# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""The ledger as it runs: MeteredRouter → flush → a usage sink, with stages and savings.

24.09.2026, live: one question — embed 16, a long-document reading 7 332, answer 7 835 prompt
tokens — left no row at all. The staged kind broke the ledger table's CHECK constraint (it listed
six plain kinds), and flush writes a request in one commit, so the whole request was lost. The
stage tests checked the in-memory buckets and never what reached storage: the composition that
broke was not the composition under test.

In the source these tests wrote through the same composition into a test database. Here the
storage is a :class:`~llmkit.metering.UsageSink`; the sink below refuses a kind the way that table
does (``KIND_PATTERN``), so a kind storage would refuse fails here, not in a log.

Savings: a token saver removes prompt tokens BEFORE a call; the ledger keeps them on the row of the
call they belong to (the same stage), so "what this call cost" and "what it would have cost" stay
side by side, per stage, per saver.
"""

from __future__ import annotations

import re
import uuid
from types import SimpleNamespace

import pytest

from llmkit import metering

#: The kinds the metering produces: a base kind, "_unmeasured" when counters could not be had,
#: and ":<stage>" inside a stage — the shape a storage constraint can rely on.
KIND_PATTERN = r"^(generate|stream|embed)(_unmeasured)?(:[a-z][a-z0-9_]{0,39})?$"


class _Inner:
    """A provider that reports usage the way Ollama and cloud providers do."""

    def __init__(self, prompt: int = 100, completion: int = 10, **extra):
        self.p, self.c, self.extra = prompt, completion, extra

    async def generate(self, request, **kw):
        return SimpleNamespace(text="ok", model="qwen3.6:27b", plane="trusted",
                               usage={"prompt_tokens": self.p, "completion_tokens": self.c, **self.extra})

    async def embed(self, texts, *, usage=None, **kw):
        if usage is not None:
            usage.update(prompt_tokens=16, model="snowflake-arctic-embed2", plane="trusted")
        return [[0.0]]


class _ConstrainedSink(metering.InMemoryUsageSink):
    """Refuses a kind the way a ledger table with a CHECK on the kind does."""

    async def record(self, row: metering.UsageRow) -> bool:
        if not re.fullmatch(KIND_PATTERN, row.kind):
            raise ValueError(f"kind {row.kind!r} violates the ledger's kind constraint")
        return await super().record(row)


async def test_a_staged_request_reaches_the_sink_whole():
    metering.start("chat")
    router = metering.MeteredRouter(_Inner())
    await router.embed(["question"])
    with metering.stage("doc_read"):
        await router.generate(object())
    with metering.stage("fact_extract"):
        await router.generate(object())
    await router.generate(object())

    sink = _ConstrainedSink()
    assert await metering.flush(sink) == 4
    kinds = sorted(r.kind for r in sink.rows)
    assert kinds == ["embed", "generate", "generate:doc_read", "generate:fact_extract"]
    assert all(re.fullmatch(KIND_PATTERN, k) for k in kinds)
    assert sink.commits == 1


async def test_savings_sit_on_the_row_of_the_call_they_reduced():
    metering.start("chat")
    router = metering.MeteredRouter(_Inner())
    with metering.stage("doc_read"):
        metering.save("observation_pack", 9_000)
        await router.generate(object())
    metering.save("context_dedupe", 120)
    metering.save("whitespace", 30)
    await router.generate(object())

    sink = _ConstrainedSink()
    await metering.flush(sink)
    by_kind = {r.kind: r for r in sink.rows}
    assert by_kind["generate:doc_read"].saved_by == {"observation_pack": 9_000}
    assert sum(by_kind["generate:doc_read"].saved_by.values()) == 9_000
    assert by_kind["generate"].saved_by == {"context_dedupe": 120, "whitespace": 30}
    assert sum(by_kind["generate"].saved_by.values()) == 150


async def test_a_saving_whose_call_failed_is_kept_on_the_largest_call():
    metering.start("chat")
    router = metering.MeteredRouter(_Inner(prompt=5_000))
    with metering.stage("query_translate"):
        metering.save("context_dedupe", 40)          # the translation never ran: no bucket of its own
    await router.generate(object())

    sink = _ConstrainedSink()
    await metering.flush(sink)
    (row,) = sink.rows
    assert row.kind == "generate" and row.saved_by == {"context_dedupe": 40}


async def test_mode_organization_and_cache_are_written():
    org = uuid.uuid4()
    meter = metering.start("chat")
    meter.mode, meter.org_id = "cloud", org
    router = metering.MeteredRouter(_Inner(prompt=1_200, cache_read_tokens=4_000, cache_write_tokens=0))
    await router.generate(object())

    sink = _ConstrainedSink()
    await metering.flush(sink)
    (row,) = sink.rows
    assert (row.mode, row.org_id, row.usage["cache_read_tokens"], row.usage["prompt_tokens"]) == (
        "cloud", org, 4_000, 1_200)


async def test_totals_of_a_turn_count_what_was_sent_and_what_was_saved():
    metering.start("chat")
    router = metering.MeteredRouter(_Inner(prompt=1_000, completion=50))
    await router.embed(["q"])                         # embeddings are not prompt tokens of an answer
    metering.save("context_dedupe", 200)
    await router.generate(object())
    t = metering.totals()
    assert t["prompt_tokens"] == 1_000 and t["completion_tokens"] == 50
    assert t["tokens_saved"] == 200 and t["saved_by"] == {"context_dedupe": 200}


async def test_a_row_the_sink_refuses_is_logged_not_silent(caplog):
    metering.start("chat")
    metering.current().add("generate:Not Allowed", "m", "trusted", prompt=1)   # bypasses stage()
    sink = _ConstrainedSink()
    with caplog.at_level("WARNING", logger="llmkit.metering"):
        assert await metering.flush(sink) == 0
    assert "failed to write" in caplog.text
    assert sink.commits == 0
    # The meter is cleared anyway: a retry must not write the same spend twice.
    assert metering.current().buckets == {}


@pytest.fixture(autouse=True)
def _no_meter_leaks():
    yield
    metering._METER.set(None)
