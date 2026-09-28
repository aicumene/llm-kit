# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""Completeness of the accounting: the wrapper is installed once and counts every path.

Pinned here is what the meter was moved into the task context for, and the wrapper into the
factory.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest

from llmkit import metering


class _Fake:
    def __init__(self, p=0, c=0):
        self._p, self._c = p, c

    async def generate(self, request, **kw):
        class R:
            usage = {"prompt_tokens": self._p, "completion_tokens": self._c}
            model = "qwen3.6:27b"
        return R()

    async def stream(self, request, *, usage=None, **kw):
        if usage is not None:
            usage["prompt_tokens"] = self._p
            usage["completion_tokens"] = self._c
        yield "part"

    async def embed(self, texts, *, usage=None, **kw):
        if usage is not None:
            usage["prompt_tokens"] = self._p
        return [[0.0]]


async def _drain(gen):
    return [c async for c in gen]


@pytest.mark.anyio
async def test_without_a_meter_nothing_is_recorded_and_nothing_breaks():
    """Scripts, tests and one-off runs do not start a meter.

    The accounting has no right to break the work it accounts for — so without a context the
    wrapper is a transparent pass-through.
    """
    metering._METER.set(None)
    r = metering.MeteredRouter(_Fake(10, 5))
    await r.generate(object())
    await _drain(r.stream(object()))
    await r.embed(["x"])
    assert metering.current() is None


@pytest.mark.anyio
async def test_concurrent_tasks_do_not_share_a_counter():
    """THIS IS THE REASON the meter lives in the context and not in a field of the wrapper.

    The router from build_router is ONE for the whole application. A field on it would add up the
    consumption of all concurrent requests in one counter: both a race and nonsense. The bug would
    show only under load, i.e. in production.
    """
    r = metering.MeteredRouter(_Fake(100, 10))

    async def one(source: str, times: int) -> dict:
        metering.start(source)
        for _ in range(times):
            await r.generate(object())
        m = metering.current()
        return {"source": m.source, "prompt": sum(b["prompt_tokens"] for b in m.buckets.values())}

    a, b = await asyncio.gather(one("chat", 2), one("draft", 3))
    assert a == {"source": "chat", "prompt": 200}
    assert b == {"source": "draft", "prompt": 300}


@pytest.mark.anyio
async def test_a_thousand_embeddings_fold_into_one_bucket():
    """Folding in place: re-indexing a corpus must not become a thousand ledger rows."""
    metering.start("kb")
    r = metering.MeteredRouter(_Fake(9, 0))
    for _ in range(1000):
        await r.embed(["a line"])
    m = metering.current()
    assert len(m.buckets) == 1
    (kind, _model, plane), b = next(iter(m.buckets.items()))
    assert (kind, plane) == ("embed", "trusted")
    assert b["calls"] == 1000 and b["prompt_tokens"] == 9000


@pytest.mark.anyio
async def test_kinds_are_kept_apart():
    """The holes in the accounting were BY KIND: the stream was not counted, the embedding was not
    counted.

    Merging them into one bucket would make the next such hole indistinguishable from zero again.
    """
    metering.start("chat")
    r = metering.MeteredRouter(_Fake(7, 3))
    await r.generate(object())
    await _drain(r.stream(object()))
    await r.embed(["x"])
    kinds = {k for k, _, _ in metering.current().buckets}
    assert kinds == {"generate", "stream", "embed"}


@pytest.mark.anyio
async def test_attribution_can_be_refined_later_and_is_safe_without_a_meter():
    metering.start("other")
    sid = uuid.uuid4()
    metering.attribute(source="draft", scope_id=sid)
    assert metering.current().source == "draft"
    assert metering.current().scope_id == sid
    # An unknown field is skipped silently, like a missing meter: the call site need not know
    # what the meter carries.
    metering.attribute(no_such_field=uuid.uuid4())
    assert not hasattr(metering.current(), "no_such_field")
    metering._METER.set(None)
    metering.attribute(source="chat")   # does not fail without a meter


def test_the_factory_wraps_so_a_new_path_is_counted_without_being_remembered():
    """The accounting stood on TWO paths of six, and drafts, research, document intake and corpus
    re-indexing — probably the largest item — went past it. The wrapper in the factory removes the
    need to remember: everything that goes to a model at all is counted."""
    from llmkit.config import ModelConfig
    from llmkit.factory import build_router
    assert isinstance(build_router(ModelConfig()), metering.MeteredRouter)


@pytest.mark.anyio
async def test_flush_clears_so_a_retry_does_not_double_count():
    metering.start("chat")
    r = metering.MeteredRouter(_Fake(50, 5))
    await r.generate(object())
    sink = metering.InMemoryUsageSink()
    assert await metering.flush(sink) == 1
    assert metering.current().buckets == {}
    assert await metering.flush(sink) == 0
    assert len(sink.rows) == 1
