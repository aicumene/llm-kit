# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""Stage attribution: the ledger must say WHICH stage burned the tokens, not only which path.

A chat's 19k tokens on one question are one long-document reading plus a short answer; without
the stage the two are one number and no attempt to make the reading cheaper can be measured.
Storage may accept a kind only in the shape of ``KIND_PATTERN``: until 24.09.2026 the source's
ledger table accepted none of the staged kinds, and one staged bucket dropped the whole request's
usage — tests/test_metering_savings.py checks that through a sink.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from llmkit import metering

KIND_PATTERN = r"^(generate|stream|embed)(_unmeasured)?(:[a-z][a-z0-9_]{0,39})?$"


class _Inner:
    def __init__(self):
        self.seen = []

    async def generate(self, request, **kw):
        self.seen.append(request)
        return SimpleNamespace(text="ok", model="qwen3.6:27b", plane="trusted",
                               usage={"prompt_tokens": 100, "completion_tokens": 10})


def _kinds(meter):
    return sorted(k for (k, _model, _plane) in meter.buckets)


async def test_a_stage_names_the_work_inside_the_source():
    meter = metering.start("chat")
    router = metering.MeteredRouter(_Inner())
    with metering.stage("doc_read"):
        await router.generate(object())
    await router.generate(object())
    assert _kinds(meter) == ["generate", "generate:doc_read"]
    assert meter.buckets[("generate:doc_read", "qwen3.6:27b", "trusted")]["prompt_tokens"] == 100


async def test_stages_nest_and_the_inner_one_wins():
    meter = metering.start("agent")
    router = metering.MeteredRouter(_Inner())
    with metering.stage("draft_generate"):
        with metering.stage("draft_critic"):
            await router.generate(object())
        await router.generate(object())
    assert _kinds(meter) == ["generate:draft_critic", "generate:draft_generate"]
    assert metering.current_stage() == ""        # restored on exit


async def test_any_stage_name_becomes_a_kind_storage_accepts():
    meter = metering.start("chat")
    router = metering.MeteredRouter(_Inner())
    for name in ("a_very_long_stage_name_that_would_not_fit_in_twenty_four", "Fact Extract!", "2nd pass"):
        with metering.stage(name):
            await router.generate(object())
    kinds = _kinds(meter)
    assert "generate:fact_extract" in kinds and "generate:s_2nd_pass" in kinds
    assert all(re.fullmatch(KIND_PATTERN, k) and len(k) <= 64 for k in kinds), kinds


async def test_without_a_stage_the_kind_is_byte_for_byte_what_it_was():
    meter = metering.start("ingest")
    router = metering.MeteredRouter(_Inner())
    await router.generate(object())
    assert _kinds(meter) == ["generate"]


def test_the_stage_survives_an_exception_inside_the_block():
    with pytest.raises(RuntimeError):
        with metering.stage("doc_read"):
            raise RuntimeError("boom")
    assert metering.current_stage() == ""


async def test_breakdown_is_per_stage():
    """breakdown() = rows per (stage, model, plane), NOT one folded total. Triage and the answer on
    one model/plane are still two rows — the stage separates them (that is the whole per-step
    point)."""
    metering.start("chat", mode="local")
    router = metering.MeteredRouter(_Inner())
    with metering.stage("triage"):
        await router.generate(object())
    with metering.stage("answer"):
        await router.generate(object())
    rows = metering.breakdown()
    assert sorted(r["stage"] for r in rows) == ["answer", "triage"]
    ans = next(r for r in rows if r["stage"] == "answer")
    assert ans["kind"] == "generate" and ans["model"] == "qwen3.6:27b" and ans["plane"] == "trusted"
    assert ans["prompt_tokens"] == 100 and ans["completion_tokens"] == 10 and ans["calls"] == 1


def test_no_meter_no_breakdown():
    """Without a meter (a script, a test without start) the breakdown is empty, like everything in
    metering."""
    from llmkit.metering import _METER

    _METER.set(None)
    assert metering.breakdown() == []
