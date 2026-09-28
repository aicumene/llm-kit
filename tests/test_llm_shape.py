# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""Tests of the response shaper.

What is checked is not "does parsing work" but the four rules the layer exists for: do not guess
from prose, refuse a value outside the domain, record the mechanism, continue the ladder past the
preferred mechanism.
"""
from __future__ import annotations

import pytest

from llmkit import shape as sh
from llmkit.base import LLMMessage, LLMRequest, LLMResponse, Plane


# An unknown model — so the general ladder tests walk the FULL ladder, not a shortened measured
# one. Measured ladders are checked by separate tests below.
def _req(model: str = "unknown-model") -> LLMRequest:
    return LLMRequest(messages=[LLMMessage(role="user", content="Pick one.")], model=model)


class FakeRouter:
    """Answers from a script: one reply for each step up the ladder."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.seen: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.seen.append(request)
        text = self._replies.pop(0) if self._replies else ""
        return LLMResponse(text=text, model=request.model or "", provider="fake",
                           plane=Plane.TRUSTED)


# ── rule 1: do not guess from prose ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("prose", [
    "The paragraph discusses candidate 5 and its relevance.",
    "I think the answer is probably 3, but candidate 7 also fits.",
    "Answer: the fifth one.",
])
def test_prose_containing_a_number_is_not_a_value(prose: str) -> None:
    """A paragraph with a number inside is NOT an answer.

    This is the layer's main rule. Pulling the first number out of prose turns "the model did not
    answer" into "the model answered wrongly". In the 04.09 measurement a model with nowhere to go
    named a known-wrong option in 83% of cases — silence is more useful.
    """
    value, why = sh.choice(10).parse(prose)
    assert value is None
    assert why


# ── rule 2: outside the domain is a refusal, not a fit ───────────────────────────────────────

def test_out_of_range_choice_is_refused_not_clamped() -> None:
    value, why = sh.choice(10).parse("11")
    assert value is None
    assert "11" in why, "the reason must name the refused value"


def test_zero_is_a_value_only_when_none_is_allowed() -> None:
    assert sh.choice(10, allow_none=True).parse("0")[0] == 0
    assert sh.choice(10, allow_none=False).parse("0")[0] is None


def test_label_outside_the_set_is_refused() -> None:
    value, _ = sh.label(("approved", "rejected")).parse("deferred")
    assert value is None


# ── parsing both shapes ──────────────────────────────────────────────────────────────────────

def test_bare_digit_and_schema_object_both_parse() -> None:
    assert sh.choice(10).parse("7")[0] == 7
    assert sh.choice(10).parse('{"value": 7}')[0] == 7
    assert sh.choice(10).parse("  7. ")[0] == 7


def test_schema_object_out_of_range_still_refused() -> None:
    """Constrained decoding guarantees the SHAPE, not the domain: a backend that does not apply
    minimum/maximum may still return the schema. The domain check here is not redundant."""
    value, _ = sh.choice(5).parse('{"value": 9}')
    assert value is None


# ── rules 3 and 4: the ladder and recording the mechanism ────────────────────────────────────

@pytest.mark.anyio
async def test_ladder_escalates_and_records_the_mechanism() -> None:
    router = FakeRouter(["The paragraph discusses...", "4"])
    out = await sh.ask(router, _req(), sh.choice(10))
    assert out.value == 4
    assert out.mechanism is sh.Mechanism.PREFILL
    assert len(router.seen) == 2
    assert out.attempts[0][0] is sh.Mechanism.NATIVE
    # The prefill must travel as a separate assistant turn, not be glued into the prompt: glued
    # into the user turn it becomes part of the question and changes the task.
    assert router.seen[1].messages[-1].role == "assistant"


@pytest.mark.anyio
async def test_all_mechanisms_failing_returns_none_with_reasons() -> None:
    router = FakeRouter(["prose one", "prose two", "prose three"])
    out = await sh.ask(router, _req(), sh.choice(10))
    assert out.value is None and not out.ok
    assert len(out.attempts) == len(sh.LADDER)
    assert out.reason


@pytest.mark.anyio
async def test_schema_travels_as_response_format_not_as_text() -> None:
    router = FakeRouter(["prose", "prose", '{"value": 2}'])
    out = await sh.ask(router, _req(), sh.choice(10))
    assert out.value == 2 and out.mechanism is sh.Mechanism.SCHEMA
    assert router.seen[-1].response_format is not None
    assert "minimum" in str(router.seen[-1].response_format)


def test_measured_ladder_is_an_order_not_a_starting_point() -> None:
    """A model's ladder is an ORDER from a measurement, not "where to start, then as usual".

    For ``qwen3.6`` the prefill breaks the format completely: 1.00 on its own against 0.00 under
    a prefill. Walking the prefill "just in case" spends a call on a known miss, so the rung was
    removed from its ladder, not moved back.
    """
    assert sh.Mechanism.PREFILL not in sh._mechanisms_for("qwen3.6:27b")
    assert sh._mechanisms_for("qwen3.6:27b")[0] is sh.Mechanism.NATIVE


def test_a_registered_ladder_for_a_tuned_model_is_used_as_registered(monkeypatch) -> None:
    """Tuned models are registered by the deployment after its own measurement. Both measured
    domain-tune lines held the format at 0.00 and 0.02 on their own, so their ladder starts at the
    prefill and has no native rung: starting with native would lose a call."""
    monkeypatch.setitem(sh.LADDERS, "tuned-4b", (sh.Mechanism.PREFILL, sh.Mechanism.SCHEMA))
    ladder = sh._mechanisms_for("tuned-4b:latest")
    assert ladder[0] is sh.Mechanism.PREFILL
    assert sh.Mechanism.NATIVE not in ladder


def test_schema_is_the_last_rung_everywhere() -> None:
    """The schema is the only mechanism with 1.00 on all four measured models.

    If it is not the last rung of some ladder, that model has a way to end in refusal while an
    unused mechanism that would certainly have worked remains.
    """
    for model, ladder in sh.LADDERS.items():
        assert sh.Mechanism.SCHEMA in ladder, model
        assert ladder[-1] is sh.Mechanism.SCHEMA or sh.Mechanism.SCHEMA in ladder[:-1], model
    assert sh.DEFAULT_LADDER[-1] is sh.Mechanism.SCHEMA


def test_unknown_model_walks_the_full_ladder() -> None:
    ladder = sh._mechanisms_for("never-seen-model")
    assert ladder == sh.DEFAULT_LADDER
    assert set(ladder) == set(sh.LADDER)


def test_request_is_not_mutated() -> None:
    r = _req()
    before = r.messages[0].content
    sh.apply(r, sh.choice(4), sh.Mechanism.PREFILL)
    assert r.messages[0].content == before
    assert r.response_format is None


def test_empty_think_wrapper_is_stripped_but_real_reasoning_is_not() -> None:
    """An empty `<think></think>` is a wrapper of the ollama build, not the model's answer.

    Measured 05.09 on 100 tasks: without stripping it, a model that answered correctly showed a
    format score of 0.00 instead of 1.00 — a fixed model read as broken. NON-empty reasoning must
    not be stripped, though: where one digit was asked for, a paragraph of reasoning is a genuine
    format violation, and swallowing it silently would stop us noticing a tune regressing.
    """
    assert sh.choice(10).parse("<think>\n\n</think>\n\n5")[0] == 5
    assert sh.choice(10).parse("<think></think>7")[0] == 7
    assert sh.choice(10).parse("<think>I reason at length</think>\n\n5")[0] is None
    assert sh.label(("approved", "rejected")).parse("<think>\n</think>\napproved")[0] == "approved"
