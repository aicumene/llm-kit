# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""Completeness of the accounting — defects found by an adversarial review.

What they all share: what was not counted looked like zero, and what was not checked looked like
"all good".
"""
from __future__ import annotations

import uuid

import pytest

from llmkit import metering
from llmkit.errors import LLMUnavailable


class _DiesMidStream:
    """A provider that gave part of the answer and broke off — a saturated model host."""

    async def stream(self, request, *, usage=None, **kw):
        if usage is not None:
            usage["prompt_tokens"] = 30_000
            usage["model"] = "qwen3.6:27b"
        yield "start of the answer"
        raise LLMUnavailable("ollama: ReadTimeout")


class _EmbedsWithModel:
    async def embed(self, texts, *, usage=None, **kw):
        if usage is not None:
            usage["prompt_tokens"] = 9
            usage["model"] = "snowflake-arctic-embed2"
        return [[0.0]]


@pytest.mark.anyio
async def test_a_stream_that_dies_midway_still_leaves_a_row():
    """The prompt tokens are burnt — flush()'s docstring calls them spend just like a successful
    request's. Before, m.add stood AFTER the loop, the failure went past it, and nothing was left
    in the ledger: no tokens, no call count. The spend was indistinguishable from no request."""
    m = metering.start("chat")
    r = metering.MeteredRouter(_DiesMidStream())
    with pytest.raises(LLMUnavailable):
        async for _ in r.stream(object()):
            pass

    assert m.buckets, "a broken stream left no row in the ledger"
    (key,) = list(m.buckets)
    assert key == ("stream", "qwen3.6:27b", "trusted")
    assert m.buckets[key]["prompt_tokens"] == 30_000
    assert m.buckets[key]["calls"] == 1


@pytest.mark.anyio
async def test_a_stream_abandoned_by_the_client_still_leaves_a_row():
    """The user closed the tab: closing a streaming response throws GeneratorExit into the yield."""
    m = metering.start("chat")
    gen = metering.MeteredRouter(_DiesMidStream()).stream(object())
    assert await gen.__anext__() == "start of the answer"
    await gen.aclose()

    assert m.buckets, "an abandoned stream burnt tokens too"
    assert list(m.buckets)[0][0] == "stream"


@pytest.mark.anyio
async def test_the_embedding_row_carries_the_model_that_did_the_work():
    """After an embedder swap the whole re-indexing run landed under an empty key — together with
    the previous model's embeddings. The model column exists exactly for this question."""
    m = metering.start("kb")
    await metering.MeteredRouter(_EmbedsWithModel()).embed(["a line"])

    assert list(m.buckets) == [("embed", "snowflake-arctic-embed2", "trusted")]


class _Cloud:
    """A cloud provider's response: plane is a field, provider is a STRING."""

    class _R:
        model = "claude-opus-4-7"
        provider = "anthropic"
        plane = "cloud"
        usage = {"prompt_tokens": 100, "completion_tokens": 20}

    async def generate(self, request, **kw):
        return self._R()


class _Falls:
    async def generate(self, request, **kw):
        raise LLMUnavailable("ollama: ConnectError")

    async def embed(self, texts, *, usage=None, **kw):
        raise LLMUnavailable("embedder: ReadTimeout")


class _NoCounters:
    """A live Ollama puts counters only in the final frame — on a break the sink is empty."""

    async def stream(self, request, *, usage=None, **kw):
        yield "start"
        raise LLMUnavailable("ollama: ReadTimeout")


@pytest.mark.anyio
async def test_a_cloud_answer_is_not_booked_as_trusted():
    """The plane was computed as type(resp.provider).__name__, and provider is a STRING, so it was
    always 'str' and therefore always 'trusted'. The breakdown by plane — the only one that costs
    money and privacy — was 100% wrong exactly on the paid path."""
    m = metering.start("research")
    await metering.MeteredRouter(_Cloud()).generate(object())
    assert list(m.buckets) == [("generate", "claude-opus-4-7", "cloud")], m.buckets


@pytest.mark.anyio
async def test_a_failed_generate_leaves_an_UNMEASURED_row_not_a_zero():
    """A failed call left no row at all — the same hole that was fixed for the stream. A zero does
    not do either: the model has already read the prompt. A kind of its own tells "measured zero"
    from "could not measure"."""
    m = metering.start("chat")
    with pytest.raises(LLMUnavailable):
        await metering.MeteredRouter(_Falls()).generate(object())
    assert list(m.buckets) == [("generate_unmeasured", "", "trusted")], m.buckets


@pytest.mark.anyio
async def test_a_failed_embed_leaves_an_UNMEASURED_row_too():
    m = metering.start("kb")
    with pytest.raises(LLMUnavailable):
        await metering.MeteredRouter(_Falls()).embed(["a line"])
    assert list(m.buckets) == [("embed_unmeasured", "", "trusted")], m.buckets


@pytest.mark.anyio
async def test_an_aborted_stream_says_unmeasured_instead_of_zero():
    """The earlier test of a broken stream was misleading: its fake filled the sink BEFORE the first
    yield, which a live provider does not do. On a real break the sink is empty, and the row would
    claim that 30 thousand prompt tokens cost zero."""
    m = metering.start("chat")
    with pytest.raises(LLMUnavailable):
        async for _ in metering.MeteredRouter(_NoCounters()).stream(object()):
            pass
    (key,) = list(m.buckets)
    assert key[0] == "stream_unmeasured", m.buckets
    assert m.buckets[key]["calls"] == 1


# ── who is spending: known only at flush time ────────────────────────────────────────────────
# Re-expressed from the source's database test: there the identity came from the live row-level
# security context; here it comes from the ``identity`` hook of ``flush``.


@pytest.mark.anyio
async def test_the_flush_learns_who_is_spending_from_the_identity_hook():
    """On a web path the meter is started BEFORE the profile is loaded, so there is no author yet,
    and the row went out with ``profile_id = NULL`` — which an "authors see their own rows" rule
    never lets through. The author is learnt at flush time, when the identity is in force."""
    actor = uuid.uuid4()
    m = metering.start("chat")          # no author passed — as on a web path
    assert m.profile_id is None
    m.add("stream", "qwen3.6:27b", "trusted", prompt=5, completion=2)
    sink = metering.InMemoryUsageSink()
    assert await metering.flush(sink, identity=lambda: str(actor)) == 1
    assert sink.rows[0].profile_id == actor, "the row left without an author"


@pytest.mark.anyio
async def test_the_identity_hook_never_replaces_a_known_author_and_ignores_garbage():
    known = uuid.uuid4()
    asked: list[int] = []
    m = metering.start("chat", profile_id=known)
    m.add("stream", "qwen3.6:27b", "trusted", prompt=5)
    sink = metering.InMemoryUsageSink()
    await metering.flush(sink, identity=lambda: asked.append(1) or str(uuid.uuid4()))
    assert sink.rows[0].profile_id == known and asked == []

    m = metering.start("chat")
    m.add("stream", "qwen3.6:27b", "trusted", prompt=5)
    sink = metering.InMemoryUsageSink()
    await metering.flush(sink, identity=lambda: "not-a-uuid")
    assert sink.rows[0].profile_id is None
