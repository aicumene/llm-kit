# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""Token accounting: complete measurement through the compositions the factory really builds.

Work on one's own models has no marginal cost in money, so the unit here is the token. Pinned is
what the accounting used to under-count silently.

Ported from the database-free half of the source's ledger tests (``test_token_ledger.py``); the
other half tested the product's billing ledger and stayed with it.
"""
from __future__ import annotations

import pytest

from llmkit import metering
from llmkit.base import LLMRequest, Plane
from llmkit.classification import Sensitivity
from llmkit.router import LLMRouter


class _Recorder:
    """A provider that hands out counters through the sink, as Ollama does."""

    def __init__(self, prompt: int, completion: int) -> None:
        self._p, self._c = prompt, completion

    async def generate(self, request, **kw):
        class R:
            usage = {"prompt_tokens": self._p, "completion_tokens": self._c}
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


class _Deaf:
    """A provider that cannot hand out counters: a cloud one, or vLLM."""

    async def stream(self, request, **kw):
        yield "part"

    async def embed(self, texts, **kw):
        return [[0.0]]


async def _drain(gen):
    return [c async for c in gen]


# ── complete measurement ─────────────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_streaming_is_counted_because_chat_is_the_busiest_path():
    """Before, a stream raised only the call counter: "streaming responses don't reliably carry
    usage". For Ollama that is wrong — the final chunk carries prompt_eval_count and eval_count,
    and the code saw them and threw them away. So CHAT contributed exactly zero."""
    from llmkit.usage import UsageRouter
    r = UsageRouter(_Recorder(120, 34))
    await _drain(r.stream(object()))
    assert r.usage["prompt_tokens"] == 120
    assert r.usage["completion_tokens"] == 34
    assert r.usage["calls"] == 1


@pytest.mark.anyio
async def test_embedding_is_counted_at_all():
    """``embed`` went to the provider past the counter through ``__getattr__``. On a corpus of 5301
    chunks that is a visible share of GPU work, invisible in the report."""
    from llmkit.usage import UsageRouter
    r = UsageRouter(_Recorder(9, 0))
    await r.embed(["a line"])
    assert r.usage["prompt_tokens"] == 9
    assert r.usage["calls"] == 1


@pytest.mark.anyio
async def test_a_provider_that_cannot_report_still_counts_the_call():
    """Zero tokens with non-zero calls is distinguishable from "consumed nothing" — and it is the
    only way to see that a provider without counters appeared somewhere."""
    from llmkit.usage import UsageRouter
    r = UsageRouter(_Deaf())
    await _drain(r.stream(object()))
    await r.embed(["a line"])
    assert r.usage["calls"] == 2
    assert r.usage["prompt_tokens"] == 0


# ── the composition that is actually deployed ───────────────────────────────────────────────


class _TrustedRecorder(_Recorder):
    """The same provider, with the fields the ROUTER needs: plane and name."""

    name = "recorder"
    plane = Plane.TRUSTED

    async def stream(self, request, *, usage=None, **kw):
        if usage is not None:
            usage["prompt_tokens"] = self._p
            usage["completion_tokens"] = self._c
            usage["model"] = "test-model"
        yield "part"

    async def health(self):
        return True


class _DeafTrusted(_Deaf):
    name = "deaf"
    plane = Plane.TRUSTED

    async def health(self):
        return True


def _req():
    return LLMRequest(messages=[], sensitivity=Sensitivity.CONFIDENTIAL)


def _deployed(rec):
    """EXACTLY THE CHAIN THE FACTORY BUILDS on a deployment with per-model routes and a separate
    embedder.

    ``OLLAMA_MODEL_HOSTS`` is non-empty and ``EMBED_BASE_URL`` differs from the main host, so
    ``_build_trusted`` gives ``SplitTrustedProvider(chat=ModelRoutedProvider(...))``, and the
    metering wrapper sits on top of the router. Building a bare provider in a test would check a
    composition that does not exist in production.
    """
    from llmkit.vllm import ModelRoutedProvider, SplitTrustedProvider

    return metering.MeteredRouter(LLMRouter([SplitTrustedProvider(
        chat=ModelRoutedProvider(default=rec, routes={"small-9b:latest": rec}),
        embed_provider=rec,
    )]))


@pytest.mark.anyio
async def test_stream_is_counted_through_the_composition_production_actually_runs():
    """The hole lived unnoticed because the tests wrapped the PROVIDER directly, while production
    runs ``MeteredRouter(LLMRouter(...))``. The router did NOT ACCEPT the sink, the wrapper caught
    ``TypeError``, fell back and wrote a row with zero tokens."""
    m = metering.start("chat")
    router = metering.MeteredRouter(LLMRouter([_TrustedRecorder(120, 34)]))
    assert await _drain(router.stream(_req())) == ["part"]

    assert list(m.buckets) == [("stream", "test-model", "trusted")], (
        "model and plane used to be hard-coded '' and 'trusted' — the breakdown lied"
    )
    b = m.buckets[("stream", "test-model", "trusted")]
    assert b["prompt_tokens"] == 120
    assert b["completion_tokens"] == 34
    assert b["calls"] == 1


@pytest.mark.anyio
async def test_stream_is_counted_through_the_DEPLOYED_chain_not_a_bare_provider():
    """THE SECOND ITERATION OF THE SAME MISTAKE, and that is why the test is separate.

    The previous test is named "the composition that is really deployed" but built a bare
    provider — while the factory builds ``SplitTrustedProvider(chat=ModelRoutedProvider(...))``.
    The sink hit ``ModelRoutedProvider.stream``, which did not accept it, the ``TypeError``
    fallback fired, and the sink NEVER reached Ollama: every streamed answer wrote zero tokens to
    the ledger. The test was green, production was broken.

    The lesson costs more than the fix: the test's name made a claim about the deployed chain, and
    the code did not build it.
    """
    m = metering.start("chat")
    assert await _drain(_deployed(_TrustedRecorder(120, 34)).stream(_req())) == ["part"]

    assert list(m.buckets) == [("stream", "test-model", "trusted")], m.buckets
    b = m.buckets[("stream", "test-model", "trusted")]
    assert b["prompt_tokens"] == 120, "the sink did not get through the composite providers"
    assert b["completion_tokens"] == 34
    assert b["calls"] == 1


@pytest.mark.anyio
async def test_a_wrapped_stream_still_rejects_an_unknown_kwarg_AT_CALL_TIME():
    """The ``except TypeError`` guards around ``stream()`` are alive only if argument binding
    happens AT THE CALL. The error wrapper used to be an async generator, and binding moved to the
    first ``__anext__`` — outside the caller's ``try``. All three guards died at once and silently,
    and the stream broke off at the first chunk with no token and no error frame."""
    from llmkit.errors import streaming_transport_errors

    @streaming_transport_errors("probe")
    async def only_request(request):
        yield "part"

    with pytest.raises(TypeError):
        only_request(object(), usage={})      # must fail HERE, not during iteration


@pytest.mark.anyio
async def test_a_provider_that_cannot_count_is_booked_as_UNMEASURED_not_as_zero():
    """A deaf provider (cloud, vLLM) must neither break the stream nor vanish from the ledger.

    Before, a ``stream`` row with zeros and a non-zero call counter was expected here. The
    expectation was wrong in exactly what this ledger exists against: a zero reads as "cost
    nothing", although the model read the prompt. A deaf path and a broken stream are ONE case,
    "could not measure", and now it is named by its own kind instead of hiding behind a zero."""
    m = metering.start("chat")
    router = metering.MeteredRouter(LLMRouter([_DeafTrusted()]))
    assert await _drain(router.stream(_req())) == ["part"]

    (key,) = list(m.buckets)
    assert key[0] == "stream_unmeasured", m.buckets
    b = m.buckets[key]
    assert b["calls"] == 1, "the call must stay visible"
    assert b["prompt_tokens"] == 0 and b["completion_tokens"] == 0
