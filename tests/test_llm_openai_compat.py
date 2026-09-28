# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""The OpenAI-compatible backend ("/v1" = SGLang/vLLM) — the same contract as Ollama.

Introduced on 12.09.2026 for an SGLang pilot: one route by model name leads to a server of another
dialect, and everything around the route must not notice — token accounting, the constrained
decoding schema, the counter sink in a stream, the embedder's query prefix, the failure types.
Also here is what must NOT change: trust in an address does not depend on the dialect, and an
untrusted "/v1" is refused exactly like an untrusted Ollama.

No network is brought up: the wire is intercepted by ``httpx.MockTransport``, and the tests check
what actually went out.
"""
from __future__ import annotations

import json

import httpx
import pytest

from llmkit import openai_compat
from llmkit.base import QUERY, LLMMessage, LLMRequest, Plane
from llmkit.config import ModelConfig
from llmkit.errors import LLMBackendError, LLMUnavailable
from llmkit.openai_compat import OpenAICompatProvider, is_openai_url

SGLANG = "http://127.0.0.1:30000/v1"
OLLAMA = "http://192.0.2.10:11434"


def _client(handler):
    """Swap the httpx transport without bringing up a network."""

    class _AC(httpx.AsyncClient):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    return _AC


def _provider(**kw) -> OpenAICompatProvider:
    return OpenAICompatProvider(base_url=SGLANG, default_model="served-model",
                                embed_model="arctic", **kw)


def _req(**kw) -> LLMRequest:
    return LLMRequest(messages=[LLMMessage(role="user", content="hi")], **kw)


def _completion(text="ok", model="served-model", prompt=11, completion=3) -> dict:
    return {"model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}


# ── the dialect rule ─────────────────────────────────────────────────────────────────────────


def test_the_dialect_is_decided_by_the_end_of_the_path_not_the_port():
    assert is_openai_url(SGLANG)
    assert is_openai_url("http://127.0.0.1:30000/v1/")
    assert not is_openai_url(OLLAMA)
    assert not is_openai_url("http://127.0.0.1:30000")       # SGLang's port without /v1 — Ollama
    assert not is_openai_url("http://host/v1/models")        # /v1 not at the end — not the root
    assert not is_openai_url("") and not is_openai_url(None)


# ── generate ─────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_schema_goes_out_as_strict_json_schema(monkeypatch):
    """A schema is a grammar, not a hint: without the json_schema/strict wrapper the server takes
    it as a wish, and "supports schemas" stays words."""
    sent: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        sent.append({"url": str(req.url), "body": json.loads(req.content)})
        return httpx.Response(200, json=_completion('{"a":1}'))

    monkeypatch.setattr(httpx, "AsyncClient", _client(handler))
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    resp = await _provider().generate(_req(response_format=schema, max_tokens=64, temperature=0.0))

    assert sent[0]["url"] == SGLANG + "/chat/completions"
    body = sent[0]["body"]
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "llmkit", "schema": schema, "strict": True},
    }
    assert body["model"] == "served-model" and body["stream"] is False
    assert body["max_tokens"] == 64 and body["temperature"] == 0.0
    assert "stream_options" not in body, "stream counters belong to streams only"
    assert resp.text == '{"a":1}' and resp.plane is Plane.TRUSTED
    assert resp.provider == "openai-compat"


@pytest.mark.asyncio
async def test_the_schema_name_is_read_on_every_request(monkeypatch):
    """A deployment may name its requests in the server's log once, at start-up."""
    sent: list[dict] = []

    def handler(req):
        sent.append(json.loads(req.content))
        return httpx.Response(200, json=_completion())

    monkeypatch.setattr(httpx, "AsyncClient", _client(handler))
    monkeypatch.setattr(openai_compat, "SCHEMA_NAME", "my-deployment")
    await _provider().generate(_req(response_format={"type": "object"}))
    assert sent[0]["response_format"]["json_schema"]["name"] == "my-deployment"


@pytest.mark.asyncio
async def test_without_a_schema_no_response_format_is_sent(monkeypatch):
    sent: list[dict] = []

    def handler(req):
        sent.append(json.loads(req.content))
        return httpx.Response(200, json=_completion())

    monkeypatch.setattr(httpx, "AsyncClient", _client(handler))
    await _provider().generate(_req())
    assert "response_format" not in sent[0]
    assert "chat_template_kwargs" not in sent[0], "think=None — the model's default, no key"


@pytest.mark.asyncio
async def test_think_false_turns_reasoning_off_through_chat_template_kwargs(monkeypatch):
    sent: list[dict] = []

    def handler(req):
        sent.append(json.loads(req.content))
        return httpx.Response(200, json=_completion())

    monkeypatch.setattr(httpx, "AsyncClient", _client(handler))
    await _provider().generate(_req(think=False))
    assert sent[0]["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_response_counters_arrive_in_the_same_shape_as_ollamas(monkeypatch):
    """MeteredRouter reads usage.prompt_tokens / completion_tokens; the model name comes from the
    response."""
    monkeypatch.setattr(httpx, "AsyncClient", _client(
        lambda req: httpx.Response(200, json=_completion(model="served-name", prompt=120, completion=7))))
    resp = await _provider().generate(_req(model="ornith:9b"))
    assert resp.usage == {"prompt_tokens": 120, "completion_tokens": 7}
    assert resp.model == "served-name"


@pytest.mark.asyncio
async def test_the_requested_model_name_beats_the_default(monkeypatch):
    sent: list[dict] = []

    def handler(req):
        sent.append(json.loads(req.content))
        return httpx.Response(200, json=_completion())

    monkeypatch.setattr(httpx, "AsyncClient", _client(handler))
    await _provider().generate(_req(model="other"))
    assert sent[0]["model"] == "other"


@pytest.mark.asyncio
async def test_empty_content_takes_the_reasoning_channel(monkeypatch):
    """As with Ollama: an answer cut off by max_tokens lies in reasoning_content, and JSON in it
    must not be lost silently."""
    body = {"model": "m", "choices": [{"message": {"content": "", "reasoning_content": "{\"x\":1}"}}],
            "usage": {}}
    monkeypatch.setattr(httpx, "AsyncClient", _client(lambda req: httpx.Response(200, json=body)))
    resp = await _provider().generate(_req())
    assert resp.text == '{"x":1}'
    assert resp.usage == {"prompt_tokens": 0, "completion_tokens": 0}


# ── stream ───────────────────────────────────────────────────────────────────────────────────


def _sse(*chunks: dict, done: bool = True) -> bytes:
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks]
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


@pytest.mark.asyncio
async def test_the_stream_yields_deltas_and_fills_the_counter_sink(monkeypatch):
    """The final frame with usage (empty choices) is the only place counters arrive; without
    stream_options.include_usage it does not exist at all."""
    sent: list[dict] = []

    def handler(req):
        sent.append(json.loads(req.content))
        return httpx.Response(200, content=_sse(
            {"model": "served", "choices": [{"delta": {"role": "assistant", "content": ""}}]},
            {"model": "served", "choices": [{"delta": {"content": "Hel"}}], "usage": None},
            {"model": "served", "choices": [{"delta": {"content": "lo"}}]},
            {"model": "served", "choices": [],
             "usage": {"prompt_tokens": 40, "completion_tokens": 2}},
        ), headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(httpx, "AsyncClient", _client(handler))
    sink: dict = {}
    pieces = [p async for p in _provider().stream(_req(), usage=sink)]
    assert pieces == ["Hel", "lo"]
    assert sent[0]["stream"] is True
    assert sent[0]["stream_options"] == {"include_usage": True}
    assert sink == {"prompt_tokens": 40, "completion_tokens": 2, "model": "served"}


@pytest.mark.asyncio
async def test_a_stream_without_a_usage_frame_leaves_the_sink_empty(monkeypatch):
    """An empty sink = "not measured" in the ledger; a zero would claim the prompt cost nothing."""
    monkeypatch.setattr(httpx, "AsyncClient", _client(lambda req: httpx.Response(
        200, content=_sse({"choices": [{"delta": {"content": "x"}}]}))))
    sink: dict = {}
    assert [p async for p in _provider().stream(_req(), usage=sink)] == ["x"]
    assert sink == {}


@pytest.mark.asyncio
async def test_a_stream_without_a_sink_works(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", _client(lambda req: httpx.Response(
        200, content=_sse({"choices": [{"delta": {"content": "y"}}]},
                          {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}))))
    assert [p async for p in _provider().stream(_req())] == ["y"]


# ── embed ────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_embedder_prefixes_a_query_and_not_a_document(monkeypatch):
    sent: list[dict] = []

    def handler(req):
        sent.append({"url": str(req.url), "body": json.loads(req.content)})
        return httpx.Response(200, json={
            "data": [{"index": 1, "embedding": [0.2]}, {"index": 0, "embedding": [0.1]}],
            "usage": {"prompt_tokens": 9}})

    monkeypatch.setattr(httpx, "AsyncClient", _client(handler))
    p = _provider(query_prefix="query: ")
    sink: dict = {}
    original = ["a", "b"]
    out = await p.embed(original, kind=QUERY, usage=sink)
    assert sent[0]["url"] == SGLANG + "/embeddings"
    assert sent[0]["body"] == {"model": "arctic", "input": ["query: a", "query: b"]}
    assert original == ["a", "b"], "the caller's string changed — the prefix would leak into SQL"
    assert out == [[0.1], [0.2]], "order follows index, not the order in the response"
    assert sink == {"prompt_tokens": 9, "completion_tokens": 0, "model": "arctic"}

    await p.embed(["doc"])
    assert sent[1]["body"]["input"] == ["doc"]


# ── failures — the same types as Ollama's ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_5xx_is_LLMBackendError_without_the_address_in_the_text(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", _client(lambda req: httpx.Response(503, text="busy")))
    with pytest.raises(LLMBackendError) as exc:
        await _provider().generate(_req())
    assert "503" in str(exc.value) and "127.0.0.1" not in str(exc.value)


@pytest.mark.asyncio
async def test_an_unreachable_host_is_LLMUnavailable(monkeypatch):
    def handler(req):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "AsyncClient", _client(handler))
    with pytest.raises(LLMUnavailable):
        await _provider().generate(_req())
    with pytest.raises(LLMUnavailable):
        await _provider().embed(["x"])
    with pytest.raises(LLMUnavailable):
        async for _ in _provider().stream(_req()):
            pass


# ── the factory: "/v1" → OpenAI-compatible, otherwise Ollama ─────────────────────────────────


def _settings(**kw) -> ModelConfig:
    base = dict(ollama_base_url=OLLAMA, embed_base_url="", ollama_model_hosts="",
                trusted_model_cidrs="192.0.2.0/24,127.0.0.0/8")
    base.update(kw)
    return ModelConfig(**base)


def test_the_factory_builds_openai_compat_for_a_v1_route():
    from llmkit.factory import _build_trusted
    from llmkit.ollama import OllamaProvider
    from llmkit.vllm import ModelRoutedProvider

    p = _build_trusted(_settings(ollama_model_hosts=f"served-model={SGLANG}"))
    assert isinstance(p, ModelRoutedProvider)
    assert isinstance(p._default, OllamaProvider), "the main chat stays on Ollama"
    route = p._routes["served-model"]
    assert isinstance(route, OpenAICompatProvider)
    assert route._base_url == SGLANG and route.plane is Plane.TRUSTED


def test_the_factory_builds_ollama_for_an_address_without_v1():
    from llmkit.factory import _build_trusted
    from llmkit.ollama import OllamaProvider
    from llmkit.vllm import ModelRoutedProvider

    p = _build_trusted(_settings(ollama_model_hosts="served-model=http://192.0.2.50:11434"))
    assert isinstance(p, ModelRoutedProvider)
    assert isinstance(p._routes["served-model"], OllamaProvider)


def test_the_factory_understands_v1_for_chat_and_for_the_embedder():
    from llmkit.factory import _build_trusted
    from llmkit.vllm import SplitTrustedProvider

    p = _build_trusted(_settings(ollama_base_url=SGLANG, embed_base_url=OLLAMA))
    assert isinstance(p, SplitTrustedProvider)
    assert isinstance(p._chat, OpenAICompatProvider)
    assert type(p._embed).__name__ == "OllamaProvider"

    p = _build_trusted(_settings(ollama_base_url=OLLAMA, embed_base_url=SGLANG))
    assert isinstance(p._embed, OpenAICompatProvider)


def test_the_factory_gives_openai_compat_the_same_models_and_prefix():
    """Another wire, the same model: otherwise new vectors land in another space."""
    from llmkit.factory import _build_trusted

    p = _build_trusted(_settings(ollama_base_url=OLLAMA, embed_base_url=SGLANG,
                                 ollama_embed_model="snowflake-arctic-embed2",
                                 ollama_embed_query_prefix="query: "))
    assert p._embed._embed_model == "snowflake-arctic-embed2"
    assert p._embed._query_prefix == "query: "


def test_an_untrusted_v1_address_is_refused():
    """The dialect does not weaken the perimeter: "/v1" on a public address or a domain name gets
    the same refusal as Ollama. Otherwise OpenAI compatibility would become a road to the
    internet."""
    from llmkit.factory import _build_trusted
    from llmkit.trust import UntrustedModelHost, check_model_url

    with pytest.raises(UntrustedModelHost):
        _build_trusted(_settings(ollama_model_hosts="served-model=http://203.0.113.8:30000/v1"))
    with pytest.raises(UntrustedModelHost):
        _build_trusted(_settings(ollama_base_url="https://models.example.com/v1"))
    with pytest.raises(UntrustedModelHost):
        check_model_url("https://models.example.com/v1", cidrs="192.0.2.0/24")
    # Positive control: the same path on a trusted address passes.
    assert check_model_url(SGLANG, cidrs="127.0.0.0/8") == SGLANG
