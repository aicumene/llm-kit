# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""Generation routing by model: each model has its own host, and without configuration the shape
stays as it was."""

from __future__ import annotations

from dataclasses import replace

import pytest

from llmkit.base import LLMMessage, LLMRequest, LLMResponse, LLMProvider, Plane, Sensitivity
from llmkit.config import ModelConfig
from llmkit.factory import _build_trusted
from llmkit.ollama import OllamaProvider
from llmkit.vllm import ModelRoutedProvider, SplitTrustedProvider


class _Spy(LLMProvider):
    """Counts where a call went."""

    plane = Plane.TRUSTED

    def __init__(self, name: str, plane: Plane = Plane.TRUSTED) -> None:
        self.name = name
        self.plane = plane
        self.generated: list[str | None] = []
        self.embedded: list[int] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.generated.append(request.model)
        return LLMResponse(text=self.name, model=request.model or "", provider=self.name,
                           plane=self.plane)

    def stream(self, request: LLMRequest):  # pragma: no cover - not called in these tests
        self.generated.append(request.model)

        async def _it():
            yield self.name

        return _it()

    async def embed(self, texts, *, kind: str = "document"):
        self.embedded.append(len(texts))
        return [[0.0]] * len(texts)


def _req(model: str | None) -> LLMRequest:
    return LLMRequest(messages=[LLMMessage(role="user", content="q")],
                      model=model, sensitivity=Sensitivity.CONFIDENTIAL)


#: Exactly the fields _build_trusted reads; the real ModelConfig, so addresses pass the same check
#: as in production.
_SETTINGS = ModelConfig(
    chat_backend="ollama",
    ollama_base_url="http://gpu-a:11434",
    ollama_chat_model="qwen3.6:27b",
    ollama_embed_model="nomic-embed-text",
    ollama_timeout=900,
    ollama_keep_alive="30m",
    embed_base_url="",
    ollama_model_hosts="",
    #: Query prefix of an asymmetric embedder. Empty = the model is symmetric.
    ollama_embed_query_prefix="",
)


@pytest.mark.asyncio
async def test_named_model_goes_to_its_own_host():
    home, away = _Spy("home"), _Spy("away")
    p = ModelRoutedProvider(default=home, routes={"small-9b:latest": away})
    await p.generate(_req("small-9b:latest"))
    assert away.generated == ["small-9b:latest"]
    assert home.generated == []


@pytest.mark.asyncio
async def test_unnamed_model_falls_through_to_default():
    home, away = _Spy("home"), _Spy("away")
    p = ModelRoutedProvider(default=home, routes={"small-9b:latest": away})
    await p.generate(_req("qwen3.6:27b"))
    assert home.generated == ["qwen3.6:27b"]
    assert away.generated == []


@pytest.mark.asyncio
async def test_model_none_falls_through_to_default():
    home, away = _Spy("home"), _Spy("away")
    p = ModelRoutedProvider(default=home, routes={"small-9b:latest": away})
    await p.generate(_req(None))
    assert home.generated == [None]
    assert away.generated == []


@pytest.mark.asyncio
async def test_model_none_follows_the_route_of_the_chat_model():
    """No name = the chat model. When the chat model has a host of its own (25.09.2026: llama-server
    with MTP on a separate host), a request that names no model goes there, not to a default that
    lacks it."""
    home, mtp = _Spy("ollama"), _Spy("llama-chat")
    p = ModelRoutedProvider(default=home, routes={"qwen3.6-27b-mtp": mtp}, default_model="qwen3.6-27b-mtp")
    await p.generate(_req(None))
    await p.generate(_req("qwen3.6:27b"))
    assert mtp.generated == [None]
    assert home.generated == ["qwen3.6:27b"]


@pytest.mark.asyncio
async def test_embeddings_never_follow_a_generation_route():
    """The class has an opinion about generation only. Otherwise we get a silent second embedder."""
    home, away = _Spy("home"), _Spy("away")
    p = ModelRoutedProvider(default=home, routes={"small-9b:latest": away})
    await p.embed(["a", "b"])
    assert home.embedded == [2]
    assert away.embedded == []


def test_untrusted_backend_is_refused():
    home = _Spy("home")
    cloud = _Spy("anthropic", plane=Plane.CLOUD)
    with pytest.raises(ValueError):
        ModelRoutedProvider(default=home, routes={"claude": cloud})


def test_no_routes_configured_keeps_the_plain_shape():
    """A deployment that did not ask for routing must not get it."""
    assert isinstance(_build_trusted(_SETTINGS), OllamaProvider)


def test_routes_produce_a_routed_provider():
    s = replace(_SETTINGS, ollama_model_hosts="small-9b:latest=http://gpu-b:11434")
    built = _build_trusted(s)
    assert isinstance(built, ModelRoutedProvider)
    assert list(built._routes) == ["small-9b:latest"]
    assert built._routes["small-9b:latest"]._base_url.rstrip("/").endswith("gpu-b:11434")


def test_route_pointing_at_the_default_host_is_not_a_route():
    """A route to the same address is not a route: an extra wrapper hides the real shape."""
    s = replace(_SETTINGS, ollama_model_hosts=f"qwen3.6:27b={_SETTINGS.ollama_base_url}")
    assert isinstance(_build_trusted(s), OllamaProvider)


def test_malformed_entries_are_ignored_not_fatal():
    s = replace(_SETTINGS, ollama_model_hosts=",  , =http://x:1, broken, small=http://gpu-b:11434 ,")
    built = _build_trusted(s)
    assert isinstance(built, ModelRoutedProvider)
    assert list(built._routes) == ["small"]


def test_routing_and_embed_split_compose():
    """Both splits together: generations by model, embeddings to their own host."""
    s = replace(_SETTINGS, embed_base_url="http://gpu-b:11434",
                ollama_model_hosts="small-9b:latest=http://gpu-b:11434")
    built = _build_trusted(s)
    assert isinstance(built, SplitTrustedProvider)
    assert isinstance(built._chat, ModelRoutedProvider)
    assert built._embed._base_url.rstrip("/").endswith("gpu-b:11434")
