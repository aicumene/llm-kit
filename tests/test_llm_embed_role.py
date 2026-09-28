# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""The query / document axis in the embedding layer.

An asymmetric model expects a prefix on a query and none on a document. The tests hold three
properties, and the second is the reason the whole construction is built this way and not more
simply.
"""

from __future__ import annotations

import httpx
import pytest

from llmkit.base import DOCUMENT, QUERY, LLMProvider, Plane, Sensitivity
from llmkit.ollama import OllamaProvider
from llmkit.router import LLMRouter
from llmkit.vllm import ModelRoutedProvider, SplitTrustedProvider

PREFIX = "query: "


def _provider(monkeypatch, sink: list[dict]) -> OllamaProvider:
    """A provider whose wire is intercepted: we look at what actually went out over HTTP."""

    class _Resp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"embeddings": [[0.1, 0.2]]}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, **kw):
            sink.append(json)
            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: _Client())
    return OllamaProvider(base_url="http://x:11434", default_model="m", embed_model="e",
                          query_prefix=PREFIX)


@pytest.mark.asyncio
async def test_query_is_prefixed(monkeypatch):
    sink: list[dict] = []
    await _provider(monkeypatch, sink).embed(["how do I reset the router?"], kind=QUERY)
    assert sink[0]["input"] == ["query: how do I reset the router?"]


@pytest.mark.asyncio
async def test_document_is_not_prefixed(monkeypatch):
    sink: list[dict] = []
    await _provider(monkeypatch, sink).embed(["Hold the reset button for ten seconds."], kind=DOCUMENT)
    assert sink[0]["input"] == ["Hold the reset button for ten seconds."]


@pytest.mark.asyncio
async def test_default_is_document(monkeypatch):
    """A forgotten role must mean DOCUMENT.

    A forgotten query loses a little accuracy on one search. A forgotten document would be
    embedded in the wrong role in EVERY row of the corpus, and nothing would fail: provenance
    records the model name and the dimension, not the role.
    """
    sink: list[dict] = []
    await _provider(monkeypatch, sink).embed(["text"])
    assert sink[0]["input"] == ["text"]


@pytest.mark.asyncio
async def test_the_callers_string_is_never_mutated(monkeypatch):
    """THIS IS THE MAIN TEST.

    In a hybrid search the same string goes BOTH to embed AND to the full-text arm. If the
    prefix were glued on at the call site, the word "query" would become a term of the full-text
    query — with Postgres or_tsquery capped at 24 terms, the first of them — and the lexical half
    of every hybrid search would be silently corrupted. So the prefix lives on the wire, and the
    caller's list stays untouched.
    """
    sink: list[dict] = []
    original = ["reset procedure"]
    await _provider(monkeypatch, sink).embed(original, kind=QUERY)
    assert original == ["reset procedure"], "the caller's string changed — the prefix would leak into SQL"
    assert sink[0]["input"] == ["query: reset procedure"]


@pytest.mark.asyncio
async def test_empty_prefix_changes_nothing(monkeypatch):
    """A symmetric model must behave exactly as before the change."""
    sink: list[dict] = []

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"embeddings": [[0.0]]}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, **kw):
            sink.append(json)
            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: _Client())
    p = OllamaProvider(base_url="http://x:11434", default_model="m", embed_model="e")
    await p.embed(["a"], kind=QUERY)
    await p.embed(["a"], kind=DOCUMENT)
    assert sink[0]["input"] == sink[1]["input"] == ["a"]


class _Spy(LLMProvider):
    name = "spy"
    plane = Plane.TRUSTED

    def __init__(self) -> None:
        self.kinds: list[str] = []

    async def generate(self, request):  # pragma: no cover - not needed by these tests
        raise NotImplementedError

    def stream(self, request):  # pragma: no cover
        raise NotImplementedError

    async def embed(self, texts, *, kind: str = DOCUMENT):
        self.kinds.append(kind)
        return [[0.0]] * len(texts)


@pytest.mark.asyncio
async def test_split_provider_forwards_the_role():
    """A pass-through wrapper that loses kind gives a silent fallback to DOCUMENT."""
    chat, emb = _Spy(), _Spy()
    await SplitTrustedProvider(chat=chat, embed_provider=emb).embed(["q"], kind=QUERY)
    assert emb.kinds == [QUERY]


@pytest.mark.asyncio
async def test_model_routed_provider_forwards_the_role():
    home, away = _Spy(), _Spy()
    await ModelRoutedProvider(default=home, routes={"m": away}).embed(["q"], kind=QUERY)
    assert home.kinds == [QUERY]


@pytest.mark.asyncio
async def test_router_forwards_the_role():
    spy = _Spy()
    await LLMRouter([spy]).embed(["q"], sensitivity=Sensitivity.CONFIDENTIAL, kind=QUERY)
    assert spy.kinds == [QUERY]


@pytest.mark.asyncio
async def test_router_defaults_to_document():
    spy = _Spy()
    await LLMRouter([spy]).embed(["t"], sensitivity=Sensitivity.CONFIDENTIAL)
    assert spy.kinds == [DOCUMENT]
