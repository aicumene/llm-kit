# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""Trusted-plane provider backed by a vLLM OpenAI-compatible server.

vLLM serves the same on-prem models as Ollama but with **continuous batching** and
PagedAttention, so one GPU sustains many more concurrent generations — the lever for the
user-concurrency axis (Ollama's ``OLLAMA_NUM_PARALLEL`` caps in-flight requests hard).
It speaks the OpenAI REST dialect (``/chat/completions``, ``/completions``, ``/embeddings``).

This is a :data:`Plane.TRUSTED` provider — it runs on the operator's own GPUs, so the router
treats it exactly like Ollama and the privacy invariant is unchanged.

Two classes ship here:

* :class:`VLLMProvider` — chat + embeddings over a vLLM endpoint.
* :class:`SplitTrustedProvider` — a composite trusted provider that routes *generation* to
  one backend (vLLM) and *embeddings* to another (a dedicated embedder). The router picks a
  single trusted provider and calls ``generate``/``embed`` on it, so wrapping the two behind
  one provider keeps the router untouched while letting chat and embeddings scale apart.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence

import httpx

from llmkit.errors import streaming_transport_errors, transport_errors

from llmkit.base import DOCUMENT, QUERY, LLMProvider, LLMRequest, LLMResponse, Plane


class VLLMProvider(LLMProvider):
    name = "vllm"
    plane = Plane.TRUSTED

    def __init__(
        self,
        base_url: str,
        chat_model: str,
        *,
        embed_model: str | None = None,
        timeout: float = 900.0,
        query_prefix: str = "",
    ) -> None:
        # base_url is the OpenAI root, e.g. "http://vllm:8000/v1".
        self._base_url = base_url.rstrip("/")
        self._chat_model = chat_model
        self._embed_model = embed_model or chat_model
        self._timeout = timeout
        self._http_timeout = httpx.Timeout(timeout, connect=5.0)
        self._query_prefix = query_prefix

    def _payload(self, request: LLMRequest, *, stream: bool) -> dict:
        payload: dict = {
            "model": request.model or self._chat_model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": stream,
            "temperature": request.temperature,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        # Hybrid reasoning toggle for qwen3 / deepseek-r1 under vLLM. The OpenAI server exposes
        # the chat-template switch via extra body; mirrors Ollama's ``think`` so callers that
        # pass ``think=False`` keep the whole budget for the answer.
        if request.think is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": request.think}
        return payload

    @transport_errors("vllm")
    async def generate(self, request: LLMRequest) -> LLMResponse:
        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions", json=self._payload(request, stream=False)
            )
            resp.raise_for_status()
            data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=data.get("model", request.model or self._chat_model),
            provider=self.name,
            plane=self.plane,
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
        )

    @streaming_transport_errors("vllm")
    async def stream(self, request: LLMRequest, *, usage: dict | None = None
                     ) -> AsyncIterator[str]:
        # ``usage`` ПРИНИМАЕТСЯ и не заполняется: SSE vLLM счётчиков не несёт. Принять и
        # промолчать честнее, чем не принять: иначе каждый вызывающий, пробрасывающий приёмник,
        # уходит в откат по TypeError, и «не умеет считать» становится неотличимо от «сломан».
        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            async with client.stream(
                "POST", f"{self._base_url}/chat/completions", json=self._payload(request, stream=True)
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if not body or body == "[DONE]":
                        if body == "[DONE]":
                            break
                        continue
                    chunk = json.loads(body)
                    delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        yield piece

    @transport_errors("vllm")
    async def embed(self, texts: Sequence[str], *, kind: str = DOCUMENT) -> list[list[float]]:
        #: Спящий путь (chat_backend="vllm"), и правится вместе с ollama именно поэтому:
        #: провод, который сегодня не под нагрузкой, завтра включают одной настройкой.
        prefix = self._query_prefix if kind == QUERY else ""
        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            resp = await client.post(
                f"{self._base_url}/embeddings",
                json={"model": self._embed_model,
                      "input": [prefix + t for t in texts] if prefix else list(texts)},
            )
            resp.raise_for_status()
            rows = resp.json().get("data", [])
        # Preserve input order — the OpenAI schema carries an explicit ``index`` per row.
        rows = sorted(rows, key=lambda r: r.get("index", 0))
        return [r.get("embedding", []) for r in rows]

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/models")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False


class ModelRoutedProvider(LLMProvider):
    """One trusted provider whose generations go to different hosts depending on the model asked for.

    ``SplitTrustedProvider`` splits by *kind of work* (chat vs embed). This splits by *which
    model* — because a fleet is not homogeneous. Measured 02.09.2026: the 27B general models and
    a 9B domain tune both lived on one host, so a 9B request that needs 5s of GPU queued behind
    27B generations that need 10s, and the second GPU in the fleet sat at 0% utilisation with 22
    of its 24 GB free. Model identity, not protocol, is what decides where the weights already are.

    ``routes`` maps a model name to the backend that holds it. Anything not named falls through to
    ``default``, so a deployment that configures no routes keeps the single-backend shape exactly.
    Embeddings always go to ``default``: this class has an opinion about generation only, and
    layering two independent opinions about the embed path is how you get a silent second embedder.
    """

    name = "trusted-model-routed"
    plane = Plane.TRUSTED

    def __init__(self, *, default: LLMProvider, routes: Mapping[str, LLMProvider],
                 default_model: str = "") -> None:
        if any(b.plane is not Plane.TRUSTED for b in (default, *routes.values())):
            raise ValueError("ModelRoutedProvider composes trusted-plane backends only.")
        self._default = default
        self._routes = dict(routes)
        self._default_model = default_model

    def _backend_for(self, request: LLMRequest) -> LLMProvider:
        # A request that names no model means THE CHAT MODEL — every backend fills that name in
        # itself. Since 25.09.2026 the chat model can have a route of its own (llama-server with
        # MTP on a separate host); routing the empty name to ``default`` sent such requests to
        # Ollama, which then asked itself for a model it does not hold.
        return self._routes.get(request.model or self._default_model or "", self._default)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return await self._backend_for(request).generate(request)

    def stream(self, request: LLMRequest, *, usage: dict | None = None) -> AsyncIterator[str]:
        """Приёмник счётчиков пробрасывается НАСКВОЗЬ.

        Без этого на развёрнутой конфигурации (``OLLAMA_MODEL_HOSTS`` непуст ⇒ фабрика строит
        ``SplitTrustedProvider(chat=ModelRoutedProvider(...))``) приёмник упирался ИМЕННО СЮДА:
        Split звал ``stream(request, usage=...)``, здесь его не принимали, срабатывал откат — и
        до Ollama, которая приёмник понимает, он не доходил НИКОГДА. Каждый потоковый ответ
        писал в книгу нулевые токены, то есть дефект, объявленный закрытым, оставался живым.
        """
        backend = self._backend_for(request)
        try:
            return backend.stream(request, usage=usage)
        except TypeError:
            return backend.stream(request)

    async def embed(self, texts: Sequence[str], *, kind: str = DOCUMENT,
                    usage: dict | None = None) -> list[list[float]]:
        # То же самое и здесь: составной провайдер, глотающий приёмник, делает отказ молчаливым.
        try:
            return await self._default.embed(texts, kind=kind, usage=usage)
        except TypeError:
            return await self._default.embed(texts, kind=kind)

    async def health(self) -> bool:
        results = []
        for backend in (self._default, *self._routes.values()):
            health = getattr(backend, "health", None)
            results.append(await health() if health else True)
        return all(results)


class SplitTrustedProvider(LLMProvider):
    """One trusted provider whose generation and embedding run on different backends.

    Generation goes to ``chat`` (e.g. vLLM, continuous-batched); embeddings go to
    ``embed_provider`` (e.g. a dedicated embedder), so the embed path never queues behind
    27B generations and each scales on its own hardware. Both must be :data:`Plane.TRUSTED`.
    """

    name = "trusted-split"
    plane = Plane.TRUSTED

    def __init__(self, *, chat: LLMProvider, embed_provider: LLMProvider) -> None:
        if chat.plane is not Plane.TRUSTED or embed_provider.plane is not Plane.TRUSTED:
            raise ValueError("SplitTrustedProvider composes trusted-plane backends only.")
        self._chat = chat
        self._embed = embed_provider

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return await self._chat.generate(request)

    def stream(self, request: LLMRequest, *, usage: dict | None = None) -> AsyncIterator[str]:
        # Приёмник счётчиков пробрасывается НАСКВОЗЬ. Составной провайдер, глотающий его, делает
        # отказ молчаливым: строка учёта пишется с нулём токенов, и «ничего не стоило»
        # становится неотличимо от «не смогли посчитать». Поймано сквозным прогоном — приёмник
        # упирался именно сюда, при том что и роутер, и Ollama его уже понимали.
        try:
            return self._chat.stream(request, usage=usage)
        except TypeError:
            return self._chat.stream(request)

    async def embed(self, texts: Sequence[str], *, kind: str = DOCUMENT,
                    usage: dict | None = None) -> list[list[float]]:
        try:
            return await self._embed.embed(texts, kind=kind, usage=usage)
        except TypeError:
            return await self._embed.embed(texts, kind=kind)

    async def health(self) -> bool:
        results = []
        for backend in (self._chat, self._embed):
            health = getattr(backend, "health", None)
            results.append(await health() if health else True)
        return all(results)
