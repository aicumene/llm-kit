# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""Trusted-plane provider backed by a local Ollama server."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence

import httpx

from llmkit.errors import streaming_transport_errors, transport_errors

from llmkit.base import DOCUMENT, QUERY, LLMProvider, LLMRequest, LLMResponse, Plane


class OllamaProvider(LLMProvider):
    name = "ollama"
    plane = Plane.TRUSTED

    def __init__(
        self,
        base_url: str,
        default_model: str,
        embed_model: str,
        *,
        timeout: float = 300.0,
        connect_timeout: float = 5.0,
        keep_alive: str = "30m",
        query_prefix: str = "",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._embed_model = embed_model
        self._timeout = timeout
        #: Установка TCP — не генерация. Один скаляр на connect+read означал, что мёртвый
        #: хост выясняется за 900 секунд; столько на рукопожатие не покупает ничего.
        self._http_timeout = httpx.Timeout(timeout, connect=connect_timeout)
        # Sent on every request so resident runners recycle on a sane cadence (avoids the stale
        # pinned-runner wedge that froze ingest). Per-request keep_alive overrides the server env.
        self._keep_alive = keep_alive
        #: Принадлежит МОДЕЛИ, не провайдеру: пусто для симметричного эмбеддера.
        self._query_prefix = query_prefix

    def _payload(self, request: LLMRequest, *, stream: bool) -> dict:
        payload = {
            "model": request.model or self._default_model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": stream,
            "keep_alive": self._keep_alive,
            "options": {
                "temperature": request.temperature,
                **({"num_predict": request.max_tokens} if request.max_tokens else {}),
            },
        }
        # Toggle reasoning on hybrid models (qwen3, deepseek-r1, …). Omitting the key leaves
        # the model's own default; sending False keeps the whole budget for the answer.
        if request.think is not None:
            payload["think"] = request.think
        # Ограниченное декодирование. Ollama называет это `format`; со схемой на выходе
        # физически не может появиться ничего, кроме объекта по ней.
        if request.response_format is not None:
            payload["format"] = request.response_format
        return payload

    @transport_errors("ollama")
    async def generate(self, request: LLMRequest) -> LLMResponse:
        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            resp = await client.post(
                f"{self._base_url}/api/chat", json=self._payload(request, stream=False)
            )
            resp.raise_for_status()
            data = resp.json()
        message = data.get("message", {}) or {}
        # Hybrid reasoning models put chain-of-thought in `thinking`. If `content` is empty
        # (e.g. the answer was truncated mid-thought by num_predict), fall back to `thinking`
        # so a parseable JSON block in the reasoning isn't silently lost.
        text = message.get("content") or message.get("thinking") or ""
        return LLMResponse(
            text=text,
            model=data.get("model", request.model or self._default_model),
            provider=self.name,
            plane=self.plane,
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
        )

    @streaming_transport_errors("ollama")
    async def stream(self, request: LLMRequest, *, usage: dict | None = None
                     ) -> AsyncIterator[str]:
        """``usage`` — приёмник счётчиков, заполняется по финальному чанку.

        Эти счётчики ВСЁ ВРЕМЯ приходили и всё время выбрасывались: цикл ниже видит чанк
        ``done`` и прерывается, не заглянув в него. Проверено на живой модели — финальный чанк
        несёт ``prompt_eval_count`` и ``eval_count``. Следствие было тихим и дорогим: чат, самый
        нагруженный путь, давал в учёт потребления РОВНО НОЛЬ, и общий счёт токенов
        занижался на всё потоковое.

        Приёмником, а не полем на провайдере: провайдер один на несколько одновременных потоков,
        и ``self.last_usage`` был бы гонкой, которая проявляется только под нагрузкой.
        """
        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            async with client.stream(
                "POST", f"{self._base_url}/api/chat", json=self._payload(request, stream=True)
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    piece = chunk.get("message", {}).get("content")
                    if piece:
                        yield piece
                    if chunk.get("done"):
                        if usage is not None:
                            usage["prompt_tokens"] = int(chunk.get("prompt_eval_count", 0) or 0)
                            usage["completion_tokens"] = int(chunk.get("eval_count", 0) or 0)
                            # Имя модели знает только провайдер, и оно приходит в финальном
                            # кадре. Без него книга разносила весь поток по пустой модели.
                            usage["model"] = str(
                                chunk.get("model") or request.model or self._default_model
                            )
                        break

    @transport_errors("ollama")
    async def embed(self, texts: Sequence[str], *, kind: str = DOCUMENT,
                    usage: dict | None = None) -> list[list[float]]:
        #: Префикс применяется ЗДЕСЬ, на самом проводе, и нигде выше. Выше по стеку та же строка
        #: уходит ещё и в лексическое плечо поиска — см. LLMProvider.embed.
        prefix = self._query_prefix if kind == QUERY else ""
        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            resp = await client.post(
                f"{self._base_url}/api/embed",
                json={
                    "model": self._embed_model,
                    "input": [prefix + t for t in texts] if prefix else list(texts),
                    "keep_alive": self._keep_alive,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            if usage is not None:
                # Эмбеддинг — настоящая работа GPU, и до сих пор она не учитывалась ВООБЩЕ:
                # ``embed`` проходил мимо счётчика через ``__getattr__`` обёртки. Ответ несёт
                # ``prompt_eval_count`` (проверено на живой модели), так что измерять было чем.
                usage["prompt_tokens"] = int(data.get("prompt_eval_count", 0) or 0)
                usage["completion_tokens"] = 0
                # Имя модели знает только провайдер. Без него книга разносит весь эмбеддинг
                # по пустому ключу — см. MeteredRouter.embed.
                usage["model"] = self._embed_model
            return data.get("embeddings", [])

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False
