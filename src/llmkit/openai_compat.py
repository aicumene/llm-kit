# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""Доверенный провайдер по протоколу OpenAI (SGLang / vLLM) — адрес с «/v1» на конце.

ЗАЧЕМ, КОГДА ЕСТЬ ``vllm.py``. ``VLLMProvider`` живёт в отдельной ветке фабрики
(``chat_backend="vllm"``): один адрес на весь чат, свой набор настроек, приёмник счётчиков
принимается и не заполняется. Пилоту SGLang (12.09.2026) нужно другое: ОДИН маршрут по имени
модели (``model=http://…:30000/v1``) рядом с обычными Ollama-маршрутами, через ту же карту
маршрутов (``ollama_model_hosts``), с тем же учётом токенов и тем же ограниченным декодированием,
что у Ollama. Поэтому этот класс повторяет договор :class:`llmkit.ollama.OllamaProvider`
буквально — сигнатуры ``generate``/``stream``/``embed``, форма ``usage``, приёмник счётчиков,
префикс запроса у эмбеддера, типы отказов, — и отличается только проводом.

ДИАЛЕКТ ВЫБИРАЕТСЯ ПО АДРЕСУ: «/v1» на конце пути = OpenAI-совместимый сервер, иначе Ollama
(см. :func:`is_openai_url` и докстроку ``llmkit.factory``). Это правило, а не эвристика: у
OpenAI-серверов корень API так и называется, а у Ollama такого пути нет.

ДОВЕРИЕ — ВЫШЕ. ``plane = TRUSTED`` здесь атрибут класса, как у Ollama, и об адресе он не
знает; периметр держит :func:`llmkit.trust.check_model_url`, через который фабрика и
резолвер пропускают КАЖДЫЙ адрес до того, как он попадёт сюда.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence

import httpx

from llmkit.base import DOCUMENT, QUERY, LLMProvider, LLMRequest, LLMResponse, Plane
from llmkit.errors import streaming_transport_errors, transport_errors

#: Имя схемы в ``response_format``: серверу оно нужно формально, нам — чтобы в его журнале
#: было видно, чей запрос. Читается при каждом запросе, поэтому развёртывание может задать
#: своё (``openai_compat.SCHEMA_NAME = "..."``) один раз при старте.
SCHEMA_NAME = "llmkit"


def is_openai_url(url: str | None) -> bool:
    """«/v1» на конце пути — OpenAI-совместимый сервер (SGLang/vLLM); всё остальное — Ollama."""
    return (url or "").strip().rstrip("/").endswith("/v1")


class OpenAICompatProvider(LLMProvider):
    name = "openai-compat"
    plane = Plane.TRUSTED

    def __init__(
        self,
        base_url: str,
        default_model: str,
        embed_model: str,
        *,
        timeout: float = 300.0,
        connect_timeout: float = 5.0,
        query_prefix: str = "",
    ) -> None:
        # Корень OpenAI API, например "http://127.0.0.1:30000/v1".
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._embed_model = embed_model
        self._timeout = timeout
        #: Как у Ollama: рукопожатие — не генерация, мёртвый хост должен выясняться за секунды.
        self._http_timeout = httpx.Timeout(timeout, connect=connect_timeout)
        #: Принадлежит МОДЕЛИ, не провайдеру: пусто для симметричного эмбеддера.
        self._query_prefix = query_prefix

    def _payload(self, request: LLMRequest, *, stream: bool) -> dict:
        payload: dict = {
            "model": request.model or self._default_model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": stream,
            "temperature": request.temperature,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        # Переключатель рассуждений гибридных моделей (qwen3.x). У клиента OpenAI это
        # ``extra_body``, на проводе — обычное поле верхнего уровня; SGLang и vLLM читают его
        # как аргумент шаблона чата. Зеркало ``think`` у Ollama: None = умолчание модели.
        if request.think is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": request.think}
        # Ограниченное декодирование. Ollama берёт схему как есть в ``format``; здесь она
        # заворачивается в ``json_schema`` со ``strict`` — иначе сервер примет её как подсказку,
        # а не как грамматику, и провайдер станет «умеет схему» только на словах.
        if isinstance(request.response_format, dict):
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": SCHEMA_NAME, "schema": request.response_format,
                                "strict": True},
            }
        if stream:
            # Без этого счётчики в потоке не приходят ВОВСЕ, и книга учёта писала бы ноль на
            # самом нагруженном пути — ровно тот дефект, который у Ollama чинили в stream().
            payload["stream_options"] = {"include_usage": True}
        return payload

    @transport_errors("openai-compat")
    async def generate(self, request: LLMRequest) -> LLMResponse:
        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions", json=self._payload(request, stream=False)
            )
            resp.raise_for_status()
            data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        # Как у Ollama: если видимый ответ пуст (обрезан max_tokens посреди рассуждения),
        # берём канал рассуждений — разбираемый JSON в нём не должен пропасть молча.
        text = message.get("content") or message.get("reasoning_content") or ""
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=data.get("model", request.model or self._default_model),
            provider=self.name,
            plane=self.plane,
            usage={
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            },
        )

    @streaming_transport_errors("openai-compat")
    async def stream(self, request: LLMRequest, *, usage: dict | None = None
                     ) -> AsyncIterator[str]:
        """``usage`` — приёмник счётчиков, заполняется по кадру с ``usage``.

        Сервер OpenAI-диалекта с ``stream_options.include_usage`` присылает счётчики ОТДЕЛЬНЫМ
        последним кадром: ``choices`` в нём пуст, ``usage`` непуст. Заполняем ту же форму, что
        Ollama (``prompt_tokens``, ``completion_tokens``, ``model``), чтобы ``MeteredRouter``
        не отличал бэкенды. Если кадра не пришло — приёмник остаётся пуст, и книга честно
        пишет «не измерено», а не ноль.
        """
        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            async with client.stream(
                "POST", f"{self._base_url}/chat/completions",
                json=self._payload(request, stream=True),
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if not body:
                        continue
                    if body == "[DONE]":
                        break
                    chunk = json.loads(body)
                    delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        yield piece
                    counters = chunk.get("usage")
                    if counters and usage is not None:
                        usage["prompt_tokens"] = int(counters.get("prompt_tokens", 0) or 0)
                        usage["completion_tokens"] = int(counters.get("completion_tokens", 0) or 0)
                        usage["model"] = str(
                            chunk.get("model") or request.model or self._default_model
                        )

    @transport_errors("openai-compat")
    async def embed(self, texts: Sequence[str], *, kind: str = DOCUMENT,
                    usage: dict | None = None) -> list[list[float]]:
        #: Префикс применяется ЗДЕСЬ, на самом проводе, и нигде выше — см. LLMProvider.embed.
        prefix = self._query_prefix if kind == QUERY else ""
        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            resp = await client.post(
                f"{self._base_url}/embeddings",
                json={
                    "model": self._embed_model,
                    "input": [prefix + t for t in texts] if prefix else list(texts),
                },
            )
            resp.raise_for_status()
            data = resp.json()
        if usage is not None:
            counters = data.get("usage") or {}
            usage["prompt_tokens"] = int(counters.get("prompt_tokens", 0) or 0)
            usage["completion_tokens"] = 0
            usage["model"] = self._embed_model
        # Порядок входа сохраняется по явному ``index`` строки — схема OpenAI его несёт.
        rows = sorted(data.get("data") or [], key=lambda r: r.get("index", 0))
        return [r.get("embedding", []) for r in rows]

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/models")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False
