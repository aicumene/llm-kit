# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""A transparent router wrapper that accumulates token usage across an agent run.

Each ``LLMResponse`` carries a ``usage`` dict (prompt/completion tokens). Wrapping the run's router
in ``UsageRouter`` sums those across every ``generate``, ``stream`` and ``embed`` call, so after the
run we can attribute exactly what the agent consumed (tokens → cost). Everything not defined here
(``providers``, ``health``, …) passes through unchanged, so it composes with other router wrappers.

For per-task accounting on a router shared by a whole process, see :mod:`llmkit.metering`.
"""

from __future__ import annotations

from typing import Any


class UsageRouter:
    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    async def generate(self, request: Any, **kwargs: Any):
        resp = await self._inner.generate(request, **kwargs)
        u = getattr(resp, "usage", None) or {}
        self.usage["prompt_tokens"] += int(u.get("prompt_tokens", 0) or 0)
        self.usage["completion_tokens"] += int(u.get("completion_tokens", 0) or 0)
        self.usage["calls"] += 1
        return resp

    async def stream(self, request: Any, **kwargs: Any):
        """Поток считается наравне с обычной генерацией.

        Прежний комментарий гласил, что «потоковые ответы не несут usage надёжно», и это было
        неверно для Ollama: финальный чанк несёт ``prompt_eval_count`` и ``eval_count``, код их
        видел и выбрасывал. Цена ошибки была не в точности, а в полноте: ЧАТ — самый нагруженный
        путь — давал в учёт потребления ровно ноль.

        Приёмник передаётся вниз, а не читается с провайдера: один провайдер обслуживает
        несколько одновременных потоков, и поле на нём было бы гонкой.
        """
        self.usage["calls"] += 1
        sink: dict[str, int] = {}
        try:
            async for chunk in self._inner.stream(request, usage=sink, **kwargs):
                yield chunk
        except TypeError:
            # Провайдер, не знающий про приёмник (облачный, vLLM): считаем вызов и не врём про
            # токены. Молчаливый ноль лучше выдуманного числа, но он обязан быть отличим —
            # поэтому ``calls`` растёт всегда.
            async for chunk in self._inner.stream(request, **kwargs):
                yield chunk
            return
        self.usage["prompt_tokens"] += int(sink.get("prompt_tokens", 0) or 0)
        self.usage["completion_tokens"] += int(sink.get("completion_tokens", 0) or 0)

    async def embed(self, texts: Any, **kwargs: Any):
        """Эмбеддинг — тоже потребление, и до сих пор оно не учитывалось вовсе.

        ``embed`` уходил в провайдер через ``__getattr__`` и счётчика не касался. На корпусе в
        5301 кусок это заметная доля работы GPU, невидимая в отчёте о расходе.
        """
        self.usage["calls"] += 1
        sink: dict[str, int] = {}
        try:
            out = await self._inner.embed(texts, usage=sink, **kwargs)
        except TypeError:
            return await self._inner.embed(texts, **kwargs)
        self.usage["prompt_tokens"] += int(sink.get("prompt_tokens", 0) or 0)
        return out

    def __getattr__(self, name: str) -> Any:
        # Delegate everything not defined here (providers, health, …) to the wrapped router.
        return getattr(self._inner, name)
