# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""Ошибки слоя моделей — то, чего у него не было.

Провайдер — единственное место, которое знает, что вызов модели не удался, но до сих пор он
выбрасывал наружу тип ЧУЖОЙ библиотеки (``httpx.*``). Из-за этого назвать отказ не мог никто
выше: вызывающие либо не ловили вовсе, либо ловили ``except Exception`` вслепую, и один и тот же
отказ наблюдался тремя разными способами, а наружу уходил текстовый 500 в обход CORS.

Предок — ``RuntimeError``, и это НЕ косметика: код, который уже отображает ``RuntimeError`` в
503, продолжает работать без единой правки.

Текст сообщения СОЗНАТЕЛЬНО не несёт URL. ``str(httpx.HTTPStatusError)`` содержит адрес хоста
моделей, и он уходил прямо в браузер — это разведка внутренней сети, а не диагностика.
"""

from __future__ import annotations

import functools
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx

__all__ = [
    "LLMError",
    "LLMUnavailable",
    "LLMBackendError",
    "transport_errors",
    "streaming_transport_errors",
]


class LLMError(RuntimeError):
    """Слой моделей не смог ответить."""


class LLMUnavailable(LLMError):
    """До хоста моделей не достучались: сеть, таймаут, отказ соединения. → 503."""


class LLMBackendError(LLMError):
    """Хост ответил отказом: 4xx/5xx, или CLI вышел с ненулевым кодом. → 502."""


def _backend(provider: str, exc: httpx.HTTPStatusError) -> LLMBackendError:
    return LLMBackendError(f"{provider}: backend returned {exc.response.status_code}")


def _unavailable(provider: str, exc: httpx.HTTPError) -> LLMUnavailable:
    return LLMUnavailable(f"{provider}: {type(exc).__name__}")


def transport_errors(provider: str) -> Callable[..., Any]:
    """Обычный (не потоковый) вызов: httpx-исключение → типизированный отказ слоя."""

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await fn(*args, **kwargs)
            except httpx.HTTPStatusError as exc:
                raise _backend(provider, exc) from exc
            except httpx.HTTPError as exc:
                raise _unavailable(provider, exc) from exc

        return wrapper

    return deco


async def _guarded_stream(provider: str, inner: AsyncIterator[Any]) -> AsyncIterator[Any]:
    try:
        async for chunk in inner:
            yield chunk
    except httpx.HTTPStatusError as exc:
        raise _backend(provider, exc) from exc
    except httpx.HTTPError as exc:
        raise _unavailable(provider, exc) from exc


def streaming_transport_errors(provider: str) -> Callable[..., Any]:
    """То же для асинхронного генератора.

    ``wrapper`` — ОБЫЧНАЯ функция, а не асинхронный генератор, и это не стилистика. Если бы она
    сама была генератором, связывание аргументов откладывалось бы до первого ``__anext__`` — то
    есть происходило бы УЖЕ ВНЕ ``try`` вызывающего. А вокруг ``stream()`` в этом пакете стоят
    сторожа ``except TypeError`` (``LLMRouter.stream``, ``SplitTrustedProvider.stream``,
    ``MeteredRouter.stream``): ими отличают провайдера, который умеет принять приёмник счётчиков,
    от того, который не умеет. Отложенное связывание убивает их все разом и молча — поток
    обрывается на первом чанке без единого токена и без кадра ошибки.

    Поэтому ``fn(*args, **kwargs)`` вызывается ЗДЕСЬ, синхронно: несоответствие сигнатуры даёт
    ``TypeError`` на вызове, ровно там, где его ждут.
    """

    def deco(fn: Callable[..., AsyncIterator[Any]]) -> Callable[..., AsyncIterator[Any]]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            return _guarded_stream(provider, fn(*args, **kwargs))

        return wrapper

    return deco
