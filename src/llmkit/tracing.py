# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""Optional tracing hook.

The router opens a span around every dispatch (``llm.generate``, ``llm.stream``, ``llm.embed``)
whose attributes are identifiers only — plane, provider, model, sensitivity label, batch size —
never content. The toolkit itself carries no tracing dependency: until a span factory is
installed, :func:`start_span` is a no-op that yields ``None``.

A deployment installs one at process start, for example a facade over OpenTelemetry::

    from llmkit import tracing

    tracing.set_span_factory(start_span)   # start_span(name, **attrs) -> context manager

The factory owns every decision about the span: which attributes to keep, how to record an
exception, what to do when tracing is off. It is process-wide on purpose — routers are built in
several places, and a per-router setting would leave the ones built elsewhere untraced.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any

#: ``factory(name, **attrs)`` → a context manager around the span (it may yield the span object).
SpanFactory = Callable[..., AbstractContextManager[Any]]

_factory: SpanFactory | None = None


def set_span_factory(factory: SpanFactory | None) -> None:
    """Install the process-wide span factory; ``None`` removes it."""
    global _factory
    _factory = factory


def span_factory() -> SpanFactory | None:
    """The installed span factory, or ``None``."""
    return _factory


@contextmanager
def start_span(name: str, **attrs: Any) -> Iterator[Any]:
    """Open a span through the installed factory; a no-op yielding ``None`` without one."""
    factory = _factory
    if factory is None:
        yield None
        return
    with factory(name, **attrs) as span:
        yield span


__all__ = ["SpanFactory", "set_span_factory", "span_factory", "start_span"]
