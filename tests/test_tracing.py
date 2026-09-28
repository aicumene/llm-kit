# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""The tracing hook: the router's spans reach an installed span factory, carry identifiers only,
and are no-ops without one."""
from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import contextmanager

import pytest

from llmkit import tracing
from llmkit.base import LLMMessage, LLMProvider, LLMRequest, LLMResponse, Plane
from llmkit.router import LLMRouter

CLIENT_TEXT = "text of a client document"


class _Trusted(LLMProvider):
    name = "fake"
    plane = Plane.TRUSTED

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(text="ok", model="m", provider=self.name, plane=self.plane)

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        yield "ok"

    async def embed(self, texts: Sequence[str], *, kind: str = "document") -> list[list[float]]:
        return [[0.0] for _ in texts]


def _req() -> LLMRequest:
    return LLMRequest(messages=[LLMMessage(role="user", content=CLIENT_TEXT)], model="m")


@pytest.fixture(autouse=True)
def _no_factory_leaks():
    yield
    tracing.set_span_factory(None)


async def test_router_spans_reach_the_installed_factory_with_identifiers_only():
    spans: list[tuple[str, dict]] = []

    @contextmanager
    def factory(name, **attrs):
        spans.append((name, attrs))
        yield {"name": name}

    tracing.set_span_factory(factory)
    router = LLMRouter([_Trusted()])
    await router.generate(_req())
    assert [c async for c in router.stream(_req())] == ["ok"]
    await router.embed(["a", "b"])

    assert [name for name, _ in spans] == ["llm.generate", "llm.stream", "llm.embed"]
    assert spans[0][1] == {"llm.plane": "trusted", "llm.provider": "fake", "llm.model": "m",
                           "llm.sensitivity": "Confidential"}
    assert spans[2][1]["llm.batch"] == 2
    assert not any(CLIENT_TEXT in str(v) for _, attrs in spans for v in attrs.values())


async def test_without_a_factory_spans_are_no_ops():
    tracing.set_span_factory(None)
    with tracing.start_span("anything", a=1) as span:
        assert span is None
    resp = await LLMRouter([_Trusted()]).generate(_req())
    assert resp.text == "ok"


def test_an_exception_inside_a_span_reaches_the_factory_and_propagates():
    recorded: list[BaseException] = []

    @contextmanager
    def factory(name, **attrs):
        try:
            yield None
        except Exception as exc:
            recorded.append(exc)
            raise

    tracing.set_span_factory(factory)
    with pytest.raises(RuntimeError):
        with tracing.start_span("failing"):
            raise RuntimeError("boom")
    assert len(recorded) == 1 and str(recorded[0]) == "boom"
    assert tracing.span_factory() is factory
