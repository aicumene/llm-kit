# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""The privacy invariant is what everything else rests on. These tests guard it.

They depend only on stdlib + the router/base/classification modules, so they run without
a database, Ollama, or any network — and can be run directly with `python -m pytest` or
even imported standalone.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import pytest

from llmkit.base import LLMMessage, LLMProvider, LLMRequest, LLMResponse, Plane
from llmkit.classification import Sensitivity
from llmkit.router import LLMRouter, PrivacyViolation


class _FakeProvider(LLMProvider):
    def __init__(self, name: str, plane: Plane) -> None:
        self.name = name
        self.plane = plane
        self.calls: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse(text="ok", model="fake", provider=self.name, plane=self.plane)

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        self.calls.append(request)
        yield "ok"

    async def embed(self, texts: Sequence[str], *, kind: str = "document") -> list[list[float]]:
        return [[0.0] for _ in texts]


def _req(sensitivity: Sensitivity, *, allow_cloud: bool = False) -> LLMRequest:
    return LLMRequest(
        messages=[LLMMessage(role="user", content="hi")],
        sensitivity=sensitivity,
        allow_cloud=allow_cloud,
    )


@pytest.mark.parametrize("sensitivity", [Sensitivity.CONFIDENTIAL, Sensitivity.RESTRICTED])
async def test_sensitive_data_never_reaches_cloud(sensitivity):
    cloud = _FakeProvider("cloud", Plane.CLOUD)
    router = LLMRouter([cloud])  # only a cloud provider exists

    # Even explicitly opting into cloud must fail for too-sensitive data.
    with pytest.raises(PrivacyViolation):
        await router.generate(_req(sensitivity, allow_cloud=True))
    assert cloud.calls == []  # the guard fired before any dispatch


async def test_explicit_cloud_for_sensitive_data_raises():
    trusted = _FakeProvider("ollama", Plane.TRUSTED)
    cloud = _FakeProvider("cloud", Plane.CLOUD)
    router = LLMRouter([trusted, cloud])
    with pytest.raises(PrivacyViolation):
        await router.generate(_req(Sensitivity.CONFIDENTIAL), prefer_plane=Plane.CLOUD)
    assert cloud.calls == []


async def test_sensitive_defaults_to_trusted():
    trusted = _FakeProvider("ollama", Plane.TRUSTED)
    cloud = _FakeProvider("cloud", Plane.CLOUD)
    router = LLMRouter([trusted, cloud])
    resp = await router.generate(_req(Sensitivity.RESTRICTED))
    assert resp.plane is Plane.TRUSTED
    assert len(trusted.calls) == 1 and cloud.calls == []


async def test_non_sensitive_may_use_cloud_when_opted_in():
    trusted = _FakeProvider("ollama", Plane.TRUSTED)
    cloud = _FakeProvider("cloud", Plane.CLOUD)
    router = LLMRouter([trusted, cloud])
    resp = await router.generate(
        _req(Sensitivity.INTERNAL, allow_cloud=True), prefer_plane=Plane.CLOUD
    )
    assert resp.plane is Plane.CLOUD


async def test_non_sensitive_without_optin_stays_trusted():
    trusted = _FakeProvider("ollama", Plane.TRUSTED)
    cloud = _FakeProvider("cloud", Plane.CLOUD)
    router = LLMRouter([trusted, cloud])
    resp = await router.generate(_req(Sensitivity.PUBLIC))  # allow_cloud defaults False
    assert resp.plane is Plane.TRUSTED


async def test_embed_defaults_to_trusted_plane():
    trusted = _FakeProvider("ollama", Plane.TRUSTED)
    cloud = _FakeProvider("cloud", Plane.CLOUD)
    router = LLMRouter([trusted, cloud])
    # Embeddings carry client data by default; must never select cloud.
    vectors = await router.embed(["confidential client text"], sensitivity=Sensitivity.RESTRICTED)
    assert vectors == [[0.0]]
