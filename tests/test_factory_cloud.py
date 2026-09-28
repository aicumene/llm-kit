# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""The factory's cloud slot: which cloud provider is registered, and the hook that may wrap it.

Re-expressed from the source's relay tests. The relay itself (organization keys) stayed with the
product; what is generic is that the factory hands its default cloud provider to a hook and
registers what comes back — and that air-gap means no cloud at all, hook or not.
"""
from __future__ import annotations

from llmkit.base import LLMProvider, Plane
from llmkit.claude_cli import ClaudeCliProvider
from llmkit.cloud import AnthropicProvider
from llmkit.config import ModelConfig
from llmkit.factory import build_router


class _Wrapper(LLMProvider):
    name = "wrapper"
    plane = Plane.CLOUD

    def __init__(self, fallback: LLMProvider | None) -> None:
        self.fallback = fallback

    async def generate(self, request):  # pragma: no cover - not called here
        raise NotImplementedError

    def stream(self, request):  # pragma: no cover
        raise NotImplementedError

    async def embed(self, texts, *, kind: str = "document"):  # pragma: no cover
        raise NotImplementedError


def _cloud(router) -> list:
    return [p for p in router.providers if p.plane is Plane.CLOUD]


def test_the_hook_gets_the_default_cloud_provider_and_what_it_returns_is_registered():
    seen: list = []

    def wrap(cloud):
        seen.append(cloud)
        return _Wrapper(cloud)

    cloud = _cloud(build_router(ModelConfig(claude_cli_enabled=True), wrap_cloud=wrap))
    assert len(cloud) == 1 and isinstance(cloud[0], _Wrapper)
    assert isinstance(seen[0], ClaudeCliProvider) and cloud[0].fallback is seen[0]


def test_without_a_hook_nothing_changes():
    cloud = _cloud(build_router(ModelConfig(claude_cli_enabled=True)))
    assert len(cloud) == 1 and isinstance(cloud[0], ClaudeCliProvider)


def test_the_hook_is_asked_even_when_there_is_no_default_cloud_provider():
    seen: list = []
    cloud = _cloud(build_router(ModelConfig(), wrap_cloud=lambda c: seen.append(c)))
    assert seen == [None] and cloud == []


def test_air_gap_still_means_no_cloud_at_all():
    asked: list = []
    router = build_router(ModelConfig(airgap=True, claude_cli_enabled=True, anthropic_api_key="placeholder"),
                          wrap_cloud=lambda c: asked.append(c) or _Wrapper(c))
    assert _cloud(router) == [] and asked == []


def test_an_api_key_wins_over_the_cli_and_stays_out_of_the_repr():
    cfg = ModelConfig(anthropic_api_key="placeholder", claude_cli_enabled=True)
    cloud = _cloud(build_router(cfg))
    assert len(cloud) == 1 and isinstance(cloud[0], AnthropicProvider)
    assert "placeholder" not in repr(cfg)
