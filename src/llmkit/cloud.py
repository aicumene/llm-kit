# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""Cloud-plane provider (Anthropic Claude).

CRITICAL: this provider runs in :data:`Plane.CLOUD`. The :class:`~llmkit.router.LLMRouter`
guarantees it only ever receives requests whose sensitivity is at or below INTERNAL — i.e.
data that has already been de-identified. It must never be called directly; always go through
the router.

A minimal implementation: no prompt caching, no tools.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from llmkit.base import LLMProvider, LLMRequest, LLMResponse, Plane


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    plane = Plane.CLOUD

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def _client(self):
        # Lazy import: the SDK is an optional extra (``llm-kit[anthropic]``), needed only when
        # the cloud plane is on.
        from anthropic import AsyncAnthropic

        return AsyncAnthropic(api_key=self._api_key)

    @staticmethod
    def _split(request: LLMRequest) -> tuple[str | None, list[dict]]:
        system = next((m.content for m in request.messages if m.role == "system"), None)
        msgs = [
            {"role": m.role, "content": m.content}
            for m in request.messages
            if m.role in ("user", "assistant")
        ]
        return system, msgs

    async def generate(self, request: LLMRequest) -> LLMResponse:
        system, msgs = self._split(request)
        resp = await self._client().messages.create(
            model=request.model or self._model,
            max_tokens=request.max_tokens or 2048,
            temperature=request.temperature,
            system=system or "",
            messages=msgs,
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        return LLMResponse(
            text=text,
            model=resp.model,
            provider=self.name,
            plane=self.plane,
            usage={
                "prompt_tokens": resp.usage.input_tokens,
                "completion_tokens": resp.usage.output_tokens,
            },
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        system, msgs = self._split(request)
        async with self._client().messages.stream(
            model=request.model or self._model,
            max_tokens=request.max_tokens or 2048,
            temperature=request.temperature,
            system=system or "",
            messages=msgs,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def embed(self, texts: Sequence[str], *, kind: str = "document") -> list[list[float]]:
        # Embeddings are always computed in the trusted plane (over client data), so the
        # cloud provider deliberately does not implement them.
        raise NotImplementedError("Cloud plane does not provide embeddings by design.")
