# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""Cloud-plane provider backed by the on-machine ``claude`` CLI (Claude Code, headless).

This is the MVP cloud backend: instead of a raw Anthropic API key it shells out to the
``claude`` binary running in non-interactive ``--print`` mode, reusing the operator's OAuth
subscription. In a container, the binary and the ``~/.claude`` credentials have to be mounted
in; swap this for :class:`~llmkit.cloud.AnthropicProvider` (a real service key) in production.

CRITICAL: like every cloud provider this runs in :data:`Plane.CLOUD`. The
:class:`~llmkit.router.LLMRouter` guarantees it only ever receives requests whose
sensitivity is at or below INTERNAL — data that has already been de-identified. Never call it
directly; always go through the router.

The call is deliberately tool-free (``--allowed-tools ""``) and uses ``--system-prompt`` to
replace the Claude Code agent persona with a plain assistant: we want a single completion,
not an autonomous coding agent touching the container filesystem.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Sequence

from llmkit.base import LLMProvider, LLMRequest, LLMResponse, Plane

from llmkit.errors import LLMBackendError

log = logging.getLogger(__name__)


class ClaudeCliError(LLMBackendError):
    """The ``claude`` CLI exited non-zero or returned an error result.

    Переподчинён ``LLMBackendError`` (а тот — ``RuntimeError``): имя сохраняется, MRO
    сохраняется, существующие ``pytest.raises(ClaudeCliError)`` продолжают проходить, а отказ
    облачного провайдера начинает отображаться в 502 тем же обработчиком, что и остальные."""


class ClaudeCliProvider(LLMProvider):
    name = "claude-cli"
    plane = Plane.CLOUD

    def __init__(self, *, binary: str = "claude", model: str = "", timeout: float = 300.0) -> None:
        self._binary = binary
        self._model = model
        self._timeout = timeout

    @staticmethod
    def _split(request: LLMRequest) -> tuple[str, str]:
        """Return (system_prompt, user_prompt). Non-system turns are flattened into a single
        prompt; multi-turn conversations get explicit role prefixes so context survives."""
        system = next((m.content for m in request.messages if m.role == "system"), "")
        convo = [m for m in request.messages if m.role in ("user", "assistant")]
        if len(convo) == 1:
            user = convo[0].content
        else:
            user = "\n\n".join(f"{m.role.capitalize()}: {m.content}" for m in convo)
        return system, user

    def _args(self, request: LLMRequest, system: str) -> list[str]:
        args = [self._binary, "-p", "--output-format", "json", "--allowed-tools", ""]
        if system:
            args += ["--system-prompt", system]
        model = request.model or self._model
        if model:
            args += ["--model", model]
        return args

    async def generate(self, request: LLMRequest) -> LLMResponse:
        system, user = self._split(request)
        args = self._args(request, system)
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            # binary missing / not executable / not found → treat as a CLI failure so callers
            # degrade gracefully rather than crashing the run.
            raise ClaudeCliError(f"claude CLI could not be launched: {exc}") from exc
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(user.encode()), timeout=self._timeout
            )
        except asyncio.TimeoutError as exc:
            proc.kill()
            raise ClaudeCliError(f"claude CLI timed out after {self._timeout}s") from exc

        if proc.returncode != 0:
            raise ClaudeCliError(
                f"claude CLI exited {proc.returncode}: {err.decode(errors='replace')[:500]}"
            )
        try:
            payload = json.loads(out.decode())
        except json.JSONDecodeError as exc:
            raise ClaudeCliError(f"claude CLI returned non-JSON: {out[:300]!r}") from exc
        if payload.get("is_error"):
            raise ClaudeCliError(f"claude CLI error result: {payload.get('result')!r}")

        usage = payload.get("usage", {}) or {}
        # The headless CLI may use a small model (haiku) for background bookkeeping alongside
        # the answer model; report whichever produced the most output as the primary model.
        model_usage = payload.get("modelUsage", {}) or {}
        model_used = (
            max(model_usage, key=lambda m: model_usage[m].get("outputTokens", 0))
            if model_usage
            else (request.model or self._model or "claude")
        )
        return LLMResponse(
            text=(payload.get("result") or "").strip(),
            model=model_used,
            provider=self.name,
            plane=self.plane,
            usage={
                "prompt_tokens": int(usage.get("input_tokens", 0) or 0),
                "completion_tokens": int(usage.get("output_tokens", 0) or 0),
            },
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        # Headless single-shot: no incremental streaming for the MVP. Yield the full
        # completion as one chunk so callers expecting an async iterator still work.
        resp = await self.generate(request)
        yield resp.text

    async def embed(self, texts: Sequence[str], *, kind: str = "document") -> list[list[float]]:
        # Embeddings are always computed in the trusted plane (over client data), so the
        # cloud provider deliberately does not implement them.
        raise NotImplementedError("Cloud plane does not provide embeddings by design.")
