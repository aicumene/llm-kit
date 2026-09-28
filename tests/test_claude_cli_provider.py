# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""ClaudeCliProvider — the MVP cloud backend (on-machine `claude` CLI, headless).

These tests stub the subprocess so they run without the binary, network, or creds. They
cover prompt assembly, the JSON-result parsing, model-from-modelUsage selection, and error
paths. The privacy invariant itself is guarded in test_router_invariant.py — here we only
check the provider declares Plane.CLOUD so the router gates it.
"""

from __future__ import annotations

import json

import pytest

from llmkit.base import LLMMessage, LLMRequest, Plane
from llmkit.claude_cli import ClaudeCliError, ClaudeCliProvider


class _FakeProc:
    def __init__(self, *, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0) -> None:
        self._stdout, self._stderr, self.returncode = stdout, stderr, returncode
        self.stdin_seen: bytes | None = None

    async def communicate(self, data: bytes | None = None):
        self.stdin_seen = data
        return self._stdout, self._stderr

    def kill(self) -> None:  # pragma: no cover - only hit on timeout
        pass


def _patch_exec(monkeypatch, proc: _FakeProc, captured: dict) -> None:
    async def _fake_exec(*args, **kwargs):
        captured["args"] = list(args)
        return proc

    monkeypatch.setattr("llmkit.claude_cli.asyncio.create_subprocess_exec", _fake_exec)


def test_provider_is_cloud_plane():
    p = ClaudeCliProvider()
    assert p.plane is Plane.CLOUD and p.name == "claude-cli"


def test_split_single_and_multi_turn():
    p = ClaudeCliProvider()
    system, user = p._split(LLMRequest(messages=[
        LLMMessage(role="system", content="SYS"),
        LLMMessage(role="user", content="hello"),
    ]))
    assert system == "SYS" and user == "hello"

    _, user2 = p._split(LLMRequest(messages=[
        LLMMessage(role="user", content="q1"),
        LLMMessage(role="assistant", content="a1"),
        LLMMessage(role="user", content="q2"),
    ]))
    assert "User: q1" in user2 and "Assistant: a1" in user2 and "User: q2" in user2


def test_args_include_tool_free_json_and_model_and_system():
    p = ClaudeCliProvider(binary="claude", model="claude-sonnet-4-6")
    args = p._args(LLMRequest(messages=[LLMMessage(role="user", content="x")]), system="SYS")
    assert args[0] == "claude"
    assert "-p" in args and "--output-format" in args and "json" in args
    # tool-free: --allowed-tools followed by an empty string
    i = args.index("--allowed-tools")
    assert args[i + 1] == ""
    assert "--system-prompt" in args and "SYS" in args
    assert "--model" in args and "claude-sonnet-4-6" in args


@pytest.mark.asyncio
async def test_generate_parses_result_and_picks_answer_model(monkeypatch):
    payload = {
        "is_error": False,
        "result": "  The answer is 42.  ",
        "usage": {"input_tokens": 10, "output_tokens": 20},
        # haiku does background work; sonnet produced the answer (more output tokens)
        "modelUsage": {
            "claude-haiku-4-5": {"outputTokens": 3},
            "claude-sonnet-4-6": {"outputTokens": 20},
        },
    }
    proc = _FakeProc(stdout=json.dumps(payload).encode())
    captured: dict = {}
    _patch_exec(monkeypatch, proc, captured)

    p = ClaudeCliProvider()
    resp = await p.generate(LLMRequest(messages=[
        LLMMessage(role="system", content="SYS"),
        LLMMessage(role="user", content="what is the answer?"),
    ]))
    assert resp.text == "The answer is 42."                    # stripped
    assert resp.model == "claude-sonnet-4-6"                    # highest-output model wins
    assert resp.plane is Plane.CLOUD and resp.provider == "claude-cli"
    assert resp.usage == {"prompt_tokens": 10, "completion_tokens": 20}
    # the user prompt was fed on stdin, not as an argv (avoids escaping/length limits)
    assert proc.stdin_seen == b"what is the answer?"


@pytest.mark.asyncio
async def test_generate_raises_on_nonzero_exit(monkeypatch):
    proc = _FakeProc(stderr=b"boom", returncode=1)
    _patch_exec(monkeypatch, proc, {})
    with pytest.raises(ClaudeCliError):
        await ClaudeCliProvider().generate(
            LLMRequest(messages=[LLMMessage(role="user", content="x")])
        )


@pytest.mark.asyncio
async def test_generate_raises_on_error_result(monkeypatch):
    proc = _FakeProc(stdout=json.dumps({"is_error": True, "result": "rate limited"}).encode())
    _patch_exec(monkeypatch, proc, {})
    with pytest.raises(ClaudeCliError):
        await ClaudeCliProvider().generate(
            LLMRequest(messages=[LLMMessage(role="user", content="x")])
        )


@pytest.mark.asyncio
async def test_generate_raises_on_non_json(monkeypatch):
    proc = _FakeProc(stdout=b"not json at all")
    _patch_exec(monkeypatch, proc, {})
    with pytest.raises(ClaudeCliError):
        await ClaudeCliProvider().generate(
            LLMRequest(messages=[LLMMessage(role="user", content="x")])
        )
