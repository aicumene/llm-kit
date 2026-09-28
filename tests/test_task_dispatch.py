# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""The task dispatcher: a local-first policy, memoization that never runs the same graph twice,
privacy intact.

THE MAIN THING is a cache-degeneracy detector from both ends at once: (1) a hit REALLY happens (the
same extract → the router is called once), and (2) the cache is ISOLATED (another tenant/scope,
another model, a creative task — computed again). Otherwise "always a miss" and "everything stuck
into one answer" both look green.

NO DATABASE AND NO GPU: the router, the cache and the configuration are doubles; what is checked is
the dispatcher's logic, not a model.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from llmkit import dispatch
from llmkit.base import LLMMessage, LLMResponse, Plane
from llmkit.classification import Sensitivity
from llmkit.config import ModelConfig


class _FakeRouter:
    def __init__(self):
        self.calls = 0
        self.last = None

    async def generate(self, request, *, prefer_plane=None):
        self.calls += 1
        self.last = request
        return LLMResponse(
            text=f"out{self.calls}", model=request.model or "?", provider="fake", plane=Plane.TRUSTED
        )


class _FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}

    async def get(self, k):
        return self.store.get(k)

    async def set(self, k, v, ex=None):
        self.store[k] = v


S = ModelConfig(ollama_reasoning_model="qwen3.6:27b", ollama_chat_model="qwen3.6-27b-mtp")
MSG = [LLMMessage(role="user", content="list the key facts in this text")]


def test_policy_is_local_first_and_differentiated():
    reason = dispatch.plan_for(dispatch.REASON, Sensitivity.CONFIDENTIAL, S)
    draft = dispatch.plan_for(dispatch.DRAFT, Sensitivity.CONFIDENTIAL, S)
    chat = dispatch.plan_for(dispatch.CHAT, Sensitivity.CONFIDENTIAL, S)
    # local-first: everything on trusted, no cloud
    for p in (reason, draft, chat):
        assert p.allow_cloud is False
        assert p.cost == "local-free"
    # measured: reasoning on qwen3.6 (not ornith)
    assert reason.model == "qwen3.6:27b"
    assert reason.memoizable is True and reason.temperature == 0.0
    # differentiated: creative kinds are not memoized, chat has a model of its own
    assert draft.memoizable is False
    assert chat.model == "qwen3.6-27b-mtp"


def test_any_object_with_the_model_fields_works_as_settings():
    """A settings object that already has the fields (e.g. the deployment's own) can be passed
    as is; one without them gets the defaults."""
    duck = SimpleNamespace(ollama_reasoning_model="r-model", ollama_chat_model="c-model")
    assert dispatch.plan_for(dispatch.REASON, Sensitivity.CONFIDENTIAL, duck).model == "r-model"
    assert dispatch.plan_for(dispatch.CHAT, Sensitivity.CONFIDENTIAL, duck).model == "c-model"
    assert dispatch.plan_for(dispatch.REASON, Sensitivity.CONFIDENTIAL, object()).model == "qwen3.6:27b"


@pytest.mark.asyncio
async def test_memo_saves_the_repeat_call():
    r, redis = _FakeRouter(), _FakeRedis()
    a = await dispatch.run(r, dispatch.EXTRACT, MSG, settings=S, scope="tenant-a", redis=redis)
    b = await dispatch.run(r, dispatch.EXTRACT, MSG, settings=S, scope="tenant-a", redis=redis)
    assert r.calls == 1, "the second identical extract must come from the memo"
    assert a.text == b.text
    assert b.provider == "memo" and b.usage.get("memo_hit") == 1


@pytest.mark.asyncio
async def test_nonmemoizable_kind_runs_each_time():
    r, redis = _FakeRouter(), _FakeRedis()
    await dispatch.run(r, dispatch.DRAFT, MSG, settings=S, scope="tenant-a", redis=redis)
    await dispatch.run(r, dispatch.DRAFT, MSG, settings=S, scope="tenant-a", redis=redis)
    assert r.calls == 2, "a creative task (temp>0) is not memoized"


@pytest.mark.asyncio
async def test_scope_isolates_tenants():
    r, redis = _FakeRouter(), _FakeRedis()
    await dispatch.run(r, dispatch.EXTRACT, MSG, settings=S, scope="tenant-a", redis=redis)
    await dispatch.run(r, dispatch.EXTRACT, MSG, settings=S, scope="tenant-b", redis=redis)
    assert r.calls == 2, "the same input, another tenant → its own entry, no cross-tenant answer"


@pytest.mark.asyncio
async def test_confidential_without_scope_is_not_memoized():
    r, redis = _FakeRouter(), _FakeRedis()
    await dispatch.run(r, dispatch.EXTRACT, MSG, settings=S, sensitivity=Sensitivity.CONFIDENTIAL, redis=redis)
    await dispatch.run(r, dispatch.EXTRACT, MSG, settings=S, sensitivity=Sensitivity.CONFIDENTIAL, redis=redis)
    assert r.calls == 2, "confidential data without a scope is NOT memoized (cross-tenant protection)"


@pytest.mark.asyncio
async def test_model_change_invalidates_memo():
    r, redis = _FakeRouter(), _FakeRedis()
    await dispatch.run(r, dispatch.EXTRACT, MSG, settings=S, scope="tenant-a", redis=redis)

    s2 = ModelConfig(ollama_reasoning_model="qwen3.7:27b",   # the model changed
                     ollama_chat_model="qwen3.6-27b-mtp")

    await dispatch.run(r, dispatch.EXTRACT, MSG, settings=s2, scope="tenant-a", redis=redis)
    assert r.calls == 2, "the model version is in the key → a model change resets the memo"


def test_confidential_plan_never_allows_cloud():
    # even if the kind is later marked cloud-eligible, confidential data is forced to False.
    p = dispatch.plan_for(dispatch.CHAT, Sensitivity.CONFIDENTIAL, S)
    assert p.allow_cloud is False


@pytest.mark.asyncio
async def test_redis_down_still_answers():
    class _Dead:
        async def get(self, k):
            raise RuntimeError("redis down")

        async def set(self, k, v, ex=None):
            raise RuntimeError("redis down")

    r = _FakeRouter()
    resp = await dispatch.run(r, dispatch.EXTRACT, MSG, settings=S, scope="tenant-a", redis=_Dead())
    assert r.calls == 1 and resp.text == "out1", "a cache failure does not bring the task down"
