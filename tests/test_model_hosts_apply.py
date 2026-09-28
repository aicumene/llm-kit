# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""Saved ≠ applied: an override of the addresses must REACH the routers.

On 10.09.2026 a route was moved through the admin panel, the panel showed the new address — and
neither the API nor the worker logged "router rebuilt": the write changed the addresses but not the
``updated_at`` stamp, and the watchers rebuild only when the stamp moves. Both processes kept using
the old address until the next restart — exactly the failure all of this was started for.

Pinned here against :class:`InMemoryHostRegistry`: every write moves the stamp; the holder rebuilds
on it and remembers what it was built on; a refused address leaves the old router in place and says
why; an unreadable registry changes nothing. The product's HTTP routes (the write and read
endpoints, the worker's agreement report, the refusal of a malformed route map) stayed with the
product.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from llmkit.config import ModelConfig
from llmkit.hosts import InMemoryHostRegistry, resolve, stamp
from llmkit.metering import MeteredRouter
from llmkit.refresh import RouterHolder, describe_embed_host, resolved_router
from llmkit.vllm import SplitTrustedProvider

NEW = "http://192.0.2.50:11434"
OLD = "http://192.0.2.51:11434"
CFG = ModelConfig(ollama_base_url="http://192.0.2.10:11434", trusted_model_cidrs="192.0.2.0/24")


async def test_a_write_moves_the_stamp_and_the_holder_rebuilds_on_it():
    """The control is inside: before a write, a holder on the current stamp does NOT rebuild."""
    reg = InMemoryHostRegistry()
    reg.set(ollama_model_hosts=f"small:latest={OLD}")
    holder = RouterHolder(router=None)
    assert await holder.refresh(reg, CFG) is True          # the first build
    before = await stamp(reg)
    assert holder.at == before and holder.hosts.model_hosts == {"small:latest": OLD}
    assert isinstance(holder.router, MeteredRouter)
    assert await holder.refresh(reg, CFG) is False         # same stamp — nothing to do

    reg.set(ollama_model_hosts=f"small:latest={NEW}")
    after = await stamp(reg)
    assert after != before, "a write must move the stamp — both processes rebuild on it"
    assert await holder.refresh(reg, CFG) is True
    assert holder.hosts.model_hosts == {"small:latest": NEW}
    assert holder.error is None


def test_two_writes_within_one_clock_tick_still_move_the_stamp(monkeypatch):
    import llmkit.hosts as hosts_mod

    frozen = datetime(2026, 1, 1, tzinfo=UTC)

    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return frozen

    monkeypatch.setattr(hosts_mod, "datetime", _Frozen)
    reg = InMemoryHostRegistry()
    first = reg.set(ollama_base_url=NEW).updated_at
    second = reg.set(ollama_base_url=OLD).updated_at
    assert second > first


def test_the_registry_refuses_a_field_it_does_not_have():
    with pytest.raises(TypeError):
        InMemoryHostRegistry().set(chat_url=NEW)


async def test_clearing_the_override_is_a_change_too():
    reg = InMemoryHostRegistry()
    reg.set(ollama_base_url=NEW)
    holder = RouterHolder(router=None)
    await holder.refresh(reg, CFG)
    assert holder.hosts.chat == NEW
    reg.clear()
    assert await holder.refresh(reg, CFG) is True
    assert holder.hosts.chat == CFG.ollama_base_url


async def test_a_refused_address_leaves_the_old_router_and_says_why():
    reg = InMemoryHostRegistry()
    holder = RouterHolder(router=None)
    assert await holder.refresh(reg, CFG) is True
    built = holder.router
    reg.set(embed_base_url="http://203.0.113.8:11435")
    assert await holder.refresh(reg, CFG) is False
    assert holder.router is built
    assert holder.error and holder.error.startswith("UntrustedModelHost")


async def test_an_unreadable_registry_changes_nothing():
    class _Down:
        async def load(self):
            raise ConnectionError("registry unreachable")

        async def stamp(self):
            raise ConnectionError("registry unreachable")

    holder = RouterHolder(router="working-router", at="t0")
    assert await holder.refresh(_Down(), CFG) is False
    assert holder.router == "working-router" and holder.at == "t0" and holder.error is None


async def test_a_deployment_builds_with_its_own_builder():
    seen: list[str] = []

    def build(settings, hosts):
        seen.append(hosts.chat)
        return "deployment-router"

    holder = RouterHolder(router=None, build=build)
    assert await holder.refresh(InMemoryHostRegistry(), CFG) is True
    assert holder.router == "deployment-router" and seen == [CFG.ollama_base_url]

    router, hosts = await resolved_router(InMemoryHostRegistry(), CFG, build=build)
    assert router == "deployment-router" and hosts.chat == CFG.ollama_base_url


async def test_a_script_gets_the_router_on_the_effective_addresses():
    """Scripts that built the router from the configuration alone went to the host the
    administrator had just retired."""
    reg = InMemoryHostRegistry()
    reg.set(embed_base_url="http://192.0.2.60:11435")
    router, hosts = await resolved_router(reg, CFG)
    assert hosts.embed == "http://192.0.2.60:11435"
    provider = router.providers[0]
    assert isinstance(provider, SplitTrustedProvider)
    assert provider._embed._base_url == "http://192.0.2.60:11435"

    env_hosts = await resolve(InMemoryHostRegistry(), CFG)
    assert describe_embed_host(hosts).startswith("http://192.0.2.60:11435 (")
    # The label names the source, and the two sources are labelled differently.
    assert describe_embed_host(hosts).split(" (", 1)[1] != describe_embed_host(env_hosts).split(" (", 1)[1]
