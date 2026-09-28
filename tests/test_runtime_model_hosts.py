# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""Effective model addresses: the administrator's override on top of the configuration.

THE MAIN CLAIM of these checks is not "the override works" but "the override does NOT widen the
perimeter". A value from the registry passes the same range check as a value from the
configuration; otherwise an input field in an admin panel would become a legitimate way to send
client material outside, while ``Plane.TRUSTED`` in the log kept claiming the bytes stayed inside.

Re-expressed from the source's database-backed tests against :class:`InMemoryHostRegistry`; the
tests of the product's admin routes, audit log and panel stayed with the product.
"""
from __future__ import annotations

import pytest

from llmkit.config import ModelConfig
from llmkit.hosts import (DB, ENV, InMemoryHostRegistry, ResolvedHosts, describe, fingerprint,
                          report, reported_fingerprint, resolve)
from llmkit.trust import UntrustedModelHost

LAN = "192.0.2.0/24,198.51.100.0/24"

CFG = ModelConfig(
    ollama_base_url="http://192.0.2.10:11434",
    embed_base_url="http://192.0.2.10:11435",
    ollama_model_hosts="",
    trusted_model_cidrs=LAN,
)


def _override(**fields) -> InMemoryHostRegistry:
    reg = InMemoryHostRegistry()
    reg.set(**fields)
    return reg


async def test_without_an_override_the_configuration_is_used():
    r = await resolve(InMemoryHostRegistry(), CFG)
    assert r.chat == CFG.ollama_base_url
    assert r.embed == CFG.embed_base_url
    assert r.source["embed_base_url"] == ENV
    assert r.updated_at is None


async def test_an_override_wins_and_is_named_by_its_source():
    reg = InMemoryHostRegistry()
    reg.set(embed_base_url="http://198.51.100.20:11435", updated_by="admin-1")
    r = await resolve(reg, CFG)
    assert r.embed == "http://198.51.100.20:11435"
    assert r.source["embed_base_url"] == DB
    # The other fields are untouched — exactly what was set is overridden.
    assert r.chat == CFG.ollama_base_url and r.source["ollama_base_url"] == ENV
    assert r.updated_by == "admin-1" and r.updated_at is not None


async def test_a_machine_moving_inside_the_perimeter_is_accepted():
    # This case is what it was all for: a GPU host moved between two subnets.
    r = await resolve(_override(ollama_base_url="http://198.51.100.20:11434"), CFG)
    assert r.chat == "http://198.51.100.20:11434"


async def test_an_override_outside_the_perimeter_is_REFUSED():
    # What this whole check exists for.
    with pytest.raises(UntrustedModelHost):
        await resolve(_override(embed_base_url="http://203.0.113.8:11435"), CFG)


async def test_a_domain_name_in_an_override_is_refused():
    with pytest.raises(UntrustedModelHost):
        await resolve(_override(embed_base_url="https://models.example.com"), CFG)


async def test_the_per_model_map_is_checked_too():
    with pytest.raises(UntrustedModelHost):
        await resolve(_override(ollama_model_hosts="small:latest=http://203.0.113.8:11434"), CFG)


async def test_an_empty_string_means_not_set_not_an_empty_address():
    # A negative control: an empty override must not replace the configuration with emptiness.
    r = await resolve(_override(embed_base_url=""), CFG)
    assert r.embed == CFG.embed_base_url and r.source["embed_base_url"] == ENV


async def test_a_registry_row_of_another_type_is_read_by_its_fields():
    """A deployment's registry returns its own row type (a database row, say); only the fields
    count."""

    class _Row:
        ollama_base_url = ""
        embed_base_url = "http://198.51.100.30:11435"
        ollama_model_hosts = "small:latest=http://192.0.2.30:11434"
        updated_at = None
        updated_by = None

    class _Registry:
        async def load(self):
            return _Row()

        async def stamp(self):
            return None

    r = await resolve(_Registry(), CFG)
    assert r.embed == "http://198.51.100.30:11435" and r.source["embed_base_url"] == DB
    assert r.model_hosts == {"small:latest": "http://192.0.2.30:11434"}
    assert r.chat == CFG.ollama_base_url and r.source["ollama_base_url"] == ENV


def test_a_route_moving_to_another_address_changes_the_description_and_the_fingerprint():
    """A route move once went into the log as "before" = "after": routes were written by model
    names. A record of where client material goes must name the address."""
    a = ResolvedHosts(chat=CFG.ollama_base_url, embed=CFG.embed_base_url,
                      model_hosts={"small:latest": "http://192.0.2.30:11434"})
    b = ResolvedHosts(chat=CFG.ollama_base_url, embed=CFG.embed_base_url,
                      model_hosts={"small:latest": "http://192.0.2.31:11434"})
    assert "routes=small:latest=http://192.0.2.30:11434" in describe(a)
    assert fingerprint(a) != fingerprint(b)
    assert reported_fingerprint(report(a)) == fingerprint(a)
    assert reported_fingerprint("router not built yet") == ""


def test_a_long_report_is_cut_but_its_fingerprint_survives():
    many = ResolvedHosts(chat=CFG.ollama_base_url, embed=CFG.embed_base_url,
                         model_hosts={f"model-{i}:latest": "http://192.0.2.40:11434" for i in range(20)})
    assert len(report(many)) == 200
    assert reported_fingerprint(report(many)) == fingerprint(many)
