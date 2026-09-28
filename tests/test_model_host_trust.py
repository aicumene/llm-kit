# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""Where client text may and may not be sent.

The ban on crossing the boundary in router.py looks at ``provider.plane``, and that is a CLASS
attribute: ``OllamaProvider.plane = Plane.TRUSTED`` does not depend on the address. So until
09.09.2026 "trusted plane" meant "we used an Ollama client", not "the bytes stayed inside the
perimeter"; the containers' way to the internet was open all the while (checked: example.com
answers 200). These checks are what ties trust to the address.

Addresses are from the documentation ranges: 192.0.2.0/24 and 198.51.100.0/24 play the trusted
networks, 203.0.113.0/24 the internet.
"""
from __future__ import annotations

import pytest

from llmkit.config import ModelConfig
from llmkit.trust import DEFAULT_TRUSTED_CIDRS, UntrustedModelHost, check_model_url

LAN = "192.0.2.0/24,198.51.100.0/24"


def test_an_address_inside_a_trusted_network_passes():
    assert check_model_url("http://192.0.2.10:11435", cidrs=LAN) == "http://192.0.2.10:11435"
    # The same host on another port is ours too: the port tells chat and embedder apart on one
    # machine.
    assert check_model_url("http://192.0.2.10:11434", cidrs=LAN)


def test_a_machine_moving_inside_the_perimeter_needs_no_deploy():
    # A GPU host showed up in two different subnets within a month — both inside the allowed
    # ranges. That is why a range was chosen instead of a list of exact addresses.
    assert check_model_url("http://192.0.2.20:11434", cidrs=LAN)
    assert check_model_url("http://198.51.100.20:11434", cidrs=LAN)


def test_a_public_address_is_refused():
    with pytest.raises(UntrustedModelHost):
        check_model_url("http://203.0.113.8:11434", cidrs=LAN)


def test_a_domain_name_is_refused_even_if_it_points_inside():
    # DNS answers one thing today and another tomorrow; there is nothing to check a name with,
    # so it is not let in.
    with pytest.raises(UntrustedModelHost):
        check_model_url("https://models.example.com", cidrs=LAN)


def test_a_docker_service_name_is_allowed():
    # A short name without dots resolves only inside the docker network and leads nowhere
    # outside; the configuration's default is exactly that (http://ollama:11434).
    assert check_model_url("http://ollama:11434", cidrs=LAN)


def test_a_foreign_scheme_is_refused():
    with pytest.raises(UntrustedModelHost):
        check_model_url("file:///etc/passwd", cidrs=LAN)


def test_an_empty_string_is_legal():
    # It has a meaning of its own: "not set, take the chat address". It is not an address and
    # there is nothing to check.
    assert check_model_url("", cidrs=LAN) == ""


def test_an_empty_network_list_allows_nothing():
    # A negative control on the rule itself: an empty list means "nothing to allow", not
    # "everything is allowed". The opposite default would make the check an ornament.
    with pytest.raises(UntrustedModelHost):
        check_model_url("http://192.0.2.10:11435", cidrs="")


def test_the_default_cuts_off_the_internet_but_lets_private_networks_through():
    # A deployment without explicit configuration must still be unable to send a document out.
    assert check_model_url("http://172.16.5.4:11435", cidrs=DEFAULT_TRUSTED_CIDRS)
    assert check_model_url("http://10.1.2.3:11434", cidrs=DEFAULT_TRUSTED_CIDRS)
    assert check_model_url("http://127.0.0.1:11434", cidrs=DEFAULT_TRUSTED_CIDRS)
    with pytest.raises(UntrustedModelHost):
        check_model_url("http://203.0.113.34:11434", cidrs=DEFAULT_TRUSTED_CIDRS)


def test_the_configuration_hands_out_only_checked_addresses():
    s = ModelConfig(ollama_base_url="http://192.0.2.10:11434",
                    embed_base_url="http://203.0.113.8:11435",
                    trusted_model_cidrs=LAN)
    assert s.trusted_ollama_base_url == "http://192.0.2.10:11434"
    with pytest.raises(UntrustedModelHost):
        _ = s.trusted_embed_base_url


def test_per_model_overrides_are_checked_on_a_par():
    # The same client material goes to chat; an override by model name is the same kind of
    # destination, not a detail of configuration.
    s = ModelConfig(ollama_model_hosts="small:latest=http://203.0.113.8:11434", trusted_model_cidrs=LAN)
    with pytest.raises(UntrustedModelHost):
        _ = s.trusted_model_host_map


# ── THE FACTORY, NOT ONLY THE PURE FUNCTION ──────────────────────────────────────────────────
#
# The mutation "the factory takes the raw address again" SURVIVED the first edition of these
# checks: they checked the function and the configuration, but not the path the data actually
# leaves by. The place to check is where the provider is built — there and only there does an
# address become a destination.

def test_the_factory_refuses_to_build_an_embedder_on_an_untrusted_address():
    from llmkit.factory import _build_trusted
    s = ModelConfig(ollama_base_url="http://192.0.2.10:11434",
                    embed_base_url="http://203.0.113.8:11435",
                    trusted_model_cidrs=LAN)
    with pytest.raises(UntrustedModelHost):
        _build_trusted(s)


def test_the_factory_refuses_to_build_chat_on_an_untrusted_address():
    from llmkit.factory import _build_trusted
    s = ModelConfig(ollama_base_url="https://models.example.com",
                    trusted_model_cidrs=LAN)
    with pytest.raises(UntrustedModelHost):
        _build_trusted(s)


def test_the_factory_builds_on_trusted_addresses():
    # The positive control: without it the previous two would be green on any factory failure.
    from llmkit.factory import _build_trusted
    s = ModelConfig(ollama_base_url="http://192.0.2.10:11434",
                    embed_base_url="http://192.0.2.10:11435",
                    trusted_model_cidrs=LAN)
    assert _build_trusted(s) is not None
