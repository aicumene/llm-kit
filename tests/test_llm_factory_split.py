# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""Embeddings can be moved to a host of their own without touching chat.

Why: on 01.09.2026 one host served both. A computation loop filled both of its slots with
generations that outlived their own client timeout, and EVERY embedding waited behind them for
hours. Chat is heavy, rare and tolerant of latency; embeddings are light, constant, and block the
loading of a corpus. The two shared only a protocol, not a nature.

The trap this test was written for: the ``embed_base_url`` field had existed for a long time but
was read only in the ``chat_backend="vllm"`` branch. The working configuration took the other
branch, and the environment variable silently did nothing — a setting you can set and never
notice that it did not take effect.

The configuration is the real :class:`ModelConfig`, so the addresses go through the same checked
properties as in production — a stand-in that skipped the check would let these tests pass past
the very check they claim to cover.
"""
from __future__ import annotations

from llmkit.config import ModelConfig
from llmkit.factory import _build_trusted
from llmkit.ollama import OllamaProvider
from llmkit.vllm import SplitTrustedProvider

CHAT = "http://192.0.2.10:11434"
EMBED = "http://192.0.2.20:11434"


def _cfg(**kw) -> ModelConfig:
    base = dict(
        ollama_base_url=CHAT,
        embed_base_url="",
        #: Generation routing by model (see ModelRoutedProvider). Empty = as before.
        ollama_model_hosts="",
        #: Query prefix of an asymmetric embedder. Empty = the model is symmetric.
        ollama_embed_query_prefix="",
        ollama_chat_model="qwen3.6:27b",
        ollama_embed_model="nomic-embed-text",
        ollama_timeout=120.0,
        ollama_keep_alive="5m",
        trusted_model_cidrs="192.0.2.0/24",
    )
    base.update(kw)
    return ModelConfig(**base)


def test_without_a_split_the_shape_is_unchanged():
    """A deployment that did not ask for the split must not quietly get a second backend."""
    p = _build_trusted(_cfg())
    assert isinstance(p, OllamaProvider)


def test_the_same_url_is_not_a_split():
    """Identical addresses are one host, not two. A composite here would be false complexity."""
    p = _build_trusted(_cfg(embed_base_url=CHAT))
    assert isinstance(p, OllamaProvider)


def test_a_different_embed_host_splits_chat_from_embeddings():
    p = _build_trusted(_cfg(embed_base_url=EMBED))
    assert isinstance(p, SplitTrustedProvider)
    assert p._chat._base_url == CHAT, "chat must stay where it was"
    assert p._embed._base_url == EMBED, "embeddings must move — that is the whole point"


def test_the_split_keeps_the_embedding_model():
    """Different hosts — the same model. Otherwise new vectors land in another space and search
    breaks silently: the divergence shows only on a gold set, and not right away."""
    p = _build_trusted(_cfg(embed_base_url=EMBED))
    assert p._embed._embed_model == "nomic-embed-text"
