# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""The configuration the factory, the host resolver and the dispatcher read.

Explicit and dependency-free. The field names are the environment names of the deployment the
toolkit was extracted from (``OLLAMA_BASE_URL``, ``EMBED_BASE_URL``, …), so a settings object that
already has them — a pydantic-settings class, for example — can be passed wherever a
:class:`ModelConfig` is expected: every reader uses plain attribute access.

Addresses are read ONLY through the checked properties (``trusted_*``). ``Plane.TRUSTED`` is a
class attribute of a provider and knows nothing about the address, so without the check a
"trusted plane" would mean "we used an Ollama client", not "the bytes stayed inside the
perimeter". They are properties rather than a validator on construction on purpose: a bad address
must not crash a process at start-up — it must close the path for DATA and leave the process
alive, so the operator has somewhere to read the reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llmkit.hosts import _parse_map
from llmkit.trust import DEFAULT_TRUSTED_CIDRS, check_model_url


@dataclass(frozen=True)
class ModelConfig:
    #: The main trusted host: chat, and embeddings unless ``embed_base_url`` is set. "/v1" at the
    #: end of the path means an OpenAI-compatible server (SGLang, vLLM); anything else is Ollama.
    ollama_base_url: str = "http://ollama:11434"
    ollama_chat_model: str = "qwen3.6:27b"
    #: The model of deterministic work in :mod:`llmkit.dispatch` (extract, classify, reason, draft).
    ollama_reasoning_model: str = "qwen3.6:27b"
    ollama_embed_model: str = "snowflake-arctic-embed2"
    #: Per-request timeout, seconds. A 27B model at ~10 tok/s asked for thousands of JSON tokens
    #: runs well past five minutes.
    ollama_timeout: float = 900.0
    #: Sent with every Ollama request, so resident runners recycle on a sane cadence instead of
    #: staying pinned forever.
    ollama_keep_alive: str = "30m"
    #: The prefix an asymmetric embedder expects on a QUERY (never on a document). Empty = the
    #: model is symmetric. It belongs to the MODEL, which is why it lives next to its name.
    ollama_embed_query_prefix: str = "query: "
    #: A separate embeddings host. Empty = the main host.
    embed_base_url: str = ""
    #: Models served by another host: "model=url,model2=url2". Empty = all on the main host.
    ollama_model_hosts: str = ""
    #: THE PERIMETER for model addresses: a numeric IP inside these networks, or a dot-less service
    #: name. Set at deploy time; an administrator's override cannot widen it.
    trusted_model_cidrs: str = DEFAULT_TRUSTED_CIDRS
    #: "ollama" | "vllm". "vllm" serves chat from ``vllm_base_url`` (continuous batching) and
    #: keeps embeddings on the trusted embedder host.
    chat_backend: str = "ollama"
    vllm_base_url: str = "http://vllm:8000/v1"
    vllm_chat_model: str = "qwen3.6:27b"
    vllm_timeout: float = 900.0
    #: Air-gap: register NO cloud provider, so the router physically cannot route off-premise.
    airgap: bool = False
    #: Cloud plane: a real Anthropic key wins; otherwise the on-machine ``claude`` CLI if enabled.
    anthropic_api_key: str = field(default="", repr=False)
    anthropic_model: str = "claude-sonnet-4-6"
    claude_cli_enabled: bool = False
    claude_cli_path: str = "claude"
    claude_cli_model: str = ""
    claude_cli_timeout: float = 300.0

    @property
    def trusted_ollama_base_url(self) -> str:
        return check_model_url(self.ollama_base_url, cidrs=self.trusted_model_cidrs)

    @property
    def trusted_embed_base_url(self) -> str:
        return check_model_url(self.embed_base_url, cidrs=self.trusted_model_cidrs)

    @property
    def trusted_model_host_map(self) -> dict[str, str]:
        """``model=url,model=url`` — every address is checked on its own.

        A chat request carries the same client material as an embedding (``LLMRequest.sensitivity``
        defaults to CONFIDENTIAL), so per-model overrides are checked on a par with the main
        addresses, not treated as "a detail of configuration".
        """
        return _parse_map(self.ollama_model_hosts, self.trusted_model_cidrs)


__all__ = ["ModelConfig"]
