# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""LLM provider abstraction shared by the trusted and cloud planes.

Every model call in the system goes through an ``LLMProvider``. Each provider declares the
plane it belongs to (TRUSTED or CLOUD); the :class:`~llmkit.router.LLMRouter` uses that,
together with the request's sensitivity label, to enforce the core privacy invariant.

Dependency-free (stdlib only) so it can be imported by the router and tested in isolation.
Concrete providers (``ollama``, ``cloud``) pull in their own transport libraries.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from enum import Enum

from llmkit.classification import Sensitivity


class Plane(str, Enum):
    """Which trust plane a provider executes in."""

    TRUSTED = "trusted"  # runs on the operator's own hardware (e.g. Ollama on local GPUs)
    CLOUD = "cloud"      # runs off-prem; only ever sees de-identified / public inputs


@dataclass(slots=True)
class LLMMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(slots=True)
class LLMRequest:
    """A single inference request, carrying the sensitivity of its payload.

    ``sensitivity`` is the classification of the *content of the messages*. It is the
    contract the router enforces against: a CLOUD provider may never receive a request
    whose sensitivity exceeds ``max_crossable()``.
    """

    messages: list[LLMMessage]
    sensitivity: Sensitivity = Sensitivity.CONFIDENTIAL  # safe default: assume client data
    model: str | None = None
    temperature: float = 0.2
    max_tokens: int | None = None
    # Reasoning-model control. Hybrid models (qwen3, deepseek-r1, …) emit chain-of-thought
    # into a separate ``thinking`` channel and can exhaust the token budget before producing
    # the visible answer. For structured-extraction tasks (classify, critic, judge) set this
    # to False so the model answers directly. None = leave the model's default.
    think: bool | None = None
    #: JSON-схема для ОГРАНИЧЕННОГО ДЕКОДИРОВАНИЯ. Не «подсказка формата», а гарантия:
    #: бэкенд, который её поддерживает, физически не может выдать ничего другого. Нужна
    #: потому, что дообученные модели теряют следование формату (замер: две линейки доменного
    #: тюна — 0.00 и 0.08 против 1.00 у нетюненных ровесников) — см. llmkit/shape.py. Провайдер,
    #: который схему не умеет, ОБЯЗАН её игнорировать: слой shape всё равно проверит разбор.
    response_format: dict | None = None
    # Caller opt-in: this request *may* be served by the cloud plane if eligible. Defaults
    # to False so nothing leaves the trusted plane unless explicitly allowed.
    allow_cloud: bool = False
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class LLMResponse:
    text: str
    model: str
    provider: str
    plane: Plane
    usage: dict[str, int] = field(default_factory=dict)


#: Роль текста в асимметричном эмбеддере. Две константы, а не bool: bool на месте вызова
#: читается как «embed(texts, True)» и не говорит ничего.
DOCUMENT: str = "document"
QUERY: str = "query"


class LLMProvider(abc.ABC):
    """Base class for all model backends."""

    #: Stable identifier, e.g. "ollama", "anthropic".
    name: str
    #: The trust plane this provider runs in.
    plane: Plane

    @abc.abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Return a full completion."""

    @abc.abstractmethod
    def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Yield completion token chunks as they arrive."""

    @abc.abstractmethod
    async def embed(self, texts: Sequence[str], *, kind: str = DOCUMENT) -> list[list[float]]:
        """Return one embedding vector per input text.

        ``kind`` says what the text IS, and it exists because asymmetric embedders exist.
        A model like ``snowflake-arctic-embed2`` expects a query to arrive prefixed and a document
        not to; measured 2026-09-02 on 376 bilingual (Greek/English) sections, getting that right
        moved MRR from 0.948 to 0.962, and on the harder gold questions Greek recall@3 from 0.391
        to 0.522.

        Two properties of the design matter more than the numbers:

        * **The role travels as an argument, never by mutating the string.** In a hybrid search
          the same string is handed to ``embed`` AND to the lexical (full-text) arm. Prefix it at
          the call site and ``"query"`` becomes a term of the full-text query — with Postgres
          ``or_tsquery`` capped at 24 terms, the FIRST of them — silently corrupting the lexical
          half of every hybrid search. Only the embedding layer may see the prefix.
        * **The default is DOCUMENT, not QUERY.** A query that forgets its role loses a little
          accuracy on that one search. A corpus embedded in the wrong role is wrong in every row
          and nothing raises: provenance records the model name and the dimension, never the role.
        """
