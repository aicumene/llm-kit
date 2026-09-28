# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""The LLM router — the single chokepoint that enforces the privacy invariant.

> **Invariant:** a payload whose sensitivity exceeds ``max_crossable()`` (i.e. CONFIDENTIAL
> or RESTRICTED) can never be dispatched to a CLOUD-plane provider. Any attempt raises
> :class:`PrivacyViolation`.

The guard (:meth:`LLMRouter._assert_allowed`) runs before *every* dispatch — generate,
stream, and embed — so even a mis-selected provider cannot leak data. This is the
guarantee everything built on the router rests on, which is why it lives in dependency-free
code with dedicated tests (``tests/test_router_invariant.py``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Sequence

from llmkit.base import DOCUMENT, LLMProvider, LLMRequest, LLMResponse, Plane
from llmkit.classification import Sensitivity, may_cross_to_cloud
from llmkit.tracing import start_span


class PrivacyViolation(RuntimeError):
    """Raised when a dispatch would send too-sensitive data off the trusted plane."""


class LLMRouter:
    def __init__(self, providers: Iterable[LLMProvider] = ()) -> None:
        self._providers: list[LLMProvider] = []
        for provider in providers:
            self.register(provider)

    # ── registration ──────────────────────────────────────────────────────
    def register(self, provider: LLMProvider) -> None:
        self._providers.append(provider)

    @property
    def providers(self) -> tuple[LLMProvider, ...]:
        return tuple(self._providers)

    def _by_plane(self, plane: Plane) -> list[LLMProvider]:
        return [p for p in self._providers if p.plane == plane]

    # ── the invariant ─────────────────────────────────────────────────────
    @staticmethod
    def _assert_allowed(provider: LLMProvider, request: LLMRequest) -> None:
        """Hard gate. Runs before every dispatch; never bypass it."""
        if provider.plane is Plane.CLOUD and not may_cross_to_cloud(request.sensitivity):
            raise PrivacyViolation(
                f"Refusing to send {Sensitivity(request.sensitivity).label} data to "
                f"cloud provider {provider.name!r}: only data at or below "
                f"{Sensitivity.INTERNAL.label} may leave the trusted plane."
            )

    # ── selection ─────────────────────────────────────────────────────────
    def select(
        self,
        request: LLMRequest,
        *,
        prefer_plane: Plane | None = None,
    ) -> LLMProvider:
        """Choose a provider, defaulting to the trusted plane.

        Cloud is chosen only when the caller opted in (``allow_cloud``), the request is
        eligible to cross, and either ``prefer_plane`` is CLOUD or no trusted provider
        exists. If the caller explicitly demands cloud for ineligible data, we raise here
        rather than silently downgrading — the caller's intent was unsafe.
        """
        cloud_ok = request.allow_cloud and may_cross_to_cloud(request.sensitivity)

        if prefer_plane is Plane.CLOUD:
            if not cloud_ok:
                # Explicit unsafe request — surface it, do not silently reroute.
                raise PrivacyViolation(
                    f"Cloud plane requested for {Sensitivity(request.sensitivity).label} "
                    "data that may not cross the gateway."
                )
            chosen = self._first(self._by_plane(Plane.CLOUD), request)
            if chosen is not None:
                return chosen
            raise PrivacyViolation("No cloud provider registered.")

        # Default path: prefer trusted.
        trusted = self._first(self._by_plane(Plane.TRUSTED), request)
        if trusted is not None:
            return trusted
        if cloud_ok:
            cloud = self._first(self._by_plane(Plane.CLOUD), request)
            if cloud is not None:
                return cloud
        raise PrivacyViolation(
            "No eligible provider: no trusted provider available and the request is not "
            "eligible for the cloud plane."
        )

    @staticmethod
    def _first(candidates: list[LLMProvider], request: LLMRequest) -> LLMProvider | None:
        # Future: rank trusted candidates by spare capacity to burst to an external trusted
        # GPU node. For now, registration order.
        return candidates[0] if candidates else None

    @staticmethod
    def _span_attrs(provider: LLMProvider, request: LLMRequest, **extra) -> dict:
        return {
            "llm.plane": provider.plane.value,
            "llm.provider": provider.name,
            "llm.model": request.model or "",
            "llm.sensitivity": Sensitivity(request.sensitivity).label,
            **extra,
        }

    # ── dispatch ──────────────────────────────────────────────────────────
    async def generate(
        self, request: LLMRequest, *, prefer_plane: Plane | None = None
    ) -> LLMResponse:
        provider = self.select(request, prefer_plane=prefer_plane)
        self._assert_allowed(provider, request)
        with start_span("llm.generate", **self._span_attrs(provider, request)):
            return await provider.generate(request)

    async def stream(
        self, request: LLMRequest, *, prefer_plane: Plane | None = None,
        usage: dict | None = None,
    ) -> AsyncIterator[str]:
        """``usage`` — приёмник счётчиков, проброшенный НАСКВОЗЬ до провайдера.

        Ровно та же дыра, что была у :meth:`embed`, и с тем же молчаливым исходом. Обёртка
        учёта передавала приёмник сюда, этот метод его НЕ ПРИНИМАЛ, обёртка ловила ``TypeError``
        и уходила в откат — строка книги писалась с нулём токенов. «Поток ничего не стоил» и
        «мы не смогли посчитать» выглядели одинаково, и весь потоковый расход — самый нагруженный
        путь — числился нулём.

        Плоскость кладётся здесь, потому что провайдера знает роутер, а модель — сам провайдер:
        иначе книга отчитывалась бы захардкоженным ``trusted`` и пустым именем модели.
        """
        provider = self.select(request, prefer_plane=prefer_plane)
        self._assert_allowed(provider, request)
        if usage is not None:
            usage["plane"] = provider.plane.value
        with start_span("llm.stream", **self._span_attrs(provider, request)):
            try:
                stream = provider.stream(request, usage=usage)
            except TypeError:
                # Провайдер, не знающий про приёмник. Счётчик ВЫЗОВОВ останется ненулевым,
                # и глухой путь будет виден в отчёте, а не растворится в нуле.
                stream = provider.stream(request)
            async for chunk in stream:
                yield chunk

    async def embed(
        self,
        texts: Sequence[str],
        *,
        sensitivity: Sensitivity = Sensitivity.CONFIDENTIAL,
        allow_cloud: bool = False,
        prefer_plane: Plane | None = None,
        kind: str = DOCUMENT,
        usage: dict | None = None,
    ) -> list[list[float]]:
        """``usage`` — приёмник счётчиков, проброшенный НАСКВОЗЬ до провайдера.

        Без этого проброса обёртка учёта передаёт приёмник роутеру, роутер его не знает и не
        передаёт дальше, а провайдер отвечает без счётчиков. Отказ при этом молчаливый: строка
        книги пишется, но с нулём токенов, и «эмбеддинг ничего не стоил» неотличимо от
        «мы не смогли посчитать». Именно так это и выглядело при первом сквозном прогоне —
        поймано счётчиком ВЫЗОВОВ, который для того и заведён отдельно от токенов.
        """
        request = LLMRequest(messages=[], sensitivity=sensitivity, allow_cloud=allow_cloud)
        provider = self.select(request, prefer_plane=prefer_plane)
        self._assert_allowed(provider, request)
        with start_span(
            "llm.embed", **self._span_attrs(provider, request, **{"llm.batch": len(texts)})
        ):
            try:
                return await provider.embed(texts, kind=kind, usage=usage)
            except TypeError:
                # Провайдер, не знающий про приёмник. Счётчик вызовов останется ненулевым, и
                # глухой путь будет виден в отчёте, а не растворится в нуле.
                return await provider.embed(texts, kind=kind)
