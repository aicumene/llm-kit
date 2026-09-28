# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""Диспетчер задач: одна политика «вид задачи → исполнитель» + МЕМОИЗАЦИЯ, поверх LLMRouter.

ЗАЧЕМ. Выбор модели по видам задач был РАЗБРОСАН: шесть кол-сайтов хардкодили
``model=settings.ollama_reasoning_model``. Одна политика в одном месте: (1) видно и меняется
централизованно; (2) стоимостно-осознанно — локально бесплатно, потому предпочитаем; в облако
только по виду задачи И по замеру; (3) сюда равноправно встают НЕ-Claude исполнители (локальные
модели на картах, другие GPU-узлы, воркеры) — план просто называет модель+плоскость, а роутер
находит хост через карту маршрутов по моделям (``ollama_model_hosts``).

МЕМОИЗАЦИЯ (не гонять одни графы дважды). Диспетчер — единственная точка исполнения задачи, потому
кеш тут: детерминированная задача (temp 0: extract/classify/reason) с теми же входами и той же
моделью не считается второй раз (ноль токенов). Ключ = sha256(вид ∥ модель ∥ temp ∥ scope ∥
входы). Инвалидация версией-в-ключе (сменилась модель → другой ключ), TTL как сборка мусора.

ЧЕГО ЭТО НЕ ДЕЛАЕТ. (1) Не обходит приватный инвариант: план лишь ПРЕДЛАГАЕТ модель/плоскость, а
жёсткие ворота (sensitivity ↛ cloud) держит LLMRouter — единственное место, где это можно;
диспетчер НАД роутером, не вместо. (2) Не мемоизирует творческие задачи (temp>0: draft/chat) — их
вывод не обязан повторяться. (3) КОНФИДЕНЦИАЛЬНОЕ БЕЗ scope НЕ мемоизирует — иначе результат по
данным одного клиента мог бы прийти другому; scope (тенант/проект) в ключе изолирует. Публичное
(справочные данные) можно и без scope.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from llmkit.base import LLMMessage, LLMRequest, LLMResponse, Plane
from llmkit.classification import Sensitivity, may_cross_to_cloud

# Виды задач.
EXTRACT = "extract"
CLASSIFY = "classify"
REASON = "reason"
DRAFT = "draft"
CHAT = "chat"

_TTL_SECONDS = 3600


@dataclass(frozen=True)
class Plan:
    model: str
    plane: Plane | None  # предпочитаемая плоскость (None = дефолт роутера, обычно trusted)
    allow_cloud: bool
    temperature: float
    memoizable: bool
    cost: str  # "local-free" | "cloud-paid"


def plan_for(task_kind: str, sensitivity: Sensitivity, settings: Any) -> Plan:
    """Политика вид→исполнитель. По умолчанию ЛОКАЛЬНО (trusted, бесплатно).

    Модели — из настроек (замер 20.09: чтение длинного документа — qwen3.6 4/4, ornith 0/4 →
    рассуждение и чтение на qwen3.6, не на ornith). Эскалация в облако — только там, где вид её
    разрешает И секретность пускает; по умолчанию нигде (local-first).

    ``settings`` — :class:`llmkit.config.ModelConfig` или любой объект с полями
    ``ollama_reasoning_model`` / ``ollama_chat_model``; без них — умолчания ниже.
    """
    local = getattr(settings, "ollama_reasoning_model", "qwen3.6:27b")
    chat = getattr(settings, "ollama_chat_model", local)
    table = {
        EXTRACT: Plan(local, Plane.TRUSTED, False, 0.0, True, "local-free"),
        CLASSIFY: Plan(local, Plane.TRUSTED, False, 0.0, True, "local-free"),
        REASON: Plan(local, Plane.TRUSTED, False, 0.0, True, "local-free"),
        DRAFT: Plan(local, Plane.TRUSTED, False, 0.2, False, "local-free"),
        CHAT: Plan(chat, None, False, 0.2, False, "local-free"),
    }
    plan = table.get(task_kind) or Plan(local, Plane.TRUSTED, False, 0.2, False, "local-free")
    # Приватность (двойная защита к роутеру): облако только если вид его хочет И данные пускают.
    if plan.allow_cloud and not may_cross_to_cloud(sensitivity):
        plan = Plan(plan.model, plan.plane, False, plan.temperature, plan.memoizable, plan.cost)
    return plan


def fingerprint(
    task_kind: str, model: str, temperature: float, scope: str, messages: Sequence[LLMMessage]
) -> str:
    body = "\x00".join(
        [
            task_kind,
            model,
            f"{temperature:.3f}",
            scope or "",
            "\x01".join(f"{m.role}:{m.content}" for m in messages),
        ]
    )
    return "task:v1:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


async def _memo_get(redis: Any, key: str) -> str | None:
    if redis is None or not key:
        return None
    try:
        raw = await redis.get(key)
        if not raw:
            return None
        return raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
    except Exception:
        return None


async def _memo_put(redis: Any, key: str, text: str) -> None:
    if redis is None or not key:
        return
    try:
        await redis.set(key, text, ex=_TTL_SECONDS)
    except Exception:
        pass


async def run(
    router: Any,
    task_kind: str,
    messages: Sequence[LLMMessage],
    *,
    settings: Any,
    sensitivity: Sensitivity = Sensitivity.CONFIDENTIAL,
    scope: str = "",
    redis: Any = None,
    **req_kw: Any,
) -> LLMResponse:
    """Выполнить задачу вида ``task_kind``: план → (мемо-попадание | маршрут через роутер) → мемо-запись.

    ``scope`` — тенант/проект; обязателен, чтобы мемоизировать конфиденциальное (иначе не мемоизируем).
    ``redis`` — клиент кеша с ``get(key)`` и ``set(key, value, ex=секунды)`` (например, асинхронный
    Redis); None ⇒ без мемоизации. Приватный инвариант держит роутер: сюда он не дублируется,
    только предлагается план.
    """
    plan = plan_for(task_kind, sensitivity, settings)

    memoize = plan.memoizable and redis is not None
    # Конфиденциальное без scope не мемоизируем — защита от кросс-тенантной выдачи.
    if memoize and not may_cross_to_cloud(sensitivity) and not scope:
        memoize = False

    key = fingerprint(task_kind, plan.model, plan.temperature, scope, messages) if memoize else ""
    if key:
        hit = await _memo_get(redis, key)
        if hit is not None:
            return LLMResponse(
                text=hit, model=plan.model, provider="memo", plane=Plane.TRUSTED,
                usage={"memo_hit": 1},
            )

    request = LLMRequest(
        messages=list(messages),
        sensitivity=sensitivity,
        model=plan.model,
        temperature=plan.temperature,
        allow_cloud=plan.allow_cloud,
        **req_kw,
    )
    resp = await router.generate(request, prefer_plane=plan.plane)
    if key:
        await _memo_put(redis, key, resp.text)
    return resp
