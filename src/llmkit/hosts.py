# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""Какие адреса моделей ДЕЙСТВУЮТ прямо сейчас — и откуда они взялись.

Порядок один и тот же для всех: переопределение из реестра, иначе окружение (конфигурация), и в
обоих случаях через :func:`llmkit.trust.check_model_url`. Значение из реестра НЕ доверенней
значения из окружения — граница периметра задаётся диапазонами на деплое и здесь не
редактируется, иначе администратор расширял бы периметр сам себе, и проверка стала бы украшением.

ОТКУДА ВЗЯЛОСЬ — часть ответа, а не отладочная мелочь. Месяц API и воркер смотрели в разные
адреса, и ни одна панель этого не показывала: панель печатала адрес процесса, который её
обслуживает, то есть всегда API. Поэтому источник каждого значения возвращается наружу.

ГДЕ ЛЕЖИТ ПЕРЕОПРЕДЕЛЕНИЕ — решает развёртывание: :class:`HostRegistry` — договор (строка в
базе, файл, служба конфигурации), :class:`InMemoryHostRegistry` — реализация в памяти процесса.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol

from llmkit.trust import check_model_url

if TYPE_CHECKING:
    from llmkit.config import ModelConfig

ENV = "окружение"
DB = "настройка администратора"

#: Поля, которые переопределяются: те же имена, что в конфигурации.
FIELDS = ("ollama_base_url", "embed_base_url", "ollama_model_hosts")


@dataclass(frozen=True)
class ResolvedHosts:
    chat: str
    embed: str
    model_hosts: dict[str, str]
    source: dict[str, str] = field(default_factory=dict)
    updated_at: datetime | None = None
    updated_by: object = None


@dataclass(frozen=True)
class HostOverrides:
    """Переопределение администратора. Пустая строка — «не переопределено», берётся окружение."""

    ollama_base_url: str = ""
    embed_base_url: str = ""
    #: "model=url,model2=url2".
    ollama_model_hosts: str = ""
    updated_at: datetime | None = None
    updated_by: object = None


class HostRegistry(Protocol):
    """Where the administrator's override lives.

    ``load`` returns the override — a :class:`HostOverrides` or any object with the same
    attributes (a database row will do) — or ``None`` when nothing is stored. ``stamp`` is the
    moment of the last change: a cheap read for a watcher that decides whether to rebuild the
    router, so every write MUST move it (a write that changes the addresses but not the stamp
    is saved and never applied).
    """

    async def load(self) -> Any: ...

    async def stamp(self) -> datetime | None: ...


class InMemoryHostRegistry:
    """The override kept in process memory — tests, scripts, single-process deployments."""

    def __init__(self, overrides: HostOverrides | None = None) -> None:
        self._row = overrides

    async def load(self) -> HostOverrides | None:
        return self._row

    async def stamp(self) -> datetime | None:
        return self._row.updated_at if self._row is not None else None

    def set(self, *, updated_by: object = None, **fields: str) -> HostOverrides:
        """Store an override of some of :data:`FIELDS`; the others keep their stored values.

        The stamp always moves forward, even for two writes within one clock tick.
        """
        unknown = sorted(set(fields) - set(FIELDS))
        if unknown:
            raise TypeError(f"unknown host override fields: {unknown}")
        prev = self._row or HostOverrides()
        now = datetime.now(UTC)
        if prev.updated_at is not None and now <= prev.updated_at:
            now = prev.updated_at + timedelta(microseconds=1)
        self._row = replace(prev, **fields, updated_at=now, updated_by=updated_by)
        return self._row

    def clear(self) -> None:
        """Forget the override: every address comes from the configuration again."""
        self._row = None


def _parse_map(raw: str, cidrs: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        model, url = pair.split("=", 1)
        model, url = model.strip(), url.strip()
        if model and url:
            out[model] = check_model_url(url, cidrs=cidrs)
    return out


async def resolve(registry: HostRegistry, settings: ModelConfig) -> ResolvedHosts:
    """Действующие адреса. Поднимает :class:`UntrustedModelHost`, если адрес вне периметра.

    ``settings`` — :class:`llmkit.config.ModelConfig` или любой объект с полями :data:`FIELDS`
    и ``trusted_model_cidrs``.
    """
    row = await registry.load()

    cidrs = settings.trusted_model_cidrs
    src: dict[str, str] = {}

    def pick(field_name: str, env_value: str) -> str:
        override = (getattr(row, field_name, "") or "").strip() if row else ""
        if override:
            src[field_name] = DB
            return override
        src[field_name] = ENV
        return env_value

    chat = check_model_url(pick("ollama_base_url", settings.ollama_base_url), cidrs=cidrs)
    embed = check_model_url(pick("embed_base_url", settings.embed_base_url), cidrs=cidrs)
    hosts = _parse_map(pick("ollama_model_hosts", settings.ollama_model_hosts), cidrs)
    return ResolvedHosts(chat=chat, embed=embed, model_hosts=hosts, source=src,
                         updated_at=getattr(row, "updated_at", None),
                         updated_by=getattr(row, "updated_by", None))


def describe(r: ResolvedHosts) -> str:
    """Одна строка про действующие адреса — С АДРЕСАМИ маршрутов, а не только с именами моделей.

    10.09.2026 один из GPU-хостов переехал на другой адрес. Весть воркера и запись аудита
    перечисляли маршруты по именам моделей, и смена адреса выглядела как «ничего не изменилось»:
    сверка API с воркером была слепа ровно к тому расхождению, ради которого её завели, а в журнале
    «до» совпадало с «после». Сравнивать имена — сравнивать ключи словаря, а расходятся значения.
    """
    routes = ",".join(f"{m}={u}" for m, u in sorted(r.model_hosts.items()))
    return f"chat={r.chat} embed={r.embed} routes={routes}"


def fingerprint(r: ResolvedHosts) -> str:
    """Короткий отпечаток описания — то, что сверяется между процессами."""
    return hashlib.sha256(describe(r).encode()).hexdigest()[:12]


def report(r: ResolvedHosts) -> str:
    """Весть процесса о своих адресах: отпечаток впереди, описание следом.

    Поле вести — 200 знаков; описание с несколькими маршрутами в него не помещается, и
    обрезанная строка сравнивалась бы с необрезанной. Поэтому сверяются ОТПЕЧАТКИ, а описание
    остаётся для человека — обрезка его не портит, она портит только сравнение, которого нет.
    """
    return f"{fingerprint(r)} {describe(r)}"[:200]


def reported_fingerprint(detail: str) -> str:
    """Отпечаток из вести процесса; пустая строка, если весть в другом формате."""
    head = (detail or "").split(" ", 1)[0]
    return head if len(head) == 12 and all(c in "0123456789abcdef" for c in head) else ""


async def stamp(registry: HostRegistry) -> datetime | None:
    """Отметка последнего изменения — дешёвое чтение для наблюдателя, который решает,
    надо ли пересобирать маршрутизатор."""
    return await registry.stamp()
