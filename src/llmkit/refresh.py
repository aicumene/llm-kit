# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""Пересборка маршрутизатора, когда администратор сменил адрес.

ЗАЧЕМ ОТДЕЛЬНЫЙ МЕХАНИЗМ. ``build_router`` зовётся РОВНО ОДИН РАЗ на процесс — при старте
приложения и при старте воркера, — и провайдер живёт до перезапуска. Значит без наблюдателя запись
в реестре так и осталась бы записью в реестре: «настройка сохранилась» и не применилась. Это ровно
тот класс отказа, ради которого всё и затевалось, только с другого конца.

ДВА ПРОЦЕССА, ОДИН ПОРЯДОК. Наблюдатель одинаков для API и воркера: сравнить отметку времени
реестра с той, на которой собран текущий маршрутизатор, и пересобрать, если отличается. Чтение
дешёвое — одна отметка.

ОТКАЗ ЧТЕНИЯ НЕ МЕНЯЕТ НИЧЕГО. Если реестр недоступен или адрес оказался вне периметра, прежний
маршрутизатор остаётся на месте: работающая конфигурация лучше отсутствующей, а о беде скажет
``RouterHolder.error``. Молча подставлять окружение вместо непрочитанного переопределения нельзя —
это подмена одного другим без ведома того, кто настраивал.

ЧЕМ СОБИРАТЬ — по умолчанию :func:`llmkit.factory.build_router` (ищется в момент сборки);
развёртывание со своей сборкой (например, с обёрткой облачного провайдера) передаёт её в ``build``.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from llmkit.factory import build_router
from llmkit.hosts import DB, HostRegistry, ResolvedHosts, resolve, stamp

logger = logging.getLogger(__name__)

#: ``build(settings, hosts)`` → маршрутизатор.
RouterBuilder = Callable[[Any, ResolvedHosts], Any]


async def resolved_router(registry: HostRegistry, settings: Any, *,
                          build: RouterBuilder | None = None):
    """Маршрутизатор на ДЕЙСТВУЮЩИХ адресах — для скриптов обслуживания.

    ЗАЧЕМ (13.09.2026). Эмбеддер в бою перевели на другой сервер переопределением администратора,
    а окружение по-прежнему называло старый хост — запасной, клинящийся под нагрузкой. API и
    воркер видят переопределение через :class:`RouterHolder`; скрипты же звали
    ``build_router(settings)`` без ``hosts`` и молча ходили в окружение — то есть в запасной хост,
    который администратор как раз и отставил. Здесь тот же порядок, что у
    ``RouterHolder.refresh``: ``resolve`` (реестр, иначе окружение, оба через проверку периметра)
    → ``build(settings, hosts)``. Возвращает пару (маршрутизатор, адреса), чтобы скрипт мог
    НАПЕЧАТАТЬ, куда пойдёт.
    """
    hosts = await resolve(registry, settings)
    return (build or build_router)(settings, hosts), hosts


def describe_embed_host(hosts: ResolvedHosts) -> str:
    """Одна строка для журнала скрипта: адрес эмбеддера и откуда он взят."""
    src = hosts.source.get("embed_base_url", "")
    return f"{hosts.embed} ({'переопределение администратора' if src == DB else 'окружение'})"


class RouterHolder:
    """Маршрутизатор и отметка настроек, на которой он собран."""

    def __init__(self, router, at=None, hosts=None, *, build: RouterBuilder | None = None) -> None:
        self.router = router
        self.at = at
        #: Адреса, на которых собран ТЕКУЩИЙ маршрутизатор. Это и есть то, чем процесс ходит;
        #: запись в реестре — лишь то, что он прочитает при следующей пересборке.
        self.hosts = hosts
        #: Последняя причина, по которой пересборка не удалась; пусто, если удалась.
        self.error: str | None = None
        self._build = build

    async def refresh(self, registry: HostRegistry, settings: Any) -> bool:
        """Пересобрать, если настройки изменились. Возвращает True, если пересобрали."""
        try:
            at = await stamp(registry)
        except Exception as exc:  # noqa: BLE001
            logger.warning("не смогли прочитать отметку настроек адресов: %s", exc)
            return False
        if at == self.at and self.router is not None:
            return False
        try:
            hosts = await resolve(registry, settings)
        except Exception as exc:  # noqa: BLE001
            # Недоверенный или нечитаемый адрес — оставляем прежний маршрутизатор.
            logger.error("адреса моделей не приняты, маршрутизатор не тронут: %s", exc)
            self.error = f"{type(exc).__name__}: {exc}"[:200]
            return False
        self.router = (self._build or build_router)(settings, hosts)
        self.at = at
        self.hosts = hosts
        self.error = None
        logger.info("маршрутизатор пересобран: chat=%s embed=%s routes=%s",
                    hosts.chat, hosts.embed, hosts.model_hosts)
        return True
