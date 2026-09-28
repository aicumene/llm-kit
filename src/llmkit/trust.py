# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""Куда МОЖНО отправлять клиентский текст.

ПОЧЕМУ ЭТОТ МОДУЛЬ СУЩЕСТВУЕТ. Запрет пересечения границы в :mod:`llmkit.router` смотрит на
``provider.plane``, а ``plane`` — это атрибут КЛАССА (``OllamaProvider.plane = Plane.TRUSTED``),
не зависящий от адреса. Значит «доверенная плоскость» до сих пор означала «мы воспользовались
клиентом Ollama», а не «байты остались внутри периметра». Проверено 09.09.2026: контейнеры API и
воркера свободно достукиваются до ``https://example.com`` (код 200), то есть сеть этого тоже не
ограничивает. Пока адреса лежали в ``.env`` и правились деплоем, это было латентно; как только
их станет менять администратор в рантайме, это стало бы полем ввода, за которым клиентские
материалы уходят по указанному адресу — с пометкой «доверенная плоскость» в журнале.

ЧТО ЗДЕСЬ. Адрес модели допустим, только если он либо (а) числовой IP внутри доверенных
подсетей, либо (б) короткое имя службы без точек — такое имя разрешается лишь внутри сети
docker и наружу не ведёт. Полное доменное имя отвергается всегда: именно оно нужно, чтобы
увести данные за периметр.

ПОЧЕМУ ДИАПАЗОН, А НЕ СПИСОК. Список точных адресов пришлось бы править деплоем при каждой
смене адреса машины — а они плавают (один GPU-хост за месяц побывал в двух разных подсетях).
Диапазон позволяет администратору менять адрес в рантайме и при этом делает выход за периметр
структурно невозможным, а не запрещённым правилом.
"""
from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

#: Частные диапазоны по умолчанию. Умолчание НЕ «разрешено всё»: приложение, развёрнутое без
#: явной настройки, всё равно не должно уметь отправить документ клиента в интернет.
DEFAULT_TRUSTED_CIDRS = "10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,127.0.0.0/8"


class UntrustedModelHost(ValueError):
    """Адрес модели вне доверенных подсетей. Отказ, а не предупреждение."""


def _networks(cidrs: str) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    out = []
    for raw in cidrs.split(","):
        raw = raw.strip()
        if raw:
            out.append(ipaddress.ip_network(raw, strict=False))
    return out


def check_model_url(url: str, *, cidrs: str = DEFAULT_TRUSTED_CIDRS) -> str:
    """Вернуть адрес, если он доверенный; иначе поднять :class:`UntrustedModelHost`.

    Пустая строка проходит: у неё есть отдельный смысл — «не задано, взять адрес чата».
    """
    if not url or not url.strip():
        return url
    url = url.strip()
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise UntrustedModelHost(
            f"адрес модели {url!r}: схема {parts.scheme!r} не разрешена, только http/https")
    host = parts.hostname
    if not host:
        raise UntrustedModelHost(f"адрес модели {url!r}: в нём нет хоста")

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        # Не число. Короткое имя без точек — это имя службы внутри сети docker, наружу оно не
        # разрешается вовсе. Полное доменное имя отвергаем: именно им уводят данные за периметр,
        # и «проверить» его нельзя — DNS может ответить сегодня одно, а завтра другое.
        if "." in host:
            raise UntrustedModelHost(
                f"адрес модели {url!r}: доменное имя {host!r} не разрешено — "
                "укажите адрес внутри доверенных подсетей или имя службы docker"
            ) from None
        return url

    nets = _networks(cidrs)
    if not nets:
        raise UntrustedModelHost(
            "доверенные подсети не заданы (TRUSTED_MODEL_CIDRS пуст) — "
            "при пустом списке разрешать нечего")
    if not any(ip in n for n in nets):
        raise UntrustedModelHost(
            f"адрес модели {url!r}: {ip} вне доверенных подсетей ({cidrs})")
    return url
