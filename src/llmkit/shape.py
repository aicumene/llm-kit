# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""Преобразователь формы ответа: вызывающий объявляет, ЧТО принять, а не как спросить.

ЗАЧЕМ ЭТО ЕСТЬ. Замер 04.09.2026 на одной и той же задаче с растущей длиной промпта:

===========================  =========  ======  ======  ======
модель                        919 симв    1377    2140    4163
===========================  =========  ======  ======  ======
``tuned-4b``                      0.00    0.00    0.00    0.00
``tuned-ornith-v31``              0.08    0.08    0.08    0.04
``gemma4:12b``                    1.00    1.00    1.00    1.00
``qwen3.6:27b``                   1.00    1.00    1.00    1.00
===========================  =========  ======  ======  ======

Доля ответов, состоящих ровно из затребованного (одна цифра). ``tuned-4b`` и
``tuned-ornith-v31`` — две линейки доменного дообучения на разных базах и разных размерах. Длина
промпта ни при чём — тюн на Ornith валится уже на самом коротком. Обе линейки сломаны одинаково:
обучающие ответы состоят из объяснительных абзацев, и модель выучила, что отвечать = писать
абзац. Нетюненные ровесники держат формат идеально.

ЧТО ЭТО МЕНЯЕТ. Без этого слоя своя же дообученная модель непригодна везде, где ответ
разбирается кодом: выбор из списка, метка, цифра, JSON. С ним — пригодна без переобучения:
на том же промпте ограниченное декодирование даёт ``{"choice": 5}``, а затравка ответа
ассистента — ``5``.

ЧЕТЫРЕ ПРАВИЛА, ИЗ КОТОРЫХ ЗДЕСЬ ВСЁ СЛЕДУЕТ.

1. **Не угадывать из прозы.** Если ответ не разобран однозначно, возвращается ``None`` с
   причиной. Вытащить первое попавшееся число из абзаца — значит подменить «модель не
   ответила» на «модель ответила неверно», а это худший из двух исходов: в замере отказа
   выяснилось, что модель, которой некуда деться, называет заведомо неверный вариант в 83%
   случаев. Пусть лучше молчит.
2. **Значение вне области — это отказ, а не значение.** Ответ «11» при десяти вариантах не
   обрезается до 10 и не берётся по модулю: он отвергается.
3. **Механизм записывается в результат.** Без этого нельзя увидеть, что модель перестала
   отвечать сама и её тянет затравка — то есть что тюн деградировал.
4. **Таблица способностей выведена из замера и перепроверяема.** Незнакомая модель проходит
   всю лестницу, а не считается «наверное умеет».
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from llmkit.base import LLMMessage, LLMRequest


class Mechanism(str, Enum):
    """Как добиться формы. Порядок объявления — порядок предпочтения по умолчанию."""

    #: Модель отвечает как просили, ничего добавлять не нужно.
    NATIVE = "native"
    #: Реплика ассистента затравлена началом ответа. Дёшево и работает даже там, где
    #: ограниченного декодирования нет; измеренно самый быстрый путь (0.8 с против 3.3 с).
    PREFILL = "prefill"
    #: Ограниченное декодирование по JSON-схеме (``format`` в ollama).
    SCHEMA = "schema"


LADDER: tuple[Mechanism, ...] = (Mechanism.NATIVE, Mechanism.PREFILL, Mechanism.SCHEMA)


@dataclass(frozen=True, slots=True)
class Shape:
    """Что должно прийти обратно.

    ``kind`` определяет разбор и проверку области значений; ``instruction`` — хвост, который
    дописывается к промпту; ``prefill`` — затравка ответа; ``schema`` — JSON-схема.
    """

    kind: str
    instruction: str
    prefill: str
    schema: dict[str, Any]
    #: Домен допустимых значений. Пусто = проверка только по типу.
    domain: tuple[Any, ...] = ()

    def parse(self, raw: str) -> tuple[Any, str]:
        """``(значение, причина)``. Значение ``None`` — не разобрано, причина объясняет."""
        text = (raw or "").strip()
        # ПУСТОЙ блок рассуждения — обёртка сборки ollama, а не ответ модели: при
        # `think: false` эта сборка всё равно печатает `<think></think>` перед ответом.
        # Замер 05.09 на 100 задачах: без снятия обёртки правильно отвечающая модель
        # показывала формат 0.00 вместо 1.00 — то есть починенная модель читалась как
        # сломанная. Снимаем ТОЛЬКО пустой блок: непустой это настоящее рассуждение,
        # и его наличие в ответе, где просили одну цифру, — законное нарушение формата.
        text = re.sub(r"^<think>\s*</think>", "", text, flags=re.S).strip()
        if not text:
            return None, "пустой ответ"
        # Ограниченное декодирование отдаёт объект — разбираем его первым.
        if text.startswith("{"):
            try:
                obj = json.loads(text)
            except ValueError:
                return None, "ответ начинается с { но не разбирается как JSON"
            if self.kind == "json":
                return obj, ""
            if not isinstance(obj, dict) or "value" not in obj:
                return None, f"в объекте нет поля value: {sorted(obj)[:5]}"
            text = str(obj["value"]).strip()
        if self.kind == "json":
            return None, "ожидался объект JSON, пришёл текст"
        # Завершающий знак препинания снимается, и только он: «7.» — это цифра с точкой,
        # а не проза. Всё, что остаётся после снятия, обязано совпасть ЦЕЛИКОМ — иначе
        # «the answer is 7» прошло бы как 7 (правило 1).
        text = text.strip().rstrip(".,;:)\u00a0").strip().strip("\"'")
        if self.kind == "integer":
            m = re.fullmatch(r"-?\d{1,9}", text)
            if not m:
                return None, f"не целое число целиком: {text[:40]!r}"
            val: Any = int(m.group(0))
        else:  # label
            val = text.strip(" .\"'")
        if self.domain and val not in self.domain:
            # Правило 2: вне области — отказ, а не подгонка под область.
            return None, f"значение {val!r} вне допустимого набора"
        return val, ""


def choice(n: int, *, allow_none: bool = True) -> Shape:
    """Выбор одного из ``n`` пронумерованных вариантов. ``0`` = «ни один», если разрешено."""
    lo = 0 if allow_none else 1
    dom = tuple(range(lo, n + 1))
    tail = (f"\nAnswer with the candidate number ({1}-{n})"
            + (", or 0 if none of them fits" if allow_none else "")
            + ".\nReply with the number only.\nAnswer:")
    return Shape(kind="integer", instruction=tail, prefill="Answer: ",
                 schema={"type": "object", "required": ["value"],
                         "properties": {"value": {"type": "integer",
                                                  "minimum": lo, "maximum": n}}},
                 domain=dom)


def label(options: Sequence[str]) -> Shape:
    """Одна метка из закрытого набора."""
    opts = tuple(options)
    tail = ("\nAnswer with exactly one of: " + ", ".join(opts)
            + ".\nReply with that word only.\nAnswer:")
    return Shape(kind="label", instruction=tail, prefill="Answer: ",
                 schema={"type": "object", "required": ["value"],
                         "properties": {"value": {"type": "string", "enum": list(opts)}}},
                 domain=opts)


def obj(schema: dict[str, Any], *, hint: str = "") -> Shape:
    """Произвольный объект по схеме."""
    return Shape(kind="json", instruction=("\n" + hint if hint else "") + "\nAnswer:",
                 prefill="{", schema=schema)


#: Измеренная лестница для известной модели. НЕ точка старта, а ПОРЯДОК: у одной из
#: моделей ступень оказалась мёртвой, и лестница «на всякий случай пройдём все» была бы
#: тратой вызова на заведомый промах.
#:
#: Замер 04.09.2026, 50 задач на модель и механизм, доля однозначно разобранных ответов:
#:
#: ============================  ======  ========  ======
#: модель                         plain   prefill  schema
#: ============================  ======  ========  ======
#: ``tuned-4b``                    0.00      1.00    1.00
#: ``tuned-ornith-v31``            0.02      1.00    1.00
#: ``gemma4:12b``                  1.00      1.00    1.00
#: ``qwen3.6:27b``                 1.00      0.00    1.00
#: ============================  ======  ========  ======
#:
#: Две вещи, которые видно только из замера и нельзя было угадать:
#:
#: * **Схема даёт 1.00 у ВСЕХ четырёх** — это единственный универсальный механизм, и он
#:   стоит последней ступенью везде.
#: * **У ``qwen3.6`` затравка ломает формат ПОЛНОСТЬЮ** (1.00 своими силами против 0.00 под
#:   затравкой). Модель, которая и так отвечает верно, под затравленной репликой ассистента
#:   скатывается в прозу. Поэтому ``prefill`` из её лестницы убран, а не отодвинут.
#:
#: Оговорка о точности. Механизм меняет не только форму ответа, но и сам ответ: у
#: ``tuned-ornith-v31`` доля ВЕРНЫХ 0.20 под затравкой против 0.10 под схемой, у ``gemma4`` наоборот —
#: 0.16 против 0.26. При n=50 полуширина около 11 п.п., так что это направление, а не
#: установленный факт; но «ограниченное декодирование бесплатно» — неверно, и порядок
#: ступеней стоит перемерять при смене модели.
#: Замер 19.09.2026 перед переводом чата на ``ornith:35b-q4_K_M`` (Ornith-1.5-35B-A3B, MoE,
#: без тюна): 60 задач (40 выбор из пяти + 20 метка из трёх), ``think=False`` — plain 1.00,
#: prefill 1.00, schema 1.00, верных 60/60 под каждым механизмом. Ключ ``ornith`` покрывает и
#: ``ornith:9b`` — та же линейка 1.5 без тюна, но 9B отдельно не мерилась.
#:
#: ДООБУЧЕННЫЕ МОДЕЛИ СЮДА ДОБАВЛЯЕТ РАЗВЁРТЫВАНИЕ, после собственного замера. Ключ — имя модели
#: до «:». У обеих измеренных линеек доменного тюна своими силами формат держался на 0.00 и 0.02,
#: поэтому их лестница — ``(Mechanism.PREFILL, Mechanism.SCHEMA)``, без native::
#:
#:     shape.LADDERS["my-tune"] = (Mechanism.PREFILL, Mechanism.SCHEMA)
LADDERS: dict[str, tuple[Mechanism, ...]] = {
    "ornith": (Mechanism.NATIVE, Mechanism.SCHEMA, Mechanism.PREFILL),
    "gemma4": (Mechanism.NATIVE, Mechanism.SCHEMA, Mechanism.PREFILL),
    "qwen3.6": (Mechanism.NATIVE, Mechanism.SCHEMA),
}

#: Незнакомая модель проходит всю лестницу: «наверное умеет» — не измерение.
DEFAULT_LADDER: tuple[Mechanism, ...] = LADDER


def _mechanisms_for(model: str) -> tuple[Mechanism, ...]:
    key = (model or "").split(":")[0]
    return LADDERS.get(key, DEFAULT_LADDER)


@dataclass(frozen=True, slots=True)
class Shaped:
    """Результат. ``value is None`` — форма не получена; ``reason`` говорит почему."""

    value: Any
    mechanism: Mechanism | None
    attempts: tuple[tuple[Mechanism, str], ...] = ()
    raw: str = ""
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.value is not None


def apply(request: LLMRequest, shape: Shape, mech: Mechanism) -> LLMRequest:
    """Запрос, оформленный под ``mech``. Исходный не мутируется."""
    msgs = [LLMMessage(role=m.role, content=m.content) for m in request.messages]
    if msgs and msgs[-1].role == "user":
        msgs[-1] = LLMMessage(role="user", content=msgs[-1].content + shape.instruction)
    else:
        msgs.append(LLMMessage(role="user", content=shape.instruction.lstrip("\n")))
    out = replace(request, messages=msgs)
    if mech is Mechanism.PREFILL:
        msgs.append(LLMMessage(role="assistant", content=shape.prefill))
        out = replace(out, messages=msgs)
    elif mech is Mechanism.SCHEMA:
        out = replace(out, response_format=shape.schema)
    return out


async def ask(router: Any, request: LLMRequest, shape: Shape) -> Shaped:
    """Спросить и получить ФОРМУ. Поднимается по лестнице, пока не разберётся.

    Возвращает ``Shaped`` с ``value=None``, если ни один механизм не дал разбираемого
    ответа. Это НЕ исключение: «модель не смогла» — обычный исход, и вызывающий обязан
    решить, что с ним делать, а не получить подделанное значение.
    """
    attempts: list[tuple[Mechanism, str]] = []
    last = ""
    for mech in _mechanisms_for(request.model or ""):
        resp = await router.generate(apply(request, shape, mech))
        last = resp.text or ""
        value, why = shape.parse(last)
        if value is not None:
            return Shaped(value=value, mechanism=mech,
                          attempts=tuple(attempts), raw=last)
        attempts.append((mech, why))
    return Shaped(value=None, mechanism=None, attempts=tuple(attempts), raw=last,
                  reason="; ".join(f"{m.value}: {w}" for m, w in attempts))
