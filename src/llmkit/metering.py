# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""Полный учёт потребления моделей — обёртка ставится ОДИН РАЗ, в фабрике.

ЗАЧЕМ ПЕРЕПИСЫВАТЬ, А НЕ ПЕРЕИСПОЛЬЗОВАТЬ ``UsageRouter``. Тот копит в поле экземпляра, и это
верно, пока его оборачивают вокруг ОДНОГО прогона. Роутер из ``build_router`` — общий: один на всё
приложение и один на воркер. Поле на нём стало бы общим счётчиком всех запросов разом — и гонкой
под нагрузкой, и бессмыслицей по смыслу.

Поэтому счётчик живёт в КОНТЕКСТЕ ЗАДАЧИ (``contextvars``), который asyncio ведёт отдельно для
каждой сопрограммы. Обёртка одна, накопители независимы.

ПОЧЕМУ ОДИН РАЗ В ФАБРИКЕ, А НЕ В КАЖДОМ ПУТИ. Учёт стоял на двух путях из шести, и мимо шли
черновики (три вызова модели на один черновик), ресёрч, чат через маршрут, приём документов и
переиндексация корпуса — вероятно крупнейшая статья. Учёт, покрывающий часть путей, даёт число,
которое выглядит ответом, но занижает неизвестно во сколько раз, и отличить его от правды нечем.
Обёртка в фабрике считает НОВЫЙ путь сама, не дожидаясь, пока о нём вспомнят.

БЕЗ КОНТЕКСТА НИЧЕГО НЕ ПИШЕТСЯ И НИЧЕГО НЕ ЛОМАЕТСЯ. Скрипты, тесты и разовые прогоны просто не
заводят накопитель; обёртка тогда — прозрачный проброс. Это осознанный выбор: учёт не имеет права
ронять работу, ради которой он ведётся.

КУДА ПИШЕТСЯ — решает развёртывание. :func:`flush` отдаёт строки приёмнику :class:`UsageSink`
(таблица, файл, очередь); :class:`InMemoryUsageSink` держит их в памяти — для тестов и скриптов.
"""
from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Meter:
    """Накопитель одной задачи: чем занимались и сколько это стоило в токенах."""

    source: str = "other"
    #: Область, к которой относится расход (проект, тенант, рабочее пространство); None — никакая.
    scope_id: uuid.UUID | None = None
    profile_id: uuid.UUID | None = None
    run_id: uuid.UUID | None = None
    #: (kind, model, plane) → {prompt, completion, calls, cache_read_tokens, cache_write_tokens}.
    #: Свёртка на месте, чтобы тысяча эмбеддингов одной переиндексации не стала тысячей строк книги.
    buckets: dict[tuple[str, str, str], dict[str, int]] = field(default_factory=dict)
    #: Как сделан ход чата: 'local' | 'cloud' | 'device'. None — не ход чата.
    mode: str | None = None
    #: Организация, от имени которой пришёл запрос (например, из утверждения токена), если известна.
    org_id: uuid.UUID | None = None
    #: стадия → {экономайзер: токены промпта, убранные ДО вызова этой стадии}. Стадия, а не вызов:
    #: экономайзер работает над промптом, который ещё не отправлен, и строка книги, к которой
    #: относится экономия, — строка вызова той же стадии.
    savings: dict[str, dict[str, int]] = field(default_factory=dict)

    def add(self, kind: str, model: str, plane: str, *, prompt: int = 0,
            completion: int = 0, calls: int = 1, cache_read: int = 0, cache_write: int = 0,
            est_prompt: int = 0) -> None:
        b = self.buckets.setdefault((kind, model or "", plane),
                                    {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0,
                                     "cache_read_tokens": 0, "cache_write_tokens": 0,
                                     "est_prompt_tokens": 0})
        b["prompt_tokens"] += max(0, prompt)
        b["completion_tokens"] += max(0, completion)
        b["calls"] += max(0, calls)
        b["cache_read_tokens"] += max(0, cache_read)
        b["cache_write_tokens"] += max(0, cache_write)
        # Оценка того же промпта тем же счётчиком, каким экономайзеры меряют убранное. Отношение
        # «настоящий счёт модели / оценка» переводит убранное в токены ЭТОЙ модели (см. _scale).
        b["est_prompt_tokens"] += max(0, est_prompt) if prompt else 0

    def save(self, stage_name: str, saver: str, tokens: int) -> None:
        if tokens > 0:
            s = self.savings.setdefault(stage_name, {})
            s[saver] = s.get(saver, 0) + int(tokens)

    def totals(self) -> dict:
        """Этот ход целиком: сколько токенов ушло в модели, сколько убрали экономайзеры и кэш.

        Убранное — в токенах той модели, чьему вызову оно принадлежит (см. ``_scale``)."""
        sent = sum(b["prompt_tokens"] for (k, _m, _p), b in self.buckets.items() if not k.startswith("embed"))
        out = sum(b["completion_tokens"] for b in self.buckets.values())
        by: dict[str, int] = {}
        for key, per_saver in _place_savings(self).items():
            ratio = _scale(self.buckets[key])
            for saver, n in per_saver.items():
                by[saver] = by.get(saver, 0) + round(n * ratio)
        cache_read = sum(b.get("cache_read_tokens", 0) for b in self.buckets.values())
        cache_write = sum(b.get("cache_write_tokens", 0) for b in self.buckets.values())
        return {"prompt_tokens": sent, "completion_tokens": out, "tokens_saved": sum(by.values()),
                "saved_by": by, "cache_read_tokens": cache_read, "cache_write_tokens": cache_write,
                "mode": self.mode}


_METER: ContextVar[Meter | None] = ContextVar("llmkit_token_meter", default=None)

#: Какая СТАДИЯ работы сейчас идёт внутри источника. Источник ('chat', 'agent', 'ingest')
#: отвечает на вопрос «кто потратил», стадия — «на что»: чтение длинного документа, извлечение
#: фактов, перевод вопроса, критик черновика. Без неё книга показывает, что чат сжёг 19 тыс.
#: токенов, но не показывает, что 16 из них — одно чтение длинного документа, и значит не даёт
#: измерить ни одну попытку это удешевить.
_STAGE: ContextVar[str] = ContextVar("llmkit_token_stage", default="")


@contextmanager
def stage(name: str):
    """Пометить вызовы модели внутри блока стадией. Вложенность: внутренняя побеждает.

    Ничего не знает про накопитель и не требует его: разметка стадии — свойство КОДА, а не
    запроса, поэтому она одинаково работает в проде, в скрипте замера и в тестах.
    """
    token = _STAGE.set(_stage_name(name))
    try:
        yield
    finally:
        _STAGE.reset(token)


def _stage_name(name: str) -> str:
    """Имя стадии в устойчивой форме — такую примет и таблица учёта с ограничением на вид:
    строчные латиница, цифры и «_», с буквы, не длиннее 40. Раньше имя резалось до 24 знаков
    вместе с видом, а в таблицу не проходило вовсе."""
    s = re.sub(r"[^a-z0-9_]+", "_", (name or "").strip().lower()).strip("_")
    if s and not s[0].isalpha():
        s = "s_" + s
    return s[:40]


def current_stage() -> str:
    return _STAGE.get()


def _kind(kind: str) -> str:
    """``generate`` → ``generate:doc_read``. Без стадии — прежняя строка байт в байт, чтобы
    старые записи книги и новые лежали в одних и тех же видах."""
    st = _STAGE.get()
    return f"{kind}:{st}" if st else kind


def save(saver: str, tokens: int) -> None:
    """Экономайзер убрал ``tokens`` токенов промпта ДО вызова модели в текущей стадии.

    Тихо ничего не делает без накопителя — как и всё остальное здесь."""
    m = _METER.get()
    if m is not None:
        m.save(_STAGE.get(), saver, tokens)


def totals() -> dict | None:
    """Сводка текущего хода для ответа клиенту (сколько ушло, сколько сэкономлено)."""
    m = _METER.get()
    return m.totals() if m is not None else None


def breakdown() -> list[dict]:
    """Пофазная разбивка текущего хода: одна строка на (стадия, модель, плоскость) с токенами и
    вызовами, в порядке первого появления (порядок вызовов). Пусто без накопителя.

    Это НЕ книга (книгу пишет ``flush`` в приёмник) — это витрина «куда ушёл каждый шаг» для ответа
    клиенту: тот же источник, что и ``totals``, но НЕ свёрнутый в один итог. ``kind`` бакета уже
    несёт стадию (``generate:triage``) — она разбирается обратно на вид и стадию."""
    m = _METER.get()
    if m is None or not m.buckets:
        return []
    rows: list[dict] = []
    for (kind, model, plane), b in m.buckets.items():
        base, _, st = kind.partition(":")
        rows.append({
            "stage": st or base,
            "kind": base,
            "model": model or "",
            "plane": plane,
            "calls": int(b.get("calls", 0) or 0),
            "prompt_tokens": int(b.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(b.get("completion_tokens", 0) or 0),
            "cache_read_tokens": int(b.get("cache_read_tokens", 0) or 0),
        })
    return rows


def start(source: str = "other", **attribution: Any) -> Meter:
    """Завести накопитель для текущей задачи."""
    m = Meter(source=source, **attribution)
    _METER.set(m)
    return m


def current() -> Meter | None:
    return _METER.get()


def attribute(**kw: Any) -> None:
    """Уточнить, на что списывать, когда это стало известно позже начала запроса.

    Тихо ничего не делает без накопителя: место вызова не обязано знать, ведётся ли учёт.
    Неизвестные поля тоже молча пропускаются.
    """
    m = _METER.get()
    if m is None:
        return
    for k, v in kw.items():
        if v is not None and hasattr(m, k):
            setattr(m, k, v)


def _plane_of(provider: Any) -> str:
    name = type(provider).__name__.lower()
    return "cloud" if ("anthropic" in name or "claude" in name or "cloud" in name) else "trusted"


def _plane_of_response(resp: Any, fallback: Any) -> str:
    """Плоскость берётся С ОТВЕТА, а не угадывается по имени класса.

    Прежде сюда передавалось ``resp.provider``, а это СТРОКА ('anthropic', 'claude-cli',
    'ollama'), и ``_plane_of`` считал ``type(x).__name__`` — то есть всегда 'str'. Ни одного из
    искомых слов там не бывает, поэтому КАЖДЫЙ облачный вызов ложился в книгу как 'trusted', и
    разбивка по плоскостям — единственная, которая стоит денег и приватности, — врала на 100%
    именно на платном пути. У ``LLMResponse`` при этом есть готовое поле ``plane``.
    """
    plane = getattr(resp, "plane", None)
    value = getattr(plane, "value", plane)
    if isinstance(value, str) and value.lower() in ("trusted", "cloud"):
        return value.lower()
    return _plane_of(fallback)


def _unmeasured(kind: str) -> str:
    """Вид работы, счётчики которой получить не удалось.

    Ноль в книге читается как «не стоило ничего». Для упавшего вызова это неправда: промпт уже
    прочитан моделью, GPU уже поработал. Отдельный вид отличает «измерили ноль» от «не смогли
    измерить» — то же различие, которое эта книга и заведена хранить.
    """
    return f"{kind}_unmeasured"


# ── куда пишется книга ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class UsageRow:
    """One ledger row: one (kind, model, plane) bucket of one meter, ready to be persisted.

    ``usage`` is the bucket itself — ``prompt_tokens``, ``completion_tokens``, ``calls``,
    ``cache_read_tokens``, ``cache_write_tokens``, ``est_prompt_tokens``. ``saved_by`` is what
    each saver removed before this call, already scaled to this model's tokens.
    """

    source: str
    kind: str
    model: str
    plane: str
    usage: dict[str, int]
    saved_by: dict[str, int]
    scope_id: uuid.UUID | None = None
    profile_id: uuid.UUID | None = None
    run_id: uuid.UUID | None = None
    mode: str | None = None
    org_id: uuid.UUID | None = None


class UsageSink(Protocol):
    """Where :func:`flush` writes. The deployment implements it over its own storage.

    ``record`` stages one row and returns a truthy value when a row was actually written (a sink
    may decline, e.g. an empty bucket); ``commit`` makes the staged rows durable. An exception
    from either is logged by :func:`flush` and counts as nothing written — the metering never
    breaks the work it measures.
    """

    async def record(self, row: UsageRow) -> Any: ...

    async def commit(self) -> None: ...


class InMemoryUsageSink:
    """Keeps the rows in a list — for tests, scripts and one-off runs."""

    def __init__(self) -> None:
        self.rows: list[UsageRow] = []
        self.commits = 0

    async def record(self, row: UsageRow) -> bool:
        self.rows.append(row)
        return True

    async def commit(self) -> None:
        self.commits += 1


async def flush(sink: UsageSink, *, identity: Callable[[], Any] | None = None) -> int:
    """Записать накопленное в приёмник и очистить накопитель. Возвращает число строк.

    КОММИТИТ САМ (``sink.commit()``), и это не небрежность: сброс происходит после того, как
    обработчик уже закрыл свою транзакцию, а потребление случилось независимо от того, чем
    кончилась работа. Токены, сожжённые упавшим запросом, — такой же расход, как и успешным.

    ``identity`` — КТО ТРАТИТ, если это выяснилось только к сбросу. Накопитель заводится в начале
    запроса, часто раньше, чем известен автор (например, до загрузки профиля), и строка ушла бы
    без автора — а правило доступа «автор видит свои строки» такую строку не пропустит. В момент
    сброса личность уже в силе: хук зовётся, только если у накопителя нет ``profile_id``, и его
    значение (UUID или строка с ним) становится автором; пустое или негодное значение
    пропускается.
    """
    m = _METER.get()
    if m is None or not m.buckets:
        return 0

    if m.profile_id is None and identity is not None:
        ident = identity()
        if ident:
            try:
                m.profile_id = uuid.UUID(str(ident))
            except (TypeError, ValueError):
                pass

    placed = _place_savings(m)
    written = 0
    try:
        for key, b in m.buckets.items():
            kind, model, plane = key
            ratio = _scale(b)
            saved = {saver: round(n * ratio) for saver, n in placed.get(key, {}).items()}
            row = await sink.record(UsageRow(
                source=m.source, kind=kind, model=model, plane=plane, usage=b, saved_by=saved,
                scope_id=m.scope_id, profile_id=m.profile_id, run_id=m.run_id,
                mode=m.mode, org_id=m.org_id))
            written += bool(row)
        await sink.commit()
    except Exception as exc:  # noqa: BLE001
        # Учёт не имеет права ронять работу. Но и молчать нельзя: хранилище, которое «успешно»
        # записало ноль строк, тихо недосчитывает — это тот самый класс отказов, ради которого
        # книга и заводилась.
        log.warning("token metering: failed to write %d buckets for source=%s: %s",
                    len(m.buckets), m.source, exc)
        written = 0
    finally:
        m.buckets.clear()
        m.savings.clear()
    return written


def _scale(bucket: dict) -> float:
    """Во сколько раз токен модели «тяжелее» единицы оценщика — на ЭТОМ вызове.

    Экономайзер не может спросить модель, сколько токенов в тексте, который он убрал: убранное в
    модель не уходит. Поэтому и убранное, и отправленное меряются одним оценщиком
    (``llmkit.savers.estimate``), а в токены модели переводятся отношением её настоящего счёта к
    оценке отправленного. Масштаб оценщика сокращается; остаётся ошибка ФОРМЫ текста (убирают
    пробелы и повторы, а они токенизируются иначе, чем прочее), и её честно называть «оценкой».
    Нет оценки или счёта — множитель 1: в единицах оценщика, но не выдуманное число.
    """
    est, real = bucket.get("est_prompt_tokens", 0), bucket.get("prompt_tokens", 0)
    if est <= 0 or real <= 0:
        return 1.0
    return max(0.25, min(4.0, real / est))


def _place_savings(m: Meter) -> dict[tuple[str, str, str], dict[str, int]]:
    """К какой строке книги относится экономия: к вызову модели той же стадии (генерация или
    поток — эмбеддинг экономайзеры не трогают). Нет такого вызова (он упал раньше счётчиков) —
    к самому крупному вызову хода: экономия была, и терять её в отчёте так же неверно, как
    писать нулём неизмеренный расход."""
    placed: dict[tuple[str, str, str], dict[str, int]] = {}
    calls = [k for k in m.buckets if not k[0].startswith("embed")]
    for st, by in m.savings.items():
        want = {f"generate:{st}", f"stream:{st}"} if st else {"generate", "stream"}
        target = next((k for k in calls if k[0] in want), None)
        if target is None and calls:
            target = max(calls, key=lambda k: m.buckets[k]["prompt_tokens"])
        if target is None:
            log.warning("token metering: %d saved tokens of stage %r have no call to belong to",
                        sum(by.values()), st)
            continue
        acc = placed.setdefault(target, {})
        for saver, n in by.items():
            acc[saver] = acc.get(saver, 0) + n
    return placed


def _estimate_request(request: Any) -> int:
    """Оценка промпта запроса тем же счётчиком, каким экономайзеры меряют убранное."""
    try:
        from llmkit.savers.count import estimate
        return sum(estimate(str(getattr(msg, "content", "") or "")) for msg in (getattr(request, "messages", None) or ()))
    except Exception:  # noqa: BLE001 — оценка не имеет права ронять вызов
        return 0


class MeteredRouter:
    """Прозрачная обёртка: считает каждый вызов модели в накопитель текущей задачи."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def _model_of(self, resp: Any) -> str:
        return str(getattr(resp, "model", "") or "")[:64]

    async def generate(self, request: Any, **kwargs: Any):
        m = _METER.get()
        est = _estimate_request(request) if m is not None else 0
        try:
            resp = await self._inner.generate(request, **kwargs)
        except BaseException:
            # Упавший вызов — такой же расход: модель прочитала промпт. Счётчиков нет, поэтому
            # вид помечается неизмеренным, а не пишется нулём.
            if m is not None:
                m.add(_kind(_unmeasured("generate")), "", "trusted")
            raise
        if m is not None:
            u = getattr(resp, "usage", None) or {}
            m.add(_kind("generate"), self._model_of(resp),
                  _plane_of_response(resp, self._inner),
                  prompt=int(u.get("prompt_tokens", 0) or 0),
                  completion=int(u.get("completion_tokens", 0) or 0),
                  cache_read=int(u.get("cache_read_tokens", 0) or 0),
                  cache_write=int(u.get("cache_write_tokens", 0) or 0),
                  est_prompt=est)
        return resp

    async def stream(self, request: Any, **kwargs: Any):
        m = _METER.get()
        sink: dict[str, int] = {"est_prompt": _estimate_request(request) if m is not None else 0}
        try:
            try:
                async for chunk in self._inner.stream(request, usage=sink, **kwargs):
                    yield chunk
            except TypeError:
                # Провайдер, не знающий про приёмник. Вызов всё равно считается: строка с
                # нулевыми токенами и ненулевым счётчиком отличима от «не потребляли», и это
                # единственный способ заметить, что появился глухой путь.
                async for chunk in self._inner.stream(request, **kwargs):
                    yield chunk
        finally:
            # FINALLY, а не после цикла. Оборванный поток — сожжённые токены промпта, и
            # докстрока flush() прямо называет их таким же расходом, как у успешного запроса.
            # Прежде отказ модели на середине генерации и закрытая вкладка (GeneratorExit в
            # точке yield) не оставляли в книге НИ ОДНОЙ строки: расход был неотличим от
            # отсутствия запроса. Соседняя обёртка UsageRouter считает вызов ДО цикла именно
            # поэтому.
            self._record_stream(m, sink)

    @staticmethod
    def _record_stream(m: Any, sink: dict) -> None:
        if m is not None:
            # Живая Ollama кладёт счётчики ТОЛЬКО в финальном кадре: у оборванного потока приёмник
            # пуст, и запись нулём утверждала бы, что 30 тысяч токенов промпта ничего не стоили.
            measured = "prompt_tokens" in sink or "completion_tokens" in sink
            if not measured:
                m.add(_kind(_unmeasured("stream")), "", str(sink.get("plane", "trusted")))
                return
            # Модель и плоскость раньше были захардкожены ('' и 'trusted'). Разбивка по
            # плоскостям при этом врала бы тем увереннее, чем больше потока идёт в облако.
            m.add(_kind("stream"), str(sink.get("model", "")), str(sink.get("plane", "trusted")),
                  prompt=int(sink.get("prompt_tokens", 0) or 0),
                  completion=int(sink.get("completion_tokens", 0) or 0),
                  est_prompt=int(sink.get("est_prompt", 0) or 0))


    async def embed(self, texts: Any, **kwargs: Any):
        m = _METER.get()
        sink: dict[str, int] = {}
        try:
            try:
                out = await self._inner.embed(texts, usage=sink, **kwargs)
            except TypeError:
                out = await self._inner.embed(texts, **kwargs)
        except BaseException:
            # Та же дыра, что была у потока: насыщенный хост эмбеддера рвёт батч на середине,
            # работа сделана, а в книге не появлялось ни строки.
            if m is not None:
                m.add(_kind(_unmeasured("embed")), "", "trusted")
            raise
        if m is not None:
            # Модель НЕ константа, по той же причине, по которой она перестала быть константой
            # у потока: после смены эмбеддера весь прогон переиндексации ложился в отчёт под
            # пустым ключом, вместе с эмбеддингами прежней модели, и вопрос «сколько GPU-работы
            # съел новый эмбеддер» книга не отвечала.
            m.add(_kind("embed"), str(sink.get("model", "")), str(sink.get("plane", "trusted")),
                  prompt=int(sink.get("prompt_tokens", 0) or 0))
        return out

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
