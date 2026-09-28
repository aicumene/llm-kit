# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""Savers that work on the passages found for a question, before any prompt is built from them.

Both are lossless in what the model can learn from the prompt:
  * whitespace — runs of spaces and blank lines collapse; a line that is only a page number goes (the
    passage label already carries the page); a short line repeated in several passages of ONE document
    (a running header, a footer) stays in the first of them only.
  * context_dedupe — a passage whose text, whitespace aside, equals or is contained in a passage found
    earlier for the same scope goes once. Across scopes nothing is merged: an answer that spans scopes
    must name the scope every fact comes from.
The remaining passages are renumbered 1…n, so every [n] the model writes still points at its source;
a citations payload built from the same list afterwards stays aligned.

A passage is any object with ``n`` (its marker number), ``document_id``, ``filename``, ``text``,
optionally ``page``, and a scope attribute — ``scope_id`` unless ``scope_attr`` names another.
Dataclasses are copied with :func:`dataclasses.replace`; other objects are changed in place.

Each saver records what it removed (``metering.save``), measured with the ledger's one estimator.
"""

from __future__ import annotations

import dataclasses
import re
from collections import Counter
from typing import Any, Sequence

from llmkit import metering
from llmkit.savers.count import estimate

#: A line that is ONLY a page marker: "Page 3", "Seite 3 von 12", "p. 3", "3 of 12", "- 3 -". A bare
#: number is NOT one: in many documents a paragraph number ("17") stands on its own line and is
#: exactly what an answer cites.
_PAGE_LINE = re.compile(
    r"^(?:(?:page|seite|p\.|s\.)\s*\d{1,4}(?:\s*(?:of|von|/)\s*\d{1,4})?"
    r"|\d{1,4}\s*(?:of|von|/)\s*\d{1,4}"
    r"|[-\u2013\u2014]\s*\d{1,4}\s*[-\u2013\u2014])$", re.IGNORECASE)
_SPACES = re.compile(r"[ \t\u00a0]{2,}")
_TRAILING = re.compile(r"[ \t\u00a0]+\n")
_BLANKS = re.compile(r"\n{3,}")
_WS = re.compile(r"\s+")

#: The attribute a passage's scope is read from, unless the caller names another.
SCOPE_ATTR = "scope_id"


def _label(c: Any) -> str:
    return f"[{c.n}] ({c.filename}" + (f", p.{c.page}" if getattr(c, "page", None) else "") + "):\n"


def _key(text: str) -> str:
    return _WS.sub(" ", text).strip().casefold()


def _replace(c: Any, **changes: Any) -> Any:
    if dataclasses.is_dataclass(c):
        return dataclasses.replace(c, **changes)
    for k, v in changes.items():
        setattr(c, k, v)
    return c


def _furniture(chunks: Sequence[Any]) -> dict[str, set[str]]:
    """Per document: short lines that repeat like a running header or footer — on two or more
    different pages, or (page unknown) in three or more passages. Two neighbouring passages of one
    page share lines because passages overlap; that is content, not furniture, and stays."""
    pages: dict[str, dict[str, set]] = {}
    counts: dict[str, Counter] = {}
    for c in chunks:
        doc = str(c.document_id)
        for ln in {ln.strip() for ln in (c.text or "").splitlines() if 3 <= len(ln.strip()) <= 80}:
            pages.setdefault(doc, {}).setdefault(ln, set()).add(getattr(c, "page", None))
            counts.setdefault(doc, Counter())[ln] += 1
    out: dict[str, set[str]] = {}
    for doc, lines in pages.items():
        out[doc] = {ln for ln, pg in lines.items()
                    if len({p for p in pg if p is not None}) >= 2 or (None in pg and counts[doc][ln] >= 3)}
    return out


def whitespace(chunks: Sequence[Any]) -> tuple[list[Any], int]:
    """Collapse whitespace, drop page-number lines and running headers repeated within a document."""
    furniture = _furniture(chunks)
    seen: dict[str, set[str]] = {}
    out, removed = [], 0
    for c in chunks:
        doc = str(c.document_id)
        repeated = furniture.get(doc, set())
        kept_lines = []
        for ln in (c.text or "").splitlines():
            s = ln.strip()
            if s and _PAGE_LINE.match(s):
                continue
            if s in repeated:
                if s in seen.setdefault(doc, set()):
                    continue
                seen[doc].add(s)
            kept_lines.append(ln)
        text = "\n".join(kept_lines)
        text = _TRAILING.sub("\n", _SPACES.sub(" ", text))
        text = _BLANKS.sub("\n\n", text).strip()
        removed += max(0, estimate(c.text or "") - estimate(text))
        out.append(_replace(c, text=text) if text != c.text else c)
    return out, removed


def dedupe(chunks: Sequence[Any], *, scope_attr: str = SCOPE_ATTR) -> tuple[list[Any], int]:
    """Drop a passage equal to, or contained in, one found earlier for the same scope."""
    kept: list[Any] = []
    keys: list[tuple[str | None, str]] = []
    removed = 0
    for c in chunks:
        k = _key(c.text or "")
        scope = str(getattr(c, scope_attr, None) or "")
        if not k:
            removed += estimate(_label(c))
            continue
        if any(s == scope and (k == other or (len(k) >= 40 and k in other)) for s, other in keys):
            removed += estimate(_label(c)) + estimate(c.text or "")
            continue
        kept.append(c)
        keys.append((scope, k))
    return kept, removed


def apply_to_chunks(chunks: Sequence[Any], enabled: frozenset[str] | set[str], *,
                    scope_attr: str = SCOPE_ATTR) -> list[Any]:
    """Run the enabled passage savers, record what they saved, and renumber the passages 1…n."""
    out = list(chunks)
    if not out:
        return out
    if "whitespace" in enabled:
        out, saved = whitespace(out)
        metering.save("whitespace", saved)
    if "context_dedupe" in enabled:
        out, saved = dedupe(out, scope_attr=scope_attr)
        metering.save("context_dedupe", saved)
    return [c if c.n == i else _replace(c, n=i) for i, c in enumerate(out, start=1)]


__all__ = ["SCOPE_ATTR", "apply_to_chunks", "dedupe", "whitespace"]
