# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""Every token saver this toolkit knows, where it came from, and whether it may run.

The rule is the efficiency track's: a saver that only removes repetition and whitespace, or makes
the provider bill a repeat cheaper, is on; a saver that changes what the model reads stays off until
it survives the track's floor on the gold sets — for an answer someone relies on, the missing few
per cent are exactly the questions that matter most. Rejected ones stay listed with the measurement
that rejected them, so nobody proposes them again without new data.

Statuses:
  on        implemented, on unless an organization turns it off;
  off       implemented, off unless an organization turns it on (not yet measured on the gold sets);
  built_in  part of how a mode works; not switchable;
  planned   in the track, not implemented yet;
  rejected  measured, no effect worth its risk; kept for the record.

``enabled`` and ``describe`` read :data:`CATALOG` unless given another catalog, so a deployment
can present the same savers in its own words (the ids, statuses and modes are what decide).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

Status = Literal["on", "off", "built_in", "planned", "rejected"]
Mode = Literal["local", "cloud", "device"]


@dataclass(frozen=True)
class Saver:
    id: str
    title: str
    origin: str
    effect: Literal["lossless", "lossy"]
    status: Status
    modes: tuple[str, ...]
    note: str

    def as_dict(self) -> dict:
        d = asdict(self)
        d["modes"] = list(self.modes)
        return d


CATALOG: tuple[Saver, ...] = (
    Saver("context_dedupe", "Repeated passages sent once", "llm-kit", "lossless", "on",
          ("local", "cloud", "device"),
          "A passage found twice, or contained in a longer passage of the same scope, goes into the "
          "prompt once; the remaining passages are renumbered, so every [n] still points at its source."),
    Saver("whitespace", "Whitespace and page furniture", "llm-kit", "lossless", "on",
          ("local", "cloud", "device"),
          "Runs of spaces and blank lines collapse; page-number lines go (the passage label keeps the page), "
          "and a line repeated in several passages of one document stays in the first of them."),
    Saver("prompt_cache", "Prompt cache (Anthropic)", "Anthropic prompt caching", "lossless", "on",
          ("cloud",),
          "The part of the prompt that repeats within a scope — the instructions, the glossary and other "
          "shared context — is marked for Anthropic's cache; a read from the cache is billed at a tenth. "
          "Counted exactly from Anthropic's own usage report."),
    Saver("shape_routing", "Utility steps on local models", "Portal (Spotify engineering), 07.09.2026",
          "lossless", "built_in", ("cloud",),
          "In cloud mode the utility steps — extracting facts, translating the question, reading a long "
          "source document — run on the deployment's own models; only the answer is written by the cloud "
          "model."),
    Saver("evidence_reducer", "Checked quotes instead of whole documents", "SoL-Pi (NVlabs), 23.09.2026",
          "lossless", "built_in", ("local", "cloud"),
          "A long source document is read once, and only verbatim quotes, each checked against the "
          "document's text, go into the answer's prompt."),
    Saver("observation_pack", "Windows around candidate paragraphs", "SoL-Pi (NVlabs), 23.09.2026",
          "lossy", "planned", ("local", "cloud"),
          "The document reader would see the paragraphs around the candidates and a pointer to the rest. "
          "Turned on only if the decisive paragraph is still found 4/4 on the pinned probes."),
    Saver("prefix_reuse", "Stable prompt order for the model server's cache", "Efficiency track, 22.09.2026",
          "lossless", "planned", ("local",),
          "The retrieved-context prompt and the system context ordered so the model server's prefix cache "
          "hits across turns."),
    Saver("early_exit", "Stop a stage once the answer is in hand", "Efficiency track, 22.09.2026",
          "lossy", "planned", ("local", "cloud"),
          "Skip the cross-lingual search when the named source is already found; stop reading when the "
          "checked quotes already answer."),
    Saver("action_fusion", "Critic and judge in one call", "SoL-Pi (NVlabs), 23.09.2026", "lossy", "planned", (),
          "Drafts, not the chat: the critic and the judge read the same two texts."),
    Saver("context_compact", "Only step results travel between agent steps", "SoL-Pi (NVlabs), 23.09.2026",
          "lossy", "planned", (),
          "Agent runs, not the chat."),
    Saver("system1_gate", "A classifier instead of a model call", "receptron/laya, 20.09.2026", "lossy", "rejected", (),
          "Measured 23.09.2026 on 247 gold records (does the retrieved source answer the question?): AUC "
          "0.471 on the hard stratum, i.e. no signal. Not used."),
)

BY_ID = {s.id: s for s in CATALOG}


def enabled(overrides: dict[str, bool] | None, mode: str,
            catalog: Sequence[Saver] | None = None) -> frozenset[str]:
    """The savers that run for a chat turn in ``mode`` under an organization's choices."""
    chosen = overrides or {}
    out = set()
    for s in (CATALOG if catalog is None else catalog):
        if mode not in s.modes:
            continue
        if s.status == "built_in":
            out.add(s.id)
        elif s.status == "on" and chosen.get(s.id, True) is not False:
            out.add(s.id)
        elif s.status == "off" and chosen.get(s.id) is True:
            out.add(s.id)
    return frozenset(out)


def describe(overrides: dict[str, bool] | None,
             catalog: Sequence[Saver] | None = None) -> list[dict]:
    """The catalog as a panel shows it, with each saver's effective state for this organization."""
    chosen = overrides or {}
    rows = []
    for s in (CATALOG if catalog is None else catalog):
        d = s.as_dict()
        if s.status in ("on", "off"):
            default = s.status == "on"
            d["enabled"] = chosen.get(s.id, default) if isinstance(chosen.get(s.id, default), bool) else default
            d["switchable"] = True
        else:
            d["enabled"] = s.status == "built_in"
            d["switchable"] = False
        rows.append(d)
    return rows


__all__ = ["BY_ID", "CATALOG", "Saver", "describe", "enabled"]
