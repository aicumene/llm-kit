# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""Token savers: what they may remove, what they must keep, and what the catalog lets run.

The two passage savers are lossless in what the model can learn from the prompt — that is the claim
their "on" status rests on, so it is pinned here: page markers and running headers go, a paragraph
number on its own line stays, a line two overlapping passages of one page share stays, a passage of
another scope is never merged away, and every [n] still points at its source.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import pytest

from llmkit import metering, savers
from llmkit.savers import catalog, passages


@dataclass(slots=True)
class Passage:
    """The shape of a retrieved passage the savers work on."""

    n: int
    document_id: str
    filename: str
    page: int | None
    text: str
    scope_id: str | None = None


def _c(n, text, *, doc="d1", page=1, scope="s1", name="report.pdf"):
    return Passage(n=n, document_id=doc, filename=name, page=page, text=text, scope_id=scope)


# ── whitespace ───────────────────────────────────────────────────────────────────────────────
def test_whitespace_collapses_runs_and_drops_page_markers_but_keeps_a_paragraph_number():
    text = ("Section 8.2   (Retention limits)\n\n\n\nNo backup shall be kept    longer.\n"
            "Page 3 of 12\n17\nThe audit found …")
    (out,), removed = passages.whitespace([_c(1, text)])
    assert "Page 3 of 12" not in out.text
    assert "\n17\n" in out.text                                    # the paragraph number an answer cites
    assert "   " not in out.text and "\n\n\n" not in out.text
    assert removed > 0


def test_a_no_break_space_run_collapses_like_any_other():
    (out,), _ = passages.whitespace([_c(1, "alpha" + chr(0xA0) * 3 + "beta")])
    assert out.text == "alpha beta"


def test_a_running_header_on_several_pages_stays_once_and_overlap_on_one_page_stays_twice():
    header = "ACME HOLDINGS GMBH — OPERATIONS HANDBOOK"
    shared = "Section 9.1 (Escalation)"
    chunks = [_c(1, f"{header}\nSection 8.2 applies.\n{shared}", page=3),
              _c(2, f"{shared}\nAny incident must first be reported.", page=3),   # overlap, same page
              _c(3, f"{header}\nSection 12 governs notices.", page=4)]
    out, _ = passages.whitespace(chunks)
    assert header in out[0].text and header not in out[2].text        # furniture: once
    assert shared in out[0].text and shared in out[1].text            # content: stays where it is


# ── dedupe ───────────────────────────────────────────────────────────────────────────────────
LONG = "No backup shall be kept for more than one thousand days in any rolling twelve-month period."


def test_dedupe_drops_repeats_and_contained_passages_of_the_same_scope_only():
    chunks = [_c(1, LONG), _c(2, "  " + LONG.upper() + " "),                   # same text, other case/space
              _c(3, "one thousand days in any rolling twelve-month period"),   # contained
              _c(4, LONG, scope="s2", doc="d9")]                             # another scope: kept
    out, removed = passages.dedupe(chunks)
    assert [c.scope_id for c in out] == ["s1", "s2"]
    assert removed > 0


def test_the_scope_attribute_can_be_named_by_the_caller():
    """A passage type that carries its scope under another name says so; the wrong name would put
    every passage into one scope and merge across scopes."""

    @dataclass
    class Other:
        n: int
        document_id: str
        filename: str
        page: int | None
        text: str
        tenant: str

    chunks = [Other(1, "d1", "a.pdf", 1, LONG, "t1"), Other(2, "d2", "b.pdf", 1, LONG, "t2")]
    assert len(passages.dedupe(chunks, scope_attr="tenant")[0]) == 2
    assert len(passages.dedupe(chunks)[0]) == 1        # no scope_id on these: one scope for all


def test_apply_renumbers_so_every_marker_points_at_its_source_and_records_the_saving():
    meter = metering.start("chat")
    chunks = [_c(1, "Alpha clause text that is long enough to matter here."),
              _c(2, "Alpha clause text that is long enough to matter here."),
              _c(3, "Beta clause, a different provision entirely.", doc="d2", name="minutes.pdf")]
    out = savers.apply_to_chunks(chunks, frozenset({"context_dedupe", "whitespace"}))
    assert [c.n for c in out] == [1, 2]
    assert [c.filename for c in out] == ["report.pdf", "minutes.pdf"]
    assert meter.savings[""]["context_dedupe"] > 0
    metering._METER.set(None)


def test_disabled_savers_change_nothing():
    chunks = [_c(1, "a  b"), _c(2, "a  b")]
    assert savers.apply_to_chunks(chunks, frozenset()) == chunks


# ── the catalog ──────────────────────────────────────────────────────────────────────────────
def test_the_catalog_names_an_origin_for_every_saver_and_the_rejected_one_keeps_its_measurement():
    assert all(s.origin and s.note for s in catalog.CATALOG)
    laya = catalog.BY_ID["system1_gate"]
    assert laya.status == "rejected" and "0.471" in laya.note
    origins = " ".join(s.origin for s in catalog.CATALOG)
    for sent in ("Portal", "SoL-Pi", "laya"):                       # the published sources are all there
        assert sent in origins


@pytest.mark.parametrize("mode", ["local", "cloud", "device"])
def test_only_implemented_savers_run_and_an_organization_can_turn_an_on_saver_off(mode):
    on = catalog.enabled({}, mode)
    assert not on & {s.id for s in catalog.CATALOG if s.status in ("planned", "rejected")}
    assert "context_dedupe" in on and "whitespace" in on
    assert "context_dedupe" not in catalog.enabled({"context_dedupe": False}, mode)
    assert ("prompt_cache" in on) == (mode == "cloud")


def test_built_in_savers_cannot_be_switched_off_and_describe_says_so():
    assert "shape_routing" in catalog.enabled({"shape_routing": False}, "cloud")
    rows = {r["id"]: r for r in catalog.describe({"whitespace": False})}
    assert rows["whitespace"]["enabled"] is False and rows["whitespace"]["switchable"] is True
    assert rows["shape_routing"]["enabled"] is True and rows["shape_routing"]["switchable"] is False
    assert rows["observation_pack"]["enabled"] is False and rows["observation_pack"]["switchable"] is False


def test_a_deployment_can_present_the_catalog_in_its_own_words():
    own = tuple(dataclasses.replace(s, title=s.title.upper()) for s in catalog.CATALOG)
    rows = catalog.describe({}, catalog=own)
    assert rows[0]["title"] == catalog.CATALOG[0].title.upper()
    for mode in ("local", "cloud", "device"):
        assert catalog.enabled({}, mode, catalog=own) == catalog.enabled({}, mode)


def test_the_estimator_counts_what_the_whitespace_saver_removes():
    assert savers.estimate("a    b\n\n\n\nc") > savers.estimate("a b\n\nc")
    assert savers.estimate("") == 0
