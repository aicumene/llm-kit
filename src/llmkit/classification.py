# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""Data-sensitivity classification.

The four labels govern what may cross from the trusted plane to the cloud plane. Ordering
matters: higher value == more sensitive. Only data at or below ``max_crossable()`` (INTERNAL)
may ever leave the trusted plane, and only after de-identification.

This module is intentionally dependency-free (stdlib only) so the core privacy invariant
can be imported and tested anywhere, including inside the LLM router.
"""

from __future__ import annotations

from enum import IntEnum


class Sensitivity(IntEnum):
    """Sensitivity label assigned to every piece of data at ingest."""

    PUBLIC = 0        # already public (published documents, public web pages, marketing copy)
    INTERNAL = 1      # non-public but non-confidential (de-identified facts, generic Qs)
    CONFIDENTIAL = 2  # client content, work product
    RESTRICTED = 3    # PII, private communications, health/financial records

    @property
    def label(self) -> str:
        return self.name.capitalize()


def max_crossable() -> Sensitivity:
    """The highest sensitivity that may cross to the cloud plane (after de-identification)."""
    return Sensitivity.INTERNAL


def may_cross_to_cloud(sensitivity: Sensitivity) -> bool:
    """True iff data at this sensitivity is *eligible* to cross to the cloud plane.

    Eligibility is necessary but not sufficient — a gateway in front of the cloud plane still
    de-identifies and audits. CONFIDENTIAL / RESTRICTED data is never eligible.
    """
    return sensitivity <= max_crossable()
