# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""One token counter for both sides of a saving.

A saver cannot ask the model how many tokens the text it removed would have cost — removed text is
never sent. So the removed text and the prompt that was sent are measured with this one estimator,
and the ledger turns the removed amount into the model's own tokens with the ratio "the model's real
count of the prompt / this estimate of it", call by call (``metering._scale``). The estimator's
absolute scale cancels out; what remains is the difference in SHAPE between what savers remove
(spaces, repeated lines) and ordinary text, and a report should call the result an estimate.

The shape it approximates is a BPE tokenizer's: a short word is one token, a long word several,
every punctuation mark one, and runs of whitespace beyond a single space cost tokens of their own.
"""

from __future__ import annotations

import re

_PIECE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_WS_RUN = re.compile(r"[ \t]{2,}|\n")


def estimate(text: str) -> int:
    if not text:
        return 0
    n = 0
    for m in _PIECE.finditer(text):
        w = m.group(0)
        n += 1 if len(w) <= 4 else (len(w) + 3) // 4
    return n + len(_WS_RUN.findall(text))


__all__ = ["estimate"]
