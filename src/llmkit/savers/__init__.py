# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""Token savers: what makes a prompt smaller before it is sent, and the record of what each saved.

The catalog (``catalog.py``) lists every saver this toolkit knows — the ones taken from published
work (Portal by Spotify, NVlabs SoL-Pi, receptron/laya) and the ones found since — with its origin
and its status in the efficiency track. What a saver removes is recorded per call
(``metering.save`` → the ``saved_by`` of the call's usage row), in the model's own tokens.
"""

from llmkit.savers.catalog import BY_ID, CATALOG, Saver, describe, enabled
from llmkit.savers.count import estimate
from llmkit.savers.passages import apply_to_chunks

__all__ = ["BY_ID", "CATALOG", "Saver", "apply_to_chunks", "describe", "enabled", "estimate"]
