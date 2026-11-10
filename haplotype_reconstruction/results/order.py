"""Recover the original sample column order from a run's bound source descriptor.

These helpers read the bound sample order for output rendering only.
They infer no biological labels and leave permutation validation to contracts.samples.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["recorded_input_order"]


def recorded_input_order(manifest: Mapping[str, Any] | None) -> tuple[str, ...] | None:
    """Return recorded input sample identifiers, or None when that order was not stored."""
    source = (manifest or {}).get("source") or {}
    ids = source.get("input_sample_ids")
    return None if ids is None else tuple(str(name) for name in ids)
