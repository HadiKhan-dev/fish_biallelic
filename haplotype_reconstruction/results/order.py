"""Recover the original sample column order from a run's bound source descriptor.

These are the storage-reading helpers of 3.5 evidence.sample_order (EC03; OU09).
They infer no biological labels and leave permutation validation to contracts.samples.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

__all__ = ["recorded_input_order", "store_input_order"]


class _ManifestReader(Protocol):
    def manifest(self, key: str) -> Mapping[str, Any] | None: ...


def recorded_input_order(manifest: Mapping[str, Any] | None) -> tuple[str, ...] | None:
    """Return recorded input sample identifiers, or None when that order was not stored."""
    source = (manifest or {}).get("source") or {}
    ids = source.get("input_sample_ids")
    return None if ids is None else tuple(str(name) for name in ids)


def store_input_order(store: _ManifestReader) -> tuple[str, ...] | None:
    """Read the source's input sample order from manifest/run without any scientific payload."""
    return recorded_input_order(store.manifest("run"))
