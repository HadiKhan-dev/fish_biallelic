"""Preserve the run and published sample orders (EC03), without reading storage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ..core.errors import InvariantError
from ..core.ties import run_sample_order

__all__ = ["SAMPLE_ORDER_RULE", "INPUT_ORDER_KEY", "run_sample_order", "recorded_input_order", "input_rows"]
SAMPLE_ORDER_RULE = "sample-name-v1"
INPUT_ORDER_KEY = "input_sample_ids"


def recorded_input_order(manifest: Mapping | None) -> tuple[str, ...] | None:
    """The input sample order a run manifest records (``manifest["source"]["input_sample_ids"]``), or None for
    runs without the canonical order (their records are in input order)."""
    source = (manifest or {}).get("source") or {}
    ids = source.get(INPUT_ORDER_KEY)
    return None if ids is None else tuple(str(name) for name in ids)


def input_rows(sample_ids: Sequence[str], input_ids: Sequence[str] | None) -> np.ndarray | None:
    """Rows of a sample axis ``sample_ids`` in the order ``input_ids`` (int64[N]: output row i is input_ids[i]),
    or None when the orders already agree (or no input order is recorded)."""
    if input_ids is None:
        return None
    sample_ids, input_ids = tuple(map(str, sample_ids)), tuple(map(str, input_ids))
    if sample_ids == input_ids:
        return None
    index = {name: row for row, name in enumerate(sample_ids)}
    if len(index) != len(sample_ids) or len(input_ids) != len(sample_ids) or set(input_ids) != set(index):
        raise InvariantError("the recorded input sample order is not a permutation of the run's samples")
    return np.asarray([index[name] for name in input_ids], dtype=np.int64)
