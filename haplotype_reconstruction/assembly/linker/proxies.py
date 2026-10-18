"""Linking proxies: the sites of an input component the linker (and the batch partition) reads."""
from __future__ import annotations

import math

import numpy as np


def proxy_from_usable(usable: np.ndarray, max_sites: int) -> np.ndarray:
    """Site indices (int64, ascending) of a block's linking proxy from its usable-site mask.

    A block with at most ``max_sites`` sites keeps every site (emissions marginalise unknown alleles
    at partial sites). A larger block keeps a stride over its usable sites (kept, and called by at
    least one founder's pre-fill calls) so that at most ``max_sites`` remain; a block without a
    usable site keeps every site (it stays an unresolved pass-through).
    """
    total = len(usable)
    if total <= max_sites:
        return np.arange(total, dtype=np.int64)
    candidates = np.flatnonzero(usable)
    if not len(candidates):
        return np.arange(total, dtype=np.int64)
    stride = max(1, math.ceil(len(candidates) / max_sites))
    return candidates[::stride].astype(np.int64)


def proxy_indices(inference: np.ndarray, kept: np.ndarray, max_sites: int) -> np.ndarray:
    """Linking-proxy site indices of a block from its pre-fill calls int8[K, L] and kept mask."""
    return proxy_from_usable(np.any(inference >= 0, axis=0) & kept, max_sites)
