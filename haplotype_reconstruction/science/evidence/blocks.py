"""Partition kept sites into the fixed discovery blocks."""

import numpy as np


def _block_spans(n_sites: int, width: int) -> np.ndarray:
    starts = np.arange(0, n_sites, width, dtype=np.int64)
    return np.column_stack((starts, np.minimum(starts + width, n_sites)))


__all__ = []
