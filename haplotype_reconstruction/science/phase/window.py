"""Whole-bin windows preserve phase orientation medians (P0 FP22)."""

import numpy as np

from .parameters import SITE_WINDOW


def site_windows(site_bin, lo=0, hi=None, window=SITE_WINDOW):
    """Half-open site windows of at least ``window`` sites (but the last), extending the last site to the end of its
    bin."""
    hi = len(site_bin) if hi is None else hi
    while lo < hi:
        stop = min(hi, lo + window)
        if stop < hi:
            stop = min(hi, int(np.searchsorted(site_bin, site_bin[stop - 1], side="right")))
        yield lo, stop
        lo = stop


__all__ = ["site_windows"]
