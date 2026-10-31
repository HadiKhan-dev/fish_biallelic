"""Family-phase site windows; chromosome-wide bin recursions remain in release/model.

Windows end at complete bin boundaries, preserving each orientation median. A very large individual bin is
processed whole. The record reader bounds its compressed cache independently of these decoded site windows.
"""
from __future__ import annotations

import numpy as np


SITE_WINDOW = 4096
# Exact decoded evidence-array bytes, not a prediction of the chromosome's total peak RSS.
RESIDENT_EVIDENCE_BYTES = 8 << 30


def use_windows(evidence_bytes, storage="auto"):
    """Use resident buffers up to 8 GiB of decoded likelihoods; larger chromosomes use site windows.

    Overrides exercise both storage paths in replays. No scientific configuration, threshold or record changes.
    """
    if storage == "auto":
        return evidence_bytes > RESIDENT_EVIDENCE_BYTES
    if storage not in ("resident", "windowed"):
        raise ValueError("family-phase storage must be auto, resident or windowed")
    return storage == "windowed"


class SampleSites:
    """Float32 site likelihoods of one sample, gathered only as a release requests a site window."""

    def __init__(self, likelihoods, sample, columns):
        self.likelihoods, self.sample, self.columns = likelihoods, sample, columns

    def __getitem__(self, sites):
        return self.likelihoods.gather(slice(self.sample, self.sample + 1), self.columns[sites])[0].astype(np.float32)


def site_windows(site_bin, lo=0, hi=None):
    """Half-open site windows, extending the last site to the end of its bin."""
    hi = len(site_bin) if hi is None else hi
    while lo < hi:
        stop = min(hi, lo + SITE_WINDOW)
        if stop < hi:
            stop = min(hi, int(np.searchsorted(site_bin, site_bin[stop - 1], side="right")))
        yield lo, stop
        lo = stop
