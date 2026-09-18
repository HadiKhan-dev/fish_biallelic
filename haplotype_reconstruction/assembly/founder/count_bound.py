"""Optimistic full-site score for a reduced-count founder panel.

Allow each prepared block to choose its own panel and reset every sample's
diplotype freely between blocks. When a block has target+1 local rows, its
best one-row deletion is exact for that block; otherwise allow every local
row (possibly more than target). Both relaxations enlarge the feasible set.
Thus the sum bounds above every chromosome panel with <=target founder paths
drawn from these prepared candidates. The caller must use the same fixed
complete-site mask, emissions and chromosome-wide switch penalty as fitting.

This bound may be loose, but must never rule out an improving admissible fit.
"""
import math
import numpy as np
from numba import njit, prange
from.scoring import _unordered_pairs


@njit(cache=True)
def _block_score(haps, logs, start, stop, penalty, removed):
    rows = np.arange(len(haps))
    if removed >= 0:
        rows = rows[rows != removed]
    first, second = _unordered_pairs(len(rows))
    total = 0.
    center = math.log(1. / 3.)
    for sample in range(logs.shape[0]):
        scores = np.zeros(len(first), np.float64)
        for site in range(start, stop):
            emission = logs[sample, site]
            # prepare_log_evidence returns exactly zero for unobserved,
            # equal-likelihood and incomplete sites. Skip before indexing haps.
            if emission[0] == 0. and emission[1] == 0. and emission[2] == 0.:
                continue
            switched = np.max(scores) - penalty
            for state in range(len(first)):
                dosage = haps[rows[first[state]], site - start] + haps[rows[second[state]], site - start]
                scores[state] = max(scores[state], switched) + emission[dosage] - center
        total += np.max(scores)
    return total


@njit(cache=True, parallel=True)
def upper_bound(leaves, offsets, logs, penalty, target):
    bounds = np.empty(len(leaves), np.float64)
    constrained = np.zeros(len(leaves), np.int64)
    for block_index in prange(len(leaves)):
        block = np.int64(block_index)
        haps = leaves[block]
        if len(haps) == target + 1:
            best = -np.inf
            for removed in range(len(haps)):
                best = max(best, _block_score(haps, logs, offsets[block], offsets[block + 1],
                                              penalty, removed))
            bounds[block] = best
            constrained[block] = 1
        else:
            bounds[block] = _block_score(haps, logs, offsets[block], offsets[block + 1], penalty, -1)
    return bounds, constrained
