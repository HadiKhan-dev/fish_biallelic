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
from .scoring import _unordered_pairs


@njit(cache=True)
def _block_sample_scores(haps, logs, start, stop, penalty, removed):
    """Per-sample optimistic scores of one block (its rows minus row ``removed`` if >= 0)."""
    rows = np.arange(len(haps))
    if removed >= 0:
        rows = rows[rows != removed]
    first, second = _unordered_pairs(len(rows))
    result = np.empty(logs.shape[0], np.float64)
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
        result[sample] = np.max(scores)
    return result


@njit(cache=True, parallel=True)
def upper_bound(leaves, offsets, logs, penalty, target):
    """Per-block optimistic scores (bounds, constrained) and their per-sample form.

    bounds[b] is the block's score summed over samples under the relaxation above; a
    target+1-row block takes its best one-row deletion (constrained[b] = 1). per_sample[b]
    is the same relaxation further relaxed so that every sample may drop a different row
    of a target+1-row block (a maximum of sums becomes a sum of maxima); summed over blocks
    it bounds each sample, as count criteria that weigh samples separately need. Both come
    from one evaluation of the per-sample block scores (a block total is their in-order sum).
    """
    samples = logs.shape[0]
    bounds = np.empty(len(leaves), np.float64)
    constrained = np.zeros(len(leaves), np.int64)
    per_sample = np.empty((len(leaves), samples), np.float64)
    for block_index in prange(len(leaves)):
        block = np.int64(block_index)
        haps = leaves[block]
        deletion = len(haps) == target + 1
        best = -np.inf
        best_sample = np.full(samples, -np.inf)
        for removed in range(len(haps) if deletion else 1):
            scores = _block_sample_scores(haps, logs, offsets[block], offsets[block + 1], penalty,
                                          removed if deletion else -1)
            total = 0.
            for sample in range(samples):
                total += scores[sample]
            best = max(best, total)
            best_sample = np.maximum(best_sample, scores)
        bounds[block] = best
        per_sample[block] = best_sample
        constrained[block] = int(deletion)
    return bounds, constrained, per_sample
