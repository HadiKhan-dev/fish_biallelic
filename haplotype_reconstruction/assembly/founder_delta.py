"""Exact full-site scores for edits inside unchanged block-aligned flanks.

Boundary messages use the canonical complete-site/neutral-observation mask.
The cached panel and evidence are fixed; a proposal must have the same founder
labels/count and unchanged rows outside the rescored interval.
"""
import math
import numpy as np
from numba import njit, prange
from .founder_scoring import _unordered_pairs


@njit(cache=True, parallel=True, nogil=True)
def messages(haps, logs, offsets, penalty):
    first, second = _unordered_pairs(len(haps))
    samples, blocks = len(logs), len(offsets)-1
    prefix = np.zeros((blocks+1, samples, len(first)))
    suffix = np.zeros_like(prefix)
    center = math.log(1./3.)
    for sample in prange(samples):
        row = np.zeros(len(first))
        for block in range(blocks):
            for site in range(offsets[block], offsets[block+1]):
                emission = logs[sample, site]
                if emission[0] == 0. and emission[1] == 0. and emission[2] == 0.:
                    continue
                switched = np.max(row)-penalty
                for state in range(len(first)):
                    dosage = haps[first[state], site]+haps[second[state], site]
                    row[state] = max(row[state], switched)+(emission[dosage]-center)
            prefix[block+1, sample] = row
        row[:] = 0.
        for block in range(blocks-1, -1, -1):
            for site in range(offsets[block+1]-1, offsets[block]-1, -1):
                emission = logs[sample, site]
                if emission[0] == 0. and emission[1] == 0. and emission[2] == 0.:
                    continue
                for state in range(len(first)):
                    dosage = haps[first[state], site]+haps[second[state], site]
                    row[state] += emission[dosage]-center
                switched = np.max(row)-penalty
                for state in range(len(first)):
                    row[state] = max(row[state], switched)
            suffix[block, sample] = row
    return prefix, suffix


@njit(cache=True, parallel=True, nogil=True)
def score_interval(leaves, offsets, selected, logs, penalty, prefix, suffix, start, stop):
    first, second = _unordered_pairs(len(selected))
    result = np.empty(len(logs))
    center = math.log(1./3.)
    for sample in prange(len(logs)):
        row = prefix[sample].copy()
        for block in range(start, stop):
            local = leaves[block]
            for site in range(offsets[block], offsets[block+1]):
                emission = logs[sample, site]
                if emission[0] == 0. and emission[1] == 0. and emission[2] == 0.:
                    continue
                switched = np.max(row)-penalty
                position = site-offsets[block]
                for state in range(len(first)):
                    dosage = local[selected[first[state], block], position]+local[selected[second[state], block], position]
                    row[state] = max(row[state], switched)+(emission[dosage]-center)
        result[sample] = np.max(row+suffix[sample])
    return result
