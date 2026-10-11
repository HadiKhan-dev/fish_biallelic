"""Exact site-level Potts scoring kernels sharing genotype dosages across samples.

The prepared robust log likelihood is zero for masked or unobserved calls.
Neutral sites therefore never index an uncalled founder allele; scoring skips
them.

The site-major int8 dosage table costs O(L*K²) bytes and removes repeated
founder-pair lookup from O(N*L*K²) scans. Callers retain a direct bounded-memory
kernel when that table is too large.
"""
import math
import numpy as np
from numba import njit, prange

@njit(cache=True, parallel=True, nogil=True)
def prepare_dosages(haps, first, second):
    out = np.empty((haps.shape[1], len(first)), np.int8)
    for site in prange(haps.shape[1]):
        for state in range(len(first)):
            out[site, state] = haps[first[state], site] + haps[second[state], site]
    return out

@njit(cache=True, parallel=True, nogil=True)
def score_dosages(dosages, logs, penalty):
    samples, sites = logs.shape[:2]
    states = dosages.shape[1]
    answer = np.empty(samples)
    center = math.log(1.0 / 3.0)
    for sample in prange(samples):
        scores = np.zeros(states)
        for site in range(sites):
            p0, p1, p2 = (logs[sample, site, 0], logs[sample, site, 1], logs[sample, site, 2])
            if p0 == 0.0 and p1 == 0.0 and (p2 == 0.0):
                continue
            switched = np.max(scores) - penalty
            e0, e1, e2 = (p0 - center, p1 - center, p2 - center)
            for state in range(states):
                dosage = dosages[site, state]
                value = e0 if dosage == 0 else e1 if dosage == 1 else e2
                scores[state] = max(scores[state], switched) + value
        answer[sample] = np.max(scores)
    return answer

