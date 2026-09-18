"""Exact site-level Potts kernels sharing genotype dosages across samples.

The prepared robust log likelihood is zero for masked or unobserved calls.
Neutral sites therefore never index an uncalled founder allele. Scoring skips
neutral sites; traceback and switch counting preserve the canonical strict
comparison and first-argmax tie conventions at every site.

The site-major int8 dosage table costs O(L*K²) bytes and removes repeated
founder-pair lookup from O(N*L*K²) scans. Callers retain a direct bounded-memory
kernel when that table is too large. Up to 64 unordered states, one uint64
encodes traceback switches at a site; wider panels use the direct traceback.
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


@njit(cache=True, parallel=True, nogil=True)
def paint_dosages(dosages, logs, penalty, first, second, founders):
    samples, sites = logs.shape[:2]
    states = dosages.shape[1]
    answer = np.empty((samples, sites), np.int32)
    likelihood = np.empty(samples)
    center = math.log(1.0 / 3.0)
    for sample in prange(samples):
        scores = np.zeros(states)
        switched_from = np.empty(sites, np.int32)
        flags = np.empty(sites, np.uint64)
        for site in range(sites):
            p0, p1, p2 = (logs[sample, site, 0], logs[sample, site, 1], logs[sample, site, 2])
            neutral = p0 == 0.0 and p1 == 0.0 and (p2 == 0.0)
            previous = int(np.argmax(scores))
            switched_from[site] = previous
            switched = scores[previous] - penalty
            mask = np.uint64(0)
            for state in range(states):
                if scores[state] < switched:
                    mask |= np.uint64(1) << np.uint64(state)
                value = 0.0 if neutral else logs[sample, site, dosages[site, state]] - center
                scores[state] = max(scores[state], switched) + value
            flags[site] = mask
        state = int(np.argmax(scores))
        likelihood[sample] = scores[state]
        for site in range(sites - 1, -1, -1):
            answer[sample, site] = first[state] * founders + second[state]
            if flags[site] & np.uint64(1) << np.uint64(state):
                state = switched_from[site]
    return (answer, likelihood)

@njit(parallel=True, cache=True, nogil=True)
def count_switches(dosages, logs, penalty):
    """Canonical Viterbi score and switch count without a site traceback.

    Carry the count of the chosen predecessor with each state. Strict switch
    comparisons and first-argmax ties match paint_panel, including neutral
    sites. The initial scores are equal and the switch penalty is positive,
    so the first site never contributes a counted transition.
    """
    samples, sites, _ = logs.shape
    states = dosages.shape[1]
    likelihood = np.empty(samples)
    counts = np.empty(samples, np.int64)
    center = math.log(1.0 / 3.0)
    for sample in prange(samples):
        scores = np.zeros(states)
        switches = np.zeros(states, np.int64)
        for site in range(sites):
            neutral = logs[sample, site, 0] == 0.0 and logs[sample, site, 1] == 0.0 and (logs[sample, site, 2] == 0.0)
            previous = int(np.argmax(scores))
            switched = scores[previous] - penalty
            next_count = switches[previous] + 1
            for state in range(states):
                if scores[state] < switched:
                    switches[state] = next_count
                value = 0.0 if neutral else logs[sample, site, dosages[site, state]] - center
                scores[state] = max(scores[state], switched) + value
        state = int(np.argmax(scores))
        likelihood[sample], counts[sample] = (scores[state], switches[state])
    return (likelihood, counts)

@njit(cache=True, parallel=True)
def exchange_messages(dosages, logs, cuts, penalty):
    states = dosages.shape[1]
    samples, sites, _ = logs.shape
    forward = np.empty((samples, len(cuts), states), np.float64)
    backward = np.empty_like(forward)
    center = np.log(1.0 / 3.0)
    for sample in prange(samples):
        row = np.zeros(states)
        cut = 0
        for site in range(sites):
            if cut < len(cuts) and site == cuts[cut]:
                forward[sample, cut] = row
                cut += 1
            switched = np.max(row) - penalty
            for state in range(states):
                neutral = logs[sample, site, 0] == 0.0 and logs[sample, site, 1] == 0.0 and (logs[sample, site, 2] == 0.0)
                value = 0.0 if neutral else logs[sample, site, dosages[site, state]] - center
                row[state] = max(row[state], switched) + value
        row[:] = 0.0
        cut = len(cuts) - 1
        for site in range(sites - 1, -1, -1):
            switched = np.max(row) - penalty
            for state in range(states):
                neutral = logs[sample, site, 0] == 0.0 and logs[sample, site, 1] == 0.0 and (logs[sample, site, 2] == 0.0)
                value = 0.0 if neutral else logs[sample, site, dosages[site, state]] - center
                row[state] = max(row[state], switched) + value
            if cut >= 0 and site == cuts[cut]:
                backward[sample, cut] = row
                cut -= 1
    return (forward, backward)
