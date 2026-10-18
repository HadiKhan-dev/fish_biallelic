"""The refinement objective F and the max-plus site steps every kernel of this package uses.

F(X) = sum over samples s of F_s(X), F_s the full-site Viterbi score of an unordered-diplotype
Potts chain over the panel X (K founder paths of prepared local rows): states are the
Q = K(K+1)/2 unordered founder pairs, a site emits the robust centred log evidence of the pair's
dosage, every state change costs SWITCH_PENALTY, the start is free and there is no length
scaling. Only complete informative sites count: the component's keep flag set, every local row
of the prepared block called (a candidate-independent mask), and evidence that is not exactly
uniform. Unobserved sample calls and missing founder alleles are never imputed.

Robust emission of dosage d: max(log(0.99 p_d + 0.01/3), -2) - log(1/3) with p the normalised
genotype likelihoods (prepare_log_evidence stores the uncentred term, exactly 0 at skipped sites;
CENTER is subtracted as the site is read). Symmetric genotype emissions and one uniform switch
penalty make the unordered-pair reduction exact for this model (not for the pipeline's
homologue-specific ancestry HMMs).

Site steps. Forward (carried) messages switch, then emit: sw = max(row) - penalty;
row[q] = max(row[q], sw) + e(d_q). Backward (stored) messages emit, then switch. The
lexicographic variants carry switch counts and prefer, among exact score ties, fewer switches
(the coarse-move guard's genotype-fit term). Every kernel of the package is built on these
inline steps, so the forward recursion of a message kernel, its in-place update, the score and
the candidate kernels perform the same operations in the same order: their values are bitwise
equal, which the message updates' early stops and the proposal cache rely on.

Memory layout: a state's pair dosage haps[first[q], j] + haps[second[q], j] is the same for every
sample, so kernels read it from a site-major int8 table shared by all samples and read a site's
three emissions once per sample; every sample's chain is its own (samples run in parallel).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from ... import execution
from ...maths.emissions import CENTER, normalised_likelihoods, robust_log_likelihood

# Uniform cost of a diplotype state change at every level. Growing it with component length can
# prefer a poorer founder phase merely because it saves sample switches.
SWITCH_PENALTY = 20.0


@njit(inline="always")
def robust_log_evidence(p0, p1, p2):
    """Uncentred robust emissions of dosages 0, 1, 2; exactly (0, 0, 0) for uniform or empty
    evidence (such a site is skipped)."""
    informative, p0, p1, p2 = normalised_likelihoods(p0, p1, p2)
    if not informative:
        return 0.0, 0.0, 0.0
    return robust_log_likelihood(p0), robust_log_likelihood(p1), robust_log_likelihood(p2)


@njit(parallel=True, cache=True, nogil=True)
def prepare_log_evidence(evidence, complete):
    """float64 (S, L, 3) robust log evidence of the complete sites (zero elsewhere), once per
    component; evidence is the component's float32 (S, L, 3) neutral evidence."""
    result = np.zeros(evidence.shape, np.float64)
    for sample in prange(evidence.shape[0]):
        for site in range(evidence.shape[1]):
            if complete[site]:
                value = robust_log_evidence(evidence[sample, site, 0], evidence[sample, site, 1],
                                            evidence[sample, site, 2])
                for genotype in range(3):
                    result[sample, site, genotype] = value[genotype]
    return result


@njit(inline="always")
def unordered_pairs(founders):
    """The Q = K(K+1)/2 unordered states (first[q] <= second[q]) in upper-triangle order."""
    first = np.empty(founders * (founders + 1) // 2, np.int64)
    second = np.empty_like(first)
    state = 0
    for a in range(founders):
        for b in range(a, founders):
            first[state], second[state] = a, b
            state += 1
    return first, second


@njit(parallel=True, cache=True, nogil=True)
def selected_alleles(leaves, offsets, selected):
    """int8 (K, L) alleles of the panel `selected` (K x B prepared-row indices): founder k's
    alleles on block b are row selected[k, b] of leaves[b].

    One task per block, which reads its leaf from the typed List once: a List element read inside
    a parallel loop updates a reference count shared by all threads, and a row copy is too little
    work to hide it."""
    founders, blocks = selected.shape
    result = np.empty((founders, offsets[-1]), np.int8)
    for block_index in prange(blocks):
        block = np.int64(block_index)
        leaf = leaves[block]
        for founder in range(founders):
            result[founder, offsets[block]:offsets[block + 1]] = leaf[selected[founder, block]]
    return result


@njit(cache=True, parallel=True, nogil=True)
def pair_dosages(haps, first, second, lo, hi):
    """Site-major pair dosages of sites [lo, hi): out[j - lo, q] = haps[first[q], j] +
    haps[second[q], j] (int8)."""
    states = len(first)
    out = np.empty((hi - lo, states), np.int8)
    for j in prange(lo, hi):
        for q in range(states):
            out[j - lo, q] = haps[first[q], j] + haps[second[q], j]
    return out


# ------------------------------------------------------------------ inline site steps
@njit(inline="always")
def is_skipped(logs, s, j):
    """Site j carries no evidence for sample s (incomplete or uniform): it is skipped."""
    return logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0


@njit(inline="always")
def site_emissions(logs, s, j):
    """The centred emissions of dosages 0, 1, 2 of sample s at site j."""
    return logs[s, j, 0] - CENTER, logs[s, j, 1] - CENTER, logs[s, j, 2] - CENTER


@njit(inline="always")
def pick(d, e0, e1, e2):
    return e0 if d == 0 else (e1 if d == 1 else e2)


@njit(inline="always")
def row_max(row):
    """max(row) of a non-empty row (no NaN): eight interleaved running maxima, then combined. The
    maximum of the same values as a sequential scan, without its serial dependence chain (every
    message and score kernel takes one per informative site)."""
    n = len(row)
    m0 = row[0]
    m1 = m0
    m2 = m0
    m3 = m0
    m4 = m0
    m5 = m0
    m6 = m0
    m7 = m0
    q = 1
    while q + 8 <= n:
        if row[q] > m0:
            m0 = row[q]
        if row[q + 1] > m1:
            m1 = row[q + 1]
        if row[q + 2] > m2:
            m2 = row[q + 2]
        if row[q + 3] > m3:
            m3 = row[q + 3]
        if row[q + 4] > m4:
            m4 = row[q + 4]
        if row[q + 5] > m5:
            m5 = row[q + 5]
        if row[q + 6] > m6:
            m6 = row[q + 6]
        if row[q + 7] > m7:
            m7 = row[q + 7]
        q += 8
    while q < n:
        if row[q] > m0:
            m0 = row[q]
        q += 1
    if m1 > m0:
        m0 = m1
    if m2 > m0:
        m0 = m2
    if m3 > m0:
        m0 = m3
    if m4 > m0:
        m0 = m4
    if m5 > m0:
        m0 = m5
    if m6 > m0:
        m0 = m6
    if m7 > m0:
        m0 = m7
    return m0


@njit(inline="always")
def switch_then_emit(row, sw, dos, i, e0, e1, e2):
    """row[q] = max(row[q], sw) + e(dos[i, q]) for every state (the forward site update given the
    switch floor sw = max(row) - penalty)."""
    for q in range(len(row)):
        row[q] = max(row[q], sw) + pick(dos[i, q], e0, e1, e2)


@njit(inline="always")
def emit_then_switch(row, dos, i, e0, e1, e2, penalty):
    """The backward (stored-convention) site update: emit, then allow one switch."""
    for q in range(len(row)):
        row[q] += pick(dos[i, q], e0, e1, e2)
    sw = row_max(row) - penalty
    for q in range(len(row)):
        row[q] = max(row[q], sw)


@njit(inline="always")
def forward_sites(row, dos, lo, logs, s, j0, j1, penalty):
    """Carry sample s's forward message row over sites [j0, j1) (dosage rows dos[j - lo])."""
    for j in range(j0, j1):
        if is_skipped(logs, s, j):
            continue
        e0, e1, e2 = site_emissions(logs, s, j)
        switch_then_emit(row, row_max(row) - penalty, dos, j - lo, e0, e1, e2)


@njit(inline="always")
def backward_sites(row, dos, lo, logs, s, j0, j1, penalty):
    """Carry sample s's backward message row over sites [j0, j1) from right to left."""
    for j in range(j1 - 1, j0 - 1, -1):
        if is_skipped(logs, s, j):
            continue
        e0, e1, e2 = site_emissions(logs, s, j)
        emit_then_switch(row, dos, j - lo, e0, e1, e2, penalty)


@njit(inline="always")
def lex_best(row, cnt):
    """Maximum score and the fewest switches among exact maxima (a sequential scan: interleaved
    running lexicographic maxima measured slower at every K from 6 to 30, unlike row_max)."""
    m = row[0]
    c = cnt[0]
    for q in range(1, len(row)):
        if row[q] > m or (row[q] == m and cnt[q] < c):
            m = row[q]
            c = cnt[q]
    return m, c


@njit(inline="always")
def lex_switch(row, cnt, q, sw, c):
    """Take the switch into state q when it scores more, or ties with fewer switches; returns
    whether it was taken."""
    if sw > row[q] or (sw == row[q] and c + 1 < cnt[q]):
        row[q] = sw
        cnt[q] = c + 1
        return True
    return False


@njit(inline="always")
def switch_then_emit_lex(row, cnt, sw, c, dos, i, e0, e1, e2):
    """switch_then_emit with switch counts (sw, c: the lexicographic maximum less the penalty)."""
    for q in range(len(row)):
        lex_switch(row, cnt, q, sw, c)
        row[q] += pick(dos[i, q], e0, e1, e2)


@njit(inline="always")
def emit_then_switch_lex(row, cnt, dos, i, e0, e1, e2, penalty):
    for q in range(len(row)):
        row[q] += pick(dos[i, q], e0, e1, e2)
    m, c = lex_best(row, cnt)
    sw = m - penalty
    for q in range(len(row)):
        lex_switch(row, cnt, q, sw, c)


@njit(inline="always")
def forward_sites_lex(row, cnt, dos, lo, logs, s, j0, j1, penalty):
    for j in range(j0, j1):
        if is_skipped(logs, s, j):
            continue
        e0, e1, e2 = site_emissions(logs, s, j)
        m, c = lex_best(row, cnt)
        switch_then_emit_lex(row, cnt, m - penalty, c, dos, j - lo, e0, e1, e2)


@njit(inline="always")
def backward_sites_lex(row, cnt, dos, lo, logs, s, j0, j1, penalty):
    for j in range(j1 - 1, j0 - 1, -1):
        if is_skipped(logs, s, j):
            continue
        e0, e1, e2 = site_emissions(logs, s, j)
        emit_then_switch_lex(row, cnt, dos, j - lo, e0, e1, e2, penalty)


@njit(inline="always")
def normalise(row):
    """Subtract the maximum (returned) so a block-boundary message has maximum 0; x - x = +0, so
    normalised messages hold no negative zero and equal values are equal bits."""
    m = row_max(row)
    for q in range(len(row)):
        row[q] -= m
    return m


# ------------------------------------------------------------------ per-sample scores
@njit(cache=True, parallel=True, nogil=True)
def score_dosages(dosages, logs, penalty):
    """Per-sample F_s from the panel's site-major pair-dosage table (L x Q int8)."""
    samples, sites = logs.shape[:2]
    answer = np.empty(samples)
    for s in prange(samples):
        row = np.zeros(dosages.shape[1])
        forward_sites(row, dosages, 0, logs, s, 0, sites, penalty)
        answer[s] = row_max(row)
    return answer


@njit(cache=True, parallel=True, nogil=True)
def score_direct(haps, logs, penalty):
    """Per-sample F_s without a dosage table (O(threads x Q) memory; the same arithmetic)."""
    samples, sites = logs.shape[:2]
    first, second = unordered_pairs(len(haps))
    answer = np.empty(samples)
    for s in prange(samples):
        row = np.zeros(len(first))
        for j in range(sites):
            if is_skipped(logs, s, j):
                continue
            e0, e1, e2 = site_emissions(logs, s, j)
            sw = row_max(row) - penalty
            for q in range(len(first)):
                row[q] = max(row[q], sw) + pick(haps[first[q], j] + haps[second[q], j], e0, e1, e2)
        answer[s] = row_max(row)
    return answer


def score_panel(haps, logs, penalty, ranges, table_bytes):
    """Per-sample F_s of the panel with alleles haps (int8 K x L) on the log evidence logs.

    ranges: consecutive sample ranges (s0, s1, threads) covering every sample
    (workspace.sample_ranges), each scored by its own call at its thread count (a sample's score
    is its own recursion). The shared pair-dosage table (L x Q int8) is built when it fits
    table_bytes, else the direct kernel runs; both give the same values."""
    founders, sites = haps.shape
    states = founders * (founders + 1) // 2
    answer = np.empty(len(logs))
    if sites * states > table_bytes:
        for s0, s1, threads in ranges:
            with execution.numba_threads(threads):
                answer[s0:s1] = score_direct(haps, logs[s0:s1], float(penalty))
        return answer
    first, second = unordered_pairs(founders)
    dosages = pair_dosages(haps, first, second, 0, sites)
    for s0, s1, threads in ranges:
        with execution.numba_threads(threads):
            answer[s0:s1] = score_dosages(dosages, logs[s0:s1], float(penalty))
    return answer
