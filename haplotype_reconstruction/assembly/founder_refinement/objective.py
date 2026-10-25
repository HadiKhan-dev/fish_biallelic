"""The refinement objective F and the max-plus site steps every kernel of this package uses.

Release 2.8: F(X) = sum over samples s of F_s(X), F_s a robust full-site Viterbi (max-path) score of
the sample's diplotype under a per-homologue founder-switch model on the panel X (log-probability-like:
the emissions below are clipped and centred, so F_s is not a literal log probability)
(K founder paths of prepared local rows). Each homologue follows its own K-state founder chain
with transition T = (1 - rho) I + rho / K between consecutive informative sites of the sample,
rho the map-derived reset probability of the gap (workspace.map_hazards: physical-edge hazards
reset_rate x Morgans composed over the sites the sample skips). Relative to staying, a homologue
that changes founder pays the cost c = log(1 - rho (1 - 1/K)) - log(rho / K) >= 0; the
panel-independent constants (start -2 log K, per site 2 log(1 - rho (1 - 1/K))) are the sample's
normaliser, added where scores of different K are compared (SweepEngine.per_sample). Both
homologues share rho, so the ordered K x K chain stays symmetric and reduces exactly to the
Q = K(K+1)/2 unordered pairs:

    U'{a, b} = max(U{a, b}, max(R[a], R[b]) - c, G - 2c) + e(d_ab),
    R[x] = max_y U{x, y}, G = max U,

O(Q + K) per site. The sites are the component's kept sites with at least one called local row of
the prepared block (a candidate-independent mask, workspace.kept) where the sample's evidence is not
exactly uniform. A pair's emission is that of its pair code (pair_code7): the dosage when both rows
are called; otherwise the predictive class of the unknown allele(s) under a uniform allele prior
(maths.emissions), one latent allele when both homologues copy the same uncalled row. Unobserved
sample calls and missing founder alleles are never imputed into the panel.

Robust emission of dosage d: max(log(0.99 p_d + 0.01/3), -2) - log(1/3) with p the normalised
genotype likelihoods (prepare_log_evidence stores the uncentred term, exactly 0 at skipped sites;
CENTER is subtracted as the site is read).

Site steps. Forward (carried) messages take the transition entering the site, then emit; backward
(stored) messages emit, then take the transition entering the site. A sample's first informative
site has no entering transition (cost +inf: the step leaves the row unchanged). The lexicographic
variants carry the switch cost of the maximum path and prefer, among exact score ties, the lower
switch cost (the coarse-move guard's genotype-fit term E = F + switch cost). Every kernel of the
package is built on these inline steps, so the forward recursion of a message kernel, its in-place
update, the score and the candidate kernels perform the same operations in the same order: their
values are bitwise equal, which the message updates' early stops and the proposal cache rely on.

Memory layout: a state's pair dosage haps[first[q], j] + haps[second[q], j] is the same for every
sample, so kernels read it from a site-major int8 table shared by all samples and read a site's
three emissions once per sample; every sample's chain is its own (samples run in parallel).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from ... import execution
from ...maths.emissions import CENTER, normalised_likelihoods, predictive_likelihood, robust_log_likelihood

# The binned surrogate's (proposals only) uniform cost of a diplotype state change. The exact
# objective uses the map-derived per-homologue costs instead.
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
def prepare_log_evidence(evidence, kept):
    """float64 (S, L, 7) uncentred robust log evidence of the pair codes (objective.pair_code7) at the
    objective's sites (workspace.kept: kept sites with at least one called prepared row; zero elsewhere and
    for uniform evidence), once per component; evidence is the component's float32 (S, L, 3)
    neutral evidence. Codes 3-6 are the predictive classes of unknown alleles (maths.emissions)."""
    result = np.zeros((evidence.shape[0], evidence.shape[1], 7), np.float64)
    for sample in prange(evidence.shape[0]):
        for site in range(evidence.shape[1]):
            if kept[site]:
                informative, p0, p1, p2 = normalised_likelihoods(
                    evidence[sample, site, 0], evidence[sample, site, 1], evidence[sample, site, 2])
                if not informative:
                    continue
                for code in range(7):
                    result[sample, site, code] = robust_log_likelihood(
                        predictive_likelihood(p0, p1, p2, code))
    return result


@njit(inline="always")
def pair_code7(x, y):
    """Pair code of two coded alleles (called allele 0/1, or -2 - prepared row when uncalled): dosage
    0-2 when both are called, 3 + the known allele when one is, else 6 when both copy the same
    prepared row (one latent allele) and 5 when they do not (maths.emissions.pair_code)."""
    if x >= 0 and y >= 0:
        return x + y
    if x >= 0:
        return 3 + x
    if y >= 0:
        return 3 + y
    return 6 if x == y else 5


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


@njit(inline="always")
def pair_founders(states):
    """K of a row of Q = K(K+1)/2 unordered states."""
    founders = 1
    while founders * (founders + 1) // 2 < states:
        founders += 1
    return founders


@njit(parallel=True, cache=True, nogil=True)
def selected_alleles(leaves, offsets, selected):
    """int8 (K, L) alleles of the panel `selected` (K x B prepared-row indices): founder k's
    alleles on block b are row selected[k, b] of leaves[b].

    One task per block, which reads its leaf from the typed List once: a List element read inside
    a parallel loop updates a reference count shared by all threads, and a row copy is too little
    work to hide it."""
    founders, blocks = selected.shape
    result = np.empty((founders, offsets[-1]), np.int16)
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
            out[j - lo, q] = pair_code7(haps[first[q], j], haps[second[q], j])
    return out


# ------------------------------------------------------------------ inline site steps
@njit(inline="always")
def is_skipped(logs, s, j):
    """Site j carries no evidence for sample s (not an objective site, or uniform): it is skipped."""
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
def transition(row, c, R):
    """One site's per-homologue transition on unordered states, in place (module docstring):
    row{a, b} = max(row{a, b}, max(R[a], R[b]) - c, G - 2c). c = +inf (no preceding informative
    site) leaves the row unchanged. R: float64 (K,) scratch (K = len(R)). O(Q + K).

    The founder maxima come from contiguous row segments (states {a, b >= a}) and their
    columns, with branch-free max selects (bitwise the plain scan's values, 1.6x-2.2x
    faster at K = 6-30)."""
    if not c < np.inf:
        return
    K = len(R)
    for x in range(K):
        R[x] = -np.inf
    q = 0
    for a in range(K):
        m = R[a]
        for t in range(K - a):
            v = row[q + t]
            m = max(m, v)
            R[a + t] = max(R[a + t], v)
        R[a] = m
        q += K - a
    G = R[0]
    for x in range(1, K):
        G = max(G, R[x])
    both = G - 2.0 * c
    q = 0
    for a in range(K):
        ra = R[a]
        for t in range(K - a):
            row[q + t] = max(max(row[q + t], max(ra, R[a + t]) - c), both)
        q += K - a


@njit(inline="always")
def lex_better(v, k, w, l):
    """(v, k) beats (w, l) lexicographically: higher score, then lower switch cost."""
    return v > w or (v == w and k < l)


@njit(inline="always")
def transition_lex(row, cst, c, R, Rc):
    """transition() on the lexicographic pair (score, switch cost of the maximum path): among exact
    score ties the lower switch cost; a one-homologue change adds c, a change of both 2c (the
    stay route wins exact pair ties, then the first founder's route). R, Rc: float64 (K,).
    Segment/column layout with selects (bitwise the plain scan's values)."""
    if not c < np.inf:
        return
    K = len(R)
    for x in range(K):
        R[x] = -np.inf
        Rc[x] = 0.0
    q = 0
    for a in range(K):
        m = R[a]
        mc = Rc[a]
        for t in range(K - a):
            v = row[q + t]
            k = cst[q + t]
            take = v > m or (v == m and k < mc)
            m = v if take else m
            mc = k if take else mc
            r = R[a + t]
            rc = Rc[a + t]
            take = v > r or (v == r and k < rc)
            R[a + t] = v if take else r
            Rc[a + t] = k if take else rc
        R[a] = m
        Rc[a] = mc
        q += K - a
    G = R[0]
    Gc = Rc[0]
    for x in range(1, K):
        take = R[x] > G or (R[x] == G and Rc[x] < Gc)
        G = R[x] if take else G
        Gc = Rc[x] if take else Gc
    both = G - 2.0 * c
    both_c = Gc + 2.0 * c
    q = 0
    for a in range(K):
        ra = R[a]
        rac = Rc[a]
        for t in range(K - a):
            rb = R[a + t]
            rbc = Rc[a + t]
            useb = rb > ra or (rb == ra and rbc < rac)
            w = (rb if useb else ra) - c
            wc = (rbc if useb else rac) + c
            v = row[q + t]
            k = cst[q + t]
            take = w > v or (w == v and wc < k)
            v = w if take else v
            k = wc if take else k
            take = both > v or (both == v and both_c < k)
            row[q + t] = both if take else v
            cst[q + t] = both_c if take else k
        q += K - a


@njit(inline="always")
def transition_count(row, cnt, c, R, Rc):
    """transition_lex() with integer homologue-switch counts as the secondary key (the fit tie
    prior: fewest exact-optimal switches). Rc: int64 (K,)."""
    if not c < np.inf:
        return
    K = len(R)
    G = -np.inf
    Gc = 0
    for x in range(K):
        R[x] = -np.inf
        Rc[x] = 0
    q = 0
    for a in range(K):
        for b in range(a, K):
            v = row[q]
            k = cnt[q]
            if v > G or (v == G and k < Gc):
                G = v
                Gc = k
            if v > R[a] or (v == R[a] and k < Rc[a]):
                R[a] = v
                Rc[a] = k
            if v > R[b] or (v == R[b] and k < Rc[b]):
                R[b] = v
                Rc[b] = k
            q += 1
    both = G - 2.0 * c
    q = 0
    for a in range(K):
        for b in range(a, K):
            v = row[q]
            k = cnt[q]
            if R[b] > R[a] or (R[b] == R[a] and Rc[b] < Rc[a]):
                w, wc = R[b] - c, Rc[b] + 1
            else:
                w, wc = R[a] - c, Rc[a] + 1
            if w > v or (w == v and wc < k):
                v, k = w, wc
            if both > v or (both == v and Gc + 2 < k):
                v, k = both, Gc + 2
            row[q] = v
            cnt[q] = k
            q += 1


@njit(inline="always")
def emit(row, dos, i, logs, s, j):
    """row[q] += the centred emission of pair code dos[i, q] of sample s at site j, every state."""
    for q in range(len(row)):
        row[q] += logs[s, j, dos[i, q]] - CENTER


@njit(inline="always")
def forward_sites(row, dos, lo, logs, s, j0, j1, costs, R):
    """Carry sample s's forward message row over sites [j0, j1): per informative site the
    transition entering it (costs[s, j]) then its emission (dosage rows dos[j - lo]). R: (K,)
    scratch."""
    for j in range(j0, j1):
        if is_skipped(logs, s, j):
            continue
        transition(row, costs[s, j], R)
        emit(row, dos, j - lo, logs, s, j)


@njit(inline="always")
def backward_sites(row, dos, lo, logs, s, j0, j1, costs, R):
    """Carry sample s's backward (stored-convention) row over sites [j0, j1) from right to left:
    per informative site its emission, then the transition entering it."""
    for j in range(j1 - 1, j0 - 1, -1):
        if is_skipped(logs, s, j):
            continue
        emit(row, dos, j - lo, logs, s, j)
        transition(row, costs[s, j], R)


@njit(inline="always")
def forward_sites_lex(row, cst, dos, lo, logs, s, j0, j1, costs, R, Rc):
    """forward_sites() with the lexicographic switch costs."""
    for j in range(j0, j1):
        if is_skipped(logs, s, j):
            continue
        transition_lex(row, cst, costs[s, j], R, Rc)
        emit(row, dos, j - lo, logs, s, j)


@njit(inline="always")
def backward_sites_lex(row, cst, dos, lo, logs, s, j0, j1, costs, R, Rc):
    """backward_sites() with the lexicographic switch costs."""
    for j in range(j1 - 1, j0 - 1, -1):
        if is_skipped(logs, s, j):
            continue
        emit(row, dos, j - lo, logs, s, j)
        transition_lex(row, cst, costs[s, j], R, Rc)


@njit(inline="always")
def lex_best(row, cst):
    """Maximum score and the lowest switch cost among exact maxima (a sequential scan)."""
    m = row[0]
    c = cst[0]
    for q in range(1, len(row)):
        if row[q] > m or (row[q] == m and cst[q] < c):
            m = row[q]
            c = cst[q]
    return m, c


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
def score_dosages(dosages, logs, costs):
    """Per-sample max-path F_s (without the panel-independent normalisers) from the panel's
    site-major pair-dosage table (L x Q int8); costs (S, L): per-homologue switch costs."""
    samples, sites = logs.shape[:2]
    founders = pair_founders(dosages.shape[1])
    answer = np.empty(samples)
    for s in prange(samples):
        row = np.zeros(dosages.shape[1])
        R = np.empty(founders)
        forward_sites(row, dosages, 0, logs, s, 0, sites, costs, R)
        answer[s] = row_max(row)
    return answer


@njit(cache=True, parallel=True, nogil=True)
def score_direct(haps, logs, costs):
    """Per-sample F_s without a dosage table (O(threads x Q) memory; the same arithmetic)."""
    samples, sites = logs.shape[:2]
    first, second = unordered_pairs(len(haps))
    answer = np.empty(samples)
    for s in prange(samples):
        row = np.zeros(len(first))
        R = np.empty(len(haps))
        for j in range(sites):
            if is_skipped(logs, s, j):
                continue
            transition(row, costs[s, j], R)
            for q in range(len(first)):
                row[q] += logs[s, j, pair_code7(haps[first[q], j], haps[second[q], j])] - CENTER
        answer[s] = row_max(row)
    return answer


@njit(cache=True, parallel=True, nogil=True)
def score_switches_direct(haps, logs, costs):
    """Exact score and the fewest homologue switches among its exact maximisers, per sample."""
    samples, sites = logs.shape[:2]
    first, second = unordered_pairs(len(haps))
    scores = np.empty(samples)
    switches = np.empty(samples, np.int64)
    for s in prange(samples):
        row = np.zeros(len(first))
        counts = np.zeros(len(first), np.int64)
        R = np.empty(len(haps))
        Rc = np.empty(len(haps), np.int64)
        for j in range(sites):
            if is_skipped(logs, s, j):
                continue
            transition_count(row, counts, costs[s, j], R, Rc)
            for q in range(len(first)):
                row[q] += logs[s, j, pair_code7(haps[first[q], j], haps[second[q], j])] - CENTER
        m = row[0]
        k = counts[0]
        for q in range(1, len(row)):
            if row[q] > m or (row[q] == m and counts[q] < k):
                m = row[q]
                k = counts[q]
        scores[s] = m
        switches[s] = k
    return scores, switches


def score_panel(haps, logs, costs, ranges, table_bytes):
    """Per-sample F_s of the panel with alleles haps (int8 K x L) on the log evidence logs and
    switch costs costs (S x L, the panel's K).

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
                answer[s0:s1] = score_direct(haps, logs[s0:s1], costs[s0:s1])
        return answer
    first, second = unordered_pairs(founders)
    dosages = pair_dosages(haps, first, second, 0, sites)
    for s0, s1, threads in ranges:
        with execution.numba_threads(threads):
            answer[s0:s1] = score_dosages(dosages, logs[s0:s1], costs[s0:s1])
    return answer
