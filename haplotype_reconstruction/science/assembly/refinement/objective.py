"""The refinement objective F and the max-plus site steps every kernel of this package uses.

Owns unordered diplotype recurrences and exact panel scores.
Must not choose K, read storage or configure runtime.
Method: docs/methods/founder_refinement.md#objective (RS01; RC02).
STYLE_40 exception: numerical kernels retain flat scalar/array arguments because
Numba cannot lower the frozen Python boundary records; the stage uses those records.
Hot recurrences remain whole to preserve their loop/update order and avoid temporary
materialisations. Parallel kernels write disjoint sample slices; sample reductions
are fixed serial folds, independent of the explicit stage team's size.


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
genotype likelihoods (window.site_values gives the uncentred term; CENTER is subtracted as the site
is read).

Release 3.4: kernels read the log evidence and switch costs of a sample-site through a window
(window.py: a stretch of sites held in memory, anything else computed on the spot with the same
arithmetic), never through whole-component arrays; a whole-component kernel runs stretch by stretch,
covering each before it reads it. The values, and so every kernel's result, are those of 3.3's
arrays bit for bit.

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
from numba import prange

from haplotype_reconstruction.core.kernels import kernel

from .emissions import CENTER
from .overloads import overloads_for
from .window import capacity, cover, locate, site_cost, site_logs, sites_of

# The binned surrogate's (proposals only) uniform cost of a diplotype state change. The exact
# objective uses the map-derived per-homologue costs instead.
SWITCH_PENALTY = 20.0


@kernel(overloads=overloads_for(__name__, "pair_code7"), inline="always")
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


@kernel(overloads=overloads_for(__name__, "unordered_pairs"), inline="always")
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


@kernel(overloads=overloads_for(__name__, "pair_founders"), inline="always")
def pair_founders(states):
    """K of a row of Q = K(K+1)/2 unordered states."""
    founders = 1
    while founders * (founders + 1) // 2 < states:
        founders += 1
    return founders


@kernel(overloads=overloads_for(__name__, "selected_alleles"), parallel=True, cache=True, nogil=True)
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
            result[founder, offsets[block] : offsets[block + 1]] = leaf[selected[founder, block]]
    return result


@kernel(overloads=overloads_for(__name__, "pair_dosages"), cache=True, parallel=True, nogil=True)
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
@kernel(overloads=overloads_for(__name__, "row_max"), inline="always")
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


@kernel(overloads=overloads_for(__name__, "transition"), inline="always")
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


@kernel(overloads=overloads_for(__name__, "lex_better"), inline="always")
def lex_better(v, k, w, l):
    """(v, k) beats (w, l) lexicographically: higher score, then lower switch cost."""
    return v > w or (v == w and k < l)


@kernel(overloads=overloads_for(__name__, "transition_lex"), inline="always")
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


@kernel(overloads=overloads_for(__name__, "transition_count"), inline="always")
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


@kernel(overloads=overloads_for(__name__, "emit"), inline="always")
def emit(row, dos, i, e):
    """row[q] += the centred emission of pair code dos[i, q] (e: the site's uncentred log evidence of
    the seven codes, window.site_logs), every state."""
    for q in range(len(row)):
        row[q] += e[dos[i, q]] - CENTER


@kernel(overloads=overloads_for(__name__, "founder_maxima"), inline="always")
def founder_maxima(row, R):
    """R[x] = the maximum of row over the states containing founder x: transition()'s scan (row
    segments {a, b >= a} and their columns, in that order)."""
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


@kernel(overloads=overloads_for(__name__, "centred"), inline="always")
def centred(e, ec):
    """ec[code] = e[code] - CENTER, the subtraction emit() makes per state, once per site."""
    for x in range(7):
        ec[x] = e[x] - CENTER


@kernel(overloads=overloads_for(__name__, "fused_step"), inline="always")
def fused_step(row, c, R, Rn, ec, dos, i, scanned):
    """Release 3.5: transition(row, c, R) (c finite; R the founder maxima of row when scanned, else
    computed here) then the emission ec[dos[i, q]], in one pass over the states that also collects
    the emitted row's founder maxima into Rn in founder_maxima()'s order. Every state takes the
    same operations in the same order as transition() then emit(), and Rn's maxima are taken over
    the same values in the same order, so rows and maxima are bitwise theirs (about 25% fewer
    cycles per state: the separate scan pass of the next transition is gone)."""
    K = len(R)
    if not scanned:
        founder_maxima(row, R)
    G = R[0]
    for x in range(1, K):
        G = max(G, R[x])
    both = G - 2.0 * c
    for x in range(K):
        Rn[x] = -np.inf
    q = 0
    for a in range(K):
        ra = R[a]
        m = Rn[a]
        for t in range(K - a):
            v = max(max(row[q + t], max(ra, R[a + t]) - c), both) + ec[dos[i, q + t]]
            row[q + t] = v
            m = max(m, v)
            Rn[a + t] = max(Rn[a + t], v)
        Rn[a] = m
        q += K - a


@kernel(overloads=overloads_for(__name__, "emit_scan"), inline="always")
def emit_scan(row, Rn, ec, dos, i):
    """emit() with centred evidence ec, collecting the emitted row's founder maxima into Rn
    (founder_maxima()'s order)."""
    K = len(Rn)
    for x in range(K):
        Rn[x] = -np.inf
    q = 0
    for a in range(K):
        m = Rn[a]
        for t in range(K - a):
            v = row[q + t] + ec[dos[i, q + t]]
            row[q + t] = v
            m = max(m, v)
            Rn[a + t] = max(Rn[a + t], v)
        Rn[a] = m
        q += K - a


@kernel(overloads=overloads_for(__name__, "finish_transition"), inline="always")
def finish_transition(row, c, R):
    """transition() given R = the founder maxima of row (c finite)."""
    K = len(R)
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


@kernel(overloads=overloads_for(__name__, "forward_sites"), inline="always")
def forward_sites(row, dos, lo, win, s, j0, j1, R, e):
    """Carry sample s's forward message row over sites [j0, j1) of the window's orientation: per
    informative site the transition entering it (its switch cost) then its emission (dosage rows
    dos[j - lo]). R: (K,) and e: (7,) scratch.

    Release 3.5: each site's transition and emission run as one pass (fused_step) that also takes
    the founder maxima the next site's transition reads; bitwise transition() then emit()."""
    K = len(R)
    Rn = np.empty(K)
    ec = np.empty(7)
    scanned = False
    for j in range(j0, j1):
        f, k, r = locate(win, s, j)
        if not site_logs(win, s, f, k, r, e):
            continue
        centred(e, ec)
        c = site_cost(win, s, f, k, r)
        if c < np.inf:
            fused_step(row, c, R, Rn, ec, dos, j - lo, scanned)
        else:
            emit_scan(row, Rn, ec, dos, j - lo)
        for x in range(K):
            R[x] = Rn[x]
        scanned = True


@kernel(overloads=overloads_for(__name__, "backward_sites"), inline="always")
def backward_sites(row, dos, lo, win, s, j0, j1, R, e):
    """Carry sample s's backward (stored-convention) row over sites [j0, j1) from right to left:
    per informative site its emission, then the transition entering it.

    Release 3.5: a site's transition is completed in the same pass as the next site's emission
    (fused_step), which also takes that site's founder maxima; the last transition is completed
    alone. Bitwise emit() then transition() per site."""
    K = len(R)
    Rn = np.empty(K)
    ec = np.empty(7)
    owed = False
    owed_c = 0.0
    for j in range(j1 - 1, j0 - 1, -1):
        f, k, r = locate(win, s, j)
        if not site_logs(win, s, f, k, r, e):
            continue
        centred(e, ec)
        if owed:
            fused_step(row, owed_c, R, Rn, ec, dos, j - lo, True)
        else:
            emit_scan(row, Rn, ec, dos, j - lo)
        for x in range(K):
            R[x] = Rn[x]
        c = site_cost(win, s, f, k, r)
        owed = c < np.inf
        owed_c = c
    if owed:
        finish_transition(row, owed_c, R)


@kernel(overloads=overloads_for(__name__, "lex_founder_maxima"), inline="always")
def lex_founder_maxima(row, cst, R, Rc):
    """(R[x], Rc[x]) = the lexicographic maximum (score, then lower switch cost) of the states containing
    founder x: transition_lex()'s first scan, with its comparisons in its order."""
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


@kernel(overloads=overloads_for(__name__, "lex_fused_step"), inline="always")
def lex_fused_step(row, cst, c, R, Rc, Rn, Rcn, ec, dos, i, emit_here):
    """Release 3.5: the rest of transition_lex(row, cst, c, R, Rc) given (R, Rc) = lex_founder_maxima(row, cst)
    (c finite), then, when emit_here, emit()'s addition ec[dos[i, q]] (ec = the site's e[code] - CENTER), in one
    pass over the states that also collects the resulting row's founder maxima into (Rn, Rcn) with
    lex_founder_maxima()'s comparisons in its order. Every state takes transition_lex()'s operations, then emit()'s,
    so rows, switch costs and maxima are bitwise theirs; only the separate scan of the next transition is gone."""
    K = len(R)
    G = R[0]
    Gc = Rc[0]
    for x in range(1, K):
        take = R[x] > G or (R[x] == G and Rc[x] < Gc)
        G = R[x] if take else G
        Gc = Rc[x] if take else Gc
    both = G - 2.0 * c
    both_c = Gc + 2.0 * c
    for x in range(K):
        Rn[x] = -np.inf
        Rcn[x] = 0.0
    q = 0
    for a in range(K):
        ra = R[a]
        rac = Rc[a]
        m = Rn[a]
        mc = Rcn[a]
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
            v = both if take else v
            k = both_c if take else k
            if emit_here:
                v += ec[dos[i, q + t]]
            row[q + t] = v
            cst[q + t] = k
            take = v > m or (v == m and k < mc)
            m = v if take else m
            mc = k if take else mc
            r = Rn[a + t]
            rc = Rcn[a + t]
            take = v > r or (v == r and k < rc)
            Rn[a + t] = v if take else r
            Rcn[a + t] = k if take else rc
        Rn[a] = m
        Rcn[a] = mc
        q += K - a


@kernel(overloads=overloads_for(__name__, "lex_emit_scan"), inline="always")
def lex_emit_scan(row, cst, Rn, Rcn, ec, dos, i):
    """emit() with centred evidence ec, collecting the emitted row's founder maxima into (Rn, Rcn)
    (lex_founder_maxima()'s comparisons in its order)."""
    K = len(Rn)
    for x in range(K):
        Rn[x] = -np.inf
        Rcn[x] = 0.0
    q = 0
    for a in range(K):
        m = Rn[a]
        mc = Rcn[a]
        for t in range(K - a):
            v = row[q + t] + ec[dos[i, q + t]]
            k = cst[q + t]
            row[q + t] = v
            take = v > m or (v == m and k < mc)
            m = v if take else m
            mc = k if take else mc
            r = Rn[a + t]
            rc = Rcn[a + t]
            take = v > r or (v == r and k < rc)
            Rn[a + t] = v if take else r
            Rcn[a + t] = k if take else rc
        Rn[a] = m
        Rcn[a] = mc
        q += K - a


@kernel(overloads=overloads_for(__name__, "forward_sites_lex"), inline="always")
def forward_sites_lex(row, cst, dos, lo, win, s, j0, j1, R, Rc, e):
    """forward_sites() with the lexicographic switch costs.

    Release 3.5: each site's transition and emission run as one pass (lex_fused_step) that also takes the
    founder maxima the next site's transition reads; bitwise transition_lex() then emit()."""
    K = len(R)
    Rn = np.empty(K)
    Rcn = np.empty(K)
    ec = np.empty(7)
    scanned = False
    for j in range(j0, j1):
        f, k, r = locate(win, s, j)
        if not site_logs(win, s, f, k, r, e):
            continue
        for x in range(7):
            ec[x] = e[x] - CENTER
        c = site_cost(win, s, f, k, r)
        if c < np.inf:
            if not scanned:
                lex_founder_maxima(row, cst, R, Rc)
            lex_fused_step(row, cst, c, R, Rc, Rn, Rcn, ec, dos, j - lo, True)
        else:
            lex_emit_scan(row, cst, Rn, Rcn, ec, dos, j - lo)
        for x in range(K):
            R[x] = Rn[x]
            Rc[x] = Rcn[x]
        scanned = True


@kernel(overloads=overloads_for(__name__, "backward_sites_lex"), inline="always")
def backward_sites_lex(row, cst, dos, lo, win, s, j0, j1, R, Rc, e):
    """backward_sites() with the lexicographic switch costs.

    Release 3.5: a site's transition is completed in the same pass as the next site's emission
    (lex_fused_step), which also takes that site's founder maxima; the last transition is completed alone.
    Bitwise emit() then transition_lex() per site."""
    K = len(R)
    Rn = np.empty(K)
    Rcn = np.empty(K)
    ec = np.empty(7)
    owed = False
    owed_c = 0.0
    for j in range(j1 - 1, j0 - 1, -1):
        f, k, r = locate(win, s, j)
        if not site_logs(win, s, f, k, r, e):
            continue
        for x in range(7):
            ec[x] = e[x] - CENTER
        if owed:
            lex_fused_step(row, cst, owed_c, R, Rc, Rn, Rcn, ec, dos, j - lo, True)
        else:
            lex_emit_scan(row, cst, Rn, Rcn, ec, dos, j - lo)
        for x in range(K):
            R[x] = Rn[x]
            Rc[x] = Rcn[x]
        c = site_cost(win, s, f, k, r)
        owed = c < np.inf
        owed_c = c
    if owed:
        lex_fused_step(row, cst, owed_c, R, Rc, Rn, Rcn, ec, dos, 0, False)


@kernel(overloads=overloads_for(__name__, "normalise"), inline="always")
def normalise(row):
    """Subtract the maximum (returned) so a block-boundary message has maximum 0; x - x = +0, so
    normalised messages hold no negative zero and equal values are equal bits."""
    m = row_max(row)
    for q in range(len(row)):
        row[q] -= m
    return m


# ------------------------------------------------------------------ per-sample scores
# The whole-component score kernels run stretch by stretch (at most the window's capacity of sites, the
# whole component for a resident window), each sample's row carried from stretch to stretch: the same
# site steps in the same order as one pass.
@kernel(overloads=overloads_for(__name__, "score_dosages"), cache=True, parallel=True, nogil=True)
def score_dosages(dosages, win, s0, s1):
    """Per-sample max-path F_s (without the panel-independent normalisers) of samples [s0, s1) from the
    panel's site-major pair-dosage table (L x Q int8); the window gives the log evidence and the
    per-homologue switch costs of the panel's K."""
    sites = sites_of(win)
    states = dosages.shape[1]
    founders = pair_founders(states)
    rows = np.zeros((s1 - s0, states))
    step = capacity(win)
    for lo in range(0, sites, step):
        hi = min(sites, lo + step)
        cover(win, lo, hi, s0, s1, False)
        for s in prange(s0, s1):
            R = np.empty(founders)
            e = np.empty(7)
            forward_sites(rows[s - s0], dosages, 0, win, s, lo, hi, R, e)
    answer = np.empty(s1 - s0)
    for s in prange(s0, s1):
        answer[s - s0] = row_max(rows[s - s0])
    return answer


@kernel(overloads=overloads_for(__name__, "score_direct"), cache=True, parallel=True, nogil=True)
def score_direct(haps, win, s0, s1):
    """Per-sample F_s of samples [s0, s1) without a dosage table (O(threads x Q) memory; the same
    arithmetic)."""
    sites = sites_of(win)
    first, second = unordered_pairs(len(haps))
    rows = np.zeros((s1 - s0, len(first)))
    step = capacity(win)
    for lo in range(0, sites, step):
        hi = min(sites, lo + step)
        cover(win, lo, hi, s0, s1, False)
        for s in prange(s0, s1):
            row = rows[s - s0]
            R = np.empty(len(haps))
            e = np.empty(7)
            for j in range(lo, hi):
                f, k, r = locate(win, s, j)
                if not site_logs(win, s, f, k, r, e):
                    continue
                transition(row, site_cost(win, s, f, k, r), R)
                for q in range(len(first)):
                    row[q] += e[pair_code7(haps[first[q], j], haps[second[q], j])] - CENTER
    answer = np.empty(s1 - s0)
    for s in prange(s0, s1):
        answer[s - s0] = row_max(rows[s - s0])
    return answer


@kernel(overloads=overloads_for(__name__, "score_switches_direct"), cache=True, parallel=True, nogil=True)
def score_switches_direct(haps, win, s0, s1):
    """Exact score and the fewest homologue switches among its exact maximisers, per sample of
    [s0, s1)."""
    sites = sites_of(win)
    first, second = unordered_pairs(len(haps))
    rows = np.zeros((s1 - s0, len(first)))
    counts = np.zeros((s1 - s0, len(first)), np.int64)
    step = capacity(win)
    for lo in range(0, sites, step):
        hi = min(sites, lo + step)
        cover(win, lo, hi, s0, s1, False)
        for s in prange(s0, s1):
            row = rows[s - s0]
            count = counts[s - s0]
            R = np.empty(len(haps))
            Rc = np.empty(len(haps), np.int64)
            e = np.empty(7)
            for j in range(lo, hi):
                f, k, r = locate(win, s, j)
                if not site_logs(win, s, f, k, r, e):
                    continue
                transition_count(row, count, site_cost(win, s, f, k, r), R, Rc)
                for q in range(len(first)):
                    row[q] += e[pair_code7(haps[first[q], j], haps[second[q], j])] - CENTER
    scores = np.empty(s1 - s0)
    switches = np.empty(s1 - s0, np.int64)
    for s in prange(s0, s1):
        row = rows[s - s0]
        count = counts[s - s0]
        m = row[0]
        k = count[0]
        for q in range(1, len(row)):
            if row[q] > m or (row[q] == m and count[q] < k):
                m = row[q]
                k = count[q]
        scores[s - s0] = m
        switches[s - s0] = k
    return scores, switches


def score_panel(haps, win, ranges, table_bytes, team):
    """Per-sample F_s of the panel with alleles haps (int8 K x L) through the window win (forward
    orientation, the panel's K: window.Windows.view).

    ranges: consecutive sample ranges (s0, s1, threads) covering every sample
    (workspace.sample_ranges), each scored by its own call at its thread count (a sample's score
    is its own recursion). The shared pair-dosage table (L x Q int8) is built when it fits
    table_bytes, else the direct kernel runs; both give the same values."""
    founders, sites = haps.shape
    states = founders * (founders + 1) // 2
    answer = np.empty(win[0].shape[0])
    if sites * states > table_bytes:
        for s0, s1, threads in ranges:
            with team.scope(threads):
                answer[s0:s1] = score_direct(haps, win, s0, s1)
        return answer
    first, second = unordered_pairs(founders)
    dosages = pair_dosages(haps, first, second, 0, sites)
    for s0, s1, threads in ranges:
        with team.scope(threads):
            answer[s0:s1] = score_dosages(dosages, win, s0, s1)
    return answer


__all__ = [
    "CENTER",
    "SWITCH_PENALTY",
    "pair_code7",
    "unordered_pairs",
    "pair_founders",
    "selected_alleles",
    "pair_dosages",
    "row_max",
    "transition",
    "lex_better",
    "transition_lex",
    "transition_count",
    "emit",
    "founder_maxima",
    "centred",
    "fused_step",
    "emit_scan",
    "finish_transition",
    "forward_sites",
    "backward_sites",
    "lex_founder_maxima",
    "lex_fused_step",
    "lex_emit_scan",
    "forward_sites_lex",
    "backward_sites_lex",
    "normalise",
    "score_dosages",
    "score_direct",
    "score_switches_direct",
    "score_panel",
]
