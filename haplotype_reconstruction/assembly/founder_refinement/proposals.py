"""Window proposals: exact founder-row maximisers of a binned surrogate on bounded windows.

Window problem (one founder f, a window of W units: prepared blocks for fine windows, L1 spans for
coarse copy moves): maximise the binned surrogate

    H(x) = sum_s max over bin-constant state paths [fwd_s(a) (x) bins of [a, c) with rows (X_-f, x) (x) bwd_s(c)]

over f's rows x (one alphabet per unit, every other founder fixed). The rest of the component enters
exactly through the current panel's normalised flank messages; the interior uses the component's
binned emissions (workspace.binned_emissions, bins of at least the level's minimum size) with a
switch-then-emission bin recursion. The surrogate only proposes: every proposal is re-scored
exactly against the current panel by the acceptance pass.

Solver (solve_focal, one nogil call per focal founder). The box bound (every sample choosing f's
later rows freely) certifies windows whose incumbent is already optimal without a programme.
Otherwise an exact forward frontier dynamic programme runs: a partial path keeps its closed
normalised per-sample message; it is dropped only if another partial path y is at least as good for
EVERY completion, sum_s max_z (M_p - M_y) + (T_p - T_y) <= tol (closed-message dominance), or if a
valid upper bound on all its completions lies below a feasible value. A programme whose frontier
passes EXPLORE_FRONTIER restarts with tighter pruning: Lagrangian suffix bounds (each sample
chooses f's later rows freely but pays zero-sum multipliers lam_s(unit, row), which cancel on every
common path; the multipliers come from block-coordinate sweeps that equalise the samples'
max-marginals) and the best feasible value among the incumbent, a beam and the dual's decoded
consensus paths. Pruning keeps every path within the margin of a feasible value, so the optimum and
the canonical answer are unchanged. A programme whose frontier would exceed the fixed cap
FounderRefinementConfig.frontier_bytes (frontier_limit) stops and keeps the incumbent (a reported
fallback; it binds at no reference size). Large programmes take memory from the run's memory
governor as they grow (programme_budgets): a solve that passes a budget is repeated within the next
one, so the result always equals one solve at the cap.

Canonical answer (a pure function of the problem): among paths within tol = 1e-7 + 1e-9 |H(incumbent)|
of the optimum, the incumbent if it is one, else the lexicographically first path, units left to
right, each in the order (incumbent row, then ascending index), found by exact restricted programmes.

Shared background: only the K states containing f depend on f's row, so children carry their K focal
values and per-bin switch floors and the background enters through per-unit tables
(window_dp). Work that does not depend on a focal founder, a row or a child is done once: the
per-state unit sums once per window for all its focal problems (_state_tables; per focal only the
background's segment maxima differ), with the closed start message and the incumbent's value; a
parent's row-independent quantities once per parent (_parent, _parent_floor); and a child sample
whose focal states can reach neither a bin maximum of the background nor the closing floor (an
upper bound by monotone rounding) takes the parent's shared outcome. Maxima over states use
interleaved running maxima (objective.row_max): the values of a sequential scan. The all-state
reference recursions of the box bound and the programme are kept as tests (tests/T5), which check
that the shared-background forms give the same answers.
"""
from __future__ import annotations

from contextlib import nullcontext

import numpy as np
from numba import njit, prange
from numba.typed import List

from ... import execution
from .objective import row_max

DOMINANCE_TOLERANCE = 1e-6
# Relative float margin between the box bound's and the programme's summation orders: a window is
# certified by the box only clear of it, so a certified window is one the programme would also
# leave unchanged.
BOX_ARITHMETIC_MARGIN = 5e-10
# Memory of the child messages a programme stores per unit (else they are recomputed).
STORE_BYTES = 128 * 1024 ** 2

OK, INFEASIBLE, BACKSTOP = 0, 1, 2
# Empty arrays: disable _child's floor shortcut (the beam and the canonical-tie scan; read only).
NO_FLOOR = np.empty(0)
NO_FLOOR2 = np.empty((0, 0))
# A programme whose frontier passes EXPLORE_FRONTIER restarts with tighter pruning (work only, never
# the answer): the Lagrangian suffix bounds of the window problem and the best feasible value among
# the incumbent, a BEAM_WIDTH-wide beam and the dual's decoded paths. The multiplier search runs
# block-coordinate sweeps until its bound is within the canonical tolerance of a feasible value, at
# most DUAL_SWEEPS (a sweep can leave the bound unchanged while the decoded paths still improve);
# any zero-sum multipliers give valid bounds, so the limit never decides an answer.
EXPLORE_FRONTIER = 256
BEAM_WIDTH = 16
DUAL_SWEEPS = 100

# Window templates: fine windows of FINE_WINDOW prepared blocks every FINE_STRIDE blocks (plus the
# window ending at the last block); coarse windows of COARSE_WINDOW L1 spans every COARSE_STRIDE.
FINE_WINDOW = 8
FINE_STRIDE = 4
COARSE_WINDOW = 4
COARSE_STRIDE = 2

# solve_focal statuses
UNCHANGED, CHANGED, FRONTIER_LIMIT, LOST = 0, 1, 2, 3


@njit(inline="always")
def _sum_max(a, b):
    """max_z (a[z] + b[z]) (non-empty, no NaN) with four interleaved running maxima: the maximum of
    the same sums as a sequential scan."""
    n = len(a)
    m0 = a[0] + b[0]
    m1 = m0
    m2 = m0
    m3 = m0
    z = 1
    while z + 4 <= n:
        v = a[z] + b[z]
        if v > m0:
            m0 = v
        v = a[z + 1] + b[z + 1]
        if v > m1:
            m1 = v
        v = a[z + 2] + b[z + 2]
        if v > m2:
            m2 = v
        v = a[z + 3] + b[z + 3]
        if v > m3:
            m3 = v
        z += 4
    while z < n:
        v = a[z] + b[z]
        if v > m0:
            m0 = v
        z += 1
    if m1 > m0:
        m0 = m1
    if m2 > m0:
        m0 = m2
    if m3 > m0:
        m0 = m3
    return m0


@njit(inline="always")
def _diff_max(a, b):
    """max_z (a[z] - b[z]) (non-empty, no NaN) with four interleaved running maxima."""
    n = len(a)
    m0 = a[0] - b[0]
    m1 = m0
    m2 = m0
    m3 = m0
    z = 1
    while z + 4 <= n:
        v = a[z] - b[z]
        if v > m0:
            m0 = v
        v = a[z + 1] - b[z + 1]
        if v > m1:
            m1 = v
        v = a[z + 2] - b[z + 2]
        if v > m2:
            m2 = v
        v = a[z + 3] - b[z + 3]
        if v > m3:
            m3 = v
        z += 4
    while z < n:
        v = a[z] - b[z]
        if v > m0:
            m0 = v
        z += 1
    if m1 > m0:
        m0 = m1
    if m2 > m0:
        m0 = m2
    if m3 > m0:
        m0 = m3
    return m0


@njit(cache=True, nogil=True)
def _extend(parent_n, parent_c, emis, col, focal, row, first, second, penalty, out_n, out_c):
    """Closed normalised message of a partial path extended by `row` on one unit."""
    samples, states = parent_n.shape
    bins = emis.shape[3]
    total = 0.0
    for s in range(samples):
        for z in range(states):
            out_n[s, z] = parent_n[s, z]
        for bn in range(bins):
            sw = row_max(out_n[s]) - penalty
            for z in range(states):
                a = first[z]
                b = second[z]
                ra = row if a == focal else col[a]
                rb = row if b == focal else col[b]
                v = out_n[s, z]
                if sw > v:
                    v = sw
                out_n[s, z] = v + emis[s, ra, rb, bn]
        m = row_max(out_n[s])
        lo = m - penalty
        for z in range(states):
            v = out_n[s, z]
            if lo > v:
                v = lo
            out_n[s, z] = v - m
        out_c[s] = parent_c[s] + m
        total += out_c[s]
    return total


@njit(cache=True, nogil=True)
def _join(msg_n, msg_c, tail):
    """sum_s [C_s + max_z (N_s(z) + tail_s(z))]."""
    samples = msg_n.shape[0]
    total = 0.0
    for s in range(samples):
        total += msg_c[s] + _sum_max(msg_n[s], tail[s])
    return total


@njit(cache=True, nogil=True)
def _dominance(n_i, t_i, n_j, t_j, tol):
    """True iff no completion makes i better than j by more than tol."""
    samples = n_i.shape[0]
    d = t_i - t_j
    if d > tol:
        return False
    for s in range(samples):
        d += _diff_max(n_i[s], n_j[s])
        if d > tol:
            return False
    return True


@njit(cache=True, nogil=True)
def _stable_order(keys_float, keys_inc, par, cho):
    """Order by (-T, incumbent row first, parent, row), stable and deterministic."""
    n = len(par)
    order = np.argsort(cho, kind="mergesort")
    order = order[np.argsort(par[order], kind="mergesort")]
    order = order[np.argsort(-keys_inc[order], kind="mergesort")]
    order = order[np.argsort(-keys_float[order], kind="mergesort")]
    return order


# --------------------------------------------------------------------------------------
# Shared-background kernels (O(K^2) per unit and sample for one focal founder).
#
# Only the K states containing the focal founder f depend on f's row. Every other
# ("background") state z has emissions fixed by the other founders' rows, so within a
# unit of t bins its value, from a parent message N, is exactly
#     x[z] = max(P[z], max_{l >= 1} (s_l + E_{l..t-1}[z])),
#     P[z] = max(N[z], m_0 - pen) + E_{0..t-1}[z],  s_l = m_l - pen,
# where m_l is the maximum over all states before bin l (the switch floor) and E are
# sums of z's bin emissions (one bin: x[z] = max(N[z], -pen) + e[z], shared by all
# children). A child therefore carries its K focal values and t scalars; the background
# enters through per-unit segment maxima max_z E_{a..c}[z] and per-parent maxima. For
# one-bin units every child quantity used for ordering (totals) and every frontier
# message (materialised by _extend) equals the all-state reference recursion's (tests); for
# multi-bin units the decomposition is exact in real arithmetic. Bounds only prune (a margin
# covers rounding); dominance decisions are taken from the shared-part formulas only when
# they clear the tolerance by a float margin, otherwise both messages are materialised
# and _dominance (the all-state test) decides.
# --------------------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _focal_layout(focal, first, second):
    """fz[c] = state {f, c}; bg = the states without f (in state order)."""
    states = len(first)
    founders = second[states - 1] + 1
    fz = np.empty(founders, np.int64)
    bg = np.empty(states - founders, np.int64)
    count = 0
    for z in range(states):
        a = first[z]
        b = second[z]
        if a == focal:
            fz[b] = z
        elif b == focal:
            fz[a] = z
        else:
            bg[count] = z
            count += 1
    return fz, bg


@njit(cache=True, nogil=True)
def _state_tables(emis, col, first, second):
    """Focal-independent sums of one unit for every state z (rows fixed by col), shared by every
    focal problem of a window: P[s, z, k] = sum_{m <= k} e_m(z), T[s, z, k] = sum_{m >= k} e_m(z)
    (T[.., t] = 0) and, per (s, a, c) with a <= c, the largest run sum sum_{m=a}^{c} e_m(z) over
    all states (best), the founders (bx, by) of the first state attaining it and the largest over
    the states without founder bx (mx) and without founder by (my)."""
    samples = emis.shape[0]
    bins = emis.shape[3]
    states = len(first)
    P = np.empty((samples, states, bins))
    T = np.zeros((samples, states, bins + 1))
    best = np.full((samples, bins, bins), -np.inf)
    bx = np.zeros((samples, bins, bins), np.int64)
    by = np.zeros((samples, bins, bins), np.int64)
    mx = np.full((samples, bins, bins), -np.inf)
    my = np.full((samples, bins, bins), -np.inf)
    runs = np.empty((states, bins, bins))
    for s in range(samples):
        for z in range(states):
            ra = col[first[z]]
            rb = col[second[z]]
            acc = 0.0
            for k in range(bins):
                acc += emis[s, ra, rb, k]
                P[s, z, k] = acc
            acc = 0.0
            for k in range(bins - 1, -1, -1):
                acc += emis[s, ra, rb, k]
                T[s, z, k] = acc
            for a in range(bins):
                run = 0.0
                for c in range(a, bins):
                    run += emis[s, ra, rb, c]
                    runs[z, a, c] = run
                    if run > best[s, a, c]:
                        best[s, a, c] = run
                        bx[s, a, c] = first[z]
                        by[s, a, c] = second[z]
        for z in range(states):
            for a in range(bins):
                for c in range(a, bins):
                    v = runs[z, a, c]
                    x = bx[s, a, c]
                    y = by[s, a, c]
                    if first[z] != x and second[z] != x and v > mx[s, a, c]:
                        mx[s, a, c] = v
                    if first[z] != y and second[z] != y and v > my[s, a, c]:
                        my[s, a, c] = v
    return P, T, best, bx, by, mx, my


@njit(cache=True, nogil=True)
def _unit_tables(P, T, best, bx, by, mx, my, focal, bg):
    """Background sums of one unit for focal founder f (from _state_tables): prefix[s, b, k] =
    sum_{m <= k} e_m, suffix[s, b, k] = sum_{m >= k} e_m (suffix[.., t] = 0) of background state
    bg[b], and seg[s, a, c] = max_b sum_{m=a}^{c} e_m over the background (the states without f)."""
    samples = P.shape[0]
    bins = P.shape[2]
    nb = len(bg)
    prefix = np.empty((samples, nb, bins))
    suffix = np.empty((samples, nb, bins + 1))
    seg = np.full((samples, bins, bins), -np.inf)
    for s in range(samples):
        for b in range(nb):
            z = bg[b]
            for k in range(bins):
                prefix[s, b, k] = P[s, z, k]
            for k in range(bins + 1):
                suffix[s, b, k] = T[s, z, k]
        for a in range(bins):
            for c in range(a, bins):
                if focal == bx[s, a, c]:
                    seg[s, a, c] = mx[s, a, c]
                elif focal == by[s, a, c]:
                    seg[s, a, c] = my[s, a, c]
                else:
                    seg[s, a, c] = best[s, a, c]
    return prefix, suffix, seg


@njit(cache=True, nogil=True)
def window_box(emissions, cols, cands, offsets, focal, first, second, penalty, end, fz, bg,
               prefixes, suffixes, segs):
    """U[u] >= the value of units u.. plus the end flank from any state, per sample.

    Each sample may choose f's row of every later unit freely (a relaxation); the
    backward bin recursion is emission-then-switch, matching the forward one. Shared
    background: per row only the K focal states and t switch maxima m_k are carried; the
    background states take max over rows of the floors, which commutes with the
    (monotone) float subtraction of the penalty, so one-bin units equal the all-state
    reference recursion (tests) bitwise.
    """
    units = len(emissions)
    samples, states = end.shape
    founders = len(fz)
    nb = len(bg)
    upper = np.empty((units + 1, samples, states))
    upper[units] = end
    yf = np.empty(founders)
    for u in range(units - 1, -1, -1):
        emis = emissions[u]
        bins = emis.shape[3]
        col = cols[u]
        prefix, suffix, seg = prefixes[u], suffixes[u], segs[u]
        best_m = np.empty(bins)
        mrow = np.empty(bins)
        ue = np.empty(bins)
        begin = offsets[u]
        rows = offsets[u + 1] - begin
        ra, rb = _unit_rows(cands, begin, rows, col, focal, first, second, fz)
        start = np.empty(founders)
        top = np.empty(founders)
        for s in range(samples):
            # ue[k] = max over background of the no-floor chain U + e_k + ... + e_{t-1}.
            for k in range(bins):
                ue[k] = -np.inf
            for b in range(nb):
                ub = upper[u + 1, s, bg[b]]
                for k in range(bins):
                    v = ub + suffix[s, b, k]
                    if v > ue[k]:
                        ue[k] = v
            for c in range(founders):
                start[c] = upper[u + 1, s, fz[c]]
                top[c] = -np.inf
            for k in range(bins):
                best_m[k] = -np.inf
            for g in range(rows):
                ga = ra[g]
                gb = rb[g]
                if bins == 1:
                    # The general step below for one bin, its passes merged (same operations and
                    # comparison order per value).
                    m = ue[0]
                    for c in range(founders):
                        y = start[c] + emis[s, ga[c], gb[c], 0]
                        yf[c] = y
                        if y > m:
                            m = y
                    sw = m - penalty
                    for c in range(founders):
                        y = yf[c]
                        if sw > y:
                            y = sw
                        if y > top[c]:
                            top[c] = y
                    if m > best_m[0]:
                        best_m[0] = m
                    continue
                for c in range(founders):
                    yf[c] = start[c]
                for k in range(bins - 1, -1, -1):
                    m = ue[k]
                    for l in range(k + 1, bins):
                        v = mrow[l] - penalty + seg[s, k, l - 1]
                        if v > m:
                            m = v
                    for c in range(founders):
                        yf[c] += emis[s, ga[c], gb[c], k]
                        if yf[c] > m:
                            m = yf[c]
                    mrow[k] = m
                    sw = m - penalty
                    for c in range(founders):
                        if sw > yf[c]:
                            yf[c] = sw
                for c in range(founders):
                    if yf[c] > top[c]:
                        top[c] = yf[c]
                for k in range(bins):
                    if mrow[k] > best_m[k]:
                        best_m[k] = mrow[k]
            for c in range(founders):
                upper[u, s, fz[c]] = top[c]
            for b in range(nb):
                z = bg[b]
                v = upper[u + 1, s, z] + suffix[s, b, 0]
                for k in range(bins):
                    w = best_m[k] - penalty + (prefix[s, b, k - 1] if k > 0 else 0.0)
                    if w > v:
                        v = w
                upper[u, s, z] = v
    return upper


@njit(cache=True, nogil=True)
def _parent(parent_n, bg, fz, prefix, unext, penalty, pmax, pb, tp, hi, lo, arg, a0, background):
    """Row-independent tables of one parent on one unit, per sample: the parent's maximum pmax,
    pb[s, k] = max_b P_k[b] (the background chain after bin k without later floors), tp[s] =
    max_b (P_{t-1}[b] + U[b]), the extremes and the argmax of P_{t-1} over the background (the
    dominance shortcuts), the focal states after the unit's first switch a0[s, c] = max(N[fz[c]],
    pmax - pen) (every child's first step) and, when `background` has rows, the background states
    P_{t-1}[z] of every child (its shared form: max(N[z], pmax - pen) + E_{0..t-1}[z]). One pass
    over the background serves all of them; computed once per parent, not per child."""
    samples = parent_n.shape[0]
    bins = prefix.shape[2]
    nb = len(bg)
    founders = len(fz)
    store = background.shape[0] > 0
    if bins == 1 and nb > 0:
        _parent_one_bin(parent_n, bg, fz, prefix, unext, penalty, pmax, pb, tp, hi, lo, arg, a0, background,
                        store)
        return
    for s in range(samples):
        m = row_max(parent_n[s])
        pmax[s] = m
        sw = m - penalty
        for k in range(bins):
            pb[s, k] = -np.inf
        tp[s] = -np.inf
        hi[s] = -np.inf
        lo[s] = np.inf
        arg[s] = 0
        for b in range(nb):
            v = parent_n[s, bg[b]]
            if sw > v:
                v = sw
            for k in range(bins):
                w = v + prefix[s, b, k]
                if w > pb[s, k]:
                    pb[s, k] = w
            last = v + prefix[s, b, bins - 1]
            if last > hi[s]:
                hi[s] = last
                arg[s] = b
            if last < lo[s]:
                lo[s] = last
            w = last + unext[s, bg[b]]
            if w > tp[s]:
                tp[s] = w
            if store:
                background[s, bg[b]] = last
        for c in range(founders):
            v = parent_n[s, fz[c]]
            if sw > v:
                v = sw
            a0[s, c] = v


@njit(cache=True, nogil=True)
def _parent_one_bin(parent_n, bg, fz, prefix, unext, penalty, pmax, pb, tp, hi, lo, arg, a0, background,
                    store):
    """_parent on a one-bin unit: the background values P[b] = max(N[bg[b]], pmax - pen) + e[b] first,
    then their maximum (= pb, = hi) with its first index (arg), minimum (lo) and the bound term
    tp = max_b (P[b] + U[bg[b]]) by four interleaved running extremes each, combined so that the
    values (and the first index of the maximum) are those of the sequential scan."""
    samples = parent_n.shape[0]
    nb = len(bg)
    founders = len(fz)
    last = np.empty(nb)
    for s in range(samples):
        m = row_max(parent_n[s])
        pmax[s] = m
        sw = m - penalty
        for b in range(nb):
            v = parent_n[s, bg[b]]
            if sw > v:
                v = sw
            last[b] = v + prefix[s, b, 0]
        if store:
            for b in range(nb):
                background[s, bg[b]] = last[b]
        # running extremes over b = j (mod 4)
        h0 = last[0]
        i0 = 0
        l0 = last[0]
        t0 = last[0] + unext[s, bg[0]]
        h1 = h0
        i1 = 0
        l1 = l0
        t1 = t0
        h2 = h0
        i2 = 0
        l2 = l0
        t2 = t0
        h3 = h0
        i3 = 0
        l3 = l0
        t3 = t0
        b = 1
        while b + 4 <= nb:
            x = last[b]
            if x > h0:
                h0 = x
                i0 = b
            if x < l0:
                l0 = x
            w = x + unext[s, bg[b]]
            if w > t0:
                t0 = w
            x = last[b + 1]
            if x > h1:
                h1 = x
                i1 = b + 1
            if x < l1:
                l1 = x
            w = x + unext[s, bg[b + 1]]
            if w > t1:
                t1 = w
            x = last[b + 2]
            if x > h2:
                h2 = x
                i2 = b + 2
            if x < l2:
                l2 = x
            w = x + unext[s, bg[b + 2]]
            if w > t2:
                t2 = w
            x = last[b + 3]
            if x > h3:
                h3 = x
                i3 = b + 3
            if x < l3:
                l3 = x
            w = x + unext[s, bg[b + 3]]
            if w > t3:
                t3 = w
            b += 4
        while b < nb:
            x = last[b]
            if x > h0:
                h0 = x
                i0 = b
            if x < l0:
                l0 = x
            w = x + unext[s, bg[b]]
            if w > t0:
                t0 = w
            b += 1
        if h1 > h0 or (h1 == h0 and i1 < i0):
            h0 = h1
            i0 = i1
        if h2 > h0 or (h2 == h0 and i2 < i0):
            h0 = h2
            i0 = i2
        if h3 > h0 or (h3 == h0 and i3 < i0):
            h0 = h3
            i0 = i3
        if l1 < l0:
            l0 = l1
        if l2 < l0:
            l0 = l2
        if l3 < l0:
            l0 = l3
        if t1 > t0:
            t0 = t1
        if t2 > t0:
            t0 = t2
        if t3 > t0:
            t0 = t3
        pb[s, 0] = h0
        hi[s] = h0
        arg[s] = i0
        lo[s] = l0
        tp[s] = t0
        for c in range(founders):
            v = parent_n[s, fz[c]]
            if sw > v:
                v = sw
            a0[s, c] = v


@njit(cache=True, nogil=True)
def _row_pairs(row, col, focal, first, second, fz, out_a, out_b):
    """Emission row indices (out_a[c], out_b[c]) of the focal states {f, c} when f takes `row`."""
    for c in range(len(fz)):
        z = fz[c]
        out_a[c] = row if first[z] == focal else col[first[z]]
        out_b[c] = row if second[z] == focal else col[second[z]]


@njit(cache=True, nogil=True)
def _unit_rows(cands, begin, rows, col, focal, first, second, fz):
    """_row_pairs of a unit's candidate rows cands[begin:begin + rows]: (rows, K) int64 arrays."""
    founders = len(fz)
    ra = np.empty((rows, founders), np.int64)
    rb = np.empty((rows, founders), np.int64)
    for g in range(rows):
        _row_pairs(cands[begin + g], col, focal, first, second, fz, ra[g], rb[g])
    return ra, rb


@njit(cache=True, nogil=True)
def _focal_values(values, fz):
    """values[:, fz] as a contiguous (S, K) array."""
    samples = values.shape[0]
    out = np.empty((samples, len(fz)))
    for s in range(samples):
        for c in range(len(fz)):
            out[s, c] = values[s, fz[c]]
    return out


@njit(cache=True, nogil=True)
def _row_maxima(emis, ra, rb):
    """emax[g, s, k] = max_c e_k(ra[g, c], rb[g, c]): the largest bin-k emission of any focal
    state of a child taking row g (with _parent_floor, the floor test of _child)."""
    rows, founders = ra.shape
    samples = emis.shape[0]
    bins = emis.shape[3]
    emax = np.empty((rows, samples, bins))
    for g in range(rows):
        for s in range(samples):
            for k in range(bins):
                m = emis[s, ra[g, 0], rb[g, 0], k]
                for c in range(1, founders):
                    v = emis[s, ra[g, c], rb[g, c], k]
                    if v > m:
                        m = v
                emax[g, s, k] = m
    return emax


@njit(cache=True, nogil=True)
def _parent_floor(parent_c, pmax, pb, tp, a0, seg, te, ubmax, ufz, penalty, amax, fmb, fs, fm, flo,
                  fd, fcs, fsb):
    """The outcome every child of the parent shares in sample s when its focal states never reach
    the background's maximum within the unit nor its closing floor (_child's floor test): the
    bin maxima fmb[s, k] and switch floors fs[s, k] are then the background's (pb and the
    earlier floors through seg), the unit-end maximum fm, every closed focal value fd = lo - m,
    and the sample's terms of the child's total (fcs) and bound (fsb) are fixed. Computed by
    _child's operations in its order; amax[s] = max_c a0[s, c] starts the test."""
    samples, founders = a0.shape
    bins = pb.shape[1]
    for s in range(samples):
        a = a0[s, 0]
        for c in range(1, founders):
            if a0[s, c] > a:
                a = a0[s, c]
        amax[s] = a
        fs[s, 0] = pmax[s] - penalty
        fmb[s, 0] = pmax[s]
        for k in range(1, bins):
            m = pb[s, k - 1]
            for l in range(1, k):
                v = fs[s, l] + seg[s, l, k - 1]
                if v > m:
                    m = v
            fmb[s, k] = m
            fs[s, k] = m - penalty
        m = pb[s, bins - 1]
        for l in range(1, bins):
            v = fs[s, l] + seg[s, l, bins - 1]
            if v > m:
                m = v
        fm[s] = m
        lo = m - penalty
        flo[s] = lo
        cs = parent_c[s] + m
        best = tp[s] - m
        v = lo - m + ubmax[s]
        if v > best:
            best = v
        for l in range(1, bins):
            v = fs[s, l] - m + te[s, l]
            if v > best:
                best = v
        d = lo - m
        fd[s] = d
        for c in range(founders):
            w = d + ufz[s, c]
            if w > best:
                best = w
        fcs[s] = cs
        fsb[s] = cs + best


@njit(cache=True, nogil=True)
def _child(parent_c, pmax, pb, tp, a0, seg, te, ubmax, ufz, emis, ra, rb, penalty, out_m, out_f,
           out_s, amax, emax, fmb, fs, fm, flo, fd, fcs, fsb, out_b):
    """One child (a parent extended by one row on one unit) without its background states.

    From the parent's tables (_parent: pmax, pb, tp, a0), the unit's suffix bound at the focal
    states ufz = U[u + 1][:, fz] and the row's focal emission indices ra, rb (_row_pairs); fills
    per sample the unit-end maximum out_m, the closed focal values out_f and the switch floors
    out_s[k] = m_k - pen; returns the child's total (sum of its constants, the reference's
    ordering key) and its bound (the reference's _join against U[u + 1]); when out_b has entries,
    out_b[s] = the sample's term of the bound (the dual sweep's max-marginal).

    Floor shortcut (when amax has entries: _parent_floor and the row's _row_maxima emax): X_k, an
    upper bound on every focal state after bin k (X_0 = amax + emax_0, X_k = max(X_{k-1}, fs_k) +
    emax_k; rounding is monotone), is checked against the background's maximum of each next bin
    and the closing floor. A sample that passes never lets a focal state set a maximum or close
    above the floor, so it takes the parent's shared outcome: the values the full computation
    produces.
    """
    samples = a0.shape[0]
    founders = a0.shape[1]
    bins = emis.shape[3]
    xf = np.empty(founders)
    total = 0.0
    bound = 0.0
    floor = len(amax) > 0
    terms = len(out_b) > 0
    for s in range(samples):
        sw = pmax[s] - penalty
        out_s[s, 0] = sw
        if floor:
            x = amax[s] + emax[s, 0]
            below = True
            for k in range(1, bins):
                if x > fmb[s, k]:
                    below = False
                    break
                y = fs[s, k]
                if x > y:
                    y = x
                x = y + emax[s, k]
            if below and x <= flo[s]:
                d = fd[s]
                out_m[s] = fm[s]
                for k in range(1, bins):
                    out_s[s, k] = fs[s, k]
                for c in range(founders):
                    out_f[s, c] = d
                total += fcs[s]
                bound += fsb[s]
                if terms:
                    out_b[s] = fsb[s]
                continue
        if bins == 1:
            # The unit-end maximum over the background (pb) and the focal states, taken as the
            # focal states are formed (the comparisons of the general case, in the same order).
            m = pb[s, 0]
            for c in range(founders):
                x = a0[s, c] + emis[s, ra[c], rb[c], 0]
                xf[c] = x
                if x > m:
                    m = x
        else:
            for c in range(founders):
                xf[c] = a0[s, c] + emis[s, ra[c], rb[c], 0]
            for k in range(1, bins):
                m = pb[s, k - 1]
                for l in range(1, k):
                    v = out_s[s, l] + seg[s, l, k - 1]
                    if v > m:
                        m = v
                for c in range(founders):
                    if xf[c] > m:
                        m = xf[c]
                sw = m - penalty
                out_s[s, k] = sw
                for c in range(founders):
                    v = xf[c]
                    if sw > v:
                        v = sw
                    xf[c] = v + emis[s, ra[c], rb[c], k]
            m = pb[s, bins - 1]
            for l in range(1, bins):
                v = out_s[s, l] + seg[s, l, bins - 1]
                if v > m:
                    m = v
            for c in range(founders):
                if xf[c] > m:
                    m = xf[c]
        out_m[s] = m
        lo = m - penalty
        cs = parent_c[s] + m
        total += cs
        best = tp[s] - m
        v = lo - m + ubmax[s]
        if v > best:
            best = v
        for l in range(1, bins):
            v = out_s[s, l] - m + te[s, l]
            if v > best:
                best = v
        for c in range(founders):
            v = xf[c]
            if lo > v:
                v = lo
            v = v - m
            out_f[s, c] = v
            w = v + ufz[s, c]
            if w > best:
                best = w
        term = cs + best
        bound += term
        if terms:
            out_b[s] = term
    return total, bound


@njit(cache=True, nogil=True)
def _parent_alone(parent_n, bg, fz, prefix, unext, penalty):
    """_parent's tables of one parent for its children's totals and bounds: (pmax, pb, tp, a0)."""
    samples, states = parent_n.shape
    bins = prefix.shape[2]
    pmax = np.empty(samples)
    pb = np.empty((samples, bins))
    tp = np.empty(samples)
    a0 = np.empty((samples, len(fz)))
    _parent(parent_n, bg, fz, prefix, unext, penalty, pmax, pb, tp, np.empty(samples), np.empty(samples),
            np.empty(samples, np.int64), a0, np.empty((0, states)))
    return pmax, pb, tp, a0


@njit(cache=True, nogil=True)
def _assemble_one_bin(parent_c, background, bg, fz, m_i, f_i, penalty, out_n, out_c):
    """The closed message of a one-bin child from its parent's shared background (_parent) and its
    own focal values (_child): the values _extend computes, by the same operations (the unit-end
    maximum m_i and the closed focal values are the child's; background states close as
    max(P[z], m - pen) - m; constants parent_c + m). Returns the child's total."""
    samples = len(m_i)
    total = 0.0
    for s in range(samples):
        m = m_i[s]
        lo = m - penalty
        for b in range(len(bg)):
            z = bg[b]
            v = background[s, z]
            if lo > v:
                v = lo
            out_n[s, z] = v - m
        for c in range(len(fz)):
            out_n[s, fz[c]] = f_i[s, c]
        out_c[s] = parent_c[s] + m
        total += out_c[s]
    return total


@njit(cache=True, nogil=True)
def _background_value(p_last, m, floors, suffix_b, bins, penalty):
    """Closed background value max(-pen, P_{t-1} - M, max_{l >= 1} (s_l - M + E_{l..t-1}))."""
    v = p_last - m
    for l in range(1, bins):
        w = floors[l] - m + suffix_b[l]
        if w > v:
            v = w
    if -penalty > v:
        v = -penalty
    return v


@njit(cache=True, nogil=True)
def _same_parent_bounds(mi, fi, si, ti, mj, fj, sj, tj, bins, pmax, hi, lo, arg, parent_n, bg,
                        prefix, suffix, probes, penalty, stop_above):
    """Upper and lower bounds on d(i, j) = t_i - t_j + sum_s max_z (n_i - n_j) for two
    children of one parent, from their focal values and floors.

    Focal states are compared directly. Background: one bin -> the closed values are
    max(P - M, -pen), a monotone function of the shared P, so the largest difference is
    at the extremes of P (bounds equal); several bins -> upper bound
    max(0, max_l alpha_i,l - alpha_j,l) (alpha_0 = -M, alpha_l = s_l - M) and, as the lower
    bound, the exact difference at probe states (the argmax of P_{t-1} and of each
    E_{l..t-1}). Stops once the lower bound exceeds stop_above (upper returned as inf).
    """
    samples = fi.shape[0]
    founders = fi.shape[1]
    d = ti - tj
    if d > stop_above:
        return np.inf, d
    upper_sum = d
    lower_sum = d
    for s in range(samples):
        fd = fi[s, 0] - fj[s, 0]
        for c in range(1, founders):
            v = fi[s, c] - fj[s, c]
            if v > fd:
                fd = v
        a = mi[s]
        b2 = mj[s]
        if bins == 1:
            h1 = max(hi[s] - a, -penalty) - max(hi[s] - b2, -penalty)
            h2 = max(lo[s] - a, -penalty) - max(lo[s] - b2, -penalty)
            ub = max(h1, h2)
            lb = ub
        else:
            ub = b2 - a
            if ub < 0.0:
                ub = 0.0
            for l in range(1, bins):
                v = (si[s, l] - a) - (sj[s, l] - b2)
                if v > ub:
                    ub = v
            lb = -np.inf
            floor0 = pmax[s] - penalty
            for q in range(bins):
                b = arg[s] if q == 0 else probes[s, q]
                v0 = parent_n[s, bg[b]]
                if floor0 > v0:
                    v0 = floor0
                p_last = v0 + prefix[s, b, bins - 1]
                v = (_background_value(p_last, a, si[s], suffix[s, b], bins, penalty)
                     - _background_value(p_last, b2, sj[s], suffix[s, b], bins, penalty))
                if v > lb:
                    lb = v
        upper_sum += max(fd, ub)
        lower_sum += max(0.0, max(fd, lb))
        if lower_sum > stop_above:
            return np.inf, lower_sum
    return upper_sum, lower_sum


@njit(cache=True, nogil=True)
def _safety(ti, tj, tk):
    """Float margin around a dominance threshold (covers both arithmetics' rounding)."""
    return 1e-9 + 1e-13 * (abs(ti) + abs(tj) + abs(tk))


@njit(cache=True, nogil=True)
def _dominance_value(n_i, t_i, n_j, t_j, tol):
    """_dominance with the accumulated value: (dominated, d), d complete when dominated."""
    samples = n_i.shape[0]
    d = t_i - t_j
    if d > tol:
        return False, d
    for s in range(samples):
        d += _diff_max(n_i[s], n_j[s])
        if d > tol:
            return False, d
    return True, d


@njit(cache=True, nogil=True)
def _one_bin_background(parent_n, pmax, emis, col, first, second, bg, penalty, out):
    """Background states of every child of a parent on a one-bin unit, before closing:
    max(N[z], m - pen) + e[z] (the reference's _extend arithmetic, shared by all rows)."""
    samples = parent_n.shape[0]
    for s in range(samples):
        sw = pmax[s] - penalty
        for b in range(len(bg)):
            z = bg[b]
            v = parent_n[s, z]
            if sw > v:
                v = sw
            out[s, z] = v + emis[s, col[first[z]], col[second[z]], 0]


@njit(cache=True, nogil=True)
def _cross_lower_bound(m_i, f_i, t_i, background, fz, probe, n_j, t_j, penalty, one_bin,
                       s_i, suffix, bins, threshold):
    """True when a lower bound of d(i, j) exceeds threshold: the running sum of the
    per-sample maxima restricted to the focal states and the background state where the
    parent's shared chain is largest (O(K) per sample, stopping as soon as it exceeds)."""
    samples = f_i.shape[0]
    d = t_i - t_j
    if d > threshold:
        return True
    for s in range(samples):
        m = m_i[s]
        best = 0.0
        for c in range(len(fz)):
            v = f_i[s, c] - n_j[s, fz[c]]
            if v > best:
                best = v
        z = probe[s, 0]
        b = probe[s, 1]
        if one_bin:
            x = background[s, z]
            lo = m - penalty
            if lo > x:
                x = lo
            v = (x - m) - n_j[s, z]
        else:
            v = _background_value(background[s, z], m, s_i[s], suffix[s, b], bins, penalty) - n_j[s, z]
        if v > best:
            best = v
        d += best
        if d > threshold:
            return True
    return False


@njit(cache=True, nogil=True)
def _dominance_one_bin(m_i, f_i, t_i, background, fz, bg, n_j, t_j, tol, penalty):
    """_dominance_value(n_i, t_i, n_j, t_j, tol) for a one-bin child kept in shared form:
    every closed entry of n_i is formed with the reference's operations (focal values
    f_i; background max(x, M - pen) - M from the parent's shared x), so the values,
    the per-sample maxima, the accumulation order and the early exit are the
    reference's."""
    samples = f_i.shape[0]
    d = t_i - t_j
    if d > tol:
        return False, d
    for s in range(samples):
        m = m_i[s]
        lo = m - penalty
        best = -np.inf
        for c in range(len(fz)):
            v = f_i[s, c] - n_j[s, fz[c]]
            if v > best:
                best = v
        for b in range(len(bg)):
            z = bg[b]
            x = background[s, z]
            if lo > x:
                x = lo
            v = (x - m) - n_j[s, z]
            if v > best:
                best = v
        d += best
        if d > tol:
            return False, d
    return True, d


@njit(cache=True, nogil=True)
def _multi_bin_background(parent_n, pmax, prefix, bg, penalty, out):
    """P_{t-1}[z] = max(N[z], m - pen) + E_{0..t-1}[z] of a parent's background states."""
    samples = parent_n.shape[0]
    bins = prefix.shape[2]
    for s in range(samples):
        sw = pmax[s] - penalty
        for b in range(len(bg)):
            z = bg[b]
            v = parent_n[s, z]
            if sw > v:
                v = sw
            out[s, z] = v + prefix[s, b, bins - 1]


@njit(cache=True, nogil=True)
def _dominance_multi_bin(m_i, f_i, s_i, t_i, p_last, suffix, fz, bg, n_j, t_j, tol, penalty):
    """_dominance of a multi-bin child in shared form against a materialised path: 1
    dominated, 0 not, -1 open (the closed background values are formed from the
    decomposition, exact in real arithmetic; decided only clear of tol by the float
    margin, the caller materialises otherwise)."""
    samples = f_i.shape[0]
    bins = s_i.shape[1]
    safety = _safety(t_i, t_j, 0.0)
    d = t_i - t_j
    if d > tol + safety:
        return 0
    for s in range(samples):
        m = m_i[s]
        best = -np.inf
        for c in range(len(fz)):
            v = f_i[s, c] - n_j[s, fz[c]]
            if v > best:
                best = v
        for b in range(len(bg)):
            z = bg[b]
            v = _background_value(p_last[s, z], m, s_i[s], suffix[s, b], bins, penalty) - n_j[s, z]
            if v > best:
                best = v
        d += best
        if d > tol + safety:
            return 0
    if d <= tol - safety:
        return 1
    return -1


@njit(cache=True, nogil=True)
def _rows3(buffer, rows, a, b):
    """A scratch array of at least `rows` leading entries (a fresh one only when `buffer` is too
    small; contents are not kept)."""
    if buffer.shape[0] >= rows:
        return buffer
    return np.empty((max(rows, 2 * buffer.shape[0]), a, b))


@njit(cache=True, nogil=True)
def _rows2(buffer, rows, a):
    if buffer.shape[0] >= rows:
        return buffer
    return np.empty((max(rows, 2 * buffer.shape[0]), a))


@njit(cache=True, nogil=True)
def window_dp(emissions, cols, cands, offsets, focal, incumbent, first, second, penalty,
              start_n, start_c, unit0, end, upper, lower, margin, dom_tol, backstop,
              store_limit, fz, bg, prefixes, suffixes, segs):
    """Exact maximum of H over f's rows on units unit0.. from one partial path.

    Returns (status, value, path of units unit0.., maximum frontier). status 1: no
    path reaches lower - margin; status 2: the frontier exceeded `backstop`.
    The all-state programme (tests) with the same child order, bound pruning, dominance
    decisions and kept messages: children are formed per parent from the shared
    background (O(S (t K + t^2)) each instead of O(S t Q)); a child's full message is
    materialised (_extend) only when it is kept or must be compared with a child of
    another parent. A child dominated by a kept path j, through an earlier sibling c
    dominated by j, is dropped when d(i, c) + d(c, j) <= tol clears the float margin
    (d is subadditive). Work arrays are reused across units (the frontier and the kept
    paths swap buffers); their sizes never change a value or a decision.
    """
    units = len(emissions)
    samples, states = start_n.shape
    width = units - unit0
    path = np.empty(width, np.int64)
    if width == 0:
        value = _join(start_n, start_c, end)
        if value < lower - margin:
            return INFEASIBLE, value, path, 1
        return OK, value, path, 1
    founders = len(fz)
    nb = len(bg)
    # The frontier (`parents` paths) and the paths kept on the current unit (`count`) live in
    # two buffers that swap roles after every unit.
    front_n = np.empty((1, samples, states))
    front_c = np.empty((1, samples))
    front_n[0] = start_n
    front_c[0] = start_c
    parents = 1
    kept_n = np.empty((0, samples, states))
    kept_c = np.empty((0, samples))
    history_par = np.empty(64, np.int64)
    history_cho = np.empty(64, np.int64)
    history_off = np.zeros(width + 1, np.int64)
    max_frontier = 1
    p_a0 = np.empty((0, samples, founders))
    p_bg = np.empty((0, samples, states))
    no_background = np.empty((0, states))
    c_m = np.empty((0, samples))
    c_f = np.empty((0, samples, founders))
    no_terms = np.empty(0)
    for u in range(unit0, units):
        emis = emissions[u]
        col = cols[u]
        bins = emis.shape[3]
        begin, stop = offsets[u], offsets[u + 1]
        rows = stop - begin
        children = parents * rows
        unext = upper[u + 1]
        prefix, suffix, seg = prefixes[u], suffixes[u], segs[u]
        # Unit tables of the bound and of the multi-bin dominance probes.
        te = np.full((samples, bins), -np.inf)
        ubmax = np.full(samples, -np.inf)
        probes = np.zeros((samples, bins), np.int64)
        for s in range(samples):
            best = np.full(bins, -np.inf)
            for b in range(nb):
                ub = unext[s, bg[b]]
                if ub > ubmax[s]:
                    ubmax[s] = ub
                for l in range(1, bins):
                    v = suffix[s, b, l] + ub
                    if v > te[s, l]:
                        te[s, l] = v
                    if suffix[s, b, l] > best[l]:
                        best[l] = suffix[s, b, l]
                        probes[s, l] = b
        ufz = _focal_values(unext, fz)
        ra, rb = _unit_rows(cands, begin, rows, col, focal, first, second, fz)
        # Per parent: its row-independent tables and, when cached (cache_bg), its children's
        # shared background (one-bin units: compared with other parents' kept paths without
        # being materialised).
        one_bin = bins == 1
        cache_bg = parents <= store_limit
        p_max = np.empty((parents, samples))
        p_pb = np.empty((parents, samples, bins))
        p_tp = np.empty((parents, samples))
        p_hi = np.empty((parents, samples))
        p_lo = np.empty((parents, samples))
        p_arg = np.empty((parents, samples), np.int64)
        p_a0 = _rows3(p_a0, parents, samples, founders)
        if cache_bg:
            p_bg = _rows3(p_bg, parents, samples, states)
        for p in range(parents):
            _parent(front_n[p], bg, fz, prefix, unext, penalty, p_max[p], p_pb[p], p_tp[p], p_hi[p],
                    p_lo[p], p_arg[p], p_a0[p], p_bg[p] if cache_bg else no_background)
        # The children's floor shortcut (_child, _parent_floor, _row_maxima).
        emax = _row_maxima(emis, ra, rb)
        p_amax = np.empty((parents, samples))
        p_fmb = np.empty((parents, samples, bins))
        p_fs = np.empty((parents, samples, bins))
        p_fm = np.empty((parents, samples))
        p_flo = np.empty((parents, samples))
        p_fd = np.empty((parents, samples))
        p_fcs = np.empty((parents, samples))
        p_fsb = np.empty((parents, samples))
        for p in range(parents):
            _parent_floor(front_c[p], p_max[p], p_pb[p], p_tp[p], p_a0[p], seg, te, ubmax, ufz, penalty,
                          p_amax[p], p_fmb[p], p_fs[p], p_fm[p], p_flo[p], p_fd[p], p_fcs[p], p_fsb[p])
        scratch_bg = np.empty((samples, states))
        bg_parent = -1
        # Lower-bound probes of the cross-parent tests: per parent and sample the background
        # state (and its index) where the parent's shared chain P_{t-1} is largest.
        p_probe = np.empty((parents, samples, 2), np.int64)
        for p in range(parents):
            for s in range(samples):
                p_probe[p, s, 0] = bg[p_arg[p, s]] if nb > 0 else fz[0]
                p_probe[p, s, 1] = p_arg[p, s]
        par = np.empty(children, np.int64)
        cho = np.empty(children, np.int64)
        totals = np.empty(children)
        bounds = np.empty(children)
        inc = np.empty(children, np.int64)
        # Same memory budget as the reference's stored child messages (store_limit of them).
        store = children * (founders + bins + 1) <= store_limit * states
        c_m = _rows2(c_m, children if store else 1, samples)
        c_f = _rows3(c_f, children if store else 1, samples, founders)
        c_s = np.empty((children if store else 1, samples, bins))
        for i in range(children):
            p = i // rows
            g = i % rows
            r = cands[begin + g]
            par[i] = p
            cho[i] = r
            inc[i] = 1 if r == incumbent[u] else 0
            slot = i if store else 0
            totals[i], bounds[i] = _child(
                front_c[p], p_max[p], p_pb[p], p_tp[p], p_a0[p], seg, te, ubmax, ufz, emis, ra[g],
                rb[g], penalty, c_m[slot], c_f[slot], c_s[slot], p_amax[p], emax[g], p_fmb[p],
                p_fs[p], p_fm[p], p_flo[p], p_fd[p], p_fcs[p], p_fsb[p], no_terms)
        alive = 0
        for i in range(children):
            if bounds[i] >= lower - margin:
                alive += 1
        if alive == 0:
            return INFEASIBLE, -np.inf, path, max_frontier
        order = _stable_order(totals, inc, par, cho)
        cap = min(alive, 64)
        if kept_n.shape[0] < cap:
            kept_n = np.empty((cap, samples, states))
            kept_c = np.empty((cap, samples))
        cap = kept_n.shape[0]
        kept_t = np.empty(cap)
        kept_i = np.empty(cap, np.int64)
        kept_lt = np.empty(cap)
        kept_m = np.empty((cap, samples))
        kept_f = np.empty((cap, samples, founders))
        kept_s = np.empty((cap, samples, bins))
        count = 0
        # Per parent: the first child dropped as dominated, its dominator and d.
        sib_child = np.full(parents, -1, np.int64)
        sib_kept = np.zeros(parents, np.int64)
        sib_d = np.zeros(parents)
        sib_m = np.empty((parents, samples))
        sib_f = np.empty((parents, samples, founders))
        sib_s = np.empty((parents, samples, bins))
        sib_t = np.zeros(parents)
        cur_m = np.empty(samples)
        cur_f = np.empty((samples, founders))
        cur_s = np.empty((samples, bins))
        mat_n = np.empty((samples, states))
        mat_c = np.empty(samples)
        for position in range(children):
            i = order[position]
            if bounds[i] < lower - margin:
                continue
            p = par[i]
            if store:
                cur_m = c_m[i]
                cur_f = c_f[i]
                cur_s = c_s[i]
            else:
                g = i % rows
                _child(front_c[p], p_max[p], p_pb[p], p_tp[p], p_a0[p], seg, te, ubmax, ufz, emis,
                       ra[g], rb[g], penalty, cur_m, cur_f, cur_s, p_amax[p], emax[g], p_fmb[p],
                       p_fs[p], p_fm[p], p_flo[p], p_fd[p], p_fcs[p], p_fsb[p], no_terms)
            materialised = False
            mat_t = 0.0
            dominated = False
            dom_j = -1
            dom_d = 0.0
            # (1) Through an earlier sibling c already dropped as dominated by kept j:
            # d(i, j) <= d(i, c) + d(c, j).
            if sib_child[p] >= 0:
                up, _ = _same_parent_bounds(
                    cur_m, cur_f, cur_s, totals[i], sib_m[p], sib_f[p], sib_s[p], sib_t[p], bins,
                    p_max[p], p_hi[p], p_lo[p], p_arg[p], front_n[p], bg, prefix, suffix, probes,
                    penalty, np.inf)
                if up + sib_d[p] <= dom_tol - _safety(totals[i], sib_t[p], kept_t[sib_kept[p]]):
                    dominated = True
            # (2) Kept children of the same parent: shortcut, else the reference test.
            if not dominated:
                for j in range(count):
                    if par[kept_i[j]] != p:
                        continue
                    safety = _safety(totals[i], kept_lt[j], 0.0)
                    up, low = _same_parent_bounds(
                        cur_m, cur_f, cur_s, totals[i], kept_m[j], kept_f[j], kept_s[j], kept_lt[j],
                        bins, p_max[p], p_hi[p], p_lo[p], p_arg[p], front_n[p], bg, prefix, suffix,
                        probes, penalty, dom_tol + safety)
                    if low > dom_tol + safety:
                        continue
                    if up <= dom_tol - safety:
                        dominated = True
                        dom_j = j
                        dom_d = up
                        break
                    if cache_bg:
                        background = p_bg[p]
                    else:
                        if bg_parent != p:
                            if one_bin:
                                _one_bin_background(front_n[p], p_max[p], emis, col, first,
                                                    second, bg, penalty, scratch_bg)
                            else:
                                _multi_bin_background(front_n[p], p_max[p], prefix, bg, penalty,
                                                      scratch_bg)
                            bg_parent = p
                        background = scratch_bg
                    verdict = -1
                    if nb > 0 and _cross_lower_bound(
                            cur_m, cur_f, totals[i], background, fz, p_probe[p], kept_n[j],
                            kept_t[j], penalty, one_bin, cur_s, suffix, bins,
                            dom_tol + _safety(totals[i], kept_t[j], 0.0)):
                        # The reference's sum exceeds tol (its terms are each at least these).
                        flag, value = False, dom_tol
                        verdict = 0
                    elif one_bin:
                        flag, value = _dominance_one_bin(cur_m, cur_f, totals[i], background, fz,
                                                         bg, kept_n[j], kept_t[j], dom_tol,
                                                         penalty)
                    else:
                        verdict = _dominance_multi_bin(cur_m, cur_f, cur_s, totals[i], background,
                                                       suffix, fz, bg, kept_n[j], kept_t[j],
                                                       dom_tol, penalty)
                        flag, value = verdict == 1, dom_tol
                    if verdict == -1 and not one_bin:
                        if not materialised:
                            mat_t = _extend(front_n[p], front_c[p], emis, col, focal, cho[i],
                                            first, second, penalty, mat_n, mat_c)
                            materialised = True
                        flag, value = _dominance_value(mat_n, mat_t, kept_n[j], kept_t[j],
                                                       dom_tol)
                    if flag:
                        dominated = True
                        dom_j = j
                        dom_d = value
                        break
            # (3) Kept children of other parents: the reference test on full messages (one
            # bin: on the shared form, the same values).
            if not dominated:
                for j in range(count):
                    if par[kept_i[j]] == p:
                        continue
                    if cache_bg:
                        background = p_bg[p]
                    else:
                        if bg_parent != p:
                            if one_bin:
                                _one_bin_background(front_n[p], p_max[p], emis, col, first,
                                                    second, bg, penalty, scratch_bg)
                            else:
                                _multi_bin_background(front_n[p], p_max[p], prefix, bg, penalty,
                                                      scratch_bg)
                            bg_parent = p
                        background = scratch_bg
                    verdict = -1
                    if nb > 0 and _cross_lower_bound(
                            cur_m, cur_f, totals[i], background, fz, p_probe[p], kept_n[j],
                            kept_t[j], penalty, one_bin, cur_s, suffix, bins,
                            dom_tol + _safety(totals[i], kept_t[j], 0.0)):
                        # The reference's sum exceeds tol (its terms are each at least these).
                        flag, value = False, dom_tol
                        verdict = 0
                    elif one_bin:
                        flag, value = _dominance_one_bin(cur_m, cur_f, totals[i], background, fz,
                                                         bg, kept_n[j], kept_t[j], dom_tol,
                                                         penalty)
                    else:
                        verdict = _dominance_multi_bin(cur_m, cur_f, cur_s, totals[i], background,
                                                       suffix, fz, bg, kept_n[j], kept_t[j],
                                                       dom_tol, penalty)
                        flag, value = verdict == 1, dom_tol
                    if verdict == -1 and not one_bin:
                        if not materialised:
                            mat_t = _extend(front_n[p], front_c[p], emis, col, focal, cho[i],
                                            first, second, penalty, mat_n, mat_c)
                            materialised = True
                        flag, value = _dominance_value(mat_n, mat_t, kept_n[j], kept_t[j],
                                                       dom_tol)
                    if flag:
                        dominated = True
                        dom_j = j
                        dom_d = value
                        break
            if dominated:
                if sib_child[p] < 0 and dom_j >= 0:
                    sib_child[p] = i
                    sib_kept[p] = dom_j
                    sib_d[p] = dom_d
                    sib_m[p] = cur_m
                    sib_f[p] = cur_f
                    sib_s[p] = cur_s
                    sib_t[p] = totals[i]
                continue
            if count == kept_n.shape[0]:
                grow = max(1, 2 * count)
                new_n = np.empty((grow, samples, states))
                new_c = np.empty((grow, samples))
                new_n[:count] = kept_n[:count]
                new_c[:count] = kept_c[:count]
                kept_n, kept_c = new_n, new_c
            if count == kept_t.shape[0]:
                grow = max(1, 2 * count)
                new_t = np.empty(grow)
                new_i = np.empty(grow, np.int64)
                new_lt = np.empty(grow)
                new_m = np.empty((grow, samples))
                new_f = np.empty((grow, samples, founders))
                new_s = np.empty((grow, samples, bins))
                new_t[:count] = kept_t[:count]
                new_i[:count] = kept_i[:count]
                new_lt[:count] = kept_lt[:count]
                new_m[:count] = kept_m[:count]
                new_f[:count] = kept_f[:count]
                new_s[:count] = kept_s[:count]
                kept_t, kept_i, kept_lt = new_t, new_i, new_lt
                kept_m, kept_f, kept_s = new_m, new_f, new_s
            if materialised:
                kept_n[count] = mat_n
                kept_c[count] = mat_c
            elif one_bin and cache_bg:
                mat_t = _assemble_one_bin(front_c[p], p_bg[p], bg, fz, cur_m, cur_f, penalty,
                                          kept_n[count], kept_c[count])
            else:
                mat_t = _extend(front_n[p], front_c[p], emis, col, focal, cho[i], first, second,
                                penalty, kept_n[count], kept_c[count])
            kept_t[count] = mat_t
            kept_i[count] = i
            kept_lt[count] = totals[i]
            kept_m[count] = cur_m
            kept_f[count] = cur_f
            kept_s[count] = cur_s
            count += 1
            if count > backstop:
                return BACKSTOP, -np.inf, path, count
        # A kept child may be dominated by one kept after it (later in the order).
        remove = np.zeros(count, np.bool_)
        for x in range(count):
            px = par[kept_i[x]]
            for y in range(x + 1, count):
                verdict = -1
                if par[kept_i[y]] == px:
                    safety = _safety(kept_lt[x], kept_lt[y], 0.0)
                    up, low = _same_parent_bounds(
                        kept_m[x], kept_f[x], kept_s[x], kept_lt[x], kept_m[y], kept_f[y], kept_s[y],
                        kept_lt[y], bins, p_max[px], p_hi[px], p_lo[px], p_arg[px], front_n[px], bg,
                        prefix, suffix, probes, penalty, dom_tol + safety)
                    if low > dom_tol + safety:
                        continue
                    if up <= dom_tol - safety:
                        verdict = 1
                if verdict == 1 or _dominance(kept_n[x], kept_t[x], kept_n[y], kept_t[y], dom_tol):
                    remove[x] = True
                    break
        survivors = count - int(remove.sum())
        hp = np.empty(survivors, np.int64)
        hc = np.empty(survivors, np.int64)
        w = 0
        for x in range(count):
            if remove[x]:
                continue
            if w != x:
                kept_n[w] = kept_n[x]
                kept_c[w] = kept_c[x]
            hp[w] = par[kept_i[x]]
            hc[w] = cho[kept_i[x]]
            w += 1
        # The survivors become the frontier; the former frontier's buffers hold the next unit's
        # kept paths.
        front_n, kept_n = kept_n, front_n
        front_c, kept_c = kept_c, front_c
        parents = survivors
        step = u - unit0
        need = history_off[step] + survivors
        if need > len(history_par):
            size = max(need, 2 * len(history_par))
            grown_par = np.empty(size, np.int64)
            grown_cho = np.empty(size, np.int64)
            grown_par[:history_off[step]] = history_par[:history_off[step]]
            grown_cho[:history_off[step]] = history_cho[:history_off[step]]
            history_par, history_cho = grown_par, grown_cho
        history_par[history_off[step]:need] = hp
        history_cho[history_off[step]:need] = hc
        history_off[step + 1] = need
        if survivors > max_frontier:
            max_frontier = survivors
    best = -1
    best_value = -np.inf
    for x in range(parents):
        value = _join(front_n[x], front_c[x], end)
        if value > best_value:
            best_value = value
            best = x
    if best_value < lower - margin:
        return INFEASIBLE, best_value, path, max_frontier
    index = best
    for step in range(width - 1, -1, -1):
        path[step] = history_cho[history_off[step] + index]
        index = history_par[history_off[step] + index]
    return OK, best_value, path, max_frontier


@njit(cache=True, nogil=True)
def path_value(emissions, cols, focal, rows, first, second, penalty, start_n, start_c, end):
    """H of one complete path, in the dynamic programme's arithmetic."""
    samples, states = start_n.shape
    n = start_n.copy()
    c = start_c.copy()
    out_n = np.empty((samples, states))
    out_c = np.empty(samples)
    for u in range(len(emissions)):
        _extend(n, c, emissions[u], cols[u], focal, rows[u], first, second, penalty, out_n, out_c)
        n[:] = out_n
        c[:] = out_c
    return _join(n, c, end)

@njit(cache=True, nogil=True)
def _closed_start(start, penalty):
    """The closed normalised start message (maximum 0, entries >= -penalty) and the
    per-sample maxima subtracted."""
    samples, states = start.shape
    out = np.empty((samples, states))
    top = np.empty(samples)
    for s in range(samples):
        m = row_max(start[s])
        top[s] = m
        for z in range(states):
            out[s, z] = max(start[s, z] - m, -penalty)
    return out, top

@njit(cache=True, nogil=True)
def _beam_value(emissions, cols, cands, offsets, focal, first, second, penalty, start_n, start_c,
                end, upper, width, fz, bg, prefixes, suffixes, segs):
    """H of the best path a beam finds: per unit only the `width` partial paths of largest
    total are extended (no dominance tests). A feasible value of the window problem
    (exactly the programme's arithmetic for that path), used to raise the pruning bound."""
    units = len(emissions)
    samples, states = start_n.shape
    founders = len(fz)
    beam_n = np.empty((1, samples, states))
    beam_c = np.empty((1, samples))
    beam_n[0] = start_n
    beam_c[0] = start_c
    no_terms = np.empty(0)
    for u in range(units):
        emis = emissions[u]
        col = cols[u]
        bins = emis.shape[3]
        begin, stop = offsets[u], offsets[u + 1]
        rows = stop - begin
        unext = upper[u + 1]
        prefix, suffix, seg = prefixes[u], suffixes[u], segs[u]
        te = np.full((samples, bins), -np.inf)
        ubmax = np.full(samples, -np.inf)
        for s in range(samples):
            for b in range(len(bg)):
                ub = unext[s, bg[b]]
                if ub > ubmax[s]:
                    ubmax[s] = ub
                for l in range(1, bins):
                    v = suffix[s, b, l] + ub
                    if v > te[s, l]:
                        te[s, l] = v
        parents = beam_n.shape[0]
        children = parents * rows
        totals = np.empty(children)
        lazy_m = np.empty(samples)
        lazy_f = np.empty((samples, founders))
        lazy_s = np.empty((samples, bins))
        ufz = _focal_values(unext, fz)
        ra, rb = _unit_rows(cands, begin, rows, col, focal, first, second, fz)
        for p in range(parents):
            pm, pbv, tpv, a0 = _parent_alone(beam_n[p], bg, fz, prefix, unext, penalty)
            for r in range(rows):
                totals[p * rows + r], _ = _child(
                    beam_c[p], pm, pbv, tpv, a0, seg, te, ubmax, ufz, emis, ra[r], rb[r], penalty,
                    lazy_m, lazy_f, lazy_s, NO_FLOOR, NO_FLOOR2, NO_FLOOR2, NO_FLOOR2, NO_FLOOR, NO_FLOOR,
                    NO_FLOOR, NO_FLOOR, NO_FLOOR, no_terms)
        order = np.argsort(-totals, kind="mergesort")
        keep = min(width, children)
        new_n = np.empty((keep, samples, states))
        new_c = np.empty((keep, samples))
        for x in range(keep):
            i = order[x]
            _extend(beam_n[i // rows], beam_c[i // rows], emis, col, focal, cands[begin + i % rows],
                    first, second, penalty, new_n[x], new_c[x])
        beam_n, beam_c = new_n, new_c
    best = -np.inf
    for x in range(beam_n.shape[0]):
        value = _join(beam_n[x], beam_c[x], end)
        if value > best:
            best = value
    return best


@njit(cache=True, nogil=True)
def window_box_dual(emissions, cols, cands, offsets, focal, first, second, penalty, end, fz, bg,
                    prefixes, suffixes, segs, lam):
    """window_box with multipliers: U[u] >= max over each sample's own choice of f's rows on
    units u.. of (their value plus the end flank plus sum_{v >= u} lam[s, g_v]), per sample.

    lam[s, g] is paid by sample s for choosing candidate g. If sum_s lam[s, g] = 0 for every
    candidate of the later units, the multipliers of a common completion cancel, so
    _join(child, U[u + 1]) bounds the child's completions as window_box does (lam = 0 is the
    box). Every state of row g's backward pass is shifted by lam[s, g]: the focal states and
    the row's switch maxima take it per row, the background no-switch chain the largest.
    """
    units = len(emissions)
    samples, states = end.shape
    founders = len(fz)
    nb = len(bg)
    upper = np.empty((units + 1, samples, states))
    upper[units] = end
    yf = np.empty(founders)
    for u in range(units - 1, -1, -1):
        emis = emissions[u]
        bins = emis.shape[3]
        col = cols[u]
        prefix, suffix, seg = prefixes[u], suffixes[u], segs[u]
        best_m = np.empty(bins)
        mrow = np.empty(bins)
        ue = np.empty(bins)
        begin = offsets[u]
        rows = offsets[u + 1] - begin
        ra, rb = _unit_rows(cands, begin, rows, col, focal, first, second, fz)
        start = np.empty(founders)
        top = np.empty(founders)
        for s in range(samples):
            for k in range(bins):
                ue[k] = -np.inf
            for b in range(nb):
                ub = upper[u + 1, s, bg[b]]
                for k in range(bins):
                    v = ub + suffix[s, b, k]
                    if v > ue[k]:
                        ue[k] = v
            for c in range(founders):
                start[c] = upper[u + 1, s, fz[c]]
                top[c] = -np.inf
            for k in range(bins):
                best_m[k] = -np.inf
            lam_max = -np.inf
            for g in range(rows):
                ga = ra[g]
                gb = rb[g]
                shift = lam[s, begin + g]
                if shift > lam_max:
                    lam_max = shift
                if bins == 1:
                    # The general step below for one bin, its passes merged (same operations and
                    # comparison order per value).
                    m = ue[0]
                    for c in range(founders):
                        y = start[c] + emis[s, ga[c], gb[c], 0]
                        yf[c] = y
                        if y > m:
                            m = y
                    sw = m - penalty
                    for c in range(founders):
                        y = yf[c]
                        if sw > y:
                            y = sw
                        v = y + shift
                        if v > top[c]:
                            top[c] = v
                    v = m + shift
                    if v > best_m[0]:
                        best_m[0] = v
                    continue
                for c in range(founders):
                    yf[c] = start[c]
                for k in range(bins - 1, -1, -1):
                    m = ue[k]
                    for l in range(k + 1, bins):
                        v = mrow[l] - penalty + seg[s, k, l - 1]
                        if v > m:
                            m = v
                    for c in range(founders):
                        yf[c] += emis[s, ga[c], gb[c], k]
                        if yf[c] > m:
                            m = yf[c]
                    mrow[k] = m
                    sw = m - penalty
                    for c in range(founders):
                        if sw > yf[c]:
                            yf[c] = sw
                for c in range(founders):
                    v = yf[c] + shift
                    if v > top[c]:
                        top[c] = v
                for k in range(bins):
                    v = mrow[k] + shift
                    if v > best_m[k]:
                        best_m[k] = v
            for c in range(founders):
                upper[u, s, fz[c]] = top[c]
            for b in range(nb):
                z = bg[b]
                v = upper[u + 1, s, z] + suffix[s, b, 0] + lam_max
                for k in range(bins):
                    w = best_m[k] - penalty + (prefix[s, b, k - 1] if k > 0 else 0.0)
                    if w > v:
                        v = w
                upper[u, s, z] = v
    return upper


@njit(cache=True, nogil=True)
def _dual_sweep(emissions, cols, cands, offsets, focal, incumbent, first, second, penalty, start_n,
                start_c, upper, lam, fz, bg, prefixes, suffixes, segs, decoded):
    """One forward block-coordinate sweep of the Lagrangian dual of the window problem.

    Every sample keeps its own copy of f's rows and pays lam[s, g] for candidate g (sum_s
    lam[s, g] = 0). Unit by unit: theta[g, s] = sample s's max-marginal of candidate g (its
    best relaxed value with f's row of unit u fixed to g: its relaxed forward message, unit u
    with row g, and the suffix bound `upper` of the later units, without unit u's
    multipliers); unit u's multipliers become mean_s theta[g] - theta[g, s] (the last sample
    takes minus the others' sum), the exact minimiser of the dual over them; decoded[u] = the
    candidate of largest mean (ties: the incumbent's row, then the first); the relaxed forward
    message advances through unit u, each state taking its best candidate with the new
    multipliers. `upper` = window_box_dual of the multipliers at the start of the sweep (the
    later units are not yet updated when unit u is). Children are formed as in the
    programme (_child); the background states of the advanced message from the
    parent's shared chain and the rows' switch floors.
    """
    units = len(emissions)
    samples, states = start_n.shape
    founders = len(fz)
    nb = len(bg)
    an = start_n.copy()
    ac = start_c.copy()
    new = np.empty(states)
    # The parent's shared background and the children's floor tables (_parent, _parent_floor), per unit.
    background = np.empty((samples, states))
    amax = np.empty(samples)
    fm = np.empty(samples)
    flo = np.empty(samples)
    fd = np.empty(samples)
    fcs = np.empty(samples)
    fsb = np.empty(samples)
    for u in range(units):
        emis = emissions[u]
        col = cols[u]
        bins = emis.shape[3]
        begin = offsets[u]
        rows = offsets[u + 1] - begin
        unext = upper[u + 1]
        prefix, suffix, seg = prefixes[u], suffixes[u], segs[u]
        te = np.full((samples, bins), -np.inf)
        ubmax = np.full(samples, -np.inf)
        for s in range(samples):
            for b in range(nb):
                ub = unext[s, bg[b]]
                if ub > ubmax[s]:
                    ubmax[s] = ub
                for l in range(1, bins):
                    v = suffix[s, b, l] + ub
                    if v > te[s, l]:
                        te[s, l] = v
        pm = np.empty(samples)
        pbv = np.empty((samples, bins))
        tpv = np.empty(samples)
        a0 = np.empty((samples, founders))
        _parent(an, bg, fz, prefix, unext, penalty, pm, pbv, tpv, np.empty(samples), np.empty(samples),
                np.empty(samples, np.int64), a0, background)
        ufz = _focal_values(unext, fz)
        ra, rb = _unit_rows(cands, begin, rows, col, focal, first, second, fz)
        emax = _row_maxima(emis, ra, rb)
        fmb = np.empty((samples, bins))
        fs = np.empty((samples, bins))
        _parent_floor(ac, pm, pbv, tpv, a0, seg, te, ubmax, ufz, penalty, amax, fmb, fs, fm, flo, fd, fcs,
                      fsb)
        rm = np.empty((rows, samples))
        rf = np.empty((rows, samples, founders))
        rs = np.empty((rows, samples, bins))
        theta = np.empty((rows, samples))
        for g in range(rows):
            # theta[g, s]: the child's per-sample join with U[u + 1] (its bound terms).
            _child(ac, pm, pbv, tpv, a0, seg, te, ubmax, ufz, emis, ra[g], rb[g], penalty, rm[g], rf[g],
                   rs[g], amax, emax[g], fmb, fs, fm, flo, fd, fcs, fsb, theta[g])
        chosen = 0
        top = -np.inf
        for g in range(rows):
            mean = 0.0
            for s in range(samples):
                mean += theta[g, s]
            mean /= samples
            rest = 0.0
            for s in range(samples - 1):
                lam[s, begin + g] = mean - theta[g, s]
                rest += lam[s, begin + g]
            lam[samples - 1, begin + g] = -rest
            if mean > top or (mean == top and cands[begin + g] == incumbent[u]):
                top = mean
                chosen = g
        decoded[u] = cands[begin + chosen]
        for s in range(samples):
            lam_max = -np.inf
            floor = -np.inf
            for g in range(rows):
                x = lam[s, begin + g]
                if x > lam_max:
                    lam_max = x
                y = x + rm[g, s]
                if y > floor:
                    floor = y
            floor -= penalty
            for c in range(founders):
                v = -np.inf
                for g in range(rows):
                    y = lam[s, begin + g] + rm[g, s] + rf[g, s, c]
                    if y > v:
                        v = y
                new[fz[c]] = v
            for b in range(nb):
                z = bg[b]
                # max(N[z], pmax - pen) + E_{0..t-1}[z]: the parent's shared background (_parent).
                v = background[s, z] + lam_max
                if bins > 1:
                    for l in range(1, bins):
                        w = -np.inf
                        for g in range(rows):
                            y = lam[s, begin + g] + rs[g, s, l]
                            if y > w:
                                w = y
                        w += suffix[s, b, l]
                        if w > v:
                            v = w
                if floor > v:
                    v = floor
                new[z] = v
            mx = row_max(new)
            for z in range(states):
                an[s, z] = new[z] - mx
            ac[s] += mx


@njit(cache=True, nogil=True)
def _dual_bounds(emissions, cols, cands, offsets, focal, incumbent, first, second, penalty, start_n,
                 start_c, end, feasible, tol, fz, bg, prefixes, suffixes, segs):
    """Lagrangian suffix bounds of one window problem for a programme's restart.

    Forward block-coordinate sweeps from zero multipliers (the box); each sweep's decoded
    consensus path is a feasible value (path_value, the programme's arithmetic). Stops when
    the dual bound is within tol of the best feasible value or after DUAL_SWEEPS. Returns
    (upper, slack, best feasible value, dual bound, sweeps): upper = window_box_dual of the
    final multipliers; slack covers
    their float residual from zero sum (sum over units of the largest |sum_s lam| of a
    candidate, plus a rounding allowance) and is added to every pruning margin that uses it.
    """
    units = len(emissions)
    samples = start_n.shape[0]
    lam = np.zeros((samples, len(cands)))
    decoded = np.empty(units, np.int64)
    upper = window_box_dual(emissions, cols, cands, offsets, focal, first, second, penalty, end, fz,
                            bg, prefixes, suffixes, segs, lam)
    bound = _join(start_n, start_c, upper[0])
    best = feasible
    sweeps = 0
    # The last decoded path and its value: a sweep that decodes the same path has the same value.
    held = np.full(units, -1, np.int64)
    held_value = -np.inf
    while sweeps < DUAL_SWEEPS and bound > best + tol:
        _dual_sweep(emissions, cols, cands, offsets, focal, incumbent, first, second, penalty,
                    start_n, start_c, upper, lam, fz, bg, prefixes, suffixes, segs, decoded)
        sweeps += 1
        same = True
        for u in range(units):
            if decoded[u] != held[u]:
                same = False
        if not same:
            held_value = path_value(emissions, cols, focal, decoded, first, second, penalty, start_n,
                                    start_c, end)
            held[:] = decoded
        value = held_value
        if value > best:
            best = value
        upper = window_box_dual(emissions, cols, cands, offsets, focal, first, second, penalty, end,
                                fz, bg, prefixes, suffixes, segs, lam)
        bound = _join(start_n, start_c, upper[0])
    slack = 0.0
    for u in range(units):
        worst = 0.0
        scale = 0.0
        for g in range(offsets[u], offsets[u + 1]):
            total = 0.0
            for s in range(samples):
                total += lam[s, g]
                if abs(lam[s, g]) > scale:
                    scale = abs(lam[s, g])
            if abs(total) > worst:
                worst = abs(total)
        slack += worst + 1e-12 * samples * scale
    return upper, slack, best, bound, sweeps


@njit(cache=True, nogil=True)
def _tighten(emissions, cols, cands, offsets, focal, incumbent, first, second, penalty, start_n,
             start_c, end, upper, lower, tol, fz, bg, prefixes, suffixes, segs):
    """Tighter pruning for the restart of a programme whose frontier passed EXPLORE_FRONTIER:
    the Lagrangian suffix bounds (_dual_bounds) and the best feasible value among `lower`, a
    beam (_beam_value) and the dual's decoded paths, each less 1e-9 relative. Returns
    (upper, lower, margin, slack); margin is solve's pruning margin plus the slack."""
    units = len(emissions)
    hint = _beam_value(emissions, cols, cands, offsets, focal, first, second, penalty, start_n,
                       start_c, end, upper, BEAM_WIDTH, fz, bg, prefixes, suffixes, segs)
    lower = max(lower, hint - 1e-9 * max(1.0, abs(hint)))
    dual_upper, slack, found, _, _ = _dual_bounds(
        emissions, cols, cands, offsets, focal, incumbent, first, second, penalty, start_n, start_c,
        end, lower, tol, fz, bg, prefixes, suffixes, segs)
    lower = max(lower, found - 1e-9 * max(1.0, abs(found)))
    margin = tol + (units + 1) * DOMINANCE_TOLERANCE + 1e-9 * max(1.0, abs(lower)) + slack
    return dual_upper, lower, margin, slack


@njit(cache=True, nogil=True)
def _restrict_unit(cands, offsets, u, rows):
    """cands / offsets with unit u's alphabet replaced by rows."""
    units = len(offsets) - 1
    delta = len(rows) - (offsets[u + 1] - offsets[u])
    offsets2 = offsets.copy()
    for v in range(u + 1, units + 1):
        offsets2[v] = offsets[v] + delta
    cands2 = np.empty(offsets2[units], np.int64)
    cands2[:offsets[u]] = cands[:offsets[u]]
    cands2[offsets[u]:offsets2[u + 1]] = rows
    cands2[offsets2[u + 1]:] = cands[offsets[u + 1]:]
    return cands2, offsets2


@njit(cache=True, nogil=True)
def _restricted_programme(emissions, cols, cands, offsets, focal, incumbent, first, second,
                          penalty, start_n, start_c, end, alphabet, alphabet_offsets, prefix_n,
                          prefix_c, unit0, upper, target, backstop, store_limit, fz, bg, prefixes,
                          suffixes, segs, slack, tightened, optimum, tol):
    """One canonical-tie programme (from a partial path on units unit0.., alphabet given by
    alphabet/alphabet_offsets, pruned below target, dominance tolerance 0). Until the
    problem's Lagrangian bounds exist (tightened False) it runs with EXPLORE_FRONTIER as its
    backstop; a programme that passes it computes them once for the problem (_dual_bounds,
    full alphabet: valid suffix bounds for every partial path) and reruns. Returns (status,
    found, tail, frontier, upper, slack, tightened)."""
    limit = backstop if tightened else min(backstop, EXPLORE_FRONTIER)
    status, found, tail, reached = window_dp(
        emissions, cols, alphabet, alphabet_offsets, focal, incumbent, first, second, penalty,
        prefix_n.copy(), prefix_c.copy(), unit0, end, upper, target,
        1e-9 * max(1.0, abs(target)) + slack, 0.0, limit, store_limit, fz, bg, prefixes, suffixes,
        segs)
    if status == BACKSTOP and not tightened and backstop > EXPLORE_FRONTIER:
        upper, slack, _, _, _ = _dual_bounds(
            emissions, cols, cands, offsets, focal, incumbent, first, second, penalty, start_n,
            start_c, end, optimum, tol, fz, bg, prefixes, suffixes, segs)
        tightened = True
        status, found, tail, again = window_dp(
            emissions, cols, alphabet, alphabet_offsets, focal, incumbent, first, second, penalty,
            prefix_n.copy(), prefix_c.copy(), unit0, end, upper, target,
            1e-9 * max(1.0, abs(target)) + slack, 0.0, backstop, store_limit, fz, bg, prefixes,
            suffixes, segs)
        if again > reached:
            reached = again
    return status, found, tail, reached, upper, slack, tightened


@njit(cache=True, nogil=True)
def _canonical_restricted(emissions, cols, cands, offsets, focal, incumbent, first, second,
                          penalty, start_n, start_c, end, upper, path, target, backstop,
                          store_limit, fz, bg, prefixes, suffixes, segs, slack, tightened,
                          optimum, tol):
    """The lexicographically first path reaching target by restricted programmes (solve's
    unit-by-unit rule): rows ordered before the current choice are tried in order; a row
    whose suffix bound (the box, or the Lagrangian bounds once computed, whose float slack
    widens every margin) cannot reach the target is skipped without a programme. When
    several rows remain, one programme with unit u's alphabet restricted to them comes
    first: it finds the best completion over all of them, so if it misses the target every
    row alone does too and none is tried. Every programme takes the EXPLORE_FRONTIER restart
    of _restricted_programme until the Lagrangian bounds exist (tightened: the main
    programme restarted; optimum = its value). Returns (status, path, programmes run,
    frontier)."""
    units = len(emissions)
    samples, states = start_n.shape
    founders = len(fz)
    best = path.copy()
    prefix_n = start_n.copy()
    prefix_c = start_c.copy()
    out_n = np.empty_like(prefix_n)
    out_c = np.empty_like(prefix_c)
    lazy_m = np.empty(samples)
    lazy_f = np.empty((samples, founders))
    row_a = np.empty(founders, np.int64)
    row_b = np.empty(founders, np.int64)
    no_terms = np.empty(0)
    fixing = 0
    frontier = 0
    for u in range(units):
        skip_below = target - 1e-9 * max(1.0, abs(target)) - 1e-9 - 1e-12 * abs(target) - slack
        held = incumbent[u]
        count = offsets[u + 1] - offsets[u]
        emis = emissions[u]
        bins = emis.shape[3]
        unext = upper[u + 1]
        prefix, suffix, seg = prefixes[u], suffixes[u], segs[u]
        lazy_s = np.empty((samples, bins))
        te = np.full((samples, bins), -np.inf)
        ubmax = np.full(samples, -np.inf)
        for s in range(samples):
            for b in range(len(bg)):
                ub = unext[s, bg[b]]
                if ub > ubmax[s]:
                    ubmax[s] = ub
                for l in range(1, bins):
                    v = suffix[s, b, l] + ub
                    if v > te[s, l]:
                        te[s, l] = v
        pm, pbv, tpv, a0 = _parent_alone(prefix_n, bg, fz, prefix, unext, penalty)
        ufz = _focal_values(unext, fz)
        tests = np.empty(count + 1, np.int64)
        n_tests = 0
        for position in range(count + 1):
            if position == 0:
                r = held
            else:
                r = cands[offsets[u] + position - 1]
                if r == held:
                    continue
            if r == best[u]:
                break
            _row_pairs(r, cols[u], focal, first, second, fz, row_a, row_b)
            _, bound = _child(prefix_c, pm, pbv, tpv, a0, seg, te, ubmax, ufz, emis, row_a, row_b,
                              penalty, lazy_m, lazy_f, lazy_s, NO_FLOOR, NO_FLOOR2, NO_FLOOR2,
                              NO_FLOOR2, NO_FLOOR, NO_FLOOR, NO_FLOOR, NO_FLOOR, NO_FLOOR, no_terms)
            if bound < skip_below:
                continue
            tests[n_tests] = r
            n_tests += 1
        reachable = True
        if n_tests > 1:
            cands2, offsets2 = _restrict_unit(cands, offsets, u, tests[:n_tests])
            fixing += 1
            status, found, _, reached, upper, slack, tightened = _restricted_programme(
                emissions, cols, cands, offsets, focal, incumbent, first, second, penalty,
                start_n, start_c, end, cands2, offsets2, prefix_n, prefix_c, u, upper, target,
                backstop, store_limit, fz, bg, prefixes, suffixes, segs, slack, tightened,
                optimum, tol)
            if status == BACKSTOP:
                return BACKSTOP, best, fixing, reached
            if reached > frontier:
                frontier = reached
            reachable = status == OK and found >= target
        for x in range(n_tests if reachable else 0):
            r = tests[x]
            _extend(prefix_n, prefix_c, emis, cols[u], focal, r, first, second, penalty, out_n,
                    out_c)
            fixing += 1
            status, found, tail, reached, upper, slack, tightened = _restricted_programme(
                emissions, cols, cands, offsets, focal, incumbent, first, second, penalty,
                start_n, start_c, end, cands, offsets, out_n, out_c, u + 1, upper, target,
                backstop, store_limit, fz, bg, prefixes, suffixes, segs, slack, tightened,
                optimum, tol)
            if status == BACKSTOP:
                return BACKSTOP, best, fixing, reached
            if reached > frontier:
                frontier = reached
            if status == OK and found >= target:
                best[u] = r
                best[u + 1:] = tail
                break
        _extend(prefix_n, prefix_c, emis, cols[u], focal, best[u], first, second, penalty, out_n,
                out_c)
        prefix_n[:] = out_n
        prefix_c[:] = out_c
    return OK, best, fixing, frontier


@njit(cache=True, nogil=True)
def solve_focal(emissions, cols, cands, offsets, focal, incumbent, first, second, penalty,
                start_n, start_c, end, incumbent_value, tables, backstop, store_limit):
    """The canonical exact maximiser of one founder's window problem (module docstring).

    start_n, start_c: the closed start message (_closed_start); incumbent_value: H of the panel's
    own rows; tables: the window's per-unit _state_tables (all three shared by the window's focal
    problems). Returns (status, rows, box_certified, dp, frontier, fixing_dps); status UNCHANGED =
    the incumbent is the canonical answer, CHANGED = rows is a different canonical answer,
    FRONTIER_LIMIT = a frontier exceeded `backstop` (the caller keeps the incumbent), LOST = the
    programme lost every feasible path (an internal error). The canonical-tie order: incumbent
    row first, then ascending index.
    """
    units = len(emissions)
    rows = np.empty(units, np.int64)
    fz, bg = _focal_layout(focal, first, second)
    prefixes = List()
    suffixes = List()
    segs = List()
    for u in range(units):
        P, T, best, bx, by, mx, my = tables[u]
        prefix, suffix, seg = _unit_tables(P, T, best, bx, by, mx, my, focal, bg)
        prefixes.append(prefix)
        suffixes.append(suffix)
        segs.append(seg)
    upper = window_box(emissions, cols, cands, offsets, focal, first, second, penalty, end, fz, bg,
                       prefixes, suffixes, segs)
    tol = 1e-7 + 1e-9 * max(1.0, abs(incumbent_value))
    certificate = (incumbent_value + tol
                   - BOX_ARITHMETIC_MARGIN * max(1.0, abs(incumbent_value)))
    if _join(start_n, start_c, upper[0]) <= certificate:
        return UNCHANGED, rows, True, False, 0, 0
    lower = incumbent_value
    margin = tol + (units + 1) * DOMINANCE_TOLERANCE + 1e-9 * max(1.0, abs(lower))
    status, value, path, frontier = window_dp(
        emissions, cols, cands, offsets, focal, incumbent, first, second, penalty, start_n,
        start_c, 0, end, upper, lower, margin, DOMINANCE_TOLERANCE, min(backstop, EXPLORE_FRONTIER),
        store_limit, fz, bg, prefixes, suffixes, segs)
    slack = 0.0
    tightened = False
    if status == BACKSTOP and backstop > EXPLORE_FRONTIER:
        # A large frontier: restart with the Lagrangian suffix bounds and the best feasible
        # value of a beam and the dual's decoded paths.
        tightened = True
        upper, lower, margin, slack = _tighten(
            emissions, cols, cands, offsets, focal, incumbent, first, second, penalty, start_n,
            start_c, end, upper, lower, tol, fz, bg, prefixes, suffixes, segs)
        status, value, path, reached = window_dp(
            emissions, cols, cands, offsets, focal, incumbent, first, second, penalty, start_n,
            start_c, 0, end, upper, lower, margin, DOMINANCE_TOLERANCE, backstop, store_limit, fz,
            bg, prefixes, suffixes, segs)
        if reached > frontier:
            frontier = reached
    if status == BACKSTOP:
        return FRONTIER_LIMIT, rows, False, True, frontier, 0
    if status != OK:
        return LOST, rows, False, True, frontier, 0
    if incumbent_value >= value - tol:
        return UNCHANGED, rows, False, True, frontier, 0
    # Canonical ties: the lexicographically first path within tol of the optimum.
    target = value - tol
    status, best, fixing, reached = _canonical_restricted(
        emissions, cols, cands, offsets, focal, incumbent, first, second, penalty, start_n,
        start_c, end, upper, path, target, backstop, store_limit, fz, bg, prefixes, suffixes,
        segs, slack, tightened, value, tol)
    if reached > frontier:
        frontier = reached
    if status == BACKSTOP:
        return FRONTIER_LIMIT, rows, False, True, frontier, fixing
    changed = False
    for u in range(units):
        if best[u] != incumbent[u]:
            changed = True
    rows[:] = best
    return (CHANGED if changed else UNCHANGED), rows, False, True, frontier, fixing


@njit(cache=True, nogil=True)
def solve_template(emissions, cols, sizes, focals, penalty, start, end, backstop, store_limit):
    """Every listed focal founder's problem of one window (shared emissions and flanks).

    cols[u, k] is founder k's row on unit u (the focal's incumbent); the alphabet of
    unit u is 0..sizes[u]-1. backstop: the largest frontier a programme may hold.
    Returns per focal (status, rows, box_certified, dp, frontier, fixing_dps) as arrays.
    """
    units = len(emissions)
    founders = cols.shape[1]
    offsets = np.zeros(units + 1, np.int64)
    for u in range(units):
        offsets[u + 1] = offsets[u] + sizes[u]
    cands = np.empty(offsets[units], np.int64)
    for u in range(units):
        for r in range(sizes[u]):
            cands[offsets[u] + r] = r
    states = founders * (founders + 1) // 2
    first = np.empty(states, np.int64)
    second = np.empty(states, np.int64)
    q = 0
    for a in range(founders):
        for b in range(a, founders):
            first[q] = a
            second[q] = b
            q += 1
    n = len(focals)
    status = np.zeros(n, np.int64)
    rows = np.zeros((n, units), np.int64)
    box = np.zeros(n, np.bool_)
    dp = np.zeros(n, np.bool_)
    frontier = np.zeros(n, np.int64)
    fixing = np.zeros(n, np.int64)
    if n == 0:
        return status, rows, box, dp, frontier, fixing
    # Shared by every focal problem: the closed start message, the value of the panel's own rows
    # (an incumbent path is the panel itself whatever the focal founder, so path_value performs
    # the same operations for every focal) and the per-state unit sums.
    start_n, start_c = _closed_start(start, penalty)
    k0 = focals[0]
    incumbent_value = path_value(emissions, cols, k0, np.ascontiguousarray(cols[:, k0]), first,
                                 second, penalty, start_n, start_c, end)
    tables = List()
    for u in range(units):
        tables.append(_state_tables(emissions[u], cols[u], first, second))
    for i in range(n):
        k = focals[i]
        incumbent = np.ascontiguousarray(cols[:, k])
        st, r, bx, used, fr, fx = solve_focal(emissions, cols, cands, offsets, k, incumbent, first,
                                             second, penalty, start_n, start_c, end,
                                             incumbent_value, tables, backstop, store_limit)
        status[i] = st
        rows[i] = r
        box[i] = bx
        dp[i] = used
        frontier[i] = fr
        fixing[i] = fx
    return status, rows, box, dp, frontier, fixing

@njit(cache=True, nogil=True)
def slice_list(items, a, c):
    """Typed list of items[a:c] (without a Python-level list construction)."""
    out = List()
    for i in range(a, c):
        out.append(items[i])
    return out



@njit(cache=True, parallel=True, nogil=True)
def gather_macro(emissions, groups, rows_by_group):
    """Coarse-unit emissions: for L1 span groups[g] = (first, end) prepared blocks and its
    alphabet rows_by_group[g] (A_g row tuples over the span), out[g][s, a, b, :] concatenates
    the blocks' binned emissions of the row pairs (rows[a, block], rows[b, block]).

    One task per group, which reads each source block of its group from the typed List once and
    copies it for every sample: every List element read updates a reference count shared by all
    threads, including the window workers that gather their problems concurrently from the
    component's one List, so reads per (group, sample, block) serialised them.
    """
    outputs = List()
    for group in range(len(groups)):
        start, stop = groups[group]
        bins = 0
        for block in range(start, stop):
            bins += emissions[block].shape[3]
        count = len(rows_by_group[group])
        outputs.append(np.empty((emissions[0].shape[0], count, count, bins)))
    samples = emissions[0].shape[0]
    for group_index in prange(len(groups)):
        group = np.int64(group_index)
        rows, output = rows_by_group[group], outputs[group]
        start, stop = groups[group]
        offset = 0
        for block in range(start, stop):
            value = emissions[block]
            for sample in range(samples):
                for a in range(len(rows)):
                    aa = rows[a, block - start]
                    for b in range(len(rows)):
                        bb = rows[b, block - start]
                        for site in range(value.shape[3]):
                            output[sample, a, b, offset + site] = value[sample, aa, bb, site]
            offset += value.shape[3]
    return outputs


# ------------------------------------------------------------------ Python level
def window_templates(blocks, groups):
    """Fine windows over blocks and coarse windows over L1 spans: [(kind, first unit, end unit)]."""
    tasks = []
    if FINE_WINDOW >= blocks:
        starts = [0]
    else:
        starts = sorted(set(range(0, blocks - FINE_WINDOW + 1, FINE_STRIDE)) | {blocks - FINE_WINDOW})
    tasks.extend(("fine", a, min(blocks, a + FINE_WINDOW)) for a in starts)
    if groups is not None:
        units = len(groups)
        if COARSE_WINDOW >= units:
            starts = [0]
        else:
            starts = sorted(set(range(0, units - COARSE_WINDOW + 1, COARSE_STRIDE))
                            | {units - COARSE_WINDOW})
        tasks.extend(("coarse", i, min(units, i + COARSE_WINDOW)) for i in starts)
    return tasks


# A window's programmes are first solved within a first budget (covered by the window solve's own memory
# estimate, WindowSolver.template_bytes); programmes whose frontier passes it are solved again within budgets
# PROGRAMME_GROWTH times larger, each reserved from the run's memory governor (execution.memory), up to
# FounderRefinementConfig.frontier_bytes. A solve that stays within a budget computes exactly what a solve at the
# cap computes, so results never depend on the budgets or on pacing; a repeat costs at most the passed budget's
# work. The first budget holds a frontier just past the EXPLORE_FRONTIER restart trigger (every reference and
# K = 30 frontier so far stopped there, at 257 paths), and at least PROGRAMME_FIRST_BYTES; a window solve's own
# estimate plans for PLANNING_FRONTIER paths (most programmes stay far below the trigger or are certified).
PROGRAMME_FIRST_BYTES = 256 << 20
PROGRAMME_GROWTH = 8
PLANNING_FRONTIER = EXPLORE_FRONTIER // 4 + 1


def programme_budgets(frontier_bytes, samples, founders, bins):
    """The frontier budgets (bytes) a window's programmes are solved within, smallest first, ending at the cap."""
    entry = 4 * 8 * samples * (founders * (founders + 1) // 2 + founders + bins + 2)
    budgets = [min(int(frontier_bytes), max(PROGRAMME_FIRST_BYTES, entry * (EXPLORE_FRONTIER + 1)))]
    while budgets[-1] < frontier_bytes:
        budgets.append(min(int(frontier_bytes), budgets[-1] * PROGRAMME_GROWTH))
    return budgets


def frontier_limit(frontier_bytes, samples, founders, bins):
    """Largest frontier (kept partial paths per unit) a programme may hold within frontier_bytes
    (None: no cap): a kept path stores S x (Q + K + bins + 2) float64 values, and
    growth and the next unit's copy hold up to four times the frontier at once."""
    if frontier_bytes is None:
        return 2 ** 62
    states = founders * (founders + 1) // 2
    entry = 8 * samples * (states + founders + bins + 2)
    return max(1, int(frontier_bytes) // (4 * entry))


class WindowSolver:
    """The window problems of one component: templates, coarse alphabets and their exact solves.

    arrays: the prepared blocks' binned emissions float64 (S, R_b, R_b, bins_b) (workspace).
    context: None, or (groups, paths) of the L1 context (stage.l1_context): groups[u] = (first,
    end) prepared blocks of L1 span u (a partition of the component), paths[u] = int64[P_u, end -
    first] the distinct L1 row paths offered on that span. penalty: SWITCH_PENALTY. Every solve
    takes frontier_bytes, the memory one programme's frontier may hold (frontier_limit).
    """

    def __init__(self, arrays, context, penalty):
        self.arrays = list(arrays)
        # One typed List for the kernels (built once: its construction copies every reference).
        self.emissions = List(self.arrays)
        self.blocks = len(self.arrays)
        self.penalty = float(penalty)
        self.coarse = None
        if context is not None:
            groups, paths = context
            self.coarse = ([tuple(int(x) for x in group) for group in groups],
                           [np.ascontiguousarray(path, np.int64) for path in paths])
        self.templates = window_templates(self.blocks,
                                          None if self.coarse is None else self.coarse[0])
        # Block range [a, c) of every template (proposal caches hold flank messages there).
        self.spans = [(a, c) if kind == "fine" else (self.coarse[0][a][0], self.coarse[0][c - 1][1])
                      for kind, a, c in self.templates]

    @property
    def coarse_units(self):
        return None if self.coarse is None else len(self.coarse[0])

    def template_bytes(self, kind, a, c, samples, founders):
        """Planned bytes of one window solve (the search's item estimate, execution.memory): the window's per-state
        unit sums (_state_tables) and one focal problem's unit tables (each two per unit and bin plus one per unit)
        and box bounds of S x Q float64, a planned frontier of PLANNING_FRONTIER paths (two and a half copies of S
        (Q + K + bins + 2) float64 each) and, for coarse windows, the gathered span emissions (S x A^2 x bins
        float64 per unit, A <= founders + context paths) and their multi-bin tables. Growth past the first frontier
        budget reserves its own bytes (programme_budgets)."""
        states = founders * (founders + 1) // 2
        if kind == "fine":
            unit_bins = [int(array.shape[3]) for array in self.arrays[a:c]]
            alphabets = None
        else:
            groups, context = self.coarse
            unit_bins = [sum(int(self.arrays[b].shape[3]) for b in range(*groups[u])) for u in range(a, c)]
            alphabets = [founders + len(context[u]) for u in range(a, c)]
        bins = max(unit_bins)
        tables = 8 * samples * states * (4 * sum(unit_bins) + 2 * len(unit_bins) + 1)
        frontier = 20 * samples * (states + founders + bins + 2) * PLANNING_FRONTIER
        coarse = 0 if alphabets is None else sum(8 * samples * width * width * unit + 16 * samples * states * unit
                                                 for width, unit in zip(alphabets, unit_bins))
        return tables + frontier + coarse

    def _template(self, emissions, bins, cols, sizes, start, end, frontier_bytes, focals=None):
        """Every (listed) founder's problem of one window in one nogil call (bins: the largest
        unit's bins). Returns [(focal, unit choices or None, diagnostic)]; a frontier fallback
        keeps the incumbent."""
        founders = cols.shape[1]
        focals = (np.arange(founders, dtype=np.int64) if focals is None
                  else np.asarray(focals, np.int64))
        samples, states = start.shape
        store_limit = max(1, STORE_BYTES // max(1, 8 * samples * states))
        budgets = programme_budgets(frontier_bytes, samples, founders, bins)
        cols = np.ascontiguousarray(cols, np.int64)
        sizes = np.asarray(sizes, np.int64)
        solved, pending = {}, focals
        for step, budget_bytes in enumerate(budgets):
            final = step == len(budgets) - 1
            limit = frontier_limit(budget_bytes, samples, founders, bins)
            with (nullcontext() if step == 0 else
                  execution.reserve_memory(budget_bytes, label="founder_refinement window frontier")):
                status, rows, box, dp, frontier, fixing = solve_template(
                    emissions, cols, sizes, pending, self.penalty, start, end, int(limit), int(store_limit))
            again = []
            for i, k in enumerate(pending):
                if status[i] == FRONTIER_LIMIT and not final:
                    again.append(k)
                else:
                    solved[int(k)] = (status[i], rows[i].copy(), box[i], dp[i], frontier[i], fixing[i])
            if not again:
                break
            pending = np.asarray(again, np.int64)
        out = []
        for k in focals:
            status, rows, box, dp, frontier, fixing = solved[int(k)]
            if status == LOST:
                raise RuntimeError("window dynamic programme lost every feasible path")
            out.append((int(k), rows if status == CHANGED else None,
                        dict(certified=bool(box), dp=bool(dp), frontier=int(frontier),
                             fixing_dps=int(fixing), fallback=bool(status == FRONTIER_LIMIT))))
        return out

    def fine(self, X, fwd, bwd, a, c, frontier_bytes, focals=None):
        """Proposals of every (listed) founder on blocks [a, c): [(kind, a, c, k, rows | None,
        diagnostic)]."""
        emissions = slice_list(self.emissions, a, c)
        sizes = [self.arrays[b].shape[1] for b in range(a, c)]
        bins = max(self.arrays[b].shape[3] for b in range(a, c))
        return [("fine", a, c, k, rows, diagnostic)
                for k, rows, diagnostic in self._template(emissions, bins, X[:, a:c].T, sizes,
                                                          fwd[a], bwd[c], frontier_bytes, focals)]

    def coarse_window(self, X, fwd, bwd, i, j, frontier_bytes):
        """Proposals of every founder over L1 spans [i, j). A span's alphabet: the distinct row
        tuples the founders use there, then the L1 context paths."""
        groups, context = self.coarse
        alphabets, lookups = [], []
        for unit in range(i, j):
            start, stop = groups[unit]
            rows, lookup = [], {}
            for row in [*X[:, start:stop], *context[unit]]:
                key = np.ascontiguousarray(row, np.int64).tobytes()
                if key not in lookup:
                    lookup[key] = len(rows)
                    rows.append(np.ascontiguousarray(row, np.int64))
            alphabets.append(np.asarray(rows, np.int64))
            lookups.append(lookup)
        emissions = gather_macro(self.emissions, np.asarray(groups[i:j], np.int64), List(alphabets))
        cols = np.empty((j - i, len(X)), np.int64)
        for u, unit in enumerate(range(i, j)):
            start, stop = groups[unit]
            for founder in range(len(X)):
                cols[u, founder] = lookups[u][np.ascontiguousarray(X[founder, start:stop],
                                                                   np.int64).tobytes()]
        a, c = groups[i][0], groups[j - 1][1]
        bins = max(sum(self.arrays[b].shape[3] for b in range(*groups[unit])) for unit in range(i, j))
        out = []
        for k, choice, diagnostic in self._template(emissions, bins, cols,
                                                    [len(x) for x in alphabets], fwd[a], bwd[c],
                                                    frontier_bytes):
            rows = None
            if choice is not None:
                rows = np.concatenate([alphabets[u][int(choice[u])] for u in range(j - i)])
                if np.array_equal(rows, X[k, a:c]):
                    rows = None
            out.append(("coarse", a, c, k, rows, diagnostic))
        return out

    def solve(self, index, X, fwd, bwd, frontier_bytes):
        """The proposals of template `index` for panel X with messages fwd, bwd (a pure function of
        X's rows in the window and the flank messages fwd[a], bwd[c], unless a frontier fallback)."""
        kind, a, c = self.templates[index]
        if kind == "fine":
            return self.fine(X, fwd, bwd, a, c, frontier_bytes)
        return self.coarse_window(X, fwd, bwd, a, c, frontier_bytes)
