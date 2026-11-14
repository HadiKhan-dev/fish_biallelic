"""Hybrid window proposals for the binned max-plus surrogate H.

The box certificate precedes an exact frontier programme with at most 64 kept
paths. Overflow selects heuristic seeded search with shared max-plus messages;
the ordinary recurrence rescores its offer without an optimality certificate.
Seeded search does not count as a beam truncation. Exact site-level F and the
acceptance guards decide which proposals enter the panel. Canonical ties prefer
the incumbent, then content order. One template account covers all focal work;
exhaustion discards the whole template, independently of its thread-map width.
"""

from __future__ import annotations

from math import isqrt

import numpy as np
from numba import prange
from numba.typed import List

from haplotype_reconstruction.core.errors import InputError, InvariantError, SettingsError
from haplotype_reconstruction.core.kernels import THREAD_WORK_FLOOR, kernel

from . import seeded_search
from .costs import seeded_search_bytes
from .emissions import pair_index
from .objective import row_max
from .overloads import overloads_for
from .parameters import (
    BOX_ARITHMETIC_MARGIN,
    CANONICAL_SKIP_ABSOLUTE_MARGIN,
    CANONICAL_SKIP_RELATIVE_MARGIN,
    CANONICAL_TOL,
    COARSE_STRIDE,
    COARSE_WINDOW,
    DOMINANCE_ABSOLUTE_MARGIN,
    DOMINANCE_RELATIVE_MARGIN,
    DOMINANCE_TOLERANCE,
    FINE_STRIDE,
    FINE_WINDOW,
    HYBRID_FRONTIER,
    INTERVAL_SEED_STARTS,
    LOCAL_ITERATIONS,
    PROGRAMME_RELATIVE_MARGIN,
    SEED_STARTS,
)
from .work_limit import WindowWorkState

# Memory of the child messages a programme stores per unit (else they are recomputed).
STORE_BYTES = 128 * 1024**2

OK, INFEASIBLE, BACKSTOP = 0, 1, 2
# Empty arrays disable _child's floor shortcut during canonical fixing (read only).
NO_FLOOR = np.empty(0)
NO_FLOOR2 = np.empty((0, 0))
# Focal statuses. BEAM_LIMIT is retained for existing receipt readers; the exact
# hybrid attempt stops on frontier overflow and never truncates a beam.
UNCHANGED, CHANGED, FRONTIER_LIMIT, LOST = 0, 1, 2, 3
BEAM_LIMIT = 5

# Artifact-only numerical work account; caller supplies an explicit budget, default off.
WORK_LIMIT = 4
# work[WINDOW_SHARED], work[WINDOW_SHARED + 1]: the units of solve_template's window-shared debits (phases 0 and 13,
# once per call whatever its focal count), which merge_focal_chunks counts once per template.
WINDOW_SHARED = 22


@kernel(overloads=overloads_for(__name__, "_debit"), cache=True, nogil=True)
def _debit(work, phase, a, b=1, c=1, d=1, e=1, f=1):
    """Charge ceil(product / 2**shift), once, for positive loop-domain factors.

    O(1) time and storage.
    """
    if len(work) == 0:
        return
    shift = work[20]
    if shift == 0:
        # Use exact integer debits for ordinary-sized accounts.
        amount = 1
        for factor in (a, b, c, d, e, f):
            if factor == 0:
                return
            if amount > work[0] // factor:
                work[2], work[3] = WORK_LIMIT, phase
                raise RuntimeError("window work limit")
            amount *= factor
    else:
        # Exact partial product = quotient * quantum + remainder. Shape checks
        # bound individual factors; shift<=31 bounds the remainder product.
        mask = (1 << shift) - 1
        quotient, remainder = 0, 1
        for factor in (a, b, c, d, e, f):
            if factor == 0:
                return
            if quotient > work[0] // factor:
                work[2], work[3] = WORK_LIMIT, phase
                raise RuntimeError("window work limit")
            quotient *= factor
            high = factor >> shift
            if high and remainder > (work[0] - quotient) // high:
                work[2], work[3] = WORK_LIMIT, phase
                raise RuntimeError("window work limit")
            quotient += remainder * high
            product = remainder * (factor & mask)
            carry = product >> shift
            if carry > work[0] - quotient:
                work[2], work[3] = WORK_LIMIT, phase
                raise RuntimeError("window work limit")
            quotient += carry
            remainder = product & mask
            if quotient == work[0] and remainder:
                work[2], work[3] = WORK_LIMIT, phase
                raise RuntimeError("window work limit")
        amount = quotient + int(remainder != 0)
    work[0] -= amount
    work[1] += amount
    work[4 + phase] += amount


@kernel(overloads=overloads_for(__name__, "_charge_window"), cache=True, nogil=True)
def _charge_window(work, phase, emissions, cols, domain):
    """Conservative normalized loop domains; factor128 is not a literal FLOP count.

    O(W) dimension scan; debits normalized loop domains.
    """
    if len(work) == 0:
        return
    n, w, k = emissions[0].shape[0], len(emissions), cols.shape[1]
    q, b, r = k * (k + 1) // 2, 0, 0
    for u in range(w):
        b = max(b, emissions[u].shape[2])
        r = max(r, domain[u] if phase == 0 else domain[u + 1] - domain[u])
    if phase == 0:
        _debit(work, phase, 128, n, w, q * (b + 1) ** 2 + k + r + 1)
        _debit(work, phase, 128, w * (k + r + 1) + q + 1)
    elif phase == 1:
        _debit(work, phase, 128, n, w, q * (b + 1) + r * (b * k + b * b + 1) + q + b * b + k + r + 1)
    elif phase == 12:
        _debit(work, phase, 128, n, w, (r + 1) * (q * (b + 1) + b * k + b * b + 1) + q * (b + 1))
        _debit(work, phase, 128, w, w, r + 1, r + 1)
    elif phase == 13:
        _debit(work, phase, 128, n, w * (q * (b + 1) + 1) + q + 1)
    elif phase == 14:
        _debit(work, phase, 128, 4 * n * q + w + 1)


@kernel(overloads=overloads_for(__name__, "_sum_max"), inline="always", nogil=True)
def _sum_max(a, b):
    """max_z (a[z] + b[z]) (non-empty, no NaN) with four interleaved running maxima: the maximum of the same sums
    as a sequential scan.

    O(Q) time for Q entries.
    """
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


@kernel(overloads=overloads_for(__name__, "_diff_max"), inline="always", nogil=True)
def _diff_max(a, b):
    """max_z (a[z] - b[z]) (non-empty, no NaN) with four interleaved running maxima.

    O(Q) time for Q entries.
    """
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


@kernel(overloads=overloads_for(__name__, "_extend"), cache=True, nogil=True)
def _extend(parent_n, parent_c, emis, col, focal, row, first, second, penalty, out_n, out_c):
    """Closed normalised message of a partial path extended by `row` on one unit.

    O(N b Q) time for N samples, b unit bins and Q pair states.
    """
    samples, states = parent_n.shape
    bins = emis.shape[2]
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
                out_n[s, z] = v + emis[s, pair_index(ra, rb), bn]
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


@kernel(overloads=overloads_for(__name__, "_join"), cache=True, nogil=True)
def _join(msg_n, msg_c, tail):
    """sum_s [C_s + max_z (N_s(z) + tail_s(z))].

    O(N Q) time.
    """
    samples = msg_n.shape[0]
    total = 0.0
    for s in range(samples):
        total += msg_c[s] + _sum_max(msg_n[s], tail[s])
    return total


@kernel(overloads=overloads_for(__name__, "_dominance"), cache=True, nogil=True)
def _dominance(n_i, t_i, n_j, t_j, tol):
    """True iff no completion makes i better than j by more than tol.

    O(N Q) time.
    """
    samples = n_i.shape[0]
    d = t_i - t_j
    if d > tol:
        return False
    for s in range(samples):
        d += _diff_max(n_i[s], n_j[s])
        if d > tol:
            return False
    return True


@kernel(overloads=overloads_for(__name__, "_stable_order"), cache=True, nogil=True)
def _stable_order(keys_float, keys_inc, par, cho):
    """Order by (-T, incumbent row first, parent, row), stable and deterministic.

    O(C log C) time for C children.
    """
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
@kernel(overloads=overloads_for(__name__, "_focal_layout"), cache=True, nogil=True)
def _focal_layout(focal, first, second):
    """fz[c] = state {f, c}; bg = the states without f (in state order).

    O(Q) time.
    """
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


@kernel(overloads=overloads_for(__name__, "_state_tables"), cache=True, nogil=True)
def _state_tables(emis, col, first, second):
    """Focal-independent sums of one unit for every state z (rows fixed by col), shared by every focal problem of a
    window: P[s, z, k] = sum_{m <= k} e_m(z), T[s, z, k] = sum_{m >= k} e_m(z) (T[.., t] = 0) and, per (s, a, c)
    with a <= c, the largest run sum sum_{m=a}^{c} e_m(z) over all states (best), the founders (bx, by) of the
    first state attaining it and the largest over the states without founder bx (mx) and without founder by
    (my).

    O(N Q b^2) time.
    """
    samples = emis.shape[0]
    bins = emis.shape[2]
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
                acc += emis[s, pair_index(ra, rb), k]
                P[s, z, k] = acc
            acc = 0.0
            for k in range(bins - 1, -1, -1):
                acc += emis[s, pair_index(ra, rb), k]
                T[s, z, k] = acc
            for a in range(bins):
                run = 0.0
                for c in range(a, bins):
                    run += emis[s, pair_index(ra, rb), c]
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


@kernel(overloads=overloads_for(__name__, "_unit_tables"), cache=True, nogil=True)
def _unit_tables(P, T, best, bx, by, mx, my, focal, bg):
    """Background sums of one unit for focal founder f (from _state_tables): prefix[s, b, k] = sum_{m <= k} e_m,
    suffix[s, b, k] = sum_{m >= k} e_m (suffix[.., t] = 0) of background state bg[b], and seg[s, a, c] = max_b
    sum_{m=a}^{c} e_m over the background (the states without f).

    O(N Q b^2) time.
    """
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


@kernel(overloads=overloads_for(__name__, "window_box"), cache=True, nogil=True)
def window_box(emissions, cols, cands, offsets, focal, first, second, penalty, end, fz, bg, prefixes, suffixes, segs):
    """U[u] >= the value of units u.. plus the end flank from any state, per sample.

    Each sample may choose f's row of every later unit freely (a relaxation); the backward bin recursion is
    emission-then-switch, matching the forward one. Shared background: per row only the K focal states and t
    switch maxima m_k are carried; the background states take max over rows of the floors, which commutes with
    the (monotone) float subtraction of the penalty, so one-bin units equal the all-state reference recursion
    (tests) bitwise.

    O(N W (Q (b+1)+R (b K+b^2))) time.
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
        bins = emis.shape[2]
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
                        y = start[c] + emis[s, pair_index(ga[c], gb[c]), 0]
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
                        yf[c] += emis[s, pair_index(ga[c], gb[c]), k]
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


@kernel(overloads=overloads_for(__name__, "_parent"), cache=True, nogil=True)
def _parent(parent_n, bg, fz, prefix, unext, penalty, pmax, pb, tp, hi, lo, arg, a0, background):
    """Row-independent tables of one parent on one unit, per sample: the parent's maximum pmax, pb[s, k] = max_b
    P_k[b] (the background chain after bin k without later floors), tp[s] = max_b (P_{t-1}[b] + U[b]), the
    extremes and the argmax of P_{t-1} over the background (the dominance shortcuts), the focal states after the
    unit's first switch a0[s, c] = max(N[fz[c]], pmax - pen) (every child's first step) and, when `background`
    has rows, the background states P_{t-1}[z] of every child (its shared form: max(N[z], pmax - pen) +
    E_{0..t-1}[z]). One pass over the background serves all of them; computed once per parent, not per child.

    O(N (Q b+K+b^2)) time.
    """
    samples = parent_n.shape[0]
    bins = prefix.shape[2]
    nb = len(bg)
    founders = len(fz)
    store = background.shape[0] > 0
    if bins == 1 and nb > 0:
        _parent_one_bin(parent_n, bg, fz, prefix, unext, penalty, pmax, pb, tp, hi, lo, arg, a0, background, store)
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


@kernel(overloads=overloads_for(__name__, "_parent_one_bin"), cache=True, nogil=True)
def _parent_one_bin(parent_n, bg, fz, prefix, unext, penalty, pmax, pb, tp, hi, lo, arg, a0, background, store):
    """_parent on a one-bin unit: the background values P[b] = max(N[bg[b]], pmax - pen) + e[b] first, then their
    maximum (= pb, = hi) with its first index (arg), minimum (lo) and the bound term tp = max_b (P[b] +
    U[bg[b]]) by four interleaved running extremes each, combined so that the values (and the first index of the
    maximum) are those of the sequential scan.

    O(N (Q+K)) time.
    """
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


@kernel(overloads=overloads_for(__name__, "_row_pairs"), cache=True, nogil=True)
def _row_pairs(row, col, focal, first, second, fz, out_a, out_b):
    """Emission row indices (out_a[c], out_b[c]) of the focal states {f, c} when f takes `row`.

    O(K) time.
    """
    for c in range(len(fz)):
        z = fz[c]
        out_a[c] = row if first[z] == focal else col[first[z]]
        out_b[c] = row if second[z] == focal else col[second[z]]


@kernel(overloads=overloads_for(__name__, "_unit_rows"), cache=True, nogil=True)
def _unit_rows(cands, begin, rows, col, focal, first, second, fz):
    """_row_pairs of a unit's candidate rows cands[begin:begin + rows]: (rows, K) int64 arrays.

    O(R K) time.
    """
    founders = len(fz)
    ra = np.empty((rows, founders), np.int64)
    rb = np.empty((rows, founders), np.int64)
    for g in range(rows):
        _row_pairs(cands[begin + g], col, focal, first, second, fz, ra[g], rb[g])
    return ra, rb


@kernel(overloads=overloads_for(__name__, "_focal_values"), cache=True, nogil=True)
def _focal_values(values, fz):
    """values[:, fz] as a contiguous (S, K) array.

    O(N K) time.
    """
    samples = values.shape[0]
    out = np.empty((samples, len(fz)))
    for s in range(samples):
        for c in range(len(fz)):
            out[s, c] = values[s, fz[c]]
    return out


@kernel(overloads=overloads_for(__name__, "_row_maxima"), cache=True, nogil=True)
def _row_maxima(emis, ra, rb):
    """emax[g, s, k] = max_c e_k(ra[g, c], rb[g, c]): the largest bin-k emission of any focal state of a child
    taking row g (with _parent_floor, the floor test of _child).

    O(N R b K) time.
    """
    rows, founders = ra.shape
    samples = emis.shape[0]
    bins = emis.shape[2]
    emax = np.empty((rows, samples, bins))
    for g in range(rows):
        for s in range(samples):
            for k in range(bins):
                m = emis[s, pair_index(ra[g, 0], rb[g, 0]), k]
                for c in range(1, founders):
                    v = emis[s, pair_index(ra[g, c], rb[g, c]), k]
                    if v > m:
                        m = v
                emax[g, s, k] = m
    return emax


@kernel(overloads=overloads_for(__name__, "_parent_floor"), cache=True, nogil=True)
def _parent_floor(parent_c, pmax, pb, tp, a0, seg, te, ubmax, ufz, penalty, amax, fmb, fs, fm, flo, fd, fcs, fsb):
    """The outcome every child of the parent shares in sample s when its focal states never reach the background's
    maximum within the unit nor its closing floor (_child's floor test): the bin maxima fmb[s, k] and switch
    floors fs[s, k] are then the background's (pb and the earlier floors through seg), the unit-end maximum fm,
    every closed focal value fd = lo - m, and the sample's terms of the child's total (fcs) and bound (fsb) are
    fixed. Computed by _child's operations in its order; amax[s] = max_c a0[s, c] starts the test.

    O(N (K+b^2)) time.
    """
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


@kernel(overloads=overloads_for(__name__, "_child"), cache=True, nogil=True)
def _child(
    parent_c,
    pmax,
    pb,
    tp,
    a0,
    seg,
    te,
    ubmax,
    ufz,
    emis,
    ra,
    rb,
    penalty,
    out_m,
    out_f,
    out_s,
    amax,
    emax,
    fmb,
    fs,
    fm,
    flo,
    fd,
    fcs,
    fsb,
):
    """One child (a parent extended by one row on one unit) without its background states.

    From the parent's tables (_parent: pmax, pb, tp, a0), the unit's suffix bound at the focal states ufz = U[u
    + 1][:, fz] and the row's focal emission indices ra, rb (_row_pairs); fills per sample the unit-end maximum
    out_m, the closed focal values out_f and the switch floors out_s[k] = m_k - pen; returns the child's total
    (sum of its constants, the reference's ordering key) and its bound (the reference's _join against U[u + 1]);

    Floor shortcut (when amax has entries: _parent_floor and the row's _row_maxima emax): X_k, an upper bound on
    every focal state after bin k (X_0 = amax + emax_0, X_k = max(X_{k-1}, fs_k) + emax_k; rounding is
    monotone), is checked against the background's maximum of each next bin and the closing floor. A sample that
    passes never lets a focal state set a maximum or close above the floor, so it takes the parent's shared
    outcome: the values the full computation produces.

    O(N (b K+b^2)) time.
    """
    samples = a0.shape[0]
    founders = a0.shape[1]
    bins = emis.shape[2]
    xf = np.empty(founders)
    total = 0.0
    bound = 0.0
    floor = len(amax) > 0
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
                continue
        if bins == 1:
            # The unit-end maximum over the background (pb) and the focal states, taken as the
            # focal states are formed (the comparisons of the general case, in the same order).
            m = pb[s, 0]
            for c in range(founders):
                x = a0[s, c] + emis[s, pair_index(ra[c], rb[c]), 0]
                xf[c] = x
                if x > m:
                    m = x
        else:
            for c in range(founders):
                xf[c] = a0[s, c] + emis[s, pair_index(ra[c], rb[c]), 0]
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
                    xf[c] = v + emis[s, pair_index(ra[c], rb[c]), k]
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
    return total, bound


@kernel(overloads=overloads_for(__name__, "_parent_alone"), cache=True, nogil=True)
def _parent_alone(parent_n, bg, fz, prefix, unext, penalty):
    """_parent's tables of one parent for its children's totals and bounds: (pmax, pb, tp, a0).

    O(N (Q b+K)) time.
    """
    samples, states = parent_n.shape
    bins = prefix.shape[2]
    pmax = np.empty(samples)
    pb = np.empty((samples, bins))
    tp = np.empty(samples)
    a0 = np.empty((samples, len(fz)))
    _parent(
        parent_n,
        bg,
        fz,
        prefix,
        unext,
        penalty,
        pmax,
        pb,
        tp,
        np.empty(samples),
        np.empty(samples),
        np.empty(samples, np.int64),
        a0,
        np.empty((0, states)),
    )
    return pmax, pb, tp, a0


@kernel(overloads=overloads_for(__name__, "_assemble_one_bin"), cache=True, nogil=True)
def _assemble_one_bin(parent_c, background, bg, fz, m_i, f_i, penalty, out_n, out_c):
    """The closed message of a one-bin child from its parent's shared background (_parent) and its own focal values
    (_child): the values _extend computes, by the same operations (the unit-end maximum m_i and the closed focal
    values are the child's; background states close as max(P[z], m - pen) - m; constants parent_c + m). Returns
    the child's total.

    O(N Q) time.
    """
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


@kernel(overloads=overloads_for(__name__, "_background_at"), inline="always", nogil=True)
def _background_at(p_last, m, floors, s, suffix, b, bins, penalty):
    """Closed background value max(-pen, P_{t-1} - M, max_{l >= 1} (s_l - M + E_{l..t-1})) of sample s and background
    state b, with floors[s, l] = s_l and suffix[s, b, l] = E_{l..t-1}. Inlined and read in place: it runs for every
    sample and background state of a dominance test, and passing it the rows floors[s] and suffix[s, b] would form two
    array views per call, each an atomic reference-count increment and decrement around O(b) work (several times the
    arithmetic on some processors, e.g. Sapphire Rapids).

    O(b) time.
    """
    v = p_last - m
    for l in range(1, bins):
        w = floors[s, l] - m + suffix[s, b, l]
        if w > v:
            v = w
    if -penalty > v:
        v = -penalty
    return v


@kernel(overloads=overloads_for(__name__, "_same_parent_bounds"), cache=True, nogil=True)
def _same_parent_bounds(
    mi, fi, si, ti, mj, fj, sj, tj, bins, pmax, hi, lo, arg, parent_n, bg, prefix, suffix, probes, penalty, stop_above
):
    """Upper and lower bounds on d(i, j) = t_i - t_j + sum_s max_z (n_i - n_j) for two children of one parent, from
    their focal values and floors.

    Focal states are compared directly. Background: one bin -> the closed values are max(P - M, -pen), a
    monotone function of the shared P, so the largest difference is at the extremes of P (bounds equal); several
    bins -> upper bound max(0, max_l alpha_i,l - alpha_j,l) (alpha_0 = -M, alpha_l = s_l - M) and, as the lower
    bound, the exact difference at probe states (the argmax of P_{t-1} and of each E_{l..t-1}). Stops once the
    lower bound exceeds stop_above (upper returned as inf).

    O(N (K+b^2)) time.
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
                v = _background_at(p_last, a, si, s, suffix, b, bins, penalty) - _background_at(
                    p_last, b2, sj, s, suffix, b, bins, penalty
                )
                if v > lb:
                    lb = v
        upper_sum += max(fd, ub)
        lower_sum += max(0.0, max(fd, lb))
        if lower_sum > stop_above:
            return np.inf, lower_sum
    return upper_sum, lower_sum


@kernel(overloads=overloads_for(__name__, "_safety"), cache=True, nogil=True)
def _safety(ti, tj, tk):
    """Float margin around a dominance threshold (covers both arithmetics' rounding).

    O(1) time.
    """
    return DOMINANCE_ABSOLUTE_MARGIN + DOMINANCE_RELATIVE_MARGIN * (abs(ti) + abs(tj) + abs(tk))


@kernel(overloads=overloads_for(__name__, "_dominance_value"), cache=True, nogil=True)
def _dominance_value(n_i, t_i, n_j, t_j, tol):
    """_dominance with the accumulated value: (dominated, d), d complete when dominated.

    O(N Q) time.
    """
    samples = n_i.shape[0]
    d = t_i - t_j
    if d > tol:
        return False, d
    for s in range(samples):
        d += _diff_max(n_i[s], n_j[s])
        if d > tol:
            return False, d
    return True, d


@kernel(overloads=overloads_for(__name__, "_one_bin_background"), cache=True, nogil=True)
def _one_bin_background(parent_n, pmax, emis, col, first, second, bg, penalty, out):
    """Background states of every child of a parent on a one-bin unit, before closing: max(N[z], m - pen) + e[z]
    (the reference's _extend arithmetic, shared by all rows).

    O(N Q) time.
    """
    samples = parent_n.shape[0]
    for s in range(samples):
        sw = pmax[s] - penalty
        for b in range(len(bg)):
            z = bg[b]
            v = parent_n[s, z]
            if sw > v:
                v = sw
            out[s, z] = v + emis[s, pair_index(col[first[z]], col[second[z]]), 0]


@kernel(overloads=overloads_for(__name__, "_cross_lower_bound"), cache=True, nogil=True)
def _cross_lower_bound(m_i, f_i, t_i, background, fz, probe, n_j, t_j, penalty, one_bin, s_i, suffix, bins, threshold):
    """True when a lower bound of d(i, j) exceeds threshold: the running sum of the per-sample maxima restricted to
    the focal states and the background state where the parent's shared chain is largest (O(K) per sample,
    stopping as soon as it exceeds).

    O(N (Q+b+K)) time.
    """
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
            v = _background_at(background[s, z], m, s_i, s, suffix, b, bins, penalty) - n_j[s, z]
        if v > best:
            best = v
        d += best
        if d > threshold:
            return True
    return False


@kernel(overloads=overloads_for(__name__, "_dominance_one_bin"), cache=True, nogil=True)
def _dominance_one_bin(m_i, f_i, t_i, background, fz, bg, n_j, t_j, tol, penalty):
    """_dominance_value(n_i, t_i, n_j, t_j, tol) for a one-bin child kept in shared form: every closed entry of n_i
    is formed with the reference's operations (focal values f_i; background max(x, M - pen) - M from the
    parent's shared x), so the values, the per-sample maxima, the accumulation order and the early exit are the
    reference's.

    O(N Q) time.
    """
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


@kernel(overloads=overloads_for(__name__, "_multi_bin_background"), cache=True, nogil=True)
def _multi_bin_background(parent_n, pmax, prefix, bg, penalty, out):
    """P_{t-1}[z] = max(N[z], m - pen) + E_{0..t-1}[z] of a parent's background states.

    O(N Q b) time.
    """
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


@kernel(overloads=overloads_for(__name__, "_dominance_multi_bin"), cache=True, nogil=True)
def _dominance_multi_bin(m_i, f_i, s_i, t_i, p_last, suffix, fz, bg, n_j, t_j, tol, penalty):
    """_dominance of a multi-bin child in shared form against a materialised path: 1 dominated, 0 not, -1 open (the
    closed background values are formed from the decomposition, exact in real arithmetic; decided only clear of
    tol by the float margin, the caller materialises otherwise).

    O(N Q b) time.
    """
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
            v = _background_at(p_last[s, z], m, s_i, s, suffix, b, bins, penalty) - n_j[s, z]
            if v > best:
                best = v
        d += best
        if d > tol + safety:
            return 0
    if d <= tol - safety:
        return 1
    return -1


@kernel(overloads=overloads_for(__name__, "_rows3"), cache=True, nogil=True)
def _rows3(buffer, rows, a, b):
    """A scratch array of at least `rows` leading entries (a fresh one only when `buffer` is too small; contents
    are not kept).

    O(1) view construction.
    """
    if buffer.shape[0] >= rows:
        return buffer
    return np.empty((max(rows, 2 * buffer.shape[0]), a, b))


@kernel(overloads=overloads_for(__name__, "_rows2"), cache=True, nogil=True)
def _rows2(buffer, rows, a):
    """O(1) view construction."""
    if buffer.shape[0] >= rows:
        return buffer
    return np.empty((max(rows, 2 * buffer.shape[0]), a))


@kernel(overloads=overloads_for(__name__, "window_dp"), cache=True, nogil=True)
def window_dp(
    emissions,
    cols,
    cands,
    offsets,
    focal,
    incumbent,
    first,
    second,
    penalty,
    start_n,
    start_c,
    unit0,
    end,
    upper,
    lower,
    margin,
    dom_tol,
    backstop,
    store_limit,
    fz,
    bg,
    prefixes,
    suffixes,
    segs,
    work,
):
    """Exact maximum of H over f's rows on units unit0.. from one partial path.

    Returns (status, value, path of units unit0.., maximum frontier, zero). Status 1 means no path reaches
    lower-margin; status 2 means the frontier exceeded backstop. With fixed frontier F, time per unit is O(N
    (F^2 R (Q (b+1)+b K+b^2)+F Q b^2)); Q=K(K+1)/2. The all-state programme (tests) with the same child order,
    bound pruning, dominance decisions and kept messages: children are formed per parent from the shared
    background (O(S (t K + t^2)) each instead of O(S t Q)); a child's full message is materialised (_extend)
    only when it is kept or must be compared with a child of another parent. A child dominated by a kept path j,
    through an earlier sibling c dominated by j, is dropped when d(i, c) + d(c, j) <= tol clears the float
    margin (d is subadditive). Work arrays are reused across units (the frontier and the kept paths swap
    buffers); their sizes never change a value or a decision.
    """
    units = len(emissions)
    samples, states = start_n.shape
    width = units - unit0
    _debit(work, 3, 128, samples, states + width + 1)
    path = np.empty(width, np.int64)
    if width == 0:
        value = _join(start_n, start_c, end)
        if value < lower - margin:
            return INFEASIBLE, value, path, 1, 0
        return OK, value, path, 1, 0
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
    truncated = 0
    p_a0 = np.empty((0, samples, founders))
    p_bg = np.empty((0, samples, states))
    no_background = np.empty((0, states))
    c_m = np.empty((0, samples))
    c_f = np.empty((0, samples, founders))
    for u in range(unit0, units):
        emis = emissions[u]
        col = cols[u]
        bins = emis.shape[2]
        begin, stop = offsets[u], offsets[u + 1]
        rows = stop - begin
        # Gate complete child family before allocating or evaluating it.
        _debit(work, 4, 128, samples, states, bins + 1, parents + 1)
        _debit(work, 4, 128, samples, bins * founders + bins * bins + 1, parents + rows + 1)
        _debit(work, 4, 128, samples, bins * founders + bins * bins + 1, parents, rows)
        _debit(work, 4, 128, parents, rows)
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
            _parent(
                front_n[p],
                bg,
                fz,
                prefix,
                unext,
                penalty,
                p_max[p],
                p_pb[p],
                p_tp[p],
                p_hi[p],
                p_lo[p],
                p_arg[p],
                p_a0[p],
                p_bg[p] if cache_bg else no_background,
            )
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
            _parent_floor(
                front_c[p],
                p_max[p],
                p_pb[p],
                p_tp[p],
                p_a0[p],
                seg,
                te,
                ubmax,
                ufz,
                penalty,
                p_amax[p],
                p_fmb[p],
                p_fs[p],
                p_fm[p],
                p_flo[p],
                p_fd[p],
                p_fcs[p],
                p_fsb[p],
            )
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
                front_c[p],
                p_max[p],
                p_pb[p],
                p_tp[p],
                p_a0[p],
                seg,
                te,
                ubmax,
                ufz,
                emis,
                ra[g],
                rb[g],
                penalty,
                c_m[slot],
                c_f[slot],
                c_s[slot],
                p_amax[p],
                emax[g],
                p_fmb[p],
                p_fs[p],
                p_fm[p],
                p_flo[p],
                p_fd[p],
                p_fcs[p],
                p_fsb[p],
            )
        alive = 0
        for i in range(children):
            if bounds[i] >= lower - margin:
                alive += 1
        if alive == 0:
            return INFEASIBLE, -np.inf, path, max_frontier, truncated
        levels = 1
        power = 2
        while power < children:
            levels += 1
            power *= 2
        _debit(work, 5, 128, 4, children, levels + 1)
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
        cell_cost = states * (bins + 1) + bins * founders + bins * bins + 1
        for position in range(children):
            _debit(work, 6, 128, samples, cell_cost)
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
                _child(
                    front_c[p],
                    p_max[p],
                    p_pb[p],
                    p_tp[p],
                    p_a0[p],
                    seg,
                    te,
                    ubmax,
                    ufz,
                    emis,
                    ra[g],
                    rb[g],
                    penalty,
                    cur_m,
                    cur_f,
                    cur_s,
                    p_amax[p],
                    emax[g],
                    p_fmb[p],
                    p_fs[p],
                    p_fm[p],
                    p_flo[p],
                    p_fd[p],
                    p_fcs[p],
                    p_fsb[p],
                )
            materialised = False
            mat_t = 0.0
            dominated = False
            dom_j = -1
            dom_d = 0.0
            # (1) Through an earlier sibling c already dropped as dominated by kept j:
            # d(i, j) <= d(i, c) + d(c, j).
            if sib_child[p] >= 0:
                up, _ = _same_parent_bounds(
                    cur_m,
                    cur_f,
                    cur_s,
                    totals[i],
                    sib_m[p],
                    sib_f[p],
                    sib_s[p],
                    sib_t[p],
                    bins,
                    p_max[p],
                    p_hi[p],
                    p_lo[p],
                    p_arg[p],
                    front_n[p],
                    bg,
                    prefix,
                    suffix,
                    probes,
                    penalty,
                    np.inf,
                )
                if up + sib_d[p] <= dom_tol - _safety(totals[i], sib_t[p], kept_t[sib_kept[p]]):
                    dominated = True
            # (2) Kept children of the same parent: shortcut, else the reference test.
            if not dominated:
                for j in range(count):
                    _debit(work, 7, 128, samples, cell_cost)
                    if par[kept_i[j]] != p:
                        continue
                    safety = _safety(totals[i], kept_lt[j], 0.0)
                    up, low = _same_parent_bounds(
                        cur_m,
                        cur_f,
                        cur_s,
                        totals[i],
                        kept_m[j],
                        kept_f[j],
                        kept_s[j],
                        kept_lt[j],
                        bins,
                        p_max[p],
                        p_hi[p],
                        p_lo[p],
                        p_arg[p],
                        front_n[p],
                        bg,
                        prefix,
                        suffix,
                        probes,
                        penalty,
                        dom_tol + safety,
                    )
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
                                _one_bin_background(
                                    front_n[p], p_max[p], emis, col, first, second, bg, penalty, scratch_bg
                                )
                            else:
                                _multi_bin_background(front_n[p], p_max[p], prefix, bg, penalty, scratch_bg)
                            bg_parent = p
                        background = scratch_bg
                    verdict = -1
                    if nb > 0 and _cross_lower_bound(
                        cur_m,
                        cur_f,
                        totals[i],
                        background,
                        fz,
                        p_probe[p],
                        kept_n[j],
                        kept_t[j],
                        penalty,
                        one_bin,
                        cur_s,
                        suffix,
                        bins,
                        dom_tol + _safety(totals[i], kept_t[j], 0.0),
                    ):
                        # The reference's sum exceeds tol (its terms are each at least these).
                        flag, value = False, dom_tol
                        verdict = 0
                    elif one_bin:
                        flag, value = _dominance_one_bin(
                            cur_m, cur_f, totals[i], background, fz, bg, kept_n[j], kept_t[j], dom_tol, penalty
                        )
                    else:
                        verdict = _dominance_multi_bin(
                            cur_m,
                            cur_f,
                            cur_s,
                            totals[i],
                            background,
                            suffix,
                            fz,
                            bg,
                            kept_n[j],
                            kept_t[j],
                            dom_tol,
                            penalty,
                        )
                        flag, value = verdict == 1, dom_tol
                    if verdict == -1 and not one_bin:
                        if not materialised:
                            mat_t = _extend(
                                front_n[p], front_c[p], emis, col, focal, cho[i], first, second, penalty, mat_n, mat_c
                            )
                            materialised = True
                        flag, value = _dominance_value(mat_n, mat_t, kept_n[j], kept_t[j], dom_tol)
                    if flag:
                        dominated = True
                        dom_j = j
                        dom_d = value
                        break
            # (3) Kept children of other parents: the reference test on full messages (one
            # bin: on the shared form, the same values).
            if not dominated:
                for j in range(count):
                    _debit(work, 7, 128, samples, cell_cost)
                    if par[kept_i[j]] == p:
                        continue
                    if cache_bg:
                        background = p_bg[p]
                    else:
                        if bg_parent != p:
                            if one_bin:
                                _one_bin_background(
                                    front_n[p], p_max[p], emis, col, first, second, bg, penalty, scratch_bg
                                )
                            else:
                                _multi_bin_background(front_n[p], p_max[p], prefix, bg, penalty, scratch_bg)
                            bg_parent = p
                        background = scratch_bg
                    verdict = -1
                    if nb > 0 and _cross_lower_bound(
                        cur_m,
                        cur_f,
                        totals[i],
                        background,
                        fz,
                        p_probe[p],
                        kept_n[j],
                        kept_t[j],
                        penalty,
                        one_bin,
                        cur_s,
                        suffix,
                        bins,
                        dom_tol + _safety(totals[i], kept_t[j], 0.0),
                    ):
                        # The reference's sum exceeds tol (its terms are each at least these).
                        flag, value = False, dom_tol
                        verdict = 0
                    elif one_bin:
                        flag, value = _dominance_one_bin(
                            cur_m, cur_f, totals[i], background, fz, bg, kept_n[j], kept_t[j], dom_tol, penalty
                        )
                    else:
                        verdict = _dominance_multi_bin(
                            cur_m,
                            cur_f,
                            cur_s,
                            totals[i],
                            background,
                            suffix,
                            fz,
                            bg,
                            kept_n[j],
                            kept_t[j],
                            dom_tol,
                            penalty,
                        )
                        flag, value = verdict == 1, dom_tol
                    if verdict == -1 and not one_bin:
                        if not materialised:
                            mat_t = _extend(
                                front_n[p], front_c[p], emis, col, focal, cho[i], first, second, penalty, mat_n, mat_c
                            )
                            materialised = True
                        flag, value = _dominance_value(mat_n, mat_t, kept_n[j], kept_t[j], dom_tol)
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
                mat_t = _assemble_one_bin(
                    front_c[p], p_bg[p], bg, fz, cur_m, cur_f, penalty, kept_n[count], kept_c[count]
                )
            else:
                mat_t = _extend(
                    front_n[p],
                    front_c[p],
                    emis,
                    col,
                    focal,
                    cho[i],
                    first,
                    second,
                    penalty,
                    kept_n[count],
                    kept_c[count],
                )
            kept_t[count] = mat_t
            kept_i[count] = i
            kept_lt[count] = totals[i]
            kept_m[count] = cur_m
            kept_f[count] = cur_f
            kept_s[count] = cur_s
            count += 1
            if count > backstop:
                return BACKSTOP, -np.inf, path, count, truncated
        # A kept child may be dominated by one kept after it (later in the order).
        _debit(work, 9, 128, count + 1, samples * (states + founders + bins + 1) + width + 1)
        remove = np.zeros(count, np.bool_)
        for x in range(count):
            px = par[kept_i[x]]
            for y in range(x + 1, count):
                _debit(work, 8, 128, samples, cell_cost)
                verdict = -1
                if par[kept_i[y]] == px:
                    safety = _safety(kept_lt[x], kept_lt[y], 0.0)
                    up, low = _same_parent_bounds(
                        kept_m[x],
                        kept_f[x],
                        kept_s[x],
                        kept_lt[x],
                        kept_m[y],
                        kept_f[y],
                        kept_s[y],
                        kept_lt[y],
                        bins,
                        p_max[px],
                        p_hi[px],
                        p_lo[px],
                        p_arg[px],
                        front_n[px],
                        bg,
                        prefix,
                        suffix,
                        probes,
                        penalty,
                        dom_tol + safety,
                    )
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
            grown_par[: history_off[step]] = history_par[: history_off[step]]
            grown_cho[: history_off[step]] = history_cho[: history_off[step]]
            history_par, history_cho = grown_par, grown_cho
        history_par[history_off[step] : need] = hp
        history_cho[history_off[step] : need] = hc
        history_off[step + 1] = need
        if survivors > max_frontier:
            max_frontier = survivors
    _debit(work, 9, 128, samples, states, parents)
    _debit(work, 9, 128, width + 1)
    best = -1
    best_value = -np.inf
    for x in range(parents):
        value = _join(front_n[x], front_c[x], end)
        if value > best_value:
            best_value = value
            best = x
    if best_value < lower - margin:
        return INFEASIBLE, best_value, path, max_frontier, truncated
    index = best
    for step in range(width - 1, -1, -1):
        path[step] = history_cho[history_off[step] + index]
        index = history_par[history_off[step] + index]
    return OK, best_value, path, max_frontier, truncated


@kernel(overloads=overloads_for(__name__, "path_value"), cache=True, nogil=True)
def path_value(emissions, cols, focal, rows, first, second, penalty, start_n, start_c, end):
    """H of one complete path, in the dynamic programme's arithmetic.

    O(N B Q) time for B total window bins.
    """
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


@kernel(overloads=overloads_for(__name__, "_closed_start"), cache=True, nogil=True)
def _closed_start(start, penalty):
    """The closed normalised start message (maximum 0, entries >= -penalty) and the per-sample maxima subtracted.

    O(N Q) time.
    """
    samples, states = start.shape
    out = np.empty((samples, states))
    top = np.empty(samples)
    for s in range(samples):
        m = row_max(start[s])
        top[s] = m
        for z in range(states):
            out[s, z] = max(start[s, z] - m, -penalty)
    return out, top


@kernel(overloads=overloads_for(__name__, "_closed_end"), cache=True, nogil=True)
def _closed_end(end, penalty):
    """The closed end message: per sample max(end - max(end), -penalty) (maximum 0, entries >= -penalty), the
    stored backward message's Potts closure.

    O(N Q) time.
    """
    samples, states = end.shape
    out = np.empty((samples, states))
    for s in range(samples):
        m = row_max(end[s])
        for z in range(states):
            out[s, z] = max(end[s, z] - m, -penalty)
    return out


@kernel(overloads=overloads_for(__name__, "_restrict_unit"), cache=True, nogil=True)
def _restrict_unit(cands, offsets, u, rows):
    """cands / offsets with unit u's alphabet replaced by rows.

    O(C) time for C candidate rows over the window.
    """
    units = len(offsets) - 1
    delta = len(rows) - (offsets[u + 1] - offsets[u])
    offsets2 = offsets.copy()
    for v in range(u + 1, units + 1):
        offsets2[v] = offsets[v] + delta
    cands2 = np.empty(offsets2[units], np.int64)
    cands2[: offsets[u]] = cands[: offsets[u]]
    cands2[offsets[u] : offsets2[u + 1]] = rows
    cands2[offsets2[u + 1] :] = cands[offsets[u + 1] :]
    return cands2, offsets2


@kernel(overloads=overloads_for(__name__, "_restricted_programme"), cache=True, nogil=True)
def _restricted_programme(
    emissions,
    cols,
    focal,
    incumbent,
    first,
    second,
    penalty,
    end,
    alphabet,
    alphabet_offsets,
    prefix_n,
    prefix_c,
    unit0,
    upper,
    target,
    backstop,
    store_limit,
    fz,
    bg,
    prefixes,
    suffixes,
    segs,
    work,
):
    """Canonical reachability under box bounds. One window_dp, O(W N (F^2 R (Q (b+1)+b K+b^2)+F Q b^2))."""
    _charge_window(work, 14, emissions, cols, alphabet_offsets)
    status, found, tail, reached, _ = window_dp(
        emissions,
        cols,
        alphabet,
        alphabet_offsets,
        focal,
        incumbent,
        first,
        second,
        penalty,
        prefix_n.copy(),
        prefix_c.copy(),
        unit0,
        end,
        upper,
        target,
        PROGRAMME_RELATIVE_MARGIN * max(1.0, abs(target)),
        0.0,
        backstop,
        store_limit,
        fz,
        bg,
        prefixes,
        suffixes,
        segs,
        work,
    )
    return status, found, tail, reached


@kernel(overloads=overloads_for(__name__, "_canonical_restricted"), cache=True, nogil=True)
def _canonical_restricted(
    emissions,
    cols,
    cands,
    offsets,
    focal,
    incumbent,
    first,
    second,
    penalty,
    start_n,
    start_c,
    end,
    upper,
    path,
    target,
    backstop,
    store_limit,
    fz,
    bg,
    prefixes,
    suffixes,
    segs,
    work,
):
    """First path reaching target in incumbent-first content order.

    Box bounds skip impossible rows; restricted programmes test remaining rows. At most W(R+1) programmes; each
    uses the same bounded frontier. The row advance costs O(N b Q), and bound tests O(N R (b K+b^2)) per unit.
    """
    _charge_window(work, 12, emissions, cols, offsets)
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
    fixing = 0
    frontier = 0
    for u in range(units):
        skip_below = (
            target
            - PROGRAMME_RELATIVE_MARGIN * max(1.0, abs(target))
            - CANONICAL_SKIP_ABSOLUTE_MARGIN
            - CANONICAL_SKIP_RELATIVE_MARGIN * abs(target)
        )
        held = incumbent[u]
        count = offsets[u + 1] - offsets[u]
        emis = emissions[u]
        bins = emis.shape[2]
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
            _, bound = _child(
                prefix_c,
                pm,
                pbv,
                tpv,
                a0,
                seg,
                te,
                ubmax,
                ufz,
                emis,
                row_a,
                row_b,
                penalty,
                lazy_m,
                lazy_f,
                lazy_s,
                NO_FLOOR,
                NO_FLOOR2,
                NO_FLOOR2,
                NO_FLOOR2,
                NO_FLOOR,
                NO_FLOOR,
                NO_FLOOR,
                NO_FLOOR,
                NO_FLOOR,
            )
            if bound < skip_below:
                continue
            tests[n_tests] = r
            n_tests += 1
        reachable = True
        if n_tests > 1:
            cands2, offsets2 = _restrict_unit(cands, offsets, u, tests[:n_tests])
            fixing += 1
            status, found, _, reached = _restricted_programme(
                emissions,
                cols,
                focal,
                incumbent,
                first,
                second,
                penalty,
                end,
                cands2,
                offsets2,
                prefix_n,
                prefix_c,
                u,
                upper,
                target,
                backstop,
                store_limit,
                fz,
                bg,
                prefixes,
                suffixes,
                segs,
                work,
            )
            if status == BACKSTOP:
                return BACKSTOP, best, fixing, reached
            if reached > frontier:
                frontier = reached
            reachable = status == OK and found >= target
        for x in range(n_tests if reachable else 0):
            r = tests[x]
            _extend(prefix_n, prefix_c, emis, cols[u], focal, r, first, second, penalty, out_n, out_c)
            fixing += 1
            status, found, tail, reached = _restricted_programme(
                emissions,
                cols,
                focal,
                incumbent,
                first,
                second,
                penalty,
                end,
                cands,
                offsets,
                out_n,
                out_c,
                u + 1,
                upper,
                target,
                backstop,
                store_limit,
                fz,
                bg,
                prefixes,
                suffixes,
                segs,
                work,
            )
            if status == BACKSTOP:
                return BACKSTOP, best, fixing, reached
            if reached > frontier:
                frontier = reached
            if status == OK and found >= target:
                best[u] = r
                best[u + 1 :] = tail
                break
        _extend(prefix_n, prefix_c, emis, cols[u], focal, best[u], first, second, penalty, out_n, out_c)
        prefix_n[:] = out_n
        prefix_c[:] = out_c
    return OK, best, fixing, frontier


@kernel(overloads=overloads_for(__name__, "solve_focal_reference"), cache=True, nogil=True)
def solve_focal_reference(
    emissions,
    cols,
    cands,
    offsets,
    focal,
    incumbent,
    first,
    second,
    penalty,
    start_n,
    start_c,
    end,
    incumbent_value,
    tables,
    backstop,
    store_limit,
    work,
):
    """The canonical exact maximiser of one founder's window problem when completed.

    start_n, start_c: the closed start message (_closed_start); incumbent_value: H of the panel's own rows;
    tables: the window's per-unit _state_tables (all three shared by the window's focal problems). Returns
    (status, rows, box_certified, dp, frontier, fixing_dps); status UNCHANGED = the incumbent is the canonical
    answer, CHANGED = rows is a different canonical answer, FRONTIER_LIMIT = a frontier exceeded `backstop`
    without an optimality certificate, LOST = the programme lost every feasible path (an internal error). The
    canonical-tie order: incumbent row first, then ascending index.

    At fixed frontier F, at most W(R+1) restricted programmes give O(N W^2 F^2 R^2 Q (b+1)^2) time before work-
    account stops.
    """
    _charge_window(work, 1, emissions, cols, offsets)
    units = len(emissions)
    rows = incumbent.copy()
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
    upper = window_box(
        emissions, cols, cands, offsets, focal, first, second, penalty, end, fz, bg, prefixes, suffixes, segs
    )
    tol = CANONICAL_TOL + PROGRAMME_RELATIVE_MARGIN * max(1.0, abs(incumbent_value))
    certificate = incumbent_value + tol - BOX_ARITHMETIC_MARGIN * max(1.0, abs(incumbent_value))
    if _join(start_n, start_c, upper[0]) <= certificate:
        return UNCHANGED, rows, True, False, 0, 0
    lower = incumbent_value
    margin = tol + (units + 1) * DOMINANCE_TOLERANCE + PROGRAMME_RELATIVE_MARGIN * max(1.0, abs(lower))
    status, value, path, frontier, truncated = window_dp(
        emissions,
        cols,
        cands,
        offsets,
        focal,
        incumbent,
        first,
        second,
        penalty,
        start_n,
        start_c,
        0,
        end,
        upper,
        lower,
        margin,
        DOMINANCE_TOLERANCE,
        backstop,
        store_limit,
        fz,
        bg,
        prefixes,
        suffixes,
        segs,
        work,
    )
    if status == BACKSTOP:
        return FRONTIER_LIMIT, rows, False, True, frontier, 0
    if status != OK:
        return LOST, rows, False, True, frontier, 0
    if incumbent_value >= value - tol:
        return UNCHANGED, rows, False, True, frontier, 0
    # Canonical ties: the lexicographically first path within tol of the optimum.
    target = value - tol
    status, best, fixing, reached = _canonical_restricted(
        emissions,
        cols,
        cands,
        offsets,
        focal,
        incumbent,
        first,
        second,
        penalty,
        start_n,
        start_c,
        end,
        upper,
        path,
        target,
        backstop,
        store_limit,
        fz,
        bg,
        prefixes,
        suffixes,
        segs,
        work,
    )
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


@kernel(overloads=overloads_for(__name__, "solve_focal"), cache=True, nogil=True)
def solve_focal(
    emissions,
    cols,
    cands,
    offsets,
    focal,
    incumbent,
    first,
    second,
    penalty,
    start_n,
    start_c,
    end,
    incumbent_value,
    tables,
    store_limit,
    work,
):
    """Box certificate, frontier-64 exact attempt, heuristic seeded answer on overflow.

    The seeded offer is rescored by ordinary H, carries no optimality certificate and does not count as a
    beam truncation.

    Preparation is O(N B (Q+R K)); seeded search is polynomial in its declared domains (seeded_search.search).
    Ordinary H rescoring is O(N B Q). Returns the exact solver's six-field status/rows/box/dp/frontier/fixing
    tuple and the seeded search's founder-order code (seeded_search.latent: 0 none, 1 ties that cannot change its
    wishes, 2 order-sensitive).
    """
    _charge_window(work, 1, emissions, cols, offsets)
    units = len(emissions)
    fz, bg = _focal_layout(focal, first, second)
    prefixes, suffixes, segs = List(), List(), List()
    for u in range(units):
        P, T, best, bx, by, mx, my = tables[u]
        prefix, suffix, seg = _unit_tables(P, T, best, bx, by, mx, my, focal, bg)
        prefixes.append(prefix)
        suffixes.append(suffix)
        segs.append(seg)
    upper = window_box(
        emissions,
        cols,
        cands,
        offsets,
        focal,
        first,
        second,
        penalty,
        end,
        fz,
        bg,
        prefixes,
        suffixes,
        segs,
    )
    tol = CANONICAL_TOL + PROGRAMME_RELATIVE_MARGIN * max(1.0, abs(incumbent_value))
    certificate = incumbent_value + tol - BOX_ARITHMETIC_MARGIN * max(1.0, abs(incumbent_value))
    if _join(start_n, start_c, upper[0]) <= certificate:
        return UNCHANGED, incumbent.copy(), True, False, 0, 0, 0
    st, exact_rows, exact_box, exact_dp, exact_frontier, exact_fixing = solve_focal_reference(
        emissions,
        cols,
        cands,
        offsets,
        focal,
        incumbent,
        first,
        second,
        penalty,
        start_n,
        start_c,
        end,
        incumbent_value,
        tables,
        HYBRID_FRONTIER,
        store_limit,
        work,
    )
    if st != FRONTIER_LIMIT and st != BEAM_LIMIT:
        return st, exact_rows, exact_box, exact_dp, exact_frontier, exact_fixing, 0
    samples, states = start_n.shape
    total_bins = 0
    max_rows = 0
    reversed_order = False
    for u in range(units):
        total_bins += emissions[u].shape[2]
        max_rows = max(max_rows, offsets[u + 1] - offsets[u])
        if offsets[u + 1] - offsets[u] > 1:
            reversed_order = cands[offsets[u]] > cands[offsets[u + 1] - 1]
    # Bound every search domain before allocating the gathered emissions.
    # N samples, W units, B bins, R rows, Q pair states, P<=N+1 wishes,
    # L<=13 starts, I<=32 ascent sweeps. Factor 128 is the account's common
    # normalization, not a literal FLOP estimate; method: refinement.md rs07.
    seeds = samples + 1
    starts = 1 + SEED_STARTS + INTERVAL_SEED_STARTS
    _debit(work, 15, 128, samples, total_bins + units + 1, states + max_rows * len(fz) + 1)
    _debit(work, 15, 128, samples, total_bins, total_bins, states + 1)
    _debit(work, 15, 128, samples, max_rows, total_bins, len(fz) + total_bins + 1)
    _debit(work, 15, 128, seeds, seeds, units)
    _debit(work, 15, 128, samples, units, units, total_bins)
    _debit(work, 15, 128, samples, seeds, units, total_bins * (len(fz) + total_bins + 1) + units * len(fz))
    _debit(work, 15, 128, INTERVAL_SEED_STARTS + SEED_STARTS + 1, seeds, units, units * (units + 1))
    _debit(
        work,
        15,
        128,
        starts,
        LOCAL_ITERATIONS,
        samples,
        (max_rows + 1) * total_bins * (len(fz) + total_bins + 1) + units * total_bins,
    )
    _debit(work, 15, 128, starts, LOCAL_ITERATIONS, len(cands) + 1, units)
    _debit(work, 15, 128, samples, total_bins, states + 1)
    fe, be, unit, bins = seeded_search.prepare(emissions, cols, cands, offsets, focal, first, second, fz, bg)
    start = start_n + start_c.reshape((len(start_c), 1))
    best, value, stats = seeded_search.search(
        fe, be, unit, bins, cands, offsets, incumbent, fz, bg, penalty, start, end, upper, tol, reversed_order
    )
    # The shared-background evaluator is mathematical elimination of the same
    # max-plus chain. The ordinary recurrence remains the final score authority.
    exact = path_value(emissions, cols, focal, best, first, second, penalty, start_n, start_c, end)
    stats[0] += 1
    changed = exact > incumbent_value + tol
    return (CHANGED if changed else UNCHANGED), best, False, True, exact_frontier, exact_fixing, stats[7]


@kernel(overloads=overloads_for(__name__, "solve_template"), cache=True, nogil=True)
def solve_template(emissions, cols, sizes, focals, penalty, start, end, store_limit, reverse, work):
    """Every listed focal founder's problem of one window (shared emissions and flanks).

    cols[u, k] is founder k's row on unit u (the focal's incumbent); the alphabet of unit u is 0..sizes[u]-1.
    Returns per focal (status, rows, box_certified, dp, frontier, fixing_dps, seeded_order) as arrays; seeded_order:
    the focal's seeded-search founder-order code (solve_focal).

    The flanks are the exact objective's messages (per-homologue transitions), whose entries can lie more than
    `penalty` below their maximum (a change of both homologues costs 2c). The surrogate's bounds assume Potts-
    closed flanks, so both the start (_closed_start) and the end (_closed_end) are closed here: the surrogate is
    a Potts window between relaxed flanks (proposals only).

    O(K) focal calls sharing O(N W Q b^2) tables; the template account bounds admitted work.
    """
    end = _closed_end(end, penalty)
    shared = len(work) > WINDOW_SHARED + 1
    used = work[1] if shared else 0
    _charge_window(work, 0, emissions, cols, sizes)
    if shared:
        work[WINDOW_SHARED] += work[1] - used
    units = len(emissions)
    founders = cols.shape[1]
    offsets = np.zeros(units + 1, np.int64)
    for u in range(units):
        offsets[u + 1] = offsets[u] + sizes[u]
    cands = np.empty(offsets[units], np.int64)
    for u in range(units):
        for r in range(sizes[u]):
            # ascending alphabet index (content order), descending under the convention reversal
            cands[offsets[u] + r] = sizes[u] - 1 - r if reverse else r
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
    tied = np.zeros(n, np.int64)
    if n == 0:
        return status, rows, box, dp, frontier, fixing, tied
    # Shared by every focal problem: the closed start message, the value of the panel's own rows
    # (an incumbent path is the panel itself whatever the focal founder, so path_value performs
    # the same operations for every focal) and the per-state unit sums.
    start_n, start_c = _closed_start(start, penalty)
    k0 = focals[0]
    used = work[1] if shared else 0
    _charge_window(work, 13, emissions, cols, offsets)
    if shared:
        work[WINDOW_SHARED + 1] += work[1] - used
    incumbent_value = path_value(
        emissions, cols, k0, np.ascontiguousarray(cols[:, k0]), first, second, penalty, start_n, start_c, end
    )
    tables = List()
    for u in range(units):
        tables.append(_state_tables(emissions[u], cols[u], first, second))
    for i in range(n):
        k = focals[i]
        incumbent = np.ascontiguousarray(cols[:, k])
        st, r, bx, used, fr, fx, tie = solve_focal(
            emissions,
            cols,
            cands,
            offsets,
            k,
            incumbent,
            first,
            second,
            penalty,
            start_n,
            start_c,
            end,
            incumbent_value,
            tables,
            store_limit,
            work,
        )
        status[i] = st
        rows[i] = r
        box[i] = bx
        dp[i] = used
        frontier[i] = fr
        fixing[i] = fx
        tied[i] = tie
    return status, rows, box, dp, frontier, fixing, tied


# ------------------------------------------------------------------ two-stage template solve (shared sweep 0)
# solve_template split where the content-order convention first enters (a fit's two searches share stage 1 of every
# sweep-0 template: SweepEngine._shared_sweep0). Stage 1 (convention-free): the window-shared
# tables, and per focal the box certificate and the frontier programme (window_dp), whose children are ordered by
# (-total, incumbent, parent, row) whatever the alphabet's listing order. Stage 2 (per convention): the canonical
# fixing (_canonical_restricted walks the alphabet in convention order) and the seeded search on frontier overflow.
# Stage 2 recomputes the tables and box bounds it needs without debiting them again and continues stage 1's work
# account, so the template's account is the one-call account.
S1_CERTIFIED, S1_UNCHANGED, S1_LOST, S1_CANONICAL, S1_LIMIT = 0, 1, 2, 3, 4


@kernel(overloads=overloads_for(__name__, "solve_template_stage1"), cache=True, nogil=True)
def solve_template_stage1(emissions, cols, sizes, focals, penalty, start, end, store_limit, work):
    """Per focal: (code S1_*, programme value, programme path, frontier) of the convention-free stage."""
    end = _closed_end(end, penalty)
    shared = len(work) > WINDOW_SHARED + 1
    used = work[1] if shared else 0
    _charge_window(work, 0, emissions, cols, sizes)
    if shared:
        work[WINDOW_SHARED] += work[1] - used
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
    code = np.zeros(n, np.int64)
    value = np.zeros(n)
    paths = np.zeros((n, units), np.int64)
    frontier = np.zeros(n, np.int64)
    if n == 0:
        return code, value, paths, frontier
    start_n, start_c = _closed_start(start, penalty)
    k0 = focals[0]
    used = work[1] if shared else 0
    _charge_window(work, 13, emissions, cols, offsets)
    if shared:
        work[WINDOW_SHARED + 1] += work[1] - used
    incumbent_value = path_value(
        emissions, cols, k0, np.ascontiguousarray(cols[:, k0]), first, second, penalty, start_n, start_c, end
    )
    tables = List()
    for u in range(units):
        tables.append(_state_tables(emissions[u], cols[u], first, second))
    for i in range(n):
        k = focals[i]
        incumbent = np.ascontiguousarray(cols[:, k])
        _charge_window(work, 1, emissions, cols, offsets)
        fz, bg = _focal_layout(k, first, second)
        prefixes, suffixes, segs = List(), List(), List()
        for u in range(units):
            P, T, best, bx, by, mx, my = tables[u]
            prefix, suffix, seg = _unit_tables(P, T, best, bx, by, mx, my, k, bg)
            prefixes.append(prefix)
            suffixes.append(suffix)
            segs.append(seg)
        upper = window_box(
            emissions, cols, cands, offsets, k, first, second, penalty, end, fz, bg, prefixes, suffixes, segs
        )
        tol = CANONICAL_TOL + PROGRAMME_RELATIVE_MARGIN * max(1.0, abs(incumbent_value))
        certificate = incumbent_value + tol - BOX_ARITHMETIC_MARGIN * max(1.0, abs(incumbent_value))
        if _join(start_n, start_c, upper[0]) <= certificate:
            code[i] = S1_CERTIFIED
            continue
        _charge_window(work, 1, emissions, cols, offsets)
        lower = incumbent_value
        margin = tol + (units + 1) * DOMINANCE_TOLERANCE + PROGRAMME_RELATIVE_MARGIN * max(1.0, abs(lower))
        status, found, path, reached, truncated = window_dp(
            emissions,
            cols,
            cands,
            offsets,
            k,
            incumbent,
            first,
            second,
            penalty,
            start_n,
            start_c,
            0,
            end,
            upper,
            lower,
            margin,
            DOMINANCE_TOLERANCE,
            HYBRID_FRONTIER,
            store_limit,
            fz,
            bg,
            prefixes,
            suffixes,
            segs,
            work,
        )
        frontier[i] = reached
        if status == BACKSTOP:
            code[i] = S1_LIMIT
        elif status != OK:
            code[i] = S1_LOST
        elif incumbent_value >= found - tol:
            code[i] = S1_UNCHANGED
        else:
            code[i] = S1_CANONICAL
            value[i] = found
            paths[i] = path
    return code, value, paths, frontier


@kernel(overloads=overloads_for(__name__, "solve_template_stage2"), cache=True, nogil=True)
def solve_template_stage2(
    emissions, cols, sizes, focals, penalty, start, end, store_limit, reverse, work, code, value, paths, frontier1
):
    """solve_template's results from stage 1's: per focal (status, rows, box_certified, dp, frontier, fixing_dps,
    seeded_tie), equal to solve_template's for the same convention."""
    end = _closed_end(end, penalty)
    units = len(emissions)
    founders = cols.shape[1]
    offsets = np.zeros(units + 1, np.int64)
    for u in range(units):
        offsets[u + 1] = offsets[u] + sizes[u]
    cands = np.empty(offsets[units], np.int64)
    for u in range(units):
        for r in range(sizes[u]):
            cands[offsets[u] + r] = sizes[u] - 1 - r if reverse else r
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
    tied = np.zeros(n, np.int64)
    if n == 0:
        return status, rows, box, dp, frontier, fixing, tied
    start_n, start_c = _closed_start(start, penalty)
    needed = False
    for i in range(n):
        if code[i] == S1_CANONICAL or code[i] == S1_LIMIT:
            needed = True
    incumbent_value = 0.0
    tables = List()
    if needed:
        k0 = focals[0]
        incumbent_value = path_value(
            emissions, cols, k0, np.ascontiguousarray(cols[:, k0]), first, second, penalty, start_n, start_c, end
        )
    for u in range(units):
        if needed:
            tables.append(_state_tables(emissions[u], cols[u], first, second))
    for i in range(n):
        k = focals[i]
        incumbent = np.ascontiguousarray(cols[:, k])
        rows[i] = incumbent
        c = code[i]
        if c == S1_CERTIFIED:
            status[i] = UNCHANGED
            box[i] = True
            continue
        dp[i] = True
        frontier[i] = frontier1[i]
        if c == S1_LOST:
            status[i] = LOST
            continue
        if c == S1_UNCHANGED:
            status[i] = UNCHANGED
            continue
        fz, bg = _focal_layout(k, first, second)
        prefixes, suffixes, segs = List(), List(), List()
        for u in range(units):
            P, T, best, bx, by, mx, my = tables[u]
            prefix, suffix, seg = _unit_tables(P, T, best, bx, by, mx, my, k, bg)
            prefixes.append(prefix)
            suffixes.append(suffix)
            segs.append(seg)
        upper = window_box(
            emissions, cols, cands, offsets, k, first, second, penalty, end, fz, bg, prefixes, suffixes, segs
        )
        tol = CANONICAL_TOL + PROGRAMME_RELATIVE_MARGIN * max(1.0, abs(incumbent_value))
        reached_most = frontier1[i]
        fixed = 0
        if c == S1_CANONICAL:
            target = value[i] - tol
            st, chosen, fixed, reached = _canonical_restricted(
                emissions,
                cols,
                cands,
                offsets,
                k,
                incumbent,
                first,
                second,
                penalty,
                start_n,
                start_c,
                end,
                upper,
                np.ascontiguousarray(paths[i]),
                target,
                HYBRID_FRONTIER,
                store_limit,
                fz,
                bg,
                prefixes,
                suffixes,
                segs,
                work,
            )
            if reached > reached_most:
                reached_most = reached
            if st != BACKSTOP:
                changed = False
                for u in range(units):
                    if chosen[u] != incumbent[u]:
                        changed = True
                rows[i] = chosen
                status[i] = CHANGED if changed else UNCHANGED
                frontier[i] = reached_most
                fixing[i] = fixed
                continue
        # Frontier overflow (in the programme or in the canonical fixing): solve_focal's seeded search.
        samples = start_n.shape[0]
        total_bins = 0
        max_rows = 0
        reversed_order = False
        for u in range(units):
            total_bins += emissions[u].shape[2]
            max_rows = max(max_rows, offsets[u + 1] - offsets[u])
            if offsets[u + 1] - offsets[u] > 1:
                reversed_order = cands[offsets[u]] > cands[offsets[u + 1] - 1]
        seeds = samples + 1
        starts = 1 + SEED_STARTS + INTERVAL_SEED_STARTS
        _debit(work, 15, 128, samples, total_bins + units + 1, states + max_rows * len(fz) + 1)
        _debit(work, 15, 128, samples, total_bins, total_bins, states + 1)
        _debit(work, 15, 128, samples, max_rows, total_bins, len(fz) + total_bins + 1)
        _debit(work, 15, 128, seeds, seeds, units)
        _debit(work, 15, 128, samples, units, units, total_bins)
        _debit(work, 15, 128, samples, seeds, units, total_bins * (len(fz) + total_bins + 1) + units * len(fz))
        _debit(work, 15, 128, INTERVAL_SEED_STARTS + SEED_STARTS + 1, seeds, units, units * (units + 1))
        _debit(
            work,
            15,
            128,
            starts,
            LOCAL_ITERATIONS,
            samples,
            (max_rows + 1) * total_bins * (len(fz) + total_bins + 1) + units * total_bins,
        )
        _debit(work, 15, 128, starts, LOCAL_ITERATIONS, len(cands) + 1, units)
        _debit(work, 15, 128, samples, total_bins, states + 1)
        fe, be, unit, bins = seeded_search.prepare(emissions, cols, cands, offsets, k, first, second, fz, bg)
        whole = start_n + start_c.reshape((len(start_c), 1))
        seeded, gained, stats = seeded_search.search(
            fe, be, unit, bins, cands, offsets, incumbent, fz, bg, penalty, whole, end, upper, tol, reversed_order
        )
        exact = path_value(emissions, cols, k, seeded, first, second, penalty, start_n, start_c, end)
        rows[i] = seeded
        status[i] = CHANGED if exact > incumbent_value + tol else UNCHANGED
        frontier[i] = reached_most
        fixing[i] = fixed
        tied[i] = stats[7]
    return status, rows, box, dp, frontier, fixing, tied


@kernel(overloads=overloads_for(__name__, "slice_list"), cache=True, nogil=True)
def slice_list(items, a, c):
    """Typed list of items[a:c] (without a Python-level list construction).

    O(W) time for W selected units.
    """
    out = List()
    for i in range(a, c):
        out.append(items[i])
    return out


@kernel(overloads=overloads_for(__name__, "gather_macro"), cache=True, parallel=True, nogil=True)
def gather_macro(emissions, groups, rows_by_group):
    """Coarse-unit emissions: for L1 span groups[g] = (first, end) prepared blocks and its alphabet
    rows_by_group[g] (A_g row tuples over the span), out[g][s, pair_index(a, b), :] concatenates the blocks'
    binned emissions of the row pairs (rows[a, block], rows[b, block]).

    One task per group, which reads each source block of its group from the typed List once and copies it for
    every sample: every List element read updates a reference count shared by all threads, including the window
    workers that gather their problems concurrently from the component's one List, so reads per (group, sample,
    block) serialised them.

    O(N sum_g A_g^2 B_g) time for group alphabet A_g and bin count B_g.
    """
    outputs = List()
    for group in range(len(groups)):
        start, stop = groups[group]
        bins = 0
        for block in range(start, stop):
            bins += emissions[block].shape[2]
        count = len(rows_by_group[group])
        outputs.append(np.empty((emissions[0].shape[0], count * (count + 1) // 2, bins)))
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
                    for b in range(a, len(rows)):
                        bb = rows[b, block - start]
                        for site in range(value.shape[2]):
                            output[sample, pair_index(a, b), offset + site] = value[sample, pair_index(aa, bb), site]
            offset += value.shape[2]
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
            starts = sorted(set(range(0, units - COARSE_WINDOW + 1, COARSE_STRIDE)) | {units - COARSE_WINDOW})
        tasks.extend(("coarse", i, min(units, i + COARSE_WINDOW)) for i in starts)
    return tasks


def _work_shape_terms(samples, width, founders, rows, bins, frontier):
    """Python integers for shape expressions formed before compiled factored debits."""
    n, w, k, r, b, p = map(int, (samples, width, founders, rows, bins, frontier))
    q = k * (k + 1) // 2
    row = b * k + b * b + 1
    return (
        k * (k + 1),
        r * (r + 1),
        w * r,
        q * (b + 1) ** 2 + k + r + 1,
        w * (k + r + 1) + q + 1,
        q * (b + 1) + r * row + q + b * b + k + r + 1,
        (r + 1) * (q * (b + 1) + row) + q * (b + 1),
        w * (q * (b + 1) + 1) + q + 1,
        4 * n * q + w + 1,
        q + w + 1,
        q * (b + 1) + row,
        n * (q + k + b + 1) + w + 1,
        p + r + 1,
        # Children, sort doubling, frontier/history growth and storage comparison.
        2 * p * r,
        p * r * (k + b + 1),
        2 * p * w,
        n + 1,
        w * b + w + 1,
        q + r * k + 1,
        (w * b) * (k + w * b + 1) + w * k,
        (r + 1) * w * b * (k + w * b + 1) + w * w * b,
        w * (w + 1),
    )


def _new_work(work_budget, samples, width, founders, rows, bins, frontier):
    budget = int(work_budget)
    if budget <= 0:
        raise SettingsError("window work budget must be positive")
    shift = max(0, budget.bit_length() - 61)
    if budget >> shift > 1 << 60:
        shift += 1
    if shift > 31:
        raise SettingsError("window work budget exceeds scaled int64 accounting range")
    if max(_work_shape_terms(samples, width, founders, rows, bins, frontier)) >= 2**63:
        raise SettingsError("window work shape exceeds supported int64 accounting")
    # work[21]: reserved zero for existing beam-truncation receipt readers; work[22:24]: shared debits.
    work = np.zeros(WINDOW_SHARED + 2, np.int64)
    work[0], work[20] = budget >> shift, shift
    return work


def _work_receipt(work, budget):
    """One final cumulative receipt per template; raw units use Python integers."""
    shift = int(work[20])
    phases = tuple(map(int, work[4:20]))
    return dict(
        account_scope="template",
        work_budget=int(budget),
        work_used=int(work[1]) << shift,
        phase_work=tuple(p << shift for p in phases),
        work_quantum=1 << shift,
        work_used_units=int(work[1]),
        phase_work_units=phases,
        beam_truncations=int(work[21]) if len(work) > 21 else 0,
    )


def focal_chunks(samples, units, bins, founders, focals):
    """The most chunks a template's focal problems are worth splitting into: each chunk keeps at least
    THREAD_WORK_FLOOR cells of focal work (a focal problem's sample-unit-state-bin cells, samples x units x
    states x (bins + 1), a lower bound on its work), the work floor of the sample kernels' threads, so a chunk's
    fixed costs (its thread and its own copy of the window-shared tables) stay small beside its work. Chunks
    change scheduling only (merge_focal_chunks).
    """
    states = founders * (founders + 1) // 2
    cells = samples * units * states * (bins + 1)
    return max(1, min(focals, focals * cells // THREAD_WORK_FLOOR))


def merge_focal_chunks(chunks):
    """Merge independent focal chunks into the one-template result and account.

    Every chunk solves once and pays the same window-shared debits. Count them once in the merged account. If
    any chunk or the total exhausts the allowance, return None so the single call supplies the deterministic
    stop receipt.
    """
    if any(work[2] == WORK_LIMIT for _, _, work in chunks):
        return None
    work = chunks[0][2].copy()
    budget = int(work[0]) + int(work[1])
    for _, _, other in chunks[1:]:
        work[1] += other[1]
        work[4:20] += other[4:20]
        work[21:] += other[21:]
    for phase, slot in ((0, WINDOW_SHARED), (13, WINDOW_SHARED + 1)):
        repeated = int(work[slot]) - int(chunks[0][2][slot])
        work[4 + phase] -= repeated
        work[slot] -= repeated
        work[1] -= repeated
    if int(work[1]) > budget:
        return None
    work[0] = budget - int(work[1])
    solved = {}
    for part, _, _ in chunks:
        solved.update(part)
    return solved, 0, work


class WindowSolver:
    """The window problems of one component: templates, coarse alphabets and their hybrid solves.

    arrays: the prepared blocks' binned emissions float64 (S, R_b * (R_b + 1) // 2, bins_b) (workspace).
    context: None, or (groups, paths) of the L1 context (run.macro_context): groups[u] = (first,
    end) prepared blocks of L1 span u (fully contained spans; uncovered edges are allowed), paths[u] = int64[P_u, end -
    first] the distinct L1 row paths offered on that span. penalty: SWITCH_PENALTY. Every solve
    uses a fixed frontier and dimension-derived seeded-search storage.
    """

    def __init__(self, arrays, context, penalty, work_coefficient, source_start=0):
        if not isinstance(work_coefficient, (int, np.integer)) or work_coefficient <= 0:
            raise SettingsError("window_work_coefficient must be a positive integer")
        self.work_coefficient = int(work_coefficient)
        self.work = WindowWorkState(source_start)
        self.arrays = list(arrays)
        self.sizes = [(isqrt(8 * array.shape[1] + 1) - 1) // 2 for array in self.arrays]
        # One typed List for the kernels (built once: its construction copies every reference).
        self.emissions = List(self.arrays)
        self.blocks = len(self.arrays)
        self.penalty = float(penalty)
        self.coarse = None
        if context is not None:
            groups, paths = context
            self.coarse = (
                [tuple(int(x) for x in group) for group in groups],
                [np.ascontiguousarray(path, np.int64) for path in paths],
            )
        self.templates = window_templates(self.blocks, None if self.coarse is None else self.coarse[0])
        # Block range [a, c) of every template (proposal caches hold flank messages there).
        self.spans = [
            (a, c) if kind == "fine" else (self.coarse[0][a][0], self.coarse[0][c - 1][1])
            for kind, a, c in self.templates
        ]

    @property
    def coarse_units(self):
        return None if self.coarse is None else len(self.coarse[0])

    def template_bytes(self, kind, a, c, samples, founders):
        """Planned bytes of one window solve (the search's item estimate): the window's per-state
        unit sums (_state_tables) and one focal problem's unit tables (each two per unit and bin plus one per unit)
        and box bounds of S x Q float64, a planned frontier of HYBRID_FRONTIER paths (two and a half copies of S
        (Q + K + bins + 2) float64 each) and, for coarse windows, the gathered span emissions (S x A(A+1)/2 x bins
        float64 per unit, A <= founders + context paths) and their multi-bin tables.
        The seeded search holds gathered emissions, background runs, wish restrictions,
        boundary focal values and suffix covectors for one active focal."""
        states = founders * (founders + 1) // 2
        if kind == "fine":
            unit_bins = [int(array.shape[2]) for array in self.arrays[a:c]]
            alphabets = None
        else:
            groups, context = self.coarse
            unit_bins = [sum(int(self.arrays[b].shape[2]) for b in range(*groups[u])) for u in range(a, c)]
            alphabets = [founders + len(context[u]) for u in range(a, c)]
        bins = max(unit_bins)
        tables = 8 * samples * states * (4 * sum(unit_bins) + 2 * len(unit_bins) + 1)
        frontier = 20 * samples * (states + founders + bins + 2) * HYBRID_FRONTIER
        coarse = (
            0
            if alphabets is None
            else sum(
                8 * samples * (width * (width + 1) // 2) * unit + 16 * samples * states * unit
                for width, unit in zip(alphabets, unit_bins, strict=True)
            )
        )
        total_bins = sum(unit_bins)
        rows = max(self.sizes[a:c]) if alphabets is None else max(alphabets)
        search = seeded_search_bytes(samples, len(unit_bins), total_bins, rows, founders)
        return tables + frontier + coarse + search

    def _template(
        self, emissions, bins, cols, sizes, start, end, focals=None, convention=False, team=None, stage1=None, split=1
    ):
        """Every (listed) founder's problem of one window in one nogil call (bins: the largest
        unit's bins). Returns [(focal, unit choices or None, diagnostic)].

        team: None, or the template's CPU share (a RefinementExecution). With more than one CPU and enough work
        (focal_chunks), the focal problems, which are independent, run as interleaved chunks on the team's map,
        each with its own account, and merge_focal_chunks merges them into the one-call result; when the merged
        account shows the one-call account exhausted, the template is solved in one call, which gives its
        exhaustion receipt. stage1: this window problem's convention-free stage (WindowSolver.stage1, all focals),
        or None; it is then finished in this convention (solve_template_stage2: the same results and account).
        split: the chunks the caller's map wants this template in (SweepEngine.proposals: a template that alone would
        set the map's critical path), run on the team's map as its share grows; at most focal_chunks."""
        # The exact objective's backward messages end with a per-homologue transition
        # (up to 2c below the maximum, c map-derived), while the surrogate's bounds and dominance
        # assume a flank closed under its own flat switch (entries >= max - penalty). Close the end
        # flank under the surrogate's switch, as _closed_start does for the start flank; proposals only.
        end = np.maximum(end, end.max(axis=1, keepdims=True) - self.penalty)
        # Dimensions are those of this actual fine/coarse problem, including temporary
        # duplicate-seed founders and all distinct context paths. Python-int products
        # precede scaled integer work accounting; there is no small-K cap or off mode.
        founders = int(cols.shape[1])
        samples = int(start.shape[0])
        width = len(emissions)
        bins = max(int(array.shape[2]) for array in emissions)
        rows = max(map(int, sizes))
        height = max(founders, rows)
        if min(samples, width, bins, founders, rows) <= 0:
            raise InputError("window work requires nonempty samples, units, bins and rows")
        work_budget = self.work_coefficient * samples * width * bins * bins * height**3
        work_shape = dict(samples=samples, units=width, bins=bins, founders=founders, alphabet=rows, height=height)
        focals = np.arange(founders, dtype=np.int64) if focals is None else np.asarray(focals, np.int64)
        samples, states = start.shape
        store_limit = max(1, STORE_BYTES // max(1, 8 * samples * states))
        cols = np.ascontiguousarray(cols, np.int64)
        sizes = np.asarray(sizes, np.int64)
        account = (
            work_budget,
            samples,
            width,
            founders,
            rows,
            bins,
            HYBRID_FRONTIER,
        )

        def solve_chunk(chosen):
            return self._solve_chunk(emissions, cols, sizes, chosen, start, end, store_limit, convention, account)

        solution = None
        if stage1 is not None and len(focals) == founders:
            solution = self._solve_stage2(emissions, cols, sizes, focals, start, end, store_limit, convention, stage1)
            if solution[2][2] == WORK_LIMIT:
                # The two stages debit in another order than one call: the one call below gives the exhaustion
                # receipt (phase and units at the limit), as for exhausted focal chunks.
                solution = None
        if solution is not None:
            solved, step, work = solution
        else:
            chunks = 1
            if team is not None:
                chunks = min(
                    len(focals),
                    max(int(team.threads()), int(split)),
                    focal_chunks(samples, width, bins, founders, len(focals)),
                )
            solution = None
            if chunks > 1:
                # Contiguous chunks: the kernels' declared signatures take C-ordered focal lists.
                parts = [np.ascontiguousarray(focals[c::chunks]) for c in range(chunks)]
                solution = merge_focal_chunks(
                    team.thread_map(solve_chunk, parts, label="founder_refinement focal chunks")
                )
            solved, step, work = solve_chunk(focals) if solution is None else solution
        if work[2] == WORK_LIMIT:
            # Discard every focal result of the template.
            out = [
                (
                    int(k),
                    None,
                    dict(
                        certified=False,
                        dp=False,
                        frontier=0,
                        fixing_dps=0,
                        fallback=False,
                        status=WORK_LIMIT,
                        work_limit=True,
                        work_phase=int(work[3]),
                        work_retries=step,
                        work_discarded=len(solved),
                    ),
                )
                for k in focals
            ]
            if out:
                out[0][2].update(
                    _work_receipt(work, work_budget), work_coefficient=self.work_coefficient, work_shape=work_shape
                )
            return out
        out = []
        for k in focals:
            status, rows, box, dp, frontier, fixing, tied = solved[int(k)]
            if status == LOST:
                raise InvariantError("window dynamic programme lost every feasible path")
            out.append(
                (
                    int(k),
                    rows if status == CHANGED else None,
                    dict(
                        certified=bool(box),
                        dp=bool(dp),
                        frontier=int(frontier),
                        fixing_dps=int(fixing),
                        fallback=bool(status == FRONTIER_LIMIT),
                        beam_fallback=bool(status == BEAM_LIMIT),
                        status=int(status),
                        work_limit=False,
                        work_retries=step,
                        seeded_tie=bool(tied == 2),
                        seeded_tie_any=bool(tied >= 1),
                    ),
                )
            )
        if out:
            out[0][2].update(
                _work_receipt(work, work_budget), work_coefficient=self.work_coefficient, work_shape=work_shape
            )
        return out

    def _solve_chunk(self, emissions, cols, sizes, focals, start, end, store_limit, convention, account):
        """Solve listed focals once; the merger owns the one-template account."""
        work = _new_work(*account)
        solved = {}
        try:
            status, rows, box, dp, frontier, fixing, tied = solve_template(
                emissions,
                cols,
                sizes,
                focals,
                self.penalty,
                start,
                end,
                int(store_limit),
                bool(convention),
                work,
            )
        except RuntimeError:
            if work[2] != WORK_LIMIT:
                raise
            return solved, 0, work
        for i, k in enumerate(focals):
            solved[int(k)] = (status[i], rows[i].copy(), box[i], dp[i], frontier[i], fixing[i], tied[i])
        return solved, 0, work

    def _stage1(self, emissions, cols, sizes, start, end):
        """The convention-free stage of one window problem (solve_template_stage1, every focal) with its work
        account; the inputs are prepared as _template prepares them."""
        end = np.maximum(end, end.max(axis=1, keepdims=True) - self.penalty)
        founders = int(cols.shape[1])
        samples = int(start.shape[0])
        width = len(emissions)
        bins = max(int(array.shape[2]) for array in emissions)
        rows = max(map(int, sizes))
        height = max(founders, rows)
        work_budget = self.work_coefficient * samples * width * bins * bins * height**3
        focals = np.arange(founders, dtype=np.int64)
        samples, states = start.shape
        store_limit = max(1, STORE_BYTES // max(1, 8 * samples * states))
        cols = np.ascontiguousarray(cols, np.int64)
        sizes = np.asarray(sizes, np.int64)
        work = _new_work(work_budget, samples, width, founders, rows, bins, HYBRID_FRONTIER)
        try:
            code, value, paths, frontier = solve_template_stage1(
                emissions, cols, sizes, focals, self.penalty, start, end, int(store_limit), work
            )
        except RuntimeError:
            if work[2] != WORK_LIMIT:
                raise
            return dict(work=work, limited=True)
        return dict(code=code, value=value, paths=paths, frontier=frontier, work=work, limited=False)

    def _solve_stage2(self, emissions, cols, sizes, focals, start, end, store_limit, convention, stage1):
        """_solve_chunk of every focal from a stage-1 state: the account continues stage 1's (a copy)."""
        work = stage1["work"].copy()
        solved = {}
        if stage1["limited"]:
            return solved, 0, work
        try:
            status, rows, box, dp, frontier, fixing, tied = solve_template_stage2(
                emissions,
                cols,
                sizes,
                focals,
                self.penalty,
                start,
                end,
                int(store_limit),
                bool(convention),
                work,
                stage1["code"],
                stage1["value"],
                stage1["paths"],
                stage1["frontier"],
            )
        except RuntimeError:
            if work[2] != WORK_LIMIT:
                raise
            return solved, 0, work
        for i, k in enumerate(focals):
            solved[int(k)] = (status[i], rows[i].copy(), box[i], dp[i], frontier[i], fixing[i], tied[i])
        return solved, 0, work

    def stage1(self, index, X, fwd, bwd):
        """The convention-free stage of template `index` for panel X with flanks fwd, bwd (every focal)."""
        kind, a, c = self.templates[index]
        if kind == "fine":
            emissions = slice_list(self.emissions, a, c)
            return self._stage1(emissions, X[:, a:c].T, self.sizes[a:c], fwd[a], bwd[c])
        emissions, cols, sizes, _, a0, c0 = self._coarse_problem(X, a, c)
        return self._stage1(emissions, cols, sizes, fwd[a0], bwd[c0])

    def _coarse_problem(self, X, i, j):
        """coarse_window's inputs over L1 spans [i, j): (gathered emissions, cols, sizes, alphabets, first block,
        end block)."""
        groups, context = self.coarse
        alphabets, lookups = [], []
        for unit in range(i, j):
            start, stop = groups[unit]
            distinct = {}
            for row in [*X[:, start:stop], *context[unit]]:
                row = np.ascontiguousarray(row, np.int64)
                distinct.setdefault(row.tobytes(), row)
            rows = sorted(distinct.values(), key=lambda row: tuple(int(v) for v in row))
            lookup = {row.tobytes(): index for index, row in enumerate(rows)}
            alphabets.append(np.asarray(rows, np.int64))
            lookups.append(lookup)
        emissions = gather_macro(self.emissions, np.asarray(groups[i:j], np.int64), List(alphabets))
        cols = np.empty((j - i, len(X)), np.int64)
        for u, unit in enumerate(range(i, j)):
            start, stop = groups[unit]
            for founder in range(len(X)):
                cols[u, founder] = lookups[u][np.ascontiguousarray(X[founder, start:stop], np.int64).tobytes()]
        return emissions, cols, [len(x) for x in alphabets], alphabets, groups[i][0], groups[j - 1][1]

    def focal_chunk_cap(self, index, samples, founders):
        """focal_chunks of template ``index`` for a panel of ``founders``: the most CPUs its solve can use."""
        kind, a, c = self.templates[index]
        if kind == "fine":
            bins = max(int(self.arrays[b].shape[2]) for b in range(a, c))
        else:
            groups = self.coarse[0]
            bins = max(sum(int(self.arrays[b].shape[2]) for b in range(*groups[unit])) for unit in range(a, c))
        return focal_chunks(samples, c - a, bins, founders, founders)

    def fine(
        self,
        X,
        fwd,
        bwd,
        a,
        c,
        focals=None,
        convention=False,
        *,
        ordinal=None,
        team=None,
        stage1=None,
        completion=False,
        split=1,
    ):
        """Proposals of every (listed) founder on blocks [a, c): [(kind, a, c, k, rows | None,
        diagnostic)]; team: the solve's CPU share or None (_template); completion: no work receipt is recorded
        (solve)."""
        self.work.check(ordinal)
        emissions = slice_list(self.emissions, a, c)
        sizes = self.sizes[a:c]
        bins = max(self.arrays[b].shape[2] for b in range(a, c))
        solved = self._template(
            emissions, bins, X[:, a:c].T, sizes, fwd[a], bwd[c], focals, convention, team, stage1=stage1, split=split
        )
        if not completion:
            self.work.finish_bound(
                ordinal, solved, X, "fine", [(b, b + 1) for b in range(a, c)], sizes, None, bool(convention)
            )
        return [("fine", a, c, k, rows, diagnostic) for k, rows, diagnostic in solved]

    def coarse_window(
        self,
        X,
        fwd,
        bwd,
        i,
        j,
        convention=False,
        *,
        ordinal=None,
        team=None,
        stage1=None,
        focals=None,
        completion=False,
        split=1,
    ):
        """Proposals of every founder over L1 spans [i, j). A span's alphabet: the distinct row
        tuples the founders use there and the L1 context paths, sorted lexicographically. The
        window programme's canonical-tie order "incumbent, then ascending index" therefore
        means incumbent, then content."""
        self.work.check(ordinal)
        groups, _ = self.coarse
        emissions, cols, sizes, alphabets, a, c = self._coarse_problem(X, i, j)
        bins = max(sum(self.arrays[b].shape[2] for b in range(*groups[unit])) for unit in range(i, j))
        solved = self._template(
            emissions,
            bins,
            cols,
            sizes,
            fwd[a],
            bwd[c],
            focals,
            convention=convention,
            team=team,
            stage1=stage1,
            split=split,
        )
        if not completion:
            self.work.finish_bound(ordinal, solved, X, "coarse", groups[i:j], sizes, alphabets, bool(convention))
        out = []
        for k, choice, diagnostic in solved:
            rows = None
            if choice is not None:
                rows = np.concatenate([alphabets[u][int(choice[u])] for u in range(j - i)])
                if np.array_equal(rows, X[k, a:c]):
                    rows = None
            out.append(("coarse", a, c, k, rows, diagnostic))
        return out

    def solve(
        self,
        index,
        X,
        fwd,
        bwd,
        convention=False,
        *,
        ordinal=None,
        team=None,
        stage1=None,
        focals=None,
        completion=False,
        split=1,
    ):
        """The proposals of template `index` for panel X with messages fwd, bwd (a pure function of
        X's rows in the window and the flank messages fwd[a], bwd[c], within its work account);
        team: the solve's CPU share or None (_template); stage1: its convention-free stage for this panel and
        these flanks (WindowSolver.stage1), or None; focals: the founders to solve (default all). completion:
        the listed focals of a template whose other focals are reused exactly (a restated template: cache.relabel):
        the call is checked against the logical stop (work.check) but records no work receipt, since the caller
        receipts the whole template."""
        kind, a, c = self.templates[index]
        if kind == "fine":
            return self.fine(
                X,
                fwd,
                bwd,
                a,
                c,
                focals,
                convention=convention,
                ordinal=ordinal,
                team=team,
                stage1=stage1,
                completion=completion,
                split=split,
            )
        return self.coarse_window(
            X,
            fwd,
            bwd,
            a,
            c,
            convention,
            ordinal=ordinal,
            team=team,
            stage1=stage1,
            focals=focals,
            completion=completion,
            split=split,
        )


__all__ = [
    "window_box",
    "window_dp",
    "path_value",
    "solve_focal",
    "solve_focal_reference",
    "solve_template",
    "solve_template_stage1",
    "solve_template_stage2",
    "slice_list",
    "gather_macro",
    "window_templates",
    "focal_chunks",
    "merge_focal_chunks",
    "WindowSolver",
]
