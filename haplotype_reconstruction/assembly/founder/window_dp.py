"""Exact founder-row proposals on bounded windows of the binned refinement model.

Window problem (one founder f, window of W units = prepared blocks, or L1 spans for
coarse copy moves): maximise the binned surrogate

    H(x) = sum_s max over bin-constant state paths [ fwd_s(a) (x) bins of [a, c) with rows (X_-f, x) (x) bwd_s(c) ]

over f's rows x (one alphabet per unit, every other founder fixed). The rest of the
chromosome enters exactly through the canonical normalised flank messages of the
current panel; the interior uses the component's production binned emissions
(founder/evidence.py, the level's own bin size), with the same switch-then-emission
bin recursion as packing.score_selected.

Solver. (1) An optional SCREEN runs the production Lagrangian dual
(dual_short/dual_search kernels, from zero multipliers) on the identical problem
posed with two virtual flank blocks. Any zero-sum multipliers give a valid upper
bound, so a bound <= H(incumbent) + tol certifies that no rewrite improves on the
incumbent; the screen's sweep limit decides only how much work is spent. (2) Every
window the screen does not certify is solved by an exact forward frontier dynamic
programme (DX exactdp2): a partial path keeps its closed normalised per-sample
message; a partial path p is dropped only if another partial path y is at least as
good for EVERY completion, sum_s max_z (M_p - M_y) + (T_p - T_y) <= tol (closed-
message dominance), or if a valid upper bound on all its completions (each sample
choosing f's later rows freely) lies below a feasible value. A programme whose frontier
passes EXPLORE_FRONTIER restarts with tighter pruning: Lagrangian suffix bounds (each
sample chooses f's later rows freely but pays zero-sum multipliers lam_s(unit, row),
which cancel on every common path; the multipliers come from block-coordinate sweeps
that equalise the samples' max-marginals) and the best feasible value among the
incumbent, a beam and the dual's decoded consensus paths. Near-identical founders make
the box bound loose by a sum of small per-sample gains, so the frontier keeps growing;
the dual bound is usually tight there. The canonical-tie programmes take the same restart
(the bounds are computed once per problem). Pruning still keeps every path within the
margin of a feasible value, so the optimum and the canonical answer are unchanged. No
frontier cap exists; a frontier larger than the backstop raises an error naming the window.

Canonical answer (a pure function of the problem): among paths within
tol = 1e-7 + 1e-9 |H(incumbent)| of the optimum, the incumbent if it is one, else the
lexicographically first path, units left to right, each in the order (incumbent row,
then ascending index), found by exact restricted dynamic programmes.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange
from numba.typed import List

from .packing import PreparedModels, emission_arrays, packed_emissions, score_rows, short_models
from . import dual_search as ds, dual_short as dsh

DOMINANCE_TOLERANCE = 1e-6
# Relative float margin between the dual screen's and the programme's summation orders.
SCREEN_ARITHMETIC_MARGIN = 5e-10
STORE_BYTES = 128 * 1024 ** 2
SCREEN_STALL = 1e-2
SCREEN_STALL_WINDOW = 10

OK, INFEASIBLE, BACKSTOP = 0, 1, 2
# A programme whose frontier passes EXPLORE_FRONTIER restarts with tighter pruning (work only,
# never the answer): the Lagrangian suffix bounds of the window problem and the best feasible
# value among the incumbent, a BEAM_WIDTH-wide beam and the dual's decoded paths. The multiplier
# search runs block-coordinate sweeps until its bound is within the canonical tolerance of a
# feasible value, at most DUAL_SWEEPS (a sweep can leave the bound unchanged while the decoded
# paths still improve, so no stall rule); any zero-sum multipliers give valid bounds, so the
# limit never decides an answer.
EXPLORE_FRONTIER = 256
BEAM_WIDTH = 16
DUAL_SWEEPS = 100


def canonical_tolerance(value):
    return 1e-7 + 1e-9 * max(1.0, abs(value))


@njit(cache=True, parallel=True, nogil=True)
def site_emissions(leaf, logs, start, stop):
    """Per-site canonical emissions of one block for every pair of its local rows.

    out[s, a, b, j] = log-evidence of dosage leaf[a, j] + leaf[b, j] at site start + j,
    centred as founder/scoring.py; exactly 0 at sites without evidence for sample s
    (incomplete or uninformative), where a switch opportunity changes no value. Used as
    one-site "bins", the window problem is the canonical full-site F of the window.
    """
    rows = leaf.shape[0]
    samples = logs.shape[0]
    center = np.log(1.0 / 3.0)
    out = np.zeros((samples, rows, rows, stop - start))
    for s in prange(samples):
        for j in range(stop - start):
            site = start + j
            if logs[s, site, 0] == 0.0 and logs[s, site, 1] == 0.0 and logs[s, site, 2] == 0.0:
                continue
            for a in range(rows):
                for b in range(rows):
                    out[s, a, b, j] = logs[s, site, leaf[a, j] + leaf[b, j]] - center
    return out


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
            m = out_n[s, 0]
            for z in range(1, states):
                if out_n[s, z] > m:
                    m = out_n[s, z]
            sw = m - penalty
            for z in range(states):
                a = first[z]
                b = second[z]
                ra = row if a == focal else col[a]
                rb = row if b == focal else col[b]
                v = out_n[s, z]
                if sw > v:
                    v = sw
                out_n[s, z] = v + emis[s, ra, rb, bn]
        m = out_n[s, 0]
        for z in range(1, states):
            if out_n[s, z] > m:
                m = out_n[s, z]
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
    samples, states = msg_n.shape
    total = 0.0
    for s in range(samples):
        best = -np.inf
        for z in range(states):
            v = msg_n[s, z] + tail[s, z]
            if v > best:
                best = v
        total += msg_c[s] + best
    return total


@njit(cache=True, nogil=True)
def window_box_reference(emissions, cols, cands, offsets, focal, first, second, penalty, end):
    """Validation reference of window_box (every row over every state). U[u] >= the value of units u.. plus the end flank from any state, per sample.

    Each sample may choose f's row of every later unit freely (a relaxation); the
    backward bin recursion is emission-then-switch, matching the forward one.
    """
    units = len(emissions)
    samples, states = end.shape
    upper = np.empty((units + 1, samples, states))
    upper[units] = end
    tmp = np.empty(states)
    for u in range(units - 1, -1, -1):
        emis = emissions[u]
        bins = emis.shape[3]
        col = cols[u]
        for s in range(samples):
            for z in range(states):
                upper[u, s, z] = -np.inf
            for g in range(offsets[u], offsets[u + 1]):
                row = cands[g]
                for z in range(states):
                    tmp[z] = upper[u + 1, s, z]
                for bn in range(bins - 1, -1, -1):
                    m = -np.inf
                    for z in range(states):
                        a = first[z]
                        b = second[z]
                        ra = row if a == focal else col[a]
                        rb = row if b == focal else col[b]
                        tmp[z] += emis[s, ra, rb, bn]
                        if tmp[z] > m:
                            m = tmp[z]
                    sw = m - penalty
                    for z in range(states):
                        if sw > tmp[z]:
                            tmp[z] = sw
                for z in range(states):
                    if tmp[z] > upper[u, s, z]:
                        upper[u, s, z] = tmp[z]
    return upper


@njit(cache=True, nogil=True)
def _dominance(n_i, t_i, n_j, t_j, tol):
    """True iff no completion makes i better than j by more than tol."""
    samples, states = n_i.shape
    d = t_i - t_j
    if d > tol:
        return False
    for s in range(samples):
        best = n_i[s, 0] - n_j[s, 0]
        for z in range(1, states):
            v = n_i[s, z] - n_j[s, z]
            if v > best:
                best = v
        d += best
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


@njit(cache=True, nogil=True)
def window_dp_reference(emissions, cols, cands, offsets, focal, incumbent, first, second, penalty,
                        start_n, start_c, unit0, end, upper, lower, margin, dom_tol, backstop,
                        store_limit):
    """Validation reference of window_dp (every child extended over every state). Exact maximum of H over f's rows on units unit0.. from one partial path.

    Returns (status, value, path of units unit0.., maximum frontier). status 1: no
    path reaches lower - margin; status 2: the frontier exceeded `backstop`.
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
    front_n = np.empty((1, samples, states))
    front_c = np.empty((1, samples))
    front_n[0] = start_n
    front_c[0] = start_c
    front_t = np.empty(1)
    front_t[0] = 0.0
    history_par = np.empty(64, np.int64)
    history_cho = np.empty(64, np.int64)
    history_off = np.zeros(width + 1, np.int64)
    max_frontier = 1
    scratch_n = np.empty((samples, states))
    scratch_c = np.empty(samples)
    for u in range(unit0, units):
        emis = emissions[u]
        col = cols[u]
        begin, stop = offsets[u], offsets[u + 1]
        rows = stop - begin
        parents = front_n.shape[0]
        children = parents * rows
        par = np.empty(children, np.int64)
        cho = np.empty(children, np.int64)
        totals = np.empty(children)
        bounds = np.empty(children)
        inc = np.empty(children, np.int64)
        store = children <= store_limit
        stored_n = np.empty((children if store else 0, samples, states))
        stored_c = np.empty((children if store else 0, samples))
        for i in range(children):
            p = i // rows
            r = cands[begin + i % rows]
            par[i] = p
            cho[i] = r
            inc[i] = 1 if r == incumbent[u] else 0
            if store:
                totals[i] = _extend(front_n[p], front_c[p], emis, col, focal, r, first, second,
                                    penalty, stored_n[i], stored_c[i])
                bounds[i] = _join(stored_n[i], stored_c[i], upper[u + 1])
            else:
                totals[i] = _extend(front_n[p], front_c[p], emis, col, focal, r, first, second,
                                    penalty, scratch_n, scratch_c)
                bounds[i] = _join(scratch_n, scratch_c, upper[u + 1])
        alive = 0
        for i in range(children):
            if bounds[i] >= lower - margin:
                alive += 1
        if alive == 0:
            return INFEASIBLE, -np.inf, path, max_frontier
        order = _stable_order(totals, inc, par, cho)
        kept_n = np.empty((min(alive, 64), samples, states))
        kept_c = np.empty((min(alive, 64), samples))
        kept_t = np.empty(min(alive, 64))
        kept_i = np.empty(min(alive, 64), np.int64)
        count = 0
        for position in range(children):
            i = order[position]
            if bounds[i] < lower - margin:
                continue
            if store:
                child_n = stored_n[i]
                child_c = stored_c[i]
            else:
                _extend(front_n[par[i]], front_c[par[i]], emis, col, focal, cho[i], first, second,
                        penalty, scratch_n, scratch_c)
                child_n = scratch_n
                child_c = scratch_c
            dominated = False
            for j in range(count):
                if _dominance(child_n, totals[i], kept_n[j], kept_t[j], dom_tol):
                    dominated = True
                    break
            if dominated:
                continue
            if count == kept_n.shape[0]:
                grow = 2 * count
                new_n = np.empty((grow, samples, states))
                new_c = np.empty((grow, samples))
                new_t = np.empty(grow)
                new_i = np.empty(grow, np.int64)
                new_n[:count] = kept_n[:count]
                new_c[:count] = kept_c[:count]
                new_t[:count] = kept_t[:count]
                new_i[:count] = kept_i[:count]
                kept_n, kept_c, kept_t, kept_i = new_n, new_c, new_t, new_i
            kept_n[count] = child_n
            kept_c[count] = child_c
            kept_t[count] = totals[i]
            kept_i[count] = i
            count += 1
            if count > backstop:
                return BACKSTOP, -np.inf, path, count
        # A kept child may be dominated by one kept after it (later in the order).
        remove = np.zeros(count, np.bool_)
        for x in range(count):
            for y in range(x + 1, count):
                if _dominance(kept_n[x], kept_t[x], kept_n[y], kept_t[y], dom_tol):
                    remove[x] = True
                    break
        survivors = count - int(remove.sum())
        front_n = np.empty((survivors, samples, states))
        front_c = np.empty((survivors, samples))
        front_t = np.empty(survivors)
        hp = np.empty(survivors, np.int64)
        hc = np.empty(survivors, np.int64)
        w = 0
        for x in range(count):
            if remove[x]:
                continue
            front_n[w] = kept_n[x]
            front_c[w] = kept_c[x]
            front_t[w] = kept_t[x]
            hp[w] = par[kept_i[x]]
            hc[w] = cho[kept_i[x]]
            w += 1
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
    for x in range(front_n.shape[0]):
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
# message (materialised by _extend) is bitwise the reference's; for multi-bin units the
# decomposition is exact in real arithmetic. Bounds only prune (a margin covers
# rounding); dominance decisions are taken from the shared-part formulas only when
# they clear the tolerance by a float margin, otherwise both messages are materialised
# and _dominance (the reference test) decides.
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
def _unit_tables(emis, col, first, second, bg):
    """Background sums of one unit: prefix[s, b, k] = sum_{m <= k} e_m, suffix[s, b, k] =
    sum_{m >= k} e_m (suffix[.., t] = 0) and seg[s, a, c] = max_b sum_{m=a}^{c} e_m."""
    samples = emis.shape[0]
    bins = emis.shape[3]
    nb = len(bg)
    prefix = np.empty((samples, nb, bins))
    suffix = np.zeros((samples, nb, bins + 1))
    seg = np.full((samples, bins, bins), -np.inf)
    for s in range(samples):
        for b in range(nb):
            z = bg[b]
            ra = col[first[z]]
            rb = col[second[z]]
            acc = 0.0
            for k in range(bins):
                acc += emis[s, ra, rb, k]
                prefix[s, b, k] = acc
            acc = 0.0
            for k in range(bins - 1, -1, -1):
                acc += emis[s, ra, rb, k]
                suffix[s, b, k] = acc
            for a in range(1 if bins == 1 else bins):
                run = 0.0
                for c in range(a, bins):
                    run += emis[s, ra, rb, c]
                    if run > seg[s, a, c]:
                        seg[s, a, c] = run
    return prefix, suffix, seg


@njit(cache=True, nogil=True)
def _problem_tables(emissions, cols, first, second, bg):
    """_unit_tables of every unit of one problem (shared by the box and every programme)."""
    prefixes = List()
    suffixes = List()
    segs = List()
    for u in range(len(emissions)):
        prefix, suffix, seg = _unit_tables(emissions[u], cols[u], first, second, bg)
        prefixes.append(prefix)
        suffixes.append(suffix)
        segs.append(seg)
    return prefixes, suffixes, segs


@njit(cache=True, nogil=True)
def window_box(emissions, cols, cands, offsets, focal, first, second, penalty, end, fz, bg,
               prefixes, suffixes, segs):
    """U[u] >= the value of units u.. plus the end flank from any state, per sample.

    Each sample may choose f's row of every later unit freely (a relaxation); the
    backward bin recursion is emission-then-switch, matching the forward one. Shared
    background: per row only the K focal states and t switch maxima m_k are carried; the
    background states take max over rows of the floors, which commutes with the
    (monotone) float subtraction of the penalty, so one-bin units equal the reference
    (window_box_reference) bitwise.
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
                upper[u, s, fz[c]] = -np.inf
            for k in range(bins):
                best_m[k] = -np.inf
            for g in range(offsets[u], offsets[u + 1]):
                row = cands[g]
                for c in range(founders):
                    yf[c] = upper[u + 1, s, fz[c]]
                for k in range(bins - 1, -1, -1):
                    m = ue[k]
                    for l in range(k + 1, bins):
                        v = mrow[l] - penalty + seg[s, k, l - 1]
                        if v > m:
                            m = v
                    for c in range(founders):
                        z = fz[c]
                        ra = row if first[z] == focal else col[first[z]]
                        rb = row if second[z] == focal else col[second[z]]
                        yf[c] += emis[s, ra, rb, k]
                        if yf[c] > m:
                            m = yf[c]
                    mrow[k] = m
                    sw = m - penalty
                    for c in range(founders):
                        if sw > yf[c]:
                            yf[c] = sw
                for c in range(founders):
                    if yf[c] > upper[u, s, fz[c]]:
                        upper[u, s, fz[c]] = yf[c]
                for k in range(bins):
                    if mrow[k] > best_m[k]:
                        best_m[k] = mrow[k]
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
def _parent_tables(parent_n, emis, bg, prefix, unext, penalty):
    """Per parent and sample: the parent's maximum, pb[s, k] = max_b P_k[b] (the background
    chain after bin k without later floors), tp[s] = max_b (P_{t-1}[b] + U[b]) and, for
    the dominance shortcuts, the extremes and the argmax of P_{t-1} over the background."""
    samples = parent_n.shape[0]
    bins = emis.shape[3]
    nb = len(bg)
    pmax = np.empty(samples)
    pb = np.empty((samples, bins))
    tp = np.empty(samples)
    hi = np.empty(samples)
    lo = np.empty(samples)
    arg = np.zeros(samples, np.int64)
    for s in range(samples):
        m = parent_n[s, 0]
        for z in range(1, parent_n.shape[1]):
            if parent_n[s, z] > m:
                m = parent_n[s, z]
        pmax[s] = m
        sw = m - penalty
        for k in range(bins):
            pb[s, k] = -np.inf
        tp[s] = -np.inf
        hi[s] = -np.inf
        lo[s] = np.inf
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
    return pmax, pb, tp, hi, lo, arg


@njit(cache=True, nogil=True)
def _child_lazy(parent_n, parent_c, pmax, pb, tp, seg, te, ubmax, unext, emis, col, focal, row,
                first, second, penalty, fz, out_m, out_f, out_s):
    """One child (parent extended by `row` on one unit) without its background states.

    Fills per sample the unit-end maximum out_m, the closed focal values out_f and the
    switch floors out_s[k] = m_k - pen; returns the child's total (sum of its constants,
    the reference's ordering key) and its bound (the reference's _join against U[u+1]).
    """
    samples = parent_n.shape[0]
    bins = emis.shape[3]
    founders = len(fz)
    xf = np.empty(founders)
    total = 0.0
    bound = 0.0
    for s in range(samples):
        for c in range(founders):
            xf[c] = parent_n[s, fz[c]]
        m = pmax[s]
        for k in range(bins):
            if k > 0:
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
                z = fz[c]
                ra = row if first[z] == focal else col[first[z]]
                rb = row if second[z] == focal else col[second[z]]
                v = xf[c]
                if sw > v:
                    v = sw
                xf[c] = v + emis[s, ra, rb, k]
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
            w = v + unext[s, fz[c]]
            if w > best:
                best = w
        bound += cs + best
    return total, bound


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
    samples, states = n_i.shape
    d = t_i - t_j
    if d > tol:
        return False, d
    for s in range(samples):
        best = n_i[s, 0] - n_j[s, 0]
        for z in range(1, states):
            v = n_i[s, z] - n_j[s, z]
            if v > best:
                best = v
        d += best
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
def window_dp(emissions, cols, cands, offsets, focal, incumbent, first, second, penalty,
              start_n, start_c, unit0, end, upper, lower, margin, dom_tol, backstop,
              store_limit, fz, bg, prefixes, suffixes, segs):
    """Exact maximum of H over f's rows on units unit0.. from one partial path.

    Returns (status, value, path of units unit0.., maximum frontier). status 1: no
    path reaches lower - margin; status 2: the frontier exceeded `backstop`.
    Same programme as window_dp_reference (same child order, bound pruning, dominance
    decisions and kept messages): children are formed per parent from the shared
    background (O(S (t K + t^2)) each instead of O(S t Q)); a child's full message is
    materialised (_extend) only when it is kept or must be compared with a child of
    another parent. A child dominated by a kept path j, through an earlier sibling c
    dominated by j, is dropped when d(i, c) + d(c, j) <= tol clears the float margin
    (d is subadditive).
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
    front_n = np.empty((1, samples, states))
    front_c = np.empty((1, samples))
    front_n[0] = start_n
    front_c[0] = start_c
    history_par = np.empty(64, np.int64)
    history_cho = np.empty(64, np.int64)
    history_off = np.zeros(width + 1, np.int64)
    max_frontier = 1
    scratch_n = np.empty((samples, states))
    scratch_c = np.empty(samples)
    for u in range(unit0, units):
        emis = emissions[u]
        col = cols[u]
        bins = emis.shape[3]
        begin, stop = offsets[u], offsets[u + 1]
        rows = stop - begin
        parents = front_n.shape[0]
        children = parents * rows
        unext = upper[u + 1]
        prefix, suffix, seg = prefixes[u], suffixes[u], segs[u]
        nb = len(bg)
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
        p_max = np.empty((parents, samples))
        p_pb = np.empty((parents, samples, bins))
        p_tp = np.empty((parents, samples))
        p_hi = np.empty((parents, samples))
        p_lo = np.empty((parents, samples))
        p_arg = np.empty((parents, samples), np.int64)
        for p in range(parents):
            pm, pbv, tpv, hv, lv, av = _parent_tables(front_n[p], emis, bg, prefix, unext, penalty)
            p_max[p] = pm
            p_pb[p] = pbv
            p_tp[p] = tpv
            p_hi[p] = hv
            p_lo[p] = lv
            p_arg[p] = av
        # One-bin units: the parents' shared background (children are compared with the
        # kept paths of other parents without being materialised).
        one_bin = bins == 1
        cache_bg = parents <= store_limit
        p_bg = np.empty((parents if cache_bg else 1, samples, states))
        if cache_bg:
            for p in range(parents):
                if one_bin:
                    _one_bin_background(front_n[p], p_max[p], emis, col, first, second, bg,
                                        penalty, p_bg[p])
                else:
                    _multi_bin_background(front_n[p], p_max[p], prefix, bg, penalty, p_bg[p])
        scratch_bg = np.empty((samples, states))
        bg_parent = -1
        # Lower-bound probes of the cross-parent tests: per parent and sample the background
        # state (and its index) where the parent's shared chain P_{t-1} is largest.
        p_probe = np.empty((parents, samples, 2), np.int64)
        for p in range(parents):
            for s in range(samples):
                p_probe[p, s, 0] = bg[p_arg[p, s]] if len(bg) > 0 else fz[0]
                p_probe[p, s, 1] = p_arg[p, s]
        par = np.empty(children, np.int64)
        cho = np.empty(children, np.int64)
        totals = np.empty(children)
        bounds = np.empty(children)
        inc = np.empty(children, np.int64)
        # Same memory budget as the reference's stored child messages (store_limit of them).
        store = children * (founders + bins + 1) <= store_limit * states
        c_m = np.empty((children if store else 1, samples))
        c_f = np.empty((children if store else 1, samples, founders))
        c_s = np.empty((children if store else 1, samples, bins))
        for i in range(children):
            p = i // rows
            r = cands[begin + i % rows]
            par[i] = p
            cho[i] = r
            inc[i] = 1 if r == incumbent[u] else 0
            slot = i if store else 0
            totals[i], bounds[i] = _child_lazy(
                front_n[p], front_c[p], p_max[p], p_pb[p], p_tp[p], seg, te, ubmax, unext, emis,
                col, focal, r, first, second, penalty, fz, c_m[slot], c_f[slot], c_s[slot])
        alive = 0
        for i in range(children):
            if bounds[i] >= lower - margin:
                alive += 1
        if alive == 0:
            return INFEASIBLE, -np.inf, path, max_frontier
        order = _stable_order(totals, inc, par, cho)
        cap = min(alive, 64)
        kept_n = np.empty((cap, samples, states))
        kept_c = np.empty((cap, samples))
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
                cur_m[:] = c_m[i]
                cur_f[:] = c_f[i]
                cur_s[:] = c_s[i]
            else:
                _child_lazy(front_n[p], front_c[p], p_max[p], p_pb[p], p_tp[p], seg, te, ubmax,
                            unext, emis, col, focal, cho[i], first, second, penalty, fz,
                            cur_m, cur_f, cur_s)
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
                    if len(bg) > 0 and _cross_lower_bound(
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
                    if len(bg) > 0 and _cross_lower_bound(
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
            if not materialised:
                mat_t = _extend(front_n[p], front_c[p], emis, col, focal, cho[i], first, second,
                                penalty, mat_n, mat_c)
            if count == kept_n.shape[0]:
                grow = 2 * count
                new_n = np.empty((grow, samples, states))
                new_c = np.empty((grow, samples))
                new_t = np.empty(grow)
                new_i = np.empty(grow, np.int64)
                new_lt = np.empty(grow)
                new_m = np.empty((grow, samples))
                new_f = np.empty((grow, samples, founders))
                new_s = np.empty((grow, samples, bins))
                new_n[:count] = kept_n[:count]
                new_c[:count] = kept_c[:count]
                new_t[:count] = kept_t[:count]
                new_i[:count] = kept_i[:count]
                new_lt[:count] = kept_lt[:count]
                new_m[:count] = kept_m[:count]
                new_f[:count] = kept_f[:count]
                new_s[:count] = kept_s[:count]
                kept_n, kept_c, kept_t, kept_i = new_n, new_c, new_t, new_i
                kept_lt, kept_m, kept_f, kept_s = new_lt, new_m, new_f, new_s
            kept_n[count] = mat_n
            kept_c[count] = mat_c
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
        front_n = np.empty((survivors, samples, states))
        front_c = np.empty((survivors, samples))
        hp = np.empty(survivors, np.int64)
        hc = np.empty(survivors, np.int64)
        w = 0
        for x in range(count):
            if remove[x]:
                continue
            front_n[w] = kept_n[x]
            front_c[w] = kept_c[x]
            hp[w] = par[kept_i[x]]
            hc[w] = cho[kept_i[x]]
            w += 1
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
    for x in range(front_n.shape[0]):
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
def _closed_start_nb(start, penalty):
    """_closed_start in one nogil call (same element operations)."""
    samples, states = start.shape
    out = np.empty((samples, states))
    top = np.empty(samples)
    for s in range(samples):
        m = start[s, 0]
        for z in range(1, states):
            if start[s, z] > m:
                m = start[s, z]
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
        for p in range(parents):
            pm, pbv, tpv, _, _, _ = _parent_tables(beam_n[p], emis, bg, prefix, unext, penalty)
            for r in range(rows):
                totals[p * rows + r], _ = _child_lazy(
                    beam_n[p], beam_c[p], pm, pbv, tpv, seg, te, ubmax, unext, emis, col, focal,
                    cands[begin + r], first, second, penalty, fz, lazy_m, lazy_f, lazy_s)
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
                upper[u, s, fz[c]] = -np.inf
            for k in range(bins):
                best_m[k] = -np.inf
            lam_max = -np.inf
            for g in range(offsets[u], offsets[u + 1]):
                row = cands[g]
                shift = lam[s, g]
                if shift > lam_max:
                    lam_max = shift
                for c in range(founders):
                    yf[c] = upper[u + 1, s, fz[c]]
                for k in range(bins - 1, -1, -1):
                    m = ue[k]
                    for l in range(k + 1, bins):
                        v = mrow[l] - penalty + seg[s, k, l - 1]
                        if v > m:
                            m = v
                    for c in range(founders):
                        z = fz[c]
                        ra = row if first[z] == focal else col[first[z]]
                        rb = row if second[z] == focal else col[second[z]]
                        yf[c] += emis[s, ra, rb, k]
                        if yf[c] > m:
                            m = yf[c]
                    mrow[k] = m
                    sw = m - penalty
                    for c in range(founders):
                        if sw > yf[c]:
                            yf[c] = sw
                for c in range(founders):
                    v = yf[c] + shift
                    if v > upper[u, s, fz[c]]:
                        upper[u, s, fz[c]] = v
                for k in range(bins):
                    v = mrow[k] + shift
                    if v > best_m[k]:
                        best_m[k] = v
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
    programme (_child_lazy); the background states of the advanced message from the
    parent's shared chain and the rows' switch floors.
    """
    units = len(emissions)
    samples, states = start_n.shape
    founders = len(fz)
    nb = len(bg)
    an = start_n.copy()
    ac = start_c.copy()
    new = np.empty(states)
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
        pm, pbv, tpv, _, _, _ = _parent_tables(an, emis, bg, prefix, unext, penalty)
        rm = np.empty((rows, samples))
        rf = np.empty((rows, samples, founders))
        rs = np.empty((rows, samples, bins))
        theta = np.empty((rows, samples))
        for g in range(rows):
            _child_lazy(an, ac, pm, pbv, tpv, seg, te, ubmax, unext, emis, col, focal,
                        cands[begin + g], first, second, penalty, fz, rm[g], rf[g], rs[g])
            for s in range(samples):
                # The child's per-sample join with U[u + 1] (_child_lazy's bound terms).
                m = rm[g, s]
                lo = m - penalty
                best = tpv[s] - m
                v = lo - m + ubmax[s]
                if v > best:
                    best = v
                for l in range(1, bins):
                    v = rs[g, s, l] - m + te[s, l]
                    if v > best:
                        best = v
                for c in range(founders):
                    v = rf[g, s, c] + unext[s, fz[c]]
                    if v > best:
                        best = v
                theta[g, s] = ac[s] + m + best
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
            sw = pm[s] - penalty
            for b in range(nb):
                z = bg[b]
                x = an[s, z]
                if sw > x:
                    x = sw
                if bins == 1:
                    v = x + emis[s, col[first[z]], col[second[z]], 0] + lam_max
                else:
                    v = x + prefix[s, b, bins - 1] + lam_max
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
            mx = new[0]
            for z in range(1, states):
                if new[z] > mx:
                    mx = new[z]
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
    while sweeps < DUAL_SWEEPS and bound > best + tol:
        _dual_sweep(emissions, cols, cands, offsets, focal, incumbent, first, second, penalty,
                    start_n, start_c, upper, lam, fz, bg, prefixes, suffixes, segs, decoded)
        sweeps += 1
        value = path_value(emissions, cols, focal, decoded, first, second, penalty, start_n, start_c,
                           end)
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
        pm, pbv, tpv, _, _, _ = _parent_tables(prefix_n, emis, bg, prefix, unext, penalty)
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
            _, bound = _child_lazy(prefix_n, prefix_c, pm, pbv, tpv, seg, te, ubmax, unext, emis,
                                   cols[u], focal, r, first, second, penalty, fz, lazy_m, lazy_f,
                                   lazy_s)
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
def solve_focal(emissions, cols, cands, offsets, focal, incumbent, first, second, penalty, start,
                end, backstop, store_limit):
    """solve() without the Lagrangian-dual iterations, in one nogil call (same answer).

    Returns (status, rows, box_certified, dp, frontier, fixing_dps); status 0 = the
    incumbent is the canonical answer, 1 = rows is a different canonical answer,
    2 = a frontier exceeded the backstop, 3 = the programme lost every feasible path.
    The canonical-tie order is solve()'s: incumbent row first, then ascending index.
    """
    units = len(emissions)
    rows = np.empty(units, np.int64)
    start_n, start_c = _closed_start_nb(start, penalty)
    fz, bg = _focal_layout(focal, first, second)
    prefixes, suffixes, segs = _problem_tables(emissions, cols, first, second, bg)
    upper = window_box(emissions, cols, cands, offsets, focal, first, second, penalty, end, fz, bg,
                       prefixes, suffixes, segs)
    incumbent_value = path_value(emissions, cols, focal, incumbent, first, second, penalty,
                                 start_n, start_c, end)
    tol = 1e-7 + 1e-9 * max(1.0, abs(incumbent_value))
    certificate = (incumbent_value + tol
                   - SCREEN_ARITHMETIC_MARGIN * max(1.0, abs(incumbent_value)))
    if _join(start_n, start_c, upper[0]) <= certificate:
        return 0, rows, True, False, 0, 0
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
        return 2, rows, False, True, frontier, 0
    if status != OK:
        return 3, rows, False, True, frontier, 0
    if incumbent_value >= value - tol:
        return 0, rows, False, True, frontier, 0
    # Canonical ties: the lexicographically first path within tol of the optimum.
    target = value - tol
    status, best, fixing, reached = _canonical_restricted(
        emissions, cols, cands, offsets, focal, incumbent, first, second, penalty, start_n,
        start_c, end, upper, path, target, backstop, store_limit, fz, bg, prefixes, suffixes,
        segs, slack, tightened, value, tol)
    if reached > frontier:
        frontier = reached
    if status == BACKSTOP:
        return 2, rows, False, True, frontier, fixing
    changed = False
    for u in range(units):
        if best[u] != incumbent[u]:
            changed = True
    rows[:] = best
    return (1 if changed else 0), rows, False, True, frontier, fixing


@njit(cache=True, nogil=True)
def solve_template(emissions, cols, sizes, focals, penalty, start, end, backstop, store_limit):
    """Every listed focal founder's problem of one window (shared emissions and flanks).

    cols[u, k] is founder k's row on unit u (the focal's incumbent); the alphabet of
    unit u is 0..sizes[u]-1. Returns per focal (status, rows, box_certified, dp,
    frontier, fixing_dps) as arrays.
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
    for i in range(n):
        k = focals[i]
        incumbent = np.ascontiguousarray(cols[:, k])
        st, r, bx, used, fr, fx = solve_focal(emissions, cols, cands, offsets, k, incumbent, first,
                                             second, penalty, start, end, backstop, store_limit)
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


def _closed_start(start, penalty):
    """Closed normalised start message; max 0, entries >= -penalty."""
    start = np.asarray(start, np.float64)
    top = start.max(axis=1)
    return (np.ascontiguousarray(np.maximum(start - top[:, None], -penalty)),
            np.ascontiguousarray(top.astype(np.float64)))


class WindowProblem:
    """One focal founder on one window; units are blocks (fine) or L1 spans (coarse)."""

    def __init__(self, emissions, cols, alphabets_sizes, focal, incumbent, start, end, penalty):
        self.emissions = emissions if isinstance(emissions, List) else List(emissions)
        self.cols = np.ascontiguousarray(cols, np.int64)
        sizes = np.asarray(alphabets_sizes, np.int64)
        self.offsets = np.asarray([0, *np.cumsum(sizes)], np.int64)
        self.cands = np.concatenate([np.arange(n, dtype=np.int64) for n in sizes])
        self.focal = int(focal)
        self.incumbent = np.ascontiguousarray(incumbent, np.int64)
        self.start = np.ascontiguousarray(start, np.float64)
        self.end = np.ascontiguousarray(end, np.float64)
        self.penalty = float(penalty)
        founders = self.cols.shape[1]
        first, second = np.triu_indices(founders)
        self.first = np.ascontiguousarray(first, np.int64)
        self.second = np.ascontiguousarray(second, np.int64)


def screen(problem, max_sweeps):
    """Production Lagrangian dual from zero multipliers on the virtual-flank form.

    Returns (certified, best feasible binned value found, its path or None, sweeps).
    The value scale is that of packing.score_rows on the virtual-flank model.
    """
    founders = problem.cols.shape[1]
    focal = problem.focal
    order = [f for f in range(founders) if f != focal] + [focal]
    index = np.empty((founders, founders), np.int64)
    for q, (a, b) in enumerate(zip(problem.first, problem.second)):
        index[a, b] = index[b, a] = q
    reindex = index[np.ix_(order, order)]
    samples = problem.start.shape[0]
    flank_in = problem.start[:, reindex].reshape(samples, founders, founders, 1)
    flank_out = problem.end[:, reindex].reshape(samples, founders, founders, 1)
    models = PreparedModels(
        [dict(bin_emissions=np.ascontiguousarray(flank_in))]
        + [dict(bin_emissions=e) for e in problem.emissions]
        + [dict(bin_emissions=np.ascontiguousarray(flank_out))])
    units = len(problem.emissions)
    known = np.empty((founders - 1, units + 2), np.int64)
    known[:, 0] = np.arange(founders - 1)
    known[:, -1] = np.arange(founders - 1)
    known[:, 1:-1] = problem.cols[:, order[:-1]].T
    incumbent = np.concatenate([[founders - 1], problem.incumbent, [founders - 1]]).astype(np.int64)
    choices = np.concatenate([[founders - 1], problem.cands, [founders - 1]]).astype(np.int64)
    sizes = np.diff(problem.offsets)
    offsets = np.asarray([0, 1, *(1 + np.cumsum(sizes)), 2 + sizes.sum()], np.int64)
    penalty = problem.penalty
    first, second = problem.first, problem.second
    emissions = emission_arrays(models)
    short = short_models(models)
    packed = packed_emissions(models) if short else None
    coordinate_packed = dsh.pack_coordinate(models, packed) if short else None

    def score(path):
        return float(score_rows(models, np.vstack((known, path)), penalty))

    threshold = score(incumbent)
    # Certify only with a margin covering the different summation order of the exact
    # programme, so that a certified window is exactly one the programme would also
    # leave unchanged: the screen decides the work spent, never the proposal.
    certificate = (threshold + canonical_tolerance(threshold)
                   - SCREEN_ARITHMETIC_MARGIN * max(1.0, abs(threshold)))
    best_score, best_path = threshold, None
    messages = np.zeros((samples, len(choices)))
    bound = np.inf
    history = []
    sweeps = 0
    for sweep in range(max_sweeps):
        backwards = bool(sweep % 2)
        if short:
            suffix, before = dsh.dual_suffix(*packed, known, choices, offsets, messages, penalty,
                                             backwards, first, second)
        else:
            suffix, before = ds.suffix_bound(emissions, known, choices, offsets, messages, penalty,
                                             backwards, first, second, None, None, 0)
        bound = min(bound, float(before - ds.residual_correction(messages.sum(axis=0), offsets)))
        sweeps += 1
        if bound <= certificate:
            return True, best_score, best_path, sweeps
        if short:
            path, after, _ = dsh.coordinate(*coordinate_packed, known, incumbent, choices, offsets,
                                            messages, suffix, penalty, backwards, first, second)
        else:
            path, after, _ = ds.coordinate_sweep(emissions, known, incumbent, choices, offsets,
                                                 messages, suffix, penalty, backwards, first,
                                                 second, None, None, 0)
        value = score(path)
        if value > best_score:
            best_score, best_path = value, path[1:-1].copy()
        bound = min(bound, float(after - ds.residual_correction(messages.sum(axis=0), offsets)))
        history.append(bound)
        if bound <= certificate:
            return True, best_score, best_path, sweeps
        if bound <= best_score + canonical_tolerance(best_score):
            break
        if (len(history) > SCREEN_STALL_WINDOW
                and history[-SCREEN_STALL_WINDOW - 1] - bound < SCREEN_STALL):
            break
    return False, best_score, best_path, sweeps


def solve(problem, *, backstop, screen_sweeps=200, use_screen=True):
    """Canonical exact maximiser of H for one window problem.

    The screen runs before the programme: first the box bound (the dual at zero
    multipliers: every sample choosing f's rows freely), then, if that does not
    certify and use_screen is set, the Lagrangian dual (screen). A window whose bound
    lies within tol of the incumbent, less a float margin covering the arithmetics, has
    the incumbent as its canonical answer, so no programme runs; the screens decide only
    the work spent. Otherwise the shared-background exact programme (window_dp) and the
    canonical-tie programmes. Same answers as solve_reference.
    Returns (rows, diagnostic); rows is None when the incumbent is the canonical answer.
    """
    p = problem
    start_n, start_c = _closed_start(p.start, p.penalty)
    store_limit = max(1, STORE_BYTES // max(1, 8 * start_n.size))
    diagnostic = dict(screened=True, certified=False, box_certified=False, dp=False, frontier=0,
                      screen_sweeps=0, fixing_dps=0)
    fz, bg = _focal_layout(p.focal, p.first, p.second)
    prefixes, suffixes, segs = _problem_tables(p.emissions, p.cols, p.first, p.second, bg)
    upper = window_box(p.emissions, p.cols, p.cands, p.offsets, p.focal, p.first, p.second,
                       p.penalty, p.end, fz, bg, prefixes, suffixes, segs)
    incumbent_value = path_value(p.emissions, p.cols, p.focal, p.incumbent, p.first, p.second,
                                 p.penalty, start_n, start_c, p.end)
    tol = canonical_tolerance(incumbent_value)
    certificate = (incumbent_value + tol
                   - SCREEN_ARITHMETIC_MARGIN * max(1.0, abs(incumbent_value)))
    if _join(start_n, start_c, upper[0]) <= certificate:
        diagnostic.update(certified=True, box_certified=True)
        return None, diagnostic
    hint = -np.inf
    if use_screen:
        certified, best, _, sweeps = screen(p, screen_sweeps)
        diagnostic.update(certified=bool(certified), screen_sweeps=int(sweeps))
        if certified:
            return None, diagnostic
        hint = best
    lower = incumbent_value
    if np.isfinite(hint):
        # A feasible value of the screen, on the virtual-flank arithmetic.
        lower = max(lower, hint - 1e-9 * max(1.0, abs(hint)))
    # Bound pruning keeps every path within tol of a feasible value, including the
    # dominance tolerance a surviving lineage may have accumulated over the units.
    units = len(p.emissions)
    margin = tol + (units + 1) * DOMINANCE_TOLERANCE + 1e-9 * max(1.0, abs(lower))
    status, value, path, frontier = window_dp(
        p.emissions, p.cols, p.cands, p.offsets, p.focal, p.incumbent, p.first, p.second,
        p.penalty, start_n, start_c, 0, p.end, upper, lower, margin, DOMINANCE_TOLERANCE,
        min(int(backstop), EXPLORE_FRONTIER), int(store_limit), fz, bg, prefixes, suffixes, segs)
    slack = 0.0
    tightened = False
    if status == BACKSTOP and int(backstop) > EXPLORE_FRONTIER:
        # A large frontier: restart with the Lagrangian suffix bounds and the best feasible
        # value of a beam and the dual's decoded paths.
        tightened = True
        upper, lower, margin, slack = _tighten(
            p.emissions, p.cols, p.cands, p.offsets, p.focal, p.incumbent, p.first, p.second,
            p.penalty, start_n, start_c, p.end, upper, lower, tol, fz, bg, prefixes, suffixes, segs)
        status, value, path, reached = window_dp(
            p.emissions, p.cols, p.cands, p.offsets, p.focal, p.incumbent, p.first, p.second,
            p.penalty, start_n, start_c, 0, p.end, upper, lower, margin, DOMINANCE_TOLERANCE,
            int(backstop), int(store_limit), fz, bg, prefixes, suffixes, segs)
        frontier = max(frontier, reached)
    diagnostic.update(dp=True, frontier=int(frontier))
    if status == BACKSTOP:
        raise RuntimeError(f"window dynamic programme frontier exceeded its backstop {backstop}")
    if status != OK:
        raise RuntimeError("window dynamic programme lost every feasible path")
    if incumbent_value >= value - tol:
        return None, dict(diagnostic, value=float(value), incumbent_value=float(incumbent_value))
    # Canonical ties: the lexicographically first path within tol of the optimum.
    target = value - tol
    status, best, fixing, frontier = _canonical_restricted(
        p.emissions, p.cols, p.cands, p.offsets, p.focal, p.incumbent, p.first, p.second,
        p.penalty, start_n, start_c, p.end, upper, path, target, int(backstop), int(store_limit),
        fz, bg, prefixes, suffixes, segs, float(slack), bool(tightened), float(value), float(tol))
    diagnostic["fixing_dps"] = int(fixing)
    diagnostic["frontier"] = max(diagnostic["frontier"], int(frontier))
    if status == BACKSTOP:
        raise RuntimeError(
            f"window canonical-tie programme frontier exceeded its backstop {backstop}")
    found = path_value(p.emissions, p.cols, p.focal, best, p.first, p.second, p.penalty, start_n,
                       start_c, p.end)
    diagnostic.update(value=float(found), incumbent_value=float(incumbent_value))
    if np.array_equal(best, p.incumbent):
        return None, diagnostic
    return best, diagnostic


def solve_reference(problem, *, backstop, screen_sweeps=200, use_screen=True):
    """Validation reference of solve (the all-state kernels). Canonical exact maximiser of H.

    Returns (rows, diagnostic); rows is None when the incumbent is the canonical answer.
    """
    p = problem
    start_n, start_c = _closed_start(p.start, p.penalty)
    store_limit = max(1, STORE_BYTES // max(1, 8 * start_n.size))
    diagnostic = dict(screened=False, certified=False, dp=False, frontier=0, screen_sweeps=0,
                      fixing_dps=0)
    hint = -np.inf
    if use_screen:
        certified, best, _, sweeps = screen(p, screen_sweeps)
        diagnostic.update(screened=True, certified=bool(certified), screen_sweeps=int(sweeps))
        if certified:
            return None, diagnostic
        hint = best
    upper = window_box_reference(p.emissions, p.cols, p.cands, p.offsets, p.focal, p.first, p.second,
                       p.penalty, p.end)
    incumbent_value = path_value(p.emissions, p.cols, p.focal, p.incumbent, p.first, p.second,
                                 p.penalty, start_n, start_c, p.end)
    tol = canonical_tolerance(incumbent_value)
    lower = incumbent_value
    if np.isfinite(hint):
        # A feasible value of the screen, on the virtual-flank arithmetic.
        lower = max(lower, hint - 1e-9 * max(1.0, abs(hint)))
    # Bound pruning keeps every path within tol of a feasible value, including the
    # dominance tolerance a surviving lineage may have accumulated over the units.
    units = len(p.emissions)
    margin = tol + (units + 1) * DOMINANCE_TOLERANCE + 1e-9 * max(1.0, abs(lower))
    status, value, path, frontier = window_dp_reference(
        p.emissions, p.cols, p.cands, p.offsets, p.focal, p.incumbent, p.first, p.second,
        p.penalty, start_n, start_c, 0, p.end, upper, lower, margin, DOMINANCE_TOLERANCE,
        int(backstop), int(store_limit))
    diagnostic.update(dp=True, frontier=int(frontier))
    if status == BACKSTOP:
        raise RuntimeError(f"window dynamic programme frontier exceeded its backstop {backstop}")
    if status != OK:
        raise RuntimeError("window dynamic programme lost every feasible path")
    if incumbent_value >= value - tol:
        return None, dict(diagnostic, value=float(value), incumbent_value=float(incumbent_value))
    target = value - tol
    best = path.copy()
    prefix_n, prefix_c = start_n.copy(), start_c.copy()
    out_n = np.empty_like(prefix_n)
    out_c = np.empty_like(prefix_c)
    for u in range(units):
        rows = sorted(int(r) for r in p.cands[p.offsets[u]:p.offsets[u + 1]])
        ordered = [int(p.incumbent[u])] + [r for r in rows if r != int(p.incumbent[u])]
        for r in ordered:
            if r == best[u]:
                break
            _extend(prefix_n, prefix_c, p.emissions[u], p.cols[u], p.focal, r, p.first,
                    p.second, p.penalty, out_n, out_c)
            diagnostic["fixing_dps"] += 1
            status, found, tail, frontier = window_dp_reference(
                p.emissions, p.cols, p.cands, p.offsets, p.focal, p.incumbent, p.first,
                p.second, p.penalty, out_n.copy(), out_c.copy(), u + 1, p.end, upper, target,
                1e-9 * max(1.0, abs(target)), 0.0, int(backstop), int(store_limit))
            if status == BACKSTOP:
                raise RuntimeError(
                    f"window canonical-tie programme frontier exceeded its backstop {backstop}")
            diagnostic["frontier"] = max(diagnostic["frontier"], int(frontier))
            if status == OK and found >= target:
                best[u] = r
                best[u + 1:] = tail
                break
        _extend(prefix_n, prefix_c, p.emissions[u], p.cols[u], p.focal, int(best[u]), p.first,
                p.second, p.penalty, out_n, out_c)
        prefix_n[:] = out_n
        prefix_c[:] = out_c
    final = _join(prefix_n, prefix_c, p.end)
    diagnostic.update(value=float(final), incumbent_value=float(incumbent_value))
    if np.array_equal(best, p.incumbent):
        return None, diagnostic
    return best, diagnostic
