"""Upper bounds of the configuration chains with the child's read state (LR phase 2, Tier 2): the M2 pair bound
(pair_bound) carries the read class. (The M1 screen's certificate with the read state is the exact M1:
selective_screen.EXACT_CERTIFICATE.)

Every site factor of the chains is nonnegative and multilinear in the slots' transmitted ALT probabilities:
E_c(a, b) = G(a, b) R_c(a, b) = 3 sum_g W_g(a, b) L'_g R_c(g) (child_reads), bilinear in (a, b), and its tempered
power E_c ** eta is monotone in each. Where a bound relaxes a slot's allele to an interval (the pair bound:
BACKGROUND's in the component's range), every class entry is
largest at a corner of the box, so per site

    rbar_c = max_corner E_c / max_corner G      (full strength),
    rbar_c = max_corner E_c ** eta / (max_corner G) ** eta      (tempered),

relative to the scalar bound's own site factor max_corner G (certificates.global_common_logs, pair_bound
.emission_bound). The bin's class transfer bound Mbar(x, y) = prod_k T_k diag(rbar_k(x, y)) over the child's observed
markers (T_k the nonnegative class operators) is entrywise >= the chain's class transfer for every allele the box
allows, and a chain whose class vectors move by Mbar and whose scalar emissions are bounded dominates the exact chain
class by class (all operations are nonnegative and monotone). A corner the chain reads from the float32 tempered table
(child_reads.resident_terms) counts with the larger of that value and its float64 power.

The bound keeps the pair bound's state aggregation (pair_bound: the aggregated forward carries the class; the
entrywise maximum of Mbar times the scalar bound over each group pair; the partner caps of the scalar route are not
applied to the class route).

Method: docs/methods/pedigree.md#child-read-state
"""

from __future__ import annotations

import math

import numpy as np
from numba import prange

from ...core.kernels import kernel
from .forward import site_likelihood
from .kernels.bounds import OVERLOADS
from .read_transfer import multiply_classes, read_ratio

__all__ = ["site_bound", "class_bound_matrices", "slot_intervals", "pair_bounds_classes"]


@kernel(overloads=OVERLOADS["read_bounds.site_bound"], cache=True, inline="always", nogil=True)
def site_bound(out, ratio, powered, slot, lo1, hi1, lo2, hi2, gl, coefficient, eta, tempered, corner):
    """out[C]: the read factor bound rbar_c of one site and view over the allele box [lo1, hi1] x [lo2, hi2] (module
    docstring), relative to max_corner G; ``corner`` [C] is a workspace."""
    classes = out.shape[0]
    if tempered and eta == 0.0:
        for c in range(classes):
            out[c] = 1.0
        return
    power = tempered and eta != 1.0
    for c in range(classes):
        out[c] = 0.0
    # Named founder states have point intervals. Evaluate only distinct
    # corners and use each scalar likelihood for both maxima.
    top = 0.0
    for i in range(1 + int(lo1 != hi1)):
        a = lo1 if i == 0 else hi1
        for j in range(1 + int(lo2 != hi2)):
            b = lo2 if j == 0 else hi2
            g = site_likelihood(gl, coefficient, a, b, 0.0)
            if g > top:
                top = g
            read_ratio(corner, ratio, slot, a, b, gl)
            if power:
                scale = g**eta if g > 0.0 else 0.0
                exact = (a == 0.0 or a == 1.0) and (b == 0.0 or b == 1.0)
                config = 2 * int(a) + int(b) if exact else 0
                for c in range(classes):
                    value = corner[c] ** eta
                    if exact:
                        value = max(value, float(powered[slot, config, c]))
                    value *= scale
                    if value > out[c]:
                        out[c] = value
            else:
                for c in range(classes):
                    value = g * corner[c]
                    if value > out[c]:
                        out[c] = value
    scale = top**eta if power and top > 0.0 else top
    for c in range(classes):
        out[c] = out[c] / scale if scale > 0.0 else 1.0


@kernel(overloads=OVERLOADS["read_bounds.class_bound_matrices"], cache=True, parallel=True, nogil=True)
def class_bound_matrices(children, first_bin, stop_bin, lo1, hi1, lo2, hi2, gl, coefficient, site_index, site_start,
                         site_stop, exponent, read_offset, read_reset, read_operator, read_ratio_table, read_powered):
    """float64 [n, nb, 2, S, S, C, C]: per listed child, bin of [first_bin, stop_bin), view (0 tempered, 1 full) and
    state pair (x, y), the bin's class transfer bound prod_k T_k diag(rbar_k(x, y)) over the child's observed markers
    (the identity without any, or for a child without read state). lo1 / hi1 float64 [S, M] slot 1's allele interval
    of each state at each marker, lo2 / hi2 slot 2's. States whose boxes agree at every observed marker of the bin
    (the same box word, per slot) have the same bounds: each pair of words is computed once and copied."""
    n = len(children)
    bins = stop_bin - first_bin
    states = lo1.shape[0]
    classes = read_operator.shape[1]
    out = np.zeros((n, bins, 2, states, states, classes, classes))
    for task in prange(n * bins):
        ci = task // bins
        local = task - ci * bins
        child = children[ci]
        block = first_bin + local
        for view in range(2):
            for x in range(states):
                for y in range(states):
                    for c in range(classes):
                        out[ci, local, view, x, y, c, c] = 1.0
        start = read_offset[child]
        if read_offset[child + 1] == start:
            continue
        base = start - site_start[child, 0]
        first_slot, stop_slot = site_start[child, block], site_stop[child, block]
        # word1[x] / word2[y]: the first state with the same slot-1 / slot-2 box at every observed marker of the bin
        word1 = np.empty(states, dtype=np.int64)
        word2 = np.empty(states, dtype=np.int64)
        for x in range(states):
            word1[x] = x
            word2[x] = x
            for z in range(x):
                if word1[x] == x and word1[z] == z:
                    same = True
                    for slot in range(first_slot, stop_slot):
                        site = site_index[slot]
                        if lo1[x, site] != lo1[z, site] or hi1[x, site] != hi1[z, site]:
                            same = False
                            break
                    if same:
                        word1[x] = z
                if word2[x] == x and word2[z] == z:
                    same = True
                    for slot in range(first_slot, stop_slot):
                        site = site_index[slot]
                        if lo2[x, site] != lo2[z, site] or hi2[x, site] != hi2[z, site]:
                            same = False
                            break
                    if same:
                        word2[x] = z
        factor = np.empty(classes)
        corner = np.empty(classes)
        work = np.empty((classes, classes))
        eta = exponent[child, block]
        for x in range(states):
            if word1[x] != x:
                continue
            for y in range(states):
                if word2[y] != y:
                    continue
                for view in range(2):
                    matrix = out[ci, local, view, x, y]
                    for slot in range(site_start[child, block], site_stop[child, block]):
                        k = base + slot
                        site = site_index[slot]
                        site_bound(factor, read_ratio_table, read_powered, k, lo1[x, site], hi1[x, site], lo2[y, site],
                                   hi2[y, site], gl[child, site], coefficient[child, site], eta, view == 0,
                                   corner)
                        multiply_classes(matrix, read_operator[k], factor, work, classes, read_reset[k])
        for view in range(2):
            for x in range(states):
                for y in range(states):
                    if word1[x] != x or word2[y] != y:
                        out[ci, local, view, x, y] = out[ci, local, view, word1[x], word2[y]]
    return out


def slot_intervals(state_alt: np.ndarray, relaxed: np.ndarray | None = None, low: np.ndarray | None = None,
                   high: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) float64 [S, M] allele intervals of one slot: exact state_alt, except states/markers where ``relaxed``
    (bool [S, M]) holds, which take [low, high] (per marker; [0, 1] when not given)."""
    lo = np.array(state_alt, dtype=np.float64, copy=True)
    hi = lo.copy()
    if relaxed is not None:
        lower = np.zeros(lo.shape[1]) if low is None else np.asarray(low, dtype=np.float64)
        upper = np.ones(lo.shape[1]) if high is None else np.asarray(high, dtype=np.float64)
        lo = np.where(relaxed, lower[None, :], lo)
        hi = np.where(relaxed, upper[None, :], hi)
    return np.ascontiguousarray(lo), np.ascontiguousarray(hi)


@kernel(overloads=OVERLOADS["read_bounds._group_step"], cache=True, nogil=True)
def _group_step(alpha, work, n_from_p, n_from_q, n_to_p, n_to_q, tp, tq):
    """pair_bound._step on one class slice: alpha[to_p, to_q] = sum alpha[from_p, from_q] tp[from_p, to_p]
    tq[from_q, to_q] (axis by axis)."""
    for i in range(n_from_p):
        for j in range(n_to_q):
            value = 0.0
            for k in range(n_from_q):
                value += alpha[i, k] * tq[k, j]
            work[i, j] = value
    for i in range(n_to_p):
        for j in range(n_to_q):
            value = 0.0
            for k in range(n_from_p):
                value += work[k, j] * tp[k, i]
            alpha[i, j] = value


@kernel(overloads=OVERLOADS["read_bounds._absorb_classes"], cache=True, nogil=True)
def _absorb_classes(alpha, moved, n_p, n_q, group_p, group_q, scaled, log_emission, matrices, local, block, view,
                    maximum, eta, group_matrix, log_group, states):
    """One view's aggregated class forward ``alpha`` [C, G, G] times its group emission bounds; returns the log
    normaliser (-inf when no mass remains). The probability path takes, per group pair, the entrywise maximum over
    its state pairs of scaled(x, y) Mbar(x, y); when its total underflows, the log path bounds each group pair by
    exp(max log scalar bound) times the entrywise maximum of Mbar (separate maxima: a valid, looser bound)."""
    classes = alpha.shape[0]
    for attempt in range(2):
        for i in range(n_p):
            for j in range(n_q):
                group_matrix[i, j] = 0.0
                log_group[i, j] = -np.inf
        for x in range(states):
            gx = group_p[x]
            for y in range(states):
                gy = group_q[y]
                if attempt == 0:
                    weight = scaled[local, block, x, y]
                else:
                    weight = 1.0
                    v = 0.0 if eta == 0.0 else eta * (log_emission[local, block, x, y] - maximum)
                    if v > log_group[gx, gy]:
                        log_group[gx, gy] = v
                matrix = matrices[local, block, view, x, y]
                for c in range(classes):
                    for d in range(classes):
                        value = weight * matrix[c, d]
                        if value > group_matrix[gx, gy, c, d]:
                            group_matrix[gx, gy, c, d] = value
        shift = 0.0
        if attempt == 1:
            shift = -np.inf
            for i in range(n_p):
                for j in range(n_q):
                    mass = 0.0
                    for c in range(classes):
                        mass += alpha[c, i, j]
                    if mass > 0.0 and log_group[i, j] > shift:
                        shift = log_group[i, j]
            if not np.isfinite(shift):
                return -np.inf
        total = 0.0
        for i in range(n_p):
            for j in range(n_q):
                factor = 1.0 if attempt == 0 else (math.exp(log_group[i, j] - shift)
                                                   if np.isfinite(log_group[i, j]) else 0.0)
                for d in range(classes):
                    acc = 0.0
                    for c in range(classes):
                        acc += alpha[c, i, j] * group_matrix[i, j, c, d]
                    moved[d, i, j] = acc * factor
                    total += moved[d, i, j]
        if attempt == 0 and not (total > 1e-280 and np.isfinite(total)):
            continue
        if total <= 0.0 or not np.isfinite(total):
            return -np.inf
        for d in range(classes):
            for i in range(n_p):
                for j in range(n_q):
                    alpha[d, i, j] = moved[d, i, j] / total
        return shift + math.log(total)
    return -np.inf


@kernel(overloads=OVERLOADS["read_bounds.pair_bounds_classes"], cache=True, parallel=True, nogil=True)
def pair_bounds_classes(trios, row_child, log_emission, full_e, tempered_e, top, offset_full, offset_tempered,
                        exponent, group, count, initial, transfer, matrices, read_initial, read_width, run_start):
    """float64 [T]: pair_bound.pair_bounds with the read class carried (module docstring): the aggregated forward
    over (class, group of p, group of q), its group transfers applied to every class slice, each bin's group emission
    the entrywise maximum of the scaled scalar bound times the class transfer bound (``matrices`` [n, B, 2, S, S, C,
    C], class_bound_matrices); no partner caps. The rows come in runs of one (child, first parent) (``run_start``
    [R + 1] the runs' first rows, the canonical order): per run, bin and view the partial maxima over the first
    parent's groups, H[gp, y] = max_{x in gp} scaled(x, y) Mbar(x, y), are formed once, and each row's group emission
    is max_{y in q's group} H[gp, y] (the same maxima, regrouped). A bin whose probability path underflows is redone
    on the log path (_absorb_classes)."""
    states = log_emission.shape[2]
    bins = log_emission.shape[1]
    groups = initial.shape[1]
    classes = matrices.shape[5]
    out = np.empty(trios.shape[0])
    for run in prange(len(run_start) - 1):
        first_row = run_start[run]
        rows = run_start[run + 1] - first_row
        child, p = trios[first_row, 0], trios[first_row, 1]
        local = row_child[first_row]
        # the run's rows advance together bin by bin (each row's operations in the same order as alone), so the
        # first parent's partial maxima are needed for one bin at a time
        partial = np.zeros((2, groups, states, classes, classes))
        a_full = np.zeros((rows, classes, groups, groups))
        a_temp = np.zeros((rows, classes, groups, groups))
        log_full = np.zeros(rows)
        log_temp = np.zeros(rows)
        alive = np.ones(rows, dtype=np.bool_)
        moved = np.zeros((classes, groups, groups))
        work = np.empty((groups, groups))
        group_matrix = np.empty((groups, groups, classes, classes))
        log_group = np.empty((groups, groups))
        for r in range(rows):
            q = trios[first_row + r, 2]
            for c in range(classes):
                for i in range(count[p, 0]):
                    for j in range(count[q, 0]):
                        a_full[r, c, i, j] = initial[p, i] * initial[q, j] * read_initial[child, c]
                        a_temp[r, c, i, j] = a_full[r, c, i, j]
        for block in range(bins):
            enter, leave = read_width[child, block], read_width[child, block + 1]
            partial[:] = 0.0
            for view in range(2):
                for x in range(states):
                    gx = group[p, block, x]
                    for y in range(states):
                        weight = full_e[local, block, x, y] if view == 1 else tempered_e[local, block, x, y]
                        matrix = matrices[local, block, view, x, y]
                        for c in range(enter):
                            for d in range(classes):
                                value = weight * matrix[c, d]
                                if value > partial[view, gx, y, c, d]:
                                    partial[view, gx, y, c, d] = value
            for r in range(rows):
                if not alive[r]:
                    continue
                q = trios[first_row + r, 2]
                n_p, n_q = count[p, block], count[q, block]
                if block > 0:
                    a_p, a_q = count[p, block - 1], count[q, block - 1]
                    for c in range(enter):
                        _group_step(a_full[r, c], work, a_p, a_q, n_p, n_q, transfer[p, block - 1],
                                    transfer[q, block - 1])
                        _group_step(a_temp[r, c], work, a_p, a_q, n_p, n_q, transfer[p, block - 1],
                                    transfer[q, block - 1])
                maximum = top[local, block]
                if not np.isfinite(maximum):
                    alive[r] = False
                    continue
                eta = exponent[child, block]
                for view in (1, 0):
                    alpha = a_full[r] if view == 1 else a_temp[r]
                    for i in range(n_p):
                        for j in range(n_q):
                            for c in range(enter):
                                for d in range(classes):
                                    group_matrix[i, j, c, d] = 0.0
                    for i in range(n_p):
                        for y in range(states):
                            gy = group[q, block, y]
                            for c in range(enter):
                                for d in range(classes):
                                    value = partial[view, i, y, c, d]
                                    if value > group_matrix[i, gy, c, d]:
                                        group_matrix[i, gy, c, d] = value
                    total = 0.0
                    for i in range(n_p):
                        for j in range(n_q):
                            for d in range(classes):
                                acc = 0.0
                                for c in range(enter):
                                    acc += alpha[c, i, j] * group_matrix[i, j, c, d]
                                moved[d, i, j] = acc
                                total += acc
                    if total > 1e-280 and np.isfinite(total):
                        for i in range(n_p):
                            for j in range(n_q):
                                if leave == 1:
                                    mass = 0.0
                                    for d in range(classes):
                                        mass += moved[d, i, j]
                                    alpha[0, i, j] = mass / total
                                else:
                                    for d in range(classes):
                                        alpha[d, i, j] = moved[d, i, j] / total
                        value = math.log(total)
                    else:
                        # The underflow fallback uses the full workspace. Expand the scalar as mass in class
                        # zero (the incoming operator has equal rows, or this is an empty bin), then marginalise.
                        if enter == 1:
                            alpha[1:] = 0.0
                        value = _absorb_classes(alpha, moved, n_p, n_q, group[p, block], group[q, block],
                                                full_e if view == 1 else tempered_e, log_emission, matrices, local,
                                                block, view, maximum, 1.0 if view == 1 else eta, group_matrix,
                                                log_group, states)
                        if leave == 1:
                            for i in range(n_p):
                                for j in range(n_q):
                                    for d in range(1, classes):
                                        alpha[0, i, j] += alpha[d, i, j]
                    if not np.isfinite(value):
                        alive[r] = False
                        break
                    if view == 1:
                        log_full[r] += value
                    else:
                        log_temp[r] += value
        for r in range(rows):
            offsets = 0.5 * (offset_full[local] + offset_tempered[local])
            out[first_row + r] = 0.5 * (log_full[r] + log_temp[r]) + offsets if alive[r] else -np.inf
    return out
