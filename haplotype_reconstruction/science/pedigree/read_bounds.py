"""Upper bounds of the configuration chains with the child's read state (LR phase 2, Tier 2): the M2 pair bound
(pair_bound) carries the read class. (The M1 screen's certificate with the read state is the exact M1:
selective_screen.EXACT_CERTIFICATE.)

Every site factor of the chains is nonnegative and multilinear in the slots' transmitted ALT probabilities:
E_c(a, b) = G(a, b) R_c(a, b) = 3 sum_g W_g(a, b) L'_g R_c(g) (child_reads), bilinear in (a, b). Where a bound
relaxes a slot's allele to an interval (the pair bound: BACKGROUND's in the component's range), every class entry is
largest at a corner of the box, so per site

    rbar_c = max_corner E_c / max_corner G,

relative to the scalar bound's own site factor max_corner G (certificates.global_common_logs, pair_bound
.emission_bound). The bin's class transfer bound Mbar(x, y) = prod_k T_k diag(rbar_k(x, y)) over the child's observed
markers (T_k the nonnegative class operators) is entrywise >= the chain's class transfer for every allele the box
allows, and a chain whose class vectors move by Mbar and whose scalar emissions are bounded dominates the exact chain
class by class (all operations are nonnegative and monotone).

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
from .scheduling import balanced_units

__all__ = ["site_bound", "class_bound_matrices", "slot_intervals", "pair_bounds_classes"]


@kernel(overloads=OVERLOADS["read_bounds.site_bound"], cache=True, inline="always", nogil=True)
def site_bound(out, ratio, slot, lo1, hi1, lo2, hi2, gl, coefficient, corner):
    """out[C]: the read factor bound rbar_c of one site over the allele box [lo1, hi1] x [lo2, hi2] (module docstring),
    relative to max_corner G; ``corner`` [C] is a workspace."""
    classes = out.shape[0]
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
            for c in range(classes):
                value = g * corner[c]
                if value > out[c]:
                    out[c] = value
    for c in range(classes):
        out[c] = out[c] / top if top > 0.0 else 1.0


@kernel(overloads=OVERLOADS["read_bounds.class_bound_matrices"], cache=True, parallel=True, nogil=True)
def class_bound_matrices(children, first_bin, stop_bin, lo1, hi1, lo2, hi2, gl, coefficient, site_index, site_start,
                         site_stop, read_offset, read_reset, read_operator, read_ratio_table):
    """float64 [n, nb, S, S, C, C]: per listed child, bin of [first_bin, stop_bin) and state pair (x, y), the bin's
    class transfer bound prod_k T_k diag(rbar_k(x, y)) over the child's observed markers (the identity without any, or
    for a child without read state). lo1 / hi1 float64 [S, M] slot 1's allele interval
    of each state at each marker, lo2 / hi2 slot 2's. States whose boxes agree at every observed marker of the bin
    (the same box word, per slot) have the same bounds: each pair of words is computed once and copied."""
    n = len(children)
    bins = stop_bin - first_bin
    states = lo1.shape[0]
    classes = read_operator.shape[1]
    out = np.zeros((n, bins, states, states, classes, classes))
    for task in prange(n * bins):
        ci = task // bins
        local = task - ci * bins
        child = children[ci]
        block = first_bin + local
        for x in range(states):
            for y in range(states):
                for c in range(classes):
                    out[ci, local, x, y, c, c] = 1.0
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
        for x in range(states):
            if word1[x] != x:
                continue
            for y in range(states):
                if word2[y] != y:
                    continue
                matrix = out[ci, local, x, y]
                for slot in range(site_start[child, block], site_stop[child, block]):
                    k = base + slot
                    site = site_index[slot]
                    site_bound(factor, read_ratio_table, k, lo1[x, site], hi1[x, site], lo2[y, site], hi2[y, site],
                               gl[child, site], coefficient[child, site], corner)
                    multiply_classes(matrix, read_operator[k], factor, work, classes, read_reset[k])
        for x in range(states):
            for y in range(states):
                if word1[x] != x or word2[y] != y:
                    out[ci, local, x, y] = out[ci, local, word1[x], word2[y]]
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
def _absorb_classes(alpha, moved, n_p, n_q, group_p, group_q, scaled, log_emission, matrices, local, block,
                    maximum, group_matrix, log_group, states):
    """The aggregated class forward ``alpha`` [C, G, G] times its group emission bounds; returns the log
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
                    v = log_emission[local, block, x, y] - maximum
                    if v > log_group[gx, gy]:
                        log_group[gx, gy] = v
                matrix = matrices[local, block, x, y]
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
def pair_bounds_classes(trios, row_child, log_emission, scaled, top, offset, group, count, initial, transfer, matrices,
                        read_initial, read_width, run_start, threads):
    """float64 [T]: pair_bound.pair_bounds with the read class carried (module docstring): the aggregated forward
    over (class, group of p, group of q), its group transfers applied to every class slice, each bin's group emission
    the entrywise maximum of the scaled scalar bound times the class transfer bound (``matrices`` [n, B, S, S, C, C],
    class_bound_matrices); no partner caps. The rows come in runs of one (child, first parent) (``run_start`` [R + 1]
    the runs' first rows, the canonical order): per run and bin the partial maxima over the first parent's groups,
    H[gp, y] = max_{x in gp} scaled(x, y) Mbar(x, y), are formed once, and each row's group emission is
    max_{y in q's group} H[gp, y] (the same maxima, regrouped). A bin whose probability path underflows is redone on
    the log path (_absorb_classes)."""
    states = log_emission.shape[2]
    bins = log_emission.shape[1]
    groups = initial.shape[1]
    classes = matrices.shape[4]
    out = np.empty(trios.shape[0])
    cost = np.empty(len(run_start) - 1, np.int64)
    child_width = read_width[:, :bins].sum(axis=1)
    for run in range(len(cost)):
        first_row = run_start[run]
        rows = run_start[run + 1] - first_row
        # Each piece pays once for the first-parent partial maxima, then
        # aggregates them and moves the group/class state for every pair.
        cost[run] = (1 + child_width[trios[first_row, 0]]) * (states + rows * groups)
    units, offsets = balanced_units(cost, threads)
    for tile in prange(len(offsets) - 1):
        for cell in range(offsets[tile], offsets[tile + 1]):
            run = units[cell]
            first_row = run_start[run]
            rows = run_start[run + 1] - first_row
            child, p = trios[first_row, 0], trios[first_row, 1]
            local = row_child[first_row]
            # the run's rows advance together bin by bin (each row's operations in the same order as alone), so the
            # first parent's partial maxima are needed for one bin at a time
            partial = np.zeros((groups, states, classes, classes))
            forward = np.zeros((rows, classes, groups, groups))
            log_bound = np.zeros(rows)
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
                            forward[r, c, i, j] = initial[p, i] * initial[q, j] * read_initial[child, c]
            for block in range(bins):
                enter, leave = read_width[child, block], read_width[child, block + 1]
                partial[:] = 0.0
                for x in range(states):
                    gx = group[p, block, x]
                    for y in range(states):
                        weight = scaled[local, block, x, y]
                        matrix = matrices[local, block, x, y]
                        for c in range(enter):
                            for d in range(classes):
                                value = weight * matrix[c, d]
                                if value > partial[gx, y, c, d]:
                                    partial[gx, y, c, d] = value
                for r in range(rows):
                    if not alive[r]:
                        continue
                    q = trios[first_row + r, 2]
                    n_p, n_q = count[p, block], count[q, block]
                    alpha = forward[r]
                    if block > 0:
                        a_p, a_q = count[p, block - 1], count[q, block - 1]
                        for c in range(enter):
                            _group_step(alpha[c], work, a_p, a_q, n_p, n_q,
                                        transfer[p, block - 1], transfer[q, block - 1])
                    maximum = top[local, block]
                    if not np.isfinite(maximum):
                        alive[r] = False
                        continue
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
                                    value = partial[i, y, c, d]
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
                        value = _absorb_classes(alpha, moved, n_p, n_q, group[p, block], group[q, block], scaled,
                                                log_emission, matrices, local, block, maximum, group_matrix, log_group,
                                                states)
                        if leave == 1:
                            for i in range(n_p):
                                for j in range(n_q):
                                    for d in range(1, classes):
                                        alpha[0, i, j] += alpha[d, i, j]
                    if not np.isfinite(value):
                        alive[r] = False
                        continue
                    log_bound[r] += value
            for r in range(rows):
                out[first_row + r] = log_bound[r] + offset[local] if alive[r] else -np.inf
    return out
