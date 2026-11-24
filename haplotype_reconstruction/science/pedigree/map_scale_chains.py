"""The class-carried transmission chains of the painting-based map-scale estimate (rev 5, DESIGN_MAPSCALE; LR Tier 3).

Every chain of the estimate and of its relationship guard runs on frame sequences (map_scale_frames) whose emissions
carry the target's read state. Frame x holds a transfer matrix em[x, g] [C, C] per emitted state g and the misfit
matrix em[x, misfit] (map_scale_reads: the painting's tempered pair emission at the path strength times the pair's
read matrix, with the label-free class operator over the sites since the previous frame folded in), and entry[x] [C]
the class law at the start of the frame's gap, read where a sequence starts. A hidden state is (s, c): the chain's
transmission state s and the read class c of the target's homologues in the transmission frame (homologue 1 carries
parent 0's transmitted copy, homologue 2 parent 1's). Between frames the transmission state moves as before, on
every class alike (the transitions do not depend on the reads); the frame then moves the class:

    alpha_x(s', c') = sum_c [sum_s alpha_{x-1}(s, c) A_x(s, s')] [(1 - delta) em[x, g(s')] + delta em[x, misfit]][c, c']

and a sequence starts from its transmission prior times entry[x0]. Without read state (C = 1: unit matrices, entry 1)
every kernel is the scalar chain exactly. Inputs are C-contiguous writable arrays; outputs are freshly allocated.

Method: docs/methods/pedigree.md#map-scale.
"""

from __future__ import annotations

import heapq
import math

import numba
import numpy as np
from numba import prange, types

from ...core.kernels import kernel

__all__ = ["frame_loglik", "target_loglik", "target_posterior", "error_run_loglik"]


def _a(dtype, ndim):
    return types.Array(dtype, ndim, "C", readonly=False, aligned=True)


_i64_1 = _a(types.int64, 1)
_f1, _f2, _f4 = _a(types.float64, 1), _a(types.float64, 2), _a(types.float64, 4)
_MISFIT_5 = 4    # misfit matrix of a [X, 5, C, C] frame emission (four full-sib configurations, then misfit)
_MISFIT_25 = 25  # misfit matrix of a [X, 26, C, C] frame emission (25 track-symbol pairs, then misfit)


def _tiles(lengths: np.ndarray, per: int) -> tuple[np.ndarray, np.ndarray]:
    """(units, tile offsets) of a kernel whose units are (sequence s, j) for j < ``per``, numbered s * per + j, each
    costing the sequence's length (``lengths``, frames): one tile per Numba thread, of nearly equal summed cost.

    Numba's omp threading layer gives each thread one contiguous block of a parallel loop's iterations
    (numba.parallel_chunksize only shrinks the blocks), and the frame sequences arrive grouped by contig and family: on
    R4's tl roots the block holding the long sequences carried about five times the mean work (core use 0.2 at 112
    threads). A parallel loop over these tiles runs one tile per thread. A sequence's units stay together in one tile
    (its frames' emissions stay in the thread's cache across rates), split into equal pieces only while there are
    fewer than four pieces per thread; the pieces go longest first to the least loaded tile. Which tile holds a unit
    changes no unit's arithmetic: the kernels' outputs do not depend on the thread count or the tiling."""
    lengths = np.asarray(lengths, dtype=np.int64)
    threads = max(1, numba.get_num_threads())
    count = len(lengths)
    if count == 0 or per == 0:
        return np.zeros(0, np.int64), np.zeros(2, np.int64)
    pieces = min(per, max(1, -(-4 * threads // count)))   # pieces per sequence
    size = -(-per // pieces)                                # units per piece
    starts = np.arange(0, per, size, dtype=np.int64)
    sequence = np.repeat(np.arange(count, dtype=np.int64), len(starts))
    low = np.tile(starts, count)
    high = np.minimum(low + size, per)
    cost = lengths[sequence] * (high - low)
    tiles = min(threads, len(cost))
    heap = [(0, t) for t in range(tiles)]
    tile = np.empty(len(cost), np.int64)
    order = np.argsort(-cost, kind="stable")
    for k in order.tolist():
        load, t = heapq.heappop(heap)
        tile[k] = t
        heapq.heappush(heap, (load + int(cost[k]), t))
    placed = order[np.argsort(tile[order], kind="stable")]   # pieces by tile, longest first within a tile
    width = high[placed] - low[placed]
    first = np.repeat(sequence[placed] * per + low[placed], width)
    step = np.arange(int(width.sum()), dtype=np.int64) - np.repeat(np.cumsum(width) - width, width)
    units = np.ascontiguousarray(first + step, dtype=np.int64)
    offsets = np.zeros(tiles + 1, np.int64)
    offsets[1:] = np.cumsum(np.bincount(tile, weights=high - low, minlength=tiles).astype(np.int64))
    return units, offsets


@kernel(overloads=((_i64_1, _f1, _f4, _f2, _f1, _f1, _i64_1, _i64_1),), cache=True, parallel=True, nogil=True)
def _frame_loglik_tiles(offsets, pos, em, entry, r_grid, delta_grid, units, tile_offsets):
    """4-state chain of a sib's two transmissions (independent homologue bits flipping at 1 per Morgan) with the read
    class carried, on each frame sequence [S, R, D]: em[x, :4] the configurations' transfer matrices (states 00, 01,
    10, 11), em[x, 4] the misfit matrix mixed in at weight delta; entry[x] the class law where a sequence starts."""
    S = len(offsets) - 1
    R = len(r_grid)
    C = em.shape[2]
    out = np.zeros((S, R, len(delta_grid)))
    # one unit per (sequence, rate), the units dealt to balanced tiles (_tiles); a unit's arithmetic is the same
    # whatever the thread count or tiling
    for t in prange(len(tile_offsets) - 1):
        for k in range(tile_offsets[t], tile_offsets[t + 1]):
            cell = units[k]
            s = cell // R
            ri = cell % R
            lo = offsets[s]
            hi = offsets[s + 1]
            a = np.empty((4, C))
            w = np.empty((4, C))
            rate = r_grid[ri] * 1e-8
            for di in range(len(delta_grid)):
                dl = delta_grid[di]
                for k in range(4):
                    for c in range(C):
                        a[k, c] = 0.25 * entry[lo, c]
                ll = 0.0
                for x in range(lo, hi):
                    if x > lo:
                        q = 0.5 * (1.0 - math.exp(-2.0 * (pos[x] - pos[x - 1]) * rate))
                        st = 1.0 - q
                        for c in range(C):
                            b0 = st * a[0, c] + q * a[2, c]
                            b1 = st * a[1, c] + q * a[3, c]
                            b2 = q * a[0, c] + st * a[2, c]
                            b3 = q * a[1, c] + st * a[3, c]
                            a[0, c] = st * b0 + q * b1
                            a[1, c] = q * b0 + st * b1
                            a[2, c] = st * b2 + q * b3
                            a[3, c] = q * b2 + st * b3
                    tot = 0.0
                    for k in range(4):
                        for j in range(C):
                            v = 0.0
                            for i in range(C):
                                v += a[k, i] * ((1.0 - dl) * em[x, k, i, j] + dl * em[x, _MISFIT_5, i, j])
                            w[k, j] = v
                            tot += v
                    ll += math.log(tot)
                    for k in range(4):
                        for j in range(C):
                            a[k, j] = w[k, j] / tot
                out[s, ri, di] = ll
    return out


def frame_loglik(offsets, pos, em, entry, r_grid, delta_grid):
    """[S, R, D] log-likelihoods of _frame_loglik_tiles, its (sequence, rate) units in balanced tiles."""
    units, tile_offsets = _tiles(np.diff(offsets), len(r_grid))
    return _frame_loglik_tiles(offsets, pos, em, entry, r_grid, delta_grid, units, tile_offsets)


@kernel(overloads=((_f2, types.float64, types.float64, types.float64),), cache=True, nogil=True)
def _first_track_transition(T, q, ein, eout):
    """Track 1 (parent 0's classes 0, 1 are its full-sib symbols) transition matrix at switch probability q."""
    for u in range(5):
        fs_u = u < 2
        for v in range(5):
            fs_v = v < 2
            if fs_u:
                if fs_v:
                    T[u, v] = (1.0 - ein) * ((1.0 - q) if u == v else q)
                else:
                    T[u, v] = ein / 3.0
            else:
                if fs_v:
                    T[u, v] = eout / 2.0
                else:
                    T[u, v] = (1.0 - eout) * ((1.0 - q) if u == v else q / 2.0)


@kernel(overloads=((_f2, types.float64, types.float64, types.float64),), cache=True, nogil=True)
def _second_track_transition(T, q, ein, eout):
    """Track 2 (parent 1's classes 2, 3 are its full-sib symbols) transition matrix at switch probability q."""
    for u in range(5):
        fs_u = u == 2 or u == 3
        for v in range(5):
            fs_v = v == 2 or v == 3
            if fs_u:
                if fs_v:
                    T[u, v] = (1.0 - ein) * ((1.0 - q) if u == v else q)
                else:
                    T[u, v] = ein / 3.0
            else:
                if fs_v:
                    T[u, v] = eout / 2.0
                else:
                    T[u, v] = (1.0 - eout) * ((1.0 - q) if u == v else q / 2.0)


@kernel(overloads=((_f2, _f2, _f2, _f2),), cache=True, inline="always", nogil=True)
def _track_step(a, T1, T2, tmp):
    """a[25, C] <- the two tracks' symbol transitions applied to every class: track 1 (first symbol) by T1, then
    track 2 by T2."""
    C = a.shape[1]
    for c in range(C):
        for v1 in range(5):
            for s2 in range(5):
                acc = 0.0
                for u1 in range(5):
                    acc += a[5 * u1 + s2, c] * T1[u1, v1]
                tmp[5 * v1 + s2, c] = acc
        for v1 in range(5):
            for v2 in range(5):
                acc = 0.0
                for u2 in range(5):
                    acc += tmp[5 * v1 + u2, c] * T2[u2, v2]
                a[5 * v1 + v2, c] = acc


@kernel(overloads=((_i64_1, _f1, _f4, _f2, types.float64, types.float64, _f1, _f1, _i64_1, _i64_1),), cache=True,
        parallel=True, nogil=True)
def _target_loglik_tiles(offsets, pos, em25, entry, rate_cm, dl, ein_grid, eout_grid, units, tile_offsets):
    """Target relationship chain of each frame sequence [S, Ein, Eout], the read class carried. Track 1 symbols: 0, 1 =
    parent-0 classes (its full-sib symbols), 2, 3 = parent-1 classes, 4 = any class outside the four; track 2:
    full-sib symbols 2, 3. A full-sib symbol flips to the other with the Haldane probability q, leaves to each
    non-full-sib symbol with e_in / 3 per frame; a non-full-sib symbol returns to each full-sib symbol with e_out / 2,
    else moves among the non-full-sib symbols with probability q. em25[x, 5 s1 + s2]: the transfer matrix of track 1
    on homologue 1 and track 2 on homologue 2, em25[x, 25] the misfit matrix. e_in = 0 is the full-sib chain."""
    S = len(offsets) - 1
    NI = len(ein_grid)
    NO = len(eout_grid)
    C = em25.shape[2]
    out = np.zeros((S, NI, NO))
    rate = rate_cm * 1e-8
    # one unit per (sequence, entry rate), in balanced tiles (as frame_loglik's units)
    for t in prange(len(tile_offsets) - 1):
        for k in range(tile_offsets[t], tile_offsets[t + 1]):
            cell = units[k]
            s = cell // NI
            ii = cell % NI
            lo = offsets[s]
            hi = offsets[s + 1]
            T1 = np.zeros((5, 5))
            T2 = np.zeros((5, 5))
            a = np.zeros((25, C))
            tmp = np.zeros((25, C))
            w = np.zeros((25, C))
            ein = ein_grid[ii]
            for jj in range(NO):
                if ein == 0.0 and jj > 0:
                    out[s, ii, jj] = out[s, ii, 0]
                    continue
                eout = eout_grid[jj]
                pf = eout / (ein + eout)
                pn = 1.0 - pf
                for s1 in range(5):
                    p1 = pf / 2.0 if s1 < 2 else pn / 3.0
                    for s2 in range(5):
                        p2 = pf / 2.0 if (s2 == 2 or s2 == 3) else pn / 3.0
                        for c in range(C):
                            a[5 * s1 + s2, c] = p1 * p2 * entry[lo, c]
                ll = 0.0
                for x in range(lo, hi):
                    if x > lo:
                        q = 0.5 * (1.0 - math.exp(-2.0 * (pos[x] - pos[x - 1]) * rate))
                        _first_track_transition(T1, q, ein, eout)
                        _second_track_transition(T2, q, ein, eout)
                        _track_step(a, T1, T2, tmp)
                    tot = 0.0
                    for k in range(25):
                        for j in range(C):
                            v = 0.0
                            for i in range(C):
                                v += a[k, i] * ((1.0 - dl) * em25[x, k, i, j] + dl * em25[x, _MISFIT_25, i, j])
                            w[k, j] = v
                            tot += v
                    ll += math.log(tot)
                    for k in range(25):
                        for j in range(C):
                            a[k, j] = w[k, j] / tot
                out[s, ii, jj] = ll
    return out


def target_loglik(offsets, pos, em25, entry, rate_cm, dl, ein_grid, eout_grid):
    """[S, Ein, Eout] log-likelihoods of _target_loglik_tiles, its (sequence, entry rate) units in balanced tiles."""
    units, tile_offsets = _tiles(np.diff(offsets), len(ein_grid))
    return _target_loglik_tiles(offsets, pos, em25, entry, rate_cm, dl, ein_grid, eout_grid, units, tile_offsets)


@kernel(overloads=((_i64_1, _f1, _f4, _f2, types.float64, types.float64, types.float64, types.float64, _i64_1,
                    _i64_1),), cache=True, parallel=True, nogil=True)
def _target_posterior_tiles(offsets, pos, em25, entry, rate_cm, dl, ein, eout, units, tile_offsets):
    """Posterior probability [X] that the target's pair at each frame is not a full-sib configuration (one class of
    each parent), the read class summed, under the relationship chain of target_loglik at (r, delta, e_in, e_out)."""
    X = em25.shape[0]
    C = em25.shape[2]
    out = np.zeros(X)
    rate = rate_cm * 1e-8
    fsconf = np.zeros(25, np.bool_)
    for s1 in range(5):
        for s2 in range(5):
            if (s1 < 2 and (s2 == 2 or s2 == 3)) or ((s1 == 2 or s1 == 3) and s2 < 2):
                fsconf[5 * s1 + s2] = True
    pf = eout / (ein + eout)
    pn = 1.0 - pf
    for t in prange(len(tile_offsets) - 1):
        for k in range(tile_offsets[t], tile_offsets[t + 1]):
            s = units[k]
            lo = offsets[s]
            hi = offsets[s + 1]
            n = hi - lo
            T1 = np.zeros((5, 5))
            T2 = np.zeros((5, 5))
            fwd = np.zeros((n, 25, C))
            a = np.zeros((25, C))
            tmp = np.zeros((25, C))
            w = np.zeros((25, C))
            for s1 in range(5):
                p1 = pf / 2.0 if s1 < 2 else pn / 3.0
                for s2 in range(5):
                    p2 = pf / 2.0 if (s2 == 2 or s2 == 3) else pn / 3.0
                    for c in range(C):
                        a[5 * s1 + s2, c] = p1 * p2 * entry[lo, c]
            for i in range(n):
                x = lo + i
                if i > 0:
                    q = 0.5 * (1.0 - math.exp(-2.0 * (pos[x] - pos[x - 1]) * rate))
                    _first_track_transition(T1, q, ein, eout)
                    _second_track_transition(T2, q, ein, eout)
                    _track_step(a, T1, T2, tmp)
                tot = 0.0
                for k in range(25):
                    for j in range(C):
                        v = 0.0
                        for c in range(C):
                            v += a[k, c] * ((1.0 - dl) * em25[x, k, c, j] + dl * em25[x, _MISFIT_25, c, j])
                        w[k, j] = v
                        tot += v
                for k in range(25):
                    for j in range(C):
                        a[k, j] = w[k, j] / tot
                        fwd[i, k, j] = a[k, j]
            # backward messages bw[k, c]: the later frames given the state (k, c) after frame i (any positive scale)
            bw = np.ones((25, C)) / 25.0
            cv = np.zeros((25, C))
            for i in range(n - 1, -1, -1):
                x = lo + i
                tot = 0.0
                nf = 0.0
                for k in range(25):
                    for c in range(C):
                        pk = fwd[i, k, c] * bw[k, c]
                        tot += pk
                        if not fsconf[k]:
                            nf += pk
                out[x] = nf / tot
                if i > 0:
                    # cv(v, c) = sum_c' E_x(v)[c, c'] bw(v, c'); bw(u, c) <- sum_v A(u, v) cv(v, c)
                    for k in range(25):
                        for c in range(C):
                            v = 0.0
                            for j in range(C):
                                v += ((1.0 - dl) * em25[x, k, c, j] + dl * em25[x, _MISFIT_25, c, j]) * bw[k, j]
                            cv[k, c] = v
                    q = 0.5 * (1.0 - math.exp(-2.0 * (pos[x] - pos[x - 1]) * rate))
                    _first_track_transition(T1, q, ein, eout)
                    _second_track_transition(T2, q, ein, eout)
                    tot = 0.0
                    for c in range(C):
                        for u1 in range(5):
                            for v2 in range(5):
                                acc = 0.0
                                for v1 in range(5):
                                    acc += T1[u1, v1] * cv[5 * v1 + v2, c]
                                tmp[5 * u1 + v2, c] = acc
                        for u1 in range(5):
                            for u2 in range(5):
                                acc = 0.0
                                for v2 in range(5):
                                    acc += T2[u2, v2] * tmp[5 * u1 + v2, c]
                                bw[5 * u1 + u2, c] = acc
                                tot += acc
                    for k in range(25):
                        for c in range(C):
                            bw[k, c] /= tot
    return out


def target_posterior(offsets, pos, em25, entry, rate_cm, dl, ein, eout):
    """[X] posteriors of _target_posterior_tiles, its sequences in balanced tiles."""
    units, tile_offsets = _tiles(np.diff(offsets), 1)
    return _target_posterior_tiles(offsets, pos, em25, entry, rate_cm, dl, ein, eout, units, tile_offsets)


@kernel(overloads=((_i64_1, _f1, _f4, _f2, _i64_1, types.int64, _f1, types.float64, _f1, _f1, _i64_1, _i64_1),),
        cache=True, parallel=True, nogil=True)
def _error_run_loglik_tiles(offsets, pos, em, entry, segment, n_segments, r_grid, dl, eps_grid, xi_grid, units,
                            tile_offsets):
    """Error-run chain log-likelihood of every frame sequence, split into its segments [n_segments, R, E, X], the read
    class carried.

    Hidden (true transmission state s, shown state g, read class c) of the 4-state per-sib chain. s follows the
    Haldane chain (two bits, one meiosis each). Faithful (g = s): the next shown state is the next true state with
    probability 1 - eps, otherwise one of the three others (an error run starts). In a run (g != s): it ends with
    probability xi (g' = s'), else g persists (unless the true state moved onto it). The frame moves the class by the
    shown state's transfer matrix em[x, g] mixed with the misfit matrix em[x, 4] (weight dl). eps = 0 is the 4-state
    chain exactly.

    ``segment[x]``: the segment of frame x; a sequence's frames carry consecutive segment numbers (one per material
    unit it crosses, numbered sequence by sequence), so every segment belongs to one sequence and one thread. Each
    frame's predictive term log p(y_x | y_<x) is added to its segment: a segment's value is its frames' share of the
    sequence's forward log-likelihood, and the segments of a sequence sum to it exactly (up to summation order)."""
    R = len(r_grid)
    E = len(eps_grid)
    X = len(xi_grid)
    C = em.shape[2]
    out = np.zeros((n_segments, R, E, X))
    # one unit per (sequence, rate), in balanced tiles (as frame_loglik's units); a unit writes only its sequence's
    # segments at its rate
    for t in prange(len(tile_offsets) - 1):
        for k in range(tile_offsets[t], tile_offsets[t + 1]):
            cell = units[k]
            s = cell // R
            ri = cell % R
            lo = offsets[s]
            hi = offsets[s + 1]
            if hi <= lo:
                continue
            seg_lo = segment[lo]
            seg_hi = segment[hi - 1] + 1
            a = np.zeros((16, C))
            b = np.zeros((16, C))
            P = np.zeros((4, 4))
            rate = r_grid[ri] * 1e-8
            for ei in range(E):
                eps = eps_grid[ei]
                for xj in range(X):
                    if eps == 0.0 and xj > 0:
                        for g_ in range(seg_lo, seg_hi):
                            out[g_, ri, ei, xj] = out[g_, ri, ei, 0]
                        continue
                    xi = xi_grid[xj]
                    perr = eps / (eps + xi)
                    for k in range(16):
                        st = k // 4
                        g = k % 4
                        for c in range(C):
                            a[k, c] = 0.25 * ((1.0 - perr) if g == st else perr / 3.0) * entry[lo, c]
                    for x in range(lo, hi):
                        if x > lo:
                            q = 0.5 * (1.0 - math.exp(-2.0 * (pos[x] - pos[x - 1]) * rate))
                            for u in range(4):
                                for v in range(4):
                                    d = ((u ^ v) >> 1) + ((u ^ v) & 1)
                                    P[u, v] = (1.0 - q) ** (2 - d) * q ** d
                            for c in range(C):
                                for k in range(16):
                                    b[k, c] = 0.0
                                for st in range(4):
                                    for g in range(4):
                                        m = a[4 * st + g, c]
                                        if m == 0.0:
                                            continue
                                        for s2 in range(4):
                                            w = m * P[st, s2]
                                            if g == st:
                                                b[4 * s2 + s2, c] += w * (1.0 - eps)
                                                for g2 in range(4):
                                                    if g2 != s2:
                                                        b[4 * s2 + g2, c] += w * eps / 3.0
                                            else:
                                                if g == s2:
                                                    b[4 * s2 + s2, c] += w
                                                else:
                                                    b[4 * s2 + s2, c] += w * xi
                                                    b[4 * s2 + g, c] += w * (1.0 - xi)
                                for k in range(16):
                                    a[k, c] = b[k, c]
                        tot = 0.0
                        for k in range(16):
                            g = k % 4
                            for j in range(C):
                                v = 0.0
                                for i in range(C):
                                    v += a[k, i] * ((1.0 - dl) * em[x, g, i, j] + dl * em[x, _MISFIT_5, i, j])
                                b[k, j] = v
                                tot += v
                        out[segment[x], ri, ei, xj] += math.log(tot)
                        for k in range(16):
                            for j in range(C):
                                a[k, j] = b[k, j] / tot
    return out


def error_run_loglik(offsets, pos, em, entry, segment, n_segments, r_grid, dl, eps_grid, xi_grid):
    """[n_segments, R, E, X] segment log-likelihoods of _error_run_loglik_tiles, its (sequence, rate) units in
    balanced tiles."""
    units, tile_offsets = _tiles(np.diff(offsets), len(r_grid))
    return _error_run_loglik_tiles(offsets, pos, em, entry, segment, n_segments, r_grid, dl, eps_grid, xi_grid, units,
                                   tile_offsets)
