"""A step: every sample's best diploid state for a fixed panel H.

States: row pairs (i, j), i <= j, row-major; a row with an unknown copy (k, U), k ascending; both copies unknown
(U, U). A state's cost is minus its summed binned log emission (uncapped). The tie band (release 2.1) is
[cheapest, cheapest + the block's ``sample`` tie quantum]: costs that differ by rounding or by perturbation-level
amounts do not decide. Inside the band the category with the fewest unknown copies wins (a stated preference: row
pairs, then (k, U), then (U, U)). Release 2.2 (the meet rule; replaces the 2.1 convention D1, which took the first
tied state in row order): the sample is assigned the multiset intersection of the founder pairs of the winning
category's tied states, a copy that differs between them being an unknown copy U. One tied state is assigned as it
is; {(i, j), (i, j')} gives (i, U); {(1, 1), (1, 2), (2, 2)} or {(0, 1), (2, 3)} give (U, U). The intersection
depends only on the tied set, so the assignment is invariant to row order, sample order and the convention reversal:
the data cannot tell which tied founder supplies a copy, and only the founders every tied explanation shares receive
the sample's evidence.
The reported per-sample cost is minus the best binned switching-Viterbi log likelihood over all
states (switch penalty between adjacent bins), the model's NLL. Samples without any observed
site are set to (U, U) afterwards (their emissions are all zero, so ties would otherwise pick a
row pair and change the H step's founder order).
"""
from __future__ import annotations

from functools import lru_cache

import numba
import numpy as np
from numba import njit, prange

from ..conventions import rev


@lru_cache(maxsize=64)
def pair_indices(k: int) -> tuple[np.ndarray, np.ndarray]:
    """Read-only row-major (i, j), i <= j, index arrays of the row-pair states of K rows."""
    first, second = np.triu_indices(int(k))
    first = np.ascontiguousarray(first, dtype=np.int64)
    second = np.ascontiguousarray(second, dtype=np.int64)
    first.setflags(write=False)
    second.setflags(write=False)
    return first, second


@njit(cache=True, nogil=True)
def haplotype_patterns(haplotypes, first, second, snps_per_bin, n_bins):
    """Within-bin allele patterns (bit t = site t of the bin) of every row and of every row
    pair's AND, shapes (K, n_bins) and (pairs, n_bins)."""
    k, length = haplotypes.shape
    row_patterns = np.zeros((k, n_bins), dtype=np.int64)
    pair_patterns = np.empty((first.shape[0], n_bins), dtype=np.int64)
    for row in range(k):
        for b in range(n_bins):
            pattern = 0
            for t in range(snps_per_bin):
                site = b * snps_per_bin + t
                if site < length and haplotypes[row, site] == 1:
                    pattern |= 1 << t
            row_patterns[row, b] = pattern
    for p in range(first.shape[0]):
        for b in range(n_bins):
            pair_patterns[p, b] = row_patterns[first[p], b] & row_patterns[second[p], b]
    return row_patterns, pair_patterns


@njit(cache=True, nogil=True, inline="always")
def meet_pairs(m0, m1, c, d, k):
    """Multiset intersection of the sorted founder pairs (m0, m1) and (c, d) (k = an unknown copy, which is never a
    shared founder), as a sorted pair padded with unknown copies (k). Commutative and associative."""
    y0 = c
    y1 = d
    out0 = k
    out1 = k
    shared = 0
    if m0 < k:
        if y0 == m0:
            y0 = -1
            out0 = m0
            shared = 1
        elif y1 == m0:
            y1 = -1
            out0 = m0
            shared = 1
    if m1 < k:
        matched = False
        if y0 == m1:
            y0 = -1
            matched = True
        elif y1 == m1:
            y1 = -1
            matched = True
        if matched:
            if shared == 0:
                out0 = m1
            else:
                out1 = m1
    if out1 < out0:
        out0, out1 = out1, out0
    return out0, out1


@njit(cache=True, nogil=True, inline="always")
def band_meet(state_cost, threshold, first, second, n_pairs, k):
    """(a0, a1, category, tied) of a sample: inside the tie band [cheapest, cheapest + quantum] the category with the
    fewest unknown copies (0 row pairs, 1 (k, U), 2 (U, U); a stated preference), and in it the meet (multiset
    intersection, unknown copies k for differing copies) of its tied states' founder pairs; ``tied`` = the number of
    tied states. Order-free: the result depends only on the tied set."""
    n_states = n_pairs + k + 1
    for category in range(3):
        lo = 0 if category == 0 else (n_pairs if category == 1 else n_pairs + k)
        hi = n_pairs if category == 0 else (n_pairs + k if category == 1 else n_states)
        tied = 0
        m0 = k
        m1 = k
        for st in range(lo, hi):
            if state_cost[st] <= threshold:
                if st < n_pairs:
                    c = first[st]
                    d = second[st]
                elif st < n_pairs + k:
                    c = st - n_pairs
                    d = k
                else:
                    c = k
                    d = k
                if tied == 0:
                    m0 = c
                    m1 = d
                else:
                    m0, m1 = meet_pairs(m0, m1, c, d, k)
                tied += 1
        if tied > 0:
            return m0, m1, category, tied
    return k, k, 2, 1


@njit(cache=True, nogil=True, inline="always")
def band_state(state_cost, threshold, n_pairs, k, reverse):
    """Release 2.1 rule (workspaces without ``tie_meet``, e.g. local selection's bank): the state chosen inside the tie
    band: fewest unknown copies first (row pairs, then (k, U), then (U, U)); inside that category the first state in
    row order, or the last under reversed conventions (maths.ties.reverse_conventions)."""
    n_states = n_pairs + k + 1
    for category in range(3):
        lo = 0 if category == 0 else (n_pairs if category == 1 else n_pairs + k)
        hi = n_pairs if category == 0 else (n_pairs + k if category == 1 else n_states)
        if reverse:
            for st in range(hi - 1, lo - 1, -1):
                if state_cost[st] <= threshold:
                    return st
        else:
            for st in range(lo, hi):
                if state_cost[st] <= threshold:
                    return st
    return n_states - 1


@njit(cache=True, nogil=True, inline="always")
def _assign_sample(s, c0b, diff1_table, w_table, kw_cb, kw_diff_table, row_slots, pair_slots,
                   ww_bin_emissions, ww_total_cost, first, second, penalty, k, n_bins,
                   founder_diff1, emissions, assignments, unknown_slots, viterbi, state_cost, tie,
                   reverse, meet):
    """One sample's state choice and Viterbi likelihood (scratch: founder_diff1 (K, bins),
    emissions (states, bins)). ``meet``: the release 2.2 meet rule (band_meet; ``reverse`` then has no effect, there
    is no order to reverse), else the 2.1 rule (band_state)."""
    n_pairs = pair_slots.shape[0]
    n_states = n_pairs + k + 1
    # Above K = 3, gather each row's diff1 contraction once per sample: the same table scalar
    # enters the same expression, with K x bins instead of 2 x pairs x bins scattered reads.
    gather = k >= 4
    if gather:
        for row in range(k):
            for b in range(n_bins):
                founder_diff1[row, b] = diff1_table[b, row_slots[row, b], s]
    best_cost = np.inf
    state = 0
    for p in range(n_pairs):
        i = first[p]
        j = second[p]
        total = 0.0
        for b in range(n_bins):
            if gather:
                e = (c0b[s, b] + founder_diff1[i, b] + founder_diff1[j, b]
                     + w_table[b, pair_slots[p, b], s])
            else:
                e = (c0b[s, b] + diff1_table[b, row_slots[i, b], s]
                     + diff1_table[b, row_slots[j, b], s] + w_table[b, pair_slots[p, b], s])
            emissions[state, b] = e
            total += e
        state_cost[state] = -total
        if -total < best_cost:
            best_cost = -total
        state += 1
    for row in range(k):
        total = 0.0
        for b in range(n_bins):
            e = kw_cb[s, b] + kw_diff_table[b, row_slots[row, b], s]
            emissions[state, b] = e
            total += e
        state_cost[state] = -total
        if -total < best_cost:
            best_cost = -total
        state += 1
    for b in range(n_bins):
        emissions[state, b] = ww_bin_emissions[s, b]
    state_cost[state] = ww_total_cost[s]
    if ww_total_cost[s] < best_cost:
        best_cost = ww_total_cost[s]
    if meet:
        # Inside the tie quantum of the cheapest: the meet of the tied states (band_meet; order-free, release 2.2).
        # A copy that differs between tied states goes to no founder row (k in the pair); the unknown-copy count
        # stays that of the tied states (their category), so ambiguity is not counted as copies from outside.
        a0, a1, category, _ = band_meet(state_cost, best_cost + tie, first, second, n_pairs, k)
        assignments[s, 0] = a0
        assignments[s, 1] = a1
        unknown_slots[s] = category
    else:
        best_state = band_state(state_cost, best_cost + tie, n_pairs, k, reverse)
        if best_state < n_pairs:
            assignments[s, 0] = first[best_state]
            assignments[s, 1] = second[best_state]
            unknown_slots[s] = 0
        elif best_state < n_pairs + k:
            assignments[s, 0] = best_state - n_pairs
            assignments[s, 1] = k
            unknown_slots[s] = 1
        else:
            assignments[s, 0] = k
            assignments[s, 1] = k
            unknown_slots[s] = 2
    # Viterbi forward over bins, in place in the dead first emission column.
    for b in range(1, n_bins):
        best_previous = -np.inf
        for st in range(n_states):
            if emissions[st, 0] > best_previous:
                best_previous = emissions[st, 0]
        switch_base = best_previous - penalty
        for st in range(n_states):
            stay = emissions[st, 0]
            if stay > switch_base:
                emissions[st, 0] = stay + emissions[st, b]
            else:
                emissions[st, 0] = switch_base + emissions[st, b]
    best_final = -np.inf
    for st in range(n_states):
        if emissions[st, 0] > best_final:
            best_final = emissions[st, 0]
    viterbi[s] = best_final


@njit(cache=True, parallel=True, nogil=True)
def _assign_parallel(c0b, diff1_table, w_table, kw_cb, kw_diff_table, row_slots, pair_slots,
                     ww_bin_emissions, ww_total_cost, first, second, penalty, k, n_bins, tie, reverse, meet):
    n = c0b.shape[0]
    n_states = pair_slots.shape[0] + k + 1
    assignments = np.empty((n, 2), dtype=np.int64)
    unknown_slots = np.empty(n, dtype=np.int64)
    viterbi = np.empty(n, dtype=np.float64)
    for s in prange(n):
        founder_diff1 = np.empty((k, n_bins), dtype=np.float64)
        emissions = np.empty((n_states, n_bins), dtype=np.float64)
        state_cost = np.empty(n_states, dtype=np.float64)
        _assign_sample(s, c0b, diff1_table, w_table, kw_cb, kw_diff_table, row_slots,
                       pair_slots, ww_bin_emissions, ww_total_cost, first, second, penalty, k,
                       n_bins, founder_diff1, emissions, assignments, unknown_slots, viterbi,
                       state_cost, tie, reverse, meet)
    return assignments, -viterbi, unknown_slots


@njit(cache=True, nogil=True)
def assign_serial(c0b, diff1_table, w_table, kw_cb, kw_diff_table, row_slots, pair_slots,
                  ww_bin_emissions, ww_total_cost, first, second, penalty, k, n_bins, tie, reverse, meet):
    """The A step on one thread, with one scratch slab for all samples (no per-sample
    allocation); the same arithmetic as the parallel driver."""
    n = c0b.shape[0]
    n_states = pair_slots.shape[0] + k + 1
    assignments = np.empty((n, 2), dtype=np.int64)
    unknown_slots = np.empty(n, dtype=np.int64)
    viterbi = np.empty(n, dtype=np.float64)
    founder_diff1 = np.empty((k, n_bins), dtype=np.float64)
    emissions = np.empty((n_states, n_bins), dtype=np.float64)
    state_cost = np.empty(n_states, dtype=np.float64)
    for s in range(n):
        _assign_sample(s, c0b, diff1_table, w_table, kw_cb, kw_diff_table, row_slots,
                       pair_slots, ww_bin_emissions, ww_total_cost, first, second, penalty, k,
                       n_bins, founder_diff1, emissions, assignments, unknown_slots, viterbi,
                       state_cost, tie, reverse, meet)
    return assignments, -viterbi, unknown_slots


def assign(workspace, haplotypes):
    """The A step of a panel: (assignments int64[N, 2], per-sample cost float64[N], unknown
    slots int64[N]), with samples without any observed site set to (U, U)."""
    k = haplotypes.shape[0]
    first, second = pair_indices(k)
    row_patterns, pair_patterns = haplotype_patterns(
        np.ascontiguousarray(haplotypes, dtype=np.int64), first, second,
        workspace.snps_per_bin, workspace.n_bins)
    tables = workspace.patterns
    # Take the slots (adding unseen patterns) before reading the tables.
    row_slots = tables.slots(row_patterns)
    pair_slots = tables.slots(pair_patterns)
    kernel = assign_serial if numba.get_num_threads() == 1 else _assign_parallel
    assignments, cost, unknown_slots = kernel(
        workspace.c0b, tables.diff1_table, tables.w_table, workspace.kw_cb,
        tables.kw_diff_table, row_slots, pair_slots, workspace.ww_bin_emissions,
        workspace.ww_total_cost, first, second, workspace.switch_penalty, k, workspace.n_bins,
        float(workspace.quanta.sample), rev("d1"), bool(workspace.tie_meet))
    uninformative = workspace.uninformative
    assignments[uninformative] = k
    unknown_slots[uninformative] = 2
    return assignments, cost, unknown_slots


@njit(cache=True, parallel=True, nogil=True)
def _tie_mask_kernel(c0b, diff1_table, w_table, kw_cb, kw_diff_table, row_slots, pair_slots, ww_total_cost,
                     first, second, k, n_bins, tie):
    """uint8[N, states]: 1 at the winning category's tied states of every sample (the set T whose meet band_meet
    assigns), with the A step's state costs (the same expressions in the same order, so the same band)."""
    n = c0b.shape[0]
    n_pairs = pair_slots.shape[0]
    n_states = n_pairs + k + 1
    mask = np.zeros((n, n_states), dtype=np.uint8)
    for s in prange(n):
        state_cost = np.empty(n_states, dtype=np.float64)
        best_cost = np.inf
        for p in range(n_pairs):
            i = first[p]
            j = second[p]
            total = 0.0
            for b in range(n_bins):
                total += (c0b[s, b] + diff1_table[b, row_slots[i, b], s]
                          + diff1_table[b, row_slots[j, b], s] + w_table[b, pair_slots[p, b], s])
            state_cost[p] = -total
            if -total < best_cost:
                best_cost = -total
        for row in range(k):
            total = 0.0
            for b in range(n_bins):
                total += kw_cb[s, b] + kw_diff_table[b, row_slots[row, b], s]
            state_cost[n_pairs + row] = -total
            if -total < best_cost:
                best_cost = -total
        state_cost[n_states - 1] = ww_total_cost[s]
        if ww_total_cost[s] < best_cost:
            best_cost = ww_total_cost[s]
        threshold = best_cost + tie
        for category in range(3):
            lo = 0 if category == 0 else (n_pairs if category == 1 else n_pairs + k)
            hi = n_pairs if category == 0 else (n_pairs + k if category == 1 else n_states)
            found = False
            for st in range(lo, hi):
                if state_cost[st] <= threshold:
                    mask[s, st] = 1
                    found = True
            if found:
                break
    return mask


def tie_mask(workspace, haplotypes):
    """uint8[N, states] of a panel (states in the A step's order: row pairs row-major, (k, U), (U, U)): every sample's
    tied states T in the winning category (one state when the band holds one); samples without any observed site
    hold (U, U) only. The cavity score counts a tied sample 1/|T| at each state of T (release 2.2)."""
    k = haplotypes.shape[0]
    first, second = pair_indices(k)
    row_patterns, pair_patterns = haplotype_patterns(
        np.ascontiguousarray(haplotypes, dtype=np.int64), first, second,
        workspace.snps_per_bin, workspace.n_bins)
    tables = workspace.patterns
    row_slots = tables.slots(row_patterns)
    pair_slots = tables.slots(pair_patterns)
    mask = _tie_mask_kernel(
        workspace.c0b, tables.diff1_table, tables.w_table, workspace.kw_cb, tables.kw_diff_table, row_slots,
        pair_slots, workspace.ww_total_cost, first, second, k, workspace.n_bins, float(workspace.quanta.sample))
    uninformative = workspace.uninformative
    mask[uninformative] = 0
    mask[uninformative, -1] = 1
    return mask
