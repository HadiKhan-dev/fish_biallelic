"""A step: every sample's best diploid state for a fixed panel H.

States, in this order (the tie rule is the first strict minimum): row pairs (i, j), i <= j,
row-major; a row with an unknown copy (k, U), k ascending; both copies unknown (U, U). A state's
cost is minus its summed binned log emission (uncapped); the chosen state is the cheapest one.
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
def _assign_sample(s, c0b, diff1_table, w_table, kw_cb, kw_diff_table, row_slots, pair_slots,
                   ww_bin_emissions, ww_total_cost, first, second, penalty, k, n_bins,
                   founder_diff1, emissions, assignments, unknown_slots, viterbi):
    """One sample's state choice and Viterbi likelihood (scratch: founder_diff1 (K, bins),
    emissions (states, bins))."""
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
    best_state = 0
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
        if -total < best_cost:
            best_cost = -total
            best_state = state
        state += 1
    for row in range(k):
        total = 0.0
        for b in range(n_bins):
            e = kw_cb[s, b] + kw_diff_table[b, row_slots[row, b], s]
            emissions[state, b] = e
            total += e
        if -total < best_cost:
            best_cost = -total
            best_state = state
        state += 1
    for b in range(n_bins):
        emissions[state, b] = ww_bin_emissions[s, b]
    if ww_total_cost[s] < best_cost:
        best_cost = ww_total_cost[s]
        best_state = state
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
                     ww_bin_emissions, ww_total_cost, first, second, penalty, k, n_bins):
    n = c0b.shape[0]
    n_states = pair_slots.shape[0] + k + 1
    assignments = np.empty((n, 2), dtype=np.int64)
    unknown_slots = np.empty(n, dtype=np.int64)
    viterbi = np.empty(n, dtype=np.float64)
    for s in prange(n):
        founder_diff1 = np.empty((k, n_bins), dtype=np.float64)
        emissions = np.empty((n_states, n_bins), dtype=np.float64)
        _assign_sample(s, c0b, diff1_table, w_table, kw_cb, kw_diff_table, row_slots,
                       pair_slots, ww_bin_emissions, ww_total_cost, first, second, penalty, k,
                       n_bins, founder_diff1, emissions, assignments, unknown_slots, viterbi)
    return assignments, -viterbi, unknown_slots


@njit(cache=True, nogil=True)
def assign_serial(c0b, diff1_table, w_table, kw_cb, kw_diff_table, row_slots, pair_slots,
                  ww_bin_emissions, ww_total_cost, first, second, penalty, k, n_bins):
    """The A step on one thread, with one scratch slab for all samples (no per-sample
    allocation); the same arithmetic as the parallel driver."""
    n = c0b.shape[0]
    n_states = pair_slots.shape[0] + k + 1
    assignments = np.empty((n, 2), dtype=np.int64)
    unknown_slots = np.empty(n, dtype=np.int64)
    viterbi = np.empty(n, dtype=np.float64)
    founder_diff1 = np.empty((k, n_bins), dtype=np.float64)
    emissions = np.empty((n_states, n_bins), dtype=np.float64)
    for s in range(n):
        _assign_sample(s, c0b, diff1_table, w_table, kw_cb, kw_diff_table, row_slots,
                       pair_slots, ww_bin_emissions, ww_total_cost, first, second, penalty, k,
                       n_bins, founder_diff1, emissions, assignments, unknown_slots, viterbi)
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
        workspace.ww_total_cost, first, second, workspace.switch_penalty, k, workspace.n_bins)
    uninformative = workspace.uninformative
    assignments[uninformative] = k
    unknown_slots[uninformative] = 2
    return assignments, cost, unknown_slots
