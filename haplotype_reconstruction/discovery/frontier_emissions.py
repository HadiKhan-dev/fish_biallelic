"""Bounded, ordered emission reuse within one fixed-K search frontier."""
from __future__ import annotations

import numpy as np
from numba import njit, prange


# Per numerical worker, not per block or per search trajectory. A one-sample
# tile that would exceed this budget instead uses the established batch path.
_EMISSION_TILE_BYTES = 8 * 1024 * 1024
_SAMPLE_TILE_WIDTH = 16


def prepare_frontier(H_batch, rr_i, rr_j):
    """Index exact rows and ordered pairs; never canonicalize pair operands.

    Sorting compact integer/byte keys changes only private storage order. Each
    panel retains its original RR, RW, WW state order through ``state_indices``.
    No evidence-dependent state or search start is omitted.
    """
    B, K, L = H_batch.shape
    packed = np.packbits(H_batch, axis=2, bitorder="little")
    row_keys = packed.reshape(B * K, -1).view(
        np.dtype((np.void, packed.shape[2]))
    ).reshape(-1)
    _, row_first, row_inverse = np.unique(
        row_keys, return_index=True, return_inverse=True
    )
    row_inverse = row_inverse.reshape(B, K)
    n_rows = len(row_first)
    pair_keys = (
        row_inverse[:, rr_i] * n_rows + row_inverse[:, rr_j]
    )
    unique_pairs, pair_inverse = np.unique(pair_keys, return_inverse=True)
    n_pairs = len(unique_pairs)
    state_indices = np.empty((B, len(rr_i) + K), dtype=np.int64)
    state_indices[:,:len(rr_i)] = pair_inverse.reshape(B, -1)
    state_indices[:, len(rr_i):] = row_inverse + n_pairs
    rows = np.ascontiguousarray(H_batch.reshape(B * K, L)[row_first])
    return (
        rows, unique_pairs // n_rows, unique_pairs % n_rows,
        state_indices,
    )


@njit(cache=True, nogil=True)
def _row_patterns(rows, snps_per_bin, n_bins):
    patterns = np.zeros((rows.shape[0], n_bins), dtype=np.int64)
    for row in range(rows.shape[0]):
        for b in range(n_bins):
            value = 0
            for t in range(snps_per_bin):
                site = b * snps_per_bin + t
                if site < rows.shape[1]:
                    value |= rows[row, site] << t
            patterns[row, b] = value
    return patterns


@njit(cache=True, parallel=True, fastmath=False, nogil=True)
def _assign_frontier(
        C0b, diff1_table, w_table, kW_Cb, kWdiff_table,
        WW_bin_emis, WW_total_cost, patterns, pair_i, pair_j,
        state_indices, rr_i, rr_j, penalty, K, n_bins, tile_width):
    """Share immutable emissions, retaining panel-local argmin and Viterbi.

    Tile columns are contiguous samples in the evidence pattern tables. The
    bin reduction for every sample/state remains strictly ascending, as does
    the state traversal within every baseline argmin and Viterbi step.
    """
    N = C0b.shape[0]
    B = state_indices.shape[0]
    n_pairs = pair_i.shape[0]
    n_rows = patterns.shape[0]
    n_shared = n_pairs + n_rows
    n_rr = rr_i.shape[0]
    n_states = n_rr + K + 1
    A = np.empty((B, N, 2), dtype=np.int64)
    cost = np.empty((B, N), dtype=np.float64)
    wildcard = np.empty((B, N), dtype=np.int64)
    for tile in prange((N + tile_width - 1) // tile_width):
        first = tile * tile_width
        width = min(tile_width, N - first)
        emissions = np.empty((n_shared, n_bins, width), dtype=np.float64)
        totals = np.zeros((n_shared, width), dtype=np.float64)
        for p in range(n_pairs):
            i = pair_i[p]
            j = pair_j[p]
            for b in range(n_bins):
                pi = patterns[i, b]
                pj = patterns[j, b]
                for t in range(width):
                    s = first + t
                    # Preserve the ordered left-associated scalar expression.
                    e = (C0b[s, b] + diff1_table[b, pi, s]
                         + diff1_table[b, pj, s] + w_table[b, pi & pj, s])
                    emissions[p, b, t] = e
                    totals[p, t] += e
        for row in range(n_rows):
            state = n_pairs + row
            for b in range(n_bins):
                pattern = patterns[row, b]
                for t in range(width):
                    s = first + t
                    e = kW_Cb[s, b] + kWdiff_table[b, pattern, s]
                    emissions[state, b, t] = e
                    totals[state, t] += e

        alpha = np.empty((n_states, width), dtype=np.float64)
        best_cost = np.empty(width, dtype=np.float64)
        best_state = np.empty(width, dtype=np.int64)
        best_previous = np.empty(width, dtype=np.float64)
        for start in range(B):
            for t in range(width):
                best_cost[t] = np.inf
                best_state[t] = 0
            for state in range(n_states - 1):
                shared = state_indices[start, state]
                for t in range(width):
                    candidate = -totals[shared, t]
                    if candidate < best_cost[t]:
                        best_cost[t] = candidate
                        best_state[t] = state
                    alpha[state, t] = emissions[shared, 0, t]
            for t in range(width):
                s = first + t
                if WW_total_cost[s] < best_cost[t]:
                    best_state[t] = n_states - 1
                alpha[n_states - 1, t] = WW_bin_emis[s, 0]
                state = best_state[t]
                if state < n_rr:
                    A[start, s, 0] = rr_i[state]
                    A[start, s, 1] = rr_j[state]
                    wildcard[start, s] = 0
                elif state < n_rr + K:
                    A[start, s, 0] = state - n_rr
                    A[start, s, 1] = K
                    wildcard[start, s] = 1
                else:
                    A[start, s, 0] = K
                    A[start, s, 1] = K
                    wildcard[start, s] = 2
            for b in range(1, n_bins):
                for t in range(width):
                    best_previous[t] = -np.inf
                for state in range(n_states):
                    for t in range(width):
                        if alpha[state, t] > best_previous[t]:
                            best_previous[t] = alpha[state, t]
                for state in range(n_states - 1):
                    shared = state_indices[start, state]
                    for t in range(width):
                        switch_base = best_previous[t] - penalty
                        stay = alpha[state, t]
                        e = emissions[shared, b, t]
                        if stay > switch_base:
                            alpha[state, t] = stay + e
                        else:
                            alpha[state, t] = switch_base + e
                for t in range(width):
                    switch_base = best_previous[t] - penalty
                    stay = alpha[n_states - 1, t]
                    e = WW_bin_emis[first + t, b]
                    if stay > switch_base:
                        alpha[n_states - 1, t] = stay + e
                    else:
                        alpha[n_states - 1, t] = switch_base + e
            for t in range(width):
                best_final = -np.inf
                for state in range(n_states):
                    if alpha[state, t] > best_final:
                        best_final = alpha[state, t]
                cost[start, first + t] = -best_final
    return A, cost, wildcard


@njit(cache=True, parallel=True, fastmath=False, nogil=True)
def _update_frontier_H(
        H_batch, A, wildcard, uninformative_samples,
        h_genotype_cost, h_wildcard_cost):
    """Apply the established missing policy and exact panel-local H sweep."""
    H_next = H_batch.copy()
    changes = np.empty(H_batch.shape[0], dtype=np.int64)
    K = H_batch.shape[1]
    for start in prange(H_batch.shape[0]):
        if uninformative_samples is not None:
            for sample in range(A.shape[1]):
                if uninformative_samples[sample]:
                    A[start, sample, 0] = K
                    A[start, sample, 1] = K
                    wildcard[start, sample] = 2
        (kk_idx, kj_idx, kW_idx, partner_J,
         n_kk, n_J, n_P, usage) = (
            discovery_founder_updates._classify_all_founder_buckets(A[start], K)
        )
        order = discovery_founder_updates._stable_descending_usage_order(usage)
        changes[start] = discovery_founder_updates._update_H_ordered_sweep_kernel_serial(
            H_next[start], order, kk_idx, kj_idx, kW_idx, partner_J,
            n_kk, n_J, n_P, h_genotype_cost, h_wildcard_cost,
        )
    return H_next, changes


def transition_batch(
        C0b, diff1_table, w_table, kW_Cb, kWdiff_table,
        WW_bin_emis, WW_total_cost, rr_i, rr_j, penalty, snps_per_bin,
        n_bins, h_genotype_cost, h_wildcard_cost, uninformative_samples,
        H_batch):
    """Evaluate a frontier with bounded emission reuse or the original kernel."""
    rows, pair_i, pair_j, state_indices = prepare_frontier(H_batch, rr_i, rr_j)
    n_shared = len(pair_i) + len(rows)
    bytes_per_sample = 8 * (n_shared * (n_bins + 1)
                            + len(rr_i) + H_batch.shape[1] + 4)
    tile_width = min(_SAMPLE_TILE_WIDTH, _EMISSION_TILE_BYTES // bytes_per_sample)
    if tile_width < 1:
        return discovery_founder_updates._fixed_k_transition_batch_pattern_kernel(
            C0b, diff1_table, w_table, kW_Cb, kWdiff_table,
            WW_bin_emis, WW_total_cost, rr_i, rr_j, penalty,
            snps_per_bin, n_bins, h_genotype_cost, h_wildcard_cost,
            uninformative_samples, H_batch,
        )
    patterns = _row_patterns(rows, snps_per_bin, n_bins)
    A, cost, wildcard = _assign_frontier(
        C0b, diff1_table, w_table, kW_Cb, kWdiff_table,
        WW_bin_emis, WW_total_cost, patterns, pair_i, pair_j,
        state_indices, rr_i, rr_j, penalty, H_batch.shape[1], n_bins,
        tile_width,
    )
    H_next, changes = _update_frontier_H(
        H_batch, A, wildcard, uninformative_samples,
        h_genotype_cost, h_wildcard_cost,
    )
    return H_next, A, cost, wildcard, changes


import haplotype_reconstruction.discovery.founder_updates as discovery_founder_updates
