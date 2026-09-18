"""Founder allele updates conditional on sample assignments."""
from __future__ import annotations


import numpy as np

import numba


from numba import njit, prange


def _update_H(probs_k, H_k, A, lam, cost_WW=None, log_probs=None,
              h_genotype_cost=None, h_wildcard_cost=None):
    """For each (founder, kept site), pick the binary value that minimises
    NLL contribution from samples carrying that founder.

    Updates H_k in-place and returns the number of bits flipped (so the
    coordinate descent loop can detect convergence).

    Arguments:
        probs_k: (N, L_kept, 3)
        H_k:     (K, L_kept) — modified in place
        A:       (N, 2)   pair assignments, with K used as the wildcard sentinel
        lam:     wildcard penalty
        cost_WW: (N, L_kept) optional — precomputed per-(sample, site) WW
                  cost from `_per_site_cost_W_W(probs_k, lam)`.  When
                  provided, threads through to `_update_one_founder` and
                  ultimately to the kernel, which reads cost_WW[s, l]
                  instead of recomputing it inline for every (k, l).
                  When None, the kernel falls back to inline computation
                  (so the wrapper computes it once here for consistency).
        log_probs: (N, L_kept, 3) optional — precomputed
                  log(max(probs_k[s, l, g], LOG_EPS_LOCAL)) (Tier 0).
                  When provided, the kernel uses it in place of every
                  inline log call.  Same caching pattern as cost_WW.

    Returns:
        n_changes: int — number of (founder, site) bits that flipped
    """
    K, L = H_k.shape
    N = probs_k.shape[0]
    W = K

    # Compute cost_WW once here if not supplied, so the K founder
    # updates below all share the same precomputed array (no per-
    # founder re-derivation).
    if cost_WW is None:
        cost_WW = discovery_objectives._per_site_cost_W_W(probs_k, lam)

    # Same pattern for log_probs (Tier 0).  When called from
    # `_fit_at_fixed_K`, log_probs arrives precomputed and is reused
    # across all K founder updates AND across all CD iterations.
    if log_probs is None:
        log_probs = discovery_objectives._log_probs_kernel(
            discovery_objectives._maybe_c_contig(probs_k, np.float64)
        )

    if h_genotype_cost is None or h_wildcard_cost is None:
        (_, h_genotype_cost, h_wildcard_cost) = discovery_fitting._prepare_fit_cost_tables(
            discovery_objectives._maybe_c_contig(cost_WW, np.float64),
            discovery_objectives._maybe_c_contig(log_probs, np.float64),
            float(lam),
        )
        h_genotype_cost = np.ascontiguousarray(
            h_genotype_cost.transpose(1, 0, 2)
        )
        h_wildcard_cost = np.ascontiguousarray(
            h_wildcard_cost.transpose(1, 0, 2)
        )

    # We update founders in decreasing order of usage.  Compute usage from A
    # via an njit kernel — the inner Python loop over K with two boolean-
    # mask sums per K was a small hot spot (4K mask scans, each scanning
    # N entries).  The kernel does it in a single pass over A.
    A_c = discovery_objectives._maybe_c_contig(A, np.int64)
    (kk_idx, kj_idx, kW_idx, partner_J,
     n_kk, n_J, n_P, usage) = _classify_all_founder_buckets(A_c, K)
    update_order = np.argsort(-usage, kind='stable')
    update_order = update_order[usage[update_order] > 0]

    # A full H sweep is Gauss-Seidel in founder order but independent across
    # sites.  Run one parallel site kernel for the whole ordered sweep instead
    # of launching and synchronising a separate prange kernel for every
    # founder.  Within each site the kernel preserves the exact founder order,
    # H/J/P bucket order, ascending sample order, cap, accumulation order, and
    # tie rule of the historical per-founder path.
    H_c = discovery_objectives._maybe_c_contig(H_k, np.int64)
    sweep_kernel = _update_H_ordered_sweep_kernel
    if numba.get_num_threads() == 1:
        sweep_kernel = _update_H_ordered_sweep_kernel_serial
    n_changes = sweep_kernel(
        H_c, np.ascontiguousarray(update_order, dtype=np.int64),
        kk_idx, kj_idx, kW_idx, partner_J, n_kk, n_J, n_P,
        discovery_objectives._maybe_c_contig(h_genotype_cost, np.float64),
        discovery_objectives._maybe_c_contig(h_wildcard_cost, np.float64),
    )
    if H_c is not H_k:
        H_k[...] = H_c
    return int(n_changes)


@njit(cache=True, nogil=True)
def _classify_all_founder_buckets(A, K):
    """Classify every founder's supporting samples in one compiled pass.

    This is the all-founder equivalent of :func:`_classify_founder_buckets`.
    Rows within each H/J/P bucket are appended in ascending sample order, so
    the downstream floating-point accumulation order is unchanged.  The
    returned ``usage`` is exactly the count used for the stable decreasing-
    usage Gauss-Seidel order; a homozygous pair contributes two copies.

    The four dense ``(K, N)`` buffers are bounded: at experimental K=16 and
    N=116 they occupy under 60 KiB in total, while replacing K allocations and
    K compiled-dispatch boundaries.
    """
    N = A.shape[0]
    W = K
    kk = np.empty((K, N), dtype=np.int64)
    kj = np.empty((K, N), dtype=np.int64)
    kW = np.empty((K, N), dtype=np.int64)
    partner = np.empty((K, N), dtype=np.int64)
    n_kk = np.zeros(K, dtype=np.int64)
    n_J = np.zeros(K, dtype=np.int64)
    n_P = np.zeros(K, dtype=np.int64)
    usage = np.zeros(K, dtype=np.int64)

    # Each assignment contains at most two real founders.  Visit samples once
    # and append directly to the one or two affected founder rows.  Because s
    # is ascending, every per-founder bucket retains the established sample
    # accumulation order even for unsorted but otherwise valid A rows.
    for s in range(N):
        a0 = A[s, 0]
        a1 = A[s, 1]
        if a0 != W:
            usage[a0] += 1
        if a1 != W:
            usage[a1] += 1

        if a0 != W:
            if a0 == a1:
                position = n_kk[a0]
                kk[a0, position] = s
                n_kk[a0] = position + 1
            elif a1 == W:
                position = n_P[a0]
                kW[a0, position] = s
                n_P[a0] = position + 1
            else:
                position = n_J[a0]
                kj[a0, position] = s
                partner[a0, position] = a1
                n_J[a0] = position + 1

        if a1 != W and a1 != a0:
            position = n_J[a1]
            kj[a1, position] = s
            partner[a1, position] = a0
            n_J[a1] = position + 1

    return kk, kj, kW, partner, n_kk, n_J, n_P, usage


@njit(cache=True, parallel=True, fastmath=False, nogil=True)
def _update_H_ordered_sweep_kernel(H_full, update_order,
                                    kk_idx, kj_idx, kW_idx, partner_J,
                                    n_kk, n_J, n_P,
                                    genotype_cost, real_wildcard_cost):
    """Parallel ordered founder sweep using evidence-only cost tables."""
    L = H_full.shape[1]
    n_changes = 0
    for l in prange(L):
        for order_position in range(update_order.shape[0]):
            k = update_order[order_position]
            cur_val = H_full[k, l]
            nll0 = 0.0
            nll1 = 0.0

            for ii in range(n_kk[k]):
                s = kk_idx[k, ii]
                nll0 += genotype_cost[l, s, 0]
                nll1 += genotype_cost[l, s, 2]

            for ii in range(n_J[k]):
                s = kj_idx[k, ii]
                j = partner_J[k, ii]
                partner_h = H_full[j, l]
                nll0 += genotype_cost[l, s, partner_h]
                nll1 += genotype_cost[l, s, partner_h + 1]

            for ii in range(n_P[k]):
                s = kW_idx[k, ii]
                nll0 += real_wildcard_cost[l, s, 0]
                nll1 += real_wildcard_cost[l, s, 1]

            diff = nll0 - nll1
            if diff < 0.0:
                diff = -diff
            if diff < 1e-9:
                new_val = cur_val
            elif nll0 < nll1:
                new_val = 0
            else:
                new_val = 1
            H_full[k, l] = new_val
            if new_val != cur_val:
                n_changes += 1
    return n_changes


@njit(cache=True, parallel=False, fastmath=False, nogil=True)
def _update_H_ordered_sweep_kernel_serial(H_full, update_order,
                                           kk_idx, kj_idx, kW_idx, partner_J,
                                           n_kk, n_J, n_P,
                                           genotype_cost,
                                           real_wildcard_cost):
    """One-thread counterpart of `_update_H_ordered_sweep_kernel`."""
    L = H_full.shape[1]
    n_changes = 0
    for l in range(L):
        for order_position in range(update_order.shape[0]):
            k = update_order[order_position]
            cur_val = H_full[k, l]
            nll0 = 0.0
            nll1 = 0.0

            for ii in range(n_kk[k]):
                s = kk_idx[k, ii]
                nll0 += genotype_cost[l, s, 0]
                nll1 += genotype_cost[l, s, 2]

            for ii in range(n_J[k]):
                s = kj_idx[k, ii]
                j = partner_J[k, ii]
                partner_h = H_full[j, l]
                nll0 += genotype_cost[l, s, partner_h]
                nll1 += genotype_cost[l, s, partner_h + 1]

            for ii in range(n_P[k]):
                s = kW_idx[k, ii]
                nll0 += real_wildcard_cost[l, s, 0]
                nll1 += real_wildcard_cost[l, s, 1]

            diff = nll0 - nll1
            if diff < 0.0:
                diff = -diff
            if diff < 1e-9:
                new_val = cur_val
            elif nll0 < nll1:
                new_val = 0
            else:
                new_val = 1
            H_full[k, l] = new_val
            if new_val != cur_val:
                n_changes += 1
    return n_changes


@njit(cache=True, nogil=True)
def _stable_descending_usage_order(usage):
    """Stable decreasing positive-usage order with lower index on ties.

    A zero-use founder has empty H/J/P buckets and therefore retains every
    allele under the established tie rule; omitting it is exact.
    """
    K = usage.shape[0]
    order = np.arange(K, dtype=np.int64)
    for index in range(1, K):
        current = order[index]
        position = index
        while position > 0 and usage[current] > usage[order[position - 1]]:
            order[position] = order[position - 1]
            position -= 1
        order[position] = current
    positive = 0
    while positive < K and usage[order[positive]] > 0:
        positive += 1
    return order[:positive]


@njit(cache=True, parallel=True, fastmath=False, nogil=True)
def _fixed_k_transition_batch_pattern_kernel(
        C0b, diff1_table, w_table, kW_Cb, kWdiff_table,
        WW_bin_emis, WW_total_cost, rr_i, rr_j, penalty, snps_per_bin, n_bins,
        h_genotype_cost, h_wildcard_cost, uninformative_samples, H_batch):
    """Evaluate independent ordered-H coordinate transitions in parallel."""
    B = H_batch.shape[0]
    K = H_batch.shape[1]
    N = C0b.shape[0]
    H_next = np.empty_like(H_batch)
    A_batch = np.empty((B, N, 2), dtype=np.int64)
    cost_batch = np.empty((B, N), dtype=np.float64)
    wildcard_batch = np.empty((B, N), dtype=np.int64)
    h_changes = np.empty(B, dtype=np.int64)
    for start in prange(B):
        is_binary, h_patterns, pair_patterns = discovery_assignments._binary_haplotype_patterns(
            H_batch[start], rr_i, rr_j,
            snps_per_bin, n_bins, H_batch.shape[2],
        )
        if not is_binary:
            h_changes[start] = -1
            continue
        A, cost, wildcard = discovery_assignments._update_A_fused_pattern_kernel_serial(
            C0b, diff1_table, w_table, kW_Cb, kWdiff_table,
            h_patterns, pair_patterns, WW_bin_emis, WW_total_cost,
            rr_i, rr_j, penalty, K, n_bins,
        )
        # Missing cells have zero weight in the cost tables. Wholly unobserved
        # samples must also be WW before counting founder usage: arbitrary
        # zero-cost RR ties would otherwise change the H sweep order.
        if uninformative_samples is not None:
            for sample in range(N):
                if uninformative_samples[sample]:
                    A[sample, 0] = K
                    A[sample, 1] = K
                    wildcard[sample] = 2
        A_batch[start] = A
        cost_batch[start] = cost
        wildcard_batch[start] = wildcard
        H_work = H_batch[start].copy()
        (kk_idx, kj_idx, kW_idx, partner_J,
         n_kk, n_J, n_P, usage) = _classify_all_founder_buckets(A, K)
        update_order = _stable_descending_usage_order(usage)
        changes = _update_H_ordered_sweep_kernel_serial(
            H_work, update_order,
            kk_idx, kj_idx, kW_idx, partner_J,
            n_kk, n_J, n_P,
            h_genotype_cost, h_wildcard_cost,
        )
        H_next[start] = H_work
        h_changes[start] = changes
    return (
        H_next, A_batch, cost_batch, wildcard_batch, h_changes,
    )

import haplotype_reconstruction.discovery.assignments as discovery_assignments
import haplotype_reconstruction.discovery.fitting as discovery_fitting
import haplotype_reconstruction.discovery.objectives as discovery_objectives
