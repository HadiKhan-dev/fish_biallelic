"""Pool emission cache and subset-scoring kernels (precomputed-tensor path plus the
fused log-free fallback), split out of bhd_kernels.  Imports primitives from the
bhd_kernels foundation and _compute_nll_for_subset from bhd_fit (single-subset
fallback / equivalence reference)."""

import numpy as np
import math
from numba import njit, prange

from bhd_kernels import (
    _log_probs_kernel,
    _per_site_cost_W_W,
)
from bhd_fit import (
    _compute_nll_for_subset,
)
from bhd_config import (
    DEFAULT_LAMBDA,
    POOL_EMISSION_CACHE_MAX_BYTES,
    VITERBI_SNPS_PER_BIN,
    VITERBI_SWITCH_PENALTY,
)


# =============================================================================
# POOL EMISSION CACHE — precompute Viterbi emissions once per BIC-search
# =============================================================================
#
# The recovery loop (_subtraction_recovery_round_loop, _late_low_carrier_-
# rescue) and the trio/pairwise seed-trim (in block_haplotypes.py's
# _grow_K_with_recovery) each do hundreds-to-thousands of `_compute_nll_-
# for_subset` calls over subsets drawn from a FIXED candidate pool.  Each
# call independently rebuilds the (N, K_states, n_bins) binned-emission
# tensor via _viterbi_binned_emissions_kernel.  Since the emission for
# state (i, j) depends ONLY on the haps H[i], H[j] and the data probs_k
# — NOT on what other states are in the subset — the rebuild is mostly
# redundant work.
#
# PoolEmissionCache precomputes the full pool emission tensor ONCE per
# search session.  Each `nll_for_subset` call then:
#   1. Maps the subset's state slots back to the pool's state slots via
#      a tiny index map (njit kernel).
#   2. Fancy-indexes the pool tensor along the state axis to assemble
#      the subset's (N, K_sub_states, n_bins) tensor.
#   3. Runs viterbi_score_selection on the assembled subset tensor.
#
# Memory: O(N * n_states_pool * n_bins).  For pool size K_pool=25, N=320,
# n_bins=20: n_states_pool = K_pool*(K_pool+1)/2 + K_pool + 1 = 351
# states; tensor size 320*351*20*8 = ~18 MB.  Bounded by typical pool
# sizes (≤ 30) so cache stays under 30 MB.
#
# Lifetime: one BIC-search session (one recovery round, one late-rescue
# invocation, or one seed-trim).  Built, queried thousands of times,
# discarded.
#
# Bit-exact equivalence: the cache uses the same _viterbi_binned_-
# emissions_kernel as the fresh build, so the pool emission tensor is
# bit-identical to what fresh-build-on-subset would produce at the same
# slot positions (modulo permutation, since the subset's state layout
# is a re-ordering of the corresponding pool slots).  viterbi_score_-
# selection's output is invariant under state-slot permutation (the
# best-path NLL doesn't depend on which integer label the states carry),
# so cache.nll_for_subset(indices) == _compute_nll_for_subset(
# [pool[i] for i in indices], ...) bit-identically.  Verified at machine
# precision against the legacy path.

@njit(cache=True)
def _build_pool_state_index_map_kernel(subset_indices, K_pool):
    """Build the (K_sub_states,) array mapping each subset state slot
    to the corresponding slot in the pool's collapsed-state-space layout.

    Subset state layout (same as _viterbi_binned_emissions_kernel post-
    collapse, but over the K_sub = len(subset_indices) selected pool
    members):
        [0 .. K_sub*(K_sub+1)/2)        unordered (a, b) with a <= b
                                          (positions into subset_indices)
        [.. + K_sub)                     (k_sub, W) for k_sub in [0, K_sub)
        [.. + 1]                         (W, W) — single slot

    For each subset state slot, we compute the corresponding pool state
    slot.  Pool state layout (same convention, but for the K_pool pool):
        rr pairs at index `lo * K_pool - lo*(lo-1)/2 + (hi - lo)` for
            lo <= hi
        kW pairs at index `n_rr_pool + k`
        WW   at index `n_rr_pool + K_pool`

    For a subset (a, b) with a <= b in the subset's index space, the
    pool pair is (subset_indices[a], subset_indices[b]); we re-order to
    (lo, hi) with lo = min, hi = max before looking up.  Subset (k_sub,
    W) maps to pool (subset_indices[k_sub], W).  (W, W) maps to (W, W).

    Arguments:
        subset_indices: (K_sub,) int64 — pool indices in any order,
                        no duplicates required by us (caller's contract)
        K_pool: int — total pool size

    Returns:
        (K_sub_states,) int64 — pool slot index per subset slot, in the
        subset's state-slot order.
    """
    K_sub = subset_indices.shape[0]
    n_rr_pool = K_pool * (K_pool + 1) // 2
    n_rr_sub = K_sub * (K_sub + 1) // 2
    K_sub_states = n_rr_sub + K_sub + 1

    out = np.empty(K_sub_states, dtype=np.int64)
    slot = 0
    # Real-real (a, b) with a <= b in the subset's index space
    for a in range(K_sub):
        pi_a = subset_indices[a]
        for b in range(a, K_sub):
            pi_b = subset_indices[b]
            if pi_a <= pi_b:
                lo = pi_a
                hi = pi_b
            else:
                lo = pi_b
                hi = pi_a
            out[slot] = lo * K_pool - lo * (lo - 1) // 2 + (hi - lo)
            slot += 1
    # Real-W (k_sub, W) — each pool member's (k, W) slot
    for k_sub in range(K_sub):
        out[slot] = n_rr_pool + subset_indices[k_sub]
        slot += 1
    # (W, W) — single state, last pool slot
    out[slot] = n_rr_pool + K_pool

    return out


@njit(cache=True, parallel=True, fastmath=False)
def _viterbi_partial_binned_emissions_kernel(
        H_pool,
        rr_pair_i, rr_pair_j, rr_state_slots,
        kW_pair_k, kW_state_slots,
        WW_state_slot, compute_WW,
        snps_per_bin, n_bins, lam,
        log_probs,
        out_tensor):
    """Fill specified state-slots in `out_tensor` with their Viterbi
    binned emissions, leaving all other slots untouched.

    This is the incremental-build companion to
    `_viterbi_binned_emissions_kernel`.  The full kernel builds every
    state in the pool's collapsed layout; this one builds ONLY the
    state slots whose corresponding pair of pool members is new (or
    has changed) relative to a previous pool.  The caller (typically
    `PoolEmissionCache.__init__` with prev_cache supplied) is
    responsible for:
      - pre-allocating `out_tensor` at (N, K_states_new, n_bins),
      - filling already-reusable rows with `prev_cache._pool_tensor`'s
        corresponding rows (canonical (lo, hi) → row mapping),
      - passing the COMPLEMENT of those rows (i.e., the slots that
        still need fresh emission values) to this kernel.

    Bit-equivalence with `_viterbi_binned_emissions_kernel`:
      - rr state (i, j): identical per-l accumulator on the same
        scalars (log_probs[s, l, d] with d = H[i,l] + H[j,l]),
        summed in the same left-to-right order within each bin.
      - kW state (k): identical per-l accumulator on
        max(log_probs[s, l, d0], log_probs[s, l, d1]) - lam, summed
        in the same order.
      - WW state: identical per-l accumulator on
        max(log_probs[s, l, 0..2]) - 2*lam.

    Uses log_probs (precomputed via _log_probs_kernel) in place of
    inline math.log calls.  By monotonicity of log over positives,
    max(log_probs[..., d0..d1]) = log(max(probs[..., d0..d1], EPS))
    after the LOG_EPS clamp baked into log_probs.  Same scalars in
    the same accumulation order yield bit-identical output to the
    legacy full-build kernel.

    Arguments:
        H_pool:             (K_pool_new, L) int64 — current pool's
                             haps (full set; only rows referenced by
                             rr_pair_* / kW_pair_k are read).
        rr_pair_i:          (n_rr_miss,) int64 — first pool index of
                             each rr pair to compute (i in 0..K_pool).
        rr_pair_j:          (n_rr_miss,) int64 — second pool index of
                             each rr pair to compute, with i <= j.
        rr_state_slots:     (n_rr_miss,) int64 — destination state-
                             slot in out_tensor for each rr pair.
        kW_pair_k:          (n_kW_miss,) int64 — pool index of each
                             (k, W) state to compute.
        kW_state_slots:     (n_kW_miss,) int64 — destination state-
                             slot in out_tensor for each kW state.
        WW_state_slot:      int — destination slot for (W, W) state
                             (only used if compute_WW=True).
        compute_WW:         bool — whether to (re)compute the (W, W)
                             state.  False when the caller copied it
                             from prev_cache.
        snps_per_bin:       bin size in SNPs.
        n_bins:             total bin count.
        lam:                wildcard penalty per strand-site.
        log_probs:          (N, L, 3) float64, C-contig — precomputed
                             log(max(probs[s, l, g], LOG_EPS_LOCAL))
                             (from _log_probs_kernel).
        out_tensor:         (N, K_states_new, n_bins) float64, C-contig
                             — destination buffer.  This kernel writes
                             ONLY to the specified slots; other slots
                             must already contain correct values
                             (typically copied from prev_cache).

    Returns:
        out_tensor (the same buffer that was passed in).
    """
    N = log_probs.shape[0]
    L = log_probs.shape[1]

    n_rr_miss = rr_pair_i.shape[0]
    n_kW_miss = kW_pair_k.shape[0]

    for s in prange(N):
        # ----- rr pairs -----
        for slot_iter in range(n_rr_miss):
            i = rr_pair_i[slot_iter]
            j = rr_pair_j[slot_iter]
            out_slot = rr_state_slots[slot_iter]
            for b in range(n_bins):
                start = b * snps_per_bin
                end = start + snps_per_bin
                if end > L:
                    end = L
                acc = 0.0
                for l in range(start, end):
                    d = H_pool[i, l] + H_pool[j, l]
                    acc += log_probs[s, l, d]
                out_tensor[s, out_slot, b] = acc

        # ----- kW pairs -----
        for slot_iter in range(n_kW_miss):
            k = kW_pair_k[slot_iter]
            out_slot = kW_state_slots[slot_iter]
            for b in range(n_bins):
                start = b * snps_per_bin
                end = start + snps_per_bin
                if end > L:
                    end = L
                acc = 0.0
                for l in range(start, end):
                    d0 = H_pool[k, l]
                    d1 = d0 + 1
                    lp0 = log_probs[s, l, d0]
                    lp1 = log_probs[s, l, d1]
                    lp_max = lp0 if lp0 > lp1 else lp1
                    acc += lp_max - lam
                out_tensor[s, out_slot, b] = acc

        # ----- (W, W) state, if requested -----
        if compute_WW:
            for b in range(n_bins):
                start = b * snps_per_bin
                end = start + snps_per_bin
                if end > L:
                    end = L
                acc = 0.0
                for l in range(start, end):
                    lp0 = log_probs[s, l, 0]
                    lp1 = log_probs[s, l, 1]
                    lp2 = log_probs[s, l, 2]
                    lp_max = lp0
                    if lp1 > lp_max:
                        lp_max = lp1
                    if lp2 > lp_max:
                        lp_max = lp2
                    acc += lp_max - 2.0 * lam
                out_tensor[s, WW_state_slot, b] = acc

    return out_tensor


@njit(cache=True, parallel=True, fastmath=False)
def _viterbi_subset_from_pool_kernel(pool_tensor, pool_state_indices, penalty):
    """Flat-penalty Viterbi best-path score per sample, scoring ONLY the
    subset of states named by `pool_state_indices`, read in place from
    the full pool emission tensor.

    Bit-identical to (and a drop-in for) the legacy query path::

        sub_tensor = np.ascontiguousarray(
            pool_tensor[:, pool_state_indices, :])
        best_ll    = viterbi_score_selection(sub_tensor, penalty)

    but without materialising `sub_tensor`.  The fancy-index gather plus
    its ascontiguousarray copy were the dominant cost of nll_for_subset
    (each subset query rebuilt a (N, K_sub_states, n_bins) tensor); the
    Viterbi DP itself is cheap.  This kernel folds the state selection
    into the DP's emission reads: wherever viterbi_score_selection reads
    ll_tensor[s, k, i], this reads pool_tensor[s, pool_state_indices[k], i]
    — the SAME scalar by construction of the gather — and runs the
    identical recurrence in the identical left-to-right order, so the
    returned best-path scores match the legacy path to the last bit.

    Arguments:
        pool_tensor:        (N, K_pool_states, n_bins) float64 — the
                            cache's full emission tensor.
        pool_state_indices: (K_sub_states,) int64 — state-axis indices
                            (into pool_tensor) of the subset's states,
                            as produced by _build_pool_state_index_map_kernel.
        penalty:            flat switch penalty (float).

    Returns:
        (N,) float64 best-path score per sample (NLL = -sum).
    """
    n_samples = pool_tensor.shape[0]
    K = pool_state_indices.shape[0]
    n_sites = pool_tensor.shape[2]
    best_scores = np.empty(n_samples, dtype=np.float64)

    for s in prange(n_samples):
        # Buffer for current scores (faster than allocation in loop)
        current_scores = np.empty(K, dtype=np.float64)
        for k in range(K):
            current_scores[k] = pool_tensor[s, pool_state_indices[k], 0]

        for i in range(1, n_sites):
            # 1. Find best previous score globally
            best_prev = -np.inf
            for k in range(K):
                if current_scores[k] > best_prev:
                    best_prev = current_scores[k]

            # The baseline score if we switch INTO a state
            switch_base = best_prev - penalty

            # 2. Update states
            for k in range(K):
                emission = pool_tensor[s, pool_state_indices[k], i]
                stay = current_scores[k]

                # Max(Stay, Switch)
                if stay > switch_base:
                    current_scores[k] = stay + emission
                else:
                    current_scores[k] = switch_base + emission

        # Final max
        final_max = -np.inf
        for k in range(K):
            if current_scores[k] > final_max:
                final_max = current_scores[k]
        best_scores[s] = final_max

    return best_scores


@njit(cache=True, parallel=True, fastmath=False)
def _batch_subset_scores_kernel(pool_tensor, subsets, K_pool, penalty):
    """Batched _viterbi_subset_from_pool_kernel: score MANY equal-size
    subsets at once, parallelised across candidates.

    `prange` is over the candidate axis (the outer loop), with the
    per-sample Viterbi DP run SERIALLY inside each candidate (no nested
    prange).  This is the win on pathological large-pool blocks where the
    greedy / swap selection evaluates many candidate subsets: instead of a
    sequential Python loop launching one tiny per-sample-parallel kernel per
    candidate (Python + dispatch overhead dominating, few cores busy), all
    candidates run concurrently here.

    Each candidate inlines _build_pool_state_index_map_kernel followed by the
    exact _viterbi_subset_from_pool_kernel recurrence, reading the SAME
    pool_tensor scalars in the SAME left-to-right order, so out[c, s] equals
    the best-path score _viterbi_subset_from_pool_kernel would return for
    subset c, sample s — to the last bit.

    Arguments:
        pool_tensor: (N, K_pool_states, n_bins) float64 — the cache's full
            emission tensor.
        subsets:     (n_cand, K_sub) int64 — each row a subset of pool
            indices (K_sub >= 1; rows need not be sorted, matching
            _build_pool_state_index_map_kernel's contract).
        K_pool:      int — total pool size.
        penalty:     flat switch penalty (float).

    Returns:
        (n_cand, N) float64 best-path score per (candidate, sample).  The
        caller forms NLL per candidate as -out.sum(axis=1), reusing the same
        numpy pairwise reduction as nll_for_subset.
    """
    n_cand = subsets.shape[0]
    K_sub = subsets.shape[1]
    N = pool_tensor.shape[0]
    n_sites = pool_tensor.shape[2]
    n_rr_pool = K_pool * (K_pool + 1) // 2
    n_rr_sub = K_sub * (K_sub + 1) // 2
    K_sub_states = n_rr_sub + K_sub + 1

    out = np.empty((n_cand, N), dtype=np.float64)

    for c in prange(n_cand):
        # --- build this subset's pool-state-index map (inline of
        # _build_pool_state_index_map_kernel) ---
        sidx = subsets[c]
        psi = np.empty(K_sub_states, dtype=np.int64)
        slot = 0
        for a in range(K_sub):
            pa = sidx[a]
            for b in range(a, K_sub):
                pb = sidx[b]
                if pa <= pb:
                    lo = pa
                    hi = pb
                else:
                    lo = pb
                    hi = pa
                psi[slot] = lo * K_pool - lo * (lo - 1) // 2 + (hi - lo)
                slot += 1
        for k_sub in range(K_sub):
            psi[slot] = n_rr_pool + sidx[k_sub]
            slot += 1
        psi[slot] = n_rr_pool + K_pool

        # --- Viterbi DP over samples, SERIAL here (inline of
        # _viterbi_subset_from_pool_kernel with range instead of prange) ---
        for s in range(N):
            current = np.empty(K_sub_states, dtype=np.float64)
            for k in range(K_sub_states):
                current[k] = pool_tensor[s, psi[k], 0]
            for i in range(1, n_sites):
                best_prev = -np.inf
                for k in range(K_sub_states):
                    if current[k] > best_prev:
                        best_prev = current[k]
                switch_base = best_prev - penalty
                for k in range(K_sub_states):
                    emission = pool_tensor[s, psi[k], i]
                    stay = current[k]
                    if stay > switch_base:
                        current[k] = stay + emission
                    else:
                        current[k] = switch_base + emission
            final_max = -np.inf
            for k in range(K_sub_states):
                if current[k] > final_max:
                    final_max = current[k]
            out[c, s] = final_max

    return out


@njit(cache=True, parallel=True, fastmath=True)
def _fused_subset_scores_kernel(pool_haps, log_probs, subsets,
                                snps_per_bin, n_bins, lam, penalty):
    """Fused, log-free per-subset scorer for the memory-guard fallback path.

    Companion to _batch_subset_scores_kernel for the regime where the
    full-pool emission tensor was too large to materialise (the MEMORY GUARD
    block in PoolEmissionCache.__init__).  Where the tensor path amortises a
    one-off emission build across many queries, the fallback has no reuse, so
    this kernel folds emission and Viterbi into a single pass per candidate:
    no (N, K_states, n_bins) intermediate is materialised (only a length-
    K_states running-score buffer per sample), and the per-(state, bin)
    emissions are read from the precomputed `log_probs` instead of calling
    math.log inline at every (state, site) visit -- the same Tier-0 win
    _log_probs_kernel gives _update_A, whose emission kernel was log()-bound.

    The candidate axis is the prange, so greedy / swap selection on a huge
    low-depth pool scores all candidates across cores at once.

    Emission semantics are identical to _viterbi_binned_emissions_kernel's
    collapsed state space -- unordered real-real pairs (i <= j), then (k, W)
    real-wildcard states, then a single (W, W) state -- and the recurrence is
    viterbi_score_selection's flat-penalty Viterbi, so out[c] matches
    _viterbi_nll([pool_haps[i] for i in row], probs, ...) to within a couple
    of ULP: log_probs makes the emissions bit-identical (log is monotonic, so
    max-of-probs-then-log == max-of-log-probs, same 1e-12 clamp), and only
    fastmath reassociation of the bin sums moves the last bit.  State order is
    irrelevant to the best-path value under a flat penalty, so the collapse is
    purely a state-count saving.

    Arguments:
        pool_haps:  (K_pool, L) int64 -- the pool, one 0/1 hap per row.
        log_probs:  (N, L, 3) float64 -- log(max(probs[s, l, g], 1e-12)) from
            _log_probs_kernel; invariant across queries on this cache.
        subsets:    (n_cand, K_sub) int64 -- each row a subset of pool row
            indices (K_sub >= 1; rows need not be sorted).
        snps_per_bin, n_bins, lam, penalty: the cache's stored scoring params.

    Returns:
        (n_cand, N) float64 best-path score per (candidate, sample).  The
        caller forms NLL as -out.sum(axis=1), reusing the same numpy pairwise
        reduction as nll_for_subset.
    """
    n_cand = subsets.shape[0]
    K_sub = subsets.shape[1]
    L = pool_haps.shape[1]
    N = log_probs.shape[0]
    n_rr = K_sub * (K_sub + 1) // 2
    K_states = n_rr + K_sub + 1
    out = np.empty((n_cand, N), dtype=np.float64)
    for c in prange(n_cand):
        sidx = subsets[c]
        current = np.empty(K_states, dtype=np.float64)   # running Viterbi scores
        emis = np.empty(K_states, dtype=np.float64)      # this bin's emissions
        for s in range(N):
            for b in range(n_bins):
                start = b * snps_per_bin
                end = start + snps_per_bin
                if end > L:
                    end = L
                # real-real pairs (i <= j): sum_l log P(dosage H[i]+H[j])
                p = 0
                for i in range(K_sub):
                    hi = sidx[i]
                    for j in range(i, K_sub):
                        hj = sidx[j]
                        acc = 0.0
                        for l in range(start, end):
                            acc += log_probs[s, l, pool_haps[hi, l] + pool_haps[hj, l]]
                        emis[p] = acc
                        p += 1
                # real-wildcard (k, W): wildcard strand picks the better allele
                for k in range(K_sub):
                    hk = sidx[k]
                    acc = 0.0
                    for l in range(start, end):
                        a = log_probs[s, l, pool_haps[hk, l]]
                        bb = log_probs[s, l, pool_haps[hk, l] + 1]
                        acc += (a if a > bb else bb) - lam
                    emis[p] = acc
                    p += 1
                # (W, W): both strands wildcard
                acc = 0.0
                for l in range(start, end):
                    a0 = log_probs[s, l, 0]
                    a1 = log_probs[s, l, 1]
                    a2 = log_probs[s, l, 2]
                    m = a0
                    if a1 > m:
                        m = a1
                    if a2 > m:
                        m = a2
                    acc += m - 2.0 * lam
                emis[p] = acc
                # fold this bin into the flat-penalty Viterbi recurrence
                if b == 0:
                    for st in range(K_states):
                        current[st] = emis[st]
                else:
                    best_prev = current[0]
                    for st in range(1, K_states):
                        if current[st] > best_prev:
                            best_prev = current[st]
                    switch_base = best_prev - penalty
                    for st in range(K_states):
                        stay = current[st]
                        if stay < switch_base:
                            stay = switch_base
                        current[st] = stay + emis[st]
            fm = current[0]
            for st in range(1, K_states):
                if current[st] > fm:
                    fm = current[st]
            out[c, s] = fm
    return out


class PoolEmissionCache:
    """Precomputed Viterbi-emission tensor for a fixed hap pool.

    Lets the recovery loop and seed-trim evaluate per-subset Viterbi
    NLL without rebuilding the (N, K_states, n_bins) emission tensor
    on each call.  See module-header comment block (POOL EMISSION
    CACHE) for design rationale and bit-exact equivalence argument.

    Usage:
        cache = PoolEmissionCache(pool_haps, probs_k,
                                   lam=DEFAULT_LAMBDA,
                                   penalty=VITERBI_SWITCH_PENALTY,
                                   snps_per_bin=VITERBI_SNPS_PER_BIN)
        nll0 = cache.nll_for_subset([])          # K=0 baseline
        nll  = cache.nll_for_subset([0, 3, 7])   # by pool indices

    Public attributes (read-only after construction):
        pool_haps:    list of (L_kept,) int64 arrays — original pool
        K_pool:       len(pool_haps)
        N, L_kept:    probs_k.shape[0], probs_k.shape[1]
        lam, penalty, snps_per_bin, n_bins
        n_rr_pool:    K_pool*(K_pool+1)//2
    """

    def __init__(self, pool_haps, probs_k,
                  lam=None, penalty=None, snps_per_bin=None,
                  prev_cache=None):
        # Default to module-level constants (matching _viterbi_ll_per_-
        # sample's parameter semantics) when callers omit them.
        if lam is None:
            lam = DEFAULT_LAMBDA
        if penalty is None:
            penalty = VITERBI_SWITCH_PENALTY
        if snps_per_bin is None:
            snps_per_bin = VITERBI_SNPS_PER_BIN

        # Normalise pool_haps to a list (callers pass lists today; this
        # also accepts numpy 2D arrays or tuples for flexibility).
        if not isinstance(pool_haps, list):
            pool_haps = list(pool_haps)
        self.pool_haps = pool_haps
        self.K_pool = len(pool_haps)

        self.N = int(probs_k.shape[0])
        self.L_kept = int(probs_k.shape[1])
        self.lam = float(lam)
        self.penalty = float(penalty)

        # Bin sizing (same logic as _viterbi_ll_per_sample).  snps_per_-
        # bin == 1 collapses to per-site bins (no binning); otherwise
        # ceil(L / snps_per_bin) bins covering the SNPs.
        L = self.L_kept
        if snps_per_bin > 1 and snps_per_bin < L:
            n_bins = int(math.ceil(L / snps_per_bin))
        else:
            n_bins = L
            snps_per_bin = 1
        self.snps_per_bin = int(snps_per_bin)
        self.n_bins = int(n_bins)

        self.n_rr_pool = self.K_pool * (self.K_pool + 1) // 2

        # Memory-guard fallback flag.  Set True by the MEMORY GUARD block
        # below when the full-pool emission tensor would exceed
        # POOL_EMISSION_CACHE_MAX_BYTES, in which case nll_for_subset scores
        # each subset on the fly instead of indexing a precomputed tensor.
        # Initialised here so every constructor exit path — including the
        # K_pool == 0 early return — leaves it defined for nll_for_subset.
        self._fallback = False

        # Store reference to probs_k for the incremental-build path's
        # compatibility check.  We use object identity (`is`) below to
        # decide if a `prev_cache` argument is compatible — this is
        # cheap and unambiguous when the recovery loop is invoking the
        # same probs_k object across rounds (the typical pattern).
        self._probs_k_ref = probs_k

        # K=0 NLL is independent of pool — it's just the (W, W)-only
        # path's NLL on probs_k.  Compute via _per_site_cost_W_W to
        # match the legacy _compute_nll_for_subset([], ...) value
        # exactly bit-for-bit (the alternative — summing the cache's
        # WW slot over bins — would sum the same scalars in a slightly
        # different order, risking tiny float differences).
        cost_WW_per_site = _per_site_cost_W_W(probs_k, lam)
        self._nll_K0 = float(cost_WW_per_site.sum())

        # Per-pool-hap content keys (bytes of the hap's int64 buffer).
        # Used by the incremental-build path to match haps across
        # successive caches by content (not by object identity, since
        # `_fit_at_fixed_K` produces fresh arrays each round even when
        # the bits are unchanged).  Cheap: K_pool * tobytes() ≤ ~30 µs
        # at K_pool=20, L=200.
        self._hap_keys = [h.tobytes() for h in pool_haps]

        if self.K_pool == 0:
            # Empty pool: only K=0 subsets queryable.  No tensor needed.
            self._pool_tensor = None
            self._log_probs = None
            return

        # ---------------------------------------------------------------
        # MEMORY GUARD: fall back to on-the-fly scoring for huge pools.
        # The full-pool emission tensor built below is O(K_pool^2) in its
        # state axis (K_states = n_rr_pool + K_pool + 1).  At normal read
        # depth the candidate pool is small (<= ~30) and this tensor is a
        # ~30 MB speed win, but at low read depth the trio/pairwise pool can
        # balloon to thousands and the tensor then runs to hundreds of GiB
        # (e.g. K_pool=3398, N=320, n_bins=20 -> 276 GiB), OOM-killing the
        # worker.  When the tensor would exceed POOL_EMISSION_CACHE_MAX_BYTES
        # we skip building it and set self._fallback; nll_for_subset then
        # scores each subset directly via _compute_nll_for_subset, which the
        # class-header equivalence argument certifies is bit-identical to the
        # cached path (every construction site uses the default penalty /
        # snps_per_bin that _compute_nll_for_subset also uses; verified for
        # all call sites).  This reproduces the pre-cache behaviour — slower
        # per query, bounded memory — so it changes run time, never results.
        # _pool_tensor / _log_probs are left None so a later cache's
        # can_reuse_prev check (which requires prev._pool_tensor is not None)
        # correctly declines to reuse a fallback cache as prev_cache.
        # ---------------------------------------------------------------
        pool_tensor_bytes = (
            self.N * (self.n_rr_pool + self.K_pool + 1) * self.n_bins * 8
        )
        if pool_tensor_bytes > POOL_EMISSION_CACHE_MAX_BYTES:
            self._fallback = True
            self._pool_tensor = None
            self._log_probs = None
            # Inputs for the parallel-over-candidates fallback scorer
            # (_fused_subset_scores_kernel): stack the pool haps and
            # precompute log(max(probs, 1e-12)) once here — the kernel is
            # log-free — so batch_nll_for_subsets need not rebuild either on
            # every call.
            self._pool_haps_arr = np.stack(pool_haps, axis=0).astype(np.int64)
            self._log_probs_c = _log_probs_kernel(
                np.ascontiguousarray(probs_k, dtype=np.float64))
            return

        # ---------------------------------------------------------------
        # log_probs precompute (Tier 0 sharing across cache builds).
        # If a compatible prev_cache provided its log_probs (same
        # probs_k object), reuse it directly.  Otherwise compute fresh.
        # The fresh compute is ~1-2 ms at N=320 L=200, but avoiding it
        # across the 7-ish round_cache builds per block is a clean
        # marginal saving.
        # ---------------------------------------------------------------
        can_reuse_prev = (
            prev_cache is not None
            and prev_cache._pool_tensor is not None
            and prev_cache._probs_k_ref is probs_k
            and prev_cache.lam == self.lam
            and prev_cache.snps_per_bin == self.snps_per_bin
            and prev_cache.n_bins == self.n_bins
        )

        if can_reuse_prev and prev_cache._log_probs is not None:
            log_probs = prev_cache._log_probs
        else:
            probs_c = np.ascontiguousarray(probs_k, dtype=np.float64)
            log_probs = _log_probs_kernel(probs_c)
        self._log_probs = log_probs

        # ---------------------------------------------------------------
        # Build the pool emission tensor.  Two paths:
        #   (a) No prev_cache (or incompatible): full build via the
        #       partial kernel with ALL slots marked missing.
        #   (b) Compatible prev_cache: identify state slots whose pool-
        #       member pair matches by content in prev_cache (via hap
        #       keys), pre-fill those slots from prev's tensor, and
        #       compute only the truly missing rows via the partial
        #       kernel.
        #
        # Bit-equivalence with the legacy full-build path:
        #   - The partial kernel's per-slot accumulator is identical
        #     to the legacy _viterbi_binned_emissions_kernel's (same
        #     scalars from log_probs[s, l, d], same accumulation
        #     order within each bin).  See the partial kernel's
        #     docstring for the monotonicity-of-log argument used to
        #     show max(log_probs[..., d0..d1]) = log(max(probs[..., d0..d1], EPS)).
        #   - Rows COPIED from prev_cache were themselves produced by
        #     the same kernel against the same probs_k / lam / binning
        #     in the prev cache's __init__, so they are bit-identical
        #     to what a fresh recompute would produce.
        # ---------------------------------------------------------------

        K_states = self.n_rr_pool + self.K_pool + 1
        out_tensor = np.empty((self.N, K_states, self.n_bins), dtype=np.float64)

        # Decide which slots are reusable from prev_cache.
        rr_pair_i_list = []
        rr_pair_j_list = []
        rr_state_slots_list = []
        kW_pair_k_list = []
        kW_state_slots_list = []
        compute_WW = True

        if can_reuse_prev:
            # Build a {hap_key -> prev_pool_idx} map for O(1) lookup.
            prev_key_to_idx = {}
            for prev_idx, key in enumerate(prev_cache._hap_keys):
                # If duplicates exist in prev's pool (shouldn't, but
                # the cache doesn't enforce uniqueness), keep the
                # FIRST occurrence — its row is canonical.
                if key not in prev_key_to_idx:
                    prev_key_to_idx[key] = prev_idx
            K_prev = prev_cache.K_pool
            prev_tensor = prev_cache._pool_tensor

            # rr state (i, j) for i <= j in NEW pool's row-major order
            new_slot = 0
            for i in range(self.K_pool):
                key_i = self._hap_keys[i]
                prev_i = prev_key_to_idx.get(key_i, -1)
                for j in range(i, self.K_pool):
                    key_j = self._hap_keys[j]
                    prev_j = prev_key_to_idx.get(key_j, -1)
                    if prev_i >= 0 and prev_j >= 0:
                        # Both haps existed in prev — copy the row.
                        # In prev's layout, the rr slot for (a, b)
                        # with a <= b is `a * K_prev - a*(a-1)/2 + (b - a)`.
                        if prev_i <= prev_j:
                            pa, pb = prev_i, prev_j
                        else:
                            pa, pb = prev_j, prev_i
                        prev_slot = pa * K_prev - pa * (pa - 1) // 2 + (pb - pa)
                        out_tensor[:, new_slot, :] = prev_tensor[:, prev_slot, :]
                    else:
                        # At least one hap is new — compute fresh.
                        rr_pair_i_list.append(i)
                        rr_pair_j_list.append(j)
                        rr_state_slots_list.append(new_slot)
                    new_slot += 1

            # kW state (k, W) — reusable iff hap k matched in prev
            for k in range(self.K_pool):
                key_k = self._hap_keys[k]
                prev_k = prev_key_to_idx.get(key_k, -1)
                if prev_k >= 0:
                    prev_slot = prev_cache.n_rr_pool + prev_k
                    out_tensor[:, new_slot, :] = prev_tensor[:, prev_slot, :]
                else:
                    kW_pair_k_list.append(k)
                    kW_state_slots_list.append(new_slot)
                new_slot += 1

            # (W, W) state — depends only on probs_k and lam, which we
            # already verified match prev.  Always reusable.
            WW_slot_new = self.n_rr_pool + self.K_pool
            WW_slot_prev = prev_cache.n_rr_pool + prev_cache.K_pool
            out_tensor[:, WW_slot_new, :] = prev_tensor[:, WW_slot_prev, :]
            compute_WW = False
        else:
            # No reusable rows — every slot needs computing.
            new_slot = 0
            for i in range(self.K_pool):
                for j in range(i, self.K_pool):
                    rr_pair_i_list.append(i)
                    rr_pair_j_list.append(j)
                    rr_state_slots_list.append(new_slot)
                    new_slot += 1
            for k in range(self.K_pool):
                kW_pair_k_list.append(k)
                kW_state_slots_list.append(new_slot)
                new_slot += 1
            # compute_WW = True (default)

        # Build the H_pool array once for the partial kernel.
        H_pool_arr = np.stack(self.pool_haps, axis=0).astype(np.int64)
        H_c = np.ascontiguousarray(H_pool_arr, dtype=np.int64)

        rr_pair_i_arr = np.asarray(rr_pair_i_list, dtype=np.int64)
        rr_pair_j_arr = np.asarray(rr_pair_j_list, dtype=np.int64)
        rr_state_slots_arr = np.asarray(rr_state_slots_list, dtype=np.int64)
        kW_pair_k_arr = np.asarray(kW_pair_k_list, dtype=np.int64)
        kW_state_slots_arr = np.asarray(kW_state_slots_list, dtype=np.int64)
        WW_slot = self.n_rr_pool + self.K_pool

        # Run the partial kernel.  If all the *_list arrays are empty
        # AND compute_WW is False (everything reused), the kernel will
        # walk N samples doing nothing — still cheap (~10 µs for N=320),
        # so no special-cased early-return needed.
        _viterbi_partial_binned_emissions_kernel(
            H_c,
            rr_pair_i_arr, rr_pair_j_arr, rr_state_slots_arr,
            kW_pair_k_arr, kW_state_slots_arr,
            int(WW_slot), bool(compute_WW),
            int(self.snps_per_bin), int(self.n_bins), float(self.lam),
            log_probs,
            out_tensor)

        self._pool_tensor = out_tensor

    def nll_for_subset(self, subset_indices):
        """Return scalar UNCAPPED NLL for the subset given by pool indices.

        Bit-identical to:
            _compute_nll_for_subset(
                [self.pool_haps[i] for i in subset_indices],
                probs_k_used_at_construction,
                self.lam)
        but reuses the precomputed pool emission tensor.

        Arguments:
            subset_indices: sequence of ints, each an index into
                self.pool_haps.  May be empty (returns K=0 NLL).
                Order doesn't affect the result (the Viterbi best-path
                value is invariant under state-slot permutation under
                flat penalty).
        """
        K_sub = len(subset_indices)
        if K_sub == 0:
            # K=0 path: only state is (W, W); return precomputed value
            # for bit-identity with legacy _compute_nll_for_subset([], ...).
            return self._nll_K0

        if self._fallback:
            # Memory-guard fallback (see the MEMORY GUARD block in __init__):
            # the full-pool emission tensor was too large to materialise, so
            # score this subset directly.  Bit-identical to the cached path
            # per the class-header equivalence argument — every construction
            # site uses the default penalty / snps_per_bin that
            # _compute_nll_for_subset also uses — so only run time differs.
            return _compute_nll_for_subset(
                [self.pool_haps[i] for i in subset_indices],
                self._probs_k_ref, self.lam)

        if self._pool_tensor is None:
            raise ValueError(
                "PoolEmissionCache was built with empty pool; only "
                "K=0 subsets are queryable.")

        # Build pool-state-index map (njit kernel for the small inner
        # loops; the K_sub_states=O(K_sub^2) iterations are cheap but
        # called often enough that staying in compiled code matters).
        si = np.asarray(subset_indices, dtype=np.int64)
        pool_state_indices = _build_pool_state_index_map_kernel(si, self.K_pool)

        # Score the subset's states directly from the pool tensor via the
        # state-index map, with NO intermediate (N, K_sub_states, n_bins)
        # tensor.  The legacy path fancy-indexed
        # self._pool_tensor[:, pool_state_indices, :] and copied it to a
        # contiguous sub_tensor for viterbi_score_selection; that gather +
        # ascontiguousarray copy was the dominant per-query cost (the
        # Viterbi DP itself is cheap).  _viterbi_subset_from_pool_kernel
        # folds the state selection into the DP's emission reads and is
        # bit-identical to viterbi_score_selection(sub_tensor, penalty) —
        # it reads the SAME scalars in the SAME recurrence order.  Returns
        # (N,) of best-path LL per sample.  NLL = -sum.
        best_ll = _viterbi_subset_from_pool_kernel(
            self._pool_tensor, pool_state_indices, self.penalty)
        return -float(best_ll.sum())

    def batch_nll_for_subsets(self, subsets):
        """Vectorised nll_for_subset over many EQUAL-size subsets at once.

        Arguments:
            subsets: (n_cand, K_sub) int64 — each row a subset of pool
                indices (K_sub >= 1).  Rows need not be sorted.

        Returns:
            (n_cand,) float64 — NLL per row, each bit-identical to
            nll_for_subset(subsets[c]): the per-candidate emission recurrence
            is identical, and the per-sample sum reuses the same numpy
            pairwise reduction (-scores.sum(axis=1)).

        Parallelised across candidates (prange in _batch_subset_scores_kernel)
        rather than per-candidate-over-samples — the win on pathological
        large-pool blocks where there are many candidates but each one's
        Viterbi is tiny.  In the memory-guard / empty-pool paths the tensor is
        absent, so scoring goes through _fused_subset_scores_kernel instead,
        which keeps the same candidate-parallel shape (and the same per-
        candidate result, to a couple of ULP) while fusing emissions into the
        Viterbi in one log-free pass; empty subsets short-circuit to the
        precomputed K=0 NLL.
        """
        subsets = np.ascontiguousarray(subsets, dtype=np.int64)
        n_cand = subsets.shape[0]
        if n_cand == 0:
            return np.empty(0, dtype=np.float64)
        if self._fallback or self._pool_tensor is None:
            # Memory-guard / empty-pool path.  Empty subsets (K=0) are pool-
            # independent — return the precomputed (W, W) NLL.  Otherwise the
            # parallel fallback kernel scores every candidate across cores and
            # we reduce with the same -scores.sum(axis=1) as the tensor path.
            if subsets.shape[1] == 0:
                return np.full(n_cand, self._nll_K0, dtype=np.float64)
            scores = _fused_subset_scores_kernel(
                self._pool_haps_arr, self._log_probs_c, subsets,
                self.snps_per_bin, self.n_bins, self.lam, self.penalty)
            return -scores.sum(axis=1)
        scores = _batch_subset_scores_kernel(
            self._pool_tensor, subsets, self.K_pool, self.penalty)
        return -scores.sum(axis=1)