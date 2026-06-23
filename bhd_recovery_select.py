# bhd_recovery_select.py - BIC subset selection + hap-dedup toolkit.
#
# Given a pool of candidate haplotypes, selects/prunes/deduplicates the final
# founder set by BIC: greedy forward selection (_greedy_bic_select), swap
# refinement (_swap_refine), BIC pruning (_bic_prune), plus hap-equality and
# Hamming-overlap dedup primitives.  Shared by the bhd_recovery orchestrators
# and by block_haplotypes' seeding path.
#
# Leaf module of the bhd_recovery 4-file split: numpy, bhd_fit._compute_cc, and
# a few BIC/dedup thresholds from bhd_config.  See bhd_recovery.py for subsystem
# context.

import numpy as np

import warnings
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    warnings.warn(
        "Numba not found; bhd_recovery_select kernels fall back to pure Python "
        "(slower but numerically identical).",
        ImportWarning,
    )
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return decorator
    prange = range

import dynamic_threads
from bhd_fit import _compute_cc
from bhd_config import (
    RECOVERY_HAPS_EQUAL_EPS_PCT,
    RECOVERY_MAX_K,
    RECOVERY_OUTER_CC_SCALE,
    RECOVERY_SWAP_NLL_TOLERANCE,
)


def _greedy_bic_select(cache, cc_scale=RECOVERY_OUTER_CC_SCALE,
                        max_k=RECOVERY_MAX_K,
                        use_log_bic=False,
                        verbose=False):
    """Greedy forward selection by BIC over a fixed candidate pool.

    Haps are FIXED during scoring (no coord descent).  At each step,
    pick the candidate giving the lowest NLL when added; accept iff
    BIC improves (equivalent to NLL_improvement > cc/2).

    Uses PoolEmissionCache to avoid rebuilding the Viterbi emission
    tensor for each trial subset — the pool's emissions are precomputed
    once when the cache is built, and each trial only does the (much
    cheaper) state-axis selection + Viterbi forward pass.

    Args:
      cache: PoolEmissionCache wrapping the candidate pool.  cache.pool_-
        haps is the candidate list; cache.N and cache.L_kept are used
        for cc computation.
      cc_scale: BIC complexity-cost scale (outer; default 0.5)
      max_k: hard cap on selected size
      use_log_bic: if True, use log-BIC formula for cc (cc_scale *
        log(N*L) * snp_growth); if False (default, project standard),
        use linear formula (cc_scale * snp_growth * N).  Must match
        the use_log_bic of the surrounding K-growth so accept/reject
        criteria are consistent across the pipeline.
      verbose: print accept/reject decisions

    Returns:
      selected_indices: list of indices into cache.pool_haps
      selected_haps:    list of arrays (same as cache.pool_haps[i] for
                        i in selected_indices)
      current_nll:      NLL at final selection
    """
    candidate_haps = cache.pool_haps
    N = cache.N
    L_kept = cache.L_kept
    cc = _compute_cc(cc_scale, N, L_kept, use_log_bic=use_log_bic)

    # K=0 baseline NLL via the cache (precomputed at construction).
    nll_K0 = cache.nll_for_subset([])
    bic_K0 = 0 * cc + 2 * nll_K0

    selected_indices = []
    used = set()
    current_bic = bic_K0
    current_nll = nll_K0

    if verbose:
        print(f'    Forward: K=0 NLL={nll_K0:.1f}, BIC={bic_K0:.1f}, '
              f'cc={cc:.1f}, threshold cc/2={cc/2:.1f}')

    while len(selected_indices) < min(len(candidate_haps), max_k):
        # Re-check the live thread budget before this round's parallel batch_nll
        # sweep, so a straggler block grows into cores freed by finished peers
        # mid-selection (no-op on the sequential path).
        dynamic_threads.apply_dynamic_threads()
        # Evaluate adding each still-unused candidate, parallelised ACROSS
        # candidates (prange in cache.batch_nll_for_subsets) instead of a
        # sequential Python loop of one tiny per-sample Viterbi each — the
        # latter pins ~1 core on pathological large-pool blocks.  `unused` is
        # built in increasing-ci order and the reduction below uses strict <,
        # so the chosen candidate (and lowest-ci tie-break) is identical to the
        # old loop; each NLL is bit-identical to cache.nll_for_subset.
        unused = [ci for ci in range(len(candidate_haps)) if ci not in used]
        best_ci = -1
        best_nll = float('inf')
        if unused:
            trial_subsets = np.array(
                [selected_indices + [ci] for ci in unused], dtype=np.int64)
            trial_nlls = cache.batch_nll_for_subsets(trial_subsets)
            for j in range(len(unused)):
                if trial_nlls[j] < best_nll:
                    best_nll = float(trial_nlls[j])
                    best_ci = unused[j]

        if best_ci < 0:
            break

        k_new = len(selected_indices) + 1
        bic_new = k_new * cc + 2 * best_nll
        d_nll = current_nll - best_nll

        if bic_new < current_bic:
            selected_indices.append(best_ci)
            used.add(best_ci)
            if verbose:
                print(f'    Forward: K={k_new} ACCEPT cand[{best_ci}], '
                      f'NLL={best_nll:.1f}, BIC={bic_new:.1f}, dNLL={d_nll:.1f}')
            current_bic = bic_new
            current_nll = best_nll
        else:
            if verbose:
                print(f'    Forward: K={k_new} REJECT cand[{best_ci}], '
                      f'NLL={best_nll:.1f}, BIC={bic_new:.1f}, dNLL={d_nll:.1f} '
                      f'< cc/2={cc/2:.1f}')
            break

    selected_haps = [candidate_haps[i] for i in selected_indices]
    return selected_indices, selected_haps, current_nll


def _swap_refine(cache, selected_indices, current_nll,
                  nll_tolerance=RECOVERY_SWAP_NLL_TOLERANCE,
                  max_passes=10, verbose=False):
    """Try swapping each selected hap with each unselected pool member.
    Apply swap if NLL improves by more than nll_tolerance.  Iterate over
    passes until no improvement.

    Sometimes greedy forward selection picks a near-optimal hap early
    that becomes redundant after later picks; swap lets us replace it
    with a better one without requiring a full re-search.

    Uses PoolEmissionCache so each trial costs O(N · K_sub_states ·
    n_bins) for the state-axis selection + Viterbi forward pass, rather
    than the original O(N · K_sub_states · L) for emission rebuild +
    Viterbi.  At K_sub=6 this is a roughly order-of-magnitude saving per
    trial, and `_swap_refine` performs many trials per call (~K * pool
    per pass, ~K passes).

    Args:
      cache: PoolEmissionCache over the swap-pool of candidates.
      selected_indices: list of indices into cache.pool_haps — current
        selection.  Replaced (or kept) one-at-a-time during refinement.
      current_nll: scalar — NLL at the current selection, used as the
        comparison point.  Caller should pass the NLL value at the
        SAME indices the cache would score (typically the output of
        _greedy_bic_select that just preceded the swap).
      nll_tolerance: minimum NLL improvement required to accept a swap.
      max_passes: defensive cap on the outer pass loop.
      verbose: print accept decisions per swap.

    Returns:
      (selected_indices, selected_haps, current_nll, n_swaps)
    """
    sel_ind = list(selected_indices)
    K = len(sel_ind)
    if K == 0:
        return sel_ind, [], current_nll, 0

    pool_size = cache.K_pool
    n_swaps = 0
    for pass_num in range(max_passes):
        # Re-check the live thread budget once per pass, so a straggler block
        # grows into cores freed by finished peers mid-refinement (no-op on the
        # sequential path).  Per-pass, not per-position: keeps the shared-counter
        # poll off the inner sweep loop.
        dynamic_threads.apply_dynamic_threads()
        improved_in_pass = False
        for si in range(K):
            # Sweep all unused pool members for position si, parallelised
            # ACROSS candidates (prange) instead of a sequential per-candidate
            # Python loop.  Threshold start (current_nll - nll_tolerance) and
            # strict-< reduction over `unused` in increasing-ci order reproduce
            # the old loop exactly (same accepted swap, same lowest-ci
            # tie-break); each NLL is bit-identical to cache.nll_for_subset.
            unused = [ci for ci in range(pool_size) if ci not in sel_ind]
            best_ci = -1
            best_nll = current_nll - nll_tolerance
            if unused:
                base = np.array(sel_ind, dtype=np.int64)
                trial_subsets = np.empty((len(unused), K), dtype=np.int64)
                for j in range(len(unused)):
                    trial_subsets[j] = base
                    trial_subsets[j, si] = unused[j]
                trial_nlls = cache.batch_nll_for_subsets(trial_subsets)
                for j in range(len(unused)):
                    if trial_nlls[j] < best_nll:
                        best_nll = float(trial_nlls[j])
                        best_ci = unused[j]
            if best_ci >= 0:
                if verbose:
                    print(f'    Swap: pos {si} (cand[{sel_ind[si]}]) -> cand[{best_ci}], '
                          f'NLL {current_nll:.1f} -> {best_nll:.1f}')
                sel_ind[si] = best_ci
                current_nll = best_nll
                improved_in_pass = True
                n_swaps += 1
                break
        if not improved_in_pass:
            break

    sel_haps = [cache.pool_haps[i] for i in sel_ind]
    return sel_ind, sel_haps, current_nll, n_swaps


def _bic_prune(cache, selected_indices,
                cc_scale=RECOVERY_OUTER_CC_SCALE, use_log_bic=False,
                verbose=False):
    """BIC pruning: try dropping each selected hap.  Drop if the NLL
    increase from removal is less than cc/2 (i.e., the hap isn't pulling
    enough weight to justify the +cc penalty for keeping it).

    Iterates: each drop may enable another (cascading prune of redundant
    haps that propped each other up).  Matches the project's
    refine_selection_by_pruning pattern in beam_search_core.

    Uses PoolEmissionCache so each leave-one-out trial reuses the pool's
    precomputed emissions (no per-trial rebuild).  For a K-sized
    selection, one prune iteration evaluates K trials (one per dropped
    hap) plus one full-K-NLL reference; the cache makes each trial
    cheap.

    use_log_bic: bool — if True, use log-BIC formula for cc; if False
      (default, project standard), use linear formula.  Must match the
      use_log_bic of the surrounding K-growth so the prune threshold
      cc/2 is consistent with K-growth's growth threshold.

    Args:
      cache: PoolEmissionCache over the candidate pool.
      selected_indices: list of pool indices forming the current
        selection (subset of cache.pool_haps).
      cc_scale, use_log_bic: BIC parameters; cc/2 is the drop threshold.
      verbose: print drop decisions.

    Returns: (pruned_indices, pruned_haps, final_nll, n_dropped)
    """
    N = cache.N
    L_kept = cache.L_kept
    cc = _compute_cc(cc_scale, N, L_kept, use_log_bic=use_log_bic)

    sel_ind = list(selected_indices)
    n_dropped = 0

    while len(sel_ind) > 0:
        # Re-check the live thread budget before this round's parallel batch_nll
        # leave-one-out sweep, so a straggler block grows into cores freed by
        # finished peers mid-prune (no-op on the sequential path).
        dynamic_threads.apply_dynamic_threads()
        nll_full = cache.nll_for_subset(sel_ind)
        K = len(sel_ind)

        best_drop_idx = -1
        best_dnll = cc / 2   # threshold; only drop if dnll_increase < this

        # Evaluate every leave-one-out drop in parallel across positions
        # (prange in cache.batch_nll_for_subsets) instead of a sequential
        # Python loop.  Trials are built in increasing-position order and the
        # reduction uses strict <, so the dropped position (and lowest-index
        # tie-break) matches the old loop; each leave-one-out NLL is
        # bit-identical to cache.nll_for_subset.  K == 1's only trial is the
        # empty subset, scored via nll_for_subset (the precomputed K=0 NLL),
        # since the batch kernel's contract is K_sub >= 1.
        if K == 1:
            nll_trial = cache.nll_for_subset([])
            if nll_trial - nll_full < best_dnll:
                best_dnll = float(nll_trial - nll_full)
                best_drop_idx = 0
        else:
            sel_arr = np.array(sel_ind, dtype=np.int64)
            trial_subsets = np.empty((K, K - 1), dtype=np.int64)
            for i in range(K):
                trial_subsets[i, :i] = sel_arr[:i]
                trial_subsets[i, i:] = sel_arr[i + 1:]
            trial_nlls = cache.batch_nll_for_subsets(trial_subsets)
            for i in range(K):
                dnll = trial_nlls[i] - nll_full   # NLL increase from dropping
                if dnll < best_dnll:
                    best_dnll = float(dnll)
                    best_drop_idx = i

        if best_drop_idx < 0:
            break

        if verbose:
            print(f'    Prune: drop pos {best_drop_idx} (cand[{sel_ind[best_drop_idx]}]), '
                  f'NLL increase {best_dnll:.1f} < cc/2={cc/2:.1f} -- DROPPED')
        del sel_ind[best_drop_idx]
        n_dropped += 1

    final_nll = cache.nll_for_subset(sel_ind)
    sel_haps = [cache.pool_haps[i] for i in sel_ind]
    return sel_ind, sel_haps, final_nll, n_dropped


@njit(cache=True, parallel=True, fastmath=False)
def _dedup_vs_refset_kernel(cands, refset, threshold_pct):
    """For each row of `cands`, flag whether it duplicates ANY row of
    `refset` -- i.e. lies within threshold_pct Hamming-% of it.

    Parallel pre-screen for the rescue dedup loops.  The per-pair distance
    is computed exactly as _hamming_pct_kept: count differing sites, divide
    by L, multiply by 100.  prange is over candidates; each candidate scans
    the reference rows with an early break on the first match.  Because the
    "within threshold of ANY reference row" decision does not depend on the
    order of candidates, it is safe to compute for all candidates in
    parallel -- unlike the dedup-vs-already-accepted check, which is
    order-dependent and stays sequential in the caller.

    Args:
        cands: (n, L) int64 -- candidate haps.
        refset: (R, L) int64 -- reference haps (e.g. the current founders).
        threshold_pct: float -- duplicate if Hamming-% < this.

    Returns:
        (n,) bool -- True where the candidate duplicates some reference row.
    """
    n = cands.shape[0]
    L = cands.shape[1]
    R = refset.shape[0]
    out = np.zeros(n, dtype=np.bool_)
    for i in prange(n):
        dup = False
        for r in range(R):
            diff = 0
            for l in range(L):
                if cands[i, l] != refset[r, l]:
                    diff += 1
            if (diff / L) * 100.0 < threshold_pct:
                dup = True
                break
        out[i] = dup
    return out


@njit(cache=True)
def _hamming_pct_kept_kernel(a, b):
    """Numba kernel for _hamming_pct_kept.

    Scalar loop over the two arrays, counting differing positions.
    Returns the percentage of differing sites (0.0 to 100.0).

    Both inputs must be int64 1D arrays of the same length — the
    wrapper handles dtype casting.

    Mathematical equivalence to `float(np.mean(a != b)) * 100.0`:
      np.mean over a bool array returns count_true / L as float64,
      then multiplied by 100.  We compute the same thing: count
      differing positions, divide by L, multiply by 100.  No round-
      off difference because the intermediate count is an integer
      and the final division is float64.
    """
    L = a.shape[0]
    diff_count = 0
    for i in range(L):
        if a[i] != b[i]:
            diff_count += 1
    return (diff_count / L) * 100.0


def _hamming_pct_kept(a, b):
    """Per-site Hamming distance as a percentage (0-100).

    Implementation: delegates to a numba kernel that uses a scalar
    loop, avoiding the (L,) bool array allocation per call.  Called
    very frequently (~120 times per block in the round loop + late
    rescue dedup loops, at typically 3-4 µs per call); the cumulative
    cost is small but the numba kernel halves it cleanly.

    Inputs are cast to int64 inside the wrapper so the kernel compiles
    only once regardless of caller-side dtype (some callers pass int8
    from candidate pools, others pass int64 from H rows or fully-
    resolved consensus haps).
    """
    a_arr = np.ascontiguousarray(a, dtype=np.int64)
    b_arr = np.ascontiguousarray(b, dtype=np.int64)
    return _hamming_pct_kept_kernel(a_arr, b_arr)


@njit(cache=True)
def _haps_equal_kernel(haps_a, haps_b, eps_pct):
    """Numba kernel for _haps_equal.

    Bipartite matching: for each row in haps_a, find an unmatched row
    in haps_b within eps_pct Hamming.  Returns True iff every row in
    haps_a finds a unique match.

    The semantics are EXACTLY the original's:
      - First-match wins: for each ha, we walk haps_b in index order
        and accept the first unmatched hb within eps_pct.  This is
        greedy and may fail to find a valid matching that exists if
        ordering is unlucky, but matches the original's behavior bit-
        for-bit.
      - `_hamming_pct_kept(ha, hb) < eps_pct` uses strict inequality.
      - matched_b is a per-call bool array of length K.

    Args:
        haps_a, haps_b: (K, L) int64 arrays — same shape, otherwise
            the wrapper returns False without calling this kernel.
        eps_pct: float — Hamming-percent tolerance.

    Returns:
        bool — True iff a valid bipartite matching exists.
    """
    K, L = haps_a.shape
    # bool scratch — reset per call.  np.zeros works fine in numba.
    matched_b = np.zeros(K, dtype=np.bool_)
    for ai in range(K):
        found = False
        for bi in range(K):
            if matched_b[bi]:
                continue
            # Inline _hamming_pct_kept: count differing positions
            diff_count = 0
            for l in range(L):
                if haps_a[ai, l] != haps_b[bi, l]:
                    diff_count += 1
            ham_pct = (diff_count / L) * 100.0
            if ham_pct < eps_pct:
                matched_b[bi] = True
                found = True
                break
        if not found:
            return False
    # All ai matched.  In a bipartite K x K matching where each ai matched
    # exactly one bi, all matched_b must be True — but we don't need to
    # check `all(matched_b)` separately: K matches across K positions
    # forces it.  Match the original's final `return all(matched_b)`
    # for safety; equivalent here since we never reach this point unless
    # K matches found.
    for bi in range(K):
        if not matched_b[bi]:
            return False
    return True


def _haps_equal(haps_a, haps_b, eps_pct=RECOVERY_HAPS_EQUAL_EPS_PCT):
    """Test if two hap collections are equal (within eps_pct Hamming
    tolerance per pair, with bipartite matching).

    Returns True iff:
      - len(haps_a) == len(haps_b), AND
      - there's a 1-to-1 matching where each matched pair is within eps_pct.

    Used for convergence detection between recovery rounds and between
    outer iterations.  Tolerance accommodates near-identical haps that
    differ only at a few uncertain sites.

    Implementation: delegates to a numba kernel that takes (K, L)
    int64 arrays.  Wrapper stacks list-of-arrays inputs into 2D
    contiguous arrays.  Output is byte-identical to the original
    (first-match-wins greedy bipartite matching with strict-inequality
    Hamming threshold).
    """
    if len(haps_a) != len(haps_b):
        return False
    # Handle empty case before stacking (np.stack rejects empty inputs)
    if len(haps_a) == 0:
        return True
    # Stack into (K, L) int64 arrays.  The original accepts either lists
    # of (L,) arrays or 2D ndarrays; both forms produce a valid 2D
    # array under np.stack with axis=0.
    a_arr = np.ascontiguousarray(np.stack(list(haps_a), axis=0), dtype=np.int64)
    b_arr = np.ascontiguousarray(np.stack(list(haps_b), axis=0), dtype=np.int64)
    if a_arr.shape != b_arr.shape:
        return False
    return _haps_equal_kernel(a_arr, b_arr, float(eps_pct))


# =============================================================================
# CARRIER-COUNTING KERNEL
# =============================================================================
# Pure-numerical inner loop pulled out of _late_low_carrier_rescue so the
# double loop over (sample, strand) runs at njit speed.  Counts how many
# real-strand carriers each hap has, treating the wildcard sentinel
# (A[s, slot] == K) as "not a real carrier" -- those strands are
# excluded from the per-hap usage tally.
#
# Inputs:
#   A: (N, 2) int — pair assignments; entries in [0, K] where K is the
#                   wildcard sentinel
#   K: int       — number of real haps (the wildcard sentinel value)
#
# Returns:
#   usage: (K,) int64 — per-hap real-strand carrier counts

@njit(cache=True)
def _count_real_carriers_kernel(A, K):
    N = A.shape[0]
    usage = np.zeros(K, dtype=np.int64)
    for s in range(N):
        for slot in range(2):
            f = int(A[s, slot])
            if f != K:
                usage[f] += 1
    return usage


def _count_real_carriers(A, K):
    """Thin wrapper that ensures contiguous int input before dispatching
    to the njit kernel.  At N=320 this is ~640 inner iterations, so
    contiguity overhead matters less than the kernel call cost, but the
    contract is uniform with the rest of this module's wrappers."""
    A_arr = np.ascontiguousarray(A, dtype=np.int64)
    return _count_real_carriers_kernel(A_arr, int(K))