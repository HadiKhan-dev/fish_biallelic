"""Chimera detection / pair-error kernels and prune_chimeras, split out of
bhd_kernels.  Uses dynamic_threads for the scoring-loop thread hook."""

import numpy as np
from numba import njit, prange

from bhd_kernels import (
    HAS_NUMBA,
)
import dynamic_threads


@njit(parallel=False, fastmath=True)
def viterbi_chimera_check_free_recombs(ll_tensor, max_recombs):
    """
    Viterbi that allows FREE recombinations up to max_recombs.
    
    Returns the minimum mismatch count achievable with at most max_recombs switches.
    This cleanly separates the structural question (can it be reconstructed?) from
    the quality question (with how many mismatches?).
    
    Args:
        ll_tensor: (n_samples, K_haps, n_sites) - values are 0 (match) or negative (mismatch)
        max_recombs: Maximum allowed recombinations (free, no penalty)
    
    Returns:
        min_mismatches: (n_samples,) - minimum mismatches achievable within recomb budget
        best_recombs: (n_samples,) - number of recombs used for best path
    """
    n_samples, K, n_sites = ll_tensor.shape
    min_mismatches = np.empty(n_samples, dtype=np.int64)
    best_recombs = np.empty(n_samples, dtype=np.int64)
    
    INF = 999999999
    
    for s in range(n_samples):
        # dp[k][r] = minimum mismatches to reach hap k with exactly r recombs used
        # Initialize at site 0
        dp = np.full((K, max_recombs + 1), INF, dtype=np.int64)
        
        for k in range(K):
            is_mismatch = 1 if ll_tensor[s, k, 0] < -0.1 else 0
            dp[k, 0] = is_mismatch  # Start on hap k with 0 recombs
        
        # Process sites 1 to n_sites-1
        for i in range(1, n_sites):
            new_dp = np.full((K, max_recombs + 1), INF, dtype=np.int64)
            
            # First, find best (minimum mismatch) state for each recomb count
            # to avoid O(K^2) work per site
            best_at_r = np.full(max_recombs + 1, INF, dtype=np.int64)
            for r in range(max_recombs + 1):
                for k in range(K):
                    if dp[k, r] < best_at_r[r]:
                        best_at_r[r] = dp[k, r]
            
            for k in range(K):
                is_mismatch = 1 if ll_tensor[s, k, i] < -0.1 else 0
                
                for r in range(max_recombs + 1):
                    # Option 1: Stay on hap k (no new recomb)
                    if dp[k, r] < INF:
                        new_val = dp[k, r] + is_mismatch
                        if new_val < new_dp[k, r]:
                            new_dp[k, r] = new_val
                    
                    # Option 2: Switch from any other hap (costs 1 recomb)
                    if r > 0 and best_at_r[r - 1] < INF:
                        # We can switch from the best state at r-1 recombs
                        new_val = best_at_r[r - 1] + is_mismatch
                        if new_val < new_dp[k, r]:
                            new_dp[k, r] = new_val
            
            # Copy new_dp to dp
            for k in range(K):
                for r in range(max_recombs + 1):
                    dp[k, r] = new_dp[k, r]
        
        # Find best (minimum mismatches across all ending states)
        best_mm = INF
        best_r = 0
        for k in range(K):
            for r in range(max_recombs + 1):
                if dp[k, r] < best_mm:
                    best_mm = dp[k, r]
                    best_r = r
        
        min_mismatches[s] = best_mm
        best_recombs[s] = best_r
    
    return min_mismatches, best_recombs

@njit(cache=True, parallel=True, fastmath=False)
def _compute_best_pair_errors_kernel(H_stack, sample_geno_stack):
    """JIT inner kernel for compute_best_pair_errors (used by prune_chimeras).
    
    For each sample s:
      sample_geno = sample_geno_stack[s]  (n_sites,) int8 in {0,1,2}
      best_error = min over (i, j) with j>=i of:
          mean(H_stack[i] + H_stack[j] != sample_geno) * 100
    
    Strict < for tie-breaking matches the original: on a tie the FIRST
    pair (lexicographically lowest) wins, since we iterate i outer, j>=i inner.
    
    Args:
        H_stack: (K, n_sites) int8 — concretised haplotypes (caller has
            already converted any 2D probabilistic haps via np.argmax).
        sample_geno_stack: (num_samples, n_sites) int8 — caller has already
            done np.argmax(probs_array[s], axis=1) for each sample.
    
    Returns:
        sample_errors: (num_samples,) float64 — per-sample best-pair error
            in percent (0.0 to 100.0).
    """
    K, n_sites = H_stack.shape
    num_samples = sample_geno_stack.shape[0]
    
    sample_errors = np.empty(num_samples, dtype=np.float64)
    
    # Per-sample parallelism: each thread fully processes its samples
    # (lock-free; sample_errors[si] is each thread's exclusive slot).
    # The within-sample iteration order matches the original Python:
    # i outer (0..K-1), j inner (i..K-1), so on a tie the lower (i,j) wins.
    for si in prange(num_samples):
        best_error = 100.0  # match the original initial value
        
        for i in range(K):
            for j in range(i, K):
                # Count mismatches between (H_stack[i] + H_stack[j]) and sample_geno
                mismatches = 0
                for s in range(n_sites):
                    g = H_stack[i, s] + H_stack[j, s]
                    if g != sample_geno_stack[si, s]:
                        mismatches += 1
                # mean(...)*100 == (count/n_sites)*100
                error = (mismatches / n_sites) * 100.0
                # Strict < to match original tie-break (first pair wins)
                if error < best_error:
                    best_error = error
        
        sample_errors[si] = best_error
    
    return sample_errors


@njit(cache=True, parallel=True, fastmath=False)
def _argmax3_numba(probs):
    """Parallel equivalent of np.argmax(probs, axis=2).astype(np.int8) for a
    (N, L, 3) array (per-site genotype argmax over the 3 dosage classes).

    Returns (N, L) int8.  Matches np.argmax exactly, including the
    first-occurrence tie-break: the branch order returns the lowest index whose
    value equals the row max.  Parallelised over samples instead of running the
    reduction single-threaded in numpy, which scans the whole probs array (~5.4
    GB at L4) on one core.
    """
    N = probs.shape[0]
    L = probs.shape[1]
    out = np.empty((N, L), dtype=np.int8)
    for s in prange(N):
        for t in range(L):
            p0 = probs[s, t, 0]
            p1 = probs[s, t, 1]
            p2 = probs[s, t, 2]
            if p0 >= p1:
                if p0 >= p2:
                    out[s, t] = 0
                else:
                    out[s, t] = 2
            else:
                if p1 >= p2:
                    out[s, t] = 1
                else:
                    out[s, t] = 2
    return out


@njit(cache=True, parallel=True, fastmath=False)
def _compute_pair_errors_matrix_kernel(H_stack, sample_geno_stack):
    """JIT kernel returning the FULL per-sample, per-pair error matrix.
    
    Computes M[s, i, j] = (mismatch_count[s, i, j] / n_sites) * 100 for
    every sample s and every pair (i, j) with j >= i.  Entries with j < i
    are filled with +inf so they're effectively ignored by min operations.
    
    Used by prune_chimeras' refactored outer loop, which builds M once per
    iteration and then derives both the "baseline best error" (min over
    all pairs) and the "best error excluding hap k" (min over pairs not
    involving k) from the same matrix — saving the K-fold redundant work
    that the original closure-per-call structure incurred.
    
    Asymptotic improvement: prune_chimeras' inner loop went from K+1
    calls of O(num_samples * K^2 * n_sites) to a single matrix build of
    O(num_samples * K^2 * n_sites) plus K+1 O(num_samples * K^2) min
    operations.  Net speedup factor ~K when many candidates pass the
    structural chimera test.
    
    Args:
        H_stack: (K, n_sites) int8 — concretised haplotypes.
        sample_geno_stack: (num_samples, n_sites) int8 — pre-computed
            argmax of probs_array along the allele axis.
    
    Returns:
        M: (num_samples, K, K) float64 — error percent for each (s, i, j)
            with j >= i; M[s, i, j] = +inf for j < i.
    """
    K, n_sites = H_stack.shape
    num_samples = sample_geno_stack.shape[0]
    
    M = np.full((num_samples, K, K), np.inf, dtype=np.float64)
    
    # Per-sample parallelism: each thread writes to its own (s, :, :) slab.
    for si in prange(num_samples):
        for i in range(K):
            for j in range(i, K):
                mismatches = 0
                for s in range(n_sites):
                    g = H_stack[i, s] + H_stack[j, s]
                    if g != sample_geno_stack[si, s]:
                        mismatches += 1
                M[si, i, j] = (mismatches / n_sites) * 100.0
    
    return M


@njit(cache=True, parallel=True, fastmath=False)
def _min_pair_error_excluding(M, excluded_idx):
    """Compute, for each sample, the min over (i, j) pairs with j >= i and
    neither i nor j equal to excluded_idx.
    
    If excluded_idx < 0, returns the unrestricted min (used for baseline).
    
    Strict < tie-breaking matches the original (earliest-encountered pair
    wins on ties).  The iteration order (i outer 0..K-1, j inner i..K-1)
    matches _compute_best_pair_errors_kernel's order, so when excluded_idx
    is -1 this function returns exactly the same per-sample errors as
    _compute_best_pair_errors_kernel applied to the same H_stack.
    """
    num_samples, K, _ = M.shape
    out = np.empty(num_samples, dtype=np.float64)
    
    for si in prange(num_samples):
        best_error = 100.0
        for i in range(K):
            if i == excluded_idx:
                continue
            for j in range(i, K):
                if j == excluded_idx:
                    continue
                e = M[si, i, j]
                if e < best_error:
                    best_error = e
        out[si] = best_error
    return out


def prune_chimeras(hap_dict, probs_array, 
                   max_recombs=1,
                   max_mismatch_percent=0.5,
                   min_mean_delta_to_protect=0.25):
    """
    Identifies and removes haplotypes that can be explained as 
    recombinations of OTHER haplotypes, using mean_delta to protect
    essential haplotypes.
    
    Algorithm:
    1. Find all candidate chimeras (structural: >= 1 recomb, <= max_recombs, <= max_mismatch_percent)
    2. For each candidate, compute mean_delta = mean increase in sample error if removed
    3. Remove the candidate with minimum mean_delta (if below protection threshold)
    4. Repeat until no removable candidates remain
    
    A haplotype is protected if:
    - It cannot be explained as a chimera of others, OR
    - Its mean_delta >= min_mean_delta_to_protect (removing it hurts too much)
    
    Args:
        hap_dict: Dictionary of haplotypes {idx: array}
        probs_array: Sample genotype probabilities (n_samples, n_sites, 3)
        max_recombs: Maximum recombinations for chimera detection (default 1)
        max_mismatch_percent: Maximum mismatch % for chimera (default 1.0)
        min_mean_delta_to_protect: Protect haplotypes with mean_delta above this (default 0.25%)
    """
    if len(hap_dict) < 3:
        return hap_dict
    
    if probs_array is None:
        return hap_dict
    
    # Pre-compute sample genotype stack once.  In the original code this
    # was done inside compute_best_pair_errors per call (and per sample),
    # but probs_array is invariant during the prune loop — only the hap
    # subset changes — so we hoist the argmax out.  Caller's probs_array
    # is typically (num_samples, num_sites, 3) float32 in production.
    num_samples_outer = probs_array.shape[0]
    num_sites_outer = probs_array.shape[1]
    # np.argmax over axis=2 returns int64 by default; cast to int8 to match
    # the JIT kernel's expected dtype.  Values are always in {0, 1, 2}.
    # _argmax3_numba is bit-identical to np.argmax(probs_array, axis=2) (same
    # first-occurrence tie-break) but parallel; the numpy reduction scans the
    # whole probs_array single-threaded (~5.4 GB at L4).
    if HAS_NUMBA:
        try:
            sample_geno_stack = _argmax3_numba(probs_array)
        except Exception as _argmax_err:
            # _argmax3_numba is @njit(cache=True, parallel=True).  Under heavy
            # forkserver worker recycling with the numba cache on a network
            # filesystem, a freshly-spawned worker can intermittently fail to
            # unbox its argument ("can't unbox array from PyObject into native
            # value") even for a perfectly valid (num_samples, num_sites, 3)
            # float32 array -- a numba cache/runtime glitch in that worker, not
            # bad data.  np.argmax over axis=2 is bit-identical to the kernel
            # (same first-occurrence tie-break, see the comment above), so fall
            # back to it rather than aborting the whole hierarchical-assembly
            # batch.  The array's signature is logged so a *genuine* wrong-type
            # input would still be visible: np.argmax raises loudly on object /
            # ragged / wrong-ndim arrays and never silently mis-reduces a valid
            # one, so the fallback can only produce the identical result or fail.
            print(f"  [prune_chimeras] _argmax3_numba raised "
                  f"{type(_argmax_err).__name__}: {_argmax_err} | probs_array "
                  f"type={type(probs_array).__name__} "
                  f"dtype={getattr(probs_array, 'dtype', None)} "
                  f"shape={getattr(probs_array, 'shape', None)} "
                  f"C_contig={getattr(getattr(probs_array, 'flags', None), 'c_contiguous', None)} "
                  f"-> using bit-identical np.argmax fallback", flush=True)
            sample_geno_stack = np.argmax(probs_array, axis=2).astype(np.int8)
    else:
        sample_geno_stack = np.argmax(probs_array, axis=2).astype(np.int8)
    
    def compute_best_pair_errors(hap_dict_local, sample_geno_stack_local):
        """For each sample, compute error of best haplotype pair (no recomb within block).
        
        Uses the JIT kernel when numba is available; falls back to the
        original pure-Python triply-nested loop otherwise.  The two
        paths are byte-identical because the inner work is integer-only
        (concretised haps + integer sample genotypes, integer mismatch
        count, then a single float64 division).
        """
        # Convert the hap_dict's values to a stacked (K, n_sites) int8 array
        # matching what the kernel (and the original Python loop) expects.
        hap_list = [np.argmax(h, axis=1) if h.ndim > 1 else h for h in hap_dict_local.values()]
        num_haps = len(hap_list)
        
        if HAS_NUMBA:
            # Build H_stack as int8 (values in {0, 1}).  hap_list elements
            # may already be int8 (if 1D) or int64 (from np.argmax of 2D);
            # vstack-then-cast unifies to int8.
            H_stack = np.vstack(hap_list).astype(np.int8)
            return _compute_best_pair_errors_kernel(H_stack, sample_geno_stack_local)
        
        # ---- Pure-Python fallback ----
        # Identical math to the JIT kernel and to the original
        # implementation (just reads from the pre-computed sample_geno_stack
        # rather than recomputing argmax per sample).
        num_samples = sample_geno_stack_local.shape[0]
        num_sites = sample_geno_stack_local.shape[1]
        sample_errors = np.zeros(num_samples)
        
        for s in range(num_samples):
            sample_geno = sample_geno_stack_local[s]
            best_error = 100.0
            
            for i in range(num_haps):
                for j in range(i, num_haps):
                    geno = hap_list[i] + hap_list[j]
                    error = np.mean(geno != sample_geno) * 100
                    if error < best_error:
                        best_error = error
            
            sample_errors[s] = best_error
        
        return sample_errors
    
    kept_keys = sorted(list(hap_dict.keys()))
    n_sites = next(iter(hap_dict.values())).shape[0]
    
    while len(kept_keys) >= 3:
        dynamic_threads.apply_dynamic_threads()
        
        # Build haplotype stack
        H_stack = np.array([np.argmax(hap_dict[k], axis=1) if hap_dict[k].ndim == 2 
                           else hap_dict[k] for k in kept_keys])
        n_haps = len(kept_keys)
        
        # ----------------------------------------------------------------
        # Build the full per-sample, per-pair error matrix M[s, i, j] once
        # for the current hap set.  baseline_errors and every reduced_errors
        # (one per chimera candidate) are derived from M via cheap O(K^2)
        # min operations, eliminating the K-fold redundant work that the
        # original closure-per-call structure incurred.
        # 
        # When numba is unavailable, fall back to calling the closure
        # baseline call only and computing reduced calls per candidate
        # later (slow path; preserves correctness).
        # ----------------------------------------------------------------
        if HAS_NUMBA:
            H_stack_i8 = H_stack.astype(np.int8) if H_stack.dtype != np.int8 else H_stack
            pair_error_matrix = _compute_pair_errors_matrix_kernel(H_stack_i8, sample_geno_stack)
            baseline_errors = _min_pair_error_excluding(pair_error_matrix, -1)
        else:
            pair_error_matrix = None
            current_dict = {k: hap_dict[k] for k in kept_keys}
            baseline_errors = compute_best_pair_errors(current_dict, sample_geno_stack)
        
        # Find candidate chimeras
        candidate_chimeras = []
        
        for i, target_key in enumerate(kept_keys):
            target_hap = H_stack[i]
            
            other_indices = [idx for idx in range(n_haps) if idx != i]
            other_haps = H_stack[other_indices]
            
            # Build mismatch tensor
            mismatches = np.abs(target_hap[None, :] - other_haps)
            ll_tensor = np.where(mismatches < 0.5, 0.0, -1.0)
            ll_tensor = ll_tensor[np.newaxis, :, :].astype(np.float64)
            
            # Check if chimera (structural test)
            mismatch_counts, recombs = viterbi_chimera_check_free_recombs(ll_tensor, max_recombs)
            
            n_recombs_val = recombs[0]
            n_mismatches_val = mismatch_counts[0]
            mismatch_pct = (n_mismatches_val / n_sites) * 100
            
            # Must have at least 1 recomb (otherwise it's just a similar haplotype, not a chimera)
            is_chimera = (n_recombs_val >= 1) and (n_recombs_val <= max_recombs) and (mismatch_pct <= max_mismatch_percent)
            
            if is_chimera:
                # Compute mean_delta: how much does removing this haplotype hurt?
                if HAS_NUMBA:
                    # Cheap O(K^2 * num_samples) min over the precomputed matrix.
                    reduced_errors = _min_pair_error_excluding(pair_error_matrix, i)
                else:
                    # Slow fallback: rebuild from scratch (matches original behaviour).
                    reduced_dict = {k: hap_dict[k] for k in kept_keys if k != target_key}
                    reduced_errors = compute_best_pair_errors(reduced_dict, sample_geno_stack)
                delta_errors = reduced_errors - baseline_errors
                mean_delta = np.mean(delta_errors)
                
                candidate_chimeras.append((target_key, n_recombs_val, n_mismatches_val, mismatch_pct, mean_delta))
        
        if len(candidate_chimeras) == 0:
            break
        
        # Sort by mean_delta (ascending) - remove the least essential first
        candidate_chimeras.sort(key=lambda x: x[4])
        
        # Check if the least essential candidate is below protection threshold
        worst_key, worst_recombs, worst_mismatches, worst_pct, worst_delta = candidate_chimeras[0]
        
        if worst_delta >= min_mean_delta_to_protect:
            # Even the least essential chimera is too important to remove
            break
        
        # Remove the least essential chimera
        kept_keys.remove(worst_key)
    
    return {k: hap_dict[k] for k in kept_keys}