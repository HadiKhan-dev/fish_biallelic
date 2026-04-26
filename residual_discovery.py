"""
Residual Discovery Module

Discovers missing founder haplotypes that HDBSCAN failed to find,
typically due to high IBS with an existing founder in that region.

Algorithm:
  1. Paint samples against K known founders per block
  2. Identify high-residual samples (carrying the missing founder)
  3. Subtract the known partner to extract the missing founder's alleles
  4. Iteratively refine via re-painting and re-extraction
  5. Filter via consistency (neighbour overlap), residual reduction,
     and structural chimera pruning

Runs post-refinement, pre-L1 assembly. Adds discovered haplotypes
to blocks and lets existing dedup + chimera pruning handle cleanup.

Main entry point: discover_missing_haplotypes()
"""

import numpy as np
import math
import time
import multiprocessing as mp
import multiprocessing.pool
import warnings
from multiprocessing.shared_memory import SharedMemory

import block_haplotypes
import block_haplotype_refinement


# =============================================================================
# JIT-COMPILED KERNELS
# =============================================================================
# 
# Numba kernels for the inner loops of _paint_samples_fast,
# _extract_missing_hap, and the partner-finding inlined loops.  These
# operate on float64 internally for log accumulation; the wrapping
# Python functions below cast block_probs to float64 on entry to ensure
# consistent dispatch.  Validation against the original Python
# implementations on tropheops-shape inputs (116 samples, 200 sites,
# K~8, float32 block_probs) shows:
#   - best_pair: byte-identical
#   - residuals: differ by <1e-8 absolute, <1e-7 relative
#   - MAD outlier set: byte-identical
#   - extracted haps: byte-identical
#   - partner indices: byte-identical
# The tiny residual differences come from float64 vs float32 log
# accumulation order; they are below any decision boundary in the
# downstream MAD threshold (which has a hard floor of 0.01) and
# do not affect any production output.

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    warnings.warn("Numba not found in residual_discovery; falling back to Python loops.",
                  ImportWarning)
    def njit(*args, **kwargs):
        # Support both @njit and @njit(...) forms.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def decorator(func): return func
        return decorator
    prange = range


@njit(cache=True, parallel=True, fastmath=False)
def _paint_kernel(hap_array, block_probs, num_samples):
    """JIT inner kernel for _paint_samples_fast.
    
    For each sample, iterate over all (i, j) pairs with j>=i, compute
    log-likelihood log(max(block_probs[si, s, hap_i[s]+hap_j[s]], 1e-10))
    summed over sites, and track the pair with maximum likelihood.
    Strict > comparison matches the original tie-break rule (earlier pair
    wins on tie, since earlier pair is processed first).
    
    After picking each sample's best pair, computes per-site residual
    1.0 - block_probs[si, s, expected[s]] summed over sites and divided
    by n_sites.
    
    Args:
        hap_array: (K, n_sites) int8.
        block_probs: (num_samples, n_sites, 3) float64.
        num_samples: int.
    
    Returns:
        (best_pair_idx, best_ll, residuals, pair_i, pair_j) — see wrapper
        for return value mapping.  pair_i and pair_j are returned so the
        wrapper can decode best_pair_idx into (i, j) tuples without
        re-deriving the iteration order.
    """
    K, n_sites = hap_array.shape
    n_pairs = K * (K + 1) // 2
    
    pair_i = np.empty(n_pairs, dtype=np.int64)
    pair_j = np.empty(n_pairs, dtype=np.int64)
    pi = 0
    for i in range(K):
        for j in range(i, K):
            pair_i[pi] = i
            pair_j[pi] = j
            pi += 1
    
    best_pair_idx = np.zeros(num_samples, dtype=np.int64)
    best_ll = np.full(num_samples, -np.inf, dtype=np.float64)
    residuals = np.zeros(num_samples, dtype=np.float64)
    
    # Per-sample parallelism: each thread fully processes its samples,
    # so there are no shared writes (lock-free).  The within-sample
    # iteration order matches the original Python (pair index outer,
    # site index inner), which preserves the per-sample accumulation
    # order — important for byte-identity of best_pair selection.
    for si in prange(num_samples):
        local_best_ll = -np.inf
        local_best_pi = 0
        
        for ppi in range(n_pairs):
            ii = pair_i[ppi]
            jj = pair_j[ppi]
            ll = 0.0
            for s in range(n_sites):
                # expected = hap_array[ii, s] + hap_array[jj, s]; values in {0,1,2}
                ee = hap_array[ii, s] + hap_array[jj, s]
                p = block_probs[si, s, ee]
                if p < 1e-10:
                    p = 1e-10
                ll += math.log(p)
            if ll > local_best_ll:
                local_best_ll = ll
                local_best_pi = ppi
        
        best_ll[si] = local_best_ll
        best_pair_idx[si] = local_best_pi
        
        # Residual pass for this sample, using its winning pair.
        ii = pair_i[local_best_pi]
        jj = pair_j[local_best_pi]
        r = 0.0
        for s in range(n_sites):
            ee = hap_array[ii, s] + hap_array[jj, s]
            r += 1.0 - block_probs[si, s, ee]
        residuals[si] = r / n_sites
    
    return best_pair_idx, best_ll, residuals, pair_i, pair_j


@njit(cache=True, parallel=True, fastmath=False)
def _extract_missing_hap_kernel(hap_array, block_probs, sample_indices, partner_indices):
    """JIT inner kernel for _extract_missing_hap.
    
    For each sample carrying the missing founder, partition the genotype
    probability mass between (partner_allele) and (partner_allele + 1):
        p0 = block_probs[si, s, a]      (sample_allele == partner_allele)
        p1 = block_probs[si, s, a + 1]  (sample_allele == partner_allele + 1)
    Renormalise (p0, p1) and accumulate as soft votes for sites where
    sample_allele matches/differs from partner_allele.  Final output is
    1 at sites where votes_1 > votes_0 (strict, not >=, matching original).
    
    Site-level parallelism (each site has its own slot in the vote arrays).
    """
    K, n_sites = hap_array.shape
    n_assigned = sample_indices.shape[0]
    
    allele_votes_0 = np.zeros(n_sites, dtype=np.float64)
    allele_votes_1 = np.zeros(n_sites, dtype=np.float64)
    
    for s in prange(n_sites):
        v0 = 0.0
        v1 = 0.0
        for idx in range(n_assigned):
            si = sample_indices[idx]
            partner_ki = partner_indices[idx]
            a = hap_array[partner_ki, s]
            p0 = block_probs[si, s, a]
            # Clamp a+1 at 2 (since alleles are 0/1, but if a==1 then a+1==2 is valid;
            # if a==0 then a+1==1; if a==2 (shouldn't happen for hap_array) then a+1 stays 2)
            a1 = a + 1
            if a1 > 2:
                a1 = 2
            p1 = block_probs[si, s, a1]
            total = p0 + p1
            if total > 1e-10:
                v0 += p0 / total
                v1 += p1 / total
        allele_votes_0[s] = v0
        allele_votes_1[s] = v1
    
    new_hap = np.zeros(n_sites, dtype=np.int8)
    for s in range(n_sites):
        if allele_votes_1[s] > allele_votes_0[s]:
            new_hap[s] = 1
    return new_hap


@njit(cache=True, parallel=True, fastmath=False)
def _find_partners_kernel(hap_array, block_probs, sample_indices):
    """JIT inner kernel for the partner-finding loop in
    _residual_discover_core and _seeded_discover_from_arrays.
    
    For each sample in sample_indices, find the haplotype index ki in
    hap_array maximising sum over sites of log(max(p0 + p1, 1e-10)),
    where p0 and p1 are as in _extract_missing_hap_kernel.
    """
    K, n_sites = hap_array.shape
    n_samples = sample_indices.shape[0]
    
    out = np.zeros(n_samples, dtype=np.int64)
    
    for idx in prange(n_samples):
        si = sample_indices[idx]
        best_ki = -1
        best_ll = -np.inf
        for ki in range(K):
            ll = 0.0
            for s in range(n_sites):
                a = hap_array[ki, s]
                p0 = block_probs[si, s, a]
                a1 = a + 1
                if a1 > 2:
                    a1 = 2
                p1 = block_probs[si, s, a1]
                total = p0 + p1
                if total < 1e-10:
                    total = 1e-10
                ll += math.log(total)
            if ll > best_ll:
                best_ll = ll
                best_ki = ki
        out[idx] = best_ki
    return out


# =============================================================================
# MULTIPROCESSING INFRASTRUCTURE
# =============================================================================

try:
    _forkserver_ctx = mp.get_context('forkserver')
except ValueError:
    _forkserver_ctx = mp.get_context('fork')


class _ForkserverPool(multiprocessing.pool.Pool):
    """A Pool using forkserver context."""
    def __init__(self, *args, **kwargs):
        kwargs['context'] = _forkserver_ctx
        super().__init__(*args, **kwargs)


# Worker globals — set by _init_pass1_worker, used by _pass1_worker
_RD_SHM_REF = None          # SharedMemory handle (kept alive for worker lifetime)
_RD_GLOBAL_PROBS = None     # numpy view into shared memory
_RD_NUM_SAMPLES = None


def _init_pass1_worker(probs_shm_name, probs_shape, probs_dtype, num_samples):
    """Initializer for Pass 1 pool workers.
    
    Attaches to shared memory containing global_probs once per worker
    (not per task), avoiding repeated attach/detach overhead.
    Also caps numba threads to 1 to prevent oversubscription from
    imported modules.
    """
    global _RD_SHM_REF, _RD_GLOBAL_PROBS, _RD_NUM_SAMPLES
    _RD_SHM_REF = SharedMemory(name=probs_shm_name, create=False)
    _RD_GLOBAL_PROBS = np.ndarray(probs_shape, dtype=probs_dtype, buffer=_RD_SHM_REF.buf)
    _RD_NUM_SAMPLES = num_samples
    try:
        import numba
        numba.set_num_threads(1)
    except Exception:
        pass


def _pass1_worker(args):
    """Process a single block for Pass 1 residual discovery.
    
    Slices block_probs from shared global_probs and runs the core
    residual discovery algorithm. Returns (bi, new_hap, n_assigned,
    assigned_set) or None if no discovery.
    """
    bi, hap_array, site_indices, min_assigned = args
    block_probs = _RD_GLOBAL_PROBS[:, site_indices, :]
    
    new_hap, n_assigned, assigned_set = _residual_discover_core(
        hap_array, block_probs, _RD_NUM_SAMPLES)
    
    if new_hap is not None and n_assigned >= min_assigned:
        return (bi, new_hap, n_assigned, assigned_set)
    return None


# -----------------------------------------------------------------------------
# Pass 3/4/5 worker — operates on pre-extracted arrays so the BlockResult
# object never has to be pickled to the worker.  Takes a snapshot of the
# accepted set's seed_samples (looked up by the main process) so workers
# don't need access to the full accepted dict.
# -----------------------------------------------------------------------------

def _pass345_worker(args):
    """Process a single block for Pass 3, 4, or 5.
    
    Args is a tuple whose first element is a mode string:
      'qc_only': args = (mode, bi, base_hap_array, hap_dict_2d, site_indices,
                          new_hap, min_rr, min_assigned)
                 — Pass 3 path: new_hap precomputed, just run quality check.
      'discover_qc': args = (mode, bi, base_hap_array, hap_dict_2d, site_indices,
                              seed_samples, min_rr, min_assigned, overlap_threshold)
                 — Pass 4 / Pass 5 path: run _seeded_discover, then overlap
                   filter, then _quality_check.
    
    Returns one of:
      ('accepted', bi, new_hap, assigned_set)  — fully passed all filters
      ('rejected', bi, reason)                 — failed at some filter
      None                                     — _seeded_discover produced no candidate
    
    The hap_dict_2d argument is the original block.haplotypes dict (mapping
    haplotype-key -> (n_sites, 2) float32 probabilistic array), which
    _quality_check_from_arrays needs to feed into prune_chimeras.  It's
    pickled to the worker; for K~8 haps and n_sites=200 that's ~12 KB —
    same order of magnitude as the existing Pass 1 task payload.
    """
    mode = args[0]
    if mode == 'qc_only':
        _, bi, base_hap_array, hap_dict_2d, site_indices, new_hap, min_rr, min_assigned = args
        block_probs = _RD_GLOBAL_PROBS[:, site_indices, :]
        passed, rr, reason = _quality_check_from_arrays(
            base_hap_array, hap_dict_2d, block_probs, new_hap,
            _RD_NUM_SAMPLES, min_rr)
        if passed:
            # For 'qc_only' (Pass 3) the assigned_set is provided by the caller
            # via the precomputed raw_discoveries entry, but we don't have it
            # here in the worker — Pass 3 main loop will look it up after
            # receiving the result. Return None for assigned_set as a sentinel.
            return ('accepted', bi, new_hap, None)
        return ('rejected', bi, reason)
    
    if mode == 'discover_qc':
        (_, bi, base_hap_array, hap_dict_2d, site_indices,
         seed_samples, min_rr, min_assigned, overlap_threshold) = args
        block_probs = _RD_GLOBAL_PROBS[:, site_indices, :]
        
        new_hap, n_assigned, assigned_set = _seeded_discover_from_arrays(
            base_hap_array, block_probs, seed_samples, _RD_NUM_SAMPLES)
        
        if new_hap is None or n_assigned < min_assigned:
            return None
        
        if _sample_overlap(assigned_set, seed_samples) < overlap_threshold:
            return ('rejected', bi, 'low_overlap')
        
        passed, rr, reason = _quality_check_from_arrays(
            base_hap_array, hap_dict_2d, block_probs, new_hap,
            _RD_NUM_SAMPLES, min_rr)
        if not passed:
            return ('rejected', bi, reason)
        return ('accepted', bi, new_hap, assigned_set)
    
    raise ValueError(f"_pass345_worker: unknown mode {mode!r}")


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def _get_block_hap_array(block):
    """Extract concretised haplotype array from a BlockResult.
    
    Returns:
        (hap_array, hap_keys) where hap_array is (K, n_sites) int8.
    """
    known_haps = {}
    for h_key, hap in block.haplotypes.items():
        if hap.ndim > 1:
            known_haps[h_key] = np.argmax(hap, axis=1)
        else:
            known_haps[h_key] = hap.copy()
    hap_keys = sorted(known_haps.keys())
    return np.array([known_haps[k] for k in hap_keys]), hap_keys


def _paint_samples_fast(hap_array, block_probs, num_samples):
    """Paint samples against known haplotypes using diploid likelihood.
    
    For each sample, finds the best diploid pair (i, j) and computes
    a per-site residual measuring unexplained signal.
    
    Args:
        hap_array: (K, n_sites) concretised haplotypes.
        block_probs: (num_samples, n_sites, 3) genotype probabilities.
        num_samples: Number of samples.
    
    Returns:
        (best_pairs_decoded, best_ll, residuals)
        best_pairs_decoded: List of (i, j) tuples per sample.
        best_ll: (num_samples,) log-likelihood of best pair.
        residuals: (num_samples,) mean per-site residual.
    """
    if HAS_NUMBA:
        # JIT path: cast inputs to expected dtypes and call kernel.
        # block_probs cast to float64 means math is f64 throughout
        # (slightly more precise than the original which silently used
        # f32 for log when block_probs was f32).
        if hap_array.dtype != np.int8:
            hap_array_k = hap_array.astype(np.int8)
        else:
            hap_array_k = hap_array
        if block_probs.dtype != np.float64:
            block_probs_k = block_probs.astype(np.float64)
        else:
            block_probs_k = block_probs
        
        best_pair_idx, best_ll, residuals, pair_i, pair_j = _paint_kernel(
            hap_array_k, block_probs_k, num_samples)
        
        best_pairs_decoded = [
            (int(pair_i[best_pair_idx[si]]), int(pair_j[best_pair_idx[si]]))
            for si in range(num_samples)
        ]
        return best_pairs_decoded, best_ll, residuals
    
    # ---- Pure-Python fallback (used when numba is not available) ----
    # Cast block_probs to float64 so the fallback produces byte-identical
    # output to the JIT path.  Without this cast, when block_probs is
    # float32, np.log(np.maximum(...)) operates in float32 and the
    # subsequent ll += ... mixes f32 and f64 — slightly different
    # numerics from the JIT kernel, which works in f64 throughout.
    if block_probs.dtype != np.float64:
        block_probs = block_probs.astype(np.float64)
    K, n_sites = hap_array.shape
    
    pairs = []
    for i in range(K):
        for j in range(i, K):
            pairs.append((i, j))
    
    best_pair = np.zeros(num_samples, dtype=int)
    best_ll = np.full(num_samples, -np.inf)
    
    for pi, (i, j) in enumerate(pairs):
        expected = hap_array[i] + hap_array[j]
        ll = np.zeros(num_samples)
        for s in range(n_sites):
            ll += np.log(np.maximum(block_probs[:, s, expected[s]], 1e-10))
        better = ll > best_ll
        best_ll[better] = ll[better]
        best_pair[better] = pi
    
    residuals = np.zeros(num_samples)
    for si in range(num_samples):
        pi = best_pair[si]
        i, j = pairs[pi]
        expected = hap_array[i] + hap_array[j]
        for s in range(n_sites):
            residuals[si] += 1.0 - block_probs[si, s, expected[s]]
    residuals /= n_sites
    
    best_pairs_decoded = [pairs[best_pair[si]] for si in range(num_samples)]
    return best_pairs_decoded, best_ll, residuals


def _extract_missing_hap(hap_array, block_probs, sample_indices, partner_indices):
    """Extract a missing haplotype by subtracting known partners.
    
    For each high-residual sample, we know it carries the missing founder
    paired with a known partner. We subtract the partner's contribution
    to recover the missing founder's alleles via soft voting.
    
    Args:
        hap_array: (K, n_sites) known haplotypes.
        block_probs: (num_samples, n_sites, 3) genotype probabilities.
        sample_indices: Indices of samples carrying the missing founder.
        partner_indices: Index into hap_array of each sample's known partner.
    
    Returns:
        (n_sites,) int8 array of the extracted haplotype.
    """
    if HAS_NUMBA:
        # JIT path.
        if hap_array.dtype != np.int8:
            hap_array_k = hap_array.astype(np.int8)
        else:
            hap_array_k = hap_array
        if block_probs.dtype != np.float64:
            block_probs_k = block_probs.astype(np.float64)
        else:
            block_probs_k = block_probs
        sample_indices_k = np.asarray(sample_indices, dtype=np.int64)
        partner_indices_k = np.asarray(partner_indices, dtype=np.int64)
        return _extract_missing_hap_kernel(
            hap_array_k, block_probs_k, sample_indices_k, partner_indices_k)
    
    # ---- Pure-Python fallback ----
    # Cast block_probs to float64 so the fallback produces byte-identical
    # output to the JIT path.  See note in _paint_samples_fast above.
    if block_probs.dtype != np.float64:
        block_probs = block_probs.astype(np.float64)
    K, n_sites = hap_array.shape
    allele_votes_0 = np.zeros(n_sites, dtype=np.float64)
    allele_votes_1 = np.zeros(n_sites, dtype=np.float64)
    
    for idx, si in enumerate(sample_indices):
        partner = hap_array[partner_indices[idx]]
        for s in range(n_sites):
            a = partner[s]
            p0 = block_probs[si, s, a]
            p1 = block_probs[si, s, min(a + 1, 2)]
            total = p0 + p1
            if total > 1e-10:
                allele_votes_0[s] += p0 / total
                allele_votes_1[s] += p1 / total
    
    new_hap = (allele_votes_1 > allele_votes_0).astype(np.int8)
    return new_hap


def _find_partners(hap_array, block_probs, sample_indices):
    """Find each sample's best-fit partner haplotype index.
    
    For each sample in sample_indices, identifies the haplotype index ki
    in hap_array that maximises sum over sites of log(max(p0 + p1, 1e-10)),
    where p0 = block_probs[si, s, hap_array[ki, s]] and p1 = block_probs[
    si, s, min(hap_array[ki, s] + 1, 2)].  This is the per-sample partner
    score used by _residual_discover_core, _seeded_discover, and
    _seeded_discover_from_arrays.
    
    Args:
        hap_array: (K, n_sites) int8 known haplotypes.
        block_probs: (num_samples, n_sites, 3) genotype probabilities.
        sample_indices: iterable of sample indices to score.
    
    Returns:
        list[int] of partner indices (parallel to sample_indices).
        Returned as a Python list to match the call sites that
        previously built up `partner_indices = []` then `.append(...)`.
    """
    if HAS_NUMBA:
        if hap_array.dtype != np.int8:
            hap_array_k = hap_array.astype(np.int8)
        else:
            hap_array_k = hap_array
        if block_probs.dtype != np.float64:
            block_probs_k = block_probs.astype(np.float64)
        else:
            block_probs_k = block_probs
        sample_indices_k = np.asarray(sample_indices, dtype=np.int64)
        return _find_partners_kernel(
            hap_array_k, block_probs_k, sample_indices_k).tolist()
    
    # ---- Pure-Python fallback (matches the original inlined loops
    # in _residual_discover_core / _seeded_discover / _seeded_discover_from_arrays) ----
    # Cast block_probs to float64 so the fallback produces byte-identical
    # output to the JIT path.  See note in _paint_samples_fast above.
    if block_probs.dtype != np.float64:
        block_probs = block_probs.astype(np.float64)
    K, n_sites = hap_array.shape
    partner_indices = []
    for si in sample_indices:
        best_ki = -1
        best_ki_ll = -np.inf
        for ki in range(K):
            partner = hap_array[ki]
            ll = 0.0
            for s in range(n_sites):
                a = partner[s]
                p0 = block_probs[si, s, a]
                p1 = block_probs[si, s, min(a + 1, 2)]
                ll += np.log(max(p0 + p1, 1e-10))
            if ll > best_ki_ll:
                best_ki_ll = ll
                best_ki = ki
        partner_indices.append(best_ki)
    return partner_indices


def _iterative_refine(base_hap_array, new_hap, block_probs, num_samples, max_rounds=10):
    """Iteratively refine a discovered haplotype.
    
    After initial extraction, re-paint samples with the expanded set
    (known + new), re-identify which samples use the new hap, and
    re-extract. Converges in 2-5 rounds typically.
    
    Returns:
        (refined_hap, n_assigned, assigned_set)
    """
    K_base = base_hap_array.shape[0]
    assigned_set = set()
    
    for round_i in range(max_rounds):
        expanded = np.vstack([base_hap_array, new_hap.reshape(1, -1)])
        new_idx = K_base
        
        best_pairs, _, _ = _paint_samples_fast(expanded, block_probs, num_samples)
        
        assigned_samples = []
        assigned_partners = []
        for si in range(num_samples):
            i, j = best_pairs[si]
            if i == new_idx or j == new_idx:
                partner = j if i == new_idx else i
                assigned_samples.append(si)
                assigned_partners.append(partner)
        
        if len(assigned_samples) < 3:
            break
        
        prev_hap = new_hap.copy()
        new_hap = _extract_missing_hap(
            expanded, block_probs, assigned_samples, assigned_partners)
        assigned_set = set(assigned_samples)
        
        if np.sum(new_hap != prev_hap) == 0:
            break
    
    return new_hap, len(assigned_set), assigned_set


def _residual_discover_core(base_hap_array, block_probs, num_samples):
    """Core residual discovery logic operating on pre-concretised arrays.
    
    Identifies samples with unusually high residuals (MAD-based outlier
    detection), finds their best-fit known partner, subtracts to extract
    the missing haplotype, then iteratively refines.
    
    Args:
        base_hap_array: (K, n_sites) int8 concretised haplotypes.
        block_probs: (num_samples, n_sites, 3) genotype probabilities.
        num_samples: Number of samples.
    
    Returns:
        (new_hap, n_assigned, assigned_set) or (None, 0, set()) if no signal.
    """
    K_base = base_hap_array.shape[0]
    n_sites = block_probs.shape[1]
    
    best_pairs, best_ll, residuals = _paint_samples_fast(
        base_hap_array, block_probs, num_samples)
    
    # MAD-based outlier detection for high-residual samples
    median_res = np.median(residuals)
    mad = np.median(np.abs(residuals - median_res))
    threshold = median_res + 3 * max(mad, 0.01)
    high_indices = np.where(residuals > threshold)[0]
    
    if len(high_indices) < 5:
        return None, 0, set()
    
    # Find best partner for each high-residual sample
    partner_indices = _find_partners(base_hap_array, block_probs, high_indices)
    
    new_hap = _extract_missing_hap(
        base_hap_array, block_probs, high_indices, partner_indices)
    new_hap, n_assigned, assigned_set = _iterative_refine(
        base_hap_array, new_hap, block_probs, num_samples)
    return new_hap, n_assigned, assigned_set


def _residual_discover(block, block_probs, num_samples):
    """Attempt to discover a missing haplotype via residual analysis.
    
    Wrapper that extracts concretised hap array from the block,
    then delegates to _residual_discover_core.
    
    Returns:
        (new_hap, n_assigned, assigned_set) or (None, 0, set()) if no signal.
    """
    base_hap_array, hap_keys = _get_block_hap_array(block)
    return _residual_discover_core(base_hap_array, block_probs, num_samples)


def _seeded_discover(block, block_probs, seed_samples, num_samples):
    """Discover a missing haplotype using seed samples from a neighbour block.
    
    Instead of MAD-based outlier detection, uses a pre-identified set of
    samples known to carry the missing founder in adjacent blocks.
    
    Returns:
        (new_hap, n_assigned, assigned_set) or (None, 0, set()) if too few seeds.
    """
    base_hap_array, hap_keys = _get_block_hap_array(block)
    K_base = len(hap_keys)
    n_sites = block_probs.shape[1]
    
    seed_list = sorted(seed_samples)
    if len(seed_list) < 3:
        return None, 0, set()
    
    partner_indices = _find_partners(base_hap_array, block_probs, seed_list)
    
    new_hap = _extract_missing_hap(
        base_hap_array, block_probs, seed_list, partner_indices)
    new_hap, n_assigned, assigned_set = _iterative_refine(
        base_hap_array, new_hap, block_probs, num_samples)
    return new_hap, n_assigned, assigned_set


def _sample_overlap(set_a, set_b):
    """Jaccard-like overlap: intersection / min(|A|, |B|)."""
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / min(len(set_a), len(set_b))


def _quality_check(block, block_probs, new_hap, num_samples, min_rr):
    """Check whether a discovered haplotype passes quality filters.
    
    Two checks:
      1. Residual reduction ≥ min_rr (adding the hap must meaningfully
         reduce total sample residuals)
      2. prune_chimeras survival (the hap must not be explainable as
         a recombination of existing haps)
    
    Returns:
        (passed, residual_reduction, reason_if_failed)
    """
    base_hap_array, hap_keys = _get_block_hap_array(block)
    n_sites = block_probs.shape[1]
    
    # Residual reduction check
    _, _, base_residuals = _paint_samples_fast(base_hap_array, block_probs, num_samples)
    base_total = np.sum(base_residuals)
    
    expanded_array = np.vstack([base_hap_array, new_hap.reshape(1, -1)])
    _, _, new_residuals = _paint_samples_fast(expanded_array, block_probs, num_samples)
    new_total = np.sum(new_residuals)
    
    rr = base_total - new_total
    
    if rr < min_rr:
        return False, rr, 'low_residual_reduction'
    
    # Structural chimera pruning check
    expanded_hap_dict = {}
    for ki, k in enumerate(hap_keys):
        expanded_hap_dict[ki] = block.haplotypes[k]
    
    new_hap_2d = np.zeros((n_sites, 2), dtype=np.float32)
    new_hap_2d[np.arange(n_sites), new_hap] = 1.0
    expanded_hap_dict[len(hap_keys)] = new_hap_2d
    
    pruned = block_haplotypes.prune_chimeras(
        expanded_hap_dict, block_probs,
        max_recombs=1, max_mismatch_percent=0.5,
        min_mean_delta_to_protect=0.25
    )
    
    survived = len(pruned) > len(hap_keys)
    
    if not survived:
        return False, rr, 'pruned'
    
    return True, rr, None


# -----------------------------------------------------------------------------
# Array-based variants of _seeded_discover and _quality_check.  These accept
# the pre-extracted hap_array and hap_dict_2d directly instead of pulling
# them off a BlockResult, which lets us pickle a tiny payload to a worker
# process instead of the full BlockResult object.  The math is byte-identical
# to _seeded_discover / _quality_check above; only the input plumbing differs.
# -----------------------------------------------------------------------------

def _seeded_discover_from_arrays(base_hap_array, block_probs, seed_samples, num_samples):
    """Same algorithm as _seeded_discover, but takes base_hap_array directly
    instead of pulling it off a BlockResult via _get_block_hap_array.
    
    Returns:
        (new_hap, n_assigned, assigned_set) or (None, 0, set()) if too few seeds.
    """
    K_base = base_hap_array.shape[0]
    n_sites = block_probs.shape[1]
    
    seed_list = sorted(seed_samples)
    if len(seed_list) < 3:
        return None, 0, set()
    
    partner_indices = _find_partners(base_hap_array, block_probs, seed_list)
    
    new_hap = _extract_missing_hap(
        base_hap_array, block_probs, seed_list, partner_indices)
    new_hap, n_assigned, assigned_set = _iterative_refine(
        base_hap_array, new_hap, block_probs, num_samples)
    return new_hap, n_assigned, assigned_set


def _quality_check_from_arrays(base_hap_array, hap_dict_2d, block_probs,
                                new_hap, num_samples, min_rr):
    """Same algorithm as _quality_check, but takes base_hap_array and the
    probabilistic 2D-form hap_dict_2d directly instead of pulling them off
    a BlockResult.
    
    hap_dict_2d should be {key: (n_sites, 2) float array} matching the
    layout of block.haplotypes for each key whose concretised form is
    represented in base_hap_array.  Iteration order over hap_dict_2d
    must match the row order of base_hap_array (i.e. sorted by key, as
    _get_block_hap_array does).
    
    Returns:
        (passed, residual_reduction, reason_if_failed)
    """
    n_sites = block_probs.shape[1]
    
    # Residual reduction check
    _, _, base_residuals = _paint_samples_fast(base_hap_array, block_probs, num_samples)
    base_total = np.sum(base_residuals)
    
    expanded_array = np.vstack([base_hap_array, new_hap.reshape(1, -1)])
    _, _, new_residuals = _paint_samples_fast(expanded_array, block_probs, num_samples)
    new_total = np.sum(new_residuals)
    
    rr = base_total - new_total
    
    if rr < min_rr:
        return False, rr, 'low_residual_reduction'
    
    # Structural chimera pruning check.
    # hap_dict_2d's keys must be iterated in the same order as
    # _get_block_hap_array would have produced (sorted), since
    # base_hap_array's row i corresponds to that order.
    hap_keys = sorted(hap_dict_2d.keys())
    expanded_hap_dict = {}
    for ki, k in enumerate(hap_keys):
        expanded_hap_dict[ki] = hap_dict_2d[k]
    
    new_hap_2d = np.zeros((n_sites, 2), dtype=np.float32)
    new_hap_2d[np.arange(n_sites), new_hap] = 1.0
    expanded_hap_dict[len(hap_keys)] = new_hap_2d
    
    pruned = block_haplotypes.prune_chimeras(
        expanded_hap_dict, block_probs,
        max_recombs=1, max_mismatch_percent=0.5,
        min_mean_delta_to_protect=0.25
    )
    
    survived = len(pruned) > len(hap_keys)
    
    if not survived:
        return False, rr, 'pruned'
    
    return True, rr, None


def _add_hap_to_block(block, new_hap):
    """Add a discovered haplotype to a block, returning a new BlockResult.
    
    The new hap gets the next available integer key. The haplotype is
    stored in probabilistic (n_sites, 2) format to match existing haps.
    """
    n_sites = len(block.positions)
    existing_keys = sorted(block.haplotypes.keys())
    new_key = max(existing_keys) + 1 if existing_keys else 0
    
    new_hap_2d = np.zeros((n_sites, 2), dtype=np.float32)
    new_hap_2d[np.arange(n_sites), new_hap] = 1.0
    
    new_haps = dict(block.haplotypes)
    new_haps[new_key] = new_hap_2d
    
    return block_haplotypes.BlockResult(
        positions=block.positions,
        haplotypes=new_haps,
        keep_flags=block.keep_flags,
        reads_count_matrix=getattr(block, 'reads_count_matrix', None),
        probs_array=block.probs_array,
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def _find_seed_nbi(bi, accepted, neighbour_range):
    """Return the bi-index of bi's nearest accepted neighbour within
    neighbour_range (with tie-break preferring the lower-indexed side
    via the [bi-dist, bi+dist] iteration order), or None if none.
    
    This must match the seed-lookup loop used inline in the original
    sequential code so that 'newly eligible' tracking is correct.
    """
    for dist in range(1, neighbour_range + 1):
        for nbi in (bi - dist, bi + dist):
            if nbi in accepted:
                return nbi
    return None


def _run_seeded_subiter(*, eligible_blocks, blocks, block_data, accepted,
                          global_probs, num_samples, min_assigned,
                          overlap_threshold, min_residual_reduction,
                          neighbour_range, pool):
    """Run ONE sub-iteration of seeded recovery against a snapshot of
    `accepted`.  This is the inner step of Option (iii): the caller
    decides which blocks are "eligible" for re-evaluation in this
    sub-iter (i.e. blocks whose seed-neighbour just became more
    favourable) and passes them in.  This function takes a fresh
    snapshot of `accepted` at entry, dispatches the eligible blocks
    in parallel, merges new accepts back in deterministic order, and
    returns the set of newly-accepted block indices plus diagnostic
    stats.
    
    Snapshot semantics within a sub-iter: blocks that get accepted in
    this sub-iter do NOT propagate to other blocks in the same sub-iter
    (only in the NEXT sub-iter).  This is the same parallelism-friendly
    rule as Variant A, but applied at sub-iter granularity rather than
    round granularity, so the wave of propagation completes within a
    single Pass-5-style outer loop.
    
    Args:
        eligible_blocks: iterable of unaccepted block indices to evaluate
            in this sub-iter.  Caller is responsible for computing this
            set (see _compute_newly_eligible).
        blocks: BlockResults — only used in sequential fallback to look
            up block.haplotypes for _quality_check.
        block_data: list of (base_hap_array, hap_dict_2d, site_indices)
            tuples (or None) per block.
        accepted: dict mapping bi -> {'new_hap': ..., 'assigned_samples': ...}.
            MUTATED in place to add new recoveries from this sub-iter.
        global_probs, num_samples: as in discover_missing_haplotypes.
        min_assigned, overlap_threshold, min_residual_reduction,
        neighbour_range: algorithm parameters.
        pool: a Pool with _init_pass1_worker initializer, or None.
    
    Returns:
        (new_accepts: set[int], stats: dict).
        new_accepts is the set of block indices newly added to `accepted`
        in this sub-iter.  stats has the same keys as before:
        visited, seeded, discover_calls, overlap_pass, qc_calls,
        t_discover, t_qc.
    """
    stats = {
        'visited': 0, 'seeded': 0, 'discover_calls': 0,
        'overlap_pass': 0, 'qc_calls': 0,
        't_discover': 0.0, 't_qc': 0.0,
    }
    new_accepts = set()
    
    # Snapshot accepted at sub-iter entry.  All seed lookups read from
    # this snapshot; merges into `accepted` happen after the dispatch
    # phase completes.  Equivalent to the Variant-A round snapshot, just
    # at finer granularity.
    accepted_snapshot = dict(accepted)
    
    # Phase 1: build task list from eligible blocks.  We re-derive each
    # block's seed_samples from the snapshot here (the caller's
    # eligibility logic told us "this block's seed changed", but we
    # still need to look up the actual seed value).
    task_list = []
    for bi in eligible_blocks:
        if bi in accepted_snapshot:
            # Caller shouldn't have included this, but be defensive.
            continue
        stats['visited'] += 1
        
        if block_data[bi] is None:
            continue
        
        nbi = _find_seed_nbi(bi, accepted_snapshot, neighbour_range)
        if nbi is None:
            # Eligibility tracking says this block had a seed; if it
            # doesn't now, something went wrong in tracking.  Defensive
            # skip.
            continue
        seed_samples = accepted_snapshot[nbi]['assigned_samples']
        if len(seed_samples) < min_assigned:
            continue
        stats['seeded'] += 1
        
        base_hap_array, hap_dict_2d, site_indices = block_data[bi]
        task_list.append((
            'discover_qc', bi, base_hap_array, hap_dict_2d, site_indices,
            seed_samples, min_residual_reduction, min_assigned,
            overlap_threshold,
        ))
    
    if not task_list:
        return new_accepts, stats
    
    # Phase 2: dispatch (parallel or sequential).
    results = []
    if pool is not None:
        # Wall-time deltas for the parallel path can't separate
        # _seeded_discover time from _quality_check time at the
        # main-process level (the worker does both inside one task).
        # We attribute the entire sub-iter wall to t_discover for
        # display purposes; the round-total wall is computed by the
        # caller independently and remains accurate.
        _t_subiter_dispatch = time.time()
        for r in pool.imap_unordered(_pass345_worker, task_list, chunksize=4):
            stats['discover_calls'] += 1
            if r is None:
                continue
            results.append(r)
            if r[0] == 'rejected' and r[2] == 'low_overlap':
                # discover ran but overlap filter rejected — count discover only
                continue
            # Anything else means overlap passed and qc ran
            stats['overlap_pass'] += 1
            stats['qc_calls'] += 1
        _t_subiter = time.time() - _t_subiter_dispatch
        stats['t_discover'] = _t_subiter  # rough — see comment
        stats['t_qc'] = 0.0
    else:
        # Sequential fallback — same algorithm, can measure precisely.
        for args in task_list:
            (_, bi, base_hap_array, hap_dict_2d, site_indices,
             seed_samples, min_rr, _, ovt) = args
            block_probs = global_probs[:, site_indices, :]
            
            _t = time.time()
            new_hap, n_assigned, assigned_set = _seeded_discover_from_arrays(
                base_hap_array, block_probs, seed_samples, num_samples)
            stats['t_discover'] += time.time() - _t
            stats['discover_calls'] += 1
            
            if new_hap is None or n_assigned < min_assigned:
                continue
            
            if _sample_overlap(assigned_set, seed_samples) < ovt:
                continue
            stats['overlap_pass'] += 1
            
            _t = time.time()
            passed, rr, reason = _quality_check_from_arrays(
                base_hap_array, hap_dict_2d, block_probs, new_hap,
                num_samples, min_rr)
            stats['t_qc'] += time.time() - _t
            stats['qc_calls'] += 1
            if not passed:
                continue
            
            results.append(('accepted', bi, new_hap, assigned_set))
    
    # Phase 3: merge accepted results into `accepted` dict in
    # deterministic (sorted-by-bi) order so the final state is
    # independent of worker scheduling order.
    results.sort(key=lambda r: r[1])
    for r in results:
        if r[0] != 'accepted':
            continue
        _, bi, new_hap, assigned_set = r
        if assigned_set is None:
            # Should not happen for 'discover_qc' mode (only for 'qc_only'
            # which this function doesn't dispatch), but guard anyway.
            continue
        accepted[bi] = {
            'new_hap': new_hap,
            'assigned_samples': assigned_set,
        }
        new_accepts.add(bi)
    
    return new_accepts, stats


def _compute_newly_eligible(*, new_accepts, accepted, last_seed_nbi,
                              n_blocks, neighbour_range):
    """Compute the set of unaccepted blocks whose seed-neighbour just
    changed as a result of `new_accepts` being added to `accepted`.
    
    A block bi is "newly eligible" iff its nearest accepted neighbour
    (per _find_seed_nbi) just changed identity.  Equivalently:
        last_seed_nbi[bi]  != _find_seed_nbi(bi, accepted, neighbour_range)
        AND _find_seed_nbi(bi, accepted, neighbour_range) is not None
    
    Implementation note: we only need to check candidates within
    [a - neighbour_range, a + neighbour_range] for each new accept `a`,
    because no block outside that window can have its nearest neighbour
    affected by `a`.  This makes the work proportional to
    |new_accepts| * neighbour_range rather than n_blocks per sub-iter.
    
    Side effect: `last_seed_nbi` is UPDATED in place for each block whose
    seed changed, and for any block whose seed transitioned from None to
    a value (first-time eligibility).  Blocks that just got accepted
    are also removed from last_seed_nbi (they're no longer unaccepted).
    
    Args:
        new_accepts: set[int] of block indices added in the just-completed
            sub-iter.
        accepted: dict, the post-merge accepted set.
        last_seed_nbi: dict mapping unaccepted bi -> nbi-index of nearest
            accepted neighbour at last evaluation, or None if no neighbour
            within range. MUTATED.
        n_blocks: total block count.
        neighbour_range: algorithm parameter.
    
    Returns:
        eligible: set[int] of unaccepted bi's whose seed-neighbour changed.
    """
    eligible = set()
    
    # Drop just-accepted blocks from the tracker.
    for a in new_accepts:
        last_seed_nbi.pop(a, None)
    
    # Candidates: unaccepted blocks within neighbour_range of any new accept.
    candidates = set()
    for a in new_accepts:
        for delta in range(1, neighbour_range + 1):
            for c in (a - delta, a + delta):
                if 0 <= c < n_blocks and c not in accepted:
                    candidates.add(c)
    
    for bi in candidates:
        new_nbi = _find_seed_nbi(bi, accepted, neighbour_range)
        old_nbi = last_seed_nbi.get(bi, None)  # may be missing for never-seen blocks
        if new_nbi != old_nbi:
            last_seed_nbi[bi] = new_nbi
            if new_nbi is not None:
                eligible.add(bi)
    
    return eligible


def discover_missing_haplotypes(blocks, global_probs, global_sites,
                                 min_residual_reduction=0.10,
                                 overlap_threshold=0.50,
                                 neighbour_range=5,
                                 min_assigned=5,
                                 max_propagation_rounds=100,
                                 num_processes=1,
                                 verbose=True):
    """
    Discover missing founder haplotypes and add them to blocks.
    
    Runs a 5-pass pipeline:
      Pass 1: Residual discovery on all blocks (MAD outlier detection)
      Pass 2: Consistency filter (assigned samples must overlap with neighbours)
      Pass 3: Quality check (residual reduction + chimera pruning)
      Pass 4: Neighbour-seeded recovery (use accepted neighbours' samples as seeds)
      Pass 5: Propagation (iterate seeded recovery until no new discoveries)
    
    Discovered haplotypes are added to their blocks. Existing dedup and
    chimera pruning in the pipeline handle any duplicates or noise.
    
    PARALLELISM (when num_processes > 1):
      Passes 1, 3, 4, and 5 all run in parallel against a single pool of
      `num_processes` workers, sharing one POSIX shared-memory segment for
      global_probs.  Pass 2 (set ops on raw_discoveries) and the final
      output/dedup/prune phase run in the main process.
      
      Passes 4 and 5 use a hierarchical snapshot scheme.  Conceptually a
      single inner loop of "sub-iterations": each sub-iter snapshots the
      current `accepted` set, dispatches all currently-eligible blocks in
      parallel against that snapshot, merges new accepts back in
      deterministic (sorted-by-bi) order, and computes the next sub-iter's
      eligible set as just those unaccepted blocks whose nearest accepted
      neighbour CHANGED IDENTITY as a result of the merge.  The wave of
      propagation thus completes within Pass 4 + Pass 5 combined; we
      maintain the historical Pass-4 / Pass-5 split in printed output for
      readability (Pass 4 = first sub-iter, Pass 5 = subsequent sub-iters).
      
      The eligibility-diff filter is what makes this efficient: a block
      whose seed neighbour is unchanged across sub-iters is NOT
      re-dispatched (its _seeded_discover and _quality_check would
      produce the same result against the same input).  This eliminates
      the round-by-round redundant re-evaluation that earlier parallel
      designs (Variant A) suffered from.
      
      Snapshot semantics within a sub-iter: blocks accepted in this
      sub-iter do NOT propagate to other blocks in the same sub-iter
      (only in the NEXT one).  This is a one-sub-iter propagation lag
      vs. the original sequential implementation, which is much smaller
      than the per-round lag of earlier parallel designs.  The final
      accepted set converges to the same fixed point modulo this lag.
    
    Args:
        blocks: BlockResults from refinement (or HDBSCAN if no refinement).
        global_probs: (num_samples, num_sites, 3) genotype probability array.
        global_sites: Array of all genomic site positions.
        min_residual_reduction: Minimum total residual decrease to accept a
            discovered haplotype. Default 0.10, validated on chr7/chr23.
        overlap_threshold: Minimum sample overlap with neighbours for
            consistency check. Default 0.50.
        neighbour_range: How many blocks to search for neighbours. Default 5.
        min_assigned: Minimum samples assigned to the new hap. Default 5.
        max_propagation_rounds: Maximum sub-iter count for the Pass 4 + Pass 5
            propagation loop.  Default 100 — generously above any realistic
            propagation depth (chr7 needed ~10-20 sub-iters in practice;
            an entire chromosome of unaccepted blocks could need at most
            ~n_blocks / neighbour_range sub-iters).  Hitting this cap
            indicates either a pathological input or a bug.
        num_processes: Number of parallel workers. Default 1 (sequential
            for all passes). Set to n_cores for full parallelism across
            Passes 1, 3, 4, 5.
        verbose: Print progress and summary.
    
    Returns:
        BlockResults with discovered haplotypes added to relevant blocks.
    """
    t0 = time.time()
    n_blocks = len(blocks)
    num_samples = global_probs.shape[0]
    site_to_idx = {s: idx for idx, s in enumerate(global_sites)}
    
    if verbose:
        print(f"  Residual discovery: {n_blocks} blocks, {num_samples} samples, "
              f"{num_processes} workers")
    
    # =====================================================================
    # Pre-extract per-block data used by Passes 3, 4, 5 workers.
    # This is computed once on the main side and shipped per-task to the
    # worker.  Per-block payload is small (K x n_sites for hap_array plus
    # K x n_sites x 2 for hap_dict_2d ~ 12KB for K=8, n_sites=200).
    # block_data[bi] = (base_hap_array, hap_dict_2d, site_indices) or None
    # if site_indices couldn't be resolved (missing positions).
    # =====================================================================
    block_data = [None] * n_blocks
    for bi in range(n_blocks):
        block = blocks[bi]
        try:
            site_indices = np.array([site_to_idx[p] for p in block.positions])
        except KeyError:
            # A block position not present in global_sites — skip this
            # block from parallel passes (it would have errored sequentially
            # at the same line).  Keep block_data[bi] = None.
            continue
        base_hap_array, _hap_keys = _get_block_hap_array(block)
        # Shallow-copy block.haplotypes so each task sees a stable mapping
        # without pickling the full BlockResult.  Values are numpy arrays
        # which pickle by reference within a single pickle stream.
        hap_dict_2d = dict(block.haplotypes)
        block_data[bi] = (base_hap_array, hap_dict_2d, site_indices)
    
    # =====================================================================
    # Pool + shared-memory lifetime spans Passes 1, 3, 4, 5.
    # Pass 2 runs in main process while pool sits idle (cheap, set ops only).
    # The pool is torn down after Pass 5; the final dedup/prune runs in
    # main process.
    # =====================================================================
    raw_discoveries = {}
    accepted = {}
    reject_counts = {'low_residual_reduction': 0, 'pruned': 0}
    
    pool = None
    shm = None
    try:
        if num_processes > 1:
            # Create shared memory for global_probs (~2 GB for chr3 etc.)
            shm = SharedMemory(create=True, size=global_probs.nbytes)
            shm_probs = np.ndarray(global_probs.shape, dtype=global_probs.dtype,
                                   buffer=shm.buf)
            shm_probs[:] = global_probs
            
            if verbose:
                shm_mb = global_probs.nbytes / (1024 * 1024)
                print(f"    Shared memory: {shm_mb:.0f} MB")
            
            pool = _ForkserverPool(
                processes=num_processes,
                initializer=_init_pass1_worker,
                initargs=(shm.name, global_probs.shape,
                          global_probs.dtype, num_samples))
        
        # =================================================================
        # Pass 1: Residual discovery on all blocks (parallel)
        # =================================================================
        t_pass1_start = time.time()
        
        if pool is not None:
            # Pre-compute per-block task arguments (lightweight: ~3KB each)
            task_args = []
            for bi in range(n_blocks):
                if block_data[bi] is None:
                    continue
                base_hap_array, _, site_indices = block_data[bi]
                task_args.append((bi, base_hap_array, site_indices, min_assigned))
            
            n_done = 0
            for result in pool.imap_unordered(_pass1_worker, task_args,
                                               chunksize=4):
                n_done += 1
                if result is not None:
                    bi, new_hap, n_assigned, assigned_set = result
                    raw_discoveries[bi] = (new_hap, n_assigned, assigned_set)
                if verbose and n_done % 500 == 0:
                    print(f"    Pass 1: {n_done}/{n_blocks} blocks...")
        else:
            # Sequential fallback (num_processes == 1)
            for bi in range(n_blocks):
                block = blocks[bi]
                site_indices = np.array([site_to_idx[p] for p in block.positions])
                block_probs = global_probs[:, site_indices, :]
                
                new_hap, n_assigned, assigned_set = _residual_discover(
                    block, block_probs, num_samples)
                
                if new_hap is not None and n_assigned >= min_assigned:
                    raw_discoveries[bi] = (new_hap, n_assigned, assigned_set)
                
                if verbose and (bi + 1) % 500 == 0:
                    print(f"    Pass 1: {bi+1}/{n_blocks} blocks...")
        
        if verbose:
            t_pass1 = time.time() - t_pass1_start
            print(f"    Pass 1: {len(raw_discoveries)} raw discoveries  [{t_pass1:.1f}s]")
        
        if not raw_discoveries:
            if verbose:
                print(f"    No discoveries — returning blocks unchanged ({time.time()-t0:.1f}s)")
            return blocks
        
        # =================================================================
        # Pass 2: Consistency filter (set ops, runs in main process)
        # =================================================================
        t_pass2_start = time.time()
        consistent = set()
        for bi in raw_discoveries:
            _, _, assigned_a = raw_discoveries[bi]
            for dist in range(1, neighbour_range + 1):
                for nbi in [bi - dist, bi + dist]:
                    if nbi in raw_discoveries:
                        _, _, assigned_b = raw_discoveries[nbi]
                        if _sample_overlap(assigned_a, assigned_b) >= overlap_threshold:
                            consistent.add(bi)
                            break
                if bi in consistent:
                    break
        
        if verbose:
            t_pass2 = time.time() - t_pass2_start
            print(f"    Pass 2: {len(consistent)} consistent (filtered {len(raw_discoveries) - len(consistent)})  [{t_pass2:.1f}s]")
        
        # =================================================================
        # Pass 3: Quality check (residual reduction + chimera pruning)
        # Parallel: each consistent candidate is independent.
        # =================================================================
        t_pass3_start = time.time()
        pass3_qc_calls = 0
        
        if pool is not None and consistent:
            # Build per-candidate task args. Each candidate is an
            # ('qc_only', bi, base_hap_array, hap_dict_2d, site_indices,
            #  new_hap, min_rr, min_assigned) tuple.
            qc_args = []
            qc_consistent_list = []  # parallel list to look up assigned_set after
            for bi in consistent:
                if block_data[bi] is None:
                    continue
                base_hap_array, hap_dict_2d, site_indices = block_data[bi]
                new_hap = raw_discoveries[bi][0]
                qc_args.append(('qc_only', bi, base_hap_array, hap_dict_2d,
                                site_indices, new_hap,
                                min_residual_reduction, min_assigned))
                qc_consistent_list.append(bi)
            
            # Collect results — order doesn't affect correctness here
            # (each bi is independent, no shared state) — but we sort
            # afterwards for deterministic accepted iteration order.
            results = []
            for r in pool.imap_unordered(_pass345_worker, qc_args, chunksize=4):
                if r is not None:
                    results.append(r)
                pass3_qc_calls += 1
            
            # Merge into accepted in sorted-by-bi order
            results.sort(key=lambda x: x[1])
            for r in results:
                if r[0] == 'accepted':
                    _, bi, new_hap, _ = r
                    accepted[bi] = {
                        'new_hap': new_hap,
                        'assigned_samples': raw_discoveries[bi][2],
                    }
                else:
                    _, bi, reason = r
                    if reason in reject_counts:
                        reject_counts[reason] += 1
        else:
            # Sequential fallback
            for bi in consistent:
                block = blocks[bi]
                site_indices = np.array([site_to_idx[p] for p in block.positions])
                block_probs = global_probs[:, site_indices, :]
                new_hap = raw_discoveries[bi][0]
                
                passed, rr, reason = _quality_check(
                    block, block_probs, new_hap, num_samples, min_residual_reduction)
                pass3_qc_calls += 1
                
                if passed:
                    accepted[bi] = {
                        'new_hap': new_hap,
                        'assigned_samples': raw_discoveries[bi][2],
                    }
                elif reason in reject_counts:
                    reject_counts[reason] += 1
        
        if verbose:
            t_pass3 = time.time() - t_pass3_start
            avg_qc_ms = (t_pass3 / pass3_qc_calls * 1000) if pass3_qc_calls else 0.0
            print(f"    Pass 3: {len(accepted)} passed quality "
                  f"(rr_reject={reject_counts['low_residual_reduction']}, "
                  f"pruned={reject_counts['pruned']})  "
                  f"[{t_pass3:.1f}s, {pass3_qc_calls} qc calls @ {avg_qc_ms:.1f} ms/call]")
        
        # =================================================================
        # Pass 4 + Pass 5 (Option-iii hierarchical snapshot propagation):
        # 
        # Conceptually a single inner loop of "sub-iterations".  Each
        # sub-iter takes a snapshot of `accepted`, dispatches all
        # currently-eligible blocks in parallel against the snapshot,
        # then merges new accepts back in deterministic order.  After
        # the merge, we compute the NEXT sub-iter's eligible set: just
        # those unaccepted blocks whose nearest accepted neighbour
        # changed identity (typically a small fraction of all unaccepted
        # blocks; only those within neighbour_range of a new accept).
        # Loop terminates when no block is newly eligible (true fixed
        # point) or max_propagation_rounds is hit (safety cap).
        # 
        # For backwards-compatible verbose output we print "Pass 4" for
        # the first sub-iter and "Pass 5 sub-iter K" for the rest.
        # 
        # Eligibility tracking: last_seed_nbi[bi] = nbi-index of bi's
        # nearest accepted neighbour at last evaluation, or None if no
        # neighbour within range.  A block is "newly eligible" iff this
        # value just changed to a different non-None value (or from None
        # to a non-None value).  The tracker avoids re-dispatching
        # blocks whose seed input hasn't changed across sub-iters,
        # which was the source of redundant work in Variant A.
        # =================================================================
        
        # Initial sub-iter: every unaccepted block with any seed neighbour
        # in `accepted` (i.e. starting from Pass 3's accepted set).  We
        # also seed last_seed_nbi for all unaccepted blocks here so the
        # diff-based eligibility logic has a starting point.
        last_seed_nbi = {}
        initial_eligible = set()
        for bi in range(n_blocks):
            if bi in accepted:
                continue
            if block_data[bi] is None:
                continue
            nbi = _find_seed_nbi(bi, accepted, neighbour_range)
            last_seed_nbi[bi] = nbi
            if nbi is not None:
                initial_eligible.add(bi)
        
        # ---- Pass 4: first sub-iter ----
        t_pass4_start = time.time()
        new_accepts, _stats = _run_seeded_subiter(
            eligible_blocks=initial_eligible,
            blocks=blocks, block_data=block_data, accepted=accepted,
            global_probs=global_probs, num_samples=num_samples,
            min_assigned=min_assigned,
            overlap_threshold=overlap_threshold,
            min_residual_reduction=min_residual_reduction,
            neighbour_range=neighbour_range,
            pool=pool,
        )
        pass4_count          = len(new_accepts)
        pass4_visited        = _stats['visited']
        pass4_seeded         = _stats['seeded']
        pass4_discover_calls = _stats['discover_calls']
        pass4_overlap_pass   = _stats['overlap_pass']
        pass4_qc_calls       = _stats['qc_calls']
        t_pass4_discover     = _stats['t_discover']
        t_pass4_qc           = _stats['t_qc']
        
        if verbose:
            t_pass4 = time.time() - t_pass4_start
            avg_disc_ms = (t_pass4_discover / pass4_discover_calls * 1000) if pass4_discover_calls else 0.0
            avg_qc_ms   = (t_pass4_qc       / pass4_qc_calls       * 1000) if pass4_qc_calls       else 0.0
            print(f"    Pass 4: {pass4_count} neighbour-seeded recoveries  [{t_pass4:.1f}s]")
            print(f"      visited={pass4_visited}, seeded={pass4_seeded}, "
                  f"discover_calls={pass4_discover_calls} ({t_pass4_discover:.1f}s, {avg_disc_ms:.1f} ms/call), "
                  f"overlap_pass={pass4_overlap_pass}, "
                  f"qc_calls={pass4_qc_calls} ({t_pass4_qc:.1f}s, {avg_qc_ms:.1f} ms/call)")
        
        # ---- Pass 5: subsequent sub-iters until convergence ----
        # Eligibility for sub-iter 2 is computed from sub-iter 1's accepts.
        eligible = _compute_newly_eligible(
            new_accepts=new_accepts, accepted=accepted,
            last_seed_nbi=last_seed_nbi, n_blocks=n_blocks,
            neighbour_range=neighbour_range,
        )
        
        t_pass5_start = time.time()
        total_propagated = 0
        pass5_visited, pass5_seeded = 0, 0
        pass5_discover_calls, pass5_overlap_pass, pass5_qc_calls = 0, 0, 0
        t_pass5_discover, t_pass5_qc = 0.0, 0.0
        
        for sub_i in range(max_propagation_rounds):
            if not eligible:
                # True fixed point — no block has a changed seed.
                break
            
            t_subiter_start = time.time()
            new_accepts, _stats = _run_seeded_subiter(
                eligible_blocks=eligible,
                blocks=blocks, block_data=block_data, accepted=accepted,
                global_probs=global_probs, num_samples=num_samples,
                min_assigned=min_assigned,
                overlap_threshold=overlap_threshold,
                min_residual_reduction=min_residual_reduction,
                neighbour_range=neighbour_range,
                pool=pool,
            )
            
            pass5_visited       += _stats['visited']
            pass5_seeded        += _stats['seeded']
            pass5_discover_calls+= _stats['discover_calls']
            pass5_overlap_pass  += _stats['overlap_pass']
            pass5_qc_calls      += _stats['qc_calls']
            t_pass5_discover    += _stats['t_discover']
            t_pass5_qc          += _stats['t_qc']
            
            total_propagated += len(new_accepts)
            if verbose and len(new_accepts) > 0:
                t_subiter = time.time() - t_subiter_start
                print(f"    Pass 5 sub-iter {sub_i+1}: {len(new_accepts)} new  "
                      f"[{t_subiter:.1f}s, eligible={len(eligible)}, "
                      f"visited={_stats['visited']}, seeded={_stats['seeded']}, "
                      f"discover={_stats['discover_calls']}, "
                      f"overlap_pass={_stats['overlap_pass']}, "
                      f"qc={_stats['qc_calls']}]")
            
            if not new_accepts:
                # No new accepts this sub-iter; the eligible set was
                # exhausted (each block's _seeded_discover/_quality_check
                # path failed against its new seed).  Stop — no further
                # eligibility changes will be triggered.
                break
            
            # Compute next sub-iter's eligible set from this sub-iter's accepts.
            eligible = _compute_newly_eligible(
                new_accepts=new_accepts, accepted=accepted,
                last_seed_nbi=last_seed_nbi, n_blocks=n_blocks,
                neighbour_range=neighbour_range,
            )
        else:
            # Loop ran to max_propagation_rounds without breaking — flag.
            # This should be very rare with the default cap; if it happens
            # routinely, the algorithm is exploring a long propagation
            # chain and the cap should be raised.
            if verbose and eligible:
                print(f"    [warn] Pass 5 hit max_propagation_rounds={max_propagation_rounds} "
                      f"with {len(eligible)} blocks still eligible; "
                      f"final accepted set may be incomplete")
        
        if verbose:
            t_pass5 = time.time() - t_pass5_start
            avg_disc_ms = (t_pass5_discover / pass5_discover_calls * 1000) if pass5_discover_calls else 0.0
            avg_qc_ms   = (t_pass5_qc       / pass5_qc_calls       * 1000) if pass5_qc_calls       else 0.0
            print(f"    Pass 5: {total_propagated} propagated total  [{t_pass5:.1f}s]")
            print(f"      cumulative: visited={pass5_visited}, seeded={pass5_seeded}, "
                  f"discover_calls={pass5_discover_calls} ({t_pass5_discover:.1f}s, {avg_disc_ms:.1f} ms/call), "
                  f"overlap_pass={pass5_overlap_pass}, "
                  f"qc_calls={pass5_qc_calls} ({t_pass5_qc:.1f}s, {avg_qc_ms:.1f} ms/call)")
    
    finally:
        # Tear down pool + shared memory regardless of how we exit.
        if pool is not None:
            pool.close()
            pool.join()
        if shm is not None:
            shm.close()
            shm.unlink()
    
    # =====================================================================
    # Build output: add discovered haps, then dedup + chimera prune
    # (runs in main process, after pool has been torn down)
    # =====================================================================
    if not accepted:
        if verbose:
            print(f"    No discoveries accepted — returning blocks unchanged ({time.time()-t0:.1f}s)")
        return blocks
    
    output_blocks = list(blocks)
    for bi, info in accepted.items():
        output_blocks[bi] = _add_hap_to_block(blocks[bi], info['new_hap'])
    
    if verbose:
        print(f"    Added haplotypes to {len(accepted)} blocks")
    
    # Dedup: merge near-identical haplotypes (catches duplicate FPs at 0% error)
    t_dedup_start = time.time()
    output_br = block_haplotype_refinement.dedup_blocks(
        block_haplotypes.BlockResults(output_blocks),
        threshold_pct=1.0,
        verbose=verbose
    )
    t_dedup = time.time() - t_dedup_start
    
    # Chimera prune: remove haps explainable as recombinations of others.
    # Only process blocks that received new haps (the rest are unchanged).
    t_prune_start = time.time()
    n_pruned = 0
    output_list = list(output_br)
    for bi in accepted:
        block = output_list[bi]
        if block.probs_array is None or len(block.haplotypes) < 3:
            continue
        pruned_haps = block_haplotypes.prune_chimeras(
            block.haplotypes, block.probs_array,
            max_recombs=1,
            max_mismatch_percent=0.25,
            min_mean_delta_to_protect=0.25
        )
        if len(pruned_haps) < len(block.haplotypes):
            n_pruned += len(block.haplotypes) - len(pruned_haps)
            block.haplotypes = {i: v for i, v in enumerate(pruned_haps.values())}
    t_prune = time.time() - t_prune_start
    
    if verbose:
        print(f"    Dedup: [{t_dedup:.1f}s]")
        print(f"    Chimera prune: removed {n_pruned} haps from modified blocks  [{t_prune:.1f}s]")
        print(f"    Residual discovery complete ({time.time()-t0:.1f}s)")
    
    return block_haplotypes.BlockResults(output_list)