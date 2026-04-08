"""
paint_samples.py (IBS-AWARE VITERBI PAINTING)

A robust module for reconstructing diploid haplotypes (painting) from probabilistic 
genotype data using a Viterbi algorithm with IBS-aware pedigree support.

BINNED VERSION: Aggregates SNP-level emissions into bins (~100 SNPs/bin) to reduce
memory usage by ~100x while preserving accuracy.

Key Features:
1.  **Binned Emissions:** Sums per-SNP log-likelihoods into bins for memory efficiency.
2.  **Single Viterbi Path:** Fast, deterministic best-path reconstruction per sample.
3.  **IBS-Aware Pedigree Support:** Instead of enumerating tolerance paths for IBS
    ambiguity, the pedigree inference stage handles IBS equivalence directly via
    allele-level comparison. This eliminates the path explosion / straggler
    problem entirely.
4.  **Double-Recomb Discount:** Prefers simultaneous switches to prevent 2^N path explosion.
5.  **Visualization:** Plots individual paintings AND whole-population.
6.  **Parallel Execution:** Uses multiprocessing with forkserver and dynamic thread scaling.
7.  **Deterministic Emissions:** Converts probabilistic haplotypes to deterministic via argmax
    to prevent epistemic uncertainty from biasing founder selection.

IMPORTANT: Uses float64 for alpha arrays to prevent precision loss over long chromosomes.

IMPORTANT FIX (v2): Emission calculations now use DETERMINISTIC haplotypes (via argmax).
This fixes the "founder aliasing" bug where epistemic uncertainty in founder haplotypes
(e.g., 50/50 probability at a site) caused systematic bias toward uncertain founders.
The probabilistic approach gave uncertain founders "moderate" scores everywhere, making
them appear better than founders who were certain but had occasional mismatches. By
converting to deterministic alleles, we treat "I don't know" as a coin flip (unbiased
noise) rather than as evidence (systematic bias).
"""

import thread_config
from thread_config import numba_thread_scope

import os
import numpy as np
import math
import pandas as pd
import warnings
from typing import List, Tuple, Dict, NamedTuple, Set, DefaultDict, Counter, Optional, Union
from collections import defaultdict
import copy
import multiprocessing as _mp
try:
    _forkserver_ctx = _mp.get_context('forkserver')
except ValueError:
    _forkserver_ctx = _mp.get_context('fork')
from tqdm import tqdm
from functools import partial

import analysis_utils

# --- VISUALIZATION IMPORTS ---
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    import networkx as nx
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

# Suppress warnings
warnings.filterwarnings("ignore")
np.seterr(divide='ignore', invalid='ignore')

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("WARNING: Numba not found. Painting will be slow.")
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator
    prange = range

# =============================================================================
# 1. DATA STRUCTURES
# =============================================================================

class PaintedChunk(NamedTuple):
    start: int
    end: int
    hap1: int
    hap2: int

class SamplePainting:
    def __init__(self, sample_index: int, chunks: List[PaintedChunk]):
        self.sample_index = sample_index
        self.chunks = chunks 
        self.num_recombinations = max(0, len(self.chunks) - 1)

    def __repr__(self):
        return f"<SamplePainting ID {self.sample_index}: {len(self.chunks)} chunks>"

    def __iter__(self):
        return iter(self.chunks)
    
    def __getitem__(self, idx):
        return self.chunks[idx]

# =============================================================================
# 2. PAINTING CONTAINERS
# =============================================================================

class BlockPainting:
    def __init__(self, block_position_range: Tuple[int, int], samples: List[SamplePainting]):
        self.start_pos = block_position_range[0]
        self.end_pos = block_position_range[1]
        self.samples = samples
        self.num_samples = len(samples)

    def __len__(self): return self.num_samples
    def __getitem__(self, idx): return self.samples[idx]
    def __iter__(self): return iter(self.samples)

# =============================================================================
# 4. HELPER: DENSE MATRIX CONVERSION
# =============================================================================

def founder_block_to_dense(block_result):
    """Convert probabilistic haplotypes to dense integer matrix via argmax."""
    positions = np.array(block_result.positions, dtype=np.int64)
    hap_dict = block_result.haplotypes
    if not hap_dict: return np.zeros((0, 0), dtype=np.int8), positions
    max_id = max(hap_dict.keys())
    n_sites = len(positions)
    dense_haps = np.full((max_id + 1, n_sites), -1, dtype=np.int8)
    for fid, hap_arr in hap_dict.items():
        if hap_arr.ndim == 2: 
            concrete = np.argmax(hap_arr, axis=1)
        else: 
            concrete = hap_arr
        dense_haps[fid, :] = concrete.astype(np.int8)
    return dense_haps, positions

# =============================================================================
# 5. BINNED EMISSION CALCULATOR (NEW - MEMORY EFFICIENT)
# =============================================================================

def calculate_binned_emissions(sample_probs_matrix, hap_dict, positions, 
                               snps_per_bin=100, robustness_epsilon=1e-2):
    """
    Calculate emission log-likelihoods aggregated into bins.
    
    MEMORY EFFICIENT: Instead of (n_samples, K, n_sites), produces (n_samples, K, n_bins)
    by summing log-likelihoods within each bin. For 50,000 SNPs with 100 SNPs/bin,
    this reduces memory by ~100x.
    
    IMPORTANT: Haplotypes are converted to DETERMINISTIC (via argmax) before emission
    calculation. This fixes a bug where epistemic uncertainty in founder haplotypes
    caused a systematic bias toward uncertain founders. When a founder has 50/50
    probability at a site, the old probabilistic approach would give it "moderate"
    emissions everywhere, making it appear better than founders who are certain but
    happen to mismatch at a few sites. The argmax approach introduces small unbiased
    noise at uncertain sites (<1%), which is preferable to systematic bias.
    
    Args:
        sample_probs_matrix: (n_samples, n_sites, 3) genotype probabilities
        hap_dict: Dict mapping founder ID -> (n_sites,) or (n_sites, 2) haplotypes
                  If (n_sites, 2), columns are P(allele=0), P(allele=1) - converted via argmax
                  If (n_sites,), values are deterministic alleles (0 or 1)
        positions: (n_sites,) array of SNP positions
        snps_per_bin: Number of SNPs to aggregate per bin
        robustness_epsilon: Numerical stability term
    
    Returns:
        binned_ll: (n_samples, K, n_bins) aggregated log-likelihoods
        state_defs: (K, 2) state definitions
        hap_keys: List of founder IDs
        bin_centers: (n_bins,) physical positions of bin centers
        bin_edges: (n_bins + 1,) bin boundary positions
    """
    hap_keys = sorted(list(hap_dict.keys()))
    num_haps = len(hap_keys)
    num_samples, num_sites, _ = sample_probs_matrix.shape
    
    # Create state definitions
    idx_i, idx_j = np.unravel_index(np.arange(num_haps**2), (num_haps, num_haps))
    state_defs = np.stack([idx_i, idx_j], axis=1).astype(np.int32)
    K = num_haps ** 2
    
    if not hap_keys or num_sites == 0:
        return np.zeros((num_samples, K, 0), dtype=np.float64), state_defs, hap_keys, np.array([]), np.array([])
    
    # Compute bin structure
    n_bins = max(1, num_sites // snps_per_bin)
    if n_bins == 0:
        n_bins = 1
    
    # Split SNP indices into approximately equal bins
    bin_snp_indices = np.array_split(np.arange(num_sites), n_bins)
    
    # Compute bin centers (average physical position of SNPs in each bin)
    bin_centers = np.array([positions[idx].mean() for idx in bin_snp_indices], dtype=np.float64)
    
    # Compute bin edges for chunk reconstruction
    bin_edges = np.zeros(n_bins + 1, dtype=np.int64)
    bin_edges[0] = positions[0]
    for i in range(n_bins - 1):
        # Edge is midpoint between last SNP of this bin and first SNP of next bin
        last_pos = positions[bin_snp_indices[i][-1]]
        next_first = positions[bin_snp_indices[i+1][0]]
        bin_edges[i+1] = (last_pos + next_first) // 2
    bin_edges[-1] = positions[-1] + 1  # Final edge past last SNP
    
    # =========================================================================
    # CRITICAL FIX: Convert haplotypes to DETERMINISTIC using argmax
    # =========================================================================
    # This fixes the "founder aliasing" bug where epistemic uncertainty in founder
    # haplotypes caused biased emissions. When founder A is 50/50 uncertain at many
    # sites but founder B is certain, the old probabilistic approach would give A
    # "moderate" scores everywhere while B would get perfect scores at matches but
    # harsh penalties at mismatches. This made uncertain founders appear artificially
    # better than they should be.
    #
    # By converting to deterministic alleles via argmax, we treat "I don't know" as
    # a coin flip rather than as evidence. This introduces small unbiased noise at
    # uncertain sites (<1% of sites typically), which is far preferable to the
    # systematic bias of the probabilistic approach.
    # =========================================================================
    
    deterministic_alleles = np.zeros((num_haps, num_sites), dtype=np.int8)
    
    for i, k in enumerate(hap_keys):
        hap = hap_dict[k]
        if hap.ndim == 2 and hap.shape[1] == 2:
            # Probabilistic: (n_sites, 2) with P(allele=0), P(allele=1)
            # Use argmax to get deterministic allele (0 or 1)
            deterministic_alleles[i] = np.argmax(hap, axis=1).astype(np.int8)
        else:
            # Already deterministic: (n_sites,) with values 0 or 1
            deterministic_alleles[i] = hap.astype(np.int8)
    
    # Compute deterministic genotypes for all state pairs
    # genotype = allele_i + allele_j (values: 0, 1, or 2)
    # Shape: (num_haps, num_haps, num_sites)
    state_genotypes = (deterministic_alleles[:, None, :] + 
                       deterministic_alleles[None, :, :])
    
    # Reshape to (K, num_sites) to match state indexing
    state_genotypes_flat = state_genotypes.reshape(K, num_sites)
    
    # Allocate binned emissions
    binned_ll = np.zeros((num_samples, K, n_bins), dtype=np.float64)
    
    # Process each bin - aggregate SNP emissions
    uniform_prob = 1.0 / 3.0
    
    for bin_idx, snp_indices in enumerate(bin_snp_indices):
        if len(snp_indices) == 0:
            continue
        
        n_snps_bin = len(snp_indices)
        
        # Extract sample probabilities for SNPs in this bin
        # Shape: (n_samples, n_snps_in_bin, 3)
        sample_probs_bin = sample_probs_matrix[:, snp_indices, :]
        
        # Extract state genotypes for this bin
        # Shape: (K, n_snps_in_bin)
        geno_bin = state_genotypes_flat[:, snp_indices]
        
        # For each state, look up P(observed | genotype) using the deterministic genotype
        # We need to gather: sample_probs_bin[sample, snp, geno[state, snp]]
        # for all (sample, state, snp) combinations
        
        # Efficient approach: compute for each genotype value separately and combine
        model_probs = np.zeros((num_samples, K, n_snps_bin), dtype=np.float64)
        
        for geno_val in range(3):
            # Mask where this genotype applies: (K, n_snps_bin)
            mask = (geno_bin == geno_val)
            # Get sample probability for this genotype: (n_samples, n_snps_bin)
            s_prob = sample_probs_bin[:, :, geno_val]
            # Expand dimensions for broadcasting: (n_samples, 1, n_snps_bin)
            s_prob_expanded = s_prob[:, np.newaxis, :]
            # mask broadcasts from (1, K, n_snps_bin) to (n_samples, K, n_snps_bin)
            # Apply where mask is True
            model_probs += mask[np.newaxis, :, :] * s_prob_expanded
        
        # Apply robustness epsilon
        final_probs = model_probs * (1.0 - robustness_epsilon) + robustness_epsilon * uniform_prob
        
        # Compute log-likelihoods
        final_probs = np.maximum(final_probs, 1e-300)
        ll_snps = np.log(final_probs)
        ll_snps = np.maximum(ll_snps, -50.0)
        
        # Sum log-likelihoods within bin
        binned_ll[:, :, bin_idx] = ll_snps.sum(axis=2)
    
    return binned_ll, state_defs, hap_keys, bin_centers, bin_edges

# =============================================================================
# 6. TOLERANCE VITERBI KERNELS (BINNED VERSION)
# =============================================================================

@njit(parallel=True, fastmath=True)
def run_forward_pass_max_sum_binned(ll_tensor, bin_centers, recomb_rate, state_definitions, 
                                     n_haps, switch_penalty, double_recomb_factor=1.5):
    """
    Forward pass of max-sum Viterbi algorithm on BINNED data.
    
    Uses physical distance between bin centers for transition probabilities.
    """
    n_samples, K, n_bins = ll_tensor.shape
    
    alpha = np.full((n_samples, n_bins, K), -np.inf, dtype=np.float64)
    
    if n_haps > 1: 
        log_N_minus_1 = math.log(float(n_haps - 1))
    else: 
        log_N_minus_1 = 0.0

    for s in prange(n_samples):
        for k in range(K): 
            alpha[s, 0, k] = ll_tensor[s, k, 0]

        for i in range(1, n_bins):
            # Distance between bin centers
            dist_bp = bin_centers[i] - bin_centers[i-1]
            if dist_bp < 1: dist_bp = 1
            theta = float(dist_bp) * recomb_rate
            if theta > 0.5: theta = 0.5
            if theta < 1e-15: theta = 1e-15
            
            log_switch = math.log(theta) - switch_penalty
            log_stay = math.log(1.0 - theta)
            
            cost_0 = 2.0 * log_stay                                    # Neither switches
            cost_1 = log_switch + log_stay - log_N_minus_1             # One switches
            cost_2 = double_recomb_factor * log_switch - 2.0 * log_N_minus_1  # Both switch
            
            for k_curr in range(K):
                h1_curr = state_definitions[k_curr, 0]
                h2_curr = state_definitions[k_curr, 1]
                best_score = -np.inf
                
                for k_prev in range(K):
                    h1_prev = state_definitions[k_prev, 0]
                    h2_prev = state_definitions[k_prev, 1]
                    
                    dist = 0
                    if h1_curr != h1_prev: dist += 1
                    if h2_curr != h2_prev: dist += 1
                    
                    if dist == 0: trans = cost_0
                    elif dist == 1: trans = cost_1
                    else: trans = cost_2
                    
                    score = alpha[s, i-1, k_prev] + trans
                    if score > best_score: best_score = score
                    
                alpha[s, i, k_curr] = best_score + ll_tensor[s, k_curr, i]
                
    return alpha


# =============================================================================
# 7. VITERBI TRACEBACK (BINNED VERSION)
# =============================================================================

def reconstruct_single_best_path_binned(alpha, ll_tensor, bin_centers, bin_edges,
                                         recomb_rate, state_definitions, n_haps, 
                                         switch_penalty, hap_keys, double_recomb_factor=1.5):
    """Reconstruct the single best path via standard Viterbi traceback - BINNED."""
    n_bins, K = alpha.shape
    if n_haps > 1: 
        log_N_minus_1 = math.log(float(n_haps - 1))
    else: 
        log_N_minus_1 = 0.0
    
    curr_k = np.argmax(alpha[n_bins-1])
    h1_idx, h2_idx = state_definitions[curr_k]
    t1, t2 = hap_keys[h1_idx], hap_keys[h2_idx]
    
    # Use bin edges for chunk positions
    chunks = [PaintedChunk(start=int(bin_edges[n_bins-1]), end=int(bin_edges[n_bins]), hap1=t1, hap2=t2)]
    
    for t in range(n_bins - 1, 0, -1):
        prev_t = t - 1
        curr_h1, curr_h2 = state_definitions[curr_k]
        
        dist_bp = bin_centers[t] - bin_centers[prev_t]
        if dist_bp < 1: dist_bp = 1
        theta = float(dist_bp) * recomb_rate
        if theta > 0.5: theta = 0.5
        if theta < 1e-15: theta = 1e-15
        
        log_switch = math.log(theta) - switch_penalty
        log_stay = math.log(1.0 - theta)
        
        cost_0 = 2.0 * log_stay
        cost_1 = log_switch + log_stay - log_N_minus_1
        cost_2 = double_recomb_factor * log_switch - 2.0 * log_N_minus_1
        
        best_prev = -1
        best_score = -np.inf
        
        for prev_k in range(K):
            if alpha[prev_t, prev_k] == -np.inf: continue
            prev_h1, prev_h2 = state_definitions[prev_k]
            dist = 0
            if curr_h1 != prev_h1: dist += 1
            if curr_h2 != prev_h2: dist += 1
            
            trans = cost_0 if dist == 0 else (cost_1 if dist == 1 else cost_2)
            score = alpha[prev_t, prev_k] + trans
            if score > best_score:
                best_score = score
                best_prev = prev_k
                
        curr_k = best_prev
        prev_h1, prev_h2 = state_definitions[curr_k]
        pt1, pt2 = hap_keys[prev_h1], hap_keys[prev_h2]
        old_chunk = chunks[0]
        
        is_extension = False
        if (pt1 == old_chunk.hap1 and pt2 == old_chunk.hap2): is_extension = True
        elif (pt1 == old_chunk.hap2 and pt2 == old_chunk.hap1): is_extension = True
        
        if is_extension:
            chunks[0] = PaintedChunk(start=int(bin_edges[prev_t]), end=old_chunk.end, 
                                     hap1=old_chunk.hap1, hap2=old_chunk.hap2)
        else:
            chunks.insert(0, PaintedChunk(start=int(bin_edges[prev_t]), end=int(bin_edges[t]), 
                                          hap1=pt1, hap2=pt2))
            
    return [SamplePainting(0, chunks)]

# =============================================================================
# 9. MULTIPROCESSING DRIVER (SharedMemory + Persistent Pool)
# =============================================================================

from multiprocessing import shared_memory as _shm

# Worker-local cache for SharedMemory arrays
_PAINT_SHARED = {}
_PAINT_SHM_REFS = []
_PAINT_CHROM_ID = None

# Dynamic thread scaling globals (set by _init_persistent_paint_worker)
_PAINT_ACTIVE_COUNTER = None
_PAINT_TOTAL_CORES = None


def _paint_get_dynamic_threads():
    """Recheck active worker count and return optimal numba thread count."""
    if _PAINT_ACTIVE_COUNTER is None or _PAINT_TOTAL_CORES is None:
        return 1
    active = max(_PAINT_ACTIVE_COUNTER.value, 1)
    return max(1, _PAINT_TOTAL_CORES // active)

def _create_shm_from_array(arr):
    """Create a SharedMemory block from a numpy array. Returns (shm, name, shape, dtype_str)."""
    shm = _shm.SharedMemory(create=True, size=arr.nbytes)
    shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    shm_arr[:] = arr
    return shm, shm.name, arr.shape, str(arr.dtype)

def _array_from_shm(name, shape, dtype_str):
    """Reconstruct a numpy array from SharedMemory. Returns (shm_ref, array)."""
    shm = _shm.SharedMemory(name=name, create=False)
    arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
    return shm, arr


def _worker_paint_batch_binned(args):
    """Worker function for parallel painting — single Viterbi path per sample."""
    indices, start_idx, end_idx = args
    
    # Read from SharedMemory (zero-copy)
    sample_probs_slice = _PAINT_SHARED['block_samples_data'][start_idx:end_idx]
    positions = _PAINT_SHARED['positions']
    hap_dict = _PAINT_SHARED['hap_dict']
    params = _PAINT_SHARED['params']
    
    recomb_rate = params['recomb_rate']
    switch_penalty = params['switch_penalty']
    robustness_epsilon = params['robustness_epsilon']
    double_recomb_factor = params.get('double_recomb_factor', 1.5)
    snps_per_bin = params.get('snps_per_bin', 100)
    numba_threads = params.get('numba_threads', 1)
    
    # Calculate BINNED emissions
    ll_tensor, state_defs, hap_keys, bin_centers, bin_edges = calculate_binned_emissions(
        sample_probs_slice, hap_dict, positions,
        snps_per_bin=snps_per_bin,
        robustness_epsilon=robustness_epsilon
    )
    num_haps = len(hap_keys)
    n_bins = len(bin_centers)
    
    if n_bins == 0:
        # No bins - return empty results
        results = []
        for global_idx in indices:
            results.append(SamplePainting(global_idx, []))
        return results
    
    # Control Numba thread count for prange loops in the forward kernel.
    # Dynamic scaling: use more threads when fewer workers are active (tail).
    dyn_threads = _paint_get_dynamic_threads()
    effective_threads = max(numba_threads, dyn_threads)
    with numba_thread_scope(effective_threads):
        # Run forward pass on BINNED data (no backward pass needed for single Viterbi)
        alpha = run_forward_pass_max_sum_binned(ll_tensor, bin_centers, recomb_rate, state_defs, 
                                                 num_haps, float(switch_penalty), double_recomb_factor)
    
    results = []
    for i, global_idx in enumerate(indices):
        # Single Viterbi best path — fast, deterministic, no beam pruning
        viterbi_path = reconstruct_single_best_path_binned(
            alpha[i], ll_tensor[i], bin_centers, bin_edges,
            recomb_rate, state_defs, num_haps, switch_penalty, hap_keys,
            double_recomb_factor=double_recomb_factor
        )
        
        painting = viterbi_path[0]  # reconstruct returns [SamplePainting(...)]
        painting.sample_index = global_idx
        results.append(painting)
        
    return results


def _init_persistent_paint_worker(total_cores=None, active_counter=None):
    """Initializer for persistent pool — sets up globals and dynamic threading."""
    global _PAINT_SHARED, _PAINT_SHM_REFS, _PAINT_CHROM_ID
    global _PAINT_ACTIVE_COUNTER, _PAINT_TOTAL_CORES
    _PAINT_SHARED = {}
    _PAINT_SHM_REFS = []
    _PAINT_CHROM_ID = None
    _PAINT_ACTIVE_COUNTER = active_counter
    _PAINT_TOTAL_CORES = total_cores
    # Cap numba threads to 1 initially — workers scale up dynamically
    try:
        import numba
        numba.set_num_threads(1)
    except Exception:
        pass

def _load_shm_for_chromosome(chrom_id, meta):
    """
    Lazy-load SharedMemory for a new chromosome. Only re-loads when chrom_id changes.
    Called by workers on first task for each chromosome.
    """
    global _PAINT_SHARED, _PAINT_SHM_REFS, _PAINT_CHROM_ID
    
    if _PAINT_CHROM_ID == chrom_id:
        return  # Already loaded
    
    # Close old SharedMemory refs (from previous chromosome)
    for shm_ref in _PAINT_SHM_REFS:
        try:
            shm_ref.close()
        except Exception:
            pass
    _PAINT_SHM_REFS = []
    _PAINT_SHARED = {}
    
    # Open new SharedMemory blocks
    shm, arr = _array_from_shm(meta['samples_name'], meta['samples_shape'], meta['samples_dtype'])
    _PAINT_SHM_REFS.append(shm)
    _PAINT_SHARED['block_samples_data'] = arr
    
    shm, arr = _array_from_shm(meta['positions_name'], meta['positions_shape'], meta['positions_dtype'])
    _PAINT_SHM_REFS.append(shm)
    _PAINT_SHARED['positions'] = arr
    
    shm, arr = _array_from_shm(meta['haps_name'], meta['haps_shape'], meta['haps_dtype'])
    _PAINT_SHM_REFS.append(shm)
    hap_keys = meta['hap_keys']
    _PAINT_SHARED['hap_dict'] = {k: arr[i] for i, k in enumerate(hap_keys)}
    
    _PAINT_SHARED['params'] = meta['params']
    _PAINT_CHROM_ID = chrom_id


def _worker_paint_persistent(args):
    """
    Worker for persistent pool. Accepts (chrom_id, meta, indices, start_idx, end_idx).
    Lazy-loads SharedMemory when chromosome changes — meta is tiny (~500 bytes),
    so including it in each task adds negligible pickle overhead.
    
    Tracks active workers for dynamic thread scaling: straggler samples
    get more numba threads as peers finish.
    """
    chrom_id, meta, indices, start_idx, end_idx = args
    _load_shm_for_chromosome(chrom_id, meta)
    
    if _PAINT_ACTIVE_COUNTER is not None:
        with _PAINT_ACTIVE_COUNTER.get_lock():
            _PAINT_ACTIVE_COUNTER.value += 1
    
    try:
        return _worker_paint_batch_binned((indices, start_idx, end_idx))
    finally:
        if _PAINT_ACTIVE_COUNTER is not None:
            with _PAINT_ACTIVE_COUNTER.get_lock():
                _PAINT_ACTIVE_COUNTER.value -= 1


class PaintingPoolManager:
    """
    Persistent pool manager for painting multiple chromosomes efficiently.
    
    Creates the multiprocessing Pool ONCE and reuses it across chromosomes.
    SharedMemory is created per chromosome; workers lazy-initialize when they
    detect a new chromosome ID.
    
    Usage:
        with paint_samples.PaintingPoolManager(num_processes=112) as painter:
            for r_name in region_keys:
                result = painter.paint_chromosome(
                    block_result, sample_probs_matrix, sample_sites, ...
                )
    
    Saves ~10s per chromosome by avoiding repeated Pool creation/teardown.
    """
    
    def __init__(self, num_processes=16):
        self.num_processes = num_processes
        self._active_counter = _forkserver_ctx.Value('i', 0)
        self.pool = _forkserver_ctx.Pool(
            num_processes,
            initializer=_init_persistent_paint_worker,
            initargs=(num_processes, self._active_counter)
        )
        self._chrom_counter = 0
    
    def paint_chromosome(self, block_result, sample_probs_matrix, sample_sites,
                         recomb_rate=1e-8, switch_penalty=10.0,
                         robustness_epsilon=1e-2, absolute_margin=5.0,
                         margin_per_snp=0.0, batch_size=1, 
                         max_active_paths=2000, double_recomb_factor=1.5,
                         snps_per_bin=100):
        """
        Paint one chromosome using the persistent pool.
        
        Uses single Viterbi best path per sample (fast, deterministic).
        IBS ambiguity is handled downstream by ibs_painting.py at the
        pedigree inference stage, not by enumerating tolerance paths.
        
        Returns:
            BlockPainting with all samples' single Viterbi paths
        """
        self._chrom_counter += 1
        chrom_id = self._chrom_counter
        
        positions = block_result.positions
        n_sites_block = len(positions)
        
        n_bins = max(1, n_sites_block // snps_per_bin)
        
        block_samples_data = analysis_utils.get_sample_data_at_sites_multiple(
            sample_probs_matrix, sample_sites, positions
        )
        num_samples = block_samples_data.shape[0]
        
        num_tasks = math.ceil(num_samples / batch_size)
        actual_pool_size = min(num_tasks, self.num_processes)
        numba_threads = max(1, self.num_processes // max(actual_pool_size, 1))
        
        print(f"Viterbi Painting (BINNED) {num_samples} samples ({n_sites_block} SNPs → {n_bins} bins) "
              f"using {self.num_processes} workers...")
        
        params = {
            'recomb_rate': recomb_rate,
            'switch_penalty': switch_penalty,
            'robustness_epsilon': robustness_epsilon,
            'double_recomb_factor': double_recomb_factor,
            'snps_per_bin': snps_per_bin,
            'numba_threads': numba_threads,
        }
        
        # Create SharedMemory for this chromosome
        shm_blocks = []
        
        try:
            samples_c = np.ascontiguousarray(block_samples_data)
            shm_s, s_name, s_shape, s_dtype = _create_shm_from_array(samples_c)
            shm_blocks.append(shm_s)
            
            positions_arr = np.ascontiguousarray(np.array(positions, dtype=np.int64))
            shm_p, p_name, p_shape, p_dtype = _create_shm_from_array(positions_arr)
            shm_blocks.append(shm_p)
            
            hap_keys_list = sorted(block_result.haplotypes.keys())
            hap_arrays = [block_result.haplotypes[k] for k in hap_keys_list]
            hap_stack = np.ascontiguousarray(np.stack(hap_arrays))
            shm_h, h_name, h_shape, h_dtype = _create_shm_from_array(hap_stack)
            shm_blocks.append(shm_h)
            
            meta = {
                'samples_name': s_name, 'samples_shape': s_shape, 'samples_dtype': s_dtype,
                'positions_name': p_name, 'positions_shape': p_shape, 'positions_dtype': p_dtype,
                'haps_name': h_name, 'haps_shape': h_shape, 'haps_dtype': h_dtype,
                'hap_keys': hap_keys_list,
                'params': params,
            }
            
            # Tasks carry chrom_id + meta (meta is ~500 bytes, negligible)
            tasks = []
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                indices = list(range(start_idx, end_idx))
                tasks.append((chrom_id, meta, indices, start_idx, end_idx))
            
            all_sample_paintings = []
            for batch_result in tqdm(
                self.pool.imap_unordered(_worker_paint_persistent, tasks),
                total=len(tasks)
            ):
                all_sample_paintings.extend(batch_result)
        
        finally:
            for shm in shm_blocks:
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass
        
        all_sample_paintings.sort(key=lambda x: x.sample_index)
        range_tuple = (int(positions[0]), int(positions[-1]))
        return BlockPainting(range_tuple, all_sample_paintings)
    
    def close(self):
        """Terminate and join the persistent pool."""
        try:
            self.pool.terminate()
            self.pool.join()
        except Exception:
            pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


# =============================================================================
# 10. VISUALIZATIONS (unchanged)
# =============================================================================

def plot_painting_topology(block_painting, sample_idx=0, output_file=None):
    """Plot the topology of a sample's painting as a graph."""
    if not HAS_PLOTTING: return
    sample_obj = block_painting[sample_idx]
    chunks = sample_obj.chunks if hasattr(sample_obj, 'chunks') else []
    if not chunks: return

    G = nx.DiGraph()
    unique_pairs = set()
    for c in chunks: 
        unique_pairs.add(tuple(sorted((c.hap1, c.hap2))))
            
    sorted_pairs = sorted(list(unique_pairs))
    pair_to_y = {p: i for i, p in enumerate(sorted_pairs)}
    
    pos = {}
    
    for i, chunk in enumerate(chunks):
        pair = tuple(sorted((chunk.hap1, chunk.hap2)))
        pos_x = (chunk.start + chunk.end) / 2
        node_id = (chunk.start, pair[0], pair[1])
        G.add_node(node_id, label=f"{pair[0]}/{pair[1]}")
        pos[node_id] = (pos_x, pair_to_y[pair])
        if i > 0:
            prev_c = chunks[i-1]
            prev_p = tuple(sorted((prev_c.hap1, prev_c.hap2)))
            prev_node = (prev_c.start, prev_p[0], prev_p[1])
            G.add_edge(prev_node, node_id)

    fig, ax = plt.subplots(figsize=(15, max(4, len(sorted_pairs))))
    for pair, y in pair_to_y.items():
        ax.axhline(y, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        ax.text(chunks[0].start, y, f" {pair[0]}/{pair[1]}", va='center', fontsize=9, fontweight='bold')

    nx.draw_networkx_nodes(G, pos, node_color='#aaccff', node_size=100, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.0, ax=ax)
    
    ax.set_title(f"Painting Topology — Sample {sample_idx} ({len(chunks)} chunks)", fontsize=14)
    ax.set_xlabel("Genomic Position (bp)")
    ax.set_yticks([])
    if output_file: plt.savefig(output_file, bbox_inches='tight')
    else: plt.show()
    plt.close()

def plot_viable_paintings(block_painting, sample_idx=0, max_paths=50, output_file=None):
    """Plot the painting for a sample."""
    if not HAS_PLOTTING: return
    sample_obj = block_painting[sample_idx]
    chunks = sample_obj.chunks if hasattr(sample_obj, 'chunks') else []
    if not chunks: return

    unique_haps = set()
    for c in chunks: 
        unique_haps.add(c.hap1)
        unique_haps.add(c.hap2)
    sorted_haps = sorted(list(unique_haps))
    hap_to_idx = {h: i for i, h in enumerate(sorted_haps)}
    
    if len(sorted_haps) <= 10: palette = sns.color_palette("tab10", len(sorted_haps))
    else: palette = sns.color_palette("husl", len(sorted_haps))

    row_height = 0.5
    y_height = 0.4 
    fig, ax = plt.subplots(figsize=(20, 2.0))
    
    # Draw painting (two tracks: hap1 bottom, hap2 top)
    y_base = 0
    for chunk in chunks:
        width = chunk.end - chunk.start
        if width <= 0: continue
        c1 = palette[hap_to_idx[chunk.hap1]]
        rect1 = mpatches.Rectangle((chunk.start, y_base), width, y_height/2, facecolor=c1, edgecolor='none')
        ax.add_patch(rect1)
        c2 = palette[hap_to_idx[chunk.hap2]]
        rect2 = mpatches.Rectangle((chunk.start, y_base + y_height/2), width, y_height/2, facecolor=c2, edgecolor='none')
        ax.add_patch(rect2)

    ax.set_xlim(block_painting.start_pos, block_painting.end_pos)
    ax.set_ylim(-0.1, row_height + 0.1)
    ax.set_yticks([row_height/2])
    ax.set_yticklabels([f"Sample {sample_idx}"], fontsize=8)
    ax.set_xlabel("Genomic Position (bp)")
    
    patches = [mpatches.Patch(color=palette[hap_to_idx[h]], label=f"H{h}") for h in sorted_haps]
    ax.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc='upper left')
    
    plt.tight_layout()
    if output_file: plt.savefig(output_file)
    else: plt.show()
    plt.close()

def plot_population_painting(block_painting, output_file=None, 
                             title="Population Painting", 
                             figsize_width=20, 
                             row_height_per_sample=0.25,
                             show_labels=True,
                             sample_names=None):
    """Plot paintings for all samples in a population view."""
    if not HAS_PLOTTING:
        print("Error: Matplotlib/Seaborn not installed.")
        return

    unique_haps = set()
    for sample in block_painting:
        for chunk in sample:
            if chunk.hap1 != -1: unique_haps.add(chunk.hap1)
            if chunk.hap2 != -1: unique_haps.add(chunk.hap2)
    
    sorted_haps = sorted(list(unique_haps))
    hap_to_idx = {h: i for i, h in enumerate(sorted_haps)}
    
    if len(sorted_haps) <= 10: palette = sns.color_palette("tab10", len(sorted_haps))
    elif len(sorted_haps) <= 20: palette = sns.color_palette("tab20", len(sorted_haps))
    else: palette = sns.color_palette("husl", len(sorted_haps))
        
    num_samples = len(block_painting)
    header_space = 2.0 
    calc_height = (num_samples * row_height_per_sample) + header_space
    if calc_height < 6: calc_height = 6
    if calc_height > 300: calc_height = 300
    
    figsize = (figsize_width, calc_height)
    fig, ax = plt.subplots(figsize=figsize)
    
    y_height = 0.8 
    
    for i, sample in enumerate(block_painting):
        y_base = i 
        for chunk in sample:
            width = chunk.end - chunk.start
            if width <= 0: continue
            
            if chunk.hap1 != -1:
                color1 = palette[hap_to_idx[chunk.hap1]]
                rect1 = mpatches.Rectangle((chunk.start, y_base), width, y_height/2, facecolor=color1, edgecolor='none')
                ax.add_patch(rect1)
            
            if chunk.hap2 != -1:
                color2 = palette[hap_to_idx[chunk.hap2]]
                rect2 = mpatches.Rectangle((chunk.start, y_base + y_height/2), width, y_height/2, facecolor=color2, edgecolor='none')
                ax.add_patch(rect2)
            
    ax.set_xlim(block_painting.start_pos, block_painting.end_pos)
    ax.set_ylim(-0.5, len(block_painting) + 0.5)
    ax.set_xlabel("Genomic Position (bp)")
    ax.set_ylabel("Samples")
    ax.set_title(title)
    
    if show_labels:
        if sample_names and len(sample_names) == len(block_painting): labels = sample_names
        else: labels = [f"S{s.sample_index}" for s in block_painting]
        ax.set_yticks(np.arange(len(block_painting)) + y_height/2)
        ax.set_yticklabels(labels, fontsize=8)
    else:
        ax.set_yticks([])

    legend_patches = []
    for h_key in sorted_haps:
        c = palette[hap_to_idx[h_key]]
        legend_patches.append(mpatches.Patch(color=c, label=f"Founder {h_key}"))
        
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.01, 1), loc='upper left')
    
    plt.tight_layout()
    if output_file:
        dpi = 100 if calc_height > 100 else 150
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


# =============================================================================
# 11. IBS-AWARE PEDIGREE SUPPORT
# =============================================================================
#
# These functions convert SamplePainting objects into allele grids with
# IBS-aware homozygosity masks for the pedigree HMM in pedigree_inference.py.
#
# The pedigree HMM compares allele values (0/1), not founder IDs. When two
# founders are IBS (identical-by-state) at a genomic region, they carry the
# same alleles — the HMM literally cannot distinguish them. The hom mask
# captures this: if alleles on track 1 == alleles on track 2 at a bin,
# the HMM allows free phase switches there (correct, since the data cannot
# resolve which parental chromosome contributed which track).
#
# This replaces the old multi-consensus tolerance painting approach which
# enumerated hundreds of paths differing only in IBS regions — all producing
# identical allele grids — then evaluated M² HMM runs per parent-child pair.
# =============================================================================

def convert_id_grid_to_allele_grid_multisnp(id_grid, bin_centers, founder_block,
                                             bin_width_bp=None, max_snps_per_bin=10):
    """
    Convert founder ID grid to allele grid with multiple SNPs per bin.
    
    For each bin, samples up to max_snps_per_bin SNPs and looks up the
    actual allele (0/1) for the assigned founder.
    
    Returns:
        allele_grid: (num_bins, 2, max_snps_per_bin) int8
    """
    num_bins = id_grid.shape[0]
    snp_positions = founder_block.positions
    n_snps = len(snp_positions)
    allele_grid = np.full((num_bins, 2, max_snps_per_bin), -1, dtype=np.int8)
    
    if n_snps == 0:
        return allele_grid
    
    if bin_width_bp is None:
        if len(bin_centers) > 1:
            bin_width_bp = bin_centers[1] - bin_centers[0]
        else:
            bin_width_bp = 10000
    
    half_width = bin_width_bp / 2.0
    
    # Build founder allele lookup
    hap_keys = sorted(list(founder_block.haplotypes.keys()))
    max_id = max(hap_keys) if hap_keys else 0
    founder_alleles = np.full((max_id + 1, n_snps), -1, dtype=np.int8)
    for fid, h_arr in founder_block.haplotypes.items():
        if h_arr.ndim == 2:
            founder_alleles[fid, :] = np.argmax(h_arr, axis=1).astype(np.int8)
        else:
            founder_alleles[fid, :] = h_arr.astype(np.int8)
    
    # Assign alleles to bins
    bin_starts = bin_centers - half_width
    bin_ends = bin_centers + half_width
    start_indices = np.searchsorted(snp_positions, bin_starts, side='left')
    end_indices = np.searchsorted(snp_positions, bin_ends, side='right')
    
    for b in range(num_bins):
        s_start = start_indices[b]
        s_end = end_indices[b]
        bin_n_snps = s_end - s_start
        if bin_n_snps == 0:
            continue
        
        if bin_n_snps <= max_snps_per_bin:
            sampled_indices = list(range(s_start, s_end))
        else:
            step = bin_n_snps / max_snps_per_bin
            sampled_indices = [s_start + int(i * step) for i in range(max_snps_per_bin)]
        
        f0 = id_grid[b, 0]
        f1 = id_grid[b, 1]
        for k_idx, snp_idx in enumerate(sampled_indices):
            if k_idx >= max_snps_per_bin:
                break
            if f0 >= 0:
                allele_grid[b, 0, k_idx] = founder_alleles[f0, snp_idx]
            if f1 >= 0:
                allele_grid[b, 1, k_idx] = founder_alleles[f1, snp_idx]
    
    return allele_grid


def convert_id_grid_to_allele_grid(id_grid, bin_centers, founder_block, bin_width_bp=None):
    """
    Convert founder ID grid to single-SNP allele grid.
    
    Returns:
        allele_grid: (num_bins, 2) int8
    """
    num_bins = id_grid.shape[0]
    bin_indices = np.searchsorted(founder_block.positions, bin_centers)
    bin_indices = np.clip(bin_indices, 0, len(founder_block.positions) - 1)
    
    if bin_width_bp is None:
        if len(bin_centers) > 1:
            bin_width_bp = bin_centers[1] - bin_centers[0]
        else:
            bin_width_bp = 10000
    
    found_snps_pos = founder_block.positions[bin_indices]
    dist_to_center = np.abs(found_snps_pos - bin_centers)
    valid_snp_mask = dist_to_center <= (bin_width_bp / 2.0)
    
    hap_keys = sorted(list(founder_block.haplotypes.keys()))
    max_id = max(hap_keys) if hap_keys else 0
    allele_lookup = np.full((max_id + 1, num_bins), -1, dtype=np.int8)
    for fid, h_arr in founder_block.haplotypes.items():
        if h_arr.ndim == 2:
            raw_alleles = np.argmax(h_arr, axis=1)
        else:
            raw_alleles = h_arr
        extracted = raw_alleles[bin_indices]
        extracted[~valid_snp_mask] = -1
        allele_lookup[fid, :] = extracted
    
    allele_grid = np.full_like(id_grid, -1, dtype=np.int8)
    b_indices = np.arange(num_bins)
    for chrom in [0, 1]:
        ids = id_grid[:, chrom]
        valid_mask = (ids != -1)
        safe_ids = ids.copy()
        safe_ids[~valid_mask] = 0
        alleles = allele_lookup[safe_ids, b_indices]
        alleles[~valid_mask] = -1
        allele_grid[:, chrom] = alleles
    
    return allele_grid


def compute_ibs_hom_mask(allele_grid):
    """
    Derive homozygosity mask from allele identity across tracks.
    
    A bin is marked as potentially homozygous (phase-ambiguous) if the
    alleles on track 1 and track 2 are identical at ALL SNPs in the bin.
    This captures both:
      - True homozygosity (same founder on both tracks)
      - Effective homozygosity from IBS (different founders, same alleles)
    
    When the HMM sees a homozygous bin, it allows free phase switches,
    which is correct because the data cannot resolve which parental
    chromosome contributed which track.
    
    Args:
        allele_grid: (num_bins, 2, max_snps_per_bin) int8 for multi-SNP,
                     or (num_bins, 2) int8 for single-SNP
    
    Returns:
        hom_mask: (num_bins,) bool
    """
    if allele_grid.ndim == 3:
        # Multi-SNP: check all SNPs in each bin
        num_bins = allele_grid.shape[0]
        hom_mask = np.ones(num_bins, dtype=np.bool_)
        for b in range(num_bins):
            a0 = allele_grid[b, 0, :]  # track 1 alleles
            a1 = allele_grid[b, 1, :]  # track 2 alleles
            # Valid SNPs: both tracks have data
            valid = (a0 != -1) & (a1 != -1)
            if not np.any(valid):
                # No valid SNPs → treat as homozygous (no information)
                hom_mask[b] = True
            else:
                # Homozygous if ALL valid SNPs match
                hom_mask[b] = np.all(a0[valid] == a1[valid])
    else:
        # Single-SNP
        hom_mask = ((allele_grid[:, 0] == allele_grid[:, 1]) |
                    (allele_grid[:, 0] == -1) |
                    (allele_grid[:, 1] == -1))
    
    return hom_mask


def process_contig_for_pedigree(contig_idx, sample_start, sample_end,
                                 painting, founder_block,
                                 snps_per_bin, recomb_rate,
                                 max_snps_per_bin=10):
    """
    Process a contig's painting into allele grids with IBS-aware hom masks
    for pedigree inference.
    
    Called by _process_contig_batch in pedigree_inference.py.
    Accesses SamplePainting.chunks directly, derives the homozygosity mask
    from allele identity instead of consensus set overlap.
    
    Returns:
        dict with keys: contig_idx, sample_start, sample_end,
             sample_allele_grids, sw_costs, st_costs, num_bins, switch_counts
    """
    start_pos = painting.start_pos
    end_pos = painting.end_pos
    total_len = end_pos - start_pos
    approx_bp_per_bin = snps_per_bin * 100
    num_bins = int(total_len / approx_bp_per_bin)
    if num_bins < 100:
        num_bins = 100
    
    bin_edges = np.linspace(start_pos, end_pos, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bin_width = 10000.0
    if num_bins > 1:
        bin_width = bin_centers[1] - bin_centers[0]
    
    sample_allele_grids = []
    switch_counts = []
    
    for i in range(sample_start, sample_end):
        sample_obj = painting[i]
        
        # Access chunks directly — painting returns SamplePainting objects
        chunks = sample_obj.chunks if sample_obj.chunks else None
        
        # Discretize to founder ID grid
        id_grid = np.full((num_bins, 2), -1, dtype=np.int32)
        if chunks:
            c_ends = np.array([c.end for c in chunks])
            c_h1 = np.array([c.hap1 for c in chunks])
            c_h2 = np.array([c.hap2 for c in chunks])
            c_starts = np.array([c.start for c in chunks])
            indices = np.searchsorted(c_ends, bin_centers)
            indices = np.clip(indices, 0, len(chunks) - 1)
            valid_mask = bin_centers >= c_starts[indices]
            id_grid[:, 0] = np.where(valid_mask, c_h1[indices], -1)
            id_grid[:, 1] = np.where(valid_mask, c_h2[indices], -1)
        
        # Convert to allele grid
        if max_snps_per_bin > 1:
            allele_grid = convert_id_grid_to_allele_grid_multisnp(
                id_grid, bin_centers, founder_block, bin_width, max_snps_per_bin
            )
        else:
            allele_grid = convert_id_grid_to_allele_grid(
                id_grid, bin_centers, founder_block, bin_width
            )
        
        # Derive hom mask from allele identity (IBS-aware)
        hom_mask = compute_ibs_hom_mask(allele_grid)
        
        # Single entry per sample (no consensus combinations needed)
        sample_allele_grids.append([(allele_grid, hom_mask, 1.0)])
        
        # Switch counts
        switches = ((id_grid[:-1, :] != id_grid[1:, :]) &
                    (id_grid[:-1, :] != -1) & (id_grid[1:, :] != -1))
        switch_counts.append(np.sum(switches))
    
    # Transition costs
    dists = np.zeros(num_bins)
    dists[1:] = np.diff(bin_centers)
    theta = np.clip(1.0 - np.exp(-dists * recomb_rate), 1e-15, 0.5)
    sw_costs = np.log(theta)
    st_costs = np.log(1.0 - theta)
    
    return {
        'contig_idx': contig_idx,
        'sample_start': sample_start,
        'sample_end': sample_end,
        'sample_allele_grids': sample_allele_grids,
        'sw_costs': sw_costs,
        'st_costs': st_costs,
        'num_bins': num_bins,
        'switch_counts': np.array(switch_counts),
    }


def precompute_founder_ibs(founder_block, snps_per_bin=100):
    """
    Precompute pairwise IBS between all founder haplotypes, per bin.
    
    This is provided for diagnostic/visualization purposes. The main
    pipeline doesn't need it — the allele-level hom mask in compute_ibs_hom_mask
    implicitly captures IBS.
    
    Returns:
        ibs_matrix: (n_haps, n_haps, n_bins) bool — True if founders i,j
                     have identical alleles at all SNPs in bin b
        bin_centers: (n_bins,) float64
        hap_keys: sorted list of founder IDs
    """
    positions = founder_block.positions
    n_snps = len(positions)
    hap_keys = sorted(list(founder_block.haplotypes.keys()))
    n_haps = len(hap_keys)
    
    # Build allele matrix (n_haps, n_snps)
    max_id = max(hap_keys) if hap_keys else 0
    alleles = np.full((max_id + 1, n_snps), -1, dtype=np.int8)
    for fid, h_arr in founder_block.haplotypes.items():
        if h_arr.ndim == 2:
            alleles[fid, :] = np.argmax(h_arr, axis=1).astype(np.int8)
        else:
            alleles[fid, :] = h_arr.astype(np.int8)
    
    # Bin structure
    num_bins = max(1, n_snps // snps_per_bin)
    if num_bins < 100 and n_snps >= 100:
        num_bins = 100
    
    bin_edges = np.linspace(positions[0], positions[-1], num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Per-bin IBS check
    ibs_matrix = np.ones((max_id + 1, max_id + 1, num_bins), dtype=np.bool_)
    
    snp_bin_idx = np.searchsorted(bin_edges[1:], positions)
    snp_bin_idx = np.clip(snp_bin_idx, 0, num_bins - 1)
    
    for b in range(num_bins):
        snp_mask = snp_bin_idx == b
        if not np.any(snp_mask):
            continue
        bin_alleles = alleles[:, snp_mask]  # (n_haps, n_snps_in_bin)
        for i in range(n_haps):
            fi = hap_keys[i]
            for j in range(i + 1, n_haps):
                fj = hap_keys[j]
                is_ibs = np.all(bin_alleles[fi] == bin_alleles[fj])
                ibs_matrix[fi, fj, b] = is_ibs
                ibs_matrix[fj, fi, b] = is_ibs
    
    return ibs_matrix, bin_centers, hap_keys


def summarize_ibs_regions(ibs_matrix, bin_centers, hap_keys):
    """
    Print a summary of IBS regions between all founder pairs.
    Useful for understanding how much IBS-driven ambiguity exists.
    """
    n_bins = len(bin_centers)
    genome_len = bin_centers[-1] - bin_centers[0] if n_bins > 1 else 0
    bin_width = genome_len / max(n_bins - 1, 1)
    
    print(f"IBS Summary: {len(hap_keys)} founders, {n_bins} bins, "
          f"{genome_len/1e6:.1f} Mb")
    print(f"{'Pair':>10s}  {'IBS bins':>10s}  {'IBS fraction':>12s}  {'IBS Mb':>8s}")
    
    for i, fi in enumerate(hap_keys):
        for j, fj in enumerate(hap_keys):
            if j <= i:
                continue
            n_ibs = np.sum(ibs_matrix[fi, fj, :])
            frac = n_ibs / n_bins if n_bins > 0 else 0
            ibs_mb = n_ibs * bin_width / 1e6
            print(f"  ({fi},{fj}):  {n_ibs:10d}  {frac:12.3f}  {ibs_mb:8.1f}")