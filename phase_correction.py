"""
Phase Correction for Viterbi Paintings using Trio + Children Information.
REFACTORED: Now operates on BINS (like pedigree_inference) for ~150x speedup.

This module resolves phase ambiguity by considering:
1. Parent derivation: which parent haplotypes explain this sample's corrected haps
2. Child transmission: how this sample's corrected haps explain transmissions to children

Key insight: The painted tracks have arbitrary phase. We find phase[k] at each BIN
that minimizes total recombinations across parents AND children.

Algorithm:
- Pre-round: Initialize LLs using parent-derivation scores
- Rounds 1-3: Phase correction using parent + child information
- Post-round: Select best-LL painting for each sample
"""
import thread_config

import numpy as np
import pandas as pd
import math
from typing import List, Tuple, Dict, Optional, Set, Callable
from dataclasses import dataclass, field

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("WARNING: Numba not found. Phase correction will be slow.")
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator
    # Fallback prange == range when numba is unavailable so the
    # parallel-decorated kernels below still execute (just sequentially).
    prange = range

# ThreadPoolExecutor is used to parallelise within-generation sample
# work in `propagate_recoloring_to_offspring` and across F1 samples in
# `apply_parsimonious_f1_recoloring`.  Threads are appropriate (rather
# than processes) because the hot inner work in both cases is numpy
# operations -- `np.argmax`, `np.sum`, `np.searchsorted` -- which all
# release the GIL.  Threads avoid the pickling cost of sharing the
# large founder_block across worker processes.
from concurrent.futures import ThreadPoolExecutor

# =============================================================================
# FORKSERVER CONTEXT (used by phase correction + greedy refinement pools)
# =============================================================================
# Why forkserver and not the default fork:
#   - fork copies the entire main-process address space via COW which is
#     fast on Linux but fragile: forking a process that has running
#     numba/OMP/MKL threads can deadlock.  Mixing this with the rest of
#     the project (block_haplotypes.py, block_linking_naive.py, etc.,
#     all use forkserver) creates two start methods in one pipeline.
#   - forkserver workers are forked from a dedicated lightweight server
#     process that has only the preloaded modules from
#     `thread_config.py`.  This avoids the deadlock risk and matches
#     the rest of the pipeline.
#
# Implications of switching to forkserver (May 2026):
#   - Workers do NOT inherit main's `_PARALLEL_DATA` module global via
#     COW.  Instead the data must be passed explicitly via the Pool
#     initializer's `initargs`; `_init_phase_worker` writes the
#     received dict back into the worker's module-level
#     `_PARALLEL_DATA` so the rest of the worker code (which reads it
#     as a global) is unchanged.
#   - `load_fn` callbacks passed via _PARALLEL_DATA MUST be picklable.
#     Closures defined inside `if __name__ == '__main__':` are NOT
#     picklable; callers must promote any such loader to module top
#     level (see `pipeline._load_contig_for_phase_correction` for the
#     pattern).
#
# Fork fallback: if forkserver isn't available (e.g. Windows in tests),
# fall back to fork so the module still imports.  In that fallback the
# old behaviour (main's state inherited via COW) is restored
# automatically -- the explicit initargs path still works, just is a
# little redundant.
import multiprocessing as mp
try:
    _forkserver_ctx = mp.get_context('forkserver')
except ValueError:
    _forkserver_ctx = mp.get_context('fork')

# Import standard painting classes
from paint_samples import SamplePainting, PaintedChunk, BlockPainting

# =============================================================================
# HELPER: DENSE MATRIX CONVERSION (for external use / validation)
# =============================================================================

def founder_block_to_dense(founder_block):
    """
    Convert BlockResult to dense matrix for fast allele lookup.
    
    Args:
        founder_block: BlockResult with positions and haplotypes dict
        
    Returns:
        dense_haps: (max_founder_id + 1, n_sites) array of alleles
        positions: Array of SNP positions
    """
    positions = founder_block.positions
    hap_dict = founder_block.haplotypes
    
    if not hap_dict:
        return np.zeros((0, 0), dtype=np.int8), positions

    max_id = max(hap_dict.keys())
    n_sites = len(positions)
    
    dense_haps = np.full((max_id + 1, n_sites), -1, dtype=np.int8)
    
    for fid, hap_arr in hap_dict.items():
        if hap_arr.ndim == 2:
            concrete = np.argmax(hap_arr, axis=1)
        else:
            concrete = hap_arr
        dense_haps[fid, :] = concrete
        
    return dense_haps, positions


@njit(parallel=True, cache=True)
def _compute_founder_equivalence_matrix_kernel(
    dense_haps: np.ndarray,           # (n_founders, n_sites) int alleles, -1 = unknown
    bin_start_indices: np.ndarray,    # (n_bins,) site index inclusive
    bin_end_indices: np.ndarray,      # (n_bins,) site index exclusive
    max_diff_fraction: float,
    min_diff_sites: int,
) -> np.ndarray:
    """
    Numba-parallel kernel for compute_founder_equivalence_matrix.

    Mirrors the original Python/NumPy implementation's semantics
    exactly; parallelism is over bins (axis 0 of the output).  The
    inner founder pair loop and the inner per-site mismatch tally
    are written as explicit scalar loops so that numba can compile
    the whole bin's work into native code without temporary
    intermediate boolean arrays.

    Returns:
        equiv: (n_bins, n_founders, n_founders) bool array, with
               equiv[b, f1, f2] True iff founders f1 and f2 are
               equivalent at bin b (i.e. their non-(-1) sites
               differ at <= max(max_diff_fraction * n_sites_in_bin,
               min_diff_sites) positions, OR all comparable sites
               were -1 -> insufficient evidence -> treat as
               equivalent, OR the bin contains no SNPs at all ->
               every founder is trivially equivalent).
    """
    n_founders = dense_haps.shape[0]
    n_bins = bin_start_indices.shape[0]
    equiv = np.zeros((n_bins, n_founders, n_founders), dtype=np.bool_)

    for b in prange(n_bins):
        # Diagonal: every founder is equivalent to itself.  Set up
        # first so that "empty-bin" and "all -1" early-outs below
        # don't have to redo it.
        for f in range(n_founders):
            equiv[b, f, f] = True

        start_idx = bin_start_indices[b]
        end_idx = bin_end_indices[b]

        if end_idx <= start_idx:
            # No SNPs in this bin - all founders equivalent.  Matches
            # the original `equiv[b, :, :] = True` short-circuit.
            for f1 in range(n_founders):
                for f2 in range(n_founders):
                    equiv[b, f1, f2] = True
            continue

        n_sites_in_bin = end_idx - start_idx
        # max_allowed_diff = max(int(frac * n_sites), min_diff_sites)
        # Preserves the exact original Python computation including
        # the int() truncation toward zero.
        max_allowed_diff = int(max_diff_fraction * n_sites_in_bin)
        if max_allowed_diff < min_diff_sites:
            max_allowed_diff = min_diff_sites

        # Pairwise founder comparison.  Only the upper triangle
        # (f1 < f2) is computed; the lower triangle is mirrored on
        # write.  This matches the original's symmetric assignment
        # at lines 139-140 of the pre-change file.
        for f1 in range(n_founders):
            for f2 in range(f1 + 1, n_founders):
                n_diff = 0
                n_valid = 0
                # Count differing sites while ignoring sites where
                # either founder has -1 (unknown allele).  The
                # original used:
                #   valid_mask = (a1 != -1) & (a2 != -1)
                #   n_diff = sum((a1 != a2) & valid_mask)
                # which we unroll as a scalar accumulator loop so
                # numba can vectorise / parallelise it.
                for s in range(start_idx, end_idx):
                    a1 = dense_haps[f1, s]
                    a2 = dense_haps[f2, s]
                    if a1 == -1 or a2 == -1:
                        continue
                    n_valid += 1
                    if a1 != a2:
                        n_diff += 1

                if n_valid == 0:
                    # No valid sites to compare - assume equivalent
                    # (matches original's `equiv[b, f1, f2] = True;
                    # equiv[b, f2, f1] = True` branch).
                    equiv[b, f1, f2] = True
                    equiv[b, f2, f1] = True
                elif n_diff <= max_allowed_diff:
                    equiv[b, f1, f2] = True
                    equiv[b, f2, f1] = True

    return equiv


def compute_founder_equivalence_matrix(
    dense_haps: np.ndarray,
    positions: np.ndarray,
    bin_edges: np.ndarray,
    max_diff_fraction: float = 0.02,
    min_diff_sites: int = 2
) -> np.ndarray:
    """
    Compute founder equivalence matrix for each bin.
    
    Two founders are considered equivalent in a bin if they differ by no more than
    max(max_diff_fraction * sites_in_bin, min_diff_sites) sites.
    
    Args:
        dense_haps: (n_founders, n_sites) array of alleles
        positions: (n_sites,) array of SNP positions
        bin_edges: (n_bins + 1,) array of bin boundaries
        max_diff_fraction: Maximum fraction of differing sites for equivalence
        min_diff_sites: Minimum absolute number of differing sites threshold
    
    Returns:
        equiv: (n_bins, n_founders, n_founders) boolean array
               equiv[bin, f1, f2] = True if founders f1 and f2 are equivalent in bin

    Implementation note (May 2026):
        The per-bin pairwise comparison was originally a triple-nested
        Python loop (over bins, then over founder pairs).  This is
        called once per contig from `_process_contig_worker`, so it
        runs inside an already-parallel contig pool but is itself
        single-threaded -- for the long-tail contigs (chr3 with
        ~15k bins) this dominates the worker's wall time once the
        smaller contigs have finished.  The bin loop has been
        delegated to a `@njit(parallel=True)` kernel with `prange`
        over bins, giving ~num_cores speedup on the late-finishing
        workers and a few-x speedup on workers that retain dynamic
        Numba threads.  Semantics are byte-identical to the previous
        Python/NumPy implementation; the kernel docstring spells out
        the case analysis.
    """
    # Convert dense_haps to int32 for the numba kernel (matches the
    # int16/int32 dtype produced by founder_block_to_dense; safe for
    # the {-1, 0, 1, 2, 3} value range used by the alleles).  We do
    # not modify the caller's array.
    if dense_haps.dtype != np.int32:
        dense_haps_for_kernel = dense_haps.astype(np.int32)
    else:
        dense_haps_for_kernel = dense_haps

    # Find SNP indices for each bin using searchsorted.  Same logic
    # as the original implementation; we precompute them once and
    # pass to the kernel so the kernel can be a pure-numerical
    # function (no Python-level positions/bin_edges objects).
    bin_start_indices = np.searchsorted(positions, bin_edges[:-1], side='left').astype(np.int64)
    bin_end_indices = np.searchsorted(positions, bin_edges[1:], side='left').astype(np.int64)

    equiv = _compute_founder_equivalence_matrix_kernel(
        dense_haps_for_kernel,
        bin_start_indices,
        bin_end_indices,
        float(max_diff_fraction),
        int(min_diff_sites),
    )
    return equiv


@njit(nogil=True)
def check_equiv(equiv: np.ndarray, bin_idx: int, f1: int, f2: int) -> bool:
    """
    Check if two founders are equivalent in a given bin.
    Handles -1 (unknown) founders by returning True.
    """
    if f1 == -1 or f2 == -1:
        return True
    if f1 == f2:
        return True
    n_founders = equiv.shape[1]
    if f1 >= n_founders or f2 >= n_founders:
        return False  # Unknown founder ID
    return equiv[bin_idx, f1, f2]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ConsensusPaintingState:
    """Tracks a consensus painting through the correction rounds."""
    painting: SamplePainting
    ll: float = 0.0
    phase_sequence: np.ndarray = None  # The phase[k] values chosen (per bin)


@dataclass
class SampleCorrectionState:
    """Tracks all consensus paintings for a sample."""
    sample_name: str
    sample_idx: int
    consensus_states: List[ConsensusPaintingState]
    parent1_name: Optional[str] = None
    parent2_name: Optional[str] = None
    children_names: List[str] = field(default_factory=list)
    
    def get_best_consensus(self) -> ConsensusPaintingState:
        """Return the consensus with best LL."""
        if not self.consensus_states:
            return None
        return max(self.consensus_states, key=lambda x: x.ll)
    
    def get_best_painting(self) -> SamplePainting:
        """Return the painting with best LL."""
        best = self.get_best_consensus()
        return best.painting if best else None


# =============================================================================
# BINNING FUNCTIONS (adapted from pedigree_inference)
# =============================================================================

def discretize_painting_to_bins(painting: SamplePainting, bin_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretize a SamplePainting to fixed bins.
    
    Args:
        painting: SamplePainting with chunks
        bin_edges: Array of bin boundaries (n_bins + 1,)
    
    Returns:
        id_grid: (n_bins, 2) array of founder IDs per bin
        hom_mask: (n_bins,) boolean array - True where hap1 == hap2 or either is -1
    """
    num_bins = len(bin_edges) - 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    id_grid = np.full((num_bins, 2), -1, dtype=np.int32)
    
    chunks = painting.chunks if hasattr(painting, 'chunks') else []
    if not chunks:
        hom_mask = np.ones(num_bins, dtype=np.bool_)
        return id_grid, hom_mask
    
    # Build arrays for vectorized lookup
    c_ends = np.array([c.end for c in chunks], dtype=np.int64)
    c_starts = np.array([c.start for c in chunks], dtype=np.int64)
    c_h1 = np.array([c.hap1 for c in chunks], dtype=np.int32)
    c_h2 = np.array([c.hap2 for c in chunks], dtype=np.int32)
    
    # Find chunk for each bin center
    indices = np.searchsorted(c_ends, bin_centers, side='right')
    indices = np.clip(indices, 0, len(chunks) - 1)
    
    # Check if bin center is actually within the chunk
    valid_mask = (bin_centers >= c_starts[indices]) & (bin_centers < c_ends[indices])
    
    id_grid[:, 0] = np.where(valid_mask, c_h1[indices], -1)
    id_grid[:, 1] = np.where(valid_mask, c_h2[indices], -1)
    
    # Homozygosity mask: True if IDs match or either is -1 (uncertain)
    hom_mask = (id_grid[:, 0] == id_grid[:, 1]) | (id_grid[:, 0] == -1) | (id_grid[:, 1] == -1)
    
    return id_grid, hom_mask


def compute_bin_edges(start_pos: int, end_pos: int, snps_per_bin: int = 100) -> np.ndarray:
    """
    Compute bin edges for a genomic region.
    
    Args:
        start_pos: Start position of region
        end_pos: End position of region  
        snps_per_bin: Approximate SNPs per bin (controls resolution)
    
    Returns:
        bin_edges: (n_bins + 1,) array of bin boundaries
    """
    total_len = end_pos - start_pos
    approx_bp_per_bin = snps_per_bin * 100  # Assuming ~1 SNP per 100bp
    num_bins = max(100, int(total_len / approx_bp_per_bin))
    
    return np.linspace(start_pos, end_pos, num_bins + 1)


# =============================================================================
# 8-STATE VITERBI FOR CHILD TRANSMISSION EXTRACTION (BINNED)
# =============================================================================

@njit(fastmath=True, nogil=True)
def run_8state_transmission_viterbi_binned(
    n_bins: int,
    sample_grid: np.ndarray,    # (n_bins, 2) Sample's founder IDs
    other_grid: np.ndarray,     # (n_bins, 2) Other parent's founder IDs
    child_grid: np.ndarray,     # (n_bins, 2) Child's founder IDs
    child_hom_mask: np.ndarray, # (n_bins,) Child homozygosity mask
    bin_widths: np.ndarray,     # (n_bins,) Physical width of each bin in bp
    equiv: np.ndarray,          # (n_bins, n_founders, n_founders) Founder equivalence
    recomb_rate: float = 5e-8,  # Per-bp recombination rate
    mismatch_cost: float = 4.6  # Soft mismatch penalty (log scale)
) -> Tuple[np.ndarray, float]:
    """
    Run 8-state Viterbi on BINS to determine which of S's tracks was transmitted to child.
    
    States (3 bits):
        - Bit 0: Which of S's tracks is being transmitted (0=track1, 1=track2)
        - Bit 1: Which of O's tracks is being transmitted (0=track1, 1=track2)
        - Bit 2: Child phase (0=S→child_track1, 1=S→child_track2)
    
    Uses founder equivalence matrix to handle aliasing - two different founder IDs
    that are equivalent in a bin do not incur a mismatch penalty.
    
    Returns:
        source_from_s: (n_bins,) array where value is 0 if S transmitted track1, 1 if track2
        score: Total Viterbi score
    """
    n_states = 8
    n_founders = equiv.shape[1]
    
    scores = np.zeros(n_states, dtype=np.float64)
    new_scores = np.zeros(n_states, dtype=np.float64)
    backpointers = np.zeros((n_bins, n_states), dtype=np.int8)
    
    # Initialize
    for state in range(n_states):
        s_choice = (state >> 0) & 1
        o_choice = (state >> 1) & 1
        c_phase = (state >> 2) & 1
        
        s_val = sample_grid[0, 1] if s_choice else sample_grid[0, 0]
        o_val = other_grid[0, 1] if o_choice else other_grid[0, 0]
        
        if c_phase == 0:
            expect_c1, expect_c2 = s_val, o_val
        else:
            expect_c1, expect_c2 = o_val, s_val
        
        cost = 0.0
        # Check equivalence instead of exact match
        if expect_c1 != -1 and child_grid[0, 0] != -1:
            if not check_equiv(equiv, 0, expect_c1, child_grid[0, 0]):
                cost -= mismatch_cost
        if expect_c2 != -1 and child_grid[0, 1] != -1:
            if not check_equiv(equiv, 0, expect_c2, child_grid[0, 1]):
                cost -= mismatch_cost
        
        scores[state] = cost
    
    prev_hom = child_hom_mask[0]
    
    # Forward pass
    for k in range(1, n_bins):
        curr_hom = child_hom_mask[k]
        free_phase_switch = prev_hom or curr_hom
        
        # Compute transition cost based on physical distance
        dist_bp = bin_widths[k]
        theta = dist_bp * recomb_rate
        if theta > 0.5:
            theta = 0.5
        if theta < 1e-15:
            theta = 1e-15
        
        log_switch = math.log(theta)
        log_stay = math.log(1.0 - theta)
        
        for curr_state in range(n_states):
            curr_s = (curr_state >> 0) & 1
            curr_o = (curr_state >> 1) & 1
            curr_phase = (curr_state >> 2) & 1
            
            best_score = -1e20
            best_prev = 0
            
            for prev_state in range(n_states):
                prev_s = (prev_state >> 0) & 1
                prev_o = (prev_state >> 1) & 1
                prev_phase = (prev_state >> 2) & 1
                
                # Transition cost
                trans_cost = 0.0
                if prev_s != curr_s:
                    trans_cost += log_switch
                else:
                    trans_cost += log_stay
                if prev_o != curr_o:
                    trans_cost += log_switch
                else:
                    trans_cost += log_stay
                if prev_phase != curr_phase:
                    if not free_phase_switch:
                        trans_cost -= mismatch_cost * 5  # Heavy penalty for illegal phase switch
                
                score = scores[prev_state] + trans_cost
                if score > best_score:
                    best_score = score
                    best_prev = prev_state
            
            # Emission cost
            s_val = sample_grid[k, 1] if curr_s else sample_grid[k, 0]
            o_val = other_grid[k, 1] if curr_o else other_grid[k, 0]
            
            if curr_phase == 0:
                expect_c1, expect_c2 = s_val, o_val
            else:
                expect_c1, expect_c2 = o_val, s_val
            
            emit_cost = 0.0
            # Check equivalence instead of exact match
            if expect_c1 != -1 and child_grid[k, 0] != -1:
                if not check_equiv(equiv, k, expect_c1, child_grid[k, 0]):
                    emit_cost -= mismatch_cost
            if expect_c2 != -1 and child_grid[k, 1] != -1:
                if not check_equiv(equiv, k, expect_c2, child_grid[k, 1]):
                    emit_cost -= mismatch_cost
            
            new_scores[curr_state] = best_score + emit_cost
            backpointers[k, curr_state] = best_prev
        
        for s in range(n_states):
            scores[s] = new_scores[s]
        
        prev_hom = curr_hom
    
    # Find best final state
    best_final = 0
    best_final_score = scores[0]
    for s in range(1, n_states):
        if scores[s] > best_final_score:
            best_final_score = scores[s]
            best_final = s
    
    # Traceback
    path = np.zeros(n_bins, dtype=np.int8)
    curr = best_final
    for k in range(n_bins - 1, -1, -1):
        path[k] = curr
        if k > 0:
            curr = backpointers[k, curr]
    
    # Extract source_from_s (bit 0)
    source_from_s = np.zeros(n_bins, dtype=np.int8)
    for k in range(n_bins):
        source_from_s[k] = (path[k] >> 0) & 1
    
    return source_from_s, best_final_score


# =============================================================================
# PHASE CORRECTION VITERBI (BINNED)
# =============================================================================

@njit(fastmath=True, nogil=True)
def run_phase_correction_viterbi_binned(
    n_bins: int,
    sample_grid: np.ndarray,      # (n_bins, 2) Sample's founder IDs
    p1_grid: np.ndarray,          # (n_bins, 2) Parent 1's founder IDs
    p2_grid: np.ndarray,          # (n_bins, 2) Parent 2's founder IDs
    hom_mask: np.ndarray,         # (n_bins,) Sample homozygosity mask
    child_sources: np.ndarray,    # (n_children, n_bins) Which track each child received
    bin_widths: np.ndarray,       # (n_bins,) Physical width of each bin
    equiv: np.ndarray,            # (n_bins, n_founders, n_founders) Founder equivalence
    parent_assignment: int,       # 0: P1→Hap1, P2→Hap2; 1: P1→Hap2, P2→Hap1
    recomb_rate: float = 5e-8,    # Per-bp recombination rate
    phase_switch_cost: float = 20.0,  # Cost for illegal phase switch
    mismatch_cost: float = 4.6,   # Soft mismatch penalty per bin
    phase_zero_preference: float = 0.0,  # Per-bin penalty for choosing phase=1.
                                        # DISABLED by default (0.0).  A non-zero
                                        # value adds an arbitrary asymmetric
                                        # prior toward the input painting; A/B
                                        # testing showed this trades one set of
                                        # failures for another (fixes F3_103
                                        # chr1, breaks F3_114 chr1 and ~35
                                        # other samples) with no net accuracy
                                        # gain.  See docstring below for the
                                        # full justification.  Kept as a
                                        # parameter for experimentation; do
                                        # not enable without a principled
                                        # reason.
    use_xor_child_recomb: bool = True   # If True, child recombination cost
                                        # depends on whether the sample's
                                        # phase ALSO flipped at the same bin
                                        # (XOR formulation).  This is the
                                        # principled formulation that lets
                                        # child information actually
                                        # influence phase decisions.  The
                                        # historical (False) behavior makes
                                        # child cost state-independent, which
                                        # is provably a no-op for the argmax
                                        # over states — child info is
                                        # computed but ignored.  See the
                                        # docstring below ("Child
                                        # recombination cost — XOR
                                        # formulation") for the full
                                        # analysis.  Default True (the fix).
                                        # Set False to A/B-test against the
                                        # historical (buggy) behavior.
) -> Tuple[np.ndarray, float]:
    """
    Run phase correction Viterbi on BINS.
    
    State (3 bits):
        - Bit 0: phase (0=Track1→CorrHap1, 1=Track2→CorrHap1)
        - Bit 1: p1_choice (which P1 haplotype we're following)
        - Bit 2: p2_choice (which P2 haplotype we're following)
    
    Uses founder equivalence matrix to handle aliasing - two different founder IDs
    that are equivalent in a bin do not incur a mismatch penalty.
    
    Child recombination cost — XOR formulation (May 2026 fix):
        The previous implementation made child recombination cost
        INDEPENDENT of the parent's phase state, with the rationale that
        "a child's recombination is a biological event that happens
        regardless of how we label the parent's phase".  This was a
        misdiagnosis.  The cost then depends ONLY on whether
        child_sources[c, k] != child_sources[c, k-1], which is a fixed
        boolean computed before the Viterbi runs (via the 8-state
        transmission Viterbi).  That makes the child contribution to
        trans_cost identical for every (prev_state, curr_state) pair at
        bin k -- a constant offset that is invisible to the inner argmax
        over prev_state and to the outer argmax over the final state.
        Child information has zero effect on phase decisions despite
        being computed.  Verified by direct synthetic test: with sample
        = (0,1), parents = (0,1), all HET, running the Viterbi with
        child_sources = [[0,0,0,0,0]] vs [[0,1,0,1,0]] vs four children
        all of [[0,1,0,1,0]] (each scenario implying very different
        recombination patterns) produces THE SAME phase sequence
        [1,1,1,1,1] in all three cases, with scores differing by exactly
        n_recombs * (log_switch - log_stay), confirming the cost is a
        pure additive constant.
        
        Correct formulation: an APPARENT child track switch at bin k
        (child_sources[c, k] != child_sources[c, k-1]) is either a real
        biological recombination OR a labeling artifact resolved by the
        parent's phase flipping at the same bin.  Both possibilities
        produce the same observation (the painting's child_sources
        sequence), so the algorithm must choose between them.  The two
        cases combine as:
        
            apparent_recomb   = child_sources[c, k] != child_sources[c, k-1]
            phase_flipped     = curr_phase != prev_phase
            biological_recomb = apparent_recomb XOR phase_flipped
        
        Only biological_recomb pays log_switch; apparent recombs that
        coincide with a parent phase flip pay log_stay (the algorithm
        interprets them as labeling artifacts, not biological events).
        With this formulation the child cost now DEPENDS on
        (curr_phase, prev_phase), so it varies across (prev_state,
        curr_state) pairs and informs the phase decision.
        
        On the prior author's concern ("XOR allowed parent phase
        switches to hide child recombinations"): with phase_switch_cost
        = 20 and log_switch ≈ -7.6, the algorithm prefers the
        "labeling artifact" interpretation when 3+ children appear to
        recombine simultaneously at the same bin (breakeven:
        20 / 7.6 ≈ 2.63).  Three independent recombs at the same bin
        have prior probability ~theta^3 ≈ 1e-10 (with theta = 5e-4 for
        10 kb bins); a parent labeling artifact has prior probability
        ~e^(-phase_switch_cost) ≈ 2e-9, about 20x more likely.  So
        choosing the "labeling artifact" interpretation when 3+
        children coincide IS Bayesian-correct, not a bug.  The
        "hiding" the author worried about is the algorithm doing the
        right thing.
        
        The use_xor_child_recomb toggle (default True) controls this:
            - True:  XOR formulation (the fix).  Child info actually
                     informs phase decisions.
            - False: Historical state-independent cost.  Child info is
                     a constant offset, no effect on argmax.  Kept as a
                     toggle for direct A/B comparison against the
                     buggy behavior.

    phase_zero_preference (May 2026 — DISABLED by default, kept for reference):
        Without this term, the Viterbi has a fundamental symmetry: for any
        state (phase=X, p1=A, p2=B) under parent_assignment=0, the state
        (phase=1-X, p1=A, p2=B) under parent_assignment=1 has IDENTICAL
        emission and transition costs (both check
        sample[k,X] vs p1[k,A] and sample[k,1-X] vs p2[k,B], just in a
        different order).  This means score_0 == score_1 exactly, and the
        two phase_seqs are global complements of each other -- producing
        paintings that are global flips.  The orchestrator's >= tie-break
        in correct_sample_phase_binned picks PA=0 arbitrarily.

        Locally, the same symmetry shows up wherever the equivalence
        matrix declares a parent's two founders to be aliasing in a bin:
        emission is exactly tied between phase=0 and phase=1.  When the
        Viterbi enters such a tied stretch in phase=1 (from a correctly-
        flipped region upstream), it has no local evidence to switch
        back, so it propagates phase=1 -- sometimes through entire chunks
        that didn't need flipping.  This produced the F3_103 chr1 failure
        (36-42 Mb chunk with sample = founders {2, 5} and aliasing
        parents incorrectly flipped, dropping Track1 from 97.76% to
        95.15%).

        Adding a small per-bin penalty for phase=1 was meant to give a
        soft prior toward "stay close to tolerance" (since tolerance is
        usually mostly right -- Track1 acc >> 50% for any sample worth
        phase correction):
            - Mismatch_cost = 4.6 nats per mismatch.  preference = 0.1
              is 46x smaller -- single real mismatches always dominate.
            - phase_switch_cost = 20.  A tied region of length L flips
              from phase=1 back to phase=0 if L * preference > 20, i.e.
              L > 200 bins.

        **A/B-tested at 0.1 vs 0.0 on the full pipeline (320 samples,
        22 contigs).  The result was a wash:**
            - F3_103 chr1 fixed (Track1 95.15% -> ~99.9%)
            - F3_114 chr1 BROKEN (Track1 dropped to 96.85%)
            - ~35 other samples regressed slightly at the margin
            - Net: 6,763 perfect phasers -> 6,728 perfect phasers (-35)
            - Net Track1: 99.95% -> 99.95% (unchanged)
        The bias is symmetric: it helps samples whose Viterbi
        over-flipped against tolerance, and equally hurts samples whose
        Viterbi correctly flipped against tolerance.  Since both happen
        in roughly equal numbers across the cohort, the net effect on
        cohort accuracy is zero.

        Conclusion: this is a one-sided tie-break, not a principled fix.
        Disabled by default.  Kept as a parameter so it can be re-enabled
        for sample-specific debugging or comparison runs.  The real fix
        for tied-emission regions needs information the per-sample
        Viterbi doesn't currently see -- sibling agreement, multi-
        consensus / beam search, or allele-level evidence at sites where
        aliased founders actually diverge.
    
    Returns:
        phase_sequence: (n_bins,) array of phase values (0 or 1)
        score: Total score
    """
    n_states = 8
    n_children = child_sources.shape[0] if child_sources.size > 0 else 0
    
    scores = np.full(n_states, -1e20, dtype=np.float64)
    new_scores = np.zeros(n_states, dtype=np.float64)
    backpointers = np.zeros((n_bins, n_states), dtype=np.int8)
    
    # Initialize at bin 0
    for state in range(n_states):
        phase = (state >> 0) & 1
        p1_choice = (state >> 1) & 1
        p2_choice = (state >> 2) & 1
        
        # Corrected haplotypes based on phase
        if phase == 0:
            corr_h1 = sample_grid[0, 0]
            corr_h2 = sample_grid[0, 1]
        else:
            corr_h1 = sample_grid[0, 1]
            corr_h2 = sample_grid[0, 0]
        
        # Expected values from parents
        if parent_assignment == 0:
            expect_h1 = p1_grid[0, 1] if p1_choice else p1_grid[0, 0]
            expect_h2 = p2_grid[0, 1] if p2_choice else p2_grid[0, 0]
        else:
            expect_h1 = p2_grid[0, 1] if p2_choice else p2_grid[0, 0]
            expect_h2 = p1_grid[0, 1] if p1_choice else p1_grid[0, 0]
        
        cost = 0.0
        # Check equivalence instead of exact match
        if corr_h1 != -1 and expect_h1 != -1:
            if not check_equiv(equiv, 0, corr_h1, expect_h1):
                cost -= mismatch_cost
        if corr_h2 != -1 and expect_h2 != -1:
            if not check_equiv(equiv, 0, corr_h2, expect_h2):
                cost -= mismatch_cost

        # Phase-zero preference: small per-bin penalty for choosing phase=1
        # (= flipping this bin from the input/tolerance painting).  See
        # function docstring for full justification.  At bin 0 this is a
        # one-off prior favoring "start in phase=0"; combined with the
        # same penalty at every subsequent bin (forward pass below), it
        # accumulates over the trajectory and breaks symmetry ties in
        # favor of the painting closer to tolerance.
        cost -= phase_zero_preference * phase

        scores[state] = cost
    
    prev_hom = hom_mask[0]
    
    # Forward pass
    for k in range(1, n_bins):
        curr_hom = hom_mask[k]
        free_phase_switch = prev_hom or curr_hom
        
        # Transition costs based on physical distance
        dist_bp = bin_widths[k]
        theta = dist_bp * recomb_rate
        if theta > 0.5:
            theta = 0.5
        if theta < 1e-15:
            theta = 1e-15
        
        log_switch = math.log(theta)
        log_stay = math.log(1.0 - theta)
        
        for curr_state in range(n_states):
            curr_phase = (curr_state >> 0) & 1
            curr_p1 = (curr_state >> 1) & 1
            curr_p2 = (curr_state >> 2) & 1
            
            best_score = -1e20
            best_prev = 0
            
            for prev_state in range(n_states):
                prev_phase = (prev_state >> 0) & 1
                prev_p1 = (prev_state >> 1) & 1
                prev_p2 = (prev_state >> 2) & 1
                
                # Transition costs
                trans_cost = 0.0
                
                # Phase switch cost
                if prev_phase != curr_phase:
                    if not free_phase_switch:
                        trans_cost -= phase_switch_cost
                
                # Parent recombination costs
                if prev_p1 != curr_p1:
                    trans_cost += log_switch
                else:
                    trans_cost += log_stay
                if prev_p2 != curr_p2:
                    trans_cost += log_switch
                else:
                    trans_cost += log_stay
                
                # Child recombination costs.
                #
                # Two modes, controlled by use_xor_child_recomb:
                #
                #   True (default, the FIX):
                #     XOR formulation -- a child's APPARENT track switch at
                #     bin k (child_sources[c,k] != child_sources[c,k-1]) is
                #     interpreted as a biological recombination ONLY if the
                #     sample's phase didn't also flip at this bin; otherwise
                #     it is treated as a labeling artifact resolved by the
                #     phase flip.  This makes the child cost depend on
                #     (curr_phase XOR prev_phase) so it varies across
                #     (prev_state, curr_state) pairs and INFORMS the phase
                #     decision.  This is the principled formulation.
                #
                #   False (historical, BUG-COMPATIBLE):
                #     Cost depends only on apparent_recomb.  Because
                #     apparent_recomb is fixed per bin (not state-dependent),
                #     the cost adds the same constant to every transition at
                #     bin k.  Invisible to the argmax over prev_state, and
                #     also invisible to the final argmax over states because
                #     it adds the same total per bin to every path -- child
                #     info has ZERO effect on phase decisions.  Kept as a
                #     toggle so the fix can be A/B tested against the prior
                #     behavior directly.
                #
                # See module docstring for the full Bayesian justification
                # ("phase_switch_cost = 20 vs log_switch ≈ -7.6 means the
                # algorithm flips parent phase only when 3+ coincident
                # apparent recombs occur, which IS the Bayesian-correct
                # prior comparison: ~theta^3 ≈ 1e-10 vs ~e^-20 ≈ 2e-9").
                if use_xor_child_recomb:
                    phase_flipped = (prev_phase != curr_phase)
                    for c_idx in range(n_children):
                        apparent_recomb = (
                            child_sources[c_idx, k-1] != child_sources[c_idx, k]
                        )
                        # XOR of two bools via inequality.  biological_recomb
                        # is True iff (apparent ⊕ phase_flipped).
                        biological_recomb = apparent_recomb != phase_flipped
                        if biological_recomb:
                            trans_cost += log_switch
                        else:
                            trans_cost += log_stay
                else:
                    # Historical state-independent path (kept for A/B test).
                    for c_idx in range(n_children):
                        if child_sources[c_idx, k-1] != child_sources[c_idx, k]:
                            trans_cost += log_switch
                        else:
                            trans_cost += log_stay
                
                score = scores[prev_state] + trans_cost
                if score > best_score:
                    best_score = score
                    best_prev = prev_state
            
            # Emission cost
            if curr_phase == 0:
                corr_h1 = sample_grid[k, 0]
                corr_h2 = sample_grid[k, 1]
            else:
                corr_h1 = sample_grid[k, 1]
                corr_h2 = sample_grid[k, 0]
            
            if parent_assignment == 0:
                expect_h1 = p1_grid[k, 1] if curr_p1 else p1_grid[k, 0]
                expect_h2 = p2_grid[k, 1] if curr_p2 else p2_grid[k, 0]
            else:
                expect_h1 = p2_grid[k, 1] if curr_p2 else p2_grid[k, 0]
                expect_h2 = p1_grid[k, 1] if curr_p1 else p1_grid[k, 0]
            
            emit_cost = 0.0
            # Check equivalence instead of exact match
            if corr_h1 != -1 and expect_h1 != -1:
                if not check_equiv(equiv, k, corr_h1, expect_h1):
                    emit_cost -= mismatch_cost
            if corr_h2 != -1 and expect_h2 != -1:
                if not check_equiv(equiv, k, corr_h2, expect_h2):
                    emit_cost -= mismatch_cost

            # Phase-zero preference: small per-bin penalty for choosing
            # phase=1 (= flipping this bin from tolerance).  Applied at
            # every bin in the forward pass; cumulative effect is what
            # breaks the PA0/PA1 symmetry and avoids global-flip outputs
            # in samples where parents have ambiguous orientation in
            # large regions (e.g. F3_103's 36-42 Mb chunk on chr1).
            # See function docstring for the full analysis.
            emit_cost -= phase_zero_preference * curr_phase

            new_scores[curr_state] = best_score + emit_cost
            backpointers[k, curr_state] = best_prev
        
        for s in range(n_states):
            scores[s] = new_scores[s]
        
        prev_hom = curr_hom
    
    # Find best final state
    best_final = 0
    best_final_score = scores[0]
    for s in range(1, n_states):
        if scores[s] > best_final_score:
            best_final_score = scores[s]
            best_final = s
    
    # Traceback
    path = np.zeros(n_bins, dtype=np.int8)
    curr = best_final
    for k in range(n_bins - 1, -1, -1):
        path[k] = curr
        if k > 0:
            curr = backpointers[k, curr]
    
    # Extract phase sequence (bit 0)
    phase_sequence = np.zeros(n_bins, dtype=np.int8)
    for k in range(n_bins):
        phase_sequence[k] = (path[k] >> 0) & 1
    
    return phase_sequence, best_final_score


# =============================================================================
# PAINTING CONSTRUCTION FROM BINNED PHASE
# =============================================================================

@njit(nogil=True, cache=True)
def _build_corrected_painting_boundaries_kernel(
    corr_h1: np.ndarray,
    corr_h2: np.ndarray,
    bin_edges: np.ndarray,
    final_end: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba kernel for build_corrected_painting_from_bins.

    Walks the corrected-bin grids and identifies chunk boundaries
    (a new chunk starts whenever either track's value changes from
    the previous bin).  Returns four parallel int arrays
    (starts, ends, h1s, h2s) describing the resulting chunks in
    ascending order.  The Python wrapper below converts these into
    a list of PaintedChunk NamedTuples.

    Math is identical to the previous Python loop:
      - chunk_start = bin_edges[0] initially
      - emit a chunk whenever corr_h1[k] or corr_h2[k] differs from
        the previous chunk's (h1, h2)
      - the new chunk's start is the current bin's edge
      - after the loop, emit one final chunk extending to final_end

    The Python loop ran in O(n_bins) Python iterations holding the
    GIL; this kernel does the same scan in native code without
    holding the GIL, so threads in the round dispatch can actually
    overlap on this step.
    """
    n_bins = len(corr_h1)
    # Pre-allocate worst-case-size buffers (one chunk per bin) and
    # final-slice at the end.  Same pattern as
    # find_hom_to_het_boundaries.
    starts = np.empty(n_bins, dtype=np.int64)
    ends = np.empty(n_bins, dtype=np.int64)
    h1s = np.empty(n_bins, dtype=np.int32)
    h2s = np.empty(n_bins, dtype=np.int32)
    cnt = 0

    if n_bins == 0:
        return starts[:0], ends[:0], h1s[:0], h2s[:0]

    chunk_start = np.int64(bin_edges[0])
    chunk_h1 = corr_h1[0]
    chunk_h2 = corr_h2[0]

    for k in range(1, n_bins):
        if corr_h1[k] != chunk_h1 or corr_h2[k] != chunk_h2:
            # End current chunk at this bin's start (matches the
            # previous Python implementation's `chunk_end = int(bin_edges[k])`
            # plus `chunk_start = chunk_end` flow).
            chunk_end = np.int64(bin_edges[k])
            starts[cnt] = chunk_start
            ends[cnt] = chunk_end
            h1s[cnt] = chunk_h1
            h2s[cnt] = chunk_h2
            cnt += 1
            chunk_start = chunk_end
            chunk_h1 = corr_h1[k]
            chunk_h2 = corr_h2[k]

    # Final chunk extends to final_end (passed in by the Python
    # wrapper, which read it from original_painting.chunks[-1].end
    # or fell back to bin_edges[-1]).
    starts[cnt] = chunk_start
    ends[cnt] = final_end
    h1s[cnt] = chunk_h1
    h2s[cnt] = chunk_h2
    cnt += 1

    return starts[:cnt], ends[:cnt], h1s[:cnt], h2s[:cnt]


def build_corrected_painting_from_bins(
    original_painting: SamplePainting,
    phase_sequence: np.ndarray,
    bin_edges: np.ndarray,
    sample_idx: int
) -> SamplePainting:
    """
    Build corrected painting by applying bin-level phase sequence.
    
    Maps phase decisions back to original chunk boundaries for cleaner output.

    Implementation note (May 2026):
        The O(n_bins) boundary-finding scan is now in the njit
        kernel `_build_corrected_painting_boundaries_kernel`; this
        Python wrapper handles the SamplePainting / PaintedChunk
        construction (NamedTuple creation can't live inside njit).
        Net Python work is now O(n_chunks) -- typically <100 per
        sample even on chr3 -- versus the previous O(n_bins)
        ~15,000-iteration Python loop holding the GIL, which used
        to serialise all threads in the round dispatch on this
        step.  Math is unchanged; the kernel docstring spells out
        the equivalence to the previous Python loop.
    """
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    n_bins = len(bin_centers)
    
    # Get original tracks at bin centers
    id_grid, _ = discretize_painting_to_bins(original_painting, bin_edges)
    
    # Apply phase correction at bin level
    corr_h1 = np.where(phase_sequence == 0, id_grid[:, 0], id_grid[:, 1])
    corr_h2 = np.where(phase_sequence == 0, id_grid[:, 1], id_grid[:, 0])

    # Build chunks from corrected bin values
    if n_bins == 0:
        return SamplePainting(sample_idx, [])

    # Final chunk extends to original painting end (preserves the
    # input's end boundary in the corrected output)
    if original_painting.chunks:
        final_end = original_painting.chunks[-1].end
    else:
        final_end = int(bin_edges[-1])

    # Delegate the per-bin scan to the njit kernel.  The
    # astype(..., copy=False) is a no-op when np.where already
    # produced int32 (the common case, since id_grid is int32),
    # and only forces a cast on the rare path where numpy
    # upcasts.  bin_edges is already int64 from compute_bin_edges.
    starts, ends, h1s, h2s = _build_corrected_painting_boundaries_kernel(
        corr_h1.astype(np.int32, copy=False),
        corr_h2.astype(np.int32, copy=False),
        bin_edges,
        np.int64(final_end),
    )

    # O(n_chunks) Python construction of the NamedTuple list --
    # typically <100 iterations even for chr3.  This is the only
    # Python work left in the function; the previous O(n_bins)
    # scan is now native.
    chunks = [
        PaintedChunk(int(starts[i]), int(ends[i]), int(h1s[i]), int(h2s[i]))
        for i in range(len(starts))
    ]

    return SamplePainting(sample_idx, chunks)


# =============================================================================
# SINGLE SAMPLE PHASE CORRECTION (BINNED)
# =============================================================================

def correct_sample_phase_binned(
    sample_state: ConsensusPaintingState,
    p1_painting: Optional[SamplePainting],
    p2_painting: Optional[SamplePainting],
    children_paintings: List[SamplePainting],
    other_parent_paintings: List[SamplePainting],
    bin_edges: np.ndarray,
    equiv: np.ndarray,
    sample_idx: int,
    recomb_rate: float = 5e-8,
    phase_switch_cost: float = 20.0,
    mismatch_cost: float = 4.6,
    phase_zero_preference: float = 0.0,  # See run_phase_correction_viterbi_binned
                                        # docstring for full justification.
                                        # Per-bin penalty for choosing phase=1
                                        # (flipping from tolerance).  DISABLED
                                        # by default (0.0) -- A/B testing on
                                        # the full cohort showed it trades one
                                        # set of failures for another with no
                                        # net accuracy gain.  Kept as a
                                        # parameter for experimentation only.
    use_xor_child_recomb: bool = True   # Whether the child recombination cost
                                        # in the phase-correction Viterbi
                                        # should depend on the sample's phase
                                        # (XOR formulation, the fix) or be
                                        # state-independent (historical, no-op
                                        # for argmax).  Default True (the
                                        # fix).  See
                                        # run_phase_correction_viterbi_binned
                                        # docstring for full analysis.
) -> Tuple[SamplePainting, float, np.ndarray]:
    """
    Correct phase for a single consensus painting using BINNED data.
    
    Uses founder equivalence matrix to handle aliasing - two different founder IDs
    that are equivalent in a bin do not incur a mismatch penalty.
    
    Returns:
        corrected_painting: The phase-corrected painting
        total_ll: Total log-likelihood
        phase_sequence: The chosen phase at each bin
    """
    sample_painting = sample_state.painting
    n_bins = len(bin_edges) - 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Compute bin widths for distance-based transition costs
    bin_widths = np.diff(bin_edges)
    
    # Discretize sample painting
    sample_grid, hom_mask = discretize_painting_to_bins(sample_painting, bin_edges)
    
    # Discretize parent paintings (use all -1 if None)
    if p1_painting is not None:
        p1_grid, _ = discretize_painting_to_bins(p1_painting, bin_edges)
    else:
        p1_grid = np.full((n_bins, 2), -1, dtype=np.int32)
    
    if p2_painting is not None:
        p2_grid, _ = discretize_painting_to_bins(p2_painting, bin_edges)
    else:
        p2_grid = np.full((n_bins, 2), -1, dtype=np.int32)
    
    # Precompute child transmission sequences
    n_children = len(children_paintings)
    child_sources = np.zeros((n_children, n_bins), dtype=np.int8)
    
    for c_idx, (child_p, other_p) in enumerate(zip(children_paintings, other_parent_paintings)):
        child_grid, child_hom = discretize_painting_to_bins(child_p, bin_edges)
        other_grid, _ = discretize_painting_to_bins(other_p, bin_edges)
        
        source_seq, _ = run_8state_transmission_viterbi_binned(
            n_bins, sample_grid, other_grid, child_grid, child_hom,
            bin_widths, equiv, recomb_rate=recomb_rate, mismatch_cost=mismatch_cost
        )
        child_sources[c_idx, :] = source_seq
    
    # Run phase correction Viterbi for both parent assignments
    phase_seq_0, score_0 = run_phase_correction_viterbi_binned(
        n_bins, sample_grid, p1_grid, p2_grid, hom_mask, child_sources,
        bin_widths, equiv, parent_assignment=0,
        recomb_rate=recomb_rate, phase_switch_cost=phase_switch_cost,
        mismatch_cost=mismatch_cost,
        phase_zero_preference=phase_zero_preference,
        use_xor_child_recomb=use_xor_child_recomb
    )
    
    phase_seq_1, score_1 = run_phase_correction_viterbi_binned(
        n_bins, sample_grid, p1_grid, p2_grid, hom_mask, child_sources,
        bin_widths, equiv, parent_assignment=1,
        recomb_rate=recomb_rate, phase_switch_cost=phase_switch_cost,
        mismatch_cost=mismatch_cost,
        phase_zero_preference=phase_zero_preference,
        use_xor_child_recomb=use_xor_child_recomb
    )
    
    # Choose better assignment
    if score_0 >= score_1:
        phase_sequence = phase_seq_0
        total_ll = score_0
    else:
        phase_sequence = phase_seq_1
        total_ll = score_1
    
    # Build corrected painting
    corrected_painting = build_corrected_painting_from_bins(
        sample_painting, phase_sequence, bin_edges, sample_idx
    )
    
    return corrected_painting, total_ll, phase_sequence


# =============================================================================
# PARENT DERIVATION SCORE (for initialization)
# =============================================================================

@njit(nogil=True, cache=True)
def _compute_parent_derivation_score_kernel(
    sample_grid: np.ndarray,
    p1_grid: np.ndarray,
    p2_grid: np.ndarray,
    bin_widths: np.ndarray,
    recomb_rate: float,
) -> float:
    """
    Numba kernel for compute_parent_derivation_score_binned.

    Mathematically identical to the original Python implementation
    but rewritten without Python sets so the inner per-bin work can
    be njit-compiled.  The original used:
        s_founders = {sample_grid[k, 0], sample_grid[k, 1]} - {-1}
        p1_founders = {p1_grid[k, 0], p1_grid[k, 1]} - {-1}
        p2_founders = {p2_grid[k, 0], p2_grid[k, 1]} - {-1}
        parent_can_provide = p1_founders | p2_founders
        if s_founders and not s_founders.issubset(parent_can_provide):
            score -= 20.0
    Per-bin set construction + set operations in Python is extremely
    slow (~5us/bin); the explicit per-slot check below is byte-equivalent
    in semantics and runs in native code.

    The semantics are: "penalty -20 iff there is at least one non-(-1)
    sample founder that does NOT equal any non-(-1) parent founder slot".
    -1 in sample is skipped; -1 in parent is treated as "this slot is
    empty, can't match", which mirrors the `- {-1}` step in the original
    set construction.

    Verified equivalence cases:
      - Sample fully -1: s_founders = empty -> `if s_founders` is False
        -> no penalty.  Kernel: both s_h1, s_h2 == -1 -> early continue.
      - Sample known, parents fully -1: s_founders non-empty,
        parent_can_provide = {} -> not subset -> penalty.  Kernel:
        no parent slot matches s_h1 or s_h2 -> penalty.
      - Sample (5, 5), parents contain 5: s_founders = {5} (set
        dedupes), parent_can_provide contains 5 -> subset -> no
        penalty.  Kernel: both s_h1 and s_h2 covered separately -> no
        penalty.  (Single penalty per bin is preserved either way.)
      - Sample (5, 7), parents have only 5: s_founders = {5, 7},
        parent_can_provide has 5 -> NOT subset (7 missing) -> penalty.
        Kernel: s_h2 = 7 uncovered -> penalty (single -20).  Same.
    """
    n_bins = sample_grid.shape[0]
    score = 0.0

    # Penalty for unexplainable founder
    for k in range(n_bins):
        s_h1 = sample_grid[k, 0]
        s_h2 = sample_grid[k, 1]

        # Skip if sample fully uncertain at this bin -- matches the
        # original "s_founders is empty" branch (`if s_founders` is
        # False when both are -1 so we skip the penalty entirely).
        if s_h1 == -1 and s_h2 == -1:
            continue

        p1_h1 = p1_grid[k, 0]
        p1_h2 = p1_grid[k, 1]
        p2_h1 = p2_grid[k, 0]
        p2_h2 = p2_grid[k, 1]

        # Check whether s_h1 (if not -1) is matched by any non-(-1)
        # parent slot.  The `pf != -1` guard mirrors the `- {-1}`
        # removal from parent_can_provide in the original set
        # implementation.
        sh1_unexplained = False
        if s_h1 != -1:
            covered = False
            if p1_h1 != -1 and s_h1 == p1_h1:
                covered = True
            elif p1_h2 != -1 and s_h1 == p1_h2:
                covered = True
            elif p2_h1 != -1 and s_h1 == p2_h1:
                covered = True
            elif p2_h2 != -1 and s_h1 == p2_h2:
                covered = True
            if not covered:
                sh1_unexplained = True

        sh2_unexplained = False
        if s_h2 != -1:
            covered = False
            if p1_h1 != -1 and s_h2 == p1_h1:
                covered = True
            elif p1_h2 != -1 and s_h2 == p1_h2:
                covered = True
            elif p2_h1 != -1 and s_h2 == p2_h1:
                covered = True
            elif p2_h2 != -1 and s_h2 == p2_h2:
                covered = True
            if not covered:
                sh2_unexplained = True

        # Single penalty per bin whenever the sample's founder-set
        # is not a subset of parent_can_provide -- equivalent to
        # "at least one s founder uncovered".
        if sh1_unexplained or sh2_unexplained:
            score -= 20.0

    # Recombination penalty
    for k in range(1, n_bins):
        s_h1_prev = sample_grid[k-1, 0]
        s_h1_curr = sample_grid[k, 0]
        s_h2_prev = sample_grid[k-1, 1]
        s_h2_curr = sample_grid[k, 1]

        # Original used `changed = ...; changed |= ...` over two
        # booleans expressing "track1 had a real change" and "track2
        # had a real change".  A change is "real" iff both ends are
        # known (-1 is wildcard, not a change).  Equivalent boolean
        # logic below.
        changed_h1 = (s_h1_curr != s_h1_prev
                      and s_h1_curr != -1
                      and s_h1_prev != -1)
        changed_h2 = (s_h2_curr != s_h2_prev
                      and s_h2_curr != -1
                      and s_h2_prev != -1)

        if changed_h1 or changed_h2:
            theta = bin_widths[k] * recomb_rate
            if theta > 0.5:
                theta = 0.5
            if theta < 1e-15:
                theta = 1e-15
            score += math.log(theta) * 0.5

    return score


def compute_parent_derivation_score_binned(
    sample_painting: SamplePainting,
    p1_painting: SamplePainting,
    p2_painting: SamplePainting,
    bin_edges: np.ndarray,
    recomb_rate: float = 5e-8
) -> float:
    """
    Compute how well sample can be derived from parents (phase-agnostic, binned).

    Implementation note (May 2026):
        This is now a thin Python wrapper that discretises the three
        paintings to bin grids and then delegates the per-bin work to
        the `_compute_parent_derivation_score_kernel` njit function.
        The previous implementation had two pure-Python loops over
        n_bins with per-bin Python set construction, which dominated
        the wall-time of `initialize_lls` for large contigs
        (~30s for chr3 with 15k bins x 320 samples).  Math is
        unchanged -- the kernel docstring spells out the case-by-case
        equivalence.
    """
    sample_grid, _ = discretize_painting_to_bins(sample_painting, bin_edges)
    p1_grid, _ = discretize_painting_to_bins(p1_painting, bin_edges)
    p2_grid, _ = discretize_painting_to_bins(p2_painting, bin_edges)

    bin_widths = np.diff(bin_edges)

    return _compute_parent_derivation_score_kernel(
        sample_grid, p1_grid, p2_grid, bin_widths, recomb_rate
    )


# =============================================================================
# MAIN DRIVER FUNCTIONS
# =============================================================================

def initialize_correction_states(
    painting,  # BlockPainting
    pedigree_df: pd.DataFrame,
    sample_names: List[str],
    bin_edges: np.ndarray
) -> Dict[str, SampleCorrectionState]:
    """
    Initialize SampleCorrectionState for each sample.
    """
    # Build parent/child lookup
    parent_map = {}
    children_map = {name: [] for name in sample_names}
    
    for _, row in pedigree_df.iterrows():
        sample = row['Sample']
        p1 = row.get('Parent1')
        p2 = row.get('Parent2')
        
        if pd.notna(p1) and pd.notna(p2):
            parent_map[sample] = (p1, p2)
            if p1 in children_map:
                children_map[p1].append(sample)
            if p2 in children_map:
                children_map[p2].append(sample)
    
    states = {}
    
    for i, name in enumerate(sample_names):
        sample_painting = painting[i]
        chunks = sample_painting.chunks if hasattr(sample_painting, 'chunks') else []
        painting_copy = SamplePainting(i, list(chunks))
        consensus_states = [ConsensusPaintingState(painting=painting_copy, ll=0.0)]
        
        p1_name, p2_name = parent_map.get(name, (None, None))
        children_names = children_map.get(name, [])
        
        states[name] = SampleCorrectionState(
            sample_name=name,
            sample_idx=i,
            consensus_states=consensus_states,
            parent1_name=p1_name,
            parent2_name=p2_name,
            children_names=children_names
        )
    
    return states


def initialize_lls(
    states: Dict[str, SampleCorrectionState],
    bin_edges: np.ndarray
) -> None:
    """
    Initialize LL scores using parent-derivation score.
    """
    for name, state in states.items():
        if state.parent1_name is None or state.parent2_name is None:
            continue
        
        p1_state = states.get(state.parent1_name)
        p2_state = states.get(state.parent2_name)
        
        if p1_state is None or p2_state is None:
            continue
        
        p1_painting = p1_state.get_best_painting()
        p2_painting = p2_state.get_best_painting()
        
        if p1_painting is None or p2_painting is None:
            continue
        
        for cons_state in state.consensus_states:
            cons_state.ll = compute_parent_derivation_score_binned(
                cons_state.painting, p1_painting, p2_painting, bin_edges
            )


def run_correction_round(
    states: Dict[str, SampleCorrectionState],
    pedigree_df: pd.DataFrame,
    sample_names: List[str],
    bin_edges: np.ndarray,
    equiv: np.ndarray,
    recomb_rate: float = 5e-8,
    phase_switch_cost: float = 20.0,
    mismatch_cost: float = 4.6,
    phase_zero_preference: float = 0.0,  # Forwarded to correct_sample_phase_binned
                                          # and ultimately to the Viterbi.  See
                                          # run_phase_correction_viterbi_binned
                                          # for full justification.  DISABLED
                                          # by default; see that docstring for
                                          # the A/B-test result that motivated
                                          # the disabling.
    use_xor_child_recomb: bool = True,    # Forwarded to correct_sample_phase_binned
                                          # and to run_phase_correction_viterbi_binned.
                                          # Controls whether child recomb cost
                                          # in the phase-correction Viterbi
                                          # actually informs phase decisions
                                          # (True, default, the May 2026 fix)
                                          # or is state-independent and
                                          # therefore a no-op for the argmax
                                          # (False, historical buggy
                                          # behavior).  See
                                          # run_phase_correction_viterbi_binned
                                          # for the full analysis.
    n_threads: int = 1,                   # Number of ThreadPoolExecutor workers
                                          # for per-sample parallelism.  See the
                                          # JACOBI-vs-GAUSS-SEIDEL discussion in
                                          # the docstring.  Default 1 (sequential
                                          # Gauss-Seidel, matches the historical
                                          # behaviour byte-for-byte); set higher
                                          # to fan out across cores within a
                                          # contig worker.
    dynamic_threads_fn: Optional[Callable[[], int]] = None,
                                          # Optional callable returning the
                                          # currently-available thread budget
                                          # (e.g. _phase_get_dynamic_threads in
                                          # the multiprocess driver).  When
                                          # provided, the round is split into
                                          # per-generation sub-batches and the
                                          # callable is invoked before each
                                          # batch to re-check the budget;
                                          # surviving contigs can therefore
                                          # ramp up MID-ROUND as peer workers
                                          # finish, rather than only between
                                          # rounds.  When None (the default),
                                          # the legacy single-dispatch path is
                                          # used and n_threads is the fixed
                                          # worker count for the whole round.
                                          # Pure Jacobi semantics are preserved
                                          # either way: every batch reads from
                                          # the same `round_start_snapshot`,
                                          # and the merge-back happens once at
                                          # the end of the round.
    verbose: bool = True
) -> int:
    """
    Run one round of phase correction on all samples using BINNED data.
    
    Uses founder equivalence matrix to handle aliasing.

    JACOBI vs GAUSS-SEIDEL (May 2026 hybrid-parallelism update):
        The original implementation iterated `sample_names` sequentially
        and read each sample's parents and children via
        `states[name].get_best_painting()` AT THE TIME OF PROCESSING.
        Because mutations to `cons_state.painting` happen inside the
        loop, a sample processed later in the same round saw the
        within-round updates of earlier samples (Gauss-Seidel
        iteration).

        When n_threads > 1, we must instead process samples in parallel
        threads, which means all samples within a round must read a
        consistent view -- the round-start snapshot (Jacobi
        iteration).  This is implemented by:
          1) Walking `sample_names` once at round start to capture
             every sample's `state.get_best_painting()`; this dict
             (`round_start_snapshot`) is then frozen for the rest of
             the round.
          2) The per-sample worker (closure `_process_one_sample`)
             reads parent/child paintings from
             `round_start_snapshot`, not from the live `states` dict.
          3) Each worker computes its (corrected, new_ll, phase_seq)
             tuple for each of its consensus states but does NOT
             write back to `cons_state` -- those mutations happen
             sequentially in the merge-back loop after all workers
             return.

        Equivalence to Gauss-Seidel at the fixed point: both
        iteration schemes converge to the same paintings once
        `corrections_made == 0`.  Jacobi may take 1 extra round to
        get there in pathological cases; the round loop already
        runs to convergence (up to `num_rounds`), so this is benign.

        When n_threads == 1, we still take the snapshot but execute
        per-sample work sequentially.  The result is JACOBI iteration
        with the same sample order, NOT byte-identical to the
        historical Gauss-Seidel output for n_threads=1, but the
        converged answer is the same.  If exact reproduction of
        historical output is required, the caller can in principle
        run the loop sequentially with n_threads=1 and the snapshot
        replaced by live-read access -- this is not implemented
        because the converged answer is what downstream code uses.
    
    Returns:
        Number of corrections made
    """
    # Build other parent lookup
    other_parent_map = {}
    for _, row in pedigree_df.iterrows():
        child = row['Sample']
        p1 = row.get('Parent1')
        p2 = row.get('Parent2')
        
        if pd.notna(p1) and pd.notna(p2):
            other_parent_map[(p1, child)] = p2
            other_parent_map[(p2, child)] = p1

    # =====================================================================
    # Round-start snapshot for Jacobi iteration.  Captures the "best"
    # painting for every sample under the current `states` mutations,
    # so that all per-sample workers in this round see a consistent
    # view of their relatives.  See JACOBI vs GAUSS-SEIDEL note in the
    # docstring.
    # =====================================================================
    round_start_snapshot = {}
    for name in sample_names:
        st = states.get(name)
        if st is not None:
            round_start_snapshot[name] = st.get_best_painting()

    def _process_one_sample(name):
        """
        Per-sample worker -- safe to call concurrently across distinct
        names.  Reads parents/children from `round_start_snapshot`,
        reads its own consensus paintings from `states[name]`
        (own-sample reads are not contended across threads), and
        returns the list of (cons_state, corrected, new_ll, phase_seq)
        tuples WITHOUT mutating anything.  The merge-back loop after
        executor.map() applies all mutations in deterministic
        sample_names order.
        """
        state = states.get(name)
        if state is None:
            return name, None

        # Get parent paintings from the round-start snapshot (Jacobi)
        p1_painting = None
        p2_painting = None
        if state.parent1_name is not None:
            p1_painting = round_start_snapshot.get(state.parent1_name)
        if state.parent2_name is not None:
            p2_painting = round_start_snapshot.get(state.parent2_name)

        # Get children and other parents (also from the snapshot)
        children_paintings = []
        other_parent_paintings = []

        for child_name in state.children_names:
            child_painting = round_start_snapshot.get(child_name)
            if child_painting is None:
                continue

            other_parent_name = other_parent_map.get((name, child_name))
            if other_parent_name is None:
                continue

            other_painting = round_start_snapshot.get(other_parent_name)
            if other_painting is None:
                continue

            children_paintings.append(child_painting)
            other_parent_paintings.append(other_painting)

        # Skip if no information
        if p1_painting is None and p2_painting is None and not children_paintings:
            return name, None

        # Process each consensus painting and collect (cons_state,
        # corrected, new_ll, phase_seq) tuples for the merge-back step.
        per_cons_results = []
        for cons_state in state.consensus_states:
            corrected, new_ll, phase_seq = correct_sample_phase_binned(
                cons_state,
                p1_painting,
                p2_painting,
                children_paintings,
                other_parent_paintings,
                bin_edges,
                equiv,
                sample_idx=state.sample_idx,
                recomb_rate=recomb_rate,
                phase_switch_cost=phase_switch_cost,
                mismatch_cost=mismatch_cost,
                phase_zero_preference=phase_zero_preference,
                use_xor_child_recomb=use_xor_child_recomb
            )
            per_cons_results.append((cons_state, corrected, new_ll, phase_seq))
        return name, per_cons_results

    # ---- Dispatch ----
    # When `dynamic_threads_fn` is supplied, split the round into
    # per-generation sub-batches (F1 -> F2 -> F3 -> any others) and
    # re-check the thread budget before each batch.  This lets a
    # surviving contig worker scale UP mid-round as peer workers
    # finish, rather than being stuck with the round-start
    # allocation.  All batches read from the same
    # `round_start_snapshot`, so Jacobi semantics are unchanged --
    # only the dispatch shape changes.
    #
    # When `dynamic_threads_fn` is None we fall back to the legacy
    # single-dispatch path which preserves byte-for-byte the
    # previous behaviour for any external callers that don't
    # opt in to mid-round reallocation.
    if dynamic_threads_fn is None:
        # ---- Legacy single-dispatch path ----
        # Use the caller-supplied n_threads; clamp to [1, len(sample_names)].
        # When n_threads == 1, call sequentially (saves the ThreadPoolExecutor
        # setup/teardown cost when no parallelism is requested).
        effective_threads = max(1, min(n_threads, len(sample_names)))
        if effective_threads == 1:
            all_results = [_process_one_sample(name) for name in sample_names]
        else:
            with ThreadPoolExecutor(max_workers=effective_threads) as executor:
                all_results = list(executor.map(_process_one_sample, sample_names))
    else:
        # ---- Per-generation batched dispatch with mid-round reallocation ----
        # Build a generation lookup from the pedigree.  Use the
        # 'Generation' column when present; otherwise fall back to
        # name-prefix inference ('F1_...', 'F2_...', etc).  Samples
        # without a recognised generation go into an 'other' bucket
        # processed last.
        gen_map = {}
        if 'Generation' in pedigree_df.columns:
            for _, row in pedigree_df.iterrows():
                gen_map[row['Sample']] = row['Generation']

        # Group sample_names by generation, preserving original
        # within-generation order so the merge-back is deterministic.
        samples_by_gen: Dict[str, List[str]] = {}
        for name in sample_names:
            gen = gen_map.get(name)
            if gen is None or (isinstance(gen, float) and pd.isna(gen)):
                # Fallback inference from the sample-name prefix --
                # matches the convention used elsewhere in the
                # pipeline (e.g. pipeline.py's lambda for
                # Generation tagging on validation DataFrames).
                if name.startswith('F1'):
                    gen = 'F1'
                elif name.startswith('F2'):
                    gen = 'F2'
                elif name.startswith('F3'):
                    gen = 'F3'
                else:
                    gen = 'other'
            samples_by_gen.setdefault(gen, []).append(name)

        # Process in pedigree order (F1 -> F2 -> F3), then any other
        # generations sorted alphabetically.  F1 first is convenient
        # for two reasons: (a) it's the smallest batch (~20 samples)
        # so peer workers have time to start finishing before we hit
        # the big F3 batch, letting F2/F3 grab freed cores; (b) it
        # matches the dependency structure of the pedigree (F2s
        # read F1 paintings as parents, F3s read F2s) so the
        # processing order is intuitive even though Jacobi
        # semantics make it order-independent.
        gen_order = ['F1', 'F2', 'F3']
        extra_gens = sorted(g for g in samples_by_gen if g not in gen_order)
        process_order = gen_order + extra_gens

        all_results = []
        for gen in process_order:
            batch = samples_by_gen.get(gen, [])
            if not batch:
                continue

            # Re-check the thread budget before each batch.  This is
            # the whole point of the mid-round dispatch path: a
            # surviving contig worker scales UP here when peer
            # workers have finished since the previous batch.
            current_threads = dynamic_threads_fn()
            effective_threads = max(1, min(current_threads, len(batch)))

            if effective_threads == 1:
                batch_results = [_process_one_sample(name) for name in batch]
            else:
                with ThreadPoolExecutor(max_workers=effective_threads) as executor:
                    batch_results = list(executor.map(_process_one_sample, batch))

            all_results.extend(batch_results)

        # `all_results` is now in process_order (F1 then F2 then F3 then ...).
        # The merge-back loop below treats the list as order-agnostic
        # (each `cons_state` mutation is independent of the others),
        # but we re-sort by sample_names order to match the original
        # deterministic iteration so that any future code that
        # depends on per-call ordering of corrections_made
        # increments is unaffected.
        results_by_name = {name: per_cons for (name, per_cons) in all_results}
        all_results = [(name, results_by_name.get(name)) for name in sample_names]

    # ---- Sequential merge-back ----
    # Applies the per-sample mutations in deterministic order
    # (sample_names order) so corrections_made is a stable counter
    # regardless of how the executor scheduled the work.
    corrections_made = 0
    for name, per_cons_results in all_results:
        if per_cons_results is None:
            continue
        for cons_state, corrected, new_ll, phase_seq in per_cons_results:
            # Check if anything changed
            old_chunks = cons_state.painting.chunks
            new_chunks = corrected.chunks

            changed = (len(old_chunks) != len(new_chunks))
            if not changed:
                for oc, nc in zip(old_chunks, new_chunks):
                    if oc.hap1 != nc.hap1 or oc.hap2 != nc.hap2:
                        changed = True
                        break

            if changed:
                corrections_made += 1

            cons_state.painting = corrected
            cons_state.ll = new_ll
            cons_state.phase_sequence = phase_seq

    return corrections_made


def build_final_painting(
    states: Dict[str, SampleCorrectionState],
    sample_names: List[str],
    start_pos: int,
    end_pos: int
) -> BlockPainting:
    """
    Build final BlockPainting by selecting best consensus for each sample.
    """
    final_samples = []
    
    for name in sample_names:
        state = states[name]
        best_painting = state.get_best_painting()
        
        if best_painting is not None:
            final_samples.append(best_painting)
        else:
            final_samples.append(SamplePainting(state.sample_idx, []))
    
    return BlockPainting((start_pos, end_pos), final_samples)


# =============================================================================
# PUBLIC API
# =============================================================================

_PARALLEL_DATA = {}

# =============================================================================
# DYNAMIC THREAD REALLOCATION (mirrors hierarchical_assembly.py / paint_samples)
# =============================================================================
# Both `correct_phase_all_contigs` and
# `post_process_phase_greedy_all_contigs` dispatch one process per
# contig.  With 22 contigs and (typically) 112 cores, the outer pool
# alone uses only 22/112 = ~20% of the machine.  To use the full
# budget we layer DYNAMIC PER-CONTIG SCALING on top:
#
#   1) Each contig worker, on entering its task, registers itself in
#      a shared `mp.Value('i', 0)` counter (`_PHASE_ACTIVE_COUNTER`).
#   2) Between phases, the worker calls `_phase_get_dynamic_threads()`
#      which returns `total_cores // active_workers`, giving the
#      surviving workers a proportional share of the machine as peers
#      finish.  This is read lock-free; being briefly off by 1 or 2
#      is fine.
#   3) The worker releases its slot in a `finally:` block on exit.
#
# Inside a contig worker, the dynamic budget is consumed in two
# ALTERNATIVE ways (never both simultaneously, to avoid
# oversubscription):
#
#   (a) NUMBA-PARALLEL phases (`compute_founder_equivalence_matrix`):
#       call `numba.set_num_threads(dyn_threads)`, then let the
#       `@njit(parallel=True)` kernel use its `prange` loop to fan
#       out across cores.  No ThreadPoolExecutor.
#
#   (b) PYTHON-DISPATCH phases (the round loop in
#       `run_correction_round` and the pass loop in
#       `post_process_phase_greedy`):  set numba threads to 1
#       (so each per-sample njit call uses 1 numba thread) and
#       use a `ThreadPoolExecutor` with `dyn_threads` workers.
#       Each Python thread calls into single-threaded njit
#       kernels (`run_phase_correction_viterbi_binned`,
#       `run_8state_transmission_viterbi_binned`,
#       `compute_parent_matching_score_fixed_phase`) which release
#       the GIL inside, giving true parallelism.
#
# JACOBI vs GAUSS-SEIDEL semantics:
#   The original `run_correction_round` and `post_process_phase_greedy`
#   processed samples sequentially and a sample reading its parent
#   would see the parent's update FROM EARLIER IN THE SAME ROUND
#   (Gauss-Seidel iteration).  Under per-sample threading we must
#   snapshot all paintings at round start so that every sample reads
#   a consistent view (Jacobi iteration).  Both schemes converge to
#   the same fixed point; Jacobi may need 1 extra round.  See the
#   round-loop docstrings for details.

_PHASE_ACTIVE_COUNTER = None   # mp.Value('i', 0) — shared across all workers
_PHASE_TOTAL_CORES = None      # Total machine cores (e.g. 112)


def _phase_get_dynamic_threads():
    """
    Compute optimal thread count for this worker based on active peers.

    Uses total_cores // active_workers, clamped to [1, total_cores].
    Lock-free read of `_PHASE_ACTIVE_COUNTER.value` -- a slightly
    stale count (off by 1-2) is fine, since we recheck between
    every major phase.  The cost of being briefly wrong is a few
    seconds of mild over/under-subscription, not correctness.
    """
    if _PHASE_ACTIVE_COUNTER is None or _PHASE_TOTAL_CORES is None:
        return 1
    active = max(_PHASE_ACTIVE_COUNTER.value, 1)
    return max(1, _PHASE_TOTAL_CORES // active)


def _init_phase_worker(total_cores, active_counter, parallel_data):
    """
    Pool initializer -- called once per worker at creation time.

    Sets the numba thread pool ceiling to total_cores (not the
    outer pool size) so that workers can later scale up their
    thread count as peers finish.  The actual active thread
    count starts at 1 and is adjusted by each phase via
    `numba.set_num_threads()` or by sizing a `ThreadPoolExecutor`.

    With OMP PASSIVE or TBB threading layers (selected by
    `thread_config.py`), idle threads in an oversized pool sleep
    and consume zero CPU, so setting the ceiling high is safe.

    `parallel_data` is the dict that under the OLD fork-based
    Pool was inherited as a module global via COW.  Under
    forkserver workers do not inherit main's address space, so
    the data must be passed explicitly via initargs; this
    function unpacks it into the module-level `_PARALLEL_DATA`
    so the rest of the worker code (which reads it as a global)
    is unchanged.  See the FORKSERVER CONTEXT comment at the top
    of this module for the full rationale.
    """
    try:
        import numba
        # Set ceiling so set_num_threads can scale up later
        numba.config.NUMBA_NUM_THREADS = total_cores
        # Start conservative — each phase will set the real value
        numba.set_num_threads(1)
    except Exception:
        pass

    global _PARALLEL_DATA, _PHASE_ACTIVE_COUNTER, _PHASE_TOTAL_CORES
    _PARALLEL_DATA = parallel_data if parallel_data is not None else {}
    _PHASE_ACTIVE_COUNTER = active_counter
    _PHASE_TOTAL_CORES = total_cores


def _process_contig_worker(r_name):
    """Worker function for processing a single contig."""
    global _PARALLEL_DATA, _PHASE_ACTIVE_COUNTER

    # ------------------------------------------------------------------
    # Per-phase timing instrumentation (May 2026 diagnostic).
    # `_phase_times` is populated as the worker progresses; printed at
    # the end so the user can see where each contig's wall-clock goes.
    # This is essential for tuning the dynamic-thread budget and
    # identifying bottlenecks (load_fn vs equiv-matrix vs rounds vs
    # serialization).  Cost is negligible (a few time.time() calls
    # per worker).
    # ------------------------------------------------------------------
    import time as _t
    _worker_start = _t.time()
    _phase_times = {}
    _t0 = _worker_start

    # Load data: use load_fn if provided, otherwise read from multi_contig_results
    load_fn = _PARALLEL_DATA.get('load_fn')
    if load_fn is not None:
        data = load_fn(r_name)
    else:
        data = _PARALLEL_DATA['multi_contig_results'][r_name]
    _phase_times['load'] = _t.time() - _t0

    pedigree_df = _PARALLEL_DATA['pedigree_df']
    sample_names = _PARALLEL_DATA['sample_names']
    num_rounds = _PARALLEL_DATA['num_rounds']
    snps_per_bin = _PARALLEL_DATA['snps_per_bin']
    recomb_rate = _PARALLEL_DATA['recomb_rate']
    max_diff_fraction = _PARALLEL_DATA.get('max_diff_fraction', 0.02)
    min_diff_sites = _PARALLEL_DATA.get('min_diff_sites', 2)
    # phase_zero_preference: defaults to 0.0 if not present in _PARALLEL_DATA
    # (back-compat for any caller that pickled an older _PARALLEL_DATA dict).
    # The 0.0 default matches the per-call default in correct_phase_all_contigs,
    # so a missing key means "use the disabled-by-default bias", not a stale
    # 0.1 value from a prior implementation.
    phase_zero_preference = _PARALLEL_DATA.get('phase_zero_preference', 0.0)
    # Match the run_correction_round defaults for phase_switch_cost /
    # mismatch_cost (forwarded via _PARALLEL_DATA when set) so the workers
    # use a consistent set of Viterbi hyperparameters.
    phase_switch_cost = _PARALLEL_DATA.get('phase_switch_cost', 20.0)
    mismatch_cost = _PARALLEL_DATA.get('mismatch_cost', 4.6)
    # use_xor_child_recomb: defaults to True if not present in _PARALLEL_DATA
    # (back-compat for any caller that pickled an older _PARALLEL_DATA dict).
    # The True default matches the per-call default in correct_phase_all_contigs
    # -- it is the May 2026 XOR fix that makes child info actually influence
    # the phase decision.  A missing key therefore means "use the fix", not
    # "fall back to the buggy state-independent behavior".
    use_xor_child_recomb = _PARALLEL_DATA.get('use_xor_child_recomb', True)
    
    tolerance_painting = data['tolerance_result']
    
    start_pos = tolerance_painting.start_pos
    end_pos = tolerance_painting.end_pos

    # =====================================================================
    # Dynamic thread reallocation -- register as active for the duration
    # of this worker.  The shared counter is decremented in the `finally`
    # so that surviving workers can scale up when this contig finishes.
    # See "DYNAMIC THREAD REALLOCATION" header comment above for the full
    # design.  Everything below `try:` is the existing contig-processing
    # code, with the equivalence matrix and round loop scaled to the
    # currently-available thread budget.
    # =====================================================================
    if _PHASE_ACTIVE_COUNTER is not None:
        with _PHASE_ACTIVE_COUNTER.get_lock():
            _PHASE_ACTIVE_COUNTER.value += 1
    try:
        try:
            import numba
        except ImportError:
            numba = None

        # Compute bin edges
        _t0 = _t.time()
        bin_edges = compute_bin_edges(start_pos, end_pos, snps_per_bin=snps_per_bin)
        n_bins = len(bin_edges) - 1
        _phase_times['bin_edges'] = _t.time() - _t0

        # ----------------------------------------------------------------
        # Phase 1: Founder equivalence matrix.
        # This is a NUMBA-PARALLEL phase (the kernel uses prange over
        # bins; see compute_founder_equivalence_matrix's docstring).
        # Give it the full dynamic thread budget so the prange fan-out
        # uses every core currently allocated to this worker.
        # ----------------------------------------------------------------
        _t0 = _t.time()
        if numba is not None:
            dyn_threads = _phase_get_dynamic_threads()
            numba.set_num_threads(dyn_threads)

        # Compute founder equivalence matrix
        if 'founder_block' in data:
            dense_haps, positions = founder_block_to_dense(data['founder_block'])
            equiv = compute_founder_equivalence_matrix(
                dense_haps, positions, bin_edges,
                max_diff_fraction=max_diff_fraction,
                min_diff_sites=min_diff_sites
            )
        else:
            # Fallback: no equivalence (identity only)
            equiv = np.zeros((n_bins, 8, 8), dtype=np.bool_)
            for b in range(n_bins):
                for f in range(8):
                    equiv[b, f, f] = True
        _phase_times['equiv'] = _t.time() - _t0

        # Initialize states
        _t0 = _t.time()
        states = initialize_correction_states(
            tolerance_painting, pedigree_df, sample_names, bin_edges
        )
        _phase_times['init_states'] = _t.time() - _t0

        # Initialize LLs
        _t0 = _t.time()
        initialize_lls(states, bin_edges)
        _phase_times['init_lls'] = _t.time() - _t0

        # ----------------------------------------------------------------
        # Phase 2: Correction rounds.
        # This is a PYTHON-DISPATCH phase: each round runs the per-sample
        # Viterbi via a ThreadPoolExecutor.  Set numba threads to 1 so
        # the python threads do not over-subscribe.  Re-check
        # _phase_get_dynamic_threads() at the start of every round so
        # late-finishing peers' freed cores are picked up between rounds.
        # ----------------------------------------------------------------
        if numba is not None:
            numba.set_num_threads(1)

        # Run correction rounds
        final_round = num_rounds
        converged = False  # True iff `corrections == 0` was actually
                           # reached inside the loop.  Distinguishes
                           # genuine convergence from hitting the
                           # num_rounds ceiling (under Jacobi iteration
                           # the algorithm typically needs 1 round more
                           # than Gauss-Seidel; if num_rounds is set too
                           # low we want to see this explicitly rather
                           # than silently report "converged round N").
        _rounds_total_t0 = _t.time()
        per_round_times = []
        per_round_threads = []
        for round_idx in range(num_rounds):
            _t0 = _t.time()
            dyn_threads = _phase_get_dynamic_threads()
            per_round_threads.append(dyn_threads)
            corrections = run_correction_round(
                states, pedigree_df, sample_names, bin_edges, equiv,
                recomb_rate=recomb_rate,
                phase_switch_cost=phase_switch_cost,
                mismatch_cost=mismatch_cost,
                phase_zero_preference=phase_zero_preference,
                use_xor_child_recomb=use_xor_child_recomb,
                n_threads=dyn_threads,
                # Pass the dynamic thread-budget callable so
                # run_correction_round can split the round into
                # per-generation batches and re-check thread
                # allocation between them.  See "Mid-round
                # reallocation" in run_correction_round's
                # `dynamic_threads_fn` parameter docstring.  This is
                # how chr3 (or any other late-finishing contig)
                # picks up freed cores from peer workers WITHIN a
                # single round, not just between rounds.
                dynamic_threads_fn=_phase_get_dynamic_threads,
                verbose=False
            )
            per_round_times.append(_t.time() - _t0)

            if corrections == 0:
                final_round = round_idx + 1
                converged = True
                break
        _phase_times['rounds_total'] = _t.time() - _rounds_total_t0
        _phase_times['per_round'] = per_round_times
        _phase_times['per_round_threads'] = per_round_threads

        # Build final painting
        _t0 = _t.time()
        final_painting = build_final_painting(states, sample_names, start_pos, end_pos)
        _phase_times['build_final'] = _t.time() - _t0

        multi_consensus = sum(1 for s in states.values() if len(s.consensus_states) > 1)

        # ------------------------------------------------------------------
        # Print compact one-line timing summary (sortable so chr3 stands
        # out from chr1).  Each line goes to stdout; main process collects
        # them via the imap_unordered result print loop, but since this
        # print happens INSIDE the worker, ordering is by completion time.
        # ------------------------------------------------------------------
        _wall = _t.time() - _worker_start
        _per_r = ",".join(f"{t:.2f}s/{n}t" for t, n in zip(per_round_times, per_round_threads))
        print(f"  [TIMING {r_name:>5}] wall={_wall:.1f}s | "
              f"load={_phase_times['load']:.1f} eq={_phase_times['equiv']:.1f} "
              f"init_states={_phase_times['init_states']:.2f} "
              f"init_lls={_phase_times['init_lls']:.2f} "
              f"rounds={_phase_times['rounds_total']:.1f}({_per_r}) "
              f"build={_phase_times['build_final']:.2f} | n_bins={n_bins}", flush=True)

        # NOTE: We deliberately do NOT return founder_block here.  Previous
        # versions did, which forced 22 workers to pickle and pipe back
        # ~30-50 GB of founder_block data sequentially through Pool's
        # IPC channel -- that was responsible for ~40s of "ghost time"
        # between worker completion (wall ~12s) and the function returning
        # (total ~58s) as observed in May 2026 timing diagnostics.
        # Downstream stages (greedy refinement, etc.) load founder_block
        # themselves via their own load_fn / _ensure_key pattern, so
        # main process never needs the founder_block in memory at all.
        return (r_name, final_painting, multi_consensus, final_round, converged)
    finally:
        if _PHASE_ACTIVE_COUNTER is not None:
            with _PHASE_ACTIVE_COUNTER.get_lock():
                _PHASE_ACTIVE_COUNTER.value -= 1


def correct_phase_all_contigs(
    multi_contig_results: Dict,
    pedigree_df: pd.DataFrame,
    sample_names: List[str],
    num_rounds: int = 3,
    snps_per_bin: int = 100,
    recomb_rate: float = 5e-8,
    max_diff_fraction: float = 0.02,
    min_diff_sites: int = 2,
    phase_switch_cost: float = 20.0,
    mismatch_cost: float = 4.6,
    phase_zero_preference: float = 0.0,  # Per-bin penalty for choosing phase=1
                                          # (= flipping from tolerance).
                                          # DISABLED by default.  See
                                          # run_phase_correction_viterbi_binned
                                          # docstring for the A/B-test result
                                          # showing this is a wash on cohort
                                          # accuracy.  Kept as a parameter for
                                          # debugging / comparison runs only.
    use_xor_child_recomb: bool = True,    # Whether the child recombination cost
                                          # in the phase-correction Viterbi
                                          # depends on the sample's phase (XOR
                                          # formulation, the May 2026 fix) or
                                          # is state-independent (historical
                                          # buggy behavior).  Default True (the
                                          # fix).  Set False to A/B-test
                                          # against the historical behavior in
                                          # which child information was
                                          # computed but did not influence
                                          # phase decisions.  See
                                          # run_phase_correction_viterbi_binned
                                          # for the full analysis and proof
                                          # via synthetic test.
    verbose: bool = True,
    max_workers: int = None,
    parallel: bool = True,
    load_fn=None
) -> Dict:
    """
    Correct phase for all contigs using BINNED data (~150x faster than per-SNP).
    
    Uses founder equivalence to handle aliasing - two founders that are nearly
    identical in a bin are treated as equivalent for emission scoring.
    
    Args:
        multi_contig_results: Dict mapping region_name -> data dict containing:
            - 'tolerance_result': BlockPainting
            - 'founder_block': BlockResult with founder haplotypes (optional but recommended)
            If load_fn is provided, this dict only needs the contig names as keys.
        pedigree_df: DataFrame with Sample, Parent1, Parent2 columns
        sample_names: List of sample names
        num_rounds: Number of correction rounds (default 3)
        snps_per_bin: Bin resolution (default 150, same as pedigree inference)
        recomb_rate: Per-bp recombination rate (default 5e-8)
        max_diff_fraction: Max fraction of differing sites for founder equivalence (default 2%)
        min_diff_sites: Min absolute number of differing sites for equivalence (default 2)
        phase_switch_cost: Penalty for illegal phase switches in Viterbi (default 20.0)
        mismatch_cost: Soft mismatch penalty per bin in Viterbi (default 4.6)
        phase_zero_preference: Per-bin penalty for choosing phase=1 (default 0.0).
            DISABLED by default after A/B testing.  When set to a non-zero
            value (e.g. 0.1), adds a soft prior toward the input painting in
            tied-emission regions.  This was found to be a one-sided
            tie-break: it helps some samples while hurting others by an
            equal amount, with no net accuracy gain.  See
            run_phase_correction_viterbi_binned docstring for the full
            analysis.  Kept as a parameter for experimentation.
        use_xor_child_recomb: Whether the child recombination cost in the
            phase-correction Viterbi uses the XOR formulation that lets
            child information actually influence phase decisions (True,
            default, the May 2026 fix), or the historical state-independent
            formulation that makes child cost a no-op for the argmax
            (False).  Verified by synthetic test: with the historical
            formulation, three scenarios with very different child_sources
            (no apparent recombs vs alternating recombs vs 4 children all
            alternating) produce the IDENTICAL phase sequence, with scores
            differing only by an additive constant equal to n_recombs
            * (log_switch - log_stay) — proving child info is dead weight
            under that formulation.  Set False to A/B-test against the
            historical behavior.  See run_phase_correction_viterbi_binned
            docstring for the full analysis and Bayesian justification.
        verbose: Print progress
        max_workers: REQUIRED.  Maximum parallel workers.  Must be passed
            from the calling pipeline (e.g. pipeline.py's `n_processes`)
            so the phase-correction stages respect the pipeline's
            machine-wide worker budget.  Passing None will raise.
        parallel: Use parallel processing
        load_fn: Optional callable(r_name) -> dict with 'tolerance_result' and 'founder_block'.
                 If provided, workers load their own data (parallelizes I/O).
    
    Returns:
        Updated multi_contig_results with 'corrected_painting' key added
    """
    # NOTE: `import multiprocessing as mp` used to live HERE (function-
    # local) so this module could be imported in environments where
    # multiprocessing isn't available.  After the May 2026 switch to
    # forkserver, the import has been promoted to module top level
    # alongside the `_forkserver_ctx` construction -- see the
    # FORKSERVER CONTEXT comment at the top of this module.  The
    # function-local import is removed here.

    if max_workers is None:
        raise ValueError(
            "correct_phase_all_contigs: max_workers must be specified "
            "(typically pipeline.py passes its n_processes here).  The "
            "previous os.cpu_count() fallback has been removed so that "
            "all phase-correction workers respect the pipeline's "
            "machine-wide worker budget.")

    if load_fn is not None:
        # With load_fn, workers load their own data — keys just need contig names
        contig_names = list(multi_contig_results.keys())
    else:
        contig_names = [
            r_name for r_name, data in multi_contig_results.items()
            if 'tolerance_result' in data
        ]
    n_contigs = len(contig_names)
    # `max_workers` is the TOTAL CORE BUDGET passed by the pipeline (e.g.
    # n_processes = 112).  The OUTER POOL SIZE is the smaller of that
    # budget and the number of contigs -- no point in starting more
    # processes than there are tasks.  But we keep `total_cores` =
    # max_workers as the dynamic-threading ceiling: with 22 contigs and
    # 112 total cores, each worker starts with 112//22 = 5 threads, and
    # as contigs finish, surviving workers scale up via
    # `_phase_get_dynamic_threads()`.
    total_cores = max_workers
    outer_pool_size = max(1, min(total_cores, n_contigs)) if n_contigs > 0 else total_cores
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Phase Correction (BINNED, {snps_per_bin} SNPs/bin, {num_rounds} rounds)")
        if parallel and n_contigs > 1:
            print(f"Using {outer_pool_size} outer workers x dynamic threads "
                  f"(total budget = {total_cores} cores)")
            if load_fn is not None:
                print(f"Workers will load data from checkpoints (parallel I/O)")
        print(f"{'='*60}")
    
    if parallel and n_contigs > 1:
        # Build the `parallel_data` payload that will be pickled into
        # each forkserver worker as part of the Pool initializer's
        # `initargs`.  Under the OLD fork-based code path this was
        # set as a module-level global (`_PARALLEL_DATA = {...}`) and
        # inherited by workers via COW; forkserver doesn't have that
        # luxury so the payload travels by pickle.
        #
        # CRITICAL (May 2026 hot-fix):  `multi_contig_results` MUST
        # be omitted from the payload when `load_fn` is set.  At this
        # point in the pipeline -- after pedigree inference (Stage 11)
        # has populated multi_contig_results with stage-9 founder
        # blocks (~30-50 GB across 22 contigs) and stage-10
        # tolerance_result paintings (~5+ GB per chr3) -- the dict can
        # hold tens of GB of data.  Pickling it into initargs for
        # 22 workers blows main-process RAM (the pickle byte string
        # alone is tens of GB, peak ~2x during serialization) and
        # hangs the pipeline before any worker is spawned.
        #
        # When `load_fn` is set, the worker uses load_fn exclusively
        # (see `_process_contig_worker`: the `multi_contig_results`
        # lookup is in the `else` branch of `if load_fn is not None`)
        # so omitting it costs us nothing.  When `load_fn` is None,
        # the worker DOES need multi_contig_results to find its
        # tolerance_result, so we still include it on that path; that
        # path is the historical in-memory (no-checkpoint) flow which
        # is only used in tests / small datasets where the dict is
        # small.
        parallel_data = {
            'pedigree_df': pedigree_df,
            'sample_names': sample_names,
            'num_rounds': num_rounds,
            'snps_per_bin': snps_per_bin,
            'recomb_rate': recomb_rate,
            'max_diff_fraction': max_diff_fraction,
            'min_diff_sites': min_diff_sites,
            # Viterbi hyperparameters forwarded to workers
            'phase_switch_cost': phase_switch_cost,
            'mismatch_cost': mismatch_cost,
            'phase_zero_preference': phase_zero_preference,
            'use_xor_child_recomb': use_xor_child_recomb,
            'load_fn': load_fn,
        }
        if load_fn is None:
            # In-memory mode -- worker needs multi_contig_results to
            # locate its tolerance_result.  Include the FULL dict
            # (no checkpoints available to load from).
            parallel_data['multi_contig_results'] = multi_contig_results
        
        if verbose:
            print(f"\nProcessing {n_contigs} contigs in parallel...")

        # Shared active-worker counter for dynamic thread reallocation.
        # Created from the same forkserver context as the pool so the
        # underlying mp.sharedctypes machinery is consistent (mixing
        # contexts is a known footgun in CPython multiprocessing).
        # See `_init_phase_worker` and the `_PHASE_ACTIVE_COUNTER`
        # machinery at the top of the public-API section.
        active_counter = _forkserver_ctx.Value('i', 0)

        with _forkserver_ctx.Pool(
            processes=outer_pool_size,
            initializer=_init_phase_worker,
            initargs=(total_cores, active_counter, parallel_data),
        ) as pool:
            results = pool.map(_process_contig_worker, contig_names)
        
        for r_name, final_painting, multi_cons, final_round, converged in results:
            multi_contig_results.setdefault(r_name, {})['corrected_painting'] = final_painting
            # NOTE: founder_block intentionally NOT included in worker
            # return tuple (saves ~40s of pickle/pipe transfer for the
            # 30-50 GB of founder_block data across 22 contigs).
            # Downstream stages load founder_block themselves via load_fn.
            if verbose:
                # Distinguish genuine convergence from hitting num_rounds.
                # Under Jacobi iteration (introduced with the May 2026
                # within-contig threading), the round count to reach
                # `corrections == 0` is typically one higher than under
                # Gauss-Seidel.  If num_rounds is set too low the
                # algorithm may exit without true convergence; this is
                # surfaced clearly here rather than masked behind a
                # vague "converged round N" message.
                status = (f"converged round {final_round}" if converged
                          else f"HIT MAX ROUNDS ({final_round}) WITHOUT CONVERGENCE")
                print(f"  {r_name}: {status}, multi-consensus: {multi_cons}")
        
        # NOTE: We no longer write back into module-level _PARALLEL_DATA
        # in the main process -- the data lives in the workers' globals
        # (set by `_init_phase_worker` from the initargs payload above)
        # and is discarded when the pool tears down.  Main's
        # _PARALLEL_DATA stays {} from initial module load.
    
    else:
        for r_name in contig_names:
            if load_fn is not None:
                data = load_fn(r_name)
            else:
                data = multi_contig_results[r_name]
            
            if verbose:
                print(f"\nProcessing {r_name}...")
            
            tolerance_painting = data['tolerance_result']
            start_pos = tolerance_painting.start_pos
            end_pos = tolerance_painting.end_pos
            
            # Compute bin edges
            bin_edges = compute_bin_edges(start_pos, end_pos, snps_per_bin=snps_per_bin)
            n_bins = len(bin_edges) - 1
            
            if verbose:
                print(f"  Region: {start_pos:,} - {end_pos:,} ({n_bins} bins)")
            
            # Compute founder equivalence matrix
            if 'founder_block' in data:
                dense_haps, positions = founder_block_to_dense(data['founder_block'])
                equiv = compute_founder_equivalence_matrix(
                    dense_haps, positions, bin_edges,
                    max_diff_fraction=max_diff_fraction,
                    min_diff_sites=min_diff_sites
                )
                if verbose:
                    n_founders = dense_haps.shape[0]
                    print(f"  Computed founder equivalence matrix ({n_founders} founders)")
            else:
                # Fallback: no equivalence (identity only)
                # Assume at most 8 founders
                equiv = np.zeros((n_bins, 8, 8), dtype=np.bool_)
                for b in range(n_bins):
                    for f in range(8):
                        equiv[b, f, f] = True
                if verbose:
                    print(f"  WARNING: No founder_block found, using identity equivalence only")
            
            # Initialize states
            states = initialize_correction_states(
                tolerance_painting, pedigree_df, sample_names, bin_edges
            )
            
            if verbose:
                print(f"  Initializing LLs...")
            initialize_lls(states, bin_edges)
            
            # Run correction rounds
            for round_idx in range(num_rounds):
                if verbose:
                    print(f"  Round {round_idx + 1}/{num_rounds}...")
                
                corrections = run_correction_round(
                    states, pedigree_df, sample_names, bin_edges, equiv,
                    recomb_rate=recomb_rate,
                    phase_switch_cost=phase_switch_cost,
                    mismatch_cost=mismatch_cost,
                    phase_zero_preference=phase_zero_preference,
                    use_xor_child_recomb=use_xor_child_recomb,
                    verbose=verbose
                )
                
                if verbose:
                    print(f"    Corrections made: {corrections}")
                
                if corrections == 0:
                    if verbose:
                        print(f"    Converged early!")
                    break
            
            final_painting = build_final_painting(states, sample_names, start_pos, end_pos)
            multi_contig_results.setdefault(r_name, {})['corrected_painting'] = final_painting
            # Also keep founder_block for greedy refinement
            if 'founder_block' in data:
                multi_contig_results[r_name]['founder_block'] = data['founder_block']
            
            if verbose:
                multi_consensus = sum(1 for s in states.values() if len(s.consensus_states) > 1)
                print(f"  Samples with multiple consensus: {multi_consensus}")
    
    if verbose:
        print(f"\nPhase correction complete.")
    
    return multi_contig_results


# =============================================================================
# GREEDY PHASE POST-PROCESSING
# =============================================================================

@njit(nogil=True, cache=True)
def find_hom_to_het_boundaries(hom_mask: np.ndarray) -> np.ndarray:
    """
    Find all bin indices where a HOM→HET transition occurs.
    
    A HOM→HET boundary at bin k means hom_mask[k-1] is True and hom_mask[k] is False.
    These are points where phase flips are biologically meaningful.
    
    Returns:
        np.ndarray (int64) of bin indices where HOM→HET transitions occur.
        (Previously a Python list; converted to ndarray as part of the
        May 2026 hot-loop njit pass.  Consumers iterate with enumerate
        or read len() in the same way, so no caller change required.)
    """
    n = len(hom_mask)
    # Allocate a worst-case-size buffer and slice down at the end.
    # numba doesn't support list.append on heterogeneous types in
    # nopython mode efficiently, so we use an explicit counter +
    # final slice; semantically identical to the previous
    # `boundaries.append(k)` loop.
    out = np.empty(n, dtype=np.int64)
    cnt = 0
    for k in range(1, n):
        if hom_mask[k-1] and not hom_mask[k]:
            out[cnt] = k
            cnt += 1
    return out[:cnt]


@njit(nogil=True, cache=True)
def find_double_recomb_boundaries(sample_grid: np.ndarray) -> np.ndarray:
    """
    Find all bin indices where a double recombination occurs (both tracks change).
    
    A double recomb at bin k means:
    - sample_grid[k-1, 0] != sample_grid[k, 0] (track 1 changed)
    - sample_grid[k-1, 1] != sample_grid[k, 1] (track 2 changed)
    
    At these points, phase is ambiguous because flipping assigns the new founders
    to opposite tracks while preserving the same number of recombinations.
    
    Example:
        Before: track1=[A], track2=[B]
        After:  track1=[C], track2=[D]
        
        Original assignment: A→C on track1, B→D on track2 (2 recombs)
        Flipped assignment:  A→D on track1, B→C on track2 (still 2 recombs!)
        
    So flipping at this boundary is "free" - both are equally valid.
    
    Returns:
        np.ndarray (int64) of bin indices where double recombinations occur.
        (Previously a Python list; converted to ndarray as part of the
        May 2026 hot-loop njit pass.)
    """
    n_bins = sample_grid.shape[0]
    # Worst-case-size buffer + final slice (see find_hom_to_het_boundaries
    # for rationale).
    out = np.empty(n_bins, dtype=np.int64)
    cnt = 0
    
    for k in range(1, n_bins):
        # Get founder IDs
        prev_h1, prev_h2 = sample_grid[k-1, 0], sample_grid[k-1, 1]
        curr_h1, curr_h2 = sample_grid[k, 0], sample_grid[k, 1]
        
        # Skip if any is uncertain
        if prev_h1 == -1 or prev_h2 == -1 or curr_h1 == -1 or curr_h2 == -1:
            continue
        
        # Check if both tracks changed
        track1_changed = (prev_h1 != curr_h1)
        track2_changed = (prev_h2 != curr_h2)
        
        if track1_changed and track2_changed:
            out[cnt] = k
            cnt += 1
    
    return out[:cnt]


@njit(nogil=True, cache=True)
def find_all_valid_flip_boundaries(
    sample_grid: np.ndarray,
    hom_mask: np.ndarray
) -> np.ndarray:
    """
    Find all valid flip boundaries: HOM→HET transitions OR double recombinations.
    
    Phase can be flipped at:
    1. HOM→HET transitions: before the transition, phase is arbitrary (both tracks same)
    2. Double recombinations: both tracks change, so swapping destinations is equivalent
    
    Note: HOM→HET with double recomb is already covered by case 1 (the HOM part
    means both tracks were the same, so phase was already arbitrary).
    
    Returns:
        np.ndarray (int64) of unique bin indices where flips are valid, sorted ascending.
        (Previously sorted(set(a) | set(b)) producing a Python list; replaced
        with np.unique(concatenate(a, b)) -- semantically identical because
        np.unique returns sorted unique values.  Part of the May 2026
        hot-loop njit pass.)
    """
    hom_het_boundaries = find_hom_to_het_boundaries(hom_mask)
    double_recomb_boundaries = find_double_recomb_boundaries(sample_grid)
    
    # Combine and deduplicate.  `np.unique(concatenate(a, b))` is the
    # numba-friendly equivalent of `sorted(set(a) | set(b))`: it
    # concatenates the two arrays, sorts them, and returns the unique
    # values.  Output is sorted ascending, matching the previous
    # `sorted(...)` result.
    merged = np.concatenate((hom_het_boundaries, double_recomb_boundaries))
    return np.unique(merged)


@njit(nogil=True, cache=True)
def _apply_phase_flip_to_grid_kernel(
    sample_grid: np.ndarray,
    start_bin: int,
    end_bin: int,
) -> np.ndarray:
    """
    Numba kernel for apply_phase_flip_to_grid -- end_bin must be a
    concrete int (sentinel handling is done by the wrapper below).
    Body is identical to the original Python implementation; only
    the wrapping changed.
    """
    flipped = sample_grid.copy()

    # Swap columns in the flipped region
    flipped[start_bin:end_bin, 0] = sample_grid[start_bin:end_bin, 1]
    flipped[start_bin:end_bin, 1] = sample_grid[start_bin:end_bin, 0]

    return flipped


def apply_phase_flip_to_grid(
    sample_grid: np.ndarray,
    start_bin: int,
    end_bin: Optional[int] = None
) -> np.ndarray:
    """
    Apply a phase flip to a sample grid from start_bin to end_bin (exclusive).
    
    If end_bin is None, flip from start_bin to end of chromosome.
    Phase flip swaps columns 0 and 1.
    
    Returns:
        New grid with flip applied (does not modify input)

    Implementation note (May 2026):
        The Optional[int] = None sentinel is handled here in the
        Python wrapper because numba's nopython mode does not support
        Optional types in argument signatures.  The numerical work
        (a .copy() plus two slice assignments) is delegated to the
        njit kernel `_apply_phase_flip_to_grid_kernel`.  Saves
        ~5-10us of Python dispatch overhead per call, which adds up
        across the ~200 calls per greedy iteration in the singleton +
        pair flip enumeration loops.
    """
    if end_bin is None:
        end_bin = len(sample_grid)
    return _apply_phase_flip_to_grid_kernel(sample_grid, int(start_bin), int(end_bin))


@njit(nogil=True, cache=True)
def check_mendelian_consistency(
    sample_grid: np.ndarray,
    p1_grid: np.ndarray,
    p2_grid: np.ndarray,
    equiv: np.ndarray
) -> bool:
    """
    Check if sample_grid is Mendelian-consistent with parents.

    For each bin, each (non -1) founder in the sample must be EQUIVALENT to at
    least one parent founder at that bin under the founder equivalence matrix.
    Returns True if consistent, False otherwise.

    Why use the equivalence matrix here?
    The previous implementation did plain set membership on raw founder IDs
    (s_h1 in {p1_grid[k,0], p1_grid[k,1], p2_grid[k,0], p2_grid[k,1]}).  Every
    OTHER scoring path in this module -- run_phase_correction_viterbi_binned,
    run_8state_transmission_viterbi_binned, compute_parent_matching_score_-
    fixed_phase -- uses check_equiv to grant zero mismatch cost when two
    different founder IDs are equivalent at a bin (i.e. their alleles match
    closely under the founder_equivalence threshold).  The old Mendelian
    check did not, so a greedy candidate that the rest of the algorithm
    would happily score as fully consistent could be rejected here before
    its score was even computed.  Using check_equiv aligns the Mendelian
    pre-filter with what the scoring functions actually treat as a match.

    Handling of -1 (uncertain) founders:
    check_equiv returns True when either operand is -1 (see its docstring).
    The semantics are therefore:
      - Sample fully -1 at bin k:                   skip the bin entirely.
      - Sample known, all four parent slots are -1: passes (-1 is wildcard;
                                                    parents provide no info).
      - Sample known, parents partially -1:         passes if the known
                                                    sample matches any of the
                                                    parent slots (a -1 slot
                                                    counts as a potential
                                                    match via check_equiv).
      - Sample known, parents all known but no
        equivalent parent slot exists for s_h1
        (or s_h2):                                  fails (true conflict).

    NOTE on the "partially -1 parents" case: the previous implementation was
    STRICTER here -- it would fail unless the sample matched one of the
    *known* parent slots, treating -1 as "no constraint from this slot but
    don't let it whitewash a mismatch elsewhere".  The new implementation
    treats -1 as a wildcard, matching what check_equiv does and what the
    Viterbi assumes (the Viterbi pays no mismatch cost when comparing
    against -1).  This is intentional: the entire point of this change is
    to align with the scoring functions, and the scoring functions assume
    wildcard -1.  In practice this case is rare (parents are usually fully
    painted) and the relaxation only matters when -1 actually appears.

    Args:
        sample_grid: (n_bins, 2) array of sample founder IDs
        p1_grid:     (n_bins, 2) array of parent 1 founder IDs
        p2_grid:     (n_bins, 2) array of parent 2 founder IDs
        equiv:       (n_bins, n_founders, n_founders) bool equivalence matrix
                     where equiv[k, f1, f2] is True iff founders f1 and f2
                     are equivalent at bin k under the equivalence threshold.
    """
    n_bins = sample_grid.shape[0]

    for k in range(n_bins):
        s_h1, s_h2 = sample_grid[k, 0], sample_grid[k, 1]

        # Skip if sample fully uncertain at this bin -- no constraint to
        # check.  Matches the previous "if s_h1 == -1 and s_h2 == -1: continue"
        # short-circuit.
        if s_h1 == -1 and s_h2 == -1:
            continue

        # Collect parent founder slots at this bin.  Use a 4-tuple (rather
        # than a set as the previous implementation did) so we can iterate
        # in a fixed order and call check_equiv on each slot individually.
        # Sets would also work, but the asymmetric wildcard semantics of
        # check_equiv make per-slot iteration clearer.
        parent_founders = (p1_grid[k, 0], p1_grid[k, 1],
                           p2_grid[k, 0], p2_grid[k, 1])

        # s_h1 must be equivalent to at least one parent founder slot.
        # check_equiv handles all -1 cases:
        #   - s_h1 == -1 is already short-circuited by the outer skip when
        #     both s_h1 and s_h2 are -1; if only s_h1 is -1 we still need
        #     to check s_h2, but we don't need to check s_h1 against parents
        #     because -1 trivially matches everything.  The `if s_h1 != -1`
        #     guard below makes this explicit (matching the previous code's
        #     `if s_h1 != -1 and ...` structure).
        #   - pf == -1 is treated as wildcard match by check_equiv.
        if s_h1 != -1:
            matched = False
            for pf in parent_founders:
                if check_equiv(equiv, k, s_h1, pf):
                    matched = True
                    break
            if not matched:
                return False

        if s_h2 != -1:
            matched = False
            for pf in parent_founders:
                if check_equiv(equiv, k, s_h2, pf):
                    matched = True
                    break
            if not matched:
                return False

    return True


@njit(fastmath=True, nogil=True)
def compute_parent_matching_score_fixed_phase(
    n_bins: int,
    sample_grid: np.ndarray,      # (n_bins, 2) Already-corrected sample
    p1_grid: np.ndarray,          # (n_bins, 2) Parent 1
    p2_grid: np.ndarray,          # (n_bins, 2) Parent 2
    bin_widths: np.ndarray,
    equiv: np.ndarray,            # (n_bins, n_founders, n_founders) Founder equivalence
    recomb_rate: float = 5e-8,
    mismatch_cost: float = 4.6
) -> float:
    """
    Compute parent matching score for a fixed (already-corrected) sample painting.
    
    Uses a 4-state Viterbi over (p1_choice, p2_choice) since phase is fixed.
    Tries both parent assignments and returns the better score.
    
    Uses founder equivalence matrix to handle aliasing.
    """
    n_states = 4  # 2 bits: p1_choice, p2_choice
    
    best_total_score = -1e20
    
    for parent_assignment in [0, 1]:
        scores = np.full(n_states, -1e20, dtype=np.float64)
        new_scores = np.zeros(n_states, dtype=np.float64)
        
        # Initialize at bin 0
        for state in range(n_states):
            p1_choice = (state >> 0) & 1
            p2_choice = (state >> 1) & 1
            
            corr_h1 = sample_grid[0, 0]
            corr_h2 = sample_grid[0, 1]
            
            if parent_assignment == 0:
                expect_h1 = p1_grid[0, 1] if p1_choice else p1_grid[0, 0]
                expect_h2 = p2_grid[0, 1] if p2_choice else p2_grid[0, 0]
            else:
                expect_h1 = p2_grid[0, 1] if p2_choice else p2_grid[0, 0]
                expect_h2 = p1_grid[0, 1] if p1_choice else p1_grid[0, 0]
            
            cost = 0.0
            # Check equivalence instead of exact match
            if corr_h1 != -1 and expect_h1 != -1:
                if not check_equiv(equiv, 0, corr_h1, expect_h1):
                    cost -= mismatch_cost
            if corr_h2 != -1 and expect_h2 != -1:
                if not check_equiv(equiv, 0, corr_h2, expect_h2):
                    cost -= mismatch_cost
            
            scores[state] = cost
        
        # Forward pass
        for k in range(1, n_bins):
            dist_bp = bin_widths[k]
            theta = dist_bp * recomb_rate
            if theta > 0.5:
                theta = 0.5
            if theta < 1e-15:
                theta = 1e-15
            
            log_switch = math.log(theta)
            log_stay = math.log(1.0 - theta)
            
            for curr_state in range(n_states):
                curr_p1 = (curr_state >> 0) & 1
                curr_p2 = (curr_state >> 1) & 1
                
                best_score = -1e20
                
                for prev_state in range(n_states):
                    prev_p1 = (prev_state >> 0) & 1
                    prev_p2 = (prev_state >> 1) & 1
                    
                    trans_cost = 0.0
                    if prev_p1 != curr_p1:
                        trans_cost += log_switch
                    else:
                        trans_cost += log_stay
                    if prev_p2 != curr_p2:
                        trans_cost += log_switch
                    else:
                        trans_cost += log_stay
                    
                    score = scores[prev_state] + trans_cost
                    if score > best_score:
                        best_score = score
                
                # Emission
                corr_h1 = sample_grid[k, 0]
                corr_h2 = sample_grid[k, 1]
                
                if parent_assignment == 0:
                    expect_h1 = p1_grid[k, 1] if curr_p1 else p1_grid[k, 0]
                    expect_h2 = p2_grid[k, 1] if curr_p2 else p2_grid[k, 0]
                else:
                    expect_h1 = p2_grid[k, 1] if curr_p2 else p2_grid[k, 0]
                    expect_h2 = p1_grid[k, 1] if curr_p1 else p1_grid[k, 0]
                
                emit_cost = 0.0
                # Check equivalence instead of exact match
                if corr_h1 != -1 and expect_h1 != -1:
                    if not check_equiv(equiv, k, corr_h1, expect_h1):
                        emit_cost -= mismatch_cost
                if corr_h2 != -1 and expect_h2 != -1:
                    if not check_equiv(equiv, k, corr_h2, expect_h2):
                        emit_cost -= mismatch_cost
                
                new_scores[curr_state] = best_score + emit_cost
            
            for s in range(n_states):
                scores[s] = new_scores[s]
        
        # Best final score for this assignment
        best_final = scores[0]
        for s in range(1, n_states):
            if scores[s] > best_final:
                best_final = scores[s]
        
        if best_final > best_total_score:
            best_total_score = best_final
    
    return best_total_score


@njit(nogil=True, cache=True)
def _compute_total_score_for_grid_kernel(
    sample_grid: np.ndarray,
    p1_grid: np.ndarray,
    p2_grid: np.ndarray,
    have_parents: bool,
    children_stacked: np.ndarray,
    child_hom_stacked: np.ndarray,
    other_parent_stacked: np.ndarray,
    bin_widths: np.ndarray,
    equiv: np.ndarray,
    recomb_rate: float,
    mismatch_cost: float,
) -> float:
    """
    Numba kernel for compute_total_score_for_grid.

    Takes pre-stacked 3D arrays for children data so that the loop
    over children stays inside njit (no Python dispatch overhead per
    child).  The wrapper `compute_total_score_for_grid` below stacks
    Python list inputs and calls this kernel; the inner loop of
    `greedy_phase_refinement_single_sample` bypasses the wrapper and
    calls this kernel directly with arrays it pre-stacks ONCE before
    the iteration loop (since the children data does not change
    between candidates -- only `sample_grid` does).  This saves the
    O(n_children * n_bins) re-stacking cost on every score evaluation.

    `have_parents` is a separate boolean rather than relying on
    Optional[ndarray] because numba's nopython mode does not support
    Optional types in argument signatures.  When False, p1_grid and
    p2_grid are zero-filled placeholders that the kernel never reads.

    Math is identical to the original compute_total_score_for_grid:
    total = parent_matching_score (if parents) + sum(child transmission Viterbi scores).
    """
    n_bins = sample_grid.shape[0]
    total_score = 0.0

    # Parent matching score
    if have_parents:
        parent_score = compute_parent_matching_score_fixed_phase(
            n_bins, sample_grid, p1_grid, p2_grid, bin_widths, equiv,
            recomb_rate, mismatch_cost
        )
        total_score += parent_score

    # Child transmission scores
    n_children = children_stacked.shape[0]
    for ci in range(n_children):
        # Slicing axis-0 of a contiguous 3D array yields a contiguous
        # 2D view -- safe for numba and zero-copy.
        _, child_score = run_8state_transmission_viterbi_binned(
            n_bins, sample_grid, other_parent_stacked[ci],
            children_stacked[ci], child_hom_stacked[ci],
            bin_widths, equiv, recomb_rate, mismatch_cost
        )
        total_score += child_score

    return total_score


def compute_total_score_for_grid(
    sample_grid: np.ndarray,
    p1_grid: Optional[np.ndarray],
    p2_grid: Optional[np.ndarray],
    children_grids: List[np.ndarray],
    children_hom_masks: List[np.ndarray],
    other_parent_grids: List[np.ndarray],
    bin_widths: np.ndarray,
    equiv: np.ndarray,
    recomb_rate: float = 5e-8,
    mismatch_cost: float = 4.6
) -> float:
    """
    Compute total score for a fixed corrected sample grid.
    
    Score = parent_matching_score + sum(child_transmission_scores)
    
    Uses full Viterbi to properly account for recombination costs.
    Uses founder equivalence matrix to handle aliasing.

    Implementation note (May 2026):
        This function is now a Python wrapper around the njit kernel
        `_compute_total_score_for_grid_kernel`.  It stacks the
        Python list inputs into contiguous 3D arrays and dispatches
        to the kernel.  The mathematical result is unchanged: it
        sums the parent-matching Viterbi score and each child's
        transmission Viterbi score, exactly as the previous Python
        implementation did.

        The hot caller `greedy_phase_refinement_single_sample`
        bypasses this wrapper to avoid re-stacking on every score
        evaluation -- it stacks ONCE before the iteration loop and
        calls the kernel directly.
    """
    n_bins = sample_grid.shape[0]
    have_parents = (p1_grid is not None and p2_grid is not None)

    # Placeholder parent arrays when parents are absent.  Their
    # contents are never read by the kernel when have_parents=False;
    # they exist only so the kernel signature accepts concrete arrays.
    if not have_parents:
        p1_grid_arg = np.zeros((n_bins, 2), dtype=np.int32)
        p2_grid_arg = np.zeros((n_bins, 2), dtype=np.int32)
    else:
        # Ensure contiguity for numba (np.ascontiguousarray is a no-op
        # if already contiguous, which is the common case).
        p1_grid_arg = np.ascontiguousarray(p1_grid)
        p2_grid_arg = np.ascontiguousarray(p2_grid)

    # Stack children data into contiguous 3D arrays.  Same dtypes as
    # the originals (int32 grids, bool hom masks) so no copy is forced
    # at downstream call sites.
    n_children = len(children_grids)
    if n_children > 0:
        children_stacked = np.stack(
            [np.ascontiguousarray(g) for g in children_grids], axis=0
        )
        child_hom_stacked = np.stack(
            [np.ascontiguousarray(h) for h in children_hom_masks], axis=0
        )
        other_parent_stacked = np.stack(
            [np.ascontiguousarray(g) for g in other_parent_grids], axis=0
        )
    else:
        # Empty placeholders of the right shape so the kernel's
        # children loop short-circuits naturally.
        children_stacked = np.zeros((0, n_bins, 2), dtype=np.int32)
        child_hom_stacked = np.zeros((0, n_bins), dtype=np.bool_)
        other_parent_stacked = np.zeros((0, n_bins, 2), dtype=np.int32)

    return _compute_total_score_for_grid_kernel(
        sample_grid, p1_grid_arg, p2_grid_arg, have_parents,
        children_stacked, child_hom_stacked, other_parent_stacked,
        bin_widths, equiv, recomb_rate, mismatch_cost
    )


def greedy_phase_refinement_single_sample(
    sample_grid: np.ndarray,
    hom_mask: np.ndarray,
    p1_grid: Optional[np.ndarray],
    p2_grid: Optional[np.ndarray],
    children_grids: List[np.ndarray],
    children_hom_masks: List[np.ndarray],
    other_parent_grids: List[np.ndarray],
    bin_widths: np.ndarray,
    equiv: np.ndarray,
    recomb_rate: float = 5e-8,
    mismatch_cost: float = 4.6,
    max_iterations: int = 100,
    verbose: bool = False
) -> Tuple[np.ndarray, int]:
    """
    Greedy phase refinement for a single sample.
    
    Iteratively tries all singleton and pair flips at valid boundaries
    (HOM→HET transitions AND double recombinations), selecting the best
    improvement until no improvement is found.
    
    Uses founder equivalence matrix to handle aliasing.
    
    Args:
        sample_grid: Current corrected sample grid (n_bins, 2)
        hom_mask: Homozygosity mask (n_bins,)
        p1_grid, p2_grid: Parent grids (may be None)
        children_grids: List of child grids
        children_hom_masks: List of child HOM masks
        other_parent_grids: List of other parent grids for each child
        bin_widths: Physical width of each bin
        equiv: Founder equivalence matrix
        recomb_rate: Per-bp recombination rate
        mismatch_cost: Mismatch penalty
        max_iterations: Maximum iterations to prevent infinite loops
        verbose: Print debug info
    
    Returns:
        refined_grid: The refined sample grid
        n_flips: Number of flip operations performed

    Implementation note (May 2026):
        The inner iteration loop evaluates the total score for
        ~n_boundaries singleton candidates + ~n_boundaries^2 / 2 pair
        candidates per iteration -- often 200+ score evaluations per
        iteration.  Each evaluation previously paid the Python dispatch
        cost of `compute_total_score_for_grid` PLUS the per-call
        stacking of `children_grids` / `children_hom_masks` /
        `other_parent_grids` into contiguous arrays.

        This refactor stacks the children data ONCE before the
        iteration loop (since those inputs are read-only throughout
        refinement -- only the sample grid changes between candidates)
        and calls the njit kernel `_compute_total_score_for_grid_kernel`
        directly via a local `_score(grid)` closure.  Math is
        unchanged: parent-matching Viterbi score + sum of child
        transmission Viterbi scores, exactly as before.
    """
    current_grid = sample_grid.copy()
    n_bins = len(hom_mask)
    
    # Find all valid flip boundaries (HOM→HET + double recombs)
    boundaries = find_all_valid_flip_boundaries(current_grid, hom_mask)
    n_boundaries = len(boundaries)
    
    if verbose:
        hom_het = find_hom_to_het_boundaries(hom_mask)
        double_recomb = find_double_recomb_boundaries(current_grid)
        print(f"    Found {len(hom_het)} HOM→HET + {len(double_recomb)} double-recomb = {n_boundaries} total boundaries")
    
    if n_boundaries == 0:
        # No boundaries to flip at
        return current_grid, 0

    # ---------------------------------------------------------------
    # Pre-stack children data ONCE.  These arrays are read-only
    # throughout the iteration loop (only current_grid changes
    # between candidates), so we amortise the stacking cost across
    # the ~200+ score evaluations per iteration.  Empty placeholders
    # are used when there are no children so the kernel's loop
    # short-circuits naturally.
    # ---------------------------------------------------------------
    have_parents = (p1_grid is not None and p2_grid is not None)
    if have_parents:
        # Ensure contiguity for numba (np.ascontiguousarray is a no-op
        # if already contiguous, which is the common case).
        p1_grid_arg = np.ascontiguousarray(p1_grid)
        p2_grid_arg = np.ascontiguousarray(p2_grid)
    else:
        # Placeholder arrays for the njit signature; values unused
        # when have_parents=False.
        p1_grid_arg = np.zeros((n_bins, 2), dtype=np.int32)
        p2_grid_arg = np.zeros((n_bins, 2), dtype=np.int32)

    n_children = len(children_grids)
    if n_children > 0:
        children_stacked = np.stack(
            [np.ascontiguousarray(g) for g in children_grids], axis=0
        )
        child_hom_stacked = np.stack(
            [np.ascontiguousarray(h) for h in children_hom_masks], axis=0
        )
        other_parent_stacked = np.stack(
            [np.ascontiguousarray(g) for g in other_parent_grids], axis=0
        )
    else:
        children_stacked = np.zeros((0, n_bins, 2), dtype=np.int32)
        child_hom_stacked = np.zeros((0, n_bins), dtype=np.bool_)
        other_parent_stacked = np.zeros((0, n_bins, 2), dtype=np.int32)

    # Local scoring closure: wraps the njit kernel with the
    # pre-stacked, read-only context.  Replaces every previous call
    # to `compute_total_score_for_grid(...)` -- mathematically
    # identical, just bypasses the per-call stacking work.
    def _score(grid):
        return _compute_total_score_for_grid_kernel(
            grid, p1_grid_arg, p2_grid_arg, have_parents,
            children_stacked, child_hom_stacked, other_parent_stacked,
            bin_widths, equiv, recomb_rate, mismatch_cost
        )
    
    # Compute initial score
    current_score = _score(current_grid)
    
    if verbose:
        print(f"    Initial score: {current_score:.2f}")
    
    total_flips = 0
    
    for iteration in range(max_iterations):
        best_improvement = 0.0
        best_candidate = None
        best_candidate_grid = None
        
        # Recompute boundaries after each flip (grid may have changed)
        boundaries = find_all_valid_flip_boundaries(current_grid, hom_mask)
        
        if len(boundaries) == 0:
            break
        
        # Try all singleton flips (flip from boundary to end)
        for i, b1 in enumerate(boundaries):
            candidate_grid = apply_phase_flip_to_grid(current_grid, b1, None)
            
            # Check Mendelian consistency if we have parents
            if p1_grid is not None and p2_grid is not None:
                if not check_mendelian_consistency(candidate_grid, p1_grid, p2_grid, equiv):
                    continue
            
            candidate_score = _score(candidate_grid)
            
            improvement = candidate_score - current_score
            if improvement > best_improvement:
                best_improvement = improvement
                best_candidate = ('singleton', b1)
                best_candidate_grid = candidate_grid
        
        # Try all pair flips (flip between b1 and b2)
        for i, b1 in enumerate(boundaries):
            for j, b2 in enumerate(boundaries):
                if j <= i:
                    continue  # Only consider pairs where b2 > b1
                
                candidate_grid = apply_phase_flip_to_grid(current_grid, b1, b2)
                
                # Check Mendelian consistency if we have parents
                if p1_grid is not None and p2_grid is not None:
                    if not check_mendelian_consistency(candidate_grid, p1_grid, p2_grid, equiv):
                        continue
                
                candidate_score = _score(candidate_grid)
                
                improvement = candidate_score - current_score
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_candidate = ('pair', b1, b2)
                    best_candidate_grid = candidate_grid
        
        # Apply best improvement if found
        if best_improvement > 0.001:  # Small threshold to avoid floating point issues
            current_grid = best_candidate_grid
            current_score = current_score + best_improvement
            total_flips += 1
            
            if verbose:
                if best_candidate[0] == 'singleton':
                    print(f"    Iteration {iteration+1}: Flip at bin {best_candidate[1]}, "
                          f"improvement: +{best_improvement:.2f}, new score: {current_score:.2f}")
                else:
                    print(f"    Iteration {iteration+1}: Flip between bins {best_candidate[1]}-{best_candidate[2]}, "
                          f"improvement: +{best_improvement:.2f}, new score: {current_score:.2f}")
        else:
            # No improvement found, we're done
            if verbose:
                print(f"    Converged after {iteration} iterations with {total_flips} flips")
            break
    
    return current_grid, total_flips


def post_process_phase_greedy(
    corrected_painting: BlockPainting,
    pedigree_df: pd.DataFrame,
    sample_names: List[str],
    bin_edges: np.ndarray,
    equiv: np.ndarray,
    recomb_rate: float = 5e-8,
    mismatch_cost: float = 4.6,
    max_global_iterations: int = 10,
    n_threads: int = 1,        # Number of ThreadPoolExecutor workers for
                                # per-sample parallelism within a pass.  See
                                # the JACOBI-vs-GAUSS-SEIDEL discussion in
                                # the docstring.  Default 1 (sequential).
    verbose: bool = True
) -> BlockPainting:
    """
    Post-process corrected paintings using greedy phase refinement.
    
    For each sample, finds valid flip boundaries (HOM→HET transitions AND
    double recombinations) and tries all singleton/pair flips to minimize
    total recombinations (parent matching + child transmission).
    
    Runs multiple global passes until no sample is refined, since refining one
    sample can create improvement opportunities for related samples.
    
    Args:
        corrected_painting: BlockPainting from initial phase correction
        pedigree_df: Pedigree information
        sample_names: List of sample names
        bin_edges: Bin edges for discretization
        equiv: Founder equivalence matrix (n_bins, n_founders, n_founders)
        recomb_rate: Per-bp recombination rate
        mismatch_cost: Mismatch penalty
        max_global_iterations: Maximum global passes
        n_threads: Per-pass ThreadPoolExecutor parallelism (default 1).
        verbose: Print progress
    
    Returns:
        Refined BlockPainting

    JACOBI vs GAUSS-SEIDEL (May 2026 hybrid-parallelism update):
        The original loop processed samples sequentially within each
        global pass and updated `sample_grids[name] = refined_grid`
        IMMEDIATELY, so a sample processed later in the same pass
        would read its parents' / children's just-updated grids
        (Gauss-Seidel iteration).  When n_threads > 1, all samples
        in a pass must read a consistent view -- the snapshot taken
        at the start of each pass (Jacobi iteration).  Implementation
        mirrors `run_correction_round`:
          1) Snapshot `sample_grids` at the start of every pass.
          2) Per-sample worker reads parents/children from the
             snapshot, computes (name, refined_grid, n_flips) without
             mutating shared state.
          3) Sequential merge-back applies updates in sample_names
             order so `pass_refined` and `pass_flips` are
             deterministic.
        Both schemes converge to the same fixed point; Jacobi may
        need one extra global pass in pathological cases.  The pass
        loop already runs to convergence (up to
        max_global_iterations), so this is benign.

        When n_threads == 1, the snapshot is still taken (so this
        path is Jacobi-but-sequential, NOT a byte-for-byte recovery
        of the historical Gauss-Seidel behaviour).  The converged
        answer is the same.
    """
    n_samples = len(sample_names)
    n_bins = len(bin_edges) - 1
    bin_widths = np.diff(bin_edges)
    
    # Build parent/child lookup
    parent_map = {}
    children_map = {name: [] for name in sample_names}
    other_parent_map = {}
    
    for _, row in pedigree_df.iterrows():
        sample = row['Sample']
        p1 = row.get('Parent1')
        p2 = row.get('Parent2')
        
        if pd.notna(p1) and pd.notna(p2):
            parent_map[sample] = (p1, p2)
            if p1 in children_map:
                children_map[p1].append(sample)
            if p2 in children_map:
                children_map[p2].append(sample)
            other_parent_map[(p1, sample)] = p2
            other_parent_map[(p2, sample)] = p1
    
    # Discretize all paintings
    sample_grids = {}
    hom_masks = {}
    
    for i, name in enumerate(sample_names):
        painting = corrected_painting.samples[i]
        grid, hom = discretize_painting_to_bins(painting, bin_edges)
        sample_grids[name] = grid
        hom_masks[name] = hom
    
    # Run multiple global passes until convergence
    total_samples_refined = 0
    total_flips = 0
    
    for global_iter in range(max_global_iterations):
        pass_refined = 0
        pass_flips = 0

        # =================================================================
        # Snapshot sample_grids at the START of this pass.  All
        # per-sample workers in this pass read parents/children from
        # the snapshot so concurrent threads see a consistent view.
        # See JACOBI vs GAUSS-SEIDEL note in the docstring.
        # =================================================================
        sample_grids_snapshot = dict(sample_grids)

        def _process_one_sample(name):
            """
            Per-sample worker -- safe to call concurrently.  Reads
            parent/child grids from `sample_grids_snapshot`; returns
            (name, refined_grid_or_None, n_flips) for the sequential
            merge-back loop to apply.
            """
            sample_grid = sample_grids_snapshot[name]
            hom_mask = hom_masks[name]
            
            # Get parent grids (from the round-start snapshot for Jacobi)
            p1_grid = None
            p2_grid = None
            if name in parent_map:
                p1_name, p2_name = parent_map[name]
                if p1_name in sample_grids_snapshot:
                    p1_grid = sample_grids_snapshot[p1_name]
                if p2_name in sample_grids_snapshot:
                    p2_grid = sample_grids_snapshot[p2_name]
            
            # Get children grids (from the snapshot)
            children_grids = []
            children_hom_masks = []
            other_parent_grids = []
            
            for child_name in children_map.get(name, []):
                if child_name not in sample_grids_snapshot:
                    continue
                other_name = other_parent_map.get((name, child_name))
                if other_name is None or other_name not in sample_grids_snapshot:
                    continue
                
                children_grids.append(sample_grids_snapshot[child_name])
                children_hom_masks.append(hom_masks[child_name])
                other_parent_grids.append(sample_grids_snapshot[other_name])
            
            # Skip if no constraints
            if p1_grid is None and p2_grid is None and not children_grids:
                return name, None, 0
            
            # Run greedy refinement
            refined_grid, n_flips = greedy_phase_refinement_single_sample(
                sample_grid, hom_mask, p1_grid, p2_grid,
                children_grids, children_hom_masks, other_parent_grids,
                bin_widths, equiv, recomb_rate, mismatch_cost,
                verbose=False
            )
            return name, refined_grid, n_flips

        # ---- Dispatch ----
        effective_threads = max(1, min(n_threads, len(sample_names)))
        if effective_threads == 1:
            results = [_process_one_sample(name) for name in sample_names]
        else:
            with ThreadPoolExecutor(max_workers=effective_threads) as executor:
                results = list(executor.map(_process_one_sample, sample_names))

        # ---- Sequential merge-back in sample_names order ----
        # Each result is (name, refined_grid_or_None, n_flips).  Updates
        # to `sample_grids` and the pass counters happen in deterministic
        # sample_names order to match the original code's accounting and
        # verbose output ordering.
        for name, refined_grid, n_flips in results:
            if refined_grid is None or n_flips <= 0:
                continue
            # Update the grid for use by other samples (in subsequent passes)
            sample_grids[name] = refined_grid
            pass_refined += 1
            pass_flips += n_flips
            if verbose:
                print(f"  Pass {global_iter+1}: {name}: {n_flips} flip(s)")
        
        total_samples_refined += pass_refined
        total_flips += pass_flips
        
        if pass_refined == 0:
            if verbose and global_iter > 0:
                print(f"  Converged after {global_iter+1} passes")
            break
    
    if verbose:
        print(f"  Total: {total_samples_refined} refinements, {total_flips} flips applied")
    
    # Build new painting from refined grids
    # Get region from the painting - try different access patterns
    if hasattr(corrected_painting, 'region'):
        start_pos = corrected_painting.region[0]
        end_pos = corrected_painting.region[1]
    elif hasattr(corrected_painting, 'samples') and corrected_painting.samples:
        # Compute from sample chunks
        start_pos = min(s.chunks[0].start for s in corrected_painting.samples if s.chunks)
        end_pos = max(s.chunks[-1].end for s in corrected_painting.samples if s.chunks)
    else:
        # Fallback: use bin_edges
        start_pos = int(bin_edges[0])
        end_pos = int(bin_edges[-1])
    
    refined_samples = []
    for i, name in enumerate(sample_names):
        refined_grid = sample_grids[name]
        
        # Convert grid back to painting
        chunks = []
        chunk_start = int(bin_edges[0])
        chunk_h1 = int(refined_grid[0, 0])
        chunk_h2 = int(refined_grid[0, 1])
        
        for k in range(1, n_bins):
            if refined_grid[k, 0] != chunk_h1 or refined_grid[k, 1] != chunk_h2:
                chunk_end = int(bin_edges[k])
                chunks.append(PaintedChunk(chunk_start, chunk_end, chunk_h1, chunk_h2))
                chunk_start = chunk_end
                chunk_h1 = int(refined_grid[k, 0])
                chunk_h2 = int(refined_grid[k, 1])
        
        # Final chunk
        chunks.append(PaintedChunk(chunk_start, end_pos, chunk_h1, chunk_h2))
        
        refined_samples.append(SamplePainting(i, chunks))
    
    return BlockPainting((start_pos, end_pos), refined_samples)


def _greedy_contig_worker(task):
    """
    Worker function for parallel greedy refinement of a single contig.

    Receives a per-task tuple:
        task = (r_name, corrected_painting, founder_block_or_None)

    DESIGN PRINCIPLE (May 2026 follow-up):  per-contig data flows via
    TASK ARGS (the argument to `pool.map`), not via INITARGS.  The two
    channels have different semantics:

      - INITARGS is pickled ONCE per worker at Pool creation and held
        for the lifetime of the pool.  Putting per-contig data here
        means the pickle payload scales with N_contigs and is
        replicated across all worker processes -- which on a warm
        pipeline (multi_contig_results holding tens of GB of stage-9
        founder_blocks + stage-10 paintings) causes the main process
        to spend minutes serialising tens of GB of pickle bytes
        before any worker can start.

      - TASK ARGS are pickled per call to `pool.map`'s consumer, one
        task at a time as workers free up.  Peak in-flight memory is
        bounded by num_workers * one_task's_size, not by
        N_contigs * payload.  This is what `pool.map` is for.

    So this worker reads:
      * Stable static config (pedigree_df, sample_names, hyperparameters,
        load_fn) from `_PARALLEL_DATA`, populated by `_init_phase_worker`
        from the small config dict passed via `initargs`.
      * Per-contig data (corrected_painting, optionally founder_block)
        from the task arg.

    founder_block_or_None handling:
      * If not None, used directly (in-memory test path, no load_fn).
      * If None, the worker calls load_fn(r_name) to fetch from disk
        -- this is the pipeline.py path, where load_fn is always
        supplied and founder_blocks live on the rds checkpoint store
        rather than in main's RAM (so we don't pickle them through the
        process boundary at all).
    """
    global _PARALLEL_DATA, _PHASE_ACTIVE_COUNTER

    r_name, corrected_painting, founder_block_from_task = task

    pedigree_df = _PARALLEL_DATA['pedigree_df']
    sample_names = _PARALLEL_DATA['sample_names']
    snps_per_bin = _PARALLEL_DATA['snps_per_bin']
    recomb_rate = _PARALLEL_DATA['recomb_rate']
    mismatch_cost = _PARALLEL_DATA['mismatch_cost']
    max_diff_fraction = _PARALLEL_DATA.get('max_diff_fraction', 0.02)
    min_diff_sites = _PARALLEL_DATA.get('min_diff_sites', 2)

    # Build the local `data` dict from task args.  founder_block is
    # filled either from the task arg (in-memory case) or load_fn (disk
    # case).  The dict shape matches what the rest of this function
    # expects so the downstream code is untouched.
    data = {'corrected_painting': corrected_painting}
    if founder_block_from_task is not None:
        data['founder_block'] = founder_block_from_task

    load_fn = _PARALLEL_DATA.get('load_fn')
    if 'founder_block' not in data and load_fn is not None:
        loaded = load_fn(r_name)
        if loaded is not None and 'founder_block' in loaded:
            data['founder_block'] = loaded['founder_block']

    corrected_painting = data['corrected_painting']
    
    # Get region
    if hasattr(corrected_painting, 'region'):
        start_pos = corrected_painting.region[0]
        end_pos = corrected_painting.region[1]
    elif hasattr(corrected_painting, 'samples') and corrected_painting.samples:
        start_pos = min(s.chunks[0].start for s in corrected_painting.samples if s.chunks)
        end_pos = max(s.chunks[-1].end for s in corrected_painting.samples if s.chunks)
    else:
        start_pos = corrected_painting.start_pos
        end_pos = corrected_painting.end_pos

    # =====================================================================
    # Dynamic thread reallocation -- same pattern as _process_contig_worker.
    # Register as active for the duration of this worker; surviving workers
    # claim freed cores when this one exits.  See "DYNAMIC THREAD
    # REALLOCATION" header comment above for the full design.
    # =====================================================================
    if _PHASE_ACTIVE_COUNTER is not None:
        with _PHASE_ACTIVE_COUNTER.get_lock():
            _PHASE_ACTIVE_COUNTER.value += 1
    try:
        try:
            import numba
        except ImportError:
            numba = None

        bin_edges = compute_bin_edges(start_pos, end_pos, snps_per_bin=snps_per_bin)

        # ----------------------------------------------------------------
        # Phase 1: Founder equivalence matrix (numba-parallel, prange).
        # Give it the full dynamic budget.
        # ----------------------------------------------------------------
        if numba is not None:
            dyn_threads = _phase_get_dynamic_threads()
            numba.set_num_threads(dyn_threads)

        # Compute founder equivalence matrix
        if 'founder_block' in data:
            dense_haps, positions = founder_block_to_dense(data['founder_block'])
            equiv = compute_founder_equivalence_matrix(
                dense_haps, positions, bin_edges,
                max_diff_fraction=max_diff_fraction,
                min_diff_sites=min_diff_sites
            )
            n_founders = dense_haps.shape[0]
        else:
            n_bins = len(bin_edges) - 1
            equiv = np.zeros((n_bins, 8, 8), dtype=np.bool_)
            for b in range(n_bins):
                for f in range(8):
                    equiv[b, f, f] = True
            n_founders = 0

        # ----------------------------------------------------------------
        # Phase 2: Greedy refinement passes.  Python-dispatch phase: each
        # pass uses a ThreadPoolExecutor over samples, with the
        # underlying njit kernels (compute_parent_matching_score_-
        # fixed_phase, run_8state_transmission_viterbi_binned) each
        # running single-threaded.  Numba threads = 1 to avoid
        # over-subscription with the python thread pool.
        # ----------------------------------------------------------------
        if numba is not None:
            numba.set_num_threads(1)
        dyn_threads = _phase_get_dynamic_threads()

        refined_painting = post_process_phase_greedy(
            corrected_painting,
            pedigree_df,
            sample_names,
            bin_edges,
            equiv,
            recomb_rate=recomb_rate,
            mismatch_cost=mismatch_cost,
            n_threads=dyn_threads,
            verbose=False  # Workers don't print (avoids interleaved output)
        )

        # Collect stats for summary
        n_bins = len(bin_edges) - 1
        sample_grids_before = {}
        sample_grids_after = {}
        for i, name in enumerate(sample_names):
            gb, _ = discretize_painting_to_bins(corrected_painting.samples[i], bin_edges)
            ga, _ = discretize_painting_to_bins(refined_painting.samples[i], bin_edges)
            if not np.array_equal(gb, ga):
                sample_grids_before[name] = gb
                sample_grids_after[name] = ga
        
        return (r_name, refined_painting, n_founders, len(sample_grids_before))
    finally:
        if _PHASE_ACTIVE_COUNTER is not None:
            with _PHASE_ACTIVE_COUNTER.get_lock():
                _PHASE_ACTIVE_COUNTER.value -= 1


def post_process_phase_greedy_all_contigs(
    multi_contig_results: Dict,
    pedigree_df: pd.DataFrame,
    sample_names: List[str],
    snps_per_bin: int = 100,
    recomb_rate: float = 5e-8,
    mismatch_cost: float = 4.6,
    max_diff_fraction: float = 0.02,
    min_diff_sites: int = 2,
    verbose: bool = True,
    max_workers: int = None,
    parallel: bool = True,
    load_fn: Optional[Callable[[str], Dict]] = None,
) -> Dict:
    """
    Apply greedy phase refinement to all contigs.
    
    This is a post-processing step that should be run after correct_phase_all_contigs.
    It refines phase by trying singleton and pair flips at valid boundaries
    (HOM→HET transitions AND double recombinations).
    
    Uses founder equivalence to handle aliasing - two founders that are nearly
    identical in a bin are treated as equivalent for emission scoring.
    
    Args:
        multi_contig_results: Dict with 'corrected_painting' and 'founder_block' from phase correction
        pedigree_df: Pedigree information
        sample_names: List of sample names
        snps_per_bin: Bin resolution (should match what was used for correction)
        recomb_rate: Per-bp recombination rate
        mismatch_cost: Mismatch penalty
        max_diff_fraction: Max fraction of differing sites for founder equivalence (default 2%)
        min_diff_sites: Min absolute number of differing sites for equivalence (default 2)
        verbose: Print progress
        max_workers: REQUIRED.  Maximum parallel workers.  Must be passed
            from the calling pipeline (e.g. pipeline.py's `n_processes`)
            so the greedy refinement respects the pipeline's machine-wide
            worker budget.  Passing None will raise.
        parallel: Use parallel processing
        load_fn: Optional callable(r_name) -> dict with 'founder_block'.
            When provided, each worker loads its own founder_block from
            checkpoint via this function rather than expecting it in
            multi_contig_results.  Mirrors the load_fn pattern used by
            `correct_phase_all_contigs`.  Added May 2026 alongside the
            IPC-cost fix: phase correction stopped returning
            founder_block to save ~40s of pickle/pipe transfer across
            22 contigs, so greedy now loads founder_block itself
            (parallel I/O across worker processes).  If None, falls
            back to data.get('founder_block') (backward compatible).
    
    Returns:
        Updated multi_contig_results with 'refined_painting' key added
    """
    # Function-local `import multiprocessing as mp` removed (May 2026);
    # see the FORKSERVER CONTEXT comment at the top of this module.
    # The forkserver context and the module-level `mp` import live
    # there.

    if max_workers is None:
        raise ValueError(
            "post_process_phase_greedy_all_contigs: max_workers must be "
            "specified (typically pipeline.py passes its n_processes "
            "here).  The previous os.cpu_count() fallback has been "
            "removed so that all greedy-refinement workers respect the "
            "pipeline's machine-wide worker budget.")

    contig_names = [
        r_name for r_name, data in multi_contig_results.items()
        if 'corrected_painting' in data
    ]
    n_contigs = len(contig_names)
    # `max_workers` is the TOTAL CORE BUDGET passed by the pipeline.
    # The OUTER POOL SIZE is the smaller of that budget and the number
    # of contigs; the total budget is forwarded to workers as
    # `total_cores` for dynamic thread reallocation.  See the
    # equivalent block in `correct_phase_all_contigs` and the
    # "DYNAMIC THREAD REALLOCATION" header comment above.
    total_cores = max_workers
    outer_pool_size = max(1, min(total_cores, n_contigs)) if n_contigs > 0 else total_cores
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Greedy Phase Refinement Post-Processing")
        if parallel and n_contigs > 1:
            print(f"Using {outer_pool_size} outer workers x dynamic threads "
                  f"(total budget = {total_cores} cores)")
        print(f"{'='*60}")
    
    if parallel and n_contigs > 1:
        # PRINCIPLE: initargs holds stable static config that every
        # worker shares; per-contig data flows via TASK ARGS
        # (pool.map's tasks list).
        #
        # The two pickle channels have very different semantics:
        #
        #   initargs is pickled ONCE per worker at Pool creation and
        #     held for the lifetime of the pool.  Putting per-contig
        #     data here means total pickle volume scales with
        #     N_workers * (full per-contig payload).  Pickling
        #     multi_contig_results -- which on a warm pipeline holds
        #     stage-9 founder_blocks (~30-50 GB) and stage-10
        #     tolerance_result (~5+ GB on chr3, more with wildcard
        #     enabled) -- into 22 worker initargs is a guaranteed
        #     OOM.  This was the May 2026 phase-correction hang.
        #
        #   task args are pickled per dispatch to a worker, ONE TASK
        #     AT A TIME as workers free up.  Peak in-flight pickle
        #     memory is bounded by N_workers * one task's payload,
        #     regardless of N_contigs.  This is the channel
        #     `pool.map` was designed for.
        #
        # So we send only the stable config via initargs, and dispatch
        # the per-contig corrected_painting (the only in-memory data
        # the greedy worker actually needs from main) via the task
        # tuple.  founder_block is fetched from disk via load_fn
        # inside the worker -- no need to pickle it through the
        # process boundary at all.
        parallel_data = {
            'pedigree_df': pedigree_df,
            'sample_names': sample_names,
            'snps_per_bin': snps_per_bin,
            'recomb_rate': recomb_rate,
            'mismatch_cost': mismatch_cost,
            'max_diff_fraction': max_diff_fraction,
            'min_diff_sites': min_diff_sites,
            # Worker-side founder_block loader -- see the May 2026
            # IPC-cost fix in `_greedy_contig_worker`'s docstring.
            'load_fn': load_fn,
        }

        # Per-contig task data.  Each task is a tuple
        # `(r_name, corrected_painting, founder_block_or_None)`.
        # founder_block is passed through ONLY when load_fn is None
        # (in-memory test path with no checkpoints) -- in the pipeline
        # flow load_fn is always set and the worker fetches
        # founder_block from disk, so we set the slot to None and
        # avoid pickling that ~1-2 GB structure into the task stream.
        if load_fn is not None:
            tasks = [
                (r_name,
                 multi_contig_results[r_name]['corrected_painting'],
                 None)
                for r_name in contig_names
            ]
        else:
            tasks = [
                (r_name,
                 multi_contig_results[r_name]['corrected_painting'],
                 multi_contig_results[r_name].get('founder_block'))
                for r_name in contig_names
            ]

        if verbose:
            print(f"\nProcessing {n_contigs} contigs in parallel...")

        # Shared active-worker counter for dynamic thread reallocation;
        # see `_init_phase_worker` and the `_PHASE_ACTIVE_COUNTER`
        # machinery for the design.  Created from `_forkserver_ctx`
        # for the same reason as in correct_phase_all_contigs.
        active_counter = _forkserver_ctx.Value('i', 0)

        with _forkserver_ctx.Pool(
            processes=outer_pool_size,
            initializer=_init_phase_worker,
            initargs=(total_cores, active_counter, parallel_data),
        ) as pool:
            results = pool.map(_greedy_contig_worker, tasks)
        
        for r_name, refined_painting, n_founders, n_refined in results:
            multi_contig_results[r_name]['refined_painting'] = refined_painting
            if verbose:
                print(f"  {r_name}: {n_founders} founders, {n_refined} samples refined")
        
        # Main process's _PARALLEL_DATA was never touched in this
        # function under the forkserver design (the data lives in the
        # workers' globals only).  Nothing to clear here.
    
    else:
        for r_name in contig_names:
            data = multi_contig_results[r_name]
            # Same load_fn fallback as in `_greedy_contig_worker` -- if
            # founder_block isn't in multi_contig_results (because
            # phase correction no longer returns it; see the IPC-cost
            # fix above) and a load_fn was supplied, fetch it here in
            # the sequential path too.
            if 'founder_block' not in data and load_fn is not None:
                loaded = load_fn(r_name)
                if loaded is not None and 'founder_block' in loaded:
                    data = dict(data)
                    data['founder_block'] = loaded['founder_block']
            corrected_painting = data['corrected_painting']
            
            if verbose:
                print(f"\n{r_name}:")
            
            # Get region from the painting - try different access patterns
            if hasattr(corrected_painting, 'region'):
                start_pos = corrected_painting.region[0]
                end_pos = corrected_painting.region[1]
            elif hasattr(corrected_painting, 'samples') and corrected_painting.samples:
                # Compute from sample chunks
                start_pos = min(s.chunks[0].start for s in corrected_painting.samples if s.chunks)
                end_pos = max(s.chunks[-1].end for s in corrected_painting.samples if s.chunks)
            else:
                # Try tuple-like access
                try:
                    start_pos = corrected_painting[0][0]
                    end_pos = corrected_painting[0][1]
                except:
                    print(f"  WARNING: Could not determine region for {r_name}, skipping")
                    continue
            
            bin_edges = compute_bin_edges(start_pos, end_pos, snps_per_bin=snps_per_bin)
            
            # Compute founder equivalence matrix
            if 'founder_block' in data:
                dense_haps, positions = founder_block_to_dense(data['founder_block'])
                equiv = compute_founder_equivalence_matrix(
                    dense_haps, positions, bin_edges,
                    max_diff_fraction=max_diff_fraction,
                    min_diff_sites=min_diff_sites
                )
                if verbose:
                    n_founders = dense_haps.shape[0]
                    print(f"  Computed founder equivalence matrix ({n_founders} founders)")
            else:
                # Fallback: no equivalence (identity only)
                n_bins = len(bin_edges) - 1
                # Assume at most 8 founders
                equiv = np.zeros((n_bins, 8, 8), dtype=np.bool_)
                for b in range(n_bins):
                    for f in range(8):
                        equiv[b, f, f] = True
                if verbose:
                    print(f"  WARNING: No founder_block found, using identity equivalence only")
            
            refined_painting = post_process_phase_greedy(
                corrected_painting,
                pedigree_df,
                sample_names,
                bin_edges,
                equiv,
                recomb_rate=recomb_rate,
                mismatch_cost=mismatch_cost,
                verbose=verbose
            )

            # Always write back to multi_contig_results[r_name], NOT
            # to `data`.  `data` may be a local dict copy (if the
            # load_fn fallback was taken above) -- writing to that
            # would lose the result.  Writing through the
            # multi_contig_results dict makes the assignment visible
            # to the caller regardless.
            multi_contig_results[r_name]['refined_painting'] = refined_painting
    
    if verbose:
        print(f"\nGreedy refinement complete.")
    
    return multi_contig_results


# =============================================================================
# PARSIMONIOUS F1 RECOLORING
# =============================================================================

def _compute_founder_ibs_in_region(founder_block, f1, f2, start_pos, end_pos,
                                    max_mismatch_rate=0.02):
    """
    Check if two founders are effectively identical (IBS) in a genomic region.
    
    Uses argmax alleles from the founder block. Two founders are considered
    equivalent if their mismatch rate is below max_mismatch_rate in the region.
    
    Args:
        founder_block: BlockResult with .positions and .haplotypes
        f1, f2: Founder IDs to compare
        start_pos, end_pos: Genomic region boundaries (bp)
        max_mismatch_rate: Maximum fraction of mismatching sites (default 2%)
    
    Returns:
        True if founders are effectively identical in the region
    """
    if f1 == f2:
        return True
    if f1 == -1 or f2 == -1:
        return True
    
    positions = founder_block.positions
    haplotypes = founder_block.haplotypes
    
    if f1 not in haplotypes or f2 not in haplotypes:
        return False
    
    # Find sites in region
    idx_start = np.searchsorted(positions, start_pos, side='left')
    idx_end = np.searchsorted(positions, end_pos, side='right')
    
    if idx_end <= idx_start:
        # No sites in region — assume equivalent (no evidence to distinguish)
        return True
    
    # Get alleles via argmax
    h1 = haplotypes[f1]
    h2 = haplotypes[f2]
    
    if h1.ndim == 2:
        a1 = np.argmax(h1[idx_start:idx_end], axis=1)
    else:
        a1 = h1[idx_start:idx_end]
    
    if h2.ndim == 2:
        a2 = np.argmax(h2[idx_start:idx_end], axis=1)
    else:
        a2 = h2[idx_start:idx_end]
    
    n_sites = len(a1)
    n_mismatches = np.sum(a1 != a2)
    mismatch_rate = n_mismatches / n_sites
    
    return mismatch_rate <= max_mismatch_rate


def _parsimonious_track_dp(segments, founder_block, all_founder_ids,
                            max_mismatch_rate=0.02):
    """
    DP to find the founder assignment that minimizes switches along one track.
    
    For each segment, computes which founders are IBS-equivalent to the
    originally assigned founder. Then finds the assignment sequence that
    minimizes the number of founder transitions.
    
    Args:
        segments: List of (start, end, original_founder_id) tuples
        founder_block: BlockResult for IBS checking
        all_founder_ids: List of all valid founder IDs
        max_mismatch_rate: Threshold for IBS equivalence
    
    Returns:
        List of (start, end, chosen_founder_id) tuples — same length as segments
    """
    n_segs = len(segments)
    if n_segs == 0:
        return []
    
    # For each segment, compute the set of compatible (IBS-equivalent) founders
    compatible = []
    for start, end, orig_fid in segments:
        equiv_set = set()
        for fid in all_founder_ids:
            if _compute_founder_ibs_in_region(founder_block, orig_fid, fid,
                                               start, end, max_mismatch_rate):
                equiv_set.add(fid)
        # Always include the original as fallback
        equiv_set.add(orig_fid)
        compatible.append(equiv_set)
    
    # DP: cost[seg][fid] = minimum switches to reach segment seg with founder fid
    # Use dict since founder IDs may be sparse
    INF = float('inf')
    
    # Initialize first segment
    prev_costs = {}
    for fid in compatible[0]:
        prev_costs[fid] = 0  # No switches needed at first segment
    
    # Backpointers: backptr[seg] = {fid: best_prev_fid}
    backptrs = [None] * n_segs
    backptrs[0] = {fid: None for fid in compatible[0]}
    
    # Forward pass
    for seg_idx in range(1, n_segs):
        curr_costs = {}
        curr_backptr = {}
        
        for fid in compatible[seg_idx]:
            best_cost = INF
            best_prev = None
            
            for prev_fid, prev_cost in prev_costs.items():
                cost = prev_cost + (0 if fid == prev_fid else 1)
                if cost < best_cost:
                    best_cost = cost
                    best_prev = prev_fid
            
            curr_costs[fid] = best_cost
            curr_backptr[fid] = best_prev
        
        prev_costs = curr_costs
        backptrs[seg_idx] = curr_backptr
    
    # Traceback: find best final founder
    best_final_fid = min(prev_costs, key=prev_costs.get)
    
    chosen = [None] * n_segs
    chosen[n_segs - 1] = best_final_fid
    
    for seg_idx in range(n_segs - 2, -1, -1):
        chosen[seg_idx] = backptrs[seg_idx + 1][chosen[seg_idx + 1]]
    
    # Build result
    result = []
    for i, (start, end, _orig) in enumerate(segments):
        result.append((start, end, chosen[i]))
    
    return result


def apply_parsimonious_f1_recoloring(
    block_painting,
    founder_block,
    pedigree_df: pd.DataFrame,
    sample_names: List[str],
    max_workers: int,
    max_mismatch_rate: float = 0.02,
    verbose: bool = True
) -> BlockPainting:
    """
    Recolor F1 (top-level parent) paintings for parsimony.
    
    For each F1 individual (no parents in the discovered pedigree), examines
    each haplotype track independently and replaces founder assignments with
    IBS-equivalent founders that minimize the total number of recombination
    switches along the track.
    
    Example:
        Original:  H1(0-5Mb) → H3(5-9Mb) → H4(9-10Mb)
        If H4 ≈ H1 in 9-10Mb region:
        Recolored: H1(0-5Mb) → H3(5-9Mb) → H1(9-10Mb)  (one fewer distinct founder)
    
    This makes the painting consistent with a simpler recombination history
    and helps downstream analysis (e.g., IBD detection, haplotype sharing).
    
    Args:
        block_painting: BlockPainting with SamplePainting objects
        founder_block: BlockResult with .positions and .haplotypes for IBS checking
        pedigree_df: DataFrame with Sample, Parent1, Parent2 columns
        sample_names: List of sample names (order matches block_painting)
        max_workers: REQUIRED.  Maximum parallel threads for the per-F1
            ThreadPoolExecutor.  Must be passed from the calling pipeline
            (e.g. pipeline.py's `n_processes`) so the recoloring stage
            respects the pipeline's machine-wide worker budget.  Passing
            None will raise.
        max_mismatch_rate: Maximum allele mismatch rate for IBS equivalence (default 2%)
        verbose: Print progress
    
    Returns:
        New BlockPainting with F1 samples recolored for parsimony

    Implementation note (May 2026):
        The per-F1 loop was previously sequential.  Each F1 calls
        _parsimonious_track_dp twice (once per track); each track DP
        calls _compute_founder_ibs_in_region for every (segment x
        all_founder_ids) pair.  With ~20 segments per F1 track and
        6 founders that's ~240 IBS comparisons per F1, each scanning
        thousands of SNPs via np.argmax/np.sum -- numpy calls that
        release the GIL.  We therefore parallelise the per-F1 work
        with a ThreadPoolExecutor; the heavy lifting in
        _compute_founder_ibs_in_region runs concurrently across F1s.
        The outer mutation of `new_samples` happens serially after
        the parallel pass, so there's no race on shared state.
        Verbose prints are buffered per-F1 and emitted in sorted-name
        order at the end to keep log output deterministic and tidy.
    """
    if max_workers is None:
        raise ValueError(
            "apply_parsimonious_f1_recoloring: max_workers must be "
            "specified (typically pipeline.py passes its n_processes "
            "here).  This stage runs after the contig pool has closed "
            "and therefore has access to the full machine-wide worker "
            "budget; the caller must pass that budget explicitly.")
    # Identify F1 samples (no parents in pedigree)
    f1_names = set()
    for _, row in pedigree_df.iterrows():
        if pd.isna(row.get('Parent1')) and pd.isna(row.get('Parent2')):
            f1_names.add(row['Sample'])
    
    if not f1_names:
        if verbose:
            print("  No F1 samples found — nothing to recolor")
        return block_painting
    
    # Get all founder IDs from the founder block
    all_founder_ids = sorted(list(founder_block.haplotypes.keys()))
    
    name_to_idx = {name: i for i, name in enumerate(sample_names)}
    
    new_samples = list(block_painting.samples)  # shallow copy

    # ---- Per-F1 worker (closure captures shared read-only data) ----
    def _process_one_f1(f1_name):
        """
        Process a single F1: compute new chunks (or return None if
        the parsimony pass yields no improvement) plus a small info
        dict for accounting and verbose logging.

        This function is intended to be called from a thread pool;
        it mutates nothing outside its return value and is therefore
        safe to call concurrently across F1 samples.

        Returns:
            (idx, new_chunks_or_None, info_dict) where info_dict has
            'orig_switches_0', 'new_switches_0', 'orig_switches_1',
            'new_switches_1', 'saved'.
        """
        if f1_name not in name_to_idx:
            return None
        idx = name_to_idx[f1_name]
        sample_painting = block_painting[idx]
        
        if not sample_painting.chunks:
            return None
        
        # Extract track segments from chunks
        # Track 0 (hap1): merge consecutive chunks with same founder
        # Track 1 (hap2): same
        def extract_track_segments(chunks, track):
            """Extract merged segments for one track from painting chunks."""
            if not chunks:
                return []
            segments = []
            curr_start = chunks[0].start
            curr_end = chunks[0].end
            curr_fid = chunks[0].hap1 if track == 0 else chunks[0].hap2
            
            for c in chunks[1:]:
                fid = c.hap1 if track == 0 else c.hap2
                if fid == curr_fid:
                    curr_end = c.end  # extend
                else:
                    segments.append((curr_start, curr_end, curr_fid))
                    curr_start = c.start
                    curr_end = c.end
                    curr_fid = fid
            
            segments.append((curr_start, curr_end, curr_fid))
            return segments
        
        track0_segs = extract_track_segments(sample_painting.chunks, 0)
        track1_segs = extract_track_segments(sample_painting.chunks, 1)
        
        # Count original switches
        orig_switches_0 = sum(1 for i in range(1, len(track0_segs))
                              if track0_segs[i][2] != track0_segs[i-1][2])
        orig_switches_1 = sum(1 for i in range(1, len(track1_segs))
                              if track1_segs[i][2] != track1_segs[i-1][2])
        
        # Run DP for each track
        new_track0 = _parsimonious_track_dp(track0_segs, founder_block,
                                             all_founder_ids, max_mismatch_rate)
        new_track1 = _parsimonious_track_dp(track1_segs, founder_block,
                                             all_founder_ids, max_mismatch_rate)
        
        # Count new switches
        new_switches_0 = sum(1 for i in range(1, len(new_track0))
                             if new_track0[i][2] != new_track0[i-1][2])
        new_switches_1 = sum(1 for i in range(1, len(new_track1))
                             if new_track1[i][2] != new_track1[i-1][2])
        
        saved = (orig_switches_0 + orig_switches_1) - (new_switches_0 + new_switches_1)
        
        info = {
            'orig_switches_0': orig_switches_0,
            'new_switches_0': new_switches_0,
            'orig_switches_1': orig_switches_1,
            'new_switches_1': new_switches_1,
            'saved': saved,
        }

        if saved <= 0:
            return (idx, None, info)  # No improvement
        
        # Rebuild painting from the two recolored tracks
        # Collect all breakpoints from both tracks
        breakpoints = set()
        for start, end, _ in new_track0:
            breakpoints.add(start)
            breakpoints.add(end)
        for start, end, _ in new_track1:
            breakpoints.add(start)
            breakpoints.add(end)
        sorted_bp = sorted(breakpoints)
        
        new_chunks = []
        for bp_idx in range(len(sorted_bp) - 1):
            seg_start = sorted_bp[bp_idx]
            seg_end = sorted_bp[bp_idx + 1]
            mid = (seg_start + seg_end) // 2
            
            # Find track0 founder at this position
            h1 = -1
            for s, e, fid in new_track0:
                if s <= mid < e:
                    h1 = fid
                    break
            
            # Find track1 founder at this position
            h2 = -1
            for s, e, fid in new_track1:
                if s <= mid < e:
                    h2 = fid
                    break
            
            # Merge with previous chunk if same founders
            if new_chunks and new_chunks[-1].hap1 == h1 and new_chunks[-1].hap2 == h2:
                prev = new_chunks[-1]
                new_chunks[-1] = PaintedChunk(prev.start, seg_end, h1, h2)
            else:
                new_chunks.append(PaintedChunk(seg_start, seg_end, h1, h2))
        
        return (idx, new_chunks, info)

    # ---- Submit one task per F1 to the thread pool ----
    # Threads (rather than processes) because the hot inner work in
    # _compute_founder_ibs_in_region (np.argmax, np.sum, numpy
    # comparisons) releases the GIL.  The thread count is capped at
    # the number of F1 samples, since over-subscription wastes
    # threads when there are more workers than tasks.  Iteration
    # order over `sorted(f1_names)` preserves the original code's
    # deterministic processing order.
    sorted_f1 = sorted(f1_names)
    if len(sorted_f1) == 0:
        return block_painting
    # Clamp to the number of F1 samples -- over-subscribing wastes
    # threads when there are more workers than tasks.  The lower
    # bound of 1 is for safety; any caller passing max_workers < 1
    # would otherwise produce a zero-thread executor.
    effective_workers = max(1, min(len(sorted_f1), max_workers))
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        worker_results = list(executor.map(_process_one_f1, sorted_f1))

    # ---- Sequential apply of worker results (matches original
    # semantics: same set of updates, same accounting counters,
    # same per-F1 verbose lines in deterministic order). ----
    total_switches_saved = 0
    total_samples_changed = 0
    for f1_name, result in zip(sorted_f1, worker_results):
        if result is None:
            continue
        idx, new_chunks, info = result
        if new_chunks is None:
            continue  # No improvement -- do not update new_samples[idx]

        total_switches_saved += info['saved']
        total_samples_changed += 1

        if verbose:
            print(f"  {f1_name}: track1 {info['orig_switches_0']}→"
                  f"{info['new_switches_0']} switches, "
                  f"track2 {info['orig_switches_1']}→"
                  f"{info['new_switches_1']} switches "
                  f"(saved {info['saved']})")

        new_samples[idx] = SamplePainting(idx, new_chunks)

    if verbose:
        print(f"  Recolored {total_samples_changed} F1 samples, "
              f"saved {total_switches_saved} total switches")
    
    return BlockPainting((block_painting.start_pos, block_painting.end_pos), new_samples)


def propagate_recoloring_to_offspring(
    block_painting,
    founder_block,
    pedigree_df: pd.DataFrame,
    sample_names: List[str],
    max_workers: int,
    max_mismatch_rate: float = 0.02,
    verbose: bool = True
) -> BlockPainting:
    """
    Propagate parsimonious founder IDs from parents to offspring.
    
    After F1 recoloring, parents use the most parsimonious founder IDs.
    But their children may still use IBS-equivalent but different founder IDs
    for the same inherited segment (because the Viterbi painting chose
    independently). This function walks the pedigree top-down and replaces
    each child segment's founder ID with the matching parent's founder ID
    when they are IBS-equivalent.
    
    This does not change alleles (IBS-equivalent founders have the same alleles
    by definition) — it only makes founder ID labels consistent across the
    pedigree for cleaner visualization and downstream analysis.
    
    Args:
        block_painting: BlockPainting with all samples (F1s already recolored)
        founder_block: BlockResult with .positions and .haplotypes
        pedigree_df: DataFrame with Sample, Parent1, Parent2, Generation columns
        sample_names: List of sample names (order matches block_painting)
        max_workers: REQUIRED.  Maximum parallel threads for the
            per-generation ThreadPoolExecutor.  Must be passed from the
            calling pipeline (e.g. pipeline.py's `n_processes`) so the
            propagation stage respects the pipeline's machine-wide
            worker budget.  Passing None will raise.
        max_mismatch_rate: IBS equivalence threshold (default 2%)
        verbose: Print progress
    
    Returns:
        New BlockPainting with offspring founder IDs propagated from parents

    Implementation note (May 2026):
        Propagation is strictly top-down across generations: F2 reads
        only F1 (already finalised by apply_parsimonious_f1_recoloring
        plus any earlier-generation propagation), F3 reads F1 and F2,
        etc.  Within a single generation, however, samples are
        independent -- each child reads only its two (already-final)
        parents and writes its own slot in new_samples.  We exploit
        this by processing each generation's samples in a thread pool
        while keeping the generation-by-generation outer loop
        sequential.  Threads are appropriate because the hot work in
        _compute_founder_ibs_in_region (np.argmax, np.sum, array
        comparisons) releases the GIL; using processes would force
        founder_block to be pickled across worker boundaries, which
        is wasteful.

        The previous implementation iterated `samples_by_gen`
        directly; the new implementation first groups by gen_num
        (preserving the original tie-break order within a gen, which
        is the pedigree_df row order), then runs each gen's group in
        parallel.  Per-child work is identical to the original
        per-iteration body; we extracted it into a closure
        `_process_one_sample` so the thread pool has a target.  The
        sequential merge-back step preserves total_replacements and
        total_samples_changed accounting exactly.
    """
    if max_workers is None:
        raise ValueError(
            "propagate_recoloring_to_offspring: max_workers must be "
            "specified (typically pipeline.py passes its n_processes "
            "here).  This stage runs after the contig pool has closed "
            "and therefore has access to the full machine-wide worker "
            "budget; the caller must pass that budget explicitly.")
    name_to_idx = {name: i for i, name in enumerate(sample_names)}
    
    # Build parent lookup
    parent_map = {}
    for _, row in pedigree_df.iterrows():
        sample = row['Sample']
        p1 = row.get('Parent1')
        p2 = row.get('Parent2')
        if pd.notna(p1) and pd.notna(p2):
            parent_map[sample] = (p1, p2)
    
    # Sort by generation so parents are processed before children
    gen_order = {'F1': 0, 'F2': 1, 'F3': 2, 'F4': 3, 'F5': 4}
    samples_by_gen = []
    for _, row in pedigree_df.iterrows():
        gen = row.get('Generation', 'F1')
        gen_num = gen_order.get(gen, int(gen[1:]) if gen.startswith('F') else 99)
        samples_by_gen.append((gen_num, gen, row['Sample']))
    samples_by_gen.sort(key=lambda x: x[0])
    
    # Work on a mutable copy of paintings
    new_samples = list(block_painting.samples)
    
    total_replacements = 0
    total_samples_changed = 0

    # ---- Group samples by generation while preserving within-gen
    # order (which matches pedigree_df iteration order after the
    # stable sort by gen_num above).  Each group is then processed
    # in a thread pool.  Inter-generation order is preserved (F2
    # finished before F3 starts) so that F3 reads finalised F2
    # parents from new_samples.
    from itertools import groupby
    groups_by_gen = []
    for gen_num, items in groupby(samples_by_gen, key=lambda x: x[0]):
        groups_by_gen.append((gen_num, list(items)))

    # ---- Per-sample worker (closure over read-only state and a
    # snapshot of `new_samples` taken once at the start of each
    # generation).  The snapshot is taken outside the closure when
    # we dispatch a generation so all workers within that generation
    # see a consistent view of parents.  Within a generation, no
    # worker reads or writes another worker's slot, so concurrent
    # execution is safe.
    def _process_one_sample(sample_name, parents_snapshot):
        """
        Process a single child sample: compute new chunks and
        replacement count using the parents_snapshot view of
        new_samples.  Returns (idx, new_chunks_or_None,
        replacements_count) or None if the sample has no parents,
        no painting, or no parent indices.

        parents_snapshot is the new_samples list as captured at
        the start of this generation -- safe to read concurrently
        across threads.
        """
        # Skip F1s — they have no parents, already recolored
        if sample_name not in parent_map:
            return None
        if sample_name not in name_to_idx:
            return None

        p1_name, p2_name = parent_map[sample_name]
        if p1_name not in name_to_idx or p2_name not in name_to_idx:
            return None

        child_idx = name_to_idx[sample_name]
        p1_idx = name_to_idx[p1_name]
        p2_idx = name_to_idx[p2_name]

        child_painting = parents_snapshot[child_idx]
        p1_painting = parents_snapshot[p1_idx]
        p2_painting = parents_snapshot[p2_idx]

        if not child_painting.chunks:
            return None

        # Build parent track segment lookups for fast midpoint queries
        def build_track_lookup(painting, track):
            """Build list of (start, end, founder_id) for one track."""
            if not painting.chunks:
                return []
            segs = []
            for c in painting.chunks:
                fid = c.hap1 if track == 0 else c.hap2
                segs.append((c.start, c.end, fid))
            return segs

        def find_founder_at_pos(track_segs, pos):
            """Find founder ID at a given position in a track."""
            for start, end, fid in track_segs:
                if start <= pos < end:
                    return fid
            return -1

        p1_track0 = build_track_lookup(p1_painting, 0)
        p1_track1 = build_track_lookup(p1_painting, 1)
        p2_track0 = build_track_lookup(p2_painting, 0)
        p2_track1 = build_track_lookup(p2_painting, 1)

        replacements = 0
        new_chunks = []

        for chunk in child_painting.chunks:
            new_h1 = chunk.hap1
            new_h2 = chunk.hap2
            mid = (chunk.start + chunk.end) // 2

            # For track 0 (hap1): find which parent track has an IBS-equivalent founder
            if chunk.hap1 != -1:
                # Check all 4 parent tracks for IBS match
                best_replacement = chunk.hap1
                for parent_tracks in [p1_track0, p1_track1, p2_track0, p2_track1]:
                    parent_fid = find_founder_at_pos(parent_tracks, mid)
                    if parent_fid != -1 and parent_fid != chunk.hap1:
                        if _compute_founder_ibs_in_region(
                            founder_block, chunk.hap1, parent_fid,
                            chunk.start, chunk.end, max_mismatch_rate
                        ):
                            best_replacement = parent_fid
                            break  # Take first IBS match from parents

                if best_replacement != chunk.hap1:
                    new_h1 = best_replacement
                    replacements += 1

            # For track 1 (hap2): same logic
            if chunk.hap2 != -1:
                best_replacement = chunk.hap2
                for parent_tracks in [p1_track0, p1_track1, p2_track0, p2_track1]:
                    parent_fid = find_founder_at_pos(parent_tracks, mid)
                    if parent_fid != -1 and parent_fid != chunk.hap2:
                        if _compute_founder_ibs_in_region(
                            founder_block, chunk.hap2, parent_fid,
                            chunk.start, chunk.end, max_mismatch_rate
                        ):
                            best_replacement = parent_fid
                            break

                if best_replacement != chunk.hap2:
                    new_h2 = best_replacement
                    replacements += 1

            # Merge with previous chunk if same founders
            if new_chunks and new_chunks[-1].hap1 == new_h1 and new_chunks[-1].hap2 == new_h2:
                prev = new_chunks[-1]
                new_chunks[-1] = PaintedChunk(prev.start, chunk.end, new_h1, new_h2)
            else:
                new_chunks.append(PaintedChunk(chunk.start, chunk.end, new_h1, new_h2))

        if replacements > 0:
            return (child_idx, new_chunks, replacements)
        else:
            return (child_idx, None, 0)

    # ---- Per-generation parallel pass ----
    # Use the caller-supplied max_workers (typically pipeline.py's
    # n_processes); clamping happens per-generation below so a small
    # generation (e.g. 20 F2s) does not over-subscribe threads.
    for gen_num, items in groups_by_gen:
        # Snapshot the parents view ONCE per generation -- all workers
        # within this generation use the same view, so concurrent
        # execution does not race on new_samples updates.  Updates
        # from this generation become visible to the NEXT generation
        # via the merge-back step below.
        parents_snapshot = list(new_samples)
        gen_sample_names = [name for (_gn, _g, name) in items]
        if not gen_sample_names:
            continue

        effective_workers = max(1, min(len(gen_sample_names), max_workers))
        if effective_workers == 1 or len(gen_sample_names) == 1:
            # No benefit from a thread pool for a single task; call
            # directly (and also useful for testing / lighter workloads).
            results = [_process_one_sample(n, parents_snapshot)
                       for n in gen_sample_names]
        else:
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                results = list(executor.map(
                    lambda n: _process_one_sample(n, parents_snapshot),
                    gen_sample_names
                ))

        # Sequential merge-back.  Iteration order matches the original
        # per-sample loop's order (same as samples_by_gen within this
        # generation) so the side effects on total_replacements /
        # total_samples_changed are identical to the original.
        for sample_name, result in zip(gen_sample_names, results):
            if result is None:
                continue
            child_idx, new_chunks, replacements = result
            if new_chunks is None or replacements <= 0:
                continue
            new_samples[child_idx] = SamplePainting(child_idx, new_chunks)
            total_replacements += replacements
            total_samples_changed += 1

    if verbose:
        print(f"  Propagated to {total_samples_changed} offspring, "
              f"{total_replacements} total segment replacements")
    
    return BlockPainting((block_painting.start_pos, block_painting.end_pos), new_samples)