"""
Phase Correction for Tolerance Paintings using Trio + Children Information.
REFACTORED: Now operates on BINS (like pedigree_inference) for ~150x speedup.

This module resolves phase ambiguity by considering:
1. Parent derivation: which parent haplotypes explain this sample's corrected haps
2. Child transmission: how this sample's corrected haps explain transmissions to children

Key insight: The painted tracks have arbitrary phase. We find phase[k] at each BIN
that minimizes total recombinations across parents AND children.

Algorithm:
- Pre-round: Initialize LLs using parent-derivation scores
- Rounds 1-3: Phase correction using parent + child information
- Post-rounds: Select best-LL consensus for each sample
"""

import numpy as np
import pandas as pd
import math
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("WARNING: Numba not found. Phase correction will be slow.")
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator

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
    """
    n_founders = dense_haps.shape[0]
    n_bins = len(bin_edges) - 1
    
    # Pre-allocate equivalence matrix - all founders equivalent to themselves
    equiv = np.zeros((n_bins, n_founders, n_founders), dtype=np.bool_)
    for b in range(n_bins):
        for f in range(n_founders):
            equiv[b, f, f] = True
    
    # Find SNP indices for each bin using searchsorted
    bin_start_indices = np.searchsorted(positions, bin_edges[:-1], side='left')
    bin_end_indices = np.searchsorted(positions, bin_edges[1:], side='left')
    
    # For each bin, compute pairwise founder differences
    for b in range(n_bins):
        start_idx = bin_start_indices[b]
        end_idx = bin_end_indices[b]
        
        if end_idx <= start_idx:
            # No SNPs in this bin - all founders equivalent
            equiv[b, :, :] = True
            continue
        
        n_sites_in_bin = end_idx - start_idx
        max_allowed_diff = max(int(max_diff_fraction * n_sites_in_bin), min_diff_sites)
        
        # Extract alleles for this bin
        bin_alleles = dense_haps[:, start_idx:end_idx]  # (n_founders, n_sites_in_bin)
        
        # Compare all pairs of founders
        for f1 in range(n_founders):
            for f2 in range(f1 + 1, n_founders):
                # Count differing sites (ignoring -1 which means unknown)
                valid_mask = (bin_alleles[f1, :] != -1) & (bin_alleles[f2, :] != -1)
                if np.sum(valid_mask) == 0:
                    # No valid sites to compare - assume equivalent
                    equiv[b, f1, f2] = True
                    equiv[b, f2, f1] = True
                else:
                    n_diff = np.sum((bin_alleles[f1, :] != bin_alleles[f2, :]) & valid_mask)
                    if n_diff <= max_allowed_diff:
                        equiv[b, f1, f2] = True
                        equiv[b, f2, f1] = True
    
    return equiv


@njit
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


def discretize_consensus_to_bins(consensus_painting, bin_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretize a SampleConsensusPainting to fixed bins, using uncertainty info for hom_mask.
    
    Args:
        consensus_painting: SampleConsensusPainting with chunks containing uncertainty sets
        bin_edges: Array of bin boundaries (n_bins + 1,)
    
    Returns:
        id_grid: (n_bins, 2) array of representative founder IDs
        potential_hom_mask: (n_bins,) boolean - True where homozygosity is possible
    """
    num_bins = len(bin_edges) - 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    id_grid = np.full((num_bins, 2), -1, dtype=np.int32)
    potential_hom_mask = np.ones(num_bins, dtype=np.bool_)
    
    cons_chunks = consensus_painting.chunks
    if not cons_chunks:
        return id_grid, potential_hom_mask
    
    # Get representative path for concrete IDs
    if consensus_painting.representative_path is not None:
        rep_chunks = consensus_painting.representative_path.chunks
    else:
        rep_chunks = None
    
    # Build lookup for consensus chunks
    c_ends = np.array([c.end for c in cons_chunks], dtype=np.int64)
    c_starts = np.array([c.start for c in cons_chunks], dtype=np.int64)
    
    chunk_indices = np.searchsorted(c_ends, bin_centers, side='right')
    chunk_indices = np.clip(chunk_indices, 0, len(cons_chunks) - 1)
    
    # Check potential homozygosity from uncertainty sets
    for b in range(num_bins):
        cidx = chunk_indices[b]
        chunk = cons_chunks[cidx]
        
        if bin_centers[b] < chunk.start or bin_centers[b] >= chunk.end:
            potential_hom_mask[b] = True
            continue
        
        # Intersection of possible founders non-empty => potentially homozygous
        intersection = chunk.possible_hap1 & chunk.possible_hap2
        potential_hom_mask[b] = len(intersection) > 0
    
    # Fill IDs from representative path
    if rep_chunks:
        rep_ends = np.array([c.end for c in rep_chunks], dtype=np.int64)
        rep_starts = np.array([c.start for c in rep_chunks], dtype=np.int64)
        rep_h1 = np.array([c.hap1 for c in rep_chunks], dtype=np.int32)
        rep_h2 = np.array([c.hap2 for c in rep_chunks], dtype=np.int32)
        
        rep_indices = np.searchsorted(rep_ends, bin_centers, side='right')
        rep_indices = np.clip(rep_indices, 0, len(rep_chunks) - 1)
        valid_mask = (bin_centers >= rep_starts[rep_indices]) & (bin_centers < rep_ends[rep_indices])
        
        id_grid[:, 0] = np.where(valid_mask, rep_h1[rep_indices], -1)
        id_grid[:, 1] = np.where(valid_mask, rep_h2[rep_indices], -1)
    
    return id_grid, potential_hom_mask


def compute_bin_edges(start_pos: int, end_pos: int, snps_per_bin: int = 150) -> np.ndarray:
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

@njit(fastmath=True)
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

@njit(fastmath=True)
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
    mismatch_cost: float = 4.6    # Soft mismatch penalty per bin
) -> Tuple[np.ndarray, float]:
    """
    Run phase correction Viterbi on BINS.
    
    State (3 bits):
        - Bit 0: phase (0=Track1→CorrHap1, 1=Track2→CorrHap1)
        - Bit 1: p1_choice (which P1 haplotype we're following)
        - Bit 2: p2_choice (which P2 haplotype we're following)
    
    Uses founder equivalence matrix to handle aliasing - two different founder IDs
    that are equivalent in a bin do not incur a mismatch penalty.
    
    IMPORTANT: Child recombination costs are computed INDEPENDENTLY of the parent's
    phase state. A child switching which track they inherit is a biological 
    recombination event that must be paid for regardless of how the parent's
    phase is labeled. This prevents the algorithm from "hiding" real child
    recombinations by flipping the parent's phase at the same location.
    
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
                
                # Child recombination costs - independent of parent phase
                # A child's recombination is a biological event that happens regardless
                # of how we label the parent's phase. The XOR trick was WRONG because
                # it allowed parent phase switches to "hide" child recombinations.
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

def build_corrected_painting_from_bins(
    original_painting: SamplePainting,
    phase_sequence: np.ndarray,
    bin_edges: np.ndarray,
    sample_idx: int
) -> SamplePainting:
    """
    Build corrected painting by applying bin-level phase sequence.
    
    Maps phase decisions back to original chunk boundaries for cleaner output.
    """
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    n_bins = len(bin_centers)
    
    # Get original tracks at bin centers
    id_grid, _ = discretize_painting_to_bins(original_painting, bin_edges)
    
    # Apply phase correction at bin level
    corr_h1 = np.where(phase_sequence == 0, id_grid[:, 0], id_grid[:, 1])
    corr_h2 = np.where(phase_sequence == 0, id_grid[:, 1], id_grid[:, 0])
    
    # Build chunks from corrected bin values
    chunks = []
    if n_bins == 0:
        return SamplePainting(sample_idx, [])
    
    chunk_start = int(bin_edges[0])
    chunk_h1 = corr_h1[0]
    chunk_h2 = corr_h2[0]
    
    for k in range(1, n_bins):
        if corr_h1[k] != chunk_h1 or corr_h2[k] != chunk_h2:
            # End current chunk at this bin's start
            chunk_end = int(bin_edges[k])
            chunks.append(PaintedChunk(chunk_start, chunk_end, int(chunk_h1), int(chunk_h2)))
            chunk_start = chunk_end
            chunk_h1 = corr_h1[k]
            chunk_h2 = corr_h2[k]
    
    # Final chunk extends to original painting end
    if original_painting.chunks:
        final_end = original_painting.chunks[-1].end
    else:
        final_end = int(bin_edges[-1])
    
    chunks.append(PaintedChunk(chunk_start, final_end, int(chunk_h1), int(chunk_h2)))
    
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
    mismatch_cost: float = 4.6
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
        mismatch_cost=mismatch_cost
    )
    
    phase_seq_1, score_1 = run_phase_correction_viterbi_binned(
        n_bins, sample_grid, p1_grid, p2_grid, hom_mask, child_sources,
        bin_widths, equiv, parent_assignment=1,
        recomb_rate=recomb_rate, phase_switch_cost=phase_switch_cost,
        mismatch_cost=mismatch_cost
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

def compute_parent_derivation_score_binned(
    sample_painting: SamplePainting,
    p1_painting: SamplePainting,
    p2_painting: SamplePainting,
    bin_edges: np.ndarray,
    recomb_rate: float = 5e-8
) -> float:
    """
    Compute how well sample can be derived from parents (phase-agnostic, binned).
    """
    sample_grid, _ = discretize_painting_to_bins(sample_painting, bin_edges)
    p1_grid, _ = discretize_painting_to_bins(p1_painting, bin_edges)
    p2_grid, _ = discretize_painting_to_bins(p2_painting, bin_edges)
    
    n_bins = len(bin_edges) - 1
    bin_widths = np.diff(bin_edges)
    
    score = 0.0
    
    for k in range(n_bins):
        s_founders = {sample_grid[k, 0], sample_grid[k, 1]} - {-1}
        p1_founders = {p1_grid[k, 0], p1_grid[k, 1]} - {-1}
        p2_founders = {p2_grid[k, 0], p2_grid[k, 1]} - {-1}
        
        parent_can_provide = p1_founders | p2_founders
        
        if s_founders and not s_founders.issubset(parent_can_provide):
            score -= 20.0  # Penalty for unexplainable founder
    
    # Recombination penalty
    for k in range(1, n_bins):
        changed = (sample_grid[k, 0] != sample_grid[k-1, 0] and 
                   sample_grid[k, 0] != -1 and sample_grid[k-1, 0] != -1)
        changed |= (sample_grid[k, 1] != sample_grid[k-1, 1] and
                    sample_grid[k, 1] != -1 and sample_grid[k-1, 1] != -1)
        if changed:
            theta = bin_widths[k] * recomb_rate
            if theta > 0.5:
                theta = 0.5
            if theta < 1e-15:
                theta = 1e-15
            score += math.log(theta) * 0.5
    
    return score


# =============================================================================
# MAIN DRIVER FUNCTIONS
# =============================================================================

def initialize_correction_states(
    tolerance_painting,  # BlockTolerancePainting
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
        tol_sample = tolerance_painting[i]
        
        consensus_states = []
        
        if hasattr(tol_sample, 'paths') and tol_sample.paths:
            for path in tol_sample.paths:
                painting = SamplePainting(i, list(path.chunks))
                consensus_states.append(ConsensusPaintingState(painting=painting, ll=0.0))
        elif hasattr(tol_sample, 'consensus_list') and tol_sample.consensus_list:
            for cons in tol_sample.consensus_list:
                if cons.representative_path:
                    painting = SamplePainting(i, list(cons.representative_path.chunks))
                else:
                    chunks = []
                    for cc in cons.chunks:
                        h1 = next(iter(cc.possible_hap1)) if cc.possible_hap1 else -1
                        h2 = next(iter(cc.possible_hap2)) if cc.possible_hap2 else -1
                        chunks.append(PaintedChunk(cc.start, cc.end, h1, h2))
                    painting = SamplePainting(i, chunks)
                consensus_states.append(ConsensusPaintingState(painting=painting, ll=0.0))
        else:
            chunks = tol_sample.chunks if hasattr(tol_sample, 'chunks') else []
            painting = SamplePainting(i, list(chunks))
            consensus_states.append(ConsensusPaintingState(painting=painting, ll=0.0))
        
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
    verbose: bool = True
) -> int:
    """
    Run one round of phase correction on all samples using BINNED data.
    
    Uses founder equivalence matrix to handle aliasing.
    
    Returns:
        Number of corrections made
    """
    corrections_made = 0
    
    # Build other parent lookup
    other_parent_map = {}
    for _, row in pedigree_df.iterrows():
        child = row['Sample']
        p1 = row.get('Parent1')
        p2 = row.get('Parent2')
        
        if pd.notna(p1) and pd.notna(p2):
            other_parent_map[(p1, child)] = p2
            other_parent_map[(p2, child)] = p1
    
    for name in sample_names:
        state = states[name]
        
        # Get parent paintings
        p1_painting = None
        p2_painting = None
        
        if state.parent1_name is not None:
            p1_state = states.get(state.parent1_name)
            if p1_state is not None:
                p1_painting = p1_state.get_best_painting()
        
        if state.parent2_name is not None:
            p2_state = states.get(state.parent2_name)
            if p2_state is not None:
                p2_painting = p2_state.get_best_painting()
        
        # Get children and other parents
        children_paintings = []
        other_parent_paintings = []
        
        for child_name in state.children_names:
            child_state = states.get(child_name)
            if child_state is None:
                continue
            
            child_painting = child_state.get_best_painting()
            if child_painting is None:
                continue
            
            other_parent_name = other_parent_map.get((name, child_name))
            if other_parent_name is None:
                continue
            
            other_parent_state = states.get(other_parent_name)
            if other_parent_state is None:
                continue
            
            other_painting = other_parent_state.get_best_painting()
            if other_painting is None:
                continue
            
            children_paintings.append(child_painting)
            other_parent_paintings.append(other_painting)
        
        # Skip if no information
        if p1_painting is None and p2_painting is None and not children_paintings:
            continue
        
        # Process each consensus painting
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
                mismatch_cost=mismatch_cost
            )
            
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

def _process_contig_worker(r_name):
    """Worker function for processing a single contig."""
    global _PARALLEL_DATA
    
    data = _PARALLEL_DATA['multi_contig_results'][r_name]
    pedigree_df = _PARALLEL_DATA['pedigree_df']
    sample_names = _PARALLEL_DATA['sample_names']
    num_rounds = _PARALLEL_DATA['num_rounds']
    snps_per_bin = _PARALLEL_DATA['snps_per_bin']
    recomb_rate = _PARALLEL_DATA['recomb_rate']
    max_diff_fraction = _PARALLEL_DATA.get('max_diff_fraction', 0.02)
    min_diff_sites = _PARALLEL_DATA.get('min_diff_sites', 2)
    
    tolerance_painting = data['tolerance_result']
    
    start_pos = tolerance_painting.start_pos
    end_pos = tolerance_painting.end_pos
    
    # Compute bin edges
    bin_edges = compute_bin_edges(start_pos, end_pos, snps_per_bin=snps_per_bin)
    n_bins = len(bin_edges) - 1
    
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
    
    # Initialize states
    states = initialize_correction_states(
        tolerance_painting, pedigree_df, sample_names, bin_edges
    )
    
    # Initialize LLs
    initialize_lls(states, bin_edges)
    
    # Run correction rounds
    final_round = num_rounds
    for round_idx in range(num_rounds):
        corrections = run_correction_round(
            states, pedigree_df, sample_names, bin_edges, equiv,
            recomb_rate=recomb_rate, verbose=False
        )
        
        if corrections == 0:
            final_round = round_idx + 1
            break
    
    # Build final painting
    final_painting = build_final_painting(states, sample_names, start_pos, end_pos)
    
    multi_consensus = sum(1 for s in states.values() if len(s.consensus_states) > 1)
    
    return (r_name, final_painting, multi_consensus, final_round)


def correct_phase_all_contigs(
    multi_contig_results: Dict,
    pedigree_df: pd.DataFrame,
    sample_names: List[str],
    num_rounds: int = 3,
    snps_per_bin: int = 150,
    recomb_rate: float = 5e-8,
    max_diff_fraction: float = 0.02,
    min_diff_sites: int = 2,
    verbose: bool = True,
    max_workers: Optional[int] = None,
    parallel: bool = True
) -> Dict:
    """
    Correct phase for all contigs using BINNED data (~150x faster than per-SNP).
    
    Uses founder equivalence to handle aliasing - two founders that are nearly
    identical in a bin are treated as equivalent for emission scoring.
    
    Args:
        multi_contig_results: Dict mapping region_name -> data dict containing:
            - 'tolerance_result': BlockTolerancePainting
            - 'founder_block': BlockResult with founder haplotypes (optional but recommended)
        pedigree_df: DataFrame with Sample, Parent1, Parent2 columns
        sample_names: List of sample names
        num_rounds: Number of correction rounds (default 3)
        snps_per_bin: Bin resolution (default 150, same as pedigree inference)
        recomb_rate: Per-bp recombination rate (default 5e-8)
        max_diff_fraction: Max fraction of differing sites for founder equivalence (default 2%)
        min_diff_sites: Min absolute number of differing sites for equivalence (default 2)
        verbose: Print progress
        max_workers: Maximum parallel workers (default: num contigs)
        parallel: Use parallel processing
    
    Returns:
        Updated multi_contig_results with 'corrected_painting' key added
    """
    global _PARALLEL_DATA
    import os
    import multiprocessing as mp
    
    contig_names = [
        r_name for r_name, data in multi_contig_results.items()
        if 'tolerance_result' in data
    ]
    n_contigs = len(contig_names)
    
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, n_contigs)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Phase Correction (BINNED, {snps_per_bin} SNPs/bin, {num_rounds} rounds)")
        if parallel and n_contigs > 1:
            print(f"Using {max_workers} parallel workers")
        print(f"{'='*60}")
    
    if parallel and n_contigs > 1:
        _PARALLEL_DATA = {
            'multi_contig_results': multi_contig_results,
            'pedigree_df': pedigree_df,
            'sample_names': sample_names,
            'num_rounds': num_rounds,
            'snps_per_bin': snps_per_bin,
            'recomb_rate': recomb_rate,
            'max_diff_fraction': max_diff_fraction,
            'min_diff_sites': min_diff_sites
        }
        
        if verbose:
            print(f"\nProcessing {n_contigs} contigs in parallel...")
        
        with mp.Pool(processes=max_workers) as pool:
            results = pool.map(_process_contig_worker, contig_names)
        
        for r_name, final_painting, multi_cons, final_round in results:
            multi_contig_results[r_name]['corrected_painting'] = final_painting
            if verbose:
                print(f"  {r_name}: converged round {final_round}, multi-consensus: {multi_cons}")
        
        _PARALLEL_DATA = {}
    
    else:
        for r_name in contig_names:
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
                    recomb_rate=recomb_rate, verbose=verbose
                )
                
                if verbose:
                    print(f"    Corrections made: {corrections}")
                
                if corrections == 0:
                    if verbose:
                        print(f"    Converged early!")
                    break
            
            final_painting = build_final_painting(states, sample_names, start_pos, end_pos)
            data['corrected_painting'] = final_painting
            
            if verbose:
                multi_consensus = sum(1 for s in states.values() if len(s.consensus_states) > 1)
                print(f"  Samples with multiple consensus: {multi_consensus}")
    
    if verbose:
        print(f"\nPhase correction complete.")
    
    return multi_contig_results


# =============================================================================
# GREEDY PHASE POST-PROCESSING
# =============================================================================

def find_hom_to_het_boundaries(hom_mask: np.ndarray) -> List[int]:
    """
    Find all bin indices where a HOM→HET transition occurs.
    
    A HOM→HET boundary at bin k means hom_mask[k-1] is True and hom_mask[k] is False.
    These are the only points where phase flips are biologically meaningful.
    
    Returns:
        List of bin indices where HOM→HET transitions occur
    """
    boundaries = []
    for k in range(1, len(hom_mask)):
        if hom_mask[k-1] and not hom_mask[k]:
            boundaries.append(k)
    return boundaries


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
    """
    flipped = sample_grid.copy()
    if end_bin is None:
        end_bin = len(sample_grid)
    
    # Swap columns in the flipped region
    flipped[start_bin:end_bin, 0] = sample_grid[start_bin:end_bin, 1]
    flipped[start_bin:end_bin, 1] = sample_grid[start_bin:end_bin, 0]
    
    return flipped


def check_mendelian_consistency(
    sample_grid: np.ndarray,
    p1_grid: np.ndarray,
    p2_grid: np.ndarray
) -> bool:
    """
    Check if sample_grid is Mendelian-consistent with parents.
    
    For each bin, each founder in the sample must be present in at least one parent.
    Returns True if consistent, False otherwise.
    """
    n_bins = sample_grid.shape[0]
    
    for k in range(n_bins):
        s_h1, s_h2 = sample_grid[k, 0], sample_grid[k, 1]
        p1_founders = {p1_grid[k, 0], p1_grid[k, 1]}
        p2_founders = {p2_grid[k, 0], p2_grid[k, 1]}
        all_parent_founders = p1_founders | p2_founders
        
        # Skip if uncertain
        if s_h1 == -1 and s_h2 == -1:
            continue
        
        # Check each sample founder
        if s_h1 != -1 and s_h1 not in all_parent_founders:
            # Allow if all parent values are -1 (uncertain)
            if -1 not in all_parent_founders or len(all_parent_founders - {-1}) > 0:
                if s_h1 not in all_parent_founders:
                    return False
        if s_h2 != -1 and s_h2 not in all_parent_founders:
            if -1 not in all_parent_founders or len(all_parent_founders - {-1}) > 0:
                if s_h2 not in all_parent_founders:
                    return False
    
    return True


@njit(fastmath=True)
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
    """
    n_bins = sample_grid.shape[0]
    total_score = 0.0
    
    # Parent matching score
    if p1_grid is not None and p2_grid is not None:
        parent_score = compute_parent_matching_score_fixed_phase(
            n_bins, sample_grid, p1_grid, p2_grid, bin_widths, equiv,
            recomb_rate=recomb_rate, mismatch_cost=mismatch_cost
        )
        total_score += parent_score
    
    # Child transmission scores
    for child_grid, child_hom, other_grid in zip(children_grids, children_hom_masks, other_parent_grids):
        _, child_score = run_8state_transmission_viterbi_binned(
            n_bins, sample_grid, other_grid, child_grid, child_hom,
            bin_widths, equiv, recomb_rate=recomb_rate, mismatch_cost=mismatch_cost
        )
        total_score += child_score
    
    return total_score


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
    
    Iteratively tries all singleton and pair flips at HOM→HET boundaries,
    selecting the best improvement until no improvement is found.
    
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
    """
    current_grid = sample_grid.copy()
    n_bins = len(hom_mask)
    
    # Find HOM→HET boundaries
    boundaries = find_hom_to_het_boundaries(hom_mask)
    n_boundaries = len(boundaries)
    
    if verbose:
        print(f"    Found {n_boundaries} HOM→HET boundaries")
    
    if n_boundaries == 0:
        # No boundaries to flip at
        return current_grid, 0
    
    # Compute initial score
    current_score = compute_total_score_for_grid(
        current_grid, p1_grid, p2_grid,
        children_grids, children_hom_masks, other_parent_grids,
        bin_widths, equiv, recomb_rate, mismatch_cost
    )
    
    if verbose:
        print(f"    Initial score: {current_score:.2f}")
    
    total_flips = 0
    
    for iteration in range(max_iterations):
        best_improvement = 0.0
        best_candidate = None
        best_candidate_grid = None
        
        # Try all singleton flips (flip from boundary to end)
        for i, b1 in enumerate(boundaries):
            candidate_grid = apply_phase_flip_to_grid(current_grid, b1, None)
            
            # Check Mendelian consistency if we have parents
            if p1_grid is not None and p2_grid is not None:
                if not check_mendelian_consistency(candidate_grid, p1_grid, p2_grid):
                    continue
            
            candidate_score = compute_total_score_for_grid(
                candidate_grid, p1_grid, p2_grid,
                children_grids, children_hom_masks, other_parent_grids,
                bin_widths, equiv, recomb_rate, mismatch_cost
            )
            
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
                    if not check_mendelian_consistency(candidate_grid, p1_grid, p2_grid):
                        continue
                
                candidate_score = compute_total_score_for_grid(
                    candidate_grid, p1_grid, p2_grid,
                    children_grids, children_hom_masks, other_parent_grids,
                    bin_widths, equiv, recomb_rate, mismatch_cost
                )
                
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
    verbose: bool = True
) -> BlockPainting:
    """
    Post-process corrected paintings using greedy phase refinement.
    
    For each sample, finds HOM→HET boundaries and tries all singleton/pair flips
    to minimize total recombinations (parent matching + child transmission).
    
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
        verbose: Print progress
    
    Returns:
        Refined BlockPainting
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
        
        for name in sample_names:
            sample_grid = sample_grids[name]
            hom_mask = hom_masks[name]
            
            # Get parent grids (use current versions, which may have been refined)
            p1_grid = None
            p2_grid = None
            if name in parent_map:
                p1_name, p2_name = parent_map[name]
                if p1_name in sample_grids:
                    p1_grid = sample_grids[p1_name]
                if p2_name in sample_grids:
                    p2_grid = sample_grids[p2_name]
            
            # Get children grids (use current versions)
            children_grids = []
            children_hom_masks = []
            other_parent_grids = []
            
            for child_name in children_map.get(name, []):
                if child_name not in sample_grids:
                    continue
                other_name = other_parent_map.get((name, child_name))
                if other_name is None or other_name not in sample_grids:
                    continue
                
                children_grids.append(sample_grids[child_name])
                children_hom_masks.append(hom_masks[child_name])
                other_parent_grids.append(sample_grids[other_name])
            
            # Skip if no constraints
            if p1_grid is None and p2_grid is None and not children_grids:
                continue
            
            # Run greedy refinement
            refined_grid, n_flips = greedy_phase_refinement_single_sample(
                sample_grid, hom_mask, p1_grid, p2_grid,
                children_grids, children_hom_masks, other_parent_grids,
                bin_widths, equiv, recomb_rate, mismatch_cost,
                verbose=False
            )
            
            if n_flips > 0:
                # Update the grid for use by other samples
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


def post_process_phase_greedy_all_contigs(
    multi_contig_results: Dict,
    pedigree_df: pd.DataFrame,
    sample_names: List[str],
    snps_per_bin: int = 150,
    recomb_rate: float = 5e-8,
    mismatch_cost: float = 4.6,
    max_diff_fraction: float = 0.02,
    min_diff_sites: int = 2,
    verbose: bool = True
) -> Dict:
    """
    Apply greedy phase refinement to all contigs.
    
    This is a post-processing step that should be run after correct_phase_all_contigs.
    It refines phase by trying singleton and pair flips at HOM→HET boundaries.
    
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
    
    Returns:
        Updated multi_contig_results with 'refined_painting' key added
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Greedy Phase Refinement Post-Processing")
        print(f"{'='*60}")
    
    contig_names = [
        r_name for r_name, data in multi_contig_results.items()
        if 'corrected_painting' in data
    ]
    
    for r_name in contig_names:
        data = multi_contig_results[r_name]
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
        
        data['refined_painting'] = refined_painting
    
    if verbose:
        print(f"\nGreedy refinement complete.")
    
    return multi_contig_results