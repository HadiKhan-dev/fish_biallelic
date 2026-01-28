"""
Phase Correction for Tolerance Paintings using Trio + Children Information.

This module resolves phase ambiguity by considering:
1. Parent derivation: which parent haplotypes explain this sample's corrected haps
2. Child transmission: how this sample's corrected haps explain transmissions to children

Key insight: The painted tracks have arbitrary phase. We find phase[k] at each position
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
# DATA STRUCTURES
# =============================================================================

@dataclass
class ConsensusPaintingState:
    """Tracks a consensus painting through the correction rounds."""
    painting: SamplePainting
    ll: float = 0.0
    phase_sequence: np.ndarray = None  # The phase[k] values chosen


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
# HELPER FUNCTIONS
# =============================================================================

def founder_block_to_dense(founder_block):
    """Convert BlockResult to dense matrix for fast lookup."""
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


def extract_tracks_at_positions(painting: SamplePainting, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract track1 and track2 founder IDs at each position.
    
    Returns:
        track1: (n_positions,) array of founder IDs
        track2: (n_positions,) array of founder IDs
    """
    n_pos = len(positions)
    track1 = np.full(n_pos, -1, dtype=np.int32)
    track2 = np.full(n_pos, -1, dtype=np.int32)
    
    chunks = painting.chunks if hasattr(painting, 'chunks') else []
    if not chunks:
        return track1, track2
    
    # Build arrays for vectorized lookup
    n_chunks = len(chunks)
    chunk_starts = np.array([c.start for c in chunks], dtype=np.int64)
    chunk_ends = np.array([c.end for c in chunks], dtype=np.int64)
    chunk_h1 = np.array([c.hap1 for c in chunks], dtype=np.int32)
    chunk_h2 = np.array([c.hap2 for c in chunks], dtype=np.int32)
    
    # Use searchsorted for fast lookup
    indices = np.searchsorted(chunk_ends, positions, side='right')
    indices = np.clip(indices, 0, n_chunks - 1)
    
    valid = (positions >= chunk_starts[indices]) & (positions < chunk_ends[indices])
    track1[valid] = chunk_h1[indices[valid]]
    track2[valid] = chunk_h2[indices[valid]]
    
    return track1, track2


def compute_hom_mask(track1: np.ndarray, track2: np.ndarray) -> np.ndarray:
    """
    Compute homozygosity mask: True where track1 == track2.
    Phase switches are free at HOM→HET boundaries.
    """
    return track1 == track2


# =============================================================================
# 8-STATE VITERBI FOR CHILD TRANSMISSION EXTRACTION
# =============================================================================

@njit(fastmath=True)
def run_8state_transmission_viterbi(
    n_positions: int,
    sample_track1: np.ndarray,  # (n_pos,) Parent S track 1
    sample_track2: np.ndarray,  # (n_pos,) Parent S track 2
    other_track1: np.ndarray,   # (n_pos,) Other parent track 1
    other_track2: np.ndarray,   # (n_pos,) Other parent track 2
    child_track1: np.ndarray,   # (n_pos,) Child track 1
    child_track2: np.ndarray,   # (n_pos,) Child track 2
    recomb_cost: float = 7.0,   # -log(recomb_prob) per recombination
    mismatch_cost: float = 50.0  # Cost for impossible inheritance
) -> Tuple[np.ndarray, float]:
    """
    Run 8-state Viterbi to determine which of S's tracks was transmitted to child at each position.
    
    States (3 bits):
        - Bit 0: Which of S's tracks is being transmitted (0=track1, 1=track2)
        - Bit 1: Which of O's tracks is being transmitted (0=track1, 1=track2)
        - Bit 2: Child phase (0=S→child_track1, 1=S→child_track2)
    
    Phase switches are FREE at HOM→HET boundaries in the child, since phase is
    meaningless in homozygous regions.
    
    Returns:
        source_from_s: (n_pos,) array where value is 0 if S transmitted track1, 1 if track2
        score: Total Viterbi score
    """
    n_states = 8
    
    # DP tables
    scores = np.zeros(n_states, dtype=np.float64)
    new_scores = np.zeros(n_states, dtype=np.float64)
    backpointers = np.zeros((n_positions, n_states), dtype=np.int8)
    
    # Compute child homozygosity mask
    # Phase is meaningless when child is homozygous
    child_hom = np.zeros(n_positions, dtype=np.bool_)
    for k in range(n_positions):
        child_hom[k] = (child_track1[k] == child_track2[k])
    
    # Initialize
    for state in range(n_states):
        s_choice = (state >> 0) & 1
        o_choice = (state >> 1) & 1
        c_phase = (state >> 2) & 1
        
        s_val = sample_track2[0] if s_choice else sample_track1[0]
        o_val = other_track2[0] if o_choice else other_track1[0]
        
        if c_phase == 0:
            expect_c1, expect_c2 = s_val, o_val
        else:
            expect_c1, expect_c2 = o_val, s_val
        
        cost = 0.0
        if expect_c1 != -1 and child_track1[0] != -1 and expect_c1 != child_track1[0]:
            cost -= mismatch_cost
        if expect_c2 != -1 and child_track2[0] != -1 and expect_c2 != child_track2[0]:
            cost -= mismatch_cost
        
        scores[state] = cost
    
    # Track previous homozygosity for HOM↔HET detection
    prev_hom = child_hom[0]
    
    # Forward pass
    for k in range(1, n_positions):
        curr_hom = child_hom[k]
        
        # Phase switch is FREE at ANY HOM↔HET boundary (in either direction)
        # because phase is meaningless in homozygous regions
        free_phase_switch = prev_hom or curr_hom
        
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
                    trans_cost -= recomb_cost
                if prev_o != curr_o:
                    trans_cost -= recomb_cost
                if prev_phase != curr_phase:
                    # Phase switch is free at HOM→HET boundary, expensive otherwise
                    if not free_phase_switch:
                        trans_cost -= mismatch_cost
                
                score = scores[prev_state] + trans_cost
                if score > best_score:
                    best_score = score
                    best_prev = prev_state
            
            # Emission cost
            s_val = sample_track2[k] if curr_s else sample_track1[k]
            o_val = other_track2[k] if curr_o else other_track1[k]
            
            if curr_phase == 0:
                expect_c1, expect_c2 = s_val, o_val
            else:
                expect_c1, expect_c2 = o_val, s_val
            
            emit_cost = 0.0
            if expect_c1 != -1 and child_track1[k] != -1 and expect_c1 != child_track1[k]:
                emit_cost -= mismatch_cost
            if expect_c2 != -1 and child_track2[k] != -1 and expect_c2 != child_track2[k]:
                emit_cost -= mismatch_cost
            
            new_scores[curr_state] = best_score + emit_cost
            backpointers[k, curr_state] = best_prev
        
        # Swap
        for s in range(n_states):
            scores[s] = new_scores[s]
        
        # Update previous homozygosity for next iteration
        prev_hom = curr_hom
    
    # Find best final state
    best_final = 0
    best_final_score = scores[0]
    for s in range(1, n_states):
        if scores[s] > best_final_score:
            best_final_score = scores[s]
            best_final = s
    
    # Traceback
    path = np.zeros(n_positions, dtype=np.int8)
    curr = best_final
    for k in range(n_positions - 1, -1, -1):
        path[k] = curr
        if k > 0:
            curr = backpointers[k, curr]
    
    # Extract source_from_s (bit 0 of state)
    source_from_s = np.zeros(n_positions, dtype=np.int8)
    for k in range(n_positions):
        source_from_s[k] = (path[k] >> 0) & 1
    
    return source_from_s, best_final_score


# =============================================================================
# PHASE CORRECTION VITERBI
# =============================================================================

@njit(fastmath=True)
def run_phase_correction_viterbi(
    n_positions: int,
    sample_track1: np.ndarray,      # (n_pos,) Sample's current track 1
    sample_track2: np.ndarray,      # (n_pos,) Sample's current track 2
    p1_track1: np.ndarray,          # (n_pos,) Parent 1 track 1
    p1_track2: np.ndarray,          # (n_pos,) Parent 1 track 2
    p2_track1: np.ndarray,          # (n_pos,) Parent 2 track 1
    p2_track2: np.ndarray,          # (n_pos,) Parent 2 track 2
    hom_mask: np.ndarray,           # (n_pos,) True where sample is homozygous
    child_sources: np.ndarray,      # (n_children, n_pos) Which track each child received
    parent_assignment: int,         # 0: P1→Hap1, P2→Hap2; 1: P1→Hap2, P2→Hap1
    recomb_cost: float = 7.0,       # Cost per recombination
    phase_switch_cost: float = 50.0, # Cost for phase switch (except at HOM→HET)
    mismatch_cost: float = 20.0     # Cost for parent mismatch (soft)
) -> Tuple[np.ndarray, float]:
    """
    Run phase correction Viterbi.
    
    State (3 bits):
        - Bit 0: phase (0=Track1→CorrHap1, 1=Track2→CorrHap1)
        - Bit 1: p1_choice (which P1 haplotype we're following)
        - Bit 2: p2_choice (which P2 haplotype we're following)
    
    Returns:
        phase_sequence: (n_pos,) array of phase values (0 or 1)
        score: Total score
    """
    n_states = 8
    n_children = child_sources.shape[0] if child_sources.size > 0 else 0
    
    # DP tables
    scores = np.full(n_states, -1e20, dtype=np.float64)
    new_scores = np.zeros(n_states, dtype=np.float64)
    backpointers = np.zeros((n_positions, n_states), dtype=np.int8)
    
    # Initialize at position 0
    for state in range(n_states):
        phase = (state >> 0) & 1
        p1_choice = (state >> 1) & 1
        p2_choice = (state >> 2) & 1
        
        # Corrected haplotypes based on phase
        if phase == 0:
            corr_h1 = sample_track1[0]
            corr_h2 = sample_track2[0]
        else:
            corr_h1 = sample_track2[0]
            corr_h2 = sample_track1[0]
        
        # Expected values from parents based on parent_assignment
        if parent_assignment == 0:
            # P1 → CorrHap1, P2 → CorrHap2
            expect_h1 = p1_track2[0] if p1_choice else p1_track1[0]
            expect_h2 = p2_track2[0] if p2_choice else p2_track1[0]
        else:
            # P1 → CorrHap2, P2 → CorrHap1
            expect_h1 = p2_track2[0] if p2_choice else p2_track1[0]
            expect_h2 = p1_track2[0] if p1_choice else p1_track1[0]
        
        # Emission cost for parent matching
        cost = 0.0
        if corr_h1 != -1 and expect_h1 != -1 and corr_h1 != expect_h1:
            cost -= mismatch_cost
        if corr_h2 != -1 and expect_h2 != -1 and corr_h2 != expect_h2:
            cost -= mismatch_cost
        
        scores[state] = cost
    
    # Track previous hom state for HOM↔HET detection
    prev_hom = hom_mask[0]
    
    # Forward pass
    for k in range(1, n_positions):
        curr_hom = hom_mask[k]
        
        # Phase switch is FREE at ANY HOM↔HET boundary (in either direction)
        # because phase is meaningless in homozygous regions
        free_phase_switch = prev_hom or curr_hom
        
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
                    trans_cost -= recomb_cost
                if prev_p2 != curr_p2:
                    trans_cost -= recomb_cost
                
                # Child recombination costs
                for c_idx in range(n_children):
                    # Child was following track source_from_s
                    # With phase correction, child follows:
                    #   CorrHap1 if source XOR phase == 0
                    #   CorrHap2 if source XOR phase == 1
                    prev_which_hap = child_sources[c_idx, k-1] ^ prev_phase
                    curr_which_hap = child_sources[c_idx, k] ^ curr_phase
                    
                    if prev_which_hap != curr_which_hap:
                        trans_cost -= recomb_cost
                
                score = scores[prev_state] + trans_cost
                if score > best_score:
                    best_score = score
                    best_prev = prev_state
            
            # Emission cost for parent matching
            if curr_phase == 0:
                corr_h1 = sample_track1[k]
                corr_h2 = sample_track2[k]
            else:
                corr_h1 = sample_track2[k]
                corr_h2 = sample_track1[k]
            
            if parent_assignment == 0:
                expect_h1 = p1_track2[k] if curr_p1 else p1_track1[k]
                expect_h2 = p2_track2[k] if curr_p2 else p2_track1[k]
            else:
                expect_h1 = p2_track2[k] if curr_p2 else p2_track1[k]
                expect_h2 = p1_track2[k] if curr_p1 else p1_track1[k]
            
            emit_cost = 0.0
            if corr_h1 != -1 and expect_h1 != -1 and corr_h1 != expect_h1:
                emit_cost -= mismatch_cost
            if corr_h2 != -1 and expect_h2 != -1 and corr_h2 != expect_h2:
                emit_cost -= mismatch_cost
            
            new_scores[curr_state] = best_score + emit_cost
            backpointers[k, curr_state] = best_prev
        
        # Swap scores
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
    path = np.zeros(n_positions, dtype=np.int8)
    curr = best_final
    for k in range(n_positions - 1, -1, -1):
        path[k] = curr
        if k > 0:
            curr = backpointers[k, curr]
    
    # Extract phase sequence (bit 0 of state)
    phase_sequence = np.zeros(n_positions, dtype=np.int8)
    for k in range(n_positions):
        phase_sequence[k] = (path[k] >> 0) & 1
    
    return phase_sequence, best_final_score


# =============================================================================
# PAINTING CONSTRUCTION
# =============================================================================

def build_corrected_painting(
    original_painting: SamplePainting,
    phase_sequence: np.ndarray,
    positions: np.ndarray,
    sample_idx: int
) -> SamplePainting:
    """
    Build corrected painting by applying phase sequence.
    
    CorrectedHap1[k] = Track1[k] if phase[k]=0 else Track2[k]
    CorrectedHap2[k] = Track2[k] if phase[k]=0 else Track1[k]
    """
    track1, track2 = extract_tracks_at_positions(original_painting, positions)
    
    # Apply phase correction
    corr_h1 = np.where(phase_sequence == 0, track1, track2)
    corr_h2 = np.where(phase_sequence == 0, track2, track1)
    
    # Build chunks by finding transitions
    chunks = []
    if len(positions) == 0:
        return SamplePainting(sample_idx, [])
    
    chunk_start = positions[0]
    chunk_h1 = corr_h1[0]
    chunk_h2 = corr_h2[0]
    
    for k in range(1, len(positions)):
        if corr_h1[k] != chunk_h1 or corr_h2[k] != chunk_h2:
            # End current chunk
            chunks.append(PaintedChunk(int(chunk_start), int(positions[k]), int(chunk_h1), int(chunk_h2)))
            chunk_start = positions[k]
            chunk_h1 = corr_h1[k]
            chunk_h2 = corr_h2[k]
    
    # Final chunk - extend to original painting end
    if original_painting.chunks:
        final_end = original_painting.chunks[-1].end
    else:
        final_end = int(positions[-1]) + 1
    
    chunks.append(PaintedChunk(int(chunk_start), final_end, int(chunk_h1), int(chunk_h2)))
    
    return SamplePainting(sample_idx, chunks)


# =============================================================================
# PARENT DERIVATION SCORE (for initialization)
# =============================================================================

def compute_parent_derivation_score(
    sample_painting: SamplePainting,
    p1_painting: SamplePainting,
    p2_painting: SamplePainting,
    positions: np.ndarray,
    recomb_cost: float = 7.0
) -> float:
    """
    Compute how well sample can be derived from parents (phase-agnostic).
    
    This is used for initializing LL before round 1.
    """
    s_t1, s_t2 = extract_tracks_at_positions(sample_painting, positions)
    p1_t1, p1_t2 = extract_tracks_at_positions(p1_painting, positions)
    p2_t1, p2_t2 = extract_tracks_at_positions(p2_painting, positions)
    
    n_pos = len(positions)
    
    # Check if sample founders can be explained by parents at each position
    score = 0.0
    
    for k in range(n_pos):
        s_founders = {s_t1[k], s_t2[k]} - {-1}
        p1_founders = {p1_t1[k], p1_t2[k]} - {-1}
        p2_founders = {p2_t1[k], p2_t2[k]} - {-1}
        
        # Check if sample founders can come from parents
        parent_can_provide = p1_founders | p2_founders
        
        if s_founders and not s_founders.issubset(parent_can_provide):
            score -= 50.0  # Penalty for unexplainable founder
    
    # Simple recombination estimate based on track changes
    for k in range(1, n_pos):
        if s_t1[k] != s_t1[k-1] or s_t2[k] != s_t2[k-1]:
            score -= recomb_cost * 0.5  # Rough estimate
    
    return score


# =============================================================================
# SINGLE SAMPLE PHASE CORRECTION
# =============================================================================

def correct_sample_phase(
    sample_state: ConsensusPaintingState,
    p1_painting: Optional[SamplePainting],
    p2_painting: Optional[SamplePainting],
    children_paintings: List[SamplePainting],
    other_parent_paintings: List[SamplePainting],  # Other parent for each child
    positions: np.ndarray,
    sample_idx: int,
    recomb_cost: float = 7.0,
    phase_switch_cost: float = 50.0,
    mismatch_cost: float = 20.0
) -> Tuple[SamplePainting, float, np.ndarray]:
    """
    Correct phase for a single consensus painting.
    
    Handles:
    - F1 samples: p1_painting and p2_painting are None, use children only
    - F2 samples: All inputs available, use both parents and children
    - F3 samples: children_paintings is empty, use parents only
    
    Returns:
        corrected_painting: The phase-corrected painting
        total_ll: Total log-likelihood
        phase_sequence: The chosen phase at each position
    """
    sample_painting = sample_state.painting
    n_pos = len(positions)
    
    # Extract tracks
    s_t1, s_t2 = extract_tracks_at_positions(sample_painting, positions)
    
    # Handle None parent paintings by creating all-(-1) tracks
    # This way the Viterbi won't penalize any parent mismatches
    if p1_painting is not None:
        p1_t1, p1_t2 = extract_tracks_at_positions(p1_painting, positions)
    else:
        p1_t1 = np.full(n_pos, -1, dtype=np.int32)
        p1_t2 = np.full(n_pos, -1, dtype=np.int32)
    
    if p2_painting is not None:
        p2_t1, p2_t2 = extract_tracks_at_positions(p2_painting, positions)
    else:
        p2_t1 = np.full(n_pos, -1, dtype=np.int32)
        p2_t2 = np.full(n_pos, -1, dtype=np.int32)
    
    # Compute homozygosity mask
    hom_mask = compute_hom_mask(s_t1, s_t2)
    
    # Precompute child transmission sequences
    n_children = len(children_paintings)
    child_sources = np.zeros((n_children, n_pos), dtype=np.int8)
    
    for c_idx, (child_p, other_p) in enumerate(zip(children_paintings, other_parent_paintings)):
        c_t1, c_t2 = extract_tracks_at_positions(child_p, positions)
        o_t1, o_t2 = extract_tracks_at_positions(other_p, positions)
        
        source_seq, _ = run_8state_transmission_viterbi(
            n_pos, s_t1, s_t2, o_t1, o_t2, c_t1, c_t2,
            recomb_cost=recomb_cost, mismatch_cost=mismatch_cost * 2
        )
        child_sources[c_idx, :] = source_seq
    
    # Run phase correction Viterbi for both parent assignments
    # Assignment 0: P1→CorrHap1, P2→CorrHap2
    phase_seq_0, score_0 = run_phase_correction_viterbi(
        n_pos, s_t1, s_t2, p1_t1, p1_t2, p2_t1, p2_t2,
        hom_mask, child_sources, parent_assignment=0,
        recomb_cost=recomb_cost, phase_switch_cost=phase_switch_cost,
        mismatch_cost=mismatch_cost
    )
    
    # Assignment 1: P1→CorrHap2, P2→CorrHap1
    phase_seq_1, score_1 = run_phase_correction_viterbi(
        n_pos, s_t1, s_t2, p1_t1, p1_t2, p2_t1, p2_t2,
        hom_mask, child_sources, parent_assignment=1,
        recomb_cost=recomb_cost, phase_switch_cost=phase_switch_cost,
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
    corrected_painting = build_corrected_painting(sample_painting, phase_sequence, positions, sample_idx)
    
    return corrected_painting, total_ll, phase_sequence


# =============================================================================
# MAIN DRIVER
# =============================================================================

def initialize_correction_states(
    tolerance_painting,  # BlockTolerancePainting
    pedigree_df: pd.DataFrame,
    sample_names: List[str],
    positions: np.ndarray
) -> Dict[str, SampleCorrectionState]:
    """
    Initialize SampleCorrectionState for each sample.
    """
    # Build parent/child lookup
    parent_map = {}  # sample -> (p1, p2)
    children_map = {name: [] for name in sample_names}  # sample -> list of children
    
    for _, row in pedigree_df.iterrows():
        sample = row['Sample']
        p1 = row.get('Parent1')
        p2 = row.get('Parent2')
        
        if pd.notna(p1) and pd.notna(p2):
            parent_map[sample] = (p1, p2)
            # Add this sample as child of both parents
            if p1 in children_map:
                children_map[p1].append(sample)
            if p2 in children_map:
                children_map[p2].append(sample)
    
    # Build states
    states = {}
    
    for i, name in enumerate(sample_names):
        # Get consensus paintings from tolerance result
        tol_sample = tolerance_painting[i]
        
        consensus_states = []
        
        if hasattr(tol_sample, 'paths') and tol_sample.paths:
            # Each path becomes a consensus painting
            for path in tol_sample.paths:
                painting = SamplePainting(i, list(path.chunks))
                consensus_states.append(ConsensusPaintingState(painting=painting, ll=0.0))
        elif hasattr(tol_sample, 'consensus_list') and tol_sample.consensus_list:
            for cons in tol_sample.consensus_list:
                if cons.representative_path:
                    painting = SamplePainting(i, list(cons.representative_path.chunks))
                else:
                    # Build from consensus chunks (take first from each set)
                    chunks = []
                    for cc in cons.chunks:
                        h1 = next(iter(cc.possible_hap1)) if cc.possible_hap1 else -1
                        h2 = next(iter(cc.possible_hap2)) if cc.possible_hap2 else -1
                        chunks.append(PaintedChunk(cc.start, cc.end, h1, h2))
                    painting = SamplePainting(i, chunks)
                consensus_states.append(ConsensusPaintingState(painting=painting, ll=0.0))
        else:
            # Fallback: use chunks directly if available
            chunks = tol_sample.chunks if hasattr(tol_sample, 'chunks') else []
            painting = SamplePainting(i, list(chunks))
            consensus_states.append(ConsensusPaintingState(painting=painting, ll=0.0))
        
        # Get parents and children
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
    positions: np.ndarray
) -> None:
    """
    Initialize LL scores using parent-derivation score (before round 1).
    """
    for name, state in states.items():
        if state.parent1_name is None or state.parent2_name is None:
            # F1 sample - no parents, LL stays at 0
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
            cons_state.ll = compute_parent_derivation_score(
                cons_state.painting, p1_painting, p2_painting, positions
            )


def run_correction_round(
    states: Dict[str, SampleCorrectionState],
    pedigree_df: pd.DataFrame,
    sample_names: List[str],
    positions: np.ndarray,
    recomb_cost: float = 7.0,
    phase_switch_cost: float = 50.0,
    mismatch_cost: float = 20.0,
    verbose: bool = True
) -> int:
    """
    Run one round of phase correction on all samples.
    
    Handles:
    - F1 samples (no parents): Use children only
    - F2 samples (parents + children): Use both
    - F3 samples (parents, no children): Use parents only
    
    Returns:
        Number of corrections made
    """
    corrections_made = 0
    
    # Build parent/child lookup for "other parent" resolution
    other_parent_map = {}  # (sample, child) -> other_parent
    for _, row in pedigree_df.iterrows():
        child = row['Sample']
        p1 = row.get('Parent1')
        p2 = row.get('Parent2')
        
        if pd.notna(p1) and pd.notna(p2):
            other_parent_map[(p1, child)] = p2
            other_parent_map[(p2, child)] = p1
    
    # Process each sample
    for name in sample_names:
        state = states[name]
        
        # Get parent paintings (None for F1 samples)
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
        
        # Get children paintings and their other parents
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
        
        # Skip only if we have NO information at all (no parents AND no children)
        if p1_painting is None and p2_painting is None and not children_paintings:
            continue
        
        # Process each consensus painting
        for cons_state in state.consensus_states:
            corrected, new_ll, phase_seq = correct_sample_phase(
                cons_state,
                p1_painting,
                p2_painting,
                children_paintings,
                other_parent_paintings,
                positions,
                sample_idx=state.sample_idx,
                recomb_cost=recomb_cost,
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
            
            # Update state
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
            # Fallback: empty painting
            final_samples.append(SamplePainting(state.sample_idx, []))
    
    return BlockPainting((start_pos, end_pos), final_samples)


# =============================================================================
# PUBLIC API
# =============================================================================

# Module-level variable for sharing data with worker processes (via fork)
_PARALLEL_DATA = {}

def _process_contig_worker(r_name):
    """
    Worker function for processing a single contig.
    Accesses data via module-level _PARALLEL_DATA (inherited via fork).
    
    Args:
        r_name: Name of the region/contig to process
    
    Returns:
        (region_name, final_painting, multi_consensus_count, final_round)
    """
    global _PARALLEL_DATA
    
    data = _PARALLEL_DATA['multi_contig_results'][r_name]
    pedigree_df = _PARALLEL_DATA['pedigree_df']
    sample_names = _PARALLEL_DATA['sample_names']
    num_rounds = _PARALLEL_DATA['num_rounds']
    
    tolerance_painting = data['tolerance_result']
    founder_block = data['control_founder_block']
    positions = founder_block.positions
    
    # Initialize states
    states = initialize_correction_states(
        tolerance_painting, pedigree_df, sample_names, positions
    )
    
    # Initialize LLs
    initialize_lls(states, positions)
    
    # Run correction rounds
    final_round = num_rounds
    for round_idx in range(num_rounds):
        corrections = run_correction_round(
            states, pedigree_df, sample_names, positions,
            verbose=False
        )
        
        if corrections == 0:
            final_round = round_idx + 1
            break
    
    # Build final painting
    start_pos = tolerance_painting.start_pos
    end_pos = tolerance_painting.end_pos
    
    final_painting = build_final_painting(states, sample_names, start_pos, end_pos)
    
    # Count multi-consensus samples
    multi_consensus = sum(1 for s in states.values() if len(s.consensus_states) > 1)
    
    return (r_name, final_painting, multi_consensus, final_round)


def correct_phase_all_contigs(
    multi_contig_results: Dict,
    pedigree_df: pd.DataFrame,
    sample_names: List[str],
    num_rounds: int = 3,
    verbose: bool = True,
    max_workers: Optional[int] = None,
    parallel: bool = True
) -> Dict:
    """
    Correct phase for all contigs using parent + children information.
    
    Args:
        multi_contig_results: Dict mapping region_name -> data dict containing:
            - 'tolerance_result': BlockTolerancePainting
            - 'control_founder_block': BlockResult
        pedigree_df: DataFrame with Sample, Parent1, Parent2 columns
        sample_names: List of sample names
        num_rounds: Number of correction rounds (default 3)
        verbose: Print progress
        max_workers: Maximum number of parallel workers (default: number of contigs)
        parallel: If True, use parallel processing; if False, process sequentially
    
    Returns:
        Updated multi_contig_results with 'corrected_painting' key added
    """
    global _PARALLEL_DATA
    import os
    import multiprocessing as mp
    
    # Get list of contigs to process
    contig_names = [
        r_name for r_name, data in multi_contig_results.items()
        if 'tolerance_result' in data and 'control_founder_block' in data
    ]
    n_contigs = len(contig_names)
    
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, n_contigs)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Phase Correction (Parent + Children, {num_rounds} rounds)")
        if parallel and n_contigs > 1:
            print(f"Using {max_workers} parallel workers")
        print(f"{'='*60}")
    
    if parallel and n_contigs > 1:
        # Set up shared data for workers (accessed via fork, no pickling)
        _PARALLEL_DATA = {
            'multi_contig_results': multi_contig_results,
            'pedigree_df': pedigree_df,
            'sample_names': sample_names,
            'num_rounds': num_rounds
        }
        
        if verbose:
            print(f"\nProcessing {n_contigs} contigs in parallel...")
        
        # Use multiprocessing.Pool with fork (default on Linux)
        # This inherits parent memory without pickling
        with mp.Pool(processes=max_workers) as pool:
            results = pool.map(_process_contig_worker, contig_names)
        
        # Update original dict with results
        for r_name, final_painting, multi_cons, final_round in results:
            multi_contig_results[r_name]['corrected_painting'] = final_painting
            if verbose:
                print(f"  {r_name}: converged round {final_round}, multi-consensus: {multi_cons}")
        
        # Clear shared data
        _PARALLEL_DATA = {}
    
    else:
        # Sequential processing
        for r_name in contig_names:
            data = multi_contig_results[r_name]
            
            if verbose:
                print(f"\nProcessing {r_name}...")
            
            tolerance_painting = data['tolerance_result']
            founder_block = data['control_founder_block']
            positions = founder_block.positions
            
            # Initialize states
            states = initialize_correction_states(
                tolerance_painting, pedigree_df, sample_names, positions
            )
            
            # Initialize LLs
            if verbose:
                print(f"  Initializing LLs...")
            initialize_lls(states, positions)
            
            # Run correction rounds
            for round_idx in range(num_rounds):
                if verbose:
                    print(f"  Round {round_idx + 1}/{num_rounds}...")
                
                corrections = run_correction_round(
                    states, pedigree_df, sample_names, positions,
                    verbose=verbose
                )
                
                if verbose:
                    print(f"    Corrections made: {corrections}")
                
                if corrections == 0:
                    if verbose:
                        print(f"    Converged early!")
                    break
            
            # Build final painting
            start_pos = tolerance_painting.start_pos
            end_pos = tolerance_painting.end_pos
            
            final_painting = build_final_painting(states, sample_names, start_pos, end_pos)
            
            data['corrected_painting'] = final_painting
            
            if verbose:
                multi_consensus = sum(1 for s in states.values() if len(s.consensus_states) > 1)
                print(f"  Samples with multiple consensus: {multi_consensus}")
    
    if verbose:
        print(f"\nPhase correction complete.")
    
    return multi_contig_results