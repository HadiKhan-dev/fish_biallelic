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
                mismatch_cost=mismatch_cost,
                phase_zero_preference=phase_zero_preference,
                use_xor_child_recomb=use_xor_child_recomb
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
    
    # Load data: use load_fn if provided, otherwise read from multi_contig_results
    load_fn = _PARALLEL_DATA.get('load_fn')
    if load_fn is not None:
        data = load_fn(r_name)
    else:
        data = _PARALLEL_DATA['multi_contig_results'][r_name]
    
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
            recomb_rate=recomb_rate,
            phase_switch_cost=phase_switch_cost,
            mismatch_cost=mismatch_cost,
            phase_zero_preference=phase_zero_preference,
            use_xor_child_recomb=use_xor_child_recomb,
            verbose=False
        )
        
        if corrections == 0:
            final_round = round_idx + 1
            break
    
    # Build final painting
    final_painting = build_final_painting(states, sample_names, start_pos, end_pos)
    
    multi_consensus = sum(1 for s in states.values() if len(s.consensus_states) > 1)
    
    # Return founder_block so main process can store it for greedy refinement
    founder_block = data.get('founder_block')
    return (r_name, final_painting, multi_consensus, final_round, founder_block)


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
    max_workers: Optional[int] = None,
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
        max_workers: Maximum parallel workers (default: num contigs)
        parallel: Use parallel processing
        load_fn: Optional callable(r_name) -> dict with 'tolerance_result' and 'founder_block'.
                 If provided, workers load their own data (parallelizes I/O).
    
    Returns:
        Updated multi_contig_results with 'corrected_painting' key added
    """
    global _PARALLEL_DATA
    import os
    import multiprocessing as mp
    
    if load_fn is not None:
        # With load_fn, workers load their own data — keys just need contig names
        contig_names = list(multi_contig_results.keys())
    else:
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
            if load_fn is not None:
                print(f"Workers will load data from checkpoints (parallel I/O)")
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
            'min_diff_sites': min_diff_sites,
            # Viterbi hyperparameters forwarded to workers
            'phase_switch_cost': phase_switch_cost,
            'mismatch_cost': mismatch_cost,
            'phase_zero_preference': phase_zero_preference,
            'use_xor_child_recomb': use_xor_child_recomb,
            'load_fn': load_fn
        }
        
        if verbose:
            print(f"\nProcessing {n_contigs} contigs in parallel...")
        
        with mp.Pool(processes=max_workers) as pool:
            results = pool.map(_process_contig_worker, contig_names)
        
        for r_name, final_painting, multi_cons, final_round, founder_block in results:
            multi_contig_results.setdefault(r_name, {})['corrected_painting'] = final_painting
            if founder_block is not None:
                multi_contig_results[r_name]['founder_block'] = founder_block
            if verbose:
                print(f"  {r_name}: converged round {final_round}, multi-consensus: {multi_cons}")
        
        _PARALLEL_DATA = {}
    
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

def find_hom_to_het_boundaries(hom_mask: np.ndarray) -> List[int]:
    """
    Find all bin indices where a HOM→HET transition occurs.
    
    A HOM→HET boundary at bin k means hom_mask[k-1] is True and hom_mask[k] is False.
    These are points where phase flips are biologically meaningful.
    
    Returns:
        List of bin indices where HOM→HET transitions occur
    """
    boundaries = []
    for k in range(1, len(hom_mask)):
        if hom_mask[k-1] and not hom_mask[k]:
            boundaries.append(k)
    return boundaries


def find_double_recomb_boundaries(sample_grid: np.ndarray) -> List[int]:
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
        List of bin indices where double recombinations occur
    """
    boundaries = []
    n_bins = sample_grid.shape[0]
    
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
            boundaries.append(k)
    
    return boundaries


def find_all_valid_flip_boundaries(
    sample_grid: np.ndarray,
    hom_mask: np.ndarray
) -> List[int]:
    """
    Find all valid flip boundaries: HOM→HET transitions OR double recombinations.
    
    Phase can be flipped at:
    1. HOM→HET transitions: before the transition, phase is arbitrary (both tracks same)
    2. Double recombinations: both tracks change, so swapping destinations is equivalent
    
    Note: HOM→HET with double recomb is already covered by case 1 (the HOM part
    means both tracks were the same, so phase was already arbitrary).
    
    Returns:
        Sorted list of unique bin indices where flips are valid
    """
    hom_het_boundaries = find_hom_to_het_boundaries(hom_mask)
    double_recomb_boundaries = find_double_recomb_boundaries(sample_grid)
    
    # Combine and deduplicate
    all_boundaries = set(hom_het_boundaries) | set(double_recomb_boundaries)
    
    return sorted(all_boundaries)


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
                    if not check_mendelian_consistency(candidate_grid, p1_grid, p2_grid, equiv):
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


def _greedy_contig_worker(r_name):
    """Worker function for parallel greedy refinement of a single contig."""
    global _PARALLEL_DATA
    
    data = _PARALLEL_DATA['multi_contig_results'][r_name]
    pedigree_df = _PARALLEL_DATA['pedigree_df']
    sample_names = _PARALLEL_DATA['sample_names']
    snps_per_bin = _PARALLEL_DATA['snps_per_bin']
    recomb_rate = _PARALLEL_DATA['recomb_rate']
    mismatch_cost = _PARALLEL_DATA['mismatch_cost']
    max_diff_fraction = _PARALLEL_DATA.get('max_diff_fraction', 0.02)
    min_diff_sites = _PARALLEL_DATA.get('min_diff_sites', 2)
    
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
    
    bin_edges = compute_bin_edges(start_pos, end_pos, snps_per_bin=snps_per_bin)
    
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
    
    refined_painting = post_process_phase_greedy(
        corrected_painting,
        pedigree_df,
        sample_names,
        bin_edges,
        equiv,
        recomb_rate=recomb_rate,
        mismatch_cost=mismatch_cost,
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
    max_workers: Optional[int] = None,
    parallel: bool = True
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
        max_workers: Maximum parallel workers (default: num contigs)
        parallel: Use parallel processing
    
    Returns:
        Updated multi_contig_results with 'refined_painting' key added
    """
    global _PARALLEL_DATA
    import os
    import multiprocessing as mp
    
    contig_names = [
        r_name for r_name, data in multi_contig_results.items()
        if 'corrected_painting' in data
    ]
    n_contigs = len(contig_names)
    
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, n_contigs)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Greedy Phase Refinement Post-Processing")
        if parallel and n_contigs > 1:
            print(f"Using {max_workers} parallel workers")
        print(f"{'='*60}")
    
    if parallel and n_contigs > 1:
        _PARALLEL_DATA = {
            'multi_contig_results': multi_contig_results,
            'pedigree_df': pedigree_df,
            'sample_names': sample_names,
            'snps_per_bin': snps_per_bin,
            'recomb_rate': recomb_rate,
            'mismatch_cost': mismatch_cost,
            'max_diff_fraction': max_diff_fraction,
            'min_diff_sites': min_diff_sites
        }
        
        if verbose:
            print(f"\nProcessing {n_contigs} contigs in parallel...")
        
        with mp.Pool(processes=max_workers) as pool:
            results = pool.map(_greedy_contig_worker, contig_names)
        
        for r_name, refined_painting, n_founders, n_refined in results:
            multi_contig_results[r_name]['refined_painting'] = refined_painting
            if verbose:
                print(f"  {r_name}: {n_founders} founders, {n_refined} samples refined")
        
        _PARALLEL_DATA = {}
    
    else:
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
        max_mismatch_rate: Maximum allele mismatch rate for IBS equivalence (default 2%)
        verbose: Print progress
    
    Returns:
        New BlockPainting with F1 samples recolored for parsimony
    """
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
    
    total_switches_saved = 0
    total_samples_changed = 0
    
    new_samples = list(block_painting.samples)  # shallow copy
    
    for f1_name in sorted(f1_names):
        if f1_name not in name_to_idx:
            continue
        idx = name_to_idx[f1_name]
        sample_painting = block_painting[idx]
        
        if not sample_painting.chunks:
            continue
        
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
        
        if saved <= 0:
            continue  # No improvement
        
        total_switches_saved += saved
        total_samples_changed += 1
        
        if verbose:
            print(f"  {f1_name}: track1 {orig_switches_0}→{new_switches_0} switches, "
                  f"track2 {orig_switches_1}→{new_switches_1} switches "
                  f"(saved {saved})")
        
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
        max_mismatch_rate: IBS equivalence threshold (default 2%)
        verbose: Print progress
    
    Returns:
        New BlockPainting with offspring founder IDs propagated from parents
    """
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
    
    for gen_num, gen, sample_name in samples_by_gen:
        # Skip F1s — they have no parents, already recolored
        if sample_name not in parent_map:
            continue
        if sample_name not in name_to_idx:
            continue
        
        p1_name, p2_name = parent_map[sample_name]
        if p1_name not in name_to_idx or p2_name not in name_to_idx:
            continue
        
        child_idx = name_to_idx[sample_name]
        p1_idx = name_to_idx[p1_name]
        p2_idx = name_to_idx[p2_name]
        
        child_painting = new_samples[child_idx]
        p1_painting = new_samples[p1_idx]
        p2_painting = new_samples[p2_idx]
        
        if not child_painting.chunks:
            continue
        
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
            new_samples[child_idx] = SamplePainting(child_idx, new_chunks)
            total_replacements += replacements
            total_samples_changed += 1
    
    if verbose:
        print(f"  Propagated to {total_samples_changed} offspring, "
              f"{total_replacements} total segment replacements")
    
    return BlockPainting((block_painting.start_pos, block_painting.end_pos), new_samples)