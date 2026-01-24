"""
paint_samples_testing.py

A robust module for reconstructing diploid haplotypes (painting) from probabilistic 
genotype data using a Tolerance-based Viterbi algorithm.

Key Features:
1.  **Tolerance Viterbi:** Finds ALL valid paths within a log-likelihood margin of the optimal path.
2.  **Topological Deduplication:** Collapses paths that are identical sequences of states.
3.  **Allele-Aware Clustering:** Decomposes diploid paths into haploid tracks and clusters them
    based on sequence identity (checking alleles, not just labels).
4.  **Visualization:** Plots valid paths and the resulting consensus options.
"""

import numpy as np
import math
import pandas as pd
import warnings
from typing import List, Tuple, Dict, NamedTuple, Set, DefaultDict, Counter, Optional, Union
from collections import defaultdict
import copy

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
    """
    Represents a contiguous genomic interval where the diploid state 
    (Founder H1, Founder H2) remains constant.
    """
    start: int
    end: int
    hap1: int
    hap2: int

class ConsensusChunk(NamedTuple):
    """
    Represents an interval in the Consensus painting.
    Tracks the set of possible founders for each track.
    """
    start: int
    end: int
    possible_hap1: frozenset  # Set of possible IDs for track 1
    possible_hap2: frozenset  # Set of possible IDs for track 2

class SamplePainting:
    """
    Holds a single concrete painting (one valid path through the HMM).
    """
    def __init__(self, sample_index: int, chunks: List[PaintedChunk]):
        self.sample_index = sample_index
        self.chunks = chunks 
        self.num_recombinations = max(0, len(self.chunks) - 1)
        
    def __repr__(self):
        return f"<SamplePainting ID {self.sample_index}: {len(self.chunks)} chunks>"

class SampleConsensusPainting:
    """
    Holds the consensus of a cluster of paths.
    """
    def __init__(self, sample_index: int, chunks: List[ConsensusChunk], weight: float = 1.0):
        self.sample_index = sample_index
        self.chunks = chunks
        self.weight = weight # Percentage of paths supporting this consensus
        
    def __repr__(self):
        return f"<SampleConsensus ID {self.sample_index}: {len(self.chunks)} chunks, Weight {self.weight:.2f}>"

# =============================================================================
# 2. NUMBA KERNELS FOR ALLELE COMPARISON
# =============================================================================

@njit(fastmath=True)
def check_allelic_match(dense_haps, positions, h1, h2, start, end, mismatch_threshold=0.01):
    """
    Checks if two haplotypes (h1, h2) are genetically identical in the region [start, end).
    """
    if h1 == h2: return True
    if h1 == -1 or h2 == -1: return True # Wildcard match
    
    n_sites = len(positions)
    # Find start index
    idx_start = -1
    for i in range(n_sites):
        if positions[i] >= start:
            idx_start = i
            break
            
    if idx_start == -1: return True # No SNPs in region
    
    # Check SNPs
    matches = 0
    total = 0
    
    for i in range(idx_start, n_sites):
        if positions[i] >= end: break
        
        a1 = dense_haps[h1, i]
        a2 = dense_haps[h2, i]
        
        if a1 != -1 and a2 != -1:
            total += 1
            if a1 == a2:
                matches += 1
                
    if total == 0: return True
    
    mismatch_rate = 1.0 - (matches / total)
    return mismatch_rate <= mismatch_threshold

# =============================================================================
# 3. CONTAINER & CLUSTERING LOGIC
# =============================================================================

class SampleTolerancePainting:
    """
    Container for the results of a Tolerance Painting run.
    Holds multiple valid paths and provides methods to generate clustered consensus.
    """
    def __init__(self, sample_index: int, paths: List[SamplePainting]):
        self.sample_index = sample_index
        self.paths = paths 
        
    def generate_all_paths(self) -> List[SamplePainting]:
        """Returns the list of valid paths found, sorted by parsimony/score."""
        return self.paths

    def _calculate_haploid_similarity(self, path_a: SamplePainting, path_b: SamplePainting, 
                                      founder_data=None) -> Tuple[float, bool]:
        """
        Calculates the exact base-pair overlap similarity between two diploid paths.
        
        Args:
            founder_data: Optional tuple (dense_haps, positions). 
                          If provided, compares Alleles. If None, compares IDs.
        
        Returns:
            (max_score, should_flip)
        """
        if not path_a.chunks or not path_b.chunks:
            return 0.0, False

        total_len = path_a.chunks[-1].end - path_a.chunks[0].start
        if total_len == 0: return 1.0, False

        def extract_tracks(p: SamplePainting):
            t1, t2 = [], []
            for c in p.chunks:
                t1.append((c.start, c.end, c.hap1))
                t2.append((c.start, c.end, c.hap2))
            return t1, t2

        a1, a2 = extract_tracks(path_a)
        b1, b2 = extract_tracks(path_b)

        dense_haps, positions = founder_data if founder_data else (None, None)

        def score_track_overlap(track_x, track_y):
            score = 0
            idx_x, idx_y = 0, 0
            n_x, n_y = len(track_x), len(track_y)
            
            while idx_x < n_x and idx_y < n_y:
                sx, ex, idx = track_x[idx_x]
                sy, ey, idy = track_y[idx_y]
                
                start = max(sx, sy)
                end = min(ex, ey)
                
                if start < end:
                    match = False
                    if idx == idy:
                        match = True
                    elif dense_haps is not None:
                        # Check allele identity
                        if check_allelic_match(dense_haps, positions, idx, idy, start, end):
                            match = True
                    
                    if match:
                        score += (end - start)
                
                if ex < ey: idx_x += 1
                else: idx_y += 1
            return score

        match_11 = score_track_overlap(a1, b1)
        match_22 = score_track_overlap(a2, b2)
        match_12 = score_track_overlap(a1, b2)
        match_21 = score_track_overlap(a2, b1)
        
        direct_score = (match_11 + match_22) / (2 * total_len)
        cross_score = (match_12 + match_21) / (2 * total_len)
        
        if cross_score > direct_score:
            return cross_score, True
        else:
            return direct_score, False

    def generate_clustered_consensus(self, similarity_threshold=0.999, founder_data=None) -> List[SampleConsensusPainting]:
        """
        Groups valid paths into clusters based on sequence identity.
        
        Args:
            similarity_threshold (float): Fraction of base-pairs that must match (0.0-1.0).
            founder_data (tuple): (dense_matrix, positions) for allele-level comparison.
        """
        if not self.paths: return []
        
        clusters = [] # List of [ (SamplePainting, is_flipped) ]
        
        for p in self.paths:
            assigned = False
            for cluster_data in clusters:
                rep_path, _ = cluster_data[0]
                
                score, should_flip = self._calculate_haploid_similarity(rep_path, p, founder_data)
                
                if score >= similarity_threshold:
                    cluster_data.append((p, should_flip))
                    assigned = True
                    break
            
            if not assigned:
                clusters.append([(p, False)])
                
        results = []
        total_paths = len(self.paths)
        
        for cluster_data in clusters:
            aligned_paths = []
            for p, flip in cluster_data:
                if flip:
                    new_chunks = [PaintedChunk(c.start, c.end, c.hap2, c.hap1) for c in p.chunks]
                    aligned_paths.append(SamplePainting(p.sample_index, new_chunks))
                else:
                    aligned_paths.append(p)
            
            # Note: We do NOT pass founder_data here anymore to avoid collapsing IBS labels in the visualization object
            cons_chunks = self._merge_aligned_paths(aligned_paths)
            
            weight = len(aligned_paths) / total_paths
            results.append(SampleConsensusPainting(self.sample_index, cons_chunks, weight))
            
        return results

    def _merge_aligned_paths(self, paths: List[SamplePainting]) -> List[ConsensusChunk]:
        """
        Merges phase-aligned paths into a consensus.
        Retains ALL candidate IDs in the set (does not collapse IBS).
        """
        breakpoints = set()
        for p in paths:
            for c in p.chunks:
                breakpoints.add(c.start); breakpoints.add(c.end)
        sorted_bp = sorted(list(breakpoints))
        
        consensus_chunks = []
        
        for i in range(len(sorted_bp) - 1):
            start, end = sorted_bp[i], sorted_bp[i+1]
            midpoint = (start + end) // 2
            
            t1_options = set()
            t2_options = set()
            
            for p in paths:
                for c in p.chunks:
                    if c.start <= midpoint < c.end:
                        t1_options.add(c.hap1)
                        t2_options.add(c.hap2)
                        break
            
            # Create Consensus Chunk with FULL sets
            new_chunk = ConsensusChunk(
                start, end, 
                frozenset(t1_options), frozenset(t2_options)
            )
            
            # Merge adjacent identical chunks
            if consensus_chunks:
                prev = consensus_chunks[-1]
                if (prev.possible_hap1 == new_chunk.possible_hap1 and 
                    prev.possible_hap2 == new_chunk.possible_hap2):
                    consensus_chunks[-1] = ConsensusChunk(
                        prev.start, new_chunk.end, 
                        prev.possible_hap1, prev.possible_hap2
                    )
                else:
                    consensus_chunks.append(new_chunk)
            else:
                consensus_chunks.append(new_chunk)
                
        return consensus_chunks

    def generate_consensus(self, founder_data=None) -> SampleConsensusPainting:
        """Helper to generate a single consensus (forcing all paths into one cluster)."""
        res = self.generate_clustered_consensus(similarity_threshold=0.0, founder_data=founder_data)
        return res[0] if res else None

    def get_best_representative_path(self, founder_data=None) -> SamplePainting:
        """
        Returns the Best Path (Path 0), masking uncertain regions.
        If founder_data is provided, checks if uncertain regions are IBS and rescues them.
        """
        if not self.paths: return None
        
        # Get dominant consensus
        cons_list = self.generate_clustered_consensus(similarity_threshold=0.999, founder_data=founder_data)
        if not cons_list: return self.paths[0]
        
        cons_list.sort(key=lambda x: x.weight, reverse=True)
        dominant_cons = cons_list[0]
        
        raw_best = self.paths[0]
        masked_chunks = []
        
        dense_haps, positions = founder_data if founder_data else (None, None)
        
        for c_cons in dominant_cons.chunks:
            # Determine if track is uncertain
            t1_ids = list(c_cons.possible_hap1)
            t2_ids = list(c_cons.possible_hap2)
            
            is_t1_uncertain = len(t1_ids) > 1
            is_t2_uncertain = len(t2_ids) > 1
            
            # IBS Check: Can we rescue uncertainty?
            if is_t1_uncertain and dense_haps is not None:
                ref = t1_ids[0]
                all_match = True
                for other in t1_ids[1:]:
                    if not check_allelic_match(dense_haps, positions, ref, other, c_cons.start, c_cons.end):
                        all_match = False
                        break
                if all_match: is_t1_uncertain = False
                
            if is_t2_uncertain and dense_haps is not None:
                ref = t2_ids[0]
                all_match = True
                for other in t2_ids[1:]:
                    if not check_allelic_match(dense_haps, positions, ref, other, c_cons.start, c_cons.end):
                        all_match = False
                        break
                if all_match: is_t2_uncertain = False
            
            midpoint = (c_cons.start + c_cons.end) // 2
            h1, h2 = -1, -1
            
            # Find raw values
            for c_raw in raw_best.chunks:
                if c_raw.start <= midpoint < c_raw.end:
                    h1, h2 = c_raw.hap1, c_raw.hap2
                    break
            
            # Apply Mask
            if is_t1_uncertain: h1 = -1
            if is_t2_uncertain: h2 = -1
            
            # Merge
            if masked_chunks:
                prev = masked_chunks[-1]
                if prev.hap1 == h1 and prev.hap2 == h2:
                    masked_chunks[-1] = PaintedChunk(prev.start, c_cons.end, h1, h2)
                else:
                    masked_chunks.append(PaintedChunk(c_cons.start, c_cons.end, h1, h2))
            else:
                masked_chunks.append(PaintedChunk(c_cons.start, c_cons.end, h1, h2))
                
        return SamplePainting(self.sample_index, masked_chunks)

class BlockTolerancePainting:
    def __init__(self, block_position_range: Tuple[int, int], samples: List[SampleTolerancePainting]):
        self.start_pos = block_position_range[0]
        self.end_pos = block_position_range[1]
        self.samples = samples

    def __getitem__(self, idx): return self.samples[idx]
    def __len__(self): return len(self.samples)

# =============================================================================
# 4. HELPER: DENSE MATRIX CONVERSION
# =============================================================================

def founder_block_to_dense(block_result):
    positions = block_result.positions
    hap_dict = block_result.haplotypes
    
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

# =============================================================================
# 5. EMISSION CALCULATOR
# =============================================================================

def calculate_batch_emissions(sample_probs_matrix, hap_dict, robustness_epsilon=1e-3):
    hap_keys = sorted(list(hap_dict.keys()))
    num_haps = len(hap_keys)
    num_samples, num_sites, _ = sample_probs_matrix.shape
    
    idx_i, idx_j = np.unravel_index(np.arange(num_haps**2), (num_haps, num_haps))
    state_defs = np.stack([idx_i, idx_j], axis=1).astype(np.int32)
    
    hap_list = [hap_dict[k] for k in hap_keys]
    if not hap_list:
        return np.zeros((num_samples, 0, num_sites)), state_defs, hap_keys
        
    haps_tensor = np.array(hap_list)
    h0 = haps_tensor[:, :, 0]
    h1 = haps_tensor[:, :, 1]
    
    c00 = h0[:, None, :] * h0[None, :, :]
    c11 = h1[:, None, :] * h1[None, :, :]
    c01 = (h0[:, None, :] * h1[None, :, :]) + (h1[:, None, :] * h0[None, :, :])
    
    combos_0 = c00.reshape(num_haps**2, -1)
    combos_1 = c01.reshape(num_haps**2, -1)
    combos_2 = c11.reshape(num_haps**2, -1)
    
    s0 = sample_probs_matrix[:, :, 0][:, np.newaxis, :]
    s1 = sample_probs_matrix[:, :, 1][:, np.newaxis, :]
    s2 = sample_probs_matrix[:, :, 2][:, np.newaxis, :]
    
    c0 = combos_0[np.newaxis, :, :]
    c1 = combos_1[np.newaxis, :, :]
    c2 = combos_2[np.newaxis, :, :]
    
    model_probs = (s0 * c0) + (s1 * c1) + (s2 * c2)
    uniform_prob = 1.0 / 3.0
    final_probs = (model_probs * (1.0 - robustness_epsilon)) + (robustness_epsilon * uniform_prob)
    
    min_prob = 1e-300
    final_probs[final_probs < min_prob] = min_prob
    ll_matrix = np.log(final_probs)
    ll_matrix = np.maximum(ll_matrix, -50.0) 
    
    return ll_matrix, state_defs, hap_keys

# =============================================================================
# 6. TOLERANCE VITERBI KERNELS
# =============================================================================

@njit(parallel=True, fastmath=True)
def run_forward_pass_max_sum(ll_tensor, positions, recomb_rate, state_definitions, n_haps, switch_penalty):
    """Calculates Alpha (Max-Sum) matrix forward."""
    n_samples, K, n_sites = ll_tensor.shape
    alpha = np.full((n_samples, n_sites, K), -np.inf, dtype=np.float32)
    
    if n_haps > 1: log_N_minus_1 = math.log(float(n_haps - 1))
    else: log_N_minus_1 = 0.0

    for s in prange(n_samples):
        for k in range(K):
            alpha[s, 0, k] = ll_tensor[s, k, 0]

        for i in range(1, n_sites):
            dist_bp = positions[i] - positions[i-1]
            if dist_bp < 1: dist_bp = 1
            theta = float(dist_bp) * recomb_rate
            if theta > 0.5: theta = 0.5
            if theta < 1e-15: theta = 1e-15
            
            log_switch = math.log(theta) - switch_penalty
            log_stay = math.log(1.0 - theta)
            cost_0 = 2.0 * log_stay
            cost_1 = log_switch + log_stay - log_N_minus_1
            cost_2 = -1e20 # Forbidden
            
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

@njit(parallel=True, fastmath=True)
def run_backward_pass_max_sum(ll_tensor, positions, recomb_rate, state_definitions, n_haps, switch_penalty):
    """Calculates Beta (Max-Sum) matrix backward."""
    n_samples, K, n_sites = ll_tensor.shape
    beta = np.full((n_samples, n_sites, K), -np.inf, dtype=np.float32)
    
    if n_haps > 1: log_N_minus_1 = math.log(float(n_haps - 1))
    else: log_N_minus_1 = 0.0

    for s in prange(n_samples):
        for k in range(K):
            beta[s, n_sites-1, k] = 0.0

        for i in range(n_sites - 2, -1, -1):
            dist_bp = positions[i+1] - positions[i]
            if dist_bp < 1: dist_bp = 1
            theta = float(dist_bp) * recomb_rate
            if theta > 0.5: theta = 0.5
            if theta < 1e-15: theta = 1e-15
            
            log_switch = math.log(theta) - switch_penalty
            log_stay = math.log(1.0 - theta)
            cost_0 = 2.0 * log_stay
            cost_1 = log_switch + log_stay - log_N_minus_1
            cost_2 = -1e20
            
            for k_curr in range(K): 
                h1_curr = state_definitions[k_curr, 0]
                h2_curr = state_definitions[k_curr, 1]
                best_score = -np.inf
                for k_next in range(K): 
                    h1_next = state_definitions[k_next, 0]
                    h2_next = state_definitions[k_next, 1]
                    dist = 0
                    if h1_curr != h1_next: dist += 1
                    if h2_curr != h2_next: dist += 1
                    
                    if dist == 0: trans = cost_0
                    elif dist == 1: trans = cost_1
                    else: trans = cost_2
                    score = trans + ll_tensor[s, k_next, i+1] + beta[s, i+1, k_next]
                    if score > best_score: best_score = score
                beta[s, i, k_curr] = best_score
    return beta

# =============================================================================
# 7. DEDUPLICATED TRACEBACK
# =============================================================================

class PathState:
    """Helper class for traceback to store partial paths."""
    def __init__(self, current_k, chunks, backward_score):
        self.current_k = current_k
        self.chunks = chunks 
        self.backward_score = backward_score 

    @property
    def topology_key(self):
        """Normalized sequence of genotype sets for deduplication."""
        sig = []
        for c in self.chunks:
            h1, h2 = sorted((c.hap1, c.hap2))
            sig.append((h1, h2))
        return tuple(sig)

def reconstruct_deduplicated_paths(alpha, beta, ll_tensor, positions, recomb_rate, state_definitions, n_haps, switch_penalty, total_margin, hap_keys):
    """
    Backtracks to find all valid paths.
    Applies deduplication and Phase Alignment to minimize visual track flipping.
    """
    n_sites, K = alpha.shape
    global_max = np.max(alpha[n_sites-1])
    min_total_score = global_max - total_margin
    
    active_paths = defaultdict(dict)
    valid_ends = np.where((alpha[n_sites-1] + beta[n_sites-1]) >= min_total_score)[0]
    
    for k in valid_ends:
        h1_idx, h2_idx = state_definitions[k]
        t1, t2 = hap_keys[h1_idx], hap_keys[h2_idx]
        initial_chunk = PaintedChunk(start=int(positions[n_sites-1]), end=int(positions[n_sites-1]), hap1=t1, hap2=t2)
        p = PathState(k, [initial_chunk], 0.0)
        active_paths[k][p.topology_key] = p

    if n_haps > 1: log_N_minus_1 = math.log(float(n_haps - 1))
    else: log_N_minus_1 = 0.0

    for t in range(n_sites - 1, 0, -1):
        prev_t = t - 1
        
        dist_bp = positions[t] - positions[prev_t]
        if dist_bp < 1: dist_bp = 1
        theta = float(dist_bp) * recomb_rate
        if theta > 0.5: theta = 0.5
        if theta < 1e-15: theta = 1e-15
        
        log_switch = math.log(theta) - switch_penalty
        log_stay = math.log(1.0 - theta)
        cost_0 = 2.0 * log_stay
        cost_1 = log_switch + log_stay - log_N_minus_1
        cost_2 = -1e20 
        
        next_active_paths = defaultdict(dict)
        
        for curr_k, topo_dict in active_paths.items():
            curr_h1, curr_h2 = state_definitions[curr_k]
            emission_t = ll_tensor[curr_k, t]
            
            for topo_key, path_obj in topo_dict.items():
                valid_prev_indices = np.where(alpha[prev_t] > -1e10)[0]
                
                for prev_k in valid_prev_indices:
                    prev_h1, prev_h2 = state_definitions[prev_k]
                    
                    dist = 0
                    if curr_h1 != prev_h1: dist += 1
                    if curr_h2 != prev_h2: dist += 1
                    
                    if dist == 0: trans = cost_0
                    elif dist == 1: trans = cost_1
                    else: trans = cost_2
                    
                    new_backward_score = trans + emission_t + path_obj.backward_score
                    if alpha[prev_t, prev_k] + new_backward_score < min_total_score: continue
                        
                    pt1, pt2 = hap_keys[prev_h1], hap_keys[prev_h2]
                    old_chunk = path_obj.chunks[0]
                    new_chunks = list(path_obj.chunks)
                    
                    is_extension = False
                    if (pt1 == old_chunk.hap1 and pt2 == old_chunk.hap2): is_extension = True
                    elif (pt1 == old_chunk.hap2 and pt2 == old_chunk.hap1): is_extension = True
                    
                    if is_extension:
                        updated_chunk = PaintedChunk(start=int(positions[prev_t]), end=old_chunk.end, hap1=old_chunk.hap1, hap2=old_chunk.hap2)
                        new_chunks[0] = updated_chunk
                        new_topo_key = topo_key
                    else:
                        # PHASE ALIGNMENT: Order new chunk to minimize edit distance vs old_chunk
                        d_direct = (0 if pt1 == old_chunk.hap1 else 1) + (0 if pt2 == old_chunk.hap2 else 1)
                        d_cross  = (0 if pt1 == old_chunk.hap2 else 1) + (0 if pt2 == old_chunk.hap1 else 1)
                        
                        final_h1, final_h2 = (pt1, pt2) if d_direct <= d_cross else (pt2, pt1)
                        
                        new_chunk = PaintedChunk(start=int(positions[prev_t]), end=int(positions[t]), hap1=final_h1, hap2=final_h2)
                        new_chunks.insert(0, new_chunk)
                        
                        h1n, h2n = sorted((final_h1, final_h2))
                        new_topo_key = ((h1n, h2n),) + topo_key
                        
                    existing = next_active_paths[prev_k].get(new_topo_key)
                    if existing is None or new_backward_score >= existing.backward_score:
                        next_active_paths[prev_k][new_topo_key] = PathState(prev_k, new_chunks, new_backward_score)
                            
        active_paths = next_active_paths
        if not active_paths: break 
        
    all_paths_flat = []
    seen_topologies = set()
    
    for k, topo_dict in active_paths.items():
        for path_obj in topo_dict.values():
            all_paths_flat.append(path_obj)
            
    # PARSIMONY SORTING
    SHORT_THRESH = 50_000
    def get_parsimony_score(p_obj):
        penalty = 0.0
        for c in p_obj.chunks:
            if (c.end - c.start) < SHORT_THRESH:
                penalty += 5.0
        return p_obj.backward_score - penalty

    all_paths_flat.sort(key=get_parsimony_score, reverse=True)
    
    final_results = []
    for path_obj in all_paths_flat:
        norm_chunks = []
        for c in path_obj.chunks:
            h1, h2 = sorted((c.hap1, c.hap2))
            norm_chunks.append(PaintedChunk(c.start, c.end, h1, h2))
        norm_sig = tuple((c.hap1, c.hap2) for c in norm_chunks)
        
        if norm_sig not in seen_topologies:
            seen_topologies.add(norm_sig)
            # Use original phased chunks (phase-aligned)
            final_results.append(SamplePainting(0, path_obj.chunks)) 
                
    return final_results

# =============================================================================
# 8. MAIN DRIVER
# =============================================================================

def paint_samples_tolerance(block_result, sample_probs_matrix, sample_sites, 
                            recomb_rate=1e-8, switch_penalty=10.0,
                            robustness_epsilon=1e-3, absolute_margin=5.0,
                            margin_per_snp=0.0, batch_size=10):
    """
    Main function to run tolerance painting on a batch of samples.
    """
    positions = block_result.positions
    hap_dict = block_result.haplotypes
    n_sites_block = len(positions)
    total_margin = absolute_margin + (margin_per_snp * n_sites_block)
    
    block_samples_data = analysis_utils.get_sample_data_at_sites_multiple(sample_probs_matrix, sample_sites, positions)
    num_samples = block_samples_data.shape[0]
    all_sample_paintings = []
    
    print(f"Tolerance Painting {num_samples} samples (Margin={total_margin:.2f})...")
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_data = block_samples_data[start_idx:end_idx]
        ll_tensor, state_defs, hap_keys = calculate_batch_emissions(batch_data, hap_dict, robustness_epsilon=robustness_epsilon)
        num_haps = len(hap_keys)
        
        alpha = run_forward_pass_max_sum(ll_tensor, positions, recomb_rate, state_defs, num_haps, float(switch_penalty))
        beta = run_backward_pass_max_sum(ll_tensor, positions, recomb_rate, state_defs, num_haps, float(switch_penalty))
        
        for i in range(end_idx - start_idx):
            global_sample_idx = start_idx + i
            valid_paintings = reconstruct_deduplicated_paths(alpha[i], beta[i], ll_tensor[i], positions, recomb_rate, state_defs, num_haps, switch_penalty, total_margin, hap_keys)
            for p in valid_paintings: p.sample_index = global_sample_idx
            all_sample_paintings.append(SampleTolerancePainting(global_sample_idx, valid_paintings))
            
    range_tuple = (int(positions[0]), int(positions[-1]))
    return BlockTolerancePainting(range_tuple, all_sample_paintings)

# =============================================================================
# 9. VISUALIZATIONS
# =============================================================================

def plot_tolerance_graph_topology(block_painting, sample_idx=0, output_file=None):
    """Plots the topology graph of bifurcating paths."""
    if not HAS_PLOTTING: return
    sample_obj = block_painting[sample_idx]
    paths = sample_obj.generate_all_paths() 
    if not paths: return

    G = nx.DiGraph()
    unique_pairs = set()
    for p in paths:
        for c in p.chunks: unique_pairs.add(tuple(sorted((c.hap1, c.hap2))))
            
    sorted_pairs = sorted(list(unique_pairs))
    pair_to_y = {p: i for i, p in enumerate(sorted_pairs)}
    
    pos = {}
    best_nodes = set()
    
    for p_idx, p in enumerate(paths):
        for i, chunk in enumerate(p.chunks):
            pair = tuple(sorted((chunk.hap1, chunk.hap2)))
            pos_x = (chunk.start + chunk.end) / 2
            node_id = (chunk.start, pair[0], pair[1])
            G.add_node(node_id, label=f"{pair[0]}/{pair[1]}")
            pos[node_id] = (pos_x, pair_to_y[pair])
            if p_idx == 0: best_nodes.add(node_id)
            if i > 0:
                prev_c = p.chunks[i-1]
                prev_p = tuple(sorted((prev_c.hap1, prev_c.hap2)))
                prev_node = (prev_c.start, prev_p[0], prev_p[1])
                G.add_edge(prev_node, node_id)

    fig, ax = plt.subplots(figsize=(15, max(4, len(sorted_pairs))))
    for pair, y in pair_to_y.items():
        ax.axhline(y, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        ax.text(paths[0].chunks[0].start, y, f" {pair[0]}/{pair[1]}", va='center', fontsize=9, fontweight='bold')

    node_colors = ['#ffaaaa' if n in best_nodes else '#aaccff' for n in G.nodes()]
    edge_colors = ['red' if u in best_nodes and v in best_nodes else 'gray' for u, v in G.edges()]
    widths = [2.0 if u in best_nodes and v in best_nodes else 0.5 for u, v in G.edges()]
            
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=widths, ax=ax)
    
    ax.set_title(f"Topology of {len(paths)} Valid Paths", fontsize=14)
    ax.set_xlabel("Genomic Position (bp)")
    ax.set_yticks([])
    if output_file: plt.savefig(output_file, bbox_inches='tight')
    else: plt.show()
    plt.close()

def plot_viable_paintings(block_tolerance, sample_idx=0, max_paths=50, output_file=None):
    """Plots valid paths as horizontal bars, plus the Clustered Consensus bars."""
    if not HAS_PLOTTING: return
    sample_obj = block_tolerance[sample_idx]
    paths = sample_obj.generate_all_paths()
    if not paths: return

    if len(paths) > max_paths: paths = paths[:max_paths]

    unique_haps = set()
    for p in paths:
        for c in p.chunks: unique_haps.add(c.hap1); unique_haps.add(c.hap2)
    sorted_haps = sorted(list(unique_haps))
    hap_to_idx = {h: i for i, h in enumerate(sorted_haps)}
    
    if len(sorted_haps) <= 10: palette = sns.color_palette("tab10", len(sorted_haps))
    else: palette = sns.color_palette("husl", len(sorted_haps))

    row_height = 0.5
    
    # Get Clustered Consensus
    cons_list = sample_obj.generate_clustered_consensus()
    
    num_paths = len(paths)
    num_cons = len(cons_list)
    calc_height = (num_paths * row_height) + (num_cons * row_height) + 2.0
    
    fig, ax = plt.subplots(figsize=(20, calc_height))
    y_height = 0.4 
    
    # 1. Plot Individual Paths
    for i, path_obj in enumerate(paths):
        y_base = i * row_height
        for chunk in path_obj.chunks:
            width = chunk.end - chunk.start
            if width <= 0: continue
            c1 = palette[hap_to_idx[chunk.hap1]]
            rect1 = mpatches.Rectangle((chunk.start, y_base), width, y_height/2, facecolor=c1, edgecolor='none')
            ax.add_patch(rect1)
            c2 = palette[hap_to_idx[chunk.hap2]]
            rect2 = mpatches.Rectangle((chunk.start, y_base + y_height/2), width, y_height/2, facecolor=c2, edgecolor='none')
            ax.add_patch(rect2)
            
    # 2. Plot Consensus Clusters
    y_cons_start = num_paths * row_height + 0.5
    
    for i, cons in enumerate(cons_list):
        y_c = y_cons_start + (i * row_height)
        label_text = f"CONSENSUS {i+1} ({cons.weight*100:.0f}%)"
        ax.text(block_tolerance.start_pos, y_c + y_height/4, label_text, fontsize=9, fontweight='bold', va='center')
        
        for chunk in cons.chunks:
            width = chunk.end - chunk.start
            if width <= 0: continue
            
            def draw_stacked_track(possible_set, y_bottom, total_h):
                candidates = sorted(list(possible_set))
                if not candidates: return
                sub_h = total_h / len(candidates)
                for k, hap_id in enumerate(candidates):
                    color = palette[hap_to_idx[hap_id]]
                    rect = mpatches.Rectangle((chunk.start, y_bottom + k*sub_h), width, sub_h, facecolor=color, edgecolor='none')
                    ax.add_patch(rect)

            draw_stacked_track(chunk.possible_hap1, y_c, y_height/2)
            draw_stacked_track(chunk.possible_hap2, y_c + y_height/2, y_height/2)

    ax.set_xlim(block_tolerance.start_pos, block_tolerance.end_pos)
    ax.set_ylim(0, y_cons_start + (num_cons * row_height) + 0.5)
    
    path_labels = [f"Path {i}" for i in range(len(paths))]
    cons_labels = [""] * num_cons # Text already placed manually
    
    ticks = list(np.arange(len(paths)) * row_height + row_height/2)
    # Add dummy ticks for spacing
    for i in range(num_cons):
        ticks.append(y_cons_start + (i*row_height) + row_height/2)
        path_labels.append("")
        
    ax.set_yticks(ticks)
    ax.set_yticklabels(path_labels, fontsize=8)
    ax.set_xlabel("Genomic Position (bp)")
    
    patches = [mpatches.Patch(color=palette[hap_to_idx[h]], label=f"H{h}") for h in sorted_haps]
    ax.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc='upper left')
    
    plt.tight_layout()
    if output_file: plt.savefig(output_file)
    else: plt.show()
    plt.close()