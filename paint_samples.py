"""
paint_samples.py (BINNED VERSION)

A robust module for reconstructing diploid haplotypes (painting) from probabilistic 
genotype data using a Tolerance-based Viterbi algorithm.

BINNED VERSION: Aggregates SNP-level emissions into bins (~100 SNPs/bin) to reduce
memory usage by ~100x while preserving accuracy.

Key Features:
1.  **Binned Emissions:** Sums per-SNP log-likelihoods into bins for memory efficiency.
2.  **Tolerance Viterbi:** Finds ALL valid paths within a log-likelihood margin.
3.  **Topological Deduplication:** Collapses paths that are identical sequences of states.
4.  **Beam-Capped Traceback:** Prevents combinatorial explosion in ambiguous regions.
5.  **Double-Recomb Discount:** Prefers simultaneous switches to prevent 2^N path explosion.
6.  **Allele-Aware Clustering:** Decomposes diploid paths into haploid tracks and clusters them.
7.  **Visualization:** Plots individual fuzzy paths AND whole-population consensus.
8.  **Parallel Execution:** Uses multiprocessing and tqdm.
9.  **Deterministic Emissions:** Converts probabilistic haplotypes to deterministic via argmax
    to prevent epistemic uncertainty from biasing founder selection.

IMPORTANT: Uses float64 for alpha/beta arrays to prevent precision loss over long chromosomes.

IMPORTANT FIX (v2): Emission calculations now use DETERMINISTIC haplotypes (via argmax).
This fixes the "founder aliasing" bug where epistemic uncertainty in founder haplotypes
(e.g., 50/50 probability at a site) caused systematic bias toward uncertain founders.
The probabilistic approach gave uncertain founders "moderate" scores everywhere, making
them appear better than founders who were certain but had occasional mismatches. By
converting to deterministic alleles, we treat "I don't know" as a coin flip (unbiased
noise) rather than as evidence (systematic bias).
"""

import numpy as np
import math
import pandas as pd
import warnings
from typing import List, Tuple, Dict, NamedTuple, Set, DefaultDict, Counter, Optional, Union
from collections import defaultdict
import copy
from multiprocess import Pool
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
    from numba import njit, prange, set_num_threads
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("WARNING: Numba not found. Painting will be slow.")
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator
    prange = range
    def set_num_threads(n): pass

# =============================================================================
# 1. DATA STRUCTURES
# =============================================================================

class PaintedChunk(NamedTuple):
    start: int
    end: int
    hap1: int
    hap2: int

class ConsensusChunk(NamedTuple):
    start: int
    end: int
    possible_hap1: frozenset
    possible_hap2: frozenset

class PhasedSegment(NamedTuple):
    start: int
    end: int
    founder_id: int

class ConsensusSegment(NamedTuple):
    start: int
    end: int
    possible_ids: frozenset

class SamplePainting:
    def __init__(self, sample_index: int, chunks: List[PaintedChunk]):
        self.sample_index = sample_index
        self.chunks = chunks 
        self.num_recombinations = max(0, len(self.chunks) - 1)
        
        self.hap1_phased = self._extract_phased_track(track_idx=0)
        self.hap2_phased = self._extract_phased_track(track_idx=1)

    def _extract_phased_track(self, track_idx: int) -> List[PhasedSegment]:
        segments = []
        if not self.chunks: return segments
        
        curr_start = self.chunks[0].start
        curr_end = self.chunks[0].end
        curr_id = self.chunks[0].hap1 if track_idx == 0 else self.chunks[0].hap2
        
        for i in range(1, len(self.chunks)):
            c = self.chunks[i]
            next_id = c.hap1 if track_idx == 0 else c.hap2
            
            if next_id == curr_id:
                curr_end = c.end
            else:
                segments.append(PhasedSegment(curr_start, curr_end, curr_id))
                curr_start = c.start
                curr_end = c.end
                curr_id = next_id
                
        segments.append(PhasedSegment(curr_start, curr_end, curr_id))
        return segments

    def __repr__(self):
        return f"<SamplePainting ID {self.sample_index}: {len(self.chunks)} chunks>"

    def __iter__(self):
        return iter(self.chunks)
    
    def __getitem__(self, idx):
        return self.chunks[idx]

class SampleConsensusPainting:
    def __init__(self, sample_index: int, chunks: List[ConsensusChunk], weight: float = 1.0,
                 representative_path: Optional['SamplePainting'] = None):
        self.sample_index = sample_index
        self.chunks = chunks
        self.weight = weight
        self.representative_path = representative_path
        
        self.hap1_consensus = self._extract_consensus_track(track_idx=0)
        self.hap2_consensus = self._extract_consensus_track(track_idx=1)

    def _extract_consensus_track(self, track_idx: int) -> List[ConsensusSegment]:
        segments = []
        if not self.chunks: return segments
        
        curr_start = self.chunks[0].start
        curr_end = self.chunks[0].end
        curr_set = self.chunks[0].possible_hap1 if track_idx == 0 else self.chunks[0].possible_hap2
        
        for i in range(1, len(self.chunks)):
            c = self.chunks[i]
            next_set = c.possible_hap1 if track_idx == 0 else c.possible_hap2
            
            if next_set == curr_set:
                curr_end = c.end
            else:
                segments.append(ConsensusSegment(curr_start, curr_end, curr_set))
                curr_start = c.start
                curr_end = c.end
                curr_set = next_set
                
        segments.append(ConsensusSegment(curr_start, curr_end, curr_set))
        return segments

    def __repr__(self):
        return f"<SampleConsensus ID {self.sample_index}: {len(self.chunks)} chunks, Weight {self.weight:.2f}>"

# =============================================================================
# 2. NUMBA KERNELS
# =============================================================================

@njit(fastmath=True)
def check_allelic_match(dense_haps, positions, h1, h2, start, end, mismatch_threshold=0.01):
    if h1 == h2: return True
    if h1 == -1 or h2 == -1: return True
    
    n_sites = len(positions)
    idx_start = -1
    for i in range(n_sites):
        if positions[i] >= start:
            idx_start = i
            break
            
    if idx_start == -1: return True
    
    matches = 0
    total = 0
    
    for i in range(idx_start, n_sites):
        if positions[i] >= end: break
        
        a1 = dense_haps[h1, i]
        a2 = dense_haps[h2, i]
        
        if a1 != -1 and a2 != -1:
            total += 1
            if a1 == a2: matches += 1
                
    if total == 0: return True
    
    mismatch_rate = 1.0 - (matches / total)
    return mismatch_rate <= mismatch_threshold

# =============================================================================
# 3. CONTAINER & CLUSTERING LOGIC
# =============================================================================

class SampleTolerancePainting:
    def __init__(self, sample_index: int, paths: List[SamplePainting]):
        self.sample_index = sample_index
        self.paths = paths 
        self.consensus_list = [] 
        
    def generate_all_paths(self) -> List[SamplePainting]:
        return self.paths

    def _calculate_haploid_similarity(self, path_a: SamplePainting, path_b: SamplePainting, 
                                      founder_data=None) -> Tuple[float, bool]:
        if not path_a.chunks or not path_b.chunks: return 0.0, False

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
                    if idx == idy: match = True
                    elif dense_haps is not None:
                        if check_allelic_match(dense_haps, positions, idx, idy, start, end):
                            match = True
                    if match: score += (end - start)
                
                if ex < ey: idx_x += 1
                else: idx_y += 1
            return score

        match_11 = score_track_overlap(a1, b1)
        match_22 = score_track_overlap(a2, b2)
        match_12 = score_track_overlap(a1, b2)
        match_21 = score_track_overlap(a2, b1)
        
        direct_score = (match_11 + match_22) / (2 * total_len)
        cross_score = (match_12 + match_21) / (2 * total_len)
        
        if cross_score > direct_score: return cross_score, True
        else: return direct_score, False

    def generate_clustered_consensus(self, similarity_threshold=0.999, founder_data=None,
                                      collapse_ibs=False) -> List[SampleConsensusPainting]:
        if not self.paths: return []
        
        clusters = []
        for p in self.paths:
            assigned = False
            for cluster_data in clusters:
                rep_path, _ = cluster_data[0]
                score, should_flip = self._calculate_haploid_similarity(rep_path, p, founder_data)
                
                if score >= similarity_threshold:
                    cluster_data.append((p, should_flip))
                    assigned = True
                    break
            
            if not assigned: clusters.append([(p, False)])
                
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
            
            cons_chunks = self._merge_aligned_paths(aligned_paths, founder_data=founder_data,
                                                     collapse_ibs=collapse_ibs)
            weight = len(aligned_paths) / total_paths
            
            representative_path = aligned_paths[0] if aligned_paths else None
            results.append(SampleConsensusPainting(self.sample_index, cons_chunks, weight, 
                                                    representative_path=representative_path))
            
        self.consensus_list = results
        return results

    def _merge_aligned_paths(self, paths: List[SamplePainting], founder_data=None,
                              collapse_ibs=False) -> List[ConsensusChunk]:
        breakpoints = set()
        for p in paths:
            for c in p.chunks:
                breakpoints.add(c.start); breakpoints.add(c.end)
        sorted_bp = sorted(list(breakpoints))
        
        consensus_chunks = []
        dense_haps, positions = founder_data if founder_data else (None, None)
        
        for i in range(len(sorted_bp) - 1):
            start, end = sorted_bp[i], sorted_bp[i+1]
            midpoint = (start + end) // 2
            t1_options = set()
            t2_options = set()
            
            for p in paths:
                for c in p.chunks:
                    if c.start <= midpoint < c.end:
                        t1_options.add(c.hap1); t2_options.add(c.hap2)
                        break
            
            if collapse_ibs and dense_haps is not None:
                def collapse_set(id_set):
                    if len(id_set) <= 1: return id_set
                    ids = list(id_set)
                    ref = ids[0]
                    all_match = True
                    for other in ids[1:]:
                        if not check_allelic_match(dense_haps, positions, ref, other, start, end):
                            all_match = False
                            break
                    if all_match: return frozenset([ref])
                    return id_set
                
                t1_options = collapse_set(t1_options)
                t2_options = collapse_set(t2_options)

            new_chunk = ConsensusChunk(start, end, frozenset(t1_options), frozenset(t2_options))
            
            if consensus_chunks:
                prev = consensus_chunks[-1]
                if (prev.possible_hap1 == new_chunk.possible_hap1 and 
                    prev.possible_hap2 == new_chunk.possible_hap2):
                    consensus_chunks[-1] = ConsensusChunk(prev.start, new_chunk.end, prev.possible_hap1, prev.possible_hap2)
                else:
                    consensus_chunks.append(new_chunk)
            else:
                consensus_chunks.append(new_chunk)
                
        return consensus_chunks

    def get_best_representative_path(self, founder_data=None) -> SamplePainting:
        if not self.paths: return SamplePainting(self.sample_index, [])
        
        if self.consensus_list: cons_list = self.consensus_list
        else: cons_list = self.generate_clustered_consensus(similarity_threshold=0.999, founder_data=founder_data)
            
        if not cons_list: return self.paths[0]
        
        dominant_cons = max(cons_list, key=lambda x: x.weight)
        
        raw_best = dominant_cons.representative_path
        if raw_best is None:
            raw_best = self.paths[0]
        
        masked_chunks = []
        dense_haps, positions = founder_data if founder_data else (None, None)
        
        for c_cons in dominant_cons.chunks:
            t1_ids = list(c_cons.possible_hap1)
            t2_ids = list(c_cons.possible_hap2)
            is_t1_uncertain = len(t1_ids) > 1
            is_t2_uncertain = len(t2_ids) > 1
            
            if is_t1_uncertain and dense_haps is not None:
                ref = t1_ids[0]; all_match = True
                for other in t1_ids[1:]:
                    if not check_allelic_match(dense_haps, positions, ref, other, c_cons.start, c_cons.end):
                        all_match = False; break
                if all_match: is_t1_uncertain = False
                
            if is_t2_uncertain and dense_haps is not None:
                ref = t2_ids[0]; all_match = True
                for other in t2_ids[1:]:
                    if not check_allelic_match(dense_haps, positions, ref, other, c_cons.start, c_cons.end):
                        all_match = False; break
                if all_match: is_t2_uncertain = False
            
            midpoint = (c_cons.start + c_cons.end) // 2
            h1, h2 = -1, -1
            
            for c_raw in raw_best.chunks:
                if c_raw.start <= midpoint < c_raw.end:
                    h1, h2 = c_raw.hap1, c_raw.hap2; break
            
            if is_t1_uncertain: h1 = -1
            if is_t2_uncertain: h2 = -1
            
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

@njit(parallel=True, fastmath=True)
def run_backward_pass_max_sum_binned(ll_tensor, bin_centers, recomb_rate, state_definitions, 
                                      n_haps, switch_penalty, double_recomb_factor=1.5):
    """
    Backward pass of max-sum Viterbi algorithm on BINNED data.
    """
    n_samples, K, n_bins = ll_tensor.shape
    
    beta = np.full((n_samples, n_bins, K), -np.inf, dtype=np.float64)
    
    if n_haps > 1: 
        log_N_minus_1 = math.log(float(n_haps - 1))
    else: 
        log_N_minus_1 = 0.0

    for s in prange(n_samples):
        for k in range(K): 
            beta[s, n_bins-1, k] = 0.0

        for i in range(n_bins - 2, -1, -1):
            dist_bp = bin_centers[i+1] - bin_centers[i]
            if dist_bp < 1: dist_bp = 1
            theta = float(dist_bp) * recomb_rate
            if theta > 0.5: theta = 0.5
            if theta < 1e-15: theta = 1e-15
            
            log_switch = math.log(theta) - switch_penalty
            log_stay = math.log(1.0 - theta)
            
            cost_0 = 2.0 * log_stay
            cost_1 = log_switch + log_stay - log_N_minus_1
            cost_2 = double_recomb_factor * log_switch - 2.0 * log_N_minus_1
            
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
# 7. PRE-COMPUTATION KERNEL (BINNED VERSION)
# =============================================================================

@njit(parallel=True, fastmath=True)
def precompute_valid_transitions_binned(alpha, beta, ll_tensor, bin_centers, recomb_rate, 
                                         state_definitions, n_haps, switch_penalty, 
                                         min_total_score, double_recomb_factor=1.5):
    """
    Pre-compute which transitions are valid (within margin of optimal) - BINNED version.
    """
    n_bins, K = alpha.shape
    valid_edges = np.zeros((n_bins, K, K), dtype=np.bool_)
    
    if n_haps > 1: 
        log_N_minus_1 = math.log(float(n_haps - 1))
    else: 
        log_N_minus_1 = 0.0
    
    for t in prange(1, n_bins):
        dist_bp = bin_centers[t] - bin_centers[t-1]
        if dist_bp < 1: dist_bp = 1
        theta = float(dist_bp) * recomb_rate
        if theta > 0.5: theta = 0.5
        if theta < 1e-15: theta = 1e-15
        
        log_switch = math.log(theta) - switch_penalty
        log_stay = math.log(1.0 - theta)
        
        cost_0 = 2.0 * log_stay
        cost_1 = log_switch + log_stay - log_N_minus_1
        cost_2 = double_recomb_factor * log_switch - 2.0 * log_N_minus_1
        
        for k_curr in range(K):
            if (alpha[t, k_curr] + beta[t, k_curr]) < min_total_score: 
                continue
                
            h1_curr = state_definitions[k_curr, 0]
            h2_curr = state_definitions[k_curr, 1]
            
            for k_prev in range(K):
                if alpha[t-1, k_prev] == -np.inf: 
                    continue
                    
                h1_prev = state_definitions[k_prev, 0]
                h2_prev = state_definitions[k_prev, 1]
                
                dist = 0
                if h1_curr != h1_prev: dist += 1
                if h2_curr != h2_prev: dist += 1
                
                if dist == 0: trans = cost_0
                elif dist == 1: trans = cost_1
                else: trans = cost_2
                
                if trans > -1e19:
                    score = alpha[t-1, k_prev] + trans + ll_tensor[k_curr, t] + beta[t, k_curr]
                    if score >= min_total_score:
                        valid_edges[t, k_prev, k_curr] = True
                        
    return valid_edges

# =============================================================================
# 8. DEDUPLICATED TRACEBACK (BINNED VERSION)
# =============================================================================

class PathState:
    def __init__(self, current_k, chunks, backward_score):
        self.current_k = current_k
        self.chunks = chunks 
        self.backward_score = backward_score 

    @property
    def topology_key(self):
        sig = []
        for c in self.chunks:
            h1, h2 = sorted((c.hap1, c.hap2))
            sig.append((h1, h2))
        return tuple(sig)

def reconstruct_single_best_path_binned(alpha, beta, ll_tensor, bin_centers, bin_edges,
                                         recomb_rate, state_definitions, n_haps, 
                                         switch_penalty, hap_keys, double_recomb_factor=1.5):
    """Fallback: reconstruct the single best path via standard Viterbi traceback - BINNED."""
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

def reconstruct_deduplicated_paths_binned(alpha, beta, ll_tensor, bin_centers, bin_edges,
                                           recomb_rate, state_definitions, n_haps, 
                                           switch_penalty, total_margin, hap_keys, 
                                           max_active_paths=2000, double_recomb_factor=1.5):
    """
    Reconstruct all paths within margin of optimal using beam-capped traceback - BINNED.
    """
    n_bins, K = alpha.shape
    global_max = np.max(alpha[n_bins-1])
    min_total_score = global_max - total_margin - 1e-5
    
    valid_edges = precompute_valid_transitions_binned(
        alpha, beta, ll_tensor, bin_centers, recomb_rate, 
        state_definitions, n_haps, switch_penalty, min_total_score,
        double_recomb_factor
    )
    
    active_paths = defaultdict(dict)
    valid_ends = np.where((alpha[n_bins-1] + beta[n_bins-1]) >= min_total_score)[0]
    
    for k in valid_ends:
        h1_idx, h2_idx = state_definitions[k]
        t1, t2 = hap_keys[h1_idx], hap_keys[h2_idx]
        initial_chunk = PaintedChunk(start=int(bin_edges[n_bins-1]), end=int(bin_edges[n_bins]), 
                                     hap1=t1, hap2=t2)
        p = PathState(k, [initial_chunk], 0.0)
        active_paths[k][p.topology_key] = p

    if n_haps > 1: 
        log_N_minus_1 = math.log(float(n_haps - 1))
    else: 
        log_N_minus_1 = 0.0

    for t in range(n_bins - 1, 0, -1):
        prev_t = t - 1
        
        rows, cols = np.where(valid_edges[t])
        valid_prev_map = defaultdict(list)
        for r, c in zip(rows, cols):
            valid_prev_map[c].append(r)
            
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
        
        next_active_paths = defaultdict(dict)
        
        for curr_k, topo_dict in active_paths.items():
            valid_parents = valid_prev_map.get(curr_k, [])
            if not valid_parents: continue
            
            curr_h1, curr_h2 = state_definitions[curr_k]
            emission_t = ll_tensor[curr_k, t]
            
            for topo_key, path_obj in topo_dict.items():
                for prev_k in valid_parents:
                    prev_h1, prev_h2 = state_definitions[prev_k]
                    dist = 0
                    if curr_h1 != prev_h1: dist += 1
                    if curr_h2 != prev_h2: dist += 1
                    
                    if dist == 0: trans = cost_0
                    elif dist == 1: trans = cost_1
                    else: trans = cost_2
                    
                    new_backward_score = trans + emission_t + path_obj.backward_score
                    
                    if (alpha[prev_t, prev_k] + new_backward_score) < min_total_score:
                        continue

                    pt1, pt2 = hap_keys[prev_h1], hap_keys[prev_h2]
                    old_chunk = path_obj.chunks[0]
                    new_chunks = list(path_obj.chunks)
                    
                    is_extension = False
                    if (pt1 == old_chunk.hap1 and pt2 == old_chunk.hap2): is_extension = True
                    elif (pt1 == old_chunk.hap2 and pt2 == old_chunk.hap1): is_extension = True
                    
                    if is_extension:
                        updated_chunk = PaintedChunk(start=int(bin_edges[prev_t]), end=old_chunk.end, 
                                                     hap1=old_chunk.hap1, hap2=old_chunk.hap2)
                        new_chunks[0] = updated_chunk
                        new_topo_key = topo_key
                    else:
                        d_direct = (0 if pt1 == old_chunk.hap1 else 1) + (0 if pt2 == old_chunk.hap2 else 1)
                        d_cross = (0 if pt1 == old_chunk.hap2 else 1) + (0 if pt2 == old_chunk.hap1 else 1)
                        final_h1, final_h2 = (pt1, pt2) if d_direct <= d_cross else (pt2, pt1)
                        
                        new_chunk = PaintedChunk(start=int(bin_edges[prev_t]), end=int(bin_edges[t]), 
                                                 hap1=final_h1, hap2=final_h2)
                        new_chunks.insert(0, new_chunk)
                        h1n, h2n = sorted((final_h1, final_h2))
                        new_topo_key = ((h1n, h2n),) + topo_key
                        
                    existing = next_active_paths[prev_k].get(new_topo_key)
                    if existing is None or new_backward_score >= existing.backward_score:
                        next_active_paths[prev_k][new_topo_key] = PathState(prev_k, new_chunks, new_backward_score)
        
        total_paths = sum(len(d) for d in next_active_paths.values())
        if total_paths > max_active_paths:
            all_entries = []
            for k, topo_dict in next_active_paths.items():
                for key, p_obj in topo_dict.items():
                    total_s = alpha[prev_t, k] + p_obj.backward_score
                    all_entries.append((total_s, k, key))
            
            all_entries.sort(key=lambda x: x[0], reverse=True)
            keep_entries = all_entries[:max_active_paths]
            
            pruned_active = defaultdict(dict)
            for _, k, key in keep_entries:
                pruned_active[k][key] = next_active_paths[k][key]
            next_active_paths = pruned_active

        active_paths = next_active_paths
        if not active_paths: break 
        
    all_paths_flat = []
    seen_topologies = set()
    
    for k, topo_dict in active_paths.items():
        for path_obj in topo_dict.values():
            all_paths_flat.append(path_obj)
            
    if not all_paths_flat:
        return reconstruct_single_best_path_binned(alpha, beta, ll_tensor, bin_centers, bin_edges,
                                                    recomb_rate, state_definitions, n_haps, 
                                                    switch_penalty, hap_keys, double_recomb_factor)

    SHORT_THRESH = 50_000
    def get_parsimony_score(p_obj):
        penalty = 0.0
        for c in p_obj.chunks:
            if (c.end - c.start) < SHORT_THRESH: penalty += 5.0
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
            final_results.append(SamplePainting(0, path_obj.chunks)) 
                
    return final_results

# =============================================================================
# 9. MULTIPROCESSING DRIVER (BINNED VERSION)
# =============================================================================

def _worker_paint_tolerance_batch_binned(args):
    """Worker function for parallel painting - BINNED version."""
    indices, sample_probs_slice, block_result, params = args
    positions = np.array(block_result.positions, dtype=np.int64)
    hap_dict = block_result.haplotypes
    recomb_rate = params['recomb_rate']
    switch_penalty = params['switch_penalty']
    robustness_epsilon = params['robustness_epsilon']
    total_margin = params['total_margin']
    founder_data = params['founder_data']
    max_active_paths = params.get('max_active_paths', 2000)
    double_recomb_factor = params.get('double_recomb_factor', 1.5)
    snps_per_bin = params.get('snps_per_bin', 100)
    
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
            sample_obj = SampleTolerancePainting(global_idx, [])
            results.append(sample_obj)
        return results
    
    if HAS_NUMBA: 
        set_num_threads(1)
        
    # Run forward/backward on BINNED data
    alpha = run_forward_pass_max_sum_binned(ll_tensor, bin_centers, recomb_rate, state_defs, 
                                             num_haps, float(switch_penalty), double_recomb_factor)
    beta = run_backward_pass_max_sum_binned(ll_tensor, bin_centers, recomb_rate, state_defs, 
                                             num_haps, float(switch_penalty), double_recomb_factor)
    
    results = []
    for i, global_idx in enumerate(indices):
        valid_paintings = reconstruct_deduplicated_paths_binned(
            alpha[i], beta[i], ll_tensor[i], bin_centers, bin_edges,
            recomb_rate, state_defs, num_haps, switch_penalty, total_margin, hap_keys,
            max_active_paths=max_active_paths,
            double_recomb_factor=double_recomb_factor
        )
        
        for p in valid_paintings: 
            p.sample_index = global_idx
            
        sample_obj = SampleTolerancePainting(global_idx, valid_paintings)
        sample_obj.consensus_list = sample_obj.generate_clustered_consensus(founder_data=founder_data)
        results.append(sample_obj)
        
    return results

def paint_samples_tolerance(block_result, sample_probs_matrix, sample_sites, 
                            recomb_rate=1e-8, switch_penalty=10.0,
                            robustness_epsilon=1e-2, absolute_margin=5.0,
                            margin_per_snp=0.0, batch_size=10, num_processes=16,
                            max_active_paths=2000, double_recomb_factor=1.5,
                            snps_per_bin=100):
    """
    Paint samples using tolerance Viterbi to find all paths within margin of optimal.
    
    BINNED VERSION: Aggregates SNP-level emissions into bins for ~100x memory reduction.
    
    Args:
        block_result: BlockResult with positions and haplotypes dict
        sample_probs_matrix: (n_samples, n_sites, 3) genotype probabilities
        sample_sites: Array of site positions in sample_probs_matrix
        recomb_rate: Per-bp recombination rate
        switch_penalty: Additional penalty for haplotype switches
        robustness_epsilon: Numerical stability term
        absolute_margin: Log-likelihood margin for accepting paths
        margin_per_snp: Additional margin per SNP (usually 0)
        batch_size: Samples per worker batch
        num_processes: Number of parallel workers
        max_active_paths: Beam width for traceback
        double_recomb_factor: Multiplier for double-switch cost (default 1.5)
        snps_per_bin: Number of SNPs to aggregate per bin (default 100)
        
    Returns:
        BlockTolerancePainting with all samples' tolerance results
    """
    positions = block_result.positions
    n_sites_block = len(positions)
    
    # Margin is now per-bin, not per-SNP
    n_bins = max(1, n_sites_block // snps_per_bin)
    total_margin = absolute_margin + (margin_per_snp * n_bins)
    
    dense_haps, dense_pos = founder_block_to_dense(block_result)
    founder_data = (dense_haps, dense_pos)
    
    block_samples_data = analysis_utils.get_sample_data_at_sites_multiple(sample_probs_matrix, sample_sites, positions)
    num_samples = block_samples_data.shape[0]
    
    print(f"Tolerance Painting (BINNED) {num_samples} samples ({n_sites_block} SNPs → {n_bins} bins, "
          f"Margin={total_margin:.2f}, Cap={max_active_paths}) using {num_processes} workers...")
    
    tasks = []
    params = {
        'recomb_rate': recomb_rate,
        'switch_penalty': switch_penalty,
        'robustness_epsilon': robustness_epsilon,
        'total_margin': total_margin,
        'founder_data': founder_data,
        'max_active_paths': max_active_paths,
        'double_recomb_factor': double_recomb_factor,
        'snps_per_bin': snps_per_bin,
    }
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_slice = block_samples_data[start_idx:end_idx]
        indices = list(range(start_idx, end_idx))
        tasks.append((indices, batch_slice, block_result, params))
        
    all_sample_paintings = []
    with Pool(num_processes) as pool:
        for batch_result in tqdm(pool.imap(_worker_paint_tolerance_batch_binned, tasks), total=len(tasks)):
            all_sample_paintings.extend(batch_result)
            
    all_sample_paintings.sort(key=lambda x: x.sample_index)
    range_tuple = (int(positions[0]), int(positions[-1]))
    return BlockTolerancePainting(range_tuple, all_sample_paintings)

# =============================================================================
# 10. VISUALIZATIONS (unchanged)
# =============================================================================

def plot_tolerance_graph_topology(block_painting, sample_idx=0, output_file=None):
    """Plot the topology of valid paths as a graph."""
    if not HAS_PLOTTING: return
    sample_obj = block_painting[sample_idx]
    paths = sample_obj.generate_all_paths() 
    if not paths: return

    G = nx.DiGraph()
    unique_pairs = set()
    for p in paths:
        for c in p.chunks: 
            unique_pairs.add(tuple(sorted((c.hap1, c.hap2))))
            
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
    """Plot all viable paintings for a sample with consensus clusters."""
    if not HAS_PLOTTING: return
    sample_obj = block_tolerance[sample_idx]
    paths = sample_obj.generate_all_paths()
    if not paths: return

    if len(paths) > max_paths: paths = paths[:max_paths]

    unique_haps = set()
    for p in paths:
        for c in p.chunks: 
            unique_haps.add(c.hap1)
            unique_haps.add(c.hap2)
    sorted_haps = sorted(list(unique_haps))
    hap_to_idx = {h: i for i, h in enumerate(sorted_haps)}
    
    if len(sorted_haps) <= 10: palette = sns.color_palette("tab10", len(sorted_haps))
    else: palette = sns.color_palette("husl", len(sorted_haps))

    row_height = 0.5
    cons_list = sample_obj.consensus_list
    
    num_paths = len(paths)
    num_cons = len(cons_list)
    calc_height = (num_paths * row_height) + (num_cons * row_height) + 2.0
    
    fig, ax = plt.subplots(figsize=(20, calc_height))
    y_height = 0.4 
    
    # Draw individual paths
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
            
    # Draw consensus clusters
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
    ticks = list(np.arange(len(paths)) * row_height + row_height/2)
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