import numpy as np
import math
import pandas as pd
import warnings
from typing import List, Tuple, Dict, NamedTuple

import analysis_utils

# Suppress warnings
warnings.filterwarnings("ignore")
np.seterr(divide='ignore', invalid='ignore')

# Constants
DEFAULT_ROBUSTNESS_EPSILON = 1e-2

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
# 1. RESULT CLASSES
# =============================================================================

class PaintedChunk(NamedTuple):
    """Represents a contiguous segment of a specific haplotype pair."""
    start: int
    end: int
    hap1: int  # Haplotype Key (e.g. 0)
    hap2: int  # Haplotype Key (e.g. 2)
    state_idx: int # Internal Viterbi state index

class SamplePainting:
    """
    Holds the painting results for a single sample.
    """
    def __init__(self, sample_index: int, chunks: List[PaintedChunk]):
        self.sample_index = sample_index
        self.chunks = chunks
        self.num_recombinations = len(chunks) - 1

    def __repr__(self):
        return f"<SamplePainting: ID {self.sample_index}, {len(self.chunks)} chunks>"

    def __iter__(self):
        return iter(self.chunks)
    
    def __getitem__(self, idx):
        return self.chunks[idx]

    def to_dict_list(self, sample_name=None):
        """Helper for dataframe conversion."""
        name = sample_name if sample_name else f"Sample_{self.sample_index}"
        rows = []
        for c in self.chunks:
            rows.append({
                'Sample': name,
                'Start': c.start,
                'End': c.end,
                'Hap1': c.hap1,
                'Hap2': c.hap2,
                'Length': c.end - c.start + 1
            })
        return rows

class BlockPainting:
    """
    The main container returned by paint_samples_in_block.
    Holds the painting results for ALL samples in a specific genomic block.
    """
    def __init__(self, block_position_range: Tuple[int, int], samples: List[SamplePainting]):
        self.start_pos = block_position_range[0]
        self.end_pos = block_position_range[1]
        self.samples = samples
        self.num_samples = len(samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Access a specific SamplePainting by index."""
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)

    def __repr__(self):
        return f"<BlockPainting: {self.num_samples} samples, Range {self.start_pos}-{self.end_pos}>"

    def get_recombinant_samples(self) -> List[SamplePainting]:
        """Returns a list of samples that have at least one internal recombination."""
        return [s for s in self.samples if s.num_recombinations > 0]

    def to_dataframe(self, sample_names=None) -> pd.DataFrame:
        """
        Converts the entire block painting to a Pandas DataFrame.
        """
        all_rows = []
        for i, sample in enumerate(self.samples):
            name = sample_names[i] if sample_names else None
            all_rows.extend(sample.to_dict_list(name))
        return pd.DataFrame(all_rows)

# =============================================================================
# 2. VECTORIZED VITERBI KERNEL
# =============================================================================

@njit(parallel=True, fastmath=False)
def viterbi_batch_solver(ll_tensor, positions, recomb_rate, state_definitions, n_haps):
    """
    Solves the Viterbi path for N samples simultaneously using parallel threads.
    """
    n_samples, K, n_sites = ll_tensor.shape
    final_paths = np.zeros((n_samples, n_sites), dtype=np.int32)
    
    # --- BURST PARAMETERS ---
    GAP_OPEN = -10.0 
    GAP_EXTEND = 0.0 
    UNIFORM_LOG_PROB = -1.0986 # ln(1/3)
    BURST_STEP = UNIFORM_LOG_PROB + GAP_EXTEND
    
    if n_haps > 1:
        log_N_minus_1 = math.log(float(n_haps - 1))
    else:
        log_N_minus_1 = 0.0

    # Parallel Loop over Samples
    for s in prange(n_samples):
        # Local DP Tables
        backpointers = np.zeros((n_sites, K), dtype=np.int32)
        curr_norm = np.empty(K, dtype=np.float64)
        curr_burst = np.empty(K, dtype=np.float64)
        next_norm = np.empty(K, dtype=np.float64)
        next_burst = np.empty(K, dtype=np.float64)
        current_scores = np.empty(K, dtype=np.float64)

        # Initialization
        for k in range(K):
            curr_norm[k] = ll_tensor[s, k, 0]
            curr_burst[k] = GAP_OPEN + UNIFORM_LOG_PROB
            current_scores[k] = max(curr_norm[k], curr_burst[k])

        # Forward Pass
        for i in range(1, n_sites):
            dist_bp = positions[i] - positions[i-1]
            if dist_bp < 1: dist_bp = 1
            theta = float(dist_bp) * recomb_rate
            if theta > 0.5: theta = 0.5
            
            min_prob = 1e-15
            if theta < min_prob:
                log_switch = -1e20
                log_stay = 0.0
            else:
                log_switch = math.log(theta)
                log_stay = math.log(1.0 - theta)
            
            cost_0 = 2.0 * log_stay
            cost_1 = log_switch + log_stay - log_N_minus_1
            cost_2 = 2.0 * log_switch - 2.0 * log_N_minus_1
            
            for k_curr in range(K):
                h1_curr = state_definitions[k_curr, 0]
                h2_curr = state_definitions[k_curr, 1]
                
                best_prev_k = -1
                best_score_to_norm = -np.inf
                
                for k_prev in range(K):
                    h1_prev = state_definitions[k_prev, 0]
                    h2_prev = state_definitions[k_prev, 1]
                    
                    dist = 0
                    if h1_curr != h1_prev: dist += 1
                    if h2_curr != h2_prev: dist += 1
                    
                    trans_cost = cost_2
                    if dist == 0: trans_cost = cost_0
                    elif dist == 1: trans_cost = cost_1
                    
                    prev_score = current_scores[k_prev] 
                    score = prev_score + trans_cost
                    
                    if score > best_score_to_norm:
                        best_score_to_norm = score
                        best_prev_k = k_prev
                
                backpointers[i, k_curr] = best_prev_k
                
                extend = curr_burst[k_curr] + BURST_STEP
                open_burst = best_score_to_norm + GAP_OPEN + BURST_STEP
                next_burst[k_curr] = max(extend, open_burst)
                
                close_burst = curr_burst[k_curr]
                combined_incoming = max(best_score_to_norm, close_burst)
                next_norm[k_curr] = combined_incoming + ll_tensor[s, k_curr, i]
                
            for k in range(K):
                curr_norm[k] = next_norm[k]
                curr_burst[k] = next_burst[k]
                current_scores[k] = max(curr_norm[k], curr_burst[k])

        # Backward Pass
        best_end_k = -1
        best_end_score = -np.inf
        for k in range(K):
            if current_scores[k] > best_end_score:
                best_end_score = current_scores[k]
                best_end_k = k
        
        final_paths[s, n_sites - 1] = best_end_k
        
        for i in range(n_sites - 1, 0, -1):
            curr_k = final_paths[s, i]
            prev_k = backpointers[i, curr_k]
            final_paths[s, i-1] = prev_k
            
    return final_paths

# =============================================================================
# 3. VECTORIZED EMISSION CALCULATOR
# =============================================================================

def calculate_batch_emissions(sample_probs_matrix, hap_dict, robustness_epsilon=1e-2):
    """
    Calculates the log-likelihood tensor for ALL samples against ALL haplotype pairs.
    """
    hap_keys = sorted(list(hap_dict.keys()))
    num_haps = len(hap_keys)
    num_samples, num_sites, _ = sample_probs_matrix.shape
    
    idx_i, idx_j = np.unravel_index(np.arange(num_haps**2), (num_haps, num_haps))
    state_defs = np.stack([idx_i, idx_j], axis=1).astype(np.int32)
    
    hap_list = [hap_dict[k] for k in hap_keys]
    if not hap_list:
        return np.zeros((num_samples, 0, num_sites)), state_defs, hap_keys
        
    haps_tensor = np.array(hap_list) # (Num_Haps, Sites, 2)
    
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
    ll_matrix = np.maximum(ll_matrix, -5.0)
    
    return ll_matrix, state_defs, hap_keys

# =============================================================================
# 4. DATA CONVERSION & DRIVER
# =============================================================================

def compress_path_to_chunks(path_indices, positions, state_defs, hap_keys):
    """Converts site-by-site path to RLE Chunk objects."""
    chunks = []
    if len(path_indices) == 0: return chunks
    
    current_state = path_indices[0]
    start_pos = positions[0]
    
    for i in range(1, len(path_indices)):
        state = path_indices[i]
        if state != current_state:
            end_pos = positions[i-1]
            h1_idx, h2_idx = state_defs[current_state]
            
            chunks.append(PaintedChunk(
                start=int(start_pos),
                end=int(end_pos),
                hap1=hap_keys[h1_idx],
                hap2=hap_keys[h2_idx],
                state_idx=int(current_state)
            ))
            current_state = state
            start_pos = positions[i]
            
    end_pos = positions[-1]
    h1_idx, h2_idx = state_defs[current_state]
    chunks.append(PaintedChunk(
        start=int(start_pos),
        end=int(end_pos),
        hap1=hap_keys[h1_idx],
        hap2=hap_keys[h2_idx],
        state_idx=int(current_state)
    ))
    return chunks

def paint_samples_in_block(block_result, sample_probs_matrix, sample_sites, 
                           recomb_rate=1e-8):
    """
    Paints all samples in a block using Vectorized Viterbi.
    
    Returns:
        BlockPainting object containing results for all samples.
    """
    
    # 1. Extract Data
    positions = block_result.positions
    hap_dict = block_result.haplotypes
    
    # Subset sample data
    block_samples_data = analysis_utils.get_sample_data_at_sites_multiple(
        sample_probs_matrix, sample_sites, positions
    )
    
    # 2. Calculate Emissions
    ll_tensor, state_defs, hap_keys = calculate_batch_emissions(
        block_samples_data, hap_dict, robustness_epsilon=DEFAULT_ROBUSTNESS_EPSILON
    )
    
    # 3. Run Viterbi
    num_haps = len(hap_keys)
    raw_paths = viterbi_batch_solver(ll_tensor, positions, recomb_rate, state_defs, num_haps)
    
    # 4. Post-Process to Objects
    sample_paintings = []
    num_samples = len(raw_paths)
    
    for s in range(num_samples):
        chunks = compress_path_to_chunks(raw_paths[s], positions, state_defs, hap_keys)
        sample_paintings.append(SamplePainting(s, chunks))
        
    range_tuple = (int(positions[0]), int(positions[-1]))
    return BlockPainting(range_tuple, sample_paintings)