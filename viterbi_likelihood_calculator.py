import numpy as np
import math
import warnings
from multiprocess import Pool
from scipy.special import logsumexp
from functools import partial

# Import existing utilities
import analysis_utils 
import block_linking_em

# Disable underflow warnings common in log-space HMMs
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Define defaults
DEFAULT_LOG_BASE = math.e
# Robustness parameter: 1e-2 means 1% chance any read is random noise/error.
DEFAULT_ROBUSTNESS_EPSILON = 1e-2

# =============================================================================
# 0. NUMBA SETUP
# =============================================================================
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("WARNING: Numba not found. Recombination scanning will be slow.")
    # Dummy decorators
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator
    prange = range

# =============================================================================
# 1. CORE MATH & VITERBI (MULTI-STATE INTRA-BLOCK RECOMBINATION)
# =============================================================================

@njit(parallel=True, fastmath=True)
def viterbi_distance_aware_forward(ll_tensor, positions, recomb_rate, state_definitions):
    """
    Performs a Viterbi Forward Scan (Site 0 -> N) through a genomic block.
    Optimizes the path to find the max likelihood of data ending in state K.
    """
    n_samples, K, n_sites = ll_tensor.shape
    end_scores = np.empty((n_samples, K), dtype=np.float64)
    
    log_2 = 0.69314718056
    
    for s in prange(n_samples):
        # 1. Initialize with First Site
        current_scores = np.empty(K, dtype=np.float64)
        for k in range(K):
            current_scores[k] = ll_tensor[s, k, 0]
            
        # 2. Iterate Forward through the block
        for i in range(1, n_sites):
            next_step_scores = np.empty(K, dtype=np.float64)
            
            # Calculate physical distance
            dist_bp = positions[i] - positions[i-1]
            if dist_bp < 1: dist_bp = 1
            
            theta = float(dist_bp) * recomb_rate
            if theta > 0.5: theta = 0.5
            
            # Transition Log-Probs
            if theta <= 1e-300:
                log_switch = -1e300 # Effectively -inf
                log_stay = 0.0
            else:
                log_switch = math.log(theta)
                log_stay = math.log(1.0 - theta)
            
            # Costs for 0, 1, or 2 switches
            # 0 switches: (1-t)*(1-t)
            cost_0 = 2.0 * log_stay                 
            # 1 switch: 2 * t * (1-t)
            cost_1 = log_2 + log_switch + log_stay
            # 2 switches: t * t
            cost_2 = 2.0 * log_switch               
            
            for k_curr in range(K):
                h1_curr = state_definitions[k_curr, 0]
                h2_curr = state_definitions[k_curr, 1]
                
                best_incoming_score = -np.inf
                
                for k_prev in range(K):
                    h1_prev = state_definitions[k_prev, 0]
                    h2_prev = state_definitions[k_prev, 1]
                    
                    # Determine transition distance
                    dist = 2
                    
                    # Case 0: Exact match
                    if h1_curr == h1_prev and h2_curr == h2_prev: 
                        dist = 0
                    # Case 0 (Flip): Free phase switching allowed to match Unordered Genotype model
                    elif h1_curr == h2_prev and h2_curr == h1_prev: 
                        dist = 0
                    # Case 1: Single Recombination
                    elif (h1_curr == h1_prev or h1_curr == h2_prev or 
                          h2_curr == h1_prev or h2_curr == h2_prev): 
                        dist = 1
                    
                    trans_log_prob = -np.inf
                    if dist == 0: trans_log_prob = cost_0
                    elif dist == 1: trans_log_prob = cost_1
                    else: trans_log_prob = cost_2
                    
                    score = current_scores[k_prev] + trans_log_prob
                    if score > best_incoming_score:
                        best_incoming_score = score
                
                next_step_scores[k_curr] = ll_tensor[s, k_curr, i] + best_incoming_score
            
            current_scores = next_step_scores
        
        # 3. Save result (Score at Last Site)
        for k in range(K):
            end_scores[s, k] = current_scores[k]
        
            
    return end_scores

@njit(parallel=True, fastmath=True)
def viterbi_distance_aware_backward(ll_tensor, positions, recomb_rate, state_definitions):
    """
    Performs a Viterbi Backward Scan (Site N -> 0) through a genomic block.
    Optimizes the path to find the max likelihood of data starting in state K.
    """
    n_samples, K, n_sites = ll_tensor.shape
    start_scores = np.empty((n_samples, K), dtype=np.float64)
    
    log_2 = 0.69314718056
    
    for s in prange(n_samples):
        # 1. Initialize with Last Site
        next_scores = np.empty(K, dtype=np.float64)
        for k in range(K):
            next_scores[k] = ll_tensor[s, k, n_sites - 1]
            
        # 2. Iterate Backward
        for i in range(n_sites - 2, -1, -1):
            current_scores_scratch = np.empty(K, dtype=np.float64)
            
            dist_bp = positions[i+1] - positions[i]
            if dist_bp < 1: dist_bp = 1
            
            theta = float(dist_bp) * recomb_rate
            if theta > 0.5: theta = 0.5
            
            if theta <= 1e-300:
                log_switch = -1e300
                log_stay = 0.0
            else:
                log_switch = math.log(theta)
                log_stay = math.log(1.0 - theta)
            
            cost_0 = 2.0 * log_stay
            cost_1 = log_2 + log_switch + log_stay
            cost_2 = 2.0 * log_switch
            
            for k_curr in range(K): 
                h1_curr = state_definitions[k_curr, 0]
                h2_curr = state_definitions[k_curr, 1]
                
                best_path_score = -np.inf
                
                for k_next in range(K):
                    h1_next = state_definitions[k_next, 0]
                    h2_next = state_definitions[k_next, 1]
                    
                    dist = 2
                    if h1_curr == h1_next and h2_curr == h2_next: dist = 0
                    elif h1_curr == h2_next and h2_curr == h1_next: dist = 0
                    elif (h1_curr == h1_next or h1_curr == h2_next or 
                          h2_curr == h1_next or h2_curr == h2_next): dist = 1
                    
                    trans_log_prob = -np.inf
                    if dist == 0: trans_log_prob = cost_0
                    elif dist == 1: trans_log_prob = cost_1
                    else: trans_log_prob = cost_2
                    
                    score_from_k_next = next_scores[k_next] + trans_log_prob
                    if score_from_k_next > best_path_score:
                        best_path_score = score_from_k_next
                
                current_scores_scratch[k_curr] = ll_tensor[s, k_curr, i] + best_path_score
            
            next_scores = current_scores_scratch
                    
        # 3. Save result
        for k in range(K):
            start_scores[s, k] = next_scores[k]
            
    return start_scores

# =============================================================================
# 2. DATA STRUCTURES & EMISSION CALCULATION
# =============================================================================

class ViterbiSplitLikelihood:
    """
    Container for the Viterbi emission results of a single block.
    """
    def __init__(self, start_matrix, end_matrix):
        self.starting = start_matrix
        self.ending = end_matrix

class ViterbiSplitLikelihoods:
    def __init__(self, data_list):
        self.blocks = []
        for item in data_list:
            if isinstance(item, ViterbiSplitLikelihood):
                self.blocks.append(item)
            else:
                self.blocks.append(ViterbiSplitLikelihood(item[0], item[1]))

    def __len__(self): return len(self.blocks)
    def __getitem__(self, idx): return self.blocks[idx]
    def __iter__(self): return iter(self.blocks)

def calculate_viterbi_emission_tensors(samples_matrix, block_result,
                                       log_likelihood_base=math.e,
                                       robustness_epsilon=DEFAULT_ROBUSTNESS_EPSILON,
                                       allow_intra_block_recomb=True,
                                       recomb_rate=5e-7):
    """
    Calculates the 'Start' and 'End' Viterbi likelihood matrices for a block.
    """
    num_samples = samples_matrix.shape[0]
    hap_dict = block_result.haplotypes
    
    if block_result.keep_flags is not None:
        keep_flags = np.array(block_result.keep_flags, dtype=bool)
    else:
        keep_flags = np.ones(len(block_result.positions), dtype=bool)
        
    hap_keys = sorted(list(hap_dict.keys()))
    num_haps = len(hap_keys)
    
    if num_haps == 0:
        empty = np.zeros((num_samples, 0, 0))
        return (empty, empty)

    # 1. Prepare Data
    hap_list = [hap_dict[k] for k in hap_keys]
    if len(hap_list) > 0 and (hap_list[0].size == 0):
        haps_tensor = np.zeros((num_haps, 0, 2))
    else:
        haps_tensor = np.array(hap_list)
        
    samples_masked = samples_matrix[:, keep_flags, :]
    haps_masked = haps_tensor[:, keep_flags, :]
    
    valid_positions = np.array(block_result.positions)[keep_flags]
    valid_positions = valid_positions.astype(np.int64)
    
    num_active_sites = samples_masked.shape[1]
    
    # 2. Calculate Genotype Probabilities implied by Haplotypes
    # h0: (H, Sites) -> (H, 1, Sites)
    # h1: (H, Sites) -> (H, 1, Sites)
    h0 = haps_masked[:, :, 0]
    h1 = haps_masked[:, :, 1]
    
    c00 = h0[:, None, :] * h0[None, :, :]
    c11 = h1[:, None, :] * h1[None, :, :]
    # c01 = P(0,1) + P(1,0) (Unordered Genotype Probability)
    c01 = (h0[:, None, :] * h1[None, :, :]) + (h1[:, None, :] * h0[None, :, :])
    
    # Flatten: (Num_Haps^2, Sites)
    c00_flat = c00.reshape(-1, num_active_sites)
    c01_flat = c01.reshape(-1, num_active_sites)
    c11_flat = c11.reshape(-1, num_active_sites)
    
    # 3. Calculate P(Data | Haps) = sum_g P(Data|g) * P(g|Haps)
    # Samples: (N, 1, Sites)
    term_0 = samples_masked[:, np.newaxis, :, 0] * c00_flat[np.newaxis, :, :]
    term_1 = samples_masked[:, np.newaxis, :, 1] * c01_flat[np.newaxis, :, :]
    term_2 = samples_masked[:, np.newaxis, :, 2] * c11_flat[np.newaxis, :, :]
    
    model_probs = term_0 + term_1 + term_2
    
    # Apply Robust Mixture
    uniform_prob = 1.0 / 3.0
    final_probs = (model_probs * (1.0 - robustness_epsilon)) + (robustness_epsilon * uniform_prob)
    
    min_prob = 1e-300
    final_probs[final_probs < min_prob] = min_prob
    ll_per_site = np.log(final_probs)
    
    # 4. Total Log-Likelihoods (Viterbi Scan)
    idx_i, idx_j = np.unravel_index(np.arange(num_haps**2), (num_haps, num_haps))
    state_definitions = np.stack([idx_i, idx_j], axis=1).astype(np.int32)
    ll_tensor = np.ascontiguousarray(ll_per_site)
    
    # Backward Scan (Optimized for Start)
    total_ll_start = viterbi_distance_aware_backward(
        ll_tensor, valid_positions, float(recomb_rate), state_definitions
    )
    # Forward Scan (Optimized for End)
    total_ll_end = viterbi_distance_aware_forward(
        ll_tensor, valid_positions, float(recomb_rate), state_definitions
    )

    dense_start = total_ll_start.reshape(num_samples, num_haps, num_haps)
    dense_end = total_ll_end.reshape(num_samples, num_haps, num_haps)
    
    # Ensure symmetry by taking maximum (accounting for numerical precision)
    # This is valid because free phase flipping means (i,j) ≡ (j,i)
    dense_start = np.maximum(dense_start, dense_start.transpose(0, 2, 1))
    dense_end = np.maximum(dense_end, dense_end.transpose(0, 2, 1))
    
    return (dense_start, dense_end)
# --- Worker Utilities ---
_worker_data = {}

def _viterbi_init_worker(samples_matrix, sites, block_results, recomb_rate):
    _worker_data['samples_matrix'] = samples_matrix
    _worker_data['sample_sites'] = sites
    _worker_data['block_results'] = block_results
    _worker_data['recomb_rate'] = recomb_rate

def _viterbi_worker_calc_emissions(indices):
    samples_matrix = _worker_data['samples_matrix']
    sites = _worker_data['sample_sites']
    block_results = _worker_data['block_results']
    recomb_rate = _worker_data['recomb_rate']
    results = []
    
    for i in indices:
        block = block_results[i]
        block_samples = analysis_utils.get_sample_data_at_sites_multiple(
            samples_matrix, sites, block.positions
        )
        tensors = calculate_viterbi_emission_tensors(
            block_samples, block, recomb_rate=recomb_rate
        )
        results.append(tensors)
    return results

def generate_all_viterbi_likelihoods(samples_matrix, sample_sites, block_results, 
                                     recomb_rate=5e-7, num_processes=16):
    n_blocks = len(block_results)
    indices = list(range(n_blocks))
    chunk_size = math.ceil(n_blocks / num_processes) if n_blocks > 0 else 1
    chunks = [indices[i:i + chunk_size] for i in range(0, n_blocks, chunk_size)]
    
    init_args = (samples_matrix, sample_sites, block_results, recomb_rate)
    with Pool(num_processes, initializer=_viterbi_init_worker, initargs=init_args) as pool:
        results_chunks = pool.map(_viterbi_worker_calc_emissions, chunks)
        
    full_list = [item for sublist in results_chunks for item in sublist]
    return ViterbiSplitLikelihoods(full_list)