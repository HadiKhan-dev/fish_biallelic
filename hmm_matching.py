import numpy as np
import math
import warnings
from multiprocess import Pool
from scipy.special import logsumexp
from functools import partial

import analysis_utils 
import block_linking_em

# Suppress divide by zero warnings in log-space calculations
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Standard robustness parameter to prevent zero-probability crashes
DEFAULT_ROBUSTNESS_EPSILON = 1e-2

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("WARNING: Numba not found. Computations will be extremely slow.")
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator
    prange = range

# =============================================================================
# SHARED MEMORY MANAGEMENT
# =============================================================================

_SHARED_DATA = {}

def _init_shared_data(data_dict):
    """
    Initializer for the worker pool.
    Updates the global _SHARED_DATA dict in the worker process.
    """
    global _SHARED_DATA
    _SHARED_DATA.clear()
    _SHARED_DATA.update(data_dict)

# =============================================================================
# 1. OPTIMIZED NUMBA KERNELS (O(N^2) Single-Switch)
# =============================================================================

@njit(fastmath=True)
def log_add_exp(a, b):
    """
    Numerically stable log-add-exp helper for scalars.
    Calculates log(exp(a) + exp(b)).
    """
    if a == -np.inf: return b
    if b == -np.inf: return a
    
    if a > b:
        return a + math.log(1.0 + math.exp(b - a))
    else:
        return b + math.log(1.0 + math.exp(a - b))

@njit(parallel=True, fastmath=True)
def scan_distance_aware_forward(ll_tensor, positions, recomb_rate, state_definitions, incoming_priors, n_haps):
    """
    Performs the 'Micro-HMM' Forward Scan (Log-Sum-Exp) inside a single block.
    
    OPTIMIZATION:
    Uses the Single-Switch Assumption (at most one chromosome recombines per site).
    This reduces complexity from O(Sites * Haps^4) to O(Sites * Haps^2).
    
    Includes BURST LOGIC:
    Maintains parallel 'Normal' and 'Burst' states to handle gene conversions/errors.
    
    Args:
        ll_tensor (np.ndarray): Shape (Samples, K, Sites). Log-likelihood of data given state.
        positions (np.ndarray): Genomic positions of sites in this block.
        recomb_rate (float): Probability of recombination per base pair.
        state_definitions (np.ndarray): Shape (K, 2). Maps state index to (Hap1, Hap2).
        incoming_priors (np.ndarray): Shape (Samples, K). The accumulated probability 
                                      mass arriving at the *start* of this block.
        n_haps (int): Number of haplotypes in this block.
        
    Returns:
        np.ndarray: End probabilities (Samples, K).
    """
    n_samples, K, n_sites = ll_tensor.shape
    end_probs = np.full((n_samples, K), -np.inf, dtype=np.float64)
    min_prob = 1e-15 
    
    # --- BURST PARAMETERS ---
    GAP_OPEN = -10.0 
    GAP_EXTEND = 0.0 
    UNIFORM_LOG_PROB = -1.0986 
    BURST_STEP = UNIFORM_LOG_PROB + GAP_EXTEND
    
    if n_haps > 1:
        log_N_minus_1 = math.log(float(n_haps - 1))
    else:
        log_N_minus_1 = 0.0

    for s in prange(n_samples):
        # 1. INJECTION: Site 0 gets Emission + Incoming Prior (Macro-Transition)
        current_normal = np.empty(K, dtype=np.float64)
        current_burst = np.empty(K, dtype=np.float64)
        
        for k in range(K):
            prior = incoming_priors[s, k]
            emission = ll_tensor[s, k, 0]
            current_normal[k] = prior + emission
            current_burst[k] = prior + GAP_OPEN + UNIFORM_LOG_PROB
            
        # 2. SCAN: Propagate from Site 1 to N (Micro-Transition)
        for i in range(1, n_sites):
            next_normal = np.empty(K, dtype=np.float64)
            next_burst = np.empty(K, dtype=np.float64)
            
            # Costs
            dist_bp = positions[i] - positions[i-1]
            if dist_bp < 1: dist_bp = 1
            theta = float(dist_bp) * recomb_rate
            if theta > 0.5: theta = 0.5 
            
            if theta < min_prob:
                log_switch = -1e20
                log_stay = 0.0
            else:
                log_switch = math.log(theta)
                log_stay = math.log(1.0 - theta)
            
            cost_0 = 2.0 * log_stay
            cost_1 = log_switch + log_stay - log_N_minus_1
            # Cost 2 (Double Switch) is banned (-inf) under this assumption
            
            # --- OPTIMIZATION: PRE-CALCULATE ROW/COL AGGREGATES ---
            # row_sums[h1] = Sum over h2 of P(h1, h2) -> Mass where Chr1 is h1
            # col_sums[h2] = Sum over h1 of P(h1, h2) -> Mass where Chr2 is h2
            
            row_sums = np.full(n_haps, -np.inf, dtype=np.float64)
            col_sums = np.full(n_haps, -np.inf, dtype=np.float64)
            
            for h1 in range(n_haps):
                for h2 in range(n_haps):
                    k = h1 * n_haps + h2
                    val = current_normal[k]
                    row_sums[h1] = log_add_exp(row_sums[h1], val)
                    col_sums[h2] = log_add_exp(col_sums[h2], val)
            
            # 3. Update States
            for k_curr in range(K):
                h1_curr = k_curr // n_haps
                h2_curr = k_curr % n_haps
                
                # Incoming Mass Logic:
                
                # 1. Stay: (h1, h2) -> (h1, h2)
                term_stay = current_normal[k_curr] + cost_0
                
                # 2. Switch Chr 2: (h1, *) -> (h1, h2)
                term_switch1_a = row_sums[h1_curr] + cost_1
                
                # 3. Switch Chr 1: (*, h2) -> (h1, h2)
                term_switch1_b = col_sums[h2_curr] + cost_1
                
                # Combine (Sum-Product)
                total_incoming = log_add_exp(term_stay, term_switch1_a)
                total_incoming = log_add_exp(total_incoming, term_switch1_b)
                
                # Burst Update (Viterbi/Max style)
                extend = current_burst[k_curr] + BURST_STEP
                open_path = total_incoming + GAP_OPEN + BURST_STEP
                next_burst[k_curr] = max(extend, open_path)
                
                # Normal Update
                close_path = current_burst[k_curr]
                combined = max(total_incoming, close_path)
                
                next_normal[k_curr] = combined + ll_tensor[s, k_curr, i]
            
            # Swap buffers
            for k in range(K):
                current_normal[k] = next_normal[k]
                current_burst[k] = next_burst[k]
        
        # Save final state at last site
        for k in range(K):
            end_probs[s, k] = max(current_normal[k], current_burst[k])
            
    return end_probs

@njit(parallel=True, fastmath=True)
def scan_distance_aware_backward(ll_tensor, positions, recomb_rate, state_definitions, incoming_priors, n_haps):
    """
    Optimized Backward Scan (O(Sites * Haps^2)).
    Assumes Single-Switch Only.
    """
    n_samples, K, n_sites = ll_tensor.shape
    start_probs = np.full((n_samples, K), -np.inf, dtype=np.float64)
    min_prob = 1e-15
    
    GAP_OPEN = -10.0 
    GAP_EXTEND = 0.0 
    UNIFORM_LOG_PROB = -1.0986 
    BURST_STEP = UNIFORM_LOG_PROB + GAP_EXTEND
    
    if n_haps > 1:
        log_N_minus_1 = math.log(float(n_haps - 1))
    else:
        log_N_minus_1 = 0.0
    
    for s in prange(n_samples):
        # 1. Init (Site N)
        next_normal = np.empty(K, dtype=np.float64)
        next_burst = np.empty(K, dtype=np.float64)
        
        for k in range(K):
            val = ll_tensor[s, k, n_sites - 1] + incoming_priors[s, k]
            next_normal[k] = val
            next_burst[k] = UNIFORM_LOG_PROB + incoming_priors[s, k]
            
        # 2. Scan Backwards
        for i in range(n_sites - 2, -1, -1):
            curr_norm_scratch = np.empty(K, dtype=np.float64)
            curr_burst_scratch = np.empty(K, dtype=np.float64)
            
            dist_bp = positions[i+1] - positions[i]
            if dist_bp < 1: dist_bp = 1
            theta = float(dist_bp) * recomb_rate
            if theta > 0.5: theta = 0.5
            
            if theta < min_prob:
                log_switch = -1e20
                log_stay = 0.0
            else:
                log_switch = math.log(theta)
                log_stay = math.log(1.0 - theta)
            
            cost_0 = 2.0 * log_stay
            cost_1 = log_switch + log_stay - log_N_minus_1
            # Double switch forbidden
            
            # --- AGGREGATES (Future States) ---
            row_sums = np.full(n_haps, -np.inf, dtype=np.float64)
            col_sums = np.full(n_haps, -np.inf, dtype=np.float64)
            
            for h1 in range(n_haps):
                for h2 in range(n_haps):
                    k = h1 * n_haps + h2
                    val = next_normal[k]
                    row_sums[h1] = log_add_exp(row_sums[h1], val)
                    col_sums[h2] = log_add_exp(col_sums[h2], val)
            
            for k_curr in range(K):
                h1_curr = k_curr // n_haps
                h2_curr = k_curr % n_haps
                
                # Flow FROM Current TO Future
                term_stay = next_normal[k_curr] + cost_0
                term_switch1_a = row_sums[h1_curr] + cost_1
                term_switch1_b = col_sums[h2_curr] + cost_1
                
                total_to_future = log_add_exp(term_stay, term_switch1_a)
                total_to_future = log_add_exp(total_to_future, term_switch1_b)
                
                # Burst Logic
                extend = next_burst[k_curr] + BURST_STEP 
                close_path = next_normal[k_curr]
                curr_burst_scratch[k_curr] = max(extend, close_path)
                
                # Normal Logic
                recomb_path = total_to_future 
                open_path = next_burst[k_curr] + GAP_OPEN + BURST_STEP
                combined = max(recomb_path, open_path)
                
                curr_norm_scratch[k_curr] = combined + ll_tensor[s, k_curr, i]
            
            for k in range(K):
                next_normal[k] = curr_norm_scratch[k]
                next_burst[k] = curr_burst_scratch[k]
        
        for k in range(K):
            start_probs[s, k] = max(next_normal[k], next_burst[k])
            
    return start_probs

# =============================================================================
# 2. DATA CONTAINERS & GENERATION
# =============================================================================

class ViterbiBlockLikelihood:
    """
    Holds the per-site log-likelihood tensor for a block, optimized for the Viterbi scan.
    """
    def __init__(self, tensor, positions, state_defs, num_haps):
        self.tensor = tensor         # (Samples, K_States, Sites)
        self.positions = positions   # (Sites,)
        self.state_defs = state_defs # (K_States, 2)
        self.num_haps = num_haps 

class ViterbiBlockList:
    """Simple container for ViterbiBlockLikelihood objects."""
    def __init__(self, blocks_list):
        self.blocks = blocks_list
    def __len__(self): return len(self.blocks)
    def __getitem__(self, idx): return self.blocks[idx]
    def __iter__(self): return iter(self.blocks)

def _worker_generate_viterbi_emissions(args):
    """
    Worker function to calculate P(Data_site | State) for all sites and states.
    Uses the Robust Mixture Model: P = (1-e)*Model + e*Uniform.
    """
    samples_matrix, block_hap, params = args
    # Robustness parameter to prevent outlier sites from zeroing out the likelihood
    epsilon = params.get('robustness_epsilon', DEFAULT_ROBUSTNESS_EPSILON)

    hap_dict = block_hap.haplotypes
    # Handling Flags
    if block_hap.keep_flags is not None:
        keep_flags = np.array(block_hap.keep_flags, dtype=bool)
    else:
        keep_flags = np.ones(len(block_hap.positions), dtype=bool)
        
    hap_keys = sorted(list(hap_dict.keys()))
    num_haps = len(hap_keys)
    
    # State Definitions: Map flattened index 0..K-1 to (h1, h2)
    # Full Directed State Space (no symmetry folding)
    idx_i, idx_j = np.unravel_index(np.arange(num_haps**2), (num_haps, num_haps))
    state_defs = np.stack([idx_i, idx_j], axis=1).astype(np.int32)
    
    hap_list = [hap_dict[k] for k in hap_keys]
    if not hap_list or hap_list[0].size == 0:
        haps_tensor = np.zeros((num_haps, 0, 2))
    else:
        haps_tensor = np.array(hap_list)
        
    # Apply Flags
    samples_masked = samples_matrix[:, keep_flags, :]
    haps_masked = haps_tensor[:, keep_flags, :]
    valid_positions = np.array(block_hap.positions)[keep_flags].astype(np.int64)
    
    # --- PROBABILISTIC MIXTURE CALCULATION ---
    h0 = haps_masked[:, :, 0]
    h1 = haps_masked[:, :, 1]
    
    # Calculate implied genotype probabilities for each pair
    # Shape: (K, Sites, 3)
    c00 = h0[:, None, :] * h0[None, :, :]
    c11 = h1[:, None, :] * h1[None, :, :]
    c01 = (h0[:, None, :] * h1[None, :, :]) + (h1[:, None, :] * h0[None, :, :])
    
    # Flatten to (K, Sites) per genotype channel
    combos_flat_0 = c00.reshape(num_haps**2, -1)
    combos_flat_1 = c01.reshape(num_haps**2, -1)
    combos_flat_2 = c11.reshape(num_haps**2, -1)
    
    # Calculate Model Probability: Sum_g P(Data|g) * P(g|Model)
    # Samples: (N, Sites, 3) -> Extract columns (N, Sites)
    # Broadcasting: (N, 1, Sites) * (1, K, Sites)
    
    # Term 0: Read=0 * Genotype=0
    term_0 = samples_masked[:, np.newaxis, :, 0] * combos_flat_0[np.newaxis, :, :]
    term_1 = samples_masked[:, np.newaxis, :, 1] * combos_flat_1[np.newaxis, :, :]
    term_2 = samples_masked[:, np.newaxis, :, 2] * combos_flat_2[np.newaxis, :, :]
    
    model_probs = term_0 + term_1 + term_2 # (N, K, Sites)
    
    # Robustness Mixture: (1-eps)*Model + eps*Uniform
    # Uniform for 3 genotype states is 1/3
    uniform_prob = 1.0 / 3.0
    
    final_probs = (model_probs * (1.0 - epsilon)) + (epsilon * uniform_prob)
    
    # Safety floor for log (avoids -inf)
    min_prob = 1e-300
    final_probs[final_probs < min_prob] = min_prob
    
    ll_per_site = np.log(final_probs)
    
    # FIX: Apply Hard Floor of -2.0 per site
    ll_per_site = np.maximum(ll_per_site, -2.0)
    
    return ViterbiBlockLikelihood(np.ascontiguousarray(ll_per_site), valid_positions, state_defs, num_haps)

def generate_viterbi_block_emissions(samples_matrix, sample_sites, block_results, num_processes=16):
    """
    Parallel generator for ViterbiBlockLikelihood objects used in the scan.
    
    Updated to support both contiguous blocks and sparse (proxy) blocks by using 
    exact index mapping rather than slicing.
    
    Args:
        samples_matrix: (Samples x Global_Sites x 3) probability matrix.
        sample_sites: (Global_Sites,) array of positions.
        block_results: List of BlockResult objects (contiguous or proxy).
        num_processes: Parallel workers.
        
    Returns:
        ViterbiBlockList containing site-specific likelihood tensors.
    """
    tasks = []
    # Params dict allows passing configs easily
    params = {'robustness_epsilon': DEFAULT_ROBUSTNESS_EPSILON}
    
    for block in block_results:
        # ROBUST DATA FETCHING:
        # We find the exact indices of the block's positions in the global array.
        # This handles both contiguous ranges (0,1,2) and sparse proxies (0,10,20) correctly.
        indices = np.searchsorted(sample_sites, block.positions)
        
        # Fancy indexing returns a copy containing only the requested sites.
        # This guarantees the dimensions of block_samples match block.positions exactly.
        block_samples = samples_matrix[:, indices, :]
        
        tasks.append((block_samples, block, params))

    if num_processes > 1 and len(tasks) > 1:
        with Pool(num_processes) as pool:
            results = pool.map(_worker_generate_viterbi_emissions, tasks)
    else:
        results = list(map(_worker_generate_viterbi_emissions, tasks))
        
    return ViterbiBlockList(results)

# =============================================================================
# 3. GLOBAL FORWARD-BACKWARD PASS
# =============================================================================

def build_dense_transition_matrix(trans_dict, prev_keys, curr_keys, prev_idx, curr_idx):
    """
    Converts sparse dictionary transition probs to dense log-prob matrix T.
    
    Args:
        trans_dict: Dictionary {(prev_hap, curr_hap): prob}.
        prev_keys: List of haplotype IDs in previous block.
        curr_keys: List of haplotype IDs in current block.
        prev_idx: Index of previous block.
        curr_idx: Index of current block.
        
    Returns:
        np.ndarray: Matrix of shape (K_prev, K_curr) containing log probabilities.
    """
    n_prev = len(prev_keys)
    n_curr = len(curr_keys)
    K_prev = n_prev * n_prev
    K_curr = n_curr * n_curr
    
    T = np.full((K_prev, K_curr), -np.inf)
    
    # Optimization: Pre-lookup sparse dict to avoid string/tuple hashing in inner loop
    log_trans_cache = {}
    
    for u_i, u_key in enumerate(prev_keys):
        log_trans_cache[u_i] = {}
        for x_i, x_key in enumerate(curr_keys):
            key = ((prev_idx, u_key), (curr_idx, x_key))
            if key in trans_dict:
                log_trans_cache[u_i][x_i] = math.log(trans_dict[key])
            else:
                log_trans_cache[u_i][x_i] = -np.inf

    # Fill T: State U=(u1, u2) -> State V=(v1, v2)
    # Using FULL DIRECTED STATE SPACE
    for r in range(K_prev):
        u_idx, v_idx = divmod(r, n_prev)
        for c in range(K_curr):
            x_idx, y_idx = divmod(c, n_curr)
            
            val_1 = log_trans_cache[u_idx][x_idx]
            val_2 = log_trans_cache[v_idx][y_idx]
            
            if val_1 != -np.inf and val_2 != -np.inf:
                T[r, c] = val_1 + val_2
                
    return T

def global_forward_backward_pass(raw_blocks, block_results, transition_probs, space_gap, recomb_rate):
    """
    Orchestrates the genome-wide Forward and Backward passes using the Viterbi Sum kernels.
    
    Returns:
        tuple: (S_results, R_results, total_log_likelihood)
    """
    num_blocks = len(raw_blocks)
    num_samples = raw_blocks[0].tensor.shape[0]
    
    S_results = [] # Stores Forward scores for each block
    R_results = [None] * num_blocks # Stores Backward scores
    
    # --- PHASE 1: FORWARD (Calculating S) ---
    for i in range(num_blocks):
        block = raw_blocks[i]
        K_curr = block.tensor.shape[1] 
        n_haps = block.num_haps
        
        # 1. Calculate Incoming Priors (Macro-Transition)
        if i < space_gap:
            # First block(s): Uniform Priors
            priors = np.zeros((num_samples, K_curr))
        else:
            # Prior = S[i-gap] * T
            prev_idx = i - space_gap
            prev_S_internal = S_results[prev_idx] # RAW
            
            prev_keys = sorted(list(block_results[prev_idx].haplotypes.keys()))
            curr_keys = sorted(list(block_results[i].haplotypes.keys()))
            
            T = build_dense_transition_matrix(transition_probs[0][prev_idx], prev_keys, curr_keys, prev_idx, i)
            
            # Log-Space Matrix Multiplication
            priors = analysis_utils.log_matmul(prev_S_internal, T)
        
        # 2. Run Forward Scan (Micro-Transition)
        S_raw = scan_distance_aware_forward(
            block.tensor, block.positions, float(recomb_rate), block.state_defs, priors, block.num_haps
        )
        
        # STORE RAW RESULTS FOR RECURSION
        S_results.append(S_raw)
        
    # --- CALCULATE TOTAL LOG LIKELIHOOD ---
    # We sum the log-probabilities of the final states of the last block for each sample.
    # This acts as the P(Data | Model) for convergence checking.
    last_S = S_results[-1] # Shape (Samples, K)
    sample_likelihoods = logsumexp(last_S, axis=1)
    total_ll = np.sum(sample_likelihoods)
        
    # --- PHASE 2: BACKWARD (Calculating R) ---
    for i in range(num_blocks - 1, -1, -1):
            
        block = raw_blocks[i]
        K_curr = block.tensor.shape[1]
        n_haps = block.num_haps
        
        # 1. Calculate Future Priors
        if i >= num_blocks - space_gap:
            # Last block(s): Uniform Future
            priors = np.zeros((num_samples, K_curr))
            
        else:
            # Prior = R[i+gap] * T.Transpose
            next_idx = i + space_gap
            next_R_internal = R_results[next_idx] # RAW
            
            curr_keys = sorted(list(block_results[i].haplotypes.keys()))
            next_keys = sorted(list(block_results[next_idx].haplotypes.keys()))
            
            # NOTE: For Backward pass, we use T_bwd(Next -> Curr)
            T = build_dense_transition_matrix(transition_probs[1][next_idx], next_keys, curr_keys, next_idx, i)
            
            priors = analysis_utils.log_matmul(next_R_internal, T)
        
        # 2. Run Backward Scan
        R_raw = scan_distance_aware_backward(
            block.tensor, block.positions, float(recomb_rate), block.state_defs, priors, block.num_haps
        )
        
        R_results[i] = R_raw
        
    return S_results, R_results, total_ll

def update_transitions_layered_hmm(S_results, R_results, block_results, current_trans, 
                                     space_gap, use_standard_baum_welch=True):
    """
    Calculates expected transition counts (Xi) for the HMM.
    
    Fixes Applied:
    1. Always includes T_mat in 'Total' to capture Diploid Partner probability.
    2. Subtracts specific Edge Prior if using Reset/Viterbi EM.
    3. Removes Cross-Edge summation to prevent blurring.
    """
    new_trans_fwd = {}
    new_trans_bwd = {}
    num_blocks = len(S_results)
    num_samples = S_results[0].shape[0] 
    
    MIN_LOG_PROB = -10.0
    BATCH = 100
    PSEUDO_COUNT = 0.1 
    LOG_PSEUDO = math.log(PSEUDO_COUNT)
    
    # -----------------------------------------------------
    # LOOP 1: FORWARD TRANSITION UPDATE (Earlier -> Later)
    # -----------------------------------------------------
    for i in range(num_blocks - space_gap):
        next_idx = i + space_gap
        S_earlier = S_results[i]
        R_later = R_results[next_idx]
        
        curr_keys = sorted(list(block_results[i].haplotypes.keys()))
        next_keys = sorted(list(block_results[next_idx].haplotypes.keys()))
        
        # Build dense transition matrix (Required for Partner Priors)
        T_mat = build_dense_transition_matrix(current_trans[0][i], curr_keys, next_keys, i, next_idx)
        
        numerators = np.full((len(curr_keys)**2, len(next_keys)**2), -np.inf)
        
        # Vectorized Batch Processing
        for start_s in range(0, num_samples, BATCH):
            end_s = min(start_s + BATCH, num_samples)
            S_batch = S_earlier[start_s:end_s]
            R_batch = R_later[start_s:end_s]
            
            # Base: S (History) + R (Future)
            Total = S_batch[:, :, np.newaxis] + R_batch[:, np.newaxis, :]
            
            # ALWAYS add T_mat.
            # Even in Reset EM, we need T(b->d) to explain the partner chromosome.
            Total += T_mat[np.newaxis, :, :]
            
            # Normalize per sample to get Posterior (Soft Count)
            sample_totals = logsumexp(Total, axis=(1, 2), keepdims=True)
            Normalized_Total = Total - sample_totals
            
            # Sum counts across batch
            Batch_Log_Sum = logsumexp(Normalized_Total, axis=0)
            numerators = np.logaddexp(numerators, Batch_Log_Sum)

        # Collapse Diploid States -> Haplotype Transitions
        n_c = len(curr_keys)
        n_n = len(next_keys)
        hap_masses = {} 
        
        # Access sparse dict for fast lookup of priors to subtract (if needed)
        sparse_trans = current_trans[0][i]

        for r in range(numerators.shape[0]):
            u1, u2 = divmod(r, n_c)
            k_u1 = curr_keys[u1]
            k_u2 = curr_keys[u2]

            for c in range(numerators.shape[1]):
                v1, v2 = divmod(c, n_n)
                k_v1 = next_keys[v1]
                k_v2 = next_keys[v2]
                
                mass = numerators[r, c]
                if mass == -np.inf: continue
                
                # --- EDGE 1: Chromosome 1 (u1 -> v1) ---
                val = mass
                if not use_standard_baum_welch:
                    # Subtract ONLY this edge's prior. 
                    # T_mat included (prior_1 + prior_2). We want (prior_2).
                    # So: (S + R + p1 + p2) - p1 = S + R + p2
                    edge_prior = sparse_trans.get(((i, k_u1), (next_idx, k_v1)), 1e-9)
                    val = mass - math.log(edge_prior)
                hap_masses.setdefault((u1, v1), []).append(val)

                # --- EDGE 2: Chromosome 2 (u2 -> v2) ---
                val = mass
                if not use_standard_baum_welch:
                    edge_prior = sparse_trans.get(((i, k_u2), (next_idx, k_v2)), 1e-9)
                    val = mass - math.log(edge_prior)
                hap_masses.setdefault((u2, v2), []).append(val)
                
                # NOTE: Cross-edges (u1->v2, u2->v1) are NOT explicitly added.
                # They are covered by the iteration where the destination is swapped (v2, v1).

        # Apply Smoothing and Normalize
        fwd_raw_edges = {u: {} for u in curr_keys}
        
        for u_i in range(n_c):
            for v_i in range(n_n):
                if (u_i, v_i) in hap_masses:
                    data_log_count = logsumexp(hap_masses[(u_i, v_i)])
                else:
                    data_log_count = -np.inf
                
                # Add Pseudocounts (prevents death spiral)
                smoothed_val = np.logaddexp(data_log_count, LOG_PSEUDO)
                
                src = curr_keys[u_i]
                dst = next_keys[v_i]
                fwd_raw_edges[src][dst] = smoothed_val

        # Row Normalization
        final_fwd = {}
        for src, targets in fwd_raw_edges.items():
            if not targets: continue
            row_vals = list(targets.values())
            row_total = logsumexp(row_vals)
            
            renorm_sum = 0.0
            temp_probs = {}
            for dst, log_val in targets.items():
                log_p = log_val - row_total if row_total != -np.inf else -np.inf
                if log_p < MIN_LOG_PROB: log_p = MIN_LOG_PROB
                p = math.exp(log_p)
                temp_probs[dst] = p
                renorm_sum += p
            
            if renorm_sum == 0: renorm_sum = 1.0
            
            # Robust Mixture (1% Uniform)
            uniform_val = 1.0 / len(temp_probs)
            mix_rate = 0.01
            
            for dst, p in temp_probs.items():
                norm_p = p / renorm_sum
                final_p = (norm_p * (1.0 - mix_rate)) + (uniform_val * mix_rate)
                key = ((i, src), (next_idx, dst))
                final_fwd[key] = final_p
                
        new_trans_fwd[i] = final_fwd

    # -----------------------------------------------------
    # LOOP 2: BACKWARD TRANSITION UPDATE (Later -> Earlier)
    # -----------------------------------------------------
    for i in range(num_blocks - 1, space_gap - 1, -1):
        prev_idx = i - space_gap
        
        R_later_source = R_results[i]
        S_earlier_dest = S_results[prev_idx]
        
        curr_keys = sorted(list(block_results[i].haplotypes.keys()))      
        prev_keys = sorted(list(block_results[prev_idx].haplotypes.keys())) 
        
        T_mat = build_dense_transition_matrix(current_trans[1][i], curr_keys, prev_keys, i, prev_idx)
        
        numerators = np.full((len(curr_keys)**2, len(prev_keys)**2), -np.inf)
        
        for start_s in range(0, num_samples, BATCH):
            end_s = min(start_s + BATCH, num_samples)
            R_batch = R_later_source[start_s:end_s]
            S_batch = S_earlier_dest[start_s:end_s]
            
            Total = R_batch[:, :, np.newaxis] + S_batch[:, np.newaxis, :]
            Total += T_mat[np.newaxis, :, :] # Always add T
            
            sample_totals = logsumexp(Total, axis=(1, 2), keepdims=True)
            Normalized_Total = Total - sample_totals
            Batch_Log_Sum = logsumexp(Normalized_Total, axis=0)
            numerators = np.logaddexp(numerators, Batch_Log_Sum)
            
        n_c = len(curr_keys)
        n_p = len(prev_keys)
        hap_masses = {} 
        sparse_trans = current_trans[1][i]

        for r in range(numerators.shape[0]):
            u1, u2 = divmod(r, n_c)
            k_u1 = curr_keys[u1]
            k_u2 = curr_keys[u2]

            for c in range(numerators.shape[1]):
                v1, v2 = divmod(c, n_p)
                k_v1 = prev_keys[v1]
                k_v2 = prev_keys[v2]
                
                mass = numerators[r, c]
                if mass == -np.inf: continue

                # Edge 1
                val = mass
                if not use_standard_baum_welch:
                    edge_prior = sparse_trans.get(((i, k_u1), (prev_idx, k_v1)), 1e-9)
                    val = mass - math.log(edge_prior)
                hap_masses.setdefault((u1, v1), []).append(val)

                # Edge 2
                val = mass
                if not use_standard_baum_welch:
                    edge_prior = sparse_trans.get(((i, k_u2), (prev_idx, k_v2)), 1e-9)
                    val = mass - math.log(edge_prior)
                hap_masses.setdefault((u2, v2), []).append(val)
                
                # No Cross Edges
                
        # Normalize Backward
        bwd_raw_edges = {u: {} for u in curr_keys}
        
        for u_i in range(n_c):
            for v_i in range(n_p):
                if (u_i, v_i) in hap_masses:
                    data_log_count = logsumexp(hap_masses[(u_i, v_i)])
                else:
                    data_log_count = -np.inf
                    
                smoothed_val = np.logaddexp(data_log_count, LOG_PSEUDO)
                
                src = curr_keys[u_i]
                dst = prev_keys[v_i]
                bwd_raw_edges[src][dst] = smoothed_val

        final_bwd = {}
        for src, targets in bwd_raw_edges.items():
            if not targets: continue
            row_vals = list(targets.values())
            row_total = logsumexp(row_vals)
            
            renorm_sum = 0.0
            temp_probs = {}
            for dst, log_val in targets.items():
                log_p = log_val - row_total if row_total != -np.inf else -np.inf
                if log_p < MIN_LOG_PROB: log_p = MIN_LOG_PROB
                p = math.exp(log_p)
                temp_probs[dst] = p
                renorm_sum += p
            
            if renorm_sum == 0: renorm_sum = 1.0
            
            uniform_val = 1.0 / len(temp_probs)
            mix_rate = 0.01
            
            for dst, p in temp_probs.items():
                norm_p = p / renorm_sum
                final_p = (norm_p * (1.0 - mix_rate)) + (uniform_val * mix_rate)
                key = ((i, src), (prev_idx, dst))
                final_bwd[key] = final_p
                
        new_trans_bwd[i] = final_bwd
            
    return [new_trans_fwd, new_trans_bwd]


# =============================================================================
# 4. MAIN LOOP & API
# =============================================================================

def calculate_hap_transition_probabilities(full_samples_data, sample_sites, haps_data,
                                           max_num_iterations=10, space_gap=1,
                                           recomb_rate=5e-7, learning_rate=1.0,
                                           num_processes=16,
                                           ll_improvement_cutoff=1e-4,
                                           use_standard_baum_welch=True,
                                           precalculated_viterbi_emissions=None): # NEW ARGUMENT
    """
    Main driver for HMM-EM transition calculation.
    Supports pre-calculated emissions to avoid passing massive raw data to workers.
    """
    current_trans = block_linking_em.initial_transition_probabilities(haps_data, space_gap)
    
    # Use pre-calculated emissions if provided, otherwise calculate them here
    if precalculated_viterbi_emissions is not None:
        raw_blocks = precalculated_viterbi_emissions
    else:
        # Fallback to internal calculation (High Memory usage if data is large)
        raw_blocks = generate_viterbi_block_emissions(
            full_samples_data, sample_sites, haps_data, num_processes=num_processes
        )
    
    prev_ll = -np.inf
    
    for it in range(max_num_iterations):
        # Match decay schedule
        effective_lr = learning_rate * (0.9 ** it)
        effective_lr = max(effective_lr, 0.1)

        # E-Step
        S_res, R_res, current_ll = global_forward_backward_pass(
            raw_blocks, haps_data, current_trans, space_gap, recomb_rate
        )
        
        # M-Step
        new_trans = update_transitions_layered_hmm(
            S_res, R_res, haps_data, current_trans, space_gap, 
            use_standard_baum_welch=use_standard_baum_welch
        )
        
        # Smoothing
        smoothed = analysis_utils.smoothen_probs_vectorized(current_trans, new_trans, effective_lr)
        
        if isinstance(smoothed, dict): 
            current_trans = [smoothed[0], smoothed[1]]
        else: 
            current_trans = smoothed
            
        # Convergence Check
        rel_improvement = 0.0
        if prev_ll != -np.inf and prev_ll != 0:
            rel_improvement = (current_ll - prev_ll) / abs(prev_ll)
        elif prev_ll == -np.inf:
            rel_improvement = float('inf') 
            
        if it > 0 and 0 <= rel_improvement < ll_improvement_cutoff:
            break
            
        prev_ll = current_ll
            
    return current_trans

def _gap_worker(args):
    """Worker for multiprocessing gap calculations"""
    # Unpack the new flag from the arguments tuple (now includes emissions)
    gap, samples, sites, haps, max_iter, rate, use_std_bw, use_shared_emissions = args
    
    if use_shared_emissions:
        # Retrieve the massive emissions object from Shared Memory
        precalc_ems = _SHARED_DATA['viterbi_emissions']
    else:
        # If not using shared memory, this argument would have been passed directly (or None)
        # But in our new architecture, we aim to rely on shared memory for the large object.
        precalc_ems = None

    return calculate_hap_transition_probabilities(
        samples, sites, haps, 
        max_num_iterations=max_iter, 
        space_gap=gap, 
        recomb_rate=rate, 
        num_processes=1, # No nested pool
        use_standard_baum_welch=use_std_bw,
        precalculated_viterbi_emissions=precalc_ems
    )

def generate_transition_probability_mesh_double_hmm(full_samples_data, sample_sites, haps_data, 
                                                 max_num_iterations=5, recomb_rate=5e-7,
                                                 use_standard_baum_welch=True,
                                                 precalculated_viterbi_emissions=None): # NEW ARGUMENT
    """
    Generates a full mesh of transition probabilities for all gap sizes using Viterbi-EM.
    
    Args:
        precalculated_viterbi_emissions: Optional ViterbiBlockList. If provided, 
        full_samples_data and sample_sites are IGNORED to prevent memory pickling overhead.
    """
    max_gap = len(haps_data) - 1
    gaps = list(range(1, max_gap + 1))
    
    use_shared_emissions = False
    
    # CRITICAL MEMORY FIX:
    # If using pre-calculated emissions, we MUST put them in Shared Memory
    # rather than passing them as arguments to the worker.
    # Passing as args duplicates the object (pickles) for every worker task.
    shared_context = {}
    
    if precalculated_viterbi_emissions is not None:
        data_arg = None
        sites_arg = None
        use_shared_emissions = True
        shared_context['viterbi_emissions'] = precalculated_viterbi_emissions
    else:
        data_arg = full_samples_data
        sites_arg = sample_sites
    
    worker_args = []
    for gap in gaps:
        worker_args.append((
            gap, 
            data_arg, 
            sites_arg, 
            haps_data, 
            max_num_iterations, 
            recomb_rate, 
            use_standard_baum_welch,
            use_shared_emissions # Flag to tell worker to check shared memory
        ))
    
    print(f"Calculating HMM-based transitions for {len(gaps)} gaps (StandardBW={use_standard_baum_welch})...")
    
    # Initialize Pool with Shared Data
    with Pool(16, initializer=_init_shared_data, initargs=(shared_context,)) as pool:
        results = pool.map(_gap_worker, worker_args)
    
    mesh_dict = dict(zip(gaps, results))
    return block_linking_em.TransitionMesh(mesh_dict)