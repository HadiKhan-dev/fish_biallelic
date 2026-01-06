import numpy as np
import math
import warnings
from multiprocess import Pool
from scipy.special import logsumexp

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
# 1. NUMBA KERNELS (LOG-SPACE SCANS)
# =============================================================================

@njit(fastmath=False)
def log_add_exp(a, b):
    """
    Numerically stable log-add-exp helper for scalars.
    Calculates log(exp(a) + exp(b)).
    
    Args:
        a (float): First log probability.
        b (float): Second log probability.
        
    Returns:
        float: The result of log(exp(a) + exp(b)).
    """
    if a == -np.inf: return b
    if b == -np.inf: return a
    
    if a > b:
        return a + math.log(1.0 + math.exp(b - a))
    else:
        return b + math.log(1.0 + math.exp(a - b))

@njit(parallel=True, fastmath=False)
def scan_distance_aware_forward(ll_tensor, positions, recomb_rate, state_definitions, incoming_priors, n_haps):
    """
    Performs the 'Micro-HMM' Forward Scan (Log-Sum-Exp) inside a single block.
    
    Includes BURST LOGIC:
    Maintains parallel 'Normal' and 'Burst' states to handle gene conversions/errors
    without geometric likelihood decay.
    
    Args:
        ll_tensor (np.ndarray): Shape (Samples, K, Sites). Log-likelihood of data given state.
        positions (np.ndarray): Genomic positions of sites in this block.
        recomb_rate (float): Probability of recombination per base pair.
        state_definitions (np.ndarray): Shape (K, 2). Maps state index to (Hap1, Hap2).
        incoming_priors (np.ndarray): Shape (Samples, K). The accumulated probability 
                                      mass arriving at the *start* of this block from previous blocks.
        n_haps (int): Number of haplotypes in this block.
        
    Returns:
        np.ndarray: End probabilities (Samples, K) representing the likelihood of the 
                    entire path ending in specific states at the last site of the block.
    """
    n_samples, K, n_sites = ll_tensor.shape
    end_probs = np.full((n_samples, K), -np.inf, dtype=np.float64)
    min_prob = 1e-15 
    
    # --- BURST PARAMETERS ---
    # Cost to start ignoring data. -8.0 ~= 1/3000 chance.
    GAP_OPEN = -10.0 
    # Cost to continue ignoring data. 0.0 means decay is just uniform prob.
    GAP_EXTEND = 0.0 
    # Log likelihood of random match (ln(1/3))
    UNIFORM_LOG_PROB = -1.0986 
    
    BURST_STEP = UNIFORM_LOG_PROB + GAP_EXTEND
    
    # Pre-calculate log terms for recombination penalty
    if n_haps > 1:
        log_N_minus_1 = math.log(float(n_haps - 1))
    else:
        log_N_minus_1 = 0.0

    for s in prange(n_samples):
        # 1. INJECTION: Site 0 gets Emission + Incoming Prior (Macro-Transition)
        current_normal = np.empty(K, dtype=np.float64)
        current_burst = np.empty(K, dtype=np.float64)
        
        for k in range(K):
            # Prior flows into Normal state
            prior = incoming_priors[s, k]
            emission = ll_tensor[s, k, 0]
            
            # Start Normal: Prior + Emission
            current_normal[k] = prior + emission
            # Start Burst: Prior + Open + Uniform
            current_burst[k] = prior + GAP_OPEN + UNIFORM_LOG_PROB
            
        # 2. SCAN: Propagate from Site 1 to N (Micro-Transition)
        for i in range(1, n_sites):
            
            next_normal = np.full(K, -np.inf, dtype=np.float64)
            next_burst = np.full(K, -np.inf, dtype=np.float64)
            
            # Calculate distance-dependent transition probs
            dist_bp = positions[i] - positions[i-1]
            if dist_bp < 1: dist_bp = 1
            theta = float(dist_bp) * recomb_rate
            if theta > 0.5: theta = 0.5 
            
            if theta < min_prob:
                log_switch = -np.inf
                log_stay = 0.0
            else:
                log_switch = math.log(theta)
                log_stay = math.log(1.0 - theta)
            
            # Transition Costs
            cost_0 = 2.0 * log_stay # Stay in both lineages
            
            if log_switch == -np.inf:
                cost_1 = -np.inf
                cost_2 = -np.inf
            else:
                # Switch in one lineage (normalized by choices)
                cost_1 = log_switch + log_stay - log_N_minus_1
                # Switch in both lineages
                cost_2 = 2.0 * log_switch - 2.0 * log_N_minus_1

            # Inner Loop: Transition from prev_site (k_prev) to curr_site (k_curr)
            for k_curr in range(K):
                h1_curr = state_definitions[k_curr, 0]
                h2_curr = state_definitions[k_curr, 1]
                
                # A. Calculate Incoming Mass from Recombination (Normal -> Normal)
                # We use LOG_ADD_EXP (Sum-Product) for Haplotype Recombination 
                # to preserve the "Forward Variable" definition.
                total_incoming_recomb = -np.inf
                
                for k_prev in range(K):
                    h1_prev = state_definitions[k_prev, 0]
                    h2_prev = state_definitions[k_prev, 1]
                    
                    dist = 0
                    if h1_curr != h1_prev: dist += 1
                    if h2_curr != h2_prev: dist += 1
                    
                    # Select Transition Cost
                    trans_log_prob = cost_2
                    if dist == 0: trans_log_prob = cost_0
                    elif dist == 1: trans_log_prob = cost_1
                    
                    # Sum-Product from Previous Normal
                    path_prob = current_normal[k_prev] + trans_log_prob
                    total_incoming_recomb = log_add_exp(total_incoming_recomb, path_prob)
                
                # B. Update BURST State
                # We use MAX (Viterbi) for Burst switching to match Standard Code behavior
                # and avoid diluting the likelihood with the "Error" hypothesis.
                
                # 1. Extend previous burst (Burst -> Burst)
                extend_path = current_burst[k_curr] + BURST_STEP
                # 2. Open new burst from Recombined Normal (Normal -> Burst)
                open_path = total_incoming_recomb + GAP_OPEN + BURST_STEP
                
                next_burst[k_curr] = max(extend_path, open_path)
                
                # C. Update NORMAL State
                # 1. Continue from Recombined Normal (Normal -> Normal)
                continue_path = total_incoming_recomb
                # 2. Close previous Burst (Burst -> Normal)
                close_path = current_burst[k_curr]
                
                combined_incoming = max(continue_path, close_path)
                
                # Add Emission
                next_normal[k_curr] = combined_incoming + ll_tensor[s, k_curr, i]
            
            # Swap buffers
            for k in range(K):
                current_normal[k] = next_normal[k]
                current_burst[k] = next_burst[k]
        
        # Save final state at last site
        # Use MAX to select the best explanation (Burst vs Normal)
        for k in range(K):
            end_probs[s, k] = max(current_normal[k], current_burst[k])
            
    return end_probs

@njit(parallel=True, fastmath=False)
def scan_distance_aware_backward(ll_tensor, positions, recomb_rate, state_definitions, incoming_priors, n_haps):
    """
    Performs the 'Micro-HMM' Backward Scan (Log-Sum-Exp) inside a single block.
    
    Includes BURST LOGIC symmetric to the Forward scan.
    
    Args:
        ll_tensor (np.ndarray): Shape (Samples, K, Sites).
        positions (np.ndarray): Genomic positions.
        recomb_rate (float): Recombination rate.
        state_definitions (np.ndarray): State mapping.
        incoming_priors (np.ndarray): Shape (Samples, K). Probabilities from the *next* block.
        n_haps (int): Number of haplotypes.
        
    Returns:
        np.ndarray: Start probabilities (Samples, K) representing the likelihood of 
                    all future data given the state at Site 0.
    """
    n_samples, K, n_sites = ll_tensor.shape
    start_probs = np.full((n_samples, K), -np.inf, dtype=np.float64)
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
        # 1. INJECTION: Site N gets Emission + Incoming Prior (from Next Block)
        # We assume the "Next Block" expects a valid state, so we close any bursts here.
        next_normal = np.empty(K, dtype=np.float64)
        next_burst = np.empty(K, dtype=np.float64)
        
        for k in range(K):
            total_future = ll_tensor[s, k, n_sites - 1] + incoming_priors[s, k]
            next_normal[k] = total_future
            # In backward pass, "Burst" at N means we emit Uniform at N and go to Future.
            next_burst[k] = UNIFORM_LOG_PROB + incoming_priors[s, k]
            
        # 2. SCAN: Propagate from Site N-1 down to 0
        for i in range(n_sites - 2, -1, -1):
            current_normal_scratch = np.full(K, -np.inf, dtype=np.float64)
            current_burst_scratch = np.full(K, -np.inf, dtype=np.float64)
            
            # Distance logic (using distance to i+1)
            dist_bp = positions[i+1] - positions[i]
            if dist_bp < 1: dist_bp = 1
            theta = float(dist_bp) * recomb_rate
            if theta > 0.5: theta = 0.5
            
            if theta < min_prob:
                log_switch = -np.inf
                log_stay = 0.0
            else:
                log_switch = math.log(theta)
                log_stay = math.log(1.0 - theta)
            
            cost_0 = 2.0 * log_stay
            if log_switch == -np.inf:
                cost_1 = -np.inf
                cost_2 = -np.inf
            else:
                cost_1 = log_switch + log_stay - log_N_minus_1
                cost_2 = 2.0 * log_switch - 2.0 * log_N_minus_1
            
            # Transition: From Current(i) to Next(i+1)
            for k_curr in range(K): 
                h1_curr = state_definitions[k_curr, 0]
                h2_curr = state_definitions[k_curr, 1]
                
                # A. Calculate Flow to Future Normal (Recombination allowed)
                total_to_future_normal = -np.inf
                
                for k_next in range(K):
                    h1_next = state_definitions[k_next, 0]
                    h2_next = state_definitions[k_next, 1]
                    
                    dist = 0
                    if h1_curr != h1_next: dist += 1
                    if h2_curr != h2_next: dist += 1
                    
                    trans_log_prob = cost_2
                    if dist == 0: trans_log_prob = cost_0
                    elif dist == 1: trans_log_prob = cost_1
                    
                    path_prob = next_normal[k_next] + trans_log_prob
                    total_to_future_normal = log_add_exp(total_to_future_normal, path_prob)
                
                # B. Update BURST State (at i)
                # Use MAX (Viterbi)
                
                # 1. Extend to Future Burst (Burst i -> Burst i+1)
                extend_path = next_burst[k_curr] + BURST_STEP 
                # 2. Close to Future Normal (Burst i -> Normal i+1)
                close_path = next_normal[k_curr]
                
                current_burst_scratch[k_curr] = max(extend_path, close_path)
                
                # C. Update NORMAL State (at i)
                # 1. Recombine to Future Normal (Normal i -> Normal i+1)
                recomb_path = total_to_future_normal 
                # 2. Open to Future Burst (Normal i -> Burst i+1)
                open_path = next_burst[k_curr] + GAP_OPEN + BURST_STEP
                
                combined_future = max(recomb_path, open_path)
                
                # Add Emission at current site i
                current_normal_scratch[k_curr] = combined_future + ll_tensor[s, k_curr, i]
            
            for k in range(K):
                next_normal[k] = current_normal_scratch[k]
                next_burst[k] = current_burst_scratch[k]
        
        # Save state at Site 0
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
    """
    tasks = []
    # Params dict allows passing configs easily
    params = {'robustness_epsilon': DEFAULT_ROBUSTNESS_EPSILON}
    
    for block in block_results:
        block_samples = analysis_utils.get_sample_data_at_sites_multiple(
            samples_matrix, sample_sites, block.positions
        )
        tasks.append((block_samples, block, params))

    if num_processes > 1:
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
                                           use_standard_baum_welch=True): # Default True, but set False for Viterbi Fix
    """
    Main driver for HMM-EM transition calculation.
    """
    current_trans = block_linking_em.initial_transition_probabilities(haps_data, space_gap)
    
    print(f"  - Pre-calculating raw HMM emissions (Gap {space_gap})...")
    raw_blocks = generate_viterbi_block_emissions(full_samples_data, sample_sites, haps_data, num_processes=num_processes)
    
    prev_ll = -np.inf
    
    for it in range(max_num_iterations):
        # Match decay schedule
        effective_lr = learning_rate * (0.9 ** it)
        effective_lr = max(effective_lr, 0.1)

        # E-Step
        S_res, R_res, current_ll = global_forward_backward_pass(
            raw_blocks, haps_data, current_trans, space_gap, recomb_rate
        )
        
        # M-Step (With Flag)
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
            print(f"  - Viterbi EM converged at iteration {it} (Improvement {rel_improvement:.2e})")
            break
            
        prev_ll = current_ll
            
    return current_trans

def _gap_worker(args):
    """Worker for multiprocessing gap calculations"""
    # Unpack the new flag from the arguments tuple
    gap, samples, sites, haps, max_iter, rate, use_std_bw = args
    
    return calculate_hap_transition_probabilities(
        samples, sites, haps, 
        max_num_iterations=max_iter, 
        space_gap=gap, 
        recomb_rate=rate, 
        num_processes=1,
        use_standard_baum_welch=use_std_bw  # Pass flag to calculator
    )

def generate_transition_probability_mesh_double_hmm(full_samples_data, sample_sites, haps_data, 
                                                 max_num_iterations=5, recomb_rate=5e-7,
                                                 use_standard_baum_welch=True):
    """
    Generates a full mesh of transition probabilities for all gap sizes using Viterbi-EM.
    
    Args:
        use_standard_baum_welch (bool): 
            If True: Uses standard update (sensitive to initialization/priors).
            If False: Uses Reset update (recommended for Viterbi to prevent path death).
    """
    max_gap = len(haps_data) - 1
    gaps = list(range(1, max_gap + 1))
    
    worker_args = []
    for gap in gaps:
        # Append the boolean flag to the arguments tuple
        worker_args.append((
            gap, 
            full_samples_data, 
            sample_sites, 
            haps_data, 
            max_num_iterations, 
            recomb_rate, 
            use_standard_baum_welch
        ))
    
    print(f"Calculating HMM-based transitions for {len(gaps)} gaps (StandardBW={use_standard_baum_welch})...")
    
    with Pool(16) as pool:
        results = pool.map(_gap_worker, worker_args)
    
    mesh_dict = dict(zip(gaps, results))
    return block_linking_em.TransitionMesh(mesh_dict)