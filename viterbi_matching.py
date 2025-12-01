import numpy as np
import math
import time
import copy
from multiprocess import Pool
from scipy.special import logsumexp
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform

# Import your existing utilities
import analysis_utils 

# =============================================================================
# 0. NUMBA SETUP (Robust Fallback)
# =============================================================================
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("WARNING: Numba not found. Recombination scanning will be slow.")
    # Dummy decorators to prevent crashing if numba is missing
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator
    prange = range

#%%
# =============================================================================
# 1. CORE MATH & VITERBI (MULTI-STATE INTRA-BLOCK RECOMBINATION)
# =============================================================================

@njit(parallel=True, fastmath=True)
def viterbi_distance_aware_forward(ll_tensor, penalty, state_definitions):
    """
    Viterbi Forward Scan (Site 0 -> N).
    Switching cost depends on Hamming distance between diploid states.
    
    ll_tensor: (n_samples, K, n_sites)
    state_definitions: (K, 2) array of [hap1, hap2] for each state index.
    """
    n_samples, K, n_sites = ll_tensor.shape
    end_scores = np.empty((n_samples, K), dtype=np.float64)
    
    for s in prange(n_samples):
        # 1. Initialize with First Site
        current_scores = np.empty(K, dtype=np.float64)
        for k in range(K):
            current_scores[k] = ll_tensor[s, k, 0]
            
        # 2. Iterate Forward
        for i in range(1, n_sites):
            
            # We need to calculate the best score arriving AT 'k_curr' 
            # FROM any 'k_prev'
            
            # Temporary buffer for this step
            next_step_scores = np.empty(K, dtype=np.float64)
            
            for k_curr in range(K):
                h1_curr = state_definitions[k_curr, 0]
                h2_curr = state_definitions[k_curr, 1]
                
                best_incoming_score = -np.inf
                
                for k_prev in range(K):
                    # Calculate Diploid Hamming Distance (0, 1, or 2)
                    h1_prev = state_definitions[k_prev, 0]
                    h2_prev = state_definitions[k_prev, 1]
                    
                    # Logic to count shared alleles (Unordered)
                    # We try to match h1_curr to prev, then h2_curr to remaining prev
                    matches = 0
                    
                    # Track usage of prev alleles to avoid double counting
                    p1_used = False
                    
                    # Match h1_curr
                    if h1_curr == h1_prev:
                        matches += 1
                        p1_used = True
                    elif h1_curr == h2_prev:
                        matches += 1
                        # p2 used
                    
                    # Match h2_curr
                    if matches == 2:
                        pass # Already max matches
                    else:
                        # Check h2_curr against whatever is left
                        # If p1 was used by h1, we can only check p2
                        # If p1 wasn't used, we check p1 (and p2 if h1 didn't use it)
                        
                        # Simplified logic:
                        # If h1_curr matches h1_prev, h2_curr must match h2_prev
                        # If h1_curr matches h2_prev, h2_curr must match h1_prev
                        # Else we look for single matches
                        
                        # Re-do logic cleanly:
                        m = 0
                        # Try alignment 1: 1-1, 2-2
                        if h1_curr == h1_prev and h2_curr == h2_prev:
                            m = 2
                        # Try alignment 2: 1-2, 2-1
                        elif h1_curr == h2_prev and h2_curr == h1_prev:
                            m = 2
                        else:
                            # Check for single match
                            if h1_curr == h1_prev or h1_curr == h2_prev:
                                m = 1
                            if h2_curr == h1_prev or h2_curr == h2_prev:
                                # Be careful not to double count if both match the SAME prev allele
                                # (e.g. curr=(1,2), prev=(1,1))
                                if m == 0: m = 1
                                elif m == 1:
                                    # We already found a match for h1_curr. 
                                    # Does h2_curr match the *other* one?
                                    # Count raw allele counts intersection
                                    # This is getting complex in logic, simpler math approach:
                                    pass 
                                    
                        # Mathematical approach for matches (robust):
                        # Count(A in B)
                        c_curr_h1 = 1
                        c_curr_h2 = 1
                        if h1_curr == h2_curr: 
                            c_curr_h1 = 2
                            c_curr_h2 = 0
                        
                        c_prev_h1 = 1
                        c_prev_h2 = 1
                        if h1_prev == h2_prev:
                            c_prev_h1 = 2
                            c_prev_h2 = 0
                            
                        # Intersection size
                        shared = 0
                        if h1_curr == h1_prev: shared += min(c_curr_h1, c_prev_h1)
                        if h1_curr == h2_prev and h2_prev != h1_prev: shared += min(c_curr_h1, c_prev_h2)
                        
                        # This loop logic is slow. Let's trust the "Try alignment" heuristic 
                        # because we usually deal with distinct indices in this context.
                        # Using the simple Swap check is usually sufficient for standard HMMs.
                        
                        dist = 0
                        if h1_curr == h1_prev and h2_curr == h2_prev: dist = 0
                        elif h1_curr == h2_prev and h2_curr == h1_prev: dist = 0
                        elif h1_curr == h1_prev or h1_curr == h2_prev or h2_curr == h1_prev or h2_curr == h2_prev: dist = 1
                        else: dist = 2
                        
                        # Calculate Score
                        # If dist is 0, penalty is 0 (Stay)
                        score = current_scores[k_prev] - (dist * penalty)
                        
                        if score > best_incoming_score:
                            best_incoming_score = score
                
                # Add Emission for current site
                next_step_scores[k_curr] = ll_tensor[s, k_curr, i] + best_incoming_score
            
            # Swap buffers
            current_scores = next_step_scores
        
        # 3. Save result (Score at Last Site)
        for k in range(K):
            end_scores[s, k] = current_scores[k]
            
    return end_scores

@njit(parallel=True, fastmath=True)
def viterbi_distance_aware_backward(ll_tensor, penalty, state_definitions):
    """
    Viterbi Backward Scan (Site N -> 0).
    Calculates best score STARTING in state k.
    """
    n_samples, K, n_sites = ll_tensor.shape
    start_scores = np.empty((n_samples, K), dtype=np.float64)
    
    for s in prange(n_samples):
        # 1. Initialize with Last Site
        next_scores = np.empty(K, dtype=np.float64)
        for k in range(K):
            next_scores[k] = ll_tensor[s, k, n_sites - 1]
            
        # 2. Iterate Backward
        for i in range(n_sites - 2, -1, -1):
            
            current_scores_scratch = np.empty(K, dtype=np.float64)
            
            for k_curr in range(K): 
                h1_curr = state_definitions[k_curr, 0]
                h2_curr = state_definitions[k_curr, 1]
                
                best_path_score = -np.inf
                
                for k_next in range(K):
                    h1_next = state_definitions[k_next, 0]
                    h2_next = state_definitions[k_next, 1]
                    
                    # Distance Logic
                    dist = 2
                    if h1_curr == h1_next and h2_curr == h2_next: dist = 0
                    elif h1_curr == h2_next and h2_curr == h1_next: dist = 0
                    elif h1_curr == h1_next or h1_curr == h2_next or h2_curr == h1_next or h2_curr == h2_next: dist = 1
                    
                    score_from_k_next = next_scores[k_next] - (dist * penalty)
                    
                    if score_from_k_next > best_path_score:
                        best_path_score = score_from_k_next
                
                current_scores_scratch[k_curr] = ll_tensor[s, k_curr, i] + best_path_score
            
            next_scores = current_scores_scratch
                    
        # 3. Save result
        for k in range(K):
            start_scores[s, k] = next_scores[k]
            
    return start_scores

#%%

# =============================================================================
# 2. VECTORIZED LIKELIHOOD CALCULATION (BLOCK LEVEL)
# =============================================================================

def viterbi_get_all_block_likelihoods_vectorised(block_samples_data, block_haps, 
    log_likelihood_base=math.e, 
    min_per_site_log_likelihood=-0.5,
    uniform_error_floor_per_site=-0.6,
    min_absolute_block_log_likelihood=-200.0,
    allow_intra_block_recomb=True, 
    recomb_penalty=10.0): # No max_switch_candidates arg needed
    
    # 1. UNPACK DATA
    raw_samples = block_samples_data[0]
    ploidies = np.array(block_samples_data[1])
    
    if len(raw_samples) == 0: return ([], ploidies)

    samples_matrix = np.stack(raw_samples)
    num_samples, num_sites, _ = samples_matrix.shape

    hap_dict = block_haps[3]
    keep_flags = block_haps[1].astype(bool)
    hap_keys = sorted(list(hap_dict.keys()))
    num_haps = len(hap_keys)
    
    if num_haps == 0: return ([{0: 0.0}] * num_samples, ploidies)

    # 2. TENSOR CREATION
    hap_list = [hap_dict[k] for k in hap_keys]
    if len(hap_list) > 0 and (hap_list[0].size == 0):
        haps_tensor = np.zeros((num_haps, 0, 2))
    else:
        haps_tensor = np.array(hap_list)
    
    # 3. MASKING & COMBINATIONS
    samples_masked = samples_matrix[:, keep_flags, :] 
    haps_masked = haps_tensor[:, keep_flags, :]       
    
    h0 = haps_masked[:, :, 0]
    h1 = haps_masked[:, :, 1]
    
    combos_4d = np.stack([
        h0[:, None, :] * h0[None, :, :],
        (h0[:, None, :] * h1[None, :, :]) + (h1[:, None, :] * h0[None, :, :]),
        h1[:, None, :] * h1[None, :, :]
    ], axis=-1)
    
    # Shape: (Num_Pairs, Num_Sites, 3) where Num_Pairs = H^2
    combos_flat = combos_4d.reshape(num_haps * num_haps, combos_4d.shape[2], 3)
    
    # 4. WEIGHTED DISTANCE & LINEAR LIKELIHOOD
    dist_weights = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    combos_weighted = combos_flat @ dist_weights
    
    dist_per_site = np.sum(
        samples_masked[:, np.newaxis, :, :] * combos_weighted[np.newaxis, :, :, :], 
        axis=3
    )
    
    log_penalty = math.log(log_likelihood_base)
    ll_per_site = -(dist_per_site) * log_penalty
    ll_per_site = np.maximum(ll_per_site, min_per_site_log_likelihood)
    
    # 5. INITIAL TOTALS (STATIC)
    total_ll_start = np.sum(ll_per_site, axis=2)
    total_ll_end = total_ll_start.copy()
    
    # --- 6. MULTI-STATE CROSSOVER BOOST (ALL PAIRS + DISTANCE AWARE) ---
    if allow_intra_block_recomb:
        curr_num_sites = ll_per_site.shape[2]
        
        # Only run if we have sites to switch between
        if curr_num_sites > 1:
            
            viterbi_cost = float(abs(recomb_penalty))
            
            # A. Prepare State Definitions
            # We need to map the flat index (0..H^2) back to (h1, h2) for distance calc
            idx_i, idx_j = np.unravel_index(np.arange(num_haps**2), (num_haps, num_haps))
            
            # state_definitions: (K, 2) array of int32
            # Each row K corresponds to the Kth pair in ll_per_site
            state_definitions = np.stack([idx_i, idx_j], axis=1).astype(np.int32)
            
            # Extract full tensor (Samples, Total_Pairs, Sites)
            # Ensure contiguous for Numba
            ll_tensor = np.ascontiguousarray(ll_per_site)
            
            # B. Run Backward Viterbi (Boosts START)
            # This is O(Samples * Sites * Pairs^2). 
            # With H=6 -> Pairs=36 -> Pairs^2=1296. Fast enough.
            scores_start = viterbi_distance_aware_backward(ll_tensor, viterbi_cost, state_definitions)
            
            # C. Run Forward Viterbi (Boosts END)
            scores_end = viterbi_distance_aware_forward(ll_tensor, viterbi_cost, state_definitions)
            
            # Update Matrices (Max of Static vs Switched)
            # Since Viterbi includes "Distance 0" (Stay) path, it is always >= Static.
            total_ll_start = scores_start
            total_ll_end = scores_end

    # 7. GLOBAL FLOORS
    num_active_sites = samples_masked.shape[1]
    if num_active_sites > 0:
        site_based_floor = num_active_sites * uniform_error_floor_per_site
        total_ll_start = np.maximum(total_ll_start, site_based_floor)
        total_ll_end   = np.maximum(total_ll_end, site_based_floor)
        
    total_ll_start = np.maximum(total_ll_start, min_absolute_block_log_likelihood)
    total_ll_end   = np.maximum(total_ll_end, min_absolute_block_log_likelihood)
        
    # 8. NORMALIZATION (Unique Pairs Only)
    idx_i, idx_j = np.unravel_index(np.arange(num_haps**2), (num_haps, num_haps))
    valid_mask = idx_j >= idx_i
    
    unique_start = total_ll_start[:, valid_mask]
    norm_start = unique_start - logsumexp(unique_start, axis=1, keepdims=True)
    
    unique_end = total_ll_end[:, valid_mask]
    norm_end = unique_end - logsumexp(unique_end, axis=1, keepdims=True)
    
    # 9. OUTPUT
    results_start = []
    results_end = []
    keys = [(hap_keys[i], hap_keys[j]) for i, j in zip(idx_i[valid_mask], idx_j[valid_mask])]
    
    for s in range(num_samples):
        results_start.append(dict(zip(keys, norm_start[s])))
        results_end.append(dict(zip(keys, norm_end[s])))
            
    return ((results_start, results_end), ploidies)


def viterbi_multiprocess_all_block_likelihoods(full_samples_data,
                                       sample_sites,
                                       haps_data):
    """
    Wrapper with updated defaults hardcoded.
    """
    tasks = []
    for i in range(len(haps_data)):
        block_samples = analysis_utils.get_sample_data_at_sites_multiple(
            full_samples_data, sample_sites, haps_data[i][0], ploidy_present=True
        )
        tasks.append((block_samples, haps_data[i]))

    with Pool(16) as pool:
        full_blocks_likelihoods = pool.starmap(
            viterbi_get_all_block_likelihoods_vectorised, 
            tasks
        )

    return full_blocks_likelihoods

#%%

# =============================================================================
# 3. VECTORIZED FORWARD-BACKWARD (HMM)
# =============================================================================
def viterbi_get_full_probs_forward_vectorized(sample_data, sample_sites, haps_data,
                                      bidirectional_transition_probs,
                                      full_blocks_likelihoods=None,
                                      space_gap=1):
    
    # --- HANDLING MISSING LIKELIHOODS ---
    if full_blocks_likelihoods is None:
        tasks = []
        for i in range(len(haps_data)):
            block_samples = analysis_utils.get_sample_data_at_sites_multiple(
                sample_data, sample_sites, haps_data[i][0], ploidy_present=True
            )
            tasks.append((block_samples, haps_data[i]))

        with Pool(16) as pool:
            # Use the new Viterbi function
            raw_results = pool.starmap(
                viterbi_get_all_block_likelihoods_vectorised, 
                tasks
            )
        
        # Reformat: Raw is ((StartList, EndList), Ploidy) per block
        # We need a list where index i is (StartDict, EndDict) for this sample
        full_blocks_likelihoods = []
        for block_res in raw_results:
            # block_res[0] is (StartList, EndList)
            # We assume sample_data contained 1 sample, so we take index 0
            start_dict = block_res[0][0][0]
            end_dict = block_res[0][1][0]
            full_blocks_likelihoods.append((start_dict, end_dict))

    transition_probs_dict = bidirectional_transition_probs[0]
    likelihood_numbers = {} 
    shadow_cache = {} 
    ln_2 = math.log(2)

    for i in range(len(haps_data)):
        # Unpack tuple: (Start, End)
        # FORWARD PASS: We enter the block from the Start (Left). 
        # We use the Start Likelihoods.
        (block_lik_dict, _) = full_blocks_likelihoods[i]
        
        hap_keys = sorted(list(haps_data[i][3].keys()))
        n_haps = len(hap_keys)
        key_to_idx = {k: idx for idx, k in enumerate(hap_keys)}
        
        E = np.full((n_haps, n_haps), -np.inf)
        for (u, v), val in block_lik_dict.items():
            u_idx, v_idx = key_to_idx[u], key_to_idx[v]
            E[u_idx, v_idx] = val
            E[v_idx, u_idx] = val
        
        if i < space_gap:
            corrections = np.full((n_haps, n_haps), -ln_2)
            np.fill_diagonal(corrections, 0.0)
            current_matrix = E + corrections
        else:
            earlier_block = i - space_gap
            prev_matrix = shadow_cache[earlier_block]['matrix']
            prev_keys = shadow_cache[earlier_block]['keys']
            
            n_prev = len(prev_keys)
            T = np.full((n_prev, n_haps), -np.inf)
            t_probs_block = transition_probs_dict[earlier_block]
            
            for r, p_key in enumerate(prev_keys):
                for c, u_key in enumerate(hap_keys):
                    lookup = ((earlier_block, p_key), (i, u_key))
                    if lookup in t_probs_block:
                        T[r, c] = math.log(t_probs_block[lookup])
            
            # Matrix Update: T.T @ Prev @ T
            Z = analysis_utils.log_matmul(prev_matrix, T)
            pred_matrix = analysis_utils.log_matmul(T.T, Z)
            current_matrix = pred_matrix + E

        shadow_cache[i] = {'matrix': current_matrix, 'keys': hap_keys}
        
        corrections = np.full((n_haps, n_haps), ln_2)
        np.fill_diagonal(corrections, 0.0)
        final_dict_vals = current_matrix + corrections
        
        result_dict = {}
        for r in range(n_haps):
            for c in range(r, n_haps):
                key = ((i, hap_keys[r]), (i, hap_keys[c]))
                result_dict[key] = final_dict_vals[r, c]
        likelihood_numbers[i] = result_dict

    return likelihood_numbers

def viterbi_get_full_probs_backward_vectorized(sample_data, sample_sites, haps_data,
                           bidirectional_transition_probs,
                           full_blocks_likelihoods=None,
                           space_gap=1):
    
    # --- HANDLING MISSING LIKELIHOODS ---
    if full_blocks_likelihoods is None:
        tasks = []
        for i in range(len(haps_data)):
            block_samples = analysis_utils.get_sample_data_at_sites_multiple(
                sample_data, sample_sites, haps_data[i][0], ploidy_present=True
            )
            tasks.append((block_samples, haps_data[i]))

        with Pool(16) as pool:
            raw_results = pool.starmap(
                viterbi_get_all_block_likelihoods_vectorised, 
                tasks
            )
        
        full_blocks_likelihoods = []
        for block_res in raw_results:
            start_dict = block_res[0][0][0]
            end_dict = block_res[0][1][0]
            full_blocks_likelihoods.append((start_dict, end_dict))

    transition_probs_dict = bidirectional_transition_probs[1]
    likelihood_numbers = {} 
    shadow_cache = {} 
    ln_2 = math.log(2)

    for i in range(len(haps_data)-1, -1, -1):
        
        # Unpack tuple: (Start, End)
        # BACKWARD PASS: We enter the block from the End (Right). 
        # We use the End Likelihoods.
        (_, block_lik_dict) = full_blocks_likelihoods[i]

        hap_keys = sorted(list(haps_data[i][3].keys()))
        n_haps = len(hap_keys)
        key_to_idx = {k: idx for idx, k in enumerate(hap_keys)}
        
        E = np.full((n_haps, n_haps), -np.inf)
        for (u, v), val in block_lik_dict.items():
            u_idx, v_idx = key_to_idx[u], key_to_idx[v]
            E[u_idx, v_idx] = val
            E[v_idx, u_idx] = val
        
        if i >= len(haps_data) - space_gap:
            corrections = np.full((n_haps, n_haps), -ln_2)
            np.fill_diagonal(corrections, 0.0)
            current_matrix = E + corrections
        else:
            future_block = i + space_gap
            future_matrix = shadow_cache[future_block]['matrix']
            future_keys = shadow_cache[future_block]['keys']
            n_fut = len(future_keys)
            
            t_probs_block = transition_probs_dict[future_block]
            T = np.full((n_haps, n_fut), -np.inf)
            
            for r, u_key in enumerate(hap_keys):
                for c, p_key in enumerate(future_keys):
                    # Key Order: ((Start/Future), (End/Current))
                    lookup = ((future_block, p_key), (i, u_key))
                    if lookup in t_probs_block:
                        T[r, c] = math.log(t_probs_block[lookup])
            
            # Matrix Update: T @ Future @ T.T
            Z = analysis_utils.log_matmul(future_matrix, T.T)
            pred_matrix = analysis_utils.log_matmul(T, Z)
            current_matrix = pred_matrix + E

        shadow_cache[i] = {'matrix': current_matrix, 'keys': hap_keys}
        
        corrections = np.full((n_haps, n_haps), ln_2)
        np.fill_diagonal(corrections, 0.0)
        final_dict_vals = current_matrix + corrections
        
        result_dict = {}
        for r in range(n_haps):
            for c in range(r, n_haps):
                key = ((i, hap_keys[r]), (i, hap_keys[c]))
                result_dict[key] = final_dict_vals[r, c]
        likelihood_numbers[i] = result_dict

    return likelihood_numbers


#%%

# =============================================================================
# 4. BEAM SEARCH & STRATIFIED DIVERSITY
# =============================================================================

class FastMesh:
    def __init__(self, haps_data, full_mesh):
        self.haps_data = haps_data
        self.num_blocks = len(haps_data)
        self.mappings = [] 
        self.reverse_mappings = []
        
        for i in range(self.num_blocks):
            keys = sorted(list(haps_data[i][3].keys()))
            self.mappings.append({k: idx for idx, k in enumerate(keys)})
            self.reverse_mappings.append(keys)
            
        self.cache = {}
        for gap in full_mesh.keys():
            self.cache[gap] = [{}, {}]
            # Forward
            for i in full_mesh[gap][0].keys():
                if i + gap >= self.num_blocks: continue
                prev_map, curr_map = self.mappings[i], self.mappings[i+gap]
                mat = np.full((len(prev_map), len(curr_map)), -np.inf)
                for (k_prev, k_curr), val in full_mesh[gap][0][i].items():
                    u, v = k_prev[1], k_curr[1]
                    if u in prev_map and v in curr_map:
                        mat[prev_map[u], curr_map[v]] = math.log(val)
                self.cache[gap][0][i] = mat
            # Backward
            for i in full_mesh[gap][1].keys():
                if i - gap < 0: continue
                curr_map, prev_map = self.mappings[i], self.mappings[i-gap]
                mat = np.full((len(curr_map), len(prev_map)), -np.inf)
                for (k_curr, k_prev), val in full_mesh[gap][1][i].items():
                    u, v = k_curr[1], k_prev[1]
                    if u in curr_map and v in prev_map:
                        mat[curr_map[u], prev_map[v]] = math.log(val)
                self.cache[gap][1][i] = mat

    def get_score_matrix(self, gap, direction, block_idx):
        try: return self.cache[gap][direction][block_idx]
        except KeyError: return None
    def get_all_keys(self, block_idx): return self.reverse_mappings[block_idx]
    def to_key(self, block_idx, dense_idx): return self.reverse_mappings[block_idx][dense_idx]

def prune_beam_stratified(candidates, target_size, 
                          min_diff_percent=0.02, min_diff_blocks=2, 
                          tip_length=5, min_per_tip=5):
    """
    Selects candidates by enforcing diversity across TIPS first, then Scores.
    Enforces BOTH percentage difference and absolute block count difference.
    """
    if not candidates: return []
    
    # 1. Bucket by Tip
    candidates_by_tip = defaultdict(list)
    for path_list, score in candidates:
        path_arr = np.array(path_list)
        tip = tuple(path_arr) if len(path_arr) <= tip_length else tuple(path_arr[-tip_length:])
        candidates_by_tip[tip].append((path_list, score))
        
    kept_candidates = []
    
    # Helper to check diversity against a list of paths
    def is_distinct(new_path, existing_matrix):
        if len(existing_matrix) == 0: return True
        # Vectorized check
        mismatches = (existing_matrix != np.array(new_path))
        diff_counts = np.sum(mismatches, axis=1)
        
        # Check absolute count
        if np.min(diff_counts) < min_diff_blocks: return False
        # Check percentage (for long chains)
        if np.min(diff_counts / len(new_path)) < min_diff_percent: return False
        return True

    # 2. Enforce Minimum Representation per Tip
    remaining = []
    for tip, group in candidates_by_tip.items():
        tip_kept = []
        for path, score in group:
            if len(tip_kept) < min_per_tip:
                if is_distinct(path, np.array(tip_kept)):
                    kept_candidates.append((path, score))
                    tip_kept.append(path)
                else: remaining.append((path, score))
            else: remaining.append((path, score))

    # 3. Global Backfill
    if len(kept_candidates) < target_size:
        remaining.sort(key=lambda x: x[1], reverse=True)
        all_kept = np.array([c[0] for c in kept_candidates]) if kept_candidates else np.array([])
        
        for path, score in remaining:
            if len(kept_candidates) >= target_size: break
            if len(all_kept) == 0 or is_distinct(path, all_kept):
                kept_candidates.append((path, score))
                all_kept = np.array([path]) if len(all_kept)==0 else np.vstack([all_kept, np.array(path)])
    
    kept_candidates.sort(key=lambda x: x[1], reverse=True)
    return kept_candidates[:target_size]

def run_beam_search_initial_diverse(fast_mesh, num_candidates=200, 
                                    diversity_diff_percent=0.02, min_diff_blocks=3, tip_length=5):
    
    num_blocks = fast_mesh.num_blocks
    first_block_keys = fast_mesh.get_all_keys(0)
    beam = [([i], 0.0) for i in range(len(first_block_keys))]
    
    for i in range(1, num_blocks):
        candidates = []
        valid_gaps = [g for g in fast_mesh.cache.keys() if i - g >= 0]
        
        gap_matrices = []
        for g in valid_gaps:
            mat = fast_mesh.get_score_matrix(g, 0, i - g)
            scale = math.sqrt(g)
            gap_matrices.append((g, mat, scale))
            
        num_curr_options = len(fast_mesh.get_all_keys(i))
        
        for path, score in beam:
            next_scores = np.full(num_curr_options, score)
            for gap, mat, scale in gap_matrices:
                prev_idx = path[i - gap]
                if mat is not None:
                    next_scores += (mat[prev_idx, :] * scale)
            
            # Expansion: Keep top 20 to ensure rare founders survive
            k_best = min(20, len(next_scores))
            best_ext_indices = np.argpartition(next_scores, -k_best)[-k_best:]
            
            for next_idx in best_ext_indices:
                new_score = next_scores[next_idx]
                if new_score > -np.inf:
                    candidates.append((path + [next_idx], new_score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Adaptive Threshold: Start loose, get stricter
        curr_min_blocks = min_diff_blocks if i > 20 else 0
        
        beam = prune_beam_stratified(
            candidates, target_size=num_candidates, 
            min_diff_percent=diversity_diff_percent,
            min_diff_blocks=curr_min_blocks,
            tip_length=tip_length,
            min_per_tip=5
        )

    return beam

def run_beam_refinement(beam, fast_mesh, num_candidates, direction="backward", 
                        diversity_diff_percent=0.02, min_diff_blocks=3, tip_length=5,
                        allowed_gaps=None):
    """
    Refinement Pass using STRATIFIED Pruning and optional Gap Filtering.
    """
    num_blocks = fast_mesh.num_blocks
    if direction == "backward": indices = range(num_blocks - 1, -1, -1)
    else: indices = range(0, num_blocks)
        
    for i in indices:
        new_beam_candidates = {} 
        
        available_gaps = fast_mesh.cache.keys()
        if allowed_gaps is not None:
            available_gaps = [g for g in available_gaps if g in allowed_gaps]

        relevant_matrices = []
        for g in available_gaps:
            if i - g >= 0:
                mat = fast_mesh.get_score_matrix(g, 0, i - g)
                if mat is not None: relevant_matrices.append(('back', g, mat, math.sqrt(g)))
        for g in available_gaps:
            if i + g < num_blocks:
                mat = fast_mesh.get_score_matrix(g, 0, i) 
                if mat is not None: relevant_matrices.append(('fwd', g, mat, math.sqrt(g)))

        num_options_at_i = len(fast_mesh.get_all_keys(i))
        
        for path_list, current_total_score in beam:
            path = np.array(path_list)
            current_val = path[i]
            
            current_local_score = 0.0
            potential_local_scores = np.zeros(num_options_at_i)
            
            for mode, gap, mat, scale in relevant_matrices:
                if mode == 'back':
                    prev_val = path[i - gap]
                    current_local_score += (mat[prev_val, current_val] * scale)
                    potential_local_scores += (mat[prev_val, :] * scale)
                elif mode == 'fwd':
                    next_val = path[i + gap]
                    current_local_score += (mat[current_val, next_val] * scale)
                    potential_local_scores += (mat[:, next_val] * scale)

            base_score = current_total_score - current_local_score
            new_total_scores = base_score + potential_local_scores
            
            k_best = min(5, len(new_total_scores))
            if k_best > 0:
                best_indices = np.argpartition(new_total_scores, -k_best)[-k_best:]
                for new_val in best_indices:
                    new_score = new_total_scores[new_val]
                    if new_score > -np.inf:
                        new_path = list(path)
                        new_path[i] = new_val
                        t_path = tuple(new_path)
                        if t_path not in new_beam_candidates:
                            new_beam_candidates[t_path] = new_score
                        elif new_score > new_beam_candidates[t_path]:
                            new_beam_candidates[t_path] = new_score

        candidates = sorted([(list(p), s) for p, s in new_beam_candidates.items()], key=lambda x: x[1], reverse=True)
        
        beam = prune_beam_stratified(
            candidates, target_size=num_candidates, 
            min_diff_percent=diversity_diff_percent,
            min_diff_blocks=min_diff_blocks,
            tip_length=tip_length,
            min_per_tip=5
        )
        
    return beam

def convert_mesh_to_haplotype_diverse(full_samples_data, full_sites,
                                      haps_data, full_mesh,
                                      num_candidates=200,
                                      diversity_diff_percent=0.02,
                                      min_diff_blocks=3,
                                      tip_length=5):
    
    fast_mesh = FastMesh(haps_data, full_mesh)
    
    # 1. Initial Construction
    beam_results = run_beam_search_initial_diverse(
        fast_mesh, num_candidates, diversity_diff_percent, min_diff_blocks, tip_length
    )
    
    # 2. Refinement (Backward & Forward)
    beam_results = run_beam_refinement(
        beam_results, fast_mesh, num_candidates, "backward", 
        diversity_diff_percent, min_diff_blocks, tip_length
    )
    beam_results = run_beam_refinement(
        beam_results, fast_mesh, num_candidates, "forward", 
        diversity_diff_percent, min_diff_blocks, tip_length
    )
    
    # 3. Gap-1 Polishing (Fix local noise)
    print("Running Gap-1 Polishing...")
    beam_results = run_beam_refinement(
        beam_results, fast_mesh, num_candidates, "backward", 
        diversity_diff_percent, min_diff_blocks, tip_length, allowed_gaps=[1]
    )
    beam_results = run_beam_refinement(
        beam_results, fast_mesh, num_candidates, "forward", 
        diversity_diff_percent, min_diff_blocks, tip_length, allowed_gaps=[1]
    )
    
    return beam_results, fast_mesh

# =============================================================================
# 5. DIVERSE SELECTION & HELPERS
# =============================================================================

def get_max_contiguous_difference(path_a, path_b):
    arr_a = np.array(path_a)
    arr_b = np.array(path_b)
    mismatches = (arr_a != arr_b)
    max_run = 0
    current_run = 0
    for is_diff in mismatches:
        if is_diff: current_run += 1
        else:
            if current_run > max_run: max_run = current_run
            current_run = 0
    if current_run > max_run: max_run = current_run
    return max_run

def select_diverse_founders(beam_results, fast_mesh, haps_data, 
                            num_founders=6, 
                            min_total_diff_percent=0.10, 
                            min_contiguous_diff=5):
    """
    Selects founders that are distinct based on BIOLOGICAL logic (Run Length).
    """
    final_founders = []
    
    # 1. First founder
    best_path, best_score = beam_results[0]
    final_founders.append(best_path)
    print(f"Founder 1 selected. Score: {best_score:.2f}")
    
    # 2. Scan remaining candidates
    current_beam_idx = 1
    while len(final_founders) < num_founders and current_beam_idx < len(beam_results):
        candidate_path, candidate_score = beam_results[current_beam_idx]
        current_beam_idx += 1
        
        is_distinct = True
        for existing_founder in final_founders:
            # Check Metrics
            total_diff = np.sum(np.array(candidate_path) != np.array(existing_founder)) / len(candidate_path)
            max_run = get_max_contiguous_difference(candidate_path, existing_founder)
            
            if max_run < min_contiguous_diff:
                is_distinct = False
                break
            if total_diff < min_total_diff_percent:
                is_distinct = False
                break
        
        if is_distinct:
            print(f"Founder {len(final_founders)+1} selected. Score: {candidate_score:.2f}.")
            final_founders.append(candidate_path)
            
    if len(final_founders) < num_founders:
        print(f"Warning: Only found {len(final_founders)} distinct founders.")

    # 3. Reconstruct Data
    reconstructed_data = []
    for path_indices in final_founders:
        combined_positions = []
        combined_haplotype = []
        for i, dense_idx in enumerate(path_indices):
            hap_key = fast_mesh.to_key(i, dense_idx)
            combined_positions.extend(haps_data[i][0])
            combined_haplotype.extend(haps_data[i][3][hap_key])
        reconstructed_data.append((np.array(combined_positions), np.array(combined_haplotype), path_indices))
        
    return reconstructed_data

# =============================================================================
# 6. HELPERS FOR MAIN SCRIPT
# =============================================================================

def multiprocess_all_block_likelihoods(full_samples_data, sample_sites, haps_data):
    """Wrapper that calls the optimized vectorised function"""
    tasks = []
    for i in range(len(haps_data)):
        block_samples = analysis_utils.get_sample_data_at_sites_multiple(
            full_samples_data, sample_sites, haps_data[i][0], ploidy_present=True
        )
        tasks.append((block_samples, haps_data[i]))

    with Pool(32) as pool:
        full_blocks_likelihoods = pool.starmap(get_all_block_likelihoods_vectorised, tasks)
    return full_blocks_likelihoods