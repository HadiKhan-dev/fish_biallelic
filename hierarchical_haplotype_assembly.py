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
def viterbi_general_k(ll_tensor, penalty):
    """
    Calculates the max score allowing multiple switches between K candidates.
    This allows transitions like A -> B -> C within a single block.
    
    ll_tensor: (n_samples, K, n_sites) 
               Log-likelihoods for the top K candidates at every site.
    penalty:   Positive float cost per switch.
    
    Returns:
    final_scores: (n_samples, K)
                  The score of the best path ENDING in state k for each sample.
    """
    n_samples, K, n_sites = ll_tensor.shape
    
    # Output: Best score ending in state k
    final_scores = np.empty((n_samples, K), dtype=np.float64)
    
    # Parallelize over samples
    for s in prange(n_samples):
        
        # 1. Initialize current scores (Site 0)
        # Manual copy is often faster than slicing in Numba
        current_scores = np.empty(K, dtype=np.float64)
        for k in range(K):
            current_scores[k] = ll_tensor[s, k, 0]
            
        # 2. Iterate through sites
        for i in range(1, n_sites):
            
            # Find the Global Best score from the previous step
            # This is the best score we can switch FROM.
            best_prev_score = -np.inf
            for k in range(K):
                if current_scores[k] > best_prev_score:
                    best_prev_score = current_scores[k]
            
            # The score one gets if they switch *into* a state
            switch_score_base = best_prev_score - penalty
            
            # Update every state
            for k in range(K):
                stay_score = current_scores[k]
                emission = ll_tensor[s, k, i]
                
                # Logic: Either we extended the path staying in k,
                # or we jumped into k from the best previous state.
                if stay_score > switch_score_base:
                    current_scores[k] = stay_score + emission
                else:
                    current_scores[k] = switch_score_base + emission
        
        # 3. Save final scores for this sample
        for k in range(K):
            final_scores[s, k] = current_scores[k]
            
    return final_scores

def log_matmul(A, B):
    """
    Performs Matrix Multiplication in Log-Space.
    Equivalent to C = A @ B, but for log-probabilities.
    C[i, j] = logsumexp(A[i, :] + B[:, j])
    """
    return logsumexp(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)

# =============================================================================
# 2. VECTORIZED LIKELIHOOD CALCULATION (BLOCK LEVEL)
# =============================================================================

def get_all_block_likelihoods_vectorised(block_samples_data, block_haps, 
    log_likelihood_base=math.e, 
    min_per_site_log_likelihood=-0.5, # Cap penalty (20 errors ~ -10)
    uniform_error_floor_per_site=-0.6,
    min_absolute_block_log_likelihood=-200.0,
    allow_intra_block_recomb=True, 
    recomb_penalty=10.0,
    max_switch_candidates=10):
    """
    Calculates likelihood of samples matching haplotypes.
    Includes Generalized Viterbi boost for intra-block recombination.
    """
    
    # 1. UNPACK DATA
    raw_samples = block_samples_data[0]
    ploidies = np.array(block_samples_data[1])
    
    if len(raw_samples) == 0:
        return ([], ploidies)

    samples_matrix = np.stack(raw_samples) # (S, Sites, 3)
    num_samples, num_sites, _ = samples_matrix.shape

    hap_dict = block_haps[3]
    keep_flags = block_haps[1].astype(bool)
    hap_keys = sorted(list(hap_dict.keys()))
    num_haps = len(hap_keys)
    
    if num_haps == 0:
        return ([{0: 0.0}] * num_samples, ploidies)

    # 2. ROBUST TENSOR CREATION (Handle Empty Blocks)
    hap_list = [hap_dict[k] for k in hap_keys]
    if len(hap_list) > 0 and (hap_list[0].size == 0):
        # Force correct 3D shape: (Num_Haps, 0, 2)
        haps_tensor = np.zeros((num_haps, 0, 2))
    else:
        haps_tensor = np.array(hap_list)
    
    # 3. MASKING
    samples_masked = samples_matrix[:, keep_flags, :] 
    haps_masked = haps_tensor[:, keep_flags, :]       
    
    # 4. GENERATE DIPLOID COMBINATIONS
    h0 = haps_masked[:, :, 0]
    h1 = haps_masked[:, :, 1]
    
    combos_4d = np.stack([
        h0[:, None, :] * h0[None, :, :], # 00
        (h0[:, None, :] * h1[None, :, :]) + (h1[:, None, :] * h0[None, :, :]), # 01
        h1[:, None, :] * h1[None, :, :] # 11
    ], axis=-1)
    
    combos_flat = combos_4d.reshape(-1, combos_4d.shape[2], 3)
    
    # 5. WEIGHTED DISTANCE CALCULATION
    dist_weights = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    combos_weighted = combos_flat @ dist_weights
    
    dist_per_site = np.sum(
        samples_masked[:, np.newaxis, :, :] * combos_weighted[np.newaxis, :, :, :], 
        axis=3
    )
    
    # 6. LOG LIKELIHOOD (PER SITE)
    log_penalty = math.log(log_likelihood_base)
    # Use Linear distance (robust) and Per-Site Cap
    ll_per_site = -(dist_per_site) * log_penalty
    ll_per_site = np.maximum(ll_per_site, min_per_site_log_likelihood)
    
    # 7. STATIC TOTALS
    total_ll_matrix = np.sum(ll_per_site, axis=2)
    
    # 8. MULTI-STATE CROSSOVER BOOST (GENERALIZED)
    if allow_intra_block_recomb:
        curr_num_sites = ll_per_site.shape[2]
        num_total_pairs = total_ll_matrix.shape[1]
        
        # Only run if we have sites and enough pairs to make switching meaningful
        if curr_num_sites > 1 and num_total_pairs > 1:
            
            K = min(max_switch_candidates, num_total_pairs)
            viterbi_cost = float(abs(recomb_penalty))
            
            # 1. Identify Top K pairs for each sample
            top_k_indices = np.argpartition(total_ll_matrix, -K, axis=1)[:, -K:]
            
            # 2. Extract Likelihood Tensor for these K candidates
            row_indices = np.arange(num_samples)[:, None]
            top_k_ll_per_site = ll_per_site[row_indices, top_k_indices, :]
            
            # Ensure it is contiguous for Numba speed
            top_k_ll_per_site = np.ascontiguousarray(top_k_ll_per_site)
            
            # 3. Run Generalized Viterbi
            viterbi_scores = viterbi_general_k(top_k_ll_per_site, viterbi_cost)
            
            # 4. Update the Main Matrix
            # Boost the static score with the Viterbi score if better
            current_vals = total_ll_matrix[row_indices, top_k_indices]
            
            total_ll_matrix[row_indices, top_k_indices] = np.maximum(
                current_vals,
                viterbi_scores
            )

    # 9. GLOBAL FLOORS
    num_active_sites = samples_masked.shape[1]
    if num_active_sites > 0:
        site_based_floor = num_active_sites * uniform_error_floor_per_site
        total_ll_matrix = np.maximum(total_ll_matrix, site_based_floor)
        
    total_ll_matrix = np.maximum(total_ll_matrix, min_absolute_block_log_likelihood)
        
    # 10. EXACT DIPLOID NORMALIZATION (Upper Triangle Only)
    idx_i, idx_j = np.unravel_index(np.arange(num_haps**2), (num_haps, num_haps))
    valid_mask = idx_j >= idx_i
    unique_ll_matrix = total_ll_matrix[:, valid_mask]
    
    norm_factors = logsumexp(unique_ll_matrix, axis=1, keepdims=True)
    final_normalized_matrix = unique_ll_matrix - norm_factors
    
    # 11. OUTPUT
    results = []
    keys = [(hap_keys[i], hap_keys[j]) for i, j in zip(idx_i[valid_mask], idx_j[valid_mask])]
    for s in range(num_samples):
        results.append(dict(zip(keys, final_normalized_matrix[s])))
            
    return (results, ploidies)

# =============================================================================
# 3. VECTORIZED FORWARD-BACKWARD (HMM)
# =============================================================================

def get_full_probs_forward_vectorized(sample_data, sample_sites, haps_data,
                                      bidirectional_transition_probs,
                                      full_blocks_likelihoods=None,
                                      space_gap=1):
    
    if full_blocks_likelihoods is None:
        raise ValueError("Must provide full_blocks_likelihoods for speed")

    transition_probs_dict = bidirectional_transition_probs[0]
    likelihood_numbers = {} 
    shadow_cache = {} 
    ln_2 = math.log(2)

    for i in range(len(haps_data)):
        (block_lik_dict, _) = full_blocks_likelihoods[i]
        if isinstance(block_lik_dict, list): block_lik_dict = block_lik_dict[0]

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
            Z = log_matmul(prev_matrix, T)
            pred_matrix = log_matmul(T.T, Z)
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

def get_full_probs_backward_vectorized(sample_data, sample_sites, haps_data,
                           bidirectional_transition_probs,
                           full_blocks_likelihoods=None,
                           space_gap=1):
    
    if full_blocks_likelihoods is None:
        raise ValueError("Must provide full_blocks_likelihoods for speed")

    transition_probs_dict = bidirectional_transition_probs[1]
    likelihood_numbers = {} 
    shadow_cache = {} 
    ln_2 = math.log(2)

    for i in range(len(haps_data)-1, -1, -1):
        (block_lik_dict, _) = full_blocks_likelihoods[i]
        if isinstance(block_lik_dict, list): block_lik_dict = block_lik_dict[0]

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
            Z = log_matmul(future_matrix, T.T)
            pred_matrix = log_matmul(T, Z)
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