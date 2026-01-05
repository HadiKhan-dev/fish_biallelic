import numpy as np
from multiprocess import Pool
import warnings
import pandas as pd

import analysis_utils

warnings.filterwarnings("ignore")
np.seterr(divide='ignore',invalid="ignore")

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("WARNING: Numba not found. Model selection will be extremely slow.")
    # Dummy decorator
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator
    prange = range

#%% --- CORE MATCHING FUNCTIONS ---

def match_best_vectorised(haps_dict, diploids, keep_flags=None):
    """
    Vectorized matching of diploid samples to haplotype pairs.
    Uses Matrix Multiplication for high performance.
    """
    diploids = np.array(diploids) 
    num_samples, total_sites, _ = diploids.shape
    
    if keep_flags is None:
        keep_flags = slice(None)
    elif keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags, dtype=bool)
        
    diploids_masked = diploids[:, keep_flags, :]
    masked_sites = diploids_masked.shape[1]
    
    if masked_sites == 0:
        return ([], {}, np.zeros(num_samples))
    
    diploids_flat = diploids_masked.reshape(num_samples, -1)

    hap_keys = list(haps_dict.keys())
    num_haps = len(hap_keys)
    
    if num_haps == 0:
        return ([], {}, np.zeros(num_samples))
    
    # Stack haps: (Num_Haps, Masked_Sites, 2)
    hap_tensor = np.array([haps_dict[k][keep_flags] for k in hap_keys])

    p0 = hap_tensor[:, :, 0] 
    p1 = hap_tensor[:, :, 1]

    # Broadcasting: (Num_Haps, Num_Haps, Masked_Sites)
    # This generates the probability for every possible combination (i, j)
    prob_00 = p0[:, None, :] * p0[None, :, :]
    prob_11 = p1[:, None, :] * p1[None, :, :]
    prob_01 = (p0[:, None, :] * p1[None, :, :]) + (p1[:, None, :] * p0[None, :, :])

    combinations_4d = np.stack([prob_00, prob_01, prob_11], axis=-1)
    # Reshape to (N*N, Sites, 3)
    combinations_list = combinations_4d.reshape(-1, masked_sites, 3)

    # Calculate expected distance for each pair against [0,1,2] states
    dist_weights = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    combinations_weighted = combinations_list @ dist_weights
    combinations_weighted_flat = combinations_weighted.reshape(-1, masked_sites * 3)

    # Matrix Mult: (Samples, Features) @ (Combinations, Features).T
    dists = diploids_flat @ combinations_weighted_flat.T
    dists *= (100.0 / masked_sites)

    best_indices_flat = np.argmin(dists, axis=1)
    best_errors = dists[np.arange(num_samples), best_indices_flat]

    # Map flat index back to (i, j)
    idx_grid_i, idx_grid_j = np.indices((num_haps, num_haps))
    idx_grid_i = idx_grid_i.flatten()
    idx_grid_j = idx_grid_j.flatten()

    best_parents_i = idx_grid_i[best_indices_flat]
    best_parents_j = idx_grid_j[best_indices_flat]

    all_used = np.concatenate([best_parents_i, best_parents_j])
    unique_idx, counts = np.unique(all_used, return_counts=True)
    
    haps_usage = {k: 0 for k in hap_keys}
    for idx, count in zip(unique_idx, counts):
        haps_usage[hap_keys[idx]] = count

    dips_matches = [
        ((hap_keys[p1], hap_keys[p2]), err)
        for p1, p2, err in zip(best_parents_i, best_parents_j, best_errors)
    ]

    return (dips_matches, haps_usage, best_errors)

@njit(parallel=True, fastmath=True)
def viterbi_constrained_k(dist_tensor, max_k, state_defs):
    """
    Finds the best path with AT MOST 'max_k' haplotype switches.
    """
    n_samples, n_pairs, n_sites = dist_tensor.shape
    budget_size = max_k + 1
    
    final_costs = np.empty(n_samples, dtype=np.float64)
    # CRITICAL FIX: Changed int16 to int32. 
    # int16 overflows if n_pairs > 32767 (approx >256 haplotypes).
    backpointers = np.empty((n_samples, n_sites, n_pairs, budget_size), dtype=np.int32)
    
    for s in prange(n_samples):
        curr_dp = np.full((n_pairs, budget_size), np.inf, dtype=np.float64)
        prev_dp = np.full((n_pairs, budget_size), np.inf, dtype=np.float64)
        
        # Initialize t=0
        for p in range(n_pairs):
            prev_dp[p, 0] = dist_tensor[s, p, 0]
            
        for t in range(1, n_sites):
            emission = dist_tensor[s, :, t]
            curr_dp[:] = np.inf
            
            for k_prev_used in range(budget_size):
                for p_prev in range(n_pairs):
                    cost_prev = prev_dp[p_prev, k_prev_used]
                    if cost_prev == np.inf: continue
                    
                    h1_prev = state_defs[p_prev, 0]
                    h2_prev = state_defs[p_prev, 1]
                    
                    for p_curr in range(n_pairs):
                        h1_curr = state_defs[p_curr, 0]
                        h2_curr = state_defs[p_curr, 1]
                        
                        # Calculate switch cost (hamming distance of parents)
                        dist = 2
                        # Case: Identical pair (0 switch)
                        if h1_curr == h1_prev and h2_curr == h2_prev: dist = 0
                        elif h1_curr == h2_prev and h2_curr == h1_prev: dist = 0
                        # Case: One parent shared (1 switch)
                        elif (h1_curr == h1_prev or h1_curr == h2_prev or 
                              h2_curr == h1_prev or h2_curr == h2_prev): dist = 1
                        
                        k_new = k_prev_used + dist
                        
                        if k_new < budget_size:
                            new_total = cost_prev + emission[p_curr]
                            if new_total < curr_dp[p_curr, k_new]:
                                curr_dp[p_curr, k_new] = new_total
                                backpointers[s, t, p_curr, k_new] = p_prev

            # Swap buffers
            prev_dp[:] = curr_dp[:]

        # Traceback
        best_cost = np.inf
        best_p = -1
        best_k = -1
        
        for p in range(n_pairs):
            for k in range(budget_size):
                if prev_dp[p, k] < best_cost:
                    best_cost = prev_dp[p, k]
                    best_p = p
                    best_k = k
        
        final_costs[s] = best_cost
        
        curr_p = best_p
        curr_k = best_k
        
        path_storage = np.empty(n_sites, dtype=np.int32)
        if best_p != -1:
            path_storage[n_sites-1] = curr_p
            
            for t in range(n_sites-1, 0, -1):
                prev_p = backpointers[s, t, curr_p, curr_k]
                
                h1_curr = state_defs[curr_p, 0]
                h2_curr = state_defs[curr_p, 1]
                h1_prev = state_defs[prev_p, 0]
                h2_prev = state_defs[prev_p, 1]
                
                dist = 2
                if h1_curr == h1_prev and h2_curr == h2_prev: dist = 0
                elif h1_curr == h2_prev and h2_curr == h1_prev: dist = 0
                elif (h1_curr == h1_prev or h1_curr == h2_prev or 
                      h2_curr == h1_prev or h2_curr == h2_prev): dist = 1
                
                curr_k = curr_k - dist
                curr_p = prev_p
                path_storage[t-1] = curr_p
        
        # Save path to backpointers array for return (reusing memory)
        for t in range(n_sites):
            backpointers[s, t, 0, 0] = path_storage[t]

    return final_costs, backpointers[:, :, 0, 0]

def match_best_k_limited(haps_dict, diploids, keep_flags=None, max_recombinations=2):
    """
    Matches samples to the best haplotype pair allowing AT MOST 'max_recombinations'.
    """
    diploids = np.array(diploids) 
    num_samples, total_sites, _ = diploids.shape
    
    if keep_flags is None: keep_flags = slice(None)
    elif keep_flags.dtype != bool: keep_flags = np.array(keep_flags, dtype=bool)
        
    diploids_masked = diploids[:, keep_flags, :]
    masked_sites = diploids_masked.shape[1]
    
    if masked_sites == 0:
        return ([], {}, np.zeros(num_samples))

    hap_keys = list(haps_dict.keys())
    num_haps = len(hap_keys)
    hap_tensor = np.array([haps_dict[k][keep_flags] for k in hap_keys])

    # Triu indices creates the list of unique pairs (excluding order)
    idx_i, idx_j = np.triu_indices(num_haps)
    state_defs = np.stack([idx_i, idx_j], axis=1).astype(np.int32)
    
    h0 = hap_tensor[idx_i, :, 0]
    h1 = hap_tensor[idx_i, :, 1]
    h2_0 = hap_tensor[idx_j, :, 0]
    h2_1 = hap_tensor[idx_j, :, 1]
    
    # Precompute genotype probs for all pairs
    pair_genotypes = np.stack([
        h0 * h2_0, 
        (h0 * h2_1) + (h1 * h2_0),
        h1 * h2_1
    ], axis=-1)
    
    dist_weights = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    weighted_pairs = pair_genotypes @ dist_weights
    
    # Calculate raw errors (Dist Tensor)
    dist_tensor = np.einsum('nls,pls->npl', diploids_masked, weighted_pairs)
    
    best_costs, best_paths = viterbi_constrained_k(
        dist_tensor.astype(np.float64), 
        int(max_recombinations), 
        state_defs
    )
    
    normalized_errors = (best_costs / masked_sites) * 100.0
    
    dips_matches = []
    
    for s in range(num_samples):
        path = best_paths[s].astype(np.int32)
        # Identify indices where the state (pair index) changes
        change_mask = np.insert(np.diff(path) != 0, 0, True)
        path_sequence_indices = path[change_mask]
        
        path_resolved = []
        for pair_idx in path_sequence_indices:
            h1_idx, h2_idx = state_defs[pair_idx]
            path_resolved.append((hap_keys[h1_idx], hap_keys[h2_idx]))
            
        dips_matches.append((path_resolved, normalized_errors[s]))

    used_hap_indices = state_defs[best_paths.astype(np.int32)].flatten()
    u_idx, u_counts = np.unique(used_hap_indices, return_counts=True)
    haps_usage = {k: 0 for k in hap_keys}
    for idx, count in zip(u_idx, u_counts):
        haps_usage[hap_keys[idx]] = count / masked_sites

    return (dips_matches, haps_usage, normalized_errors)

#%% --- HELPER FUNCTIONS FOR BLOCK HAPLOTYPES ---

def combined_best_hap_matches(block_result):
    if hasattr(block_result, 'haplotypes'):
        reads_array = block_result.reads_count_matrix
        haps = block_result.haplotypes
        keep_flags = getattr(block_result, 'keep_flags', None)
    else:
        # Assuming tuple structure (pos, keep_flags, reads, haps)
        keep_flags = block_result[1]
        reads_array = block_result[2]
        haps = block_result[3]
        

    (site_priors, actual_probs) = analysis_utils.reads_to_probabilities(reads_array)
    
    matches = match_best_vectorised(haps, actual_probs, keep_flags=keep_flags)
    return matches

def get_best_matches_all_blocks(block_results, num_processes=16):
    with Pool(processes=num_processes) as pool:
        processing_results = pool.map(combined_best_hap_matches, block_results)
    return processing_results

#%% --- MATCHING COMPARISON & STITCHING HELPERS ---

def relative_haplotype_usage(first_hap, first_matches, second_matches):
    use_indices = []
    match_list_1 = first_matches[0]
    
    for sample_idx, (parents, _) in enumerate(match_list_1):
        if first_hap in parents:
            use_indices.append(sample_idx)
            
    second_usages = {}
    match_list_2 = second_matches[0]
    
    for sample_idx in use_indices:
        parents_2, _ = match_list_2[sample_idx]
        
        for parent in parents_2:
            second_usages[parent] = second_usages.get(parent, 0) + 1
    
    return dict(sorted(second_usages.items(), key=lambda item: item[1]))

def hap_matching_comparison(haps_data, matches_data, first_block_index, second_block_index):
    forward_scores = {}
    backward_scores = {}
        
    b1 = haps_data[first_block_index]
    b2 = haps_data[second_block_index]
    
    first_haps_dict = b1.haplotypes if hasattr(b1, 'haplotypes') else b1[3]
    second_haps_dict = b2.haplotypes if hasattr(b2, 'haplotypes') else b2[3]
        
    first_matches = matches_data[first_block_index]
    second_matches = matches_data[second_block_index]
        
    for hap in first_haps_dict.keys():
        hap_usages = relative_haplotype_usage(hap, first_matches, second_matches)
        total_matches = sum(hap_usages.values())
        if total_matches == 0: continue
        
        hap_percs = {x: 100 * count / total_matches for x, count in hap_usages.items()}
            
        for other_hap in second_haps_dict.keys():
            perc = hap_percs.get(other_hap, 0)
            scaled_val = 100 * (min(1, 2 * perc / 100))**2
            key = ((first_block_index, hap), (second_block_index, other_hap))
            forward_scores[key] = scaled_val
            
    for hap in second_haps_dict.keys():
        hap_usages = relative_haplotype_usage(hap, second_matches, first_matches)
        total_matches = sum(hap_usages.values())
        if total_matches == 0: continue
        
        hap_percs = {x: 100 * count / total_matches for x, count in hap_usages.items()}
            
        for other_hap in first_haps_dict.keys():
            perc = hap_percs.get(other_hap, 0)
            scaled_val = 100 * (min(1, 2 * perc / 100))**2
            key = ((first_block_index, other_hap), (second_block_index, hap))
            backward_scores[key] = scaled_val
        
    return (forward_scores, backward_scores)

def get_block_hap_similarities(block_result):
    scores = []
    
    if hasattr(block_result, 'haplotypes'):
        hap_vals = block_result.haplotypes
        flags = getattr(block_result, 'keep_flags', None)
    else:
        hap_vals = block_result[3]
        flags = block_result[1]
        
    if flags is None:
        any_key = next(iter(hap_vals))
        flags = np.ones(len(hap_vals[any_key]), dtype=bool)
    else:
        flags = np.array(flags, dtype=bool)
    
    keys = sorted(hap_vals.keys())
    
    for i in keys:
        row_scores = []
        for j in keys:
            if j < i:
                row_scores.append(0)
            else:
                first_hap = hap_vals[i][flags]
                second_hap = hap_vals[j][flags]
                hap_len = len(first_hap)
                
                if hap_len == 0:
                    similarity = 0
                else:
                    dist = analysis_utils.calc_distance(first_hap, second_hap, calc_type="haploid")
                    scoring = 2.0 * dist / hap_len
                    similarity = 1.0 - min(1.0, scoring)
                
                row_scores.append(similarity)
        scores.append(row_scores)
                
    scores = np.array(scores)
    scores = scores + scores.T - np.diag(scores.diagonal())
    
    scr_diag = np.sqrt(scores.diagonal())
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = scores / scr_diag
        scores = scores / scr_diag.reshape(1, -1).T
    
    return scores

#%% --- ANALYSIS TOOLS ---

def find_intra_block_recombinations(viterbi_block_likes):
    """
    Identifies samples where the optimal haplotype pair changes between 
    the START and END of a block (indicating intra-block recombination).
    """
    recomb_events = []
    num_blocks = len(viterbi_block_likes)
    if num_blocks == 0: return [], None

    # Detect object type and sample count from first valid block
    num_samples = 0
    is_tensor_mode = False
    
    for b in viterbi_block_likes:
        if hasattr(b, 'starting') and b.starting is not None:
            num_samples = b.starting.shape[0]
            is_tensor_mode = True
            break
        elif b and len(b) > 0 and isinstance(b[0], (list, tuple)):
            # Old format: (start_list, end_list)
            if len(b[0]) > 0:
                num_samples = len(b[0][0])
                break

    print(f"Scanning {num_blocks} blocks and {num_samples} samples for internal crossovers...")

    for block_idx in range(num_blocks):
        block_data = viterbi_block_likes[block_idx]
        
        # --- PATH A: TENSOR MODE ---
        if is_tensor_mode:
            start_mat = block_data.starting # (N, H, H) or flattened
            end_mat = block_data.ending     
            
            if start_mat is None or start_mat.size == 0: continue
            
            n_s, n_h = start_mat.shape[0], start_mat.shape[1]
            
            # Flatten to find argmax (best pair index)
            flat_start = start_mat.reshape(n_s, -1)
            flat_end = end_mat.reshape(n_s, -1)
            
            best_start_idx = np.argmax(flat_start, axis=1)
            best_end_idx = np.argmax(flat_end, axis=1)
            
            # Find mismatches
            mismatch_indices = np.where(best_start_idx != best_end_idx)[0]
            
            for sample_idx in mismatch_indices:
                s_idx = best_start_idx[sample_idx]
                e_idx = best_end_idx[sample_idx]
                
                # Recover h1, h2 from flattened index
                s_h1, s_h2 = divmod(s_idx, n_h)
                e_h1, e_h2 = divmod(e_idx, n_h)
                
                start_pair = tuple(sorted((s_h1, s_h2)))
                end_pair = tuple(sorted((e_h1, e_h2)))
                
                if start_pair != end_pair:
                    ll_start = flat_start[sample_idx, s_idx]
                    ll_end = flat_end[sample_idx, e_idx]
                    
                    recomb_events.append({
                        'block_idx': block_idx,
                        'sample_idx': sample_idx,
                        'start_hap': start_pair,
                        'end_hap': end_pair,
                        'log_lik_start': float(ll_start),
                        'log_lik_end': float(ll_end)
                    })

        # --- PATH B: LEGACY DICT MODE ---
        else:
            if not block_data or not block_data[0]: continue
            start_list = block_data[0][0]
            end_list = block_data[0][1]
            
            if not start_list or not end_list: continue

            for sample_idx in range(num_samples):
                start_dict = start_list[sample_idx]
                end_dict = end_list[sample_idx]
                
                if not start_dict: continue

                best_start_pair = max(start_dict, key=start_dict.get)
                best_end_pair = max(end_dict, key=end_dict.get)
                
                if best_start_pair != best_end_pair:
                    recomb_events.append({
                        'block_idx': block_idx,
                        'sample_idx': sample_idx,
                        'start_hap': best_start_pair,
                        'end_hap': best_end_pair,
                        'log_lik_start': start_dict[best_start_pair],
                        'log_lik_end': end_dict[best_end_pair]
                    })

    print(f"Found {len(recomb_events)} total intra-block recombination events.")
    
    if recomb_events:
        df = pd.DataFrame(recomb_events)
        summary = df.groupby('block_idx').size().reset_index(name='crossover_count')
        summary = summary.sort_values('crossover_count', ascending=False)
    else:
        summary = pd.DataFrame()

    return recomb_events, summary

def analyze_crossover_types(recomb_events):
    """
    Analyzes whether crossovers are familial or systematic error.
    """
    if not recomb_events:
        return None

    df = pd.DataFrame(recomb_events)
    
    def get_signature(row):
        start = tuple(sorted(row['start_hap']))
        end = tuple(sorted(row['end_hap']))
        return f"{start} -> {end}"

    df['transition_type'] = df.apply(get_signature, axis=1)

    block_stats = []
    for block_idx, group in df.groupby('block_idx'):
        total = len(group)
        counts = group['transition_type'].value_counts()
        
        block_stats.append({
            'block_idx': block_idx,
            'total_crossovers': total,
            'unique_switch_types': len(counts),
            'dominant_switch': counts.index[0],
            'dominant_count': counts.iloc[0],
            'diversity_ratio': len(counts) / total
        })
        
    summary_df = pd.DataFrame(block_stats)
    return summary_df.sort_values('total_crossovers', ascending=False)