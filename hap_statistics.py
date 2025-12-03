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
    # Dummy decorators
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator
    prange = range

#%%

def match_best_vectorised(haps_dict, diploids, keep_flags=None):
    """
    Vectorized matching.
    Uses Matrix Multiplication instead of loops.
    """
    
    # 1. SETUP DATA & MASKS
    diploids = np.array(diploids) 
    num_samples, total_sites, _ = diploids.shape
    
    if keep_flags is None:
        keep_flags = slice(None)
    elif keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags, dtype=bool)
        
    # Slice and Flatten Diploids
    # Input: (Num_Samples, Masked_Sites, 3)
    diploids_masked = diploids[:, keep_flags, :]
    masked_sites = diploids_masked.shape[1]
    
    # Flatten to 2D for Matrix Mult: (Num_Samples, Masked_Sites * 3)
    diploids_flat = diploids_masked.reshape(num_samples, -1)

    # 2. PREPARE HAPLOTYPES
    hap_keys = list(haps_dict.keys())
    num_haps = len(hap_keys)
    
    # Stack haps: (Num_Haps, Masked_Sites, 2)
    hap_tensor = np.array([haps_dict[k][keep_flags] for k in hap_keys])

    # 3. COMBINE HAPLOTYPES (Vectorized)
    # Calculate P(00), P(01), P(11) for all pairs
    p0 = hap_tensor[:, :, 0] 
    p1 = hap_tensor[:, :, 1]

    # Broadcasting: (Num_Haps, Num_Haps, Masked_Sites)
    prob_00 = p0[:, None, :] * p0[None, :, :]
    prob_11 = p1[:, None, :] * p1[None, :, :]
    prob_01 = (p0[:, None, :] * p1[None, :, :]) + (p1[:, None, :] * p0[None, :, :])

    # Stack: (Num_Haps, Num_Haps, Masked_Sites, 3)
    combinations_4d = np.stack([prob_00, prob_01, prob_11], axis=-1)

    # Flatten to list of combinations: (Num_Haps^2, Masked_Sites, 3)
    combinations_list = combinations_4d.reshape(-1, masked_sites, 3)

    # 4. APPLY CUSTOM DISTANCE WEIGHTS
    # Your distance matrix defined in calc_distance
    # 0 vs 0 = 0 cost, 0 vs 2 = 2 cost, etc.
    dist_weights = np.array([[0, 1, 2],
                             [1, 0, 1],
                             [2, 1, 0]])

    # Transform the combinations by the weights
    # We multiply the prob vector (size 3) by the weight matrix (3x3)
    # Shape: (Num_Combinations, Masked_Sites, 3)
    combinations_weighted = combinations_list @ dist_weights

    # Flatten for final dot product: (Num_Combinations, Masked_Sites * 3)
    combinations_weighted_flat = combinations_weighted.reshape(-1, masked_sites * 3)

    # 5. CALCULATE SCORES (Matrix Multiplication)
    # We want sum(Diploid * Weighted_Combination)
    # Matrix Mult: (Samples, Features) @ (Combinations, Features).T
    # Result: (Num_Samples, Num_Combinations)
    dists = diploids_flat @ combinations_weighted_flat.T

    # Normalize: (dist * 100) / original_length
    # Note: You used dip_length in your original code, which corresponds to total_sites
    dists *= (100.0 / total_sites)

    # 6. FIND BEST MATCHES
    # argmin finds the index with the lowest distance cost
    best_indices_flat = np.argmin(dists, axis=1)
    best_errors = dists[np.arange(num_samples), best_indices_flat]

    # 7. RECONSTRUCT KEYS (Same as previous answer)
    idx_grid_i, idx_grid_j = np.indices((num_haps, num_haps))
    idx_grid_i = idx_grid_i.flatten()
    idx_grid_j = idx_grid_j.flatten()

    best_parents_i = idx_grid_i[best_indices_flat]
    best_parents_j = idx_grid_j[best_indices_flat]

    # Usage Counts
    all_used = np.concatenate([best_parents_i, best_parents_j])
    unique_idx, counts = np.unique(all_used, return_counts=True)
    
    haps_usage = {k: 0 for k in hap_keys}
    for idx, count in zip(unique_idx, counts):
        haps_usage[hap_keys[idx]] = count

    # Format Output
    dips_matches = [
        ((hap_keys[p1], hap_keys[p2]), err)
        for p1, p2, err in zip(best_parents_i, best_parents_j, best_errors)
    ]

    return (dips_matches, haps_usage, best_errors)

@njit(parallel=True, fastmath=True)
def viterbi_constrained_k(dist_tensor, max_k, state_defs):
    """
    Finds the best path with AT MOST 'max_k' haplotype switches.
    
    dist_tensor: (Samples, Pairs, Sites) Error scores
    max_k:       Max number of switches allowed (e.g. 3)
    state_defs:  (Pairs, 2) Haplotype definitions for Hamming dist
    
    Returns:
        best_costs: (Samples,) Min error found within budget
        best_paths: (Samples, Sites) Indices of best pair at each site
    """
    n_samples, n_pairs, n_sites = dist_tensor.shape
    
    # Allow budget from 0 to max_k (size is k+1)
    budget_size = max_k + 1
    
    final_costs = np.empty(n_samples, dtype=np.float64)
    
    # Backpointers: (Samples, Sites, 1) - We only store the resulting path
    # We reconstruct the path on the fly or store minimal info. 
    # To keep memory low (O(N)), we can't store the full (Site x Pair x K) backpointer table.
    # HOWEVER, since we need the exact path, and K is small, we can store a 
    # lighter 'decision' table.
    
    # Memory Trade-off: 
    # Storing (Samples, Sites, Pairs, K) bytes is too big.
    # We will accept a simplified approach: The function returns the BEST COST and 
    # the ENDING STATE. Reconstructing the full path exactly for K-limited 
    # without the full table is hard.
    
    # PRACTICAL COMPROMISE: 
    # Since you mostly want to know "Which block hap suits best" and the error,
    # we will return the path of the "Best K" found.
    # To do this efficiently, we store backpointers only for the 'best' per site? 
    # No, that breaks optimality.
    
    # We will use the standard O(N*H*K) backpointer table but assume n_pairs is small (~20-50).
    # 300 * 50000 * 50 * 5 * 1 byte = 3.7 GB. This fits in 128GB RAM.
    # backpointers[s, t, pair_idx, k_used] = prev_pair_idx
    
    # To save space, we use int16 (-32000 to 32000) for pair indices
    backpointers = np.empty((n_samples, n_sites, n_pairs, budget_size), dtype=np.int16)
    
    for s in prange(n_samples):
        # dp[pair_idx, k_used] = min_cost
        curr_dp = np.full((n_pairs, budget_size), np.inf, dtype=np.float64)
        prev_dp = np.full((n_pairs, budget_size), np.inf, dtype=np.float64)
        
        # Init Site 0 (Cost 0, k=0)
        for p in range(n_pairs):
            prev_dp[p, 0] = dist_tensor[s, p, 0]
            
        # Iterate Sites
        for t in range(1, n_sites):
            emission = dist_tensor[s, :, t] # (n_pairs,)
            
            # Reset current
            curr_dp[:] = np.inf
            
            # This inner loop structure is O(H^2 * K). 
            # With H=30, H^2=900, K=5 -> 4500 ops per site. Fast enough in C.
            
            for k_prev_used in range(budget_size):
                
                # Check transitions from every previous pair
                for p_prev in range(n_pairs):
                    cost_prev = prev_dp[p_prev, k_prev_used]
                    if cost_prev == np.inf: continue
                    
                    h1_prev = state_defs[p_prev, 0]
                    h2_prev = state_defs[p_prev, 1]
                    
                    for p_curr in range(n_pairs):
                        # Calculate Cost to Switch (Hamming Distance)
                        h1_curr = state_defs[p_curr, 0]
                        h2_curr = state_defs[p_curr, 1]
                        
                        dist = 2
                        if h1_curr == h1_prev and h2_curr == h2_prev: dist = 0
                        elif h1_curr == h2_prev and h2_curr == h1_prev: dist = 0
                        elif (h1_curr == h1_prev or h1_curr == h2_prev or 
                              h2_curr == h1_prev or h2_curr == h2_prev): dist = 1
                        
                        k_new = k_prev_used + dist
                        
                        if k_new < budget_size:
                            new_total = cost_prev + emission[p_curr]
                            if new_total < curr_dp[p_curr, k_new]:
                                curr_dp[p_curr, k_new] = new_total
                                backpointers[s, t, p_curr, k_new] = p_prev

            # Swap
            prev_dp[:] = curr_dp[:]

        # 4. Find Best Endpoint
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
        
        # 5. Backtrack Path
        # We reuse the backpointers array to store the path in [s, :, 0, 0] to simplify return
        curr_p = best_p
        curr_k = best_k
        
        # Store last step
        # We handle t=0 separately, loop t from N-1 down to 1
        path_storage = np.empty(n_sites, dtype=np.int32)
        path_storage[n_sites-1] = curr_p
        
        for t in range(n_sites-1, 0, -1):
            prev_p = backpointers[s, t, curr_p, curr_k]
            
            # Recalculate dist to know which k came before
            # (We didn't store prev_k to save RAM, we infer it)
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
            
        # Write to backpointers[0,0] just to smuggle it out
        # (Hack to keep signature simple, or just return separate array)
        # Let's return separate array actually, simpler.
        # But for Numba parallel, writing to a pre-allocated shared array is best.
        # Let's just use the backpointers array's first slice.
        for t in range(n_sites):
            backpointers[s, t, 0, 0] = path_storage[t]

    return final_costs, backpointers[:, :, 0, 0]

def match_best_k_limited(haps_dict, diploids, keep_flags=None, max_recombinations=2):
    """
    Matches samples to the best haplotype pair allowing AT MOST 'max_recombinations'.
    
    Returns:
        (matches_list, haps_usage, errors)
        
        matches_list structure: [ ( [Path_Sequence], Error ), ... ]
        Example Path Sequence: [('Hap1', 'Hap2'), ('Hap1', 'Hap5')] 
        (Indicates a switch from 2->5 occurred in the middle of the block)
    """
    
    # 1. SETUP (Identical to before)
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

    # 2. GENERATE PAIRS (Identical)
    idx_i, idx_j = np.triu_indices(num_haps)
    state_defs = np.stack([idx_i, idx_j], axis=1).astype(np.int32)
    
    h0 = hap_tensor[idx_i, :, 0]
    h1 = hap_tensor[idx_i, :, 1]
    h2_0 = hap_tensor[idx_j, :, 0]
    h2_1 = hap_tensor[idx_j, :, 1]
    
    pair_genotypes = np.stack([
        h0 * h2_0, 
        (h0 * h2_1) + (h1 * h2_0),
        h1 * h2_1
    ], axis=-1)
    
    # 3. ERROR TENSOR (Identical)
    dist_weights = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    weighted_pairs = pair_genotypes @ dist_weights
    
    dist_tensor = np.sum(
        diploids_masked[:, np.newaxis, :, :] * weighted_pairs[np.newaxis, :, :, :],
        axis=3
    )
    
    # 4. RUN VITERBI (Identical)
    # Re-use the viterbi_constrained_k function from previous answers
    # (Assuming it is imported or available in this file)
    best_costs, best_paths = viterbi_constrained_k(
        dist_tensor.astype(np.float64), 
        int(max_recombinations), 
        state_defs
    )
    
    # 5. FORMAT OUTPUT (CHANGED)
    normalized_errors = (best_costs / masked_sites) * 100.0
    
    # Usage Stats
    used_hap_indices = state_defs[best_paths.astype(np.int32)]
    flat_indices = used_hap_indices.flatten()
    u_idx, u_counts = np.unique(flat_indices, return_counts=True)
    haps_usage = {k: 0 for k in hap_keys}
    for idx, count in zip(u_idx, u_counts):
        haps_usage[hap_keys[idx]] = count / masked_sites

    dips_matches = []
    
    for s in range(num_samples):
        path = best_paths[s].astype(np.int32)
        
        # --- LOGIC CHANGE START ---
        # Instead of Majority Vote, we extract the sequence of states
        
        # 1. Identify indices where the path changes
        # np.diff != 0 gives True where value changes. Insert True at pos 0 to keep start.
        change_mask = np.insert(np.diff(path) != 0, 0, True)
        
        # 2. Get the unique sequence of pair indices in order
        path_sequence_indices = path[change_mask]
        
        # 3. Convert indices to Haplotype Keys
        path_resolved = []
        for pair_idx in path_sequence_indices:
            h1_idx, h2_idx = state_defs[pair_idx]
            path_resolved.append((hap_keys[h1_idx], hap_keys[h2_idx]))
            
        # Store list of pairs: [('A','B'), ('A','C')]
        dips_matches.append((path_resolved, normalized_errors[s]))
        # --- LOGIC CHANGE END ---

    return (dips_matches, haps_usage, normalized_errors)

def get_addition_statistics(starting_haps,
                            candidate_haps,
                            addition_index,
                            probs_array,
                            keep_flags = None):
    """
    Add one hap from our list of candidate haps and see how much better
    of a total fit we get.
    """
    
    for x in starting_haps.keys():
        orig_hap_len = len(starting_haps[x])
        break
    
    if keep_flags is None:
        keep_flags = np.array([True for _ in range(orig_hap_len)])
    
    if keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags,dtype=bool)
        
    added_haps = starting_haps.copy()
    adding_name = max(added_haps.keys())+1
    
    added_haps[adding_name] = candidate_haps[addition_index]
    
    added_matches = match_best_vectorised(added_haps,probs_array,keep_flags=keep_flags)
    
    added_mean = np.mean(added_matches[2])
    added_max = np.max(added_matches[2])
    added_std = np.std(added_matches[2])
    
    return (added_mean,added_max,added_std,added_matches)

def get_removal_statistics(candidate_haps,
                           candidate_matches,
                           removal_value,
                           probs_array):
    """
    Remove one hap from our list of candidate haps and see how much worse
    of a total fit we get.
    """
    truncated_haps = candidate_haps.copy()
    truncated_haps.pop(removal_value)
    truncated_matches = match_best_vectorised(truncated_haps,probs_array)
    
    truncated_mean = np.mean(truncated_matches[2])
    truncated_max = np.max(truncated_matches[2])
    truncated_std = np.std(truncated_matches[2])
    
    return (truncated_mean,truncated_max,truncated_std,truncated_matches)

def combined_best_hap_matches(haps_data_block):
    """
    Takes full haps data for a single block and calculates 
    the best matches for the haps for that block
    """
    keep_flags = haps_data_block[1]
    reads_array = haps_data_block[2]
    haps = haps_data_block[3]
    (site_priors,probs_array) = analysis_utils.reads_to_probabilities(reads_array)
    
    actual_probs = probs_array[0]
    
    matches = match_best_vectorised(haps,actual_probs,keep_flags=keep_flags)
    
    return matches

def get_best_matches_all_blocks(haps_data):
    """
    Multithreaded function to calculate the best matches for 
    each block in haps data, applies combined_best_hap_matches
    to each element of haps_data
    """
    
    processing_pool = Pool(8)
    
    processing_results = processing_pool.starmap(combined_best_hap_matches,
                                                 zip(haps_data))
    
    return processing_results

def relative_haplotype_usage(first_hap,first_matches,second_matches):
    """
    Counts the relative usage of haps in second_matches for those
    samples which include first_hap in first_matches
    
    first_matches and second_matches must correspond to the same 
    samples in order (so the first element of both are from the same 
    sample etc. etc.)
    """
    use_indices = []
    
    for sample in range(len(first_matches[0])):
        if first_matches[0][sample][0][0] == first_hap:
            use_indices.append(sample)
        if first_matches[0][sample][0][1] == first_hap:
            use_indices.append(sample)
            
    second_usages = {}
    
    for sample in use_indices:
        dat = second_matches[0][sample][0]
        
        for s in dat:
            if s not in second_usages.keys():
                second_usages[s] = 0
            second_usages[s] += 1
    
    second_usages = {k: v for k, v in sorted(second_usages.items(), key=lambda item: item[1])}
    
    return second_usages

def relative_haplotype_usage_indicator(first_hap,first_matches,second_matches):
    """
    Like relative_haplotype_usage but instead of giving a count gives
    a list for each hap in second_matches which contains those sample
    ids which use first_hap in first_matches and that hap in second_matches
    """
    use_indices = []
    
    for sample in range(len(first_matches[0])):
        if first_matches[0][sample][0][0] == first_hap:
            use_indices.append(sample)
        if first_matches[0][sample][0][1] == first_hap:
            use_indices.append(sample)
    
    second_usages = {}
    
    for sample in use_indices:
        dat = second_matches[0][sample][0]
        
        for s in dat:
            if s not in second_usages.keys():
                second_usages[s] = []
            second_usages[s].append(sample)
    
    
    second_usages = {k: v for k, v in sorted(second_usages.items(), key=lambda item: item[1])}
    
    return second_usages

def hap_matching_comparison(haps_data,matches_data,first_block_index,second_block_index):
    """
    For each hap at the first_block_index block this fn. compares
    where the samples which use that hap end up for the block at
    second_block_index and converts these numbers into percentage
    usages for each hap at index first_block_index
        
    It also then scales these scores and returns the scaled version of 
    these scores back to us as a dictionary
    """
        
    forward_scores = {}
    backward_scores = {}
        
    first_haps_data = haps_data[first_block_index][3]
    second_haps_data = haps_data[second_block_index][3]
        
    first_matches_data = matches_data[first_block_index]
    second_matches_data = matches_data[second_block_index]
        
    for hap in first_haps_data.keys():
        hap_usages = relative_haplotype_usage(hap,first_matches_data,second_matches_data)
        total_matches = sum(hap_usages.values())
        hap_percs = {x:100*hap_usages[x]/total_matches for x in hap_usages.keys()}
            
        scaled_scores = {}
            
        for other_hap in second_haps_data.keys():
            if other_hap in hap_percs.keys():
                scaled_val = 100*(min(1,2*hap_percs[other_hap]/100))**2
                    
                scaled_scores[((first_block_index,hap),
                               (second_block_index,other_hap))] = scaled_val
            elif other_hap not in hap_percs.keys():
                scaled_scores[((first_block_index,hap),
                               (second_block_index,other_hap))] = 0
        forward_scores.update(scaled_scores)
            
    for hap in second_haps_data.keys():
        hap_usages = relative_haplotype_usage(hap,second_matches_data,first_matches_data)
        total_matches = sum(hap_usages.values())
        hap_percs = {x:100*hap_usages[x]/total_matches for x in hap_usages.keys()}
            
        scaled_scores = {}
            
        for other_hap in first_haps_data.keys():
            if other_hap in hap_percs.keys():
                scaled_val = 100*(min(1,2*hap_percs[other_hap]/100))**2
                    
                scaled_scores[((first_block_index,other_hap),
                               (second_block_index,hap))] = scaled_val
            elif other_hap not in hap_percs.keys():
                scaled_scores[((first_block_index,other_hap),
                               (second_block_index,hap))] = 0
        backward_scores.update(scaled_scores)
        
    return (forward_scores,backward_scores)

def get_block_hap_similarities(haps_data):
    """
    Takes as input a list of haplotypes for a single block 
    such as generated from generate_haplotypes_all and 
    calculates a similarity matrix between them with values 
    from 0 to 1 with higher values denoting more similar haps
    """
    
    scores = []
    
    keep_flags = np.array(haps_data[1],dtype=bool)
    
    hap_vals = haps_data[3]
    
    for i in hap_vals.keys():
        scores.append([])
        for j in hap_vals.keys():
            if j < i:
                scores[-1].append(0)
            else:
                
                first_hap = hap_vals[i][keep_flags]
                second_hap = hap_vals[j][keep_flags]
                hap_len = len(first_hap)
                
                scoring = 2.0*analysis_utils.calc_distance(first_hap,second_hap,calc_type="haploid")/hap_len
                
                similarity = 1.0-min(1.0,scoring)
                scores[-1].append(similarity)
                
    scores = np.array(scores)
    scores = scores+scores.T-np.diag(scores.diagonal())
    
    scr_diag = np.sqrt(scores.diagonal())
    
    scores = scores/scr_diag
    scores = scores/scr_diag.reshape(1,-1).T
    
    return scores

def find_intra_block_recombinations(viterbi_block_likes):
    """
    Identifies samples where the optimal haplotype pair at the start of the block
    differs from the optimal pair at the end of the block.
    
    Returns:
        recomb_events: List of dictionaries containing details of each event.
        summary_stats: A pandas DataFrame summarizing crossovers per block.
    """
    recomb_events = []
    
    num_blocks = len(viterbi_block_likes)
    
    if num_blocks == 0:
        return [], None

    # Determine number of samples from the first non-empty block
    # viterbi_block_likes[i] is ((StartList, EndList), Ploidy)
    # StartList is a list of dicts, one per sample
    num_samples = 0
    for b in viterbi_block_likes:
        if b and b[0] and b[0][0]:
            num_samples = len(b[0][0])
            break
            
    print(f"Scanning {num_blocks} blocks and {num_samples} samples for internal crossovers...")

    for block_idx in range(num_blocks):
        block_data = viterbi_block_likes[block_idx]
        
        # Skip empty blocks
        if not block_data or not block_data[0]:
            continue
            
        start_list = block_data[0][0]
        end_list = block_data[0][1]
        
        # Double check we have data
        if not start_list or not end_list:
            continue

        for sample_idx in range(num_samples):
            start_dict = start_list[sample_idx]
            end_dict = end_list[sample_idx]
            
            if not start_dict: continue

            # Find the best pair at the Start (Left edge)
            # This represents the state the Viterbi path started in
            best_start_pair = max(start_dict, key=start_dict.get)
            
            # Find the best pair at the End (Right edge)
            # This represents the state the Viterbi path ended in
            best_end_pair = max(end_dict, key=end_dict.get)
            
            # If they differ, the optimal path switched states inside the block
            if best_start_pair != best_end_pair:
                
                # Get the scores to see confidence
                start_score = start_dict[best_start_pair]
                end_score = end_dict[best_end_pair]
                
                recomb_events.append({
                    'block_idx': block_idx,
                    'sample_idx': sample_idx,
                    'start_hap': best_start_pair,
                    'end_hap': best_end_pair,
                    'log_lik_start': start_score,
                    'log_lik_end': end_score
                })

    print(f"Found {len(recomb_events)} total intra-block recombination events.")
    
    # Create Summary Statistics
    if recomb_events:
        df = pd.DataFrame(recomb_events)
        
        # Group by Block to see hotspots
        summary = df.groupby('block_idx').size().reset_index(name='crossover_count')
        summary = summary.sort_values('crossover_count', ascending=False)
    else:
        summary = pd.DataFrame()

    return recomb_events, summary

def analyze_crossover_types(recomb_events):
    """
    Analyzes recombination events to distinguish between:
    1. Familial Inheritance (Many samples having the EXACT SAME switch).
    2. Systematic Error (Many samples having DIFFERENT switches at the same block).
    """
    if not recomb_events:
        print("No recombination events to analyze.")
        return None

    df = pd.DataFrame(recomb_events)
    
    # 1. Normalize the Transitions
    # We treat (1, 2) the same as (2, 1) for this analysis
    def get_signature(row):
        start = tuple(sorted(row['start_hap']))
        end = tuple(sorted(row['end_hap']))
        return f"{start} -> {end}"

    df['transition_type'] = df.apply(get_signature, axis=1)

    # 2. Group by Block
    block_stats = []
    
    for block_idx, group in df.groupby('block_idx'):
        total_crossovers = len(group)
        
        # Count how many distinct types of switches happened here
        counts = group['transition_type'].value_counts()
        most_common_type = counts.index[0]
        most_common_count = counts.iloc[0]
        
        # Diversity Score (0.0 = All identical, 1.0 = All different)
        # Low diversity suggests familial inheritance.
        # High diversity suggests a structural problem with the block.
        unique_switches = len(counts)
        diversity_ratio = unique_switches / total_crossovers
        
        block_stats.append({
            'block_idx': block_idx,
            'total_crossovers': total_crossovers,
            'unique_switch_types': unique_switches,
            'dominant_switch': most_common_type,
            'dominant_count': most_common_count,
            'diversity_ratio': diversity_ratio
        })
        
    summary_df = pd.DataFrame(block_stats)
    
    # Sort by total count to find hotspots
    summary_df = summary_df.sort_values('total_crossovers', ascending=False)
    
    return summary_df
