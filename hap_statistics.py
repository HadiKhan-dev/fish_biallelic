import numpy as np
from multiprocess import Pool
import warnings

import analysis_utils

warnings.filterwarnings("ignore")
np.seterr(divide='ignore',invalid="ignore")
pass

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

def match_best(haps_dict,diploids,keep_flags=None):
    """
    Find the best matches of a pair of haploids for each diploid in the diploid list
    """
    
    dip_length = len(diploids[0])
    
    if keep_flags is None:
        keep_flags = np.array([True for _ in range(dip_length)])
    
    if keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags,dtype=bool)
        
    haps_dict = {x:haps_dict[x][keep_flags] for x in haps_dict.keys()}

    diploids = diploids[:,keep_flags]
    
    dips_matches = []
    haps_usage = {}
    errs = []
    
    combined_haps = {}
    
    for i in haps_dict.keys():
        for j in haps_dict.keys():
            comb = analysis_utils.combine_haploids(haps_dict[i],haps_dict[j])
            if (j,i) not in combined_haps.keys():
                combined_haps[(i,j)] = comb
    
    for i in range(len(diploids)):
        cur_best = (None,None)
        cur_div = np.inf
        
        for combination_index in combined_haps.keys():
                combi = combined_haps[combination_index]
                div = 100*analysis_utils.calc_distance(diploids[i],combi)/dip_length
                if div < cur_div:
                    cur_div = div
                    cur_best = combination_index
        
        for index in cur_best:
            if index not in haps_usage.keys():
                haps_usage[index] = 0
            haps_usage[index] += 1
            
        errs.append(cur_div)
        dips_matches.append((cur_best,cur_div))
    
    return (dips_matches,haps_usage,np.array(errs))

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
    
    added_matches = match_best(added_haps,probs_array,keep_flags=keep_flags)
    
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
    truncated_matches = match_best(truncated_haps,probs_array)
    
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
    
    matches = match_best(haps,actual_probs,keep_flags=keep_flags)
    
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