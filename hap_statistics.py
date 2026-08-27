import thread_config

import numpy as np
from multiprocess import Pool
import warnings

import analysis_utils

warnings.filterwarnings("ignore")
np.seterr(divide='ignore',invalid="ignore")


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

#%% --- HELPER FUNCTIONS FOR BLOCK HAPLOTYPES ---

def combined_best_hap_matches(block_result):
    if hasattr(block_result, 'haplotypes'):
        reads_array = block_result.reads_count_matrix
        haps = block_result.haplotypes
        keep_flags = getattr(block_result, 'keep_flags', None)
        probs_array = getattr(block_result, 'probs_array', None)
    else:
        # Assuming tuple structure (pos, keep_flags, reads, haps)
        keep_flags = block_result[1]
        reads_array = block_result[2]
        haps = block_result[3]
        probs_array = None
        
    # Handle Empty Block Case
    if len(haps) == 0:
        return ([], {}, [])

    # Determine which probability source to use
    if reads_array is not None and reads_array.size > 0:
        # Prefer computing from reads (original behavior)
        (site_priors, actual_probs) = analysis_utils.reads_to_probabilities(
            reads_array,
            use_hwe_prior=False,
        )
    elif probs_array is not None and probs_array.size > 0:
        # Fallback to pre-computed probs_array (when reads discarded for memory)
        actual_probs = probs_array
    else:
        # No probability data available
        return ([], {}, [])
    
    matches = match_best_vectorised(haps, actual_probs, keep_flags=keep_flags)
    return matches

def get_best_matches_all_blocks(block_results, num_processes=16):
    with Pool(processes=num_processes) as pool:
        processing_results = pool.map(combined_best_hap_matches, block_results)
    return processing_results

#%% --- MATCHING COMPARISON & STITCHING HELPERS ---

def relative_haplotype_usage(first_hap, first_matches, second_matches):
    """
    Calculates usages of haplotypes in the second block for samples that
    used 'first_hap' in the first block.
    
    Includes bounds checking to handle cases where one block is empty/invalid.
    """
    use_indices = []
    
    # 1. Validate Inputs
    if not first_matches or len(first_matches) < 1: return {}
    if not second_matches or len(second_matches) < 1: return {}
    
    match_list_1 = first_matches[0]
    match_list_2 = second_matches[0]
    
    if not match_list_1 or not match_list_2: return {}
    
    len_1 = len(match_list_1)
    len_2 = len(match_list_2)
    
    # 2. Collect Indices from First Block
    for sample_idx, (parents, _) in enumerate(match_list_1):
        if first_hap in parents:
            # Only add if this sample ALSO exists in the second block
            if sample_idx < len_2:
                use_indices.append(sample_idx)
            
    second_usages = {}
    
    # 3. Aggregate Usage in Second Block
    for sample_idx in use_indices:
        # Tuple unpacking safety
        entry = match_list_2[sample_idx]
        if entry:
            parents_2, _ = entry
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
    
    # Handle empty blocks
    if not first_haps_dict or not second_haps_dict:
        return ({}, {})
        
    first_matches = matches_data[first_block_index]
    second_matches = matches_data[second_block_index]
    
    # Validate match data structure
    if not first_matches or not second_matches:
        return ({}, {})
        
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
        
    if not hap_vals:
        return np.array([])

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
