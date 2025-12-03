

import numpy as np
import math
import hdbscan
from multiprocess import Pool
import warnings
import time
from scipy.spatial.distance import cdist
from functools import partial

import analysis_utils
import hap_statistics

warnings.filterwarnings("ignore")
np.seterr(divide='ignore',invalid="ignore")
pass

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
def hdbscan_cluster(dist_matrix,
                    min_cluster_size=3,
                    min_samples=1,
                    cluster_selection_method="eom",
                    alpha=1,
                    allow_single_cluster=False):
    """
    Create clusters of similar rows for the given dist_matrix
    using HDBSCAN
    """
    #Create clustering object from sklearn
    base_clustering = hdbscan.HDBSCAN(metric="precomputed",
                                      min_cluster_size=min_cluster_size,
                                      min_samples=min_samples,
                                      cluster_selection_method=cluster_selection_method,
                                      alpha=alpha,allow_single_cluster=allow_single_cluster)
    
    
    #Fit data to clustering
    base_clustering.fit(dist_matrix)
    
    #Plot the condensed tree
    # color_palette = sns.color_palette('Paired', 20)
    # base_clustering.condensed_tree_.plot(select_clusters=True,selection_palette=color_palette)
    # plt.show()
    
    initial_labels = np.array(base_clustering.labels_)
    initial_probabilities = np.array(base_clustering.probabilities_)
    
    all_clusters = set(initial_labels)
    all_clusters.discard(-1)

    
    return [initial_labels,initial_probabilities,base_clustering]

def fix_cluster_labels(c_labels):
    """
    Convert jumpbled up cluster labels into ones following
    ascending order as we move across the list of labels
    """
    seen_old = set([])
    new_mapping = {}
    cur_number = 0
    
    for i in range(len(c_labels)):
        test = c_labels[i]
        if test not in seen_old:
            seen_old.add(test)
            new_mapping[test] = cur_number
            cur_number += 1
    
    new_labels = list(map(lambda x: new_mapping[x],c_labels))
    
    return new_labels

def get_representatives_reads(site_priors,
                        reads_array,
                        cluster_labels,
                        cluster_probs=np.array([]),
                        prob_cutoff=0.8,
                        read_error_prob=0.02):
    """
    Get representatives for each of the clusters we have,
    cluster_labels must be a list of length len(reads_array)
    showing which cluster each sample maps to
    
    This version works by taking as input an array of read counts
    for each sample for each site
    """
    
    num_sites = len(site_priors)
    
    singleton_priors = np.array([np.sqrt(site_priors[:,0]),np.sqrt(site_priors[:,2])]).T
    reads_array = np.array(reads_array)
    
    if len(cluster_probs) == 0:
        cluster_probs = np.array([1]*reads_array.shape[0])

    cluster_names = set(cluster_labels)
    cluster_representatives = {}
    
    #Remove the outliers
    cluster_names.discard(-1)
    
    
    for cluster in cluster_names:
        
        #Find those indices which strongly map to the current cluster
        indices = np.where(np.logical_and(cluster_labels == cluster,cluster_probs >= prob_cutoff))[0]

        #Pick out these indices and get the data for the cluster
        cluster_data = reads_array[indices,:]
        
        
        cluster_representative = []
    
        #Calculate total count of reads at each site for the samples which fall into the cluster
        site_sum = np.nansum(cluster_data,axis=0)
        
        log_read_error = math.log(read_error_prob)
        log_read_nonerror = math.log(1-read_error_prob)
        
        for i in range(num_sites):
            
            priors = singleton_priors[i]
            log_priors = np.log(priors)
            
            zeros = site_sum[i][0]
            ones = site_sum[i][1]
            total = zeros+ones
        
            log_likelihood_0 = analysis_utils.log_binomial(total,ones)+zeros*log_read_nonerror+ones*log_read_error
            log_likelihood_1 = analysis_utils.log_binomial(total,zeros)+zeros*log_read_error+ones*log_read_nonerror
            
            log_likli = np.array([log_likelihood_0,log_likelihood_1])
            
            nonnorm_log_postri = log_priors + log_likli
            nonnorm_log_postri -= np.mean(nonnorm_log_postri)
            
            #Numerical correction when we have a very wide range in loglikihoods to prevent np.inf
            while np.max(nonnorm_log_postri) > 200:
                nonnorm_log_postri -= 100
            
            nonnorm_post = np.exp(nonnorm_log_postri)
            posterior = nonnorm_post/sum(nonnorm_post)
            
            cluster_representative.append(posterior)
        
        #Add the representative for the cluster to the dictionary
        cluster_representatives[cluster] = np.array(cluster_representative)
        

    #Return the results
    return cluster_representatives

def get_representatives_probs(site_priors,
                        probs_array,
                        cluster_labels,
                        cluster_probs=np.array([]),
                        prob_cutoff=0.8,
                        site_max_singleton_sureness=0.98,
                        read_error_prob=0.02):
    """
    Get representatives for each of the clusters we have,
    cluster_labels must be a list of length len(probs_array) showing
    which cluster each sample maps to
    
    This version works by taking as input an array of ref/alt probabilities
    for each sample/site
    
    """
    
    singleton_priors = np.array([np.sqrt(site_priors[:,0]),np.sqrt(site_priors[:,2])]).T
    probs_array = np.array(probs_array)
    
    if len(cluster_probs) == 0:
        cluster_probs = np.array([1]*probs_array.shape[0])

    cluster_names = set(cluster_labels)
    cluster_representatives = {}
    
    #Remove the outliers
    cluster_names.discard(-1)
    
    for cluster in cluster_names:
        
        #Find those indices which strongly map to the current cluster
        indices = np.where(
            np.logical_and(
                cluster_labels == cluster,
                cluster_probs >= prob_cutoff))[0]

        #Pick out these indices and get the data for the cluster
        cluster_data = probs_array[indices,:]
        
        cluster_representative = []
    
        num_sites = cluster_data.shape[1]
        
        for i in range(num_sites):
            
            priors = singleton_priors[i]
            log_priors = np.log(priors)
            
            site_data = cluster_data[:,i,:].copy()
            
            site_data[site_data > site_max_singleton_sureness] = site_max_singleton_sureness
            site_data[site_data < 1-site_max_singleton_sureness] = 1-site_max_singleton_sureness
            
            zero_logli = 0
            one_logli = 0
            
            for info in site_data:
                zero_logli += math.log((1-read_error_prob)*info[0]+read_error_prob*info[1])
                one_logli += math.log(read_error_prob*info[0]+(1-read_error_prob)*info[1])
            
            logli_array = np.array([zero_logli,one_logli])
            
            nonorm_log_post = log_priors+logli_array
            nonorm_log_post -= np.mean(nonorm_log_post)
            
            nonorm_post = np.exp(nonorm_log_post)
            
            posterior = nonorm_post/np.sum(nonorm_post)
            
            cluster_representative.append(posterior)
        
        #Add the representative for the cluster to the dictionary
        cluster_representatives[cluster] = np.array(cluster_representative)
        

    #Return the results
    return cluster_representatives

def consolidate_similar_candidates(candidates, diff_threshold=0.02):
    """
    Greedily merges candidates that are nearly identical (e.g. < 2% difference).
    This prevents the Viterbi selection from selecting multiple noise variations 
    of the same founder to explain specific outlier samples.
    
    Args:
        candidates: Dictionary {index: hap_array} or List of hap_arrays
    Returns:
        dict: {new_index: hap_array} containing only distinct haplotypes
    """
    if not candidates: return {}
    
    # Normalize input to list of arrays
    if isinstance(candidates, dict):
        candidate_list = list(candidates.values())
    else:
        candidate_list = candidates

    unique_haps = []
    
    for hap in candidate_list:
        is_duplicate = False
        for existing in unique_haps:
            # Calculate Hamming distance (percentage of sites that differ)
            diff = np.mean(hap != existing)
            
            if diff < diff_threshold:
                # It's a duplicate (or noise variant)
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_haps.append(hap)
            
    # Rebuild dictionary with sequential keys
    return {i: h for i, h in enumerate(unique_haps)}

@njit(parallel=True, fastmath=True)
def viterbi_score_selection(ll_tensor, penalty):
    """
    Calculates the BEST Viterbi path score for each sample given a set of active pairs.
    Used for Model Selection (BIC).
    
    ll_tensor: (n_samples, K_active_pairs, n_sites)
    penalty:   Cost per switch (positive float).
    
    Returns: (n_samples,) array of best log-likelihoods.
    """
    n_samples, K, n_sites = ll_tensor.shape
    best_scores = np.empty(n_samples, dtype=np.float64)
    
    for s in prange(n_samples):
        # Buffer for current scores (faster than allocation in loop)
        current_scores = np.empty(K, dtype=np.float64)
        for k in range(K):
            current_scores[k] = ll_tensor[s, k, 0]
            
        for i in range(1, n_sites):
            # 1. Find best previous score globally
            best_prev = -np.inf
            for k in range(K):
                if current_scores[k] > best_prev:
                    best_prev = current_scores[k]
            
            # The baseline score if we switch INTO a state
            switch_base = best_prev - penalty
            
            # 2. Update states
            for k in range(K):
                emission = ll_tensor[s, k, i]
                stay = current_scores[k]
                
                # Max(Stay, Switch)
                if stay > switch_base:
                    current_scores[k] = stay + emission
                else:
                    current_scores[k] = switch_base + emission
        
        # Final max
        final_max = -np.inf
        for k in range(K):
            if current_scores[k] > final_max:
                final_max = current_scores[k]
        best_scores[s] = final_max
        
    return best_scores

def prune_chimeras(hap_dict, probs_array=None, recomb_penalty=10.0, error_tolerance_sites=10):
    """
    Identifies and removes haplotypes that can be explained as 
    recombinations of OTHER haplotypes.
    
    TOPOLOGY-AWARE VERSION WITH FREQUENCY TIE-BREAKER:
    1. Iteratively finds the haplotype most easily explained by the others (Highest Score).
    2. If scores are tied (symmetric redundancy), it protects the haplotype with higher usage.
    """
    if len(hap_dict) < 3:
        return hap_dict
        
    kept_keys = sorted(list(hap_dict.keys()))
    
    # Calculate usage counts if data is available (for tie-breaking)
    usage_map = {}
    if probs_array is not None:
        matches = hap_statistics.match_best_vectorised(hap_dict, probs_array)
        usage_map = matches[1]
    
    # Mismatch penalty matching your error floor
    mismatch_penalty = 0.5 
    
    # We loop until we can no longer explain ANY haplotype using the others
    # or until we hit the minimum of 2 haplotypes
    while len(kept_keys) >= 3:
        
        candidate_scores = []
        
        # Convert all kept haps to a tensor for fast comparison
        H_stack = np.array([hap_dict[k] for k in kept_keys])
        n_haps, n_sites, _ = H_stack.shape
        
        for i, target_key in enumerate(kept_keys):
            target_hap = H_stack[i]
            
            # Basis = Everyone except i
            other_indices = [idx for idx in range(n_haps) if idx != i]
            other_haps = H_stack[other_indices]
            
            # 1. Calculate Likelihoods (Target vs Basis)
            # Distance 0 -> Score 0. Distance > 0 -> Score -0.5
            mismatches = np.sum(np.abs(target_hap[None, :, :] - other_haps), axis=2)
            ll_tensor = np.where(mismatches < 0.5, 0.0, -mismatch_penalty)
            
            # Add dimension: (1_Sample, N-1_States, Sites)
            ll_tensor = ll_tensor[np.newaxis, :, :]
            
            # 2. Run Viterbi
            # "How well can I construct Target using the Others?"
            best_score = viterbi_score_selection(
                ll_tensor, float(recomb_penalty)
            )[0]
            
            # 3. Apply Tie-Breaker
            # We want to remove High Scores (Easy to explain).
            # If scores are equal, we want to remove Low Usage first.
            # So we SUBTRACT a tiny epsilon based on usage.
            # High Usage -> Lower Score -> Harder to remove.
            usage = usage_map.get(target_key, 0)
            adjusted_score = best_score - (usage * 1e-9)
            
            candidate_scores.append((target_key, adjusted_score, best_score))
            
        # Sort candidates by Adjusted Score Descending
        # The top element is the "Most Redundant" (Best fit by others + Lowest usage)
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        
        worst_key, _, raw_score = candidate_scores[0]
        
        # 4. Decision Logic
        # Allow 2 switches (A->B->A) + tolerance
        allowed_switches = 2
        threshold = -((recomb_penalty * allowed_switches) + 
                      (mismatch_penalty * error_tolerance_sites))
        
        if raw_score > threshold:
            kept_keys.remove(worst_key)
            print(f"  Pruned Chimera {worst_key} (Score {raw_score:.1f}). Explained by others.")
        else:
            # The "easiest" haplotype to explain is still impossible to explain.
            # We have reached the irreducible Founders.
            break
            
    return {k: hap_dict[k] for k in kept_keys}

def select_optimal_haplotype_set_viterbi(candidate_haps, probs_array, 
                                         recomb_penalty=10.0,
                                         penalty_strength=1.0,
                                         read_error_prob=0.02,
                                         max_sites_for_selection=2000):
    """
    Selects the smallest set of haplotypes that explains the data best,
    ALLOWING for recombinations within the block via Viterbi.
    
    Features:
    1. Downsamples sites if block is massive to save RAM.
    2. Scales likelihoods to account for downsampling.
    3. Batches pair creation to prevent memory explosion.
    4. Rejects 'Chimera' haplotypes that are just mixes of existing ones.
    """
    
    # --- 1. SETUP DATA ---
    if isinstance(candidate_haps, dict):
        hap_keys = list(candidate_haps.keys())
        H = np.array([candidate_haps[k] for k in hap_keys])
    else:
        hap_keys = list(range(len(candidate_haps)))
        H = np.array(candidate_haps)
        
    num_candidates = len(H)
    num_samples, total_sites, _ = probs_array.shape
    
    if num_candidates == 0: return []

    # --- 2. DOWNSAMPLING STRATEGY ---
    # If we have too many sites, we downsample to save RAM/CPU.
    # We multiply the resulting scores by 'stride' to estimate the full block likelihood.
    
    if total_sites > max_sites_for_selection:
        stride = math.ceil(total_sites / max_sites_for_selection)
        # Slice the data
        probs_active = probs_array[:, ::stride, :]
        H_active = H[:, ::stride, :]
        # print(f"Downsampling sites from {total_sites} to {probs_active.shape[1]} (Stride {stride}) for selection.")
    else:
        stride = 1
        probs_active = probs_array
        H_active = H

    num_active_sites = probs_active.shape[1]

    # --- 3. PRE-CALCULATE ALL PAIR LIKELIHOODS (MEMORY SAFE) ---
    # Tensor Target: (Samples, Total_Pairs, Active_Sites)
    
    idx_i, idx_j = np.triu_indices(num_candidates)
    num_pairs = len(idx_i)
    
    # Pre-allocate the final tensor (Destination)
    # Size example: 300 samples * 465 pairs * 2000 sites * 4 bytes ~= 1.1 GB (Safe)
    ll_tensor = np.empty((num_samples, num_pairs, num_active_sites), dtype=np.float32)
    
    # Weights matrix for distance (00=0, 01=1, 11=2)
    W = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=np.float32)
    
    # Process pairs in batches to avoid intermediate broadcasting explosion
    batch_size = 50
    
    for start_p in range(0, num_pairs, batch_size):
        end_p = min(start_p + batch_size, num_pairs)
        
        # Get indices for this batch
        batch_idx_i = idx_i[start_p:end_p]
        batch_idx_j = idx_j[start_p:end_p]
        
        # Extract Haps: (Batch, Sites)
        h0 = H_active[batch_idx_i, :, 0]
        h1 = H_active[batch_idx_i, :, 1]
        h2_0 = H_active[batch_idx_j, :, 0]
        h2_1 = H_active[batch_idx_j, :, 1]
        
        # Generate Genotypes: (Batch, Sites, 3)
        g00 = h0 * h2_0
        g11 = h1 * h2_1
        g01 = (h0 * h2_1) + (h1 * h2_0)
        
        batch_pairs = np.stack([g00, g01, g11], axis=-1)
        
        # Apply Weights: (Batch, Sites, 3)
        weighted_pairs = batch_pairs @ W
        
        # Calc Distance: (N, Batch, Sites)
        # Sum over alleles axis (3)
        batch_dist = np.sum(
            probs_active[:, np.newaxis, :, :] * weighted_pairs[np.newaxis, :, :, :],
            axis=3
        )
        
        # Store into main tensor
        # Convert to Log Likelihood (-Dist)
        # SCALE UP by stride to approximate full block signal so Penalty matches magnitude
        ll_tensor[:, start_p:end_p, :] = (-batch_dist * stride)

    # Cap floor to prevent -inf issues
    ll_tensor = np.maximum(ll_tensor, -2.0 * stride)
    
    # Ensure contiguous for Numba (float64 preferred for accumulation accuracy)
    ll_tensor = np.ascontiguousarray(ll_tensor, dtype=np.float64)

    # --- 4. SELECTION LOOP ---
    
    selected_indices = []
    current_best_bic = float('inf')
    
    # BIC Constants
    # Complexity must be > Recomb Penalty to prevent adding 'chimera' haps
    min_complexity = recomb_penalty * 1.5
    calculated_complexity = math.log(num_samples) * total_sites * penalty_strength * 0.01
    complexity_cost = max(calculated_complexity, min_complexity)
    
    # print(f"Selection Parameters: Recomb Cost={recomb_penalty}, New Hap Cost={complexity_cost:.1f}")

    while len(selected_indices) < num_candidates:
        
        best_new_index = -1
        best_new_bic = float('inf')
        
        remaining = [x for x in range(num_candidates) if x not in selected_indices]
        
        for cand_idx in remaining:
            trial_set = selected_indices + [cand_idx]
            
            # Create mask for valid pairs in this trial set
            # A pair (i, j) is valid only if both i and j are in trial_set
            subset_mask = np.zeros(num_candidates, dtype=bool)
            subset_mask[trial_set] = True
            
            # Vectorized lookup using pre-calc indices
            valid_pairs_mask = subset_mask[idx_i] & subset_mask[idx_j]
            
            if not np.any(valid_pairs_mask):
                continue
            
            # Slice Tensor: (N, Active_Pairs, L)
            active_ll_tensor = ll_tensor[:, valid_pairs_mask, :]
            
            # RUN VITERBI
            # This returns the best score per sample allowing switches between Active Pairs
            best_scores = viterbi_score_selection(active_ll_tensor, float(recomb_penalty))
            
            total_log_likelihood = np.sum(best_scores)
            
            # BIC Calculation
            k = len(trial_set)
            bic = (k * complexity_cost) - (2 * total_log_likelihood)
            
            if bic < best_new_bic:
                best_new_bic = bic
                best_new_index = cand_idx
        
        if best_new_bic < current_best_bic:
            selected_indices.append(best_new_index)
            current_best_bic = best_new_bic
        else:
            # Diminishing returns reached
            break
            
    return [hap_keys[i] for i in selected_indices]

def add_distinct_haplotypes(initial_haps,
                       new_candidate_haps,
                       keep_flags=None,
                       unique_cutoff=10,
                       max_hap_add=1000):
    """
    Takes two dictionaries of haplotypes and a list of new potential
    haptotype and creates a new dictionary of haplotypes containing
    all the first ones as well as those from the second which are at
    least unique_cutoff percent different from all of the
    ones in the first list/any of those in the second list already chosen.
    
    usages is an optional parameter which is an indicator of 
    how often a new candidate haplotype is used in the generated
    haplotype list. It works with max_hap_add to ensure that if
    we have a limit on the maximum number of added haplotypes 
    we preferentially add the haplotypes that have highest usage
    
    Returns a dictionary detailing the new set of haplotypes
    as well as a dictionary showing where all the haps in the 
    second list map to in the new thing.

    """
    
    
    for x in initial_haps.keys(): #Get the legth of a haplotype
        haplotype_length = len(initial_haps[x])
        break
    
    if keep_flags is None:
        keep_flags = np.array([True for _ in range(haplotype_length)])
    
    if keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags,dtype=bool)
    
    keep_length = np.count_nonzero(keep_flags)
    
    new_haps_mapping = {-1:-1}

    cutoff = math.ceil(unique_cutoff*keep_length/100)
    
    i = 0
    j = 0
    num_added = 0
    
    cur_haps = {}
    for idx in initial_haps.keys():
        cur_haps[i] = initial_haps[idx]
        i += 1
    
    for identifier in new_candidate_haps.keys():
        hap = new_candidate_haps[identifier]
        hap_keep = hap[keep_flags]
        add = True
        for k in range(len(cur_haps)):
            compare = cur_haps[k]
            compare_keep = compare[keep_flags]
            
            distance = analysis_utils.calc_distance(hap_keep,compare_keep,calc_type="haploid")
            
            if distance < cutoff:
                add = False
                new_haps_mapping[j] = k
                j += 1
                break
            
        if add and num_added < max_hap_add:
            cur_haps[i] = hap
            new_haps_mapping[j] = i
            i += 1
            j += 1
            num_added += 1
        if num_added >= max_hap_add:
            break
    
    return (cur_haps,new_haps_mapping)

def add_distinct_haplotypes_smart(initial_haps,
                        new_candidate_haps,
                        probs_array,
                        keep_flags=None,
                        loss_reduction_cutoff_ratio =0.98,
                        use_multiprocessing=False):
    """
    Takes two lists of haplotypes and creates a new dictionary of
    haplotypes containing all the first ones as well as those
    from the second which are at least unique_cutoff percent
    different from all of the ones in the first list/any of 
    those in the second list already chosen.
    
    Returns a dictionary detailing the new set of haplotypes
    as well as a dictionary showing where all the haps in the 
    second list map to in the new thing.
    
    Alternate method to add_distinct_haplotypes that smartly looks at 
    which candidate hap will reduce the mean error the most, adds
    that to the list and continues until the reduction is too small
    to matter
    
    Unlike add_distinct_haplotypes not all candidate haplotypes get mapped,
    so there is no new_haps_mapping being returned

    """
    
    if use_multiprocessing:
        processing_pool = Pool(processes=32)
    
    for x in initial_haps.keys():
        orig_hap_len = len(initial_haps[x])
        break
    
    if keep_flags is None:
        keep_flags = np.array([True for _ in range(orig_hap_len)])
    
    if keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags,dtype=bool)
        
    if len(new_candidate_haps) == 0:
        return initial_haps
    
    i = 0
    cur_haps = {}
    for idx in initial_haps.keys():
        cur_haps[i] = initial_haps[idx]
        i += 1
        
    cur_matches = hap_statistics.match_best_vectorised(cur_haps,probs_array,keep_flags=keep_flags)
    cur_error = np.mean(cur_matches[2])
    
    
    
    candidate_haps = new_candidate_haps.copy()
    
    addition_complete = False
    
    while not addition_complete:
        cand_keys = list(candidate_haps.keys())
        
        if use_multiprocessing:
            addition_indicators = processing_pool.starmap(lambda x:
                            hap_statistics.get_addition_statistics(cur_haps,
                            candidate_haps,x,
                            probs_array,keep_flags=keep_flags),
                            zip(cand_keys))            
        else:
            addition_indicators = []
            for i in range(len(cand_keys)):
                addition_indicators.append(hap_statistics.get_addition_statistics(
                    cur_haps,candidate_haps,cand_keys[i],probs_array,keep_flags=keep_flags))
        
        smallest_result = min(addition_indicators,key=lambda x:x[0])
        smallest_index = addition_indicators.index(smallest_result)
        smallest_name = cand_keys[smallest_index]
        smallest_value = smallest_result[0]
        
        if smallest_value/cur_error < loss_reduction_cutoff_ratio:
            new_index = max(cur_haps.keys())+1
            cur_haps[new_index] = candidate_haps[smallest_name]
            candidate_haps.pop(smallest_name)
            
            cur_matches = hap_statistics.match_best_vectorised(cur_haps,probs_array,keep_flags=keep_flags)
            cur_error = np.mean(cur_matches[2])
            
            
        else:
            addition_complete = True
        
        if len(candidate_haps) == 0:
            addition_complete = True
            
    final_haps = {}
    i = 0
    for idx in cur_haps.keys():
        final_haps[i] = cur_haps[idx]
        i += 1
    
    return final_haps

def add_distinct_haplotypes_smart_vectorised(initial_haps,
                                   new_candidate_haps,
                                   probs_array,
                                   keep_flags=None,
                                   loss_reduction_cutoff_ratio=0.98,
                                   use_multiprocessing=False): # Argument kept for compatibility, but ignored
    """
    Fast, vectorized Greedy Forward Selection of haplotypes.
    Uses incremental score updates to avoid re-running full matching.
    
    Significantly faster than the non-vectorised version
    """
    
    # --- 1. SETUP AND PRE-PROCESSING ---
    
    # Copy inputs to avoid modifying originals
    cur_haps_dict = initial_haps.copy()
    cand_haps_dict = new_candidate_haps.copy()
    
    if not cand_haps_dict:
        return cur_haps_dict

    # Convert dicts to keys list and Matrix
    cur_keys = list(cur_haps_dict.keys())
    cand_keys = list(cand_haps_dict.keys())
    
    # Determine next available key index
    if cur_keys:
        next_key_idx = max(cur_keys) + 1
    else:
        next_key_idx = 0

    # Prepare Flags
    num_samples, num_sites, _ = probs_array.shape
    if keep_flags is None:
        keep_flags = np.ones(num_sites, dtype=bool)
    elif keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags, dtype=bool)
        
    # Slice Data (Shape: N x Masked_Sites x 3)
    probs_masked = probs_array[:, keep_flags, :]
    masked_len = probs_masked.shape[1]
    
    # Flatten Diploids for Matrix Mult (Shape: N x 3L)
    diploids_flat = probs_masked.reshape(num_samples, -1)
    
    # Prepare Haplotype Matrices (Shape: H x Masked_Sites x 2)
    # Current Haps
    if cur_keys:
        cur_haps_mat = np.array([cur_haps_dict[k][keep_flags] for k in cur_keys])
    else:
        cur_haps_mat = np.empty((0, masked_len, 2))
        
    # Candidate Haps
    cand_haps_mat = np.array([cand_haps_dict[k][keep_flags] for k in cand_keys])
    
    # Distance Weights (from your calc_distance logic)
    dist_weights = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])

    # --- 2. HELPER: BATCH SCORE CALCULATION ---
    def compute_pair_scores(hap_matrix_A, hap_matrix_B):
        """
        Computes distance between Samples and ALL combinations of A+B.
        Returns: (Num_Samples, Num_A, Num_B)
        """
        nA = len(hap_matrix_A)
        nB = len(hap_matrix_B)
        if nA == 0 or nB == 0: return np.zeros((num_samples, 0, 0))
        
        # Extract columns
        p0_A, p1_A = hap_matrix_A[:, :, 0], hap_matrix_A[:, :, 1]
        p0_B, p1_B = hap_matrix_B[:, :, 0], hap_matrix_B[:, :, 1]
        
        # Broadcast Combine (nA, nB, L)
        # Logic: P(00)=A0*B0, P(11)=A1*B1, P(01)=A0*B1 + A1*B0
        comb_00 = p0_A[:, None, :] * p0_B[None, :, :]
        comb_11 = p1_A[:, None, :] * p1_B[None, :, :]
        comb_01 = (p0_A[:, None, :] * p1_B[None, :, :]) + (p1_A[:, None, :] * p0_B[None, :, :])
        
        # Stack & Flatten for dot product
        # Stack: (nA, nB, L, 3) -> Reshape: (nA * nB, L, 3)
        comb_stacked = np.stack([comb_00, comb_01, comb_11], axis=-1)
        comb_list = comb_stacked.reshape(nA * nB, masked_len, 3)
        
        # Apply Weights
        comb_weighted = comb_list @ dist_weights # (nA*nB, L, 3)
        comb_flat = comb_weighted.reshape(nA * nB, masked_len * 3)
        
        # Dot Product with Diploids
        # (N, 3L) @ (nA*nB, 3L).T -> (N, nA*nB)
        scores_flat = diploids_flat @ comb_flat.T
        
        # Reshape back to (N, nA, nB)
        scores = scores_flat.reshape(num_samples, nA, nB)
        
        # Normalize (100/L) assuming original logic used full length
        # Note: You likely want to normalize by masked_len here, but sticking to input
        scores *= (100.0 / num_sites) 
        return scores

    # --- 3. INITIALIZE BASELINE SCORES ---
    # Calculate best score for every sample using ONLY current haplotypes
    if len(cur_keys) > 0:
        # Self pairs (cur + cur) and Cross pairs (cur + cur)
        # We just compute all cur vs cur
        base_dists = compute_pair_scores(cur_haps_mat, cur_haps_mat) # (N, H_cur, H_cur)
        # Flatten last two dims to find min across all combinations
        base_dists_flat = base_dists.reshape(num_samples, -1)
        current_best_errors = np.min(base_dists_flat, axis=1) # Vector (N,)
    else:
        current_best_errors = np.full(num_samples, np.inf)

    current_mean_error = np.mean(current_best_errors) if len(cur_keys) > 0 else np.inf

    # --- 4. INITIALIZE CANDIDATE POTENTIALS ---
    # We need to know: For every candidate, what is the best score a sample COULD get
    # if that candidate were added?
    # Possibilities for Candidate C:
    # 1. C + C (Homozygous)
    # 2. C + Existing_Hap_i (Heterozygous)
    
    num_cand = len(cand_keys)
    
    # 4a. Homozygous Scores (Cand + Cand)
    # (N, Num_Cand, Num_Cand) -> We only need diagonal (N, Num_Cand)
    # Optimization: Just compute diagonal directly to save memory? 
    # For simplicity, we reuse the function but inputs are (Cand, Cand). 
    # Actually, calculating C vs C for all pairs is wasteful. 
    # Let's just compute C vs C (diagonal) manually to be fast.
    
    # Optimized Homozygous Calc:
    c_p0, c_p1 = cand_haps_mat[:, :, 0], cand_haps_mat[:, :, 1]
    # Homozygous: 00=p0*p0, 11=p1*p1, 01=2*p0*p1
    homo_00 = c_p0 * c_p0
    homo_11 = c_p1 * c_p1
    homo_01 = 2 * c_p0 * c_p1
    homo_comb = np.stack([homo_00, homo_01, homo_11], axis=-1) @ dist_weights
    homo_flat = homo_comb.reshape(num_cand, -1)
    homo_scores = (diploids_flat @ homo_flat.T) * (100.0 / num_sites) # (N, Num_Cand)
    
    # 4b. Heterozygous Scores (Cand + Existing)
    if len(cur_keys) > 0:
        hetero_scores_3d = compute_pair_scores(cand_haps_mat, cur_haps_mat) # (N, Cand, Cur)
        # Best hetero score for each candidate across all existing haps
        hetero_best = np.min(hetero_scores_3d, axis=2) # (N, Cand)
    else:
        hetero_best = np.full((num_samples, num_cand), np.inf)

    # 4c. Best possible score for each candidate (Self OR Pair with Existing)
    cand_best_potentials = np.minimum(homo_scores, hetero_best) # (N, Num_Cand)
    
    # Active mask for candidates (True = still available)
    cand_active = np.ones(num_cand, dtype=bool)
    
    # --- 5. GREEDY SELECTION LOOP ---
    
    while True:
        # If no candidates left
        if not np.any(cand_active):
            break
            
        # A. CALCULATE IMPROVEMENT FOR ALL ACTIVE CANDIDATES
        # New Error if we add Cand C = min(Current_Best, Potential_Best_C)
        # We broadcast Current_Best (N, 1) against Potentials (N, Cand)
        
        active_indices = np.where(cand_active)[0]
        
        # Compare current error vs candidate potentials
        # Shape: (N, Num_Active)
        potential_errors = np.minimum(
            current_best_errors[:, np.newaxis], 
            cand_best_potentials[:, active_indices]
        )
        
        # Mean error for each candidate
        potential_means = np.mean(potential_errors, axis=0)
        
        # Find best candidate
        best_local_idx = np.argmin(potential_means)
        best_global_idx = active_indices[best_local_idx]
        best_new_mean = potential_means[best_local_idx]
        
        # B. CHECK CUTOFF
        if current_mean_error == 0 or (best_new_mean / current_mean_error) >= loss_reduction_cutoff_ratio:
            # Improvement too small, stop
            break
            
        # C. COMMIT THE ADDITION
        # Add best candidate to results
        best_cand_key = cand_keys[best_global_idx]
        cur_haps_dict[next_key_idx] = cand_haps_dict[best_cand_key]
        
        # Extract the matrix for the new winner
        new_hap_vec = cand_haps_mat[best_global_idx][np.newaxis, :, :] # (1, L, 2)
        
        # Update Global Current Errors
        # The new baseline is the best we found with this candidate
        current_best_errors = potential_errors[:, best_local_idx]
        current_mean_error = best_new_mean
        
        # Mark candidate as used
        cand_active[best_global_idx] = False
        
        # Increment key for next time
        next_key_idx += 1
        
        # D. INCREMENTAL UPDATE OF POTENTIALS
        # Crucial Step: The remaining candidates might now pair better with the 
        # *newly added* haplotype than they did with previous ones.
        # We calculate (Remaining_Cands + New_Winner) scores.
        
        if not np.any(cand_active):
            break
            
        remaining_indices = np.where(cand_active)[0]
        remaining_cands_mat = cand_haps_mat[remaining_indices] # (R, L, 2)
        
        # Compute scores: (Remaining) + (New_Winner)
        # Result: (N, Remaining, 1) -> Flatten to (N, Remaining)
        new_pairing_scores = compute_pair_scores(remaining_cands_mat, new_hap_vec)
        new_pairing_scores = new_pairing_scores[:, :, 0]
        
        # Update potentials: min(Old_Potential, New_Pairing)
        cand_best_potentials[:, remaining_indices] = np.minimum(
            cand_best_potentials[:, remaining_indices],
            new_pairing_scores
        )

    return cur_haps_dict

def truncate_haps(candidate_haps,
                  candidate_matches,
                  probs_array,
                  max_cutoff_error_increase=1.1,
                  use_multiprocessing=False):
    """
    Truncate a list of haplotypes so that only the necessary ones remain
    """
    
    if use_multiprocessing:
        processing_pool = Pool(processes=32)
    
    cand_copy = candidate_haps.copy()
    cand_matches = candidate_matches
    used_haps = cand_matches[1].keys()
    
    starting_error = np.mean(cand_matches[2])
    
    for hap in list(cand_copy.keys()):
        if hap not in used_haps:
            cand_copy.pop(hap)
            
    haps_names = list(cand_copy.keys())
    
    errors_limit = max_cutoff_error_increase*starting_error
    
    truncation_complete = False
    
    while not truncation_complete:
        
        if use_multiprocessing:
            removal_indicators = processing_pool.starmap(lambda x:
                            hap_statistics.get_removal_statistics(cand_copy,
                            cand_matches,x,probs_array),
                            zip(haps_names))
        else:
            removal_indicators = []
            
            for i in range(len(haps_names)):
                removal_indicators.append(hap_statistics.get_removal_statistics(
                    cand_copy,cand_matches,haps_names[i],probs_array))
        
        smallest_value = min(removal_indicators,key=lambda x:x[0])
        smallest_index = removal_indicators.index(smallest_value)
        
        if removal_indicators[smallest_index][0] > errors_limit:
            truncation_complete = True
        else:
            hap_name = haps_names[smallest_index]
            cand_copy.pop(hap_name)
            cand_matches = smallest_value[3]
            errors_limit = max_cutoff_error_increase*smallest_value[0]
            haps_names = list(cand_copy.keys())
            
    final_haps = {}
    i = 0
    for j in cand_copy.keys():
        final_haps[i] = cand_copy[j]
        i += 1
    
    return final_haps

def get_initial_haps(site_priors,
                     probs_array,
                     keep_flags=None,
                     het_cutoff_start_percentage=10,
                     het_excess_add=2,
                     het_max_cutoff_percentage=20,
                     deeper_analysis=False,
                     uniqueness_tolerance=5,
                     read_error_prob = 0.02,
                     verbose=False):
    """
    Get our initial haplotypes by finding high homozygosity samples
    
    keep_flags is a optional boolean array which is 1 at those sites which
    we wish to consider for purposes of the analysis and 0 elsewhere,
    if not provided we use all sites
    """
    if keep_flags is None:
        keep_flags = np.array([True for _ in range(probs_array.shape[1])])
    
    if keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags,dtype=bool)
        
        

    het_vals = np.array([analysis_utils.get_heterozygosity(probs_array[i],keep_flags) for i in range(len(probs_array))])

    found_homs = False
    cur_het_cutoff = het_cutoff_start_percentage
    
    accept_singleton = (het_cutoff_start_percentage+het_max_cutoff_percentage)/2
    
    while not found_homs:
        
        if cur_het_cutoff > het_max_cutoff_percentage:
            if verbose:
                print("Unable to find samples with high homozygosity in region")
            
            base_probs = []
            for i in range(len(site_priors)):
                ref_prob = math.sqrt(site_priors[i,0])
                alt_prob = 1-ref_prob
                base_probs.append([ref_prob,alt_prob])
            base_probs = np.array(base_probs)
            
            return {0:base_probs}
        
        #print("HV",het_vals)
        
        homs_where = np.where(het_vals <= cur_het_cutoff)[0]
    
        homs_array = probs_array[homs_where]
        #corresp_reads_array = reads_array[homs_where]
        
        if len(homs_array) < 5:
            cur_het_cutoff += het_excess_add
            continue
        
        
        homs_array[:,:,0] += 0.5*homs_array[:,:,1]
        homs_array[:,:,2] += 0.5*homs_array[:,:,1]
        
        homs_array = homs_array[:,:,[0,2]]
    
        dist_submatrix = analysis_utils.generate_distance_matrix(
            homs_array,keep_flags=keep_flags,
            calc_type="haploid")        
        
        #First do clustering looking for at least 2 clusters, if that fails rerun allowing single clusters
        try:
            initial_clusters = hdbscan_cluster(
                                dist_submatrix,
                                min_cluster_size=2,
                                min_samples=1,
                                cluster_selection_method="eom",
                                alpha=1.0)
            num_clusters = 1+np.max(initial_clusters[0])
            
            if num_clusters == 0:
                assert False
        except:
            if cur_het_cutoff < accept_singleton:
                cur_het_cutoff += het_excess_add
                continue
            initial_clusters = hdbscan_cluster(
                                dist_submatrix,
                                min_cluster_size=2,
                                min_samples=1,
                                cluster_selection_method="eom",
                                alpha=1.0,
                                allow_single_cluster=True)
            num_clusters = 1+np.max(initial_clusters[0])

        
        #If we don't find any clusters the increase the het threshold and repeat
        if num_clusters < 1:
            cur_het_cutoff += het_excess_add
            continue
        else:
            found_homs = True
    
    representatives = get_representatives_probs(site_priors,
        homs_array,initial_clusters[0],
        read_error_prob=read_error_prob)
    
    #Hacky way to remove any haps that are too close to each other
    (representatives,label_mappings) = add_distinct_haplotypes(
              {},representatives,keep_flags=keep_flags,
              unique_cutoff=uniqueness_tolerance)
        
    return representatives

def generate_further_haps(site_priors,
                          probs_array,
                          initial_haps,
                          keep_flags=None,
                          wrongness_cutoff=10,
                          uniqueness_threshold=5,
                          max_hap_add = 1000,
                          verbose=False):
    """
    Given a genotype array and a set of initial haplotypes
    which are present in some of the samples of the array
    calculates other new haplotypes which are also present.
    
    het_cutoff is the maximum percentage of sites which are not 0,1
    for a candidate hap to consider it further as a new haplotype
    
    uniqueness_threshold is a percentage lower bound of how different
    a new candidate hap has to be from all the initial haps to be considered
    further.
    
    max_hap_add is the maximum number of additional haplotypes to add
    
    """
    if keep_flags is None:
        keep_flags = np.array([True for _ in range(probs_array.shape[1])])
    
    if keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags,dtype=bool)
        
    candidate_haps = []
    initial_list = np.array(list(initial_haps.values()))
    
    for geno in probs_array:
        for init_hap in initial_list:
            
            (fixed_diff,wrongness) = analysis_utils.get_diff_wrongness(geno,init_hap,keep_flags=keep_flags)
            
            if wrongness <= wrongness_cutoff:

                add = True
                for comp_hap in initial_list:
                    
                    fixed_keep = fixed_diff[keep_flags]
                    comp_keep = comp_hap[keep_flags]
                    
                    perc_diff = 100*analysis_utils.calc_distance(fixed_keep,comp_keep,calc_type="haploid")/len(comp_keep)
                    
                    
                    if perc_diff < uniqueness_threshold:
                        add = False
                        break
                
                if add:
                    candidate_haps.append(fixed_diff)
    
    candidate_haps = np.array(candidate_haps)
    
    if len(candidate_haps) == 0:
        if verbose:
            print("Unable to find candidate haplotypes when generating further haps")
        print("No extra haps found")
        return initial_haps
    
    
    dist_submatrix = analysis_utils.generate_distance_matrix(
        candidate_haps,keep_flags=keep_flags,
        calc_type="haploid")

    if len(candidate_haps) > 1:
        initial_clusters = hdbscan_cluster(
                            dist_submatrix,
                            min_cluster_size=len(initial_haps)+1,
                            min_samples=1,
                            cluster_selection_method="eom",
                            alpha=1.0)
    else: #Here candidate haps will have length = 1
        final_haps = add_distinct_haplotypes_smart(initial_haps,
                    {0:candidate_haps[0]},probs_array,
                    keep_flags=keep_flags)

        return final_haps
    
    
    representatives = get_representatives_probs(
        site_priors,candidate_haps,initial_clusters[0])
    

    final_haps = add_distinct_haplotypes_smart_vectorised(initial_haps,
                representatives,probs_array,
                keep_flags=keep_flags)
    
    return final_haps

def generate_further_haps_vectorised(site_priors,
                          probs_array,
                          initial_haps,
                          keep_flags=None,
                          wrongness_cutoff=10,
                          uniqueness_threshold=5,
                          max_hap_add=1000,
                          verbose=False):
    """
    Fully vectorized version of generate_further_haps.
    Mathematically identical to the loop-based version but 100x faster.
    """
    
    # --- 1. PREPARE DATA ---
    num_samples, num_sites, _ = probs_array.shape
    
    if keep_flags is None:
        keep_flags = np.ones(num_sites, dtype=bool)
    elif keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags, dtype=bool)
        
    # Slice to keep only relevant sites (Speeds up math significantly)
    # Diploid columns: 0=Ref(00), 1=Het(01), 2=Alt(11)
    probs_masked = probs_array[:, keep_flags, :]
    masked_sites_len = probs_masked.shape[1]
    
    # Haploid columns: 0=Ref(0), 1=Alt(1)
    # Convert dict values to array: (Num_Haps, Masked_Sites, 2)
    initial_list = np.array(list(initial_haps.values()))
    initial_list_masked = initial_list[:, keep_flags, :]
    
    # --- 2. BROADCASTING SETUP ---
    # We want to compare Every Sample (N) vs Every Hap (K)
    # D: Diploids (N, 1, L)
    # H: Haplotypes (1, K, L)
    
    D0 = probs_masked[:, :, 0][:, np.newaxis, :] # P(Dip=00)
    D1 = probs_masked[:, :, 1][:, np.newaxis, :] # P(Dip=01)
    D2 = probs_masked[:, :, 2][:, np.newaxis, :] # P(Dip=11)
    
    H0 = initial_list_masked[:, :, 0][np.newaxis, :, :] # P(Hap=0)
    H1 = initial_list_masked[:, :, 1][np.newaxis, :, :] # P(Hap=1)
    
    # --- 3. CALCULATE WRONGNESS (Vectorized) ---
    # Formula derived from your code: Wrong = (H1 * D0) + (H0 * D2)
    # Shape: (N, K, L)
    wrong_per_site = (H1 * D0) + (H0 * D2)
    
    # Sum over sites to get score per pair: Shape (N, K)
    # Perc Wrong = 100 * Sum(Wrong) / Sites
    wrongness_scores = (np.sum(wrong_per_site, axis=2) * 100.0) / masked_sites_len
    
    # Find valid pairs
    # valid_indices is a tuple (sample_indices, hap_indices)
    valid_mask = wrongness_scores <= wrongness_cutoff
    
    if not np.any(valid_mask):
        if verbose: print("No extra haps found (wrongness check).")
        return initial_haps

    # --- 4. CALCULATE NEW CANDIDATES (Only for valid pairs) ---
    # We only compute the full arrays for the pairs that passed the check
    # to save memory.
    
    # Extract the specific rows/cols needed
    idx_s, idx_h = np.where(valid_mask)
    
    # D_subset: (Num_Valid, L)
    D0_sub = probs_masked[idx_s, :, 0]
    D1_sub = probs_masked[idx_s, :, 1]
    D2_sub = probs_masked[idx_s, :, 2]
    
    # H_subset: (Num_Valid, L)
    # Note: initial_list_masked shape is (K, L, 2)
    H0_sub = initial_list_masked[idx_h, :, 0]
    H1_sub = initial_list_masked[idx_h, :, 1]
    
    # Apply your exact algebra derived from 'einsum':
    # New_0 = D0 + H1*D1
    # New_1 = D2 + H0*D1
    new_haps_0 = D0_sub + (H1_sub * D1_sub)
    new_haps_1 = D2_sub + (H0_sub * D1_sub)
    
    # Stack back to (Num_Valid, L, 2)
    candidate_haps_masked = np.stack([new_haps_0, new_haps_1], axis=-1)
    
    # --- 5. UNIQUENESS CHECK ---
    # We check if candidates are distinct from ALL initial haps
    # Compare Candidates (Num_Valid) vs Initial (Num_Init)
    
    # We use P(1) (the second column) for distance checks, essentially Manhattan distance
    cand_p1 = candidate_haps_masked[:, :, 1]
    init_p1 = initial_list_masked[:, :, 1] # (K, L)
    
    # cdist 'cityblock' = sum(|a - b|)
    # Result: (Num_Candidates, Num_Init_Haps)
    dists = cdist(cand_p1, init_p1, metric='cityblock')
    perc_dists = (dists * 100.0) / masked_sites_len
    
    # For each candidate, find the closest distance to ANY existing hap
    min_dist_to_existing = np.min(perc_dists, axis=1)
    
    # Keep only those sufficiently far away
    is_unique = min_dist_to_existing >= uniqueness_threshold
    
    unique_candidates = candidate_haps_masked[is_unique]
    
    if len(unique_candidates) == 0:
        if verbose: print("No extra haps found (uniqueness check).")
        return initial_haps

    # Cap number of additions
    if len(unique_candidates) > max_hap_add:
        unique_candidates = unique_candidates[:max_hap_add]

    # --- 6. CLUSTERING & CLEANUP ---
    # Use the P(1) column for clustering distance
    unique_p1 = unique_candidates[:, :, 1]
    
    # Distance between NEW candidates
    candidate_dist_matrix = cdist(unique_p1, unique_p1, metric='cityblock')

    if len(unique_candidates) > 1:
        try:
            initial_clusters = hdbscan_cluster(
                                candidate_dist_matrix,
                                min_cluster_size=len(initial_haps)+1,
                                min_samples=1,
                                cluster_selection_method="eom",
                                alpha=1.0)
            cluster_labels = initial_clusters[0]
        except:
             cluster_labels = np.zeros(len(unique_candidates), dtype=int)
    else:
         cluster_labels = np.array([0])

    # Reinflate to full size (Num_Sites, 2) for get_representatives
    # We fill masked sites with 0 (or site priors if you prefer, but 0 is safe for 'add')
    final_candidates_full = np.zeros((len(unique_candidates), num_sites, 2))
    final_candidates_full[:, keep_flags, :] = unique_candidates
    
    # If using site_priors for masked regions is required by get_representatives:
    # (Uncomment if masked regions need valid probs)
    # inverse_mask = ~keep_flags
    # if np.any(inverse_mask):
    #     final_candidates_full[:, inverse_mask, :] = site_priors[inverse_mask]
        
    representatives = get_representatives_probs(
        site_priors, final_candidates_full, cluster_labels)
    
    final_haps = add_distinct_haplotypes_smart_vectorised(initial_haps,
                representatives, probs_array,
                keep_flags=keep_flags)
    
    return final_haps

def generate_haplotypes_block(positions, reads_array, keep_flags=None,
                              error_reduction_cutoff=0.98,
                              max_cutoff_error_increase=1.02,
                              max_hapfind_iter=5,
                              deeper_analysis_initial=False,
                              min_num_haps=0,
                              penalty_strength=5.0,
                              max_intermediate_haps=25,
                              known_haplotypes=None): # NEW ARGUMENT
    """
    Given the read count array of our sample data for a single block
    generates the haplotypes that make up the samples present in our data.
    
    known_haplotypes: Optional list or dict of haplotypes (arrays) that 
                      must be included in the starting search pool.
                      Useful for injecting missing founders found via residual analysis.
    """
    
    # --- 1. SETUP & INITIALIZATION ---
    if keep_flags is None:
        keep_flags = np.array([1 for _ in range(reads_array.shape[1])])
    
    if keep_flags.dtype != int:
        keep_flags = np.array(keep_flags, dtype=int)
    
    # Calculate probabilities from reads
    (site_priors, (probs_array, ploidy)) = analysis_utils.reads_to_probabilities(reads_array)
    
    # Get seed haplotypes from homozygotes
    initial_haps = get_initial_haps(site_priors, probs_array, keep_flags=keep_flags)
    
    if len(positions) == 0:
        return (np.array([]), np.array([]), np.array([]), initial_haps)

    # --- 2. INJECT KNOWN HAPLOTYPES (NEW LOGIC) ---
    if known_haplotypes is not None:
        # Normalize input to list of arrays
        if isinstance(known_haplotypes, dict):
            known_list = list(known_haplotypes.values())
        elif isinstance(known_haplotypes, list):
            known_list = known_haplotypes
        else:
            known_list = []
            
        if len(known_list) > 0:
            # Combine Auto-Detected seeds with Injected seeds
            combined_list = list(initial_haps.values()) + known_list
            
            # Consolidate to remove duplicates (e.g. if Injected == Auto-Detected)
            # This ensures we start with a clean, unique set
            initial_haps = consolidate_similar_candidates(combined_list, diff_threshold=0.01)
            
            # print(f"  Injected {len(known_list)} known haplotypes. Starting set size: {len(initial_haps)}")

    # --- 3. INITIAL STATISTICS ---
    initial_matches = hap_statistics.match_best_vectorised(initial_haps, probs_array, keep_flags=keep_flags)
    initial_error = np.mean(initial_matches[2])
    
    matches_history = [initial_matches]
    errors_history = [initial_error]
    haps_history = [initial_haps]
    
    all_found = False
    cur_haps = initial_haps
    
    minimum_strikes = 0 
    striking_up = False
    uniqueness_threshold = 5
    wrongness_cutoff = 10
    
    print(f"Processing Block {positions[0]} (Size: {len(positions)} sites)...")
    
    # --- 4. ITERATIVE DISCOVERY LOOP ---
    while not all_found:
        
        # A. Generate new candidates based on residuals
        cur_haps = generate_further_haps_vectorised(
            site_priors, 
            probs_array,
            cur_haps,
            keep_flags=keep_flags,
            uniqueness_threshold=uniqueness_threshold,
            wrongness_cutoff=wrongness_cutoff
        )
        
        # B. INTERMEDIATE PRUNING
        if len(cur_haps) > max_intermediate_haps:
            keep_keys = select_optimal_haplotype_set_viterbi(
                cur_haps, 
                probs_array, 
                recomb_penalty=10.0,
                penalty_strength=0.5 
            )
            pruned_haps = {i: cur_haps[k] for i, k in enumerate(keep_keys)}
            
            if len(pruned_haps) < len(cur_haps):
                cur_haps = pruned_haps

        # C. Evaluate Improvement
        cur_matches = hap_statistics.match_best_vectorised(cur_haps, probs_array)
        cur_error = np.mean(cur_matches[2])
        
        # D. Convergence Checks
        if cur_error/errors_history[-1] >= error_reduction_cutoff and len(errors_history) >= 2:
            if len(cur_haps) >= min_num_haps or minimum_strikes >= 3:
                all_found = True
                break
            else:
                minimum_strikes += 1
                uniqueness_threshold -= 1
                wrongness_cutoff += 2
                striking_up = True
                
        if len(cur_haps) == len(haps_history[-1]) and not striking_up: 
            all_found = True
            break
            
        if len(errors_history) > max_hapfind_iter + 1:
            all_found = True
            
        matches_history.append(cur_matches)
        errors_history.append(cur_error)
        haps_history.append(cur_haps)
        striking_up = False
    
    # --- 5. FINAL CLEANUP STEPS ---
    
    candidates_to_filter = haps_history[-1]
    
    if len(candidates_to_filter) > 1:
        
        # Step A: Pre-Merge
        merged_haps = consolidate_similar_candidates(candidates_to_filter, diff_threshold=0.02)
        
        # Step B: Viterbi-BIC Selection
        best_keys = select_optimal_haplotype_set_viterbi(
            merged_haps,
            probs_array,
            recomb_penalty=10.0,
            penalty_strength=penalty_strength 
        )
        
        selected_haps_dict = {i: merged_haps[k] for i, k in enumerate(best_keys)}
        
        # Step C: Post-Usage Pruning
        final_matches = hap_statistics.match_best_vectorised(selected_haps_dict, probs_array)
        usage_counts = final_matches[1] 
        min_samples = max(2, int(probs_array.shape[0] * 0.01))
        
        used_haps = {}
        new_idx = 0
        for h_idx, count in usage_counts.items():
            if count >= min_samples:
                used_haps[new_idx] = selected_haps_dict[h_idx]
                new_idx += 1
        
        # Step D: Chimera Pruning
        final_haps = prune_chimeras(
            used_haps, 
            probs_array, 
            recomb_penalty=10.0, 
            error_tolerance_sites=10
        )
        
        final_haps = {i: v for i, v in enumerate(final_haps.values())}
        
        print(f"Block {positions[0]}: Candidates {len(candidates_to_filter)} -> Merged {len(merged_haps)} -> Selected {len(selected_haps_dict)} -> Used {len(used_haps)} -> Final {len(final_haps)}")
        
    else:
        final_haps = candidates_to_filter

    return (positions, keep_flags, reads_array, final_haps)

def find_missing_haplotypes_iterative(positions, reads_array, current_haps, 
                                      keep_flags=None,
                                      error_threshold_percent=2.0, 
                                      min_bad_samples=5):
    """
    Analyzes the fit of the current haplotypes to the data.
    Identifies samples with high error rates, isolates them, and runs
    a targeted haplotype generation on that subset to find missing founders.
    
    Args:
        current_haps: Dict of currently found haplotypes.
        error_threshold_percent: Samples with error > this are considered "unexplained".
        min_bad_samples: Need at least this many unexplained samples to trigger a search.
        
    Returns:
        dict: A dictionary of NEW, DISTINCT haplotypes found in the residuals.
              (Returns empty dict if no new haplotypes found).
    """
    
    if len(current_haps) == 0:
        return {}

    # 1. Setup Data
    if keep_flags is None:
        keep_flags = np.array([1 for _ in range(reads_array.shape[1])])
    
    # Calculate probabilities for the full set
    (_, (probs_array, _)) = analysis_utils.reads_to_probabilities(reads_array)
    
    # 2. Check Fit using K-Limited (Allowing recombinations)
    # We use k-limited because if a sample is a recombinant of known haps, 
    # it is NOT missing a founder. We only want samples that truly don't fit.
    match_results = hap_statistics.match_best_k_limited(
        current_haps, 
        probs_array, 
        keep_flags=keep_flags,
        max_recombinations=2
    )
    
    # Extract error scores (Index 2 of return tuple)
    error_scores = match_results[2]
    
    # 3. Identify Outliers
    bad_fit_indices = np.where(error_scores > error_threshold_percent)[0]
    num_bad = len(bad_fit_indices)
    
    print(f"Residual Analysis: {num_bad} samples ({num_bad/len(reads_array)*100:.1f}%) have >{error_threshold_percent}% error.")
    
    if num_bad < min_bad_samples:
        print("  -> Too few outliers to constitute a missing founder. Skipping.")
        return {}
        
    # 4. Run Targeted Generation
    print("  -> Running targeted search on outlier subset...")
    
    subset_reads = reads_array[bad_fit_indices]
    
    # We use LENIENT settings here. We want to pick up *any* signal in this noise.
    # penalty_strength=0.5 (Lenient)
    # error_tolerance=10 (Forgiving of noise)
    
    _, _, _, sub_result_haps = generate_haplotypes_block(
        positions,
        subset_reads,
        keep_flags=keep_flags,
        penalty_strength=0.5, # Very lenient
        min_num_haps=1,       # Just find what's there
        max_intermediate_haps=100 # Allow exploration
    )
    
    # 5. Filter Redundancy
    # The sub-search might just rediscover 'Hap 0' but with some noise.
    # We only want to return haps that are significantly different from what we already have.
    
    existing_matrix = np.array(list(current_haps.values()))
    newly_found_unique = {}
    new_idx = 0
    
    for k, sub_hap in sub_result_haps.items():
        # Check against EXISTING founders
        diffs = np.mean(existing_matrix != sub_hap, axis=1)
        min_diff = np.min(diffs)
        
        # Threshold: Must be at least 2% different from any existing founder
        if min_diff > 0.02:
            newly_found_unique[new_idx] = sub_hap
            new_idx += 1
            # print(f"  -> Found NEW distinct haplotype (Diff: {min_diff*100:.1f}%)")
        # else:
            # print(f"  -> Rediscovered existing haplotype (Diff: {min_diff*100:.1f}%). Ignoring.")
            
    print(f"  -> Residual Analysis yielded {len(newly_found_unique)} new valid haplotypes.")
    
    return newly_found_unique

def generate_haplotypes_block_robust(positions, reads_array, keep_flags=None, 
                                     max_robust_passes=3, # Safety limit
                                     **kwargs):
    """
    Wrapper that runs the standard generation, checks for missing data (residuals),
    and re-runs iteratively until all distinct founders are found.
    """
    
    # Track known haplotypes across iterations
    # We start with whatever was passed in kwargs, or empty list
    current_known_haps = kwargs.get('known_haplotypes', [])
    if isinstance(current_known_haps, dict):
        current_known_haps = list(current_known_haps.values())
    
    # Store the final result variables
    final_res_pos = positions
    final_res_flags = keep_flags
    final_res_reads = reads_array
    final_haps_dict = {}
    
    for pass_num in range(1, max_robust_passes + 1):
        print(f"\n--- Robust Pass {pass_num}: Generating with {len(current_known_haps)} known injected haps ---")
        
        # 1. Run Generation (passing the accumulated known list)
        # We update kwargs to include the current known list
        run_kwargs = kwargs.copy()
        run_kwargs['known_haplotypes'] = current_known_haps
        
        (res_pos, res_flags, res_reads, generated_haps) = generate_haplotypes_block(
            positions, reads_array, keep_flags=keep_flags, **run_kwargs
        )
        
        # Update our current best result
        final_res_pos = res_pos
        final_res_flags = res_flags
        final_res_reads = res_reads
        final_haps_dict = generated_haps
        
        # 2. Residual Analysis
        # Check if there are still samples that fit poorly
        missing_haps_dict = find_missing_haplotypes_iterative(
            positions, 
            reads_array, 
            final_haps_dict, 
            keep_flags=keep_flags,
            error_threshold_percent=2.0,
            min_bad_samples=5 # Don't re-run for < 5 samples (likely just noise)
        )
        
        # 3. Decision
        if len(missing_haps_dict) == 0:
            print("--- Convergence reached: No significant missing haplotypes found in residuals. ---")
            break
        else:
            print(f"--- Detected {len(missing_haps_dict)} missing haplotypes in residuals. Preparing next pass... ---")
            
            # Add new findings to our known list for the next pass
            # We convert dict values to a list and extend
            new_haps_list = list(missing_haps_dict.values())
            
            # Consolidate to ensure we don't just add duplicates
            # (Combine current known + newly found)
            combined_pool = current_known_haps + new_haps_list
            consolidated_dict = consolidate_similar_candidates(combined_pool, diff_threshold=0.01)
            current_known_haps = list(consolidated_dict.values())
            
            if pass_num == max_robust_passes:
                print("--- Max passes reached. Returning best effort. ---")

    return (final_res_pos, final_res_flags, final_res_reads, final_haps_dict)

def generate_haplotypes_all(positions_data, reads_array_data, keep_flags_data=None, 
                            penalty_strength=1.0, 
                            max_intermediate_haps=100):
    """
    Generate a list of block haplotypes using multiprocessing.
    
    Args:
        penalty_strength: Controls strictness of final Viterbi selection (1.0 = standard).
        max_intermediate_haps: Trigger for intermediate pruning during the loop.
    
    Assumes OMP_NUM_THREADS=1 is set in the main script.
    """
    
    if keep_flags_data is None:
        keep_flags_data = [None] * len(positions_data)

    # Combine the varying arguments (Position, Reads, Flags) into tuples
    tasks = zip(positions_data, reads_array_data, keep_flags_data)
    
    # 16 processes is usually safe given the new memory optimizations
    # (Viterbi selection reduces the output size significantly, lowering memory pressure during gather)
    num_processes = 16
    
    # Create a partial function with the fixed configuration arguments
    # When starmap unpacks the 'tasks', it will append these fixed args automatically
    worker_func = partial(
        generate_haplotypes_block_robust, 
        penalty_strength=penalty_strength, 
        max_intermediate_haps=max_intermediate_haps
    )
    
    # Use a context manager ('with') to ensure processes are cleaned up
    with Pool(processes=num_processes) as processing_pool:
        
        # starmap calls: worker_func(pos, reads, flags)
        # which internally becomes: generate_haplotypes_block(pos, reads, flags, penalty_strength=..., max=...)
        overall_haplotypes = processing_pool.starmap(
            worker_func, 
            tasks
        )

    return overall_haplotypes

#%%
def match_long_hap_to_blocks(long_hap,long_hap_sites,block_haps_data):
    """
    Function which takes a full long hap and the full block level haps
    data and computes which block haps matches best to the long hap
    at each block
    """
    best_matches = []
    lowest_diffs = []
    
    for i in range(len(block_haps_data)):
        current_sites = block_haps_data[i][0]

        current_data = analysis_utils.get_sample_data_at_sites(long_hap, long_hap_sites, current_sites)
    
        best_hap = -1
        lowest_diff = 100
        
        for j in range(len(block_haps_data[i][3])):
            diff = analysis_utils.calc_perc_difference(
                block_haps_data[i][3][j],
                current_data,calc_type="haploid")
            if diff < lowest_diff:
                lowest_diff = diff
                best_hap = j
        best_matches.append(best_hap)
        lowest_diffs.append(lowest_diff)
        
    return [best_matches,lowest_diffs]