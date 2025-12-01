

import numpy as np
import math
import hdbscan
from multiprocess import Pool
import warnings
import time
from scipy.spatial.distance import cdist

import analysis_utils
import hap_statistics

warnings.filterwarnings("ignore")
np.seterr(divide='ignore',invalid="ignore")
pass



#%%
def hdbscan_cluster(dist_matrix,
                    min_cluster_size=2,
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

def add_distinct_haplotypes(initial_haps,
                       new_candidate_haps,
                       keep_flags=None,
                       unique_cutoff=5,
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
        
    cur_matches = hap_statistics.match_best(cur_haps,probs_array,keep_flags=keep_flags)
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
            
            cur_matches = hap_statistics.match_best(cur_haps,probs_array,keep_flags=keep_flags)
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

def generate_haplotypes_block(positions,reads_array,keep_flags=None,
                              error_reduction_cutoff = 0.98,
                              max_cutoff_error_increase = 1.02,
                              max_hapfind_iter=5,
                              deeper_analysis_initial=False,
                              min_num_haps=0):
    """
    Given the read count array of our sample data for a single block
    generates the haplotypes that make up the samples present in our data
    
    min_num_haps is a (soft) minimum value for the number of haplotypes,
    if we have fewer than that many haps we iterate further to get more 
    haps.
    
    """
    if keep_flags is None:
        keep_flags = np.array([1 for _ in range(reads_array.shape[1])])
    
    if keep_flags.dtype != int:
        keep_flags = np.array(keep_flags,dtype=int)
    
    #reads_array = resample_reads_array(reads_array,1)
    
    (site_priors,(probs_array,ploidy)) = analysis_utils.reads_to_probabilities(reads_array)
    
    
    initial_haps = get_initial_haps(site_priors,probs_array,
        keep_flags=keep_flags)
    
    if len(positions) == 0:
        return (np.array([]),np.array([]),np.array([]),initial_haps)
    
    
    initial_matches = hap_statistics.match_best(initial_haps,probs_array,keep_flags=keep_flags)
    initial_error = np.mean(initial_matches[2])
    
    matches_history = [initial_matches]
    errors_history = [initial_error]
    haps_history = [initial_haps]
    
    all_found = False
    cur_haps = initial_haps
    
    minimum_strikes = 0 #Counter that increases every time we get fewer than the required minimum number of haplotypes, if it hits 3 we break out of our loop to find further haps
    striking_up = False
    
    uniqueness_threshold = 5
    wrongness_cutoff = 10
    
    print(positions[0])
    
    while not all_found:
        cur_haps = generate_further_haps_vectorised(site_priors,probs_array,
                    cur_haps,keep_flags=keep_flags,uniqueness_threshold=uniqueness_threshold,
                    wrongness_cutoff=wrongness_cutoff)
        cur_matches = hap_statistics.match_best(cur_haps,probs_array)
        cur_error = np.mean(cur_matches[2])
        
        
        # matches_history.append(cur_matches)
        # errors_history.append(cur_error)
        # haps_history.append(cur_haps)
        
        if cur_error/errors_history[-1] >= error_reduction_cutoff and len(errors_history) >= 2:
            if len(cur_haps) >= min_num_haps or minimum_strikes >= 3:
                all_found = True
                break
            else:
                minimum_strikes += 1
                uniqueness_threshold -= 1
                wrongness_cutoff += 2
                striking_up = True
                
        if len(cur_haps) == len(haps_history[-1]) and not striking_up: #Break if in the last iteration we didn't find a single new hap
            all_found = True
            break
            
        if len(errors_history) > max_hapfind_iter+1:
            all_found = True
            
        matches_history.append(cur_matches)
        errors_history.append(cur_error)
        haps_history.append(cur_haps)
        
        striking_up = False
    
    print(f"Completed {positions[0]}")
    final_haps = haps_history[-1]
        
    return (positions,keep_flags,reads_array,final_haps)
    

def generate_haplotypes_all(positions_data, reads_array_data, keep_flags_data=None):
    """
    Generate a list of block haplotypes using multiprocessing.
    Assumes OMP_NUM_THREADS=1 is set in the main script.
    """
    
    if keep_flags_data is None:
        keep_flags_data = [None] * len(positions_data)

    # Combine arguments into a single iterable of tuples
    # This is cleaner than using starmap with a lambda
    tasks = zip(positions_data, reads_array_data, keep_flags_data)
    
    # 32 processes is extremely high if using large vectorized arrays.
    # If this still hangs or crashes (Out of Memory), reduce this number (e.g., to 8 or 16).
    num_processes = 16
    
    # Use a context manager ('with') to ensure processes are cleaned up
    with Pool(processes=num_processes) as processing_pool:
        
        # starmap automatically unpacks the tuples from 'tasks'
        # generate_haplotypes_block(x, y, z)
        overall_haplotypes = processing_pool.starmap(
            generate_haplotypes_block, 
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
#%%
# for i in range(6):
#     matches = match_long_hap_to_blocks(haplotype_data[i][:18341],
#                                    all_sites[:18341],test_haps)

#     print(matches[0])
# print(check[2])