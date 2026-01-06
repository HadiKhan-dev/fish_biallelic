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

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    warnings.warn("Numba not found. Model selection will be extremely slow.", ImportWarning)
    # Dummy decorators
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator
    prange = range

class BlockResult:
    """
    Container for the reconstructed haplotypes of a single genomic block.
    """
    def __init__(self, positions, haplotypes, reads_count_matrix=None, keep_flags=None, probs_array=None):
        self.positions = positions
        self.haplotypes = haplotypes # Dictionary {id: numpy_array}
        self.reads_count_matrix = reads_count_matrix # Optional: source reads (Samples x Sites x 2)
        self.keep_flags = keep_flags
        self.probs_array = probs_array # New Optional: genotype probabilities (Samples x Sites x 3)
        
    def __len__(self):
        return len(self.haplotypes)

    def __repr__(self):
        active_sites = np.sum(self.keep_flags) if self.keep_flags is not None else len(self.positions)
        has_probs = "with probs" if self.probs_array is not None else "no probs"
        return f"<BlockResult: {len(self.haplotypes)} haplotypes at {len(self.positions)} sites ({active_sites} active), {has_probs}>"


class BlockResults:
    """
    A container class holding a list of BlockResult objects, representing
    the reconstruction results for an entire genomic region.
    """
    def __init__(self, block_result_list):
        self.blocks = block_result_list

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, index):
        """Allows accessing blocks by index: block_results[i]"""
        return self.blocks[index]

    def __iter__(self):
        """Allows iterating: for block in block_results: ..."""
        return iter(self.blocks)

    def __repr__(self):
        return f"<BlockResults: containing {len(self.blocks)} processed blocks>"

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
                                      alpha=alpha,
                                      allow_single_cluster=allow_single_cluster)
    
    #Fit data to clustering
    base_clustering.fit(dist_matrix)
    
    initial_labels = np.array(base_clustering.labels_)
    initial_probabilities = np.array(base_clustering.probabilities_)
    
    all_clusters = set(initial_labels)
    all_clusters.discard(-1)

    return [initial_labels,initial_probabilities,base_clustering]

def get_representatives_probs(site_priors,
                        probs_array,
                        cluster_labels,
                        cluster_probs=np.array([]),
                        prob_cutoff=0.8,
                        site_max_singleton_sureness=0.98,
                        read_error_prob=0.02):
    """
    Get representatives for each of the clusters we have.
    """
    
    # Calculate singleton priors (Haploid) from Site Priors (Diploid)
    # P(0) ~ sqrt(P(00)), P(1) ~ sqrt(P(11))
    singleton_priors = np.array([np.sqrt(site_priors[:,0]), np.sqrt(site_priors[:,2])]).T
    # Normalize rows
    row_sums = singleton_priors.sum(axis=1)[:, np.newaxis]
    singleton_priors = np.divide(singleton_priors, row_sums, out=np.full_like(singleton_priors, 0.5), where=row_sums!=0)

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
            # Safety for log(0)
            log_priors = np.log(priors + 1e-12) 
            
            site_data = cluster_data[:,i,:].copy()
            
            site_data[site_data > site_max_singleton_sureness] = site_max_singleton_sureness
            site_data[site_data < 1-site_max_singleton_sureness] = 1-site_max_singleton_sureness
            
            zero_logli = 0
            one_logli = 0
            
            for info in site_data:
                # Calc prob of seeing this read given Hyp(0) or Hyp(1)
                
                # prob_obs_given_0 = (1-e)*P(R0) + e*P(R1)
                p_obs_0 = (1-read_error_prob)*info[0] + read_error_prob*info[1]
                
                # prob_obs_given_1 = e*P(R0) + (1-e)*P(R1)
                p_obs_1 = read_error_prob*info[0] + (1-read_error_prob)*info[1]
                
                # Clamp to avoid log(0)
                if p_obs_0 < 1e-12: p_obs_0 = 1e-12
                if p_obs_1 < 1e-12: p_obs_1 = 1e-12
                
                zero_logli += math.log(p_obs_0)
                one_logli += math.log(p_obs_1)
            
            logli_array = np.array([zero_logli,one_logli])
            
            nonorm_log_post = log_priors+logli_array
            
            # Avoid nan in mean if one branch is -inf
            max_val = np.max(nonorm_log_post)
            if np.isinf(max_val): 
                # Both are -inf?
                posterior = priors # Fallback to priors
            else:
                nonorm_log_post -= max_val # Safer than mean for overflow prevention
                nonorm_post = np.exp(nonorm_log_post)
                posterior = nonorm_post/np.sum(nonorm_post)
            
            cluster_representative.append(posterior)
        
        #Add the representative for the cluster to the dictionary
        cluster_representatives[cluster] = np.array(cluster_representative)
        

    #Return the results
    return cluster_representatives

def consolidate_similar_candidates(candidates, diff_threshold_percent=1.0):
    """
    Greedily merges candidates that are nearly identical.
    
    Args:
        candidates: dict or list of haplotype arrays.
        diff_threshold_percent: Percentage difference (0-100) below which to merge.
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
            diff = np.mean(hap != existing) * 100.0
            
            if diff < diff_threshold_percent:
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
    
    Args:
        recomb_penalty: Cost to switch haplotypes in the Viterbi model.
        error_tolerance_sites: Number of sites of mismatch allowed before deletion.
                               (Each mismatch penalizes the score by 0.5)
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
            usage = usage_map.get(target_key, 0)
            adjusted_score = best_score - (usage * 1e-9)
            
            candidate_scores.append((target_key, adjusted_score, best_score))
            
        # Sort candidates by Adjusted Score Descending
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        
        worst_key, _, raw_score = candidate_scores[0]
        
        # 4. Decision Logic
        allowed_switches = 2
        threshold = -((recomb_penalty * allowed_switches) + 
                      (mismatch_penalty * error_tolerance_sites))
        
        # If score is BETTER (less negative) than threshold, it means 
        # we found a good way to mimic this haplotype using others. Delete it.
        if raw_score > threshold:
            kept_keys.remove(worst_key)
        else:
            break
            
    return {k: hap_dict[k] for k in kept_keys}

def select_optimal_haplotype_set_viterbi(candidate_haps, probs_array, 
                                         recomb_penalty=10.0,
                                         penalty_strength=1.0,
                                         read_error_prob=0.02,
                                         max_sites_for_selection=2000):
    """
    Selects the smallest set of haplotypes that explains the data best.
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
    if total_sites > max_sites_for_selection:
        stride = math.ceil(total_sites / max_sites_for_selection)
        # Slice the data
        probs_active = probs_array[:, ::stride, :]
        H_active = H[:, ::stride, :]
    else:
        stride = 1
        probs_active = probs_array
        H_active = H

    num_active_sites = probs_active.shape[1]

    # --- 3. PRE-CALCULATE ALL PAIR LIKELIHOODS (MEMORY SAFE) ---
    idx_i, idx_j = np.triu_indices(num_candidates)
    num_pairs = len(idx_i)
    
    ll_tensor = np.empty((num_samples, num_pairs, num_active_sites), dtype=np.float32)
    
    W = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=np.float32)
    
    batch_size = 50
    
    for start_p in range(0, num_pairs, batch_size):
        end_p = min(start_p + batch_size, num_pairs)
        
        batch_idx_i = idx_i[start_p:end_p]
        batch_idx_j = idx_j[start_p:end_p]
        
        h0 = H_active[batch_idx_i, :, 0]
        h1 = H_active[batch_idx_i, :, 1]
        h2_0 = H_active[batch_idx_j, :, 0]
        h2_1 = H_active[batch_idx_j, :, 1]
        
        g00 = h0 * h2_0
        g11 = h1 * h2_1
        g01 = (h0 * h2_1) + (h1 * h2_0)
        
        batch_pairs = np.stack([g00, g01, g11], axis=-1)
        
        weighted_pairs = batch_pairs @ W
        
        batch_dist = np.sum(
            probs_active[:, np.newaxis, :, :] * weighted_pairs[np.newaxis, :, :, :],
            axis=3
        )
        
        ll_tensor[:, start_p:end_p, :] = (-batch_dist * stride)

    ll_tensor = np.maximum(ll_tensor, -2.0 * stride)
    ll_tensor = np.ascontiguousarray(ll_tensor, dtype=np.float64)

    # --- 4. SELECTION LOOP ---
    
    selected_indices = []
    current_best_bic = float('inf')
    
    min_complexity = recomb_penalty * 1.5
    calculated_complexity = math.log(num_samples) * total_sites * penalty_strength * 0.01
    complexity_cost = max(calculated_complexity, min_complexity)
    
    while len(selected_indices) < num_candidates:
        
        best_new_index = -1
        best_new_bic = float('inf')
        
        remaining = [x for x in range(num_candidates) if x not in selected_indices]
        
        for cand_idx in remaining:
            trial_set = selected_indices + [cand_idx]
            
            subset_mask = np.zeros(num_candidates, dtype=bool)
            subset_mask[trial_set] = True
            
            valid_pairs_mask = subset_mask[idx_i] & subset_mask[idx_j]
            
            if not np.any(valid_pairs_mask):
                continue
            
            active_ll_tensor = ll_tensor[:, valid_pairs_mask, :]
            
            best_scores = viterbi_score_selection(active_ll_tensor, float(recomb_penalty))
            
            total_log_likelihood = np.sum(best_scores)
            
            k = len(trial_set)
            bic = (k * complexity_cost) - (2 * total_log_likelihood)
            
            if bic < best_new_bic:
                best_new_bic = bic
                best_new_index = cand_idx
        
        if best_new_bic < current_best_bic:
            selected_indices.append(best_new_index)
            current_best_bic = best_new_bic
        else:
            break
            
    return [hap_keys[i] for i in selected_indices]

def add_distinct_haplotypes(initial_haps,
                       new_candidate_haps,
                       keep_flags=None,
                       uniqueness_threshold_percent=2.0,
                       max_hap_add=1000):
    """
    Helper to add distinct haplotypes.
    This uses simple Hamming distance to ensure diversity.
    Do NOT replace with smart_vectorised as this is used for initialization.
    """
    
    for x in initial_haps.keys():
        haplotype_length = len(initial_haps[x])
        break
    
    if keep_flags is None:
        keep_flags = np.array([True for _ in range(haplotype_length)])
    
    if keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags,dtype=bool)
    
    keep_length = np.count_nonzero(keep_flags)
    
    new_haps_mapping = {-1:-1}

    cutoff = math.ceil(uniqueness_threshold_percent*keep_length/100.0)
    
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
                                   loss_reduction_cutoff_ratio=0.98,
                                   use_multiprocessing=False):
    """
    Fast, vectorized Greedy Forward Selection of haplotypes.
    Adds haplotypes that significantly reduce the error when explaining samples.
    """
    
    # --- 1. SETUP AND PRE-PROCESSING ---
    cur_haps_dict = initial_haps.copy()
    cand_haps_dict = new_candidate_haps.copy()
    
    if not cand_haps_dict:
        return cur_haps_dict

    cur_keys = list(cur_haps_dict.keys())
    cand_keys = list(cand_haps_dict.keys())
    
    if cur_keys:
        next_key_idx = max(cur_keys) + 1
    else:
        next_key_idx = 0

    num_samples, num_sites, _ = probs_array.shape
    if keep_flags is None:
        keep_flags = np.ones(num_sites, dtype=bool)
    elif keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags, dtype=bool)
        
    probs_masked = probs_array[:, keep_flags, :]
    masked_len = probs_masked.shape[1]
    
    diploids_flat = probs_masked.reshape(num_samples, -1)
    
    if cur_keys:
        cur_haps_mat = np.array([cur_haps_dict[k][keep_flags] for k in cur_keys])
    else:
        cur_haps_mat = np.empty((0, masked_len, 2))
        
    cand_haps_mat = np.array([cand_haps_dict[k][keep_flags] for k in cand_keys])
    
    dist_weights = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])

    # --- 2. HELPER: BATCH SCORE CALCULATION ---
    def compute_pair_scores(hap_matrix_A, hap_matrix_B):
        nA = len(hap_matrix_A)
        nB = len(hap_matrix_B)
        if nA == 0 or nB == 0: return np.zeros((num_samples, 0, 0))
        
        p0_A, p1_A = hap_matrix_A[:, :, 0], hap_matrix_A[:, :, 1]
        p0_B, p1_B = hap_matrix_B[:, :, 0], hap_matrix_B[:, :, 1]
        
        comb_00 = p0_A[:, None, :] * p0_B[None, :, :]
        comb_11 = p1_A[:, None, :] * p1_B[None, :, :]
        comb_01 = (p0_A[:, None, :] * p1_B[None, :, :]) + (p1_A[:, None, :] * p0_B[None, :, :])
        
        comb_stacked = np.stack([comb_00, comb_01, comb_11], axis=-1)
        comb_list = comb_stacked.reshape(nA * nB, masked_len, 3)
        
        comb_weighted = comb_list @ dist_weights 
        comb_flat = comb_weighted.reshape(nA * nB, masked_len * 3)
        
        scores_flat = diploids_flat @ comb_flat.T
        
        scores = scores_flat.reshape(num_samples, nA, nB)
        scores *= (100.0 / num_sites) 
        return scores

    # --- 3. INITIALIZE BASELINE SCORES ---
    if len(cur_keys) > 0:
        base_dists = compute_pair_scores(cur_haps_mat, cur_haps_mat) 
        base_dists_flat = base_dists.reshape(num_samples, -1)
        current_best_errors = np.min(base_dists_flat, axis=1)
    else:
        current_best_errors = np.full(num_samples, np.inf)

    current_mean_error = np.mean(current_best_errors) if len(cur_keys) > 0 else np.inf

    # --- 4. INITIALIZE CANDIDATE POTENTIALS ---
    num_cand = len(cand_keys)
    
    c_p0, c_p1 = cand_haps_mat[:, :, 0], cand_haps_mat[:, :, 1]
    homo_00 = c_p0 * c_p0
    homo_11 = c_p1 * c_p1
    homo_01 = 2 * c_p0 * c_p1
    homo_comb = np.stack([homo_00, homo_01, homo_11], axis=-1) @ dist_weights
    homo_flat = homo_comb.reshape(num_cand, -1)
    homo_scores = (diploids_flat @ homo_flat.T) * (100.0 / num_sites) 
    
    if len(cur_keys) > 0:
        hetero_scores_3d = compute_pair_scores(cand_haps_mat, cur_haps_mat)
        hetero_best = np.min(hetero_scores_3d, axis=2)
    else:
        hetero_best = np.full((num_samples, num_cand), np.inf)

    cand_best_potentials = np.minimum(homo_scores, hetero_best)
    
    cand_active = np.ones(num_cand, dtype=bool)
    
    # --- 5. GREEDY SELECTION LOOP ---
    
    while True:
        if not np.any(cand_active):
            break
            
        active_indices = np.where(cand_active)[0]
        
        potential_errors = np.minimum(
            current_best_errors[:, np.newaxis], 
            cand_best_potentials[:, active_indices]
        )
        
        potential_means = np.mean(potential_errors, axis=0)
        
        best_local_idx = np.argmin(potential_means)
        best_global_idx = active_indices[best_local_idx]
        best_new_mean = potential_means[best_local_idx]
        
        if current_mean_error == 0 or (best_new_mean / current_mean_error) >= loss_reduction_cutoff_ratio:
            break
            
        best_cand_key = cand_keys[best_global_idx]
        cur_haps_dict[next_key_idx] = cand_haps_dict[best_cand_key]
        
        new_hap_vec = cand_haps_mat[best_global_idx][np.newaxis, :, :]
        
        current_best_errors = potential_errors[:, best_local_idx]
        current_mean_error = best_new_mean
        
        cand_active[best_global_idx] = False
        next_key_idx += 1
        
        if not np.any(cand_active):
            break
            
        remaining_indices = np.where(cand_active)[0]
        remaining_cands_mat = cand_haps_mat[remaining_indices]
        
        new_pairing_scores = compute_pair_scores(remaining_cands_mat, new_hap_vec)
        new_pairing_scores = new_pairing_scores[:, :, 0]
        
        cand_best_potentials[:, remaining_indices] = np.minimum(
            cand_best_potentials[:, remaining_indices],
            new_pairing_scores
        )

    return cur_haps_dict

def get_initial_haps(site_priors,
                     probs_array,
                     keep_flags=None,
                     het_cutoff_start_percentage=10,
                     het_excess_add=2,
                     het_max_cutoff_percentage=20,
                     deeper_analysis=False,
                     uniqueness_threshold_percent=2.0, # Renamed for clarity
                     read_error_prob = 0.02,
                     verbose=False):
    """
    Get our initial haplotypes by finding high homozygosity samples
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
        
        homs_where = np.where(het_vals <= cur_het_cutoff)[0]
    
        homs_array = probs_array[homs_where]
        
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
            
            # Using assertion to trigger the except block if no clusters found
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
    
    (representatives,label_mappings) = add_distinct_haplotypes(
              {},representatives,keep_flags=keep_flags,
              uniqueness_threshold_percent=uniqueness_threshold_percent)
        
    return representatives

def generate_further_haps(site_priors,
                          probs_array,
                          initial_haps,
                          keep_flags=None,
                          wrongness_cutoff=10,
                          uniqueness_threshold_percent=2.0,
                          max_hap_add=1000,
                          verbose=False):
    """
    Fully vectorized version of generate_further_haps.
    """
    
    # --- 1. PREPARE DATA ---
    num_samples, num_sites, _ = probs_array.shape
    
    if keep_flags is None:
        keep_flags = np.ones(num_sites, dtype=bool)
    elif keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags, dtype=bool)
        
    probs_masked = probs_array[:, keep_flags, :]
    masked_sites_len = probs_masked.shape[1]
    
    initial_list = np.array(list(initial_haps.values()))
    initial_list_masked = initial_list[:, keep_flags, :]
    
    # --- 2. BROADCASTING SETUP ---
    D0 = probs_masked[:, :, 0][:, np.newaxis, :] # P(Dip=00)
    D1 = probs_masked[:, :, 1][:, np.newaxis, :] # P(Dip=01)
    D2 = probs_masked[:, :, 2][:, np.newaxis, :] # P(Dip=11)
    
    H0 = initial_list_masked[:, :, 0][np.newaxis, :, :] # P(Hap=0)
    H1 = initial_list_masked[:, :, 1][np.newaxis, :, :] # P(Hap=1)
    
    # --- 3. CALCULATE WRONGNESS (Vectorized) ---
    wrong_per_site = (H1 * D0) + (H0 * D2)
    wrongness_scores = (np.sum(wrong_per_site, axis=2) * 100.0) / masked_sites_len
    
    valid_mask = wrongness_scores <= wrongness_cutoff
    
    if not np.any(valid_mask):
        return initial_haps

    # --- 4. CALCULATE NEW CANDIDATES (Only for valid pairs) ---
    idx_s, idx_h = np.where(valid_mask)
    
    D0_sub = probs_masked[idx_s, :, 0]
    D1_sub = probs_masked[idx_s, :, 1]
    D2_sub = probs_masked[idx_s, :, 2]
    
    H0_sub = initial_list_masked[idx_h, :, 0]
    H1_sub = initial_list_masked[idx_h, :, 1]
    
    new_haps_0 = D0_sub + (H1_sub * D1_sub)
    new_haps_1 = D2_sub + (H0_sub * D1_sub)
    
    candidate_haps_masked = np.stack([new_haps_0, new_haps_1], axis=-1)
    
    # --- 5. UNIQUENESS CHECK ---
    cand_p1 = candidate_haps_masked[:, :, 1]
    init_p1 = initial_list_masked[:, :, 1] 
    
    dists = cdist(cand_p1, init_p1, metric='cityblock')
    perc_dists = (dists * 100.0) / masked_sites_len
    
    min_dist_to_existing = np.min(perc_dists, axis=1)
    
    is_unique = min_dist_to_existing >= uniqueness_threshold_percent
    
    unique_candidates = candidate_haps_masked[is_unique]
    
    if len(unique_candidates) == 0:
        return initial_haps

    if len(unique_candidates) > max_hap_add:
        unique_candidates = unique_candidates[:max_hap_add]

    # --- 6. CLUSTERING & CLEANUP ---
    unique_p1 = unique_candidates[:, :, 1]
    
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

    # --- FILLING MASKED SITES (Fix for NaN issue) ---
    final_candidates_full = np.zeros((len(unique_candidates), num_sites, 2))
    final_candidates_full[:, keep_flags, :] = unique_candidates
    
    # Fill masked regions with site priors to avoid log(0) in get_representatives
    # Calculate haploid priors: P(Ref) ~ sqrt(P(RefRef))
    haploid_priors = np.zeros((num_sites, 2))
    haploid_priors[:, 0] = np.sqrt(site_priors[:, 0]) # P(0)
    haploid_priors[:, 1] = np.sqrt(site_priors[:, 2]) # P(1)
    # Normalize rows
    row_sums = haploid_priors.sum(axis=1)[:, np.newaxis]
    haploid_priors = np.divide(haploid_priors, row_sums, out=np.full_like(haploid_priors, 0.5), where=row_sums!=0)
    
    inverse_mask = ~keep_flags
    if np.any(inverse_mask):
        # Broadcast the priors to all candidates at masked sites
        final_candidates_full[:, inverse_mask, :] = haploid_priors[inverse_mask, :]
        
    representatives = get_representatives_probs(
        site_priors, final_candidates_full, cluster_labels)
    
    final_haps = add_distinct_haplotypes_smart(initial_haps,
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
                              known_haplotypes=None,
                              uniqueness_threshold_percent=2.0, # PARAM
                              diff_threshold_percent=1.0,       # PARAM
                              wrongness_threshold=10.0,         # PARAM
                              chimera_switch_cost=10.0,         # PARAM
                              chimera_mismatch_tol=1):          # PARAM (Default 1 site)
    """
    Given the read count array of our sample data for a single block
    generates the haplotypes that make up the samples present in our data.
    
    Returns a BlockResult object.
    """
    
    # --- 1. SETUP & INITIALIZATION ---
    if keep_flags is None:
        keep_flags = np.array([1 for _ in range(reads_array.shape[1])])
    
    if keep_flags.dtype != int:
        keep_flags = np.array(keep_flags, dtype=int)
    
    # Calculate probabilities from reads - STORE THIS to return later
    (site_priors, probs_array) = analysis_utils.reads_to_probabilities(reads_array)
    
    # Get seed haplotypes from homozygotes
    initial_haps = get_initial_haps(site_priors, probs_array, keep_flags=keep_flags,
                                    uniqueness_threshold_percent=uniqueness_threshold_percent)
    
    if len(positions) == 0:
        return BlockResult(np.array([]), initial_haps, np.array([]), keep_flags=keep_flags, probs_array=probs_array)

    # --- 2. INJECT KNOWN HAPLOTYPES ---
    if known_haplotypes is not None:
        if isinstance(known_haplotypes, dict):
            known_list = list(known_haplotypes.values())
        elif isinstance(known_haplotypes, list):
            known_list = known_haplotypes
        else:
            known_list = []
            
        if len(known_list) > 0:
            combined_list = list(initial_haps.values()) + known_list
            initial_haps = consolidate_similar_candidates(combined_list, diff_threshold_percent=diff_threshold_percent)

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
    
    # Use variable threshold starting at passed arg
    current_unique_thresh = uniqueness_threshold_percent
    
    # Use the passed wrongness threshold as the starting point
    current_wrongness = wrongness_threshold
    
    # --- 4. ITERATIVE DISCOVERY LOOP ---
    while not all_found:
        
        # A. Generate new candidates based on residuals
        cur_haps = generate_further_haps(
            site_priors, 
            probs_array,
            cur_haps,
            keep_flags=keep_flags,
            uniqueness_threshold_percent=current_unique_thresh,
            wrongness_cutoff=current_wrongness
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
                current_unique_thresh = max(1.0, current_unique_thresh - 1.0)
                # Relax wrongness if stuck
                current_wrongness += 2.0 
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
        merged_haps = consolidate_similar_candidates(candidates_to_filter, diff_threshold_percent=diff_threshold_percent)
        
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
        # DISABLED per biological reasoning (Inbreeding indistinguishable from Chimerism in small blocks)
        # final_haps = prune_chimeras(
        #     used_haps, 
        #     probs_array, 
        #     recomb_penalty=chimera_switch_cost, 
        #     error_tolerance_sites=chimera_mismatch_tol
        # )
        
        final_haps = {i: v for i, v in enumerate(used_haps.values())}
        
    else:
        final_haps = candidates_to_filter

    return BlockResult(positions, final_haps, reads_array, keep_flags=keep_flags, probs_array=probs_array)

def find_missing_haplotypes_iterative(positions, reads_array, current_haps, 
                                      keep_flags=None,
                                      error_threshold_percent=2.0, 
                                      min_bad_samples=5):
    """
    Analyzes the fit of the current haplotypes to the data.
    Identifies samples with high error rates, isolates them, and runs
    a targeted haplotype generation on that subset.
    """
    
    if len(current_haps) == 0:
        return {}

    # 1. Setup Data
    if keep_flags is None:
        keep_flags = np.array([1 for _ in range(reads_array.shape[1])])
    
    (_, probs_array) = analysis_utils.reads_to_probabilities(reads_array)
    
    # 2. Check Fit using K-Limited
    match_results = hap_statistics.match_best_k_limited(
        current_haps, 
        probs_array, 
        keep_flags=keep_flags,
        max_recombinations=2
    )
    
    error_scores = match_results[2]
    
    # 3. Identify Outliers
    bad_fit_indices = np.where(error_scores > error_threshold_percent)[0]
    num_bad = len(bad_fit_indices)
    
    if num_bad < min_bad_samples:
        return {}
        
    # 4. Run Targeted Generation
    subset_reads = reads_array[bad_fit_indices]
    
    # We catch the BlockResult object here
    sub_block_result = generate_haplotypes_block(
        positions,
        subset_reads,
        keep_flags=keep_flags,
        penalty_strength=0.5, 
        min_num_haps=1,       
        max_intermediate_haps=100 
    )
    
    sub_result_haps = sub_block_result.haplotypes
    
    # 5. Filter Redundancy
    existing_matrix = np.array(list(current_haps.values()))
    newly_found_unique = {}
    new_idx = 0
    
    for k, sub_hap in sub_result_haps.items():
        diffs = np.mean(existing_matrix != sub_hap, axis=1) * 100.0
        min_diff = np.min(diffs)
        
        if min_diff > 2.0: # Compare using Percentage
            newly_found_unique[new_idx] = sub_hap
            new_idx += 1
            
    return newly_found_unique

def generate_haplotypes_block_robust(positions, reads_array, keep_flags=None, 
                                     max_robust_passes=3, 
                                     **kwargs):
    """
    Wrapper that runs the standard generation, checks for missing data (residuals),
    and re-runs iteratively until all distinct founders are found.
    """
    
    # Track known haplotypes across iterations
    current_known_haps = kwargs.get('known_haplotypes', [])
    if isinstance(current_known_haps, dict):
        current_known_haps = list(current_known_haps.values())
    
    # Variable to hold our final BlockResult object
    final_result = None
    
    for pass_num in range(1, max_robust_passes + 1):
        
        # 1. Run Generation (passing the accumulated known list)
        run_kwargs = kwargs.copy()
        run_kwargs['known_haplotypes'] = current_known_haps
        
        # Capture the full result object (now contains probs_array)
        final_result = generate_haplotypes_block(
            positions, reads_array, keep_flags=keep_flags, **run_kwargs
        )
        
        # 2. Residual Analysis (uses haps from the recent pass)
        missing_haps_dict = find_missing_haplotypes_iterative(
            positions, 
            reads_array, 
            final_result.haplotypes, 
            keep_flags=keep_flags,
            error_threshold_percent=2.0,
            min_bad_samples=5 
        )
        
        # 3. Decision
        if len(missing_haps_dict) == 0:
            break
        else:
            
            new_haps_list = list(missing_haps_dict.values())
            
            combined_pool = current_known_haps + new_haps_list
            consolidated_dict = consolidate_similar_candidates(combined_pool, diff_threshold_percent=0.01)
            current_known_haps = list(consolidated_dict.values())
            
    return final_result

def generate_all_block_haplotypes(genomic_data, # Accepts GenomicData object
                            penalty_strength=1.0, 
                            max_intermediate_haps=100,
                            uniqueness_threshold_percent=2.0, # NEW PARAM
                            diff_threshold_percent=1.0,       # NEW PARAM
                            wrongness_threshold=10.0,         # NEW PARAM
                            chimera_switch_cost=10.0,         # NEW PARAM
                            chimera_mismatch_tol=1,           # NEW PARAM
                            num_processes=16):
    """
    Generate a list of block haplotypes using multiprocessing.
    
    Args:
        genomic_data: A GenomicData object containing blocks to process.
        penalty_strength: Controls strictness of final Viterbi selection (1.0 = standard).
        max_intermediate_haps: Trigger for intermediate pruning during the loop.
    
    Assumes OMP_NUM_THREADS=1 is set in the main script.
    """
    
    # Create a partial function with the fixed configuration arguments
    # When starmap unpacks the 'tasks' (GenomicData yield), it will append these fixed args as keyword args
    worker_func = partial(
        generate_haplotypes_block_robust, 
        penalty_strength=penalty_strength, 
        max_intermediate_haps=max_intermediate_haps,
        uniqueness_threshold_percent=uniqueness_threshold_percent,
        diff_threshold_percent=diff_threshold_percent,
        wrongness_threshold=wrongness_threshold,
        chimera_switch_cost=chimera_switch_cost,
        chimera_mismatch_tol=chimera_mismatch_tol
    )

    with Pool(processes=num_processes) as processing_pool:
        
        # GenomicData is iterable yielding (pos, reads, flags)
        # starmap will call: worker_func(pos, reads, flags)
        # worker_func (partial) adds: penalty_strength=..., max_intermediate_haps=...
        overall_haplotypes = processing_pool.starmap(
            worker_func, 
            genomic_data
        )

    return BlockResults(overall_haplotypes)