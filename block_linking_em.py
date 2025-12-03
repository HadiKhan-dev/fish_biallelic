import numpy as np
import math
import time
from multiprocess import Pool
import itertools
import copy
import seaborn as sns
from scipy.special import logsumexp

import analysis_utils
import block_haplotypes
import simulate_sequences
import hap_statistics

#%%
def find_runs(data, min_length):
    """
    Function which takes as input a list of bools and a minimum length,
    then returns all runs of True of length at least min_length
    """
    runs = []
    run_start = -1  # -1 indicates we are not in a run of Trues

    # Enumerate to get both index and value
    for i, value in enumerate(data):
        if value:  # Current value is True
            if run_start == -1:
                run_start = i  # Start of a new run
        else:  # Current value is False
            if run_start != -1:  # We were in a run, and it just ended
                run_length = i - run_start
                if run_length >= min_length:
                    runs.append((run_start, i))
                run_start = -1 # Reset for the next run

    # After the loop, check if a run was ongoing to the end of the list
    if run_start != -1:
        run_length = len(data) - run_start
        if run_length >= min_length:
            runs.append((run_start, len(data)))
            
    return runs

def initial_transition_probabilities(hap_data,
                                     space_gap=1,
                                     found_haps=[],
                                     ):
    """
    Creates a dict of initial equal transition probabilities
    for a list where each element of the list contains info about
    block haps for that block.
    
    space_gap is the number of blocks we jump over at each step
    for calculating the transition probabilities. By default this
    is equal to 1.

    """
    transition_dict_forward = {}
    transition_dict_reverse = {}
    
    for i in range(0,len(hap_data)-space_gap):
        transition_dict_forward[i] = {}
        
        these_haps = hap_data[i][3]
        next_haps = hap_data[i+space_gap][3]
        
        for first_idx in these_haps.keys():
            first_hap_name = (i,first_idx)
            
            for second_idx in next_haps.keys():
                second_hap_name = (i+space_gap,second_idx)
                
                present = False
                
                for fhap in found_haps:
                    if fhap[i] == first_idx and fhap[i+space_gap] == second_idx:
                        present = True
                
                if not present:
                    transition_dict_forward[i][(first_hap_name,second_hap_name)] = 1
                else:
                    transition_dict_forward[i][(first_hap_name,second_hap_name)] = 1
                
    for i in range(len(hap_data)-1,space_gap-1,-1):
        transition_dict_reverse[i] = {}
        
        these_haps = hap_data[i][3]
        next_haps = hap_data[i-space_gap][3]
        
        for first_idx in these_haps.keys():
            first_hap_name = (i,first_idx)
            
            for second_idx in next_haps.keys():
                second_hap_name = (i-space_gap,second_idx)
                
                present = False
                
                for fhap in found_haps:
                    if fhap[i] == first_idx and fhap[i-space_gap] == second_idx:
                        present = True
                
                if not present:
                    transition_dict_reverse[i][(first_hap_name,second_hap_name)] = 1
                else:
                    transition_dict_reverse[i][(first_hap_name,second_hap_name)] = 1
    
    #At this point we have the unscaled transition probabilities, we now scale them
    scaled_dict_forward = {}
    scaled_dict_reverse = {}
    
    for idx in transition_dict_forward.keys():
        scaled_dict_forward[idx] = {}
        
        start_dict = {}
        for s in transition_dict_forward[idx].keys():
            if s[0] not in start_dict.keys():
                start_dict[s[0]] = 0
            start_dict[s[0]] += transition_dict_forward[idx][s]
        
        for s in transition_dict_forward[idx].keys():
            scaled_dict_forward[idx][s] = transition_dict_forward[idx][s]/start_dict[s[0]]
        
    for idx in transition_dict_reverse.keys():
        scaled_dict_reverse[idx] = {}
        
        start_dict = {}
        for s in transition_dict_reverse[idx].keys():
            if s[0] not in start_dict.keys():
                start_dict[s[0]] = 0
            start_dict[s[0]] += transition_dict_reverse[idx][s]
        
        for s in transition_dict_reverse[idx].keys():
            scaled_dict_reverse[idx][s] = transition_dict_reverse[idx][s]/start_dict[s[0]]
        
    return [scaled_dict_forward,scaled_dict_reverse]

def get_block_likelihoods(sample_data,
                          haps_data,
                          log_likelihood_base=math.e**2,
                          min_per_site_log_likelihood=-10,
                          normalize=True,                          
                          ):
    """
    Get the log-likelihoods for each combination of haps in how closely
    they are matching the sample
    
    sample_ploidy is a list of length the number of sites in the block
    which indicates whether this particular block is made of two, one or zero haplotypes
    
    haps_data must be the full list of haplotype information for a block, including
    the positions,keep_flags,read_counts and haplotype info elements
    
    For each site we compute the distance of the sample data probabalistic genotype
    from the probabalistic genotype given by combining each pair of haps in haps_data.
    We then get the log_likelihood for that site as
    max(-dist*log(log_likelihood_base),min_per_site_log_likelihood)
    
    If normalize=True then ensures the sum of the probabilities for 
    the sample over all possible pairs adds up to 1
    """
    
    main_sample_data = sample_data[0]
    sample_ploidy = sample_data[1]
    
    assert len(haps_data[0]) == len(main_sample_data), f"Number of sites in sample {len(main_sample_data)} don't match number of sites in haps {len(haps_data[0])}"
    
    
    block_ploidy = sample_ploidy[0]
    
    bool_keepflags = haps_data[1].astype(bool)
    
    ll_dict = {}
    
    ll_list = []
    
    if block_ploidy == 2:
        sample_keep = main_sample_data[bool_keepflags,:]
        
        for i in haps_data[3].keys():
            for j in haps_data[3].keys():
                if j < i:
                    continue
                
                combined_haps = analysis_utils.combine_haploids(haps_data[3][i],haps_data[3][j])
                combined_keep = combined_haps[bool_keepflags,:]
                
                dist = analysis_utils.calc_distance_by_site(sample_keep,combined_keep)
                
                bdist = -(dist**2)*math.log(log_likelihood_base)
                combined_logs = np.concatenate([np.array(bdist.reshape(1,-1)),min_per_site_log_likelihood*np.ones((1,len(dist)))])
                
                combined_dist = np.max(combined_logs,axis=0)
                
                total_ll = np.sum(combined_dist)
                
                ll_dict[(i,j)] = total_ll
                ll_list.append(total_ll)
                
    elif block_ploidy == 1:
        sample_keep = main_sample_data[bool_keepflags,:2]
        
        for i in haps_data[3].keys():
            use_hap = haps_data[3][i]
            use_keep = use_hap[bool_keepflags,:]
            
            dist = analysis_utils.calc_distance_by_site(sample_keep,use_keep,calc_type="haploid")
            
            bdist = -(dist**2)*math.log(log_likelihood_base)
            combined_logs = np.concatenate([np.array(bdist.reshape(1,-1)),min_per_site_log_likelihood*np.ones((1,len(dist)))])
            
            combined_dist = np.max(combined_logs,axis=0)
            
            total_ll = np.sum(combined_dist)
            
            ll_dict[i] = total_ll
            ll_list.append(total_ll)
            
    else: #This is the block_ploidy = 0 case, here the log likelihood is 0 as we don't have any haps left
        ll_dict[0] = 0
        ll_list.append(0)
            
    if normalize:
        combined_ll = logsumexp(ll_list)
        
        for k in ll_dict.keys():
            ll_dict[k] = ll_dict[k]-combined_ll
            
    return ll_dict

def get_all_block_likelihoods(block_samples_data,block_haps):
    """
    Function which calculates the block likelihoods for each 
    sample in block_samples_data
    
    block_samples_data must contain the ploidy for each sample
    """
    sample_likelihoods = []
    sample_ploidies = []
    
    for i in range(len(block_samples_data[0])):
        sample_likelihoods.append(
            get_block_likelihoods([block_samples_data[0][i],
                                  block_samples_data[1][i]],
                                  block_haps))
        sample_ploidies.append(block_samples_data[1][i])
    
    
    return (sample_likelihoods,sample_ploidies)

def get_all_block_likelihoods_vectorised(block_samples_data, block_haps, 
    log_likelihood_base=math.e,       
    min_per_site_log_likelihood=-0.5, 
    uniform_error_floor_per_site=-0.6, 
    min_absolute_block_log_likelihood=-200.0):
    
    # 1. UNPACK DATA
    raw_samples = block_samples_data[0]
    ploidies = np.array(block_samples_data[1])
    
    if len(raw_samples) == 0:
        return ([], ploidies)

    samples_matrix = np.stack(raw_samples)
    num_samples, num_sites, _ = samples_matrix.shape

    hap_dict = block_haps[3]
    keep_flags = block_haps[1].astype(bool)
    hap_keys = sorted(list(hap_dict.keys()))
    num_haps = len(hap_keys)
    
    if num_haps == 0:
        return ([{0: 0.0}] * num_samples, ploidies)

    # 2. ROBUST TENSOR CREATION
    hap_list = [hap_dict[k] for k in hap_keys]
    if len(hap_list) > 0 and (hap_list[0].size == 0):
        haps_tensor = np.zeros((num_haps, 0, 2))
    else:
        haps_tensor = np.array(hap_list)
    
    # 3. MASKING
    samples_masked = samples_matrix[:, keep_flags, :] 
    haps_masked = haps_tensor[:, keep_flags, :]       
    
    # --- FIX START ---
    # Check if we have any sites left to process
    num_active_sites = samples_masked.shape[1]
    
    if num_active_sites > 0:
        # 4. GENERATE DIPLOID COMBINATIONS
        h0 = haps_masked[:, :, 0]
        h1 = haps_masked[:, :, 1]
        
        c00 = h0[:, None, :] * h0[None, :, :]
        c11 = h1[:, None, :] * h1[None, :, :]
        c01 = (h0[:, None, :] * h1[None, :, :]) + (h1[:, None, :] * h0[None, :, :])
        
        combos_4d = np.stack([c00, c01, c11], axis=-1)
        
        # Flatten HxH -> (H^2, Sites, 3)
        # This works because Sites > 0, so NumPy can solve for -1
        combos_flat = combos_4d.reshape(-1, combos_4d.shape[2], 3)
        
        # 5. WEIGHTED DISTANCE CALCULATION
        dist_weights = np.array([[0, 1, 2], 
                                 [1, 0, 1], 
                                 [2, 1, 0]])
        
        combos_weighted = combos_flat @ dist_weights
        
        dist_per_site = np.sum(
            samples_masked[:, np.newaxis, :, :] * combos_weighted[np.newaxis, :, :, :], 
            axis=3
        )
        
        # 6. LOG LIKELIHOOD (TUNED)
        log_penalty = math.log(log_likelihood_base)
        ll_per_site = -(dist_per_site) * log_penalty
        ll_per_site = np.maximum(ll_per_site, min_per_site_log_likelihood)
        
        # Sum over sites
        total_ll_matrix = np.sum(ll_per_site, axis=2)
        
    else:
        # Handle 0 sites case
        # If there are no sites, the distance is 0, penalty is 0.
        # All pairs are equally likely (Uniform).
        # We explicitly create the zero matrix because reshape(-1) would crash.
        total_ll_matrix = np.zeros((num_samples, num_haps**2))
    # --- FIX END ---

    # 7. APPLY GLOBAL FLOORS
    if num_active_sites > 0:
        site_based_floor = num_active_sites * uniform_error_floor_per_site
        total_ll_matrix = np.maximum(total_ll_matrix, site_based_floor)
        
    total_ll_matrix = np.maximum(total_ll_matrix, min_absolute_block_log_likelihood)
        
    # 8. NORMALIZATION
    idx_i, idx_j = np.unravel_index(np.arange(num_haps**2), (num_haps, num_haps))
    valid_mask = idx_j >= idx_i
    unique_ll_matrix = total_ll_matrix[:, valid_mask]
    
    norm_factors = logsumexp(unique_ll_matrix, axis=1, keepdims=True)
    final_normalized_matrix = unique_ll_matrix - norm_factors
    
    # 9. FORMAT OUTPUT
    results = []
    keys = [(hap_keys[i], hap_keys[j]) for i, j in zip(idx_i[valid_mask], idx_j[valid_mask])]
    
    for s in range(num_samples):
        results.append(dict(zip(keys, final_normalized_matrix[s])))
            
    return (results, ploidies)

def multiprocess_all_block_likelihoods(full_samples_data,
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
            get_all_block_likelihoods_vectorised, 
            tasks
        )

    return full_blocks_likelihoods


# --- 2. FORWARD PASS (FIXED) ---
def get_full_probs_forward(sample_data, sample_sites, haps_data,
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

        # Calculate serially or use a pool if not nested
        with Pool(16) as pool:
            full_blocks_likelihoods = pool.starmap(
                get_all_block_likelihoods_vectorised, 
                tasks
            )

    transition_probs_dict = bidirectional_transition_probs[0]
    
    # OUTPUT: Dictionary of dictionaries (Unordered pairs)
    likelihood_numbers = {} 
    
    # CACHE: Stores the matrices needed for recursion (Ordered)
    shadow_cache = {} 
    
    ln_2 = math.log(2)

    for i in range(len(haps_data)):
        
        # 1. Get Current Observation Likelihoods
        (block_lik_dict, _) = full_blocks_likelihoods[i]
        
        # Handle list-wrapper if present
        if isinstance(block_lik_dict, list):
            block_lik_dict = block_lik_dict[0] 

        hap_keys = sorted(list(haps_data[i][3].keys()))
        n_haps = len(hap_keys)
        key_to_idx = {k: idx for idx, k in enumerate(hap_keys)}
        
        # Create Emission Matrix E
        E = np.full((n_haps, n_haps), -np.inf)
        for (u, v), val in block_lik_dict.items():
            u_idx, v_idx = key_to_idx[u], key_to_idx[v]
            E[u_idx, v_idx] = val
            E[v_idx, u_idx] = val
        
        # 2. Compute Forward Probabilities
        if i < space_gap:
            # Base Case: Just Emission, adjusted for Ordered space
            # Matrix Entry = DictEntry - log(2) for heterozygotes
            corrections = np.full((n_haps, n_haps), -ln_2)
            np.fill_diagonal(corrections, 0.0)
            current_matrix = E + corrections
            
        else:
            # Recursive Step
            earlier_block = i - space_gap
            
            # --- FIX: READ FROM SHADOW CACHE ---
            prev_matrix = shadow_cache[earlier_block]['matrix']
            prev_keys = shadow_cache[earlier_block]['keys']
            
            n_prev = len(prev_keys)
            T = np.full((n_prev, n_haps), -np.inf)
            
            t_probs_block = transition_probs_dict[earlier_block]
            
            # Fill Transition Matrix T (O(H^2))
            for r, p_key in enumerate(prev_keys):
                for c, u_key in enumerate(hap_keys):
                    lookup = ((earlier_block, p_key), (i, u_key))
                    if lookup in t_probs_block:
                        T[r, c] = math.log(t_probs_block[lookup])
            
            # Matrix Update (O(H^3))
            # Curr = T.T @ Prev @ T
            Z = analysis_utils.log_matmul(prev_matrix, T)
            pred_matrix = analysis_utils.log_matmul(T.T, Z)
            current_matrix = pred_matrix + E

        # 3. Store Results
        
        # A. Save Matrix to Shadow Cache for future iterations
        shadow_cache[i] = {'matrix': current_matrix, 'keys': hap_keys}
        
        # B. Convert Matrix to Dictionary for Output
        # Dict(u, v) = Matrix(u, v) + log(2) (if u != v)
        corrections = np.full((n_haps, n_haps), ln_2)
        np.fill_diagonal(corrections, 0.0)
        final_dict_vals = current_matrix + corrections
        
        result_dict = {}
        for r in range(n_haps):
            for c in range(r, n_haps): # Upper triangle
                key = ((i, hap_keys[r]), (i, hap_keys[c]))
                result_dict[key] = final_dict_vals[r, c]

        likelihood_numbers[i] = result_dict

    return likelihood_numbers

# --- 3. BACKWARD PASS (FIXED KEYS) ---
def get_full_probs_backward(sample_data, sample_sites, haps_data,
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
            full_blocks_likelihoods = pool.starmap(get_all_block_likelihoods_vectorised, tasks)

    # Use Backward Dictionary (Index 1)
    transition_probs_dict = bidirectional_transition_probs[1]
    
    likelihood_numbers = {} 
    shadow_cache = {} 
    
    ln_2 = math.log(2)

    # Iterate Backwards (N-1 -> 0)
    for i in range(len(haps_data)-1, -1, -1):
        
        # 1. Get Current Observation Likelihoods
        (block_lik_dict, _) = full_blocks_likelihoods[i]
        if isinstance(block_lik_dict, list): block_lik_dict = block_lik_dict[0]

        hap_keys = sorted(list(haps_data[i][3].keys()))
        n_haps = len(hap_keys)
        key_to_idx = {k: idx for idx, k in enumerate(hap_keys)}
        
        # Emission Matrix E
        E = np.full((n_haps, n_haps), -np.inf)
        for (u, v), val in block_lik_dict.items():
            u_idx, v_idx = key_to_idx[u], key_to_idx[v]
            E[u_idx, v_idx] = val
            E[v_idx, u_idx] = val
        
        # 2. Compute Backward Probabilities
        if i >= len(haps_data) - space_gap:
            # Base Case (End of chain)
            corrections = np.full((n_haps, n_haps), -ln_2)
            np.fill_diagonal(corrections, 0.0)
            current_matrix = E + corrections
            
        else:
            # Recursive Step
            # We look at the "Future" block (physically higher index)
            future_block = i + space_gap
            
            future_matrix = shadow_cache[future_block]['matrix']
            future_keys = shadow_cache[future_block]['keys']
            n_fut = len(future_keys)
            
            # --- FIX: Access Correct Dictionary Key ---
            # The backward dict is indexed by the START of the backward transition
            # which is 'future_block' (i + gap).
            t_probs_block = transition_probs_dict[future_block]
            
            # Build Transition Matrix T (Curr_Haps x Future_Haps)
            # We want to sum over Future states p.
            # L_curr(u) = Sum_p ( L_future(p) * P(p -> u) )
            # T[u, p] = P(Future=p -> Curr=u)
            T = np.full((n_haps, n_fut), -np.inf)
            
            for r, u_key in enumerate(hap_keys): # u (Current)
                for c, p_key in enumerate(future_keys): # p (Future)
                    # --- FIX: Key Order ---
                    # Backward Dict keys are: ((Start_Block, Start_Hap), (End_Block, End_Hap))
                    # Here Start is Future (i+gap), End is Current (i)
                    lookup = ((future_block, p_key), (i, u_key))
                    
                    if lookup in t_probs_block:
                        T[r, c] = math.log(t_probs_block[lookup])
            
            # Matrix Update
            # We want: Curr[u, v] = Sum_{p,q} ( T[u,p] + T[v,q] + Future[p,q] )
            # Matrix Algebra: Curr = T @ Future @ T.T
            
            # 1. Z = Future @ T.T (Dimensions: n_fut x n_curr)
            # Future (n_fut, n_fut) @ T.T (n_fut, n_curr)
            Z = analysis_utils.log_matmul(future_matrix, T.T)
            
            # 2. Res = T @ Z (Dimensions: n_curr x n_curr)
            # T (n_curr, n_fut) @ Z (n_fut, n_curr)
            pred_matrix = analysis_utils.log_matmul(T, Z)
            
            # 3. Add Current Emission
            current_matrix = pred_matrix + E

        # Store Results
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


def get_all_data_forward_probs(full_samples_data,sample_sites,
                               haps_data,
                               bidirectional_transition_probs,
                               full_blocks_likelihoods=None,
                               space_gap=1):
    
    
    if full_blocks_likelihoods == None:
        
        tasks = []
        for i in range(len(haps_data)):
            # Extract data for block i
            # This slicing part can be optimized if full_samples_data is massive,
            # but the worker optimization is the main gain.
            block_samples = analysis_utils.get_sample_data_at_sites_multiple(
                full_samples_data, sample_sites, haps_data[i][0], ploidy_present=True
            )
            tasks.append((block_samples, haps_data[i]))

        with Pool(16) as pool:
            full_blocks_likelihoods = pool.starmap(
                get_all_block_likelihoods_vectorised, 
                tasks
            )

    all_block_likelihoods = []
    
    for i in range(len(full_samples_data)):
        all_block_likelihoods.append(
            [full_blocks_likelihoods[x][i] for x in range(len(full_blocks_likelihoods))])
        
    forward_nums = list(itertools.starmap(
        lambda i : get_full_probs_forward(full_samples_data[i],
                    sample_sites,
                    haps_data,
                    bidirectional_transition_probs,
                    all_block_likelihoods[i],
                    space_gap=space_gap),
        zip(range(len(full_samples_data)))))
    
    overall_likelihoods = []
    
    num_vals = len(forward_nums[0])
    
    for i in range(len(forward_nums)):
        sample_vals = forward_nums[i][num_vals-1]
        
        max_likelihood_pair = max(sample_vals,key=sample_vals.get)
        
        max_likelihood_val = sample_vals[max_likelihood_pair]
        
        overall_likelihoods.append(max_likelihood_val)
    
    return (overall_likelihoods,forward_nums)

def get_all_data_backward_probs(full_samples_data,sample_sites,
                               haps_data,
                               bidirectional_transition_probs,
                               full_blocks_likelihoods=None,
                               space_gap=1):
    
    
    if full_blocks_likelihoods == None:
        
        tasks = []
        for i in range(len(haps_data)):
            # Extract data for block i
            # This slicing part can be optimized if full_samples_data is massive,
            # but the worker optimization is the main gain.
            block_samples = analysis_utils.get_sample_data_at_sites_multiple(
                full_samples_data, sample_sites, haps_data[i][0], ploidy_present=True
            )
            tasks.append((block_samples, haps_data[i]))

        with Pool(16) as pool:
            full_blocks_likelihoods = pool.starmap(
                get_all_block_likelihoods_vectorised, 
                tasks
            )

    all_block_likelihoods = []
    
    for i in range(len(full_samples_data)):
        all_block_likelihoods.append(
            [full_blocks_likelihoods[x][i] for x in range(len(full_blocks_likelihoods))])
        
    backward_nums = list(itertools.starmap(
        lambda i : get_full_probs_backward(full_samples_data[i],
                    sample_sites,
                    haps_data,
                    bidirectional_transition_probs,
                    all_block_likelihoods[i],
                    space_gap=space_gap),
        zip(range(len(full_samples_data)))))
    
    overall_likelihoods = []
    
    for i in range(len(backward_nums)):
        sample_vals = backward_nums[i][0]
        
        max_likelihood_pair = max(sample_vals,key=sample_vals.get)
        
        max_likelihood_val = sample_vals[max_likelihood_pair]
        
        overall_likelihoods.append(max_likelihood_val)
    
    return (overall_likelihoods,backward_nums)


def get_updated_transition_probabilities(
        full_samples_data,
        sample_sites,
        haps_data,
        current_transition_probs,
        full_blocks_likelihoods,
        currently_found_long_haps=[],
        space_gap=1,
        minimum_transition_log_likelihood=-10,
        BATCH_SIZE=100): # Batch size to control RAM usage
    """
    Calculates updated transition probabilities using Vectorized Forward-Backward
    and Batched Memory-Safe Tensor aggregation.
    """

    # 1. SETUP & HELPER DATA GATHERING
    prior_a_posteriori = initial_transition_probabilities(haps_data, space_gap=space_gap,
                        found_haps=currently_found_long_haps)

    full_samples_likelihoods = full_samples_data[0]
    
    # Restructure block likelihoods
    all_block_likelihoods = []
    for i in range(len(full_samples_likelihoods)):
        all_block_likelihoods.append(
            [(full_blocks_likelihoods[x][0][i], full_blocks_likelihoods[x][1][i])
             for x in range(len(full_blocks_likelihoods))])
             
    # Calculate Forward and Backward probabilities using VECTORIZED functions
    # No starmap/lambda overhead here, just direct calls
    forward_nums = [get_full_probs_forward(
                        (full_samples_data[0][i], full_samples_data[1][i]),
                        sample_sites, haps_data, current_transition_probs, 
                        all_block_likelihoods[i], space_gap=space_gap
                    ) for i in range(len(full_samples_likelihoods))]

    backward_nums = [get_full_probs_backward(
                        (full_samples_data[0][i], full_samples_data[1][i]),
                        sample_sites, haps_data, current_transition_probs, 
                        all_block_likelihoods[i], space_gap=space_gap
                    ) for i in range(len(full_samples_likelihoods))]
    
    samples_probs = list(zip(forward_nums, backward_nums))
    num_samples = len(samples_probs)

    # 2. CORE VECTORIZATION LOGIC (Batched)
    def _run_batched_pass(indices, is_forward):
        new_transition_probs = {}
        dir_idx = 0 if is_forward else 1
        
        for i in indices:
            next_bundle = i + space_gap if is_forward else i - space_gap
            
            hap_keys_current = sorted(list(haps_data[i][3].keys()))
            hap_keys_next    = sorted(list(haps_data[next_bundle][3].keys()))
            n_curr = len(hap_keys_current)
            n_next = len(hap_keys_next)
            
            # Build Static Tensors (Small enough for memory)
            T_matrix = np.full((n_curr, n_next), -np.inf)
            P_matrix = np.full((n_curr, n_next), -np.inf)

            for u_idx, u in enumerate(hap_keys_current):
                for v_idx, v in enumerate(hap_keys_next):
                    trans_key = ((i, u), (next_bundle, v))
                    if trans_key in current_transition_probs[dir_idx][i]:
                        T_matrix[u_idx, v_idx] = math.log(current_transition_probs[dir_idx][i][trans_key])
                    if trans_key in prior_a_posteriori[dir_idx][i]:
                        P_matrix[u_idx, v_idx] = math.log(prior_a_posteriori[dir_idx][i][trans_key])

            # Build Sample Tensors (F & B)
            # Allocation (Samples * 20 * 20 * 8 bytes) is generally safe
            F_tensor = np.full((num_samples, n_curr, n_curr), -np.inf)
            B_tensor = np.full((num_samples, n_next, n_next), -np.inf)

            # Fill Tensors from Dictionaries
            for s in range(num_samples):
                fwd_dict = samples_probs[s][0][i]
                bwd_dict = samples_probs[s][1][next_bundle]
                
                # Fill F
                for u_out_idx, u_out in enumerate(hap_keys_current):
                    for u_in_idx, u_in in enumerate(hap_keys_current):
                        key = ((i, u_out), (i, u_in)) if u_out <= u_in else ((i, u_in), (i, u_out))
                        if key in fwd_dict: F_tensor[s, u_out_idx, u_in_idx] = fwd_dict[key]

                # Fill B
                for v_out_idx, v_out in enumerate(hap_keys_next):
                    for v_in_idx, v_in in enumerate(hap_keys_next):
                        key = ((next_bundle, v_out), (next_bundle, v_in)) if v_out <= v_in else ((next_bundle, v_in), (next_bundle, v_out))
                        if key in bwd_dict: B_tensor[s, v_out_idx, v_in_idx] = bwd_dict[key]

            # BATCHED CALCULATION
            batch_results = []
            
            for start_idx in range(0, num_samples, BATCH_SIZE):
                end_idx = min(start_idx + BATCH_SIZE, num_samples)
                
                F_batch = F_tensor[start_idx:end_idx]
                B_batch = B_tensor[start_idx:end_idx]
                
                # 5D Broadcasting: (Batch, u_out, v_out, u_in, v_in)
                combined = (F_batch[:, :, np.newaxis, :, np.newaxis] + 
                            B_batch[:, np.newaxis, :, np.newaxis, :] + 
                            T_matrix[np.newaxis, np.newaxis, np.newaxis, :, :])
                
                # Marginalize Inner States & Normalize
                sample_lik_batch = logsumexp(combined, axis=(-2, -1))
                total_per_sample = logsumexp(sample_lik_batch, axis=(1, 2), keepdims=True)
                batch_aggregated = logsumexp(sample_lik_batch - total_per_sample, axis=0)
                batch_results.append(batch_aggregated)
                del combined # Free RAM
            
            # Merge & Finalize
            if len(batch_results) > 0:
                final_aggregated = logsumexp(batch_results, axis=0)
            else:
                final_aggregated = np.full((n_curr, n_next), -np.inf)
            
            posterior_with_prior = final_aggregated + P_matrix
            row_sums = logsumexp(posterior_with_prior, axis=1, keepdims=True)
            log_probs = posterior_with_prior - row_sums
            log_probs_clipped = np.maximum(log_probs, minimum_transition_log_likelihood)
            
            probs_nonnorm = np.exp(log_probs_clipped)
            row_sums_final = np.sum(probs_nonnorm, axis=1, keepdims=True)
            row_sums_final[row_sums_final == 0] = 1.0 
            final_probs_matrix = probs_nonnorm / row_sums_final
            
            block_dict = {}
            for u_idx, u in enumerate(hap_keys_current):
                for v_idx, v in enumerate(hap_keys_next):
                    key = ((i, u), (next_bundle, v))
                    block_dict[key] = final_probs_matrix[u_idx, v_idx]
            
            new_transition_probs[i] = block_dict
            
        return new_transition_probs

    forward_indices = range(len(haps_data) - space_gap)
    new_transition_probs_forward = _run_batched_pass(forward_indices, is_forward=True)
    
    backward_indices = range(len(haps_data) - 1, space_gap - 1, -1)
    new_transition_probs_backwards = _run_batched_pass(backward_indices, is_forward=False)
    
    return [new_transition_probs_forward, new_transition_probs_backwards]

def standard_get_updated_transition_probabilities(
        full_samples_data,
        sample_sites,
        haps_data,
        current_transition_probs,
        full_blocks_likelihoods,
        currently_found_long_haps=[],
        space_gap=1,
        minimum_transition_log_likelihood=-10,
        BATCH_SIZE=100):
    """
    Like get_updated_transition_probabilities but using 
    standard Baum-Welch where we use the full prior for 
    update purposes instead of only the part of the prior
    corresponding to the other haplotype.
    """

    # ... [Setup code remains exactly the same] ...
    prior_a_posteriori = initial_transition_probabilities(haps_data, space_gap=space_gap,
                        found_haps=currently_found_long_haps)

    full_samples_likelihoods = full_samples_data[0]
    all_block_likelihoods = []
    for i in range(len(full_samples_likelihoods)):
        all_block_likelihoods.append(
            [(full_blocks_likelihoods[x][0][i], full_blocks_likelihoods[x][1][i])
             for x in range(len(full_blocks_likelihoods))])
             
    forward_nums = [get_full_probs_forward(
                        (full_samples_data[0][i], full_samples_data[1][i]),
                        sample_sites, haps_data, current_transition_probs, 
                        all_block_likelihoods[i], space_gap=space_gap
                    ) for i in range(len(full_samples_likelihoods))]

    backward_nums = [get_full_probs_backward(
                        (full_samples_data[0][i], full_samples_data[1][i]),
                        sample_sites, haps_data, current_transition_probs, 
                        all_block_likelihoods[i], space_gap=space_gap
                    ) for i in range(len(full_samples_likelihoods))]
    
    samples_probs = list(zip(forward_nums, backward_nums))
    num_samples = len(samples_probs)

    def _run_batched_pass(indices, is_forward):
        new_transition_probs = {}
        dir_idx = 0 if is_forward else 1
        
        for i in indices:
            next_bundle = i + space_gap if is_forward else i - space_gap
            
            hap_keys_current = sorted(list(haps_data[i][3].keys()))
            hap_keys_next    = sorted(list(haps_data[next_bundle][3].keys()))
            n_curr = len(hap_keys_current)
            n_next = len(hap_keys_next)
            
            # Static Tensors
            T_matrix = np.full((n_curr, n_next), -np.inf)
            P_matrix = np.full((n_curr, n_next), -np.inf)

            for u_idx, u in enumerate(hap_keys_current):
                for v_idx, v in enumerate(hap_keys_next):
                    trans_key = ((i, u), (next_bundle, v))
                    if trans_key in current_transition_probs[dir_idx][i]:
                        T_matrix[u_idx, v_idx] = math.log(current_transition_probs[dir_idx][i][trans_key])
                    if trans_key in prior_a_posteriori[dir_idx][i]:
                        P_matrix[u_idx, v_idx] = math.log(prior_a_posteriori[dir_idx][i][trans_key])

            # Sample Tensors
            F_tensor = np.full((num_samples, n_curr, n_curr), -np.inf)
            B_tensor = np.full((num_samples, n_next, n_next), -np.inf)

            for s in range(num_samples):
                fwd_dict = samples_probs[s][0][i]
                bwd_dict = samples_probs[s][1][next_bundle]
                
                # Fill F (Current)
                # Matches image S(i, c, d)
                for u_out_idx, u_out in enumerate(hap_keys_current):
                    for u_in_idx, u_in in enumerate(hap_keys_current):
                        key = ((i, u_out), (i, u_in)) if u_out <= u_in else ((i, u_in), (i, u_out))
                        if key in fwd_dict: F_tensor[s, u_out_idx, u_in_idx] = fwd_dict[key]

                # Fill B (Next/Future)
                # Matches image R(i+1, a, b)
                for v_out_idx, v_out in enumerate(hap_keys_next):
                    for v_in_idx, v_in in enumerate(hap_keys_next):
                        key = ((next_bundle, v_out), (next_bundle, v_in)) if v_out <= v_in else ((next_bundle, v_in), (next_bundle, v_out))
                        if key in bwd_dict: B_tensor[s, v_out_idx, v_in_idx] = bwd_dict[key]

            # BATCHED CALCULATION
            batch_results = []
            
            for start_idx in range(0, num_samples, BATCH_SIZE):
                end_idx = min(start_idx + BATCH_SIZE, num_samples)
                
                F_batch = F_tensor[start_idx:end_idx] # (Batch, c, d)
                B_batch = B_tensor[start_idx:end_idx] # (Batch, a, b)
                
                # We want to broadcast to shape: (Batch, c, a, d, b)
                # c = u_out (Current Main Hap) -> Dim 1
                # a = v_out (Next Main Hap)    -> Dim 2
                # d = u_in  (Current Partner)  -> Dim 3
                # b = v_in  (Next Partner)     -> Dim 4
                
                # F(Batch, c, d) -> (Batch, c, 1, d, 1)
                F_broad = F_batch[:, :, np.newaxis, :, np.newaxis]
                
                # B(Batch, a, b) -> (Batch, 1, a, 1, b)
                B_broad = B_batch[:, np.newaxis, :, np.newaxis, :]
                
                # T_main(c, a) -> (1, c, a, 1, 1)
                # This corresponds to t^f(a, c) in your writeup (The one being updated)
                T_main_broad = T_matrix[np.newaxis, :, :, np.newaxis, np.newaxis]
                
                # --- THE FIX: T_partner(d, b) ---
                # This corresponds to t^f(b, d) in your writeup (The partner transition)
                # Shape: (1, 1, 1, d, b)
                T_partner_broad = T_matrix[np.newaxis, np.newaxis, np.newaxis, :, :]
                
                # Combined Sum (Log Space)
                combined = F_broad + B_broad + T_main_broad + T_partner_broad
                
                # Marginalize Inner States (d, b)
                sample_lik_batch = logsumexp(combined, axis=(-2, -1))
                
                # Normalize & Aggregate
                total_per_sample = logsumexp(sample_lik_batch, axis=(1, 2), keepdims=True)
                batch_aggregated = logsumexp(sample_lik_batch - total_per_sample, axis=0)
                batch_results.append(batch_aggregated)
                del combined 
            
            # Merge & Finalize (Same as before)
            if len(batch_results) > 0:
                final_aggregated = logsumexp(batch_results, axis=0)
            else:
                final_aggregated = np.full((n_curr, n_next), -np.inf)
            
            posterior_with_prior = final_aggregated + P_matrix
            row_sums = logsumexp(posterior_with_prior, axis=1, keepdims=True)
            log_probs = posterior_with_prior - row_sums
            log_probs_clipped = np.maximum(log_probs, minimum_transition_log_likelihood)
            
            probs_nonnorm = np.exp(log_probs_clipped)
            row_sums_final = np.sum(probs_nonnorm, axis=1, keepdims=True)
            row_sums_final[row_sums_final == 0] = 1.0 
            final_probs_matrix = probs_nonnorm / row_sums_final
            
            block_dict = {}
            for u_idx, u in enumerate(hap_keys_current):
                for v_idx, v in enumerate(hap_keys_next):
                    key = ((i, u), (next_bundle, v))
                    block_dict[key] = final_probs_matrix[u_idx, v_idx]
            
            new_transition_probs[i] = block_dict
            
        return new_transition_probs

    forward_indices = range(len(haps_data) - space_gap)
    new_transition_probs_forward = _run_batched_pass(forward_indices, is_forward=True)
    
    backward_indices = range(len(haps_data) - 1, space_gap - 1, -1)
    new_transition_probs_backwards = _run_batched_pass(backward_indices, is_forward=False)
    
    return [new_transition_probs_forward, new_transition_probs_backwards]

def calculate_hap_transition_probabilities(full_samples_data,
            sample_sites,
            haps_data,
            full_blocks_likelihoods=None,
            max_num_iterations=10,
            space_gap=1,
            currently_found_long_haps=[],
            min_cutoff_change=0.001, # Relaxed from 1e-8
            learning_rate=1.0,       # Start high
            minimum_transition_log_likelihood=-10):
    
    # 1. INITIALIZATION
    start_probs = initial_transition_probabilities(haps_data, space_gap=space_gap)
    
    # Check/Calc Likelihoods
    if full_blocks_likelihoods is None:
        print("Warning: full_blocks_likelihoods not provided. Calculating serially.")
        full_blocks_likelihoods = []
        for i in range(len(haps_data)):
            block_samples = analysis_utils.get_sample_data_at_sites_multiple(
                full_samples_data, sample_sites, haps_data[i][0], ploidy_present=True
            )
            full_blocks_likelihoods.append(
                get_all_block_likelihoods_vectorised((block_samples, haps_data[i]))
            )

    current_probs = start_probs
    
    # 2. EM LOOP
    for i in range(max_num_iterations):
        
        # --- ADAPTIVE LEARNING RATE ---
        # Decay the learning rate as iterations progress.
        # Iter 0: 1.0
        # Iter 5: 0.59
        # Iter 10: 0.34
        # This allows fast movement early on, but forces stability later.
        effective_lr = learning_rate * (0.9 ** i)
        
        # Lower bound to ensure we don't stop updating completely
        effective_lr = max(effective_lr, 0.1)
        
        # A. M-STEP
        new_probs_raw = standard_get_updated_transition_probabilities(
            full_samples_data,
            sample_sites,
            haps_data,
            current_probs, 
            full_blocks_likelihoods,
            currently_found_long_haps=currently_found_long_haps,
            space_gap=space_gap,
            minimum_transition_log_likelihood=minimum_transition_log_likelihood,
            BATCH_SIZE=100
        )
        
        # B. SMOOTHING (With Effective LR)
        current_probs_new = analysis_utils.smoothen_probs_vectorized(current_probs, new_probs_raw, effective_lr)

        # C. CONVERGENCE CHECK
        global_max_diff = 0.0
        
        for direction in [0, 1]:
            for block_idx in current_probs_new[direction]:
                d_new = current_probs_new[direction][block_idx]
                d_old = current_probs[direction][block_idx]
                
                v_new = np.fromiter(d_new.values(), dtype=float)
                v_old = np.fromiter(d_old.values(), dtype=float)
                
                block_max = np.max(np.abs(v_new - v_old))
                if block_max > global_max_diff:
                    global_max_diff = block_max
        
        # Debugging Output
        print(f"Gap {space_gap} | Iter {i} | LR {effective_lr:.2f} | Max Diff: {global_max_diff:.6f}")
        
        current_probs = current_probs_new
        
        if global_max_diff < min_cutoff_change:
            break
            
    return current_probs

# --- 1. GLOBAL STORAGE FOR WORKERS ---
# This dictionary will live in the memory of the worker processes
_worker_data = {}

def _init_worker_transition(samples_data, sites, haps, block_likes, 
                            found_haps, min_log_lik, lr, max_iter):
    """
    This runs once per process when the pool starts.
    It saves the heavy read-only data into the worker's global scope.
    """
    _worker_data['full_samples_data'] = samples_data
    _worker_data['sample_sites'] = sites
    _worker_data['haps_data'] = haps
    _worker_data['full_blocks_likelihoods'] = block_likes
    _worker_data['currently_found_long_haps'] = found_haps
    _worker_data['minimum_transition_log_likelihood'] = min_log_lik
    _worker_data['learning_rate'] = lr
    _worker_data['max_num_iterations'] = max_iter

def _worker_calculate_gap_wrapper(space_gap):
    """
    The lightweight wrapper that calls your actual logic using the cached global data.
    """
    # Import your logic function here if it's in another module
    # form analysis_utils import calculate_hap_transition_probabilities 
    
    return calculate_hap_transition_probabilities(
        _worker_data['full_samples_data'],
        _worker_data['sample_sites'],
        _worker_data['haps_data'],
        full_blocks_likelihoods=_worker_data['full_blocks_likelihoods'],
        max_num_iterations=_worker_data['max_num_iterations'],
        space_gap=space_gap,
        currently_found_long_haps=_worker_data['currently_found_long_haps'],
        minimum_transition_log_likelihood=_worker_data['minimum_transition_log_likelihood'],
        learning_rate=_worker_data['learning_rate']
    )

def generate_transition_probability_mesh(full_samples_data,
                                         sample_sites,
                                         haps_data,
                                         max_num_iterations=10,
                                         currently_found_long_haps=[],
                                         minimum_transition_log_likelihood=-10,
                                         learning_rate=1):
    """
    Optimized version using Initializers to avoid copying heavy data repeatedly.
    """
    
    # --- PHASE 1: GENERATE BLOCK LIKELIHOODS ---
    
    # Use your existing vectorized function here
    tasks = []
    for i in range(len(haps_data)):
        # Note: If full_samples_data is HUGE, slicing it here might still be slow.
        # Ideally, get_sample_data... happens inside the worker too, but
        # keeping your logic for now:
        block_samples = analysis_utils.get_sample_data_at_sites_multiple(
            full_samples_data, sample_sites, haps_data[i][0], ploidy_present=True
        )
        tasks.append((block_samples, haps_data[i]))

    # Use a smaller pool for the first step if RAM is tight
    # Ensure OMP_NUM_THREADS=1 is set in your main script!
    print("Calculating block likelihoods...")
    with Pool(16) as pool:
        full_blocks_likelihoods = pool.starmap(
            get_all_block_likelihoods_vectorised, 
            tasks
        )
    
    # --- PHASE 2: CALCULATE TRANSITION MESH ---
    
    num_sqrs = math.floor(math.sqrt(len(haps_data)-1))
    sqrs = [i**2 for i in range(1, num_sqrs+1)]
    
    print(f"Calculating transitions for {len(sqrs)} different gaps...")

    # We pack all the heavy/static arguments into a tuple for the initializer
    init_args = (full_samples_data, sample_sites, haps_data, full_blocks_likelihoods,
                 currently_found_long_haps,
                 minimum_transition_log_likelihood, learning_rate, max_num_iterations)

    # We use the initializer to send data ONCE per process
    with Pool(16, initializer=_init_worker_transition, initargs=init_args) as pool:
        
        # We only pass the changing variable 'x' (the gap size)
        # map/imap is cleaner than starmap for single arguments
        results = pool.map(_worker_calculate_gap_wrapper, sqrs)
    
    # Construct dictionary
    mesh_dict = dict(zip(sqrs, results))
    
    return mesh_dict