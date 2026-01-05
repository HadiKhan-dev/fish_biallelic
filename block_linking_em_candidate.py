import numpy as np
import math
import time
from multiprocess import Pool
from scipy.special import logsumexp
from functools import partial

import analysis_utils
import block_haplotypes

# Define defaults as constants
DEFAULT_LOG_BASE = math.e
DEFAULT_MIN_PER_SITE = -0.5
DEFAULT_UNIFORM_FLOOR = -0.6
DEFAULT_MIN_BLOCK_LL = -200.0

class TransitionMesh:
    """
    A specialized container for transition probability meshes across different gap sizes.
    
    Attributes:
        forward (dict): Maps gap_size -> Forward transition dictionary.
                        Forward dict maps ((block_i, hap_u), (block_j, hap_v)) -> prob
        backward (dict): Maps gap_size -> Backward transition dictionary.
                         Backward dict maps ((block_j, hap_v), (block_i, hap_u)) -> prob
    """
    def __init__(self, raw_gap_results=None):
        self.forward = {}
        self.backward = {}
        
        if raw_gap_results:
            for gap, probs_pair in raw_gap_results.items():
                # probs_pair is [fwd_dict, bwd_dict]
                self.forward[gap] = probs_pair[0]
                self.backward[gap] = probs_pair[1]

    def __getitem__(self, gap):
        """
        Legacy compatibility: allows accessing mesh[gap] which returns [forward, backward].
        This ensures existing analysis code using mesh[gap][0] still works.
        """
        return [self.forward.get(gap), self.backward.get(gap)]
    
    def __contains__(self, gap):
        return gap in self.forward
    
    def keys(self):
        return self.forward.keys()
    
    def items(self):
        # Yields (gap, [fwd, bwd]) to mimic dict.items()
        for gap in self.forward:
            yield gap, [self.forward[gap], self.backward[gap]]

# %% --- LIKELIHOOD GENERATION ---

def _worker_calculate_single_block_likelihood(args):
    """
    Internal worker function to calculate likelihoods for a single block.
    Args tuple: (block_samples_data, block_result, params_dict)
    """
    samples_matrix, block_hap, params = args
    
    # Unpack params
    log_likelihood_base = params.get('log_likelihood_base', DEFAULT_LOG_BASE)
    min_per_site_log_likelihood = params.get('min_per_site_log_likelihood', DEFAULT_MIN_PER_SITE)
    uniform_error_floor_per_site = params.get('uniform_error_floor_per_site', DEFAULT_UNIFORM_FLOOR)
    min_absolute_block_log_likelihood = params.get('min_absolute_block_log_likelihood', DEFAULT_MIN_BLOCK_LL)

    if len(samples_matrix) == 0:
        return []

    num_samples, num_sites, _ = samples_matrix.shape

    hap_dict = block_hap.haplotypes
    # Ensure flags are boolean
    if block_hap.keep_flags is None:
        keep_flags = np.ones(num_sites, dtype=bool)
    else:
        keep_flags = block_hap.keep_flags.astype(bool)
        
    hap_keys = sorted(list(hap_dict.keys()))
    num_haps = len(hap_keys)
    
    if num_haps == 0:
        return [{0: 0.0}] * num_samples

    # --- 1. ROBUST TENSOR CREATION ---
    hap_list = [hap_dict[k] for k in hap_keys]
    
    # Handle edge case where haplotypes exist but are empty arrays
    if len(hap_list) > 0 and (hap_list[0].size == 0):
        haps_tensor = np.zeros((num_haps, 0, 2))
    else:
        haps_tensor = np.array(hap_list)
    
    # --- 2. MASKING ---
    # Apply keep_flags to reduce computation to only valid sites
    samples_masked = samples_matrix[:, keep_flags, :] 
    haps_masked = haps_tensor[:, keep_flags, :]       
    
    num_active_sites = samples_masked.shape[1]
    
    if num_active_sites > 0:
        # --- 3. GENERATE DIPLOID COMBINATIONS ---
        # Haps are (N_Haps, Sites, 2)
        h0 = haps_masked[:, :, 0]
        h1 = haps_masked[:, :, 1]
        
        # Broadcasting to create (N_Haps, N_Haps, Sites)
        c00 = h0[:, None, :] * h0[None, :, :]
        c11 = h1[:, None, :] * h1[None, :, :]
        c01 = (h0[:, None, :] * h1[None, :, :]) + (h1[:, None, :] * h0[None, :, :])
        
        # Stack: (N_Haps, N_Haps, Sites, 3)
        combos_4d = np.stack([c00, c01, c11], axis=-1)
        
        # Flatten HxH -> (H^2, Sites, 3) for vectorized matmul
        combos_flat = combos_4d.reshape(-1, num_active_sites, 3)
        
        # --- 4. WEIGHTED DISTANCE CALCULATION ---
        # Weights: 0 vs 0 (0), 0 vs 1 (1), 0 vs 2 (2)
        dist_weights = np.array([[0, 1, 2], 
                                 [1, 0, 1], 
                                 [2, 1, 0]])
        
        # (H^2, Sites, 3) @ (3, 3) -> (H^2, Sites, 3)
        combos_weighted = combos_flat @ dist_weights
        
        # Calculate expected distance per site
        # Samples: (Samples, 1, Sites, 3)
        # Combos:  (1, H^2, Sites, 3)
        # Result:  (Samples, H^2, Sites) (Summed over feature dim)
        dist_per_site = np.sum(
            samples_masked[:, np.newaxis, :, :] * combos_weighted[np.newaxis, :, :, :], 
            axis=3
        )
        
        # --- 5. LOG LIKELIHOOD ---
        log_penalty = math.log(log_likelihood_base)
        ll_per_site = -(dist_per_site) * log_penalty
        
        # Apply per-site floor
        ll_per_site = np.maximum(ll_per_site, min_per_site_log_likelihood)
        
        # Sum over sites -> (Samples, H^2)
        total_ll_matrix = np.sum(ll_per_site, axis=2)
        
    else:
        # Handle 0 sites case
        total_ll_matrix = np.zeros((num_samples, num_haps**2))

    # --- 6. APPLY GLOBAL FLOORS & NORMALIZE ---
    if num_active_sites > 0:
        site_based_floor = num_active_sites * uniform_error_floor_per_site
        total_ll_matrix = np.maximum(total_ll_matrix, site_based_floor)
        
    total_ll_matrix = np.maximum(total_ll_matrix, min_absolute_block_log_likelihood)
        
    # Filter to unique pairs (Upper Triangle)
    idx_i, idx_j = np.unravel_index(np.arange(num_haps**2), (num_haps, num_haps))
    valid_mask = idx_j >= idx_i
    
    unique_ll_matrix = total_ll_matrix[:, valid_mask]
    
    # LogSumExp Normalization (Vectorized across samples)
    norm_factors = logsumexp(unique_ll_matrix, axis=1, keepdims=True)
    final_normalized_matrix = unique_ll_matrix - norm_factors
    
    # --- 7. FORMAT OUTPUT ---
    results = []
    keys = [(hap_keys[i], hap_keys[j]) for i, j in zip(idx_i[valid_mask], idx_j[valid_mask])]
    
    for s in range(num_samples):
        results.append(dict(zip(keys, final_normalized_matrix[s])))
            
    return results

def generate_all_block_likelihoods(
    sample_probs_matrix,
    global_site_locations,
    haplotype_data,
    num_processes=16,
    log_likelihood_base=math.e,
    min_per_site_log_likelihood=-0.5,
    uniform_error_floor_per_site=-0.6,
    min_absolute_block_log_likelihood=-200.0
):
    """
    Calculates diploid genotype likelihoods for all blocks against all samples.
    """
    
    # 1. Normalize Input (Single vs List)
    is_single_block = False
    
    if isinstance(haplotype_data, (list, tuple, block_haplotypes.BlockResults)):
        blocks_to_process = haplotype_data
    else:
        blocks_to_process = [haplotype_data]
        is_single_block = True
        
    # 2. Prepare Tasks
    tasks = []
    
    params = {
        'log_likelihood_base': log_likelihood_base,
        'min_per_site_log_likelihood': min_per_site_log_likelihood,
        'uniform_error_floor_per_site': uniform_error_floor_per_site,
        'min_absolute_block_log_likelihood': min_absolute_block_log_likelihood
    }
    
    for block in blocks_to_process:
        block_samples = analysis_utils.get_sample_data_at_sites_multiple(
            sample_probs_matrix, global_site_locations, block.positions
        )
        tasks.append((block_samples, block, params))

    # 3. Execution
    if num_processes > 1 and len(tasks) > 1:
        with Pool(num_processes) as pool:
            results = pool.map(_worker_calculate_single_block_likelihood, tasks)
    else:
        results = list(map(_worker_calculate_single_block_likelihood, tasks))

    # 4. Return Format
    if is_single_block:
        return results[0]
    return results

# %% --- EM HELPERS ---

def initial_transition_probabilities(haps_data, space_gap=1, found_haps=[]):
    """
    Creates a dict of initial equal transition probabilities.
    """
    transition_dict_forward = {}
    transition_dict_reverse = {}
    
    for i in range(0,len(haps_data)-space_gap):
        transition_dict_forward[i] = {}
        
        these_haps = haps_data[i].haplotypes
        next_haps = haps_data[i+space_gap].haplotypes
        
        for first_idx in these_haps.keys():
            first_hap_name = (i,first_idx)
            for second_idx in next_haps.keys():
                second_hap_name = (i+space_gap,second_idx)
                transition_dict_forward[i][(first_hap_name,second_hap_name)] = 1
                
    for i in range(len(haps_data)-1,space_gap-1,-1):
        transition_dict_reverse[i] = {}
        
        these_haps = haps_data[i].haplotypes
        next_haps = haps_data[i-space_gap].haplotypes
        
        for first_idx in these_haps.keys():
            first_hap_name = (i,first_idx)
            for second_idx in next_haps.keys():
                second_hap_name = (i-space_gap,second_idx)
                transition_dict_reverse[i][(first_hap_name,second_hap_name)] = 1
    
    # Scale probabilities
    scaled_dict_forward = {}
    scaled_dict_reverse = {}
    
    for idx in transition_dict_forward.keys():
        scaled_dict_forward[idx] = {}
        start_dict = {}
        for s in transition_dict_forward[idx].keys():
            start_dict[s[0]] = start_dict.get(s[0], 0) + transition_dict_forward[idx][s]
        
        for s in transition_dict_forward[idx].keys():
            scaled_dict_forward[idx][s] = transition_dict_forward[idx][s]/start_dict[s[0]]
        
    for idx in transition_dict_reverse.keys():
        scaled_dict_reverse[idx] = {}
        start_dict = {}
        for s in transition_dict_reverse[idx].keys():
            start_dict[s[0]] = start_dict.get(s[0], 0) + transition_dict_reverse[idx][s]
        
        for s in transition_dict_reverse[idx].keys():
            scaled_dict_reverse[idx][s] = transition_dict_reverse[idx][s]/start_dict[s[0]]
        
    return [scaled_dict_forward, scaled_dict_reverse]

# %% --- EM FORWARD/BACKWARD ---

def get_full_probs_forward(sample_data, sample_sites, haps_data,
                           bidirectional_transition_probs,
                           full_blocks_likelihoods=None,
                           space_gap=1):
    """
    Calculates forward variables (S).
    Returns (likelihood_dictionary, total_log_likelihood_of_sample).
    """
    
    if full_blocks_likelihoods is None:
        full_blocks_likelihoods = generate_all_block_likelihoods(
            sample_data[np.newaxis, :, :], 
            sample_sites, haps_data, num_processes=1
        )
        full_blocks_likelihoods = [x[0] for x in full_blocks_likelihoods]

    transition_probs_dict = bidirectional_transition_probs[0]
    
    likelihood_numbers = {} 
    shadow_cache = {} 
    
    ln_2 = math.log(2)
    final_sample_ll = -np.inf

    for i in range(len(haps_data)):
        
        block_lik_dict = full_blocks_likelihoods[i]
        if isinstance(block_lik_dict, list): block_lik_dict = block_lik_dict[0]

        hap_keys = sorted(list(haps_data[i].haplotypes.keys()))
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
        
        if i == len(haps_data) - 1:
            unique_states_ll = []
            for r in range(n_haps):
                for c in range(r, n_haps):
                    unique_states_ll.append(final_dict_vals[r,c])
            final_sample_ll = logsumexp(unique_states_ll)

    return (likelihood_numbers, final_sample_ll)

def get_full_probs_backward(sample_data, sample_sites, haps_data,
                           bidirectional_transition_probs,
                           full_blocks_likelihoods=None,
                           space_gap=1):
    
    if full_blocks_likelihoods is None:
        full_blocks_likelihoods = generate_all_block_likelihoods(
            sample_data[np.newaxis, :, :], 
            sample_sites, haps_data, num_processes=1
        )
        full_blocks_likelihoods = [x[0] for x in full_blocks_likelihoods]

    transition_probs_dict = bidirectional_transition_probs[1]
    
    likelihood_numbers = {} 
    shadow_cache = {} 
    
    ln_2 = math.log(2)

    for i in range(len(haps_data)-1, -1, -1):
        
        block_lik_dict = full_blocks_likelihoods[i]
        if isinstance(block_lik_dict, list): block_lik_dict = block_lik_dict[0]

        hap_keys = sorted(list(haps_data[i].haplotypes.keys()))
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
                    lookup = ((future_block, p_key), (i, u_key))
                    if lookup in t_probs_block:
                        T[r, c] = math.log(t_probs_block[lookup])
            
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

# %% --- UNIFIED UPDATE FUNCTION ---

def get_updated_transition_probabilities_unified(
        full_samples_data,
        sample_sites,
        haps_data,
        current_transition_probs,
        full_blocks_likelihoods,
        currently_found_long_haps=[],
        space_gap=1,
        minimum_transition_log_likelihood=-10,
        BATCH_SIZE=100,
        use_standard_baum_welch=True): 
    """
    Unified function to calculate updated transition probabilities.
    Returns: (new_transition_probs_list, total_data_log_likelihood)
    """

    prior_a_posteriori = initial_transition_probabilities(haps_data, space_gap=space_gap,
                        found_haps=currently_found_long_haps)

    full_samples_likelihoods = full_samples_data
    
    all_block_likelihoods = []
    num_blocks = len(full_blocks_likelihoods)
    num_samples = len(full_samples_likelihoods)
    
    for s in range(num_samples):
        sample_blocks = []
        for b in range(num_blocks):
            sample_blocks.append(full_blocks_likelihoods[b][s])
        all_block_likelihoods.append(sample_blocks)
             
    forward_results = [get_full_probs_forward(
                        full_samples_data[i],
                        sample_sites, haps_data, current_transition_probs, 
                        all_block_likelihoods[i], space_gap=space_gap
                    ) for i in range(len(full_samples_likelihoods))]
    
    forward_nums = [x[0] for x in forward_results]
    total_data_log_likelihood = sum([x[1] for x in forward_results])

    backward_nums = [get_full_probs_backward(
                        full_samples_data[i],
                        sample_sites, haps_data, current_transition_probs, 
                        all_block_likelihoods[i], space_gap=space_gap
                    ) for i in range(len(full_samples_likelihoods))]
    
    samples_probs = list(zip(forward_nums, backward_nums))
    
    # 2. CORE VECTORIZATION LOGIC (Batched)
    def _run_batched_pass(indices, is_forward):
        new_transition_probs = {}
        dir_idx = 0 if is_forward else 1
        
        for i in indices:
            next_bundle = i + space_gap if is_forward else i - space_gap
            
            hap_keys_current = sorted(list(haps_data[i].haplotypes.keys()))
            hap_keys_next    = sorted(list(haps_data[next_bundle].haplotypes.keys()))
            n_curr = len(hap_keys_current)
            n_next = len(hap_keys_next)
            
            T_matrix = np.full((n_curr, n_next), -np.inf)
            P_matrix = np.full((n_curr, n_next), -np.inf)

            for u_idx, u in enumerate(hap_keys_current):
                for v_idx, v in enumerate(hap_keys_next):
                    trans_key = ((i, u), (next_bundle, v))
                    if trans_key in current_transition_probs[dir_idx][i]:
                        T_matrix[u_idx, v_idx] = math.log(current_transition_probs[dir_idx][i][trans_key])
                    if trans_key in prior_a_posteriori[dir_idx][i]:
                        P_matrix[u_idx, v_idx] = math.log(prior_a_posteriori[dir_idx][i][trans_key])

            F_tensor = np.full((num_samples, n_curr, n_curr), -np.inf)
            B_tensor = np.full((num_samples, n_next, n_next), -np.inf)

            for s in range(num_samples):
                fwd_dict = samples_probs[s][0][i]
                bwd_dict = samples_probs[s][1][next_bundle]
                
                # Fill F - Keys: ((i, u_out), (i, u_in))
                for u_out_idx, u_out in enumerate(hap_keys_current):
                    for u_in_idx, u_in in enumerate(hap_keys_current):
                        # Ensure consistent key ordering (symmetric)
                        key = ((i, u_out), (i, u_in)) if u_out <= u_in else ((i, u_in), (i, u_out))
                        if key in fwd_dict: F_tensor[s, u_out_idx, u_in_idx] = fwd_dict[key]

                # Fill B - Keys: ((j, v_out), (j, v_in))
                for v_out_idx, v_out in enumerate(hap_keys_next):
                    for v_in_idx, v_in in enumerate(hap_keys_next):
                        key = ((next_bundle, v_out), (next_bundle, v_in)) if v_out <= v_in else ((next_bundle, v_in), (next_bundle, v_out))
                        if key in bwd_dict: B_tensor[s, v_out_idx, v_in_idx] = bwd_dict[key]

            batch_results = []
            
            for start_idx in range(0, num_samples, BATCH_SIZE):
                end_idx = min(start_idx + BATCH_SIZE, num_samples)
                
                F_batch = F_tensor[start_idx:end_idx]
                B_batch = B_tensor[start_idx:end_idx]
                
                # Broadcast
                F_broad = F_batch[:, :, :, np.newaxis, np.newaxis]
                B_broad = B_batch[:, np.newaxis, np.newaxis, :, :]
                
                # T_partner is used in both versions (Transitions of the 'other' haplotype)
                T_partner_broad = T_matrix[np.newaxis, np.newaxis, :, np.newaxis, :]
                
                combined = F_broad + B_broad + T_partner_broad
                
                if use_standard_baum_welch:
                    # T_main corresponds to U_main (dim 1) -> V_main (dim 3)
                    T_main_broad = T_matrix[np.newaxis, :, np.newaxis, :, np.newaxis]
                    combined += T_main_broad
                
                # Marginalize over Partners (dim 2 and 4)
                sample_lik_batch = logsumexp(combined, axis=(2, 4))
                
                total_per_sample = logsumexp(sample_lik_batch, axis=(1, 2), keepdims=True)
                batch_aggregated = logsumexp(sample_lik_batch - total_per_sample, axis=0)
                batch_results.append(batch_aggregated)
                del combined 
            
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
    
    return ([new_transition_probs_forward, new_transition_probs_backwards], total_data_log_likelihood)

def calculate_hap_transition_probabilities(full_samples_data,
            sample_sites,
            haps_data,
            full_blocks_likelihoods=None,
            max_num_iterations=10,
            space_gap=1,
            currently_found_long_haps=[],
            min_cutoff_change=0.001,
            ll_improvement_cutoff=1e-4,
            learning_rate=1.0, 
            minimum_transition_log_likelihood=-10,
            use_standard_baum_welch=True):
    
    start_probs = initial_transition_probabilities(haps_data, space_gap=space_gap,
                        found_haps=currently_found_long_haps)
    
    if full_blocks_likelihoods is None:
        print("Warning: full_blocks_likelihoods not provided. Calculating.")
        full_blocks_likelihoods = generate_all_block_likelihoods(
            full_samples_data, sample_sites, haps_data
        )

    current_probs = start_probs
    prev_ll = -np.inf
    
    for i in range(max_num_iterations):
        effective_lr = learning_rate * (0.9 ** i)
        effective_lr = max(effective_lr, 0.1)
        
        new_probs_raw, current_ll = get_updated_transition_probabilities_unified(
            full_samples_data,
            sample_sites,
            haps_data,
            current_probs, 
            full_blocks_likelihoods,
            currently_found_long_haps=currently_found_long_haps,
            space_gap=space_gap,
            minimum_transition_log_likelihood=minimum_transition_log_likelihood,
            BATCH_SIZE=100,
            use_standard_baum_welch=use_standard_baum_welch
        )
        
        # Returns dict if mixed LR, so ensure it is a list [fwd, bwd]
        current_probs_smoothed = analysis_utils.smoothen_probs_vectorized(current_probs, new_probs_raw, effective_lr)
        
        if isinstance(current_probs_smoothed, dict):
            current_probs_new = [current_probs_smoothed[0], current_probs_smoothed[1]]
        else:
            current_probs_new = current_probs_smoothed

        ll_diff = current_ll - prev_ll
        current_probs = current_probs_new
        
        # Relative improvement check
        rel_improvement = 0.0
        if prev_ll != -np.inf and prev_ll != 0:
            rel_improvement = (current_ll - prev_ll) / abs(prev_ll)
        elif prev_ll == -np.inf:
            rel_improvement = float('inf') # First iteration always improves
            
        if i > 0 and 0 <= rel_improvement < ll_improvement_cutoff:
            break
            
        prev_ll = current_ll
            
    return current_probs

# %% --- WORKER WRAPPER FOR POOL ---

def _gap_worker(args):
    """
    Unpacks arguments and calls the calculation function.
    """
    (gap, full_samples, sites, haps, likes, max_iter, found, min_ll, lr) = args
    
    return calculate_hap_transition_probabilities(
        full_samples, 
        sites, 
        haps, 
        full_blocks_likelihoods=likes,
        max_num_iterations=max_iter,
        space_gap=gap,
        currently_found_long_haps=found,
        minimum_transition_log_likelihood=min_ll,
        learning_rate=lr
    )

def generate_transition_probability_mesh(full_samples_data,
                                         sample_sites,
                                         haps_data,
                                         max_num_iterations=10,
                                         currently_found_long_haps=[],
                                         minimum_transition_log_likelihood=-10,
                                         learning_rate=1):
    
    print("Calculating block likelihoods (Mesh Generation)...")
    full_blocks_likelihoods = generate_all_block_likelihoods(
        full_samples_data, sample_sites, haps_data
    )
    
    max_gap = len(haps_data) - 1
    gaps = list(range(1, max_gap + 1))
    
    print(f"Calculating transitions for {len(gaps)} different gaps...")

    worker_args = []
    for gap in gaps:
        worker_args.append((
            gap,
            full_samples_data,
            sample_sites,
            haps_data,
            full_blocks_likelihoods,
            max_num_iterations,
            currently_found_long_haps,
            minimum_transition_log_likelihood,
            learning_rate
        ))

    with Pool(16) as pool:
        results = pool.map(_gap_worker, worker_args)
    
    mesh_dict = dict(zip(gaps, results))
    return TransitionMesh(mesh_dict)