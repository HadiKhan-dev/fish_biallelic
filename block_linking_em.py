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
# Robustness parameter: 1e-2 means 1% chance any read is random noise/error.
# This prevents high-depth outliers from forcing incorrect recombinations.
DEFAULT_ROBUSTNESS_EPSILON = 1e-2

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("WARNING: Numba not found. Computations will be extremely slow.")
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator
    prange = range

# %% --- NUMBA KERNELS ---

@njit(fastmath=True)
def calculate_burst_score_vectorized(ll_matrix_sites_last, 
                                     gap_open_penalty=-10.0, 
                                     gap_extend_penalty=0.0, 
                                     uniform_log_prob=-1.1):
    """
    Calculates P(Data | Haplotype) using a 2-state HMM (Normal vs Burst) 
    along the sequence length to prevent geometric decay of likelihoods.
    
    States:
    0: Normal Match (Uses provided ll_matrix values)
    1: Error Burst (Uses uniform_log_prob)
    
    Transitions:
    Normal -> Normal: 0
    Normal -> Burst:  gap_open_penalty
    Burst  -> Burst:  gap_extend_penalty
    Burst  -> Normal: 0 (Free recovery)
    
    Args:
        ll_matrix_sites_last: (N_Samples, N_Haps, N_Haps, N_Sites) log-likelihoods.
        gap_open_penalty: Cost to START ignoring data (entering burst).
        gap_extend_penalty: Cost to CONTINUE ignoring data.
        uniform_log_prob: The score of a site inside a burst (ln(1/3) approx -1.1).
        
    Returns:
        (N_Samples, N_Haps, N_Haps) matrix of total log-likelihoods.
    """
    n_samples, n_h1, n_h2, n_sites = ll_matrix_sites_last.shape
    results = np.empty((n_samples, n_h1, n_h2), dtype=np.float64)
    
    # Pre-calc values for the Burst State
    # Inside a burst, the score is always: UniformLikelihood + TransitionCost
    burst_step_score = uniform_log_prob + gap_extend_penalty
    
    for s in range(n_samples):
        for h1 in range(n_h1):
            for h2 in range(n_h2):
                
                # State 0: Normal Mode
                # State 1: Burst Mode
                
                # Initialization (Site 0)
                # We assume we start in Normal mode. 
                # Starting in Burst immediately costs open_penalty.
                score_normal = ll_matrix_sites_last[s, h1, h2, 0]
                score_burst = gap_open_penalty + burst_step_score
                
                for i in range(1, n_sites):
                    emission = ll_matrix_sites_last[s, h1, h2, i]
                    
                    # 1. Update Normal State
                    # Transition Normal->Normal (Cost 0) OR Burst->Normal (Cost 0)
                    # We take max because we want the most likely path (Viterbi approx)
                    prev_best_for_normal = max(score_normal, score_burst) 
                    new_score_normal = prev_best_for_normal + emission
                    
                    # 2. Update Burst State
                    # Transition Normal->Burst (Open Cost) OR Burst->Burst (Extend Cost)
                    from_normal = score_normal + gap_open_penalty + burst_step_score
                    from_burst  = score_burst + burst_step_score
                    
                    new_score_burst = max(from_normal, from_burst)
                    
                    score_normal = new_score_normal
                    score_burst = new_score_burst
                    
                # Final score is the best of finishing in either state
                results[s, h1, h2] = max(score_normal, score_burst)
                
    return results

# %% --- CLASSES ---

class TransitionMesh:
    """
    A specialized container for transition probability meshes across different gap sizes.
    This structure allows efficient lookups of Forward and Backward transition 
    probabilities between genomic blocks separated by variable distances.

    Attributes:
        forward (dict): Maps gap_size (int) -> Forward Transition Dictionary.
                        Structure: { gap_size: { block_index: { ((curr_idx, curr_hap), (next_idx, next_hap)): prob } } }
        backward (dict): Maps gap_size (int) -> Backward Transition Dictionary.
                         Structure: { gap_size: { block_index: { ((curr_idx, curr_hap), (prev_idx, prev_hap)): prob } } }
    """
    def __init__(self, raw_gap_results=None):
        """
        Initializes the TransitionMesh.

        Args:
            raw_gap_results (dict, optional): A dictionary where keys are gap sizes and 
                                            values are [forward_dict, backward_dict] lists.
        """
        self.forward = {}
        self.backward = {}
        
        if raw_gap_results:
            for gap, probs_pair in raw_gap_results.items():
                self.forward[gap] = probs_pair[0]
                self.backward[gap] = probs_pair[1]

    def __getitem__(self, gap):
        """
        Retrieve the [Forward, Backward] transition dictionaries for a specific gap size.
        
        Args:
            gap (int): The distance (in number of blocks) between connected nodes.
            
        Returns:
            list: [forward_transition_dict, backward_transition_dict]
        """
        return [self.forward.get(gap), self.backward.get(gap)]
    
    def __contains__(self, gap):
        """Checks if a specific gap size has been computed in this mesh."""
        return gap in self.forward
    
    def keys(self):
        """Returns an iterator over the gap sizes available in the mesh."""
        return self.forward.keys()
    
    def items(self):
        """Yields (gap, [forward, backward]) tuples."""
        for gap in self.forward:
            yield gap, [self.forward[gap], self.backward[gap]]


class StandardBlockLikelihood:
    """
    Container for the likelihoods of ONE genomic block across ALL samples.
    Represents the emission probabilities P(Data | Genotype) for the HMM.
    
    Attributes:
        likelihood_tensor (np.ndarray): A tensor of shape (Num_Samples, Num_Haps, Num_Haps)
                                        containing log-likelihoods.
                                        Entry [s, i, j] is log(P(Sample_s | Hap_i, Hap_j)).
    """
    def __init__(self, likelihood_tensor):
        """
        Args:
            likelihood_tensor (np.ndarray): Tensor of log-likelihoods.
        """
        self.likelihood_tensor = likelihood_tensor
        
    def __len__(self):
        """Returns the number of samples."""
        return self.likelihood_tensor.shape[0]
    
    def __getitem__(self, sample_index):
        """
        Returns the (Num_Haps, Num_Haps) symmetric likelihood matrix for a specific sample.
        
        Args:
            sample_index (int): Index of the sample.
        """
        return self.likelihood_tensor[sample_index]
    
    def __repr__(self):
        return f"<StandardBlockLikelihood: {self.likelihood_tensor.shape[0]} samples, {self.likelihood_tensor.shape[1]} haps>"


class StandardBlockLikelihoods:
    """
    Container for the likelihoods of ALL genomic blocks in the dataset.
    This acts as the global 'Emission Probability' matrix for the downstream HMM.
    
    Attributes:
        blocks (list): A list of StandardBlockLikelihood objects, one per genomic block.
    """
    def __init__(self, blocks_list):
        """
        Args:
            blocks_list (list): A list of StandardBlockLikelihood objects.
        """
        # Validate input
        if blocks_list and not isinstance(blocks_list[0], StandardBlockLikelihood):
            self.blocks = [StandardBlockLikelihood(b) for b in blocks_list]
        else:
            self.blocks = blocks_list
            
    def __len__(self):
        """Returns the number of genomic blocks processed."""
        return len(self.blocks)
    
    def __getitem__(self, block_index):
        """Returns the StandardBlockLikelihood object for a specific block index."""
        return self.blocks[block_index]
    
    def __iter__(self):
        return iter(self.blocks)
    
    def __repr__(self):
        return f"<StandardBlockLikelihoods: covering {len(self.blocks)} blocks>"

# %% --- LIKELIHOOD GENERATION ---

def _worker_calculate_single_block_likelihood(args):
    """
    Internal worker function to calculate genotype likelihoods for a single block.
    
    It converts the per-site probabilistic genotypes (00, 01, 11) of the samples
    into diploid likelihoods for every pair of candidate haplotypes in the block.
    
    Uses Robust Categorical Likelihood (Mixture Model) to prevent over-penalization 
    of outliers at high read depth.
    
    Args:
        args (tuple): Contains:
            - samples_matrix (np.ndarray): (Samples x Sites x 3) probability matrix.
            - block_hap (BlockResult): The candidate haplotypes for this block.
            - params (dict): Configuration dictionary including robustness_epsilon.
    
    Returns:
        StandardBlockLikelihood: Object containing the unnormalized log-likelihood tensor.
    """
    samples_matrix, block_hap, params = args
    
    # Unpack params
    log_likelihood_base = params.get('log_likelihood_base', DEFAULT_LOG_BASE)
    epsilon = params.get('robustness_epsilon', DEFAULT_ROBUSTNESS_EPSILON)

    if len(samples_matrix) == 0:
        return StandardBlockLikelihood(np.array([]))

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
        return StandardBlockLikelihood(np.zeros((num_samples, 0, 0)))

    # --- 1. ROBUST TENSOR CREATION ---
    hap_list = [hap_dict[k] for k in hap_keys]
    
    if len(hap_list) > 0 and (hap_list[0].size == 0):
        haps_tensor = np.zeros((num_haps, 0, 2))
    else:
        haps_tensor = np.array(hap_list)
    
    # --- 2. MASKING ---
    samples_masked = samples_matrix[:, keep_flags, :] 
    haps_masked = haps_tensor[:, keep_flags, :]       
    
    num_active_sites = samples_masked.shape[1]
    
    if num_active_sites > 0:
        # --- 3. GENERATE DIPLOID COMBINATIONS ---
        h0 = haps_masked[:, :, 0]
        h1 = haps_masked[:, :, 1]
        
        # Calculate Genotype Probabilities implied by Haplotypes
        # Outer product to get all pairs (Full Grid)
        # Shape: (N_Haps, N_Haps, Sites)
        c00 = h0[:, None, :] * h0[None, :, :]
        c11 = h1[:, None, :] * h1[None, :, :]
        c01 = (h0[:, None, :] * h1[None, :, :]) + (h1[:, None, :] * h0[None, :, :])
        
        # Flatten haplotypes for broadcasting
        c00_flat = c00.reshape(-1, num_active_sites)
        c01_flat = c01.reshape(-1, num_active_sites)
        c11_flat = c11.reshape(-1, num_active_sites)
        
        # --- 4. PROBABILISTIC AGREEMENT (MIXTURE MODEL) ---
        
        # A. Calculate "Pure" Model Likelihood
        term_0 = samples_masked[:, np.newaxis, :, 0] * c00_flat[np.newaxis, :, :]
        term_1 = samples_masked[:, np.newaxis, :, 1] * c01_flat[np.newaxis, :, :]
        term_2 = samples_masked[:, np.newaxis, :, 2] * c11_flat[np.newaxis, :, :]
        
        model_probs = term_0 + term_1 + term_2
        
        # B. Apply Robust Mixture
        uniform_prob = 1.0 / 3.0
        final_probs = (model_probs * (1.0 - epsilon)) + (epsilon * uniform_prob)
        
        # --- 5. LOG LIKELIHOOD ---
        # Note: final_probs cannot be 0 if epsilon > 0, but we safety check anyway
        min_prob = 1e-300
        final_probs[final_probs < min_prob] = min_prob
        
        ll_per_site = np.log(final_probs)
        
        # --- APPLY BURST/AFFINE LOGIC UPGRADE ---
        # 1. Apply Hard Floor of -2.0 per site (prevents single-site overkill)
        ll_per_site = np.maximum(ll_per_site, -2.0)
        
        # 2. Reshape for Kernel: (Samples, Haps, Haps, Sites)
        # Original code flattened this implicitly via sum, we need explicit dimensions
        # model_probs shape logic: (N, 1, Sites) * (1, K_flat, Sites) -> (N, K_flat, Sites)
        # where K_flat is (Haps*Haps) flattened
        
        # So ll_per_site is currently (N_Samples, K_flat, Sites)
        ll_4d = ll_per_site.reshape(num_samples, num_haps, num_haps, num_active_sites)
        
        # 3. Burst Aware Summation
        # gap_open = -10.0 (High penalty to start burst)
        # gap_extend = 0.0 (No extra penalty beyond uniform)
        # uniform = -1.1 (ln(1/3))
        total_ll_matrix_4d = calculate_burst_score_vectorized(
            ll_4d, 
            gap_open_penalty=-10.0,
            gap_extend_penalty=0.0, 
            uniform_log_prob=math.log(1.0/3.0)
        )
        
        final_tensor = total_ll_matrix_4d
        
    else:
        final_tensor = np.zeros((num_samples, num_haps, num_haps))

    # --- 6. FORMAT OUTPUT ---
    # Reshape to (N_Samples, N_Haps, N_Haps)
    # The burst kernel returns exactly this shape, so just wrapping it.
    
    return StandardBlockLikelihood(final_tensor)

def generate_all_block_likelihoods(
    sample_probs_matrix,
    global_site_locations,
    haplotype_data,
    num_processes=16,
    log_likelihood_base=math.e,
    robustness_epsilon=DEFAULT_ROBUSTNESS_EPSILON):
    """
    Calculates diploid genotype log-likelihoods for all blocks against all samples.
    This generates the "Emission Matrix" for the HMM.
    
    Updated to support both contiguous blocks and sparse (proxy) blocks by using 
    exact index mapping rather than slicing.
    
    Args:
        sample_probs_matrix (np.ndarray): (N_Samples x Total_Sites x 3) probability matrix.
        global_site_locations (np.ndarray): Array of genomic positions.
        haplotype_data (list or BlockResults): List of BlockResult objects or a single object.
        num_processes (int): Number of parallel processes to use.
        log_likelihood_base (float): Base for the log calculation.
        robustness_epsilon (float): The mixture weight for the uniform error model.

    Returns:
        StandardBlockLikelihoods: A container with symmetric likelihood matrices for all blocks.
        Or a single StandardBlockLikelihood if a single block was passed.
    """
    
    is_single_block = False
    
    if hasattr(haplotype_data, 'positions') and hasattr(haplotype_data, 'haplotypes'):
        blocks_to_process = [haplotype_data]
        is_single_block = True
    else:
        blocks_to_process = haplotype_data
        
    tasks = []
    params = {
        'log_likelihood_base': log_likelihood_base,
        'robustness_epsilon': robustness_epsilon
    }
    
    for block in blocks_to_process:
        if not hasattr(block, 'positions'):
             raise ValueError(f"Encountered invalid block object in list. Type: {type(block)}")

        # ROBUST DATA FETCHING:
        # Instead of slicing (start:end) which fails for sparse/downsampled blocks,
        # we find the exact indices of the block's positions in the global array.
        indices = np.searchsorted(global_site_locations, block.positions)
        
        # Fancy indexing returns a copy containing only the requested sites.
        # This works correctly for contiguous blocks (0, 1, 2...) 
        # AND sparse proxies (0, 10, 20...), ensuring dimension alignment.
        block_samples = sample_probs_matrix[:, indices, :]
        
        tasks.append((block_samples, block, params))

    if num_processes > 1 and len(tasks) > 1:
        with Pool(num_processes) as pool:
            results = pool.map(_worker_calculate_single_block_likelihood, tasks)
    else:
        results = list(map(_worker_calculate_single_block_likelihood, tasks))

    if is_single_block:
        return results[0]
        
    return StandardBlockLikelihoods(results)
# %% --- EM HELPERS ---

def initial_transition_probabilities(haps_data, space_gap=1):
    """
    Creates a dictionary of initial transition probabilities assuming a Uniform Prior.
    Connects every haplotype in Block N to every haplotype in Block N + space_gap.
    
    Args:
        haps_data (list): List of BlockResult objects.
        space_gap (int): The distance (stride) between blocks to link.

    Returns:
        list: [forward_dict, backward_dict] containing uniform probabilities.
    """
    transition_dict_forward = {}
    transition_dict_reverse = {}
    
    # Forward Pass initialization
    for i in range(0,len(haps_data)-space_gap):
        transition_dict_forward[i] = {}
        
        these_haps = haps_data[i].haplotypes
        next_haps = haps_data[i+space_gap].haplotypes
        
        for first_idx in these_haps.keys():
            first_hap_name = (i,first_idx)
            for second_idx in next_haps.keys():
                second_hap_name = (i+space_gap,second_idx)
                transition_dict_forward[i][(first_hap_name,second_hap_name)] = 1
                
    # Backward Pass initialization
    for i in range(len(haps_data)-1,space_gap-1,-1):
        transition_dict_reverse[i] = {}
        
        these_haps = haps_data[i].haplotypes
        next_haps = haps_data[i-space_gap].haplotypes
        
        for first_idx in these_haps.keys():
            first_hap_name = (i,first_idx)
            for second_idx in next_haps.keys():
                second_hap_name = (i-space_gap,second_idx)
                transition_dict_reverse[i][(first_hap_name,second_hap_name)] = 1
    
    # Normalize Forward
    scaled_dict_forward = {}
    for idx in transition_dict_forward.keys():
        scaled_dict_forward[idx] = {}
        start_dict = {}
        for s in transition_dict_forward[idx].keys():
            start_dict[s[0]] = start_dict.get(s[0], 0) + transition_dict_forward[idx][s]
        
        for s in transition_dict_forward[idx].keys():
            scaled_dict_forward[idx][s] = transition_dict_forward[idx][s]/start_dict[s[0]]
        
    # Normalize Backward
    scaled_dict_reverse = {}
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
                           sample_block_likelihoods=None,
                           space_gap=1):
    """
    Calculates the Forward Variables (Alpha) for the HMM for a SINGLE sample.
    Computes P(State_t = i, Data_1:t) recursively using log-space matrix multiplication.
    Uses FULL directed state space (no symmetry collapsing).
    
    Args:
        sample_data (np.ndarray): (Sites x 3) probability array for one sample.
        sample_sites (np.ndarray): Site coordinates.
        haps_data (list): List of BlockResult objects.
        bidirectional_transition_probs (list): [forward_dict, backward_dict].
        sample_block_likelihoods (list, optional): Pre-computed emission probabilities for this sample.
        space_gap (int): The stride of the HMM chain.
        
    Returns:
        dict: likelihood_numbers mapping block_index -> { (HapA, HapB): log_prob }
    """
    
    if sample_block_likelihoods is None:
        full_res = generate_all_block_likelihoods(
            sample_data[np.newaxis, :, :], sample_sites, haps_data, num_processes=1
        )
        sample_block_likelihoods = [b[0] for b in full_res]

    transition_probs_dict = bidirectional_transition_probs[0]
    
    likelihood_numbers = {} 
    shadow_cache = {} 
    
    for i in range(len(haps_data)):
        
        # 1. Load Emission Probabilities (Matrix)
        E = sample_block_likelihoods[i] # (N_Haps, N_Haps)

        hap_keys = sorted(list(haps_data[i].haplotypes.keys()))
        n_haps = len(hap_keys)
        
        if i < space_gap:
            # Initialization Step: Just Emissions
            # In directed space, we don't need correction factors for hets.
            current_matrix = E
            
        else:
            # Recursion Step: Alpha_t = (Alpha_t-1 @ T) * E
            earlier_block = i - space_gap
            prev_matrix = shadow_cache[earlier_block]['matrix']
            prev_keys = shadow_cache[earlier_block]['keys']
            n_prev = len(prev_keys)
            
            # Construct Transition Matrix T (Sparse to Dense)
            T = np.full((n_prev, n_haps), -np.inf)
            t_probs_block = transition_probs_dict[earlier_block]
            
            for r, p_key in enumerate(prev_keys):
                for c, u_key in enumerate(hap_keys):
                    lookup = ((earlier_block, p_key), (i, u_key))
                    if lookup in t_probs_block:
                        T[r, c] = math.log(t_probs_block[lookup])
            
            # Z = Alpha_prev @ T
            Z = analysis_utils.log_matmul(prev_matrix, T)
            # Pred = T.T @ Z
            pred_matrix = analysis_utils.log_matmul(T.T, Z)
            
            # Combine with Emissions
            current_matrix = pred_matrix + E

        shadow_cache[i] = {'matrix': current_matrix, 'keys': hap_keys}
        
        # Output Results (Full Grid)
        result_dict = {}
        for r in range(n_haps):
            for c in range(n_haps): # Iterate full grid
                key = ((i, hap_keys[r]), (i, hap_keys[c]))
                result_dict[key] = current_matrix[r, c]

        likelihood_numbers[i] = result_dict
        
    return likelihood_numbers

def get_full_probs_backward(sample_data, sample_sites, haps_data,
                           bidirectional_transition_probs,
                           sample_block_likelihoods=None,
                           space_gap=1):
    """
    Calculates the Backward Variables (Beta) for the HMM for a SINGLE sample.
    Computes P(Data_t+1:T | State_t = i) recursively.
    Uses FULL directed state space.
    
    Args:
        sample_data (np.ndarray): (Sites x 3) probability array for one sample.
        sample_sites (np.ndarray): Site coordinates.
        haps_data (list): List of BlockResult objects.
        bidirectional_transition_probs (list): [forward_dict, backward_dict].
        sample_block_likelihoods (list, optional): Pre-computed emission probabilities.
        space_gap (int): The stride of the HMM chain.

    Returns:
        dict: likelihood_numbers mapping block_index -> { (HapA, HapB): log_prob }
    """
    
    if sample_block_likelihoods is None:
        full_res = generate_all_block_likelihoods(
            sample_data[np.newaxis, :, :], sample_sites, haps_data, num_processes=1
        )
        sample_block_likelihoods = [b[0] for b in full_res]

    transition_probs_dict = bidirectional_transition_probs[1]
    
    likelihood_numbers = {} 
    shadow_cache = {} 
    
    for i in range(len(haps_data)-1, -1, -1):
        
        # 1. Load Emission Probabilities (Matrix)
        E = sample_block_likelihoods[i]

        hap_keys = sorted(list(haps_data[i].haplotypes.keys()))
        n_haps = len(hap_keys)
        
        if i >= len(haps_data) - space_gap:
            # Initialization: Beta_T = 1 (log 0)
            current_matrix = E
            
        else:
            # Recursion: Beta_t = T @ (Beta_t+1 * E_t+1)
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
        
        # Output Results (Full Grid)
        result_dict = {}
        for r in range(n_haps):
            for c in range(n_haps): # Iterate full grid
                key = ((i, hap_keys[r]), (i, hap_keys[c]))
                result_dict[key] = current_matrix[r, c]

        likelihood_numbers[i] = result_dict

    return likelihood_numbers

# %% --- UNIFIED UPDATE FUNCTION ---

def get_updated_transition_probabilities_unified(
        full_samples_data,
        sample_sites,
        haps_data,
        current_transition_probs,
        full_blocks_likelihoods,
        space_gap=1,
        minimum_transition_log_likelihood=-10,
        BATCH_SIZE=100,
        use_standard_baum_welch=True): 
    """
    Performs the Expectation-Maximization (EM) update step (Baum-Welch).

    1. E-Step: Runs Forward and Backward algorithms for all samples to compute 
       the probability of being in state (u,v) at time t given the data.
    2. M-Step: Updates the transition probabilities T_ij to maximize the likelihood.
       Utilizes vectorized batch processing to handle the summation over samples efficiently.
       Includes Robust M-Step logic to handle diploid phase ambiguity.
    
    Args:
        full_samples_data (list): List of sample data arrays.
        sample_sites (np.ndarray): Genomic locations.
        haps_data (list): List of BlockResult objects.
        current_transition_probs (list): Current estimates [fwd, bwd].
        full_blocks_likelihoods (StandardBlockLikelihoods): Pre-computed emissions.
        space_gap (int): HMM stride.
        minimum_transition_log_likelihood (float): Floor for probabilities.
        BATCH_SIZE (int): Number of samples to process in a vectorized chunk.
        use_standard_baum_welch (bool): If True, applies standard HMM logic.

    Returns:
        tuple: ([new_fwd, new_bwd], total_data_log_likelihood)
    """

    prior_a_posteriori = initial_transition_probabilities(haps_data, space_gap=space_gap)

    full_samples_likelihoods = full_samples_data
    num_samples = len(full_samples_likelihoods)
    num_blocks = len(full_blocks_likelihoods)
    
    # Restructure: blocks[sample] -> sample[block] (matrix based)
    all_block_likelihoods_by_sample = []
    for s in range(num_samples):
        sample_chain = []
        for b in range(num_blocks):
            sample_chain.append(full_blocks_likelihoods[b][s])
        all_block_likelihoods_by_sample.append(sample_chain)
             
    # 1. E-Step: Forward Pass
    forward_nums = [get_full_probs_forward(
                        full_samples_data[i],
                        sample_sites, haps_data, current_transition_probs, 
                        all_block_likelihoods_by_sample[i], space_gap=space_gap
                    ) for i in range(num_samples)]
    

    # Calculate Total Data Log Likelihood (summing full grid)
    total_data_log_likelihood = 0.0
    last_block_idx = len(haps_data) - 1
    
    for s in range(num_samples):
        final_states_log_probs = list(forward_nums[s][last_block_idx].values())
        if final_states_log_probs:
            total_data_log_likelihood += logsumexp(final_states_log_probs)

    # 1. E-Step: Backward Pass
    backward_nums = [get_full_probs_backward(
                        full_samples_data[i],
                        sample_sites, haps_data, current_transition_probs, 
                        all_block_likelihoods_by_sample[i], space_gap=space_gap
                    ) for i in range(num_samples)]
    
    
    samples_probs = list(zip(forward_nums, backward_nums))
    
    
    # 2. M-Step: Vectorized Update (Baum-Welch ξ calculation)
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

            # Load priors
            for u_idx, u in enumerate(hap_keys_current):
                for v_idx, v in enumerate(hap_keys_next):
                    trans_key = ((i, u), (next_bundle, v))
                    if trans_key in current_transition_probs[dir_idx][i]:
                        T_matrix[u_idx, v_idx] = math.log(current_transition_probs[dir_idx][i][trans_key])
                    if trans_key in prior_a_posteriori[dir_idx][i]:
                        P_matrix[u_idx, v_idx] = math.log(prior_a_posteriori[dir_idx][i][trans_key])

            # Accumulate Forward/Backward probabilities across samples
            F_tensor = np.full((num_samples, n_curr, n_curr), -np.inf)
            B_tensor = np.full((num_samples, n_next, n_next), -np.inf)
                
            for s in range(num_samples):
                if is_forward:
                    fwd_dict = samples_probs[s][0][i]           
                    bwd_dict = samples_probs[s][1][next_bundle] 
                else:
                    fwd_dict = samples_probs[s][1][i]           
                    bwd_dict = samples_probs[s][0][next_bundle] 

                for u_out_idx, u_out in enumerate(hap_keys_current):
                    for u_in_idx, u_in in enumerate(hap_keys_current):
                        key = ((i, u_out), (i, u_in))
                        if key in fwd_dict: 
                            F_tensor[s, u_out_idx, u_in_idx] = fwd_dict[key]

                for v_out_idx, v_out in enumerate(hap_keys_next):
                    for v_in_idx, v_in in enumerate(hap_keys_next):
                        key = ((next_bundle, v_out), (next_bundle, v_in))
                        if key in bwd_dict: 
                            B_tensor[s, v_out_idx, v_in_idx] = bwd_dict[key]
                
            batch_results = []
            
            # Process batches
            for start_idx in range(0, num_samples, BATCH_SIZE):
                end_idx = min(start_idx + BATCH_SIZE, num_samples)
                
                F_batch = F_tensor[start_idx:end_idx]
                B_batch = B_tensor[start_idx:end_idx]
                
                # Broadcasting: (Batch, u_out, u_in, v_out, v_in)
                F_broad = F_batch[:, :, :, np.newaxis, np.newaxis]
                B_broad = B_batch[:, np.newaxis, np.newaxis, :, :]
                
                # T_partner_broad (u_in -> v_in) aligns with indices (2, 4)
                T_partner_broad = T_matrix[np.newaxis, np.newaxis, :, np.newaxis, :]
                
                combined = F_broad + B_broad + T_partner_broad
                                
                if use_standard_baum_welch:
                    # T_main_broad (u_out -> v_out) aligns with indices (1, 3)
                    T_main_broad = T_matrix[np.newaxis, :, np.newaxis, :, np.newaxis]
                    combined += T_main_broad
                
                # --- CORRECTED SUMMATION ---
                # Combined Indices: 0=Batch, 1=u_out, 2=u_in, 3=v_out, 4=v_in
                
                # 1. Slot 1 -> Slot 1 (u_out -> v_out): Sum over (2, 4)
                # This captures the probability mass for the primary chromosome transition.
                mass_1_1 = logsumexp(combined, axis=(2, 4))
                
                # 2. Slot 2 -> Slot 2 (u_in -> v_in): Sum over (1, 3)
                # This captures the probability mass for the partner chromosome transition.
                mass_2_2 = logsumexp(combined, axis=(1, 3))
                
                # FIX: Removed mass_1_2 and mass_2_1 (Cross-edges). 
                # Swapped transitions are already covered by mass_1_1 at swapped indices.

                # Stack only the direct evidences
                stacked_evidence = np.stack([mass_1_1, mass_2_2])
                sample_lik_batch = logsumexp(stacked_evidence, axis=0)
                
                # Normalize per sample
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
            min_cutoff_change=0.001,
            ll_improvement_cutoff=1e-4,
            learning_rate=1.0, 
            minimum_transition_log_likelihood=-10,
            use_standard_baum_welch=True):
    """
    Main loop for calculating transition probabilities between blocks using EM.
    Iteratively refines the transition matrix until the likelihood converges.
    
    Args:
        full_samples_data (list): Sample data arrays.
        sample_sites (np.ndarray): Genomic locations.
        haps_data (list): BlockResult objects.
        full_blocks_likelihoods (StandardBlockLikelihoods, optional): Pre-computed emissions.
        max_num_iterations (int): Maximum EM steps.
        space_gap (int): HMM stride.
        min_cutoff_change (float): (Unused) Threshold param.
        ll_improvement_cutoff (float): Convergence threshold for Log Likelihood.
        learning_rate (float): Smoothing factor for updates.

    Returns:
        list: [final_forward_transitions, final_backward_transitions]
    """
    
    start_probs = initial_transition_probabilities(haps_data, space_gap=space_gap,
                        )
    
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
            space_gap=space_gap,
            minimum_transition_log_likelihood=minimum_transition_log_likelihood,
            BATCH_SIZE=100,
            use_standard_baum_welch=use_standard_baum_welch
        )
        
        current_probs_smoothed = analysis_utils.smoothen_probs_vectorized(current_probs, new_probs_raw, effective_lr)
        
        if isinstance(current_probs_smoothed, dict):
            current_probs_new = [current_probs_smoothed[0], current_probs_smoothed[1]]
        else:
            current_probs_new = current_probs_smoothed

        current_probs = current_probs_new
        
        # Relative improvement check
        rel_improvement = 0.0
        if prev_ll != -np.inf and prev_ll != 0:
            rel_improvement = (current_ll - prev_ll) / abs(prev_ll)
        elif prev_ll == -np.inf:
            rel_improvement = float('inf') 
            
        if i > 0 and 0 <= rel_improvement < ll_improvement_cutoff:
            break
            
        prev_ll = current_ll
            
    return current_probs

# %% --- WORKER WRAPPER FOR POOL ---
def _gap_worker(args):
    """
    Unpacks arguments and calls the calculation function for multiprocessing.
    """
    # Unpack the new flag from the tuple
    (gap, full_samples, sites, haps, likes, max_iter, min_ll, lr, use_std_bw) = args
    
    return calculate_hap_transition_probabilities(
        full_samples, 
        sites, 
        haps, 
        full_blocks_likelihoods=likes,
        max_num_iterations=max_iter,
        space_gap=gap,
        minimum_transition_log_likelihood=min_ll,
        learning_rate=lr,
        use_standard_baum_welch=use_std_bw  # Pass flag to calculator
    )

def generate_transition_probability_mesh(full_samples_data,
                                         sample_sites,
                                         haps_data,
                                         max_num_iterations=10,
                                         minimum_transition_log_likelihood=-10,
                                         learning_rate=1,
                                         use_standard_baum_welch=True):
    """
    Generates a TransitionMesh by calculating transition probabilities 
    for ALL possible gap sizes (1 to N) in parallel.
    
    This creates a multi-scale view of the haplotype graph, allowing 
    downstream algorithms (like Beam Search) to skip over noisy blocks.
    
    Args:
        full_samples_data (list): Sample data.
        sample_sites (np.ndarray): Genomic locations.
        haps_data (list): List of BlockResult objects.
        max_num_iterations (int): EM iterations per gap size.
        use_standard_baum_welch (bool): 
            If True: Uses standard update (sensitive to initialization/priors).
            If False: Uses Reset update (recommended for Viterbi/Hard EM).
        
    Returns:
        TransitionMesh: The fully populated mesh of transition probabilities.
    """
    
    print("Calculating block likelihoods (Mesh Generation)...")
    full_blocks_likelihoods = generate_all_block_likelihoods(
        full_samples_data, sample_sites, haps_data
    )
    
    max_gap = len(haps_data) - 1
    gaps = list(range(1, max_gap + 1))
    
    print(f"Calculating transitions for {len(gaps)} different gaps (StandardBW={use_standard_baum_welch})...")

    worker_args = []
    for gap in gaps:
        worker_args.append((
            gap,
            full_samples_data,
            sample_sites,
            haps_data,
            full_blocks_likelihoods,
            max_num_iterations,
            minimum_transition_log_likelihood,
            learning_rate,
            use_standard_baum_welch # Append flag to args tuple
        ))

    with Pool(16) as pool:
        results = pool.map(_gap_worker, worker_args)
    
    mesh_dict = dict(zip(gaps, results))
    return TransitionMesh(mesh_dict)