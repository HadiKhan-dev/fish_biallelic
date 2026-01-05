import numpy as np
import math
from multiprocess import Pool
from scipy.special import softmax, gammaln, logsumexp
import warnings

# Try to import Numba, provide dummy decorators if missing
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Use warnings module instead of print for library-level warnings
    warnings.warn("Numba not found. Falling back to slow Python mode.", ImportWarning)
    
    # Dummy decorator that accepts arguments (like fastmath=True) but does nothing
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

warnings.filterwarnings("ignore")
np.seterr(divide='ignore', invalid="ignore")

#%% --- MATH HELPERS ---

def log_fac(x):
    """
    Returns log(x!) using the gamma function.
    Works for scalars and numpy arrays.
    """
    return gammaln(np.asarray(x) + 1)

def log_binomial(n, k):
    """
    Returns log(nCk).
    Works for scalars and numpy arrays.
    """
    return log_fac(n) - log_fac(k) - log_fac(n - k)

def make_upper_triangular(matrix):
    """
    Make a matrix upper triangular by adding the value at 
    index (j,i) with i < j to the value at index (i,j)
    """
    return np.triu(matrix + matrix.T - np.diag(matrix.diagonal()))

def log_matmul(A, B):
    """
    Performs Matrix Multiplication in Log-Space.
    Mathematically equivalent to C = A @ B but entirely in the log domain.
    
    Formula: C_ij = logsumexp_k(A_ik + B_kj)
    
    Args:
        A: Tensor of shape (..., M, K) or (M, K)
        B: Tensor of shape (..., K, N) or (K, N)
    
    Returns:
        Tensor of shape (..., M, N) resulting from log-space multiplication.
        Supports broadcasting for batches.
    """
    # Expand dims to broadcast the summation dimension K
    return logsumexp(A[..., np.newaxis] + B[..., np.newaxis, :, :], axis=-2)

#%% --- READS TO PROBABILITIES ---

def reads_to_probabilities(reads_array, read_error_prob=0.02, min_total_reads=5):
    """
    Convert a reads array to a probability of the underlying
    genotype being 0, 1 or 2.
    
    Returns:
        (site_priors, posterior_probs)
    """
    num_samples, num_sites, _ = reads_array.shape
    
    # 0. Handle empty case
    if num_sites == 0:
        return (np.empty((num_sites, 3)), np.empty((num_samples, num_sites, 3)))

    # --- PART 1: Calculate Site Priors ---
    
    # Sum reads across samples
    reads_sum = np.sum(reads_array, axis=0)
    total_reads_per_site = np.sum(reads_sum, axis=1)
    
    # Create mask for valid sites
    threshold = max(min_total_reads, read_error_prob * num_samples)
    valid_mask = total_reads_per_site >= threshold
    
    # Calculate ratios
    numerator = 1 + reads_sum[:, 1]
    denominator = 2 + total_reads_per_site
    calculated_ratios = numerator / denominator
    
    # Apply condition: if invalid, use error prob
    singleton = np.where(valid_mask, calculated_ratios, read_error_prob)
    
    # Calculate priors (00, 01, 11) assuming HWE approximation
    priors_00 = (1 - singleton) ** 2
    priors_01 = 2 * singleton * (1 - singleton)
    priors_11 = singleton ** 2
    
    site_priors = np.stack([priors_00, priors_01, priors_11], axis=1)
    
    # --- PART 2: Calculate Likelihoods ---
    
    log_half = math.log(0.5)
    log_read_error = math.log(read_error_prob)
    log_read_nonerror = math.log(1 - read_error_prob)
    
    zeros = reads_array[..., 0] 
    ones = reads_array[..., 1]
    total = zeros + ones
    
    # Compute log binomials
    lb_total_ones = log_binomial(total, ones)
    lb_total_zeros = log_binomial(total, zeros)
    
    # Calculate Genotype Likelihoods
    # 00: Homozygous Ref
    ll_00 = lb_total_ones + zeros * log_read_nonerror + ones * log_read_error
    # 11: Homozygous Alt
    ll_11 = lb_total_zeros + zeros * log_read_error + ones * log_read_nonerror
    # 01: Heterozygous
    ll_01 = lb_total_ones + total * log_half
    
    log_likli_matrix = np.stack([ll_00, ll_01, ll_11], axis=-1)
    
    # --- PART 3: Combine and Normalize ---
    
    # Broadcast priors (1, Sites, 3) against Likelihoods (Samples, Sites, 3)
    log_site_priors = np.log(site_priors)
    nonnorm_log_posterior = log_likli_matrix + log_site_priors[np.newaxis, :, :]
    
    # Softmax normalizes in log space
    posterior_probs = softmax(nonnorm_log_posterior, axis=-1)
    
    return (site_priors, posterior_probs)

#%% --- DISTANCE METRICS (OPTIMIZED) ---

def calc_distance(first_row, second_row, calc_type="diploid"):
    """
    Calculate the probabilistic distance between two rows (einsum based).
    Used for single comparisons.
    """
    if calc_type == "diploid":
        distances = np.array([[0,1,2],[1,0,1],[2,1,0]], dtype=float)
    else:
        distances = np.array([[0,1],[1,0]], dtype=float)
        
    ens = np.einsum("ij,ik->ijk", first_row, second_row)
    ensd = ens * distances
    
    return np.sum(ensd, axis=None)

def calc_distance_by_site(first_row, second_row, calc_type="diploid"):
    """
    Like calc_distance but returns distance per site.
    """
    if calc_type == "diploid":
        distances = np.array([[0,1,2],[1,0,1],[2,1,0]], dtype=float)
    else:
        distances = np.array([[0,1],[1,0]], dtype=float)
        
    ens = np.einsum("ij,ik->ijk", first_row, second_row)
    ensd = ens * distances
    
    return np.sum(ensd, axis=(1,2))
    
def calc_perc_difference(first_row, second_row, calc_type="diploid"):
    """
    Calculate the probabilistic percentage difference between two rows.
    """
    row_len = len(first_row)
    if row_len == 0: return 0.0
    return 100 * calc_distance(first_row, second_row, calc_type=calc_type) / row_len

def calc_distance_row(row, data_matrix, start_point, calc_type="diploid"):
    """
    Legacy helper for calculating row distance. 
    Kept for compatibility, but generate_distance_matrix is preferred.
    """
    num_samples = len(data_matrix)
    row_vals = [0] * start_point
    
    for i in range(start_point, num_samples):
        row_vals.append(calc_distance(row, data_matrix[i], calc_type=calc_type))
    
    return row_vals
    
def generate_distance_matrix(probs_array,
                             keep_flags=None,
                             calc_type="diploid"): 
    """
    Generates a pairwise distance matrix for the array using vectorized algebra.
    Note: 'num_processes' argument was removed as numpy matrix multiplication 
    is inherently optimized and parallelized.
    """
    num_samples, num_sites, feat_dim = probs_array.shape
    
    if keep_flags is None:
        keep_flags = slice(None)
    else:
        keep_flags = np.array(keep_flags, dtype=bool)
    
    # 1. Filter sites
    # Shape: (N, Active_Sites, 3)
    active_probs = probs_array[:, keep_flags, :]
    
    # 2. Setup Weights
    if calc_type == "diploid":
        # 0 vs 0=0, 0 vs 2=2
        weights = np.array([[0, 1, 2],
                            [1, 0, 1],
                            [2, 1, 0]], dtype=np.float32)
    else:
        # Haploid
        weights = np.array([[0, 1],
                            [1, 0]], dtype=np.float32)

    # 3. Transform Data for Matmul
    # Formula: Dist(u, v) = sum_sites( u_s @ W @ v_s.T )
    # We compute (X @ W) globally first.
    # Shape: (N, Active_Sites, 3)
    transformed_probs = active_probs @ weights
    
    # 4. Flatten for large dot product
    # We want to sum over sites, so we combine (Sites * Feats)
    flat_original = active_probs.reshape(num_samples, -1)
    flat_transformed = transformed_probs.reshape(num_samples, -1)
    
    # 5. Compute all pairwise distances
    # (N, Features) @ (N, Features).T -> (N, N)
    dist_matrix = flat_original @ flat_transformed.T
    
    return dist_matrix

#%% --- GENOTYPE HELPERS ---

def greatest_likelihood_hap(hap):
    """
    Convert probabilistic hap to deterministic (argmax).
    """
    return np.argmax(hap, axis=1)

def get_heterozygosity(probabalistic_genotype, keep_flags=None):
    """
    Calculate the heterozygosity of a genotype.
    """
    if keep_flags is None:
        keep_flags = slice(None)
    else:
        keep_flags = np.array(keep_flags, dtype=bool)
    
    # Sum probability of being 1 (Het)
    het_probs = probabalistic_genotype[keep_flags, 1]
    
    if het_probs.size == 0: return 0.0
    
    num_hetero = np.sum(het_probs)
    return 100 * num_hetero / het_probs.size

def size_l1(vec):
    return np.nansum(np.abs(vec))

def magnitude_percentage(vec):
    valid = np.count_nonzero(~np.isnan(vec))
    if valid == 0: return 0
    return 100 * size_l1(vec) / (2 * valid)

def perc_wrong(haplotype, keep_flags=None):
    """
    Percentage of sites that are not [0,1] or [1,0] (fuzzy).
    Checks if values are outside [0,1] bounds.
    """
    if keep_flags is None:
        keep_flags = slice(None)
    else:
        keep_flags = np.array(keep_flags, dtype=bool)
        
    hap_use = haplotype[keep_flags]
    if len(hap_use) == 0: return 0.0

    num_wrong = np.count_nonzero(np.logical_or(hap_use < 0, hap_use > 1))
    return 100 * num_wrong / len(hap_use)

def fix_hap(haplotype):
    """
    Clamp values between 0.0 and 1.0.
    """
    return np.clip(haplotype, 0.0, 1.0)

def combine_haploids(hap_one, hap_two):
    """
    Combine two haploids (Probabilistic) to get a diploid.
    """
    # Vectorized outer product per row
    # (N, 2) -> (N, 2, 1) * (N, 1, 2) -> (N, 2, 2)
    ens = hap_one[:, :, None] * hap_two[:, None, :]

    vals_00 = ens[:,0,0]
    vals_01 = ens[:,0,1] + ens[:,1,0]
    vals_11 = ens[:,1,1]
    
    together = np.column_stack((vals_00, vals_01, vals_11))
    return np.ascontiguousarray(together)

def get_diff_wrongness(diploid, haploid, keep_flags=None):
    """
    Get the probabilistic difference created by subtracting a haploid from a diploid.
    """
    if keep_flags is None:
        keep_flags = slice(None)
    else:
        keep_flags = np.array(keep_flags, dtype=bool)
    
    # Broadcast subtraction logic
    # Dip has 0, 1, 2. Hap has 0, 1.
    
    # We calculate the full tensor: (Sites, Hap_Alleles, Dip_Alleles)
    ens = haploid[:, :, None] * diploid[:, None, :]
    
    # Rem 0: (H0 & D0) + (H1 & D1)
    zeros = ens[:,0,0] + ens[:,1,1]
    
    # Rem 1: (H0 & D1) + (H1 & D2)
    ones = ens[:,0,1] + ens[:,1,2]
    
    # Wrong: (H0 & D2) + (H1 & D0)
    wrong = ens[:,0,2] + ens[:,1,0]
    
    wrong_masked = wrong[keep_flags]
    perc_wrong = 100 * np.sum(wrong_masked) / len(wrong_masked) if len(wrong_masked) > 0 else 0
    
    difference = np.column_stack((zeros, ones))
    return (np.ascontiguousarray(difference), perc_wrong)

#%% --- INFORMATION THEORY UTILS (NUMBA OPTIMIZED) ---

@njit(fastmath=True)
def probability_to_information(probs_list):
    """
    Signed information metric for a [1-p, p] pair.
    """
    p = probs_list[1]
    
    if p >= 1.0: return 1.0
    if p <= 0.0: return -1.0
    
    sgn = -1.0 if p < 0.5 else 1.0
    entropy = -(1-p)*math.log2(1-p) - p*math.log2(p)
    
    return sgn * (1.0 - entropy)

@njit(fastmath=True)
def add_informations(first_information, second_information):
    """
    Relativistic addition of information.
    """
    if abs(first_information * second_information + 1.0) < 1e-9:
        return 0.0
        
    return (first_information + second_information) / (1.0 + first_information * second_information)

@njit(fastmath=True)
def combine_probabilities(first_prob, second_prob, prior_prob, required_accuracy=1e-13):
    """
    Combine observed probabilities with a prior using binary search inversion.
    JIT-compiled for speed.
    """ 
    first_information = probability_to_information(first_prob)
    second_information = probability_to_information(second_prob)
    prior_information = probability_to_information(prior_prob)
    
    first_relative = add_informations(first_information, -prior_information)
    second_relative = add_informations(second_information, -prior_information)
    
    combined_relative = add_informations(first_relative, second_relative)
    full_combined = add_informations(combined_relative, prior_information)
    
    # Binary search inversion
    low = 0.0
    high = 1.0
    
    for _ in range(60):
        midpoint = (low + high) * 0.5
        test_prob = np.array([1.0 - midpoint, midpoint])
        test_val = probability_to_information(test_prob)
        
        if test_val < full_combined:
            low = midpoint
        else:
            high = midpoint
    
    final_prob = (low + high) * 0.5
    return np.array([1.0 - final_prob, final_prob])

def get_dips_from_long_haps(long_haps, num_processes=16):
    """
    Create each possible combination of long diploids from a set of long haplotypes.
    """
    def extract_and_combine(full_haps, first_index, second_index):
        return combine_haploids(full_haps[first_index], full_haps[second_index])
    
    num_haps = len(long_haps)
    all_combs = [(i,j) for i in range(num_haps) for j in range(num_haps)]
    
    with Pool(processes=num_processes) as pool:
        all_combined = pool.starmap(lambda x,y: extract_and_combine(long_haps, x, y), all_combs)
    
    # Reshape
    total_combined = []
    for i in range(num_haps):
        subset = all_combined[num_haps*i : num_haps*(i+1)]
        total_combined.append(subset)
    
    return np.array(total_combined)

#%% --- ANALYSIS & PATHING ---

def smoothen_probs_vectorized(old_probs_list, new_probs_list, learning_rate):
    """
    Safe version of smoothing that aligns keys explicitly.
    """
    if learning_rate >= 1.0: return new_probs_list
    
    smoothed_result = []
    for dir_idx in [0, 1]: # Forward and Backward
        dir_dict = {}
        for block_idx in new_probs_list[dir_idx]:
            old_block = old_probs_list[dir_idx][block_idx]
            new_block = new_probs_list[dir_idx][block_idx]
            
            smoothed_block = {}
            # Use new_block keys as the master list
            for key in new_block:
                new_val = new_block[key]
                old_val = old_block.get(key, new_val) # Fallback to new if missing
                smoothed_block[key] = (old_val * (1.0 - learning_rate)) + (new_val * learning_rate)
            dir_dict[block_idx] = smoothed_block
        smoothed_result.append(dir_dict)
    return smoothed_result

def map_haplotype_to_blocks(target_haplotype, haps_data):
    """
    Maps a continuous long haplotype to block indices.
    """
    best_path_indices = []
    current_site_idx = 0
    total_target_len = len(target_haplotype)
    
    for i, block in enumerate(haps_data):
        # Handle BlockResult object or tuple
        if hasattr(block, 'haplotypes'):
            block_haps_dict = block.haplotypes
        else:
            block_haps_dict = block[3]
        
        if not block_haps_dict:
            best_path_indices.append(None)
            continue
            
        any_key = next(iter(block_haps_dict))
        block_len = len(block_haps_dict[any_key])
        
        if current_site_idx + block_len > total_target_len:
            print(f"Warning: Target haplotype shorter than data. Stopped at block {i}.")
            break
            
        target_slice = target_haplotype[current_site_idx : current_site_idx + block_len]
        
        min_dist = float('inf')
        best_key = None
        
        for key, candidate_hap in block_haps_dict.items():
            dist = np.sum(np.abs(target_slice - candidate_hap))
            if dist < min_dist:
                min_dist = dist
                best_key = key
        
        best_path_indices.append(best_key)
        current_site_idx += block_len
        
    return best_path_indices

def get_path_log_likelihood(path_indices, full_mesh, verbose=False):
    """
    Calculates total log-likelihood of a sequence of block haplotypes
    based on transition probabilities in full_mesh.
    """
    total_log_likelihood = 0.0
    gap = 1
    
    if gap not in full_mesh:
        raise ValueError("Full mesh does not contain gap size 1 transitions.")
    
    for i in range(len(path_indices) - 1):
        curr_block_idx = i
        next_block_idx = i + 1
        
        curr_hap_val = path_indices[curr_block_idx]
        next_hap_val = path_indices[next_block_idx]
        
        transition_key = ((curr_block_idx, curr_hap_val), 
                          (next_block_idx, next_hap_val))
        
        try:
            prob_dict = full_mesh[gap][0][curr_block_idx]
            
            if transition_key in prob_dict:
                prob = prob_dict[transition_key]
                if prob > 0:
                    total_log_likelihood += math.log(prob)
                else:
                    if verbose: print(f"Path broken at Block {i}->{i+1}: Probability is 0.0")
                    return float('-inf')
            else:
                if verbose: print(f"Path broken at Block {i}->{i+1}: Transition not found.")
                return float('-inf')
        except KeyError:
            if verbose: print(f"Block {i} missing from mesh.")
            return float('-inf')

    return total_log_likelihood

def get_path_score_beam_view(path_indices, full_mesh, verbose=False):
    """
    Calculates the 'Score' of a path exactly as the Beam Search calculates it.
    """
    total_score = 0.0
    num_blocks = len(path_indices)
    
    for i in range(num_blocks):
        for gap in full_mesh.keys():
            earlier_index = i - gap
            if earlier_index >= 0:
                curr_val = path_indices[i]
                prev_val = path_indices[earlier_index]
                
                transition_key = ((earlier_index, prev_val), (i, curr_val))
                try:
                    prob_dict = full_mesh[gap][0][earlier_index]
                    if transition_key in prob_dict:
                        prob = prob_dict[transition_key]
                        if prob > 0:
                            log_prob = math.log(prob)
                            total_score += log_prob * math.sqrt(1/gap)
                        else:
                            return float('-inf')
                    else:
                        return float('-inf')
                except KeyError:
                    pass
    return total_score

def recombination_fudge(start_probs, distance, recomb_rate=10**-8):
    """
    Updates genotype copying probabilities based on recombination distance.
    This creates a transition matrix adjustment:
    New = Old*(prob_stay) + (Marginal_Row + Marginal_Col)*(prob_switch)
    """
    num_rows, num_cols = start_probs.shape
    num_possible_switches = num_rows + num_cols - 1
    
    non_switch_prob = (1 - recomb_rate)**distance
    each_switch_prob = (1 - non_switch_prob) / num_possible_switches
    total_non_switch_prob = non_switch_prob + each_switch_prob
    
    row_mass_sources = np.sum(start_probs, axis=1) 
    col_mass_sources = np.sum(start_probs, axis=0) 
    
    term1 = start_probs * (total_non_switch_prob - 2 * each_switch_prob)
    term2 = each_switch_prob * (row_mass_sources[:, np.newaxis] + col_mass_sources[np.newaxis, :])
    
    return term1 + term2

def get_sample_data_at_sites(sample_data, sample_sites, query_sites):
    """
    Extract subset of sample data for specific sites.
    """
    if len(query_sites) == 0:
        return np.array([])

    # Assume sorted sites
    start_idx = np.searchsorted(sample_sites, query_sites[0])
    end_idx = np.searchsorted(sample_sites, query_sites[-1])
    
    return sample_data[start_idx:end_idx+1, :]
                       
def get_sample_data_at_sites_multiple(sample_data, sample_sites, query_sites):
    """
    Vectorized extraction for multiple samples (Batch, Sites, ...).
    """
    if len(query_sites) == 0:
        return sample_data[:, :0, :]
    
    start_idx = np.searchsorted(sample_sites, query_sites[0])
    end_idx = np.searchsorted(sample_sites, query_sites[-1])
    
    return sample_data[:, start_idx:end_idx+1, :]

def get_best_block_haps_for_long_hap(long_hap, hap_sites, block_haps):
    """
    Returns the best matching local haplotype ID for a given long haplotype 
    at each block.
    """
    best_haps = []
    best_diffs = []
    
    for i, block in enumerate(block_haps):
        # Handle BlockResult or legacy tuple
        if hasattr(block, 'positions'):
            block_sites = block.positions
            block_haps_dict = block.haplotypes
        else:
            block_sites = block[0]
            block_haps_dict = block[3]
            
        block_data = get_sample_data_at_sites(long_hap, hap_sites, block_sites)
        
        # Dictionary comprehension to find diff for each candidate
        haps_diffs = {
            hap_id: calc_perc_difference(block_data, hap_arr, calc_type="haploid") 
            for hap_id, hap_arr in block_haps_dict.items()
        }
        
        best_hap = min(haps_diffs, key=haps_diffs.get)
        best_diff = haps_diffs[best_hap]
        
        best_haps.append(best_hap)
        best_diffs.append(best_diff)
    
    return (best_haps, best_diffs)