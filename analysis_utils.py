import numpy as np
import math
from multiprocess import Pool
from scipy.special import softmax, gammaln, logsumexp
import warnings

warnings.filterwarnings("ignore")
np.seterr(divide='ignore',invalid="ignore")
pass

#%%
def log_fac(x):
    """
    Returns log(x!)
    """
    x = int(x)
    
    return math.lgamma(x+1)
    
def log_binomial(n,k):
    """
    Returns log(nCk)
    """
    
    return log_fac(n)-log_fac(k)-log_fac(n-k)

def log_fac_vectorized(x):
    """
    Vectorized version of log(x!).
    scipy.special.gammaln is the array-equivalent of math.lgamma
    """
    
    return gammaln(x + 1)

def log_binomial_vectorized(n, k):
    """
    Calculates log(nCk) for entire arrays at once.
    """
    
    return log_fac_vectorized(n) - log_fac_vectorized(k) - log_fac_vectorized(n - k)


def make_upper_triangular(matrix):
    """
    Make a matrix upper triangular by adding the value at 
    index (j,i) with i < j to the value at index (i,j)
    
    matrix must be a numpy array
    """
    
    return np.triu(matrix+matrix.T-np.diag(matrix.diagonal()))

def log_matmul(A, B):
    """
    Performs Matrix Multiplication in Log-Space.
    Equivalent to C = A @ B, but for log-probabilities.
    C[i, j] = logsumexp(A[i, :] + B[:, j])
    """
    # A: (N, K) -> (N, K, 1)
    # B: (K, M) -> (1, K, M)
    # Broadcasting sum: (N, K, M)
    # Logsumexp over K (axis 1)
    return logsumexp(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)


def reads_to_probabilities(reads_array, read_error_prob=0.02, min_total_reads=5):
    """
    Convert a reads array to a probability of the underlying
    genotype being 0, 1 or 2
    
    min_total_reads is a minimum number of reads each site must have for it to be considered
    a valid alternate site (this is to reduce the chance of us considering something which is a variant site only because of errors as a real site)
    
    Fully vectorized conversion of reads to genotype probabilities.
    """
    num_samples, num_sites, _ = reads_array.shape
    
    # 0. Handle empty case
    if num_sites == 0:
        ploidy = 2 * np.ones((num_samples, num_sites))
        return (np.empty((num_sites, 3)), (np.empty((num_samples, num_sites, 3)), ploidy))

    # --- PART 1: Calculate Site Priors (Vectorized) ---
    
    # Sum reads across samples: Shape (num_sites, 2)
    reads_sum = np.sum(reads_array, axis=0)
    total_reads_per_site = np.sum(reads_sum, axis=1)
    
    # Create a mask for the threshold condition
    # condition: total reads >= max(min_total, error_prob * samples)
    threshold = max(min_total_reads, read_error_prob * num_samples)
    valid_mask = total_reads_per_site >= threshold
    
    # Calculate ratios for ALL sites at once using np.where
    # If valid: (1 + ones) / (2 + total)
    # If not valid: read_error_prob
    # Note: We compute the math for all, but only use it where valid_mask is True
    # to avoid division by zero errors, we can add a tiny epsilon or just ignore the invalid math
    # since np.where selects the result afterwards.
    
    numerator = 1 + reads_sum[:, 1]
    denominator = 2 + total_reads_per_site
    calculated_ratios = numerator / denominator
    
    # Apply the condition
    singleton = np.where(valid_mask, calculated_ratios, read_error_prob)
    
    # Calculate priors: Shape (num_sites, 3)
    priors_00 = (1 - singleton) ** 2
    priors_01 = 2 * singleton * (1 - singleton)
    priors_11 = singleton ** 2
    
    # Stack them: Shape (num_sites, 3)
    site_priors = np.stack([priors_00, priors_01, priors_11], axis=1)
    
    # --- PART 2: Calculate Likelihoods (Vectorized across Samples AND Sites) ---
    
    log_half = math.log(0.5)
    log_read_error = math.log(read_error_prob)
    log_read_nonerror = math.log(1 - read_error_prob)
    
    # Extract columns: Shapes are (num_samples, num_sites)
    zeros = reads_array[..., 0] 
    ones = reads_array[..., 1]
    total = zeros + ones
    
    # Compute log binomials on the 2D matrices directly
    lb_total_ones = log_binomial_vectorized(total, ones)
    lb_total_zeros = log_binomial_vectorized(total, zeros)
    
    # Calculate Likelihoods: Shape (num_samples, num_sites)
    ll_00 = lb_total_ones + zeros * log_read_nonerror + ones * log_read_error
    ll_11 = lb_total_zeros + zeros * log_read_error + ones * log_read_nonerror
    ll_01 = lb_total_ones + total * log_half
    
    # Stack into a 3D tensor: Shape (num_samples, num_sites, 3)
    # We use dstack to stack along the last depth dimension
    log_likli_matrix = np.stack([ll_00, ll_01, ll_11], axis=-1)
    
    # --- PART 3: Combine and Normalize ---
    
    # log_priors is (num_sites, 3). log_likli is (num_samples, num_sites, 3).
    # We need to broadcast log_priors.
    # Expand dims of log_priors to (1, num_sites, 3) so it adds to every sample row.
    log_site_priors = np.log(site_priors)
    nonnorm_log_posterior = log_likli_matrix + log_site_priors[np.newaxis, :, :]
    
    # Apply Softmax on the last axis (the 3 classes)
    # This handles the overflow/underflow (max subtraction) automatically.
    posterior_probs = softmax(nonnorm_log_posterior, axis=-1)
    
    # Create ploidy array
    ploidy = 2 * np.ones((num_samples, num_sites))
    
    # posterior_probs is already (num_samples, num_sites, 3) and contiguous
    return (site_priors, (posterior_probs, ploidy))
            
def calc_distance(first_row,second_row,calc_type="diploid"):
    """
    Calculate the probabalistic distance between two rows
    """

    if calc_type == "diploid":
        distances = [[0,1,2],[1,0,1],[2,1,0]]
    else:
        distances = [[0,1],[1,0]]
        
    ens = np.einsum("ij,ik->ijk",first_row,second_row)
    ensd = ens * distances
    
    return np.sum(ensd,axis=None)

def calc_distance_by_site(first_row,second_row,calc_type="diploid"):
    """
    Like calc_distance but instead of summing everything up at
    the end this function gives the distance by site
    """
    if calc_type == "diploid":
        distances = [[0,1,2],[1,0,1],[2,1,0]]
    else:
        distances = [[0,1],[1,0]]
        
    ens = np.einsum("ij,ik->ijk",first_row,second_row)
    ensd = ens * distances
    
    return np.sum(ensd,axis=(1,2))
    
def calc_perc_difference(first_row,second_row,calc_type="diploid"):
    """
    Calculate the probabalistic percentage difference between two rows
    """
    row_len = len(first_row)
    
    return 100*calc_distance(first_row,second_row,calc_type=calc_type)/row_len

def calc_distance_row(row,data_matrix,start_point,calc_type="diploid"):
    """
    Calculate the distance between one row and all rows in a data 
    matrix starting from a given start_point index
    """
    
    num_samples = len(data_matrix)

    row_vals = [0] * start_point
    
    for i in range(start_point,num_samples):
        row_vals.append(calc_distance(row,data_matrix[i],calc_type=calc_type))
    
    return row_vals
    
def generate_distance_matrix(probs_array,
                             keep_flags=None,
                             calc_type="diploid",
                             use_multiprocessing=False):
    """
    Generates a distance matrix for the distance between two samples
    for an array where rows represent the probabalistic genotypes for 
    a single sample
    
    multiprocessing is a bool which controls whether we use multiple threads
    for the task
    """
    
    if keep_flags is None:
        keep_flags = np.array([True for _ in range(probs_array.shape[1])])
    
    if keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags,dtype=bool)
          
    probs_array = probs_array[:,keep_flags]
    
    num_samples = probs_array.shape[0]
    
    probs_copy = probs_array.copy()

    if use_multiprocessing:
        processing_pool = Pool(processes=32)    
    

        dist_matrix = processing_pool.starmap(
        lambda x,y : calc_distance_row(x,probs_copy,y,calc_type=calc_type),
        zip(probs_array,range(num_samples)))
        
        del(processing_pool)
        
    else:
        dist_matrix = []
        for i in range(num_samples):
            dist_matrix.append(calc_distance_row(probs_array[i],probs_copy,i,calc_type=calc_type))
    
    #Convert to array and fill up lower diagonal
    dist_matrix = np.array(dist_matrix)
    dist_matrix = dist_matrix+dist_matrix.T-np.diag(dist_matrix.diagonal())
    
    return dist_matrix

#%%
def greatest_likelihood_hap(hap):
    """
    Convert a probabalistic hap to a deterministic one by
    choosing the highest probability allele at each site
    """
    return np.argmax(hap,axis=1)

def get_heterozygosity(probabalistic_genotype,keep_flags=None):
    """
    Calculate the heterozygosity of a genotype
    """
    if keep_flags is None:
        keep_flags = np.array([True for _ in range(len(probabalistic_genotype))])
        
    if keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags,dtype=bool)
        
    
    gt_use = np.array(probabalistic_genotype)
    gt_use = gt_use[keep_flags]
    
    num_sites = len(gt_use)
    num_hetero = np.sum(gt_use[:,1])
    
    return 100*num_hetero/num_sites

def size_l1(vec):
    """
    Absolute distance between a vector and zero
    """
    return np.nansum(np.abs(vec))

def magnitude_percentage(vec):
    """
    A percentage measure of how far away a diploid is from zero in the 
    L1 metric compared to a vector of 2s. Can be bigger than 100%
    """
    if np.count_nonzero(~np.isnan(vec)) == 0:
        return 0
    return 100*size_l1(vec)/(2*np.count_nonzero(~np.isnan(vec)))

def perc_wrong(haplotype,keep_flags=None):
    """
    Get the percentage of sites that are wrong (not between 0 and 1)
    for a haplotype
    """
    if keep_flags is None:
        keep_flags = np.array([True for _ in range(len(haplotype))])
        
    if keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags,dtype=bool)
        
    hap_use = haplotype[keep_flags]

    num_wrong = np.count_nonzero(np.logical_or(np.array(hap_use) < 0,np.array(hap_use) > 1))
    
    num_tot = len(hap_use)
    
    return 100*num_wrong/num_tot

def fix_hap(haplotype):
    """
    Fix a haplotype by removing negative values and
    values bigger than 1 (setting them to 0 and 1
    respectively)
    
    haplotype must be a numpy float array
    """
    haplotype[haplotype < 0] = 0.0
    haplotype[haplotype > 1] = 1.0

    return haplotype

def combine_haploids(hap_one,hap_two):
    """
    Combine together two haploids to get a diploid
    """
        
    ens = np.einsum("ij,ik->ijk",hap_one,hap_two)

    vals_00 = ens[:,0,0]
    vals_01 = ens[:,0,1]+ens[:,1,0]
    vals_11 = ens[:,1,1]
    
    together = np.ascontiguousarray(np.array([vals_00,vals_01,vals_11]).T)
    
    return together

def get_diff_wrongness(diploid,haploid,keep_flags=None):
    """
    Get the probabalistic difference created by subtracting a 
    haploid from a diploid. Also gets a wrongness metric for
    what probabalistic proportion of the sites where keep_flags=True
    were wrong (not 0 or 1)
    """
    
    if keep_flags is None:
        keep_flags = np.array([True for _ in range(len(diploid))])
    
    if keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags,dtype=bool)
    
    ens = np.einsum("ij,ik->ijk",haploid,diploid)
    
    zeros = ens[:,0,0]+ens[:,1,0]+ens[:,1,1]
    ones = ens[:,0,1]+ens[:,0,2]+ens[:,1,2]
    wrong = ens[:,1,0]+ens[:,0,2]
    
    wrong = wrong[keep_flags]
    
    difference = np.ascontiguousarray(np.array([zeros,ones]).T)
    
    
    perc_wrong = 100*np.sum(wrong)/len(wrong)
    
    return (difference,perc_wrong)

def probability_to_information(probs_list):
    """
    Takes as input a list of two elements of the form 
    [1-p,p] with p in [0,1] and gives the signed information 
    of the data
    
    This is defined so that if p = 0 we return -1, if p = 1
    we return 1 and if p = 0.5 we return 0: more concretely if 
    p < 0.5 then sgn = -1, else sgn = +1 and we reurn sgn*(1-entropy)
    where the entropy is defined in the standard way
    """
    
    p = probs_list[1]
    
    if p == 1:
        return 1
    elif p == 0:
        return -1
    
    if p < 0.5:
        sgn = -1
    else:
        sgn = 1
    
    entropy = -(1-p)*math.log2(1-p)-p*math.log2(p)
    
    return sgn*(1-entropy)

def add_informations(first_information,second_information):
    """
    Adds together two signed information values using the
    relativistic velocity addition formula
    
    Both first_information and second_information must be 
    within [-1,1]
    """
    if abs(first_information) == 1 and abs(second_information) == 1:
        if first_information != second_information:
            return 0
        else:
            return first_information
        
    return (first_information+second_information)/(1+first_information*second_information)

def combine_probabilities(first_prob,second_prob,prior_prob,
                          required_accuracy=10**-13):
    """
    Combine two lists of size equal to 2 and of the form [1-p,p]
    denoting the probability of having a ref/alt at a site for
    a hap and a given prior probability for that site into a
    single new probability
    
    required_accuracy is the amount of accuracy we want in our final answer,
    the runtime of the algorithm is linear in -log(required_accuracy)
    
    required_accuracy can not be lower than about 10**-15 for floating point
    precision reasons. For safety keep it higher than 10**-14
    """ 
    
    first_information = probability_to_information(first_prob)
    second_information = probability_to_information(second_prob)
    
    prior_information = probability_to_information(prior_prob)
    
    first_relative = add_informations(first_information,-prior_information)
    second_relative = add_informations(second_information,-prior_information)
    
    combined_relative_information = add_informations(first_relative,second_relative)
    full_combined_information = add_informations(combined_relative_information,prior_information)
    
    #Invert from information space to get our probability back
    #We do this using binary search
    search_space = [0,1]
        
    while search_space[1]-search_space[0] > required_accuracy:
        midpoint = (search_space[0]+search_space[1])/2
        test_val = probability_to_information([1-midpoint,midpoint])
        
        if test_val < full_combined_information:
            search_space = [midpoint,search_space[1]]
        else:
            search_space = [search_space[0],midpoint]
    
    final_prob = (search_space[0]+search_space[1])/2
    
    return [1-final_prob,final_prob]

def get_dips_from_long_haps(long_haps):
    """
    Create each possible combination of long diploids from 
    a set of long haplotypes and return the result as a high
    dimensional array
    
    This uses multiprocessing to speed up computation
    """
    def extract_and_combine(full_haps,first_index,second_index):
        first_hap = full_haps[first_index]
        second_hap = full_haps[second_index]
        
        return combine_haploids(first_hap,second_hap)
    
    num_haps = len(long_haps)
    all_combs = [(i,j) for i in range(num_haps) for j in range(num_haps)]
    
    processing_pool = Pool(32)
    
    all_combined = processing_pool.starmap(lambda x,y: extract_and_combine(long_haps,x,y),
                                           all_combs)
    
    #Reshape the flat array of size num_haps**2 into a num_haps*num_haps array
    total_combined = []
    for i in range(num_haps):
        subset = all_combined[num_haps*i:num_haps*(i+1)]
        total_combined.append(subset)
    
    total_combined = np.array(total_combined)
    
    return total_combined

def smoothen_probs_vectorized(old_probs, new_probs, learning_rate):
    """
    Vectorized smoothing: Result = (old * (1-lr)) + (new * lr)
    Operates on dictionary values directly using numpy.
    """
    if learning_rate == 1:
        return new_probs
    if learning_rate == 0:
        return old_probs

    smoothed_result = {}
    
    # Iterate [Forward, Backward]
    for i in [0, 1]:
        smoothed_result[i] = {}
        for block_idx in new_probs[i]:
            old_block = old_probs[i][block_idx]
            new_block = new_probs[i][block_idx]
            
            # We trust key order is preserved (Python 3.7+)
            # Extract values to numpy arrays
            keys = list(new_block.keys())
            v_old = np.array(list(old_block.values()))
            v_new = np.array(list(new_block.values()))
            
            # Vectorized Math
            v_smooth = (v_old * (1.0 - learning_rate)) + (v_new * learning_rate)
            
            # Zip back to dictionary
            smoothed_result[i][block_idx] = dict(zip(keys, v_smooth))
            
    return smoothed_result


def map_haplotype_to_blocks(target_haplotype, haps_data):
    """
    Maps a continuous long haplotype (from simulation/ground truth) 
    to the discrete block indices used in the HMM/Beam Search.
    
    Args:
        target_haplotype: (Total_Sites, 2) numpy array of probabilities.
        haps_data: The standard list of blocks used in your pipeline.
        
    Returns:
        A list of indices (e.g. [0, 5, 12, ...]) representing the best 
        matching local haplotype for each block.
    """
    best_path_indices = []
    current_site_idx = 0
    total_target_len = len(target_haplotype)
    
    for i, block in enumerate(haps_data):
        # block[3] contains the dictionary of {index: hap_array}
        block_haps_dict = block[3]
        
        if not block_haps_dict:
            # Handle empty blocks (if any)
            best_path_indices.append(None)
            continue
            
        # Get the length of this block (number of sites)
        # We grab the first hap to check shape
        any_key = next(iter(block_haps_dict))
        block_len = len(block_haps_dict[any_key])
        
        # Ensure we don't go out of bounds
        if current_site_idx + block_len > total_target_len:
            print(f"Warning: Target haplotype shorter than data. Stopped at block {i}.")
            break
            
        # Slice the target haplotype to get the segment for this block
        target_slice = target_haplotype[current_site_idx : current_site_idx + block_len]
        
        # Find best match
        best_key = None
        min_dist = float('inf')
        
        for key, candidate_hap in block_haps_dict.items():
            # Calculate L1 Distance (Sum of absolute differences)
            # This is robust and fast for probabilities
            dist = np.sum(np.abs(target_slice - candidate_hap))
            
            if dist < min_dist:
                min_dist = dist
                best_key = key
        
        best_path_indices.append(best_key)
        
        # Advance the pointer
        current_site_idx += block_len
        
    return best_path_indices

def get_path_log_likelihood(path_indices, full_mesh, verbose=False):
    """
    Calculates the total log-likelihood of a specific sequence of block haplotypes
    based on the transition probabilities in full_mesh.
    
    Args:
        path_indices: List of integer indices representing the haplotype at each block.
                      e.g. [0, 5, 2, 2, 1, ...]
        full_mesh: The dictionary containing transition probabilities.
        verbose: If True, prints exactly where the path breaks (probability 0).
        
    Returns:
        float: Total Log Likelihood. Returns -inf if the path contains 
               an impossible transition (prob=0 or missing from mesh).
    """
    total_log_likelihood = 0.0
    gap = 1 # We measure likelihood based on immediate neighbors
    
    # Check if gap 1 exists in mesh (it should)
    if gap not in full_mesh:
        raise ValueError("Full mesh does not contain gap size 1 transitions.")
    
    # Iterate through the path
    for i in range(len(path_indices) - 1):
        curr_block_idx = i
        next_block_idx = i + 1
        
        curr_hap_val = path_indices[curr_block_idx]
        next_hap_val = path_indices[next_block_idx]
        
        # Construct the key used in the mesh dictionary
        # Structure: ((block_i, hap_val), (block_i+1, hap_val))
        transition_key = ((curr_block_idx, curr_hap_val), 
                          (next_block_idx, next_hap_val))
        
        # Access the forward probability dict (Direction 0)
        try:
            prob_dict = full_mesh[gap][0][curr_block_idx]
            
            if transition_key in prob_dict:
                prob = prob_dict[transition_key]
                
                if prob > 0:
                    step_log_lik = math.log(prob)
                    total_log_likelihood += step_log_lik
                else:
                    if verbose:
                        print(f"Path broken at Block {i}->{i+1}: Probability is 0.0")
                    return float('-inf')
            else:
                # Key missing usually means probability 0 or filtering removed it
                if verbose:
                    print(f"Path broken at Block {i}->{i+1}: Transition {transition_key} not found in mesh.")
                return float('-inf')
                
        except KeyError:
            if verbose:
                print(f"Block {i} missing from mesh entirely.")
            return float('-inf')

    return total_log_likelihood

def get_path_score_beam_view(path_indices, full_mesh, verbose=False):
    """
    Calculates the 'Score' of a path exactly as the Beam Search calculates it.
    It sums evidence from ALL gap sizes (1, 4, 9...) available in the mesh.
    
    This is useful for understanding why the Beam Search preferred Path A over Path B.
    """
    total_score = 0.0
    num_blocks = len(path_indices)
    
    # Iterate through every block
    for i in range(num_blocks):
        
        # The beam search calculates a score for node 'i' based on incoming connections
        # from the PAST (Left-to-Right pass).
        
        # Check all valid gaps looking backwards
        for gap in full_mesh.keys():
            earlier_index = i - gap
            
            if earlier_index >= 0:
                # Get the values
                curr_val = path_indices[i]
                prev_val = path_indices[earlier_index]
                
                # Construct key: ((prev_idx, prev_val), (curr_idx, curr_val))
                transition_key = ((earlier_index, prev_val), (i, curr_val))
                
                try:
                    # Get Forward Probability (Direction 0)
                    prob_dict = full_mesh[gap][0][earlier_index]
                    
                    if transition_key in prob_dict:
                        prob = prob_dict[transition_key]
                        
                        # Apply the same scaling used in Beam Search
                        # score += log(prob) * sqrt(gap)
                        if prob > 0:
                            log_prob = math.log(prob)
                            scaled_score = log_prob * math.sqrt(1/gap)
                            total_score += scaled_score
                        else:
                            if verbose: print(f"Zero prob at {earlier_index}->{i} (Gap {gap})")
                            return float('-inf')
                    else:
                        if verbose: print(f"Missing key at {earlier_index}->{i} (Gap {gap})")
                        return float('-inf')
                        
                except KeyError:
                    pass # Gap/Block combination might not exist, skip
                    
    return total_score

def recombination_fudge(start_probs,distance,recomb_rate=10**-8):
    """
    Function which takes in the genotype copying probabilities and 
    updates them for a location "distance" number of sites downstream 
    based on up to a single recombination event happening within this 
    particular stretch of data
    """
    
    num_rows = start_probs.shape[0]
    num_cols = start_probs.shape[1]
    
    num_possible_switches = num_rows+num_cols-1
    
    non_switch_prob = (1-recomb_rate)**distance
    
    each_switch_prob = (1-non_switch_prob)/num_possible_switches
    
    #Account for the fact that we can recombine back to the haplotype pre recombination
    total_non_switch_prob = non_switch_prob+each_switch_prob
    
    final_mats = []
    
    for i in range(num_rows):
        for j in range(num_cols):
            base = np.zeros((num_rows,num_cols))
            base[i,:] = each_switch_prob*start_probs[i,j]*np.ones(num_cols)
            base[:,j] = each_switch_prob*start_probs[i,j]*np.ones(num_rows)
            base[i,j] = total_non_switch_prob*start_probs[i,j]
            final_mats.append(base)
    
    combined_probability = np.sum(final_mats,axis=0)
    
    return combined_probability

def get_sample_data_at_sites(sample_data,sample_sites,
                             query_sites,ploidy_present = False):
    """
    Helper function to extract a subset of the sample data which is
    for sites at locations sample_sites in order. The function will
    extract the sample data for sites at query_sites. query_sites 
    must be a subarray of sample_sites
    
    If ploidy_present == True then sample_data must be a tuple, the first
    element of which will be the likelihoods and the second the ploidies,
    otherwise sample_data is just the likelihoods
    """
    indices = np.searchsorted(sample_sites,[query_sites[0],query_sites[-1]])
    
    if ploidy_present:
        return (sample_data[0][indices[0]:indices[1]+1,:],sample_data[1][indices[0]:indices[1]+1])
    else:
        return sample_data[indices[0]:indices[1]+1,:]
                       
def get_sample_data_at_sites_multiple(sample_data, sample_sites,
                                      query_sites, ploidy_present=False):
    """
    Helper function to extract a subset of the sample data which is
    for sites at locations sample_sites in order. The function will
    extract the sample data for sites at query_sites. query_sites 
    must be a subarray of sample_sites
    
    This is like get_sample_data_at_sites but works for an array with data for multiple samples
    
    If ploidy_present == True then sample_data must be a tuple, the first
    element of which will be the likelihoods and the second the ploidies,
    otherwise sample_data is just the likelihoods
    """
    
    # Handle the empty case
    if len(query_sites) == 0:
        if ploidy_present:
            # sample_data[0] is (Samples, Sites, 3) -> Return (Samples, 0, 3)
            empty_lik = sample_data[0][:, :0, :]
            # sample_data[1] is (Samples, Sites) -> Return (Samples, 0)
            empty_ploidy = sample_data[1][:, :0]
            return (empty_lik, empty_ploidy)
        else:
            # sample_data is (Samples, Sites, 3) -> Return (Samples, 0, 3)
            return sample_data[:, :0, :]
    
    # Original logic for non-empty case
    # Find start and end indices in sample_sites
    indices = np.searchsorted(sample_sites, [query_sites[0], query_sites[-1]])
    
    if ploidy_present:
        return (sample_data[0][:, indices[0]:indices[1]+1, :],
                sample_data[1][:, indices[0]:indices[1]+1])
    else:
        return sample_data[:, indices[0]:indices[1]+1, :]

def get_best_block_haps_for_long_hap(long_hap,hap_sites,block_haps):
    """
    Takes as input a single long haplotype and returns a list of
    the hap number for each block which best fits the long hap at that
    block as well as a list containing the percentage difference
    between the long hap and the best matching block hap for each block
    """
    best_haps = []
    best_diffs = []
    
    for i in range(len(block_haps)):
        block_sites = block_haps[i][0]
        block_data = get_sample_data_at_sites(long_hap,hap_sites,block_sites)
        
        haps_diffs = {hap:calc_perc_difference(block_data,
            block_haps[i][3][hap],calc_type="haploid") for hap in block_haps[i][3].keys()}
        
        best_hap = min(haps_diffs, key=haps_diffs.get)
        best_diff = haps_diffs[best_hap]
        
        best_haps.append(best_hap)
        best_diffs.append(best_diff)
    
    return (best_haps,best_diffs)

def get_best_block_matches_for_data(samples_data,):
    pass
    