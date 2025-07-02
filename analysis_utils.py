import numpy as np
import math
from multiprocess import Pool
import warnings

warnings.filterwarnings("ignore")
np.seterr(divide='ignore',invalid="ignore")
pass

#%%
def log_fac(x):
    """
    Returns log(x!) exactly if x < 10, otherwise returns
    Stirling's approximation to it
    """
    x = int(x)
    
    if x < 4:
        return math.log(math.factorial(x))
    else:
        return (x+0.5)*math.log(x)-x+0.918938+(1/(12*x)) #0.918938 is ln(sqrt(2*pi))
    
def log_binomial(n,k):
    """
    Returns log(nCk) exactly if n,k < 10, otherwise caclulates
    the consituent factorials through Stirling's approximation
    """
    n = int(n)
    k = int(k)
    
    if n < 35:
        return math.log(math.comb(n,k))
    elif k < 17 and n < 40:
        return math.log(math.comb(n,k))
    else:
        return log_fac(n)-log_fac(k)-log_fac(n-k)
    
def add_log_likelihoods(logli_list):
    """
    Takes as input a list of log likelihoods (in base e) and 
    accurately approximates the log of the sum of 
    the actual probabilities
    """
    logli = np.array(logli_list)
    ma = max(logli)
    logli = logli-ma
    logli = logli[logli > -50]
    probs = np.exp(logli)
    sum_probs = np.sum(probs)
    
    return ma+math.log(sum_probs)

def make_upper_triangular(matrix):
    """
    Make a matrix upper triangular by adding the value at 
    index (j,i) with i < j to the value at index (i,j)
    
    matrix must be a numpy array
    """
    
    return np.triu(matrix+matrix.T-np.diag(matrix.diagonal()))

def reads_to_probabilities(reads_array,read_error_prob = 0.02,min_total_reads=5):
    """
    Convert a reads array to a probability of the underlying
    genotype being 0, 1 or 2
    
    min_total_reads is a minimum number of reads each site must have for it to be considered
    a valid alternate site (this is to reduce the chance of us considering something which is a variant site only because of errors as a real site)
    """
    reads_sum = np.sum(reads_array,axis=0)
    
    num_samples = len(reads_array)
    
    site_ratios = []
    for i in range(len(reads_sum)):
        
        if sum(reads_sum[i]) >= max(min_total_reads,read_error_prob*num_samples):
            site_ratios.append((1+reads_sum[i][1])/(2+reads_sum[i][0]+reads_sum[i][1]))
        else:
            site_ratios.append(read_error_prob)
    
    site_priors = []
    for i in range(len(site_ratios)):
        singleton = site_ratios[i]
        site_priors.append([(1-singleton)**2,2*singleton*(1-singleton),singleton**2])
    site_priors = np.array(site_priors)
    
    num_samples = reads_array.shape[0]
    num_sites = reads_array.shape[1]
    
    new_array = []
    
    log_half = math.log(1/2)
    log_read_error = math.log(read_error_prob)
    log_read_nonerror = math.log(1-read_error_prob)
    for i in range(num_sites):
        prior_vals = site_priors[i]
        log_priors = np.array(np.log(prior_vals))
        new_array.append([])
        for j in range(num_samples):
            
            zeros = reads_array[j][i][0]
            ones = reads_array[j][i][1]
            total = zeros+ones
            
            log_likelihood_00 = log_binomial(total,ones)+zeros*log_read_nonerror+ones*log_read_error
            log_likelihood_11 = log_binomial(total,zeros)+zeros*log_read_error+ones*log_read_nonerror
            log_likelihood_01 = log_binomial(total,ones)+total*log_half
            
            log_likli = np.array([log_likelihood_00,log_likelihood_01,log_likelihood_11])
            
            nonnorm_log_postri = log_priors + log_likli
            
            nonnorm_log_postri -= np.mean(nonnorm_log_postri)
            
            nonnorm_post = np.exp(nonnorm_log_postri)
            posterior = nonnorm_post/sum(nonnorm_post)
            
            new_array[-1].append(posterior)
    
    ploidy = 2*np.ones((num_samples,num_sites))
            
    new_array = np.array(new_array)
    new_array = np.ascontiguousarray(new_array.swapaxes(0,1))
   
    return (site_priors,(new_array,ploidy))
            
def calc_distance(first_row,second_row,calc_type="diploid"):
    """
    Calculate the probabalistic distance between two rows
    """

    if calc_type == "diploid":
        distances = [[0,1,2],[1,0,1],[2,1,0]]
    else:
        distances = [[0,1],[1,0]]
        
    #print(first_row[:10],second_row[:10])
    
    #print(np.where(np.isnan(second_row)))
        
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
        
    #print(first_row[:10],second_row[:10])
    
    #print(np.where(np.isnan(second_row)))
        
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
        processing_pool = Pool(processes=16)    
    

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
        
        # print("VALUE",test_val,full_combined_information,search_space)
        # print(search_space[1]-search_space[0])
        # print()
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
    
    processing_pool = Pool(16)
    
    all_combined = processing_pool.starmap(lambda x,y: extract_and_combine(long_haps,x,y),
                                           all_combs)
    
    #Reshape the flat array of size num_haps**2 into a num_haps*num_haps array
    total_combined = []
    for i in range(num_haps):
        subset = all_combined[num_haps*i:num_haps*(i+1)]
        total_combined.append(subset)
    
    total_combined = np.array(total_combined)
    
    print(total_combined.shape)
    
    return total_combined

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
                       
def get_sample_data_at_sites_multiple(sample_data,sample_sites,
                                      query_sites,ploidy_present = False):
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
    indices = np.searchsorted(sample_sites,[query_sites[0],query_sites[-1]])
    
    if ploidy_present:
        return (sample_data[0][:,indices[0]:indices[1]+1,:],sample_data[1][:,indices[0]:indices[1]+1])
    else:
        return sample_data[:,indices[0]:indices[1]+1,:]

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
    