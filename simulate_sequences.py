"""
File which contains functions that takes as input a list of
probabalistic haplotypes and use them to simulate a multi generation
progeny of founders made up of these haplotypes
"""
import random
import numpy as np
import pickle

# Removed unused imports to keep it clean
# import block_linking_naive
# import analysis_utils
from vcf_data_loader import GenomicData

#%%
def concretify_haps(haps_list):
    """
    Takes a list of probabalistic haps and turns each of them 
    into a list of 0s and 1s by taking the highest probability
    allele at each site
    """
    
    concreted = []
    
    for hap in haps_list:
        concreted.append(np.argmax(hap,axis=1))
    return concreted

def pairup_haps(haps_list,shuffle=False):
    """
    Pair up a list of concrete haps (made up of 0s and 1s)
    """
    
    haps_copy = pickle.loads(pickle.dumps(haps_list))
    
    if shuffle:
        random.shuffle(haps_copy)
    
    num_pairs = len(haps_list)//2
    
    haps_paired = []
    
    for i in range(num_pairs):
        first = haps_copy[2*i]
        second = haps_copy[2*i+1]
        
        haps_paired.append([first,second])
    
    return haps_paired

def recombine_haps(hap_pair,site_locs,
                   recomb_rate=10**-8,mutate_rate=10**-8):
    """
    Takes as input a pair of concrete haps giving the allele
    at each variable site as well as a list of positions for
    the variable sites. The function then creates a composite
    haplotype simulating meiosis by switching over based on
    an exponential distribution
    """
    
    recomb_scale = 1.0/recomb_rate
    mutate_scale = 1.0/mutate_rate
    
    assert len(hap_pair[0]) == len(hap_pair[1]), "Length of two haplotypes is different"
    assert len(hap_pair[0]) == len(site_locs), "Different length of hap and of list of site locations"
    
    #Current location
    cur_loc = site_locs[0]
    cur_loc_index = 0
    
    #Hap we are currently copying from
    using_hap = random.choice([0,1])
    
    #List which will contain the pieces that make up the final hap
    final_hap_list = []
    
    while cur_loc <= site_locs[-1]:
        next_break_distance = np.random.exponential(recomb_scale)
        
        new_loc = cur_loc+np.ceil(next_break_distance)
            
        new_loc_index = np.searchsorted(site_locs,new_loc)
        
        # Slicing is safe even if new_loc_index > len
        adding = hap_pair[using_hap][cur_loc_index:new_loc_index]
        
        final_hap_list.append(adding)
        
        #Switch around our copying haplotype
        using_hap = 1-using_hap
        
        cur_loc = new_loc
        cur_loc_index = new_loc_index
        
        if cur_loc_index >= len(site_locs):
            break
    
    return_value = np.concatenate(final_hap_list)
    
    # Truncate if we overshot (due to exponential jump past end)
    if len(return_value) > len(site_locs):
        return_value = return_value[:len(site_locs)]
    elif len(return_value) < len(site_locs):
        # Should ideally not happen if loop logic is correct, but safe guard:
        pass 

    #Add in mutations
    mutation_points = []
    
    cur_loc = site_locs[0]
    cur_loc_index = 0
    
    while cur_loc <= site_locs[-1]:
        next_mutation_distance = np.random.exponential(mutate_scale)
        
        new_loc = cur_loc+np.floor(next_mutation_distance)
            
        new_loc_index = np.searchsorted(site_locs,new_loc)
        
        if new_loc_index < len(site_locs):
            mutation_points.append(new_loc_index)
        
        cur_loc = new_loc
        cur_loc_index = new_loc_index
        
        if cur_loc_index >= len(site_locs):
            break
    
    # Apply mutations (flip 0->1, 1->0)
    if len(mutation_points) > 0:
        base_vals = return_value[mutation_points]
        mutated_vals = 1-base_vals
        return_value[mutation_points] = mutated_vals
    
    return return_value

def create_offspring(first_pair,second_pair,
                     site_locs,recomb_rate=10**-8,
                     mutate_rate=10**-8):
    """
    Create an offspring from the two parents which have haplotype pairs
    first_pair and second_pair respectively by running recombine_haps on 
    each of the pairs and then returning the combined result
    """
    
    first_hap = recombine_haps(first_pair,site_locs,recomb_rate=recomb_rate, mutate_rate=mutate_rate)
    second_hap = recombine_haps(second_pair,site_locs,recomb_rate=recomb_rate, mutate_rate=mutate_rate)
    
    return [first_hap,second_hap]

def create_new_generation(parent_haps,site_locs,num_offspring,
                          recomb_rate=10**-8,mutate_rate=10**-8):
    """
    Create a new generation of num_offspring offspring given a list
    of pairs of parental haplotypes.
    """
    new_pairs = []
    
    for i in range(num_offspring):
        parents = random.sample(parent_haps,2)
        
        offspring = create_offspring(parents[0],parents[1],
                                     site_locs,recomb_rate=recomb_rate,
                                     mutate_rate=mutate_rate)
        new_pairs.append(offspring)
    return new_pairs

def get_reads_from_sample(hap_pair,read_depth,error_rate=0.02):
    """
    Simulates sequencing an individual made up of a pair of haplotypes 
    up to read_depth average coverage. Returns an array of size len(hap_pair[0])*2
    where the first column gives the number of 0 reads and the second column gives
    the number of 1 reads
    """
    
    num_sites = len(hap_pair[0])
    
    num_reads_at_site = np.random.poisson(lam=read_depth,size=num_sites)
    
    site_sum = hap_pair[0]+hap_pair[1]
    
    zeros = np.where(site_sum == 0)[0]
    ones = np.where(site_sum == 1)[0]
    twos = np.where(site_sum == 2)[0]
    
    zero_read_counts = num_reads_at_site[zeros]
    one_read_counts = num_reads_at_site[ones]
    two_read_counts = num_reads_at_site[twos]
    
    zero_draws = np.random.binomial(zero_read_counts,error_rate)
    one_draws = np.random.binomial(one_read_counts,0.5)
    two_draws = np.random.binomial(two_read_counts,1-error_rate)
    
    zero_basics = zero_read_counts-zero_draws
    one_basics = one_read_counts-one_draws
    two_basics = two_read_counts-two_draws
    
    zero_concated = np.column_stack((zero_basics, zero_draws))
    one_concated = np.column_stack((one_basics, one_draws))
    two_concated = np.column_stack((two_basics, two_draws))
    
    full_scaffold = np.zeros((num_sites,2), dtype=int)
    
    full_scaffold[zeros,:] = zero_concated
    full_scaffold[ones,:] = one_concated
    full_scaffold[twos,:] = two_concated
    
    return full_scaffold
    
def read_sample_all_individuals(individual_list,read_depth,error_rate=0.02):
    """
    Takes a list of pairs of haps (individuals) and samples them all to
    read_depth read coverage on average
    """
    sampled = []
    
    for item in individual_list:
        reads = get_reads_from_sample(item,read_depth,error_rate=error_rate)
        sampled.append(reads)
    
    np_array = np.array(sampled,dtype=int)
    
    return np_array        

def combine_into_genotype(individual_list):
    """
    Takes as input a list of pairs of haplotypes meant to represent 
    an individual and turn them into a combined likelihood genotype.
    
    Returns likelihoods_array.
    """
    all_list = []
    
    for i in range(len(individual_list)):
        indexing = individual_list[i][0]+individual_list[i][1]
        num_sites = len(indexing)
        
        base_array = [np.array(range(num_sites)), indexing]
        
        # Advanced indexing instead of concatenation
        # scaffold[site_idx, genotype_val] = 1
        scaffold = np.zeros((num_sites,3))
        scaffold[base_array[0], base_array[1]] = 1
        
        all_list.append(scaffold)
    
    return np.array(all_list)
    
def chunk_up_data(positions_list,reads_array,
                  starting_pos,ending_pos,
                  block_size,shift_size,
                  use_snp_count=False, snps_per_block=200, snp_shift=100,
                  error_rate=0.02,
                  min_total_reads=5):
    """
    Breaks up the positions_list and reads_array into blocks.
    
    Modes:
    1. Physical Distance: Uses block_size and shift_size (bp).
    2. SNP Count: Uses snps_per_block and snp_shift (count).
    
    Returns a GenomicData object.
    """
    chunked_positions = []
    chunked_reads = []
    chunked_keep_flags = []
    
    num_samples = reads_array.shape[0]
    total_sites = len(positions_list)
    
    # Restrict to requested physical range first
    # Find indices corresponding to [starting_pos, ending_pos)
    # Using searchsorted for speed
    range_start_idx = np.searchsorted(positions_list, starting_pos)
    range_end_idx = np.searchsorted(positions_list, ending_pos)
    
    # Work on the slice
    positions_slice = positions_list[range_start_idx:range_end_idx]
    reads_slice = reads_array[:, range_start_idx:range_end_idx, :]
    
    slice_len = len(positions_slice)
    
    if slice_len == 0:
        return GenomicData([], [], [])

    if use_snp_count:
        # --- SNP COUNT LOGIC ---
        curr_idx = 0
        while curr_idx < slice_len:
            end_idx = min(curr_idx + snps_per_block, slice_len)
            
            # If block is too small (end of chromosome), keep it? 
            # Usually yes, unless < min_snps required.
            if end_idx == curr_idx: break
            
            block_positions = positions_slice[curr_idx:end_idx]
            block_reads_array = reads_slice[:, curr_idx:end_idx, :]
            
            total_read_pos = np.sum(block_reads_array, axis=(0,2))
            block_keep_flags = (total_read_pos >= max(min_total_reads, error_rate*num_samples)).astype(int)
            
            chunked_positions.append(np.array(block_positions))
            chunked_reads.append(block_reads_array)
            chunked_keep_flags.append(block_keep_flags)
            
            curr_idx += snp_shift
            
    else:
        # --- PHYSICAL DISTANCE LOGIC ---
        cur_pos = positions_slice[0]
        # We need to track index relative to the slice
        cur_idx_in_slice = 0
        
        max_phys_pos = positions_slice[-1]
        
        while cur_pos < max_phys_pos:
            block_end_pos = cur_pos + block_size
            
            # Find end index in slice
            # We search starting from cur_idx_in_slice for efficiency
            end_idx_in_slice = cur_idx_in_slice
            while end_idx_in_slice < slice_len and positions_slice[end_idx_in_slice] < block_end_pos:
                end_idx_in_slice += 1
            
            block_positions = positions_slice[cur_idx_in_slice:end_idx_in_slice]
            
            if len(block_positions) > 0:
                block_reads_array = reads_slice[:, cur_idx_in_slice:end_idx_in_slice, :]
                
                total_read_pos = np.sum(block_reads_array, axis=(0,2))
                block_keep_flags = (total_read_pos >= max(min_total_reads, error_rate*num_samples)).astype(int)
                
                chunked_positions.append(np.array(block_positions))
                chunked_reads.append(block_reads_array)
                chunked_keep_flags.append(block_keep_flags)
            
            cur_pos = cur_pos + shift_size
            
            # Advance start index
            while cur_idx_in_slice < slice_len and positions_slice[cur_idx_in_slice] < cur_pos:
                cur_idx_in_slice += 1
                
    return GenomicData(chunked_positions, chunked_keep_flags, chunked_reads)