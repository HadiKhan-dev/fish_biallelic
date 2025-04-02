"""
File which contains functions that takes as input a list of
probabalistic haplotypes and use them to simulate a multi generation
progeny of founders made up of these haplotypes
"""
import random
import numpy as np
import pickle

import block_linking_naive
import analysis_utils

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
    
    recomb_scale = 1/recomb_rate
    mutate_scale = 1/mutate_rate
    
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
        
        adding = hap_pair[using_hap][cur_loc_index:new_loc_index]
        
        final_hap_list.append(adding)
        
        #Switch around our copying haplotype
        using_hap = 1-using_hap
        
        cur_loc = new_loc
        cur_loc_index = new_loc_index
    
    return_value = np.concatenate(final_hap_list)
    
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
    
    first_hap = recombine_haps(first_pair,site_locs,recomb_rate=recomb_rate)
    second_hap = recombine_haps(second_pair,site_locs,recomb_rate=recomb_rate)
    
    return [first_hap,second_hap]

def create_new_generation(parent_haps,site_locs,num_offspring,
                          recomb_rate=10**-8,mutate_rate=10**-8):
    """
    Create a new generation of num_offspring offspring given a list
    of pairs of parental haplotypes. Each offspring is made by randomly
    choosing 
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
    
    zero_concated = np.concatenate([np.reshape(zero_basics,(-1,1)),np.reshape(zero_draws,(-1,1))],axis=1)
    one_concated = np.concatenate([np.reshape(one_basics,(-1,1)),np.reshape(one_draws,(-1,1))],axis=1)
    two_concated = np.concatenate([np.reshape(two_basics,(-1,1)),np.reshape(two_draws,(-1,1))],axis=1)
    
    full_scaffold = np.zeros((num_sites,2))
    
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

def combine_into_genotype(individual_list,sites_data):
    """
    Takes as input a list of pairs of haplotypes meant to represent 
    an individual and turn them into a combined likelihood genotype
    """
    all_list = []
    ploidy = []
    
    for i in range(len(individual_list)):
        indexing = individual_list[i][0]+individual_list[i][1]
        num_sites = len(indexing)
        
        base_array = [np.array(range(num_sites)).reshape(1,-1),indexing.reshape(1,-1)]
        
        combined_indexing = np.concatenate(base_array,axis=0).T
        
        scaffold = np.zeros((num_sites,3))
        
        scaffold[combined_indexing[:,0],combined_indexing[:,1]] = 1
        
        all_list.append(scaffold)
        ploidy.append([2 for _ in range(num_sites)])
    
    return [sites_data,(np.array(all_list),np.array(ploidy))]
    
def chunk_up_data(positions_list,reads_array,
                  starting_pos,ending_pos,
                  block_size,shift_size,
                  error_rate=0.02,
                  min_total_reads=5):
    """
    Breaks up the positions_list and reads_array (where elements of the positions_list 
    correspond to the location of the site at that index in reads_array) into chunks
    of size block_size where consecutive chunks differ by shift_size in their starting
    location
    
    If we have fewer than min_total_reads or error_rate*num_samples (whichever is higher)
    at a site then we mark it as a 0 to indicate it is low quality and not to be considered
    for our founder inference algorithm
    """
    chunked_positions = []
    chunked_reads = []
    chunked_keep_flags = []
    
    num_samples = reads_array.shape[0]
    
    cur_pos = starting_pos
    cur_index = 0
    last_site = positions_list[-1]
    
    while cur_pos < ending_pos:
        block_start_index = cur_index
        
        block_end_pos = cur_pos+block_size
        
        end_index = cur_index
        while end_index < len(positions_list) and positions_list[end_index] < block_end_pos:
            end_index += 1
        
        block_positions = positions_list[cur_index:end_index]
        extract_indices = list(range(cur_index,end_index))
        
        block_reads_array = reads_array[:,extract_indices,:]
        
        total_read_pos = np.sum(block_reads_array,axis=(0,2))
        
        block_keep_flags = (total_read_pos >= max(min_total_reads,error_rate*num_samples)).astype(int)
        
        chunked_positions.append(block_positions)
        chunked_reads.append(block_reads_array)
        chunked_keep_flags.append(block_keep_flags)
        
        cur_pos = cur_pos + shift_size
        
        while cur_index < len(positions_list) and positions_list[cur_index] < cur_pos:
            cur_index += 1
    
    return (chunked_positions,chunked_keep_flags,chunked_reads)
        
def calc_distance_concrete(first_row,second_row):
    """
    Calculate the L1 distance between two rows
    """
    
    diff = np.sum(np.abs(first_row-second_row))
    
    return diff
        
#%%
haplotype_sites = final_test[0]
haplotype_data = final_test[1]
    
cm = concretify_haps(haplotype_data)
pa = pairup_haps(cm)
#%%
f1 = create_new_generation(pa,haplotype_sites,10,recomb_rate=10**-7,mutate_rate=10**-10)
f2 = create_new_generation(f1,haplotype_sites,100,recomb_rate=10**-7,mutate_rate=10**-10)
f3 = create_new_generation(f2,haplotype_sites,200,recomb_rate=10**-7,mutate_rate=10**-10)

#%%
all_offspring = [xs for x in [f1,f2,f3] for xs in x]
offspring_genotype_likelihoods = combine_into_genotype(all_offspring,haplotype_sites)
#%%
new_reads_array = read_sample_all_individuals(all_offspring,10,error_rate=0.02)
#%%
print("Starting")
(simd_pos,simd_keep_flags,simd_reads) = chunk_up_data(
    haplotype_sites,new_reads_array,
    35000000,42700000,100000,100000)
simd_probabalistic_genotypes = analysis_utils.reads_to_probabilities(new_reads_array)
#%%
##############################
(simd_block_haps,simd_haps) = block_linking_naive.generate_long_haplotypes_naive(simd_pos,simd_reads,6,simd_keep_flags)
#%%
simd_conc = concretify_haps(simd_haps[1])

#%%
for i in range(len(haplotype_data)):
    for j in range(len(simd_haps[1])):
        print(i,j,f"{100*calc_distance_concrete(cm[i],simd_conc[j])/len(simd_haps[1][i]):.2f}%")
    print()
#%%
base_idx = 26

start_pos = 2500000+50000*base_idx
start_index = np.where(haplotype_sites > start_pos)[0][0]
block_num_sites = len(simd_block_haps[base_idx][0])
end_index = start_index + block_num_sites

for i in range(len(haplotype_data)):
    hap = haplotype_data[i][start_index:end_index]
    for j in range(len(simd_block_haps[base_idx][3])):
        print(i,j,f"{100*analysis_utils.calc_distance(hap,simd_block_haps[base_idx][3][j],calc_type='haploid')/len(hap):.2f}%")
    print()