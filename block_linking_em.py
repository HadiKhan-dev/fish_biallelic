import numpy as np
import math
import time
from multiprocess import Pool

import analysis_utils
import block_haplotypes
import hap_statistics
#%%
test_site_priors = []
test_probs_arrays = []

for i in range(len(simd_reads)):
    (a,b) = analysis_utils.reads_to_probabilities(simd_reads[i])
    
    test_site_priors.append(a)
    test_probs_arrays.append(b)

test_haps = block_haplotypes.generate_haplotypes_all(
            simd_pos,simd_reads,simd_keep_flags)
#%%

def initial_transition_probabilities(hap_data,
                                     space_gap=1,
                                     forward=True):
    """
    Creates a dict of initial equal transition probabilities
    for a list where each element of the list contains info about
    block haps for that block.
    
    space_gap is the number of blocks we jump over at each step
    for calculating the transition probabilities. By default this
    is equal to 1.
    
    If forward=True the transition probabilities are given as if
    we were iterating forwards through the list of block haplotypes
    and if forward = False then the transition probabilities are
    given as if we were iterating backwards

    """
    transition_dict = {}
    
    if forward:
        for i in range(0,len(hap_data)-space_gap):
            transition_dict[i] = {}
            
            these_haps = hap_data[i][3]
            next_haps = hap_data[i+space_gap][3]
            
            for first_idx in these_haps.keys():
                first_hap_name = (i,first_idx)
                
                num_second_haps = len(next_haps)
                
                for second_idx in next_haps.keys():
                    second_hap_name = (i+space_gap,second_idx)
                    transition_dict[i][(first_hap_name,second_hap_name)] = 1/num_second_haps
    else:
        for i in range(len(hap_data)-1,space_gap-1,-1):
            transition_dict[i] = {}
            
            these_haps = hap_data[i][3]
            next_haps = hap_data[i-space_gap][3]
            
            for first_idx in these_haps.keys():
                first_hap_name = (i,first_idx)
                
                num_second_haps = len(next_haps)
                
                for second_idx in next_haps.keys():
                    second_hap_name = (i-space_gap,second_idx)
                    transition_dict[i][(first_hap_name,second_hap_name)] = 1/num_second_haps

    return transition_dict

def get_block_likelihoods(sample_data,haps_data,
                          log_likelihood_base=math.e**3,
                          min_per_site_log_likelihood=-100):
    """
    Get the log-likelihoods for each combination of haps matching the sample
    
    haps_data must be the full list of haplotype information for a block, including
    the positions,keep_flags,read_counts and haplotype info elements
    
    For each site we copute the distance of the sample data probabalistic genotype
    from the probabalistic genotype given by combining each pair of haps in haps_data.
    We then get the log_likelihood for that site as
    max(-dist*log(log_likelihood_base),min_per_site_log_likelihood)
    """
    
    assert len(haps_data[0]) == len(sample_data), "Number of sites in sample don't match number of sites in haps"
    
    bool_keepflags = haps_data[1].astype(bool)
    
    sample_keep = sample_data[bool_keepflags,:]
    
    ll_dict = {}
    
    
    for i in haps_data[3].keys():
        for j in haps_data[3].keys():
            if j < i:
                continue
            
            combined_haps = analysis_utils.combine_haploids(haps_data[3][i],haps_data[3][j])
            combined_keep = combined_haps[bool_keepflags,:]
            
            dist = analysis_utils.calc_distance_by_site(sample_keep,combined_keep)
            
            f = min_per_site_log_likelihood*np.ones((1,len(dist)))

            
            bdist = -(dist**2)*math.log(log_likelihood_base)
            combined_logs = np.concatenate([np.array(bdist.reshape(1,-1)),min_per_site_log_likelihood*np.ones((1,len(dist)))])
            
            combined_dist = np.max(combined_logs,axis=0)
            
            total_ll = np.sum(combined_dist)
            
            ll_dict[(i,j)] = total_ll
    
    return ll_dict

def get_sample_data_at_sites(sample_data,sample_sites,query_sites):
    """
    Helper function to extract a subset of the sample data which is
    for sites at locations sample_sites in order. The function will
    extract the sample data for sites at query_sites. query_sites 
    must be a subarray of sample_sites
    """
    indices = np.searchsorted(sample_sites,[query_sites[0],query_sites[-1]])
    
    return sample_data[indices[0]:indices[1]+1,:]
    
def get_full_probs_forward(sample_data,sample_sites,haps_data,
                           transition_probs,space_gap=1):
    """
    Compute the forward step in the forward-backward inference
    algorithm for the HMM where we observe our data for a single
    sample given underlying transition probabilities
    
    space_gap is the number of blocks we jump over to get 
    consecutive transitions from, by default this is equal to 1
    
    The value of space_gap must correspond to the same space gap the
    transition_probs are for
    """
    
    likelihood_numbers = {}
    
    for i in range(len(haps_data)):
        block_haps = haps_data[i]
        block_sites = block_haps[0]
        
        block_sample_data = get_sample_data_at_sites(sample_data,sample_sites,block_sites)
        block_likelihoods = get_block_likelihoods(block_sample_data,block_haps)
        
        likelihoods = {}
        
        if i < space_gap:
            for hap_pair in block_likelihoods.keys():
                new_name = ((i,hap_pair[0]),(i,hap_pair[1]))
                likelihoods[new_name] = block_likelihoods[hap_pair]
        else:
            earlier_block = i-space_gap
            earlier_likelihoods = likelihood_numbers[earlier_block]
            earlier_haps = haps_data[earlier_block]
            
            for hap_pair in block_likelihoods.keys():
                new_name = ((i,hap_pair[0]),(i,hap_pair[1]))
                direct_likelihood = block_likelihoods[hap_pair]
                
                total_sum_probs = [] #List which will contain the likelihoods for each possible earlier pair (a,b) which could transition to our hap at this step
                
                for earlier_first in earlier_haps[3].keys():
                    for earlier_second in earlier_haps[3].keys():
                        if earlier_second < earlier_first:
                            continue
                        earlier_dip_name = ((earlier_block,earlier_first),
                                            (earlier_block,earlier_second))
                        
                        transition_prob = transition_probs[earlier_block][((earlier_block,earlier_first),(i,hap_pair[0]))]*transition_probs[earlier_block][((earlier_block,earlier_second),(i,hap_pair[1]))] \
                                        + transition_probs[earlier_block][((earlier_block,earlier_first),(i,hap_pair[1]))]*transition_probs[earlier_block][((earlier_block,earlier_second),(i,hap_pair[0]))]                                                                                                            
                        earlier_log_likelihood = earlier_likelihoods[earlier_dip_name]
                        
                        #print("T",new_name,transition_prob)
                        combined_log_likelihood = earlier_log_likelihood+math.log(transition_prob)
                        
                        total_sum_probs.append(combined_log_likelihood)
                
                combined_prob = analysis_utils.add_log_likelihoods(total_sum_probs)+direct_likelihood
                likelihoods[new_name] = combined_prob                            
        
        likelihood_numbers[i] = likelihoods
   
    return likelihood_numbers

def get_full_probs_backward(sample_data,sample_sites,haps_data,
                           transition_probs,space_gap=1):
    """
    Compute the forward step in the forward-backward inference
    algorithm for the HMM where we observe our data for a single
    sample given underlying transition probabilities
    
    space_gap is the number of blocks we jump over to get 
    consecutive transitions from, by default this is equal to 1
    
    The value of space_gap must correspond to the same space gap the
    transition_probs are for
    """
    
    likelihood_numbers = {}
    
    for i in range(len(haps_data)-1,-1,-1):
        block_haps = haps_data[i]
        block_sites = block_haps[0]
        
        block_sample_data = get_sample_data_at_sites(sample_data,sample_sites,block_sites)
        block_likelihoods = get_block_likelihoods(block_sample_data,block_haps)
        
        likelihoods = {}
        
        if i >= len(haps_data)-space_gap:
            for hap_pair in block_likelihoods.keys():
                new_name = ((i,hap_pair[0]),(i,hap_pair[1]))
                likelihoods[new_name] = block_likelihoods[hap_pair]
        else:
            earlier_block = i+space_gap
            earlier_likelihoods = likelihood_numbers[earlier_block]
            earlier_haps = haps_data[earlier_block]
            
            for hap_pair in block_likelihoods.keys():
                new_name = ((i,hap_pair[0]),(i,hap_pair[1]))
                direct_likelihood = block_likelihoods[hap_pair]
                
                total_sum_probs = [] #List which will contain the likelihoods for each possible earlier pair (a,b) which could transition to our hap at this step
                
                for earlier_first in earlier_haps[3].keys():
                    for earlier_second in earlier_haps[3].keys():
                        if earlier_second < earlier_first:
                            continue
                        earlier_dip_name = ((earlier_block,earlier_first),
                                            (earlier_block,earlier_second))

                        transition_prob = transition_probs[i][((i,hap_pair[0]),(earlier_block,earlier_first))]*transition_probs[i][((i,hap_pair[1]),(earlier_block,earlier_second))] \
                                        + transition_probs[i][((i,hap_pair[1]),(earlier_block,earlier_first))]*transition_probs[i][((i,hap_pair[0]),(earlier_block,earlier_second))]                                                                                                            
                        
                        
                        earlier_log_likelihood = earlier_likelihoods[earlier_dip_name]
                        
                        combined_log_likelihood = earlier_log_likelihood+math.log(transition_prob)
                        
                        total_sum_probs.append(combined_log_likelihood)
                
                combined_prob = analysis_utils.add_log_likelihoods(total_sum_probs)+direct_likelihood
                likelihoods[new_name] = combined_prob                            
        
        likelihood_numbers[i] = likelihoods
   
    return likelihood_numbers
    
def get_updated_transition_probabilities(full_samples_data,
                                         sample_sites,
                                         haps_data,
                                         current_transition_probs,
                                         space_gap=1,
                                         minimum_transition_log_likelihood=-15):
    """
    Uses an EM algorithm to come up with updated transition probabilities 
    for haps between blocks given data for a bunch of samples
    as well as haps_data for each block
    
    minimum_transition_log_likelihood is a parameter which gives the log of the 
    smallest possible transition probability between two adjact block haps. This
    is set to -15 (giving e**-15 = 3*10^-7 as the probability) as is done to
    avoid numerical errors caused by numbers rounding off to 0 in the EM-process
    """
    
    processing_pool = Pool(8)
    
    samples_probs = []
    
    forward_nums = []
    backward_nums = []
    
    forward_nums = processing_pool.starmap(
        lambda x : get_full_probs_forward(x,
                    sample_sites,haps_data,
                    current_transition_probs,
                    space_gap=space_gap),
        zip(full_samples_data))
    backward_nums = processing_pool.starmap(
        lambda x : get_full_probs_forward(x,
                    sample_sites,haps_data,
                    current_transition_probs,
                    space_gap=space_gap),
        zip(full_samples_data))
    
    for i in range(len(forward_nums)):
        samples_probs.append((forward_nums[i],backward_nums[i]))
    
    
    # for i in range(len(full_samples_data)): #Iterate over the samples
    #     print(i)
    
    #     sample_data = full_samples_data[i]
        
    #     forward_probs = get_full_probs_forward(sample_data,sample_sites,
    #                     haps_data,current_transition_probs,space_gap=space_gap)
    #     backward_probs = get_full_probs_backward(sample_data,sample_sites,
    #                     haps_data,current_transition_probs,space_gap=space_gap)
    #     samples_probs.append((forward_probs,backward_probs))
    
    new_transition_probs = {}
    
    #Calculate overall transition likelihoods
    for i in range(len(haps_data)-space_gap):
        print(f"Down {i}")
        next_bundle = i+space_gap
        
        first_haps = haps_data[i][3]
        second_haps = haps_data[next_bundle][3]
        
        transitions_likelihoods = {}
        
        for first in first_haps.keys():
            for second in second_haps.keys():
                tots_comb = []
                
                for s in range(len(samples_probs)):
                    data_here = samples_probs[s]
                    
                    lower_comb = []
                    
                    for first_in_data in first_haps.keys():
                        if first <= first_in_data:
                            current_key = ((i,first),(i,first_in_data))
                        else:
                            current_key = ((i,first_in_data),(i,first))
                            
                        for second_in_data in second_haps.keys():
                            if second <= second_in_data:
                                next_key = ((next_bundle,second),(next_bundle,second_in_data))
                            else:
                                next_key = ((next_bundle,second_in_data),(next_bundle,second))
                                
                            lower_comb.append(data_here[0][i][current_key]+data_here[1][next_bundle][next_key])
                            
                            
                    tots_comb.append(analysis_utils.add_log_likelihoods(lower_comb))
                
                sample_combined_likelihood = analysis_utils.add_log_likelihoods(tots_comb)
                transitions_likelihoods[((i,first),(next_bundle,second))] = sample_combined_likelihood
        
        #Now add everything up, remembering to normalize
        
        overall_likelihood_dict = {}
        for first in first_haps.keys():
            overall_likelihood_dict[(i,first)] = []
            
        #Construct the overall_likelihood_dict, first as a list of loglikeli and then to a single combined number
        for k in transitions_likelihoods.keys():
            first_part = k[0]
            
            overall_likelihood_dict[first_part].append(transitions_likelihoods[k])

        for k in overall_likelihood_dict.keys():
            overall_likelihood_dict[k] = analysis_utils.add_log_likelihoods(list(overall_likelihood_dict[k]))
        
        
        final_non_norm_likelihoods = {}
        
        #Combine to get a sums for each node almost equal to 1 dict for new transition probabilities, imposing our minimum probability constraint
        for k in transitions_likelihoods.keys():
            first_part = k[0]
            
            #Make sure no probability can be above e**minimum_transition_log_likelihood
            final_non_norm_likelihoods[k] = math.exp(max(transitions_likelihoods[k]-overall_likelihood_dict[first_part],minimum_transition_log_likelihood))
        
        final_likelihoods = {}
        
        #Renormalize so that sums for transitions out of any node add up to 1
        for first in first_haps.keys():
            probs_sum = 0
            for second in second_haps.keys():
                keyname = ((i,first),(next_bundle,second))
                
                probs_sum += final_non_norm_likelihoods[keyname]
            
            for second in second_haps.keys():
                keyname = ((i,first),(next_bundle,second))
                final_likelihoods[keyname] = final_non_norm_likelihoods[keyname]/probs_sum
        
        new_transition_probs[i] = final_likelihoods
        
    return new_transition_probs

def calculate_hap_transition_probabilities(full_samples_data,
                                         sample_sites,
                                         haps_data,
                                         num_iterations=6,
                                         space_gap=1,
                                         minimum_transition_log_likelihood=-15):
    """
    Starting out with an equal prior compute update transition probabilities
    for adjacent haps (where by adjacency we mean a gap of size space_gap) 
    by applying an EM algorithm num_iteration times.
    
    Returns the result of the final run of the algorithm
    """
    start_probs = initial_transition_probabilities(haps_data,space_gap=space_gap)
    
    probs_list = [start_probs]
    
    for i in range(num_iterations):
        new_probs = get_updated_transition_probabilities(full_samples_data,
                sample_sites,haps_data,probs_list[-1],
                space_gap=space_gap,
                minimum_transition_log_likelihood=minimum_transition_log_likelihood)
        probs_list.append(new_probs)
    
    return probs_list[-1]

def generate_transition_probability_mesh(full_samples_data,
                                         sample_sites,
                                         haps_data,
                                         num_iterations=6,
                                         minimum_transition_log_likelihood=-15):
    """
    Generates a mesh of transition probabilities where we generate the transition
    probabilities for space_gap = 1,2,4,8,... all the way up to the largest
    power of two less than the number of blocks we have
    """
    mesh_dict = {}
    
    num_powers = math.floor(math.log(len(full_samples_data)-1,2))
    
    for i in range(num_powers):
        
        mesh_dict[2**i] = calculate_hap_transition_probabilities(full_samples_data,
                                                 sample_sites,
                                                 haps_data,
                                                 num_iterations=num_iterations,
                                                 space_gap=2**i,
                                                 minimum_transition_log_likelihood=minimum_transition_log_likelihood)
    
    return mesh_dict
#%%
space_gap = 2
#%%
start = time.time()
final_mesh = generate_transition_probability_mesh(offspring_genotype_likelihoods[1],
        offspring_genotype_likelihoods[0],test_haps)
print(time.time()-start)

#%%
start = time.time()
final_probs = calculate_hap_transition_probabilities(offspring_genotype_likelihoods[1],
        offspring_genotype_likelihoods[0],test_haps,space_gap=space_gap)
print(time.time()-start)

#%%
start = time.time()
updated = get_updated_transition_probabilities(offspring_genotype_likelihoods[1],
        offspring_genotype_likelihoods[0],test_haps,starting_transition_probs,
        space_gap=space_gap)
print(time.time()-start)
#%%
start = time.time()
updated2 = get_updated_transition_probabilities(offspring_genotype_likelihoods[1],
        offspring_genotype_likelihoods[0],test_haps,updated,
        space_gap=space_gap)
print(time.time()-start)
#%%
start = time.time()
updated3 = get_updated_transition_probabilities(offspring_genotype_likelihoods[1],
        offspring_genotype_likelihoods[0],test_haps,updated2,
        space_gap=space_gap)
print(time.time()-start)
#%%
start = time.time()
updated4 = get_updated_transition_probabilities(offspring_genotype_likelihoods[1],
        offspring_genotype_likelihoods[0],test_haps,updated3,
        space_gap=space_gap)
print(time.time()-start)
#%%
start = time.time()
updated5 = get_updated_transition_probabilities(offspring_genotype_likelihoods[1],
        offspring_genotype_likelihoods[0],test_haps,updated4,
        space_gap=space_gap)
print(time.time()-start)
#%%
start = time.time()
updated6 = get_updated_transition_probabilities(offspring_genotype_likelihoods[1],
        offspring_genotype_likelihoods[0],test_haps,updated5,
        space_gap=space_gap)
print(time.time()-start)
#%% #12
start = time.time()
updated7 = get_updated_transition_probabilities(offspring_genotype_likelihoods[1],
        offspring_genotype_likelihoods[0],test_haps,updated6,
        space_gap=space_gap)
print(time.time()-start)