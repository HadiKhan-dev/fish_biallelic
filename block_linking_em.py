import numpy as np
import math
import time
from multiprocess import Pool
import itertools

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

start = time.time()
test_haps = block_haplotypes.generate_haplotypes_all(
            simd_pos,simd_reads,simd_keep_flags)

print(time.time()-start)

all_sites = offspring_genotype_likelihoods[0]
all_likelihoods = offspring_genotype_likelihoods[1]
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
    transition_dict_forward = {}
    transition_dict_reverse = {}
    
    for i in range(0,len(hap_data)-space_gap):
        transition_dict_forward[i] = {}
        
        these_haps = hap_data[i][3]
        next_haps = hap_data[i+space_gap][3]
        
        for first_idx in these_haps.keys():
            first_hap_name = (i,first_idx)
            
            num_second_haps = len(next_haps)
            
            for second_idx in next_haps.keys():
                second_hap_name = (i+space_gap,second_idx)
                transition_dict_forward[i][(first_hap_name,second_hap_name)] = 1/num_second_haps
                
    for i in range(len(hap_data)-1,space_gap-1,-1):
        transition_dict_reverse[i] = {}
        
        these_haps = hap_data[i][3]
        next_haps = hap_data[i-space_gap][3]
        
        for first_idx in these_haps.keys():
            first_hap_name = (i,first_idx)
            
            num_second_haps = len(next_haps)
            
            for second_idx in next_haps.keys():
                second_hap_name = (i-space_gap,second_idx)
                transition_dict_reverse[i][(first_hap_name,second_hap_name)] = 1/num_second_haps

    return [transition_dict_forward,transition_dict_reverse]

def get_block_likelihoods(sample_data,haps_data,
                          log_likelihood_base=math.e**3,
                          min_per_site_log_likelihood=-100,
                          normalize=True,                          
                          ):
    """
    Get the log-likelihoods for each combination of haps matching the sample
    
    haps_data must be the full list of haplotype information for a block, including
    the positions,keep_flags,read_counts and haplotype info elements
    
    For each site we copute the distance of the sample data probabalistic genotype
    from the probabalistic genotype given by combining each pair of haps in haps_data.
    We then get the log_likelihood for that site as
    max(-dist*log(log_likelihood_base),min_per_site_log_likelihood)
    
    If normalize=True then ensures the sum of the probabilities for 
    the sample over all possible pairs adds up to 1
    """
    
    assert len(haps_data[0]) == len(sample_data), "Number of sites in sample don't match number of sites in haps"
    
    bool_keepflags = haps_data[1].astype(bool)
    
    sample_keep = sample_data[bool_keepflags,:]
    
    ll_dict = {}
    
    ll_list = []
    
    
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
            
    if normalize:
        combined_ll = analysis_utils.add_log_likelihoods(ll_list)
        
        for k in ll_dict.keys():
            ll_dict[k] = ll_dict[k]-combined_ll
            
    return ll_dict

def get_all_block_likelihoods(block_samples_data,block_haps):
    """
    Function which calculates the block likelihoods for each 
    sample in block_samples_data
    """
    sample_likelihoods = []
    
    for i in range(len(block_samples_data)):
        sample_likelihoods.append(
            get_block_likelihoods(block_samples_data[i],
                                  block_haps))
    
    
    return sample_likelihoods

def multiprocess_all_block_likelihoods(full_samples_data,
                                       sample_sites,
                                       haps_data):
    
    processing_pool = Pool(8)

    full_blocks_likelihoods = processing_pool.starmap(
        lambda i: get_all_block_likelihoods(
            get_sample_data_at_sites_multiple(full_samples_data,
            sample_sites,haps_data[i][0]),
            haps_data[i]),
        zip(range(len(haps_data))))
    
    return full_blocks_likelihoods

def get_sample_data_at_sites(sample_data,sample_sites,query_sites):
    """
    Helper function to extract a subset of the sample data which is
    for sites at locations sample_sites in order. The function will
    extract the sample data for sites at query_sites. query_sites 
    must be a subarray of sample_sites
    """
    indices = np.searchsorted(sample_sites,[query_sites[0],query_sites[-1]])
    
    return sample_data[indices[0]:indices[1]+1,:]

def get_sample_data_at_sites_multiple(sample_data,sample_sites,query_sites):
    """
    Helper function to extract a subset of the sample data which is
    for sites at locations sample_sites in order. The function will
    extract the sample data for sites at query_sites. query_sites 
    must be a subarray of sample_sites
    
    This is like get_sample_data_at_sites but works for an array with data for multiple samples
    """
    indices = np.searchsorted(sample_sites,[query_sites[0],query_sites[-1]])

    return sample_data[:,indices[0]:indices[1]+1,:]

    
def get_full_probs_forward(mist,sample_data,
                           sample_sites,
                           haps_data,
                           full_blocks_likelihoods,
                           bidirectional_transition_probs,space_gap=1):
    """
    Compute the forward step in the forward-backward inference
    algorithm for the HMM where we observe our data for a single
    sample given underlying transition probabilities
    
    space_gap is the number of blocks we jump over to get 
    consecutive transitions from, by default this is equal to 1
    
    The value of space_gap must correspond to the same space gap the
    transition_probs are for
    """

    transition_probs = bidirectional_transition_probs[0]
    
    likelihood_numbers = {}
    
    for i in range(len(haps_data)):

        block_likelihoods = full_blocks_likelihoods[i]

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
                        
                        #Flip around our naming scheme for lookups if we are putting a higher index hap before a lower index one
                        if earlier_second < earlier_first:
                            lookup_first = earlier_second
                            lookup_second = earlier_first
                        else:
                            lookup_first = earlier_first
                            lookup_second = earlier_second

                        earlier_dip_name = ((earlier_block,lookup_first),
                                            (earlier_block,lookup_second))
                        
                        transition_prob = transition_probs[earlier_block][((earlier_block,earlier_first),(i,hap_pair[0]))]*transition_probs[earlier_block][((earlier_block,earlier_second),(i,hap_pair[1]))]                                                                                                       

                        earlier_log_likelihood = earlier_likelihoods[earlier_dip_name]
                        
                        
                        combined_log_likelihood = earlier_log_likelihood+math.log(transition_prob)
                        
                        
                        total_sum_probs.append(combined_log_likelihood)

                
                combined_prob = analysis_utils.add_log_likelihoods(total_sum_probs)+direct_likelihood
                likelihoods[new_name] = combined_prob                            
        
        likelihood_numbers[i] = likelihoods

    return likelihood_numbers

def get_full_probs_backward(sample_data,
                            sample_sites,
                            haps_data,
                            full_blocks_likelihoods,
                            bidirectional_transition_probs,space_gap=1):
    """
    Compute the backward step in the forward-backward inference
    algorithm for the HMM where we observe our data for a single
    sample given underlying transition probabilities
    
    space_gap is the number of blocks we jump over to get 
    consecutive transitions from, by default this is equal to 1
    
    The value of space_gap must correspond to the same space gap the
    transition_probs are for
    """
    
    transition_probs = bidirectional_transition_probs[1]
    
    likelihood_numbers = {}
    
    for i in range(len(haps_data)-1,-1,-1):
        block_likelihoods = full_blocks_likelihoods[i]

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
                        
                        #Flip around our naming scheme for lookups if we are putting a higher index hap before a lower index one
                        if earlier_second < earlier_first:
                            lookup_first = earlier_second
                            lookup_second = earlier_first
                        else:
                            lookup_first = earlier_first
                            lookup_second = earlier_second

                        earlier_dip_name = ((earlier_block,lookup_first),
                                            (earlier_block,lookup_second))

                        transition_prob = transition_probs[earlier_block][((earlier_block,earlier_first),(i,hap_pair[0]))]*transition_probs[earlier_block][((earlier_block,earlier_second),(i,hap_pair[1]))]                                                                                                       

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
                                         full_blocks_likelihoods,
                                         current_transition_probs,
                                         space_gap=1,
                                         minimum_transition_log_likelihood=-10):
    """
    Uses an EM algorithm to come up with updated transition probabilities 
    for haps between blocks given data for a bunch of samples
    as well as haps_data for each block
    
    minimum_transition_log_likelihood is a parameter which gives the log of the 
    smallest possible transition probability between two adjact block haps. This
    is set to -15 (giving e**-15 = 3*10^-7 as the probability) as is done to
    avoid numerical errors caused by numbers rounding off to 0 in the EM-process
    """
    
    all_block_likelihoods = []
    
    for i in range(len(full_samples_data)):
        all_block_likelihoods.append(
            [full_blocks_likelihoods[x][i] for x in range(len(full_blocks_likelihoods))])
    
    samples_probs = []
    
    forward_nums = []
    backward_nums = []
    
    
    forward_nums = list(itertools.starmap(
        lambda i : get_full_probs_forward(i,full_samples_data[i],
                    sample_sites,
                    haps_data,
                    all_block_likelihoods[i],
                    current_transition_probs,
                    space_gap=space_gap),
        zip(range(len(full_samples_data)))))
    backward_nums = list(itertools.starmap(
        lambda i : get_full_probs_backward(full_samples_data[i],
                    sample_sites,
                    haps_data,
                    all_block_likelihoods[i],
                    current_transition_probs,
                    space_gap=space_gap),
        zip(range(len(full_samples_data)))))
    
    for i in range(len(forward_nums)):
        samples_probs.append((forward_nums[i],backward_nums[i]))
    
    full_transitions_likelihoods = {}
    
    new_transition_probs_forward = {}
    new_transition_probs_backwards = {}
    
    
    #Calculate overall transition likelihoods
    for i in range(len(haps_data)-space_gap):
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
                            
                            transition_key = ((i,first_in_data),(next_bundle,second_in_data))
                            transition_value = math.log(current_transition_probs[0][i][transition_key])
                            
                            # if i == 0 and first == 0 and second in [2,3] and s == 19:
                                
                            #     print("IDEN",i,s,second,current_key,next_key)
                            #     print("S:",data_here[0][i][current_key])
                            #     print("R:",data_here[1][next_bundle][next_key])
                            #     print("Transition:",transition_value)
                            #     print("Adding:",data_here[0][i][current_key]+data_here[1][next_bundle][next_key]+transition_value)
                            #     print()
                                
                            
                            adding = data_here[0][i][current_key]+data_here[1][next_bundle][next_key]+transition_value
                                
                            lower_comb.append(adding)
                    

                    sample_likelihood = analysis_utils.add_log_likelihoods(lower_comb)
                    
                    tots_comb.append(sample_likelihood)
                    
                
                # if i == 0 and first == 0 and second in [2,3]:
                #     print(i,first,second)
                #     print(sorted(tots_comb)[-5:])
                #     print("MAXIMUM HERE",tots_comb.index(max(tots_comb)))
                #     print()
                
                all_sample_combined_likelihood = analysis_utils.add_log_likelihoods(tots_comb)
                
                
                transitions_likelihoods[((i,first),(next_bundle,second))] = all_sample_combined_likelihood
        
        full_transitions_likelihoods[i] = transitions_likelihoods
        
    
    print("FP")
    print(full_transitions_likelihoods[0])
    
    #Now we build our new transition probabilities going forwards
    for i in range(len(haps_data)-space_gap):
        overall_likelihood_forward_dict = {}
        
        next_bundle = i+space_gap
        
        first_haps = haps_data[i][3]
        second_haps = haps_data[next_bundle][3]
        
        for first in first_haps.keys():
            overall_likelihood_forward_dict[(i,first)] = []
        
        for k in full_transitions_likelihoods[i].keys():
            first_part = k[0]
            overall_likelihood_forward_dict[first_part].append(full_transitions_likelihoods[i][k])
        
        for first_part in overall_likelihood_forward_dict.keys():
            overall_likelihood_forward_dict[first_part] = analysis_utils.add_log_likelihoods(list(overall_likelihood_forward_dict[first_part]))
        
        final_non_norm_forward_likelihoods = {}
        
        for k in full_transitions_likelihoods[i].keys():
            first_part = k[0]
            final_non_norm_forward_likelihoods[k] = math.exp(max(full_transitions_likelihoods[i][k]-overall_likelihood_forward_dict[first_part],minimum_transition_log_likelihood))
        
        final_forward_likelihoods = {}
        
        for first in first_haps.keys():
            probs_sum = 0
            for second in second_haps.keys():
                keyname = ((i,first),(next_bundle,second))
                probs_sum += final_non_norm_forward_likelihoods[keyname]
            for second in second_haps.keys():
                keyname = ((i,first),(next_bundle,second))
                final_forward_likelihoods[keyname] = final_non_norm_forward_likelihoods[keyname]/probs_sum
         
        new_transition_probs_forward[i] = final_forward_likelihoods
        
    #And then we build our new transition probabilities going backwards
    for i in range(len(haps_data)-1,space_gap-1,-1):
        overall_likelihood_backward_dict = {}
        
        next_bundle = i-space_gap
        
        first_haps = haps_data[i][3]
        second_haps = haps_data[next_bundle][3]
        
        for first in first_haps.keys():
            overall_likelihood_backward_dict[(i,first)] = []
        
        for k in full_transitions_likelihoods[next_bundle].keys():
            first_part = k[1]
            overall_likelihood_backward_dict[first_part].append(full_transitions_likelihoods[next_bundle][k])
        
        for first_part in overall_likelihood_backward_dict.keys():
            overall_likelihood_backward_dict[first_part] = analysis_utils.add_log_likelihoods(list(overall_likelihood_backward_dict[first_part]))
        
        final_non_norm_backward_likelihoods = {}
        
        for k in full_transitions_likelihoods[next_bundle].keys():
            first_part = k[1]
            reversed_k = (k[1],k[0])
            final_non_norm_backward_likelihoods[reversed_k] = math.exp(max(full_transitions_likelihoods[next_bundle][k]-overall_likelihood_backward_dict[first_part],minimum_transition_log_likelihood))
        
        final_backward_likelihoods = {}
        
        for first in first_haps.keys():
            probs_sum = 0
            for second in second_haps.keys():
                keyname = ((i,first),(next_bundle,second))
                probs_sum += final_non_norm_backward_likelihoods[keyname]
            for second in second_haps.keys():
                keyname = ((i,first),(next_bundle,second))
                final_backward_likelihoods[keyname] = final_non_norm_backward_likelihoods[keyname]/probs_sum
         
        new_transition_probs_backwards[i] = final_backward_likelihoods
        
    return [new_transition_probs_forward,new_transition_probs_backwards]

def calculate_hap_transition_probabilities(full_samples_data,
                                         sample_sites,
                                         haps_data,
                                         full_blocks_likelihoods=None,
                                         max_num_iterations=6,
                                         space_gap=1,
                                         min_cutoff_change=0.001,
                                         minimum_transition_log_likelihood=-10):
    """
    Starting out with an equal prior compute update transition probabilities
    for adjacent haps (where by adjacency we mean a gap of size space_gap) 
    by applying an EM algorithm num_iteration times.
    
    Cuts off early if we have two successive iterations and no single transition
    probability changes by at least min_cutoff_change
    
    Returns the result of the final run of the algorithm
    
    If full_blocks_likelihoods (which contains the likelihood of seeing each 
    block for each sample given the underlying ground truth of which haplotypes
    make up the sample at that block) is not provided it is computed from the 
    other data
    """
    start_probs = initial_transition_probabilities(haps_data,space_gap=space_gap)

    probs_list = [start_probs]
    
    if full_blocks_likelihoods == None:
        #Calculate site data and underlying likelihoods for each sample
        #and possible pair making up that sample at each block
        processing_pool = Pool(8)
    
        full_blocks_likelihoods = processing_pool.starmap(
            lambda i: get_all_block_likelihoods(
                get_sample_data_at_sites_multiple(full_samples_data,
                sample_sites,haps_data[i][0]),
                haps_data[i]),
            zip(range(len(haps_data))))
    
    for i in range(max_num_iterations):
        new_probs = get_updated_transition_probabilities(
            full_samples_data,
            sample_sites,
            haps_data,
            full_blocks_likelihoods,
            probs_list[-1],
            space_gap=space_gap,
            minimum_transition_log_likelihood=minimum_transition_log_likelihood)
        
        probs_list.append(new_probs)
        
        last_probs = probs_list[-2]
        
        max_diff = 0
        
        for j in [0,1]:
            testing_currently = new_probs[j]
            testing_last = last_probs[j]

            for k in testing_currently.keys():
                for pair in testing_currently[k].keys():
                    diff = abs(testing_currently[k][pair]-testing_last[k][pair])
                    
                    if diff > max_diff:
                        max_diff = diff
        
        print("Max diff:",i,max_diff)
        print()
        if max_diff < min_cutoff_change:
            #print(f"Exiting_early {len(probs_list)}")
            break
    
    return probs_list[-1]

def generate_transition_probability_mesh(full_samples_data,
                                         sample_sites,
                                         haps_data,
                                         max_num_iterations=10,
                                         minimum_transition_log_likelihood=-10):
    """
    Generates a mesh of transition probabilities where we generate the transition
    probabilities for space_gap = 1,2,4,8,... all the way up to the largest
    power of two less than the number of blocks we have
    """
    
    #Calculate site data and underlying likelihoods for each sample
    #and possible pair making up that sample at each block
    processing_pool = Pool(8)

    full_blocks_likelihoods = processing_pool.starmap(
        lambda i: get_all_block_likelihoods(
            get_sample_data_at_sites_multiple(full_samples_data,
            sample_sites,haps_data[i][0]),
            haps_data[i]),
        zip(range(len(haps_data))))
    
    
    mesh_dict = {}
    
    num_powers = math.floor(math.log(len(haps_data)-1,2))+1
    
    powers = [2**i for i in range(num_powers)]

    mesh_list = processing_pool.starmap(lambda x: calculate_hap_transition_probabilities(full_samples_data,
    sample_sites,
    haps_data,
    full_blocks_likelihoods=full_blocks_likelihoods,
    max_num_iterations=max_num_iterations,
    space_gap=x,
    minimum_transition_log_likelihood=minimum_transition_log_likelihood),
    zip(powers))
    
    for i in range(len(powers)):
        mesh_dict[powers[i]] = mesh_list[i]
    
    return mesh_dict

def convert_mesh_to_haplotype(full_samples_data,full_sites,
                              haps_data,full_mesh):
    """
    Given a mesh of transition probabilities uses that to come up 
    with a long haplotype
    """
    
    first_block = haps_data[0]
    
    first_block_haps = first_block[3]
    first_block_sites = first_block[0]
    first_keep_flags = first_block[1]
    
    samples_first_restricted = get_sample_data_at_sites_multiple(
        full_samples_data,full_sites,first_block_sites)
    
    matches = hap_statistics.match_best(first_block_haps,
                samples_first_restricted,first_keep_flags)
    
    
    best_first_match = max(matches[1], key=matches[1].get)
    
    first_pass_hap = [best_first_match]
    
    for i in range(1,len(haps_data)):
        available_haps = haps_data[i][3]
        cur_subtract = 1
        
        while cur_subtract <= i:
            earlier = i-cur_subtract
            
        
#%%
first_block = test_haps[0]

first_block_haps = first_block[3]
first_block_sites = first_block[0]
first_keep_flags = first_block[1]

second_block = test_haps[1]

second_block_haps = second_block[3]
second_block_sites = second_block[0]
second_keep_flags = second_block[1]

samples_first_restricted = get_sample_data_at_sites_multiple(
    all_likelihoods,all_sites,first_block_sites)

samples_second_restricted = get_sample_data_at_sites_multiple(
    all_likelihoods,all_sites,second_block_sites)

first_matches = hap_statistics.match_best(first_block_haps,
            samples_first_restricted,first_keep_flags)

second_matches = hap_statistics.match_best(second_block_haps,
            samples_second_restricted,second_keep_flags)

rel_usage = hap_statistics.relative_haplotype_usage(0,first_matches,second_matches)
rel_usage_indicators = hap_statistics.relative_haplotype_usage_indicator(0,first_matches,second_matches)

#rev_rel_usage0 = hap_statistics.relative_haplotype_usage(0,second_matches,first_matches)
#rev_rel_usage3 = hap_statistics.relative_haplotype_usage(3,second_matches,first_matches)

#%%

space_gap = 8
initp = initial_transition_probabilities(test_haps,space_gap=space_gap)

block_likes =  multiprocess_all_block_likelihoods(all_likelihoods,all_sites,test_haps)

#%%
start = time.time()
final_probs = calculate_hap_transition_probabilities(all_likelihoods,
        all_sites,test_haps,max_num_iterations=20,space_gap=space_gap)
print(time.time()-start)

#%%
start = time.time()
updated_probs = get_updated_transition_probabilities(
    all_likelihoods,
    all_sites,
    test_haps,
    block_likes,
    final_probs,
    space_gap=space_gap,
    minimum_transition_log_likelihood=-10)
print(time.time()-start)

#%%
start = time.time()
final_mesh = generate_transition_probability_mesh(all_likelihoods,
        all_sites,test_haps)
print(time.time()-start)
#%%
start = time.time()
main_haplotype = convert_mesh_to_haplotype(all_sites,
        all_likelihoods,test_haps,final_mesh)
print(time.time()-start)