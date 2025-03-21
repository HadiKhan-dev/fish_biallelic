import numpy as np
import math
import time
from multiprocess import Pool
import itertools

import analysis_utils
import block_haplotypes
import hap_statistics
#%%
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
    
    assert len(haps_data[0]) == len(sample_data), f"Number of sites in sample {len(sample_data)} don't match number of sites in haps {len(haps_data[0])}"
    
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
    
def get_full_probs_forward(sample_data,
                           sample_sites,
                           haps_data,
                           bidirectional_transition_probs,
                           full_blocks_likelihoods = None,
                           space_gap=1):
    """
    Compute the forward step in the forward-backward inference
    algorithm for the HMM where we observe our data for a single
    sample given underlying transition probabilities
    
    space_gap is the number of blocks we jump over to get 
    consecutive transitions from, by default this is equal to 1
    
    The value of space_gap must correspond to the same space gap the
    transition_probs are for
    """
    
    if full_blocks_likelihoods == None:
        #Calculate site data and underlying likelihoods for each sample
        #and possible pair making up that sample at each block
        processing_pool = Pool(8)
    
        full_blocks_likelihoods = processing_pool.starmap(
            lambda i: get_block_likelihoods(
                get_sample_data_at_sites(sample_data,
                sample_sites,haps_data[i][0]),
                haps_data[i]),
            zip(range(len(haps_data))))
    

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
                
                #Account for the fact that we need to halve all probabilities if both haps
                #are the same at our location of interest
                if hap_pair[0] == hap_pair[1]:
                    dividing_val = 2
                else:
                    dividing_val = 1
                    
                direct_likelihood = block_likelihoods[hap_pair]
                
                total_sum_probs = [] #List which will contain the likelihoods for each possible earlier pair (a,b) which could transition to our hap at this step
                
                for earlier_first in earlier_haps[3].keys():
                    for earlier_second in earlier_haps[3].keys():
                        
                        if earlier_first == earlier_second:
                            doubling_correction = 2
                        else:
                            doubling_correction = 1 
                            
                        #Flip around our naming scheme for lookups if we are putting a higher index hap before a lower index one
                        if earlier_second < earlier_first:
                            lookup_first = earlier_second
                            lookup_second = earlier_first
                        else:
                            lookup_first = earlier_first
                            lookup_second = earlier_second

                        earlier_dip_name = ((earlier_block,lookup_first),
                                            (earlier_block,lookup_second))
                        
                        transition_prob = transition_probs[earlier_block][((earlier_block,earlier_first),(i,hap_pair[0]))]* \
                        transition_probs[earlier_block][((earlier_block,earlier_second),(i,hap_pair[1]))]                                                                                                       

                        transition_prob = transition_prob*doubling_correction/dividing_val

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
                            bidirectional_transition_probs,
                            full_blocks_likelihoods = None,
                            space_gap=1):
    """
    Compute the backward step in the forward-backward inference
    algorithm for the HMM where we observe our data for a single
    sample given underlying transition probabilities
    
    space_gap is the number of blocks we jump over to get 
    consecutive transitions from, by default this is equal to 1
    
    The value of space_gap must correspond to the same space gap the
    transition_probs are for
    """
    
    if full_blocks_likelihoods == None:
        #Calculate site data and underlying likelihoods for each sample
        #and possible pair making up that sample at each block
        processing_pool = Pool(8)
    
        full_blocks_likelihoods = processing_pool.starmap(
            lambda i: get_block_likelihoods(
                get_sample_data_at_sites(sample_data,
                sample_sites,haps_data[i][0]),
                haps_data[i]),
            zip(range(len(haps_data))))
    
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
                
                #Account for the fact that we need to halve all probabilities if both haps
                #are the same at our location of interest
                if hap_pair[0] == hap_pair[1]:
                    dividing_val = 2
                else:
                    dividing_val = 1
                    
                direct_likelihood = block_likelihoods[hap_pair]
                
                total_sum_probs = [] #List which will contain the likelihoods for each possible earlier pair (a,b) which could transition to our hap at this step
                
                for earlier_first in earlier_haps[3].keys():
                    for earlier_second in earlier_haps[3].keys():
                        
                        if earlier_first == earlier_second:
                            doubling_correction = 2
                        else:
                            doubling_correction = 1 
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

                        transition_prob = transition_prob*doubling_correction/dividing_val

                        earlier_log_likelihood = earlier_likelihoods[earlier_dip_name]
                        
                        combined_log_likelihood = earlier_log_likelihood+math.log(transition_prob)
                        
                        total_sum_probs.append(combined_log_likelihood)
                
                combined_prob = analysis_utils.add_log_likelihoods(total_sum_probs)+direct_likelihood
                likelihoods[new_name] = combined_prob                            
        
        likelihood_numbers[i] = likelihoods
   
    return likelihood_numbers

def get_all_data_forward_probs(full_samples_data,sample_sites,
                               haps_data,
                               bidirectional_transition_probs,
                               full_blocks_likelihoods=None,
                               space_gap=1):
    
    
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
    
    return forward_nums
    
def get_updated_transition_probabilities(full_samples_data,
                                         sample_sites,
                                         haps_data,
                                         current_transition_probs,
                                         full_blocks_likelihoods,
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
        lambda i : get_full_probs_forward(full_samples_data[i],
                    sample_sites,
                    haps_data,
                    current_transition_probs,
                    all_block_likelihoods[i],
                    space_gap=space_gap),
        zip(range(len(full_samples_data)))))
    backward_nums = list(itertools.starmap(
        lambda i : get_full_probs_backward(full_samples_data[i],
                    sample_sites,
                    haps_data,
                    current_transition_probs,
                    all_block_likelihoods[i],
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
                                
                            
                            adding = data_here[0][i][current_key]+data_here[1][next_bundle][next_key]+transition_value
                                
                            lower_comb.append(adding)
                    

                    sample_likelihood = analysis_utils.add_log_likelihoods(lower_comb)
                    
                    tots_comb.append(sample_likelihood)
                    
                all_sample_combined_likelihood = analysis_utils.add_log_likelihoods(tots_comb)
                
                
                transitions_likelihoods[((i,first),(next_bundle,second))] = all_sample_combined_likelihood
        
        full_transitions_likelihoods[i] = transitions_likelihoods

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
                                         max_num_iterations=10,
                                         space_gap=1,
                                         min_cutoff_change=0.001,
                                         averaging_lookback=3,
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
    
    averaging_lookback is the number of previous steps the program averages
    the probabilities over to come up with the latest final probabilities
    at each step. Using a higher value for this will smoothen out the probability
    changes per step more.
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
        
        print("Max diff:",space_gap,i,f"{max_diff :.3f}")
        print()
        if max_diff < min_cutoff_change:
            #print(f"Exiting_early {len(probs_list)}")
            break
    
    return probs_list

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
        
        available_haps_probs = {k: [] for k in available_haps.keys()}
        
        while cur_subtract <= i:
            earlier = i-cur_subtract
            
            earlier_block_value = first_pass_hap[earlier]
            
            mesh_probs = full_mesh[cur_subtract][0][earlier]
            
            for k in available_haps.keys():
                mesh_key = ((earlier,earlier_block_value),(i,k))
                log_prob = math.log(mesh_probs[mesh_key])
                available_haps_probs[k].append(log_prob)
        
            cur_subtract *= 2
        
        for k in available_haps_probs.keys():
            available_haps_probs[k] = sum(available_haps_probs[k])
        
        best_new_hap = max(available_haps_probs, key=available_haps_probs.get)
        
        first_pass_hap.append(best_new_hap)
        
    #This is the updated hap we are constructing in reverse order
    second_pass_hap = [first_pass_hap[-1]]
    
    for i in range(len(haps_data)-2,-1,-1):
        available_haps = haps_data[i][3]
        cur_position = len(haps_data)-1-i
        
        available_haps_probs = {k: [] for k in available_haps.keys()}
        
        forward_subtract = 1
        
        while forward_subtract <= cur_position:
            
            #For indexing in our partially constructed second pass hap
            earlier_index = cur_position-forward_subtract
            
            #For indexing throuugh our mesh dict which only goes from forward to end
            later_index = i+forward_subtract
            
            earlier_block_value = second_pass_hap[earlier_index]
            
            mesh_probs = full_mesh[forward_subtract][1][later_index]
            
            for k in available_haps.keys():
                mesh_key = ((later_index,earlier_block_value),(i,k))
                log_prob = math.log(mesh_probs[mesh_key])
                available_haps_probs[k].append(log_prob)
            
            
            forward_subtract *= 2
        
        backward_subtract = 1
        
        while backward_subtract <= i:

            #For indexing through 
            lookback_index = i-backward_subtract
            
            lookback_block_value = first_pass_hap[lookback_index]
            
            mesh_probs = full_mesh[backward_subtract][0][lookback_index]
            
            for k in available_haps.keys():
                mesh_key = ((lookback_index,lookback_block_value),(i,k))
                log_prob = math.log(mesh_probs[mesh_key])
                available_haps_probs[k].append(log_prob)
            
            backward_subtract *= 2
        
        for k in available_haps_probs.keys():
            available_haps_probs[k] = sum(available_haps_probs[k])
        
        best_new_hap = max(available_haps_probs, key=available_haps_probs.get)
        
        second_pass_hap.append(best_new_hap)
        
    #Since we build the second pass long haplotype in reverse, reverse the result to get the forward haplotype
    second_pass_hap = second_pass_hap[::-1]
    
    #Finally build the haplotype we will return
    third_pass_hap = [second_pass_hap[0]]
    
    for i in range(1,len(haps_data)):
        available_haps = haps_data[i][3]
        
        available_haps_probs = {k: [] for k in available_haps.keys()}
        
        backward_subtract = 1
        
        while backward_subtract <= i:
            
            earlier_index = i-backward_subtract
            
            earlier_block_value = third_pass_hap[earlier_index]
            
            mesh_probs = full_mesh[backward_subtract][0][earlier_index]
            
            for k in available_haps.keys():
                mesh_key = ((earlier_index,earlier_block_value),(i,k))
                log_prob = math.log(mesh_probs[mesh_key])
                available_haps_probs[k].append(log_prob)            
            
            backward_subtract *= 2
            
        forward_subtract = 1
        
        while i+forward_subtract <= len(haps_data)-1:
            
            later_index = i+forward_subtract
            
            later_block_value = second_pass_hap[later_index]
            
            mesh_probs = full_mesh[forward_subtract][1][later_index]
            
            for k in available_haps.keys():
                mesh_key = ((later_index,later_block_value),(i,k))
                log_prob = math.log(mesh_probs[mesh_key])
                available_haps_probs[k].append(log_prob)
            
            forward_subtract *= 2
            
        for k in available_haps_probs.keys():
            available_haps_probs[k] = sum(available_haps_probs[k])
            
        best_new_hap = max(available_haps_probs, key=available_haps_probs.get)
            
        third_pass_hap.append(best_new_hap)
        
    combined_positions = []
    combined_haplotype = []
    
    for i in range(len(haps_data)):
        combined_positions.extend(haps_data[i][0])
        combined_haplotype.extend(haps_data[i][3][third_pass_hap[i]])
    
    combined_positions = np.array(combined_positions)
    combined_haplotype = np.array(combined_haplotype)
        
    return (combined_positions,combined_haplotype)   

def get_other_hap_fragments(full_samples_data,
                            full_sites,
                            haps_data,
                            main_haplotype,
                            max_hap_wrongness_for_match=0.02):
    """
    Takes as input the likelihood data for all the sequences as well
    as the full data for a single haplotype and computes further 
    haplotypes from it
    """
    
    newfound_fragments = {0:((0,len(haps_data)-1),main_haplotype)}
    
    main_hap_at_blocks = []
    
    for j in range(len(haps_data)):
        main_hap_value = get_sample_data_at_sites(main_haplotype[1],full_sites,haps_data[j][0])
        main_hap_at_blocks.append(main_hap_value)
        
    for i in range(len(full_samples_data)):
        
        sample_data = full_samples_data[i]
        
        #Boolean array which will store the blocks where main_haplotype
        #seems to be a constitutent of the sample under consideration
        hap_matchings = [False for _ in range(len(haps_data))]
        
        for j in range(len(haps_data)):
            sample_block_data = get_sample_data_at_sites(sample_data,full_sites,haps_data[j][0])
            main_hap_at_block = main_hap_at_blocks[j]
            
            hap_difference = analysis_utils.get_diff_wrongness(sample_block_data,main_hap_at_block,haps_data[j][1])
            
            print(j)
            print(hap_difference[1])
            
            if hap_difference[1] < max_hap_wrongness_for_match:
                hap_matchings[j] = True
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
    final_probs,
    block_likes,
    space_gap=space_gap,
    minimum_transition_log_likelihood=-10)
print(time.time()-start)

#%%
start = time.time()
final_mesh = generate_transition_probability_mesh(all_likelihoods,
        all_sites,test_haps,max_num_iterations=30)
print(time.time()-start)
#%%
start = time.time()
main_haplotype = convert_mesh_to_haplotype(all_likelihoods,
        all_sites,test_haps,final_mesh)
print(time.time()-start)
#%%
for i in range(len(haplotype_data)):
    print(i,analysis_utils.calc_perc_difference(main_haplotype[1],
        haplotype_data[i],calc_type="haploid"))
    print(len(np.where(
        concretify_haps([main_haplotype[1]])[0] !=
        concretify_haps([haplotype_data[i]])[0])[0]))
    print(len(main_haplotype[1]))
    print()
#%%
basic_good = get_full_probs_forward(all_likelihoods[0],all_sites,test_haps,
                           final_mesh[1][20])
#%%
good = get_all_data_forward_probs(all_likelihoods,all_sites,test_haps,
                           final_mesh[1][20])
#%%
first_probs = final_mesh[1][16][0]
second_probs = final_mesh[1][17][0]

for loc in first_probs.keys():
    first_vals = first_probs[loc]
    second_vals = second_probs[loc]
    for tag in first_vals.keys():
        diff = second_vals[tag]-first_vals[tag]
        
        if abs(diff) > 0.2:
            print(loc,tag,f"{first_vals[tag] :.3f}",
                  f"{second_vals[tag] :.3f}",f"{diff : .3f}")
    
#%%
start = time.time()
all_haps = get_other_hap_fragments(all_likelihoods,
        all_sites,test_haps,main_haplotype)
print(time.time()-start)
