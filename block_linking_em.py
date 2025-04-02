import numpy as np
import math
import time
from multiprocess import Pool
import itertools
import seaborn as sns

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
            analysis_utils.get_sample_data_at_sites_multiple(full_samples_data,
            sample_sites,haps_data[i][0]),
            haps_data[i]),
        zip(range(len(haps_data))))
    
    return full_blocks_likelihoods
    
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
                analysis_utils.get_sample_data_at_sites(sample_data,
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
                analysis_utils.get_sample_data_at_sites(sample_data,
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
                analysis_utils.get_sample_data_at_sites_multiple(full_samples_data,
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
    
    overall_likelihoods = []
    
    num_vals = len(forward_nums[0])
    
    for i in range(len(forward_nums)):
        sample_vals = forward_nums[i][num_vals-1]
        
        max_likelihood_pair = max(sample_vals,key=sample_vals.get)
        
        max_likelihood_val = sample_vals[max_likelihood_pair]
        
        overall_likelihoods.append(max_likelihood_val)
    
    return (overall_likelihoods,forward_nums)

def get_all_data_backward_probs(full_samples_data,sample_sites,
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
                analysis_utils.get_sample_data_at_sites_multiple(full_samples_data,
                sample_sites,haps_data[i][0]),
                haps_data[i]),
            zip(range(len(haps_data))))
        
    all_block_likelihoods = []
    
    for i in range(len(full_samples_data)):
        all_block_likelihoods.append(
            [full_blocks_likelihoods[x][i] for x in range(len(full_blocks_likelihoods))])
        
    backward_nums = list(itertools.starmap(
        lambda i : get_full_probs_backward(full_samples_data[i],
                    sample_sites,
                    haps_data,
                    bidirectional_transition_probs,
                    all_block_likelihoods[i],
                    space_gap=space_gap),
        zip(range(len(full_samples_data)))))
    
    overall_likelihoods = []
    
    for i in range(len(backward_nums)):
        sample_vals = backward_nums[i][0]
        
        max_likelihood_pair = max(sample_vals,key=sample_vals.get)
        
        max_likelihood_val = sample_vals[max_likelihood_pair]
        
        overall_likelihoods.append(max_likelihood_val)
    
    return (overall_likelihoods,backward_nums)

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
    
    full_transitions_likelihoods_forwards = {}
    full_transitions_likelihoods_backwards = {}
    
    new_transition_probs_forward = {}
    new_transition_probs_backwards = {}
    
    
    #Calculate overall transition likelihoods going forwards
    for i in range(len(haps_data)-space_gap):
        next_bundle = i+space_gap
        
        first_haps = haps_data[i][3]
        second_haps = haps_data[next_bundle][3]
        
        transitions_likelihoods = {}
        
        likelihood_lists = {}
        
        for first in first_haps.keys():
            for second in second_haps.keys():
                likelihood_lists[(first,second)] = []
                
        for s in range(len(samples_probs)):
            data_here = samples_probs[s]
            
            sample_numbers = {}
            
            for first in first_haps.keys():
                for second in second_haps.keys():
                
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
                    

                    pair_transition_likelihood = analysis_utils.add_log_likelihoods(lower_comb)
                    
                    sample_numbers[(first,second)] = pair_transition_likelihood
                    
                      
            total_sample_loglikeli = analysis_utils.add_log_likelihoods(list(sample_numbers.values()))
            
            normalized_sample_numbers = {}
            for k in sample_numbers.keys():
                normalized_sample_numbers[k] = sample_numbers[k]-total_sample_loglikeli
                
            for k in normalized_sample_numbers.keys():
                likelihood_lists[k].append(normalized_sample_numbers[k])
            
        
        for k in likelihood_lists.keys():
            
            #Get prior value for the transition (i,first)->(next_bundle,second)
            # transition_prior_key = ((i,k[0]),(next_bundle,k[1]))
            # transition_prior_value =  math.log(current_transition_probs[0][i][transition_prior_key])
            # transition_prior_value = 0 #Debugging value to zero out impact of this transiton
        
            transitions_likelihoods[((i,k[0]),(next_bundle,k[1]))] = analysis_utils.add_log_likelihoods(likelihood_lists[k])
            
        full_transitions_likelihoods_forwards[i] = transitions_likelihoods
        
    
    #Now do the same thing going backwards
    for i in range(len(haps_data)-1,space_gap-1,-1):
        next_bundle = i-space_gap
        
        first_haps = haps_data[i][3]
        second_haps = haps_data[next_bundle][3]
        
        transitions_likelihoods = {}
        
        likelihood_lists = {}
        
        for first in first_haps.keys():
            for second in second_haps.keys():
                likelihood_lists[(first,second)] = []
                
        for s in range(len(samples_probs)):
            data_here = samples_probs[s]
            
            sample_numbers = {}
            
            for first in first_haps.keys():
                for second in second_haps.keys():
                
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
                            transition_value = math.log(current_transition_probs[1][i][transition_key])
                                
                            
                            adding = data_here[0][i][current_key]+data_here[1][next_bundle][next_key]+transition_value
                                
                            lower_comb.append(adding)
                    

                    pair_transition_likelihood = analysis_utils.add_log_likelihoods(lower_comb)
                    
                    sample_numbers[(first,second)] = pair_transition_likelihood
                    
                      
            total_sample_loglikeli = analysis_utils.add_log_likelihoods(list(sample_numbers.values()))
            
            normalized_sample_numbers = {}
            for k in sample_numbers.keys():
                normalized_sample_numbers[k] = sample_numbers[k]-total_sample_loglikeli
                
            for k in normalized_sample_numbers.keys():
                likelihood_lists[k].append(normalized_sample_numbers[k])
            
        
        for k in likelihood_lists.keys():
            
            #Get prior value for the transition (i,first)->(next_bundle,second)
            # transition_prior_key = ((i,k[0]),(next_bundle,k[1]))
            # transition_prior_value =  math.log(current_transition_probs[1][i][transition_prior_key])
            # transition_prior_value = 0 #Debugging value to zero out impact of this transiton
        
            transitions_likelihoods[((i,k[0]),(next_bundle,k[1]))] = analysis_utils.add_log_likelihoods(likelihood_lists[k])
            
        full_transitions_likelihoods_backwards[i] = transitions_likelihoods
    
    
    #Now we build our new transition probabilities going forwards
    for i in range(len(haps_data)-space_gap):
        overall_likelihood_forward_dict = {}
        
        next_bundle = i+space_gap
        
        first_haps = haps_data[i][3]
        second_haps = haps_data[next_bundle][3]
        
        for first in first_haps.keys():
            overall_likelihood_forward_dict[(i,first)] = []
        
        for k in full_transitions_likelihoods_forwards[i].keys():
            first_part = k[0]
            overall_likelihood_forward_dict[first_part].append(full_transitions_likelihoods_forwards[i][k])
        
        for first_part in overall_likelihood_forward_dict.keys():
            overall_likelihood_forward_dict[first_part] = analysis_utils.add_log_likelihoods(list(overall_likelihood_forward_dict[first_part]))
        
        final_non_norm_forward_likelihoods = {}
        
        for k in full_transitions_likelihoods_forwards[i].keys():
            first_part = k[0]
            final_non_norm_forward_likelihoods[k] = math.exp(max(full_transitions_likelihoods_forwards[i][k]-overall_likelihood_forward_dict[first_part],minimum_transition_log_likelihood))
        
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
        overall_likelihood_backwards_dict = {}
        
        next_bundle = i-space_gap
        
        first_haps = haps_data[i][3]
        second_haps = haps_data[next_bundle][3]
        
        for first in first_haps.keys():
            overall_likelihood_backwards_dict[(i,first)] = []
        
        for k in full_transitions_likelihoods_backwards[i].keys():
            first_part = k[0]
            overall_likelihood_backwards_dict[first_part].append(full_transitions_likelihoods_backwards[i][k])
        
        for first_part in overall_likelihood_backwards_dict.keys():
            overall_likelihood_backwards_dict[first_part] = analysis_utils.add_log_likelihoods(list(overall_likelihood_backwards_dict[first_part]))
        
        final_non_norm_backwards_likelihoods = {}
        
        for k in full_transitions_likelihoods_backwards[i].keys():
            first_part = k[0]
            final_non_norm_backwards_likelihoods[k] = math.exp(max(full_transitions_likelihoods_backwards[i][k]-overall_likelihood_backwards_dict[first_part],minimum_transition_log_likelihood))
        
        final_backwards_likelihoods = {}
        
        for first in first_haps.keys():
            probs_sum = 0
            for second in second_haps.keys():
                keyname = ((i,first),(next_bundle,second))
                probs_sum += final_non_norm_backwards_likelihoods[keyname]
            for second in second_haps.keys():
                keyname = ((i,first),(next_bundle,second))
                final_backwards_likelihoods[keyname] = final_non_norm_backwards_likelihoods[keyname]/probs_sum
         
        new_transition_probs_backwards[i] = final_backwards_likelihoods
    
    return [new_transition_probs_forward,new_transition_probs_backwards]

def smoothen_probs(old_probs,new_probs,alpha):
    """
    Takes in two lists of transition probabilities and returns a 
    new combined list of transition probabilities where the
    combined probability for each transition is
    (1-alpha)*old_probability+alpha*new_probability
    """
    combined_probs = [{},{}]
    
    for i in range(2):
        for block in old_probs[i].keys():
            combined_probs[i][block] = {}
            
            old_dict = old_probs[i][block]
            new_dict = new_probs[i][block]
            
            for k in old_dict.keys():
                combined_probs[i][block][k] = (1-alpha)*old_dict[k]+alpha*new_dict[k]
    
    return combined_probs
            

def calculate_hap_transition_probabilities(full_samples_data,
                                          sample_sites,
                                          haps_data,
                                          full_blocks_likelihoods=None,
                                          max_num_iterations=10,
                                          space_gap=1,
                                          min_cutoff_change=0.00000001,
                                          learning_rate=1,
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
    
    learning_rate is a parameter between [0,1] which controls how much we update
    our probabilities at each step, 1 corresponds to a full EM update while 0
    corresponds to no change.
    """
    start_probs = initial_transition_probabilities(haps_data,space_gap=space_gap)

    probs_list = [start_probs]
    
    if full_blocks_likelihoods == None:
        #Calculate site data and underlying likelihoods for each sample
        #and possible pair making up that sample at each block
        processing_pool = Pool(8)
    
        full_blocks_likelihoods = processing_pool.starmap(
            lambda i: get_all_block_likelihoods(
                analysis_utils.get_sample_data_at_sites_multiple(full_samples_data,
                sample_sites,haps_data[i][0]),
                haps_data[i]),
            zip(range(len(haps_data))))
    
    for i in range(max_num_iterations):
        
        old_probs = probs_list[-1]
        
        new_probs = get_updated_transition_probabilities(
            full_samples_data,
            sample_sites,
            haps_data,
           old_probs,
            full_blocks_likelihoods,
            space_gap=space_gap,
            minimum_transition_log_likelihood=minimum_transition_log_likelihood)
        
        combined_probs = smoothen_probs(old_probs,new_probs,learning_rate)
        
        probs_list.append(combined_probs)
        
        max_diff = 0
        
        for j in [0,1]:
            testing_currently = new_probs[j]
            testing_last = old_probs[j]

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
    
    return probs_list[-1]

def generate_transition_probability_mesh(full_samples_data,
                                         sample_sites,
                                         haps_data,
                                         max_num_iterations=10,
                                         minimum_transition_log_likelihood=-10,
                                         learning_rate=1):
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
            analysis_utils.get_sample_data_at_sites_multiple(full_samples_data,
            sample_sites,haps_data[i][0]),
            haps_data[i]),
        zip(range(len(haps_data))))
    
    
    mesh_dict = {}
    
    num_sqrs = math.floor(math.sqrt(len(haps_data)-1))
    
    sqrs = [i**2 for i in range(1,num_sqrs+1)]

    mesh_list = processing_pool.starmap(lambda x: calculate_hap_transition_probabilities(full_samples_data,
    sample_sites,
    haps_data,
    full_blocks_likelihoods=full_blocks_likelihoods,
    max_num_iterations=max_num_iterations,
    space_gap=x,
    minimum_transition_log_likelihood=minimum_transition_log_likelihood,
    learning_rate=learning_rate),
    zip(sqrs))
    
    for i in range(len(sqrs)):
        mesh_dict[sqrs[i]] = mesh_list[i]
    
    return mesh_dict

def forward_hap_construction_update(current_full_hap,haps_data,full_mesh):
    """
    Given a full block haplotype computes an updated block haplotype by running a forward pass, 
    choosing the hap at each step which is most likely present based on the input block
    haplotype and the under construction block haplotype
    """
    return_hap = [current_full_hap[0]]
    
    for i in range(1,len(haps_data)):
        available_haps = haps_data[i][3]
        
        available_haps_probs = {k: [] for k in available_haps.keys()}
        
        backward_base = 1
        backward_subtract = 1
        
        while backward_subtract <= i:
            
            earlier_index = i-backward_subtract
            
            earlier_block_value = return_hap[earlier_index]
            
            mesh_probs = full_mesh[backward_subtract][0][earlier_index]
            
            subtract_scaling = math.sqrt(backward_subtract)
            
            for k in available_haps.keys():
                mesh_key = ((earlier_index,earlier_block_value),(i,k))
                log_prob = math.log(mesh_probs[mesh_key])
                available_haps_probs[k].append(log_prob*subtract_scaling)            
            
            backward_base += 1
            backward_subtract = backward_base**2
        
        forward_base = 1
        forward_subtract = 1
        
        while i+forward_subtract <= len(haps_data)-1:
            
            later_index = i+forward_subtract
            
            later_block_value = current_full_hap[later_index]
            
            mesh_probs = full_mesh[forward_subtract][1][later_index]
            
            subtract_scaling = math.sqrt(forward_subtract)
            
            for k in available_haps.keys():
                mesh_key = ((later_index,later_block_value),(i,k))
                log_prob = math.log(mesh_probs[mesh_key])
                available_haps_probs[k].append(log_prob*subtract_scaling)
            
            forward_base += 1
            forward_subtract = forward_base**2
            
        for k in available_haps_probs.keys():
            available_haps_probs[k] = sum(available_haps_probs[k])
            
        best_new_hap = max(available_haps_probs, key=available_haps_probs.get)
            
        return_hap.append(best_new_hap)
    
    return return_hap

def backward_hap_construction_update(current_full_hap,haps_data,full_mesh):
    """
    Given a full block haplotype computes an updated block haplotype by running a backward pass, 
    choosing the hap at each step which is most likely present based on the input block
    haplotype and the under construction block haplotype
    """
    return_hap = [current_full_hap[-1]]
    
    for i in range(len(haps_data)-2,-1,-1):
        available_haps = haps_data[i][3]
        cur_position = len(haps_data)-1-i
        
        available_haps_probs = {k: [] for k in available_haps.keys()}
        
        forward_base = 1
        forward_subtract = 1
        
        while forward_subtract <= cur_position:
            
            #For indexing in our partially constructed second pass hap
            earlier_index = cur_position-forward_subtract
            
            #For indexing throuugh our mesh dict which only goes from forward to end
            later_index = i+forward_subtract
            
            earlier_block_value = return_hap[earlier_index]
            
            mesh_probs = full_mesh[forward_subtract][1][later_index]
            
            subtract_scaling = math.sqrt(forward_subtract)
            
            for k in available_haps.keys():
                mesh_key = ((later_index,earlier_block_value),(i,k))
                log_prob = math.log(mesh_probs[mesh_key])
                available_haps_probs[k].append(log_prob*subtract_scaling)
            
            forward_base += 1
            forward_subtract = forward_base**2
        
        backward_base = 1
        backward_subtract = 1
        
        while backward_subtract <= i:

            #For indexing through 
            lookback_index = i-backward_subtract
            
            lookback_block_value = current_full_hap[lookback_index]
            
            mesh_probs = full_mesh[backward_subtract][0][lookback_index]
            
            subtract_scaling = math.sqrt(backward_subtract)
            
            for k in available_haps.keys():
                mesh_key = ((lookback_index,lookback_block_value),(i,k))
                log_prob = math.log(mesh_probs[mesh_key])
                available_haps_probs[k].append(log_prob*subtract_scaling)
            
            backward_base += 1
            backward_subtract = backward_base**2
        
        for k in available_haps_probs.keys():
            available_haps_probs[k] = sum(available_haps_probs[k])
        
        best_new_hap = max(available_haps_probs, key=available_haps_probs.get)
        
        return_hap.append(best_new_hap)
        
    #Since we build the second pass long haplotype in reverse, reverse the result to get the forward haplotype
    return_hap = return_hap[::-1]
    
    return return_hap

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
    
    samples_first_restricted = analysis_utils.get_sample_data_at_sites_multiple(
        full_samples_data,full_sites,first_block_sites)
    
    matches = hap_statistics.match_best(first_block_haps,
                samples_first_restricted,first_keep_flags)
    
    
    best_first_match = max(matches[1], key=matches[1].get)
    
    first_pass_hap = [best_first_match]
    
    
    for i in range(1,len(haps_data)):
        available_haps = haps_data[i][3]
        
        base = 1
        cur_subtract = 1
        
        available_haps_probs = {k: [] for k in available_haps.keys()}
        
        while cur_subtract <= i:
            earlier = i-cur_subtract
            
            earlier_block_value = first_pass_hap[earlier]
            
            mesh_probs = full_mesh[cur_subtract][0][earlier]
            
            subtract_scaling = math.sqrt(cur_subtract)
            
            for k in available_haps.keys():
                mesh_key = ((earlier,earlier_block_value),(i,k))
                log_prob = math.log(mesh_probs[mesh_key])
                available_haps_probs[k].append(log_prob*subtract_scaling)
        
            base += 1
            cur_subtract = base**2
        
        for k in available_haps_probs.keys():
            available_haps_probs[k] = sum(available_haps_probs[k])
        
        best_new_hap = max(available_haps_probs, key=available_haps_probs.get)
        
        first_pass_hap.append(best_new_hap)
        
    second_pass_hap = backward_hap_construction_update(first_pass_hap, haps_data, full_mesh)
    third_pass_hap = forward_hap_construction_update(second_pass_hap, haps_data, full_mesh)
        
    #fourth_pass_hap = backward_hap_construction_update(third_pass_hap, haps_data, full_mesh)
    #fifth_pass_hap = forward_hap_construction_update(fourth_pass_hap, haps_data, full_mesh)
        
    
    combined_positions = []
    combined_haplotype = []
    
    for i in range(len(haps_data)):
        combined_positions.extend(haps_data[i][0])
        combined_haplotype.extend(haps_data[i][3][third_pass_hap[i]])
    
    combined_positions = np.array(combined_positions)
    combined_haplotype = np.array(combined_haplotype)
    
    print(first_pass_hap)
    print(second_pass_hap)
    print(third_pass_hap)
    
    print(np.where(np.array(third_pass_hap) != np.array(first_pass_hap)))
        
    return (combined_positions,combined_haplotype,third_pass_hap)   

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
        main_hap_value = analysis_utils.get_sample_data_at_sites(main_haplotype[1],full_sites,haps_data[j][0])
        main_hap_at_blocks.append(main_hap_value)
        
    for i in range(len(full_samples_data)):
        
        sample_data = full_samples_data[i]
        
        #Boolean array which will store the blocks where main_haplotype
        #seems to be a constitutent of the sample under consideration
        hap_matchings = [False for _ in range(len(haps_data))]
        
        for j in range(len(haps_data)):
            sample_block_data = analysis_utils.get_sample_data_at_sites(sample_data,full_sites,haps_data[j][0])
            main_hap_at_block = main_hap_at_blocks[j]
            
            hap_difference = analysis_utils.get_diff_wrongness(sample_block_data,main_hap_at_block,haps_data[j][1])
            
            print(j)
            print(hap_difference[1])
            
            if hap_difference[1] < max_hap_wrongness_for_match:
                hap_matchings[j] = True
                
def attempt_subtraction(usage_hap,haps_data,full_best_matches):
    """    
    This function tries to subtract off the haplotype given by usage_hap
    from each sample making up haps_data
    
    usage_hap is a block haplotype level representation of a (partial)
    haplotype, being a list of the block haplotype used for each block,
    with -1 used at each block where we are unsure
    
    haps_data is the data for all the block haps in the standard format
    of each element being (sites,keep_flags,read_counts_each_sample,haps)

    full_best_matches is a list where each element contrains the best matches
    for all samples for that particular index    
    """
    
    sample_hits = []
    num_samples = len(haps_data[0][2])
    num_haps = len(haps_data)
    
    for i in range(num_samples):
        matching = []
        for j in range(num_haps):
            haps_here = full_best_matches[j][0][i][0]
            if usage_hap[j] == -1:
                matching.append(-1)
            elif usage_hap[j] in haps_here:
                matching.append(1)
            else:
                matching.append(0)
        sample_hits.append(matching)
    sample_hits = np.array(sample_hits)
    
    sample_similars = np.zeros((num_haps,num_haps))
    
    start = time.time()
    for i in range(num_haps):
        for j in range(i,num_haps):
            commonality_ratio = len(np.where(
                (sample_hits[:,i] == 1) & (sample_hits[:,j] == 1))[0])/num_samples
            sample_similars[i,j] = commonality_ratio
            sample_similars[j,i] =  commonality_ratio
    print(time.time()-start)
    
    sns.heatmap(sample_similars)
    
        
    breakpoint()
    
#%%

space_gap = 49

initp = initial_transition_probabilities(test_haps,space_gap=space_gap)
block_likes =  multiprocess_all_block_likelihoods(all_likelihoods,all_sites,test_haps)

#%%
start = time.time()
final_probs = calculate_hap_transition_probabilities(all_likelihoods,
        all_sites,test_haps,max_num_iterations=20,space_gap=space_gap,
        minimum_transition_log_likelihood=-100,learning_rate=0.5)
print(time.time()-start)

#%%
start = time.time()
updated_probs = get_updated_transition_probabilities(
    all_likelihoods,all_sites,test_haps,
    initp,block_likes,space_gap=space_gap,
    minimum_transition_log_likelihood=-100)
print(time.time()-start)
#%%
start = time.time()
final_mesh = generate_transition_probability_mesh(all_likelihoods,
        all_sites,test_haps,max_num_iterations=20,learning_rate=0.5)
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
for i in range(len(test_haps)):
    sites_here = test_haps[i][0]
    
    real_data_here = analysis_utils.get_sample_data_at_sites(
        haplotype_data[5],all_sites,sites_here)
    
    block_used_here = test_haps[i][3][main_haplotype[2][i]]
    
    print(i,analysis_utils.calc_perc_difference(real_data_here,
        block_used_here,calc_type="haploid"))

#%%
sites_test = test_haps[20][0]
real_data_test = analysis_utils.get_sample_data_at_sites(haplotype_data[5],all_sites,sites_test)

for hap in test_haps[20][3].keys():
    print(hap,analysis_utils.calc_perc_difference(real_data_test,
        test_haps[20][3][hap],calc_type="haploid"))

#%%
real_dat =  test_haps[21][2][306]
real_probs = analysis_utils.reads_to_probabilities(np.array([real_dat]))[1]
print(analysis_utils.get_diff_wrongness(
    real_probs[0], test_haps[21][3][1]))

comb_check = analysis_utils.combine_haploids(test_haps[21][3][0],test_haps[21][3][1])

#%%
comp0 = analysis_utils.get_best_block_haps_for_long_hap(
    haplotype_data[2], all_sites, test_haps)
comp1 = analysis_utils.get_best_block_haps_for_long_hap(
    haplotype_data[1], all_sites, test_haps)
comp5 = analysis_utils.get_best_block_haps_for_long_hap(
    haplotype_data[5], all_sites, test_haps)
#%%
print(np.where(np.array(comp5[0]) != main_haplotype[2]))
#%%
t = 57
mat = main_haplotype[2][t]
mat = comp1[0][t]
lpro = []
for x in [1,4,9,16,25,36,49,64]:
    v = t-x
    
    if v >= 0:    
        ma = main_haplotype[2][v]
        key = ((v,ma),(t,mat))
        print(x,v,final_mesh[x][0][v][key])
    
#%%
all_best_matches = hap_statistics.get_best_matches_all_blocks(test_haps)
#%%
sub = attempt_subtraction(comp0[0],test_haps,all_best_matches)
#%%
rel_usg = hap_statistics.relative_haplotype_usage(
    2, all_best_matches[8], all_best_matches[57])
print(rel_usg)
#%%
start = time.time()
all_haps = get_other_hap_fragments(all_likelihoods,
        all_sites,test_haps,main_haplotype)
print(time.time()-start)
