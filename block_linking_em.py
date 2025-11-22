import numpy as np
import math
import time
from multiprocess import Pool
import itertools
import copy
import seaborn as sns

import analysis_utils
import block_haplotypes
import simulate_sequences
import hap_statistics

#%%
def find_runs(data, min_length):
    """
    Function which takes as input a list of bools and a minimum length,
    then returns all runs of True of length at least min_length
    """
    runs = []
    run_start = -1  # -1 indicates we are not in a run of Trues

    # Enumerate to get both index and value
    for i, value in enumerate(data):
        if value:  # Current value is True
            if run_start == -1:
                run_start = i  # Start of a new run
        else:  # Current value is False
            if run_start != -1:  # We were in a run, and it just ended
                run_length = i - run_start
                if run_length >= min_length:
                    runs.append((run_start, i))
                run_start = -1 # Reset for the next run

    # After the loop, check if a run was ongoing to the end of the list
    if run_start != -1:
        run_length = len(data) - run_start
        if run_length >= min_length:
            runs.append((run_start, len(data)))
            
    return runs

def initial_transition_probabilities(hap_data,
                                     space_gap=1,
                                     found_haps=[],
                                     found_penalty=0.1):
    """
    Creates a dict of initial equal transition probabilities
    for a list where each element of the list contains info about
    block haps for that block.
    
    space_gap is the number of blocks we jump over at each step
    for calculating the transition probabilities. By default this
    is equal to 1.
    
    found_haps is a list of block level haplotypes which correspond
    to already found haplotypes, if a transition is one which is 
    already present in one of the found haplotypes, its weighting
    is deflated by a factor of found_penalty

    """
    transition_dict_forward = {}
    transition_dict_reverse = {}
    
    for i in range(0,len(hap_data)-space_gap):
        transition_dict_forward[i] = {}
        
        these_haps = hap_data[i][3]
        next_haps = hap_data[i+space_gap][3]
        
        for first_idx in these_haps.keys():
            first_hap_name = (i,first_idx)
            
            for second_idx in next_haps.keys():
                second_hap_name = (i+space_gap,second_idx)
                
                present = False
                
                for fhap in found_haps:
                    if fhap[i] == first_idx and fhap[i+space_gap] == second_idx:
                        present = True
                
                if not present:
                    transition_dict_forward[i][(first_hap_name,second_hap_name)] = 1
                else:
                    transition_dict_forward[i][(first_hap_name,second_hap_name)] = found_penalty
                
    for i in range(len(hap_data)-1,space_gap-1,-1):
        transition_dict_reverse[i] = {}
        
        these_haps = hap_data[i][3]
        next_haps = hap_data[i-space_gap][3]
        
        for first_idx in these_haps.keys():
            first_hap_name = (i,first_idx)
            
            for second_idx in next_haps.keys():
                second_hap_name = (i-space_gap,second_idx)
                
                present = False
                
                for fhap in found_haps:
                    if fhap[i] == first_idx and fhap[i-space_gap] == second_idx:
                        present = True
                
                if not present:
                    transition_dict_reverse[i][(first_hap_name,second_hap_name)] = 1
                else:
                    transition_dict_reverse[i][(first_hap_name,second_hap_name)] = found_penalty
    
    #At this point we have the unscaled transition probabilities, we now scale them
    scaled_dict_forward = {}
    scaled_dict_reverse = {}
    
    for idx in transition_dict_forward.keys():
        scaled_dict_forward[idx] = {}
        
        start_dict = {}
        for s in transition_dict_forward[idx].keys():
            if s[0] not in start_dict.keys():
                start_dict[s[0]] = 0
            start_dict[s[0]] += transition_dict_forward[idx][s]
        
        for s in transition_dict_forward[idx].keys():
            scaled_dict_forward[idx][s] = transition_dict_forward[idx][s]/start_dict[s[0]]
        
    for idx in transition_dict_reverse.keys():
        scaled_dict_reverse[idx] = {}
        
        start_dict = {}
        for s in transition_dict_reverse[idx].keys():
            if s[0] not in start_dict.keys():
                start_dict[s[0]] = 0
            start_dict[s[0]] += transition_dict_reverse[idx][s]
        
        for s in transition_dict_reverse[idx].keys():
            scaled_dict_reverse[idx][s] = transition_dict_reverse[idx][s]/start_dict[s[0]]
        
    return [scaled_dict_forward,scaled_dict_reverse]

def get_block_likelihoods(sample_data,
                          haps_data,
                          log_likelihood_base=math.e**2,
                          min_per_site_log_likelihood=-100,
                          normalize=True,                          
                          ):
    """
    Get the log-likelihoods for each combination of haps matching the sample
    
    sample_ploidy is a list of length the number of sites in the block
    which indicates whether this particular block is made of two, one or zero haplotypes
    
    haps_data must be the full list of haplotype information for a block, including
    the positions,keep_flags,read_counts and haplotype info elements
    
    For each site we compute the distance of the sample data probabalistic genotype
    from the probabalistic genotype given by combining each pair of haps in haps_data.
    We then get the log_likelihood for that site as
    max(-dist*log(log_likelihood_base),min_per_site_log_likelihood)
    
    If normalize=True then ensures the sum of the probabilities for 
    the sample over all possible pairs adds up to 1
    """
    
    main_sample_data = sample_data[0]
    sample_ploidy = sample_data[1]
    
    assert len(haps_data[0]) == len(main_sample_data), f"Number of sites in sample {len(main_sample_data)} don't match number of sites in haps {len(haps_data[0])}"
    
    
    block_ploidy = sample_ploidy[0]
    
    bool_keepflags = haps_data[1].astype(bool)
    
    ll_dict = {}
    
    ll_list = []
    
    if block_ploidy == 2:
        sample_keep = main_sample_data[bool_keepflags,:]
        
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
                
    elif block_ploidy == 1:
        sample_keep = main_sample_data[bool_keepflags,:2]
        
        for i in haps_data[3].keys():
            use_hap = haps_data[3][i]
            use_keep = use_hap[bool_keepflags,:]
            
            dist = analysis_utils.calc_distance_by_site(sample_keep,use_keep,calc_type="haploid")
            
            bdist = -(dist**2)*math.log(log_likelihood_base)
            combined_logs = np.concatenate([np.array(bdist.reshape(1,-1)),min_per_site_log_likelihood*np.ones((1,len(dist)))])
            
            combined_dist = np.max(combined_logs,axis=0)
            
            total_ll = np.sum(combined_dist)
            
            ll_dict[i] = total_ll
            ll_list.append(total_ll)
            
    else: #This is the block_ploidy = 0 case, here the log likelihood is 0 as we don't have any haps left
        ll_dict[0] = 0
        ll_list.append(0)
            
    if normalize:
        combined_ll = analysis_utils.add_log_likelihoods(ll_list)
        
        for k in ll_dict.keys():
            ll_dict[k] = ll_dict[k]-combined_ll
            
    return ll_dict

def get_all_block_likelihoods(block_samples_data,block_haps):
    """
    Function which calculates the block likelihoods for each 
    sample in block_samples_data
    
    block_samples_data must contain the ploidy for each sample
    """
    sample_likelihoods = []
    sample_ploidies = []
    
    for i in range(len(block_samples_data[0])):
        sample_likelihoods.append(
            get_block_likelihoods([block_samples_data[0][i],
                                  block_samples_data[1][i]],
                                  block_haps))
        sample_ploidies.append(block_samples_data[1][i])
    
    
    return (sample_likelihoods,sample_ploidies)

def multiprocess_all_block_likelihoods(full_samples_data,
                                       sample_sites,
                                       haps_data):
    
    processing_pool = Pool(32)

    full_blocks_likelihoods = processing_pool.starmap(
        lambda i: get_all_block_likelihoods(
            analysis_utils.get_sample_data_at_sites_multiple(full_samples_data,
            sample_sites,haps_data[i][0],ploidy_present=True),
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
        processing_pool = Pool(32)
    
        full_blocks_likelihoods = processing_pool.starmap(
            lambda i: get_block_likelihoods(
                analysis_utils.get_sample_data_at_sites(sample_data,
                sample_sites,haps_data[i][0],ploidy_present=True),
                haps_data[i]),
            zip(range(len(haps_data))))

    transition_probs = bidirectional_transition_probs[0]
    
    likelihood_numbers = {}
    
    #Iterate over the blocks
    for i in range(len(haps_data)):

        (block_likelihoods,ploidy_list) = full_blocks_likelihoods[i]
        
        ploidy_here = ploidy_list[0]

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
        processing_pool = Pool(32)
    
        full_blocks_likelihoods = processing_pool.starmap(
            lambda i: get_block_likelihoods(
                analysis_utils.get_sample_data_at_sites(sample_data,
                sample_sites,haps_data[i][0]),
                haps_data[i]),
            zip(range(len(haps_data))))
    
    transition_probs = bidirectional_transition_probs[1]
    
    likelihood_numbers = {}
    
    for i in range(len(haps_data)-1,-1,-1):
        
        (block_likelihoods,ploidy_list) = full_blocks_likelihoods[i]
        
        ploidy_here = ploidy_list[0]

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
        processing_pool = Pool(32)
    
        full_blocks_likelihoods = processing_pool.starmap(
            lambda i: get_all_block_likelihoods(
                analysis_utils.get_sample_data_at_sites_multiple(full_samples_data,
                sample_sites,haps_data[i][0],ploidy_present=True),
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
        processing_pool = Pool(32)
    
        full_blocks_likelihoods = processing_pool.starmap(
            lambda i: get_all_block_likelihoods(
                analysis_utils.get_sample_data_at_sites_multiple(full_samples_data,
                sample_sites,haps_data[i][0],ploidy_present=True),
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
                                         currently_found_long_haps=[],
                                         found_transition_penalty=0.1,
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
    
    currently_found_long_haps is an optional input used for MAP-EM rather than pure EM,
    it adds the posterior to the likelihoods at the M step, putting a found_transition_penalty
    at each step where the transition is one found in some hap present in currently_found_long_haps
    """
    
    prior_a_posteriori = initial_transition_probabilities(haps_data,space_gap=space_gap,
                        found_haps=currently_found_long_haps,found_penalty=found_transition_penalty)

    full_samples_likelihoods = full_samples_data[0]
    full_samples_ploidies = full_samples_data[1]
    
    all_block_likelihoods = []
    
    for i in range(len(full_samples_likelihoods)):
        all_block_likelihoods.append(
            [(full_blocks_likelihoods[x][0][i],
              full_blocks_likelihoods[x][1][i])
             for x in range(len(full_blocks_likelihoods))])
    
    samples_probs = []
    
    forward_nums = []
    backward_nums = []
    
    #Log Likelihoods going forwards for each of the samples
    forward_nums = list(itertools.starmap(
        lambda i : get_full_probs_forward((full_samples_data[0][i],full_samples_data[1][i]),
                    sample_sites,
                    haps_data,
                    current_transition_probs,
                    all_block_likelihoods[i],
                    space_gap=space_gap),
        zip(range(len(full_samples_likelihoods)))))
    
    #Log Likelihoods going backwards for each of the samples
    backward_nums = list(itertools.starmap(
        lambda i : get_full_probs_backward((full_samples_data[0][i],full_samples_data[1][i]),
                    sample_sites,
                    haps_data,
                    current_transition_probs,
                    all_block_likelihoods[i],
                    space_gap=space_gap),
        zip(range(len(full_samples_likelihoods)))))
    
    #breakpoint()
    
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

            transitions_likelihoods[((i,k[0]),(next_bundle,k[1]))] = analysis_utils.add_log_likelihoods(likelihood_lists[k])
            
        full_transitions_likelihoods_backwards[i] = transitions_likelihoods
    
    
    #Now we build our new transition probabilities going forwards
    for i in range(len(haps_data)-space_gap):
        overall_posterior_forward_dict = {}
        
        next_bundle = i+space_gap
        
        first_haps = haps_data[i][3]
        second_haps = haps_data[next_bundle][3]
        
        transition_likelihoods_forwards_here = full_transitions_likelihoods_forwards[i]
        
        transition_posterior_forwards_here = {}
        
        for k in transition_likelihoods_forwards_here.keys():
            transition_posterior_forwards_here[k] = transition_likelihoods_forwards_here[k]+math.log(prior_a_posteriori[0][i][k])
        
        for first in first_haps.keys():
            overall_posterior_forward_dict[(i,first)] = []
        
        for k in transition_posterior_forwards_here.keys():
            first_part = k[0]
            overall_posterior_forward_dict[first_part].append(transition_posterior_forwards_here[k])
        
        for first_part in overall_posterior_forward_dict.keys():
            overall_posterior_forward_dict[first_part] = analysis_utils.add_log_likelihoods(list(overall_posterior_forward_dict[first_part]))
        
        final_non_norm_forward_posteriors = {}
        
        for k in transition_posterior_forwards_here.keys():
            first_part = k[0]
            final_non_norm_forward_posteriors[k] = math.exp(max(transition_posterior_forwards_here[k]-overall_posterior_forward_dict[first_part],minimum_transition_log_likelihood))
        
        
        #And now normalize
        final_forward_posteriors = {}
        
        for first in first_haps.keys():
            probs_sum = 0
            for second in second_haps.keys():
                keyname = ((i,first),(next_bundle,second))
                probs_sum += final_non_norm_forward_posteriors[keyname]
            for second in second_haps.keys():
                keyname = ((i,first),(next_bundle,second))
                final_forward_posteriors[keyname] = final_non_norm_forward_posteriors[keyname]/probs_sum
         
        
        #Now we account for any priors we may have had
        # final_forward_posteriors = {}
        
        # for first in first_haps.keys():
        #     probs_sum = 0
        #     for second in second_haps.keys():
        #         keyname = ((i,first),(next_bundle,second))
        #         probs_sum += final_non_norm_forward_likelihoods[keyname]*prior_a_posteriori[0][i][keyname]

        new_transition_probs_forward[i] = final_forward_posteriors
        
        
    
    #And then we build our new transition probabilities going backwards
    for i in range(len(haps_data)-1,space_gap-1,-1):
        overall_posterior_backwards_dict = {}
        
        next_bundle = i-space_gap
        
        first_haps = haps_data[i][3]
        second_haps = haps_data[next_bundle][3]
        
        transition_likelihoods_backwards_here = full_transitions_likelihoods_backwards[i]
        
        transition_posterior_backwards_here = {}

        for k in transition_likelihoods_backwards_here.keys():
            transition_posterior_backwards_here[k] = transition_likelihoods_backwards_here[k]+math.log(prior_a_posteriori[1][i][k])
        
        for first in first_haps.keys():
            overall_posterior_backwards_dict[(i,first)] = []
        
        for k in transition_posterior_backwards_here.keys():
            first_part = k[0]
            overall_posterior_backwards_dict[first_part].append(transition_posterior_backwards_here[k])
        
        for first_part in overall_posterior_backwards_dict.keys():
            overall_posterior_backwards_dict[first_part] = analysis_utils.add_log_likelihoods(list(overall_posterior_backwards_dict[first_part]))
        
        final_non_norm_backwards_posteriors = {}
        
        for k in transition_posterior_backwards_here.keys():
            first_part = k[0]
            final_non_norm_backwards_posteriors[k] = math.exp(max(transition_posterior_backwards_here[k]-overall_posterior_backwards_dict[first_part],minimum_transition_log_likelihood))
        
        final_backwards_posteriors = {}
        
        for first in first_haps.keys():
            probs_sum = 0
            for second in second_haps.keys():
                keyname = ((i,first),(next_bundle,second))
                probs_sum += final_non_norm_backwards_posteriors[keyname]
            for second in second_haps.keys():
                keyname = ((i,first),(next_bundle,second))
                final_backwards_posteriors[keyname] = final_non_norm_backwards_posteriors[keyname]/probs_sum
         
        new_transition_probs_backwards[i] = final_backwards_posteriors
    
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
                                          currently_found_long_haps=[],
                                          found_transition_penalty=0.1,
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
        
        processing_pool = Pool(32)
    
        full_blocks_likelihoods = processing_pool.starmap(
            lambda i: get_all_block_likelihoods(
                analysis_utils.get_sample_data_at_sites_multiple(full_samples_data,
                sample_sites,haps_data[i][0],ploidy_present=True),
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
            currently_found_long_haps=currently_found_long_haps,
            found_transition_penalty=found_transition_penalty,
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
                                         currently_found_long_haps=[],
                                         found_transition_penalty=0.1,
                                         minimum_transition_log_likelihood=-10,
                                         learning_rate=1):
    """
    Generates a mesh of transition probabilities where we generate the transition
    probabilities for space_gap = 1,2,4,8,... all the way up to the largest
    power of two less than the number of blocks we have
    """
    
    #Calculate site data and underlying likelihoods for each sample
    #and possible pair making up that sample at each block
    processing_pool = Pool(32)

    full_blocks_likelihoods = processing_pool.starmap(
        lambda i: get_all_block_likelihoods(
            analysis_utils.get_sample_data_at_sites_multiple(full_samples_data,
            sample_sites,haps_data[i][0],ploidy_present=True),
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
    currently_found_long_haps=currently_found_long_haps,
    found_transition_penalty=found_transition_penalty,
    minimum_transition_log_likelihood=minimum_transition_log_likelihood,
    learning_rate=learning_rate),
    zip(sqrs))
    
    for i in range(len(sqrs)):
        mesh_dict[sqrs[i]] = mesh_list[i]
    
    return mesh_dict