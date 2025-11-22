import numpy as np
import math
import time
from multiprocess import Pool
import itertools
import copy
import seaborn as sns

import analysis_utils
import block_haplotypes
import hap_statistics
import simulate_sequences
import block_linking_em

#%%
def forward_hap_construction_update(current_beam_results,
                                     haps_data,
                                     full_mesh,
                                     num_candidates=50):
    """
    Given a full block haplotype computes an updated block haplotype by running a forward pass, 
    choosing the hap at each step which is most likely present based on the input block
    haplotype and the under construction block haplotype
    """
    #List of best candidates so far, for the beam search
    kept_haps = copy.deepcopy(current_beam_results)[:num_candidates]
    
    for i in range(1,len(haps_data)):
        available_haps = haps_data[i][3]
        
        new_cands = []
        new_cands_dict = {}
        
        for j in range(len(kept_haps)):
            
            cur_examine = kept_haps[j] #The hap we are currently examining to extend            
        
            available_haps_probs = {k: [] for k in available_haps.keys()}
        
            backward_base = 1
            backward_subtract = 1
        
            while backward_subtract <= i:
            
                earlier_index = i-backward_subtract
            
                earlier_block_value = cur_examine[0][earlier_index]
            
                mesh_probs = full_mesh[backward_subtract][0][earlier_index]
            
                subtract_scaling = math.sqrt(backward_subtract)
            
                for k in available_haps.keys():
                    mesh_key = ((earlier_index,earlier_block_value),(i,k))
                    log_prob = math.log(mesh_probs[mesh_key])
                    available_haps_probs[k].append(log_prob*subtract_scaling)            
            
                backward_base += 1
                backward_subtract = backward_base**2
        
            forward_base = 1
            forward_add = 1
        
            while i+forward_add <= len(haps_data)-1:
            
                later_index = i+forward_add
            
                later_block_value = cur_examine[0][later_index]
            
                mesh_probs = full_mesh[forward_add][1][later_index]
            
                subtract_scaling = math.sqrt(forward_add)
            
                for k in available_haps.keys():
                    mesh_key = ((later_index,later_block_value),(i,k))
                    log_prob = math.log(mesh_probs[mesh_key])
                    available_haps_probs[k].append(log_prob*subtract_scaling)
            
                forward_base += 1
                forward_add = forward_base**2
            
            for k in available_haps_probs.keys():
                available_haps_probs[k] = sum(available_haps_probs[k])
        
            #For each new candidate block hap to move to create a copy and add to list with its log likelihood
            for k in available_haps_probs.keys():
                cur_copy = copy.deepcopy(cur_examine)
                cur_copy[0][i] = k
                cur_copy[1] += available_haps_probs[k]
                
                if tuple(cur_copy[0]) not in new_cands_dict.keys():
                    new_cands_dict[tuple(cur_copy[0])] = cur_copy[1]
                elif cur_copy[1] > new_cands_dict[tuple(cur_copy[0])]:
                    new_cands_dict[tuple(cur_copy[0])] = cur_copy[1]
            
        new_cands = [[list(x),y] for x,y in new_cands_dict.items()]
            
        #Update the kept_haps for the next stage
        new_cands = sorted(new_cands,key=lambda x:-x[1])
        kept_haps = copy.deepcopy(new_cands)[:num_candidates]
    
    return kept_haps

def backward_hap_construction_update(current_beam_results,
                                     haps_data,
                                     full_mesh,
                                     num_candidates=50):
    """
    Given a set of current beam search results computes an updated block haplotype by running a backward pass, 
    choosing the hap at each step which is most likely present based on the input block
    haplotype and the under construction block haplotype
    
    The first input is a list where each element is a list of 2 elements, the first of which is the 
    block haplotype and the second is its associated log likelihood
    """
    
    #List of best candidates so far, for the beam search
    kept_haps = copy.deepcopy(current_beam_results)[:num_candidates]

    
    for i in range(len(haps_data)-2,-1,-1):
        available_haps = haps_data[i][3]
        cur_position = len(haps_data)-1-i
        
        new_cands_dict = {}
        new_cands = []
        
        for j in range(len(kept_haps)):
            
            cur_examine = kept_haps[j] #The hap we are currently examining to extend            
        
            available_haps_probs = {k: [] for k in available_haps.keys()}
        
            forward_base = 1
            forward_add = 1
        
            #Add forward mesh likelihoods
            while forward_add <= cur_position:
            
                #For indexing in our partially constructed second pass hap
                forward_index = i+forward_add
            
            
                forward_block_value = cur_examine[0][forward_index]
            
                mesh_probs = full_mesh[forward_add][1][forward_index]
            
                subtract_scaling = math.sqrt(forward_add)
            
                for k in available_haps.keys():
                    mesh_key = ((forward_index,forward_block_value),(i,k))
                    log_prob = math.log(mesh_probs[mesh_key])
                    available_haps_probs[k].append(log_prob*subtract_scaling)
            
                forward_base += 1
                forward_add = forward_base**2
        
            backward_base = 1
            backward_subtract = 1
            
            #Add backward mesh likelihoods
            while backward_subtract <= i:

                #For indexing through 
                lookback_index = i-backward_subtract
            
                lookback_block_value = cur_examine[0][lookback_index]
            
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
        
            #For each new candidate block hap to move to create a copy and add to list with its log likelihood
            for k in available_haps_probs.keys():
                cur_copy = copy.deepcopy(cur_examine)
                cur_copy[0][i] = k
                cur_copy[1] += available_haps_probs[k]
                
                if tuple(cur_copy[0]) not in new_cands_dict.keys():
                    new_cands_dict[tuple(cur_copy[0])] = cur_copy[1]
                elif cur_copy[1] > new_cands_dict[tuple(cur_copy[0])]:
                    new_cands_dict[tuple(cur_copy[0])] = cur_copy[1]
            
        new_cands = [[list(x),y] for x,y in new_cands_dict.items()]
        
        #Update the kept_haps for the next stage
        new_cands = sorted(new_cands,key=lambda x:-x[1])
        kept_haps = copy.deepcopy(new_cands)[:num_candidates]
    
    return kept_haps

def convert_mesh_to_haplotype(full_samples_data,full_sites,
                              haps_data,full_mesh,
                              num_candidates=50,
                              keeping_diff=None):
    """
    Given a mesh of transition probabilities uses that to come up 
    with a long haplotype
    io
    This uses beam search to select the most likely haplotype, 
    keeping num_candidates=50 haplotypes at each step
    
    For best performance ideally num_candidates is at least 2x the 
    number of suspected individuals present
    
    keeping_diff is a parameter used for selection of the best
    partial haps at each stage
    """
    
    first_block = haps_data[0]
    
    first_block_haps = first_block[3]
    first_block_sites = first_block[0]
    first_keep_flags = first_block[1]
    
    samples_first_restricted = analysis_utils.get_sample_data_at_sites_multiple(
        full_samples_data,full_sites,first_block_sites,ploidy_present=True)
    
    matches = hap_statistics.match_best(first_block_haps,
                samples_first_restricted[0],first_keep_flags)
    
    best_first_match = max(matches[1], key=matches[1].get)
    
    sorted_matches_usages = sorted(matches[1],key=matches[1].get)[::-1]
    
    #List of best candidates so far, for the beam search
    kept_haps = []
    
    #Start by populating the list
    for i in range(min(num_candidates,len(matches[1]))):
        kept_haps.append([[sorted_matches_usages[i]],0])
    
    
    for i in range(1,len(haps_data)):
        available_haps = haps_data[i][3]
        
        new_cands = []
        
        for j in range(len(kept_haps)):
            cur_examine = kept_haps[j] #The hap we are currently examining to extend
        
            base = 1
            cur_subtract = 1
        
            available_haps_probs = {k: [] for k in available_haps.keys()}
        
            while cur_subtract <= i:
                earlier = i-cur_subtract
            
                earlier_block_value = cur_examine[0][earlier]
            
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
                
            #For each new candidate block hap to move to create a copy and add to list with its log likelihood
            for k in available_haps_probs.keys():
                cur_copy = copy.deepcopy(cur_examine)
                cur_copy[0].append(k)
                cur_copy[1] += available_haps_probs[k]
                
                new_cands.append(cur_copy)
            
        #Update the kept_haps for the next stage
        new_cands = sorted(new_cands,key=lambda x:-x[1])
        kept_haps = copy.deepcopy(new_cands)[:num_candidates]
        
        # print(i)
        # breakpoint()

    #At this point we have our list of num_candidate candidate haplotypes following the first pass through
    #We now do a second pass in reverse to see if changing things helps us anywhere
    second_pass_kept_haps = backward_hap_construction_update(kept_haps, haps_data, full_mesh,num_candidates=num_candidates)
    
    #And now do the forwards pass again
    third_pass_kept_haps = forward_hap_construction_update(second_pass_kept_haps, haps_data, full_mesh,num_candidates=num_candidates)

    #fourth_pass_hap = backward_hap_construction_update(third_pass_hap, haps_data, full_mesh)
    #fifth_pass_hap = forward_hap_construction_update(fourth_pass_hap, haps_data, full_mesh)
        
    
    combined_positions = []
    combined_haplotype = []
    
    for i in range(len(haps_data)):
        combined_positions.extend(haps_data[i][0])
        combined_haplotype.extend(haps_data[i][3][third_pass_kept_haps[0][0][i]])
    
    combined_positions = np.array(combined_positions)
    combined_haplotype = np.array(combined_haplotype)
    
    return (combined_positions,combined_haplotype,third_pass_kept_haps[0][0])   
   
#%%
def compute_all_haplotypes(full_samples_data,
                           sample_sites,
                           haps_data,
                           max_num_iterations=15,
                           found_transition_penalty=0.1,
                           minimum_transition_log_likelihood=-10,
                           learning_rate=0.5,
                           num_beam_search_candidates=50):
    """
    Function which takes as input the site likelihoods for all samples,
    the list fo sites and the list of block haplotypes to generate all the
    haplotypes present in the data
    """
    
    found_haps = []
    found_block_haps = []
    
    #Calculate the haps one by one using EM-MAP
    for i in range(4):
        print(f"HAP NUMBER {i}")
        mesh = block_linking_em.generate_transition_probability_mesh(full_samples_data,
                sample_sites,haps_data,max_num_iterations=max_num_iterations,
                currently_found_long_haps=found_block_haps,
                found_transition_penalty=found_transition_penalty,
                minimum_transition_log_likelihood=minimum_transition_log_likelihood,
                learning_rate=learning_rate)
        
        found_hap = convert_mesh_to_haplotype(full_samples_data,
               sample_sites,haps_data,mesh,num_candidates=num_beam_search_candidates)
        
        found_haps.append(found_hap)
        found_block_haps.append(found_hap[2])
    
    return [found_haps,found_block_haps]

def number_to_index(n):
    """
    Returns unique integer i where n = i*(i+1)/2+k with k < i+1.
    This is equivalent to finding the largest integer i such that the i-th triangular number
    T_i = i*(i+1)/2 is less than or equal to n.
    
    Raises a ValueError If n is a negative integer.
    """
    if n < 0:
        raise ValueError("Input n must be a non-negative integer.")

    discriminant = 1 + 8 * n
    i_float = (-1 + math.sqrt(discriminant)) / 2

    i = math.floor(i_float)
    
    return int(i)

def index_to_pair(idx, n):
    """
    Given an index 'idx' and a maximum value 'n', returns the (a, b) pair
    from the sequence (0,0), (0,1), ..., (0,n-1), (1,1), ..., (1,n-1), ..., (n-1,n-1).
    The pairs satisfy 0 <= a <= b < n.

    Args:
        idx: The 0-based index of the desired pair.
        n: The upper exclusive bound for b (b < n).

    Returns:
        A tuple (a, b) corresponding to the given index.

    Raises:
        ValueError: If n is not a positive integer or idx is out of bounds.
    """

    total_pairs = n * (n + 1) // 2

    discriminant = (2 * n + 1)**2 - 8 * idx
    
    discriminant = max(0, discriminant) 

    a_float = ((2 * n + 1) - math.sqrt(discriminant)) / 2
    a = math.floor(a_float)

    start_idx_a = a * n - a * (a - 1) // 2

    offset = idx - start_idx_a
    b = a + offset

    return (int(a), int(b))
        
def explain_sample_viterbi(sample_data,sample_sites,haps_data,
                           full_haplotypes,
                           full_blocks_likelihoods = None,
                           recomb_rate=5*10**-8,
                           block_size=10**5):
    """
    Finds the best possible explanation of all the samples (given as likelihoods of
    each possible haplotype at each site)
    
    Here the sample_data must be given with ploidy
    """
    
    print("NOW")
    sample_likelihoods = sample_data[0]
    sample_ploidy = sample_data[1]
    
    num_blocks = len(haps_data)
    num_haps = len(full_haplotypes[0])
    num_combs = int((num_haps*(num_haps+1))/2)
    
    block_recomb_rate = recomb_rate*block_size #Probability of recombining within a block, to be accurate we need block_size*recomb_rate << 1
    
    emissions_matrix = np.zeros(shape=(num_combs,num_blocks))
    
    print("STARTED")
    
    if full_blocks_likelihoods == None:
        #Calculate site data and underlying likelihoods for each sample
        #and possible pair making up that sample at each block
        processing_pool = Pool(32)
    
        full_blocks_likelihoods = processing_pool.starmap(
            lambda i: block_linking_em.get_block_likelihoods(
                analysis_utils.get_sample_data_at_sites(sample_data,
                sample_sites,haps_data[i][0],ploidy_present=True),
                haps_data[i]),
            zip(range(len(haps_data))))    
        
    print("HERE")

    for block_idx in range(num_blocks):
        start_point = 0
        for i in range(num_haps):
            for j in range(i,num_haps):
                hap_comb = (full_haplotypes[0][i][2][block_idx],full_haplotypes[0][j][2][block_idx])
                if hap_comb[0] > hap_comb[1]: #Flip if our indices are the wrong way around
                    hap_comb = (hap_comb[1],hap_comb[0])
                
                log_likelihood = full_blocks_likelihoods[block_idx][hap_comb]
                emissions_matrix[start_point,block_idx] = log_likelihood
                start_point += 1
                
    viterbi_probs = np.zeros(shape=(int((num_haps*(num_haps+1))/2),num_blocks))
    backpointers = np.zeros(shape=(int((num_haps*(num_haps+1))/2),num_blocks))
    
    viterbi_probs[:,0] = emissions_matrix[:,0]
    
    for block_idx in range(1,num_blocks):
        i = 0
        j = 0
        for idx in range(num_combs):
            cur_comb = (i,j)
            
            here_value = emissions_matrix[idx,block_idx]
            
            highest_log_likelihood = float("-inf")
            best_pointer = 0
            
            for h in range(num_combs):
                earlier_comb_name = index_to_pair(h,num_haps)
            
                earlier_log_likeli = emissions_matrix[h,block_idx-1]
                
                if i == earlier_comb_name[0] and j == earlier_comb_name[1]:
                    recomb_log_likeli = 2*math.log(1-block_recomb_rate)
                elif (i in earlier_comb_name) or (j in earlier_comb_name):
                    recomb_log_likeli = math.log(2)+math.log(1-block_recomb_rate)+math.log(block_recomb_rate)
                else:
                    recomb_log_likeli = 2*math.log(block_recomb_rate)
                
                total_log_likeli = earlier_log_likeli+recomb_log_likeli+here_value
                
                if total_log_likeli > highest_log_likelihood:
                    highest_log_likelihood = total_log_likeli
                    best_pointer = h
            
            viterbi_probs[idx,block_idx] = highest_log_likelihood
            backpointers[idx,block_idx] = best_pointer
            ...
            j += 1
            if j >= num_haps:
                i += 1
                j = i
            
    best_path = []
    
    next_index = int(np.argmax(viterbi_probs[:,-1]))
    best_path.append(index_to_pair(next_index,num_haps))
    
    for i in range(num_blocks-1,0,-1):
        next_index = int(backpointers[next_index,i])
        best_path.append(index_to_pair(next_index,num_haps))
        
    return best_path[::-1]
    
#%%
start = time.time()
main_haplotype = convert_mesh_to_haplotype(all_likelihoods,
        all_sites,test_haps,final_mesh,num_candidates=50)
print(time.time()-start)
#%%
start = time.time()
found_haplotypes = compute_all_haplotypes(all_likelihoods,all_sites,
                                          test_haps,max_num_iterations=10)
print(time.time()-start)
#%%
check = found_haplotypes[0][3]
for i in range(len(haplotype_data)):
    print(i,analysis_utils.calc_perc_difference(check[1],
        haplotype_data[i][:18341],calc_type="haploid"))
    print(len(np.where(
        simulate_sequences.concretify_haps([check[1]])[0] !=
        simulate_sequences.concretify_haps([haplotype_data[i][:18341]])[0])[0]))
    print(len(check[1]))
    print()

#%%
test = explain_sample_viterbi([all_likelihoods[0][4][:18341],all_likelihoods[1][0][:18341]],
                              all_sites[:18341],
                              test_haps,[found_haplotypes[0][:5]],
                              recomb_rate=5*10**-8)
print(test)