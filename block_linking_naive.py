import numpy as np
import math
import copy
from multiprocess import Pool
import warnings

import analysis_utils
import hap_statistics
import block_haplotypes

warnings.filterwarnings("ignore")
np.seterr(divide='ignore',invalid="ignore")
pass

#%%
def match_haplotypes_by_overlap(block_level_haps,
                     hap_cutoff_autoinclude=2,
                     hap_cutoff_noninclude=5):
    """
    Takes as input a list of positions and block level haplotypes and
    finds which haps from which block 
    
    hap_cutoff_autoinclude is an upper bound for how different two 
    overlapping portions from neighbouring haps can be for
    us to always have a link between them
    
    hap_cutoff_noninclude is an lower bound such that whenever two
    overlapping portions are at least this different we never have 
    a link between them
    
    """        
    
    next_starting = []
    matches = []
    
    block_haps_names = []
    block_counts = {}
    
    #Find overlap starting points
    for i in range(1,len(block_level_haps)):
        start_position = block_level_haps[i][0][0]
        insertion_point = np.searchsorted(block_level_haps[i-1][0],start_position)
        next_starting.append(insertion_point)
        
    #Create list of unique names for the haps in each of the blocks    
    for i in range(len(block_level_haps)):
        block_haps_names.append([])
        block_counts[i] = len(block_level_haps[i][3])
        for name in block_level_haps[i][3].keys():
            block_haps_names[-1].append((i,name))

    #Iterate over all the blocks
    for i in range(len(block_level_haps)-1):
        start_point = next_starting[i]
        overlap_length = len(block_level_haps[i][0])-start_point   
        
        cur_ends = {k:block_level_haps[i][3][k][start_point:] for k in block_level_haps[i][3].keys()}
        next_ends = {k:block_level_haps[i+1][3][k][:overlap_length] for k in block_level_haps[i+1][3].keys()}

        cur_keep_flags = np.array(block_level_haps[i][1][start_point:],dtype=bool)
        next_keep_flags = np.array(block_level_haps[i+1][1][:overlap_length],dtype=bool)
        
        assert (cur_keep_flags == next_keep_flags).all(),"Keep flags don't match up"
        
        matches.append([])
        
        min_expected_connections = max(block_counts[i],block_counts[i+1])
        if block_counts[i] == 0 or block_counts[i+1] == 0:
            min_expected_connections = 0
            
        amount_added = 0
        all_edges_consideration = {}
        
        for first_name in cur_ends.keys(): 
            dist_values = {}

            for second_name in next_ends.keys():
                
                first_new_hap = cur_ends[first_name][cur_keep_flags]
                second_new_hap = next_ends[second_name][next_keep_flags]
                
                common_size = len(first_new_hap)
                
                if common_size > 0:
                    haps_dist = 100*analysis_utils.calc_distance(first_new_hap,
                                      second_new_hap,
                                      calc_type="haploid")/common_size
                else:
                    haps_dist = 0
                    
                dist_values[(first_name,second_name)] = haps_dist

            #Add autoinclude edges and remove from consideration those which are too different to ever be added
            removals = []
            for k in dist_values.keys():
                if dist_values[k] <= hap_cutoff_autoinclude:
                    matches[-1].append(((i,k[0]),(i+1,k[1])))
                    amount_added += 1
                    removals.append(k)
                if dist_values[k] >= hap_cutoff_noninclude:
                    removals.append(k)
            for k in removals:
                dist_values.pop(k)
            all_edges_consideration.update(dist_values)
        
        all_edges_consideration = {k:v for k, v in sorted(all_edges_consideration.items(), key=lambda item: item[1])}
        removals = []
        
        for key in all_edges_consideration:
            if amount_added < min_expected_connections:
                matches[-1].append(((i,key[0]),(i+1,key[1])))
                amount_added += 1
                continue
            
    return (block_haps_names,matches)

def match_haplotypes_by_overlap_probabalistic(block_level_haps):
    """
    Probabalistic version of match_haplotypes_by_overlap which
    instead of returning a list of edges it returns a likelihood of 
    an edge for each pair of nodes in neighbouring layers
    
    Takes as input a list of positions and block level haplotypes and
    finds which haps from which block 
    """
    next_starting = []
    matches = []
    
    block_haps_names = []
    block_counts = {}
    
    #Find overlap starting points
    for i in range(1,len(block_level_haps)):
        start_position = block_level_haps[i][0][0]
        insertion_point = np.searchsorted(block_level_haps[i-1][0],start_position)
        next_starting.append(insertion_point)
        
    #Create list of unique names for the haps in each of the blocks    
    for i in range(len(block_level_haps)):
        block_haps_names.append([])
        block_counts[i] = len(block_level_haps[i][3])
        for name in block_level_haps[i][3].keys():
            block_haps_names[-1].append((i,name))

    #Iterate over all the blocks
    for i in range(len(block_level_haps)-1):
        start_point = next_starting[i]
        overlap_length = len(block_level_haps[i][0])-start_point   
        
        cur_ends = {k:block_level_haps[i][3][k][start_point:] for k in block_level_haps[i][3].keys()}
        next_ends = {k:block_level_haps[i+1][3][k][:overlap_length] for k in block_level_haps[i+1][3].keys()}
           
        cur_keep_flags = np.array(block_level_haps[i][1][start_point:],dtype=bool)
        next_keep_flags = np.array(block_level_haps[i+1][1][:overlap_length],dtype=bool)
        
        assert (cur_keep_flags == next_keep_flags).all(),"Keep flags don't match up"
        
        similarities = {}
        for first_name in cur_ends.keys(): 

            for second_name in next_ends.keys():
                
                first_new_hap = cur_ends[first_name][cur_keep_flags]
                second_new_hap = next_ends[second_name][next_keep_flags]
                
                common_size = len(first_new_hap)
                
                if common_size > 0:
                    haps_dist = 100*analysis_utils.calc_distance(first_new_hap,
                                      second_new_hap,
                                      calc_type="haploid")/common_size
                else:
                    haps_dist = 0
                
                if haps_dist > 50:
                    similarity = 0
                else:
                    similarity = 2*(50-haps_dist)
                
                similarities[((i,first_name),(i+1,second_name))] = similarity
            
        #Scale and transform the similarities into a score
        transform_similarities = {}
            
        for item in similarities.keys():
            val = similarities[item]/100
                
            transformed = 100*(val**2)
            transform_similarities[item] = transformed
            
        matches.append(transform_similarities)
            
    return (block_haps_names,matches)

def match_haplotypes_by_samples(full_haps_data):
    """
    Alternate method of matching haplotypes in nearby blocks together
    by matching hap A with hap B if the samples which use hap A at its location
    disproportionately use hap B at its location
    
    full_haps_data is the return from a run of generate_haplotypes_all,
    containing info about the positions, read count array and haplotypes
    for each block.
    """
    
    
    #Controls the threshold for which a score higher than results in an edge and how low other scores relative to the highest can be for an edge to be added
    auto_add_val = 70
    max_reduction_include = 0.8
    
    num_blocks = len(full_haps_data)
    
    all_matches = []
    
    block_haps_names = []
    for i in range(len(full_haps_data)):
        block_haps_names.append([])
        
        for nm in full_haps_data[i][3].keys():
            block_haps_names[-1].append((i,nm))
    
    match_best_results = []
    
    processing_pool = Pool(8)
    
    match_best_results = processing_pool.starmap(hap_statistics.combined_best_hap_matches,
                                                 zip(full_haps_data))
    
    neighbouring_usages = processing_pool.starmap(lambda x,y:
                        hap_statistics.hap_matching_comparison(full_haps_data,match_best_results,x,y),
                        zip(list(range(num_blocks-1)),list(range(1,num_blocks))))
    
        
    forward_matches = []
    backward_matches = []
    
    for x in range(len(block_haps_names)-1):
        first_names = block_haps_names[x]
        second_names = block_haps_names[x+1]
        
        forward_edges_add = []
        for first_hap in first_names:
            highest_score_found = -1
            for second_hap in second_names:
                sim_val = neighbouring_usages[x][0][(first_hap,second_hap)]
                if sim_val > highest_score_found:
                    highest_score_found = sim_val

            for second_hap in second_names:
                sim_val = neighbouring_usages[x][0][(first_hap,second_hap)]
                
                if sim_val >= auto_add_val or sim_val > highest_score_found*max_reduction_include:
                    forward_edges_add.append((first_hap,second_hap))
        
        forward_matches.append(forward_edges_add)
        
        backward_edges_add = []
        for second_hap in second_names:
            highest_score_found = -1
            for first_hap in first_names:
                sim_val = neighbouring_usages[x][1][(first_hap,second_hap)]
                if sim_val > highest_score_found:
                    highest_score_found = sim_val
                    
            for first_hap in first_names:
                sim_val = neighbouring_usages[x][1][(first_hap,second_hap)]
                
                if sim_val >= auto_add_val or sim_val > highest_score_found*max_reduction_include:
                    backward_edges_add.append((first_hap,second_hap))
        
        backward_matches.append(backward_edges_add)

    for i in range(len(forward_matches)):
        commons = []
        for x in forward_matches[i]:
            if x in backward_matches[i]:
                commons.append(x)
        all_matches.append(commons)
        
    return [(block_haps_names,forward_matches),(block_haps_names,backward_matches),(block_haps_names,all_matches)]
        
def match_haplotypes_by_samples_probabalistic(full_haps_data):
    """
    Probabalistic version of match_haplotypes_by_samples that gives 
    a likelihood for each possible edge between neighbouring layers
    
    full_haps_data is the return from a run of generate_haplotypes_all,
    containing info about the positions, read count array and haplotypes
    for each block.
    
    Returns a list of node labels and a list of dictionaries with scaled
    scores for how strong of an edge there is between the first element and the 
    second element of the key for each key in each dictionary
    """
        
    num_blocks = len(full_haps_data)
    
    block_haps_names = []
    for i in range(len(full_haps_data)):
        block_haps_names.append([])
        
        for nm in full_haps_data[i][3].keys():
            block_haps_names[-1].append((i,nm))
    
    match_best_results = []
    
    processing_pool = Pool(8)
    
    match_best_results = processing_pool.starmap(hap_statistics.combined_best_hap_matches,
                                                 zip(full_haps_data))
    
    neighbouring_usages = processing_pool.starmap(lambda x,y:
                        hap_statistics.hap_matching_comparison(full_haps_data,match_best_results,x,y),
                        zip(list(range(num_blocks-1)),list(range(1,num_blocks))))
    
    forward_match_scores = [neighbouring_usages[x][0] for x in range(num_blocks-1)]
    backward_match_scores = [neighbouring_usages[x][1] for x in range(num_blocks-1)]
    
    combined_scores = []
    

    for i in range(len(forward_match_scores)):
        commons = {}
        for x in forward_match_scores[i].keys():
            commons[x] = (forward_match_scores[i][x]+backward_match_scores[i][x])/2
            
        combined_scores.append(commons)
        
    return [(block_haps_names,forward_match_scores),(block_haps_names,backward_match_scores),(block_haps_names,combined_scores)]
        
#%%

def get_combined_hap_score(hap_overlap_scores,hap_sample_scores,
                           overlap_importance=1):
    """
    Combines the results from match_haplotypes_by_overlap_probabalistic
    and match_haplotypes_by_samples_probabalistic (just the combined output
    of this function and not the forward/backward ones) to get one single
    likelihood score for each edge which is normalised to a maximum value
    of 100. 
    
    overlap_importance is a measure of how much we weight the overlap score vs
    the sample scores (individually). A value of 1 here means we weight the
    combined sample score twice as much as the overlap score (because sample
    score is combined version of both forward and backward sample scores).
    
    """
    ovr = hap_overlap_scores[1]
    samps = hap_sample_scores[1]
    
    total_weight = overlap_importance+2
    
    combined_dict = {}
    
    for i in range(len(ovr)):
        for d in ovr[i].keys():
            comb = (overlap_importance*ovr[i][d]+2*samps[i][d])/total_weight      
            combined_dict[d] = comb
    
    return combined_dict

#%%
def calc_best_scoring(padded_nodes_list,node_scores,edge_scores):
    """
    Uses Breadth First Search going backwards to calculate the best scoring possible for each starting 
    node assuming we get to the end. This uses "I" as a dummy initial starting node
    and "S" as the final sink node
    """
    
    num_layers = len(padded_nodes_list)

    
    scorings = [{"S":0}]
    
    for layer in range(num_layers-2,-1,-1):
        this_nodes = padded_nodes_list[layer]
        next_nodes = padded_nodes_list[layer+1]
        
        layer_scores = {}
        
        for node in this_nodes:
            best_score = -np.inf
            for other in next_nodes:
                new_score = node_scores[node]+edge_scores[(node,other)]+scorings[-1][other]
                
                if new_score > best_score:
                    best_score = new_score
                
            layer_scores[node] = best_score
            
        scorings.append(layer_scores)
    
    return scorings[::-1]
    
def scorings_to_optimal_path(scorings,padded_nodes_list,node_scores,edge_scores):
    """
    Takes a list of dictionaries of optimal scorings from each node to the end and
    calculates the optimal path starting at I and ending at S
    """
    cur_path = ["I"]
    cur_node = "I"
    
    for i in range(len(padded_nodes_list)-1):

        cur_score = scorings[i][cur_node]
        
        next_nodes = padded_nodes_list[i+1]
        for new_node in next_nodes:
            score_removal = node_scores[cur_node]+edge_scores[(cur_node,new_node)]
            remaining_score = cur_score-score_removal
            
            if abs(remaining_score-scorings[i+1][new_node]) < 10**-10:
                cur_path.append(new_node)
                cur_node = new_node
                break
    
    return cur_path     

def generate_chained_block_haplotypes(haplotype_data,nodes_list,combined_scores,num_haplotypes,
                             node_usage_penalty=10,edge_usage_penalty=10):
    """
    Generates num_haplotypes many chromosome length haplotypes given
    a layered list of the nodes and a dictionary containing the combined likelihood
    scores for each possible edge between layers.
    
    Returns the haplotypes as a list of nodes, one from each layer from the start to 
    the end.
    
    This function works through a reverse Breadth First Search algorithm trying to maximize
    the score between the start and the end. The first haplotype is just the maximal path.
    
    For future haplotypes we apply a penalty to each node/edge already on a discovered 
    haplotype with a penalty also applied to other similar nodes to used nodes in the same
    layer and run the Breadth First search again. We repeat this process until we generate
    num_haplotypes many haplotypes.
    """
    
    num_layers = len(nodes_list)
    
    processing_pool = Pool(8)
    
    similarity_matrices = processing_pool.starmap(hap_statistics.get_block_hap_similarities,
                                       zip(haplotype_data))#Similarity matrices for calculating the associated penalty when we use a node
    
    current_edge_scores = combined_scores.copy()
    current_node_scores = {"I":0,"S":0}
    for i in range(len(nodes_list)):
        for node in nodes_list[i]:
            current_node_scores[node] = 0
    
    nodes_copy = nodes_list.copy()
    nodes_copy.insert(0,["I"])
    nodes_copy.append(["S"])
    
    #Add edges from the dummy nodes to first and last layers
    for xm in range(len(nodes_list[0])):
        current_edge_scores[("I",(0,xm))] = 0
    for xm in range(len(nodes_list[-1])):
        current_edge_scores[((num_layers-1,xm),"S")] = 0
    
    found_haps = []
    
    for ite in range(num_haplotypes):
        best_scores = calc_best_scoring(nodes_copy,current_node_scores,current_edge_scores)
        
        found_hap = scorings_to_optimal_path(best_scores,nodes_copy,current_node_scores,current_edge_scores)
        
        #Now that we have our hap apply node penalties
        for i in range(1,len(found_hap)-1):
            layer = found_hap[i][0]
            used_hap = found_hap[i][1]
            reductions = (node_usage_penalty)*similarity_matrices[layer][used_hap,:]
            for nm in range(len(reductions)):
                current_node_scores[(layer,nm)] -= reductions[nm]
        
        #And apply edge penalties
        for i in range(1,len(found_hap)-2):
            current_edge_scores[(found_hap[i],found_hap[i+1])] -= edge_usage_penalty
        
        found_haps.append(found_hap[1:-1])

    return found_haps

def combine_chained_blocks_to_single_hap(all_haps,
                                         hap_blocks,
                                         read_error_prob = 0.02,
                                         min_total_reads=5):
    """
    Takes in as input the block level haplotypes (such as generated by
    by generate_haplotypes_all) as well as a list giving the blocks which
    make up a haplotype and then converts this into a single long 
    chromosome length haplotype
    
    read_error_prob and min_total_reads are used to calculate the haplotype
    level priors for each site from the read counts in all_haps
    
    This function assumes that everything except for the very end of the 
    starting/finishing block ends up in exactly two blocks, i.e. the shift size
    is exactly half of the block size!!!
    
    """
    
    next_starting = []
    
    #Find overlap starting points
    for i in range(1,len(all_haps)):
        start_position = all_haps[i][0][0]
        insertion_point = np.searchsorted(all_haps[i-1][0],start_position)
        next_starting.append(insertion_point)
        
    
    final_haplotype = []
    final_locations = []
    
    for i in range(len(all_haps)-1):
        hap_here = hap_blocks[i][1]
        hap_next = hap_blocks[i+1][1]
        start_point = next_starting[i]
        overlap_length = len(all_haps[i][0])-start_point  

        
        if i == 0:
            start_data = all_haps[i][3][hap_here][:start_point]
            final_haplotype.extend(start_data)
            final_locations.extend(all_haps[i][0][:start_point])
        
        
        cur_overlap_data = all_haps[i][3][hap_here][start_point:]
        next_overlap_data = all_haps[i+1][3][hap_next][:overlap_length]
        
        reads_sum = np.sum(all_haps[i][2][:,start_point:,:],axis=0)

        num_samples = len(all_haps[i][2])
    
        hap_priors = []
        for j in range(len(reads_sum)):
        
            if sum(reads_sum[j]) >= max(min_total_reads,read_error_prob*num_samples):
                rat_val = (1+reads_sum[j][1])/(2+reads_sum[j][0]+reads_sum[j][1])
            else:
                rat_val = read_error_prob
            hap_priors.append([rat_val,1-rat_val])
        hap_priors = np.array(hap_priors)
        
        new_probs = []
        for j in range(len(hap_priors)):
            new_val = analysis_utils.combine_probabilities(cur_overlap_data[j],next_overlap_data[j],hap_priors[j])

            new_probs.append(new_val)
        new_probs = np.array(new_probs)
        
        final_haplotype.extend(new_probs)
        final_locations.extend(all_haps[i+1][0][:overlap_length])
        
        if i == len(all_haps)-2:
            end_data = all_haps[i+1][3][hap_next][overlap_length:]
            final_haplotype.extend(end_data)
            final_locations.extend(all_haps[i+1][0][overlap_length:])
    
    return [np.array(final_locations),np.array(final_haplotype)]

def combine_all_blocks_to_long_haps(all_haps,
                                    hap_blocks_list,
                                    read_error_prob = 0.02,
                                    min_total_reads=5):
    """
    Multithreaded version of combine_chained_blocks_to_single_hap
    which processes all of our haps at once
    """
        
    processing_pool = Pool(8)
    
    processing_results = processing_pool.starmap(lambda x: combine_chained_blocks_to_single_hap(
                                        all_haps,x,read_error_prob=read_error_prob,
                                        min_total_reads=min_total_reads),
                                        zip(hap_blocks_list))
    
    sites_loc = processing_results[0][0]
    long_haps = [processing_results[x][1] for x in range(len(processing_results))]
    
    return [sites_loc,long_haps]

def compute_likeliest_path(full_combined_genotypes,sample_probs,site_locations,
                            keep_flags=None,recomb_rate=10**-8,value_error_rate=10**-3):
    """
    Compute the likeliest path explaining the sample using a combination of
    genotypes from full_combined_genotypes via the Viterbi algorithm
    
    We only update on sites which have keep_flags[i] = 1
    """
    
    data_shape = full_combined_genotypes.shape
    
    if keep_flags is None:
        keep_flags = np.array([1 for _ in range(data_shape[2])])
    
    if keep_flags.dtype != int:
        keep_flags = np.array(keep_flags,dtype=int)
        
    #Matrix which records the probabilities of seeing one thing when the true value is another
    #This is used for calculating posteriors later down the line,
    #Moving across rows we see true baseline genotype and across columns we have observed sample genotype
    eps = value_error_rate 
    value_error_matrix = [[(1-eps)**2,2*eps*(1-eps),eps**2],
                          [eps*(1-eps),eps**2+(1-eps)**2,eps*(1-eps)],
                          [eps**2,2*eps*(1-eps),(1-eps)**2]]
    
        

    num_haps = data_shape[0]
    num_rows = data_shape[0]
    num_cols = data_shape[1]
    num_geno = data_shape[0]*data_shape[1]
    
    num_sites = data_shape[2]
    
    starting_probabilities = analysis_utils.make_upper_triangular((1/num_geno)*np.ones((num_haps,num_haps)))
    log_current_probabilities = np.log(starting_probabilities)
    
    prev_best = np.empty((data_shape[0],data_shape[1]),dtype=object)
    prev_best[:] = [[(i,j) for j in range(data_shape[1])] for i in range(data_shape[0])]

    log_probs_history = [copy.deepcopy(log_current_probabilities)]
    prev_best_history = [copy.deepcopy(prev_best)]
    
    last_site_loc = None
    
    for loc in range(num_sites):
        if keep_flags[loc] != 1:
            log_probs_history.append(copy.deepcopy(log_current_probabilities))
            prev_best_history.append(copy.deepcopy(prev_best))
        else:
            
            if last_site_loc == None:
                distance_since_last_site = 0
            else:
                distance_since_last_site = site_locations[loc]-last_site_loc
            
        
            #Update last_site_loc
            last_site_loc = site_locations[loc]
            
            #Create transition probability matrix            
            num_possible_switches = num_rows+num_cols-1

            non_switch_prob = (1-recomb_rate)**distance_since_last_site
            
            each_switch_prob= (1-non_switch_prob)/num_possible_switches

            #Account for the fact that we can recombine back to the haplotype pre recombination
            total_non_switch_prob = non_switch_prob+each_switch_prob
            
            transition_probs = np.zeros((num_rows,num_cols,num_rows,num_cols))
            
            for i in range(num_rows):
                for j in range(i,num_cols):
                    transition_probs[i,j,:,j] = each_switch_prob
                    transition_probs[i,j,i,:] = each_switch_prob
                    transition_probs[i,j,i,j] = total_non_switch_prob
                    
                    transition_probs[i,j,:,:] = analysis_utils.make_upper_triangular(transition_probs[i,j,:,:])
                    
            
            
            site_sample_val = sample_probs[loc]
            
            genotype_vals = full_combined_genotypes[:,:,loc,:]

            #Calculate for each genotype combination the probability it equals x and the true data equals y
            all_combs = np.einsum("i,jkl->jkil",site_sample_val,genotype_vals)
            
            #Calculate for each genotype combination the probability of seeing the data given the true underlying genotype 
            prob_data_given_comb = np.einsum("ijkl,kl->ij",all_combs,value_error_matrix)
            
            #Calculate the probability of switching to state (k,l) and seeing our observed data given that we started in state (i,j)
            prob_switch_seen = np.einsum("ijkl,kl->ijkl",transition_probs,prob_data_given_comb)
            
            log_prob_switch_seen = math.log(prob_switch_seen)
            
            extended_log_cur_probs = copy.deepcopy(log_current_probabilities)
            extended_log_cur_probs = np.expand_dims(extended_log_cur_probs,2)
            extended_log_cur_probs = np.expand_dims(extended_log_cur_probs,3)
            extended_log_cur_probs = np.repeat(extended_log_cur_probs,num_rows,axis=2)
            extended_log_cur_probs = np.repeat(extended_log_cur_probs,num_cols,axis=3)
            
            log_total_combined_probability = extended_log_cur_probs+log_prob_switch_seen
            
            best_matches = np.empty((data_shape[0],data_shape[1]),dtype=object)
            best_log_probs = np.empty((data_shape[0],data_shape[1]),dtype=float)
            
            for k in range(num_rows):
                for l in range(num_cols):
                    comb_data = log_total_combined_probability[:,:,k,l]
                    max_combination = np.unravel_index(np.argmax(comb_data), comb_data.shape)
                    max_val = comb_data[max_combination]
                    
                    best_matches[k,l] = max_combination
                    best_log_probs[k,l] = max_val
            
            log_current_probabilities = best_log_probs
            prev_best = best_matches
            
            log_probs_history.append(copy.deepcopy(log_current_probabilities))
            prev_best_history.append(copy.deepcopy(prev_best))
    
    #Now we can work backwards to figure out the most likely path for the sample
    reversed_path = []
    
    cur_place = np.unravel_index(np.argmax(log_probs_history[-1]), log_probs_history[-1].shape)
    
    reversed_path.append(cur_place)
    
    for i in range(len(prev_best_history)-1,-1,-1):
        cur_place = prev_best_history[i][cur_place[0],cur_place[1]]
        reversed_path.append(cur_place)
    
    return (reversed_path,log_probs_history,prev_best_history)

def generate_long_haplotypes_naive(full_positions_data,full_reads_data,num_long_haps,
                             full_keep_flags=None,block_size=100000,shift_size=50000,
                             node_usage_penalty=10,edge_usage_penalty=10):
    """
    Takes as input a list of VCF Record data where each element represents 
    full data for a single site for all the samples. Runs the full pipline 
    and generates num_long_haps many full length haplotypes 
    
    This version uses the naive function (no EM-algorithm) to generate
    the hapotypes
    """


    all_haps = block_haplotypes.generate_haplotypes_all(full_positions_data,full_reads_data,full_keep_flags)
    #all_matches = get_best_matches_all_blocks(all_haps)
    
    hap_matching_overlap = match_haplotypes_by_overlap_probabalistic(all_haps)
    hap_matching_samples = match_haplotypes_by_samples_probabalistic(all_haps)
    
    node_names = hap_matching_overlap[0]
    
    combined_scores = get_combined_hap_score(hap_matching_overlap,hap_matching_samples[2])
    
    chained_block_haps = generate_chained_block_haplotypes(all_haps,
                        node_names,combined_scores,num_long_haps,
                        node_usage_penalty=node_usage_penalty,
                        edge_usage_penalty=edge_usage_penalty)
    
    final_long_haps = combine_all_blocks_to_long_haps(all_haps,chained_block_haps)

    return (all_haps,final_long_haps)   

#%%
def get_match_probabilities(full_combined_genotypes,sample_probs,site_locations,
                            keep_flags=None,recomb_rate=10**-8,value_error_rate=10**-3):
    """
    Function which takes in the full (square) array of combined 
    genotypes from haplotypes and runs a HMM-esque process to match
    the sample genotype to the best combination of haplotypes which 
    make it up.
    
    This is so of like Li-Stephens but works on probabalistic genotypes
    rather than fixed ones
    
    Only updates on those sites where keep_flag=1
    """
    
    data_shape = full_combined_genotypes.shape
    
    if keep_flags is None:
        keep_flags = np.array([1 for _ in range(data_shape[2])])
    
    if keep_flags.dtype != int:
        keep_flags = np.array(keep_flags,dtype=int)
        
    #Matrix which records the probabilities of seeing one thing when the true value is another
    #This is used for calculating posteriors later down the line,
    #Moving across rows we see true baseline genotype and across columns we have observed sample genotype
    eps = value_error_rate 
    value_error_matrix = [[(1-eps)**2,2*eps*(1-eps),eps**2],
                          [eps*(1-eps),eps**2+(1-eps)**2,eps*(1-eps)],
                          [eps**2,2*eps*(1-eps),(1-eps)**2]]
    
        

    num_haps = data_shape[0]
    num_geno = data_shape[0]*data_shape[1]
    
    num_sites = data_shape[2]
    
    current_probabilities = (1/num_geno)*np.ones((num_haps,num_haps))

    posterior_probabilities = []
    
    last_site_loc = None
    
    #Iterate backwards initially
    for loc in range(num_sites-1,-1,-1):
        if keep_flags[loc] != 1:
            posterior_probabilities.append(copy.deepcopy(current_probabilities))
        else:
            if last_site_loc == None:
                distance_since_last_site = 0
            else:
                distance_since_last_site = last_site_loc-site_locations[loc]
            last_site_loc = site_locations[loc]
                
            #Fudge prior due to possible recombinations
            updated_prior = analysis_utils.recombination_fudge(current_probabilities,
                                                distance_since_last_site,
                                                recomb_rate=recomb_rate)
            
            site_sample_val = sample_probs[loc]
            
            genotype_vals = full_combined_genotypes[:,:,loc,:]

            #Calculate for each genotype combination the probability it equals x and the true data equals y
            all_combs = np.einsum("i,jkl->jkil",site_sample_val,genotype_vals)
            
            #Calculate for each genotype combination the probability of seeing the data given the true underlying value
            prob_data_given_comb = np.einsum("ijkl,kl->ij",all_combs,value_error_matrix)
            
            #Use Bayes's rule to calculate the probability of the combination given data using our prior
            nonorm_prob_comb_given_data = np.einsum("ij,ij->ij",prob_data_given_comb,updated_prior)
            
            #Normalize
            prob_comb_given_data = nonorm_prob_comb_given_data/np.sum(nonorm_prob_comb_given_data)
            
            posterior_probabilities.append(copy.deepcopy(prob_comb_given_data))
            current_probabilities = prob_comb_given_data
    
    #Repeat process going forward
    posterior_probabilities = []
    
    last_site_loc = None
    
    for loc in range(num_sites):
        if keep_flags[loc] != 1:
            posterior_probabilities.append(copy.deepcopy(current_probabilities))
        else:
            if last_site_loc == None:
                distance_since_last_site = 0
            else:
                distance_since_last_site = site_locations[loc]-last_site_loc
            
            last_site_loc = site_locations[loc]
                
            #Fudge prior due to possible recombinations
            updated_prior = analysis_utils.recombination_fudge(current_probabilities,
                                                distance_since_last_site,
                                                recomb_rate=recomb_rate)
            site_sample_val = sample_probs[loc]
            
            genotype_vals = full_combined_genotypes[:,:,loc,:]

            #Calculate for each genotype combination the probability it equals x and the true data equals y
            all_combs = np.einsum("i,jkl->jkil",site_sample_val,genotype_vals)
            
            #Calculate for each genotype combination the probability of seeing the data given the true underlying value
            prob_data_given_comb = np.einsum("ijkl,kl->ij",all_combs,value_error_matrix)
            
            #Use Bayes's rule to calculate the probability of the combination given data using our prior
            nonorm_prob_comb_given_data = np.einsum("ij,ij->ij",prob_data_given_comb,updated_prior)
            
            #Normalize
            prob_comb_given_data = nonorm_prob_comb_given_data/np.sum(nonorm_prob_comb_given_data)
            
            posterior_probabilities.append(prob_comb_given_data)
            current_probabilities = prob_comb_given_data
            
    #Convert to upper triangular matrix for easier visualization
    upp_tri_post = []
    for item in posterior_probabilities:
        new = np.triu(item+np.transpose(item)-np.diag(np.diag(item)))
        upp_tri_post.append(new)
        
    max_loc = []
    
    for i in range(len(upp_tri_post)):
        max_loc.append(np.unravel_index(upp_tri_post[i].argmax(), upp_tri_post[i].shape))
    
    return (max_loc,upp_tri_post)
        
def get_full_match_probs(full_combined_genotypes,all_sample_probs,site_locations,
                            keep_flags=None,recomb_rate=10**-8,value_error_rate=10**-3):         
    """
    Multithreaded version to match all samples to their haplotype combinations
    """
    processing_pool = Pool(8)
    
    results = processing_pool.starmap(lambda x: get_match_probabilities(full_combined_genotypes,x,site_locations,
                                        keep_flags=keep_flags,recomb_rate=recomb_rate,value_error_rate=value_error_rate),
                                      zip(all_sample_probs))
    
    return results
