import numpy as np
import math
import copy
from multiprocess import Pool
import warnings
from functools import partial

import analysis_utils
import hap_statistics
import block_haplotypes

warnings.filterwarnings("ignore")
np.seterr(divide='ignore',invalid="ignore")

# --- SHARED MEMORY MANAGEMENT ---

_SHARED_DATA = {}

def _init_shared_data(data_dict):
    """
    Initializer for the worker pool.
    Updates the global _SHARED_DATA dict in the worker process.
    """
    global _SHARED_DATA
    _SHARED_DATA.clear()
    _SHARED_DATA.update(data_dict)

# --- WORKER WRAPPERS ---

def _worker_match_overlap(i):
    """
    Worker function to calculate overlap similarity between block[i] and block[i+1].
    Accesses data from _SHARED_DATA to avoid serialization.
    """
    blocks = _SHARED_DATA['blocks']
    
    # Get the two blocks
    curr_block = blocks[i]
    next_block = blocks[i+1]
    
    # 1. Determine Overlap Region
    start_position_next = next_block.positions[0]
    
    # Assuming positions are sorted, find where next starts in current
    overlap_start_idx = np.searchsorted(curr_block.positions, start_position_next)
    
    overlap_length = len(curr_block.positions) - overlap_start_idx
    
    if overlap_length <= 0:
        return {}

    # 2. Extract Haplotype Slices
    curr_haps = curr_block.haplotypes
    next_haps = next_block.haplotypes
    
    cur_ends = {k: curr_haps[k][overlap_start_idx:] for k in curr_haps.keys()}
    next_ends = {k: next_haps[k][:overlap_length] for k in next_haps.keys()}
    
    # 3. Handle Keep Flags
    if curr_block.keep_flags is not None:
        cur_keep_flags = np.array(curr_block.keep_flags[overlap_start_idx:], dtype=bool)
    else:
        if len(cur_ends) > 0:
            cur_keep_flags = np.ones(len(list(cur_ends.values())[0]), dtype=bool)
        else:
            cur_keep_flags = np.array([], dtype=bool)

    if next_block.keep_flags is not None:
        next_keep_flags = np.array(next_block.keep_flags[:overlap_length], dtype=bool)
    else:
        next_keep_flags = np.ones(overlap_length, dtype=bool)
    
    # Align lengths
    min_len = min(len(cur_keep_flags), len(next_keep_flags))
    cur_keep_flags = cur_keep_flags[:min_len]
    next_keep_flags = next_keep_flags[:min_len]
    
    # 4. Calculate Pairwise Similarities
    similarities = {}
    
    for first_name in cur_ends.keys():
        first_new_hap = cur_ends[first_name][:min_len][cur_keep_flags]
        
        for second_name in next_ends.keys():
            second_new_hap = next_ends[second_name][:min_len][next_keep_flags]
            
            common_size = len(first_new_hap)
            
            if common_size > 0:
                haps_dist = 100 * analysis_utils.calc_distance(
                    first_new_hap,
                    second_new_hap,
                    calc_type="haploid"
                ) / common_size
            else:
                haps_dist = 0
            
            # Convert distance to similarity score
            if haps_dist > 50:
                similarity = 0
            else:
                similarity = 2 * (50 - haps_dist)
            
            similarities[((i, first_name), (i+1, second_name))] = similarity
            
    # 5. Transform to final score (Squared)
    transform_similarities = {}
    for item, sim_val in similarities.items():
        val = sim_val / 100.0
        transformed = 100 * (val**2)
        transform_similarities[item] = transformed
        
    return transform_similarities

def _worker_combined_best_hap_matches(block_idx):
    return hap_statistics.combined_best_hap_matches(_SHARED_DATA['blocks'][block_idx])

def _worker_hap_matching_comparison(block_idx_1, block_idx_2):
    """
    UPDATED WORKER: Looks up match_results from _SHARED_DATA
    instead of accepting it as a pickled argument.
    """
    return hap_statistics.hap_matching_comparison(
        _SHARED_DATA['blocks'], 
        _SHARED_DATA['match_results'], 
        block_idx_1, 
        block_idx_2
    )

def _worker_get_similarities(block_idx):
    return hap_statistics.get_block_hap_similarities(_SHARED_DATA['blocks'][block_idx])

def _worker_combine_chained_blocks(chain_data, read_error_prob, min_total_reads):
    return combine_chained_blocks_to_single_hap(
        _SHARED_DATA['blocks'], chain_data, 
        read_error_prob=read_error_prob, 
        min_total_reads=min_total_reads
    )

def _worker_get_match_probabilities(sample_idx, recomb_rate, value_error_rate):
    """
    Worker for get_full_match_probs.
    Fetches specific sample probability matrix and full genotypes from shared memory.
    """
    return get_match_probabilities(
        _SHARED_DATA['genotypes'],
        _SHARED_DATA['sample_probs'][sample_idx], # Access specific sample by index
        _SHARED_DATA['locations'],
        keep_flags=_SHARED_DATA.get('keep_flags'),
        recomb_rate=recomb_rate,
        value_error_rate=value_error_rate
    )

#%%
def match_haplotypes_by_overlap(block_level_haps,
                     hap_cutoff_autoinclude=2,
                     hap_cutoff_noninclude=5):
    """
    Takes as input a list of BlockResult objects and finds matching haps 
    based on overlapping suffix/prefix similarity.
    (Sequential legacy version)
    """        
    
    next_starting = []
    matches = []
    
    block_haps_names = []
    block_counts = {}
    
    #Find overlap starting points
    for i in range(1,len(block_level_haps)):
        start_position = block_level_haps[i].positions[0]
        insertion_point = np.searchsorted(block_level_haps[i-1].positions, start_position)
        next_starting.append(insertion_point)
        
    #Create list of unique names for the haps in each of the blocks    
    for i in range(len(block_level_haps)):
        block_haps_names.append([])
        block_counts[i] = len(block_level_haps[i].haplotypes)
        for name in block_level_haps[i].haplotypes.keys():
            block_haps_names[-1].append((i,name))

    #Iterate over all the blocks
    for i in range(len(block_level_haps)-1):
        start_point = next_starting[i]
        overlap_length = len(block_level_haps[i].positions) - start_point   
        
        curr_haps = block_level_haps[i].haplotypes
        next_haps = block_level_haps[i+1].haplotypes
        
        cur_ends = {k: curr_haps[k][start_point:] for k in curr_haps.keys()}
        next_ends = {k: next_haps[k][:overlap_length] for k in next_haps.keys()}

        # Flags
        if block_level_haps[i].keep_flags is not None:
            cur_keep_flags = np.array(block_level_haps[i].keep_flags[start_point:], dtype=bool)
        else:
            if len(cur_ends) > 0:
                cur_keep_flags = np.ones(len(list(cur_ends.values())[0]), dtype=bool)
            else:
                cur_keep_flags = np.array([], dtype=bool)

        if block_level_haps[i+1].keep_flags is not None:
            next_keep_flags = np.array(block_level_haps[i+1].keep_flags[:overlap_length], dtype=bool)
        else:
            next_keep_flags = np.ones(overlap_length, dtype=bool)
        
        # --- FIX: Define min_len and slice flags ---
        min_len = min(len(cur_keep_flags), len(next_keep_flags))
        cur_keep_flags = cur_keep_flags[:min_len]
        next_keep_flags = next_keep_flags[:min_len]
        # -------------------------------------------
        
        matches.append([])
        
        min_expected_connections = max(block_counts[i], block_counts[i+1])
        if block_counts[i] == 0 or block_counts[i+1] == 0:
            min_expected_connections = 0
            
        amount_added = 0
        all_edges_consideration = {}
        
        for first_name in cur_ends.keys(): 
            dist_values = {}

            for second_name in next_ends.keys():
                
                first_new_hap = cur_ends[first_name][:min_len][cur_keep_flags]
                second_new_hap = next_ends[second_name][:min_len][next_keep_flags]
                
                common_size = len(first_new_hap)
                
                if common_size > 0:
                    haps_dist = 100 * analysis_utils.calc_distance(
                        first_new_hap,
                        second_new_hap,
                        calc_type="haploid"
                    ) / common_size
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
        
        for key in all_edges_consideration:
            if amount_added < min_expected_connections:
                matches[-1].append(((i,key[0]),(i+1,key[1])))
                amount_added += 1
                continue
            
    return (block_haps_names,matches)

def match_haplotypes_by_overlap_probabalistic(block_level_haps, num_processes=16):
    """
    Probabilistic version of match_haplotypes_by_overlap.
    Returns likelihood scores for edges.
    
    Optimized: Parallelized using shared memory.
    """
    
    # 1. Metadata Setup (Fast, sequential)
    block_haps_names = []
    for i in range(len(block_level_haps)):
        block_haps_names.append([])
        for name in block_level_haps[i].haplotypes.keys():
            block_haps_names[-1].append((i,name))
            
    num_junctions = len(block_level_haps) - 1
    if num_junctions < 1:
        return (block_haps_names, [])

    # 2. Parallel Processing
    shared_context = {'blocks': block_level_haps}
    
    with Pool(num_processes, initializer=_init_shared_data, initargs=(shared_context,)) as processing_pool:
        # We process every junction i -> i+1
        matches = processing_pool.map(_worker_match_overlap, range(num_junctions))
            
    return (block_haps_names, matches)

def match_haplotypes_by_samples(full_haps_data, num_processes=16):
    """
    Match haplotypes based on sample usage (best matches).
    full_haps_data should be a BlockResults object.
    
    Optimized: Uses shared memory for full_haps_data and split pools.
    """
    
    #Controls the threshold
    auto_add_val = 70
    max_reduction_include = 0.8
    
    num_blocks = len(full_haps_data)
    all_matches = []
    
    block_haps_names = []
    for i in range(len(full_haps_data)):
        block_haps_names.append([])
        for nm in full_haps_data[i].haplotypes.keys():
            block_haps_names[-1].append((i,nm))
    
    # --- PHASE 1: Get Best Matches ---
    # Init Shared Data with blocks only
    shared_context_1 = {'blocks': full_haps_data}

    with Pool(num_processes, initializer=_init_shared_data, initargs=(shared_context_1,)) as processing_pool:
        match_best_results = processing_pool.map(
            _worker_combined_best_hap_matches,
            range(num_blocks)
        )
        
    # --- PHASE 2: Compare Neighboring Blocks ---
    # Init Shared Data with blocks AND the results from Phase 1
    shared_context_2 = {'blocks': full_haps_data, 'match_results': match_best_results}
    
    with Pool(num_processes, initializer=_init_shared_data, initargs=(shared_context_2,)) as processing_pool:
        # Pass indices to worker
        args_list = [(i, i+1) for i in range(num_blocks - 1)]
        neighbouring_usages = processing_pool.starmap(
            _worker_hap_matching_comparison,
            args_list
        )
    
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
        
def match_haplotypes_by_samples_probabalistic(full_haps_data, num_processes=16):
    """
    Probabilistic version of match_haplotypes_by_samples.
    Optimized: Uses shared memory and split pools.
    """
        
    num_blocks = len(full_haps_data)
    
    block_haps_names = []
    for i in range(len(full_haps_data)):
        block_haps_names.append([])
        for nm in full_haps_data[i].haplotypes.keys():
            block_haps_names[-1].append((i,nm))
    
    # --- PHASE 1 ---
    shared_context_1 = {'blocks': full_haps_data}

    with Pool(num_processes, initializer=_init_shared_data, initargs=(shared_context_1,)) as processing_pool:
        match_best_results = processing_pool.map(
            _worker_combined_best_hap_matches,
            range(num_blocks)
        )
        
    # --- PHASE 2 ---
    shared_context_2 = {'blocks': full_haps_data, 'match_results': match_best_results}
    
    with Pool(num_processes, initializer=_init_shared_data, initargs=(shared_context_2,)) as processing_pool:
        args_list = [(i, i+1) for i in range(num_blocks - 1)]
        neighbouring_usages = processing_pool.starmap(
            _worker_hap_matching_comparison,
            args_list
        )
    
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
    Combines overlap and sample matching scores.
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
    Reverse BFS to find optimal path scoring.
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
                if (node,other) in edge_scores:
                    new_score = node_scores[node]+edge_scores[(node,other)]+scorings[-1][other]
                    if new_score > best_score:
                        best_score = new_score
            
            # If dead end, keep -inf
            layer_scores[node] = best_score
            
        scorings.append(layer_scores)
    
    return scorings[::-1]
    
def scorings_to_optimal_path(scorings,padded_nodes_list,node_scores,edge_scores):
    """
    Reconstructs path from scoring dictionaries.
    """
    cur_path = ["I"]
    cur_node = "I"
    
    for i in range(len(padded_nodes_list)-1):

        cur_score = scorings[i][cur_node]
        if cur_score == -np.inf:
            break # Path broken
        
        next_nodes = padded_nodes_list[i+1]
        for new_node in next_nodes:
            if (cur_node,new_node) in edge_scores:
                score_removal = node_scores[cur_node]+edge_scores[(cur_node,new_node)]
                remaining_score = cur_score-score_removal
                
                # Check approximate equality for float
                if abs(remaining_score - scorings[i+1][new_node]) < 1e-9:
                    cur_path.append(new_node)
                    cur_node = new_node
                    break
    
    return cur_path     

def generate_chained_block_haplotypes(haplotype_data, nodes_list, combined_scores, num_haplotypes,
                             node_usage_penalty=10, edge_usage_penalty=10, similarity_matrices=None):
    """
    Generates chromosome length haplotypes by finding best paths through the graph
    and penalizing used nodes/edges iteratively.
    """
    
    num_layers = len(nodes_list)
    
    if similarity_matrices is None:
        # Compute sequentially if not provided to avoid nested pool issues
        similarity_matrices = [hap_statistics.get_block_hap_similarities(block) for block in haplotype_data]
    
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
        
        # Check if valid path found (length should cover all layers + I + S)
        if len(found_hap) < len(nodes_copy):
            print(f"Warning: Only partial path found for haplotype {ite}. Stopping early.")
            break

        #Now that we have our hap apply node penalties
        for i in range(1,len(found_hap)-1):
            layer = found_hap[i][0]
            used_hap = found_hap[i][1]
            
            # Apply penalty to the node used, AND nodes similar to it in that block
            reductions = (node_usage_penalty)*similarity_matrices[layer][used_hap,:]
            
            # Iterate through all haps in that block to apply similarity penalty
            for nm in range(len(reductions)):
                node_key = (layer, nm)
                if node_key in current_node_scores:
                    current_node_scores[node_key] -= reductions[nm]
        
        #And apply edge penalties
        for i in range(1,len(found_hap)-2):
            edge = (found_hap[i],found_hap[i+1])
            if edge in current_edge_scores:
                current_edge_scores[edge] -= edge_usage_penalty
        
        found_haps.append(found_hap[1:-1])

    return found_haps

def combine_chained_blocks_to_single_hap(all_haps,
                                         hap_blocks,
                                         read_error_prob = 0.02,
                                         min_total_reads=5):
    """
    Stitches blocks together into a single long haplotype.
    all_haps: BlockResults object.
    hap_blocks: List of tuples (layer, hap_index) for the path.
    """
    
    next_starting = []
    
    #Find overlap starting points
    for i in range(1,len(all_haps)):
        start_position = all_haps[i].positions[0]
        insertion_point = np.searchsorted(all_haps[i-1].positions, start_position)
        next_starting.append(insertion_point)
        
    final_haplotype = []
    final_locations = []
    
    for i in range(len(all_haps)-1):
        hap_here = hap_blocks[i][1]
        hap_next = hap_blocks[i+1][1]
        start_point = next_starting[i]
        overlap_length = len(all_haps[i].positions) - start_point  

        # Append non-overlapping start of first block
        if i == 0:
            start_data = all_haps[i].haplotypes[hap_here][:start_point]
            final_haplotype.extend(start_data)
            final_locations.extend(all_haps[i].positions[:start_point])
        
        cur_overlap_data = all_haps[i].haplotypes[hap_here][start_point:]
        next_overlap_data = all_haps[i+1].haplotypes[hap_next][:overlap_length]
        
        # Get read counts to weight the merger
        reads_sum = np.sum(all_haps[i].reads_count_matrix[:,start_point:,:], axis=0)
        num_samples = all_haps[i].reads_count_matrix.shape[0]
    
        hap_priors = []
        for j in range(len(reads_sum)):
            # Calculate weight based on coverage
            if sum(reads_sum[j]) >= max(min_total_reads, read_error_prob*num_samples):
                rat_val = (1+reads_sum[j][1])/(2+reads_sum[j][0]+reads_sum[j][1])
            else:
                rat_val = read_error_prob
            hap_priors.append([rat_val, 1-rat_val])
        hap_priors = np.array(hap_priors)
        
        # Merge overlapping probabilities
        new_probs = []
        
        # Ensure lengths match before looping (handle edge case of truncated overlap)
        process_len = min(len(hap_priors), len(cur_overlap_data), len(next_overlap_data))
        
        for j in range(process_len):
            new_val = analysis_utils.combine_probabilities(
                cur_overlap_data[j],
                next_overlap_data[j],
                hap_priors[j]
            )
            new_probs.append(new_val)
        new_probs = np.array(new_probs)
        
        final_haplotype.extend(new_probs)
        final_locations.extend(all_haps[i+1].positions[:overlap_length])
        
        # If this is the last intersection, append the rest of the last block
        if i == len(all_haps)-2:
            end_data = all_haps[i+1].haplotypes[hap_next][overlap_length:]
            final_haplotype.extend(end_data)
            final_locations.extend(all_haps[i+1].positions[overlap_length:])
    
    return [np.array(final_locations), np.array(final_haplotype)]

def combine_all_blocks_to_long_haps(all_haps,
                                    hap_blocks_list,
                                    read_error_prob = 0.02,
                                    min_total_reads=5,
                                    num_processes=16):
    """
    Multithreaded stitching of all long haplotypes.
    Optimized: Uses shared memory for all_haps.
    """
    
    shared_context = {'blocks': all_haps}
    
    worker = partial(_worker_combine_chained_blocks, 
                     read_error_prob=read_error_prob, 
                     min_total_reads=min_total_reads)

    with Pool(num_processes, initializer=_init_shared_data, initargs=(shared_context,)) as processing_pool:
        # Map over the list of haplotype paths (small)
        processing_results = processing_pool.map(worker, hap_blocks_list)
    
    # All paths share the same locations (approximately), so take the first one's locs
    sites_loc = processing_results[0][0]
    long_haps = [processing_results[x][1] for x in range(len(processing_results))]
    
    return [sites_loc,long_haps]

def compute_likeliest_path(full_combined_genotypes,sample_probs,site_locations,
                            keep_flags=None,recomb_rate=10**-8,value_error_rate=10**-3):
    """
    Compute the likeliest path explaining the sample using a combination of
    genotypes from full_combined_genotypes via the Viterbi algorithm.
    """
    
    data_shape = full_combined_genotypes.shape
    
    if keep_flags is None:
        keep_flags = np.array([1 for _ in range(data_shape[2])])
    
    if keep_flags.dtype != int:
        keep_flags = np.array(keep_flags,dtype=int)
        
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
            
            last_site_loc = site_locations[loc]
            
            num_possible_switches = num_rows+num_cols-1
            non_switch_prob = (1-recomb_rate)**distance_since_last_site
            each_switch_prob= (1-non_switch_prob)/num_possible_switches
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

            all_combs = np.einsum("i,jkl->jkil",site_sample_val,genotype_vals)
            prob_data_given_comb = np.einsum("ijkl,kl->ij",all_combs,value_error_matrix)
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
    
    reversed_path = []
    cur_place = np.unravel_index(np.argmax(log_probs_history[-1]), log_probs_history[-1].shape)
    reversed_path.append(cur_place)
    
    for i in range(len(prev_best_history)-1,-1,-1):
        cur_place = prev_best_history[i][cur_place[0],cur_place[1]]
        reversed_path.append(cur_place)
    
    return (reversed_path,log_probs_history,prev_best_history)

def generate_long_haplotypes_naive(block_results, num_long_haps,
                             node_usage_penalty=10, edge_usage_penalty=10, num_processes=16):
    """
    Takes the already calculated BlockResults and chains them together
    to form num_long_haps full chromosome haplotypes.
    
    Optimized: Uses shared memory for parallel steps to prevent massive serialization overhead.
    """

    # 1. Match haplotypes between neighboring blocks (Overlap & Samples)
    # Passed num_processes to enable parallel, shared-memory overlap matching
    hap_matching_overlap = match_haplotypes_by_overlap_probabalistic(block_results, num_processes=num_processes)
    
    # Run Shared Memory Optimized Sample Matching
    hap_matching_samples = match_haplotypes_by_samples_probabalistic(block_results, num_processes=num_processes)
    
    node_names = hap_matching_overlap[0]
    
    # 2. Combine scores
    combined_scores = get_combined_hap_score(hap_matching_overlap, hap_matching_samples[2])

    # 3. Pre-calculate similarity matrices in parallel (Shared Memory Optimized)
    shared_context = {'blocks': block_results}
    with Pool(num_processes, initializer=_init_shared_data, initargs=(shared_context,)) as p:
        similarity_matrices = p.map(
            _worker_get_similarities,
            range(len(block_results))
        )
    
    # 4. Find optimal paths (Chaining) - Sequential logic, fast
    chained_block_haps = generate_chained_block_haplotypes(
        block_results,
        node_names,
        combined_scores,
        num_long_haps,
        node_usage_penalty=node_usage_penalty,
        edge_usage_penalty=edge_usage_penalty,
        similarity_matrices=similarity_matrices
    )
    
    # 5. Stitch blocks together (Shared Memory Optimized)
    final_long_haps = combine_all_blocks_to_long_haps(block_results, chained_block_haps, num_processes=num_processes)

    return (block_results, final_long_haps)   

#%%
def get_match_probabilities(full_combined_genotypes,sample_probs,site_locations,
                            keep_flags=None,recomb_rate=10**-8,value_error_rate=10**-3):
    """
    HMM to match sample genotype to best combination of haplotypes.
    """
    
    data_shape = full_combined_genotypes.shape
    
    if keep_flags is None:
        keep_flags = np.array([1 for _ in range(data_shape[2])])
    
    if keep_flags.dtype != int:
        keep_flags = np.array(keep_flags,dtype=int)
        
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
    
    #Iterate backwards
    for loc in range(num_sites-1,-1,-1):
        if keep_flags[loc] != 1:
            posterior_probabilities.append(copy.deepcopy(current_probabilities))
        else:
            if last_site_loc == None:
                distance_since_last_site = 0
            else:
                distance_since_last_site = last_site_loc-site_locations[loc]
            last_site_loc = site_locations[loc]
                
            updated_prior = analysis_utils.recombination_fudge(current_probabilities,
                                                distance_since_last_site,
                                                recomb_rate=recomb_rate)
            
            site_sample_val = sample_probs[loc]
            genotype_vals = full_combined_genotypes[:,:,loc,:]

            all_combs = np.einsum("i,jkl->jkil",site_sample_val,genotype_vals)
            prob_data_given_comb = np.einsum("ijkl,kl->ij",all_combs,value_error_matrix)
            nonorm_prob_comb_given_data = np.einsum("ij,ij->ij",prob_data_given_comb,updated_prior)
            prob_comb_given_data = nonorm_prob_comb_given_data/np.sum(nonorm_prob_comb_given_data)
            
            posterior_probabilities.append(copy.deepcopy(prob_comb_given_data))
            current_probabilities = prob_comb_given_data
    
    #Iterate forward
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
                
            updated_prior = analysis_utils.recombination_fudge(current_probabilities,
                                                distance_since_last_site,
                                                recomb_rate=recomb_rate)
            site_sample_val = sample_probs[loc]
            genotype_vals = full_combined_genotypes[:,:,loc,:]

            all_combs = np.einsum("i,jkl->jkil",site_sample_val,genotype_vals)
            prob_data_given_comb = np.einsum("ijkl,kl->ij",all_combs,value_error_matrix)
            nonorm_prob_comb_given_data = np.einsum("ij,ij->ij",prob_data_given_comb,updated_prior)
            prob_comb_given_data = nonorm_prob_comb_given_data/np.sum(nonorm_prob_comb_given_data)
            
            posterior_probabilities.append(prob_comb_given_data)
            current_probabilities = prob_comb_given_data
            
    upp_tri_post = []
    for item in posterior_probabilities:
        new = np.triu(item+np.transpose(item)-np.diag(np.diag(item)))
        upp_tri_post.append(new)
        
    max_loc = []
    for i in range(len(upp_tri_post)):
        max_loc.append(np.unravel_index(upp_tri_post[i].argmax(), upp_tri_post[i].shape))
    
    return (max_loc,upp_tri_post)
        
def get_full_match_probs(full_combined_genotypes,all_sample_probs,site_locations,
                            keep_flags=None,recomb_rate=10**-8,value_error_rate=10**-3, num_processes=16):         
    """
    Multithreaded version to match all samples to their haplotype combinations.
    Optimized: Uses shared memory to avoid pickling the massive genotype tensor and sample probabilities.
    """
    
    # Prepare shared context
    shared_context = {
        'genotypes': full_combined_genotypes,
        'sample_probs': all_sample_probs,
        'locations': site_locations,
        'keep_flags': keep_flags
    }
    
    worker = partial(_worker_get_match_probabilities, 
                     recomb_rate=recomb_rate, 
                     value_error_rate=value_error_rate)

    with Pool(num_processes, initializer=_init_shared_data, initargs=(shared_context,)) as processing_pool:
        # Pass the indices of samples (0 to N-1)
        # Workers look up specific sample_probs[i] from shared memory
        num_samples = len(all_sample_probs)
        results = processing_pool.map(worker, range(num_samples))
    
    return results