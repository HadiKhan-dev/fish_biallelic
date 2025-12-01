import numpy as np
import math
import time
from multiprocess import Pool
import itertools
import copy
import seaborn as sns
from collections import defaultdict


import analysis_utils
import block_haplotypes
import hap_statistics
import simulate_sequences
import block_linking_em
#%%

class FastMesh:
    """
    Helper to convert the dictionary-based mesh into dense numpy matrices
    for O(1) vectorised lookups.
    """
    def __init__(self, haps_data, full_mesh):
        self.haps_data = haps_data
        self.num_blocks = len(haps_data)
        
        # We map the sparse "Haplotype Keys" (e.g. 5, 12, 33) to Dense Indices (0, 1, 2)
        # mapping[block_index][hap_key] = dense_index
        self.mappings = [] 
        self.reverse_mappings = []
        
        for i in range(self.num_blocks):
            keys = sorted(list(haps_data[i][3].keys()))
            self.mappings.append({k: idx for idx, k in enumerate(keys)})
            self.reverse_mappings.append(keys)
            
        # Cache the log-probability matrices
        # cache[gap][direction][block_index] = Matrix(Prev_Haps x Curr_Haps)
        self.cache = {}
        
        # Pre-process the mesh into matrices
        # We only need to process the gaps that actually exist in full_mesh
        for gap in full_mesh.keys():
            self.cache[gap] = [{}, {}] # [Forward, Backward]
            
            # 1. Forward Matrices
            for i in full_mesh[gap][0].keys():
                if i + gap >= self.num_blocks: continue
                
                prev_map = self.mappings[i]
                curr_map = self.mappings[i+gap]
                
                # Matrix shape: (Num_Prev_Haps, Num_Curr_Haps)
                mat = np.full((len(prev_map), len(curr_map)), -np.inf)
                
                # Fill from dict
                raw_dict = full_mesh[gap][0][i]
                for (k_prev, k_curr), val in raw_dict.items():
                    # k_prev is ((i, hap), (i+gap, hap))... wait, check key structure
                    # Usually keys are ((i, u), (i+gap, v))
                    u = k_prev[1]
                    v = k_curr[1]
                    if u in prev_map and v in curr_map:
                        mat[prev_map[u], curr_map[v]] = math.log(val)
                
                self.cache[gap][0][i] = mat

            # 2. Backward Matrices
            for i in full_mesh[gap][1].keys():
                if i - gap < 0: continue
                
                # Key structure for backward: ((i, u), (i-gap, v))
                curr_map = self.mappings[i]
                prev_map = self.mappings[i-gap]
                
                mat = np.full((len(curr_map), len(prev_map)), -np.inf)
                
                raw_dict = full_mesh[gap][1][i]
                for (k_curr, k_prev), val in raw_dict.items():
                    u = k_curr[1]
                    v = k_prev[1]
                    if u in curr_map and v in prev_map:
                        mat[curr_map[u], prev_map[v]] = math.log(val)
                        
                self.cache[gap][1][i] = mat

    def get_score_matrix(self, gap, direction, block_idx):
        """Returns the dense log-prob matrix or None if missing."""
        try:
            return self.cache[gap][direction][block_idx]
        except KeyError:
            return None

    def to_dense(self, block_idx, hap_key):
        """Convert Hap Key to Dense Index"""
        return self.mappings[block_idx][hap_key]

    def to_key(self, block_idx, dense_idx):
        """Convert Dense Index to Hap Key"""
        return self.reverse_mappings[block_idx][dense_idx]
    
    def get_all_keys(self, block_idx):
        return self.reverse_mappings[block_idx]
    
#%%
def prune_beam_stratified(candidates, target_size, 
                          min_diff_percent=0.02, 
                          min_diff_blocks=3,  # NEW: Force at least this many blocks diff
                          tip_length=3, 
                          min_per_tip=5):
    """
    Selects candidates with Stratified Diversity (Bucket by Tip).
    Enforces BOTH percentage difference and absolute block count difference.
    """
    if not candidates:
        return []
    
    # 1. Bucket by Tip
    candidates_by_tip = {}
    
    for path_list, score in candidates:
        path_arr = np.array(path_list)
        if len(path_arr) <= tip_length:
            tip = tuple(path_arr)
        else:
            tip = tuple(path_arr[-tip_length:])
        
        if tip not in candidates_by_tip:
            candidates_by_tip[tip] = []
        candidates_by_tip[tip].append((path_list, score))
        
    kept_candidates = []
    
    # Helper to check diversity
    def is_distinct(new_path, existing_paths_matrix):
        if len(existing_paths_matrix) == 0: return True
        
        new_arr = np.array(new_path)
        
        # Boolean matrix of mismatches
        mismatches = (existing_paths_matrix != new_arr)
        
        # Count mismatches per row
        diff_counts = np.sum(mismatches, axis=1)
        
        # 1. Check Absolute Count (Must differ by at least N blocks)
        min_count = np.min(diff_counts)
        if min_count < min_diff_blocks:
            return False
            
        # 2. Check Percentage (Must differ by X%)
        # (Only relevant for very long chains where N blocks is tiny percent)
        diff_percs = diff_counts / len(new_path)
        if np.min(diff_percs) < min_diff_percent:
            return False
            
        return True

    # 2. Enforce Minimum Representation per Tip
    remaining_candidates = [] 
    
    for tip, group in candidates_by_tip.items():
        tip_kept_paths = [] 
        
        for path_list, score in group:
            if len(tip_kept_paths) < min_per_tip:
                if is_distinct(path_list, np.array(tip_kept_paths)):
                    kept_candidates.append((path_list, score))
                    tip_kept_paths.append(path_list)
                else:
                    remaining_candidates.append((path_list, score))
            else:
                remaining_candidates.append((path_list, score))

    # 3. Backfill Global
    if len(kept_candidates) < target_size:
        remaining_candidates.sort(key=lambda x: x[1], reverse=True)
        
        all_kept_matrix = np.array([c[0] for c in kept_candidates]) if kept_candidates else np.array([])
        
        for path_list, score in remaining_candidates:
            if len(kept_candidates) >= target_size:
                break
            
            if len(all_kept_matrix) == 0 or is_distinct(path_list, all_kept_matrix):
                kept_candidates.append((path_list, score))
                if len(all_kept_matrix) == 0:
                    all_kept_matrix = np.array([path_list])
                else:
                    all_kept_matrix = np.vstack([all_kept_matrix, np.array(path_list)])
                    
    # Final Sort
    kept_candidates.sort(key=lambda x: x[1], reverse=True)
    return kept_candidates[:target_size]

def run_beam_search_initial_diverse(fast_mesh, num_candidates=100, 
                                    diversity_diff_percent=0.02, min_diff_blocks=3, tip_length=3):
    
    num_blocks = fast_mesh.num_blocks
    first_block_keys = fast_mesh.get_all_keys(0)
    beam = [([i], 0.0) for i in range(len(first_block_keys))]
    
    for i in range(1, num_blocks):
        candidates = []
        valid_gaps = [g for g in fast_mesh.cache.keys() if i - g >= 0]
        gap_matrices = []
        for g in valid_gaps:
            mat = fast_mesh.get_score_matrix(g, 0, i - g)
            scale = 1/math.sqrt(g) #scale = math.sqrt(g)
            gap_matrices.append((g, mat, scale))
            
        num_curr_options = len(fast_mesh.get_all_keys(i))
        
        for path, score in beam:
            next_scores = np.full(num_curr_options, score)
            for gap, mat, scale in gap_matrices:
                prev_idx = path[i - gap]
                if mat is not None:
                    next_scores += (mat[prev_idx, :] * scale)
            
            k_best = min(20, len(next_scores))
            best_ext_indices = np.argpartition(next_scores, -k_best)[-k_best:]
            
            for next_idx in best_ext_indices:
                new_score = next_scores[next_idx]
                if new_score > -np.inf:
                    new_path = path + [next_idx] 
                    candidates.append((new_path, new_score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Adjust threshold for early blocks where length < min_diff_blocks
        curr_min_blocks = min_diff_blocks
        if i <= min_diff_blocks:
            curr_min_blocks = 0 # Accept duplicates at the very start to allow branching
            
        beam = prune_beam_stratified(
            candidates, 
            target_size=num_candidates, 
            min_diff_percent=diversity_diff_percent,
            min_diff_blocks=curr_min_blocks,
            tip_length=tip_length,
            min_per_tip=5
        )

    return beam

def run_beam_refinement(beam, fast_mesh, num_candidates, direction="backward", 
                        diversity_diff_percent=0.02, min_diff_blocks=2, tip_length=5,
                        allowed_gaps=None): # NEW ARGUMENT
    """
    Refinement Pass with optional gap filtering.
    """
    num_blocks = fast_mesh.num_blocks
    
    if direction == "backward":
        indices = range(num_blocks - 1, -1, -1)
    else:
        indices = range(0, num_blocks)
        
    for i in indices:
        new_beam_candidates = {} 
        
        relevant_matrices = []
        
        # Filter gaps if specified
        available_gaps = fast_mesh.cache.keys()
        if allowed_gaps is not None:
            available_gaps = [g for g in available_gaps if g in allowed_gaps]
            
        # Look Backward (i-g -> i)
        for g in available_gaps:
            if i - g >= 0:
                mat = fast_mesh.get_score_matrix(g, 0, i - g)
                if mat is not None:
                    scale = 1/math.sqrt(g)
                    relevant_matrices.append(('back', g, mat, scale))

        # Look Forward (i -> i+g)
        for g in available_gaps:
            if i + g < num_blocks:
                mat = fast_mesh.get_score_matrix(g, 0, i) 
                if mat is not None:
                    scale = 1/math.sqrt(g)
                    relevant_matrices.append(('fwd', g, mat, scale))

        num_options_at_i = len(fast_mesh.get_all_keys(i))
        
        # ... (Rest of the function is identical to previous version) ...
        
        for path_list, current_total_score in beam:
            path = np.array(path_list)
            current_val = path[i]
            
            current_local_score = 0.0
            potential_local_scores = np.zeros(num_options_at_i)
            
            for mode, gap, mat, scale in relevant_matrices:
                if mode == 'back':
                    prev_val = path[i - gap]
                    current_local_score += (mat[prev_val, current_val] * scale)
                    potential_local_scores += (mat[prev_val, :] * scale)
                elif mode == 'fwd':
                    next_val = path[i + gap]
                    current_local_score += (mat[current_val, next_val] * scale)
                    potential_local_scores += (mat[:, next_val] * scale)

            base_score = current_total_score - current_local_score
            new_total_scores = base_score + potential_local_scores
            
            # Keep top 3 local swaps
            k_best = min(3, len(new_total_scores))
            if k_best > 0:
                best_indices = np.argpartition(new_total_scores, -k_best)[-k_best:]
    
                for new_val in best_indices:
                    new_score = new_total_scores[new_val]
                    if new_score > -np.inf:
                        new_path = list(path)
                        new_path[i] = new_val
                        t_path = tuple(new_path)
                        
                        if t_path not in new_beam_candidates:
                            new_beam_candidates[t_path] = new_score
                        elif new_score > new_beam_candidates[t_path]:
                            new_beam_candidates[t_path] = new_score

        # Sort and Prune
        candidates = sorted(
            [(list(p), s) for p, s in new_beam_candidates.items()], 
            key=lambda x: x[1], reverse=True
        )
        
        beam = prune_beam_stratified(
            candidates, 
            target_size=num_candidates, 
            min_diff_percent=diversity_diff_percent,
            min_diff_blocks=min_diff_blocks,
            tip_length=tip_length,
            min_per_tip=5
        )
        
    return beam

def convert_mesh_to_haplotype_diverse(full_samples_data, full_sites,
                                      haps_data, full_mesh,
                                      num_candidates=200,
                                      diversity_diff_percent=0.02,
                                      min_diff_blocks=3,
                                      tip_length=5):
    
    fast_mesh = FastMesh(haps_data, full_mesh)
    
    # 1. Initial Construction (Uses all gaps)
    beam_results = run_beam_search_initial_diverse(
        fast_mesh, num_candidates, diversity_diff_percent, min_diff_blocks, tip_length
    )
    
    # 2. Refinement (Backward - All Gaps)
    beam_results = run_beam_refinement(
        beam_results, fast_mesh, num_candidates, "backward", 
        diversity_diff_percent, min_diff_blocks, tip_length
    )
    
    # 3. Refinement (Forward - All Gaps)
    beam_results = run_beam_refinement(
        beam_results, fast_mesh, num_candidates, "forward", 
        diversity_diff_percent, min_diff_blocks, tip_length
    )
    
    # # --- 4. POLISHING STEP (GAP 1 ONLY) ---
    # # This fixes single-block errors where long-range noise overpowered local signal.
    # # We run it Backward then Forward to ripple corrections through.
    
    # print("Running Gap-1 Polishing...")
    
    # beam_results = run_beam_refinement(
    #     beam_results, fast_mesh, num_candidates, "backward", 
    #     diversity_diff_percent, min_diff_blocks, tip_length,
    #     allowed_gaps=[1] # ONLY GAP 1
    # )
    
    # beam_results = run_beam_refinement(
    #     beam_results, fast_mesh, num_candidates, "forward", 
    #     diversity_diff_percent, min_diff_blocks, tip_length,
    #     allowed_gaps=[1] # ONLY GAP 1
    # )
    
    return beam_results, fast_mesh

#%%
def get_max_contiguous_difference(path_a, path_b):
    """
    Returns the length of the longest contiguous run of mismatches 
    between two paths (of block labels).
    """
    arr_a = np.array(path_a)
    arr_b = np.array(path_b)
    
    # Boolean array: True where they differ
    mismatches = (arr_a != arr_b)
    
    # Find longest run of True
    max_run = 0
    current_run = 0
    
    for is_diff in mismatches:
        if is_diff:
            current_run += 1
        else:
            if current_run > max_run:
                max_run = current_run
            current_run = 0
            
    # Check final run
    if current_run > max_run:
        max_run = current_run
        
    return max_run

def select_diverse_founders(beam_results, fast_mesh, haps_data, 
                            num_founders=6, 
                            min_total_diff_percent=0.10, # Must differ by at least 10% total
                            min_contiguous_diff=5):      # AND contain a run of 5 different blocks
    """
    Selects founders that are distinct based on BIOLOGICAL logic.
    Filters out 'sparkle' noise (scattered differences) while keeping 
    recombination segments.
    """
    
    final_founders = []
    
    # 1. The first founder is always the highest scoring path
    best_path, best_score = beam_results[0]
    final_founders.append(best_path)
    
    print(f"Founder 1 selected. Score: {best_score:.2f}")
    
    # 2. Iterate through candidates
    current_beam_idx = 1
    
    while len(final_founders) < num_founders and current_beam_idx < len(beam_results):
        candidate_path, candidate_score = beam_results[current_beam_idx]
        current_beam_idx += 1
        
        is_distinct = True
        
        # Compare candidate against ALL currently selected founders
        for existing_founder in final_founders:
            
            # Metric 1: Total Hamming Distance
            total_diff_len = np.sum(np.array(candidate_path) != np.array(existing_founder))
            total_diff_perc = total_diff_len / len(candidate_path)
            
            # Metric 2: Longest Contiguous Run of Differences
            # (This is the anti-noise filter)
            max_run = get_max_contiguous_difference(candidate_path, existing_founder)
            
            # CONDITION:
            # To be a "New Founder", it must be significantly different...
            # BUT if the difference is just scattered noise (max_run is small), 
            # we assume it's the SAME founder with artifacts.
            
            if max_run < min_contiguous_diff:
                is_distinct = False
                # Optional debug to see what we are rejecting
                # print(f"Rejecting candidate (Score {candidate_score:.2f}). Too similar to Founder (Max run {max_run} < {min_contiguous_diff})")
                break
            
            # Double check total diff just in case
            if total_diff_perc < min_total_diff_percent:
                is_distinct = False
                break
        
        if is_distinct:
            print(f"Founder {len(final_founders)+1} selected. Score: {candidate_score:.2f}.")
            final_founders.append(candidate_path)
            
    if len(final_founders) < num_founders:
        print(f"Warning: Only found {len(final_founders)} distinct founders out of {len(beam_results)} candidates.")

    # 3. Reconstruct Data
    reconstructed_data = []
    
    for path_indices in final_founders:
        combined_positions = []
        combined_haplotype = []
        
        for i, dense_idx in enumerate(path_indices):
            hap_key = fast_mesh.to_key(i, dense_idx)
            combined_positions.extend(haps_data[i][0])
            combined_haplotype.extend(haps_data[i][3][hap_key])
            
        reconstructed_data.append((np.array(combined_positions), np.array(combined_haplotype), path_indices))
        
    return reconstructed_data

#%%
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
# 1. Run the heavy calculation ONCE (No re-EM, no re-mesh)
# Use a larger candidate pool (200) to ensure "Sibling 2" is in the list somewhere
raw_candidates, fast_mesh_obj = convert_mesh_to_haplotype_diverse(
    all_likelihoods,
    all_sites,
    test_haps,
    final_mesh1,
    num_candidates=5000
)

# 2. Select the 6 distinct parents
# min_diff_threshold=0.05 means a haplotype is "new" if it differs by at least 5% 
# from all previously selected ones. Adjust this based on how related your parents are.
founders_list = select_diverse_founders(
    raw_candidates, 
    fast_mesh_obj, 
    test_haps, 
    num_founders=20,
    min_total_diff_percent=0.10, # Must differ by at least 10% total
    min_contiguous_diff=5
)

print(time.time()-start)


#%%
check = founders_list[16]
for i in range(len(haplotype_data)):
    print(i,analysis_utils.calc_perc_difference(check[1],
        haplotype_data[i][:54559],calc_type="haploid"))
    print(len(np.where(
        simulate_sequences.concretify_haps([check[1]])[0] !=
        simulate_sequences.concretify_haps([haplotype_data[i][:54559]])[0])[0]))
    print(len(check[1]))
    print()
    
#%%
original_haplotype_number = 3
found_haplotype_number = 16

ideal_path_indices = analysis_utils.map_haplotype_to_blocks(haplotype_data[original_haplotype_number], test_haps)
print(np.where(np.array(ideal_path_indices) != founders_list[found_haplotype_number][2]))

found_exact = False
for cand in raw_candidates:
    if list(cand[0]) == list(ideal_path_indices):
        print(f"Found exact match! Score: {cand[1]}")
        break
        
score = analysis_utils.get_path_log_likelihood(ideal_path_indices, final_mesh1, verbose=True)
print("Ideal Path Score:", score)
best_found_score = analysis_utils.get_path_log_likelihood(founders_list[found_haplotype_number][2], final_mesh1)
print("Best Found Score:", best_found_score)
print()
beam_score = analysis_utils.get_path_score_beam_view(ideal_path_indices, final_mesh1, verbose=True)
print("Ideal Path Score:", beam_score)
beam_best_found_score = analysis_utils.get_path_score_beam_view(founders_list[found_haplotype_number][2], final_mesh1)
print("Best Found Score:", beam_best_found_score)
#%%
def diagnose_block_score(block_idx, ideal_path, found_path, full_mesh):
    """
    Prints the score contribution of EVERY gap size for a specific block index.
    Compares 'ideal_path' (Truth) vs 'found_path' (Error).
    """
    print(f"\n--- DIAGNOSIS FOR BLOCK {block_idx} ---")
    print(f"Ideal Val: {ideal_path[block_idx]} | Found Val: {found_path[block_idx]}")
    
    total_ideal = 0
    total_found = 0
    
    # Check all gaps that connect to this block
    for gap in sorted(full_mesh.keys()):
        
        # 1. Incoming from Past (i-gap -> i)
        prev_idx = block_idx - gap
        if prev_idx >= 0:
            # Ideal
            key_ideal = ((prev_idx, ideal_path[prev_idx]), (block_idx, ideal_path[block_idx]))
            prob_ideal = full_mesh[gap][0][prev_idx].get(key_ideal, 0)
            log_ideal = math.log(prob_ideal) if prob_ideal > 0 else -999
            
            # Found
            key_found = ((prev_idx, found_path[prev_idx]), (block_idx, found_path[block_idx]))
            prob_found = full_mesh[gap][0][prev_idx].get(key_found, 0)
            log_found = math.log(prob_found) if prob_found > 0 else -999
            
            diff = log_found - log_ideal
            winner = "FOUND" if diff > 0 else "IDEAL"
            
            print(f"Gap {gap} (Back): Ideal={log_ideal:.2f} vs Found={log_found:.2f} | Diff={diff:.2f} -> {winner}")
            
            total_ideal += log_ideal
            total_found += log_found

        # 2. Incoming from Future (i -> i+gap) (Technically outgoing, but part of the score)
        next_idx = block_idx + gap
        if next_idx < len(ideal_path):
            # Ideal
            key_ideal = ((block_idx, ideal_path[block_idx]), (next_idx, ideal_path[next_idx]))
            prob_ideal = full_mesh[gap][0][block_idx].get(key_ideal, 0)
            log_ideal = math.log(prob_ideal) if prob_ideal > 0 else -999
            
            # Found
            key_found = ((block_idx, found_path[block_idx]), (next_idx, found_path[next_idx]))
            prob_found = full_mesh[gap][0][block_idx].get(key_found, 0)
            log_found = math.log(prob_found) if prob_found > 0 else -999
            
            diff = log_found - log_ideal
            winner = "FOUND" if diff > 0 else "IDEAL"
            
            print(f"Gap {gap} (Fwd) : Ideal={log_ideal:.2f} vs Found={log_found:.2f} | Diff={diff:.2f} -> {winner}")

            total_ideal += log_ideal
            total_found += log_found

    print(f"--- TOTAL (Uniform Weight) ---")
    print(f"Ideal: {total_ideal:.2f}")
    print(f"Found: {total_found:.2f}")
    print(f"Winner: {'FOUND (Error)' if total_found > total_ideal else 'IDEAL (Truth)'}")

# RUN IT
diagnose_block_score(19, ideal_path_indices, founders_list[1][2], final_mesh1)
#%%
test = explain_sample_viterbi([all_likelihoods[0][4][:18341],all_likelihoods[1][0][:18341]],
                              all_sites[:18341],
                              test_haps,[found_haplotypes[0][:5]],
                              recomb_rate=5*10**-8)
print(test)