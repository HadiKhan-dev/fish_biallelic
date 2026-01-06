import numpy as np
import math
import time
from collections import defaultdict

# Import the Viterbi kernel for the selection phase
from block_haplotypes import viterbi_score_selection

# =============================================================================
# 1. FAST MESH (O(1) Lookup)
# =============================================================================

class FastMesh:
    """
    Optimized container for the Transition Mesh.
    
    Converts the sparse dictionary-based probabilities from the HMM step into 
    dense NumPy log-probability matrices. This allows for O(1) lookups and 
    vectorized broadcasting during the Beam Search.
    
    Attributes:
        num_blocks (int): Total number of genomic blocks.
        mappings (list): List of dicts mapping {real_hap_key: dense_index} for each block.
        reverse_mappings (list): List of lists mapping [dense_index] -> real_hap_key.
        registry (dict): Nested dictionary storing dense matrices.
                         registry[from_block_idx][to_block_idx] = Log_Prob_Matrix (N_from x N_to).
    """
    def __init__(self, block_results, transition_mesh):
        """
        Initialize the FastMesh.

        Args:
            block_results (BlockResults): List of BlockResult objects containing local haplotypes.
            transition_mesh (TransitionMesh): The sparse mesh calculated by block_linking_em.
        """
        self.num_blocks = len(block_results)
        
        # 1. Build Index Mappings (Dense ID <-> Real Key)
        self.mappings = [] 
        self.reverse_mappings = []
        
        for i in range(self.num_blocks):
            keys = sorted(list(block_results[i].haplotypes.keys()))
            self.mappings.append({k: idx for idx, k in enumerate(keys)})
            self.reverse_mappings.append(keys)
            
        # 2. Build Dense Matrices Registry
        self.registry = {}

        # Iterate over all gap sizes available in the mesh
        for gap in transition_mesh.keys():
            # We use the Forward Dictionary: P(Next | Curr)
            # Structure: { block_idx: { ((i, u), (j, v)): prob } }
            fwd_dict = transition_mesh[gap][0] 
            
            if fwd_dict is None: continue

            for i_idx, transitions in fwd_dict.items():
                j_idx = i_idx + gap
                
                # Check bounds
                if j_idx >= self.num_blocks: continue
                
                # Create dense matrix initialized to -inf (log(0))
                n_from = len(self.mappings[i_idx])
                n_to = len(self.mappings[j_idx])
                
                mat = np.full((n_from, n_to), -np.inf, dtype=np.float32)
                
                # Fill matrix
                for (key_from, key_to), prob in transitions.items():
                    u_key = key_from[1]
                    v_key = key_to[1]
                    
                    if u_key in self.mappings[i_idx] and v_key in self.mappings[j_idx]:
                        r = self.mappings[i_idx][u_key]
                        c = self.mappings[j_idx][v_key]
                        mat[r, c] = math.log(prob)
                
                if i_idx not in self.registry: self.registry[i_idx] = {}
                self.registry[i_idx][j_idx] = mat

    def get_transition_matrix(self, from_block, to_block):
        """
        Returns the dense log-probability matrix P(to_block | from_block).
        Returns None if no transition data exists for this pair.
        """
        if from_block in self.registry and to_block in self.registry[from_block]:
            return self.registry[from_block][to_block]
        return None

    def get_key_from_dense(self, block_idx, dense_idx):
        """Converts internal dense integer index back to original haplotype key."""
        return self.reverse_mappings[block_idx][dense_idx]
    
    def get_num_haps(self, block_idx):
        """Returns the number of haplotypes in a specific block."""
        return len(self.reverse_mappings[block_idx])

# =============================================================================
# 2. BEAM SEARCH UTILITIES
# =============================================================================

def enforce_tip_diversity(candidates, beam_width, num_possible_tips):
    """
    Selects the next beam to ensure DIVERSITY.
    
    Logic:
    1. Identify the BEST scoring path for EVERY possible local haplotype (tip).
    2. Add these to the beam first (ensuring coverage).
    3. Fill the rest of the beam with the highest scoring remaining paths.
    
    Args:
        candidates: List of (path_list, score, tip_index) tuples.
        beam_width: Total size of beam.
        num_possible_tips: Number of haplotypes in the current block.
    
    Returns:
        List of (path_list, score, tip_index) selected for the beam.
    """
    # 1. Bucketing by Tip
    best_per_tip = {} # tip_idx -> (path, score, tip)
    remainder = []    # List of all other candidates
    
    # Sort candidates by score descending first
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    for cand in candidates:
        path, score, tip = cand
        
        if tip not in best_per_tip:
            # First time seeing this tip -> it's the best one (due to sort)
            best_per_tip[tip] = cand
        else:
            remainder.append(cand)
            
    # 2. Construct Beam
    # First, add the best representative for every tip
    beam = list(best_per_tip.values())
    
    # 3. Fill Remainder
    # We already sorted candidates, so 'remainder' is also sorted desc
    slots_left = beam_width - len(beam)
    if slots_left > 0:
        beam.extend(remainder[:slots_left])
        
    return beam

# =============================================================================
# 3. BIDIRECTIONAL BEAM SEARCH
# =============================================================================

def run_forward_pass_history(fast_mesh, beam_width=100):
    """
    PASS 1: Forward Scoring (Viterbi-like).
    
    Calculates the 'Best Possible History Score' to reach every node in the graph.
    This does NOT build paths; it builds a score map used to guide the backward pass.
    
    Args:
        fast_mesh: FastMesh object.
        beam_width: (Unused in pure scoring, but implies scope).
    
    Returns:
        history_scores: List of arrays. history_scores[b][h] = Max LogProb to reach Hap h at Block b from Start.
    """
    num_blocks = fast_mesh.num_blocks
    
    # Init: [Block_Idx] -> Array of scores (shape: N_haps)
    history_scores = []
    
    # Block 0: Uniform (or 0.0 log prob)
    n_0 = fast_mesh.get_num_haps(0)
    history_scores.append(np.zeros(n_0, dtype=np.float32))
    
    for curr_block in range(1, num_blocks):
        n_curr = fast_mesh.get_num_haps(curr_block)
        current_block_scores = np.full(n_curr, -np.inf, dtype=np.float32)
        
        # Look back at history (All-to-All / Multi-Gap)
        # Score[curr] = Max_over_past ( Score[past] + P(past -> curr) )
        
        updated = False
        for past_block in range(curr_block):
            mat = fast_mesh.get_transition_matrix(past_block, curr_block)
            if mat is None: continue
            
            # mat is (N_past, N_curr)
            prev_scores = history_scores[past_block]
            
            # Mask -inf scores to avoid useless computation
            valid_mask = (prev_scores > -1e20)
            if not np.any(valid_mask): continue
            
            # Broadcasting: (N_past, 1) + (N_past, N_curr) -> (N_past, N_curr)
            scores_expanded = prev_scores[valid_mask, np.newaxis] + mat[valid_mask, :]
            
            # Max over the past nodes -> Best way to reach 'curr' from 'past_block'
            best_from_this_gap = np.max(scores_expanded, axis=0)
            
            # Update current block max (Accumulate evidence from all gaps)
            current_block_scores = np.maximum(current_block_scores, best_from_this_gap)
            updated = True
            
        if not updated:
            # Handle disconnected blocks (rare)
            current_block_scores = np.zeros(n_curr, dtype=np.float32) - 1000.0
            
        history_scores.append(current_block_scores)
        
    return history_scores

def run_backward_pass_guided(fast_mesh, forward_history, beam_width=100, weight_decay_func=None):
    """
    PASS 2: Backward Construction.
    
    Builds paths from Right (End) to Left (Start).
    At each step, it selects the best parents based on:
      Score = (Accumulated Future Score) + (Transition Prob) + (Forward History Score)
      
    This guarantees global optimality (or near-optimality) because the Forward History
    encodes the best possible path from the start.
    
    Args:
        fast_mesh: FastMesh object.
        forward_history: Output of run_forward_pass_history.
        beam_width: Number of paths to keep.
        weight_decay_func: Optional scaling for long-range transitions.
    
    Returns:
        List of (path_indices, score), sorted by score.
    """
    num_blocks = fast_mesh.num_blocks
    
    # 1. Initialize Beam at Last Block
    # Candidates are stored as: ( [path_indices], backward_accumulated_score, tip_index )
    # Note: 'path_indices' are built in reverse order [N, N-1, ...]. Reversed at end.
    
    last_block_scores = forward_history[-1]
    n_last = fast_mesh.get_num_haps(num_blocks - 1)
    
    candidates = []
    for i in range(n_last):
        if last_block_scores[i] > -1e20:
            # Score for sorting = Forward History (since Backward Score is 0 at start)
            candidates.append( ([i], 0.0, i) )
            
    # Initial Pruning with Diversity
    # We sort by total likelihood (Fwd + Bwd)
    candidates.sort(key=lambda x: x[1] + last_block_scores[x[2]], reverse=True)
    beam = enforce_tip_diversity(candidates, beam_width, n_last)
    
    # 2. Iterate Backwards (from N-2 down to 0)
    # We are choosing nodes for 'curr_block' to attach to 'curr_block + 1'
    for curr_block in range(num_blocks - 2, -1, -1):
        candidates = []
        n_curr = fast_mesh.get_num_haps(curr_block)
        fwd_scores_loc = forward_history[curr_block]
        
        # Pre-fetch transition matrices from 'curr_block' to all 'future_blocks'
        # We need P(Future | Curr)
        future_matrices = []
        for future_block in range(curr_block + 1, num_blocks):
            mat = fast_mesh.get_transition_matrix(curr_block, future_block)
            if mat is not None:
                gap = future_block - curr_block
                weight = 1.0
                if weight_decay_func: weight = weight_decay_func(gap)
                future_matrices.append((future_block, mat, weight))
                
        # Expand Beam
        for path_indices, path_bwd_score, _ in beam:
            
            # The node we are immediately connecting to is the last one added to the path list
            prev_node_idx = path_indices[-1] 
            
            # Vectorized calculation for all 'u' in curr_block
            transition_total = np.zeros(n_curr, dtype=np.float32)
            
            # 1. Base Transition (Gap 1)
            # Matrix: curr -> curr+1
            mat_1 = fast_mesh.get_transition_matrix(curr_block, curr_block + 1)
            if mat_1 is not None:
                # Column for 'prev_node_idx' gives P(v | u) for all u
                transition_total += mat_1[:, prev_node_idx]
            
            # 2. Long Range Transitions
            for future_abs, mat, weight in future_matrices:
                # Find corresponding node index in the reversed path
                path_idx = (num_blocks - 1) - future_abs
                
                if path_idx >= 0 and path_idx < len(path_indices):
                    future_node = path_indices[path_idx]
                    transition_total += (mat[:, future_node] * weight)

            # 3. Total Score for Ranking
            rank_scores = fwd_scores_loc + path_bwd_score + transition_total
            
            # 4. Filter & Add Candidates
            valid_u = np.where(rank_scores > -1e20)[0]
            
            # Optimization: Only take top K extensions for this specific path 
            if len(valid_u) > 20:
                top_local = valid_u[np.argpartition(rank_scores[valid_u], -20)[-20:]]
                valid_u = top_local

            for u in valid_u:
                u_int = int(u)
                new_path = path_indices + [u_int]
                
                # New Bwd Score: Accumulate the transitions only
                new_bwd_score = path_bwd_score + transition_total[u]
                
                # Candidate: (Path, Rank_Score, Tip_Index)
                candidates.append((new_path, rank_scores[u], u_int))
                
        # 5. Select Beam with Diversity
        beam = enforce_tip_diversity(candidates, beam_width, n_curr)
    
    # 3. Finalize
    # Reverse paths to be 0 -> N
    final_results = []
    for path, score, _ in beam:
        final_results.append((path[::-1], score))
        
    final_results.sort(key=lambda x: x[1], reverse=True)
    return final_results

def run_full_mesh_beam_search(haps_data, transition_mesh, beam_width=100, 
                              weight_decay_func=None):
    """
    Main Driver for Bidirectional Beam Search.
    
    Args:
        haps_data: List of BlockResult objects.
        transition_mesh: TransitionMesh object.
        beam_width: Number of paths to keep.
        weight_decay_func: Function to weight long-range connections.
        
    Returns:
        List of (path_indices, score) sorted by score.
    """
    print("Building FastMesh for Beam Search...")
    mesh = FastMesh(haps_data, transition_mesh)
    
    print("Pass 1: Forward History Calculation...")
    fwd_history = run_forward_pass_history(mesh, beam_width)
    
    print(f"Pass 2: Guided Backward Beam Search (Width={beam_width})...")
    results = run_backward_pass_guided(mesh, fwd_history, beam_width, weight_decay_func)
    
    return results

def reconstruct_haplotypes_from_beam(beam_results, fast_mesh, haps_data):
    """
    Converts the dense indices from the beam search back into 
    full genomic data arrays.
    
    Returns:
        List of dicts with keys 'score', 'positions', 'haplotype', 'path_indices'.
    """
    reconstructed = []
    
    for path_indices, score in beam_results:
        combined_pos = []
        combined_hap = []
        
        for block_idx, dense_idx in enumerate(path_indices):
            key = fast_mesh.get_key_from_dense(block_idx, dense_idx)
            block_obj = haps_data[block_idx]
            pos = block_obj.positions
            hap_array = block_obj.haplotypes[key]
            
            combined_pos.extend(pos)
            combined_hap.extend(hap_array)
            
        reconstructed.append({
            "score": score,
            "positions": np.array(combined_pos),
            "haplotype": np.array(combined_hap),
            "path_indices": path_indices
        })
        
    return reconstructed

# =============================================================================
# 4. FOUNDER SELECTION LOGIC (Merged)
# =============================================================================

def precompute_beam_likelihoods(beam_results, block_emissions, fast_mesh):
    """
    Constructs the master tensor required for model selection.
    Tensor Shape: (Num_Samples, Num_Beam_Pairs, Num_Blocks)
    """
    num_samples = len(block_emissions[0])
    num_blocks = len(block_emissions)
    num_candidates = len(beam_results)
    
    print(f"Building Selection Tensor for {num_candidates} candidates...")
    
    map_matrix = np.zeros((num_candidates, num_blocks), dtype=int)
    
    for c_idx, (path, _) in enumerate(beam_results):
        for b_idx, dense_idx in enumerate(path):
            map_matrix[c_idx, b_idx] = dense_idx

    num_pairs = num_candidates * num_candidates
    tensor = np.zeros((num_samples, num_pairs, num_blocks), dtype=np.float32)
    
    start_t = time.time()
    
    for b in range(num_blocks):
        block_ll = block_emissions[b].likelihood_tensor
        local_indices = map_matrix[:, b]
        
        grid_i = np.repeat(local_indices, num_candidates)
        grid_j = np.tile(local_indices, num_candidates)
        
        vals = block_ll[:, grid_i, grid_j]
        tensor[:, :, b] = vals
        
    print(f"Tensor built in {time.time() - start_t:.2f}s. Size: {tensor.nbytes / 1024**2:.1f} MB")
    return tensor

def calculate_score_for_subset(subset_indices, full_tensor, recomb_penalty):
    """Helper to calculate total log-likelihood for a specific subset of candidates."""
    num_candidates = int(np.sqrt(full_tensor.shape[1]))
    
    active_cands = np.zeros(num_candidates, dtype=bool)
    active_cands[subset_indices] = True
    
    active_pairs_mat = active_cands[:, None] & active_cands[None, :]
    active_pairs_mask = active_pairs_mat.flatten()
    
    sub_tensor = full_tensor[:, active_pairs_mask, :]
    
    sample_scores = viterbi_score_selection(
        np.ascontiguousarray(sub_tensor, dtype=np.float64), 
        float(recomb_penalty)
    )
    
    return np.sum(sample_scores)

def refine_selection_by_swapping(selected_indices, full_tensor, num_total_candidates, recomb_penalty):
    """
    Iteratively attempts to swap a selected founder with an unselected one
    to improve the score.
    """
    current_indices = list(selected_indices)
    current_score = calculate_score_for_subset(current_indices, full_tensor, recomb_penalty)
    
    improved = True
    iteration = 0
    
    print("\n--- Starting Swap Refinement ---")
    
    while improved:
        improved = False
        iteration += 1
        
        unselected = [x for x in range(num_total_candidates) if x not in current_indices]
        best_swap = None
        best_swap_gain = 0.0
        
        # Try removing each current founder
        for i, remove_idx in enumerate(current_indices):
            temp_set = current_indices[:i] + current_indices[i+1:]
            
            for add_idx in unselected:
                trial_set = temp_set + [add_idx]
                trial_score = calculate_score_for_subset(trial_set, full_tensor, recomb_penalty)
                
                gain = trial_score - current_score
                if gain > 1e-4 and gain > best_swap_gain:
                    best_swap_gain = gain
                    best_swap = (remove_idx, add_idx)
        
        if best_swap:
            remove_idx, add_idx = best_swap
            print(f"  Iter {iteration}: Swapped {remove_idx} -> {add_idx} (Gain: {best_swap_gain:.2f})")
            idx_to_replace = current_indices.index(remove_idx)
            current_indices[idx_to_replace] = add_idx
            current_score += best_swap_gain
            improved = True
        else:
            print("  Converged. No further beneficial swaps found.")
            
    return current_indices, current_score

def refine_selection_by_pruning(selected_indices, full_tensor, recomb_penalty, complexity_cost_per_founder):
    """
    Backward Elimination: Iteratively tries to remove the least useful founder
    from the set to improve the BIC score.
    """
    current_indices = list(selected_indices)
    
    # Calculate initial state
    current_ll = calculate_score_for_subset(current_indices, full_tensor, recomb_penalty)
    k = len(current_indices)
    current_bic = (k * complexity_cost_per_founder) - (2 * current_ll)
    
    improved = True
    iteration = 0
    
    print("\n--- Starting Pruning Refinement (Backward Elimination) ---")
    
    while improved and len(current_indices) > 1:
        improved = False
        iteration += 1
        
        best_removal_idx = -1
        best_new_bic = float('inf')
        
        # Try removing each founder
        for i, idx_to_remove in enumerate(current_indices):
            # Form trial set
            trial_set = current_indices[:i] + current_indices[i+1:]
            
            # Calculate Score
            trial_ll = calculate_score_for_subset(trial_set, full_tensor, recomb_penalty)
            
            # Calculate BIC (k is now k-1)
            new_penalty = (len(current_indices) - 1) * complexity_cost_per_founder
            trial_bic = new_penalty - (2 * trial_ll)
            
            if trial_bic < current_bic:
                if trial_bic < best_new_bic:
                    best_new_bic = trial_bic
                    best_removal_idx = idx_to_remove
        
        if best_new_bic < current_bic:
            diff = current_bic - best_new_bic
            print(f"  Iter {iteration}: Removed Founder {best_removal_idx} (BIC Improved by {diff:.2f})")
            
            current_indices.remove(best_removal_idx)
            current_bic = best_new_bic
            improved = True
        else:
            print("  Converged. No deletions improve the BIC.")
            
    return current_indices

def select_founders_likelihood(beam_results, block_emissions, fast_mesh, 
                             max_founders=16, 
                             recomb_penalty=10.0, 
                             penalty_strength=1.0,
                             do_refinement=True):
    """
    Selects the optimal set of founders using Forward Selection (Greedy + BIC),
    Swap Refinement, and Backward Pruning.
    
    Args:
        beam_results: Output of run_full_mesh_beam_search.
        block_emissions: The StandardBlockLikelihoods object used in the HMM.
        fast_mesh: FastMesh object for index mapping.
        max_founders: Maximum K to try.
        recomb_penalty: Cost for samples to switch between founders (global).
        penalty_strength: Scaling factor for BIC complexity (controls sparsity).
        do_refinement: Whether to run the swap and prune steps.
    
    Returns:
        List of selected (path, score) tuples.
    """
    
    full_tensor = precompute_beam_likelihoods(beam_results, block_emissions, fast_mesh)
    
    num_candidates = len(beam_results)
    num_samples, _, num_blocks = full_tensor.shape
    
    log_n = math.log(num_samples * num_blocks)
    complexity_cost_per_founder = log_n * penalty_strength * num_blocks
    
    selected_indices = []
    current_best_bic = float('inf')
    
    print("\n--- Starting Forward Selection (Greedy) ---")
    
    for k in range(max_founders):
        
        best_new_idx = -1
        best_new_score = -np.inf
        
        remaining = [x for x in range(num_candidates) if x not in selected_indices]
        
        for cand_idx in remaining:
            trial_set = selected_indices + [cand_idx]
            total_ll = calculate_score_for_subset(trial_set, full_tensor, recomb_penalty)
            
            if total_ll > best_new_score:
                best_new_score = total_ll
                best_new_idx = cand_idx
        
        num_params = len(selected_indices) + 1
        penalty = num_params * complexity_cost_per_founder
        new_bic = penalty - (2 * best_new_score)
        
        print(f"Round {k+1}: Adding Cand {best_new_idx} -> LL {best_new_score:.1f}, BIC {new_bic:.1f}")
        
        if new_bic < current_best_bic:
            current_best_bic = new_bic
            selected_indices.append(best_new_idx)
            print(f"  >> ACCEPTED. (Founder {len(selected_indices)})")
        else:
            print(f"  >> REJECTED. Improvement didn't justify complexity.")
            break
    
    if do_refinement and len(selected_indices) > 1:
        # Phase 1: Swap
        selected_indices, final_score = refine_selection_by_swapping(
            selected_indices, full_tensor, num_candidates, recomb_penalty
        )
        
        # Phase 2: Prune
        selected_indices = refine_selection_by_pruning(
            selected_indices, full_tensor, recomb_penalty, complexity_cost_per_founder
        )
        
        final_score = calculate_score_for_subset(selected_indices, full_tensor, recomb_penalty)
        final_penalty = len(selected_indices) * complexity_cost_per_founder
        final_bic = final_penalty - (2 * final_score)
        print(f"Final Refined BIC: {final_bic:.1f}")

    selected_founders = []
    for idx in selected_indices:
        selected_founders.append(beam_results[idx])
        
    return selected_founders