import numpy as np
import math
import time
from collections import defaultdict

# Import the Viterbi kernel for the selection phase
from block_haplotypes import viterbi_score_selection, prune_chimeras

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
            # which corresponds to block (curr_block + 1)
            prev_node_idx = path_indices[-1] 
            
            # We want to calculate the 'Transition Cost' of adding node 'u' to this path.
            # This includes the immediate link (u -> prev_node)
            # AND all long-range links (u -> node_at_curr+2, u -> node_at_curr+3...)
            
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
            # Total = Forward_History(u) + Path_Backward_Accumulated + New_Transitions
            rank_scores = fwd_scores_loc + path_bwd_score + transition_total
            
            # 4. Filter & Add Candidates
            valid_u = np.where(rank_scores > -1e20)[0]
            
            # Optimization: Only take top K extensions for this specific path 
            # to avoid explosion before the diversity filter
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
                              weight_decay_func=None, verbose=True):
    """
    Main Driver for Bidirectional Beam Search.
    
    Args:
        haps_data: List of BlockResult objects.
        transition_mesh: TransitionMesh object.
        beam_width: Number of paths to keep.
        weight_decay_func: Function to weight long-range connections.
        verbose: If True, print progress messages.
        
    Returns:
        List of (path_indices, score) sorted by score.
    """
    if verbose:
        print("Building FastMesh for Beam Search...")
    mesh = FastMesh(haps_data, transition_mesh)
    
    if verbose:
        print("Pass 1: Forward History Calculation...")
    fwd_history = run_forward_pass_history(mesh, beam_width)
    
    if verbose:
        print(f"Pass 2: Guided Backward Beam Search (Width={beam_width})...")
    results = run_backward_pass_guided(mesh, fwd_history, beam_width, weight_decay_func)
    
    return results

def reconstruct_haplotypes_from_beam(beam_results, fast_mesh, haps_data):
    """
    Converts the dense indices from the beam search back into 
    full genomic data arrays.
    
    Args:
        beam_results: Output from run_full_mesh_beam_search.
        fast_mesh: FastMesh object used during search.
        haps_data: Original BlockResults list.
        
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
# 4. FOUNDER SELECTION LOGIC (Overshoot & Prune)
# =============================================================================

def precompute_beam_likelihoods(beam_results, block_emissions, fast_mesh, verbose=True):
    """
    Constructs the master tensor required for model selection.
    Tensor Shape: (Num_Samples, Num_Beam_Pairs, Num_Blocks)
    """
    num_samples = len(block_emissions[0])
    num_blocks = len(block_emissions)
    num_candidates = len(beam_results)
    
    if verbose:
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
        
    if verbose:
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

def refine_selection_by_swapping(selected_indices, full_tensor, num_total_candidates, recomb_penalty, verbose=True):
    """
    Iteratively attempts to swap a selected founder with an unselected one
    to improve the score.
    """
    current_indices = list(selected_indices)
    current_score = calculate_score_for_subset(current_indices, full_tensor, recomb_penalty)
    
    improved = True
    iteration = 0
    
    if verbose:
        print("\n--- Starting Swap Refinement ---")
    
    while improved:
        improved = False
        iteration += 1
        
        unselected = [x for x in range(num_total_candidates) if x not in current_indices]
        best_swap = None
        best_swap_gain = 0.0
        
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
            if verbose:
                print(f"  Iter {iteration}: Swapped {remove_idx} -> {add_idx} (Gain: {best_swap_gain:.2f})")
            idx_to_replace = current_indices.index(remove_idx)
            current_indices[idx_to_replace] = add_idx
            current_score += best_swap_gain
            improved = True
        else:
            if verbose:
                print("  Converged. No further beneficial swaps found.")
            
    return current_indices, current_score

def refine_selection_by_pruning(selected_indices, full_tensor, recomb_penalty, 
                              complexity_cost_per_founder, force_prune_to=None, verbose=True):
    """
    Backward Elimination: Iteratively tries to remove the least useful founder
    from the set to improve the BIC score.
    
    Args:
        recomb_penalty: This should be passed as the LOWER pruning_recomb_penalty
                        when collapsing chimeras.
        force_prune_to (int): If set, forces deletion of least useful founders
                              until this count is reached, ignoring BIC score.
        verbose: If True, print progress messages.
    """
    current_indices = list(selected_indices)
    
    # Init Scores
    current_ll = calculate_score_for_subset(current_indices, full_tensor, recomb_penalty)
    k = len(current_indices)
    current_bic = (k * complexity_cost_per_founder) - (2 * current_ll)
    
    if verbose:
        print("\n--- Starting Pruning Refinement (Backward Elimination) ---")
    
    while True:
        k = len(current_indices)
        if k <= 1: break
        
        # Determine Mode: "Forced Pruning" or "BIC Optimization"
        forced_mode = (force_prune_to is not None) and (k > force_prune_to)
        
        best_removal_idx = -1
        best_new_bic = float('inf')
        min_ll_loss = float('inf') # For forced mode
        
        # Try removing each founder
        for i, idx_to_remove in enumerate(current_indices):
            # Form trial set
            trial_set = current_indices[:i] + current_indices[i+1:]
            
            # Calculate Score
            trial_ll = calculate_score_for_subset(trial_set, full_tensor, recomb_penalty)
            
            # Calculate BIC (k is now k-1)
            new_penalty = (k - 1) * complexity_cost_per_founder
            trial_bic = new_penalty - (2 * trial_ll)
            
            # Track best BIC
            if trial_bic < best_new_bic:
                best_new_bic = trial_bic
                if not forced_mode: best_removal_idx = idx_to_remove
            
            # Track minimum Loss (for forced mode)
            ll_loss = current_ll - trial_ll
            if ll_loss < min_ll_loss:
                min_ll_loss = ll_loss
                if forced_mode: best_removal_idx = idx_to_remove

        # Decision Logic
        should_prune = False
        reason = ""
        
        if forced_mode:
            should_prune = True
            reason = f"Forced (Count {k} > {force_prune_to})"
        elif best_new_bic < current_bic:
            should_prune = True
            reason = f"BIC Improved ({current_bic:.1f} -> {best_new_bic:.1f})"
            
        if should_prune:
            if verbose:
                print(f"  Removed Founder {best_removal_idx} [{reason}]")
            current_indices.remove(best_removal_idx)
            
            # Update current state vars for next loop
            new_ll = calculate_score_for_subset(current_indices, full_tensor, recomb_penalty)
            current_ll = new_ll
            current_bic = ((k-1) * complexity_cost_per_founder) - (2 * new_ll)
        else:
            if verbose:
                print("  Converged. No deletions improve the BIC.")
            break
            
    return current_indices

def select_founders_likelihood(beam_results, block_emissions, fast_mesh, 
                             max_founders=12, 
                             recomb_penalty=10.0, 
                             pruning_recomb_penalty=None, # NEW ARGUMENT
                             complexity_penalty_scale=0.1, 
                             do_refinement=True,
                             overshoot_limit=30,
                             use_standard_bic=False,
                             verbose=True): 
    """
    Overshoot & Prune Strategy with Configurable Penalty Scaling.
    
    Args:
        recomb_penalty: Cost to switch used during Addition/Swapping (Should be HIGH).
        pruning_recomb_penalty: Cost to switch used during Pruning (Should be LOW).
        complexity_penalty_scale: Multiplier for the complexity term.
        use_standard_bic: If True, uses log(N) scaling. If False, uses Linear N scaling.
        verbose: If True, print progress messages.
    """
    
    # Default to symmetric penalty if not specified
    if pruning_recomb_penalty is None:
        pruning_recomb_penalty = recomb_penalty

    full_tensor = precompute_beam_likelihoods(beam_results, block_emissions, fast_mesh, verbose=verbose)
    
    num_candidates = len(beam_results)
    num_samples, _, num_blocks = full_tensor.shape
    
    # --- COST FORMULA ---
    if use_standard_bic:
        # Standard BIC: k * log(n) * scale
        log_n = math.log(num_samples * num_blocks)
        complexity_cost_per_founder = log_n * complexity_penalty_scale * num_blocks
        cost_type = "Standard BIC (Log)"
    else:
        # Linear Loss: k * n * scale
        # Prevents founder explosion with large sample counts
        complexity_cost_per_founder = complexity_penalty_scale * num_samples * num_blocks
        cost_type = "Linear Loss (N)"
    
    if verbose:
        print(f"Selection Cost per Founder: {complexity_cost_per_founder:.1f} [{cost_type}, Scale={complexity_penalty_scale}]")
        print(f"Penalties -> Add/Swap: {recomb_penalty:.1f}, Prune: {pruning_recomb_penalty:.1f}")

    selected_indices = []
    current_best_bic = float('inf')
    
    if verbose:
        print(f"\n--- Starting Forward Selection (Attempting up to {overshoot_limit}) ---")
    
    # 1. FORWARD SELECTION (Use strict/high recomb_penalty)
    for k in range(overshoot_limit):
        best_new_idx = -1
        best_new_score = -np.inf
        
        remaining = [x for x in range(num_candidates) if x not in selected_indices]
        if not remaining: break
        
        for cand_idx in remaining:
            trial_set = selected_indices + [cand_idx]
            # Use HIGH penalty here to ensure we pick data that fits well without assuming switching
            total_ll = calculate_score_for_subset(trial_set, full_tensor, recomb_penalty)
            
            if total_ll > best_new_score:
                best_new_score = total_ll
                best_new_idx = cand_idx
        
        num_params = len(selected_indices) + 1
        penalty = num_params * complexity_cost_per_founder
        new_bic = penalty - (2 * best_new_score)
        
        if verbose:
            print(f"  Round {k+1}: Cand {best_new_idx} -> BIC {new_bic:.1f} (Current Best: {current_best_bic:.1f})")
        
        # Stop if BIC stops improving
        if new_bic < current_best_bic:
            current_best_bic = new_bic
            selected_indices.append(best_new_idx)
            if verbose:
                print(f"    >> ACCEPTED.")
        else:
            if verbose:
                print(f"    >> REJECTED. BIC did not improve.")
            break
    
    # 2. SWAP (Use strict/high recomb_penalty)
    if do_refinement and len(selected_indices) > 1:
        selected_indices, final_score = refine_selection_by_swapping(
            selected_indices, full_tensor, num_candidates, recomb_penalty, verbose=verbose
        )
        
        # 3. FORCE PRUNE & BIC PRUNE (Use LOOSE/LOW pruning_recomb_penalty)
        # By lowering the penalty, the Viterbi path becomes willing to switch. 
        # If a founder is just a chimera of two others, the likelihood loss of removing it 
        # becomes small (because we can just switch), but the BIC gain (complexity reward) 
        # remains high. Thus, it gets pruned.
        selected_indices = refine_selection_by_pruning(
            selected_indices, full_tensor, 
            pruning_recomb_penalty, # <--- Uses low penalty
            complexity_cost_per_founder,
            force_prune_to=max_founders,
            verbose=verbose
        )

    selected_founders = []
    for idx in selected_indices:
        selected_founders.append(beam_results[idx])
        
    return selected_founders


# =============================================================================
# 5. STRUCTURAL CHIMERA PRUNING FOR SUPER-BLOCKS
# =============================================================================

def prune_superblock_chimeras(super_block, max_recombs=1, max_mismatch_percent=1.0,
                               min_mean_delta_to_protect=0.25):
    """
    Applies structural chimera pruning to a super-block after reconstruction.
    
    This is a post-processing step that removes haplotypes which can be explained
    as recombinations of other haplotypes in the set, using mean_delta (average
    sample error increase) to protect essential haplotypes.
    
    Args:
        super_block: A BlockResult object containing the reconstructed super-block.
        max_recombs: Maximum recombinations for chimera detection (default 1).
        max_mismatch_percent: Maximum mismatch % for chimera (default 1.0%).
        min_mean_delta_to_protect: Protect haplotypes with mean_delta above this (default 0.25%).
    
    Returns:
        The super_block with pruned and reindexed haplotypes.
        Returns the original super_block unchanged if pruning cannot be performed.
    """
    if super_block is None:
        return super_block
    
    if super_block.probs_array is None:
        # Cannot prune without sample data
        return super_block
    
    if len(super_block.haplotypes) < 3:
        # Need at least 3 haplotypes to have a chimera
        return super_block
    
    # Apply structural chimera pruning
    pruned_haps = prune_chimeras(
        super_block.haplotypes,
        super_block.probs_array,
        max_recombs=max_recombs,
        max_mismatch_percent=max_mismatch_percent,
        min_mean_delta_to_protect=min_mean_delta_to_protect
    )
    
    # Reindex haplotypes to be sequential
    super_block.haplotypes = {i: v for i, v in enumerate(pruned_haps.values())}
    
    return super_block