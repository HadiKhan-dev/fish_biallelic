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
    
    Stores BOTH forward and backward transition matrices:
      - Forward:  P(later_hap | earlier_hap)  — used when scoring from past to future
      - Backward: P(earlier_hap | later_hap)  — used when scoring from future to past
    
    Attributes:
        num_blocks (int): Total number of genomic blocks.
        mappings (list): List of dicts mapping {real_hap_key: dense_index} for each block.
        reverse_mappings (list): List of lists mapping [dense_index] -> real_hap_key.
        registry (dict): Forward matrices. registry[from_block][to_block] = Log_Prob_Matrix (N_from x N_to).
        backward_registry (dict): Backward matrices. backward_registry[later_block][earlier_block] = Log_Prob_Matrix (N_later x N_earlier).
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
            
        # 2. Build Forward Dense Matrices Registry
        self.registry = {}

        for gap in transition_mesh.keys():
            # Forward Dictionary: P(Next | Curr)
            fwd_dict = transition_mesh[gap][0] 
            
            if fwd_dict is None: continue

            for i_idx, transitions in fwd_dict.items():
                j_idx = i_idx + gap
                
                if j_idx >= self.num_blocks: continue
                
                n_from = len(self.mappings[i_idx])
                n_to = len(self.mappings[j_idx])
                
                mat = np.full((n_from, n_to), -np.inf, dtype=np.float32)
                
                for (key_from, key_to), prob in transitions.items():
                    u_key = key_from[1]
                    v_key = key_to[1]
                    
                    if u_key in self.mappings[i_idx] and v_key in self.mappings[j_idx]:
                        r = self.mappings[i_idx][u_key]
                        c = self.mappings[j_idx][v_key]
                        mat[r, c] = math.log(prob)
                
                if i_idx not in self.registry: self.registry[i_idx] = {}
                self.registry[i_idx][j_idx] = mat

        # 3. Build Backward Dense Matrices Registry
        self.backward_registry = {}

        for gap in transition_mesh.keys():
            # Backward Dictionary: P(Prev | Curr)
            bwd_dict = transition_mesh[gap][1]
            
            if bwd_dict is None: continue
            
            for j_idx, transitions in bwd_dict.items():
                i_idx = j_idx - gap  # the earlier block
                
                if i_idx < 0: continue
                
                # Rows = later block (j_idx), Cols = earlier block (i_idx)
                # mat[h_later, h_earlier] = log P(h_earlier | h_later)
                n_later = len(self.mappings[j_idx])
                n_earlier = len(self.mappings[i_idx])
                
                mat = np.full((n_later, n_earlier), -np.inf, dtype=np.float32)
                
                for (key_from, key_to), prob in transitions.items():
                    u_key = key_from[1]  # hap in later block (j_idx)
                    v_key = key_to[1]    # hap in earlier block (i_idx)
                    
                    if u_key in self.mappings[j_idx] and v_key in self.mappings[i_idx]:
                        r = self.mappings[j_idx][u_key]
                        c = self.mappings[i_idx][v_key]
                        mat[r, c] = math.log(prob)
                
                if j_idx not in self.backward_registry:
                    self.backward_registry[j_idx] = {}
                self.backward_registry[j_idx][i_idx] = mat

    def get_transition_matrix(self, from_block, to_block):
        """
        Returns the forward dense log-probability matrix P(to_block | from_block).
        from_block < to_block.
        Returns None if no transition data exists for this pair.
        """
        if from_block in self.registry and to_block in self.registry[from_block]:
            return self.registry[from_block][to_block]
        return None

    def get_backward_matrix(self, later_block, earlier_block):
        """
        Returns the backward dense log-probability matrix P(earlier_hap | later_hap).
        later_block > earlier_block.
        Shape: (n_later_haps, n_earlier_haps)
        Entry [h_later, h_earlier] = log P(h_earlier | h_later)
        Returns None if no transition data exists for this pair.
        """
        if later_block in self.backward_registry and earlier_block in self.backward_registry[later_block]:
            return self.backward_registry[later_block][earlier_block]
        return None

    def get_key_from_dense(self, block_idx, dense_idx):
        """Converts internal dense integer index back to original haplotype key."""
        return self.reverse_mappings[block_idx][dense_idx]
    
    def get_num_haps(self, block_idx):
        """Returns the number of haplotypes in a specific block."""
        return len(self.reverse_mappings[block_idx])

# =============================================================================
# 2. BIDIRECTIONAL BEAM SEARCH
# =============================================================================

def run_bidirectional_beam_search(haps_data, transition_mesh, beam_width=200, 
                                  max_gap=None, verbose=True):
    """
    Bidirectional Beam Search - forward pass builds paths, backward pass refines
    using both backward transitions AND the forward path as context.
    
    Uses MEAN-PER-STEP scoring: at each block, adds the mean of transition scores
    rather than the sum. This prevents later blocks from dominating the score
    (block N would otherwise add N transitions while block 1 adds only 1).
    
    For each forward path F = [f0, f1, ..., fN-1]:
    - Backward pass builds B = [b0, b1, ..., bN-1]
    - When choosing b_k, score includes:
      - P(b_k | b_{k+1}, b_{k+2}, ...) - backward transitions from future blocks
      - P(b_k | f_0, f_1, ..., f_{k-1}) - forward transitions from past blocks
    
    Args:
        haps_data: List of BlockResult objects.
        transition_mesh: TransitionMesh object.
        beam_width: Number of paths to keep.
        max_gap: Maximum gap between blocks to consider transitions for.
                 If None, use all available transitions (no limit).
                 gap=1 means adjacent blocks (essentially 0bp between them).
                 gap=G spans G-1 full blocks of physical distance.
        verbose: If True, print progress.
    
    Returns:
        List of (path_indices, score) sorted by score.
    """
    if verbose:
        print("Building FastMesh for Bidirectional Beam Search...")
    fast_mesh = FastMesh(haps_data, transition_mesh)
    num_blocks = fast_mesh.num_blocks
    
    if num_blocks < 2:
        n_0 = fast_mesh.get_num_haps(0)
        return [([h], 0.0) for h in range(n_0)]
    
    # Effective max_gap: if None, use all blocks
    eff_max_gap = max_gap if max_gap is not None else num_blocks
    
    if verbose:
        print(f"  max_gap={max_gap} (effective: {eff_max_gap})")
    
    # ========== FORWARD PASS ==========
    # Build full paths from block 0 to block N-1
    # Uses MEAN scoring: add mean of transitions at each step
    if verbose:
        print(f"Pass 1: Forward Beam (blocks 0 to {num_blocks-1})...")
    
    n_0 = fast_mesh.get_num_haps(0)
    # (path, cumulative_score, tip)
    forward_beam = [([h], 0.0, h) for h in range(n_0)]
    
    for curr_block in range(1, num_blocks):
        candidates = []
        n_curr = fast_mesh.get_num_haps(curr_block)
        
        # Earliest past block to consider (limited by max_gap)
        earliest_past = max(0, curr_block - eff_max_gap)
        
        for path, path_score, _ in forward_beam:
            # Compute transition scores from previous blocks within max_gap to curr_block
            # Using FORWARD matrices: P(curr_h | past_h)
            transition_to_curr = np.zeros(n_curr, dtype=np.float32)
            n_transitions = 0
            
            for past_idx in range(earliest_past, curr_block):
                past_h = path[past_idx]
                mat = fast_mesh.get_transition_matrix(past_idx, curr_block)
                if mat is not None:
                    transition_to_curr += mat[past_h, :]
                    n_transitions += 1
            
            # MEAN scoring: divide by number of transitions actually used
            if n_transitions > 0:
                mean_transition = transition_to_curr / n_transitions
            else:
                mean_transition = transition_to_curr
            
            for h in range(n_curr):
                new_path = path + [h]
                new_score = path_score + mean_transition[h]
                candidates.append((new_path, new_score, h))
        
        forward_beam = _enforce_diversity_simple(candidates, beam_width, n_curr)
    
    if verbose:
        print(f"  Forward beam has {len(forward_beam)} full paths")
        print(f"Pass 2: Backward Beam conditioned on forward paths...")
    
    # ========== BACKWARD PASS (conditioned on each forward path) ==========
    all_backward_paths = []
    
    for fwd_path, fwd_score, _ in forward_beam:
        # For this forward path, build backward paths
        # backward_beam contains (path, bwd_score, tip) where path is in forward order [k, k+1, ..., N-1]
        
        n_last = fast_mesh.get_num_haps(num_blocks - 1)
        
        # Initialize backward beam with forward path context at the last block
        # Score for haplotype h at block N-1 = mean of FORWARD transitions from past blocks within max_gap
        # (These are forward because we're asking: given the forward path's past, how likely is h?)
        earliest_past = max(0, (num_blocks - 1) - eff_max_gap)
        
        backward_beam = []
        for h in range(n_last):
            init_score = 0.0
            n_transitions = 0
            for past_block in range(earliest_past, num_blocks - 1):
                past_h = fwd_path[past_block]
                mat = fast_mesh.get_transition_matrix(past_block, num_blocks - 1)
                if mat is not None:
                    init_score += mat[past_h, h]
                    n_transitions += 1
            if n_transitions > 0:
                init_score /= n_transitions
            backward_beam.append(([h], init_score, h))
        
        for curr_block in range(num_blocks - 2, -1, -1):
            candidates = []
            n_curr = fast_mesh.get_num_haps(curr_block)
            
            for bwd_path, bwd_score, _ in backward_beam:
                # bwd_path = [h_{curr+1}, h_{curr+2}, ..., h_{N-1}] in forward order
                
                # Score for extending with haplotype h at curr_block:
                # 1. BACKWARD transitions: P(curr_h | future_h) for future blocks in bwd_path
                # 2. FORWARD transitions: P(curr_h | past_h) for past blocks in fwd_path
                
                total_trans = np.zeros(n_curr, dtype=np.float32)
                n_transitions = 0
                
                # Backward transitions: use BACKWARD matrix P(curr_h | future_h)
                for future_idx, future_h in enumerate(bwd_path):
                    future_block = curr_block + 1 + future_idx
                    if future_block - curr_block > eff_max_gap:
                        break
                    bwd_mat = fast_mesh.get_backward_matrix(future_block, curr_block)
                    if bwd_mat is not None:
                        # bwd_mat[future_h, :] gives log P(curr_h | future_h) for all curr_h
                        total_trans += bwd_mat[future_h, :]
                        n_transitions += 1
                
                # Forward transitions: use FORWARD matrix P(curr_h | past_h)
                # Limited to past blocks within max_gap
                earliest_past = max(0, curr_block - eff_max_gap)
                for past_block in range(earliest_past, curr_block):
                    past_h = fwd_path[past_block]
                    mat = fast_mesh.get_transition_matrix(past_block, curr_block)
                    if mat is not None:
                        total_trans += mat[past_h, :]
                        n_transitions += 1
                
                # MEAN scoring: divide by number of transitions
                if n_transitions > 0:
                    mean_trans = total_trans / n_transitions
                else:
                    mean_trans = total_trans
                
                for h in range(n_curr):
                    new_path = [h] + bwd_path
                    new_score = bwd_score + mean_trans[h]
                    candidates.append((new_path, new_score, h))
            
            backward_beam = _enforce_diversity_simple(candidates, beam_width, n_curr)
        
        # backward_beam now contains full paths [0, 1, ..., N-1]
        # Add best backward path(s) to results
        for bwd_path, bwd_score, _ in backward_beam:
            all_backward_paths.append((bwd_path, bwd_score))
    
    if verbose:
        print(f"  Generated {len(all_backward_paths)} backward paths from {len(forward_beam)} forward paths")
    
    # ========== COMBINE AND SELECT FINAL ==========
    # De-duplicate paths and keep best score for each
    path_scores = {}
    for path, score in all_backward_paths:
        path_tuple = tuple(path)
        if path_tuple not in path_scores or score > path_scores[path_tuple]:
            path_scores[path_tuple] = score
    
    # Also add forward paths (they might be good too)
    for path, score, _ in forward_beam:
        path_tuple = tuple(path)
        if path_tuple not in path_scores or score > path_scores[path_tuple]:
            path_scores[path_tuple] = score
    
    # Sort and return
    final_results = [(list(p), s) for p, s in path_scores.items()]
    final_results.sort(key=lambda x: x[1], reverse=True)
    
    if verbose:
        print(f"  Final unique paths: {len(final_results)}")
    
    return final_results[:beam_width]


def _enforce_diversity_simple(candidates, beam_width, num_tips):
    """Simple diversity enforcement for forward/backward passes."""
    if not candidates:
        return []
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    best_per_tip = {}
    remainder = []
    
    for path, score, tip in candidates:
        if tip not in best_per_tip:
            best_per_tip[tip] = (path, score, tip)
        else:
            remainder.append((path, score, tip))
    
    beam = list(best_per_tip.values())
    slots_left = beam_width - len(beam)
    if slots_left > 0:
        beam.extend(remainder[:slots_left])
    
    return beam


def run_full_mesh_beam_search(haps_data, transition_mesh, beam_width=100, 
                              max_gap=None, weight_decay_func=None, verbose=True):
    """
    Main Driver for Bidirectional Beam Search.
    
    Args:
        haps_data: List of BlockResult objects.
        transition_mesh: TransitionMesh object.
        beam_width: Number of paths to keep.
        max_gap: Maximum gap between blocks for transition lookups. None = no limit.
        weight_decay_func: Function to weight long-range connections (unused).
        verbose: If True, print progress messages.
        
    Returns:
        List of (path_indices, score) sorted by score.
    """
    return run_bidirectional_beam_search(haps_data, transition_mesh, beam_width, 
                                         max_gap=max_gap, verbose=verbose)

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