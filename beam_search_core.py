import thread_config

import numpy as np
import math
from collections import defaultdict

# Import structural chimera pruning.
from bhd_chimera import prune_chimeras

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
# 2. SCORING HELPERS
# =============================================================================

def _compute_step_scores(path, fast_mesh, eff_max_gap):
    """
    Compute the per-step mean-transition scores for a complete path.
    
    Step k (for k=1..N-1) is the mean of forward transition log-probs
    from all earlier blocks within max_gap to block k.
    Step 0 is always 0.0 (no incoming transitions).
    
    Args:
        path: List of dense hap indices, one per block.
        fast_mesh: FastMesh object.
        eff_max_gap: Effective maximum gap for transitions.
    
    Returns:
        np.ndarray of shape (num_blocks,) with per-step scores.
    """
    num_blocks = len(path)
    step_scores = np.zeros(num_blocks, dtype=np.float64)
    
    for curr_block in range(1, num_blocks):
        earliest_past = max(0, curr_block - eff_max_gap)
        step_total = 0.0
        n_transitions = 0
        
        for past_idx in range(earliest_past, curr_block):
            mat = fast_mesh.get_transition_matrix(past_idx, curr_block)
            if mat is not None:
                step_total += mat[path[past_idx], path[curr_block]]
                n_transitions += 1
        
        if n_transitions > 0:
            step_scores[curr_block] = step_total / n_transitions
    
    return step_scores


def _recompute_affected_steps(path, step_scores, changed_block, fast_mesh, eff_max_gap):
    """
    Recompute only the step scores affected by changing one block.
    
    When block k changes, the affected steps are:
      - Step k itself (transitions from past blocks INTO k)
      - Steps k+1 through min(k + max_gap, N-1) (transitions FROM k into future blocks)
    
    Args:
        path: The full path (already modified at changed_block).
        step_scores: The current per-step scores array (will be modified in-place copy).
        changed_block: Index of the block that was changed.
        fast_mesh: FastMesh object.
        eff_max_gap: Effective maximum gap for transitions.
    
    Returns:
        New step_scores array with affected steps recomputed, and new total score.
    """
    num_blocks = len(path)
    new_steps = step_scores.copy()
    
    # Range of steps to recompute:
    # - Step changed_block (if > 0): transitions into changed_block
    # - Steps changed_block+1 .. min(changed_block+max_gap, N-1): 
    #   transitions where changed_block is a past block
    recompute_start = max(1, changed_block)  # step 0 is always 0
    recompute_end = min(changed_block + eff_max_gap, num_blocks - 1)
    
    for curr_block in range(recompute_start, recompute_end + 1):
        earliest_past = max(0, curr_block - eff_max_gap)
        step_total = 0.0
        n_transitions = 0
        
        for past_idx in range(earliest_past, curr_block):
            mat = fast_mesh.get_transition_matrix(past_idx, curr_block)
            if mat is not None:
                step_total += mat[path[past_idx], path[curr_block]]
                n_transitions += 1
        
        if n_transitions > 0:
            new_steps[curr_block] = step_total / n_transitions
        else:
            new_steps[curr_block] = 0.0
    
    return new_steps, np.sum(new_steps)

# =============================================================================
# 3. BIDIRECTIONAL BEAM SEARCH (Scaffold Refinement)
# =============================================================================

def run_bidirectional_beam_search(haps_data, transition_mesh, beam_width=200, 
                                  max_gap=None, mmr_lambda=0.7, verbose=True):
    """
    Bidirectional Beam Search with scaffold-based backward refinement.
    
    Pass 1 (Forward): Builds full paths left-to-right using MMR-based selection
    to maintain diversity. This prevents the beam from filling up with minor 
    variants of the top-scoring path while crowding out genuinely distinct paths.
    
    Pass 2 (Backward Refinement): Uses each forward path as a scaffold.
    At each block k (from N-1 down to 0), tries all possible haplotype choices
    for block k while keeping the rest of the scaffold fixed, then scores the
    full resulting path using cached per-step scores (only recomputing steps
    affected by the change). All candidates across all scaffolds are pooled
    and selected via MMR to maintain diversity.
    
    This ensures:
    - The beam_width budget is respected at every step
    - Cross-scaffold competition happens at every block, not just at the end
    - Diverse paths are maintained (via MMR) preventing dominant founders from
      crowding out rare ones
    
    Args:
        haps_data: List of BlockResult objects.
        transition_mesh: TransitionMesh object.
        beam_width: Number of paths to keep at each step.
        max_gap: Maximum gap between blocks to consider transitions for.
                 If None, use all available transitions (no limit).
        mmr_lambda: Balance between score and diversity (0=pure diversity, 1=pure score).
                    Default 0.7 gives 70% weight to score, 30% to novelty.
        verbose: If True, print progress.
    
    Returns:
        List of (path_indices, score) sorted by score descending.
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
        
        forward_beam = _select_beam_mmr_forward(candidates, beam_width, mmr_lambda)
    
    if verbose:
        print(f"  Forward beam has {len(forward_beam)} full paths")
    
    # ========== BACKWARD PASS (Scaffold Refinement with Cached Scores) ==========
    # Start with the forward paths as scaffolds.
    # At each block k (N-1 down to 0), for each scaffold, try all possible
    # hap choices at block k, score using cached per-step scores (only recomputing
    # the affected steps), pool across all scaffolds, and keep top beam_width
    # using MMR to maintain path diversity.
    
    if verbose:
        print(f"Pass 2: Backward Refinement (blocks {num_blocks-1} to 0)...")
    
    # Initialize scaffold pool from forward beam (deduplicated)
    # Each scaffold is (path_list, total_score, step_scores_array)
    scaffold_cache = {}  # path_tuple -> (path_list, total_score, step_scores)
    for path, score, _ in forward_beam:
        path_tuple = tuple(path)
        if path_tuple not in scaffold_cache:
            step_scores = _compute_step_scores(path, fast_mesh, eff_max_gap)
            scaffold_cache[path_tuple] = (list(path), np.sum(step_scores), step_scores)
    
    current_scaffolds = list(scaffold_cache.values())
    
    for refine_block in range(num_blocks - 1, -1, -1):
        n_choices = fast_mesh.get_num_haps(refine_block)
        
        # Generate candidates: for each scaffold, try every hap at refine_block
        # candidates maps path_tuple -> (path_list, total_score, step_scores)
        candidates = {}
        
        for scaffold_path, scaffold_total, scaffold_steps in current_scaffolds:
            original_h = scaffold_path[refine_block]
            
            for h in range(n_choices):
                if h == original_h:
                    # Unchanged — reuse cached scores
                    path_tuple = tuple(scaffold_path)
                    if path_tuple not in candidates:
                        candidates[path_tuple] = (scaffold_path, scaffold_total, scaffold_steps)
                else:
                    # Create variant with one block changed
                    new_path = scaffold_path[:refine_block] + [h] + scaffold_path[refine_block + 1:]
                    path_tuple = tuple(new_path)
                    
                    if path_tuple not in candidates:
                        # Recompute only affected steps
                        new_steps, new_total = _recompute_affected_steps(
                            new_path, scaffold_steps, refine_block, fast_mesh, eff_max_gap
                        )
                        candidates[path_tuple] = (new_path, new_total, new_steps)
        
        # Select top beam_width using MMR
        cand_list = list(candidates.values())  # [(path, total, steps), ...]
        current_scaffolds = _select_beam_mmr_backward(cand_list, beam_width, mmr_lambda)
    
    if verbose:
        print(f"  Backward refinement complete. {len(current_scaffolds)} paths.")
    
    # ========== FINAL OUTPUT ==========
    # Convert to (path, score) format and sort by score descending
    final_results = [(path, total) for path, total, _ in current_scaffolds]
    final_results.sort(key=lambda x: x[1], reverse=True)
    
    return final_results[:beam_width]


def _select_beam_mmr(candidates, beam_width, mmr_lambda=0.7):
    """Select high-scoring diverse path tuples with stable greedy MMR."""
    if not candidates or beam_width <= 0:
        return []
    n_candidates = len(candidates)
    if n_candidates <= beam_width:
        return candidates

    all_paths = np.array([candidate[0] for candidate in candidates],
                         dtype=np.int32)
    all_scores = np.array([candidate[1] for candidate in candidates],
                          dtype=np.float64)
    score_min = all_scores.min()
    score_max = all_scores.max()
    if score_max > score_min:
        normalized_scores = ((all_scores - score_min)
                             / (score_max - score_min))
    else:
        normalized_scores = np.ones(n_candidates, dtype=np.float64)

    selected = []
    remaining = np.ones(n_candidates, dtype=np.bool_)
    max_similarity = np.full(n_candidates, -1.0, dtype=np.float64)
    for _ in range(min(beam_width, n_candidates)):
        if not np.any(remaining):
            break
        if not selected:
            candidate_scores = np.where(remaining, all_scores, -np.inf)
            best = np.argmax(candidate_scores)
        else:
            novelty = 1.0 - max_similarity
            mmr = (mmr_lambda * normalized_scores
                   + (1.0 - mmr_lambda) * novelty)
            mmr[~remaining] = -np.inf
            best = np.argmax(mmr)
        selected.append(best)
        remaining[best] = False
        similarities = np.mean(
            all_paths == all_paths[best][None, :], axis=1
        )
        max_similarity = np.maximum(max_similarity, similarities)
    return [candidates[index] for index in selected]


def _select_beam_mmr_forward(candidates, beam_width, mmr_lambda=0.7):
    """Compatibility wrapper for forward path tuples."""
    return _select_beam_mmr(candidates, beam_width, mmr_lambda)


def _select_beam_mmr_backward(candidates, beam_width, mmr_lambda=0.7):
    """Compatibility wrapper for backward path tuples."""
    return _select_beam_mmr(candidates, beam_width, mmr_lambda)


def run_full_mesh_beam_search(haps_data, transition_mesh, beam_width=100, 
                              max_gap=None, mmr_lambda=0.7, weight_decay_func=None, verbose=True):
    """
    Main Driver for Bidirectional Beam Search.
    
    Args:
        haps_data: List of BlockResult objects.
        transition_mesh: TransitionMesh object.
        beam_width: Number of paths to keep.
        max_gap: Maximum gap between blocks for transition lookups. None = no limit.
        mmr_lambda: Balance between score and diversity (0=pure diversity, 1=pure score).
        weight_decay_func: Function to weight long-range connections (unused).
        verbose: If True, print progress messages.
        
    Returns:
        List of (path_indices, score) sorted by score.
    """
    return run_bidirectional_beam_search(haps_data, transition_mesh, beam_width, 
                                         max_gap=max_gap, mmr_lambda=mmr_lambda, verbose=verbose)

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
        # Concatenate the per-block positions/haplotypes with np.concatenate
        # rather than extending Python lists element-by-element and calling
        # np.array() on a ~1.5M-entry list: the list path iterates every site
        # as a Python object (single-threaded, ~5s at L4), while concatenate is
        # a C-level copy.  Same values in the same block order, so the result is
        # identical.
        pos_parts = []
        hap_parts = []
        for block_idx, dense_idx in enumerate(path_indices):
            key = fast_mesh.get_key_from_dense(block_idx, dense_idx)
            block_obj = haps_data[block_idx]
            pos_parts.append(np.asarray(block_obj.positions))
            hap_parts.append(np.asarray(block_obj.haplotypes[key]))

        reconstructed.append({
            "score": score,
            "positions": np.concatenate(pos_parts) if pos_parts else np.array([]),
            "haplotype": np.concatenate(hap_parts) if hap_parts else np.array([]),
            "path_indices": path_indices
        })
        
    return reconstructed

# =============================================================================
# 5. STRUCTURAL CHIMERA PRUNING FOR SUPER-BLOCKS
# =============================================================================

def prune_superblock_chimeras(super_block, max_recombs=1, max_mismatch_percent=0.25,
                               min_mean_delta_to_protect=0.25):
    """
    Applies structural chimera pruning to a super-block after reconstruction.
    
    This is a post-processing step that removes haplotypes which can be explained
    as recombinations of other haplotypes in the set, using mean_delta (average
    sample error increase) to protect essential haplotypes.
    
    Args:
        super_block: A BlockResult object containing the reconstructed super-block.
        max_recombs: Maximum recombinations for chimera detection (default 1).
        max_mismatch_percent: Maximum mismatch % for chimera (default 0.25%).
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