import numpy as np
import math
from collections import defaultdict

# =============================================================================
# 0. OPTIONAL NUMBA SETUP
# =============================================================================
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Dummy decorator if Numba is missing
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator

# =============================================================================
# 1. FAST MESH & CACHING
# =============================================================================

class FastMesh:
    """
    A high-performance wrapper around the TransitionMesh.
    Converts sparse/dictionary-based transition probabilities into dense 
    Numpy log-probability matrices for rapid lookup during Beam Search.
    
    Attributes:
        num_blocks (int): Total number of genomic blocks.
        mappings (list): List of dicts mapping {hap_key: dense_index} per block.
        reverse_mappings (list): List of lists mapping [dense_index] -> hap_key.
        cache (dict): Stores transition matrices for each gap size.
                      Structure: {gap: [forward_list, backward_list]}
                      Where forward_list[i] is the matrix for block i -> i+gap.
    """
    def __init__(self, block_results, transition_mesh):
        """
        Args:
            block_results: List of BlockResult objects.
            transition_mesh: TransitionMesh object from block_linking_em.
        """
        self.block_results = block_results
        self.num_blocks = len(block_results)
        self.mappings = [] 
        self.reverse_mappings = []
        
        # 1. Build Index Mappings
        for i in range(self.num_blocks):
            keys = sorted(list(block_results[i].haplotypes.keys()))
            self.mappings.append({k: idx for idx, k in enumerate(keys)})
            self.reverse_mappings.append(keys)
            
        self.cache = {}
        
        # 2. Build Dense Matrices for each Gap
        for gap in transition_mesh.gaps:
            model = transition_mesh[gap]
            
            # Storage for matrices indexed by block_idx
            fwd_matrices = [None] * self.num_blocks
            bwd_matrices = [None] * self.num_blocks
            
            # --- Process Forward Transitions ---
            # Model keys: ((i, u), (j, v)) -> prob
            for (key_from, key_to), prob in model.forward.items():
                i_idx = key_from[0] # Block index
                
                # Create matrix if it doesn't exist for this block
                if fwd_matrices[i_idx] is None:
                    n_curr = len(self.mappings[i_idx])
                    n_next = len(self.mappings[i_idx + gap])
                    fwd_matrices[i_idx] = np.full((n_curr, n_next), -np.inf)
                
                # Map keys to dense indices
                u = key_from[1]
                v = key_to[1]
                
                if u in self.mappings[i_idx] and v in self.mappings[i_idx + gap]:
                    r = self.mappings[i_idx][u]
                    c = self.mappings[i_idx + gap][v]
                    fwd_matrices[i_idx][r, c] = math.log(prob)

            # --- Process Backward Transitions ---
            # Model keys: ((j, v), (i, u)) -> prob (j is Next, i is Prev)
            for (key_from, key_to), prob in model.backward.items():
                j_idx = key_from[0] # Next Block index
                
                if bwd_matrices[j_idx] is None:
                    n_curr = len(self.mappings[j_idx])
                    n_prev = len(self.mappings[j_idx - gap])
                    bwd_matrices[j_idx] = np.full((n_curr, n_prev), -np.inf)
                    
                v = key_from[1]
                u = key_to[1]
                
                if v in self.mappings[j_idx] and u in self.mappings[j_idx - gap]:
                    r = self.mappings[j_idx][v]
                    c = self.mappings[j_idx - gap][u]
                    bwd_matrices[j_idx][r, c] = math.log(prob)

            self.cache[gap] = [fwd_matrices, bwd_matrices]

    def get_score_matrix(self, gap, direction, block_idx):
        """
        Retrieves the dense log-prob matrix.
        Direction: 0 for Forward (i -> i+gap), 1 for Backward (i -> i-gap).
        """
        try:
            matrix_list = self.cache[gap][direction]
            if block_idx < 0 or block_idx >= len(matrix_list):
                return None
            return matrix_list[block_idx]
        except KeyError:
            return None

    def get_all_keys(self, block_idx):
        """Returns the list of haplotype keys for a block."""
        return self.reverse_mappings[block_idx]

    def to_key(self, block_idx, dense_idx):
        """Converts a dense index back to the original haplotype key."""
        return self.reverse_mappings[block_idx][dense_idx]

# =============================================================================
# 2. BEAM SEARCH & STRATIFIED PRUNING
# =============================================================================

def prune_beam_stratified(candidates, target_size, 
                          min_diff_percent=0.02, min_diff_blocks=2, 
                          tip_length=5, min_per_tip=5):
    """
    Selects candidates by enforcing diversity across TIPS first, then Scores.
    Enforces BOTH percentage difference and absolute block count difference to
    prevent the beam from filling up with minor variations of one good path.
    
    Args:
        candidates: List of (path_list, score).
        target_size: Max number of candidates to keep.
        min_diff_percent: Minimum Hamming distance (%) required to be 'distinct'.
        min_diff_blocks: Minimum absolute block differences required.
        tip_length: Length of the path tail to consider for 'tip diversity'.
        min_per_tip: Minimum candidates to keep per unique tip.
        
    Returns:
        List of (path_list, score) sorted by score.
    """
    if not candidates: return []
    
    # 1. Bucket by Tip (Last N blocks)
    candidates_by_tip = defaultdict(list)
    for path_list, score in candidates:
        path_arr = np.array(path_list)
        if len(path_arr) <= tip_length:
            tip = tuple(path_arr)
        else:
            tip = tuple(path_arr[-tip_length:])
        candidates_by_tip[tip].append((path_list, score))
        
    kept_candidates = []
    
    # Helper: Vectorized Diversity Check
    def is_distinct(new_path, existing_matrix):
        if len(existing_matrix) == 0: return True
        
        # Compare new_path against all rows in existing_matrix
        # (Assumes paths are equal length at this stage of beam search)
        mismatches = (existing_matrix != np.array(new_path))
        diff_counts = np.sum(mismatches, axis=1)
        
        # Strict Check: Must differ by at least X blocks AND Y percent
        if np.min(diff_counts) < min_diff_blocks: return False
        if np.min(diff_counts / len(new_path)) < min_diff_percent: return False
        return True

    # 2. Enforce Minimum Representation per Tip
    # This ensures we explore different "endings" even if they currently score lower
    remaining = []
    for tip, group in candidates_by_tip.items():
        # Sort group by score descending
        group.sort(key=lambda x: x[1], reverse=True)
        
        tip_kept = []
        for path, score in group:
            if len(tip_kept) < min_per_tip:
                # Intra-tip diversity check
                if is_distinct(path, np.array(tip_kept)):
                    kept_candidates.append((path, score))
                    tip_kept.append(path)
                else: 
                    remaining.append((path, score))
            else: 
                remaining.append((path, score))

    # 3. Global Backfill
    # Fill the rest of the beam with the best remaining paths that are distinct globally
    if len(kept_candidates) < target_size:
        remaining.sort(key=lambda x: x[1], reverse=True)
        
        # Build matrix of currently kept paths for fast comparison
        if kept_candidates:
            all_kept = np.array([c[0] for c in kept_candidates])
        else:
            all_kept = np.array([])
        
        for path, score in remaining:
            if len(kept_candidates) >= target_size: break
            
            if len(all_kept) == 0:
                kept_candidates.append((path, score))
                all_kept = np.array([path])
            elif is_distinct(path, all_kept):
                kept_candidates.append((path, score))
                all_kept = np.vstack([all_kept, np.array(path)])
    
    # Final Sort
    kept_candidates.sort(key=lambda x: x[1], reverse=True)
    return kept_candidates[:target_size]

def run_beam_search_initial_diverse(fast_mesh, num_candidates=200, 
                                    diversity_diff_percent=0.02, min_diff_blocks=3, tip_length=5):
    """
    Performs the initial Forward Beam Search.
    Builds paths from Block 0 to Block N.
    """
    num_blocks = fast_mesh.num_blocks
    first_block_keys = fast_mesh.get_all_keys(0)
    
    # Init Beam: Paths of length 1
    # Format: ( [path_indices], score )
    beam = [([i], 0.0) for i in range(len(first_block_keys))]
    
    for i in range(1, num_blocks):
        candidates = []
        
        # Identify valid gaps looking backward from current block i
        valid_gaps = [g for g in fast_mesh.cache.keys() if i - g >= 0]
        
        # Pre-fetch matrices for speed
        gap_matrices = []
        for g in valid_gaps:
            # get matrix for transition: (i-g) -> i
            mat = fast_mesh.get_score_matrix(g, 0, i - g)
            scale = math.sqrt(g) # Weight longer range transitions less per step? 
            # Note: Previously sqrt(g) was used to normalize impact.
            # Usually: sum(log_prob / gap) or sum(log_prob * weight).
            # Here keeping consistent with previous logic.
            gap_matrices.append((g, mat, scale))
            
        num_curr_options = len(fast_mesh.get_all_keys(i))
        
        # Expand Beam
        for path, score in beam:
            # Base score for next step starts at current path score
            # We will add contributions from all valid history gaps
            next_scores = np.full(num_curr_options, score)
            
            for gap, mat, scale in gap_matrices:
                prev_idx = path[i - gap]
                if mat is not None:
                    # Add row corresponding to prev_idx
                    next_scores += (mat[prev_idx, :] * scale)
            
            # Local Pruning: Only keep top K extensions per path to avoid explosion
            k_best = min(20, len(next_scores))
            best_ext_indices = np.argpartition(next_scores, -k_best)[-k_best:]
            
            for next_idx in best_ext_indices:
                new_score = next_scores[next_idx]
                if new_score > -np.inf:
                    candidates.append((path + [next_idx], new_score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Apply Adaptive Pruning
        # Stricter absolute difference required as paths get longer
        curr_min_blocks = min_diff_blocks if i > 20 else 0
        
        beam = prune_beam_stratified(
            candidates, target_size=num_candidates, 
            min_diff_percent=diversity_diff_percent,
            min_diff_blocks=curr_min_blocks,
            tip_length=tip_length,
            min_per_tip=5
        )

    return beam

def run_beam_refinement(beam, fast_mesh, num_candidates, direction="backward", 
                        diversity_diff_percent=0.02, min_diff_blocks=3, tip_length=5,
                        allowed_gaps=None):
    """
    Iterative Refinement Pass.
    Walks along the existing paths (Backward or Forward) and tries to optimize 
    single-block choices using the full multi-scale mesh context.
    """
    num_blocks = fast_mesh.num_blocks
    if direction == "backward": 
        indices = range(num_blocks - 1, -1, -1)
    else: 
        indices = range(0, num_blocks)
        
    for i in indices:
        new_beam_candidates = {} 
        
        # 1. Identify relevant matrices for position i
        available_gaps = fast_mesh.cache.keys()
        if allowed_gaps is not None:
            available_gaps = [g for g in available_gaps if g in allowed_gaps]

        relevant_matrices = []
        for g in available_gaps:
            # Incoming from Past (Backward-looking edges)
            if i - g >= 0:
                mat = fast_mesh.get_score_matrix(g, 0, i - g)
                if mat is not None: 
                    # mode, gap, matrix, scale
                    relevant_matrices.append(('back', g, mat, math.sqrt(g)))
            
            # Incoming from Future (Forward-looking edges)
            if i + g < num_blocks:
                # Matrix for i -> i+g
                mat = fast_mesh.get_score_matrix(g, 0, i) 
                if mat is not None: 
                    relevant_matrices.append(('fwd', g, mat, math.sqrt(g)))

        num_options_at_i = len(fast_mesh.get_all_keys(i))
        
        # 2. Iterate over beam
        for path_list, current_total_score in beam:
            path = np.array(path_list)
            current_val = path[i]
            
            # Calculate what the score contribution OF NODE i IS currently
            current_local_score = 0.0
            
            # Calculate potential scores for ALL options at node i
            potential_local_scores = np.zeros(num_options_at_i)
            
            for mode, gap, mat, scale in relevant_matrices:
                if mode == 'back':
                    prev_val = path[i - gap]
                    # Current score includes: mat[prev, curr]
                    current_local_score += (mat[prev_val, current_val] * scale)
                    # Potentials: mat[prev, :]
                    potential_local_scores += (mat[prev_val, :] * scale)
                elif mode == 'fwd':
                    next_val = path[i + gap]
                    # Current score includes: mat[curr, next]
                    current_local_score += (mat[current_val, next_val] * scale)
                    # Potentials: mat[:, next]
                    potential_local_scores += (mat[:, next_val] * scale)

            # 3. Delta Update
            # New Total = Old Total - Contribution(Old_Node) + Contribution(New_Node)
            base_score = current_total_score - current_local_score
            new_total_scores = base_score + potential_local_scores
            
            # 4. Select top local swaps
            k_best = min(5, len(new_total_scores))
            if k_best > 0:
                best_indices = np.argpartition(new_total_scores, -k_best)[-k_best:]
                for new_val in best_indices:
                    new_score = new_total_scores[new_val]
                    if new_score > -np.inf:
                        # Construct candidate
                        new_path = list(path)
                        new_path[i] = new_val
                        t_path = tuple(new_path)
                        
                        # Dedup: Keep best score for this path
                        if t_path not in new_beam_candidates:
                            new_beam_candidates[t_path] = new_score
                        elif new_score > new_beam_candidates[t_path]:
                            new_beam_candidates[t_path] = new_score

        # 5. Repack and Prune
        candidates = sorted([(list(p), s) for p, s in new_beam_candidates.items()], 
                            key=lambda x: x[1], reverse=True)
        
        beam = prune_beam_stratified(
            candidates, target_size=num_candidates, 
            min_diff_percent=diversity_diff_percent,
            min_diff_blocks=min_diff_blocks,
            tip_length=tip_length,
            min_per_tip=5
        )
        
    return beam

def convert_mesh_to_haplotype_diverse(haps_data, full_mesh,
                                      num_candidates=200,
                                      diversity_diff_percent=0.02,
                                      min_diff_blocks=3,
                                      tip_length=5):
    """
    Main driver for the Beam Search process.
    
    Args:
        haps_data: List of BlockResult objects.
        full_mesh: TransitionMesh object.
        num_candidates: Beam width.
        diversity_diff_percent: Minimum difference % to keep distinct paths.
        
    Returns:
        (beam_results, fast_mesh_object)
    """
    
    fast_mesh = FastMesh(haps_data, full_mesh)
    
    # 1. Initial Construction
    print("Running Initial Beam Search...")
    beam_results = run_beam_search_initial_diverse(
        fast_mesh, num_candidates, diversity_diff_percent, min_diff_blocks, tip_length
    )
    
    # 2. Refinement (Backward & Forward)
    print("Refining Paths (Global)...")
    beam_results = run_beam_refinement(
        beam_results, fast_mesh, num_candidates, "backward", 
        diversity_diff_percent, min_diff_blocks, tip_length
    )
    beam_results = run_beam_refinement(
        beam_results, fast_mesh, num_candidates, "forward", 
        diversity_diff_percent, min_diff_blocks, tip_length
    )
    
    # 3. Gap-1 Polishing (Fix local noise by restricting to adjacent blocks)
    print("Refining Paths (Local Polishing)...")
    beam_results = run_beam_refinement(
        beam_results, fast_mesh, num_candidates, "backward", 
        diversity_diff_percent, min_diff_blocks, tip_length, allowed_gaps=[1]
    )
    beam_results = run_beam_refinement(
        beam_results, fast_mesh, num_candidates, "forward", 
        diversity_diff_percent, min_diff_blocks, tip_length, allowed_gaps=[1]
    )
    
    return beam_results, fast_mesh

# =============================================================================
# 3. FINAL SELECTION & RECONSTRUCTION
# =============================================================================

def get_max_contiguous_difference(path_a, path_b):
    """Calculates the longest run of consecutive mismatched blocks."""
    arr_a = np.array(path_a)
    arr_b = np.array(path_b)
    mismatches = (arr_a != arr_b)
    max_run = 0
    current_run = 0
    for is_diff in mismatches:
        if is_diff: current_run += 1
        else:
            if current_run > max_run: max_run = current_run
            current_run = 0
    if current_run > max_run: max_run = current_run
    return max_run

def select_diverse_founders(beam_results, fast_mesh, haps_data, 
                            num_founders=6, 
                            min_total_diff_percent=0.10, 
                            min_contiguous_diff=5):
    """
    Selects the final set of founders from the beam candidates using 
    biological diversity metrics (Run Length and Total Hamming Distance).
    
    Returns:
        reconstructed_data: List of tuples (positions, haplotypes, path_indices)
    """
    final_founders = []
    
    # 1. First founder is always the highest score
    best_path, best_score = beam_results[0]
    final_founders.append(best_path)
    print(f"Founder 1 selected. Score: {best_score:.2f}")
    
    # 2. Scan remaining candidates
    current_beam_idx = 1
    while len(final_founders) < num_founders and current_beam_idx < len(beam_results):
        candidate_path, candidate_score = beam_results[current_beam_idx]
        current_beam_idx += 1
        
        is_distinct = True
        for existing_founder in final_founders:
            # Check Metrics
            total_diff = np.sum(np.array(candidate_path) != np.array(existing_founder)) / len(candidate_path)
            max_run = get_max_contiguous_difference(candidate_path, existing_founder)
            
            # Prune if too similar to ANY existing founder
            if max_run < min_contiguous_diff:
                is_distinct = False
                break
            if total_diff < min_total_diff_percent:
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
            # Map dense index back to original haplotype key/array
            hap_key = fast_mesh.to_key(i, dense_idx)
            
            # Access BlockResult attributes
            block_pos = haps_data[i].positions
            block_hap = haps_data[i].haplotypes[hap_key]
            
            combined_positions.extend(block_pos)
            combined_haplotype.extend(block_hap)
            
        reconstructed_data.append((np.array(combined_positions), np.array(combined_haplotype), path_indices))
        
    return reconstructed_data