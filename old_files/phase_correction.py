import numpy as np
import pandas as pd
import math
from typing import List, Tuple
from multiprocess import Pool

from paint_samples import SamplePainting, PaintedChunk, BlockPainting

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("WARNING: Numba not found. Phase correction will be slow.")
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator

# =============================================================================
# 1. INTERVAL INTERSECTION & HELPERS
# =============================================================================

def get_atomic_intervals(paintings: List[SamplePainting]):
    """
    Takes a list of SamplePaintings (Child, P1, P2) and returns a sorted
    array of unique breakpoints.
    """
    breakpoints = set()
    for p in paintings:
        for chunk in p.chunks:
            breakpoints.add(chunk.start)
            breakpoints.add(chunk.end)
    return sorted(list(breakpoints))

def founder_block_to_dense(founder_block):
    """
    Converts a BlockResult object into a dense matrix for fast Numba lookup.
    Returns: (hap_matrix, positions)
    """
    positions = founder_block.positions
    hap_dict = founder_block.haplotypes
    
    if not hap_dict:
        return np.zeros((0, 0), dtype=np.int8), positions

    max_id = max(hap_dict.keys())
    n_sites = len(positions)
    
    # Init with -1 (Missing)
    dense_haps = np.full((max_id + 1, n_sites), -1, dtype=np.int8)
    
    for fid, hap_arr in hap_dict.items():
        if hap_arr.ndim == 2:
            concrete = np.argmax(hap_arr, axis=1)
        else:
            concrete = hap_arr
        dense_haps[fid, :] = concrete
        
    return dense_haps, positions

# =============================================================================
# 2. 8-STATE TRIO VITERBI KERNEL
# =============================================================================

@njit(fastmath=True)
def run_8state_trio_viterbi(
    intervals,      # (N_Ints+1,) Coordinates
    child_ids,      # (N_Ints, 2) Founder IDs
    p1_ids,         # (N_Ints, 2)
    p2_ids,         # (N_Ints, 2)
    hap_matrix,     # (MaxID, Sites) Dense Alleles
    snp_positions,  # (Sites,)
    recomb_rate=5e-8,
    phase_penalty=20.0,
    mismatch_penalty=4.605 # ln(0.01)
):
    n_ints = len(child_ids)
    n_states = 8
    
    # DP Tables
    scores = np.zeros(n_states, dtype=np.float64)
    backpointers = np.zeros((n_ints, n_states), dtype=np.int8)
    
    # SNP Cursor
    snp_cursor = 0
    n_sites = len(snp_positions)
    wildcard_id = -1
    
    for i in range(n_ints):
        start = intervals[i]
        end = intervals[i+1]
        dist_bp = end - start
        
        # --- 1. TRANSITION COSTS ---
        theta = dist_bp * recomb_rate
        if theta > 0.5: theta = 0.5
        if theta < 1e-15: theta = 1e-15
        
        log_stay = math.log(1.0 - theta)
        log_swap = math.log(theta)
        
        c0, c1 = child_ids[i]
        is_child_hom = (c0 == c1) or (c0 == wildcard_id) or (c1 == wildcard_id)
        cost_phase_switch = 0.0 if is_child_hom else -phase_penalty
        
        # --- 2. EMISSIONS (Allele Matching) ---
        e_scores = np.zeros(n_states, dtype=np.float64)
        
        while snp_cursor < n_sites and snp_positions[snp_cursor] < start:
            snp_cursor += 1
        
        temp_cursor = snp_cursor
        
        while temp_cursor < n_sites and snp_positions[temp_cursor] < end:
            a_c0 = -1 if c0 == wildcard_id else hap_matrix[c0, temp_cursor]
            a_c1 = -1 if c1 == wildcard_id else hap_matrix[c1, temp_cursor]
            
            p1_h = p1_ids[i]
            p2_h = p2_ids[i]
            
            a_p1_0 = -1 if p1_h[0] == wildcard_id else hap_matrix[p1_h[0], temp_cursor]
            a_p1_1 = -1 if p1_h[1] == wildcard_id else hap_matrix[p1_h[1], temp_cursor]
            
            a_p2_0 = -1 if p2_h[0] == wildcard_id else hap_matrix[p2_h[0], temp_cursor]
            a_p2_1 = -1 if p2_h[1] == wildcard_id else hap_matrix[p2_h[1], temp_cursor]
            
            for state in range(n_states):
                p1_choice = (state >> 0) & 1
                p2_choice = (state >> 1) & 1
                phase     = (state >> 2) & 1 # 0=Direct, 1=Flipped
                
                val_p1 = a_p1_1 if p1_choice == 1 else a_p1_0
                val_p2 = a_p2_1 if p2_choice == 1 else a_p2_0
                
                match_1 = False
                match_2 = False
                
                if phase == 0: # Direct
                    match_1 = (val_p1 == -1) or (a_c0 == -1) or (val_p1 == a_c0)
                    match_2 = (val_p2 == -1) or (a_c1 == -1) or (val_p2 == a_c1)
                else: # Flipped
                    match_1 = (val_p1 == -1) or (a_c1 == -1) or (val_p1 == a_c1)
                    match_2 = (val_p2 == -1) or (a_c0 == -1) or (val_p2 == a_c0)
                
                if not match_1: e_scores[state] -= mismatch_penalty
                if not match_2: e_scores[state] -= mismatch_penalty
            
            temp_cursor += 1
            
        # --- 3. TRANSITION UPDATE ---
        prev = scores.copy()
        
        for curr_k in range(n_states):
            curr_p1 = (curr_k >> 0) & 1
            curr_p2 = (curr_k >> 1) & 1
            curr_ph = (curr_k >> 2) & 1
            
            best_prev_score = -1e20
            best_prev_k = -1
            
            for prev_k in range(n_states):
                prev_p1 = (prev_k >> 0) & 1
                prev_p2 = (prev_k >> 1) & 1
                prev_ph = (prev_k >> 2) & 1
                
                cost = 0.0
                if prev_p1 == curr_p1: cost += log_stay
                else:                  cost += log_swap
                if prev_p2 == curr_p2: cost += log_stay
                else:                  cost += log_swap
                if prev_ph != curr_ph: cost += cost_phase_switch
                    
                score = prev[prev_k] + cost
                if score > best_prev_score:
                    best_prev_score = score
                    best_prev_k = prev_k
            
            scores[curr_k] = best_prev_score + e_scores[curr_k]
            backpointers[i, curr_k] = best_prev_k
            
    # Traceback
    path = np.zeros(n_ints, dtype=np.int8)
    best_final_k = -1
    best_final_score = -1e20
    
    for k in range(n_states):
        if scores[k] > best_final_score:
            best_final_score = scores[k]
            best_final_k = k
            
    curr = best_final_k
    for i in range(n_ints - 1, -1, -1):
        path[i] = curr
        curr = backpointers[i, curr]
        
    return path

# =============================================================================
# 3. RECONSTRUCTION
# =============================================================================

def reconstruct_from_trio_path(original_painting: SamplePainting, 
                               intervals: List[int], 
                               path: np.ndarray,
                               p1_ids: np.ndarray,
                               p2_ids: np.ndarray) -> List[PaintedChunk]:
    new_chunks = []
    
    curr_start = -1
    curr_end = -1
    curr_h1 = -1
    curr_h2 = -1
    
    for i in range(len(intervals) - 1):
        start, end = intervals[i], intervals[i+1]
        state = path[i]
        
        p1_choice = (state >> 0) & 1
        p2_choice = (state >> 1) & 1
        
        new_h1 = p1_ids[i][p1_choice] 
        new_h2 = p2_ids[i][p2_choice]
        
        if curr_h1 == -1:
            curr_start, curr_end = start, end
            curr_h1, curr_h2 = new_h1, new_h2
        else:
            if (start == curr_end) and (new_h1 == curr_h1) and (new_h2 == curr_h2):
                curr_end = end
            else:
                new_chunks.append(PaintedChunk(curr_start, curr_end, curr_h1, curr_h2))
                curr_start, curr_end = start, end
                curr_h1, curr_h2 = new_h1, new_h2
                
    if curr_h1 != -1:
        new_chunks.append(PaintedChunk(curr_start, curr_end, curr_h1, curr_h2))
        
    return new_chunks

# =============================================================================
# 4. CONTIG-LEVEL WORKER & DRIVER
# =============================================================================

def process_single_contig_phase(args):
    """
    Worker: Runs iterative correction for one contig using minimal passed data.
    """
    region_name, original_bp, hap_matrix, snp_positions, parent_map, sample_names, num_rounds, config = args
    
    MISMATCH_PENALTY = config['mismatch']
    PHASE_PENALTY    = config['phase']
    RECOMB_RATE      = config['recomb']
    
    # Initialize state
    current_bp = original_bp
    
    # Iterative Refinement
    for round_idx in range(num_rounds):
        new_samples = []
        
        for i, sample_name in enumerate(sample_names):
            current_sample_obj = current_bp[i]
            
            if sample_name not in parent_map:
                new_samples.append(current_sample_obj)
                continue
                
            p1_name, p2_name = parent_map[sample_name]
            try:
                p1_idx = sample_names.index(p1_name)
                p2_idx = sample_names.index(p2_name)
            except ValueError:
                new_samples.append(current_sample_obj)
                continue
            
            p1_obj = current_bp[p1_idx]
            p2_obj = current_bp[p2_idx]
            
            # A. Intervals
            intervals = get_atomic_intervals([current_sample_obj, p1_obj, p2_obj])
            n_ints = len(intervals) - 1
            
            if n_ints == 0:
                new_samples.append(current_sample_obj)
                continue
            
            # B. ID Lookup
            c_ids = np.zeros((n_ints, 2), dtype=np.int32)
            p1_ids = np.zeros((n_ints, 2), dtype=np.int32)
            p2_ids = np.zeros((n_ints, 2), dtype=np.int32)
            
            c_cur = 0; p1_cur = 0; p2_cur = 0
            
            for k in range(n_ints):
                start = intervals[k]
                while current_sample_obj.chunks[c_cur].end <= start: c_cur += 1
                while p1_obj.chunks[p1_cur].end <= start: p1_cur += 1
                while p2_obj.chunks[p2_cur].end <= start: p2_cur += 1
                
                c_ids[k] = (current_sample_obj.chunks[c_cur].hap1, current_sample_obj.chunks[c_cur].hap2)
                p1_ids[k] = (p1_obj.chunks[p1_cur].hap1, p1_obj.chunks[p1_cur].hap2)
                p2_ids[k] = (p2_obj.chunks[p2_cur].hap1, p2_obj.chunks[p2_cur].hap2)
            
            # C. 8-State Viterbi
            intervals_arr = np.array(intervals, dtype=np.int64)
            path = run_8state_trio_viterbi(
                intervals_arr, c_ids, p1_ids, p2_ids,
                hap_matrix, snp_positions,
                recomb_rate=RECOMB_RATE,
                phase_penalty=PHASE_PENALTY,
                mismatch_penalty=MISMATCH_PENALTY
            )
            
            # D. Reconstruct
            new_chunks = reconstruct_from_trio_path(current_sample_obj, intervals, path, p1_ids, p2_ids)
            new_samples.append(SamplePainting(i, new_chunks))
            
        # Update for next round
        current_bp = BlockPainting((original_bp.start_pos, original_bp.end_pos), new_samples)
        
    return region_name, current_bp

def correct_phase_all_contigs(multi_contig_results, pedigree_df, sample_names, num_rounds=3, num_processes=8):
    """
    Parallel Driver: Corrects phase for all contigs independently.
    Optimized to pass only essential data to workers to prevent OOM.
    """
    
    config = {
        'mismatch': 4.605,
        'phase': 20.0,
        'recomb': 5e-8
    }
    
    parent_map = {}
    for _, row in pedigree_df.iterrows():
        if pd.notna(row['Parent1']) and pd.notna(row['Parent2']):
            parent_map[row['Sample']] = (row['Parent1'], row['Parent2'])
            
    print(f"\n--- Phase Correction (8-State Trio HMM, {num_rounds} rounds, {num_processes} jobs) ---")
    
    # Create Task List
    tasks = []
    
    for r_name, data in multi_contig_results.items():
        # 1. Identify Inputs
        if 'control_painting' in data:
            painting_key = 'control_painting'
            founder_key = 'control_founder_block'
        elif 'final_painting' in data:
            painting_key = 'final_painting'
            founder_key = 'block_results' 
        else:
            print(f"Skipping {r_name} (Missing input)")
            continue
            
        original_bp = data[painting_key]
        founder_obj = data.get(founder_key)
        
        # 2. Prepare Dense Genotypes (MAIN PROCESS)
        # This converts the dict-of-lists to numpy arrays *before* forking
        # significantly reducing pickle overhead and memory fragmentation.
        if hasattr(founder_obj, 'haplotypes'):
            hap_matrix, snp_positions = founder_block_to_dense(founder_obj)
        elif isinstance(founder_obj, list) or hasattr(founder_obj, 'blocks'):
            if hasattr(founder_obj, 'blocks') and len(founder_obj) == 1:
                 hap_matrix, snp_positions = founder_block_to_dense(founder_obj[0])
            else:
                print(f"Skipping {r_name} (Founder format issue)")
                continue
        else:
            print(f"Skipping {r_name} (No founder data)")
            continue
            
        # 3. Pack Task
        # Note: We do NOT pass 'data' (the huge dict containing reads/probs).
        tasks.append((
            r_name, 
            original_bp, 
            hap_matrix, 
            snp_positions, 
            parent_map, 
            sample_names, 
            num_rounds, 
            config
        ))
        
    # Execute
    if num_processes > 1:
        with Pool(num_processes) as pool:
            results = pool.map(process_single_contig_phase, tasks)
    else:
        results = list(map(process_single_contig_phase, tasks))
        
    # Unpack Results
    for r_name, corrected_bp in results:
        if corrected_bp is not None:
            multi_contig_results[r_name]['corrected_painting'] = corrected_bp
        else:
            print(f"  {r_name}: Returned None")
            
    return multi_contig_results