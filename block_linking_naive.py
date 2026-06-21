import thread_config

import numpy as np
import math
import copy
import multiprocessing as _mp
import multiprocessing.pool
from multiprocessing import shared_memory as _shm
import warnings
from contextlib import contextmanager
from functools import partial

import analysis_utils
import hap_statistics
import block_haplotypes

warnings.filterwarnings("ignore")
np.seterr(divide='ignore',invalid="ignore")

# =============================================================================
# FORKSERVER POOL
# =============================================================================
# Workers spawn from a lightweight forkserver process (~500 MB) instead of
# forking from the parent, which may hold 100+ GB after loading multiple
# chromosomes.  This eliminates the O(parent_RSS) fork overhead that caused
# the naive linker to slow down linearly as chromosomes accumulated.
#
# Block data is passed directly as task arguments (~20 KB per block after
# stripping probs_array), NOT via pool initializer.  This avoids the
# O(num_workers × data_size) serialization cost that made the first
# forkserver attempt 10x slower.
#
# For pool 6 (get_full_match_probs), large numpy arrays are placed in
# POSIX SharedMemory (/dev/shm) for zero-copy worker access.

try:
    _forkserver_ctx = _mp.get_context('forkserver')
except (ValueError, AttributeError):
    _forkserver_ctx = _mp.get_context('fork')

class _ForkserverPool(multiprocessing.pool.Pool):
    """A Pool using forkserver context."""
    def __init__(self, *args, **kwargs):
        kwargs['context'] = _forkserver_ctx
        super().__init__(*args, **kwargs)

@contextmanager
def _safe_forkserver_pool(processes, initializer=None, initargs=()):
    """
    Create a forkserver pool with __main__ safety.
    
    Temporarily clears __main__.__file__ so forkserver workers don't
    re-execute the entry script.  Restores it on exit.
    """
    import sys as _sys
    _main_mod = _sys.modules.get('__main__')
    _saved_file = getattr(_main_mod, '__file__', None)
    _saved_spec = getattr(_main_mod, '__spec__', None)
    if _main_mod is not None:
        if hasattr(_main_mod, '__file__'):
            del _main_mod.__file__
        _main_mod.__spec__ = None
    try:
        with _ForkserverPool(processes=processes, initializer=initializer, initargs=initargs) as pool:
            yield pool
    finally:
        if _main_mod is not None:
            if _saved_file is not None:
                _main_mod.__file__ = _saved_file
            _main_mod.__spec__ = _saved_spec


def _strip_block(block):
    """
    Create a lightweight copy of a BlockResult without probs_array.
    
    probs_array is ~1.5 MB per block (320 samples × 200 sites × 3 × 8 bytes)
    but is never used by the naive linker's workers.  Stripping it reduces
    each block from ~1.5 MB to ~20 KB, making direct task arguments practical.
    Keeps reads_count_matrix (used by overlap merge in pools 1-3, 5).
    """
    return block_haplotypes.BlockResult(
        positions=block.positions,
        haplotypes=block.haplotypes,
        keep_flags=block.keep_flags,
        reads_count_matrix=block.reads_count_matrix,  # keep — used by overlap merge
        probs_array=None,
    )


def _strip_block_light(block):
    """
    Create a minimal copy of a BlockResult without probs_array OR reads_count_matrix.
    
    Used by Pool 4 where reads are placed in SharedMemory separately.
    The resulting block is ~5-10 KB (just positions, haplotypes, keep_flags).
    """
    return block_haplotypes.BlockResult(
        positions=block.positions,
        haplotypes=block.haplotypes,
        keep_flags=block.keep_flags,
        reads_count_matrix=None,
        probs_array=None,
    )


# --- SHARED MEMORY MANAGEMENT (Pools 4 and 6 — large numpy arrays) ---

_SHARED_DATA = {}
_SHM_REFS = []

def _init_shared_data(data_dict):
    """
    Initializer for forkserver pool workers.
    Attaches to POSIX SharedMemory segments for zero-copy numpy access.
    Non-SharedMemory values are stored directly.
    """
    global _SHARED_DATA, _SHM_REFS
    _SHARED_DATA.clear()
    for ref in _SHM_REFS:
        try: ref.close()
        except Exception: pass
    _SHM_REFS = []

    for key, meta in data_dict.items():
        if isinstance(meta, dict) and 'shm_name' in meta:
            # SharedMemory-backed array
            shm = _shm.SharedMemory(name=meta['shm_name'], create=False)
            _SHM_REFS.append(shm)
            _SHARED_DATA[key] = np.ndarray(meta['shape'], dtype=np.dtype(meta['dtype']), buffer=shm.buf)
        else:
            # Small metadata — passed directly
            _SHARED_DATA[key] = meta

def _create_shm_array(arr, label=""):
    """Copy a numpy array into POSIX SharedMemory. Returns (shm_handle, metadata_dict)."""
    arr = np.ascontiguousarray(arr)
    shm = _shm.SharedMemory(create=True, size=arr.nbytes)
    view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    np.copyto(view, arr)
    meta = {'shm_name': shm.name, 'shape': arr.shape, 'dtype': str(arr.dtype)}
    return shm, meta


# --- DIRECT WORKER FUNCTIONS (Pools 1–5) ---
# Workers receive block data as task arguments, NOT from shared memory.
# This works with forkserver because the per-task pickle is small (~20 KB
# per stripped block), unlike the old initializer approach which pickled
# ALL blocks for every worker process.

def _worker_match_overlap_direct(args):
    """
    Pool 1 worker: overlap similarity between block[i] and block[i+1].
    Receives both blocks directly as arguments.
    """
    i, curr_block, next_block = args
    gap_bridge_score = 1.0

    if len(curr_block.positions) == 0 or len(next_block.positions) == 0:
        return {}

    start_position_next = next_block.positions[0]
    overlap_start_idx = np.searchsorted(curr_block.positions, start_position_next)
    overlap_length = len(curr_block.positions) - overlap_start_idx

    if overlap_length <= 0:
        bridge_edges = {}
        for curr_name in curr_block.haplotypes.keys():
            for next_name in next_block.haplotypes.keys():
                bridge_edges[((i, curr_name), (i+1, next_name))] = gap_bridge_score
        return bridge_edges

    curr_haps = curr_block.haplotypes
    next_haps = next_block.haplotypes

    cur_ends = {k: curr_haps[k][overlap_start_idx:] for k in curr_haps.keys()}
    next_ends = {k: next_haps[k][:overlap_length] for k in next_haps.keys()}

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

    min_len = min(len(cur_keep_flags), len(next_keep_flags))
    cur_keep_flags = cur_keep_flags[:min_len]
    next_keep_flags = next_keep_flags[:min_len]

    similarities = {}
    for first_name in cur_ends.keys():
        first_new_hap = cur_ends[first_name][:min_len][cur_keep_flags]
        for second_name in next_ends.keys():
            second_new_hap = next_ends[second_name][:min_len][next_keep_flags]
            common_size = len(first_new_hap)
            if common_size > 0:
                haps_dist = 100 * analysis_utils.calc_distance(
                    first_new_hap, second_new_hap, calc_type="haploid") / common_size
            else:
                haps_dist = 0
            similarity = 0 if haps_dist > 50 else 2 * (50 - haps_dist)
            similarities[((i, first_name), (i+1, second_name))] = similarity

    transform_similarities = {}
    for item, sim_val in similarities.items():
        val = sim_val / 100.0
        transform_similarities[item] = 100 * (val**2)

    return transform_similarities


def _worker_combined_best_hap_matches_direct(args):
    """Pool 2 worker: best hap matches for a single block."""
    block_idx, block = args
    return hap_statistics.combined_best_hap_matches(block)


def _worker_hap_matching_comparison_direct(args):
    """Pool 3 worker: hap matching comparison between two blocks."""
    block_idx_1, block_idx_2, block_1, block_2, matches_1, matches_2 = args
    # Build minimal containers that hap_matching_comparison can index into
    haps_data = {block_idx_1: block_1, block_idx_2: block_2}
    matches_data = {block_idx_1: matches_1, block_idx_2: matches_2}

    class _IndexableDict:
        """Allows dict[int] access for hap_matching_comparison."""
        def __init__(self, d): self._d = d
        def __getitem__(self, idx): return self._d[idx]

    return hap_statistics.hap_matching_comparison(
        _IndexableDict(haps_data),
        _IndexableDict(matches_data),
        block_idx_1,
        block_idx_2
    )


def _worker_combine_chained_blocks_direct(args):
    """
    Pool 4 worker: stitch a chain of blocks into one long haplotype.
    Reads light blocks from _SHARED_DATA, reconstructs reads_count_matrix
    from SharedMemory on demand.
    """
    chain_data, read_error_prob, min_total_reads = args
    
    light_blocks = _SHARED_DATA['blocks']
    reads_index = _SHARED_DATA['reads_index']
    reads_flat = _SHARED_DATA.get('reads_flat')  # may be None if no reads exist
    
    # Reconstruct blocks with reads for the blocks this chain touches
    # (only ~10-20 blocks per chain, not all 850)
    needed_indices = set(b[0] for b in chain_data)
    # Also need adjacent blocks for overlap calculation
    for b in chain_data:
        idx = b[0]
        if idx > 0:
            needed_indices.add(idx - 1)
        if idx < len(light_blocks) - 1:
            needed_indices.add(idx + 1)
    
    # Build full block list with reads restored where needed
    full_blocks = []
    for i in range(len(light_blocks)):
        lb = light_blocks[i]
        if i in needed_indices and reads_index[i] is not None and reads_flat is not None:
            offset, n_samples, n_sites = reads_index[i]
            rcm = reads_flat[offset : offset + n_samples * n_sites * 2].reshape(n_samples, n_sites, 2)
        else:
            rcm = lb.reads_count_matrix  # None
        full_blocks.append(block_haplotypes.BlockResult(
            positions=lb.positions,
            haplotypes=lb.haplotypes,
            keep_flags=lb.keep_flags,
            reads_count_matrix=rcm,
            probs_array=None,
        ))
    
    reconstituted = block_haplotypes.BlockResults(full_blocks)
    return combine_chained_blocks_to_single_hap(
        reconstituted, chain_data,
        read_error_prob=read_error_prob,
        min_total_reads=min_total_reads
    )


def _worker_get_similarities_direct(args):
    """Pool 5 worker: similarity matrix for a single block."""
    block_idx, block = args
    return hap_statistics.get_block_hap_similarities(block)


def _worker_get_match_probabilities(sample_idx, recomb_rate, value_error_rate):
    """
    Pool 6 worker: match sample to haplotype combinations.
    Accesses large arrays from POSIX SharedMemory (zero-copy).
    """
    return get_match_probabilities(
        _SHARED_DATA['genotypes'],
        _SHARED_DATA['sample_probs'][sample_idx],
        _SHARED_DATA['locations'],
        keep_flags=_SHARED_DATA.get('keep_flags'),
        recomb_rate=recomb_rate,
        value_error_rate=value_error_rate
    )

#%%

def match_haplotypes_by_overlap_probabalistic(block_level_haps, num_processes=16):
    """
    Probabilistic version of match_haplotypes_by_overlap.
    Returns likelihood scores for edges.
    
    Forkserver + direct args: each task receives only the two blocks it needs
    (~20 KB each after stripping probs_array).
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

    # 2. Parallel Processing — pass stripped block pairs as task arguments
    task_args = [
        (i, _strip_block(block_level_haps[i]), _strip_block(block_level_haps[i+1]))
        for i in range(num_junctions)
    ]
    
    with _safe_forkserver_pool(num_processes) as pool:
        matches = pool.map(_worker_match_overlap_direct, task_args)
            
    return (block_haps_names, matches)
       
def match_haplotypes_by_samples_probabalistic(full_haps_data, num_processes=16):
    """
    Probabilistic version of match_haplotypes_by_samples.
    
    Forkserver + direct args: Phase 1 sends one stripped block per task;
    Phase 2 sends two stripped blocks + their match results per task.
    """
        
    num_blocks = len(full_haps_data)
    
    block_haps_names = []
    for i in range(len(full_haps_data)):
        block_haps_names.append([])
        for nm in full_haps_data[i].haplotypes.keys():
            block_haps_names[-1].append((i,nm))
    
    # --- PHASE 1: one block per task ---
    task_args_1 = [(i, _strip_block(full_haps_data[i])) for i in range(num_blocks)]

    with _safe_forkserver_pool(num_processes) as pool:
        match_best_results = pool.map(
            _worker_combined_best_hap_matches_direct,
            task_args_1
        )
        
    # --- PHASE 2: two adjacent blocks + their match results per task ---
    task_args_2 = [
        (i, i+1,
         _strip_block(full_haps_data[i]), _strip_block(full_haps_data[i+1]),
         match_best_results[i], match_best_results[i+1])
        for i in range(num_blocks - 1)
    ]
    
    with _safe_forkserver_pool(num_processes) as pool:
        neighbouring_usages = pool.map(
            _worker_hap_matching_comparison_direct,
            task_args_2
        )
    
    forward_match_scores = [neighbouring_usages[x][0] for x in range(num_blocks-1)]
    backward_match_scores = [neighbouring_usages[x][1] for x in range(num_blocks-1)]
    
    combined_scores = []
    
    for i in range(len(forward_match_scores)):
        commons = {}
        # Union of keys to handle cases where one direction is empty
        all_keys = set(forward_match_scores[i].keys()) | set(backward_match_scores[i].keys())
        
        for x in all_keys:
            f_score = forward_match_scores[i].get(x, 0.0)
            b_score = backward_match_scores[i].get(x, 0.0)
            commons[x] = (f_score + b_score) / 2.0
            
        combined_scores.append(commons)
        
    return [(block_haps_names,forward_match_scores),(block_haps_names,backward_match_scores),(block_haps_names,combined_scores)]
        
#%%

def get_combined_hap_score(hap_overlap_scores, hap_sample_scores, overlap_importance=1):
    """
    Combines overlap and sample matching scores.
    Robust to missing keys in either dictionary.
    """
    ovr = hap_overlap_scores[1]
    samps = hap_sample_scores[1]
    
    total_weight = overlap_importance + 2
    
    combined_dict = {}
    
    for i in range(len(ovr)):
        # 1. Process keys in Overlap (checking Sample)
        for d in ovr[i].keys():
            ovr_val = ovr[i][d]
            samp_val = samps[i].get(d, 0.0) # Handle missing sample score
            
            comb = (overlap_importance * ovr_val + 2 * samp_val) / total_weight      
            combined_dict[d] = comb
            
        # 2. Process keys in Sample (checking Overlap)
        # Include edges found by Samples but not Overlap
        for d in samps[i].keys():
            if d not in combined_dict:
                ovr_val = 0.0
                samp_val = samps[i][d]
                
                comb = (overlap_importance * ovr_val + 2 * samp_val) / total_weight
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
            # print(f"Warning: Only partial path found for haplotype {ite}. Stopping early.")
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
    Handles gaps between blocks correctly by skipping empty regions
    and only appending valid data from subsequent blocks.
    """
    
    final_haplotype = []
    final_locations = []
    
    num_blocks = len(all_haps)
    
    # Pre-calculate where next block starts inside current (if overlap exists)
    # If no overlap (gap), index is len(current)
    next_starting = []
    
    for i in range(num_blocks - 1):
        if len(all_haps[i].positions) == 0 or len(all_haps[i+1].positions) == 0:
            next_starting.append(0) # Dummy
            continue
            
        start_position_next = all_haps[i+1].positions[0]
        # Find where next block starts in current block's coordinates
        insertion_point = np.searchsorted(all_haps[i].positions, start_position_next)
        next_starting.append(insertion_point)

    # Process blocks
    for i in range(num_blocks):
        # Determine the range of the current block to include
        # Start: Overlap with Previous (or 0 if first)
        # End: Overlap with Next (or len if last/gap)
        
        current_hap_idx = hap_blocks[i][1]
        current_data = all_haps[i].haplotypes[current_hap_idx]
        current_pos = all_haps[i].positions
        
        # 1. Determine trim_start (Overlap from previous iteration)
        if i == 0:
            trim_start = 0
        else:
            # Look back: Where did this block start inside the previous one?
            # We want to start AFTER the merged region.
            # But the 'Merge' happened at the *end* of the previous block.
            # So for *this* block, we start at the beginning of the merge?
            # Actually, `next_overlap_data` in prev iteration covered `[:overlap_len]`.
            
            # Recalculate overlap with previous
            prev_pos_start_next = current_pos[0]
            idx_in_prev = np.searchsorted(all_haps[i-1].positions, prev_pos_start_next)
            overlap_len = len(all_haps[i-1].positions) - idx_in_prev
            
            if overlap_len > 0:
                trim_start = overlap_len
            else:
                trim_start = 0
                
        # 2. Determine trim_end (Where to start merging with next)
        if i == num_blocks - 1:
            trim_end = len(current_data)
        else:
            # Check overlap with next
            idx_start_next = next_starting[i]
            overlap_len_next = len(current_pos) - idx_start_next
            
            if overlap_len_next > 0:
                trim_end = idx_start_next
            else:
                trim_end = len(current_data)
        
        # 3. Append the "Middle" (Unique) part of the current block
        # Safety clamp
        if trim_start < len(current_data):
            # Append unique part
            # If gap (trim_end == len), we append until end.
            final_haplotype.extend(current_data[trim_start : trim_end])
            final_locations.extend(current_pos[trim_start : trim_end])
            
        # 4. Perform Merge (only if not last block AND overlap exists)
        if i < num_blocks - 1:
            idx_start_next = next_starting[i]
            overlap_len_next = len(current_pos) - idx_start_next
            
            if overlap_len_next > 0:
                # We have overlap. Merge the tail of curr with head of next.
                hap_next_idx = hap_blocks[i+1][1]
                
                curr_overlap_data = current_data[trim_end:] # Should be len = overlap_len_next
                next_overlap_data = all_haps[i+1].haplotypes[hap_next_idx][:overlap_len_next]
                
                # Read counts for weighting
                if all_haps[i].reads_count_matrix is not None and all_haps[i].reads_count_matrix.size > 0:
                    reads_sum = np.sum(all_haps[i].reads_count_matrix[:, trim_end:, :], axis=0)
                    num_samples = all_haps[i].reads_count_matrix.shape[0]
                    hap_priors = []
                    for k in range(len(reads_sum)):
                        if sum(reads_sum[k]) >= max(min_total_reads, read_error_prob*num_samples):
                            rat_val = (1+reads_sum[k][1])/(2+reads_sum[k][0]+reads_sum[k][1])
                        else:
                            rat_val = read_error_prob
                        hap_priors.append([rat_val, 1-rat_val])
                    hap_priors = np.array(hap_priors)
                else:
                    hap_priors = np.full((len(curr_overlap_data), 2), 0.5)
                
                # Merge
                merged_probs = []
                process_len = min(len(curr_overlap_data), len(next_overlap_data), len(hap_priors))
                
                for k in range(process_len):
                    new_val = analysis_utils.combine_probabilities(
                        curr_overlap_data[k], next_overlap_data[k], hap_priors[k]
                    )
                    merged_probs.append(new_val)
                
                final_haplotype.extend(merged_probs)
                final_locations.extend(current_pos[trim_end : trim_end+process_len])
    
    return [np.array(final_locations), np.array(final_haplotype)]

def combine_all_blocks_to_long_haps(all_haps,
                                    hap_blocks_list,
                                    read_error_prob = 0.02,
                                    min_total_reads=5,
                                    num_processes=16):
    """
    Multithreaded stitching of all long haplotypes.
    
    Forkserver + POSIX SharedMemory: light blocks (positions, haplotypes,
    keep_flags) are sent via pool initializer (~5 KB per block).
    reads_count_matrices are concatenated into a single SharedMemory buffer
    (~500 KB per block × N blocks) for zero-copy worker access.
    Only ~6 tasks (one per long hap), each receives just chain_data + scalars.
    """
    if not hap_blocks_list:
        return [[], []]

    # 1. Strip blocks to light versions (no reads, no probs)
    light_blocks = [_strip_block_light(b) for b in all_haps]
    
    # 2. Concatenate all reads_count_matrices into a single SharedMemory buffer
    reads_index = []  # block_idx -> (offset, n_samples, n_sites) or None
    reads_chunks = []
    offset = 0
    reads_dtype = None
    
    for b in all_haps:
        if b.reads_count_matrix is not None and b.reads_count_matrix.size > 0:
            rcm = np.ascontiguousarray(b.reads_count_matrix)
            if reads_dtype is None:
                reads_dtype = rcm.dtype
            reads_index.append((offset, rcm.shape[0], rcm.shape[1]))
            reads_chunks.append(rcm.ravel())
            offset += rcm.size
        else:
            reads_index.append(None)
    
    shm_handles = []
    try:
        if reads_chunks:
            all_reads_flat = np.concatenate(reads_chunks)
            shm_reads, meta_reads = _create_shm_array(all_reads_flat, "pool4_reads")
            shm_handles.append(shm_reads)
        else:
            meta_reads = None
        
        # 3. Build initializer context — light blocks + reads SharedMemory
        shared_context = {
            'blocks': light_blocks,          # small — pickle via initializer
            'reads_index': reads_index,       # small — list of tuples
        }
        if meta_reads is not None:
            shared_context['reads_flat'] = meta_reads  # SharedMemory-backed
        
        # 4. Task args are just chain_data + scalars
        task_args = [
            (chain_data, read_error_prob, min_total_reads)
            for chain_data in hap_blocks_list
        ]

        with _safe_forkserver_pool(min(num_processes, len(task_args)),
                                   initializer=_init_shared_data,
                                   initargs=(shared_context,)) as pool:
            processing_results = pool.map(_worker_combine_chained_blocks_direct, task_args)
    
    finally:
        for shm in shm_handles:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
    
    # Filter empty results if any path failed completely
    valid_results = [r for r in processing_results if len(r[0]) > 0]
    
    if not valid_results:
        return [[], []]
    
    # Use result with max length to set sites
    longest_idx = np.argmax([len(r[0]) for r in valid_results])
    sites_loc = valid_results[longest_idx][0]
    
    long_haps = [r[1] for r in valid_results]
    
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
    
    # --- CRITICAL FIX: Filter out empty blocks to prevent graph disconnection ---
    valid_blocks = []
    for b in block_results:
        # Check Positions, Haplotypes, AND Active Flags
        if len(b.positions) == 0: continue
        if len(b.haplotypes) == 0: continue
        if b.keep_flags is not None and np.sum(b.keep_flags) == 0: continue
        valid_blocks.append(b)
        
    if len(valid_blocks) < 2:
        return (block_results, [[], []])

    # 1. Match haplotypes between neighboring blocks (Overlap & Samples)
    # Passed num_processes to enable parallel, shared-memory overlap matching
    hap_matching_overlap = match_haplotypes_by_overlap_probabalistic(valid_blocks, num_processes=num_processes)
    
    # Run Shared Memory Optimized Sample Matching
    hap_matching_samples = match_haplotypes_by_samples_probabalistic(valid_blocks, num_processes=num_processes)
    
    node_names = hap_matching_overlap[0]
    
    # 2. Combine scores
    combined_scores = get_combined_hap_score(hap_matching_overlap, hap_matching_samples[2])

    # 3. Pre-calculate similarity matrices in parallel (forkserver + direct args)
    task_args_sim = [(i, _strip_block(valid_blocks[i])) for i in range(len(valid_blocks))]
    with _safe_forkserver_pool(num_processes) as p:
        similarity_matrices = p.map(
            _worker_get_similarities_direct,
            task_args_sim
        )
    
    # 4. Find optimal paths (Chaining) - Sequential logic, fast
    chained_block_haps = generate_chained_block_haplotypes(
        valid_blocks,
        node_names,
        combined_scores,
        num_long_haps,
        node_usage_penalty=node_usage_penalty,
        edge_usage_penalty=edge_usage_penalty,
        similarity_matrices=similarity_matrices
    )
    
    # 5. Stitch blocks together (Shared Memory Optimized)
    final_long_haps = combine_all_blocks_to_long_haps(valid_blocks, chained_block_haps, num_processes=num_processes)

    return (valid_blocks, final_long_haps)   

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
    
    Forkserver + POSIX SharedMemory: the genotypes tensor (~150 MB) and
    sample_probs matrix (~1.3 GB) are placed in /dev/shm for zero-copy
    access by workers.  Small arrays (locations, keep_flags) are passed
    directly via the pool initializer.
    """
    
    # Place large arrays in POSIX SharedMemory
    shm_handles = []
    try:
        shm_geno, meta_geno = _create_shm_array(full_combined_genotypes, "genotypes")
        shm_handles.append(shm_geno)
        
        shm_probs, meta_probs = _create_shm_array(all_sample_probs, "sample_probs")
        shm_handles.append(shm_probs)
        
        # Build initializer dict — large arrays as shm metadata, small arrays directly
        shared_context = {
            'genotypes': meta_geno,
            'sample_probs': meta_probs,
            'locations': site_locations,           # small — pass directly
        }
        if keep_flags is not None:
            shared_context['keep_flags'] = keep_flags  # small — pass directly
        
        worker = partial(_worker_get_match_probabilities, 
                         recomb_rate=recomb_rate, 
                         value_error_rate=value_error_rate)

        with _safe_forkserver_pool(num_processes, initializer=_init_shared_data,
                                   initargs=(shared_context,)) as pool:
            num_samples = len(all_sample_probs)
            results = pool.map(worker, range(num_samples))
    
    finally:
        # Clean up SharedMemory segments
        for shm in shm_handles:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
    
    return results