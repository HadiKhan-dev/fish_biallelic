import numpy as np
import math
import pandas as pd
import warnings
from typing import List, Tuple, Dict, NamedTuple

import analysis_utils

# --- VISUALIZATION IMPORTS ---
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

# Suppress warnings
warnings.filterwarnings("ignore")
np.seterr(divide='ignore', invalid='ignore')

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("WARNING: Numba not found. Painting will be slow.")
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator
    prange = range

# =============================================================================
# 1. RESULT CLASSES
# =============================================================================

class PaintedChunk(NamedTuple):
    """Represents a specific interval where the state (Hap1, Hap2) is constant."""
    start: int
    end: int
    hap1: int  # Assigned to Track 1 (Bottom)
    hap2: int  # Assigned to Track 2 (Top)

class PhasedSegment(NamedTuple):
    """Represents a contiguous segment of a single phased haplotype."""
    start: int
    end: int
    founder_id: int
    
class SamplePainting:
    """
    Holds the painting results for a single sample.
    """
    def __init__(self, sample_index: int, chunks: List[PaintedChunk]):
        self.sample_index = sample_index
        self.chunks = chunks
        self.num_recombinations = max(0, len(chunks) - 1)
        
        # Extract fully phased haplotypes by merging contiguous chunks per track
        self.hap1_phased = self._extract_phased_track(chunks, track_idx=0)
        self.hap2_phased = self._extract_phased_track(chunks, track_idx=1)

    def _extract_phased_track(self, chunks, track_idx) -> List[PhasedSegment]:
        segments = []
        if not chunks: 
            return segments
        
        curr_start = chunks[0].start
        curr_end = chunks[0].end
        # track_idx 0 is hap1, 1 is hap2
        curr_id = chunks[0].hap1 if track_idx == 0 else chunks[0].hap2
        
        for i in range(1, len(chunks)):
            c = chunks[i]
            next_id = c.hap1 if track_idx == 0 else c.hap2
            
            if next_id == curr_id:
                # Same founder, extend the segment
                curr_end = c.end
            else:
                # Founder changed, seal current segment
                segments.append(PhasedSegment(curr_start, curr_end, curr_id))
                # Start new segment
                curr_start = c.start
                curr_end = c.end
                curr_id = next_id
                
        # Append final segment
        segments.append(PhasedSegment(curr_start, curr_end, curr_id))
        return segments

    def __repr__(self):
        return f"<SamplePainting ID {self.sample_index}: {len(self.hap1_phased)} segs in Hap1, {len(self.hap2_phased)} segs in Hap2>"

    def __iter__(self):
        return iter(self.chunks)
    
    def __getitem__(self, idx):
        return self.chunks[idx]

    def to_dict_list(self, sample_name=None):
        """Helper for dataframe conversion (Exploded view)."""
        name = sample_name if sample_name else f"Sample_{self.sample_index}"
        rows = []
        for c in self.chunks:
            rows.append({
                'Sample': name,
                'Start': c.start,
                'End': c.end,
                'Hap1': c.hap1,
                'Hap2': c.hap2,
                'Length': c.end - c.start + 1
            })
        return rows

class BlockPainting:
    def __init__(self, block_position_range: Tuple[int, int], samples: List[SamplePainting]):
        self.start_pos = block_position_range[0]
        self.end_pos = block_position_range[1]
        self.samples = samples
        self.num_samples = len(samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)

    def __repr__(self):
        return f"<BlockPainting: {self.num_samples} samples, Range {self.start_pos}-{self.end_pos}>"

    def to_dataframe(self, sample_names=None) -> pd.DataFrame:
        all_rows = []
        for i, sample in enumerate(self.samples):
            name = sample_names[i] if sample_names else None
            all_rows.extend(sample.to_dict_list(name))
        return pd.DataFrame(all_rows)

# =============================================================================
# 2. HELPER: DENSE MATRIX CONVERSION (For Cleanup)
# =============================================================================

def founder_block_to_dense(block_result):
    """
    Converts a BlockResult object (dictionary of haps) into a dense matrix
    for fast lookup during cleanup.
    Returns: (hap_matrix, positions)
    """
    positions = block_result.positions
    hap_dict = block_result.haplotypes
    
    if not hap_dict:
        return np.zeros((0, 0), dtype=np.int8), positions

    # Determine max ID
    max_id = max(hap_dict.keys())
    n_sites = len(positions)
    
    # Init with -1 (Missing)
    # Shape: (MaxID+1, Sites)
    dense_haps = np.full((max_id + 1, n_sites), -1, dtype=np.int8)
    
    for fid, hap_arr in hap_dict.items():
        # Handle probabilistic (N,2) vs concrete (N,)
        if hap_arr.ndim == 2:
            concrete = np.argmax(hap_arr, axis=1)
        else:
            concrete = hap_arr
        dense_haps[fid, :] = concrete
        
    return dense_haps, positions

# =============================================================================
# 3. CLEANUP KERNELS (INDEPENDENT TRACK RESOLUTION)
# =============================================================================

@njit(fastmath=True)
def find_equivalent_founders_for_chunk(
    hap_matrix,      # (MaxID, Sites)
    snp_positions,   # (Sites,)
    chunk_start,
    chunk_end,
    current_id,
    tolerance=0.01   # 1% difference allowed to be considered "Equal"
):
    """
    Checks all founders to see which ones match the 'current_id' haplotype
    in the specific region [chunk_start, chunk_end).
    
    Returns a boolean mask of valid candidates.
    """
    n_founders, n_sites = hap_matrix.shape
    candidates = np.zeros(n_founders, dtype=np.bool_)
    
    # Find SNP range
    start_idx = -1
    end_idx = -1
    
    for i in range(n_sites):
        if snp_positions[i] >= chunk_start:
            start_idx = i
            break
            
    if start_idx == -1: return candidates # No SNPs
    
    for i in range(start_idx, n_sites):
        if snp_positions[i] >= chunk_end:
            break
        end_idx = i
    
    # If region is empty or 1 SNP, difficult to judge, but we proceed
    if end_idx < start_idx:
        candidates[current_id] = True
        return candidates
        
    # Extract reference sequence (what we assigned)
    ref_seq = hap_matrix[current_id, start_idx:end_idx+1]
    
    valid_sites_count = 0
    for x in ref_seq:
        if x != -1: valid_sites_count += 1
        
    if valid_sites_count == 0:
        candidates[current_id] = True
        return candidates

    # Compare against all other founders
    for f_id in range(n_founders):
        if f_id == current_id:
            candidates[f_id] = True
            continue
            
        test_seq = hap_matrix[f_id, start_idx:end_idx+1]
        
        matches = 0
        total_comp = 0
        
        for k in range(len(ref_seq)):
            r = ref_seq[k]
            t = test_seq[k]
            if r != -1 and t != -1:
                total_comp += 1
                if r == t:
                    matches += 1
        
        if total_comp > 0:
            diff = 1.0 - (matches / total_comp)
            if diff <= tolerance:
                candidates[f_id] = True
        
    return candidates

def resolve_single_track(segments: List[PhasedSegment], hap_matrix, snp_positions) -> List[PhasedSegment]:
    """
    Resolves ambiguous IDs on a single haploid track by checking neighbors.
    Merges adjacent segments if they resolve to the same ID.
    """
    if not segments: return []
    
    # 1. Calculate Options for every segment
    # Store: (original_segment, valid_ids_set, resolved_id)
    seg_meta = []
    
    for seg in segments:
        cands = find_equivalent_founders_for_chunk(
            hap_matrix, snp_positions, seg.start, seg.end, seg.founder_id
        )
        valid_set = set(np.where(cands)[0])
        
        seg_meta.append({
            'seg': seg,
            'opts': valid_set,
            'final_id': seg.founder_id
        })
        
    n_segs = len(seg_meta)
    
    # 2. Resolve Ambiguities
    for i in range(n_segs):
        opts = seg_meta[i]['opts']
        curr_id = seg_meta[i]['final_id']
        
        if len(opts) > 1:
            # Ambiguous! Search neighbors.
            best_choice = curr_id
            min_dist = float('inf')
            
            # Search Left
            for j in range(i-1, -1, -1):
                neighbor_id = seg_meta[j]['final_id'] # Use resolved neighbor
                if neighbor_id in opts:
                    dist = i - j
                    if dist < min_dist:
                        min_dist = dist
                        best_choice = neighbor_id
                    break 
            
            # Search Right
            for j in range(i+1, n_segs):
                neighbor_id = seg_meta[j]['seg'].founder_id # Use original neighbor (future)
                if neighbor_id in opts:
                    dist = j - i
                    if dist < min_dist:
                        min_dist = dist
                        best_choice = neighbor_id
                    break
            
            seg_meta[i]['final_id'] = best_choice
            
    # 3. Merge Adjacent Segments
    new_segments = []
    
    curr = seg_meta[0]
    curr_start = curr['seg'].start
    curr_end = curr['seg'].end
    curr_id = curr['final_id']
    
    for i in range(1, n_segs):
        next_s = seg_meta[i]
        next_id = next_s['final_id']
        
        # Continuity check + ID check
        if (curr_end == next_s['seg'].start) and (curr_id == next_id):
            curr_end = next_s['seg'].end
        else:
            new_segments.append(PhasedSegment(curr_start, curr_end, curr_id))
            curr_start = next_s['seg'].start
            curr_end = next_s['seg'].end
            curr_id = next_id
            
    new_segments.append(PhasedSegment(curr_start, curr_end, curr_id))
    return new_segments

def zip_tracks_to_chunks(track1: List[PhasedSegment], track2: List[PhasedSegment]) -> List[PaintedChunk]:
    """
    Combines two independent haploid tracks back into diploid chunks.
    Takes the Union of breakpoints from both tracks.
    """
    breakpoints = set()
    for s in track1:
        breakpoints.add(s.start)
        breakpoints.add(s.end)
    for s in track2:
        breakpoints.add(s.start)
        breakpoints.add(s.end)
        
    sorted_bp = sorted(list(breakpoints))
    chunks = []
    
    # Cursors
    t1_idx = 0
    t2_idx = 0
    n1 = len(track1)
    n2 = len(track2)
    
    for k in range(len(sorted_bp) - 1):
        start = sorted_bp[k]
        end = sorted_bp[k+1]
        
        # Find active segment for Track 1
        # Since sorted_bp are derived from track1, we are guaranteed exact alignment or containment
        while t1_idx < n1 and track1[t1_idx].end <= start:
            t1_idx += 1
        
        h1 = track1[t1_idx].founder_id if t1_idx < n1 else -1
        
        # Find active segment for Track 2
        while t2_idx < n2 and track2[t2_idx].end <= start:
            t2_idx += 1
            
        h2 = track2[t2_idx].founder_id if t2_idx < n2 else -1
        
        chunks.append(PaintedChunk(start, end, h1, h2))
        
    return chunks

# =============================================================================
# 4. VECTORIZED VITERBI KERNEL (STRICT PAINTING)
# =============================================================================

@njit(parallel=True, fastmath=True)
def viterbi_painting_solver(ll_tensor, positions, recomb_rate, state_definitions, n_haps, switch_penalty):
    """
    Solves the Viterbi path for N samples simultaneously.
    """
    n_samples, K, n_sites = ll_tensor.shape
    final_paths = np.zeros((n_samples, n_sites), dtype=np.int32)
    
    if n_haps > 1:
        log_N_minus_1 = math.log(float(n_haps - 1))
    else:
        log_N_minus_1 = 0.0

    for s in prange(n_samples):
        backpointers = np.zeros((n_sites, K), dtype=np.int32)
        current_scores = np.empty(K, dtype=np.float64)
        prev_scores = np.empty(K, dtype=np.float64)

        for k in range(K):
            current_scores[k] = ll_tensor[s, k, 0]

        for i in range(1, n_sites):
            prev_scores[:] = current_scores[:]
            dist_bp = positions[i] - positions[i-1]
            if dist_bp < 1: dist_bp = 1
            theta = float(dist_bp) * recomb_rate
            
            if theta > 0.5: theta = 0.5
            if theta < 1e-15: theta = 1e-15
            
            log_switch = math.log(theta) - switch_penalty
            log_stay = math.log(1.0 - theta)
            
            cost_0 = 2.0 * log_stay
            cost_1 = log_switch + log_stay - log_N_minus_1
            cost_2 = 2.0 * log_switch - 2.0 * log_N_minus_1
            
            for k_curr in range(K):
                h1_curr = state_definitions[k_curr, 0]
                h2_curr = state_definitions[k_curr, 1]
                
                best_prev_k = -1
                best_score = -np.inf
                
                for k_prev in range(K):
                    h1_prev = state_definitions[k_prev, 0]
                    h2_prev = state_definitions[k_prev, 1]
                    
                    dist = 0
                    if h1_curr != h1_prev: dist += 1
                    if h2_curr != h2_prev: dist += 1
                    
                    if dist == 0: trans_cost = cost_0
                    elif dist == 1: trans_cost = cost_1
                    else: trans_cost = cost_2
                    
                    score = prev_scores[k_prev] + trans_cost
                    if score > best_score:
                        best_score = score
                        best_prev_k = k_prev
                
                backpointers[i, k_curr] = best_prev_k
                current_scores[k_curr] = best_score + ll_tensor[s, k_curr, i]

        best_end_k = -1
        best_end_score = -np.inf
        for k in range(K):
            if current_scores[k] > best_end_score:
                best_end_score = current_scores[k]
                best_end_k = k
        
        final_paths[s, n_sites - 1] = best_end_k
        
        for i in range(n_sites - 1, 0, -1):
            curr_k = final_paths[s, i]
            prev_k = backpointers[i, curr_k]
            final_paths[s, i-1] = prev_k
            
    return final_paths

def calculate_batch_emissions(sample_probs_matrix, hap_dict, robustness_epsilon=1e-3):
    hap_keys = sorted(list(hap_dict.keys()))
    num_haps = len(hap_keys)
    num_samples, num_sites, _ = sample_probs_matrix.shape
    
    idx_i, idx_j = np.unravel_index(np.arange(num_haps**2), (num_haps, num_haps))
    state_defs = np.stack([idx_i, idx_j], axis=1).astype(np.int32)
    
    hap_list = [hap_dict[k] for k in hap_keys]
    if not hap_list:
        return np.zeros((num_samples, 0, num_sites)), state_defs, hap_keys
        
    haps_tensor = np.array(hap_list)
    h0 = haps_tensor[:, :, 0]
    h1 = haps_tensor[:, :, 1]
    
    c00 = h0[:, None, :] * h0[None, :, :]
    c11 = h1[:, None, :] * h1[None, :, :]
    c01 = (h0[:, None, :] * h1[None, :, :]) + (h1[:, None, :] * h0[None, :, :])
    
    combos_0 = c00.reshape(num_haps**2, -1)
    combos_1 = c01.reshape(num_haps**2, -1)
    combos_2 = c11.reshape(num_haps**2, -1)
    
    s0 = sample_probs_matrix[:, :, 0][:, np.newaxis, :]
    s1 = sample_probs_matrix[:, :, 1][:, np.newaxis, :]
    s2 = sample_probs_matrix[:, :, 2][:, np.newaxis, :]
    
    c0 = combos_0[np.newaxis, :, :]
    c1 = combos_1[np.newaxis, :, :]
    c2 = combos_2[np.newaxis, :, :]
    
    model_probs = (s0 * c0) + (s1 * c1) + (s2 * c2)
    uniform_prob = 1.0 / 3.0
    final_probs = (model_probs * (1.0 - robustness_epsilon)) + (robustness_epsilon * uniform_prob)
    
    min_prob = 1e-300
    final_probs[final_probs < min_prob] = min_prob
    ll_matrix = np.log(final_probs)
    ll_matrix = np.maximum(ll_matrix, -50.0)
    
    return ll_matrix, state_defs, hap_keys

def compress_path_to_chunks(path_indices, positions, state_defs, hap_keys):
    chunks = []
    if len(path_indices) == 0: return chunks

    h1_idx, h2_idx = state_defs[path_indices[0]]
    curr_t1 = hap_keys[h1_idx]
    curr_t2 = hap_keys[h2_idx]
    
    if curr_t1 > curr_t2: curr_t1, curr_t2 = curr_t2, curr_t1
        
    start_pos = positions[0]
    
    for i in range(1, len(path_indices)):
        h1_idx_next, h2_idx_next = state_defs[path_indices[i]]
        cand_a = hap_keys[h1_idx_next]
        cand_b = hap_keys[h2_idx_next]
        
        score_1 = (1 if cand_a == curr_t1 else 0) + (1 if cand_b == curr_t2 else 0)
        score_2 = (1 if cand_b == curr_t1 else 0) + (1 if cand_a == curr_t2 else 0)
        
        if score_1 >= score_2:
            next_t1, next_t2 = cand_a, cand_b
        else:
            next_t1, next_t2 = cand_b, cand_a
            
        if (next_t1 != curr_t1) or (next_t2 != curr_t2):
            midpoint = (positions[i-1] + positions[i]) // 2
            chunks.append(PaintedChunk(int(start_pos), int(midpoint), curr_t1, curr_t2))
            curr_t1, curr_t2 = next_t1, next_t2
            start_pos = midpoint
            
    end_pos = positions[-1]
    chunks.append(PaintedChunk(int(start_pos), int(end_pos), curr_t1, curr_t2))
    return chunks

# =============================================================================
# 5. MAIN DRIVER
# =============================================================================

def paint_samples_in_block(block_result, sample_probs_matrix, sample_sites, 
                           recomb_rate=1e-8,
                           switch_penalty=10.0,
                           robustness_epsilon=1e-3, 
                           batch_size=10):
    """
    Paints all samples using Viterbi, then applies INDEPENDENT TRACK CLEANUP.
    """
    
    positions = block_result.positions
    hap_dict = block_result.haplotypes
    
    # 1. Prepare for Cleanup (Dense Matrix)
    hap_matrix, snp_positions = founder_block_to_dense(block_result)
    
    block_samples_data = analysis_utils.get_sample_data_at_sites_multiple(
        sample_probs_matrix, sample_sites, positions
    )
    
    num_samples = block_samples_data.shape[0]
    all_sample_paintings = []
    
    print(f"Painting {num_samples} samples in batches of {batch_size} (eps={robustness_epsilon:.1e})...")
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        
        batch_data = block_samples_data[start_idx:end_idx]
        
        ll_tensor, state_defs, hap_keys = calculate_batch_emissions(
            batch_data, hap_dict, robustness_epsilon=robustness_epsilon
        )
        
        num_haps = len(hap_keys)
        raw_paths = viterbi_painting_solver(
            ll_tensor, positions, recomb_rate, state_defs, num_haps, float(switch_penalty)
        )
        
        batch_count = raw_paths.shape[0]
        for i in range(batch_count):
            global_sample_idx = start_idx + i
            
            # A. Get Raw Diplotype Chunks
            chunks = compress_path_to_chunks(raw_paths[i], positions, state_defs, hap_keys)
            
            # B. Split into Independent Tracks
            # We reuse the logic from SamplePainting init
            temp_obj = SamplePainting(global_sample_idx, chunks)
            track1_segs = temp_obj.hap1_phased
            track2_segs = temp_obj.hap2_phased
            
            # C. Clean Independently (Resolves IBS flickering using long-range neighbors)
            clean_t1 = resolve_single_track(track1_segs, hap_matrix, snp_positions)
            clean_t2 = resolve_single_track(track2_segs, hap_matrix, snp_positions)
            
            # D. Stitch back together
            final_chunks = zip_tracks_to_chunks(clean_t1, clean_t2)
            
            all_sample_paintings.append(SamplePainting(global_sample_idx, final_chunks))
            
    range_tuple = (int(positions[0]), int(positions[-1]))
    return BlockPainting(range_tuple, all_sample_paintings)

# =============================================================================
# 6. VISUALIZATION
# =============================================================================

def plot_painting(block_painting, output_file=None, 
                  title="Chromosome Painting", 
                  figsize_width=20, 
                  row_height_per_sample=0.25,
                  show_labels=True,
                  sample_names=None):
    
    if not HAS_PLOTTING:
        print("Error: Matplotlib/Seaborn not installed.")
        return

    unique_haps = set()
    for sample in block_painting:
        for chunk in sample:
            unique_haps.add(chunk.hap1)
            unique_haps.add(chunk.hap2)
    
    sorted_haps = sorted(list(unique_haps))
    hap_to_idx = {h: i for i, h in enumerate(sorted_haps)}
    
    if len(sorted_haps) <= 10:
        palette = sns.color_palette("tab10", len(sorted_haps))
    elif len(sorted_haps) <= 20:
        palette = sns.color_palette("tab20", len(sorted_haps))
    else:
        palette = sns.color_palette("husl", len(sorted_haps))
        
    num_samples = len(block_painting)
    header_space = 2.0 
    calc_height = (num_samples * row_height_per_sample) + header_space
    
    if calc_height < 6: calc_height = 6
    if calc_height > 300: calc_height = 300
    
    figsize = (figsize_width, calc_height)
    fig, ax = plt.subplots(figsize=figsize)
    
    y_height = 0.8 
    
    for i, sample in enumerate(block_painting):
        y_base = i 
        for chunk in sample:
            width = chunk.end - chunk.start
            if width <= 0: continue
            
            color1 = palette[hap_to_idx[chunk.hap1]]
            rect1 = mpatches.Rectangle(
                (chunk.start, y_base), width, y_height/2,
                facecolor=color1, edgecolor='none'
            )
            ax.add_patch(rect1)
            
            color2 = palette[hap_to_idx[chunk.hap2]]
            rect2 = mpatches.Rectangle(
                (chunk.start, y_base + y_height/2), width, y_height/2,
                facecolor=color2, edgecolor='none'
            )
            ax.add_patch(rect2)
            
    ax.set_xlim(block_painting.start_pos, block_painting.end_pos)
    ax.set_ylim(-0.5, len(block_painting) + 0.5)
    
    ax.set_xlabel("Genomic Position (bp)")
    ax.set_ylabel("Samples")
    ax.set_title(title)
    
    if show_labels:
        if sample_names and len(sample_names) == len(block_painting):
            labels = sample_names
        else:
            labels = [f"S{s.sample_index}" for s in block_painting]
        ax.set_yticks(np.arange(len(block_painting)) + y_height/2)
        ax.set_yticklabels(labels, fontsize=8)
    else:
        ax.set_yticks([])

    legend_patches = []
    for h_key in sorted_haps:
        c = palette[hap_to_idx[h_key]]
        legend_patches.append(mpatches.Patch(color=c, label=f"Founder {h_key}"))
        
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.01, 1), loc='upper left')
    
    plt.tight_layout()
    
    if output_file:
        dpi = 100 if calc_height > 100 else 150
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()
    plt.close()