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

# Constants
DEFAULT_ROBUSTNESS_EPSILON = 1e-2

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
    Includes both the raw chunks and the fully consolidated phased haplotypes.
    """
    def __init__(self, sample_index: int, chunks: List[PaintedChunk]):
        self.sample_index = sample_index
        self.chunks = chunks
        self.num_recombinations = max(0, len(chunks) - 1)
        
        # Extract fully phased haplotypes by merging contiguous chunks per track
        self.hap1_phased = self._extract_phased_track(chunks, track_idx=0)
        self.hap2_phased = self._extract_phased_track(chunks, track_idx=1)

    def _extract_phased_track(self, chunks, track_idx) -> List[PhasedSegment]:
        """Merges adjacent chunks if the founder ID on this track doesn't change."""
        segments = []
        if not chunks: 
            return segments
        
        # Get initial state
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
    """
    The main container returned by paint_samples_in_block.
    Holds the painting results for ALL samples in a specific genomic block.
    """
    def __init__(self, block_position_range: Tuple[int, int], samples: List[SamplePainting]):
        self.start_pos = block_position_range[0]
        self.end_pos = block_position_range[1]
        self.samples = samples
        self.num_samples = len(samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Access a specific SamplePainting by index."""
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)

    def __repr__(self):
        return f"<BlockPainting: {self.num_samples} samples, Range {self.start_pos}-{self.end_pos}>"

    def to_dataframe(self, sample_names=None) -> pd.DataFrame:
        """
        Converts the entire block painting to a Pandas DataFrame.
        """
        all_rows = []
        for i, sample in enumerate(self.samples):
            name = sample_names[i] if sample_names else None
            all_rows.extend(sample.to_dict_list(name))
        return pd.DataFrame(all_rows)

# =============================================================================
# 2. VECTORIZED VITERBI KERNEL (STRICT PAINTING)
# =============================================================================

@njit(parallel=True, fastmath=True)
def viterbi_painting_solver(ll_tensor, positions, recomb_rate, state_definitions, n_haps, switch_penalty):
    """
    Solves the Viterbi path for N samples simultaneously.
    
    Optimized for 'Strict Painting':
    - No Burst/Gap states (forces assignment to a founder pair).
    - Includes 'switch_penalty' to suppress noise-induced flickering.
    """
    n_samples, K, n_sites = ll_tensor.shape
    final_paths = np.zeros((n_samples, n_sites), dtype=np.int32)
    
    # Pre-calculate transition cost constants
    if n_haps > 1:
        log_N_minus_1 = math.log(float(n_haps - 1))
    else:
        log_N_minus_1 = 0.0

    # Parallel Loop over Samples
    for s in prange(n_samples):
        # Local DP Tables
        backpointers = np.zeros((n_sites, K), dtype=np.int32)
        current_scores = np.empty(K, dtype=np.float64)
        prev_scores = np.empty(K, dtype=np.float64)

        # Initialization
        for k in range(K):
            current_scores[k] = ll_tensor[s, k, 0]

        # Forward Pass
        for i in range(1, n_sites):
            # Swap buffers
            prev_scores[:] = current_scores[:]
            
            # Distance based transition prob
            dist_bp = positions[i] - positions[i-1]
            if dist_bp < 1: dist_bp = 1
            theta = float(dist_bp) * recomb_rate
            
            # Cap theta to reasonable bounds
            if theta > 0.5: theta = 0.5
            if theta < 1e-15: theta = 1e-15
            
            log_switch = math.log(theta) - switch_penalty # Apply stickiness
            log_stay = math.log(1.0 - theta)
            
            # Transition Costs
            cost_0 = 2.0 * log_stay
            cost_1 = log_switch + log_stay - log_N_minus_1
            cost_2 = 2.0 * log_switch - 2.0 * log_N_minus_1
            
            # Finding best predecessor
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
                    
                    if dist == 0:
                        trans_cost = cost_0
                    elif dist == 1:
                        trans_cost = cost_1
                    else:
                        trans_cost = cost_2
                    
                    score = prev_scores[k_prev] + trans_cost
                    
                    if score > best_score:
                        best_score = score
                        best_prev_k = k_prev
                
                backpointers[i, k_curr] = best_prev_k
                current_scores[k_curr] = best_score + ll_tensor[s, k_curr, i]

        # Backward Pass (Traceback)
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

# =============================================================================
# 3. VECTORIZED EMISSION CALCULATOR
# =============================================================================

def calculate_batch_emissions(sample_probs_matrix, hap_dict, robustness_epsilon=1e-3):
    """
    Calculates the log-likelihood tensor for ALL samples against ALL haplotype pairs.
    Includes robustness mixing to prevent -inf.
    """
    hap_keys = sorted(list(hap_dict.keys()))
    num_haps = len(hap_keys)
    num_samples, num_sites, _ = sample_probs_matrix.shape
    
    # Generate all pairs indices (N^2)
    idx_i, idx_j = np.unravel_index(np.arange(num_haps**2), (num_haps, num_haps))
    state_defs = np.stack([idx_i, idx_j], axis=1).astype(np.int32)
    
    hap_list = [hap_dict[k] for k in hap_keys]
    if not hap_list:
        return np.zeros((num_samples, 0, num_sites)), state_defs, hap_keys
        
    haps_tensor = np.array(hap_list) # (Num_Haps, Sites, 2)
    
    # Extract prob of allele 0 and 1 from haps (Probabilistic Haplotypes)
    h0 = haps_tensor[:, :, 0]
    h1 = haps_tensor[:, :, 1]
    
    # Precompute Genotype Probabilities for every Pair
    c00 = h0[:, None, :] * h0[None, :, :]
    c11 = h1[:, None, :] * h1[None, :, :]
    c01 = (h0[:, None, :] * h1[None, :, :]) + (h1[:, None, :] * h0[None, :, :])
    
    # Flatten to list of pairs
    combos_0 = c00.reshape(num_haps**2, -1)
    combos_1 = c01.reshape(num_haps**2, -1)
    combos_2 = c11.reshape(num_haps**2, -1)
    
    # Sample Probs: (Samples, Sites, 1) to broadcast against Pairs
    s0 = sample_probs_matrix[:, :, 0][:, np.newaxis, :]
    s1 = sample_probs_matrix[:, :, 1][:, np.newaxis, :]
    s2 = sample_probs_matrix[:, :, 2][:, np.newaxis, :]
    
    # Pair Probs: (1, Pairs, Sites)
    c0 = combos_0[np.newaxis, :, :]
    c1 = combos_1[np.newaxis, :, :]
    c2 = combos_2[np.newaxis, :, :]
    
    # Dot Product logic via broadcasting
    model_probs = (s0 * c0) + (s1 * c1) + (s2 * c2)
    
    # Robustness: Mix with Uniform
    uniform_prob = 1.0 / 3.0
    final_probs = (model_probs * (1.0 - robustness_epsilon)) + (robustness_epsilon * uniform_prob)
    
    # Log conversion with safety
    min_prob = 1e-300
    final_probs[final_probs < min_prob] = min_prob
    ll_matrix = np.log(final_probs)
    ll_matrix = np.maximum(ll_matrix, -50.0)
    
    return ll_matrix, state_defs, hap_keys

# =============================================================================
# 4. DATA CONVERSION & DRIVER
# =============================================================================

def compress_path_to_chunks(path_indices, positions, state_defs, hap_keys):
    """
    Converts site-by-site path to RLE Chunk objects.
    Uses PHASE SMOOTHING to reduce flickering.
    Uses MIDPOINT EXTENSION to fill gap artifacts (white slivers).
    """
    chunks = []
    if len(path_indices) == 0: return chunks

    # Initialize first site
    h1_idx, h2_idx = state_defs[path_indices[0]]
    curr_t1 = hap_keys[h1_idx]
    curr_t2 = hap_keys[h2_idx]
    
    # Optional: deterministic sort for the very first chunk only
    if curr_t1 > curr_t2:
        curr_t1, curr_t2 = curr_t2, curr_t1
        
    start_pos = positions[0]
    
    for i in range(1, len(path_indices)):
        # Decode next state
        h1_idx_next, h2_idx_next = state_defs[path_indices[i]]
        cand_a = hap_keys[h1_idx_next]
        cand_b = hap_keys[h2_idx_next]
        
        # --- PHASE SMOOTHING LOGIC ---
        # Match new candidates to current tracks to minimize changes
        score_1 = (1 if cand_a == curr_t1 else 0) + (1 if cand_b == curr_t2 else 0)
        score_2 = (1 if cand_b == curr_t1 else 0) + (1 if cand_a == curr_t2 else 0)
        
        if score_1 >= score_2:
            next_t1, next_t2 = cand_a, cand_b
        else:
            next_t1, next_t2 = cand_b, cand_a
            
        # Check if the state actually changed content
        if (next_t1 != curr_t1) or (next_t2 != curr_t2):
            
            # --- GAP FILLING LOGIC ---
            # Extend both chunks to the midpoint to eliminate the white gap.
            midpoint = (positions[i-1] + positions[i]) // 2
            
            # Seal current chunk at midpoint
            chunks.append(PaintedChunk(
                start=int(start_pos),
                end=int(midpoint),
                hap1=curr_t1,
                hap2=curr_t2
            ))
            
            # Start new chunk at midpoint
            curr_t1, curr_t2 = next_t1, next_t2
            start_pos = midpoint
            
    # Final chunk
    end_pos = positions[-1]
    chunks.append(PaintedChunk(
        start=int(start_pos),
        end=int(end_pos),
        hap1=curr_t1,
        hap2=curr_t2
    ))
    
    return chunks

def paint_samples_in_block(block_result, sample_probs_matrix, sample_sites, 
                           recomb_rate=1e-8,
                           switch_penalty=10.0,
                           robustness_epsilon=1e-3, # Tunable param
                           batch_size=10):
    """
    Paints all samples in a block using Vectorized Viterbi.
    
    Args:
        robustness_epsilon: Lower values (1e-12) enforce stricter matching.
    """
    
    # 1. Extract Data
    positions = block_result.positions
    hap_dict = block_result.haplotypes
    
    # Subset sample data to match the block's positions
    block_samples_data = analysis_utils.get_sample_data_at_sites_multiple(
        sample_probs_matrix, sample_sites, positions
    )
    
    num_samples = block_samples_data.shape[0]
    all_sample_paintings = []
    
    # 2. Iterate in batches to save memory
    print(f"Painting {num_samples} samples in batches of {batch_size} (eps={robustness_epsilon:.1e})...")
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        
        # Batch data
        batch_data = block_samples_data[start_idx:end_idx]
        
        # Calculate Emissions for batch (Passing user-defined epsilon)
        ll_tensor, state_defs, hap_keys = calculate_batch_emissions(
            batch_data, hap_dict, robustness_epsilon=robustness_epsilon
        )
        
        # Run Viterbi for batch
        num_haps = len(hap_keys)
        raw_paths = viterbi_painting_solver(
            ll_tensor, 
            positions, 
            recomb_rate, 
            state_defs, 
            num_haps,
            float(switch_penalty)
        )
        
        # Post-Process
        batch_count = raw_paths.shape[0]
        for i in range(batch_count):
            global_sample_idx = start_idx + i
            chunks = compress_path_to_chunks(raw_paths[i], positions, state_defs, hap_keys)
            all_sample_paintings.append(SamplePainting(global_sample_idx, chunks))
            
    range_tuple = (int(positions[0]), int(positions[-1]))
    return BlockPainting(range_tuple, all_sample_paintings)

# =============================================================================
# 5. VISUALIZATION
# =============================================================================

def plot_painting(block_painting, output_file=None, 
                  title="Chromosome Painting", 
                  figsize_width=20, 
                  row_height_per_sample=0.25,
                  show_labels=True,
                  sample_names=None):
    """
    Plots the painted haplotypes for all samples using DYNAMIC SIZING.
    """
    if not HAS_PLOTTING:
        print("Error: Matplotlib/Seaborn not installed.")
        return

    # 1. Identify all unique haplotypes for coloring
    unique_haps = set()
    for sample in block_painting:
        for chunk in sample:
            unique_haps.add(chunk.hap1)
            unique_haps.add(chunk.hap2)
    
    sorted_haps = sorted(list(unique_haps))
    hap_to_idx = {h: i for i, h in enumerate(sorted_haps)}
    
    # 2. Generate Color Palette
    if len(sorted_haps) <= 10:
        palette = sns.color_palette("tab10", len(sorted_haps))
    elif len(sorted_haps) <= 20:
        palette = sns.color_palette("tab20", len(sorted_haps))
    else:
        palette = sns.color_palette("husl", len(sorted_haps))
        
    # 3. Dynamic Figure Sizing
    num_samples = len(block_painting)
    header_space = 2.0 
    calc_height = (num_samples * row_height_per_sample) + header_space
    
    # Ensure reasonable limits
    if calc_height < 6: calc_height = 6
    if calc_height > 300: 
        print(f"Warning: Plot height ({calc_height:.1f} in) is very large.")
    
    figsize = (figsize_width, calc_height)
    fig, ax = plt.subplots(figsize=figsize)
    
    y_height = 0.8  # Leave 0.1 margin top/bottom within the row
    
    # Loop over samples (Y-axis)
    # Reverse index so Sample 0 is at the top
    for i, sample in enumerate(block_painting):
        y_base = i 
        
        # Loop over chunks (X-axis)
        for chunk in sample:
            width = chunk.end - chunk.start
            if width <= 0: continue
            
            # Hap 1 Rectangle (Bottom half of the row)
            color1 = palette[hap_to_idx[chunk.hap1]]
            rect1 = mpatches.Rectangle(
                (chunk.start, y_base), width, y_height/2,
                facecolor=color1, edgecolor='none'
            )
            ax.add_patch(rect1)
            
            # Hap 2 Rectangle (Top half of the row)
            color2 = palette[hap_to_idx[chunk.hap2]]
            rect2 = mpatches.Rectangle(
                (chunk.start, y_base + y_height/2), width, y_height/2,
                facecolor=color2, edgecolor='none'
            )
            ax.add_patch(rect2)
            
    # 4. Formatting
    ax.set_xlim(block_painting.start_pos, block_painting.end_pos)
    ax.set_ylim(-0.5, len(block_painting) + 0.5)
    
    ax.set_xlabel("Genomic Position (bp)")
    ax.set_ylabel("Samples")
    ax.set_title(title)
    
    # Y-Ticks
    if show_labels:
        if sample_names and len(sample_names) == len(block_painting):
            labels = sample_names
        else:
            labels = [f"S{s.sample_index}" for s in block_painting]
        
        ax.set_yticks(np.arange(len(block_painting)) + y_height/2)
        ax.set_yticklabels(labels, fontsize=8)
    else:
        ax.set_yticks([])

    # Legend
    legend_patches = []
    for h_key in sorted_haps:
        c = palette[hap_to_idx[h_key]]
        legend_patches.append(mpatches.Patch(color=c, label=f"Founder {h_key}"))
        
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.01, 1), loc='upper left')
    
    plt.tight_layout()
    
    if output_file:
        dpi = 100 if calc_height > 100 else 150
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Painting saved to {output_file} (Height: {calc_height:.1f} in)")
    else:
        plt.show()
    plt.close()