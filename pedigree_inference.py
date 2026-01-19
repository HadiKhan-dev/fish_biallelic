import numpy as np
import pandas as pd
import warnings
import math
from itertools import combinations
from tqdm import tqdm

# Visualization Imports
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_VIS = True
except ImportError:
    HAS_VIS = False

warnings.filterwarnings("ignore")

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("WARNING: Numba not found. Pedigree inference will be slow.")
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator
    prange = range

# =============================================================================
# 1. DATA STRUCTURES
# =============================================================================

class PedigreeResult:
    def __init__(self, samples, relationships, parent_candidates, recombination_map, systematic_errors, kinship_matrix, ibd0_matrix):
        self.samples = samples
        self.relationships = relationships 
        self.parent_candidates = parent_candidates 
        self.recombination_map = recombination_map 
        self.systematic_errors = systematic_errors 
        self.kinship_matrix = kinship_matrix
        self.ibd0_matrix = ibd0_matrix

# =============================================================================
# 2. DISCRETIZATION & ALLELE CONVERSION
# =============================================================================

def discretize_paintings(block_painting, snps_per_bin=50):
    """
    Converts RLE paintings into a fixed-grid NumPy array of IDs.
    Returns grid, bin_edges, and bin_centers.
    """
    start_pos = block_painting.start_pos
    end_pos = block_painting.end_pos
    total_len = end_pos - start_pos
    approx_bp_per_bin = snps_per_bin * 100 
    num_bins = int(total_len / approx_bp_per_bin)
    if num_bins < 100: num_bins = 100 
    
    bin_edges = np.linspace(start_pos, end_pos, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    num_samples = len(block_painting)
    grid = np.zeros((num_samples, num_bins, 2), dtype=np.int32) - 1 
    
    # print(f"Discretizing genome into {num_bins} bins...")
    
    for i, sample in enumerate(block_painting):
        chunks = sample.chunks
        if not chunks: continue
        c_ends = np.array([c.end for c in chunks])
        c_h1 = np.array([c.hap1 for c in chunks])
        c_h2 = np.array([c.hap2 for c in chunks])
        c_starts = np.array([c.start for c in chunks])
        
        indices = np.searchsorted(c_ends, bin_centers)
        indices = np.clip(indices, 0, len(chunks) - 1)
        valid_mask = bin_centers >= c_starts[indices]
        
        grid[i, :, 0] = np.where(valid_mask, c_h1[indices], -1)
        grid[i, :, 1] = np.where(valid_mask, c_h2[indices], -1)
            
    return grid, bin_edges, bin_centers

def convert_id_grid_to_allele_grid(id_grid, bin_centers, founder_block):
    """Translates Founder IDs to Alleles (0/1/missing)."""
    # print("Translating Founder IDs to Alleles...")
    num_samples, num_bins, _ = id_grid.shape
    bin_indices = np.searchsorted(founder_block.positions, bin_centers)
    bin_indices = np.clip(bin_indices, 0, len(founder_block.positions) - 1)
    
    hap_keys = sorted(list(founder_block.haplotypes.keys()))
    max_id = max(hap_keys) if hap_keys else 0
    allele_lookup = np.full((max_id + 1, num_bins), -1, dtype=np.int8)
    
    for fid, h_arr in founder_block.haplotypes.items():
        if h_arr.ndim == 2: raw_alleles = np.argmax(h_arr, axis=1)
        else: raw_alleles = h_arr
        allele_lookup[fid, :] = raw_alleles[bin_indices]
        
    allele_grid = np.full_like(id_grid, -1, dtype=np.int8)
    b_indices = np.arange(num_bins)
    
    for chrom in [0, 1]:
        ids = id_grid[:, :, chrom]
        valid_mask = (ids != -1)
        safe_ids = ids.copy()
        safe_ids[~valid_mask] = 0
        alleles = allele_lookup[safe_ids, b_indices[np.newaxis, :]]
        alleles[~valid_mask] = -1
        allele_grid[:, :, chrom] = alleles
        
    return allele_grid

# =============================================================================
# 3. STATISTICAL CALCULATIONS
# =============================================================================

def calculate_ibd_matrices(grid):
    num_samples, num_bins, _ = grid.shape
    kinship_sum = np.zeros((num_samples, num_samples))
    ibd0_sum = np.zeros((num_samples, num_samples))
    valid_counts = np.zeros((num_samples, num_samples))
    batch_size = 500
    for b_start in range(0, num_bins, batch_size):
        b_end = min(b_start + batch_size, num_bins)
        g_sub = grid[:, b_start:b_end, :]
        A = g_sub[:, np.newaxis, :, :]
        B = g_sub[np.newaxis, :, :, :]
        valid_mask = (A[..., 0] != -1) & (B[..., 0] != -1)
        valid_counts += np.sum(valid_mask, axis=2)
        match_direct = (A[...,0] == B[...,0]) & (A[...,1] == B[...,1])
        match_cross  = (A[...,0] == B[...,1]) & (A[...,1] == B[...,0])
        match_both   = match_direct | match_cross
        any_match = (A[...,0] == B[...,0]) | (A[...,0] == B[...,1]) | (A[...,1] == B[...,0]) | (A[...,1] == B[...,1])
        k_scores = np.zeros_like(match_both, dtype=float)
        k_scores[any_match] = 0.5; k_scores[match_both] = 1.0; k_scores[~valid_mask] = 0.0
        kinship_sum += np.sum(k_scores, axis=2)
        ibd0_scores = (~any_match).astype(float); ibd0_scores[~valid_mask] = 0.0
        ibd0_sum += np.sum(ibd0_scores, axis=2)
    valid_counts[valid_counts == 0] = 1.0
    return kinship_sum / valid_counts, ibd0_sum / valid_counts

def analyze_recombinations(grid, bin_edges, gen_map, systematic_thresh=0.25):
    num_samples, num_bins, _ = grid.shape
    switches = (grid[:, :-1, :] != grid[:, 1:, :]) & (grid[:, :-1, :] != -1)
    any_switch = np.any(switches, axis=2).astype(int)
    global_freq = np.mean(any_switch, axis=0)
    bad_bin_indices = np.where(global_freq > systematic_thresh)[0]
    events = []
    for i in range(num_samples):
        my_switches = np.where(any_switch[i, :] == 1)[0]
        for bin_idx in my_switches:
            if not any(abs(bad - bin_idx) <= 1 for bad in bad_bin_indices):
                events.append({'Sample_Index': i, 'Generation': gen_map.get(i, 'Unknown'), 'Approx_Position': bin_edges[bin_idx+1]})
    return pd.DataFrame(events), bad_bin_indices

# =============================================================================
# 4. HMM KERNEL (DISTANCE AWARE)
# =============================================================================

@njit(fastmath=True)
def run_inheritance_hmm_dynamic(target_hap, source_dip, switch_costs, stay_costs, error_penalty):
    """
    Calculates the Viterbi Score of generating 'target_hap' from 'source_dip'
    using distance-dependent transition probabilities.
    """
    n_sites = len(target_hap)
    scores = np.zeros(3) 
    
    BURST_EMISSION = -0.693 
    
    scores[0] = 0.0
    scores[1] = 0.0
    scores[2] = -error_penalty 
    
    for i in range(n_sites):
        obs = target_hap[i]
        if obs == -1: 
            # Simplified: Apply transition cost, 0 emission.
            e0 = 0.0
            e1 = 0.0
            e2 = 0.0
        else:
            s0_allele = source_dip[i, 0]
            s1_allele = source_dip[i, 1]
            e0 = 0.0 if (s0_allele == -1 or s0_allele == obs) else -1e9
            e1 = 0.0 if (s1_allele == -1 or s1_allele == obs) else -1e9
            e2 = BURST_EMISSION
        
        # Get dynamic costs for this step
        c_switch = switch_costs[i]
        c_stay = stay_costs[i]
        
        prev = scores.copy()
        
        # 0: From 0(Stay), 1(Switch), 2(Recovery)
        v0 = max(prev[0] + c_stay, prev[1] + c_switch, prev[2]) + e0
        
        # 1: From 1(Stay), 0(Switch), 2(Recovery)
        v1 = max(prev[1] + c_stay, prev[0] + c_switch, prev[2]) + e1
        
        # 2: Burst (From 0/1 pay penalty, From 2 free)
        v2 = max(prev[0] - error_penalty, prev[1] - error_penalty, prev[2]) + e2
        
        scores[0] = v0
        scores[1] = v1
        scores[2] = v2
        
    return max(scores[0], scores[1], scores[2])

@njit(parallel=True)
def batch_calculate_parent_scores(allele_grid, candidates_mask, switch_costs, stay_costs, error_penalty):
    n_samples, n_bins, _ = allele_grid.shape
    scores = np.full((n_samples, 2, n_samples), -np.inf)
    
    for i in prange(n_samples): 
        child_h0 = allele_grid[i, :, 0]
        child_h1 = allele_grid[i, :, 1]
        
        for j in range(n_samples): 
            if i == j: continue
            if not candidates_mask[i, j]: continue
            
            parent_dip = allele_grid[j]
            s0 = run_inheritance_hmm_dynamic(child_h0, parent_dip, switch_costs, stay_costs, error_penalty)
            scores[i, 0, j] = s0
            s1 = run_inheritance_hmm_dynamic(child_h1, parent_dip, switch_costs, stay_costs, error_penalty)
            scores[i, 1, j] = s1
            
    return scores

# =============================================================================
# 5. MULTI-CONTIG INFERENCE LOGIC
# =============================================================================

def score_contig_raw(block_painting, founder_block, sample_ids, 
                     snps_per_bin=150, recomb_rate=5e-8, robustness=1e-2):
    """
    Step 1: Calculates raw Parent-Offspring HMM scores for a single contig.
    Returns:
        scores_matrix: (N_Samples x 2 x N_Samples) float array.
                       scores[i, 0, j] = Score of Child i (Hap 0) coming from Parent j.
        recomb_counts: (N_Samples,) int array of switch counts for this contig.
    """
    # 1. Discretize
    id_grid, bin_edges, bin_centers = discretize_paintings(block_painting, snps_per_bin=snps_per_bin)
    
    # 2. Convert to Alleles
    allele_grid = convert_id_grid_to_allele_grid(id_grid, bin_centers, founder_block)
    
    num_samples = len(sample_ids)
    num_bins = len(bin_centers)
    
    # 3. Calculate HMM Costs
    dists = np.zeros(num_bins)
    dists[1:] = np.diff(bin_centers)
    
    theta = 1.0 - np.exp(-dists * recomb_rate)
    theta = np.clip(theta, 1e-15, 0.5)
    
    switch_costs = np.log(theta)
    stay_costs = np.log(1.0 - theta)
    
    # Convert Robustness epsilon to Log Penalty
    # e.g. 1e-2 -> -ln(0.01) ~= 4.6
    error_penalty = -math.log(robustness)
    
    # 4. Count Switches (for Directionality Filter later)
    switches = (id_grid[:, :-1, :] != id_grid[:, 1:, :]) & \
               (id_grid[:, :-1, :] != -1) & (id_grid[:, 1:, :] != -1)
    recomb_counts = np.sum(switches, axis=(1, 2))
    
    # 5. Run HMM Scoring (No filtering yet, we filter globally later)
    # We pass a "True" mask to score ALL pairs
    full_mask = np.ones((num_samples, num_samples), dtype=bool)
    np.fill_diagonal(full_mask, False) # Don't score self-parentage
    
    # print(f"  Scoring {num_samples} samples (HMM penalty={error_penalty:.2f})...")
    scores = batch_calculate_parent_scores(
        allele_grid, full_mask, switch_costs, stay_costs, error_penalty
    )
    
    return scores, recomb_counts

def infer_pedigree_multi_contig(contig_data_list, sample_ids, top_k=20):
    """
    Step 2: Aggregates scores across multiple contigs to infer trios.
    
    Args:
        contig_data_list: List of dicts, each containing:
                          {'painting': ..., 'founder_block': ...}
    """
    num_samples = len(sample_ids)
    num_contigs = len(contig_data_list)
    
    # Accumulators
    total_switches = np.zeros(num_samples)
    
    # Temporary storage for trio scoring
    # We need to store (N, 2, N) for each contig to handle Trio Phasing
    all_contig_scores = [] 
    
    print(f"\n--- Aggregating Scores across {num_contigs} Contigs ---")
    
    for c_idx, data in enumerate(contig_data_list):
        print(f"Processing Contig {c_idx+1}/{num_contigs}...")
        
        # Calculate Raw Scores
        scores, switches = score_contig_raw(
            data['painting'], 
            data['founder_block'], 
            sample_ids,
            snps_per_bin=150,
            robustness=1e-2
        )
        
        all_contig_scores.append(scores)
        total_switches += switches
        
    # --- 1. Global Directionality Filter ---
    # A parent must have fewer recombinations than the child (globally).
    # We allow a margin of error (e.g. 5 switches per contig)
    cand_mask = np.zeros((num_samples, num_samples), dtype=bool)
    margin = 5 * num_contigs
    
    for i in range(num_samples):
        # Candidates must have fewer switches than Child i
        valid_gen = total_switches <= (total_switches[i] + margin)
        cand_mask[i, :] = valid_gen
        cand_mask[i, i] = False # No self-parenting

    # --- 2. Trio Formation (Phase-Aware Aggregation) ---
    relationships = []
    parent_candidates = {}
    
    for i in tqdm(range(num_samples), desc="Inferring Trios"):
        
        # A. Pre-select Top Candidates (to avoid N^2 loop)
        # We sum the Max likelihoods across contigs to find "Good Individual Parents"
        agg_single_scores = np.zeros(num_samples)
        for j in range(num_samples):
            if not cand_mask[i, j]: 
                agg_single_scores[j] = -np.inf
                continue
            
            sum_score = 0
            for c in range(num_contigs):
                # Max of matching H0 or matching H1 on this contig
                s_mat = all_contig_scores[c]
                best_fit = max(s_mat[i, 0, j], s_mat[i, 1, j])
                sum_score += best_fit
            agg_single_scores[j] = sum_score
            
        # Select Top K candidates
        top_candidates = np.argsort(agg_single_scores)[-top_k:][::-1]
        top_candidates = [x for x in top_candidates if agg_single_scores[x] > -1e10]
        
        parent_candidates[sample_ids[i]] = [(sample_ids[x], agg_single_scores[x]) for x in top_candidates]

        if len(top_candidates) < 1:
            relationships.append({'Sample': sample_ids[i], 'Generation': 'F1', 
                                  'Parent1': None, 'Parent2': None})
            continue

        # B. Score Trios (Phase-Correct Summation)
        best_trio = None
        best_trio_score = -np.inf
        
        # Try all pairs of top candidates
        combinations_list = [(p1, p2) for p1 in top_candidates for p2 in top_candidates if p1 != p2]
        
        if not combinations_list:
             # Fallback if only 1 candidate exists
             combinations_list = [(top_candidates[0], top_candidates[0])]

        for p1, p2 in combinations_list:
            
            trio_total_log_lik = 0.0
            
            for c in range(num_contigs):
                s_mat = all_contig_scores[c]
                
                # Option A: P1->H0, P2->H1
                lik_a = s_mat[i, 0, p1] + s_mat[i, 1, p2]
                
                # Option B: P1->H1, P2->H0
                lik_b = s_mat[i, 1, p1] + s_mat[i, 0, p2]
                
                # We don't know phase, so we take the configuration that fits best for this contig
                trio_total_log_lik += max(lik_a, lik_b)
            
            # Tie-Breaker: Complexity (Total Genome Switches)
            complexity = (total_switches[p1] + total_switches[p2]) * 0.1
            final_score = trio_total_log_lik - complexity
            
            if final_score > best_trio_score:
                best_trio_score = final_score
                best_trio = (p1, p2)
        
        # Result
        if best_trio:
            p1_name, p2_name = sample_ids[best_trio[0]], sample_ids[best_trio[1]]
            relationships.append({'Sample': sample_ids[i], 'Generation': 'Unknown', 
                                  'Parent1': p1_name, 'Parent2': p2_name})
        else:
            relationships.append({'Sample': sample_ids[i], 'Generation': 'F1', 
                                  'Parent1': None, 'Parent2': None})
            
    # --- 3. Final Formatting ---
    rel_df = pd.DataFrame(relationships)
    
    # Infer Generations labels (iterative)
    name_to_gen = {row['Sample']: 'F1' for _, row in rel_df.iterrows() if pd.isna(row['Parent1'])}
    for _ in range(10): # Depth of tree
        for idx, row in rel_df.iterrows():
            if row['Generation'] != 'Unknown': continue
            p1, p2 = row['Parent1'], row['Parent2']
            if p1 in name_to_gen and p2 in name_to_gen:
                try:
                    g1 = int(name_to_gen[p1][1:])
                    g2 = int(name_to_gen[p2][1:])
                    gen = f"F{max(g1, g2) + 1}"
                    rel_df.at[idx, 'Generation'] = gen
                    name_to_gen[row['Sample']] = gen
                except: pass

    # Return Result Object (Empty maps/stats since they are per-contig specific)
    return PedigreeResult(sample_ids, rel_df, parent_candidates, None, [], None, None)

# =============================================================================
# 6. VISUALIZATION & WRAPPER
# =============================================================================

def draw_pedigree_tree(relationships_df, output_file="pedigree_tree.png"):
    if not HAS_VIS: return
    G = nx.DiGraph()
    gen_nodes = {'F1':[], 'F2':[], 'F3':[], 'Unknown':[]}
    parents_of = {}
    for _, row in relationships_df.iterrows():
        sample = row['Sample']; gen = row['Generation']
        if gen not in gen_nodes: gen_nodes[gen] = []
        gen_nodes[gen].append(sample)
        color = "#999999"
        if gen == 'F1': color = "#1f77b4"
        elif gen == 'F2': color = "#ff7f0e"
        elif gen == 'F3': color = "#2ca02c"
        G.add_node(sample, color=color, gen=gen)
        if pd.notna(row['Parent1']): 
            G.add_edge(row['Parent1'], sample); parents_of.setdefault(sample, []).append(row['Parent1'])
        if pd.notna(row['Parent2']): 
            G.add_edge(row['Parent2'], sample); parents_of.setdefault(sample, []).append(row['Parent2'])
    pos = {}
    node_y = {}
    layers = sorted([k for k in gen_nodes.keys() if k.startswith('F')], key=lambda x: int(x[1:]))
    if 'Unknown' in gen_nodes: layers.append('Unknown')
    for x_idx, gen in enumerate(layers):
        nodes = gen_nodes[gen]
        if not nodes: continue
        if x_idx == 0: nodes.sort()
        else: nodes.sort(key=lambda n: sum([node_y.get(p, 0.5) for p in parents_of.get(n, [])])/len(parents_of.get(n, [])) if parents_of.get(n) else 0.5, reverse=True)
        for i, n in enumerate(nodes):
            y = 1.0 - (i + 0.5) / len(nodes); pos[n] = (x_idx, y); node_y[n] = y
    plt.figure(figsize=(20, max(10, len(gen_nodes.get('F3', []))*0.2)))
    node_colors = [G.nodes[n]['color'] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=80, node_color=node_colors, edgecolors='black', linewidths=0.5)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3, width=0.5, arrows=False)
    to_label = gen_nodes.get('F1', []) + gen_nodes.get('F2', [])
    labels = {n:n for n in to_label}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    plt.axis('off'); plt.tight_layout(); plt.savefig(output_file, dpi=150); plt.close()

def run_pedigree_inference(block_painting, sample_ids=None, snps_per_bin=150, 
                           founder_block=None, recomb_rate=5e-8,
                           output_prefix="pedigree"):
    """
    Legacy wrapper for single-contig inference.
    Packages the single input into the list format required by infer_pedigree_multi_contig.
    """
    
    if founder_block is None:
        raise ValueError("Founder Block is now required for allele-based inference.")

    if sample_ids is None:
        sample_ids = [f"S_{i}" for i in range(len(block_painting))]

    # Package as list of dicts
    contig_input = [{
        'painting': block_painting,
        'founder_block': founder_block
    }]
    
    result = infer_pedigree_multi_contig(contig_input, sample_ids, top_k=20)
    
    result.relationships.to_csv(f"{output_prefix}.ped", index=False)
    draw_pedigree_tree(result.relationships, output_file=f"{output_prefix}_tree.png")
    
    return result