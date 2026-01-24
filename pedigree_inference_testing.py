import numpy as np
import pandas as pd
import warnings
import math
from itertools import combinations
from tqdm import tqdm
from sklearn.cluster import KMeans

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
    def __init__(self, samples, relationships, parent_candidates, 
                 recombination_map, systematic_errors, kinship_matrix, ibd0_matrix,
                 trio_scores=None, total_bins=0):
        self.samples = samples
        self.relationships = relationships 
        self.parent_candidates = parent_candidates 
        self.recombination_map = recombination_map 
        self.systematic_errors = systematic_errors 
        self.kinship_matrix = kinship_matrix
        self.ibd0_matrix = ibd0_matrix
        self.trio_scores = trio_scores if trio_scores is not None else {}
        self.total_bins = total_bins

    def _recalculate_generations(self):
        """
        Internal helper to propagate generation labels (F1 -> F2 -> F3)
        down the tree after the pedigree structure has been modified.
        """
        # 1. Reset all to 'Unknown'
        self.relationships['Generation'] = 'Unknown'
        
        # 2. Identify Roots (F1) - Samples with No Parents
        # Note: In your dataframe, None/NaN indicates no parents
        is_root = self.relationships['Parent1'].isna()
        self.relationships.loc[is_root, 'Generation'] = 'F1'
        
        # 3. Create Lookup
        # sample_name -> generation_string
        name_to_gen = dict(zip(self.relationships['Sample'], self.relationships['Generation']))
        
        # 4. Propagate
        # We loop enough times to cover the depth of the tree
        for _ in range(10):
            updates = 0
            for idx, row in self.relationships.iterrows():
                if row['Generation'] != 'Unknown': continue
                
                p1 = row['Parent1']
                p2 = row['Parent2']
                
                # Check if parents have assigned generations
                if p1 in name_to_gen and p2 in name_to_gen:
                    g1_str = name_to_gen[p1]
                    g2_str = name_to_gen[p2]
                    
                    if g1_str != 'Unknown' and g2_str != 'Unknown':
                        # Parse "F1" -> 1
                        try:
                            g1 = int(g1_str[1:])
                            g2 = int(g2_str[1:])
                            new_gen_num = max(g1, g2) + 1
                            new_gen_str = f"F{new_gen_num}"
                            
                            self.relationships.at[idx, 'Generation'] = new_gen_str
                            name_to_gen[row['Sample']] = new_gen_str
                            updates += 1
                        except:
                            pass
            if updates == 0:
                break

    def perform_automatic_cutoff(self, force_clusters=2):
        """
        Uses K-Means on Score-Per-Bin to distinguish True Trios (F2/F3) from
        False Trios (F1s matched to siblings).
        """
        if self.total_bins == 0: return
        
        scores = []
        indices = []
        
        for i, s in enumerate(self.samples):
            raw = self.trio_scores.get(s, -1e9)
            norm_score = raw / self.total_bins
            scores.append(norm_score)
            indices.append(i)
            
        scores = np.array(scores).reshape(-1, 1)
        
        try:
            kmeans = KMeans(n_clusters=force_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(scores)
            centers = kmeans.cluster_centers_.flatten()
            
            # The cluster with the HIGHER score center is the "True Parents" cluster
            good_cluster = np.argmax(centers)
            
            print(f"\n[Auto-Cutoff] Score Centers: {centers}")
            print(f"[Auto-Cutoff] True Parent Cluster Center: {centers[good_cluster]:.4f} (Score/Bin)")
            print(f"[Auto-Cutoff] Noise/Sibling Cluster Center: {centers[1-good_cluster]:.4f} (Score/Bin)")
            
            updates = 0
            for i, label in zip(indices, labels):
                if label != good_cluster:
                    # Identify as Root (F1)
                    self.relationships.at[i, 'Parent1'] = None
                    self.relationships.at[i, 'Parent2'] = None
                    updates += 1
                    
            print(f"[Auto-Cutoff] Reclassified {updates} samples as Founders/F1 (No Parents found).")
            
            # NOW calculate generations based on this clean structure
            self._recalculate_generations()
            
        except Exception as e:
            print(f"[Auto-Cutoff] Failed: {e}. Skipping.")
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
# 4. HMM KERNELS (8-STATE FILTER & 16-STATE VERIFIER)
# =============================================================================

@njit(fastmath=True)
def run_phase_agnostic_hmm(child_dip_alleles, child_dip_ids, parent_dip_alleles, switch_costs, stay_costs, error_penalty, phase_penalty):
    """
    Calculates the Viterbi Score of Parent -> Child inheritance allowing for
    phase switching in the Child AND Split-Burst error handling.
    """
    n_sites = len(child_dip_alleles)
    
    # Init Scores (8 states) [Norm0..3, Burst0..3]
    scores = np.zeros(8)
    
    # Burst Emission: log(0.5)
    BURST_EMISSION = -0.693147 
    
    # Initialize Bursts as valid starting points (entering with penalty)
    for k in range(4, 8):
        scores[k] = -error_penalty
    
    for i in range(n_sites):
        # Alleles for Emissions
        c0_a, c1_a = child_dip_alleles[i, 0], child_dip_alleles[i, 1]
        p0_a, p1_a = parent_dip_alleles[i, 0], parent_dip_alleles[i, 1]
        
        # IDs for Phase Logic
        c0_id, c1_id = child_dip_ids[i, 0], child_dip_ids[i, 1]
        
        # 1. Normal Emissions (Based on Alleles)
        # -1 (Missing) is treated as a wildcard match (0 cost)
        e0 = 0.0 if (c0_a == -1 or p0_a == -1 or c0_a == p0_a) else -1e9
        e1 = 0.0 if (c1_a == -1 or p0_a == -1 or c1_a == p0_a) else -1e9
        e2 = 0.0 if (c0_a == -1 or p1_a == -1 or c0_a == p1_a) else -1e9
        e3 = 0.0 if (c1_a == -1 or p1_a == -1 or c1_a == p1_a) else -1e9
        
        emissions = np.array([e0, e1, e2, e3])
        
        # 2. Transition Costs
        c_recomb = switch_costs[i]
        c_stay = stay_costs[i]
        
        # Child Phase Switch Cost (Based on Founder IDs)
        is_ibd_ambiguous = (c0_id == c1_id) or (c0_id == -1) or (c1_id == -1)
        c_phase = 0.0 if is_ibd_ambiguous else -phase_penalty
        
        prev = scores.copy()
        new_scores = np.zeros(8)
        
        # --- A. UPDATE BURST STATES (4-7) ---
        for k in range(4):
            burst_idx = k + 4
            from_burst = prev[burst_idx] 
            from_normal = prev[k] - error_penalty
            new_scores[burst_idx] = max(from_burst, from_normal) + BURST_EMISSION

        # --- B. UPDATE NORMAL STATES (0-3) ---
        prev_b0 = prev[4]; prev_b1 = prev[5]; prev_b2 = prev[6]; prev_b3 = prev[7]
        
        # State 0 (P0, C0): From 0(Stay), 1(Phase), 2(Recomb), Burst0(Recov)
        src0 = prev[0] + c_stay
        src1 = prev[1] + c_stay + c_phase
        src2 = prev[2] + c_recomb
        new_scores[0] = max(src0, src1, src2, prev_b0) + emissions[0]
        
        # State 1 (P0, C1): From 1, 0, 3, Burst1
        src1 = prev[1] + c_stay
        src0 = prev[0] + c_stay + c_phase
        src3 = prev[3] + c_recomb
        new_scores[1] = max(src1, src0, src3, prev_b1) + emissions[1]
        
        # State 2 (P1, C0): From 2, 3, 0, Burst2
        src2 = prev[2] + c_stay
        src3 = prev[3] + c_stay + c_phase
        src0 = prev[0] + c_recomb
        new_scores[2] = max(src2, src3, src0, prev_b2) + emissions[2]
        
        # State 3 (P1, C1): From 3, 2, 1, Burst3
        src3 = prev[3] + c_stay
        src2 = prev[2] + c_stay + c_phase
        src1 = prev[1] + c_recomb
        new_scores[3] = max(src3, src2, src1, prev_b3) + emissions[3]
        
        scores = new_scores
        
    # Return max over all 8 states
    best_final = -np.inf
    for k in range(8):
        if scores[k] > best_final:
            best_final = scores[k]
    return best_final

@njit(parallel=True)
def batch_calculate_parent_scores(allele_grid, id_grid, candidates_mask, switch_costs, stay_costs, error_penalty, phase_penalty):
    n_samples, n_bins, _ = allele_grid.shape
    # Returns (N, N) matrix of single best scores
    scores = np.full((n_samples, n_samples), -np.inf)
    
    for i in prange(n_samples): 
        child_alleles = allele_grid[i]
        child_ids = id_grid[i]
        
        for j in range(n_samples): 
            if i == j: continue
            if not candidates_mask[i, j]: continue
            
            parent_alleles = allele_grid[j]
            s = run_phase_agnostic_hmm(child_alleles, child_ids, parent_alleles, switch_costs, stay_costs, error_penalty, phase_penalty)
            scores[i, j] = s
            
    return scores

@njit(fastmath=True)
def run_trio_phase_aware_hmm(child_dip_alleles, child_dip_ids, p1_dip_alleles, p2_dip_alleles, 
                             switch_costs, stay_costs, error_penalty, phase_penalty):
    """
    Calculates the Joint Likelihood of P1 and P2 explaining the Child.
    Includes 16-State Split-Burst Logic to handle redundant coverage and genotyping errors.
    """
    n_sites = len(child_dip_alleles)
    BURST_EMISSION = -1.386 # ln(0.25)
    
    # Init Scores (16 states)
    scores = np.zeros(16)
    for k in range(8, 16): scores[k] = -error_penalty
    
    for i in range(n_sites):
        # 1. Data Fetch
        c0, c1 = child_dip_alleles[i, 0], child_dip_alleles[i, 1]
        p1_h0, p1_h1 = p1_dip_alleles[i, 0], p1_dip_alleles[i, 1]
        p2_h0, p2_h1 = p2_dip_alleles[i, 0], p2_dip_alleles[i, 1]
        cid0, cid1 = child_dip_ids[i, 0], child_dip_ids[i, 1]
        
        # 2. Emissions
        def match(a, b): return 0.0 if (a == -1 or b == -1 or a == b) else -1e9
        
        m_p1h0_c0 = match(p1_h0, c0); m_p1h1_c0 = match(p1_h1, c0)
        m_p1h0_c1 = match(p1_h0, c1); m_p1h1_c1 = match(p1_h1, c1)
        m_p2h0_c0 = match(p2_h0, c0); m_p2h1_c0 = match(p2_h1, c0)
        m_p2h0_c1 = match(p2_h0, c1); m_p2h1_c1 = match(p2_h1, c1)
        
        # Group A (P1->C0, P2->C1)
        e = np.zeros(8)
        e[0] = m_p1h0_c0 + m_p2h0_c1
        e[1] = m_p1h0_c0 + m_p2h1_c1
        e[2] = m_p1h1_c0 + m_p2h0_c1
        e[3] = m_p1h1_c0 + m_p2h1_c1
        
        # Group B (P1->C1, P2->C0)
        e[4] = m_p1h0_c1 + m_p2h0_c0
        e[5] = m_p1h0_c1 + m_p2h1_c0
        e[6] = m_p1h1_c1 + m_p2h0_c0
        e[7] = m_p1h1_c1 + m_p2h1_c0
        
        # 3. Transitions
        c_recomb = switch_costs[i]
        c_stay = stay_costs[i]
        
        is_hom = (cid0 == cid1) or (cid0 == -1) or (cid1 == -1)
        c_phase = 0.0 if is_hom else -phase_penalty
        
        prev = scores.copy()
        new_scores = np.zeros(16)
        
        # --- A. UPDATE BURST STATES (8-15) ---
        for k in range(8):
            burst_idx = k + 8
            from_burst = prev[burst_idx]
            from_normal = prev[k] - error_penalty
            new_scores[burst_idx] = max(from_burst, from_normal) + BURST_EMISSION

        # --- B. UPDATE NORMAL STATES (0-7) ---
        cc_0 = 2 * c_stay
        cc_1 = c_recomb + c_stay
        cc_2 = 2 * c_recomb
        
        # Helper: Max over previous 4 states in a group (Parent moves)
        def get_best_incoming_groupA(prev_arr):
            p0, p1, p2, p3 = prev_arr[0], prev_arr[1], prev_arr[2], prev_arr[3]
            t0 = max(p0+cc_0, p1+cc_1, p2+cc_1, p3+cc_2)
            t1 = max(p0+cc_1, p1+cc_0, p2+cc_2, p3+cc_1)
            t2 = max(p0+cc_1, p1+cc_2, p2+cc_0, p3+cc_1)
            t3 = max(p0+cc_2, p1+cc_1, p2+cc_1, p3+cc_0)
            return t0, t1, t2, t3

        def get_best_incoming_groupB(prev_arr):
            p4, p5, p6, p7 = prev_arr[4], prev_arr[5], prev_arr[6], prev_arr[7]
            t4 = max(p4+cc_0, p5+cc_1, p6+cc_1, p7+cc_2)
            t5 = max(p4+cc_1, p5+cc_0, p6+cc_2, p7+cc_1)
            t6 = max(p4+cc_1, p5+cc_2, p6+cc_0, p7+cc_1)
            t7 = max(p4+cc_2, p5+cc_1, p6+cc_1, p7+cc_0)
            return t4, t5, t6, t7

        a0, a1, a2, a3 = get_best_incoming_groupA(prev)
        b4, b5, b6, b7 = get_best_incoming_groupB(prev)
        
        # Previous Bursts
        pb = prev[8:16]
        
        new_scores[0] = max(a0 + c_stay, b4 + c_stay + c_phase, pb[0]) + e[0]
        new_scores[1] = max(a1 + c_stay, b5 + c_stay + c_phase, pb[1]) + e[1]
        new_scores[2] = max(a2 + c_stay, b6 + c_stay + c_phase, pb[2]) + e[2]
        new_scores[3] = max(a3 + c_stay, b7 + c_stay + c_phase, pb[3]) + e[3]
            
        new_scores[4] = max(b4 + c_stay, a0 + c_stay + c_phase, pb[4]) + e[4]
        new_scores[5] = max(b5 + c_stay, a1 + c_stay + c_phase, pb[5]) + e[5]
        new_scores[6] = max(b6 + c_stay, a2 + c_stay + c_phase, pb[6]) + e[6]
        new_scores[7] = max(b7 + c_stay, a3 + c_stay + c_phase, pb[7]) + e[7]
            
        scores = new_scores
        
    # Return max over all 16 states
    best_final = -np.inf
    for k in range(16):
        if scores[k] > best_final:
            best_final = scores[k]
    return best_final

# =============================================================================
# 5. MULTI-CONTIG INFERENCE LOGIC
# =============================================================================

def score_contig_raw(block_painting, founder_block, sample_ids, 
                     snps_per_bin=150, recomb_rate=5e-8, robustness=1e-2):
    """
    Step 1: Calculates raw Parent-Offspring HMM scores for a single contig.
    """
    id_grid, bin_edges, bin_centers = discretize_paintings(block_painting, snps_per_bin=snps_per_bin)
    allele_grid = convert_id_grid_to_allele_grid(id_grid, bin_centers, founder_block)
    
    num_samples = len(sample_ids)
    num_bins = len(bin_centers)
    
    dists = np.zeros(num_bins); dists[1:] = np.diff(bin_centers)
    theta = np.clip(1.0 - np.exp(-dists * recomb_rate), 1e-15, 0.5)
    switch_costs = np.log(theta)
    stay_costs = np.log(1.0 - theta)
    
    error_penalty = -math.log(robustness)
    phase_penalty = 50.0 
    
    switches = (id_grid[:, :-1, :] != id_grid[:, 1:, :]) & \
               (id_grid[:, :-1, :] != -1) & (id_grid[:, 1:, :] != -1)
    recomb_counts = np.sum(switches, axis=(1, 2))
    
    # Run Single HMM
    full_mask = np.ones((num_samples, num_samples), dtype=bool)
    np.fill_diagonal(full_mask, False) 
    
    scores = batch_calculate_parent_scores(
        allele_grid, id_grid, full_mask, switch_costs, stay_costs, error_penalty, phase_penalty
    )
    
    return scores, recomb_counts

def infer_pedigree_multi_contig(contig_data_list, sample_ids, top_k=20):
    """
    Step 2: Aggregates scores across multiple contigs.
    """
    num_samples = len(sample_ids)
    num_contigs = len(contig_data_list)
    
    # 1. Pre-Calc & Filter
    contig_caches = []
    total_scores = np.zeros((num_samples, num_samples))
    total_switches = np.zeros(num_samples)
    global_total_bins = 0
    
    print(f"\n--- Phase 1: Filtering Candidates ({num_contigs} Contigs) ---")
    
    for c_idx, data in enumerate(contig_data_list):
        id_grid, bin_edges, bin_centers = discretize_paintings(data['painting'], snps_per_bin=150)
        allele_grid = convert_id_grid_to_allele_grid(id_grid, bin_centers, data['founder_block'])
        
        global_total_bins += len(bin_centers)
        
        dists = np.zeros(len(bin_centers)); dists[1:] = np.diff(bin_centers)
        theta = np.clip(1.0 - np.exp(-dists * 5e-8), 1e-15, 0.5)
        sw_costs = np.log(theta); st_costs = np.log(1.0 - theta)
        
        contig_caches.append({
            'allele_grid': allele_grid,
            'id_grid': id_grid,
            'sw_costs': sw_costs,
            'st_costs': st_costs
        })
        
        # Score Singles
        error_pen = -math.log(1e-2)
        full_mask = np.ones((num_samples, num_samples), dtype=bool); np.fill_diagonal(full_mask, False)
        
        scores = batch_calculate_parent_scores(allele_grid, id_grid, full_mask, sw_costs, st_costs, error_pen, 50.0)
        
        row_switches = (id_grid[:, :-1, :] != id_grid[:, 1:, :]) & (id_grid[:, :-1, :] != -1) & (id_grid[:, 1:, :] != -1)
        recomb_counts = np.sum(row_switches, axis=(1, 2))
        
        scores[scores == -np.inf] = -1e9
        total_scores += scores
        total_switches += recomb_counts

    cand_mask = np.zeros((num_samples, num_samples), dtype=bool)
    margin = 5
    for i in range(num_samples):
        valid_gen = total_switches <= (total_switches[i] + margin)
        cand_mask[i, :] = valid_gen
        cand_mask[i, i] = False

    # --- 2. Trio Verification ---
    relationships = []
    parent_candidates = {}
    trio_scores_map = {}
    COMPLEXITY_PENALTY = 0.0
    
    print(f"\n--- Phase 2: Trio Verification (Top {top_k} Pairs) ---")
    
    for i in tqdm(range(num_samples), desc="Inferring Trios"):
        
        # Get Top Candidates
        valid_scores = total_scores[i].copy()
        valid_scores[~cand_mask[i, :]] = -np.inf
        # Apply Penalty for ranking
        for j in range(num_samples):
            if valid_scores[j] > -1e9: valid_scores[j] -= (total_switches[j] * COMPLEXITY_PENALTY)
            
        top_indices = np.argsort(valid_scores)[-top_k:][::-1]
        top_indices = [x for x in top_indices if valid_scores[x] > -1e10]
        
        parent_candidates[sample_ids[i]] = [(sample_ids[x], valid_scores[x]) for x in top_indices]
        
        if len(top_indices) < 1:
            relationships.append({'Sample': sample_ids[i], 'Generation': 'F1', 'Parent1': None, 'Parent2': None})
            trio_scores_map[sample_ids[i]] = -1e9
            continue
            
        # Form Trios
        pairs = [(p1, p2) for p1 in top_indices for p2 in top_indices if p1 != p2]
        if not pairs: pairs = [(top_indices[0], top_indices[0])]
        
        best_trio = None
        best_trio_score = -np.inf
        
        # Child Data per contig
        child_data_per_contig = []
        for c in range(num_contigs):
            child_data_per_contig.append({
                'alleles': contig_caches[c]['allele_grid'][i],
                'ids': contig_caches[c]['id_grid'][i],
                'sw': contig_caches[c]['sw_costs'],
                'st': contig_caches[c]['st_costs']
            })
            
        # Score Pairs
        for p1, p2 in pairs:
            trio_ll = 0.0
            error_pen = -math.log(1e-2)
            
            for c in range(num_contigs):
                p1_all = contig_caches[c]['allele_grid'][p1]
                p2_all = contig_caches[c]['allele_grid'][p2]
                dat = child_data_per_contig[c]
                
                score = run_trio_phase_aware_hmm(
                    dat['alleles'], dat['ids'], p1_all, p2_all,
                    dat['sw'], dat['st'], error_pen, 50.0
                )
                trio_ll += score
                
            final = trio_ll - (total_switches[p1] + total_switches[p2]) * COMPLEXITY_PENALTY
            
            if final > best_trio_score:
                best_trio_score = final
                best_trio = (p1, p2)
        
        trio_scores_map[sample_ids[i]] = best_trio_score
        
        if best_trio:
            p1n, p2n = sample_ids[best_trio[0]], sample_ids[best_trio[1]]
            relationships.append({'Sample': sample_ids[i], 'Generation': 'Unknown', 'Parent1': p1n, 'Parent2': p2n})
        else:
            relationships.append({'Sample': sample_ids[i], 'Generation': 'F1', 'Parent1': None, 'Parent2': None})

    rel_df = pd.DataFrame(relationships)
    
    # Generation Logic
    name_to_gen = {row['Sample']: 'F1' for _, row in rel_df.iterrows() if pd.isna(row['Parent1'])}
    for _ in range(10): 
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

    # Return object with cutoff method
    res = PedigreeResult(sample_ids, rel_df, parent_candidates, None, [], None, None, trio_scores_map, global_total_bins)
    
    # Run automatic cutoff internally
    res.perform_automatic_cutoff()
    
    return res

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
    if founder_block is None: raise ValueError("Founder Block is now required.")
    if sample_ids is None: sample_ids = [f"S_{i}" for i in range(len(block_painting))]

    contig_input = [{'painting': block_painting, 'founder_block': founder_block}]
    result = infer_pedigree_multi_contig(contig_input, sample_ids, top_k=20)
    
    result.relationships.to_csv(f"{output_prefix}.ped", index=False)
    draw_pedigree_tree(result.relationships, output_file=f"{output_prefix}_tree.png")
    
    return result