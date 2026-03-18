"""
pedigree_inference.py

Pedigree inference using tolerance paintings from paint_samples.py.

- Uses SampleTolerancePainting/SampleConsensusPainting from paint_samples.py
- When scoring parent-child relationships, considers ALL consensus paintings
- Score = max over all (parent_consensus, child_consensus) combinations
- This handles uncertainty in founder assignment properly

The 16-state HMM and free switches in homozygous regions are preserved.
"""
import thread_config

import numpy as np
import pandas as pd
import warnings
import math
import os
from itertools import combinations, product
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
# SHARED DATA FOR POOL WORKERS (avoids pickling large objects per task)
# =============================================================================

_PEDIGREE_SHARED = {}

def _init_pedigree_shared(shared_dict):
    """Pool initializer: store shared data in worker's global scope."""
    global _PEDIGREE_SHARED
    _PEDIGREE_SHARED.clear()
    _PEDIGREE_SHARED.update(shared_dict)

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
        self.relationships['Generation'] = 'Unknown'
        is_root = self.relationships['Parent1'].isna()
        self.relationships.loc[is_root, 'Generation'] = 'F1'
        name_to_gen = dict(zip(self.relationships['Sample'], self.relationships['Generation']))
        for _ in range(10):
            updates = 0
            for idx, row in self.relationships.iterrows():
                if row['Generation'] != 'Unknown': continue
                p1 = row['Parent1']
                p2 = row['Parent2']
                if p1 in name_to_gen and p2 in name_to_gen:
                    g1_str = name_to_gen[p1]
                    g2_str = name_to_gen[p2]
                    if g1_str != 'Unknown' and g2_str != 'Unknown':
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

    def perform_automatic_cutoff(self, force_clusters=2, sigma_threshold=5.0):
        if self.total_bins == 0: return
        scores = []
        indices = []
        for i, s in enumerate(self.samples):
            raw = self.trio_scores.get(s, -1e9)
            norm_score = raw / self.total_bins
            scores.append(norm_score)
            indices.append(i)
        scores_arr = np.array(scores).reshape(-1, 1)
        try:
            kmeans = KMeans(n_clusters=force_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(scores_arr)
            centers = kmeans.cluster_centers_.flatten()
            good_cluster_idx = np.argmax(centers)
            good_scores = scores_arr[labels == good_cluster_idx]
            if len(good_scores) < 2:
                cutoff = np.mean(centers)
                print(f"[Auto-Cutoff] Not enough good samples for stats. Using midpoint: {cutoff:.4f}")
            else:
                mean_good = np.mean(good_scores)
                std_good = np.std(good_scores)
                cutoff = mean_good - (sigma_threshold * std_good)
                bad_center = centers[1-good_cluster_idx]
                if cutoff < bad_center:
                    cutoff = (mean_good + bad_center) / 2
                    print("[Auto-Cutoff] Sigma cutoff too wide, reverting to midpoint.")
            print(f"\n[Auto-Cutoff] Centers: {centers}")
            print(f"[Auto-Cutoff] Good Cluster Stats: Mean={np.mean(good_scores):.4f}, Std={np.std(good_scores):.4f}")
            print(f"[Auto-Cutoff] Calculated Threshold (Mean - {sigma_threshold}*Std): {cutoff:.4f}")
            updates = 0
            for i, score in zip(indices, scores):
                if score < cutoff:
                    self.relationships.at[i, 'Parent1'] = None
                    self.relationships.at[i, 'Parent2'] = None
                    updates += 1
            print(f"[Auto-Cutoff] Reclassified {updates} samples as Founders/F1 (Score < {cutoff:.4f}).")
            self._recalculate_generations()
        except Exception as e:
            print(f"[Auto-Cutoff] Failed: {e}")

    def resolve_cycles(self, verbose=True):
        G = nx.DiGraph()
        for _, row in self.relationships.iterrows():
            child = row['Sample']
            G.add_node(child)
            if pd.notna(row['Parent1']):
                G.add_edge(row['Parent1'], child)
            if pd.notna(row['Parent2']):
                G.add_edge(row['Parent2'], child)
        if nx.is_directed_acyclic_graph(G):
            if verbose:
                print("[Cycle Resolution] Pedigree is already acyclic.")
            return
        trio_scores_norm = {}
        if self.total_bins > 0:
            for s in self.samples:
                raw = self.trio_scores.get(s, 0)
                trio_scores_norm[s] = raw / self.total_bins
        iteration = 0
        max_iterations = 20
        while iteration < max_iterations:
            iteration += 1
            G = nx.DiGraph()
            for _, row in self.relationships.iterrows():
                child = row['Sample']
                G.add_node(child)
                if pd.notna(row['Parent1']):
                    G.add_edge(row['Parent1'], child)
                if pd.notna(row['Parent2']):
                    G.add_edge(row['Parent2'], child)
            if nx.is_directed_acyclic_graph(G):
                if verbose:
                    print(f"[Cycle Resolution] Pedigree is acyclic after {iteration-1} fix(es).")
                break
            cycles = list(nx.simple_cycles(G))
            if verbose:
                print(f"[Cycle Resolution] Iteration {iteration}: Found {len(cycles)} cycle(s)")
            samples_in_cycles = set()
            for cycle in cycles:
                samples_in_cycles.update(cycle)
            candidates_to_fix = []
            for sample in samples_in_cycles:
                row = self.relationships[self.relationships['Sample'] == sample]
                if len(row) == 0:
                    continue
                row = row.iloc[0]
                has_parents = pd.notna(row['Parent1']) or pd.notna(row['Parent2'])
                n_children = G.out_degree(sample)
                if has_parents and n_children > 0:
                    score = trio_scores_norm.get(sample, 0.0)
                    candidates_to_fix.append({
                        'sample': sample,
                        'score': score,
                        'n_children': n_children,
                        'parents': (row['Parent1'], row['Parent2'])
                    })
            if not candidates_to_fix:
                if verbose:
                    print(f"[Cycle Resolution] No candidates found to fix")
                break
            candidates_to_fix.sort(key=lambda x: (-x['n_children'], x['score']))
            to_fix = candidates_to_fix[0]
            sample_to_fix = to_fix['sample']
            if verbose:
                print(f"[Cycle Resolution] Fixing {sample_to_fix}: "
                      f"n_children={to_fix['n_children']}, score={to_fix['score']:.4f}, "
                      f"removing parents {to_fix['parents']}")
            idx = self.relationships[self.relationships['Sample'] == sample_to_fix].index[0]
            self.relationships.at[idx, 'Parent1'] = None
            self.relationships.at[idx, 'Parent2'] = None
            self.relationships.at[idx, 'Generation'] = 'F1'
        self._recalculate_generations()

# =============================================================================
# 2. DISCRETIZATION & ALLELE CONVERSION FOR TOLERANCE PAINTINGS
# =============================================================================

def discretize_consensus_with_uncertainty(consensus_painting, start_pos, end_pos, snps_per_bin=100):
    total_len = end_pos - start_pos
    approx_bp_per_bin = snps_per_bin * 100 
    num_bins = int(total_len / approx_bp_per_bin)
    if num_bins < 100: num_bins = 100 
    bin_edges = np.linspace(start_pos, end_pos, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    id_grid = np.zeros((num_bins, 2), dtype=np.int32) - 1
    potential_hom_mask = np.ones(num_bins, dtype=np.bool_)
    cons_chunks = consensus_painting.chunks
    if not cons_chunks:
        return id_grid, potential_hom_mask, bin_centers
    if consensus_painting.representative_path is not None:
        rep_chunks = consensus_painting.representative_path.chunks
    else:
        rep_chunks = None
    c_ends = np.array([c.end for c in cons_chunks])
    c_starts = np.array([c.start for c in cons_chunks])
    chunk_indices = np.searchsorted(c_ends, bin_centers)
    chunk_indices = np.clip(chunk_indices, 0, len(cons_chunks) - 1)
    for b in range(num_bins):
        cidx = chunk_indices[b]
        chunk = cons_chunks[cidx]
        if bin_centers[b] < chunk.start or bin_centers[b] >= chunk.end:
            potential_hom_mask[b] = True
            continue
        intersection = chunk.possible_hap1 & chunk.possible_hap2
        potential_hom_mask[b] = len(intersection) > 0
    if rep_chunks:
        rep_ends = np.array([c.end for c in rep_chunks])
        rep_h1 = np.array([c.hap1 for c in rep_chunks])
        rep_h2 = np.array([c.hap2 for c in rep_chunks])
        rep_starts = np.array([c.start for c in rep_chunks])
        rep_indices = np.searchsorted(rep_ends, bin_centers)
        rep_indices = np.clip(rep_indices, 0, len(rep_chunks) - 1)
        valid_mask = bin_centers >= rep_starts[rep_indices]
        id_grid[:, 0] = np.where(valid_mask, rep_h1[rep_indices], -1)
        id_grid[:, 1] = np.where(valid_mask, rep_h2[rep_indices], -1)
    return id_grid, potential_hom_mask, bin_centers


def discretize_sample_painting_with_hom_mask(sample_painting, start_pos, end_pos, snps_per_bin=100):
    total_len = end_pos - start_pos
    approx_bp_per_bin = snps_per_bin * 100 
    num_bins = int(total_len / approx_bp_per_bin)
    if num_bins < 100: num_bins = 100 
    bin_edges = np.linspace(start_pos, end_pos, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    id_grid = np.zeros((num_bins, 2), dtype=np.int32) - 1
    chunks = sample_painting.chunks
    if not chunks:
        potential_hom_mask = np.ones(num_bins, dtype=np.bool_)
        return id_grid, potential_hom_mask, bin_centers
    c_ends = np.array([c.end for c in chunks])
    c_h1 = np.array([c.hap1 for c in chunks])
    c_h2 = np.array([c.hap2 for c in chunks])
    c_starts = np.array([c.start for c in chunks])
    indices = np.searchsorted(c_ends, bin_centers)
    indices = np.clip(indices, 0, len(chunks) - 1)
    valid_mask = bin_centers >= c_starts[indices]
    id_grid[:, 0] = np.where(valid_mask, c_h1[indices], -1)
    id_grid[:, 1] = np.where(valid_mask, c_h2[indices], -1)
    potential_hom_mask = (id_grid[:, 0] == id_grid[:, 1]) | (id_grid[:, 0] == -1) | (id_grid[:, 1] == -1)
    return id_grid, potential_hom_mask, bin_centers


def discretize_tolerance_paintings(block_tolerance_painting, snps_per_bin=100):
    start_pos = block_tolerance_painting.start_pos
    end_pos = block_tolerance_painting.end_pos
    total_len = end_pos - start_pos
    approx_bp_per_bin = snps_per_bin * 100 
    num_bins = int(total_len / approx_bp_per_bin)
    if num_bins < 100: num_bins = 100 
    bin_edges = np.linspace(start_pos, end_pos, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    num_samples = len(block_tolerance_painting)
    sample_grids = []
    for i in range(num_samples):
        sample_obj = block_tolerance_painting[i]
        grids_for_sample = []
        consensus_list = sample_obj.consensus_list
        if consensus_list:
            for cons in consensus_list:
                id_grid, potential_hom_mask, _ = discretize_consensus_with_uncertainty(
                    cons, start_pos, end_pos, snps_per_bin
                )
                grids_for_sample.append((id_grid, potential_hom_mask, cons.weight))
        if not grids_for_sample and sample_obj.paths:
            id_grid, potential_hom_mask, _ = discretize_sample_painting_with_hom_mask(
                sample_obj.paths[0], start_pos, end_pos, snps_per_bin
            )
            grids_for_sample.append((id_grid, potential_hom_mask, 1.0))
        if not grids_for_sample:
            id_grid = np.zeros((num_bins, 2), dtype=np.int32) - 1
            potential_hom_mask = np.ones(num_bins, dtype=np.bool_)
            grids_for_sample.append((id_grid, potential_hom_mask, 1.0))
        sample_grids.append(grids_for_sample)
    return sample_grids, bin_centers


def convert_id_grid_to_allele_grid(id_grid, bin_centers, founder_block, bin_width_bp=None):
    num_bins = id_grid.shape[0]
    bin_indices = np.searchsorted(founder_block.positions, bin_centers)
    bin_indices = np.clip(bin_indices, 0, len(founder_block.positions) - 1)
    if bin_width_bp is None:
        if len(bin_centers) > 1:
            bin_width_bp = bin_centers[1] - bin_centers[0]
        else:
            bin_width_bp = 10000 
    found_snps_pos = founder_block.positions[bin_indices]
    dist_to_center = np.abs(found_snps_pos - bin_centers)
    valid_snp_mask = dist_to_center <= (bin_width_bp / 2.0)
    hap_keys = sorted(list(founder_block.haplotypes.keys()))
    max_id = max(hap_keys) if hap_keys else 0
    allele_lookup = np.full((max_id + 1, num_bins), -1, dtype=np.int8)
    for fid, h_arr in founder_block.haplotypes.items():
        if h_arr.ndim == 2: raw_alleles = np.argmax(h_arr, axis=1)
        else: raw_alleles = h_arr
        extracted = raw_alleles[bin_indices]
        extracted[~valid_snp_mask] = -1
        allele_lookup[fid, :] = extracted
    allele_grid = np.full_like(id_grid, -1, dtype=np.int8)
    b_indices = np.arange(num_bins)
    for chrom in [0, 1]:
        ids = id_grid[:, chrom]
        valid_mask = (ids != -1)
        safe_ids = ids.copy()
        safe_ids[~valid_mask] = 0
        alleles = allele_lookup[safe_ids, b_indices]
        alleles[~valid_mask] = -1
        allele_grid[:, chrom] = alleles
    return allele_grid


def convert_id_grid_to_allele_grid_multisnp(id_grid, bin_centers, founder_block, 
                                             bin_width_bp=None, max_snps_per_bin=10):
    num_bins = id_grid.shape[0]
    snp_positions = founder_block.positions
    n_snps = len(snp_positions)
    allele_grid = np.full((num_bins, 2, max_snps_per_bin), -1, dtype=np.int8)
    if n_snps == 0:
        return allele_grid
    if bin_width_bp is None:
        if len(bin_centers) > 1:
            bin_width_bp = bin_centers[1] - bin_centers[0]
        else:
            bin_width_bp = 10000
    half_width = bin_width_bp / 2.0
    hap_keys = sorted(list(founder_block.haplotypes.keys()))
    max_id = max(hap_keys) if hap_keys else 0
    founder_alleles = np.full((max_id + 1, n_snps), -1, dtype=np.int8)
    for fid, h_arr in founder_block.haplotypes.items():
        if h_arr.ndim == 2:
            founder_alleles[fid, :] = np.argmax(h_arr, axis=1).astype(np.int8)
        else:
            founder_alleles[fid, :] = h_arr.astype(np.int8)
    bin_starts = bin_centers - half_width
    bin_ends = bin_centers + half_width
    start_indices = np.searchsorted(snp_positions, bin_starts, side='left')
    end_indices = np.searchsorted(snp_positions, bin_ends, side='right')
    for b in range(num_bins):
        s_start = start_indices[b]
        s_end = end_indices[b]
        bin_n_snps = s_end - s_start
        if bin_n_snps == 0:
            continue
        if bin_n_snps <= max_snps_per_bin:
            sampled_indices = list(range(s_start, s_end))
        else:
            step = bin_n_snps / max_snps_per_bin
            sampled_indices = [s_start + int(i * step) for i in range(max_snps_per_bin)]
        f0 = id_grid[b, 0]
        f1 = id_grid[b, 1]
        for k_idx, snp_idx in enumerate(sampled_indices):
            if k_idx >= max_snps_per_bin:
                break
            if f0 >= 0:
                allele_grid[b, 0, k_idx] = founder_alleles[f0, snp_idx]
            if f1 >= 0:
                allele_grid[b, 1, k_idx] = founder_alleles[f1, snp_idx]
    return allele_grid


# =============================================================================
# 2b. SNP-LEVEL DATA STRUCTURES FOR BINNED AGGREGATION
# =============================================================================

def get_snp_level_founder_ids(painting_chunks, snp_positions):
    n_snps = len(snp_positions)
    id_array = np.full((n_snps, 2), -1, dtype=np.int32)
    if not painting_chunks:
        return id_array
    c_ends = np.array([c.end for c in painting_chunks])
    c_h1 = np.array([c.hap1 for c in painting_chunks])
    c_h2 = np.array([c.hap2 for c in painting_chunks])
    c_starts = np.array([c.start for c in painting_chunks])
    indices = np.searchsorted(c_ends, snp_positions)
    indices = np.clip(indices, 0, len(painting_chunks) - 1)
    valid_mask = snp_positions >= c_starts[indices]
    id_array[:, 0] = np.where(valid_mask, c_h1[indices], -1)
    id_array[:, 1] = np.where(valid_mask, c_h2[indices], -1)
    return id_array


def get_snp_level_hom_mask_from_consensus(consensus_painting, snp_positions):
    n_snps = len(snp_positions)
    hom_mask = np.ones(n_snps, dtype=np.bool_)
    cons_chunks = consensus_painting.chunks
    if not cons_chunks:
        return hom_mask
    c_ends = np.array([c.end for c in cons_chunks])
    c_starts = np.array([c.start for c in cons_chunks])
    chunk_indices = np.searchsorted(c_ends, snp_positions)
    chunk_indices = np.clip(chunk_indices, 0, len(cons_chunks) - 1)
    for s in range(n_snps):
        cidx = chunk_indices[s]
        chunk = cons_chunks[cidx]
        if snp_positions[s] < chunk.start or snp_positions[s] >= chunk.end:
            hom_mask[s] = True
            continue
        intersection = chunk.possible_hap1 & chunk.possible_hap2
        hom_mask[s] = len(intersection) > 0
    return hom_mask


def build_founder_allele_lookup(founder_block):
    snp_positions = founder_block.positions
    n_snps = len(snp_positions)
    hap_keys = sorted(list(founder_block.haplotypes.keys()))
    max_id = max(hap_keys) if hap_keys else 0
    allele_lookup = np.full((max_id + 1, n_snps), -1, dtype=np.int8)
    for fid, h_arr in founder_block.haplotypes.items():
        if h_arr.ndim == 2:
            raw_alleles = np.argmax(h_arr, axis=1)
        else:
            raw_alleles = h_arr
        allele_lookup[fid, :] = raw_alleles.astype(np.int8)
    return allele_lookup, snp_positions


def create_bin_structure(snp_positions, snps_per_bin=100):
    n_snps = len(snp_positions)
    if n_snps == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([])
    num_bins = max(1, n_snps // snps_per_bin)
    if num_bins < 100 and n_snps >= 100:
        num_bins = 100
    bin_starts = np.zeros(num_bins, dtype=np.int32)
    bin_ends = np.zeros(num_bins, dtype=np.int32)
    bin_centers = np.zeros(num_bins, dtype=np.float64)
    snps_per = n_snps // num_bins
    remainder = n_snps % num_bins
    start_idx = 0
    for b in range(num_bins):
        extra = 1 if b < remainder else 0
        end_idx = start_idx + snps_per + extra
        bin_starts[b] = start_idx
        bin_ends[b] = end_idx
        if end_idx > start_idx:
            bin_centers[b] = np.mean(snp_positions[start_idx:end_idx])
        start_idx = end_idx
    return bin_starts, bin_ends, bin_centers


def aggregate_hom_mask_to_bins(snp_hom_mask, bin_starts, bin_ends):
    n_bins = len(bin_starts)
    bin_hom_mask = np.zeros(n_bins, dtype=np.bool_)
    for b in range(n_bins):
        bin_hom_mask[b] = np.any(snp_hom_mask[bin_starts[b]:bin_ends[b]])
    return bin_hom_mask


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

# =============================================================================
# 4. HMM KERNELS (8-STATE FILTER & 16-STATE VERIFIER)
# =============================================================================

@njit(fastmath=True)
def run_phase_agnostic_hmm(child_dip_alleles, child_potential_hom_mask, parent_dip_alleles, 
                           switch_costs, stay_costs, error_penalty, phase_penalty,
                           mismatch_penalty=-4.6):
    n_sites = len(child_dip_alleles)
    scores = np.zeros(8)
    BURST_EMISSION = -0.693147 
    for k in range(4, 8):
        scores[k] = -error_penalty
    for i in range(n_sites):
        c0_a, c1_a = child_dip_alleles[i, 0], child_dip_alleles[i, 1]
        p0_a, p1_a = parent_dip_alleles[i, 0], parent_dip_alleles[i, 1]
        def soft_match(child_allele, parent_allele):
            if child_allele == -1 or parent_allele == -1:
                return 0.0
            elif child_allele == parent_allele:
                return 0.0
            else:
                return mismatch_penalty
        e0 = soft_match(c0_a, p0_a)
        e1 = soft_match(c1_a, p0_a)
        e2 = soft_match(c0_a, p1_a)
        e3 = soft_match(c1_a, p1_a)
        emissions = np.array([e0, e1, e2, e3])
        c_recomb = switch_costs[i]
        c_stay = stay_costs[i]
        c_phase = 0.0 if child_potential_hom_mask[i] else -phase_penalty
        prev = scores.copy()
        new_scores = np.zeros(8)
        for k in range(4):
            burst_idx = k + 4
            from_burst = prev[burst_idx] 
            from_normal = prev[k] - error_penalty
            new_scores[burst_idx] = max(from_burst, from_normal) + BURST_EMISSION
        prev_b0 = prev[4]; prev_b1 = prev[5]; prev_b2 = prev[6]; prev_b3 = prev[7]
        src0 = prev[0] + c_stay
        src1 = prev[1] + c_stay + c_phase
        src2 = prev[2] + c_recomb
        new_scores[0] = max(src0, src1, src2, prev_b0) + emissions[0]
        src1 = prev[1] + c_stay
        src0 = prev[0] + c_stay + c_phase
        src3 = prev[3] + c_recomb
        new_scores[1] = max(src1, src0, src3, prev_b1) + emissions[1]
        src2 = prev[2] + c_stay
        src3 = prev[3] + c_stay + c_phase
        src0 = prev[0] + c_recomb
        new_scores[2] = max(src2, src3, src0, prev_b2) + emissions[2]
        src3 = prev[3] + c_stay
        src2 = prev[2] + c_stay + c_phase
        src1 = prev[1] + c_recomb
        new_scores[3] = max(src3, src2, src1, prev_b3) + emissions[3]
        scores = new_scores
    best_final = -np.inf
    for k in range(8):
        if scores[k] > best_final:
            best_final = scores[k]
    return best_final


@njit(fastmath=True)
def run_trio_phase_aware_hmm(child_dip_alleles, child_potential_hom_mask, 
                             p1_dip_alleles, p2_dip_alleles, 
                             switch_costs, stay_costs, error_penalty, phase_penalty,
                             mismatch_penalty=-4.6):
    n_sites = len(child_dip_alleles)
    BURST_EMISSION = -1.386
    scores = np.zeros(16)
    for k in range(8, 16): scores[k] = -error_penalty
    for i in range(n_sites):
        c0, c1 = child_dip_alleles[i, 0], child_dip_alleles[i, 1]
        p1_h0, p1_h1 = p1_dip_alleles[i, 0], p1_dip_alleles[i, 1]
        p2_h0, p2_h1 = p2_dip_alleles[i, 0], p2_dip_alleles[i, 1]
        def soft_match(parent_allele, child_allele):
            if parent_allele == -1 or child_allele == -1:
                return 0.0
            elif parent_allele == child_allele:
                return 0.0
            else:
                return mismatch_penalty
        m_p1h0_c0 = soft_match(p1_h0, c0); m_p1h1_c0 = soft_match(p1_h1, c0)
        m_p1h0_c1 = soft_match(p1_h0, c1); m_p1h1_c1 = soft_match(p1_h1, c1)
        m_p2h0_c0 = soft_match(p2_h0, c0); m_p2h1_c0 = soft_match(p2_h1, c0)
        m_p2h0_c1 = soft_match(p2_h0, c1); m_p2h1_c1 = soft_match(p2_h1, c1)
        e = np.zeros(8)
        e[0] = m_p1h0_c0 + m_p2h0_c1
        e[1] = m_p1h0_c0 + m_p2h1_c1
        e[2] = m_p1h1_c0 + m_p2h0_c1
        e[3] = m_p1h1_c0 + m_p2h1_c1
        e[4] = m_p1h0_c1 + m_p2h0_c0
        e[5] = m_p1h0_c1 + m_p2h1_c0
        e[6] = m_p1h1_c1 + m_p2h0_c0
        e[7] = m_p1h1_c1 + m_p2h1_c0
        c_recomb = switch_costs[i]
        c_stay = stay_costs[i]
        c_phase = 0.0 if child_potential_hom_mask[i] else -phase_penalty
        prev = scores.copy()
        new_scores = np.zeros(16)
        for k in range(8):
            burst_idx = k + 8
            from_burst = prev[burst_idx]
            from_normal = prev[k] - error_penalty
            new_scores[burst_idx] = max(from_burst, from_normal) + BURST_EMISSION
        cc_0 = 2 * c_stay
        cc_1 = c_recomb + c_stay
        cc_2 = 2 * c_recomb
        p0, p1, p2, p3 = prev[0], prev[1], prev[2], prev[3]
        a0 = max(p0+cc_0, p1+cc_1, p2+cc_1, p3+cc_2)
        a1 = max(p0+cc_1, p1+cc_0, p2+cc_2, p3+cc_1)
        a2 = max(p0+cc_1, p1+cc_2, p2+cc_0, p3+cc_1)
        a3 = max(p0+cc_2, p1+cc_1, p2+cc_1, p3+cc_0)
        p4, p5, p6, p7 = prev[4], prev[5], prev[6], prev[7]
        b4 = max(p4+cc_0, p5+cc_1, p6+cc_1, p7+cc_2)
        b5 = max(p4+cc_1, p5+cc_0, p6+cc_2, p7+cc_1)
        b6 = max(p4+cc_1, p5+cc_2, p6+cc_0, p7+cc_1)
        b7 = max(p4+cc_2, p5+cc_1, p6+cc_1, p7+cc_0)
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
    best_final = -np.inf
    for k in range(16):
        if scores[k] > best_final: best_final = scores[k]
    return best_final


# =============================================================================
# 4b. MULTI-SNP HMM KERNELS (k samples per bin)
# =============================================================================

@njit(fastmath=True)
def run_phase_agnostic_hmm_multisnp(child_dip_alleles, child_potential_hom_mask, parent_dip_alleles, 
                                     switch_costs, stay_costs, error_penalty, phase_penalty,
                                     mismatch_penalty=-4.6):
    n_bins = child_dip_alleles.shape[0]
    k_snps = child_dip_alleles.shape[2]
    scores = np.zeros(8)
    BURST_EMISSION_PER_SNP = -0.693147
    for state in range(4, 8):
        scores[state] = -error_penalty
    for i in range(n_bins):
        e0, e1, e2, e3 = 0.0, 0.0, 0.0, 0.0
        valid_snps = 0
        for s in range(k_snps):
            c0_a = child_dip_alleles[i, 0, s]
            c1_a = child_dip_alleles[i, 1, s]
            p0_a = parent_dip_alleles[i, 0, s]
            p1_a = parent_dip_alleles[i, 1, s]
            if c0_a < 0 or c1_a < 0 or p0_a < 0 or p1_a < 0:
                continue
            valid_snps += 1
            if c0_a != p0_a: e0 += mismatch_penalty
            if c1_a != p0_a: e1 += mismatch_penalty
            if c0_a != p1_a: e2 += mismatch_penalty
            if c1_a != p1_a: e3 += mismatch_penalty
        emissions = np.array([e0, e1, e2, e3])
        burst_emission = BURST_EMISSION_PER_SNP * max(valid_snps, 1)
        c_recomb = switch_costs[i]
        c_stay = stay_costs[i]
        c_phase = 0.0 if child_potential_hom_mask[i] else -phase_penalty
        prev = scores.copy()
        new_scores = np.zeros(8)
        for state in range(4):
            burst_idx = state + 4
            from_burst = prev[burst_idx]
            from_normal = prev[state] - error_penalty
            new_scores[burst_idx] = max(from_burst, from_normal) + burst_emission
        prev_b0 = prev[4]; prev_b1 = prev[5]; prev_b2 = prev[6]; prev_b3 = prev[7]
        src0 = prev[0] + c_stay
        src1 = prev[1] + c_stay + c_phase
        src2 = prev[2] + c_recomb
        new_scores[0] = max(src0, src1, src2, prev_b0) + emissions[0]
        src1 = prev[1] + c_stay
        src0 = prev[0] + c_stay + c_phase
        src3 = prev[3] + c_recomb
        new_scores[1] = max(src1, src0, src3, prev_b1) + emissions[1]
        src2 = prev[2] + c_stay
        src3 = prev[3] + c_stay + c_phase
        src0 = prev[0] + c_recomb
        new_scores[2] = max(src2, src3, src0, prev_b2) + emissions[2]
        src3 = prev[3] + c_stay
        src2 = prev[2] + c_stay + c_phase
        src1 = prev[1] + c_recomb
        new_scores[3] = max(src3, src2, src1, prev_b3) + emissions[3]
        scores = new_scores
    best_final = -np.inf
    for state in range(8):
        if scores[state] > best_final:
            best_final = scores[state]
    return best_final


@njit(fastmath=True)
def run_trio_phase_aware_hmm_multisnp(child_dip_alleles, child_potential_hom_mask, 
                                       p1_dip_alleles, p2_dip_alleles, 
                                       switch_costs, stay_costs, error_penalty, phase_penalty,
                                       mismatch_penalty=-4.6):
    n_bins = child_dip_alleles.shape[0]
    k_snps = child_dip_alleles.shape[2]
    BURST_EMISSION_PER_SNP = -1.386
    scores = np.zeros(16)
    for state in range(8, 16):
        scores[state] = -error_penalty
    for i in range(n_bins):
        e = np.zeros(8)
        valid_snps = 0
        for s in range(k_snps):
            c0 = child_dip_alleles[i, 0, s]
            c1 = child_dip_alleles[i, 1, s]
            p1_h0 = p1_dip_alleles[i, 0, s]
            p1_h1 = p1_dip_alleles[i, 1, s]
            p2_h0 = p2_dip_alleles[i, 0, s]
            p2_h1 = p2_dip_alleles[i, 1, s]
            if c0 < 0 or c1 < 0 or p1_h0 < 0 or p1_h1 < 0 or p2_h0 < 0 or p2_h1 < 0:
                continue
            valid_snps += 1
            if c0 != p1_h0: e[0] += mismatch_penalty
            if c1 != p2_h0: e[0] += mismatch_penalty
            if c0 != p1_h0: e[1] += mismatch_penalty
            if c1 != p2_h1: e[1] += mismatch_penalty
            if c0 != p1_h1: e[2] += mismatch_penalty
            if c1 != p2_h0: e[2] += mismatch_penalty
            if c0 != p1_h1: e[3] += mismatch_penalty
            if c1 != p2_h1: e[3] += mismatch_penalty
            if c1 != p1_h0: e[4] += mismatch_penalty
            if c0 != p2_h0: e[4] += mismatch_penalty
            if c1 != p1_h0: e[5] += mismatch_penalty
            if c0 != p2_h1: e[5] += mismatch_penalty
            if c1 != p1_h1: e[6] += mismatch_penalty
            if c0 != p2_h0: e[6] += mismatch_penalty
            if c1 != p1_h1: e[7] += mismatch_penalty
            if c0 != p2_h1: e[7] += mismatch_penalty
        burst_emission = BURST_EMISSION_PER_SNP * max(valid_snps, 1)
        c_recomb = switch_costs[i]
        c_stay = stay_costs[i]
        c_phase = 0.0 if child_potential_hom_mask[i] else -phase_penalty
        prev = scores.copy()
        new_scores = np.zeros(16)
        for state in range(8):
            burst_idx = state + 8
            from_burst = prev[burst_idx]
            from_normal = prev[state] - error_penalty
            new_scores[burst_idx] = max(from_burst, from_normal) + burst_emission
        cc_0 = 2 * c_stay
        cc_1 = c_recomb + c_stay
        cc_2 = 2 * c_recomb
        p0, p1, p2, p3 = prev[0], prev[1], prev[2], prev[3]
        a0 = max(p0+cc_0, p1+cc_1, p2+cc_1, p3+cc_2)
        a1 = max(p0+cc_1, p1+cc_0, p2+cc_2, p3+cc_1)
        a2 = max(p0+cc_1, p1+cc_2, p2+cc_0, p3+cc_1)
        a3 = max(p0+cc_2, p1+cc_1, p2+cc_1, p3+cc_0)
        p4, p5, p6, p7 = prev[4], prev[5], prev[6], prev[7]
        b4 = max(p4+cc_0, p5+cc_1, p6+cc_1, p7+cc_2)
        b5 = max(p4+cc_1, p5+cc_0, p6+cc_2, p7+cc_1)
        b6 = max(p4+cc_1, p5+cc_2, p6+cc_0, p7+cc_1)
        b7 = max(p4+cc_2, p5+cc_1, p6+cc_1, p7+cc_0)
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
    best_final = -np.inf
    for state in range(16):
        if scores[state] > best_final:
            best_final = scores[state]
    return best_final

# =============================================================================
# 5. TOLERANCE-AWARE SCORING FUNCTIONS
# =============================================================================

DEFAULT_MISMATCH_PENALTY = -4.605170  # math.log(0.01)

def score_parent_child_all_consensus_precomputed(child_grids, parent_grids, 
                                                  switch_costs, stay_costs, 
                                                  error_penalty, phase_penalty,
                                                  mismatch_penalty=DEFAULT_MISMATCH_PENALTY):
    best_score = -np.inf
    for child_allele_grid, child_hom_mask, child_weight in child_grids:
        for parent_allele_grid, parent_hom_mask, parent_weight in parent_grids:
            if child_allele_grid.ndim == 3:
                score = run_phase_agnostic_hmm_multisnp(
                    child_allele_grid, child_hom_mask,
                    parent_allele_grid,
                    switch_costs, stay_costs,
                    error_penalty, phase_penalty,
                    mismatch_penalty
                )
            else:
                score = run_phase_agnostic_hmm(
                    child_allele_grid, child_hom_mask,
                    parent_allele_grid,
                    switch_costs, stay_costs,
                    error_penalty, phase_penalty,
                    mismatch_penalty
                )
            if score > best_score:
                best_score = score
    return best_score


def score_trio_all_consensus_precomputed(child_grids, p1_grids, p2_grids,
                                          switch_costs, stay_costs,
                                          error_penalty, phase_penalty,
                                          mismatch_penalty=DEFAULT_MISMATCH_PENALTY):
    best_score = -np.inf
    for child_allele_grid, child_hom_mask, child_weight in child_grids:
        for p1_allele_grid, p1_hom_mask, p1_weight in p1_grids:
            for p2_allele_grid, p2_hom_mask, p2_weight in p2_grids:
                if child_allele_grid.ndim == 3:
                    score = run_trio_phase_aware_hmm_multisnp(
                        child_allele_grid, child_hom_mask,
                        p1_allele_grid, p2_allele_grid,
                        switch_costs, stay_costs,
                        error_penalty, phase_penalty,
                        mismatch_penalty
                    )
                else:
                    score = run_trio_phase_aware_hmm(
                        child_allele_grid, child_hom_mask,
                        p1_allele_grid, p2_allele_grid,
                        switch_costs, stay_costs,
                        error_penalty, phase_penalty,
                        mismatch_penalty
                    )
                if score > best_score:
                    best_score = score
    return best_score


# =============================================================================
# 6. MULTI-CONTIG INFERENCE LOGIC (PARALLELIZED WITH SHARED DATA)
# =============================================================================

def _process_contig_batch(args):
    """
    Worker: process a BATCH of samples for one contig in Phase 1.
    Reads contig data from shared memory, processes only the assigned samples.
    """
    contig_idx, sample_start, sample_end = args
    contig_data = _PEDIGREE_SHARED['contig_data_list'][contig_idx]
    tol_painting = contig_data['tolerance_painting']
    founder_block = contig_data['founder_block']
    snps_per_bin = _PEDIGREE_SHARED['snps_per_bin']
    recomb_rate = _PEDIGREE_SHARED['recomb_rate']
    max_snps_per_bin = _PEDIGREE_SHARED['max_snps_per_bin']
    
    start_pos = tol_painting.start_pos
    end_pos = tol_painting.end_pos
    total_len = end_pos - start_pos
    approx_bp_per_bin = snps_per_bin * 100
    num_bins = int(total_len / approx_bp_per_bin)
    if num_bins < 100: num_bins = 100
    bin_edges = np.linspace(start_pos, end_pos, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bin_width = 10000.0
    if num_bins > 1:
        bin_width = bin_centers[1] - bin_centers[0]
    
    # Discretize and convert only the assigned sample range
    sample_allele_grids = []
    switch_counts = []
    
    for i in range(sample_start, sample_end):
        sample_obj = tol_painting[i]
        grids_for_sample = []
        
        # Discretize this sample's consensus paintings
        consensus_list = sample_obj.consensus_list
        id_grids_for_sample = []
        
        if consensus_list:
            for cons in consensus_list:
                id_grid, hom_mask, _ = discretize_consensus_with_uncertainty(
                    cons, start_pos, end_pos, snps_per_bin
                )
                id_grids_for_sample.append((id_grid, hom_mask, cons.weight))
        
        if not id_grids_for_sample and sample_obj.paths:
            id_grid, hom_mask, _ = discretize_sample_painting_with_hom_mask(
                sample_obj.paths[0], start_pos, end_pos, snps_per_bin
            )
            id_grids_for_sample.append((id_grid, hom_mask, 1.0))
        
        if not id_grids_for_sample:
            id_grid = np.zeros((num_bins, 2), dtype=np.int32) - 1
            hom_mask = np.ones(num_bins, dtype=np.bool_)
            id_grids_for_sample.append((id_grid, hom_mask, 1.0))
        
        # Convert to allele grids
        allele_grids = []
        for id_grid, hom_mask, weight in id_grids_for_sample:
            if max_snps_per_bin > 1:
                allele_grid = convert_id_grid_to_allele_grid_multisnp(
                    id_grid, bin_centers, founder_block, bin_width, max_snps_per_bin
                )
            else:
                allele_grid = convert_id_grid_to_allele_grid(
                    id_grid, bin_centers, founder_block, bin_width
                )
            allele_grids.append((allele_grid, hom_mask, weight))
        sample_allele_grids.append(allele_grids)
        
        # Switch counts from first id_grid
        first_id_grid = id_grids_for_sample[0][0]
        switches = (first_id_grid[:-1, :] != first_id_grid[1:, :]) & \
                  (first_id_grid[:-1, :] != -1) & (first_id_grid[1:, :] != -1)
        switch_counts.append(np.sum(switches))
    
    # Transition costs (same for all samples in this contig)
    dists = np.zeros(num_bins)
    dists[1:] = np.diff(bin_centers)
    theta = np.clip(1.0 - np.exp(-dists * recomb_rate), 1e-15, 0.5)
    sw_costs = np.log(theta)
    st_costs = np.log(1.0 - theta)
    
    return {
        'contig_idx': contig_idx,
        'sample_start': sample_start,
        'sample_end': sample_end,
        'sample_allele_grids': sample_allele_grids,
        'sw_costs': sw_costs,
        'st_costs': st_costs,
        'num_bins': num_bins,
        'switch_counts': np.array(switch_counts),
    }


def _score_contig_pairs(contig_idx):
    """Worker: score all parent-child pairs for one contig. Reads from shared data."""
    cache = _PEDIGREE_SHARED['contig_caches'][contig_idx]
    sample_allele_grids = cache['sample_allele_grids']
    sw_costs = cache['sw_costs']
    st_costs = cache['st_costs']
    error_pen = _PEDIGREE_SHARED['error_pen']
    phase_pen = _PEDIGREE_SHARED['phase_pen']
    mismatch_penalty = _PEDIGREE_SHARED['mismatch_penalty']
    num_samples = _PEDIGREE_SHARED['num_samples']
    
    scores = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(num_samples):
            if i == j:
                continue
            score = score_parent_child_all_consensus_precomputed(
                sample_allele_grids[i],
                sample_allele_grids[j],
                sw_costs, st_costs,
                error_pen, phase_pen,
                mismatch_penalty
            )
            scores[i, j] = score
    return scores


def _score_pairs_by_children(child_indices):
    """
    Worker: score a batch of children against ALL parents across ALL contigs.
    
    Parallelizes by child rows instead of by contig, giving many more tasks
    (e.g. 32-64 batches of 5-10 children) to fill all available cores.
    """
    contig_caches = _PEDIGREE_SHARED['contig_caches']
    error_pen = _PEDIGREE_SHARED['error_pen']
    phase_pen = _PEDIGREE_SHARED['phase_pen']
    mismatch_penalty = _PEDIGREE_SHARED['mismatch_penalty']
    num_samples = _PEDIGREE_SHARED['num_samples']
    
    n_children = len(child_indices)
    scores = np.zeros((n_children, num_samples))
    
    for ci_local, ci in enumerate(child_indices):
        for j in range(num_samples):
            if ci == j:
                continue
            total = 0.0
            for cache in contig_caches:
                s = score_parent_child_all_consensus_precomputed(
                    cache['sample_allele_grids'][ci],
                    cache['sample_allele_grids'][j],
                    cache['sw_costs'], cache['st_costs'],
                    error_pen, phase_pen,
                    mismatch_penalty
                )
                total += s
            scores[ci_local, j] = total
    
    return child_indices, scores


def _score_trios_batch(batch_sample_args):
    """Worker: score trios for a batch of samples. Reads from shared data."""
    contig_caches = _PEDIGREE_SHARED['contig_caches']
    total_switches = _PEDIGREE_SHARED['total_switches']
    error_pen = _PEDIGREE_SHARED['error_pen']
    phase_pen = _PEDIGREE_SHARED['phase_pen']
    mismatch_penalty = _PEDIGREE_SHARED['mismatch_penalty']
    complexity_penalty = _PEDIGREE_SHARED['complexity_penalty']
    
    results = []
    for sample_idx, top_indices in batch_sample_args:
        if len(top_indices) < 1:
            results.append((sample_idx, None, -1e9, []))
            continue
        pairs = [(p1, p2) for p1 in top_indices for p2 in top_indices if p1 != p2]
        if not pairs:
            pairs = [(top_indices[0], top_indices[0])]
        best_trio = None
        best_trio_score = -np.inf
        for p1, p2 in pairs:
            trio_ll = 0.0
            for cache in contig_caches:
                sample_allele_grids = cache['sample_allele_grids']
                sw_costs = cache['sw_costs']
                st_costs = cache['st_costs']
                score = score_trio_all_consensus_precomputed(
                    sample_allele_grids[sample_idx],
                    sample_allele_grids[p1],
                    sample_allele_grids[p2],
                    sw_costs, st_costs,
                    error_pen, phase_pen,
                    mismatch_penalty
                )
                trio_ll += score
            final = trio_ll - (total_switches[p1] + total_switches[p2]) * complexity_penalty
            if final > best_trio_score:
                best_trio_score = final
                best_trio = (p1, p2)
        results.append((sample_idx, best_trio, best_trio_score, top_indices))
    return results


def infer_pedigree_multi_contig_tolerance(contig_data_list, sample_ids, top_k=20,
                                          snps_per_bin=100, recomb_rate=5e-8,
                                          mismatch_penalty=DEFAULT_MISMATCH_PENALTY,
                                          max_snps_per_bin=10,
                                          n_workers=None):
    """
    Multi-contig pedigree inference using tolerance paintings with multi-SNP voting.
    """
    import multiprocessing as mp
    
    num_samples = len(sample_ids)
    num_contigs = len(contig_data_list)
    
    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, 16)
    
    # Phase 1: Discretize and convert allele grids — parallelize across
    # (contig, sample_batch) pairs to use all cores, not just 7.
    # With 7 contigs × ~10 batches each = ~70 tasks across 112 cores.
    samples_per_phase1_batch = max(1, num_samples // max(1, n_workers // num_contigs))
    
    phase1_tasks = []
    for c_idx in range(num_contigs):
        for s_start in range(0, num_samples, samples_per_phase1_batch):
            s_end = min(s_start + samples_per_phase1_batch, num_samples)
            phase1_tasks.append((c_idx, s_start, s_end))
    
    print(f"\n--- Phase 1: Processing {num_contigs} Contigs × {len(phase1_tasks)} tasks "
          f"({n_workers} workers, {max_snps_per_bin} SNPs/bin) ---")
    
    error_pen = -math.log(1e-2)
    phase_pen = 50.0
    
    # Share contig data via initializer — tolerance paintings + founder blocks
    shared_phase1 = {
        'contig_data_list': contig_data_list,
        'snps_per_bin': snps_per_bin,
        'recomb_rate': recomb_rate,
        'max_snps_per_bin': max_snps_per_bin,
    }
    
    ctx = mp.get_context('fork')
    with ctx.Pool(processes=min(n_workers, len(phase1_tasks)),
                  initializer=_init_pedigree_shared,
                  initargs=(shared_phase1,)) as pool:
        phase1_results = list(tqdm(
            pool.imap_unordered(_process_contig_batch, phase1_tasks),
            total=len(phase1_tasks),
            desc="Processing contigs in parallel"
        ))
    
    # Reassemble results into per-contig caches
    # First pass: collect transition costs and num_bins per contig
    contig_meta = {}
    for result in phase1_results:
        c_idx = result['contig_idx']
        if c_idx not in contig_meta:
            contig_meta[c_idx] = {
                'sw_costs': result['sw_costs'],
                'st_costs': result['st_costs'],
                'num_bins': result['num_bins'],
            }
    
    # Second pass: assemble per-sample allele grids in order
    contig_sample_grids = {c_idx: [None] * num_samples for c_idx in range(num_contigs)}
    total_switches = np.zeros(num_samples)
    global_total_bins = 0
    
    for result in phase1_results:
        c_idx = result['contig_idx']
        s_start = result['sample_start']
        s_end = result['sample_end']
        for local_i, global_i in enumerate(range(s_start, s_end)):
            contig_sample_grids[c_idx][global_i] = result['sample_allele_grids'][local_i]
        total_switches[s_start:s_end] += result['switch_counts']
    
    contig_caches = []
    for c_idx in range(num_contigs):
        meta = contig_meta[c_idx]
        contig_caches.append({
            'sample_allele_grids': contig_sample_grids[c_idx],
            'sw_costs': meta['sw_costs'],
            'st_costs': meta['st_costs'],
        })
        global_total_bins += meta['num_bins']
    
    # Phase 1b + Phase 2: Use ONE Pool for both phases to avoid double fork overhead.
    # Set all shared data before Pool creation — workers see it via fork COW.
    COMPLEXITY_PENALTY = 0.0
    
    shared_all = {
        'contig_caches': contig_caches,
        'error_pen': error_pen,
        'phase_pen': phase_pen,
        'mismatch_penalty': mismatch_penalty,
        'num_samples': num_samples,
        # Phase 2 fields (set now so workers have them at fork time)
        'total_switches': total_switches,
        'complexity_penalty': COMPLEXITY_PENALTY,
    }
    
    # Phase 1b: Score all pairs — parallelize by CHILD ROWS (not by contig)
    print(f"\n--- Phase 1b: Scoring Parent-Child Pairs ({n_workers} workers) ---")
    
    children_per_batch = max(1, num_samples // (n_workers * 2))
    child_batches = []
    for start in range(0, num_samples, children_per_batch):
        end = min(start + children_per_batch, num_samples)
        child_batches.append(list(range(start, end)))
    
    total_scores = np.zeros((num_samples, num_samples))
    
    with ctx.Pool(processes=min(n_workers, len(child_batches)),
                  initializer=_init_pedigree_shared,
                  initargs=(shared_all,)) as pool:
        
        # --- Phase 1b ---
        results = list(tqdm(
            pool.imap_unordered(_score_pairs_by_children, child_batches),
            total=len(child_batches),
            desc="Scoring pairs in parallel"
        ))
        
        for child_indices, scores_block in results:
            for ci_local, ci in enumerate(child_indices):
                total_scores[ci, :] = scores_block[ci_local, :]
        
        total_scores[total_scores == -np.inf] = -1e9
        
        cand_mask = np.zeros((num_samples, num_samples), dtype=bool)
        margin = 5
        for i in range(num_samples):
            valid_gen = total_switches <= (total_switches[i] + margin)
            cand_mask[i, :] = valid_gen
            cand_mask[i, i] = False

        # --- Phase 2: Trio Verification (same Pool, no re-fork) ---
        print(f"\n--- Phase 2: Trio Verification (Top {top_k} Pairs, reusing pool) ---")
        
        all_sample_args = []
        for i in range(num_samples):
            valid_scores = total_scores[i].copy()
            valid_scores[~cand_mask[i, :]] = -np.inf
            for j in range(num_samples):
                if valid_scores[j] > -1e9:
                    valid_scores[j] -= (total_switches[j] * COMPLEXITY_PENALTY)
            top_indices = np.argsort(valid_scores)[-top_k:][::-1]
            top_indices = [x for x in top_indices if valid_scores[x] > -1e10]
            all_sample_args.append((i, top_indices))
        
        batch_size = max(1, num_samples // (n_workers * 4))
        batched_args = []
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_sample_args = all_sample_args[batch_start:batch_end]
            batched_args.append(batch_sample_args)
        
        batch_results = list(tqdm(
            pool.imap_unordered(_score_trios_batch, batched_args),
            total=len(batched_args),
            desc="Inferring Trios in parallel"
        ))
    
    trio_results = []
    for batch in batch_results:
        trio_results.extend(batch)
    
    # CRITICAL: Sort by sample_idx so DataFrame rows match self.samples order.
    # imap_unordered returns results in arbitrary order; perform_automatic_cutoff
    # assumes self.relationships.at[i, ...] corresponds to self.samples[i].
    trio_results.sort(key=lambda x: x[0])
    
    # Collect results
    relationships = []
    parent_candidates = {}
    trio_scores_map = {}
    
    for sample_idx, best_trio, best_trio_score, top_indices in trio_results:
        valid_scores = total_scores[sample_idx].copy()
        valid_scores[~cand_mask[sample_idx, :]] = -np.inf
        parent_candidates[sample_ids[sample_idx]] = [
            (sample_ids[x], valid_scores[x]) for x in top_indices
        ]
        trio_scores_map[sample_ids[sample_idx]] = best_trio_score
        if best_trio:
            p1n, p2n = sample_ids[best_trio[0]], sample_ids[best_trio[1]]
            relationships.append({
                'Sample': sample_ids[sample_idx], 
                'Generation': 'Unknown', 
                'Parent1': p1n, 
                'Parent2': p2n
            })
        else:
            relationships.append({
                'Sample': sample_ids[sample_idx], 
                'Generation': 'F1', 
                'Parent1': None, 
                'Parent2': None
            })

    rel_df = pd.DataFrame(relationships)
    
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

    res = PedigreeResult(sample_ids, rel_df, parent_candidates, None, [], None, None, 
                         trio_scores_map, global_total_bins)
    res.perform_automatic_cutoff()
    res.resolve_cycles()
    return res


# =============================================================================
# 7. CYCLE DETECTION AND RESOLUTION
# =============================================================================

def build_pedigree_graph(relationships_df):
    G = nx.DiGraph()
    for _, row in relationships_df.iterrows():
        child = row['Sample']
        G.add_node(child)
        p1 = row.get('Parent1')
        p2 = row.get('Parent2')
        if pd.notna(p1):
            G.add_edge(p1, child)
        if pd.notna(p2):
            G.add_edge(p2, child)
    return G


def detect_cycles(relationships_df):
    G = build_pedigree_graph(relationships_df)
    if nx.is_directed_acyclic_graph(G):
        return []
    return list(nx.simple_cycles(G))


def propagate_generations(relationships_df):
    fixed_df = relationships_df.copy()
    name_to_gen = {}
    for idx, row in fixed_df.iterrows():
        if pd.isna(row['Parent1']) and pd.isna(row['Parent2']):
            fixed_df.at[idx, 'Generation'] = 'F1'
            name_to_gen[row['Sample']] = 'F1'
        else:
            fixed_df.at[idx, 'Generation'] = 'Unknown'
    for _ in range(10):
        changed = False
        for idx, row in fixed_df.iterrows():
            if row['Generation'] != 'Unknown':
                continue
            p1, p2 = row['Parent1'], row['Parent2']
            p1_gen = name_to_gen.get(p1)
            p2_gen = name_to_gen.get(p2)
            if p1_gen and p2_gen:
                try:
                    g1 = int(p1_gen[1:])
                    g2 = int(p2_gen[1:])
                    gen = f"F{max(g1, g2) + 1}"
                    fixed_df.at[idx, 'Generation'] = gen
                    name_to_gen[row['Sample']] = gen
                    changed = True
                except:
                    pass
        if not changed:
            break
    return fixed_df


# =============================================================================
# 8. VISUALIZATION & WRAPPER
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


def run_pedigree_inference_tolerance(tolerance_painting, sample_ids=None, snps_per_bin=100, 
                                     founder_block=None, recomb_rate=5e-8,
                                     output_prefix="pedigree",
                                     mismatch_penalty=DEFAULT_MISMATCH_PENALTY,
                                     n_workers=None):
    if founder_block is None: 
        raise ValueError("Founder Block is required.")
    if sample_ids is None: 
        sample_ids = [f"S_{i}" for i in range(len(tolerance_painting))]
    contig_input = [{'tolerance_painting': tolerance_painting, 'founder_block': founder_block}]
    result = infer_pedigree_multi_contig_tolerance(contig_input, sample_ids, top_k=20,
                                                    snps_per_bin=snps_per_bin, recomb_rate=recomb_rate,
                                                    mismatch_penalty=mismatch_penalty,
                                                    n_workers=n_workers)
    result.relationships.to_csv(f"{output_prefix}.ped", index=False)
    draw_pedigree_tree(result.relationships, output_file=f"{output_prefix}_tree.png")
    return result