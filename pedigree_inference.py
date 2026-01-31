"""
pedigree_inference.py

Pedigree inference using tolerance paintings from paint_samples.py.

- Uses SampleTolerancePainting/SampleConsensusPainting from paint_samples.py
- When scoring parent-child relationships, considers ALL consensus paintings
- Score = max over all (parent_consensus, child_consensus) combinations
- This handles uncertainty in founder assignment properly

The 16-state HMM and free switches in homozygous regions are preserved.
"""

import numpy as np
import pandas as pd
import warnings
import math
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
        is_root = self.relationships['Parent1'].isna()
        self.relationships.loc[is_root, 'Generation'] = 'F1'
        
        # 3. Create Lookup
        name_to_gen = dict(zip(self.relationships['Sample'], self.relationships['Generation']))
        
        # 4. Propagate
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

    def perform_automatic_cutoff(self, force_clusters=2, sigma_threshold=4.0):
        """
        Uses Sigma Clipping on Score-Per-Bin to distinguish True Trios (F2/F3) from
        False Trios (F1s). 
        Calculates stats of the "Good" cluster and rejects outliers.
        """
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
            # 1. Initial Split with K-Means
            kmeans = KMeans(n_clusters=force_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(scores_arr)
            centers = kmeans.cluster_centers_.flatten()
            
            # Identify the "Good" cluster (Higher score, closer to 0)
            good_cluster_idx = np.argmax(centers)
            
            # 2. Refine using Statistics of the Good Cluster
            good_scores = scores_arr[labels == good_cluster_idx]
            
            if len(good_scores) < 2:
                cutoff = np.mean(centers)
                print(f"[Auto-Cutoff] Not enough good samples for stats. Using midpoint: {cutoff:.4f}")
            else:
                mean_good = np.mean(good_scores)
                std_good = np.std(good_scores)
                
                # Define cutoff as X standard deviations below the mean
                cutoff = mean_good - (sigma_threshold * std_good)
                
                # Safety Check: cutoff shouldn't be lower than the bad center
                bad_center = centers[1-good_cluster_idx]
                if cutoff < bad_center:
                    cutoff = (mean_good + bad_center) / 2
                    print("[Auto-Cutoff] Sigma cutoff too wide, reverting to midpoint.")
            
            print(f"\n[Auto-Cutoff] Centers: {centers}")
            print(f"[Auto-Cutoff] Good Cluster Stats: Mean={np.mean(good_scores):.4f}, Std={np.std(good_scores):.4f}")
            print(f"[Auto-Cutoff] Calculated Threshold (Mean - {sigma_threshold}*Std): {cutoff:.4f}")
            
            updates = 0
            for i, score in zip(indices, scores):
                # 3. Apply Threshold
                if score < cutoff:
                    self.relationships.at[i, 'Parent1'] = None
                    self.relationships.at[i, 'Parent2'] = None
                    updates += 1
                    
            print(f"[Auto-Cutoff] Reclassified {updates} samples as Founders/F1 (Score < {cutoff:.4f}).")
            
            # 4. Fix Generation Labels
            self._recalculate_generations()
            
        except Exception as e:
            print(f"[Auto-Cutoff] Failed: {e}")

    def resolve_cycles(self, verbose=True):
        """
        Detect and resolve cycles in the pedigree.
        
        Strategy: Samples that have many children but also have parents assigned
        are likely mis-classified. We remove parent assignments from the sample
        with the most children and worst trio score.
        
        Args:
            verbose: Print progress information
        """
        # Build directed graph (parent -> child)
        G = nx.DiGraph()
        for _, row in self.relationships.iterrows():
            child = row['Sample']
            G.add_node(child)
            if pd.notna(row['Parent1']):
                G.add_edge(row['Parent1'], child)
            if pd.notna(row['Parent2']):
                G.add_edge(row['Parent2'], child)
        
        # Check if already valid
        if nx.is_directed_acyclic_graph(G):
            if verbose:
                print("[Cycle Resolution] Pedigree is already acyclic.")
            return
        
        # Get normalized trio scores
        trio_scores_norm = {}
        if self.total_bins > 0:
            for s in self.samples:
                raw = self.trio_scores.get(s, 0)
                trio_scores_norm[s] = raw / self.total_bins
        
        iteration = 0
        max_iterations = 20
        
        while iteration < max_iterations:
            iteration += 1
            
            # Rebuild graph
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
            
            # Find cycles
            cycles = list(nx.simple_cycles(G))
            if verbose:
                print(f"[Cycle Resolution] Iteration {iteration}: Found {len(cycles)} cycle(s)")
            
            # Collect all samples involved in cycles
            samples_in_cycles = set()
            for cycle in cycles:
                samples_in_cycles.update(cycle)
            
            # Find candidates: samples that are both parent AND child
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
            
            # Sort by: most children, then worst score
            candidates_to_fix.sort(key=lambda x: (-x['n_children'], x['score']))
            
            # Fix the top candidate
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
        
        # Re-propagate generations
        self._recalculate_generations()

# =============================================================================
# 2. DISCRETIZATION & ALLELE CONVERSION FOR TOLERANCE PAINTINGS
# =============================================================================

def discretize_consensus_with_uncertainty(consensus_painting, start_pos, end_pos, snps_per_bin=50):
    """
    Discretize a SampleConsensusPainting, preserving uncertainty information.
    
    Computes a potential_hom_mask based on whether the intersection of
    possible_hap1 and possible_hap2 is non-empty at each bin.
    
    Args:
        consensus_painting: SampleConsensusPainting with chunks containing uncertainty sets
        start_pos: Start position of the region
        end_pos: End position of the region
        snps_per_bin: Approximate SNPs per bin for grid resolution
    
    Returns:
        id_grid: (num_bins, 2) array of representative founder IDs
        potential_hom_mask: (num_bins,) boolean array - True if homozygosity is possible
        bin_centers: array of bin center positions
    """
    total_len = end_pos - start_pos
    approx_bp_per_bin = snps_per_bin * 100 
    num_bins = int(total_len / approx_bp_per_bin)
    if num_bins < 100: num_bins = 100 
    
    bin_edges = np.linspace(start_pos, end_pos, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    id_grid = np.zeros((num_bins, 2), dtype=np.int32) - 1
    potential_hom_mask = np.ones(num_bins, dtype=np.bool_)  # Default: assume potentially hom (conservative)
    
    # Get the consensus chunks (with uncertainty sets)
    cons_chunks = consensus_painting.chunks
    if not cons_chunks:
        return id_grid, potential_hom_mask, bin_centers
    
    # Get representative path for concrete IDs
    if consensus_painting.representative_path is not None:
        rep_chunks = consensus_painting.representative_path.chunks
    else:
        rep_chunks = None
    
    # Build lookup for consensus chunks (uncertainty sets)
    c_ends = np.array([c.end for c in cons_chunks])
    c_starts = np.array([c.start for c in cons_chunks])
    
    # For each bin, find the covering consensus chunk
    chunk_indices = np.searchsorted(c_ends, bin_centers)
    chunk_indices = np.clip(chunk_indices, 0, len(cons_chunks) - 1)
    
    for b in range(num_bins):
        cidx = chunk_indices[b]
        chunk = cons_chunks[cidx]
        
        # Check if bin is actually within this chunk
        if bin_centers[b] < chunk.start or bin_centers[b] >= chunk.end:
            # Bin not covered - treat as uncertain (potentially hom)
            potential_hom_mask[b] = True
            continue
        
        # Check if intersection of possible founders is non-empty
        # possible_hap1 and possible_hap2 are frozensets
        intersection = chunk.possible_hap1 & chunk.possible_hap2
        potential_hom_mask[b] = len(intersection) > 0
    
    # Fill in concrete IDs from representative path
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


def discretize_sample_painting_with_hom_mask(sample_painting, start_pos, end_pos, snps_per_bin=50):
    """
    Discretize a SamplePainting (no uncertainty info).
    
    For a concrete painting without uncertainty sets, homozygosity is determined
    by whether the two founder IDs are identical.
    
    Args:
        sample_painting: SamplePainting object with chunks
        start_pos: Start position of the region
        end_pos: End position of the region
        snps_per_bin: Approximate SNPs per bin for grid resolution
    
    Returns:
        id_grid: (num_bins, 2) array of founder IDs
        potential_hom_mask: (num_bins,) boolean array - True if hap1 == hap2 or either is -1
        bin_centers: array of bin center positions
    """
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
    
    # For concrete paintings: hom if IDs match or either is -1
    potential_hom_mask = (id_grid[:, 0] == id_grid[:, 1]) | (id_grid[:, 0] == -1) | (id_grid[:, 1] == -1)
    
    return id_grid, potential_hom_mask, bin_centers


def discretize_tolerance_paintings(block_tolerance_painting, snps_per_bin=100):
    """
    Converts BlockTolerancePainting into discretized grids for ALL consensus paintings.
    
    For each sample, returns a list of (id_grid, potential_hom_mask, weight) tuples,
    one per consensus. The potential_hom_mask is computed from set intersection
    of possible founders.
    
    Args:
        block_tolerance_painting: BlockTolerancePainting from paint_samples
        snps_per_bin: Grid resolution
        
    Returns:
        sample_grids: List of lists, sample_grids[i] = [(id_grid, potential_hom_mask, weight), ...]
        bin_centers: Array of bin center positions (same for all)
    """
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
        
        # Get consensus paintings
        consensus_list = sample_obj.consensus_list
        
        if consensus_list:
            for cons in consensus_list:
                id_grid, potential_hom_mask, _ = discretize_consensus_with_uncertainty(
                    cons, start_pos, end_pos, snps_per_bin
                )
                grids_for_sample.append((id_grid, potential_hom_mask, cons.weight))
        
        # Fallback: if no consensus, use the best path
        if not grids_for_sample and sample_obj.paths:
            id_grid, potential_hom_mask, _ = discretize_sample_painting_with_hom_mask(
                sample_obj.paths[0], start_pos, end_pos, snps_per_bin
            )
            grids_for_sample.append((id_grid, potential_hom_mask, 1.0))
        
        # Ultimate fallback: empty grid (all potentially hom)
        if not grids_for_sample:
            id_grid = np.zeros((num_bins, 2), dtype=np.int32) - 1
            potential_hom_mask = np.ones(num_bins, dtype=np.bool_)
            grids_for_sample.append((id_grid, potential_hom_mask, 1.0))
        
        sample_grids.append(grids_for_sample)
    
    return sample_grids, bin_centers


def convert_id_grid_to_allele_grid(id_grid, bin_centers, founder_block, bin_width_bp=None):
    """
    Translates Founder IDs to Alleles (0/1/missing).
    Strictly handles empty bins by setting alleles to -1 if no SNP is nearby.
    
    Args:
        id_grid: (num_bins, 2) array of founder IDs
        bin_centers: array of bin center positions
        founder_block: BlockResult with positions and haplotypes
        bin_width_bp: Width of each bin in bp
        
    Returns:
        allele_grid: (num_bins, 2) array of alleles (0, 1, or -1)
    """
    num_bins = id_grid.shape[0]
    
    # 1. Find nearest SNP for every bin
    bin_indices = np.searchsorted(founder_block.positions, bin_centers)
    bin_indices = np.clip(bin_indices, 0, len(founder_block.positions) - 1)
    
    # 2. Check if the SNP found is actually CLOSE to the bin center
    if bin_width_bp is None:
        if len(bin_centers) > 1:
            bin_width_bp = bin_centers[1] - bin_centers[0]
        else:
            bin_width_bp = 10000 
            
    # Get positions of the found SNPs
    found_snps_pos = founder_block.positions[bin_indices]
    
    # A bin is valid if the SNP is within the bin's radius (half width)
    dist_to_center = np.abs(found_snps_pos - bin_centers)
    valid_snp_mask = dist_to_center <= (bin_width_bp / 2.0)
    
    # 3. Lookup Alleles
    hap_keys = sorted(list(founder_block.haplotypes.keys()))
    max_id = max(hap_keys) if hap_keys else 0
    allele_lookup = np.full((max_id + 1, num_bins), -1, dtype=np.int8)
    
    for fid, h_arr in founder_block.haplotypes.items():
        if h_arr.ndim == 2: raw_alleles = np.argmax(h_arr, axis=1)
        else: raw_alleles = h_arr
        
        # Get alleles for the selected indices
        extracted = raw_alleles[bin_indices]
        
        # MASK OUT bins where the SNP was too far away (Empty Bin)
        extracted[~valid_snp_mask] = -1
        
        allele_lookup[fid, :] = extracted
        
    # 4. Fill Grid
    allele_grid = np.full_like(id_grid, -1, dtype=np.int8)
    b_indices = np.arange(num_bins)
    
    for chrom in [0, 1]:
        ids = id_grid[:, chrom]
        valid_mask = (ids != -1)
        
        safe_ids = ids.copy()
        safe_ids[~valid_mask] = 0
        
        # Look up
        alleles = allele_lookup[safe_ids, b_indices]
        
        # Restore missing IDs
        alleles[~valid_mask] = -1
        
        allele_grid[:, chrom] = alleles
        
    return allele_grid


def convert_id_grid_to_allele_grid_multisnp(id_grid, bin_centers, founder_block, 
                                             bin_width_bp=None, max_snps_per_bin=10):
    """
    Translates Founder IDs to Alleles, storing multiple sampled SNPs per bin.
    
    For each bin, samples up to max_snps_per_bin evenly-spaced SNPs and stores
    all their alleles. This allows the HMM to count actual mismatches across
    multiple SNPs per bin for more robust scoring.
    
    Args:
        id_grid: (num_bins, 2) array of founder IDs
        bin_centers: array of bin center positions  
        founder_block: BlockResult with positions and haplotypes
        bin_width_bp: Width of each bin in bp
        max_snps_per_bin: Number of SNPs to sample per bin (default 10)
        
    Returns:
        allele_grid: (num_bins, 2, max_snps_per_bin) array of alleles (0, 1, or -1)
    """
    num_bins = id_grid.shape[0]
    snp_positions = founder_block.positions
    n_snps = len(snp_positions)
    
    # Initialize with -1 (missing)
    allele_grid = np.full((num_bins, 2, max_snps_per_bin), -1, dtype=np.int8)
    
    if n_snps == 0:
        return allele_grid
    
    # Calculate bin width
    if bin_width_bp is None:
        if len(bin_centers) > 1:
            bin_width_bp = bin_centers[1] - bin_centers[0]
        else:
            bin_width_bp = 10000
    
    half_width = bin_width_bp / 2.0
    
    # Build founder allele lookup (founder_id -> alleles at all SNPs)
    hap_keys = sorted(list(founder_block.haplotypes.keys()))
    max_id = max(hap_keys) if hap_keys else 0
    founder_alleles = np.full((max_id + 1, n_snps), -1, dtype=np.int8)
    
    for fid, h_arr in founder_block.haplotypes.items():
        if h_arr.ndim == 2:
            founder_alleles[fid, :] = np.argmax(h_arr, axis=1).astype(np.int8)
        else:
            founder_alleles[fid, :] = h_arr.astype(np.int8)
    
    # For each bin, find SNPs within the bin boundaries
    bin_starts = bin_centers - half_width
    bin_ends = bin_centers + half_width
    
    # Use searchsorted to find SNP ranges for each bin
    start_indices = np.searchsorted(snp_positions, bin_starts, side='left')
    end_indices = np.searchsorted(snp_positions, bin_ends, side='right')
    
    for b in range(num_bins):
        s_start = start_indices[b]
        s_end = end_indices[b]
        bin_n_snps = s_end - s_start
        
        if bin_n_snps == 0:
            # No SNPs in this bin
            continue
        
        # Determine which SNPs to sample
        if bin_n_snps <= max_snps_per_bin:
            # Use all SNPs in bin, pad rest with -1
            sampled_indices = list(range(s_start, s_end))
        else:
            # Sample evenly-spaced SNPs
            step = bin_n_snps / max_snps_per_bin
            sampled_indices = [s_start + int(i * step) for i in range(max_snps_per_bin)]
        
        # Get founder IDs for this bin
        f0 = id_grid[b, 0]
        f1 = id_grid[b, 1]
        
        # Store alleles at each sampled SNP position
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
    """
    Convert painting chunks to SNP-level founder IDs.
    
    Args:
        painting_chunks: List of chunks with .start, .end, .hap1, .hap2
        snp_positions: Array of SNP positions
        
    Returns:
        id_array: (n_snps, 2) array of founder IDs at each SNP
    """
    n_snps = len(snp_positions)
    id_array = np.full((n_snps, 2), -1, dtype=np.int32)
    
    if not painting_chunks:
        return id_array
    
    c_ends = np.array([c.end for c in painting_chunks])
    c_h1 = np.array([c.hap1 for c in painting_chunks])
    c_h2 = np.array([c.hap2 for c in painting_chunks])
    c_starts = np.array([c.start for c in painting_chunks])
    
    # Find chunk covering each SNP
    indices = np.searchsorted(c_ends, snp_positions)
    indices = np.clip(indices, 0, len(painting_chunks) - 1)
    valid_mask = snp_positions >= c_starts[indices]
    
    id_array[:, 0] = np.where(valid_mask, c_h1[indices], -1)
    id_array[:, 1] = np.where(valid_mask, c_h2[indices], -1)
    
    return id_array


def get_snp_level_hom_mask_from_consensus(consensus_painting, snp_positions):
    """
    Get SNP-level homozygosity mask from consensus painting.
    
    Args:
        consensus_painting: SampleConsensusPainting with uncertainty sets
        snp_positions: Array of SNP positions
        
    Returns:
        hom_mask: (n_snps,) boolean array - True where homozygosity is possible
    """
    n_snps = len(snp_positions)
    hom_mask = np.ones(n_snps, dtype=np.bool_)  # Default: assume potentially hom
    
    cons_chunks = consensus_painting.chunks
    if not cons_chunks:
        return hom_mask
    
    c_ends = np.array([c.end for c in cons_chunks])
    c_starts = np.array([c.start for c in cons_chunks])
    
    # Find chunk covering each SNP
    chunk_indices = np.searchsorted(c_ends, snp_positions)
    chunk_indices = np.clip(chunk_indices, 0, len(cons_chunks) - 1)
    
    for s in range(n_snps):
        cidx = chunk_indices[s]
        chunk = cons_chunks[cidx]
        
        # Check if SNP is actually within this chunk
        if snp_positions[s] < chunk.start or snp_positions[s] >= chunk.end:
            hom_mask[s] = True
            continue
        
        # Check if intersection of possible founders is non-empty
        intersection = chunk.possible_hap1 & chunk.possible_hap2
        hom_mask[s] = len(intersection) > 0
    
    return hom_mask


def build_founder_allele_lookup(founder_block):
    """
    Build a lookup table for founder alleles at each SNP.
    
    Args:
        founder_block: BlockResult with positions and haplotypes
        
    Returns:
        allele_lookup: (max_founder_id+1, n_snps) array of alleles
        snp_positions: Array of SNP positions
    """
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
    """
    Create bin structure mapping SNPs to bins.
    
    Args:
        snp_positions: Array of SNP positions
        snps_per_bin: Target number of SNPs per bin
        
    Returns:
        bin_starts: (n_bins,) array of first SNP index in each bin
        bin_ends: (n_bins,) array of last SNP index + 1 in each bin
        bin_centers: (n_bins,) array of bin center positions
    """
    n_snps = len(snp_positions)
    
    if n_snps == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([])
    
    # Create bins based on actual SNP count
    num_bins = max(1, n_snps // snps_per_bin)
    if num_bins < 100 and n_snps >= 100:
        num_bins = 100
    
    # Distribute SNPs evenly across bins
    bin_starts = np.zeros(num_bins, dtype=np.int32)
    bin_ends = np.zeros(num_bins, dtype=np.int32)
    bin_centers = np.zeros(num_bins, dtype=np.float64)
    
    snps_per = n_snps // num_bins
    remainder = n_snps % num_bins
    
    start_idx = 0
    for b in range(num_bins):
        # Distribute remainder SNPs across first bins
        extra = 1 if b < remainder else 0
        end_idx = start_idx + snps_per + extra
        
        bin_starts[b] = start_idx
        bin_ends[b] = end_idx
        
        # Bin center is average position of SNPs in bin
        if end_idx > start_idx:
            bin_centers[b] = np.mean(snp_positions[start_idx:end_idx])
        
        start_idx = end_idx
    
    return bin_starts, bin_ends, bin_centers


def aggregate_hom_mask_to_bins(snp_hom_mask, bin_starts, bin_ends):
    """
    Aggregate SNP-level hom_mask to bin-level.
    A bin is potentially homozygous if ANY SNP in it is potentially homozygous.
    
    Args:
        snp_hom_mask: (n_snps,) boolean array
        bin_starts, bin_ends: Bin structure arrays
        
    Returns:
        bin_hom_mask: (n_bins,) boolean array
    """
    n_bins = len(bin_starts)
    bin_hom_mask = np.zeros(n_bins, dtype=np.bool_)
    
    for b in range(n_bins):
        # If any SNP in bin is potentially hom, bin is potentially hom
        bin_hom_mask[b] = np.any(snp_hom_mask[bin_starts[b]:bin_ends[b]])
    
    return bin_hom_mask


# =============================================================================
# 3. STATISTICAL CALCULATIONS
# =============================================================================

def calculate_ibd_matrices(grid):
    """Calculate IBD matrices from a standard grid (not used in tolerance version)."""
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
    """
    Calculates the Viterbi Score of Parent -> Child inheritance allowing for
    phase switching in the Child AND Split-Burst error handling.
    
    Uses SOFT EMISSIONS: mismatches incur a penalty (default log(0.01) ≈ -4.6)
    rather than being impossible (-1e9). Burst states are retained for handling
    runs of errors (e.g., coverage dropout).
    
    Args:
        child_dip_alleles: (n_sites, 2) allele array for child
        child_potential_hom_mask: (n_sites,) boolean array - True where homozygosity is possible
        parent_dip_alleles: (n_sites, 2) allele array for parent
        switch_costs, stay_costs: Transition cost arrays
        error_penalty: Cost to enter burst state
        phase_penalty: Cost for phase switch in non-homozygous regions
        mismatch_penalty: Log-probability of a single allele mismatch (default log(0.01))
    """
    n_sites = len(child_dip_alleles)
    
    # Init Scores (8 states) [Norm0..3, Burst0..3]
    scores = np.zeros(8)
    
    # Burst Emission: log(0.5) - random coin flip
    BURST_EMISSION = -0.693147 
    
    # Initialize Bursts as valid starting points (entering with penalty)
    for k in range(4, 8):
        scores[k] = -error_penalty
    
    for i in range(n_sites):
        # Alleles for Emissions
        c0_a, c1_a = child_dip_alleles[i, 0], child_dip_alleles[i, 1]
        p0_a, p1_a = parent_dip_alleles[i, 0], parent_dip_alleles[i, 1]
        
        # SOFT EMISSIONS: mismatch incurs penalty, not impossible
        # -1 (Missing) is treated as a wildcard match (0 cost)
        def soft_match(child_allele, parent_allele):
            if child_allele == -1 or parent_allele == -1:
                return 0.0  # Missing = wildcard
            elif child_allele == parent_allele:
                return 0.0  # Match
            else:
                return mismatch_penalty  # Soft mismatch
        
        e0 = soft_match(c0_a, p0_a)
        e1 = soft_match(c1_a, p0_a)
        e2 = soft_match(c0_a, p1_a)
        e3 = soft_match(c1_a, p1_a)
        
        emissions = np.array([e0, e1, e2, e3])
        
        # 2. Transition Costs
        c_recomb = switch_costs[i]
        c_stay = stay_costs[i]
        
        # Child Phase Switch Cost
        # FREE SWITCH if potentially homozygous (set intersection non-empty)
        c_phase = 0.0 if child_potential_hom_mask[i] else -phase_penalty
        
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


@njit(fastmath=True)
def run_trio_phase_aware_hmm(child_dip_alleles, child_potential_hom_mask, 
                             p1_dip_alleles, p2_dip_alleles, 
                             switch_costs, stay_costs, error_penalty, phase_penalty,
                             mismatch_penalty=-4.6):
    """
    Calculates the Joint Likelihood of P1 and P2 explaining the Child.
    Includes 16-State Split-Burst Logic to handle redundant coverage and genotyping errors.
    
    Uses SOFT EMISSIONS: mismatches incur a penalty (default log(0.01) ≈ -4.6)
    rather than being impossible. Both parent contributions must match for full
    score; each mismatch adds the penalty.
    
    FREE SWITCHES in potentially homozygous regions (where possible_hap1 ∩ possible_hap2 ≠ ∅).
    
    Args:
        child_dip_alleles: (n_sites, 2) allele array for child
        child_potential_hom_mask: (n_sites,) boolean array - True where homozygosity is possible
        p1_dip_alleles, p2_dip_alleles: (n_sites, 2) allele arrays for parents
        switch_costs, stay_costs: Transition cost arrays
        error_penalty: Cost to enter burst state
        phase_penalty: Cost for phase switch in non-homozygous regions
        mismatch_penalty: Log-probability of a single allele mismatch (default log(0.01))
    """
    n_sites = len(child_dip_alleles)
    
    # Burst Emission: log(0.25) - two independent coin flips
    BURST_EMISSION = -1.386
    
    # Init Scores (16 states)
    scores = np.zeros(16)
    for k in range(8, 16): scores[k] = -error_penalty
    
    for i in range(n_sites):
        # 1. Data Fetch
        c0, c1 = child_dip_alleles[i, 0], child_dip_alleles[i, 1]
        p1_h0, p1_h1 = p1_dip_alleles[i, 0], p1_dip_alleles[i, 1]
        p2_h0, p2_h1 = p2_dip_alleles[i, 0], p2_dip_alleles[i, 1]
        
        # 2. SOFT EMISSIONS
        # Each match contributes 0.0, each mismatch contributes mismatch_penalty
        # Missing (-1) is wildcard (0.0 cost)
        def soft_match(parent_allele, child_allele):
            if parent_allele == -1 or child_allele == -1:
                return 0.0  # Missing = wildcard
            elif parent_allele == child_allele:
                return 0.0  # Match
            else:
                return mismatch_penalty  # Soft mismatch
        
        m_p1h0_c0 = soft_match(p1_h0, c0); m_p1h1_c0 = soft_match(p1_h1, c0)
        m_p1h0_c1 = soft_match(p1_h0, c1); m_p1h1_c1 = soft_match(p1_h1, c1)
        m_p2h0_c0 = soft_match(p2_h0, c0); m_p2h1_c0 = soft_match(p2_h1, c0)
        m_p2h0_c1 = soft_match(p2_h0, c1); m_p2h1_c1 = soft_match(p2_h1, c1)
        
        # Group A (P1->C0, P2->C1): sum of both contributions
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
        
        # FREE SWITCH if potentially homozygous
        c_phase = 0.0 if child_potential_hom_mask[i] else -phase_penalty
        
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
    """
    8-state HMM for parent-child scoring with k sampled SNPs per bin.
    
    Sums mismatch penalties across all k samples per bin for more robust emission scoring.
    
    Args:
        child_dip_alleles: (n_bins, 2, k) allele array for child
        child_potential_hom_mask: (n_bins,) boolean array - True where homozygosity is possible
        parent_dip_alleles: (n_bins, 2, k) allele array for parent
        switch_costs, stay_costs: Transition cost arrays
        error_penalty: Cost to enter burst state
        phase_penalty: Cost for phase switch in non-homozygous regions
        mismatch_penalty: Log-probability of a single allele mismatch (default log(0.01))
    """
    n_bins = child_dip_alleles.shape[0]
    k_snps = child_dip_alleles.shape[2]
    
    # Init Scores (8 states) [Norm0..3, Burst0..3]
    scores = np.zeros(8)
    
    # Burst Emission scales with k
    BURST_EMISSION_PER_SNP = -0.693147  # log(0.5)
    
    # Initialize Bursts as valid starting points (entering with penalty)
    for state in range(4, 8):
        scores[state] = -error_penalty
    
    for i in range(n_bins):
        # Count mismatches across all k SNPs for each state
        e0, e1, e2, e3 = 0.0, 0.0, 0.0, 0.0
        valid_snps = 0
        
        for s in range(k_snps):
            c0_a = child_dip_alleles[i, 0, s]
            c1_a = child_dip_alleles[i, 1, s]
            p0_a = parent_dip_alleles[i, 0, s]
            p1_a = parent_dip_alleles[i, 1, s]
            
            # Skip if any allele is missing
            if c0_a < 0 or c1_a < 0 or p0_a < 0 or p1_a < 0:
                continue
            
            valid_snps += 1
            
            # State 0: P_hap0 -> C_hap0
            if c0_a != p0_a:
                e0 += mismatch_penalty
            # State 1: P_hap0 -> C_hap1
            if c1_a != p0_a:
                e1 += mismatch_penalty
            # State 2: P_hap1 -> C_hap0
            if c0_a != p1_a:
                e2 += mismatch_penalty
            # State 3: P_hap1 -> C_hap1
            if c1_a != p1_a:
                e3 += mismatch_penalty
        
        emissions = np.array([e0, e1, e2, e3])
        
        # Burst emission scales with number of valid SNPs
        burst_emission = BURST_EMISSION_PER_SNP * max(valid_snps, 1)
        
        # Transition Costs
        c_recomb = switch_costs[i]
        c_stay = stay_costs[i]
        
        # Child Phase Switch Cost
        c_phase = 0.0 if child_potential_hom_mask[i] else -phase_penalty
        
        prev = scores.copy()
        new_scores = np.zeros(8)
        
        # --- A. UPDATE BURST STATES (4-7) ---
        for state in range(4):
            burst_idx = state + 4
            from_burst = prev[burst_idx]
            from_normal = prev[state] - error_penalty
            new_scores[burst_idx] = max(from_burst, from_normal) + burst_emission

        # --- B. UPDATE NORMAL STATES (0-3) ---
        prev_b0 = prev[4]; prev_b1 = prev[5]; prev_b2 = prev[6]; prev_b3 = prev[7]
        
        # State 0 (P0, C0)
        src0 = prev[0] + c_stay
        src1 = prev[1] + c_stay + c_phase
        src2 = prev[2] + c_recomb
        new_scores[0] = max(src0, src1, src2, prev_b0) + emissions[0]
        
        # State 1 (P0, C1)
        src1 = prev[1] + c_stay
        src0 = prev[0] + c_stay + c_phase
        src3 = prev[3] + c_recomb
        new_scores[1] = max(src1, src0, src3, prev_b1) + emissions[1]
        
        # State 2 (P1, C0)
        src2 = prev[2] + c_stay
        src3 = prev[3] + c_stay + c_phase
        src0 = prev[0] + c_recomb
        new_scores[2] = max(src2, src3, src0, prev_b2) + emissions[2]
        
        # State 3 (P1, C1)
        src3 = prev[3] + c_stay
        src2 = prev[2] + c_stay + c_phase
        src1 = prev[1] + c_recomb
        new_scores[3] = max(src3, src2, src1, prev_b3) + emissions[3]
        
        scores = new_scores
    
    # Return max over all 8 states
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
    """
    16-state HMM for trio scoring with k sampled SNPs per bin.
    
    Sums mismatch penalties across all k samples per bin for more robust emission scoring.
    
    Args:
        child_dip_alleles: (n_bins, 2, k) allele array for child
        child_potential_hom_mask: (n_bins,) boolean array - True where homozygosity is possible
        p1_dip_alleles, p2_dip_alleles: (n_bins, 2, k) allele arrays for parents
        switch_costs, stay_costs: Transition cost arrays
        error_penalty: Cost to enter burst state
        phase_penalty: Cost for phase switch in non-homozygous regions
        mismatch_penalty: Log-probability of a single allele mismatch (default log(0.01))
    """
    n_bins = child_dip_alleles.shape[0]
    k_snps = child_dip_alleles.shape[2]
    
    # Burst Emission scales with k: log(0.25) per SNP
    BURST_EMISSION_PER_SNP = -1.386
    
    # Init Scores (16 states)
    scores = np.zeros(16)
    for state in range(8, 16):
        scores[state] = -error_penalty
    
    for i in range(n_bins):
        # Count mismatches across all k SNPs for each of 8 states
        e = np.zeros(8)
        valid_snps = 0
        
        for s in range(k_snps):
            c0 = child_dip_alleles[i, 0, s]
            c1 = child_dip_alleles[i, 1, s]
            p1_h0 = p1_dip_alleles[i, 0, s]
            p1_h1 = p1_dip_alleles[i, 1, s]
            p2_h0 = p2_dip_alleles[i, 0, s]
            p2_h1 = p2_dip_alleles[i, 1, s]
            
            # Skip if any allele is missing
            if c0 < 0 or c1 < 0 or p1_h0 < 0 or p1_h1 < 0 or p2_h0 < 0 or p2_h1 < 0:
                continue
            
            valid_snps += 1
            
            # Group A: P1 -> C0, P2 -> C1
            if c0 != p1_h0:
                e[0] += mismatch_penalty
            if c1 != p2_h0:
                e[0] += mismatch_penalty
            
            if c0 != p1_h0:
                e[1] += mismatch_penalty
            if c1 != p2_h1:
                e[1] += mismatch_penalty
            
            if c0 != p1_h1:
                e[2] += mismatch_penalty
            if c1 != p2_h0:
                e[2] += mismatch_penalty
            
            if c0 != p1_h1:
                e[3] += mismatch_penalty
            if c1 != p2_h1:
                e[3] += mismatch_penalty
            
            # Group B: P1 -> C1, P2 -> C0
            if c1 != p1_h0:
                e[4] += mismatch_penalty
            if c0 != p2_h0:
                e[4] += mismatch_penalty
            
            if c1 != p1_h0:
                e[5] += mismatch_penalty
            if c0 != p2_h1:
                e[5] += mismatch_penalty
            
            if c1 != p1_h1:
                e[6] += mismatch_penalty
            if c0 != p2_h0:
                e[6] += mismatch_penalty
            
            if c1 != p1_h1:
                e[7] += mismatch_penalty
            if c0 != p2_h1:
                e[7] += mismatch_penalty
        
        # Burst emission scales with valid SNPs
        burst_emission = BURST_EMISSION_PER_SNP * max(valid_snps, 1)
        
        # Transitions
        c_recomb = switch_costs[i]
        c_stay = stay_costs[i]
        
        # FREE SWITCH if potentially homozygous
        c_phase = 0.0 if child_potential_hom_mask[i] else -phase_penalty
        
        prev = scores.copy()
        new_scores = np.zeros(16)
        
        # --- A. UPDATE BURST STATES (8-15) ---
        for state in range(8):
            burst_idx = state + 8
            from_burst = prev[burst_idx]
            from_normal = prev[state] - error_penalty
            new_scores[burst_idx] = max(from_burst, from_normal) + burst_emission

        # --- B. UPDATE NORMAL STATES (0-7) ---
        cc_0 = 2 * c_stay
        cc_1 = c_recomb + c_stay
        cc_2 = 2 * c_recomb
        
        # Group A transitions
        p0, p1, p2, p3 = prev[0], prev[1], prev[2], prev[3]
        a0 = max(p0+cc_0, p1+cc_1, p2+cc_1, p3+cc_2)
        a1 = max(p0+cc_1, p1+cc_0, p2+cc_2, p3+cc_1)
        a2 = max(p0+cc_1, p1+cc_2, p2+cc_0, p3+cc_1)
        a3 = max(p0+cc_2, p1+cc_1, p2+cc_1, p3+cc_0)
        
        # Group B transitions
        p4, p5, p6, p7 = prev[4], prev[5], prev[6], prev[7]
        b4 = max(p4+cc_0, p5+cc_1, p6+cc_1, p7+cc_2)
        b5 = max(p4+cc_1, p5+cc_0, p6+cc_2, p7+cc_1)
        b6 = max(p4+cc_1, p5+cc_2, p6+cc_0, p7+cc_1)
        b7 = max(p4+cc_2, p5+cc_1, p6+cc_1, p7+cc_0)
        
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
    
    best_final = -np.inf
    for state in range(16):
        if scores[state] > best_final:
            best_final = scores[state]
    return best_final
# =============================================================================
# 5. TOLERANCE-AWARE SCORING FUNCTIONS
# =============================================================================

# Default mismatch penalty: log(0.01) ≈ -4.6
DEFAULT_MISMATCH_PENALTY = -4.605170  # math.log(0.01)

def score_parent_child_all_consensus(child_grids, parent_grids, 
                                     switch_costs, stay_costs, 
                                     error_penalty, phase_penalty,
                                     founder_block, bin_centers, bin_width,
                                     mismatch_penalty=DEFAULT_MISMATCH_PENALTY):
    """
    Score a parent-child pair considering ALL consensus paintings.
    NOTE: This version converts grids on-the-fly. Use score_parent_child_all_consensus_precomputed
    for better performance when scoring many pairs.
    """
    best_score = -np.inf
    
    for child_id_grid, child_hom_mask, child_weight in child_grids:
        child_allele_grid = convert_id_grid_to_allele_grid(
            child_id_grid, bin_centers, founder_block, bin_width
        )
        
        for parent_id_grid, parent_hom_mask, parent_weight in parent_grids:
            parent_allele_grid = convert_id_grid_to_allele_grid(
                parent_id_grid, bin_centers, founder_block, bin_width
            )
            
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


def score_parent_child_all_consensus_precomputed(child_grids, parent_grids, 
                                                  switch_costs, stay_costs, 
                                                  error_penalty, phase_penalty,
                                                  mismatch_penalty=DEFAULT_MISMATCH_PENALTY):
    """
    Score a parent-child pair considering ALL consensus paintings.
    Uses PRE-COMPUTED allele grids for efficiency.
    
    Args:
        child_grids: List of (allele_grid, potential_hom_mask, weight) tuples
        parent_grids: List of (allele_grid, potential_hom_mask, weight) tuples
        switch_costs, stay_costs: Transition cost arrays
        error_penalty, phase_penalty: HMM parameters
        mismatch_penalty: Log-probability of single allele mismatch
        
    Returns:
        best_score: Maximum HMM score over all consensus combinations
    """
    best_score = -np.inf
    
    for child_allele_grid, child_hom_mask, child_weight in child_grids:
        for parent_allele_grid, parent_hom_mask, parent_weight in parent_grids:
            # Check if 3D (multi-SNP) or 2D (single midpoint)
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
    """
    Score a trio (child, parent1, parent2) considering ALL consensus paintings.
    Uses PRE-COMPUTED allele grids for efficiency.
    
    Args:
        child_grids, p1_grids, p2_grids: Lists of (allele_grid, hom_mask, weight) tuples
        switch_costs, stay_costs: Transition cost arrays
        error_penalty, phase_penalty: HMM parameters
        mismatch_penalty: Log-probability of single allele mismatch
    
    Returns:
        best_score: Maximum HMM score over all consensus combinations
    """
    best_score = -np.inf
    
    for child_allele_grid, child_hom_mask, child_weight in child_grids:
        for p1_allele_grid, p1_hom_mask, p1_weight in p1_grids:
            for p2_allele_grid, p2_hom_mask, p2_weight in p2_grids:
                # Check if 3D (multi-SNP) or 2D (single midpoint)
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


def score_trio_all_consensus(child_grids, p1_grids, p2_grids,
                             switch_costs, stay_costs,
                             error_penalty, phase_penalty,
                             founder_block, bin_centers, bin_width,
                             mismatch_penalty=DEFAULT_MISMATCH_PENALTY):
    """
    Score a trio (child, parent1, parent2) considering ALL consensus paintings.
    NOTE: This version converts grids on-the-fly. Use score_trio_all_consensus_precomputed
    for better performance.
    """
    best_score = -np.inf
    
    for child_id_grid, child_hom_mask, child_weight in child_grids:
        child_allele_grid = convert_id_grid_to_allele_grid(
            child_id_grid, bin_centers, founder_block, bin_width
        )
        
        for p1_id_grid, p1_hom_mask, p1_weight in p1_grids:
            p1_allele_grid = convert_id_grid_to_allele_grid(
                p1_id_grid, bin_centers, founder_block, bin_width
            )
            
            for p2_id_grid, p2_hom_mask, p2_weight in p2_grids:
                p2_allele_grid = convert_id_grid_to_allele_grid(
                    p2_id_grid, bin_centers, founder_block, bin_width
                )
                
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
# 6. MULTI-CONTIG INFERENCE LOGIC (TOLERANCE VERSION - PARALLELIZED)
# =============================================================================


def _process_contig(args):
    """
    Worker function to process a single contig in Phase 1.
    Discretizes paintings and converts to allele grids using multi-SNP voting.
    
    Returns:
        dict with 'sample_allele_grids', 'sw_costs', 'st_costs', 'num_bins', 'switch_counts'
    """
    (tol_painting, founder_block, snps_per_bin, recomb_rate, num_samples, max_snps_per_bin) = args
    
    # Discretize all consensus paintings (returns id_grid, potential_hom_mask, weight)
    sample_grids, bin_centers = discretize_tolerance_paintings(
        tol_painting, snps_per_bin=snps_per_bin
    )
    
    num_bins = len(bin_centers)
    
    # Calculate bin width
    bin_width = 10000.0
    if num_bins > 1:
        bin_width = bin_centers[1] - bin_centers[0]
    
    # PRE-COMPUTE ALLELE GRIDS FOR ALL SAMPLES using multi-SNP voting
    sample_allele_grids = []
    for i in range(num_samples):
        allele_grids_for_sample = []
        for id_grid, hom_mask, weight in sample_grids[i]:
            if max_snps_per_bin > 1:
                # Use multi-SNP voting for more robust allele determination
                allele_grid = convert_id_grid_to_allele_grid_multisnp(
                    id_grid, bin_centers, founder_block, bin_width, max_snps_per_bin
                )
            else:
                # Use original single-SNP midpoint approach
                allele_grid = convert_id_grid_to_allele_grid(
                    id_grid, bin_centers, founder_block, bin_width
                )
            allele_grids_for_sample.append((allele_grid, hom_mask, weight))
        sample_allele_grids.append(allele_grids_for_sample)
    
    # Transition costs
    dists = np.zeros(num_bins)
    dists[1:] = np.diff(bin_centers)
    theta = np.clip(1.0 - np.exp(-dists * recomb_rate), 1e-15, 0.5)
    sw_costs = np.log(theta)
    st_costs = np.log(1.0 - theta)
    
    # Count switches (use first consensus for complexity estimate)
    switch_counts = np.zeros(num_samples)
    for i in range(num_samples):
        if sample_grids[i]:
            id_grid = sample_grids[i][0][0]
            switches = (id_grid[:-1, :] != id_grid[1:, :]) & \
                      (id_grid[:-1, :] != -1) & (id_grid[1:, :] != -1)
            switch_counts[i] = np.sum(switches)
    
    return {
        'sample_allele_grids': sample_allele_grids,
        'sw_costs': sw_costs,
        'st_costs': st_costs,
        'num_bins': num_bins,
        'switch_counts': switch_counts
    }


def _score_contig_pairs(args):
    """
    Worker function to score all parent-child pairs for a single contig.
    Uses pre-computed allele grids.
    Returns a (num_samples, num_samples) score matrix.
    """
    (sample_allele_grids, sw_costs, st_costs, error_pen, phase_pen, 
     mismatch_penalty, num_samples) = args
    
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


def _score_trios_for_sample(args):
    """
    Worker function to score all trio combinations for a single sample (child).
    Uses pre-computed allele grids.
    Returns (sample_idx, best_trio, best_trio_score, top_indices).
    """
    (sample_idx, top_indices, contig_caches, total_switches, 
     error_pen, phase_pen, mismatch_penalty, complexity_penalty) = args
    
    if len(top_indices) < 1:
        return (sample_idx, None, -1e9, [])
    
    # Form all possible parent pairs
    pairs = [(p1, p2) for p1 in top_indices for p2 in top_indices if p1 != p2]
    if not pairs:
        pairs = [(top_indices[0], top_indices[0])]
    
    best_trio = None
    best_trio_score = -np.inf
    
    # Score all pairs across all contigs
    for p1, p2 in pairs:
        trio_ll = 0.0
        
        for cache in contig_caches:
            sample_allele_grids = cache['sample_allele_grids']
            sw_costs = cache['sw_costs']
            st_costs = cache['st_costs']
            
            score = score_trio_all_consensus_precomputed(
                sample_allele_grids[sample_idx],  # child
                sample_allele_grids[p1],          # parent 1
                sample_allele_grids[p2],          # parent 2
                sw_costs, st_costs,
                error_pen, phase_pen,
                mismatch_penalty
            )
            
            trio_ll += score
        
        final = trio_ll - (total_switches[p1] + total_switches[p2]) * complexity_penalty
        
        if final > best_trio_score:
            best_trio_score = final
            best_trio = (p1, p2)
    
    return (sample_idx, best_trio, best_trio_score, top_indices)


def _score_trios_batch(args):
    """
    Worker function to score trios for a BATCH of samples.
    Uses pre-computed allele grids.
    Reduces serialization overhead by processing multiple samples per worker.
    
    Returns list of (sample_idx, best_trio, best_trio_score, top_indices) tuples.
    """
    (batch_sample_args, contig_caches, total_switches,
     error_pen, phase_pen, mismatch_penalty, complexity_penalty) = args
    
    results = []
    
    for sample_idx, top_indices in batch_sample_args:
        if len(top_indices) < 1:
            results.append((sample_idx, None, -1e9, []))
            continue
        
        # Form all possible parent pairs
        pairs = [(p1, p2) for p1 in top_indices for p2 in top_indices if p1 != p2]
        if not pairs:
            pairs = [(top_indices[0], top_indices[0])]
        
        best_trio = None
        best_trio_score = -np.inf
        
        # Score all pairs across all contigs
        for p1, p2 in pairs:
            trio_ll = 0.0
            
            for cache in contig_caches:
                sample_allele_grids = cache['sample_allele_grids']
                sw_costs = cache['sw_costs']
                st_costs = cache['st_costs']
                
                score = score_trio_all_consensus_precomputed(
                    sample_allele_grids[sample_idx],  # child
                    sample_allele_grids[p1],          # parent 1
                    sample_allele_grids[p2],          # parent 2
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
                                          snps_per_bin=150, recomb_rate=5e-8,
                                          mismatch_penalty=DEFAULT_MISMATCH_PENALTY,
                                          max_snps_per_bin=10,
                                          n_workers=None):
    """
    Multi-contig pedigree inference using tolerance paintings with multi-SNP voting.
    
    Args:
        contig_data_list: List of dicts with:
            - 'tolerance_painting': BlockTolerancePainting from paint_samples
            - 'founder_block': BlockResult with positions and haplotypes
        sample_ids: List of sample names
        top_k: Number of top parent candidates to consider for trio verification
        snps_per_bin: Grid resolution (target SNPs per bin)
        recomb_rate: Per-bp recombination rate
        mismatch_penalty: Log-probability of single allele mismatch (default log(0.01) ≈ -4.6)
        max_snps_per_bin: Number of SNPs to sample per bin for allele voting (default 10).
                          Set to 1 for original midpoint behavior.
        n_workers: Number of parallel workers (default: auto)
        
    Returns:
        PedigreeResult object
    """
    import multiprocessing as mp
    import os
    
    num_samples = len(sample_ids)
    num_contigs = len(contig_data_list)
    
    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, 16)
    
    # Phase 1: Pre-calculate and discretize all paintings (PARALLELIZED with fork)
    print(f"\n--- Phase 1: Processing {num_contigs} Contigs ({n_workers} workers, {max_snps_per_bin} SNPs/bin) ---")
    
    error_pen = -math.log(1e-2)
    phase_pen = 50.0
    
    # Prepare arguments for parallel workers
    phase1_args = [
        (data['tolerance_painting'], data['founder_block'], snps_per_bin, recomb_rate, 
         num_samples, max_snps_per_bin)
        for data in contig_data_list
    ]
    
    # Process contigs in parallel using Pool with fork
    ctx = mp.get_context('fork')
    with ctx.Pool(processes=n_workers) as pool:
        contig_results = list(tqdm(
            pool.imap(_process_contig, phase1_args),
            total=num_contigs,
            desc="Processing contigs in parallel"
        ))
    
    # Aggregate results
    contig_caches = []
    total_switches = np.zeros(num_samples)
    global_total_bins = 0
    
    for result in contig_results:
        contig_caches.append({
            'sample_allele_grids': result['sample_allele_grids'],
            'sw_costs': result['sw_costs'],
            'st_costs': result['st_costs']
        })
        global_total_bins += result['num_bins']
        total_switches += result['switch_counts']
    
    # Phase 1b: Score all pairs in PARALLEL across contigs
    print(f"\n--- Phase 1b: Scoring Parent-Child Pairs ({n_workers} workers) ---")
    
    # Prepare arguments for parallel workers
    worker_args = [
        (cache['sample_allele_grids'], cache['sw_costs'], cache['st_costs'],
         error_pen, phase_pen, mismatch_penalty, num_samples)
        for cache in contig_caches
    ]
    
    # Use Pool with fork for true parallelization
    total_scores = np.zeros((num_samples, num_samples))
    
    with ctx.Pool(processes=n_workers) as pool:
        score_matrices = list(tqdm(
            pool.imap(_score_contig_pairs, worker_args),
            total=num_contigs,
            desc="Scoring contigs in parallel"
        ))
    
    # Sum score matrices from all contigs
    for score_matrix in score_matrices:
        total_scores += score_matrix
    
    # Handle -inf scores
    total_scores[total_scores == -np.inf] = -1e9
    
    # Build candidate mask based on complexity
    cand_mask = np.zeros((num_samples, num_samples), dtype=bool)
    margin = 5
    for i in range(num_samples):
        valid_gen = total_switches <= (total_switches[i] + margin)
        cand_mask[i, :] = valid_gen
        cand_mask[i, i] = False

    # Phase 2: Trio Verification (PARALLELIZED with Pool and fork)
    print(f"\n--- Phase 2: Trio Verification (Top {top_k} Pairs, {n_workers} workers) ---")
    
    COMPLEXITY_PENALTY = 0.0
    
    # Prepare arguments for each sample
    all_sample_args = []
    for i in range(num_samples):
        # Get Top Candidates
        valid_scores = total_scores[i].copy()
        valid_scores[~cand_mask[i, :]] = -np.inf
        
        # Apply Penalty for ranking
        for j in range(num_samples):
            if valid_scores[j] > -1e9:
                valid_scores[j] -= (total_switches[j] * COMPLEXITY_PENALTY)
        
        top_indices = np.argsort(valid_scores)[-top_k:][::-1]
        top_indices = [x for x in top_indices if valid_scores[x] > -1e10]
        
        all_sample_args.append((i, top_indices))
    
    # Batch samples together to reduce serialization overhead
    batch_size = max(1, num_samples // (n_workers * 4))  # ~4 batches per worker
    batched_args = []
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_sample_args = all_sample_args[batch_start:batch_end]
        batched_args.append((
            batch_sample_args, contig_caches, total_switches,
            error_pen, phase_pen, mismatch_penalty, COMPLEXITY_PENALTY
        ))
    
    # Run trio scoring in parallel using Pool with fork
    with ctx.Pool(processes=n_workers) as pool:
        batch_results = list(tqdm(
            pool.imap(_score_trios_batch, batched_args),
            total=len(batched_args),
            desc="Inferring Trios in parallel"
        ))
    
    # Flatten batch results
    trio_results = []
    for batch in batch_results:
        trio_results.extend(batch)
    
    # Collect results
    relationships = []
    parent_candidates = {}
    trio_scores_map = {}
    
    for sample_idx, best_trio, best_trio_score, top_indices in trio_results:
        # Store parent candidates
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
    res = PedigreeResult(sample_ids, rel_df, parent_candidates, None, [], None, None, 
                         trio_scores_map, global_total_bins)
    
    # Run automatic cutoff internally
    res.perform_automatic_cutoff()
    
    # Resolve any cycles introduced by incorrect parent assignments
    res.resolve_cycles()
    
    return res

# =============================================================================
# 7. CYCLE DETECTION AND RESOLUTION
# =============================================================================

def build_pedigree_graph(relationships_df):
    """
    Build a directed graph from pedigree relationships.
    Edges go from parent -> child.
    """
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
    """
    Detect cycles in a pedigree.
    
    Returns:
        List of cycles, where each cycle is a list of sample names.
        Empty list if pedigree is a valid DAG.
    """
    G = build_pedigree_graph(relationships_df)
    
    if nx.is_directed_acyclic_graph(G):
        return []
    
    return list(nx.simple_cycles(G))


def resolve_pedigree_cycles(relationships_df, trio_scores=None, verbose=True):
    """
    Detect and resolve cycles in a pedigree by identifying samples that
    appear in cycles and removing their parent assignments.
    
    Strategy: Samples that have many children but also have parents assigned
    are likely mis-classified. We remove parent assignments from the sample
    with the most children (they're clearly a parent, not a child) and 
    worst trio score (their parent assignment is least confident).
    
    Args:
        relationships_df: DataFrame with Sample, Generation, Parent1, Parent2
        trio_scores: Dict mapping sample -> trio score (more negative = worse fit)
        verbose: Print progress information
    
    Returns:
        Fixed DataFrame with cycles broken
    """
    fixed_df = relationships_df.copy()
    
    # Default trio scores to 0 if not provided
    if trio_scores is None:
        trio_scores = {}
    
    iteration = 0
    max_iterations = 20  # Safety limit
    
    while iteration < max_iterations:
        iteration += 1
        
        # Build graph
        G = build_pedigree_graph(fixed_df)
        
        # Check if already valid
        if nx.is_directed_acyclic_graph(G):
            if verbose and iteration > 1:
                print(f"[Cycle Resolution] Pedigree is acyclic after {iteration-1} fix(es).")
            break
        
        # Find cycles
        cycles = list(nx.simple_cycles(G))
        
        if verbose:
            print(f"[Cycle Resolution] Iteration {iteration}: Found {len(cycles)} cycle(s)")
        
        # Collect all samples involved in cycles
        samples_in_cycles = set()
        for cycle in cycles:
            samples_in_cycles.update(cycle)
        
        # For each sample in a cycle, check if it's both a parent AND a child
        candidates_to_fix = []
        
        for sample in samples_in_cycles:
            row = fixed_df[fixed_df['Sample'] == sample]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            
            has_parents = pd.notna(row['Parent1']) or pd.notna(row['Parent2'])
            n_children = G.out_degree(sample)
            
            if has_parents and n_children > 0:
                # This sample is in the middle of a cycle - both parent and child
                score = trio_scores.get(sample, 0.0)
                candidates_to_fix.append({
                    'sample': sample,
                    'score': score,
                    'n_children': n_children,
                    'parents': (row['Parent1'], row['Parent2'])
                })
        
        if not candidates_to_fix:
            if verbose:
                print(f"[Cycle Resolution] No candidates found to fix - breaking")
            break
        
        # Sort by: 1) Most children (likely a true parent), 2) Worst trio score
        candidates_to_fix.sort(key=lambda x: (-x['n_children'], x['score']))
        
        # Fix the top candidate
        to_fix = candidates_to_fix[0]
        sample_to_fix = to_fix['sample']
        
        if verbose:
            print(f"[Cycle Resolution] Fixing {sample_to_fix}: "
                  f"n_children={to_fix['n_children']}, score={to_fix['score']:.4f}, "
                  f"removing parents {to_fix['parents']}")
        
        idx = fixed_df[fixed_df['Sample'] == sample_to_fix].index[0]
        fixed_df.at[idx, 'Parent1'] = None
        fixed_df.at[idx, 'Parent2'] = None
        fixed_df.at[idx, 'Generation'] = 'F1'
    
    # Final verification
    G_final = build_pedigree_graph(fixed_df)
    if not nx.is_directed_acyclic_graph(G_final):
        remaining = list(nx.simple_cycles(G_final))
        if verbose:
            print(f"[Cycle Resolution] WARNING: {len(remaining)} cycles remain after max iterations")
    
    return fixed_df


def propagate_generations(relationships_df):
    """
    Re-propagate generation labels after cycle resolution.
    F1s have no parents, F2s have F1 parents, F3s have F2 parents, etc.
    """
    fixed_df = relationships_df.copy()
    
    # Start fresh: samples with no parents are F1
    name_to_gen = {}
    for idx, row in fixed_df.iterrows():
        if pd.isna(row['Parent1']) and pd.isna(row['Parent2']):
            fixed_df.at[idx, 'Generation'] = 'F1'
            name_to_gen[row['Sample']] = 'F1'
        else:
            fixed_df.at[idx, 'Generation'] = 'Unknown'
    
    # Propagate generations iteratively
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
    """Draw pedigree tree visualization."""
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


def run_pedigree_inference_tolerance(tolerance_painting, sample_ids=None, snps_per_bin=150, 
                                     founder_block=None, recomb_rate=5e-8,
                                     output_prefix="pedigree",
                                     mismatch_penalty=DEFAULT_MISMATCH_PENALTY,
                                     n_workers=None):
    """
    Convenience wrapper for single-contig pedigree inference with tolerance paintings.
    
    Args:
        tolerance_painting: BlockTolerancePainting from paint_samples.py
        sample_ids: List of sample names
        snps_per_bin: Grid resolution
        founder_block: BlockResult with founder haplotypes
        recomb_rate: Per-bp recombination rate
        output_prefix: Prefix for output files
        mismatch_penalty: Log-probability of single allele mismatch (default log(0.01))
        n_workers: Number of parallel workers (default: auto)
        
    Returns:
        PedigreeResult object
    """
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