"""
File which contains functions that takes as input a list of
probabalistic haplotypes and use them to simulate a multi generation
progeny of founders made up of these haplotypes

PARALLELIZED VERSION: Contigs processed in parallel within each generation.
"""
import thread_config

import random
import numpy as np
import pickle
import pandas as pd
import warnings
import os
import multiprocessing as mp
import multiprocessing.pool
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from vcf_data_loader import GenomicData

# Import painting classes for direct conversion
from paint_samples import SamplePainting, PaintedChunk, BlockPainting

# =============================================================================
# NON-DAEMONIC POOL (allows inner worker pools)
# =============================================================================
# Outer contig workers need to spawn inner offspring-batch pools.
# Standard Pool creates daemonic workers that cannot have children.
# NoDaemonPool overrides this, mirroring hierarchical_assembly.py.

try:
    _forkserver_ctx = mp.get_context('forkserver')
except (ValueError, AttributeError):
    _forkserver_ctx = mp.get_context('fork')

class _NoDaemonProcess(_forkserver_ctx.Process):
    @property
    def daemon(self):
        return False
    
    @daemon.setter
    def daemon(self, value):
        pass

class _NoDaemonContext(type(_forkserver_ctx)):
    Process = _NoDaemonProcess

class NoDaemonPool(multiprocessing.pool.Pool):
    """A Pool whose workers can spawn child pools (non-daemonic)."""
    def __init__(self, *args, **kwargs):
        kwargs['context'] = _NoDaemonContext()
        super().__init__(*args, **kwargs)

# Visualization imports
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_VIS = True
except ImportError:
    HAS_VIS = False

#%%
def concretify_haps(haps_list):
    """
    Takes a list of probabalistic haps and turns each of them 
    into a list of 0s and 1s by taking the highest probability
    allele at each site
    """
    concreted = []
    for hap in haps_list:
        concreted.append(np.argmax(hap, axis=1))
    return concreted

def pairup_haps(haps_list, shuffle=False):
    """
    Pair up a list of concrete haps (made up of 0s and 1s)
    """
    haps_copy = pickle.loads(pickle.dumps(haps_list))
    
    if shuffle:
        random.shuffle(haps_copy)
    
    num_pairs = len(haps_list) // 2
    haps_paired = []
    
    for i in range(num_pairs):
        first = haps_copy[2 * i]
        second = haps_copy[2 * i + 1]
        haps_paired.append([first, second])
    
    return haps_paired

def get_segments_in_range(source_painting, range_start, range_end):
    """
    Helper to extract and clip ancestry segments that fall within a specific window.
    source_painting: List of (start, end, founder_id)
    """
    result = []
    for (seg_start, seg_end, fid) in source_painting:
        overlap_start = max(seg_start, range_start)
        overlap_end = min(seg_end, range_end)
        if overlap_start < overlap_end:
            result.append((overlap_start, overlap_end, fid))
    return result

def recombine_haps(hap_pair, ancestry_pair, site_locs,
                   recomb_rate=10**-8, mutate_rate=10**-8, rng=None):
    """
    Simulates meiosis for ONE gamete.
    Tracks both ALLELES (0/1) and ANCESTRY (Founder IDs).
    """
    if rng is None:
        rng = np.random.default_rng()
    
    recomb_scale = 1.0 / recomb_rate
    mutate_scale = 1.0 / mutate_rate
    
    assert len(hap_pair[0]) == len(hap_pair[1]), "Length of two haplotypes is different"
    assert len(hap_pair[0]) == len(site_locs), "Different length of hap and of list of site locations"
    
    start_phys = site_locs[0]
    end_phys = site_locs[-1]
    
    cur_loc = start_phys
    cur_loc_index = 0
    
    using_hap = rng.choice([0, 1])
    
    final_hap_alleles = []
    final_hap_ancestry = []
    
    while cur_loc <= end_phys:
        next_break_distance = rng.exponential(recomb_scale)
        new_loc = cur_loc + np.ceil(next_break_distance)
        new_loc_index = np.searchsorted(site_locs, new_loc)
        
        adding = hap_pair[using_hap][cur_loc_index:new_loc_index]
        final_hap_alleles.append(adding)
        
        segment_phys_end = min(new_loc, end_phys)
        parent_segments = get_segments_in_range(
            ancestry_pair[using_hap], cur_loc, segment_phys_end
        )
        final_hap_ancestry.extend(parent_segments)
        
        using_hap = 1 - using_hap
        cur_loc = new_loc
        cur_loc_index = new_loc_index
        
        if cur_loc_index >= len(site_locs):
            break
    
    return_alleles = np.concatenate(final_hap_alleles)
    
    if len(return_alleles) > len(site_locs):
        return_alleles = return_alleles[:len(site_locs)]

    # Apply Mutations (only affects alleles, not ancestry)
    mutation_points = []
    cur_loc = start_phys
    
    while cur_loc <= end_phys:
        next_mutation_distance = rng.exponential(mutate_scale)
        new_loc = cur_loc + np.floor(next_mutation_distance)
        new_loc_index = np.searchsorted(site_locs, new_loc)
        
        if new_loc_index < len(site_locs):
            mutation_points.append(new_loc_index)
        
        cur_loc = new_loc
        
    if len(mutation_points) > 0:
        base_vals = return_alleles[mutation_points]
        mutated_vals = 1 - base_vals
        return_alleles[mutation_points] = mutated_vals
    
    return return_alleles, final_hap_ancestry

def create_offspring(first_pair, second_pair,
                     first_ancestry, second_ancestry,
                     site_locs, recomb_rate=10**-8,
                     mutate_rate=10**-8, rng=None):
    """
    Creates an offspring (Diploid) from two parents.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    h1_alleles, h1_ancestry = recombine_haps(
        first_pair, first_ancestry, site_locs, 
        recomb_rate=recomb_rate, mutate_rate=mutate_rate, rng=rng
    )
    
    h2_alleles, h2_ancestry = recombine_haps(
        second_pair, second_ancestry, site_locs, 
        recomb_rate=recomb_rate, mutate_rate=mutate_rate, rng=rng
    )
    
    return [h1_alleles, h2_alleles], [h1_ancestry, h2_ancestry]

def get_reads_from_sample(hap_pair, read_depth, error_rate=0.02):
    """
    Simulates sequencing an individual made up of a pair of haplotypes 
    up to read_depth average coverage.
    """
    num_sites = len(hap_pair[0])
    num_reads_at_site = np.random.poisson(lam=read_depth, size=num_sites)
    
    site_sum = hap_pair[0] + hap_pair[1]
    
    zeros = np.where(site_sum == 0)[0]
    ones = np.where(site_sum == 1)[0]
    twos = np.where(site_sum == 2)[0]
    
    zero_read_counts = num_reads_at_site[zeros]
    one_read_counts = num_reads_at_site[ones]
    two_read_counts = num_reads_at_site[twos]
    
    zero_draws = np.random.binomial(zero_read_counts, error_rate)
    one_draws = np.random.binomial(one_read_counts, 0.5)
    two_draws = np.random.binomial(two_read_counts, 1 - error_rate)
    
    zero_basics = zero_read_counts - zero_draws
    one_basics = one_read_counts - one_draws
    two_basics = two_read_counts - two_draws
    
    zero_concated = np.column_stack((zero_basics, zero_draws))
    one_concated = np.column_stack((one_basics, one_draws))
    two_concated = np.column_stack((two_basics, two_draws))
    
    full_scaffold = np.zeros((num_sites, 2), dtype=int)
    
    full_scaffold[zeros, :] = zero_concated
    full_scaffold[ones, :] = one_concated
    full_scaffold[twos, :] = two_concated
    
    return full_scaffold


def read_sample_all_individuals(individual_list, read_depth, error_rate=0.02, rng=None):
    """
    Vectorized read sampling across ALL individuals at once.
    
    Instead of looping over 320 individuals each doing separate
    np.random calls, performs single bulk poisson + masked binomial
    calls over the full (N_individuals, N_sites) array.
    
    Args:
        individual_list: List of (hap0, hap1) pairs.
        read_depth: Average sequencing depth.
        error_rate: Per-base sequencing error rate.
        rng: Optional numpy Generator for reproducibility. If None, uses
             np.random (legacy global state, not reproducible).
    """
    if rng is None:
        rng = np.random.default_rng()
    
    num_individuals = len(individual_list)
    num_sites = len(individual_list[0][0])
    
    hap0 = np.array([ind[0] for ind in individual_list])  # (N, S)
    hap1 = np.array([ind[1] for ind in individual_list])  # (N, S)
    site_sum = hap0 + hap1  # (N, S) — genotype: 0, 1, or 2
    
    # Sample read depths for all individuals x sites at once
    num_reads = rng.poisson(lam=read_depth, size=(num_individuals, num_sites))
    
    # Alt read count depends on genotype
    alt_reads = np.zeros_like(num_reads)
    
    mask0 = (site_sum == 0)
    mask1 = (site_sum == 1)
    mask2 = (site_sum == 2)
    
    if np.any(mask0):
        alt_reads[mask0] = rng.binomial(num_reads[mask0], error_rate)
    if np.any(mask1):
        alt_reads[mask1] = rng.binomial(num_reads[mask1], 0.5)
    if np.any(mask2):
        alt_reads[mask2] = rng.binomial(num_reads[mask2], 1 - error_rate)
    
    ref_reads = num_reads - alt_reads
    
    return np.stack([ref_reads, alt_reads], axis=-1).astype(int)


def combine_into_genotype(individual_list):
    """
    Vectorized genotype conversion for all individuals at once.
    
    Uses np.eye one-hot encoding instead of per-individual scatter
    indexing loops.
    """
    hap0 = np.array([ind[0] for ind in individual_list])  # (N, S)
    hap1 = np.array([ind[1] for ind in individual_list])  # (N, S)
    genotypes = hap0 + hap1  # (N, S) values in {0, 1, 2}
    
    eye = np.eye(3, dtype=np.float64)
    return eye[genotypes]  # (N, S, 3)


def chunk_up_data(positions_list, reads_array,
                  starting_pos, ending_pos,
                  block_size, shift_size,
                  use_snp_count=False, snps_per_block=200, snp_shift=100,
                  error_rate=0.02,
                  min_total_reads=5):
    """
    Breaks up the positions_list and reads_array into blocks.
    """
    chunked_positions = []
    chunked_reads = []
    chunked_keep_flags = []
    
    num_samples = reads_array.shape[0]
    total_sites = len(positions_list)
    
    range_start_idx = np.searchsorted(positions_list, starting_pos)
    range_end_idx = np.searchsorted(positions_list, ending_pos)
    
    positions_slice = positions_list[range_start_idx:range_end_idx]
    reads_slice = reads_array[:, range_start_idx:range_end_idx, :]
    
    slice_len = len(positions_slice)
    
    if slice_len == 0:
        return GenomicData([], [], [])

    if use_snp_count:
        curr_idx = 0
        while curr_idx < slice_len:
            end_idx = min(curr_idx + snps_per_block, slice_len)
            if end_idx == curr_idx:
                break
            
            block_positions = positions_slice[curr_idx:end_idx]
            block_reads_array = reads_slice[:, curr_idx:end_idx, :]
            
            total_read_pos = np.sum(block_reads_array, axis=(0, 2))
            block_keep_flags = (total_read_pos >= max(min_total_reads, error_rate * num_samples)).astype(int)
            
            chunked_positions.append(np.array(block_positions))
            chunked_reads.append(block_reads_array)
            chunked_keep_flags.append(block_keep_flags)
            
            curr_idx += snp_shift
            
    else:
        cur_pos = positions_slice[0]
        cur_idx_in_slice = 0
        max_phys_pos = positions_slice[-1]
        
        while cur_pos < max_phys_pos:
            block_end_pos = cur_pos + block_size
            end_idx_in_slice = cur_idx_in_slice
            while end_idx_in_slice < slice_len and positions_slice[end_idx_in_slice] < block_end_pos:
                end_idx_in_slice += 1
            
            block_positions = positions_slice[cur_idx_in_slice:end_idx_in_slice]
            
            if len(block_positions) > 0:
                block_reads_array = reads_slice[:, cur_idx_in_slice:end_idx_in_slice, :]
                total_read_pos = np.sum(block_reads_array, axis=(0, 2))
                block_keep_flags = (total_read_pos >= max(min_total_reads, error_rate * num_samples)).astype(int)
                
                chunked_positions.append(np.array(block_positions))
                chunked_reads.append(block_reads_array)
                chunked_keep_flags.append(block_keep_flags)
            
            cur_pos = cur_pos + shift_size
            while cur_idx_in_slice < slice_len and positions_slice[cur_idx_in_slice] < cur_pos:
                cur_idx_in_slice += 1
                
    return GenomicData(chunked_positions, chunked_keep_flags, chunked_reads)


def plot_ground_truth_pedigree(relationships_df, output_file="ground_truth_pedigree.png"):
    """
    Plots the Ground Truth Pedigree structure.
    Gracefully handles disk quota / IO errors.
    """
    if not HAS_VIS:
        print("Visualization libraries not found.")
        return
    if output_file is None:
        return
    
    try:
        folder = os.path.dirname(output_file)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        G = nx.DiGraph()
        unique_gens = sorted(relationships_df['Generation'].unique())
        
        all_samples = set(relationships_df['Sample'])
        all_parents = set(relationships_df['Parent1'].dropna()) | set(relationships_df['Parent2'].dropna())
        founders = list(all_parents - all_samples)
        
        if founders:
            unique_gens.insert(0, "Founder")
        
        cmap = plt.get_cmap("tab10")
        gen_colors = {gen: cmap(i) for i, gen in enumerate(unique_gens)}
        
        generations_nodes = {gen: [] for gen in unique_gens}
        parents_of = {}

        for f in founders:
            G.add_node(f, color=gen_colors["Founder"], gen="Founder")
            generations_nodes["Founder"].append(f)

        for _, row in relationships_df.iterrows():
            sample = row['Sample']
            gen = row['Generation']
            generations_nodes[gen].append(sample)
            G.add_node(sample, color=gen_colors[gen], gen=gen)
            
            if pd.notna(row['Parent1']): 
                G.add_edge(row['Parent1'], sample)
                parents_of.setdefault(sample, []).append(row['Parent1'])
            if pd.notna(row['Parent2']): 
                G.add_edge(row['Parent2'], sample)
                parents_of.setdefault(sample, []).append(row['Parent2'])
                
        pos = {}
        node_y_map = {} 
        
        def gen_sort_key(g):
            if g == "Founder":
                return -1
            if g.startswith("F") and g[1:].isdigit():
                return int(g[1:])
            return 999
            
        sorted_gens = sorted(unique_gens, key=gen_sort_key)
        
        for x_idx, gen in enumerate(sorted_gens):
            nodes = generations_nodes[gen]
            if not nodes:
                continue
            
            if x_idx == 0:
                nodes.sort()
            else:
                def get_parent_avg_y(node):
                    parents = parents_of.get(node, [])
                    if not parents:
                        return 0.5
                    ys = [node_y_map.get(p, 0.5) for p in parents]
                    return sum(ys) / len(ys)
                nodes.sort(key=get_parent_avg_y, reverse=True)
                
            for i, node in enumerate(nodes):
                y = 1.0 - (i + 0.5) / len(nodes)
                pos[node] = (x_idx, y)
                node_y_map[node] = y

        max_nodes = max([len(n) for n in generations_nodes.values()])
        fig_height = max(10, max_nodes * 0.25)
        
        plt.figure(figsize=(20, fig_height))
        node_colors = [G.nodes[n]['color'] for n in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_size=120, node_color=node_colors, edgecolors='black', linewidths=0.5)
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3, width=0.5, arrows=False)
        
        first_gen = sorted_gens[0]
        labels_first = {n: n for n in generations_nodes[first_gen]}
        pos_first = {n: (x - 0.03, y) for n, (x, y) in pos.items() if n in labels_first}
        nx.draw_networkx_labels(G, pos_first, labels=labels_first, font_size=10, horizontalalignment='right')
        
        labels_others = {}
        for gen in sorted_gens[1:]:
            for n in generations_nodes[gen]:
                labels_others[n] = n
        pos_others = {n: (x + 0.03, y) for n, (x, y) in pos.items() if n in labels_others}
        nx.draw_networkx_labels(G, pos_others, labels=labels_others, font_size=8, horizontalalignment='left')

        patches = [mpatches.Patch(color=gen_colors[g], label=g) for g in sorted_gens]
        plt.legend(handles=patches, loc='upper right')
        
        plt.title("Ground Truth Pedigree")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
    except OSError as e:
        print(f"WARNING: Could not save pedigree plot to {output_file}: {e}")
        plt.close('all')


# =============================================================================
# PARALLEL WORKER FUNCTIONS
# =============================================================================

def _process_offspring_batch(args):
    """
    Worker function to process a BATCH of offspring for a single contig.
    
    Each batch gets its own RNG seed derived from the contig seed, so results
    are deterministic regardless of how many inner workers are used.
    """
    (batch_indices, offspring_parent_indices_batch,
     parents, ancestries, site_locs,
     recomb_rate, mutate_rate, seed) = args
    
    rng = np.random.default_rng(seed)
    
    batch_haps = []
    batch_ancs = []
    
    for (p1_idx, p2_idx) in offspring_parent_indices_batch:
        child_haps, child_paintings = create_offspring(
            parents[p1_idx], parents[p2_idx],
            ancestries[p1_idx], ancestries[p2_idx],
            site_locs,
            recomb_rate=recomb_rate,
            mutate_rate=mutate_rate,
            rng=rng
        )
        
        batch_haps.append(child_haps)
        batch_ancs.append(child_paintings)
    
    return (batch_indices, batch_haps, batch_ancs)


def _process_contig_for_generation(args):
    """
    Worker function to process all offspring for a single contig within a generation.
    
    Two-level parallelism: this function is called as an outer worker (one per
    contig), and internally parallelises across offspring batches using an inner
    Pool.  The core budget is divided as num_processes // num_contigs per contig.
    
    When inner_workers <= 1 or the offspring count is small, falls back to
    sequential processing to avoid Pool overhead.
    """
    contig_idx = args[0]
    data = args[1]
    
    parents = data['parents']
    ancestries = data['ancestries']
    site_locs = data['site_locs']
    offspring_parent_indices = data['offspring_parent_indices']
    recomb_rate = data['recomb_rate']
    mutate_rate = data['mutate_rate']
    seed = data['seed']
    inner_workers = data.get('inner_workers', 1)
    
    num_offspring = len(offspring_parent_indices)
    
    # --- Sequential path (small batch or single inner worker) ---
    if inner_workers <= 1 or num_offspring <= inner_workers:
        rng = np.random.default_rng(seed)
        
        offspring_haps = []
        offspring_ancs = []
        
        for (p1_idx, p2_idx) in offspring_parent_indices:
            child_haps, child_paintings = create_offspring(
                parents[p1_idx], parents[p2_idx],
                ancestries[p1_idx], ancestries[p2_idx],
                site_locs,
                recomb_rate=recomb_rate,
                mutate_rate=mutate_rate,
                rng=rng
            )
            offspring_haps.append(child_haps)
            offspring_ancs.append(child_paintings)
        
        return (contig_idx, offspring_haps, offspring_ancs)
    
    # --- Parallel path: split offspring into batches for inner workers ---
    # Derive per-batch seeds deterministically from the contig seed so that
    # results are reproducible regardless of the number of inner workers.
    batch_rng = np.random.default_rng(seed)
    
    # Divide offspring into roughly equal batches, one per inner worker
    batch_size = max(1, (num_offspring + inner_workers - 1) // inner_workers)
    
    worker_args = []
    for b_start in range(0, num_offspring, batch_size):
        b_end = min(b_start + batch_size, num_offspring)
        batch_seed = int(batch_rng.integers(0, 2**63))
        worker_args.append((
            (b_start, b_end),
            offspring_parent_indices[b_start:b_end],
            parents, ancestries, site_locs,
            recomb_rate, mutate_rate, batch_seed
        ))
    
    with Pool(processes=min(inner_workers, len(worker_args))) as inner_pool:
        results = inner_pool.map(_process_offspring_batch, worker_args)
    
    # Reassemble in original order
    offspring_haps = [None] * num_offspring
    offspring_ancs = [None] * num_offspring
    
    for ((b_start, b_end), batch_haps, batch_ancs) in results:
        for local_i, global_i in enumerate(range(b_start, b_end)):
            offspring_haps[global_i] = batch_haps[local_i]
            offspring_ancs[global_i] = batch_ancs[local_i]
    
    return (contig_idx, offspring_haps, offspring_ancs)


def simulate_pedigree(founders, site_locs, generation_sizes, 
                      recomb_rate=1e-8, mutate_rate=1e-10, 
                      output_plot="ground_truth_pedigree.png",
                      max_workers=None, parallel=True,
                      num_processes=None, seed=None):
    """
    Simulates a multi-generation pedigree while tracking ANCESTRY.
    
    TWO-LEVEL PARALLELISM (mirrors hierarchical_assembly pattern):
      outer_workers = min(num_contigs, num_processes)
      inner_workers = max(1, num_processes // outer_workers)
    
    This saturates all available cores even with few contigs.  For example,
    5 contigs with num_processes=110 gives 5 outer × 22 inner = 110 cores.
    Single-contig mode also benefits: 1 outer × N inner parallelises across
    offspring batches within the single contig.
    
    SUPPORTS MULTI-CONTIG INPUT:
    - If `founders` is a single list of individuals (pairs), runs single simulation.
    - If `founders` is a list of lists of individuals (contigs), runs multi-contig simulation
      where the pedigree structure (parent-child links) is consistent across all contigs.
    
    Args:
        founders: Founder haplotype pairs (single contig) or list-of-lists (multi-contig).
        site_locs: Site positions array (single) or list of arrays (multi-contig).
        generation_sizes: List of offspring counts per generation, e.g. [20, 100, 200].
        recomb_rate: Per-bp recombination rate.
        mutate_rate: Per-bp mutation rate.
        output_plot: Path to save pedigree plot, or None to skip.
        max_workers: Deprecated — use num_processes instead. If set and num_processes
                     is None, used as outer worker count (backward-compatible).
        parallel: Whether to use parallel processing at all.
        num_processes: Total core budget.  Divided between outer (contig) and inner
                       (offspring) workers.  Defaults to num_contigs if None.
        seed: Optional integer seed for full reproducibility. Seeds both the
              pedigree structure (parent selection) and the meiosis simulation
              (recombination + mutation). If None, uses system entropy (not
              reproducible). Note: the parallel path uses deterministic
              per-batch sub-seeds derived from this seed, so results are
              reproducible regardless of the number of workers.
    """
    # 1. Detect Input Mode (Single vs Multi Contig)
    is_multi_mode = False
    
    if len(founders) > 0 and isinstance(founders[0], list):
        if len(founders[0]) > 0 and isinstance(founders[0][0], list):
            is_multi_mode = True
            
    if is_multi_mode:
        founders_list = founders
        site_locs_list = site_locs
        num_contigs = len(founders_list)
        print(f"Detected Multi-Contig Simulation ({num_contigs} contigs).")
    else:
        founders_list = [founders]
        site_locs_list = [site_locs]
        num_contigs = 1
    
    # Resolve core budget
    # num_processes is the total budget; max_workers is the legacy parameter.
    if num_processes is None:
        if max_workers is not None:
            num_processes = max_workers
        else:
            num_processes = num_contigs
    
    # Two-level split (same pattern as hierarchical_assembly)
    outer_workers = min(num_contigs, num_processes)
    inner_workers = max(1, num_processes // outer_workers)
    
    use_parallel = parallel and num_processes > 1
    
    if use_parallel:
        print(f"Parallelism: {outer_workers} outer (contigs) x {inner_workers} inner (offspring) "
              f"= {outer_workers * inner_workers} total")
        
    # 2. Initialize Ancestries for Founders (Per Contig)
    current_parents_list = founders_list
    current_ancestries_list = []
    
    for c in range(num_contigs):
        c_sites = site_locs_list[c]
        start_pos, end_pos = c_sites[0], c_sites[-1]
        
        c_ancestries = []
        c_founders = founders_list[c]
        
        hap_id_counter = 0
        for _ in c_founders:
            ancestry_pair = []
            for _ in range(2):
                ancestry_pair.append([(start_pos, end_pos, hap_id_counter)])
                hap_id_counter += 1
            c_ancestries.append(ancestry_pair)
        current_ancestries_list.append(c_ancestries)
        
    # 3. Storage for results
    all_individuals_flat_by_contig = [[] for _ in range(num_contigs)]
    all_paintings_flat_by_contig = [[] for _ in range(num_contigs)]
    
    relationships = []
    
    num_initial_founders = len(founders_list[0])
    current_parent_ids = [f"Founder_{i}" for i in range(num_initial_founders)]
    
    master_rng = np.random.default_rng(seed)
    
    # Seed stdlib random for pedigree structure (parent selection).
    # Derive a sub-seed from master_rng so a single seed controls everything.
    if seed is not None:
        pedigree_seed = int(master_rng.integers(0, 2**31))
        random.seed(pedigree_seed)
        print(f"Seeded simulation: master={seed}, pedigree_structure={pedigree_seed}")
    
    # 4. Simulation Loop
    for gen_idx, num_offspring in enumerate(generation_sizes):
        
        gen_name = f"F{gen_idx + 1}"
        print(f"Simulating {gen_name}: {num_offspring} individuals...")
        
        # A. Determine Pedigree Structure (Shared across contigs)
        offspring_parent_indices = []
        next_gen_ids = []
        
        for i in range(num_offspring):
            p1_idx, p2_idx = random.sample(range(len(current_parent_ids)), 2)
            
            parent1_id = current_parent_ids[p1_idx]
            parent2_id = current_parent_ids[p2_idx]
            child_id = f"{gen_name}_{i}"
            
            next_gen_ids.append(child_id)
            offspring_parent_indices.append((p1_idx, p2_idx))
            
            relationships.append({
                'Sample': child_id,
                'Generation': gen_name,
                'Parent1': parent1_id,
                'Parent2': parent2_id
            })
        
        # B. Generate Genetics (Two-level parallel: contigs x offspring)
        if use_parallel:
            worker_args = []
            for c in range(num_contigs):
                contig_data = {
                    'parents': current_parents_list[c],
                    'ancestries': current_ancestries_list[c],
                    'site_locs': site_locs_list[c],
                    'offspring_parent_indices': offspring_parent_indices,
                    'recomb_rate': recomb_rate,
                    'mutate_rate': mutate_rate,
                    'seed': master_rng.integers(0, 2**31),
                    'inner_workers': inner_workers,
                }
                worker_args.append((c, contig_data))
            
            if outer_workers > 1:
                with NoDaemonPool(processes=outer_workers) as pool:
                    results = pool.map(_process_contig_for_generation, worker_args)
            else:
                # Single contig: run the worker directly (inner Pool handles parallelism)
                results = [_process_contig_for_generation(wa) for wa in worker_args]
            
            next_gen_individuals_list = [None] * num_contigs
            next_gen_ancestries_list = [None] * num_contigs
            
            for (contig_idx, offspring_haps, offspring_ancs) in results:
                next_gen_individuals_list[contig_idx] = offspring_haps
                next_gen_ancestries_list[contig_idx] = offspring_ancs
                
                all_individuals_flat_by_contig[contig_idx].extend(offspring_haps)
                all_paintings_flat_by_contig[contig_idx].extend(offspring_ancs)
        
        else:
            next_gen_individuals_list = [[] for _ in range(num_contigs)]
            next_gen_ancestries_list = [[] for _ in range(num_contigs)]
            
            for i, (p1_idx, p2_idx) in enumerate(offspring_parent_indices):
                for c in range(num_contigs):
                    p1_pair = current_parents_list[c][p1_idx]
                    p2_pair = current_parents_list[c][p2_idx]
                    
                    p1_anc = current_ancestries_list[c][p1_idx]
                    p2_anc = current_ancestries_list[c][p2_idx]
                    
                    c_sites = site_locs_list[c]
                    
                    child_haps, child_paintings = create_offspring(
                        p1_pair, p2_pair,
                        p1_anc, p2_anc,
                        c_sites, 
                        recomb_rate=recomb_rate, 
                        mutate_rate=mutate_rate
                    )
                    
                    next_gen_individuals_list[c].append(child_haps)
                    next_gen_ancestries_list[c].append(child_paintings)
                    
                    all_individuals_flat_by_contig[c].append(child_haps)
                    all_paintings_flat_by_contig[c].append(child_paintings)
                
        # Move to next generation
        current_parents_list = next_gen_individuals_list
        current_ancestries_list = next_gen_ancestries_list
        current_parent_ids = next_gen_ids

    # 5. Wrap up
    df = pd.DataFrame(relationships)
    if output_plot is not None:
        plot_ground_truth_pedigree(df, output_file=output_plot)
    
    if is_multi_mode:
        return all_individuals_flat_by_contig, df, all_paintings_flat_by_contig
    else:
        return all_individuals_flat_by_contig[0], df, all_paintings_flat_by_contig[0]


def convert_truth_to_painting_objects(all_paintings_flat, num_workers=8):
    """
    Converts the raw simulation output (lists of tuples) into 
    SamplePainting/PaintedChunk objects compatible with paint_samples.py.
    
    Uses ThreadPoolExecutor + binary search for segment lookup.
    """
    def _process_one_sample(args):
        i, p1, p2 = args
        
        breaks = set()
        for s, e, _ in p1:
            breaks.add(s); breaks.add(e)
        for s, e, _ in p2:
            breaks.add(s); breaks.add(e)
        
        sorted_breaks = sorted(breaks)
        chunks = []
        
        # Build arrays for binary search (faster than linear scan)
        p1_starts = np.array([s for s, e, _ in p1])
        p1_ends = np.array([e for s, e, _ in p1])
        p1_ids = [fid for _, _, fid in p1]
        
        p2_starts = np.array([s for s, e, _ in p2])
        p2_ends = np.array([e for s, e, _ in p2])
        p2_ids = [fid for _, _, fid in p2]
        
        for k in range(len(sorted_breaks) - 1):
            start, end = sorted_breaks[k], sorted_breaks[k + 1]
            if start == end:
                continue
            
            mid = (start + end) / 2
            
            # Find owner in p1 via binary search
            h1_id = -1
            idx = np.searchsorted(p1_starts, mid, side='right') - 1
            if 0 <= idx < len(p1_starts) and p1_starts[idx] <= start and p1_ends[idx] >= end:
                h1_id = p1_ids[idx]
            
            # Find owner in p2 via binary search
            h2_id = -1
            idx = np.searchsorted(p2_starts, mid, side='right') - 1
            if 0 <= idx < len(p2_starts) and p2_starts[idx] <= start and p2_ends[idx] >= end:
                h2_id = p2_ids[idx]
            
            chunks.append(PaintedChunk(
                start=int(start),
                end=int(end),
                hap1=h1_id,
                hap2=h2_id
            ))
        
        return SamplePainting(i, chunks)
    
    args_list = [(i, p1, p2) for i, (p1, p2) in enumerate(all_paintings_flat)]
    
    if num_workers <= 1 or len(args_list) <= 1:
        block_samples = [_process_one_sample(a) for a in args_list]
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            block_samples = list(executor.map(_process_one_sample, args_list))
    
    if not block_samples:
        return None
    
    g_min = block_samples[0].chunks[0].start
    g_max = block_samples[0].chunks[-1].end
    
    return BlockPainting((g_min, g_max), block_samples)


def _process_single_contig_postprocessing(args):
    """
    Worker: process one contig's post-simulation steps (read sampling,
    chunking, probability conversion, truth painting conversion).
    
    All operations are numpy/scipy — they release the GIL, so
    ThreadPoolExecutor gives true parallelism here.
    """
    import analysis_utils
    
    (r_name, offspring_haps, paintings_raw, sites, read_depth, 
     error_rate, snps_per_block, snp_shift, seed) = args
    
    # 1. Convert truth paintings to SamplePainting objects
    true_biological_painting = convert_truth_to_painting_objects(paintings_raw)
    
    # 2. Simulate sequencing reads (vectorized across all individuals)
    reads_rng = np.random.default_rng(seed)
    new_reads_array = read_sample_all_individuals(
        offspring_haps, read_depth, error_rate=error_rate, rng=reads_rng
    )
    
    # 3. Chunk into blocks
    min_pos = sites[0]
    max_pos = sites[-1] + 1
    simd_genomic_data = chunk_up_data(
        sites, new_reads_array,
        min_pos, max_pos, 0, 0,
        use_snp_count=True,
        snps_per_block=snps_per_block,
        snp_shift=snp_shift
    )
    
    # 4. Convert reads to genotype probabilities
    (simd_site_priors, simd_probabalistic_genotypes) = analysis_utils.reads_to_probabilities(
        new_reads_array
    )
    
    return {
        'r_name': r_name,
        'simulated_reads': new_reads_array,
        'simd_genomic_data': simd_genomic_data,
        'simd_probs': simd_probabalistic_genotypes,
        'simd_priors': simd_site_priors,
        'truth_painting': true_biological_painting,
    }


def process_all_contigs_parallel(region_keys, all_offspring_lists, truth_paintings_lists,
                                  sites_list, read_depth=30, error_rate=0.02,
                                  snps_per_block=200, snp_shift=200,
                                  num_workers=None, num_processes=None, seed=None):
    """
    Process all contigs' post-simulation steps in parallel using threads.
    
    Two-level parallelism: outer threads across contigs, inner numpy BLAS
    parallelism via thread count.  All operations are numpy/scipy which
    release the GIL, so ThreadPoolExecutor gives true parallelism.
    
    Args:
        region_keys: List of region names (e.g. ['chr1', 'chr2', ...])
        all_offspring_lists: List of offspring haplotype lists per contig
        truth_paintings_lists: List of raw truth paintings per contig
        sites_list: List of site position arrays per contig
        read_depth: Average sequencing depth (default 30)
        error_rate: Sequencing error rate (default 0.02)
        snps_per_block: SNPs per block for chunking (default 200)
        snp_shift: Block shift for chunking (default 200)
        num_workers: Deprecated — use num_processes instead.
        num_processes: Total core budget. Divided between outer (contig)
                       threads. Defaults to num_contigs if None.
        seed: Optional integer seed for reproducible read simulation.
              Each contig gets a deterministic sub-seed derived from this.
              If None, uses system entropy (not reproducible).
    
    Returns:
        Dict mapping r_name -> dict with keys:
            'simulated_reads', 'simd_genomic_data', 'simd_probs',
            'simd_priors', 'truth_painting'
    """
    n_contigs = len(region_keys)
    
    if num_processes is None:
        if num_workers is not None:
            num_processes = num_workers
        else:
            num_processes = n_contigs
    
    outer_workers = min(n_contigs, num_processes)
    
    # Derive per-contig seeds deterministically from master seed
    seed_rng = np.random.default_rng(seed)
    contig_seeds = [int(seed_rng.integers(0, 2**63)) for _ in range(n_contigs)]
    
    print(f"Post-processing parallelism: {outer_workers} threads across {n_contigs} contigs"
          + (f" (seed={seed})" if seed is not None else ""))
    
    # Build args for each contig
    worker_args = []
    for i, r_name in enumerate(region_keys):
        worker_args.append((
            r_name,
            all_offspring_lists[i],
            truth_paintings_lists[i],
            sites_list[i],
            read_depth,
            error_rate,
            snps_per_block,
            snp_shift,
            contig_seeds[i],
        ))
    
    # Process in parallel — numpy/scipy release GIL so threads work
    results_dict = {}
    
    if outer_workers <= 1 or n_contigs <= 1:
        for args in worker_args:
            result = _process_single_contig_postprocessing(args)
            results_dict[result['r_name']] = result
    else:
        with ThreadPoolExecutor(max_workers=outer_workers) as executor:
            for result in executor.map(_process_single_contig_postprocessing, worker_args):
                results_dict[result['r_name']] = result
    
    return results_dict