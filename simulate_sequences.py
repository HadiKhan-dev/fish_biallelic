"""
File which contains functions that takes as input a list of
probabalistic haplotypes and use them to simulate a multi generation
progeny of founders made up of these haplotypes
"""
import random
import numpy as np
import pickle
import pandas as pd
import warnings
import os

from vcf_data_loader import GenomicData

# Import painting classes for direct conversion
from paint_samples import SamplePainting, PaintedChunk, BlockPainting

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
        concreted.append(np.argmax(hap,axis=1))
    return concreted

def pairup_haps(haps_list,shuffle=False):
    """
    Pair up a list of concrete haps (made up of 0s and 1s)
    """
    haps_copy = pickle.loads(pickle.dumps(haps_list))
    
    if shuffle:
        random.shuffle(haps_copy)
    
    num_pairs = len(haps_list)//2
    haps_paired = []
    
    for i in range(num_pairs):
        first = haps_copy[2*i]
        second = haps_copy[2*i+1]
        haps_paired.append([first,second])
    
    return haps_paired

def get_segments_in_range(source_painting, range_start, range_end):
    """
    Helper to extract and clip ancestry segments that fall within a specific window.
    source_painting: List of (start, end, founder_id)
    """
    result = []
    for (seg_start, seg_end, fid) in source_painting:
        # Calculate overlap
        overlap_start = max(seg_start, range_start)
        overlap_end = min(seg_end, range_end)
        
        # If there is a valid overlap, add it
        if overlap_start < overlap_end:
            result.append((overlap_start, overlap_end, fid))
    return result

def recombine_haps(hap_pair, ancestry_pair, site_locs,
                   recomb_rate=10**-8, mutate_rate=10**-8):
    """
    Simulates meiosis for ONE gamete.
    Tracks both ALLELES (0/1) and ANCESTRY (Founder IDs).
    
    Args:
        hap_pair: [array_hap0, array_hap1] (Alleles)
        ancestry_pair: [list_segments_0, list_segments_1] (Truth Painting)
                       Where list_segments = [(start, end, ID), ...]
    """
    
    recomb_scale = 1.0/recomb_rate
    mutate_scale = 1.0/mutate_rate
    
    assert len(hap_pair[0]) == len(hap_pair[1]), "Length of two haplotypes is different"
    assert len(hap_pair[0]) == len(site_locs), "Different length of hap and of list of site locations"
    
    # Current location tracking
    start_phys = site_locs[0]
    end_phys = site_locs[-1]
    
    cur_loc = start_phys
    cur_loc_index = 0
    
    # Randomly choose starting haplotype (0 or 1)
    using_hap = random.choice([0,1])
    
    final_hap_alleles = []
    final_hap_ancestry = []
    
    while cur_loc <= end_phys:
        # Determine next crossover point (Physical Distance)
        next_break_distance = np.random.exponential(recomb_scale)
        new_loc = cur_loc + np.ceil(next_break_distance)
        
        # Find corresponding indices for ALLELE slicing
        new_loc_index = np.searchsorted(site_locs, new_loc)
        
        # 1. Slice ALLELES
        # Slicing is safe even if new_loc_index > len
        adding = hap_pair[using_hap][cur_loc_index:new_loc_index]
        final_hap_alleles.append(adding)
        
        # 2. Slice ANCESTRY
        # The segment covers [cur_loc, new_loc] physically.
        # We extract relevant pieces from the parent's ancestry.
        segment_phys_end = min(new_loc, end_phys) # Clip to chromosome end
        
        # Get chunks from parent
        parent_segments = get_segments_in_range(
            ancestry_pair[using_hap], 
            cur_loc, 
            segment_phys_end
        )
        final_hap_ancestry.extend(parent_segments)
        
        # Switch haplotype source
        using_hap = 1 - using_hap
        
        cur_loc = new_loc
        cur_loc_index = new_loc_index
        
        if cur_loc_index >= len(site_locs):
            break
    
    # Concatenate alleles
    return_alleles = np.concatenate(final_hap_alleles)
    
    # Truncate alleles if we overshot
    if len(return_alleles) > len(site_locs):
        return_alleles = return_alleles[:len(site_locs)]

    # Apply Mutations (only affects alleles, not ancestry)
    mutation_points = []
    cur_loc = start_phys
    
    while cur_loc <= end_phys:
        next_mutation_distance = np.random.exponential(mutate_scale)
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
                     mutate_rate=10**-8):
    """
    Creates an offspring (Diploid) from two parents.
    Now handles ancestry tracking.
    """
    
    # Gamete from Parent 1
    h1_alleles, h1_ancestry = recombine_haps(
        first_pair, first_ancestry, site_locs, 
        recomb_rate=recomb_rate, mutate_rate=mutate_rate
    )
    
    # Gamete from Parent 2
    h2_alleles, h2_ancestry = recombine_haps(
        second_pair, second_ancestry, site_locs, 
        recomb_rate=recomb_rate, mutate_rate=mutate_rate
    )
    
    return [h1_alleles, h2_alleles], [h1_ancestry, h2_ancestry]

def get_reads_from_sample(hap_pair,read_depth,error_rate=0.02):
    """
    Simulates sequencing an individual made up of a pair of haplotypes 
    up to read_depth average coverage.
    """
    num_sites = len(hap_pair[0])
    num_reads_at_site = np.random.poisson(lam=read_depth,size=num_sites)
    
    site_sum = hap_pair[0]+hap_pair[1]
    
    zeros = np.where(site_sum == 0)[0]
    ones = np.where(site_sum == 1)[0]
    twos = np.where(site_sum == 2)[0]
    
    zero_read_counts = num_reads_at_site[zeros]
    one_read_counts = num_reads_at_site[ones]
    two_read_counts = num_reads_at_site[twos]
    
    zero_draws = np.random.binomial(zero_read_counts,error_rate)
    one_draws = np.random.binomial(one_read_counts,0.5)
    two_draws = np.random.binomial(two_read_counts,1-error_rate)
    
    zero_basics = zero_read_counts-zero_draws
    one_basics = one_read_counts-one_draws
    two_basics = two_read_counts-two_draws
    
    zero_concated = np.column_stack((zero_basics, zero_draws))
    one_concated = np.column_stack((one_basics, one_draws))
    two_concated = np.column_stack((two_basics, two_draws))
    
    full_scaffold = np.zeros((num_sites,2), dtype=int)
    
    full_scaffold[zeros,:] = zero_concated
    full_scaffold[ones,:] = one_concated
    full_scaffold[twos,:] = two_concated
    
    return full_scaffold
    
def read_sample_all_individuals(individual_list,read_depth,error_rate=0.02):
    """
    Takes a list of pairs of haps (individuals) and samples them all.
    """
    sampled = []
    for item in individual_list:
        reads = get_reads_from_sample(item,read_depth,error_rate=error_rate)
        sampled.append(reads)
    return np.array(sampled,dtype=int)

def combine_into_genotype(individual_list):
    """
    Takes as input a list of pairs of haplotypes meant to represent 
    an individual and turn them into a combined likelihood genotype.
    """
    all_list = []
    
    for i in range(len(individual_list)):
        indexing = individual_list[i][0]+individual_list[i][1]
        num_sites = len(indexing)
        
        base_array = [np.array(range(num_sites)), indexing]
        
        scaffold = np.zeros((num_sites,3))
        scaffold[base_array[0], base_array[1]] = 1
        
        all_list.append(scaffold)
    
    return np.array(all_list)
    
def chunk_up_data(positions_list,reads_array,
                  starting_pos,ending_pos,
                  block_size,shift_size,
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
            if end_idx == curr_idx: break
            
            block_positions = positions_slice[curr_idx:end_idx]
            block_reads_array = reads_slice[:, curr_idx:end_idx, :]
            
            total_read_pos = np.sum(block_reads_array, axis=(0,2))
            block_keep_flags = (total_read_pos >= max(min_total_reads, error_rate*num_samples)).astype(int)
            
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
                total_read_pos = np.sum(block_reads_array, axis=(0,2))
                block_keep_flags = (total_read_pos >= max(min_total_reads, error_rate*num_samples)).astype(int)
                
                chunked_positions.append(np.array(block_positions))
                chunked_reads.append(block_reads_array)
                chunked_keep_flags.append(block_keep_flags)
            
            cur_pos = cur_pos + shift_size
            while cur_idx_in_slice < slice_len and positions_slice[cur_idx_in_slice] < cur_pos:
                cur_idx_in_slice += 1
                
    return GenomicData(chunked_positions, chunked_keep_flags, chunked_reads)

# =============================================================================
# NEW: AUTOMATED PEDIGREE SIMULATION & VISUALIZATION
# =============================================================================

def plot_ground_truth_pedigree(relationships_df, output_file="ground_truth_pedigree.png"):
    """
    Plots the Ground Truth Pedigree structure.
    """
    if not HAS_VIS:
        print("Visualization libraries not found.")
        return
        
    # Check if folder path exists in output_file, create if not
    folder = os.path.dirname(output_file)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    G = nx.DiGraph()
    unique_gens = sorted(relationships_df['Generation'].unique())
    
    all_samples = set(relationships_df['Sample'])
    all_parents = set(relationships_df['Parent1'].dropna()) | set(relationships_df['Parent2'].dropna())
    founders = list(all_parents - all_samples)
    
    if founders: unique_gens.insert(0, "Founder")
    
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
        if g == "Founder": return -1
        if g.startswith("F") and g[1:].isdigit(): return int(g[1:])
        return 999
        
    sorted_gens = sorted(unique_gens, key=gen_sort_key)
    
    for x_idx, gen in enumerate(sorted_gens):
        nodes = generations_nodes[gen]
        if not nodes: continue
        
        if x_idx == 0:
            nodes.sort()
        else:
            def get_parent_avg_y(node):
                parents = parents_of.get(node, [])
                if not parents: return 0.5
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
    
    # Labels
    first_gen = sorted_gens[0]
    labels_first = {n: n for n in generations_nodes[first_gen]}
    pos_first = {n: (x-0.03, y) for n, (x,y) in pos.items() if n in labels_first}
    nx.draw_networkx_labels(G, pos_first, labels=labels_first, font_size=10, horizontalalignment='right')
    
    labels_others = {}
    for gen in sorted_gens[1:]:
        for n in generations_nodes[gen]: labels_others[n] = n
    pos_others = {n: (x+0.03, y) for n, (x,y) in pos.items() if n in labels_others}
    nx.draw_networkx_labels(G, pos_others, labels=labels_others, font_size=8, horizontalalignment='left')

    patches = [mpatches.Patch(color=gen_colors[g], label=g) for g in sorted_gens]
    plt.legend(handles=patches, loc='upper right')
    
    plt.title("Ground Truth Pedigree")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()

def simulate_pedigree(founders, site_locs, generation_sizes, 
                      recomb_rate=1e-8, mutate_rate=1e-10, 
                      output_plot="ground_truth_pedigree.png"):
    """
    Simulates a multi-generation pedigree while tracking ANCESTRY.
    
    SUPPORTS MULTI-CONTIG INPUT:
    - If `founders` is a single list of individuals (pairs), runs single simulation.
    - If `founders` is a list of lists of individuals (contigs), runs multi-contig simulation
      where the pedigree structure (parent-child links) is consistent across all contigs.
    
    Returns:
        If Single Contig:
             (all_offspring, relationships_df, all_paintings)
        If Multi Contig:
             (all_offspring_by_contig, relationships_df, all_paintings_by_contig)
    """
    
    # 1. Detect Input Mode (Single vs Multi Contig)
    is_multi_mode = False
    
    # Check if founders is [ [ind1, ind2...], [ind1, ind2...] ]
    # A single individual is a list [hap0, hap1].
    # So single mode: founders[0] is [h0, h1]. type(founders[0][0]) is ndarray.
    # Multi mode: founders[0] is [ [h0, h1]... ]. type(founders[0][0]) is list.
    
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
        
    # 2. Initialize Ancestries for Founders (Per Contig)
    # current_parents_list[c] = List of parents for contig c
    current_parents_list = founders_list
    
    current_ancestries_list = []
    
    for c in range(num_contigs):
        # Initialize ancestry for this contig's founders
        # ID is just an integer index. Founder i on Contig 0 and Founder i on Contig 1 have same ID.
        c_sites = site_locs_list[c]
        start_pos, end_pos = c_sites[0], c_sites[-1]
        
        c_ancestries = []
        c_founders = founders_list[c] # List of [h0, h1]
        
        hap_id_counter = 0
        for _ in c_founders:
            ancestry_pair = []
            for _ in range(2):
                ancestry_pair.append([(start_pos, end_pos, hap_id_counter)])
                hap_id_counter += 1
            c_ancestries.append(ancestry_pair)
        current_ancestries_list.append(c_ancestries)
        
    # 3. Storage for results
    # For multi-contig, we store list of lists
    all_individuals_flat_by_contig = [[] for _ in range(num_contigs)]
    all_paintings_flat_by_contig = [[] for _ in range(num_contigs)]
    
    # Add founders to history?
    # Usually we only return offspring, or user requests.
    # The original function accumulated `child_haps` into `all_individuals_flat`.
    # It did NOT add founders to `all_individuals_flat` initially? 
    # Let's check original logic: "all_individuals_flat.append(child_haps)" inside loop.
    # So founders are NOT in the output list, only offspring. Correct.
    
    relationships = []
    
    num_initial_founders = len(founders_list[0])
    current_parent_ids = [f"Founder_{i}" for i in range(num_initial_founders)]
    
    # 4. Simulation Loop
    for gen_idx, num_offspring in enumerate(generation_sizes):
        
        gen_name = f"F{gen_idx + 1}"
        print(f"Simulating {gen_name}: {num_offspring} individuals...")
        
        next_gen_individuals_list = [[] for _ in range(num_contigs)]
        next_gen_ancestries_list = [[] for _ in range(num_contigs)]
        next_gen_ids = []
        
        for i in range(num_offspring):
            # A. Determine Pedigree Structure (Shared across contigs)
            p1_idx, p2_idx = random.sample(range(len(current_parent_ids)), 2)
            
            parent1_id = current_parent_ids[p1_idx]
            parent2_id = current_parent_ids[p2_idx]
            child_id = f"{gen_name}_{i}"
            
            next_gen_ids.append(child_id)
            
            relationships.append({
                'Sample': child_id,
                'Generation': gen_name,
                'Parent1': parent1_id,
                'Parent2': parent2_id
            })
            
            # B. Generate Genetics (Independent per contig)
            for c in range(num_contigs):
                # Get parents for this contig
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
    plot_ground_truth_pedigree(df, output_file=output_plot)
    
    if is_multi_mode:
        return all_individuals_flat_by_contig, df, all_paintings_flat_by_contig
    else:
        return all_individuals_flat_by_contig[0], df, all_paintings_flat_by_contig[0]

def convert_truth_to_painting_objects(all_paintings_flat):
    """
    Converts the raw simulation output (lists of tuples) into 
    SamplePainting/PaintedChunk objects compatible with paint_samples.py.
    """
    block_samples = []
    
    for i, (p1, p2) in enumerate(all_paintings_flat):
        # We need to merge two haplotype paintings into one 'SamplePainting'.
        # This requires chopping them into common intervals where state is constant.
        
        # 1. Collect all break points
        breaks = set()
        for s, e, _ in p1:
            breaks.add(s); breaks.add(e)
        for s, e, _ in p2:
            breaks.add(s); breaks.add(e)
            
        sorted_breaks = sorted(list(breaks))
        chunks = []
        
        # 2. Iterate through intervals
        for k in range(len(sorted_breaks) - 1):
            start, end = sorted_breaks[k], sorted_breaks[k+1]
            if start == end: continue
            
            # Find owner in p1
            h1_id = -1
            for (s, e, fid) in p1:
                if s <= start and e >= end:
                    h1_id = fid
                    break
            
            # Find owner in p2
            h2_id = -1
            for (s, e, fid) in p2:
                if s <= start and e >= end:
                    h2_id = fid
                    break
                    
            chunks.append(PaintedChunk(
                start=int(start),
                end=int(end),
                hap1=h1_id,
                hap2=h2_id
            ))
            
        block_samples.append(SamplePainting(i, chunks))
        
    # We create a dummy BlockPainting container
    if not block_samples: return None
    
    # Find global min/max
    g_min = block_samples[0].chunks[0].start
    g_max = block_samples[0].chunks[-1].end
    
    return BlockPainting((g_min, g_max), block_samples)