import os

# FORCE NUMPY TO USE 1 THREAD PER PROCESS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import time
import warnings
import platform
import importlib
import math
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict
from multiprocess import Pool


import vcf_data_loader
import analysis_utils
import hap_statistics
import block_haplotypes
import block_linking_naive
import block_linking_em
import simulate_sequences
import hmm_matching
import viterbi_likelihood_calculator
import beam_search_core
import hierarchical_assembly
import paint_samples
import pedigree_inference
import phase_correction

warnings.filterwarnings("ignore")
np.seterr(divide='ignore',invalid="ignore")

if platform.system() != "Windows":
    os.nice(15)
    print(f"Main process ({os.getpid()}) niceness set to: {os.nice(0)}")


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#%%
vcf_path = "./fish_vcf/AsAc.AulStuGenome.biallelic.bcf.gz"

# Define the regions you want to use for inference.
# Each entry requires a unique name, the VCF contig name, and the block range.
regions_config = [
    {"contig": "chr1", "start": 0, "end": 600}]

block_size = 100000
shift_size = 50000

# Dictionary to store the processed data for every region
# Structure: { region_name : { 'genomic_data': ..., 'block_results': ..., 'naive_long_haps': ... } }
multi_contig_results = {}

total_start = time.time()

for region in regions_config:
    print(f"\n" + "="*60)
    print(f"PROCESSING REGION: ({region['contig']} blocks {region['start']}-{region['end']})")
    print("="*60)
    
    # 1. Load Data
    start = time.time()
    genomic_data = vcf_data_loader.cleanup_block_reads_list(
        vcf_path, 
        region['contig'],
        start_block_idx=region['start'],
        end_block_idx=region['end'],
        block_size=block_size,
        shift_size=shift_size,
        num_processes=16
    )
    print(f"  [Loader] Loaded {len(genomic_data)} blocks in {time.time() - start:.2f}s")

    # 2. Run Haplotype Discovery
    start = time.time()
    block_results = block_haplotypes.generate_all_block_haplotypes(genomic_data)

    # Remove empty blocks (no SNPs found) to prevent linker crashes
    valid_blocks = [b for b in block_results if len(b.positions) > 0]
    block_results = block_haplotypes.BlockResults(valid_blocks)
    
    print(f"  [Discovery] Haplotypes generated in {time.time() - start:.2f}s")
    # 3. Run Naive Linker (to get long templates for simulation)
    start = time.time()
    (naive_blocks, naive_long_haps) = block_linking_naive.generate_long_haplotypes_naive(
        block_results, 
        num_long_haps=6
    )
    print(f"  [Naive Linker] Chained {len(naive_long_haps[1])} haps in {time.time() - start:.2f}s")
    
    # Store results
    multi_contig_results[region['contig']] = {
        "genomic_data": genomic_data,
        "block_results": block_results,
        "naive_long_haps": naive_long_haps # Tuple: (sites, haps_list)
    }

print(f"\nAll regions processed in {time.time() - total_start:.2f}s")
#%%
start = time.time()
# Create a specific folder for simulation outputs
output_dir = "results_simulation"
os.makedirs(output_dir, exist_ok=True)

# 1. Prepare Founders and Sites for ALL regions
founders_list = []
sites_list = []
region_keys = []

# Collect data from the previous step
for r_name, data in multi_contig_results.items():
    # naive_long_haps is tuple: (sites_array, haps_list)
    sites, haps_data = data['naive_long_haps']
    
    # Convert probabilistic haplotypes to concrete 0/1 founders
    concrete_haps = simulate_sequences.concretify_haps(haps_data)
    parents = simulate_sequences.pairup_haps(concrete_haps)
    
    founders_list.append(parents)
    sites_list.append(sites)
    region_keys.append(r_name)

# 2. Run Multi-Contig Simulation (PARALLELIZED)
# This generates a SINGLE pedigree (relationships) applied to ALL contigs independently
generation_sizes = [10, 100, 200]
print(f"Running Multi-Contig Simulation for {len(region_keys)} regions...")

# Calculate mutation rate for ~1% of SNPs per generation (stress testing)
# mutate_rate = 0.01 * num_snps / chromosome_length
# For typical data: ~1e-5 gives roughly 1% mutation rate
# Use 1e-10 for essentially no mutations (original behavior)
STRESS_TEST_MUTATIONS = False  # Set to True for 1% mutation stress test
if STRESS_TEST_MUTATIONS:
    # Approximate: assume ~1 SNP per 100bp on average
    mutate_rate = 1e-5  # ~1% of SNPs mutated per generation
    print(f"STRESS TEST MODE: Using mutation rate {mutate_rate} (~1% per generation)")
else:
    mutate_rate = 1e-10  # Essentially no mutations
    print(f"Normal mode: Using mutation rate {mutate_rate} (minimal mutations)")

# Returns lists-of-lists (one sub-list per contig)
all_offspring_lists, truth_pedigree, truth_paintings_lists = simulate_sequences.simulate_pedigree(
    founders_list, 
    sites_list, 
    generation_sizes, 
    recomb_rate=5e-8, 
    mutate_rate=mutate_rate,
    output_plot=os.path.join(output_dir, "ground_truth_pedigree.png"),
    parallel=True,       # Enable parallel processing across contigs
    max_workers=None     # Use all available cores (or set to specific number)
)

print(f"Pedigree simulation completed in {time.time()-start:.1f}s")

# 3. Save Truth for later validation
truth_csv_path = os.path.join(output_dir, "ground_truth_pedigree.csv")
truth_pedigree.to_csv(truth_csv_path, index=False)
print(f"Ground Truth Pedigree data saved to '{truth_csv_path}'")

sample_names = truth_pedigree['Sample'].tolist()

# 4. Process Each Contig Individually (Visualization, Sequencing, Chunking)
read_depth = 2000

for i, r_name in enumerate(region_keys):
    print(f"\nProcessing Simulated Data for Region: {r_name}")
    
    # Unpack specific results for this contig
    offspring_haps = all_offspring_lists[i]   # List of [hapA, hapB] for this contig
    paintings_raw = truth_paintings_lists[i]  # List of raw painting tuples
    sites = sites_list[i]                     # Site locations
    
    # A. Visualize Truth
    true_biological_painting = simulate_sequences.convert_truth_to_painting_objects(paintings_raw)
    
    paint_samples.plot_population_painting(
        true_biological_painting,
        output_file=os.path.join(output_dir, f"{r_name}_biological_truth.png"),
        title=f"Biological Truth - {r_name}",
        sample_names=sample_names,
        figsize_width=20,
        row_height_per_sample=0.25
    )
    
    # B. Simulate Sequencing (Reads)
    # Generates reads for this specific contig
    new_reads_array = simulate_sequences.read_sample_all_individuals(
        offspring_haps, read_depth, error_rate=0.02
    )
    
    # C. Chunk Data (Create SIMD/Proxy data for inference)
    # We dynamically find the start/end positions
    min_pos = sites[0]
    max_pos = sites[-1] + 1
    
    simd_genomic_data = simulate_sequences.chunk_up_data(
        sites, 
        new_reads_array,
        min_pos, max_pos, 
        0, 0, # Physical block size ignored when using snp_count
        use_snp_count=True,
        snps_per_block=200,
        snp_shift=200 # Disjoint blocks
    )
    
    # D. Calculate Probabilities (Genotypes and Priors)
    (simd_site_priors, simd_probabalistic_genotypes) = analysis_utils.reads_to_probabilities(new_reads_array)
    
    # E. Store results back into the dictionary for downstream steps
    # We add new keys to the existing region dictionary
    multi_contig_results[r_name]['simulated_reads'] = new_reads_array
    multi_contig_results[r_name]['simd_genomic_data'] = simd_genomic_data
    multi_contig_results[r_name]['simd_probs'] = simd_probabalistic_genotypes
    multi_contig_results[r_name]['simd_priors'] = simd_site_priors
    
    # Also store the truth painting object for validation later
    multi_contig_results[r_name]['truth_painting'] = true_biological_painting

print("\nSimulation, Sequencing, and Chunking complete for all regions.")
print(f"Total time: {time.time()-start:.1f}s")
#%%
# ==========================================================================
# STEP: DISCOVER BLOCK HAPLOTYPES FROM SIMULATED READS
# ==========================================================================
print(f"\n{'='*60}")
print("Discovering Block Haplotypes from Simulated Reads")
print(f"{'='*60}")

start = time.time()

for r_name in region_keys:
    print(f"\n  Processing {r_name}...")
    
    simd_genomic_data = multi_contig_results[r_name]['simd_genomic_data']
    
    # Discover block haplotypes from simulated reads
    simd_block_results = block_haplotypes.generate_all_block_haplotypes(
        simd_genomic_data,
        uniqueness_threshold_percent=1.0,
        diff_threshold_percent=0.5,
        wrongness_threshold=1.0,
        num_processes=16
    )
    
    # Remove empty blocks
    valid_blocks = [b for b in simd_block_results if len(b.positions) > 0]
    simd_block_results = block_haplotypes.BlockResults(valid_blocks)
    
    # Store
    multi_contig_results[r_name]['simd_block_results'] = simd_block_results
    
    # Summary stats
    hap_counts = [len(b.haplotypes) for b in valid_blocks]
    print(f"    {len(valid_blocks)} blocks, haps/block: min={min(hap_counts)}, max={max(hap_counts)}, mean={np.mean(hap_counts):.1f}")

print(f"\nBlock haplotype discovery complete in {time.time()-start:.1f}s")


#%% ==========================================================================
# STEP: VALIDATE BLOCK HAPLOTYPES AGAINST GROUND TRUTH
# ==========================================================================
print(f"\n{'='*60}")
print("Validating Discovered Block Haplotypes Against Ground Truth")
print(f"{'='*60}")

def validate_block_haplotypes(simd_block_results, orig_sites, orig_haps_concrete):
    """
    Compare discovered block haplotypes against true founder haplotypes.
    
    Returns per-block statistics and overall summary.
    """
    # Build site -> index lookup for original haplotypes
    orig_site_to_idx = {s: i for i, s in enumerate(orig_sites)}
    
    block_stats = []
    
    for block in simd_block_results:
        block_positions = block.positions
        block_haps = block.haplotypes  # Dict {idx: hap_array}
        
        if len(block_positions) == 0:
            continue
        
        # Find indices in original data for this block's positions
        common_indices = []
        block_indices = []
        for bi, pos in enumerate(block_positions):
            if pos in orig_site_to_idx:
                common_indices.append(orig_site_to_idx[pos])
                block_indices.append(bi)
        
        if len(common_indices) == 0:
            continue
        
        # Extract true founder haplotypes at these positions
        true_at_block = [h[common_indices] for h in orig_haps_concrete]
        num_true_founders = len(true_at_block)
        
        # Extract discovered haplotypes at these positions
        # Block haps are probabilistic (n_sites, 2) - concretify them
        discovered_at_block = []
        for hap_idx, hap_arr in block_haps.items():
            # hap_arr is (n_sites, 2) probabilities
            concrete = np.argmax(hap_arr, axis=1)
            discovered_at_block.append(concrete[block_indices])
        
        num_discovered = len(discovered_at_block)
        
        # For each true founder, find best matching discovered haplotype
        true_to_best_discovered = []
        for ti, true_h in enumerate(true_at_block):
            best_diff = 100.0
            best_idx = -1
            for di, disc_h in enumerate(discovered_at_block):
                diff = np.mean(true_h != disc_h) * 100
                if diff < best_diff:
                    best_diff = diff
                    best_idx = di
            true_to_best_discovered.append((ti, best_idx, best_diff))
        
        # For each discovered, find best matching true founder
        discovered_to_best_true = []
        for di, disc_h in enumerate(discovered_at_block):
            best_diff = 100.0
            best_idx = -1
            for ti, true_h in enumerate(true_at_block):
                diff = np.mean(true_h != disc_h) * 100
                if diff < best_diff:
                    best_diff = diff
                    best_idx = ti
            discovered_to_best_true.append((di, best_idx, best_diff))
        
        # Count how many true founders are "found" (match < 2% diff)
        founders_found = sum(1 for _, _, diff in true_to_best_discovered if diff < 2.0)
        
        # Average best-match error
        avg_true_match_error = np.mean([diff for _, _, diff in true_to_best_discovered])
        avg_disc_match_error = np.mean([diff for _, _, diff in discovered_to_best_true])
        
        block_stats.append({
            'start_pos': block_positions[0],
            'n_sites': len(common_indices),
            'n_true': num_true_founders,
            'n_discovered': num_discovered,
            'founders_found': founders_found,
            'avg_true_match_err': avg_true_match_error,
            'avg_disc_match_err': avg_disc_match_error,
            'true_matches': true_to_best_discovered,
            'disc_matches': discovered_to_best_true
        })
    
    return block_stats


for r_name in region_keys:
    print(f"\n{r_name}:")
    
    simd_block_results = multi_contig_results[r_name]['simd_block_results']
    orig_sites, orig_haps = multi_contig_results[r_name]['naive_long_haps']
    
    # Concretify original haplotypes
    orig_haps_concrete = simulate_sequences.concretify_haps(orig_haps)
    
    # Run validation
    block_stats = validate_block_haplotypes(simd_block_results, orig_sites, orig_haps_concrete)
    
    # Store for later use
    multi_contig_results[r_name]['block_validation_stats'] = block_stats
    
    # Summary statistics
    n_blocks = len(block_stats)
    avg_discovered = np.mean([bs['n_discovered'] for bs in block_stats])
    avg_founders_found = np.mean([bs['founders_found'] for bs in block_stats])
    avg_true_err = np.mean([bs['avg_true_match_err'] for bs in block_stats])
    avg_disc_err = np.mean([bs['avg_disc_match_err'] for bs in block_stats])
    
    # Count blocks where all founders found
    all_found_count = sum(1 for bs in block_stats if bs['founders_found'] == bs['n_true'])
    
    print(f"  Blocks analyzed: {n_blocks}")
    print(f"  True founders per block: {block_stats[0]['n_true']}")
    print(f"  Avg discovered haps per block: {avg_discovered:.1f}")
    print(f"  Avg founders found per block (<2% diff): {avg_founders_found:.1f} / {block_stats[0]['n_true']}")
    print(f"  Blocks with ALL founders found: {all_found_count} / {n_blocks} ({100*all_found_count/n_blocks:.1f}%)")
    print(f"  Avg best-match error (true->discovered): {avg_true_err:.2f}%")
    print(f"  Avg best-match error (discovered->true): {avg_disc_err:.2f}%")
    
    # Distribution of founders found
    founders_found_dist = {}
    for bs in block_stats:
        ff = bs['founders_found']
        founders_found_dist[ff] = founders_found_dist.get(ff, 0) + 1
    
    print(f"  Founders found distribution:")
    for k in sorted(founders_found_dist.keys()):
        print(f"    {k} founders: {founders_found_dist[k]} blocks ({100*founders_found_dist[k]/n_blocks:.1f}%)")

print(f"\n{'='*60}")
print("Block Haplotype Validation Complete")
print(f"{'='*60}")
#%% 
# ==========================================================================
# RUN: Hierarchical Assembly (Level 1) - Parallel Batches
# ==========================================================================
start = time.time()

for r_name in region_keys:
    print(f"\n  Processing {r_name}...")
    
    simd_block_results = multi_contig_results[r_name]['simd_block_results']
    global_probs = multi_contig_results[r_name]['global_probs']
    global_sites = multi_contig_results[r_name]['global_sites']
    
    print(f"    Input: {len(simd_block_results)} blocks of 200 SNPs each")
    
    # Run hierarchical step (internally parallelized across batches)
    super_blocks = hierarchical_assembly.run_hierarchical_step(
        input_blocks=simd_block_results,
        global_probs=global_probs,
        global_sites=global_sites,
        batch_size=10,
        # Linking parameters
        use_hmm_linking=False,  # Use block_linking_em
        # Search parameters
        beam_width=200,
        # Selection parameters
        max_founders=12,
        complexity_penalty_scale=0.1,
        switch_cost_scale=0.1,
        pruning_switch_cost_scale=0.05,
        use_standard_bic=False,
        # Memory safety
        max_sites_for_linking=2000,
        # Parallelization
        num_processes=16
    )
    
    # Store results
    multi_contig_results[r_name]['super_blocks_L1'] = super_blocks
    
    # Summary
    hap_counts = [len(b.haplotypes) for b in super_blocks]
    total_sites = sum(len(b.positions) for b in super_blocks)
    print(f"\n    Output: {len(super_blocks)} super-blocks")
    print(f"    Total sites: {total_sites}")
    print(f"    Haps per super-block: min={min(hap_counts)}, max={max(hap_counts)}, mean={np.mean(hap_counts):.1f}")

print(f"\nHierarchical Assembly (Level 1) complete in {time.time()-start:.1f}s")

#%% ==========================================================================
# VALIDATE: Compare Level 1 Super Blocks against True Founders
# ==========================================================================
print(f"\n{'='*60}")
print("Validating Level 1 Super Blocks against Ground Truth")
print(f"{'='*60}")

for r_name in region_keys:
    print(f"\n  Processing {r_name}...")
    
    super_blocks = multi_contig_results[r_name]['super_blocks_L1']
    
    # Get ground truth founder haplotypes
    orig_sites, orig_haps = multi_contig_results[r_name]['naive_long_haps']
    orig_haps_concrete = simulate_sequences.concretify_haps(orig_haps)
    orig_site_to_idx = {s: i for i, s in enumerate(orig_sites)}
    
    num_true_founders = len(orig_haps_concrete)
    
    # Track results
    founders_found_per_block = []
    all_best_errors = []
    
    for block_idx, block in enumerate(super_blocks):
        positions = block.positions
        
        # Get true founder haplotypes at this block's positions
        true_founders_at_block = []
        for f_idx in range(num_true_founders):
            founder_vals = np.array([orig_haps_concrete[f_idx][orig_site_to_idx[pos]] 
                                      for pos in positions])
            true_founders_at_block.append(founder_vals)
        
        # Compare each discovered haplotype against all true founders
        discovered_haps = []
        for hap_idx, hap_arr in block.haplotypes.items():
            # Convert probabilistic to concrete
            if hap_arr.ndim > 1:
                hap_concrete = np.argmax(hap_arr, axis=1)
            else:
                hap_concrete = hap_arr
            discovered_haps.append(hap_concrete)
        
        # For each true founder, find the best matching discovered haplotype
        founders_found = 0
        block_errors = []
        
        for f_idx, true_founder in enumerate(true_founders_at_block):
            best_error = 100.0
            for disc_hap in discovered_haps:
                error = np.mean(disc_hap != true_founder) * 100
                if error < best_error:
                    best_error = error
            
            block_errors.append(best_error)
            if best_error < 2.0:  # Threshold for "found"
                founders_found += 1
        
        founders_found_per_block.append(founders_found)
        all_best_errors.extend(block_errors)
    
    # Summary statistics
    blocks_with_all = sum(1 for f in founders_found_per_block if f == num_true_founders)
    avg_found = np.mean(founders_found_per_block)
    avg_error = np.mean(all_best_errors)
    
    print(f"\n  Results for {r_name}:")
    print(f"    Super blocks analyzed: {len(super_blocks)}")
    print(f"    True founders: {num_true_founders}")
    print(f"    Avg founders found per block: {avg_found:.1f} / {num_true_founders}")
    print(f"    Blocks with ALL founders found: {blocks_with_all} / {len(super_blocks)} ({100*blocks_with_all/len(super_blocks):.1f}%)")
    print(f"    Avg best-match error: {avg_error:.2f}%")
    
    # Distribution of founders found
    from collections import Counter
    dist = Counter(founders_found_per_block)
    print(f"    Founders found distribution:")
    for k in sorted(dist.keys()):
        print(f"      {k} founders: {dist[k]} blocks ({100*dist[k]/len(super_blocks):.1f}%)")
    
    # Show worst blocks
    worst_blocks = sorted(enumerate(founders_found_per_block), key=lambda x: x[1])[:5]
    print(f"\n    Worst 5 blocks:")
    for block_idx, found in worst_blocks:
        n_haps = len(super_blocks[block_idx].haplotypes)
        print(f"      Block {block_idx}: {found}/{num_true_founders} founders found, {n_haps} discovered haps")
    
    # Store validation results
    multi_contig_results[r_name]['L1_validation'] = {
        'founders_found_per_block': founders_found_per_block,
        'blocks_with_all': blocks_with_all,
        'avg_found': avg_found,
        'avg_error': avg_error
    }
