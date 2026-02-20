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
import chimera_resolution
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
    {"contig": "chr1", "start": 0, "end": 1000},]

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
    #time.sleep(60) #Give machine time to cool down

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
generation_sizes = [20, 100, 200]
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
read_depth = 4

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

#%%
# ==========================================================================
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
    global_probs = multi_contig_results[r_name]['simd_probs']
    global_sites = multi_contig_results[r_name]['naive_long_haps'][0]

    print(f"    Input: {len(simd_block_results)} blocks of 200 SNPs each")
    
    super_blocks = hierarchical_assembly.run_hierarchical_step(
        input_blocks=simd_block_results,
        global_probs=global_probs,
        global_sites=global_sites,
        batch_size=10,
        use_hmm_linking=False,
        beam_width=200,
        max_founders=12,
        max_sites_for_linking=2000,
        cc_scale=0.2,
        num_processes=16
    )
    
    multi_contig_results[r_name]['super_blocks_L1'] = super_blocks
    
    hap_counts = [len(b.haplotypes) for b in super_blocks]
    total_sites = sum(len(b.positions) for b in super_blocks)
    print(f"\n    Output: {len(super_blocks)} super-blocks")
    print(f"    Total sites: {total_sites}")
    print(f"    Haps per super-block: min={min(hap_counts)}, max={max(hap_counts)}, mean={np.mean(hap_counts):.1f}")

print(f"\nHierarchical Assembly (Level 1) complete in {time.time()-start:.1f}s")

#%% 
# ==========================================================================
# VALIDATE: Compare Level 1 Super Blocks against True Founders
# ==========================================================================
print(f"\n{'='*60}")
print("Validating Level 1 Super Blocks against Ground Truth")
print(f"{'='*60}")

for r_name in region_keys:
    super_blocks = multi_contig_results[r_name]['super_blocks_L1']
    
    orig_sites, orig_haps = multi_contig_results[r_name]['naive_long_haps']
    orig_haps_concrete = simulate_sequences.concretify_haps(orig_haps)
    orig_site_to_idx = {s: i for i, s in enumerate(orig_sites)}
    
    num_true_founders = len(orig_haps_concrete)
    
    total_discovered = 0
    total_good = 0
    total_chimeras = 0
    blocks_with_all_founders = 0
    
    chimera_details = []
    
    for block_idx, block in enumerate(super_blocks):
        positions = block.positions
        
        true_at_block = []
        for f_idx in range(num_true_founders):
            founder_vals = np.array([orig_haps_concrete[f_idx][orig_site_to_idx[pos]] 
                                      for pos in positions])
            true_at_block.append(founder_vals)
        
        founders_found = 0
        for f_idx, tf in enumerate(true_at_block):
            best_error = 100
            for h_idx, hap in block.haplotypes.items():
                if hap.ndim > 1:
                    hap = np.argmax(hap, axis=1)
                error = np.mean(hap != tf) * 100
                if error < best_error:
                    best_error = error
            if best_error < 2.0:
                founders_found += 1
        
        for h_idx, hap in block.haplotypes.items():
            if hap.ndim > 1:
                hap = np.argmax(hap, axis=1)
            errors = [np.mean(hap != tf) * 100 for tf in true_at_block]
            best_f = np.argmin(errors)
            best_error = errors[best_f]
            total_discovered += 1
            if best_error < 2.0:
                total_good += 1
            else:
                total_chimeras += 1
                chimera_details.append({
                    'block': block_idx,
                    'hap': h_idx,
                    'best_f': best_f,
                    'error': best_error,
                    'n_sites': len(positions)
                })
        
        if founders_found == num_true_founders:
            blocks_with_all_founders += 1
    
    print(f"\nResults:")
    print(f"  L1 super-blocks: {len(super_blocks)}")
    print(f"  Blocks with ALL founders: {blocks_with_all_founders} / {len(super_blocks)} ({100*blocks_with_all_founders/len(super_blocks):.1f}%)")
    print(f"  Total haplotypes: {total_discovered}")
    print(f"  Good haplotypes (<2% error): {total_good}")
    print(f"  Chimeras (>2% error): {total_chimeras}")
    
    if chimera_details:
        print(f"\nChimera details:")
        for c in sorted(chimera_details, key=lambda x: x['error'], reverse=True):
            print(f"  Block {c['block']}, H{c['hap']}: F{c['best_f']} @ {c['error']:.2f}%")

#%% 
# ==========================================================================
# Level 2 Hierarchical Assembly
# ==========================================================================

print("="*60)
print("Level 2 Hierarchical Assembly")
print("="*60)

start_time = time.time()

for r_name in region_keys:
    print(f"\n  Processing {r_name}...")
    
    super_blocks_L1 = multi_contig_results[r_name]['super_blocks_L1']
    
    print(f"    Input: {len(super_blocks_L1)} L1 super-blocks")
    print(f"    Sites per L1 block: {[len(b.positions) for b in super_blocks_L1[:5]]}...")
    
    global_probs = multi_contig_results[r_name]['simd_probs']
    global_sites = multi_contig_results[r_name]['naive_long_haps'][0]
    
    super_blocks_L2 = hierarchical_assembly.run_hierarchical_step(
        super_blocks_L1,
        global_probs,
        global_sites,
        batch_size=10,
        use_hmm_linking=True,
        recomb_rate=5e-8,
        beam_width=200,
        max_founders=12,
        cc_scale=0.2,
        num_processes=16,
        n_generations=3,
        verbose=False
    )
    
    multi_contig_results[r_name]['super_blocks_L2'] = super_blocks_L2
    
    haps_per_block = [len(b.haplotypes) for b in super_blocks_L2]
    total_sites = sum(len(b.positions) for b in super_blocks_L2)
    print(f"\n    Output: {len(super_blocks_L2)} L2 super-blocks")
    print(f"    Total sites: {total_sites}")
    print(f"    Haps per super-block: min={min(haps_per_block)}, max={max(haps_per_block)}, mean={np.mean(haps_per_block):.1f}")

print(f"\nHierarchical Assembly (Level 2) complete in {time.time()-start_time:.1f}s")

#%%
print("="*60)
print("Validating Level 2 Super Blocks against Ground Truth")
print("="*60)

super_blocks_L2 = multi_contig_results['chr1']['super_blocks_L2']

orig_sites, orig_haps = multi_contig_results['chr1']['naive_long_haps']
orig_haps_concrete = simulate_sequences.concretify_haps(orig_haps)
orig_site_to_idx = {s: idx for idx, s in enumerate(orig_sites)}

total_discovered = 0
total_good = 0
total_chimeras = 0
blocks_with_all_founders = 0

for block_idx, block in enumerate(super_blocks_L2):
    positions = block.positions
    
    true_at_block = []
    for f_idx in range(len(orig_haps_concrete)):
        founder_vals = np.array([orig_haps_concrete[f_idx][orig_site_to_idx[pos]] 
                                  for pos in positions])
        true_at_block.append(founder_vals)
    
    founders_found = 0
    for f_idx, tf in enumerate(true_at_block):
        best_error = 100
        for h_idx, hap in block.haplotypes.items():
            if hap.ndim > 1:
                hap = np.argmax(hap, axis=1)
            error = np.mean(hap != tf) * 100
            if error < best_error:
                best_error = error
        if best_error < 2.0:
            founders_found += 1
    
    for h_idx, hap in block.haplotypes.items():
        if hap.ndim > 1:
            hap = np.argmax(hap, axis=1)
        errors = [np.mean(hap != tf) * 100 for tf in true_at_block]
        best_error = min(errors)
        total_discovered += 1
        if best_error < 2.0:
            total_good += 1
        else:
            total_chimeras += 1
    
    if founders_found == 6:
        blocks_with_all_founders += 1

print(f"\nResults:")
print(f"  L2 super-blocks: {len(super_blocks_L2)}")
print(f"  Blocks with ALL founders: {blocks_with_all_founders} / {len(super_blocks_L2)} ({100*blocks_with_all_founders/len(super_blocks_L2):.1f}%)")
print(f"  Total haplotypes: {total_discovered}")
print(f"  Good haplotypes (<2% error): {total_good}")
print(f"  Chimeras (>2% error): {total_chimeras}")

#%% 
# ==========================================================================
# Level 3 Hierarchical Assembly
# ==========================================================================
print("="*60)
print("Level 3 Hierarchical Assembly")
print("="*60)

start_time = time.time()

for r_name in region_keys:
    print(f"\n  Processing {r_name}...")
    
    super_blocks_L2 = multi_contig_results[r_name]['super_blocks_L2']
    
    print(f"    Input: {len(super_blocks_L2)} L2 super-blocks")
    print(f"    Sites per L2 block: {[len(b.positions) for b in super_blocks_L2]}")
    
    global_probs = multi_contig_results[r_name]['simd_probs']
    global_sites = multi_contig_results[r_name]['naive_long_haps'][0]
    
    super_blocks_L3 = hierarchical_assembly.run_hierarchical_step(
        super_blocks_L2,
        global_probs,
        global_sites,
        batch_size=10,
        use_hmm_linking=True,
        recomb_rate=5e-8,
        beam_width=200,
        max_founders=12,
        cc_scale=0.2,
        num_processes=16,
        n_generations=3,
        verbose=False
    )
    
    multi_contig_results[r_name]['super_blocks_L3'] = super_blocks_L3
    
    haps_per_block = [len(b.haplotypes) for b in super_blocks_L3]
    print(f"\n    Output: {len(super_blocks_L3)} L3 super-blocks")
    print(f"    Sites per block: {[len(b.positions) for b in super_blocks_L3]}")
    print(f"    Haps per super-block: {haps_per_block}")

print(f"\nHierarchical Assembly (Level 3) complete in {time.time()-start_time:.1f}s")

#%%
# ==========================================================================
# Validate Level 3 Super Blocks
# ==========================================================================

print("="*60)
print("Validating Level 3 Super Blocks against Ground Truth")
print("="*60)

super_blocks_L3 = multi_contig_results['chr1']['super_blocks_L3']

total_discovered = 0
total_good = 0
total_chimeras = 0
blocks_with_all_founders = 0

chimera_details = []

for block_idx, block in enumerate(super_blocks_L3):
    positions = block.positions
    
    true_at_block = []
    for f_idx in range(len(orig_haps_concrete)):
        founder_vals = np.array([orig_haps_concrete[f_idx][orig_site_to_idx[pos]] 
                                  for pos in positions])
        true_at_block.append(founder_vals)
    
    founders_found = 0
    for f_idx, tf in enumerate(true_at_block):
        best_error = 100
        for h_idx, hap in block.haplotypes.items():
            if hap.ndim > 1:
                hap = np.argmax(hap, axis=1)
            error = np.mean(hap != tf) * 100
            if error < best_error:
                best_error = error
        if best_error < 2.0:
            founders_found += 1
    
    for h_idx, hap in block.haplotypes.items():
        if hap.ndim > 1:
            hap = np.argmax(hap, axis=1)
        errors = [np.mean(hap != tf) * 100 for tf in true_at_block]
        best_f = np.argmin(errors)
        best_error = errors[best_f]
        total_discovered += 1
        if best_error < 2.0:
            total_good += 1
        else:
            total_chimeras += 1
            chimera_details.append({
                'block': block_idx,
                'hap': h_idx,
                'best_f': best_f,
                'error': best_error,
                'n_sites': len(positions)
            })
    
    if founders_found == 6:
        blocks_with_all_founders += 1
    
    print(f"Block {block_idx}: {len(positions)} sites, {len(block.haplotypes)} haps, {founders_found}/6 founders")

print(f"\nResults:")
print(f"  L3 super-blocks: {len(super_blocks_L3)}")
print(f"  Blocks with ALL founders: {blocks_with_all_founders} / {len(super_blocks_L3)}")
print(f"  Total haplotypes: {total_discovered}")
print(f"  Good haplotypes (<2% error): {total_good}")
print(f"  Chimeras (>2% error): {total_chimeras}")

if chimera_details:
    print(f"\nChimera details:")
    for c in sorted(chimera_details, key=lambda x: x['error'], reverse=True):
        print(f"  Block {c['block']}, H{c['hap']}: F{c['best_f']} @ {c['error']:.2f}%")
        
#%% 
# ==========================================================================
# Level 4 Hierarchical Assembly
# ==========================================================================
print("="*60)
print("Level 4 Hierarchical Assembly")
print("="*60)

start_time = time.time()

for r_name in region_keys:
    print(f"\n  Processing {r_name}...")
    
    super_blocks_L3 = multi_contig_results[r_name]['super_blocks_L3']
    
    if len(super_blocks_L3) < 2:
        print("    Only 1 L3 block — no L4 needed.")
        multi_contig_results[r_name]['super_blocks_L4'] = super_blocks_L3
        continue
    
    print(f"    Input: {len(super_blocks_L3)} L3 super-blocks")
    print(f"    Sites per L3 block: {[len(b.positions) for b in super_blocks_L3]}")
    
    global_probs = multi_contig_results[r_name]['simd_probs']
    global_sites = multi_contig_results[r_name]['naive_long_haps'][0]
    
    super_blocks_L4 = hierarchical_assembly.run_hierarchical_step(
        super_blocks_L3,
        global_probs,
        global_sites,
        batch_size=10,
        use_hmm_linking=True,
        recomb_rate=5e-8,
        beam_width=200,
        max_founders=12,
        cc_scale=0.2,
        num_processes=16,
        n_generations=3,
        verbose=False
    )
    
    multi_contig_results[r_name]['super_blocks_L4'] = super_blocks_L4
    
    haps_per_block = [len(b.haplotypes) for b in super_blocks_L4]
    print(f"\n    Output: {len(super_blocks_L4)} L4 super-blocks")
    print(f"    Sites per block: {[len(b.positions) for b in super_blocks_L4]}")
    print(f"    Haps per super-block: {haps_per_block}")

print(f"\nHierarchical Assembly (Level 4) complete in {time.time()-start_time:.1f}s")

#%%
# ==========================================================================
# Validate Level 4 Super Blocks (FINAL)
# ==========================================================================

print("="*60)
print("Validating Level 4 Super Blocks against Ground Truth")
print("="*60)

# Use L4 if available, else L3
if 'super_blocks_L4' in multi_contig_results['chr1']:
    final_blocks = multi_contig_results['chr1']['super_blocks_L4']
    level_name = "L4"
else:
    final_blocks = multi_contig_results['chr1']['super_blocks_L3']
    level_name = "L3 (final)"

orig_sites, orig_haps = multi_contig_results['chr1']['naive_long_haps']
orig_haps_concrete = simulate_sequences.concretify_haps(orig_haps)
orig_site_to_idx = {s: idx for idx, s in enumerate(orig_sites)}

total_discovered = 0
total_good = 0
total_chimeras = 0
blocks_with_all_founders = 0

chimera_details = []

for block_idx, block in enumerate(final_blocks):
    positions = block.positions
    
    true_at_block = []
    for f_idx in range(len(orig_haps_concrete)):
        founder_vals = np.array([orig_haps_concrete[f_idx][orig_site_to_idx[pos]] 
                                  for pos in positions])
        true_at_block.append(founder_vals)
    
    founders_found = 0
    for f_idx, tf in enumerate(true_at_block):
        best_error = 100
        for h_idx, hap in block.haplotypes.items():
            if hap.ndim > 1:
                hap = np.argmax(hap, axis=1)
            error = np.mean(hap != tf) * 100
            if error < best_error:
                best_error = error
        if best_error < 2.0:
            founders_found += 1
    
    for h_idx, hap in block.haplotypes.items():
        if hap.ndim > 1:
            hap = np.argmax(hap, axis=1)
        errors = [np.mean(hap != tf) * 100 for tf in true_at_block]
        best_f = np.argmin(errors)
        best_error = errors[best_f]
        total_discovered += 1
        if best_error < 2.0:
            total_good += 1
        else:
            total_chimeras += 1
            chimera_details.append({
                'block': block_idx,
                'hap': h_idx,
                'best_f': best_f,
                'error': best_error,
                'n_sites': len(positions)
            })
    
    if founders_found == 6:
        blocks_with_all_founders += 1
    
    print(f"Block {block_idx}: {len(positions)} sites, {len(block.haplotypes)} haps, {founders_found}/6 founders")

print(f"\nFinal Results ({level_name}):")
print(f"  Super-blocks: {len(final_blocks)}")
print(f"  Blocks with ALL founders: {blocks_with_all_founders} / {len(final_blocks)}")
print(f"  Total haplotypes: {total_discovered}")
print(f"  Good haplotypes (<2% error): {total_good}")
print(f"  Chimeras (>2% error): {total_chimeras}")

if chimera_details:
    print(f"\nChimera details:")
    for c in sorted(chimera_details, key=lambda x: x['error'], reverse=True):
        print(f"  Block {c['block']}, H{c['hap']}: F{c['best_f']} @ {c['error']:.2f}%")#%%
#%%




#%%
# =============================================================================
# TOLERANCE PAINTING
# =============================================================================

print("\n" + "="*60)
print("RUNNING: Tolerance Painting")
print("="*60)

for r_name in region_keys:
    print(f"\n[Tolerance Painting] Processing Region: {r_name}")

    # 1. Retrieve Data
    gt_block_result = multi_contig_results[r_name]['control_founder_block']
    global_probs = multi_contig_results[r_name]['simd_probs']
    sites, _ = multi_contig_results[r_name]['naive_long_haps']

    # 2. Run Tolerance Painting
    tol_painting_result = paint_samples.paint_samples_tolerance(
        gt_block_result,
        global_probs,
        sites,
        recomb_rate=5e-8,
        switch_penalty=10.0,
        absolute_margin=5.0,
        batch_size=10
    )

    multi_contig_results[r_name]['tolerance_result'] = tol_painting_result

    # 3. Visualization A: Detailed Uncertainty View
    print(f"  Generating detailed tolerance plots for first 3 samples...")
    for i in range(3):
        if i < len(sample_names):
            detail_filename = os.path.join(output_dir, f"{r_name}_tolerance_detail_{sample_names[i]}.png")
            paint_samples.plot_viable_paintings(
                tol_painting_result,
                sample_idx=i,
                output_file=detail_filename
            )

    # 4. Visualization B: Population Consensus
    print(f"  Generating Population Consensus Plot...")

    dense_haps, dense_pos = paint_samples.founder_block_to_dense(gt_block_result)
    founder_data = (dense_haps, dense_pos)

    consensus_samples = []
    for i in range(len(sample_names)):
        best_rep = tol_painting_result[i].get_best_representative_path(founder_data=founder_data)
        consensus_samples.append(best_rep)

    consensus_block = paint_samples.BlockPainting(
        (int(sites[0]), int(sites[-1])), 
        consensus_samples
    )

    cons_filename = os.path.join(output_dir, f"{r_name}_tolerance_consensus_population.png")
    
    paint_samples.plot_population_painting(
        consensus_block,
        output_file=cons_filename,
        title=f"Tolerance Consensus (Uncertainty Masked) - {r_name}",
        sample_names=sample_names,
        figsize_width=20,
        row_height_per_sample=0.25
    )

print("\nTolerance Painting complete.")

#%%
# =============================================================================
# MULTI-CONTIG PEDIGREE INFERENCE
# =============================================================================
print("\n" + "="*60)
print("RUNNING: Multi-Contig Pedigree Inference")
print("="*60)

# 1. Gather Data from all regions
contig_inputs = []
for r_name in region_keys:
    if 'tolerance_result' in multi_contig_results[r_name]:
        entry = {
            'tolerance_painting': multi_contig_results[r_name]['tolerance_result'],
            'founder_block': multi_contig_results[r_name]['control_founder_block']
        }
        contig_inputs.append(entry)
    else:
        print(f"Warning: Tolerance painting missing for {r_name}")

# 2. Run Inference (16-State HMM with tolerance-aware scoring)
pedigree_result = pedigree_inference.infer_pedigree_multi_contig_tolerance(
    contig_inputs, 
    sample_ids=sample_names,
    top_k=20
)

# 3. Apply Auto-Cutoff (Crucial for F1 identification)
pedigree_result.perform_automatic_cutoff()

# 4. Save & Visualize
pedigree_df = pedigree_result.relationships
output_csv = os.path.join(output_dir, "pedigree_inference.csv")
pedigree_df.to_csv(output_csv, index=False)
print(f"Pedigree saved to: {output_csv}")

output_tree = os.path.join(output_dir, "pedigree_tree.png")
pedigree_inference.draw_pedigree_tree(pedigree_df, output_file=output_tree)

# 5. Validate against Truth (if available)
if 'truth_pedigree' in dir():
    print("\n--- Pedigree Validation ---")
    validation_df = pd.merge(
        truth_pedigree[['Sample', 'Generation', 'Parent1', 'Parent2']],
        pedigree_df[['Sample', 'Generation', 'Parent1', 'Parent2']],
        on='Sample',
        suffixes=('_True', '_Inf')
    )

    def check_parent_match(row):
        true_p = {row['Parent1_True'], row['Parent2_True']}
        true_p = {x for x in true_p if pd.notna(x)}
        inf_p = {row['Parent1_Inf'], row['Parent2_Inf']}
        inf_p = {x for x in inf_p if pd.notna(x)}
        
        # F1 check (Truth has Founders, Inf has None)
        if any("Founder" in str(x) for x in true_p):
            return len(inf_p) == 0
        return true_p == inf_p

    validation_df['Gen_Match'] = validation_df['Generation_True'] == validation_df['Generation_Inf']
    validation_df['Parents_Match'] = validation_df.apply(check_parent_match, axis=1)

    gen_acc = validation_df['Gen_Match'].mean() * 100
    descendant_mask = validation_df['Generation_True'].isin(['F2', 'F3'])
    parent_acc = validation_df[descendant_mask]['Parents_Match'].mean() * 100

    print(f"Generation Accuracy: {gen_acc:.2f}%")
    print(f"Parentage Accuracy (F2+F3): {parent_acc:.2f}%")

#%%
# =============================================================================
# PHASE CORRECTION
# =============================================================================
print("\n" + "="*60)
print("RUNNING: Phase Correction (Parent + Children)")
print("="*60)

# Copy founder_block reference so phase correction can find it
for r_name in multi_contig_results:
    if 'control_founder_block' in multi_contig_results[r_name]:
        multi_contig_results[r_name]['founder_block'] = multi_contig_results[r_name]['control_founder_block']

start = time.time()
# Run phase correction on all contigs
multi_contig_results = phase_correction.correct_phase_all_contigs(
    multi_contig_results,
    pedigree_df,
    sample_names,
    num_rounds=3,
    verbose=True
)
print(f"Phase correction time: {time.time()-start:.1f}s")

# =============================================================================
# GREEDY PHASE REFINEMENT POST-PROCESSING
# =============================================================================
print("\n" + "="*60)
print("RUNNING: Greedy Phase Refinement (HOM→HET boundary flips)")
print("="*60)

start_refine = time.time()
multi_contig_results = phase_correction.post_process_phase_greedy_all_contigs(
    multi_contig_results,
    pedigree_df,
    sample_names,
    snps_per_bin=100,
    recomb_rate=5e-8,
    mismatch_cost=4.6,
    verbose=True
)
print(f"Greedy refinement time: {time.time()-start_refine:.1f}s")

#%%
# =============================================================================
# VALIDATE PHASE CORRECTION AGAINST GROUND TRUTH (ALLELE-LEVEL)
# =============================================================================
print("\n" + "="*60)
print("VALIDATING: Phase Correction vs Ground Truth (Allele-Level)")
print("="*60)

from concurrent.futures import ThreadPoolExecutor

def extract_founder_ids_at_positions(painting, positions):
    """
    Extract founder IDs using binary search on chunk boundaries.
    """
    n_pos = len(positions)
    hap1_ids = np.full(n_pos, -1, dtype=np.int32)
    hap2_ids = np.full(n_pos, -1, dtype=np.int32)
    
    # Get chunks
    if hasattr(painting, 'chunks'):
        chunks = painting.chunks
    elif hasattr(painting, 'paths') and painting.paths:
        chunks = painting.paths[0].chunks
    else:
        return hap1_ids, hap2_ids
    
    if not chunks:
        return hap1_ids, hap2_ids
    
    # Build chunk arrays for vectorized lookup
    n_chunks = len(chunks)
    chunk_starts = np.array([c.start for c in chunks], dtype=np.int64)
    chunk_ends = np.array([c.end for c in chunks], dtype=np.int64)
    chunk_hap1 = np.array([c.hap1 for c in chunks], dtype=np.int32)
    chunk_hap2 = np.array([c.hap2 for c in chunks], dtype=np.int32)
    
    # Use searchsorted on ENDS to find potential chunk
    chunk_indices = np.searchsorted(chunk_ends, positions, side='right')
    chunk_indices = np.clip(chunk_indices, 0, n_chunks - 1)
    
    # Check which positions are actually within their assigned chunk
    valid_mask = (positions >= chunk_starts[chunk_indices]) & (positions < chunk_ends[chunk_indices])
    
    # Assign founder IDs where valid
    hap1_ids[valid_mask] = chunk_hap1[chunk_indices[valid_mask]]
    hap2_ids[valid_mask] = chunk_hap2[chunk_indices[valid_mask]]
    
    return hap1_ids, hap2_ids


def evaluate_single_sample(args):
    """Worker function to evaluate a single sample at ALLELE level."""
    i, name, corrected_sample, truth_sample, positions, dense_haps = args
    
    # Extract founder IDs at each position
    corr_hap1, corr_hap2 = extract_founder_ids_at_positions(corrected_sample, positions)
    true_hap1, true_hap2 = extract_founder_ids_at_positions(truth_sample, positions)
    
    n_pos = len(positions)
    max_founder_id = dense_haps.shape[0]
    
    # Convert founder IDs to alleles
    corr_allele1 = np.full(n_pos, -1, dtype=np.int8)
    corr_allele2 = np.full(n_pos, -1, dtype=np.int8)
    true_allele1 = np.full(n_pos, -1, dtype=np.int8)
    true_allele2 = np.full(n_pos, -1, dtype=np.int8)
    
    # Vectorized allele extraction
    valid_corr1 = (corr_hap1 >= 0) & (corr_hap1 < max_founder_id)
    valid_corr2 = (corr_hap2 >= 0) & (corr_hap2 < max_founder_id)
    valid_true1 = (true_hap1 >= 0) & (true_hap1 < max_founder_id)
    valid_true2 = (true_hap2 >= 0) & (true_hap2 < max_founder_id)
    
    pos_indices = np.arange(n_pos)
    corr_allele1[valid_corr1] = dense_haps[corr_hap1[valid_corr1], pos_indices[valid_corr1]]
    corr_allele2[valid_corr2] = dense_haps[corr_hap2[valid_corr2], pos_indices[valid_corr2]]
    true_allele1[valid_true1] = dense_haps[true_hap1[valid_true1], pos_indices[valid_true1]]
    true_allele2[valid_true2] = dense_haps[true_hap2[valid_true2], pos_indices[valid_true2]]
    
    # Compare ALLELES
    direct_match = (corr_allele1 == true_allele1) & (corr_allele2 == true_allele2)
    flipped_match = (corr_allele1 == true_allele2) & (corr_allele2 == true_allele1)
    correct_either = direct_match | flipped_match
    
    n_direct = np.sum(direct_match)
    n_flipped = np.sum(flipped_match)
    
    if n_direct >= n_flipped:
        track1_correct = (corr_allele1 == true_allele1)
        track2_correct = (corr_allele2 == true_allele2)
        dominant_phase = "Direct"
    else:
        track1_correct = (corr_allele1 == true_allele2)
        track2_correct = (corr_allele2 == true_allele1)
        dominant_phase = "Flipped"
    
    # Only count sites where we have valid alleles in BOTH
    valid_mask = (corr_allele1 != -1) & (corr_allele2 != -1) & (true_allele1 != -1) & (true_allele2 != -1)
    n_valid = np.sum(valid_mask)
    
    if n_valid > 0:
        accuracy = np.sum(correct_either & valid_mask) / n_valid
        track1_acc = np.sum(track1_correct & valid_mask) / n_valid
        track2_acc = np.sum(track2_correct & valid_mask) / n_valid
    else:
        accuracy = 0.0
        track1_acc = 0.0
        track2_acc = 0.0
    
    return {
        'Sample': name,
        'Total_sites': len(positions),
        'Valid_sites': int(n_valid),
        'Correct_sites': int(np.sum(correct_either & valid_mask)),
        'Accuracy': accuracy,
        'Track1_accuracy': track1_acc,
        'Track2_accuracy': track2_acc,
        'Direct_matches': int(n_direct),
        'Flipped_matches': int(n_flipped),
        'Dominant_phase': dominant_phase
    }


def evaluate_painting_accuracy(corrected_painting, truth_painting, sample_names, positions, dense_haps, n_workers=8):
    """Compare corrected paintings against ground truth painting at ALLELE level."""
    args_list = []
    for i, name in enumerate(sample_names):
        corrected_sample = corrected_painting[i]
        truth_sample = truth_painting[i]
        args_list.append((i, name, corrected_sample, truth_sample, positions, dense_haps))
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(evaluate_single_sample, args_list))
    
    return pd.DataFrame(results)


def evaluate_contig(args):
    """Worker function to evaluate a single contig."""
    r_name, painting, truth, positions, dense_haps, sample_names = args
    
    contig_eval = evaluate_painting_accuracy(
        painting, truth, sample_names, positions, dense_haps, n_workers=4
    )
    contig_eval['Contig'] = r_name
    
    return r_name, contig_eval


# Evaluate refined paintings
print("Evaluating phase correction accuracy (allele-level)...")

eval_args = []
for r_name in region_keys:
    if 'truth_painting' not in multi_contig_results[r_name]:
        continue
    
    truth = multi_contig_results[r_name]['truth_painting']
    founder_block = multi_contig_results[r_name]['control_founder_block']
    positions = founder_block.positions
    dense_haps, _ = phase_correction.founder_block_to_dense(founder_block)
    
    # Use refined painting (after greedy refinement)
    if 'refined_painting' in multi_contig_results[r_name]:
        painting = multi_contig_results[r_name]['refined_painting']
    elif 'corrected_painting' in multi_contig_results[r_name]:
        painting = multi_contig_results[r_name]['corrected_painting']
    else:
        continue
    
    eval_args.append((r_name, painting, truth, positions, dense_haps, sample_names))

# Run contig evaluations in parallel
all_contig_results = []
with ThreadPoolExecutor(max_workers=len(region_keys)) as executor:
    for r_name, contig_eval in executor.map(evaluate_contig, eval_args):
        mean_acc = contig_eval['Accuracy'].mean()*100
        mean_t1 = contig_eval['Track1_accuracy'].mean()*100
        mean_t2 = contig_eval['Track2_accuracy'].mean()*100
        print(f"  {r_name}: Allele={mean_acc:.2f}%, Track1={mean_t1:.2f}%, Track2={mean_t2:.2f}%")
        all_contig_results.append(contig_eval)

# Aggregate results
if all_contig_results:
    full_eval_df = pd.concat(all_contig_results, ignore_index=True)
    
    # Save detailed results
    eval_output = os.path.join(output_dir, "phase_correction_evaluation.csv")
    full_eval_df.to_csv(eval_output, index=False)
    print(f"\nDetailed evaluation saved to: {eval_output}")
    
    # Group by generation
    full_eval_df['Generation'] = full_eval_df['Sample'].apply(
        lambda x: 'F1' if x.startswith('F1') else ('F2' if x.startswith('F2') else 'F3')
    )
    
    print("\n" + "="*60)
    print("PHASE CORRECTION RESULTS")
    print("="*60)
    
    print("\nAccuracy by Generation:")
    for gen in ['F1', 'F2', 'F3']:
        gen_df = full_eval_df[full_eval_df['Generation'] == gen]
        if len(gen_df) > 0:
            print(f"  {gen}: Accuracy={gen_df['Accuracy'].mean()*100:.2f}%, "
                  f"Track1={gen_df['Track1_accuracy'].mean()*100:.2f}%, "
                  f"Track2={gen_df['Track2_accuracy'].mean()*100:.2f}%, "
                  f"N={len(gen_df)}")
    
    print(f"\nOverall Accuracy:  {full_eval_df['Accuracy'].mean()*100:.2f}%")
    print(f"Overall Track1:    {full_eval_df['Track1_accuracy'].mean()*100:.2f}%")
    print(f"Overall Track2:    {full_eval_df['Track2_accuracy'].mean()*100:.2f}%")
    
    # Phase consistency
    n_direct = (full_eval_df['Dominant_phase'] == 'Direct').sum()
    n_flipped = (full_eval_df['Dominant_phase'] == 'Flipped').sum()
    print(f"\nPhase assignment: {n_direct} samples Direct, {n_flipped} samples Flipped")
    
    # Worst samples
    print("\nWorst 10 samples by accuracy:")
    worst = full_eval_df.nsmallest(10, 'Accuracy')[['Sample', 'Contig', 'Accuracy', 'Track1_accuracy', 'Track2_accuracy', 'Dominant_phase']]
    worst_display = worst.copy()
    worst_display['Accuracy'] = worst_display['Accuracy'] * 100
    worst_display['Track1_accuracy'] = worst_display['Track1_accuracy'] * 100
    worst_display['Track2_accuracy'] = worst_display['Track2_accuracy'] * 100
    print(worst_display.to_string(index=False, float_format='%.2f'))
    
    # Perfect phasing summary
    print("\n" + "="*60)
    print("PERFECT PHASING SUMMARY")
    print("="*60)
    
    perfect_threshold = 0.999
    perfect_samples = full_eval_df[full_eval_df['Track1_accuracy'] >= perfect_threshold]
    n_perfect = len(perfect_samples)
    n_total = len(full_eval_df)
    
    print(f"\nSamples with >=99.9% Track1 accuracy: {n_perfect}/{n_total} ({100*n_perfect/n_total:.1f}%)")
    
    for gen in ['F1', 'F2', 'F3']:
        gen_df = full_eval_df[full_eval_df['Generation'] == gen]
        gen_perfect = gen_df[gen_df['Track1_accuracy'] >= perfect_threshold]
        if len(gen_df) > 0:
            print(f"  {gen}: {len(gen_perfect)}/{len(gen_df)} ({100*len(gen_perfect)/len(gen_df):.1f}%)")
    
    # Samples with internal phase switches
    internal_switch = full_eval_df[
        (full_eval_df['Track1_accuracy'] < perfect_threshold) & 
        (full_eval_df['Track1_accuracy'] > 0.5)
    ]
    print(f"\nSamples with internal phase switches: {len(internal_switch)}")
    if len(internal_switch) > 0:
        print(internal_switch[['Sample', 'Contig', 'Track1_accuracy', 'Track2_accuracy']].head(20).to_string(index=False))

print(f"\nPhase correction validation complete.")
print(f"Total time: {time.time()-start:.1f}s")
