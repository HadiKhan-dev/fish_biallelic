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
#import paint_samples_testing 
#import pedigree_inference_testing
#import phase_correction_testing


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
    {"contig": "chr1", "start": 0, "end": 600},
    {"contig": "chr2", "start": 0, "end": 600},
    {"contig": "chr3", "start": 0, "end": 600},
    {"contig": "chr4", "start": 0, "end": 600},
    {"contig": "chr5", "start": 0, "end": 600},
    {"contig": "chr6", "start": 0, "end": 600}]

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

@dataclass
class FounderBlock:
    """Simple container for founder haplotype data."""
    positions: np.ndarray
    haplotypes: Dict[int, np.ndarray]

for r_name in region_keys:
    print(f"  Creating founder block for {r_name}...")
    
    # Get the naive long haplotypes (sites, haps_list)
    sites, haps_list = multi_contig_results[r_name]['naive_long_haps']
    
    # Convert to the format expected by paint_samples
    # haps_list is a list of probabilistic haplotypes, one per founder
    haplotypes_dict = {}
    for fid, hap_arr in enumerate(haps_list):
        # hap_arr should be (n_sites, 2) with probabilities for allele 0 and 1
        haplotypes_dict[fid] = np.array(hap_arr)
    
    # Create the founder block
    founder_block = FounderBlock(
        positions=np.array(sites, dtype=np.int64),
        haplotypes=haplotypes_dict
    )
    
    # Store it
    multi_contig_results[r_name]['control_founder_block'] = founder_block
    
    print(f"    {r_name}: {len(sites)} sites, {len(haplotypes_dict)} founders")

print("\nFounder blocks created for all regions.")
print(f"Total time: {time.time()-start:.1f}s")
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
    snps_per_bin=150,
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






#%%
vcf_path = "./fish_vcf/AsAc.AulStuGenome.biallelic.bcf.gz"
contig_name = "chr1"

block_size = 100000
shift_size = 50000
starting = 0
ending = 1000

start = time.time()


mgenomic_data = vcf_data_loader.cleanup_block_reads_list(
    vcf_path, 
    contig_name,
    start_block_idx=starting,
    end_block_idx=ending,
    block_size=block_size,
    shift_size=shift_size,
    num_processes=16
)
print("Time taken:", time.time() - start)
#%%
start = time.time()

# 1. Run Haplotype Discovery on all blocks
# This uses the GenomicData object directly and returns a list of BlockResult objects
block_results = block_haplotypes.generate_all_block_haplotypes(
    genomic_data)


print("Time taken:", time.time() - start)
#%%
# 2. Run the Naive Linker
# Now we pass the BlockResults object instead of raw arrays
start = time.time()

(naive_blocks, naive_long_haps) = block_linking_naive.generate_long_haplotypes_naive(
    block_results, 
    num_long_haps=6
)

print("Time taken:", time.time() - start)
#%%
haplotype_sites = naive_long_haps[0]
haplotype_data = naive_long_haps[1]
    
# 1. Prepare Founders (F0)
concrete_haps = simulate_sequences.concretify_haps(haplotype_data)
parents = simulate_sequences.pairup_haps(concrete_haps)

generation_sizes = [10, 100, 200]

# UPDATE: Unpack 3 values now (Offspring, Pedigree DataFrame, Ancestry Data)
all_offspring, truth_pedigree, truth_paintings_raw = simulate_sequences.simulate_pedigree(
    parents, 
    haplotype_sites, 
    generation_sizes, 
    recomb_rate=5*10**-8, 
    mutate_rate=10**-10,
    output_plot="ground_truth_pedigree.png"
)

# 2. Convert Raw Truth to BlockPainting object
# This allows us to visualize exactly what the simulation did before sequencing errors
true_biological_painting = simulate_sequences.convert_truth_to_painting_objects(truth_paintings_raw)

# 3. Save Truth for later validation
truth_pedigree.to_csv("ground_truth_pedigree.csv", index=False)
print("Ground Truth Pedigree data saved to 'ground_truth_pedigree.csv'")

sample_names = truth_pedigree['Sample'].tolist()

# 4. Visualize the Biological Truth
print("Generating Biological Truth Painting Plot...")
paint_samples.plot_population_painting(
    true_biological_painting,
    output_file="biological_truth_painting.png",
    title="Biological Truth (Actual Recombinations during Simulation)",
    sample_names=sample_names,
    figsize_width=20,
    row_height_per_sample=0.25
)

all_sites = haplotype_sites
all_likelihoods = simulate_sequences.combine_into_genotype(all_offspring)

read_depth = 2000
new_reads_array = simulate_sequences.read_sample_all_individuals(all_offspring, read_depth, error_rate=0.02)

simd_genomic_data = simulate_sequences.chunk_up_data(
    haplotype_sites, 
    new_reads_array,
    0, 50000000, 
    0, 0, # Block Size / Shift ignored
    use_snp_count=True,
    snps_per_block=200,
    snp_shift=200) # Disjoint blocks (Shift = Size)

(simd_site_priors, simd_probabalistic_genotypes) = analysis_utils.reads_to_probabilities(new_reads_array)

#%%
start = time.time()

# Generate Haplotypes on the 200-SNP blocks
test_haps = block_haplotypes.generate_all_block_haplotypes(
    simd_genomic_data,               # The data chunked by SNP count
    uniqueness_threshold_percent=1.0, # 1.0% difference required to be distinct (2 SNPs)
    diff_threshold_percent=0.5,       # 0.5% difference threshold for merging (1 SNP)
    wrongness_threshold=1.0,
    num_processes=16
)

print(f"Haplotype Discovery Time: {time.time()-start:.2f}s")

#%%
min_pos = simd_genomic_data.positions[0][0]
max_pos = simd_genomic_data.positions[-1][-1]

# Find indices in the global simulated arrays (haplotype_sites/new_reads_array)
# that correspond to the range covered by the chunked data.
start_idx = np.searchsorted(haplotype_sites, min_pos)
end_idx = np.searchsorted(haplotype_sites, max_pos, side='right')

global_sites = haplotype_sites[start_idx:end_idx]
global_reads = new_reads_array[:, start_idx:end_idx, :]

(_, global_probs) = analysis_utils.reads_to_probabilities(global_reads)#%%
start_time = time.time()
    
initial_block_results = block_haplotypes.generate_all_block_haplotypes(
    simd_genomic_data,               # The data chunked by SNP count
    uniqueness_threshold_percent=1.0, # 1.0% difference required to be distinct (2 SNPs)
    diff_threshold_percent=0.5,       # 0.5% difference threshold for merging (1 SNP)
    wrongness_threshold=1.0,
    num_processes=16)

#%%

# =============================================================================
# HIERARCHICAL ASSEMBLY (WITH DUAL SCALING)
# =============================================================================

start = time.time()
super_blocks_level_1 = hierarchical_assembly.run_hierarchical_step(
    initial_block_results,
    global_probs,
    global_sites,
    batch_size=10,
    use_hmm_linking=False, # Use standard linking for first step (faster/cleaner on small blocks)
    recomb_rate=5e-8,
    beam_width=200,
    max_founders=12,
    switch_cost_scale=0.1,          
    pruning_switch_cost_scale=0.1,
    recomb_penalty=None, 
    pruning_recomb_penalty=None,
    complexity_penalty_scale=0.2      
)
print(f"Time for Level 1: {time.time()-start}")
#%%
start = time.time()
super_blocks_level_2 = hierarchical_assembly.run_hierarchical_step(
    super_blocks_level_1,
    global_probs,
    global_sites,
    batch_size=10,
    use_hmm_linking=True,
    recomb_rate=5e-8,
    beam_width=200,
    max_founders=12,
    switch_cost_scale=0.075,
    pruning_switch_cost_scale=0.005,
    recomb_penalty=None,
    pruning_recomb_penalty=None,
    complexity_penalty_scale=0.2      
)
print(f"Time for Level 2: {time.time()-start}")

#%%
start = time.time()
super_blocks_level_3 = hierarchical_assembly.run_hierarchical_step(
    super_blocks_level_2,
    global_probs,
    global_sites,
    batch_size=10,
    use_hmm_linking=True,
    recomb_rate=5e-8,
    beam_width=200,
    max_founders=12,    
    switch_cost_scale=0.075,
    pruning_switch_cost_scale=0.005,
    recomb_penalty=None,
    pruning_recomb_penalty=None,
    complexity_penalty_scale=0.2      
)
print(f"Time for Level 3: {time.time()-start}")

#%%
start = time.time()
super_blocks_level_4 = hierarchical_assembly.run_hierarchical_step(
    super_blocks_level_3,
    global_probs,
    global_sites,
    batch_size=10,
    use_hmm_linking=True,
    recomb_rate=5e-8,
    beam_width=200,
    max_founders=12,
    switch_cost_scale=0.075,
    pruning_switch_cost_scale=0.005,
    recomb_penalty=None,
    pruning_recomb_penalty=None,
    complexity_penalty_scale=0.2      
)
print(f"Time for Level 4: {time.time()-start}")
#%%
# =============================================================================
# PAINTING & VISUALIZATION
# =============================================================================

print("\n--- Starting Final Sample Painting ---")
start = time.time()

# 1. Select the final Super-Block
# Ideally, Level 4 has collapsed the region into a single contiguous block.
# We take the first one (which covers the start of the region).
if len(super_blocks_level_4) > 0:
    final_consensus_block = super_blocks_level_4[0]
else:
    raise ValueError("Level 4 produced no blocks!")

# 2. Paint the samples
# We use the global probability arrays that cover the simulation range.
# switch_penalty=15.0 is strict to prevent single-SNP noise from creating fake crossovers.
painting_result = paint_samples.paint_samples_in_block(
    final_consensus_block,
    global_probs,
    global_sites,
    recomb_rate=5e-8,
    switch_penalty=15.0  
)

print(f"Painting complete in {time.time() - start:.2f}s")

# 3. Generate Sample Names for the Plot
# Your simulation structure was: F1 (10), F2 (100), F3 (200) -> Total 310
sample_names = []
for i in range(10): sample_names.append(f"F1_{i}")
for i in range(100): sample_names.append(f"F2_{i}")
for i in range(200): sample_names.append(f"F3_{i}")

# Safety check in case you changed simulation numbers
if len(sample_names) != len(painting_result):
    sample_names = None

# 4. Visualize
output_filename = "final_painting.png"
paint_samples.plot_painting(
    painting_result,
    output_file=output_filename,
    title="Reconstructed Haplotype Painting (Simulated F1 -> F3)",
    figsize=(20, 20), # Tall figure to fit 310 samples
    sample_names=sample_names
)

# 5. Quick Stats Check
# Calculate average crossovers per generation to verify biological logic
recomb_counts = [s.num_recombinations for s in painting_result]

# F1s are first 10
f1_avg = np.mean(recomb_counts[:10])
# F2s are next 100
f2_avg = np.mean(recomb_counts[10:110])
# F3s are last 200
f3_avg = np.mean(recomb_counts[110:])

print("\n--- Biological Verification ---")
print(f"Avg Crossovers F1: {f1_avg:.2f} (Expected ~0 relative to parents)")
print(f"Avg Crossovers F2: {f2_avg:.2f} (Expected Low)")
print(f"Avg Crossovers F3: {f3_avg:.2f} (Expected Higher)")
#%%
print("\n--- Starting Pedigree Inference ---")
start = time.time()

# 1. Run the full inference pipeline
# We pass the 'painting_result' from the previous step.
# snps_per_bin=150 corresponds to roughly 10-15kb resolution in your simulated data 
# (depending on SNP density), which is a good balance for detecting chimeras vs real crossovers.
pedigree_data = pedigree_inference.run_pedigree_inference(
    painting_result,
    sample_ids=sample_names,  # Passing the generated F1_x, F2_x, F3_x names
    snps_per_bin=150,
    output_prefix="simulated_pedigree"
)

print(f"Pedigree Inference complete in {time.time() - start:.2f}s")

# 2. Validation / Accuracy Check (Since we know the truth from simulation)
# Let's verify if the inferred generations match the simulation labels.

print("\n--- Generation Inference Accuracy ---")
inferred_gens = pedigree_data.relationships['Generation'].values
true_gens = []

# Reconstruct truth labels based on your simulation loop order:
# 10 F1s, then 100 F2s, then 200 F3s
for i in range(10): true_gens.append("F1")
for i in range(100): true_gens.append("F2")
for i in range(200): true_gens.append("F3")

matches = 0
for i, (inf, true) in enumerate(zip(inferred_gens, true_gens)):
    if inf == true:
        matches += 1
    else:
        # Optional: Print mismatches to debug
        # print(f"Sample {i}: Truth={true}, Inferred={inf}")
        pass

accuracy = (matches / len(true_gens)) * 100
print(f"Generation Label Accuracy: {accuracy:.2f}% ({matches}/{len(true_gens)})")

# 3. Systematic Error Report
print("\n--- Systematic Error Analysis ---")
bad_bins = pedigree_data.systematic_errors
if len(bad_bins) > 0:
    print(f"WARNING: Detected {len(bad_bins)} systematic error zones (potential chimeric founders).")
    print(f"These regions were automatically filtered from the recombination map.")
else:
    print("SUCCESS: No systematic linking errors detected.")

# 4. Recombination Map Stats
map_df = pedigree_data.recombination_map
print(f"\nTotal Detected Crossovers (Filtered): {len(map_df)}")
print(map_df.head())
#%%
# =============================================================================
# VALIDATION: INFERRED VS GROUND TRUTH
# =============================================================================

print("\n--- Final Pedigree Validation ---")

# 1. Merge Dataframes for row-by-row comparison
# We use an inner join on 'Sample'
validation_df = pd.merge(
    truth_pedigree[['Sample', 'Generation', 'Parent1', 'Parent2']],
    pedigree_data.relationships[['Sample', 'Generation', 'Parent1', 'Parent2']],
    on='Sample',
    suffixes=('_True', '_Inf')
)

def check_parent_match(row):
    """
    Compares sets of parents. Order doesn't matter.
    """
    # 1. Get True Parents set (filtering out NaNs)
    true_p = {row['Parent1_True'], row['Parent2_True']}
    true_p = {x for x in true_p if pd.notna(x)}
    
    # 2. Get Inferred Parents set
    inf_p = {row['Parent1_Inf'], row['Parent2_Inf']}
    inf_p = {x for x in inf_p if pd.notna(x)}
    
    # 3. Handling F1s (Founders are not in sample list, so Inf should be empty)
    # If True parents are "Founder_X", we expect Inference to be empty.
    is_founder_child = any("Founder" in str(x) for x in true_p)
    if is_founder_child:
        return len(inf_p) == 0 # Correct if we inferred 'No Parents' for F1s
        
    # 4. Standard Check
    return true_p == inf_p

# Apply validation logic
validation_df['Gen_Match'] = validation_df['Generation_True'] == validation_df['Generation_Inf']
validation_df['Parents_Match'] = validation_df.apply(check_parent_match, axis=1)

# --- STATISTICS ---

# 1. Generation Accuracy
gen_acc = validation_df['Gen_Match'].mean() * 100
print(f"Generation Classification Accuracy: {gen_acc:.2f}%")

# 2. Parentage Accuracy (Broken down by Generation)
print("\nParentage Accuracy by Generation:")
for gen in ['F1', 'F2', 'F3']:
    subset = validation_df[validation_df['Generation_True'] == gen]
    if len(subset) == 0: continue
    
    # Accuracy
    acc = subset['Parents_Match'].mean() * 100
    print(f"  {gen}: {acc:.2f}% ({subset['Parents_Match'].sum()}/{len(subset)})")
    
    # Show Failures (Limit to 5)
    failures = subset[~subset['Parents_Match']]
    if len(failures) > 0:
        print(f"    Sample Failures (First 5):")
        for i, row in failures.head(5).iterrows():
            t_p = {row['Parent1_True'], row['Parent2_True']}
            i_p = {row['Parent1_Inf'], row['Parent2_Inf']}
            print(f"      {row['Sample']}: True={t_p} vs Inf={i_p}")

# 3. Overall Parentage Accuracy (Excluding F1s who have no internal parents)
# We only care if we found parents for F2 and F3
descendant_mask = validation_df['Generation_True'].isin(['F2', 'F3'])
descendant_acc = validation_df[descendant_mask]['Parents_Match'].mean() * 100

print(f"\nOverall Descendant Parentage Accuracy (F2+F3): {descendant_acc:.2f}%")

if descendant_acc > 95.0:
    print("\n[SUCCESS] Pipeline successfully reconstructed the pedigree structure!")
else:
    print("\n[WARNING] Pedigree reconstruction had errors. Check IBD thresholds in pedigree_inference.py.")
