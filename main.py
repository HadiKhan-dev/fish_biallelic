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
import paint_samples
import hierarchical_assembly
import pedigree_inference
import pedigree_inference_testing

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
    {"contig": "chr1", "start": 0, "end": 400},
    {"contig": "chr2", "start": 0, "end": 400},
    {"contig": "chr3", "start": 0, "end": 400},
    {"contig": "chr4", "start": 0, "end": 400},
    {"contig": "chr5", "start": 0, "end": 400},
    {"contig": "chr6", "start": 0, "end": 400},
    {"contig": "chr7", "start": 0, "end": 400},
    {"contig": "chr8", "start": 0, "end": 400},
    {"contig": "chr9", "start": 0, "end": 400},
    {"contig": "chr10", "start": 0, "end": 400}]

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

# 2. Run Multi-Contig Simulation
# This generates a SINGLE pedigree (relationships) applied to ALL contigs independently
generation_sizes = [10, 100, 200]
print(f"Running Multi-Contig Simulation for {len(region_keys)} regions...")

# Returns lists-of-lists (one sub-list per contig)
all_offspring_lists, truth_pedigree, truth_paintings_lists = simulate_sequences.simulate_pedigree(
    founders_list, 
    sites_list, 
    generation_sizes, 
    recomb_rate=5*10**-8, 
    mutate_rate=10**-10,
    output_plot=os.path.join(output_dir, "ground_truth_pedigree.png")
)

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
    
    paint_samples.plot_painting(
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
#%%
print("\n" + "="*60)
print("CONTROL EXPERIMENT: Multi-Contig Painting using GROUND TRUTH Haplotypes")
print("="*60)

for r_name in region_keys:
    print(f"\n[Control Painting] Processing Region: {r_name}")
    
    # 1. Retrieve Data
    # 'naive_long_haps' contains the exact sites and haplotypes used to start the simulation
    sites, haps_list = multi_contig_results[r_name]['naive_long_haps']
    
    # 'simulated_reads' are the raw reads generated from the simulation
    reads_matrix = multi_contig_results[r_name]['simulated_reads']
    
    # 2. Calculate Genotype Probabilities for the FULL contig
    # (We do this here because previous steps might have only calculated probs for chunks)
    # This gives us (Samples, Sites, 3) for the whole chromosome
    _, global_probs = analysis_utils.reads_to_probabilities(reads_matrix)
    
    # 3. Package Ground Truth Haplotypes into a BlockResult
    # The simulation output 'haps_list' is a list of arrays. We need a dict {id: array}
    gt_haps_dict = {i: h_arr for i, h_arr in enumerate(haps_list)}
    
    # Create valid flags (all true)
    gt_flags = np.ones(len(sites), dtype=int)
    
    gt_block_result = block_haplotypes.BlockResult(
        positions=sites,
        haplotypes=gt_haps_dict,
        keep_flags=gt_flags
    )
    
    # 4. Paint samples using these PERFECT haplotypes
    print(f"  Painting {len(sample_names)} samples...")
    gt_painting_result = paint_samples.paint_samples_in_block(
        gt_block_result,
        global_probs,
        sites,
        recomb_rate=5e-8,
        switch_penalty=15.0 
    )
    
    # 5. Visualize Ground Truth Painting
    plot_filename = os.path.join(output_dir, f"{r_name}_control_painting.png")
    print(f"  Generating Plot: {plot_filename}")
    
    paint_samples.plot_painting(
        gt_painting_result,
        output_file=plot_filename,
        title=f"Control Experiment: Ground Truth Painting - {r_name}",
        sample_names=sample_names,
        figsize_width=20,
        row_height_per_sample=0.25
    )
    
    # 6. Store results for the Multi-Contig Pedigree Inference step
    multi_contig_results[r_name]['control_painting'] = gt_painting_result
    multi_contig_results[r_name]['control_founder_block'] = gt_block_result

print("\nControl Painting complete for all regions.")
#%%
# =============================================================================
# MULTI-CONTIG PEDIGREE INFERENCE
# =============================================================================

print("\n" + "="*60)
print("RUNNING: Multi-Contig Pedigree Inference (Control)")
print("="*60)

# 1. Gather Data from all regions
contig_inputs = []

for r_name in region_keys:
    # We need the Painting and the Founder Block for every region
    if 'control_painting' in multi_contig_results[r_name]:
        entry = {
            'painting': multi_contig_results[r_name]['control_painting'],
            'founder_block': multi_contig_results[r_name]['control_founder_block']
        }
        contig_inputs.append(entry)
    else:
        print(f"Warning: Control painting missing for {r_name}")

# 2. Run Inference
# Note: sample_names is defined earlier in your script from the truth csv
pedigree_result = pedigree_inference.infer_pedigree_multi_contig(
    contig_inputs, 
    sample_ids=sample_names,
    top_k=20
)

# 3. Save & Visualize
ped_df = pedigree_result.relationships
output_csv = os.path.join(output_dir, "multi_contig_pedigree_control.csv")
ped_df.to_csv(output_csv, index=False)
print(f"Multi-Contig Pedigree saved to: {output_csv}")

output_tree = os.path.join(output_dir, "multi_contig_tree_control.png")
pedigree_inference.draw_pedigree_tree(ped_df, output_file=output_tree)

# 4. Validate
print("\n--- Multi-Contig Validation ---")
control_validation_df = pd.merge(
    truth_pedigree[['Sample', 'Generation', 'Parent1', 'Parent2']],
    ped_df[['Sample', 'Generation', 'Parent1', 'Parent2']],
    on='Sample',
    suffixes=('_True', '_Inf')
)

def check_parent_match(row):
    true_p = {row['Parent1_True'], row['Parent2_True']}
    true_p = {x for x in true_p if pd.notna(x)}
    inf_p = {row['Parent1_Inf'], row['Parent2_Inf']}
    inf_p = {x for x in inf_p if pd.notna(x)}
    
    # F1 check
    if any("Founder" in str(x) for x in true_p):
        return len(inf_p) == 0
    return true_p == inf_p

control_validation_df['Gen_Match'] = control_validation_df['Generation_True'] == control_validation_df['Generation_Inf']
control_validation_df['Parents_Match'] = control_validation_df.apply(check_parent_match, axis=1)

gen_acc = control_validation_df['Gen_Match'].mean() * 100
descendant_mask = control_validation_df['Generation_True'].isin(['F2', 'F3'])
parent_acc = control_validation_df[descendant_mask]['Parents_Match'].mean() * 100

print(f"Generation Accuracy: {gen_acc:.2f}%")
print(f"Parentage Accuracy (F2+F3): {parent_acc:.2f}%")
#%%
import numpy as np
import pandas as pd
import math
from pedigree_inference import discretize_paintings, convert_id_grid_to_allele_grid, run_inheritance_hmm_dynamic

# =============================================================================
# DEBUG CONFIGURATION
# =============================================================================
target_kid = "F3_22"
true_parents = ["F2_65", "F2_84"]
inferred_parents = ["F1_6", "F2_84"] # The wrong set

print(f"\n=== DEBUGGING TRIO: {target_kid} ===")
print(f"Comparing True: {true_parents}")
print(f"vs Inferred:    {inferred_parents}")

# Check if indices exist
try:
    k_idx = sample_names.index(target_kid)
    tp_indices = [sample_names.index(p) for p in true_parents]
    ip_indices = [sample_names.index(p) for p in inferred_parents]
except ValueError as e:
    print(f"Error: Sample not found in sample_names list. {e}")
    # Stop execution if names don't match
    raise e

# =============================================================================
# METRIC ACCUMULATORS
# =============================================================================
# We need to track the global sums across all contigs
global_scores = {
    "True_Pair": 0.0,
    "Inferred_Pair": 0.0
}

global_switches = {
    target_kid: 0,
    true_parents[0]: 0,
    true_parents[1]: 0,
    inferred_parents[0]: 0, 
    # Note: inferred_parents[1] is usually same as true_parents[1] in your case (F1_4)
}
if inferred_parents[1] not in global_switches:
    global_switches[inferred_parents[1]] = 0

# =============================================================================
# PER-CONTIG ANALYSIS
# =============================================================================

for r_name in region_keys:
    print(f"\n--- Analyzing Region: {r_name} ---")
    
    # 1. Get Data
    painting = multi_contig_results[r_name]['control_painting']
    founder_block = multi_contig_results[r_name]['control_founder_block']
    
    # 2. Re-Discretize
    id_grid, bin_edges, bin_centers = discretize_paintings(painting, snps_per_bin=150)
    allele_grid = convert_id_grid_to_allele_grid(id_grid, bin_centers, founder_block)
    
    # 3. Setup HMM Costs locally
    recomb_rate = 5e-8
    robustness = 1e-2
    error_penalty = -math.log(robustness)
    
    dists = np.zeros(len(bin_centers))
    dists[1:] = np.diff(bin_centers)
    theta = 1.0 - np.exp(-dists * recomb_rate)
    theta = np.clip(theta, 1e-15, 0.5)
    switch_costs = np.log(theta)
    stay_costs = np.log(1.0 - theta)
    
    # 4. Count Switches (Directionality Check)
    def get_switches(idx):
        row = id_grid[idx]
        sw = (row[:-1, :] != row[1:, :]) & (row[:-1, :] != -1) & (row[1:, :] != -1)
        return np.sum(sw)

    # Accumulate Switches
    k_sw = get_switches(k_idx)
    global_switches[target_kid] += k_sw
    
    for p_name in set(true_parents + inferred_parents):
        pidx = sample_names.index(p_name)
        global_switches[p_name] += get_switches(pidx)

    # 5. Score The Pairs (HMM)
    def get_pair_score(p1_idx, p2_idx):
        kid_h0 = allele_grid[k_idx, :, 0]
        kid_h1 = allele_grid[k_idx, :, 1]
        
        par1_dip = allele_grid[p1_idx]
        par2_dip = allele_grid[p2_idx]
        
        # Calculate raw likelihoods for all combinations
        # Score(Kid_H0 | Parent1), Score(Kid_H0 | Parent2), etc.
        s_k0_p1 = run_inheritance_hmm_dynamic(kid_h0, par1_dip, switch_costs, stay_costs, error_penalty)
        s_k1_p1 = run_inheritance_hmm_dynamic(kid_h1, par1_dip, switch_costs, stay_costs, error_penalty)
        
        s_k0_p2 = run_inheritance_hmm_dynamic(kid_h0, par2_dip, switch_costs, stay_costs, error_penalty)
        s_k1_p2 = run_inheritance_hmm_dynamic(kid_h1, par2_dip, switch_costs, stay_costs, error_penalty)
        
        # Option A: P1 -> H0, P2 -> H1
        score_a = s_k0_p1 + s_k1_p2
        
        # Option B: P1 -> H1, P2 -> H0
        score_b = s_k1_p1 + s_k0_p2
        
        # In multi-contig inference, we take the MAX phase configuration per contig
        return max(score_a, score_b)

    score_true = get_pair_score(tp_indices[0], tp_indices[1])
    score_inf = get_pair_score(ip_indices[0], ip_indices[1])
    
    print(f"  True Pair Score:     {score_true:.2f}")
    print(f"  Inferred Pair Score: {score_inf:.2f}")
    
    global_scores["True_Pair"] += score_true
    global_scores["Inferred_Pair"] += score_inf

# =============================================================================
# FINAL REPORT
# =============================================================================

print("\n" + "="*40)
print("FINAL DIAGNOSTIC REPORT")
print("="*40)

print("\n1. GLOBAL HMM SCORES (Higher is Better)")
print(f"  True Pair ({true_parents}):     {global_scores['True_Pair']:.2f}")
print(f"  Inferred Pair ({inferred_parents}): {global_scores['Inferred_Pair']:.2f}")

diff = global_scores['Inferred_Pair'] - global_scores['True_Pair']
if diff > 0:
    print(f"  -> RESULT: Inferred pair fits genetic data BETTER by {diff:.2f} log-units.")
    print("     (The F3 actually looks more like a parent than the F1 on these contigs)")
else:
    print(f"  -> RESULT: True pair fits genetic data BETTER by {abs(diff):.2f} log-units.")
    print("     (Logic error: Why was True Pair rejected?)")

print("\n2. DIRECTIONALITY FILTER (Switch Counts)")
print("  Parents must have FEWER switches than the child.")
print(f"  Target Child ({target_kid}): {global_switches[target_kid]}")

print("\n  Candidate Comparisons:")
for p_name in set(true_parents + inferred_parents):
    sw = global_switches[p_name]
    diff = global_switches[target_kid] - sw
    # In `infer_pedigree_multi_contig`, we use a margin (e.g., 5 * num_contigs)
    # Filter: Parent_Switches <= Child_Switches + Margin
    # So if Parent_Switches is massively higher, it fails.
    
    status = "VALID" if sw <= (global_switches[target_kid] + 25) else "FILTERED" # Assuming margin=25 (5*5)
    print(f"    {p_name}: {sw} switches. (Child - Parent = {diff}). Status: {status}")

# SPECIFIC CHECK FOR THE FAILURE CASE
fp_name = "F3_5"
if fp_name in global_switches:
    kid_sw = global_switches[target_kid]
    fp_sw = global_switches[fp_name]
    
    if fp_sw < kid_sw:
        print(f"\n[DIAGNOSIS] {fp_name} (F3) has FEWER switches ({fp_sw}) than {target_kid} (F2) ({kid_sw}).")
        print("The algorithm correctly interprets this as F3 being 'older/purer' than F2.")
        print("Root Cause: Painting noise or short contigs allowed F3 to look like a founder.")
    else:
        print(f"\n[DIAGNOSIS] {fp_name} has MORE switches. It should have been filtered.")



















#%%
vcf_path = "./fish_vcf/AsAc.AulStuGenome.biallelic.bcf.gz"
contig_name = "chr1"

block_size = 100000
shift_size = 50000
starting = 0
ending = 480

start = time.time()


genomic_data = vcf_data_loader.cleanup_block_reads_list(
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
paint_samples.plot_painting(
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
    num_processes=16
)

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
#%%
# =============================================================================
# CONTROL EXPERIMENT: PEDIGREE INFERENCE ON GROUND TRUTH HAPLOTYPES
# =============================================================================

print("\n" + "="*60)
print("CONTROL EXPERIMENT: Pedigree Inference using GROUND TRUTH Haplotypes")
print("="*60)

# 1. Package Ground Truth Haplotypes
gt_indices = np.searchsorted(haplotype_sites, global_sites)

gt_haps_sliced = {}
for i, hap_arr in enumerate(haplotype_data):
    gt_haps_sliced[i] = hap_arr[gt_indices]

gt_flags = np.ones(len(global_sites), dtype=int)

gt_block_result = block_haplotypes.BlockResult(
    positions=global_sites,
    haplotypes=gt_haps_sliced,
    keep_flags=gt_flags
)

# 2. Paint samples using these PERFECT haplotypes
print("Painting samples using Ground Truth Haplotypes...")
gt_painting_result = paint_samples.paint_samples_in_block(
    gt_block_result,
    global_probs,
    global_sites,
    recomb_rate=5e-8,
    switch_penalty=15.0 
)

# --- VISUALIZE GROUND TRUTH PAINTING ---
print("Generating Ground Truth Painting Plot...")
paint_samples.plot_painting(
    gt_painting_result,
    output_file="control_truth_painting.png",
    title="Control Experiment: Ground Truth Haplotype Painting",
    sample_names=sample_names,
    figsize_width=20,          # Wide figure
    row_height_per_sample=0.25 # 0.25 inches per sample -> tall image
)

#%%
# =============================================================================
# PEDIGREE RECONSTRUCTION v5 (HMM-Likelihood Approach)
# =============================================================================

print("\n" + "="*60)
print("RE-RUNNING Pedigree Inference (HMM-Likelihood Approach)")
print("="*60)

# Run the new HMM-based inference
gt_pedigree_data = pedigree_inference.run_pedigree_inference(
    gt_painting_result,
    sample_ids=sample_names,
    snps_per_bin=150,              # Bin size for discretization
    founder_block=gt_block_result, # <--- CRITICAL: Pass Founder Data for Allele conversion
    output_prefix="control_truth_pedigree_hmm"
)

# Merge & Check
control_validation_df = pd.merge(
    truth_pedigree[['Sample', 'Generation', 'Parent1', 'Parent2']],
    gt_pedigree_data.relationships[['Sample', 'Generation', 'Parent1', 'Parent2']],
    on='Sample',
    suffixes=('_True', '_Control')
)

def check_control_parent_match(row):
    # 1. Get True Parents set (filtering out NaNs)
    true_p = {row['Parent1_True'], row['Parent2_True']}
    true_p = {x for x in true_p if pd.notna(x)}
    
    # 2. Get Inferred Parents set
    inf_p = {row['Parent1_Control'], row['Parent2_Control']}
    inf_p = {x for x in inf_p if pd.notna(x)}
    
    # 3. F1 Handling (Founders are not in sample list)
    is_founder_child = any("Founder" in str(x) for x in true_p)
    if is_founder_child: return len(inf_p) == 0
    
    # 4. Equality Check
    return true_p == inf_p

control_validation_df['Gen_Match'] = control_validation_df['Generation_True'] == control_validation_df['Generation_Control']
control_validation_df['Parents_Match'] = control_validation_df.apply(check_control_parent_match, axis=1)

print("\n--- Validation Results ---")
print(f"Generation Accuracy: {control_validation_df['Gen_Match'].mean()*100:.2f}%")

descendant_mask = control_validation_df['Generation_True'].isin(['F2', 'F3'])
parent_acc = control_validation_df[descendant_mask]['Parents_Match'].mean() * 100
print(f"Parentage Accuracy (F2+F3): {parent_acc:.2f}%")


#%%
# =============================================================================
# DEBUG: COMPARATIVE TRIO ANALYSIS (FIXED)
# =============================================================================

print("\n" + "="*60)
print("DEBUG: Head-to-Head Parent Comparison (Allele Aware)")
print("="*60)

# Define Targets
kid = "F2_0"
true_p1, true_p2 = "F1_6", "F1_7"
false_p1, false_p2 = "F1_7", "F3_187"

if all(x in sample_names for x in [kid, true_p1, true_p2, false_p1, false_p2]):
    k_idx = sample_names.index(kid)
    tp1_idx, tp2_idx = sample_names.index(true_p1), sample_names.index(true_p2)
    fp1_idx, fp2_idx = sample_names.index(false_p1), sample_names.index(false_p2)
    
    # 1. Regenerate Grids
    # ID Grid
    grid_id, bin_edges, bin_centers = pedigree_inference.discretize_paintings(gt_painting_result, snps_per_bin=150)
    
    # Allele Grid (Requires Founder Block)
    grid_allele = pedigree_inference.convert_id_grid_to_allele_grid(grid_id, bin_centers, gt_block_result)

    # --- HELPER FUNCTIONS ---
    def get_ibd0(idx_a, idx_b, use_alleles=True):
        g = grid_allele if use_alleles else grid_id
        A, B = g[idx_a], g[idx_b]
        matches = (A[:,0]==B[:,0]) | (A[:,0]==B[:,1]) | (A[:,1]==B[:,0]) | (A[:,1]==B[:,1])
        valid = (A[:,0]!=-1) & (B[:,0]!=-1)
        return np.sum((~matches) & valid) / np.sum(valid) if np.sum(valid) > 0 else 1.0

    def count_switches(idx):
        g = grid_id[idx] 
        # Count changes along bins (axis 0)
        switches = (g[:-1, :] != g[1:, :]) & (g[:-1, :] != -1) & (g[1:, :] != -1)
        return np.sum(switches)

    def calc_mendel(k_idx, p1_idx, p2_idx, use_alleles=True):
        g = grid_allele if use_alleles else grid_id
        k, pa, pb = g[k_idx], g[p1_idx], g[p2_idx]
        k0_in_pa = (k[:,0] == pa[:,0]) | (k[:,0] == pa[:,1])
        k0_in_pb = (k[:,0] == pb[:,0]) | (k[:,0] == pb[:,1])
        k1_in_pa = (k[:,1] == pa[:,0]) | (k[:,1] == pa[:,1])
        k1_in_pb = (k[:,1] == pb[:,0]) | (k[:,1] == pb[:,1])
        
        valid_inherit = (k0_in_pa & k1_in_pb) | (k0_in_pb & k1_in_pa)
        valid_bins = (k[:,0]!=-1) & (pa[:,0]!=-1) & (pb[:,0]!=-1)
        
        return np.sum(valid_inherit & valid_bins) / np.sum(valid_bins) if np.sum(valid_bins) > 0 else 0.0

    # --- METRICS ---
    print(f"\n[1] Candidate Checks (Allele IBD0):")
    print(f"  True P1 ({true_p1}):  {get_ibd0(k_idx, tp1_idx):.4f}")
    print(f"  True P2 ({true_p2}):  {get_ibd0(k_idx, tp2_idx):.4f}")
    print(f"  False P1 ({false_p1}): {get_ibd0(k_idx, fp1_idx):.4f}")
    print(f"  False P2 ({false_p2}): {get_ibd0(k_idx, fp2_idx):.4f}")

    k_sw = count_switches(k_idx)
    tp1_sw = count_switches(tp1_idx)
    tp2_sw = count_switches(tp2_idx)
    fp1_sw = count_switches(fp1_idx)
    fp2_sw = count_switches(fp2_idx)

    print(f"\n[2] Directionality (ID Switches):")
    print(f"  Kid ({kid}): {k_sw}")
    print(f"  True ({true_p1}, {true_p2}): {tp1_sw} + {tp2_sw} = {tp1_sw + tp2_sw}")
    print(f"  False ({false_p1}, {false_p2}): {fp1_sw} + {fp2_sw} = {fp1_sw + fp2_sw}")

    print(f"\n[3] Mendelian Score (Alleles):")
    print(f"  True Pair:  {calc_mendel(k_idx, tp1_idx, tp2_idx):.5f}")
    print(f"  False Pair: {calc_mendel(k_idx, fp1_idx, fp2_idx):.5f}")
    
    true_metric_ibd = get_ibd0(k_idx, tp1_idx) + get_ibd0(k_idx, tp2_idx)
    false_metric_ibd = get_ibd0(k_idx, fp1_idx) + get_ibd0(k_idx, fp2_idx)
    
    print(f"\n[4] IBD Tie-Breaker (Sum):")
    print(f"  True Pair: {true_metric_ibd:.5f}")
    print(f"  False Pair: {false_metric_ibd:.5f}")
    
    print(f"\n[5] Switch Tie-Breaker (Sum):")
    print(f"  True Pair: {tp1_sw + tp2_sw}")
    print(f"  False Pair: {fp1_sw + fp2_sw}")

else:
    print("Samples not found.")

#%%
# =============================================================================
# DEBUG: HMM CANDIDATE INSPECTION FOR F2_0 (Fixed for Dynamic HMM)
# =============================================================================

print("\n" + "="*60)
print("DEBUG: Inspecting HMM Scores for Sample F3_187")
print("="*60)

target_sample = "F2_1"

if target_sample in sample_names:
    kid_idx = sample_names.index(target_sample)
    
    # 1. Regenerate Grids
    print("Regenerating grids...")
    # FIX: Unpack 3 values now
    id_grid, bin_edges, bin_centers = pedigree_inference.discretize_paintings(gt_painting_result, snps_per_bin=150)
    
    # Use the returned bin_centers directly
    allele_grid = pedigree_inference.convert_id_grid_to_allele_grid(id_grid, bin_centers, gt_block_result)
    
    # 2. Calculate Recombination Counts (for filtering check)
    switches = (id_grid[:, :-1, :] != id_grid[:, 1:, :]) & (id_grid[:, :-1, :] != -1) & (id_grid[:, 1:, :] != -1)
    recomb_counts = np.sum(switches, axis=(1, 2))
    kid_switches = recomb_counts[kid_idx]
    
    print(f"\nTarget {target_sample} Switch Count: {kid_switches}")
    print(f"Filter Threshold (Kid + 5): {kid_switches + 5}")
    
    # 3. Setup HMM Scoring
    from pedigree_inference import run_inheritance_hmm_dynamic
    
    # Pre-calculate Distance-Dependent Transition Costs
    recomb_rate = 5e-8
    error_penalty = 10.0
    
    num_bins = len(bin_centers)
    dists = np.zeros(num_bins)
    dists[1:] = np.diff(bin_centers)
    
    # Prob switch = 1 - exp(-r * d)
    theta = 1.0 - np.exp(-dists * recomb_rate)
    theta = np.clip(theta, 1e-15, 0.5)
    
    switch_costs = np.log(theta)
    stay_costs = np.log(1.0 - theta)
    
    print(f"Running HMM (Dynamic Distances, Error Penalty={error_penalty})...")
    
    child_h0 = allele_grid[kid_idx, :, 0]
    child_h1 = allele_grid[kid_idx, :, 1]
    
    candidates_h0 = []
    candidates_h1 = []
    
    for cand_idx, cand_name in enumerate(sample_names):
        if cand_idx == kid_idx: continue
        
        parent_dip = allele_grid[cand_idx]
        switches = recomb_counts[cand_idx]
        
        # Calculate Scores using the new Dynamic Kernel
        score_h0 = run_inheritance_hmm_dynamic(child_h0, parent_dip, switch_costs, stay_costs, error_penalty)
        score_h1 = run_inheritance_hmm_dynamic(child_h1, parent_dip, switch_costs, stay_costs, error_penalty)
        
        # Check if it passes the directionality filter
        passed_filter = switches <= (kid_switches + 5)
        status = "PASS" if passed_filter else "FAIL"
        
        candidates_h0.append((cand_name, score_h0, switches, status))
        candidates_h1.append((cand_name, score_h1, switches, status))
        
    # 4. Sort and Display
    candidates_h0.sort(key=lambda x: x[1], reverse=True)
    candidates_h1.sort(key=lambda x: x[1], reverse=True)
    
    def print_top_k(title, cands, k=20):
        print(f"\n--- {title} (Top {k}) ---")
        print(f"{'Rank':<5} {'Sample':<10} {'HMM Score':<12} {'Switches':<10} {'Filter'}")
        print("-" * 50)
        for i, (name, score, sw, status) in enumerate(cands[:k]):
            print(f"{i+1:<5} {name:<10} {score:<12.4f} {sw:<10} {status}")
            
    print_top_k(f"Best Parents for {target_sample} Haplotype 0", candidates_h0)
    print_top_k(f"Best Parents for {target_sample} Haplotype 1", candidates_h1)
    
else:
    print(f"Sample {target_sample} not found.")









