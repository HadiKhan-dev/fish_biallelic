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
import hmm_matching_testing
import viterbi_likelihood_calculator
import beam_search_core
import paint_samples

import hierarchical_assembly


warnings.filterwarnings("ignore")
np.seterr(divide='ignore',invalid="ignore")

if platform.system() != "Windows":
    os.nice(15)
    print(f"Main process ({os.getpid()}) niceness set to: {os.nice(0)}")


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


#%%
vcf_path = "./fish_vcf/AsAc.AulStuGenome.biallelic.bcf.gz"
contig_name = "chr1"

block_size = 100000
shift_size = 50000
starting = 0
ending = 100

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

print(genomic_data.positions[-1][-1])
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
    
concrete_haps = simulate_sequences.concretify_haps(haplotype_data)
parents = simulate_sequences.pairup_haps(concrete_haps)
#%%
f1 = simulate_sequences.create_new_generation(parents, haplotype_sites, 10, recomb_rate=5*10**-8, mutate_rate=10**-10)
f2 = simulate_sequences.create_new_generation(f1, haplotype_sites, 100, recomb_rate=5*10**-8, mutate_rate=10**-10)
f3 = simulate_sequences.create_new_generation(f2, haplotype_sites, 200, recomb_rate=5*10**-8, mutate_rate=10**-10)

#%%
all_offspring = [xs for x in [f1, f2, f3] for xs in x]
all_sites = haplotype_sites

all_likelihoods = simulate_sequences.combine_into_genotype(all_offspring)
#%%
read_depth = 2000
new_reads_array = simulate_sequences.read_sample_all_individuals(all_offspring, read_depth, error_rate=0.02)
#%%
simd_genomic_data = simulate_sequences.chunk_up_data(
    haplotype_sites, 
    new_reads_array,
    0, 5000000, 
    0, 0, # Block Size / Shift ignored
    use_snp_count=True,
    snps_per_block=200,
    snp_shift=200 # Disjoint blocks (Shift = Size)
)

start = time.time()
(simd_site_priors, simd_probabalistic_genotypes) = analysis_utils.reads_to_probabilities(new_reads_array)
print(time.time() - start)
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
start = time.time()
all_block_emissions = block_linking_em.generate_all_block_likelihoods(
    simd_probabalistic_genotypes,
    haplotype_sites,
    test_haps,
    num_processes=16)
print(time.time()-start)

start = time.time()
viterbi_emissions = viterbi_likelihood_calculator.generate_all_viterbi_likelihoods(
    simd_probabalistic_genotypes,
    all_sites,
    test_haps,recomb_rate=10**-10000)        
print(time.time()-start)  
#%%
all_paths = []

for long_hap in haplotype_data:
    path = analysis_utils.map_haplotype_to_blocks(long_hap, test_haps) 
    all_paths.append(path)          
#%%
start = time.time()
final_mesh = block_linking_em.generate_transition_probability_mesh(
    simd_probabalistic_genotypes,
    all_sites,
    test_haps,
    max_num_iterations=20,
    learning_rate=1)
print(time.time()-start)
#%%
start = time.time()

# Using the new Viterbi-enhanced function
final_mesh_viterbi = hmm_matching.generate_transition_probability_mesh_double_hmm(
    simd_probabalistic_genotypes,
    all_sites,
    test_haps,
    max_num_iterations=20,
    learning_rate=1,
)

print(f"Viterbi Mesh Generation Time: {time.time() - start:.2f}s")

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
start_time = time.time()
portion1 = block_haplotypes.BlockResults(initial_block_results[:10])
portion2 = block_haplotypes.BlockResults(initial_block_results[10:20])
portion3 = block_haplotypes.BlockResults(initial_block_results[20:30])
portion4 = block_haplotypes.BlockResults(initial_block_results[30:40])
portion = block_haplotypes.BlockResults(initial_block_results[40:50])
print(f"Initial Haplotypes Generated in {time.time() - start_time:.2f}s")
#%%
portion_emissions = block_linking_em.generate_all_block_likelihoods(
    global_probs,      # (Samples, Sites, 3)
    global_sites,      # (Sites,)
    portion,           # The exact BlockResults object used for Beam Search
    num_processes=16
)
#%%
start = time.time()
portion_standard_mesh = block_linking_em.generate_transition_probability_mesh(
    global_probs, global_sites, portion,max_num_iterations=100,
    use_standard_baum_welch=False)
print("Standard Time:",time.time()-start)
#%%

start = time.time()
portion_hmm_mesh = hmm_matching.generate_transition_probability_mesh_double_hmm(
    global_probs, global_sites, portion,recomb_rate=5*10**-8,
    use_standard_baum_welch=False)
print("Deep HMM Time:",time.time()-start)

#%%
start = time.time()
super_blocks_level_1 = hierarchical_assembly.run_hierarchical_step(
    initial_block_results,
    global_probs,
    global_sites,
    batch_size=10,
    recomb_rate=5e-8,
    max_founders=16,
    complexity_penalty_scale=0.25 # Lower penalty for micro-assembly
)
print(f"Time for 12.5% of Chr1: {time.time()-start}")
#%%
start = time.time()
super_blocks_level_2 = hierarchical_assembly.run_hierarchical_step(
    super_blocks_level_1,
    global_probs,
    global_sites,
    batch_size=10,
    use_hmm_linking=True,
    recomb_rate=5e-8,          # KEEP CONSTANT (Biologically accurate)
    beam_width=200,
    max_founders=16,
    recomb_penalty=15.0,       
    complexity_penalty_scale=0.1      
)
print(f"Time for 12.5% of Chr1, 2nd round: {time.time()-start}")

#%%
start = time.time()
super_blocks_level_3 = hierarchical_assembly.run_hierarchical_step(
    super_blocks_level_2,
    global_probs,
    global_sites,
    batch_size=10,
    use_hmm_linking=True,
    recomb_rate=5e-8,          # KEEP CONSTANT (Biologically accurate)
    beam_width=100,
    max_founders=16,
    recomb_penalty=20.0,       
    complexity_penalty_scale=0.1       
)
print(f"Time for 12.5% of Chr1, 3rd round: {time.time()-start}")
#%%























#%%
import numpy as np
import pandas as pd
from collections import Counter
import hap_statistics

def analyze_haplotype_transition(block_results, 
                                 source_block_idx, 
                                 source_hap_id, 
                                 target_block_idx):
    """
    Analyzes which haplotypes appear in a target block for the subset of samples
    that possess a specific haplotype in a source block.

    Args:
        block_results: The list of BlockResult objects.
        source_block_idx: Index of the 'from' block (e.g., 8).
        source_hap_id: The ID of the haplotype to filter by (e.g., 4).
        target_block_idx: Index of the 'to' block (e.g., 9).

    Returns:
        pd.DataFrame: Statistics of haplotype usage in the target block for the filtered subset.
    """
    
    # 1. Validation
    if source_block_idx >= len(block_results) or target_block_idx >= len(block_results):
        print(f"Error: Block indices out of range (Max: {len(block_results)-1})")
        return None
    
    source_block = block_results[source_block_idx]
    
    # Check if source hap exists (handles int vs string keys)
    if source_hap_id not in source_block.haplotypes:
        print(f"Warning: Haplotype {source_hap_id} not found in Block {source_block_idx} keys: {list(source_block.haplotypes.keys())}")

    # 2. Get Best Matches (Viterbi/Likelihood) for both blocks
    # This uses the cached probability calculations if available in block_results, 
    # otherwise calculates them from reads.
    print(f"Calculating best matches for Block {source_block_idx}...")
    source_matches = hap_statistics.combined_best_hap_matches(block_results[source_block_idx])
    # Structure: [((hap_A, hap_B), error), ...] for each sample
    source_pairs = source_matches[0] 

    print(f"Calculating best matches for Block {target_block_idx}...")
    target_matches = hap_statistics.combined_best_hap_matches(block_results[target_block_idx])
    target_pairs = target_matches[0]

    # 3. Filter Samples
    relevant_indices = []
    
    for i, ((h1, h2), error) in enumerate(source_pairs):
        # Check if the sample possesses the source haplotype (on either allele)
        if h1 == source_hap_id or h2 == source_hap_id:
            relevant_indices.append(i)

    total_samples = len(source_pairs)
    subset_count = len(relevant_indices)

    print(f"\n--- Transition Report ---")
    print(f"Filter Condition: Block {source_block_idx} must contain Hap {source_hap_id}")
    print(f"Total Samples: {total_samples}")
    print(f"Matching Samples: {subset_count} ({subset_count/total_samples:.2%})")
    
    if subset_count == 0:
        return pd.DataFrame()

    # 4. Compile Target Statistics
    target_allele_counts = Counter()
    target_samples_carrying = Counter() # Counts samples that have at least one copy
    
    for idx in relevant_indices:
        (t1, t2), t_err = target_pairs[idx]
        
        # Count total allele occurrences
        target_allele_counts[t1] += 1
        target_allele_counts[t2] += 1
        
        # Count unique samples carrying the hap
        unique_in_sample = set([t1, t2])
        for h in unique_in_sample:
            target_samples_carrying[h] += 1

    # 5. Create DataFrame
    stats = []
    total_alleles_in_subset = subset_count * 2
    
    for hap_id, allele_count in target_allele_counts.items():
        sample_carrier_count = target_samples_carrying[hap_id]
        
        stats.append({
            "Target_Block": target_block_idx,
            "Hap_ID": hap_id,
            "Allele_Count": allele_count,
            "Allele_Freq_Subset": allele_count / total_alleles_in_subset,
            "Sample_Carrier_Count": sample_carrier_count,
            "Sample_Carrier_Perc": sample_carrier_count / subset_count
        })

    df = pd.DataFrame(stats)
    
    # Sort by frequency
    df = df.sort_values("Allele_Count", ascending=False).reset_index(drop=True)
    
    # Format percentages for display
    df["Allele_Freq_Subset"] = df["Allele_Freq_Subset"].map("{:.1%}".format)
    df["Sample_Carrier_Perc"] = df["Sample_Carrier_Perc"].map("{:.1%}".format)

    return df

# --- Usage Example ---
# Assuming you have your 'block_results' list loaded:
stats_df = analyze_haplotype_transition(portion,
    source_block_idx=6, source_hap_id=0, target_block_idx=7)
print(stats_df)