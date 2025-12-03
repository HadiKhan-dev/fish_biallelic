import os

# FORCE NUMPY TO USE 1 THREAD PER PROCESS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import math
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
from multiprocess import Pool
import time
import warnings
import networkx as nx
import platform
import importlib

import vcf_data_loader
import analysis_utils
import hap_statistics
import block_haplotypes
import block_linking_naive
import block_linking_em
import simulate_sequences
import viterbi_matching

warnings.filterwarnings("ignore")
np.seterr(divide='ignore',invalid="ignore")

if platform.system() != "Windows":
    os.nice(15)
    print(f"Main process ({os.getpid()}) niceness set to: {os.nice(0)}")

importlib.reload(hap_statistics)
importlib.reload(block_haplotypes)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


#%%
vcf_path = "./fish_vcf/AsAc.AulStuGenome.biallelic.bcf.gz"
contig_name = "chr1"
bcf = vcf_data_loader.read_bcf_file(vcf_path)
contigs = bcf.header.contigs
names = bcf.header.samples
#%%
block_size = 100000
shift_size = 50000
starting = 0
ending = 854

start = time.time()


(pos_broken, keep_flags_broken, reads_array_broken) = vcf_data_loader.cleanup_block_reads_list_cyvcf2(
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
(final_blocks,final_test) = block_linking_naive.generate_long_haplotypes_naive(pos_broken,reads_array_broken,6,keep_flags_broken)
print(time.time()-start)
#%%
haplotype_sites = final_test[0]
haplotype_data = final_test[1]
    
cm = simulate_sequences.concretify_haps(haplotype_data)
pa = simulate_sequences.pairup_haps(cm)

#%%
f1 = simulate_sequences.create_new_generation(pa,haplotype_sites,10,recomb_rate=10**-7,mutate_rate=10**-10)
f2 = simulate_sequences.create_new_generation(f1,haplotype_sites,100,recomb_rate=10**-7,mutate_rate=10**-10)
f3 = simulate_sequences.create_new_generation(f2,haplotype_sites,200,recomb_rate=10**-7,mutate_rate=10**-10)

#%%
all_offspring = [xs for x in [f1,f2,f3] for xs in x]
offspring_genotype_likelihoods = simulate_sequences.combine_into_genotype(all_offspring,haplotype_sites)
#%%
read_depth = 2000
new_reads_array = simulate_sequences.read_sample_all_individuals(all_offspring,read_depth,error_rate=0.02)
#%%
(simd_pos,simd_keep_flags,simd_reads) = simulate_sequences.chunk_up_data(
    haplotype_sites,new_reads_array,
    0,44000000,100000,100000)

start = time.time()
simd_probabalistic_genotypes = analysis_utils.reads_to_probabilities(new_reads_array)
print(time.time()-start)
#%%
start = time.time()
test_haps = block_haplotypes.generate_haplotypes_all(
            simd_pos,simd_reads,simd_keep_flags)

print(time.time()-start)
#%%
start = time.time()
new_test_haps = block_haplotypes.generate_haplotypes_all(
            simd_pos,simd_reads,simd_keep_flags)

print(time.time()-start)
#%%
# Use the specific index
block_idx = 420

# 1. Extract the data for this block
# (Assuming your list of positions is 'all_sites' and list of read arrays is 'all_reads' or 'simd_reads')
# Replace 'all_reads' with whatever your variable name for the list of raw read counts is.
current_pos = simd_pos[block_idx]
current_reads = simd_reads[block_idx] 

print(f"Testing Block {block_idx} with {len(current_pos)} sites...")

start = time.time()

# 2. Run the function with STRICT parameters
# penalty_strength=5.0 is aggressive to kill noise (default 1.0)
results_420 = block_haplotypes.generate_haplotypes_block_robust(
    current_pos,
    current_reads,
    keep_flags=None,
    penalty_strength=1.0,        # <-- The fix for overfitting (try 5.0 or 10.0)
    max_intermediate_haps=25     # Keeps the loop clean
)

print(f"Time taken: {time.time() - start:.2f}s")
print(f"Final Haplotypes Found: {len(results_420[3])}")

# Optional: Print the keys to ensure they are re-indexed correctly (0, 1, 2...)
print("Haplotype Keys:", list(results_420[3].keys()))
#%%

# 1. Unpack the results from your function call
# Assuming the output of generate_haplotypes_block is stored in 'results_420'
# results_420 = (positions, keep_flags, reads_array, final_haps)
# If you ran it interactively, replace these variable names with what you have.
positions = results_420[0]
keep_flags = results_420[1]
reads_array = results_420[2]
final_haps = results_420[3]

print(f"Analyzing fit for {len(final_haps)} haplotypes on {len(reads_array)} samples...")

# 2. Convert Reads to Probabilities (if you don't have this variable in scope)
(_, (probs_array, _)) = analysis_utils.reads_to_probabilities(reads_array)

# 3. Run match_best
# Returns: ( [(best_pair, score), ...], {hap_idx: count}, [all_scores] )
match_output = hap_statistics.match_best_k_limited(
    final_haps, 
    probs_array, 
    keep_flags=keep_flags,
    max_recombinations=3
)

matches_list = match_output[0]  # List of ((h1, h2), error)
usage_counts = match_output[1]  # Dict of hap usage
error_scores = match_output[2]  # Array of error % per sample

# --- 4. PRINT STATISTICS ---

print("\n=== ERROR STATISTICS (Lower is Better) ===")
print(f"Mean Error:   {np.mean(error_scores):.4f}%")
print(f"Median Error: {np.median(error_scores):.4f}%")
print(f"Max Error:    {np.max(error_scores):.4f}%")
print(f"Std Dev:      {np.std(error_scores):.4f}")

# Check for outliers (samples that fit poorly)
bad_fit_threshold = 2.0 # e.g., > 2% mismatch
bad_fits = np.where(error_scores > bad_fit_threshold)[0]
print(f"\nSamples with >{bad_fit_threshold}% error: {len(bad_fits)}")
if len(bad_fits) > 0:
    print(f"Indices of bad fits: {bad_fits}")

print("\n=== HAPLOTYPE USAGE ===")
print(f"{'Hap ID':<8} | {'Count':<8} | {'% of Pop':<10}")
print("-" * 35)
total_hap_slots = len(reads_array) * 2 # Diploid
for k in sorted(final_haps.keys()):
    count = usage_counts.get(k, 0)
    perc = (count / total_hap_slots) * 100
    print(f"{k:<8} | {count:<8} | {perc:.1f}%")

print("\n=== GENOTYPE BREAKDOWN (Top 10) ===")
# Count which PAIRS are most common
geno_counts = {}
for item in matches_list:
    pair = tuple(sorted(item[0])) # Sort so (0,1) == (1,0)
    geno_counts[pair] = geno_counts.get(pair, 0) + 1

sorted_genos = sorted(geno_counts.items(), key=lambda x: x[1], reverse=True)
print(f"{'Pair':<10} | {'Count':<8}")
print("-" * 20)
for pair, count in sorted_genos[:10]:
    print(f"{str(pair):<10} | {count:<8}")
#%%
all_sites = offspring_genotype_likelihoods[0]
all_likelihoods = offspring_genotype_likelihoods[1]
haplotype_data = haplotype_data
#%%

start = time.time()
space_gap = 1

block_likes =  block_linking_em.multiprocess_all_block_likelihoods(all_likelihoods,
        all_sites,test_haps)
print(time.time()-start)

#%%

start = time.time()
space_gap = 1

viterbi_block_likes =  viterbi_matching.viterbi_multiprocess_all_block_likelihoods(all_likelihoods,
        all_sites,test_haps)
print(time.time()-start)


#%%
for j in range(440):
    for s in range(len(block_likes[j][0])):
        val = max(block_likes[j][0][s].values())
        if val < -0.5:
            print(j,s,val)
        
        
#%%
start = time.time()
final_mesh = block_linking_em.generate_transition_probability_mesh(
    all_likelihoods,
    all_sites,
    test_haps,
    max_num_iterations=100,
    learning_rate=1)
print(time.time()-start)

#%%
start = time.time()

# Using the new Viterbi-enhanced function
final_mesh_viterbi = viterbi_matching.viterbi_generate_transition_probability_mesh(
    all_likelihoods,
    all_sites,
    test_haps,
    max_num_iterations=100,
    learning_rate=1
)

print(f"Viterbi Mesh Generation Time: {time.time() - start:.2f}s")

#%%
recombs = hap_statistics.find_intra_block_recombinations(viterbi_block_likes)

error_analysis = hap_statistics.analyze_crossover_types(recombs[0])