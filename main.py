import os

# FORCE NUMPY TO USE 1 THREAD PER PROCESS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
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

importlib.reload(viterbi_matching)
importlib.reload(block_linking_em)

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
final_mesh = block_linking_em.generate_transition_probability_mesh(all_likelihoods,
        all_sites,test_haps,max_num_iterations=100,learning_rate=1)
print(time.time()-start)
