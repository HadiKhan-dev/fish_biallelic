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
import os
import platform
import importlib

import vcf_data_loader
import analysis_utils
import hap_statistics
import block_haplotypes
import block_linking_naive
import block_linking_em
import simulate_sequences

warnings.filterwarnings("ignore")
np.seterr(divide='ignore',invalid="ignore")

if platform.system() != "Windows":
    os.nice(15)
    print(f"Main process ({os.getpid()}) niceness set to: {os.nice(0)}")

importlib.reload(analysis_utils)
importlib.reload(block_haplotypes)

#%%
bcf = vcf_data_loader.read_bcf_file("./fish_vcf/AsAc.AulStuGenome.biallelic.bcf.gz")
contigs = bcf.header.contigs
names = bcf.header.samples
#%%
start = time.time()
block_size = 100000
shift_size = 50000

chr1 = list(vcf_data_loader.break_contig(bcf,"chr1",block_size=block_size,shift=shift_size))

starting = 0
ending = 300


combi = [chr1[i] for i in range(starting,ending)]

(pos_broken,keep_flags_broken,reads_array_broken) = vcf_data_loader.cleanup_block_reads_list(combi)
print(time.time()-start)
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
print("Starting")
print(read_depth)
(simd_pos,simd_keep_flags,simd_reads) = simulate_sequences.chunk_up_data(
    haplotype_sites,new_reads_array,
    0,15000000,100000,100000)
print("Reached HERE")
start = time.time()
simd_probabalistic_genotypes = analysis_utils.reads_to_probabilities(new_reads_array)
print("Reads to probs time:",time.time()-start)
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

space_gap = 1

initp = block_linking_em.initial_transition_probabilities(test_haps,space_gap=space_gap)
block_likes =  block_linking_em.multiprocess_all_block_likelihoods(all_likelihoods,all_sites,test_haps)

#%%
start = time.time()
final_mesh = block_linking_em.generate_transition_probability_mesh(all_likelihoods,
        all_sites,test_haps,max_num_iterations=20,learning_rate=0.5)
print(time.time()-start)
