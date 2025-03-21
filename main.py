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

import vcf_data_loader
import analysis_utils
import hap_statistics
import block_haplotypes
import block_linking_naive

warnings.filterwarnings("ignore")
np.seterr(divide='ignore',invalid="ignore")
pass

#%%
bcf = vcf_data_loader.read_bcf_file("./fish_vcf/AsAc.AulStuGenome.biallelic.bcf.gz")
contigs = bcf.header.contigs
names = bcf.header.samples
#%%
start = time.time()
block_size = 100000
shift_size = 50000

chr1 = list(vcf_data_loader.break_contig(bcf,"chr1",block_size=block_size,shift=shift_size))

starting = 700
ending = len(chr1)


combi = [chr1[i] for i in range(starting,ending)]

(pos_broken,keep_flags_broken,reads_array_broken) = vcf_data_loader.cleanup_block_reads_list(combi)
print(time.time()-start)
#%%
start = time.time()
(final_blocks,final_test) = block_linking_naive.generate_long_haplotypes_naive(pos_broken,reads_array_broken,6,keep_flags_broken)
print(time.time()-start)
#%%
mark = get_dips_from_long_haps(final_test[1])

full_data = get_vcf_subset(bcf,"chr1",starting*shift_size,(ending+1)*shift_size)
(full_positions,full_keep_flags,full_reads_array) = cleanup_block_reads(full_data,min_frequency=0)
(full_site_priors,full_probs_array) = reads_to_probabilities(full_reads_array)

recomb_rate = 10**-8
resu = compute_likeliest_path(mark,full_probs_array[0],
            full_positions,keep_flags=full_keep_flags,
            recomb_rate=recomb_rate,value_error_rate=10**-3)    

make_heatmap_path(resu[0],6)

