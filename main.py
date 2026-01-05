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

import vcf_data_loader
import analysis_utils
import hap_statistics
import block_haplotypes
import block_linking_naive
import block_linking_em
import block_linking_em_old
import simulate_sequences
import hmm_matching
import hmm_matching_testing
import viterbi_likelihood_calculator
import hierarchical_assembly

import block_linking_em_testing

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

(final_blocks, final_long_haps) = block_linking_naive.generate_long_haplotypes_naive(
    block_results, 
    num_long_haps=6
)

print("Time taken:", time.time() - start)
#%%
haplotype_sites = final_long_haps[0]
haplotype_data = final_long_haps[1]
    
concrete_haps = simulate_sequences.concretify_haps(haplotype_data)
pa = simulate_sequences.pairup_haps(concrete_haps)
#%%
f1 = simulate_sequences.create_new_generation(pa, haplotype_sites, 10, recomb_rate=10**-7, mutate_rate=10**-10)
f2 = simulate_sequences.create_new_generation(f1, haplotype_sites, 100, recomb_rate=10**-7, mutate_rate=10**-10)
f3 = simulate_sequences.create_new_generation(f2, haplotype_sites, 200, recomb_rate=10**-7, mutate_rate=10**-10)

#%%
all_offspring = [xs for x in [f1, f2, f3] for xs in x]
all_sites = haplotype_sites

all_likelihoods = simulate_sequences.combine_into_genotype(all_offspring)
#%%
read_depth = 2000
new_reads_array = simulate_sequences.read_sample_all_individuals(all_offspring, read_depth, error_rate=0.02)
#%%
simd_genomic_data = simulate_sequences.chunk_up_data(
    haplotype_sites, new_reads_array,
    0, 5000000, 100000, 100000)

start = time.time()
(simd_site_priors, simd_probabalistic_genotypes) = analysis_utils.reads_to_probabilities(new_reads_array)
print(time.time() - start)
#%%
start = time.time()
test_haps = block_haplotypes.generate_all_block_haplotypes(
            simd_genomic_data)

print(time.time()-start)
#%%
start = time.time()
all_block_emissions_testing = block_linking_em_testing.generate_all_block_likelihoods(
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
final_mesh2 = block_linking_em.generate_transition_probability_mesh(
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
        simd_genomic_data)
    
portion = block_haplotypes.BlockResults(initial_block_results[:10])
print(f"Initial Haplotypes Generated in {time.time() - start_time:.2f}s")
#%%
start = time.time()
portion_standard_mesh = block_linking_em_testing.generate_transition_probability_mesh(
    global_probs, global_sites, portion,max_num_iterations=100,
    use_standard_baum_welch=False)
print("Standard Time:",time.time()-start)
#%%
importlib.reload(hmm_matching_testing)

start = time.time()
portion_hmm_mesh5 = hmm_matching_testing.generate_transition_probability_mesh_double_hmm(
    global_probs, global_sites, portion,recomb_rate=5*10**-7,
    use_standard_baum_welch=True)
print("Deep HMM Time:",time.time()-start)
#%%
import time
import importlib
import hmm_matching_testing  # or block_linking_em, whatever you named the file

# Reload ensures your commented-out @njit changes are picked up
importlib.reload(hmm_matching_testing)

print("Starting Single-Gap Debug Run...")
start = time.time()

# We call the core calculation function directly.
# Crucially: set num_processes=1 to prevent internal pooling
gap_1_transitions2 = hmm_matching_testing.calculate_hap_transition_probabilities(
    full_samples_data=global_probs,    # Your sample data matrix
    sample_sites=global_sites,         # Your genomic positions
    haps_data=portion,                 # Your list of BlockResult objects
    max_num_iterations=5,              # Iterations
    space_gap=1,                       # <--- RESTRICT TO GAP 1
    recomb_rate=5*10**-7,
    learning_rate=1,
    num_processes=1,                   # <--- CRITICAL: Runs in main thread for breakpoints
    use_standard_baum_welch=True
)

print("Single Gap Calculation Time:", time.time() - start)
#%%
# 3. The Function Call
transition_mesh = block_linking_em_old.generate_transition_probability_mesh(
    full_samples_data=global_probs,   # The (Samples x Sites x 3) probability matrix
    sample_sites=global_sites,    # The (Sites,) position array
    haps_data=portion,        # The BlockResults object containing reconstructed blocks
    max_num_iterations=100,          # EM iterations (usually 10-20 is sufficient)
    learning_rate=1.0               # Initial learning rate
)
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
#%%
import time
import numpy as np
import block_linking_em
import hmm_matching



importlib.reload(hmm_matching_testing)
importlib.reload(block_linking_em_testing)

# --- CONFIGURATION ---
NUM_ITERATIONS = 10
GAP_SIZE = 1
RECOMB_RATE_DEBUG = 5*10**-7 # Effectively zero

print(f"Running comparison for {NUM_ITERATIONS} iteration(s) at gap {GAP_SIZE}...")

# =============================================================================
# 1. RUN STANDARD EM (block_linking_em.py)
# =============================================================================
print("\n--- Running Standard EM ---")
start_std = time.time()

# We call the internal calculator directly to control 'space_gap' and 'max_num_iterations'
# instead of generating the full mesh.
std_transitions = block_linking_em_testing.calculate_hap_transition_probabilities(
    global_probs,      # full_samples_data
    global_sites,      # sample_sites
    portion,           # haps_data (BlockResults)
    max_num_iterations=NUM_ITERATIONS,
    space_gap=GAP_SIZE,
    learning_rate=1.0  # Full update, no smoothing history
)

end_std = time.time()
print(f"Standard EM finished in {end_std - start_std:.4f}s")


# =============================================================================
# 2. RUN HMM EM (hmm_matching.py)
# =============================================================================
print("\n--- Running deep HMM EM (Recomb=0) ---")
start_hmm = time.time()

hmm_transitions = hmm_matching_testing.calculate_hap_transition_probabilities(
    global_probs,
    global_sites,
    portion,
    max_num_iterations=NUM_ITERATIONS,
    space_gap=GAP_SIZE,
    recomb_rate=RECOMB_RATE_DEBUG, # Should force rigid blocks
    learning_rate=1.0,
    num_processes=16,  # Use parallel for fair comparison if available
    use_standard_baum_welch=False
)

end_hmm = time.time()
print(f"Two layer EM finished in {end_hmm - start_hmm:.4f}s")


# =============================================================================
# 3. COMPARE RESULTS
# =============================================================================
print("\n--- Comparing Transitions ---")

def compare_transition_dicts(std_dict, vit_dict, label="Forward"):
    """
    Compares two transition dictionaries block by block.
    """
    all_blocks = sorted(list(std_dict.keys()))
    total_diff = 0.0
    count = 0
    max_diff = 0.0
    
    print(f"Checking {label} transitions for {len(all_blocks)} blocks...")
    
    for b_idx in all_blocks:
        std_block = std_dict[b_idx]
        
        if b_idx not in vit_dict:
            print(f"WARNING: Block {b_idx} missing from Viterbi results!")
            continue
            
        vit_block = vit_dict[b_idx]
        
        for key, val_std in std_block.items():
            if key not in vit_block:
                if val_std > 1e-9:
                    print(f"Mismatch Block {b_idx}: Key {key} missing in Viterbi. Std val: {val_std}")
                continue
            
            val_vit = vit_block[key]
            diff = abs(val_std - val_vit)
            total_diff += diff
            count += 1
            if diff > max_diff:
                max_diff = diff
                # Debug print for significant single-sample divergences
                if diff > 0.01:
                    print(f"  [Diff {diff:.4f}] Blk {b_idx} {key}: Std={val_std:.4f} vs Vit={val_vit:.4f}")
             
    avg_diff = total_diff / count if count > 0 else 0.0
    print(f"  > Max Diff: {max_diff:.6f}")
    print(f"  > Avg Diff: {avg_diff:.6f}")
    
    if max_diff < 1e-5:
        print("  > RESULT: MATCH")
    else:
        print("  > RESULT: DIVERGENCE DETECTED")

# Compare Forward Dicts (Index 0)
compare_transition_dicts(std_transitions[0], hmm_transitions[0], label="Forward")

# Compare Backward Dicts (Index 0)
compare_transition_dicts(std_transitions[1], hmm_transitions[1], label="Backward")
#%%














import time
import numpy as np
import copy
import importlib
import block_linking_em
import viterbi_matching

importlib.reload(viterbi_matching)
importlib.reload(block_linking_em)

# --- 1. DATA PREPARATION (SINGLE SAMPLE ISOLATION) ---

print("Isolating Sample 0 for Debugging...")

# 1. Slice the global inputs
# global_probs shape is (Samples, Sites, 3). We take index 0 but keep dims.
global_probs_single = global_probs[0:2, :, :] 

# 2. Slice the BlockResults object (portion)
# We perform a deep copy to ensure we don't mess up the original 'portion' object
portion_single = copy.deepcopy(portion)

# Iterate through the blocks and slice any sample-specific matrices 
# (reads_count_matrix or probs_array) that might be stored inside.
for i in range(len(portion_single)):
    block = portion_single.blocks[i]
    
    if block.reads_count_matrix is not None:
        # Check if it has samples dimension (usually axis 0)
        if block.reads_count_matrix.shape[0] > 1:
            block.reads_count_matrix = block.reads_count_matrix[0:2, :, :]
            
    if block.probs_array is not None:
        if block.probs_array.shape[0] > 1:
            block.probs_array = block.probs_array[0:2, :, :]

# --- CONFIGURATION ---
NUM_ITERATIONS = 2
GAP_SIZE = 1
RECOMB_RATE_DEBUG = 10**-19  # Effectively zero

print(f"Running comparison for {NUM_ITERATIONS} iteration(s) at gap {GAP_SIZE} using ONLY SAMPLE 0...")

# =============================================================================
# 2. RUN STANDARD EM (block_linking_em.py)
# =============================================================================
print("\n--- Running Standard EM ---")
start_std = time.time()

std_transitions = block_linking_em_testing.calculate_hap_transition_probabilities(
    global_probs_single, # <--- Uses single sample
    global_sites,
    portion_single,      # <--- Uses single sample blocks
    max_num_iterations=NUM_ITERATIONS,
    space_gap=GAP_SIZE,
    learning_rate=1.0
)

end_std = time.time()
print(f"Standard EM finished in {end_std - start_std:.4f}s")


# =============================================================================
# 3. RUN VITERBI EM (viterbi_matching.py)
# =============================================================================
print("\n--- Running Viterbi EM (Recomb=0) ---")
start_vit = time.time()

vit_transitions = viterbi_matching_testing.calculate_hap_transition_probabilities(
    global_probs_single, # <--- Uses single sample
    global_sites,
    portion_single,      # <--- Uses single sample blocks
    max_num_iterations=NUM_ITERATIONS,
    space_gap=GAP_SIZE,
    recomb_rate=RECOMB_RATE_DEBUG, 
    learning_rate=1.0,
    num_processes=1  # No need for parallel with 1 sample
)

end_vit = time.time()
print(f"Viterbi EM finished in {end_vit - start_vit:.4f}s")


# =============================================================================
# 4. COMPARE RESULTS
# =============================================================================
print("\n--- Comparing Transitions ---")

def compare_transition_dicts(std_dict, vit_dict, label="Forward"):
    """
    Compares two transition dictionaries block by block.
    """
    all_blocks = sorted(list(std_dict.keys()))
    total_diff = 0.0
    count = 0
    max_diff = 0.0
    
    print(f"Checking {label} transitions for {len(all_blocks)} blocks...")
    
    for b_idx in all_blocks:
        std_block = std_dict[b_idx]
        
        if b_idx not in vit_dict:
            print(f"WARNING: Block {b_idx} missing from Viterbi results!")
            continue
            
        vit_block = vit_dict[b_idx]
        
        for key, val_std in std_block.items():
            if key not in vit_block:
                if val_std > 1e-9:
                    print(f"Mismatch Block {b_idx}: Key {key} missing in Viterbi. Std val: {val_std}")
                continue
            
            val_vit = vit_block[key]
            diff = abs(val_std - val_vit)
            total_diff += diff
            count += 1
            if diff > max_diff:
                max_diff = diff
                # Debug print for significant single-sample divergences
                if diff > 0.01:
                    print(f"  [Diff {diff:.4f}] Blk {b_idx} {key}: Std={val_std:.4f} vs Vit={val_vit:.4f}")
                
    avg_diff = total_diff / count if count > 0 else 0.0
    print(f"  > Max Diff: {max_diff:.6f}")
    print(f"  > Avg Diff: {avg_diff:.6f}")
    
    if max_diff < 1e-5:
        print("  > RESULT: MATCH")
    else:
        print("  > RESULT: DIVERGENCE DETECTED")

# Compare Forward Dicts (Index 0)
compare_transition_dicts(std_transitions[0], vit_transitions[0], label="Forward")

# Compare Backward Dicts (Index 0)
compare_transition_dicts(std_transitions[1], vit_transitions[1], label="Backward")
#%%
hap2 = portion[7].haplotypes[0]
hap4 = portion[7].haplotypes[3]

# Check the Mean Absolute Error (how far apart representatively are they?)
diff = np.mean(np.abs(hap2 - hap4))
print(f"Mean Absolute Difference: {diff:.6f}")

# Check correlation (do they move together?)
# Flatten to 1D for correlation check
flat2 = hap2.flatten()
flat4 = hap4.flatten()
corr = np.corrcoef(flat2, flat4)[0, 1]
print(f"Correlation: {corr:.6f}")

#%%
import numpy as np
import analysis_utils

# 1. Define the Ground Truth Haplotypes (assuming 'haplotype_data' holds the long sequences)
# and 'portion' holds your reconstructed blocks.

print(f"{'GT Hap':<10} | {'Block 7 Rec ID':<15} | {'Block 8 Rec ID':<15} | {'HMM Prob (7->8)':<15}")
print("-" * 65)

for i, long_hap in enumerate(haplotype_data):
    # Find which Reconstructed ID matches the Ground Truth best in Block 7
    # Note: We need the positions from the block to extract the relevant part of the long hap
    
    # --- Block 7 Mapping ---
    b7_pos = portion[7].positions
    b7_rec_haps = portion[7].haplotypes
    
    # Extract the relevant part of the Ground Truth long haplotype
    gt_seq_7 = analysis_utils.get_sample_data_at_sites(long_hap, global_sites, b7_pos)
    
    # Find best match in Reconstructed Haps
    best_id_7 = -1
    min_dist_7 = float('inf')
    
    for rid, rseq in b7_rec_haps.items():
        dist = np.mean(np.abs(gt_seq_7 - rseq))
        if dist < min_dist_7:
            min_dist_7 = dist
            best_id_7 = rid
            
    # --- Block 8 Mapping ---
    b8_pos = portion[8].positions
    b8_rec_haps = portion[8].haplotypes
    
    gt_seq_8 = analysis_utils.get_sample_data_at_sites(long_hap, global_sites, b8_pos)
    
    best_id_8 = -1
    min_dist_8 = float('inf')
    
    for rid, rseq in b8_rec_haps.items():
        dist = np.mean(np.abs(gt_seq_8 - rseq))
        if dist < min_dist_8:
            min_dist_8 = dist
            best_id_8 = rid

    # --- Check Transition Probability ---
    # Look up the transition calculated by HMM
    trans_key = ((7, best_id_7), (8, best_id_8))
    
    prob = 0.0
    if 7 in hmm_transitions[0] and trans_key in hmm_transitions[0][7]:
        prob = hmm_transitions[0][7][trans_key]
        
    print(f"{i:<10} | {best_id_7:<15} | {best_id_8:<15} | {prob:.4f}")
#%%
import numpy as np
import hmm_matching_testing
import analysis_utils

# --- DIAGNOSTIC: CHECK BLOCK 8 EMISSIONS ---

print("Recalculating raw emissions for Block 8...")

# 1. Generate Viterbi Likelihoods for just Block 8
# We wrap portion[8] in a list because the function expects a list
raw_blocks_8 = hmm_matching_testing.generate_viterbi_block_emissions(
    global_probs, 
    global_sites, 
    [portion[8]], 
    num_processes=16
)

# Extract tensor: Shape is (Samples, K_states, Sites)
b8_tensor = raw_blocks_8[0].tensor
# Sum over sites to get Total Log Likelihood per Sample per State
b8_total_ll = np.sum(b8_tensor, axis=2)

# 2. Identify State Indices
hap_keys = sorted(list(portion[8].haplotypes.keys()))
try:
    idx_2 = hap_keys.index(2)
    idx_4 = hap_keys.index(4)
except ValueError:
    print("Error: Haplotype 2 or 4 not found in Block 8 keys:", hap_keys)
    raise

n_haps = len(hap_keys)
# In the flattened state space (N*N), homozygous states are at index: i * N + i
state_2_2 = idx_2 * n_haps + idx_2
state_4_4 = idx_4 * n_haps + idx_4

# 3. Compare Likelihoods
print(f"\nComparing Emission Likelihoods: Homozygous 2 vs Homozygous 4")
print(f"{'Sample':<8} | {'LL(2,2)':<12} | {'LL(4,4)':<12} | {'Diff (4-2)':<12} | {'Winner'}")
print("-" * 65)

count_prefer_4 = 0
count_prefer_2 = 0
toxic_count = 0

# Check first 30 samples (or all if fewer)
for s in range(min(30, b8_total_ll.shape[0])):
    ll_2 = b8_total_ll[s, state_2_2]
    ll_4 = b8_total_ll[s, state_4_4]
    diff = ll_4 - ll_2
    
    winner = "Hap 4" if diff > 0 else "Hap 2"
    if diff > 0: count_prefer_4 += 1
    else: count_prefer_2 += 1
    
    # Check for "Toxicity" (Massive difference indicating broken sites)
    if diff > 50: 
        toxic_count += 1
        winner += " (!)"

    print(f"{s:<8} | {ll_2:<12.2f} | {ll_4:<12.2f} | {diff:<12.2f} | {winner}")

print("-" * 65)
print(f"Samples preferring Hap 4: {count_prefer_4}")
print(f"Samples preferring Hap 2: {count_prefer_2}")
if toxic_count > 0:
    print(f"NOTE: {toxic_count} samples show massive preference (>50 LL), suggesting Hap 2 contains fatal mismatches.")
#%%
import numpy as np
import hmm_matching_testing
import analysis_utils

print("Regenerating internal HMM state...")
# 1. Generate Emissions
raw_blocks = hmm_matching_testing.generate_viterbi_block_emissions(
    global_probs, global_sites, portion, num_processes=16
)
# 2. Run Forward-Backward Pass
S_res, R_res, _ = hmm_matching_testing.global_forward_backward_pass(
    raw_blocks, portion, hmm_transitions, space_gap=GAP_SIZE, recomb_rate=RECOMB_RATE_DEBUG
)

def diagnose_state(block_idx, hap_id, name):
    print(f"\n=== DIAGNOSING STATE ({block_idx}, {hap_id}) [{name}] ===")
    
    hap_keys = sorted(list(portion[block_idx].haplotypes.keys()))
    try:
        h_idx = hap_keys.index(hap_id)
    except ValueError:
        print(f"Haplotype {hap_id} not found in Block {block_idx}")
        return

    n_haps = len(hap_keys)
    state_idx = h_idx * n_haps + h_idx # Homozygous index
    
    # Find sample with MAX probability in this state (even if low)
    s_probs = S_res[block_idx][:, state_idx]
    best_s = np.argmax(s_probs)
    max_val = s_probs[best_s]
    
    print(f"Sample with most mass in ({block_idx}, {hap_id}): Sample {best_s} (LogProb: {max_val:.2f})")
    
    # Check where this sample wants to go in Block 8
    print(f"--- Block 8 Emission Preference for Sample {best_s} ---")
    
    raw_b8 = raw_blocks[8].tensor[best_s]
    raw_b8_sum = np.sum(raw_b8, axis=1)
    hap_keys_8 = sorted(list(portion[8].haplotypes.keys()))
    
    try:
        i2 = hap_keys_8.index(2)
        i4 = hap_keys_8.index(4)
        
        ll_2 = raw_b8_sum[i2 * len(hap_keys_8) + i2]
        ll_4 = raw_b8_sum[i4 * len(hap_keys_8) + i4]
        
        diff = ll_4 - ll_2
        print(f"  LL(8, 2): {ll_2:.2f}")
        print(f"  LL(8, 4): {ll_4:.2f}")
        print(f"  Winner:   {'Hap 4' if diff > 0 else 'Hap 2'} (Diff: {diff:.2f})")
    except ValueError:
        print("  Target haplotypes 2 or 4 not in Block 8.")

# Diagnose the "Zombie" state (7, 2)
diagnose_state(7, 2, "The Zombie")

# Diagnose the "Sink" state (7, 4) where lineage likely went
diagnose_state(7, 4, "The Sink")
#%%
import numpy as np

# Configuration
BLOCK_SRC = 7
BLOCK_DST = 8
TARGET_HAP = 2
TOP_N = 10

print(f"--- Analyzing Top {TOP_N} Samples for State ({BLOCK_SRC}, {TARGET_HAP}, {TARGET_HAP}) ---")

# 1. Get State Index for Block 7 (Source)
hap_keys_src = sorted(list(portion[BLOCK_SRC].haplotypes.keys()))
try:
    idx_src = hap_keys_src.index(TARGET_HAP)
    # Homozygous index: i * N + i
    state_src_idx = idx_src * len(hap_keys_src) + idx_src
except ValueError:
    raise ValueError(f"Haplotype {TARGET_HAP} not found in Block {BLOCK_SRC}")

# 2. Find Top Samples based on Forward Probability (S_res)
# We look at the log-probability of being in state (2,2)
src_probs = S_res[BLOCK_SRC][:, state_src_idx]
# Argsort gives ascending, so we slice from the end to get highest probs
top_sample_indices = np.argsort(src_probs)[-TOP_N:][::-1]

# 3. Analyze Block 8 (Destination) Preferences
hap_keys_dst = sorted(list(portion[BLOCK_DST].haplotypes.keys()))
n_haps_dst = len(hap_keys_dst)

print(f"{'Sample':<8} | {'LogProb @ B7':<14} | {'Preferred Pair @ B8':<20} | {'LogLikelihood @ B8'}")
print("-" * 65)

for s in top_sample_indices:
    # Get Block 7 Confidence
    prob_b7 = src_probs[s]
    
    # Calculate Block 8 Emissions (Sum over sites)
    # raw_blocks[8].tensor is (Samples, K, Sites)
    b8_emissions = np.sum(raw_blocks[BLOCK_DST].tensor[s], axis=1)
    
    # Find the best fitting state in Block 8
    best_state_idx = np.argmax(b8_emissions)
    max_ll = b8_emissions[best_state_idx]
    
    # Decode the state index to Haplotype IDs
    u, v = divmod(best_state_idx, n_haps_dst)
    h1 = hap_keys_dst[u]
    h2 = hap_keys_dst[v]
    
    print(f"{s:<8} | {prob_b7:<14.2f} | ({h1}, {h2}):<20 | {max_ll:.2f}")
    
#%%
import hap_statistics
import analysis_utils
import numpy as np

# 1. Select Block 7
block_idx = 7
block = portion[block_idx]

# 2. Extract the specific probability slice for this block
# We use the helper to get the columns from global_probs that match this block's positions
block_probs = analysis_utils.get_sample_data_at_sites_multiple(
    global_probs, 
    global_sites, 
    block.positions
)

# 3. Run the matching
print(f"Running match_best_vectorised for Block {block_idx}...")
matches, usage, errors = hap_statistics.match_best_vectorised(
    block.haplotypes, 
    block_probs, 
    keep_flags=block.keep_flags
)

# 4. Display Results
print("\n--- Haplotype Usage (Hard Assignment) ---")
# Sort by Haplotype ID
for h_id in sorted(usage.keys()):
    count = usage[h_id]
    print(f"Haplotype {h_id}: assigned to {count} samples")

print(f"\nAverage Reconstruction Error: {np.mean(errors):.4f}%")

# 5. Peak at a few samples (e.g. Sample 11 and 35 from previous context)
targets = [0, 11, 35]
print("\n--- Specific Sample Assignments ---")
for t in targets:
    if t < len(matches):
        pair, err = matches[t]
        print(f"Sample {t:<3} | Assigned Pair: {str(pair):<10} | Error: {err:.4f}%")
        
#%%
import numpy as np
import hmm_matching_testing

s_idx = 112  # One of the samples you found
blk = 7

# 1. Get Indices
hap_keys = sorted(list(portion[blk].haplotypes.keys()))
idx_2 = hap_keys.index(2)
idx_4 = hap_keys.index(4)
n_haps = len(hap_keys)

state_2_2 = idx_2 * n_haps + idx_2
state_4_4 = idx_4 * n_haps + idx_4

# 2. Get Scores
# A. Raw Local Fit (Emission)
# We need to regenerate the raw block emission for this sample
raw_block_7 = hmm_matching_testing.generate_viterbi_block_emissions(
    global_probs, global_sites, [portion[blk]], num_processes=1
)[0]
# Sum over sites
local_ll = np.sum(raw_block_7.tensor[s_idx], axis=1)

ll_2 = local_ll[state_2_2]
ll_4 = local_ll[state_4_4]

# B. Incoming History (Prior from Block 6)
# We extract this from the Forward result S_res
# S_res = Emission + Prior. Therefore: Prior = S_res - Emission
total_score_2 = S_res[blk][s_idx, state_2_2]
total_score_4 = S_res[blk][s_idx, state_4_4]

prior_2 = total_score_2 - ll_2
prior_4 = total_score_4 - ll_4

print(f"--- Analysis of Sample {s_idx} in Block {blk} ---")
print(f"{'Component':<15} | {'State (2,2)':<15} | {'State (4,4)':<15} | {'Winner'}")
print("-" * 60)
print(f"{'Local Fit':<15} | {ll_2:<15.2f} | {ll_4:<15.2f} | {'(2,2)' if ll_2 > ll_4 else '(4,4)'}")
print(f"{'Transition':<15} | {prior_2:<15.2f} | {prior_4:<15.2f} | {'(2,2)' if prior_2 > prior_4 else '(4,4)'}")
print("-" * 60)
print(f"{'HMM Total':<15} | {total_score_2:<15.2f} | {total_score_4:<15.2f} | {'(2,2)' if total_score_2 > total_score_4 else '(4,4)'}")

#%%
import numpy as np
import math
import analysis_utils

# --- CONFIGURATION ---
SAMPLE_IDX = 112
BLOCK_IDX = 7
HAP_A = 2
HAP_B = 4
EPSILON = 1e-2  # The robustness param used in standard EM

print(f"--- FORENSIC ANALYSIS: Sample {SAMPLE_IDX} in Block {BLOCK_IDX} ---")
print(f"Comparing Haplotype {HAP_A} vs Haplotype {HAP_B}\n")

# 1. GET DATA
block = portion[BLOCK_IDX]
positions = block.positions

# CORRECTION: Use the 'multiple' version which handles (Samples, Sites, 3) arrays
# Returns (Samples, Subset_Sites, 3)
subset_probs = analysis_utils.get_sample_data_at_sites_multiple(
    global_probs, 
    global_sites, 
    positions
)
# Extract the specific sample: (Subset_Sites, 3)
sample_data = subset_probs[SAMPLE_IDX]

# Get Haplotypes (Sites x 2) - these are haploid probabilities [P(Ref), P(Alt)]
hap_a_vec = block.haplotypes[HAP_A]
hap_b_vec = block.haplotypes[HAP_B]

# Filter by keep_flags
if block.keep_flags is not None:
    flags = np.array(block.keep_flags, dtype=bool)
    # Check if flags match sliced data length
    if len(flags) == len(sample_data):
        positions = positions[flags]
        sample_data = sample_data[flags]
        hap_a_vec = hap_a_vec[flags]
        hap_b_vec = hap_b_vec[flags]

n_sites = len(positions)

# 2. CALCULATE METRICS PER SITE
site_stats = []

total_dist_a = 0
total_dist_b = 0
total_ll_a = 0
total_ll_b = 0

uniform_prob = 1.0/3.0

for i in range(n_sites):
    # --- A. DISTANCE CALCULATION (match_best style) ---
    
    # Hap A Diploid Projection
    p0_a, p1_a = hap_a_vec[i]
    dip_a = np.array([p0_a**2, 2*p0_a*p1_a, p1_a**2])
    
    # Hap B Diploid Projection
    p0_b, p1_b = hap_b_vec[i]
    dip_b = np.array([p0_b**2, 2*p0_b*p1_b, p1_b**2])
    
    # Sample Probabilities
    s_probs = sample_data[i] # [P(00), P(01), P(11)]
    
    # Distance Weights
    w_mat = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    
    # Dist(Sample, HapDiploids)
    dist_a = np.sum(s_probs * (w_mat @ dip_a))
    dist_b = np.sum(s_probs * (w_mat @ dip_b))
    
    total_dist_a += dist_a
    total_dist_b += dist_b
    
    # --- B. LIKELIHOOD CALCULATION (HMM style) ---
    model_p_a = np.sum(s_probs * dip_a)
    model_p_b = np.sum(s_probs * dip_b)
    
    # Robust Mixture
    final_p_a = (model_p_a * (1 - EPSILON)) + (EPSILON * uniform_prob)
    final_p_b = (model_p_b * (1 - EPSILON)) + (EPSILON * uniform_prob)
    
    ll_a = math.log(max(1e-300, final_p_a))
    ll_b = math.log(max(1e-300, final_p_b))
    
    total_ll_a += ll_a
    total_ll_b += ll_b
    
    # Record Stats
    diff_ll = ll_b - ll_a # Positive means B (Hap 4) is better
    diff_dist = dist_b - dist_a # Positive means A (Hap 2) is better (lower dist)
    
    site_stats.append({
        'pos': positions[i],
        'sample': s_probs,
        'hap_a': hap_a_vec[i],
        'hap_b': hap_b_vec[i],
        'll_a': ll_a,
        'll_b': ll_b,
        'diff_ll': diff_ll,
        'dist_a': dist_a,
        'dist_b': dist_b
    })

# 3. GLOBAL SUMMARY
print("--- GLOBAL SUMMARY ---")
print(f"{'Metric':<15} | {'Hap 2 (A)':<15} | {'Hap 4 (B)':<15} | {'Winner'}")
print("-" * 60)
print(f"{'Total Dist':<15} | {total_dist_a:<15.2f} | {total_dist_b:<15.2f} | {'Hap 2' if total_dist_a < total_dist_b else 'Hap 4'}")
print(f"{'Total LL':<15} | {total_ll_a:<15.2f} | {total_ll_b:<15.2f} | {'Hap 2' if total_ll_a > total_ll_b else 'Hap 4'}")
print("-" * 60)

# 4. IDENTIFY "POISON PILLS"
poison_sites = sorted(site_stats, key=lambda x: x['diff_ll'], reverse=True)

print(f"\n--- TOP 10 SITES PREFERRING HAP 4 (By Likelihood) ---")
print("Look for cases where Hap 2 Likelihood crashes (e.g. -5.0) while Hap 4 survives.")
print(f"{'Pos':<10} | {'Sample(00,01,11)':<20} | {'Hap2(0,1)':<12} | {'Hap4(0,1)':<12} | {'LL(2)':<8} | {'LL(4)':<8} | {'Diff'}")

for i in range(min(10, len(poison_sites))):
    s = poison_sites[i]
    samp_str = f"{s['sample'][0]:.2f},{s['sample'][1]:.2f},{s['sample'][2]:.2f}"
    hap_a_str = f"{s['hap_a'][0]:.2f},{s['hap_a'][1]:.2f}"
    hap_b_str = f"{s['hap_b'][0]:.2f},{s['hap_b'][1]:.2f}"
    print(f"{s['pos']:<10} | {samp_str:<20} | {hap_a_str:<12} | {hap_b_str:<12} | {s['ll_a']:<8.2f} | {s['ll_b']:<8.2f} | +{s['diff_ll']:.2f}")

print(f"\n--- TOP 10 SITES PREFERRING HAP 2 (By Likelihood) ---")
friendly_sites = sorted(site_stats, key=lambda x: x['diff_ll'], reverse=False)

for i in range(min(10, len(friendly_sites))):
    s = friendly_sites[i]
    samp_str = f"{s['sample'][0]:.2f},{s['sample'][1]:.2f},{s['sample'][2]:.2f}"
    hap_a_str = f"{s['hap_a'][0]:.2f},{s['hap_a'][1]:.2f}"
    hap_b_str = f"{s['hap_b'][0]:.2f},{s['hap_b'][1]:.2f}"
    print(f"{s['pos']:<10} | {samp_str:<20} | {hap_a_str:<12} | {hap_b_str:<12} | {s['ll_a']:<8.2f} | {s['ll_b']:<8.2f} | {s['diff_ll']:.2f}")
    
#%%
import numpy as np

def regularize_haplotypes(block_results, min_prob=0.01):
    """
    In-place modifies the block_results to ensure no haplotype probability 
    is ever exactly 0.0 or 1.0. This prevents 'Poison Pill' sites.
    """
    print(f"Regularizing haplotypes (clamping to {min_prob:.3f} - {1-min_prob:.3f})...")
    
    count = 0
    for block in block_results:
        for hap_id in block.haplotypes:
            # Get original array
            original = block.haplotypes[hap_id]
            
            # Clip values
            # We assume the array is (Sites x 2) [Prob_Ref, Prob_Alt]
            # Renormalize rows to ensure sum is 1.0 after clipping
            
            # 1. Clip
            clipped = np.clip(original, min_prob, 1.0 - min_prob)
            
            # 2. Renormalize (so [0.01, 0.01] becomes [0.5, 0.5])
            row_sums = clipped.sum(axis=1)[:, np.newaxis]
            normalized = clipped / row_sums
            
            block.haplotypes[hap_id] = normalized
            count += 1
            
    print(f"Regularized {count} haplotypes across {len(block_results)} blocks.")

# --- APPLY THE FIX ---
# Apply this to your data variable 'portion' BEFORE running the HMM steps
regularize_haplotypes(portion, min_prob=0.2) 