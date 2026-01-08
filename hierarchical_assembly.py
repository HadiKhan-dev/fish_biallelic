import numpy as np
import math
import time
from tqdm import tqdm
from multiprocess import Pool

# Import your specific modules
import block_haplotypes
import block_linking_em
import hmm_matching
import beam_search_core
import analysis_utils

# =============================================================================
# 1. PROXY MANAGEMENT
# =============================================================================

def create_downsampled_proxy(block, max_sites=5000):
    """
    Creates a lightweight 'Proxy' of a BlockResult.
    Downsamples sites to ensure HMM tensors fit in RAM.
    """
    total_sites = len(block.positions)
    
    if total_sites <= max_sites:
        return block
        
    stride = math.ceil(total_sites / max_sites)
    
    # Slice Data
    new_pos = np.ascontiguousarray(block.positions[::stride])
    
    new_haps = {}
    for k, v in block.haplotypes.items():
        if v.ndim > 1:
            new_haps[k] = np.ascontiguousarray(v[::stride, :])
        else:
            new_haps[k] = np.ascontiguousarray(v[::stride])
            
    if block.keep_flags is not None:
        new_flags = np.ascontiguousarray(block.keep_flags[::stride])
    else:
        new_flags = None
        
    # Check consistency
    if new_flags is not None:
        assert len(new_pos) == len(new_flags)

    # Note: We do NOT store reads/probs here to save memory. 
    # They are re-extracted using the sparse positions in the local generators below.
    proxy = block_haplotypes.BlockResult(
        positions=new_pos,
        haplotypes=new_haps,
        keep_flags=new_flags,
        reads_count_matrix=None,
        probs_array=None
    )
    
    return proxy

def convert_reconstruction_to_superblock(reconstructed_data, original_blocks):
    """Packages beam search result into a new BlockResult."""
    if not reconstructed_data:
        return None

    super_haplotypes = {}
    for i, data in enumerate(reconstructed_data):
        super_haplotypes[i] = data['haplotype'] 

    super_positions = reconstructed_data[0]['positions']

    super_flags = []
    for b in original_blocks:
        if b.keep_flags is not None:
            super_flags.extend(b.keep_flags)
        else:
            super_flags.extend(np.ones(len(b.positions), dtype=int))
    super_flags = np.array(super_flags)

    super_block = block_haplotypes.BlockResult(
        positions=super_positions,
        haplotypes=super_haplotypes,
        keep_flags=super_flags,
        reads_count_matrix=None,
        probs_array=None 
    )
    
    return super_block

# =============================================================================
# 2. LOCAL GENERATORS (Fixing the Index Error)
# =============================================================================

def extract_sparse_data(global_probs, global_sites, block_positions):
    """
    Correctly extracts non-contiguous (strided) data from global arrays.
    Replaces analysis_utils.get_sample_data_at_sites_multiple for proxies.
    """
    # Find indices of the block's positions in the global site array
    # Assumes both are sorted
    indices = np.searchsorted(global_sites, block_positions)
    
    # Safety check: ensure we didn't go out of bounds (if block_pos not in global)
    # In a clean pipeline, this shouldn't happen, but good to clamp
    indices = np.clip(indices, 0, len(global_sites) - 1)
    
    # Fancy Indexing to get specific columns
    # Shape: (Samples, Len_Block_Pos, 3)
    return global_probs[:, indices, :]

def calculate_proxy_emissions(global_probs, global_sites, proxies, num_processes=16):
    """
    Generates StandardBlockLikelihoods for proxies using correct sparse extraction.
    """
    tasks = []
    params = {
        'log_likelihood_base': math.e, 
        'robustness_epsilon': 1e-2
    }
    
    for block in proxies:
        # Extract data correctly for strided positions
        block_samples = extract_sparse_data(global_probs, global_sites, block.positions)
        tasks.append((block_samples, block, params))
        
    if num_processes > 1 and len(tasks) > 1:
        with Pool(num_processes) as pool:
            # Call the worker from the imported module
            results = pool.map(block_linking_em._worker_calculate_single_block_likelihood, tasks)
    else:
        results = list(map(block_linking_em._worker_calculate_single_block_likelihood, tasks))
        
    return block_linking_em.StandardBlockLikelihoods(results)

def calculate_proxy_mesh(global_probs, global_sites, proxies, recomb_rate, num_processes=16):
    """
    Generates Transition Mesh for proxies using correct sparse extraction.
    Re-implements the HMM driver loop to handle pre-calculated sparse data.
    """
    # 1. Pre-calculate Viterbi Emissions (P(Data|State))
    # We use the worker from hmm_matching
    tasks = []
    params = {'robustness_epsilon': 1e-2}
    
    for block in proxies:
        block_samples = extract_sparse_data(global_probs, global_sites, block.positions)
        tasks.append((block_samples, block, params))
        
    if num_processes > 1 and len(tasks) > 1:
        with Pool(num_processes) as pool:
            viterbi_emissions_list = pool.map(hmm_matching._worker_generate_viterbi_emissions, tasks)
    else:
        viterbi_emissions_list = list(map(hmm_matching._worker_generate_viterbi_emissions, tasks))
    
    # 2. Run Mesh Generation (Gap Workers)
    # We need to calculate transitions for all gaps.
    max_gap = len(proxies) - 1
    gaps = list(range(1, max_gap + 1))
    
    # We define a local worker to run the HMM loop on pre-calculated data
    # to avoid pickling huge global arrays
    
    results_map = {}
    
    # Prepare data for all gaps (shared)
    # We run the loop serially or parallel over gaps
    # Since we already have emissions, the HMM is fast.
    
    # Initialize transitions (Uniform)
    # Note: We calculate this once, but it changes per gap inside the loop? 
    # No, initial_transition_probabilities generates for specific gap.
    
    def _local_gap_worker(gap):
        # 2a. Initial Trans
        current_trans = block_linking_em.initial_transition_probabilities(proxies, gap)
        
        # 2b. EM Loop
        # We reuse the logic from hmm_matching but pass our pre-calculated emissions
        prev_ll = -np.inf
        for it in range(10): # Max Iter
            lr = max(1.0 * (0.9**it), 0.1)
            
            # E-Step
            S_res, R_res, curr_ll = hmm_matching.global_forward_backward_pass(
                viterbi_emissions_list, proxies, current_trans, gap, recomb_rate
            )
            
            # M-Step
            new_trans = hmm_matching.update_transitions_layered_hmm(
                S_res, R_res, proxies, current_trans, gap, use_standard_baum_welch=False
            )
            
            # Smooth
            current_trans = analysis_utils.smoothen_probs_vectorized(current_trans, new_trans, lr)
            
            # Converge
            if prev_ll != -np.inf and abs(curr_ll - prev_ll) / abs(prev_ll) < 1e-4:
                break
            prev_ll = curr_ll
            
        return current_trans

    # Run gaps in parallel
    if num_processes > 1:
        with Pool(num_processes) as pool:
            results = pool.map(_local_gap_worker, gaps)
    else:
        results = list(map(_local_gap_worker, gaps))
        
    mesh_dict = dict(zip(gaps, results))
    return block_linking_em.TransitionMesh(mesh_dict)

# =============================================================================
# 3. MAIN RUNNER
# =============================================================================

def run_hierarchical_step(input_blocks, global_probs, global_sites,
                          batch_size=10,
                          # Linking Parameters
                          use_hmm_linking=True,
                          recomb_rate=5e-8,
                          # Search Parameters
                          beam_width=200,
                          # Selection Parameters
                          max_founders=12,
                          complexity_penalty_scale=0.1,
                          recomb_penalty=15.0,
                          use_standard_bic=False,
                          # Memory Safety
                          max_sites_for_linking=5000): 
    
    total_blocks = len(input_blocks)
    num_batches = math.ceil(total_blocks / batch_size)
    output_super_blocks = []
    
    print(f"\n--- Starting Hierarchical Step ---")
    print(f"Input: {total_blocks} blocks -> Target: ~{num_batches} Super-Blocks")
    print(f"Params: Batch={batch_size} | HMM={use_hmm_linking} | Scale={complexity_penalty_scale}")
    
    for b_idx in tqdm(range(num_batches), desc="Processing Batches"):
        start_i = b_idx * batch_size
        end_i = min(start_i + batch_size, total_blocks)
        
        original_portion = block_haplotypes.BlockResults(input_blocks[start_i:end_i])
        
        if len(original_portion) < 2:
            print(f"\n[Batch {b_idx}] Single block tail. Passing through.")
            output_super_blocks.append(original_portion[0])
            continue

        # 1. Create Proxies
        proxy_list = []
        for b in original_portion:
            proxy_list.append(create_downsampled_proxy(b, max_sites_for_linking))
        portion_proxy = block_haplotypes.BlockResults(proxy_list)

        # 2. Emissions on Proxy (Using LOCAL extractor)
        portion_emissions = calculate_proxy_emissions(
            global_probs, global_sites, portion_proxy, num_processes=16
        )
        
        # 3. Mesh on Proxy (Using LOCAL extractor/loop)
        if use_hmm_linking:
            mesh = calculate_proxy_mesh(
                global_probs, global_sites, portion_proxy, recomb_rate, num_processes=16
            )
        else:
            # Standard linking handles proxies poorly if not updated, 
            # but usually we use HMM. If standard needed, implement similar local wrapper.
            # Fallback to HMM logic or raise warning.
            print("Warning: Standard Linking not optimized for proxies. Using HMM.")
            mesh = calculate_proxy_mesh(
                global_probs, global_sites, portion_proxy, recomb_rate, num_processes=16
            )
            
        # 4. Beam on Proxy
        beam_results = beam_search_core.run_full_mesh_beam_search(
            portion_proxy, mesh, beam_width=beam_width
        )
        
        if not beam_results:
            print(f"\n[Batch {b_idx}] Beam Search failed. Skipping.")
            continue
            
        # 5. Selection on Proxy
        fast_mesh = beam_search_core.FastMesh(portion_proxy, mesh)
        
        selected_founders = beam_search_core.select_founders_likelihood(
            beam_results, 
            portion_emissions, 
            fast_mesh,
            max_founders=max_founders,
            recomb_penalty=recomb_penalty,
            complexity_penalty_scale=complexity_penalty_scale,
            do_refinement=True,
            use_standard_bic=use_standard_bic
        )
        
        # 6. Reconstruction on Originals (Using FastMesh map from Proxy)
        # Key Mapping is identical between Proxy and Original
        reconstructed_data = beam_search_core.reconstruct_haplotypes_from_beam(
            selected_founders, fast_mesh, original_portion
        )
        
        super_block = convert_reconstruction_to_superblock(reconstructed_data, original_portion)
        
        if super_block:
            output_super_blocks.append(super_block)
            
    print(f"Hierarchical Step Complete. Produced {len(output_super_blocks)} Super-Blocks.")
    
    return block_haplotypes.BlockResults(output_super_blocks)