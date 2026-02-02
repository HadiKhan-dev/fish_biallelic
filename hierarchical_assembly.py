import numpy as np
import math
import time
from tqdm import tqdm

# Import your specific modules
import block_haplotypes
import block_linking_em
import hmm_matching
import beam_search_core
import analysis_utils

# =============================================================================
# DATA HELPERS
# =============================================================================

def create_downsampled_proxy(block, max_sites=2000):
    """
    Creates a lightweight 'Proxy' of a BlockResult.
    Returns the original block if it's small enough.
    
    Correctly slices internal arrays to maintain consistency.
    """
    total_sites = len(block.positions)
    
    if total_sites <= max_sites:
        return block
        
    stride = math.ceil(total_sites / max_sites)
    
    # Force copy and contiguous to avoid view issues
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

    # Optional: Slice reads/probs if they exist on the block object
    # (Though usually we fetch from global arrays in the pipeline)
    new_reads = None
    if block.reads_count_matrix is not None:
        new_reads = np.ascontiguousarray(block.reads_count_matrix[:, ::stride, :])
        
    new_probs = None
    if block.probs_array is not None:
        new_probs = np.ascontiguousarray(block.probs_array[:, ::stride, :])

    proxy = block_haplotypes.BlockResult(
        positions=new_pos,
        haplotypes=new_haps,
        keep_flags=new_flags,
        reads_count_matrix=new_reads,
        probs_array=new_probs
    )
    
    return proxy

def convert_reconstruction_to_superblock(reconstructed_data, original_blocks, global_probs=None, global_sites=None):
    """
    Packages reconstruction results into a BlockResult (Super-Block).
    
    Args:
        reconstructed_data: Output from beam search reconstruction.
        original_blocks: Original BlockResults that were merged.
        global_probs: Global probability matrix (n_samples, n_total_sites, 3).
        global_sites: Global site positions array.
    """
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
    
    # Get probs - prefer slicing from global_probs if available
    super_probs = None
    if global_probs is not None and global_sites is not None:
        # Find indices of super_positions in global_sites
        indices = np.searchsorted(global_sites, super_positions)
        super_probs = global_probs[:, indices, :]
    else:
        # Fallback: try to get from original blocks
        probs_list = []
        for b in original_blocks:
            if b.probs_array is not None:
                probs_list.append(b.probs_array)
            elif b.reads_count_matrix is not None:
                _, probs = analysis_utils.reads_to_probabilities(b.reads_count_matrix)
                probs_list.append(probs)
        
        if probs_list:
            super_probs = np.concatenate(probs_list, axis=1)
    
    # Concatenate reads from original blocks (optional)
    reads_list = []
    for b in original_blocks:
        if b.reads_count_matrix is not None:
            reads_list.append(b.reads_count_matrix)
    
    super_reads = None
    if reads_list and len(reads_list) == len(original_blocks):
        super_reads = np.concatenate(reads_list, axis=1)

    super_block = block_haplotypes.BlockResult(
        positions=super_positions,
        haplotypes=super_haplotypes,
        keep_flags=super_flags,
        reads_count_matrix=super_reads,
        probs_array=super_probs
    )
    
    return super_block

# =============================================================================
# BATCH WORKER FUNCTION (For Parallel Processing)
# =============================================================================

def _process_single_batch(args):
    """
    Worker function to process a single batch.
    Returns the super_block or None if processing failed.
    """
    (b_idx, start_i, end_i, original_blocks_list, global_probs, global_sites,
     use_hmm_linking, recomb_rate, beam_width, max_founders,
     switch_cost_scale, recomb_penalty, pruning_switch_cost_scale, 
     pruning_recomb_penalty, complexity_penalty_scale, use_standard_bic,
     max_sites_for_linking, verbose) = args
    
    # Convert list back to BlockResults
    original_portion = block_haplotypes.BlockResults(original_blocks_list)
    
    if len(original_portion) < 2:
        # Single block tail - pass through
        return {
            'batch_idx': b_idx,
            'super_block': original_portion[0],
            'status': 'passthrough'
        }

    # 1. Create Proxies
    proxy_list = []
    for b in original_portion:
        proxy_list.append(create_downsampled_proxy(b, max_sites_for_linking))
    portion_proxy = block_haplotypes.BlockResults(proxy_list)

    # --- DYNAMIC PENALTY LOGIC ---
    avg_snps = np.mean([len(b.positions) for b in portion_proxy])
    
    # Determine Add/Swap Penalty (High)
    if switch_cost_scale is not None:
        current_recomb_penalty = avg_snps * switch_cost_scale
    else:
        current_recomb_penalty = recomb_penalty if recomb_penalty is not None else 15.0

    # Determine Pruning Penalty (Low)
    if pruning_switch_cost_scale is not None:
        current_pruning_penalty = avg_snps * pruning_switch_cost_scale
    elif pruning_recomb_penalty is not None:
        current_pruning_penalty = pruning_recomb_penalty
    else:
        current_pruning_penalty = current_recomb_penalty
        
    # Auto-scale complexity penalty
    snp_growth_factor = avg_snps / 200.0
    effective_complexity_scale = complexity_penalty_scale * snp_growth_factor

    # 2. Generate Mesh
    if use_hmm_linking:
        viterbi_emissions = hmm_matching.generate_viterbi_block_emissions(
            global_probs, global_sites, portion_proxy, num_processes=1
        )
        mesh = hmm_matching.generate_transition_probability_mesh_double_hmm(
            None, None, portion_proxy, 
            recomb_rate=recomb_rate, 
            use_standard_baum_welch=False,
            precalculated_viterbi_emissions=viterbi_emissions
        )
    else:
        # Standard Linker - use num_processes=1 to avoid nested Pool
        mesh = block_linking_em.generate_transition_probability_mesh(
            global_probs, global_sites, portion_proxy,
            use_standard_baum_welch=True,
            num_processes=1
        )
        
    # 3. Beam Search on Proxy
    beam_results = beam_search_core.run_full_mesh_beam_search(
        portion_proxy, mesh, beam_width=beam_width, verbose=verbose
    )
    
    if not beam_results:
        return {
            'batch_idx': b_idx,
            'super_block': None,
            'status': 'beam_search_failed'
        }
        
    # 4. Selection on Proxy
    portion_emissions_standard = block_linking_em.generate_all_block_likelihoods(
        global_probs, global_sites, portion_proxy, num_processes=1
    )
    
    fast_mesh = beam_search_core.FastMesh(portion_proxy, mesh)
    
    selected_founders = beam_search_core.select_founders_likelihood(
        beam_results, 
        portion_emissions_standard, 
        fast_mesh,
        max_founders=max_founders,
        recomb_penalty=current_recomb_penalty,
        pruning_recomb_penalty=current_pruning_penalty,
        complexity_penalty_scale=effective_complexity_scale,
        do_refinement=True,
        use_standard_bic=use_standard_bic,
        verbose=verbose
    )
    
    # 5. Reconstruction on Originals
    reconstructed_data = beam_search_core.reconstruct_haplotypes_from_beam(
        selected_founders, fast_mesh, original_portion
    )
    
    # 6. Package (pass global_probs/sites so probs get stored on super_block)
    super_block = convert_reconstruction_to_superblock(
        reconstructed_data, original_portion, global_probs, global_sites
    )
    
    # 7. Structural Chimera Pruning
    super_block = beam_search_core.prune_superblock_chimeras(super_block)
    
    return {
        'batch_idx': b_idx,
        'super_block': super_block,
        'status': 'success' if super_block else 'reconstruction_failed'
    }


# =============================================================================
# MAIN DRIVER
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
                          recomb_penalty=None,          # Raw High penalty (if scale not used)
                          pruning_recomb_penalty=None,  # Raw Low penalty (if scale not used)
                          switch_cost_scale=0.1,      # Scaling factor for High Penalty
                          pruning_switch_cost_scale=None, # Scaling factor for Low Penalty
                          use_standard_bic=False,
                          # Memory Safety
                          max_sites_for_linking=2000,
                          # Parallelization
                          num_processes=16,
                          # Output control
                          verbose=False): 
    """
    Performs one level of Hierarchical Assembly.
    Includes Proxy Logic and Pre-Calculated Emissions to prevent OOM.
    
    Args:
        switch_cost_scale: Factor to calculate High Penalty from block size (e.g. 0.075).
        pruning_switch_cost_scale: Factor to calculate Low Penalty from block size (e.g. 0.001).
                                   If None, falls back to raw pruning_recomb_penalty or defaults to High.
        num_processes: Number of parallel processes for batch processing.
                      Use 1 for sequential execution.
        verbose: If True, print detailed progress from beam search and selection.
    """
    from multiprocess import Pool
    
    total_blocks = len(input_blocks)
    num_batches = math.ceil(total_blocks / batch_size)
    
    print(f"\n--- Starting Hierarchical Step ---")
    print(f"Input: {total_blocks} blocks -> Target: ~{num_batches} Super-Blocks")
    print(f"Processing with {num_processes} workers...")
    
    # Prepare worker arguments
    worker_args = []
    batch_blocks_storage = []
    
    for b_idx in range(num_batches):
        start_i = b_idx * batch_size
        end_i = min(start_i + batch_size, total_blocks)
        
        original_blocks_list = list(input_blocks[start_i:end_i])
        batch_blocks_storage.append(original_blocks_list)
        
        worker_args.append((
            b_idx, start_i, end_i, original_blocks_list, global_probs, global_sites,
            use_hmm_linking, recomb_rate, beam_width, max_founders,
            switch_cost_scale, recomb_penalty, pruning_switch_cost_scale,
            pruning_recomb_penalty, complexity_penalty_scale, use_standard_bic,
            max_sites_for_linking, verbose
        ))
    
    # Process batches
    if num_processes > 1:
        with Pool(num_processes) as pool:
            results = list(tqdm(
                pool.imap(_process_single_batch, worker_args),
                total=num_batches,
                desc="Processing Batches"
            ))
    else:
        # Sequential processing
        results = []
        for args in tqdm(worker_args, desc="Processing Batches"):
            results.append(_process_single_batch(args))
    
    # Sort by batch index and collect super blocks
    results = sorted(results, key=lambda x: x['batch_idx'])
    
    output_super_blocks = []
    success_count = 0
    passthrough_count = 0
    failed_count = 0
    
    for result in results:
        if result['super_block'] is not None:
            output_super_blocks.append(result['super_block'])
            if result['status'] == 'success':
                success_count += 1
            elif result['status'] == 'passthrough':
                passthrough_count += 1
        else:
            failed_count += 1
            
    print(f"Hierarchical Step Complete. Produced {len(output_super_blocks)} Super-Blocks.")
    print(f"  Success: {success_count}, Passthrough: {passthrough_count}, Failed: {failed_count}")
    
    return block_haplotypes.BlockResults(output_super_blocks)