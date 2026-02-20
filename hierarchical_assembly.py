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
import chimera_resolution

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
        indices = np.searchsorted(global_sites, super_positions)
        super_probs = global_probs[:, indices, :]
    else:
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


def compute_max_gap(blocks, recomb_rate, n_generations, recomb_tolerance):
    """
    Compute the maximum gap to use for beam search transition lookups.
    
    gap=1 corresponds to ~0bp (adjacent block boundary).
    gap=G spans (G-1) full blocks of physical distance.
    
    We want: (G-1) * avg_block_span * recomb_rate * n_generations < recomb_tolerance
    
    Args:
        blocks: List of BlockResult/proxy objects with .positions.
        recomb_rate: Per-bp recombination rate.
        n_generations: Average number of generations between founders and samples.
        recomb_tolerance: Maximum expected recombinations before we stop trusting linkage.
    
    Returns:
        int: max_gap (always >= 1).
    """
    block_spans = [b.positions[-1] - b.positions[0] for b in blocks if len(b.positions) > 1]
    if not block_spans:
        return 1
    
    avg_block_span = np.mean(block_spans)
    recombs_per_step = avg_block_span * recomb_rate * n_generations
    
    if recombs_per_step <= 0:
        return len(blocks)
    
    max_gap = max(1, 1 + int(math.floor(recomb_tolerance / recombs_per_step)))
    
    return max_gap


# =============================================================================
# BATCH WORKER FUNCTION (For Parallel Processing)
# =============================================================================

def _process_single_batch(args):
    """
    Worker function to process a single batch.
    
    Uses chimera_resolution.select_and_resolve for sub-block Viterbi
    selection, top-N swap refinement, BIC pruning, and chimera resolution.
    
    Returns dict with 'batch_idx', 'super_block', and 'status'.
    """
    (b_idx, start_i, end_i, original_blocks_list, global_probs, global_sites,
     use_hmm_linking, recomb_rate, beam_width, max_founders,
     max_sites_for_linking, n_generations, recomb_tolerance,
     top_n_swap, max_cr_iterations, paint_penalty, min_hotspot_samples,
     cc_scale, verbose) = args
    
    # Convert list back to BlockResults
    original_portion = block_haplotypes.BlockResults(original_blocks_list)
    
    if len(original_portion) < 2:
        # Single block tail - pass through
        return {
            'batch_idx': b_idx,
            'super_block': original_portion[0],
            'status': 'passthrough'
        }

    # 1. Create Proxies (for mesh generation and beam search only)
    proxy_list = []
    for b in original_portion:
        proxy_list.append(create_downsampled_proxy(b, max_sites_for_linking))
    portion_proxy = block_haplotypes.BlockResults(proxy_list)

    # 2. Compute Max Gap
    if n_generations is not None and recomb_tolerance is not None:
        beam_max_gap = compute_max_gap(original_blocks_list, recomb_rate, 
                                        n_generations, recomb_tolerance)
    else:
        beam_max_gap = None

    # 3. Generate Mesh (on proxy blocks)
    if use_hmm_linking:
        viterbi_emissions = hmm_matching.generate_viterbi_block_emissions(
            global_probs, global_sites, portion_proxy, num_processes=1
        )
        mesh = hmm_matching.generate_transition_probability_mesh_double_hmm(
            None, None, portion_proxy, 
            recomb_rate=recomb_rate, 
            use_standard_baum_welch=False,
            precalculated_viterbi_emissions=viterbi_emissions,
            num_processes=1
        )
    else:
        mesh = block_linking_em.generate_transition_probability_mesh(
            global_probs, global_sites, portion_proxy,
            use_standard_baum_welch=True,
            num_processes=1
        )
        
    # 4. Beam Search on Proxy
    beam_results = beam_search_core.run_full_mesh_beam_search(
        portion_proxy, mesh, beam_width=beam_width, 
        max_gap=beam_max_gap, verbose=verbose
    )
    
    if not beam_results:
        return {
            'batch_idx': b_idx,
            'super_block': None,
            'status': 'beam_search_failed'
        }
        
    fast_mesh = beam_search_core.FastMesh(portion_proxy, mesh)
    
    # 5. Selection + Swap + CR (on original blocks via sub-block emissions)
    resolved_beam = chimera_resolution.select_and_resolve(
        beam_results=beam_results,
        fast_mesh=fast_mesh,
        batch_blocks=list(original_portion),
        global_probs=global_probs,
        global_sites=global_sites,
        max_founders=max_founders,
        top_n_swap=top_n_swap,
        max_cr_iterations=max_cr_iterations,
        paint_penalty=paint_penalty,
        min_hotspot_samples=min_hotspot_samples,
        cc_scale=cc_scale,
    )
    
    # 6. Reconstruction on Originals
    reconstructed_data = beam_search_core.reconstruct_haplotypes_from_beam(
        resolved_beam, fast_mesh, original_portion
    )
    
    # 7. Package
    super_block = convert_reconstruction_to_superblock(
        reconstructed_data, original_portion, global_probs, global_sites
    )
    
    # 8. Structural Chimera Pruning
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
                          # Memory Safety
                          max_sites_for_linking=2000,
                          # Max Gap Parameters
                          n_generations=None,
                          recomb_tolerance=0.5,
                          # Chimera Resolution Parameters
                          top_n_swap=20,
                          max_cr_iterations=10,
                          paint_penalty=10.0,
                          min_hotspot_samples=5,
                          cc_scale=0.2,
                          # Parallelization
                          num_processes=16,
                          # Output control
                          verbose=False): 
    """
    Performs one level of Hierarchical Assembly.
    
    Uses sub-block Viterbi forward selection, top-N swap refinement,
    BIC pruning with actual-size CC, and chimera resolution.
    
    Args:
        input_blocks: BlockResults from the previous level.
        global_probs: (n_samples, n_sites, 3) genotype probability array.
        global_sites: Array of genomic site positions.
        batch_size: Number of input blocks per batch.
        use_hmm_linking: If True, use HMM matching for mesh. If False, use EM linking.
        recomb_rate: Per-bp recombination rate for HMM mesh.
        beam_width: Number of beam search candidates.
        max_founders: Maximum number of founders to retain.
        max_sites_for_linking: Maximum sites for proxy blocks used in mesh generation.
        n_generations: Average meioses between founders and samples (for max_gap).
                       If None, max_gap is unlimited.
        recomb_tolerance: Maximum expected recombinations before linkage is degraded.
        top_n_swap: Number of candidates to evaluate per swap position.
        max_cr_iterations: Maximum chimera resolution iterations.
        paint_penalty: Viterbi penalty for sample painting in CR.
        min_hotspot_samples: Minimum samples for a hotspot to be actionable.
        cc_scale: Complexity cost scaling factor.
        num_processes: Number of parallel processes for batch processing.
        verbose: If True, print detailed progress.
    """
    from multiprocess import Pool
    
    total_blocks = len(input_blocks)
    num_batches = math.ceil(total_blocks / batch_size)
    
    # Preview
    if n_generations is not None:
        preview_max_gap = compute_max_gap(list(input_blocks), recomb_rate, 
                                           n_generations, recomb_tolerance)
        print(f"\n--- Starting Hierarchical Step ---")
        print(f"Input: {total_blocks} blocks -> Target: ~{num_batches} Super-Blocks")
        print(f"Max gap: {preview_max_gap} (n_gen={n_generations}, tol={recomb_tolerance}, rate={recomb_rate})")
        print(f"Processing with {num_processes} workers...")
    else:
        print(f"\n--- Starting Hierarchical Step ---")
        print(f"Input: {total_blocks} blocks -> Target: ~{num_batches} Super-Blocks")
        print(f"Max gap: unlimited (n_generations not specified)")
        print(f"Processing with {num_processes} workers...")
    
    # Prepare worker arguments
    worker_args = []
    
    for b_idx in range(num_batches):
        start_i = b_idx * batch_size
        end_i = min(start_i + batch_size, total_blocks)
        
        original_blocks_list = list(input_blocks[start_i:end_i])
        
        worker_args.append((
            b_idx, start_i, end_i, original_blocks_list, global_probs, global_sites,
            use_hmm_linking, recomb_rate, beam_width, max_founders,
            max_sites_for_linking, n_generations, recomb_tolerance,
            top_n_swap, max_cr_iterations, paint_penalty, min_hotspot_samples,
            cc_scale, verbose
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