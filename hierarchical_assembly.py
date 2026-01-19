import numpy as np
import math
import time
from tqdm import tqdm

# Import your specific modules
import block_haplotypes
import block_linking_em
import hmm_matching
import beam_search_core

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

def convert_reconstruction_to_superblock(reconstructed_data, original_blocks):
    """
    Packages reconstruction results into a BlockResult (Super-Block).
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

    super_block = block_haplotypes.BlockResult(
        positions=super_positions,
        haplotypes=super_haplotypes,
        keep_flags=super_flags,
        reads_count_matrix=None,
        probs_array=None 
    )
    
    return super_block

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
                          switch_cost_scale=0.075,      # Scaling factor for High Penalty
                          pruning_switch_cost_scale=None, # Scaling factor for Low Penalty
                          use_standard_bic=False,
                          # Memory Safety
                          max_sites_for_linking=2000): 
    """
    Performs one level of Hierarchical Assembly.
    Includes Proxy Logic and Pre-Calculated Emissions to prevent OOM.
    
    Args:
        switch_cost_scale: Factor to calculate High Penalty from block size (e.g. 0.075).
        pruning_switch_cost_scale: Factor to calculate Low Penalty from block size (e.g. 0.001).
                                   If None, falls back to raw pruning_recomb_penalty or defaults to High.
    """
    
    total_blocks = len(input_blocks)
    num_batches = math.ceil(total_blocks / batch_size)
    
    output_super_blocks = []
    
    print(f"\n--- Starting Hierarchical Step ---")
    print(f"Input: {total_blocks} blocks -> Target: ~{num_batches} Super-Blocks")
    
    for b_idx in tqdm(range(num_batches), desc="Processing Batches"):
        start_i = b_idx * batch_size
        end_i = min(start_i + batch_size, total_blocks)
        
        original_portion = block_haplotypes.BlockResults(input_blocks[start_i:end_i])
        
        if len(original_portion) < 2:
            print(f"\n[Batch {b_idx}] Single block tail. Passing through.")
            output_super_blocks.append(original_portion[0])
            continue

        # 1. Create Proxies (Downsample first so stats match the data used for likelihoods)
        proxy_list = []
        for b in original_portion:
            proxy_list.append(create_downsampled_proxy(b, max_sites_for_linking))
        portion_proxy = block_haplotypes.BlockResults(proxy_list)

        # --- DYNAMIC PENALTY LOGIC (Calculated on PROXY size) ---
        avg_snps = np.mean([len(b.positions) for b in portion_proxy])
        
        # 1. Determine Add/Swap Penalty (High)
        if switch_cost_scale is not None:
            current_recomb_penalty = avg_snps * switch_cost_scale
        else:
            current_recomb_penalty = recomb_penalty if recomb_penalty is not None else 15.0

        # 2. Determine Pruning Penalty (Low)
        if pruning_switch_cost_scale is not None:
            # Scaled Low Penalty
            current_pruning_penalty = avg_snps * pruning_switch_cost_scale
        elif pruning_recomb_penalty is not None:
            # Fixed Low Penalty (User provided raw number)
            current_pruning_penalty = pruning_recomb_penalty
        else:
            # Fallback: Same as High (Legacy behavior)
            current_pruning_penalty = current_recomb_penalty
            
        # 3. AUTO-SCALE COMPLEXITY PENALTY
        # Normalize against a baseline of 200 SNPs (Level 1 size).
        # Since we are using Proxy size now, this will usually be near 1.0 or capped at 10.0 (2000/200).
        # This prevents the penalty from exploding when the "real" block size is 20k.
        snp_growth_factor = avg_snps / 200.0
        effective_complexity_scale = complexity_penalty_scale * snp_growth_factor

        # 2. Generate Mesh
        if use_hmm_linking:
            # OOM FIX: Pre-calculate Viterbi Emissions in the main process
            # block_linking now handles sparse proxies natively
            viterbi_emissions = hmm_matching.generate_viterbi_block_emissions(
                global_probs, global_sites, portion_proxy, num_processes=16
            )
            print("Mesh Stage")
            # Generate Mesh using pre-calculated emissions
            mesh = hmm_matching.generate_transition_probability_mesh_double_hmm(
                None, None, portion_proxy, 
                recomb_rate=recomb_rate, 
                use_standard_baum_welch=False,
                precalculated_viterbi_emissions=viterbi_emissions
            )
        else:
            # Standard Linker
            mesh = block_linking_em.generate_transition_probability_mesh(
                global_probs, global_sites, portion_proxy,
                use_standard_baum_welch=True
            )
            
        # 3. Beam Search on Proxy
        beam_results = beam_search_core.run_full_mesh_beam_search(
            portion_proxy, mesh, beam_width=beam_width
        )
        
        if not beam_results:
            print(f"\n[Batch {b_idx}] Beam Search failed. Skipping.")
            continue
            
        # 4. Selection on Proxy
        # Calculate Standard Emissions (Block-level) for Selection
        portion_emissions_standard = block_linking_em.generate_all_block_likelihoods(
            global_probs, global_sites, portion_proxy, num_processes=16
        )
        
        fast_mesh = beam_search_core.FastMesh(portion_proxy, mesh)
        
        # Pass both penalties to the selection core
        selected_founders = beam_search_core.select_founders_likelihood(
            beam_results, 
            portion_emissions_standard, 
            fast_mesh,
            max_founders=max_founders,
            recomb_penalty=current_recomb_penalty,       # HIGH: Strict Entry
            pruning_recomb_penalty=current_pruning_penalty, # LOW: Easy Exit (Chimera Cleanup)
            
            # Use the AUTO-SCALED complexity penalty
            complexity_penalty_scale=effective_complexity_scale,
            
            do_refinement=True,
            use_standard_bic=use_standard_bic
        )
        
        # 5. Reconstruction on Originals
        # Map indices (calculated on Proxy) back to Full Resolution data
        reconstructed_data = beam_search_core.reconstruct_haplotypes_from_beam(
            selected_founders, fast_mesh, original_portion
        )
        
        # 6. Package
        super_block = convert_reconstruction_to_superblock(reconstructed_data, original_portion)
        
        if super_block:
            output_super_blocks.append(super_block)
            
    print(f"Hierarchical Step Complete. Produced {len(output_super_blocks)} Super-Blocks.")
    
    return block_haplotypes.BlockResults(output_super_blocks)