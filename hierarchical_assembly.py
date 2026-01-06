import numpy as np
import math
import time
from tqdm import tqdm

# Import your specific modules
# Assuming these are the filenames where the working logic resides
import block_haplotypes
import block_linking_em
import hmm_matching
import beam_search_core

def convert_reconstruction_to_superblock(reconstructed_data, original_blocks):
    """
    Helper function to package the reconstruction results into a BlockResult object.
    
    Args:
        reconstructed_data: List of dicts from beam_search_core.reconstruct_haplotypes_from_beam.
        original_blocks: The list of BlockResult objects that were stitched.
        
    Returns:
        A single BlockResult object representing the consolidated Super-Block.
    """
    if not reconstructed_data:
        return None

    # 1. Consolidate Haplotypes
    # We assign simple integer IDs (0, 1, 2...) based on their selection rank
    super_haplotypes = {}
    for i, data in enumerate(reconstructed_data):
        super_haplotypes[i] = data['haplotype'] 

    # 2. Consolidate Positions
    # All reconstructed paths share the same positions structure
    super_positions = reconstructed_data[0]['positions']

    # 3. Consolidate Flags
    # We concatenate the flags from the original blocks to maintain site metadata
    super_flags = []
    for b in original_blocks:
        if b.keep_flags is not None:
            super_flags.extend(b.keep_flags)
        else:
            super_flags.extend(np.ones(len(b.positions), dtype=int))
    super_flags = np.array(super_flags)

    # 4. Create the Object
    # Note: We do NOT store reads_count_matrix or probs_array here to save memory.
    # They will be re-extracted from the global arrays if this Super-Block 
    # is used in a subsequent hierarchy level.
    super_block = block_haplotypes.BlockResult(
        positions=super_positions,
        haplotypes=super_haplotypes,
        keep_flags=super_flags,
        reads_count_matrix=None,
        probs_array=None 
    )
    
    return super_block

def run_hierarchical_step(input_blocks, global_probs, global_sites,
                          batch_size=10,
                          use_hmm_linking=True,
                          recomb_rate=5e-7,
                          beam_width=200,
                          max_founders=16,
                          penalty_strength=1.0,
                          recomb_penalty=15.0):
    """
    Performs one level of Hierarchical Assembly.
    
    Takes a list of N blocks, groups them into batches of 'batch_size',
    and consolidates each batch into a single 'Super-Block'.
    
    Args:
        input_blocks: List (or BlockResults) of BlockResult objects.
        global_probs: (Samples x Sites x 3) Probability Matrix.
        global_sites: (Sites,) Position Array.
        batch_size: Number of blocks to group together (e.g. 10).
        use_hmm_linking: If True, uses Deep HMM. If False, uses Standard/Naive.
        recomb_rate: Per-site recombination rate for the HMM Mesh.
        beam_width: Candidates to keep during Beam Search.
        max_founders: Maximum number of haplotypes to keep in the Super-Block.
        penalty_strength: BIC penalty multiplier for selection (Higher = fewer haps).
        recomb_penalty: Fixed log-penalty for switching founders during Selection.
        
    Returns:
        BlockResults: A new container with approx N/batch_size Super-Blocks.
    """
    
    total_blocks = len(input_blocks)
    num_batches = math.ceil(total_blocks / batch_size)
    
    output_super_blocks = []
    
    print(f"\n--- Starting Hierarchical Step ---")
    print(f"Input: {total_blocks} blocks -> Target: ~{num_batches} Super-Blocks")
    print(f"Batch Size: {batch_size} | Beam Width: {beam_width} | HMM: {use_hmm_linking}")
    
    # Iterate through batches
    for b_idx in tqdm(range(num_batches), desc="Processing Batches"):
        start_i = b_idx * batch_size
        end_i = min(start_i + batch_size, total_blocks)
        
        # 1. Slice the Batch
        # We wrap it in BlockResults to ensure it behaves like a container
        portion = block_haplotypes.BlockResults(input_blocks[start_i:end_i])
        
        # Handle Edge Case: Single Block (Tail)
        if len(portion) < 2:
            # Cannot link a single block. 
            # We just pass it through as a Super-Block (identity transformation)
            # or we could try to append it to the previous super-block, but passing through is safer.
            print(f"\n[Batch {b_idx}] Single block tail. Passing through.")
            output_super_blocks.append(portion[0])
            continue

        # 2. Calculate Emissions (Required for both Linking and Selection)
        # We calculate P(Data | Local_Hap) for every block in the batch
        portion_emissions = block_linking_em.generate_all_block_likelihoods(
            global_probs, global_sites, portion, num_processes=16
        )
        
        # 3. Generate Transition Mesh
        if use_hmm_linking:
            # Deep HMM (Viterbi EM) - High Precision
            mesh = hmm_matching.generate_transition_probability_mesh_double_hmm(
                global_probs, global_sites, portion,
                recomb_rate=recomb_rate,
                use_standard_baum_welch=False # Use Viterbi EM for sharpness
            )
        else:
            # Standard Linker - Faster, Robust for very small blocks
            mesh = block_linking_em.generate_transition_probability_mesh(
                global_probs, global_sites, portion,
                use_standard_baum_welch=False
            )
            
        # 4. Beam Search (Pathfinding)
        beam_results = beam_search_core.run_full_mesh_beam_search(
            portion, mesh, beam_width=beam_width
        )
        
        if not beam_results:
            print(f"\n[Batch {b_idx}] Beam Search failed (no valid paths). Skipping.")
            continue
            
        # 5. Founder Selection (Model Selection)
        # Initialize FastMesh helper for the selection logic
        fast_mesh = beam_search_core.FastMesh(portion, mesh)
        
        selected_founders = beam_search_core.select_founders_likelihood(
            beam_results, 
            portion_emissions, 
            fast_mesh,
            max_founders=max_founders,
            recomb_penalty=recomb_penalty,
            penalty_strength=penalty_strength,
            do_refinement=True
        )
        
        # 6. Reconstruction
        reconstructed_data = beam_search_core.reconstruct_haplotypes_from_beam(
            selected_founders, fast_mesh, portion
        )
        
        # 7. Package into Super-Block
        super_block = convert_reconstruction_to_superblock(reconstructed_data, portion)
        
        if super_block:
            output_super_blocks.append(super_block)
            
    print(f"Hierarchical Step Complete. Produced {len(output_super_blocks)} Super-Blocks.")
    
    return block_haplotypes.BlockResults(output_super_blocks)