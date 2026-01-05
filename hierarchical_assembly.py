import numpy as np
import math
from multiprocess import Pool

# Import your pipeline modules
import analysis_utils
import block_haplotypes
import hmm_matching
import beam_search_linking

class HierarchicalAssembler:
    """
    Manages the recursive stitching of genomic blocks into larger contigs.
    """
    def __init__(self, samples_matrix, sample_sites, initial_blocks, 
                 recomb_rate=5e-7, num_processes=16):
        """
        Args:
            samples_matrix: (Samples x Sites x 3) Genotype probabilities.
            sample_sites: Array of genomic positions corresponding to the matrix.
            initial_blocks: List of BlockResult objects (Level 0).
            recomb_rate: Recombination rate per base pair (for Viterbi).
        """
        self.samples_matrix = samples_matrix
        self.sample_sites = sample_sites
        self.current_blocks = initial_blocks
        self.recomb_rate = recomb_rate
        self.num_processes = num_processes

    def run_assembly_step(self, batch_size=10, 
                          num_founders_to_keep=12, 
                          em_iterations=20, 
                          beam_width=200):
        """
        Performs one level of assembly (e.g., stitching groups of 10 blocks).
        
        Args:
            batch_size: Number of blocks to group together (e.g., 10).
            num_founders_to_keep: Number of diverse paths to retain in the new Super-Block.
            em_iterations: Max iterations for Viterbi-EM training.
            beam_width: Number of candidates in Beam Search.
            
        Returns:
            list: A list of new, larger BlockResult objects.
        """
        new_super_blocks = []
        num_batches = math.ceil(len(self.current_blocks) / batch_size)
        
        print(f"\n=== Starting Assembly Step ===")
        print(f"Input Blocks: {len(self.current_blocks)} | Target Batch Size: {batch_size}")
        
        for batch_idx in range(num_batches):
            # 1. Slice Batch
            start_i = batch_idx * batch_size
            end_i = min((batch_idx + 1) * batch_size, len(self.current_blocks))
            
            # Skip if single block left (cannot stitch) - just pass it through
            if end_i - start_i < 2:
                new_super_blocks.extend(self.current_blocks[start_i:end_i])
                continue

            batch_blocks = self.current_blocks[start_i:end_i]
            print(f"Processing Batch {batch_idx+1}/{num_batches} (Blocks {start_i}-{end_i})...")
            
            # 2. Generate Dense Transition Mesh (All-to-All within batch)
            # specific_gaps=None defaults to range(1, len(batch)), i.e., all gaps.
            mesh = viterbi_matching.viterbi_generate_transition_probability_mesh(
                self.samples_matrix,
                self.sample_sites,
                batch_blocks,
                max_num_iterations=em_iterations,
                learning_rate=1.0,
                recomb_rate=self.recomb_rate,
                num_processes=self.num_processes,
                specific_gaps=None # Creates dense mesh (1, 2, ... 9)
            )
            
            breakpoint()
            
            # 3. Beam Search Stitching
            # Find the best N diverse paths through this batch
            beam_results, fast_mesh = beam_search_linking.convert_mesh_to_haplotype_diverse(
                batch_blocks, 
                mesh, 
                num_candidates=beam_width,
                diversity_diff_percent=0.01,
                min_diff_blocks=1
            )
            
            # 4. Select Founders & Flatten
            # reconstructed_data is list of (positions, hap_array, path_indices)
            reconstructed_data = beam_search_linking.select_diverse_founders(
                beam_results, fast_mesh, batch_blocks,
                num_founders=num_founders_to_keep,
                min_total_diff_percent=0.01,
                min_contiguous_diff=1
            )
            
            # 5. Create Super-Block Object
            if not reconstructed_data:
                print(f"Warning: Batch {batch_idx} failed to reconstruct. Passing original blocks.")
                new_super_blocks.extend(batch_blocks)
                continue

            # Convert list of arrays back into a dictionary {0: hap0, 1: hap1...}
            super_hap_dict = {}
            for idx, (_, hap, _) in enumerate(reconstructed_data):
                super_hap_dict[idx] = hap
                
            # Positions are consistent across all founders in the stitch
            super_positions = reconstructed_data[0][0]
            
            # Create new BlockResult
            # We assume all sites in the stitched block are valid (mask=1)
            # because they come from previously filtered blocks.
            super_block = block_haplotypes.BlockResult(
                super_positions,
                super_hap_dict,
                keep_flags=np.ones(len(super_positions), dtype=int)
            )
            
            new_super_blocks.append(super_block)
            
        self.current_blocks = new_super_blocks
        print(f"=== Step Complete. New Block Count: {len(self.current_blocks)} ===\n")
        return self.current_blocks