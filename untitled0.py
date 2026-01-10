import numpy as np

def evaluate_superblocks_against_truth(super_blocks, true_haps_list, global_sites):
    """
    Evaluates how well the reconstructed super-blocks match the ground truth.
    """
    
    print(f"\n{'Block':<5} | {'Range':<20} | {'True':<5} | {'Found':<5} | {'Recovered':<10} | {'Avg Error %':<12} | {'Status'}")
    print("-" * 90)
    
    total_true_haps = len(true_haps_list)
    total_recovered = 0
    total_checks = 0
    
    # Convert truth to numpy array
    true_matrix = []
    for h in true_haps_list:
        if h.ndim > 1: true_matrix.append(h[:, 1])
        else: true_matrix.append(h)
    true_matrix = np.array(true_matrix)
    
    for b_idx, block in enumerate(super_blocks):
        # 1. Align Coordinates
        start_pos = block.positions[0]
        end_pos = block.positions[-1]
        
        # Binary search for global indices
        start_global_idx = np.searchsorted(global_sites, start_pos)
        end_global_idx = np.searchsorted(global_sites, end_pos, side='right')
        
        # Intersection to be safe
        common_sites, block_indices, global_indices = np.intersect1d(
            block.positions, global_sites[start_global_idx:end_global_idx], return_indices=True
        )
        
        # 2. Slice Ground Truth for this Super-Region
        true_slice = true_matrix[:, start_global_idx:end_global_idx][:, global_indices]
        
        # 3. Get Unique True Haplotypes
        unique_true_rows, true_ids = np.unique(true_slice, axis=0, return_index=True)
        num_true_unique = len(unique_true_rows)
        
        # 4. Get Reconstructed Haplotypes
        rec_matrix = []
        # Sort keys to ensure deterministic order
        for k in sorted(block.haplotypes.keys()):
            h = block.haplotypes[k]
            if h.ndim > 1: h = h[:, 1]
            rec_matrix.append(h[block_indices])
        rec_matrix = np.array(rec_matrix)
        num_found = len(rec_matrix)
        
        # 5. Distance Matrix
        if num_found == 0:
            print(f"{b_idx:<5} | {start_pos}-{end_pos} | {num_true_unique:<5} | 0     | 0          | N/A          | FAIL")
            continue
            
        distances = np.zeros((num_true_unique, num_found))
        for t in range(num_true_unique):
            for r in range(num_found):
                d = np.mean(np.abs(unique_true_rows[t] - rec_matrix[r]))
                distances[t, r] = d * 100 # Percentage
        
        # 6. Scoring
        recovered_count = 0
        errors = []
        
        for t in range(num_true_unique):
            best_match_idx = np.argmin(distances[t])
            best_error = distances[t, best_match_idx]
            errors.append(best_error)
            
            # Threshold: < 1% difference = Recovered
            if best_error < 1.0:
                recovered_count += 1
        
        mean_error = np.mean(errors)
        status = "OK"
        if recovered_count < num_true_unique:
            status = "MISSING"
        elif num_found > num_true_unique + 2: 
            status = "NOISY"
            
        print(f"{b_idx:<5} | {start_pos:<9}-{end_pos:<9} | {num_true_unique:<5} | {num_found:<5} | {recovered_count:<10} | {mean_error:<12.4f} | {status}")
        
        total_recovered += recovered_count
        total_checks += num_true_unique

    print("-" * 90)
    print(f"Total Unique True Haplotypes across all blocks: {total_checks}")
    print(f"Total Successfully Recovered: {total_recovered}")
    if total_checks > 0:
        print(f"Global Recovery Rate: {total_recovered / total_checks * 100:.2f}%")

# =============================================================================
# RUN TEST
# =============================================================================

# Replace 's_level_2' with your actual variable name if different
evaluate_superblocks_against_truth(
    super_blocks_level_2, 
    haplotype_data, 
    haplotype_sites
)