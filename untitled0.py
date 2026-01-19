import numpy as np

def evaluate_reconstruction_robust(super_blocks, true_haps_list, global_sites, error_threshold=1.0):
    """
    Evaluates reconstruction accuracy against the 'Concrete' reality of the simulation.
    
    Fixes the 'Fuzzy Truth' artifact by thresholding the probabilistic input 
    (Truth > 0.5) before comparison. This matches how the simulation generates 
    discrete individuals from fuzzy priors.
    """
    print(f"\n{'Block':<5} | {'Range':<20} | {'True':<5} | {'Found':<5} | {'Recov':<5} | {'AvgErr%':<8} | {'Status'}")
    print("-" * 95)
    
    total_true = 0
    total_recovered = 0
    
    # 1. Flatten Truth to Matrix (Sites x Haps)
    true_matrix_full = []
    for h in true_haps_list:
        if h.ndim > 1: true_matrix_full.append(h[:, 1])
        else: true_matrix_full.append(h)
    true_matrix_full = np.array(true_matrix_full) # (Num_Founders, Global_Sites)

    for b_idx, block in enumerate(super_blocks):
        # 2. Align Data
        start_pos = block.positions[0]
        end_pos = block.positions[-1]
        
        start_g = np.searchsorted(global_sites, start_pos)
        end_g = np.searchsorted(global_sites, end_pos, side='right')
        
        # Intersect to handle downsampling/proxies safely
        # matching_sites -> sites present in BOTH block and global truth
        common, block_idxs, global_idxs = np.intersect1d(
            block.positions, global_sites[start_g:end_g], return_indices=True
        )
        
        if len(common) == 0:
            print(f"{b_idx:<5} | {start_pos}-{end_pos} | 0     | 0     | 0     | N/A      | EMPTY")
            continue

        # 3. Get Concrete Truth
        # Slice global truth to the specific sites active in this block
        prob_slice = true_matrix_full[:, start_g:end_g][:, global_idxs]
        
        # *** CRITICAL FIX: CONCRETIZATION ***
        # Convert probabilistic input (e.g. 0.6) to concrete reality (1.0)
        concrete_slice = (prob_slice > 0.5).astype(np.float64)
        
        # Find unique rows (Founders active in this region)
        unique_true_rows = np.unique(concrete_slice, axis=0)
        n_true = len(unique_true_rows)
        
        # 4. Get Found Haplotypes
        found_rows = []
        for k in sorted(block.haplotypes.keys()):
            h = block.haplotypes[k]
            if h.ndim > 1: h = h[:, 1]
            
            # Extract only the intersecting sites
            found_rows.append(h[block_idxs])
        
        found_rows = np.array(found_rows)
        n_found = len(found_rows)
        
        if n_found == 0:
            print(f"{b_idx:<5} | {start_pos:<9}-{end_pos:<9} | {n_true:<5} | 0     | 0     | N/A      | MISSING")
            total_true += n_true
            continue

        # 5. Compare (Greedy Matching)
        # We want to know: For every True Hap, is there a Found Hap close enough?
        
        recovered_this_block = 0
        errors = []
        
        for t_idx in range(n_true):
            target = unique_true_rows[t_idx]
            
            # Calculate distance to ALL found haps
            # Dist = % of sites that disagree
            dists = np.mean(np.abs(found_rows - target), axis=1) * 100
            
            best_err = np.min(dists)
            errors.append(best_err)
            
            if best_err < error_threshold:
                recovered_this_block += 1
        
        avg_err = np.mean(errors)
        
        # Status
        if recovered_this_block == n_true:
            if n_found > n_true + 2:
                status = "OK (Noisy)"
            else:
                status = "OK"
        else:
            status = "MISSING"
            
        print(f"{b_idx:<5} | {start_pos:<9}-{end_pos:<9} | {n_true:<5} | {n_found:<5} | {recovered_this_block:<5} | {avg_err:<8.3f} | {status}")
        
        total_true += n_true
        total_recovered += recovered_this_block
        
    print("-" * 95)
    print(f"Total True Haplotypes: {total_true}")
    print(f"Total Recovered:       {total_recovered}")
    if total_true > 0:
        print(f"Global Recovery Rate:  {total_recovered/total_true*100:.2f}%")

# =======================================================
# RUN EVALUATION
# =======================================================

evaluate_reconstruction_robust(
    super_blocks_level_1,  # Your reconstruction variable
    haplotype_data,        # Your probabilistic ground truth list
    haplotype_sites,       # Global site positions
    error_threshold=1.0    # 1% error tolerance
)