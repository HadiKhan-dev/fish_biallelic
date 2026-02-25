"""
haplotype_refinement.py

Iterative refinement of raw block haplotypes via Viterbi painting
and deconvolution. Supports L1 and L2 refinement levels.

Pipeline:
  HDBSCAN raw blocks
    → L1 assembly → Viterbi paint → deconvolve → L1 refined raw blocks
    → dedup → L1 v2 → L2 → Viterbi paint → deconvolve → L2 refined raw blocks
"""

import numpy as np
import math
import time
import block_haplotypes


# =============================================================================
# VITERBI PAINTING
# =============================================================================

def paint_viterbi(super_blocks, raw_blocks, raw_per_super,
                  global_probs, global_sites, num_samples,
                  penalty_scale=20.0, recomb_rate=5e-8, n_generations=3):
    """
    Paint samples with Viterbi switching at raw block boundaries.
    
    For each super-block, compute per-raw-block emissions for all diplotype
    pairs, then run Viterbi per sample allowing pair-switches with a
    recombination-based penalty.
    
    Args:
        super_blocks: BlockResults from L1 or L2 assembly.
        raw_blocks: Original 200-SNP raw blocks.
        raw_per_super: Number of raw blocks per super-block
            (BATCH_SIZE for L1, BATCH_SIZE^2 for L2).
        global_probs: (n_samples, n_sites, 3) genotype probability array.
        global_sites: Array of all genomic site positions.
        num_samples: Number of samples.
        penalty_scale: Multiplier on the base switching penalty.
            Higher = fewer switches. Default 20.0.
        recomb_rate: Per-bp recombination rate.
        n_generations: Average meioses between founders and samples.
    
    Returns:
        List of dicts, one per super-block:
            {'raw_start': int, 'raw_end': int,
             'paintings': list[list[tuple(int,int)]]}
        paintings[sample][raw_block_local] = (hap_a, hap_b)
    """
    n_raw = len(raw_blocks)
    all_paintings = []
    
    for sb_idx, sb in enumerate(super_blocks):
        hap_keys = sorted(sb.haplotypes.keys())
        K = len(hap_keys)
        sb_positions = sb.positions
        n_sb_sites = len(sb_positions)
        
        # Concretify haplotypes
        haps = np.zeros((K, n_sb_sites), dtype=np.int8)
        for ki, k in enumerate(hap_keys):
            h = sb.haplotypes[k]
            haps[ki] = np.argmax(h, axis=1) if h.ndim > 1 else h
        
        # Sample probabilities at super-block sites
        indices = np.searchsorted(global_sites, sb_positions)
        sample_probs = global_probs[:, indices, :]
        
        # Build pair list (upper triangle including diagonal)
        pair_indices = []
        for a in range(K):
            for b in range(a, K):
                pair_indices.append((a, b))
        n_pairs = len(pair_indices)
        
        # Pair genotypes
        pair_genos = np.zeros((n_pairs, n_sb_sites), dtype=np.int8)
        for pi, (a, b) in enumerate(pair_indices):
            pair_genos[pi] = haps[a] + haps[b]
        
        # Raw block boundaries within this super-block
        raw_start_idx = sb_idx * raw_per_super
        raw_end_idx = min(raw_start_idx + raw_per_super, n_raw)
        n_raw_local = raw_end_idx - raw_start_idx
        
        raw_block_boundaries = []
        site_offset = 0
        for ri in range(raw_start_idx, raw_end_idx):
            n_sites_ri = len(raw_blocks[ri].positions)
            raw_block_boundaries.append((site_offset, site_offset + n_sites_ri))
            site_offset += n_sites_ri
        
        # Per-raw-block emissions: (n_samples, n_pairs, n_raw_local)
        block_emissions = np.zeros((num_samples, n_pairs, n_raw_local),
                                   dtype=np.float64)
        for ri_local, (s_start, s_end) in enumerate(raw_block_boundaries):
            for pi in range(n_pairs):
                geno = pair_genos[pi, s_start:s_end]
                site_idx = np.arange(s_start, s_end)
                probs = sample_probs[:, site_idx, geno]
                block_emissions[:, pi, ri_local] = np.sum(
                    np.log(np.maximum(probs, 1e-300)), axis=1)
        
        # Switching penalties based on physical distance
        switch_penalties = np.zeros(n_raw_local - 1)
        for ri_local in range(n_raw_local - 1):
            ri_abs = raw_start_idx + ri_local
            if ri_abs + 1 < n_raw:
                pos_a = raw_blocks[ri_abs].positions
                pos_b = raw_blocks[ri_abs + 1].positions
                dist = pos_b[len(pos_b) // 2] - pos_a[len(pos_a) // 2]
                expected_recombs = dist * recomb_rate * n_generations
                if expected_recombs > 0:
                    base_pen = min(20.0, -np.log(max(expected_recombs, 1e-10)))
                else:
                    base_pen = 20.0
                switch_penalties[ri_local] = base_pen * penalty_scale
        
        # Viterbi per sample
        sample_paintings = []
        for s in range(num_samples):
            scores = np.full((n_pairs, n_raw_local), -np.inf)
            backptr = np.zeros((n_pairs, n_raw_local), dtype=np.int32)
            
            scores[:, 0] = block_emissions[s, :, 0]
            
            for t in range(1, n_raw_local):
                pen_t = switch_penalties[t - 1]
                for p in range(n_pairs):
                    stay = scores[p, t - 1] + block_emissions[s, p, t]
                    best_switch = (np.max(scores[:, t - 1]) - pen_t
                                   + block_emissions[s, p, t])
                    if stay >= best_switch:
                        scores[p, t] = stay
                        backptr[p, t] = p
                    else:
                        scores[p, t] = best_switch
                        backptr[p, t] = np.argmax(scores[:, t - 1])
            
            # Traceback
            path = np.zeros(n_raw_local, dtype=np.int32)
            path[-1] = np.argmax(scores[:, -1])
            for t in range(n_raw_local - 1, 0, -1):
                path[t - 1] = backptr[path[t], t]
            
            sample_path = [pair_indices[path[t]] for t in range(n_raw_local)]
            sample_paintings.append(sample_path)
        
        all_paintings.append({
            'raw_start': raw_start_idx,
            'raw_end': raw_end_idx,
            'paintings': sample_paintings
        })
    
    return all_paintings


# =============================================================================
# DECONVOLUTION
# =============================================================================

def deconvolve(super_blocks, viterbi_paintings, raw_blocks,
               global_probs, global_sites, num_samples):
    """
    Deconvolve raw blocks using per-raw-block Viterbi painting.
    
    For each haplotype in the super-block, collect all carrier samples
    (from the painting), then re-estimate the allele at each site using
    the carrier's read data and the partner haplotype as a template.
    
    Args:
        super_blocks: BlockResults from L1 or L2 assembly.
        viterbi_paintings: Output of paint_viterbi.
        raw_blocks: Original 200-SNP raw blocks.
        global_probs: (n_samples, n_sites, 3) genotype probability array.
        global_sites: Array of all genomic site positions.
        num_samples: Number of samples.
    
    Returns:
        BlockResults with refined raw blocks (same count as raw_blocks).
    """
    n_raw = len(raw_blocks)
    refined_blocks = [None] * n_raw
    
    for paint_data in viterbi_paintings:
        raw_start = paint_data['raw_start']
        raw_end = paint_data['raw_end']
        sample_paintings = paint_data['paintings']
        
        # Find corresponding super-block by position match
        sb_idx = None
        for li, sb in enumerate(super_blocks):
            if len(sb.positions) > 0:
                if sb.positions[0] == raw_blocks[raw_start].positions[0]:
                    sb_idx = li
                    break
        if sb_idx is None:
            for li, pd in enumerate(viterbi_paintings):
                if pd['raw_start'] == raw_start:
                    sb_idx = li
                    break
        
        sb = super_blocks[sb_idx]
        hap_keys = sorted(sb.haplotypes.keys())
        K = len(hap_keys)
        sb_positions = sb.positions
        n_sb_sites = len(sb_positions)
        
        # Concretify super-block haplotypes
        haps_sb = np.zeros((K, n_sb_sites), dtype=np.int8)
        for ki, k in enumerate(hap_keys):
            h = sb.haplotypes[k]
            haps_sb[ki] = np.argmax(h, axis=1) if h.ndim > 1 else h
        
        # Sample probabilities at super-block sites
        indices = np.searchsorted(global_sites, sb_positions)
        sample_probs_sb = global_probs[:, indices, :]
        
        # Process each raw block
        site_offset = 0
        for ri in range(raw_start, raw_end):
            ri_local = ri - raw_start
            raw_block = raw_blocks[ri]
            n_raw_sites = len(raw_block.positions)
            site_slice = slice(site_offset, site_offset + n_raw_sites)
            site_idx_local = np.arange(site_offset, site_offset + n_raw_sites)
            
            # Build per-haplotype carrier list for this raw block
            carriers = {ki: [] for ki in range(K)}
            for s in range(num_samples):
                a, b = sample_paintings[s][ri_local]
                if a == b:
                    carriers[a].append((s, a, True))
                else:
                    carriers[a].append((s, b, False))
                    carriers[b].append((s, a, False))
            
            # Deconvolve each haplotype
            new_haps = np.zeros((K, n_raw_sites, 2), dtype=np.float64)
            
            for ki in range(K):
                if not carriers[ki]:
                    # No carriers: use super-block alleles as fallback
                    alleles = haps_sb[ki, site_slice]
                    new_haps[ki, :, 0] = 1.0 - alleles
                    new_haps[ki, :, 1] = alleles.astype(np.float64)
                    continue
                
                log_p0 = np.zeros(n_raw_sites, dtype=np.float64)
                log_p1 = np.zeros(n_raw_sites, dtype=np.float64)
                
                for s, partner_ki, is_hom in carriers[ki]:
                    if is_hom:
                        log_p0 += np.log(np.maximum(
                            sample_probs_sb[s, site_idx_local, 0], 1e-300))
                        log_p1 += np.log(np.maximum(
                            sample_probs_sb[s, site_idx_local, 2], 1e-300))
                    else:
                        pa = haps_sb[partner_ki, site_slice]
                        log_p0 += np.log(np.maximum(
                            sample_probs_sb[s, site_idx_local, pa], 1e-300))
                        log_p1 += np.log(np.maximum(
                            sample_probs_sb[s, site_idx_local, pa + 1],
                            1e-300))
                
                max_log = np.maximum(log_p0, log_p1)
                p0 = np.exp(log_p0 - max_log)
                p1 = np.exp(log_p1 - max_log)
                total = p0 + p1
                new_haps[ki, :, 0] = p0 / total
                new_haps[ki, :, 1] = p1 / total
            
            refined_haps = {ki: new_haps[ki].copy() for ki in range(K)}
            refined_block = block_haplotypes.BlockResult(
                positions=raw_block.positions.copy(),
                haplotypes=refined_haps,
                keep_flags=raw_block.keep_flags,
                probs_array=raw_block.probs_array
            )
            refined_blocks[ri] = refined_block
            site_offset += n_raw_sites
    
    # Fill any gaps (safety net for rounding in last super-block)
    for ri in range(n_raw):
        if refined_blocks[ri] is None:
            refined_blocks[ri] = raw_blocks[ri]
    
    return block_haplotypes.BlockResults(refined_blocks)


# =============================================================================
# DEDUPLICATION
# =============================================================================

def dedup_blocks(blocks, threshold_pct=1.0, verbose=True):
    """
    Merge near-identical haplotypes within each block.
    
    Prevents degenerate EM meshes in subsequent assembly steps by
    removing haplotypes that differ by less than threshold_pct.
    When merging, keeps the haplotype with higher sharpness
    (mean max probability across sites).
    
    Args:
        blocks: BlockResults to deduplicate.
        threshold_pct: Merge haplotypes differing by less than this %.
        verbose: If True, print summary.
    
    Returns:
        Deduplicated BlockResults.
    """
    deduped = []
    total_before = 0
    total_after = 0
    n_collapsed = 0
    
    for block in blocks:
        hap_keys = sorted(block.haplotypes.keys())
        K = len(hap_keys)
        total_before += K
        
        if K <= 1:
            deduped.append(block)
            total_after += K
            continue
        
        # Concretify and compute sharpness
        concrete = {}
        sharpness = {}
        for k in hap_keys:
            h = block.haplotypes[k]
            if h.ndim > 1:
                concrete[k] = np.argmax(h, axis=1)
                sharpness[k] = np.mean(np.max(h, axis=1))
            else:
                concrete[k] = np.array(h)
                sharpness[k] = 1.0
        
        # Greedy dedup
        kept = [hap_keys[0]]
        for k in hap_keys[1:]:
            is_dup = False
            for kk in kept:
                if np.mean(concrete[k] != concrete[kk]) * 100 < threshold_pct:
                    is_dup = True
                    if sharpness[k] > sharpness[kk]:
                        kept[kept.index(kk)] = k
                    break
            if not is_dup:
                kept.append(k)
        
        if len(kept) < K:
            n_collapsed += 1
        
        new_haps = {new_idx: block.haplotypes[old_key]
                    for new_idx, old_key in enumerate(kept)}
        new_block = block_haplotypes.BlockResult(
            positions=block.positions.copy(),
            haplotypes=new_haps,
            keep_flags=block.keep_flags,
            probs_array=block.probs_array
        )
        deduped.append(new_block)
        total_after += len(kept)
    
    if verbose:
        print(f"  Dedup: {total_before} -> {total_after} total haps "
              f"({n_collapsed} blocks had merges)")
    
    return block_haplotypes.BlockResults(deduped)


# =============================================================================
# HIGH-LEVEL REFINEMENT
# =============================================================================

def refine_at_level(assembly_blocks, raw_blocks,
                    global_probs, global_sites, num_samples,
                    raw_per_super,
                    penalty_scale=20.0,
                    recomb_rate=5e-8, n_generations=3,
                    verbose=True):
    """
    Run one round of refinement: Viterbi paint → deconvolve.
    
    Args:
        assembly_blocks: Super-blocks from L1 or L2 assembly.
        raw_blocks: Original 200-SNP raw blocks.
        global_probs: (n_samples, n_sites, 3) array.
        global_sites: Site position array.
        num_samples: Number of samples.
        raw_per_super: Raw blocks per super-block
            (batch_size for L1, batch_size^2 for L2).
        penalty_scale: Viterbi switching penalty scale.
        recomb_rate: Per-bp recombination rate.
        n_generations: Average meioses.
        verbose: If True, print progress and switching stats.
    
    Returns:
        BlockResults with refined raw blocks.
    """
    t0 = time.time()
    if verbose:
        print(f"  Viterbi painting (penalty_scale={penalty_scale})...")
    
    paintings = paint_viterbi(
        assembly_blocks, raw_blocks, raw_per_super,
        global_probs, global_sites, num_samples,
        penalty_scale=penalty_scale,
        recomb_rate=recomb_rate,
        n_generations=n_generations
    )
    
    if verbose:
        total_switches = 0
        total_positions = 0
        for pd in paintings:
            for s in range(num_samples):
                path = pd['paintings'][s]
                for t in range(1, len(path)):
                    if path[t] != path[t - 1]:
                        total_switches += 1
                total_positions += len(path) - 1
        pct = 100 * total_switches / max(total_positions, 1)
        print(f"  Switches: {total_switches}/{total_positions} ({pct:.2f}%)")
        print(f"  Painting done in {time.time() - t0:.1f}s")
    
    t0 = time.time()
    if verbose:
        print(f"  Deconvolving...")
    
    refined = deconvolve(
        assembly_blocks, paintings, raw_blocks,
        global_probs, global_sites, num_samples
    )
    
    if verbose:
        print(f"  Deconvolution done in {time.time() - t0:.1f}s")
    
    return refined


def run_refinement_pipeline(raw_blocks,
                            global_probs, global_sites, num_samples,
                            run_l1_assembly_fn,
                            run_l2_assembly_fn,
                            batch_size=10,
                            penalty_scale=20.0,
                            recomb_rate=5e-8, n_generations=3,
                            verbose=True):
    """
    Run the full L1 + L2 refinement pipeline.
    
    Steps:
      1. L1 assembly on raw blocks
      2. L1 refinement (Viterbi paint + deconvolve)
      3. Dedup
      4. L1 v2 assembly on refined blocks
      5. L2 assembly
      6. L2 refinement (Viterbi paint + deconvolve)
    
    Args:
        raw_blocks: HDBSCAN-discovered raw blocks.
        global_probs: (n_samples, n_sites, 3) array.
        global_sites: Site position array.
        num_samples: Number of samples.
        run_l1_assembly_fn: Callable that takes BlockResults and returns
            L1 super-blocks. Should handle its own parallelization.
        run_l2_assembly_fn: Callable that takes BlockResults and returns
            L2 super-blocks (with HMM linking).
        batch_size: Number of blocks per batch (default 10).
        penalty_scale: Viterbi switching penalty scale.
        recomb_rate: Per-bp recombination rate.
        n_generations: Average meioses.
        verbose: If True, print progress.
    
    Returns:
        dict with keys:
            'l1_refined': L1-refined raw blocks
            'l2_refined': L2-refined raw blocks
            'l1_assembly': L1 super-blocks (from refined input)
            'l2_assembly': L2 super-blocks (from refined input)
    """
    results = {}
    
    # ── Step 1: L1 Assembly ──
    if verbose:
        print(f"\n  Step 1: L1 assembly on raw blocks...")
    t0 = time.time()
    l1_blocks = run_l1_assembly_fn(raw_blocks)
    if verbose:
        print(f"  L1 assembly done in {time.time() - t0:.0f}s")
    
    # ── Step 2: L1 Refinement ──
    if verbose:
        print(f"\n  Step 2: L1 refinement...")
    l1_refined = refine_at_level(
        l1_blocks, raw_blocks,
        global_probs, global_sites, num_samples,
        raw_per_super=batch_size,
        penalty_scale=penalty_scale,
        recomb_rate=recomb_rate,
        n_generations=n_generations,
        verbose=verbose
    )
    results['l1_refined'] = l1_refined
    
    # ── Step 3: Dedup ──
    if verbose:
        print(f"\n  Step 3: Dedup L1 refined blocks...")
    l1_refined_dd = dedup_blocks(l1_refined, verbose=verbose)
    
    # ── Step 4: L1 v2 Assembly ──
    if verbose:
        print(f"\n  Step 4: L1 v2 assembly on refined blocks...")
    t0 = time.time()
    l1_v2 = run_l1_assembly_fn(l1_refined_dd)
    if verbose:
        print(f"  L1 v2 assembly done in {time.time() - t0:.0f}s")
    results['l1_assembly'] = l1_v2
    
    # ── Step 5: L2 Assembly ──
    if verbose:
        print(f"\n  Step 5: L2 assembly...")
    t0 = time.time()
    l2_blocks = run_l2_assembly_fn(l1_v2)
    if verbose:
        print(f"  L2 assembly done in {time.time() - t0:.0f}s")
    results['l2_assembly'] = l2_blocks
    
    # ── Step 6: L2 Refinement ──
    if verbose:
        print(f"\n  Step 6: L2 refinement...")
    l2_refined = refine_at_level(
        l2_blocks, raw_blocks,
        global_probs, global_sites, num_samples,
        raw_per_super=batch_size * batch_size,
        penalty_scale=penalty_scale,
        recomb_rate=recomb_rate,
        n_generations=n_generations,
        verbose=verbose
    )
    results['l2_refined'] = l2_refined
    
    return results