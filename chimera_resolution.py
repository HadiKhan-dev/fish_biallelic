"""
Chimera Resolution Module

Sub-block forward selection, top-N swap refinement, BIC pruning,
and chimera resolution via hotspot-guided splicing.

Main entry point: select_and_resolve()
"""

import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

import block_haplotypes

# =============================================================================
# NUMBA JIT FUNCTIONS
# =============================================================================

try:
    from numba import njit, prange
    _HAS_NUMBA = True

    @njit(parallel=True, fastmath=True)
    def _batched_viterbi_score(stacked_tensor, penalty):
        """Score multiple candidate sets in parallel via Viterbi."""
        n_batch, n_samples, n_pairs, n_bins = stacked_tensor.shape
        scores = np.empty(n_batch, dtype=np.float64)
        for b in prange(n_batch):
            total = 0.0
            for s in range(n_samples):
                current_scores = np.empty(n_pairs, dtype=np.float64)
                for k in range(n_pairs):
                    current_scores[k] = stacked_tensor[b, s, k, 0]
                for t in range(1, n_bins):
                    best_prev = -np.inf
                    for k in range(n_pairs):
                        if current_scores[k] > best_prev:
                            best_prev = current_scores[k]
                    switch_base = best_prev - penalty
                    for k in range(n_pairs):
                        emission = stacked_tensor[b, s, k, t]
                        stay = current_scores[k]
                        if stay > switch_base:
                            current_scores[k] = stay + emission
                        else:
                            current_scores[k] = switch_base + emission
                final_max = -np.inf
                for k in range(n_pairs):
                    if current_scores[k] > final_max:
                        final_max = current_scores[k]
                total += final_max
            scores[b] = total
        return scores

    @njit(parallel=True, fastmath=True)
    def _viterbi_traceback(tensor, penalty):
        """Viterbi traceback for sample painting."""
        n_samples, n_pairs, n_bins = tensor.shape
        sample_paths = np.zeros((n_samples, n_bins), dtype=np.int32)
        for s in prange(n_samples):
            current_scores = np.empty(n_pairs, dtype=np.float64)
            for p in range(n_pairs):
                current_scores[p] = tensor[s, p, 0]
            backptrs = np.zeros((n_bins, n_pairs), dtype=np.int32)
            for t in range(1, n_bins):
                best_prev = -np.inf; best_prev_idx = 0
                for p in range(n_pairs):
                    if current_scores[p] > best_prev:
                        best_prev = current_scores[p]; best_prev_idx = p
                switch_base = best_prev - penalty
                new_scores = np.empty(n_pairs, dtype=np.float64)
                for p in range(n_pairs):
                    emission = tensor[s, p, t]
                    stay = current_scores[p] + emission
                    switch = switch_base + emission
                    if stay >= switch:
                        new_scores[p] = stay; backptrs[t, p] = p
                    else:
                        new_scores[p] = switch; backptrs[t, p] = best_prev_idx
                for p in range(n_pairs):
                    current_scores[p] = new_scores[p]
            best_final = -np.inf; best_final_idx = 0
            for p in range(n_pairs):
                if current_scores[p] > best_final:
                    best_final = current_scores[p]; best_final_idx = p
            sample_paths[s, n_bins - 1] = best_final_idx
            for t in range(n_bins - 1, 0, -1):
                sample_paths[s, t - 1] = backptrs[t, sample_paths[s, t]]
        return sample_paths

except ImportError:
    _HAS_NUMBA = False


def warmup_jit(num_samples):
    """Call once at startup to compile JIT functions."""
    if not _HAS_NUMBA:
        return
    dummy = np.zeros((1, num_samples, 1, 10), dtype=np.float64)
    _batched_viterbi_score(dummy, 10.0)
    dummy2 = np.zeros((num_samples, 1, 10), dtype=np.float64)
    _viterbi_traceback(dummy2, 10.0)


# =============================================================================
# PARAMETER COMPUTATION
# =============================================================================

def compute_penalty(batch_blocks):
    """Compute switching penalty based on block sizes.
    
    L1 (200 SNP input, 2k output):   pen=10
    L2 (2k SNP input, 20k output):   pen≈63
    L3 (20k SNP input, 200k output): pen≈200
    """
    avg_input_sites = np.mean([len(b.positions) for b in batch_blocks])
    avg_output_sites = avg_input_sites * len(batch_blocks)
    if avg_output_sites <= 5000:
        return 20.0
    return 20.0 * math.sqrt(avg_output_sites / 2000.0)


def compute_spb(batch_blocks):
    """Compute SNPs per bin. Targets ~20 bins per input block, clamped [10, 100]."""
    avg_sites = np.mean([len(b.positions) for b in batch_blocks])
    return int(min(100, max(10, avg_sites // 20)))


def compute_cc(batch_blocks, num_samples, cc_scale=0.2):
    """Compute complexity cost per founder using actual block sizes.
    
    CC = cc_scale * (avg_snps / 200) * num_samples * num_blocks
    """
    avg_snps = np.mean([len(b.positions) for b in batch_blocks])
    snp_growth_factor = avg_snps / 200.0
    return cc_scale * snp_growth_factor * num_samples * len(batch_blocks)


# =============================================================================
# EMISSION COMPUTATION
# =============================================================================

def compute_subblock_emissions(input_blocks, global_probs, global_sites, snps_per_bin):
    """Compute binned diploid emission log-likelihoods for each block.
    
    Returns list of dicts with keys: 'hap_keys', 'bin_emissions', 'n_bins'.
    bin_emissions shape: (num_samples, n_haps, n_haps, n_bins)
    """
    num_samples = global_probs.shape[0]
    all_emissions = []
    for block in input_blocks:
        positions = block.positions
        n_sites = len(positions)
        n_bins = math.ceil(n_sites / snps_per_bin)
        indices = np.searchsorted(global_sites, positions)
        block_samples = global_probs[:, indices, :]
        hap_keys = sorted(block.haplotypes.keys())
        n_haps = len(hap_keys)
        haps_tensor = np.array([block.haplotypes[k] for k in hap_keys])
        h0, h1 = haps_tensor[:, :, 0], haps_tensor[:, :, 1]
        c00 = h0[:, None, :] * h0[None, :, :]
        c01 = (h0[:, None, :] * h1[None, :, :]) + (h1[:, None, :] * h0[None, :, :])
        c11 = h1[:, None, :] * h1[None, :, :]
        s0, s1, s2 = block_samples[:, :, 0], block_samples[:, :, 1], block_samples[:, :, 2]
        model_prob = (s0[:, None, None, :] * c00[None, :, :, :] +
                      s1[:, None, None, :] * c01[None, :, :, :] +
                      s2[:, None, None, :] * c11[None, :, :, :])
        final_prob = np.maximum(model_prob * 0.99 + 0.01 / 3.0, 1e-300)
        ll_per_site = np.maximum(np.log(final_prob), -2.0)
        bin_emissions = np.zeros((num_samples, n_haps, n_haps, n_bins), dtype=np.float64)
        for sb in range(n_bins):
            start = sb * snps_per_bin
            end = min(start + snps_per_bin, n_sites)
            bin_emissions[:, :, :, sb] = np.sum(ll_per_site[:, :, :, start:end], axis=3)
        all_emissions.append({
            'hap_keys': hap_keys,
            'bin_emissions': bin_emissions,
            'n_bins': n_bins
        })
    return all_emissions


# =============================================================================
# TENSOR BUILDING AND SCORING
# =============================================================================

def _build_tensor_from_paths(path_set, sub_emissions, num_samples):
    """Build Viterbi scoring tensor from a set of key-paths."""
    K = len(path_set)
    n_pairs = K * K
    total_bins = sum(e['n_bins'] for e in sub_emissions)
    tensor = np.zeros((num_samples, n_pairs, total_bins), dtype=np.float64)
    bin_offset = 0
    for b_idx, em_data in enumerate(sub_emissions):
        n_bins_b = em_data['n_bins']
        bin_em = em_data['bin_emissions']
        hap_keys = em_data['hap_keys']
        local_indices = np.array([hap_keys.index(path[b_idx]) for path in path_set])
        grid_i = np.repeat(local_indices, K)
        grid_j = np.tile(local_indices, K)
        tensor[:, :, bin_offset:bin_offset + n_bins_b] = bin_em[:, grid_i, grid_j, :]
        bin_offset += n_bins_b
    return np.ascontiguousarray(tensor)


def score_path_set(path_set, sub_emissions, penalty, num_samples):
    """Score a set of key-paths using Viterbi."""
    tensor = _build_tensor_from_paths(path_set, sub_emissions, num_samples)
    return float(np.sum(block_haplotypes.viterbi_score_selection(tensor, float(penalty))))


def score_path_sets_parallel(path_sets, sub_emissions, penalty, num_samples,
                             chunk_size=64):
    """Score multiple path sets, grouped by size for batched Viterbi."""
    groups = defaultdict(list)
    for i, ps in enumerate(path_sets):
        groups[len(ps)].append((i, ps))
    results = [None] * len(path_sets)
    for K, group_items in groups.items():
        n_pairs = K * K
        total_b = sum(e['n_bins'] for e in sub_emissions)
        adaptive_cs = max(4, min(64, int(2e9 / (num_samples * n_pairs * total_b * 8))))
        cs = min(adaptive_cs, chunk_size)
        for chunk_start in range(0, len(group_items), cs):
            chunk = group_items[chunk_start:chunk_start + cs]
            def _build_one(item):
                _, ps = item
                return _build_tensor_from_paths(ps, sub_emissions, num_samples)
            with ThreadPoolExecutor(max_workers=8) as executor:
                chunk_tensors = list(executor.map(_build_one, chunk))
            stacked = np.stack(chunk_tensors, axis=0); del chunk_tensors
            chunk_scores = _batched_viterbi_score(stacked, float(penalty)); del stacked
            for j, (orig_idx, _) in enumerate(chunk):
                results[orig_idx] = float(chunk_scores[j])
    return results


# =============================================================================
# SAMPLE PAINTING AND HOTSPOT DETECTION
# =============================================================================

def paint_samples_viterbi(path_set, sub_emissions, penalty, num_samples):
    """Paint samples using Viterbi traceback."""
    K = len(path_set)
    tensor = _build_tensor_from_paths(path_set, sub_emissions, num_samples)
    return _viterbi_traceback(tensor, float(penalty)), K


def find_hotspots(sample_paths, K, num_blocks, sub_emissions, path_set,
                  num_samples, min_samples=5, ambiguity_threshold=1.0):
    """Find recombination hotspots between path pairs at block boundaries.
    
    Zone-based detection: extends scan range into ambiguous (low-diff) bins,
    then counts samples switching between the pair within the zone.
    """
    hotspots = []
    bin_offsets = [0]
    for e in sub_emissions:
        bin_offsets.append(bin_offsets[-1] + e['n_bins'])

    def _compute_block_diffs(block_idx, si, sj):
        em = sub_emissions[block_idx]
        hap_keys = em['hap_keys']
        si_idx = hap_keys.index(path_set[si][block_idx])
        sj_idx = hap_keys.index(path_set[sj][block_idx])
        bin_em = em['bin_emissions']
        n_haps = len(hap_keys)
        diff = np.zeros((num_samples, em['n_bins']))
        for k in range(n_haps):
            diff += np.abs(bin_em[:, si_idx, k, :] - bin_em[:, sj_idx, k, :])
        diff /= n_haps
        return np.mean(diff, axis=0)

    for b in range(1, num_blocks):
        boundary_bin = bin_offsets[b]
        for si in range(K):
            for sj in range(si + 1, K):
                # Compute left ambiguity zone
                left_zone = 0
                for blk in range(b - 1, -1, -1):
                    mean_diff = _compute_block_diffs(blk, si, sj)
                    extended = False
                    for t in range(len(mean_diff) - 1, -1, -1):
                        if mean_diff[t] < ambiguity_threshold:
                            left_zone += 1; extended = True
                        else:
                            break
                    if not extended or mean_diff[0] >= ambiguity_threshold:
                        break

                # Compute right ambiguity zone
                right_zone = 0
                for blk in range(b, num_blocks):
                    mean_diff = _compute_block_diffs(blk, si, sj)
                    extended = False
                    for t in range(len(mean_diff)):
                        if mean_diff[t] < ambiguity_threshold:
                            right_zone += 1; extended = True
                        else:
                            break
                    if not extended or mean_diff[-1] >= ambiguity_threshold:
                        break

                # Determine scan range (zone_end always includes boundary_bin)
                if left_zone == 0 and right_zone == 0:
                    zone_start = boundary_bin - 1
                    zone_end = boundary_bin
                else:
                    zone_start = boundary_bin - left_zone
                    zone_end = max(boundary_bin, boundary_bin + right_zone - 1)

                # Count samples switching between si and sj in zone
                swap_count = 0
                for s in range(num_samples):
                    for t in range(max(zone_start, 1),
                                   min(zone_end + 1, sample_paths.shape[1])):
                        pb, pa = sample_paths[s, t - 1], sample_paths[s, t]
                        if pb != pa:
                            out = {pb // K, pb % K} - {pa // K, pa % K}
                            inn = {pa // K, pa % K} - {pb // K, pb % K}
                            if (si in out and sj in inn) or (sj in out and si in inn):
                                swap_count += 1
                                break

                if swap_count >= min_samples:
                    hotspots.append({
                        'count': swap_count,
                        'boundary': b,
                        'hap_out': si,
                        'hap_in': sj,
                        'zone': (left_zone, right_zone)
                    })

    hotspots.sort(key=lambda x: -x['count'])
    return hotspots


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def select_and_resolve(beam_results, fast_mesh, batch_blocks,
                       global_probs, global_sites,
                       # Tuning parameters (with sensible defaults)
                       max_founders=12,
                       top_n_swap=20,
                       max_cr_iterations=10,
                       paint_penalty=10.0,
                       min_hotspot_samples=5,
                       cc_scale=0.2,
                       chunk_size=64,
                       penalty_override=None,
                       spb_override=None,
                       cc_override=None):
    """
    Sub-block forward selection + top-N swap + BIC prune + chimera resolution.
    
    Args:
        beam_results: List of (path, score) from beam search.
        fast_mesh: FastMesh object with reverse_mappings.
        batch_blocks: List of original BlockResult objects for this batch.
        global_probs: (num_samples, num_sites, 3) genotype probabilities.
        global_sites: Array of site positions.
        max_founders: Maximum founders to keep.
        top_n_swap: Number of candidates to evaluate per swap position.
        max_cr_iterations: Maximum chimera resolution iterations.
        paint_penalty: Viterbi penalty for sample painting in CR.
        min_hotspot_samples: Minimum samples for a hotspot to be actionable.
        cc_scale: Complexity cost scaling factor.
        chunk_size: Maximum batch size for parallel scoring.
        penalty_override: If set, use this penalty instead of auto-computed.
        spb_override: If set, use this SPB instead of auto-computed.
        cc_override: If set, use this CC instead of auto-computed.
        
    Returns:
        List of resolved beam entries [(dense_path, score), ...] ready for
        reconstruct_haplotypes_from_beam.
    """
    num_samples = global_probs.shape[0]
    n_blocks = len(batch_blocks)
    n_cands = len(beam_results)
    
    # --- Compute parameters ---
    pen_sel = penalty_override if penalty_override is not None else compute_penalty(batch_blocks)
    spb = spb_override if spb_override is not None else compute_spb(batch_blocks)
    batch_cc = cc_override if cc_override is not None else compute_cc(batch_blocks, num_samples, cc_scale)
    
    # --- Sub-block emissions ---
    sub_em = compute_subblock_emissions(batch_blocks, global_probs, global_sites, spb)
    total_bins = sum(e['n_bins'] for e in sub_em)
    
    # --- Map matrix: beam index → dense hap index per block ---
    map_matrix = np.zeros((n_cands, n_blocks), dtype=int)
    for c_idx, (path, _) in enumerate(beam_results):
        for b_idx, dense_idx in enumerate(path):
            map_matrix[c_idx, b_idx] = dense_idx
    
    # --- Local tensor builders ---
    def build_tensor_sel(subset_indices):
        n_sub = len(subset_indices)
        n_pairs = n_sub * n_sub
        tensor = np.zeros((num_samples, n_pairs, total_bins), dtype=np.float64)
        bin_off = 0
        for b_i, em_data in enumerate(sub_em):
            nb = em_data['n_bins']
            bin_em = em_data['bin_emissions']
            local_haps = map_matrix[subset_indices, b_i]
            tensor[:, :, bin_off:bin_off + nb] = \
                bin_em[:, np.repeat(local_haps, n_sub), np.tile(local_haps, n_sub), :]
            bin_off += nb
        return np.ascontiguousarray(tensor)
    
    def build_tensors_threaded(subset_list):
        with ThreadPoolExecutor(max_workers=8) as executor:
            return list(executor.map(build_tensor_sel, subset_list))
    
    def score_subset(subset_indices):
        tensor = build_tensor_sel(subset_indices)
        return float(np.sum(
            block_haplotypes.viterbi_score_selection(tensor, float(pen_sel))))
    
    # =========================================================================
    # STEP 1: Forward Selection
    # =========================================================================
    selected = []
    current_best_bic = float('inf')
    for k in range(20):
        remaining = [x for x in range(n_cands) if x not in selected]
        if not remaining:
            break
        K_next = len(selected) + 1
        n_pairs = K_next * K_next
        max_chunk = max(4, min(64,
            int(2e9 / (num_samples * n_pairs * total_bins * 8))))
        all_scores = {}
        for cs in range(0, len(remaining), max_chunk):
            chunk = remaining[cs:cs + max_chunk]
            tensors = build_tensors_threaded([selected + [ci] for ci in chunk])
            stacked = np.stack(tensors, axis=0); del tensors
            scores = _batched_viterbi_score(stacked, float(pen_sel)); del stacked
            for j, ci in enumerate(chunk):
                all_scores[ci] = float(scores[j])
        best_idx = max(all_scores, key=all_scores.get)
        new_bic = ((len(selected) + 1) * batch_cc) - (2 * all_scores[best_idx])
        if new_bic < current_best_bic:
            current_best_bic = new_bic
            selected.append(best_idx)
        else:
            break
    
    # =========================================================================
    # STEP 2: Top-N Swap Refinement
    # =========================================================================
    def precompute_base_max(base_set):
        K_base = len(base_set)
        base_maxes = []
        for b_i, em_data in enumerate(sub_em):
            bin_em = em_data['bin_emissions']
            local_haps = map_matrix[base_set, b_i]
            pair_em = bin_em[:, np.repeat(local_haps, K_base),
                             np.tile(local_haps, K_base), :]
            base_maxes.append(np.max(pair_em, axis=1))
        return base_maxes
    
    def cheap_score_fn(base_maxes, base_set, candidate):
        full_set = base_set + [candidate]
        K = len(full_set)
        total = 0.0
        for b_i, em_data in enumerate(sub_em):
            bin_em = em_data['bin_emissions']
            cand_hap = map_matrix[candidate, b_i]
            full_haps = map_matrix[full_set, b_i]
            cand_with_all = bin_em[:, cand_hap, full_haps, :]
            all_with_cand = bin_em[:, full_haps, cand_hap, :]
            new_max = np.maximum(np.max(cand_with_all, axis=1),
                                 np.max(all_with_cand, axis=1))
            combined_max = np.maximum(base_maxes[b_i], new_max)
            total += np.sum(combined_max)
        return total
    
    current_score = score_subset(selected)
    K_sel = len(selected)
    
    if K_sel >= 2:
        swap_chunk = max(4, min(64,
            int(2e9 / (num_samples * K_sel * K_sel * total_bins * 8))))
        
        while True:
            unselected = [x for x in range(n_cands) if x not in selected]
            best_swap = None; best_swap_gain = 0.0
            for i, remove_idx in enumerate(selected):
                temp_set = selected[:i] + selected[i + 1:]
                if not temp_set:
                    continue
                base_maxes = precompute_base_max(temp_set)
                cheap_scores = {
                    add: cheap_score_fn(base_maxes, temp_set, add)
                    for add in unselected
                }
                ranked = sorted(cheap_scores, key=cheap_scores.get,
                               reverse=True)[:top_n_swap]
                for cs in range(0, len(ranked), swap_chunk):
                    chunk = ranked[cs:cs + swap_chunk]
                    tensors = build_tensors_threaded(
                        [temp_set + [add] for add in chunk])
                    stacked = np.stack(tensors, axis=0); del tensors
                    scores = _batched_viterbi_score(stacked, float(pen_sel))
                    del stacked
                    for j, add_idx in enumerate(chunk):
                        gain = float(scores[j]) - current_score
                        if gain > 1e-4 and gain > best_swap_gain:
                            best_swap_gain = gain
                            best_swap = (remove_idx, add_idx)
            if best_swap:
                selected[selected.index(best_swap[0])] = best_swap[1]
                current_score += best_swap_gain
            else:
                break
    
    # =========================================================================
    # STEP 3: Force Prune + BIC Prune
    # =========================================================================
    while len(selected) > max_founders:
        cur_ll = score_subset(selected)
        worst = min(selected, key=lambda idx:
            cur_ll - score_subset([x for x in selected if x != idx]))
        selected.remove(worst)
    
    while len(selected) > 1:
        cur_ll = score_subset(selected)
        k_now = len(selected)
        cur_bic = (k_now * batch_cc) - (2 * cur_ll)
        best_rem, best_bic = None, cur_bic
        for idx in selected:
            trial = [x for x in selected if x != idx]
            trial_bic = ((k_now - 1) * batch_cc) - (2 * score_subset(trial))
            if trial_bic < best_bic:
                best_bic = trial_bic; best_rem = idx
        if best_rem is not None:
            selected.remove(best_rem)
        else:
            break
    
    # Convert beam indices to key-paths
    paths = []
    for bi in selected:
        path, _ = beam_results[bi]
        keys = [fast_mesh.reverse_mappings[b][d] for b, d in enumerate(path)]
        paths.append(keys)
    
    # =========================================================================
    # STEP 4: Chimera Resolution
    # =========================================================================
    current_paths = list(paths)
    for iteration in range(max_cr_iterations):
        # 4a. Paint samples
        sp, K_cr = paint_samples_viterbi(
            current_paths, sub_em, paint_penalty, num_samples)
        
        # 4b. Find hotspots
        hotspots = find_hotspots(
            sp, K_cr, n_blocks, sub_em, current_paths,
            num_samples, min_hotspot_samples)
        
        # 4c. Build pair-hotspot map + shared-key pairs
        pair_hotspots = {}
        for hs in hotspots:
            pk = (min(hs['hap_out'], hs['hap_in']),
                  max(hs['hap_out'], hs['hap_in']))
            if pk not in pair_hotspots:
                pair_hotspots[pk] = []
            pair_hotspots[pk].append((hs['boundary'], hs['count']))
        
        K_cur = len(current_paths)
        for si in range(K_cur):
            for sj in range(si + 1, K_cur):
                n_shared = sum(1 for b in range(n_blocks)
                             if current_paths[si][b] == current_paths[sj][b])
                if n_shared >= n_blocks * 0.4 and n_shared < n_blocks:
                    pk = (si, sj)
                    if pk not in pair_hotspots:
                        pair_hotspots[pk] = [
                            (b, 0) for b in range(1, n_blocks)]
        
        if not pair_hotspots:
            break
        
        # 4d. Generate and score splice candidates
        current_ll = score_path_set(
            current_paths, sub_em, pen_sel, num_samples)
        all_path_sets = []
        all_task_info = []
        candidate_groups = []
        
        for (si, sj), boundaries in pair_hotspots.items():
            tried = set()
            for boundary, count in boundaries:
                if boundary in tried:
                    continue
                tried.add(boundary)
                pi, pj = current_paths[si], current_paths[sj]
                rA = pi[:boundary] + pj[boundary:]
                rB = pj[:boundary] + pi[boundary:]
                if rA == pi and rB == pj:
                    continue
                if rA == pj and rB == pi:
                    continue
                ns = [p for idx, p in enumerate(current_paths)
                      if idx != si and idx != sj]
                options = [
                    ('both', ns + [rA, rB]),
                    ('si→A', [p for idx, p in enumerate(current_paths)
                              if idx != si] + [rA]),
                    ('si→B', [p for idx, p in enumerate(current_paths)
                              if idx != si] + [rB]),
                    ('sj→A', [p for idx, p in enumerate(current_paths)
                              if idx != sj] + [rA]),
                    ('sj→B', [p for idx, p in enumerate(current_paths)
                              if idx != sj] + [rB]),
                    ('add_A', current_paths + [rA]),
                    ('add_B', current_paths + [rB]),
                ]
                gs = len(all_path_sets)
                for opt_name, new_paths in options:
                    all_path_sets.append(new_paths)
                    all_task_info.append({
                        'size_delta': len(new_paths) - len(current_paths),
                        'new_paths': new_paths
                    })
                candidate_groups.append(
                    {'option_range': (gs, len(all_path_sets))})
        
        if not all_path_sets:
            break
        
        all_scores = score_path_sets_parallel(
            all_path_sets, sub_em, pen_sel, num_samples, chunk_size)
        
        # 4e. Pick best improving option
        best_option = None; best_gain = 0.0
        for group in candidate_groups:
            gs, ge = group['option_range']
            for i in range(gs, ge):
                info = all_task_info[i]
                adj = ((all_scores[i] - current_ll)
                       - (info['size_delta'] * batch_cc / 2.0))
                if adj > best_gain:
                    best_gain = adj
                    best_option = info['new_paths']
        
        if best_option is None:
            break
        current_paths = best_option
        
        # 4f. BIC prune after each CR iteration
        while len(current_paths) > 1:
            cur_ll = score_path_set(
                current_paths, sub_em, pen_sel, num_samples)
            k_now = len(current_paths)
            cur_bic = (k_now * batch_cc) - (2 * cur_ll)
            best_rem, best_bic = None, cur_bic
            for i in range(len(current_paths)):
                trial = current_paths[:i] + current_paths[i + 1:]
                trial_bic = ((k_now - 1) * batch_cc) - (
                    2 * score_path_set(trial, sub_em, pen_sel, num_samples))
                if trial_bic < best_bic:
                    best_bic = trial_bic; best_rem = i
            if best_rem is not None:
                current_paths = (current_paths[:best_rem]
                                + current_paths[best_rem + 1:])
            else:
                break
    
    # =========================================================================
    # STEP 5: Convert resolved key-paths back to beam format
    # =========================================================================
    resolved_beam = []
    for path_keys in current_paths:
        dense_path = [fast_mesh.reverse_mappings[b].index(key)
                      for b, key in enumerate(path_keys)]
        resolved_beam.append((dense_path, 0.0))
    
    return resolved_beam