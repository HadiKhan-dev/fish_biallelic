"""
Chimera Resolution Module

Sub-block forward selection, top-N swap refinement, BIC pruning,
and chimera resolution via hotspot-guided splicing.

Main entry point: select_and_resolve()
"""

import numpy as np
import math
import ctypes
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

import block_haplotypes

# glibc malloc_trim — releases freed pages back to OS
try:
    _libc = ctypes.CDLL("libc.so.6")
    def _malloc_trim():
        _libc.malloc_trim(0)
except OSError:
    def _malloc_trim():
        pass


def _resolve_threads(num_threads):
    """
    Resolve num_threads: if callable, call it to get the current value.
    
    This enables dynamic thread reallocation — hierarchical_assembly passes
    a function (e.g. _get_dynamic_threads) instead of a fixed integer.
    Each time _resolve_threads is called, the function re-checks how many
    peer workers are active and returns the appropriate thread count.
    
    Also updates numba's active thread count to match, so prange loops
    in scoring functions use the correct number of threads.
    """
    if callable(num_threads):
        n = num_threads()
        try:
            import numba
            numba.set_num_threads(n)
        except Exception:
            pass
        return n
    return num_threads

# =============================================================================
# NUMBA JIT FUNCTIONS
# =============================================================================

try:
    from numba import njit, prange
    _HAS_NUMBA = True

    @njit(parallel=True, fastmath=True)
    def _batched_viterbi_score(stacked_tensor, penalty):
        """Score multiple candidate sets via Viterbi.
        Parallelizes over batch x samples (e.g. 4 x 320 = 1280 parallel units),
        giving full utilization of 112 threads even with small batch sizes.
        Thread count controlled by numba_thread_scope.
        """
        n_batch, n_samples, n_pairs, n_bins = stacked_tensor.shape
        
        # Phase 1: compute per-(batch, sample) scores in parallel
        sample_scores = np.empty((n_batch, n_samples), dtype=np.float64)
        n_total = n_batch * n_samples
        
        for idx in prange(n_total):
            b = idx // n_samples
            s = idx % n_samples
            
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
            sample_scores[b, s] = final_max
        
        # Phase 2: sum across samples for each batch
        scores = np.empty(n_batch, dtype=np.float64)
        for b in range(n_batch):
            total = 0.0
            for s in range(n_samples):
                total += sample_scores[b, s]
            scores[b] = total
        return scores

    @njit(parallel=True, fastmath=True)
    def _viterbi_traceback(tensor, penalty):
        """Viterbi traceback for sample painting.
        Parallelizes over samples — thread count controlled by numba_thread_scope.
        """
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

    @njit(parallel=True, fastmath=True)
    def _compute_bin_emissions_numba(block_samples, hap0, hap1, n_haps, n_bins, snps_per_bin, n_sites):
        """
        Compute binned diploid emission log-likelihoods for a single block.
        Parallelizes over samples — thread count controlled by numba_thread_scope.
        
        10x faster than numpy version, uses no large temporaries (no 2.5 GB
        broadcasting arrays), and validated to match numpy to machine epsilon.
        """
        num_samples = block_samples.shape[0]
        bin_emissions = np.zeros((num_samples, n_haps, n_haps, n_bins), dtype=np.float64)
        
        for s in prange(num_samples):
            s0 = block_samples[s, :, 0]
            s1 = block_samples[s, :, 1]
            s2 = block_samples[s, :, 2]
            
            for h1_idx in range(n_haps):
                h1_0 = hap0[h1_idx]
                h1_1 = hap1[h1_idx]
                
                for h2_idx in range(n_haps):
                    h2_0 = hap0[h2_idx]
                    h2_1 = hap1[h2_idx]
                    
                    for site in range(n_sites):
                        c00 = h1_0[site] * h2_0[site]
                        c01 = (h1_0[site] * h2_1[site]) + (h1_1[site] * h2_0[site])
                        c11 = h1_1[site] * h2_1[site]
                        
                        model = s0[site] * c00 + s1[site] * c01 + s2[site] * c11
                        
                        final = model * 0.99 + 0.01 / 3.0
                        if final < 1e-300:
                            final = 1e-300
                        
                        ll = math.log(final)
                        if ll < -2.0:
                            ll = -2.0
                        
                        b = site // snps_per_bin
                        bin_emissions[s, h1_idx, h2_idx, b] += ll
        
        return bin_emissions

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
    # Warmup emission kernel
    tiny_probs = np.random.rand(2, 10, 3)
    tiny_h0 = np.random.rand(2, 10)
    tiny_h1 = 1.0 - tiny_h0
    _compute_bin_emissions_numba(tiny_probs, tiny_h0, tiny_h1, 2, 2, 5, 10)


# =============================================================================
# PARAMETER COMPUTATION
# =============================================================================

def compute_penalty(batch_blocks):
    """Compute switching penalty based on block sizes.
    
    L1 (200 SNP input, 2k output):   pen=10
    L2 (2k SNP input, 20k output):   pen~63
    L3 (20k SNP input, 200k output): pen~200
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


def compute_cc(batch_blocks, num_samples, cc_scale=0.5):
    """Compute complexity cost per founder using actual block sizes.
    
    CC = cc_scale * (avg_snps / 200) * num_samples * num_blocks
    """
    avg_snps = np.mean([len(b.positions) for b in batch_blocks])
    snp_growth_factor = avg_snps / 200.0
    return cc_scale * snp_growth_factor * num_samples * len(batch_blocks)


# =============================================================================
# EMISSION COMPUTATION
# =============================================================================

def compute_subblock_emissions(input_blocks, global_probs, global_sites, snps_per_bin,
                                num_threads=1):
    """Compute binned diploid emission log-likelihoods for each block.
    
    Uses numba kernel with prange over samples when available — 10x faster
    than numpy, uses no large temporary arrays, and parallelism is controlled
    by numba_thread_scope (set in the caller).
    
    num_threads parameter is accepted for API compatibility but unused —
    parallelism comes from numba_thread_scope instead.
    
    Returns list of dicts with keys: 'hap_keys', 'bin_emissions', 'n_bins'.
    bin_emissions shape: (num_samples, n_haps, n_haps, n_bins)
    """
    all_emissions = []
    for block in input_blocks:
        positions = block.positions
        n_sites = len(positions)
        n_bins = math.ceil(n_sites / snps_per_bin)
        indices = np.searchsorted(global_sites, positions)
        block_samples = np.ascontiguousarray(global_probs[:, indices, :])
        hap_keys = sorted(block.haplotypes.keys())
        n_haps = len(hap_keys)
        haps_tensor = np.array([block.haplotypes[k] for k in hap_keys])
        hap0 = np.ascontiguousarray(haps_tensor[:, :, 0])
        hap1 = np.ascontiguousarray(haps_tensor[:, :, 1])
        
        if _HAS_NUMBA:
            bin_emissions = _compute_bin_emissions_numba(
                block_samples, hap0, hap1, n_haps, n_bins, snps_per_bin, n_sites
            )
        else:
            # Numpy fallback
            h0, h1 = hap0, hap1
            c00 = h0[:, None, :] * h0[None, :, :]
            c01 = (h0[:, None, :] * h1[None, :, :]) + (h1[:, None, :] * h0[None, :, :])
            c11 = h1[:, None, :] * h1[None, :, :]
            s0 = block_samples[:, :, 0]
            s1_arr = block_samples[:, :, 1]
            s2 = block_samples[:, :, 2]
            model_prob = (s0[:, None, None, :] * c00[None, :, :, :] +
                          s1_arr[:, None, None, :] * c01[None, :, :, :] +
                          s2[:, None, None, :] * c11[None, :, :, :])
            del c00, c01, c11, s0, s1_arr, s2
            final_prob = np.maximum(model_prob * 0.99 + 0.01 / 3.0, 1e-300)
            del model_prob
            ll_per_site = np.maximum(np.log(final_prob), -2.0)
            del final_prob
            bin_emissions = np.zeros((block_samples.shape[0], n_haps, n_haps, n_bins), dtype=np.float64)
            for sb in range(n_bins):
                start_s = sb * snps_per_bin
                end_s = min(start_s + snps_per_bin, n_sites)
                bin_emissions[:, :, :, sb] = np.sum(ll_per_site[:, :, :, start_s:end_s], axis=3)
            del ll_per_site
        
        del block_samples, haps_tensor, hap0, hap1
        all_emissions.append({
            'hap_keys': hap_keys,
            'bin_emissions': bin_emissions,
            'n_bins': n_bins
        })
    _malloc_trim()
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
                             chunk_size=64, num_threads=8):
    """Score multiple path sets, grouped by size for batched Viterbi."""
    groups = defaultdict(list)
    for i, ps in enumerate(path_sets):
        groups[len(ps)].append((i, ps))
    results = [None] * len(path_sets)
    for K, group_items in groups.items():
        n_pairs = K * K
        total_b = sum(e['n_bins'] for e in sub_emissions)
        adaptive_cs = max(4, min(64, int(5e8 / (num_samples * n_pairs * total_b * 8))))
        cs = min(adaptive_cs, chunk_size)
        for chunk_start in range(0, len(group_items), cs):
            chunk = group_items[chunk_start:chunk_start + cs]
            def _build_one(item):
                _, ps = item
                return _build_tensor_from_paths(ps, sub_emissions, num_samples)
            nt = _resolve_threads(num_threads)
            if nt <= 1 or len(chunk) <= 1:
                chunk_tensors = [_build_one(item) for item in chunk]
            else:
                with ThreadPoolExecutor(max_workers=nt) as executor:
                    chunk_tensors = list(executor.map(_build_one, chunk))
            # Pre-allocate and fill to avoid double memory from np.stack
            stacked = np.empty((len(chunk), num_samples, n_pairs, total_b),
                               dtype=np.float64)
            for j, t in enumerate(chunk_tensors):
                stacked[j] = t
            del chunk_tensors
            chunk_scores = _batched_viterbi_score(stacked, float(penalty)); del stacked
            _malloc_trim()
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
                       cc_scale=0.5,
                       chunk_size=64,
                       penalty_override=None,
                       spb_override=None,
                       cc_override=None,
                       max_bins_for_cr=2000,
                       num_threads=8):
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
        max_bins_for_cr: Maximum total bins for CR tensors. If the default spb
            would produce more bins than this, spb is increased to cap total_bins.
            Prevents memory/time blowup on large batches (e.g. chr3 L4 with
            8 blocks × 200k sites = 16000 bins → 2 GB tensors per candidate).
        
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
    
    # --- Cap total bins to prevent memory blowup ---
    total_sites = sum(len(b.positions) for b in batch_blocks)
    estimated_bins = math.ceil(total_sites / spb)
    if estimated_bins > max_bins_for_cr:
        spb = math.ceil(total_sites / max_bins_for_cr)
    
    # --- Sub-block emissions ---
    sub_em = compute_subblock_emissions(batch_blocks, global_probs, global_sites, spb)
    total_bins = sum(e['n_bins'] for e in sub_em)
    
    # --- Map matrix: beam index -> dense hap index per block ---
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
        nt = _resolve_threads(num_threads)
        if nt <= 1 or len(subset_list) <= 1:
            return [build_tensor_sel(s) for s in subset_list]
        with ThreadPoolExecutor(max_workers=nt) as executor:
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
            int(5e8 / (num_samples * n_pairs * total_bins * 8))))
        all_scores = {}
        
        # Build template with base pairs
        template = np.zeros((num_samples, n_pairs, total_bins), dtype=np.float64)
        per_block_sel_haps = []
        if selected:
            bin_off = 0
            for b_i, em_data in enumerate(sub_em):
                bin_em = em_data['bin_emissions']
                nb = em_data['n_bins']
                sel_haps = map_matrix[np.array(selected), b_i]
                per_block_sel_haps.append(sel_haps)
                for ii in range(K_next - 1):
                    for jj in range(K_next - 1):
                        pos = ii * K_next + jj
                        template[:, pos, bin_off:bin_off + nb] = bin_em[:, sel_haps[ii], sel_haps[jj], :]
                bin_off += nb
        else:
            for b_i in range(len(sub_em)):
                per_block_sel_haps.append(np.array([], dtype=int))
        
        # Pre-allocate stacked buffer ONCE for this k-iteration.
        stacked = np.empty((max_chunk, num_samples, n_pairs, total_bins),
                           dtype=np.float64)
        
        for cs in range(0, len(remaining), max_chunk):
            chunk = remaining[cs:cs + max_chunk]
            n_chunk = len(chunk)
            stacked_view = stacked[:n_chunk]
            chunk_arr = np.array(chunk)
            
            # Per-candidate threaded fill: copy template + overwrite pairs.
            # Each candidate writes to stacked_view[local_idx] — independent
            # memory. Multiple cores drive memory bandwidth simultaneously.
            def _fill_candidate(local_idx):
                ci = chunk_arr[local_idx]
                stacked_view[local_idx] = template  # contiguous memcpy
                bin_off_t = 0
                for b_i, em_data in enumerate(sub_em):
                    bin_em = em_data['bin_emissions']
                    nb = em_data['n_bins']
                    sel_haps = per_block_sel_haps[b_i]
                    h_c = map_matrix[ci, b_i]
                    for ii in range(K_next - 1):
                        pos = ii * K_next + (K_next - 1)
                        stacked_view[local_idx, :, pos, bin_off_t:bin_off_t + nb] = \
                            bin_em[:, sel_haps[ii], h_c, :]
                    for jj in range(K_next - 1):
                        pos = (K_next - 1) * K_next + jj
                        stacked_view[local_idx, :, pos, bin_off_t:bin_off_t + nb] = \
                            bin_em[:, h_c, sel_haps[jj], :]
                    pos = K_next * K_next - 1
                    stacked_view[local_idx, :, pos, bin_off_t:bin_off_t + nb] = \
                        bin_em[:, h_c, h_c, :]
                    bin_off_t += nb
            
            nt = _resolve_threads(num_threads)
            if nt <= 1 or n_chunk <= 1:
                for i in range(n_chunk):
                    _fill_candidate(i)
            else:
                with ThreadPoolExecutor(max_workers=min(nt, n_chunk)) as executor:
                    list(executor.map(_fill_candidate, range(n_chunk)))
            
            scores = _batched_viterbi_score(
                np.ascontiguousarray(stacked_view), float(pen_sel))
            for j, ci in enumerate(chunk):
                all_scores[ci] = float(scores[j])
        del stacked
        del template
        _malloc_trim()
        best_idx = max(all_scores, key=all_scores.get)
        new_bic = ((len(selected) + 1) * batch_cc) - (2 * all_scores[best_idx])
        if new_bic < current_best_bic:
            current_best_bic = new_bic
            selected.append(best_idx)
        else:
            break
    
    # =========================================================================
    # STEP 2: Swap Refinement
    #
    # Outer loop:
    #   Phase A — 1-for-1 swaps (top-N, batched) until no improvement
    #   Phase B — one round of 2-for-1 swaps (top-N, BIC-aware, batched)
    #             If improved → back to Phase A
    #   Phase C — one round of brute-force 1-for-1 (all candidates, batched)
    #             If improved → back to Phase A
    #   All three found nothing → done
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
    
    def cheap_score_all(base_maxes, temp_set, candidates):
        """Score ALL candidates at once using per-block hap grouping."""
        candidates_arr = np.array(candidates, dtype=int)
        n_cands_local = len(candidates_arr)
        if n_cands_local == 0:
            return {}
        scores = np.zeros(n_cands_local, dtype=np.float64)
        for b_i, em_data in enumerate(sub_em):
            bin_em = em_data['bin_emissions']
            n_haps_local = bin_em.shape[1]
            temp_haps = map_matrix[temp_set, b_i]
            bm = base_maxes[b_i]
            hap_contribs = np.empty(n_haps_local, dtype=np.float64)
            for h in range(n_haps_local):
                cwt = bin_em[:, h, temp_haps, :]
                twc = bin_em[:, temp_haps, h, :]
                self_pair = bin_em[:, h, h, :]
                new_max = np.maximum(
                    np.maximum(np.max(cwt, axis=1), np.max(twc, axis=1)),
                    self_pair)
                combined = np.maximum(bm, new_max)
                hap_contribs[h] = np.sum(combined)
            cand_haps = map_matrix[candidates_arr, b_i]
            scores += hap_contribs[cand_haps]
        return {cand: scores[i] for i, cand in enumerate(candidates)}
    
    # -----------------------------------------------------------------
    # Batched 1-for-1 swap round
    # -----------------------------------------------------------------
    def _run_1for1_round(selected, get_candidates_fn, cur_score):
        """One round of batched 1-for-1 swaps across all positions.
        
        Precomputes templates for all K swap positions, builds a flat task
        list from get_candidates_fn per position, then fills and scores
        in chunks (ThreadPoolExecutor fill + batched numba score).
        
        Returns:
            (remove_idx, add_idx, score_gain) or None
        """
        K = len(selected)
        if K < 2:
            return None
        n_pairs = K * K
        unselected = [x for x in range(n_cands) if x not in selected]
        if not unselected:
            return None
        
        # Precompute templates for all K positions
        pos_templates = []
        pos_haps = []
        K_base = K - 1
        for i in range(K):
            temp_set = selected[:i] + selected[i + 1:]
            tmpl = np.zeros((num_samples, n_pairs, total_bins), dtype=np.float64)
            hpb = {}
            bin_off = 0
            for b_i, em_data in enumerate(sub_em):
                bin_em = em_data['bin_emissions']
                nb = em_data['n_bins']
                t_haps = map_matrix[np.array(temp_set), b_i]
                hpb[b_i] = t_haps
                for ii in range(K_base):
                    for jj in range(K_base):
                        pos = ii * K + jj
                        tmpl[:, pos, bin_off:bin_off + nb] = \
                            bin_em[:, t_haps[ii], t_haps[jj], :]
                bin_off += nb
            pos_templates.append(tmpl)
            pos_haps.append(hpb)
        
        # Get candidates per position, build flat task list
        all_tasks = []
        for i in range(K):
            temp_set = selected[:i] + selected[i + 1:]
            cands = get_candidates_fn(temp_set, unselected)
            for ci in cands:
                all_tasks.append((i, ci))
        
        if not all_tasks:
            del pos_templates
            return None
        
        # Process in chunks
        sc = max(4, min(64, int(5e8 / (num_samples * n_pairs * total_bins * 8))))
        stacked = np.empty((sc, num_samples, n_pairs, total_bins), dtype=np.float64)
        best_swap = None
        best_gain = 0.0
        cand_pos = K - 1
        
        for cs in range(0, len(all_tasks), sc):
            chunk_tasks = all_tasks[cs:cs + sc]
            n_chunk = len(chunk_tasks)
            sv = stacked[:n_chunk]
            
            def _fill(local_idx):
                pi, ci = chunk_tasks[local_idx]
                sv[local_idx] = pos_templates[pi]
                bin_off_t = 0
                for b_i, em_data in enumerate(sub_em):
                    bin_em = em_data['bin_emissions']
                    nb = em_data['n_bins']
                    t_haps = pos_haps[pi][b_i]
                    h_c = map_matrix[ci, b_i]
                    for ii in range(cand_pos):
                        p = ii * K + cand_pos
                        sv[local_idx, :, p, bin_off_t:bin_off_t + nb] = \
                            bin_em[:, t_haps[ii], h_c, :]
                    for jj in range(cand_pos):
                        p = cand_pos * K + jj
                        sv[local_idx, :, p, bin_off_t:bin_off_t + nb] = \
                            bin_em[:, h_c, t_haps[jj], :]
                    p = cand_pos * K + cand_pos
                    sv[local_idx, :, p, bin_off_t:bin_off_t + nb] = \
                        bin_em[:, h_c, h_c, :]
                    bin_off_t += nb
            
            nt = _resolve_threads(num_threads)
            if nt <= 1 or n_chunk <= 1:
                for idx in range(n_chunk):
                    _fill(idx)
            else:
                with ThreadPoolExecutor(
                        max_workers=min(nt, n_chunk)) as exc:
                    list(exc.map(_fill, range(n_chunk)))
            
            scores = _batched_viterbi_score(
                np.ascontiguousarray(sv), float(pen_sel))
            for j, (pi, ci) in enumerate(chunk_tasks):
                gain = float(scores[j]) - cur_score
                if gain > 1e-4 and gain > best_gain:
                    best_gain = gain
                    best_swap = (selected[pi], ci)
        
        del stacked, pos_templates
        _malloc_trim()
        return (best_swap[0], best_swap[1], best_gain) if best_swap else None
    
    # -----------------------------------------------------------------
    # Batched 2-for-1 swap round
    # -----------------------------------------------------------------
    def _run_2for1_round(selected, cur_score):
        """One round of batched 2-for-1 swaps (BIC-aware).
        
        For each pair (i,j) in selected, removes both and tries adding
        one candidate. Uses top-N cheap scoring per pair. Compares BIC
        since the result has K-1 members vs current K.
        
        Returns:
            (remove1, remove2, add_idx) or None
        """
        K = len(selected)
        if K < 3:
            return None
        
        current_bic = K * batch_cc - 2 * cur_score
        K_result = K - 1
        n_pairs_r = K_result * K_result
        unselected = [x for x in range(n_cands) if x not in selected]
        if not unselected:
            return None
        
        # Precompute templates for all C(K,2) pairs
        pair_info = []  # list of (i, j, template, haps_per_block)
        K_base = K - 2
        for i in range(K):
            for j in range(i + 1, K):
                temp_set = [selected[k] for k in range(K) if k != i and k != j]
                tmpl = np.zeros((num_samples, n_pairs_r, total_bins),
                                dtype=np.float64)
                hpb = {}
                bin_off = 0
                for b_i, em_data in enumerate(sub_em):
                    bin_em = em_data['bin_emissions']
                    nb = em_data['n_bins']
                    t_haps = map_matrix[np.array(temp_set), b_i]
                    hpb[b_i] = t_haps
                    for ii in range(K_base):
                        for jj in range(K_base):
                            pos = ii * K_result + jj
                            tmpl[:, pos, bin_off:bin_off + nb] = \
                                bin_em[:, t_haps[ii], t_haps[jj], :]
                    bin_off += nb
                pair_info.append((i, j, tmpl, hpb))
        
        # Get candidates per pair, build flat task list
        all_tasks = []  # (pair_idx, candidate_beam_idx)
        for p_idx, (i, j, tmpl, hpb) in enumerate(pair_info):
            temp_set = [selected[k] for k in range(K) if k != i and k != j]
            bm = precompute_base_max(temp_set)
            cs = cheap_score_all(bm, temp_set, unselected)
            ranked = sorted(cs, key=cs.get, reverse=True)[:top_n_swap]
            for ci in ranked:
                all_tasks.append((p_idx, ci))
        
        if not all_tasks:
            for _, _, tmpl, _ in pair_info:
                del tmpl
            return None
        
        # Process in chunks
        sc = max(4, min(64, int(5e8 / (num_samples * n_pairs_r * total_bins * 8))))
        stacked = np.empty((sc, num_samples, n_pairs_r, total_bins),
                           dtype=np.float64)
        best_2for1 = None
        best_bic = current_bic
        cand_pos = K_result - 1
        
        for cs_start in range(0, len(all_tasks), sc):
            chunk_tasks = all_tasks[cs_start:cs_start + sc]
            n_chunk = len(chunk_tasks)
            sv = stacked[:n_chunk]
            
            def _fill_2(local_idx):
                p_idx, ci = chunk_tasks[local_idx]
                _, _, tmpl, hpb = pair_info[p_idx]
                sv[local_idx] = tmpl
                bin_off_t = 0
                for b_i, em_data in enumerate(sub_em):
                    bin_em = em_data['bin_emissions']
                    nb = em_data['n_bins']
                    t_haps = hpb[b_i]
                    h_c = map_matrix[ci, b_i]
                    for ii in range(cand_pos):
                        p = ii * K_result + cand_pos
                        sv[local_idx, :, p, bin_off_t:bin_off_t + nb] = \
                            bin_em[:, t_haps[ii], h_c, :]
                    for jj in range(cand_pos):
                        p = cand_pos * K_result + jj
                        sv[local_idx, :, p, bin_off_t:bin_off_t + nb] = \
                            bin_em[:, h_c, t_haps[jj], :]
                    p = cand_pos * K_result + cand_pos
                    sv[local_idx, :, p, bin_off_t:bin_off_t + nb] = \
                        bin_em[:, h_c, h_c, :]
                    bin_off_t += nb
            
            nt = _resolve_threads(num_threads)
            if nt <= 1 or n_chunk <= 1:
                for idx in range(n_chunk):
                    _fill_2(idx)
            else:
                with ThreadPoolExecutor(
                        max_workers=min(nt, n_chunk)) as exc:
                    list(exc.map(_fill_2, range(n_chunk)))
            
            scores = _batched_viterbi_score(
                np.ascontiguousarray(sv), float(pen_sel))
            for j_idx, (p_idx, ci) in enumerate(chunk_tasks):
                new_score = float(scores[j_idx])
                new_bic = K_result * batch_cc - 2 * new_score
                if new_bic < best_bic - 1e-4:
                    best_bic = new_bic
                    i_sel, j_sel = pair_info[p_idx][0], pair_info[p_idx][1]
                    best_2for1 = (selected[i_sel], selected[j_sel], ci)
        
        del stacked
        for _, _, tmpl, _ in pair_info:
            del tmpl
        del pair_info
        _malloc_trim()
        return best_2for1
    
    # -----------------------------------------------------------------
    # Candidate selection strategies
    # -----------------------------------------------------------------
    def _top_n_candidates(temp_set, unselected):
        bm = precompute_base_max(temp_set)
        cs = cheap_score_all(bm, temp_set, unselected)
        return sorted(cs, key=cs.get, reverse=True)[:top_n_swap]
    
    def _all_candidates(temp_set, unselected):
        return list(unselected)
    
    # -----------------------------------------------------------------
    # Main swap loop
    # -----------------------------------------------------------------
    current_score = score_subset(selected)
    
    if len(selected) >= 2:
      while True:  # Outer loop: Phase A → Phase B → Phase C → repeat
        
        # Phase A: top-N 1-for-1 until convergence
        while True:
            result = _run_1for1_round(selected, _top_n_candidates,
                                      current_score)
            if result:
                rm, add, gain = result
                selected[selected.index(rm)] = add
                current_score += gain
            else:
                break
        
        # Phase B: one round of top-N 2-for-1 (BIC-aware)
        current_score = score_subset(selected)  # resync after Phase A
        result_b = _run_2for1_round(selected, current_score)
        if result_b:
            rm1, rm2, add = result_b
            selected = [x for x in selected if x != rm1 and x != rm2] + [add]
            current_score = score_subset(selected)
            continue  # Back to Phase A
        
        # Phase C: one round of brute-force 1-for-1
        result_c = _run_1for1_round(selected, _all_candidates,
                                     current_score)
        if result_c:
            rm, add, gain = result_c
            selected[selected.index(rm)] = add
            current_score += gain
            continue  # Back to Phase A
        
        break  # All three phases found nothing → done
    
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
                    ('si->A', [p for idx, p in enumerate(current_paths)
                              if idx != si] + [rA]),
                    ('si->B', [p for idx, p in enumerate(current_paths)
                              if idx != si] + [rB]),
                    ('sj->A', [p for idx, p in enumerate(current_paths)
                              if idx != sj] + [rA]),
                    ('sj->B', [p for idx, p in enumerate(current_paths)
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
            all_path_sets, sub_em, pen_sel, num_samples, chunk_size, num_threads)
        del all_path_sets
        _malloc_trim()
        
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