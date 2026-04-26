"""
Chimera Resolution Module

Sub-block forward selection, top-N swap refinement, BIC pruning,
and chimera resolution via hotspot-guided splicing.

Main entry point: select_and_resolve()
"""

import numpy as np
import math
import ctypes
import os as _os_cr
import time as _time_cr
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

# =============================================================================
# DEBUG INSTRUMENTATION (opt-in via HIERARCHICAL_DEBUG_DIR env var)
# =============================================================================
# Same mechanism as the phase markers in hierarchical_assembly.py, but writes
# fine-grained progress inside select_and_resolve() so we can pinpoint which
# sub-step of chimera resolution is where the silent failure / segfault lives.
# Each worker writes to its own file: cr_pid<PID>_phases.log (since select_
# and_resolve doesn't know which batch_idx it's inside — that info is one
# stack frame up).  When the crash happens, the worker's most recent
# cr_pid<PID>_phases.log line tells us exactly where it died.
def _cr_debug_dir():
    return _os_cr.environ.get('HIERARCHICAL_DEBUG_DIR', '')

def _cr_mark(phase, extra=""):
    d = _cr_debug_dir()
    if not d:
        return
    try:
        path = _os_cr.path.join(d, f'cr_pid{_os_cr.getpid()}_phases.log')
        with open(path, 'a', buffering=1) as f:
            f.write(f"{_time_cr.time():.6f} {phase} {extra}\n")
    except Exception:
        pass

# Maximum samples to process at once in emission scoring tensors.
# Bounds peak memory per worker: (n_candidates, _SAMPLE_CHUNK, K², n_bins).
# Each sample is independent, so chunking is mathematically exact.
# 50 samples keeps the largest tensors under ~100 MB regardless of K.
_SAMPLE_CHUNK = 10


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

from numba import njit, prange, get_num_threads as _numba_get_num_threads

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

@njit(parallel=True, fastmath=False)
def _fill_all_blocks_numba(sv, tmpl_slice, bin_ems_stacked,
                            chunk_cands_arr, map_matrix,
                            t_haps_stacked, bin_offs, nbs,
                            _s0, _s1, stride, cand_pos):
    """Fused broadcast + per-block fill kernel. Replaces an earlier
    Python loop that dispatched a separate per-block numba kernel
    once per block (N-blocks dispatches per fill call).

    Per-fill work is now ONE kernel dispatch instead of N-blocks
    dispatches, and the serial
    `sv[:] = tmpl_slice[np.newaxis, :, :, :]` broadcast is now
    inside prange, running in parallel across candidates.

    Observed on L3 chr5 run 20260424_184837:
      CR_1for1_fill wall   = 251.3s  (8064 calls)
      CR_1for1_kernel wall =  17.7s  (kernel time only)
    So ~93% of fill wall was Python overhead (broadcast + 10
    np.asarray/np.ascontiguousarray calls + 10 kernel dispatches
    with their OMP barriers) rather than actual numba work.
    This kernel absorbs all of that into one parallel region.

    Mathematically identical to the reference per-block Python
    loop (validated byte-exact by validate_fused_fill.py):

        sv[:] = tmpl_slice[None, :, :, :]   # broadcast
        for b_i in range(n_blocks):
            bin_off = bin_offs[b_i]; nb = nbs[b_i]
            for local_idx in range(n_chunk):
                ci = chunk_cands[local_idx]
                h_c = map_matrix[ci, b_i]
                for ii in range(cand_pos):
                    p = ii*stride + cand_pos
                    sv[local_idx, :, p, bin_off:bin_off+nb] = \\
                        bin_ems_stacked[
                            _s0:_s1, t_haps_stacked[ii, b_i], h_c,
                            bin_off:bin_off+nb]
                for jj in range(cand_pos):
                    p = cand_pos*stride + jj
                    sv[local_idx, :, p, bin_off:bin_off+nb] = \\
                        bin_ems_stacked[
                            _s0:_s1, h_c, t_haps_stacked[jj, b_i],
                            bin_off:bin_off+nb]
                p = cand_pos*stride + cand_pos
                sv[local_idx, :, p, bin_off:bin_off+nb] = \\
                    bin_ems_stacked[_s0:_s1, h_c, h_c,
                                     bin_off:bin_off+nb]

    but parallelized over local_idx via prange (GIL-free) AND
    with the broadcast fused inside the prange so the template
    fill also runs in parallel (instead of being a serial memcpy).

    REQUIRES: bin_ems_stacked must be built by concatenating
    per-block em['bin_emissions'] along axis=-1. This requires
    all blocks in sub_em to have the same (num_samples, n_haps,
    n_haps) leading-dim shape — i.e. uniform n_haps across
    sub-blocks within the super-block. If that invariant doesn't
    hold, np.concatenate raises ValueError at chimera setup
    (no fallback path).

    Args:
        sv:                (n_chunk, _sn, n_pairs, total_bins) float64,
                           written in-place, fully overwritten.
        tmpl_slice:        (_sn, n_pairs, total_bins) float64, read-only,
                           broadcast into every sv[local_idx].
        bin_ems_stacked:   (num_samples, n_haps, n_haps, total_bins)
                           float64, read-only, concatenated bins.
        chunk_cands_arr:   (n_chunk,) int64, candidate row indices into
                           map_matrix.
        map_matrix:        (n_cands_total, n_blocks) int64, contiguous;
                           maps candidate idx -> per-block hap idx.
        t_haps_stacked:    (cand_pos, n_blocks) int64, stacked from
                           hpb[b_i]. t_haps_stacked[ii, b_i] is the hap
                           id at position ii for block b_i in the
                           current basis.
        bin_offs:          (n_blocks,) int64, cumulative bin offsets in
                           the stacked bin dimension.
        nbs:               (n_blocks,) int64, per-block bin counts.
        _s0, _s1:          int, sample range. _sn = _s1 - _s0.
        stride:            int, pair-index row stride (== K for 1for1,
                           == K_result for 2for1 — both == cand_pos+1).
        cand_pos:          int, row/col of the candidate in the pair
                           index (== K_base in both 1for1 and 2for1).
    """
    n_chunk = chunk_cands_arr.shape[0]
    n_blocks = bin_offs.shape[0]
    _sn = _s1 - _s0
    n_pairs = sv.shape[2]
    total_bins = sv.shape[3]
    for local_idx in prange(n_chunk):
        ci = chunk_cands_arr[local_idx]
        # --- Phase 1: broadcast tmpl_slice into sv[local_idx] ---
        # This is the `sv[:] = tmpl_slice[None, :, :, :]` operation
        # done in parallel across candidates instead of as a serial
        # 30 MB memcpy.  Each thread writes its own sv[local_idx]
        # slab (~0.5 MB at L3 shapes) so different threads exercise
        # different DRAM channels.
        for s in range(_sn):
            for p in range(n_pairs):
                for b in range(total_bins):
                    sv[local_idx, s, p, b] = tmpl_slice[s, p, b]
        # --- Phase 2: overwrite cand row / col / diagonal slots ---
        # For each block, the candidate being tested lives at row
        # cand_pos and column cand_pos of the (K_result, K_result)
        # pair grid.  We overwrite:
        #   (ii, cand_pos) for ii in [0, cand_pos)   -- cand-row slots
        #   (cand_pos, jj) for jj in [0, cand_pos)   -- cand-col slots
        #   (cand_pos, cand_pos)                     -- diagonal
        # using bin_ems_stacked indexed by the candidate's per-block
        # hap id h_c = map_matrix[ci, b_i] and the basis haps
        # t_haps_stacked[ii/jj, b_i] for the non-candidate positions.
        for b_i in range(n_blocks):
            bin_off = bin_offs[b_i]
            nb = nbs[b_i]
            h_c = map_matrix[ci, b_i]
            # cand-row slots: (ii*stride + cand_pos) for ii in [0, cand_pos)
            for ii in range(cand_pos):
                p = ii * stride + cand_pos
                t_h = t_haps_stacked[ii, b_i]
                for s in range(_sn):
                    for bin_local in range(nb):
                        sv[local_idx, s, p, bin_off + bin_local] = \
                            bin_ems_stacked[_s0 + s, t_h, h_c,
                                            bin_off + bin_local]
            # cand-col slots: (cand_pos*stride + jj) for jj in [0, cand_pos)
            for jj in range(cand_pos):
                p = cand_pos * stride + jj
                t_h = t_haps_stacked[jj, b_i]
                for s in range(_sn):
                    for bin_local in range(nb):
                        sv[local_idx, s, p, bin_off + bin_local] = \
                            bin_ems_stacked[_s0 + s, h_c, t_h,
                                            bin_off + bin_local]
            # Diagonal: (cand_pos*stride + cand_pos)
            p = cand_pos * stride + cand_pos
            for s in range(_sn):
                for bin_local in range(nb):
                    sv[local_idx, s, p, bin_off + bin_local] = \
                        bin_ems_stacked[_s0 + s, h_c, h_c,
                                        bin_off + bin_local]


@njit(parallel=True, fastmath=False)
def _precompute_base_max_block_numba(bin_em, local_haps):
    """Numba kernel for precompute_base_max — replaces a per-block
    np.repeat / np.tile fancy-index + np.max chain that allocated a
    (samples, K_base*K_base, bins) intermediate per call.

    Computes, for each (sample s, bin b):
        out[s, b] = max over (ii, jj) in [0, K_base)^2 of
                        bin_em[s, local_haps[ii], local_haps[jj], b]

    Mathematically equivalent to:
        pair_em = bin_em[:, np.repeat(local_haps, K_base),
                         np.tile(local_haps, K_base), :]
        out     = np.max(pair_em, axis=1)

    Same set of K_base^2 values entered into the max for each (s, b);
    max is order-independent for the finite log-likelihood values used
    here (clamped >= -2.0 in _compute_bin_emissions_numba), so bit-exact
    with the numpy reference under fastmath=False. Parallelizes over
    samples via prange (each thread writes its own out[s, :] row, no
    race).

    K_base == 0 edge case: the inner loops are skipped, leaving
    out[s, b] = -np.inf. This matches the semantics of np.max on an
    empty axis (which would raise; in practice K_base >= 1 at every
    call site since _run_1for1_round requires K >= 2 and
    _run_2for1_round requires K >= 3).
    """
    n_samples = bin_em.shape[0]
    n_bins = bin_em.shape[3]
    K_base = local_haps.shape[0]
    out = np.empty((n_samples, n_bins), dtype=np.float64)
    for s in prange(n_samples):
        for b in range(n_bins):
            best = -np.inf
            for ii in range(K_base):
                for jj in range(K_base):
                    v = bin_em[s, local_haps[ii], local_haps[jj], b]
                    if v > best:
                        best = v
            out[s, b] = best
    return out


@njit(parallel=True, fastmath=False)
def _cheap_score_all_block_numba(bin_em, temp_haps, bm, n_haps_local):
    """Numba kernel for cheap_score_all's per-block inner loop —
    replaces a chain of three numpy slice copies + chained
    np.max / np.maximum / np.sum that allocated ~60 MB of intermediates
    per call (cwt = bin_em[:, h, temp_haps, :] etc.).

    Computes, for each candidate-row hap h in [0, n_haps_local):
        hap_contribs[h] = sum over (s, b) of
                            max(bm[s, b],
                                max(max_t bin_em[s, h, temp_haps[t], b],
                                    max_t bin_em[s, temp_haps[t], h, b],
                                    bin_em[s, h, h, b]))

    Mathematically equivalent to the per-hap numpy block:
        cwt       = bin_em[:, h, temp_haps, :]
        twc       = bin_em[:, temp_haps, h, :]
        self_pair = bin_em[:, h, h, :]
        new_max   = np.maximum(np.maximum(np.max(cwt, axis=1),
                                          np.max(twc, axis=1)),
                               self_pair)
        combined  = np.maximum(bm, new_max)
        hap_contribs[h] = np.sum(combined)

    With fastmath=False:
      - Max operations are order-independent for finite values (the
        bin_em values are clamped log-likelihoods >= -2.0).
      - Sum order matches numpy's row-major np.sum on a (samples,
        bins) C-contiguous array: outer loop over s, inner over b
        — bit-exact with the numpy reference.

    Parallelizes over h via prange (each thread writes its own
    hap_contribs[h] slot, no race).

    K_base == 0 edge case: m_cwt and m_twc stay at -np.inf, so
    nm = max(-inf, -inf, sp) = sp; cb = max(bm[s, b], sp). Matches
    the numpy version when temp_haps is empty (cwt/twc would be
    zero-width arrays where np.max raises; in practice K_base >= 1
    at every call site).
    """
    n_samples = bin_em.shape[0]
    n_bins = bin_em.shape[3]
    K_base = temp_haps.shape[0]
    hap_contribs = np.empty(n_haps_local, dtype=np.float64)
    for h in prange(n_haps_local):
        total = 0.0
        for s in range(n_samples):
            for b in range(n_bins):
                # max over temp_haps of bin_em[s, h, temp_haps[k], b]
                m_cwt = -np.inf
                for k in range(K_base):
                    v = bin_em[s, h, temp_haps[k], b]
                    if v > m_cwt:
                        m_cwt = v
                # max over temp_haps of bin_em[s, temp_haps[k], h, b]
                m_twc = -np.inf
                for k in range(K_base):
                    v = bin_em[s, temp_haps[k], h, b]
                    if v > m_twc:
                        m_twc = v
                # self-pair contribution
                sp = bin_em[s, h, h, b]
                # new_max = max(m_cwt, m_twc, sp)
                nm = m_cwt
                if m_twc > nm:
                    nm = m_twc
                if sp > nm:
                    nm = sp
                # combined = max(bm[s, b], new_max)
                cb = bm[s, b]
                if nm > cb:
                    cb = nm
                total += cb
        hap_contribs[h] = total
    return hap_contribs




def warmup_jit(num_samples):
    """Call once at startup to compile JIT functions."""
    dummy = np.zeros((1, num_samples, 1, 10), dtype=np.float64)
    _batched_viterbi_score(dummy, 10.0)
    dummy2 = np.zeros((num_samples, 1, 10), dtype=np.float64)
    _viterbi_traceback(dummy2, 10.0)
    # Warmup emission kernel
    tiny_probs = np.random.rand(2, 10, 3)
    tiny_h0 = np.random.rand(2, 10)
    tiny_h1 = 1.0 - tiny_h0
    _compute_bin_emissions_numba(tiny_probs, tiny_h0, tiny_h1, 2, 2, 5, 10)
    # Warmup cheap-score precompute and per-hap kernels (used by
    # precompute_base_max and cheap_score_all in step2 phase_A/C and
    # _run_2for1_round). Tiny shapes (2, 2, 2, 4) just trigger JIT.
    tiny_bin_em = np.zeros((2, 2, 2, 4), dtype=np.float64)
    tiny_haps = np.array([0, 1], dtype=np.int64)
    _precompute_base_max_block_numba(tiny_bin_em, tiny_haps)
    tiny_bm = np.zeros((2, 4), dtype=np.float64)
    _cheap_score_all_block_numba(tiny_bin_em, tiny_haps, tiny_bm, 2)


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
                                num_threads=None):
    """Compute binned diploid emission log-likelihoods for each block.
    
    Uses numba kernel with prange over samples when available — 10x faster
    than numpy, uses no large temporary arrays, and parallelism is controlled
    by numba_thread_scope (set in the caller).
    
    num_threads parameter is accepted for API compatibility but unused —
    parallelism comes from numba_thread_scope instead.

    NOTE: when num_threads is callable (e.g. _get_dynamic_threads from
    hierarchical_assembly), _resolve_threads is invoked once per block
    iteration so the per-block _compute_bin_emissions_numba kernel picks
    up the latest total_cores // active_workers value as peers finish.
    Default None preserves the original behaviour (no thread resync) for
    any existing callers that don't pass the argument.
    
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

        # Dynamic thread resync (gated on num_threads being provided, so
        # any external caller that doesn't pass it sees the original
        # "no thread management" behaviour).
        if num_threads is not None:
            _resolve_threads(num_threads)
        bin_emissions = _compute_bin_emissions_numba(
            block_samples, hap0, hap1, n_haps, n_bins, snps_per_bin, n_sites
        )
        
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

def _build_tensor_from_paths(path_set, sub_emissions, num_samples, sample_range=None):
    """Build Viterbi scoring tensor from a set of key-paths.
    If sample_range=(s0, s1) is given, only builds for those samples."""
    K = len(path_set)
    n_pairs = K * K
    total_bins = sum(e['n_bins'] for e in sub_emissions)
    if sample_range is not None:
        s0, s1 = sample_range
        n_s = s1 - s0
    else:
        s0, s1 = 0, num_samples
        n_s = num_samples
    tensor = np.zeros((n_s, n_pairs, total_bins), dtype=np.float64)
    bin_offset = 0
    for b_idx, em_data in enumerate(sub_emissions):
        n_bins_b = em_data['n_bins']
        bin_em = em_data['bin_emissions']
        hap_keys = em_data['hap_keys']
        local_indices = np.array([hap_keys.index(path[b_idx]) for path in path_set], dtype=np.intp)
        grid_i = np.repeat(local_indices, K)
        grid_j = np.tile(local_indices, K)
        tensor[:, :, bin_offset:bin_offset + n_bins_b] = bin_em[s0:s1, grid_i, grid_j, :]
        bin_offset += n_bins_b
    return np.ascontiguousarray(tensor)


def score_path_set(path_set, sub_emissions, penalty, num_samples,
                   num_threads=None):
    """Score a set of key-paths using Viterbi.

    If num_threads is provided (int or callable), it is resolved via
    _resolve_threads before the numba dispatch so the kernel picks up the
    latest dynamic thread allocation.  Default None preserves the original
    behaviour (no thread resync) for any existing callers that don't pass
    the argument.
    """
    tensor = _build_tensor_from_paths(path_set, sub_emissions, num_samples)
    if num_threads is not None:
        _resolve_threads(num_threads)
    return float(np.sum(block_haplotypes.viterbi_score_selection(tensor, float(penalty))))


def score_path_sets_parallel(path_sets, sub_emissions, penalty, num_samples,
                             chunk_size=64, num_threads=8):
    """Score multiple path sets, grouped by size for batched Viterbi.
    Processes samples in chunks of _SAMPLE_CHUNK to bound memory."""
    groups = defaultdict(list)
    for i, ps in enumerate(path_sets):
        groups[len(ps)].append((i, ps))
    results = [None] * len(path_sets)
    for K, group_items in groups.items():
        n_pairs = K * K
        total_b = sum(e['n_bins'] for e in sub_emissions)
        _sc = min(num_samples, _SAMPLE_CHUNK)
        adaptive_cs = max(4, min(64, int(5e8 / (_sc * n_pairs * total_b * 8))))
        cs = min(adaptive_cs, chunk_size)
        for chunk_start in range(0, len(group_items), cs):
            chunk = group_items[chunk_start:chunk_start + cs]
            n_chunk = len(chunk)
            
            # Accumulate scores across sample chunks
            accum_scores = np.zeros(n_chunk, dtype=np.float64)
            for _s0 in range(0, num_samples, _SAMPLE_CHUNK):
                _s1 = min(_s0 + _SAMPLE_CHUNK, num_samples)
                
                def _build_one(item, _s0=_s0, _s1=_s1):
                    _, ps = item
                    return _build_tensor_from_paths(ps, sub_emissions, num_samples,
                                                    sample_range=(_s0, _s1))
                nt = _resolve_threads(num_threads)
                if nt <= 1 or n_chunk <= 1:
                    chunk_tensors = [_build_one(item) for item in chunk]
                else:
                    with ThreadPoolExecutor(max_workers=nt) as executor:
                        chunk_tensors = list(executor.map(_build_one, chunk))
                # Pre-allocate and fill to avoid double memory from np.stack
                stacked = np.empty((n_chunk, _s1 - _s0, n_pairs, total_b),
                                   dtype=np.float64)
                for j, t in enumerate(chunk_tensors):
                    stacked[j] = t
                del chunk_tensors
                partial = _batched_viterbi_score(stacked, float(penalty))
                accum_scores += partial
                del stacked
            _malloc_trim()
            for j, (orig_idx, _) in enumerate(chunk):
                results[orig_idx] = float(accum_scores[j])
    return results


# =============================================================================
# SAMPLE PAINTING AND HOTSPOT DETECTION
# =============================================================================

def paint_samples_viterbi(path_set, sub_emissions, penalty, num_samples,
                          num_threads=None):
    """Paint samples using Viterbi traceback.

    If num_threads is provided (int or callable), it is resolved via
    _resolve_threads before the numba dispatch so the kernel picks up the
    latest dynamic thread allocation.  Default None preserves the original
    behaviour (no thread resync) for any existing callers that don't pass
    the argument.
    """
    K = len(path_set)
    # Fix: K=0 produces a (n_samples, 0, total_bins) tensor with n_pairs=0.
    # _viterbi_traceback is a numba JIT function that assumes n_pairs >= 1 —
    # with n_pairs=0 it performs out-of-bounds reads on backptrs/current_scores
    # and segfaults (or returns garbage).  Return an all-zero painting with
    # K=0 here so downstream find_hotspots / pair-loop code harmlessly processes
    # an empty path set.
    if K == 0:
        total_bins = sum(e['n_bins'] for e in sub_emissions)
        sample_paths = np.zeros((num_samples, total_bins), dtype=np.int32)
        return sample_paths, K
    tensor = _build_tensor_from_paths(path_set, sub_emissions, num_samples)
    if num_threads is not None:
        _resolve_threads(num_threads)
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
    
    _cr_mark("CR_entry",
             f"n_samples={num_samples} n_blocks={n_blocks} n_cands={n_cands} "
             f"max_founders={max_founders} cc_scale={cc_scale}")
    
    # --- Compute parameters ---
    _cr_mark("CR_compute_params_start")
    pen_sel = penalty_override if penalty_override is not None else compute_penalty(batch_blocks)
    spb = spb_override if spb_override is not None else compute_spb(batch_blocks)
    batch_cc = cc_override if cc_override is not None else compute_cc(batch_blocks, num_samples, cc_scale)
    _cr_mark("CR_compute_params_done",
             f"pen_sel={pen_sel} spb={spb} batch_cc={batch_cc}")
    
    # --- Cap total bins to prevent memory blowup ---
    total_sites = sum(len(b.positions) for b in batch_blocks)
    estimated_bins = math.ceil(total_sites / spb)
    if estimated_bins > max_bins_for_cr:
        spb = math.ceil(total_sites / max_bins_for_cr)
    
    # --- Sub-block emissions ---
    _cr_mark("CR_sub_em_start", f"spb={spb} total_sites={total_sites}")
    sub_em = compute_subblock_emissions(batch_blocks, global_probs, global_sites, spb,
                                        num_threads=num_threads)
    total_bins = sum(e['n_bins'] for e in sub_em)
    _cr_mark("CR_sub_em_done",
             f"total_bins={total_bins} per_block_bins={[e['n_bins'] for e in sub_em]}")
    
    # --- Map matrix: beam index -> dense hap index per block ---
    _cr_mark("CR_map_matrix_start")
    map_matrix = np.zeros((n_cands, n_blocks), dtype=int)
    for c_idx, (path, _) in enumerate(beam_results):
        for b_idx, dense_idx in enumerate(path):
            map_matrix[c_idx, b_idx] = dense_idx
    _cr_mark("CR_map_matrix_done", f"shape={map_matrix.shape}")

    # --- Fused-fill-kernel precompute ---
    # Build bin_ems_stacked = concatenated per-block bin_emissions along
    # the bins axis. The fused _fill_all_blocks_numba kernel then runs
    # as a single dispatch per fill call (instead of n_blocks dispatches
    # plus a serial pre-broadcast).
    #
    # Concatenation requires uniform leading-dim shape
    # (num_samples, n_haps, n_haps) across blocks. When sub-blocks have
    # different n_haps (e.g. a super-block with one n_haps=4 block and
    # nine n_haps=6 blocks), we pad the smaller blocks up to the
    # super-block's max n_haps with zeros. The kernel only ever indexes
    # bin_ems_stacked[s, t_h, h_c, ...] with t_h, h_c in [0, n_haps_b)
    # for block b — those indices come from map_matrix which is bounded
    # by each block's actual n_haps_b — so the padded slots
    # [n_haps_b : max_n_haps] are never read at runtime. Padding is
    # purely additive memory, no math change. Verified byte-exact by
    # the jagged-padded scenario in validate_fused_fill.py.
    #
    # Also: cache int64-contiguous map_matrix so the fused kernel
    # doesn't pay an astype/ascontiguousarray per fill call.
    _cr_mark("CR_stack_bin_em_start")
    _max_n_haps = max(em['bin_emissions'].shape[1] for em in sub_em)
    _padded_bin_ems = []
    _n_padded = 0
    for em in sub_em:
        bin_em = em['bin_emissions']
        n_haps_b = bin_em.shape[1]
        if n_haps_b == _max_n_haps:
            _padded_bin_ems.append(bin_em)
        else:
            # Pad n_haps dims with zeros — slots [n_haps_b:_max_n_haps]
            # are never indexed at kernel runtime. Inert memory.
            padded = np.zeros(
                (bin_em.shape[0], _max_n_haps, _max_n_haps, bin_em.shape[3]),
                dtype=bin_em.dtype)
            padded[:, :n_haps_b, :n_haps_b, :] = bin_em
            _padded_bin_ems.append(padded)
            _n_padded += 1
    stacked_bin_em = np.concatenate(_padded_bin_ems, axis=-1)
    bin_offs_arr = np.zeros(len(sub_em), dtype=np.int64)
    nbs_arr = np.zeros(len(sub_em), dtype=np.int64)
    _off = 0
    for _i_em, em in enumerate(sub_em):
        bin_offs_arr[_i_em] = _off
        nbs_arr[_i_em] = em['n_bins']
        _off += em['n_bins']
    assert _off == total_bins, (
        f"bin offset mismatch: cumulative={_off} total_bins={total_bins}")
    map_matrix_c = np.ascontiguousarray(map_matrix).astype(np.int64)
    _cr_mark("CR_stack_bin_em_done",
             f"shape={stacked_bin_em.shape} "
             f"max_n_haps={_max_n_haps} "
             f"n_padded={_n_padded}/{len(sub_em)} "
             f"per_block_nbs={list(int(x) for x in nbs_arr)}")
    del _padded_bin_ems
    
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
        # Dynamic thread resync: viterbi_score_selection is a numba @njit
        # kernel with prange over samples.  score_subset is the workhorse
        # of Step 2's `current_score = score_subset(selected)` resyncs and
        # of Step 3's force-prune + BIC-prune loops (O(K^2) calls per
        # pruning pass with K up to max_founders), so resyncing here lets
        # the worker scale up its numba thread count as peers finish.
        _resolve_threads(num_threads)
        return float(np.sum(
            block_haplotypes.viterbi_score_selection(tensor, float(pen_sel))))
    
    # =========================================================================
    # STEP 1: Forward Selection (sample-chunked)
    # =========================================================================
    _cr_mark("CR_step1_start")
    selected = []
    current_best_bic = float('inf')
    for k in range(20):
        remaining = [x for x in range(n_cands) if x not in selected]
        if not remaining:
            break
        K_next = len(selected) + 1
        n_pairs = K_next * K_next
        _cr_mark("CR_step1_iter_start",
                 f"k={k} K_next={K_next} n_pairs={n_pairs} n_remaining={len(remaining)}")
        # Dynamic thread resync: pull the latest total_cores // active_workers
        # value before doing any per-iteration work.  Step 1's fused-fill and
        # batched-Viterbi kernels at lines 1041 / 1054 below use whatever
        # numba.set_num_threads() was last set to; resyncing once per k
        # iteration is enough granularity since each k does many fills/scores
        # under the same thread count, and as peers finish during a long
        # forward-selection pass we want this worker to scale up.
        nt = _resolve_threads(num_threads)
        # Size max_chunk using _SAMPLE_CHUNK to bound stacked tensor
        _sc = min(num_samples, _SAMPLE_CHUNK)
        max_chunk = max(4, min(64,
            int(5e8 / (_sc * n_pairs * total_bins * 8))))
        all_scores = {}
        
        # Build template with base pairs (full samples — moderate size)
        _cr_mark("CR_step1_template_build_start", f"k={k}")
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
        # For the fused fill kernel: stack per_block_sel_haps into a
        # (cand_pos, n_blocks) int64 array once per k-iteration. cand_pos
        # for step1 is K_next - 1 (the candidate lives at row/col index
        # K_next-1 in the pair grid, so there are K_next-1 "other" base
        # positions). When K_next == 1 the basis is empty and
        # per_block_sel_haps contains (0,)-shape arrays, so the stacked
        # array has shape (0, n_blocks). Numba accepts int64[:, :]
        # parameters regardless of runtime shape; the row/col loops
        # (range(cand_pos) = range(0)) are skipped so t_haps_stacked is
        # never indexed at runtime. Verified byte-exact by scenario 9
        # of validate_fused_fill.py (step1_Knext=1_empty_basis).
        t_haps_stacked_step1 = np.stack(
            per_block_sel_haps, axis=1
        ).astype(np.int64)
        _cr_mark("CR_step1_template_build_done", f"k={k}")
        
        _cr_mark("CR_step1_chunks_start", f"k={k} max_chunk={max_chunk}")
        for cs in range(0, len(remaining), max_chunk):
            chunk = remaining[cs:cs + max_chunk]
            n_chunk = len(chunk)
            chunk_arr = np.array(chunk)
            
            # Accumulate scores across sample chunks
            accum_scores = np.zeros(n_chunk, dtype=np.float64)
            for _s0 in range(0, num_samples, _SAMPLE_CHUNK):
                _s1 = min(_s0 + _SAMPLE_CHUNK, num_samples)
                _sn = _s1 - _s0
                
                stacked_chunk = np.empty((n_chunk, _sn, n_pairs, total_bins),
                                         dtype=np.float64)
                _tmpl_slice = template[_s0:_s1]
                
                _cr_mark("CR_step1_fill_start",
                         f"k={k} cs={cs} _s0={_s0} n_chunk={n_chunk} "
                         f"K_next={K_next}")
                # Fused numba fill kernel.  Byte-exact to step2's fused
                # path with the mapping:
                #     stride  ↔  K_next
                #     cand_pos ↔ K_next - 1
                # Row pos   = ii * stride + cand_pos
                #           = ii * K_next + (K_next - 1)        ← step1 row
                # Col pos   = cand_pos * stride + jj
                #           = (K_next - 1) * K_next + jj        ← step1 col
                # Diag pos  = cand_pos * stride + cand_pos
                #           = (K_next - 1) * K_next + (K_next - 1)
                #           = K_next * K_next - 1               ← step1 diag
                # validate_fused_fill.py verifies this identity on
                # synthetic step1-shaped inputs (K_next = 1..6).
                chunk_cands_arr_s1 = np.asarray(chunk_arr, dtype=np.int64)
                stride_s1 = K_next
                cand_pos_s1 = K_next - 1
                _cr_mark("CR_step1_numba_dispatch",
                         f"nt_numba={_numba_get_num_threads()} "
                         f"n_chunk={n_chunk} n_blocks={len(sub_em)} "
                         f"K_next={K_next}")
                _cr_mark("CR_step1_kernel_start",
                         f"k={k} cs={cs} _s0={_s0}")
                _fill_all_blocks_numba(
                    stacked_chunk, _tmpl_slice, stacked_bin_em,
                    chunk_cands_arr_s1, map_matrix_c,
                    t_haps_stacked_step1,
                    bin_offs_arr, nbs_arr,
                    _s0, _s1, stride_s1, cand_pos_s1)
                _cr_mark("CR_step1_kernel_done",
                         f"k={k} cs={cs} _s0={_s0}")
                _cr_mark("CR_step1_fill_done",
                         f"k={k} cs={cs} _s0={_s0}")
                
                _cr_mark("CR_step1_viterbi_score_start",
                         f"k={k} cs={cs} n_chunk={n_chunk} sample_chunk=[{_s0},{_s1})")
                partial = _batched_viterbi_score(
                    np.ascontiguousarray(stacked_chunk), float(pen_sel))
                _cr_mark("CR_step1_viterbi_score_done", f"k={k} cs={cs} _s0={_s0}")
                accum_scores += partial
                del stacked_chunk
            
            for j, ci in enumerate(chunk):
                all_scores[ci] = float(accum_scores[j])
        del template
        _malloc_trim()
        _cr_mark("CR_step1_chunks_done", f"k={k}")
        best_idx = max(all_scores, key=all_scores.get)
        new_bic = ((len(selected) + 1) * batch_cc) - (2 * all_scores[best_idx])
        _cr_mark("CR_step1_iter_done",
                 f"k={k} best_idx={best_idx} best_score={all_scores[best_idx]:.2f} "
                 f"new_bic={new_bic:.2f} cur_best_bic={current_best_bic:.2f}")
        if new_bic < current_best_bic:
            current_best_bic = new_bic
            selected.append(best_idx)
        else:
            break
    _cr_mark("CR_step1_done", f"selected={selected}")
    
    # Fix: if Step 1 selected nothing (all candidates scored -inf or failed
    # the BIC comparison on the first iteration), fall back to the single
    # top-scoring beam result.  This avoids cascading an empty `selected`
    # through Steps 2-5, which produces n_paths=0 going into paint_samples_
    # viterbi and crashes the numba kernel on a (n_samples, 0, n_bins) tensor.
    # The fallback preserves correctness: we still return SOMETHING that
    # downstream reconstruct_haplotypes_from_beam can process, even if the
    # selection machinery numerically failed for this batch.
    if not selected:
        _cr_mark("CR_step1_fallback",
                 "Step 1 selected nothing -- using top beam result as fallback")
        # beam_results is sorted best-first coming out of run_full_mesh_beam_search.
        # Pick index 0 (the top-scoring path) as our single selected founder.
        if n_cands > 0:
            selected = [0]
        else:
            # Truly degenerate: no beam results at all.  Return empty list
            # immediately; _process_single_batch already handles this path
            # (it ends up with status='reconstruction_failed' rather than
            # crashing).
            _cr_mark("CR_step1_fallback_empty",
                     "beam_results is also empty -- returning []")
            _cr_mark("CR_return", "returning [] (degenerate input)")
            return []
    
    # =========================================================================
    # OUTER LOOP: iterate Steps 2, 3, 4 to full convergence (max 3 rounds)
    #
    # Step 4's chimera resolution can introduce splice products (key-paths
    # that aren't in the original beam) and transform the path set in ways
    # that enable new Step 2 swap moves, new Step 3 prunes, or additional
    # Step 4 splice opportunities.  Iterating Steps 2+3+4 together to
    # convergence catches those cross-step improvements.
    #
    # Strict-BIC acceptance throughout (Step 2's gain>1e-4 rule; Step 3's
    # BIC-improvement prune; Step 4's adj_gain>0 rule) ensures monotone BIC
    # descent and cannot regress.  Loop exits as soon as one iteration
    # produces no change in the selected set, or after MAX_OUTER_ITERATIONS
    # rounds -- whichever comes first.
    # =========================================================================
    MAX_OUTER_ITERATIONS = 3
    _cr_mark("CR_outer_loop_start",
             f"max_iters={MAX_OUTER_ITERATIONS} selected_in={selected}")

    # Reverse-lookup dict: beam key-paths -> beam indices.  Used to map any
    # splice product that happens to equal a beam entry back to that entry,
    # avoiding duplicate rows in map_matrix.  Built once before the loop,
    # then extended as new splices are introduced.
    beam_keypaths = {}
    for _ci, (_path, _) in enumerate(beam_results):
        _keys = tuple(fast_mesh.reverse_mappings[b][d]
                      for b, d in enumerate(_path))
        beam_keypaths[_keys] = _ci

    for _outer_iter in range(MAX_OUTER_ITERATIONS):
        _state_before = frozenset(selected)
        _cr_mark("CR_outer_iter_start",
                 f"outer_iter={_outer_iter} K={len(selected)} "
                 f"n_cands={n_cands}")

        # =========================================================================
        # STEP 2: Swap Refinement (sample-chunked)
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
                # Numba kernel replaces:
                #   pair_em = bin_em[:, np.repeat(local_haps, K_base),
                #                    np.tile(local_haps, K_base), :]
                #   base_maxes.append(np.max(pair_em, axis=1))
                # which allocated a (samples, K_base^2, bins) intermediate
                # per call (~10 MB at chr5 shapes, K_base^2 = 25). The
                # kernel computes the same per-(s, b) max over the K_base^2
                # (ii, jj) pairs without materializing the intermediate.
                # Bit-exact with fastmath=False — see kernel docstring.
                base_maxes.append(_precompute_base_max_block_numba(
                    bin_em,
                    np.ascontiguousarray(local_haps).astype(np.int64)))
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
                # Numba kernel replaces the per-hap numpy chain:
                #   for h in range(n_haps_local):
                #       cwt = bin_em[:, h, temp_haps, :]
                #       twc = bin_em[:, temp_haps, h, :]
                #       self_pair = bin_em[:, h, h, :]
                #       new_max = np.maximum(
                #           np.maximum(np.max(cwt, axis=1),
                #                      np.max(twc, axis=1)),
                #           self_pair)
                #       combined = np.maximum(bm, new_max)
                #       hap_contribs[h] = np.sum(combined)
                # which allocated ~60 MB of intermediates per call.
                # Bit-exact with fastmath=False — see kernel docstring.
                hap_contribs = _cheap_score_all_block_numba(
                    bin_em,
                    np.ascontiguousarray(temp_haps).astype(np.int64),
                    bm,
                    n_haps_local)
                cand_haps = map_matrix[candidates_arr, b_i]
                scores += hap_contribs[cand_haps]
            return {cand: scores[i] for i, cand in enumerate(candidates)}
    
        # -----------------------------------------------------------------
        # Batched 1-for-1 swap round (sample-chunked)
        # -----------------------------------------------------------------
        def _run_1for1_round(selected, get_candidates_fn, cur_score):
            """One round of batched 1-for-1 swaps across all positions.
        
            Processes each swap position independently — only ONE template slice
            exists at a time, avoiding the K × (num_samples, K², bins) memory
            explosion from precomputing all K templates simultaneously.
        
            Returns:
                (remove_idx, add_idx, score_gain) or None
            """
            _cr_mark("CR_1for1_start",
                     f"K={len(selected)} mode={get_candidates_fn.__name__ if hasattr(get_candidates_fn, '__name__') else '?'}")
            K = len(selected)
            if K < 2:
                _cr_mark("CR_1for1_done", "reason=K<2")
                return None
            n_pairs = K * K
            unselected = [x for x in range(n_cands) if x not in selected]
            if not unselected:
                _cr_mark("CR_1for1_done", "reason=no_unselected")
                return None
        
            K_base = K - 1
            cand_pos = K - 1
            _sc = min(num_samples, _SAMPLE_CHUNK)
            sc = max(4, min(64, int(5e8 / (_sc * n_pairs * total_bins * 8))))
            best_swap = None
            best_gain = 0.0
        
            # Process each swap position independently — only 1 template alive at a time
            for i in range(K):
                _cr_mark("CR_1for1_pos_iter_start", f"i={i}")
                temp_set = selected[:i] + selected[i + 1:]
                _cr_mark("CR_1for1_get_cands_start", f"i={i}")
                cands = get_candidates_fn(temp_set, unselected)
                _cr_mark("CR_1for1_get_cands_done", f"i={i} n_cands={len(cands) if cands else 0}")
                if not cands:
                    _cr_mark("CR_1for1_pos_iter_done", f"i={i} skipped=no_cands")
                    continue
            
                # Store hap indices for this position (tiny — just integer arrays)
                _cr_mark("CR_1for1_hpb_build_start", f"i={i}")
                hpb = {}
                for b_i, em_data in enumerate(sub_em):
                    hpb[b_i] = map_matrix[np.array(temp_set), b_i]
                # For the fused fill kernel: stack hpb[0..n_blocks-1]
                # into a single (cand_pos, n_blocks) int64 array once
                # per pos_iter so the fused kernel call doesn't rebuild
                # it on every fill dispatch.  cand_pos == K_base here.
                t_haps_stacked = np.stack(
                    [hpb[b_i] for b_i in range(len(sub_em))],
                    axis=1,
                ).astype(np.int64)
                _cr_mark("CR_1for1_hpb_build_done", f"i={i}")
            
                # Precompute templates_by_sample once per position iter.
                # The template slice depends on (i, _s0) but NOT on the
                # candidate chunk cs, so the original `for cs: for _s0:
                # build` rebuilt the same template once per cs chunk
                # (8064 builds aggregate at chr5 shapes vs 5760 needed).
                # Building once per _s0 here and reusing across cs chunks
                # eliminates the redundant rebuilds (most pronounced in
                # phase_C's all-candidates passes where each position
                # spawns multiple cs chunks). Mathematically byte-exact:
                # same templates, fewer rebuilds.
                _cr_mark("CR_1for1_template_build_start", f"i={i}")
                _sample_starts = list(range(0, num_samples, _SAMPLE_CHUNK))
                templates_by_sample = []
                for _s0 in _sample_starts:
                    _s1 = min(_s0 + _SAMPLE_CHUNK, num_samples)
                    _sn = _s1 - _s0
                    tmpl_slice = np.zeros((_sn, n_pairs, total_bins), dtype=np.float64)
                    bin_off = 0
                    for b_i, em_data in enumerate(sub_em):
                        bin_em = em_data['bin_emissions']
                        nb = em_data['n_bins']
                        t_haps = hpb[b_i]
                        for ii in range(K_base):
                            for jj in range(K_base):
                                pos = ii * K + jj
                                tmpl_slice[:, pos, bin_off:bin_off + nb] = \
                                    bin_em[_s0:_s1, t_haps[ii], t_haps[jj], :]
                        bin_off += nb
                    templates_by_sample.append(tmpl_slice)
                _cr_mark("CR_1for1_template_build_done",
                         f"i={i} n_templates={len(templates_by_sample)}")
            
                # Process candidates in chunks
                _cr_mark("CR_1for1_chunks_start", f"i={i} n_cands={len(cands)} sc={sc}")
                for cs in range(0, len(cands), sc):
                    chunk_cands = cands[cs:cs + sc]
                    n_chunk = len(chunk_cands)
                
                    # Accumulate scores across sample chunks
                    accum_scores = np.zeros(n_chunk, dtype=np.float64)
                    for _s0_idx, _s0 in enumerate(_sample_starts):
                        _s1 = min(_s0 + _SAMPLE_CHUNK, num_samples)
                        _sn = _s1 - _s0
                        # Reuse precomputed template (built once per
                        # position above, see CR_1for1_template_build).
                        tmpl_slice = templates_by_sample[_s0_idx]
                    
                        _cr_mark("CR_1for1_alloc_sv_start",
                                 f"i={i} cs={cs} _s0={_s0} "
                                 f"shape=({n_chunk},{_sn},{n_pairs},{total_bins})")
                        sv = np.empty((n_chunk, _sn, n_pairs, total_bins), dtype=np.float64)
                        _cr_mark("CR_1for1_alloc_sv_done",
                                 f"i={i} cs={cs} _s0={_s0}")
                    
                        _cr_mark("CR_1for1_fill_start",
                                 f"i={i} cs={cs} _s0={_s0} n_chunk={n_chunk}")
                        nt = _resolve_threads(num_threads)
                        # Fused numba fill kernel — one dispatch does the
                        # broadcast (sv[:] = tmpl[None,...]) AND per-block
                        # cand-row/col/diagonal fill in parallel via prange.
                        # Handles all n_haps cases via padding at chimera
                        # setup (see CR_stack_bin_em build).  Byte-exact
                        # with the pure-Python reference (verified by
                        # validate_fused_fill.py on synthetic L3-shaped
                        # inputs including jagged-padded scenarios).
                        chunk_cands_arr = np.asarray(chunk_cands, dtype=np.int64)
                        stride_n = cand_pos + 1  # == K for 1for1
                        # DIAGNOSTIC: log numba's actual thread count at
                        # kernel dispatch time and bracket the kernel-only
                        # portion so we can separate kernel wall from
                        # broadcast/setup wall.
                        _cr_mark("CR_1for1_numba_dispatch",
                                 f"nt_numba={_numba_get_num_threads()} "
                                 f"n_chunk={n_chunk} n_blocks={len(sub_em)}")
                        _cr_mark("CR_1for1_kernel_start",
                                 f"i={i} cs={cs} _s0={_s0}")
                        _fill_all_blocks_numba(
                            sv, tmpl_slice, stacked_bin_em,
                            chunk_cands_arr, map_matrix_c,
                            t_haps_stacked,
                            bin_offs_arr, nbs_arr,
                            _s0, _s1, stride_n, cand_pos)
                        _cr_mark("CR_1for1_kernel_done",
                                 f"i={i} cs={cs} _s0={_s0}")
                        _cr_mark("CR_1for1_fill_done",
                                 f"i={i} cs={cs} _s0={_s0} nt={nt}")
                    
                        _cr_mark("CR_1for1_viterbi_score_start",
                                 f"i={i} cs={cs} _s0={_s0} n_chunk={n_chunk}")
                        partial = _batched_viterbi_score(
                            np.ascontiguousarray(sv), float(pen_sel))
                        _cr_mark("CR_1for1_viterbi_score_done",
                                 f"i={i} cs={cs} _s0={_s0}")
                        accum_scores += partial
                        # tmpl_slice is a reference into templates_by_sample
                        # (precomputed once per position iter above). Free
                        # sv here; the template list is freed after the
                        # chunks loop ends.
                        del sv
                
                    for j, ci in enumerate(chunk_cands):
                        gain = float(accum_scores[j]) - cur_score
                        if gain > 1e-4 and gain > best_gain:
                            best_gain = gain
                            best_swap = (selected[i], ci)
                _cr_mark("CR_1for1_chunks_done", f"i={i}")
                # Free per-position templates before moving to next i.
                del templates_by_sample
                _cr_mark("CR_1for1_pos_iter_done", f"i={i}")
        
            _malloc_trim()
            _cr_mark("CR_1for1_done",
                     f"best_gain={best_gain:.4f} "
                     f"{'found' if best_swap else 'no_swap'}")
            return (best_swap[0], best_swap[1], best_gain) if best_swap else None
    
        # -----------------------------------------------------------------
        # Batched 2-for-1 swap round (sample-chunked)
        # -----------------------------------------------------------------
        def _run_2for1_round(selected, cur_score):
            """One round of batched 2-for-1 swaps (BIC-aware).
        
            For each pair (i,j) in selected, removes both and tries adding
            one candidate. Uses top-N cheap scoring per pair. Compares BIC
            since the result has K-1 members vs current K.
        
            Returns:
                (remove1, remove2, add_idx) or None
            """
            _cr_mark("CR_2for1_start", f"K={len(selected)}")
            K = len(selected)
            if K < 3:
                _cr_mark("CR_2for1_done", "reason=K<3")
                return None
        
            current_bic = K * batch_cc - 2 * cur_score
            K_result = K - 1
            n_pairs_r = K_result * K_result
            unselected = [x for x in range(n_cands) if x not in selected]
            if not unselected:
                _cr_mark("CR_2for1_done", "reason=no_unselected")
                return None
        
            K_base = K - 2
            cand_pos = K_result - 1
            _sc = min(num_samples, _SAMPLE_CHUNK)
            sc = max(4, min(64, int(5e8 / (_sc * n_pairs_r * total_bins * 8))))
            best_2for1 = None
            best_bic = current_bic
        
            # Process each pair independently — only 1 template slice alive at a time
            # (avoids C(K,2) × full-sample templates = 22 GB for K=12)
            for i in range(K):
                for j in range(i + 1, K):
                    _cr_mark("CR_2for1_pair_iter_start", f"i={i} j={j}")
                    temp_set = [selected[k] for k in range(K) if k != i and k != j]
                
                    # Get candidates for this pair via cheap scoring
                    _cr_mark("CR_2for1_cheap_score_start", f"i={i} j={j}")
                    bm = precompute_base_max(temp_set)
                    cs_scores = cheap_score_all(bm, temp_set, unselected)
                    ranked = sorted(cs_scores, key=cs_scores.get, reverse=True)[:top_n_swap]
                    _cr_mark("CR_2for1_cheap_score_done",
                             f"i={i} j={j} n_ranked={len(ranked)}")
                    if not ranked:
                        _cr_mark("CR_2for1_pair_iter_done", f"i={i} j={j} skipped=no_ranked")
                        continue
                
                    # Store hap indices for this pair (tiny)
                    _cr_mark("CR_2for1_hpb_build_start", f"i={i} j={j}")
                    hpb = {}
                    for b_i, em_data in enumerate(sub_em):
                        hpb[b_i] = map_matrix[np.array(temp_set), b_i]
                    # For the fused fill kernel: stack hpb[0..n_blocks-1]
                    # into a single (cand_pos, n_blocks) int64 array once
                    # per pair_iter so the fused kernel doesn't rebuild
                    # it on every fill dispatch.  cand_pos == K_base here,
                    # matching the 1for1 case.
                    t_haps_stacked = np.stack(
                        [hpb[b_i] for b_i in range(len(sub_em))],
                        axis=1,
                    ).astype(np.int64)
                    _cr_mark("CR_2for1_hpb_build_done", f"i={i} j={j}")
                
                    # Precompute templates_by_sample once per pair iter.
                    # Same restructure as _run_1for1_round: the template
                    # slice depends on (i, j, _s0) but NOT on the
                    # candidate chunk cs_start. Building once per _s0
                    # here and reusing across cs_start chunks eliminates
                    # the redundant rebuilds. Mathematically byte-exact:
                    # same templates, fewer rebuilds.
                    _cr_mark("CR_2for1_template_build_start", f"i={i} j={j}")
                    _sample_starts = list(range(0, num_samples, _SAMPLE_CHUNK))
                    templates_by_sample = []
                    for _s0 in _sample_starts:
                        _s1 = min(_s0 + _SAMPLE_CHUNK, num_samples)
                        _sn = _s1 - _s0
                        tmpl_slice = np.zeros((_sn, n_pairs_r, total_bins), dtype=np.float64)
                        bin_off = 0
                        for b_i, em_data in enumerate(sub_em):
                            bin_em = em_data['bin_emissions']
                            nb = em_data['n_bins']
                            t_haps = hpb[b_i]
                            for ii in range(K_base):
                                for jj in range(K_base):
                                    pos = ii * K_result + jj
                                    tmpl_slice[:, pos, bin_off:bin_off + nb] = \
                                        bin_em[_s0:_s1, t_haps[ii], t_haps[jj], :]
                            bin_off += nb
                        templates_by_sample.append(tmpl_slice)
                    _cr_mark("CR_2for1_template_build_done",
                             f"i={i} j={j} n_templates={len(templates_by_sample)}")
                
                    # Process candidates in chunks
                    _cr_mark("CR_2for1_chunks_start",
                             f"i={i} j={j} n_ranked={len(ranked)} sc={sc}")
                    for cs_start in range(0, len(ranked), sc):
                        chunk_cands = ranked[cs_start:cs_start + sc]
                        n_chunk = len(chunk_cands)
                    
                        # Accumulate scores across sample chunks
                        accum_scores = np.zeros(n_chunk, dtype=np.float64)
                        for _s0_idx, _s0 in enumerate(_sample_starts):
                            _s1 = min(_s0 + _SAMPLE_CHUNK, num_samples)
                            _sn = _s1 - _s0
                            # Reuse precomputed template (built once per
                            # pair above, see CR_2for1_template_build).
                            tmpl_slice = templates_by_sample[_s0_idx]
                        
                            _cr_mark("CR_2for1_alloc_sv_start",
                                     f"i={i} j={j} cs={cs_start} _s0={_s0} "
                                     f"shape=({n_chunk},{_sn},{n_pairs_r},{total_bins})")
                            sv = np.empty((n_chunk, _sn, n_pairs_r, total_bins), dtype=np.float64)
                            _cr_mark("CR_2for1_alloc_sv_done",
                                     f"i={i} j={j} cs={cs_start} _s0={_s0}")
                        
                            _cr_mark("CR_2for1_fill_start",
                                     f"i={i} j={j} cs={cs_start} _s0={_s0} n_chunk={n_chunk}")
                            nt = _resolve_threads(num_threads)
                            # Numba parallel path (identical pattern to
                            # 1for1, but stride = K_result since the pair
                            # layout is (K_result, K_result) instead of
                            # (K, K)). Two possible kernels — see 1for1
                            # comment for the full rationale.
                            chunk_cands_arr = np.asarray(chunk_cands, dtype=np.int64)
                            stride_n = cand_pos + 1  # == K_result for 2for1
                            # DIAGNOSTIC: same probe as 1for1 path.
                            _cr_mark("CR_2for1_numba_dispatch",
                                     f"nt_numba={_numba_get_num_threads()} "
                                     f"n_chunk={n_chunk} n_blocks={len(sub_em)}")
                            _cr_mark("CR_2for1_kernel_start",
                                     f"i={i} j={j} cs={cs_start} _s0={_s0}")
                            _fill_all_blocks_numba(
                                sv, tmpl_slice, stacked_bin_em,
                                chunk_cands_arr, map_matrix_c,
                                t_haps_stacked,
                                bin_offs_arr, nbs_arr,
                                _s0, _s1, stride_n, cand_pos)
                            _cr_mark("CR_2for1_kernel_done",
                                     f"i={i} j={j} cs={cs_start} _s0={_s0}")
                            _cr_mark("CR_2for1_fill_done",
                                     f"i={i} j={j} cs={cs_start} _s0={_s0} nt={nt}")
                        
                            _cr_mark("CR_2for1_viterbi_score_start",
                                     f"i={i} j={j} cs={cs_start} _s0={_s0} n_chunk={n_chunk}")
                            partial = _batched_viterbi_score(
                                np.ascontiguousarray(sv), float(pen_sel))
                            _cr_mark("CR_2for1_viterbi_score_done",
                                     f"i={i} j={j} cs={cs_start} _s0={_s0}")
                            accum_scores += partial
                            # tmpl_slice is a reference into templates_by_sample
                            # (precomputed once per pair iter above). Free
                            # sv here; the template list is freed after the
                            # chunks loop ends.
                            del sv
                    
                        for j_idx, ci in enumerate(chunk_cands):
                            new_score = float(accum_scores[j_idx])
                            new_bic = K_result * batch_cc - 2 * new_score
                            if new_bic < best_bic - 1e-4:
                                best_bic = new_bic
                                best_2for1 = (selected[i], selected[j], ci)
                    _cr_mark("CR_2for1_chunks_done", f"i={i} j={j}")
                    # Free per-pair templates before moving to next (i, j).
                    del templates_by_sample
                    _cr_mark("CR_2for1_pair_iter_done", f"i={i} j={j}")
        
            _malloc_trim()
            _cr_mark("CR_2for1_done",
                     f"{'found' if best_2for1 else 'no_swap'}")
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
        _cr_mark("CR_step2_start", f"selected={selected}")
        current_score = score_subset(selected)
    
        if len(selected) >= 2:
          _iter_step2 = 0
          while True:  # Outer loop: Phase A → Phase B → Phase C → repeat
            _iter_step2 += 1
            _cr_mark("CR_step2_outer_iter",
                     f"iter={_iter_step2} K={len(selected)} score={current_score:.2f}")
        
            # Phase A: top-N 1-for-1 until convergence
            _cr_mark("CR_step2_phase_A_start", f"iter={_iter_step2}")
            while True:
                result = _run_1for1_round(selected, _top_n_candidates,
                                          current_score)
                if result:
                    rm, add, gain = result
                    selected[selected.index(rm)] = add
                    current_score += gain
                else:
                    break
            _cr_mark("CR_step2_phase_A_done",
                     f"iter={_iter_step2} score={current_score:.2f}")
        
            # Phase B: one round of top-N 2-for-1 (BIC-aware)
            _cr_mark("CR_step2_phase_B_start", f"iter={_iter_step2}")
            current_score = score_subset(selected)  # resync after Phase A
            result_b = _run_2for1_round(selected, current_score)
            if result_b:
                rm1, rm2, add = result_b
                selected = [x for x in selected if x != rm1 and x != rm2] + [add]
                current_score = score_subset(selected)
                _cr_mark("CR_step2_phase_B_found", f"iter={_iter_step2}")
                continue  # Back to Phase A
            _cr_mark("CR_step2_phase_B_none", f"iter={_iter_step2}")
        
            # Phase C: one round of brute-force 1-for-1
            _cr_mark("CR_step2_phase_C_start", f"iter={_iter_step2}")
            result_c = _run_1for1_round(selected, _all_candidates,
                                         current_score)
            if result_c:
                rm, add, gain = result_c
                selected[selected.index(rm)] = add
                current_score += gain
                _cr_mark("CR_step2_phase_C_found", f"iter={_iter_step2}")
                continue  # Back to Phase A
            _cr_mark("CR_step2_phase_C_none", f"iter={_iter_step2}")
        
            break  # All three phases found nothing → done
        _cr_mark("CR_step2_done", f"selected={selected}")
    
        # =========================================================================
        # STEP 3: Force Prune + BIC Prune
        # =========================================================================
        _cr_mark("CR_step3_start",
                 f"K={len(selected)} max_founders={max_founders}")
        while len(selected) > max_founders:
            cur_ll = score_subset(selected)
            worst = min(selected, key=lambda idx:
                cur_ll - score_subset([x for x in selected if x != idx]))
            selected.remove(worst)
        _cr_mark("CR_step3_force_prune_done", f"K={len(selected)}")
    
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
        _cr_mark("CR_step3_done", f"selected={selected}")
    
        # Convert beam indices (possibly including splice-extension
        # indices from prior outer-loop iterations) to key-paths via
        # map_matrix.  map_matrix[bi] is byte-identical to
        # beam_results[bi][0] for original beam rows (populated in the
        # shared setup) and holds splice dense paths for extension rows
        # added at end-of-iteration conversions.
        _cr_mark("CR_key_paths_start")
        paths = []
        for bi in selected:
            path = map_matrix[bi]
            keys = [fast_mesh.reverse_mappings[b][int(d)]
                    for b, d in enumerate(path)]
            paths.append(keys)
        _cr_mark("CR_key_paths_done", f"n_paths={len(paths)}")
    
        # =========================================================================
        # STEP 4: Chimera Resolution
        # =========================================================================
        _cr_mark("CR_step4_start",
                 f"n_paths={len(paths)} max_cr_iterations={max_cr_iterations}")
        current_paths = list(paths)
        for iteration in range(max_cr_iterations):
            _cr_mark("CR_step4_iter_start",
                     f"iteration={iteration} K={len(current_paths)}")
            # 4a. Paint samples
            sp, K_cr = paint_samples_viterbi(
                current_paths, sub_em, paint_penalty, num_samples,
                num_threads=num_threads)
            _cr_mark("CR_step4_paint_done", f"iteration={iteration} K_cr={K_cr}")
        
            # 4b. Find hotspots
            hotspots = find_hotspots(
                sp, K_cr, n_blocks, sub_em, current_paths,
                num_samples, min_hotspot_samples)
            _cr_mark("CR_step4_hotspots_done",
                     f"iteration={iteration} n_hotspots={len(hotspots)}")
        
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
                current_paths, sub_em, pen_sel, num_samples,
                num_threads=num_threads)
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
                    current_paths, sub_em, pen_sel, num_samples,
                    num_threads=num_threads)
                k_now = len(current_paths)
                cur_bic = (k_now * batch_cc) - (2 * cur_ll)
                best_rem, best_bic = None, cur_bic
                for i in range(len(current_paths)):
                    trial = current_paths[:i] + current_paths[i + 1:]
                    trial_bic = ((k_now - 1) * batch_cc) - (
                        2 * score_path_set(trial, sub_em, pen_sel, num_samples,
                                           num_threads=num_threads))
                    if trial_bic < best_bic:
                        best_bic = trial_bic; best_rem = i
                if best_rem is not None:
                    current_paths = (current_paths[:best_rem]
                                    + current_paths[best_rem + 1:])
                else:
                    break
        _cr_mark("CR_step4_done", f"n_paths={len(current_paths)}")
    

        # =====================================================================
        # End-of-iteration: convert Step 4's current_paths (key-paths) back
        # into selected (beam indices or splice-extension indices).  Any new
        # splice products get assigned new indices n_cands, n_cands+1, ...
        # and corresponding new rows in map_matrix.  Step 2's closures
        # (which reference n_cands and map_matrix by name) automatically see
        # the extended values on the next outer-loop iteration.
        # =====================================================================
        _new_selected = []
        _new_dense_rows = []
        for _path_keys in current_paths:
            _keys_tup = tuple(_path_keys)
            if _keys_tup in beam_keypaths:
                _new_selected.append(beam_keypaths[_keys_tup])
            else:
                _dense = [fast_mesh.reverse_mappings[b].index(k)
                          for b, k in enumerate(_path_keys)]
                _new_idx = n_cands + len(_new_dense_rows)
                beam_keypaths[_keys_tup] = _new_idx
                _new_selected.append(_new_idx)
                _new_dense_rows.append(_dense)
        if _new_dense_rows:
            map_matrix = np.vstack([
                map_matrix,
                np.array(_new_dense_rows, dtype=map_matrix.dtype),
            ])
            n_cands = map_matrix.shape[0]
            # Rebuild the int64-contiguous cache used by the fused fill
            # kernel.  Without this, the next outer iteration's fill calls
            # would index map_matrix_c[ci, b_i] for ci in the newly-added
            # rows and read past the end of the cached array (out of bounds
            # — caught by boundscheck=True as IndexError, otherwise SIGSEGV).
            # bin_offs_arr / nbs_arr / stacked_bin_em are per-block, not
            # per-candidate, so they don't need rebuilding here.
            map_matrix_c = np.ascontiguousarray(map_matrix).astype(np.int64)
        selected = _new_selected
        _cr_mark("CR_outer_convert_done",
                 f"outer_iter={_outer_iter} K={len(selected)} "
                 f"n_splices_added={len(_new_dense_rows)} "
                 f"new_n_cands={n_cands}")

        # Early-break if this iteration didn't change the state.
        if frozenset(selected) == _state_before:
            _cr_mark("CR_outer_iter_converged",
                     f"outer_iter={_outer_iter} selected={selected}")
            break
    else:
        # Exited because MAX_OUTER_ITERATIONS reached (loop didn't break)
        _cr_mark("CR_outer_iter_max_reached",
                 f"max_iters={MAX_OUTER_ITERATIONS} selected={selected}")
    _cr_mark("CR_outer_loop_done",
             f"final_K={len(selected)} final_n_cands={n_cands}")

    # Rebuild current_paths from the final selected one more time so Step 5
    # sees a consistent key-path list (map_matrix rows cover both original
    # beam entries and any splice-extension indices added during the loop).
    current_paths = [
        [fast_mesh.reverse_mappings[b][int(d)]
         for b, d in enumerate(map_matrix[_bi])]
        for _bi in selected
    ]

    # =========================================================================
    # STEP 5: Convert resolved key-paths back to beam format
    # =========================================================================
    _cr_mark("CR_step5_start", f"n_paths={len(current_paths)}")
    resolved_beam = []
    for path_keys in current_paths:
        dense_path = [fast_mesh.reverse_mappings[b].index(key)
                      for b, key in enumerate(path_keys)]
        resolved_beam.append((dense_path, 0.0))
    _cr_mark("CR_step5_done", f"n_resolved={len(resolved_beam)}")
    _cr_mark("CR_return", f"returning {len(resolved_beam)} beam entries")
    
    return resolved_beam