"""
Chimera Resolution Module

Sub-block forward selection, top-N swap refinement, BIC pruning,
and chimera resolution via hotspot-guided splicing.

Main entry point: select_and_resolve()
"""

import numpy as np
import math
import ctypes
import heapq
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

from scipy.optimize import linear_sum_assignment

import block_haplotypes

# glibc malloc_trim — releases freed pages back to OS
try:
    _libc = ctypes.CDLL("libc.so.6")
    def _malloc_trim():
        _libc.malloc_trim(0)
except OSError:
    def _malloc_trim():
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
def _batched_viterbi_score_split(tmpl_slice, cand_sv, cand_idx_for_p, penalty):
    """Score multiple candidate sets via Viterbi, reading emissions from
    a SPLIT layout (shared template + per-candidate cand cells) instead
    of one big single-source sv tensor.

    A single-source layout would duplicate the per-position template
    into n_chunk separate slabs of sv (`sv[local_idx, s, p, b] =
    tmpl[s, p, b]` for non-cand pairs), so a fill on n_chunk = 50
    candidates would write ~1.8 GB per call.  At K=6 1for1, more than
    69% of those cells are identical across all n_chunk slabs.  The
    split layout avoids the duplication by storing the shared portion
    once.

    Split layout:
      - tmpl_slice  shape (n_samples, n_pairs, total_bins): the shared
        template, allocated once per swap position.  Used for the
        ~70% of pair indices that don't involve the candidate row or
        column.
      - cand_sv     shape (n_chunk, n_samples, n_cand_pairs, total_bins):
        per-candidate cand-row + cand-col + diagonal emissions only.
        n_cand_pairs == 2*cand_pos + 1 (e.g. 11 at K=6 1for1, vs 36 in
        a full pair grid — 3.3x reduction on this dim).
      - cand_idx_for_p shape (n_pairs,) int64: maps pair-index k to:
            -1                      → use tmpl_slice[s, k, t]
            0..(cand_pos-1)         → cand_sv[b, s, k_idx, t] (cand-row[ii])
            cand_pos..(2*cand_pos-1) → cand_sv[b, s, k_idx, t] (cand-col[jj])
            2*cand_pos              → cand_sv[b, s, k_idx, t] (diagonal)

    Same Viterbi recurrence as _batched_viterbi_score; only the emission
    read is split.  The set of double-precision values fed into the
    recurrence is bit-identical to what a single-source sv would supply
    (the cand_idx_for_p dispatch reads exactly the same double from
    whichever source holds it).  Output sample sums also bit-identical
    under fastmath=True since the loop order and value sequence are
    unchanged.

    Parallelism: prange over (batch × samples) as before.  Each thread
    reads its own batch's cand_sv slab plus a shared tmpl_slice; tmpl is
    L3-cache-resident at production shapes (~25 MB non-cand portion)
    and shared across all threads, so the cache-hit rate on tmpl reads
    is much higher than the old per-batch sv layout.

    Args:
        tmpl_slice:        (n_samples, n_pairs, total_bins) float64,
                           read-only, shared across batches.  Cells at
                           cand_idx_for_p[k] >= 0 are NEVER read (those
                           positions come from cand_sv instead), so the
                           caller may leave them uninitialised (np.empty)
                           if convenient.
        cand_sv:           (n_chunk, n_samples, 2*cand_pos+1, total_bins)
                           float64, read-only.  Per-batch slab; written
                           by _build_cand_sv_numba beforehand.
        cand_idx_for_p:    (n_pairs,) int64, precomputed pair-index →
                           cand-slot lookup (see above).  Constant for
                           a given (stride, cand_pos), so built once per
                           swap round in Python.
        penalty:           float, Viterbi switch penalty.

    Returns:
        scores: (n_chunk,) float64, per-batch summed log-likelihoods.
    """
    n_chunk = cand_sv.shape[0]
    n_samples = cand_sv.shape[1]
    n_bins = cand_sv.shape[3]
    n_pairs = tmpl_slice.shape[1]

    sample_scores = np.empty((n_chunk, n_samples), dtype=np.float64)
    n_total = n_chunk * n_samples

    for idx in prange(n_total):
        b = idx // n_samples
        s = idx % n_samples

        current_scores = np.empty(n_pairs, dtype=np.float64)
        # Init at t=0 — same value sequence as the old single-source kernel
        # would have read from sv[b, s, k, 0].
        for k in range(n_pairs):
            cidx = cand_idx_for_p[k]
            if cidx < 0:
                current_scores[k] = tmpl_slice[s, k, 0]
            else:
                current_scores[k] = cand_sv[b, s, cidx, 0]

        # Viterbi recurrence — identical structure to _batched_viterbi_score:
        # find max of current_scores, subtract penalty for the switch_base,
        # then for each k pick max(stay, switch) + emission.  Only the
        # emission source differs.
        for t in range(1, n_bins):
            best_prev = -np.inf
            for k in range(n_pairs):
                if current_scores[k] > best_prev:
                    best_prev = current_scores[k]
            switch_base = best_prev - penalty
            for k in range(n_pairs):
                cidx = cand_idx_for_p[k]
                if cidx < 0:
                    emission = tmpl_slice[s, k, t]
                else:
                    emission = cand_sv[b, s, cidx, t]
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

    # Phase 2: sum across samples per batch — same as _batched_viterbi_score.
    scores = np.empty(n_chunk, dtype=np.float64)
    for b in range(n_chunk):
        total = 0.0
        for s in range(n_samples):
            total += sample_scores[b, s]
        scores[b] = total
    return scores


@njit(parallel=True, fastmath=False)
def _build_cand_sv_numba(cand_sv, bin_ems_stacked,
                          chunk_cands_arr, map_matrix,
                          t_haps_stacked, bin_offs, nbs,
                          _s0, _s1, cand_pos):
    """Build the compact per-candidate emission tensor for the split-layout
    Viterbi (_batched_viterbi_score_split).

    Writes ONLY the cand-row + cand-col + diagonal emissions — the
    template (non-cand) data lives separately in tmpl_slice and is read
    directly by the Viterbi kernel.

    cand_sv[local_idx, s, cidx, b]:
      - cidx in [0, cand_pos):
          cand-row[ii=cidx], emission =
            bin_ems_stacked[_s0+s, t_haps_stacked[ii, b_i], h_c[b_i], b]
      - cidx in [cand_pos, 2*cand_pos):
          cand-col[jj=cidx-cand_pos], emission =
            bin_ems_stacked[_s0+s, h_c[b_i], t_haps_stacked[jj, b_i], b]
      - cidx = 2*cand_pos:
          diagonal, emission = bin_ems_stacked[_s0+s, h_c[b_i], h_c[b_i], b]

    where h_c[b_i] = map_matrix[chunk_cands_arr[local_idx], b_i] is the
    candidate's hap-id for block b_i, and b runs over [bin_offs[b_i],
    bin_offs[b_i] + nbs[b_i]) — i.e. the bins owned by block b_i in the
    stacked total_bins axis.

    The compact `(n_chunk, _sn, 2*cand_pos+1, total_bins)` layout stores
    each candidate's per-pair emissions for the cand-row + cand-col +
    diagonal cells only (the cells whose values differ across
    candidates).  Non-cand cells are read directly from tmpl_slice by
    the downstream Viterbi kernel, avoiding any full-pair-grid sv
    materialisation.

    Parallelism: prange over local_idx (each thread writes its own
    cand_sv[local_idx] slab; no race).

    Args:
        cand_sv:           (n_chunk, _sn, 2*cand_pos+1, total_bins) float64,
                           OUTPUT, written in-place via np.empty().
                           This kernel fully overwrites all cand_sv
                           cells, so the np.empty garbage is safe.
        bin_ems_stacked:   (num_samples, n_haps, n_haps, total_bins)
                           float64, read-only.  Per-block emission
                           tensors padded to a common n_haps and
                           concatenated along the bins axis.
        chunk_cands_arr:   (n_chunk,) int64, candidate row indices.
        map_matrix:        (n_cands_total, n_blocks) int64.
        t_haps_stacked:    (cand_pos, n_blocks) int64.  Note: cand_pos here
                           equals K_base.
        bin_offs:          (n_blocks,) int64.
        nbs:               (n_blocks,) int64.
        _s0, _s1:          int, sample range.  _sn = _s1 - _s0.
        cand_pos:          int, the candidate position index in the
                           (stride, stride) pair grid (== K_base in both
                           1for1 and 2for1).
    """
    n_chunk = chunk_cands_arr.shape[0]
    n_blocks = bin_offs.shape[0]
    _sn = _s1 - _s0
    for local_idx in prange(n_chunk):
        ci = chunk_cands_arr[local_idx]
        for b_i in range(n_blocks):
            bin_off = bin_offs[b_i]
            nb = nbs[b_i]
            h_c = map_matrix[ci, b_i]
            # cand-row: cidx = ii  for ii in [0, cand_pos)
            for ii in range(cand_pos):
                t_h = t_haps_stacked[ii, b_i]
                for s in range(_sn):
                    for bin_local in range(nb):
                        cand_sv[local_idx, s, ii, bin_off + bin_local] = \
                            bin_ems_stacked[_s0 + s, t_h, h_c,
                                            bin_off + bin_local]
            # cand-col: cidx = cand_pos + jj  for jj in [0, cand_pos)
            for jj in range(cand_pos):
                t_h = t_haps_stacked[jj, b_i]
                for s in range(_sn):
                    for bin_local in range(nb):
                        cand_sv[local_idx, s, cand_pos + jj, bin_off + bin_local] = \
                            bin_ems_stacked[_s0 + s, h_c, t_h,
                                            bin_off + bin_local]
            # Diagonal: cidx = 2 * cand_pos
            for s in range(_sn):
                for bin_local in range(nb):
                    cand_sv[local_idx, s, 2 * cand_pos, bin_off + bin_local] = \
                        bin_ems_stacked[_s0 + s, h_c, h_c,
                                        bin_off + bin_local]


def _build_cand_idx_for_p(stride, cand_pos):
    """Build the pair-index → cand-slot lookup for the split-layout
    Viterbi.

    For a (stride, stride) pair grid where cand_pos = stride - 1:
        row = p // stride, col = p - row * stride
        result[p] = -1                  if row < cand_pos and col < cand_pos
                    row                  if col == cand_pos and row < cand_pos
                    cand_pos + col       if row == cand_pos and col < cand_pos
                    2 * cand_pos         if row == cand_pos and col == cand_pos

    Pure-Python; runs once per swap round and the result (int64 array of
    length n_pairs = stride²) is fed into _batched_viterbi_score_split.
    """
    n_pairs = stride * stride
    out = np.empty(n_pairs, dtype=np.int64)
    for p in range(n_pairs):
        row = p // stride
        col = p - row * stride
        if row == cand_pos and col == cand_pos:
            out[p] = 2 * cand_pos
        elif row == cand_pos:
            out[p] = cand_pos + col
        elif col == cand_pos:
            out[p] = row
        else:
            out[p] = -1
    return out


@njit(fastmath=False)
def _build_tensor_block_numba(tensor, bin_em, local_indices,
                                s0, s1, K, n_bins_b, bin_offset):
    """Write one sub-block's (n_pairs × n_bins_b) slice into the scoring
    tensor, called from _build_tensor_from_paths once per sub-block per
    path-set.

    Equivalent to the numpy reference:
        grid_i = np.repeat(local_indices, K)     # shape (K*K,)
        grid_j = np.tile(local_indices, K)       # shape (K*K,)
        tensor[:, :, bin_offset:bin_offset+n_bins_b] = \\
            bin_em[s0:s1, grid_i, grid_j, :]

    with the indexing relation
        grid_i[p] = local_indices[p // K]
        grid_j[p] = local_indices[p %  K]

    The numpy fancy-indexing path would allocate an intermediate
    (n_s, K², n_bins_b) array per call and copy through Python-level
    indexing machinery.  This kernel writes directly via three nested
    loops, eliminating the temporary and inlining the index resolution.
    At K=6, n_pairs=36, n_bins_b≈200 production shapes, the per-call
    work is ~460k writes (~3.7 MB); numba executes that serially in
    well under a millisecond.

    SERIAL (no `parallel=True`, no `prange`) — IMPORTANT:
    `_build_tensor_from_paths` is reached from `score_path_sets_parallel`,
    which already runs the per-path-set builds inside a Python
    `ThreadPoolExecutor` (one Python thread per builder).  If THIS
    kernel also used `prange`, every Python thread would try to acquire
    the same numba OpenMP thread pool, causing the OMP barrier to
    serialise everything and adding contention overhead — empirically
    a ~2.4x slowdown vs the parallel-kernel attempt at 112-way
    production scale.  Serial execution lets the outer
    ThreadPoolExecutor own the parallelism instead.

    The per-call work is small (~460k writes per block per call) and
    the function is called from many parallel contexts simultaneously,
    so the outer-pool parallelism is the right place to scale.  Inside
    `_run_1for1_round`, `_run_2for1_round`, and Step 1 (where this
    function is also reached, but under a single Python thread), the
    numba thread budget is correctly used by the *other* parallel
    kernels in the same call (`_build_cand_sv_numba`,
    `_batched_viterbi_score_split`) — this serial kernel just removes
    Python-level indexing overhead from its own slice of the call.

    Args:
        tensor:        (n_s, n_pairs, total_bins) float64, OUTPUT.
                       Slice [:, :, bin_offset:bin_offset+n_bins_b] is
                       written in-place; cells outside that bin window
                       are untouched.  Caller is responsible for
                       initialising regions not covered by this block.
        bin_em:        (num_samples, n_haps, n_haps, n_bins_b) float64,
                       READ-only.  This block's emission tensor.
        local_indices: (K,) int64.  Per-path local hap indices for this
                       block.
        s0, s1:        int, sample range.  n_s = s1 - s0.
        K:             int, path-set size.  n_pairs = K * K.
        n_bins_b:      int, this block's bin count.
        bin_offset:    int, where to start writing in the tensor's
                       total_bins axis (cumulative bin offset).
    """
    n_s = s1 - s0
    n_pairs = K * K
    for s in range(n_s):
        s_src = s0 + s
        for p in range(n_pairs):
            row = p // K
            col = p - row * K
            i_local = local_indices[row]
            j_local = local_indices[col]
            for b in range(n_bins_b):
                tensor[s, p, bin_offset + b] = \
                    bin_em[s_src, i_local, j_local, b]


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
def _build_template_numba(tmpl_full, bin_ems_stacked,
                           t_haps_stacked, bin_offs, nbs,
                           K_base, stride):
    """Build the per-position/pair template tensor for one swap round.

    Computes, for each sample s, per-block (b_i), per-pair (ii, jj) with
    ii, jj in [0, K_base):

        pos = ii * stride + jj
        tmpl_full[s, pos, bin_off + bin_local] =
            bin_ems_stacked[s, t_haps_stacked[ii, b_i],
                            t_haps_stacked[jj, b_i],
                            bin_off + bin_local]

    Cells at (row == cand_pos) or (col == cand_pos), where
    cand_pos = stride - 1 = K_base, are left untouched (uninitialised
    from the np.empty allocator).  The Viterbi kernel never reads those
    cells from tmpl_slice — the candidate fill kernel
    (_build_cand_sv_numba) supplies real data for every cand-row /
    cand-col / diagonal cell, and the Viterbi recurrence dispatches
    through `cand_idx_for_p[k]` to read from cand_sv whenever k is a
    cand cell.

    Equivalent to the Python reference:

        for b_i: for ii: for jj:
            tmpl[:, pos, bin_off:bin_off+nb] = \
                bin_em[_s0:_s1, t_haps[ii], t_haps[jj], :]

    which ran K * n_sample_chunks * n_blocks * K_base² slice-copy
    operations per swap round (~7500 at K=6, 10 blocks, 5 chunks),
    each through numpy's fancy-indexing machinery with Python
    interpreter overhead.  This kernel folds the whole thing into one
    prange'd numba region and writes via direct indexing.

    Parallelism: prange over samples (each thread writes its own
    tmpl_full[s, :, :] row; no race).  Numba thread count is set by
    the caller via numba.set_num_threads.

    Args:
        tmpl_full:        (num_samples, n_pairs, total_bins) float64,
                          OUTPUT, written in-place. Caller pre-allocates
                          via np.empty; this kernel writes only the
                          non-cand cells (see docstring above).
        bin_ems_stacked:  (num_samples, n_haps, n_haps, total_bins)
                          float64, read-only. Per-block emission tensors
                          padded to a common n_haps and concatenated
                          along the bins axis.
        t_haps_stacked:   (K_base, n_blocks) int64. t_haps_stacked[ii, b_i]
                          gives the hap-id for basis position ii in block b_i.
        bin_offs:         (n_blocks,) int64, cumulative bin offsets.
        nbs:              (n_blocks,) int64, per-block bin counts.
        K_base:           int, number of basis (non-candidate) positions.
                          Equal to K-1 for 1for1, K-2 for 2for1.
        stride:           int, pair-index row stride. Equal to
                          K (1for1) or K_result=K-1 (2for1). Also equal
                          to K_base + 1 since cand_pos = K_base.

    Mathematically byte-equivalent to the Python reference:

        tmpl_full = np.zeros((num_samples, n_pairs, total_bins), dtype=float64)
        bin_off = 0
        for b_i in range(n_blocks):
            nb = nbs[b_i]
            for ii in range(K_base):
                for jj in range(K_base):
                    pos = ii * stride + jj
                    tmpl_full[:, pos, bin_off:bin_off+nb] = bin_ems_stacked[
                        :, t_haps_stacked[ii, b_i],
                        t_haps_stacked[jj, b_i],
                        bin_off:bin_off+nb]
            bin_off += nb

    except: tmpl_full is allocated via np.empty (cand cells uninitialised);
    np.zeros vs np.empty differs only at cand cells which downstream never
    reads, so the difference is mathematically and observably equivalent.
    """
    num_samples = tmpl_full.shape[0]
    n_blocks = bin_offs.shape[0]
    for s in prange(num_samples):
        for b_i in range(n_blocks):
            bin_off = bin_offs[b_i]
            nb = nbs[b_i]
            for ii in range(K_base):
                t_h_i = t_haps_stacked[ii, b_i]
                for jj in range(K_base):
                    pos = ii * stride + jj
                    t_h_j = t_haps_stacked[jj, b_i]
                    for bin_local in range(nb):
                        tmpl_full[s, pos, bin_off + bin_local] = \
                            bin_ems_stacked[s, t_h_i, t_h_j,
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
        # Precompute a {hap_key: local_index_in_hap_keys} dict so downstream
        # call sites that map path entries → local hap indices can use an
        # O(1) lookup instead of repeating list.index() (O(n_haps) per call).
        # _build_tensor_from_paths is called ~1500 times per L1 batch and does
        # K such lookups per block, so cache hit-rate is high; the other two
        # call sites (_compute_block_diffs, all-pair-diffs build) consume the
        # same field.  Built once per block here; never mutated downstream.
        key_to_local_idx = {k: i for i, k in enumerate(hap_keys)}
        all_emissions.append({
            'hap_keys': hap_keys,
            'bin_emissions': bin_emissions,
            'n_bins': n_bins,
            'key_to_local_idx': key_to_local_idx,
        })
    _malloc_trim()
    return all_emissions


# =============================================================================
# TENSOR BUILDING AND SCORING
# =============================================================================

def _build_tensor_from_paths(path_set, sub_emissions, num_samples, sample_range=None):
    """Build Viterbi scoring tensor from a set of key-paths.
    If sample_range=(s0, s1) is given, only builds for those samples.

    Per-block fill goes through _build_tensor_block_numba, a serial
    numba kernel that writes the (n_pairs × n_bins_b) slice directly
    via three nested loops, avoiding the numpy fancy-index + slice-
    assign intermediate `tensor[:, :, off:off+nb] = bin_em[s0:s1,
    grid_i, grid_j, :]` (which would allocate a (n_s, K², n_bins_b)
    temporary per block per call).
    """
    K = len(path_set)
    n_pairs = K * K
    total_bins = sum(e['n_bins'] for e in sub_emissions)
    if sample_range is not None:
        s0, s1 = sample_range
        n_s = s1 - s0
    else:
        s0, s1 = 0, num_samples
        n_s = num_samples
    # np.empty rather than np.zeros — the per-block numba kernel fully
    # writes every (s, p, b) cell in its own bin window (b in
    # [bin_offset, bin_offset + n_bins_b)); the union of those windows
    # is exactly [0, total_bins), so the entire tensor is overwritten
    # by the end of the loop.  np.zeros' implicit page-zeroing was
    # wasted work.
    tensor = np.empty((n_s, n_pairs, total_bins), dtype=np.float64)
    bin_offset = 0
    for b_idx, em_data in enumerate(sub_emissions):
        n_bins_b = em_data['n_bins']
        bin_em = em_data['bin_emissions']
        # Use the precomputed {hap_key: local_idx} dict (built once in
        # compute_subblock_emissions) instead of K linear searches via
        # list.index().  Falls back gracefully for any externally-built
        # sub_emissions that didn't supply the cache.
        key_to_local = em_data.get('key_to_local_idx')
        if key_to_local is not None:
            local_indices = np.array(
                [key_to_local[path[b_idx]] for path in path_set],
                dtype=np.int64,
            )
        else:
            hap_keys = em_data['hap_keys']
            local_indices = np.array(
                [hap_keys.index(path[b_idx]) for path in path_set],
                dtype=np.int64,
            )
        # Numba kernel writes tensor[:, :, bin_offset:bin_offset+n_bins_b]
        # via direct three-loop indexing — same values as the numpy
        # fancy-index path, no intermediate allocation.  Edge case K=0
        # gives n_pairs=0; the kernel's inner loop is a no-op and the
        # tensor's bin window stays at np.empty garbage.  In practice
        # K=0 doesn't reach this function (paint_samples_viterbi has an
        # explicit early-return at line ~1314); guarding here is a
        # defensive harmless cost.
        if n_pairs > 0:
            _build_tensor_block_numba(
                tensor, bin_em, local_indices,
                s0, s1, K, n_bins_b, bin_offset,
            )
        bin_offset += n_bins_b
    # `tensor` was allocated via np.empty and overwritten in-place by
    # the per-block kernel; it remains C-contiguous.  No
    # np.ascontiguousarray wrap needed.
    return tensor


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
# BEAM WARMSTART INITIALISATION (seeds Step 1's selected set)
# =============================================================================

def beam_warmstart_select(beam_results, fast_mesh, sub_emissions,
                          penalty, num_samples, batch_cc,
                          max_K_cap=20, num_threads=None):
    """Greedy beam-warmstart that produces an initial `selected` list
    to seed select_and_resolve's Step 1 forward selection.

    Walks the beam in score order (best LL first), trying to ADD each
    candidate to the current selected set; if the addition lowers BIC
    it is committed.  After every successful add, runs a DROP-LOOP that
    tests removing each currently-selected member; any drop that
    further lowers BIC is committed, and the loop repeats until no
    single drop improves further.

    The output seeds Step 1's `selected` and `current_best_bic` so
    Step 1's per-set-best-add greedy starts from a non-empty set rather
    than from K=0.  Step 1 still runs on top (warmstart and Step 1 use
    different acceptance criteria — warmstart is "first improvement
    in beam order", Step 1 is "best improvement across all candidates"
    — so Step 1 can still find additions warmstart missed).

    Motivation: the original Step 1, starting from empty, committed to
    splice paths at K=1 in cases where the data has within-sub-block
    recombinations (chr23 SB 129).  No 1-for-1 or 2-for-1 swap in the
    outer loop could escape the splice basin afterwards (basin-shape
    verification: requires >=4 simultaneous swaps).  Seeding Step 1
    with a beam-walk-derived set avoids the basin entirely because
    the beam-walk considers each high-LL beam path as an independent
    candidate add rather than only the per-set winner at each k.

    Mathematical equivalence with the rest of the resolver: the BIC
    used here (K * batch_cc - 2 * LL) is the same one Step 2's swap
    acceptance, Step 3's prune, and Step 4's adj_gain all use.  LL is
    obtained from score_path_set / score_path_sets_parallel, which
    dispatch the same _batched_viterbi_score / viterbi_score_selection
    kernels used everywhere else in this module.  The penalty argument
    matches pen_sel inside select_and_resolve, so warmstart's
    `current_best_bic = bic_S` returned to Step 1 is byte-identical to
    what Step 1 would compute for that same selected set.

    Threading: score_path_set is called O(beam_size) times for the add
    step; score_path_sets_parallel is called once per accept for the
    drop loop (K candidate (K-1)-sets in one batched dispatch).  Both
    accept num_threads as int OR callable, so passing a callable (e.g.
    hierarchical_assembly._get_dynamic_threads) enables dynamic thread
    reallocation as peer workers finish — same convention as the rest
    of select_and_resolve.

    Args:
        beam_results: list of (dense_path, score) tuples, sorted best
            score first (the order returned by run_full_mesh_beam_search).
        fast_mesh: FastMesh object — used to decode dense paths to
            key-paths via fast_mesh.reverse_mappings[b][d].
        sub_emissions: per-sub-block emission cache from
            compute_subblock_emissions.
        penalty: per-bin Viterbi recomb penalty (== pen_sel inside
            select_and_resolve).
        num_samples: number of samples in the batch.
        batch_cc: per-founder BIC complexity penalty (== batch_cc
            inside select_and_resolve).
        max_K_cap: hard upper bound on K reached by warmstart.  The
            beam-walk stops attempting further adds once K hits this
            cap.  Default 20; in practice the BIC criterion stops
            growth far below this on realistic data (K typically 5-8
            on L1 super-blocks).  Caller in select_and_resolve passes
            its step1_max_iters argument so the two phases scale
            together if the user adjusts that parameter.
        num_threads: int or callable; passed through to score_path_set
            and score_path_sets_parallel.

    Returns:
        (selected_beam_indices, final_bic)
            selected_beam_indices: list of indices into beam_results
                (== indices into map_matrix in select_and_resolve, since
                 map_matrix is built by row-aligned iteration over
                 beam_results).  May be empty if the beam is empty;
                 in that case Step 1 starts from K=0 just like the
                 original behaviour.
            final_bic: float, BIC of the final selected set; assigned
                to current_best_bic at Step 1 entry so Step 1's accept
                criterion compares against the warmstart baseline.
                float('inf') for empty selected (matches original
                behaviour).
    """
    n_cands = len(beam_results)
    if n_cands == 0:
        return [], float('inf')

    # Decode every beam path's dense_path to a key-path once.
    # score_path_set / score_path_sets_parallel both expect key-paths
    # (lists of stage-3 hap keys), not dense fast_mesh indices.
    decoded = []
    for dp, _score in beam_results:
        keys = [int(fast_mesh.reverse_mappings[b][int(d)])
                for b, d in enumerate(dp)]
        decoded.append(keys)

    # Deduplicate by key-path tuple.  Two beam paths can in principle
    # decode identically even if their dense paths differ (FastMesh can
    # have duplicate local-index slots for the same key in some sub-
    # block).  Keep the first occurrence (best LL).  Original Step 1
    # implicitly avoided duplicates because at each k it skipped indices
    # already in selected; we replicate that effect explicitly here.
    seen = set()
    unique_beam_idx = []
    unique_keypaths = []
    for bi, kp in enumerate(decoded):
        kt = tuple(kp)
        if kt in seen:
            continue
        seen.add(kt)
        unique_beam_idx.append(bi)
        unique_keypaths.append(kp)

    selected_keypaths = []          # list of key-paths (parallel to ...)
    selected_beam_idx = []          # ... this list of beam-row indices
    selected_tuples = set()         # for O(1) "is path in S?" checks
    bic_S = float('inf')            # empty set has +inf BIC; any K>=1
                                    # set with finite LL strictly improves
    n_adds = 0
    n_drops = 0
    n_rejects = 0

    for beam_idx, kp in zip(unique_beam_idx, unique_keypaths):
        kt = tuple(kp)
        if kt in selected_tuples:
            continue                # already in selected, skip
        if len(selected_keypaths) >= max_K_cap:
            break

        # ------ Try ADD ------
        candidate_set = selected_keypaths + [list(kp)]
        ll_cand = score_path_set(
            candidate_set, sub_emissions, penalty, num_samples,
            num_threads=num_threads)
        bic_cand = len(candidate_set) * batch_cc - 2.0 * ll_cand

        if bic_cand < bic_S:
            # Accept: bind in the new path, update BIC.
            selected_keypaths = candidate_set
            selected_beam_idx.append(beam_idx)
            selected_tuples.add(kt)
            bic_S = bic_cand
            n_adds += 1

            # ------ DROP LOOP ------
            # After each successful add, check whether any current
            # member can be dropped for further BIC improvement.  An
            # earlier drop may expose a later one (e.g. once path X is
            # gone, path Y becomes redundant), so we loop until no
            # single drop improves further.
            #
            # All K candidate (K-1)-sets are evaluated in ONE call to
            # score_path_sets_parallel, which batches them into a
            # single _batched_viterbi_score dispatch with batch dim K
            # and sample dim num_samples — gives K * num_samples
            # parallel work units, fully utilising whatever thread
            # count _resolve_threads(num_threads) returns.
            while len(selected_keypaths) > 1:
                K_cur = len(selected_keypaths)
                cand_sets = [
                    [selected_keypaths[j] for j in range(K_cur) if j != i]
                    for i in range(K_cur)
                ]
                lls = score_path_sets_parallel(
                    cand_sets, sub_emissions, penalty, num_samples,
                    num_threads=num_threads)
                bics = [(K_cur - 1) * batch_cc - 2.0 * ll for ll in lls]

                best_drop_i = -1
                best_drop_bic = bic_S
                for i, b in enumerate(bics):
                    if b < best_drop_bic:
                        best_drop_bic = b
                        best_drop_i = i

                if best_drop_i < 0:
                    break                       # no drop improves; done

                # Commit drop.  Remove from all parallel structures so
                # selected_beam_idx[i] continues to align with
                # selected_keypaths[i].  Also remove from
                # selected_tuples — defensive: lets the path become
                # eligible for re-addition later if some future accept
                # exposes it as optimal again.  (Doesn't fire on
                # well-behaved blocks; pathological beams only.)
                dropped_kp = selected_keypaths[best_drop_i]
                dropped_bidx = selected_beam_idx[best_drop_i]
                dropped_kt = tuple(dropped_kp)
                selected_tuples.discard(dropped_kt)
                selected_keypaths = [selected_keypaths[i]
                                     for i in range(K_cur)
                                     if i != best_drop_i]
                selected_beam_idx = [selected_beam_idx[i]
                                     for i in range(K_cur)
                                     if i != best_drop_i]
                bic_S = best_drop_bic
                n_drops += 1
        else:
            n_rejects += 1

    return selected_beam_idx, bic_S


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


@njit(cache=True, parallel=True)
def _compute_all_pair_diffs_kernel(bin_em, pair_si_idx, pair_sj_idx,
                                     out_mean_diffs, out_offset):
    """Per-block mean-diff computation for all hap-pairs in one block.

    For each pair p, computes:
        out_mean_diffs[p, out_offset + t] = (1 / (n_haps * num_samples)) *
            sum_{s, k} | bin_em[s, pair_si_idx[p], k, t]
                       - bin_em[s, pair_sj_idx[p], k, t] |

    Mathematically equivalent to the numpy reference:

        diff = np.zeros((num_samples, n_bins))
        for k in range(n_haps):
            diff += np.abs(bin_em[:, si_idx, k, :] - bin_em[:, sj_idx, k, :])
        diff /= n_haps
        return np.mean(diff, axis=0)

    The kernel's outer loop ordering (k, then s, then t) matches the
    numpy reference's per-k broadcast addition into diff[s, t], so the
    final per-t accumulation differs from numpy only by 1-ULP-class
    floating-point rounding (validated via end-to-end hotspot
    equivalence).

    Args:
        bin_em: (num_samples, n_haps, n_haps, n_bins) float64 -- block's
            per-pair bin emission tensor.  n_haps here is the number of
            haplotypes in this block; pair_si_idx / pair_sj_idx index into
            that axis.
        pair_si_idx: (n_pairs,) int64 -- for each pair, the hap index in
            this block corresponding to the pair's si-side path.
        pair_sj_idx: (n_pairs,) int64 -- for each pair, the hap index in
            this block corresponding to the pair's sj-side path.
        out_mean_diffs: (n_pairs, total_bins) float64 -- preallocated by
            caller.  Written in place; only the column slice [:, out_offset
            : out_offset + n_bins] is modified.
        out_offset: int64 -- column offset (== global bin_offsets[block_idx]).
    """
    n_pairs = pair_si_idx.shape[0]
    num_samples = bin_em.shape[0]
    n_haps_dim = bin_em.shape[1]
    n_bins = bin_em.shape[3]
    inv_norm = 1.0 / (n_haps_dim * num_samples)

    for p in prange(n_pairs):
        si = pair_si_idx[p]
        sj = pair_sj_idx[p]
        # Per-pair scratch: sum-over-(s, k)-of-abs(diff) per t.  Allocated
        # inside prange so each parallel iteration has its own buffer.
        scratch = np.zeros(n_bins, dtype=np.float64)
        # Match numpy's per-k broadcast accumulation order: outer k loop,
        # inner s loop, innermost t loop (contiguous memory walk).
        for k in range(n_haps_dim):
            for s in range(num_samples):
                for t in range(n_bins):
                    d = bin_em[s, si, k, t] - bin_em[s, sj, k, t]
                    if d < 0.0:
                        d = -d
                    scratch[t] += d
        # Apply the combined 1/(n_haps * num_samples) factor when writing
        # back (uniform scale, so it commutes with the sum order).
        for t in range(n_bins):
            out_mean_diffs[p, out_offset + t] = scratch[t] * inv_norm


@njit(cache=True)
def _find_hotspots_kernel(mean_diffs_flat, bin_offsets, sample_paths,
                            K, num_samples, num_blocks,
                            ambiguity_threshold, min_samples,
                            pair_si, pair_sj,
                            out_count, out_boundary,
                            out_hap_out, out_hap_in,
                            out_left_zone, out_right_zone):
    """Hotspot detection kernel — fused numba implementation of the
    zone-extension + swap-counting algorithm used by find_hotspots.

    Iterates over every (boundary b, pair p) combination.  For each, walks
    left and right of the boundary through ambiguous-mean_diff bins to
    establish a zone, then counts how many samples switch between the
    pair (si, sj) within that zone.  Emits a hotspot record if the count
    meets min_samples.

    Mathematically identical to the equivalent Python implementation
    (1-ULP class differences in mean_diffs may shift threshold-boundary
    decisions in theory, but verified end-to-end byte-equivalent on the
    hotspot list at production scale).  The Python reference is
    preserved verbatim as a comment block inside find_hotspots() above.

    Args:
        mean_diffs_flat: (n_pairs, total_bins) float64 -- per-pair per-bin
            mean diff (output of _compute_all_pair_diffs_kernel calls).
        bin_offsets: (num_blocks + 1,) int64 -- cumulative bin offsets.
        sample_paths: (num_samples, total_bins) int32 -- per-sample
            per-bin Viterbi-decoded state index (in 0..K**2 - 1).
        K: int -- number of haplotypes (state space is K * K).
        num_samples, num_blocks: ints.
        ambiguity_threshold: float -- bins with mean_diff < threshold are
            "ambiguous" and contribute to zone extension.
        min_samples: int -- minimum swap count to emit a hotspot.
        pair_si, pair_sj: (n_pairs,) int64 -- pair indices (0 <= si < sj < K).
        out_count, out_boundary, out_hap_out, out_hap_in,
        out_left_zone, out_right_zone: preallocated int64 output buffers
            of size at least n_pairs * (num_blocks - 1).

    Returns:
        n_found: number of hotspot records written to the output buffers.
    """
    n_pairs = pair_si.shape[0]
    n_found = 0
    max_pos = sample_paths.shape[1]

    for b in range(1, num_blocks):
        boundary_bin = bin_offsets[b]

        for p in range(n_pairs):
            si = pair_si[p]
            sj = pair_sj[p]

            # ----- Left ambiguity zone -----
            # Walks blocks right-to-left from b-1.  Within each block,
            # walks bins right-to-left until hitting a non-ambiguous bin.
            # Continues to the previous block only if (a) at least one bin
            # in this block was ambiguous (extended) AND (b) the leftmost
            # bin of this block is still ambiguous (so a contiguous-zone
            # crossing is still possible).
            left_zone = 0
            for blk in range(b - 1, -1, -1):
                blk_start = bin_offsets[blk]
                blk_end = bin_offsets[blk + 1]
                blk_n_bins = blk_end - blk_start
                extended = False
                # Right-to-left bin walk
                for t in range(blk_n_bins - 1, -1, -1):
                    if mean_diffs_flat[p, blk_start + t] < ambiguity_threshold:
                        left_zone += 1
                        extended = True
                    else:
                        break
                # Decide whether to continue to the previous block
                if (not extended) or \
                   mean_diffs_flat[p, blk_start] >= ambiguity_threshold:
                    break

            # ----- Right ambiguity zone -----
            # Mirror of the left walk.  Walks blocks left-to-right from b;
            # within each block, walks bins left-to-right; continues only
            # if extended AND the rightmost bin of this block is ambiguous.
            right_zone = 0
            for blk in range(b, num_blocks):
                blk_start = bin_offsets[blk]
                blk_end = bin_offsets[blk + 1]
                blk_n_bins = blk_end - blk_start
                extended = False
                for t in range(blk_n_bins):
                    if mean_diffs_flat[p, blk_start + t] < ambiguity_threshold:
                        right_zone += 1
                        extended = True
                    else:
                        break
                if (not extended) or \
                   mean_diffs_flat[p, blk_end - 1] >= ambiguity_threshold:
                    break

            # ----- Determine scan range -----
            # Matches the original:
            #   if left_zone == 0 and right_zone == 0:
            #       zone_start = boundary_bin - 1
            #       zone_end = boundary_bin
            #   else:
            #       zone_start = boundary_bin - left_zone
            #       zone_end = max(boundary_bin, boundary_bin + right_zone - 1)
            if left_zone == 0 and right_zone == 0:
                zone_start = boundary_bin - 1
                zone_end = boundary_bin
            else:
                zone_start = boundary_bin - left_zone
                ze_candidate = boundary_bin + right_zone - 1
                if ze_candidate > boundary_bin:
                    zone_end = ze_candidate
                else:
                    zone_end = boundary_bin

            # ----- Count swapping samples within zone -----
            # Per sample, walk timesteps within zone; on the first
            # state-transition that matches the (si <-> sj) pattern,
            # increment swap_count and break to next sample.
            #
            # Set ops `{pb//K, pb%K} - {pa//K, pa%K}` unrolled to scalar
            # comparisons; preserves behaviour for homozygous-pair states
            # (where pb//K == pb%K, set is a singleton, contains-check is
            # equivalent to two scalar equality checks).
            t_start = zone_start if zone_start > 1 else 1
            t_end = zone_end + 1
            if t_end > max_pos:
                t_end = max_pos
            swap_count = 0
            if t_end > t_start:
                for s in range(num_samples):
                    for t in range(t_start, t_end):
                        pb = sample_paths[s, t - 1]
                        pa = sample_paths[s, t]
                        if pb != pa:
                            pb_a = pb // K
                            pb_b = pb % K
                            pa_a = pa // K
                            pa_b = pa % K
                            si_in_pb = (si == pb_a) or (si == pb_b)
                            si_in_pa = (si == pa_a) or (si == pa_b)
                            sj_in_pb = (sj == pb_a) or (sj == pb_b)
                            sj_in_pa = (sj == pa_a) or (sj == pa_b)
                            si_in_out = si_in_pb and (not si_in_pa)
                            sj_in_inn = sj_in_pa and (not sj_in_pb)
                            sj_in_out = sj_in_pb and (not sj_in_pa)
                            si_in_inn = si_in_pa and (not si_in_pb)
                            if (si_in_out and sj_in_inn) or \
                               (sj_in_out and si_in_inn):
                                swap_count += 1
                                break

            if swap_count >= min_samples:
                out_count[n_found] = swap_count
                out_boundary[n_found] = b
                out_hap_out[n_found] = si
                out_hap_in[n_found] = sj
                out_left_zone[n_found] = left_zone
                out_right_zone[n_found] = right_zone
                n_found += 1

    return n_found


def find_hotspots(sample_paths, K, num_blocks, sub_emissions, path_set,
                  num_samples, min_samples=5, ambiguity_threshold=1.0):
    """Find recombination hotspots between path pairs at block boundaries.
    
    Zone-based detection: extends scan range into ambiguous (low-diff) bins,
    then counts samples switching between the pair within the zone.

    Numba-accelerated reimplementation: precomputes per-(pair, bin) mean-diff
    values via _compute_all_pair_diffs_kernel (one call per block), then
    runs _find_hotspots_kernel which does zone extension + swap counting
    in a tight C-level loop.  Mathematically equivalent to the original
    pure-python implementation (max-error ULP-class on mean_diffs;
    end-to-end hotspot output validated byte-equivalent except at the
    pathological case of mean_diff values exactly equal to the threshold,
    which is a measure-zero event for floating-point data).

    The original python implementation is preserved verbatim below as
    reference -- both the inner `_compute_block_diffs` nested function and
    the triple-nested boundary/pair/zone+swap loop -- so the math intent
    is fully documented at this call site:

        hotspots = []
        bin_offsets = [0]
        for e in sub_emissions:
            bin_offsets.append(bin_offsets[-1] + e['n_bins'])

        def _compute_block_diffs(block_idx, si, sj):
            em = sub_emissions[block_idx]
            hap_keys = em['hap_keys']
            # Use the precomputed {hap_key: local_idx} dict (added in
            # compute_subblock_emissions) for O(1) lookup; falls back to
            # list.index() if the cache is missing (externally-built
            # sub_emissions).
            key_to_local = em.get('key_to_local_idx')
            if key_to_local is not None:
                si_idx = key_to_local[path_set[si][block_idx]]
                sj_idx = key_to_local[path_set[sj][block_idx]]
            else:
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
    """
    # Trivial-case fast paths.  K < 2 means no pairs; num_blocks < 2 means
    # no boundaries to scan.  Either way, no hotspots are possible -- match
    # the original by returning an empty list.
    if K < 2 or num_blocks < 2:
        return []

    # Step 1: bin offsets (cumulative).  Matches the original `bin_offsets =
    # [0]; for e in sub_emissions: bin_offsets.append(...)` pattern, but as
    # an int64 numpy array that the kernel can index.
    bin_offsets = np.zeros(num_blocks + 1, dtype=np.int64)
    for i in range(num_blocks):
        bin_offsets[i + 1] = bin_offsets[i] + sub_emissions[i]['n_bins']
    total_bins = int(bin_offsets[num_blocks])

    # Step 2: enumerate pairs (si < sj) once.
    n_pairs = K * (K - 1) // 2
    pair_si = np.empty(n_pairs, dtype=np.int64)
    pair_sj = np.empty(n_pairs, dtype=np.int64)
    idx = 0
    for si in range(K):
        for sj in range(si + 1, K):
            pair_si[idx] = si
            pair_sj[idx] = sj
            idx += 1

    # Step 3: precompute mean_diffs for every (pair, bin) via a per-block
    # kernel call.  Dict lookups (hap_keys.index) live in Python because
    # hap_keys can contain arbitrary objects; per-block kernel call then
    # vectorises the per-pair work across the prange.
    mean_diffs_flat = np.empty((n_pairs, total_bins), dtype=np.float64)
    for block_idx in range(num_blocks):
        em = sub_emissions[block_idx]
        hap_keys = em['hap_keys']
        bin_em = em['bin_emissions']
        # Build per-block per-pair hap-index arrays from the path_set view
        # of this block.  Use the precomputed {hap_key: local_idx} dict for
        # O(1) lookup (n_pairs × 2 lookups per block); falls back to
        # list.index() if the cache is missing.
        key_to_local = em.get('key_to_local_idx')
        pair_si_idx_block = np.empty(n_pairs, dtype=np.int64)
        pair_sj_idx_block = np.empty(n_pairs, dtype=np.int64)
        if key_to_local is not None:
            for p in range(n_pairs):
                pair_si_idx_block[p] = key_to_local[
                    path_set[pair_si[p]][block_idx]]
                pair_sj_idx_block[p] = key_to_local[
                    path_set[pair_sj[p]][block_idx]]
        else:
            for p in range(n_pairs):
                pair_si_idx_block[p] = hap_keys.index(
                    path_set[pair_si[p]][block_idx])
                pair_sj_idx_block[p] = hap_keys.index(
                    path_set[pair_sj[p]][block_idx])
        # bin_em is sourced from compute_subblock_emissions which produces
        # float64 contiguous tensors.  Pass-through (no copy) is the common
        # path; np.ascontiguousarray is a defensive no-op if the caller
        # already supplied contiguous float64 data.
        bin_em_c = np.ascontiguousarray(bin_em, dtype=np.float64)
        _compute_all_pair_diffs_kernel(
            bin_em_c, pair_si_idx_block, pair_sj_idx_block,
            mean_diffs_flat, int(bin_offsets[block_idx]))

    # Step 4: hotspot detection kernel.  Preallocate output buffers sized
    # to the worst case (every (pair, boundary) qualifies as a hotspot).
    max_hotspots = n_pairs * (num_blocks - 1)
    if max_hotspots < 1:
        max_hotspots = 1
    out_count = np.empty(max_hotspots, dtype=np.int64)
    out_boundary = np.empty(max_hotspots, dtype=np.int64)
    out_hap_out = np.empty(max_hotspots, dtype=np.int64)
    out_hap_in = np.empty(max_hotspots, dtype=np.int64)
    out_left_zone = np.empty(max_hotspots, dtype=np.int64)
    out_right_zone = np.empty(max_hotspots, dtype=np.int64)

    # sample_paths may arrive as int32 from _viterbi_traceback or as int64
    # from other paths; the kernel accesses scalars and uses pb // K / pb % K
    # so any integer dtype is fine.  Ensure contiguity.
    sample_paths_c = np.ascontiguousarray(sample_paths)

    n_found = _find_hotspots_kernel(
        mean_diffs_flat, bin_offsets, sample_paths_c,
        int(K), int(num_samples), int(num_blocks),
        float(ambiguity_threshold), int(min_samples),
        pair_si, pair_sj,
        out_count, out_boundary, out_hap_out, out_hap_in,
        out_left_zone, out_right_zone)

    # Step 5: convert kernel output back to the production list-of-dicts
    # format and sort by -count to match the original return value.
    hotspots = []
    for i in range(n_found):
        hotspots.append({
            'count': int(out_count[i]),
            'boundary': int(out_boundary[i]),
            'hap_out': int(out_hap_out[i]),
            'hap_in': int(out_hap_in[i]),
            'zone': (int(out_left_zone[i]), int(out_right_zone[i]))
        })

    hotspots.sort(key=lambda x: -x['count'])
    return hotspots


# =============================================================================
# STEP 5 HELPERS: Painter-Guided Escape (V10 + V11)
# =============================================================================
# When Step 4's hotspot-guided 1-for-1 splicing converges with hotspots
# remaining at a single boundary, the resolver may be stuck in a "splice
# basin" — a cyclic permutation of suffixes across paths where no 2-path
# swap improves BIC even though a coordinated k-path permutation would
# (k>=4 for the chr23 SB 129 case we verified).
#
# Step 5 escapes the basin by reading the painter's per-strand transitions
# at hotspot boundaries: each strand's switch i->j at the boundary votes
# for sigma(i)=j in the suffix-permutation we apply.  Aggregated across
# samples, this yields a vote matrix W[i, j].  We then:
#
#   1. Identify "active" paths — those with off-diagonal weight >= threshold
#      in either their row or column of W.  Inactive paths get sigma(i)=i
#      forced (they have no painter signal suggesting they should change).
#
#   2. V10: Hungarian on the active sub-matrix (after zeroing its diagonal
#      so stay-votes don't drown out switch-votes).  One BIC eval.
#
#   3. V11: if V10 doesn't help, fall back to Murty's top-N permutations
#      restricted to the active sub-matrix.  Up to step5_top_k BIC evals.
#
# Each candidate is BIC-tested; the best improving (boundary, sigma) across
# all hotspot boundaries is applied.  Iterate until no boundary improves.
#
# Verified on six synthetic basin structures (cycles of size 2, 3, 4, 5,
# two disjoint 3-cycles, and a cycle at a non-default boundary): all
# converge to the truth-clean path set in 1 iteration with <=6 BIC evals
# each.  See test_step45_active_variants.py for the verification.
# =============================================================================
def _step5_find_active_paths(W, threshold):
    """Active path: any path whose row OR column off-diagonal max is
    >= threshold.  Inactive paths are forced sigma(i)=i.

    The threshold matches min_hotspot_samples (default 5) so we use
    the same "noteworthy" criterion as find_hotspots.
    """
    K_ = W.shape[0]
    if K_ <= 1:
        return list(range(K_))
    W_off = W.copy()
    np.fill_diagonal(W_off, 0)
    active = []
    for i in range(K_):
        row_max = W_off[i, :].max()
        col_max = W_off[:, i].max()
        if row_max >= threshold or col_max >= threshold:
            active.append(i)
    return active


def _step5_solve_constrained_assignment(W, must_assign, must_not_assign):
    """Maximize sum W[i, sigma(i)] subject to:
        must_assign: dict {i: j} -- sigma(i) must equal j
        must_not_assign: set/iterable of (i, j) -- sigma(i) must not equal j

    Returns (sigma, total_weight) or (None, -inf) if infeasible.
    Used by Murty's top-N enumeration (V11).
    """
    K_ = W.shape[0]
    BIG = 1e9
    cost = -W.copy()
    for i, j in must_not_assign:
        cost[i, j] = BIG
    locked_rows = list(must_assign.keys())
    locked_cols = [must_assign[i] for i in locked_rows]
    free_rows = [i for i in range(K_) if i not in locked_rows]
    free_cols = [j for j in range(K_) if j not in locked_cols]
    if len(free_rows) != len(free_cols):
        return None, float('-inf')
    if len(free_rows) == 0:
        sigma = np.zeros(K_, dtype=np.int64)
        for i, j in must_assign.items():
            sigma[i] = j
        for i, j in must_not_assign:
            if sigma[i] == j:
                return None, float('-inf')
        total = float(sum(W[i, sigma[i]] for i in range(K_)))
        return sigma, total
    sub_cost = cost[np.ix_(free_rows, free_cols)]
    try:
        ri, ci = linear_sum_assignment(sub_cost)
    except ValueError:
        return None, float('-inf')
    sigma = np.zeros(K_, dtype=np.int64)
    for i, j in must_assign.items():
        sigma[i] = j
    for fr, fc in zip(ri, ci):
        sigma[free_rows[fr]] = free_cols[fc]
    total_cost = float(sum(cost[i, sigma[i]] for i in range(K_)))
    if total_cost > BIG / 2:
        return None, float('-inf')
    total = float(sum(W[i, sigma[i]] for i in range(K_)))
    return sigma, total


def _step5_murty_top_n(W, N):
    """Top-N permutations of {0..K-1} by descending sum W[i, sigma(i)].

    Murty's algorithm: at each pop, the partition node yields its
    optimal sigma_P; spawn children that exclude sigma_P's edges one
    by one (force prior positions to sigma_P's value, forbid current
    position from sigma_P's value).  Verified against brute force on
    3x3 and 6x6 random matrices (matches sorted order exactly).
    """
    K_ = W.shape[0]
    sigma_1, w_1 = _step5_solve_constrained_assignment(W, {}, set())
    if sigma_1 is None:
        return []
    results = [(sigma_1, w_1)]
    counter = [0]

    def make_node(must_assign, must_not_assign):
        sigma, w = _step5_solve_constrained_assignment(
            W, must_assign, must_not_assign)
        if sigma is None:
            return None
        counter[0] += 1
        return (-w, counter[0], must_assign, must_not_assign,
                tuple(int(x) for x in sigma))

    heap = []
    n0 = make_node({}, set())
    if n0 is None:
        return [(sigma_1, w_1)]
    heapq.heappush(heap, n0)
    while len(results) < N and heap:
        neg_w, _, ma, mna, sigma_t = heapq.heappop(heap)
        sigma_P = np.array(sigma_t, dtype=np.int64)
        already = any(
            tuple(int(x) for x in sigma_P) == tuple(int(x) for x in s)
            for s, _ in results)
        if not already:
            results.append((sigma_P, -neg_w))
            if len(results) >= N:
                break
        free_positions = [i for i in range(K_) if i not in ma]
        running_ma = dict(ma)
        for pos in free_positions:
            new_mna = set(mna)
            new_mna.add((pos, int(sigma_P[pos])))
            child = make_node(dict(running_ma), new_mna)
            if child is not None:
                heapq.heappush(heap, child)
            running_ma[pos] = int(sigma_P[pos])
    return results[:N]


def _step5_hungarian_active(W, threshold):
    """V10: Hungarian on the active sub-matrix only.  Inactive paths
    get sigma(i)=i forced.  Returns (sigma, active_set).

    The active sub-matrix has its diagonal zeroed so stay-votes (which
    dominate the row sums for all paths -- the painter often prefers
    to stay) don't drown out the switch-votes that actually carry
    structural information about the suffix permutation.
    """
    K_ = W.shape[0]
    active = _step5_find_active_paths(W, threshold)
    sigma = np.arange(K_, dtype=np.int64)         # identity baseline
    if len(active) == 0:
        return sigma, active
    W_active = W[np.ix_(active, active)].copy()
    np.fill_diagonal(W_active, 0)
    row_ind, col_ind = linear_sum_assignment(-W_active)
    for r, c in zip(row_ind, col_ind):
        sigma[active[r]] = active[c]
    return sigma, active


def _step5_murty_active(W, N, threshold):
    """V11: Murty top-N restricted to the active sub-matrix.  Inactive
    paths get sigma(i)=i fixed across all returned candidates.
    Returns list of (sigma, vote_sum).

    Useful when the active sub-matrix has tied or near-tied solutions
    (e.g. when degenerate suffixes — like T0 == T5 over a sub-block —
    create multiple permutations with the same vote sum).  Cost is
    bounded by N BIC evals (caller iterates over the returned list).
    """
    K_ = W.shape[0]
    active = _step5_find_active_paths(W, threshold)
    if len(active) == 0:
        return [(np.arange(K_, dtype=np.int64), 0.0)]
    W_active = W[np.ix_(active, active)].copy()
    np.fill_diagonal(W_active, 0)
    top_active = _step5_murty_top_n(W_active, N)
    results = []
    for sigma_a, w_a in top_active:
        sigma = np.arange(K_, dtype=np.int64)
        for r in range(len(active)):
            sigma[active[r]] = active[int(sigma_a[r])]
        results.append((sigma, w_a))
    return results


def _step5_build_W_at_boundary(sample_paths, boundary_bin, K):
    """Vote matrix at given absolute bin index.  W[i, j] counts strand-
    units that were on path i at bin (boundary_bin - 1) and on path j
    at bin boundary_bin.  Each sample contributes 2 strand votes (one
    per diploid strand) per boundary.
    """
    W = np.zeros((K, K), dtype=np.float64)
    n_samples_, _ = sample_paths.shape
    for s in range(n_samples_):
        pre = int(sample_paths[s, boundary_bin - 1])
        post = int(sample_paths[s, boundary_bin])
        pre_a, pre_b = pre // K, pre % K
        post_a, post_b = post // K, post % K
        W[pre_a, post_a] += 1
        W[pre_b, post_b] += 1
    return W


def _step5_apply_sigma(paths, sigma, boundary_sb):
    """Construct the candidate path set obtained by applying suffix
    permutation sigma at sub-block boundary boundary_sb:

        new_paths[i] = paths[i][:boundary_sb] + paths[sigma[i]][boundary_sb:]

    Identity sigma yields paths unchanged.
    """
    K_ = len(paths)
    return [list(paths[i][:boundary_sb]) +
            list(paths[int(sigma[i])][boundary_sb:])
            for i in range(K_)]


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def select_and_resolve(beam_results, fast_mesh, batch_blocks,
                       global_probs, global_sites,
                       # Tuning parameters (with sensible defaults)
                       max_founders=12,
                       top_n_swap=20,
                       max_cr_iterations=10,
                       step1_max_iters=20,
                       step5_max_iters=10,
                       step5_top_k=6,
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
        step1_max_iters: Maximum number of forward-selection iterations
            in Step 1 (i.e. maximum number of additional `selected.append`
            calls Step 1 can make above whatever warmstart produced).
            Default 20 matches the original hard-coded `for k in range(20)`
            cap.  Also passed as warmstart's `max_K_cap` so the two
            phases scale together when this parameter is adjusted.
        step5_max_iters: Maximum number of Step 5 painter-guided escape
            iterations.  Each iteration paints samples on the current
            path set, scans every sub-block boundary for hotspot
            structure, and proposes a suffix-permutation via V10
            (Hungarian on active sub-matrix, 1 BIC eval per boundary)
            with V11 (Murty top-K on active sub-matrix, up to step5_top_k
            BIC evals per boundary) as fallback.  The best improving
            (boundary, sigma) across all boundaries is applied and
            iteration repeats.  Default 10 — in practice convergence is
            reached in 1-2 iterations on every basin we've verified.
        step5_top_k: Number of top permutations to try via V11 (Murty)
            as the fallback when V10 (single-shot Hungarian on the
            active sub-matrix) doesn't yield improvement at a boundary.
            Higher values explore more candidates near the linear-vote
            optimum at the cost of more BIC evals (each iteration is
            bounded by step5_top_k * num_hotspot_boundaries evals).
            Default 6 sufficed for the L=3 cycle test case (which V10
            alone missed because of T5/T0 sub-block degeneracy creating
            tied permutations); harder cases may benefit from larger.
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
    sub_em = compute_subblock_emissions(batch_blocks, global_probs, global_sites, spb,
                                        num_threads=num_threads)
    total_bins = sum(e['n_bins'] for e in sub_em)
    
    # --- Map matrix: beam index -> dense hap index per block ---
    map_matrix = np.zeros((n_cands, n_blocks), dtype=int)
    for c_idx, (path, _) in enumerate(beam_results):
        for b_idx, dense_idx in enumerate(path):
            map_matrix[c_idx, b_idx] = dense_idx

    # --- Stacked emission tensor precompute ---
    # Build bin_ems_stacked = concatenated per-block bin_emissions along
    # the bins axis.  Downstream kernels (_build_template_numba and
    # _build_cand_sv_numba) consume this stacked array as a single
    # dispatch per call instead of paying n_blocks dispatches plus a
    # serial per-block pre-broadcast.
    #
    # Concatenation requires uniform leading-dim shape
    # (num_samples, n_haps, n_haps) across blocks. When sub-blocks have
    # different n_haps (e.g. a super-block with one n_haps=4 block and
    # nine n_haps=6 blocks), we pad the smaller blocks up to the
    # super-block's max n_haps with zeros. The kernels only ever index
    # bin_ems_stacked[s, t_h, h_c, ...] with t_h, h_c in [0, n_haps_b)
    # for block b — those indices come from map_matrix which is bounded
    # by each block's actual n_haps_b — so the padded slots
    # [n_haps_b : max_n_haps] are never read at runtime. Padding is
    # purely additive memory, no math change.
    #
    # Also: cache int64-contiguous map_matrix so the downstream kernels
    # don't pay an astype/ascontiguousarray per call.
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
        # `tensor` was allocated via np.zeros (C-contiguous) and only
        # slice-assigned in-place; ascontiguousarray would be a no-op.
        return tensor
    
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
    # STEP 0: Beam Warmstart (seeds Step 1's selected set)
    #
    # Before Step 1 runs, walk the beam in score order (best LL first)
    # and accept any candidate whose addition lowers BIC; after every
    # accept run a drop-loop that removes any current member whose
    # removal further lowers BIC.  The result becomes the starting
    # `selected` list for Step 1 (instead of starting from empty).
    #
    # Step 1 still runs on top: its per-set-best-add greedy can find
    # additions that warmstart's beam-order walk missed.  Warmstart's
    # acceptance criterion is "first improvement in beam order"; Step 1's
    # is "best improvement across all candidates" — different criteria,
    # so Step 1 still adds value after warmstart.  On a converged
    # warmstart, Step 1 typically breaks on the first iteration with
    # no further improvement.
    #
    # Motivation: the original Step 1, starting from empty, committed
    # to splice paths at K=1 in cases where the data has within-sub-
    # block recombinations (chr23 SB 129).  No 1-for-1 or 2-for-1 swap
    # in the outer loop could escape the splice basin afterwards (basin-
    # shape verification: requires >=4 simultaneous swaps).  Seeding
    # Step 1 with a beam-walk-derived set avoids the basin entirely.
    # See beam_warmstart_select's docstring for full details.
    # =========================================================================
    _warmstart_selected, _warmstart_bic = beam_warmstart_select(
        beam_results=beam_results,
        fast_mesh=fast_mesh,
        sub_emissions=sub_em,
        penalty=pen_sel,
        num_samples=num_samples,
        batch_cc=batch_cc,
        max_K_cap=step1_max_iters,
        num_threads=num_threads,
    )
    
    # =========================================================================
    # STEP 1: Forward Selection (sample-chunked)
    # =========================================================================
    selected = list(_warmstart_selected)
    current_best_bic = _warmstart_bic
    for k in range(step1_max_iters):
        remaining = [x for x in range(n_cands) if x not in selected]
        if not remaining:
            break
        K_next = len(selected) + 1
        n_pairs = K_next * K_next
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
        # Stack per_block_sel_haps into a (cand_pos, n_blocks) int64
        # array once per k-iteration, for downstream
        # _build_cand_sv_numba.  cand_pos for step1 is K_next - 1 (the
        # candidate lives at row/col index K_next-1 in the pair grid,
        # so there are K_next-1 "other" base positions).  When
        # K_next == 1 the basis is empty and per_block_sel_haps contains
        # (0,)-shape arrays, so the stacked array has shape
        # (0, n_blocks).  Numba accepts int64[:, :] parameters regardless
        # of runtime shape; the row/col loops (range(cand_pos) =
        # range(0)) are skipped so t_haps_stacked is never indexed at
        # runtime.
        t_haps_stacked_step1 = np.stack(
            per_block_sel_haps, axis=1
        ).astype(np.int64)

        # Split-layout precomputation for this k iteration.  In Step 1,
        # stride = K_next and cand_pos = K_next - 1, both of which change
        # per k (selected grows each iteration), so the cand-slot lookup
        # must be rebuilt each iteration.  Special case K_next == 1:
        # stride=1, cand_pos=0, n_pairs=1; the single pair is the
        # diagonal cand cell, cand_idx_for_p = [0], n_cand_pairs = 1.
        stride_s1 = K_next
        cand_pos_s1 = K_next - 1
        cand_idx_for_p_s1 = _build_cand_idx_for_p(stride_s1, cand_pos_s1)
        n_cand_pairs_s1 = 2 * cand_pos_s1 + 1
        chunk_cands_arr_s1_dtype = np.int64  # captured for use below

        for cs in range(0, len(remaining), max_chunk):
            chunk = remaining[cs:cs + max_chunk]
            n_chunk = len(chunk)
            chunk_arr = np.array(chunk)
            
            # Accumulate scores across sample chunks
            accum_scores = np.zeros(n_chunk, dtype=np.float64)
            for _s0 in range(0, num_samples, _SAMPLE_CHUNK):
                _s1 = min(_s0 + _SAMPLE_CHUNK, num_samples)
                _sn = _s1 - _s0

                # Compact per-candidate emission tensor of shape
                # (n_chunk, _sn, 2*cand_pos+1, total_bins).  Stores only
                # the cand-row + cand-col + diagonal cells (the cells
                # whose values differ across candidates); non-cand cells
                # are read directly from tmpl_slice by the Viterbi
                # kernel via the cand_idx_for_p[k] dispatch.  At
                # K_next=6 that's 11 cand pairs vs 36 total in the
                # full pair grid, a 3.3x memory reduction.  At K_next=1,
                # n_cand_pairs=1 (the diagonal only).
                cand_sv = np.empty(
                    (n_chunk, _sn, n_cand_pairs_s1, total_bins),
                    dtype=np.float64,
                )
                _tmpl_slice = template[_s0:_s1]

                # _build_cand_sv_numba writes only cand-row + cand-col +
                # diagonal emissions into the compact (n_chunk, _sn,
                # 2*cand_pos+1, total_bins) layout.  The Viterbi kernel
                # reads non-cand emissions directly from tmpl_slice via
                # the cand_idx_for_p[k] dispatch (k=-1 → template,
                # k>=0 → cand_sv[..., k, ...]).
                chunk_cands_arr_s1 = np.asarray(chunk_arr, dtype=chunk_cands_arr_s1_dtype)
                _build_cand_sv_numba(
                    cand_sv, stacked_bin_em,
                    chunk_cands_arr_s1, map_matrix_c,
                    t_haps_stacked_step1,
                    bin_offs_arr, nbs_arr,
                    _s0, _s1, cand_pos_s1)
                
                # Split-source Viterbi reads non-cand emissions from
                # _tmpl_slice (shared across candidates) and cand emissions
                # from cand_sv (per-candidate).  No ascontiguousarray wrap
                # needed — both inputs are already C-contiguous.
                partial = _batched_viterbi_score_split(
                    _tmpl_slice, cand_sv, cand_idx_for_p_s1,
                    float(pen_sel))
                accum_scores += partial
                del cand_sv
            
            for j, ci in enumerate(chunk):
                all_scores[ci] = float(accum_scores[j])
        del template
        _malloc_trim()
        best_idx = max(all_scores, key=all_scores.get)
        new_bic = ((len(selected) + 1) * batch_cc) - (2 * all_scores[best_idx])
        if new_bic < current_best_bic:
            current_best_bic = new_bic
            selected.append(best_idx)
        else:
            break
    
    # Fix: if Step 1 selected nothing (all candidates scored -inf or failed
    # the BIC comparison on the first iteration), fall back to the single
    # top-scoring beam result.  This avoids cascading an empty `selected`
    # through Steps 2-5, which produces n_paths=0 going into paint_samples_
    # viterbi and crashes the numba kernel on a (n_samples, 0, n_bins) tensor.
    # The fallback preserves correctness: we still return SOMETHING that
    # downstream reconstruct_haplotypes_from_beam can process, even if the
    # selection machinery numerically failed for this batch.
    if not selected:
        # beam_results is sorted best-first coming out of run_full_mesh_beam_search.
        # Pick index 0 (the top-scoring path) as our single selected founder.
        if n_cands > 0:
            selected = [0]
        else:
            # Truly degenerate: no beam results at all.  Return empty list
            # immediately; _process_single_batch already handles this path
            # (it ends up with status='reconstruction_failed' rather than
            # crashing).
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
            K = len(selected)
            if K < 2:
                return None
            n_pairs = K * K
            unselected = [x for x in range(n_cands) if x not in selected]
            if not unselected:
                return None
        
            K_base = K - 1
            cand_pos = K - 1
            _sc = min(num_samples, _SAMPLE_CHUNK)
            sc = max(4, min(64, int(5e8 / (_sc * n_pairs * total_bins * 8))))
            best_swap = None
            best_gain = 0.0

            # Sample chunk starts don't depend on the swap position i,
            # so build the list once outside the K-position loop instead
            # of rebuilding it K times.
            _sample_starts = list(range(0, num_samples, _SAMPLE_CHUNK))

            # Precompute the pair-index → cand-slot lookup for the
            # split-layout Viterbi (_batched_viterbi_score_split).  For
            # 1for1, stride = K and cand_pos = K - 1; this is constant for
            # the entire round, so compute once here rather than per swap
            # position or per candidate chunk.  Length n_pairs = K*K = 36
            # at production shapes — tiny array.
            cand_idx_for_p = _build_cand_idx_for_p(K, cand_pos)
            n_cand_pairs = 2 * cand_pos + 1

            # Process each swap position independently — only 1 template alive at a time
            for i in range(K):
                temp_set = selected[:i] + selected[i + 1:]
                cands = get_candidates_fn(temp_set, unselected)
                if not cands:
                    continue
            
                # Store hap indices for this position (tiny — just integer arrays)
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
            
                # Precompute the full-sample template once per position iter
                # via the parallel numba kernel _build_template_numba.  The
                # template depends on (i, samples) — once built we can slice
                # zero-copy per sample chunk inside the candidate loop.
                #
                # The kernel writes via direct indexing in a single prange'd
                # region (one dispatch per round), avoiding the Python-level
                # K * n_sample_chunks * n_blocks * K_base² slice-copy loop
                # that the equivalent numpy implementation would run.
                # Cand cells in tmpl_full are deliberately left uninitialised
                # by the kernel; the Viterbi recurrence never reads them
                # because cand_idx_for_p[k] dispatches those k values to
                # cand_sv (built separately by _build_cand_sv_numba).
                tmpl_full = np.empty((num_samples, n_pairs, total_bins),
                                     dtype=np.float64)
                _build_template_numba(tmpl_full, stacked_bin_em,
                                      t_haps_stacked, bin_offs_arr, nbs_arr,
                                      K_base, K)
            
                # Process candidates in chunks
                for cs in range(0, len(cands), sc):
                    chunk_cands = cands[cs:cs + sc]
                    n_chunk = len(chunk_cands)
                
                    # Accumulate scores across sample chunks
                    accum_scores = np.zeros(n_chunk, dtype=np.float64)
                    for _s0_idx, _s0 in enumerate(_sample_starts):
                        _s1 = min(_s0 + _SAMPLE_CHUNK, num_samples)
                        _sn = _s1 - _s0
                        # Zero-copy slice into the per-position full
                        # template built once above by _build_template_numba.
                        # tmpl_full is C-contiguous; slicing the leading
                        # axis preserves contiguity, which numba prefers.
                        tmpl_slice = tmpl_full[_s0:_s1]

                        # Allocate the compact per-candidate emission
                        # tensor — only cand-row + cand-col + diagonal
                        # cells (2*cand_pos+1 entries on the pair axis,
                        # vs n_pairs = stride² for the full pair grid).
                        # At K=6 1for1: 11 vs 36 ≈ 3.3x smaller; non-cand
                        # emissions come from tmpl_slice, which is
                        # L3-cache-resident and shared across all n_chunk
                        # batches inside _batched_viterbi_score_split.
                        cand_sv = np.empty(
                            (n_chunk, _sn, n_cand_pairs, total_bins),
                            dtype=np.float64,
                        )
                    
                        nt = _resolve_threads(num_threads)
                        # _build_cand_sv_numba writes ONLY the cand-row +
                        # cand-col + diagonal emissions into cand_sv.  No
                        # full-pair-grid broadcast — the Viterbi kernel
                        # reads non-cand pairs directly from tmpl_slice
                        # via the cand_idx_for_p[k] dispatch.
                        chunk_cands_arr = np.asarray(chunk_cands, dtype=np.int64)
                        _build_cand_sv_numba(
                            cand_sv, stacked_bin_em,
                            chunk_cands_arr, map_matrix_c,
                            t_haps_stacked,
                            bin_offs_arr, nbs_arr,
                            _s0, _s1, cand_pos)
                    
                        # Split-source Viterbi: reads non-cand emissions
                        # from tmpl_slice (shared) and cand emissions from
                        # cand_sv (per-batch).  Same recurrence as the
                        # single-source _batched_viterbi_score.
                        partial = _batched_viterbi_score_split(
                            tmpl_slice, cand_sv, cand_idx_for_p,
                            float(pen_sel))
                        accum_scores += partial
                        # tmpl_slice is a view into tmpl_full (built once
                        # above by _build_template_numba). Free cand_sv
                        # here; tmpl_full is freed after the chunks loop.
                        del cand_sv
                
                    for j, ci in enumerate(chunk_cands):
                        gain = float(accum_scores[j]) - cur_score
                        if gain > 1e-4 and gain > best_gain:
                            best_gain = gain
                            best_swap = (selected[i], ci)
                # Free the per-position template tensor before moving to next i.
                del tmpl_full
        
            _malloc_trim()
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
            K = len(selected)
            if K < 3:
                return None
        
            current_bic = K * batch_cc - 2 * cur_score
            K_result = K - 1
            n_pairs_r = K_result * K_result
            unselected = [x for x in range(n_cands) if x not in selected]
            if not unselected:
                return None
        
            K_base = K - 2
            cand_pos = K_result - 1
            _sc = min(num_samples, _SAMPLE_CHUNK)
            sc = max(4, min(64, int(5e8 / (_sc * n_pairs_r * total_bins * 8))))
            best_2for1 = None
            best_bic = current_bic

            # Sample chunk starts don't depend on (i, j), so build once
            # outside the pair loop instead of K*(K-1)/2 times.
            _sample_starts = list(range(0, num_samples, _SAMPLE_CHUNK))

            # Precompute the pair-index → cand-slot lookup for the
            # split-layout Viterbi (_batched_viterbi_score_split).  For
            # 2for1, stride = K_result and cand_pos = K_result - 1; this
            # is constant for the entire round, so compute once here
            # rather than per pair.  Length n_pairs_r = K_result² (e.g. 25
            # at K=6 2for1).
            cand_idx_for_p = _build_cand_idx_for_p(K_result, cand_pos)
            n_cand_pairs = 2 * cand_pos + 1

            # Process each pair independently — only 1 template slice alive at a time
            # (avoids C(K,2) × full-sample templates = 22 GB for K=12)
            for i in range(K):
                for j in range(i + 1, K):
                    temp_set = [selected[k] for k in range(K) if k != i and k != j]
                
                    # Get candidates for this pair via cheap scoring
                    bm = precompute_base_max(temp_set)
                    cs_scores = cheap_score_all(bm, temp_set, unselected)
                    ranked = sorted(cs_scores, key=cs_scores.get, reverse=True)[:top_n_swap]
                    if not ranked:
                        continue
                
                    # Store hap indices for this pair (tiny)
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
                
                    # Precompute the full-sample template once per pair iter
                    # via the parallel numba kernel _build_template_numba.
                    # Same shape/semantics as in _run_1for1_round, with the
                    # only difference being stride = K_result instead of K
                    # (the pair layout is (K_result, K_result) for 2for1).
                    # Cand cells are left uninitialised — see kernel docstring.
                    tmpl_full = np.empty((num_samples, n_pairs_r, total_bins),
                                         dtype=np.float64)
                    _build_template_numba(tmpl_full, stacked_bin_em,
                                          t_haps_stacked, bin_offs_arr, nbs_arr,
                                          K_base, K_result)
                
                    # Process candidates in chunks
                    for cs_start in range(0, len(ranked), sc):
                        chunk_cands = ranked[cs_start:cs_start + sc]
                        n_chunk = len(chunk_cands)
                    
                        # Accumulate scores across sample chunks
                        accum_scores = np.zeros(n_chunk, dtype=np.float64)
                        for _s0_idx, _s0 in enumerate(_sample_starts):
                            _s1 = min(_s0 + _SAMPLE_CHUNK, num_samples)
                            _sn = _s1 - _s0
                            # Zero-copy slice into the per-pair full template.
                            tmpl_slice = tmpl_full[_s0:_s1]

                            # Compact per-candidate emission tensor —
                            # cand-row + cand-col + diagonal cells only.
                            # See _run_1for1_round above for the full
                            # rationale.  The 2for1 stride is K_result
                            # instead of K, but cand_pos is still equal
                            # to stride - 1 so the kernel works unchanged.
                            cand_sv = np.empty(
                                (n_chunk, _sn, n_cand_pairs, total_bins),
                                dtype=np.float64,
                            )
                        
                            nt = _resolve_threads(num_threads)
                            # _build_cand_sv_numba writes only the cand-row,
                            # cand-col, and diagonal emissions — Phase 1
                            # template broadcast is eliminated entirely.
                            chunk_cands_arr = np.asarray(chunk_cands, dtype=np.int64)
                            _build_cand_sv_numba(
                                cand_sv, stacked_bin_em,
                                chunk_cands_arr, map_matrix_c,
                                t_haps_stacked,
                                bin_offs_arr, nbs_arr,
                                _s0, _s1, cand_pos)
                        
                            # Split-source Viterbi — reads non-cand emissions
                            # from tmpl_slice (shared) and cand emissions
                            # from cand_sv (per-batch).
                            partial = _batched_viterbi_score_split(
                                tmpl_slice, cand_sv, cand_idx_for_p,
                                float(pen_sel))
                            accum_scores += partial
                            # tmpl_slice is a view into tmpl_full (built
                            # once above by _build_template_numba). Free
                            # cand_sv here; tmpl_full is freed after the
                            # chunks loop.
                            del cand_sv
                    
                        for j_idx, ci in enumerate(chunk_cands):
                            new_score = float(accum_scores[j_idx])
                            new_bic = K_result * batch_cc - 2 * new_score
                            if new_bic < best_bic - 1e-4:
                                best_bic = new_bic
                                best_2for1 = (selected[i], selected[j], ci)
                    # Free the per-pair template tensor before moving to next (i, j).
                    del tmpl_full
        
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
          _iter_step2 = 0
          while True:  # Outer loop: Phase A → Phase B → Phase C → repeat
            _iter_step2 += 1
        
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
    
        # Convert beam indices (possibly including splice-extension
        # indices from prior outer-loop iterations) to key-paths via
        # map_matrix.  map_matrix[bi] is byte-identical to
        # beam_results[bi][0] for original beam rows (populated in the
        # shared setup) and holds splice dense paths for extension rows
        # added at end-of-iteration conversions.
        paths = []
        for bi in selected:
            path = map_matrix[bi]
            keys = [fast_mesh.reverse_mappings[b][int(d)]
                    for b, d in enumerate(path)]
            paths.append(keys)
    
        # =========================================================================
        # STEP 4: Chimera Resolution
        # =========================================================================
        current_paths = list(paths)
        for iteration in range(max_cr_iterations):
            # 4a. Paint samples
            sp, K_cr = paint_samples_viterbi(
                current_paths, sub_em, paint_penalty, num_samples,
                num_threads=num_threads)
        
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

        # Early-break if this iteration didn't change the state.
        if frozenset(selected) == _state_before:
            break
    else:
        # Exited because MAX_OUTER_ITERATIONS reached (loop didn't break)
        pass

    # Rebuild current_paths from the final selected one more time so Steps
    # 5 and 6 see a consistent key-path list (map_matrix rows cover both
    # original beam entries and any splice-extension indices added during
    # the loop).
    current_paths = [
        [fast_mesh.reverse_mappings[b][int(d)]
         for b, d in enumerate(map_matrix[_bi])]
        for _bi in selected
    ]

    # =========================================================================
    # STEP 5: Painter-Guided Escape (V10 + V11)
    #
    # Step 4's hotspot-guided 1-for-1 splicing (the 7-option machinery) can
    # converge with hotspots remaining — a "splice basin" of cyclic suffix
    # permutations across paths that no 2-path swap can escape (k>=4
    # simultaneous changes required for the chr23 SB 129 case we verified).
    # Step 5 escapes such basins using only the input path set.
    #
    # Mechanism: paint samples; for each sub-block boundary, build a vote
    # matrix W[i,j] from per-strand transitions; identify "active" paths
    # (off-diagonal weight >= min_hotspot_samples in row or column);
    # propose sigma via Hungarian on the active sub-matrix (V10) with
    # Murty top-K fallback (V11).  Each candidate is BIC-tested; the best
    # improving (boundary, sigma) across all hotspot boundaries is applied.
    # Iterate until no boundary improves.
    #
    # Math identical to existing steps: BIC = K * batch_cc - 2 * LL via
    # score_path_set; pen_sel and num_threads semantics identical.  Inactive
    # paths get sigma(i)=i forced, so the search space is bounded by the
    # active-set size — never K! enumeration even for large K.
    # =========================================================================

    # Need at least 2 paths to permute meaningfully (and score_path_set
    # would segfault on K=0 — _build_tensor_from_paths produces a
    # (num_samples, 0, total_bins) tensor and viterbi_score_selection
    # is undefined for n_pairs=0).  K=1 is a no-op via the threshold
    # check (W is 1x1 with no off-diagonal mass) but we skip it explicitly
    # to avoid one wasted painting + score_path_set call per empty case.
    if len(current_paths) < 2:
        pass
    else:
        # Bin offsets per sub-block (translates sub-block index to bin index)
        _step5_bin_offsets = [0]
        for _step5_e in sub_em:
            _step5_bin_offsets.append(
                _step5_bin_offsets[-1] + _step5_e['n_bins'])

        for _step5_iter in range(step5_max_iters):
            K_s5 = len(current_paths)

            # Paint samples on current state
            _step5_sample_paths, _step5_K_check = paint_samples_viterbi(
                current_paths, sub_em, pen_sel, num_samples,
                num_threads=num_threads)

            # Current BIC
            _step5_cur_ll = score_path_set(
                current_paths, sub_em, pen_sel, num_samples,
                num_threads=num_threads)
            _step5_cur_bic = K_s5 * batch_cc - 2 * _step5_cur_ll

            _step5_best_bic = _step5_cur_bic
            _step5_best_paths = current_paths
            _step5_best_method = None
            _step5_best_boundary = None
            _step5_best_sigma = None
            _step5_best_active = None
            _step5_n_evals = 0

            # Scan every internal sub-block boundary.  sb_idx 1..N_SUB-1 are
            # the boundaries between adjacent sub-blocks; sb_idx=0 is the start
            # of the region (no boundary), sb_idx=N_SUB is the end.
            for _step5_sb in range(1, len(_step5_bin_offsets)):
                _step5_b_bin = _step5_bin_offsets[_step5_sb]
                if _step5_b_bin == 0 or \
                        _step5_b_bin >= _step5_bin_offsets[-1]:
                    continue
                _step5_W = _step5_build_W_at_boundary(
                    _step5_sample_paths, _step5_b_bin, K_s5)
                _step5_n_switches = float(
                    _step5_W.sum() - np.diag(_step5_W).sum())
                # Skip boundaries with too few strand-switches (no signal)
                if _step5_n_switches < min_hotspot_samples:
                    continue

                # ---- V10: Hungarian on active sub-matrix ----
                _step5_sigma_v10, _step5_active = _step5_hungarian_active(
                    _step5_W, min_hotspot_samples)
                if len(_step5_active) == 0:
                    continue
                if not np.array_equal(_step5_sigma_v10, np.arange(K_s5)):
                    _step5_new_paths = _step5_apply_sigma(
                        current_paths, _step5_sigma_v10, _step5_sb)
                    _step5_new_ll = score_path_set(
                        _step5_new_paths, sub_em, pen_sel, num_samples,
                        num_threads=num_threads)
                    _step5_new_bic = K_s5 * batch_cc - 2 * _step5_new_ll
                    _step5_n_evals += 1
                    if _step5_new_bic < _step5_best_bic:
                        _step5_best_bic = _step5_new_bic
                        _step5_best_paths = _step5_new_paths
                        _step5_best_method = "V10"
                        _step5_best_boundary = _step5_sb
                        _step5_best_sigma = _step5_sigma_v10.copy()
                        _step5_best_active = list(_step5_active)

                # ---- V11: Murty top-K on active sub-matrix (fallback) ----
                # Murty's first candidate is V10's sigma; we skip it to avoid
                # double-evaluation.  Up to (step5_top_k - 1) further evals
                # per boundary.
                _step5_v11_candidates = _step5_murty_active(
                    _step5_W, step5_top_k, min_hotspot_samples)
                for _step5_sig_v11, _ in _step5_v11_candidates:
                    if np.array_equal(_step5_sig_v11, np.arange(K_s5)):
                        continue
                    if np.array_equal(_step5_sig_v11, _step5_sigma_v10):
                        continue              # already V10-evaluated
                    _step5_new_paths = _step5_apply_sigma(
                        current_paths, _step5_sig_v11, _step5_sb)
                    _step5_new_ll = score_path_set(
                        _step5_new_paths, sub_em, pen_sel, num_samples,
                        num_threads=num_threads)
                    _step5_new_bic = K_s5 * batch_cc - 2 * _step5_new_ll
                    _step5_n_evals += 1
                    if _step5_new_bic < _step5_best_bic:
                        _step5_best_bic = _step5_new_bic
                        _step5_best_paths = _step5_new_paths
                        _step5_best_method = "V11"
                        _step5_best_boundary = _step5_sb
                        _step5_best_sigma = _step5_sig_v11.copy()
                        _step5_best_active = list(_step5_active)

            if _step5_best_method is None:
                break

            _step5_sigma_str = ','.join(
                str(int(s)) for s in _step5_best_sigma)
            current_paths = _step5_best_paths
        else:
            # Loop didn't break — hit step5_max_iters
            pass


    # =========================================================================
    # STEP 6: Convert resolved key-paths back to beam format
    # =========================================================================
    resolved_beam = []
    for path_keys in current_paths:
        dense_path = [fast_mesh.reverse_mappings[b].index(key)
                      for b, key in enumerate(path_keys)]
        resolved_beam.append((dense_path, 0.0))
    
    return resolved_beam