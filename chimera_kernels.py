"""
Chimera Resolution — Numba JIT Kernels

Low-level numba-compiled numeric primitives extracted verbatim from
chimera_resolution.py to keep that file manageable.  Contents: batched and
split Viterbi scoring, scoring-tensor construction (serial and parallel-over-
samples builders), Viterbi traceback, bin-emission computation and single-hap
update, dosage rounding, sample gather, partner-carrier and consensus kernels,
distance-to-H and template builders, cheap-score precompute, and the JIT
warmup helper.

These are pure numeric kernels (no module-level scoring state).  They are
re-imported into chimera_resolution so that existing
`chimera_resolution.<kernel>` references — both external callers and the
orchestration code in that module — keep resolving unchanged.
"""

import numpy as np
import math
import ctypes

from numba import njit, prange


# -----------------------------------------------------------------------------
# Shared low-level utilities (thread resolution + memory trim) used by both the
# scoring machinery (chimera_scoring) and the orchestrator (chimera_resolution).
# Kept here, in the leaf module, so both can import them without a cycle.
# -----------------------------------------------------------------------------
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

    (Single-Python-thread callers — score_path_set, paint_samples_viterbi
    — instead use the prange variant _build_tensor_block_numba_par, where
    there is no outer pool to contend with and the per-call build is the
    dominant cost.)

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


@njit(parallel=True, fastmath=False)
def _build_tensor_block_numba_par(tensor, bin_em, local_indices,
                                  s0, s1, K, n_bins_b, bin_offset):
    """Parallel-over-samples variant of _build_tensor_block_numba.

    Identical writes (and therefore identical values) to the serial
    kernel, but the outer sample loop is a `prange`, so a single
    Python-thread caller gets the full numba OpenMP thread pool for the
    build.  Each iteration writes a disjoint `tensor[s, :, :]` row, so
    there is no write race.

    Used by the SINGLE-Python-thread scorers (`score_path_set`,
    `paint_samples_viterbi`) reached from the sequential warmstart walk
    and CR Steps 1-5, where there is no competing `ThreadPoolExecutor`
    and the per-call build (a ~250 MB tensor write at production K) is
    otherwise serial and dominates the call.  Parallelising it over
    samples is what actually keeps the cores busy in those loops.

    The serial `_build_tensor_block_numba` is still used inside
    `score_path_sets_parallel`, whose own `ThreadPoolExecutor` owns the
    parallelism — a `prange` there would make every builder thread
    contend for the same OMP pool (~2.4x slowdown; see that kernel's
    docstring).  This kernel and that one are deliberately the two halves
    of that split: prange for single-threaded callers, serial+nogil for
    the pooled caller.
    """
    n_s = s1 - s0
    n_pairs = K * K
    for s in prange(n_s):
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
        
        # The diploid emission at a site depends only on the dosage (h1 allele +
        # h2 allele = 0, 1 or 2), so there are just THREE possible log-emissions
        # per site -- one per dosage.  Compute them once per site and index by
        # dosage in the pair loops instead of recomputing model+log for all
        # n_haps^2 pairs (for n_haps=6 that is 3 logs/site rather than 36).  The
        # dosage selects exactly the s0/s1/s2 that the one-hot products
        # c00/c01/c11 selected before, and each bin still accumulates its sites in
        # the same (ascending) order, so the result is bit-for-bit identical.
        for site in range(n_sites):
            f0 = s0[site] * 0.99 + 0.01 / 3.0
            if f0 < 1e-300:
                f0 = 1e-300
            l0 = math.log(f0)
            if l0 < -2.0:
                l0 = -2.0

            f1 = s1[site] * 0.99 + 0.01 / 3.0
            if f1 < 1e-300:
                f1 = 1e-300
            l1 = math.log(f1)
            if l1 < -2.0:
                l1 = -2.0

            f2 = s2[site] * 0.99 + 0.01 / 3.0
            if f2 < 1e-300:
                f2 = 1e-300
            l2 = math.log(f2)
            if l2 < -2.0:
                l2 = -2.0

            b = site // snps_per_bin
            for h1_idx in range(n_haps):
                a1 = hap1[h1_idx][site]
                for h2_idx in range(n_haps):
                    dosage = a1 + hap1[h2_idx][site]
                    if dosage < 0.5:
                        ll = l0
                    elif dosage < 1.5:
                        ll = l1
                    else:
                        ll = l2
                    bin_emissions[s, h1_idx, h2_idx, b] += ll
    
    return bin_emissions


@njit(parallel=True, fastmath=True)
def _update_bin_emissions_one_hap_numba(bin_emissions, block_samples, hap0, hap1,
                                        estar, n_haps, n_bins, snps_per_bin, n_sites):
    """In-place update of `bin_emissions` for the (estar, *) ROW and (*, estar)
    COLUMN of the (n_haps, n_haps) hap-pair grid, after hap `estar` has been
    replaced (hap0[estar] / hap1[estar] hold the NEW hap's one-hot alleles).

    The per-site arithmetic is identical to _compute_bin_emissions_numba (the
    same three per-site dosage log-emissions l0/l1/l2), and every emission cell
    depends ONLY on the
    two haps of its pair.  A single-hap replacement therefore changes EXACTLY the
    estar row and estar column; all other cells are unchanged.  So overwriting
    just those cells here yields a tensor bit-identical to a full
    _compute_bin_emissions_numba recompute on the modified hap matrix — at
    O(n_haps) pair-cost instead of O(n_haps^2).  Parallelises over samples (each
    thread owns its own s-slice, so the result is deterministic regardless of
    thread count); the thread count is controlled by the caller via
    numba.set_num_threads, exactly like the full kernel.

    The diagonal (estar, estar) is written once in the ROW loop; the COLUMN loop
    skips h1_idx == estar so it is not double-accumulated.
    """
    num_samples = block_samples.shape[0]
    e0 = hap0[estar]
    e1 = hap1[estar]

    for s in prange(num_samples):
        s0 = block_samples[s, :, 0]
        s1 = block_samples[s, :, 1]
        s2 = block_samples[s, :, 2]

        # clear the estar row & column for this sample before re-accumulating
        for h in range(n_haps):
            for b in range(n_bins):
                bin_emissions[s, estar, h, b] = 0.0
                bin_emissions[s, h, estar, b] = 0.0

        # Same per-site dosage dedup as the full kernel: three log-emissions per
        # site (one per dosage 0/1/2), indexed by each pair's dosage, rather than
        # recomputing model+log for all 2*n_haps-1 affected pairs.  Re-accumulate
        # the estar ROW (estar, h2) and COLUMN (h1, estar); each bin sums its sites
        # in ascending order exactly as a full recompute would, so the updated
        # cells are bit-identical.  The diagonal (estar, estar) is written in the
        # ROW loop; the COLUMN loop skips h1_idx == estar.
        for site in range(n_sites):
            f0 = s0[site] * 0.99 + 0.01 / 3.0
            if f0 < 1e-300:
                f0 = 1e-300
            l0 = math.log(f0)
            if l0 < -2.0:
                l0 = -2.0

            f1 = s1[site] * 0.99 + 0.01 / 3.0
            if f1 < 1e-300:
                f1 = 1e-300
            l1 = math.log(f1)
            if l1 < -2.0:
                l1 = -2.0

            f2 = s2[site] * 0.99 + 0.01 / 3.0
            if f2 < 1e-300:
                f2 = 1e-300
            l2 = math.log(f2)
            if l2 < -2.0:
                l2 = -2.0

            b = site // snps_per_bin
            ea = e1[site]

            # ROW: pairs (estar, h2_idx) for every h2 (includes (estar, estar))
            for h2_idx in range(n_haps):
                dosage = ea + hap1[h2_idx][site]
                if dosage < 0.5:
                    ll = l0
                elif dosage < 1.5:
                    ll = l1
                else:
                    ll = l2
                bin_emissions[s, estar, h2_idx, b] += ll

            # COLUMN: pairs (h1_idx, estar) for every h1 except estar (done above)
            for h1_idx in range(n_haps):
                if h1_idx == estar:
                    continue
                dosage = hap1[h1_idx][site] + ea
                if dosage < 0.5:
                    ll = l0
                elif dosage < 1.5:
                    ll = l1
                else:
                    ll = l2
                bin_emissions[s, h1_idx, estar, b] += ll

    return bin_emissions


@njit(parallel=True, fastmath=True)
def _dosage_round_numba(gp, idx_g):
    """Rounded per-sample dosage over a window, parallelised over samples.

    For each sample s and window position t (idx_g[t] = the site's index in the
    global probs array):

        d_r[s, t] = clip(rint(gp[s, idx_g[t], 1] + 2 * gp[s, idx_g[t], 2]), 0, 2)

    returned as int16.  This is the numba equivalent of the numpy expression

        d   = gp[:, idx_g, 1] * 1.0 + gp[:, idx_g, 2] * 2.0
        d_r = np.clip(np.rint(d), 0, 2).astype(np.int16)

    and is bit-identical to it: the dosage is formed in float64 exactly as above,
    Python/numba round() and np.rint() both round half to even, and the clip maps
    into {0, 1, 2}.  Computing it here avoids the (N, L) float64 gather + rint +
    clip + astype temporaries AND the subsequent ascontiguousarray copy of the
    numpy path (the output is written contiguously), and spreads the work over the
    inner-core pool instead of running single-threaded in numpy.
    """
    N = gp.shape[0]
    L = idx_g.shape[0]
    out = np.empty((N, L), dtype=np.int16)
    for s in prange(N):
        for t in range(L):
            g = idx_g[t]
            d = gp[s, g, 1] * 1.0 + gp[s, g, 2] * 2.0
            r = round(d)
            if r < 0.0:
                r = 0.0
            elif r > 2.0:
                r = 2.0
            out[s, t] = np.int16(r)
    return out


@njit(parallel=True, fastmath=True)
def _gather_samples_numba(probs, indices):
    """Parallel equivalent of np.ascontiguousarray(probs[:, indices, :]).

    probs: (N, n_global, 3); indices: (n_sites,) integer positions into axis 1.
    Returns a C-contiguous (N, n_sites, 3) array with
    out[s, t, c] = probs[s, indices[t], c].

    This is a pure copy (no arithmetic), so it is bit-identical to numpy's
    advanced-index gather.  numpy runs that gather single-threaded, and at L4 the
    probs slice is ~5.4 GB, so the fancy-index copy costs ~10-20s on one core;
    spreading the copy over samples lets the inner-core pool absorb it.  The
    output layout (N, n_sites, 3) matches numpy's, so block_samples[s, :, c] has
    the same stride the emission kernels already expect.
    """
    N = probs.shape[0]
    n_sites = indices.shape[0]
    out = np.empty((N, n_sites, 3), dtype=probs.dtype)
    for s in prange(N):
        for t in range(n_sites):
            g = indices[t]
            out[s, t, 0] = probs[s, g, 0]
            out[s, t, 1] = probs[s, g, 1]
            out[s, t, 2] = probs[s, g, 2]
    return out


@njit(parallel=True, fastmath=True)
def _viterbi_partner_carriers_numba(d_r, carrier_idx, cur, N_tilde, bof, nbins,
                                    log_invalid, near_bonus, switch_pen):
    """Per-carrier partner-haplotype Viterbi for the refinement carrier
    re-derivation, PARALLELISED over carriers (prange — carriers are independent).

    For each carrier c this builds its (nbins, m) binned stay-score matrix E_c:
        E_c[b, k] = sum over the sites of bin b of the per-site score, where the
        per-site score is `log_invalid` when the carrier's dosage minus hap k is
        outside {0, 1}, else `near_bonus` when the extracted allele equals
        N_tilde, else 0.0.
    Then it runs the stay/switch Viterbi over bins and writes the per-site partner
    index path[bof[site]] into partner_site[c].

    This is the parallel counterpart of the serial numpy code it replaces —
    the `for k: E[:, :, k] = Lsite @ Bmat` bin-emission build together with
    `for c: partner_site[c] = _viterbi_partner(E[c], switch_pen)[bof]`.  It is
    mathematically equivalent: the per-bin sums differ from the BLAS matmul only
    by floating-point summation order (a few ULP), and the argmax / second-argmax
    feeding the switch back-pointer use a first-occurrence tie-break (numpy's
    argsort uses an unspecified one), which only ever changes the chosen partner
    among haps that score EXACTLY equal — i.e. among equally-optimal partners.
    Each carrier owns its own E_c / V / back / path scratch, so the result is
    independent of the thread count; the count is set by the caller via
    numba.set_num_threads (so a worker uses total_cores // active_workers here too).

    Takes the full per-sample dosage `d_r` (N, L) plus the carrier sample indices
    `carrier_idx` and reads carrier rows as views (d_r[carrier_idx[c]]) instead of
    a pre-gathered (C, L) `d_rC` -- avoids materialising that per-rep gather copy.
    """
    nC = carrier_idx.shape[0]
    L = d_r.shape[1]
    m = cur.shape[0]
    partner_site = np.empty((nC, L), dtype=np.int64)

    for c in prange(nC):
        d_r_c = d_r[carrier_idx[c]]                # carrier dosage row (view; no copy)
        # --- build E_c (nbins, m): binned per-site stay-scores for each hap ---
        E_c = np.zeros((nbins, m), dtype=np.float64)
        for k in range(m):
            for site in range(L):
                diff = d_r_c[site] - cur[k, site]
                if diff < 0 or diff > 1:
                    Lsite = log_invalid
                elif diff == N_tilde[site]:
                    Lsite = near_bonus
                else:
                    Lsite = 0.0
                E_c[bof[site], k] += Lsite

        # --- stay/switch Viterbi over bins (matches _viterbi_partner) ---
        V = np.empty((nbins, m), dtype=np.float64)
        back = np.zeros((nbins, m), dtype=np.int64)
        for k in range(m):
            V[0, k] = E_c[0, k]
        for b in range(1, nbins):
            # am = argmax(V[b-1]); sec = argmax over the rest (first-occurrence ties)
            am = 0
            for k in range(1, m):
                if V[b - 1, k] > V[b - 1, am]:
                    am = k
            sec = -1
            for k in range(m):
                if k == am:
                    continue
                if sec < 0 or V[b - 1, k] > V[b - 1, sec]:
                    sec = k
            m1 = V[b - 1, am]
            m2 = V[b - 1, sec]
            for j in range(m):
                if j == am:
                    best_other = m2
                    sw_src = sec
                else:
                    best_other = m1
                    sw_src = am
                stay = V[b - 1, j]
                switch_val = best_other - switch_pen
                if stay >= switch_val:
                    V[b, j] = E_c[b, j] + stay
                    back[b, j] = j
                else:
                    V[b, j] = E_c[b, j] + switch_val
                    back[b, j] = sw_src

        # --- backtrack: path[-1] = argmax(V[-1]); path[b-1] = back[b, path[b]] ---
        path = np.empty(nbins, dtype=np.int64)
        last = 0
        for k in range(1, m):
            if V[nbins - 1, k] > V[nbins - 1, last]:
                last = k
        path[nbins - 1] = last
        for b in range(nbins - 1, 0, -1):
            path[b - 1] = back[b, path[b]]

        # --- map the bin-path to per-site partner indices ---
        for site in range(L):
            partner_site[c, site] = path[bof[site]]

    return partner_site


@njit(parallel=True, fastmath=True)
def _consensus_from_carriers_numba(d_r, carrier_idx, cur, partner_site, N_tilde):
    """Per-site consensus of the carrier partner-painting, PARALLELISED over
    SITES (prange).  For each site it sums, over carriers, the partner-dosage
    support and returns the new hap H plus frac1 / has / win_frac.

    This replaces the serial numpy block
        present_by_partner = take_along_axis(cur, partner_site, 0)
        Fdiff = d_rC - present_by_partner ; Fest = clip(Fdiff, 0, 1)
        contribute = (Fdiff == 0) | (Fdiff == 1) ; mask = contribute.astype(f8)
        denom = mask.sum(0) ; num = (Fest * mask).sum(0)
        H = N_tilde; has = denom > 0; frac1[has] = num[has]/denom[has]
        H[has] = frac1[has] >= 0.5 ; win_frac = where(H==1, frac1, 1-frac1)
    which materialises several (C, L) arrays and reduces them single-threaded —
    the dominant serial cost when L is large (e.g. the L4 super-block, L ~ 1.5M).

    Computing denom/num on the fly per site avoids the (C, L) temporaries and
    parallelises over the L axis (controlled by numba.set_num_threads); reading
    d_r[carrier_idx[c]] directly also avoids materialising the per-rep (C, L)
    carrier-dosage gather d_rC.  num and denom are integer-valued counts, so the
    H decision num/denom >= 0.5 is equivalent to 2*num >= denom and is unaffected
    by the ULP of the division (the float result is at least 1/(2*denom) away from
    0.5 unless exactly 0.5, where it is exact) -- H, and therefore the downstream
    estar / deciding / n_supported, are identical to the numpy version; only
    frac1 / win_frac can differ by a ULP, exactly as for any reordered sum.
    """
    nC = carrier_idx.shape[0]
    L = d_r.shape[1]
    H = np.empty(L, dtype=np.int8)
    frac1 = np.zeros(L, dtype=np.float64)
    has = np.zeros(L, dtype=np.bool_)
    win_frac = np.empty(L, dtype=np.float64)

    for site in prange(L):
        denom = 0.0
        num = 0.0
        for c in range(nC):
            ps = partner_site[c, site]
            fdiff = d_r[carrier_idx[c], site] - cur[ps, site]
            if fdiff == 0 or fdiff == 1:          # contribute & Fest==fdiff here
                denom += 1.0
                num += fdiff
        if denom > 0.0:
            has[site] = True
            f1 = num / denom
            frac1[site] = f1
            if f1 >= 0.5:
                H[site] = 1
                win_frac[site] = f1
            else:
                H[site] = 0
                win_frac[site] = 1.0 - f1
        else:
            hh = N_tilde[site]
            H[site] = hh
            frac1[site] = 0.0                      # frac1 stays 0 where denom==0
            if hh == 1:
                win_frac[site] = 0.0               # where(H==1, frac1, .) -> frac1 == 0
            else:
                win_frac[site] = 1.0               # where(H==0, ., 1-frac1) -> 1 - 0

    return H, frac1, has, win_frac


@njit(parallel=True, fastmath=False)
def _dists_to_H_numba(cur, H):
    """Per-hap Hamming distance (in percent) from each hap to the consensus hap H,
    PARALLELISED over haps (prange).  Returns out[k] = 100 * mean(cur[k] != H),
    the parallel equivalent of  np.array([_hamming_pct(cur[k], H) for k in range(m)]).

    The per-hap mismatch count is an integer and the value is computed as
    100.0 * (cnt / L) -- the same operation order as numpy's
    100.0 * float(np.mean(cur[k] != H)) -- so the result is identical (and the
    downstream argmin / pmin are over integer-count distances separated by 100/L,
    far larger than any ULP, so estar is unaffected regardless).  m is small, but
    the work is tiny (m * L compares); haps are independent so there are no races
    and the result is independent of the thread count.
    """
    m = cur.shape[0]
    L = cur.shape[1]
    out = np.empty(m, dtype=np.float64)
    for k in prange(m):
        cnt = 0
        for site in range(L):
            if cur[k, site] != H[site]:
                cnt += 1
        out[k] = 100.0 * (cnt / L)
    return out


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