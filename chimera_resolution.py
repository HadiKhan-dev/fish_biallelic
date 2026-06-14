"""
Chimera Resolution Module

Sub-block forward selection, top-N swap refinement, BIC pruning,
and chimera resolution via hotspot-guided splicing.

Main entry point: select_and_resolve()

Low-level numba kernels live in chimera_kernels.py; the scoring/selection
machinery lives in chimera_scoring.py.  Both are re-imported below so that
external callers' `chimera_resolution.<name>` references keep working and
select_and_resolve can call everything unqualified, exactly as before.
"""

import numpy as np
import math
import os
import time as _time
from concurrent.futures import ThreadPoolExecutor

import block_haplotypes

# env-gated coarse phase timing inside select_and_resolve (no effect on results
# when off): set BHD_CR_PROFILE=1 to print Step1 / Step2-3-4 / Step5 wall times.
_CR_PROFILE = os.environ.get('BHD_CR_PROFILE', '') not in ('', '0', 'false', 'False')

# =============================================================================
# RE-EXPORTS from the kernel and machinery layers
# =============================================================================

from chimera_kernels import (
    _resolve_threads,
    _malloc_trim,
    _SAMPLE_CHUNK,
    _batched_viterbi_score,
    _batched_viterbi_score_split,
    _build_cand_sv_numba,
    _build_cand_idx_for_p,
    _build_tensor_block_numba,
    _build_tensor_block_numba_par,
    _viterbi_traceback,
    _compute_bin_emissions_numba,
    _update_bin_emissions_one_hap_numba,
    _dosage_round_numba,
    _gather_samples_numba,
    _viterbi_partner_carriers_numba,
    _consensus_from_carriers_numba,
    _dists_to_H_numba,
    _build_template_numba,
    _precompute_base_max_block_numba,
    _cheap_score_all_block_numba,
    warmup_jit,
)
from chimera_scoring import (
    compute_penalty,
    compute_spb,
    compute_cc,
    compute_subblock_emissions,
    _build_tensor_from_paths,
    score_path_set,
    score_path_sets_parallel,
    beam_warmstart_select,
    paint_samples_viterbi,
    _compute_all_pair_diffs_kernel,
    _find_hotspots_kernel,
    find_hotspots,
    _step5_find_active_paths,
    _step5_solve_constrained_assignment,
    _step5_murty_top_n,
    _step5_hungarian_active,
    _step5_murty_active,
    _step5_build_W_at_boundary,
    _step5_apply_sigma,
)


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

    # --- env-gated coarse phase timing (no effect on results when off:
    # _cacc only mutates _cpt inside `if _crp`, and the report is gated) ---
    _crp = _CR_PROFILE
    _cpt = {}
    def _cacc(_key, _t0):
        _e = _cpt.get(_key)
        _dt = _time.perf_counter() - _t0
        if _e is None:
            _cpt[_key] = [_dt, 1]
        else:
            _e[0] += _dt
            _e[1] += 1

    # --- Cap total bins to prevent memory blowup ---
    total_sites = sum(len(b.positions) for b in batch_blocks)
    estimated_bins = math.ceil(total_sites / spb)
    if estimated_bins > max_bins_for_cr:
        spb = math.ceil(total_sites / max_bins_for_cr)
    
    # --- Sub-block emissions ---
    _ct = _time.perf_counter()
    sub_em = compute_subblock_emissions(batch_blocks, global_probs, global_sites, spb,
                                        num_threads=num_threads)
    if _crp: _cacc('emission', _ct)
    _ct = _time.perf_counter()
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
    if _crp: _cacc('cr_setup_stack', _ct)
    _ct = _time.perf_counter()
    
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
    
    if _crp: _cacc('warmstart', _ct)

    # =========================================================================
    # STEP 1: Forward Selection (sample-chunked)
    # =========================================================================
    selected = list(_warmstart_selected)
    current_best_bic = _warmstart_bic
    _ct = _time.perf_counter()
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
    
    if _crp: _cacc('step1_forward_select', _ct)

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

    _ct = _time.perf_counter()
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

    if _crp: _cacc('step2_3_4_loop', _ct)
    _ct = _time.perf_counter()
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


    if _crp: _cacc('step5_escape', _ct)
    if _crp:
        try:
            _tot = sum(v[0] for v in _cpt.values())
            _lines = [f"  [CR profile] K_final={len(current_paths)} "
                      f"n_cands={n_cands} blocks={len(sub_em)} | sum={_tot:.1f}s"]
            for _k, _v in sorted(_cpt.items(), key=lambda kv: -kv[1][0]):
                _lines.append(f"        {_k:22s} {_v[0]:7.2f}s")
            print("\n".join(_lines), flush=True)
        except Exception:
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