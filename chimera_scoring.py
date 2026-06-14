"""
Chimera Resolution — Scoring & Selection Machinery

The algorithm layer between the numba kernels (chimera_kernels) and the
orchestrator (chimera_resolution.select_and_resolve).  Contents, moved verbatim
from chimera_resolution.py: parameter computation (penalty / spb / cc),
sub-block emission assembly, scoring-tensor construction and path-set scoring
(serial + batched), beam-warmstart initialisation, Viterbi sample painting and
hotspot detection, and the Step-5 constrained-assignment / Murty helpers.

These are re-imported into chimera_resolution so existing
`chimera_resolution.<name>` references keep resolving unchanged.
"""

import numpy as np
import math
import heapq
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

from scipy.optimize import linear_sum_assignment
from numba import njit, prange, get_num_threads as _numba_get_num_threads

import bhd_kernels

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
        # Dynamic thread resync (gated on num_threads being provided, so
        # any external caller that doesn't pass it sees the original
        # "no thread management" behaviour).  Done BEFORE the gather so the
        # (multi-GB at L4) block_samples copy is parallelised as well.
        if num_threads is not None:
            _resolve_threads(num_threads)
        # block_samples = np.ascontiguousarray(global_probs[:, indices, :]), but
        # via a parallel kernel.  numpy's advanced-index gather of the
        # (N, n_global, 3) probs runs single-threaded and costs ~10-20s at L4,
        # where it dominated the emission phase; _gather_samples_numba is a
        # bit-identical pure copy spread over the inner-core pool.
        block_samples = _gather_samples_numba(global_probs, indices)
        hap_keys = sorted(block.haplotypes.keys())
        n_haps = len(hap_keys)
        haps_tensor = np.array([block.haplotypes[k] for k in hap_keys])
        hap0 = np.ascontiguousarray(haps_tensor[:, :, 0])
        hap1 = np.ascontiguousarray(haps_tensor[:, :, 1])

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

def _build_tensor_from_paths(path_set, sub_emissions, num_samples, sample_range=None,
                             parallel_build=False):
    """Build Viterbi scoring tensor from a set of key-paths.
    If sample_range=(s0, s1) is given, only builds for those samples.

    Per-block fill goes through _build_tensor_block_numba, a serial
    numba kernel that writes the (n_pairs × n_bins_b) slice directly
    via three nested loops, avoiding the numpy fancy-index + slice-
    assign intermediate `tensor[:, :, off:off+nb] = bin_em[s0:s1,
    grid_i, grid_j, :]` (which would allocate a (n_s, K², n_bins_b)
    temporary per block per call).

    parallel_build: when True, the per-block fill uses the prange-over-
    samples kernel `_build_tensor_block_numba_par` instead of the serial
    one (identical values).  Callers reached from a single Python thread
    (score_path_set, paint_samples_viterbi -> the sequential warmstart
    walk and CR Steps 1-5) pass True so the otherwise-serial ~250 MB
    tensor write is spread across the numba thread pool.  The pooled
    caller `score_path_sets_parallel` keeps the default False — its own
    ThreadPoolExecutor owns the parallelism and a prange there would
    contend on the OMP pool.
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
            if parallel_build:
                _build_tensor_block_numba_par(
                    tensor, bin_em, local_indices,
                    s0, s1, K, n_bins_b, bin_offset,
                )
            else:
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

    This scorer is only ever called from single-Python-thread contexts
    (the sequential warmstart add walk and CR Steps 1-5), so the tensor
    build uses the prange-over-samples kernel (parallel_build=True): the
    build is the dominant per-call cost and there is no outer
    ThreadPoolExecutor to contend with.
    """
    tensor = _build_tensor_from_paths(path_set, sub_emissions, num_samples,
                                      parallel_build=True)
    if num_threads is not None:
        _resolve_threads(num_threads)
    return float(np.sum(bhd_kernels.viterbi_score_selection(tensor, float(penalty))))


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

    Like score_path_set, this is only reached from single-Python-thread
    contexts (CR Steps 2/4), so the build uses the prange-over-samples
    kernel (parallel_build=True).
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
    tensor = _build_tensor_from_paths(path_set, sub_emissions, num_samples,
                                      parallel_build=True)
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