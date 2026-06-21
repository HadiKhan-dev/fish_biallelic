#%% =====================================================================
# bhd_recovery.py — Subtraction-based recovery + late low-carrier rescue
#
# 4-WAY SPLIT (this module = orchestration + public API): the Bernoulli-
# mixture machinery, residual-candidate generation, and BIC subset-selection
# now live in bhd_recovery_mixture.py, bhd_recovery_candidates.py, and
# bhd_recovery_select.py respectively.  This module retains the recovery
# orchestration (_subtraction_recovery_round_loop, _residual_trio_rescue,
# _late_low_carrier_rescue), the recovery public API, and the cross-module
# wiring to bhd_trio / bhd_pairwise.  The component descriptions and numba-
# optimisation history below document the whole subsystem.
#
# Split out of block_haplotypes.py as part of the 4-file split.
# Contains the candidate-pool-driven recovery subsystems that run after
# the initial K-growth pass:
#
#   - Recovery constants (RECOVERY_*, RECOVERY_LOW_CARRIER_*)
#   - Bernoulli mixture machinery (_kmeans_pp_init,
#     _bernoulli_mixture_em, _fit_bernoulli_mixture_select_K) — fits
#     a K-component mixture over candidate haps with inner BIC over K.
#   - Candidate generation (_run_subtraction_round, _generate_carrier_-
#     residuals) — produces residual-candidate haps for downstream
#     subset selection.
#   - BIC subset selection (_greedy_bic_select, _swap_refine, _bic_-
#     prune) — outer BIC scoring on sample data; the BIC criterion
#     here uses Viterbi NLL via bhd_fit._compute_nll_for_subset
#     when _VITERBI_BIC_ENABLED is True (the production default).
#   - Hap-comparison helpers (_hamming_pct_kept, _haps_equal) — used by
#     the round loop, late rescue, and the orchestrator's
#     _grow_K_with_recovery for convergence detection.
#   - Round loop (_subtraction_recovery_round_loop) — outer iteration
#     wrapping candidate generation + BIC selection until convergence.
#   - Late low-carrier rescue (_late_low_carrier_rescue) — targeted
#     post-convergence pass that re-generates trio + pairwise
#     candidates and tries to BIC-improve the current solution.
#
# Dependencies:
#   - bhd_kernels (BIC scorer, CD primitives, constants)
#   - bhd_trio (_trio_recovery_candidate_haps in late rescue)
#   - bhd_pairwise (pairwise_recovery_candidate_haps + PAIRWISE_-
#     RECOVERY_ENABLED flag in late rescue)
#
# NUMBA OPTIMIZATION (added after bhd_trio + bhd_pairwise numba passes,
# based on production profiling at N=320, L=200, K=6):
#   _run_subtraction_round           — 7.1 ms -> ~1 ms (5-7x), via scalar-
#                                       loop kernel that preallocates the
#                                       output buffer and eliminates the
#                                       per-pool-member numpy temporaries
#   _generate_carrier_residuals      — 1.2 ms -> ~0.2 ms (5-6x), via
#                                       scalar-loop kernel with bool-mask
#                                       (not Python set) for low_idx
#                                       lookups; verbose path stays in
#                                       Python (diagnostics-only)
#   _bernoulli_mixture_em            — 0.30 ms -> 0.18 ms (1.6x), via
#                                       full E/M loop in a numba kernel
#                                       with inlined axis-1 logsumexp;
#                                       matmuls remain BLAS-dispatched
#                                       so per-iter perf bounded by
#                                       OpenBLAS, no regression risk.
#                                       Numerically equivalent to the
#                                       Python version to 1e-9 on theta
#                                       and 1e-12 on log-likelihood.
#   _kmeans_pp_init                  — 0.45 ms -> 0.04 ms (11x), via
#                                       scalar-loop kernel with running
#                                       min-distance update (vs the
#                                       original's full recomputation
#                                       each iteration) and inverse-CDF
#                                       weighted sampling.
#                                       SEMANTIC CAVEAT: the wrapper
#                                       draws K uniform [0, 1) values
#                                       from the user's Generator and
#                                       feeds them to the kernel for
#                                       weighted sampling, rather than
#                                       calling rng.integers/rng.choice
#                                       directly.  This advances the
#                                       PRNG state differently — the
#                                       specific initial centers picked
#                                       may shift across multi-start
#                                       restarts.  Project owner has
#                                       accepted this in exchange for
#                                       the 11x inner speedup; tested
#                                       empirically on 20 seeds —
#                                       _fit_bernoulli_mixture_select_K
#                                       picks the same K and the same
#                                       binary thetas downstream as the
#                                       legacy implementation.
# The first two kernels (_run_subtraction_round, _generate_carrier_-
# residuals) preserve byte-identical output to the pre-numba versions.
# The EM kernel is numerically equivalent to ~1e-12 LL drift.  The
# kmeans_pp kernel may pick different specific centers due to different
# PRNG-state advancement, but with downstream stability verified.
#
# SECOND-PASS ADDITIONS (small high-volume utilities):
#   _hamming_pct_kept                — 4.2 us -> 0.6 us (7x), scalar-
#                                       loop kernel that eliminates the
#                                       (L,) bool array allocation per
#                                       call.  Called ~120x per block
#                                       inside dedup loops and the late-
#                                       rescue trio/pairwise candidate
#                                       merging.  Wrapper casts both
#                                       inputs to int64 so the kernel
#                                       compiles once regardless of
#                                       caller dtype.
#   _haps_equal                      — 30 us -> 13 us (2.3x), stacking
#                                       the input lists/arrays into 2D
#                                       int64 then doing bipartite
#                                       matching with inlined scalar-
#                                       loop Hamming.  Modest speedup
#                                       because the np.stack overhead
#                                       in the wrapper dominates the
#                                       tiny inner work; called only
#                                       twice per round loop for
#                                       convergence detection.
# All 6 (now: 4 main + the 2 utility) bhd_recovery kernels are output-
# equivalent to the pre-numba implementations.
#
# These are not the hot path in stage 3 (combined ~50ms/call vs ~100s/block
# total) — bhd_kernels._viterbi_nll and _fit_at_fixed_K dominate.  Numba
# applied here for consistency with the other bhd_* modules and to remove
# the small per-recovery-round overhead.
# =======================================================================


import numpy as np

# Shared dynamic-thread reallocation, re-checked at this module's recovery
# phase boundaries so a straggler Stage-3 block grows into cores freed as its
# peers finish; no-ops on the sequential path until a pool wires the counters.
import dynamic_threads

import bhd_trio
import bhd_pairwise
from bhd_trio import _trio_recovery_candidate_haps
from bhd_fit import (
    _compute_cc,
    _compute_bic,
    _fit_at_fixed_K,
)
from bhd_pool import PoolEmissionCache
from bhd_config import (
    PAIRWISE_RECOVERY_ENABLED,
    RECOVERY_CLEANNESS_THRESHOLD,
    RECOVERY_HAPS_EQUAL_EPS_PCT,
    RECOVERY_INTRA_ROUND_DEDUP_PCT,
    RECOVERY_LOW_CARRIER_TRIGGER_FRAC,
    RECOVERY_MAX_K,
    RECOVERY_MAX_ROUNDS,
    RECOVERY_MIXTURE_K_MAX,
    RECOVERY_MIXTURE_N_RESTARTS,
    RECOVERY_MIXTURE_PATIENCE,
    RECOVERY_MIXTURE_RNG_SEED,
    RECOVERY_OUTER_CC_SCALE,
    RECOVERY_RESIDUAL_MODE,
    RECOVERY_SWAP_NLL_TOLERANCE,
    RESIDUAL_TRIO_CLEANNESS_THRESHOLD,
    RESIDUAL_TRIO_DEDUP_PCT,
    RESIDUAL_TRIO_DEDUP_VS_H_PCT,
    RESIDUAL_TRIO_MIN_CLUSTER_SIZE,
)

# Recovery subsystem internals, extracted in the 4-file split:
from bhd_recovery_mixture import (
    _fit_bernoulli_mixture_ml_select_K,
    _fit_bernoulli_mixture_select_K,
)
from bhd_recovery_candidates import (
    _generate_all_sample_residuals,
    _generate_carrier_residuals,
    _run_subtraction_round,
    _run_subtraction_round_soft,
)
from bhd_recovery_select import (
    _bic_prune,
    _count_real_carriers,
    _dedup_vs_refset_kernel,
    _greedy_bic_select,
    _hamming_pct_kept,
    _haps_equal,
    _swap_refine,
)


# =============================================================================
# SUBTRACTION-BASED RECOVERY — APPROACH (mixture + BIC + outer iteration)
# =============================================================================
#
# These constants govern the subtraction-recovery pass that runs after
# K-growth.  Recovery generates clean residual candidates by subtracting
# each current founder from each sample's argmax dosage, fits a Bernoulli
# mixture model with K selected by BIC over candidate-density, then runs
# outer BIC subset-selection on (existing founders ∪ mixture consensus)
# against actual sample data.  The outer pass is iterated with K-growth
# so that worst-fit-sample seeding (K-growth's mechanism) and density-
# based seeding (mixture's mechanism) catch different failure modes.


# =============================================================================
# LATE LOW-CARRIER RESCUE — BACKGROUND (post-convergence targeted refinement)
# =============================================================================
#
# Background: at blocks where one truth founder is carried by very few
# samples (say, 4 out of N=320 = 0.6% of strands), trio recovery may
# produce a candidate close to that truth (e.g., 2.5% Hamming) that is
# BIC-trimmed in correctly, but joint CD on H_trio_seed drifts the
# candidate slightly further from truth (e.g., 3% Hamming).  The drifted
# hap then captures the same 4 carriers as truth would have, fitting
# them at the noise floor — so NLL is identical to truth's NLL.  At
# the same K, NLL is identical (degenerate plateau).  The mixture in
# subtraction recovery clusters at K=1..mixture_K_max and uses inner
# BIC over candidate density to pick K; sparse low-frequency-founder
# residuals (here, 4 candidates out of ~1668) are absorbed into a
# larger mixture component, so the mixture's consensus haps do NOT
# include a near-truth alternative that would let forward selection
# pick the K-optimal subset.  The pipeline ends at K=K_truth+1 with
# a chimera in place of the low-frequency truth, losing by exactly cc
# on BIC.  Diagnosed at chr6:23624234; see diagnose_chr6_23624234.py.
#
# Late rescue mechanism: after _grow_K_with_recovery's outer iteration
# converges, identify selected haps with very low carrier counts.  For
# each such hap h_low, generate per-strand residuals from h_low's
# carrier samples (residual = sample.argmax_dosage - other_strand_hap).
# By construction, these residuals are clean approximations of the
# "true" version of h_low — for chr6:23624234, the 4 carriers of
# alg_row_5 are EXACTLY the 4 truth_4-carrying samples, and the
# residuals after subtracting the truth-matching other strand yield
# 4 candidates near truth_4.  Add these to the candidate pool, run
# greedy BIC forward selection, and accept iff BIC strictly improves.
# Forward selection is K-aware (BIC stops naturally at the optimal K),
# so it can REDUCE K by 1 in the chimera-replacement case.
#
# Cost: triggered only on blocks with at least one low-carrier hap
# (typically <5% of blocks).  Per triggered block: residual generation
# is O(N*L), pool grows by 4-20 candidates, forward selection +
# _fit_at_fixed_K cost ~1 sec.  Negligible aggregate cost (<1% of
# stage 3 runtime).  Untriggered blocks have zero overhead.


# =============================================================================
# SUBTRACTION-BASED RECOVERY: ROUND LOOP
# =============================================================================

def _subtraction_recovery_round_loop(probs_k, H_init, lam,
                                       outer_cc_scale=RECOVERY_OUTER_CC_SCALE,
                                       max_K=RECOVERY_MAX_K,
                                       max_rounds=RECOVERY_MAX_ROUNDS,
                                       max_iter_per_K=50,
                                       intra_round_dedup_pct=RECOVERY_INTRA_ROUND_DEDUP_PCT,
                                       mixture_K_max=RECOVERY_MIXTURE_K_MAX,
                                       mixture_n_restarts=RECOVERY_MIXTURE_N_RESTARTS,
                                       mixture_seed_base=RECOVERY_MIXTURE_RNG_SEED,
                                       mixture_patience=RECOVERY_MIXTURE_PATIENCE,
                                       cleanness_threshold=RECOVERY_CLEANNESS_THRESHOLD,
                                       swap_nll_tolerance=RECOVERY_SWAP_NLL_TOLERANCE,
                                       haps_equal_eps_pct=RECOVERY_HAPS_EQUAL_EPS_PCT,
                                       use_log_bic=False,
                                       site_priors=None,
                                       verbose=False):
    """Iterative subtraction-recovery rounds until convergence.

    Each round:
      1. Subtract each hap in current selected from each sample's argmax
         dosage; collect clean clipped residuals as raw candidates.
      2. Fit Bernoulli mixture for K=1..mixture_K_max with K-means++
         init + multi-restart EM.  Pick K minimising inner BIC.  Output:
         K consensus haps (binary).
      3. Intra-round dedup (safety net at intra_round_dedup_pct: tight
         duplicates that survived numerical rounding get merged).
      4. Pool = current selected union new consensus haps.
      5. Greedy BIC forward-select on sample data with FIXED haps
         (outer BIC, cc_scale=outer_cc_scale).
      6. Swap refinement.
      7. BIC pruning.
      8. Coord descent on the selected haps via _fit_at_fixed_K (this
         is where haps actually move toward truth).
      9. Convergence check: if selected unchanged from previous round
         (within haps_equal_eps_pct), exit.

    Args:
      probs_k: (N, L_kept, 3) — kept-site posteriors
      H_init: (K_init, L_kept) starting hap set (typically from K-growth)
      lam: wildcard penalty
      outer_cc_scale: BIC scale for outer subset selection on sample data
      max_K: hard cap on selected size
      max_rounds: hard cap on round iterations (defensive)
      max_iter_per_K: cap on coord-descent iterations within each round
      intra_round_dedup_pct: tight dedup threshold for consensus haps
      mixture_K_max, mixture_n_restarts, mixture_seed_base: inner mixture params
      mixture_patience: K-sweep early-stop patience for the inner mixture
        (see RECOVERY_MIXTURE_PATIENCE); None disables the early stop.
      cleanness_threshold: min residual cleanness to admit a candidate
      swap_nll_tolerance: NLL improvement floor for accepting a swap
      haps_equal_eps_pct: tolerance for round-convergence detection
      site_priors: (L_kept, 3) per-site genotype priors, or None.  Used
        only in RECOVERY_RESIDUAL_MODE == "soft" to divide the HWE site
        prior out of probs_k and recover the genotype LIKELIHOODS for the
        marginal-likelihood mixture (real data); None is correct when
        probs_k already is the likelihood (flat prior, e.g. synthetic).
      verbose: print per-round trace

    Returns:
      H_final: (K_final, L_kept) — refined hap set after recovery
    """
    if H_init is None or len(H_init) == 0:
        # Nothing to subtract from; recovery has no anchor
        return np.empty((0, probs_k.shape[1]), dtype=np.int64)

    selected = [np.asarray(H_init[k], dtype=np.int64).copy() for k in range(H_init.shape[0])]

    # Pre-compute argmax dosage on kept sites for subtraction
    argmax_dosage_kept = probs_k.argmax(axis=2)                              # (N, L_kept)

    prev_selected = [s.copy() for s in selected]

    # Carry a `prev_round_cache` reference across rounds so each round's
    # PoolEmissionCache build can reuse rows for haps whose CONTENT
    # appeared in the previous round's pool.  Saves ~30-50% of the cache
    # construction time when CD didn't move existing `selected` haps and
    # only a small number of new consensus haps were added — exactly the
    # late-round-convergence pattern.  See PoolEmissionCache.__init__ in
    # bhd_kernels.py for the row-reuse logic and bit-equivalence proof.
    prev_round_cache = None

    for round_num in range(1, max_rounds + 1):
        # Re-check thread allocation at the top of each recovery round.
        dynamic_threads.apply_dynamic_threads()
        # 1+2. Generate consensus haps (residuals -> mixture -> BIC over K).
        #    In "soft" mode the residuals are the other-strand LIKELIHOODS
        #    (L0, L1) per site, fed to the marginal-likelihood Bernoulli-
        #    haplotype mixture (_fit_bernoulli_mixture_ml_select_K), which
        #    marginalises the latent other-strand allele rather than hard-
        #    calling it.  In the default "argmax" mode the residuals are the
        #    binary clipped residuals of the argmax dosage, fed to the binary
        #    Bernoulli mixture.  Both return K consensus haps (binary).
        if RECOVERY_RESIDUAL_MODE == "soft":
            L0_cand, L1_cand = _run_subtraction_round_soft(
                selected, probs_k, site_priors=site_priors,
                cleanness_threshold=cleanness_threshold)
            n_raw = L0_cand.shape[0]
            if n_raw == 0:
                if verbose:
                    print(f'  [recovery round {round_num}] no clean residuals -- CONVERGED')
                break
            if verbose:
                print(f'  [recovery round {round_num}] {n_raw} raw candidates')
            consensus_haps = _fit_bernoulli_mixture_ml_select_K(
                L0_cand, L1_cand,
                K_max=mixture_K_max,
                n_restarts=mixture_n_restarts,
                seed=mixture_seed_base + round_num,
                patience=mixture_patience,
                verbose=verbose)
        else:
            raw_candidates = _run_subtraction_round(
                selected, argmax_dosage_kept,
                cleanness_threshold=cleanness_threshold)
            if len(raw_candidates) == 0:
                if verbose:
                    print(f'  [recovery round {round_num}] no clean residuals -- CONVERGED')
                break
            if verbose:
                print(f'  [recovery round {round_num}] {len(raw_candidates)} raw candidates')
            consensus_haps = _fit_bernoulli_mixture_select_K(
                raw_candidates,
                K_max=mixture_K_max,
                n_restarts=mixture_n_restarts,
                seed=mixture_seed_base + round_num,
                patience=mixture_patience,
                verbose=verbose)
        if len(consensus_haps) == 0:
            if verbose:
                print(f'  [recovery round {round_num}] no mixture consensus -- CONVERGED')
            break

        # 3. Intra-round dedup (safety net only)
        new_haps = []
        for consensus in consensus_haps:
            is_dup = False
            for other in new_haps:
                if _hamming_pct_kept(consensus, other) < intra_round_dedup_pct:
                    is_dup = True
                    break
            if not is_dup:
                new_haps.append(consensus)

        # 4. Pool = selected union new_haps
        pool = list(selected) + list(new_haps)

        # Build the per-round PoolEmissionCache.  All `_compute_nll_for_-
        # subset` calls inside _greedy_bic_select / _swap_refine /
        # _bic_prune in this round will be evaluated against this fixed
        # pool, so the cache amortises the Viterbi emission build over
        # the hundreds of subset queries that follow.  See bhd_kernels.
        # PoolEmissionCache for the design rationale.  Memory cost:
        # O(N * |pool|² * n_bins / 2) ≈ a few MB to ~30 MB depending on
        # pool size; discarded at the end of this iteration.
        #
        # `prev_round_cache` lets the constructor copy emission rows for
        # haps whose content matches a hap in the previous round's
        # pool — saving the dominant rr-pair build cost in the common
        # case where most of `selected` is unchanged round-to-round.
        round_cache = PoolEmissionCache(pool, probs_k, lam=lam,
                                         prev_cache=prev_round_cache)
        prev_round_cache = round_cache

        # 5. Greedy BIC forward selection (haps frozen)
        sel_indices, sel_haps, sel_nll = _greedy_bic_select(
            round_cache,
            cc_scale=outer_cc_scale, max_k=max_K,
            use_log_bic=use_log_bic, verbose=verbose)

        # 6. Swap refinement (haps still frozen)
        if len(sel_haps) > 0:
            sel_indices, sel_haps, sel_nll, n_swaps = _swap_refine(
                round_cache, sel_indices,
                current_nll=sel_nll,
                nll_tolerance=swap_nll_tolerance,
                verbose=verbose)

        # 7. BIC pruning (haps still frozen)
        if len(sel_haps) > 0:
            sel_indices, sel_haps, sel_nll, n_dropped = _bic_prune(
                round_cache, sel_indices,
                cc_scale=outer_cc_scale, use_log_bic=use_log_bic,
                verbose=verbose)

        # 8. Coord descent on selected to refine haps.  This is the
        #    step that actually MOVES haps — until now the candidates
        #    were frozen (binary outputs of mixture).  CD aligns them
        #    to the actual data.
        if len(sel_haps) > 0:
            H_sel = np.stack(sel_haps, axis=0)
            H_refined, A_ref, costs_ref, wcs_ref, n_iter_ref, nll_ref = \
                _fit_at_fixed_K(probs_k, H_sel, lam, max_iter=max_iter_per_K)
            new_selected = [H_refined[k].copy() for k in range(H_refined.shape[0])]
        else:
            new_selected = sel_haps

        # 9. Convergence check
        if _haps_equal(new_selected, prev_selected, eps_pct=haps_equal_eps_pct):
            if verbose:
                print(f'  [recovery round {round_num}] selected unchanged -- CONVERGED')
            selected = new_selected
            break

        selected = new_selected
        prev_selected = [s.copy() for s in selected]

    # Return as np.array (consistent with K-growth's H format)
    if len(selected) == 0:
        return np.empty((0, probs_k.shape[1]), dtype=np.int64)
    return np.stack(selected, axis=0)


def _residual_trio_rescue(probs_k, H, A, costs, wcs, NLL,
                          lam, cc_scale, use_log_bic, max_iter,
                          cleanness_threshold=RESIDUAL_TRIO_CLEANNESS_THRESHOLD,
                          dedup_pct=RESIDUAL_TRIO_DEDUP_PCT,
                          dedup_vs_h_pct=RESIDUAL_TRIO_DEDUP_VS_H_PCT,
                          min_cluster_size=RESIDUAL_TRIO_MIN_CLUSTER_SIZE,
                          verbose=False):
    """Post-K-growth pass surfacing near-clone founders via per-sample
    residual mining + clustering.

    Architecture mirrors _late_low_carrier_rescue but with two
    differences:
      - Mines residuals across ALL samples (not just low-carrier-hap
        carriers).  This is the new candidate source.
      - No usage-based trigger gate.  The implicit gate is "did the
        residual clustering produce any candidate that isn't already
        in H?"  If not, returns the inputs unchanged at near-zero cost.

    Args:
      probs_k: (N, L_kept, 3) — kept-site posteriors
      H: (K, L_kept) int64 — current discrete founder bits
      A: (N, 2) int64 — current pair assignments (K = wildcard sentinel)
      costs: (N,) — per-sample CAPPED cost
      wcs: (N,) — per-sample wildcard slot count
      NLL: float — current UNCAPPED total NLL
      lam: wildcard penalty
      cc_scale, use_log_bic: BIC formula parameters (must match outer
        pipeline)
      max_iter: cap on _fit_at_fixed_K coord-descent iterations
      cleanness_threshold: min admissible-site fraction for residuals
        (default RESIDUAL_TRIO_CLEANNESS_THRESHOLD = 1.0)
      dedup_pct: candidate-vs-candidate clustering threshold (% Hamming)
      dedup_vs_h_pct: candidate-vs-H dedup threshold (% Hamming) — drop
        candidates already within this distance of any H row
      min_cluster_size: minimum supporting samples per admitted candidate
      verbose: if True, print diagnostic trace

    Returns:
      (H, A, costs, wcs, NLL) — possibly updated; identical to inputs
      if no admitted candidate or BIC did not improve.
    """
    N, L_kept = probs_k.shape[0], probs_k.shape[1]
    K = H.shape[0]
    if K == 0:
        return H, A, costs, wcs, NLL

    # Step 1: Generate residuals across all samples
    residuals_list = _generate_all_sample_residuals(
        probs_k, H, A, cleanness_threshold=cleanness_threshold)

    if not residuals_list:
        if verbose:
            print(f'[residual-trio] no clean residuals — skipping '
                  f'(cleanness_threshold={cleanness_threshold})')
        return H, A, costs, wcs, NLL

    if verbose:
        print(f'[residual-trio] generated {len(residuals_list)} clean '
              f'residuals (cleanness_threshold={cleanness_threshold})')

    # Step 2: Cluster the residuals.  Use bhd_trio's clustering kernel
    # which we already use for hom-recovery candidates — same single-
    # pass online clustering with majority-vote consensus, ensuring
    # candidate sets from different sources cluster compatibly.
    threshold_bits = int(dedup_pct / 100.0 * L_kept)
    residuals_arr = np.ascontiguousarray(
        np.stack(residuals_list, axis=0).astype(np.int64))
    cluster_buf, n_clusters = bhd_trio._cluster_haps_consensus_kernel(
        residuals_arr, threshold_bits, min_cluster_size)
    cluster_haps = [cluster_buf[i].copy() for i in range(n_clusters)]

    if verbose:
        print(f'[residual-trio] clustered {len(residuals_list)} residuals -> '
              f'{n_clusters} clusters (dedup_pct={dedup_pct}%, '
              f'min_cluster_size={min_cluster_size})')

    if not cluster_haps:
        if verbose:
            print(f'[residual-trio] no clusters survived min_cluster_size — '
                  f'skipping')
        return H, A, costs, wcs, NLL

    # Step 3: Drop cluster centroids that are already in H (within
    # dedup_vs_h_pct).  The remaining candidates are the "new" near-
    # clone founders that residual-trio is proposing.
    H_list = [H[k] for k in range(K)]
    new_candidates = []
    for cand in cluster_haps:
        is_dup = False
        for h in H_list:
            if _hamming_pct_kept(cand, h) < dedup_vs_h_pct:
                is_dup = True
                break
        if is_dup:
            continue
        new_candidates.append(cand)

    if not new_candidates:
        if verbose:
            print(f'[residual-trio] all {n_clusters} cluster centroids are '
                  f'already in H (within {dedup_vs_h_pct}%) — skipping')
        return H, A, costs, wcs, NLL

    if verbose:
        print(f'[residual-trio] {len(new_candidates)} candidates not in H '
              f'(dedup_vs_h_pct={dedup_vs_h_pct}%):')
        for ci, cand in enumerate(new_candidates):
            hams = [_hamming_pct_kept(cand, H[k]) for k in range(K)]
            ham_str = ', '.join(f'H{k}={h:5.2f}%' for k, h in enumerate(hams))
            print(f'    new_cand[{ci}]: [{ham_str}]')

    # Step 4: BIC subset selection on the enriched pool.  Same as
    # _late_low_carrier_rescue: build a PoolEmissionCache once (precom-
    # putes Viterbi emissions for the whole pool) then run greedy
    # forward → swap refinement → BIC prune → refit at fixed K →
    # compare BIC.
    pool = H_list + new_candidates

    # Build the residual-trio PoolEmissionCache.  All _greedy_bic_select /
    # _swap_refine / _bic_prune calls below evaluate subsets of this
    # fixed pool, so we precompute the Viterbi emission tensor once.
    rescue_cache = PoolEmissionCache(pool, probs_k, lam=lam)

    cc = _compute_cc(cc_scale, N, L_kept, use_log_bic=use_log_bic)
    BIC_orig = _compute_bic(K, NLL, cc)

    # Greedy BIC forward selection on enriched pool.  max_k = K + 1
    # allows BIC to grow K by 1, OR shrink K by stopping early when a
    # smaller subset has lower BIC.
    sel_indices, sel_haps, sel_nll = _greedy_bic_select(
        rescue_cache,
        cc_scale=cc_scale, max_k=K + 1,
        use_log_bic=use_log_bic, verbose=verbose)

    if not sel_haps:
        if verbose:
            print(f'[residual-trio] forward selection picked K=0 — keeping '
                  f'original')
        return H, A, costs, wcs, NLL

    sel_indices, sel_haps, sel_nll, n_swaps = _swap_refine(
        rescue_cache, sel_indices, sel_nll,
        nll_tolerance=RECOVERY_SWAP_NLL_TOLERANCE,
        max_passes=10, verbose=verbose)

    sel_indices, sel_haps, sel_nll, n_dropped = _bic_prune(
        rescue_cache, sel_indices,
        cc_scale=cc_scale, use_log_bic=use_log_bic, verbose=verbose)

    if not sel_haps:
        if verbose:
            print(f'[residual-trio] post-swap-prune picked K=0 — keeping '
                  f'original')
        return H, A, costs, wcs, NLL

    # Step 5: Refit at chosen K and compare BIC.  Same convention as
    # _late_low_carrier_rescue: strict improvement by > 0.1 to avoid
    # float-noise oscillation.
    H_new = np.array(sel_haps, dtype=np.int64)
    H_new, A_new, costs_new, wcs_new, n_iter_new, NLL_new = _fit_at_fixed_K(
        probs_k, H_new, lam, max_iter=max_iter)
    K_new = H_new.shape[0]
    BIC_new = _compute_bic(K_new, NLL_new, cc)

    if verbose:
        swap_str = f'{n_swaps} swap{"s" if n_swaps != 1 else ""}'
        prune_str = f'{n_dropped} prune{"s" if n_dropped != 1 else ""}'
        print(f'[residual-trio] orig: K={K}, NLL={NLL:.1f}, BIC={BIC_orig:.1f}')
        print(f'[residual-trio] new:  K={K_new}, NLL={NLL_new:.1f}, '
              f'BIC={BIC_new:.1f} (after {swap_str}, {prune_str})')

    if BIC_new < BIC_orig - 0.1:
        if verbose:
            print(f'[residual-trio] BIC improved by {BIC_orig - BIC_new:.1f} '
                  f'— ACCEPT')
        return H_new, A_new, costs_new, wcs_new, NLL_new
    if verbose:
        print(f'[residual-trio] BIC did not improve '
              f'(delta={BIC_orig - BIC_new:+.1f}) — KEEP ORIGINAL')
    return H, A, costs, wcs, NLL


# =============================================================================
# LATE LOW-CARRIER RESCUE (targeted post-convergence pass)
# =============================================================================

def _late_low_carrier_rescue(probs_k, H, A, costs, wcs, NLL,
                              lam, cc_scale, use_log_bic, max_iter,
                              low_carrier_frac=RECOVERY_LOW_CARRIER_TRIGGER_FRAC,
                              cleanness_threshold=RECOVERY_CLEANNESS_THRESHOLD,
                              verbose=False):
    """Late targeted refinement for blocks with a low-carrier hap that
    may be a chimeric stand-in for a low-frequency founder.

    Triggered when min(carrier_count) < low_carrier_frac * 2*N.  When
    triggered, this pass:

      1. Identifies all suspect haps (carrier count below threshold).
      2. (Diagnostic only when verbose:) generates per-(sample, strand)
         carrier residuals via _generate_carrier_residuals and prints
         their Hamming-% to current H rows.  These residuals are NOT
         used as candidates because by the optimality of A's pair
         assignment, argmax_dosage[s] - H[partner_X] ≈ h_low for every
         carrier — the residual just reproduces the chimera (verified
         empirically at chr6:23624234, all 4 carriers' residuals at
         0.00% from H[chim_5]).
      3. Re-runs trio recovery (_trio_recovery_candidate_haps) on the
         block's probs_k.  Trio candidates come from XOR-triangulation
         of within-cluster sample triples — a structural/algebraic
         property of the genotype data, INDEPENDENT of current H or
         A.  They retain pre-CD-drift information about candidate
         founders, including low-frequency truths that the rest of
         the pipeline drifted away from.  Trio is deterministic
         (soft-agreement clustering via HDBSCAN) and cheap (~100ms at N=320).
      4. Dedups trio candidates against current H at 0.5% (tight; only
         collapse near-exact duplicates so we don't waste evaluations
         on candidates that are already in H).
      5. Builds pool = current H ∪ surviving trio candidates and runs
         greedy BIC forward selection with max_k = K + 1.
      6. Swap refinement: tests "drop selected_i, insert pool_j" at
         fixed K for every (i, j) pair — the operation needed to
         displace a chimera from a slot in favour of a trio candidate.
      7. BIC pruning: drops any selected hap whose NLL contribution
         falls below cc/2.  After swap pulls in a truth-near hap, the
         secondary chimera that was absorbing 'overflow' carriers
         becomes redundant and prunes cleanly — this is what gives
         the K=7 → K=6 BIC win at chr6:23624234.
      8. Refits via _fit_at_fixed_K.  With the secondary chimera
         pruned, the truth_4 carriers' M-step votes on the slot-5
         hap are no longer diluted, and CD pulls the trio candidate
         (from 2.5%) toward truth_4 (at 0%).
      9. Replaces the input state iff BIC strictly improves (by more
         than 0.1 to avoid float-noise oscillation).

    Cost: trio_recovery O(N²), pool size = K + (≤6 trio candidates),
    forward + swap (≤10 passes) + prune + fit each O(K * pool * N *
    L).  About 1-2 sec per triggered block.  Untriggered blocks (min
    carrier count above threshold) return immediately at near-zero
    cost.

    See RECOVERY_LOW_CARRIER_TRIGGER_FRAC for the chr6:23624234
    motivating diagnosis.

    Args:
      probs_k: (N, L_kept, 3) — kept-site posteriors
      H: (K, L_kept) — current discrete founder bits
      A: (N, 2) — current pair assignments
      costs: (N,) — per-sample CAPPED cost (matches _fit_at_fixed_K
        return convention)
      wcs: (N,) — per-sample wildcard slot count
      NLL: float — current UNCAPPED total NLL
      lam: wildcard penalty
      cc_scale: BIC complexity-cost scale (must match outer pipeline)
      use_log_bic: BIC formula selector (must match outer pipeline)
      max_iter: cap on _fit_at_fixed_K coord-descent iterations
      low_carrier_frac: trigger threshold; min carrier fraction of 2N
      cleanness_threshold: min admissible-site fraction for residuals
      verbose: print diagnostic trace

    Returns:
      (H, A, costs, wcs, NLL) — possibly updated; identical to inputs
      if rescue did not trigger or did not yield a BIC improvement.
    """
    N, L_kept = probs_k.shape[0], probs_k.shape[1]
    K = H.shape[0]
    if K == 0:
        return H, A, costs, wcs, NLL

    # Compute per-hap real-strand carrier counts (excluding wildcards).
    # The inner double loop over (sample, strand) runs at njit speed
    # via _count_real_carriers_kernel; W = K is the wildcard sentinel
    # the kernel uses to skip non-real-strand entries.
    usage = _count_real_carriers(A, K)

    # Trigger condition: any hap below the low-carrier threshold.
    # Floor of 2 ensures we never trigger on degenerate K=0 or K=1
    # blocks where threshold rounds to 0 (e.g., very small N).
    threshold = max(2, int(low_carrier_frac * 2 * N))
    low_idx_list = [k for k in range(K) if int(usage[k]) < threshold]
    if not low_idx_list:
        return H, A, costs, wcs, NLL

    if verbose:
        usage_str = ','.join(f'{k}:{int(usage[k])}' for k in low_idx_list)
        print(f'[late-rescue] triggered: K={K}, low-carrier haps '
              f'(threshold={threshold}): {{{usage_str}}}')

    # Generate candidate haplotypes from carrier residuals.
    #
    # For each (carrier_sample, carrier_strand) of a low-carrier hap
    # h_low, _generate_carrier_residuals iterates over EVERY current H
    # row as a candidate subtractor.  By the algebra (clean data, the
    # other 5 truths are perfect), exactly one subtractor — the actual
    # other strand of that carrier — produces a residual = (the missing
    # truth) at 100% cleanness.  All other subtractors produce residuals
    # with out-of-range bits at heterozygous sites, failing the cleanness
    # threshold.  This is the user's insight (chr6:23624234 message): if
    # we subtract the right partner, we get the missing hap perfectly.
    #
    # The previous broken version subtracted only A[s, other_slot] (the
    # algorithm's *fitted* partner for the given chimera, not the
    # *actual* other strand), producing residual = chim_low for every
    # carrier — verified empirically.  The fix is to test all H rows
    # and let cleanness filter discriminate.
    #
    # Trio candidates are also added as a complementary source: trio
    # generates candidates from XOR-triangulation of within-cluster
    # sample triples — independent of H or A.  At chr6:23624234 trio
    # originally produced a candidate at 2.5% from truth_4, which got
    # drifted to 3% (= chim_5) during downstream CD; re-running trio
    # here regenerates the pre-drift candidate.  Trio is deterministic
    # (soft-agreement clustering via HDBSCAN) and cheap (~100ms at N=320).
    if verbose:
        residuals, provenance = _generate_carrier_residuals(
            probs_k, H, A, low_idx_list,
            cleanness_threshold=cleanness_threshold,
            verbose=True)
        n_examined = len(provenance)
        n_normal = sum(1 for e in provenance if e['partner_kind'] == 'normal')
        print(f'[late-rescue] _generate_carrier_residuals: '
              f'{n_examined} (sample, slot, subtractor) triples examined '
              f'({n_normal} normal, {n_examined - n_normal} skipped), '
              f'{len(residuals)} residuals accepted:')
        for entry in provenance:
            base = (f'    s={entry["sample_idx"]:>3d} slot={entry["slot"]} '
                    f'low={entry["low_idx"]} sub={entry["partner_idx"]} '
                    f'kind={entry["partner_kind"]:<11s}')
            if not entry['accepted']:
                if entry['partner_kind'] == 'normal':
                    print(f'{base} cleanness={entry["cleanness"]:.3f} '
                          f'-- REJECTED (cleanness < threshold)')
                # 'self_low' / 'low_carrier' skipped without printing each
                # — they're high-volume and uninformative; just count them
                # in the header line.
                continue
            r = entry['residual']
            hams = [_hamming_pct_kept(r, H[k]) for k in range(K)]
            ham_str = ', '.join(f'H{k}={h:5.2f}%' for k, h in enumerate(hams))
            print(f'{base} cleanness={entry["cleanness"]:.3f} ACCEPTED  '
                  f'[{ham_str}]')
    else:
        residuals = _generate_carrier_residuals(
            probs_k, H, A, low_idx_list,
            cleanness_threshold=cleanness_threshold)

    # Re-generate trio candidates fresh (deterministic, ~100ms at N=320).
    trio_candidates = _trio_recovery_candidate_haps(probs_k, verbose=False)
    if verbose:
        print(f'[late-rescue] trio produced {trio_candidates.shape[0]} '
              f'candidates.  Hamming-% to current H:')
        for ti in range(trio_candidates.shape[0]):
            hams = [_hamming_pct_kept(trio_candidates[ti], H[k])
                    for k in range(K)]
            ham_str = ', '.join(f'H{k}={h:5.2f}%' for k, h in enumerate(hams))
            print(f'    trio_cand[{ti}]: [{ham_str}]')

    # Also re-generate v6 pairwise common-hap candidates fresh.  Pairwise
    # is deterministic given probs_k and covers complementary failure
    # modes to trio (see pairwise_common_hap.py and the seed-stage call
    # at the start of _grow_K_with_recovery).  This is the rescue-pool
    # addition that distinguishes V6 from V3 in the integration test;
    # on the 50-block sample it was never triggered because no block
    # produced a low-carrier founder in production output, but it's the
    # right rescue mechanism for the chr6:23624234-style pathological
    # cases that motivated late-rescue in the first place.
    if PAIRWISE_RECOVERY_ENABLED:
        pairwise_candidates_list = bhd_pairwise.pairwise_recovery_candidate_haps(
            probs_k, verbose=False)
    else:
        pairwise_candidates_list = []
    if verbose:
        print(f'[late-rescue] pairwise produced '
              f'{len(pairwise_candidates_list)} candidates.  '
              f'Hamming-% to current H:')
        for pi in range(len(pairwise_candidates_list)):
            hams = [_hamming_pct_kept(pairwise_candidates_list[pi], H[k])
                    for k in range(K)]
            ham_str = ', '.join(f'H{k}={h:5.2f}%' for k, h in enumerate(hams))
            print(f'    pairwise_cand[{pi}]: [{ham_str}]')

    # Combine sources, dedup against current H and against each other
    # at 0.5% (only collapse near-exact duplicates so candidates that
    # differ at a few sites are kept as distinct pool members).
    raw_pool_candidates = (list(residuals)
                           + [trio_candidates[ti]
                              for ti in range(trio_candidates.shape[0])]
                           + pairwise_candidates_list)
    if not raw_pool_candidates:
        if verbose:
            print(f'[late-rescue] no candidates to admit — skipping')
        return H, A, costs, wcs, NLL

    H_list = [H[k] for k in range(K)]
    # The dedup-vs-current-H decision is order-independent (a candidate within
    # 0.5% Hamming of ANY current founder is dropped regardless of accept
    # order), so flag all such candidates in one parallel kernel pass (prange
    # over candidates x founders) instead of the nested Python loop.  The
    # dedup-vs-already-accepted check is inherently order-dependent and stays
    # sequential, but it now only runs over the H-survivors with the same
    # cheap early-break.  The resulting new_candidates is identical to the old
    # loop: the per-pair Hamming-% is computed the same way
    # (_dedup_vs_refset_kernel mirrors _hamming_pct_kept exactly), and a
    # candidate is admitted iff it duplicates neither H nor an earlier accept.
    raw_arr = np.empty((len(raw_pool_candidates), L_kept), dtype=np.int64)
    for ci in range(len(raw_pool_candidates)):
        raw_arr[ci] = np.asarray(raw_pool_candidates[ci])
    if K > 0:
        H_ref = np.ascontiguousarray(np.asarray(H_list), dtype=np.int64)
    else:
        H_ref = np.empty((0, L_kept), dtype=np.int64)
    dup_vs_H = _dedup_vs_refset_kernel(raw_arr, H_ref, 0.5)
    new_candidates = []
    for ci in range(len(raw_pool_candidates)):
        if dup_vs_H[ci]:
            continue
        cand = raw_pool_candidates[ci]
        is_dup = False
        for nc in new_candidates:
            if _hamming_pct_kept(cand, nc) < 0.5:
                is_dup = True
                break
        if not is_dup:
            new_candidates.append(cand)

    if not new_candidates:
        if verbose:
            print(f'[late-rescue] all candidates duplicate current H '
                  f'or each other — skipping')
        return H, A, costs, wcs, NLL

    if verbose:
        print(f'[late-rescue] {len(raw_pool_candidates)} raw candidates → '
              f'{len(new_candidates)} after dedup against H + each other '
              f'(threshold 0.5%)')

    # Pool = current H + new candidates
    pool = H_list + new_candidates

    # Build the late-rescue PoolEmissionCache.  All _greedy_bic_select /
    # _swap_refine / _bic_prune calls below evaluate subsets of this
    # fixed pool, so we precompute the Viterbi emission tensor once.
    rescue_cache = PoolEmissionCache(pool, probs_k, lam=lam)

    # Compute current BIC for comparison
    cc = _compute_cc(cc_scale, N, L_kept, use_log_bic=use_log_bic)
    BIC_orig = _compute_bic(K, NLL, cc)

    # Greedy BIC forward selection on enriched pool.  max_k = K + 1
    # allows BIC to grow K by 1, OR shrink K by stopping early when a
    # smaller subset has lower BIC.
    sel_indices, sel_haps, sel_nll = _greedy_bic_select(
        rescue_cache,
        cc_scale=cc_scale, max_k=K + 1,
        use_log_bic=use_log_bic, verbose=verbose)

    if not sel_haps:
        if verbose:
            print(f'[late-rescue] forward selection picked K=0 — '
                  f'unexpected, keeping original')
        return H, A, costs, wcs, NLL

    # Swap refinement (1-for-1 swap): for each selected slot, try every
    # pool member as a replacement.  Accepts a swap if NLL improves by
    # more than tolerance.  Mirrors the swap step inside
    # _subtraction_recovery_round_loop.  This is the operation that
    # displaces a chimera from its slot when a clean carrier residual
    # (= the missing truth founder) is in the pool.
    sel_indices, sel_haps, sel_nll, n_swaps = _swap_refine(
        rescue_cache, sel_indices, sel_nll,
        nll_tolerance=RECOVERY_SWAP_NLL_TOLERANCE,
        max_passes=10, verbose=verbose)

    # BIC-pruning: after a swap pulls in a truth-near hap, the
    # secondary chimera that was absorbing 'overflow' carriers (e.g.
    # chim_6 at chr6:23624234, ~21 carriers fitting truth_1 with 3%
    # drift) becomes redundant — the truth-near slot can absorb its
    # own carriers cleanly, and the remaining carriers' true partners
    # are the existing truths.  Drops any hap whose NLL contribution
    # falls below cc/2.  This is what actually gives the K=7 → K=6
    # BIC win.
    sel_indices, sel_haps, sel_nll, n_dropped = _bic_prune(
        rescue_cache, sel_indices,
        cc_scale=cc_scale, use_log_bic=use_log_bic, verbose=verbose)

    if not sel_haps:
        if verbose:
            print(f'[late-rescue] post-swap-prune picked K=0 — '
                  f'unexpected, keeping original')
        return H, A, costs, wcs, NLL

    # Refit at chosen K (forward+swap+prune used FIXED haps; refit lets
    # CD drift candidate haps toward the actual truth bits.  In the
    # chr6:23624234 case, after swap pulls truth_4 (= a clean carrier
    # residual) into slot 5 and prune drops chim_6, refit polishes the
    # K=6 state).
    H_new = np.array(sel_haps, dtype=np.int64)
    H_new, A_new, costs_new, wcs_new, n_iter_new, NLL_new = _fit_at_fixed_K(
        probs_k, H_new, lam, max_iter=max_iter)
    K_new = H_new.shape[0]
    BIC_new = _compute_bic(K_new, NLL_new, cc)

    if verbose:
        swap_str = f'{n_swaps} swap{"s" if n_swaps != 1 else ""}'
        prune_str = f'{n_dropped} prune{"s" if n_dropped != 1 else ""}'
        print(f'[late-rescue] orig: K={K}, NLL={NLL:.1f}, BIC={BIC_orig:.1f}')
        print(f'[late-rescue] new:  K={K_new}, NLL={NLL_new:.1f}, '
              f'BIC={BIC_new:.1f} (after {swap_str}, {prune_str})')

    if BIC_new < BIC_orig - 0.1:
        if verbose:
            print(f'[late-rescue] BIC improved by {BIC_orig - BIC_new:.1f} '
                  f'— ACCEPT')
        return H_new, A_new, costs_new, wcs_new, NLL_new
    if verbose:
        print(f'[late-rescue] BIC did not improve '
              f'(delta={BIC_orig - BIC_new:+.1f}) — KEEP ORIGINAL')
    return H, A, costs, wcs, NLL