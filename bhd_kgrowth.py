# bhd_kgrowth.py - K-growth founder discovery for block haplotype reconstruction.
#
# The coordinate-descent core that grows the founder count K within a single
# block: medoid-seeded initialisation (_initial_kgrowth_with_medoids,
# _soft_cluster_seed_haps), the iterative K-growth coordinate descent (_grow_K),
# and the recovery-augmented driver (_grow_K_with_recovery) that folds in
# subtraction / residual-trio / late-low-carrier rescue plus BIC selection.
#
# Extracted from block_haplotypes.py; block_haplotypes.generate_haplotypes_block
# calls _grow_K_with_recovery as its per-block founder-discovery entry point.

import numpy as np

import dynamic_threads
import bhd_pairwise

from bhd_config import (
    K_MEDOID_STARTS_DEFAULT,
    MEDOID_MIN_N_FOR_MULTISTART,
    PAIRWISE_RECOVERY_ENABLED,
    RECOVERY_CLEANNESS_THRESHOLD,
    RECOVERY_HAPS_EQUAL_EPS_PCT,
    RECOVERY_INTRA_ROUND_DEDUP_PCT,
    RECOVERY_MAX_K,
    RECOVERY_MAX_OUTER_ITERATIONS,
    RECOVERY_MAX_ROUNDS,
    RECOVERY_MIXTURE_K_MAX,
    RECOVERY_MIXTURE_N_RESTARTS,
    RECOVERY_MIXTURE_PATIENCE,
    RECOVERY_MIXTURE_RNG_SEED,
    RECOVERY_OUTER_CC_SCALE,
    RECOVERY_SWAP_NLL_TOLERANCE,
    RESIDUAL_TRIO_ENABLED,
    SEED_SOFT_MIN_CLUSTER_SIZE,
    TRIO_RECOVERY_ENABLED,
)
from bhd_kernels import (
    _init_hap_from_sample_dosage,
    _select_initial_seed,
)
from bhd_fit import (
    _compute_bic,
    _compute_cc,
    _fit_at_fixed_K,
)
from bhd_pool import PoolEmissionCache
from bhd_trio import _trio_recovery_candidate_haps
from bhd_recovery import (
    _subtraction_recovery_round_loop,
    _late_low_carrier_rescue,
    _residual_trio_rescue,
)
from bhd_recovery_select import (
    _greedy_bic_select,
    _hamming_pct_kept,
    _haps_equal,
)


def _grow_K(probs_k, kept_mask_full, lam,
            wildcard_mass_threshold=0.0,
            min_relative_improvement=0.10,
            K_max=10,
            max_iter_per_K=50,
            known_haps_full=None,
            cc_scale=0.5,
            use_log_bic=False,
            min_nll_improvement=1e-6,
            H_init=None):
    """Iteratively grow K, starting at K=0 (empty founder set), seeding
    each new founder from the current worst-fit sample's subtraction
    against existing founders.  Stops when either:
      - a candidate new founder fails to improve the BIC score
        (NLL improvement does not exceed the per-founder complexity cost), OR
      - K_max is reached.

    Acceptance criterion history:
      v1: relative wildcard-mass improvement >= 10%.  Rejected real
          founders whose individual contribution was small but whose
          combination with later founders unlocked substantial
          improvement.
      v2: strict-positive absolute wildcard-mass improvement (>= 1e-6).
          Fixed the v1 problem but introduced a new one: when a new
          real founder lets samples upgrade from (close-to-truth, W) to
          (exact-truth, W), the wildcard slot count is unchanged (still
          1 per upgraded sample) even though NLL drops substantially.
          The wildcard-mass criterion missed these improvements.
      v3: strict-positive NLL improvement (>= 1e-6).  Fixed v2 but
          accepted ANY positive NLL improvement, including spurious
          K-additions that absorbed a small amount of noise.  This
          showed up in the benchmark as small but non-zero K=7+ blocks
          past truth K=6, with reduced quality at the over-grown K.
      v4: linear-BIC-based acceptance.  A new founder is
          accepted iff adding it strictly reduces the BIC score
              BIC(K) = K * cc + 2 * NLL_K
          where cc = cc_scale * (L_kept/200) * N is the per-founder
          complexity cost (linear in N as in the project's
          beam_search_core / chimera_resolution standard).  This
          requires NLL_improvement > cc/2 to accept, calibrated so
          spurious noise-absorbing founders are rejected while real
          founders (which typically save thousands of NLL) easily pass.

          Linear BIC is preferred over standard log-BIC for the same
          reason as in beam_search_core: log(N) scaling is too weak
          when N is large, allowing founder explosion.  See
          chimera_resolution.compute_cc for the project-wide formula.

          v4 still had a top-of-loop early-stop on wildcard_mass <=
          wildcard_mass_threshold (default 0.0).  See v5 for why it
          was removed.
      v5 (current): wildcard-mass-based early-stop REMOVED.  BIC alone
          (v4 acceptance criterion) decides when to stop.  Diagnosed
          on chr3:27772468 (a 4/6 K-collapse case): wildcard_mass=0
          was firing at K=4 even though BIC overwhelmingly justified
          continuing — truth K=6 had BIC=379 while alg K=4 had
          BIC=12585, a 12,200-unit gap.  The mechanism: with LAM=0.5
          and L_kept=200, the wildcard penalty per slot is ≈100 NLL,
          so a sample with (real, real) cost up to ≈100 will pick
          (real, real) over (real, W) even when the (real, real) fit
          is terrible.  wm=0 then means "no sample chose a wildcard
          pair under current cost arithmetic", NOT "every sample is
          well fit".  K-growth was stopping on a misleading signal.
          BIC's NLL_improvement > cc/2 check is the principled stop;
          if all samples are truly well-fit, the next candidate's
          NLL drop will be small and BIC will reject naturally.

    Arguments:
        probs_k: (N, L_kept, 3) — kept-site posteriors
        kept_mask_full: (L_full,) bool — for downstream tracking; not used here
        lam:    wildcard penalty
        wildcard_mass_threshold: float — RETAINED for backward
            compatibility and signature stability.  Since v5, this
            parameter does NOT control K-growth's stopping behavior
            (BIC alone decides when to stop).  It is still consumed
            elsewhere in the pipeline (e.g. the uncertainty flag in
            generate_haplotypes_block uses
            `wildcard_mass > 2 * wildcard_mass_threshold` to mark
            blocks with high wildcard usage as untrustworthy).
        min_relative_improvement: float — RETAINED for backward compatibility
            but no longer used; current criterion is BIC-based.
        K_max: int — hard cap on K
        known_haps_full: list of (L_full,) binary arrays or None — if given,
            these are used as the initial K founders and never updated.
            (Note: they're passed as full-length arrays; we slice to kept.)
            **Currently unused** — known_haplotypes integration is a v2
            feature; ignored for now.
        cc_scale: float — complexity-cost scale (per-founder per-sample
            per-200-SNPs).  Default 0.5.  Higher values penalise extra
            founders more strongly.

            Calibration history:
              cc_scale=0.5 (project-wide default in chimera_resolution.py
              and beam_search_core.py): too aggressive at the per-block
              EM stage.  Empirically rejected real founders saving
              50-80 NLL (e.g., founders with 20-30 carriers showing as
              "founder upgrade" type savings rather than full wildcard-
              slot reduction).  Benchmark dropped 70.8% → 69.5%
              all-found, with 250+ blocks regressing from K=6
              (mostly-recovered) to K=4 (multiple-founders-missed).

              cc_scale=0.05 (current default): threshold of cc/2 =
              0.05 * (L/200) * N / 2 ≈ 8 NLL for typical N=320, L=200.
              This is just above floating-point noise: rejects pure
              "noise absorption" K-additions saving < 8 NLL while
              preserving every realistic founder addition (real
              founders typically save 50+ NLL per K transition).

              cc_scale=0.0 effectively disables the BIC penalty,
              reverting to strict-positive NLL improvement criterion
              with only the min_nll_improvement floor in effect.

            Note: the project's default 0.5 in beam_search_core was
            calibrated for whole-genome long-haplotype assembly, where
            the complexity penalty needs to suppress recombinant-
            founder false-positives across thousands of blocks.  At
            the per-block EM stage we operate on a much smaller scale
            and need gentler regularisation.

            Update (May 2026): default reverted to 0.5 to match the
            project-wide standard, after diagnosing the K-growth /
            recovery oscillation at chr3:16378549.  The historical
            70.8% → 69.5% regression noted above (and the 250+ block
            K-collapse from cc_scale=0.5) was confounded by other
            pipeline bugs that have since been fixed:
              - The wm-stop bug in K-growth (fixed by removing the
                wildcard_mass <= threshold short-circuit; lifted
                benchmark to 99.82% all-found).
              - Viterbi-BIC subset selection in _final_cleanup
                (Step B), with a per-hap inclusion penalty ~7×
                stricter than discrete-CD's BIC, systematically
                dropped legitimate low-frequency-carrier founders
                (disabled).
              - Chimera pruning in _final_cleanup (Step D), with
                mean_delta protection scaled to carrier frequency,
                systematically dropped legitimate low-frequency
                founders (disabled).
              - Step C usage threshold of max(2, 1% of N) = 3 for
                N=320 dropped legitimate founders with usage = 2
                strands (lowered to 1).
            With those pipeline issues removed, the project-standard
            cc_scale=0.5 is the principled choice — it removes the
            asymmetry between K-growth and recovery's outer BIC, and
            relies on K-growth's BIC at cc/2 ≈ 80 NLL as the
            authoritative data-justification filter for each founder
            (with the recovery's mixture-derived candidate pool
            providing the diversity of seeds K-growth's worst-fit
            seeding alone might miss).
        use_log_bic: bool — if True, use standard BIC with log(N*L) scaling
            instead of linear scaling.  Default False (linear, project standard).
        min_nll_improvement: float — additional numerical-noise floor.
            Effective threshold is max(min_nll_improvement, cc/2).
        H_init: optional (K_init, L_kept) array — if provided, K-growth
            starts from these K_init founders rather than from K=0
            (empty set).  H_init is treated as MUTABLE: the initial
            _fit_at_fixed_K call refines them via coord descent before
            any growth attempts.  This supports the outer K-growth ↔
            recovery iteration in _grow_K_with_recovery, where each
            K-growth call starts from the previous recovery output.
            Default None = original empty-set behaviour.

    Returns:
        H:               final (K, L_kept)
        A:               final (N, 2)
        per_sample_cost: (N,)
        wildcard_slots:  (N,)
        K_final:         int
        wildcard_mass:   float in [0, 1]
        history:         list of (K, BIC, wildcard_mass, n_iter) per growth step
                         (BIC = K * cc + 2 * NLL with the same cc used
                         in the acceptance criterion; comparable across K)
    """
    N, L_kept, _ = probs_k.shape
    history = []

    # Defensive guard: N=0 (no samples) means there's nothing to fit or
    # grow.  Return an empty result early.  Without this, _grow_K would
    # later call _select_initial_seed in the K=0 → K=1 fallback path,
    # which crashes on `argmax of empty sequence` because there's no
    # sample to score for decisiveness.  In production, the upstream
    # generate_haplotypes_block guards N=0 at the top level, but _grow_K
    # is also called directly (e.g., from the outer recovery loop) so
    # the guard belongs here for defense in depth.
    if N == 0:
        H_out = (np.empty((0, L_kept), dtype=np.int64) if H_init is None
                 else np.asarray(H_init, dtype=np.int64).copy())
        return (H_out,
                np.empty((0, 2), dtype=np.int64),         # A
                np.empty((0,), dtype=np.float64),         # per_sample_cost
                np.empty((0,), dtype=np.int64),           # wildcard_slots
                H_out.shape[0],                            # K_final
                0.0,                                        # wildcard_mass
                history)

    # === BIC-based acceptance threshold ===
    # Linear BIC: cc = cc_scale * (L_kept/200) * N
    # Standard BIC: cc = cc_scale * log(N * L_kept) * (L_kept/200)
    # Acceptance criterion: BIC(K+1) < BIC(K)
    #   => K*cc + 2*NLL_K > (K+1)*cc + 2*NLL_{K+1}
    #   => 2*(NLL_K - NLL_{K+1}) > cc
    #   => NLL_improvement > cc/2
    # Effective threshold combines BIC term with a numerical-noise floor.
    cc = _compute_cc(cc_scale, N, L_kept, use_log_bic=use_log_bic)
    bic_threshold = cc / 2.0
    accept_threshold = max(min_nll_improvement, bic_threshold)

    # === K=K_init baseline: empty founder set, or given H_init ===
    #
    # Per philosophical principle: start with NO founders at all (or
    # with the given H_init founders), measure how badly we fit the
    # data with pure wildcards (or with H_init).  Each subsequent K is
    # grown by introducing one new founder, warm-starting existing
    # founders from the previous K's converged values, and re-running
    # coordinate descent (no founders are held fixed).  This gives every
    # K transition the same uniform shape (K → K+1 grow + fit + accept).
    #
    # When H_init is None or empty, the baseline is K=0 (empty founder
    # set, all samples assigned to (W, W)) — the original behaviour.
    # When H_init is provided, the baseline is K=K_init (founders
    # refined via coord descent) — used by _grow_K_with_recovery to
    # continue growth after a recovery pass produced a better starting
    # set.
    if H_init is None or len(H_init) == 0:
        H = np.empty((0, L_kept), dtype=np.int64)
    else:
        H = np.asarray(H_init, dtype=np.int64).copy()
        if H.shape[1] != L_kept:
            raise ValueError(
                f"H_init has L={H.shape[1]} but probs_k has L_kept={L_kept}")
    H, A, per_sample_cost, wildcard_slots, n_iter, nll = _fit_at_fixed_K(
        probs_k, H, lam, max_iter=max_iter_per_K)
    wildcard_mass = float(wildcard_slots.sum()) / max(2 * N, 1)
    # History entries record BIC = K*cc + 2*NLL so callers can compare
    # entries across different K values directly.  At fixed K this is
    # NLL + const, so it preserves within-K ordering; across K it
    # correctly accounts for the per-founder complexity penalty.
    history.append((H.shape[0], _compute_bic(H.shape[0], nll, cc),
                    wildcard_mass, n_iter))
    # Track NLL of the last accepted state — used as the comparison point
    # for the next K candidate's acceptance check.  Note: comparisons
    # use NLL_improvement vs cc/2 here, which is algebraically the same
    # as comparing BIC(K+1) vs BIC(K); we keep the NLL form for numerical
    # accuracy (avoids double-floating-point recombination).
    prev_nll = nll

    # === K-growth loop (handles K=0→1, K=1→2, ... uniformly) ===
    #
    # Stopping criteria (in priority order):
    #   1. K_max reached (safety cap)
    #   2. BIC reject: nll_improvement < cc/2 → adding this founder
    #      would not reduce K*cc + 2*NLL.  This is the principled
    #      "BIC no longer improves" stop and lives further down the
    #      loop, after we've fit the candidate.
    #
    # Earlier versions also had a wildcard-mass-based early-stop here
    # (`if wildcard_mass <= wildcard_mass_threshold: break`), with
    # default threshold 0.0.  This was REMOVED after the
    # chr3:27772468 diagnostic: wildcard_mass=0 does not mean "all
    # samples well-fit"; it means "no sample chose a (real, W) or
    # (W, W) pair under the current LAM-vs-real-pair-cost arithmetic".
    # A sample can be assigned to (real, real) with per-sample-cost
    # 100+ NLL units (terrible fit) and still produce wm=0, because
    # (real, real) at cost 100 is still cheaper than (real, W) at
    # cost ≈ best_real + lam*L.  Stopping K-growth on wm=0 in such
    # cases caused premature exit at K_alg < K_truth even when BIC
    # overwhelmingly justified continuing (truth K=6 had BIC=379 vs
    # alg K=4 BIC=12585 on chr3:27772468 — a 12,200-unit gap).
    #
    # The wildcard_mass_threshold parameter is retained in the
    # signature for backward compatibility and is still used elsewhere
    # in the codebase (e.g. the uncertainty flag in
    # generate_haplotypes_block), but does NOT affect K-growth's stop.
    while True:
        # Re-check thread allocation at each K-growth step.
        dynamic_threads.apply_dynamic_threads()
        K_cur = H.shape[0]
        if K_cur >= K_max:
            break

        # Seed new founder via SUBTRACTION from the worst-fit sample.
        # Rationale: the worst-fit sample is one whose pair (and thus
        # whose two real founders) the current set fails to explain.
        # If we hypothesise that ONE of its strands is an existing
        # founder F_i, the OTHER strand has a determined value at sites
        # where dosage and F_i agree, and is ambiguous at sites where
        # they conflict (those sites become MASK or rounded).  We try
        # each existing founder F_i as the "known strand" hypothesis,
        # producing K_cur candidate other-strand haps; pick the one
        # most distinct from existing founders (max min-Hamming).  This
        # gives a real-founder-hypothesis seed (per principle 8) rather
        # than a hybrid-average seed.
        #
        # If the worst sample has 2 wildcard strands (no real founder
        # hypothesis), or all samples are pure wildcards, fall back to
        # the dosage / 2 heuristic on the most-decisive sample.
        worst_candidate_mask = (wildcard_slots < 2)        # exclude (W, W)
        if not worst_candidate_mask.any():
            # All samples are (W, W) — fall back to most-decisive sample
            worst_idx = _select_initial_seed(probs_k, kept_mask=None)
            new_h = _init_hap_from_sample_dosage(
                probs_k, worst_idx, kept_mask=None)
            # Single candidate in this branch, no picker needed.  Run
            # CD once on the chosen seed.
            H_try = np.vstack([H, new_h[None, :]])         # (K+1, L_kept)
            H_try, A_try, cost_try, wcs_try, n_iter_try, nll_try = \
                _fit_at_fixed_K(probs_k, H_try, lam, max_iter=max_iter_per_K)
            wm_try = float(wcs_try.sum()) / max(2 * N, 1)
        else:
            adjusted_cost = np.where(worst_candidate_mask,
                                     per_sample_cost, -np.inf)
            worst_idx = int(adjusted_cost.argmax())
            # Subtraction-based seed candidates
            worst_dosage = probs_k[worst_idx].argmax(axis=1)   # (L_kept,)
            seed_candidates = []
            for i in range(K_cur):
                # Implied other strand: dosage - F_i, clipped to [0, 1].
                # Where this is fractional / ambiguous (e.g., dosage=2 but
                # F_i = 0 implies other = 2 which is impossible), the
                # subtraction is invalid at that site — we fall back to
                # the data's argmax-favored single-strand value (i.e.,
                # if dosage = 2 we set other_strand = 1; if dosage = 0,
                # other_strand = 0).
                other = worst_dosage - H[i]
                # Clip values: anything outside {0, 1} indicates the
                # F_i hypothesis is inconsistent at that site.  Project
                # to nearest valid {0, 1} value to keep going.
                other = np.clip(other, 0, 1).astype(np.int64)
                seed_candidates.append(other)
            # Also include the simple dosage / 2 heuristic as a fallback
            seed_candidates.append(_init_hap_from_sample_dosage(
                probs_k, worst_idx, kept_mask=None))
            # Pick the candidate that's most distinct from existing
            # founders (maximises min-Hamming to any existing F_i)
            #
            # HISTORICAL NOTE — old picker and the bug it caused:
            # Originally we picked by max-min-Hamming (the candidate
            # furthest from any existing founder).  This optimises the
            # WRONG criterion: hap-space distance, not data fit.  At
            # blocks where K-growth has converged into a "chimera basin"
            # (existing founders are weighted-averages of multiple
            # truths rather than any single truth), the max-distance
            # candidate is typically a chimera-residual that no sample
            # in the data actually wants as a strand.  Such a seed has
            # zero carriers, M-step cannot update it, the trial CD fit
            # produces dNLL ≈ 0, and BIC rejects K_cur+1.  The algorithm
            # then halts at the local minimum.  Diagnostic on
            # chr1:34921614 confirmed: at K=6 stuck-NLL=26777, the
            # max-distance candidate gave post-CD NLL=26777 (no change),
            # but a DIFFERENT candidate in the same set gave post-CD
            # NLL=19131 (a 7600-unit drop, escaping the basin).  Truth
            # NLL on this block is 551 — the algorithm is 49x above
            # truth in NLL because the picker selected the "different
            # but useless" candidate over the "less different but
            # actually fits the data" one.
            #
            # NEW picker: evaluate every candidate by running CD and
            # picking the one with lowest post-CD NLL.  This trades
            # ~K extra CD fits per K-growth step for correctness — at
            # K_max=10 the total cost is 1+2+...+10 = 55 fits instead
            # of 10, but the resulting K-growth trajectory escapes
            # local minima that the old picker couldn't.
            #
            # Determinism is preserved: candidate generation is
            # deterministic (np.clip arithmetic + dosage-init), each
            # CD trial is deterministic (given the same probs_k and
            # initial H), the argmin is deterministic with stable
            # tie-breaking via candidate-index order.  We tie-break
            # on max-min-Hamming (the legacy criterion) when post-CD
            # NLL values are equal to within a small tolerance, so
            # that in the limit of no NLL difference (e.g. K=0 case
            # which doesn't enter this branch anyway) we recover the
            # legacy behaviour.
            #
            # Note on BIC vs NLL: every candidate at this branch has
            # the same target K = K_cur+1, so BIC = K*cc + 2*NLL
            # differs from NLL only by the constant (K_cur+1)*cc.
            # Picking by lowest NLL is therefore identical to picking
            # by lowest BIC — no need to add the cc term here.
            best_NLL = float('inf')
            best_min_d = -1.0
            best_seed = None
            best_fit = None
            for cand in seed_candidates:
                H_cand = np.vstack([H, cand[None, :]])
                fit_state = _fit_at_fixed_K(probs_k, H_cand, lam,
                                             max_iter=max_iter_per_K)
                cand_nll = float(fit_state[5])     # nll_try is index 5
                # Hamming to each existing founder, for tie-break
                ds = [float(np.mean(cand != H[i])) for i in range(K_cur)]
                min_d = min(ds) if ds else 1.0
                # Pick by NLL primarily, max-min-Hamming as tiebreak
                if (cand_nll < best_NLL - 1e-9 or
                        (abs(cand_nll - best_NLL) <= 1e-9 and
                         min_d > best_min_d)):
                    best_NLL = cand_nll
                    best_min_d = min_d
                    best_seed = cand
                    best_fit = fit_state
            new_h = best_seed
            # Reuse the captured fit for the chosen candidate — no
            # need to refit.
            H_try, A_try, cost_try, wcs_try, n_iter_try, nll_try = best_fit
            wm_try = float(wcs_try.sum()) / max(2 * N, 1)

        # Did the new founder reduce BIC?
        #
        # BIC(K) = K * cc + 2 * NLL_K, where cc is the per-founder
        # complexity cost (linear in N as in the project's
        # beam_search_core / chimera_resolution standard).  Adding a
        # founder reduces BIC iff
        #     NLL_improvement = NLL_K - NLL_{K+1} > cc / 2
        # i.e., the likelihood gain (in NLL units) outweighs half the
        # complexity cost (the factor of 2 cancels with the 2*NLL form
        # of BIC).
        #
        # This replaces v3's "any positive NLL improvement" criterion
        # which incorrectly accepted spurious K-additions absorbing
        # tiny amounts of noise, producing K=truth+1 or K=truth+2
        # blocks past the real K.  See benchmark_stage3_em K-distribution
        # showing K=7+ blocks past truth K=6 — those are now rejected.
        #
        # NLL captures every source of fit improvement:
        #   - wildcard-slot reductions (samples switching from (W,W) to
        #     (real, W) or to (real, real))
        #   - better-fitting real founders (samples switching from
        #     (close-to-truth, W) to (exact-truth, W) — same number of
        #     wildcard slots but lower per-site data-fit cost)
        #   - pair-assignment reorganisations that improve overall fit
        #
        # See trace_discrete_block on chr3:26562266 — K=4→K=5 reduced
        # NLL by 1380 (real improvement: hap4 became exact t3, letting
        # (t1,t3) samples upgrade from (hap1, W) to (hap4, W)).  With
        # cc=160 and threshold=80, this passes easily.
        #
        # The min_wildcard_relative_improvement parameter is preserved
        # in the signature for backward compatibility but is unused.
        nll_improvement = prev_nll - nll_try
        history.append((K_cur + 1,
                        _compute_bic(K_cur + 1, nll_try, cc),
                        wm_try, n_iter_try))
        if nll_improvement < accept_threshold:
            # New founder didn't sufficiently improve BIC — reject and stop
            break

        # Accept
        H = H_try
        A = A_try
        per_sample_cost = cost_try
        wildcard_slots = wcs_try
        wildcard_mass = wm_try
        prev_nll = nll_try

    return H, A, per_sample_cost, wildcard_slots, H.shape[0], wildcard_mass, history


# =============================================================================
# SOFT-CLUSTERING SEEDS FOR INITIAL-K-GROWTH MULTI-START
# =============================================================================
# Generates the diverse K=1 seed haps for _initial_kgrowth_with_medoids by
# clustering the samples on the posterior soft-agreement similarity and
# emitting one denoised pooled-alt consensus seed per cluster.  This keeps
# the genotype posteriors rather than hard-calling each sample's dosage,
# which preserves the low-read-depth signal.  hdbscan and bhd_kernels are
# imported lazily.

def _soft_cluster_seed_haps(probs_k, n_seeds,
                              min_cluster_size=SEED_SOFT_MIN_CLUSTER_SIZE,
                              verbose=False):
    """Generate up to ``n_seeds`` diverse K=1 seed haps by soft clustering.

    Clusters samples on the expected-genotype-agreement similarity
    (bhd_kernels.soft_agreement_similarity) via HDBSCAN on the derived
    distance (S.max() - S), ranks clusters by membership size (largest
    first), and returns one binary seed hap per cluster (up to n_seeds) as
    the per-site pooled-alt consensus of the cluster
    (bhd_kernels.alt_fractions averaged over members -> pooled_alt_to_hap).

    A homozygous-looking cluster yields a clean founder readout; a
    heterozygous (pair-type) cluster yields the same forced-bits / majority
    readout the per-sample seed would, but denoised by pooling over the
    whole cluster — a much better K-growth starting point at low read depth.

    Arguments:
        probs_k: (N, L, 3) genotype posteriors restricted to kept sites
        n_seeds: int — maximum number of seed haps to return (the multi-
            start branch count)
        min_cluster_size: int — HDBSCAN minimum cluster size
        verbose: bool

    Returns:
        list of (L,) np.int64 seed haps, length in [0, n_seeds].  May be
        shorter than n_seeds (or empty) when HDBSCAN finds fewer clusters;
        the caller falls back to its single-branch path when empty.
    """
    import bhd_kernels as _bk
    import hdbscan

    N, L = probs_k.shape[0], probs_k.shape[1]

    # Soft-agreement similarity -> precomputed distance for HDBSCAN.
    S = _bk.soft_agreement_similarity(probs_k)            # (N, N) in [0, 1]
    dist = (S.max() - S)
    np.fill_diagonal(dist, 0.0)
    dist = np.ascontiguousarray(dist, dtype=np.float64)

    labels = hdbscan.HDBSCAN(
        metric='precomputed',
        min_cluster_size=int(min_cluster_size),
    ).fit(dist).labels_

    # Rank clusters by size, largest first (label -1 is HDBSCAN noise).
    clusters = [(c, np.where(labels == c)[0])
                for c in np.unique(labels) if c != -1]
    clusters.sort(key=lambda cm: -cm[1].shape[0])

    if verbose:
        sizes = [int(mem.shape[0]) for _c, mem in clusters]
        print(f'[seed-soft] N={N}, clusters={len(clusters)} sizes={sizes}, '
              f'taking up to {n_seeds}')

    alt = _bk.alt_fractions(probs_k)                      # (N, L) E[alt dose]/2
    seeds = []
    for _c, mem in clusters[:n_seeds]:
        pooled = alt[mem].mean(axis=0)                    # (L,)
        seeds.append(_bk.pooled_alt_to_hap(pooled).astype(np.int64))
    return seeds


def _initial_kgrowth_with_medoids(probs_k, kept_mask_full, lam,
                                    n_medoid_starts,
                                    wildcard_mass_threshold,
                                    min_relative_improvement,
                                    K_max,
                                    max_iter_per_K,
                                    known_haps_full,
                                    cc_scale,
                                    use_log_bic,
                                    min_nll_improvement,
                                    H_trio_seed=None,
                                    run_per_branch_recovery=False,
                                    recovery_outer_cc_scale=RECOVERY_OUTER_CC_SCALE,
                                    recovery_max_K=RECOVERY_MAX_K,
                                    recovery_max_rounds=RECOVERY_MAX_ROUNDS,
                                    recovery_intra_round_dedup_pct=RECOVERY_INTRA_ROUND_DEDUP_PCT,
                                    recovery_mixture_K_max=RECOVERY_MIXTURE_K_MAX,
                                    recovery_mixture_n_restarts=RECOVERY_MIXTURE_N_RESTARTS,
                                    recovery_mixture_seed_base=RECOVERY_MIXTURE_RNG_SEED,
                                    recovery_mixture_patience=RECOVERY_MIXTURE_PATIENCE,
                                    recovery_cleanness_threshold=RECOVERY_CLEANNESS_THRESHOLD,
                                    recovery_swap_nll_tolerance=RECOVERY_SWAP_NLL_TOLERANCE,
                                    recovery_haps_equal_eps_pct=RECOVERY_HAPS_EQUAL_EPS_PCT,
                                    verbose=False):
    """Run K-growth (optionally seeded from H_trio_seed) with k-medoid
    multi-start over sample seeds, plus optional per-branch subtraction
    recovery before BIC arbitration.

    Builds N sample-dosage seeds (one per sample), computes the (N, N)
    Hamming distance matrix between seeds, runs PAM at K=n_medoid_starts
    to pick diverse seed samples, then for each medoid m runs a full
    branch:

      a. H_init for branch m = stack([H_trio_seed, seed_array[m:m+1]])
         when H_trio_seed is non-empty, else just seed_array[m:m+1].
      b. Run K-growth from H_init.
      c. If run_per_branch_recovery: run subtraction-recovery on the
         K-growth output, then re-fit at fixed K to populate the full
         result tuple.
      d. Compute branch BIC.

    Returns the trajectory with lowest final BIC = K_final * cc + 2 * NLL.

    Selection by BIC (not raw NLL) properly handles the case where
    different medoids land at different K_final values: a trajectory
    that grew to K=8 with marginally lower NLL than one at K=6 will
    correctly lose if those two extra founders don't pay their
    complexity cost.

    H_trio_seed parameter:
      When provided (non-empty), serves as a SHARED prefix in every
      branch's H_init.  All branches start with the trio-derived
      founder set and then add one medoid-derived candidate hap on
      top, so different branches differ only in their candidate
      "K_trio+1-th founder" hypothesis.  CD inside K-growth refines
      the medoid candidate (and possibly the trio haps) to fit the
      data; per-branch recovery (when enabled) then runs subtraction-
      recovery to find any additional founders.  BIC arbitration
      across branches picks the winner.

      Plus a NO-MEDOID BASELINE branch is run alongside (only when
      H_trio_seed is non-empty): it starts with H_init = H_trio_seed
      alone (size K_trio).  This branch's K-growth first does CD at
      K_trio, then tries K_trio+1 via worst-fit-sample seeding with
      proper BIC comparison (does K_trio+1 improve over K_trio?).
      Required because the medoid branches all start at K_trio+1
      and never compare against K_trio — so degenerate K_trio+1
      attractors that happen to have the same NLL as truth K_trio
      (which occur on all-hets symmetry cases) would otherwise win
      every branch and trap us in a wrong-K basin.  The baseline
      branch is the only one that gets to test K_trio+1 → K_trio.

      When None or empty, falls back to legacy behavior: each branch's
      H_init is just seed_array[m:m+1] (K=0 -> K=1 starting set).
      No no-medoid baseline branch in this case (would be K=0 which
      is meaningless).

    run_per_branch_recovery parameter:
      When True, runs _subtraction_recovery_round_loop on each branch's
      K-growth output before computing branch BIC.  This gives each
      branch a chance to BIC-discover additional founders via mixture
      recovery before cross-branch arbitration — without it, a branch
      with marginally better K-growth-only BIC would win even if
      another branch had recoverable founders that would have flipped
      the ranking.

      The recovery_* parameters are passed through to
      _subtraction_recovery_round_loop and are ignored when
      run_per_branch_recovery=False.

    Arguments mirror _grow_K (which is called per medoid).  When
    n_medoid_starts <= 1 OR N < MEDOID_MIN_N_FOR_MULTISTART, falls
    back to a single branch (using H_trio_seed if provided, else None
    for K=0 start).  Per-branch recovery still runs in this fallback
    when enabled — there's just only one branch to arbitrate over.

    Returns: same tuple as _grow_K:
      (H, A, per_sample_cost, wildcard_slots, K_final, wildcard_mass,
       history)

    History follows _grow_K's format — a list of (K, BIC, wildcard_mass,
    n_iter) tuples, one per K-growth step inside the WINNING branch.
    Recovery doesn't add growth steps so its effects (which may change
    K) are not recorded in history; verbose logging shows them via the
    [recovery] / [medoid] tag prints.
    """
    N, L_kept, _ = probs_k.shape

    has_trio = (H_trio_seed is not None) and (H_trio_seed.shape[0] >= 1)
    K_trio = int(H_trio_seed.shape[0]) if has_trio else 0

    def _process_one_branch(H_init):
        """Run K-growth + optional subtraction-recovery on a given H_init.
        Returns the same 7-tuple as _grow_K."""
        result = _grow_K(
            probs_k, kept_mask_full, lam,
            wildcard_mass_threshold=wildcard_mass_threshold,
            min_relative_improvement=min_relative_improvement,
            K_max=K_max,
            max_iter_per_K=max_iter_per_K,
            known_haps_full=known_haps_full,
            cc_scale=cc_scale,
            use_log_bic=use_log_bic,
            min_nll_improvement=min_nll_improvement,
            H_init=H_init)
        if not run_per_branch_recovery:
            return result
        # result tuple: (H, A, per_sample_cost, wildcard_slots,
        #                K_final, wildcard_mass, history)
        H_after_grow = result[0]
        if H_after_grow.shape[0] < 1:
            # K=0 result: nothing to subtract from, recovery is a no-op
            return result
        H_after_recov = _subtraction_recovery_round_loop(
            probs_k, H_after_grow, lam,
            outer_cc_scale=recovery_outer_cc_scale,
            max_K=recovery_max_K,
            max_rounds=recovery_max_rounds,
            max_iter_per_K=max_iter_per_K,
            intra_round_dedup_pct=recovery_intra_round_dedup_pct,
            mixture_K_max=recovery_mixture_K_max,
            mixture_n_restarts=recovery_mixture_n_restarts,
            mixture_seed_base=recovery_mixture_seed_base,
            mixture_patience=recovery_mixture_patience,
            cleanness_threshold=recovery_cleanness_threshold,
            swap_nll_tolerance=recovery_swap_nll_tolerance,
            haps_equal_eps_pct=recovery_haps_equal_eps_pct,
            use_log_bic=use_log_bic,
            verbose=verbose)
        if H_after_recov.shape[0] < 1:
            # Recovery returned empty (degenerate case) — keep K-growth result
            return result
        # Re-fit at fixed K to populate the full tuple after recovery's
        # internal CD may have changed things.  Note _fit_at_fixed_K
        # returns 6 elements; we reshape to the 7-tuple form _grow_K
        # uses by adding K_final and wildcard_mass.
        H_final, A_final, costs_final, wcs_final, _it, _nll_final = \
            _fit_at_fixed_K(probs_k, H_after_recov, lam,
                              max_iter=max_iter_per_K)
        K_final_recov = H_final.shape[0]
        wm_final = float(wcs_final.sum()) / max(2 * N, 1)
        # History follows _grow_K's contract (list of (K, BIC, wm,
        # n_iter) per growth step).  Recovery isn't a growth step so we
        # preserve the K-growth history unchanged.  Recovery's effects
        # on K and NLL appear via verbose [recovery] tag prints.
        return (H_final, A_final, costs_final, wcs_final,
                  K_final_recov, wm_final, list(result[6]))

    # Per-founder complexity cost for branch BIC comparison.  Must
    # match the cc used inside _grow_K (each branch uses the same cc
    # for its own acceptance criterion); this lets us compare final
    # solutions ACROSS branches at potentially different K.
    cc = _compute_cc(cc_scale, N, L_kept, use_log_bic=use_log_bic)

    # Single-branch fallback: too few samples or single-start requested.
    # Per-branch recovery (if enabled) still applies — we just have
    # only one branch to choose from.
    if n_medoid_starts <= 1 or N < MEDOID_MIN_N_FOR_MULTISTART:
        H_init = H_trio_seed if has_trio else None
        return _process_one_branch(H_init)

    # Cap medoid count at N (PAM degenerate otherwise)
    n_medoid_starts = min(n_medoid_starts, N)

    # Build the diverse K=1 seed haps for the branches.  Each entry of
    # seed_haps is a (L_kept,) binary hap; seed_labels[i] is a short tag
    # used only for verbose logging.  Seeds come from the posterior soft-
    # clustering front-end: cluster on the soft-agreement similarity and
    # use up to n_medoid_starts cluster pooled-alt consensuses as the
    # diverse seeds (a denoised, low-read-depth-robust analogue of the
    # per-sample argmax seed).
    soft_seeds = _soft_cluster_seed_haps(
        probs_k, n_medoid_starts,
        min_cluster_size=SEED_SOFT_MIN_CLUSTER_SIZE, verbose=verbose)
    if len(soft_seeds) == 0:
        # HDBSCAN found no clusters (e.g. too few samples per pair-type
        # at very low depth) — fall back to the single branch.
        if verbose:
            print('[medoid] soft clustering found no clusters — '
                  'single-branch fallback')
        H_init = H_trio_seed if has_trio else None
        return _process_one_branch(H_init)
    seed_haps = soft_seeds
    seed_labels = [f'soft cluster {i}' for i in range(len(seed_haps))]
    if verbose:
        if has_trio:
            print(f'[medoid] {len(seed_haps)} soft-cluster seeds, '
                  f'each branch H_init = stack([H_trio_seed '
                  f'(K={K_trio}), cluster_seed])')
        else:
            print(f'[medoid] {len(seed_haps)} soft-cluster seeds')

    # Run full per-branch processing from each medoid; keep the best
    # by final BIC.  BIC = K_final * cc + 2 * NLL_final correctly
    # penalises trajectories that grew to a larger K than the data
    # justifies, so a marginally lower NLL at K=8 will lose to a
    # slightly higher NLL at K=6 if the extra two founders aren't
    # paying their complexity cost.
    best_BIC = float('inf')
    best_result = None
    best_label = None

    # No-medoid baseline branch (only when has_trio).  This branch
    # starts with H_init = H_trio_seed alone (size K_trio), so
    # K-growth first runs CD at K_trio, then tries K_trio+1 via worst-
    # fit-sample seeding with proper BIC comparison (does K_trio+1
    # improve over K_trio?).  Without this baseline, the medoid
    # branches all start at K_trio+1 directly and never compare
    # against K_trio, so degenerate K_trio+1 attractors with same NLL
    # as truth K_trio (which happen on the all-hets symmetry case)
    # win every branch and the truth K_trio basin is unreachable.
    #
    # Note: this is NOT a redundant computation when the medoid
    # branches happen to also drop down to K_trio internally — those
    # branches start CD at K_trio+1 and only have K_trio+1 → K_trio+2
    # transitions to test, never K_trio+1 → K_trio.  The baseline
    # branch is the only one that gets to test K_trio+1 → K_trio.
    if has_trio:
        if verbose:
            print(f'[medoid] no-medoid baseline branch: H_init = '
                  f'H_trio_seed (K_trio={K_trio})')
        baseline_result = _process_one_branch(H_trio_seed)
        baseline_K = int(baseline_result[4])
        baseline_NLL = float(baseline_result[2].sum())
        baseline_BIC = _compute_bic(baseline_K, baseline_NLL, cc)
        if verbose:
            tag = ' + recovery' if run_per_branch_recovery else ''
            print(f'[medoid] no-medoid baseline{tag}: '
                  f'K_final={baseline_K}, NLL={baseline_NLL:.1f}, '
                  f'BIC={baseline_BIC:.1f}')
        best_BIC = baseline_BIC
        best_result = baseline_result
        best_label = 'no-medoid baseline'

    for seed_hap, label in zip(seed_haps, seed_labels):
        # Re-check thread allocation at the top of each medoid branch (each
        # branch runs its own per-branch recovery -- a heavy phase).
        dynamic_threads.apply_dynamic_threads()
        # Build per-branch H_init: trio_seed prefix + this branch's seed hap
        if has_trio:
            H_init = np.vstack([H_trio_seed, seed_hap[None, :]])
        else:
            H_init = seed_hap[None, :]
        result = _process_one_branch(H_init)
        # result tuple: (H, A, per_sample_cost, wildcard_slots,
        #                K_final, wildcard_mass, history)
        result_K = int(result[4])
        result_NLL = float(result[2].sum())
        result_BIC = _compute_bic(result_K, result_NLL, cc)
        if verbose:
            tag = ' + recovery' if run_per_branch_recovery else ''
            print(f'[medoid] start at {label}{tag}: '
                  f'K_final={result_K}, NLL={result_NLL:.1f}, '
                  f'BIC={result_BIC:.1f}')
        if result_BIC < best_BIC:
            best_BIC = result_BIC
            best_result = result
            best_label = label

    if verbose:
        print(f'[medoid] best trajectory: {best_label}, '
              f'BIC={best_BIC:.1f}')

    return best_result


# =============================================================================
# K-GROWTH WITH SUBTRACTION-RECOVERY ITERATION (top-level entry)
# =============================================================================

def _grow_K_with_recovery(probs_k, kept_mask_full, lam,
                            wildcard_mass_threshold=0.0,
                            min_relative_improvement=0.10,
                            K_max=10,
                            max_iter_per_K=50,
                            known_haps_full=None,
                            cc_scale=0.5,
                            use_log_bic=False,
                            min_nll_improvement=1e-6,
                            n_medoid_starts=K_MEDOID_STARTS_DEFAULT,
                            recovery_outer_cc_scale=RECOVERY_OUTER_CC_SCALE,
                            recovery_max_K=RECOVERY_MAX_K,
                            recovery_max_rounds=RECOVERY_MAX_ROUNDS,
                            recovery_max_outer_iterations=RECOVERY_MAX_OUTER_ITERATIONS,
                            recovery_cleanness_threshold=RECOVERY_CLEANNESS_THRESHOLD,
                            recovery_intra_round_dedup_pct=RECOVERY_INTRA_ROUND_DEDUP_PCT,
                            recovery_mixture_K_max=RECOVERY_MIXTURE_K_MAX,
                            recovery_mixture_n_restarts=RECOVERY_MIXTURE_N_RESTARTS,
                            recovery_mixture_seed_base=RECOVERY_MIXTURE_RNG_SEED,
                            recovery_mixture_patience=RECOVERY_MIXTURE_PATIENCE,
                            recovery_swap_nll_tolerance=RECOVERY_SWAP_NLL_TOLERANCE,
                            recovery_haps_equal_eps_pct=RECOVERY_HAPS_EQUAL_EPS_PCT,
                            verbose=False):
    """Drop-in replacement for _grow_K with subtraction-recovery iteration.

    Algorithm:
      0. Trio recovery (XOR-based group-trio algorithm) generates
         candidate founder haplotypes via algebraic composition of
         heterozygous samples.  Greedy forward-selection BIC trim
         keeps only haps that strictly improve BIC over K=0 baseline.
         May produce zero seed haps on blocks where the trio scheme
         doesn't apply (K<3, all-hom data, etc.).
      1. K-medoid multistart with per-branch recovery (single unified
         call to _initial_kgrowth_with_medoids):
           - Build N sample-dosage seeds, run PAM to pick
             n_medoid_starts diverse medoids.
           - For each medoid m: H_init = stack([H_trio_seed, medoid_m])
             when H_trio_seed is non-empty, else just medoid_m.
           - When H_trio_seed is non-empty, ALSO run a no-medoid
             baseline branch with H_init = H_trio_seed alone (size
             K_trio).  Required because medoid branches all start at
             K_trio+1 and never test K_trio+1 → K_trio; without the
             baseline, degenerate K_trio+1 attractors with the same
             NLL as truth K_trio (which happen on all-hets symmetry
             cases) win every branch and trap us in a wrong-K basin.
           - K-growth from H_init, then subtraction-recovery on the
             K-growth output, then BIC compute for the branch.
           - Pick branch with lowest BIC as the initial winner.
         When n_medoid_starts <= 1 or N is too small, falls back to a
         single branch (using H_trio_seed if provided, else None).
      2. Iterate up to recovery_max_outer_iterations times:
         a. Recovery on current H (multi-round subtraction + mixture +
            outer BIC subset selection until recovery's own internal
            convergence).
         b. If recovery didn't change H (within haps_equal_eps_pct), exit.
         c. K-growth from recovery's output (continues from K_init =
            K_after_recovery; worst-fit-sample seeding tries to add
            founders that the mixture missed, e.g., low-carrier truths).
         d. If K-growth didn't add anything, exit.
      3. Final _fit_at_fixed_K to populate the full return tuple.

    Why include H_trio_seed in every branch's H_init (instead of just
    the trio path)?  On blocks where trio recovers all true founders,
    every branch converges to K=K_trio (any extra medoid hap gets BIC-
    rejected) and the BIC-tied result is correct.  On blocks where
    trio recovers K_trio_correct < K_truth founders (e.g., one founder
    is hom-only or noise-defeated), different medoids on top of the
    shared trio seed give different starting positions for the K_trio
    +1-th founder; CD inside K-growth refines them, and BIC arbitra-
    tion picks the best.  This is more thorough than running a single
    trajectory from H_trio_seed (which would only try worst-fit-sample
    seeding for the K_trio+1-th founder, missing the multi-hypothesis
    benefit of multistart).

    Why per-branch recovery?  Without it, branches with marginally
    better K-growth-only BIC win even when another branch had recover-
    able founders that would have flipped the ranking after recovery.
    Per-branch recovery gives each branch a chance to BIC-discover its
    full founder set before cross-branch arbitration, so the winner
    is selected on its true post-recovery BIC rather than its K-
    growth-only BIC.

    Why iterate (step 2)?  K-growth (worst-fit-sample seeding) and
    recovery (Bernoulli mixture density) catch DIFFERENT failure modes:
      - Recovery's mixture finds founders supported by many candidates
        clustering in candidate-space (good for moderate-carrier counts).
      - K-growth's worst-fit-sample picks one sample's strand directly
        (good for low-carrier founders whose candidates don't form a
        density cluster but whose individual samples have high cost).
    Iteration ensures both mechanisms get a turn against the residual
    after the other has run.

    Returns: same tuple as _grow_K:
      (H, A, per_sample_cost, wildcard_slots, K_final, wildcard_mass, history)
    """
    N = probs_k.shape[0]

    # 0. Trio recovery: generate candidate founder haps and BIC-trim.
    #
    # The all-hets failure mode (no homozygous samples for some founder
    # pair) traps standard K-growth in wrong basins because every data-
    # driven seed candidate is a heterozygous strand (a blend of two
    # true founders).  Trio recovery sidesteps this by working in XOR
    # space, where het-pair samples have a clean structural composition
    # that lets us algebraically extract individual founders from
    # triangles of samples with overlapping pair-types.
    #
    # We then BIC-trim trio's output via greedy forward selection (the
    # same _greedy_bic_select used by the recovery loop) so spurious
    # haps (e.g., from chain-merged clusters at low-diversity
    # boundaries, or noise floor false positives) don't contaminate the
    # seed.  Each accepted hap strictly improves BIC; rejected haps are
    # dropped.  May produce zero seed haps if trio gives nothing
    # usable (K<3, all-hom data, no triangles match thresholds, or all
    # candidates fail BIC trim).  In that case we fall through to step
    # 1's multistart path.
    H_seed = np.zeros((0, probs_k.shape[1]), dtype=np.int64)
    if TRIO_RECOVERY_ENABLED or PAIRWISE_RECOVERY_ENABLED:
        # Gather trio candidates (XOR-triangle algebraic algorithm — see
        # _trio_recovery_candidate_haps above).  When TRIO_RECOVERY_ENABLED
        # is False, trio is skipped entirely (production behavior before
        # trio integration).
        if TRIO_RECOVERY_ENABLED:
            H_trio_candidates = _trio_recovery_candidate_haps(
                probs_k, verbose=verbose)
        else:
            H_trio_candidates = np.zeros((0, probs_k.shape[1]), dtype=np.int64)
        trio_list = [H_trio_candidates[k]
                     for k in range(H_trio_candidates.shape[0])]
        # Gather v6 pairwise common-hap candidates (partial-haps clustered
        # by mutual compatibility + quality filters A-E).  Pairwise covers
        # complementary failure modes to trio: trio excels on all-hets data
        # via XOR triangulation, pairwise excels when clean homozygous
        # samples for some founders exist (its pair-of-carriers signal is
        # strong there).  Feeding both into the combined seed gives BIC-
        # trim a richer pool.  See pairwise_common_hap.py for the v6
        # algorithm and the 50-block integration test results.
        if PAIRWISE_RECOVERY_ENABLED:
            pairwise_list = bhd_pairwise.pairwise_recovery_candidate_haps(
                probs_k, verbose=verbose)
        else:
            pairwise_list = []
        # Combine the two pools and dedup at 0.5% Hamming — collapses
        # near-exact duplicates (e.g., when trio and pairwise independently
        # recover the same clean truth).  Threshold matches the one used
        # inside _late_low_carrier_rescue (see ~line 3275 below).  When
        # PAIRWISE_RECOVERY_ENABLED is False this dedup is a no-op on
        # trio-only output (trio's internal dedup is at TRIO_HAP_DEDUP_PCT
        # = 2.0%, so trio candidates are guaranteed ≥2.0% apart and never
        # within 0.5%), making the pre-integration code path numerically
        # identical to the baseline.
        cand_list = []
        for cand in trio_list + pairwise_list:
            is_dup = False
            for kept in cand_list:
                if _hamming_pct_kept(cand, kept) < 0.5:
                    is_dup = True
                    break
            if not is_dup:
                cand_list.append(cand)
        if cand_list:
            # Greedy forward-selection BIC trim.  Uses the same cc_scale
            # AND use_log_bic as the K-growth that follows, so trim and
            # grow share an identical BIC criterion.  Each accepted hap
            # strictly improves BIC; rejected haps are dropped.
            #
            # Build a PoolEmissionCache wrapping the combined trio +
            # pairwise candidate pool.  _greedy_bic_select makes
            # O(|cand_list|² / 2) calls to _compute_nll_for_subset
            # internally (forward selection trials each remaining
            # candidate at each K step), and the cache amortises the
            # Viterbi emission build across all those calls.
            seed_cache = PoolEmissionCache(cand_list, probs_k,
                                            lam=lam)
            sel_indices, sel_haps, _trim_nll = _greedy_bic_select(
                seed_cache,
                cc_scale=cc_scale,
                max_k=K_max,
                use_log_bic=use_log_bic,
                verbose=verbose)
            if sel_haps:
                H_seed = np.stack(sel_haps, axis=0).astype(np.int64)
            if verbose:
                # Rewritten from the original `[trio] N candidates -> ...`
                # print to reflect that the BIC trim now consumes a
                # combined pool of trio + pairwise candidates.
                print(f'[seed] trio={len(trio_list)} + '
                      f'pairwise={len(pairwise_list)} -> '
                      f'combined+deduped={len(cand_list)} -> '
                      f'BIC-trimmed to {H_seed.shape[0]} seed haps')

    # 1. K-medoid multistart with per-branch recovery, optionally
    # seeded from trio.
    #
    # Each branch m starts from H_init = stack([H_seed, seed_array[m]])
    # when H_seed is non-empty, or just seed_array[m] when empty.
    # Per-branch recovery runs subtraction-recovery on the K-growth
    # output before computing branch BIC, so cross-branch arbitration
    # happens on post-recovery BIC.
    #
    # Why include H_seed in every branch (vs single trajectory from
    # H_seed when it's non-empty): different medoids give different
    # starting positions for the K_trio+1-th founder hypothesis,
    # giving multi-shot exploration in cases where trio finds K_trio
    # correct founders but the truth has K_trio+1 (e.g., one founder
    # is hom-only and trio missed it).
    #
    # Why per-branch recovery (vs only running recovery once on the
    # winner): without it, branches with marginally better K-growth-
    # only BIC win even when another branch's recovery would have
    # found additional founders that flipped the ranking.
    #
    # Cost: roughly 2x slower than the previous "single trajectory or
    # multistart-K-growth-only" design.  On production scale, pushes
    # stage-3 from ~30 min to ~1 hour single-threaded (proportional on
    # parallel cores).  Trade-off accepted for more thorough
    # exploration on the rare hard blocks where it matters.
    # Re-check thread allocation before the medoid multistart (the per-branch
    # recovery here is the heaviest single phase): a straggler block claims
    # cores freed as its peers finish.
    dynamic_threads.apply_dynamic_threads()
    H, A, costs, wcs, K_final, wm, history = _initial_kgrowth_with_medoids(
        probs_k, kept_mask_full, lam,
        n_medoid_starts=n_medoid_starts,
        wildcard_mass_threshold=wildcard_mass_threshold,
        min_relative_improvement=min_relative_improvement,
        K_max=K_max,
        max_iter_per_K=max_iter_per_K,
        known_haps_full=known_haps_full,
        cc_scale=cc_scale,
        use_log_bic=use_log_bic,
        min_nll_improvement=min_nll_improvement,
        H_trio_seed=H_seed,
        run_per_branch_recovery=True,
        recovery_outer_cc_scale=recovery_outer_cc_scale,
        recovery_max_K=recovery_max_K,
        recovery_max_rounds=recovery_max_rounds,
        recovery_intra_round_dedup_pct=recovery_intra_round_dedup_pct,
        recovery_mixture_K_max=recovery_mixture_K_max,
        recovery_mixture_n_restarts=recovery_mixture_n_restarts,
        recovery_mixture_seed_base=recovery_mixture_seed_base,
        recovery_mixture_patience=recovery_mixture_patience,
        recovery_cleanness_threshold=recovery_cleanness_threshold,
        recovery_swap_nll_tolerance=recovery_swap_nll_tolerance,
        recovery_haps_equal_eps_pct=recovery_haps_equal_eps_pct,
        verbose=verbose)

    if verbose:
        print(f'[recovery] Initial K-growth: K_final={K_final}, '
              f'wildcard_mass={wm:.4f}')

    # 2. Outer iteration: alternate recovery and K-growth
    for outer_it in range(recovery_max_outer_iterations):
        # Re-check thread allocation at the top of each outer iteration.
        dynamic_threads.apply_dynamic_threads()
        if verbose:
            print(f'[recovery] === Outer iteration {outer_it + 1} ===')

        # 2a. Recovery on current H
        H_after_recovery = _subtraction_recovery_round_loop(
            probs_k, H, lam,
            outer_cc_scale=recovery_outer_cc_scale,
            max_K=recovery_max_K,
            max_rounds=recovery_max_rounds,
            max_iter_per_K=max_iter_per_K,
            intra_round_dedup_pct=recovery_intra_round_dedup_pct,
            mixture_K_max=recovery_mixture_K_max,
            mixture_n_restarts=recovery_mixture_n_restarts,
            mixture_seed_base=recovery_mixture_seed_base + outer_it * 1000,
            mixture_patience=recovery_mixture_patience,
            cleanness_threshold=recovery_cleanness_threshold,
            swap_nll_tolerance=recovery_swap_nll_tolerance,
            haps_equal_eps_pct=recovery_haps_equal_eps_pct,
            use_log_bic=use_log_bic,
            verbose=verbose)

        # 2b. Did recovery change H?
        H_list = [H[k] for k in range(H.shape[0])] if H.shape[0] > 0 else []
        H_rec_list = [H_after_recovery[k] for k in range(H_after_recovery.shape[0])] \
                      if H_after_recovery.shape[0] > 0 else []
        if _haps_equal(H_rec_list, H_list, eps_pct=recovery_haps_equal_eps_pct):
            if verbose:
                print(f'[recovery] Outer iteration {outer_it + 1}: '
                      f'recovery did not change H -- CONVERGED')
            break

        if verbose:
            print(f'[recovery] Outer iteration {outer_it + 1}: '
                  f'recovery K {H.shape[0]} -> {H_after_recovery.shape[0]}')

        # 2c. K-growth from recovery's output
        H_after_grow, A, costs, wcs, K_after_grow, wm, hist_grow = _grow_K(
            probs_k, kept_mask_full, lam,
            wildcard_mass_threshold=wildcard_mass_threshold,
            min_relative_improvement=min_relative_improvement,
            K_max=K_max,
            max_iter_per_K=max_iter_per_K,
            known_haps_full=known_haps_full,
            cc_scale=cc_scale,
            use_log_bic=use_log_bic,
            min_nll_improvement=min_nll_improvement,
            H_init=H_after_recovery)
        history.extend(hist_grow)

        # 2d. Did K-growth add anything?
        H_grow_list = [H_after_grow[k] for k in range(H_after_grow.shape[0])] \
                       if H_after_grow.shape[0] > 0 else []
        if _haps_equal(H_grow_list, H_rec_list, eps_pct=recovery_haps_equal_eps_pct):
            if verbose:
                print(f'[recovery] Outer iteration {outer_it + 1}: '
                      f'K-growth did not add -- CONVERGED')
            H = H_after_grow
            break

        if verbose:
            print(f'[recovery] Outer iteration {outer_it + 1}: '
                  f'K-growth K {H_after_recovery.shape[0]} -> {H_after_grow.shape[0]}')

        H = H_after_grow

    # 3. Final fit to populate return values consistently
    H_final, A_final, costs_final, wcs_final, n_iter_final, nll_final = \
        _fit_at_fixed_K(probs_k, H, lam, max_iter=max_iter_per_K)

    # 3.5. Late low-carrier rescue (added May 2026): targeted post-
    # convergence pass that detects suspect low-carrier haps (potential
    # chimeric stand-ins for low-frequency founders) and tries to
    # replace them with carrier-derived residual candidates via BIC-
    # aware forward selection.  Triggers only when min carrier count
    # is below RECOVERY_LOW_CARRIER_TRIGGER_FRAC of 2N (typical: <5%
    # of blocks); for triggered blocks, accepts the new state iff it
    # strictly improves BIC.  Cannot regress.  See chr6:23624234
    # diagnostic for the motivating analysis.
    H_final, A_final, costs_final, wcs_final, nll_final = _late_low_carrier_rescue(
        probs_k, H_final, A_final, costs_final, wcs_final, nll_final,
        lam=lam, cc_scale=cc_scale, use_log_bic=use_log_bic,
        max_iter=max_iter_per_K, verbose=verbose)

    # 3.6. Residual-trio rescue (added 2026-05): post-convergence pass
    # that mines per-sample residuals across ALL samples (not just low-
    # carrier-hap carriers) to surface near-clone founders that K-
    # growth's residual-mass seeding missed.  Complements low-carrier
    # rescue (which handles low-frequency chimeric replacements) by
    # targeting the orthogonal pattern: all haps healthy but one
    # absorbs carriers of a near-clone partner founder.  Internal
    # gate: skip if no admitted candidate or BIC does not improve.
    # Cannot regress.  See chr10:503 diagnostic for the motivating
    # case (F0 vs F4 at 5-bit distance, 14 clean F0 carriers absorbed
    # into the F4 slot).
    if RESIDUAL_TRIO_ENABLED:
        H_final, A_final, costs_final, wcs_final, nll_final = _residual_trio_rescue(
            probs_k, H_final, A_final, costs_final, wcs_final, nll_final,
            lam=lam, cc_scale=cc_scale, use_log_bic=use_log_bic,
            max_iter=max_iter_per_K, verbose=verbose)

    wm_final = float(wcs_final.sum()) / max(2 * N, 1)

    if verbose:
        # Report BIC (not raw NLL) since "FINAL" is the comparison point
        # external callers might use to compare across different K
        # outcomes from this function.
        cc_final = _compute_cc(cc_scale, N, probs_k.shape[1],
                                use_log_bic=use_log_bic)
        bic_final = _compute_bic(H_final.shape[0], nll_final, cc_final)
        print(f'[recovery] FINAL: K={H_final.shape[0]}, '
              f'BIC={bic_final:.1f}, NLL={nll_final:.1f}, '
              f'wildcard_mass={wm_final:.4f}')

    return H_final, A_final, costs_final, wcs_final, H_final.shape[0], wm_final, history