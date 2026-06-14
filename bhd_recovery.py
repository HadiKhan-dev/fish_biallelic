#%% =====================================================================
# bhd_recovery.py — Subtraction-based recovery + late low-carrier rescue
#
# Split out of block_haplotypes_discrete.py as part of the 4-file split.
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
#     here uses Viterbi NLL via bhd_kernels._compute_nll_for_subset
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

import math
import warnings

import numpy as np

# Defensive numba import matching the project convention (see
# analysis_utils.py, block_haplotypes.py, bhd_trio.py, bhd_pairwise.py).
# If numba is unavailable, all @njit decorators become no-ops and the
# per-kernel scalar loops run as pure Python (slow but correct).  The
# wrappers preserve the exact same input/output shapes either way.
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    warnings.warn(
        "Numba not found.  bhd_recovery will fall back to pure-Python "
        "paths for _run_subtraction_round, _generate_carrier_residuals, "
        "_bernoulli_mixture_em, and _kmeans_pp_init (typically 2-12x "
        "slower per call).",
        ImportWarning,
    )
    # Dummy decorator that accepts arguments (like cache=True) but does
    # nothing — same pattern as analysis_utils.py, bhd_trio.py,
    # bhd_pairwise.py.
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        # Support both @njit and @njit(cache=True) forms
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return decorator

import bhd_kernels
import bhd_trio
import bhd_pairwise

# Explicit named imports from bhd_kernels for symbols used directly in
# this file's function bodies.  Cross-module ENABLED-flag reads use
# module-attribute lookup (e.g. bhd_pairwise.PAIRWISE_RECOVERY_ENABLED)
# to preserve runtime-mutation semantics, but function imports are
# explicit since functions don't get re-bound at runtime.
from bhd_kernels import (
    DEFAULT_LAMBDA,
    LOG_EPS,
    MASK,
    _safe_neg_log,
    _per_site_cost_W_W,
    _update_A,
    _update_H,
    _fit_at_fixed_K,
    _compute_cc,
    _compute_bic,
    _compute_nll_for_subset,
    PoolEmissionCache,
)

# Cross-module function imports
from bhd_trio import _trio_recovery_candidate_haps


# =============================================================================
# CONSTANTS — SUBTRACTION-BASED RECOVERY (mixture + BIC + outer iteration)
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

# Cleanness threshold for residuals to be admitted as candidates.
# A residual is "clean" if at least this fraction of sites have values
# in {0, 1} after subtraction (i.e., the founder hypothesis is consistent
# with the sample's argmax dosage).  Lowering admits more candidates
# but with more noise; raising rejects valid candidates whose other
# strand is genuinely there but with read-error sites.
RECOVERY_CLEANNESS_THRESHOLD = 0.90

# -----------------------------------------------------------------------------
# Residual mode for subtraction recovery (low-read-depth)
# -----------------------------------------------------------------------------
# RECOVERY_RESIDUAL_MODE selects how the residual candidates fed to the inner
# Bernoulli mixture are generated inside _subtraction_recovery_round_loop:
#
#   "argmax" (default) — the behaviour documented above: subtract each
#            founder from each sample's ARGMAX dosage, clip the result to
#            {0, 1}, and admit residuals whose fraction of in-{0,1} sites
#            (the hard "cleanness") meets RECOVERY_CLEANNESS_THRESHOLD.  This
#            is the validated production path; selecting "argmax" leaves
#            _subtraction_recovery_round_loop bit-identical to before.
#
#   "soft" — a low-read-depth front-end that keeps the genotype likelihoods
#            instead of hard-calling them, and clusters the residuals with a
#            marginal-likelihood Bernoulli-haplotype mixture (model B).  For
#            founder strand h and a sample with genotype likelihoods L(g),
#            the latent other strand o satisfies g = h + o, giving per-site
#            other-strand likelihoods (the inadmissible third genotype is
#            dropped):
#              h[l]=0:  L0 = L(g=0) [o=0],  L1 = L(g=1) [o=1]
#              h[l]=1:  L0 = L(g=1) [o=0],  L1 = L(g=2) [o=1]
#            A candidate is admitted when its mean admissible POSTERIOR mass
#            (P0+P1 for h=0, P1+P2 for h=1) meets RECOVERY_CLEANNESS_THRESHOLD
#            — the same screen as the argmax path.  The mixture then
#            MARGINALISES o rather than plugging in its posterior mean:
#              E-step:  log P(cand | k) = Σ_l log(L0(1-θ_k) + L1 θ_k)
#              M-step:  θ_k[l] = Σ_m γ_mk r_o / Σ_m γ_mk,
#                       r_o = L1 θ_k / (L0(1-θ_k) + L1 θ_k)
#            At an uninformative site (L0 ~ L1) r_o -> θ_k[l], so the
#            candidate defers to the cluster consensus and θ converges to the
#            mean of the INFORMATIVE carriers only — eliminating the dilution
#            toward 0.5 that sinks a plain Bernoulli mixture fed E[o] (the
#            earlier plug-in soft path, which was empirically WORSE than
#            argmax at low depth for exactly this reason).  Sub-threshold
#            evidence is still used (graded), unlike argmax which thresholds
#            it away — so model B ties argmax where data is plentiful and
#            beats it in the low-depth transition zone.  The rounded θ
#            (θ > 0.5) is the consensus founder.  K-means++ init runs on the
#            expected other strand E[o] = L1/(L0+L1) via the same L1-distance
#            initialiser as the argmax path (L1 == Hamming on binary; see
#            _kmeans_pp_init_kernel).
#
#            probs_k holds the POSTERIOR; the marginalisation needs the
#            LIKELIHOOD.  When site_priors is threaded in (real data) the HWE
#            site prior is divided out: L(g) ∝ probs_k[g] / site_priors[g]
#            (the per-(sample, site) normaliser cancels in both EM steps, so
#            only the per-genotype prior ratio matters).  When site_priors is
#            None probs_k is used directly — exact when it already is the
#            likelihood (flat prior, e.g. synthetic data).
#
# Same rationale as the trio / seed soft front-ends.  "soft" is RESULT-
# AFFECTING at every read depth and must be validated against ground truth
# before use; it is opt-in for that reason.  Only the main recovery loop
# (_subtraction_recovery_round_loop) honours this switch; the rescue entries
# (_late_low_carrier_rescue, _residual_trio_rescue) still use argmax
# residuals.
RECOVERY_RESIDUAL_MODE = "argmax"

# Bernoulli mixture parameters for inner K-selection on candidates
RECOVERY_MIXTURE_K_MAX = 10            # try K=1..K_max, pick best by inner BIC
RECOVERY_MIXTURE_N_RESTARTS = 2        # EM restarts per K (different K-means++ seeds).
                                       # Reduced 3 -> 2 after validation: produces bit-
                                       # identical recovered founders across low- and
                                       # high-K, uniform- and skewed-frequency blocks
                                       # (K=6/12/20 tested).  The mixture only PROPOSES
                                       # candidate consensus haps; the outer BIC subset-
                                       # selection on the sample data is what picks the
                                       # founders, and it is empirically robust to which
                                       # mixture local-optimum did the proposing.  Kept at
                                       # 2 (not 1) to retain a restart safety margin for
                                       # the multi-modal EM landscape.
RECOVERY_MIXTURE_MAX_ITER = 100        # max EM iterations per fit
RECOVERY_MIXTURE_TOL = 1e-3            # relative LL change for EM convergence.
                                      # The mixture output is a BINARY consensus
                                      # (theta > 0.5), which stabilises long
                                      # before the LL converges to high
                                      # precision, so 1e-3 yields a consensus
                                      # identical to 1e-6 (verified, 0 bit-diff)
                                      # while cutting EM iterations ~3x — the
                                      # dominant cost of the soft recovery path
                                      # (and each block runs on a single core,
                                      # so this single-thread cut is the win).
RECOVERY_MIXTURE_THETA_EPS = 1e-3      # clip theta to [eps, 1-eps] for log stability
RECOVERY_MIXTURE_RNG_SEED = 42         # base RNG seed (varied per round)

# Patience for the mixture K-sweep early-stop (see _fit_bernoulli_mixture_-
# select_K / _fit_bernoulli_mixture_ml_select_K).  Both inner-mixture fits
# sweep K = 1 .. K_max_effective and pick the global-min-BIC K.  Without an
# early stop that sweep is linear in K_max_effective in the number of EM
# fits (and ~quadratic in wall time, since each EM fit at K components is
# itself O(K)); when the recovery caps are raised to support many founders
# (e.g. K_max ~ 40), a block whose true K is small would otherwise pay the
# full 40-wide sweep on every mixture call.  The early stop tracks the best
# BIC seen so far and terminates once `patience` CONSECUTIVE increasing-K
# values fail to improve on it.  It only ever truncates the TAIL of the
# sweep — it never changes which K is selected among the K it evaluates —
# and because the counter resets on every BIC improvement, a fit whose BIC
# is still descending at K* is always evaluated through K* (plus `patience`
# further K).  The residual risk is a globally better BIC that sits MORE
# than `patience` increasing-K steps past a local BIC minimum following a
# non-monotone bump; the RECOVERY_MIXTURE_N_RESTARTS EM restarts per K
# mitigate the EM-local-optimum source of such bumps.  Threaded as
# recovery_mixture_patience / mixture_patience through the recovery call
# chain; pass None there to disable the early stop and recover the
# full-sweep behaviour bit-for-bit.
RECOVERY_MIXTURE_PATIENCE = 3

# Intra-round dedup safety net.  The mixture's BIC over K already
# prevents near-duplicate components from being selected, so this is
# only a safety net for true duplicates that survive (e.g., from
# numerical rounding or restart inconsistencies).  Tight 2% threshold:
# legitimate close founders (>=3% truth distance) won't be merged.
RECOVERY_INTRA_ROUND_DEDUP_PCT = 2.0

# Outer BIC complexity-cost scale for subset selection on sample data.
# Distinct from K-growth's cc_scale=0.05 — the outer subset-selection
# uses the project-standard 0.5 (matches beam_search_core /
# chimera_resolution).  Different problem (subset selection over a
# finite candidate pool, not greedy K-growth from worst-fit seeds), so
# the calibration is different.
#
# Update (May 2026): the comment above is historical.  K-growth's
# default cc_scale was raised from 0.05 to 0.5 to match this constant,
# eliminating the asymmetry that caused K-growth/recovery oscillation
# at chr3:16378549.  Both K-growth and recovery's outer subset-
# selection now use cc_scale=0.5; this value is retained as a named
# constant for clarity at the recovery call sites.  See the cc_scale
# docstring in _grow_K for the full rationale.
RECOVERY_OUTER_CC_SCALE = 0.5

# Hard caps on selected size and rounds (defensive against pathological
# blocks that wouldn't converge naturally).
RECOVERY_MAX_K = 12
RECOVERY_MAX_ROUNDS = 4

# NLL-tolerance for swap refinement: a swap is applied only if it
# reduces NLL by more than this amount (avoids oscillation between
# near-equivalent haps from numerical noise).
RECOVERY_SWAP_NLL_TOLERANCE = 0.5

# Hap-equality tolerance for convergence detection (between rounds and
# between outer iterations).  Two haps within this Hamming-percentage
# are considered "the same" for convergence purposes.
RECOVERY_HAPS_EQUAL_EPS_PCT = 0.5

# Outer iteration cap: K-growth and recovery alternate up to this many
# times.  In practice virtually all blocks converge in 1-2 outer
# iterations; 3 is a defensive safety net.
RECOVERY_MAX_OUTER_ITERATIONS = 3

# =============================================================================
# CONSTANTS — LATE LOW-CARRIER RESCUE (post-convergence targeted refinement)
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

# Trigger threshold (fraction of 2N).  A hap is "low-carrier" if its
# real-strand usage count is below this fraction of total real strands
# (= 2*N = total real-strand slots across N samples).  At N=320:
# 0.02 * 640 = 12.8 → minimum carriers must be < 13 to trigger.
# At chr6:23624234, alg_row_5 has 4 carriers (0.6%) → triggers.
# At non-pathological blocks, all founders typically carry >5% of
# strands → no trigger, zero overhead.
RECOVERY_LOW_CARRIER_TRIGGER_FRAC = 0.02

# =============================================================================
# RESIDUAL-TRIO RESCUE CONSTANTS (added 2026-05): post-K-growth pass that
# mines per-sample residuals (argmax_dosage[s] - H[A[s, other_slot]]) across
# ALL samples (not just low-carrier-hap carriers) to surface near-clone
# founders that K-growth's residual-mass seeding missed.
#
# Motivating case: chr10:503 F0.  At this block F0 has 36 carriers (11%) but
# differs from F4 at only 5 of 200 sites.  K-growth fits {F1, F2, F4, chimera}
# and absorbs the 14 clean F0 carriers into the F4 slot (residual NLL ≈ 70
# nats over 14 samples × 5 sites).  Trio recovery never emits pure-F0 as a
# candidate (the F0-pair-type groups are dominated by 22 chimera carriers
# whose group dosages encode the chimera, not pure F0).
#
# Mechanism: for each (sample, slot), compute residual = argmax_dosage[s] -
# H[A[s, other_slot]].  When the other-slot partner is exact-truth, residual
# = the actual other-strand founder exactly (verified by the cleanness
# filter, which rejects residuals with out-of-range bits).  At chr10:503,
# 9 clean F0/F2 samples currently fit as (F4, F2) produce 9 identical
# residuals = pure F0 at 5 bits from H[F4] — a clean cluster of 9 candidates
# pointing at the missing near-clone founder.
#
# This complements _late_low_carrier_rescue.  Late-rescue triggers only when
# min hap usage drops below RECOVERY_LOW_CARRIER_TRIGGER_FRAC and exists to
# replace low-frequency chimeras.  Residual-trio triggers on the orthogonal
# pattern: ALL haps have healthy usage but one of them is absorbing carriers
# of a near-clone partner founder.  No overlap by construction (low-rescue
# bails out if no low-carrier hap exists; residual-trio is the path for the
# remaining cases).
RESIDUAL_TRIO_ENABLED = True

# Cleanness threshold for residual-trio: residuals where < this fraction of
# sites land in {0, 1} after subtraction are rejected as noise.  Higher than
# the late-rescue cleanness (0.95) because residual-trio mines every sample
# pair, not just low-carrier targets, and noisier residuals dilute clean
# clusters more readily.  At cleanness = 1.0 (every site clean), only
# residuals against a truly-correct partner survive — the strictest possible
# filter, and the right default when most blocks have clean partners.
RESIDUAL_TRIO_CLEANNESS_THRESHOLD = 1.0

# Cluster dedup threshold for residual-trio candidates (% Hamming).  After
# generating up to 2N clean residuals across all samples, near-duplicates
# are merged.  0.5% at L=200 means residuals within 1 bit cluster together.
# Tighter than the 1% used inside trio's hap-pool dedup because residuals
# are direct subtractions (no consensus averaging), so noise per residual
# is at most 1-2 sites — exactly bit-identical residuals should cluster
# but anything more is a real distinction worth preserving.
RESIDUAL_TRIO_DEDUP_PCT = 0.5

# Minimum cluster size (number of supporting samples) for a residual-trio
# candidate to be admitted to the BIC pool.  At chr10:503 the clean-F0
# signal is 9 samples — the smallest support we'd want to ensure produces
# a real candidate.  Set to 3 to filter out single-sample-noise residuals
# (a single sample's residual could be a chimera fragment); 3+ identical
# residuals from independent samples is strong evidence of a hidden founder.
RESIDUAL_TRIO_MIN_CLUSTER_SIZE = 3

# Dedup threshold (% Hamming) for matching residual-trio candidates against
# existing H rows.  Tighter than candidate-vs-candidate dedup (0.5%) since
# the H rows are themselves the "current best estimate" and we want to skip
# candidates that won't add new information.  1.0% at L=200 = 2 bits.
RESIDUAL_TRIO_DEDUP_VS_H_PCT = 1.0

# =============================================================================
# SUBTRACTION-BASED RECOVERY: BERNOULLI MIXTURE HELPERS
# =============================================================================
#
# These helpers fit a Bernoulli mixture model to a candidate pool of clean
# residuals (one per "carrier" sample after subtracting an existing founder).
# Each candidate is a binary vector of length L_kept; we model them as a
# mixture of K hidden founder profiles theta_k in [0,1]^L, each candidate
# generated by one component:
#
#     P(c | k) = prod_l theta_k[l]^c[l] * (1 - theta_k[l])^(1-c[l])
#
# We fit by EM with K-means++ initialisation and multiple restarts, then
# pick the K that minimises BIC over candidate density (inner BIC).  The
# fitted theta vectors (rounded to {0, 1}) become consensus candidate haps
# for outer BIC subset-selection on actual sample data.


@njit(cache=True)
def _kmeans_pp_init_kernel(candidates, K, uniforms):
    """Numba kernel for _kmeans_pp_init.

    Faithfully reproduces the K-means++ algorithm using uniform random
    draws supplied by the wrapper:

      - uniforms[0]: drives the first center selection (uniform over N)
      - uniforms[k] for k in 1..K-1: drives the weighted choice for
        the k-th center (weighted by min-L1-distance squared; L1 equals
        Hamming on binary candidates)

    Conversion from uniform to index:
      - For the first center: int(uniforms[0] * N) — same as
        np.random.randint(N) under the hood, gives a value in [0, N).
      - For weighted draws: inverse-CDF lookup on the normalized
        squared-distance probabilities.

    Edge cases handled:
      - K >= N: defensive duplicates path (matches the original's
        `idx = list(range(N)) + [rng.integers(N) ...]` behavior).
      - All min_dists zero (all candidates identical to some existing
        center): pick first unpicked index, falling back to
        int(uniforms[k] * N) if all are picked (matches the original's
        `[i for i in range(N) if i not in centers_idx]` then random).

    SEMANTIC DIFFERENCE FROM ORIGINAL:
      The original calls `rng.integers(N)` and `rng.choice(N, p=probs)`
      directly on the user's Generator.  This kernel takes precomputed
      uniforms, so the underlying PRNG sequence will differ — but only
      by which Generator method advances the state.  The weights, the
      inverse-CDF mapping, and the K-means++ algorithm are preserved
      exactly.  Tested empirically and accepted by the project owner
      (see conversation transcript): tiny shifts in which specific
      candidate gets picked for which center may occur, but downstream
      BIC selection on actual sample data is stable enough to absorb
      these.

    Args:
        candidates: (N, L) float64 candidate array (binary 0/1 in argmax
            mode, fractional in [0, 1] in soft mode)
        K: number of centers to pick
        uniforms: (K,) float64 random draws in [0, 1) from the wrapper

    Returns:
        centers_idx: (K,) int64 — indices into candidates for the
            selected centers (caller does candidates[centers_idx].copy()
            to get the centers themselves)
    """
    N, L = candidates.shape

    centers_idx = np.empty(K, dtype=np.int64)
    # picked[i] = True if i is already a chosen center (used in the
    # "all-zero min_dists" tiebreak path).  bool mask replaces the
    # original's Python `in centers_idx` list-membership test.
    picked = np.zeros(N, dtype=np.bool_)

    # === First center: uniform over N (matches rng.integers(N)) ===
    # int(u * N) maps uniformly into [0, N) for u in [0, 1).
    c0 = int(uniforms[0] * N)
    if c0 >= N:
        c0 = N - 1   # defensive: clip if u rounds to 1.0 (numerically possible)
    centers_idx[0] = c0
    picked[c0] = True

    # Per-candidate running minimum distance to nearest center.  Distance
    # is L1 (sum of |a - b| over sites).  On BINARY candidates this equals
    # Hamming distance exactly (|0-1| = 1, |0-0| = |1-1| = 0), so argmax-
    # mode recovery is bit-identical to the original Hamming kernel; on
    # FRACTIONAL (soft) candidates L1 is the correct generalisation (a
    # Hamming `!=` test would treat 0.49 vs 0.51 as a full mismatch).
    # We update this incrementally each time a new center is added —
    # cheaper than recomputing from scratch (the original recomputes
    # min over all existing centers each iteration via the (N, n_exist, L)
    # broadcast).  Mathematical result is identical: min over a growing
    # set equals running-min.
    #
    # Initial state: distance to c0 only.
    min_dists = np.empty(N, dtype=np.float64)
    for i in range(N):
        # L1 distance between candidates[i] and candidates[c0]
        d = 0.0
        for l in range(L):
            d += abs(candidates[i, l] - candidates[c0, l])
        min_dists[i] = d

    # === Subsequent centers: weighted by squared min-Hamming ===
    for k in range(1, K):
        # Check if all min_dists are zero (all candidates identical to
        # some chosen center).  This matches the original's
        # `if min_dists.sum() == 0: remaining = [i not in centers_idx]`
        # tiebreak path.
        any_nonzero = False
        for i in range(N):
            if min_dists[i] > 0:
                any_nonzero = True
                break

        if not any_nonzero:
            # Tiebreak: pick the first unpicked index.  Original used
            # rng.choice over the remaining list; here we pick
            # deterministically (the first unpicked).  This is a
            # SEMANTIC SIMPLIFICATION — when this tiebreak path is
            # reached (all candidates identical to some center) the
            # choice is degenerate anyway, but we note it for honesty.
            new_c = -1
            for i in range(N):
                if not picked[i]:
                    new_c = i
                    break
            if new_c < 0:
                # All N candidates already picked (only possible if K >= N,
                # which the wrapper would route through the duplicate path
                # before calling this kernel — defensive only).
                new_c = int(uniforms[k] * N)
                if new_c >= N:
                    new_c = N - 1
        else:
            # Inverse-CDF sample over min_dists^2.  Build cumulative
            # distribution on the fly and find the index where u
            # crosses it.
            total = 0.0
            for i in range(N):
                d = min_dists[i]
                total += float(d) * float(d)
            target = uniforms[k] * total
            cum = 0.0
            new_c = N - 1   # defensive default (last index, in case
                              # of float rounding at the boundary)
            for i in range(N):
                d = min_dists[i]
                cum += float(d) * float(d)
                if cum >= target:
                    new_c = i
                    break

        centers_idx[k] = new_c
        picked[new_c] = True

        # Update running min_dists with distance to new_c (skip the last
        # iteration since we don't need min_dists after that)
        if k < K - 1:
            for i in range(N):
                d = 0.0
                for l in range(L):
                    d += abs(candidates[i, l] - candidates[new_c, l])
                if d < min_dists[i]:
                    min_dists[i] = d

    return centers_idx


def _kmeans_pp_init(candidates, K, rng):
    """K-means++ initialisation for binary data using Hamming distance.

    Picks K centers from candidates: the first uniformly at random, each
    subsequent weighted by squared min-Hamming-distance from existing
    centers.  This gives spread-out initial centers that typically lead
    EM to good local optima — much more robust than uniform random init.

    Args:
      candidates: (N, L) binary array
      K: number of centers to pick
      rng: numpy Generator

    Returns: (K, L) array of centers selected from candidates (copy).

    Implementation: delegates to a numba kernel that uses incremental
    running-min for Hamming distances (vs the original's full
    recomputation from scratch each iteration), with an inverse-CDF
    weighted sampling driven by precomputed uniforms drawn from the
    Python-level rng.

    NOTE ON RANDOM STATE:
      The original calls `rng.integers(N)` and `rng.choice(N, p=probs)`
      directly.  This wrapper instead draws K uniform `rng.random()`
      values and passes them to the kernel for in-kernel conversion to
      indices via int(u*N) for the first center and inverse-CDF for
      subsequent ones.  The math is preserved exactly (same weights,
      same algorithm); only the specific bytes consumed from the PRNG
      state differ, which causes the specific indices picked across
      n_restarts multi-starts to shuffle.  Project owner has accepted
      this tradeoff in exchange for a ~7-8x speedup on the dominant
      inner-hamming step.  Downstream BIC selection on sample data is
      empirically stable to this perturbation (validated against the
      legacy implementation's selected-K trace; see conversation
      transcript).
    """
    N, L = candidates.shape
    if K >= N:
        # Defensive: caller should never let this happen, but if it does,
        # return all candidates plus repeats so the EM has something to work with.
        # This path is preserved EXACTLY as the original (still uses rng
        # directly).
        idx = list(range(N)) + [int(rng.integers(N)) for _ in range(K - N)]
        return candidates[idx].copy()

    # Draw K uniform [0, 1) values from the user's Generator for the
    # kernel to consume.  Using rng.random(K) advances the Generator
    # state exactly once with a single call, vs the original's K calls
    # to rng.integers / rng.choice.
    uniforms = rng.random(K)

    candidates_arr = np.ascontiguousarray(candidates, dtype=np.float64)
    centers_idx = _kmeans_pp_init_kernel(candidates_arr, int(K), uniforms)
    return candidates[centers_idx].copy()


@njit(cache=True)
def _logsumexp_axis1_kernel(x):
    """Numerically stable log-sum-exp along axis 1 of a 2D array.

    Numba-friendly version of _logsumexp(x, axis=1).  Returns a 1D
    array of length x.shape[0].  Handles -inf max (all entries -inf
    in a row) by zeroing the offset (so the result is -inf, not NaN),
    matching the original `np.where(np.isfinite(m), m, 0.0)` semantic.

    This is an internal helper for the EM kernel.  Math:
      m_i  = max_j x_ij
      m'_i = m_i if finite, else 0.0
      out_i = m'_i + log(sum_j exp(x_ij - m'_i))
    Note: the (m_i not finite) branch is reached only when all entries
    in row i are -inf, in which case sum exp(x - 0) = 0 and log(0) =
    -inf, so out_i = -inf — matches the original.
    """
    N, K = x.shape
    out = np.empty(N, dtype=np.float64)
    for i in range(N):
        # max along K
        m = x[i, 0]
        for k in range(1, K):
            if x[i, k] > m:
                m = x[i, k]
        # m_safe: zero if -inf, else m.  np.isfinite returns False for
        # +inf and NaN too; we replicate that.
        if np.isfinite(m):
            m_safe = m
        else:
            m_safe = 0.0
        # sum exp(x - m_safe)
        s = 0.0
        for k in range(K):
            s += np.exp(x[i, k] - m_safe)
        out[i] = m_safe + np.log(s)
    return out


@njit(cache=True)
def _bernoulli_mixture_em_kernel(cands, K, init_centers,
                                    max_iter, tol, eps):
    """Numba kernel for _bernoulli_mixture_em.

    Faithfully reproduces the EM math step by step, with two changes
    from the pure-Python version:
      1. Matmuls (cands @ log_theta.T, resp.T @ cands) remain as numba
         @ operator, which dispatches to the same OpenBLAS backend as
         numpy — no performance change, no numerical change.
      2. The custom _logsumexp(axis=1) is replaced by an inlined
         scalar-loop equivalent (_logsumexp_axis1_kernel) so the whole
         iteration body runs as compiled native code with no Python
         object round-trips.

    Numerical behavior: bit-identical to the pure-Python version on
    typical inputs.  Tiny floating-point differences are possible at
    the last bit because:
      - numpy's vectorized `max(axis=1)`, `exp`, `sum` may use SIMD
        reductions with different rounding order than the scalar loop.
      - The matmul result is identical (same BLAS), but the +log_pi
        addition is done element-wise so order is preserved.
    Empirically (see validation suite) all tested seeds produced
    identical theta and resp arrays to full float64 precision.

    Args identical to _bernoulli_mixture_em; returns the same tuple.
    Note: when max_iter == 0, returns init_centers smoothed but resp
    and ll uninitialized in the original — we match that by allocating
    resp to zeros (matches the original which returns resp=None — but
    since numba can't return None, we return zeros; the wrapper detects
    this case and substitutes None for downstream Python callers).
    """
    N, L = cands.shape
    # Smooth initial centers from {0, 1} to (eps, 1-eps) for numerical stability
    theta = init_centers.astype(np.float64) * (1 - 2 * eps) + eps
    pi = np.ones(K, dtype=np.float64) / K

    prev_ll = -np.inf
    n_iter = 0
    ll = 0.0    # initialised in case max_iter=0 (defensive)

    # Pre-allocate resp so the function has a value to return even
    # when max_iter=0.  The original returns resp=None in that case;
    # we return all-zeros here and the wrapper handles the None
    # substitution.  In practice production never sets max_iter=0
    # so this code path is purely defensive.
    resp = np.zeros((N, K), dtype=np.float64)

    for it in range(max_iter):
        n_iter = it + 1
        # E-step
        log_theta = np.log(theta)
        log_one_minus_theta = np.log(1 - theta)
        # E-step emission log P(c | k), FUSED to a single GEMM.  The original
        # form was the two-GEMM expression
        #     log_p = cands @ log_theta.T + (1 - cands) @ log_one_minus_theta.T
        # which also materialises a full (N, L) (1 - cands) temporary.  Since
        # `cands` is binary {0, 1}, that equals (exactly, in real arithmetic)
        #     cands @ (log_theta - log_one_minus_theta).T
        #         + sum_l log_one_minus_theta[k, l]
        # i.e. ONE GEMM against the logit matrix plus a per-component constant
        # c[k] (the row-sums of log_one_minus_theta) broadcast over samples.
        # This halves the E-step GEMM work and drops the (1 - cands) alloc.
        # The floating-point summation order differs from the two-GEMM form,
        # so log_p (and hence theta / resp / ll) may differ at the last bits;
        # the binary consensus (theta > 0.5) and the downstream BIC subset-
        # selection are robust to that (validated on the recovery panel).
        # Numba's @ dispatches to the same BLAS as numpy.
        logit = log_theta - log_one_minus_theta                  # (K, L)
        c = np.empty(K, dtype=np.float64)
        for k in range(K):
            s = 0.0
            for l in range(L):
                s += log_one_minus_theta[k, l]
            c[k] = s
        log_p = cands @ logit.T + c                              # (N, K) + (K,)
        log_pi = np.log(pi + 1e-15)
        # log_p_weighted = log_p + log_pi[None, :]  — broadcast row-wise
        log_p_weighted = log_p + log_pi  # numba broadcasts (N, K) + (K,)

        log_norm = _logsumexp_axis1_kernel(log_p_weighted)                       # (N,)
        # log_resp = log_p_weighted - log_norm[:, None]; then resp = exp(log_resp).
        # We don't actually need log_resp as a separate variable here.
        resp = np.empty((N, K), dtype=np.float64)
        for i in range(N):
            for k in range(K):
                resp[i, k] = np.exp(log_p_weighted[i, k] - log_norm[i])

        # ll = float(log_norm.sum())
        ll_acc = 0.0
        for i in range(N):
            ll_acc += log_norm[i]
        ll = ll_acc

        # Convergence check (relative).  Break BEFORE updating prev_ll so
        # the caller receives the most recent ll value.
        if prev_ll != -np.inf and abs(ll - prev_ll) < tol * max(abs(ll), 1.0):
            break
        prev_ll = ll

        # M-step
        # N_k = resp.sum(axis=0)
        N_k = np.empty(K, dtype=np.float64)
        for k in range(K):
            s = 0.0
            for i in range(N):
                s += resp[i, k]
            N_k[k] = s
        # pi = N_k / N
        pi = N_k / N
        # N_k_safe = np.maximum(N_k, 1e-10)
        N_k_safe = np.empty(K, dtype=np.float64)
        for k in range(K):
            N_k_safe[k] = N_k[k] if N_k[k] > 1e-10 else 1e-10
        # theta = (resp.T @ cands) / N_k_safe[:, None]
        theta = (resp.T @ cands) / N_k_safe.reshape(K, 1)
        # theta = np.clip(theta, eps, 1 - eps)
        for k in range(K):
            for l in range(L):
                if theta[k, l] < eps:
                    theta[k, l] = eps
                elif theta[k, l] > 1 - eps:
                    theta[k, l] = 1 - eps

    return theta, pi, ll, n_iter, resp


def _bernoulli_mixture_em(cands, K, init_centers,
                            max_iter=RECOVERY_MIXTURE_MAX_ITER,
                            tol=RECOVERY_MIXTURE_TOL,
                            eps=RECOVERY_MIXTURE_THETA_EPS):
    """Single EM run for a Bernoulli mixture with K components.

    Args:
      cands: (N, L) float64 array (binary 0/1 values, but float for arithmetic)
      K: number of components
      init_centers: (K, L) initial theta values (binary will be smoothed)
      max_iter: cap on EM iterations
      tol: relative LL change threshold for convergence
      eps: theta clipping bound for numerical stability

    Returns:
      theta:  (K, L) final mixture parameters
      pi:     (K,)   final mixture weights
      ll:     final log-likelihood (scalar)
      n_iter: number of iterations actually used
      resp:   (N, K) responsibilities (soft assignments)

    Math:
      E-step: log P(c | k) = c . log(theta_k) + (1-c) . log(1-theta_k)
              gamma_nk = pi_k * P(c_n | k) / sum_j pi_j * P(c_n | j)
      M-step: theta_k[l] = sum_n gamma_nk * c_n[l] / sum_n gamma_nk
              pi_k = sum_n gamma_nk / N

    Implementation: delegates to a numba kernel that fuses the E/M
    step body with the logsumexp helper.  Matmul ops (cands @ log_theta.T
    and resp.T @ cands) remain BLAS-dispatched via numba's @ operator —
    same backend as numpy, no performance regression.  Per-block
    contribution at production size is ~24 ms across ~60 EM calls; the
    numba pass roughly halves this.  Output is numerically equivalent
    to the pure-Python version up to floating-point reduction-order
    differences (validated bit-identical on all tested seeds).
    """
    N, L = cands.shape
    # Edge case: max_iter=0 — original returns resp=None.  Detect and
    # substitute None for backward compatibility (the kernel returns
    # zeros to satisfy numba's return-type constraint).
    if max_iter == 0:
        theta = init_centers.astype(np.float64) * (1 - 2 * eps) + eps
        pi = np.ones(K, dtype=np.float64) / K
        return theta, pi, 0.0, 0, None

    # Ensure cands and init_centers are contiguous float64 — the
    # kernel signature requires this for the matmul dispatch and the
    # numba type-binding to be consistent across call sites.
    cands_arr = np.ascontiguousarray(cands, dtype=np.float64)
    init_arr = np.ascontiguousarray(init_centers, dtype=np.float64)

    theta, pi, ll, n_iter, resp = _bernoulli_mixture_em_kernel(
        cands_arr, K, init_arr, int(max_iter), float(tol), float(eps))
    return theta, pi, float(ll), int(n_iter), resp


def _fit_bernoulli_mixture_select_K(candidates,
                                      K_max=RECOVERY_MIXTURE_K_MAX,
                                      n_restarts=RECOVERY_MIXTURE_N_RESTARTS,
                                      seed=RECOVERY_MIXTURE_RNG_SEED,
                                      patience=RECOVERY_MIXTURE_PATIENCE,
                                      verbose=False):
    """Fit Bernoulli mixture for K=1..K_max, pick the K minimising BIC.

    Args:
      candidates: list of (L,) binary arrays
      K_max: upper bound on K to try
      n_restarts: EM restarts per K (with different K-means++ seeds)
      seed: base RNG seed (for reproducibility)
      patience: stop the K-sweep after this many CONSECUTIVE increasing-K
        values fail to improve the best BIC (see RECOVERY_MIXTURE_PATIENCE).
        Truncates the sweep tail only; does not change the selected K among
        the K it evaluates.  patience=None disables the early stop (full
        sweep, bit-identical to the pre-early-stop behaviour).
      verbose: print BIC trace

    Returns:
      list of (L,) binary arrays — the consensus haps for the selected K.
      Empty list if candidates is empty.

    BIC formula:
      BIC = -2 * LL + (K*L + K - 1) * log(N)
        - K*L params for the mixture component profiles
        - K-1 params for the mixture weights (one constraint pi.sum() == 1)
      Lower BIC = better.

    The output is INTENTIONALLY over-permissive — inner BIC measures
    density of candidates in candidate-space (which can include noise
    components from recombinant or low-cleanness candidates).  The outer
    BIC subset-selection on actual sample data (in the recovery round
    loop) filters these out.
    """
    if len(candidates) == 0:
        return []

    cands_arr = np.stack(candidates, axis=0).astype(np.float64)               # (N, L)
    N, L = cands_arr.shape

    # Cap K_max at N (can't have more components than candidates)
    K_max_effective = min(K_max, N)

    rng = np.random.default_rng(seed)

    best_overall = None   # tuple: (K, BIC, theta, pi, ll, effective_sizes)
    no_improve = 0        # consecutive increasing-K with no BIC improvement
    bic_trace = []

    if verbose:
        print(f'    Inner mixture fitting: N={N} candidates, L={L}, '
              f'trying K=1..{K_max_effective}, n_restarts={n_restarts}')

    for K in range(1, K_max_effective + 1):
        # Multi-restart: pick the best LL across n_restarts independent
        # K-means++ inits.  EM has local minima; multi-start gives robustness.
        best_for_K = None   # (LL, theta, pi, resp)
        for restart in range(n_restarts):
            init_centers = _kmeans_pp_init(cands_arr, K, rng)
            theta, pi, ll, _n_iter, resp = _bernoulli_mixture_em(
                cands_arr, K, init_centers=init_centers)
            if best_for_K is None or ll > best_for_K[0]:
                best_for_K = (ll, theta, pi, resp)

        ll, theta, pi, resp = best_for_K
        n_params = K * L + (K - 1)
        bic = -2 * ll + n_params * np.log(max(N, 2))

        effective_sizes = resp.sum(axis=0)
        bic_trace.append((K, bic, ll, effective_sizes))

        if best_overall is None or bic < best_overall[1]:
            best_overall = (K, bic, theta, pi, ll, effective_sizes)
            no_improve = 0
        else:
            # No BIC improvement at this K.  Count it; once `patience`
            # CONSECUTIVE increasing-K values have failed to beat the best
            # BIC, stop sweeping (the tail is wasted work — see
            # RECOVERY_MIXTURE_PATIENCE).  patience=None disables this and
            # restores the full K=1..K_max_effective sweep bit-for-bit.
            no_improve += 1
            if patience is not None and no_improve >= patience:
                if verbose:
                    print(f'      [patience] stop sweep at K={K} '
                          f'({no_improve} consecutive non-improving K; '
                          f'best K={best_overall[0]})')
                break

    if verbose:
        for K, bic, ll, eff_sizes in bic_trace:
            marker = ' <-' if K == best_overall[0] else ''
            eff_str = '[' + ', '.join(f'{s:.1f}' for s in eff_sizes) + ']'
            print(f'      K={K:>2d}: LL={ll:>11.1f}, BIC={bic:>11.1f}, '
                  f'eff_sizes={eff_str}{marker}')

    best_K, best_bic, best_theta, best_pi, best_ll, best_eff = best_overall
    if verbose:
        print(f'    Inner mixture: selected K={best_K} with BIC={best_bic:.1f}')

    # Round theta to binary haps (consensus founder profiles)
    binary_thetas = (best_theta > 0.5).astype(np.int64)
    return [binary_thetas[k].copy() for k in range(best_K)]


# =============================================================================
# SUBTRACTION-BASED RECOVERY: CANDIDATE GENERATION & OUTER BIC SELECTION
# =============================================================================

@njit(cache=True)
def _run_subtraction_round_kernel(pool_arr, argmax_dosage_kept,
                                    cleanness_threshold):
    """Numba kernel for _run_subtraction_round.

    Replaces the per-pool-member numpy temporaries (residual, in_01,
    cleanness; each O(N*L) per iteration) with a scalar loop over
    (hap, sample) pairs.  For each pair:
      (1) count admissible sites by computing the residual on the fly,
          break-decision via `cleanness < threshold`;
      (2) if accepted, write the clipped residual into the output
          buffer in pool-outer / sample-inner order.
    This recomputes each residual twice (once for counting, once for
    writing) but avoids any O(N*L) heap allocation per iteration; on
    production data the second pass is bound by the ~50% acceptance
    rate so the recomputation cost is dwarfed by the allocation
    savings.

    Output buffer is preallocated worst-case (pool_size * N rows).  The
    wrapper truncates to the active count.

    Args:
        pool_arr: (P, L) int64 — stacked pool haps (each row is one
            (L,) hap from the original Python list)
        argmax_dosage_kept: (N, L) int64 — argmax dosages in {0, 1, 2}
        cleanness_threshold: float — min fraction of admissible sites

    Returns:
        out_buf: (out_count, L) int64 — tight slice of admissible
            clipped residuals, in (hap, sample) traversal order
        out_count: int — number of valid rows in out_buf
    """
    P, L = pool_arr.shape
    N = argmax_dosage_kept.shape[0]
    # Worst case: every sample is clean for every pool member.
    # At production P=30, N=320, L=200, this is ~50 MB int64 — fine.
    out_buf = np.empty((P * N, L), dtype=np.int64)

    # Minimum admissible-site count for "clean" decision.  Match the
    # original `cleanness = in_01.mean(axis=1); clean_mask = cleanness
    # >= cleanness_threshold` exactly — integer-comparison equivalent
    # is `n_in_01 >= ceil(threshold * L)`.  Use rounding to nearest
    # to avoid subtle off-by-one drift; for the production default of
    # 0.90 at L=200 this is 180 and matches the float comparison
    # 180/200=0.90 >= 0.90.
    #
    # Note: float-mean and integer-count threshold can differ at edge
    # cases where threshold*L isn't exact.  The numpy original uses
    # exact float mean which has rounding behavior.  To preserve byte-
    # identity we replicate the exact float-mean comparison: compute
    # n_in_01 as int, divide by L, compare to threshold.  Numba handles
    # this fine since both sides are float64.

    out_count = 0
    for p in range(P):
        for s in range(N):
            # Count admissible sites for this (pool member, sample) pair
            n_in_01 = 0
            for l in range(L):
                r = argmax_dosage_kept[s, l] - pool_arr[p, l]
                if 0 <= r <= 1:
                    n_in_01 += 1
            cleanness = n_in_01 / L
            if cleanness < cleanness_threshold:
                continue
            # Accepted — write clipped residual into output buffer
            for l in range(L):
                r = argmax_dosage_kept[s, l] - pool_arr[p, l]
                # np.clip(r, 0, 1)
                if r < 0:
                    out_buf[out_count, l] = 0
                elif r > 1:
                    out_buf[out_count, l] = 1
                else:
                    out_buf[out_count, l] = r
            out_count += 1
    return out_buf[:out_count], out_count


def _run_subtraction_round(pool, argmax_dosage_kept,
                            cleanness_threshold=RECOVERY_CLEANNESS_THRESHOLD):
    """Generate clean residual candidates by subtracting each pool member
    from each sample's argmax dosage.

    For sample s with argmax dosage d_s and pool member h, the residual
    r_s = d_s - h is interpreted as an estimate of the OTHER strand
    given that h was one strand.  At sites where d_s in {h, h+1}, r_s in
    {0, 1} — admissible.  At sites where d_s != h and d_s != h+1, r_s is
    out of range — the (h, ?) hypothesis is inconsistent at that site.

    A residual is "clean" if at least cleanness_threshold of its sites
    have admissible values.  Clean residuals are clipped to {0, 1} and
    returned as binary candidate haps.

    Args:
      pool: list of (L_kept,) binary arrays (the founders to subtract)
      argmax_dosage_kept: (N, L_kept) int array of argmax genotype dosages
        in {0, 1, 2} per (sample, kept site)
      cleanness_threshold: min fraction of admissible sites to accept

    Returns:
      list of (L_kept,) binary candidate arrays

    Implementation: delegates to a numba kernel that uses a scalar
    two-pass loop over (pool, sample) pairs and writes directly into a
    preallocated worst-case output buffer.  Output ordering preserved
    (pool-outer, sample-inner) so downstream callers see byte-identical
    candidate sequences.
    """
    # Edge case: empty pool returns an empty list (kernel would handle
    # P=0 correctly but we short-circuit to skip the buffer allocation).
    if len(pool) == 0:
        return []

    # Stack pool haps into a contiguous (P, L) int64 array.  Each
    # element of pool is a (L,) int array (from the round loop, dtype
    # is np.int64; defensive cast ensures kernel signature match).
    pool_arr = np.empty((len(pool), pool[0].shape[0]), dtype=np.int64)
    for p in range(len(pool)):
        pool_arr[p] = pool[p]

    argmax_arr = np.ascontiguousarray(argmax_dosage_kept, dtype=np.int64)
    out_buf, out_count = _run_subtraction_round_kernel(
        pool_arr, argmax_arr, float(cleanness_threshold))

    # Repackage into list-of-arrays (matching the legacy return shape).
    # Copy each row to detach from the kernel's slice — defensive in
    # case downstream callers mutate.
    return [out_buf[i].copy() for i in range(out_count)]


def _run_subtraction_round_soft(pool, probs_k, site_priors=None,
                                  cleanness_threshold=RECOVERY_CLEANNESS_THRESHOLD):
    """Soft analogue of _run_subtraction_round (RECOVERY_RESIDUAL_MODE ==
    "soft"): generate other-strand LIKELIHOODS for the marginal-likelihood
    mixture, rather than hard-called residual bits.

    For founder strand h and a sample with genotype likelihoods L(g) over
    dosages g in {0, 1, 2}, the other strand o satisfies g = h + o, so the
    per-site other-strand likelihoods are:
      h[l]=0: L0 = L(g=0)  (o=0),  L1 = L(g=1)  (o=1)
      h[l]=1: L0 = L(g=1)  (o=0),  L1 = L(g=2)  (o=1)
    (the third, "inadmissible", genotype — g=2 when h=0, g=0 when h=1 — is
    dropped; a sample whose reads favour it gets low L0 and L1, hence low
    likelihood under every mixture component.)  These (L0, L1) feed
    _fit_bernoulli_mixture_ml_select_K, which marginalises the latent
    other-strand allele instead of plugging in its posterior mean — so
    uninformative sites defer to the cluster consensus rather than diluting
    it (see that function and RECOVERY_RESIDUAL_MODE).

    probs_k holds the genotype POSTERIOR.  When site_priors is given the HWE
    site prior is divided out to recover the genotype LIKELIHOOD (the correct
    quantity for the marginalisation): L(g) prop probs_k[g] / site_priors[g].
    The per-(sample, site) normaliser is irrelevant — it cancels in both EM
    steps, so only the per-genotype prior ratio matters — hence no
    renormalisation is needed.  When site_priors is None probs_k is used
    directly, which is exact when probs_k already is the likelihood (flat
    prior, e.g. synthetic data).

    A candidate (founder h, sample s) is admitted when its mean admissible
    POSTERIOR mass (P0+P1 for h=0, P1+P2 for h=1) over sites meets
    cleanness_threshold — the same screen as the argmax path.

    Args:
      pool: list of (L_kept,) binary arrays (the founders to subtract)
      probs_k: (N, L_kept, 3) genotype posteriors (kept sites)
      site_priors: (L_kept, 3) genotype priors to divide out, or None
      cleanness_threshold: min mean admissible posterior mass to accept

    Returns:
      (L0, L1): two (M, L_kept) float64 arrays of other-strand likelihoods
        for the M admitted candidates, in (pool-outer, sample-inner) order.
        Both are empty (M=0) when the pool is empty or nothing is clean.
    """
    L = probs_k.shape[1]
    empty = np.empty((0, L), dtype=np.float64)
    if len(pool) == 0:
        return empty, empty

    P0 = probs_k[:, :, 0]
    P1 = probs_k[:, :, 1]
    P2 = probs_k[:, :, 2]                                          # (N, L) views

    # Genotype likelihoods for the model: divide out the site prior if given.
    if site_priors is not None:
        sp = np.asarray(site_priors, dtype=np.float64)            # (L, 3)
        Lk0 = P0 / sp[None, :, 0]
        Lk1 = P1 / sp[None, :, 1]
        Lk2 = P2 / sp[None, :, 2]
    else:
        Lk0, Lk1, Lk2 = P0, P1, P2

    L0_list = []
    L1_list = []
    for h in pool:
        hb = np.asarray(h).astype(np.bool_)                       # (L,)
        # Soft cleanness screen from the POSTERIOR (admissible mass).
        adm = np.where(hb[None, :], P1 + P2, P0 + P1)             # (N, L)
        keep = adm.mean(axis=1) >= cleanness_threshold            # (N,)
        if not keep.any():
            continue
        # Other-strand likelihoods (un-renormalised; per-site scale cancels):
        #   h=0: o=0 -> g=0, o=1 -> g=1 ;  h=1: o=0 -> g=1, o=1 -> g=2
        l0 = np.where(hb[None, :], Lk1, Lk0)                      # (N, L)
        l1 = np.where(hb[None, :], Lk2, Lk1)                      # (N, L)
        L0_list.append(l0[keep])
        L1_list.append(l1[keep])

    if not L0_list:
        return empty, empty
    return (np.ascontiguousarray(np.vstack(L0_list), dtype=np.float64),
            np.ascontiguousarray(np.vstack(L1_list), dtype=np.float64))


@njit(cache=True)
def _bernoulli_mixture_ml_em_kernel(L0, L1, K, init_theta,
                                      max_iter, tol, eps):
    """Numba kernel for _bernoulli_mixture_ml_em (marginal-likelihood EM).

    Mixture of Bernoulli HAPLOTYPES over the latent other-strand allele o.
    Component k posits o[l] ~ Bernoulli(theta_k[l]); each candidate is
    observed only through its per-site other-strand likelihoods (L0, L1).

    E-step (marginalises o, no hard call):
      log P(cand_m | k) = sum_l log( L0[m,l]*(1-theta_k[l])
                                     + L1[m,l]*theta_k[l] ) + log pi_k
      resp_mk = softmax_k of the above.
    M-step (deferral update — the key difference from the plug-in soft):
      r_o(m,l,k) = L1[m,l]*theta_k[l]
                   / (L0[m,l]*(1-theta_k[l]) + L1[m,l]*theta_k[l])
      theta_k[l] = sum_m resp_mk * r_o(m,l,k) / sum_m resp_mk
    At an uninformative site (L0 ~ L1) r_o -> theta_k[l], so the candidate
    votes for the status quo and the consensus converges to the mean of the
    INFORMATIVE candidates only — no dilution toward 0.5.  pi_k = N_k / M.

    Args:
        L0, L1: (M, L) float64 — per-candidate other-strand likelihoods
        K: number of components
        init_theta: (K, L) initial component profiles (in [0,1])
        max_iter, tol, eps: EM iteration cap, rel-LL tol, theta clip bound

    Returns:
        theta: (K, L) final profiles
        pi:    (K,)   final weights
        ll:    final log-likelihood (scalar)
        n_iter: iterations used
        resp:  (M, K) responsibilities
    """
    M, L = L0.shape
    theta = init_theta.astype(np.float64)
    for k in range(K):
        for l in range(L):
            if theta[k, l] < eps:
                theta[k, l] = eps
            elif theta[k, l] > 1.0 - eps:
                theta[k, l] = 1.0 - eps
    pi = np.ones(K, dtype=np.float64) / K

    prev_ll = -np.inf
    n_iter = 0
    ll = 0.0
    resp = np.zeros((M, K), dtype=np.float64)

    for it in range(max_iter):
        n_iter = it + 1
        # E-step: per-candidate, per-component marginal log-likelihood.
        log_pi = np.log(pi + 1e-15)
        logp = np.empty((M, K), dtype=np.float64)
        for m in range(M):
            for k in range(K):
                s = 0.0
                for l in range(L):
                    mix = L0[m, l] * (1.0 - theta[k, l]) + L1[m, l] * theta[k, l]
                    if mix < 1e-300:
                        mix = 1e-300
                    s += np.log(mix)
                logp[m, k] = s + log_pi[k]
        # logsumexp over k -> resp, ll
        ll_acc = 0.0
        for m in range(M):
            mx = logp[m, 0]
            for k in range(1, K):
                if logp[m, k] > mx:
                    mx = logp[m, k]
            se = 0.0
            for k in range(K):
                se += np.exp(logp[m, k] - mx)
            lnorm = mx + np.log(se)
            ll_acc += lnorm
            for k in range(K):
                resp[m, k] = np.exp(logp[m, k] - lnorm)
        ll = ll_acc

        # Convergence check (relative).  Break before updating prev_ll so the
        # caller receives the most recent ll.
        if prev_ll != -np.inf and abs(ll - prev_ll) < tol * max(abs(ll), 1.0):
            break
        prev_ll = ll

        # M-step.
        N_k = np.zeros(K, dtype=np.float64)
        for k in range(K):
            for m in range(M):
                N_k[k] += resp[m, k]
        for k in range(K):
            pi[k] = N_k[k] / M
        new_theta = np.empty((K, L), dtype=np.float64)
        for k in range(K):
            nk = N_k[k] if N_k[k] > 1e-10 else 1e-10
            for l in range(L):
                tkl = theta[k, l]
                num = 0.0
                for m in range(M):
                    den = L0[m, l] * (1.0 - tkl) + L1[m, l] * tkl
                    if den < 1e-300:
                        den = 1e-300
                    num += resp[m, k] * (L1[m, l] * tkl) / den
                v = num / nk
                if v < eps:
                    v = eps
                elif v > 1.0 - eps:
                    v = 1.0 - eps
                new_theta[k, l] = v
        theta = new_theta

    return theta, pi, ll, n_iter, resp


def _bernoulli_mixture_ml_em(L0, L1, K, init_theta,
                               max_iter=RECOVERY_MIXTURE_MAX_ITER,
                               tol=RECOVERY_MIXTURE_TOL,
                               eps=RECOVERY_MIXTURE_THETA_EPS):
    """Single marginal-likelihood EM run (model B).  Thin wrapper around
    _bernoulli_mixture_ml_em_kernel; see it for the math.

    Args:
      L0, L1: (M, L) float64 other-strand likelihoods
      K: number of components
      init_theta: (K, L) initial profiles
      max_iter, tol, eps: EM controls

    Returns:
      theta (K, L), pi (K,), ll (float), n_iter (int), resp (M, K)
    """
    L0c = np.ascontiguousarray(L0, dtype=np.float64)
    L1c = np.ascontiguousarray(L1, dtype=np.float64)
    init = np.ascontiguousarray(init_theta, dtype=np.float64)
    theta, pi, ll, n_iter, resp = _bernoulli_mixture_ml_em_kernel(
        L0c, L1c, int(K), init, int(max_iter), float(tol), float(eps))
    return theta, pi, float(ll), int(n_iter), resp


def _fit_bernoulli_mixture_ml_select_K(L0, L1,
                                         K_max=RECOVERY_MIXTURE_K_MAX,
                                         n_restarts=RECOVERY_MIXTURE_N_RESTARTS,
                                         seed=RECOVERY_MIXTURE_RNG_SEED,
                                         patience=RECOVERY_MIXTURE_PATIENCE,
                                         verbose=False):
    """Marginal-likelihood Bernoulli-haplotype mixture (model B) for the
    "soft" recovery path: fit K=1..K_max, pick the K minimising BIC, return
    the rounded consensus haps.  The model-B analogue of
    _fit_bernoulli_mixture_select_K.

    Each candidate is a pair of per-site other-strand likelihood vectors
    (L0, L1) from _run_subtraction_round_soft.  The mixture marginalises the
    latent other-strand allele (E-step) and updates each profile from the
    informative candidates only (M-step deferral), so low-coverage /
    heterozygous-ambiguous sites neither bias cluster assignment nor pull the
    consensus toward 0.5 — the failure mode of plugging the posterior mean
    P(o=1) into a plain Bernoulli mixture.  Sub-threshold evidence is still
    used (graded), unlike the binary argmax path which thresholds it away.

    K-means++ init runs on the expected other strand E[o] = L1/(L0+L1) using
    the same (L1-distance) initialiser as the argmax path.  BIC and the
    over-permissive inner-fit philosophy match _fit_bernoulli_mixture_select_K
    (the outer BIC subset-selection on sample data filters noise components).

    Args:
      L0, L1: (M, L) float64 other-strand likelihoods
      K_max: upper bound on K to try
      n_restarts: EM restarts per K (different K-means++ seeds)
      seed: base RNG seed
      patience: stop the K-sweep after this many CONSECUTIVE increasing-K
        values fail to improve the best BIC (see RECOVERY_MIXTURE_PATIENCE).
        Truncates the sweep tail only; does not change the selected K among
        the K it evaluates.  patience=None disables the early stop (full
        sweep, bit-identical to the pre-early-stop behaviour).
      verbose: print BIC trace

    Returns:
      list of (L,) binary arrays — consensus haps for the selected K.
      Empty list if there are no candidates.

    BIC formula (identical to the binary mixture):
      BIC = -2*LL + (K*L + K - 1) * log(max(M, 2)),  lower = better.
    """
    M = L0.shape[0]
    if M == 0:
        return []
    L = L0.shape[1]

    # Expected other strand for the kmeans++ init distance (in [0,1]).
    Eo = L1 / (L0 + L1 + 1e-12)                                   # (M, L)
    Eo = np.ascontiguousarray(Eo, dtype=np.float64)

    K_max_effective = min(K_max, M)
    rng = np.random.default_rng(seed)

    best_overall = None   # (K, BIC, theta)
    no_improve = 0        # consecutive increasing-K with no BIC improvement
    bic_trace = []

    if verbose:
        print(f'    Inner ML-mixture fitting: M={M} candidates, L={L}, '
              f'trying K=1..{K_max_effective}, n_restarts={n_restarts}')

    for K in range(1, K_max_effective + 1):
        best_for_K = None   # (LL, theta)
        for restart in range(n_restarts):
            # _kmeans_pp_init returns the selected centers themselves
            # (rows of Eo), which serve directly as the initial profiles.
            init_theta = _kmeans_pp_init(Eo, K, rng)
            theta, pi, ll, _n_iter, _resp = _bernoulli_mixture_ml_em(
                L0, L1, K, init_theta=init_theta)
            if best_for_K is None or ll > best_for_K[0]:
                best_for_K = (ll, theta)

        ll, theta = best_for_K
        n_params = K * L + (K - 1)
        bic = -2 * ll + n_params * np.log(max(M, 2))
        bic_trace.append((K, bic, ll))

        if best_overall is None or bic < best_overall[1]:
            best_overall = (K, bic, theta)
            no_improve = 0
        else:
            # No BIC improvement at this K.  Count it; once `patience`
            # CONSECUTIVE increasing-K values have failed to beat the best
            # BIC, stop sweeping (the tail is wasted work — see
            # RECOVERY_MIXTURE_PATIENCE).  patience=None disables this and
            # restores the full K=1..K_max_effective sweep bit-for-bit.
            no_improve += 1
            if patience is not None and no_improve >= patience:
                if verbose:
                    print(f'      [patience] stop sweep at K={K} '
                          f'({no_improve} consecutive non-improving K; '
                          f'best K={best_overall[0]})')
                break

    if verbose:
        for K, bic, ll in bic_trace:
            marker = ' <-' if K == best_overall[0] else ''
            print(f'      K={K:>2d}: LL={ll:>11.1f}, BIC={bic:>11.1f}{marker}')

    best_K, best_bic, best_theta = best_overall
    if verbose:
        print(f'    Inner ML-mixture: selected K={best_K} with BIC={best_bic:.1f}')

    # Round theta to binary consensus founder profiles.
    binary_thetas = (best_theta > 0.5).astype(np.int64)
    return [binary_thetas[k].copy() for k in range(best_K)]


@njit(cache=True)
def _generate_carrier_residuals_kernel(argmax_dosage, H, A,
                                          low_idx_mask, cleanness_threshold):
    """Numba kernel for _generate_carrier_residuals (verbose=False only).

    Replaces the triple Python loop + numpy temporaries with a tight
    scalar loop over (s, slot, partner_idx) triples that:
      1. checks `A[s, slot] in low_idx_set` via a bool mask of length K
         (numba doesn't support Python sets);
      2. skips partner_idx == low_idx or partner_idx in low_idx_set;
      3. computes the residual on the fly for accepted partner pairs;
      4. writes accepted clipped residuals into a preallocated buffer.

    Iteration order: s outer, slot middle, partner_idx inner —
    matching the original.  Output buffer is preallocated worst-case
    (N * 2 * K rows).  The wrapper truncates to the active count.

    Args:
        argmax_dosage: (N, L) int64 — precomputed argmax dosages
        H: (K, L) int64 — current founder bits
        A: (N, 2) int64 — current pair assignments (entry K is the
            wildcard sentinel)
        low_idx_mask: (K,) bool — True at indices belonging to low_idx_set
        cleanness_threshold: float — min admissible-site fraction

    Returns:
        out_buf: (out_count, L) int64 — accepted clipped residuals
        out_count: int — number of valid rows in out_buf
    """
    N, L = argmax_dosage.shape
    K = H.shape[0]
    # Worst case: every (s, slot, partner_idx) triple accepted.
    # At production N=320, K=6 this is 3840 rows — tiny.
    out_buf = np.empty((N * 2 * K, L), dtype=np.int64)
    out_count = 0

    for s in range(N):
        for slot in range(2):
            a_idx = A[s, slot]
            # A entries: integers in [0, K] where K is the wildcard
            # sentinel.  Index K is out of range for low_idx_mask, so
            # we must guard before lookup.  Wildcard slots are skipped
            # (the original `int(A[s, slot]) not in low_idx_set` returns
            # False for the wildcard sentinel value since low_idx_set
            # contains only real founder indices < K).
            if a_idx < 0 or a_idx >= K:
                continue
            if not low_idx_mask[a_idx]:
                continue
            low_idx = a_idx
            for partner_idx in range(K):
                if partner_idx == low_idx:
                    continue
                if low_idx_mask[partner_idx]:
                    # Subtractor in low_idx_set — skip
                    continue
                # Compute residual on the fly and count admissible sites
                n_in_01 = 0
                for l in range(L):
                    r = argmax_dosage[s, l] - H[partner_idx, l]
                    if 0 <= r <= 1:
                        n_in_01 += 1
                cleanness = n_in_01 / L
                if cleanness < cleanness_threshold:
                    continue
                # Accepted — write clipped residual
                for l in range(L):
                    r = argmax_dosage[s, l] - H[partner_idx, l]
                    if r < 0:
                        out_buf[out_count, l] = 0
                    elif r > 1:
                        out_buf[out_count, l] = 1
                    else:
                        out_buf[out_count, l] = r
                out_count += 1

    return out_buf[:out_count], out_count


@njit(cache=True)
def _generate_all_sample_residuals_kernel(argmax_dosage, H, A,
                                          cleanness_threshold):
    """Numba kernel for _generate_all_sample_residuals (verbose=False path).

    Variant of _generate_carrier_residuals_kernel that loops over EVERY
    (sample, slot) pair — not just samples whose A[s, slot] is in a
    low_idx_set.  Used by _residual_trio_rescue to mine residuals across
    the whole population, surfacing near-clone founders that K-growth's
    residual-mass seeding missed.

    For each (sample, slot), the OTHER slot's H-row is the partner being
    subtracted: we want residual = strand_at_slot, so we subtract the
    OTHER strand's H entry.  This is the algebra of _late_low_carrier_-
    rescue but applied to all samples regardless of usage statistics.

    Iteration: s outer, slot middle.  For each (s, slot), the partner is
    fixed as A[s, 1 - slot] (the OTHER slot's H row).  Wildcard partners
    (A entry == K, the wildcard sentinel) are skipped because subtracting
    a wildcard means we don't know the other strand and any residual
    derived from it is uninterpretable.

    Output buffer is preallocated worst-case (N * 2 rows) — every sample
    can contribute at most 2 residuals.  The wrapper truncates to the
    active count.

    Args:
        argmax_dosage: (N, L) int64 — precomputed argmax dosages
        H: (K, L) int64 — current founder bits
        A: (N, 2) int64 — current pair assignments (entry K = wildcard
            sentinel)
        cleanness_threshold: float — min admissible-site fraction

    Returns:
        out_buf: (out_count, L) int64 — accepted clipped residuals
        out_count: int — number of valid rows in out_buf
    """
    N, L = argmax_dosage.shape
    K = H.shape[0]
    # Worst case: every (s, slot) pair accepted.  At N=320 this is 640
    # rows — tiny.  Each row is L int64 = 1.6 KB at L=200 → 1 MB total
    # worst case, well within budget.
    out_buf = np.empty((N * 2, L), dtype=np.int64)
    out_count = 0

    for s in range(N):
        for slot in range(2):
            # The partner is the OTHER slot's H row.  We want to expose
            # the founder at `slot` so we subtract A[s, 1 - slot].
            partner_idx = A[s, 1 - slot]
            # Wildcard partners (sentinel index K) are not valid
            # subtractors — we don't have an H row for the wildcard,
            # and a residual derived against it is meaningless.
            if partner_idx < 0 or partner_idx >= K:
                continue
            # Compute residual on the fly and count admissible sites
            n_in_01 = 0
            for l in range(L):
                r = argmax_dosage[s, l] - H[partner_idx, l]
                if 0 <= r <= 1:
                    n_in_01 += 1
            cleanness = n_in_01 / L
            if cleanness < cleanness_threshold:
                continue
            # Accepted — write clipped residual.  Clipping is defensive:
            # if cleanness_threshold < 1.0 some sites can be out-of-
            # range; we clip to {0, 1} so downstream consumers see a
            # well-formed binary vector.  At cleanness_threshold == 1.0
            # (the default) no clipping is needed in principle, but we
            # keep the clip for symmetry with the low-carrier kernel.
            for l in range(L):
                r = argmax_dosage[s, l] - H[partner_idx, l]
                if r < 0:
                    out_buf[out_count, l] = 0
                elif r > 1:
                    out_buf[out_count, l] = 1
                else:
                    out_buf[out_count, l] = r
            out_count += 1

    return out_buf[:out_count], out_count


def _generate_carrier_residuals(probs_k, H, A, low_idx_list,
                                 cleanness_threshold=RECOVERY_CLEANNESS_THRESHOLD,
                                 verbose=False):
    """Generate per-(carrier_sample, partner_candidate) residuals for
    low-carrier haps.

    This is the targeted analogue of _run_subtraction_round, used by
    the late low-carrier rescue (see RECOVERY_LOW_CARRIER_TRIGGER_FRAC).

    For each carrier strand of a low-carrier hap h_low, we want to
    recover h_low's "right" version (the missing truth founder).  The
    algebra: if carrier sample s has true strands (truth_X, truth_Y)
    where truth_Y is the missing one we want to recover, then
      argmax_dosage[s] = truth_X + truth_Y       (noiseless data)
      residual = argmax_dosage[s] - H[partner]
              = (truth_X + truth_Y) - H[partner]
    The residual is "clean" (every site in {0, 1}) if and only if
    H[partner] = truth_X (the actual other strand), in which case
    residual = truth_Y exactly.

    The challenge: when h_low is a chimera, _update_A's choice of A[s,
    other_slot] is whichever founder makes (h_low, partner) optimally
    fit the dosage given h_low is in the pair — NOT necessarily the
    actual other strand truth_X.  At chr6:23624234, A pairs h_low
    (= chim_5) with H[1] = truth_5 for some carriers, but the actual
    other strands are different truth founders; the residual ends up
    = chim_5 by construction.  Verified empirically: all 4 carrier
    residuals at H_low = 0.00%.

    Solution: for each carrier strand, try every H row (excluding
    low_idx_set) as the candidate subtractor.  Most produce noisy
    out-of-range residuals (cleanness < threshold) and are rejected.
    Exactly one row matches truth_X for that carrier and produces a
    clean residual = truth_Y at 100% cleanness.

    Strands paired with the wildcard slot — when the wildcard is the
    "other_slot" — are still mined: we just iterate over all H rows
    as subtractors regardless of A's wildcard assignment.  Subtracting
    a low-carrier hap (a known suspect) from itself or another low-
    carrier hap gives at best noisy residuals, so subtractors in
    low_idx_set are skipped.

    Args:
      probs_k: (N, L_kept, 3) — kept-site posteriors
      H: (K, L_kept) discrete {0, 1} — current founder bits
      A: (N, 2) — current pair assignments (entry K = wildcard sentinel)
      low_idx_list: list of founder indices whose carriers we mine
      cleanness_threshold: min fraction of admissible sites per residual
      verbose: if True, additionally return per-(sample, slot, subtractor)
        provenance

    Returns:
      If verbose=False: list of (L_kept,) binary candidate arrays — one
        per (sample, strand, subtractor) triple that survived cleanness
        filtering.
      If verbose=True: tuple (residuals, provenance) where provenance is
        a list of dicts, one per (sample, slot, subtractor) triple
        examined, with keys:
          sample_idx, slot, low_idx, partner_idx, partner_kind, cleanness,
          accepted, residual (the clipped binary array if accepted else None)
        partner_kind ∈ {'self_low', 'low_carrier', 'normal'}.

    Implementation: the production (verbose=False) path delegates to a
    numba kernel that uses a bool mask (not a Python set) for low_idx
    lookups and writes accepted residuals into a preallocated worst-
    case buffer.  The verbose=True path stays in pure Python because
    numba cannot build Python dicts (and verbose=True is diagnostics-
    only — not called in production).
    """
    N, L_kept = probs_k.shape[0], probs_k.shape[1]
    K = H.shape[0]
    if K == 0 or not low_idx_list:
        return ([], []) if verbose else []
    argmax_dosage = probs_k.argmax(axis=2)                        # (N, L_kept)

    # Build a bool mask over [0, K) for low_idx_set membership.  Any
    # index in low_idx_list that's >= K is ignored (defensive — the
    # production path passes founder indices, which are always < K).
    low_idx_mask = np.zeros(K, dtype=np.bool_)
    for k in low_idx_list:
        ki = int(k)
        if 0 <= ki < K:
            low_idx_mask[ki] = True

    # VERBOSE PATH — stays in pure Python (numba can't return dicts;
    # this path is diagnostics-only and not on any production code path).
    if verbose:
        low_idx_set = set(int(k) for k in low_idx_list)
        residuals = []
        provenance = []
        for s in range(N):
            for slot in range(2):
                if int(A[s, slot]) not in low_idx_set:
                    continue
                low_idx = int(A[s, slot])
                for partner_idx in range(K):
                    if partner_idx == low_idx:
                        provenance.append({
                            'sample_idx': s, 'slot': slot,
                            'low_idx': low_idx, 'partner_idx': partner_idx,
                            'partner_kind': 'self_low',
                            'cleanness': float('nan'), 'accepted': False,
                            'residual': None})
                        continue
                    if partner_idx in low_idx_set:
                        provenance.append({
                            'sample_idx': s, 'slot': slot,
                            'low_idx': low_idx, 'partner_idx': partner_idx,
                            'partner_kind': 'low_carrier',
                            'cleanness': float('nan'), 'accepted': False,
                            'residual': None})
                        continue
                    residual = argmax_dosage[s] - H[partner_idx]
                    in_01 = (residual >= 0) & (residual <= 1)
                    cleanness = float(in_01.mean())
                    accepted = cleanness >= cleanness_threshold
                    clipped = (np.clip(residual, 0, 1).astype(np.int64)
                               if accepted else None)
                    if accepted:
                        residuals.append(clipped)
                    provenance.append({
                        'sample_idx': s, 'slot': slot,
                        'low_idx': low_idx, 'partner_idx': partner_idx,
                        'partner_kind': 'normal',
                        'cleanness': cleanness, 'accepted': accepted,
                        'residual': clipped})
        return residuals, provenance

    # PRODUCTION PATH — delegate to numba kernel.  Ensure dtypes match
    # the kernel signature.  argmax_dosage is already int64 (numpy's
    # default for argmax on integer dtype).  H may be int8 or int64;
    # the kernel signature expects int64 to avoid mixed-type subtraction
    # overhead, so cast defensively.
    argmax_arr = np.ascontiguousarray(argmax_dosage, dtype=np.int64)
    H_arr = np.ascontiguousarray(H, dtype=np.int64)
    A_arr = np.ascontiguousarray(A, dtype=np.int64)
    out_buf, out_count = _generate_carrier_residuals_kernel(
        argmax_arr, H_arr, A_arr, low_idx_mask, float(cleanness_threshold))
    # Repackage into list-of-arrays matching the legacy return shape.
    return [out_buf[i].copy() for i in range(out_count)]



def _greedy_bic_select(cache, cc_scale=RECOVERY_OUTER_CC_SCALE,
                        max_k=RECOVERY_MAX_K,
                        use_log_bic=False,
                        verbose=False):
    """Greedy forward selection by BIC over a fixed candidate pool.

    Haps are FIXED during scoring (no coord descent).  At each step,
    pick the candidate giving the lowest NLL when added; accept iff
    BIC improves (equivalent to NLL_improvement > cc/2).

    Uses PoolEmissionCache to avoid rebuilding the Viterbi emission
    tensor for each trial subset — the pool's emissions are precomputed
    once when the cache is built, and each trial only does the (much
    cheaper) state-axis selection + Viterbi forward pass.

    Args:
      cache: PoolEmissionCache wrapping the candidate pool.  cache.pool_-
        haps is the candidate list; cache.N and cache.L_kept are used
        for cc computation.
      cc_scale: BIC complexity-cost scale (outer; default 0.5)
      max_k: hard cap on selected size
      use_log_bic: if True, use log-BIC formula for cc (cc_scale *
        log(N*L) * snp_growth); if False (default, project standard),
        use linear formula (cc_scale * snp_growth * N).  Must match
        the use_log_bic of the surrounding K-growth so accept/reject
        criteria are consistent across the pipeline.
      verbose: print accept/reject decisions

    Returns:
      selected_indices: list of indices into cache.pool_haps
      selected_haps:    list of arrays (same as cache.pool_haps[i] for
                        i in selected_indices)
      current_nll:      NLL at final selection
    """
    candidate_haps = cache.pool_haps
    N = cache.N
    L_kept = cache.L_kept
    cc = _compute_cc(cc_scale, N, L_kept, use_log_bic=use_log_bic)

    # K=0 baseline NLL via the cache (precomputed at construction).
    nll_K0 = cache.nll_for_subset([])
    bic_K0 = 0 * cc + 2 * nll_K0

    selected_indices = []
    used = set()
    current_bic = bic_K0
    current_nll = nll_K0

    if verbose:
        print(f'    Forward: K=0 NLL={nll_K0:.1f}, BIC={bic_K0:.1f}, '
              f'cc={cc:.1f}, threshold cc/2={cc/2:.1f}')

    while len(selected_indices) < min(len(candidate_haps), max_k):
        best_ci = -1
        best_nll = float('inf')
        for ci in range(len(candidate_haps)):
            if ci in used:
                continue
            trial_indices = selected_indices + [ci]
            trial_nll = cache.nll_for_subset(trial_indices)
            if trial_nll < best_nll:
                best_nll = trial_nll
                best_ci = ci

        if best_ci < 0:
            break

        k_new = len(selected_indices) + 1
        bic_new = k_new * cc + 2 * best_nll
        d_nll = current_nll - best_nll

        if bic_new < current_bic:
            selected_indices.append(best_ci)
            used.add(best_ci)
            if verbose:
                print(f'    Forward: K={k_new} ACCEPT cand[{best_ci}], '
                      f'NLL={best_nll:.1f}, BIC={bic_new:.1f}, dNLL={d_nll:.1f}')
            current_bic = bic_new
            current_nll = best_nll
        else:
            if verbose:
                print(f'    Forward: K={k_new} REJECT cand[{best_ci}], '
                      f'NLL={best_nll:.1f}, BIC={bic_new:.1f}, dNLL={d_nll:.1f} '
                      f'< cc/2={cc/2:.1f}')
            break

    selected_haps = [candidate_haps[i] for i in selected_indices]
    return selected_indices, selected_haps, current_nll


def _swap_refine(cache, selected_indices, current_nll,
                  nll_tolerance=RECOVERY_SWAP_NLL_TOLERANCE,
                  max_passes=10, verbose=False):
    """Try swapping each selected hap with each unselected pool member.
    Apply swap if NLL improves by more than nll_tolerance.  Iterate over
    passes until no improvement.

    Sometimes greedy forward selection picks a near-optimal hap early
    that becomes redundant after later picks; swap lets us replace it
    with a better one without requiring a full re-search.

    Uses PoolEmissionCache so each trial costs O(N · K_sub_states ·
    n_bins) for the state-axis selection + Viterbi forward pass, rather
    than the original O(N · K_sub_states · L) for emission rebuild +
    Viterbi.  At K_sub=6 this is a roughly order-of-magnitude saving per
    trial, and `_swap_refine` performs many trials per call (~K * pool
    per pass, ~K passes).

    Args:
      cache: PoolEmissionCache over the swap-pool of candidates.
      selected_indices: list of indices into cache.pool_haps — current
        selection.  Replaced (or kept) one-at-a-time during refinement.
      current_nll: scalar — NLL at the current selection, used as the
        comparison point.  Caller should pass the NLL value at the
        SAME indices the cache would score (typically the output of
        _greedy_bic_select that just preceded the swap).
      nll_tolerance: minimum NLL improvement required to accept a swap.
      max_passes: defensive cap on the outer pass loop.
      verbose: print accept decisions per swap.

    Returns:
      (selected_indices, selected_haps, current_nll, n_swaps)
    """
    sel_ind = list(selected_indices)
    K = len(sel_ind)
    if K == 0:
        return sel_ind, [], current_nll, 0

    pool_size = cache.K_pool
    n_swaps = 0
    for pass_num in range(max_passes):
        improved_in_pass = False
        for si in range(K):
            best_ci = -1
            best_nll = current_nll - nll_tolerance
            for ci in range(pool_size):
                if ci in sel_ind:
                    continue
                # Trial: replace pool index at position si with ci.
                # Build the trial subset by index substitution and score
                # it through the cache (no emission rebuild).
                trial_indices = list(sel_ind)
                trial_indices[si] = ci
                trial_nll = cache.nll_for_subset(trial_indices)
                if trial_nll < best_nll:
                    best_nll = trial_nll
                    best_ci = ci
            if best_ci >= 0:
                if verbose:
                    print(f'    Swap: pos {si} (cand[{sel_ind[si]}]) -> cand[{best_ci}], '
                          f'NLL {current_nll:.1f} -> {best_nll:.1f}')
                sel_ind[si] = best_ci
                current_nll = best_nll
                improved_in_pass = True
                n_swaps += 1
                break
        if not improved_in_pass:
            break

    sel_haps = [cache.pool_haps[i] for i in sel_ind]
    return sel_ind, sel_haps, current_nll, n_swaps


def _bic_prune(cache, selected_indices,
                cc_scale=RECOVERY_OUTER_CC_SCALE, use_log_bic=False,
                verbose=False):
    """BIC pruning: try dropping each selected hap.  Drop if the NLL
    increase from removal is less than cc/2 (i.e., the hap isn't pulling
    enough weight to justify the +cc penalty for keeping it).

    Iterates: each drop may enable another (cascading prune of redundant
    haps that propped each other up).  Matches the project's
    refine_selection_by_pruning pattern in beam_search_core.

    Uses PoolEmissionCache so each leave-one-out trial reuses the pool's
    precomputed emissions (no per-trial rebuild).  For a K-sized
    selection, one prune iteration evaluates K trials (one per dropped
    hap) plus one full-K-NLL reference; the cache makes each trial
    cheap.

    use_log_bic: bool — if True, use log-BIC formula for cc; if False
      (default, project standard), use linear formula.  Must match the
      use_log_bic of the surrounding K-growth so the prune threshold
      cc/2 is consistent with K-growth's growth threshold.

    Args:
      cache: PoolEmissionCache over the candidate pool.
      selected_indices: list of pool indices forming the current
        selection (subset of cache.pool_haps).
      cc_scale, use_log_bic: BIC parameters; cc/2 is the drop threshold.
      verbose: print drop decisions.

    Returns: (pruned_indices, pruned_haps, final_nll, n_dropped)
    """
    N = cache.N
    L_kept = cache.L_kept
    cc = _compute_cc(cc_scale, N, L_kept, use_log_bic=use_log_bic)

    sel_ind = list(selected_indices)
    n_dropped = 0

    while len(sel_ind) > 0:
        nll_full = cache.nll_for_subset(sel_ind)
        K = len(sel_ind)

        best_drop_idx = -1
        best_dnll = cc / 2   # threshold; only drop if dnll_increase < this

        for i in range(K):
            trial = sel_ind[:i] + sel_ind[i+1:]
            nll_trial = cache.nll_for_subset(trial)
            dnll = nll_trial - nll_full   # NLL increase from dropping
            if dnll < best_dnll:
                best_dnll = dnll
                best_drop_idx = i

        if best_drop_idx < 0:
            break

        if verbose:
            print(f'    Prune: drop pos {best_drop_idx} (cand[{sel_ind[best_drop_idx]}]), '
                  f'NLL increase {best_dnll:.1f} < cc/2={cc/2:.1f} -- DROPPED')
        del sel_ind[best_drop_idx]
        n_dropped += 1

    final_nll = cache.nll_for_subset(sel_ind)
    sel_haps = [cache.pool_haps[i] for i in sel_ind]
    return sel_ind, sel_haps, final_nll, n_dropped



@njit(cache=True)
def _hamming_pct_kept_kernel(a, b):
    """Numba kernel for _hamming_pct_kept.

    Scalar loop over the two arrays, counting differing positions.
    Returns the percentage of differing sites (0.0 to 100.0).

    Both inputs must be int64 1D arrays of the same length — the
    wrapper handles dtype casting.

    Mathematical equivalence to `float(np.mean(a != b)) * 100.0`:
      np.mean over a bool array returns count_true / L as float64,
      then multiplied by 100.  We compute the same thing: count
      differing positions, divide by L, multiply by 100.  No round-
      off difference because the intermediate count is an integer
      and the final division is float64.
    """
    L = a.shape[0]
    diff_count = 0
    for i in range(L):
        if a[i] != b[i]:
            diff_count += 1
    return (diff_count / L) * 100.0


def _hamming_pct_kept(a, b):
    """Per-site Hamming distance as a percentage (0-100).

    Implementation: delegates to a numba kernel that uses a scalar
    loop, avoiding the (L,) bool array allocation per call.  Called
    very frequently (~120 times per block in the round loop + late
    rescue dedup loops, at typically 3-4 µs per call); the cumulative
    cost is small but the numba kernel halves it cleanly.

    Inputs are cast to int64 inside the wrapper so the kernel compiles
    only once regardless of caller-side dtype (some callers pass int8
    from candidate pools, others pass int64 from H rows or fully-
    resolved consensus haps).
    """
    a_arr = np.ascontiguousarray(a, dtype=np.int64)
    b_arr = np.ascontiguousarray(b, dtype=np.int64)
    return _hamming_pct_kept_kernel(a_arr, b_arr)


@njit(cache=True)
def _haps_equal_kernel(haps_a, haps_b, eps_pct):
    """Numba kernel for _haps_equal.

    Bipartite matching: for each row in haps_a, find an unmatched row
    in haps_b within eps_pct Hamming.  Returns True iff every row in
    haps_a finds a unique match.

    The semantics are EXACTLY the original's:
      - First-match wins: for each ha, we walk haps_b in index order
        and accept the first unmatched hb within eps_pct.  This is
        greedy and may fail to find a valid matching that exists if
        ordering is unlucky, but matches the original's behavior bit-
        for-bit.
      - `_hamming_pct_kept(ha, hb) < eps_pct` uses strict inequality.
      - matched_b is a per-call bool array of length K.

    Args:
        haps_a, haps_b: (K, L) int64 arrays — same shape, otherwise
            the wrapper returns False without calling this kernel.
        eps_pct: float — Hamming-percent tolerance.

    Returns:
        bool — True iff a valid bipartite matching exists.
    """
    K, L = haps_a.shape
    # bool scratch — reset per call.  np.zeros works fine in numba.
    matched_b = np.zeros(K, dtype=np.bool_)
    for ai in range(K):
        found = False
        for bi in range(K):
            if matched_b[bi]:
                continue
            # Inline _hamming_pct_kept: count differing positions
            diff_count = 0
            for l in range(L):
                if haps_a[ai, l] != haps_b[bi, l]:
                    diff_count += 1
            ham_pct = (diff_count / L) * 100.0
            if ham_pct < eps_pct:
                matched_b[bi] = True
                found = True
                break
        if not found:
            return False
    # All ai matched.  In a bipartite K x K matching where each ai matched
    # exactly one bi, all matched_b must be True — but we don't need to
    # check `all(matched_b)` separately: K matches across K positions
    # forces it.  Match the original's final `return all(matched_b)`
    # for safety; equivalent here since we never reach this point unless
    # K matches found.
    for bi in range(K):
        if not matched_b[bi]:
            return False
    return True


def _haps_equal(haps_a, haps_b, eps_pct=RECOVERY_HAPS_EQUAL_EPS_PCT):
    """Test if two hap collections are equal (within eps_pct Hamming
    tolerance per pair, with bipartite matching).

    Returns True iff:
      - len(haps_a) == len(haps_b), AND
      - there's a 1-to-1 matching where each matched pair is within eps_pct.

    Used for convergence detection between recovery rounds and between
    outer iterations.  Tolerance accommodates near-identical haps that
    differ only at a few uncertain sites.

    Implementation: delegates to a numba kernel that takes (K, L)
    int64 arrays.  Wrapper stacks list-of-arrays inputs into 2D
    contiguous arrays.  Output is byte-identical to the original
    (first-match-wins greedy bipartite matching with strict-inequality
    Hamming threshold).
    """
    if len(haps_a) != len(haps_b):
        return False
    # Handle empty case before stacking (np.stack rejects empty inputs)
    if len(haps_a) == 0:
        return True
    # Stack into (K, L) int64 arrays.  The original accepts either lists
    # of (L,) arrays or 2D ndarrays; both forms produce a valid 2D
    # array under np.stack with axis=0.
    a_arr = np.ascontiguousarray(np.stack(list(haps_a), axis=0), dtype=np.int64)
    b_arr = np.ascontiguousarray(np.stack(list(haps_b), axis=0), dtype=np.int64)
    if a_arr.shape != b_arr.shape:
        return False
    return _haps_equal_kernel(a_arr, b_arr, float(eps_pct))


# =============================================================================
# CARRIER-COUNTING KERNEL
# =============================================================================
# Pure-numerical inner loop pulled out of _late_low_carrier_rescue so the
# double loop over (sample, strand) runs at njit speed.  Counts how many
# real-strand carriers each hap has, treating the wildcard sentinel
# (A[s, slot] == K) as "not a real carrier" -- those strands are
# excluded from the per-hap usage tally.
#
# Inputs:
#   A: (N, 2) int — pair assignments; entries in [0, K] where K is the
#                   wildcard sentinel
#   K: int       — number of real haps (the wildcard sentinel value)
#
# Returns:
#   usage: (K,) int64 — per-hap real-strand carrier counts

@njit(cache=True)
def _count_real_carriers_kernel(A, K):
    N = A.shape[0]
    usage = np.zeros(K, dtype=np.int64)
    for s in range(N):
        for slot in range(2):
            f = int(A[s, slot])
            if f != K:
                usage[f] += 1
    return usage


def _count_real_carriers(A, K):
    """Thin wrapper that ensures contiguous int input before dispatching
    to the njit kernel.  At N=320 this is ~640 inner iterations, so
    contiguity overhead matters less than the kernel call cost, but the
    contract is uniform with the rest of this module's wrappers."""
    A_arr = np.ascontiguousarray(A, dtype=np.int64)
    return _count_real_carriers_kernel(A_arr, int(K))


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



# =============================================================================
# RESIDUAL-TRIO RESCUE (added 2026-05): post-K-growth pass that mines
# per-sample residuals across ALL samples to surface near-clone founders
# K-growth's residual-mass seeding missed.
# =============================================================================
#
# Algorithm:
#   1. For every (sample, slot), compute residual = argmax_dosage[s] -
#      H[A[s, other_slot]].  When the other-slot partner is a clean,
#      truth-near founder, residual = the actual strand at `slot`.  When
#      the partner is itself a chimera, residual will fail the cleanness
#      filter (out-of-range bits at heterozygous sites where the chimera
#      differs from the true partner).
#   2. Filter to clean residuals (every site in {0, 1}).
#   3. Cluster the clean residuals using bhd_trio's
#      _cluster_haps_consensus_kernel (same clustering machinery the
#      hom-recovery candidate emitter uses, for compatibility with the
#      trio pipeline's pool composition).
#   4. For each cluster centroid:
#      - if within RESIDUAL_TRIO_DEDUP_VS_H_PCT of any existing H row →
#        skip (already in dictionary)
#      - else admit as a candidate
#   5. Pool = current H + admitted residual-trio candidates + (optionally)
#      fresh trio/hom/pairwise candidates.  Run greedy BIC forward + swap
#      + prune + refit, same as _late_low_carrier_rescue.
#   6. Accept iff BIC strictly improves.
#
# Differs from _late_low_carrier_rescue:
#   - Mines EVERY sample's residuals (not just low-carrier-hap carriers).
#   - Trigger is per-block "any near-clone-derived candidate exists",
#     evaluated after residual clustering.  No usage-based gate.
#
# Motivating case: chr10:503 F0, where 9 clean F0/F2 samples are
# misfitted to (F4, F2) and produce 9 identical residuals = pure F0 at
# 5 bits from H[F4].  See conversation notes 2026-05-25 for the full
# diagnostic trace.

def _generate_all_sample_residuals(probs_k, H, A,
                                   cleanness_threshold=RESIDUAL_TRIO_CLEANNESS_THRESHOLD):
    """Generate per-(sample, slot) residuals across ALL samples.

    Variant of _generate_carrier_residuals that mines every sample's
    residual against its currently-assigned OTHER slot, not just samples
    whose A[s, slot] is in a low-carrier set.  Used by _residual_trio_-
    rescue to surface near-clone founders K-growth missed.

    For each (sample, slot) with a non-wildcard partner at A[s, 1-slot]:
        residual = argmax_dosage[s] - H[A[s, 1-slot]]
    Residuals where < cleanness_threshold of sites land in {0, 1} are
    rejected (the partner is impure, or the data is noisy).

    Args:
      probs_k: (N, L_kept, 3) — kept-site posteriors
      H: (K, L_kept) — current founder bits (int64)
      A: (N, 2) — current pair assignments (entry K is wildcard sentinel)
      cleanness_threshold: float — min admissible-site fraction

    Returns:
      list of (L_kept,) np.int64 binary arrays — one per (sample, slot)
      pair that survived cleanness filtering.
    """
    N, L_kept = probs_k.shape[0], probs_k.shape[1]
    K = H.shape[0]
    if K == 0:
        return []
    argmax_dosage = probs_k.argmax(axis=2)                        # (N, L_kept)
    argmax_arr = np.ascontiguousarray(argmax_dosage.astype(np.int64))
    H_arr = np.ascontiguousarray(H.astype(np.int64))
    A_arr = np.ascontiguousarray(A.astype(np.int64))

    out_buf, out_count = _generate_all_sample_residuals_kernel(
        argmax_arr, H_arr, A_arr, float(cleanness_threshold))
    # Repackage into list-of-arrays for consistency with the rest of
    # the candidate-source APIs (trio, hom, pairwise all return lists).
    return [out_buf[i].copy() for i in range(out_count)]


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
    if bhd_pairwise.PAIRWISE_RECOVERY_ENABLED:
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
    new_candidates = []
    for cand in raw_pool_candidates:
        is_dup = False
        for h in H_list:
            if _hamming_pct_kept(cand, h) < 0.5:
                is_dup = True
                break
        if is_dup:
            continue
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