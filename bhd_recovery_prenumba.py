#%% =====================================================================
# bhd_recovery.py — Subtraction-based recovery + late low-carrier rescue
#
# Split out of block_haplotypes_discrete.py as part of the 4-file split.
# Contains the candidate-pool-driven recovery subsystems that run after
# the initial K-growth pass:
#
#   - Recovery constants (RECOVERY_*, RECOVERY_LOW_CARRIER_*)
#   - Bernoulli mixture machinery (_logsumexp, _kmeans_pp_init,
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
# =======================================================================

import math

import numpy as np

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

# Bernoulli mixture parameters for inner K-selection on candidates
RECOVERY_MIXTURE_K_MAX = 10            # try K=1..K_max, pick best by inner BIC
RECOVERY_MIXTURE_N_RESTARTS = 3        # EM restarts per K (different K-means++ seeds)
RECOVERY_MIXTURE_MAX_ITER = 100        # max EM iterations per fit
RECOVERY_MIXTURE_TOL = 1e-6            # relative LL change for EM convergence
RECOVERY_MIXTURE_THETA_EPS = 1e-3      # clip theta to [eps, 1-eps] for log stability
RECOVERY_MIXTURE_RNG_SEED = 42         # base RNG seed (varied per round)

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
RECOVERY_MAX_ROUNDS = 10

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

def _logsumexp(x, axis):
    """Numerically stable log-sum-exp along an axis.  Returns array with
    that axis squeezed.  Handles -inf max (all entries -inf) by zeroing
    the offset (so the result is -inf, not NaN)."""
    m = x.max(axis=axis, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    return (m_safe + np.log(np.exp(x - m_safe).sum(axis=axis, keepdims=True))).squeeze(axis)


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
    """
    N, L = candidates.shape
    if K >= N:
        # Defensive: caller should never let this happen, but if it does,
        # return all candidates plus repeats so the EM has something to work with.
        idx = list(range(N)) + [int(rng.integers(N)) for _ in range(K - N)]
        return candidates[idx].copy()

    centers_idx = [int(rng.integers(N))]
    for _ in range(K - 1):
        existing = candidates[centers_idx]                                # (n_existing, L)
        # Pairwise Hamming distance: (N, n_existing) — number of differing bits
        diffs = (candidates[:, None, :] != existing[None, :, :]).sum(axis=2)
        min_dists = diffs.min(axis=1).astype(np.float64)
        if min_dists.sum() == 0:
            # All candidates identical to some existing center; pick anything not yet picked
            remaining = [i for i in range(N) if i not in centers_idx]
            if remaining:
                new_c = int(rng.choice(remaining))
            else:
                new_c = int(rng.integers(N))
        else:
            probs = min_dists ** 2
            probs = probs / probs.sum()
            new_c = int(rng.choice(N, p=probs))
        centers_idx.append(new_c)
    return candidates[centers_idx].copy()


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
    """
    N, L = cands.shape

    # Smooth initial centers from {0, 1} to (eps, 1-eps) for numerical stability
    theta = init_centers.astype(np.float64) * (1 - 2 * eps) + eps
    pi = np.ones(K, dtype=np.float64) / K

    prev_ll = -np.inf
    n_iter = 0
    ll = 0.0    # initialised in case max_iter=0 (defensive)

    log_resp = None
    resp = None

    for it in range(max_iter):
        n_iter = it + 1
        # E-step
        log_theta = np.log(theta)
        log_one_minus_theta = np.log(1 - theta)
        log_p = cands @ log_theta.T + (1 - cands) @ log_one_minus_theta.T   # (N, K)
        log_pi = np.log(pi + 1e-15)
        log_p_weighted = log_p + log_pi[None, :]                              # (N, K)

        log_norm = _logsumexp(log_p_weighted, axis=1)                          # (N,)
        log_resp = log_p_weighted - log_norm[:, None]                          # (N, K)
        resp = np.exp(log_resp)

        ll = float(log_norm.sum())

        # Convergence check (relative).  Break BEFORE updating prev_ll so
        # the caller receives the most recent ll value.
        if prev_ll != -np.inf and abs(ll - prev_ll) < tol * max(abs(ll), 1.0):
            break
        prev_ll = ll

        # M-step
        N_k = resp.sum(axis=0)                                                 # (K,)
        pi = N_k / N
        N_k_safe = np.maximum(N_k, 1e-10)                                      # avoid /0 for empty components
        theta = (resp.T @ cands) / N_k_safe[:, None]
        theta = np.clip(theta, eps, 1 - eps)

    return theta, pi, ll, n_iter, resp


def _fit_bernoulli_mixture_select_K(candidates,
                                      K_max=RECOVERY_MIXTURE_K_MAX,
                                      n_restarts=RECOVERY_MIXTURE_N_RESTARTS,
                                      seed=RECOVERY_MIXTURE_RNG_SEED,
                                      verbose=False):
    """Fit Bernoulli mixture for K=1..K_max, pick the K minimising BIC.

    Args:
      candidates: list of (L,) binary arrays
      K_max: upper bound on K to try
      n_restarts: EM restarts per K (with different K-means++ seeds)
      seed: base RNG seed (for reproducibility)
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
    """
    raw_candidates = []
    for hap in pool:
        residual = argmax_dosage_kept - hap[None, :]              # (N, L_kept)
        in_01 = (residual >= 0) & (residual <= 1)
        cleanness = in_01.mean(axis=1)                            # (N,)
        clean_mask = cleanness >= cleanness_threshold
        if not clean_mask.any():
            continue
        clean_residuals = residual[clean_mask]                    # (n_clean, L_kept)
        clipped = np.clip(clean_residuals, 0, 1).astype(np.int64)
        for cand in clipped:
            raw_candidates.append(cand)
    return raw_candidates


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
    """
    N, L_kept = probs_k.shape[0], probs_k.shape[1]
    K = H.shape[0]
    if K == 0 or not low_idx_list:
        return ([], []) if verbose else []
    argmax_dosage = probs_k.argmax(axis=2)                        # (N, L_kept)

    low_idx_set = set(int(k) for k in low_idx_list)
    residuals = []
    provenance = [] if verbose else None
    for s in range(N):
        for slot in range(2):
            if int(A[s, slot]) not in low_idx_set:
                continue
            low_idx = int(A[s, slot])
            # Iterate over every H row as a candidate subtractor.
            # Cleanness filter discriminates: wrong-partner subtractors
            # produce out-of-range bits at heterozygous sites and fail.
            for partner_idx in range(K):
                if partner_idx == low_idx:
                    if verbose:
                        provenance.append({
                            'sample_idx': s, 'slot': slot,
                            'low_idx': low_idx, 'partner_idx': partner_idx,
                            'partner_kind': 'self_low',
                            'cleanness': float('nan'), 'accepted': False,
                            'residual': None})
                    continue
                if partner_idx in low_idx_set:
                    # Subtracting another suspect hap would taint
                    # the residual with that suspect's drift.
                    if verbose:
                        provenance.append({
                            'sample_idx': s, 'slot': slot,
                            'low_idx': low_idx, 'partner_idx': partner_idx,
                            'partner_kind': 'low_carrier',
                            'cleanness': float('nan'), 'accepted': False,
                            'residual': None})
                    continue
                residual = argmax_dosage[s] - H[partner_idx]      # (L_kept,)
                in_01 = (residual >= 0) & (residual <= 1)
                cleanness = float(in_01.mean())
                accepted = cleanness >= cleanness_threshold
                clipped = (np.clip(residual, 0, 1).astype(np.int64)
                           if accepted else None)
                if accepted:
                    residuals.append(clipped)
                if verbose:
                    provenance.append({
                        'sample_idx': s, 'slot': slot,
                        'low_idx': low_idx, 'partner_idx': partner_idx,
                        'partner_kind': 'normal',
                        'cleanness': cleanness, 'accepted': accepted,
                        'residual': clipped})
    if verbose:
        return residuals, provenance
    return residuals



def _greedy_bic_select(candidate_haps, probs_k, lam,
                        cc_scale=RECOVERY_OUTER_CC_SCALE,
                        max_k=RECOVERY_MAX_K,
                        use_log_bic=False,
                        verbose=False):
    """Greedy forward selection by BIC over a fixed candidate pool.

    Haps are FIXED during scoring (no coord descent).  At each step,
    pick the candidate giving the lowest NLL when added; accept iff
    BIC improves (equivalent to NLL_improvement > cc/2).

    Args:
      candidate_haps: list of (L_kept,) binary arrays — the pool
      probs_k, lam: scoring primitives
      cc_scale: BIC complexity-cost scale (outer; default 0.5)
      max_k: hard cap on selected size
      use_log_bic: if True, use log-BIC formula for cc (cc_scale *
        log(N*L) * snp_growth); if False (default, project standard),
        use linear formula (cc_scale * snp_growth * N).  Must match
        the use_log_bic of the surrounding K-growth so accept/reject
        criteria are consistent across the pipeline.
      verbose: print accept/reject decisions

    Returns:
      selected_indices: list of indices into candidate_haps
      selected_haps:    list of arrays (same as candidate_haps[i] for i in indices)
      current_nll:      NLL at final selection
    """
    N = probs_k.shape[0]
    L_kept = probs_k.shape[1]
    cc = _compute_cc(cc_scale, N, L_kept, use_log_bic=use_log_bic)

    nll_K0 = _compute_nll_for_subset([], probs_k, lam)
    bic_K0 = 0 * cc + 2 * nll_K0

    selected_indices = []
    selected_haps = []
    used = set()
    current_bic = bic_K0
    current_nll = nll_K0

    if verbose:
        print(f'    Forward: K=0 NLL={nll_K0:.1f}, BIC={bic_K0:.1f}, '
              f'cc={cc:.1f}, threshold cc/2={cc/2:.1f}')

    while len(selected_haps) < min(len(candidate_haps), max_k):
        best_ci = -1
        best_nll = float('inf')
        for ci in range(len(candidate_haps)):
            if ci in used:
                continue
            trial_haps = selected_haps + [candidate_haps[ci]]
            trial_nll = _compute_nll_for_subset(trial_haps, probs_k, lam)
            if trial_nll < best_nll:
                best_nll = trial_nll
                best_ci = ci

        if best_ci < 0:
            break

        k_new = len(selected_haps) + 1
        bic_new = k_new * cc + 2 * best_nll
        d_nll = current_nll - best_nll

        if bic_new < current_bic:
            selected_indices.append(best_ci)
            selected_haps.append(candidate_haps[best_ci])
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

    return selected_indices, selected_haps, current_nll


def _swap_refine(selected_indices, selected_haps, pool_haps,
                  probs_k, lam, current_nll,
                  nll_tolerance=RECOVERY_SWAP_NLL_TOLERANCE,
                  max_passes=10, verbose=False):
    """Try swapping each selected hap with each unselected pool member.
    Apply swap if NLL improves by more than nll_tolerance.  Iterate over
    passes until no improvement.

    Sometimes greedy forward selection picks a near-optimal hap early
    that becomes redundant after later picks; swap lets us replace it
    with a better one without requiring a full re-search.
    """
    sel_ind = list(selected_indices)
    sel_haps = list(selected_haps)
    K = len(sel_haps)
    if K == 0:
        return sel_ind, sel_haps, current_nll, 0

    n_swaps = 0
    for pass_num in range(max_passes):
        improved_in_pass = False
        for si in range(K):
            best_ci = -1
            best_nll = current_nll - nll_tolerance
            for ci in range(len(pool_haps)):
                if ci in sel_ind:
                    continue
                trial_haps = list(sel_haps)
                trial_haps[si] = pool_haps[ci]
                trial_nll = _compute_nll_for_subset(trial_haps, probs_k, lam)
                if trial_nll < best_nll:
                    best_nll = trial_nll
                    best_ci = ci
            if best_ci >= 0:
                if verbose:
                    print(f'    Swap: pos {si} (cand[{sel_ind[si]}]) -> cand[{best_ci}], '
                          f'NLL {current_nll:.1f} -> {best_nll:.1f}')
                sel_haps[si] = pool_haps[best_ci]
                sel_ind[si] = best_ci
                current_nll = best_nll
                improved_in_pass = True
                n_swaps += 1
                break
        if not improved_in_pass:
            break

    return sel_ind, sel_haps, current_nll, n_swaps


def _bic_prune(selected_indices, selected_haps, probs_k, lam,
                cc_scale=RECOVERY_OUTER_CC_SCALE, use_log_bic=False,
                verbose=False):
    """BIC pruning: try dropping each selected hap.  Drop if the NLL
    increase from removal is less than cc/2 (i.e., the hap isn't pulling
    enough weight to justify the +cc penalty for keeping it).

    Iterates: each drop may enable another (cascading prune of redundant
    haps that propped each other up).  Matches the project's
    refine_selection_by_pruning pattern in beam_search_core.

    use_log_bic: bool — if True, use log-BIC formula for cc; if False
      (default, project standard), use linear formula.  Must match the
      use_log_bic of the surrounding K-growth so the prune threshold
      cc/2 is consistent with K-growth's growth threshold.

    Returns: (pruned_indices, pruned_haps, final_nll, n_dropped)
    """
    N = probs_k.shape[0]
    L_kept = probs_k.shape[1]
    cc = _compute_cc(cc_scale, N, L_kept, use_log_bic=use_log_bic)

    sel_ind = list(selected_indices)
    sel_haps = list(selected_haps)
    n_dropped = 0

    while len(sel_haps) > 0:
        nll_full = _compute_nll_for_subset(sel_haps, probs_k, lam)
        K = len(sel_haps)

        best_drop_idx = -1
        best_dnll = cc / 2   # threshold; only drop if dnll_increase < this

        for i in range(K):
            trial = sel_haps[:i] + sel_haps[i+1:]
            nll_trial = _compute_nll_for_subset(trial, probs_k, lam)
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
        del sel_haps[best_drop_idx]
        n_dropped += 1

    final_nll = _compute_nll_for_subset(sel_haps, probs_k, lam)
    return sel_ind, sel_haps, final_nll, n_dropped



def _hamming_pct_kept(a, b):
    """Per-site Hamming distance as a percentage (0-100)."""
    return float(np.mean(a != b)) * 100.0


def _haps_equal(haps_a, haps_b, eps_pct=RECOVERY_HAPS_EQUAL_EPS_PCT):
    """Test if two hap collections are equal (within eps_pct Hamming
    tolerance per pair, with bipartite matching).

    Returns True iff:
      - len(haps_a) == len(haps_b), AND
      - there's a 1-to-1 matching where each matched pair is within eps_pct.

    Used for convergence detection between recovery rounds and between
    outer iterations.  Tolerance accommodates near-identical haps that
    differ only at a few uncertain sites.
    """
    if len(haps_a) != len(haps_b):
        return False
    matched_b = [False] * len(haps_b)
    for ha in haps_a:
        found = False
        for bi, hb in enumerate(haps_b):
            if matched_b[bi]:
                continue
            if _hamming_pct_kept(ha, hb) < eps_pct:
                matched_b[bi] = True
                found = True
                break
        if not found:
            return False
    return all(matched_b)


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
                                       cleanness_threshold=RECOVERY_CLEANNESS_THRESHOLD,
                                       swap_nll_tolerance=RECOVERY_SWAP_NLL_TOLERANCE,
                                       haps_equal_eps_pct=RECOVERY_HAPS_EQUAL_EPS_PCT,
                                       use_log_bic=False,
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
      cleanness_threshold: min residual cleanness to admit a candidate
      swap_nll_tolerance: NLL improvement floor for accepting a swap
      haps_equal_eps_pct: tolerance for round-convergence detection
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

    for round_num in range(1, max_rounds + 1):
        # 1. Subtraction: generate clean residual candidates
        raw_candidates = _run_subtraction_round(
            selected, argmax_dosage_kept,
            cleanness_threshold=cleanness_threshold)
        if len(raw_candidates) == 0:
            if verbose:
                print(f'  [recovery round {round_num}] no clean residuals -- CONVERGED')
            break

        # 2. Bernoulli mixture fit + BIC over K -> consensus haps
        if verbose:
            print(f'  [recovery round {round_num}] {len(raw_candidates)} raw candidates')
        consensus_haps = _fit_bernoulli_mixture_select_K(
            raw_candidates,
            K_max=mixture_K_max,
            n_restarts=mixture_n_restarts,
            seed=mixture_seed_base + round_num,
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

        # 5. Greedy BIC forward selection (haps frozen)
        sel_indices, sel_haps, sel_nll = _greedy_bic_select(
            pool, probs_k, lam,
            cc_scale=outer_cc_scale, max_k=max_K,
            use_log_bic=use_log_bic, verbose=verbose)

        # 6. Swap refinement (haps still frozen)
        if len(sel_haps) > 0:
            sel_indices, sel_haps, sel_nll, n_swaps = _swap_refine(
                sel_indices, sel_haps, pool, probs_k, lam,
                current_nll=sel_nll,
                nll_tolerance=swap_nll_tolerance,
                verbose=verbose)

        # 7. BIC pruning (haps still frozen)
        if len(sel_haps) > 0:
            sel_indices, sel_haps, sel_nll, n_dropped = _bic_prune(
                sel_indices, sel_haps, probs_k, lam,
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
         (TRIO_D_ESTIMATE_SEED=42) and cheap (~100ms at N=320).
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

    # Compute per-hap real-strand carrier counts (excluding wildcards)
    W = K
    usage = np.zeros(K, dtype=np.int64)
    for s in range(N):
        for slot in range(2):
            f = int(A[s, slot])
            if f != W:
                usage[f] += 1

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
    # (TRIO_D_ESTIMATE_SEED=42) and cheap (~100ms at N=320).
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

    # Compute current BIC for comparison
    cc = _compute_cc(cc_scale, N, L_kept, use_log_bic=use_log_bic)
    BIC_orig = _compute_bic(K, NLL, cc)

    # Greedy BIC forward selection on enriched pool.  max_k = K + 1
    # allows BIC to grow K by 1, OR shrink K by stopping early when a
    # smaller subset has lower BIC.
    sel_indices, sel_haps, sel_nll = _greedy_bic_select(
        pool, probs_k, lam,
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
        sel_indices, sel_haps, pool, probs_k, lam, sel_nll,
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
        sel_indices, sel_haps, probs_k, lam,
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