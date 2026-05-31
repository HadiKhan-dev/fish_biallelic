#%% =====================================================================
# bhd_kernels.py — Atomic computation kernel for stage-3 block haplotypes
#
# Split out of block_haplotypes_discrete.py as part of the 4-file split.
# Contains the BIC/CD primitives that every higher-level subsystem builds
# on:
#
#   - Module-level constants (MASK, DEFAULT_LAMBDA, LOG_EPS) and Viterbi
#     BIC scoring configuration (_VITERBI_BIC_ENABLED, VITERBI_SWITCH_-
#     PENALTY, VITERBI_SNPS_PER_BIN).
#   - Low-level helpers (_safe_neg_log, _decisiveness).
#   - Initialization helpers (_init_hap_from_sample_dosage,
#     _select_initial_seed).
#   - Cost-tensor computation (_per_site_cost_real_real, _per_site_cost_-
#     real_W, _per_site_cost_W_W).
#   - Viterbi log-likelihood kernel (_viterbi_ll_per_sample, _viterbi_nll).
#   - Pair-assignment / founder-update steps (_update_A, _update_H,
#     _update_one_founder).
#   - Coordinate descent at fixed K (_fit_at_fixed_K).
#   - BIC helpers (_compute_cc, _compute_bic) — the formula
#         BIC(K) = K * cc + 2 * NLL_K          (lower is better)
#     is centralised here.
#   - The unified BIC scorer (_compute_nll_for_subset) — dispatches on
#     _VITERBI_BIC_ENABLED to either Viterbi or best-pair NLL.
#
# Dependencies: only numpy + math + block_haplotypes.viterbi_score_-
# selection.  No cycles with bhd_recovery, bhd_trio, or bhd_pairwise.
# =======================================================================

import math
import warnings

import numpy as np
from numba import njit, prange

from block_haplotypes import viterbi_score_selection

warnings.filterwarnings("ignore")
np.seterr(divide='ignore', invalid='ignore')


# =============================================================================
# CONSTANTS
# =============================================================================

# Sentinel for "this site of this founder is unconstrained / no support".
# Used at output-time only; during inference H is strictly binary {0, 1}.
MASK = -1

# Wildcard sentinel for pair assignments A[s, *] = W means strand-is-wildcard.
# We use K (one past the last real founder index) as the wildcard slot since
# K varies during growth.  The W sentinel is computed as the current K at
# each call site that needs it.

# Default wildcard penalty.  λ in log-likelihood units per (strand, site)
# wildcard usage.  Sites where the real founder pair gives a likelihood at
# least 1/e^(2λ) ≈ 0.37 of the wildcard's optimal genotype likelihood
# prefer real founders; below that, wildcards take over.  λ=0.5 puts the
# crossover at "real wins until likelihood ratio of (best wildcard /
# real-pair) exceeds e^1 ≈ 2.7."
DEFAULT_LAMBDA = 0.5

# Numerical floor for log(probability) — prevents -inf when a sample's
# posterior is exactly zero at a particular genotype (which can happen
# at sites with no reads).
LOG_EPS = 1e-12


# =============================================================================
# VITERBI BIC SCORING — replaces best-pair-per-sample cost calculation
# =============================================================================
#
# When _VITERBI_BIC_ENABLED is True (the production default), the per-sample
# cost calculation inside _update_A is replaced with a Viterbi best-path
# log-likelihood over a state space of all founder-pair states (including
# wildcard pairs), allowing inter-bin switches between pair states at a flat
# cost VITERBI_SWITCH_PENALTY per switch.  This lets the BIC criterion
# correctly identify chimeric founders: a chimera that fits some samples
# better than any single existing-founder pair will, under Viterbi, be
# rejected if those samples can be modelled by switching between two
# existing-founder pairs at a within-block recombination point.
#
# Validated end-to-end on 250 K-mixed blocks (test_viterbi_full_pipeline_*):
#   - V5 (penalty=5): 88% spurious reduction vs baseline, 0/1354 captures
#     lost across K strata 1 through 10
#   - V10 (penalty=10): 79% spurious reduction, 0/1354 captures lost
# V5 strictly Pareto-dominates V10 (same captures, fewer spurious).
#
# Architecture: when enabled, _compute_nll_for_subset returns Viterbi NLL
# directly (bypassing _update_A entirely for scoring); _update_A still
# computes the baseline best-pair-per-sample assignment A (used by
# _update_H's bit-voting M-step) and wildcard_slots (used for wildcard
# mass tracking), but the returned per_sample_cost and per_sample_cost_unc
# arrays are replaced with Viterbi per-sample NLL.  This means every BIC
# accept/reject decision downstream of _fit_at_fixed_K — K-growth's NLL
# improvement signal at lines 1255 and 1455, the medoid-branch BIC
# comparison at 2581/2602, the late-rescue BIC_new at 3341 — sees
# Viterbi-based scoring.  Coord descent on H (the M-step bit voting in
# _update_H) still operates under baseline pair assignments because
# rewriting bit voting to be Viterbi-path-aware would be a much larger
# change; the compromise was validated to give the gains reported above
# without rewriting _update_H.
#
# To revert to baseline best-pair-per-sample scoring for A/B comparison,
# set _VITERBI_BIC_ENABLED = False.  Setting the flag is process-global,
# so it should be set before _grow_K_with_recovery is called and not
# changed during a run; with multiprocessing.Pool workers, each worker
# has its own module state and can have an independent setting if needed.
_VITERBI_BIC_ENABLED = True

# Switch penalty for the Viterbi BIC kernel.  At lower values, switches
# between pair states are cheaper => more aggressive chimera rejection.
# At higher values, switches approach prohibitive cost => behaviour
# approaches baseline best-pair-per-sample.  Validated sweep on K>6
# blocks gave a Pareto frontier: 0 loses captures; 5 is the production
# sweet spot (88% spurious reduction, 0 captures lost); 10-30 progressively
# fewer chimeras rejected; >= 50 collapses to baseline-equivalent.
# 5 is the default; tune by editing this constant or by setting it from
# a caller before invoking _grow_K_with_recovery.
#
# Update (2026-05): raised default from 5.0 to 10.0 to address within-
# block-recombination "switch-trap" failures.  At chr10:503 (and the
# chr3 F0 cluster, chr6 F4 cluster, and a handful of other blocks),
# K-growth correctly identifies all distinct haplotype patterns
# including a chimera, then the residual-trio rescue (added 2026-05,
# see bhd_recovery._residual_trio_rescue) surfaces the missing pure
# founder as a candidate — but BIC rejects K=5 because the clean
# carriers of the missing founder can already fit their data by
# Viterbi-switching between the chimera (in pre-breakpoint region)
# and the near-clone founder (in post-breakpoint region) at the cost
# of just 5 nats per sample.  This 5-nat switch cost is below the
# BIC-acceptance threshold (cc/2 = 80 nats vs the typical 70-nat LL
# improvement from 14 clean carriers), so K=5 is wrongly rejected and
# the truth founder is missed.  Doubling to 10.0 makes those switches
# cost 10 nats × 14 carriers = 140 nats, comfortably above cc/2 = 80,
# so K=5 is accepted and the truth founder recovered.
#
# Trade-off: the original V5 vs V10 sweep showed V5 had 9pp more
# spurious-K reduction.  But residual-trio's strict filtering
# (cleanness=1.0, min_cluster_size=3, dedup_vs_h=1.0%) makes spurious
# K-additions much less likely at V10 than they were in the original
# sweep (which preceded residual-trio).  Empirical A/B test on
# full stage 3 + downstream pipeline is the right way to validate.
VITERBI_SWITCH_PENALTY = 10.0

# Bin granularity for Viterbi: each bin sums log-prob emissions within the
# bin before applying the inter-bin switch penalty.  At spb=10 (the
# default), a 200-SNP block has 20 bins and Viterbi can switch pair states
# at most 19 times.  Matches chimera_resolution.py's L1 anchor
# (compute_spb() = max(10, avg_sites//20) gives 10 for L=200 blocks).
# Lower spb => more switching points (more granular chimera handling but
# more compute); higher spb => fewer switch points (coarser, faster).
VITERBI_SNPS_PER_BIN = 10


# =============================================================================
# LOW-LEVEL HELPERS
# =============================================================================

def _safe_neg_log(p):
    """Element-wise -log(max(p, LOG_EPS)).  Vectorised, never returns inf."""
    return -np.log(np.maximum(p, LOG_EPS))


def _decisiveness(probs):
    """Per-sample decisiveness score: sum of per-site argmax probabilities.

    A sample with crisp posteriors (each site's argmax-prob near 1.0) has
    high decisiveness and is a good initial-founder candidate.  A sample
    with diffuse posteriors (each site's argmax-prob near 1/3) has low
    decisiveness.

    Argument:
        probs: (N, L, 3) genotype posteriors

    Returns:
        (N,) array of decisiveness scores
    """
    return probs.max(axis=2).sum(axis=1)


# =============================================================================
# INITIALIZATION
# =============================================================================

def _init_hap_from_sample_dosage(probs, sample_idx, kept_mask):
    """Build a binary founder hap from one sample's argmax dosages.

    The seed sample's per-site genotype dosage is interpreted as the sum
    of two homozygous-equivalent strands of one founder.  At dosage=0
    (seed homo-ref) the founder bit MUST be 0; at dosage=2 (seed
    homo-alt) the founder bit MUST be 1.  At dosage=1 (seed het) either
    value is consistent with the seed alone — the founder could be 0
    (with the other strand being 1) or 1 (with the other strand being 0).

    HISTORICAL NOTE — old behaviour and the bug it caused:
        Originally this function rounded dosage // 2, which at dosage=1
        deterministically picked 0.  At read depth 5x the M-step's
        carrier-pool votes were noisy enough that wrong-polarity bits
        from this floor-div could be flipped during coordinate descent.
        At read depth 20x, votes are highly confident and CD locks in
        the seed's wrong polarity at dosage=1 sites.  Diagnostic on
        chr1:14043389 (a 0/6-found block) showed 100% of the wrong-
        polarity sites in the final K-grown output were exactly the
        sites where the K=1 seed was heterozygous, i.e. the floor-div's
        arbitrary-zero choice.  All later founders inherited the same
        wrong-polarity convention via worst-fit-sample subtraction-
        seeding.  This was the dominant failure mode at high depth.

    NEW behaviour: at dosage=1 sites we break the tie using POPULATION
    allele frequency at that site, computed from `probs` as the
    expected per-site alt allele rate:
        alt_freq[l] = mean over samples of (P(g=01) * 0.5 + P(g=11))
    If alt_freq[l] > 0.5 we set the seed bit to 1 (alt is majority);
    otherwise to 0.  At dosage 0 / 2 sites the seed itself is
    unambiguous and we use it directly (population frequency is not
    consulted, since the data forces a value).

    Why this fixes the lock-in: the K=1 seed now starts with polarity
    that is correct on average across the population, rather than
    polarity that is correct only when the seed sample's true other
    strand happens to be 0.  CD's confident votes then act on a seed
    that's already in the right polarity ballpark, so the wrong-
    polarity local optimum is avoided.

    Arguments:
        probs: (N, L, 3) genotype posteriors
        sample_idx: int — which sample to use as the seed
        kept_mask: (L,) bool — which sites are scored (unkept sites get
            value 0 by convention; their value won't affect any sample's
            cost since no sample's pair likelihood is summed over them)

    Returns:
        h: (L,) int array of {0, 1} alleles
    """
    L = probs.shape[1]
    dosage = probs[sample_idx].argmax(axis=1)   # (L,) ∈ {0, 1, 2}

    # Population alt-allele frequency per site.  Posterior expected
    # P(allele=1) per site = sum over samples of (0.5 * P(g=01) + P(g=11))
    # divided by sample count.  Range: [0, 1].
    pop_alt_freq = probs[..., 1].mean(axis=0) * 0.5 + probs[..., 2].mean(axis=0)

    # Default: dosage 0 -> h=0, dosage 2 -> h=1 (forced by data).
    # At dosage 1: break the tie using population frequency.  Tied at
    # exactly 0.5 we keep the legacy convention (round to 0) — extreme
    # edge case, doesn't affect the failure mode being fixed.
    h = np.zeros(L, dtype=np.int64)
    h[dosage == 2] = 1
    het_mask = (dosage == 1)
    h[het_mask & (pop_alt_freq > 0.5)] = 1

    # Unkept sites: value doesn't matter, but set to 0 for cleanliness
    if kept_mask is not None:
        h = np.where(kept_mask, h, 0)
    return h


def _select_initial_seed(probs, kept_mask):
    """Pick the most-decisive sample to seed the K=1 founder.

    Argument:
        probs: (N, L, 3) — restricted to kept sites for fair scoring
        kept_mask: (L,) bool

    Returns:
        sample_idx: int
    """
    if kept_mask is not None:
        probs_kept = probs[:, kept_mask, :]
    else:
        probs_kept = probs
    decisiveness = _decisiveness(probs_kept)
    return int(decisiveness.argmax())


# =============================================================================
# COST TENSOR COMPUTATION
# =============================================================================
# All cost tensors are computed over kept sites only.  The "cost" of a pair
# for a sample is the negative log-likelihood plus wildcard penalties.

def _per_site_cost_real_real(probs_k, H_k):
    """For all real-real pairs (i, j) with i ≤ j, compute per-(sample, site)
    cost.

    Arguments:
        probs_k: (N, L_kept, 3) — kept-site genotype posteriors
        H_k:     (K, L_kept) — kept-site discrete founder haps in {0, 1}

    Returns:
        cost: (N, n_pairs_real, L_kept) where n_pairs_real = K(K+1)/2.
              cost[s, p, l] = -log probs_k[s, l, H_k[i, l] + H_k[j, l]]
              for the p-th pair (i, j).
        pair_indices: list of (i, j) tuples in p-order.
    """
    K, L = H_k.shape
    pair_indices = [(i, j) for i in range(K) for j in range(i, K)]
    n_pairs = len(pair_indices)
    N = probs_k.shape[0]
    if n_pairs == 0:
        return np.empty((N, 0, L)), []

    # Build (n_pairs, 2) int array of pair (i, j) endpoints so the kernel
    # can index into H_k without a Python list.  pair_indices stays as
    # the returned tuple list for downstream consumers (e.g. _update_A's
    # translate-back loop, which expects Python-tuple indexing).
    pair_idx_arr = np.empty((n_pairs, 2), dtype=np.int64)
    for p, (i, j) in enumerate(pair_indices):
        pair_idx_arr[p, 0] = i
        pair_idx_arr[p, 1] = j

    probs_c = np.ascontiguousarray(probs_k, dtype=np.float64)
    H_c = np.ascontiguousarray(H_k, dtype=np.int64)
    cost = _per_site_cost_real_real_kernel(probs_c, H_c, pair_idx_arr)
    return cost, pair_indices


@njit(cache=True, parallel=True, fastmath=False)
def _per_site_cost_real_real_kernel(probs_k, H_k, pair_idx_arr):
    """njit version of the real-real cost tensor build.

    Avoids the (N, n_pairs, L, 1) broadcast / take_along_axis that the
    numpy version had to use, by writing the inner triple loop directly.
    The site axis is the innermost loop (best cache locality on a
    C-contig probs_k that is (N, L, 3)).

    Returns:
        cost: (N, n_pairs, L) float64

    Note: -log floor is LOG_EPS, identical to _safe_neg_log's behaviour.
    LOG_EPS appears as a literal here because module-level constants are
    not importable inside @njit functions; the value 1e-12 must be kept
    in sync with LOG_EPS in the module body.
    """
    LOG_EPS_LOCAL = 1e-12

    N = probs_k.shape[0]
    L = probs_k.shape[1]
    n_pairs = pair_idx_arr.shape[0]
    cost = np.empty((N, n_pairs, L), dtype=np.float64)

    # prange over samples — N is typically 320 and each sample's work is
    # independent.  Inner loops are sequential per sample.
    for s in prange(N):
        for p in range(n_pairs):
            i = pair_idx_arr[p, 0]
            j = pair_idx_arr[p, 1]
            for l in range(L):
                d = H_k[i, l] + H_k[j, l]
                pv = probs_k[s, l, d]
                if pv < LOG_EPS_LOCAL:
                    pv = LOG_EPS_LOCAL
                cost[s, p, l] = -math.log(pv)
    return cost


def _per_site_cost_real_W(probs_k, H_k, lam):
    """For each (real founder, wildcard) pair, compute per-(sample, site) cost.

    Arguments:
        probs_k: (N, L_kept, 3)
        H_k:     (K, L_kept)
        lam:     wildcard penalty per strand-site usage

    Returns:
        cost: (N, K, L_kept) where cost[s, k, l] is the cost of pair (k, W)
              for sample s at site l, with the wildcard strand picking
              its allele optimally per site.
    """
    probs_c = np.ascontiguousarray(probs_k, dtype=np.float64)
    H_c = np.ascontiguousarray(H_k, dtype=np.int64)
    return _per_site_cost_real_W_kernel(probs_c, H_c, float(lam))


@njit(cache=True, parallel=True, fastmath=False)
def _per_site_cost_real_W_kernel(probs_k, H_k, lam):
    """njit version of the real-W cost tensor build.

    For each (real founder k, wildcard) pair the wildcard strand picks
    its allele w in {0, 1} to maximise probs_k[s, l, H_k[k, l] + w].
    Direct max over two candidates instead of the
    take_along_axis-on-two-broadcast-tensors dance the numpy version
    needed.
    """
    LOG_EPS_LOCAL = 1e-12

    N = probs_k.shape[0]
    L = probs_k.shape[1]
    K = H_k.shape[0]
    cost = np.empty((N, K, L), dtype=np.float64)

    for s in prange(N):
        for k in range(K):
            for l in range(L):
                d0 = H_k[k, l]          # w = 0
                d1 = d0 + 1              # w = 1
                p0 = probs_k[s, l, d0]
                p1 = probs_k[s, l, d1]
                pmax = p0 if p0 > p1 else p1
                if pmax < LOG_EPS_LOCAL:
                    pmax = LOG_EPS_LOCAL
                cost[s, k, l] = -math.log(pmax) + lam
    return cost


def _per_site_cost_W_W(probs_k, lam):
    """Per-(sample, site) cost of the (W, W) pair: each strand's wildcard
    picks its allele independently to maximise the genotype likelihood,
    paying 2λ per site.

    Arguments:
        probs_k: (N, L_kept, 3)
        lam:     wildcard penalty per strand-site usage

    Returns:
        cost: (N, L_kept)
    """
    probs_c = np.ascontiguousarray(probs_k, dtype=np.float64)
    return _per_site_cost_W_W_kernel(probs_c, float(lam))


@njit(cache=True, parallel=True, fastmath=False)
def _per_site_cost_W_W_kernel(probs_k, lam):
    """njit version: max over the 3 genotype probabilities per (sample,
    site), -log floor at LOG_EPS_LOCAL, plus 2*lam per site."""
    LOG_EPS_LOCAL = 1e-12

    N = probs_k.shape[0]
    L = probs_k.shape[1]
    cost = np.empty((N, L), dtype=np.float64)

    for s in prange(N):
        for l in range(L):
            p0 = probs_k[s, l, 0]
            p1 = probs_k[s, l, 1]
            p2 = probs_k[s, l, 2]
            pmax = p0
            if p1 > pmax:
                pmax = p1
            if p2 > pmax:
                pmax = p2
            if pmax < LOG_EPS_LOCAL:
                pmax = LOG_EPS_LOCAL
            cost[s, l] = -math.log(pmax) + 2.0 * lam
    return cost


@njit(cache=True, parallel=True, fastmath=False)
def _ww_bin_emis_from_cost_ww(cost_WW, snps_per_bin, n_bins):
    """Precompute the WW state's binned Viterbi emissions from a precomputed
    per-(sample, site) WW cost array.

    Derivation:
        cost_WW[s, l] = -log p_max(s, l) + 2 * lam   (with LOG_EPS clamp)
        WW per-site LL contribution = log p_max(s, l) - 2 * lam = -cost_WW[s, l]
        WW_bin_emis[s, b] = sum_{l in bin_b} (log p_max - 2*lam)
                          = -sum_{l in bin_b} cost_WW[s, l]

    Per-bin summation is left-to-right (l = start..end-1), matching
    `_viterbi_binned_emissions_kernel`'s WW path bit-for-bit.

    Used by `_fit_at_fixed_K` to amortise WW work across CD iterations:
    cost_WW depends only on (probs_k, lam) and so doesn't change as H
    is updated.  The precomputed WW_bin_emis lets the fused kernel skip
    its WW state's inner site-loop entirely (one (N, n_bins) write per
    call instead of N * L log-probability evaluations per call).
    """
    N = cost_WW.shape[0]
    L = cost_WW.shape[1]
    out = np.empty((N, n_bins), dtype=np.float64)
    for s in prange(N):
        for b in range(n_bins):
            start = b * snps_per_bin
            end = start + snps_per_bin
            if end > L:
                end = L
            acc = 0.0
            for l in range(start, end):
                acc -= cost_WW[s, l]   # = log p_max - 2*lam
            out[s, b] = acc
    return out


@njit(cache=True, parallel=True, fastmath=False)
def _log_probs_kernel(probs_k):
    """Precompute log(max(probs_k[s, l, g], LOG_EPS_LOCAL)) for every
    (sample, site, genotype) cell.

    This is the heart of Tier 0: probs_k is INVARIANT across CD
    iterations within a single `_fit_at_fixed_K` invocation, so every
    `-log(probs[s, l, d])` value the `_update_A_fused_kernel` and
    `_update_one_founder_kernel` evaluate inside their (s, state, l)
    or (l, bucket, sample) walks is itself invariant.  Computing all
    3 * N * L log values ONCE per fit replaces N * L * 3 * (~28) log
    calls per `_update_A` and several more per `_update_H`.

    Bit-equivalence with the original inline pattern:

        Old inline (per (s, l, d) visit):
            pv = probs_k[s, l, d]
            if pv < LOG_EPS_LOCAL: pv = LOG_EPS_LOCAL
            lp = math.log(pv)

        Precomputed:
            lp = log_probs[s, l, d]
            where log_probs[s, l, d] = math.log(max(probs_k[s, l, d], LOG_EPS_LOCAL))

    Identical scalars: same probs entry, same EPS clamp, same math.log
    call, just amortised across all the kernels that need it.

    Bit-equivalence for max-of-probs callsites (e.g. WW state, kW
    state, bucket-P max-pick) relies on log being strictly monotonic
    on positives, so

        log(max(max(p0, p1), EPS)) = max(log(max(p0, EPS)), log(max(p1, EPS)))

    when all pi are non-negative (which they are, being probabilities).
    Therefore replacing "max raw probs then log-with-clamp" with "max
    of pre-clamped log_probs" yields the same scalar.  The comparison
    used to select the max is identical because log is monotonic
    (p0 > p1 iff log_probs[..., 0] > log_probs[..., 1] when both are
    >= EPS-clamped).

    Inputs:
        probs_k: (N, L, 3) float64, C-contig.  Per-(sample, site)
                 genotype posteriors.

    Returns:
        log_probs: (N, L, 3) float64 — log(max(probs[s, l, g], EPS))
                   per cell.
    """
    LOG_EPS_LOCAL = 1e-12
    N = probs_k.shape[0]
    L = probs_k.shape[1]
    out = np.empty((N, L, 3), dtype=np.float64)
    for s in prange(N):
        for l in range(L):
            for g in range(3):
                pv = probs_k[s, l, g]
                if pv < LOG_EPS_LOCAL:
                    pv = LOG_EPS_LOCAL
                out[s, l, g] = math.log(pv)
    return out


def _maybe_c_contig(arr, dtype):
    """Return a C-contiguous array of the given dtype, without copying when
    the input already satisfies both conditions.

    `np.ascontiguousarray` always allocates a fresh buffer even when the
    input is already C-contiguous with the right dtype; over hundreds of
    thousands of hot-path calls (e.g. inside `_update_one_founder` and
    `_update_A`), the redundant allocations show up in profiles.  This
    helper inspects `.flags.c_contiguous` and `.dtype` first and only
    falls back to a copy when something genuinely needs converting.

    Restricted to numpy ndarrays — every caller in this module passes
    ndarrays.  Don't generalise to "anything array-like" because the
    branch logic relies on `.flags`.
    """
    if arr.flags.c_contiguous and arr.dtype == dtype:
        return arr
    return np.ascontiguousarray(arr, dtype=dtype)


# =============================================================================
# VITERBI LOG-LIKELIHOOD KERNEL
# =============================================================================
#
# Per-sample best-path log-likelihood over founder-pair states with optional
# inter-bin switching at a flat penalty cost.  Used by _update_A and
# _compute_nll_for_subset when _VITERBI_BIC_ENABLED is True (default) to
# replace the best-pair-per-sample BIC scoring.  See the VITERBI BIC
# SCORING constants block near the top of the file for rationale and
# validated parameter values.
#
# State space (with include_wildcards=True, the default):
#   - K^2 ordered real-real pair states (i, j) with i, j in [0, K)
#   - 2K ordered real-W states: K of (k, W) and K of (W, k) for k in [0, K).
#     Under flat-penalty Viterbi these are mathematically equivalent to K
#     unordered real-W states because emissions are commutative in strand
#     identity (dosage = H[k]+w == w+H[k]); we keep them ordered for
#     enumeration consistency with the K^2 real-real states and to future-
#     proof against distance-aware transition models where strand identity
#     matters for phase tracking.
#   - 1 (W, W) state.
#   Total = K^2 + 2K + 1 states.
#
# K=0 special case: only state is (W, W); no switching possible.  Per-
# sample LL = sum_l (log max_g probs_k[s, l, g] - 2*lam).  Computed inline
# so this function does NOT depend on _compute_nll_for_subset (and so the
# two functions can call each other without infinite recursion).

def _viterbi_ll_per_sample(haps_list, probs_k,
                              penalty=None, snps_per_bin=None, lam=None,
                              include_wildcards=True):
    """Return (N,) array of best Viterbi-path log-likelihood per sample.

    All parameters default to module-level constants when None:
        penalty       -> VITERBI_SWITCH_PENALTY (default 5.0)
        snps_per_bin  -> VITERBI_SNPS_PER_BIN  (default 10)
        lam           -> DEFAULT_LAMBDA         (default 0.5)
    """
    if penalty is None:
        penalty = VITERBI_SWITCH_PENALTY
    if snps_per_bin is None:
        snps_per_bin = VITERBI_SNPS_PER_BIN
    if lam is None:
        lam = DEFAULT_LAMBDA

    if not haps_list:
        # K=0 inline path: only state is (W, W), no transitions to consider.
        # Per-sample LL = sum_l (log max_g P_s(l,g) - 2*lam).  Computed
        # directly without invoking viterbi_score_selection so the K=0
        # branch is self-contained and cannot recurse through callers.
        N_, L_, _ = probs_k.shape
        p_max_g = np.maximum(probs_k.max(axis=2).astype(np.float64), LOG_EPS)
        ll_per_sample = (np.log(p_max_g) - 2.0 * lam).sum(axis=1)
        return ll_per_sample

    H = np.stack(haps_list, axis=0).astype(np.int64)        # (K, L)
    K, L = H.shape
    N = probs_k.shape[0]

    # Build the BINNED log-likelihood tensor in a single fused njit pass.
    # The old numpy version built three intermediates -- rr_emis (N, K*K, L),
    # rw_emis (N, 2K, L), ww_emis (N, 1, L) -- via broadcast + take_along_
    # axis + concatenate, then summed over snps_per_bin-sized chunks.  Each
    # of those steps allocated a fresh (N, K_states, L) tensor; the
    # concatenate further copied them into one contiguous (N, K_states, L)
    # buffer before the bin-sum reduced it to (N, K_states, n_bins).
    # In the njit version we go straight from (probs_k, H) to (N, K_states,
    # n_bins) with no per-SNP intermediate tensor.  Saves N*K_states*L*8
    # bytes of allocation (e.g. for N=320, K=6, L=200, K_states=49: ~25 MB
    # per call, hit on every coord-descent iteration).
    probs_c = np.ascontiguousarray(probs_k, dtype=np.float64)
    H_c = np.ascontiguousarray(H, dtype=np.int64)
    if snps_per_bin > 1 and snps_per_bin < L:
        n_bins = int(math.ceil(L / snps_per_bin))
    else:
        n_bins = L
        snps_per_bin = 1
    ll_tensor = _viterbi_binned_emissions_kernel(
        probs_c, H_c, int(snps_per_bin), int(n_bins),
        float(lam), bool(include_wildcards))

    # viterbi_score_selection: flat-penalty Viterbi, @njit(parallel=True)
    # over samples (block_haplotypes line 378).  Returns (N,) of best path
    # log-likelihood.
    best_ll = viterbi_score_selection(ll_tensor, float(penalty))
    return best_ll


@njit(cache=True, parallel=True, fastmath=False)
def _viterbi_binned_emissions_kernel(probs_k, H, snps_per_bin, n_bins,
                                      lam, include_wildcards):
    """Build the binned log-likelihood tensor for the Viterbi BIC kernel.

    Emission semantics match the original numpy version exactly with
    one COLLAPSED state-space optimisation: twin states (states with
    identical emissions at every bin) are deduplicated.  Under flat-
    penalty Viterbi the alpha values for twin states are identical
    throughout the forward pass (proof by induction on bin index),
    so removing one twin does not change the final per-sample best-
    path LL.  See the "VITERBI STATE-SPACE COLLAPSE" comment block
    in the module docstring header for the proof.

    Pre-collapse state layout (the original numpy version's order):
        [0 .. K*K)               real-real (i, j) full square
        [K*K .. K*K + K)         (k, W) real-W pairs in k order
        [K*K + K .. K*K + 2K)    (W, k) — DUPLICATES of (k, W)
        [K*K + 2K]               (W, W) single state
                                  Total: K*K + 2*K + 1

    Post-collapse state layout (this kernel's output):
        [0 .. n_rr)              unordered real-real pairs (i, j)
                                  with i <= j, in row-major order:
                                  (0,0), (0,1), ..., (0,K-1),
                                  (1,1), (1,2), ..., (K-1,K-1)
                                  where n_rr = K * (K + 1) // 2
        [n_rr .. n_rr + K)       (k, W) real-W pairs in k order
                                  (NO (W, k) duplicates)
        [n_rr + K]               (W, W) single state
                                  Total: n_rr + K + 1
                                       = K*(K+1)/2 + K + 1

    State-count reduction:
        K=4:  25 → 15  (40% smaller)
        K=6:  49 → 28  (43% smaller)
        K=8:  81 → 45  (44% smaller)

    Under flat-penalty Viterbi (viterbi_score_selection's model):
    transition cost is `penalty` for any state-to-state switch
    (regardless of which two states), 0 for self-loops.  This means
    the Viterbi best-path value depends ONLY on per-(bin, state)
    emissions and the count of bin-to-bin switches, not on state
    identity.  Removing emit-identical twin states therefore leaves
    the best-path value bit-identical.

    Emission formulas (unchanged from the original):
      - Real-real pair (i, j):
            log P(g = H[i, l] + H[j, l] | data)
      - Real-W pair (k, W):
            log max(P(H[k, l]), P(H[k, l] + 1)) - lam
      - (W, W):
            log max_g P(g) - 2*lam

    Site emissions are summed within each snps_per_bin-sized bin to
    yield ll_tensor[:, :, b].  The bin boundaries are
    [b*snps_per_bin, min((b+1)*snps_per_bin, L)).

    Returns:
        ll_tensor: (N, K_states, n_bins) float64, contiguous, ready
                   for viterbi_score_selection.  K_states is the
                   POST-COLLAPSE count (n_rr + K + 1 with wildcards,
                   n_rr alone without).

    Floors -log(p) at LOG_EPS_LOCAL = 1e-12 to match _safe_neg_log.
    LOG_EPS_LOCAL is inlined as a literal because module-level
    constants aren't importable inside @njit functions.
    """
    LOG_EPS_LOCAL = 1e-12

    N = probs_k.shape[0]
    L = probs_k.shape[1]
    K = H.shape[0]

    # Unordered real-real pair count: 0+1+...+K = K(K+1)/2.
    n_rr = K * (K + 1) // 2

    if include_wildcards:
        K_states = n_rr + K + 1
    else:
        K_states = n_rr

    ll_tensor = np.zeros((N, K_states, n_bins), dtype=np.float64)

    # Per-sample, per-state, per-bin emission summation.  Sample axis is
    # the outermost parallel axis; state and bin loops are sequential.
    # The inner SNP loop runs at most snps_per_bin sites.
    for s in prange(N):
        for b in range(n_bins):
            start = b * snps_per_bin
            end = start + snps_per_bin
            if end > L:
                end = L

            # Real-real pairs (i, j) with i <= j, in row-major upper-
            # triangular order.  The running index p tracks the state
            # slot; it ends at n_rr after the double loop.  Each (i, j)
            # produces emission log P(g = H[i, l] + H[j, l] | data),
            # summed over the bin's SNPs.  Note dosage H[i,l]+H[j,l] is
            # symmetric in (i, j), so (i, j) and (j, i) have identical
            # emissions — the collapse removes (j, i) for j > i.
            p = 0
            for i in range(K):
                for j in range(i, K):
                    acc = 0.0
                    for l in range(start, end):
                        d = H[i, l] + H[j, l]
                        pv = probs_k[s, l, d]
                        if pv < LOG_EPS_LOCAL:
                            pv = LOG_EPS_LOCAL
                        acc += math.log(pv)
                    ll_tensor[s, p, b] = acc
                    p += 1

            if include_wildcards:
                # (k, W) real-W states in k order.  The wildcard strand
                # picks its allele in {0, 1} to maximise the emission
                # P(H[k, l] + w), giving log max(P(H[k, l]),
                # P(H[k, l] + 1)) - lam per site.  No (W, k) slot
                # because (W, k) has identical emissions to (k, W)
                # (wildcard picks optimally regardless of which strand
                # it occupies) — collapsed.
                for k in range(K):
                    acc = 0.0
                    for l in range(start, end):
                        d0 = H[k, l]
                        d1 = d0 + 1
                        p0 = probs_k[s, l, d0]
                        p1 = probs_k[s, l, d1]
                        pmax = p0 if p0 > p1 else p1
                        if pmax < LOG_EPS_LOCAL:
                            pmax = LOG_EPS_LOCAL
                        acc += math.log(pmax) - lam
                    ll_tensor[s, n_rr + k, b] = acc

                # (W, W) state: max over genotype probabilities.
                # Unchanged from pre-collapse: there's only one (W, W)
                # state (no twin).
                acc = 0.0
                for l in range(start, end):
                    p0 = probs_k[s, l, 0]
                    p1 = probs_k[s, l, 1]
                    p2 = probs_k[s, l, 2]
                    pmax = p0
                    if p1 > pmax:
                        pmax = p1
                    if p2 > pmax:
                        pmax = p2
                    if pmax < LOG_EPS_LOCAL:
                        pmax = LOG_EPS_LOCAL
                    acc += math.log(pmax) - 2.0 * lam
                ll_tensor[s, n_rr + K, b] = acc

    return ll_tensor


def _viterbi_nll(haps_list, probs_k,
                  penalty=None, snps_per_bin=None, lam=None,
                  include_wildcards=True):
    """Scalar NLL = -sum_s best Viterbi LL per sample.  See
    _viterbi_ll_per_sample for parameter semantics."""
    return -float(_viterbi_ll_per_sample(
        haps_list, probs_k, penalty, snps_per_bin, lam,
        include_wildcards).sum())


# =============================================================================
# UPDATE STEP A: pair assignments per sample
# =============================================================================

def _update_A(probs_k, H_k, lam, cost_WW=None, WW_bin_emis=None, log_probs=None):
    """For each sample, pick the pair assignment that minimises its
    capped cost.

    Arguments:
        probs_k: (N, L_kept, 3)
        H_k:     (K, L_kept) discrete in {0, 1}
        lam:     wildcard penalty
        cost_WW: (N, L_kept) optional — precomputed per-(sample, site)
                  WW cost from `_per_site_cost_W_W(probs_k, lam)`.
                  When None, computed internally on every call.  When
                  provided, the WW state of the fused kernel skips its
                  inline pmax/log work; this is what `_fit_at_fixed_K`
                  uses to amortise WW computation across CD iterations.
        WW_bin_emis: (N, n_bins) optional — precomputed Viterbi binned
                  emissions for the WW state.  Derived from cost_WW via
                  `_ww_bin_emis_from_cost_ww`; if cost_WW is supplied
                  but WW_bin_emis is None, the latter is derived here.
        log_probs: (N, L_kept, 3) optional — precomputed
                  log(max(probs_k[s, l, g], LOG_EPS_LOCAL)).  When
                  provided, the fused kernel reads it instead of
                  computing log(probs[s, l, d]) inline at every
                  (s, state, l) visit.  Stable across CD iterations
                  (probs_k doesn't change), so `_fit_at_fixed_K`
                  precomputes once and threads through.  Tier 0 of
                  the optimisation programme; expected ~25% CPU
                  reduction at K=6 N=320 L=200 because the kernel
                  was log()-bound.

    Returns:
        A: (N, 2) int array — A[s, *] in {0..K-1, K} where K = wildcard
            sentinel (one past the last real founder index).  Entries are
            sorted ascending so each unordered pair has a canonical
            representation; W is always placed last.
        per_sample_cost: (N,) — total CAPPED cost under chosen pair (used
            internally as the M-step's view of per-sample fit; bounded
            above by N_kept_sites × cost_WW_per_site)
        per_sample_cost_unc: (N,) — total UNCAPPED cost under the same
            assignment (used as the K-growth NLL improvement signal,
            since capped NLL plateaus when adding founders only converts
            samples from "way over cost_WW" to "still over cost_WW")
        wildcard_slots: (N,) int — number of wildcard strands used by sample
            from the pair assignment alone (0, 1, or 2).  Note: with the
            cap, even (real, real)-assigned samples may effectively use
            wildcards at some sites; this slot count reflects only the
            global pair structure.
    """
    K, L = H_k.shape
    N = probs_k.shape[0]

    probs_c = _maybe_c_contig(probs_k, np.float64)
    H_c = _maybe_c_contig(H_k, np.int64)

    # === FUSED PATH (production default) ===
    # When _VITERBI_BIC_ENABLED is True (the production default) AND
    # K > 0, dispatch to _update_A_fused_kernel which produces baseline
    # (A, wildcard_slots) AND Viterbi-BIC (per_sample_cost = -viterbi_ll)
    # in a single walk through (s, state, l).  Bit-identical to the
    # previous two-pass implementation; see the kernel docstring for the
    # equivalence argument.
    #
    # K = 0 falls through to the baseline-only path below for bit-
    # identity with the legacy K=0 behaviour, which skipped the Viterbi
    # override (the "if _VITERBI_BIC_ENABLED and K > 0" guard in the
    # previous code).  At K = 0 the only state is (W, W), there are no
    # transitions, and per_sample_cost equals the baseline WW cost
    # directly via _per_site_cost_W_W's left-to-right sum.
    if _VITERBI_BIC_ENABLED and K > 0:
        # Bin sizing — same logic as _viterbi_ll_per_sample so the
        # fused kernel's bin_emis matches _viterbi_binned_emissions_-
        # kernel's output bit-for-bit.
        if VITERBI_SNPS_PER_BIN > 1 and VITERBI_SNPS_PER_BIN < L:
            n_bins = int(math.ceil(L / VITERBI_SNPS_PER_BIN))
            snps_per_bin = VITERBI_SNPS_PER_BIN
        else:
            n_bins = L
            snps_per_bin = 1

        # Precompute the WW state's arrays if not provided by the caller.
        # cost_WW depends only on (probs_k, lam) and so doesn't change
        # across CD iterations; `_fit_at_fixed_K` passes a once-per-fit
        # cached version through every call.  WW_bin_emis is derived
        # from cost_WW + (snps_per_bin, n_bins) and is also stable
        # across CD iterations once cached.
        if cost_WW is None:
            cost_WW = _per_site_cost_W_W(probs_c, float(lam))
        cost_WW_c = _maybe_c_contig(cost_WW, np.float64)
        if WW_bin_emis is None:
            WW_bin_emis = _ww_bin_emis_from_cost_ww(
                cost_WW_c, int(snps_per_bin), int(n_bins))
        WW_bin_emis_c = _maybe_c_contig(WW_bin_emis, np.float64)

        # Precompute log_probs once if not provided (Tier 0).  Like
        # cost_WW, log_probs is invariant across CD iterations because
        # probs_k doesn't change, so `_fit_at_fixed_K` caches it once
        # per invocation.  Standalone callers (e.g. unit tests, the
        # recovery loop's _update_A calls outside the CD loop) get
        # internal computation here.
        if log_probs is None:
            log_probs = _log_probs_kernel(probs_c)
        log_probs_c = _maybe_c_contig(log_probs, np.float64)

        A, baseline_cost, wildcard_slots, viterbi_ll = _update_A_fused_kernel(
            probs_c, H_c, float(lam),
            float(VITERBI_SWITCH_PENALTY),
            int(snps_per_bin), int(n_bins),
            cost_WW_c, WW_bin_emis_c, log_probs_c)

        # Per-sample cost = -log-likelihood (NLL convention used elsewhere
        # in this module).  Alias _unc to match the previous code's
        # invariant (downstream callers treat per_sample_cost ==
        # per_sample_cost_unc when Viterbi BIC is active).
        per_sample_cost = (-viterbi_ll).astype(np.float64)
        per_sample_cost_unc = per_sample_cost
        return A, per_sample_cost, per_sample_cost_unc, wildcard_slots

    # === BASELINE-ONLY PATH (K = 0 or Viterbi BIC disabled) ===
    # Fused baseline pass: build cost-per-candidate-pair AND track
    # argmin in-flight, in a single njit kernel that never materialises
    # the (N, n_pairs_rr, L), (N, K, L), or (N, L) per-site cost tensors.
    # At N=320, K=6, L=200 these three tensors total ~14 MB of allocator
    # churn per _update_A call; the fused kernel keeps the running-best
    # candidate in scalar registers and gains cache locality by walking
    # probs_k[s, l, *] for all candidates of sample s in immediate
    # succession (~28 candidates per sample at K=6).
    #
    # Pair assignment uses UNCAPPED costs.  The strict-diploid constraint
    # says each sample has exactly two strands, and the per-pair cost
    # reflects the true model's prediction error under that pair.  We
    # apply the per-(strand, site) wildcard-escape cap (Fix H) only in
    # the M-step (_update_H), where it prevents non-carrier samples from
    # contaminating the founder's update at incompatible sites.  Using
    # the cap in pair assignment would make non-carriers prefer (real, W)
    # ties with (W, W), routing them away from (W, W) and inflating
    # their effective uncapped NLL — which would break the K-growth
    # improvement signal.
    A, per_sample_cost, wildcard_slots = _update_A_baseline_kernel(
        probs_c, H_c, float(lam))
    # Uncapped is the same as the assignment cost since we used uncapped
    # to assign in the first place.  Returned for API symmetry with the
    # Fix-H-cap-in-pair-assignment design that was rejected; downstream
    # callers can treat per_sample_cost == per_sample_cost_unc.
    per_sample_cost_unc = per_sample_cost
    return A, per_sample_cost, per_sample_cost_unc, wildcard_slots


@njit(cache=True, parallel=True, fastmath=False)
def _update_A_baseline_kernel(probs_k, H_k, lam):
    """Fused baseline cost + argmin kernel for _update_A.

    For each sample s, evaluates every candidate pair (real-real with
    i <= j, real-W in k order, then W-W) in immediate succession,
    tracking the running-best (lowest-cost) candidate via scalar
    registers.  Returns the winning (a, b, per_sample_cost,
    wildcard_slots) tuple per sample, with NO intermediate (N, n_pairs,
    L), (N, K, L), or (N, L) cost tensors allocated.

    Iteration order matches the original's flat all_costs concatenation:
        [real-real pairs (i, j) with i <= j, row-major]
      + [real-W pairs (k, W) in k order]
      + [(W, W)]
    so the running-best with STRICT-< update produces the same first-
    occurrence-tiebreak result as np.argmin on the concatenated array.

    Per-site summation order also matches the numpy version: for each
    candidate, the L-sites are accumulated left-to-right (l = 0, 1,
    ..., L-1).  Combined with float64 arithmetic this gives bit-
    identical results to the original three-tensor implementation
    (verified at machine precision).

    Inputs:
        probs_k: (N, L, 3) float64, C-contig
        H_k:     (K, L)    int64,   C-contig
        lam:     wildcard penalty (per strand-site)

    Returns:
        A:               (N, 2)   int64, canonical (real-first, W-second)
        per_sample_cost: (N,)     float64, BASELINE best-pair NLL
        wildcard_slots:  (N,)     int64,  count of W strands in A[s]
                                          (0 for real-real, 1 for kW, 2 for WW)

    Floors -log(p) at LOG_EPS_LOCAL = 1e-12 to match _safe_neg_log.
    LOG_EPS_LOCAL is inlined as a literal because module-level
    constants are not importable inside @njit functions; if you change
    LOG_EPS in the module body, change LOG_EPS_LOCAL in every kernel.
    """
    LOG_EPS_LOCAL = 1e-12

    N = probs_k.shape[0]
    L = probs_k.shape[1]
    K = H_k.shape[0]
    W = K   # wildcard sentinel = one past last real founder index

    A = np.empty((N, 2), dtype=np.int64)
    per_sample_cost = np.empty(N, dtype=np.float64)
    wildcard_slots = np.empty(N, dtype=np.int64)

    # prange over samples — each sample's argmin is independent.  Inner
    # candidate loops are sequential per sample, with the L-site loop
    # innermost for cache locality on the C-contig probs_k.
    for s in prange(N):
        # Sentinels: best_cost = +inf forces the first candidate (the
        # (0, 0) real-real pair if K >= 1, or the (W, W) state if K == 0)
        # to set the initial best.  Strict-< on subsequent updates
        # preserves first-occurrence tiebreak semantics matching the
        # original's np.argmin.
        best_cost = np.inf
        best_a = 0
        best_b = 0
        best_wcs = 0

        # ----- Real-real pairs (i, j) with i <= j, in row-major -----
        # Inner L loop accumulates -log(probs[s, l, H[i,l]+H[j,l]]).
        # When K == 0 this nested loop has zero iterations and falls
        # through to the (W, W) branch below.
        for i in range(K):
            for j in range(i, K):
                cost = 0.0
                for l in range(L):
                    d = H_k[i, l] + H_k[j, l]
                    pv = probs_k[s, l, d]
                    if pv < LOG_EPS_LOCAL:
                        pv = LOG_EPS_LOCAL
                    cost -= math.log(pv)
                if cost < best_cost:
                    best_cost = cost
                    best_a = i
                    best_b = j
                    best_wcs = 0

        # ----- Real-W pairs (k, W) in k order -----
        # Wildcard strand picks its allele w in {0, 1} to maximise
        # probs[s, l, H[k, l] + w] per site; cost = -log of that max
        # plus lam per site.  Summed left-to-right over L.
        for k in range(K):
            cost = 0.0
            for l in range(L):
                d0 = H_k[k, l]
                d1 = d0 + 1
                p0 = probs_k[s, l, d0]
                p1 = probs_k[s, l, d1]
                pmax = p0 if p0 > p1 else p1
                if pmax < LOG_EPS_LOCAL:
                    pmax = LOG_EPS_LOCAL
                cost += -math.log(pmax) + lam
            if cost < best_cost:
                best_cost = cost
                best_a = k
                best_b = W
                best_wcs = 1

        # ----- (W, W) -----
        # Both strands wildcard; each picks its allele optimally,
        # giving max over (p0, p1, p2) per site, plus 2*lam.
        cost = 0.0
        for l in range(L):
            p0 = probs_k[s, l, 0]
            p1 = probs_k[s, l, 1]
            p2 = probs_k[s, l, 2]
            pmax = p0
            if p1 > pmax:
                pmax = p1
            if p2 > pmax:
                pmax = p2
            if pmax < LOG_EPS_LOCAL:
                pmax = LOG_EPS_LOCAL
            cost += -math.log(pmax) + 2.0 * lam
        if cost < best_cost:
            best_cost = cost
            best_a = W
            best_b = W
            best_wcs = 2

        A[s, 0] = best_a
        A[s, 1] = best_b
        per_sample_cost[s] = best_cost
        wildcard_slots[s] = best_wcs

    return A, per_sample_cost, wildcard_slots


@njit(cache=True, parallel=True, fastmath=False)
def _update_A_fused_kernel(probs_k, H_k, lam, penalty, snps_per_bin, n_bins,
                            cost_WW, WW_bin_emis, log_probs):
    """Fused baseline argmin + Viterbi-BIC kernel.

    Single pass through (sample, state, site) that simultaneously
    produces the BASELINE outputs (A, baseline_cost, wildcard_slots)
    consumed by _update_H's bit-voting M-step AND the VITERBI best-
    path LL per sample consumed as the K-growth BIC signal.

    The previous architecture executed two independent kernels:
        _update_A_baseline_kernel    walks (s, state, l) for argmin
        _viterbi_binned_emissions_kernel  walks (s, state, l) for binned
                                          emissions
        viterbi_score_selection       runs forward pass on the bin tensor
    The two walks compute the same per-(s, l, state) scalar log P_pair,
    differing only in summation strategy: baseline accumulates the
    uncapped sum over all L sites per state for argmin; binned-emissions
    accumulates within snps_per_bin-sized bins for the forward pass.
    Profile showed the two walks together at ~33s out of 50.7s of total
    CPU on the 5-block chr1 sample.

    The fused kernel walks (s, state, l) ONCE.  For each per-l scalar
    `lp = log p` (with the per-state offset for kW and WW), it both:
        cost_total  += -lp + offset         (baseline cost, per state)
        bin_emis[st, b] += lp - offset      (Viterbi LL emission, per bin)
    so the two summations are SAME-ORDER (left-to-right within state,
    contiguous across bins) and bit-identical to the legacy kernels.

    WW state pre-bake: cost_WW: (N, L) and WW_bin_emis: (N, n_bins) are
    precomputed by the caller (typically `_fit_at_fixed_K` ONCE per CD
    invocation) and passed in.  The fused kernel's WW state branch
    becomes a pair of memory writes instead of an inner site-loop with
    pmax/log computations.  cost_WW is also threaded through to
    `_update_one_founder_kernel` via the wrapper so the M-step doesn't
    recompute it either.

    Log-probs pre-bake (Tier 0): log_probs: (N, L, 3) is precomputed
    ONCE per `_fit_at_fixed_K` invocation as math.log(max(probs_k, EPS)).
    Inside the kernel, every `lp = log(probs[s, l, d])` becomes
    `lp = log_probs[s, l, d]` — same scalar, no log() in the inner
    loop.  Profile showed `_update_A_fused_kernel` was log()-bound
    (~1.79M log calls per call at K=6 N=320 L=200); caching them
    once per fit eliminates ~5/6 of the in-kernel arithmetic.  The
    kW state's pmax becomes `max(log_probs[s, l, d0], log_probs[s, l, d1])`
    using monotonicity of log.

    After the (state, l) walk, the kernel:
      1. Argmins state_cost[*] with strict-< first-occurrence tiebreak
         to recover (A, wildcard_slots), preserving _update_A_baseline_-
         kernel's semantics.
      2. Runs an in-place Viterbi forward on the per-sample bin_emis
         buffer (alpha updated in place, matching viterbi_score_-
         selection's pattern bit-for-bit) to compute best-path LL.

    Per-sample buffers (bin_emis (K_states, n_bins), state_cost (K_-
    states,), alpha (K_states,)) are allocated inside the prange so
    each thread gets thread-local storage with no false sharing.  At
    K=6, n_bins=20, total per-sample memory is ~4.7 KB — fits in L1.

    K = 0 should NOT call this kernel — the wrapper _update_A handles
    K = 0 via _update_A_baseline_kernel for bit-identity with the
    legacy K=0 path that skipped the Viterbi override entirely.

    Inputs:
        probs_k:       (N, L, 3) float64, C-contig.  Kept as a parameter
                       for signature symmetry; the Tier-0 changes mean
                       the kernel body itself doesn't read probs_k any
                       more (all probability lookups go through
                       log_probs).  Left in place so callers don't
                       need to change argument order.
        H_k:           (K, L)    int64,   C-contig  (K >= 1)
        lam:           wildcard penalty per strand-site
        penalty:       Viterbi switch penalty between adjacent bins
        snps_per_bin:  bin size in SNPs (>= 1)
        n_bins:        total bin count (= ceil(L / snps_per_bin))
        cost_WW:       (N, L)    float64, C-contig — precomputed per-
                       (sample, site) WW cost (-log p_max + 2*lam, with
                       LOG_EPS_LOCAL clamp).  Sum over l gives baseline
                       WW state cost; equivalent up to negation per
                       l to per-site WW LL.
        WW_bin_emis:   (N, n_bins) float64, C-contig — precomputed per-
                       (sample, bin) WW LL emission for Viterbi forward.
        log_probs:     (N, L, 3) float64, C-contig — precomputed
                       log(max(probs_k[s, l, g], LOG_EPS_LOCAL)).  Read
                       in place of every inline log call in the rr and
                       kW state branches.

    Returns:
        A:               (N, 2)   int64
        baseline_cost:   (N,)     float64 — UNCAPPED best-pair NLL per
                                            sample (== _update_A_baseline_-
                                            kernel's per_sample_cost)
        wildcard_slots:  (N,)     int64  — count of W strands in A[s]
        viterbi_ll:      (N,)     float64 — best Viterbi-path LL per
                                            sample

    Floors -log(p) at LOG_EPS_LOCAL = 1e-12 in the rr and kW state
    branches.  WW uses the caller-supplied cost_WW values, which were
    computed with the same LOG_EPS clamp in `_per_site_cost_W_W_kernel`.
    With Tier 0 (precomputed log_probs), the LOG_EPS clamp is applied
    in `_log_probs_kernel` so the kernel body itself doesn't clamp;
    the constant below is kept as a reference but is no longer reached
    on the rr/kW paths.
    """
    LOG_EPS_LOCAL = 1e-12

    N = probs_k.shape[0]
    L = probs_k.shape[1]
    K = H_k.shape[0]
    W = K   # wildcard sentinel

    n_rr = K * (K + 1) // 2
    K_states = n_rr + K + 1     # rr pairs + (k, W) pairs + (W, W)

    A = np.empty((N, 2), dtype=np.int64)
    baseline_cost = np.empty(N, dtype=np.float64)
    wildcard_slots = np.empty(N, dtype=np.int64)
    viterbi_ll = np.empty(N, dtype=np.float64)

    # prange over samples — each sample is independent.  All inner
    # buffers (bin_emis, state_cost, alpha) are allocated INSIDE the
    # prange so they're thread-local and not shared across samples.
    for s in prange(N):
        # Per-sample storage
        bin_emis = np.empty((K_states, n_bins), dtype=np.float64)
        state_cost = np.empty(K_states, dtype=np.float64)

        # =================================================================
        # Single (state, l) walk for rr and kW states.  WW state is
        # filled from precomputed arrays below (no inner l loop).
        # State iteration order matches _update_A_baseline_kernel and
        # _viterbi_binned_emissions_kernel so argmin tiebreak and Viterbi
        # state indices are aligned: rr pairs (i, j) with i <= j in row-
        # major, then (k, W) in k order, then (W, W).
        # =================================================================
        st = 0

        # ----- Real-real pairs (i, j) with i <= j -----
        # Per-site emission: log P(g = H[i,l] + H[j,l]).  Baseline cost
        # accumulates -log p; Viterbi bin accumulates +log p.  Reads
        # the precomputed log_probs[s, l, d] (Tier 0) instead of
        # computing log(probs[s, l, d]) inline.
        for i in range(K):
            for j in range(i, K):
                cost_total = 0.0
                for b in range(n_bins):
                    start = b * snps_per_bin
                    end = start + snps_per_bin
                    if end > L:
                        end = L
                    bin_acc = 0.0
                    for l in range(start, end):
                        d = H_k[i, l] + H_k[j, l]
                        lp = log_probs[s, l, d]
                        bin_acc += lp
                        cost_total += -lp
                    bin_emis[st, b] = bin_acc
                state_cost[st] = cost_total
                st += 1

        # ----- Real-W pairs (k, W) -----
        # Per-site emission: log max(P(H[k,l]), P(H[k,l]+1)) - lam.
        # Baseline cost: -log pmax + lam.  Same scalars, negated.
        # Using monotonicity of log, log max(p0, p1) = max(log p0,
        # log p1), so we compare/select directly on precomputed log
        # values.  The post-EPS-clamp comparison on raw probs (p0 > p1
        # with both >= EPS) is identical to the comparison on log_probs
        # values, because log is strictly monotonic on the positive
        # reals.  When p0 < EPS we read log_probs[..., d0] = log(EPS)
        # and likewise for d1, recovering the legacy EPS-clamp result.
        for k in range(K):
            cost_total = 0.0
            for b in range(n_bins):
                start = b * snps_per_bin
                end = start + snps_per_bin
                if end > L:
                    end = L
                bin_acc = 0.0
                for l in range(start, end):
                    d0 = H_k[k, l]
                    d1 = d0 + 1
                    lp0 = log_probs[s, l, d0]
                    lp1 = log_probs[s, l, d1]
                    lp = lp0 if lp0 > lp1 else lp1
                    bin_acc += lp - lam
                    cost_total += -lp + lam
                bin_emis[st, b] = bin_acc
            state_cost[st] = cost_total
            st += 1

        # ----- (W, W) — from precomputed cost_WW and WW_bin_emis -----
        # Baseline cost: sum_l cost_WW[s, l] (left-to-right; matches
        # the legacy inline summation order bit-for-bit).
        # Viterbi bin emissions: precomputed copy.
        cost_total = 0.0
        for l in range(L):
            cost_total += cost_WW[s, l]
        for b in range(n_bins):
            bin_emis[st, b] = WW_bin_emis[s, b]
        state_cost[st] = cost_total

        # =================================================================
        # Baseline argmin (strict-< first-occurrence tiebreak, matching
        # _update_A_baseline_kernel).  state_cost layout: rr (i,j) row-
        # major for slots [0..n_rr), then kW for slots [n_rr..n_rr+K),
        # then (W,W) at slot n_rr+K.
        # =================================================================
        best_cost = np.inf
        best_state_idx = 0
        for st_iter in range(K_states):
            if state_cost[st_iter] < best_cost:
                best_cost = state_cost[st_iter]
                best_state_idx = st_iter

        # Map best_state_idx back to (a, b, wcs).  For rr block, decode
        # row-major upper-triangular index by walking rows in order.
        if best_state_idx < n_rr:
            remaining = best_state_idx
            for i in range(K):
                row_len = K - i           # pairs (i, j) with j in [i, K)
                if remaining < row_len:
                    A[s, 0] = i
                    A[s, 1] = i + remaining
                    wildcard_slots[s] = 0
                    break
                remaining -= row_len
        elif best_state_idx < n_rr + K:
            A[s, 0] = best_state_idx - n_rr
            A[s, 1] = W
            wildcard_slots[s] = 1
        else:
            A[s, 0] = W
            A[s, 1] = W
            wildcard_slots[s] = 2

        baseline_cost[s] = best_cost

        # =================================================================
        # Viterbi forward on bin_emis (in-place alpha, matching
        # viterbi_score_selection's exact pattern).  Single buffer; the
        # update for state k only reads alpha[k] (its own old value) and
        # the pre-computed best_prev, so in-place is correct.
        # =================================================================
        alpha = np.empty(K_states, dtype=np.float64)
        for st_iter in range(K_states):
            alpha[st_iter] = bin_emis[st_iter, 0]
        for b in range(1, n_bins):
            best_prev = -np.inf
            for st_iter in range(K_states):
                if alpha[st_iter] > best_prev:
                    best_prev = alpha[st_iter]
            switch_base = best_prev - penalty
            for st_iter in range(K_states):
                em = bin_emis[st_iter, b]
                stay = alpha[st_iter]
                if stay > switch_base:
                    alpha[st_iter] = stay + em
                else:
                    alpha[st_iter] = switch_base + em
        best_final = -np.inf
        for st_iter in range(K_states):
            if alpha[st_iter] > best_final:
                best_final = alpha[st_iter]
        viterbi_ll[s] = best_final

    return A, baseline_cost, wildcard_slots, viterbi_ll


@njit(cache=True)
def _update_A_translate_kernel(best_idx, pair_idx_arr, n_rr, K):
    """Translate a flat best-pair index back to (a, b) and wildcard slots.

    Index layout in best_idx (matches all_costs concatenation in _update_A):
        [0 .. n_rr)         real-real pair states; pair_idx_arr[p] = (i, j)
        [n_rr .. n_rr + K)  real-W pair states; bi - n_rr = real founder index
        n_rr + K            (W, W) state

    Returns (A, wildcard_slots) where:
        A:               (N, 2) int64, with W = K used as the wildcard sentinel.
                         Entries are canonical (real-first, W-second for kW;
                         real-real pairs come straight from pair_idx_arr).
        wildcard_slots:  (N,) int64 in {0, 1, 2}
    """
    N = best_idx.shape[0]
    W = K
    A = np.empty((N, 2), dtype=np.int64)
    wildcard_slots = np.empty(N, dtype=np.int64)

    for s in range(N):
        bi = best_idx[s]
        if bi < n_rr:
            A[s, 0] = pair_idx_arr[bi, 0]
            A[s, 1] = pair_idx_arr[bi, 1]
            wildcard_slots[s] = 0
        elif bi < n_rr + K:
            k_real = bi - n_rr
            A[s, 0] = k_real
            A[s, 1] = W
            wildcard_slots[s] = 1
        else:
            A[s, 0] = W
            A[s, 1] = W
            wildcard_slots[s] = 2
    return A, wildcard_slots


# =============================================================================
# UPDATE STEP H: founder allele updates
# =============================================================================

def _update_H(probs_k, H_k, A, lam, cost_WW=None, log_probs=None):
    """For each (founder, kept site), pick the binary value that minimises
    NLL contribution from samples carrying that founder.

    Updates H_k in-place and returns the number of bits flipped (so the
    coordinate descent loop can detect convergence).

    Arguments:
        probs_k: (N, L_kept, 3)
        H_k:     (K, L_kept) — modified in place
        A:       (N, 2)   pair assignments, with K used as the wildcard sentinel
        lam:     wildcard penalty
        cost_WW: (N, L_kept) optional — precomputed per-(sample, site) WW
                  cost from `_per_site_cost_W_W(probs_k, lam)`.  When
                  provided, threads through to `_update_one_founder` and
                  ultimately to the kernel, which reads cost_WW[s, l]
                  instead of recomputing it inline for every (k, l).
                  When None, the kernel falls back to inline computation
                  (so the wrapper computes it once here for consistency).
        log_probs: (N, L_kept, 3) optional — precomputed
                  log(max(probs_k[s, l, g], LOG_EPS_LOCAL)) (Tier 0).
                  When provided, the kernel uses it in place of every
                  inline log call.  Same caching pattern as cost_WW.

    Returns:
        n_changes: int — number of (founder, site) bits that flipped
    """
    K, L = H_k.shape
    N = probs_k.shape[0]
    W = K

    # Compute cost_WW once here if not supplied, so the K founder
    # updates below all share the same precomputed array (no per-
    # founder re-derivation).
    if cost_WW is None:
        cost_WW = _per_site_cost_W_W(probs_k, lam)

    # Same pattern for log_probs (Tier 0).  When called from
    # `_fit_at_fixed_K`, log_probs arrives precomputed and is reused
    # across all K founder updates AND across all CD iterations.
    if log_probs is None:
        log_probs = _log_probs_kernel(_maybe_c_contig(probs_k, np.float64))

    # We update founders in decreasing order of usage.  Compute usage from A
    # via an njit kernel — the inner Python loop over K with two boolean-
    # mask sums per K was a small hot spot (4K mask scans, each scanning
    # N entries).  The kernel does it in a single pass over A.
    A_c = _maybe_c_contig(A, np.int64)
    usage = _update_H_usage_kernel(A_c, K)
    update_order = np.argsort(-usage, kind='stable')

    n_changes = 0
    for k in update_order:
        n_changes += _update_one_founder(probs_k, H_k, A, int(k), lam,
                                          cost_WW=cost_WW, log_probs=log_probs)
    return n_changes


@njit(cache=True)
def _update_H_usage_kernel(A, K):
    """Count per-founder usage from pair-assignment array A.

    A entries are in [0, K]; entries equal to K are the wildcard sentinel
    and don't count.  For each k in [0, K), usage[k] = number of (s, slot)
    entries with A[s, slot] == k.  A pair (k, k) contributes 2.
    """
    N = A.shape[0]
    usage = np.zeros(K, dtype=np.int64)
    for s in range(N):
        for slot in range(2):
            f = A[s, slot]
            if f != K:
                usage[f] += 1
    return usage


def _update_one_founder(probs_k, H_k, A, k, lam, cost_WW=None, log_probs=None):
    """For founder k, at each kept site, evaluate cost contribution from
    samples carrying k under H_k[k, l] = 0 vs = 1, and pick the lower.

    Per-(sample, site) costs are capped at cost_WW(s, l) (Fix H).  This
    implements per-strand-per-site wildcard escape: a sample whose
    pair-fit at site l exceeds cost_WW (i.e., the founder doesn't
    represent the sample at this site) contributes cost_WW to BOTH H=0
    and H=1, contributing zero preference.  This prevents non-carrier
    samples from contaminating the founder's update at incompatible
    sites while still allowing them to vote at agreeing sites.

    H_k is modified in place at row k.  Returns the number of sites flipped.

    cost_WW: optional precomputed (N, L) WW cost array.  When None,
    computed here so the kernel can read from it.  When provided
    (typical inside `_fit_at_fixed_K`'s CD loop), reused across
    K founders within one M-step iteration and across all CD iterations.

    log_probs: optional precomputed (N, L, 3) log array (Tier 0).  When
    provided, the kernel reads log_probs[s, l, g] in place of every
    inline math.log(probs[s, l, g]).  Stable across CD iterations and
    across M-step founders, so `_fit_at_fixed_K` precomputes once.
    """
    K, L = H_k.shape
    N = probs_k.shape[0]
    W = K

    # Identify samples whose pair contains k as a REAL strand.  We split
    # into three buckets by partner type:
    #   Bucket H: A[s] = (k, k)              — homozygous for k
    #   Bucket J: A[s] = (k, j) with j != k, j real   — k paired with another real founder
    #   Bucket P: A[s] = (k, W)              — k paired with wildcard
    # Note A is sorted ascending with W always last, so (k, k) means both
    # entries are k; (k, j) with j != k real means one entry is k and the
    # other is some j with j != k and j != W; (k, W) means one entry is k
    # and the other is W.
    is_kk = (A[:, 0] == k) & (A[:, 1] == k)
    is_kW = ((A[:, 0] == k) & (A[:, 1] == W))
    # A[:, 0] != A[:, 1] and one of them is k and the other is real (not W)
    has_k = (A[:, 0] == k) | (A[:, 1] == k)
    is_kj = has_k & ~is_kk & ~is_kW

    # If founder k has no support, leave H_k[k] untouched
    n_supp = int(is_kk.sum() + is_kj.sum() + is_kW.sum())
    if n_supp == 0:
        return 0

    # For each sample in bucket J, identify the partner founder index j_s
    if is_kj.any():
        # j_s = the entry in A[s] that's NOT k
        a0 = A[is_kj, 0]
        a1 = A[is_kj, 1]
        partner_J = np.where(a0 == k, a1, a0)               # (n_J,)
        kj_sample_idx = np.where(is_kj)[0]                  # (n_J,)
    else:
        partner_J = np.empty(0, dtype=np.int64)
        kj_sample_idx = np.empty(0, dtype=np.int64)

    kk_sample_idx = np.where(is_kk)[0]
    kW_sample_idx = np.where(is_kW)[0]

    # Compute cost_WW if not supplied.  This is the wrapper's last chance
    # to reuse a precomputed array; the kernel REQUIRES a passed-in
    # cost_WW (no internal fallback) because keeping the kernel signature
    # uniform avoids a code-duplicated cold path.
    if cost_WW is None:
        cost_WW = _per_site_cost_W_W(probs_k, lam)

    # Compute log_probs if not supplied (Tier 0).  Same kernel-requires-
    # uniform-signature rationale as cost_WW above.
    if log_probs is None:
        log_probs = _log_probs_kernel(_maybe_c_contig(probs_k, np.float64))

    # Hand off to the njit per-site kernel.  This replaces the original
    # Python `for l in range(L):` loop with O(L) per-site numpy slicing
    # (which had high overhead at L≈200 across many coord-descent
    # iterations).
    probs_c = _maybe_c_contig(probs_k, np.float64)
    H_row_c = _maybe_c_contig(H_k[k], np.int64)
    H_c = _maybe_c_contig(H_k, np.int64)
    kk_idx_c = _maybe_c_contig(kk_sample_idx, np.int64)
    kj_idx_c = _maybe_c_contig(kj_sample_idx, np.int64)
    kW_idx_c = _maybe_c_contig(kW_sample_idx, np.int64)
    partner_J_c = _maybe_c_contig(partner_J, np.int64)
    cost_WW_c = _maybe_c_contig(cost_WW, np.float64)
    log_probs_c = _maybe_c_contig(log_probs, np.float64)
    new_row, n_changes = _update_one_founder_kernel(
        probs_c, H_row_c, H_c,
        kk_idx_c, kj_idx_c, kW_idx_c, partner_J_c,
        float(lam), cost_WW_c, log_probs_c)
    # Write back into H_k in-place at row k.
    H_k[k] = new_row
    return int(n_changes)


@njit(cache=True, parallel=True, fastmath=False)
def _update_one_founder_kernel(probs_k, H_row, H_full,
                                kk_idx, kj_idx, kW_idx, partner_J,
                                lam, cost_WW, log_probs):
    """njit version of the per-site loop in _update_one_founder.

    Replaces the original Python `for l in range(L):` body with explicit
    per-bucket per-sample accumulation.  Each bucket's contribution to
    nll0 and nll1 is capped at cost_WW(s, l) — the cap mirrors Fix H
    exactly.

    Three changes from the original parallel=False version:
      1. cost_WW is passed in (precomputed once per `_fit_at_fixed_K`
         invocation) instead of recomputed inline per (s, l, bucket).
         Eliminates the redundant -log(p_max) + 2*lam computation that
         was previously done THREE TIMES per (s, l) within each kernel
         call (once per bucket type), times the K founders updated in
         one M-step iteration.
      2. log_probs is passed in (Tier 0 precompute).  All inline
         `-math.log(max(probs[s, l, d], EPS))` patterns become
         `-log_probs[s, l, d]`, eliminating the log() call (the kernel
         was log()-bound on bucket J under K=6).  Bucket P's max-pick
         on (p0, p1) and (p1, p2) uses monotonicity of log: select via
         comparison on log_probs values directly.
      3. The outer site loop is `prange(L)` (parallel=True) so that
         large numba thread budgets — which workers get dynamically via
         `_update_dynamic_threads` as the contig's slow-tail blocks
         finish — actually accelerate the M-step.  Each l writes an
         independent `new_row[l]` and the `n_changes` accumulator is a
         simple `+= 1` reduction that numba recognises automatically.

    Bit-equivalence: identical scalars in identical accumulation order
    within each l (sequential bucket loops); each l is independent so
    parallel iteration is safe.  Verified against the legacy parallel=
    False version at machine precision.

    Arguments:
        probs_k:    (N, L, 3) float64, C-contig — kept in the signature
                    for symmetry, but no longer read inside the kernel
                    body (all probability lookups go through log_probs).
                    Removing the parameter would force changes in every
                    caller; leaving it as a noop preserves wrapper code.
        H_row:      (L,) int64 — the CURRENT row k of H_k that will be
                    updated.  The function returns the NEW row; the
                    caller writes it back into H_k[k].  We pass the row
                    in (not just an empty buffer) so the "keep current
                    value when nll0 == nll1" branch sees the right
                    reference value.
        H_full:     (K, L) int64 — full H matrix; only column 'partner_J'
                    rows are read inside the loop (Bucket J uses partner
                    indices into H_full).
        kk_idx:     (n_kk,) int64 — sample indices in bucket H (kk)
        kj_idx:     (n_J,)  int64 — sample indices in bucket J
        kW_idx:     (n_P,)  int64 — sample indices in bucket P (kW)
        partner_J:  (n_J,)  int64 — partner founder index for each
                    bucket-J sample, in the same order as kj_idx
        lam:        wildcard penalty
        cost_WW:    (N, L) float64 — precomputed per-(sample, site) WW
                    cost from `_per_site_cost_W_W_kernel`.  Read inline
                    inside each bucket as the cap value, replacing the
                    previous inline pmax+log computation.
        log_probs:  (N, L, 3) float64 — precomputed log(max(probs[s, l, g],
                    LOG_EPS_LOCAL)).  Read in place of every inline log
                    call in the three buckets.  The LOG_EPS clamp is
                    applied at precompute time in `_log_probs_kernel`,
                    so the kernel body itself doesn't clamp.

    Returns:
        new_row:    (L,) int64 — new H_k[k] row
        n_changes:  int — number of (k, l) bits that flipped

    The "no-signal" branch (|nll0 - nll1| < 1e-9) keeps H_row[l]
    unchanged, matching the Python version's behaviour.
    """
    LOG_EPS_LOCAL = 1e-12

    L = H_row.shape[0]
    n_kk = kk_idx.shape[0]
    n_J = kj_idx.shape[0]
    n_P = kW_idx.shape[0]

    new_row = np.empty(L, dtype=np.int64)
    n_changes = 0

    for l in prange(L):
        cur_val = H_row[l]
        nll0 = 0.0
        nll1 = 0.0

        # Bucket H (k, k): dosage = 0 under hk=0, dosage = 2 under hk=1.
        for ii in range(n_kk):
            s = kk_idx[ii]
            cost_WW_s = cost_WW[s, l]

            # Raw cost under hk=0: -log p[s, l, 0]; under hk=1: -log p[s, l, 2].
            # Read precomputed log values (Tier 0); EPS clamp was applied
            # in `_log_probs_kernel`.
            c_h0 = -log_probs[s, l, 0]
            c_h1 = -log_probs[s, l, 2]

            # Cap at cost_WW.
            if c_h0 > cost_WW_s:
                c_h0 = cost_WW_s
            if c_h1 > cost_WW_s:
                c_h1 = cost_WW_s
            nll0 += c_h0
            nll1 += c_h1

        # Bucket J (k, j): dosage = partner_h_at_l + hk.
        for ii in range(n_J):
            s = kj_idx[ii]
            j = partner_J[ii]
            cost_WW_s = cost_WW[s, l]
            partner_h = H_full[j, l]

            d_h0 = partner_h          # dosage if hk=0
            d_h1 = partner_h + 1      # dosage if hk=1
            c_h0 = -log_probs[s, l, d_h0]
            c_h1 = -log_probs[s, l, d_h1]
            if c_h0 > cost_WW_s:
                c_h0 = cost_WW_s
            if c_h1 > cost_WW_s:
                c_h1 = cost_WW_s
            nll0 += c_h0
            nll1 += c_h1

        # Bucket P (k, W): wildcard strand picks its bit optimally.
        # Under hk=0: candidate dosages {0, 1}; under hk=1: {1, 2}.
        # Cost = -log max-prob + lam.
        # log(max(p_a, p_b)) = max(log p_a, log p_b) by monotonicity, so
        # we select the larger of the two precomputed log values directly
        # (no need to read probs and apply EPS clamp again — both were
        # done at precompute time).  The comparison on log_probs values
        # gives the same selection as the legacy comparison on raw probs
        # after EPS clamping, since log is strictly increasing.
        for ii in range(n_P):
            s = kW_idx[ii]
            cost_WW_s = cost_WW[s, l]
            lp0 = log_probs[s, l, 0]
            lp1 = log_probs[s, l, 1]
            lp2 = log_probs[s, l, 2]

            # hk=0: best of (p0, p1) -> max(lp0, lp1)
            best_lp0 = lp0 if lp0 > lp1 else lp1
            # hk=1: best of (p1, p2) -> max(lp1, lp2)
            best_lp1 = lp1 if lp1 > lp2 else lp2
            c_h0 = -best_lp0 + lam
            c_h1 = -best_lp1 + lam
            if c_h0 > cost_WW_s:
                c_h0 = cost_WW_s
            if c_h1 > cost_WW_s:
                c_h1 = cost_WW_s
            nll0 += c_h0
            nll1 += c_h1

        # Pick lower-NLL value.  No-signal handling: if nll0 == nll1
        # (within numerical precision), no sample expressed a meaningful
        # preference at this site (e.g., all attributing samples were
        # capped out under both H values, giving zero discriminating
        # signal).  In that case keep cur_val to avoid arbitrary flips.
        diff = nll0 - nll1
        if diff < 0.0:
            diff = -diff
        if diff < 1e-9:
            new_val = cur_val
        else:
            if nll0 < nll1:
                new_val = 0
            else:
                new_val = 1
        new_row[l] = new_val
        if new_val != cur_val:
            n_changes += 1

    return new_row, n_changes


# =============================================================================
# COORDINATE DESCENT AT FIXED K
# =============================================================================

def _fit_at_fixed_K(probs_k, H_init, lam, max_iter=50):
    """Run discrete coordinate descent at the K determined by H_init.shape[0].

    Alternates updating A (pair assignments) and H (founder bits) until
    no changes are made in a full pass, or max_iter is reached.

    Arguments:
        probs_k: (N, L_kept, 3) — kept-site posteriors
        H_init:  (K, L_kept) discrete {0, 1} — initial founder values
        lam:     wildcard penalty
        max_iter: cap on coordinate descent iterations

    Returns:
        H:               final (K, L_kept)
        A:               final (N, 2)
        per_sample_cost: (N,) total CAPPED cost per sample under final state
                          (used in worst-fit-sample selection for K-growth seed)
        wildcard_slots:  (N,) wildcard strand count per sample
        n_iter:          how many iterations were used
        total_NLL:       scalar — UNCAPPED NLL summed across samples (used
                          by K-growth as the improvement signal; the cap
                          would mask improvements where adding a founder
                          converts samples from "way over cost_WW" to
                          "still over cost_WW but less so")
    """
    H = H_init.copy()
    A_prev = None
    n_iter = 0

    # Initialise result variables so the post-loop block can refer to
    # them whether or not the loop body executed (max_iter=0 edge case)
    # or completed any iterations.
    A = None
    per_sample_cost = None
    per_sample_cost_unc = None
    wildcard_slots = None

    # ---------------------------------------------------------------------
    # Pre-bake invariant quantities ONCE per CD invocation.
    #
    # cost_WW depends only on (probs_k, lam) — neither changes inside the
    # CD loop — so it's wasteful to recompute it in every _update_A and
    # _update_H call.  Same for WW_bin_emis, which is a fixed binning of
    # cost_WW.  Same for log_probs (Tier 0), which depends only on
    # probs_k.  We compute all three here and thread them through.
    #
    # The fused _update_A_fused_kernel skips its WW state's inner site-
    # loop entirely (replaced by writes from WW_bin_emis) and reads
    # log_probs[s, l, d] inline in the rr and kW branches instead of
    # calling math.log per visit.  The M-step kernel
    # _update_one_founder_kernel reads cost_WW[s, l] for the cap value
    # and log_probs[s, l, g] for the per-genotype cost.
    # ---------------------------------------------------------------------
    L = probs_k.shape[1]
    cost_WW = _per_site_cost_W_W(probs_k, lam)
    probs_c = _maybe_c_contig(probs_k, np.float64)
    log_probs = _log_probs_kernel(probs_c)
    if _VITERBI_BIC_ENABLED:
        if VITERBI_SNPS_PER_BIN > 1 and VITERBI_SNPS_PER_BIN < L:
            _snps_per_bin = VITERBI_SNPS_PER_BIN
            _n_bins = int(math.ceil(L / VITERBI_SNPS_PER_BIN))
        else:
            _snps_per_bin = 1
            _n_bins = L
        cost_WW_c = _maybe_c_contig(cost_WW, np.float64)
        WW_bin_emis = _ww_bin_emis_from_cost_ww(
            cost_WW_c, int(_snps_per_bin), int(_n_bins))
    else:
        WW_bin_emis = None

    # Tracks whether we need to recompute A and per-sample costs after
    # the loop exits.  When the loop exits via CD CONVERGENCE
    # (not a_changed and h_changes == 0 at the break point), the last
    # _update_H call did NOT change H, so the A computed earlier in the
    # same loop iteration is still consistent with the (unchanged) H.
    # In that case we can skip the post-loop _update_A — a clean win of
    # one _update_A call per converged _fit_at_fixed_K invocation, which
    # is the common case (most blocks converge in 3-10 iterations, well
    # under max_iter=50).
    #
    # When the loop exits via max_iter (no convergence), the LAST
    # _update_H call MAY have changed H, in which case A is now stale
    # and must be recomputed.  We can't tell whether h_changes was 0 at
    # the final iteration of the max_iter path, so we conservatively
    # recompute in that case.  (Cheap insurance — max_iter is rarely
    # reached.)
    need_recompute = True

    for it in range(max_iter):
        # Update A given H — pass precomputed WW arrays and log_probs
        # for the fused kernel to consume; identical results to
        # recomputing inside.
        A, per_sample_cost, per_sample_cost_unc, wildcard_slots = _update_A(
            probs_k, H, lam, cost_WW=cost_WW, WW_bin_emis=WW_bin_emis,
            log_probs=log_probs)

        # Convergence check via A
        a_changed = (A_prev is None) or (not np.array_equal(A, A_prev))
        A_prev = A.copy()

        # Update H given A — pass precomputed cost_WW and log_probs for
        # the kernel to use directly.
        h_changes = _update_H(probs_k, H, A, lam, cost_WW=cost_WW,
                              log_probs=log_probs)

        n_iter = it + 1
        if not a_changed and h_changes == 0:
            # Converged: H didn't change in this iteration's _update_H
            # call, so the A computed above is still consistent.  Skip
            # the post-loop recompute.
            need_recompute = False
            break

    if need_recompute:
        # Either we never entered the loop (max_iter=0) or we exited via
        # max_iter without converging (in which case the final _update_H
        # call may have changed H, making A stale).  Recompute.
        A, per_sample_cost, per_sample_cost_unc, wildcard_slots = _update_A(
            probs_k, H, lam, cost_WW=cost_WW, WW_bin_emis=WW_bin_emis,
            log_probs=log_probs)
    # Use UNCAPPED NLL as the K-growth signal (see docstring).
    total_NLL = float(per_sample_cost_unc.sum())

    return H, A, per_sample_cost, wildcard_slots, n_iter, total_NLL


# =============================================================================
# BIC HELPERS (the complexity-cost and BIC formula, shared by every BIC site)
# =============================================================================
#
# BIC convention used throughout this module (linear-BIC, project standard
# matching beam_search_core / chimera_resolution):
#
#     BIC(K) = K * cc + 2 * NLL_K          (lower is better)
#
# where cc is the per-founder complexity cost.  At fixed K, BIC and NLL
# differ only by an additive constant K*cc, so any fixed-K decision (e.g.
# per-(founder, site) bit voting in _update_one_founder, picking the best
# K_cur+1 candidate among K_cur+1 same-K candidates) is identical under
# either score.  Decisions across different K values (K-growth acceptance,
# multi-medoid trajectory selection, history reporting) MUST use BIC so
# the complexity penalty is properly accounted for.
#
# The acceptance criterion BIC(K+1) < BIC(K) reduces algebraically to
# NLL_improvement > cc/2; see _grow_K for the derivation.

def _compute_cc(cc_scale, N, L_kept, use_log_bic=False):
    """Per-founder complexity cost cc used in BIC = K * cc + 2 * NLL.

    Linear BIC (default, project standard):
        cc = cc_scale * (L_kept / 200) * N
    Standard log-BIC (use_log_bic=True):
        cc = cc_scale * log(N * L_kept) * (L_kept / 200)

    Linear scaling is preferred over log(N) at the per-block EM stage
    because log(N) is too weak when N is large (~320 here), allowing
    spurious founder additions to slip past the BIC threshold.  See
    chimera_resolution.compute_cc and _grow_K's docstring for the full
    rationale.
    """
    snp_growth = L_kept / 200.0
    if use_log_bic:
        log_n = math.log(max(N * L_kept, 2))
        return cc_scale * log_n * snp_growth
    return cc_scale * snp_growth * N


def _compute_bic(K, nll, cc):
    """BIC = K * cc + 2 * NLL.  Lower is better.

    Centralises the formula so every place that compares solutions
    across different K values uses the same convention.  At fixed K
    this is a constant offset from NLL (so ordering is preserved) but
    the absolute number is informative when comparing across K.
    """
    return K * cc + 2.0 * nll



def _compute_nll_for_subset(haps_list, probs_k, lam):
    """Score a subset of haps by computing UNCAPPED NLL on sample data.
    Haps are FIXED (not refined) during scoring — used for outer BIC
    subset-selection where we want to compare different SUBSETS of a
    fixed candidate pool.

    For empty subset (K=0): NLL = sum of cost_WW per sample.

    When _VITERBI_BIC_ENABLED is True (production default), NLL is the
    Viterbi best-path NLL over founder-pair states with switching
    allowed at VITERBI_SWITCH_PENALTY cost per switch.  When False, the
    baseline best-pair-per-sample NLL is returned (preserved for A/B
    revert; see the VITERBI BIC SCORING constants block for rationale).
    """
    # K=0 path is identical under both scoring schemes: only state is
    # (W, W), no transitions to consider.  Handled inline so this branch
    # bypasses both _update_A and _viterbi_ll_per_sample.
    if len(haps_list) == 0:
        cost_WW_per_site = _per_site_cost_W_W(probs_k, lam)
        return float(cost_WW_per_site.sum())

    # Viterbi scoring (production default): call _viterbi_nll directly,
    # bypassing _update_A entirely.  This is correct AND faster than
    # going through _update_A, which would compute the baseline best-
    # pair-per-sample cost and then overwrite it with Viterbi NLL —
    # wasted work for a scoring-only call site that doesn't need A or
    # wildcard_slots.
    if _VITERBI_BIC_ENABLED:
        return _viterbi_nll(
            haps_list, probs_k,
            penalty=VITERBI_SWITCH_PENALTY,
            snps_per_bin=VITERBI_SNPS_PER_BIN,
            lam=lam)

    # Baseline best-pair-per-sample scoring (legacy path, retained for
    # A/B comparison when _VITERBI_BIC_ENABLED = False).  Stacks the
    # haps, runs _update_A which enumerates all candidate pairs, and
    # returns the sum of per-sample uncapped costs under the best pair.
    H = np.stack(haps_list, axis=0)
    A, per_sample_cost, per_sample_cost_unc, wildcard_slots = _update_A(probs_k, H, lam)
    return float(per_sample_cost_unc.sum())


# =============================================================================
# POOL EMISSION CACHE — precompute Viterbi emissions once per BIC-search
# =============================================================================
#
# The recovery loop (_subtraction_recovery_round_loop, _late_low_carrier_-
# rescue) and the trio/pairwise seed-trim (in block_haplotypes_discrete.py's
# _grow_K_with_recovery) each do hundreds-to-thousands of `_compute_nll_-
# for_subset` calls over subsets drawn from a FIXED candidate pool.  Each
# call independently rebuilds the (N, K_states, n_bins) binned-emission
# tensor via _viterbi_binned_emissions_kernel.  Since the emission for
# state (i, j) depends ONLY on the haps H[i], H[j] and the data probs_k
# — NOT on what other states are in the subset — the rebuild is mostly
# redundant work.
#
# PoolEmissionCache precomputes the full pool emission tensor ONCE per
# search session.  Each `nll_for_subset` call then:
#   1. Maps the subset's state slots back to the pool's state slots via
#      a tiny index map (njit kernel).
#   2. Fancy-indexes the pool tensor along the state axis to assemble
#      the subset's (N, K_sub_states, n_bins) tensor.
#   3. Runs viterbi_score_selection on the assembled subset tensor.
#
# Memory: O(N * n_states_pool * n_bins).  For pool size K_pool=25, N=320,
# n_bins=20: n_states_pool = K_pool*(K_pool+1)/2 + K_pool + 1 = 351
# states; tensor size 320*351*20*8 = ~18 MB.  Bounded by typical pool
# sizes (≤ 30) so cache stays under 30 MB.
#
# Lifetime: one BIC-search session (one recovery round, one late-rescue
# invocation, or one seed-trim).  Built, queried thousands of times,
# discarded.
#
# Bit-exact equivalence: the cache uses the same _viterbi_binned_-
# emissions_kernel as the fresh build, so the pool emission tensor is
# bit-identical to what fresh-build-on-subset would produce at the same
# slot positions (modulo permutation, since the subset's state layout
# is a re-ordering of the corresponding pool slots).  viterbi_score_-
# selection's output is invariant under state-slot permutation (the
# best-path NLL doesn't depend on which integer label the states carry),
# so cache.nll_for_subset(indices) == _compute_nll_for_subset(
# [pool[i] for i in indices], ...) bit-identically.  Verified at machine
# precision against the legacy path.

@njit(cache=True)
def _build_pool_state_index_map_kernel(subset_indices, K_pool):
    """Build the (K_sub_states,) array mapping each subset state slot
    to the corresponding slot in the pool's collapsed-state-space layout.

    Subset state layout (same as _viterbi_binned_emissions_kernel post-
    collapse, but over the K_sub = len(subset_indices) selected pool
    members):
        [0 .. K_sub*(K_sub+1)/2)        unordered (a, b) with a <= b
                                          (positions into subset_indices)
        [.. + K_sub)                     (k_sub, W) for k_sub in [0, K_sub)
        [.. + 1]                         (W, W) — single slot

    For each subset state slot, we compute the corresponding pool state
    slot.  Pool state layout (same convention, but for the K_pool pool):
        rr pairs at index `lo * K_pool - lo*(lo-1)/2 + (hi - lo)` for
            lo <= hi
        kW pairs at index `n_rr_pool + k`
        WW   at index `n_rr_pool + K_pool`

    For a subset (a, b) with a <= b in the subset's index space, the
    pool pair is (subset_indices[a], subset_indices[b]); we re-order to
    (lo, hi) with lo = min, hi = max before looking up.  Subset (k_sub,
    W) maps to pool (subset_indices[k_sub], W).  (W, W) maps to (W, W).

    Arguments:
        subset_indices: (K_sub,) int64 — pool indices in any order,
                        no duplicates required by us (caller's contract)
        K_pool: int — total pool size

    Returns:
        (K_sub_states,) int64 — pool slot index per subset slot, in the
        subset's state-slot order.
    """
    K_sub = subset_indices.shape[0]
    n_rr_pool = K_pool * (K_pool + 1) // 2
    n_rr_sub = K_sub * (K_sub + 1) // 2
    K_sub_states = n_rr_sub + K_sub + 1

    out = np.empty(K_sub_states, dtype=np.int64)
    slot = 0
    # Real-real (a, b) with a <= b in the subset's index space
    for a in range(K_sub):
        pi_a = subset_indices[a]
        for b in range(a, K_sub):
            pi_b = subset_indices[b]
            if pi_a <= pi_b:
                lo = pi_a
                hi = pi_b
            else:
                lo = pi_b
                hi = pi_a
            out[slot] = lo * K_pool - lo * (lo - 1) // 2 + (hi - lo)
            slot += 1
    # Real-W (k_sub, W) — each pool member's (k, W) slot
    for k_sub in range(K_sub):
        out[slot] = n_rr_pool + subset_indices[k_sub]
        slot += 1
    # (W, W) — single state, last pool slot
    out[slot] = n_rr_pool + K_pool

    return out


@njit(cache=True, parallel=True, fastmath=False)
def _viterbi_partial_binned_emissions_kernel(
        H_pool,
        rr_pair_i, rr_pair_j, rr_state_slots,
        kW_pair_k, kW_state_slots,
        WW_state_slot, compute_WW,
        snps_per_bin, n_bins, lam,
        log_probs,
        out_tensor):
    """Fill specified state-slots in `out_tensor` with their Viterbi
    binned emissions, leaving all other slots untouched.

    This is the incremental-build companion to
    `_viterbi_binned_emissions_kernel`.  The full kernel builds every
    state in the pool's collapsed layout; this one builds ONLY the
    state slots whose corresponding pair of pool members is new (or
    has changed) relative to a previous pool.  The caller (typically
    `PoolEmissionCache.__init__` with prev_cache supplied) is
    responsible for:
      - pre-allocating `out_tensor` at (N, K_states_new, n_bins),
      - filling already-reusable rows with `prev_cache._pool_tensor`'s
        corresponding rows (canonical (lo, hi) → row mapping),
      - passing the COMPLEMENT of those rows (i.e., the slots that
        still need fresh emission values) to this kernel.

    Bit-equivalence with `_viterbi_binned_emissions_kernel`:
      - rr state (i, j): identical per-l accumulator on the same
        scalars (log_probs[s, l, d] with d = H[i,l] + H[j,l]),
        summed in the same left-to-right order within each bin.
      - kW state (k): identical per-l accumulator on
        max(log_probs[s, l, d0], log_probs[s, l, d1]) - lam, summed
        in the same order.
      - WW state: identical per-l accumulator on
        max(log_probs[s, l, 0..2]) - 2*lam.

    Uses log_probs (precomputed via _log_probs_kernel) in place of
    inline math.log calls.  By monotonicity of log over positives,
    max(log_probs[..., d0..d1]) = log(max(probs[..., d0..d1], EPS))
    after the LOG_EPS clamp baked into log_probs.  Same scalars in
    the same accumulation order yield bit-identical output to the
    legacy full-build kernel.

    Arguments:
        H_pool:             (K_pool_new, L) int64 — current pool's
                             haps (full set; only rows referenced by
                             rr_pair_* / kW_pair_k are read).
        rr_pair_i:          (n_rr_miss,) int64 — first pool index of
                             each rr pair to compute (i in 0..K_pool).
        rr_pair_j:          (n_rr_miss,) int64 — second pool index of
                             each rr pair to compute, with i <= j.
        rr_state_slots:     (n_rr_miss,) int64 — destination state-
                             slot in out_tensor for each rr pair.
        kW_pair_k:          (n_kW_miss,) int64 — pool index of each
                             (k, W) state to compute.
        kW_state_slots:     (n_kW_miss,) int64 — destination state-
                             slot in out_tensor for each kW state.
        WW_state_slot:      int — destination slot for (W, W) state
                             (only used if compute_WW=True).
        compute_WW:         bool — whether to (re)compute the (W, W)
                             state.  False when the caller copied it
                             from prev_cache.
        snps_per_bin:       bin size in SNPs.
        n_bins:             total bin count.
        lam:                wildcard penalty per strand-site.
        log_probs:          (N, L, 3) float64, C-contig — precomputed
                             log(max(probs[s, l, g], LOG_EPS_LOCAL))
                             (from _log_probs_kernel).
        out_tensor:         (N, K_states_new, n_bins) float64, C-contig
                             — destination buffer.  This kernel writes
                             ONLY to the specified slots; other slots
                             must already contain correct values
                             (typically copied from prev_cache).

    Returns:
        out_tensor (the same buffer that was passed in).
    """
    N = log_probs.shape[0]
    L = log_probs.shape[1]

    n_rr_miss = rr_pair_i.shape[0]
    n_kW_miss = kW_pair_k.shape[0]

    for s in prange(N):
        # ----- rr pairs -----
        for slot_iter in range(n_rr_miss):
            i = rr_pair_i[slot_iter]
            j = rr_pair_j[slot_iter]
            out_slot = rr_state_slots[slot_iter]
            for b in range(n_bins):
                start = b * snps_per_bin
                end = start + snps_per_bin
                if end > L:
                    end = L
                acc = 0.0
                for l in range(start, end):
                    d = H_pool[i, l] + H_pool[j, l]
                    acc += log_probs[s, l, d]
                out_tensor[s, out_slot, b] = acc

        # ----- kW pairs -----
        for slot_iter in range(n_kW_miss):
            k = kW_pair_k[slot_iter]
            out_slot = kW_state_slots[slot_iter]
            for b in range(n_bins):
                start = b * snps_per_bin
                end = start + snps_per_bin
                if end > L:
                    end = L
                acc = 0.0
                for l in range(start, end):
                    d0 = H_pool[k, l]
                    d1 = d0 + 1
                    lp0 = log_probs[s, l, d0]
                    lp1 = log_probs[s, l, d1]
                    lp_max = lp0 if lp0 > lp1 else lp1
                    acc += lp_max - lam
                out_tensor[s, out_slot, b] = acc

        # ----- (W, W) state, if requested -----
        if compute_WW:
            for b in range(n_bins):
                start = b * snps_per_bin
                end = start + snps_per_bin
                if end > L:
                    end = L
                acc = 0.0
                for l in range(start, end):
                    lp0 = log_probs[s, l, 0]
                    lp1 = log_probs[s, l, 1]
                    lp2 = log_probs[s, l, 2]
                    lp_max = lp0
                    if lp1 > lp_max:
                        lp_max = lp1
                    if lp2 > lp_max:
                        lp_max = lp2
                    acc += lp_max - 2.0 * lam
                out_tensor[s, WW_state_slot, b] = acc

    return out_tensor


class PoolEmissionCache:
    """Precomputed Viterbi-emission tensor for a fixed hap pool.

    Lets the recovery loop and seed-trim evaluate per-subset Viterbi
    NLL without rebuilding the (N, K_states, n_bins) emission tensor
    on each call.  See module-header comment block (POOL EMISSION
    CACHE) for design rationale and bit-exact equivalence argument.

    Usage:
        cache = PoolEmissionCache(pool_haps, probs_k,
                                   lam=DEFAULT_LAMBDA,
                                   penalty=VITERBI_SWITCH_PENALTY,
                                   snps_per_bin=VITERBI_SNPS_PER_BIN)
        nll0 = cache.nll_for_subset([])          # K=0 baseline
        nll  = cache.nll_for_subset([0, 3, 7])   # by pool indices

    Public attributes (read-only after construction):
        pool_haps:    list of (L_kept,) int64 arrays — original pool
        K_pool:       len(pool_haps)
        N, L_kept:    probs_k.shape[0], probs_k.shape[1]
        lam, penalty, snps_per_bin, n_bins
        n_rr_pool:    K_pool*(K_pool+1)//2
    """

    def __init__(self, pool_haps, probs_k,
                  lam=None, penalty=None, snps_per_bin=None,
                  prev_cache=None):
        # Default to module-level constants (matching _viterbi_ll_per_-
        # sample's parameter semantics) when callers omit them.
        if lam is None:
            lam = DEFAULT_LAMBDA
        if penalty is None:
            penalty = VITERBI_SWITCH_PENALTY
        if snps_per_bin is None:
            snps_per_bin = VITERBI_SNPS_PER_BIN

        # Normalise pool_haps to a list (callers pass lists today; this
        # also accepts numpy 2D arrays or tuples for flexibility).
        if not isinstance(pool_haps, list):
            pool_haps = list(pool_haps)
        self.pool_haps = pool_haps
        self.K_pool = len(pool_haps)

        self.N = int(probs_k.shape[0])
        self.L_kept = int(probs_k.shape[1])
        self.lam = float(lam)
        self.penalty = float(penalty)

        # Bin sizing (same logic as _viterbi_ll_per_sample).  snps_per_-
        # bin == 1 collapses to per-site bins (no binning); otherwise
        # ceil(L / snps_per_bin) bins covering the SNPs.
        L = self.L_kept
        if snps_per_bin > 1 and snps_per_bin < L:
            n_bins = int(math.ceil(L / snps_per_bin))
        else:
            n_bins = L
            snps_per_bin = 1
        self.snps_per_bin = int(snps_per_bin)
        self.n_bins = int(n_bins)

        self.n_rr_pool = self.K_pool * (self.K_pool + 1) // 2

        # Store reference to probs_k for the incremental-build path's
        # compatibility check.  We use object identity (`is`) below to
        # decide if a `prev_cache` argument is compatible — this is
        # cheap and unambiguous when the recovery loop is invoking the
        # same probs_k object across rounds (the typical pattern).
        self._probs_k_ref = probs_k

        # K=0 NLL is independent of pool — it's just the (W, W)-only
        # path's NLL on probs_k.  Compute via _per_site_cost_W_W to
        # match the legacy _compute_nll_for_subset([], ...) value
        # exactly bit-for-bit (the alternative — summing the cache's
        # WW slot over bins — would sum the same scalars in a slightly
        # different order, risking tiny float differences).
        cost_WW_per_site = _per_site_cost_W_W(probs_k, lam)
        self._nll_K0 = float(cost_WW_per_site.sum())

        # Per-pool-hap content keys (bytes of the hap's int64 buffer).
        # Used by the incremental-build path to match haps across
        # successive caches by content (not by object identity, since
        # `_fit_at_fixed_K` produces fresh arrays each round even when
        # the bits are unchanged).  Cheap: K_pool * tobytes() ≤ ~30 µs
        # at K_pool=20, L=200.
        self._hap_keys = [h.tobytes() for h in pool_haps]

        if self.K_pool == 0:
            # Empty pool: only K=0 subsets queryable.  No tensor needed.
            self._pool_tensor = None
            self._log_probs = None
            return

        # ---------------------------------------------------------------
        # log_probs precompute (Tier 0 sharing across cache builds).
        # If a compatible prev_cache provided its log_probs (same
        # probs_k object), reuse it directly.  Otherwise compute fresh.
        # The fresh compute is ~1-2 ms at N=320 L=200, but avoiding it
        # across the 7-ish round_cache builds per block is a clean
        # marginal saving.
        # ---------------------------------------------------------------
        can_reuse_prev = (
            prev_cache is not None
            and prev_cache._pool_tensor is not None
            and prev_cache._probs_k_ref is probs_k
            and prev_cache.lam == self.lam
            and prev_cache.snps_per_bin == self.snps_per_bin
            and prev_cache.n_bins == self.n_bins
        )

        if can_reuse_prev and prev_cache._log_probs is not None:
            log_probs = prev_cache._log_probs
        else:
            probs_c = np.ascontiguousarray(probs_k, dtype=np.float64)
            log_probs = _log_probs_kernel(probs_c)
        self._log_probs = log_probs

        # ---------------------------------------------------------------
        # Build the pool emission tensor.  Two paths:
        #   (a) No prev_cache (or incompatible): full build via the
        #       partial kernel with ALL slots marked missing.
        #   (b) Compatible prev_cache: identify state slots whose pool-
        #       member pair matches by content in prev_cache (via hap
        #       keys), pre-fill those slots from prev's tensor, and
        #       compute only the truly missing rows via the partial
        #       kernel.
        #
        # Bit-equivalence with the legacy full-build path:
        #   - The partial kernel's per-slot accumulator is identical
        #     to the legacy _viterbi_binned_emissions_kernel's (same
        #     scalars from log_probs[s, l, d], same accumulation
        #     order within each bin).  See the partial kernel's
        #     docstring for the monotonicity-of-log argument used to
        #     show max(log_probs[..., d0..d1]) = log(max(probs[..., d0..d1], EPS)).
        #   - Rows COPIED from prev_cache were themselves produced by
        #     the same kernel against the same probs_k / lam / binning
        #     in the prev cache's __init__, so they are bit-identical
        #     to what a fresh recompute would produce.
        # ---------------------------------------------------------------

        K_states = self.n_rr_pool + self.K_pool + 1
        out_tensor = np.empty((self.N, K_states, self.n_bins), dtype=np.float64)

        # Decide which slots are reusable from prev_cache.
        rr_pair_i_list = []
        rr_pair_j_list = []
        rr_state_slots_list = []
        kW_pair_k_list = []
        kW_state_slots_list = []
        compute_WW = True

        if can_reuse_prev:
            # Build a {hap_key -> prev_pool_idx} map for O(1) lookup.
            prev_key_to_idx = {}
            for prev_idx, key in enumerate(prev_cache._hap_keys):
                # If duplicates exist in prev's pool (shouldn't, but
                # the cache doesn't enforce uniqueness), keep the
                # FIRST occurrence — its row is canonical.
                if key not in prev_key_to_idx:
                    prev_key_to_idx[key] = prev_idx
            K_prev = prev_cache.K_pool
            prev_tensor = prev_cache._pool_tensor

            # rr state (i, j) for i <= j in NEW pool's row-major order
            new_slot = 0
            for i in range(self.K_pool):
                key_i = self._hap_keys[i]
                prev_i = prev_key_to_idx.get(key_i, -1)
                for j in range(i, self.K_pool):
                    key_j = self._hap_keys[j]
                    prev_j = prev_key_to_idx.get(key_j, -1)
                    if prev_i >= 0 and prev_j >= 0:
                        # Both haps existed in prev — copy the row.
                        # In prev's layout, the rr slot for (a, b)
                        # with a <= b is `a * K_prev - a*(a-1)/2 + (b - a)`.
                        if prev_i <= prev_j:
                            pa, pb = prev_i, prev_j
                        else:
                            pa, pb = prev_j, prev_i
                        prev_slot = pa * K_prev - pa * (pa - 1) // 2 + (pb - pa)
                        out_tensor[:, new_slot, :] = prev_tensor[:, prev_slot, :]
                    else:
                        # At least one hap is new — compute fresh.
                        rr_pair_i_list.append(i)
                        rr_pair_j_list.append(j)
                        rr_state_slots_list.append(new_slot)
                    new_slot += 1

            # kW state (k, W) — reusable iff hap k matched in prev
            for k in range(self.K_pool):
                key_k = self._hap_keys[k]
                prev_k = prev_key_to_idx.get(key_k, -1)
                if prev_k >= 0:
                    prev_slot = prev_cache.n_rr_pool + prev_k
                    out_tensor[:, new_slot, :] = prev_tensor[:, prev_slot, :]
                else:
                    kW_pair_k_list.append(k)
                    kW_state_slots_list.append(new_slot)
                new_slot += 1

            # (W, W) state — depends only on probs_k and lam, which we
            # already verified match prev.  Always reusable.
            WW_slot_new = self.n_rr_pool + self.K_pool
            WW_slot_prev = prev_cache.n_rr_pool + prev_cache.K_pool
            out_tensor[:, WW_slot_new, :] = prev_tensor[:, WW_slot_prev, :]
            compute_WW = False
        else:
            # No reusable rows — every slot needs computing.
            new_slot = 0
            for i in range(self.K_pool):
                for j in range(i, self.K_pool):
                    rr_pair_i_list.append(i)
                    rr_pair_j_list.append(j)
                    rr_state_slots_list.append(new_slot)
                    new_slot += 1
            for k in range(self.K_pool):
                kW_pair_k_list.append(k)
                kW_state_slots_list.append(new_slot)
                new_slot += 1
            # compute_WW = True (default)

        # Build the H_pool array once for the partial kernel.
        H_pool_arr = np.stack(self.pool_haps, axis=0).astype(np.int64)
        H_c = np.ascontiguousarray(H_pool_arr, dtype=np.int64)

        rr_pair_i_arr = np.asarray(rr_pair_i_list, dtype=np.int64)
        rr_pair_j_arr = np.asarray(rr_pair_j_list, dtype=np.int64)
        rr_state_slots_arr = np.asarray(rr_state_slots_list, dtype=np.int64)
        kW_pair_k_arr = np.asarray(kW_pair_k_list, dtype=np.int64)
        kW_state_slots_arr = np.asarray(kW_state_slots_list, dtype=np.int64)
        WW_slot = self.n_rr_pool + self.K_pool

        # Run the partial kernel.  If all the *_list arrays are empty
        # AND compute_WW is False (everything reused), the kernel will
        # walk N samples doing nothing — still cheap (~10 µs for N=320),
        # so no special-cased early-return needed.
        _viterbi_partial_binned_emissions_kernel(
            H_c,
            rr_pair_i_arr, rr_pair_j_arr, rr_state_slots_arr,
            kW_pair_k_arr, kW_state_slots_arr,
            int(WW_slot), bool(compute_WW),
            int(self.snps_per_bin), int(self.n_bins), float(self.lam),
            log_probs,
            out_tensor)

        self._pool_tensor = out_tensor

    def nll_for_subset(self, subset_indices):
        """Return scalar UNCAPPED NLL for the subset given by pool indices.

        Bit-identical to:
            _compute_nll_for_subset(
                [self.pool_haps[i] for i in subset_indices],
                probs_k_used_at_construction,
                self.lam)
        but reuses the precomputed pool emission tensor.

        Arguments:
            subset_indices: sequence of ints, each an index into
                self.pool_haps.  May be empty (returns K=0 NLL).
                Order doesn't affect the result (the Viterbi best-path
                value is invariant under state-slot permutation under
                flat penalty).
        """
        K_sub = len(subset_indices)
        if K_sub == 0:
            # K=0 path: only state is (W, W); return precomputed value
            # for bit-identity with legacy _compute_nll_for_subset([], ...).
            return self._nll_K0

        if self._pool_tensor is None:
            raise ValueError(
                "PoolEmissionCache was built with empty pool; only "
                "K=0 subsets are queryable.")

        # Build pool-state-index map (njit kernel for the small inner
        # loops; the K_sub_states=O(K_sub^2) iterations are cheap but
        # called often enough that staying in compiled code matters).
        si = np.asarray(subset_indices, dtype=np.int64)
        pool_state_indices = _build_pool_state_index_map_kernel(si, self.K_pool)

        # Fancy-index the pool tensor along the state axis to assemble
        # the subset's emission tensor.  Numpy creates a copy here;
        # ascontiguousarray is a no-op if already contiguous but
        # guarantees the layout viterbi_score_selection expects.
        sub_tensor = np.ascontiguousarray(
            self._pool_tensor[:, pool_state_indices, :])

        # Run flat-penalty Viterbi on the subset tensor.  Same kernel
        # used by _viterbi_ll_per_sample.  Returns (N,) of best-path
        # LL per sample.  NLL = -sum.
        best_ll = viterbi_score_selection(sub_tensor, self.penalty)
        return -float(best_ll.sum())