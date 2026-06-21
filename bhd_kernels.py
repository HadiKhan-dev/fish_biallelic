#%% =====================================================================
# bhd_kernels.py — Atomic computation kernel for stage-3 block haplotypes
#
# Split out of block_haplotypes.py as part of the 4-file split.
# Contains the BIC/CD primitives that every higher-level subsystem builds
# on:
#
#   - Module-level constants (MASK, DEFAULT_LAMBDA, LOG_EPS) and Viterbi
#     BIC scoring configuration (_VITERBI_BIC_ENABLED, VITERBI_SWITCH_-
#     PENALTY, VITERBI_SNPS_PER_BIN).
#   - Low-level helpers (_safe_neg_log, _decisiveness).
#   - Initialization helpers (_init_hap_from_sample_dosage,
#     _select_initial_seed).
#   - Cost computation (_per_site_cost_W_W, _ww_bin_emis_from_cost_ww,
#     _log_probs_kernel).
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
from bhd_config import (
    CLUSTER_HOM_BAND_HI,
    CLUSTER_HOM_BAND_LO,
    DEFAULT_LAMBDA,
    POOLED_ALT_HI,
    POOLED_ALT_LO,
    VITERBI_SNPS_PER_BIN,
    VITERBI_SWITCH_PENALTY,
)


HAS_NUMBA = True  # bhd_kernels hard-imports numba; flag kept for the migrated prune_chimeras fast/slow branch


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


# Numerical floor for log(probability) — prevents -inf when a sample's
# posterior is exactly zero at a particular genotype (which can happen
# at sites with no reads).
LOG_EPS = 1e-12


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
    iterations within a single `_fit_at_fixed_K` invocation, so the
    `log P(g)` values that `_update_A` (via its BLAS precompute) and
    `_update_one_founder_kernel` consume are themselves invariant.
    Computing all 3 * N * L log values ONCE per fit avoids recomputing
    them in every `_update_A` and `_update_H` call.

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
# SCORING constants block in bhd_config for rationale and
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
# SHARED LOW-READ-DEPTH CANDIDATE-GENERATION PRIMITIVES
# =============================================================================
#
# The candidate generators that feed stage-3 founder discovery — trio
# recovery and homozygous-sample recovery (bhd_trio), the k-medoids
# multistart seeding (block_haplotypes), the Bernoulli-mixture /
# kmeans++ subtraction recovery (bhd_recovery), and pairwise common-hap
# recovery (bhd_pairwise) — have historically reduced each sample's
# posteriors to a hard argmax dosage (and, in trio, an XOR/parity form)
# before clustering or comparing samples.  Hard calls are fine at the read
# depths those paths were validated on (>= ~10x), but they discard exactly
# the information that survives at low read depth:
#
#   - A zero-coverage site has a ~uniform posterior; argmax calls it dosage
#     0, which is indistinguishable from a true hom-ref call and injects
#     spurious agreement into every sample-sample comparison.
#   - A heterozygous site at low depth (one read) is read as a RANDOM
#     homozygote, so two true-het samples look like a hard 0-vs-2
#     disagreement even though their posteriors still place real mass on
#     the het genotype.
#
# These primitives operate on the full posteriors instead of hard calls, so
# the low-depth signal is retained.  They are SHARED so that every generator
# uses one validated implementation:
#
#   - soft_agreement_similarity(probs_k):
#       (N, N) expected per-site genotype-agreement matrix under the
#       posteriors.  Replaces "argmax then pairwise Hamming" as the
#       sample-sample similarity used for clustering.  It is homoscedastic
#       by construction — every pair is averaged over all L sites, so unlike
#       a masked-Hamming distance (whose denominator is the variable count
#       of jointly-informative sites) it does not assign systematically
#       noisier distances to low-overlap pairs; a uniform (zero-coverage)
#       site contributes the same constant 1/3 agreement term to every pair
#       and so needs no masking.
#   - alt_fractions(probs_k):
#       (N, L) per-(sample, site) expected alt-allele fraction E[dosage]/2.
#       Pooling this across a cluster's members (alt[members].mean(axis=0))
#       gives a signal-boosted per-site estimate that reveals het sites a
#       per-sample mode/argmax consensus would collapse: a het site of the
#       cluster's pair-type pools to ~0.5 (both alleles seen across the
#       shallow members), a hom site to ~0 or ~1.
#   - pooled_alt_to_dosage / cluster_homozygosity_score / pooled_alt_to_hap:
#       small deterministic readers of a pooled-alt vector — call the
#       dosage {0, 1, 2}, score how homozygous a cluster looks (fraction of
#       sites NOT sitting in the het band), and read a binary founder hap
#       off a (presumed homozygous) cluster.
#
# IMPORTANT: nothing in the existing code path calls these.  They are
# dormant additions; each consumer opts in behind its own switch (e.g.
# bhd_trio's TRIO_CLUSTER).  With those switches off, behaviour is
# bit-identical to before, because these functions are simply never
# invoked.  This keeps the additions safe to land ahead of the consumers.
#
# Dependency note: these use only numpy (the soft-agreement matrix is built
# with BLAS matmul; the remaining primitives are elementwise / reduction
# numpy), consistent with this module's "numpy + math +
# block_haplotypes.viterbi_score_selection only" contract.  The actual
# clustering call (HDBSCAN etc.) lives in the consumer modules, not here, so
# this module pulls in no new dependency.
#
# Thresholds are module constants (documented below) so every consumer
# applies identical cut-points; tune here to retune all consumers at once.


def soft_agreement_similarity(probs_k):
    """(N, N) expected per-site genotype-agreement matrix under posteriors.

        S[i, j] = (1 / L) * sum_l sum_g probs_k[i, l, g] * probs_k[j, l, g]

    The inner sum_g probs_k[i, l, g] * probs_k[j, l, g] is P(G_i = G_j) at
    site l when G_i and G_j are independent draws from their respective
    posteriors; averaging over the L sites gives the expected fraction of
    sites at which the two samples' genotypes agree.  Range [0, 1].

    Computed via BLAS: for each genotype channel g, the (N, N) Gram matrix
    of the (N, L) slice probs_k[:, :, g] is Pg @ Pg.T; the three Gram
    matrices are summed and the result divided by L.  This is ~6x faster
    than an explicit njit triple loop even under the single-threaded BLAS
    that thread_config.py enforces (GEMM is cache-blocked and vectorised in
    ways a plain loop is not), and the matrix build is the per-block-
    dominant cost of the clustering front-end, so the speedup matters.

    DETERMINISM NOTE: unlike this module's Viterbi / BIC kernels — which use
    explicit, fixed-order summation because they back bit-identity
    equivalence claims — the matmul's reduction order over the L axis is
    BLAS-internal and therefore NOT guaranteed bit-stable across BLAS
    libraries, versions, or thread counts.  The resulting variation is at
    the float-rounding level (~1e-15 vs an explicit left-to-right loop), and
    the only consumer of this matrix is HDBSCAN clustering, whose label
    assignments are insensitive to such perturbations.  This primitive
    therefore makes NO bit-identity claim, and callers must not rely on its
    output being reproducible to the last bit across environments.

    The matrix is symmetric (Pg @ Pg.T is symmetric for every g, and a sum
    of symmetric matrices is symmetric).  The diagonal S[i, i] =
    (1/L) sum_l sum_g probs_k[i, l, g]^2 is the sample's self-agreement
    (equal to 1 only where the posterior is a point mass).  Callers that
    need a distance typically use `S.max() - S` with a zeroed diagonal.

    Inputs:
        probs_k: (N, L, 3) float64 — per-(sample, site) genotype posteriors.
                 Made C-contiguous float64 internally (no copy when already
                 so, via `_maybe_c_contig`); each per-genotype slice is also
                 forced contiguous for BLAS.

    Returns:
        S: (N, N) float64 — symmetric expected-agreement matrix.
    """
    probs_c = _maybe_c_contig(probs_k, np.float64)
    N = probs_c.shape[0]
    L = probs_c.shape[1]
    S = np.zeros((N, N), dtype=np.float64)
    # Sum the three per-genotype Gram matrices (g = 0, 1, 2), then scale by
    # 1/L.  Each slice is forced C-contiguous so BLAS takes its fast path.
    for g in range(3):
        Pg = np.ascontiguousarray(probs_c[:, :, g])
        S += Pg @ Pg.T
    S /= float(L)
    return S


def alt_fractions(probs_k):
    """(N, L) per-(sample, site) expected alt-allele fraction E[dosage]/2.

        alt[s, l] = 0.5 * probs_k[s, l, 1] + probs_k[s, l, 2]
                  = (1 * P(g=1) + 2 * P(g=2)) / 2 = E[dosage] / 2

    Range [0, 1].  Pooling across a cluster's members
    (alt[members].mean(axis=0)) yields the signal-boosted per-site estimate
    described in the section header: a homozygous-ref pair-type pools to ~0,
    a het pair-type to ~0.5, a homozygous-alt pair-type to ~1.

    Pure vectorised numpy (an elementwise combination of two genotype
    channels, with no reduction across samples), so it is fully
    deterministic and needs no kernel.
    """
    return 0.5 * probs_k[:, :, 1] + probs_k[:, :, 2]


def pooled_alt_to_dosage(pooled_alt, lo=POOLED_ALT_LO, hi=POOLED_ALT_HI):
    """Call a per-site consensus dosage in {0, 1, 2} from a pooled-alt vector.

        pooled_alt[l] < lo  -> 0 (hom-ref)
        pooled_alt[l] > hi  -> 2 (hom-alt)
        otherwise           -> 1 (het)

    Defaults lo = POOLED_ALT_LO (0.25), hi = POOLED_ALT_HI (0.75).  Values
    landing exactly on lo or hi fall through to the het call (1); this is a
    negligible measure-zero edge for continuous pooled fractions.

    Arguments:
        pooled_alt: (L,) float64 — a cluster's per-site pooled alt fraction
                    (e.g. alt_fractions(probs_k)[members].mean(axis=0)).

    Returns:
        (L,) int64 dosage in {0, 1, 2}.
    """
    d = np.ones(pooled_alt.shape[0], dtype=np.int64)
    d[pooled_alt < lo] = 0
    d[pooled_alt > hi] = 2
    return d


def cluster_homozygosity_score(pooled_alt, band_lo=CLUSTER_HOM_BAND_LO,
                               band_hi=CLUSTER_HOM_BAND_HI):
    """Fraction of a cluster's sites that are NOT in the het band.

        score = 1 - mean_l[ band_lo < pooled_alt[l] < band_hi ]

    A genuinely homozygous cluster (pair-type (A, A), so no site is truly
    het) has almost no sites near 0.5 and scores ~1; a heterozygous pair-
    type (A, B) has about half its sites near 0.5 and scores ~0.5.  Range
    [0, 1].  Defaults band (band_lo, band_hi) = (CLUSTER_HOM_BAND_LO,
    CLUSTER_HOM_BAND_HI) = (0.35, 0.65).

    Arguments:
        pooled_alt: (L,) float64 — a cluster's per-site pooled alt fraction.

    Returns:
        float in [0, 1] — higher means the cluster looks more homozygous.
    """
    in_band = (pooled_alt > band_lo) & (pooled_alt < band_hi)
    return 1.0 - float(in_band.mean())


def pooled_alt_to_hap(pooled_alt):
    """Read a binary founder hap off a (presumed homozygous) cluster.

        bit[l] = 1 if pooled_alt[l] > 0.5 else 0

    Intended for clusters that `cluster_homozygosity_score` has flagged as
    homozygous, where pooled_alt sits cleanly near 0 or 1 so the 0.5 cut is
    unambiguous; the recovered hap is the single haplotype that both strands
    of those homozygous samples carry.  (A value of exactly 0.5 maps to 0; a
    negligible measure-zero edge for continuous pooled fractions.)

    Arguments:
        pooled_alt: (L,) float64 — the cluster's per-site pooled alt fraction.

    Returns:
        (L,) int64 founder hap in {0, 1}.
    """
    return (pooled_alt > 0.5).astype(np.int64)


# =============================================================================
# MIGRATED FROM block_haplotypes.py (legacy block-hap discovery, retired).
# Viterbi/chimera scoring kernels + prune_chimeras now live here so the
# active ecosystem (beam_search_core, chimera_resolution/scoring,
# residual_discovery, block_haplotypes) imports them from the bhd_* leaf
# instead of from the legacy module.
# =============================================================================

@njit(parallel=True, fastmath=True)
def viterbi_score_selection(ll_tensor, penalty):
    """
    Calculates the BEST Viterbi path score for each sample given a set of active pairs.
    """
    n_samples, K, n_sites = ll_tensor.shape
    best_scores = np.empty(n_samples, dtype=np.float64)
    
    for s in prange(n_samples):
        # Buffer for current scores (faster than allocation in loop)
        current_scores = np.empty(K, dtype=np.float64)
        for k in range(K):
            current_scores[k] = ll_tensor[s, k, 0]
            
        for i in range(1, n_sites):
            # 1. Find best previous score globally
            best_prev = -np.inf
            for k in range(K):
                if current_scores[k] > best_prev:
                    best_prev = current_scores[k]
            
            # The baseline score if we switch INTO a state
            switch_base = best_prev - penalty
            
            # 2. Update states
            for k in range(K):
                emission = ll_tensor[s, k, i]
                stay = current_scores[k]
                
                # Max(Stay, Switch)
                if stay > switch_base:
                    current_scores[k] = stay + emission
                else:
                    current_scores[k] = switch_base + emission
        
        # Final max
        final_max = -np.inf
        for k in range(K):
            if current_scores[k] > final_max:
                final_max = current_scores[k]
        best_scores[s] = final_max
        
    return best_scores