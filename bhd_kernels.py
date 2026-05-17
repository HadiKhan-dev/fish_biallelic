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
VITERBI_SWITCH_PENALTY = 5.0

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
    if n_pairs == 0:
        return np.empty((probs_k.shape[0], 0, L)), []

    # For each pair, compute per-site dosage in {0, 1, 2}
    pair_dosage = np.empty((n_pairs, L), dtype=np.int64)
    for p, (i, j) in enumerate(pair_indices):
        pair_dosage[p] = H_k[i] + H_k[j]

    # Gather probs[:, l, dosage[p, l]] for each (sample, pair, site).
    # Use fancy indexing on the genotype axis.  Build index arrays:
    #   sample_idx[s, p, l] = s
    #   site_idx[s, p, l]   = l
    #   geno_idx[s, p, l]   = pair_dosage[p, l]
    N = probs_k.shape[0]
    # Broadcast pair_dosage to (N, n_pairs, L) — the "geno" axis index
    geno_idx = np.broadcast_to(pair_dosage[None, :, :], (N, n_pairs, L))
    # Site index broadcasts trivially via the gather
    # Use take_along_axis: probs_k has shape (N, L, 3); we want gather on
    # axis=2 of shape (N, L) array reshaped to (N, n_pairs, L) selection.
    # Reshape probs to (N, 1, L, 3) and broadcast geno_idx to (N, n_pairs, L, 1).
    probs_b = probs_k[:, None, :, :]                       # (N, 1, L, 3)
    geno_idx_b = geno_idx[:, :, :, None]                   # (N, n_pairs, L, 1)
    gathered = np.take_along_axis(probs_b, geno_idx_b, axis=3)   # (N, n_pairs, L, 1)
    cost = _safe_neg_log(gathered.squeeze(-1))             # (N, n_pairs, L)
    return cost, pair_indices


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
    K, L = H_k.shape
    N = probs_k.shape[0]
    # For each (k, l), wildcard's strand picks w ∈ {0, 1} to maximise
    # probs_k[s, l, H_k[k, l] + w].  The two candidate dosages for fixed
    # k, l are H_k[k, l] + 0 = H_k[k, l] and H_k[k, l] + 1.
    # Build (K, L) → (K, L, 2) of candidate dosages.
    dosage_w0 = H_k                       # (K, L)  — w=0
    dosage_w1 = H_k + 1                   # (K, L)  — w=1
    # Gather per (s, k, l) — for each sample, look up these two dosages
    # in probs_k[s, l, *]
    # probs_k shape (N, L, 3); we want for each k, l:
    #   p0[s, k, l] = probs_k[s, l, dosage_w0[k, l]]
    #   p1[s, k, l] = probs_k[s, l, dosage_w1[k, l]]
    # Use fancy indexing: build broadcasted indices.
    probs_b = probs_k[:, None, :, :]                                # (N, 1, L, 3)
    d0_b = np.broadcast_to(dosage_w0[None, :, :, None],
                            (N, K, L, 1))                            # (N, K, L, 1)
    d1_b = np.broadcast_to(dosage_w1[None, :, :, None],
                            (N, K, L, 1))                            # (N, K, L, 1)
    p0 = np.take_along_axis(probs_b, d0_b, axis=3).squeeze(-1)      # (N, K, L)
    p1 = np.take_along_axis(probs_b, d1_b, axis=3).squeeze(-1)      # (N, K, L)
    # Wildcard picks w to maximise p (i.e., minimise -log p)
    p_max = np.maximum(p0, p1)
    cost = _safe_neg_log(p_max) + lam
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
    # Best dosage at each (s, l) is just argmax over genotype.
    p_max = probs_k.max(axis=2)            # (N, L_kept)
    return _safe_neg_log(p_max) + 2.0 * lam


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

    # Real-real K^2 ordered states.  Dosage at site l for pair (i, j) is
    # H[i, l] + H[j, l] in {0, 1, 2}, used as the genotype index into
    # probs_k.  np.take_along_axis with the broadcast index tensor pulls
    # the per-(sample, pair, site) emission probability in one call.
    dosage_pairs = (H[:, None, :] + H[None, :, :]).astype(np.int64)  # (K, K, L)
    dosage_flat = dosage_pairs.reshape(K * K, L)
    probs_b = probs_k[:, None, :, :]                                   # (N, 1, L, 3)
    geno_idx_rr = np.broadcast_to(
        dosage_flat[None, :, :, None], (N, K * K, L, 1))
    p_rr = np.take_along_axis(probs_b, geno_idx_rr, axis=3).squeeze(-1)
    rr_emis = np.log(np.maximum(p_rr.astype(np.float64), LOG_EPS))    # (N, K*K, L)

    if include_wildcards:
        # 2K wildcard pairs: K of (k, W) + K of (W, k), identical emissions
        # under flat-penalty Viterbi by commutativity of dosage (see the
        # state-space description in the section header above).  The
        # wildcard strand picks its bit optimally per site to maximise the
        # emission probability, yielding max(P(H[k]+0), P(H[k]+1)).
        d_w0 = H
        d_w1 = (H + 1).astype(np.int64)
        d_w0_b = np.broadcast_to(d_w0[None, :, :, None], (N, K, L, 1))
        d_w1_b = np.broadcast_to(d_w1[None, :, :, None], (N, K, L, 1))
        p_w0 = np.take_along_axis(probs_b, d_w0_b, axis=3).squeeze(-1)
        p_w1 = np.take_along_axis(probs_b, d_w1_b, axis=3).squeeze(-1)
        p_max_w = np.maximum(p_w0, p_w1)
        rw_emis_kW = (
            np.log(np.maximum(p_max_w.astype(np.float64), LOG_EPS))
            - lam)                                                     # (N, K, L)
        rw_emis = np.concatenate([rw_emis_kW, rw_emis_kW], axis=1)     # (N, 2K, L)

        # (W, W) emission: both strands wildcard, both pick optimally,
        # max-over-genotypes (3 dosage states).
        p_max_g = probs_k.max(axis=2)
        ww_emis = (np.log(np.maximum(p_max_g.astype(np.float64), LOG_EPS))
                   - 2.0 * lam)[:, None, :]                            # (N, 1, L)

        all_emis = np.concatenate([rr_emis, rw_emis, ww_emis], axis=1)
    else:
        all_emis = rr_emis

    # Bin along the site axis (sum log-prob emissions within each bin
    # before applying the inter-bin switch penalty).
    K_states = all_emis.shape[1]
    if snps_per_bin > 1 and snps_per_bin < L:
        n_bins = int(math.ceil(L / snps_per_bin))
        ll_tensor = np.zeros((N, K_states, n_bins), dtype=np.float64)
        for b in range(n_bins):
            start = b * snps_per_bin
            end = min(start + snps_per_bin, L)
            ll_tensor[:, :, b] = all_emis[:, :, start:end].sum(axis=2)
    else:
        ll_tensor = all_emis.astype(np.float64)

    ll_tensor = np.ascontiguousarray(ll_tensor)
    # viterbi_score_selection: flat-penalty Viterbi, @njit(parallel=True)
    # over samples (block_haplotypes line 378).  Returns (N,) of best path
    # log-likelihood.
    best_ll = viterbi_score_selection(ll_tensor, float(penalty))
    return best_ll


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

def _update_A(probs_k, H_k, lam):
    """For each sample, pick the pair assignment that minimises its
    capped cost.

    Arguments:
        probs_k: (N, L_kept, 3)
        H_k:     (K, L_kept) discrete in {0, 1}
        lam:     wildcard penalty

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
    W = K   # sentinel: wildcard strand index

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
    cost_rr_per_site, pair_indices = _per_site_cost_real_real(probs_k, H_k)  # (N, n_pairs_rr, L)
    cost_rW_per_site = _per_site_cost_real_W(probs_k, H_k, lam)              # (N, K, L)
    cost_WW_per_site = _per_site_cost_W_W(probs_k, lam)                       # (N, L)

    # Sum across kept sites — UNCAPPED for pair assignment
    cost_rr_total = cost_rr_per_site.sum(axis=2)                              # (N, n_pairs_rr)
    cost_rW_total = cost_rW_per_site.sum(axis=2)                              # (N, K)
    cost_WW_total = cost_WW_per_site.sum(axis=1)                              # (N,)

    # Concatenate all candidate pairs into one cost array per sample
    # Order: [real-real pairs (n_rr), real-W pairs (K), W-W (1)]
    all_costs = np.concatenate([
        cost_rr_total,                                                  # (N, n_pairs_rr)
        cost_rW_total,                                                  # (N, K)
        cost_WW_total[:, None],                                         # (N, 1)
    ], axis=1)                                                          # (N, n_total)

    n_rr = cost_rr_total.shape[1]
    # Best pair per sample (uncapped costs)
    best_idx = all_costs.argmin(axis=1)                                 # (N,)
    per_sample_cost = all_costs[np.arange(N), best_idx]                 # (N,)
    # Uncapped is the same as the assignment cost since we used uncapped
    # to assign in the first place.  Returned for API symmetry with the
    # Fix-H-cap-in-pair-assignment design that was rejected; downstream
    # callers can treat per_sample_cost == per_sample_cost_unc.
    per_sample_cost_unc = per_sample_cost

    # Translate best_idx back to (a, b) pair representation
    A = np.empty((N, 2), dtype=np.int64)
    wildcard_slots = np.empty(N, dtype=np.int64)

    for s in range(N):
        bi = int(best_idx[s])
        if bi < n_rr:
            i, j = pair_indices[bi]
            A[s, 0] = i
            A[s, 1] = j
            wildcard_slots[s] = 0
        elif bi < n_rr + K:
            k_real = bi - n_rr
            # Pair (k_real, W); place real first, W second (canonical)
            A[s, 0] = k_real
            A[s, 1] = W
            wildcard_slots[s] = 1
        else:
            # (W, W)
            A[s, 0] = W
            A[s, 1] = W
            wildcard_slots[s] = 2

    # === VITERBI BIC OVERRIDE ===
    # When _VITERBI_BIC_ENABLED is True (production default), replace the
    # baseline per-sample cost arrays with Viterbi best-path NLL per sample.
    # The pair assignment A and wildcard_slots computed above are preserved
    # — they are consumed by _update_H's bit-voting M-step (which is NOT
    # Viterbi-path-aware) and by wildcard-mass tracking respectively.
    # See the VITERBI BIC SCORING constants block for design rationale.
    #
    # K=0 (no real founders): Viterbi state space is just (W, W) with no
    # transitions; per-sample LL = sum_l (log max_g P_s(l, g) - 2*lam).
    # The baseline path above already produced this exact value via
    # _per_site_cost_W_W, so the override is a no-op at K=0 — we skip the
    # Viterbi call to save the cost of building a state-space-of-one tensor.
    if _VITERBI_BIC_ENABLED and K > 0:
        haps_list = [H_k[k] for k in range(K)]
        ll_per_sample = _viterbi_ll_per_sample(
            haps_list, probs_k,
            penalty=VITERBI_SWITCH_PENALTY,
            snps_per_bin=VITERBI_SNPS_PER_BIN,
            lam=lam)
        # Per-sample cost = -log-likelihood (NLL convention used elsewhere
        # in this module).  per_sample_cost and per_sample_cost_unc were
        # aliased in the baseline computation (see comment above at the
        # `per_sample_cost_unc = per_sample_cost` line), and we keep them
        # aliased here to preserve that invariant for downstream callers
        # that may inspect both fields independently.
        per_sample_cost = (-ll_per_sample).astype(np.float64)
        per_sample_cost_unc = per_sample_cost

    return A, per_sample_cost, per_sample_cost_unc, wildcard_slots


# =============================================================================
# UPDATE STEP H: founder allele updates
# =============================================================================

def _update_H(probs_k, H_k, A, lam):
    """For each (founder, kept site), pick the binary value that minimises
    NLL contribution from samples carrying that founder.

    Updates H_k in-place and returns the number of bits flipped (so the
    coordinate descent loop can detect convergence).

    Arguments:
        probs_k: (N, L_kept, 3)
        H_k:     (K, L_kept) — modified in place
        A:       (N, 2)   pair assignments, with K used as the wildcard sentinel
        lam:     wildcard penalty

    Returns:
        n_changes: int — number of (founder, site) bits that flipped
    """
    K, L = H_k.shape
    N = probs_k.shape[0]
    W = K

    # We update founders in decreasing order of usage.  Compute usage from A.
    # Usage of founder k = number of A[s, *] entries equal to k.  A pair
    # (k, k) contributes 2; (k, j) with j != k contributes 1.
    usage = np.zeros(K, dtype=np.int64)
    for k in range(K):
        usage[k] = int(((A[:, 0] == k) & (A[:, 0] != W)).sum() +
                       ((A[:, 1] == k) & (A[:, 1] != W)).sum())
    update_order = np.argsort(-usage, kind='stable')

    n_changes = 0
    for k in update_order:
        n_changes += _update_one_founder(probs_k, H_k, A, int(k), lam)
    return n_changes


def _update_one_founder(probs_k, H_k, A, k, lam):
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

    n_changes = 0

    # Pre-compute cost_WW per (attributing sample, site) once.
    # cost_WW(s, l) = -log max_d P(g=d | data) + 2λ.  This is the cap.
    all_attributing_idx = np.concatenate([
        kk_sample_idx, kj_sample_idx, kW_sample_idx
    ]) if (kk_sample_idx.size + kj_sample_idx.size + kW_sample_idx.size) > 0 else np.empty(0, dtype=np.int64)
    # We compute cost_WW per-site within the loop; lookups are cheap.

    for l in range(L):
        # Current value of H_k[k, l]
        cur_val = H_k[k, l]

        # Contribution under H_k[k, l] = 0
        nll0 = 0.0
        # Contribution under H_k[k, l] = 1
        nll1 = 0.0

        # Bucket H (k, k): cost = -log probs[s, l, 2*hk]
        # Cap each sample's contribution at cost_WW(s, l).
        if kk_sample_idx.size > 0:
            p_kk = probs_k[kk_sample_idx, l, :]              # (n_kk, 3)
            cost_WW_kk = _safe_neg_log(p_kk.max(axis=1)) + 2.0 * lam   # (n_kk,)
            cost_kk_h0 = _safe_neg_log(p_kk[:, 0])           # 2*0 = 0
            cost_kk_h1 = _safe_neg_log(p_kk[:, 2])           # 2*1 = 2
            nll0 += float(np.minimum(cost_kk_h0, cost_WW_kk).sum())
            nll1 += float(np.minimum(cost_kk_h1, cost_WW_kk).sum())

        # Bucket J (k, j): cost = -log probs[s, l, hk + H_k[j, l]]
        # Cap each sample's contribution at cost_WW(s, l).
        if kj_sample_idx.size > 0:
            partner_h_at_l = H_k[partner_J, l]               # (n_J,)
            # Dosage if H_k[k, l] = 0:  partner_h_at_l + 0 = partner_h_at_l
            # Dosage if H_k[k, l] = 1:  partner_h_at_l + 1
            p_J = probs_k[kj_sample_idx, l, :]               # (n_J, 3)
            cost_WW_J = _safe_neg_log(p_J.max(axis=1)) + 2.0 * lam       # (n_J,)
            d0 = partner_h_at_l                              # (n_J,)
            d1 = partner_h_at_l + 1
            cost_J_h0 = _safe_neg_log(p_J[np.arange(p_J.shape[0]), d0])
            cost_J_h1 = _safe_neg_log(p_J[np.arange(p_J.shape[0]), d1])
            nll0 += float(np.minimum(cost_J_h0, cost_WW_J).sum())
            nll1 += float(np.minimum(cost_J_h1, cost_WW_J).sum())

        # Bucket P (k, W): cost = min_w -log probs[s, l, hk + w] + λ
        # Cap each sample's contribution at cost_WW(s, l).
        if kW_sample_idx.size > 0:
            p_P = probs_k[kW_sample_idx, l, :]               # (n_P, 3)
            cost_WW_P = _safe_neg_log(p_P.max(axis=1)) + 2.0 * lam       # (n_P,)
            # Under H_k[k, l] = 0: dosage candidates = {0, 1}; pick max prob
            best0 = np.maximum(p_P[:, 0], p_P[:, 1])
            # Under H_k[k, l] = 1: dosage candidates = {1, 2}; pick max prob
            best1 = np.maximum(p_P[:, 1], p_P[:, 2])
            cost_P_h0 = _safe_neg_log(best0) + lam
            cost_P_h1 = _safe_neg_log(best1) + lam
            nll0 += float(np.minimum(cost_P_h0, cost_WW_P).sum())
            nll1 += float(np.minimum(cost_P_h1, cost_WW_P).sum())

        # Pick lower-NLL value.  No-signal handling: if nll0 == nll1
        # (within numerical precision), no sample expressed a meaningful
        # preference at this site (e.g., all attributing samples were
        # capped out under both H values, giving zero discriminating
        # signal).  In that case keep cur_val to avoid arbitrary flips.
        if abs(nll0 - nll1) < 1e-9:
            new_val = cur_val
        else:
            new_val = 0 if nll0 < nll1 else 1
        if new_val != cur_val:
            H_k[k, l] = new_val
            n_changes += 1

    return n_changes


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

    for it in range(max_iter):
        # Update A given H
        A, per_sample_cost, per_sample_cost_unc, wildcard_slots = _update_A(probs_k, H, lam)

        # Convergence check via A
        a_changed = (A_prev is None) or (not np.array_equal(A, A_prev))
        A_prev = A.copy()

        # Update H given A
        h_changes = _update_H(probs_k, H, A, lam)

        n_iter = it + 1
        if not a_changed and h_changes == 0:
            break

    # Recompute final A and per-sample cost after the last H update
    A, per_sample_cost, per_sample_cost_unc, wildcard_slots = _update_A(probs_k, H, lam)
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