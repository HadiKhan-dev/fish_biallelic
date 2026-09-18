"""Founder complexity, panel objectives and binary candidate utilities."""
from __future__ import annotations


import math

import numpy as np
from numba import njit, prange
import haplotype_reconstruction.core.config as core_config

_LOG_TWO = math.log(2.0)


def canonicalize_binary_panel(haplotypes, assignments):
    """Sort binary founder rows and remap diploid assignments consistently.

    The wildcard sentinel is the input founder count and remains unchanged.
    Returned assignments have each diploid pair ordered increasingly.
    """

    haps = np.asarray(haplotypes)
    assigned = np.asarray(assignments)
    if haps.ndim != 2:
        raise ValueError("haplotypes must be a two-dimensional matrix")
    if assigned.ndim != 2 or assigned.shape[1] != 2:
        raise ValueError("assignments must have shape (samples, 2)")
    k = len(haps)
    byte_rows = np.ascontiguousarray(haps, dtype=np.int8)
    if haps.shape[1] == 0:
        order = np.arange(k, dtype=np.int64)
    else:
        row_keys = byte_rows.view(
            np.dtype((np.void, haps.shape[1]))
        ).reshape(-1)
        order = np.argsort(row_keys, kind="stable")
    inverse = np.empty(k, dtype=np.int64)
    inverse[order] = np.arange(k, dtype=np.int64)
    remapping = np.empty(k + 1, dtype=np.int64)
    remapping[:k] = inverse
    remapping[k] = k
    canonical_assignments = remapping[assigned]
    first = canonical_assignments[:, 0].copy()
    np.minimum(
        first, canonical_assignments[:, 1], out=canonical_assignments[:, 0]
    )
    np.maximum(
        first, canonical_assignments[:, 1], out=canonical_assignments[:, 1]
    )
    canonical_haplotypes = np.ascontiguousarray(haps[order])
    canonical_key = np.ascontiguousarray(byte_rows[order]).tobytes()
    return canonical_haplotypes, canonical_assignments, order, inverse, canonical_key


def compute_founder_complexity_cost(
    cc_scale,
    n_samples,
    n_sites,
    use_log_bic=False,
    n_blocks=1,
):
    """Return the outer per-founder complexity cost.

    The default linear convention is
    ``cc_scale * (n_sites / 200) * n_samples * n_blocks``.  The optional
    historical log convention is preserved exactly for a single block and
    extended linearly over explicitly combined independent blocks.
    """

    site_growth = n_sites / 200.0
    if use_log_bic:
        log_n = math.log(max(n_samples * n_sites, 2))
        return cc_scale * log_n * site_growth * n_blocks
    return cc_scale * site_growth * n_samples * n_blocks


def soft_cluster_seed_haplotypes(
    genotype_likelihoods,
    n_seeds,
    min_cluster_size=core_config.DEFAULT_SOFT_SEED_MIN_CLUSTER_SIZE,
):
    """Return up to ``n_seeds`` pooled binary haplotype seeds.

    Samples are clustered by expected-genotype agreement. Each retained
    cluster contributes one pooled-alt consensus, providing deterministic,
    denoised starting basins without using sample metadata or truth.
    """


    import hdbscan

    likelihoods = np.asarray(genotype_likelihoods, dtype=np.float64)
    if likelihoods.ndim != 3 or likelihoods.shape[2] != 3:
        raise ValueError(
            "genotype_likelihoods must have shape (samples, sites, 3)"
        )
    if n_seeds < 1:
        raise ValueError("n_seeds must be positive")
    if min_cluster_size < 2:
        raise ValueError("min_cluster_size must be at least 2")
    if likelihoods.shape[0] < int(min_cluster_size):
        return []

    similarity = soft_agreement_similarity(likelihoods)
    distance = similarity.max() - similarity
    np.fill_diagonal(distance, 0.0)
    distance = np.ascontiguousarray(distance, dtype=np.float64)

    labels = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=int(min_cluster_size),
    ).fit(distance).labels_
    clusters = [
        np.flatnonzero(labels == label)
        for label in np.unique(labels)
        if label != -1
    ]
    clusters.sort(key=lambda members: -len(members))

    alt_fraction = alt_fractions(likelihoods)
    return [
        pooled_alt_to_hap(
            alt_fraction[members].mean(axis=0)
        ).astype(np.int64)
        for members in clusters[:n_seeds]
    ]


def _k_fits_binary_universe(n_sites: int, k: int) -> bool:
    """Return whether K is no larger than 2**L without constructing 2**L."""

    bits = k.bit_length()
    if bits <= n_sites:
        return True
    if bits > n_sites + 1:
        return False
    return (k & (k - 1)) == 0 and bits - 1 == n_sites


np.seterr(divide='ignore', invalid='ignore')


def exact_unique_binary_rows(matrix):
    """Return exact distinct binary rows in NumPy lexicographic order."""

    rows = np.asarray(matrix)
    if rows.ndim != 2:
        raise ValueError("binary rows must be a two-dimensional matrix")
    if np.any((rows != 0) & (rows != 1)):
        raise ValueError("binary rows must contain only zero and one")
    if len(rows) <= 1:
        return np.array(rows, copy=True, order="C")
    packed = np.packbits(rows, axis=1, bitorder="big")
    first_index_by_key = {}
    for index, packed_row in enumerate(packed):
        first_index_by_key.setdefault(packed_row.tobytes(), index)
    indices = [first_index_by_key[key] for key in sorted(first_index_by_key)]
    return np.ascontiguousarray(rows[indices])


def compute_outer_bic_from_log_likelihood(k, log_likelihood, complexity_cost):
    """Return ``k * complexity_cost - 2 * log_likelihood`` (lower is better)."""

    return k * complexity_cost - 2.0 * log_likelihood


def log_binary_haplotype_set_count(n_sites: int, k: int) -> float:
    """Return log binomial(2**L, K) without materializing the universe.

    K is small in the supported block search. Summing K stable log-ratio
    terms avoids constructing either 2**L or the combinatorial integer.
    """

    if (
        isinstance(n_sites, bool)
        or int(n_sites) != n_sites
        or int(n_sites) < 1
    ):
        raise ValueError("n_sites must be a positive integer")
    if isinstance(k, bool) or int(k) != k or int(k) < 1:
        raise ValueError("k must be a positive integer")
    n_sites = int(n_sites)
    k = int(k)
    if not _k_fits_binary_universe(n_sites, k):
        raise ValueError("k cannot exceed the number of binary haplotypes")

    universe_log = n_sites * _LOG_TWO
    terms = []
    for index in range(k):
        fraction = math.ldexp(float(index), -n_sites)
        terms.append(
            universe_log
            + math.log1p(-fraction)
            - math.log(index + 1)
        )
    result = math.fsum(terms)
    if result < 0.0 and result > -1e-12:
        return 0.0
    if result < 0.0 or not math.isfinite(result):
        raise ArithmeticError("invalid enumerative haplotype-set code length")
    return result


MASK = -1


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
    dosage = probs[sample_idx].argmax(axis=1)  # (L,) ∈ {0, 1, 2}

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
        probs_kept = probs[:, kept_mask,:]
    else:
        probs_kept = probs
    decisiveness = _decisiveness(probs_kept)
    return int(decisiveness.argmax())


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
                acc -= cost_WW[s, l]  # = log p_max - 2*lam
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
    that core/parallel.py enforces (GEMM is cache-blocked and vectorised in
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
        Pg = np.ascontiguousarray(probs_c[:,:, g])
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
    return 0.5 * probs_k[:,:, 1] + probs_k[:,:, 2]


def pooled_alt_to_hap(pooled_alt):
    """Read a binary founder hap off a (presumed homozygous) cluster.

        bit[l] = 1 if pooled_alt[l] > 0.5 else 0

    Pooled values above 0.5 map to allele 1; all others map to allele 0.
    Cluster pooling denoises shallow per-sample observations before this call.

    Arguments:
        pooled_alt: (L,) float64 — the cluster's per-site pooled alt fraction.

    Returns:
        (L,) int64 founder hap in {0, 1}.
    """
    return (pooled_alt > 0.5).astype(np.int64)


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
