#%% =====================================================================
# bhd_trio.py — Trio recovery (posterior soft-clustering + group-trio
#               algorithm)
#
# Split out of block_haplotypes_discrete.py as part of the 4-file split.
# Contains the trio-recovery subsystem: recovers founder haplotypes for
# blocks where standard K-growth fails because every sample is a 1+1
# dosage blend of two true founders.
#
# Recovery clusters samples on the genotype posteriors (a low-read-depth
# front-end; see _soft_unified_recovery and the shared primitives in
# bhd_kernels.py), then splits clusters into a homozygous read-off route
# and a heterozygous group-triangle route.  The group-triangle algebra
# (the XOR-based recovery described in the long docstring below) is
# retained and applied to the heterozygous clusters; it depends only on
# numpy + numba.  The clustering call (HDBSCAN) and the soft-similarity /
# pooled-alt primitives are imported lazily from bhd_kernels.
#
# Public entry point: _trio_recovery_candidate_haps(probs_k, ...)
# Called by block_haplotypes_discrete._grow_K_with_recovery and
# bhd_recovery's residual rescue (see TRIO_RECOVERY_ENABLED below for the
# master switch).
#
# HISTORY: the original sample-clustering front-end hard-called dosages and
# clustered on XOR-Hamming distance (helpers _cluster_samples_by_xor,
# _estimate_inter_xor_distance, _compute_group_consensus_dosages, with the
# threshold TRIO_CLUSTER_FRACTION * D).  That front-end was replaced by the
# posterior-based soft clustering above; those helpers are preserved in
# version control.
#
# NUMBA OPTIMIZATION: the retained group-triangle kernels are numba-
# accelerated with @njit, preallocate fixed-size buffers (no Python
# list-of-array growth), and use order-preserving merge semantics:
#   _find_grouped_trios        — 8.6 ms -> ~1 ms (8x) at G=16
#   _consensus_recovery_blind  — 7.8 ms -> ~1 ms (8x) at typical pool
# Output equivalence:  these kernels produce byte-identical results to the
# Python fallback paths (validated end-to-end against the pre-numba
# version as part of integration validation).
# See the long docstring below the constants for algorithm details and
# validation results.
# =======================================================================

import warnings

import numpy as np

# Shared dynamic-thread reallocation: re-checked at the entry of trio recovery
# (a major Stage-3 phase) so a straggler block grows into cores freed as peers
# finish.  thread_config is a leaf config module (no bhd_* imports) -> no
# import cycle; the helper no-ops on the sequential path.
import thread_config

# Defensive numba import matching the project convention (see
# analysis_utils.py, block_haplotypes.py).  If numba is unavailable,
# all @njit decorators become no-ops and the Python fallback paths
# inside the wrapper functions run.
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    warnings.warn(
        "Numba not found.  bhd_trio will fall back to pure-Python paths "
        "(typically 2-10x slower per trio call).",
        ImportWarning,
    )
    # Dummy decorator that accepts arguments (like cache=True) but does
    # nothing — same pattern as analysis_utils.py.
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        # Support both @njit and @njit(cache=True) forms
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return decorator


# =============================================================================
# TRIO RECOVERY (XOR-based group-trio algorithm for all-hets blocks)
# =============================================================================
#
# Motivation: standard K-growth fails on all-hets blocks (no homozygous
# samples) because the seeding step picks a heterozygous sample's strand,
# which is a blend of two true founders rather than any single founder.
# The K-growth then settles on a wrong basin and grows K to compensate.
#
# Algorithmic idea (from user proposal, verified empirically): if sample
# s_ab has dosage d_ab = h_a + h_b (with h_a, h_b binary), then the
# transformation X(s) = d_ab mod 2 (equivalently "replace 2 with 0")
# yields exactly h_a XOR h_b.  Three samples covering pair-types
# (a,b), (a,c), (b,c) form a "triangle" with the structural property
#     X(s_ab) XOR X(s_ac) = h_b XOR h_c = X(s_bc)
# i.e., the predicted XOR of a third pair-type can be computed from any
# two.  A matching third sample lets us recover all three founders
# algebraically: (d_ab + d_ac + d_bc) / 2 = h_a + h_b + h_c, then
# subtract any sample to get the founder it doesn't carry.
#
# Scaling: naive O(N^3) is intractable.  Samples of the same pair-type are
# grouped by clustering (see below); triangles are then enumerated at the
# group level (G <= K(K-1)/2 groups, where K is the number of founders),
# and per-cluster consensus dosages denoise the algebraic recovery.  Total
# cost: O(N * K^2 * L).
#
# Clustering front-end and thresholds (see _soft_unified_recovery /
# bhd_kernels):
#   - Samples are clustered on the expected-genotype-agreement similarity
#     (bhd_kernels.soft_agreement_similarity) via HDBSCAN, keeping the
#     posteriors rather than hard-calling them so the low-read-depth signal
#     is preserved.
#   - Each cluster is classified by its pooled alt-allele fraction
#     (bhd_kernels.cluster_homozygosity_score): a homozygous-looking cluster
#     yields a founder directly (bhd_kernels.pooled_alt_to_hap); a
#     heterozygous cluster supplies a consensus dosage
#     (bhd_kernels.pooled_alt_to_dosage) to the triangle algebra below.
#   - Triangle thresholds are derived from the median pairwise group-
#     centroid Hamming (centroids = consensus dosage % 2), the denoised
#     analogue of the per-sample distance the original front-end estimated:
#       Match threshold = TRIO_MATCH_FRACTION * centroid_median (= 0.4 * it)
#       Distinct check  = TRIO_DISTINCT_FRACTION * centroid_median (0.5 * it);
#       rejects trios where any two of the three group centroids are within
#       the distinct threshold of each other (Option B).
#
# Validation (test_xor_trio_noise.py + test_xor_trio_grouped.py):
#   - K=3 N=120 across noise grid (depth 5/10/20, mutation 0/2%):
#     6/6 conditions, 3/3 truths recovered at 0% Hamming.
#   - K=6 N=320 production scale, same grid: 6/6 conditions,
#     6/6 truths at 0% Hamming.
#   - Closely-related founders (diversity 2.5%, 5%, 10%, 25%):
#     all 4 levels pass at K=3.  K=6 + diversity 2.5% degrades to
#     4/6 (chain-merging at boundary) — graceful failure.
#   - Scaling: 30ms at N=120, 110ms at N=2560 (sub-linear per sample).
#   - Hom-mixed data: hom samples cluster at all-zeros centroid and
#     are correctly excluded from triangle enumeration via Option B.
#
# Integration: _grow_K_with_recovery calls _trio_recovery_candidate_haps
# after initial K-growth, fits at fixed K with the resulting candidates,
# and replaces the current H if and only if BIC(trio) < BIC(current).
# This is purely additive — trio recovery can only IMPROVE the result,
# never make it worse.
# -----------------------------------------------------------------------------

# Master switch — set to False to bypass trio recovery entirely
TRIO_RECOVERY_ENABLED = True


# Threshold fractions for the group-triangle algebra, applied as multipliers
# of the median pairwise group-centroid Hamming distance (see
# _soft_unified_recovery — the denoised analogue of the per-sample distance
# the recovery historically estimated).
TRIO_MATCH_FRACTION = 0.4
TRIO_DISTINCT_FRACTION = 0.5

# Minimum block size requirements — below these, trio scheme is skipped
# (returns empty) and the caller falls through to the existing pipeline.
TRIO_MIN_SAMPLES = 9                # need at least 3 samples per pair-type
                                    # times 3 pair-types in a triangle
TRIO_MIN_SITES = 50                 # too few sites makes XOR matching
                                    # unreliable (statistical insufficiency)

# Recovered-haplotype clustering parameters (production blind recovery,
# no ground-truth comparison).  Each group-trio yields 3 candidate
# haplotypes; across all trios this gives ~G(G-1)(G-2)/2 candidates
# (canonical g1<g2<g3 enumeration in _find_grouped_trios_kernel), and
# the same true founder appears in many of them.  We cluster by Hamming
# similarity and emit per-cluster majority-vote consensus.
TRIO_HAP_DEDUP_PCT = 1.0            # Hamming pct below which two haps
                                    # are considered the same founder.
                                    # Lowered from 2.0 to 1.0 after the
                                    # sweep across 55 failure + 200
                                    # healthy regression blocks showed
                                    # +26 net recoveries (27 fixes,
                                    # 1 break) at 1.0%.  Tighter dedup
                                    # avoids over-merging near-but-
                                    # distinct founders (e.g. closely-
                                    # related founder pairs at the
                                    # block-specific Hamming level)
                                    # that the 2.0% threshold was
                                    # collapsing into a single cluster.
TRIO_MIN_HAP_CLUSTER_SIZE = 1       # drop hap clusters with fewer than
                                    # this many supporting candidates
                                    # (likely noise rather than real
                                    # founder).  HISTORY: prior to the
                                    # canonical-enumeration fix in
                                    # _find_grouped_trios_kernel, each
                                    # unordered triangle was enumerated
                                    # 3 times under different (g1,g2,g3)
                                    # role assignments, so each
                                    # underlying triangle contributed
                                    # 9 pool haps that clustered into
                                    # groups of size 3 per founder.
                                    # A threshold of 3 was therefore a
                                    # no-op (no cluster could be
                                    # smaller).  After the fix, each
                                    # underlying triangle contributes
                                    # exactly 3 pool haps (one per
                                    # founder), so cluster size now
                                    # equals the number of supporting
                                    # triangles.  Default lowered from
                                    # 3 to 1 to preserve the prior
                                    # "no filter" effective behavior;
                                    # set to 2+ to require multiple
                                    # supporting triangles per founder.


# -----------------------------------------------------------------------------
# Soft-clustering recovery (posterior-based, low-read-depth)
# -----------------------------------------------------------------------------
# _trio_recovery_candidate_haps recovers founder haplotypes by clustering
# samples on the genotype posteriors rather than hard-called dosages, then
# splitting clusters into a homozygous read-off route and a heterozygous
# group-triangle route.  The full pipeline lives in _soft_unified_recovery at
# the bottom of this file; in outline:
#
#   - Cluster samples on the expected-genotype-agreement similarity
#     (bhd_kernels.soft_agreement_similarity) via HDBSCAN, then classify each
#     cluster by its pooled alt-allele fraction
#     (bhd_kernels.cluster_homozygosity_score):
#       - homozygous-looking clusters (score >= TRIO_SOFT_HOM_SCORE): the
#         founder is read off directly from the pooled-alt vector
#         (bhd_kernels.pooled_alt_to_hap).  This is robust to the lowest read
#         depths because a homozygous sample carries no strand-sampling noise
#         (both chromosomes read the same allele even at 1x).
#       - heterozygous-looking clusters: a per-site consensus dosage
#         (bhd_kernels.pooled_alt_to_dosage) feeds the group-triangle algebra
#         (_find_grouped_trios / _consensus_recovery_blind), with
#         match/distinct thresholds derived from the clean group-centroid
#         Hamming distances.
#   Rationale and the homozygous/heterozygous decomposition are documented in
#   the shared primitives section of bhd_kernels.py.
#
# The hdbscan and bhd_kernels imports are performed lazily inside
# _soft_unified_recovery, keeping this module's import surface (numpy +
# warnings) minimal at load time.

# HDBSCAN minimum cluster size — the minimum number
# of samples sharing a pair-type for that pair-type to be recovered as a
# cluster.  Mirrors the ">= 3 samples per pair-type" requirement behind
# TRIO_MIN_SAMPLES.  Other HDBSCAN parameters use the library defaults
# (min_samples = min_cluster_size, cluster_selection_method = "eom",
# alpha = 1.0, allow_single_cluster = False), matching block_haplotypes.py's
# precomputed-metric usage; expose them here if finer control is needed.
TRIO_SOFT_MIN_CLUSTER_SIZE = 3

# Cluster-homozygosity score (see bhd_kernels.cluster_homozygosity_score) at
# or above which a "soft"-mode cluster is treated as homozygous and its
# founder read off directly rather than routed to the triangle algebra.
# 0.90 = at most ~10% of the cluster's sites may sit in the het band before
# the cluster is treated as a heterozygous pair-type.
TRIO_SOFT_HOM_SCORE = 0.90


@njit(cache=True)
def _find_grouped_trios_kernel(centroids, match_thresh_bits,
                                  distinct_thresh_bits):
    """Numba kernel for _find_grouped_trios.

    Enumerates valid (g1, g2, g3, d_pred) trios.  Uses two passes: an
    upper-bound counting pass, then an emit pass into a preallocated
    array.  This avoids any list growth inside the numba kernel.

    Algorithm identical to the Python version — same loop order, same
    pairwise comparison logic — so output is byte-identical.
    """
    G, L = centroids.shape

    # Pass 1: count valid trios so we can allocate exact output size
    count = 0
    for g1 in range(G):
        for g2 in range(g1 + 1, G):
            d12 = 0
            for l in range(L):
                if centroids[g1, l] != centroids[g2, l]:
                    d12 += 1
            if d12 <= distinct_thresh_bits:
                continue
            # g3 > g2 enforces canonical ordering g1 < g2 < g3 — each
            # unordered triangle enumerated exactly once.  Previously
            # g3 ranged over all G groups (excluding g1, g2), which
            # enumerated every unordered triangle 3 times under
            # different (g1, g2, g3) role assignments.  The algebra
            # is symmetric under permutation of (g1, g2, g3) — XOR
            # is commutative for the match check, pairwise Hamming is
            # symmetric for the distinct check, and h_sum/slot output
            # values are independent of role assignment — so all 3
            # orderings produced identical candidate haplotype sets
            # (just permuted across the 3 slot positions).  Each
            # underlying triangle therefore contributed 9 candidate
            # haps to the recovery pool that were really only 3
            # unique haps repeated 3x.  After this canonicalization,
            # each unordered triangle contributes exactly 3 candidate
            # haps (one per founder), so recovered-hap cluster sizes
            # now equal the number of supporting underlying triangles
            # (not 3x that count).  NOTE: TRIO_MIN_HAP_CLUSTER_SIZE
            # semantics change accordingly — see its comment.
            for g3 in range(g2 + 1, G):
                d_pred = 0
                for l in range(L):
                    pred_l = centroids[g1, l] ^ centroids[g2, l]
                    if centroids[g3, l] != pred_l:
                        d_pred += 1
                if d_pred > match_thresh_bits:
                    continue
                d_g3_g1 = 0
                for l in range(L):
                    if centroids[g3, l] != centroids[g1, l]:
                        d_g3_g1 += 1
                if d_g3_g1 <= distinct_thresh_bits:
                    continue
                d_g3_g2 = 0
                for l in range(L):
                    if centroids[g3, l] != centroids[g2, l]:
                        d_g3_g2 += 1
                if d_g3_g2 <= distinct_thresh_bits:
                    continue
                count += 1

    # Pass 2: allocate and emit
    out = np.empty((count, 4), dtype=np.int64)
    idx = 0
    for g1 in range(G):
        for g2 in range(g1 + 1, G):
            d12 = 0
            for l in range(L):
                if centroids[g1, l] != centroids[g2, l]:
                    d12 += 1
            if d12 <= distinct_thresh_bits:
                continue
            # Canonical g3 > g2 — see Pass 1 commentary above for the
            # rationale (3-fold enumeration of identical triangles
            # eliminated; each unordered triangle enumerated once).
            for g3 in range(g2 + 1, G):
                d_pred = 0
                for l in range(L):
                    pred_l = centroids[g1, l] ^ centroids[g2, l]
                    if centroids[g3, l] != pred_l:
                        d_pred += 1
                if d_pred > match_thresh_bits:
                    continue
                d_g3_g1 = 0
                for l in range(L):
                    if centroids[g3, l] != centroids[g1, l]:
                        d_g3_g1 += 1
                if d_g3_g1 <= distinct_thresh_bits:
                    continue
                d_g3_g2 = 0
                for l in range(L):
                    if centroids[g3, l] != centroids[g2, l]:
                        d_g3_g2 += 1
                if d_g3_g2 <= distinct_thresh_bits:
                    continue
                out[idx, 0] = g1
                out[idx, 1] = g2
                out[idx, 2] = g3
                out[idx, 3] = d_pred
                idx += 1
    return out


def _find_grouped_trios(centroids, match_thresh_bits, distinct_thresh_bits):
    """Enumerate group-level triangles.  Returns list of (g1, g2, g3, d)
    tuples where:
      - (g1, g2, g3) are group indices forming a valid triangle in the
        sense that X(g1) XOR X(g2) is within match_thresh_bits Hamming
        of X(g3).
      - All three pairwise group-centroid Hammings exceed
        distinct_thresh_bits (Option B — kills same-pair-type queries
        and ensures three structurally-distinct pair-types).

    Cost: O(G^3 * L) where G = centroids.shape[0] is the number of
    groups (typically <= K(K-1)/2 <= 45 for K <= 10).

    Implementation: delegates to a numba kernel that returns a flat
    (n_trios, 4) array; we repackage into the legacy list-of-tuples
    shape that _consensus_recovery_blind expects.
    """
    G = centroids.shape[0]
    if G < 3:
        return []
    centroids_arr = np.ascontiguousarray(centroids, dtype=np.int64)
    out_arr = _find_grouped_trios_kernel(
        centroids_arr,
        int(match_thresh_bits),
        int(distinct_thresh_bits),
    )
    # Convert to list of 4-tuples matching the legacy Python return type
    return [(int(out_arr[i, 0]), int(out_arr[i, 1]),
              int(out_arr[i, 2]), int(out_arr[i, 3]))
            for i in range(out_arr.shape[0])]


@njit(cache=True)
def _consensus_recovery_blind_kernel(trios_arr, group_dosages,
                                        threshold_bits, min_cluster_size):
    """Numba kernel for _consensus_recovery_blind.

    Args:
        trios_arr: (n_trios, 4) np.int64 — columns (g1, g2, g3, d_pred).
            Only the first three are used; d_pred is informational.
        group_dosages: (G, L) np.int64 — per-group modal dosages
        threshold_bits: int — hap_dedup_pct fraction of L
        min_cluster_size: int — minimum cluster size to emit

    Returns:
        final_haps_buf: (C_max, L) np.int64 — preallocated buffer
        n_final: int — number of valid final haps in buf[:n_final]
        n_pool: int — total candidate haps generated (for diagnostics)

    Algorithm identical to the Python version: for each trio (g1,g2,g3),
    emit 3 candidate haps via h_sum = (S1+S2+S3)//2; cluster the pool
    by Hamming threshold; emit per-cluster majority-vote consensus
    for clusters with >= min_cluster_size members.

    Output ordering matches the Python version because the clustering
    loop visits haps in pool-build order, and we emit clusters in
    creation order.
    """
    n_trios = trios_arr.shape[0]
    L = group_dosages.shape[1]
    pool_size = 3 * n_trios

    # Build pool of recovered haplotypes
    haps_pool = np.empty((pool_size, L), dtype=np.int64)
    for t in range(n_trios):
        g1 = trios_arr[t, 0]
        g2 = trios_arr[t, 1]
        g3 = trios_arr[t, 2]
        for l in range(L):
            s1 = group_dosages[g1, l]
            s2 = group_dosages[g2, l]
            s3 = group_dosages[g3, l]
            h_sum = (s1 + s2 + s3) // 2
            # np.clip(x, 0, 1)
            v_h3 = h_sum - s3
            if v_h3 < 0:
                v_h3 = 0
            elif v_h3 > 1:
                v_h3 = 1
            v_h2 = h_sum - s2
            if v_h2 < 0:
                v_h2 = 0
            elif v_h2 > 1:
                v_h2 = 1
            v_h1 = h_sum - s1
            if v_h1 < 0:
                v_h1 = 0
            elif v_h1 > 1:
                v_h1 = 1
            haps_pool[3 * t + 0, l] = v_h3
            haps_pool[3 * t + 1, l] = v_h2
            haps_pool[3 * t + 2, l] = v_h1

    # Online clustering with preallocated buffers (no merge-pass —
    # this version of the algorithm only does append-or-create).
    C_max = pool_size  # worst case: every hap is its own cluster
    centroids = np.zeros((C_max, L), dtype=np.int64)
    bit_votes = np.zeros((C_max, L), dtype=np.int64)
    n_members = np.zeros(C_max, dtype=np.int64)
    # Per-cluster member indices (into haps_pool) — stored ragged,
    # capacity per cluster is pool_size in the worst case.
    cluster_members = np.zeros((C_max, pool_size), dtype=np.int64)

    n_clusters = 0
    for hi in range(pool_size):
        best_c = -1
        best_d = L + 1
        for c in range(n_clusters):
            d = 0
            for l in range(L):
                if haps_pool[hi, l] != centroids[c, l]:
                    d += 1
            if d < best_d:
                best_d = d
                best_c = c
        if best_c >= 0 and best_d <= threshold_bits:
            # Add to existing cluster best_c
            cluster_members[best_c, n_members[best_c]] = hi
            n_members[best_c] += 1
            for l in range(L):
                bit_votes[best_c, l] += haps_pool[hi, l]
            nm = n_members[best_c]
            for l in range(L):
                if 2 * bit_votes[best_c, l] > nm:
                    centroids[best_c, l] = 1
                else:
                    centroids[best_c, l] = 0
        else:
            # Start a new cluster
            c = n_clusters
            cluster_members[c, 0] = hi
            n_members[c] = 1
            for l in range(L):
                centroids[c, l] = haps_pool[hi, l]
                bit_votes[c, l] = haps_pool[hi, l]
            n_clusters += 1

    # Emit per-cluster consensus for clusters meeting min size.
    # Consensus is computed from the cluster's haps_pool members
    # (not from the running centroid — matches the Python version's
    # `cluster_haps.sum(axis=0) > n_members / 2.0` semantics).
    final_haps_buf = np.zeros((n_clusters, L), dtype=np.int64)
    n_final = 0
    for c in range(n_clusters):
        if n_members[c] < min_cluster_size:
            continue
        nm = n_members[c]
        for l in range(L):
            votes = 0
            for k in range(nm):
                hi = cluster_members[c, k]
                votes += haps_pool[hi, l]
            if 2 * votes > nm:
                final_haps_buf[n_final, l] = 1
            else:
                final_haps_buf[n_final, l] = 0
        n_final += 1
    return final_haps_buf, n_final, pool_size


def _consensus_recovery_blind(trios, group_dosages,
                                hap_dedup_pct=TRIO_HAP_DEDUP_PCT,
                                min_cluster_size=TRIO_MIN_HAP_CLUSTER_SIZE):
    """Production version of trio-based haplotype recovery — no ground
    truth.  Each group-trio (g1, g2, g3) gives 3 candidate haplotypes
    via the algebraic recovery:
        h_sum = (S1 + S2 + S3) // 2  where Si = group_dosages[gi]
        position 0 of recovered: h_sum - S3  (= h1 if g3 = h2+h3)
        position 1 of recovered: h_sum - S2  (= h2 if g2 = h1+h3)
        position 2 of recovered: h_sum - S1  (= h3 if g1 = h1+h2)
    Across all trios this gives a pool of ~3 * len(trios) candidate
    haps, with each true founder appearing many times.

    We cluster the candidate pool by Hamming similarity (threshold
    hap_dedup_pct of L), drop clusters with fewer than min_cluster_size
    members (likely noise rather than real founder), and emit per-
    cluster per-site majority-vote consensus.

    Returns: (G_unique, L) array of unique candidate founders.  May be
    empty if no trios were found or all clusters were dropped.

    Implementation: delegates to a numba kernel.  The trios list-of-
    tuples is repackaged into a flat (n_trios, 4) int64 array for the
    kernel; the kernel returns a preallocated (n_clusters_active, L)
    array plus an active count, which we slice to the tight output.
    """
    if len(trios) == 0:
        L = group_dosages.shape[1] if group_dosages.shape[0] > 0 else 0
        return np.zeros((0, L), dtype=np.int64)
    L = group_dosages.shape[1]

    # Repackage trios into a flat 2D int64 array for the numba kernel.
    # trios is a list of (g1, g2, g3, d_pred) tuples.
    trios_arr = np.empty((len(trios), 4), dtype=np.int64)
    for i, (g1, g2, g3, d_pred) in enumerate(trios):
        trios_arr[i, 0] = g1
        trios_arr[i, 1] = g2
        trios_arr[i, 2] = g3
        trios_arr[i, 3] = d_pred

    group_dosages_arr = np.ascontiguousarray(group_dosages, dtype=np.int64)
    threshold_bits = int(hap_dedup_pct / 100.0 * L)

    final_buf, n_final, _ = _consensus_recovery_blind_kernel(
        trios_arr,
        group_dosages_arr,
        int(threshold_bits),
        int(min_cluster_size),
    )

    if n_final == 0:
        return np.zeros((0, L), dtype=np.int64)
    # Tight-slice the active prefix.  Note: we copy to ensure the
    # returned array doesn't share memory with the kernel's larger
    # buffer (defensive — the kernel buffer is sized C_max = n_clusters
    # but the caller might mutate the returned array).
    return final_buf[:n_final].copy()


def _trio_recovery_candidate_haps(probs_k,
                                    match_fraction=TRIO_MATCH_FRACTION,
                                    distinct_fraction=TRIO_DISTINCT_FRACTION,
                                    hap_dedup_pct=TRIO_HAP_DEDUP_PCT,
                                    min_hap_cluster_size=TRIO_MIN_HAP_CLUSTER_SIZE,
                                    verbose=False):
    """Top-level trio-recovery entry point.

    Recovers candidate founder haplotypes from a (N, L, 3) probs_k array
    (already in kept-site space) via the posterior-based soft-clustering
    pipeline implemented in _soft_unified_recovery: cluster samples on the
    expected-genotype-agreement similarity, read founders off homozygous-
    looking clusters directly, and recover the rest from the group-triangle
    algebra over heterozygous clusters.

    Returns: (G_unique, L) np.int64 array of candidate haplotypes in kept-
    site space.  May be empty (shape (0, L)) when N < TRIO_MIN_SAMPLES,
    L < TRIO_MIN_SITES, clustering yields no usable clusters, or neither
    recovery route produces a candidate.

    Designed to compose with _fit_at_fixed_K — the output is a valid H_init
    array.  This is a thin entry point preserved for its callers
    (block_haplotypes_discrete, bhd_recovery's residual rescue); the recovery
    logic lives in _soft_unified_recovery.
    """
    return _soft_unified_recovery(
        probs_k,
        match_fraction=match_fraction,
        distinct_fraction=distinct_fraction,
        hap_dedup_pct=hap_dedup_pct,
        min_hap_cluster_size=min_hap_cluster_size,
        verbose=verbose)

# =============================================================================
# SHARED CANDIDATE-HAP DEDUP KERNEL
# =============================================================================
# Online clustering + per-cluster majority-vote consensus that collapses a
# pool of candidate founder haps to unique founders.  Used by
# _soft_unified_recovery's final de-duplication and externally by bhd_recovery
# (as bhd_trio._cluster_haps_consensus_kernel).

@njit(cache=True)
def _cluster_haps_consensus_kernel(haps_pool, threshold_bits,
                                       min_cluster_size):
    """Online clustering + per-cluster majority-vote consensus on a flat
    pool of candidate haplotypes.

    Args:
        haps_pool: (n_pool, L) np.int64 — input candidate haps
        threshold_bits: int — Hamming threshold for "same founder"
        min_cluster_size: int — minimum cluster size to emit

    Returns:
        final_haps_buf: (n_clusters, L) np.int64 — preallocated buffer
            (only the first n_final rows are valid)
        n_final: int — number of valid final haps in buf[:n_final]

    Algorithm: identical to the clustering step of _consensus_recovery_
    blind_kernel (single-pass online assignment with running per-bit
    majority centroid, then per-cluster majority-vote consensus
    computed from members' raw pool entries — NOT from the running
    centroid, to match the Python `cluster_haps.sum(axis=0) > nm/2.0`
    semantics).  Kept as a separate function to avoid refactoring the
    trio kernel; clustering semantics are intentionally byte-identical
    so trio + hom candidate sets cluster compatibly when concatenated.
    """
    n_pool = haps_pool.shape[0]
    L = haps_pool.shape[1]

    # n_pool == 0 should be handled by the caller — we still need >= 1
    # buffer entry to satisfy numba.  Use C_max = max(n_pool, 1).
    if n_pool < 1:
        C_max = 1
    else:
        C_max = n_pool

    centroids = np.zeros((C_max, L), dtype=np.int64)
    bit_votes = np.zeros((C_max, L), dtype=np.int64)
    n_members = np.zeros(C_max, dtype=np.int64)
    # Per-cluster member indices — ragged, capacity per cluster is
    # n_pool in the worst case (all pool haps in one cluster).
    cluster_members = np.zeros((C_max, C_max), dtype=np.int64)

    n_clusters = 0
    for hi in range(n_pool):
        # Find nearest existing cluster
        best_c = -1
        best_d = L + 1
        for c in range(n_clusters):
            d = 0
            for l in range(L):
                if haps_pool[hi, l] != centroids[c, l]:
                    d += 1
            if d < best_d:
                best_d = d
                best_c = c
        if best_c >= 0 and best_d <= threshold_bits:
            # Add to existing cluster best_c
            cluster_members[best_c, n_members[best_c]] = hi
            n_members[best_c] += 1
            for l in range(L):
                bit_votes[best_c, l] += haps_pool[hi, l]
            nm = n_members[best_c]
            for l in range(L):
                # > nm/2 (matches Python: bit_votes > n_members / 2.0)
                if 2 * bit_votes[best_c, l] > nm:
                    centroids[best_c, l] = 1
                else:
                    centroids[best_c, l] = 0
        else:
            # Start a new cluster
            c = n_clusters
            cluster_members[c, 0] = hi
            n_members[c] = 1
            for l in range(L):
                centroids[c, l] = haps_pool[hi, l]
                bit_votes[c, l] = haps_pool[hi, l]
            n_clusters += 1

    # Emit per-cluster consensus for clusters meeting min_cluster_size.
    # Consensus computed from raw pool members (not the running
    # centroid) to match the Python semantics exactly.
    if n_clusters < 1:
        # Edge case: empty pool — return an empty buffer
        return np.zeros((1, L), dtype=np.int64), 0
    final_haps_buf = np.zeros((n_clusters, L), dtype=np.int64)
    n_final = 0
    for c in range(n_clusters):
        if n_members[c] < min_cluster_size:
            continue
        nm = n_members[c]
        for l in range(L):
            votes = 0
            for k in range(nm):
                hi = cluster_members[c, k]
                votes += haps_pool[hi, l]
            if 2 * votes > nm:
                final_haps_buf[n_final, l] = 1
            else:
                final_haps_buf[n_final, l] = 0
        n_final += 1
    return final_haps_buf, n_final


# =============================================================================
# SOFT-CLUSTERING UNIFIED RECOVERY
# =============================================================================
# The recovery body called by _trio_recovery_candidate_haps.
#
# This unifies two posterior-based recovery routes behind ONE clustering of
# the samples:
#
#   soft-agreement similarity  ->  HDBSCAN clusters  ->  classify each cluster
#       homozygous cluster  ->  read founder off the pooled-alt vector
#       heterozygous cluster ->  consensus dosage  ->  group-triangle algebra
#
# Homozygous clusters are recovered directly because a homozygous sample's
# reads carry no strand-sampling ambiguity even at 1x, so the pooled-alt
# vector of such a cluster is a clean 0/1 founder readout; heterozygous
# pair-types still require the triangle algebra to separate the two founders
# that make them up.  Both routes' outputs are concatenated and de-duplicated
# so the same founder recovered by both routes collapses to one hap.

def _soft_unified_recovery(probs_k,
                            match_fraction=TRIO_MATCH_FRACTION,
                            distinct_fraction=TRIO_DISTINCT_FRACTION,
                            hap_dedup_pct=TRIO_HAP_DEDUP_PCT,
                            min_hap_cluster_size=TRIO_MIN_HAP_CLUSTER_SIZE,
                            min_cluster_size=TRIO_SOFT_MIN_CLUSTER_SIZE,
                            hom_score_threshold=TRIO_SOFT_HOM_SCORE,
                            verbose=False):
    """Posterior-based unified recovery (the body of _trio_recovery_candidate_haps).

    Returns (G_unique, L) int64 candidate haplotypes — the candidate-hap
    contract used throughout stage-3 recovery.  Pipeline:
      1. Soft-agreement similarity (bhd_kernels.soft_agreement_similarity)
         -> distance (S.max() - S, with a zeroed diagonal).
      2. HDBSCAN(metric="precomputed", min_cluster_size) clustering; noise
         points (label -1) are dropped.
      3. Per-cluster pooled alt-allele fraction (mean of
         bhd_kernels.alt_fractions over the cluster's members).
      4. Classify each cluster by bhd_kernels.cluster_homozygosity_score:
           - score >= hom_score_threshold: read the founder off directly
             via bhd_kernels.pooled_alt_to_hap (homozygous cluster).
           - else: route to the group-triangle algebra as a heterozygous
             pair-type, using bhd_kernels.pooled_alt_to_dosage as the
             cluster's consensus dosage.
      5. If >= 3 heterozygous clusters: derive match/distinct thresholds
         from the median pairwise group-centroid Hamming (centroids =
         consensus dosage % 2) and run _find_grouped_trios /
         _consensus_recovery_blind exactly as the "xor" path does.
      6. Concatenate homozygous read-offs and heterozygous triangle
         recoveries, then de-duplicate near-identical founders via
         _cluster_haps_consensus_kernel at hap_dedup_pct with
         min_cluster_size = 1 (a PURE de-duplication: support filtering
         already happened inside _consensus_recovery_blind for the het haps
         and via HDBSCAN's min_cluster_size for the hom clusters, so we must
         not drop singleton hom founders here).

    Returns empty (shape (0, L)) when N/L are below the trio minimums, when
    clustering yields no usable clusters, or when neither route produces a
    candidate.

    hdbscan and bhd_kernels are imported lazily here so that the default
    "xor" mode leaves this module's import surface (numpy + warnings)
    unchanged.  Matches block_haplotypes.py's standalone-hdbscan usage.
    """
    import bhd_kernels as _bk
    import hdbscan as _hdbscan

    N = probs_k.shape[0]
    L = probs_k.shape[1]

    if N < TRIO_MIN_SAMPLES or L < TRIO_MIN_SITES:
        if verbose:
            print(f'[trio/soft] Skipping: N={N} (min {TRIO_MIN_SAMPLES}) '
                  f'or L={L} (min {TRIO_MIN_SITES})')
        return np.zeros((0, L), dtype=np.int64)

    # Re-check thread allocation at the start of trio recovery (a major
    # phase: soft-clustering + group-triangle algebra).
    thread_config.apply_dynamic_threads()

    # Step 1: soft-agreement similarity -> precomputed distance.  The
    # distance is S.max() - S (so identical samples are closest) with the
    # diagonal forced to exactly 0 as HDBSCAN's precomputed metric expects.
    S = _bk.soft_agreement_similarity(probs_k)
    dist = S.max() - S
    np.fill_diagonal(dist, 0.0)
    dist = np.ascontiguousarray(dist, dtype=np.float64)

    # Step 2: HDBSCAN on the precomputed distance.  metric="precomputed"
    # and min_cluster_size mirror block_haplotypes.py's usage; the other
    # HDBSCAN parameters use library defaults (see TRIO_SOFT_MIN_CLUSTER_SIZE).
    clusterer = _hdbscan.HDBSCAN(metric="precomputed",
                                  min_cluster_size=int(min_cluster_size))
    labels = np.asarray(clusterer.fit(dist).labels_)
    cluster_ids = [c for c in np.unique(labels) if c != -1]
    if verbose:
        n_noise = int((labels == -1).sum())
        print(f'[trio/soft] HDBSCAN: {len(cluster_ids)} clusters, '
              f'{n_noise} noise points '
              f'(min_cluster_size={min_cluster_size})')
    if len(cluster_ids) == 0:
        return np.zeros((0, L), dtype=np.int64)

    # Steps 3-4: per-cluster pooled-alt fraction; classify homozygous vs
    # heterozygous and collect each route's inputs.
    alt = _bk.alt_fractions(probs_k)                       # (N, L)
    hom_haps = []                                          # list of (L,) int64
    het_group_dosages = []                                 # list of (L,) int64
    for c in cluster_ids:
        mem = np.where(labels == c)[0]
        pooled = alt[mem].mean(axis=0)                     # (L,)
        score = _bk.cluster_homozygosity_score(pooled)
        if score >= hom_score_threshold:
            hom_haps.append(_bk.pooled_alt_to_hap(pooled))
        else:
            het_group_dosages.append(_bk.pooled_alt_to_dosage(pooled))
    if verbose:
        print(f'[trio/soft] {len(hom_haps)} homozygous-looking clusters, '
              f'{len(het_group_dosages)} heterozygous-looking clusters')

    # Step 5: heterozygous group-triangle algebra (needs >= 3 het clusters
    # for a triangle to exist, mirroring the "xor" path's len(sizes) < 3
    # guard).
    het_haps = np.zeros((0, L), dtype=np.int64)
    if len(het_group_dosages) >= 3:
        group_dosages = np.stack(het_group_dosages, axis=0).astype(np.int64)
        centroids = (group_dosages % 2).astype(np.int64)
        # Match/distinct thresholds from the clean group-centroid Hamming
        # distribution.  The soft path has no per-sample D estimate; the
        # pairwise group-centroid distances are the denoised analogue, and
        # we apply the same match_fraction / distinct_fraction the "xor"
        # path applies to D.  Vectorised over the upper triangle (i < j)
        # via the exact integer identity for binary centroids C:
        #     H[i, j] = rowsum[i] + rowsum[j] - 2 * (C @ C.T)[i, j]
        # (because c != c' <=> c + c' - 2 c c' == 1 for c, c' in {0, 1}).
        # This is bit-identical to the previous
        #     for i: for j: (centroids[i] != centroids[j]).sum()
        # double-loop median (same set of integer distances, same
        # np.median), while replacing the O(G_het^2) Python loop with a
        # single GEMM.  G_het <= ~K(K-1)/2.
        G_het = centroids.shape[0]
        rowsum = centroids.sum(axis=1)
        hamming = (rowsum[:, None] + rowsum[None, :]
                   - 2 * (centroids @ centroids.T))
        cd_vals = hamming[np.triu_indices(G_het, k=1)]
        cd_med = float(np.median(cd_vals)) if cd_vals.size else 0.0
        match_thresh = max(1, int(match_fraction * cd_med))
        distinct_thresh = max(1, int(distinct_fraction * cd_med))
        trios = _find_grouped_trios(
            centroids,
            match_thresh_bits=match_thresh,
            distinct_thresh_bits=distinct_thresh)
        if verbose:
            print(f'[trio/soft] centroid-Hamming median={cd_med:.0f}, '
                  f'Mthr={match_thresh}, Dthr={distinct_thresh}, '
                  f'{len(trios)} triangles')
        if trios:
            het_haps = _consensus_recovery_blind(
                trios, group_dosages,
                hap_dedup_pct=hap_dedup_pct,
                min_cluster_size=min_hap_cluster_size)

    # Step 6: combine homozygous read-offs + heterozygous triangle
    # recoveries, then de-duplicate.
    pool_list = list(hom_haps)
    for k in range(het_haps.shape[0]):
        pool_list.append(het_haps[k])
    if len(pool_list) == 0:
        return np.zeros((0, L), dtype=np.int64)

    pool = np.stack(pool_list, axis=0).astype(np.int64)
    # Pure de-duplication (min_cluster_size = 1): collapse near-identical
    # founders recovered by both routes WITHOUT dropping singletons (a
    # homozygous founder may legitimately be supported by a single hom
    # cluster).  Uses the same per-cluster-consensus dedup kernel and the
    # same TRIO_HAP_DEDUP_PCT threshold the "xor" path's two recovery
    # routines use.
    threshold_bits = int(hap_dedup_pct / 100.0 * L)
    final_buf, n_final = _cluster_haps_consensus_kernel(
        np.ascontiguousarray(pool, dtype=np.int64),
        int(threshold_bits),
        1)
    if verbose:
        print(f'[trio/soft] {pool.shape[0]} candidates -> {n_final} unique '
              f'founders after dedup '
              f'({hap_dedup_pct}% = {threshold_bits} bits)')
    if n_final == 0:
        return np.zeros((0, L), dtype=np.int64)
    return final_buf[:n_final].copy()