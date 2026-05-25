#%% =====================================================================
# bhd_trio.py — Trio recovery (XOR-based group-trio algorithm)
#
# Split out of block_haplotypes_discrete.py as part of the 4-file split.
# Contains the trio-recovery subsystem: an XOR-based group-trio algorithm
# for all-hets blocks where standard K-growth fails because every sample
# is a 1+1 dosage blend of two true founders.
#
# Trio recovery is genuinely self-contained — operates purely on dosages
# and XOR Hamming distances, doesn't touch the BIC/CD machinery in
# bhd_kernels.  Only depends on numpy + (optionally) numba.
#
# Public entry point: _trio_recovery_candidate_haps(probs_k, ...)
# Called by block_haplotypes_discrete._grow_K_with_recovery and
# bhd_recovery._late_low_carrier_rescue (see TRIO_RECOVERY_ENABLED
# below for the master switch).
#
# NUMBA OPTIMIZATION (added after 4-file split + chr8 production
# validation):  The three hot functions are numba-accelerated with @njit
# kernels:
#   _cluster_samples_by_xor    — 31 ms -> ~3 ms (10x) at N=320, L=200
#   _find_grouped_trios        — 8.6 ms -> ~1 ms (8x) at G=16
#   _consensus_recovery_blind  — 7.8 ms -> ~1 ms (8x) at typical pool
# All three now preallocate fixed-size buffers (no Python list-of-array
# growth), use order-preserving merge semantics (memmove-style shifts
# rather than swap-with-last), and dispatch through small wrapper funcs
# that fall back to slow Python paths if numba isn't installed.
#
# Output equivalence:  numba paths produce byte-identical
# _trio_recovery_candidate_haps output to the Python fallback paths.
# Validated end-to-end against the pre-numba version (see _bhd_trio_
# numba_equivalence_test, run as part of integration validation).
#
# SECOND-PASS ADDITIONS (small support functions):
#   _compute_group_consensus_dosages — 0.44 ms -> 0.20 ms (2.2x), via
#                                       scalar-loop kernel that takes a
#                                       flattened (total_members,) int64
#                                       index array plus (G+1,) offsets,
#                                       avoiding the per-group (G x L)
#                                       bool arrays (one per value v in
#                                       {0, 1, 2}).  Strict-greater
#                                       tiebreak (smaller v wins ties)
#                                       preserved exactly.
#   _estimate_inter_xor_distance     — 0.73 ms -> 0.24 ms (3x), via
#                                       scalar-loop kernel for the
#                                       pairwise Hamming computation
#                                       (which allocates a (n_target, L)
#                                       bool intermediate in the
#                                       original).  RNG calls remain in
#                                       the Python wrapper to preserve
#                                       the deterministic seed=42
#                                       semantics byte-identically;
#                                       np.median / np.percentile also
#                                       remain in numpy (trivial cost
#                                       on the 1D diffs array).
# See the long docstring below the constants for algorithm details
# and validation results.
# =======================================================================

import warnings

import numpy as np

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
# Scaling: naive O(N^3) is intractable.  Same-pair-type samples have
# nearly identical XOR forms (differ only by per-sample noise), so we
# cluster samples by XOR Hamming distance to get one group per pair-
# type, then enumerate triangles at the group level (G <= K(K-1)/2
# groups, where K is the number of founders).  Within-group consensus
# dosages denoise the algebraic recovery.  Total cost: O(N * K^2 * L).
#
# Threshold strategy (Options A + B from validation):
#   - Estimate D = median pairwise sample-XOR Hamming.  For all-hets
#     data this is bimodal with the larger cluster at the inter-pair-
#     type distance.
#   - Cluster threshold = TRIO_CLUSTER_FRACTION * D (= 0.5 * D)
#   - Match threshold   = TRIO_MATCH_FRACTION   * D (= 0.4 * D)
#   - Distinct check    = TRIO_DISTINCT_FRACTION * D (= 0.5 * D);
#     rejects trios where any two of the three group centroids are
#     within the distinct threshold of each other.
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


# Number of random sample pairs sampled to estimate D = median pairwise
# XOR Hamming.  At N=320 there are ~50000 unordered pairs; sampling 1000
# is plenty for a stable median estimate.
TRIO_D_ESTIMATE_N_SAMPLES = 1000
TRIO_D_ESTIMATE_SEED = 42

# Threshold fractions (multipliers of D, the estimated inter-pair-type
# Hamming distance).  See validation results in module docstring.
TRIO_CLUSTER_FRACTION = 0.5
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
TRIO_HAP_DEDUP_PCT = 2.0            # Hamming pct below which two haps
                                    # are considered the same founder
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


@njit(cache=True)
def _pairwise_hamming_kernel(xor_forms, s1_arr, s2_arr):
    """Numba kernel for the pairwise Hamming distance computation
    inside _estimate_inter_xor_distance.

    Computes diff[k] = sum(xor_forms[s1_arr[k]] != xor_forms[s2_arr[k]])
    for each k.  Replaces the original's
        np.sum(xor_forms[s1_arr] != xor_forms[s2_arr], axis=1)
    which allocates a (n_target, L) bool intermediate (~5MB at typical
    n_target=500, L=200).

    Args:
        xor_forms: (N, L) int64 — sample XOR forms
        s1_arr, s2_arr: (n_target,) int64 — sample index arrays

    Returns:
        diffs: (n_target,) int64 — per-pair Hamming distances
    """
    n_target = s1_arr.shape[0]
    L = xor_forms.shape[1]
    diffs = np.empty(n_target, dtype=np.int64)
    for k in range(n_target):
        s1 = s1_arr[k]
        s2 = s2_arr[k]
        d = 0
        for l in range(L):
            if xor_forms[s1, l] != xor_forms[s2, l]:
                d += 1
        diffs[k] = d
    return diffs


def _estimate_inter_xor_distance(xor_forms,
                                   n_samples=TRIO_D_ESTIMATE_N_SAMPLES,
                                   seed=TRIO_D_ESTIMATE_SEED):
    """Estimate D = median pairwise Hamming distance between sample XOR
    forms.  For balanced all-hets data with K >= 3 founders, the
    distribution of pairwise sample XOR Hammings is bimodal:
      - same-pair-type pairs (~1/C(K,2) of all pairs): Hamming ~ noise
      - different-pair-type pairs (rest): Hamming ~ D, the inter-pair-
        type distance

    The median over enough random pairs falls in the larger cluster
    (D), giving a robust estimate independent of how close the founders
    are to each other.

    Returns: (median, p25, p75) — median is the D estimate, p25/p75
    are quartiles useful for diagnostic reporting.

    Implementation: RNG calls stay in the Python wrapper (preserving
    the deterministic seed=TRIO_D_ESTIMATE_SEED semantics exactly), the
    pairwise Hamming inner loop is delegated to a numba kernel that
    eliminates the (n_target, L) bool array allocation.  Median /
    percentile computation stays in numpy — they're trivial on a 1D
    array of length ~n_target.
    """
    rng = np.random.default_rng(seed)
    N = xor_forms.shape[0]
    if N < 2:
        return 0.0, 0.0, 0.0
    # Sample n_samples random distinct pairs.  Generate 2 * n_samples
    # raw indices then dedupe by s1 != s2 to get enough distinct pairs.
    n_target = min(n_samples, N * (N - 1) // 2)
    s1_arr = rng.integers(0, N, size=2 * n_target)
    s2_arr = rng.integers(0, N, size=2 * n_target)
    valid = s1_arr != s2_arr
    s1_arr = s1_arr[valid][:n_target]
    s2_arr = s2_arr[valid][:n_target]
    # Defensive: ensure kernel receives contiguous int64 arrays
    xor_arr = np.ascontiguousarray(xor_forms, dtype=np.int64)
    s1_arr_i = np.ascontiguousarray(s1_arr, dtype=np.int64)
    s2_arr_i = np.ascontiguousarray(s2_arr, dtype=np.int64)
    diffs = _pairwise_hamming_kernel(xor_arr, s1_arr_i, s2_arr_i)
    return (float(np.median(diffs)),
            float(np.percentile(diffs, 25)),
            float(np.percentile(diffs, 75)))


@njit(cache=True)
def _cluster_samples_by_xor_kernel(xor_forms, threshold_bits):
    """Numba kernel for _cluster_samples_by_xor.

    Operates on a preallocated (G_max, L) centroid array with
    G_max = N (worst case: every sample its own group).  Bookkeeping
    via fixed-size scratch arrays: n_members (G_max,), bit_votes (G_max,
    L), members_flat (N,) with members_start/end (G_max,) offsets.

    Returns:
        centroids_out: (n_groups_out, L) np.int64 — active prefix of
            the (G_max, L) centroids buffer, copied to a tight array
        members_flat_out: (N,) np.int64 — sample indices arranged
            contiguously by group
        members_offsets_out: (n_groups_out + 1,) np.int64 — group g
            members live at members_flat_out[offsets[g]:offsets[g+1]]
        sizes_out: (n_groups_out,) np.int64 — per-group sizes
            (equivalent to np.diff(offsets))

    Order-preservation: merges use memmove-style shifts (not swap-with-
    last), so the final group order matches what the pure-Python
    implementation produces.  This is required because the downstream
    _find_grouped_trios and online clustering in _consensus_recovery_-
    blind are order-sensitive.
    """
    N, L = xor_forms.shape
    G_max = N  # worst case: every sample is its own group

    # Centroid buffer + bookkeeping
    centroids = np.zeros((G_max, L), dtype=np.int64)
    bit_votes = np.zeros((G_max, L), dtype=np.int64)
    n_members = np.zeros(G_max, dtype=np.int64)
    # Per-group member lists, stored compactly:
    #   members_buf[group_offsets[g] : group_offsets[g] + n_members[g]]
    # We use a worst-case linear buffer per group: any group could hold
    # up to N samples.  To keep the kernel simple we use a (G_max, N)
    # ragged buffer with n_members[g] giving the active length.
    members_buf = np.zeros((G_max, N), dtype=np.int64)

    n_groups = 0

    # Pass 1: streaming assignment
    for s in range(N):
        # Find nearest centroid (linear scan)
        best_g = -1
        best_d = L + 1
        for g in range(n_groups):
            d = 0
            for l in range(L):
                if xor_forms[s, l] != centroids[g, l]:
                    d += 1
            if d < best_d:
                best_d = d
                best_g = g
        if best_g >= 0 and best_d <= threshold_bits:
            # Merge sample s into group best_g
            members_buf[best_g, n_members[best_g]] = s
            n_members[best_g] += 1
            # Update bit_votes and recompute centroid via per-bit
            # majority: centroid[l] = 1 iff bit_votes[l] > nm/2
            for l in range(L):
                bit_votes[best_g, l] += xor_forms[s, l]
            nm = n_members[best_g]
            for l in range(L):
                # > nm/2 (matches Python: bit_votes > n_members / 2.0)
                if 2 * bit_votes[best_g, l] > nm:
                    centroids[best_g, l] = 1
                else:
                    centroids[best_g, l] = 0
        else:
            # Start a new group at index n_groups
            g = n_groups
            members_buf[g, 0] = s
            n_members[g] = 1
            for l in range(L):
                centroids[g, l] = xor_forms[s, l]
                bit_votes[g, l] = xor_forms[s, l]
            n_groups += 1

    # Pass 2: merge-pass — same scan order as Python (i < j, find first
    # mergeable j after i, pop j, restart).  Uses memmove-style shift
    # for pop so group indices > j shift down by one, matching list.pop().
    while True:
        merged = False
        if n_groups < 2:
            break
        for i in range(n_groups):
            inner_break = False
            for j in range(i + 1, n_groups):
                d = 0
                for l in range(L):
                    if centroids[i, l] != centroids[j, l]:
                        d += 1
                if d <= threshold_bits:
                    # Merge group j INTO group i
                    nm_i = n_members[i]
                    nm_j = n_members[j]
                    # Append j's members to i's
                    for k in range(nm_j):
                        members_buf[i, nm_i + k] = members_buf[j, k]
                    n_members[i] = nm_i + nm_j
                    # Update bit_votes and recompute i's centroid
                    for l in range(L):
                        bit_votes[i, l] += bit_votes[j, l]
                    nm_new = n_members[i]
                    for l in range(L):
                        if 2 * bit_votes[i, l] > nm_new:
                            centroids[i, l] = 1
                        else:
                            centroids[i, l] = 0
                    # Shift groups j+1..n_groups-1 down by one (memmove)
                    for k in range(j, n_groups - 1):
                        n_members[k] = n_members[k + 1]
                        for l in range(L):
                            centroids[k, l] = centroids[k + 1, l]
                            bit_votes[k, l] = bit_votes[k + 1, l]
                        # members_buf row: copy active members of k+1 into k
                        nm_src = n_members[k]  # this is now the new k's count
                        for m in range(nm_src):
                            members_buf[k, m] = members_buf[k + 1, m]
                    n_groups -= 1
                    merged = True
                    inner_break = True
                    break
            if inner_break:
                break
        if not merged:
            break

    # Build the tight output arrays from the active prefix
    centroids_out = np.empty((n_groups, L), dtype=np.int64)
    sizes_out = np.empty(n_groups, dtype=np.int64)
    for g in range(n_groups):
        sizes_out[g] = n_members[g]
        for l in range(L):
            centroids_out[g, l] = centroids[g, l]

    # Flatten members into a 1-D array with offsets
    offsets_out = np.zeros(n_groups + 1, dtype=np.int64)
    for g in range(n_groups):
        offsets_out[g + 1] = offsets_out[g] + n_members[g]
    members_flat_out = np.empty(offsets_out[n_groups], dtype=np.int64)
    for g in range(n_groups):
        base = offsets_out[g]
        nm = n_members[g]
        for k in range(nm):
            members_flat_out[base + k] = members_buf[g, k]
    return centroids_out, members_flat_out, offsets_out, sizes_out


def _cluster_samples_by_xor(xor_forms, threshold_bits):
    """Online streaming cluster of samples by XOR Hamming distance.

    For each sample, find nearest existing centroid; if within
    threshold_bits, merge in (update centroid via per-bit majority);
    else start a new group.  Centroids are updated incrementally via
    per-bit '1'-vote counts, so adding a sample is O(L).

    Followed by a merge-pass: any two centroids within threshold_bits
    get merged.  This catches order-dependent splits where early-bad-
    luck samples create a group that should have stayed merged.

    Returns:
        centroids: (G, L) np.int64 array — per-bit majority of each group
        members:   list of length G — list of sample indices per group
        sizes:     list of length G — per-group member count

    Implementation note: delegates to a numba kernel for the inner
    loops.  The kernel returns flat arrays (members_flat + offsets);
    we repackage into the legacy list-of-lists output shape that
    downstream callers expect.  Output is byte-identical to the
    pre-numba implementation."""
    N, L = xor_forms.shape

    # Edge case: empty input.  Kernel handles N=0 correctly, but be
    # explicit because the shape (0, L) zero-row return path is what
    # downstream callers depend on.
    if N == 0:
        return (np.zeros((0, L), dtype=np.int64), [], [])

    # xor_forms is already np.int64 (built upstream as (dosage % 2)
    # cast to int64).  Defensive: ensure dtype matches kernel signature.
    xor_forms_arr = np.ascontiguousarray(xor_forms, dtype=np.int64)

    centroids_out, members_flat, offsets, sizes = \
        _cluster_samples_by_xor_kernel(xor_forms_arr, int(threshold_bits))

    # Repackage members_flat + offsets into the legacy list-of-lists.
    # Each group g's member indices are members_flat[offsets[g]:offsets[g+1]].
    G = centroids_out.shape[0]
    members = []
    for g in range(G):
        members.append(members_flat[offsets[g]:offsets[g + 1]].tolist())

    if G == 0:
        return (np.zeros((0, L), dtype=np.int64), [], [])
    return (centroids_out, members, sizes.tolist())


@njit(cache=True)
def _compute_group_consensus_dosages_kernel(dosage, member_indices, group_offsets):
    """Numba kernel for _compute_group_consensus_dosages.

    Per-site modal dosage in {0, 1, 2} per group, computed via the
    same vectorised tiebreak as the original:
      - Start with count of value 0; best_val = 0
      - For each v in (1, 2): if count(v) > best_count, update best.
    Strict-greater means ties go to the SMALLER value (0 before 1
    before 2).  This matches the original's `cnt > best_count` exactly.

    Args:
        dosage: (N, L) int64 — full dosage matrix
        member_indices: (total_members,) int64 — concatenated member
            indices across all groups (flattened ragged list)
        group_offsets: (G+1,) int64 — group_offsets[g] is the start
            index of group g in member_indices; group_offsets[g+1] is
            the (exclusive) end.  So group g has
            member_indices[group_offsets[g]:group_offsets[g+1]].

    Returns:
        consensus: (G, L) int64 — per-group consensus dosages.
            Groups with zero members get a row of all-zeros (matching
            the original's `if len(mem_list) == 0: continue` which
            leaves the np.zeros-initialized row in place).
    """
    G = group_offsets.shape[0] - 1
    L = dosage.shape[1]
    consensus = np.zeros((G, L), dtype=np.int64)
    for g in range(G):
        start = group_offsets[g]
        end = group_offsets[g + 1]
        if start == end:
            # Empty group — leave zero row, matching original semantics
            continue
        for l in range(L):
            # Count occurrences of 0, 1, 2 among this group's members
            c0 = 0
            c1 = 0
            c2 = 0
            for k in range(start, end):
                v = dosage[member_indices[k], l]
                if v == 0:
                    c0 += 1
                elif v == 1:
                    c1 += 1
                elif v == 2:
                    c2 += 1
                # Other values (shouldn't occur in normal data, but
                # defensive) are ignored — same as the original which
                # only counts {0, 1, 2}.
            # Tiebreak: strict-greater means smaller v wins ties.
            # Initial: best_val=0, best_count=c0.  Then check v=1, v=2.
            best_val = 0
            best_count = c0
            if c1 > best_count:
                best_count = c1
                best_val = 1
            if c2 > best_count:
                best_val = 2
            consensus[g, l] = best_val
    return consensus


def _compute_group_consensus_dosages(dosage, members):
    """For each group, compute per-site modal dosage in {0, 1, 2}
    across its members.  This denoises by majority vote; with ~21
    members per group at K=6 N=320, even per-site dosage error rates
    of 10% become near-zero after consensus.

    Implementation: flattens the ragged `members` list-of-lists into a
    single (total_members,) array plus a (G+1,) offsets array, then
    delegates to a numba kernel that uses scalar loops over groups
    and sites.  Eliminates the per-group numpy temporaries (`(member_dosage == v).sum(axis=0)`
    allocates a (group_size, L) bool plus a (L,) int per v).  Per-call
    is ~0.45 ms; cumulative per-block ~1.5-3 ms across the 1-3 trio
    invocations.

    Tiebreak preserved: strict-greater comparison means the smaller
    value (0 before 1 before 2) wins ties.
    """
    G = len(members)
    if G == 0:
        return np.zeros((0, dosage.shape[1]), dtype=np.int64)
    # Flatten ragged members list into a 1D int64 array plus offsets.
    # group_offsets[g] is the start of group g; group_offsets[G] is total.
    group_sizes = np.array([len(m) for m in members], dtype=np.int64)
    group_offsets = np.empty(G + 1, dtype=np.int64)
    group_offsets[0] = 0
    for g in range(G):
        group_offsets[g + 1] = group_offsets[g] + group_sizes[g]
    total_members = int(group_offsets[G])
    member_indices = np.empty(total_members, dtype=np.int64)
    for g, mem_list in enumerate(members):
        start = group_offsets[g]
        for j, idx in enumerate(mem_list):
            member_indices[start + j] = int(idx)

    dosage_arr = np.ascontiguousarray(dosage, dtype=np.int64)
    return _compute_group_consensus_dosages_kernel(
        dosage_arr, member_indices, group_offsets)


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
                                    cluster_fraction=TRIO_CLUSTER_FRACTION,
                                    match_fraction=TRIO_MATCH_FRACTION,
                                    distinct_fraction=TRIO_DISTINCT_FRACTION,
                                    hap_dedup_pct=TRIO_HAP_DEDUP_PCT,
                                    min_hap_cluster_size=TRIO_MIN_HAP_CLUSTER_SIZE,
                                    d_estimate_n_samples=TRIO_D_ESTIMATE_N_SAMPLES,
                                    d_estimate_seed=TRIO_D_ESTIMATE_SEED,
                                    verbose=False):
    """Top-level trio-recovery entry point.

    Given a (N, L, 3) probs_k array (already in kept-site space), runs
    the full trio-recovery pipeline:
      1. Argmax dosages and compute XOR forms (replace 2 with 0)
      2. Estimate D = median pairwise sample-XOR Hamming
      3. Cluster samples by XOR Hamming (threshold = cluster_fraction*D)
      4. Compute within-group consensus dosages (per-site mode)
      5. Enumerate group triangles (Option A match + Option B distinct)
      6. Algebraically recover candidate haps from each trio, cluster
         the pool by Hamming, emit per-cluster consensus

    Returns: (G_unique, L) np.int64 array of candidate haplotypes in
    kept-site space.  May be empty (shape (0, L)) if:
      - probs_k has fewer than TRIO_MIN_SAMPLES samples
      - probs_k has fewer than TRIO_MIN_SITES sites
      - clustering produces fewer than 3 groups (no triangle possible)
      - no triangles match within thresholds
      - all recovered-hap clusters fall below min_hap_cluster_size

    Designed to compose with _fit_at_fixed_K — the output is a valid
    H_init array.
    """
    N = probs_k.shape[0]
    L = probs_k.shape[1]

    if N < TRIO_MIN_SAMPLES or L < TRIO_MIN_SITES:
        if verbose:
            print(f'[trio] Skipping: N={N} (min {TRIO_MIN_SAMPLES}) '
                  f'or L={L} (min {TRIO_MIN_SITES})')
        return np.zeros((0, L), dtype=np.int64)

    # Step 1: argmax dosage and XOR transform
    dosage = np.argmax(probs_k, axis=2).astype(np.int64)  # (N, L)
    xor_forms = (dosage % 2).astype(np.int64)             # (N, L)

    # Step 2: estimate D
    d_med, d_p25, d_p75 = _estimate_inter_xor_distance(
        xor_forms, n_samples=d_estimate_n_samples, seed=d_estimate_seed)
    if d_med < 2:
        # All XOR forms are essentially identical — no discriminative
        # signal (all-hom or near-monomorphic block).  Trio scheme
        # cannot distinguish founders.
        if verbose:
            print(f'[trio] Skipping: D estimate too small ({d_med:.1f} bits)')
        return np.zeros((0, L), dtype=np.int64)

    cluster_thresh = max(1, int(cluster_fraction * d_med))
    match_thresh = max(1, int(match_fraction * d_med))
    distinct_thresh = max(1, int(distinct_fraction * d_med))

    # Step 3: cluster samples
    centroids, members, sizes = _cluster_samples_by_xor(
        xor_forms, threshold_bits=cluster_thresh)
    if verbose:
        print(f'[trio] D={d_med:.0f}, Cthr={cluster_thresh}, '
              f'Mthr={match_thresh}, Dthr={distinct_thresh}, '
              f'G={len(sizes)}, sizes={sizes}')
    if len(sizes) < 3:
        if verbose:
            print(f'[trio] Skipping: only {len(sizes)} groups (<3) — '
                  f'no triangle possible')
        return np.zeros((0, L), dtype=np.int64)

    # Step 4: within-group consensus dosages
    group_dosages = _compute_group_consensus_dosages(dosage, members)

    # Step 5: enumerate triangles
    trios = _find_grouped_trios(
        centroids,
        match_thresh_bits=match_thresh,
        distinct_thresh_bits=distinct_thresh)
    if verbose:
        print(f'[trio] Found {len(trios)} valid group-triangle trios')
    if not trios:
        return np.zeros((0, L), dtype=np.int64)

    # Step 6: blind consensus recovery
    candidate_haps = _consensus_recovery_blind(
        trios, group_dosages,
        hap_dedup_pct=hap_dedup_pct,
        min_cluster_size=min_hap_cluster_size)
    if verbose:
        print(f'[trio] Recovered {candidate_haps.shape[0]} unique '
              f'candidate haplotypes')
    return candidate_haps