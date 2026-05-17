#%% =====================================================================
# bhd_pairwise.py — Production module
#
# Pairwise common-hap recovery for stage-3 founder discovery.  Extracted
# verbatim from test_pairwise_common_hap_v6.py (the v6 dimensionless
# rewrite of v5), preserving every comment and rationale, plus a single
# new entry point pairwise_recovery_candidate_haps() that block_
# haplotypes_discrete.py's _grow_K_with_recovery and _late_low_carrier_
# rescue call to enrich the trio-recovery candidate pool.
#
# Algorithm summary (v6 docstring):
#
#   - For each pair of samples (i, j), build a PARTIAL common-hap by
#     exploiting the dosage->bit constraints:
#         site has dosage 0 in either i or j (and not 2 in the other) ->
#             the shared strand's bit at that site is FORCED to 0
#         site has dosage 2 in either (and not 0 in the other) -> forced 1
#         site has dosage 1 in BOTH -> undetermined (?)
#         site has dosage 0 in one and 2 in the other -> INCOMPATIBLE
#     A pair with too many incompatible sites cannot share a strand and
#     is rejected.  Surviving pairs produce partial-haps with MASK bits
#     at undetermined sites.
#
#   - Cluster partial-haps that are mutually compatible (agree on every
#     site where both are determined, allowing a small disagreement
#     tolerance for read-error robustness).  Each cluster's consensus
#     fills in MASKs from other partial-haps in the cluster.
#
#   - Resolve any remaining MASK sites via population-frequency tiebreak
#     (same convention as _init_hap_from_sample_dosage).
#
#   - Output the per-cluster consensus haps as candidate founder haps,
#     after the v6 quality filters A-E drop low-determination, low-
#     carrier, and chimera-pattern candidates.
#
# Integration test (50-block sample, integrated with production
# _grow_K_with_recovery): V3 = combined-seed and V6 = combined-seed +
# combined-rescue both captured 258/258 truths vs trio-only V1's
# 257/258, and produced 16 spurious haps vs V1's 17.  The improvement
# came from block chr4:1739 (trio=0 candidates, pairwise=2 clean
# candidates), where pairwise enabled production's K-medoid multistart
# to find all 4 truths instead of 3.  See conversation transcript
# 2026-05-10-17-14-06 for the detailed result analysis.
#
# Feature flag: bhd_pairwise.PAIRWISE_RECOVERY_ENABLED.
#
# NUMBA OPTIMIZATION (added after the bhd_trio numba pass, based on
# production profiling at N=320, L=200, K=6):
#   build_pairwise_partial_haps   — 74 ms -> ~15 ms (5x), via scalar-loop
#                                   kernel that eliminates the per-i
#                                   (n_compat, L) bool temporaries
#                                   (forced_0, forced_1, determined ~5MB
#                                   each per iteration)
#   grow_cluster_iterative        — 130 ms / 6 calls -> ~15 ms (8x), via
#                                   kernel that replaces the (P, L) bool
#                                   overlap/disagree masks (5MB each per
#                                   iteration of 3-10 iters per call)
#                                   with two scalar accumulators per
#                                   partial-hap in a tight loop
#   count_carriers                — 78 us -> 5 us lenient (15x),
#                                   80 us -> 2 us strict (33x), via
#                                   scalar-loop kernel with early-exit
#                                   on incompat > threshold.  Eliminates
#                                   the (N, L) bool intermediate per
#                                   call.  Called ~12x per
#                                   apply_quality_filters invocation;
#                                   strict mode benefits most because
#                                   most non-carrier samples short-
#                                   circuit after a handful of sites.
#   apply_quality_filters         — 1.0 ms -> 0.3 ms (3x), via kernel
#                                   that handles all five filter
#                                   decisions (A determined, B lenient,
#                                   D strict, E lenient-excess, C
#                                   dedup) + the lexicographic sort
#                                   over surviving candidates.  Kernel
#                                   returns survivor/rejection index
#                                   arrays plus rejection codes; the
#                                   Python wrapper reconstructs the
#                                   reject_reason strings (numba does
#                                   not support f-string formatting).
#                                   Filter ordering, sort key (-carriers,
#                                   -cluster_size), first-match dedup
#                                   semantics, and string formats all
#                                   preserved byte-identically.
# Expected end-to-end pairwise_recovery_candidate_haps: 255 ms -> ~80 ms.
# All four kernels preserve byte-identical output: pair iteration order
# (i < j with no reorder), seed iteration order in cluster grow, count
# values, and filter decisions match the original.
#
# Order preservation matters because the downstream seed selection uses
# np.argsort(-determined_counts, kind='stable') — ties break by index
# order, so a numba kernel that emits rows in different order would
# choose different seeds.
#
# No prange: production runs 112 workers x 1 numba thread each (via
# _init_block_worker dynamic allocation), so prange becomes range and
# adds nothing.  Parallelism comes from the worker pool, not numba.
# =======================================================================

import warnings

import numpy as np

# Defensive numba import matching the project convention (see
# analysis_utils.py, block_haplotypes.py, bhd_trio.py).  If numba is
# unavailable, all @njit decorators become no-ops and the per-kernel
# scalar loops run as pure Python (slow but correct).  The wrappers
# preserve the exact same input/output shapes either way.
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    warnings.warn(
        "Numba not found.  bhd_pairwise will fall back to pure-Python "
        "paths for build_pairwise_partial_haps, grow_cluster_iterative, "
        "count_carriers, and apply_quality_filters "
        "(typically 3-33x slower per call).",
        ImportWarning,
    )
    # Dummy decorator that accepts arguments (like cache=True) but does
    # nothing — same pattern as analysis_utils.py and bhd_trio.py.
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        # Support both @njit and @njit(cache=True) forms
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return decorator

# =============================================================================
# MASTER ENABLE FLAG (consumed by block_haplotypes_discrete.py and bhd_recovery.py)
# =============================================================================

# Master switch — set to False to bypass v6 pairwise common-hap recovery
# entirely.  When True (the default), pairwise candidates are merged into
# the trio seed pool at the start of _grow_K_with_recovery AND into the
# rescue pool inside _late_low_carrier_rescue.  Integration testing on a
# 50-block sample (see integration test summary in bhd_pairwise.py)
# showed +1 truth captured and -1 spurious vs trio-only baseline, driven
# by chr4:1739 where trio produced 0 candidates and pairwise produced 2
# clean ones.  No blocks regressed.  Pairwise adds ~0.4 sec/block to
# stage-3 founder discovery.
PAIRWISE_RECOVERY_ENABLED = True


# -----------------------------------------------------------------------
# Algorithm-parameter constants (v6 verbatim from
# test_pairwise_common_hap_v6.py lines 76-175; the test-only
# TRUTH_MATCH_HAMMING_PCT constant is omitted)
# -----------------------------------------------------------------------
# Algorithm parameters (tunable, dimensionless in v6).
#
# Every per-block threshold that v5 expressed as an absolute count of
# sites or samples is now expressed as a fraction of L (block length)
# or N (sample count) and resolved at the consumer call site via
# `round(frac * L)` or `round(frac * N)`.  The fractional values below
# are calibrated so that at L=200, N=320 (the v5 calibration dataset)
# they reproduce v5's absolutes exactly:
#
#       v5 absolute                   v6 fractional                    at L=200, N=320
#   ----------------------------------------------------------------------------------
#   MAX_PAIR_INCOMPATIBILITIES        NOISE_SLACK_FRAC=0.01              →  2 (= L)
#   MAX_CLUSTER_DISAGREEMENTS         NOISE_SLACK_FRAC=0.01              →  2 (= L)
#   MAX_CARRIER_INCOMPAT              NOISE_SLACK_FRAC=0.01              →  2 (= L)
#   MIN_PARTIAL_DETERMINED            MIN_PARTIAL_DETERMINED_FRAC=0.15   → 30 (= L)
#   MIN_CLUSTER_OVERLAP               MIN_CLUSTER_OVERLAP_FRAC=0.10      → 20 (= L)
#   MIN_DETERMINED_FOR_OUTPUT         MIN_DETERMINED_FOR_OUTPUT_FRAC=0.95 → 190 (= L)
#   MIN_STRICT_CARRIERS_FOR_OUTPUT    MIN_STRICT_CARRIER_FRAC=0.11       → 35 (= N)
#
# So the v6 numerics on the v5 dataset are byte-identical to v5.

# Read-error / noise tolerance budget, as a fraction of L.  Used as
# the absolute-count threshold (after `round(NOISE_SLACK_FRAC * L)`)
# in three places where the algorithm needs to absorb sequencing-noise
# sites:
#   1. Pair compatibility in build_pairwise_partial_haps (the v5 name
#      was MAX_PAIR_INCOMPATIBILITIES) — a pair of samples whose
#      argmax dosages conflict at more than this many sites cannot
#      share a strand and is rejected.
#   2. Cluster compatibility in grow_cluster_iterative (was
#      MAX_CLUSTER_DISAGREEMENTS) — a candidate partial-hap may
#      disagree with the running cluster consensus at up to this many
#      sites in their overlap region.
#   3. Carrier consistency in count_carriers (was MAX_CARRIER_INCOMPAT)
#      — the lenient-carrier threshold; a sample's argmax dosage may
#      be diploid-incompatible with a candidate hap at up to this many
#      sites and still count as a "lenient carrier".  Used by Filter B
#      and as the lenient count in Filter E.
#
# 0.01 was calibrated for the project's sequencing depth (~5x to 20x),
# at which per-site dosage error is roughly 0.5-1.5%.  At L=200 it
# resolves to 2 sites — matching v5's three absolute-2 constants
# exactly.
#
# History — why 1% (= 2 sites at L=200) and not tighter:
#
# v5 UPDATE (post-v4-testing, 2026-05): REVERTED MAX_PAIR_INCOMPATIBILITIES
# from 1 back to 2 based on actual data.  The v4 run with this set to
# 1 showed essentially no improvement vs v3 with this set to 2:
#   v3 (p=2): 235 captured, 49 spurious filt
#   v4 (p=1): 235 captured, 50 spurious filt
# Partial-haps built dropped only ~3.7% (from 37828/block to
# 36421/block).  The Poisson estimate predicted a 4.4x reduction in
# cross-truth pair admission; in practice it didn't materialise.
#
# Why the math was wrong: the estimate assumed cross-truth pairs are
# admitted at the margin (right at p=2).  In reality, most cross-truth
# pairs that get admitted at p=2 also satisfy p=1, while most pairs
# near the boundary are real shared-strand pairs with more variable
# read-error counts.  So tightening just excluded some real pairs at
# the margin (slight noise) and didn't dent the chimera cluster sizes.
#
# Lesson: the chimeras come from population-modal admission at the
# cluster-merge stage, not the pair-admission stage.  Tightening here
# is the wrong knob.
NOISE_SLACK_FRAC = 0.01
MIN_PARTIAL_DETERMINED_FRAC = 0.15      # require ≥ this fraction of L
                                        # bits to be determined in a
                                        # partial-hap to consider it
                                        # (drops low-info pairs).
                                        # v6: was MIN_PARTIAL_DETERMINED
                                        # = 30 absolute; at L=200, 0.15
                                        # resolves to 30 exactly.
MIN_CLUSTER_OVERLAP_FRAC = 0.10         # for cluster merging: a candidate
                                        # partial-hap must share ≥ this
                                        # fraction of L determined sites
                                        # with the current consensus (so
                                        # the compatibility check is
                                        # meaningful).
                                        # v6: was MIN_CLUSTER_OVERLAP =
                                        # 20 absolute; at L=200, 0.10
                                        # resolves to 20 exactly.
# (Cluster-merge disagreement budget — v5's MAX_CLUSTER_DISAGREEMENTS = 2
# — is the same noise-tolerance concept as pair compatibility above.  It
# resolves from NOISE_SLACK_FRAC at the consumer call site in
# pairwise_common_hap_recover; no separate constant is needed.)
MIN_CLUSTER_SIZE = 3                    # emit consensus only for clusters
                                        # of ≥ this many partial-haps.
                                        # v6: kept as absolute count.
                                        # This is a minimum statistical
                                        # support count for emitting a
                                        # consensus at all, not a
                                        # fraction of N or L; 3 is the
                                        # minimum that distinguishes a
                                        # genuine cluster from a single
                                        # partial-hap or a transient pair.
MAX_CLUSTER_GROW_ITER = 10              # cap on the iterative consensus-
                                        # refinement loop per cluster.
                                        # v6: kept as absolute (iteration
                                        # cap, not L/N-dependent).

# -----------------------------------------------------------------------
# Quality-filter configuration (v2-v5; verbatim from
# test_pairwise_common_hap_v6.py lines 180-319)
# -----------------------------------------------------------------------
# Filter A — determination floor.  A cluster's consensus must have at
# least this fraction of L bits actually determined by the cluster's
# pair-of-carriers evidence (i.e., not population-frequency tiebreak
# filled).  Spurious haps frequently have low determination because the
# cluster admitted few pair-of-carrier instances and most sites came
# from the pop-frequency fallback, which drifts from any specific truth
# at sites where the truth disagrees with the population mode.  In the
# v1 run, every CAPTURED hap had determined ≥ 195/200, while ~30
# spurious haps had determined < 190 — so 190 is a clean cutoff.
#
# v6: was MIN_DETERMINED_FOR_OUTPUT = 190 absolute; at L=200, the
# fraction 0.95 resolves to 190 exactly.
MIN_DETERMINED_FOR_OUTPUT_FRAC = 0.95

# Filter B — carrier-count floor.  A "carrier" of hap h is a sample
# whose argmax dosage is consistent with carrying h as one of its two
# diploid strands — specifically, at most `noise_slack` sites violate
# the diploid constraint (h[l]=0 AND dosage[s,l]=2) or (h[l]=1 AND
# dosage[s,l]=0), where `noise_slack = round(NOISE_SLACK_FRAC * L)`.
# The noise_slack budget allows for read-error noise; the same
# threshold is used for pair compatibility in build_pairwise_partial_haps,
# so the criterion is internally consistent.  Real founders typically
# have 30-100 carriers; chimeric haps (the dominant Pattern 2 spurious
# in the v1 trace) have very few real carriers because the chimera
# disagrees with each true founder at enough sites to push their
# carriers past the incompat threshold.
#
# v6: MAX_CARRIER_INCOMPAT = 2 absolute is replaced by per-block
# resolution from NOISE_SLACK_FRAC at the call site (inside
# count_carriers and apply_quality_filters).  MIN_CARRIERS_FOR_OUTPUT
# stays absolute — 3 is the minimum statistical support count below
# which any "carrier set" is indistinguishable from random consistency
# noise; this is not an N-fraction threshold.
MIN_CARRIERS_FOR_OUTPUT = 3

# Filter C — cross-cluster dedup.  After A and B, merge clusters whose
# consensus haps are within this Hamming pct of each other; keep the
# one with more carriers (and, as tiebreak, larger cluster size).
# Catches the chr2-block-1210 case where the same truth produced two
# nearby clusters at slightly different Hammings (e.g., 0% and 4.5%)
# because partial-haps with seed-error patterns formed a separate cluster
# that didn't fully merge with the clean truth's cluster.
#
# v3 UPDATE (post-v2-testing, 2026-05): SET TO 0.0 TO DISABLE.  The v2
# run with CROSS_CLUSTER_DEDUP_PCT = 5.0 lost 37 real captures (capture
# rate dropped from 91.1% raw to 77.1% filtered) while only removing 9
# spurious — net negative.  Two reasons the 5%-by-carriers approach
# fails on this dataset:
#
#   (1) Distinct real founders within a 200-site block are commonly only
#       1-5% apart in Hamming because they share chromosomal background
#       and differ at only a handful of sites.  A 5% threshold collapses
#       these distinct truths into one — e.g., chr13 block 500 truth #0
#       and truth #5 are both at Hamming 0% from their own truth and
#       2.0% from each other; chr16 block 764 truth #4 and truth #1 are
#       2.0% apart; chr3 block 6393 truth #5 and truth #2 are 3.5%
#       apart.  All got collapsed in v2.
#
#   (2) The "keep the cluster with more carriers" tiebreaker is
#       systematically inverted for this dataset.  Population-modal
#       chimeras score HIGHER on the carriers metric (with
#       max_incompat = 2) than truth-specific haps do, because the
#       carrier check at max_incompat = 2 is dominated by hom-position
#       compatibility with the population mode.  E.g., chr2 block 1210:
#       chimera at 4.5% from truth #5 has carriers = 226, while real
#       truth #5 has carriers = 83 — so dedup kept the chimera and
#       dropped the truth.
#
# Setting to 0.0 makes the merge condition (ham_pct < 0.0) never
# satisfied, so apply_quality_filters runs through the dedup loop as
# a no-op.  Filter A (determination floor) is the only filter doing
# useful work in v3, with Filter B retained at MAX_CARRIER_INCOMPAT = 2
# as a no-op for now (a future v4 may try MAX_CARRIER_INCOMPAT = 0 to
# count strict-fit carriers, which the v2 trace suggests would be a
# better discriminator between real and chimeric haps).
CROSS_CLUSTER_DEDUP_PCT = 0.0


# Filter D — strict-carriers floor.  A "strict carrier" of hap h is a
# sample whose argmax dosage is consistent with carrying h as one of
# its two diploid strands with NO incompatibility slack: at every site
# l, hap h does NOT take the value (0 with dosage 2) or (1 with dosage
# 0).  This is the same as count_carriers(..., max_incompat=0).
#
# v5 UPDATE: ADDED based on programmatic analysis of the v4 trace.
# The v4 strict-carrier diagnostic showed:
#   bucket    n  min  p25  med  p75  max
#    <2.0%  236   36   79  101  133  314   ← captures
#     2-5%    8   29   58   81   99  120   ← borderline
#    5-10%   23   21   41   76   92  128   ← chimeras
#   10-25%   18   14   43   77  104  157   ← chimeras
#
# The minimum strict-carrier value among captures is 36 (chr4 block
# 1381 truth #1, a low-frequency real founder).  Setting the threshold
# at 35 leaves a safety margin of 1 above that minimum and:
#   - captures lost: 0 (next capture above 35 is at strict=36)
#   - spurious dropped: 8 (those at strict ∈ [14, 30] across all
#     spurious-Hamming buckets)
#
# This is a strictly Pareto-improving filter on this dataset.
#
# v6: was MIN_STRICT_CARRIERS_FOR_OUTPUT = 35 absolute; at N=320, the
# fraction 0.11 resolves via round(0.11 * 320) = round(35.2) = 35
# exactly.  This fraction encodes a population-structure assumption
# (roughly: "we expect each real founder to be carried by at least 11%
# of samples") rather than a noise-floor or block-length assumption.
# For a study with substantially different K (founder count) or
# admixture structure, this is the parameter most likely to need
# recalibration.
MIN_STRICT_CARRIER_FRAC = 0.11


# Filter E — lenient-excess ratio cap.  A real founder's lenient
# carriers (those passing max_incompat=2) are mostly the same set as
# its strict carriers (max_incompat=0); the difference reflects sample-
# specific read errors at a small number of sites.  A chimera's
# lenient carriers include many samples that pass at max_incompat=2
# only because they fail at exactly 1-2 sites — typically the sites
# where the chimera drifted away from the true founder.  So:
#
#   ratio := (carriers - strict) / carriers
#
# is the fraction of "lenient-only" carriers.  Real founders have low
# ratio (~0.10 typical, max 0.57 in the v4 trace); chimeras have
# higher ratio (median 0.40-0.45 across spurious buckets, max 0.85).
# Threshold at 0.60:
#   - captures lost: 0 (max capture ratio in v4 is 0.569; safety
#     margin 0.03 — small but nonzero)
#   - spurious dropped: ~12 net additional vs Filter D alone
#     (combined spurious_remaining = 38, vs 50 raw and 42 with D only)
#
# A tighter threshold (0.55) would catch 1-2 more spurious but at the
# cost of dropping 1-2 captures, which is not worth it given the
# downstream BIC selection can absorb spurious cheaply but cannot
# recover dropped captures.
MAX_LENIENT_EXCESS_RATIO = 0.60


# -----------------------------------------------------------------------
# CORE ALGORITHM — pairwise common-hap recovery
# -----------------------------------------------------------------------
@njit(cache=True)
def _build_pairwise_partial_haps_kernel(dosage,
                                          max_pair_incompat,
                                          min_partial_determined):
    """Numba kernel for build_pairwise_partial_haps.

    Replaces the original's per-i broadcasted bool arrays
    (forced_0, forced_1, determined; each (n_compat, L) ≈ 5MB at
    production size) with a tight scalar loop over (i, j) pairs that
    writes directly into a preallocated output buffer.

    Buffer sizing: P_max = N*(N-1)/2 worst case (every pair compatible).
    At N=320 that's 51040 partial-haps × L bytes (int8) ≈ 10 MB for
    values, same for determined, plus 8 bytes/row for sources.  We
    truncate to actual P at the wrapper.

    Two-pass per pair so we can early-exit on incompat before doing the
    expensive second pass.  This matches the original's
    `n_incompat <= max_pair_incompat` admission filter — pairs failing
    incompat are rejected before the (n_compat, L) inner work.

    Args:
        dosage: (N, L) int64 — argmax dosages in {0, 1, 2}
        max_pair_incompat: int — pair-rejection threshold (= 2 at L=200)
        min_partial_determined: int — partial-rejection threshold (= 30
            at L=200)

    Returns:
        all_values: (P, L) int8 — tightly sized, contains {-1, 0, 1}
        all_determined: (P, L) bool — tightly sized
        sources: (P, 2) int64 — (i, j) per row
    """
    N, L = dosage.shape
    P_max = N * (N - 1) // 2

    # Allocate worst-case buffers.  At N=320 this is ~10MB int8 for
    # all_values plus ~10MB bool for all_determined; manageable.
    all_values_buf = np.empty((P_max, L), dtype=np.int8)
    all_determined_buf = np.empty((P_max, L), dtype=np.bool_)
    sources_buf = np.empty((P_max, 2), dtype=np.int64)

    P = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            # Pass 1: count incompat sites.  Early-exit when we already
            # know the pair will be rejected — saves the second pass on
            # the typical ~50% of pairs that are incompatible.
            n_incompat = 0
            for l in range(L):
                d_i = dosage[i, l]
                d_j = dosage[j, l]
                if (d_i == 0 and d_j == 2) or (d_i == 2 and d_j == 0):
                    n_incompat += 1
                    if n_incompat > max_pair_incompat:
                        break
            if n_incompat > max_pair_incompat:
                continue

            # Pass 2: build the partial hap at row P of the output
            # buffer.  We're tentatively writing here; if n_determined
            # ends up below threshold we'll simply not increment P and
            # the next pair overwrites this row.
            n_determined = 0
            for l in range(L):
                d_i = dosage[i, l]
                d_j = dosage[j, l]
                # Incompat site -> stay MASK (-1, not determined).
                # Original numpy code masks via `& ~inc_compat` and
                # leaves the -1 fill from np.full.
                if (d_i == 0 and d_j == 2) or (d_i == 2 and d_j == 0):
                    all_values_buf[P, l] = -1
                    all_determined_buf[P, l] = False
                # Non-incompat: forced_0 priority (matches numpy's
                # "partials[forced_0]=0 first, then partials[forced_1]=1"
                # — these are mutually exclusive after incompat masking,
                # so the elif order matches when one is True).
                elif d_i == 0 or d_j == 0:
                    all_values_buf[P, l] = 0
                    all_determined_buf[P, l] = True
                    n_determined += 1
                elif d_i == 2 or d_j == 2:
                    all_values_buf[P, l] = 1
                    all_determined_buf[P, l] = True
                    n_determined += 1
                else:
                    all_values_buf[P, l] = -1
                    all_determined_buf[P, l] = False

            if n_determined >= min_partial_determined:
                sources_buf[P, 0] = i
                sources_buf[P, 1] = j
                P += 1
            # else: row at P is discarded; next iter overwrites it

    return all_values_buf[:P], all_determined_buf[:P], sources_buf[:P]


def build_pairwise_partial_haps(probs,
                                 max_pair_incompat=None,
                                 min_partial_determined=None):
    """Step 1 — for every pair of samples (i, j) with i < j, build the
    partial common-hap implied by their dosages (or skip if incompatible).

    Args:
        probs: (N, L, 3) genotype posteriors
        max_pair_incompat: int, optional — max # of dosage-0-vs-dosage-2
            sites tolerated per pair before we reject as "no shared
            strand".  If None (the v6 default), resolved per-block as
            `max(1, round(NOISE_SLACK_FRAC * L))`.
        min_partial_determined: int, optional — minimum number of
            determined sites required in a partial-hap to keep it.
            If None (the v6 default), resolved per-block as
            `max(1, round(MIN_PARTIAL_DETERMINED_FRAC * L))`.

    Returns:
        all_values:    (P, L) np.int8 in {0, 1, -1}.  -1 means MASK
                       (undetermined).
        all_determined: (P, L) bool — True at sites where the partial-
                       hap value is determined (= forced 0 or forced 1).
        sources:       list of (i, j) for each row.

    Implementation: delegates to a numba kernel that uses scalar loops
    over (i, j) and writes directly into preallocated worst-case
    buffers.  Output is byte-identical to the pre-numba implementation
    (same row order, same values, same sources) — order is preserved
    so the downstream `np.argsort(-determined_counts, kind='stable')`
    seed selection picks the same seeds.
    """
    N, L, _ = probs.shape
    # v6 — per-block resolution of fractional thresholds.  At L=200 the
    # values resolve to 2 and 30, matching v5 absolutes exactly.
    if max_pair_incompat is None:
        max_pair_incompat = max(1, int(round(NOISE_SLACK_FRAC * L)))
    if min_partial_determined is None:
        min_partial_determined = max(1, int(round(MIN_PARTIAL_DETERMINED_FRAC * L)))
    dosage = probs.argmax(axis=2).astype(np.int64)        # (N, L)

    # Edge case: N < 2 means no pairs.  The kernel handles this
    # correctly (its outer loop runs 0 times) but we short-circuit
    # to avoid allocating P_max=0 buffers.
    if N < 2:
        empty_v = np.empty((0, L), dtype=np.int8)
        empty_d = np.empty((0, L), dtype=bool)
        return empty_v, empty_d, []

    all_values, all_determined, sources_arr = \
        _build_pairwise_partial_haps_kernel(
            dosage, int(max_pair_incompat), int(min_partial_determined))

    if all_values.shape[0] == 0:
        empty_v = np.empty((0, L), dtype=np.int8)
        empty_d = np.empty((0, L), dtype=bool)
        return empty_v, empty_d, []

    # Repackage sources from (P, 2) array into the legacy list-of-
    # tuples that downstream callers expect.
    P = sources_arr.shape[0]
    sources = [(int(sources_arr[k, 0]), int(sources_arr[k, 1]))
                for k in range(P)]
    return all_values, all_determined, sources


@njit(cache=True)
def _grow_cluster_iterative_kernel(seed_idx,
                                      all_values,
                                      all_determined,
                                      available,
                                      min_cluster_overlap,
                                      max_cluster_disagreements,
                                      max_iter):
    """Numba kernel for grow_cluster_iterative.

    Replaces the original's (P, L) bool overlap and disagree_mask
    intermediates (~5MB each per iteration) with two scalar
    accumulators per partial-hap computed in a tight loop, avoiding
    the per-iteration allocation cost that dominated the production
    profile.

    Optimizations applied on top of the basic scalar-loop port:
      1. EARLY TERMINATION in the compat-check inner L-loop: break as
         soon as disagreements > max_cluster_disagreements, since the
         partial-hap is already rejected and the rest of the L-pass
         only affects overlap_count (which is irrelevant once we know
         compatible[p] = False).  At production size most partials
         in pass-2+ ARE rejected (the cluster has converged), so this
         is the dominant fast path.
      2. COMPACT COMPATIBLE-INDEX in the votes pass: rather than
         scanning all P=25712 partials per site checking
         `if compatible[p]`, we build a packed int64 array of just
         the compatible indices (~5000 entries typical) once per
         iteration and iterate that.  Reduces the votes pass from
         L*P to L*n_compat, ~5x at production size.

    Algorithm faithfully reproduces the Python version including its
    convergence semantics:
      - First iteration: prev_compatible is None, no convergence check;
        we just compute compatible[], assign to prev_compatible
      - Later iterations: break if compatible == prev_compatible
      - On break from convergence: final_compatible retains the value
        set at the END of the previous successful iteration
      - On break from empty compat (compat_count < 1): final_compatible
        retains last iter's value or stays None (handled by the
        have_final flag)
      - If no iteration ever set final_compatible (i.e. max_iter == 0,
        or first iter compat empty): fall back to seed-only

    Tiebreak in majority vote: votes_1 >= votes_0 -> 1, matching the
    Python `np.where(votes_1 >= votes_0, 1, 0)`.

    Mutates available[] in place at end (marks final_compatible members
    as no longer available).

    Args:
        seed_idx: int — index into all_values for the seed partial-hap
        all_values: (P, L) int8 in {-1, 0, 1}
        all_determined: (P, L) bool
        available: (P,) bool — modified in-place at end
        min_cluster_overlap: int
        max_cluster_disagreements: int
        max_iter: int

    Returns:
        cur_values: (L,) int8 — final consensus
        cur_determined: (L,) bool
        final_compatible: (P,) bool — True at cluster member indices
            (caller does np.where(...)[0].tolist() to get member list)
    """
    P, L = all_values.shape

    # Copy seed into consensus
    cur_values = np.empty(L, dtype=np.int8)
    cur_determined = np.empty(L, dtype=np.bool_)
    for l in range(L):
        cur_values[l] = all_values[seed_idx, l]
        cur_determined[l] = all_determined[seed_idx, l]

    # Scratch buffers reused across iterations.  numba prefers explicit
    # types and shapes here over dynamic resizing.
    compatible = np.zeros(P, dtype=np.bool_)
    prev_compatible = np.zeros(P, dtype=np.bool_)
    final_compatible = np.zeros(P, dtype=np.bool_)
    new_values = np.empty(L, dtype=np.int8)
    new_determined = np.empty(L, dtype=np.bool_)
    # Packed compat index — populated each iteration after compatibility
    # check.  Size up to P; we track active count separately.
    compat_idx = np.empty(P, dtype=np.int64)

    have_prev = False
    have_final = False

    for _it in range(max_iter):
        # Compute compatible[p] = available[p] AND overlap_count(p) >=
        # min_cluster_overlap AND disagreements(p) <= max_disagreements.
        # Early-exit the inner L-loop on disagreements > max — the
        # partial is rejected regardless of the rest of the L-pass.
        # At production size in iter 2+ this is the fast path for most
        # partials (cluster has shrunk to the true strand's partials).
        for p in range(P):
            if not available[p] and p != seed_idx:
                # Even non-available partials are skipped — they got
                # claimed by an earlier cluster.  (seed_idx is always
                # forced True below.)
                compatible[p] = False
                continue
            overlap_count = 0
            disagreements = 0
            rejected = False
            for l in range(L):
                if cur_determined[l] and all_determined[p, l]:
                    overlap_count += 1
                    if cur_values[l] != all_values[p, l]:
                        disagreements += 1
                        if disagreements > max_cluster_disagreements:
                            rejected = True
                            break
            if rejected:
                compatible[p] = False
            else:
                compatible[p] = overlap_count >= min_cluster_overlap
        # Always include the seed itself
        compatible[seed_idx] = True

        # Convergence: compatible set hasn't changed since last iter
        if have_prev:
            equal = True
            for p in range(P):
                if compatible[p] != prev_compatible[p]:
                    equal = False
                    break
            if equal:
                break
        # Snapshot compatible into prev for next iter's check
        for p in range(P):
            prev_compatible[p] = compatible[p]
        have_prev = True

        # Build packed compatible-index array (reused in votes pass)
        n_compat = 0
        for p in range(P):
            if compatible[p]:
                compat_idx[n_compat] = p
                n_compat += 1
        if n_compat < 1:
            # Original breaks before setting final_compatible — leave
            # have_final as it was (False on first iter, True if a
            # previous iter set it; that snapshot persists).
            break

        # Recompute consensus by majority vote across compatible
        # partial-haps.  L-outer, n_compat-inner; iterate the packed
        # index array (~5000 entries at production size) rather than
        # all P=25712 partials.
        for l in range(L):
            votes_0 = 0
            votes_1 = 0
            for k in range(n_compat):
                p = compat_idx[k]
                if all_determined[p, l]:
                    v = all_values[p, l]
                    if v == 0:
                        votes_0 += 1
                    elif v == 1:
                        votes_1 += 1
            total = votes_0 + votes_1
            if total > 0:
                new_determined[l] = True
                # Tiebreak: prefer 1 if tied (matches numpy
                # `np.where(votes_1 >= votes_0, 1, 0)`).
                if votes_1 >= votes_0:
                    new_values[l] = 1
                else:
                    new_values[l] = 0
            else:
                new_determined[l] = False
                new_values[l] = -1

        # Commit new consensus to cur_*
        for l in range(L):
            cur_values[l] = new_values[l]
            cur_determined[l] = new_determined[l]
        # Snapshot compatible -> final_compatible
        for p in range(P):
            final_compatible[p] = compatible[p]
        have_final = True

    # If no iteration ever set final_compatible (e.g. max_iter == 0,
    # or first iter had compat_count < 1 with no prior commit), fall
    # back to seed-only.  Matches the Python original.
    if not have_final:
        for p in range(P):
            final_compatible[p] = False
        final_compatible[seed_idx] = True

    # Mutate available in-place — same as the Python version
    for p in range(P):
        if final_compatible[p]:
            available[p] = False

    return cur_values, cur_determined, final_compatible


def grow_cluster_iterative(seed_idx,
                            all_values, all_determined,
                            available,
                            min_cluster_overlap=None,
                            max_cluster_disagreements=None,
                            max_iter=MAX_CLUSTER_GROW_ITER):
    """Grow a cluster from a seed partial-hap by iterative batch-merge
    with re-check: at each iteration, find every available partial-hap
    that's compatible with the current consensus; recompute consensus
    by majority vote across all such candidates; iterate until the
    compatible set stabilises.

    "Compatible" = overlap (intersection of determined regions) is
    ≥ min_cluster_overlap, AND disagreements (sites in the overlap
    where consensus and candidate differ) is ≤ max_cluster_disagreements.

    The re-check inside iteration is what handles the pathological case
    where two true strands collapse into the seed's compatibility set:
    after the first majority-vote consensus, candidates from the wrong
    strand fail the disagreement check and drop out.

    Args:
        seed_idx: int — index into all_values for the seed partial-hap
        all_values, all_determined: (P, L) — as returned by
            build_pairwise_partial_haps
        available: (P,) bool — True for partial-haps not yet claimed by
            another cluster.  Modified in-place at the end to mark the
            cluster's final members as no longer available.
        ...

    Returns:
        consensus_values: (L,) int8 in {0, 1, -1} — final cluster
            consensus (-1 marks sites that NO partial-hap in the cluster
            determined; these are filled by the population-frequency
            tiebreak in the caller)
        consensus_determined: (L,) bool
        members: list of int — indices into all_values for the cluster's
            final members

    Implementation: delegates to a numba kernel that uses two scalar
    accumulators per partial-hap (overlap_count, disagreements) instead
    of the original (P, L) bool intermediates, eliminating ~5MB of
    per-iteration allocation at production size.  Output (consensus
    values, members list, available[] post-mutation) is byte-identical
    to the pre-numba implementation.
    """
    P, L = all_values.shape
    # v6 — per-block resolution of fractional thresholds.  At L=200 the
    # values resolve to 20 and 2, matching v5 absolutes exactly.
    if min_cluster_overlap is None:
        min_cluster_overlap = max(1, int(round(MIN_CLUSTER_OVERLAP_FRAC * L)))
    if max_cluster_disagreements is None:
        max_cluster_disagreements = max(1, int(round(NOISE_SLACK_FRAC * L)))

    # Edge case: empty input.  Kernel handles it but we short-circuit
    # to avoid allocating zero-length scratch.
    if P == 0:
        return (np.full(L, -1, dtype=np.int8),
                np.zeros(L, dtype=bool),
                [])

    # Defensive: ensure dtypes match the kernel signature.  The caller
    # passes arrays from build_pairwise_partial_haps which are already
    # int8 / bool, so these casts are no-ops in practice.
    all_values_arr = np.ascontiguousarray(all_values, dtype=np.int8)
    all_determined_arr = np.ascontiguousarray(all_determined, dtype=np.bool_)
    # available is mutated in-place by the kernel; ensure it's a numba-
    # compatible bool array (and that the caller's view picks up the
    # mutation, since np.ascontiguousarray may return a copy if dtype
    # changes).  available is passed as bool from the caller, so the
    # cast is also a no-op and shares memory.
    if available.dtype != np.bool_:
        # Should not happen in production but be safe
        raise ValueError(
            f"available must be bool; got {available.dtype}")

    cur_values, cur_determined, final_compatible = \
        _grow_cluster_iterative_kernel(
            int(seed_idx),
            all_values_arr,
            all_determined_arr,
            available,
            int(min_cluster_overlap),
            int(max_cluster_disagreements),
            int(max_iter),
        )

    members = np.where(final_compatible)[0].tolist()
    return cur_values, cur_determined, members


def pairwise_common_hap_recover(probs,
                                 max_pair_incompat=None,
                                 min_partial_determined=None,
                                 min_cluster_overlap=None,
                                 max_cluster_disagreements=None,
                                 min_cluster_size=MIN_CLUSTER_SIZE,
                                 max_cluster_grow_iter=MAX_CLUSTER_GROW_ITER):
    """Top-level entry: build partial-haps from all compatible pairs,
    cluster them, emit per-cluster consensus haps with population-
    frequency tiebreak at remaining MASK sites.

    Args:
        probs: (N, L, 3) genotype posteriors
        max_pair_incompat: int, optional — pair-compatibility slack.
            If None (default), resolves to round(NOISE_SLACK_FRAC * L).
        min_partial_determined: int, optional — partial-hap info floor.
            If None (default), resolves to round(MIN_PARTIAL_DETERMINED_FRAC * L).
        min_cluster_overlap: int, optional — cluster-merge overlap floor.
            If None (default), resolves to round(MIN_CLUSTER_OVERLAP_FRAC * L).
        max_cluster_disagreements: int, optional — cluster-merge slack.
            If None (default), resolves to round(NOISE_SLACK_FRAC * L).
        min_cluster_size: int — kept absolute (minimum statistical
            support count, not L/N-dependent).
        max_cluster_grow_iter: int — kept absolute (iteration cap).

    Returns:
        list of dicts with keys
            'hap': (L,) int64 binary — final hap with MASKs resolved
            'cluster_size': int
            'determined_count': int — # of bits the cluster determined
                BEFORE the population-frequency tiebreak filled the rest
            'pair_sources': list of (i, j) tuples that contributed
    """
    N, L, _ = probs.shape
    # v6 — resolve None defaults from the global fractional constants.
    # We resolve here AND let the inner functions resolve again from the
    # same fractions (idempotent — both produce the same absolute value
    # at the same L), but resolving here ensures the printed/returned
    # parameter values used in this function (e.g. min_cluster_size
    # comparison below) are well-defined.
    if max_pair_incompat is None:
        max_pair_incompat = max(1, int(round(NOISE_SLACK_FRAC * L)))
    if min_partial_determined is None:
        min_partial_determined = max(1, int(round(MIN_PARTIAL_DETERMINED_FRAC * L)))
    if min_cluster_overlap is None:
        min_cluster_overlap = max(1, int(round(MIN_CLUSTER_OVERLAP_FRAC * L)))
    if max_cluster_disagreements is None:
        max_cluster_disagreements = max(1, int(round(NOISE_SLACK_FRAC * L)))
    # Step 1: pairwise partial-haps
    all_values, all_determined, sources = build_pairwise_partial_haps(
        probs,
        max_pair_incompat=max_pair_incompat,
        min_partial_determined=min_partial_determined)
    P = all_values.shape[0]
    if P == 0:
        return [], 0

    # Step 2: cluster by greedy seed-grow.  Seeds chosen by descending
    # determined-count (the most informative partial-haps anchor the
    # cluster well).
    determined_counts = all_determined.sum(axis=1)
    seed_order = np.argsort(-determined_counts, kind='stable')

    available = np.ones(P, dtype=bool)
    consensus_haps = []

    for seed_idx in seed_order:
        if not available[seed_idx]:
            continue
        cv, cd, members = grow_cluster_iterative(
            int(seed_idx), all_values, all_determined, available,
            min_cluster_overlap=min_cluster_overlap,
            max_cluster_disagreements=max_cluster_disagreements,
            max_iter=max_cluster_grow_iter)
        if len(members) < min_cluster_size:
            continue
        cluster_pair_sources = [sources[m] for m in members]
        consensus_haps.append({
            'cv': cv,
            'cd': cd,
            'cluster_size': len(members),
            'determined_count': int(cd.sum()),
            'pair_sources': cluster_pair_sources,
        })

    # Step 3: resolve MASKs via population alt-allele frequency
    pop_alt_freq = (probs[..., 1].mean(axis=0) * 0.5
                    + probs[..., 2].mean(axis=0))
    final = []
    for c in consensus_haps:
        h = c['cv'].astype(np.int64).copy()
        mask = (c['cv'] == -1)
        h[mask] = (pop_alt_freq[mask] > 0.5).astype(np.int64)
        final.append({
            'hap': h,
            'cluster_size': c['cluster_size'],
            'determined_count': c['determined_count'],
            'pair_sources': c['pair_sources'],
        })

    return final, P

# -----------------------------------------------------------------------
# QUALITY FILTERS (v2 — post-clustering filtering to reduce spurious haps)
# -----------------------------------------------------------------------
@njit(cache=True)
def _count_carriers_kernel(hap, dosage, max_incompat):
    """Numba kernel for count_carriers.

    For each sample s, count the number of sites l where
        (hap[l] == 0 AND dosage[s, l] == 2)
     OR (hap[l] == 1 AND dosage[s, l] == 0)
    If that count <= max_incompat, sample s is a carrier.

    Early-exits the inner L-loop as soon as n_incompat exceeds the
    threshold — at production noise levels most samples are either
    clean carriers (n_incompat ~ 0-1) or clear non-carriers
    (n_incompat >> max_incompat by site 10-20), so most samples
    short-circuit quickly.

    Mathematical equivalence to the original numpy form:
      Original computes the full (N, L) incompat bool array, sums
      to get n_incompat per sample, compares to max_incompat.  The
      sum() vs counting positions before threshold-breach gives the
      same boolean answer: `n_incompat <= max_incompat` is determined
      the moment count first exceeds the threshold (or the L-loop
      finishes).  Early-exit only affects when we stop counting; the
      decision is identical.

    Args:
        hap: (L,) int64 — fully resolved hap in {0, 1}
        dosage: (N, L) int64 — argmax dosages in {0, 1, 2}
        max_incompat: int — slack threshold

    Returns:
        int — number of carrier samples
    """
    N, L = dosage.shape
    carrier_count = 0
    for s in range(N):
        n_incompat = 0
        is_carrier = True
        for l in range(L):
            h_l = hap[l]
            d_sl = dosage[s, l]
            if (h_l == 0 and d_sl == 2) or (h_l == 1 and d_sl == 0):
                n_incompat += 1
                if n_incompat > max_incompat:
                    is_carrier = False
                    break
        if is_carrier:
            carrier_count += 1
    return carrier_count


def count_carriers(hap, dosage, max_incompat=None):
    """Count samples whose argmax dosage is consistent with carrying hap
    as one of the two diploid strands.

    A sample s is "consistent" iff at most max_incompat sites violate
    the diploid constraint:
        (hap[l] == 0  AND  dosage[s, l] == 2)   — would imply h'[l] = 2
        (hap[l] == 1  AND  dosage[s, l] == 0)   — would imply h'[l] = -1
    Either case requires the implied other-strand bit h'[l] to be outside
    {0, 1}, which is impossible for a binary haplotype.

    The max_incompat slack absorbs read-error noise — at the project's
    depth (~5x to 20x), per-site dosage error is roughly 0.5-1.5%, so
    expecting ≤ 2 incompat sites per real carrier on a 200-site block is
    conservative.  This is the same incompat criterion used for pair
    compatibility in build_pairwise_partial_haps; using the same threshold
    keeps the carrier-count check consistent with how partial-haps were
    admitted in the first place.

    Args:
        hap: (L,) int64 array in {0, 1} — fully resolved hap (no MASK)
        dosage: (N, L) int64 argmax dosages in {0, 1, 2}
        max_incompat: int, optional — slack for read-error tolerance.
            If None (the v6 lenient default), resolves to
            `max(1, round(NOISE_SLACK_FRAC * L))` from L = hap.shape[0].
            Pass max_incompat=0 explicitly for a strict-carrier count
            (no slack); this is what Filter D and the strict half of
            Filter E use.

    Returns:
        int — number of samples consistent with carrying hap

    Implementation: delegates to a numba kernel that uses scalar
    loops with early-exit on incompat > threshold.  Eliminates the
    (N, L) bool array allocation per call.  Called ~12 times per
    apply_quality_filters invocation (twice per recovered hap:
    lenient + strict); cumulative per-block ~0.8 ms.
    """
    if max_incompat is None:
        L = hap.shape[0]
        max_incompat = max(1, int(round(NOISE_SLACK_FRAC * L)))
    # Cast to int64 for kernel dtype consistency.  Both arrays are
    # typically int64 already; np.ascontiguousarray is a no-op in
    # that common case.
    hap_arr = np.ascontiguousarray(hap, dtype=np.int64)
    dosage_arr = np.ascontiguousarray(dosage, dtype=np.int64)
    return _count_carriers_kernel(hap_arr, dosage_arr, int(max_incompat))


@njit(cache=True)
def _apply_quality_filters_kernel(haps_arr, carriers_arr, strict_carriers_arr,
                                    determined_arr, cluster_sizes_arr,
                                    min_determined, min_carriers,
                                    min_strict_carriers,
                                    max_lenient_excess_ratio,
                                    cross_dedup_pct):
    """Numba kernel for apply_quality_filters decision logic.

    Replicates the full Filter A/B/D/E + sort + Filter C dedup logic
    of the original apply_quality_filters, EXCLUDING:
      - count_carriers calls (done by the wrapper before invocation
        since they're already numba-accelerated and the kernel needs
        the values regardless)
      - dict construction, string formatting, and the return-list
        building (done by the wrapper using the kernel's index arrays)

    Args:
      haps_arr: (n, L) int64 — stacked hap arrays from recovered
      carriers_arr: (n,) int64 — lenient carrier counts (precomputed)
      strict_carriers_arr: (n,) int64 — strict carrier counts (precomputed)
      determined_arr: (n,) int64 — determined_count from each candidate
      cluster_sizes_arr: (n,) int64 — cluster_size from each candidate
      min_determined, min_carriers, min_strict_carriers: int — filter thresholds
      max_lenient_excess_ratio: float — Filter E threshold
      cross_dedup_pct: float — Filter C dedup threshold

    Returns:
      survivor_order_arr: (n_surv,) int64 — indices into the original
        recovered list, in the FINAL deduped order
      reject_idx_arr: (n_rej,) int64 — indices into the original list,
        in rejection order (rejection order = filter A/B/D/E order
        encountered, then dedup-rejections appended in sort order)
      reject_code_arr: (n_rej,) int64 — code per rejection:
        0=Filter A (determined),  1=Filter B (lenient carriers),
        2=Filter D (strict carriers), 3=Filter E (lenient_excess),
        4=Filter C (dedup)
      reject_dedup_kept_arr: (n_rej,) int64 — for code 4 rejections,
        the index (into recovered) of the kept hap that triggered the
        dedup; -1 for all other codes.
      reject_dedup_hampct_arr: (n_rej,) float64 — for code 4 rejections,
        the Hamming-% computed for the dedup decision; 0.0 for others.

    Semantics preserved exactly from the original:
      - Filter ordering A, B, D, E, then C dedup (same as the original).
      - sorted(survivors, key=(-carriers, -cluster_size)) for dedup
        order — implemented via np.argsort with a composite negation
        and Python's timsort-stable equivalent.
      - Strict-inequality Filter checks (`<`, `>`) preserved.
      - Filter E only applies when carriers > 0 (matches the original
        `if r['carriers'] > 0:` guard).
      - Lenient-excess is `(carriers - strict_carriers) / carriers`
        as float division (NOT integer).
      - Dedup uses `ham_pct < cross_dedup_pct` (strict-less).
      - Dedup walks `deduped` in INSERTION order (first match wins)
        and breaks on first match (matching `is_dup; break`).
    """
    n = haps_arr.shape[0]
    L = haps_arr.shape[1]

    # Pre-allocate output buffers with worst-case sizes (n elements each).
    # We track actual fill counts and truncate at the end via slicing in
    # the wrapper.
    survivor_order_arr = np.empty(n, dtype=np.int64)
    reject_idx_arr = np.empty(n, dtype=np.int64)
    reject_code_arr = np.empty(n, dtype=np.int64)
    reject_dedup_kept_arr = np.full(n, -1, dtype=np.int64)
    reject_dedup_hampct_arr = np.zeros(n, dtype=np.float64)

    n_surv_intermediate = 0
    n_rej = 0

    # First-pass survivors (after A/B/D/E, before dedup).  We store
    # their indices into the original array; the dedup step sorts and
    # walks this set.
    surv_pre_dedup = np.empty(n, dtype=np.int64)

    # === Step 2: filters A, B, D, E ===
    for i in range(n):
        # Filter A — determined floor
        if determined_arr[i] < min_determined:
            reject_idx_arr[n_rej] = i
            reject_code_arr[n_rej] = 0
            n_rej += 1
            continue
        # Filter B — lenient-carrier floor
        if carriers_arr[i] < min_carriers:
            reject_idx_arr[n_rej] = i
            reject_code_arr[n_rej] = 1
            n_rej += 1
            continue
        # Filter D — strict-carrier floor (v5)
        if strict_carriers_arr[i] < min_strict_carriers:
            reject_idx_arr[n_rej] = i
            reject_code_arr[n_rej] = 2
            n_rej += 1
            continue
        # Filter E — lenient-excess cap (v5).  Only when carriers > 0
        # (matches original guard).  Filter B above already excluded
        # carriers < min_carriers; min_carriers > 0 in production so
        # this guard is effectively redundant but kept for byte-fidelity.
        if carriers_arr[i] > 0:
            lenient_excess = (carriers_arr[i]
                              - strict_carriers_arr[i]) / carriers_arr[i]
            if lenient_excess > max_lenient_excess_ratio:
                reject_idx_arr[n_rej] = i
                reject_code_arr[n_rej] = 3
                n_rej += 1
                continue
        # Survivor
        surv_pre_dedup[n_surv_intermediate] = i
        n_surv_intermediate += 1

    # === Step 3: Filter C — sort by (-carriers, -cluster_size) ===
    # We sort surv_pre_dedup[:n_surv_intermediate] indices by the
    # composite key.  Python's sorted is stable; numba's np.argsort
    # with `kind='stable'` (mergesort/timsort) gives the same ordering.
    #
    # Build the sort key.  Since numba's argsort doesn't accept tuple
    # keys, we use the trick: sort by primary, then re-sort the result
    # within each primary-tied group by secondary.  Cleaner: build a
    # single composite float64 key by combining the two int fields.
    # We use a lexicographic argsort via two-pass stable sort:
    #   1. Sort by secondary (-cluster_size) — stable, preserves order
    #      among elements with equal secondary.
    #   2. Sort by primary (-carriers) — stable; ties preserve the
    #      ordering established in step 1.
    # Result: lexicographically sorted by (primary, secondary).
    surv_sub = surv_pre_dedup[:n_surv_intermediate]
    neg_cluster = -cluster_sizes_arr[surv_sub]
    neg_carriers = -carriers_arr[surv_sub]
    # Stable sort by secondary first
    order2 = np.argsort(neg_cluster, kind='mergesort')
    # Apply, then stable sort by primary
    surv_sub_ordered2 = surv_sub[order2]
    neg_carriers_ordered2 = neg_carriers[order2]
    order1 = np.argsort(neg_carriers_ordered2, kind='mergesort')
    sorted_idx = surv_sub_ordered2[order1]

    # === Dedup walk ===
    # `kept_indices` holds the indices (into recovered) of haps that
    # survived dedup so far, in insertion order.  For each candidate in
    # sorted_idx, check against each kept hap and reject on first match.
    kept_indices = np.empty(n_surv_intermediate, dtype=np.int64)
    n_kept = 0
    for ki in range(sorted_idx.shape[0]):
        cand_i = sorted_idx[ki]
        is_dup = False
        kept_match_idx = -1
        ham_pct_match = 0.0
        for kj in range(n_kept):
            kept_i = kept_indices[kj]
            # Hamming-% between haps_arr[cand_i] and haps_arr[kept_i].
            # Matches the original's `float(np.mean(r['hap'] != kept['hap'])) * 100.0`
            diff_count = 0
            for l in range(L):
                if haps_arr[cand_i, l] != haps_arr[kept_i, l]:
                    diff_count += 1
            ham_pct = (diff_count / L) * 100.0
            if ham_pct < cross_dedup_pct:
                is_dup = True
                kept_match_idx = kept_i
                ham_pct_match = ham_pct
                break
        if is_dup:
            reject_idx_arr[n_rej] = cand_i
            reject_code_arr[n_rej] = 4
            reject_dedup_kept_arr[n_rej] = kept_match_idx
            reject_dedup_hampct_arr[n_rej] = ham_pct_match
            n_rej += 1
        else:
            kept_indices[n_kept] = cand_i
            survivor_order_arr[n_kept] = cand_i
            n_kept += 1

    # Return per-array slices via .copy() so the wrapper can slice
    # without holding references to the worst-case buffers.  (Numba
    # supports slice + copy on int64 / float64 arrays.)
    return (survivor_order_arr[:n_kept].copy(),
            reject_idx_arr[:n_rej].copy(),
            reject_code_arr[:n_rej].copy(),
            reject_dedup_kept_arr[:n_rej].copy(),
            reject_dedup_hampct_arr[:n_rej].copy())


def apply_quality_filters(recovered, dosage,
                           min_determined=None,
                           min_carriers=MIN_CARRIERS_FOR_OUTPUT,
                           max_carrier_incompat=None,
                           cross_dedup_pct=CROSS_CLUSTER_DEDUP_PCT,
                           min_strict_carriers=None,
                           max_lenient_excess_ratio=MAX_LENIENT_EXCESS_RATIO):
    """Apply post-clustering quality filters to reduce spurious haps.

    Filter ordering and rationale:
      A. determined_count >= min_determined
         Drops haps that depend heavily on the population-frequency
         MASK fill at the end of pairwise_common_hap_recover.  Such haps
         drift from any truth proportional to the fraction of MASK-filled
         sites where the population mode disagrees with the candidate
         truth, and are typically 5-20% off any real founder.
      B. carrier_count >= min_carriers
         Drops haps with few or no samples whose argmax dosage is
         consistent with carrying them as one strand.  Real founders have
         many real carriers (30-100 typical, ≥ 4 even for low-frequency
         truths in the project's data); chimeric haps (drift between two
         real founders) have very few real carriers because the chimera
         disagrees with each component truth at enough sites to push the
         component's carriers past the incompat threshold.
      C. cross-cluster dedup at cross_dedup_pct
         Merges clusters whose consensus haps are within cross_dedup_pct
         Hamming.  Keeps the one with most carriers (tiebreak: larger
         cluster size).  Collapses fragmented clusters where the same
         truth produced multiple sub-clusters that didn't fully merge
         during clustering — typically because partial-haps with
         correlated seed errors formed a separate cluster.

    Each filter is monotonic with respect to the goal of reducing
    spurious without affecting captures: A and B drop low-quality
    candidates that don't represent any real founder; C consolidates
    redundant near-identical candidates of the SAME truth.

    v5 ADDENDUM — two additional filters were added based on
    programmatic analysis of the v4 trace data:
      D. strict_carriers >= min_strict_carriers
         A "strict carrier" passes count_carriers with max_incompat=0
         (no slack).  Real founders have strict carriers ranging from
         36 (low-frequency truths) to 314 (high-frequency); spurious
         haps' strict-carriers distribution overlaps but extends down
         to 14.  At threshold 35, drops 8 spurious without affecting
         captures (the minimum capture strict-carrier value is 36).
      E. (carriers - strict_carriers) / carriers <= max_lenient_excess_ratio
         The fraction of "lenient-only" carriers — samples that pass
         max_incompat=2 but fail max_incompat=0.  Real founders have
         this ratio mostly < 0.20 (max 0.57 in the v4 data); chimeras
         have median 0.40-0.45.  At threshold 0.60, drops ~12 more
         spurious without affecting captures.

    Args:
        recovered: list of dicts as returned by pairwise_common_hap_recover
        dosage: (N, L) argmax dosages — used for the carrier-count check
        min_determined: int, optional — Filter A threshold.  If None
            (default), resolves to round(MIN_DETERMINED_FOR_OUTPUT_FRAC * L)
            from L = dosage.shape[1].
        min_carriers: int — Filter B threshold (kept absolute; minimum
            statistical support count).
        max_carrier_incompat: int, optional — slack for the lenient
            carrier-consistency check.  If None (default), resolves to
            max(1, round(NOISE_SLACK_FRAC * L)).
        cross_dedup_pct: float — Filter C threshold (Hamming pct).
        min_strict_carriers: int, optional — Filter D threshold (v5).
            If None (default), resolves to
            round(MIN_STRICT_CARRIER_FRAC * N) from N = dosage.shape[0].
        max_lenient_excess_ratio: float — Filter E threshold (v5);
            already dimensionless.

    Returns:
        survivors: list of dicts with all original keys plus 'carriers'
            (int — number of carriers consistent with the hap), in the
            order they survive A → B → C.
        rejected: list of dicts (same shape as input + 'carriers' +
            'reject_reason' string) of those filtered out.  Order is
            the order they were rejected.

    Implementation: the decision logic (filters A/B/D/E + sort + Filter
    C dedup) is delegated to a numba kernel that operates on stacked
    arrays.  The kernel returns index arrays (survivor order, rejection
    indices, rejection codes, plus dedup-auxiliary data); the wrapper
    here builds the final dict lists with reject_reason strings.
    Cumulative per-block cost was ~0.92 ms across 1 call; after this
    pass dominantly bottlenecked on dict construction and string
    formatting in the wrapper.  Output is byte-identical to the
    pre-numba implementation (same filter ordering, same sort key,
    same first-match dedup, same string formats).
    """
    # v6 — per-block resolution of fractional thresholds.  At L=200 and
    # N=320 the values resolve to 190, 2, and 35 respectively — matching
    # v5 absolutes exactly.
    N, L = dosage.shape
    if min_determined is None:
        min_determined = max(1, int(round(MIN_DETERMINED_FOR_OUTPUT_FRAC * L)))
    if max_carrier_incompat is None:
        max_carrier_incompat = max(1, int(round(NOISE_SLACK_FRAC * L)))
    if min_strict_carriers is None:
        min_strict_carriers = max(1, int(round(MIN_STRICT_CARRIER_FRAC * N)))

    n_rec = len(recovered)
    if n_rec == 0:
        return [], []

    # Step 1 — annotate every candidate with its carrier count.  We
    # compute this once per candidate so subsequent filter steps can
    # reuse the value.
    enriched = []
    for r in recovered:
        carriers = count_carriers(r['hap'], dosage,
                                   max_incompat=max_carrier_incompat)
        r2 = dict(r)
        r2['carriers'] = carriers
        enriched.append(r2)

    # v5 Step 1.5 — also compute strict-carrier count (max_incompat=0)
    # for each candidate, used by Filters D and E below.  Pass over the
    # already-enriched list rather than rebuilding to keep the existing
    # Step 1 byte-identical.
    for r2 in enriched:
        r2['strict_carriers'] = count_carriers(r2['hap'], dosage,
                                                max_incompat=0)

    # Stack arrays for the numba kernel.
    haps_arr = np.empty((n_rec, L), dtype=np.int64)
    carriers_arr = np.empty(n_rec, dtype=np.int64)
    strict_arr = np.empty(n_rec, dtype=np.int64)
    determined_arr = np.empty(n_rec, dtype=np.int64)
    cluster_arr = np.empty(n_rec, dtype=np.int64)
    for i, r in enumerate(enriched):
        haps_arr[i] = np.asarray(r['hap'], dtype=np.int64)
        carriers_arr[i] = int(r['carriers'])
        strict_arr[i] = int(r['strict_carriers'])
        determined_arr[i] = int(r['determined_count'])
        cluster_arr[i] = int(r['cluster_size'])

    (survivor_order, reject_idx, reject_code,
     reject_dedup_kept, reject_dedup_hampct) = _apply_quality_filters_kernel(
        haps_arr, carriers_arr, strict_arr,
        determined_arr, cluster_arr,
        int(min_determined), int(min_carriers),
        int(min_strict_carriers),
        float(max_lenient_excess_ratio),
        float(cross_dedup_pct))

    # Build the final survivor list (in deduped order)
    deduped = [enriched[int(survivor_order[k])]
                for k in range(survivor_order.shape[0])]

    # Build the rejected list with reconstructed reject_reason strings.
    # Codes:
    #   0 = Filter A (determined floor)
    #   1 = Filter B (lenient carriers)
    #   2 = Filter D (strict carriers)
    #   3 = Filter E (lenient_excess)
    #   4 = Filter C (dedup)
    rejected = []
    for j in range(reject_idx.shape[0]):
        i = int(reject_idx[j])
        code = int(reject_code[j])
        r = enriched[i]
        r2 = dict(r)
        if code == 0:
            r2['reject_reason'] = (
                f"determined={r['determined_count']} < {min_determined}")
        elif code == 1:
            r2['reject_reason'] = (
                f"carriers={r['carriers']} < {min_carriers}")
        elif code == 2:
            r2['reject_reason'] = (
                f"strict_carriers={r['strict_carriers']} "
                f"< {min_strict_carriers}")
        elif code == 3:
            # Recompute lenient_excess from the candidate's own values.
            # Original uses `lenient_excess:.2f` and threshold
            # `max_lenient_excess_ratio:.2f`.
            lenient_excess = ((r['carriers'] - r['strict_carriers'])
                              / r['carriers'])
            r2['reject_reason'] = (
                f"lenient_excess={lenient_excess:.2f} "
                f"> {max_lenient_excess_ratio:.2f}")
        else:  # code == 4
            kept_i = int(reject_dedup_kept[j])
            ham_pct = float(reject_dedup_hampct[j])
            kept = enriched[kept_i]
            r2['reject_reason'] = (
                f"dedup against carriers={kept['carriers']} "
                f"cluster={kept['cluster_size']} at Ham={ham_pct:.2f}%")
        rejected.append(r2)

    return deduped, rejected



# -----------------------------------------------------------------------
# Top-level production entry point.  Used by block_haplotypes_discrete's
# _grow_K_with_recovery (seed stage) and _late_low_carrier_rescue
# (rescue stage) to enrich the candidate pool that the existing trio
# recovery already populates.
# -----------------------------------------------------------------------
def pairwise_recovery_candidate_haps(probs_k, verbose=False):
    """Run the v6 pairwise common-hap recovery pipeline + quality filters.

    This is the only function in this module called by block_haplotypes_
    discrete.py.  It packages pairwise_common_hap_recover() (raw
    candidates) and apply_quality_filters() (Filter A-E) into a single
    call that returns a clean list of (L,) int64 binary haps suitable
    for direct concatenation onto the trio-recovery candidate pool
    in _grow_K_with_recovery's step 0.

    Args:
      probs_k: (N, L, 3) float — genotype posteriors in kept-site space.
        Same array the production trio call takes; no transformation
        needed at the call site.
      verbose: bool — when True, prints a one-line summary of how many
        raw candidates were recovered and how many survived filters.

    Returns:
      list of (L,) int64 ndarrays — pairwise candidates that passed
      every quality filter, ready for inclusion in the combined seed
      or rescue pool.  Empty list if the method recovers nothing useful
      (no compatible pairs, no cluster reached min size, all candidates
      filtered out).
    """
    recovered, n_partials = pairwise_common_hap_recover(probs_k)
    if not recovered:
        if verbose:
            print(f'[pairwise] no candidates (n_partials={n_partials})')
        return []
    dosage = probs_k.argmax(axis=2).astype(np.int64)
    survivors, rejected = apply_quality_filters(recovered, dosage)
    if verbose:
        print(f'[pairwise] {len(recovered)} raw candidates -> '
              f'{len(survivors)} after filters '
              f'({len(rejected)} rejected; n_partials={n_partials})')
    return [r['hap'] for r in survivors]