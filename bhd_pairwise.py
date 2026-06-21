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
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    warnings.warn(
        "Numba not found.  bhd_pairwise will fall back to pure-Python "
        "paths for build_pairwise_partial_haps, the clustering kernel, "
        "count_carriers, and apply_quality_filters "
        "(typically 3-33x slower per call).",
        ImportWarning,
    )
    # Dummy decorator that accepts arguments (like cache=True or
    # parallel=True) but does nothing — same pattern as analysis_utils.py
    # and bhd_trio.py.
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        # Support both @njit and @njit(cache=True) forms
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return decorator
    # prange falls back to the builtin range when numba is unavailable, so
    # the parallel build kernels run correctly as (serial) pure Python.
    prange = range

# =============================================================================
# MASTER ENABLE FLAG (consumed by block_haplotypes.py and bhd_recovery.py)
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

# -----------------------------------------------------------------------------
# Residual mode for pairwise recovery (low-read-depth)
# -----------------------------------------------------------------------------
# PAIRWISE_RESIDUAL_MODE selects how the two argmax-dosage hard-call gates in
# this module decide pair compatibility and carrier consistency:
#
#   "argmax" (default) — the behaviour documented throughout: a pair is
#            compatible if its ARGMAX dosages conflict (one 0, the other 2)
#            at <= MAX_PAIR_INCOMPAT sites, and a sample is a carrier of a
#            hap if its ARGMAX dosage is incompatible (h=0 & dosage=2, or
#            h=1 & dosage=0) at <= max_incompat sites.  This is the validated
#            production path; selecting "argmax" leaves build_pairwise_partial_
#            haps and apply_quality_filters bit-identical to before.
#
#   "soft" — a low-read-depth front-end that keeps the genotype posteriors
#            instead of hard-calling the dosage.  Both gates count, per site,
#            only CONFIDENT conflicts/incompatibilities — sites where the
#            posterior probability of the offending genotype exceeds
#            PAIRWISE_SOFT_CONFLICT_TAU — rather than hard argmax events:
#              pair compat:   conflict prob = P_i(0)P_j(2) + P_i(2)P_j(0)
#              carrier h=0:   incompat prob = P_s(2)   (sample hom-alt)
#              carrier h=1:   incompat prob = P_s(0)   (sample hom-ref)
#            At low depth a diffuse posterior contributes a small probability,
#            not a hard +1, so argmax noise no longer spuriously rejects
#            truly-compatible pairs (which collapses the partial-hap pool to
#            nothing at ~3x) or destroys the strict-carrier counts Filters
#            D/E rely on.  Summing the per-site PROBABILITY directly would
#            overcount (it accumulates tiny residual-mass products across all
#            L sites); thresholding each site at tau first — the "more likely
#            than not to conflict here" boundary — is the correct analogue of
#            the hard count.  Determination (the partial-hap VALUES) and the
#            clustering stay argmax; the cluster majority-vote denoises them.
#            Filter A's determined-fraction floor is also left as argmax.
#
# Same rationale as the trio / seed / recovery soft front-ends.  "soft" is
# RESULT-AFFECTING at every read depth and must be validated against ground
# truth before use; it is opt-in for that reason.  NOTE: the strict-carrier
# threshold (Filter D, MIN_STRICT_CARRIER_FRAC) is hand-calibrated against
# real-data argmax strict-carrier distributions; the soft count is larger at
# low depth, so this threshold likely needs RECALIBRATION on real data when
# "soft" is enabled.
PAIRWISE_RESIDUAL_MODE = "argmax"

# Per-site posterior threshold for the "soft" gates: a site counts as a
# confident conflict / incompatibility only when the offending-genotype
# posterior probability exceeds this value.  0.5 is the principled "more
# likely than not" boundary and the empirically-best value (lower values are
# too strict — they flag too many sites and over-reject pairs).
PAIRWISE_SOFT_CONFLICT_TAU = 0.5


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
@njit(parallel=True, cache=True)
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
    values, same for determined.  We truncate to actual P at the wrapper.

    Two-pass per pair so we can early-exit on incompat before doing the
    expensive second pass.  This matches the original's
    `n_incompat <= max_pair_incompat` admission filter — pairs failing
    incompat are rejected before the (n_compat, L) inner work.

    PARALLELISM: the (i, j) pairs are independent, so Phase 1 runs the
    outer i-loop under prange.  To make parallel writes race-free AND
    keep the output ROW ORDER identical to the serial version (the
    downstream `np.argsort(-determined_counts, kind='stable')` seed
    selection depends on it), each pair writes into its DETERMINISTIC
    slot in the worst-case buffer rather than into a shared running
    counter:
        slot(i, j) = base(i) + (j - i - 1),
        base(i)    = i*(N-1) - i*(i-1)//2   (# pairs with first index < i)
    No two pairs map to the same slot, so there is no write contention
    and no need for atomics.  A `keep` flag marks slots that pass both
    the incompat and the min-determined thresholds.  Phase 2 then walks
    the slots in increasing order (= (i, j) enumeration order) and
    compacts the kept rows to the front IN PLACE — safe because the
    write index P never exceeds the read index slot.  Output (kept rows,
    their order, values, determined) is byte-identical to the serial
    two-pass implementation.

    Args:
        dosage: (N, L) int64 — argmax dosages in {0, 1, 2}
        max_pair_incompat: int — pair-rejection threshold (= 2 at L=200)
        min_partial_determined: int — partial-rejection threshold (= 30
            at L=200)

    Returns:
        all_values: (P, L) int8 — tightly sized, contains {-1, 0, 1}
        all_determined: (P, L) bool — tightly sized
    """
    N, L = dosage.shape
    P_max = N * (N - 1) // 2

    # Allocate worst-case buffers.  At N=320 this is ~10MB int8 for
    # all_values plus ~10MB bool for all_determined; manageable.
    all_values_buf = np.empty((P_max, L), dtype=np.int8)
    all_determined_buf = np.empty((P_max, L), dtype=np.bool_)
    # keep[slot] = True iff the pair at that slot passes both thresholds.
    # Zero-initialised so rejected / never-written slots stay False.
    keep = np.zeros(P_max, dtype=np.bool_)

    # Phase 1 (parallel over i): each pair writes its own fixed slot.
    for i in prange(N - 1):
        base = i * (N - 1) - (i * (i - 1)) // 2
        for j in range(i + 1, N):
            slot = base + (j - i - 1)
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

            # Pass 2: build the partial hap at this pair's slot in the
            # output buffer.  We're tentatively writing here; if
            # n_determined ends up below threshold we simply leave
            # keep[slot] False and Phase 2 skips the row.
            n_determined = 0
            for l in range(L):
                d_i = dosage[i, l]
                d_j = dosage[j, l]
                # Incompat site -> stay MASK (-1, not determined).
                # Original numpy code masks via `& ~inc_compat` and
                # leaves the -1 fill from np.full.
                if (d_i == 0 and d_j == 2) or (d_i == 2 and d_j == 0):
                    all_values_buf[slot, l] = -1
                    all_determined_buf[slot, l] = False
                # Non-incompat: forced_0 priority (matches numpy's
                # "partials[forced_0]=0 first, then partials[forced_1]=1"
                # — these are mutually exclusive after incompat masking,
                # so the elif order matches when one is True).
                elif d_i == 0 or d_j == 0:
                    all_values_buf[slot, l] = 0
                    all_determined_buf[slot, l] = True
                    n_determined += 1
                elif d_i == 2 or d_j == 2:
                    all_values_buf[slot, l] = 1
                    all_determined_buf[slot, l] = True
                    n_determined += 1
                else:
                    all_values_buf[slot, l] = -1
                    all_determined_buf[slot, l] = False

            if n_determined >= min_partial_determined:
                keep[slot] = True
            # else: keep[slot] stays False; Phase 2 discards this row

    # Phase 2 (serial): compact kept rows to the front in slot order
    # (= (i, j) enumeration order).  P <= slot always, so the in-place
    # row copy never clobbers a row we have not yet read.
    P = 0
    for slot in range(P_max):
        if keep[slot]:
            if P != slot:
                for l in range(L):
                    all_values_buf[P, l] = all_values_buf[slot, l]
                    all_determined_buf[P, l] = all_determined_buf[slot, l]
            P += 1

    return all_values_buf[:P], all_determined_buf[:P]


@njit(parallel=True, cache=True)
def _build_pairwise_partial_haps_soft_kernel(dosage, P0, P2, tau,
                                               max_conflict,
                                               min_partial_determined):
    """Soft-compatibility variant of _build_pairwise_partial_haps_kernel
    (PAIRWISE_RESIDUAL_MODE == "soft").

    The ONLY change from the argmax kernel is the pair-acceptance test in
    Pass 1.  Instead of counting hard argmax 0-vs-2 conflicts, it counts the
    number of sites where the POSTERIOR conflict probability
        P(0-vs-2 at l) = P0[i,l]*P2[j,l] + P2[i,l]*P0[j,l]
    exceeds tau (a "confident conflict" site), and admits the pair when that
    count is <= max_conflict.  At low read depth a diffuse posterior gives a
    small product (not a hard +1), so argmax noise no longer spuriously
    rejects truly-compatible pairs — which is what collapses the partial-hap
    pool to ~0 at low depth in the argmax kernel.  (Summing the per-site
    probabilities directly would overcount across all L sites; thresholding
    each site first is the correct analogue of the hard count.)

    Pass 2 — the partial-hap VALUES and determined mask — is byte-for-byte
    identical to the argmax kernel: argmax forcing (a hom call forces the
    shared strand) with per-site hard-incompat masking.  The cluster
    majority-vote denoises these values downstream, so only the gate needs
    softening.

    PARALLELISM: identical scheme to the argmax kernel — Phase 1 runs the
    outer i-loop under prange with each pair writing its DETERMINISTIC slot
        slot(i, j) = base(i) + (j - i - 1),
        base(i)    = i*(N-1) - i*(i-1)//2,
    so there is no write contention; a `keep` flag marks admitted slots;
    Phase 2 compacts kept rows to the front in slot (= (i, j)) order in
    place.  Row order and contents are byte-identical to the serial version.

    Args:
        dosage: (N, L) int64 — argmax dosages in {0, 1, 2} (Pass 2)
        P0, P2: (N, L) float64 — posteriors for dosage 0 and 2 (Pass 1)
        tau: float — per-site confident-conflict threshold
        max_conflict: int — max confident-conflict sites to admit a pair
        min_partial_determined: int — partial-rejection threshold

    Returns:
        all_values: (P, L) int8 in {-1, 0, 1}; all_determined: (P, L) bool
        — same shapes/semantics as the argmax kernel.
    """
    N, L = dosage.shape
    P_max = N * (N - 1) // 2

    all_values_buf = np.empty((P_max, L), dtype=np.int8)
    all_determined_buf = np.empty((P_max, L), dtype=np.bool_)
    # keep[slot] = True iff the pair at that slot passes both thresholds.
    keep = np.zeros(P_max, dtype=np.bool_)

    # Phase 1 (parallel over i): each pair writes its own fixed slot.
    for i in prange(N - 1):
        base = i * (N - 1) - (i * (i - 1)) // 2
        for j in range(i + 1, N):
            slot = base + (j - i - 1)
            # Pass 1 (soft): count confident-conflict sites; early-exit.
            n_conf = 0
            for l in range(L):
                cl = P0[i, l] * P2[j, l] + P2[i, l] * P0[j, l]
                if cl > tau:
                    n_conf += 1
                    if n_conf > max_conflict:
                        break
            if n_conf > max_conflict:
                continue

            # Pass 2 (argmax determination — identical to the argmax kernel).
            n_determined = 0
            for l in range(L):
                d_i = dosage[i, l]
                d_j = dosage[j, l]
                if (d_i == 0 and d_j == 2) or (d_i == 2 and d_j == 0):
                    all_values_buf[slot, l] = -1
                    all_determined_buf[slot, l] = False
                elif d_i == 0 or d_j == 0:
                    all_values_buf[slot, l] = 0
                    all_determined_buf[slot, l] = True
                    n_determined += 1
                elif d_i == 2 or d_j == 2:
                    all_values_buf[slot, l] = 1
                    all_determined_buf[slot, l] = True
                    n_determined += 1
                else:
                    all_values_buf[slot, l] = -1
                    all_determined_buf[slot, l] = False

            if n_determined >= min_partial_determined:
                keep[slot] = True

    # Phase 2 (serial): compact kept rows to the front in slot order
    # (= (i, j) enumeration order); P <= slot, so the in-place copy is safe.
    P = 0
    for slot in range(P_max):
        if keep[slot]:
            if P != slot:
                for l in range(L):
                    all_values_buf[P, l] = all_values_buf[slot, l]
                    all_determined_buf[P, l] = all_determined_buf[slot, l]
            P += 1

    return all_values_buf[:P], all_determined_buf[:P]


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

    Implementation: delegates to a numba kernel that uses scalar loops
    over (i, j) and writes directly into preallocated worst-case
    buffers.  Output preserves row order so the downstream
    `np.argsort(-determined_counts, kind='stable')` seed selection picks
    the same seeds.
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
        return empty_v, empty_d

    if PAIRWISE_RESIDUAL_MODE == "soft":
        # Soft compatibility gate: posterior confident-conflict count.
        # Determination (Pass 2) stays argmax inside the soft kernel.
        P0 = np.ascontiguousarray(probs[:, :, 0], dtype=np.float64)
        P2 = np.ascontiguousarray(probs[:, :, 2], dtype=np.float64)
        all_values, all_determined = \
            _build_pairwise_partial_haps_soft_kernel(
                dosage, P0, P2, float(PAIRWISE_SOFT_CONFLICT_TAU),
                int(max_pair_incompat), int(min_partial_determined))
    else:
        all_values, all_determined = \
            _build_pairwise_partial_haps_kernel(
                dosage, int(max_pair_incompat), int(min_partial_determined))

    if all_values.shape[0] == 0:
        empty_v = np.empty((0, L), dtype=np.int8)
        empty_d = np.empty((0, L), dtype=bool)
        return empty_v, empty_d

    return all_values, all_determined


@njit(cache=True)
def _cluster_all_seeds_kernel(all_values, all_determined, seed_order,
                               min_cluster_overlap,
                               max_cluster_disagreements,
                               max_iter, min_cluster_size):
    """Fused numba kernel for the whole pairwise clustering pass (Step 2 of
    pairwise_common_hap_recover).

    Replaces the previous structure — a Python `for seed_idx in seed_order`
    loop that called grow_cluster_iterative (and through it the single-seed
    kernel _grow_cluster_iterative_kernel) once per seed — with one kernel
    that runs the entire greedy seed-grow loop in nopython mode.  This
    removes the per-seed Python overhead that became significant once the
    soft compatibility gate enlarged the partial-hap pool: per seed the old
    path paid a wrapper call, two np.ascontiguousarray no-op checks on the
    full (P, L) arrays, an np.where(...).tolist() over (P,) bool, and a dict
    construction.  The per-seed scratch buffers (compatible, prev_compatible,
    final_compatible, compat_idx, and the consensus work arrays) are
    allocated ONCE here and reused across seeds instead of being
    re-allocated by every grow_cluster_iterative call.

    GREEDY SEED-GROW (unchanged semantics).  Seeds are visited in seed_order
    (the caller passes np.argsort(-determined_counts, kind='stable'), so the
    most informative partial-haps — those that determined the most bits —
    anchor clusters first).  A seed whose partial-hap was already claimed by
    an earlier cluster (available[seed] == False) is skipped.  Otherwise we
    grow a cluster from it and mark its members unavailable — this mutation
    happens for EVERY grown seed, including clusters that then prove too
    small to emit, exactly as the previous code mutated available[] inside
    grow_cluster_iterative BEFORE the Python min_cluster_size check.  A
    cluster is EMITTED (its consensus recorded) only when it has
    >= min_cluster_size members.

    SINGLE-CLUSTER GROW (inlined from the former _grow_cluster_iterative_
    kernel, byte-for-byte semantics):
      Grow a cluster from a seed partial-hap by iterative batch-merge with
      re-check: at each iteration, find every available partial-hap that's
      compatible with the current consensus; recompute consensus by majority
      vote across all such candidates; iterate until the compatible set
      stabilises.  "Compatible" = overlap (intersection of determined
      regions) >= min_cluster_overlap, AND disagreements (sites in the
      overlap where consensus and candidate differ) <= max_cluster_
      disagreements.  The re-check inside iteration is what handles the
      pathological case where two true strands collapse into the seed's
      compatibility set: after the first majority-vote consensus, candidates
      from the wrong strand fail the disagreement check and drop out.

      Optimizations preserved from the single-seed kernel (they replaced the
      original's (P, L) bool overlap/disagree_mask intermediates — ~5MB each
      per iteration at production size — with two scalar accumulators per
      partial-hap):
        1. EARLY TERMINATION in the compat-check inner L-loop: break as soon
           as disagreements > max_cluster_disagreements, since the partial-
           hap is already rejected and the rest of the L-pass only affects
           overlap_count (irrelevant once compatible[p] = False).  At
           production size in iter 2+ most partials ARE rejected (the
           cluster has converged), so this is the dominant fast path.
        2. COMPACT COMPATIBLE-INDEX in the votes pass: rather than scanning
           all P partials per site checking `if compatible[p]`, build a
           packed int64 array of just the compatible indices once per
           iteration and iterate that — reduces the votes pass from L*P to
           L*n_compat.

      Convergence semantics (faithfully reproduced from the Python version):
        - First iteration: prev_compatible unused; compute compatible[],
          snapshot into prev_compatible.
        - Later iterations: break if compatible == prev_compatible.
        - On convergence break: final_compatible retains the value set at
          the END of the previous committed iteration.
        - On empty-compat break (n_compat < 1): final_compatible retains
          the last committed value (or the seed-only fallback if none).
        - If no iteration ever committed (max_iter == 0, or first-iter
          compat empty): fall back to seed-only.
      Tiebreak in majority vote: votes_1 >= votes_0 -> 1 (matches the numpy
      `np.where(votes_1 >= votes_0, 1, 0)`).

    The per-seed scratch arrays do NOT need resetting between seeds:
    `compatible` is fully overwritten on the first iteration of each grow;
    `prev_compatible` is gated by have_prev (reset False per seed);
    `final_compatible` is fully written by either a committed iteration or
    the seed-only fallback — matching the fresh np.zeros the single-seed
    kernel allocated per call.

    Args:
        all_values: (P, L) int8 in {-1, 0, 1}
        all_determined: (P, L) bool
        seed_order: (P,) int64 — seed visitation order
        min_cluster_overlap: int
        max_cluster_disagreements: int
        max_iter: int
        min_cluster_size: int — minimum members required to EMIT a cluster

    Returns:
        cons_values: (n_clusters, L) int8 — per-emitted-cluster consensus
            (-1 marks sites NO member determined; the caller fills these by
            the population-frequency tiebreak)
        cons_determined: (n_clusters, L) bool
        cons_sizes: (n_clusters,) int64 — member count per emitted cluster
        cons_detcount: (n_clusters,) int64 — determined-bit count (= the
            old determined_count = int(cd.sum()))
    """
    P, L = all_values.shape

    available = np.ones(P, dtype=np.bool_)

    # Per-emitted-cluster output buffers (worst case: every seed emits, so
    # at most P clusters; members are exclusive across clusters so the total
    # emitted member count is also <= P).
    cons_values = np.empty((P, L), dtype=np.int8)
    cons_determined = np.empty((P, L), dtype=np.bool_)
    cons_sizes = np.empty(P, dtype=np.int64)
    cons_detcount = np.empty(P, dtype=np.int64)

    # Scratch reused across seeds.
    cur_values = np.empty(L, dtype=np.int8)
    cur_determined = np.empty(L, dtype=np.bool_)
    compatible = np.zeros(P, dtype=np.bool_)
    prev_compatible = np.zeros(P, dtype=np.bool_)
    final_compatible = np.zeros(P, dtype=np.bool_)
    new_values = np.empty(L, dtype=np.int8)
    new_determined = np.empty(L, dtype=np.bool_)
    compat_idx = np.empty(P, dtype=np.int64)

    n_clusters = 0

    for s in range(seed_order.shape[0]):
        seed_idx = seed_order[s]
        if not available[seed_idx]:
            continue

        # ---- grow one cluster from seed_idx (inlined single-seed kernel) ----
        # Copy seed into consensus
        for l in range(L):
            cur_values[l] = all_values[seed_idx, l]
            cur_determined[l] = all_determined[seed_idx, l]

        have_prev = False
        have_final = False

        for _it in range(max_iter):
            # Compute compatible[p] = available[p] AND overlap_count(p) >=
            # min_cluster_overlap AND disagreements(p) <= max_disagreements.
            # Early-exit the inner L-loop on disagreements > max — the
            # partial is rejected regardless of the rest of the L-pass.
            for p in range(P):
                if not available[p] and p != seed_idx:
                    # Claimed by an earlier cluster.  (seed_idx is always
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
            # index array rather than all P partials.
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

        # If no iteration ever set final_compatible (e.g. max_iter == 0, or
        # first iter had compat_count < 1 with no prior commit), fall back
        # to seed-only.  Matches the Python original.
        if not have_final:
            for p in range(P):
                final_compatible[p] = False
            final_compatible[seed_idx] = True

        # Mutate available in-place — claim this cluster's members.  This
        # happens for every grown seed, even if the cluster is too small to
        # emit (matches the old available[] mutation inside the per-seed
        # kernel, which ran before the Python min_cluster_size check).
        n_members = 0
        for p in range(P):
            if final_compatible[p]:
                available[p] = False
                n_members += 1
        # ---- end grow ----

        # Emit the cluster's consensus only if it reached min_cluster_size.
        if n_members >= min_cluster_size:
            for l in range(L):
                cons_values[n_clusters, l] = cur_values[l]
                cons_determined[n_clusters, l] = cur_determined[l]
            det_cnt = 0
            for l in range(L):
                if cur_determined[l]:
                    det_cnt += 1
            cons_sizes[n_clusters] = n_members
            cons_detcount[n_clusters] = det_cnt
            n_clusters += 1

    return (cons_values[:n_clusters], cons_determined[:n_clusters],
            cons_sizes[:n_clusters], cons_detcount[:n_clusters])


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
    all_values, all_determined = build_pairwise_partial_haps(
        probs,
        max_pair_incompat=max_pair_incompat,
        min_partial_determined=min_partial_determined)
    P = all_values.shape[0]
    if P == 0:
        return [], 0

    # Step 2: cluster by greedy seed-grow.  Seeds chosen by descending
    # determined-count (the most informative partial-haps anchor the
    # cluster well).  The entire seed loop AND the per-seed single-cluster
    # grow now run inside one fused numba kernel (_cluster_all_seeds_kernel)
    # instead of a Python loop calling grow_cluster_iterative per seed; see
    # that kernel's docstring.  Output (emitted clusters, their consensus
    # values/determined masks, sizes, determined counts) is identical to the
    # previous per-seed implementation.
    determined_counts = all_determined.sum(axis=1)
    seed_order = np.argsort(-determined_counts, kind='stable').astype(np.int64)

    # Defensive dtype/contiguity: the fused kernel expects int8 / bool
    # C-contiguous arrays.  build_pairwise_partial_haps already returns
    # these (a row-slice of a C-contiguous buffer stays C-contiguous), so
    # these are no-ops in practice.
    all_values_arr = np.ascontiguousarray(all_values, dtype=np.int8)
    all_determined_arr = np.ascontiguousarray(all_determined, dtype=np.bool_)

    (cons_values, cons_determined, cons_sizes, cons_detcount) = \
        _cluster_all_seeds_kernel(
            all_values_arr, all_determined_arr, seed_order,
            int(min_cluster_overlap), int(max_cluster_disagreements),
            int(max_cluster_grow_iter), int(min_cluster_size))

    consensus_haps = []
    for c in range(cons_sizes.shape[0]):
        consensus_haps.append({
            'cv': cons_values[c],
            'cd': cons_determined[c],
            'cluster_size': int(cons_sizes[c]),
            'determined_count': int(cons_detcount[c]),
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
def _count_carriers_soft_kernel(hap, P0, P2, tau, max_incompat):
    """Soft carrier count for apply_quality_filters Filters B/D/E when
    PAIRWISE_RESIDUAL_MODE == "soft".

    Soft analogue of _count_carriers_kernel.  A sample s carries hap iff the
    number of CONFIDENT incompatibility sites is <= max_incompat, where a
    site l is a confident incompatibility when the posterior probability of
    the offending genotype exceeds tau:
        hap[l] == 0:  offending = hom-alt (dosage 2)  -> P2[s, l] > tau
        hap[l] == 1:  offending = hom-ref (dosage 0)  -> P0[s, l] > tau
    This is the carrier-count analogue of the soft pair-compatibility gate:
    at low depth a diffuse posterior gives a small offending-genotype
    probability, not a hard argmax incompatibility, so argmax noise no
    longer destroys the (strict, max_incompat=0) carrier counts Filters D/E
    depend on.  Early-exits the inner loop on n_incompat > max_incompat.

    Args:
        hap: (L,) int64 — fully resolved hap in {0, 1}
        P0, P2: (N, L) float64 — posteriors for dosage 0 and 2
        tau: float — per-site confident-incompatibility threshold
        max_incompat: int — slack threshold (0 for strict carriers)

    Returns:
        int — number of carrier samples
    """
    N, L = P0.shape
    carrier_count = 0
    for s in range(N):
        n_incompat = 0
        is_carrier = True
        for l in range(L):
            if hap[l] == 0:
                p_off = P2[s, l]
            else:
                p_off = P0[s, l]
            if p_off > tau:
                n_incompat += 1
                if n_incompat > max_incompat:
                    is_carrier = False
                    break
        if is_carrier:
            carrier_count += 1
    return carrier_count


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
                           max_lenient_excess_ratio=MAX_LENIENT_EXCESS_RATIO,
                           probs=None):
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

    # In "soft" mode (and only when probs is provided) the lenient and
    # strict carrier counts are computed from the genotype POSTERIOR
    # (_count_carriers_soft_kernel) rather than the argmax dosage, so low-
    # depth argmax noise no longer destroys the strict-carrier counts Filters
    # D/E rely on.  "argmax" mode (or probs=None) leaves the counts — and
    # therefore Filters B/D/E — byte-identical to before.
    use_soft = (PAIRWISE_RESIDUAL_MODE == "soft") and (probs is not None)

    # PERFORMANCE (soft path): the soft carrier count needs the dosage-0 and
    # dosage-2 posterior planes.  Slicing them out of the (N, L, 3) array is
    # a strided view, so np.ascontiguousarray must allocate and copy a full
    # (N, L) float64 array.  We call _count_carriers_soft_kernel twice per
    # candidate (lenient + strict), so computing those planes per call would
    # be 4 * len(recovered) full-array copies — which dominated apply_quality_
    # filters at production size.  The planes are identical across all
    # candidates, so we slice them ONCE here and pass them into the kernel.
    if use_soft:
        P0_soft = np.ascontiguousarray(probs[:, :, 0], dtype=np.float64)
        P2_soft = np.ascontiguousarray(probs[:, :, 2], dtype=np.float64)
        tau_soft = float(PAIRWISE_SOFT_CONFLICT_TAU)

    # Step 1 — annotate every candidate with its carrier count.  We
    # compute this once per candidate so subsequent filter steps can
    # reuse the value.
    enriched = []
    for r in recovered:
        if use_soft:
            hap_arr = np.ascontiguousarray(r['hap'], dtype=np.int64)
            carriers = _count_carriers_soft_kernel(
                hap_arr, P0_soft, P2_soft, tau_soft,
                int(max_carrier_incompat))
        else:
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
        if use_soft:
            hap_arr = np.ascontiguousarray(r2['hap'], dtype=np.int64)
            r2['strict_carriers'] = _count_carriers_soft_kernel(
                hap_arr, P0_soft, P2_soft, tau_soft, 0)
        else:
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
# Top-level production entry point.  Used by block_haplotypes's
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
    survivors, rejected = apply_quality_filters(recovered, dosage, probs=probs_k)
    if verbose:
        print(f'[pairwise] {len(recovered)} raw candidates -> '
              f'{len(survivors)} after filters '
              f'({len(rejected)} rejected; n_partials={n_partials})')
    return [r['hap'] for r in survivors]