"""Centralized tunable configuration for the bhd_* haplotype-recovery ecosystem.

Single source of truth for the pipeline's tunable knobs, thresholds, and feature
flags.  Logic-free: imports nothing, so every bhd_* module (and block_haplotypes)
can pull its constants from here without import cycles.  Low-level numerical
sentinels that are not user-tuned (bhd_kernels.MASK, bhd_kernels.LOG_EPS) and the
per-module HAS_NUMBA capability flag deliberately stay in their home modules.
"""


# ============================================================================
# Viterbi scoring & similarity-band tuning  (sentinels MASK / LOG_EPS stay in bhd_kernels)
# ============================================================================
# Default wildcard penalty.  λ in log-likelihood units per (strand, site)
# wildcard usage.  Sites where the real founder pair gives a likelihood at
# least 1/e^(2λ) ≈ 0.37 of the wildcard's optimal genotype likelihood
# prefer real founders; below that, wildcards take over.  λ=0.5 puts the
# crossover at "real wins until likelihood ratio of (best wildcard /
# real-pair) exceeds e^1 ≈ 2.7."
DEFAULT_LAMBDA = 0.5
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
# Pooled alt-allele fraction cut-points for calling a consensus dosage in
# {0, 1, 2} from a cluster's pooled-alt vector (see pooled_alt_to_dosage).
# A homozygous-ref pair-type pools to ~0, het to ~0.5, homozygous-alt to ~1.
# 0.25 / 0.75 place the {0|1} and {1|2} boundaries midway between those
# modes, matching the natural 0 / 0.5 / 1 structure of the expected alt
# fraction.  Symmetric about 0.5.
POOLED_ALT_LO = 0.25
POOLED_ALT_HI = 0.75
# Het-band edges for the cluster-homozygosity score (see
# cluster_homozygosity_score).  A site whose pooled-alt sits in
# (CLUSTER_HOM_BAND_LO, CLUSTER_HOM_BAND_HI) is counted as a het site of the
# cluster's pair-type; the score is 1 minus the fraction of such sites, so a
# genuinely homozygous cluster (no het sites) scores ~1 and a heterozygous
# pair-type (about half its sites het) scores ~0.5.  The (0.35, 0.65) band
# is narrower than the (0.25, 0.75) dosage band on purpose: the score should
# count only sites that are clearly intermediate (near 0.5), not sites near
# the 0.25 / 0.75 dosage boundaries that are still essentially hom calls.
CLUSTER_HOM_BAND_LO = 0.35
CLUSTER_HOM_BAND_HI = 0.65

# ============================================================================
# Pool emission cache
# ============================================================================
# Per-cache byte budget for the precomputed full-pool emission tensor built
# in PoolEmissionCache.__init__.  When the tensor would exceed this, the
# cache skips the allocation and falls back to on-the-fly per-subset scoring
# (see the MEMORY GUARD block in __init__).  The tensor's state axis is
# K_states = K_pool*(K_pool+1)//2 + K_pool + 1, i.e. O(K_pool^2); at the
# small pools (<= ~30) that block discovery produces at normal read depth it
# is a ~30 MB speed win, but at low read depth the trio/pairwise candidate
# pool can balloon to thousands and the tensor then runs to hundreds of GiB
# (K_pool=3398, N=320, n_bins=20 -> 276 GiB), OOM-killing the worker.
# 256 MiB caps a single cache's tensor at K_pool ~ 100 (at N=320, n_bins=20).
# Block discovery runs up to n_processes workers concurrently, each of which
# may hold one seed-trim cache, so the worst-case aggregate is bounded by
# ~256 MiB * n_processes (~29 GiB at 112 workers) rather than the unbounded
# hundreds-of-GiB a single low-depth block could otherwise demand.  Tunable:
# raise it on large-memory nodes to keep the fast path for bigger pools,
# lower it when running many workers on a memory-tight node.
POOL_EMISSION_CACHE_MAX_BYTES = 256 * 1024 * 1024

# ============================================================================
# Trio-based recovery
# ============================================================================
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

# ============================================================================
# Pairwise / partial-determination recovery
# ============================================================================
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

# ============================================================================
# Residual-subtraction & Bernoulli-mixture recovery
# ============================================================================
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

# ============================================================================
# Block-haplotype seeding / medoid multistart
# ============================================================================
# Number of cluster-seed starts at the initial K=0 -> K=1 transition.
# Reducing to 1 disables multi-start (recovers single-start behavior).
# Increasing improves robustness on blocks with very heterogeneous truth
# founders but costs proportionally more time.
K_MEDOID_STARTS_DEFAULT = 5
# Minimum sample count for multi-start to be applied.  Below this we fall
# back to a single branch (soft clustering needs at least a few samples
# per pair-type to form clusters).
MEDOID_MIN_N_FOR_MULTISTART = 3
# HDBSCAN minimum cluster size for the soft seed front-end — the minimum
# number of samples that must share a pair-type for that pair-type to seed a
# branch.  Mirrors the trio front-end's TRIO_SOFT_MIN_CLUSTER_SIZE.  Other
# HDBSCAN parameters use the library defaults, matching block_haplotypes.py's
# precomputed-metric usage.
SEED_SOFT_MIN_CLUSTER_SIZE = 3