"""block_haplotypes_discrete.py — Discrete-hap founder discovery (orchestrator)

The top-level orchestration layer of stage-3 block-haplotype founder
discovery.  As of the 4-file split, this file contains:

  - Multi-start constants (K_MEDOID_STARTS_DEFAULT,
    MEDOID_MIN_N_FOR_MULTISTART, SEED_SOFT_MIN_CLUSTER_SIZE)
  - Forkserver pool scaffolding (_ForkserverPool, _init_block_worker,
    _BH_ACTIVE_COUNTER, _BH_TOTAL_CORES)
  - _grow_K — the main K-growth coordinate-descent orchestrator
  - _soft_cluster_seed_haps, _initial_kgrowth_with_medoids — soft-cluster
    multi-start seeding
  - _grow_K_with_recovery — top-level entry that combines K-growth,
    trio recovery, pairwise recovery, subtraction recovery, and late
    low-carrier rescue
  - Output construction (_compute_per_site_confidence,
    _discrete_haps_to_prob_arrays)
  - _final_cleanup — chimera pruning + Viterbi subset selection on
    the prob-array form
  - generate_haplotypes_block, generate_haplotypes_block_robust —
    the per-block public entry points
  - _worker_generate_block_direct, generate_all_block_haplotypes —
    the multi-process orchestrator over all blocks of a contig

The atomic BIC/CD primitives, subtraction recovery, trio recovery, and
pairwise recovery are now in bhd_kernels.py, bhd_recovery.py,
bhd_trio.py, and bhd_pairwise.py respectively.  This file is
import-only-downstream — nothing in the 4 split files imports back
from here, which keeps the dependency DAG acyclic.

Public callers (e.g. pipeline.py) continue to use:
    import block_haplotypes_discrete as bhd
    bhd.generate_haplotypes_block(...)
    bhd.generate_all_block_haplotypes(...)
"""

import numpy as np
import math
import multiprocessing as mp
import multiprocessing.pool
import warnings
import gc
import ctypes

from numba import njit, prange

import thread_config

import analysis_utils
import hap_statistics

# Cross-module imports from the 4 split bhd subsystems.  The atomic
# BIC/CD kernel, recovery pipeline, trio recovery, and pairwise common-
# hap recovery each live in their own file; we import what we use here.
import bhd_kernels
import bhd_recovery
import bhd_trio
import bhd_pairwise

# Explicit named imports for symbols used directly in this file's
# function bodies (function/constant references that don't need
# runtime mutation).  Cross-module ENABLED-flag reads use module-
# attribute lookup (e.g. bhd_pairwise.PAIRWISE_RECOVERY_ENABLED) to
# preserve runtime-mutation semantics.
from bhd_kernels import (
    MASK,
    DEFAULT_LAMBDA,
    LOG_EPS,
    _safe_neg_log,
    _init_hap_from_sample_dosage,
    _select_initial_seed,
    _fit_at_fixed_K,
    _update_A,
    _compute_cc,
    _compute_bic,
    PoolEmissionCache,
)
from bhd_recovery import (
    RECOVERY_HAPS_EQUAL_EPS_PCT,
    RECOVERY_OUTER_CC_SCALE,
    RECOVERY_MAX_K,
    RECOVERY_MAX_ROUNDS,
    RECOVERY_INTRA_ROUND_DEDUP_PCT,
    RECOVERY_MIXTURE_K_MAX,
    RECOVERY_MIXTURE_N_RESTARTS,
    RECOVERY_MIXTURE_RNG_SEED,
    RECOVERY_MIXTURE_PATIENCE,
    RECOVERY_CLEANNESS_THRESHOLD,
    RECOVERY_SWAP_NLL_TOLERANCE,
    RECOVERY_MAX_OUTER_ITERATIONS,
    RECOVERY_LOW_CARRIER_TRIGGER_FRAC,
    RESIDUAL_TRIO_ENABLED,
    _greedy_bic_select,
    _swap_refine,
    _bic_prune,
    _subtraction_recovery_round_loop,
    _late_low_carrier_rescue,
    _residual_trio_rescue,
    _hamming_pct_kept,
    _haps_equal,
)
from bhd_trio import _trio_recovery_candidate_haps

# prune_chimeras + viterbi_score_selection are the scoring / chimera-pruning
# kernels, now living in the bhd_kernels leaf (migrated out of the retired
# legacy block_haplotypes.py); _update_dynamic_threads is the shared dynamic-
# thread hook, also in bhd_kernels, used by select_optimal_haplotype_set_viterbi.
from bhd_kernels import (
    prune_chimeras,
    viterbi_score_selection,
    _update_dynamic_threads,
)
# BlockResult, BlockResults, consolidate_similar_candidates, and
# select_optimal_haplotype_set_viterbi were migrated out of block_haplotypes.py
# into this module — see the "MIGRATED FROM block_haplotypes.py" section at the
# end of the file.  (find_missing_haplotypes_iterative is still imported lazily
# from the legacy module inside the residual loop, where present.)

warnings.filterwarnings("ignore")
np.seterr(divide='ignore', invalid='ignore')


# =============================================================================
# CONSTANTS — INITIAL-K-GROWTH MEDOID MULTI-START
# =============================================================================
#
# At the K=0 -> K=1 transition, the M-step's voting at K=1 (where every
# sample is paired with the wildcard founder) is dominated by population
# allele frequency.  Whichever sample is chosen as the K=1 seed, CD
# pulls H[0] toward the population-majority haplotype.  When truth
# founders are heterogeneous and population-majority is itself a chimera
# of multiple truths, CD locks in a chimera at K=1.  Subsequent K-growth
# steps generate candidates as `np.clip(worst_dosage - F_i, 0, 1)`; if
# the F_i are chimeras, the candidates are also chimera-shaped (no real
# sample's strand matches them), so CD can't refine them, and BIC
# rejects further K-growth.  The trajectory is permanently trapped in
# the chimera basin.
#
# Diagnostic on chr17:29157296 confirmed: 156/320 sample seeds land
# K-growth in the truth basin (NLL=287.9, all 6 truths recovered at
# 0.0%); 164/320 sample seeds land in the chimera basin (NLL=31654.3,
# all 6 truths missed at ~21% Hamming).  Default seed selection picks
# the most-decisive sample in the all-WW case at K=0; "decisiveness"
# (sum of argmax-genotype-probabilities) is a coverage-quality measure,
# not a basin-membership predictor, so the default's chosen seed is in
# the chimera basin on these blocks.
#
# Fix: deterministic multi-start with soft-clustering seeds.  At the
# K=0 -> K=1 transition, instead of relying on a single most-decisive
# sample, we generate up to K_MEDOID_STARTS DIVERSE seed haps and run
# full K-growth from each as a separate H_init, then pick the trajectory
# with lowest final BIC = K_final * cc + 2 * NLL_final.  BIC (not raw
# NLL) is the right cross-K comparison criterion: it penalises
# trajectories that grew to a larger K than the data justifies, so
# multi-start naturally prefers parsimonious solutions of equal data-fit
# quality.
#
# The diverse seeds come from the posterior soft-clustering front-end
# (_soft_cluster_seed_haps): samples are clustered on the expected-
# genotype-agreement similarity (bhd_kernels.soft_agreement_similarity)
# via HDBSCAN, and the largest clusters' per-site pooled-alt consensus
# haps (bhd_kernels.alt_fractions -> pooled_alt_to_hap) become the seeds.
# This guarantees representation from each truth-progenitor cluster (the
# property that makes multi-start work — confirmed empirically: top-K-
# decisive seeds do NOT correlate with basin membership, e.g. chr17
# needed K=20 top-decisive samples to find a truth-basin seed) while
# keeping the posteriors rather than hard-calling them, so the seeds stay
# robust at low read depth.  A cluster's pooled consensus also averages
# out the per-sample het / zero-coverage noise that a single argmax
# sample-seed carries, and selection no longer depends on per-sample
# "decisiveness" (which collapses at low depth).  Same rationale and
# shared primitives as the trio soft front-end (bhd_kernels.py); the
# hdbscan / bhd_kernels imports are performed lazily inside
# _soft_cluster_seed_haps.
#
# Cost: K_MEDOID_STARTS x full K-growth instead of 1x.  At default
# K=5 seeds, stage 3 cost is roughly 5x the single-trajectory cost.
# Per-block parallelism unchanged.

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
# =============================================================================
# REJECTED EXPERIMENT — POST-CD TWO-STEP REFINEMENT
# =============================================================================
#
# CD converges to a JOINT local minimum of NLL(H, A): a state where
# neither the M-step (H update at fixed A) nor the E-step (A update at
# fixed H) decreases NLL.  Such a state is a fixed point of CD but not
# necessarily a local minimum of f(H) = min_A NLL(H, A).
#
# Concrete example, chr4:1695146 (founder t5, site 14): the converged
# state has H[d2, 14] = 1 with M-step margin +5 against bit=0; but
# flipping the bit AND letting _update_A re-assign reduces NLL by ~170,
# and re-running full CD from there drops NLL by ~1300 total.  CD got
# trapped in a strictly-worse basin because the bit-flip and A-update
# were jointly coupled and neither single-coordinate move alone took
# the first step.
#
# We implemented "two-step refinement": post-CD steepest descent over
# single-bit flips, scoring each flip by f(H_flipped) = min_A
# NLL(H_flipped, A) (one E-step per evaluation).  Tested on the 261
# failing blocks of the seed=50 benchmark:
#
#   Runtime: 1:09 (no refinement) -> 5:12 (with refinement) = 4.5x slower.
#   Accuracy:
#     - 13/261 failing blocks moved 5/6 -> 6/6 (5% of failing,
#       0.029% absolute improvement on the full 44794-block benchmark).
#     - 0/5 of the 4/6 blocks improved (those are K-compromise cases
#       that single-bit refinement can't fix).
#     - avg_true_match_err: 0.453% -> 0.450% (effectively unchanged).
#
# Cost-vs-gain ratio rejected.  If revisited, the cost would need to
# come down ~10x (e.g. only refine bits with small M-step margin, on
# the order of cc/2, since those are the only plausibly near-tipping
# bits) OR the gain would need to land on the harder failures (4/6
# blocks), which it structurally cannot since those are
# K_alg < K_truth compromises rather than single-bit issues.

# =============================================================================
# FORKSERVER POOL SCAFFOLDING (mirrors block_haplotypes_em_foothold.py)
# =============================================================================

try:
    _libc = ctypes.CDLL("libc.so.6")
    def _malloc_trim():
        _libc.malloc_trim(0)
except OSError:
    def _malloc_trim():
        pass


try:
    _forkserver_ctx = mp.get_context('forkserver')
except ValueError:
    _forkserver_ctx = mp.get_context('fork')


class _ForkserverPool(multiprocessing.pool.Pool):
    """Pool using forkserver context."""
    def __init__(self, *args, **kwargs):
        kwargs['context'] = _forkserver_ctx
        super().__init__(*args, **kwargs)


_BH_ACTIVE_COUNTER = None
_BH_TOTAL_CORES = None


def _init_block_worker(total_cores, active_counter, extra_counter=None):
    """Initializer for worker processes — sets up dynamic numba thread
    allocation based on number of currently-active workers.

    Wires both this module's _BH_* globals (used to increment/decrement the
    active count around each block) and thread_config's shared dynamic-thread
    state, which is read by thread_config.apply_dynamic_threads() at every
    phase boundary across this module + bhd_recovery + bhd_trio.  That lets a
    straggler block GROW into cores freed as its peers finish, instead of
    being pinned for its whole run to the thread count it got at start.
    extra_counter drives the remainder distribution (total threads in use ==
    total_cores, zero idle cores); None falls back to floor-only."""
    global _BH_ACTIVE_COUNTER, _BH_TOTAL_CORES
    _BH_ACTIVE_COUNTER = active_counter
    _BH_TOTAL_CORES = total_cores
    try:
        import os, numba
        os.environ['NUMBA_NUM_THREADS'] = str(total_cores)
        numba.config.NUMBA_NUM_THREADS = total_cores
        numba.set_num_threads(1)
    except Exception:
        pass
    # Wire the shared state so every phase boundary across this module +
    # bhd_recovery + bhd_trio re-checks the SAME pool-wide active count.
    thread_config.set_dynamic_thread_state(total_cores, active_counter, extra_counter)

def _grow_K(probs_k, kept_mask_full, lam,
            wildcard_mass_threshold=0.0,
            min_relative_improvement=0.10,
            K_max=10,
            max_iter_per_K=50,
            known_haps_full=None,
            cc_scale=0.5,
            use_log_bic=False,
            min_nll_improvement=1e-6,
            H_init=None):
    """Iteratively grow K, starting at K=0 (empty founder set), seeding
    each new founder from the current worst-fit sample's subtraction
    against existing founders.  Stops when either:
      - a candidate new founder fails to improve the BIC score
        (NLL improvement does not exceed the per-founder complexity cost), OR
      - K_max is reached.

    Acceptance criterion history:
      v1: relative wildcard-mass improvement >= 10%.  Rejected real
          founders whose individual contribution was small but whose
          combination with later founders unlocked substantial
          improvement.
      v2: strict-positive absolute wildcard-mass improvement (>= 1e-6).
          Fixed the v1 problem but introduced a new one: when a new
          real founder lets samples upgrade from (close-to-truth, W) to
          (exact-truth, W), the wildcard slot count is unchanged (still
          1 per upgraded sample) even though NLL drops substantially.
          The wildcard-mass criterion missed these improvements.
      v3: strict-positive NLL improvement (>= 1e-6).  Fixed v2 but
          accepted ANY positive NLL improvement, including spurious
          K-additions that absorbed a small amount of noise.  This
          showed up in the benchmark as small but non-zero K=7+ blocks
          past truth K=6, with reduced quality at the over-grown K.
      v4: linear-BIC-based acceptance.  A new founder is
          accepted iff adding it strictly reduces the BIC score
              BIC(K) = K * cc + 2 * NLL_K
          where cc = cc_scale * (L_kept/200) * N is the per-founder
          complexity cost (linear in N as in the project's
          beam_search_core / chimera_resolution standard).  This
          requires NLL_improvement > cc/2 to accept, calibrated so
          spurious noise-absorbing founders are rejected while real
          founders (which typically save thousands of NLL) easily pass.

          Linear BIC is preferred over standard log-BIC for the same
          reason as in beam_search_core: log(N) scaling is too weak
          when N is large, allowing founder explosion.  See
          chimera_resolution.compute_cc for the project-wide formula.

          v4 still had a top-of-loop early-stop on wildcard_mass <=
          wildcard_mass_threshold (default 0.0).  See v5 for why it
          was removed.
      v5 (current): wildcard-mass-based early-stop REMOVED.  BIC alone
          (v4 acceptance criterion) decides when to stop.  Diagnosed
          on chr3:27772468 (a 4/6 K-collapse case): wildcard_mass=0
          was firing at K=4 even though BIC overwhelmingly justified
          continuing — truth K=6 had BIC=379 while alg K=4 had
          BIC=12585, a 12,200-unit gap.  The mechanism: with LAM=0.5
          and L_kept=200, the wildcard penalty per slot is ≈100 NLL,
          so a sample with (real, real) cost up to ≈100 will pick
          (real, real) over (real, W) even when the (real, real) fit
          is terrible.  wm=0 then means "no sample chose a wildcard
          pair under current cost arithmetic", NOT "every sample is
          well fit".  K-growth was stopping on a misleading signal.
          BIC's NLL_improvement > cc/2 check is the principled stop;
          if all samples are truly well-fit, the next candidate's
          NLL drop will be small and BIC will reject naturally.

    Arguments:
        probs_k: (N, L_kept, 3) — kept-site posteriors
        kept_mask_full: (L_full,) bool — for downstream tracking; not used here
        lam:    wildcard penalty
        wildcard_mass_threshold: float — RETAINED for backward
            compatibility and signature stability.  Since v5, this
            parameter does NOT control K-growth's stopping behavior
            (BIC alone decides when to stop).  It is still consumed
            elsewhere in the pipeline (e.g. the uncertainty flag in
            generate_haplotypes_block uses
            `wildcard_mass > 2 * wildcard_mass_threshold` to mark
            blocks with high wildcard usage as untrustworthy).
        min_relative_improvement: float — RETAINED for backward compatibility
            but no longer used; current criterion is BIC-based.
        K_max: int — hard cap on K
        known_haps_full: list of (L_full,) binary arrays or None — if given,
            these are used as the initial K founders and never updated.
            (Note: they're passed as full-length arrays; we slice to kept.)
            **Currently unused** — known_haplotypes integration is a v2
            feature; ignored for now.
        cc_scale: float — complexity-cost scale (per-founder per-sample
            per-200-SNPs).  Default 0.5.  Higher values penalise extra
            founders more strongly.

            Calibration history:
              cc_scale=0.5 (project-wide default in chimera_resolution.py
              and beam_search_core.py): too aggressive at the per-block
              EM stage.  Empirically rejected real founders saving
              50-80 NLL (e.g., founders with 20-30 carriers showing as
              "founder upgrade" type savings rather than full wildcard-
              slot reduction).  Benchmark dropped 70.8% → 69.5%
              all-found, with 250+ blocks regressing from K=6
              (mostly-recovered) to K=4 (multiple-founders-missed).

              cc_scale=0.05 (current default): threshold of cc/2 =
              0.05 * (L/200) * N / 2 ≈ 8 NLL for typical N=320, L=200.
              This is just above floating-point noise: rejects pure
              "noise absorption" K-additions saving < 8 NLL while
              preserving every realistic founder addition (real
              founders typically save 50+ NLL per K transition).

              cc_scale=0.0 effectively disables the BIC penalty,
              reverting to strict-positive NLL improvement criterion
              with only the min_nll_improvement floor in effect.

            Note: the project's default 0.5 in beam_search_core was
            calibrated for whole-genome long-haplotype assembly, where
            the complexity penalty needs to suppress recombinant-
            founder false-positives across thousands of blocks.  At
            the per-block EM stage we operate on a much smaller scale
            and need gentler regularisation.

            Update (May 2026): default reverted to 0.5 to match the
            project-wide standard, after diagnosing the K-growth /
            recovery oscillation at chr3:16378549.  The historical
            70.8% → 69.5% regression noted above (and the 250+ block
            K-collapse from cc_scale=0.5) was confounded by other
            pipeline bugs that have since been fixed:
              - The wm-stop bug in K-growth (fixed by removing the
                wildcard_mass <= threshold short-circuit; lifted
                benchmark to 99.82% all-found).
              - Viterbi-BIC subset selection in _final_cleanup
                (Step B), with a per-hap inclusion penalty ~7×
                stricter than discrete-CD's BIC, systematically
                dropped legitimate low-frequency-carrier founders
                (disabled).
              - Chimera pruning in _final_cleanup (Step D), with
                mean_delta protection scaled to carrier frequency,
                systematically dropped legitimate low-frequency
                founders (disabled).
              - Step C usage threshold of max(2, 1% of N) = 3 for
                N=320 dropped legitimate founders with usage = 2
                strands (lowered to 1).
            With those pipeline issues removed, the project-standard
            cc_scale=0.5 is the principled choice — it removes the
            asymmetry between K-growth and recovery's outer BIC, and
            relies on K-growth's BIC at cc/2 ≈ 80 NLL as the
            authoritative data-justification filter for each founder
            (with the recovery's mixture-derived candidate pool
            providing the diversity of seeds K-growth's worst-fit
            seeding alone might miss).
        use_log_bic: bool — if True, use standard BIC with log(N*L) scaling
            instead of linear scaling.  Default False (linear, project standard).
        min_nll_improvement: float — additional numerical-noise floor.
            Effective threshold is max(min_nll_improvement, cc/2).
        H_init: optional (K_init, L_kept) array — if provided, K-growth
            starts from these K_init founders rather than from K=0
            (empty set).  H_init is treated as MUTABLE: the initial
            _fit_at_fixed_K call refines them via coord descent before
            any growth attempts.  This supports the outer K-growth ↔
            recovery iteration in _grow_K_with_recovery, where each
            K-growth call starts from the previous recovery output.
            Default None = original empty-set behaviour.

    Returns:
        H:               final (K, L_kept)
        A:               final (N, 2)
        per_sample_cost: (N,)
        wildcard_slots:  (N,)
        K_final:         int
        wildcard_mass:   float in [0, 1]
        history:         list of (K, BIC, wildcard_mass, n_iter) per growth step
                         (BIC = K * cc + 2 * NLL with the same cc used
                         in the acceptance criterion; comparable across K)
    """
    N, L_kept, _ = probs_k.shape
    history = []

    # Defensive guard: N=0 (no samples) means there's nothing to fit or
    # grow.  Return an empty result early.  Without this, _grow_K would
    # later call _select_initial_seed in the K=0 → K=1 fallback path,
    # which crashes on `argmax of empty sequence` because there's no
    # sample to score for decisiveness.  In production, the upstream
    # generate_haplotypes_block guards N=0 at the top level, but _grow_K
    # is also called directly (e.g., from the outer recovery loop) so
    # the guard belongs here for defense in depth.
    if N == 0:
        H_out = (np.empty((0, L_kept), dtype=np.int64) if H_init is None
                 else np.asarray(H_init, dtype=np.int64).copy())
        return (H_out,
                np.empty((0, 2), dtype=np.int64),         # A
                np.empty((0,), dtype=np.float64),         # per_sample_cost
                np.empty((0,), dtype=np.int64),           # wildcard_slots
                H_out.shape[0],                            # K_final
                0.0,                                        # wildcard_mass
                history)

    # === BIC-based acceptance threshold ===
    # Linear BIC: cc = cc_scale * (L_kept/200) * N
    # Standard BIC: cc = cc_scale * log(N * L_kept) * (L_kept/200)
    # Acceptance criterion: BIC(K+1) < BIC(K)
    #   => K*cc + 2*NLL_K > (K+1)*cc + 2*NLL_{K+1}
    #   => 2*(NLL_K - NLL_{K+1}) > cc
    #   => NLL_improvement > cc/2
    # Effective threshold combines BIC term with a numerical-noise floor.
    cc = _compute_cc(cc_scale, N, L_kept, use_log_bic=use_log_bic)
    bic_threshold = cc / 2.0
    accept_threshold = max(min_nll_improvement, bic_threshold)

    # === K=K_init baseline: empty founder set, or given H_init ===
    #
    # Per philosophical principle: start with NO founders at all (or
    # with the given H_init founders), measure how badly we fit the
    # data with pure wildcards (or with H_init).  Each subsequent K is
    # grown by introducing one new founder, warm-starting existing
    # founders from the previous K's converged values, and re-running
    # coordinate descent (no founders are held fixed).  This gives every
    # K transition the same uniform shape (K → K+1 grow + fit + accept).
    #
    # When H_init is None or empty, the baseline is K=0 (empty founder
    # set, all samples assigned to (W, W)) — the original behaviour.
    # When H_init is provided, the baseline is K=K_init (founders
    # refined via coord descent) — used by _grow_K_with_recovery to
    # continue growth after a recovery pass produced a better starting
    # set.
    if H_init is None or len(H_init) == 0:
        H = np.empty((0, L_kept), dtype=np.int64)
    else:
        H = np.asarray(H_init, dtype=np.int64).copy()
        if H.shape[1] != L_kept:
            raise ValueError(
                f"H_init has L={H.shape[1]} but probs_k has L_kept={L_kept}")
    H, A, per_sample_cost, wildcard_slots, n_iter, nll = _fit_at_fixed_K(
        probs_k, H, lam, max_iter=max_iter_per_K)
    wildcard_mass = float(wildcard_slots.sum()) / max(2 * N, 1)
    # History entries record BIC = K*cc + 2*NLL so callers can compare
    # entries across different K values directly.  At fixed K this is
    # NLL + const, so it preserves within-K ordering; across K it
    # correctly accounts for the per-founder complexity penalty.
    history.append((H.shape[0], _compute_bic(H.shape[0], nll, cc),
                    wildcard_mass, n_iter))
    # Track NLL of the last accepted state — used as the comparison point
    # for the next K candidate's acceptance check.  Note: comparisons
    # use NLL_improvement vs cc/2 here, which is algebraically the same
    # as comparing BIC(K+1) vs BIC(K); we keep the NLL form for numerical
    # accuracy (avoids double-floating-point recombination).
    prev_nll = nll

    # === K-growth loop (handles K=0→1, K=1→2, ... uniformly) ===
    #
    # Stopping criteria (in priority order):
    #   1. K_max reached (safety cap)
    #   2. BIC reject: nll_improvement < cc/2 → adding this founder
    #      would not reduce K*cc + 2*NLL.  This is the principled
    #      "BIC no longer improves" stop and lives further down the
    #      loop, after we've fit the candidate.
    #
    # Earlier versions also had a wildcard-mass-based early-stop here
    # (`if wildcard_mass <= wildcard_mass_threshold: break`), with
    # default threshold 0.0.  This was REMOVED after the
    # chr3:27772468 diagnostic: wildcard_mass=0 does not mean "all
    # samples well-fit"; it means "no sample chose a (real, W) or
    # (W, W) pair under the current LAM-vs-real-pair-cost arithmetic".
    # A sample can be assigned to (real, real) with per-sample-cost
    # 100+ NLL units (terrible fit) and still produce wm=0, because
    # (real, real) at cost 100 is still cheaper than (real, W) at
    # cost ≈ best_real + lam*L.  Stopping K-growth on wm=0 in such
    # cases caused premature exit at K_alg < K_truth even when BIC
    # overwhelmingly justified continuing (truth K=6 had BIC=379 vs
    # alg K=4 BIC=12585 on chr3:27772468 — a 12,200-unit gap).
    #
    # The wildcard_mass_threshold parameter is retained in the
    # signature for backward compatibility and is still used elsewhere
    # in the codebase (e.g. the uncertainty flag in
    # generate_haplotypes_block), but does NOT affect K-growth's stop.
    while True:
        # Re-check thread allocation at each K-growth step.
        thread_config.apply_dynamic_threads()
        K_cur = H.shape[0]
        if K_cur >= K_max:
            break

        # Seed new founder via SUBTRACTION from the worst-fit sample.
        # Rationale: the worst-fit sample is one whose pair (and thus
        # whose two real founders) the current set fails to explain.
        # If we hypothesise that ONE of its strands is an existing
        # founder F_i, the OTHER strand has a determined value at sites
        # where dosage and F_i agree, and is ambiguous at sites where
        # they conflict (those sites become MASK or rounded).  We try
        # each existing founder F_i as the "known strand" hypothesis,
        # producing K_cur candidate other-strand haps; pick the one
        # most distinct from existing founders (max min-Hamming).  This
        # gives a real-founder-hypothesis seed (per principle 8) rather
        # than a hybrid-average seed.
        #
        # If the worst sample has 2 wildcard strands (no real founder
        # hypothesis), or all samples are pure wildcards, fall back to
        # the dosage / 2 heuristic on the most-decisive sample.
        worst_candidate_mask = (wildcard_slots < 2)        # exclude (W, W)
        if not worst_candidate_mask.any():
            # All samples are (W, W) — fall back to most-decisive sample
            worst_idx = _select_initial_seed(probs_k, kept_mask=None)
            new_h = _init_hap_from_sample_dosage(
                probs_k, worst_idx, kept_mask=None)
            # Single candidate in this branch, no picker needed.  Run
            # CD once on the chosen seed.
            H_try = np.vstack([H, new_h[None, :]])         # (K+1, L_kept)
            H_try, A_try, cost_try, wcs_try, n_iter_try, nll_try = \
                _fit_at_fixed_K(probs_k, H_try, lam, max_iter=max_iter_per_K)
            wm_try = float(wcs_try.sum()) / max(2 * N, 1)
        else:
            adjusted_cost = np.where(worst_candidate_mask,
                                     per_sample_cost, -np.inf)
            worst_idx = int(adjusted_cost.argmax())
            # Subtraction-based seed candidates
            worst_dosage = probs_k[worst_idx].argmax(axis=1)   # (L_kept,)
            seed_candidates = []
            for i in range(K_cur):
                # Implied other strand: dosage - F_i, clipped to [0, 1].
                # Where this is fractional / ambiguous (e.g., dosage=2 but
                # F_i = 0 implies other = 2 which is impossible), the
                # subtraction is invalid at that site — we fall back to
                # the data's argmax-favored single-strand value (i.e.,
                # if dosage = 2 we set other_strand = 1; if dosage = 0,
                # other_strand = 0).
                other = worst_dosage - H[i]
                # Clip values: anything outside {0, 1} indicates the
                # F_i hypothesis is inconsistent at that site.  Project
                # to nearest valid {0, 1} value to keep going.
                other = np.clip(other, 0, 1).astype(np.int64)
                seed_candidates.append(other)
            # Also include the simple dosage / 2 heuristic as a fallback
            seed_candidates.append(_init_hap_from_sample_dosage(
                probs_k, worst_idx, kept_mask=None))
            # Pick the candidate that's most distinct from existing
            # founders (maximises min-Hamming to any existing F_i)
            #
            # HISTORICAL NOTE — old picker and the bug it caused:
            # Originally we picked by max-min-Hamming (the candidate
            # furthest from any existing founder).  This optimises the
            # WRONG criterion: hap-space distance, not data fit.  At
            # blocks where K-growth has converged into a "chimera basin"
            # (existing founders are weighted-averages of multiple
            # truths rather than any single truth), the max-distance
            # candidate is typically a chimera-residual that no sample
            # in the data actually wants as a strand.  Such a seed has
            # zero carriers, M-step cannot update it, the trial CD fit
            # produces dNLL ≈ 0, and BIC rejects K_cur+1.  The algorithm
            # then halts at the local minimum.  Diagnostic on
            # chr1:34921614 confirmed: at K=6 stuck-NLL=26777, the
            # max-distance candidate gave post-CD NLL=26777 (no change),
            # but a DIFFERENT candidate in the same set gave post-CD
            # NLL=19131 (a 7600-unit drop, escaping the basin).  Truth
            # NLL on this block is 551 — the algorithm is 49x above
            # truth in NLL because the picker selected the "different
            # but useless" candidate over the "less different but
            # actually fits the data" one.
            #
            # NEW picker: evaluate every candidate by running CD and
            # picking the one with lowest post-CD NLL.  This trades
            # ~K extra CD fits per K-growth step for correctness — at
            # K_max=10 the total cost is 1+2+...+10 = 55 fits instead
            # of 10, but the resulting K-growth trajectory escapes
            # local minima that the old picker couldn't.
            #
            # Determinism is preserved: candidate generation is
            # deterministic (np.clip arithmetic + dosage-init), each
            # CD trial is deterministic (given the same probs_k and
            # initial H), the argmin is deterministic with stable
            # tie-breaking via candidate-index order.  We tie-break
            # on max-min-Hamming (the legacy criterion) when post-CD
            # NLL values are equal to within a small tolerance, so
            # that in the limit of no NLL difference (e.g. K=0 case
            # which doesn't enter this branch anyway) we recover the
            # legacy behaviour.
            #
            # Note on BIC vs NLL: every candidate at this branch has
            # the same target K = K_cur+1, so BIC = K*cc + 2*NLL
            # differs from NLL only by the constant (K_cur+1)*cc.
            # Picking by lowest NLL is therefore identical to picking
            # by lowest BIC — no need to add the cc term here.
            best_NLL = float('inf')
            best_min_d = -1.0
            best_seed = None
            best_fit = None
            for cand in seed_candidates:
                H_cand = np.vstack([H, cand[None, :]])
                fit_state = _fit_at_fixed_K(probs_k, H_cand, lam,
                                             max_iter=max_iter_per_K)
                cand_nll = float(fit_state[5])     # nll_try is index 5
                # Hamming to each existing founder, for tie-break
                ds = [float(np.mean(cand != H[i])) for i in range(K_cur)]
                min_d = min(ds) if ds else 1.0
                # Pick by NLL primarily, max-min-Hamming as tiebreak
                if (cand_nll < best_NLL - 1e-9 or
                        (abs(cand_nll - best_NLL) <= 1e-9 and
                         min_d > best_min_d)):
                    best_NLL = cand_nll
                    best_min_d = min_d
                    best_seed = cand
                    best_fit = fit_state
            new_h = best_seed
            # Reuse the captured fit for the chosen candidate — no
            # need to refit.
            H_try, A_try, cost_try, wcs_try, n_iter_try, nll_try = best_fit
            wm_try = float(wcs_try.sum()) / max(2 * N, 1)

        # Did the new founder reduce BIC?
        #
        # BIC(K) = K * cc + 2 * NLL_K, where cc is the per-founder
        # complexity cost (linear in N as in the project's
        # beam_search_core / chimera_resolution standard).  Adding a
        # founder reduces BIC iff
        #     NLL_improvement = NLL_K - NLL_{K+1} > cc / 2
        # i.e., the likelihood gain (in NLL units) outweighs half the
        # complexity cost (the factor of 2 cancels with the 2*NLL form
        # of BIC).
        #
        # This replaces v3's "any positive NLL improvement" criterion
        # which incorrectly accepted spurious K-additions absorbing
        # tiny amounts of noise, producing K=truth+1 or K=truth+2
        # blocks past the real K.  See benchmark_stage3_em K-distribution
        # showing K=7+ blocks past truth K=6 — those are now rejected.
        #
        # NLL captures every source of fit improvement:
        #   - wildcard-slot reductions (samples switching from (W,W) to
        #     (real, W) or to (real, real))
        #   - better-fitting real founders (samples switching from
        #     (close-to-truth, W) to (exact-truth, W) — same number of
        #     wildcard slots but lower per-site data-fit cost)
        #   - pair-assignment reorganisations that improve overall fit
        #
        # See trace_discrete_block on chr3:26562266 — K=4→K=5 reduced
        # NLL by 1380 (real improvement: hap4 became exact t3, letting
        # (t1,t3) samples upgrade from (hap1, W) to (hap4, W)).  With
        # cc=160 and threshold=80, this passes easily.
        #
        # The min_wildcard_relative_improvement parameter is preserved
        # in the signature for backward compatibility but is unused.
        nll_improvement = prev_nll - nll_try
        history.append((K_cur + 1,
                        _compute_bic(K_cur + 1, nll_try, cc),
                        wm_try, n_iter_try))
        if nll_improvement < accept_threshold:
            # New founder didn't sufficiently improve BIC — reject and stop
            break

        # Accept
        H = H_try
        A = A_try
        per_sample_cost = cost_try
        wildcard_slots = wcs_try
        wildcard_mass = wm_try
        prev_nll = nll_try

    return H, A, per_sample_cost, wildcard_slots, H.shape[0], wildcard_mass, history


# =============================================================================
# SOFT-CLUSTERING SEEDS FOR INITIAL-K-GROWTH MULTI-START
# =============================================================================
# Generates the diverse K=1 seed haps for _initial_kgrowth_with_medoids by
# clustering the samples on the posterior soft-agreement similarity and
# emitting one denoised pooled-alt consensus seed per cluster.  This keeps
# the genotype posteriors rather than hard-calling each sample's dosage,
# which preserves the low-read-depth signal.  hdbscan and bhd_kernels are
# imported lazily.

def _soft_cluster_seed_haps(probs_k, n_seeds,
                              min_cluster_size=SEED_SOFT_MIN_CLUSTER_SIZE,
                              verbose=False):
    """Generate up to ``n_seeds`` diverse K=1 seed haps by soft clustering.

    Clusters samples on the expected-genotype-agreement similarity
    (bhd_kernels.soft_agreement_similarity) via HDBSCAN on the derived
    distance (S.max() - S), ranks clusters by membership size (largest
    first), and returns one binary seed hap per cluster (up to n_seeds) as
    the per-site pooled-alt consensus of the cluster
    (bhd_kernels.alt_fractions averaged over members -> pooled_alt_to_hap).

    A homozygous-looking cluster yields a clean founder readout; a
    heterozygous (pair-type) cluster yields the same forced-bits / majority
    readout the per-sample seed would, but denoised by pooling over the
    whole cluster — a much better K-growth starting point at low read depth.

    Arguments:
        probs_k: (N, L, 3) genotype posteriors restricted to kept sites
        n_seeds: int — maximum number of seed haps to return (the multi-
            start branch count)
        min_cluster_size: int — HDBSCAN minimum cluster size
        verbose: bool

    Returns:
        list of (L,) np.int64 seed haps, length in [0, n_seeds].  May be
        shorter than n_seeds (or empty) when HDBSCAN finds fewer clusters;
        the caller falls back to its single-branch path when empty.
    """
    import bhd_kernels as _bk
    import hdbscan

    N, L = probs_k.shape[0], probs_k.shape[1]

    # Soft-agreement similarity -> precomputed distance for HDBSCAN.
    S = _bk.soft_agreement_similarity(probs_k)            # (N, N) in [0, 1]
    dist = (S.max() - S)
    np.fill_diagonal(dist, 0.0)
    dist = np.ascontiguousarray(dist, dtype=np.float64)

    labels = hdbscan.HDBSCAN(
        metric='precomputed',
        min_cluster_size=int(min_cluster_size),
    ).fit(dist).labels_

    # Rank clusters by size, largest first (label -1 is HDBSCAN noise).
    clusters = [(c, np.where(labels == c)[0])
                for c in np.unique(labels) if c != -1]
    clusters.sort(key=lambda cm: -cm[1].shape[0])

    if verbose:
        sizes = [int(mem.shape[0]) for _c, mem in clusters]
        print(f'[seed-soft] N={N}, clusters={len(clusters)} sizes={sizes}, '
              f'taking up to {n_seeds}')

    alt = _bk.alt_fractions(probs_k)                      # (N, L) E[alt dose]/2
    seeds = []
    for _c, mem in clusters[:n_seeds]:
        pooled = alt[mem].mean(axis=0)                    # (L,)
        seeds.append(_bk.pooled_alt_to_hap(pooled).astype(np.int64))
    return seeds


def _initial_kgrowth_with_medoids(probs_k, kept_mask_full, lam,
                                    n_medoid_starts,
                                    wildcard_mass_threshold,
                                    min_relative_improvement,
                                    K_max,
                                    max_iter_per_K,
                                    known_haps_full,
                                    cc_scale,
                                    use_log_bic,
                                    min_nll_improvement,
                                    H_trio_seed=None,
                                    run_per_branch_recovery=False,
                                    recovery_outer_cc_scale=RECOVERY_OUTER_CC_SCALE,
                                    recovery_max_K=RECOVERY_MAX_K,
                                    recovery_max_rounds=RECOVERY_MAX_ROUNDS,
                                    recovery_intra_round_dedup_pct=RECOVERY_INTRA_ROUND_DEDUP_PCT,
                                    recovery_mixture_K_max=RECOVERY_MIXTURE_K_MAX,
                                    recovery_mixture_n_restarts=RECOVERY_MIXTURE_N_RESTARTS,
                                    recovery_mixture_seed_base=RECOVERY_MIXTURE_RNG_SEED,
                                    recovery_mixture_patience=RECOVERY_MIXTURE_PATIENCE,
                                    recovery_cleanness_threshold=RECOVERY_CLEANNESS_THRESHOLD,
                                    recovery_swap_nll_tolerance=RECOVERY_SWAP_NLL_TOLERANCE,
                                    recovery_haps_equal_eps_pct=RECOVERY_HAPS_EQUAL_EPS_PCT,
                                    verbose=False):
    """Run K-growth (optionally seeded from H_trio_seed) with k-medoid
    multi-start over sample seeds, plus optional per-branch subtraction
    recovery before BIC arbitration.

    Builds N sample-dosage seeds (one per sample), computes the (N, N)
    Hamming distance matrix between seeds, runs PAM at K=n_medoid_starts
    to pick diverse seed samples, then for each medoid m runs a full
    branch:

      a. H_init for branch m = stack([H_trio_seed, seed_array[m:m+1]])
         when H_trio_seed is non-empty, else just seed_array[m:m+1].
      b. Run K-growth from H_init.
      c. If run_per_branch_recovery: run subtraction-recovery on the
         K-growth output, then re-fit at fixed K to populate the full
         result tuple.
      d. Compute branch BIC.

    Returns the trajectory with lowest final BIC = K_final * cc + 2 * NLL.

    Selection by BIC (not raw NLL) properly handles the case where
    different medoids land at different K_final values: a trajectory
    that grew to K=8 with marginally lower NLL than one at K=6 will
    correctly lose if those two extra founders don't pay their
    complexity cost.

    H_trio_seed parameter:
      When provided (non-empty), serves as a SHARED prefix in every
      branch's H_init.  All branches start with the trio-derived
      founder set and then add one medoid-derived candidate hap on
      top, so different branches differ only in their candidate
      "K_trio+1-th founder" hypothesis.  CD inside K-growth refines
      the medoid candidate (and possibly the trio haps) to fit the
      data; per-branch recovery (when enabled) then runs subtraction-
      recovery to find any additional founders.  BIC arbitration
      across branches picks the winner.

      Plus a NO-MEDOID BASELINE branch is run alongside (only when
      H_trio_seed is non-empty): it starts with H_init = H_trio_seed
      alone (size K_trio).  This branch's K-growth first does CD at
      K_trio, then tries K_trio+1 via worst-fit-sample seeding with
      proper BIC comparison (does K_trio+1 improve over K_trio?).
      Required because the medoid branches all start at K_trio+1
      and never compare against K_trio — so degenerate K_trio+1
      attractors that happen to have the same NLL as truth K_trio
      (which occur on all-hets symmetry cases) would otherwise win
      every branch and trap us in a wrong-K basin.  The baseline
      branch is the only one that gets to test K_trio+1 → K_trio.

      When None or empty, falls back to legacy behavior: each branch's
      H_init is just seed_array[m:m+1] (K=0 -> K=1 starting set).
      No no-medoid baseline branch in this case (would be K=0 which
      is meaningless).

    run_per_branch_recovery parameter:
      When True, runs _subtraction_recovery_round_loop on each branch's
      K-growth output before computing branch BIC.  This gives each
      branch a chance to BIC-discover additional founders via mixture
      recovery before cross-branch arbitration — without it, a branch
      with marginally better K-growth-only BIC would win even if
      another branch had recoverable founders that would have flipped
      the ranking.

      The recovery_* parameters are passed through to
      _subtraction_recovery_round_loop and are ignored when
      run_per_branch_recovery=False.

    Arguments mirror _grow_K (which is called per medoid).  When
    n_medoid_starts <= 1 OR N < MEDOID_MIN_N_FOR_MULTISTART, falls
    back to a single branch (using H_trio_seed if provided, else None
    for K=0 start).  Per-branch recovery still runs in this fallback
    when enabled — there's just only one branch to arbitrate over.

    Returns: same tuple as _grow_K:
      (H, A, per_sample_cost, wildcard_slots, K_final, wildcard_mass,
       history)

    History follows _grow_K's format — a list of (K, BIC, wildcard_mass,
    n_iter) tuples, one per K-growth step inside the WINNING branch.
    Recovery doesn't add growth steps so its effects (which may change
    K) are not recorded in history; verbose logging shows them via the
    [recovery] / [medoid] tag prints.
    """
    N, L_kept, _ = probs_k.shape

    has_trio = (H_trio_seed is not None) and (H_trio_seed.shape[0] >= 1)
    K_trio = int(H_trio_seed.shape[0]) if has_trio else 0

    def _process_one_branch(H_init):
        """Run K-growth + optional subtraction-recovery on a given H_init.
        Returns the same 7-tuple as _grow_K."""
        result = _grow_K(
            probs_k, kept_mask_full, lam,
            wildcard_mass_threshold=wildcard_mass_threshold,
            min_relative_improvement=min_relative_improvement,
            K_max=K_max,
            max_iter_per_K=max_iter_per_K,
            known_haps_full=known_haps_full,
            cc_scale=cc_scale,
            use_log_bic=use_log_bic,
            min_nll_improvement=min_nll_improvement,
            H_init=H_init)
        if not run_per_branch_recovery:
            return result
        # result tuple: (H, A, per_sample_cost, wildcard_slots,
        #                K_final, wildcard_mass, history)
        H_after_grow = result[0]
        if H_after_grow.shape[0] < 1:
            # K=0 result: nothing to subtract from, recovery is a no-op
            return result
        H_after_recov = _subtraction_recovery_round_loop(
            probs_k, H_after_grow, lam,
            outer_cc_scale=recovery_outer_cc_scale,
            max_K=recovery_max_K,
            max_rounds=recovery_max_rounds,
            max_iter_per_K=max_iter_per_K,
            intra_round_dedup_pct=recovery_intra_round_dedup_pct,
            mixture_K_max=recovery_mixture_K_max,
            mixture_n_restarts=recovery_mixture_n_restarts,
            mixture_seed_base=recovery_mixture_seed_base,
            mixture_patience=recovery_mixture_patience,
            cleanness_threshold=recovery_cleanness_threshold,
            swap_nll_tolerance=recovery_swap_nll_tolerance,
            haps_equal_eps_pct=recovery_haps_equal_eps_pct,
            use_log_bic=use_log_bic,
            verbose=verbose)
        if H_after_recov.shape[0] < 1:
            # Recovery returned empty (degenerate case) — keep K-growth result
            return result
        # Re-fit at fixed K to populate the full tuple after recovery's
        # internal CD may have changed things.  Note _fit_at_fixed_K
        # returns 6 elements; we reshape to the 7-tuple form _grow_K
        # uses by adding K_final and wildcard_mass.
        H_final, A_final, costs_final, wcs_final, _it, _nll_final = \
            _fit_at_fixed_K(probs_k, H_after_recov, lam,
                              max_iter=max_iter_per_K)
        K_final_recov = H_final.shape[0]
        wm_final = float(wcs_final.sum()) / max(2 * N, 1)
        # History follows _grow_K's contract (list of (K, BIC, wm,
        # n_iter) per growth step).  Recovery isn't a growth step so we
        # preserve the K-growth history unchanged.  Recovery's effects
        # on K and NLL appear via verbose [recovery] tag prints.
        return (H_final, A_final, costs_final, wcs_final,
                  K_final_recov, wm_final, list(result[6]))

    # Per-founder complexity cost for branch BIC comparison.  Must
    # match the cc used inside _grow_K (each branch uses the same cc
    # for its own acceptance criterion); this lets us compare final
    # solutions ACROSS branches at potentially different K.
    cc = _compute_cc(cc_scale, N, L_kept, use_log_bic=use_log_bic)

    # Single-branch fallback: too few samples or single-start requested.
    # Per-branch recovery (if enabled) still applies — we just have
    # only one branch to choose from.
    if n_medoid_starts <= 1 or N < MEDOID_MIN_N_FOR_MULTISTART:
        H_init = H_trio_seed if has_trio else None
        return _process_one_branch(H_init)

    # Cap medoid count at N (PAM degenerate otherwise)
    n_medoid_starts = min(n_medoid_starts, N)

    # Build the diverse K=1 seed haps for the branches.  Each entry of
    # seed_haps is a (L_kept,) binary hap; seed_labels[i] is a short tag
    # used only for verbose logging.  Seeds come from the posterior soft-
    # clustering front-end: cluster on the soft-agreement similarity and
    # use up to n_medoid_starts cluster pooled-alt consensuses as the
    # diverse seeds (a denoised, low-read-depth-robust analogue of the
    # per-sample argmax seed).
    soft_seeds = _soft_cluster_seed_haps(
        probs_k, n_medoid_starts,
        min_cluster_size=SEED_SOFT_MIN_CLUSTER_SIZE, verbose=verbose)
    if len(soft_seeds) == 0:
        # HDBSCAN found no clusters (e.g. too few samples per pair-type
        # at very low depth) — fall back to the single branch.
        if verbose:
            print('[medoid] soft clustering found no clusters — '
                  'single-branch fallback')
        H_init = H_trio_seed if has_trio else None
        return _process_one_branch(H_init)
    seed_haps = soft_seeds
    seed_labels = [f'soft cluster {i}' for i in range(len(seed_haps))]
    if verbose:
        if has_trio:
            print(f'[medoid] {len(seed_haps)} soft-cluster seeds, '
                  f'each branch H_init = stack([H_trio_seed '
                  f'(K={K_trio}), cluster_seed])')
        else:
            print(f'[medoid] {len(seed_haps)} soft-cluster seeds')

    # Run full per-branch processing from each medoid; keep the best
    # by final BIC.  BIC = K_final * cc + 2 * NLL_final correctly
    # penalises trajectories that grew to a larger K than the data
    # justifies, so a marginally lower NLL at K=8 will lose to a
    # slightly higher NLL at K=6 if the extra two founders aren't
    # paying their complexity cost.
    best_BIC = float('inf')
    best_result = None
    best_label = None

    # No-medoid baseline branch (only when has_trio).  This branch
    # starts with H_init = H_trio_seed alone (size K_trio), so
    # K-growth first runs CD at K_trio, then tries K_trio+1 via worst-
    # fit-sample seeding with proper BIC comparison (does K_trio+1
    # improve over K_trio?).  Without this baseline, the medoid
    # branches all start at K_trio+1 directly and never compare
    # against K_trio, so degenerate K_trio+1 attractors with same NLL
    # as truth K_trio (which happen on the all-hets symmetry case)
    # win every branch and the truth K_trio basin is unreachable.
    #
    # Note: this is NOT a redundant computation when the medoid
    # branches happen to also drop down to K_trio internally — those
    # branches start CD at K_trio+1 and only have K_trio+1 → K_trio+2
    # transitions to test, never K_trio+1 → K_trio.  The baseline
    # branch is the only one that gets to test K_trio+1 → K_trio.
    if has_trio:
        if verbose:
            print(f'[medoid] no-medoid baseline branch: H_init = '
                  f'H_trio_seed (K_trio={K_trio})')
        baseline_result = _process_one_branch(H_trio_seed)
        baseline_K = int(baseline_result[4])
        baseline_NLL = float(baseline_result[2].sum())
        baseline_BIC = _compute_bic(baseline_K, baseline_NLL, cc)
        if verbose:
            tag = ' + recovery' if run_per_branch_recovery else ''
            print(f'[medoid] no-medoid baseline{tag}: '
                  f'K_final={baseline_K}, NLL={baseline_NLL:.1f}, '
                  f'BIC={baseline_BIC:.1f}')
        best_BIC = baseline_BIC
        best_result = baseline_result
        best_label = 'no-medoid baseline'

    for seed_hap, label in zip(seed_haps, seed_labels):
        # Re-check thread allocation at the top of each medoid branch (each
        # branch runs its own per-branch recovery -- a heavy phase).
        thread_config.apply_dynamic_threads()
        # Build per-branch H_init: trio_seed prefix + this branch's seed hap
        if has_trio:
            H_init = np.vstack([H_trio_seed, seed_hap[None, :]])
        else:
            H_init = seed_hap[None, :]
        result = _process_one_branch(H_init)
        # result tuple: (H, A, per_sample_cost, wildcard_slots,
        #                K_final, wildcard_mass, history)
        result_K = int(result[4])
        result_NLL = float(result[2].sum())
        result_BIC = _compute_bic(result_K, result_NLL, cc)
        if verbose:
            tag = ' + recovery' if run_per_branch_recovery else ''
            print(f'[medoid] start at {label}{tag}: '
                  f'K_final={result_K}, NLL={result_NLL:.1f}, '
                  f'BIC={result_BIC:.1f}')
        if result_BIC < best_BIC:
            best_BIC = result_BIC
            best_result = result
            best_label = label

    if verbose:
        print(f'[medoid] best trajectory: {best_label}, '
              f'BIC={best_BIC:.1f}')

    return best_result


# =============================================================================
# K-GROWTH WITH SUBTRACTION-RECOVERY ITERATION (top-level entry)
# =============================================================================

def _grow_K_with_recovery(probs_k, kept_mask_full, lam,
                            wildcard_mass_threshold=0.0,
                            min_relative_improvement=0.10,
                            K_max=10,
                            max_iter_per_K=50,
                            known_haps_full=None,
                            cc_scale=0.5,
                            use_log_bic=False,
                            min_nll_improvement=1e-6,
                            n_medoid_starts=K_MEDOID_STARTS_DEFAULT,
                            recovery_outer_cc_scale=RECOVERY_OUTER_CC_SCALE,
                            recovery_max_K=RECOVERY_MAX_K,
                            recovery_max_rounds=RECOVERY_MAX_ROUNDS,
                            recovery_max_outer_iterations=RECOVERY_MAX_OUTER_ITERATIONS,
                            recovery_cleanness_threshold=RECOVERY_CLEANNESS_THRESHOLD,
                            recovery_intra_round_dedup_pct=RECOVERY_INTRA_ROUND_DEDUP_PCT,
                            recovery_mixture_K_max=RECOVERY_MIXTURE_K_MAX,
                            recovery_mixture_n_restarts=RECOVERY_MIXTURE_N_RESTARTS,
                            recovery_mixture_seed_base=RECOVERY_MIXTURE_RNG_SEED,
                            recovery_mixture_patience=RECOVERY_MIXTURE_PATIENCE,
                            recovery_swap_nll_tolerance=RECOVERY_SWAP_NLL_TOLERANCE,
                            recovery_haps_equal_eps_pct=RECOVERY_HAPS_EQUAL_EPS_PCT,
                            verbose=False):
    """Drop-in replacement for _grow_K with subtraction-recovery iteration.

    Algorithm:
      0. Trio recovery (XOR-based group-trio algorithm) generates
         candidate founder haplotypes via algebraic composition of
         heterozygous samples.  Greedy forward-selection BIC trim
         keeps only haps that strictly improve BIC over K=0 baseline.
         May produce zero seed haps on blocks where the trio scheme
         doesn't apply (K<3, all-hom data, etc.).
      1. K-medoid multistart with per-branch recovery (single unified
         call to _initial_kgrowth_with_medoids):
           - Build N sample-dosage seeds, run PAM to pick
             n_medoid_starts diverse medoids.
           - For each medoid m: H_init = stack([H_trio_seed, medoid_m])
             when H_trio_seed is non-empty, else just medoid_m.
           - When H_trio_seed is non-empty, ALSO run a no-medoid
             baseline branch with H_init = H_trio_seed alone (size
             K_trio).  Required because medoid branches all start at
             K_trio+1 and never test K_trio+1 → K_trio; without the
             baseline, degenerate K_trio+1 attractors with the same
             NLL as truth K_trio (which happen on all-hets symmetry
             cases) win every branch and trap us in a wrong-K basin.
           - K-growth from H_init, then subtraction-recovery on the
             K-growth output, then BIC compute for the branch.
           - Pick branch with lowest BIC as the initial winner.
         When n_medoid_starts <= 1 or N is too small, falls back to a
         single branch (using H_trio_seed if provided, else None).
      2. Iterate up to recovery_max_outer_iterations times:
         a. Recovery on current H (multi-round subtraction + mixture +
            outer BIC subset selection until recovery's own internal
            convergence).
         b. If recovery didn't change H (within haps_equal_eps_pct), exit.
         c. K-growth from recovery's output (continues from K_init =
            K_after_recovery; worst-fit-sample seeding tries to add
            founders that the mixture missed, e.g., low-carrier truths).
         d. If K-growth didn't add anything, exit.
      3. Final _fit_at_fixed_K to populate the full return tuple.

    Why include H_trio_seed in every branch's H_init (instead of just
    the trio path)?  On blocks where trio recovers all true founders,
    every branch converges to K=K_trio (any extra medoid hap gets BIC-
    rejected) and the BIC-tied result is correct.  On blocks where
    trio recovers K_trio_correct < K_truth founders (e.g., one founder
    is hom-only or noise-defeated), different medoids on top of the
    shared trio seed give different starting positions for the K_trio
    +1-th founder; CD inside K-growth refines them, and BIC arbitra-
    tion picks the best.  This is more thorough than running a single
    trajectory from H_trio_seed (which would only try worst-fit-sample
    seeding for the K_trio+1-th founder, missing the multi-hypothesis
    benefit of multistart).

    Why per-branch recovery?  Without it, branches with marginally
    better K-growth-only BIC win even when another branch had recover-
    able founders that would have flipped the ranking after recovery.
    Per-branch recovery gives each branch a chance to BIC-discover its
    full founder set before cross-branch arbitration, so the winner
    is selected on its true post-recovery BIC rather than its K-
    growth-only BIC.

    Why iterate (step 2)?  K-growth (worst-fit-sample seeding) and
    recovery (Bernoulli mixture density) catch DIFFERENT failure modes:
      - Recovery's mixture finds founders supported by many candidates
        clustering in candidate-space (good for moderate-carrier counts).
      - K-growth's worst-fit-sample picks one sample's strand directly
        (good for low-carrier founders whose candidates don't form a
        density cluster but whose individual samples have high cost).
    Iteration ensures both mechanisms get a turn against the residual
    after the other has run.

    Returns: same tuple as _grow_K:
      (H, A, per_sample_cost, wildcard_slots, K_final, wildcard_mass, history)
    """
    N = probs_k.shape[0]

    # 0. Trio recovery: generate candidate founder haps and BIC-trim.
    #
    # The all-hets failure mode (no homozygous samples for some founder
    # pair) traps standard K-growth in wrong basins because every data-
    # driven seed candidate is a heterozygous strand (a blend of two
    # true founders).  Trio recovery sidesteps this by working in XOR
    # space, where het-pair samples have a clean structural composition
    # that lets us algebraically extract individual founders from
    # triangles of samples with overlapping pair-types.
    #
    # We then BIC-trim trio's output via greedy forward selection (the
    # same _greedy_bic_select used by the recovery loop) so spurious
    # haps (e.g., from chain-merged clusters at low-diversity
    # boundaries, or noise floor false positives) don't contaminate the
    # seed.  Each accepted hap strictly improves BIC; rejected haps are
    # dropped.  May produce zero seed haps if trio gives nothing
    # usable (K<3, all-hom data, no triangles match thresholds, or all
    # candidates fail BIC trim).  In that case we fall through to step
    # 1's multistart path.
    H_seed = np.zeros((0, probs_k.shape[1]), dtype=np.int64)
    if bhd_trio.TRIO_RECOVERY_ENABLED or bhd_pairwise.PAIRWISE_RECOVERY_ENABLED:
        # Gather trio candidates (XOR-triangle algebraic algorithm — see
        # _trio_recovery_candidate_haps above).  When TRIO_RECOVERY_ENABLED
        # is False, trio is skipped entirely (production behavior before
        # trio integration).
        if bhd_trio.TRIO_RECOVERY_ENABLED:
            H_trio_candidates = _trio_recovery_candidate_haps(
                probs_k, verbose=verbose)
        else:
            H_trio_candidates = np.zeros((0, probs_k.shape[1]), dtype=np.int64)
        trio_list = [H_trio_candidates[k]
                     for k in range(H_trio_candidates.shape[0])]
        # Gather v6 pairwise common-hap candidates (partial-haps clustered
        # by mutual compatibility + quality filters A-E).  Pairwise covers
        # complementary failure modes to trio: trio excels on all-hets data
        # via XOR triangulation, pairwise excels when clean homozygous
        # samples for some founders exist (its pair-of-carriers signal is
        # strong there).  Feeding both into the combined seed gives BIC-
        # trim a richer pool.  See pairwise_common_hap.py for the v6
        # algorithm and the 50-block integration test results.
        if bhd_pairwise.PAIRWISE_RECOVERY_ENABLED:
            pairwise_list = bhd_pairwise.pairwise_recovery_candidate_haps(
                probs_k, verbose=verbose)
        else:
            pairwise_list = []
        # Combine the two pools and dedup at 0.5% Hamming — collapses
        # near-exact duplicates (e.g., when trio and pairwise independently
        # recover the same clean truth).  Threshold matches the one used
        # inside _late_low_carrier_rescue (see ~line 3275 below).  When
        # PAIRWISE_RECOVERY_ENABLED is False this dedup is a no-op on
        # trio-only output (trio's internal dedup is at TRIO_HAP_DEDUP_PCT
        # = 2.0%, so trio candidates are guaranteed ≥2.0% apart and never
        # within 0.5%), making the pre-integration code path numerically
        # identical to the baseline.
        cand_list = []
        for cand in trio_list + pairwise_list:
            is_dup = False
            for kept in cand_list:
                if _hamming_pct_kept(cand, kept) < 0.5:
                    is_dup = True
                    break
            if not is_dup:
                cand_list.append(cand)
        if cand_list:
            # Greedy forward-selection BIC trim.  Uses the same cc_scale
            # AND use_log_bic as the K-growth that follows, so trim and
            # grow share an identical BIC criterion.  Each accepted hap
            # strictly improves BIC; rejected haps are dropped.
            #
            # Build a PoolEmissionCache wrapping the combined trio +
            # pairwise candidate pool.  _greedy_bic_select makes
            # O(|cand_list|² / 2) calls to _compute_nll_for_subset
            # internally (forward selection trials each remaining
            # candidate at each K step), and the cache amortises the
            # Viterbi emission build across all those calls.
            seed_cache = PoolEmissionCache(cand_list, probs_k,
                                            lam=lam)
            sel_indices, sel_haps, _trim_nll = _greedy_bic_select(
                seed_cache,
                cc_scale=cc_scale,
                max_k=K_max,
                use_log_bic=use_log_bic,
                verbose=verbose)
            if sel_haps:
                H_seed = np.stack(sel_haps, axis=0).astype(np.int64)
            if verbose:
                # Rewritten from the original `[trio] N candidates -> ...`
                # print to reflect that the BIC trim now consumes a
                # combined pool of trio + pairwise candidates.
                print(f'[seed] trio={len(trio_list)} + '
                      f'pairwise={len(pairwise_list)} -> '
                      f'combined+deduped={len(cand_list)} -> '
                      f'BIC-trimmed to {H_seed.shape[0]} seed haps')

    # 1. K-medoid multistart with per-branch recovery, optionally
    # seeded from trio.
    #
    # Each branch m starts from H_init = stack([H_seed, seed_array[m]])
    # when H_seed is non-empty, or just seed_array[m] when empty.
    # Per-branch recovery runs subtraction-recovery on the K-growth
    # output before computing branch BIC, so cross-branch arbitration
    # happens on post-recovery BIC.
    #
    # Why include H_seed in every branch (vs single trajectory from
    # H_seed when it's non-empty): different medoids give different
    # starting positions for the K_trio+1-th founder hypothesis,
    # giving multi-shot exploration in cases where trio finds K_trio
    # correct founders but the truth has K_trio+1 (e.g., one founder
    # is hom-only and trio missed it).
    #
    # Why per-branch recovery (vs only running recovery once on the
    # winner): without it, branches with marginally better K-growth-
    # only BIC win even when another branch's recovery would have
    # found additional founders that flipped the ranking.
    #
    # Cost: roughly 2x slower than the previous "single trajectory or
    # multistart-K-growth-only" design.  On production scale, pushes
    # stage-3 from ~30 min to ~1 hour single-threaded (proportional on
    # parallel cores).  Trade-off accepted for more thorough
    # exploration on the rare hard blocks where it matters.
    # Re-check thread allocation before the medoid multistart (the per-branch
    # recovery here is the heaviest single phase): a straggler block claims
    # cores freed as its peers finish.
    thread_config.apply_dynamic_threads()
    H, A, costs, wcs, K_final, wm, history = _initial_kgrowth_with_medoids(
        probs_k, kept_mask_full, lam,
        n_medoid_starts=n_medoid_starts,
        wildcard_mass_threshold=wildcard_mass_threshold,
        min_relative_improvement=min_relative_improvement,
        K_max=K_max,
        max_iter_per_K=max_iter_per_K,
        known_haps_full=known_haps_full,
        cc_scale=cc_scale,
        use_log_bic=use_log_bic,
        min_nll_improvement=min_nll_improvement,
        H_trio_seed=H_seed,
        run_per_branch_recovery=True,
        recovery_outer_cc_scale=recovery_outer_cc_scale,
        recovery_max_K=recovery_max_K,
        recovery_max_rounds=recovery_max_rounds,
        recovery_intra_round_dedup_pct=recovery_intra_round_dedup_pct,
        recovery_mixture_K_max=recovery_mixture_K_max,
        recovery_mixture_n_restarts=recovery_mixture_n_restarts,
        recovery_mixture_seed_base=recovery_mixture_seed_base,
        recovery_mixture_patience=recovery_mixture_patience,
        recovery_cleanness_threshold=recovery_cleanness_threshold,
        recovery_swap_nll_tolerance=recovery_swap_nll_tolerance,
        recovery_haps_equal_eps_pct=recovery_haps_equal_eps_pct,
        verbose=verbose)

    if verbose:
        print(f'[recovery] Initial K-growth: K_final={K_final}, '
              f'wildcard_mass={wm:.4f}')

    # 2. Outer iteration: alternate recovery and K-growth
    for outer_it in range(recovery_max_outer_iterations):
        # Re-check thread allocation at the top of each outer iteration.
        thread_config.apply_dynamic_threads()
        if verbose:
            print(f'[recovery] === Outer iteration {outer_it + 1} ===')

        # 2a. Recovery on current H
        H_after_recovery = _subtraction_recovery_round_loop(
            probs_k, H, lam,
            outer_cc_scale=recovery_outer_cc_scale,
            max_K=recovery_max_K,
            max_rounds=recovery_max_rounds,
            max_iter_per_K=max_iter_per_K,
            intra_round_dedup_pct=recovery_intra_round_dedup_pct,
            mixture_K_max=recovery_mixture_K_max,
            mixture_n_restarts=recovery_mixture_n_restarts,
            mixture_seed_base=recovery_mixture_seed_base + outer_it * 1000,
            mixture_patience=recovery_mixture_patience,
            cleanness_threshold=recovery_cleanness_threshold,
            swap_nll_tolerance=recovery_swap_nll_tolerance,
            haps_equal_eps_pct=recovery_haps_equal_eps_pct,
            use_log_bic=use_log_bic,
            verbose=verbose)

        # 2b. Did recovery change H?
        H_list = [H[k] for k in range(H.shape[0])] if H.shape[0] > 0 else []
        H_rec_list = [H_after_recovery[k] for k in range(H_after_recovery.shape[0])] \
                      if H_after_recovery.shape[0] > 0 else []
        if _haps_equal(H_rec_list, H_list, eps_pct=recovery_haps_equal_eps_pct):
            if verbose:
                print(f'[recovery] Outer iteration {outer_it + 1}: '
                      f'recovery did not change H -- CONVERGED')
            break

        if verbose:
            print(f'[recovery] Outer iteration {outer_it + 1}: '
                  f'recovery K {H.shape[0]} -> {H_after_recovery.shape[0]}')

        # 2c. K-growth from recovery's output
        H_after_grow, A, costs, wcs, K_after_grow, wm, hist_grow = _grow_K(
            probs_k, kept_mask_full, lam,
            wildcard_mass_threshold=wildcard_mass_threshold,
            min_relative_improvement=min_relative_improvement,
            K_max=K_max,
            max_iter_per_K=max_iter_per_K,
            known_haps_full=known_haps_full,
            cc_scale=cc_scale,
            use_log_bic=use_log_bic,
            min_nll_improvement=min_nll_improvement,
            H_init=H_after_recovery)
        history.extend(hist_grow)

        # 2d. Did K-growth add anything?
        H_grow_list = [H_after_grow[k] for k in range(H_after_grow.shape[0])] \
                       if H_after_grow.shape[0] > 0 else []
        if _haps_equal(H_grow_list, H_rec_list, eps_pct=recovery_haps_equal_eps_pct):
            if verbose:
                print(f'[recovery] Outer iteration {outer_it + 1}: '
                      f'K-growth did not add -- CONVERGED')
            H = H_after_grow
            break

        if verbose:
            print(f'[recovery] Outer iteration {outer_it + 1}: '
                  f'K-growth K {H_after_recovery.shape[0]} -> {H_after_grow.shape[0]}')

        H = H_after_grow

    # 3. Final fit to populate return values consistently
    H_final, A_final, costs_final, wcs_final, n_iter_final, nll_final = \
        _fit_at_fixed_K(probs_k, H, lam, max_iter=max_iter_per_K)

    # 3.5. Late low-carrier rescue (added May 2026): targeted post-
    # convergence pass that detects suspect low-carrier haps (potential
    # chimeric stand-ins for low-frequency founders) and tries to
    # replace them with carrier-derived residual candidates via BIC-
    # aware forward selection.  Triggers only when min carrier count
    # is below RECOVERY_LOW_CARRIER_TRIGGER_FRAC of 2N (typical: <5%
    # of blocks); for triggered blocks, accepts the new state iff it
    # strictly improves BIC.  Cannot regress.  See chr6:23624234
    # diagnostic for the motivating analysis.
    H_final, A_final, costs_final, wcs_final, nll_final = _late_low_carrier_rescue(
        probs_k, H_final, A_final, costs_final, wcs_final, nll_final,
        lam=lam, cc_scale=cc_scale, use_log_bic=use_log_bic,
        max_iter=max_iter_per_K, verbose=verbose)

    # 3.6. Residual-trio rescue (added 2026-05): post-convergence pass
    # that mines per-sample residuals across ALL samples (not just low-
    # carrier-hap carriers) to surface near-clone founders that K-
    # growth's residual-mass seeding missed.  Complements low-carrier
    # rescue (which handles low-frequency chimeric replacements) by
    # targeting the orthogonal pattern: all haps healthy but one
    # absorbs carriers of a near-clone partner founder.  Internal
    # gate: skip if no admitted candidate or BIC does not improve.
    # Cannot regress.  See chr10:503 diagnostic for the motivating
    # case (F0 vs F4 at 5-bit distance, 14 clean F0 carriers absorbed
    # into the F4 slot).
    if RESIDUAL_TRIO_ENABLED:
        H_final, A_final, costs_final, wcs_final, nll_final = _residual_trio_rescue(
            probs_k, H_final, A_final, costs_final, wcs_final, nll_final,
            lam=lam, cc_scale=cc_scale, use_log_bic=use_log_bic,
            max_iter=max_iter_per_K, verbose=verbose)

    wm_final = float(wcs_final.sum()) / max(2 * N, 1)

    if verbose:
        # Report BIC (not raw NLL) since "FINAL" is the comparison point
        # external callers might use to compare across different K
        # outcomes from this function.
        cc_final = _compute_cc(cc_scale, N, probs_k.shape[1],
                                use_log_bic=use_log_bic)
        bic_final = _compute_bic(H_final.shape[0], nll_final, cc_final)
        print(f'[recovery] FINAL: K={H_final.shape[0]}, '
              f'BIC={bic_final:.1f}, NLL={nll_final:.1f}, '
              f'wildcard_mass={wm_final:.4f}')

    return H_final, A_final, costs_final, wcs_final, H_final.shape[0], wm_final, history


# =============================================================================
# OUTPUT CONSTRUCTION
# =============================================================================

def _compute_per_site_confidence(probs_k, H_k, A, lam, min_supporters=2):
    """For each (founder, kept site), compute confidence as the fraction
    of attributing samples whose data is consistent with the founder's
    inferred allele under their pair assignment.

    "Consistent" = the per-(sample, site) cost under the real-pair beats
    the wildcard cost — i.e., the founder's allele genuinely fits this
    sample at this site rather than the data being indifferent.

    For sites with fewer than min_supporters attributing samples, the
    confidence is 0 (and the site will be MASKed at output).

    Arguments:
        probs_k: (N, L_kept, 3)
        H_k:     (K, L_kept)
        A:       (N, 2) with K used as wildcard sentinel
        lam:     wildcard penalty
        min_supporters: minimum supporting samples to compute confidence

    Returns:
        confidence: (K, L_kept) float in [0, 1]
        n_supporters: (K, L_kept) int
    """
    K, L = H_k.shape
    if K == 0:
        return (np.zeros((0, L), dtype=np.float64),
                np.zeros((0, L), dtype=np.int64))
    # Hand off to the njit kernel.  The kernel replaces the original
    # Python `for k in range(K): for l in range(L):` double loop with
    # per-site numpy slicing.  Same pattern as bhd_kernels'
    # _update_one_founder rewrite (which gave 22x on the same shape of
    # work); expect comparable speedup here.
    probs_c = np.ascontiguousarray(probs_k, dtype=np.float64)
    H_c = np.ascontiguousarray(H_k, dtype=np.int64)
    A_c = np.ascontiguousarray(A, dtype=np.int64)
    return _compute_per_site_confidence_kernel(
        probs_c, H_c, A_c, float(lam), int(min_supporters))


@njit(cache=True, parallel=True, fastmath=False)
def _compute_per_site_confidence_kernel(probs_k, H_k, A, lam, min_supporters):
    """njit version of _compute_per_site_confidence.

    For each (k, l), determine which samples are "supporting" k at l
    (via their pair-assignment bucket) and how many of those samples'
    data is "consistent" with the founder's allele.  Three buckets:

      Bucket H (k, k): consistent iff argmax of P(g | s, l) == 2*H_k[k, l]
      Bucket J (k, j) with j != k, j real: consistent iff argmax of
                                            P(g | s, l) == H_k[k, l] + H_k[j, l]
      Bucket P (k, W): consistent iff per-site real-pair cost <
                       per-site wildcard cost (so the real founder
                       actually contributed information, not just letting
                       the wildcard absorb)

    Same arithmetic as the Python version.  `prange` over founders
    because samples-in-pair-assignment are unevenly distributed across
    founders (some founders dominate carriers; parallelising over k
    balances out via the global sample-mask scan inside each k).

    Floors -log(p) at LOG_EPS_LOCAL = 1e-12 to match _safe_neg_log's
    behaviour.

    Returns:
        confidence:   (K, L) float64
        n_supporters: (K, L) int64
    """
    LOG_EPS_LOCAL = 1e-12

    N = probs_k.shape[0]
    K = H_k.shape[0]
    L = H_k.shape[1]
    W = K

    confidence = np.zeros((K, L), dtype=np.float64)
    n_supporters = np.zeros((K, L), dtype=np.int64)

    for k in prange(K):
        # Walk samples once to classify each into bucket H, J, P, or
        # not-supporting.  We don't pre-materialise the bucket masks
        # (the Python version did) because numba prefers explicit loops
        # over fancy mask indexing in the inner kernel — and at typical
        # N=320 a single sample-walk per (k, l) inner site is cheap.
        for l in range(L):
            cur_val = H_k[k, l]
            n_supp = 0
            n_consistent = 0

            for s in range(N):
                a0 = A[s, 0]
                a1 = A[s, 1]
                # Check whether sample s supports founder k under any
                # bucket.  Bucket-H test first since it's the cheapest.
                if a0 == k and a1 == k:
                    # Bucket H (k, k): consistent iff argmax P(g) == 2*cur_val
                    p0 = probs_k[s, l, 0]
                    p1 = probs_k[s, l, 1]
                    p2 = probs_k[s, l, 2]
                    # argmax of (p0, p1, p2).  Tie-break: first-max
                    # matches np.argmax's behaviour (which the Python
                    # version used).
                    if p0 >= p1 and p0 >= p2:
                        amax = 0
                    elif p1 >= p2:
                        amax = 1
                    else:
                        amax = 2
                    n_supp += 1
                    if amax == 2 * cur_val:
                        n_consistent += 1
                elif a0 == k and a1 == W:
                    # Bucket P (k, W): consistent iff real-(k,W) cost <
                    # (W, W) cost at site l.
                    p0 = probs_k[s, l, 0]
                    p1 = probs_k[s, l, 1]
                    p2 = probs_k[s, l, 2]
                    # best_real = max(p[cur_val], p[cur_val+1])
                    if cur_val == 0:
                        best_real = p0 if p0 > p1 else p1
                    else:
                        best_real = p1 if p1 > p2 else p2
                    pmax = p0
                    if p1 > pmax:
                        pmax = p1
                    if p2 > pmax:
                        pmax = p2
                    if best_real < LOG_EPS_LOCAL:
                        best_real = LOG_EPS_LOCAL
                    if pmax < LOG_EPS_LOCAL:
                        pmax = LOG_EPS_LOCAL
                    cost_real = -math.log(best_real) + lam
                    cost_WW = -math.log(pmax) + 2.0 * lam
                    n_supp += 1
                    if cost_real < cost_WW:
                        n_consistent += 1
                elif (a0 == k or a1 == k) and a0 != a1 and a0 != W and a1 != W:
                    # Bucket J (k, j) with j != k, j real.  Find
                    # partner founder index j.
                    j = a1 if a0 == k else a0
                    partner_h = H_k[j, l]
                    expected_dosage = cur_val + partner_h
                    p0 = probs_k[s, l, 0]
                    p1 = probs_k[s, l, 1]
                    p2 = probs_k[s, l, 2]
                    if p0 >= p1 and p0 >= p2:
                        amax = 0
                    elif p1 >= p2:
                        amax = 1
                    else:
                        amax = 2
                    n_supp += 1
                    if amax == expected_dosage:
                        n_consistent += 1
                # else: sample s does not support founder k at this
                # site under any bucket; skip.

            n_supporters[k, l] = n_supp
            if n_supp >= min_supporters:
                confidence[k, l] = n_consistent / n_supp
            # else: confidence stays 0 (low-support site).

    return confidence, n_supporters


def _discrete_haps_to_prob_arrays(H_k_full, n_sites_full, kept_mask, confidence_full,
                                    n_supporters_full, min_supporters):
    """Convert the (K, L_full) discrete H to a dict of (n_sites_full, 2)
    [P(allele=0), P(allele=1)] arrays.

    Sites that fall below min_supporters or are not in kept_mask are
    represented as (0.5, 0.5) — the legacy format's encoding for "no
    information."  Confident sites are crisp (1.0, 0.0) or (0.0, 1.0).

    Arguments:
        H_k_full: (K, L_full) — discrete haps padded to full block length
                  (sites outside kept_mask are 0 by default)
        n_sites_full: int
        kept_mask: (L_full,) bool — only kept sites are scored
        confidence_full: (K, L_full) float
        n_supporters_full: (K, L_full) int
        min_supporters: int — sites with fewer supporters become (0.5, 0.5)

    Returns:
        haps_dict: {k: (n_sites_full, 2)} float arrays
    """
    K = H_k_full.shape[0]
    haps_dict = {}
    for k in range(K):
        h_arr = np.full((n_sites_full, 2), 0.5, dtype=np.float64)
        # For each site, if it's kept AND has enough supporters, set crisp
        for l in range(n_sites_full):
            if kept_mask is not None and not kept_mask[l]:
                continue
            if n_supporters_full[k, l] < min_supporters:
                continue
            if H_k_full[k, l] == 0:
                h_arr[l, 0] = 1.0
                h_arr[l, 1] = 0.0
            else:
                h_arr[l, 0] = 0.0
                h_arr[l, 1] = 1.0
        haps_dict[k] = h_arr
    return haps_dict


# =============================================================================
# FINAL CLEANUP: re-uses legacy machinery on the converted prob-array form
# =============================================================================

def _final_cleanup(haps_dict, probs_array, diff_threshold_percent,
                    penalty_strength, chimera_max_recombs,
                    chimera_max_mismatch_pct, chimera_min_delta_to_protect):
    """Apply legacy final-cleanup steps: consolidate near-duplicates,
    Viterbi-BIC selection, chimera pruning.  Uses prob-array form."""
    if len(haps_dict) <= 1:
        return haps_dict

    # Step A: Consolidate near-duplicates
    merged = consolidate_similar_candidates(
        haps_dict, diff_threshold_percent=diff_threshold_percent)
    if len(merged) <= 1:
        return merged

    # Step B: Viterbi-BIC selection — DISABLED in the discrete pipeline.
    #
    # The legacy Viterbi-BIC subset selector was designed for the EM
    # era, where a ~100-candidate pool needed post-hoc subset selection
    # to pick the right ~6.  In the discrete-CD pipeline, K is already
    # authoritatively selected during K-growth via the discrete-CD BIC
    # (cc_scale=0.05, accept threshold cc/2 ≈ 8 NLL nats for N=320,
    # L=200).
    #
    # Viterbi-BIC's criterion (complexity_cost = max(recomb_penalty*1.5,
    # log(N)*L*penalty_strength*0.01) ≈ 57.7 nats per hap for our
    # defaults) is ≈7× stricter than discrete-CD's, and routinely
    # overrules K-growth.  Diagnosed at chr3:16418593 (May 2026):
    # K-growth correctly accepts K=3 with all 6 truths at 0% Hamming
    # (truths 2/3/4/5 are byte-identical at this block, so
    # K_truth_distinct=3); Viterbi-BIC then trims to K=2, dropping the
    # founder uniquely representing truth_0 → truth_0 at 3.5% Hamming
    # after carrier reassignment.  See diagnose_chr3_16418593_postproc.py
    # PART 4 for the trace.
    #
    # Step C (usage prune) and Step D (chimera prune) below still drop
    # any genuinely spurious haps via principled per-hap criteria.
    #
    # Update (May 2026): K-growth's cc_scale was raised from 0.05 to
    # 0.5 after the above diagnosis.  Under cc_scale=0.5, K-growth's
    # accept threshold is cc/2 ≈ 80 NLL nats — now higher than
    # Viterbi-BIC's 57.7 nats per-hap penalty, reversing the strictness
    # ordering described above.  The disable decision still stands:
    # the chr3:16418593 regression was driven by Viterbi-BIC's
    # ABSOLUTE per-hap penalty being applied irrespective of how much
    # data each hap genuinely explains, not by relative strictness.
    # Re-enabling Step B would still trim the truth_0-matching
    # founder at chr3:16418593, since the trim happens because that
    # founder's local likelihood gain is below 57.7 nats while
    # K-growth has independently accepted it on within-block BIC
    # grounds.  The right authority for K-selection is K-growth's
    # BIC at the chosen cc_scale; Step B's distinct (and now
    # less-strict) criterion remains misaligned with that authority.
    #
    # Original code (preserved for record):
    #     best_keys = select_optimal_haplotype_set_viterbi(
    #         merged, probs_array,
    #         recomb_penalty=10.0,
    #         penalty_strength=penalty_strength,
    #     )
    #     selected = {i: merged[k] for i, k in enumerate(best_keys)}
    #     if len(selected) <= 1:
    #         return selected
    selected = merged

    # Step C: Post-usage pruning (drop unused haps).
    #
    # Threshold lowered to 1 from the legacy max(2, 1% of N).
    #
    # The legacy threshold (= 3 for N=320) systematically dropped real
    # founders with low local carrier counts.  Diagnosed at
    # chr3:16378549 (May 2026): _grow_K_with_recovery produces K=7
    # with NLL=69.3 = noise floor; alg_row_5 (= truth_0 within 2 sites,
    # usage=2 strands) and alg_row_6 (spurious chimera, usage=2
    # strands) both fall below threshold=3 and are dropped, leaving
    # truth_0 with no representative within 2% Hamming → founders_found
    # drops from 6/6 to 5/6 with truth_0 at 3.0%.  See
    # diagnose_chr3_16418593_postproc.py PART 4 with target chr3:16378549.
    #
    # K-growth's BIC at cc/2 (≈ 80 NLL nats per hap for N=320, L=200
    # under cc_scale=0.5; was ≈ 8 nats under the prior cc_scale=0.05
    # at the time of the chr3:16378549 diagnosis above) already
    # validates each founder as data-justified.  Step C's only
    # remaining role is to drop literal-zero-carrier "phantom" haps —
    # haps that K-growth accepted at one CD iteration but lost all
    # carriers in subsequent iterations.  threshold=1 catches these
    # while preserving every founder with even a single carrier strand.
    #
    # Original code (preserved for record):
    #     min_samples = max(2, int(probs_array.shape[0] * 0.01))
    final_matches = hap_statistics.match_best_vectorised(selected, probs_array)
    usage_counts = final_matches[1]
    min_samples = 1
    used = {}
    new_idx = 0
    for h_idx, count in usage_counts.items():
        if count >= min_samples:
            used[new_idx] = selected[h_idx]
            new_idx += 1
    if len(used) < 2:
        return used

    # Step D: Chimera pruning — DISABLED in the discrete pipeline.
    #
    # prune_chimeras flags a hap as a chimera-candidate if it can be
    # reconstructed from the OTHER haps via ≤max_recombs (=1) Viterbi
    # transitions with ≤max_mismatch_percent (=0.5% = 1 site at L=200)
    # mismatches.  It then computes mean_delta = average per-sample
    # increase in pair-error if that hap were removed, and prunes any
    # candidate with mean_delta < min_mean_delta_to_protect (=0.25%).
    #
    # In a population with related founders, real founders ARE
    # structurally reconstructible from each other by ancestry — that
    # is what shared ancestry means at the haplotype level.  The
    # Viterbi chimera test cannot distinguish "structurally similar
    # due to shared ancestry" from "actually a chimeric algorithm
    # artifact."  Mean_delta protection scales with carrier frequency
    # (mean_delta ≈ (carriers/N) × per-carrier-error), so any low-
    # frequency real founder is at risk regardless of how true it is.
    #
    # Diagnosed at chr14:10136207 (May 2026): _grow_K_with_recovery
    # correctly settles at K=6 with all 6 truths matched at 0.00%
    # Hamming and NLL=117.0 = noise floor.  Step D's prune_chimeras
    # removes founder 5 (= truth_1, 36 strand-uses out of 640 total)
    # because mean_delta ≈ (36/320) × 2% ≈ 0.225% < 0.25% threshold.
    # See diagnose_chr3_16418593_postproc.py PART 4 for the trace.
    #
    # The discrete pipeline relies on K-growth's BIC at cc/2 (≈ 80 NLL
    # nats per hap for N=320, L=200 under cc_scale=0.5; was ≈ 8 nats
    # under the prior cc_scale=0.05 at the time of the chr14:10136207
    # diagnosis above) as the authoritative filter on whether each
    # founder is data-justified.  At chr14:10136207 the K=7 candidate
    # was rejected with dBIC = +8.4 (under cc_scale=0.05), confirming
    # K-growth's BIC is strict enough that spurious chimeric haps don't
    # survive; under cc_scale=0.5 the K=7 rejection at this block is
    # even stronger (dBIC ≈ +152), reinforcing the conclusion.
    # Step A (consolidate at 0.5% diff threshold) still merges near-
    # duplicates; Step C (usage prune) still drops genuinely-unused
    # haps.  Step D's structural test had no remaining role other than
    # removing legitimate low-frequency founders.
    #
    # Original code (preserved for record):
    #     final = prune_chimeras(
    #         used, probs_array,
    #         max_recombs=chimera_max_recombs,
    #         max_mismatch_percent=chimera_max_mismatch_pct,
    #         min_mean_delta_to_protect=chimera_min_delta_to_protect,
    #     )
    #     return {i: v for i, v in enumerate(final.values())}
    return used


# =============================================================================
# TOP-LEVEL ENTRY: generate_haplotypes_block
# =============================================================================

def generate_haplotypes_block(positions, reads_array, keep_flags=None,
                              # New discrete-coord-descent parameters
                              lambda_wildcard_penalty=DEFAULT_LAMBDA,
                              wildcard_mass_threshold=0.0,
                              min_wildcard_relative_improvement=0.10,
                              K_max=10,
                              coord_descent_max_iter=50,
                              min_supporters_for_confidence=2,
                              n_medoid_starts=K_MEDOID_STARTS_DEFAULT,
                              # Recovery caps and inner-mixture early-stop.
                              # recovery_max_K / recovery_mixture_K_max default
                              # to None = "auto": resolved below to
                              # max(module constant, K_max), so raising the
                              # public K_max (e.g. to support ~40 founders)
                              # auto-raises the recovery selection / inner-
                              # mixture caps WITHOUT changing the default-K_max
                              # behaviour.  recovery_mixture_patience is the
                              # mixture K-sweep early-stop patience (see
                              # RECOVERY_MIXTURE_PATIENCE); pass None to disable
                              # it (full sweep, bit-identical to pre-early-stop).
                              recovery_max_K=None,
                              recovery_mixture_K_max=None,
                              recovery_mixture_patience=RECOVERY_MIXTURE_PATIENCE,
                              # Legacy parameters that still apply (final cleanup)
                              diff_threshold_percent=1.0,
                              penalty_strength=5.0,
                              chimera_max_recombs=1,
                              chimera_max_mismatch_pct=0.5,
                              chimera_min_delta_to_protect=0.25,
                              # Legacy parameters accepted for compat (no-ops here)
                              error_reduction_cutoff=0.98,
                              max_cutoff_error_increase=1.02,
                              max_hapfind_iter=5,
                              deeper_analysis_initial=False,
                              min_num_haps=0,
                              max_intermediate_haps=25,
                              known_haplotypes=None,
                              uniqueness_threshold_percent=2.0,
                              wrongness_threshold=10.0):
    """Discrete-hap founder discovery for a single block.

    Implements an alternative to EM: discrete coordinate descent over
    binary founder haps with hard pair assignment and a wildcard
    founder.  K is grown one founder at a time until the wildcard mass
    falls below `wildcard_mass_threshold`, the wildcard improvement per
    new founder drops below `min_wildcard_relative_improvement`, or
    K_max is reached.

    Returns a BlockResult with extra attributes attached:
        result.discrete_haps:        (K, L_full) int with MASK at low-support sites
        result.per_site_confidence:  (K, L_full) float in [0, 1]
        result.n_site_supporters:    (K, L_full) int
        result.pair_assignments:     (N, 2) int with K = wildcard sentinel
        result.wildcard_mass:        float in [0, 1]
        result.uncertainty_flag:     bool (True if block is genuinely uncertain)
        result.K_final:              int
        result.growth_history:       list of (K, BIC, wildcard_mass, n_iter)
                                     where BIC = K * cc + 2 * NLL with the
                                     same cc as used in K-growth acceptance

    The `haplotypes` attribute uses the legacy (n_sites_full, 2)
    [P(0), P(1)] format for backward compat.
    """
    n_sites_full = reads_array.shape[1]

    # --- 1. SETUP ---
    if keep_flags is None:
        keep_flags = np.ones(n_sites_full, dtype=np.int64)
    if keep_flags.dtype != int:
        keep_flags = np.asarray(keep_flags, dtype=np.int64)
    kept_mask = keep_flags > 0

    # --- 2. PROBS FROM READS ---
    site_priors, probs_array = analysis_utils.reads_to_probabilities(reads_array)

    if len(positions) == 0:
        empty_haps = {}
        result = BlockResult(np.array([]), empty_haps, reads_array,
                              keep_flags=keep_flags, probs_array=probs_array)
        result.discrete_haps = np.empty((0, 0), dtype=np.int64)
        result.per_site_confidence = np.empty((0, 0), dtype=np.float64)
        result.n_site_supporters = np.empty((0, 0), dtype=np.int64)
        result.pair_assignments = np.empty((0, 2), dtype=np.int64)
        result.wildcard_mass = 0.0
        result.uncertainty_flag = True
        result.K_final = 0
        result.growth_history = []
        return result

    # --- 3. RESTRICT TO KEPT SITES FOR INFERENCE ---
    if kept_mask.any():
        # Boolean masking on the middle axis yields a NON-C-contiguous view;
        # probs_k is the largest array in the block and is handed to every CD
        # kernel, each of which requires C-contiguous input (via
        # _maybe_c_contig).  Materialise it C-contiguous ONCE here so those
        # per-call contiguity checks fast-path instead of deep-copying the
        # ~N*L*3 tensor on every founder of every coordinate-descent iteration
        # (profiled at ~28% of a K=40 block).  Pure layout change — the values
        # are identical, so results are bit-for-bit unchanged.
        probs_k = np.ascontiguousarray(probs_array[:, kept_mask, :])
    else:
        # No kept sites — degenerate case
        probs_k = probs_array[:, :0, :]

    if probs_k.shape[1] == 0 or probs_k.shape[0] == 0:
        # Truly nothing to infer
        empty_haps = {}
        result = BlockResult(positions, empty_haps, reads_array,
                              keep_flags=keep_flags, probs_array=probs_array)
        result.discrete_haps = np.empty((0, n_sites_full), dtype=np.int64)
        result.per_site_confidence = np.empty((0, n_sites_full), dtype=np.float64)
        result.n_site_supporters = np.empty((0, n_sites_full), dtype=np.int64)
        result.pair_assignments = np.zeros((reads_array.shape[0], 2), dtype=np.int64)
        result.wildcard_mass = 1.0
        result.uncertainty_flag = True
        result.K_final = 0
        result.growth_history = []
        return result

    # --- 4. K-GROWTH WITH COORDINATE DESCENT + SUBTRACTION-RECOVERY ITERATION ---
    # Uses _grow_K_with_recovery (drop-in replacement for _grow_K) which
    # alternates K-growth and subtraction-recovery rounds until convergence.
    # Recovery catches founders that K-growth's worst-fit-sample seeding
    # missed (e.g., when K-growth gets stuck at a low K_final due to dirty
    # haps causing pseudo-convergence; see chr11:28698298, chr14:14665241).
    # Returns the same 7-tuple as _grow_K, so this is a transparent change.
    # Resolve the recovery caps.  None means "auto": take the larger of the
    # module-constant default and K_max, so raising the public K_max (e.g.
    # to support ~40 founders) automatically raises the recovery selection
    # and inner-mixture caps to match, WITHOUT changing the default-K_max
    # behaviour — at the default K_max=10, max(RECOVERY_MAX_K=12, 10)=12 and
    # max(RECOVERY_MIXTURE_K_MAX=10, 10)=10, i.e. the existing constants.
    # The mixture-sweep patience early-stop (recovery_mixture_patience) is
    # what keeps the average (small-true-K) cost from scaling with these
    # raised caps; without it the inner mixture would sweep K=1..cap on every
    # call regardless of how many founders the block actually needs.
    if recovery_max_K is None:
        recovery_max_K = max(RECOVERY_MAX_K, K_max)
    if recovery_mixture_K_max is None:
        recovery_mixture_K_max = max(RECOVERY_MIXTURE_K_MAX, K_max)

    H_k, A, per_sample_cost, wildcard_slots, K_final, wildcard_mass, history = \
        _grow_K_with_recovery(probs_k, kept_mask,
                              lam=lambda_wildcard_penalty,
                              wildcard_mass_threshold=wildcard_mass_threshold,
                              min_relative_improvement=min_wildcard_relative_improvement,
                              K_max=K_max,
                              max_iter_per_K=coord_descent_max_iter,
                              n_medoid_starts=n_medoid_starts,
                              recovery_max_K=recovery_max_K,
                              recovery_mixture_K_max=recovery_mixture_K_max,
                              recovery_mixture_patience=recovery_mixture_patience)

    # --- 5. COMPUTE PER-SITE CONFIDENCE (kept-site coords) ---
    # Re-check thread allocation before the one parallel kernel in the block
    # path (this confidence pass): after the long serial recovery above, peers
    # may have finished, so pick up any cores freed in the meantime here.
    thread_config.apply_dynamic_threads()
    confidence_k, n_supporters_k = _compute_per_site_confidence(
        probs_k, H_k, A, lam=lambda_wildcard_penalty,
        min_supporters=min_supporters_for_confidence)

    # --- 6. EXPAND BACK TO FULL-LENGTH COORDS ---
    K = H_k.shape[0]
    H_full = np.zeros((K, n_sites_full), dtype=np.int64)
    confidence_full = np.zeros((K, n_sites_full), dtype=np.float64)
    n_supporters_full = np.zeros((K, n_sites_full), dtype=np.int64)
    if kept_mask.any():
        kept_idx = np.where(kept_mask)[0]
        H_full[:, kept_idx] = H_k
        confidence_full[:, kept_idx] = confidence_k
        n_supporters_full[:, kept_idx] = n_supporters_k

    # --- 7. CONVERT TO LEGACY PROB-ARRAY FORMAT ---
    haps_dict = _discrete_haps_to_prob_arrays(
        H_full, n_sites_full, kept_mask,
        confidence_full, n_supporters_full,
        min_supporters=min_supporters_for_confidence)

    # --- 8. FINAL CLEANUP (legacy machinery, safety net) ---
    if len(haps_dict) > 1:
        cleaned = _final_cleanup(
            haps_dict, probs_array,
            diff_threshold_percent=diff_threshold_percent,
            penalty_strength=penalty_strength,
            chimera_max_recombs=chimera_max_recombs,
            chimera_max_mismatch_pct=chimera_max_mismatch_pct,
            chimera_min_delta_to_protect=chimera_min_delta_to_protect)
    else:
        cleaned = haps_dict

    # --- 9. APPLY MASK FOR LOW-SUPPORT SITES IN DISCRETE OUTPUT ---
    H_with_mask = H_full.copy()
    H_with_mask[n_supporters_full < min_supporters_for_confidence] = MASK

    # --- 10. UNCERTAINTY FLAG ---
    uncertainty_flag = (
        wildcard_mass > wildcard_mass_threshold * 2 or
        K_final == 0 or
        # If most founders are mostly MASK, we don't trust this block
        (K_final > 0 and (H_with_mask == MASK).any() and
         np.mean(H_with_mask == MASK) > 0.3)
    )

    # --- 11. CONSTRUCT RESULT ---
    result = BlockResult(positions, cleaned, reads_array,
                          keep_flags=keep_flags, probs_array=probs_array)
    result.discrete_haps = H_with_mask
    result.per_site_confidence = confidence_full
    result.n_site_supporters = n_supporters_full
    result.pair_assignments = A
    result.wildcard_mass = float(wildcard_mass)
    result.uncertainty_flag = bool(uncertainty_flag)
    result.K_final = int(K_final)
    result.growth_history = history

    _malloc_trim()
    return result


# =============================================================================
# find_missing_haplotypes_iterative — discrete-native residual founder discovery
# =============================================================================
# Discrete's own replacement for the residual step the robust wrapper used to
# borrow from block_haplotypes.find_missing_haplotypes_iterative.  Rather than
# transliterate the legacy choices (a k-limited recombination matcher for the
# fit check, the legacy clustering algorithm for re-generation, a flat 2%
# Hamming redundancy filter), this uses discrete's own machinery end to end:
#
#   * Detection.  Each sample is assigned to its best founder PAIR under
#     discrete's exact cost model via _update_A, with the wildcard founder as
#     the explicit "unexplained" state.  A sample is residual iff at least one
#     of its two strands lands on the wildcard sentinel — i.e. discrete itself,
#     under the same wildcard penalty `lambda_wildcard_penalty` it uses during
#     discovery, judges that no real founder explains that strand.  The penalty
#     IS the threshold, so there is no foreign error-percentage cutoff.
#
#   * Generation.  discrete coordinate-descent founder discovery
#     (generate_haplotypes_block) is run on just the residual samples, with the
#     same discrete parameters as the parent block.  This is the whole point of
#     retiring block_haplotypes: the residual pass now uses discrete's algorithm
#     too, so the founder set is internally consistent.
#
#   * Dedup.  A discovered founder is returned only if its minimum per-site
#     Hamming distance (over kept sites, via discrete's _hamming_pct_kept) to
#     every existing founder exceeds `dedup_threshold_percent` (discrete's hap-
#     equality tolerance by default), so only genuinely new founders propagate.
#
# Founder QUALITY is delegated to generate_haplotypes_block, which already gates
# emission on wildcard-mass / per-founder support — we deliberately do not add a
# second, redundant confidence filter here.
# =============================================================================

def find_missing_haplotypes_iterative(positions, reads_array, current_haps,
                                      keep_flags=None,
                                      lambda_wildcard_penalty=DEFAULT_LAMBDA,
                                      min_residual_samples=None,
                                      dedup_threshold_percent=RECOVERY_HAPS_EQUAL_EPS_PCT,
                                      **generation_kwargs):
    """Find founders the current set cannot explain, using discrete machinery.

    Assigns every sample to its best founder pair under discrete's cost model
    (with a wildcard founder for unexplained strands), runs discrete
    coordinate-descent discovery on the samples discrete leaves on the wildcard,
    and returns the founders that are not already present.

    Args:
        positions, reads_array, keep_flags: as for generate_haplotypes_block.
        current_haps: {key: (n_sites_full, 2) [P(0), P(1)]} current founder set.
        lambda_wildcard_penalty: wildcard penalty used both for the residual-
            detection assignment and for the residual discovery, kept consistent
            with the parent block.
        min_residual_samples: minimum number of wildcard-assigned samples before
            recovery is attempted.  None (default) resolves to 2x discrete's
            per-site confidence floor (min_supporters_for_confidence, default 2)
            so the trigger scales if that floor is raised; below it the
            unexplained signal is too weak to support a confident new founder.
        dedup_threshold_percent: a discovered founder is kept only if its minimum
            kept-site Hamming distance (%) to every existing founder exceeds
            this.  Defaults to discrete's hap-equality tolerance.
        **generation_kwargs: forwarded to the residual generate_haplotypes_block
            call (e.g. K_max, penalty_strength, min_supporters_for_confidence).
            `known_haplotypes` is dropped: residual discovery runs fresh and is
            deduped against `current_haps` afterwards.

    Returns:
        {new_idx: (n_sites_full, 2)} newly discovered founders (possibly empty),
        in the same [P(0), P(1)] format as generate_haplotypes_block output.
    """
    if len(current_haps) == 0:
        return {}

    n_sites_full = reads_array.shape[1]
    if keep_flags is None:
        keep_flags = np.ones(n_sites_full, dtype=np.int64)
    keep_flags = np.asarray(keep_flags, dtype=np.int64)
    kept_mask = keep_flags > 0
    if not kept_mask.any():
        return {}

    # Trigger floor: 2x discrete's per-site confidence requirement unless the
    # caller pins it explicitly.
    if min_residual_samples is None:
        min_supporters = int(generation_kwargs.get('min_supporters_for_confidence', 2))
        min_residual_samples = max(2, 2 * min_supporters)

    # Probs on kept sites only, materialised C-contiguous so the assignment /
    # CD kernels fast-path their contiguity checks (boolean masking yields a
    # non-contiguous view).  Pure layout change — values are identical.
    (_, probs_array) = analysis_utils.reads_to_probabilities(reads_array)
    probs_k = np.ascontiguousarray(probs_array[:, kept_mask, :])
    if probs_k.shape[0] == 0 or probs_k.shape[1] == 0:
        return {}

    # Current founders -> discrete {0, 1} matrix on kept sites.  This inverts
    # _discrete_haps_to_prob_arrays: P(1) >= P(0) -> allele 1.  Confident sites
    # (1,0)/(0,1) map back exactly; "no-information" (0.5, 0.5) sites map to 1
    # but carry no discriminative weight in the assignment because the data
    # there is uniform too.  _update_A requires {0, 1} (no MASK), which the
    # argmax guarantees.
    hap_keys = list(current_haps.keys())
    L_kept = int(kept_mask.sum())
    H_kept = np.empty((len(hap_keys), L_kept), dtype=np.int64)
    for i, k in enumerate(hap_keys):
        hp = np.asarray(current_haps[k], dtype=np.float64)
        H_kept[i] = (hp[:, 1][kept_mask] >= hp[:, 0][kept_mask]).astype(np.int64)
    H_kept = np.ascontiguousarray(H_kept)

    # Discrete-native fit: assign each sample to its best founder pair, with the
    # wildcard sentinel == K marking strands no real founder explains.
    K = H_kept.shape[0]
    A, _per_sample_cost, _per_sample_cost_unc, _wildcard_slots = _update_A(
        probs_k, H_kept, lambda_wildcard_penalty)

    # Residual = samples discrete cannot fully place: at least one strand on the
    # wildcard sentinel.
    residual_idx = np.where(np.any(A == K, axis=1))[0]
    if len(residual_idx) < min_residual_samples:
        return {}

    # Discrete founder discovery on the residual samples, fresh (deduped below)
    # and with the parent block's discrete parameters.
    generation_kwargs.pop('known_haplotypes', None)
    sub_block_result = generate_haplotypes_block(
        positions, reads_array[residual_idx], keep_flags=keep_flags,
        lambda_wildcard_penalty=lambda_wildcard_penalty,
        **generation_kwargs)

    # Keep only founders not already present: minimum kept-site Hamming (%) to
    # every existing founder must exceed the tolerance.
    newly_found_unique = {}
    new_idx = 0
    for sub_hap in sub_block_result.haplotypes.values():
        sub_arr = np.asarray(sub_hap, dtype=np.float64)
        sub_H = (sub_arr[:, 1][kept_mask] >= sub_arr[:, 0][kept_mask]).astype(np.int64)
        min_diff = 100.0
        for i in range(K):
            d = _hamming_pct_kept(sub_H, H_kept[i])
            if d < min_diff:
                min_diff = d
        if min_diff > dedup_threshold_percent:
            newly_found_unique[new_idx] = sub_hap
            new_idx += 1

    return newly_found_unique


# =============================================================================
# generate_haplotypes_block_robust — same iterative-residual-discovery
# wrapper, now calling our own discrete find_missing_haplotypes_iterative.
# =============================================================================

def generate_haplotypes_block_robust(positions, reads_array, keep_flags=None,
                                     max_robust_passes=3,
                                     **kwargs):
    """Wrapper that runs generate_haplotypes_block, checks for residuals
    (samples poorly fit by current set), and re-runs targeted generation
    on the residual subset until no new founders are found or
    max_robust_passes is exceeded.

    Mirrors the legacy generate_haplotypes_block_robust contract.
    """
    current_known_haps = kwargs.get('known_haplotypes', [])
    if isinstance(current_known_haps, dict):
        current_known_haps = list(current_known_haps.values())
    elif current_known_haps is None:
        current_known_haps = []

    final_result = None
    for pass_num in range(1, max_robust_passes + 1):
        run_kwargs = kwargs.copy()
        run_kwargs['known_haplotypes'] = current_known_haps

        final_result = generate_haplotypes_block(
            positions, reads_array, keep_flags=keep_flags, **run_kwargs)

        # Residual check: identify samples the current founder set cannot
        # explain — discrete's own wildcard assignment, not the legacy
        # k-limited matcher — and run discrete founder discovery on just those
        # (see find_missing_haplotypes_iterative above).  Forward this block's
        # generation parameters so the residual pass uses identical discrete
        # settings; lambda_wildcard_penalty binds to its explicit param and
        # known_haplotypes is dropped inside (residual discovery is fresh and
        # deduped).
        missing_haps_dict = find_missing_haplotypes_iterative(
            positions, reads_array, final_result.haplotypes,
            keep_flags=keep_flags, **kwargs)

        if len(missing_haps_dict) == 0:
            break

        new_haps_list = list(missing_haps_dict.values())
        combined = current_known_haps + new_haps_list
        consolidated = consolidate_similar_candidates(
            combined, diff_threshold_percent=0.01)
        current_known_haps = list(consolidated.values())

    return final_result


# =============================================================================
# WORKERS + ORCHESTRATOR
# =============================================================================

def _worker_generate_block_direct(args):
    """Worker function used by the forkserver pool.  Receives block data
    directly, returns (idx, result).  Matches the worker signature of
    block_haplotypes_em_foothold._worker_generate_block_direct so the
    orchestrator scaffolding can be reused."""
    block_idx, positions, reads, flags, kwargs = args

    # Register this worker as active, then take an initial thread allocation
    # from the shared dynamic-thread state.  The per-block recovery path
    # (here + bhd_recovery / bhd_trio) re-checks this allocation at every
    # phase boundary via thread_config.apply_dynamic_threads(), so a straggler
    # block grows into cores freed as its peers finish.
    if _BH_ACTIVE_COUNTER is not None:
        with _BH_ACTIVE_COUNTER.get_lock():
            _BH_ACTIVE_COUNTER.value += 1
    thread_config.apply_dynamic_threads()

    try:
        result = generate_haplotypes_block_robust(
            positions, reads, keep_flags=flags, **kwargs)
        _malloc_trim()
        return (block_idx, result)
    finally:
        # Release any held extra FIRST, then decrement the active counter, so
        # peers see the freed extra-slot before the decremented active count
        # (mirrors hierarchical_assembly).  The counter WIRING persists across
        # tasks (set once in _init_block_worker) for Pool worker reuse — only
        # the per-task extra-claim is released here.
        thread_config.release_dynamic_extra()
        if _BH_ACTIVE_COUNTER is not None:
            with _BH_ACTIVE_COUNTER.get_lock():
                _BH_ACTIVE_COUNTER.value -= 1


def generate_all_block_haplotypes(genomic_data,
                                    # Discrete coordinate descent parameters
                                    lambda_wildcard_penalty=DEFAULT_LAMBDA,
                                    wildcard_mass_threshold=0.0,
                                    min_wildcard_relative_improvement=0.10,
                                    K_max=10,
                                    coord_descent_max_iter=50,
                                    min_supporters_for_confidence=2,
                                    # Recovery caps + inner-mixture sweep
                                    # early-stop, forwarded to
                                    # generate_haplotypes_block.  recovery_max_K
                                    # / recovery_mixture_K_max default to None
                                    # ("auto" = max(module constant, K_max)), so
                                    # raising K_max for high-founder runs
                                    # auto-raises the recovery caps;
                                    # recovery_mixture_patience defaults to
                                    # RECOVERY_MIXTURE_PATIENCE (None disables).
                                    recovery_max_K=None,
                                    recovery_mixture_K_max=None,
                                    recovery_mixture_patience=RECOVERY_MIXTURE_PATIENCE,
                                    # Legacy params (used in final cleanup)
                                    diff_threshold_percent=1.0,
                                    penalty_strength=5.0,
                                    chimera_max_recombs=1,
                                    chimera_max_mismatch_pct=0.5,
                                    chimera_min_delta_to_protect=0.25,
                                    # Legacy params accepted but unused in inference
                                    uniqueness_threshold_percent=2.0,
                                    wrongness_threshold=10.0,
                                    max_intermediate_haps=100,
                                    num_processes=16,
                                    discard_reads_after=True):
    """Parallel orchestrator — drop-in replacement for the legacy
    generate_all_block_haplotypes contract."""
    from tqdm import tqdm

    kwargs = {
        'lambda_wildcard_penalty': lambda_wildcard_penalty,
        'wildcard_mass_threshold': wildcard_mass_threshold,
        'min_wildcard_relative_improvement': min_wildcard_relative_improvement,
        'K_max': K_max,
        'coord_descent_max_iter': coord_descent_max_iter,
        'min_supporters_for_confidence': min_supporters_for_confidence,
        'recovery_max_K': recovery_max_K,
        'recovery_mixture_K_max': recovery_mixture_K_max,
        'recovery_mixture_patience': recovery_mixture_patience,
        'diff_threshold_percent': diff_threshold_percent,
        'penalty_strength': penalty_strength,
        'chimera_max_recombs': chimera_max_recombs,
        'chimera_max_mismatch_pct': chimera_max_mismatch_pct,
        'chimera_min_delta_to_protect': chimera_min_delta_to_protect,
        'uniqueness_threshold_percent': uniqueness_threshold_percent,
        'wrongness_threshold': wrongness_threshold,
        'max_intermediate_haps': max_intermediate_haps,
    }

    n_blocks = len(genomic_data)
    task_args = []
    for i in range(n_blocks):
        positions, reads, flags = genomic_data[i]
        task_args.append((i, positions, reads, flags, kwargs))

    # Belt-and-suspenders: clear __main__ to prevent forkserver from
    # re-executing the entry script
    import sys as _sys
    _main_mod = _sys.modules.get('__main__')
    _saved_main_file = getattr(_main_mod, '__file__', None)
    _saved_main_spec = getattr(_main_mod, '__spec__', None)
    if _main_mod is not None:
        if hasattr(_main_mod, '__file__'):
            del _main_mod.__file__
        _main_mod.__spec__ = None

    try:
        active_counter = _forkserver_ctx.Value('i', 0)
        # extra_counter: # workers currently holding the +1 remainder thread,
        # so total threads in use across the pool stays == num_processes with
        # no idle cores as the active-block count changes (see thread_config's
        # dynamic-thread mechanism).
        extra_counter = _forkserver_ctx.Value('i', 0)
        with _ForkserverPool(processes=num_processes,
                              initializer=_init_block_worker,
                              initargs=(num_processes, active_counter, extra_counter)) as pool:
            results = []
            for result in tqdm(
                pool.imap_unordered(_worker_generate_block_direct, task_args, chunksize=1),
                total=n_blocks,
                desc="Block Haplotypes (discrete)"
            ):
                results.append(result)
    finally:
        if _main_mod is not None:
            if _saved_main_file is not None:
                _main_mod.__file__ = _saved_main_file
            _main_mod.__spec__ = _saved_main_spec

    results.sort(key=lambda x: x[0])
    overall_haplotypes = [r[1] for r in results]

    if discard_reads_after:
        for block in overall_haplotypes:
            block.reads_count_matrix = None
        gc.collect()

    return BlockResults(overall_haplotypes)


# =============================================================================
# MIGRATED FROM block_haplotypes.py (legacy block-hap discovery, retired).
# The BlockResult/BlockResults output types and the consolidate / optimal-set
# selection helpers now live here so the active ecosystem imports them from
# the discrete driver instead of from the legacy module.  Verbatim moves.
# =============================================================================

class BlockResult:
    """
    Container for the reconstructed haplotypes of a single genomic block.
    """
    def __init__(self, positions, haplotypes, reads_count_matrix=None, keep_flags=None, probs_array=None):
        self.positions = positions
        self.haplotypes = haplotypes # Dictionary {id: numpy_array}
        self.reads_count_matrix = reads_count_matrix # Optional: source reads (Samples x Sites x 2)
        self.keep_flags = keep_flags
        self.probs_array = probs_array # New Optional: genotype probabilities (Samples x Sites x 3)
        
    def __len__(self):
        return len(self.haplotypes)

    def __repr__(self):
        active_sites = np.sum(self.keep_flags) if self.keep_flags is not None else len(self.positions)
        has_probs = "with probs" if self.probs_array is not None else "no probs"
        return f"<BlockResult: {len(self.haplotypes)} haplotypes at {len(self.positions)} sites ({active_sites} active), {has_probs}>"

class BlockResults:
    """
    A container class holding a list of BlockResult objects, representing
    the reconstruction results for an entire genomic region.
    """
    def __init__(self, block_result_list):
        self.blocks = block_result_list

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, index):
        """Allows accessing blocks by index: block_results[i]"""
        return self.blocks[index]

    def __iter__(self):
        """Allows iterating: for block in block_results: ..."""
        return iter(self.blocks)

    def __repr__(self):
        return f"<BlockResults: containing {len(self.blocks)} processed blocks>"

def consolidate_similar_candidates(candidates, diff_threshold_percent=1.0):
    """
    Greedily merges candidates that are nearly identical.
    
    Args:
        candidates: dict or list of haplotype arrays.
        diff_threshold_percent: Percentage difference (0-100) below which to merge.
    """
    if not candidates: return {}
    
    # Normalize input to list of arrays
    if isinstance(candidates, dict):
        candidate_list = list(candidates.values())
    else:
        candidate_list = candidates

    unique_haps = []
    
    for hap in candidate_list:
        is_duplicate = False
        for existing in unique_haps:
            # Calculate Hamming distance (percentage of sites that differ)
            diff = np.mean(hap != existing) * 100.0
            
            if diff < diff_threshold_percent:
                # It's a duplicate (or noise variant)
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_haps.append(hap)
            
    # Rebuild dictionary with sequential keys
    return {i: h for i, h in enumerate(unique_haps)}

def select_optimal_haplotype_set_viterbi(candidate_haps, probs_array, 
                                         recomb_penalty=10.0,
                                         penalty_strength=1.0,
                                         read_error_prob=0.02,
                                         max_sites_for_selection=2000):
    """
    Selects the smallest set of haplotypes that explains the data best.
    """
    
    # --- 1. SETUP DATA ---
    if isinstance(candidate_haps, dict):
        hap_keys = list(candidate_haps.keys())
        H = np.array([candidate_haps[k] for k in hap_keys])
    else:
        hap_keys = list(range(len(candidate_haps)))
        H = np.array(candidate_haps)
        
    num_candidates = len(H)
    num_samples, total_sites, _ = probs_array.shape
    
    if num_candidates == 0: return []

    # --- 2. DOWNSAMPLING STRATEGY ---
    if total_sites > max_sites_for_selection:
        stride = math.ceil(total_sites / max_sites_for_selection)
        # Slice the data
        probs_active = probs_array[:, ::stride, :]
        H_active = H[:, ::stride, :]
    else:
        stride = 1
        probs_active = probs_array
        H_active = H

    num_active_sites = probs_active.shape[1]

    # --- 3. PRE-CALCULATE ALL PAIR LIKELIHOODS (MEMORY SAFE) ---
    idx_i, idx_j = np.triu_indices(num_candidates)
    num_pairs = len(idx_i)
    
    ll_tensor = np.empty((num_samples, num_pairs, num_active_sites), dtype=np.float32)
    
    W = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=np.float32)
    
    batch_size = 50
    
    for start_p in range(0, num_pairs, batch_size):
        end_p = min(start_p + batch_size, num_pairs)
        
        batch_idx_i = idx_i[start_p:end_p]
        batch_idx_j = idx_j[start_p:end_p]
        
        h0 = H_active[batch_idx_i, :, 0]
        h1 = H_active[batch_idx_i, :, 1]
        h2_0 = H_active[batch_idx_j, :, 0]
        h2_1 = H_active[batch_idx_j, :, 1]
        
        g00 = h0 * h2_0
        g11 = h1 * h2_1
        g01 = (h0 * h2_1) + (h1 * h2_0)
        del h0, h1, h2_0, h2_1
        
        batch_pairs = np.stack([g00, g01, g11], axis=-1)
        del g00, g01, g11
        
        weighted_pairs = batch_pairs @ W
        del batch_pairs
        
        batch_dist = np.sum(
            probs_active[:, np.newaxis, :, :] * weighted_pairs[np.newaxis, :, :, :],
            axis=3
        )
        del weighted_pairs
        
        ll_tensor[:, start_p:end_p, :] = (-batch_dist * stride)
        del batch_dist

    del H_active, probs_active
    ll_tensor = np.maximum(ll_tensor, -2.0 * stride)
    ll_tensor = np.ascontiguousarray(ll_tensor, dtype=np.float64)
    _malloc_trim()

    # --- 4. SELECTION LOOP ---
    
    selected_indices = []
    current_best_bic = float('inf')
    
    min_complexity = recomb_penalty * 1.5
    calculated_complexity = math.log(num_samples) * total_sites * penalty_strength * 0.01
    complexity_cost = max(calculated_complexity, min_complexity)
    
    while len(selected_indices) < num_candidates:
        _update_dynamic_threads()
        
        best_new_index = -1
        best_new_bic = float('inf')
        
        remaining = [x for x in range(num_candidates) if x not in selected_indices]
        
        for cand_idx in remaining:
            trial_set = selected_indices + [cand_idx]
            
            subset_mask = np.zeros(num_candidates, dtype=bool)
            subset_mask[trial_set] = True
            
            valid_pairs_mask = subset_mask[idx_i] & subset_mask[idx_j]
            
            if not np.any(valid_pairs_mask):
                continue
            
            active_ll_tensor = ll_tensor[:, valid_pairs_mask, :]
            
            best_scores = viterbi_score_selection(active_ll_tensor, float(recomb_penalty))
            
            total_log_likelihood = np.sum(best_scores)
            
            k = len(trial_set)
            bic = (k * complexity_cost) - (2 * total_log_likelihood)
            
            if bic < best_new_bic:
                best_new_bic = bic
                best_new_index = cand_idx
        
        if best_new_bic < current_best_bic:
            selected_indices.append(best_new_index)
            current_best_bic = best_new_bic
        else:
            break
            
    return [hap_keys[i] for i in selected_indices]