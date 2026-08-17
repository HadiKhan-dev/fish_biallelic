"""block_haplotypes.py — Discrete-hap founder discovery (orchestrator)

The top-level orchestration layer of stage-3 block-haplotype founder
discovery.  This file contains:

  - Forkserver pool scaffolding (_ForkserverPool, _init_block_worker)
  - generate_haplotypes_block, generate_haplotypes_block_robust — the
    per-block public entry points (these call bhd_kgrowth for founder
    discovery, then post-process)
  - find_missing_haplotypes_iterative — iterative missing-hap recovery
  - Output construction (re-exported from bhd_results)
  - _final_cleanup, consolidate_similar_candidates — chimera pruning,
    Viterbi subset selection, and candidate consolidation on the
    prob-array form
  - _worker_generate_block_direct, generate_all_block_haplotypes —
    the multi-process orchestrator over all blocks of a contig
  - BlockResult, BlockResults — result containers re-exported from bhd_results

The K-growth founder-discovery core (_grow_K, _initial_kgrowth_with_medoids,
_soft_cluster_seed_haps, _grow_K_with_recovery) now lives in bhd_kgrowth.py.
The atomic BIC/CD primitives, subtraction / trio / pairwise recovery, and
tuning constants are in bhd_kernels.py, bhd_recovery*.py, bhd_trio.py,
bhd_pairwise.py, and bhd_config.py respectively.  This file is
import-only-downstream — nothing in the split modules imports back from
here, which keeps the dependency DAG acyclic.

Public callers (e.g. pipeline.py) continue to use:
    import block_haplotypes as bhd
    bhd.generate_haplotypes_block(...)
    bhd.generate_all_block_haplotypes(...)
"""

import numpy as np
import warnings
import gc
import os
import ctypes

import numba

import thread_config
import dynamic_threads
from multiprocessing_runtime import (
    ForkserverPool as _ForkserverPool,
    forkserver_context as _forkserver_ctx,
    main_module_guard as _main_module_guard,
)

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
# attribute lookup (e.g. PAIRWISE_RECOVERY_ENABLED) to
# preserve runtime-mutation semantics.
from bhd_kernels import MASK
from bhd_fit import _fit_at_fixed_K, _update_A
from bhd_recovery_select import _hamming_pct_kept
from bhd_config import (
    DEFAULT_READ_ERROR_PROBABILITY,
    DEFAULT_LAMBDA,
    K_MEDOID_STARTS_DEFAULT,
    RECOVERY_HAPS_EQUAL_EPS_PCT,
    RECOVERY_MAX_K,
    RECOVERY_MIXTURE_K_MAX,
    RECOVERY_MIXTURE_PATIENCE,
)
from bhd_kgrowth import _grow_K_with_recovery
from bhd_results import (
    BlockResult,
    BlockResults,
    _compute_per_site_confidence,
    _compute_per_site_confidence_kernel,
    _discrete_haps_to_prob_arrays,
)

# BlockResult, BlockResults, and the output-materialization helpers are
# re-exported here for caller and checkpoint compatibility.  Candidate
# consolidation remains local below.  find_missing_haplotypes_iterative also
# lives locally and is invoked directly by the residual loop.  Dynamic-thread
# rescaling uses the standalone dynamic_threads module
# (apply_dynamic_threads), wired in _init_block_worker.

warnings.filterwarnings("ignore")
np.seterr(divide='ignore', invalid='ignore')

GENOTYPE_EVIDENCE_MODES = frozenset({"hwe_posterior", "raw_likelihood"})


def _validate_genotype_evidence_mode(mode):
    """Validate and return the genotype-evidence representation name."""
    if mode not in GENOTYPE_EVIDENCE_MODES:
        raise ValueError(
            "genotype_evidence_mode must be one of "
            f"{sorted(GENOTYPE_EVIDENCE_MODES)}"
        )
    return mode


def _validate_read_error_prob(read_error_prob):
    """Validate the per-read sequencing-error probability."""
    value = float(read_error_prob)
    if not np.isfinite(value) or not 0.0 < value < 0.5:
        raise ValueError("read_error_prob must be finite and between 0 and 0.5")
    return value


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

def _nonnegative_env_int(name, default):
    """Read a non-negative operational tuning value from the environment."""
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        return int(default)
    return value if value >= 0 else int(default)


# Retain reusable Numba/pattern-table arenas between blocks.  With the default
# 1.5-GiB per-worker threshold, even 112 workers remain well inside the normal
# 512-GiB production allocation; a 32-block periodic trim bounds fragmentation
# on smaller allocations and long-lived pools.  Both controls are operationally
# configurable without changing the scientific configuration or checkpoints.
_MALLOC_TRIM_RSS_BYTES = 1024 * 1024 * _nonnegative_env_int(
    "BHD_MALLOC_TRIM_RSS_MB", 1536
)
_MALLOC_TRIM_INTERVAL = _nonnegative_env_int(
    "BHD_MALLOC_TRIM_EVERY_BLOCKS", 32
)
_BLOCKS_SINCE_MALLOC_TRIM = 0
try:
    _RSS_PAGE_SIZE = int(os.sysconf("SC_PAGE_SIZE"))
except (AttributeError, OSError, ValueError):
    _RSS_PAGE_SIZE = 0


try:
    _libc = ctypes.CDLL("libc.so.6")

    def _trim_process_heap():
        _libc.malloc_trim(0)
except OSError:
    def _trim_process_heap():
        pass


def _current_process_rss_bytes():
    """Return current Linux RSS cheaply; zero when unavailable."""
    if not _RSS_PAGE_SIZE:
        return 0
    try:
        with open("/proc/self/statm", "rt", encoding="ascii") as handle:
            fields = handle.readline().split()
        return int(fields[1]) * _RSS_PAGE_SIZE
    except (OSError, ValueError, IndexError):
        return 0


def _maybe_malloc_trim(*, completed_block=False):
    """Trim only for high RSS or periodically after completed blocks.

    The old unconditional trim discarded reusable arenas after every block,
    forcing persistent workers to fault and allocate them again immediately.
    """
    global _BLOCKS_SINCE_MALLOC_TRIM
    if completed_block:
        _BLOCKS_SINCE_MALLOC_TRIM += 1
    over_rss_limit = (
        _MALLOC_TRIM_RSS_BYTES > 0
        and _current_process_rss_bytes() >= _MALLOC_TRIM_RSS_BYTES
    )
    periodic = (
        completed_block
        and _MALLOC_TRIM_INTERVAL > 0
        and _BLOCKS_SINCE_MALLOC_TRIM >= _MALLOC_TRIM_INTERVAL
    )
    if over_rss_limit or periodic:
        _trim_process_heap()
        _BLOCKS_SINCE_MALLOC_TRIM = 0




def _init_block_worker(
    total_cores,
    active_counter,
    extra_counter=None,
    started_counter=None,
    participant_counter=None,
    batch_generation=None,
    batch_task_count=None,
    startup_target=None,
    startup_ready=None,
):
    """Initializer for worker processes — sets up dynamic numba thread
    allocation based on number of currently-active workers.

    Wires dynamic_threads' shared dynamic-thread state, which is read by
    dynamic_threads.apply_dynamic_threads() at every
    phase boundary across this module + bhd_recovery + bhd_trio.  That lets a
    straggler block GROW into cores freed as its peers finish, instead of
    being pinned for its whole run to the thread count it got at start.
    extra_counter drives the remainder distribution (total threads in use ==
    total_cores, zero idle cores); None falls back to floor-only."""
    try:
        os.environ['NUMBA_NUM_THREADS'] = str(total_cores)
        numba.config.NUMBA_NUM_THREADS = total_cores
        numba.set_num_threads(1)
    except Exception:
        pass
    # Wire the shared state so every phase boundary across this module +
    # bhd_recovery + bhd_trio re-checks the SAME pool-wide active count.
    dynamic_threads.set_dynamic_thread_state(
        total_cores,
        active_counter,
        extra_counter,
        started_counter,
        participant_counter,
        batch_generation,
        batch_task_count,
        startup_target,
        startup_ready,
    )

def _validate_block_parallelism(
    num_processes: int,
    total_numba_threads: int | None,
) -> tuple[int, int]:
    if (
        isinstance(num_processes, bool)
        or int(num_processes) != num_processes
        or int(num_processes) < 1
    ):
        raise ValueError("num_processes must be a positive integer")
    processes = int(num_processes)
    total_threads = (
        processes
        if total_numba_threads is None
        else total_numba_threads
    )
    if (
        isinstance(total_threads, bool)
        or int(total_threads) != total_threads
        or int(total_threads) < processes
    ):
        raise ValueError(
            "total_numba_threads must be an integer at least as large as "
            "num_processes"
        )
    total_threads = int(total_threads)
    available_cpus = len(os.sched_getaffinity(0))
    if processes > available_cpus or total_threads > available_cpus:
        raise ValueError(
            "block workers and Numba thread budget must lie within the "
            "current CPU affinity"
        )
    return processes, total_threads


class BlockDiscoveryPool:
    """Reusable block-worker pool with one bounded dynamic thread budget."""

    def __init__(
        self,
        num_processes: int,
        total_numba_threads: int | None = None,
    ) -> None:
        (
            self.num_processes,
            self.total_numba_threads,
        ) = _validate_block_parallelism(
            num_processes, total_numba_threads
        )
        self._closed = False
        self._active_counter = _forkserver_ctx.Value("i", 0)
        self._extra_counter = _forkserver_ctx.Value("i", 0)
        self._started_counter = _forkserver_ctx.Value("i", 0)
        self._participant_counter = _forkserver_ctx.Value("i", 0)
        self._batch_generation = _forkserver_ctx.Value("i", 0)
        self._batch_task_count = _forkserver_ctx.Value("i", 0)
        self._startup_target = _forkserver_ctx.Value("i", 1)
        self._startup_ready = _forkserver_ctx.Value("i", 0)
        self._batch_in_progress = False

        # Prevent forkserver workers from re-executing a pipeline entry point.
        with _main_module_guard():
            self._pool = _ForkserverPool(
                processes=self.num_processes,
                initializer=_init_block_worker,
                initargs=(
                    self.total_numba_threads,
                    self._active_counter,
                    self._extra_counter,
                    self._started_counter,
                    self._participant_counter,
                    self._batch_generation,
                    self._batch_task_count,
                    self._startup_target,
                    self._startup_ready,
                ),
            )

    @staticmethod
    def _store_counter(counter, value):
        with counter.get_lock():
            counter.get_obj().value = int(value)

    def _prepare_task_batch(self, n_tasks):
        if self._batch_in_progress:
            raise RuntimeError(
                "block-discovery pool already has an unfinished task batch"
            )
        if self._active_counter.value != 0:
            raise RuntimeError(
                "block-discovery workers from the preceding batch are still active"
            )
        task_count = int(n_tasks)
        initial_target = min(task_count, self.num_processes)
        self._store_counter(self._started_counter, 0)
        self._store_counter(self._participant_counter, 0)
        self._store_counter(self._batch_task_count, task_count)
        self._store_counter(self._startup_target, initial_target)
        self._store_counter(self._startup_ready, initial_target <= 1)
        with self._batch_generation.get_lock():
            generation = self._batch_generation.get_obj()
            generation.value += 1

    def imap_unordered(self, tasks):
        if self._closed:
            raise RuntimeError("block-discovery pool is closed")
        if not hasattr(tasks, "__len__"):
            tasks = tuple(tasks)
        self._prepare_task_batch(len(tasks))
        try:
            raw_results = self._pool.imap_unordered(
                _worker_generate_block_direct, tasks, chunksize=1
            )
        except Exception:
            raise
        self._batch_in_progress = True

        def consume_batch():
            completed = False
            try:
                for result in raw_results:
                    yield result
                completed = True
            finally:
                # An abandoned or failed iterator may still have queued work.
                # Keep the pool guarded in that case; close/terminate remains
                # available, but a new batch cannot corrupt the startup gate.
                if completed:
                    self._batch_in_progress = False

        return consume_batch()

    def close(self) -> None:
        if self._closed:
            return
        self._pool.close()
        self._pool.join()
        self._closed = True

    def terminate(self) -> None:
        if self._closed:
            return
        self._pool.terminate()
        self._pool.join()
        self._closed = True

    def __enter__(self) -> "BlockDiscoveryPool":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self.terminate()











# =============================================================================
# OUTPUT CONSTRUCTION
# =============================================================================



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
    # Legacy probability-array cleanup only.  Keep its pandas-bearing helper
    # outside the reversible worker startup path.
    import hap_statistics

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


def _cleanup_source_indices(haps_dict, cleaned):
    """Map cleanup survivors back to their pre-cleanup discrete rows.

    ``_final_cleanup`` only retains references to input candidate arrays; it
    does not synthesize or average rows. Recovering those source coordinates
    lets us seed a fresh discrete fit whenever cleanup changes K or row order.
    Identity is preferred so byte-identical duplicate rows remain unambiguous;
    exact equality is a defensive fallback for callers that copy arrays.
    """
    source_rows = list(haps_dict.values())
    used = set()
    indices = []
    for cleaned_row in cleaned.values():
        match = next(
            (
                index
                for index, source_row in enumerate(source_rows)
                if index not in used and cleaned_row is source_row
            ),
            None,
        )
        if match is None:
            match = next(
                (
                    index
                    for index, source_row in enumerate(source_rows)
                    if index not in used
                    and np.array_equal(cleaned_row, source_row)
                ),
                None,
            )
        if match is None:
            raise AssertionError(
                "final cleanup returned a haplotype outside its input rows"
            )
        used.add(match)
        indices.append(match)
    return tuple(indices)


def _refit_cleaned_discrete_model(probs_k, H_seed,
                                  lambda_wildcard_penalty,
                                  coord_descent_max_iter):
    """Refit cleanup survivors and remove any rows that collapse on refit."""
    n_samples = probs_k.shape[0]
    H = np.ascontiguousarray(H_seed, dtype=np.int64)
    if len(H) == 0:
        A = np.zeros((n_samples, 2), dtype=np.int64)
        wildcard_slots = np.full(n_samples, 2, dtype=np.int64)
        return H, A, wildcard_slots

    H, A, _costs, wildcard_slots, _n_iter, _nll = _fit_at_fixed_K(
        probs_k,
        H,
        lambda_wildcard_penalty,
        max_iter=coord_descent_max_iter,
    )

    # A fixed-K refit can make two seeds identical or leave a component with
    # no assigned strand. Prune and refit until row and assignment coordinates
    # stabilize, matching the cleanup's zero-use intent.
    while len(H) > 0:
        real_assignments = A[A < len(H)]
        usage = np.bincount(real_assignments, minlength=len(H))
        keep = []
        for index in range(len(H)):
            if usage[index] == 0:
                continue
            if any(np.array_equal(H[index], H[prior]) for prior in keep):
                continue
            keep.append(index)
        if len(keep) == len(H):
            break
        H = np.ascontiguousarray(H[np.asarray(keep, dtype=np.int64)])
        if len(H) == 0:
            A = np.zeros((n_samples, 2), dtype=np.int64)
            wildcard_slots = np.full(n_samples, 2, dtype=np.int64)
            break
        H, A, _costs, wildcard_slots, _n_iter, _nll = _fit_at_fixed_K(
            probs_k,
            H,
            lambda_wildcard_penalty,
            max_iter=coord_descent_max_iter,
        )

    return (
        np.ascontiguousarray(H),
        np.ascontiguousarray(A),
        np.ascontiguousarray(wildcard_slots),
    )


def _validate_synchronized_block_result(result, min_supporters):
    """Raise if public and discrete block payloads use different models."""
    reads = np.asarray(result.reads_count_matrix)
    n_samples, n_sites, allele_count = reads.shape
    if allele_count != 2:
        raise AssertionError("reads_count_matrix must end in a ref/alt axis")
    k = int(result.K_final)
    if len(result.positions) != n_sites:
        raise AssertionError("positions and reads disagree on site count")
    if tuple(sorted(result.haplotypes)) != tuple(range(k)):
        raise AssertionError("haplotype keys must be contiguous 0..K-1")
    if result.discrete_haps.shape != (k, n_sites):
        raise AssertionError("discrete_haps and K_final disagree")
    if result.per_site_confidence.shape != (k, n_sites):
        raise AssertionError("per-site confidence and K_final disagree")
    if result.n_site_supporters.shape != (k, n_sites):
        raise AssertionError("per-site support and K_final disagree")

    assignments = np.asarray(result.pair_assignments)
    if assignments.shape != (n_samples, 2):
        raise AssertionError("pair_assignments has the wrong shape")
    if np.any(assignments < 0) or np.any(assignments > k):
        raise AssertionError("pair_assignments contains an invalid final index")
    if np.any(assignments[:, 0] > assignments[:, 1]):
        raise AssertionError("pair_assignments must be canonically sorted")
    expected_wildcard_mass = (
        float(np.count_nonzero(assignments == k)) / max(2 * n_samples, 1)
    )
    if not np.isclose(
        float(result.wildcard_mass),
        expected_wildcard_mass,
        rtol=0.0,
        atol=np.finfo(np.float64).eps,
    ):
        raise AssertionError("wildcard_mass and pair_assignments disagree")

    supporters = np.asarray(result.n_site_supporters)
    expected_mask = supporters < int(min_supporters)
    if not np.array_equal(result.discrete_haps == MASK, expected_mask):
        raise AssertionError("discrete MASK cells and support counts disagree")
    for index in range(k):
        haplotype = np.asarray(result.haplotypes[index])
        if haplotype.shape != (n_sites, 2):
            raise AssertionError("a public haplotype has the wrong shape")
        if not np.all(haplotype[expected_mask[index]] == 0.5):
            raise AssertionError("low-support public cells must be uninformative")
        known = ~expected_mask[index]
        if np.any(known):
            expected_alt = result.discrete_haps[index, known]
            if not np.array_equal(haplotype[known, 1], expected_alt):
                raise AssertionError(
                    "public and discrete haplotype alleles disagree"
                )
            if not np.array_equal(haplotype[known, 0], 1 - expected_alt):
                raise AssertionError(
                    "public and discrete haplotype alleles disagree"
                )

    candidate_rows = np.asarray(result.precleanup_candidate_discrete_haps)
    if candidate_rows.ndim != 2 or candidate_rows.shape[1] != n_sites:
        raise AssertionError(
            "pre-cleanup candidate rows use wrong site coordinates"
        )
    if int(result.precleanup_candidate_k) != candidate_rows.shape[0]:
        raise AssertionError("pre-cleanup candidate K and rows disagree")
    if candidate_rows.shape[0] < k:
        raise AssertionError(
            "pre-cleanup candidate K cannot be smaller than final K"
        )
    if np.shares_memory(candidate_rows, np.asarray(result.discrete_haps)):
        raise AssertionError(
            "pre-cleanup candidate rows must not alias final rows"
        )
    if not np.all(
        (candidate_rows == 0) | (candidate_rows == 1) | (candidate_rows == MASK)
    ):
        raise AssertionError("pre-cleanup candidate rows contain invalid alleles")
    candidate_filtered = (
        np.zeros(n_sites, dtype=bool)
        if result.keep_flags is None
        else np.asarray(result.keep_flags) <= 0
    )
    if np.any(candidate_filtered) and not np.all(
        candidate_rows[:, candidate_filtered] == MASK
    ):
        raise AssertionError(
            "pre-cleanup candidate rows must mask filtered sites"
        )


def _selftest_cleanup_synchronization():
    """Focused regression checks for cleanup/refit result coordinates."""
    duplicate_a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    duplicate_b = duplicate_a.copy()
    other = np.array([[0.5, 0.5], [1.0, 0.0]], dtype=np.float64)
    source = {0: duplicate_a, 1: duplicate_b, 2: other}

    # Identity wins over an earlier equal row; cleanup may drop and reorder.
    assert _cleanup_source_indices(source, {0: duplicate_b}) == (1,)
    assert _cleanup_source_indices(
        source, {0: other, 1: duplicate_b}
    ) == (2, 1)
    # Equality fallback consumes each duplicate source coordinate only once.
    assert _cleanup_source_indices(
        source, {0: duplicate_a.copy(), 1: duplicate_b.copy()}
    ) == (0, 1)
    try:
        _cleanup_source_indices(
            source,
            {
                0: duplicate_a.copy(),
                1: duplicate_b.copy(),
                2: duplicate_a.copy(),
            },
        )
    except AssertionError as exc:
        assert "outside its input rows" in str(exc)
    else:
        raise AssertionError("cleanup accepted an excess duplicate row")

    # Decisive homozygotes make one duplicate and one unsupported seed
    # removable. Final assignments must use the refitted K=2 coordinates.
    probs = np.full((4, 3, 3), 0.0005, dtype=np.float64)
    probs[:2, :, 0] = 0.999
    probs[2:, :, 2] = 0.999
    seed_haps = np.array(
        [[0, 0, 0], [0, 0, 0], [1, 1, 1], [0, 1, 0]],
        dtype=np.int64,
    )
    final_haps, assignments, wildcard_slots = (
        _refit_cleaned_discrete_model(
            probs, seed_haps, DEFAULT_LAMBDA, 20
        )
    )
    assert np.array_equal(
        final_haps, np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
    )
    assert np.array_equal(
        assignments,
        np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.int64),
    )
    assert np.array_equal(wildcard_slots, np.zeros(4, dtype=np.int64))

    supporters = np.full(final_haps.shape, 2, dtype=np.int64)
    confidence = np.ones(final_haps.shape, dtype=np.float64)
    public_haps = _discrete_haps_to_prob_arrays(
        final_haps,
        final_haps.shape[1],
        np.ones(final_haps.shape[1], dtype=bool),
        confidence,
        supporters,
        min_supporters=2,
    )
    result = BlockResult(
        np.arange(final_haps.shape[1], dtype=np.int64),
        public_haps,
        np.zeros(
            (len(assignments), final_haps.shape[1], 2), dtype=np.int64
        ),
    )
    result.discrete_haps = final_haps.copy()
    result.per_site_confidence = confidence
    result.n_site_supporters = supporters
    result.pair_assignments = assignments
    result.wildcard_mass = 0.0
    result.K_final = len(final_haps)
    result.precleanup_candidate_discrete_haps = seed_haps.copy()
    result.precleanup_candidate_k = len(seed_haps)
    _validate_synchronized_block_result(result, min_supporters=2)

    # Candidate-only provenance deliberately keeps a single-carrier binary
    # row on retained sites even though the public two-supporter representation
    # remains completely uncertainty-masked.
    singleton_haplotype = np.array([[0, 1, 0]], dtype=np.int64)
    singleton_keep = np.array([1, 0, 1], dtype=np.int64)
    singleton_supporters = np.array([[1, 0, 1]], dtype=np.int64)
    singleton_confidence = np.ones((1, 3), dtype=np.float64)
    singleton_public = _discrete_haps_to_prob_arrays(
        singleton_haplotype,
        3,
        singleton_keep > 0,
        singleton_confidence,
        singleton_supporters,
        min_supporters=2,
    )
    singleton_result = BlockResult(
        np.arange(3, dtype=np.int64),
        singleton_public,
        np.zeros((1, 3, 2), dtype=np.int64),
        keep_flags=singleton_keep,
    )
    singleton_result.discrete_haps = np.full((1, 3), MASK, dtype=np.int64)
    singleton_result.per_site_confidence = singleton_confidence
    singleton_result.n_site_supporters = singleton_supporters
    singleton_result.pair_assignments = np.array([[0, 0]], dtype=np.int64)
    singleton_result.wildcard_mass = 0.0
    singleton_result.K_final = 1
    singleton_result.precleanup_candidate_discrete_haps = np.array(
        [[0, MASK, 0]], dtype=np.int64
    )
    singleton_result.precleanup_candidate_k = 1
    _validate_synchronized_block_result(singleton_result, min_supporters=2)
    assert np.all(singleton_result.haplotypes[0] == 0.5)
    assert np.array_equal(
        singleton_result.precleanup_candidate_discrete_haps,
        np.array([[0, MASK, 0]], dtype=np.int64),
    )

    def expect_validation_failure(expected_text):
        try:
            _validate_synchronized_block_result(result, min_supporters=2)
        except AssertionError as exc:
            assert expected_text in str(exc), str(exc)
        else:
            raise AssertionError(
                f"validator accepted invalid provenance: {expected_text}"
            )

    result.precleanup_candidate_k = len(seed_haps) - 1
    expect_validation_failure("candidate K and rows disagree")
    result.precleanup_candidate_discrete_haps = final_haps[:1].copy()
    result.precleanup_candidate_k = 1
    expect_validation_failure("cannot be smaller than final K")
    result.precleanup_candidate_discrete_haps = seed_haps.copy()
    result.precleanup_candidate_discrete_haps[0, 0] = 7
    result.precleanup_candidate_k = len(seed_haps)
    expect_validation_failure("contain invalid alleles")
    result.precleanup_candidate_discrete_haps = result.discrete_haps
    result.precleanup_candidate_k = len(final_haps)
    expect_validation_failure("must not alias final rows")

    # Exercise every production branch that previously aliased provenance.
    empty = generate_haplotypes_block(
        np.empty(0, dtype=np.int64),
        np.empty((0, 0, 2), dtype=np.int64),
    )
    assert empty.precleanup_candidate_discrete_haps is not empty.discrete_haps
    no_kept = generate_haplotypes_block(
        np.arange(3, dtype=np.int64),
        np.ones((2, 3, 2), dtype=np.int64),
        keep_flags=np.zeros(3, dtype=np.int64),
    )
    assert (
        no_kept.precleanup_candidate_discrete_haps
        is not no_kept.discrete_haps
    )
    simple_reads = np.array(
        [
            [[8, 0]] * 3,
            [[7, 0]] * 3,
            [[0, 8]] * 3,
            [[0, 7]] * 3,
        ],
        dtype=np.int64,
    )
    unchanged = generate_haplotypes_block(
        np.arange(3, dtype=np.int64),
        simple_reads,
        K_max=2,
        n_medoid_starts=1,
        recovery_max_K=2,
        recovery_mixture_K_max=2,
        recovery_mixture_patience=1,
        min_supporters_for_confidence=1,
        coord_descent_max_iter=10,
    )
    assert unchanged.precleanup_candidate_k == unchanged.K_final
    assert (
        unchanged.precleanup_candidate_discrete_haps
        is not unchanged.discrete_haps
    )
    assert not np.shares_memory(
        unchanged.precleanup_candidate_discrete_haps,
        unchanged.discrete_haps,
    )


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
                              wrongness_threshold=10.0,
                              genotype_evidence_mode="hwe_posterior",
                              read_error_prob=DEFAULT_READ_ERROR_PROBABILITY):
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
        result.precleanup_candidate_discrete_haps:
                                     candidate-only (K_pre, L_full) rows from
                                     K-growth, before cleanup/refit; these are
                                     provenance and do not share final assignment
                                     coordinates when cleanup changes the model

    The `haplotypes` attribute uses the legacy (n_sites_full, 2)
    [P(0), P(1)] format for backward compat.
    """
    genotype_evidence_mode = _validate_genotype_evidence_mode(
        genotype_evidence_mode
    )
    read_error_prob = _validate_read_error_prob(read_error_prob)
    n_sites_full = reads_array.shape[1]

    # --- 1. SETUP ---
    if keep_flags is None:
        keep_flags = np.ones(n_sites_full, dtype=np.int64)
    if keep_flags.dtype != int:
        keep_flags = np.asarray(keep_flags, dtype=np.int64)
    kept_mask = keep_flags > 0

    # --- 2. PROBS FROM READS ---
    # Probability conversion is needed only after a block enters this legacy
    # generator; reversible worker startup does not require analysis_utils.
    import analysis_utils

    site_priors, probs_array = analysis_utils.reads_to_probabilities(
        reads_array,
        read_error_prob=read_error_prob,
        use_hwe_prior=(genotype_evidence_mode == "hwe_posterior"),
    )

    if len(positions) == 0:
        empty_haps = {}
        result = BlockResult(np.array([]), empty_haps, reads_array,
                             keep_flags=keep_flags, probs_array=probs_array,
                             genotype_evidence_mode=genotype_evidence_mode)
        result.discrete_haps = np.empty((0, 0), dtype=np.int64)
        result.per_site_confidence = np.empty((0, 0), dtype=np.float64)
        result.n_site_supporters = np.empty((0, 0), dtype=np.int64)
        result.pair_assignments = np.empty((0, 2), dtype=np.int64)
        result.wildcard_mass = 0.0
        result.uncertainty_flag = True
        result.K_final = 0
        result.growth_history = []
        result.precleanup_candidate_discrete_haps = result.discrete_haps.copy()
        result.precleanup_candidate_k = 0
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
                             keep_flags=keep_flags, probs_array=probs_array,
                             genotype_evidence_mode=genotype_evidence_mode)
        result.discrete_haps = np.empty((0, n_sites_full), dtype=np.int64)
        result.per_site_confidence = np.empty((0, n_sites_full), dtype=np.float64)
        result.n_site_supporters = np.empty((0, n_sites_full), dtype=np.int64)
        result.pair_assignments = np.zeros((reads_array.shape[0], 2), dtype=np.int64)
        result.wildcard_mass = 1.0
        result.uncertainty_flag = True
        result.K_final = 0
        result.growth_history = []
        result.precleanup_candidate_discrete_haps = result.discrete_haps.copy()
        result.precleanup_candidate_k = 0
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
    dynamic_threads.apply_dynamic_threads()
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

    # Candidate-only provenance: preserve the training-fitted binary rows on
    # every retained site before public-output uncertainty masking.  A genuine
    # haplotype carried by one individual has only one supporter by design;
    # applying the public two-supporter mask here would erase that entire row
    # before a later nested held-out selector could test it.  Filtered sites
    # remain MASK, while standard output fields keep the conservative mask.
    precleanup_candidate_discrete_haps = H_full.copy()
    precleanup_candidate_discrete_haps[:, ~kept_mask] = MASK

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

    cleaned_source_indices = _cleanup_source_indices(haps_dict, cleaned)
    cleanup_changed_model = (
        cleaned_source_indices != tuple(range(H_k.shape[0]))
    )
    if cleanup_changed_model:
        H_seed = np.ascontiguousarray(
            H_k[np.asarray(cleaned_source_indices, dtype=np.int64)]
        )
        H_k, A, wildcard_slots = _refit_cleaned_discrete_model(
            probs_k,
            H_seed,
            lambda_wildcard_penalty,
            coord_descent_max_iter,
        )
        K_final = int(H_k.shape[0])
        wildcard_mass = (
            float(np.sum(wildcard_slots, dtype=np.int64))
            / max(2 * probs_k.shape[0], 1)
        )
        if K_final:
            confidence_k, n_supporters_k = _compute_per_site_confidence(
                probs_k,
                H_k,
                A,
                lam=lambda_wildcard_penalty,
                min_supporters=min_supporters_for_confidence,
            )
        else:
            confidence_k = np.empty(
                (0, probs_k.shape[1]), dtype=np.float64
            )
            n_supporters_k = np.empty(
                (0, probs_k.shape[1]), dtype=np.int64
            )

        H_full = np.zeros((K_final, n_sites_full), dtype=np.int64)
        confidence_full = np.zeros(
            (K_final, n_sites_full), dtype=np.float64
        )
        n_supporters_full = np.zeros(
            (K_final, n_sites_full), dtype=np.int64
        )
        if kept_mask.any():
            kept_idx = np.where(kept_mask)[0]
            H_full[:, kept_idx] = H_k
            confidence_full[:, kept_idx] = confidence_k
            n_supporters_full[:, kept_idx] = n_supporters_k
        cleaned = _discrete_haps_to_prob_arrays(
            H_full,
            n_sites_full,
            kept_mask,
            confidence_full,
            n_supporters_full,
            min_supporters=min_supporters_for_confidence,
        )

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
                         keep_flags=keep_flags, probs_array=probs_array,
                         genotype_evidence_mode=genotype_evidence_mode)
    result.discrete_haps = H_with_mask
    result.per_site_confidence = confidence_full
    result.n_site_supporters = n_supporters_full
    result.pair_assignments = A
    result.wildcard_mass = float(wildcard_mass)
    result.uncertainty_flag = bool(uncertainty_flag)
    result.K_final = int(K_final)
    result.growth_history = history
    result.precleanup_candidate_discrete_haps = (
        precleanup_candidate_discrete_haps.copy()
    )
    result.precleanup_candidate_k = int(
        result.precleanup_candidate_discrete_haps.shape[0]
    )
    _validate_synchronized_block_result(
        result, min_supporters_for_confidence
    )

    _maybe_malloc_trim()
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
                                      read_error_prob=DEFAULT_READ_ERROR_PROBABILITY,
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
    evidence_mode = _validate_genotype_evidence_mode(
        generation_kwargs.get('genotype_evidence_mode', 'hwe_posterior')
    )
    read_error_prob = _validate_read_error_prob(read_error_prob)
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
    # Residual recovery is a legacy compatibility path and imports its
    # probability conversion machinery only if the path is actually used.
    import analysis_utils

    (_, probs_array) = analysis_utils.reads_to_probabilities(
        reads_array,
        read_error_prob=read_error_prob,
        use_hwe_prior=(evidence_mode == 'hwe_posterior'),
    )
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
        read_error_prob=read_error_prob,
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
    # Register before decoding the task so even an early malformed-task failure
    # contributes to batch-start accounting and is always balanced in finally.
    # Later recovery phases re-check the live allocation, allowing true tail
    # stragglers to consume cores released by completed workers.
    dynamic_threads.increment_active()
    dynamic_threads.apply_dynamic_threads()

    try:
        block_idx, positions, reads, flags, kwargs, discard_reads_after = args
        worker_kwargs = kwargs.copy()
        reversible_cavity_config = worker_kwargs.pop(
            'reversible_cavity_config', None
        )

        has_kept_sites = (
            flags is None or np.any(np.asarray(flags) > 0)
        )
        if len(positions) == 0 or not has_kept_sites:
            # Preserve the historical empty/no-kept result construction; the
            # reversible protocol intentionally requires a kept site.
            result = generate_haplotypes_block_robust(
                positions, reads, keep_flags=flags, **worker_kwargs
            )
        elif reversible_cavity_config is not None:
            from bhd_reversible_discovery import (
                discover_block_reversible_cavity,
            )

            cavity_result = discover_block_reversible_cavity(
                positions,
                reads,
                keep_flags=flags,
                config=reversible_cavity_config,
            )
            result = cavity_result.to_block_result(
                block_result_class=BlockResult
            )
        else:
            result = generate_haplotypes_block_robust(
                positions, reads, keep_flags=flags, **worker_kwargs
            )
        # When callers discard the read matrix, clear it before the result is
        # serialized through the multiprocessing pipe.  The parent has always
        # observed ``None`` in this mode; doing the same operation here avoids
        # transmitting roughly samples * sites * 2 int64 values per block.
        if discard_reads_after:
            result.reads_count_matrix = None
        _maybe_malloc_trim(completed_block=True)
        return (block_idx, result)
    finally:
        # Release any held extra FIRST, then decrement the active counter, so
        # peers see the freed extra-slot before the decremented active count
        # (mirrors hierarchical_assembly).  The counter WIRING persists across
        # tasks (set once in _init_block_worker) for Pool worker reuse — only
        # the per-task extra-claim is released here.
        dynamic_threads.release_dynamic_extra()
        dynamic_threads.decrement_active()


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
                                    discard_reads_after=True,
                                    genotype_evidence_mode="hwe_posterior",
                                    read_error_prob=DEFAULT_READ_ERROR_PROBABILITY,
                                    total_numba_threads=None,
                                    block_pool=None,
                                    reversible_cavity_config=None):
    """Parallel orchestrator — drop-in replacement for the legacy
    generate_all_block_haplotypes contract."""
    from tqdm import tqdm

    genotype_evidence_mode = _validate_genotype_evidence_mode(
        genotype_evidence_mode
    )
    read_error_prob = _validate_read_error_prob(read_error_prob)

    num_processes, total_numba_threads = _validate_block_parallelism(
        num_processes, total_numba_threads
    )
    if block_pool is not None:
        if not isinstance(block_pool, BlockDiscoveryPool):
            raise TypeError("block_pool must be a BlockDiscoveryPool")
        if (
            block_pool.num_processes != num_processes
            or block_pool.total_numba_threads != total_numba_threads
        ):
            raise ValueError(
                "block_pool worker and thread budgets must match this call"
            )

    kwargs = {
        'lambda_wildcard_penalty': lambda_wildcard_penalty,
        'wildcard_mass_threshold': wildcard_mass_threshold,
        'min_wildcard_relative_improvement': min_wildcard_relative_improvement,
        'K_max': K_max,
        'coord_descent_max_iter': coord_descent_max_iter,
        'min_supporters_for_confidence': min_supporters_for_confidence,
        'genotype_evidence_mode': genotype_evidence_mode,
        'read_error_prob': read_error_prob,
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
        'reversible_cavity_config': reversible_cavity_config,
    }

    n_blocks = len(genomic_data)
    task_args = []
    for i in range(n_blocks):
        positions, reads, flags = genomic_data[i]
        task_args.append((
            i,
            positions,
            reads,
            flags,
            kwargs,
            bool(discard_reads_after),
        ))

    description = (
        "Block Haplotypes (reversible cavity)"
        if reversible_cavity_config is not None
        else "Block Haplotypes (discrete)"
    )

    def collect(pool):
        return list(tqdm(
            pool.imap_unordered(task_args),
            total=n_blocks,
            desc=description,
        ))

    if block_pool is None:
        with BlockDiscoveryPool(
            num_processes, total_numba_threads
        ) as local_pool:
            results = collect(local_pool)
    else:
        results = collect(block_pool)

    results.sort(key=lambda x: x[0])
    overall_haplotypes = [r[1] for r in results]

    if discard_reads_after:
        for block in overall_haplotypes:
            block.reads_count_matrix = None
        gc.collect()

    return BlockResults(overall_haplotypes)


# =============================================================================
# COMPATIBILITY HELPERS
# Candidate consolidation and optimal-set selection remain here because the
# discrete discovery path still uses them.  Neutral result containers and
# output materialization live in bhd_results and are re-exported above so old
# imports and historical checkpoint class paths continue to resolve.
# =============================================================================

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


def _selftest_reversible_orchestration():
    """Check reversible, default-legacy, and empty-block dispatch."""
    import bhd_reversible_discovery as reversible_discovery

    class InlineBlockDiscoveryPool(BlockDiscoveryPool):
        def __init__(self):
            self.num_processes = 1
            self.total_numba_threads = 1
            self.tasks = []

        def imap_unordered(self, tasks):
            self.tasks.extend(tasks)
            for task in tasks:
                yield _worker_generate_block_direct(task)

    reversible_calls = []
    materialization_classes = []
    legacy_calls = []

    class FakeReversibleResult:
        def __init__(self, positions, reads, flags):
            self.positions = positions
            self.reads = reads
            self.flags = flags

        def to_block_result(self, *, block_result_class=None):
            materialization_classes.append(block_result_class)
            result = block_result_class(
                self.positions,
                {},
                self.reads,
                keep_flags=self.flags,
            )
            result.dispatch_marker = "reversible"
            return result

    def fake_reversible_discovery(
        positions,
        reads,
        keep_flags=None,
        *,
        config=None,
    ):
        reversible_calls.append((positions, reads, keep_flags, config))
        return FakeReversibleResult(positions, reads, keep_flags)

    def fake_legacy_discovery(
        positions,
        reads,
        keep_flags=None,
        **kwargs,
    ):
        legacy_calls.append((positions, reads, keep_flags, kwargs))
        assert "reversible_cavity_config" not in kwargs
        result = BlockResult(
            positions,
            {},
            reads,
            keep_flags=keep_flags,
        )
        result.dispatch_marker = "legacy"
        return result

    original_reversible = (
        reversible_discovery.discover_block_reversible_cavity
    )
    original_legacy = globals()["generate_haplotypes_block_robust"]
    reversible_discovery.discover_block_reversible_cavity = (
        fake_reversible_discovery
    )
    globals()["generate_haplotypes_block_robust"] = fake_legacy_discovery
    try:
        positions = np.asarray([10, 20], dtype=np.int64)
        reads = np.zeros((1, 2, 2), dtype=np.int64)
        flags = np.ones(2, dtype=np.int64)
        empty_positions = np.asarray([], dtype=np.int64)
        empty_reads = np.zeros((1, 0, 2), dtype=np.int64)
        empty_flags = np.asarray([], dtype=np.int64)
        filtered_positions = np.asarray([30], dtype=np.int64)
        filtered_reads = np.zeros((1, 1, 2), dtype=np.int64)
        filtered_flags = np.zeros(1, dtype=np.int64)
        genomic_data = (
            (positions, reads, flags),
            (empty_positions, empty_reads, empty_flags),
            (filtered_positions, filtered_reads, filtered_flags),
        )
        reversible_marker = object()

        reversible_pool = InlineBlockDiscoveryPool()
        reversible_results = generate_all_block_haplotypes(
            genomic_data,
            num_processes=1,
            total_numba_threads=1,
            block_pool=reversible_pool,
            discard_reads_after=False,
            reversible_cavity_config=reversible_marker,
        )
        assert len(reversible_pool.tasks) == 3
        assert all(
            task[4]["reversible_cavity_config"] is reversible_marker
            for task in reversible_pool.tasks
        )
        assert len(reversible_calls) == 1
        call = reversible_calls[0]
        assert call[0] is positions
        assert call[1] is reads
        assert call[2] is flags
        assert call[3] is reversible_marker
        assert materialization_classes == [BlockResult]
        assert len(legacy_calls) == 2
        assert reversible_results[0].dispatch_marker == "reversible"
        assert reversible_results[1].dispatch_marker == "legacy"
        assert reversible_results[2].dispatch_marker == "legacy"
        assert reversible_results[0].reads_count_matrix is reads

        legacy_pool = InlineBlockDiscoveryPool()
        legacy_results = generate_all_block_haplotypes(
            ((positions, reads, flags),),
            num_processes=1,
            total_numba_threads=1,
            block_pool=legacy_pool,
            discard_reads_after=False,
        )
        assert len(reversible_calls) == 1
        assert len(legacy_calls) == 3
        assert legacy_results[0].dispatch_marker == "legacy"
        assert legacy_results[0].reads_count_matrix is reads
    finally:
        globals()["generate_haplotypes_block_robust"] = original_legacy
        reversible_discovery.discover_block_reversible_cavity = (
            original_reversible
        )

    return {
        "legacy_dispatch_unchanged": "pass",
        "empty_and_no_kept_legacy_fallback": "pass",
        "reversible_forwarding_and_exact_materialization": "pass",
        "reversible_no_refit": "pass",
    }
