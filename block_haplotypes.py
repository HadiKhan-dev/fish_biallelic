"""block_haplotypes.py — Discrete-hap founder discovery (orchestrator)

The top-level orchestration layer of stage-3 block-haplotype founder
discovery.  This file contains:

  - Forkserver pool scaffolding (_ForkserverPool, _init_block_worker)
  - generate_haplotypes_block, generate_haplotypes_block_robust — the
    per-block public entry points (these call bhd_kgrowth for founder
    discovery, then post-process)
  - find_missing_haplotypes_iterative — iterative missing-hap recovery
  - Output construction (_compute_per_site_confidence,
    _discrete_haps_to_prob_arrays)
  - _final_cleanup, consolidate_similar_candidates — chimera pruning,
    Viterbi subset selection, and candidate consolidation on the
    prob-array form
  - _worker_generate_block_direct, generate_all_block_haplotypes —
    the multi-process orchestrator over all blocks of a contig
  - BlockResult, BlockResults — result containers

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
import math
import multiprocessing as mp
import multiprocessing.pool
import warnings
import gc
import ctypes

from numba import njit, prange

import thread_config
import dynamic_threads

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
# attribute lookup (e.g. PAIRWISE_RECOVERY_ENABLED) to
# preserve runtime-mutation semantics.
from bhd_kernels import MASK
from bhd_fit import _update_A
from bhd_recovery_select import _hamming_pct_kept
from bhd_config import (
    DEFAULT_LAMBDA,
    K_MEDOID_STARTS_DEFAULT,
    RECOVERY_HAPS_EQUAL_EPS_PCT,
    RECOVERY_MAX_K,
    RECOVERY_MIXTURE_K_MAX,
    RECOVERY_MIXTURE_PATIENCE,
)
from bhd_kgrowth import _grow_K_with_recovery

# BlockResult, BlockResults, and consolidate_similar_candidates were migrated
# out of the retired legacy block_haplotypes.py into this module — see the
# "MIGRATED FROM block_haplotypes.py" section at the end of the file.
# (find_missing_haplotypes_iterative is still imported lazily from the legacy
# module inside the residual loop, where present.)  Dynamic-thread rescaling
# uses the standalone dynamic_threads module (apply_dynamic_threads), wired in
# _init_block_worker.

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


def _init_block_worker(total_cores, active_counter, extra_counter=None):
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
        import os, numba
        os.environ['NUMBA_NUM_THREADS'] = str(total_cores)
        numba.config.NUMBA_NUM_THREADS = total_cores
        numba.set_num_threads(1)
    except Exception:
        pass
    # Wire the shared state so every phase boundary across this module +
    # bhd_recovery + bhd_trio re-checks the SAME pool-wide active count.
    dynamic_threads.set_dynamic_thread_state(total_cores, active_counter, extra_counter)









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
    # phase boundary via dynamic_threads.apply_dynamic_threads(), so a straggler
    # block grows into cores freed as its peers finish.
    dynamic_threads.increment_active()
    dynamic_threads.apply_dynamic_threads()

    try:
        # --- diagnostic: time each block's discovery and print id/size/seconds
        # as it finishes (lines tagged [block-time]; grep them out of the log).
        import time as _time
        import sys as _sys
        _bt0 = _time.perf_counter()
        result = generate_haplotypes_block_robust(
            positions, reads, keep_flags=flags, **kwargs)
        _bt = _time.perf_counter() - _bt0
        try:
            _nr = reads.shape[0] if hasattr(reads, "shape") else len(reads)
            _ns = len(positions) if positions is not None else -1
            print("[block-time] id=%d reads=%s sites=%s seconds=%.2f"
                  % (block_idx, _nr, _ns, _bt), file=_sys.stderr, flush=True)
        except Exception:
            pass
        _malloc_trim()
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
        # no idle cores as the active-block count changes (see dynamic_threads'
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