"""block_haplotypes_discrete.py — Discrete-hap founder discovery with
hard pair assignment and a wildcard founder.

Replaces the EM-based discovery in block_haplotypes_em_foothold.py with
a different inference paradigm built around eight philosophical
principles:

  1. Sub-additive scoring across sites within each sample (decisive
     fits beat blended fits).  Implemented via the wildcard penalty
     mechanism — sites where no real founder pair fits well get routed
     to the wildcard at a per-site cost λ, so a sample with patchy
     fit accumulates wildcard penalties at its bad sites rather than
     contaminating real founders' learning.

  2. Strict diploid constraint — every sample is exactly two founder
     copies.  Each can be a real founder or the wildcard W, but never
     a fractional combination.

  3. Wildcard founder — a degenerate "I match anything but pay a
     penalty" founder that absorbs samples (or sample-strands at
     specific sites) that don't fit the current real founder set.
     The penalty λ is calibrated so real founders are preferred when
     they fit reasonably, and wildcards take over when they don't.

  4. Discreteness throughout — founder haps are binary {0, 1} during
     inference.  Continuous-h relaxations introduce attractors
     (compromise founders, recombinants, fuzzy fixed points) that
     don't exist in the data-generating process.  Sites where the
     binary value can't be determined from the data are MASKED at
     output time, not represented as fuzzy.

  5. Distinguishability and parsimony — founders must be distinguishable
     from each other by the data.  K is grown one founder at a time
     while wildcard mass decreases meaningfully; growth stops when
     adding a founder doesn't reduce wildcard mass.

  6. Inference model = data-generating model — the inference mirrors
     the generative process (discrete haps, hard pair assignment,
     diploid constraint), even when optimization is harder than the
     EM relaxation.

  7. Honest uncertainty — per-site confidence is tracked (fraction of
     attributing samples that agree with the inferred allele).  Blocks
     with high residual wildcard mass are flagged as uncertain rather
     than collapsed to confident-but-wrong founder sets.

  8. Real-founder-hypothesis initialization — founders are initialized
     from specific samples' implied haps (most-decisive samples'
     genotype dosage / 2), never from soft averages.  K-growth seeds
     new founders from the worst-fit sample, breaking out of local
     optima.

The math:

  For sample s, site l (kept), under pair A[s] = (a, b):

    Both real (a, b ∈ {0..K-1}):
      dosage d = H[a, l] + H[b, l] ∈ {0, 1, 2}
      cost(s, l) = -log probs[s, l, d]

    One wildcard (a ∈ {0..K-1}, b = W):
      cost(s, l) = min_{w ∈ {0,1}} -log probs[s, l, H[a, l] + w] + λ

    Both wildcard (a = b = W):
      cost(s, l) = min_{d ∈ {0,1,2}} -log probs[s, l, d] + 2λ
      (each strand picks its own allele independently per site;
      at dosage=1 the (0,1) vs (1,0) ambiguity has no effect on cost)

  Total NLL = Σ_s Σ_{l: kept} cost(s, l)

  Coordinate descent:
    repeat:
      update A given H  (assign each sample to its lowest-cost pair)
      update H given A  (for each (founder, site), pick binary value
                          that minimises NLL contribution from samples
                          carrying that founder)
    until no changes.

  K-growth:
    1. Start with K=1, init from highest-decisiveness sample.
    2. Run coord descent, measure wildcard mass.
    3. While wildcard mass > threshold and K < K_max:
         a. Try adding a founder seeded from the worst-fit sample.
         b. Re-run coord descent.
         c. Accept new K iff wildcard mass decreased meaningfully.
         d. Otherwise stop.

The module is a drop-in replacement for the legacy block_haplotypes
module: same `generate_all_block_haplotypes` orchestrator signature,
same BlockResult return type (with extra attributes attached for the
new per-site confidence and uncertainty flag).  Re-exports the legacy
final-cleanup helpers (consolidate_similar_candidates,
select_optimal_haplotype_set_viterbi, prune_chimeras) since these are
not EM-specific and provide useful safety nets.
"""

import numpy as np
import math
import multiprocessing as mp
import multiprocessing.pool
import warnings
import gc
import ctypes

import thread_config

import analysis_utils
import hap_statistics

# Re-export the public names from block_haplotypes that this module
# uses internally (consolidate_similar_candidates, BlockResult,
# BlockResults) and that legacy commented-out code blocks reference
# (select_optimal_haplotype_set_viterbi, prune_chimeras — see the
# "Original code (preserved for record)" blocks in _final_cleanup).
from block_haplotypes import (
    BlockResult,
    BlockResults,
    consolidate_similar_candidates,
    select_optimal_haplotype_set_viterbi,
    prune_chimeras,
)

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
# CONSTANTS — SUBTRACTION-BASED RECOVERY (mixture + BIC + outer iteration)
# =============================================================================
#
# These constants govern the subtraction-recovery pass that runs after
# K-growth.  Recovery generates clean residual candidates by subtracting
# each current founder from each sample's argmax dosage, fits a Bernoulli
# mixture model with K selected by BIC over candidate-density, then runs
# outer BIC subset-selection on (existing founders ∪ mixture consensus)
# against actual sample data.  The outer pass is iterated with K-growth
# so that worst-fit-sample seeding (K-growth's mechanism) and density-
# based seeding (mixture's mechanism) catch different failure modes.

# Cleanness threshold for residuals to be admitted as candidates.
# A residual is "clean" if at least this fraction of sites have values
# in {0, 1} after subtraction (i.e., the founder hypothesis is consistent
# with the sample's argmax dosage).  Lowering admits more candidates
# but with more noise; raising rejects valid candidates whose other
# strand is genuinely there but with read-error sites.
RECOVERY_CLEANNESS_THRESHOLD = 0.90

# Bernoulli mixture parameters for inner K-selection on candidates
RECOVERY_MIXTURE_K_MAX = 10            # try K=1..K_max, pick best by inner BIC
RECOVERY_MIXTURE_N_RESTARTS = 3        # EM restarts per K (different K-means++ seeds)
RECOVERY_MIXTURE_MAX_ITER = 100        # max EM iterations per fit
RECOVERY_MIXTURE_TOL = 1e-6            # relative LL change for EM convergence
RECOVERY_MIXTURE_THETA_EPS = 1e-3      # clip theta to [eps, 1-eps] for log stability
RECOVERY_MIXTURE_RNG_SEED = 42         # base RNG seed (varied per round)

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
RECOVERY_MAX_ROUNDS = 10

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

# =============================================================================
# CONSTANTS — LATE LOW-CARRIER RESCUE (post-convergence targeted refinement)
# =============================================================================
#
# Background: at blocks where one truth founder is carried by very few
# samples (say, 4 out of N=320 = 0.6% of strands), trio recovery may
# produce a candidate close to that truth (e.g., 2.5% Hamming) that is
# BIC-trimmed in correctly, but joint CD on H_trio_seed drifts the
# candidate slightly further from truth (e.g., 3% Hamming).  The drifted
# hap then captures the same 4 carriers as truth would have, fitting
# them at the noise floor — so NLL is identical to truth's NLL.  At
# the same K, NLL is identical (degenerate plateau).  The mixture in
# subtraction recovery clusters at K=1..mixture_K_max and uses inner
# BIC over candidate density to pick K; sparse low-frequency-founder
# residuals (here, 4 candidates out of ~1668) are absorbed into a
# larger mixture component, so the mixture's consensus haps do NOT
# include a near-truth alternative that would let forward selection
# pick the K-optimal subset.  The pipeline ends at K=K_truth+1 with
# a chimera in place of the low-frequency truth, losing by exactly cc
# on BIC.  Diagnosed at chr6:23624234; see diagnose_chr6_23624234.py.
#
# Late rescue mechanism: after _grow_K_with_recovery's outer iteration
# converges, identify selected haps with very low carrier counts.  For
# each such hap h_low, generate per-strand residuals from h_low's
# carrier samples (residual = sample.argmax_dosage - other_strand_hap).
# By construction, these residuals are clean approximations of the
# "true" version of h_low — for chr6:23624234, the 4 carriers of
# alg_row_5 are EXACTLY the 4 truth_4-carrying samples, and the
# residuals after subtracting the truth-matching other strand yield
# 4 candidates near truth_4.  Add these to the candidate pool, run
# greedy BIC forward selection, and accept iff BIC strictly improves.
# Forward selection is K-aware (BIC stops naturally at the optimal K),
# so it can REDUCE K by 1 in the chimera-replacement case.
#
# Cost: triggered only on blocks with at least one low-carrier hap
# (typically <5% of blocks).  Per triggered block: residual generation
# is O(N*L), pool grows by 4-20 candidates, forward selection +
# _fit_at_fixed_K cost ~1 sec.  Negligible aggregate cost (<1% of
# stage 3 runtime).  Untriggered blocks have zero overhead.

# Trigger threshold (fraction of 2N).  A hap is "low-carrier" if its
# real-strand usage count is below this fraction of total real strands
# (= 2*N = total real-strand slots across N samples).  At N=320:
# 0.02 * 640 = 12.8 → minimum carriers must be < 13 to trigger.
# At chr6:23624234, alg_row_5 has 4 carriers (0.6%) → triggers.
# At non-pathological blocks, all founders typically carry >5% of
# strands → no trigger, zero overhead.
RECOVERY_LOW_CARRIER_TRIGGER_FRAC = 0.02


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
# Fix: deterministic multi-start.  At the K=0 -> K=1 transition, pick
# K_MEDOID_STARTS samples spread across seed-hap-space (via PAM
# k-medoids on the pairwise Hamming distance matrix of sample seeds)
# and run full K-growth from each as a separate H_init.  Pick the
# trajectory with lowest final BIC = K_final * cc + 2 * NLL_final.
# BIC (not raw NLL) is the right cross-K comparison criterion: it
# penalises trajectories that grew to a larger K than the data
# justifies, so multi-start naturally prefers parsimonious solutions
# of equal data-fit quality.
#
# Why k-medoids and not top-K-decisive?  Decisiveness doesn't
# correlate with basin membership — confirmed empirically: chr17 needed
# K=20 top-decisive samples to find a truth-basin medoid, but K=5
# k-medoids found 4 of them.  K-medoids picks DIVERSE seeds in
# hap-space, which at small K guarantees representation from each
# truth-progenitor cluster.
#
# Cost: K_MEDOID_STARTS x full K-growth instead of 1x.  At default
# K=5 medoids, stage 3 cost is roughly 5x the single-trajectory cost.
# Per-block parallelism unchanged.

# Number of medoid starts at the initial K=0 -> K=1 transition.
# Reducing to 1 disables multi-start (recovers legacy single-start
# behavior).  Increasing improves robustness on blocks with very
# heterogeneous truth founders but costs proportionally more time.
K_MEDOID_STARTS_DEFAULT = 5

# Minimum sample count for medoid multi-start to be applied.  If
# N < K_MEDOID_STARTS, we fall back to single-start because PAM
# can't pick more medoids than there are points.
MEDOID_MIN_N_FOR_MULTISTART = 3

# Maximum PAM swap-phase iterations.  PAM converges quickly on
# small-K problems; 100 is a defensive cap rarely reached.
MEDOID_PAM_MAX_ITER = 100


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


def _init_block_worker(total_cores, active_counter):
    """Initializer for worker processes — sets up dynamic numba thread
    allocation based on number of currently-active workers."""
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
# K-GROWTH ORCHESTRATION
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
# SUBTRACTION-BASED RECOVERY: BERNOULLI MIXTURE HELPERS
# =============================================================================
#
# These helpers fit a Bernoulli mixture model to a candidate pool of clean
# residuals (one per "carrier" sample after subtracting an existing founder).
# Each candidate is a binary vector of length L_kept; we model them as a
# mixture of K hidden founder profiles theta_k in [0,1]^L, each candidate
# generated by one component:
#
#     P(c | k) = prod_l theta_k[l]^c[l] * (1 - theta_k[l])^(1-c[l])
#
# We fit by EM with K-means++ initialisation and multiple restarts, then
# pick the K that minimises BIC over candidate density (inner BIC).  The
# fitted theta vectors (rounded to {0, 1}) become consensus candidate haps
# for outer BIC subset-selection on actual sample data.

def _logsumexp(x, axis):
    """Numerically stable log-sum-exp along an axis.  Returns array with
    that axis squeezed.  Handles -inf max (all entries -inf) by zeroing
    the offset (so the result is -inf, not NaN)."""
    m = x.max(axis=axis, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    return (m_safe + np.log(np.exp(x - m_safe).sum(axis=axis, keepdims=True))).squeeze(axis)


def _kmeans_pp_init(candidates, K, rng):
    """K-means++ initialisation for binary data using Hamming distance.

    Picks K centers from candidates: the first uniformly at random, each
    subsequent weighted by squared min-Hamming-distance from existing
    centers.  This gives spread-out initial centers that typically lead
    EM to good local optima — much more robust than uniform random init.

    Args:
      candidates: (N, L) binary array
      K: number of centers to pick
      rng: numpy Generator

    Returns: (K, L) array of centers selected from candidates (copy).
    """
    N, L = candidates.shape
    if K >= N:
        # Defensive: caller should never let this happen, but if it does,
        # return all candidates plus repeats so the EM has something to work with.
        idx = list(range(N)) + [int(rng.integers(N)) for _ in range(K - N)]
        return candidates[idx].copy()

    centers_idx = [int(rng.integers(N))]
    for _ in range(K - 1):
        existing = candidates[centers_idx]                                # (n_existing, L)
        # Pairwise Hamming distance: (N, n_existing) — number of differing bits
        diffs = (candidates[:, None, :] != existing[None, :, :]).sum(axis=2)
        min_dists = diffs.min(axis=1).astype(np.float64)
        if min_dists.sum() == 0:
            # All candidates identical to some existing center; pick anything not yet picked
            remaining = [i for i in range(N) if i not in centers_idx]
            if remaining:
                new_c = int(rng.choice(remaining))
            else:
                new_c = int(rng.integers(N))
        else:
            probs = min_dists ** 2
            probs = probs / probs.sum()
            new_c = int(rng.choice(N, p=probs))
        centers_idx.append(new_c)
    return candidates[centers_idx].copy()


def _bernoulli_mixture_em(cands, K, init_centers,
                            max_iter=RECOVERY_MIXTURE_MAX_ITER,
                            tol=RECOVERY_MIXTURE_TOL,
                            eps=RECOVERY_MIXTURE_THETA_EPS):
    """Single EM run for a Bernoulli mixture with K components.

    Args:
      cands: (N, L) float64 array (binary 0/1 values, but float for arithmetic)
      K: number of components
      init_centers: (K, L) initial theta values (binary will be smoothed)
      max_iter: cap on EM iterations
      tol: relative LL change threshold for convergence
      eps: theta clipping bound for numerical stability

    Returns:
      theta:  (K, L) final mixture parameters
      pi:     (K,)   final mixture weights
      ll:     final log-likelihood (scalar)
      n_iter: number of iterations actually used
      resp:   (N, K) responsibilities (soft assignments)

    Math:
      E-step: log P(c | k) = c . log(theta_k) + (1-c) . log(1-theta_k)
              gamma_nk = pi_k * P(c_n | k) / sum_j pi_j * P(c_n | j)
      M-step: theta_k[l] = sum_n gamma_nk * c_n[l] / sum_n gamma_nk
              pi_k = sum_n gamma_nk / N
    """
    N, L = cands.shape

    # Smooth initial centers from {0, 1} to (eps, 1-eps) for numerical stability
    theta = init_centers.astype(np.float64) * (1 - 2 * eps) + eps
    pi = np.ones(K, dtype=np.float64) / K

    prev_ll = -np.inf
    n_iter = 0
    ll = 0.0    # initialised in case max_iter=0 (defensive)

    log_resp = None
    resp = None

    for it in range(max_iter):
        n_iter = it + 1
        # E-step
        log_theta = np.log(theta)
        log_one_minus_theta = np.log(1 - theta)
        log_p = cands @ log_theta.T + (1 - cands) @ log_one_minus_theta.T   # (N, K)
        log_pi = np.log(pi + 1e-15)
        log_p_weighted = log_p + log_pi[None, :]                              # (N, K)

        log_norm = _logsumexp(log_p_weighted, axis=1)                          # (N,)
        log_resp = log_p_weighted - log_norm[:, None]                          # (N, K)
        resp = np.exp(log_resp)

        ll = float(log_norm.sum())

        # Convergence check (relative).  Break BEFORE updating prev_ll so
        # the caller receives the most recent ll value.
        if prev_ll != -np.inf and abs(ll - prev_ll) < tol * max(abs(ll), 1.0):
            break
        prev_ll = ll

        # M-step
        N_k = resp.sum(axis=0)                                                 # (K,)
        pi = N_k / N
        N_k_safe = np.maximum(N_k, 1e-10)                                      # avoid /0 for empty components
        theta = (resp.T @ cands) / N_k_safe[:, None]
        theta = np.clip(theta, eps, 1 - eps)

    return theta, pi, ll, n_iter, resp


def _fit_bernoulli_mixture_select_K(candidates,
                                      K_max=RECOVERY_MIXTURE_K_MAX,
                                      n_restarts=RECOVERY_MIXTURE_N_RESTARTS,
                                      seed=RECOVERY_MIXTURE_RNG_SEED,
                                      verbose=False):
    """Fit Bernoulli mixture for K=1..K_max, pick the K minimising BIC.

    Args:
      candidates: list of (L,) binary arrays
      K_max: upper bound on K to try
      n_restarts: EM restarts per K (with different K-means++ seeds)
      seed: base RNG seed (for reproducibility)
      verbose: print BIC trace

    Returns:
      list of (L,) binary arrays — the consensus haps for the selected K.
      Empty list if candidates is empty.

    BIC formula:
      BIC = -2 * LL + (K*L + K - 1) * log(N)
        - K*L params for the mixture component profiles
        - K-1 params for the mixture weights (one constraint pi.sum() == 1)
      Lower BIC = better.

    The output is INTENTIONALLY over-permissive — inner BIC measures
    density of candidates in candidate-space (which can include noise
    components from recombinant or low-cleanness candidates).  The outer
    BIC subset-selection on actual sample data (in the recovery round
    loop) filters these out.
    """
    if len(candidates) == 0:
        return []

    cands_arr = np.stack(candidates, axis=0).astype(np.float64)               # (N, L)
    N, L = cands_arr.shape

    # Cap K_max at N (can't have more components than candidates)
    K_max_effective = min(K_max, N)

    rng = np.random.default_rng(seed)

    best_overall = None   # tuple: (K, BIC, theta, pi, ll, effective_sizes)
    bic_trace = []

    if verbose:
        print(f'    Inner mixture fitting: N={N} candidates, L={L}, '
              f'trying K=1..{K_max_effective}, n_restarts={n_restarts}')

    for K in range(1, K_max_effective + 1):
        # Multi-restart: pick the best LL across n_restarts independent
        # K-means++ inits.  EM has local minima; multi-start gives robustness.
        best_for_K = None   # (LL, theta, pi, resp)
        for restart in range(n_restarts):
            init_centers = _kmeans_pp_init(cands_arr, K, rng)
            theta, pi, ll, _n_iter, resp = _bernoulli_mixture_em(
                cands_arr, K, init_centers=init_centers)
            if best_for_K is None or ll > best_for_K[0]:
                best_for_K = (ll, theta, pi, resp)

        ll, theta, pi, resp = best_for_K
        n_params = K * L + (K - 1)
        bic = -2 * ll + n_params * np.log(max(N, 2))

        effective_sizes = resp.sum(axis=0)
        bic_trace.append((K, bic, ll, effective_sizes))

        if best_overall is None or bic < best_overall[1]:
            best_overall = (K, bic, theta, pi, ll, effective_sizes)

    if verbose:
        for K, bic, ll, eff_sizes in bic_trace:
            marker = ' <-' if K == best_overall[0] else ''
            eff_str = '[' + ', '.join(f'{s:.1f}' for s in eff_sizes) + ']'
            print(f'      K={K:>2d}: LL={ll:>11.1f}, BIC={bic:>11.1f}, '
                  f'eff_sizes={eff_str}{marker}')

    best_K, best_bic, best_theta, best_pi, best_ll, best_eff = best_overall
    if verbose:
        print(f'    Inner mixture: selected K={best_K} with BIC={best_bic:.1f}')

    # Round theta to binary haps (consensus founder profiles)
    binary_thetas = (best_theta > 0.5).astype(np.int64)
    return [binary_thetas[k].copy() for k in range(best_K)]


# =============================================================================
# SUBTRACTION-BASED RECOVERY: CANDIDATE GENERATION & OUTER BIC SELECTION
# =============================================================================

def _run_subtraction_round(pool, argmax_dosage_kept,
                            cleanness_threshold=RECOVERY_CLEANNESS_THRESHOLD):
    """Generate clean residual candidates by subtracting each pool member
    from each sample's argmax dosage.

    For sample s with argmax dosage d_s and pool member h, the residual
    r_s = d_s - h is interpreted as an estimate of the OTHER strand
    given that h was one strand.  At sites where d_s in {h, h+1}, r_s in
    {0, 1} — admissible.  At sites where d_s != h and d_s != h+1, r_s is
    out of range — the (h, ?) hypothesis is inconsistent at that site.

    A residual is "clean" if at least cleanness_threshold of its sites
    have admissible values.  Clean residuals are clipped to {0, 1} and
    returned as binary candidate haps.

    Args:
      pool: list of (L_kept,) binary arrays (the founders to subtract)
      argmax_dosage_kept: (N, L_kept) int array of argmax genotype dosages
        in {0, 1, 2} per (sample, kept site)
      cleanness_threshold: min fraction of admissible sites to accept

    Returns:
      list of (L_kept,) binary candidate arrays
    """
    raw_candidates = []
    for hap in pool:
        residual = argmax_dosage_kept - hap[None, :]              # (N, L_kept)
        in_01 = (residual >= 0) & (residual <= 1)
        cleanness = in_01.mean(axis=1)                            # (N,)
        clean_mask = cleanness >= cleanness_threshold
        if not clean_mask.any():
            continue
        clean_residuals = residual[clean_mask]                    # (n_clean, L_kept)
        clipped = np.clip(clean_residuals, 0, 1).astype(np.int64)
        for cand in clipped:
            raw_candidates.append(cand)
    return raw_candidates


def _generate_carrier_residuals(probs_k, H, A, low_idx_list,
                                 cleanness_threshold=RECOVERY_CLEANNESS_THRESHOLD,
                                 verbose=False):
    """Generate per-(carrier_sample, partner_candidate) residuals for
    low-carrier haps.

    This is the targeted analogue of _run_subtraction_round, used by
    the late low-carrier rescue (see RECOVERY_LOW_CARRIER_TRIGGER_FRAC).

    For each carrier strand of a low-carrier hap h_low, we want to
    recover h_low's "right" version (the missing truth founder).  The
    algebra: if carrier sample s has true strands (truth_X, truth_Y)
    where truth_Y is the missing one we want to recover, then
      argmax_dosage[s] = truth_X + truth_Y       (noiseless data)
      residual = argmax_dosage[s] - H[partner]
              = (truth_X + truth_Y) - H[partner]
    The residual is "clean" (every site in {0, 1}) if and only if
    H[partner] = truth_X (the actual other strand), in which case
    residual = truth_Y exactly.

    The challenge: when h_low is a chimera, _update_A's choice of A[s,
    other_slot] is whichever founder makes (h_low, partner) optimally
    fit the dosage given h_low is in the pair — NOT necessarily the
    actual other strand truth_X.  At chr6:23624234, A pairs h_low
    (= chim_5) with H[1] = truth_5 for some carriers, but the actual
    other strands are different truth founders; the residual ends up
    = chim_5 by construction.  Verified empirically: all 4 carrier
    residuals at H_low = 0.00%.

    Solution: for each carrier strand, try every H row (excluding
    low_idx_set) as the candidate subtractor.  Most produce noisy
    out-of-range residuals (cleanness < threshold) and are rejected.
    Exactly one row matches truth_X for that carrier and produces a
    clean residual = truth_Y at 100% cleanness.

    Strands paired with the wildcard slot — when the wildcard is the
    "other_slot" — are still mined: we just iterate over all H rows
    as subtractors regardless of A's wildcard assignment.  Subtracting
    a low-carrier hap (a known suspect) from itself or another low-
    carrier hap gives at best noisy residuals, so subtractors in
    low_idx_set are skipped.

    Args:
      probs_k: (N, L_kept, 3) — kept-site posteriors
      H: (K, L_kept) discrete {0, 1} — current founder bits
      A: (N, 2) — current pair assignments (entry K = wildcard sentinel)
      low_idx_list: list of founder indices whose carriers we mine
      cleanness_threshold: min fraction of admissible sites per residual
      verbose: if True, additionally return per-(sample, slot, subtractor)
        provenance

    Returns:
      If verbose=False: list of (L_kept,) binary candidate arrays — one
        per (sample, strand, subtractor) triple that survived cleanness
        filtering.
      If verbose=True: tuple (residuals, provenance) where provenance is
        a list of dicts, one per (sample, slot, subtractor) triple
        examined, with keys:
          sample_idx, slot, low_idx, partner_idx, partner_kind, cleanness,
          accepted, residual (the clipped binary array if accepted else None)
        partner_kind ∈ {'self_low', 'low_carrier', 'normal'}.
    """
    N, L_kept = probs_k.shape[0], probs_k.shape[1]
    K = H.shape[0]
    if K == 0 or not low_idx_list:
        return ([], []) if verbose else []
    argmax_dosage = probs_k.argmax(axis=2)                        # (N, L_kept)

    low_idx_set = set(int(k) for k in low_idx_list)
    residuals = []
    provenance = [] if verbose else None
    for s in range(N):
        for slot in range(2):
            if int(A[s, slot]) not in low_idx_set:
                continue
            low_idx = int(A[s, slot])
            # Iterate over every H row as a candidate subtractor.
            # Cleanness filter discriminates: wrong-partner subtractors
            # produce out-of-range bits at heterozygous sites and fail.
            for partner_idx in range(K):
                if partner_idx == low_idx:
                    if verbose:
                        provenance.append({
                            'sample_idx': s, 'slot': slot,
                            'low_idx': low_idx, 'partner_idx': partner_idx,
                            'partner_kind': 'self_low',
                            'cleanness': float('nan'), 'accepted': False,
                            'residual': None})
                    continue
                if partner_idx in low_idx_set:
                    # Subtracting another suspect hap would taint
                    # the residual with that suspect's drift.
                    if verbose:
                        provenance.append({
                            'sample_idx': s, 'slot': slot,
                            'low_idx': low_idx, 'partner_idx': partner_idx,
                            'partner_kind': 'low_carrier',
                            'cleanness': float('nan'), 'accepted': False,
                            'residual': None})
                    continue
                residual = argmax_dosage[s] - H[partner_idx]      # (L_kept,)
                in_01 = (residual >= 0) & (residual <= 1)
                cleanness = float(in_01.mean())
                accepted = cleanness >= cleanness_threshold
                clipped = (np.clip(residual, 0, 1).astype(np.int64)
                           if accepted else None)
                if accepted:
                    residuals.append(clipped)
                if verbose:
                    provenance.append({
                        'sample_idx': s, 'slot': slot,
                        'low_idx': low_idx, 'partner_idx': partner_idx,
                        'partner_kind': 'normal',
                        'cleanness': cleanness, 'accepted': accepted,
                        'residual': clipped})
    if verbose:
        return residuals, provenance
    return residuals


def _compute_nll_for_subset(haps_list, probs_k, lam):
    """Score a subset of haps by computing UNCAPPED NLL on sample data.
    Haps are FIXED (not refined) during scoring — used for outer BIC
    subset-selection where we want to compare different SUBSETS of a
    fixed candidate pool.

    For empty subset (K=0): NLL = sum of cost_WW per sample.
    """
    if len(haps_list) == 0:
        cost_WW_per_site = _per_site_cost_W_W(probs_k, lam)
        return float(cost_WW_per_site.sum())
    H = np.stack(haps_list, axis=0)
    A, per_sample_cost, per_sample_cost_unc, wildcard_slots = _update_A(probs_k, H, lam)
    return float(per_sample_cost_unc.sum())


def _greedy_bic_select(candidate_haps, probs_k, lam,
                        cc_scale=RECOVERY_OUTER_CC_SCALE,
                        max_k=RECOVERY_MAX_K,
                        use_log_bic=False,
                        verbose=False):
    """Greedy forward selection by BIC over a fixed candidate pool.

    Haps are FIXED during scoring (no coord descent).  At each step,
    pick the candidate giving the lowest NLL when added; accept iff
    BIC improves (equivalent to NLL_improvement > cc/2).

    Args:
      candidate_haps: list of (L_kept,) binary arrays — the pool
      probs_k, lam: scoring primitives
      cc_scale: BIC complexity-cost scale (outer; default 0.5)
      max_k: hard cap on selected size
      use_log_bic: if True, use log-BIC formula for cc (cc_scale *
        log(N*L) * snp_growth); if False (default, project standard),
        use linear formula (cc_scale * snp_growth * N).  Must match
        the use_log_bic of the surrounding K-growth so accept/reject
        criteria are consistent across the pipeline.
      verbose: print accept/reject decisions

    Returns:
      selected_indices: list of indices into candidate_haps
      selected_haps:    list of arrays (same as candidate_haps[i] for i in indices)
      current_nll:      NLL at final selection
    """
    N = probs_k.shape[0]
    L_kept = probs_k.shape[1]
    cc = _compute_cc(cc_scale, N, L_kept, use_log_bic=use_log_bic)

    nll_K0 = _compute_nll_for_subset([], probs_k, lam)
    bic_K0 = 0 * cc + 2 * nll_K0

    selected_indices = []
    selected_haps = []
    used = set()
    current_bic = bic_K0
    current_nll = nll_K0

    if verbose:
        print(f'    Forward: K=0 NLL={nll_K0:.1f}, BIC={bic_K0:.1f}, '
              f'cc={cc:.1f}, threshold cc/2={cc/2:.1f}')

    while len(selected_haps) < min(len(candidate_haps), max_k):
        best_ci = -1
        best_nll = float('inf')
        for ci in range(len(candidate_haps)):
            if ci in used:
                continue
            trial_haps = selected_haps + [candidate_haps[ci]]
            trial_nll = _compute_nll_for_subset(trial_haps, probs_k, lam)
            if trial_nll < best_nll:
                best_nll = trial_nll
                best_ci = ci

        if best_ci < 0:
            break

        k_new = len(selected_haps) + 1
        bic_new = k_new * cc + 2 * best_nll
        d_nll = current_nll - best_nll

        if bic_new < current_bic:
            selected_indices.append(best_ci)
            selected_haps.append(candidate_haps[best_ci])
            used.add(best_ci)
            if verbose:
                print(f'    Forward: K={k_new} ACCEPT cand[{best_ci}], '
                      f'NLL={best_nll:.1f}, BIC={bic_new:.1f}, dNLL={d_nll:.1f}')
            current_bic = bic_new
            current_nll = best_nll
        else:
            if verbose:
                print(f'    Forward: K={k_new} REJECT cand[{best_ci}], '
                      f'NLL={best_nll:.1f}, BIC={bic_new:.1f}, dNLL={d_nll:.1f} '
                      f'< cc/2={cc/2:.1f}')
            break

    return selected_indices, selected_haps, current_nll


def _swap_refine(selected_indices, selected_haps, pool_haps,
                  probs_k, lam, current_nll,
                  nll_tolerance=RECOVERY_SWAP_NLL_TOLERANCE,
                  max_passes=10, verbose=False):
    """Try swapping each selected hap with each unselected pool member.
    Apply swap if NLL improves by more than nll_tolerance.  Iterate over
    passes until no improvement.

    Sometimes greedy forward selection picks a near-optimal hap early
    that becomes redundant after later picks; swap lets us replace it
    with a better one without requiring a full re-search.
    """
    sel_ind = list(selected_indices)
    sel_haps = list(selected_haps)
    K = len(sel_haps)
    if K == 0:
        return sel_ind, sel_haps, current_nll, 0

    n_swaps = 0
    for pass_num in range(max_passes):
        improved_in_pass = False
        for si in range(K):
            best_ci = -1
            best_nll = current_nll - nll_tolerance
            for ci in range(len(pool_haps)):
                if ci in sel_ind:
                    continue
                trial_haps = list(sel_haps)
                trial_haps[si] = pool_haps[ci]
                trial_nll = _compute_nll_for_subset(trial_haps, probs_k, lam)
                if trial_nll < best_nll:
                    best_nll = trial_nll
                    best_ci = ci
            if best_ci >= 0:
                if verbose:
                    print(f'    Swap: pos {si} (cand[{sel_ind[si]}]) -> cand[{best_ci}], '
                          f'NLL {current_nll:.1f} -> {best_nll:.1f}')
                sel_haps[si] = pool_haps[best_ci]
                sel_ind[si] = best_ci
                current_nll = best_nll
                improved_in_pass = True
                n_swaps += 1
                break
        if not improved_in_pass:
            break

    return sel_ind, sel_haps, current_nll, n_swaps


def _bic_prune(selected_indices, selected_haps, probs_k, lam,
                cc_scale=RECOVERY_OUTER_CC_SCALE, use_log_bic=False,
                verbose=False):
    """BIC pruning: try dropping each selected hap.  Drop if the NLL
    increase from removal is less than cc/2 (i.e., the hap isn't pulling
    enough weight to justify the +cc penalty for keeping it).

    Iterates: each drop may enable another (cascading prune of redundant
    haps that propped each other up).  Matches the project's
    refine_selection_by_pruning pattern in beam_search_core.

    use_log_bic: bool — if True, use log-BIC formula for cc; if False
      (default, project standard), use linear formula.  Must match the
      use_log_bic of the surrounding K-growth so the prune threshold
      cc/2 is consistent with K-growth's growth threshold.

    Returns: (pruned_indices, pruned_haps, final_nll, n_dropped)
    """
    N = probs_k.shape[0]
    L_kept = probs_k.shape[1]
    cc = _compute_cc(cc_scale, N, L_kept, use_log_bic=use_log_bic)

    sel_ind = list(selected_indices)
    sel_haps = list(selected_haps)
    n_dropped = 0

    while len(sel_haps) > 0:
        nll_full = _compute_nll_for_subset(sel_haps, probs_k, lam)
        K = len(sel_haps)

        best_drop_idx = -1
        best_dnll = cc / 2   # threshold; only drop if dnll_increase < this

        for i in range(K):
            trial = sel_haps[:i] + sel_haps[i+1:]
            nll_trial = _compute_nll_for_subset(trial, probs_k, lam)
            dnll = nll_trial - nll_full   # NLL increase from dropping
            if dnll < best_dnll:
                best_dnll = dnll
                best_drop_idx = i

        if best_drop_idx < 0:
            break

        if verbose:
            print(f'    Prune: drop pos {best_drop_idx} (cand[{sel_ind[best_drop_idx]}]), '
                  f'NLL increase {best_dnll:.1f} < cc/2={cc/2:.1f} -- DROPPED')
        del sel_ind[best_drop_idx]
        del sel_haps[best_drop_idx]
        n_dropped += 1

    final_nll = _compute_nll_for_subset(sel_haps, probs_k, lam)
    return sel_ind, sel_haps, final_nll, n_dropped


def _hamming_pct_kept(a, b):
    """Per-site Hamming distance as a percentage (0-100)."""
    return float(np.mean(a != b)) * 100.0


def _haps_equal(haps_a, haps_b, eps_pct=RECOVERY_HAPS_EQUAL_EPS_PCT):
    """Test if two hap collections are equal (within eps_pct Hamming
    tolerance per pair, with bipartite matching).

    Returns True iff:
      - len(haps_a) == len(haps_b), AND
      - there's a 1-to-1 matching where each matched pair is within eps_pct.

    Used for convergence detection between recovery rounds and between
    outer iterations.  Tolerance accommodates near-identical haps that
    differ only at a few uncertain sites.
    """
    if len(haps_a) != len(haps_b):
        return False
    matched_b = [False] * len(haps_b)
    for ha in haps_a:
        found = False
        for bi, hb in enumerate(haps_b):
            if matched_b[bi]:
                continue
            if _hamming_pct_kept(ha, hb) < eps_pct:
                matched_b[bi] = True
                found = True
                break
        if not found:
            return False
    return all(matched_b)


# =============================================================================
# SUBTRACTION-BASED RECOVERY: ROUND LOOP
# =============================================================================

def _subtraction_recovery_round_loop(probs_k, H_init, lam,
                                       outer_cc_scale=RECOVERY_OUTER_CC_SCALE,
                                       max_K=RECOVERY_MAX_K,
                                       max_rounds=RECOVERY_MAX_ROUNDS,
                                       max_iter_per_K=50,
                                       intra_round_dedup_pct=RECOVERY_INTRA_ROUND_DEDUP_PCT,
                                       mixture_K_max=RECOVERY_MIXTURE_K_MAX,
                                       mixture_n_restarts=RECOVERY_MIXTURE_N_RESTARTS,
                                       mixture_seed_base=RECOVERY_MIXTURE_RNG_SEED,
                                       cleanness_threshold=RECOVERY_CLEANNESS_THRESHOLD,
                                       swap_nll_tolerance=RECOVERY_SWAP_NLL_TOLERANCE,
                                       haps_equal_eps_pct=RECOVERY_HAPS_EQUAL_EPS_PCT,
                                       use_log_bic=False,
                                       verbose=False):
    """Iterative subtraction-recovery rounds until convergence.

    Each round:
      1. Subtract each hap in current selected from each sample's argmax
         dosage; collect clean clipped residuals as raw candidates.
      2. Fit Bernoulli mixture for K=1..mixture_K_max with K-means++
         init + multi-restart EM.  Pick K minimising inner BIC.  Output:
         K consensus haps (binary).
      3. Intra-round dedup (safety net at intra_round_dedup_pct: tight
         duplicates that survived numerical rounding get merged).
      4. Pool = current selected union new consensus haps.
      5. Greedy BIC forward-select on sample data with FIXED haps
         (outer BIC, cc_scale=outer_cc_scale).
      6. Swap refinement.
      7. BIC pruning.
      8. Coord descent on the selected haps via _fit_at_fixed_K (this
         is where haps actually move toward truth).
      9. Convergence check: if selected unchanged from previous round
         (within haps_equal_eps_pct), exit.

    Args:
      probs_k: (N, L_kept, 3) — kept-site posteriors
      H_init: (K_init, L_kept) starting hap set (typically from K-growth)
      lam: wildcard penalty
      outer_cc_scale: BIC scale for outer subset selection on sample data
      max_K: hard cap on selected size
      max_rounds: hard cap on round iterations (defensive)
      max_iter_per_K: cap on coord-descent iterations within each round
      intra_round_dedup_pct: tight dedup threshold for consensus haps
      mixture_K_max, mixture_n_restarts, mixture_seed_base: inner mixture params
      cleanness_threshold: min residual cleanness to admit a candidate
      swap_nll_tolerance: NLL improvement floor for accepting a swap
      haps_equal_eps_pct: tolerance for round-convergence detection
      verbose: print per-round trace

    Returns:
      H_final: (K_final, L_kept) — refined hap set after recovery
    """
    if H_init is None or len(H_init) == 0:
        # Nothing to subtract from; recovery has no anchor
        return np.empty((0, probs_k.shape[1]), dtype=np.int64)

    selected = [np.asarray(H_init[k], dtype=np.int64).copy() for k in range(H_init.shape[0])]

    # Pre-compute argmax dosage on kept sites for subtraction
    argmax_dosage_kept = probs_k.argmax(axis=2)                              # (N, L_kept)

    prev_selected = [s.copy() for s in selected]

    for round_num in range(1, max_rounds + 1):
        # 1. Subtraction: generate clean residual candidates
        raw_candidates = _run_subtraction_round(
            selected, argmax_dosage_kept,
            cleanness_threshold=cleanness_threshold)
        if len(raw_candidates) == 0:
            if verbose:
                print(f'  [recovery round {round_num}] no clean residuals -- CONVERGED')
            break

        # 2. Bernoulli mixture fit + BIC over K -> consensus haps
        if verbose:
            print(f'  [recovery round {round_num}] {len(raw_candidates)} raw candidates')
        consensus_haps = _fit_bernoulli_mixture_select_K(
            raw_candidates,
            K_max=mixture_K_max,
            n_restarts=mixture_n_restarts,
            seed=mixture_seed_base + round_num,
            verbose=verbose)
        if len(consensus_haps) == 0:
            if verbose:
                print(f'  [recovery round {round_num}] no mixture consensus -- CONVERGED')
            break

        # 3. Intra-round dedup (safety net only)
        new_haps = []
        for consensus in consensus_haps:
            is_dup = False
            for other in new_haps:
                if _hamming_pct_kept(consensus, other) < intra_round_dedup_pct:
                    is_dup = True
                    break
            if not is_dup:
                new_haps.append(consensus)

        # 4. Pool = selected union new_haps
        pool = list(selected) + list(new_haps)

        # 5. Greedy BIC forward selection (haps frozen)
        sel_indices, sel_haps, sel_nll = _greedy_bic_select(
            pool, probs_k, lam,
            cc_scale=outer_cc_scale, max_k=max_K,
            use_log_bic=use_log_bic, verbose=verbose)

        # 6. Swap refinement (haps still frozen)
        if len(sel_haps) > 0:
            sel_indices, sel_haps, sel_nll, n_swaps = _swap_refine(
                sel_indices, sel_haps, pool, probs_k, lam,
                current_nll=sel_nll,
                nll_tolerance=swap_nll_tolerance,
                verbose=verbose)

        # 7. BIC pruning (haps still frozen)
        if len(sel_haps) > 0:
            sel_indices, sel_haps, sel_nll, n_dropped = _bic_prune(
                sel_indices, sel_haps, probs_k, lam,
                cc_scale=outer_cc_scale, use_log_bic=use_log_bic,
                verbose=verbose)

        # 8. Coord descent on selected to refine haps.  This is the
        #    step that actually MOVES haps — until now the candidates
        #    were frozen (binary outputs of mixture).  CD aligns them
        #    to the actual data.
        if len(sel_haps) > 0:
            H_sel = np.stack(sel_haps, axis=0)
            H_refined, A_ref, costs_ref, wcs_ref, n_iter_ref, nll_ref = \
                _fit_at_fixed_K(probs_k, H_sel, lam, max_iter=max_iter_per_K)
            new_selected = [H_refined[k].copy() for k in range(H_refined.shape[0])]
        else:
            new_selected = sel_haps

        # 9. Convergence check
        if _haps_equal(new_selected, prev_selected, eps_pct=haps_equal_eps_pct):
            if verbose:
                print(f'  [recovery round {round_num}] selected unchanged -- CONVERGED')
            selected = new_selected
            break

        selected = new_selected
        prev_selected = [s.copy() for s in selected]

    # Return as np.array (consistent with K-growth's H format)
    if len(selected) == 0:
        return np.empty((0, probs_k.shape[1]), dtype=np.int64)
    return np.stack(selected, axis=0)


# =============================================================================
# K-MEDOIDS (PAM) FOR INITIAL-K-GROWTH MULTI-START
# =============================================================================

def _kmedoids_pam(D, K, max_iter=MEDOID_PAM_MAX_ITER):
    """Partitioning Around Medoids (PAM) on a precomputed (N, N)
    pairwise distance matrix D.  Returns sorted medoid indices.

    Algorithm:
      BUILD: greedily pick K medoids.  First medoid minimises sum of
             distances to all points.  Subsequent medoids each chosen
             to minimise the resulting total cost (sum over points of
             distance-to-nearest-medoid).
      SWAP:  iteratively try swapping each medoid with each non-medoid;
             accept any swap that strictly reduces total cost.  Stop
             when no improving swap exists or max_iter reached.

    Determinism: pure greedy selection on D, ties broken by index
    order (np.argmin returns first-min).  No RNG.

    Arguments:
        D: (N, N) symmetric non-negative distance matrix
        K: int, number of medoids to pick (must be <= N)
        max_iter: defensive cap on swap-phase iterations

    Returns:
        medoids: list of K int indices, sorted ascending
    """
    N = D.shape[0]
    if K >= N:
        return list(range(N))
    if K <= 0:
        return []

    # ---- BUILD phase ----
    # First medoid minimises total distance to all points.
    sum_d = D.sum(axis=1)
    first = int(sum_d.argmin())
    medoids = [first]

    for k in range(1, K):
        # current_min[i] = min over m in medoids of D[m, i]
        current_min = np.min(D[medoids], axis=0)               # (N,)
        # For each non-medoid candidate c, total cost would be
        # sum_i min(current_min[i], D[c, i]).  We want the c that
        # minimises this.
        best_cost = float('inf')
        best_idx = -1
        for c in range(N):
            if c in medoids:
                continue
            new_min = np.minimum(current_min, D[c])
            cost = float(new_min.sum())
            if cost < best_cost - 1e-9:
                best_cost = cost
                best_idx = c
        if best_idx < 0:
            break
        medoids.append(best_idx)

    medoids = sorted(medoids)

    # ---- SWAP phase ----
    def total_cost(m_list):
        return float(np.min(D[m_list], axis=0).sum())

    cur_cost = total_cost(medoids)
    for _ in range(max_iter):
        improved = False
        for mi in range(len(medoids)):
            for c in range(N):
                if c in medoids:
                    continue
                candidate_medoids = list(medoids)
                candidate_medoids[mi] = c
                new_cost = total_cost(candidate_medoids)
                if new_cost < cur_cost - 1e-9:
                    medoids = candidate_medoids
                    cur_cost = new_cost
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return sorted(medoids)


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

    # Build sample seeds.  Each sample's seed is its argmax-dosage
    # interpretation under _init_hap_from_sample_dosage's rules
    # (forced bits at homozygous sites, population-frequency tiebreak
    # at heterozygous sites).
    seed_array = np.zeros((N, L_kept), dtype=np.int64)
    for s in range(N):
        seed_array[s] = _init_hap_from_sample_dosage(
            probs_k, s, kept_mask=None)

    # Compute pairwise Hamming distance matrix (in [0, 1] units).
    # D[i, j] = mean over l of (seed[i, l] != seed[j, l])
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        D[i] = np.mean(seed_array != seed_array[i], axis=1)

    # Run PAM to pick diverse seed samples
    medoid_indices = _kmedoids_pam(D, n_medoid_starts,
                                     max_iter=MEDOID_PAM_MAX_ITER)

    if verbose:
        if has_trio:
            print(f'[medoid] selected medoids: {medoid_indices}, '
                  f'each branch H_init = stack([H_trio_seed (K={K_trio}), '
                  f'medoid_seed])')
        else:
            print(f'[medoid] selected medoids: {medoid_indices}')

    # Run full per-branch processing from each medoid; keep the best
    # by final BIC.  BIC = K_final * cc + 2 * NLL_final correctly
    # penalises trajectories that grew to a larger K than the data
    # justifies, so a marginally lower NLL at K=8 will lose to a
    # slightly higher NLL at K=6 if the extra two founders aren't
    # paying their complexity cost.
    best_BIC = float('inf')
    best_result = None
    best_medoid = -1

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
        best_medoid = -1   # sentinel for "baseline" (no medoid sample)

    for m in medoid_indices:
        # Build per-branch H_init: trio_seed prefix + medoid m's seed
        if has_trio:
            H_init = np.vstack([H_trio_seed, seed_array[m:m + 1]])
        else:
            H_init = seed_array[m:m + 1]
        result = _process_one_branch(H_init)
        # result tuple: (H, A, per_sample_cost, wildcard_slots,
        #                K_final, wildcard_mass, history)
        result_K = int(result[4])
        result_NLL = float(result[2].sum())
        result_BIC = _compute_bic(result_K, result_NLL, cc)
        if verbose:
            tag = ' + recovery' if run_per_branch_recovery else ''
            print(f'[medoid] start at sample {m}{tag}: '
                  f'K_final={result_K}, NLL={result_NLL:.1f}, '
                  f'BIC={result_BIC:.1f}')
        if result_BIC < best_BIC:
            best_BIC = result_BIC
            best_result = result
            best_medoid = m

    if verbose:
        winner = ('no-medoid baseline' if best_medoid < 0
                  else f'sample {best_medoid}')
        print(f'[medoid] best trajectory: {winner}, '
              f'BIC={best_BIC:.1f}')

    return best_result


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
# haplotypes; across all trios this gives ~3*G^3 candidates, and the
# same true founder appears in many of them.  We cluster by Hamming
# similarity and emit per-cluster majority-vote consensus.
TRIO_HAP_DEDUP_PCT = 2.0            # Hamming pct below which two haps
                                    # are considered the same founder
TRIO_MIN_HAP_CLUSTER_SIZE = 3       # drop hap clusters with fewer than
                                    # this many supporting candidates
                                    # (likely noise rather than real
                                    # founder)


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
    diffs = np.sum(xor_forms[s1_arr] != xor_forms[s2_arr], axis=1)
    return (float(np.median(diffs)),
            float(np.percentile(diffs, 25)),
            float(np.percentile(diffs, 75)))


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
    """
    N, L = xor_forms.shape
    centroids = []   # list of (L,) np.int64 arrays
    members = []     # list of list of sample indices
    bit_votes = []   # list of (L,) np.int64 — per-bit '1' count
    n_members = []   # int per group

    for s in range(N):
        # Find nearest centroid (linear scan over current groups; G is small)
        best_g = -1
        best_d = L + 1
        for g_idx in range(len(centroids)):
            d = int(np.sum(xor_forms[s] != centroids[g_idx]))
            if d < best_d:
                best_d = d
                best_g = g_idx
        if best_g >= 0 and best_d <= threshold_bits:
            members[best_g].append(s)
            bit_votes[best_g] = bit_votes[best_g] + xor_forms[s]
            n_members[best_g] += 1
            centroids[best_g] = (
                bit_votes[best_g] > n_members[best_g] / 2.0
            ).astype(np.int64)
        else:
            centroids.append(xor_forms[s].copy().astype(np.int64))
            members.append([s])
            bit_votes.append(xor_forms[s].astype(np.int64).copy())
            n_members.append(1)

    # Merge-pass: combine any two groups within threshold
    while True:
        merged = False
        G = len(centroids)
        if G < 2:
            break
        for i in range(G):
            inner_break = False
            for j in range(i + 1, G):
                d = int(np.sum(centroids[i] != centroids[j]))
                if d <= threshold_bits:
                    members[i] = members[i] + members[j]
                    bit_votes[i] = bit_votes[i] + bit_votes[j]
                    n_members[i] = n_members[i] + n_members[j]
                    centroids[i] = (
                        bit_votes[i] > n_members[i] / 2.0
                    ).astype(np.int64)
                    centroids.pop(j)
                    members.pop(j)
                    bit_votes.pop(j)
                    n_members.pop(j)
                    merged = True
                    inner_break = True
                    break
            if inner_break:
                break
        if not merged:
            break

    if len(centroids) == 0:
        return (np.zeros((0, L), dtype=np.int64), [], [])
    return (np.stack(centroids, axis=0), members, list(n_members))


def _compute_group_consensus_dosages(dosage, members):
    """For each group, compute per-site modal dosage in {0, 1, 2}
    across its members.  This denoises by majority vote; with ~21
    members per group at K=6 N=320, even per-site dosage error rates
    of 10% become near-zero after consensus."""
    G = len(members)
    if G == 0:
        return np.zeros((0, dosage.shape[1]), dtype=np.int64)
    L = dosage.shape[1]
    consensus = np.zeros((G, L), dtype=np.int64)
    for g_idx, mem_list in enumerate(members):
        if len(mem_list) == 0:
            continue
        member_dosage = dosage[mem_list]  # (group_size, L)
        # Vectorised per-site mode in {0, 1, 2}
        best_count = (member_dosage == 0).sum(axis=0)
        best_val = np.zeros(L, dtype=np.int64)
        for v in (1, 2):
            cnt = (member_dosage == v).sum(axis=0)
            better = cnt > best_count
            best_count = np.where(better, cnt, best_count)
            best_val = np.where(better, v, best_val)
        consensus[g_idx] = best_val
    return consensus


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
    """
    G = centroids.shape[0]
    trios = []
    for g1 in range(G):
        for g2 in range(g1 + 1, G):
            d12 = int(np.sum(centroids[g1] != centroids[g2]))
            if d12 <= distinct_thresh_bits:
                continue  # same-ish centroid — Option B reject
            predicted = centroids[g1] ^ centroids[g2]
            for g3 in range(G):
                if g3 == g1 or g3 == g2:
                    continue
                d_pred = int(np.sum(centroids[g3] != predicted))
                if d_pred > match_thresh_bits:
                    continue
                d_g3_g1 = int(np.sum(centroids[g3] != centroids[g1]))
                d_g3_g2 = int(np.sum(centroids[g3] != centroids[g2]))
                if (d_g3_g1 > distinct_thresh_bits
                        and d_g3_g2 > distinct_thresh_bits):
                    trios.append((g1, g2, g3, d_pred))
    return trios


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
    """
    if len(trios) == 0:
        L = group_dosages.shape[1] if group_dosages.shape[0] > 0 else 0
        return np.zeros((0, L), dtype=np.int64)
    L = group_dosages.shape[1]

    # Build pool of recovered haplotypes
    haps_pool = []
    for g1, g2, g3, _d in trios:
        s1 = group_dosages[g1].astype(np.int64)
        s2 = group_dosages[g2].astype(np.int64)
        s3 = group_dosages[g3].astype(np.int64)
        h_sum = (s1 + s2 + s3) // 2
        haps_pool.append(np.clip(h_sum - s3, 0, 1).astype(np.int64))
        haps_pool.append(np.clip(h_sum - s2, 0, 1).astype(np.int64))
        haps_pool.append(np.clip(h_sum - s1, 0, 1).astype(np.int64))
    haps_pool = np.stack(haps_pool, axis=0)  # (3 * n_trios, L)

    # Cluster recovered haps online
    threshold_bits = int(hap_dedup_pct / 100.0 * L)
    clusters = []        # list of list-of-hap-indices
    centroids = []       # list of (L,) np.int64 arrays
    bit_votes = []       # list of (L,) np.int64 (vote counts)
    n_members = []       # int per cluster
    for hi in range(haps_pool.shape[0]):
        hap = haps_pool[hi]
        best_c = -1
        best_d = L + 1
        for c_idx in range(len(centroids)):
            d = int(np.sum(hap != centroids[c_idx]))
            if d < best_d:
                best_d = d
                best_c = c_idx
        if best_c >= 0 and best_d <= threshold_bits:
            clusters[best_c].append(hi)
            bit_votes[best_c] = bit_votes[best_c] + hap
            n_members[best_c] += 1
            centroids[best_c] = (
                bit_votes[best_c] > n_members[best_c] / 2.0
            ).astype(np.int64)
        else:
            clusters.append([hi])
            centroids.append(hap.copy())
            bit_votes.append(hap.astype(np.int64).copy())
            n_members.append(1)

    # Emit consensus for clusters that meet min size
    final_haps = []
    for c_idx in range(len(clusters)):
        if n_members[c_idx] < min_cluster_size:
            continue
        # Per-site majority vote across cluster members
        cluster_haps = haps_pool[clusters[c_idx]]
        cons = (cluster_haps.sum(axis=0)
                > n_members[c_idx] / 2.0).astype(np.int64)
        final_haps.append(cons)

    if not final_haps:
        return np.zeros((0, L), dtype=np.int64)
    return np.stack(final_haps, axis=0)


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


# =============================================================================
# LATE LOW-CARRIER RESCUE (targeted post-convergence pass)
# =============================================================================

def _late_low_carrier_rescue(probs_k, H, A, costs, wcs, NLL,
                              lam, cc_scale, use_log_bic, max_iter,
                              low_carrier_frac=RECOVERY_LOW_CARRIER_TRIGGER_FRAC,
                              cleanness_threshold=RECOVERY_CLEANNESS_THRESHOLD,
                              verbose=False):
    """Late targeted refinement for blocks with a low-carrier hap that
    may be a chimeric stand-in for a low-frequency founder.

    Triggered when min(carrier_count) < low_carrier_frac * 2*N.  When
    triggered, this pass:

      1. Identifies all suspect haps (carrier count below threshold).
      2. (Diagnostic only when verbose:) generates per-(sample, strand)
         carrier residuals via _generate_carrier_residuals and prints
         their Hamming-% to current H rows.  These residuals are NOT
         used as candidates because by the optimality of A's pair
         assignment, argmax_dosage[s] - H[partner_X] ≈ h_low for every
         carrier — the residual just reproduces the chimera (verified
         empirically at chr6:23624234, all 4 carriers' residuals at
         0.00% from H[chim_5]).
      3. Re-runs trio recovery (_trio_recovery_candidate_haps) on the
         block's probs_k.  Trio candidates come from XOR-triangulation
         of within-cluster sample triples — a structural/algebraic
         property of the genotype data, INDEPENDENT of current H or
         A.  They retain pre-CD-drift information about candidate
         founders, including low-frequency truths that the rest of
         the pipeline drifted away from.  Trio is deterministic
         (TRIO_D_ESTIMATE_SEED=42) and cheap (~100ms at N=320).
      4. Dedups trio candidates against current H at 0.5% (tight; only
         collapse near-exact duplicates so we don't waste evaluations
         on candidates that are already in H).
      5. Builds pool = current H ∪ surviving trio candidates and runs
         greedy BIC forward selection with max_k = K + 1.
      6. Swap refinement: tests "drop selected_i, insert pool_j" at
         fixed K for every (i, j) pair — the operation needed to
         displace a chimera from a slot in favour of a trio candidate.
      7. BIC pruning: drops any selected hap whose NLL contribution
         falls below cc/2.  After swap pulls in a truth-near hap, the
         secondary chimera that was absorbing 'overflow' carriers
         becomes redundant and prunes cleanly — this is what gives
         the K=7 → K=6 BIC win at chr6:23624234.
      8. Refits via _fit_at_fixed_K.  With the secondary chimera
         pruned, the truth_4 carriers' M-step votes on the slot-5
         hap are no longer diluted, and CD pulls the trio candidate
         (from 2.5%) toward truth_4 (at 0%).
      9. Replaces the input state iff BIC strictly improves (by more
         than 0.1 to avoid float-noise oscillation).

    Cost: trio_recovery O(N²), pool size = K + (≤6 trio candidates),
    forward + swap (≤10 passes) + prune + fit each O(K * pool * N *
    L).  About 1-2 sec per triggered block.  Untriggered blocks (min
    carrier count above threshold) return immediately at near-zero
    cost.

    See RECOVERY_LOW_CARRIER_TRIGGER_FRAC for the chr6:23624234
    motivating diagnosis.

    Args:
      probs_k: (N, L_kept, 3) — kept-site posteriors
      H: (K, L_kept) — current discrete founder bits
      A: (N, 2) — current pair assignments
      costs: (N,) — per-sample CAPPED cost (matches _fit_at_fixed_K
        return convention)
      wcs: (N,) — per-sample wildcard slot count
      NLL: float — current UNCAPPED total NLL
      lam: wildcard penalty
      cc_scale: BIC complexity-cost scale (must match outer pipeline)
      use_log_bic: BIC formula selector (must match outer pipeline)
      max_iter: cap on _fit_at_fixed_K coord-descent iterations
      low_carrier_frac: trigger threshold; min carrier fraction of 2N
      cleanness_threshold: min admissible-site fraction for residuals
      verbose: print diagnostic trace

    Returns:
      (H, A, costs, wcs, NLL) — possibly updated; identical to inputs
      if rescue did not trigger or did not yield a BIC improvement.
    """
    N, L_kept = probs_k.shape[0], probs_k.shape[1]
    K = H.shape[0]
    if K == 0:
        return H, A, costs, wcs, NLL

    # Compute per-hap real-strand carrier counts (excluding wildcards)
    W = K
    usage = np.zeros(K, dtype=np.int64)
    for s in range(N):
        for slot in range(2):
            f = int(A[s, slot])
            if f != W:
                usage[f] += 1

    # Trigger condition: any hap below the low-carrier threshold.
    # Floor of 2 ensures we never trigger on degenerate K=0 or K=1
    # blocks where threshold rounds to 0 (e.g., very small N).
    threshold = max(2, int(low_carrier_frac * 2 * N))
    low_idx_list = [k for k in range(K) if int(usage[k]) < threshold]
    if not low_idx_list:
        return H, A, costs, wcs, NLL

    if verbose:
        usage_str = ','.join(f'{k}:{int(usage[k])}' for k in low_idx_list)
        print(f'[late-rescue] triggered: K={K}, low-carrier haps '
              f'(threshold={threshold}): {{{usage_str}}}')

    # Generate candidate haplotypes from carrier residuals.
    #
    # For each (carrier_sample, carrier_strand) of a low-carrier hap
    # h_low, _generate_carrier_residuals iterates over EVERY current H
    # row as a candidate subtractor.  By the algebra (clean data, the
    # other 5 truths are perfect), exactly one subtractor — the actual
    # other strand of that carrier — produces a residual = (the missing
    # truth) at 100% cleanness.  All other subtractors produce residuals
    # with out-of-range bits at heterozygous sites, failing the cleanness
    # threshold.  This is the user's insight (chr6:23624234 message): if
    # we subtract the right partner, we get the missing hap perfectly.
    #
    # The previous broken version subtracted only A[s, other_slot] (the
    # algorithm's *fitted* partner for the given chimera, not the
    # *actual* other strand), producing residual = chim_low for every
    # carrier — verified empirically.  The fix is to test all H rows
    # and let cleanness filter discriminate.
    #
    # Trio candidates are also added as a complementary source: trio
    # generates candidates from XOR-triangulation of within-cluster
    # sample triples — independent of H or A.  At chr6:23624234 trio
    # originally produced a candidate at 2.5% from truth_4, which got
    # drifted to 3% (= chim_5) during downstream CD; re-running trio
    # here regenerates the pre-drift candidate.  Trio is deterministic
    # (TRIO_D_ESTIMATE_SEED=42) and cheap (~100ms at N=320).
    if verbose:
        residuals, provenance = _generate_carrier_residuals(
            probs_k, H, A, low_idx_list,
            cleanness_threshold=cleanness_threshold,
            verbose=True)
        n_examined = len(provenance)
        n_normal = sum(1 for e in provenance if e['partner_kind'] == 'normal')
        print(f'[late-rescue] _generate_carrier_residuals: '
              f'{n_examined} (sample, slot, subtractor) triples examined '
              f'({n_normal} normal, {n_examined - n_normal} skipped), '
              f'{len(residuals)} residuals accepted:')
        for entry in provenance:
            base = (f'    s={entry["sample_idx"]:>3d} slot={entry["slot"]} '
                    f'low={entry["low_idx"]} sub={entry["partner_idx"]} '
                    f'kind={entry["partner_kind"]:<11s}')
            if not entry['accepted']:
                if entry['partner_kind'] == 'normal':
                    print(f'{base} cleanness={entry["cleanness"]:.3f} '
                          f'-- REJECTED (cleanness < threshold)')
                # 'self_low' / 'low_carrier' skipped without printing each
                # — they're high-volume and uninformative; just count them
                # in the header line.
                continue
            r = entry['residual']
            hams = [_hamming_pct_kept(r, H[k]) for k in range(K)]
            ham_str = ', '.join(f'H{k}={h:5.2f}%' for k, h in enumerate(hams))
            print(f'{base} cleanness={entry["cleanness"]:.3f} ACCEPTED  '
                  f'[{ham_str}]')
    else:
        residuals = _generate_carrier_residuals(
            probs_k, H, A, low_idx_list,
            cleanness_threshold=cleanness_threshold)

    # Re-generate trio candidates fresh (deterministic, ~100ms at N=320).
    trio_candidates = _trio_recovery_candidate_haps(probs_k, verbose=False)
    if verbose:
        print(f'[late-rescue] trio produced {trio_candidates.shape[0]} '
              f'candidates.  Hamming-% to current H:')
        for ti in range(trio_candidates.shape[0]):
            hams = [_hamming_pct_kept(trio_candidates[ti], H[k])
                    for k in range(K)]
            ham_str = ', '.join(f'H{k}={h:5.2f}%' for k, h in enumerate(hams))
            print(f'    trio_cand[{ti}]: [{ham_str}]')

    # Combine sources, dedup against current H and against each other
    # at 0.5% (only collapse near-exact duplicates so candidates that
    # differ at a few sites are kept as distinct pool members).
    raw_pool_candidates = list(residuals) + [trio_candidates[ti]
                            for ti in range(trio_candidates.shape[0])]
    if not raw_pool_candidates:
        if verbose:
            print(f'[late-rescue] no candidates to admit — skipping')
        return H, A, costs, wcs, NLL

    H_list = [H[k] for k in range(K)]
    new_candidates = []
    for cand in raw_pool_candidates:
        is_dup = False
        for h in H_list:
            if _hamming_pct_kept(cand, h) < 0.5:
                is_dup = True
                break
        if is_dup:
            continue
        for nc in new_candidates:
            if _hamming_pct_kept(cand, nc) < 0.5:
                is_dup = True
                break
        if not is_dup:
            new_candidates.append(cand)

    if not new_candidates:
        if verbose:
            print(f'[late-rescue] all candidates duplicate current H '
                  f'or each other — skipping')
        return H, A, costs, wcs, NLL

    if verbose:
        print(f'[late-rescue] {len(raw_pool_candidates)} raw candidates → '
              f'{len(new_candidates)} after dedup against H + each other '
              f'(threshold 0.5%)')

    # Pool = current H + new candidates
    pool = H_list + new_candidates

    # Compute current BIC for comparison
    cc = _compute_cc(cc_scale, N, L_kept, use_log_bic=use_log_bic)
    BIC_orig = _compute_bic(K, NLL, cc)

    # Greedy BIC forward selection on enriched pool.  max_k = K + 1
    # allows BIC to grow K by 1, OR shrink K by stopping early when a
    # smaller subset has lower BIC.
    sel_indices, sel_haps, sel_nll = _greedy_bic_select(
        pool, probs_k, lam,
        cc_scale=cc_scale, max_k=K + 1,
        use_log_bic=use_log_bic, verbose=verbose)

    if not sel_haps:
        if verbose:
            print(f'[late-rescue] forward selection picked K=0 — '
                  f'unexpected, keeping original')
        return H, A, costs, wcs, NLL

    # Swap refinement (1-for-1 swap): for each selected slot, try every
    # pool member as a replacement.  Accepts a swap if NLL improves by
    # more than tolerance.  Mirrors the swap step inside
    # _subtraction_recovery_round_loop.  This is the operation that
    # displaces a chimera from its slot when a clean carrier residual
    # (= the missing truth founder) is in the pool.
    sel_indices, sel_haps, sel_nll, n_swaps = _swap_refine(
        sel_indices, sel_haps, pool, probs_k, lam, sel_nll,
        nll_tolerance=RECOVERY_SWAP_NLL_TOLERANCE,
        max_passes=10, verbose=verbose)

    # BIC-pruning: after a swap pulls in a truth-near hap, the
    # secondary chimera that was absorbing 'overflow' carriers (e.g.
    # chim_6 at chr6:23624234, ~21 carriers fitting truth_1 with 3%
    # drift) becomes redundant — the truth-near slot can absorb its
    # own carriers cleanly, and the remaining carriers' true partners
    # are the existing truths.  Drops any hap whose NLL contribution
    # falls below cc/2.  This is what actually gives the K=7 → K=6
    # BIC win.
    sel_indices, sel_haps, sel_nll, n_dropped = _bic_prune(
        sel_indices, sel_haps, probs_k, lam,
        cc_scale=cc_scale, use_log_bic=use_log_bic, verbose=verbose)

    if not sel_haps:
        if verbose:
            print(f'[late-rescue] post-swap-prune picked K=0 — '
                  f'unexpected, keeping original')
        return H, A, costs, wcs, NLL

    # Refit at chosen K (forward+swap+prune used FIXED haps; refit lets
    # CD drift candidate haps toward the actual truth bits.  In the
    # chr6:23624234 case, after swap pulls truth_4 (= a clean carrier
    # residual) into slot 5 and prune drops chim_6, refit polishes the
    # K=6 state).
    H_new = np.array(sel_haps, dtype=np.int64)
    H_new, A_new, costs_new, wcs_new, n_iter_new, NLL_new = _fit_at_fixed_K(
        probs_k, H_new, lam, max_iter=max_iter)
    K_new = H_new.shape[0]
    BIC_new = _compute_bic(K_new, NLL_new, cc)

    if verbose:
        swap_str = f'{n_swaps} swap{"s" if n_swaps != 1 else ""}'
        prune_str = f'{n_dropped} prune{"s" if n_dropped != 1 else ""}'
        print(f'[late-rescue] orig: K={K}, NLL={NLL:.1f}, BIC={BIC_orig:.1f}')
        print(f'[late-rescue] new:  K={K_new}, NLL={NLL_new:.1f}, '
              f'BIC={BIC_new:.1f} (after {swap_str}, {prune_str})')

    if BIC_new < BIC_orig - 0.1:
        if verbose:
            print(f'[late-rescue] BIC improved by {BIC_orig - BIC_new:.1f} '
                  f'— ACCEPT')
        return H_new, A_new, costs_new, wcs_new, NLL_new
    if verbose:
        print(f'[late-rescue] BIC did not improve '
              f'(delta={BIC_orig - BIC_new:+.1f}) — KEEP ORIGINAL')
    return H, A, costs, wcs, NLL


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
    if TRIO_RECOVERY_ENABLED:
        H_trio_candidates = _trio_recovery_candidate_haps(
            probs_k, verbose=verbose)
        if H_trio_candidates.shape[0] >= 1:
            cand_list = [H_trio_candidates[k]
                         for k in range(H_trio_candidates.shape[0])]
            # Greedy forward-selection BIC trim.  Uses the same cc_scale
            # AND use_log_bic as the K-growth that follows, so trim and
            # grow share an identical BIC criterion.  Each accepted hap
            # strictly improves BIC; rejected haps are dropped.
            sel_indices, sel_haps, _trim_nll = _greedy_bic_select(
                cand_list, probs_k, lam,
                cc_scale=cc_scale,
                max_k=K_max,
                use_log_bic=use_log_bic,
                verbose=verbose)
            if sel_haps:
                H_seed = np.stack(sel_haps, axis=0).astype(np.int64)
            if verbose:
                print(f'[trio] {H_trio_candidates.shape[0]} candidates -> '
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
        recovery_cleanness_threshold=recovery_cleanness_threshold,
        recovery_swap_nll_tolerance=recovery_swap_nll_tolerance,
        recovery_haps_equal_eps_pct=recovery_haps_equal_eps_pct,
        verbose=verbose)

    if verbose:
        print(f'[recovery] Initial K-growth: K_final={K_final}, '
              f'wildcard_mass={wm:.4f}')

    # 2. Outer iteration: alternate recovery and K-growth
    for outer_it in range(recovery_max_outer_iterations):
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
    N = probs_k.shape[0]
    W = K

    confidence = np.zeros((K, L), dtype=np.float64)
    n_supporters = np.zeros((K, L), dtype=np.int64)

    for k in range(K):
        is_kk = (A[:, 0] == k) & (A[:, 1] == k)
        is_kW = (A[:, 0] == k) & (A[:, 1] == W)
        has_k = (A[:, 0] == k) | (A[:, 1] == k)
        is_kj = has_k & ~is_kk & ~is_kW

        for l in range(L):
            cur_val = H_k[k, l]
            n_supp = 0
            n_consistent = 0

            # Bucket (k, k): consistent if data prefers genotype = 2*cur_val
            if is_kk.any():
                p = probs_k[is_kk, l, :]
                consistent_mask = (p.argmax(axis=1) == 2 * cur_val)
                n_supp += int(is_kk.sum())
                n_consistent += int(consistent_mask.sum())

            # Bucket (k, j): consistent if data prefers genotype = cur_val + H_k[j, l]
            if is_kj.any():
                idx = np.where(is_kj)[0]
                a0 = A[idx, 0]; a1 = A[idx, 1]
                partner = np.where(a0 == k, a1, a0)
                expected_dosage = cur_val + H_k[partner, l]
                p = probs_k[idx, l, :]
                consistent_mask = (p.argmax(axis=1) == expected_dosage)
                n_supp += len(idx)
                n_consistent += int(consistent_mask.sum())

            # Bucket (k, W): consistent if real-pair fit beats wildcard
            #  (since the wildcard might mask any data, "consistent" here
            #   means the real founder's allele actually contributed
            #   information rather than just letting the wildcard absorb)
            if is_kW.any():
                p = probs_k[is_kW, l, :]                   # (n_P, 3)
                # Cost under real-(k,W) at this site:
                #   -log max(p[:, cur_val], p[:, cur_val+1]) + λ
                best_real = np.maximum(p[:, cur_val], p[:, cur_val + 1])
                cost_real = _safe_neg_log(best_real) + lam
                # Cost under (W, W):
                cost_WW = _safe_neg_log(p.max(axis=1)) + 2.0 * lam
                # Sample is "consistent" with the real founder at this site
                # if the real-pair cost is meaningfully better than (W, W).
                consistent_mask = cost_real < cost_WW
                n_supp += int(is_kW.sum())
                n_consistent += int(consistent_mask.sum())

            n_supporters[k, l] = n_supp
            if n_supp >= min_supporters:
                confidence[k, l] = n_consistent / n_supp
            # else: confidence stays 0 (low-support site)

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
        probs_k = probs_array[:, kept_mask, :]
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
    H_k, A, per_sample_cost, wildcard_slots, K_final, wildcard_mass, history = \
        _grow_K_with_recovery(probs_k, kept_mask,
                              lam=lambda_wildcard_penalty,
                              wildcard_mass_threshold=wildcard_mass_threshold,
                              min_relative_improvement=min_wildcard_relative_improvement,
                              K_max=K_max,
                              max_iter_per_K=coord_descent_max_iter,
                              n_medoid_starts=n_medoid_starts)

    # --- 5. COMPUTE PER-SITE CONFIDENCE (kept-site coords) ---
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
# generate_haplotypes_block_robust — same iterative-residual-discovery
# wrapper as the legacy module, calling our generate_haplotypes_block.
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

        # Residual check: legacy uses find_missing_haplotypes_iterative
        # with k-limited Viterbi.  We reuse that logic via the legacy
        # module's helper — it operates on hap_dict + probs_array, and
        # our hap_dict is in the same format.
        try:
            from block_haplotypes import find_missing_haplotypes_iterative
        except ImportError:
            break

        missing_haps_dict = find_missing_haplotypes_iterative(
            positions, reads_array, final_result.haplotypes,
            keep_flags=keep_flags,
            error_threshold_percent=2.0,
            min_bad_samples=5)

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

    import numba as _numba
    if _BH_ACTIVE_COUNTER is not None and _BH_TOTAL_CORES is not None:
        with _BH_ACTIVE_COUNTER.get_lock():
            _BH_ACTIVE_COUNTER.value += 1
        active = max(_BH_ACTIVE_COUNTER.value, 1)
        n_threads = max(1, _BH_TOTAL_CORES // active)
        _numba.set_num_threads(n_threads)

    try:
        result = generate_haplotypes_block_robust(
            positions, reads, keep_flags=flags, **kwargs)
        _malloc_trim()
        return (block_idx, result)
    finally:
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
        with _ForkserverPool(processes=num_processes,
                              initializer=_init_block_worker,
                              initargs=(num_processes, active_counter)) as pool:
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