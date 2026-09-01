"""Centralized tunable configuration for the bhd_* haplotype-recovery ecosystem.

Single source of truth for the pipeline's tunable knobs, thresholds, and feature
flags.  Logic-free: imports nothing, so every bhd_* module (and block_haplotypes)
can pull its constants from here without import cycles.  Low-level numerical
sentinels that are not user-tuned (bhd_kernels.MASK, bhd_kernels.LOG_EPS)
deliberately stay in their home modules.
"""


# ============================================================================
# Shared sequencing-error model
# ============================================================================
# Per-read probability of observing the opposite allele.  All raw genotype-
# likelihood and VCF-loading entry points use this value unless overridden.
DEFAULT_READ_ERROR_PROBABILITY = 0.02


# ============================================================================
# Terminal whole-bin cavity refinement
# ============================================================================
# Post-L4 allele refinement uses 100-site HMM bins and scores the strict 2*K+2
# two-basin candidate family against an immutable structural ancestry track.
# Rho is the per-alternative transition log penalty per bin; the total switch
# probability therefore depends on K. The emission mixture and floor robustify
# individual log-likelihood contributions. Chunk sizes affect only working-
# memory use and scheduling, not the statistical model or result.
TERMINAL_CAVITY_SNPS_PER_BIN = 100
TERMINAL_CAVITY_RHO = 10.0
TERMINAL_CAVITY_SAMPLE_CHUNK_SIZE = 128
TERMINAL_CAVITY_SITE_CHUNK_SIZE = 1024
TERMINAL_CAVITY_EMISSION_UNIFORM_MIX = 0.01
TERMINAL_CAVITY_LOG_EMISSION_FLOOR = -2.0


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
# ============================================================================
# Fixed-panel Viterbi objective
# ============================================================================
# Per-sample model scores allow founder-pair changes only between fixed SNP
# bins, with one flat penalty per change. Pair assignments used by the founder
# update remain the best fixed pair; the Viterbi path supplies the model score.
_VITERBI_BIC_ENABLED = True
VITERBI_SWITCH_PENALTY = 10.0
# Bin granularity for Viterbi: each bin sums log-prob emissions within the
# bin before applying the inter-bin switch penalty.  At spb=10 (the
# default), a 200-SNP block has 20 bins and Viterbi can switch pair states
# at most 19 times.  Matches chimera_resolution.py's L1 anchor
# (compute_spb() = max(10, avg_sites//20) gives 10 for L=200 blocks).
# Lower spb => more switching points (more granular chimera handling but
# more compute); higher spb => fewer switch points (coarser, faster).
VITERBI_SNPS_PER_BIN = 10


# ============================================================================
# Fixed-panel subset scoring
# ============================================================================
# Maximum cached Viterbi-emission tensor per subset-scoring session.
POOL_EMISSION_CACHE_MAX_BYTES = 256 * 1024 * 1024
# Complexity scaling and minimum NLL improvement used by Pearly's fixed-panel
# subset selector. These parameters do not control Stage-1 founder discovery.
FIXED_PANEL_OUTER_CC_SCALE = 0.5
FIXED_PANEL_SWAP_NLL_TOLERANCE = 0.5


# ============================================================================
# Fixed-K fitter execution
# ============================================================================
# Maximum Numba threads used by one fixed-K coordinate-descent fit. Independent
# blocks still occupy the full pool in bulk; this cap only prevents one tail
# block from driving the memory-bandwidth-bound A/H kernels past their measured
# scaling optimum. The cavity-scoring phase remains uncapped and can consume
# every dynamically available core. Calibrated at 16 on 76-core Ice Lake; keep
# this explicit so another architecture can be benchmarked and adjusted without
# changing the statistical model.
FIXED_K_FIT_MAX_THREADS = 16


# ============================================================================
# Reversible-cavity candidate generation
# ============================================================================
# Maximum number of soft-clustered data basins used by the generic fixed-panel
# API. The main reversible controller may set its own explicit operational
# budget.
DEFAULT_DATA_SEED_MODES = 5
# HDBSCAN support required for a sample cluster to propose a pooled seed.
DEFAULT_SOFT_SEED_MIN_CLUSTER_SIZE = 3
# Candidate rows closer than this percentage are treated as duplicates by the
# proposal-D residual candidate builder.
CANDIDATE_DEDUP_HAMMING_PERCENT = 0.5


# ============================================================================
# Bernoulli-mixture model for F2 block-subtraction recovery
# ============================================================================
RECOVERY_MIXTURE_K_MAX = 10
RECOVERY_MIXTURE_N_RESTARTS = 2
RECOVERY_MIXTURE_MAX_ITER = 100
RECOVERY_MIXTURE_TOL = 1e-3
RECOVERY_MIXTURE_THETA_EPS = 1e-3
RECOVERY_MIXTURE_RNG_SEED = 42
# Stop after this many consecutive component counts fail to improve BIC.
RECOVERY_MIXTURE_PATIENCE = 3
