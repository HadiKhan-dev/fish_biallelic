"""Shared scientific constants; stage-specific settings live beside their models."""
from __future__ import annotations


DEFAULT_READ_ERROR_PROBABILITY = 0.02

# Shared linker's block-local quality CTMC, separate from read-error rates.
# Stationary unreliable-sequence fraction and mean error-tract length in bp.
LINKER_ERROR_FRACTION = 0.01
LINKER_ERROR_TRACT_BP = 20_000.0


_VITERBI_BIC_ENABLED = True


VITERBI_SWITCH_PENALTY = 10.0


VITERBI_SNPS_PER_BIN = 10


FIXED_K_FIT_MAX_THREADS = 16


DEFAULT_SOFT_SEED_MIN_CLUSTER_SIZE = 3


CANDIDATE_DEDUP_HAMMING_PERCENT = 0.5
