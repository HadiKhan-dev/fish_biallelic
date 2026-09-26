"""Shared scientific constants; stage-specific settings live beside their models."""
from __future__ import annotations

from dataclasses import dataclass


DEFAULT_READ_ERROR_PROBABILITY = 0.02


@dataclass(frozen=True)
class ReadCalibrationConfig:
    """Numerical/sampling settings for nested observed-AD model fitting."""

    marker_stride: int = 8
    maximum_fit_depth: int = 48
    fold_window_bp: int = 1_000_000
    initial_parameters: tuple = ((DEFAULT_READ_ERROR_PROBABILITY, 0.5),
                                (0.01, 0.35), (0.08, 0.65))
    maximum_iterations: int = 2000
    likelihood_tolerance_per_observation: float = 1e-10


@dataclass(frozen=True)
class PathSelectionConfig:
    """Bounded local-panel search; shared by initial and feedback fits."""

    max_updates: int = 20
    rounds: int = 3
    refits_per_kind: int = 8


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
