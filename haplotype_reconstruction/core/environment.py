"""core / environment for the canonical reconstruction pipeline."""
from __future__ import annotations


import os


NUMERIC_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def assembly_transition_model():
    """Assembly choice is independent of discovery; dense is the default."""
    value = os.environ.get("HAPLOTYPES_ASSEMBLY_MODEL", "dense")
    if value not in ("dense", "structured"):
        raise ValueError("HAPLOTYPES_ASSEMBLY_MODEL must be dense or structured")
    return value


def batched_discovery_enabled():
    """The experimental discovery search requires its own explicit choice."""
    value = os.environ.get("HAPLOTYPES_DISCOVERY_SEARCH", "standard")
    if value not in ("standard", "batched"):
        raise ValueError("HAPLOTYPES_DISCOVERY_SEARCH must be standard or batched")
    return value == "batched"


def force_single_threaded_numeric_libraries():
    """Force BLAS/OpenMP libraries to one thread before importing them."""

    for variable in NUMERIC_THREAD_ENV_VARS:
        os.environ[variable] = "1"
