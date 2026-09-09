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


def force_single_threaded_numeric_libraries():
    """Force BLAS/OpenMP libraries to one thread before importing them."""

    for variable in NUMERIC_THREAD_ENV_VARS:
        os.environ[variable] = "1"


