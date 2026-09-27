"""Founder-haplotype reconstruction from low-coverage experimental crosses."""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
__version__ = "1.0.0"

# Set native-library limits before any numerical submodule is imported.
from .core.environment import (
    configure_numba_cache,
    force_single_threaded_numeric_libraries,
)

force_single_threaded_numeric_libraries()
configure_numba_cache()
