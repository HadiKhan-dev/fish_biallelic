"""Standalone metadata-aware Tropheops V7-margin pedigree package.

Importing :mod:`thread_config` before NumPy/Numba preserves the repository's
oversubscription and disk-cache safeguards. Numba decorates some internal CPU
registry helpers lazily, and those helpers do not always have a cache locator.
The tiny scoped warm-up below binds those internal helpers to Numba's original
decorator without changing the process-wide project wrapper. The selected V7
kernels explicitly declare ``cache=True``, preserving their validated policy.
"""

import thread_config as _thread_config

try:
    import numba as _numba

    _project_njit_wrapper = _numba.njit
    _real_njit = getattr(
        _thread_config, "_original_njit", _project_njit_wrapper
    )
    try:
        _numba.njit = _real_njit

        @_real_njit(cache=False)
        def _pedigree_v7_numba_registry_warmup(value):
            return value + 1

        _pedigree_v7_numba_registry_warmup(0)
    finally:
        _numba.njit = _project_njit_wrapper
except ImportError:
    _numba = None

from .aggregation import (
    V7MarginResult,
    aggregate_v7_margin,
    robust_information_weighted_utilities,
)
from .design import (
    COMPATIBILITY_DESIGN,
    CONTIGS,
    TropheopsV7Design,
    candidate_sets,
    load_g0_seed_assignments,
)

__all__ = (
    "COMPATIBILITY_DESIGN", "CONTIGS", "TropheopsV7Design",
    "V7MarginResult", "aggregate_v7_margin", "candidate_sets",
    "load_g0_seed_assignments", "robust_information_weighted_utilities",
)
