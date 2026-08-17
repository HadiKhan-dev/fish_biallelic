"""Standalone metadata-aware Tropheops V7-margin pedigree package.

Importing :mod:`thread_config` before NumPy/Numba preserves the repository's
oversubscription, registry warm-up, and disk-cache safeguards.  V7 kernels
explicitly declare ``cache=True``, preserving their validated policy.
"""

import thread_config as _thread_config
_thread_config.ensure_numba_registry_warmup()

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
