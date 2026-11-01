"""Name the family model and search parameters.

Owns the relocated parameters units. No storage, environment or runtime policy.
Method: docs/methods/relatedness.md; WP_RL1. Legacy arithmetic retained for relocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AlleleFrequencyParameters:
    iterations: int = field(default=6, metadata={"provenance": "3.5 kernels.py:24"})
    initial: float = field(default=0.5, metadata={"provenance": "3.5 kernels.py:44"})
    clip: float = field(default=1e-4, metadata={"provenance": "3.5 kernels.py:63"})
    total_floor: float = field(default=1e-300, metadata={"provenance": "3.5 kernels.py:57"})


@dataclass(frozen=True)
class FamilyParameters:
    minor_allele_frequency: float = field(default=0.05, metadata={"provenance": "3.5 config.py:36"})
    minimum_observed: float = field(default=0.5, metadata={"provenance": "3.5 config.py:37"})
    family_site_step: int = field(default=20, metadata={"provenance": "3.5 config.py:38"})
    max_sweeps: int = field(default=30, metadata={"provenance": "3.5 config.py:41"})
    support_unit_morgans: float = field(default=1.0, metadata={"provenance": "3.5 config.py:42"})
    rate_is_measured: bool = field(default=False, metadata={"provenance": "3.5 config.py:43"})
    gain_tolerance: float = field(default=1e-6, metadata={"provenance": "3.5 families.py:56"})
    screen_margin: float = field(default=0.1, metadata={"provenance": "3.5 families.py:65"})
    screen_margin_per_site: float = field(default=1e-6, metadata={"provenance": "3.5 families.py:66"})
    screen_rounding: float = field(default=2.0**-53, metadata={"provenance": "3.5 families.py:67"})
    screen_padding: int = field(default=16, metadata={"provenance": "3.5 families.py:178"})
    screen_tile: int = field(default=16, metadata={"provenance": "3.5 families.py:68"})
    sum_bytes: int = field(default=32 * 1024**2, metadata={"provenance": "3.5 families.py:69"})
    pair_batch: int = field(default=262144, metadata={"provenance": "3.5 families.py:70"})
    prefetch_batch: int = field(default=64, metadata={"provenance": "3.5 families.py:71"})
    term_floor: float = field(default=1e-300, metadata={"provenance": "3.5 kernels.py:88-95"})


# Fixed numerical/scheduling constants (not user settings). Source changes enter both code identities.
LOG_ZERO = -1e300  # 3.5 kernels.py:104; max initializer, every log-domain kernel.
SCREEN_FLOOR = 1e-30  # 3.5 kernels.py:461; float32 site fallback threshold.
SCALE_EXPONENT = 500  # 3.5 kernels.py:462-464; exact binary product rescaling.
EVIDENCE_COMPARE_CHUNK = 256  # 3.5 families.py:87; early-exit storage chunk only.
SCREEN_SITE_TILE = 256  # 3.5 kernels.py:501; storage-only site tiling.

__all__ = ["AlleleFrequencyParameters", "FamilyParameters"]
