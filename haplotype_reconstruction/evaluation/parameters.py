"""Name carried scientific definitions and reported working-set estimates.

Each value cites its P0 unit and frozen 3.5 source. No values are inferred
from environment, worker count, clocks or filesystem state.
"""

LONG_SWITCH_MIN_RUN: int = 20  # OU36: B evaluation/metrics.py:34; metric definition.
SEGMENT_SWITCH_COST: float = 20.0  # OU33: B evaluation/founders.py:38; exact metric definition.
SITE_TILE: int = 65_536  # OU32: B evaluation/founders.py:33; integer counting work decomposition.
IDENTITY_WINDOW_MARKERS: int = 100  # OU35: B evaluation/identity.py:36; truth-marker grid.
LABEL_MIN_CALLS: int = 16  # OU35: B evaluation/identity.py:37.
LABEL_MARGIN: int = 2  # OU35: B evaluation/identity.py:38.
LABEL_ERROR_FRACTION: float = 0.05  # OU35: B evaluation/identity.py:94.
LABEL_MIN_ERRORS: int = 1  # OU35: B evaluation/identity.py:94.
COVERAGE_MIN_DISTANCE: int = 3  # OU35: B evaluation/identity.py:39.
COVERAGE_MAX_GAP: int = 3  # OU35: B evaluation/identity.py:40.
TASK_ALLOWANCE_BYTES: int = 256 << 20  # OU31: B evaluation/evaluate.py:69; scheduling estimate only.
PHASE_METRIC: str = "released_phase_v3"  # OU36: B evaluation/evaluate.py:40; published definition version.

__all__ = [
    "LONG_SWITCH_MIN_RUN",
    "SEGMENT_SWITCH_COST",
    "SITE_TILE",
    "IDENTITY_WINDOW_MARKERS",
    "LABEL_MIN_CALLS",
    "LABEL_MARGIN",
    "LABEL_ERROR_FRACTION",
    "LABEL_MIN_ERRORS",
    "COVERAGE_MIN_DISTANCE",
    "COVERAGE_MAX_GAP",
    "TASK_ALLOWANCE_BYTES",
    "PHASE_METRIC",
]
