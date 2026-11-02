"""Name carried scientific definitions and reported working-set estimates.

Each value cites its P0 unit and frozen 3.5 source. No values are inferred
from environment, worker count, clocks or filesystem state.
"""

MUTATION_RATE_PER_BP: float = 1e-10  # OU26: B simulation/design.py:109; fixed generating rate.
FRAGMENT_QUANTILES: int = 100_000  # OU28: B simulation/reads.py:30; deterministic quadrature.
MINIMUM_FRAGMENT_BP: int = 50  # OU28: B simulation/reads.py:31; generating length floor.
FRAGMENT_SD_LIMIT: int = 6  # OU28: B simulation/reads.py:77; upper clipping extent.
READS_BYTES_PER_CELL: int = 48  # OU30: B simulation/source.py:35; reported working-set model.
READS_WORKER_BYTES: int = 1 << 30  # OU30: B simulation/source.py:36; reported worker allowance.
EXAMPLE_HAPLOTYPES: int = 6  # OU22: B templates/example.py:13.
EXAMPLE_COHORTS: tuple[int, ...] = (20, 30, 30)  # OU22: B templates/example.py:14.
EXAMPLE_DEPTH: float = 5.0  # OU22: B templates/example.py:15.

__all__ = [
    "MUTATION_RATE_PER_BP",
    "FRAGMENT_QUANTILES",
    "MINIMUM_FRAGMENT_BP",
    "FRAGMENT_SD_LIMIT",
    "READS_BYTES_PER_CELL",
    "READS_WORKER_BYTES",
    "EXAMPLE_HAPLOTYPES",
    "EXAMPLE_COHORTS",
    "EXAMPLE_DEPTH",
]
