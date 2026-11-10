"""Generating-model constants and reported working-set estimates.

Rates, lengths and example sizes are fixed defaults. Working-set estimates
control execution reporting, never the generating model.
"""

MUTATION_RATE_PER_BP: float = 1e-10  # Mutations per bp; exponential inter-mutation distances use this rate.
FRAGMENT_QUANTILES: int = 100_000  # Quantile count for deterministic integration of covered bases per fragment.
MINIMUM_FRAGMENT_BP: int = 50  # bp; lower clipping bound of the generating fragment-length law.
FRAGMENT_SD_LIMIT: int = 6  # Standard deviations above the mean; upper fragment-length clipping bound.
READS_BYTES_PER_CELL: int = 48  # Bytes per sample-site cell in the reported read working-set estimate.
READS_WORKER_BYTES: int = 1 << 30  # Bytes of fixed worker allowance in the reported read working-set estimate.
EXAMPLE_HAPLOTYPES: int = 6  # Founder haplotype count for the synthetic onboarding example.
EXAMPLE_COHORTS: tuple[int, ...] = (20, 30, 30)  # Individuals in each successive cohort of the synthetic example.
EXAMPLE_DEPTH: float = 5.0  # Expected reads per site and sample in the synthetic example.

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
