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
READ_BLOCKS_PER_CORE: int = 2  # Sample blocks per allocated CPU; execution grain, with independent sample streams.
SOURCE_BYTES_PER_SITE: int = 3  # Bytes per source-site cell: derived, indexed and REF-coded mutation panels.
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
    "READ_BLOCKS_PER_CORE",
    "SOURCE_BYTES_PER_SITE",
    "EXAMPLE_HAPLOTYPES",
    "EXAMPLE_COHORTS",
    "EXAMPLE_DEPTH",
]

# Genealogy traversal safety limits; these refuse work without changing the SMC' law.
GENEALOGY_WORK_LIMIT: float = 5e7  # generation-Morgans; frozen first-tree length times contig-length proxy.
GENEALOGY_RECORD_BLOCK: int = 65_536  # intervals per allocation step; avoids an up-front ceiling-sized record.
GENEALOGY_INTERVAL_LIMIT: int = 60_000_000  # intervals; two float64 records use at most 960 MB of payload.
SITE_CHUNK_BP: int = 262_144  # integer bp per weight/clock chunk; bounds scratch memory, never a model scale.
PROFILE_SCALES_BP: tuple[int, ...] = (10, 100, 1_000, 10_000, 100_000)  # bp; frozen cascade cell lengths.
PROFILE_SIGMAS: tuple[float, ...] = (0.9, 0.6, 0.7, 0.6, 0.45)  # log-intensity SD per scale; prototype defaults.
PROFILE_MASKED_FRACTION: float = 0.10  # stationary covered share; prototype default, exact Boolean mask law.
PROFILE_MASKED_MEAN_BP: float = 6_000.0  # bp; mean exponential stretch length, prototype default.
SITE_BISECTION_STEPS: int = 80  # iterations; resolves the clock threshold to float64 precision.

__all__ += [
    'GENEALOGY_WORK_LIMIT', 'GENEALOGY_RECORD_BLOCK', 'GENEALOGY_INTERVAL_LIMIT', 'SITE_CHUNK_BP',
    'PROFILE_SCALES_BP', 'PROFILE_SIGMAS', 'PROFILE_MASKED_FRACTION', 'PROFILE_MASKED_MEAN_BP',
    'SITE_BISECTION_STEPS',
]

# Call-set artefact defaults; enabling the named preset uses these values.
ARTEFACT_PRESET: str = "reference_blocks"  # Name of the generating call-set preset.
ARTEFACT_ENABLED: bool = False  # Whether the count transformation is applied.
ARTEFACT_HARD_SITES: float = 0.01  # Probability that a site has the hard-site miss term.
ARTEFACT_HARD_MISS: float = 0.9  # Non-reference miss probability from a hard site.
ARTEFACT_STRETCH_COVER: float = 0.03  # Long-axis covered share of low-sensitivity stretches.
ARTEFACT_STRETCH_SITES: float = 6.0  # Sites; geometric mean length of a low-sensitivity stretch.
ARTEFACT_STRETCH_MISS: float = 0.8  # Non-reference miss probability inside a stretch.
ARTEFACT_MISS: float = 0.012  # Baseline non-reference miss probability before sample/region factors.
ARTEFACT_MISS_SAMPLE_CV: float = 0.5  # Dimensionless CV of a mean-one lognormal baseline factor.
ARTEFACT_CALL_MIN_ALT_FRACTION: float = 0.15  # Minimum ALT/read fraction calling a hom-reference cell.
ARTEFACT_BLOCK_BAND: float = 0.5  # Relative width of logarithmic depth bands, dimensionless; reproduces the measured share of cells in identical-depth runs.
ARTEFACT_BALANCE_DISPERSION: float = 0.05  # Beta-binomial intraclass correlation of heterozygous reads.
ARTEFACT_SYSTEMATIC_SITES: float = 0.03  # Probability of a shared systematic ALT-error site; reproduces the measured false-ALT read share.
ARTEFACT_SYSTEMATIC_LO: float = 0.1  # Lower endpoint of the uniform systematic ALT/read probability.
ARTEFACT_SYSTEMATIC_HI: float = 0.4  # Upper endpoint of the uniform systematic ALT/read probability.
ARTEFACT_SITE_STREAM: str = "simulation.artefact.sites"  # Stream keyed by realised seed and contig ID.
ARTEFACT_SAMPLE_STREAM: str = "simulation.artefact.sample"  # Stream keyed by seed, sample ID and contig ID.
ARTEFACT_RENEWAL_BATCH: int = 1024  # Renewal intervals per vectorised allocation; no process truncation.

__all__ += [
    "ARTEFACT_PRESET", "ARTEFACT_ENABLED", "ARTEFACT_HARD_SITES", "ARTEFACT_HARD_MISS",
    "ARTEFACT_STRETCH_COVER", "ARTEFACT_STRETCH_SITES", "ARTEFACT_STRETCH_MISS", "ARTEFACT_MISS",
    "ARTEFACT_MISS_SAMPLE_CV", "ARTEFACT_CALL_MIN_ALT_FRACTION", "ARTEFACT_BLOCK_BAND",
    "ARTEFACT_BALANCE_DISPERSION", "ARTEFACT_SYSTEMATIC_SITES", "ARTEFACT_SYSTEMATIC_LO",
    "ARTEFACT_SYSTEMATIC_HI", "ARTEFACT_SITE_STREAM", "ARTEFACT_SAMPLE_STREAM", "ARTEFACT_RENEWAL_BATCH",
]

# Optional cohort read/site extensions; all default to the released off state.
SITE_LOSS_FRACTION: float = 0.0  # Impaired share of site-depth windows, dimensionless, in [0, 1).
REGION_DEPTH_CV: float = 0.0  # Per-sample, per-window mean-one lognormal depth CV, dimensionless.
REGION_DEPTH_WINDOW_BP: int = 0  # bp; a positive physical width is required when regional CV is nonzero.
VARIANT_AGAINST: str = "cohort"  # Separating set: carried sources, or carried sources plus reference.
SPURIOUS_FRACTION: float = 0.0  # Share of written sites without a genealogical ALT allele, in [0, 1).

__all__ += ["SITE_LOSS_FRACTION", "REGION_DEPTH_CV", "REGION_DEPTH_WINDOW_BP", "VARIANT_AGAINST", "SPURIOUS_FRACTION"]

SITE_LOSS_FINE_FRACTION: float = 0.0  # Impaired share of independent fine windows, dimensionless, in [0, 1).
SITE_LOSS_FINE_WINDOW_BP: int = 0  # bp; positive when the fine impairment fraction is positive.

__all__ += ["SITE_LOSS_FINE_FRACTION", "SITE_LOSS_FINE_WINDOW_BP"]
