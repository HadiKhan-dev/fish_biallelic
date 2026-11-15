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
ARTEFACT_HOM_ALT_MISS_FACTOR: float = 0.02  # Miss probability of a hom-ALT cell over a het's: the 52-sample ONT
#                                             trios miss 0.04-0.11% of hom-ALT cells against 3.5-4.6% of hets.
# Long reads carry their own allele balance (context sites, reference bias, shared reads); the call-set layer's
# per-cell beta-binomial redraw would resample each heterozygous cell on its own, erasing the reads shared by
# neighbouring sites, and its value was fitted to a spread that included the misses the layer adds separately.
BALANCE_WITH_LONG_READS: str = (
    "[simulation.artefacts] balance_dispersion must be 0 with long reads ([simulation.reads] long_read_mean_bp > 0): "
    "the long-read model makes the heterozygote balance itself, and the per-cell redraw would erase the reads that "
    "neighbouring sites share (the long_read_cohort read preset sets it to 0)"
)
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
    "ARTEFACT_MISS_SAMPLE_CV", "ARTEFACT_HOM_ALT_MISS_FACTOR", "BALANCE_WITH_LONG_READS",
    "ARTEFACT_CALL_MIN_ALT_FRACTION", "ARTEFACT_BLOCK_BAND",
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

# Long-read mode of the read sampler (simulation/reads.py), off while long_read_mean_bp is zero. Values are the
# generating defaults of [simulation.reads] and the long_read_cohort preset; the preset file states its calibration.
LONG_READ_MEAN_BP: float = 0.0  # bp; mean read length; 0 = off (the paired-end and independent-site samplers apply).
LONG_READ_LENGTH_SIGMA: float = 0.9  # log-scale SD of the lognormal read-length law; ONT ligation libraries are broad.
LONG_READ_MIN_BP: int = 500  # bp; lower clipping bound of the read-length law (shorter reads are filtered upstream).
LONG_READ_SIGMA_LIMIT: float = 4.0  # log-scale SDs above the log-mean; upper read-length clipping bound.
READ_QUALITY_MEAN: float = 20.0  # Phred; mean per-read accuracy (error rate 10**(-Q/10) per base); R10-era ONT reads.
READ_QUALITY_SD: float = 4.0  # Phred; SD of per-read accuracy across reads (the same rate at every base of a read).
READ_QUALITY_MIN: float = 5.0  # Phred; clipping bounds of the per-read accuracy law.
READ_QUALITY_MAX: float = 50.0
SUBSTITUTION_SHARE: float = 0.35  # Share of read errors that are substitutions; the rest are indels (half deletions).
CONTEXT_SITES: float = 0.0  # Share of sites where one allele extends a homopolymer (shared by all samples).
CONTEXT_FACTOR: float = 10.0  # Mean multiplier of the indel error rate on the run-extending allele at a context site
#                               (each site's multiplier is context_factor times a mean-one exponential strength).
CONTEXT_CONVERT: float = 0.5  # Share of context indel errors read as the other allele (the rest give no base).
REFERENCE_BIAS: float = 0.0  # Loss hazard scale: a read's base at a site is lost (local clipping or misalignment
#                              against the reference) with probability 1 - exp(-reference_bias x k**power), k = the
#                              ALT alleles of the read's homologue within the window (the site included).
REFERENCE_BIAS_WINDOW_BP: int = 0  # bp; half-width of the window counting the homologue's ALT alleles; 0 = off.
REFERENCE_BIAS_POWER: float = 1.0  # Exponent of the ALT count: the loss hazard is reference_bias x k**power (clusters
#                                    of differences misalign more than their count suggests).
FOREIGN_READ_SHARE: float = 0.0  # Share of a sample's long reads that come from another sequenced sample (cross-sample
#                                  contamination or barcode misassignment); errors then cluster within those reads.
REFERENCE_PULL: float = 0.0  # Mean probability that a read's ALT base is reported as REF (alignment or calling toward
#                              the reference), scaled per read by its error rate over the mean error rate.

__all__ += [
    "LONG_READ_MEAN_BP", "LONG_READ_LENGTH_SIGMA", "LONG_READ_MIN_BP", "LONG_READ_SIGMA_LIMIT", "READ_QUALITY_MEAN",
    "READ_QUALITY_SD", "READ_QUALITY_MIN", "READ_QUALITY_MAX", "SUBSTITUTION_SHARE", "CONTEXT_SITES", "CONTEXT_FACTOR",
    "CONTEXT_CONVERT", "REFERENCE_BIAS", "REFERENCE_BIAS_WINDOW_BP", "REFERENCE_BIAS_POWER", "REFERENCE_PULL",
    "FOREIGN_READ_SHARE",
]

# Structural rearrangements (simulation/structure.py), off by default; enabling the "cichlid" preset uses these values.
# Sources: large inversions from Blumer et al. 2025 (Science 388:eadr9961): five of 22 Lake Malawi chromosomes carry
# an inversion polymorphic among lineages, each 17-23 Mb and more than half the chromosome; crossovers are suppressed
# in heterokaryotypes. Deletions and duplications: the order of the Lake Malawi pangenome (Quah et al. 2024, bioRxiv
# 2024.03.28.587230): 0.22 structural-variant bubbles per kb across eight assemblies, 13.8% of the reference backbone
# variable among them (the union of what the other haplotypes lack), transposable-element sized (kb); who lacks it
# follows the genealogy. Misorder: the lab's
# linkage breakpoints in a recent long-read assembly (23 shared-switch points in one sire's 11 daughters, about 12
# misplaced segments per Gb). Values not fitted to any cross; carriers come from the source genealogy.
STRUCTURE_PRESET: str = "cichlid"  # Name of the generating structure preset.
STRUCTURE_ENABLED: bool = False  # Whether structural variants are generated.
STRUCTURE_INVERSION_RATE: float = 5 / 22  # Expected large inversions per contig (five of 22 chromosomes).
STRUCTURE_INVERSION_SPAN_LO: float = 0.45  # Large-inversion span as a fraction of the contig, uniform in [lo, hi].
STRUCTURE_INVERSION_SPAN_HI: float = 0.65  # (17-23 Mb of 30-40 Mb chromosomes).
STRUCTURE_SMALL_INVERSION_RATE_PER_MB: float = 0.0  # Small inversions per Mb; off in the preset (weak evidence).
STRUCTURE_SMALL_INVERSION_MEAN_BP: float = 10_000.0  # bp; mean exponential small-inversion length.
STRUCTURE_DELETION_RATE_PER_MB: float = 40.0  # Deletions (reference sequence a haplotype lacks) per Mb.
STRUCTURE_DELETION_MEAN_BP: float = 3_000.0  # bp; mean exponential length: deletion intervals cover about 12% of the
#                                               reference, the pangenome's variable backbone share (13.8%).
STRUCTURE_DUPLICATION_RATE_PER_MB: float = 5.0  # Duplications per Mb (about a quarter of deletions).
STRUCTURE_DUPLICATION_MEAN_BP: float = 2_000.0  # bp; mean exponential length of the duplicated segment.
STRUCTURE_DISPERSED_FRACTION: float = 0.5  # Share of duplications whose copy lies at a uniform position (else tandem).
STRUCTURE_PARALOG_DIVERGENCE: float = 0.02  # Share of a duplicated segment's sites where the copy differs.
STRUCTURE_MISORDER_RATE_PER_MB: float = 0.012  # Reference segments in the wrong place or orientation, per Mb.
STRUCTURE_MISORDER_MEAN_BP: float = 500_000.0  # bp; mean exponential misordered-segment length (assumption).
STRUCTURE_CARRIER_FREQUENCY: float = 0.5  # File sources: probability that a source haplotype carries a variant.

__all__ += [
    "STRUCTURE_PRESET", "STRUCTURE_ENABLED", "STRUCTURE_INVERSION_RATE", "STRUCTURE_INVERSION_SPAN_LO",
    "STRUCTURE_INVERSION_SPAN_HI", "STRUCTURE_SMALL_INVERSION_RATE_PER_MB", "STRUCTURE_SMALL_INVERSION_MEAN_BP",
    "STRUCTURE_DELETION_RATE_PER_MB", "STRUCTURE_DELETION_MEAN_BP", "STRUCTURE_DUPLICATION_RATE_PER_MB",
    "STRUCTURE_DUPLICATION_MEAN_BP", "STRUCTURE_DISPERSED_FRACTION", "STRUCTURE_PARALOG_DIVERGENCE",
    "STRUCTURE_MISORDER_RATE_PER_MB", "STRUCTURE_MISORDER_MEAN_BP", "STRUCTURE_CARRIER_FREQUENCY",
]
