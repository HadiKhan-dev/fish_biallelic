"""Site- and sample-quality model controls and numerical solver bounds."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SiteQualityParameters:
    """Self-calibrated site filter r1-r5 (``evidence/quality/site_quality.py``).

    Every removal threshold comes from the contig's own distributions: an empirical null
    (histogram mode and half-maximum widths) for total depth (r1) and per-sample depth
    heterogeneity (r2), clean-core/artefact mixtures for wrong-allele reads in homozygotes (r3)
    and heterozygote allele balance (r4). These fields fix only the genotype-classification read
    model, the posterior levels and the identifiability guard. Rule r0 (not a biallelic SNP) is
    applied by the loader for every input.
    """

    enabled: bool = field(
        default=True,
        metadata={
            "provenance": (
                "Enable contig-calibrated site quality filtering (boolean); the loader still "
                "applies its biallelic-SNP rule."
            ),
        },
    )
    # Read model used only to classify each sample's genotype at a site; the clean-core error
    # rate and allele balance are then fitted, not these.
    classification_error: float = field(
        default=0.01,
        metadata={
            "provenance": (
                "Read error probability for genotype classification (probability); clean-core "
                "error rates are fitted separately."
            ),
        },
    )
    classification_heterozygote_alt: float = field(
        default=0.5,
        metadata={
            "provenance": (
                "Heterozygote ALT fraction for genotype classification (probability); clean-core "
                "balance is fitted separately."
            ),
        },
    )
    # r1/r2 windows: where the fitted core's local posterior is >= this level (histogram
    # smoothed over this many bins).
    core_posterior_level: float = field(
        default=0.5,
        metadata={
            "provenance": (
                "Minimum local core posterior for depth and heterogeneity windows (probability); "
                "defines the retained empirical-null interval."
            ),
        },
    )
    histogram_smoothing_bins: int = field(
        default=5,
        metadata={
            "provenance": (
                "Histogram smoothing width (bins); stabilizes the empirical depth and "
                "heterogeneity core."
            ),
        },
    )
    # r3/r4: a site is flagged when P(clean core) < this level, and only if the artefact class is
    # separated from the clean core by this factor (r3: rate ratio; r4: latent standard deviation or
    # mean offset over max(clean latent standard deviation, binomial standard deviation at the
    # median depth)).
    clean_posterior_level: float = field(
        default=0.5,
        metadata={
            "provenance": (
                "Clean-core posterior threshold for wrong-read and balance mixtures "
                "(probability); lower posteriors flag a site when components are separated."
            ),
        },
    )
    minimum_artefact_separation: float = field(
        default=3.0,
        metadata={
            "provenance": (
                "Minimum artefact separation (dimensionless rate ratio or standardized balance "
                "displacement); avoids labelling unresolved mixtures."
            ),
        },
    )
    # r5 (informativeness, not quality): fewer expected carriers of a non-major genotype than
    # this. It removes sites without segregating information.
    drop_uninformative_sites: bool = field(
        default=True,
        metadata={
            "provenance": (
                "Apply the expected non-major-carrier gate (boolean); removes sites without "
                "segregating information."
            ),
        },
    )
    minimum_expected_carriers: float = field(
        default=2.0,
        metadata={
            "provenance": (
                "Minimum expected non-major-genotype carriers (samples); sets the informativeness "
                "gate."
            ),
        },
    )
    # Contigs with fewer biallelic SNPs than this are not filtered.
    minimum_calibration_sites: int = field(
        default=1000,
        metadata={
            "provenance": (
                "Minimum biallelic SNPs on a contig (sites) before fitting its quality filter; "
                "smaller contigs are not filtered."
            ),
        },
    )


@dataclass(frozen=True)
class SampleQualityParameters:
    """Per-sample, per-contig quality exclusion from founder construction (``evidence/quality/sample_quality.py``).

    Runs only where the site filter is applied (it uses the r1-r4-passing SNPs, r3's clean error rate and r4's clean
    heterozygote balance h0, widened to r4's two component means when r4 records an unresolved fit). Per library, a
    mixture over its eligible cells (no other library carries the minor allele) tests for a foreign-like minor-read
    component above ``minimum_foreign_fraction`` / 2, with a common share within the non-artefact mixture, against the
    library's own errors (any distribution below that amplitude), its own heterozygotes (beta-binomial, the same in
    both orientations) and local artefacts (a free share per region, each with its own fraction); the library is
    flagged on a contig when the statistic reaches the working threshold. A flag is a quality exclusion, not a
    contamination diagnosis: elevated library error, dispersed variable-fraction artefacts and biased private alleles
    can flag too, even in a short tract if other regions have little eligible information. Flagged samples leave
    founder construction on that contig
    (``ContigEvidence.founder_view``) and the r5 carrier count. Heterozygous near-private alleles (inherited segments)
    are reported.
    """

    enabled: bool = field(
        default=True,
        metadata={
            "role": "state", "axes": (), "dtype": "bool", "stored": True, "units": "1",
            "provenance": (
                "Enable per-library quality exclusion from founder construction (boolean); "
                "painting still uses every sample."
            ),
        },
    )
    # q: a library carries the cohort-minor allele when P(K >= k | d, e) <= q / (N - 1) (a site nobody carries
    # shows a carrier anywhere with probability <= q). The same q sets the working threshold of the per-library test:
    # T >= u with P(chi2_1 >= u) = 2 q / (G N) (the boundary law (chi2_0 + chi2_1) / 2 of one foreign fraction, by
    # Bonferroni over the G fractions and the N libraries). That level is nominal only for a regular, correctly
    # specified likelihood of independent windows; the measured clean false-flag rate is what supports it.
    carrier_tail_level: float = field(
        default=0.05,
        metadata={
            "role": "state", "axes": (), "dtype": "float64", "stored": True, "units": "1",
            "provenance": (
                "Carrier-tail level and nominal library-test calibration (probability); controls "
                "cohort-wise carrier calls and the working multiple-test threshold."
            ),
        },
    )
    # alpha_min: twice the working foreign-allele amplitude a_min. The null admits any error distribution of the
    # library up to a_min = alpha_min / 2; a flag needs a component above it. Motivation (not a derived safety
    # boundary): the founder-construction and painting emissions mix each observed site's genotype evidence with a
    # uniform genotype error of mass r = 0.01 (assembly LinkerParameters.robustness, PaintingParameters.robustness_epsilon,
    # the founder refit), which limits the influence of discordant observations. At a site where foreign DNA at read
    # fraction alpha is homozygous for an allele the sample lacks, a cell of depth d shows K ~ Bin(d, alpha) foreign
    # reads; two or more contradict a homozygote decisively (heterozygote likelihood ratio above 3 (1 - r) / r at
    # clean error rates 1e-4..1e-3 and depths of about 4 to 16) with probability ~ C(d, 2) alpha^2, which is of the
    # order of r at alpha = 0.8-1.4% over 11-18x (alpha 1%: 0.72% at 13x; alpha 2%: 2.7%). A foreign allele shows at
    # alpha (foreign homozygote) or alpha/2 (heterozygote), so a = alpha_min / 2 aims to catch 1% contamination of
    # either zygosity; a flag does not establish alpha >= 1%, and sensitivity near the amplitude is measured, not
    # guaranteed. Libraries whose cells are mostly at 1-5x are more fragile; the QC has little power there. Not
    # calibrated on any real library.
    minimum_foreign_fraction: float = field(
        default=0.01,
        metadata={
            "role": "state", "axes": (), "dtype": "float64", "stored": True, "units": "1",
            "provenance": (
                "Working foreign-read fraction (fraction); half this value is the minimum "
                "foreign-allele amplitude tested against own errors, not a calibrated "
                "contamination estimate."
            ),
        },
    )
    # f: foreign alleles are resolved up to a_hi = f h_lo / 2 (h0/4: a foreign heterozygote at alpha/2 with alpha up
    # to h0/2), with h_lo the lower end of the balance interval (h0, or the lower of r4's two component means when r4's
    # labelling is unresolved); grosser contamination is detected at that ceiling, its fraction reported as at least
    # it. Own private heterozygotes are beta-binomial about the balance interval with a latent SD up to 0.1, the same
    # for a private ALT and a private REF allele. Own private alleles shown below about h_lo - 0.2 (reference or
    # haplotype bias) can resemble foreign alleles, even locally if other regions carry little eligible information.
    # The fitted fraction is not a calibrated contamination estimate.
    heterozygote_floor_ratio: float = field(
        default=0.5,
        metadata={
            "role": "state", "axes": (), "dtype": "float64", "stored": True, "units": "1",
            "provenance": (
                "Foreign-allele ceiling multiplier of half the lower clean heterozygote balance "
                "(dimensionless); separates the foreign grid from own heterozygotes."
            ),
        },
    )
    # Own private heterozygotes and errors may be commoner at some depths (mapping bias lowers depth and allele
    # fraction together): their weights are fitted in this many depth classes (the library's eligible-cell depth
    # quantiles), foreign alleles' share in none (foreign reads do not depend on a site's depth).
    heterozygote_depth_classes: int = field(
        default=4,
        metadata={
            "role": "state", "axes": (), "dtype": "int64", "stored": True, "units": "1",
            "provenance": (
                "Eligible-cell depth quantile classes (count); allows own-error and heterozygote "
                "weights to vary with depth."
            ),
        },
    )
    # Smaller cohorts have too little eligible private-allele evidence to separate foreign-like reads
    # from a library's own alleles.
    minimum_samples: int = field(
        default=20,
        metadata={
            "role": "state", "axes": (), "dtype": "int64", "stored": True, "units": "1",
            "provenance": (
                "Minimum libraries (samples) for quality fitting; smaller cohorts have too little "
                "private-allele evidence to distinguish foreign-like reads."
            ),
        },
    )
    # Local artefacts (both hypotheses): the contig's passing sites are cut into this many regions of equal site
    # counts, and in each region a free share of the library's cells may be local artefacts (a collapsed repeat, a
    # mismapped paralog, a private tract with mapping bias), each with a minor fraction of its own (uniform on [0, 1],
    # so k is uniform on 0..d). The foreign share c is common within the non-artefact mixture; its marginal weight
    # is (1 - pi_R) c in region R. No independent regional support is required; local excesses can be absorbed,
    # but need not be. The uniform law and region count are working choices. 1: one contig-wide artefact share.
    artefact_regions: int = field(
        default=16,
        metadata={
            "role": "state", "axes": (), "dtype": "int64", "stored": True, "units": "1",
            "provenance": (
                "Equal-site-count regions per contig (count); allows a separate local-artefact "
                "share in each region under both hypotheses."
            ),
        },
    )
    # Cells of one sample closer than one sequenced fragment can share reads: each eligible cell weighs 1 / (the
    # library's eligible cells in its fragment_bp window, counted from the first passing site), and segment-report
    # cells form one event.
    fragment_bp: int = field(
        default=1000,
        metadata={
            "role": "state", "axes": (), "dtype": "int64", "stored": True, "units": "bp",
            "provenance": (
                "Read-sharing window width (base pairs); eligible cells in a window share one "
                "unit of test weight."
            ),
        },
    )
    # Inherited-segment report only: heterozygous near-private events of one (sample, partner)
    # less than segment_gap_bp apart form a region; regions with fewer events are not listed.
    segment_gap_bp: int = field(
        default=1_000_000,
        metadata={
            "role": "state", "axes": (), "dtype": "int64", "stored": True, "units": "bp",
            "provenance": (
                "Maximum gap between inherited near-private events (base pairs); groups events "
                "for the segment report only."
            ),
        },
    )
    segment_minimum_events: float = field(
        default=3.0,
        metadata={
            "role": "state", "axes": (), "dtype": "float64", "stored": True, "units": "1",
            "provenance": (
                "Minimum weighted events per reported inherited segment (events); suppresses "
                "sparsely supported report regions."
            ),
        },
    )


# Maximum genotype-frequency EM updates per site (iterations).
SITE_SITE_EM_ITERATIONS = 60

# Maximum clean/artefact mixture EM updates (iterations).
SITE_MIXTURE_ITERATIONS = 300

# Histogram range endpoints (percentiles); limits the empirical core fit to the central depth/heterogeneity
# range.
SITE_HISTOGRAM_RANGE_PERCENTILES = (0.5, 99.5)

# Projected-gradient stopping scale of the error-mixture M step (dimensionless).
SITE_M_STEP_GRADIENT = 1e-09

# Relative coordinate-motion stopping scale of the error-mixture M step (dimensionless).
SITE_M_STEP_STEP = 1e-08

# Objective-rounding allowance for M-step backtracking (dimensionless).
SITE_M_STEP_ROUNDING = 1e-13

# Noise scale for M-step convergence (dimensionless).
SITE_M_STEP_NOISE = 1e-11

# Maximum error-mixture M-step Newton updates (iterations).
SITE_M_STEP_ITERATIONS = 100

# Chunks for deterministic M-step objective reductions (chunks).
SITE_M_STEP_CHUNKS = 256

# Minimum argument for the polygamma asymptotic expansion (dimensionless).
SITE_POLYGAMMA_SHIFT = 16.0

# Target cells per site-QC tile (sample-site cells); bounds dense scratch space.
SITE_SITE_TILE_CELLS = 1 << 20

# Maximum sites per site-QC tile (sites); bounds scratch even for small cohorts.
SITE_MAXIMUM_TILE_SITES = 4096

# Error-mixture optimizer bounds on log(rate) and log(shape) (dimensionless transformed coordinates).
SITE_ERROR_LOWER, SITE_ERROR_UPPER = (np.log([1e-12, 0.001]), np.log([10.0, 10000.0]))

# Clean wrong-read rate interval (probability); bounds the concave one-dimensional fit.
SITE_CLEAN_RATE_BOUNDS = (1e-12, 0.5)

# Minimum leakage slope relative to expected homozygote reads (dimensionless); requires information for the
# error-mixture fit.
SITE_LEAKAGE_SLOPE_FLOOR = 1e-09

# Stirling-series coefficients (dimensionless); evaluates beta-binomial gamma-ratio remainders.
SITE_BALANCE_STIRLING = np.array(
    [1 / 12, -1 / 360, 1 / 1260, -1 / 1680, 1 / 1188, -691 / 360360, 1 / 156, -3617 / 122400], dtype=np.float64
)

# Scale of the log1p-transformed balance dispersion (dimensionless).
SITE_BALANCE_T_SCALE = 0.001

# Bounds for balance-mixture logits and transformed dispersions (dimensionless coordinates).
SITE_BALANCE_BOUNDS = (
    (-18.0, 18.0),
    (-18.0, 18.0),
    (0.0, math.log1p(1000000.0 / SITE_BALANCE_T_SCALE)),
    (-18.0, 18.0),
    (0.0, math.log1p(1000000.0 / SITE_BALANCE_T_SCALE)),
)

# Projected-gradient stationarity threshold for balance fitting (dimensionless).
SITE_BALANCE_GRADIENT = 1e-08


# Ratio between own-error grid amplitudes (dimensionless); spaces the nuisance error grid geometrically.
SAMPLE_ERROR_GRID_RATIO = 1.6

# Lower own-error grid amplitude relative to the fitted clean error rate (fraction).
SAMPLE_ERROR_GRID_BOTTOM = 0.1

# Foreign-allele grid size (points); resolves alternatives above the own-error range.
SAMPLE_FOREIGN_GRID = 24

# Private-heterozygote dispersion grid rho (dimensionless); latent SD is sqrt(h (1-h) rho),
# at most 0.1 at h = 1/2. The working nuisance uses the same grid in both orientations.
SAMPLE_HET_DISPERSION_GRID = np.array([0.0, 0.0001, 0.0004, 0.0016, 0.0064, 0.0256, 0.04])

# Weight updates within each sample-mixture M step (iterations).
SAMPLE_M_STEP_ITERATIONS = 4

# Maximum sample-mixture EM updates (iterations).
SAMPLE_EM_ITERATIONS = 400

# Sample-mixture EM gain tolerance relative to the total log likelihood (dimensionless).
SAMPLE_EM_TOLERANCE = 1e-10

# Bisection steps for the mixture-share update (iterations).
SAMPLE_SHARE_BISECTIONS = 48

# Minimum nonzero mixture share (fraction); keeps share updates finite.
SAMPLE_SHARE_FLOOR = 1e-12

# Initial non-reference mixture weight (fraction); starts weight fitting away from zero.
SAMPLE_WEIGHT_START = 0.001

# Sites per sparse sample-quality chunk (sites); bounds extraction workspace.
SAMPLE_CHUNK_SITES = 1024

# Radix of packed sample-quality cell keys (integer codes); separates depth and allele-count coordinates.
SAMPLE_CODE = 1 << 20

# Target entries in sample-quality reductions (entries); bounds reduction workspace.
SAMPLE_REDUCE_ENTRIES = 1 << 22

# Minimum foreign amplitude range relative to its lower bound (dimensionless); requires a resolvable foreign
# grid.
SAMPLE_MINIMUM_FOREIGN_RANGE = 1.25

# Own-error amplitude separation factor (dimensionless); spaces the nuisance and foreign grids.
SAMPLE_ERROR_SEPARATION = 2.0

# Balance-fit reason strings identifying unresolved labelling (text); widen the private-heterozygote balance
# interval.
SAMPLE_AMBIGUOUS_BALANCE = ("unresolved class labelling", "no start of the balance fit")


# Central-difference step for the balance-polish Hessian (coordinate units).
BALANCE_POLISH_DIFFERENCE_STEP = 0.0001

# Relative objective improvement threshold for balance polish (dimensionless).
BALANCE_POLISH_RELATIVE_STOP = 1e-11

# Relative bracket/step tolerance of the clean-rate fit (dimensionless).
CLEAN_RATE_RELATIVE_STEP = 1e-12

# Minimum site mean depth in heterogeneity scaling (reads per sample); avoids a zero denominator.
DEPTH_MEAN_FLOOR = 1e-09

# Maximum posterior-clean change for mixture convergence (probability).
ERROR_MIXTURE_RELATIVE_STOP = 1e-05

# Minimum negative-binomial shape (dimensionless); bounds artefact dispersion.
ERROR_MIXTURE_SHAPE_FLOOR = 0.001

# Minimum clean/artefact mixture share (probability); keeps both mixture logs finite.
ERROR_MIXTURE_WEIGHT_FLOOR = 1e-06

# Minimum fitted histogram-core width (histogram coordinate units); avoids zero-width scaling.
HISTOGRAM_WIDTH_FLOOR = 1e-12

# Positive probability/weight floor (dimensionless); keeps logarithms and denominators finite.
POSITIVE_WEIGHT_FLOOR = 1e-300

# Distance of regional artefact shares below one (probability); keeps non-artefact mass finite.
REGION_SHARE_CEILING_GAP = 1e-09

# Minimum genotype frequency in site-classification EM (probability).
SITE_EM_FREQUENCY_FLOOR = 1e-06

# Per-site genotype EM log-likelihood gain threshold (nats); stops negligible updates.
SITE_EM_LOG_GAIN_TOLERANCE = 1e-07

# Minimum Newton damping for sample-mixture weights (dimensionless).
WEIGHT_DAMPING_FLOOR = 1e-10

# Minimum accepted sample-weight objective gain (objective units).
WEIGHT_GAIN_TOLERANCE = 1e-12

__all__ = ["SiteQualityParameters", "SampleQualityParameters"]


# Damped Newton weight-solver ceiling, growth and shrink factors (dimensionless).
WEIGHT_DAMPING_CEILING = 1e12
WEIGHT_DAMPING_GROWTH = 10.0
WEIGHT_DAMPING_SHRINK = 0.01
# Balance-fit stationarity controls (iterations, backtracking trials and restarts) and L-BFGS-B options.
BALANCE_POLISH_ITERATIONS = 8
BALANCE_BACKTRACK_TRIALS = 8
BALANCE_RESTART_LIMIT = 3
BALANCE_FIT_OPTIONS = dict(ftol=2e-15, gtol=1e-9, maxiter=1000, maxls=40, maxcor=20)
BALANCE_RESTART_OPTIONS = dict(ftol=0.0, gtol=1e-10, maxiter=500, maxls=60, maxcor=10)
BALANCE_START_LOGIT_BOUND = 18.0
# (mixing weight, first mean offset, first dispersion, second offset, second dispersion).
# Preserve the native sorted set after clipping each mean around the empirical mean.
BALANCE_STARTS = (
    (0.9, 0.0, 0.0, -0.02, 0.01),
    (0.8, 0.0, 0.001, -0.03, 0.02),
    (0.6, 0.0, 0.003, -0.05, 0.05),
    (0.95, 0.0, 0.0, 0.02, 0.03),
    (0.5, -0.03, 0.01, 0.02, 0.001),
)
# Gamma-ratio remainder-series switch radius (dimensionless).
BALANCE_SERIES_RADIUS = 0.01
# Initial genotype EM objective sentinel (nats); any finite supported likelihood improves it.
SITE_EM_INITIAL_LOG_LIKELIHOOD = -1e300

# Histogram bin-count bounds, posterior log-odds clamp, Newton backtracking count and error-mixture initial
# values (counts, nats and probabilities).
HISTOGRAM_MINIMUM_BINS = 50
HISTOGRAM_MAXIMUM_BINS = 2000
POSTERIOR_LOG_ODDS_BOUND = 700
ERROR_BACKTRACK_TRIALS = 40
ERROR_INITIAL_CLEAN_WEIGHT = 0.6
ERROR_INITIAL_RATE_FLOOR = 1e-5
ERROR_INITIAL_RATE_CEILING = 0.1
ERROR_INITIAL_RATE_RATIO = 10
ERROR_MIXTURE_SHAPE_CEILING = 10000.0
