"""Declare unchanged evidence parameters with their 3.5 provenance (EC01-24)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SiteQualityParameters:
    """Self-calibrated site filter r1-r5 (``evidence/site_quality.py``).

    Every removal threshold comes from the contig's own distributions: an empirical null
    (histogram mode and half-maximum widths) for total depth (r1) and per-sample depth
    heterogeneity (r2), clean-core/artefact mixtures for wrong-allele reads in homozygotes (r3)
    and heterozygote allele balance (r4). These fields fix only the genotype-classification read
    model, the posterior levels and the identifiability guard. Rule r0 (not a biallelic SNP) is
    applied by the loader for every input.
    """

    enabled: bool = field(
        default=True, metadata={"provenance": "B/evidence/config.py: enabled; P0 evidence-core.parameters.csv"}
    )
    # Read model used only to classify each sample's genotype at a site; the clean-core error
    # rate and allele balance are then fitted, not these.
    classification_error: float = field(
        default=0.01,
        metadata={"provenance": "B/evidence/config.py: classification_error; P0 evidence-core.parameters.csv"},
    )
    classification_heterozygote_alt: float = field(
        default=0.5,
        metadata={
            "provenance": "B/evidence/config.py: classification_heterozygote_alt; P0 evidence-core.parameters.csv"
        },
    )
    # r1/r2 windows: where the fitted core's local posterior is >= this level (histogram
    # smoothed over this many bins).
    core_posterior_level: float = field(
        default=0.5,
        metadata={"provenance": "B/evidence/config.py: core_posterior_level; P0 evidence-core.parameters.csv"},
    )
    histogram_smoothing_bins: int = field(
        default=5,
        metadata={"provenance": "B/evidence/config.py: histogram_smoothing_bins; P0 evidence-core.parameters.csv"},
    )
    # r3/r4: a site is flagged when P(clean core) < this level, and only if the artefact class is
    # separated from the clean core by this factor (r3: rate ratio; r4: latent standard deviation or
    # mean offset over max(clean latent standard deviation, binomial standard deviation at the
    # median depth)).
    clean_posterior_level: float = field(
        default=0.5,
        metadata={"provenance": "B/evidence/config.py: clean_posterior_level; P0 evidence-core.parameters.csv"},
    )
    minimum_artefact_separation: float = field(
        default=3.0,
        metadata={"provenance": "B/evidence/config.py: minimum_artefact_separation; P0 evidence-core.parameters.csv"},
    )
    # r5 (informativeness, not quality): fewer expected carriers of a non-major genotype than
    # this. It removes sites without segregating information.
    drop_uninformative_sites: bool = field(
        default=True,
        metadata={"provenance": "B/evidence/config.py: drop_uninformative_sites; P0 evidence-core.parameters.csv"},
    )
    minimum_expected_carriers: float = field(
        default=2.0,
        metadata={"provenance": "B/evidence/config.py: minimum_expected_carriers; P0 evidence-core.parameters.csv"},
    )
    # Contigs with fewer biallelic SNPs than this are not filtered.
    minimum_calibration_sites: int = field(
        default=1000,
        metadata={"provenance": "B/evidence/config.py: minimum_calibration_sites; P0 evidence-core.parameters.csv"},
    )


@dataclass(frozen=True)
class SampleQualityParameters:
    """Per-sample, per-contig quality exclusion from founder construction (``evidence/sample_quality.py``).

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
        default=True, metadata={"provenance": "B/evidence/config.py: enabled; P0 evidence-core.parameters.csv"}
    )
    # q: a library carries the cohort-minor allele when P(K >= k | d, e) <= q / (N - 1) (a site nobody carries
    # shows a carrier anywhere with probability <= q). The same q sets the working threshold of the per-library test:
    # T >= u with P(chi2_1 >= u) = 2 q / (G N) (the boundary law (chi2_0 + chi2_1) / 2 of one foreign fraction, by
    # Bonferroni over the G fractions and the N libraries). That level is nominal only for a regular, correctly
    # specified likelihood of independent windows; the measured clean false-flag rate is what supports it.
    carrier_tail_level: float = field(
        default=0.05,
        metadata={"provenance": "B/evidence/config.py: carrier_tail_level; P0 evidence-core.parameters.csv"},
    )
    # alpha_min: twice the working foreign-allele amplitude a_min. The null admits any error distribution of the
    # library up to a_min = alpha_min / 2; a flag needs a component above it. Motivation (not a derived safety
    # boundary): the founder-construction and painting emissions mix each observed site's genotype evidence with a
    # uniform genotype error of mass r = 0.01 (assembly LinkerConfig.robustness, PaintingConfig.robustness_epsilon,
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
        metadata={"provenance": "B/evidence/config.py: minimum_foreign_fraction; P0 evidence-core.parameters.csv"},
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
        metadata={"provenance": "B/evidence/config.py: heterozygote_floor_ratio; P0 evidence-core.parameters.csv"},
    )
    # Own private heterozygotes and errors may be commoner at some depths (mapping bias lowers depth and allele
    # fraction together): their weights are fitted in this many depth classes (the library's eligible-cell depth
    # quantiles), foreign alleles' share in none (foreign reads do not depend on a site's depth).
    heterozygote_depth_classes: int = field(
        default=4,
        metadata={"provenance": "B/evidence/config.py: heterozygote_depth_classes; P0 evidence-core.parameters.csv"},
    )
    # Kept from release 2.16: smaller cohorts flag nothing (with few libraries most alleles absent from every other
    # library are the library's own, and the eligible set is too small to separate foreign alleles from them).
    minimum_samples: int = field(
        default=20, metadata={"provenance": "B/evidence/config.py: minimum_samples; P0 evidence-core.parameters.csv"}
    )
    # Local artefacts (both hypotheses): the contig's passing sites are cut into this many regions of equal site
    # counts, and in each region a free share of the library's cells may be local artefacts (a collapsed repeat, a
    # mismapped paralog, a private tract with mapping bias), each with a minor fraction of its own (uniform on [0, 1],
    # so k is uniform on 0..d). The foreign share c is common within the non-artefact mixture; its marginal weight
    # is (1 - pi_R) c in region R. No independent regional support is required; local excesses can be absorbed,
    # but need not be. The uniform law and region count are working choices. 1: one contig-wide artefact share.
    artefact_regions: int = field(
        default=16, metadata={"provenance": "B/evidence/config.py: artefact_regions; P0 evidence-core.parameters.csv"}
    )
    # Cells of one sample closer than one sequenced fragment can share reads: each eligible cell weighs 1 / (the
    # library's eligible cells in its fragment_bp window, counted from the first passing site), and segment-report
    # cells form one event.
    fragment_bp: int = field(
        default=1000, metadata={"provenance": "B/evidence/config.py: fragment_bp; P0 evidence-core.parameters.csv"}
    )
    # Inherited-segment report only: heterozygous near-private events of one (sample, partner)
    # less than segment_gap_bp apart form a region; regions with fewer events are not listed.
    segment_gap_bp: int = field(
        default=1_000_000,
        metadata={"provenance": "B/evidence/config.py: segment_gap_bp; P0 evidence-core.parameters.csv"},
    )
    segment_minimum_events: float = field(
        default=3.0,
        metadata={"provenance": "B/evidence/config.py: segment_minimum_events; P0 evidence-core.parameters.csv"},
    )


# Provenance: B/evidence/site_quality.py; P0 evidence-core.parameters.csv.
SITE_SITE_EM_ITERATIONS = 60

# Provenance: B/evidence/site_quality.py; P0 evidence-core.parameters.csv.
SITE_MIXTURE_ITERATIONS = 300

# Provenance: B/evidence/site_quality.py; P0 evidence-core.parameters.csv.
SITE_HISTOGRAM_RANGE_PERCENTILES = (0.5, 99.5)

# Provenance: B/evidence/site_quality.py; P0 evidence-core.parameters.csv.
SITE_M_STEP_GRADIENT = 1e-09

# Provenance: B/evidence/site_quality.py; P0 evidence-core.parameters.csv.
SITE_M_STEP_STEP = 1e-08

# Provenance: B/evidence/site_quality.py; P0 evidence-core.parameters.csv.
SITE_M_STEP_ROUNDING = 1e-13

# Provenance: B/evidence/site_quality.py; P0 evidence-core.parameters.csv.
SITE_M_STEP_NOISE = 1e-11

# Provenance: B/evidence/site_quality.py; P0 evidence-core.parameters.csv.
SITE_M_STEP_ITERATIONS = 100

# Provenance: B/evidence/site_quality.py; P0 evidence-core.parameters.csv.
SITE_M_STEP_CHUNKS = 256

# Provenance: B/evidence/site_quality.py; P0 evidence-core.parameters.csv.
SITE_POLYGAMMA_SHIFT = 16.0

# Provenance: B/evidence/site_quality.py; P0 evidence-core.parameters.csv.
SITE_SITE_TILE_CELLS = 1 << 20

# Provenance: B/evidence/site_quality.py; P0 evidence-core.parameters.csv.
SITE_MAXIMUM_TILE_SITES = 4096

# Provenance: B/evidence/site_quality.py; P0 evidence-core.parameters.csv.
SITE_MINIMUM_CHUNK_SITES = 16384

# Provenance: B/evidence/site_quality.py; P0 evidence-core.parameters.csv.
SITE_ERROR_LOWER, SITE_ERROR_UPPER = (np.log([1e-12, 0.001]), np.log([10.0, 10000.0]))

# Provenance: B/evidence/site_quality.py; P0 evidence-core.parameters.csv.
SITE_CLEAN_RATE_BOUNDS = (1e-12, 0.5)

# Provenance: B/evidence/site_quality.py; P0 evidence-core.parameters.csv.
SITE_LEAKAGE_SLOPE_FLOOR = 1e-09

# Provenance: B/evidence/site_quality.py; P0 evidence-core.parameters.csv.
SITE_BALANCE_STIRLING = np.array(
    [1 / 12, -1 / 360, 1 / 1260, -1 / 1680, 1 / 1188, -691 / 360360, 1 / 156, -3617 / 122400], dtype=np.float64
)

# Provenance: B/evidence/site_quality.py; P0 evidence-core.parameters.csv.
SITE_BALANCE_T_SCALE = 0.001

# Provenance: B/evidence/site_quality.py; P0 evidence-core.parameters.csv.
SITE_BALANCE_BOUNDS = (
    (-18.0, 18.0),
    (-18.0, 18.0),
    (0.0, math.log1p(1000000.0 / SITE_BALANCE_T_SCALE)),
    (-18.0, 18.0),
    (0.0, math.log1p(1000000.0 / SITE_BALANCE_T_SCALE)),
)

# Provenance: B/evidence/site_quality.py; P0 evidence-core.parameters.csv.
SITE_BALANCE_GRADIENT = 1e-08


# Provenance: B/evidence/sample_quality.py; P0 evidence-core.parameters.csv.
SAMPLE_ERROR_GRID_RATIO = 1.6

# Provenance: B/evidence/sample_quality.py; P0 evidence-core.parameters.csv.
SAMPLE_ERROR_GRID_BOTTOM = 0.1

# Provenance: B/evidence/sample_quality.py; P0 evidence-core.parameters.csv.
SAMPLE_FOREIGN_GRID = 24

# Provenance: B/evidence/sample_quality.py; P0 evidence-core.parameters.csv.
SAMPLE_HET_DISPERSION_GRID = np.array([0.0, 0.0001, 0.0004, 0.0016, 0.0064, 0.0256, 0.04])

# Provenance: B/evidence/sample_quality.py; P0 evidence-core.parameters.csv.
SAMPLE_M_STEP_ITERATIONS = 4

# Provenance: B/evidence/sample_quality.py; P0 evidence-core.parameters.csv.
SAMPLE_EM_ITERATIONS = 400

# Provenance: B/evidence/sample_quality.py; P0 evidence-core.parameters.csv.
SAMPLE_EM_TOLERANCE = 1e-10

# Provenance: B/evidence/sample_quality.py; P0 evidence-core.parameters.csv.
SAMPLE_SHARE_BISECTIONS = 48

# Provenance: B/evidence/sample_quality.py; P0 evidence-core.parameters.csv.
SAMPLE_SHARE_FLOOR = 1e-12

# Provenance: B/evidence/sample_quality.py; P0 evidence-core.parameters.csv.
SAMPLE_WEIGHT_START = 0.001

# Provenance: B/evidence/sample_quality.py; P0 evidence-core.parameters.csv.
SAMPLE_CHUNK_SITES = 1024

# Provenance: B/evidence/sample_quality.py; P0 evidence-core.parameters.csv.
SAMPLE_CODE = 1 << 20

# Provenance: B/evidence/sample_quality.py; P0 evidence-core.parameters.csv.
SAMPLE_REDUCE_ENTRIES = 1 << 22

# Provenance: B/evidence/sample_quality.py; P0 evidence-core.parameters.csv.
SAMPLE_MINIMUM_FOREIGN_RANGE = 1.25

# Provenance: B/evidence/sample_quality.py; P0 evidence-core.parameters.csv.
SAMPLE_ERROR_SEPARATION = 2.0

# Provenance: B/evidence/sample_quality.py; P0 evidence-core.parameters.csv.
SAMPLE_AMBIGUOUS_BALANCE = ("unresolved class labelling", "no start of the balance fit")


# Provenance: B/evidence/quality; unchanged inline numerical setting (EC04-22).
BALANCE_POLISH_DIFFERENCE_STEP = 0.0001

# Provenance: B/evidence/quality; unchanged inline numerical setting (EC04-22).
BALANCE_POLISH_RELATIVE_STOP = 1e-11

# Provenance: B/evidence/quality; unchanged inline numerical setting (EC04-22).
CLEAN_RATE_RELATIVE_STEP = 1e-12

# Provenance: B/evidence/quality; unchanged inline numerical setting (EC04-22).
DEPTH_MEAN_FLOOR = 1e-09

# Provenance: B/evidence/quality; unchanged inline numerical setting (EC04-22).
ERROR_MIXTURE_RELATIVE_STOP = 1e-05

# Provenance: B/evidence/quality; unchanged inline numerical setting (EC04-22).
ERROR_MIXTURE_SHAPE_FLOOR = 0.001

# Provenance: B/evidence/quality; unchanged inline numerical setting (EC04-22).
ERROR_MIXTURE_WEIGHT_FLOOR = 1e-06

# Provenance: B/evidence/quality; unchanged inline numerical setting (EC04-22).
HISTOGRAM_WIDTH_FLOOR = 1e-12

# Provenance: B/evidence/quality; unchanged inline numerical setting (EC04-22).
POSITIVE_WEIGHT_FLOOR = 1e-300

# Provenance: B/evidence/quality; unchanged inline numerical setting (EC04-22).
REGION_SHARE_CEILING_GAP = 1e-09

# Provenance: B/evidence/quality; unchanged inline numerical setting (EC04-22).
SITE_EM_FREQUENCY_FLOOR = 1e-06

# Provenance: B/evidence/quality; unchanged inline numerical setting (EC04-22).
SITE_EM_LOG_GAIN_TOLERANCE = 1e-07

# Provenance: B/evidence/quality; unchanged inline numerical setting (EC04-22).
WEIGHT_DAMPING_FLOOR = 1e-10

# Provenance: B/evidence/quality; unchanged inline numerical setting (EC04-22).
WEIGHT_GAIN_TOLERANCE = 1e-12

__all__ = ["SiteQualityParameters", "SampleQualityParameters"]


# Provenance: B/evidence/sample_quality.py:539-587, damped Newton weight solver.
WEIGHT_DAMPING_CEILING = 1e12
WEIGHT_DAMPING_GROWTH = 10.0
WEIGHT_DAMPING_SHRINK = 0.01
# Provenance: B/evidence/site_quality.py:822-920, r4 stationarity and optimizer budgets.
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
# Provenance: B/evidence/site_quality.py: balance gamma-ratio remainder series switch.
BALANCE_SERIES_RADIUS = 0.01
# Provenance: B/evidence/site_quality.py: genotype EM initial log likelihood.
SITE_EM_INITIAL_LOG_LIKELIHOOD = -1e300

# Provenance: B/evidence/site_quality.py:_peak_core, _posterior_clean, _newton_maximum, _error_mixture.
HISTOGRAM_MINIMUM_BINS = 50
HISTOGRAM_MAXIMUM_BINS = 2000
POSTERIOR_LOG_ODDS_BOUND = 700
ERROR_BACKTRACK_TRIALS = 40
ERROR_INITIAL_CLEAN_WEIGHT = 0.6
ERROR_INITIAL_RATE_FLOOR = 1e-5
ERROR_INITIAL_RATE_CEILING = 0.1
ERROR_INITIAL_RATE_RATIO = 10
ERROR_MIXTURE_SHAPE_CEILING = 10000.0
