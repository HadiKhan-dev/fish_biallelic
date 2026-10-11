"""Shared scientific constants; stage-specific settings live beside their models."""
from __future__ import annotations

from dataclasses import dataclass


DEFAULT_READ_ERROR_PROBABILITY = 0.02


@dataclass(frozen=True)
class ReadCalibrationConfig:
    """Numerical/sampling settings for nested observed-AD model fitting."""

    marker_stride: int = 8
    maximum_fit_depth: int = 48
    fold_window_bp: int = 1_000_000
    initial_parameters: tuple = ((DEFAULT_READ_ERROR_PROBABILITY, 0.5),
                                (0.01, 0.35), (0.08, 0.65))
    maximum_iterations: int = 2000
    likelihood_tolerance_per_observation: float = 1e-10


@dataclass(frozen=True)
class SiteQualityConfig:
    """Per-contig, self-calibrated site filter on loaded allele depths.

    Applied before read calibration and discovery (core/site_quality.py). Every
    removal threshold comes from the contig's own distributions: an empirical
    null (histogram mode and half-maximum width) for total depth and
    per-sample depth heterogeneity, and clean-core/artefact mixtures for
    wrong-allele reads in homozygotes and heterozygote allele balance. These
    fields fix only the genotype-classification read model, the posterior
    levels and the identifiability guard.
    """

    enabled: bool = True
    # Read model used only to classify each sample's genotype at a site; the
    # clean-core error rate and allele balance are then fitted, not these.
    classification_error: float = 0.01
    classification_heterozygote_alt: float = 0.5
    # r1/r2 windows: where the fitted core's local posterior is >= this level
    # (histogram smoothed over this many bins).
    core_posterior_level: float = 0.5
    histogram_smoothing_bins: int = 5
    # r3/r4: a site is flagged when P(clean core) < this level, and only if
    # the artefact class is separated from the clean core by this factor
    # (rate ratio, or binomial standard deviations at the median depth).
    clean_posterior_level: float = 0.5
    minimum_artefact_separation: float = 3.0
    # r5 (informativeness, not quality): fewer expected carriers of a
    # non-major genotype than this. It removes sites without segregating
    # information, including founder-monomorphic simulated sites.
    drop_uninformative_sites: bool = True
    minimum_expected_carriers: float = 2.0
    # Contigs with fewer biallelic SNPs than this are not filtered.
    minimum_calibration_sites: int = 1000


# The simulated route keeps every site. On the stress control, per-sample
# dropout tracts make r1/r2 remove benign sites and 4% read errors (above the
# 1% classification error) make r4 separate; r5 removes founder-monomorphic
# sites. Evaluation compares phase with truth on the full simulated site axis.
SIMULATION_SITE_QUALITY = SiteQualityConfig(enabled=False)


@dataclass(frozen=True)
class SampleQualityConfig:
    """Per-sample, per-contig contamination QC inside the site filter (core/sample_quality.py).

    Statistic: at biallelic SNPs passing the site filter's quality rules r0-r4,
    a sample's private cells (no other sample carries the cohort-minor allele)
    whose minor reads fit a low fraction (half the contig's heterozygote
    balance) better than a heterozygous allele, counted as fragment events per
    observed cell on the square-root scale. A sample is flagged on a contig
    when its robust z (cohort median and 1.4826 x MAD on that contig, at least
    the statistic's counting noise) exceeds flag_robust_z; no labels,
    pedigree, benchmark or absolute rate enter.
    Heterozygous near-private alleles (inherited segments) never flag; they
    are reported as regions. On that contig, flagged samples' reads leave
    founder construction (discovery, feedback, assembly, founder refinement)
    and the r5 carrier count; they are still painted, phased and scored with
    their full evidence. Runs only where the site filter is applied.
    """

    enabled: bool = True
    # Extreme-outlier bar on the variance-stabilised scale: flagging removes a
    # library from founder evidence (real TroMau, 22 contigs: samples without
    # a contamination signature reach at most z 4.0).
    flag_robust_z: float = 5.0
    # The median/MAD spread needs a cohort; smaller cohorts flag nothing.
    minimum_samples: int = 20
    # Cells of one sample closer than one sequenced fragment (paired-end
    # inserts are a few hundred bp) can share reads and count as one event.
    fragment_bp: int = 1000
    # Inherited-segment report only (no decision): heterozygous near-private
    # events of one (sample, partner) less than segment_gap_bp apart form a
    # region (inside TroMau segments 99% of gaps are < 0.81 Mb); regions with
    # fewer events are not listed (on TroMau's 22 contigs, samples without a
    # segment never chain more than two such events).
    segment_gap_bp: int = 1_000_000
    segment_minimum_events: float = 3.0


# Harness simulations have no foreign reads; the route keeps every library.
SIMULATION_SAMPLE_QUALITY = SampleQualityConfig(enabled=False)


@dataclass(frozen=True)
class PathSelectionConfig:
    """Local-panel search shared by initial and feedback fits.

    Rounds stop when the best refit does not improve and fixed-K fits at a
    conditional fixed point. The limits (rounds: base + per_founder x K of the
    current panel; fixed-K updates: base + per_founder x K + per_site x L,
    since a slow fit can build a row one or two sites per update) are reported
    backstops, "round_backstop" and "update_backstop", not budgets (T19
    replays: at most 242 updates needed at L = 200, K = 6, within 20 + 4K + 2L
    = 444). refits_per_kind is the refit generator: the top proposals per kind
    (drop, merge, add) that are refitted.
    """

    refits_per_kind: int = 8
    round_backstop_base: int = 10
    round_backstop_per_founder: int = 2
    update_backstop_base: int = 20
    update_backstop_per_founder: int = 4
    update_backstop_per_site: int = 2


# Shared founder-allele site posterior (assembly/site_posterior.py): block
# feedback and the assembly completion. At most MAX_EXACT_BITS free founder
# bits are enumerated exactly (<= 1,024 configurations). Above that, founders
# whose single and pair flips at the posterior mode all cost more than
# SITE_CERTAIN_NATS stay at the mode; the remaining core is enumerated exactly
# when it has at most MAX_EXACT_BITS founders, otherwise over the largest
# Hamming ball with at most SITE_BALL_CONFIGURATIONS configurations, but of
# radius at least SITE_BALL_MINIMUM_RADIUS. In that ball a release is withheld
# when the marginal is within SITE_SHELL_GUARD x (outermost-shell weight that
# flips the founder) of its release threshold. The mode comes from loopy
# belief propagation (damping, tolerance, iteration limit) and single/pair
# flip ascent; the ascent and ball-recentring limits are reported backstops.
MAX_EXACT_BITS = 10
SITE_CERTAIN_NATS = 20.0
SITE_BALL_CONFIGURATIONS = 1024
SITE_BALL_MINIMUM_RADIUS = 3
SITE_SHELL_GUARD = 3.0
SITE_BP_DAMPING = 0.5
SITE_BP_TOLERANCE = 1e-9
SITE_BP_ITERATIONS = 500
SITE_ASCENT_BACKSTOP = 50
SITE_RECENTRE_BACKSTOP = 50


# Shared linker's block-local quality CTMC, separate from read-error rates.
# Stationary unreliable-sequence fraction and mean error-tract length in bp.
LINKER_ERROR_FRACTION = 0.01
LINKER_ERROR_TRACT_BP = 20_000.0


_VITERBI_BIC_ENABLED = True


VITERBI_SWITCH_PENALTY = 10.0


VITERBI_SNPS_PER_BIN = 10


FIXED_K_FIT_MAX_THREADS = 16


DEFAULT_SOFT_SEED_MIN_CLUSTER_SIZE = 3


CANDIDATE_DEDUP_HAMMING_PERCENT = 0.5
