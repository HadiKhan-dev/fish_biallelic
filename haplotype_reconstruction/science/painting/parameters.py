"""Scientific settings of the painting stage (label continuity, then component painting)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...core.ties import noise_quantum
from ...core.errors import SettingsError

# Copying-rate bounds (per Morgan), entry/exit probability bounds and fixed search step counts.
# A reached estimate bound is reported, not rejected: entry at 1e-6 indicates no support
# for error runs under this model and estimation schedule, not a test of their absence.
RATE_BOUNDS = (1e-2, 1e3)
ENTRY_BOUNDS = (1e-6, 0.5)
EXIT_BOUNDS = (1e-3, 0.999)
RATE_GRID_POINTS = 241
GOLDEN_STEPS = 60
# Smallest switch probability of a bin edge. Two bin centres inside a cold interval of the map (equal cumulative
# cM) are 0 Morgans apart; they keep a negligible chance to switch instead of none, so every transition weight
# stays finite (Viterbi, posteriors and pedigree's chains use the logs). Such edges carry no information about rho
# (painting.fidelity.rate_update leaves them out).
MINIMUM_SWITCH_PROBABILITY = 1e-15
# Normalized-GL spread threshold (probability) and Viterbi predecessor tie quantum (nats).
INFORMATIVE_SPREAD = 16.0 * np.finfo(np.float64).eps
VITERBI_TIE = noise_quantum(1.9e-6)
# int64 interval masks allow 63 labels; enforced at the decode boundary.
MAXIMUM_MASK_STATES = 63
# Dimension-only cache/window budgets (bytes); their values do not select scientific candidates.
# 8 GiB, so the largest chromosomes' estimation and label-continuity emissions stay resident (chr3 at
# N = 320: packed estimation emissions 1.0 GiB, label-continuity part emissions about 2.8 GB); a reported working
# set, not a reservation (the memory governor acts on measured pressure).
EMISSION_CACHE_MAX_BYTES = 8 << 30
WINDOW_BYTES = 64 << 20
# Read-state cache budget (bytes; dimension-only, PaintingParameters.read_classes > 1): the label-independent parts of
# the read state (painting/read_inputs.py) kept across a component's emission builds (copying-model fits, label
# continuity, decode, reference painting). Values never depend on it: a build reuses the arrays it would compute.
# 16 GiB holds every sample's parts of the largest contig of the 8-contig paired-end designs (80 samples x 329,000
# selected sites x 400 bytes, 10.5 GB); larger cohorts keep the batches that fit and recompute the rest.
READ_STATE_CACHE_BYTES = 16 << 30
# Read-state batch budget (bytes; dimension-only): one sample batch's read-state working set (read matrices at both
# strengths, label-independent parts, their transient inputs over the evidence span). Batches shrink below the
# CPU-sized batch only where that would exceed it, e.g. near 30 founders over thousands of bins (496 label pairs x
# 5,000 bins x 5 classes: 0.27 GB of read matrices per sample and strength). Results do not depend on the batch.
READ_STATE_BATCH_BYTES = 16 << 30
# Read-state table cache (bytes; dimension-only): the depth and class tables of the read state's (survival bin, depth
# value) pairs (science/shared/reads.py build_tables), kept across a contig's sample batches; about 0.5 MB per pair
# at the deepest tables. A table is a function of its key alone, so values never depend on it.
READ_TABLE_CACHE_BYTES = 1 << 30

# Probability-product floor (dimensionless): fold the product into its log below this value;
# factors below it are logged directly, so bin aggregation never underflows.
EMISSION_RESCALE_BELOW = 1e-150


@dataclass(frozen=True)
class ReadStateParameters:
    """Model constants of the painting's read state (science/shared/reads.py, painting/read_factor.py,
    painting/read_chain.py), used when ``PaintingParameters.read_classes`` > 1.

    Per sample and site of the evidence grid the read state holds the covering depth N = n + o (called depth n, loss
    offset o) and the class of the covering split between the two homologues; consecutive sites share reads through
    the library's read-survival law phi(d), fitted to the evidence stage's read-sharing curve.

    offsets: minimum number of retained loss offsets starting at zero (reads lost to deletions or third-base calls).
        Also the working width at zero fitted loss, where all the mass is at offset zero.
    loss_pilot_tail, loss_pilot_extra_offsets, loss_pilot_minimum_offsets, loss_pilot_maximum_offsets: upper-tail
        probability of the stationary Poisson lost-depth quantile, extra offsets for depth jumps, and the bounds
        on positive-loss pilot width. The 13–33 offset range was checked against a wider pilot on saved and
        high-loss fixtures. The floor exceeds the nine offsets needed by fitted-library depth-pair probes.
    loss_span_tail: maximum discarded probability mass within the pilot depth path law, by allocating an equal
        tail allowance to every evidence site in the span (union bound).
    survival_step: ratio of the log grid of 1 - phi on which the per-gap depth and class tables are shared.
    survival_floor: smallest tabulated 1 - phi.
    depth_step: per-homologue covering-depth bin width on which samples share tables (reads).
    minimum_arrival_depth: smallest per-homologue covering depth of the read-arrival law (reads).
    length_grid_low_bp, length_grid_high_bp, length_grid_points: the log-spaced read-length grid of the survival fit.
    minimum_sharing_bins: measured lag bins the survival fit needs.
    sharing_detection_alpha: the run's family-wise error of read-sharing detection. A fit detects sharing when some
        usable lag bin's excess exceeds z* jackknife standard errors, z* = Phi^-1(1 - alpha / M) (Bonferroni), M every
        one-sided test the run's detection can make: per analysed contig the lag bins holding site pairs (usable in
        the all-sample estimate or not), once for the all-sample fit and four times for each of at most
        2 ** library_levels - 1 candidate library splits (the parts' difference in both directions, each part's
        fit), a bound fixed before any test is read; library fits and the library-split test use the same z*, and
        the calibration records z*, M, alpha, the route, each library's largest excess / SE with its signed margin,
        winning bin and usable bins, and each candidate split's contrast and margin (a run-log warning when a
        library's statistic or a split's deciding contrast is within a factor of two of z*). Else a library
        shows no read sharing (independent sites) and its samples bypass the read state (one class: the emission
        without read state exactly); when no library of the run shows sharing, the whole run takes read_classes = 1
        (PaintingParameters.read_classes). Why family-wise: a fixed per-bin z of 2 over ~41 bins flags a contig
        without sharing about 60 % of the time (std, simulated without sharing: 3 of 8 contigs), and the split
        tests must be counted (std chr18: a noise split at z 4.14 passes z* = 4.0 of the all-sample bins alone).
    detection_z: z*, resolved per run by the calibration (stages/painting/read_state.py run_detection) and carried on
        the paintings' parameters; None in the configuration. Without sharing the read state
        carries nothing, while the tempered class factor R ** w (w < 1) would penalise heterozygous label pairs
        (sum_s pi_s R_s ** w < 1, about 8 nats per bin; painting/read_inputs.py ReadEmissionBuilder).
    maximum_loss: largest intermittent loss (probability) the model uses.
    loss_contrast_gap_bp, loss_quartet_span_bp, loss_cluster_bp: short-gap geometry and spatial block size for the
        repeated-depth moment check of a covariance fit at zero loss. The check uses the run's read-sharing multiplier
        as a conservative consistency criterion. Its contrasts are outside the read-sharing count of tests.
    deep_tail: P(a homologue holds <= minor of N covering reads) below which the covering depth N is 'deep' (split
        class 'balanced' with certainty to this precision); bounds the tables and the per-site work at any depth.
    table_minimum_depth, table_depth_margin: the tables cover covering depths up to max(table_minimum_depth, the
        deepest called site + offsets + table_depth_margin), capped at the deep bound + offsets + table_depth_margin.
    library_*: sequencing libraries are found by dividing the samples (two-means on their depth-correlation curves
        at library_lags_bp, seeded on library_seed_lag_bp, at most library_iterations rounds, library_levels
        divisions); a split is kept only when both parts have library_minimum_samples samples, their read-sharing
        estimates differ by more than z* standard errors (sharing_detection_alpha) at some lag bin AND their fitted
        survival laws differ by more than library_survival_tolerance somewhere on library_gaps_bp.
    fallback_error: per-read other-allele rate when the evidence stage applied its fixed binomial read model (the
        evidence stage records no fitted rate then).
    """

    offsets: int = field(
        default=3,
        metadata={
            "role": "state",
            "axes": (),
            "units": "count",
            "dtype": "int64",
            "stored": True,
            "provenance": "Minimum number of retained consecutive loss offsets starting at zero; keeps the "
                          "three-offset base support even when the depth posterior allows trimming further.",
        },
    )
    loss_pilot_tail: float = field(
        default=1e-9,
        metadata={
            "role": "state", "axes": (), "units": "probability", "dtype": "float64", "stored": True,
            "provenance": "Upper tail for the stationary Poisson lost-depth quantile at the library's largest "
                          "mean covering depth. A small tail makes it a starting point for wide support.",
        },
    )
    loss_pilot_extra_offsets: int = field(
        default=5,
        metadata={
            "role": "state", "axes": (), "units": "count", "dtype": "int64", "stored": True,
            "provenance": "Pilot width added to the stationary quantile to allow for posterior shifts at "
                          "called-depth jumps.",
        },
    )
    loss_pilot_minimum_offsets: int = field(
        default=13,
        metadata={
            "role": "state", "axes": (), "units": "count", "dtype": "int64", "stored": True,
            "provenance": "Positive-loss pilot floor. Fitted-library physical depth-pair tails were negligible "
                          "at nine offsets, so thirteen leave room for conditioning on longer sequences.",
        },
    )
    loss_pilot_maximum_offsets: int = field(
        default=33,
        metadata={
            "role": "state", "axes": (), "units": "count", "dtype": "int64", "stored": True,
            "provenance": "Positive-loss pilot ceiling to bound work; the trial checked finite pilot widths "
                          "against wider support on high-loss and deep-island fixtures.",
        },
    )
    loss_span_tail: float = field(
        default=1e-4,
        metadata={
            "role": "state", "axes": (), "units": "probability", "dtype": "float64", "stored": True,
            "provenance": "Maximum pilot depth-path mass discarded across the span, shared equally over "
                          "its evidence sites. Allows pruning while bounding the total variation within the pilot.",
        },
    )
    survival_step: float = field(
        default=1.02,
        metadata={
            "role": "state",
            "axes": (),
            "units": "ratio",
            "dtype": "float64",
            "stored": True,
            "provenance": "Log-grid ratio of 1 - phi for table reuse (2 %): a linear grid rounds short-gap survival "
                          "to one.",
        },
    )
    survival_floor: float = field(
        default=1e-7,
        metadata={
            "role": "state",
            "axes": (),
            "units": "probability",
            "dtype": "float64",
            "stored": True,
            "provenance": "Smallest tabulated read turnover 1 - phi.",
        },
    )
    loss_contrast_gap_bp: int = field(
        default=5,
        metadata={
            "role": "state",
            "axes": (),
            "units": "bp",
            "dtype": "int64",
            "stored": True,
            "provenance": "Maximum adjacent spacing for extrapolating the called-depth variance to zero gap.",
        },
    )
    loss_quartet_span_bp: int = field(
        default=10,
        metadata={
            "role": "state",
            "axes": (),
            "units": "bp",
            "dtype": "int64",
            "stored": True,
            "provenance": "Maximum span of four consecutive distinct markers in the repeated-depth check.",
        },
    )
    loss_cluster_bp: int = field(
        default=1_000_000,
        metadata={
            "role": "state",
            "axes": (),
            "units": "bp",
            "dtype": "int64",
            "stored": True,
            "provenance": "Physical block width for the spatial delete-block uncertainty of the loss contrasts.",
        },
    )
    depth_step: float = field(
        default=0.25,
        metadata={
            "role": "state",
            "axes": (),
            "units": "reads",
            "dtype": "float64",
            "stored": True,
            "provenance": "Per-homologue covering-depth bins for table reuse.",
        },
    )
    minimum_arrival_depth: float = field(
        default=0.05,
        metadata={
            "role": "state",
            "axes": (),
            "units": "reads",
            "dtype": "float64",
            "stored": True,
            "provenance": "Floor of the per-homologue covering depth in the read-arrival law: a sample with almost "
                          "no reads still receives arrivals.",
        },
    )
    length_grid_low_bp: float = field(
        default=50.0,
        metadata={
            "role": "state",
            "axes": (),
            "units": "bp",
            "dtype": "float64",
            "stored": True,
            "provenance": "Shortest read length of the survival fit's log-spaced grid.",
        },
    )
    length_grid_high_bp: float = field(
        default=2e6,
        metadata={
            "role": "state",
            "axes": (),
            "units": "bp",
            "dtype": "float64",
            "stored": True,
            "provenance": "Longest read length of the survival fit's log-spaced grid.",
        },
    )
    length_grid_points: int = field(
        default=60,
        metadata={
            "role": "state",
            "axes": (),
            "units": "count",
            "dtype": "int64",
            "stored": True,
            "provenance": "Read lengths of the survival fit's log-spaced grid.",
        },
    )
    minimum_sharing_bins: int = field(
        default=3,
        metadata={
            "role": "state",
            "axes": (),
            "units": "count",
            "dtype": "int64",
            "stored": True,
            "provenance": "Measured read-sharing lag bins the survival fit needs.",
        },
    )
    sharing_detection_alpha: float = field(
        default=0.01,
        metadata={
            "role": "state",
            "axes": (),
            "units": "probability",
            "dtype": "float64",
            "stored": True,
            "provenance": "Owner 18 Nov: family-wise error 1 % over the run's read-sharing tests (Bonferroni; "
                          "sharing_detection_alpha docstring); the per-bin z 2 it replaces flagged 3 of std's 8 "
                          "contigs without sharing (max z 2.08-2.25) and a noise library split (split z 4.14, "
                          "parts' max z 4.64 and 5.19: counting only the all-sample bins, z* 4.0, accepts it; "
                          "counting the split tests, z* 4.58, does not); every sharing design measured has max z "
                          "14-414 (lab421/lr/cost/diag, diag2).",
        },
    )
    detection_z: float | None = field(
        default=None,
        metadata={
            "role": "state",
            "axes": (),
            "units": "standard errors",
            "dtype": "float64 | None",
            "stored": True,
            "provenance": "The run's family-wise detection threshold z*, resolved by the calibration from every "
                          "analysed contig's read-sharing record and carried on the paintings' parameters (None in "
                          "the configuration).",
        },
    )
    maximum_loss: float = field(
        default=0.5,
        metadata={
            "role": "state",
            "axes": (),
            "units": "probability",
            "dtype": "float64",
            "stored": True,
            "provenance": "Largest intermittent loss the model uses. A covariance fit above one starts at zero "
                          "loss and may be corrected by the nearby-depth check.",
        },
    )
    deep_tail: float = field(
        default=1e-12,
        metadata={
            "role": "state",
            "axes": (),
            "units": "probability",
            "dtype": "float64",
            "stored": True,
            "provenance": "Tail probability of the deep covering depth (42, 47, 52 reads at minor 0, 1, 2): at and "
                          "above it the split class is 'balanced', so tables and per-site work stay fixed at any "
                          "depth.",
        },
    )
    table_minimum_depth: int = field(
        default=24,
        metadata={
            "role": "state",
            "axes": (),
            "units": "reads",
            "dtype": "int64",
            "stored": True,
            "provenance": "Smallest covering-depth range of the tables.",
        },
    )
    table_depth_margin: int = field(
        default=2,
        metadata={
            "role": "state",
            "axes": (),
            "units": "reads",
            "dtype": "int64",
            "stored": True,
            "provenance": "Covering depths tabulated beyond the deepest called site and its offsets.",
        },
    )
    library_lags_bp: tuple[int, ...] = field(
        default=(25, 100, 300, 1000, 3000, 10000, 30000),
        metadata={
            "role": "state",
            "axes": ("lag",),
            "units": "bp",
            "dtype": "int64",
            "stored": True,
            "provenance": "Lags of the samples' depth-correlation curves compared by the library grouping.",
        },
    )
    library_gaps_bp: tuple[float, ...] = field(
        default=(100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0),
        metadata={
            "role": "state",
            "axes": ("gap",),
            "units": "bp",
            "dtype": "float64",
            "stored": True,
            "provenance": "Gaps at which two candidate libraries' fitted survival laws are compared.",
        },
    )
    library_seed_lag_bp: int = field(
        default=1000,
        metadata={
            "role": "state",
            "axes": (),
            "units": "bp",
            "dtype": "int64",
            "stored": True,
            "provenance": "Lag that seeds the two-means split when the contig is long enough for it.",
        },
    )
    library_iterations: int = field(
        default=50,
        metadata={
            "role": "state",
            "axes": (),
            "units": "count",
            "dtype": "int64",
            "stored": True,
            "provenance": "Largest number of two-means rounds of a candidate split.",
        },
    )
    library_minimum_samples: int = field(
        default=3,
        metadata={
            "role": "state",
            "axes": (),
            "units": "count",
            "dtype": "int64",
            "stored": True,
            "provenance": "Samples per library for its own read-sharing estimate.",
        },
    )
    library_survival_tolerance: float = field(
        default=0.2,
        metadata={
            "role": "state",
            "axes": (),
            "units": "probability",
            "dtype": "float64",
            "stored": True,
            "provenance": "Survival difference that changes decisions (lab421: +-0.10 at 30 kb changed nothing, "
                          "-0.22 by 0.005 percentage points); within-library differences measured <= 0.08, genuine "
                          "libraries 0.95-0.97.",
        },
    )
    library_levels: int = field(
        default=2,
        metadata={
            "role": "state",
            "axes": (),
            "units": "count",
            "dtype": "int64",
            "stored": True,
            "provenance": "Divisive levels of the library grouping (at most four libraries).",
        },
    )
    fallback_error: float = field(
        default=0.02,
        metadata={
            "role": "state",
            "axes": (),
            "units": "probability",
            "dtype": "float64",
            "stored": True,
            "provenance": "Per-read other-allele rate when the evidence stage applied its fixed binomial read model "
                          "(the evidence stage's ReadModelParameters.fixed_error; no fitted rate is recorded then).",
        },
    )

    def __post_init__(self) -> None:
        if self.offsets < 1 or self.length_grid_points < 2 or self.library_minimum_samples < 1:
            raise SettingsError("read-state counts must be positive")
        if self.loss_contrast_gap_bp < 2 or self.loss_quartet_span_bp < 3 or self.loss_cluster_bp <= 0:
            raise SettingsError("read-loss checks require two distinct positive gaps, four sites and positive blocks")
        if not 1 <= self.loss_pilot_minimum_offsets <= self.loss_pilot_maximum_offsets or self.loss_pilot_extra_offsets < 0:
            raise SettingsError("invalid loss-offset pilot widths")
        if not 0.0 < self.loss_pilot_tail < 1.0 or not 0.0 < self.loss_span_tail < 1.0:
            raise SettingsError("loss-offset tails must be probabilities in (0, 1)")
        if not self.survival_step > 1.0 or not 0.0 <= self.maximum_loss < 1.0 or not 0.0 < self.deep_tail < 1.0:
            raise SettingsError("invalid read-state survival grid, loss bound or deep tail")
        if self.library_seed_lag_bp not in self.library_lags_bp:
            raise SettingsError("the library seed lag must be one of the library lags")
        if not 0.0 < self.sharing_detection_alpha < 1.0:
            raise SettingsError("sharing_detection_alpha must be a probability in (0, 1)")


@dataclass(frozen=True)
class PaintingParameters:
    """Model constants of the painting HMM; defaults are the production values.

    The copying process between bins (painting/transitions.py, painting/fidelity.py) uses the contig's genetic map
    and three parameters estimated per component from its samples (painting.fidelity.estimate_chain): the
    ancestry-change rate rho and the panel-error chain's entry and exit probabilities a and b. The settings below
    are their starting values and the estimation's fixed schedule, not the values used.

    divisor_grid: candidate full-reference-window evidence divisors. Two strengths are chosen from it per run, from
        held-out reference windows pooled over the run (painting/calibration.py): the PATH divisor maximises the
        held-out score of the decoded label pairs and tempers the fit, the decoding and the pedigree's reading of
        the painting; the CONFIDENCE divisor maximises the held-out posterior predictive density and tempers the
        release posteriors and family phase. The correlation of a divisor is
        (divisor - 1) / (reference_snps_per_bin - 1).
    calibration_folds: number of held-out folds; fold f hides the f-th of calibration_folds contiguous near-equal
        parts of every reference window's selected sites, so every window is predicted from the rest of its own data.
    initial_switch_rate: starting value of rho, ancestry changes per Morgan per track (about two meioses since the
        founders, an F2 homologue).
    initial_error_entry: starting value of a, the per-edge probability that a faithful track of an anchored class
        enters a panel-error run while keeping its label.
    initial_error_exit: starting value of b, the per-edge probability that a panel-error run ends while its label
        stays (1 / b bins: the mean length of a run that keeps its label).
    estimation_iterations: EM updates of (rho, a, b), a fixed number from the fixed starting values (a truncated,
        deterministic estimator, not a convergence test; see docs/methods/painting.md for its sensitivity).
    estimate_grid: the estimates are rounded to this step in natural-log space (2^-6: about 1.6%), so
        genotype-likelihood noise far below it changes them only at a grid boundary.
    robustness_grid: candidate genotype outlier rates: a cell's genotype is redrawn from the site's cohort genotype
        frequencies at rate eps (science.shared.observation). Fit every component at each value; select the run-wide
        largest sum of last-trajectory log likelihoods, with ties to the smaller value, and publish the whole profile,
        its boundary status and its likelihood set (4.21: half-decades from 1e-6, the numerical lower bound that keeps
        a discordant cell's cost finite, to 3e-2; ADVICE2 section 4).
    snps_per_bin: target selected sites per fine bin; split each near-equal reference window
        into reference_snps_per_bin // snps_per_bin near-equal consecutive runs.
    reference_snps_per_bin: selected-site reference window; snps_per_bin must divide it.
        Consecutive groups of reference_snps_per_bin // snps_per_bin bins share one divisor.
    emission_log_floor: lower bound of each site's log emission (nats).
    minimum_release_posterior: tau of the release rule; a track pair is released when the posterior probability
        that both tracks copy its unordered class pair faithfully (no panel error) is >= tau, or (stage 2) that of
        its local allele-equivalence pair, if directly read-supported. Predeclared 10% posterior-error budget.
    label_continuity_minimum_switch_gain: a row-suffix relabelling must lower
        the painted switch total by at least this many switches.
    label_continuity_maximum_loser_ratio: samples whose painted switches rise
        may be at most this fraction of the samples whose switches fall.
    label_continuity_check_budget: maximum charged candidate checks per search iteration, including tie peers.
        An undecided class withholds its repair and flags LabelContinuity.check_budget_reached.
    read_classes: read-state classes per sample (1 = no read state; 3, 5, 7, 9 = 2 minor + 3 for minor 0..3;
        science/shared/reads.py): the painting's emissions carry each sample's reads across neighbouring sites. When
        no library of the run shows read sharing the read model's exact limit is one class (nothing is carried
        between sites, while the tempered class factors would still penalise heterozygous label pairs), so the run
        takes read_classes = 1 throughout, calibration, paintings and their consumers included; the calibration
        records the route (read_state: 'on', 'off: no library shares reads'; stages/painting/read_state.py).
    read_state: the read state's model constants (ReadStateParameters).
    path_divisor: with the read state (read_classes > 1), the path strength's fixed full-reference-window divisor
        (painting/read_calibration.py path_divisor); without it the path divisor is calibrated from divisor_grid.
    confidence_grid: with the read state, the candidate confidence divisors, chosen per run by the read-aware held-out
        score and the paired one-standard-error rule (painting/read_calibration.py); divisor_grid serves the model
        without read state.
    calibration_morgans: with the read state, the working-map Morgans the read-aware confidence fits run on: the
        analysed contigs in configured order, whole, accumulated until their length (the working map between each
        contig's first and last evidence position) reaches this budget; every analysed contig when the run has less.
        The outlier-rate calibration and the route without read state use every analysed contig.
    """

    read_classes: int = field(
        default=5,
        metadata={
            "role": "state",
            "axes": (),
            "units": "classes",
            "dtype": "int64",
            "stored": True,
            "provenance": "Owner 17 Nov 2026: user setting 1/3/5/7/9, default 5 (= 7 = 9 in every criterion measured).",
        },
    )
    read_state: ReadStateParameters = field(
        default_factory=ReadStateParameters,
        metadata={
            "role": "state",
            "axes": (),
            "units": "1",
            "dtype": "ReadStateParameters",
            "stored": True,
            "provenance": "Read-state parameter group; offsets, survival grid, deep bound and library grouping are "
                          "declared by its fields.",
        },
    )
    path_divisor: float = field(
        default=4.0,
        metadata={
            "role": "state",
            "axes": (),
            "units": "1",
            "dtype": "float64",
            "stored": True,
            "provenance": "Read-state path strength, fixed: on the four truth designs path divisor 4 decoded the "
                          "fewest unmatched switch events at equal misses (seed 58 vs 94 at divisor 1, famhs 52 vs "
                          "79, bc8 17 vs 41, tl 26 vs 58); the within-window path score failed as a selector.",
        },
    )
    confidence_grid: tuple[float, ...] = field(
        default=(1.0, 2.0, 4.0, 8.0, 16.0),
        metadata={
            "role": "state",
            "axes": ("grid",),
            "units": "1",
            "dtype": "float64",
            "stored": True,
            "provenance": "Read-state confidence divisor candidates (4.21, from 1-64): the read-aware calibration "
                          "chose 1-2 on every simulated design (bc8 1, tl 1, k30 1, seed 2, famhs 2) and 8 on TroMau "
                          "(whole genome best 8; best 16 on chr19 alone); 32 and 64 were never the best or the "
                          "choice, and leaving them out leaves every recorded choice unchanged at 5 of 7 fits per "
                          "fold.",
        },
    )
    calibration_morgans: float = field(
        default=10.0,
        metadata={
            "role": "state",
            "axes": (),
            "units": "Morgans",
            "dtype": "float64",
            "stored": True,
            "provenance": "Read-aware confidence calibration material (4.21): on the stored per-contig fits, the "
                          "contigs in configured order reaching 10 Morgans choose what every contig chose on TroMau "
                          "(42.5 M; 8 at every prefix), bc8 (14.7 M; 1 at every prefix) and seed (14.7 M; 2 at 11.2 M "
                          "and at 14.7 M, 1 at 12.8 M and below 11 M, both good on truth); TroMau fits 5 of its 22 "
                          "contigs.",
        },
    )
    divisor_grid: tuple[float, ...] = field(
        default=(1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0),
        metadata={
            "role": "state",
            "axes": ("grid",),
            "units": "1",
            "dtype": "float64",
            "stored": True,
            "provenance": "Full-reference evidence divisors; choose the pooled held-out posterior predictive maximum.",
        },
    )
    calibration_folds: int = field(
        default=3,
        metadata={
            "role": "state",
            "axes": (),
            "units": "folds",
            "dtype": "int64",
            "stored": True,
            "provenance": "Hide every kth selected-site reference window in one of k deterministic folds.",
        },
    )
    initial_switch_rate: float = field(
        default=2.0,
        metadata={
            "provenance": (
                "Initial copying-change rate per track (per Morgan); starts component estimation "
                "at two ancestry changes per Morgan."
            ),
            "role": "state",
            "axes": (),
            "units": "Morgan^-1",
            "dtype": "float64",
            "stored": True,
        },
    )
    initial_error_entry: float = field(
        default=0.001,
        metadata={
            "provenance": (
                "Initial faithful-to-panel-error entry probability per edge (probability); starts "
                "component chain estimation."
            ),
            "role": "state",
            "axes": (),
            "units": "probability",
            "dtype": "float64",
            "stored": True,
        },
    )
    initial_error_exit: float = field(
        default=0.25,
        metadata={
            "provenance": (
                "Initial panel-error exit probability per edge (probability); starts the mean "
                "label-preserving error-run length at four bins."
            ),
            "role": "state",
            "axes": (),
            "units": "probability",
            "dtype": "float64",
            "stored": True,
        },
    )
    estimation_iterations: int = field(
        default=8,
        metadata={
            "provenance": (
                "Fixed EM updates of the copying rate and error chain (iterations); defines a "
                "deterministic truncated estimator."
            ),
            "role": "state",
            "axes": (),
            "units": "count",
            "dtype": "int64",
            "stored": True,
        },
    )
    estimate_grid: float = field(
        default=2.0 ** (-6),
        metadata={
            "provenance": (
                "Rounding step for estimated parameters (natural-log units); limits sensitivity "
                "to genotype-likelihood noise below the grid scale."
            ),
            "role": "state",
            "axes": (),
            "units": "nats",
            "dtype": "float64",
            "stored": True,
        },
    )
    robustness_grid: tuple[float, ...] = field(
        default=(1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2),
        metadata={
            "provenance": (
                "Candidate genotype outlier rates (probabilities), half-decades from the numerical bound 1e-6; "
                "selected across the run by the sum of the components' last EM-trajectory log likelihoods, ties "
                "to the smaller rate."
            ),
            "role": "state",
            "axes": ("grid",),
            "units": "probability",
            "dtype": "float64",
            "stored": True,
        },
    )
    snps_per_bin: int = field(
        default=25,
        metadata={
            "provenance": (
                "Target selected sites per emission bin (sites); sets near-equal consecutive bins "
                "for the copying chain."
            ),
            "role": "state",
            "axes": (),
            "units": "sites",
            "dtype": "int64",
            "stored": True,
        },
    )
    reference_snps_per_bin: int = field(
        default=100,
        metadata={
            "provenance": "Selected-site reference window for evidence tempering and pedigree observation bins.",
            "role": "state",
            "axes": (),
            "units": "sites",
            "dtype": "int64",
            "stored": True,
        },
    )
    emission_log_floor: float = field(
        default=-50.0,
        metadata={
            "provenance": (
                "Lower bound of a site log emission (nats); limits a discordant observation before bin aggregation."
            ),
            "role": "state",
            "axes": (),
            "units": "nats",
            "dtype": "float64",
            "stored": True,
        },
    )
    minimum_release_posterior: float = field(
        default=0.9,
        metadata={
            "provenance": (
                "Minimum faithful-copy pair posterior (probability); applies the predeclared "
                "ten-percent posterior-error budget."
            ),
            "role": "state",
            "axes": (),
            "units": "probability",
            "dtype": "float64",
            "stored": True,
        },
    )
    label_continuity_minimum_switch_gain: float = field(
        default=2.0,
        metadata={
            "provenance": (
                "Minimum reduction in painted switches for a suffix relabelling (switches); also "
                "sets the score-screen admission level in switch units."
            ),
            "role": "state",
            "axes": (),
            "units": "count",
            "dtype": "float64",
            "stored": True,
        },
    )
    label_continuity_maximum_loser_ratio: float = field(
        default=0.1,
        metadata={
            "provenance": (
                "Maximum switch-loser to switch-winner sample ratio (dimensionless); protects "
                "shared recombinant lineages during row relabelling."
            ),
            "role": "state",
            "axes": (),
            "units": "share",
            "dtype": "float64",
            "stored": True,
        },
    )
    label_continuity_check_budget: int = field(
        default=64,
        metadata={
            "provenance": (
                "Maximum charged candidate checks per search iteration, including tie peers (candidates). "
                "An unresolved class withholds its repair and ends the search with check_budget_reached; "
                "capability rejections need no decode or charge. Bounds trial work at large founder counts."
            ),
            "role": "state",
            "axes": (),
            "units": "count",
            "dtype": "int64",
            "stored": True,
        },
    )

    viterbi_tie: float = field(
        default=VITERBI_TIE,
        metadata={
            "provenance": (
                "Viterbi predecessor tie quantum (nats); noise_quantum(1.9e-6) places the "
                "preference for fewer track changes above the measured score-gap noise scale."
            ),
            "role": "state",
            "axes": (),
            "units": "nats",
            "dtype": "float64",
            "stored": True,
        },
    )

    def __post_init__(self) -> None:
        if not self.divisor_grid or any(not np.isfinite(x) or x < 1 for x in self.divisor_grid):
            raise SettingsError("divisor_grid must contain finite divisors >= 1")
        if not self.confidence_grid or any(not np.isfinite(x) or x < 1 for x in self.confidence_grid):
            raise SettingsError("confidence_grid must contain finite divisors >= 1")
        if not np.isfinite(self.path_divisor) or self.path_divisor < 1:
            raise SettingsError("path_divisor must be a finite divisor >= 1")
        if np.isnan(self.calibration_morgans) or self.calibration_morgans <= 0:
            raise SettingsError("calibration_morgans must be a positive number of Morgans")
        if (
            not np.isfinite(self.calibration_folds)
            or int(self.calibration_folds) != self.calibration_folds
            or self.calibration_folds < 2
        ):
            raise SettingsError("calibration_folds must be an integer >= 2")
        if self.reference_snps_per_bin <= 1:
            raise SettingsError("reference_snps_per_bin must exceed 1")
        if not self.robustness_grid or any(not np.isfinite(x) or not 0 <= x <= 1 for x in self.robustness_grid):
            raise SettingsError("robustness_grid must contain finite probabilities in [0, 1]")
        if self.estimation_iterations < 1:
            raise SettingsError("robustness estimation requires at least one EM update")
        if self.snps_per_bin <= 0 or self.reference_snps_per_bin <= 0:
            raise SettingsError("painting bin sizes must be positive")
        if self.reference_snps_per_bin % self.snps_per_bin:
            raise SettingsError("snps_per_bin must divide reference_snps_per_bin")
        if self.read_classes not in (1, 3, 5, 7, 9):
            raise SettingsError("read_classes must be 1, 3, 5, 7 or 9")


__all__ = [
    "PaintingParameters",
    "ReadStateParameters",
    "EMISSION_RESCALE_BELOW",
    "RATE_BOUNDS",
    "ENTRY_BOUNDS",
    "EXIT_BOUNDS",
    "RATE_GRID_POINTS",
    "GOLDEN_STEPS",
    "MINIMUM_SWITCH_PROBABILITY",
    "INFORMATIVE_SPREAD",
    "VITERBI_TIE",
    "MAXIMUM_MASK_STATES",
    "EMISSION_CACHE_MAX_BYTES",
    "WINDOW_BYTES",
    "READ_STATE_CACHE_BYTES",
    "READ_STATE_BATCH_BYTES",
    "READ_TABLE_CACHE_BYTES",
]
