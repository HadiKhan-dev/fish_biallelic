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
# Fail-fast tolerance of the copying-model EM (fidelity.estimate_chain; relative to |log-likelihood|, plus nats): an
# update may lower the log-likelihood of the tempered emissions by at most EM_DECREASE_RELATIVE x |log-likelihood| +
# EM_DECREASE_NATS. The emissions do not depend on (rho, a, b) and neither M-step lowers its part of the expected
# complete-data log-likelihood, so in exact arithmetic the log-likelihood never falls; a larger fall means a
# numerically corrupted E-step, which the fit must not return (4.23: the TroMau label-strength fits fell by 1,288 to
# 4,204 nats per contig, 1.2e-4 to 3.9e-4 relative, and collapsed to a boundary point). Rounding: the falls of 8,700
# recorded fits of the 4.23 runs were at most 2.8e-9 nats (5.6e-16 relative, the last bits of a 4e6-nat sum); the
# relative bound is 1e6 times that and 1e5 times below the smallest corrupted fall.
EM_DECREASE_RELATIVE = 1e-9
# Absolute floor of that tolerance (nats), for a log-likelihood near zero: rounding of a sum of that size is far
# smaller (2.8e-9 nats at 4e6 nats above).
EM_DECREASE_NATS = 1e-6
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
# Positive-share route choice (dimensionless fraction of the run's memory limit; read_inputs.ReadEmissionBuilder.
# compiles): bin actions are compiled while one strength's compiled actions of every sample of a component fit this
# fraction, so fits can retain them and batches keep their CPU-sized concurrency; beyond it the site walk (storage:
# the site tape) is used. Both routes are exact, so the choice changes cost only, never a value beyond rounding.
# Measured on the K ladder (lab423/paintperf): compiled actions cost less CPU than the walk at every label count to
# 31, but their storage grows as X^2 (C + X)^2 per bin: a TroMau-sized component (116 samples, 2,900 bins) needs 24 GB
# at 9 labels, 80 GB at 13, 200 GB at 17 and 1.6 TB at 31, where the compiled plan keeps 17 of 76 CPUs busy in its
# fits and 6 in its decode against the walk's 72 and 41. Half the limit keeps the retained fit beside the decode
# batches; the measured crossover on that unit lies near 21-23 labels.
READ_ACTION_MEMORY_FRACTION = 0.5

# Probability-product floor (dimensionless): fold the product into its log below this value;
# factors below it are logged directly, so bin aggregation never underflows.
EMISSION_RESCALE_BELOW = 1e-150

# Site correlation (dimensionless; divisor 1 + r (reference_snps_per_bin - 1) = 1) of the chain the label-error
# statistic h = (u, v_0, v_1) is read from (fidelity.label_errors; the pedigree's A9, A10 and PD29 allowances). h is
# the posterior probability that a released label is wrong, so it is read from the untempered chain: the copying
# model fitted at r = 0 (fidelity.estimate_chain) and its label-pair masses. The run's two strengths keep
# their roles (the path strength decodes the labels, the confidence strength gates their release). Held-out read
# prediction ignores label identity. It chooses confidence strength and, without the read state, path strength.
# With the read state, the path divisor is fixed at 4. Known-truth simulations (release 4.22): with h at
# the confidence strength, an independent-site design at half depth chose divisor 4 instead of 32 and its label
# posterior sharpened on thinner data (mean u fell x0.3 while the true label error rose x3.8); at r = 0 h rises with
# the true error on every design tested (independent-site, short-read backcross, medium- and long-read designs at
# depth 5 and 2.5), its top tenth holds 5.3 to 9.0 times the mean error, and the independent-site design's
# observed/stated error falls from 8.6 / 84 to 4.1 / 7.9. Divisors 4, 8 and 32 each failed a check on some design.
# h stays a working score (2.7 to 22 times overconfident on those designs), not a calibrated probability.
LABEL_ERROR_CORRELATION = 0.0
# Per-library contamination (4.22 QC A, science/painting/contamination.py). A library is corrected when its
# population-law alpha (read fraction of foreign DNA) is at least CONTAMINATION_ALPHA_FLOOR above the cohort's median
# alpha (the clean level) AND its robust z of alpha over the cohort is at least CONTAMINATION_Z_FLOOR. Both are needed:
# clean libraries carry a misfit floor (alpha 0.004-0.009 in known-truth designs, highest in a systematic subgroup --
# the F1s of TroMau-like crosses -- so a robust z alone flags clean F1s, z up to 6.7), and a fixed floor alone ignores a
# noisier cohort. The floor is on the excess over the median because the clean level moves with the data: under the
# realistic simulator's artefacts a clean half-sib cohort sat at alpha 0.010-0.017 (median 0.015) and a fixed 0.01
# floor let its founder subgroup (0.020, z 7-8) be flagged and corrected (4.22 release validation). 0.008 is the fixed
# 0.01 floor at the clean level of the runs it was validated on (median 0.0014-0.0018): on the saved reports of every
# known-truth run and of TroMau, the excess floor changes no decision except those three founders (TroMau's smallest
# corrected library, alpha 0.0101 with a named donor, keeps its correction; at an excess of 0.01 it would lose it).
# The relative floor was not re-run end to end (owner's ruling, 23 Nov). On five known-truth
# runs (~560 clean, 15 contaminated libraries at 2-30 %) the rule flagged no clean library and 14 of 15 contaminated
# ones (the miss: a nominal 2 % library whose realized share was ~1 %, which changed no pedigree call). The best
# sequenced donor is named when its same-homozygote residual (foreign minor reads where library and donor are the
# same homozygote, per unit of donor alpha) is at most CONTAMINATION_DONOR_RESIDUAL: true donors gave 0.001-0.016 and
# stand-ins for an unsequenced source (founders standing in for an outside fish) 0.079-0.213 on known truth; a
# stand-in's donor law lost edges (96 vs 105 with the population plug-in), so a stand-in is never used. Libraries
# above CONTAMINATION_CARE_ALPHA are corrected and reported "interpret with care" (the owner's line, 3-5 %: 5 %).
CONTAMINATION_ALPHA_FLOOR = 0.008
CONTAMINATION_Z_FLOOR = 4.0
CONTAMINATION_DONOR_RESIDUAL = 0.04
CONTAMINATION_CARE_ALPHA = 0.05


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
        carries nothing (with read_factor's bin-level correction exactly; the bypass is a shortcut to the same result;
        painting/read_inputs.py ReadEmissionBuilder).
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
    foreign_read_share: fraction of called reads assigned to the uniform-donor foreign class (dimensionless).
        The default is zero because the library's foreign share is unknown without an estimate or declaration.
        Foreign occupancy is determined by twice the share times covering depth per homologue.
    share_flank_windows: reference windows of flank on each side of every window the automatic share estimate
        scores (painting/share_estimator.py). Each window is scored given its flanks, which pin its label pair as
        the painting's flanking sequence does; without them a homozygous pair plus a foreign read from the
        sample's own other founder explains a stretch where one homologue has few reads (the 4.23 false positive
        on clean long reads). Not a user setting.
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
    foreign_read_share: float = field(
        default=0.0,
        metadata={
            "role": "state",
            "axes": (),
            "units": "fraction of reads",
            "dtype": "float64",
            "stored": True,
            "provenance": "Declared library contamination fraction. Zero assigns all reads to the sample by default.",
        },
    )
    share_flank_windows: int = field(
        default=2,
        metadata={
            "role": "state",
            "axes": (),
            "units": "reference windows",
            "dtype": "int64",
            "stored": True,
            "provenance": "4.23 share estimate (lab423/share_fp DESIGN.md sections 3-4, saved stage-D inputs): with "
                          "1, 2 or 3 windows per side the clean famhs_corrected arm estimates exactly 0 (against "
                          "4.28e-4 without flanks); the foreign ad_foreign arm (true 1e-2) gives 3.34e-3, 3.59e-3 and "
                          "3.67e-3, rising slowly toward the unflanked 4.39e-3 while the scored sites grow as 2k + 1. "
                          "Two is the smallest width at which the foreign estimate has nearly settled.",
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
        if not 0.0 <= self.foreign_read_share < 1.0:
            raise SettingsError("painting.foreign_read_share must be in [0, 1)")
        if int(self.share_flank_windows) != self.share_flank_windows or self.share_flank_windows < 0:
            raise SettingsError("painting.share_flank_windows must be an integer >= 0")


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
    "READ_ACTION_MEMORY_FRACTION",
]
