"""Scientific settings of the painting stage (label continuity, then component painting)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...core.ties import noise_quantum

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

# Probability-product floor (dimensionless): fold the product into its log below this value;
# factors below it are logged directly, so bin aggregation never underflows.
EMISSION_RESCALE_BELOW = 1e-150


@dataclass(frozen=True)
class PaintingParameters:
    """Model constants of the painting HMM; defaults are the production values.

    The copying process between bins (painting/transitions.py, painting/fidelity.py) uses the contig's genetic map
    and three parameters estimated per component from its samples (painting.fidelity.estimate_chain): the
    ancestry-change rate rho and the panel-error chain's entry and exit probabilities a and b. The settings below
    are their starting values and the estimation's fixed schedule, not the values used.

    emission_site_correlation: r; each sample's bin log emission is divided by 1 + (n - 1) r, n the sample's
        informative sites in the bin (painting.emissions.design_effect): a tuned, equicorrelation-inspired
        magnitude adjustment. The bin emission multiplies its per-site terms as if they were independent given the
        copying pair, which overstates the evidence of a bin (panel errors run along rows, reads and
        genotype-likelihood errors are correlated between nearby sites). Under an equal-variance, common-correlation
        working assumption a bin of 100 informative sites counts as about 9 variance-equivalent sites and one of 8 as
        about 4.7; the weight tends to 1 as n -> 1. r is a tuned constant chosen on the simulations and sparse-read
        controls (docs/methods/painting.md), not an estimated correlation.
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
    robustness_epsilon: uniform genotype-error mixture of the emission,
        P = (1 - eps) P(GL | pair) + eps / 3 per observed site.
    snps_per_bin: target selected sites per bin (bins split the selected sites
        into max(1, L // snps_per_bin) near-equal consecutive runs).
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
    """

    emission_site_correlation: float = field(
        default=0.1,
        metadata={
            "provenance": (
                "Working within-bin correlation r (dimensionless); divides log emissions by "
                "1+(n-1)r to temper correlated evidence, tuned on simulations and sparse-read "
                "controls."
            ),
            "role": "state",
            "axes": (),
            "units": "1",
            "dtype": "float64",
            "stored": True,
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
    robustness_epsilon: float = field(
        default=0.01,
        metadata={
            "provenance": (
                "Uniform genotype-error mixture mass per observed site (probability); bounds the "
                "influence of discordant genotype evidence."
            ),
            "role": "state",
            "axes": (),
            "units": "probability",
            "dtype": "float64",
            "stored": True,
        },
    )
    snps_per_bin: int = field(
        default=100,
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
    emission_log_floor: float = field(
        default=-50.0,
        metadata={
            "provenance": (
                "Lower bound of a site log emission (nats); limits a discordant observation "
                "before bin aggregation."
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


__all__ = [
    "PaintingParameters",
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
]
