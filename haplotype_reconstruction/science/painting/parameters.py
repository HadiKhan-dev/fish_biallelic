"""Scientific settings of the painting stage (label continuity, then component painting)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...core.ties import noise_quantum

# Production bounds and fixed search schedule, 3.5 painting/fidelity.py:49–51,700,706 (P0 PR08).
RATE_BOUNDS = (1e-2, 1e3)
ENTRY_BOUNDS = (1e-6, 0.5)
EXIT_BOUNDS = (1e-3, 0.999)
RATE_GRID_POINTS = 241
GOLDEN_STEPS = 60
# 3.5 painting/transitions.py:40; finite cold-map transition logs (P0 PR05).
MINIMUM_SWITCH_PROBABILITY = 1e-15
# 3.5 painting/hmm.py:21,47; observed normalized-GL spread and incumbent preference (PR04/PR09).
INFORMATIVE_SPREAD = 16.0 * np.finfo(np.float64).eps
VITERBI_TIE = noise_quantum(1.9e-6)
# int64 interval-mask representation, 3.5 painting/hmm.py:50 (PR11); enforced at decode boundary.
MAXIMUM_MASK_STATES = 63
# Dimension-only storage/execution budgets, WP_PA2 §7; these never enter scientific decisions.
# Release 4.7: 8 GiB, so the largest chromosomes' estimation and label-continuity emissions stay resident (chr3 at
# N = 320: packed estimation emissions 1.0 GiB, label-continuity part emissions about 2.8 GB); a reported working
# set, not a reservation (the memory governor acts on measured pressure).
EMISSION_CACHE_MAX_BYTES = 8 << 30
WINDOW_BYTES = 64 << 20
LABEL_CONTINUITY_BASE_BYTES = 64 << 20


@dataclass(frozen=True)
class PaintingParameters:
    """Model constants of the painting HMM; defaults are the production values.

    The copying process between bins (painting/transitions.py, painting/fidelity.py) uses the contig's genetic map
    and three parameters estimated per component from its samples (painting.fidelity.estimate_chain): the
    ancestry-change rate rho and the panel-error chain's entry and exit probabilities a and b. The settings below
    are their starting values and the estimation's fixed schedule, not the values used.

    emission_site_correlation: r; each sample's bin log emission is divided by 1 + (n - 1) r, n the sample's
        informative sites in the bin (release 2.14; painting.hmm.design_effect): a tuned, equicorrelation-inspired
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
    label_continuity_check_budget: candidates re-decoded per search iteration at most (release 4.7, owner 8 Nov
        2026); an iteration that reaches it without an acceptable candidate ends the search, and the outcome is
        flagged (LabelContinuity.check_budget_reached).
    """

    emission_site_correlation: float = field(
        default=0.1,
        metadata={
            "provenance": "3.5 painting/config.py: emission_site_correlation; P0 PR08/PR13/PR17; WP_PA2 §7",
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
            "provenance": "3.5 painting/config.py: initial_switch_rate; P0 PR08/PR13/PR17; WP_PA2 §7",
            "role": "state",
            "axes": (),
            "units": "1",
            "dtype": "float64",
            "stored": True,
        },
    )
    initial_error_entry: float = field(
        default=0.001,
        metadata={
            "provenance": "3.5 painting/config.py: initial_error_entry; P0 PR08/PR13/PR17; WP_PA2 §7",
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
            "provenance": "3.5 painting/config.py: initial_error_exit; P0 PR08/PR13/PR17; WP_PA2 §7",
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
            "provenance": "3.5 painting/config.py: estimation_iterations; P0 PR08/PR13/PR17; WP_PA2 §7",
            "role": "state",
            "axes": (),
            "units": "1",
            "dtype": "int64",
            "stored": True,
        },
    )
    estimate_grid: float = field(
        default=2.0 ** (-6),
        metadata={
            "provenance": "3.5 painting/config.py: estimate_grid; P0 PR08/PR13/PR17; WP_PA2 §7",
            "role": "state",
            "axes": (),
            "units": "1",
            "dtype": "float64",
            "stored": True,
        },
    )
    robustness_epsilon: float = field(
        default=0.01,
        metadata={
            "provenance": "3.5 painting/config.py: robustness_epsilon; P0 PR08/PR13/PR17; WP_PA2 §7",
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
            "provenance": "3.5 painting/config.py: snps_per_bin; P0 PR08/PR13/PR17; WP_PA2 §7",
            "role": "state",
            "axes": (),
            "units": "1",
            "dtype": "int64",
            "stored": True,
        },
    )
    emission_log_floor: float = field(
        default=-50.0,
        metadata={
            "provenance": "3.5 painting/config.py: emission_log_floor; P0 PR08/PR13/PR17; WP_PA2 §7",
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
            "provenance": "3.5 painting/config.py: minimum_release_posterior; P0 PR08/PR13/PR17; WP_PA2 §7",
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
            "provenance": "3.5 painting/config.py: label_continuity_minimum_switch_gain; P0 PR08/PR13/PR17; WP_PA2 §7",
            "role": "state",
            "axes": (),
            "units": "1",
            "dtype": "float64",
            "stored": True,
        },
    )
    label_continuity_maximum_loser_ratio: float = field(
        default=0.1,
        metadata={
            "provenance": "3.5 painting/config.py: label_continuity_maximum_loser_ratio; P0 PR08/PR13/PR17; WP_PA2 §7",
            "role": "state",
            "axes": (),
            "units": "1",
            "dtype": "float64",
            "stored": True,
        },
    )
    label_continuity_check_budget: int = field(
        default=64,
        metadata={
            "provenance": "4.7 (owner, 8 Nov 2026): bounds the final, unproductive search iteration at low depth",
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
            "provenance": "3.5 painting/hmm.py:47; noise_quantum(1.9e-6); P0 PR09; WP_PA2 §7",
            "role": "state",
            "axes": (),
            "units": "nats",
            "dtype": "float64",
            "stored": True,
        },
    )


__all__ = [
    "PaintingParameters",
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
    "LABEL_CONTINUITY_BASE_BYTES",
]
