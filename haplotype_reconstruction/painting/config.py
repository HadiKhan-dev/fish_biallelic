"""Scientific settings of the painting stage (label continuity, then component painting)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaintingConfig:
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
    """

    emission_site_correlation: float = 0.1
    initial_switch_rate: float = 2.0
    initial_error_entry: float = 1e-3
    initial_error_exit: float = 0.25
    estimation_iterations: int = 8
    estimate_grid: float = 2.0 ** -6
    robustness_epsilon: float = 0.01
    snps_per_bin: int = 100
    emission_log_floor: float = -50.0
    minimum_release_posterior: float = 0.90
    label_continuity_minimum_switch_gain: float = 2.0
    label_continuity_maximum_loser_ratio: float = 0.1
