"""Scientific settings of the painting stage (label continuity, then component painting)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaintingConfig:
    """Model constants of the painting HMM; defaults are the production values.

    The switch probability between bins comes from the contig's genetic map
    (painting/transitions.py), so no recombination rate lives here.

    switch_penalty_per_snp: nats per selected site of a bin subtracted from
        log(theta) for every track change; a change between bins of
        ``snps_per_bin`` sites costs about ``switch_penalty_per_snp *
        snps_per_bin`` nats more than its map probability (about 109 nats per
        track change in total at the defaults).
    robustness_epsilon: uniform genotype-error mixture of the emission,
        P = (1 - eps) P(GL | pair) + eps / 3 per observed site.
    double_recomb_factor: exponent on the switch weight when both tracks
        change at one bin edge.
    snps_per_bin: target selected sites per bin (bins split the selected sites
        into max(1, L // snps_per_bin) near-equal consecutive runs).
    emission_log_floor: lower bound of each site's log emission (nats).
    minimum_release_posterior: tau of the release rule; a track pair is
        released when the posterior mass of its unordered class pair (or of its
        local allele-equivalence pair, if directly read-supported) is >= tau.
        Predeclared 10% within-model posterior-error budget.
    label_continuity_minimum_switch_gain: a row-suffix relabelling must lower
        the painted switch total by at least this many switches.
    label_continuity_maximum_loser_ratio: samples whose painted switches rise
        may be at most this fraction of the samples whose switches fall.
    """

    switch_penalty_per_snp: float = 1.0
    robustness_epsilon: float = 0.01
    double_recomb_factor: float = 1.5
    snps_per_bin: int = 100
    emission_log_floor: float = -50.0
    minimum_release_posterior: float = 0.90
    label_continuity_minimum_switch_gain: float = 2.0
    label_continuity_maximum_loser_ratio: float = 0.1
