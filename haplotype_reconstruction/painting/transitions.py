"""The painting's transition between consecutive bins (shared with pedigree).

Each track (homologue) of a sample keeps its state between consecutive bins or
switches to one of the other S - 1 states. With theta the switch probability
of the bin edge (Haldane's recombination fraction of the genetic distance
between the two bin centres, see bin_switch_probabilities) and
theta' = theta * exp(-switch_penalty_per_snp * snps_per_bin), an ordered pair
of track states moves to

* the same pair with weight (1 - theta)^2,
* each of the 2 (S - 1) pairs that change one track with theta' (1 - theta) / (S - 1),
* each of the (S - 1)^2 pairs that change both tracks with theta'^d / (S - 1)^2,
  d = double_recomb_factor.

The table below is normalised over the S^2 successor states; normalising a
whole edge by one constant changes no Viterbi path or posterior.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..model.genetic_map import haldane_fraction
from .config import PaintingConfig

# Smallest switch probability of a bin edge. Two bin centres inside a cold interval of the map (equal cumulative
# cM) are 0 Morgans apart; they keep a negligible chance to switch instead of none, so every transition weight
# stays finite (Viterbi, posteriors and pedigree's chains use the logs). Any positive distance above ~1e-15 M
# (5e-8 bp at 5 cM/Mb) is unaffected.
MINIMUM_SWITCH_PROBABILITY = 1e-15


@dataclass(frozen=True)
class PaintingTransitions:
    """Normalised ordered-diplotype Hamming transition of one component.

    n_states: S (classes + BACKGROUND).
    log_same, log_one, log_two: float64[B - 1], per bin edge e (between bins
        e and e + 1), the log probability of moving to the same ordered pair,
        to one particular pair that changes exactly one track, and to one
        particular pair that changes both tracks:
        exp(log_same) + 2 (S - 1) exp(log_one) + (S - 1)^2 exp(log_two) = 1.
    """

    n_states: int
    log_same: np.ndarray
    log_one: np.ndarray
    log_two: np.ndarray

    def probabilities(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(same, one, two) per edge, the exponentials of the log table."""
        return np.exp(self.log_same), np.exp(self.log_one), np.exp(self.log_two)


def bin_switch_probabilities(centers: np.ndarray, genetic_map) -> np.ndarray:
    """theta per bin edge: Haldane's recombination fraction -expm1(-2 d) / 2 of the genetic distance d
    (Morgans, on the contig's map) between consecutive bin centres, at least MINIMUM_SWITCH_PROBABILITY.

    Haldane's map function (model.genetic_map.haldane_fraction), as in family phase, phase polishing and
    recombination. 92b3daf used the capped linear min(1/2, max(1e-15, d)) here (class-B fix, blueprint 7 and
    9.6); theta = d - d^2 + O(d^3), so a track change costs about d nats more than before, negligible next to the
    ~109-nat switch cost.
    """
    centers = np.asarray(centers, dtype=np.float64)
    morgans = np.ascontiguousarray(genetic_map.interval_morgans(centers[:-1], centers[1:]), dtype=np.float64)
    return np.maximum(MINIMUM_SWITCH_PROBABILITY, haldane_fraction(morgans))


def hamming_log_weights(theta: np.ndarray, n_states: int, *, switch_penalty_per_snp: float,
                        snps_per_bin: int, double_recomb_factor: float) -> PaintingTransitions:
    """The normalised transition table for per-edge switch probabilities ``theta``."""
    theta = np.asarray(theta, dtype=np.float64)
    log_switch = np.log(theta) - float(switch_penalty_per_snp) * int(snps_per_bin)
    log_stay = np.log1p(-theta)
    log_others = math.log(float(n_states - 1))
    same = 2.0 * log_stay
    one = log_switch + log_stay - log_others
    two = float(double_recomb_factor) * log_switch - 2.0 * log_others
    top = np.maximum(np.maximum(same, one), two)
    log_total = top + np.log(
        np.exp(same - top)
        + 2.0 * (n_states - 1) * np.exp(one - top)
        + float((n_states - 1) ** 2) * np.exp(two - top))
    return PaintingTransitions(
        n_states=int(n_states),
        log_same=np.ascontiguousarray(same - log_total),
        log_one=np.ascontiguousarray(one - log_total),
        log_two=np.ascontiguousarray(two - log_total),
    )


def painting_transitions(centers: np.ndarray, n_states: int, config: PaintingConfig,
                         genetic_map) -> PaintingTransitions:
    """The transition table of a component with bin ``centers`` and S = ``n_states``."""
    return hamming_log_weights(
        bin_switch_probabilities(centers, genetic_map), n_states,
        switch_penalty_per_snp=config.switch_penalty_per_snp,
        snps_per_bin=config.snps_per_bin,
        double_recomb_factor=config.double_recomb_factor)
