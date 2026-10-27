"""The painting's copying process between consecutive bins (shared with label continuity and pedigree).

Release 2.14. Each track (homologue) of a sample copies a label, a trajectory class or BACKGROUND (S labels). Along
the genetic map the label changes as a jump process, independently on the two tracks: jumps at rate rho per Morgan,
each to one of the other S - 1 labels uniformly. Over a bin edge of d Morgans (between the two bin centres on the
contig's map) a track therefore ends on a different label with probability

    theta = (S - 1) / S * (1 - exp(-rho d S / (S - 1))),

at least MINIMUM_SWITCH_PROBABILITY, each other label with theta / (S - 1) (theta = rho d for short edges and tends
to the uniform law, never above (S - 1) / S, across long gaps). An ordered label pair moves to

* the same pair with probability (1 - theta)^2,
* each of the 2 (S - 1) pairs that change one track with theta (1 - theta) / (S - 1),
* each of the (S - 1)^2 pairs that change both tracks with theta^2 / (S - 1)^2,

which sum to one (the product of two independent track transitions). rho, the component's effective
ancestry-change rate (a pooled copying rate of its samples, not a number of meioses), is estimated from all its
samples, with the panel-error chain that painting's decoder adds to anchored labels (painting.fidelity). This
label-level table is that process without the panel-error state: label continuity's gain screen and pedigree's
source chain use it, with the estimated rho and the tempered emissions (each sample's bin divided by its design
effect, PaintingConfig.emission_site_correlation).

Before 2.14 a change had weight theta exp(-100) (Haldane's theta, rho = 1, and a coupled double change): a
normalisable but imposed switch-suppression prior, whose effective change rate was not the map's and whose
posteriors were near 0 or 1, which 2.14 replaces by the estimated rate, the explicit panel-error state and the
emissions' design effect.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

# Smallest switch probability of a bin edge. Two bin centres inside a cold interval of the map (equal cumulative
# cM) are 0 Morgans apart; they keep a negligible chance to switch instead of none, so every transition weight
# stays finite (Viterbi, posteriors and pedigree's chains use the logs). Such edges carry no information about rho
# (painting.fidelity.rate_update leaves them out).
MINIMUM_SWITCH_PROBABILITY = 1e-15


@dataclass(frozen=True)
class PaintingTransitions:
    """Normalised ordered-diplotype Hamming transition of one component, with its copying-process parameters.

    n_states: S (classes + BACKGROUND).
    log_same, log_one, log_two: float64[B - 1], per bin edge e (between bins
        e and e + 1), the log probability of moving to the same ordered pair,
        to one particular pair that changes exactly one track, and to one
        particular pair that changes both tracks:
        exp(log_same) + 2 (S - 1) exp(log_one) + (S - 1)^2 exp(log_two) = 1.
    switch_rate: rho, ancestry changes per Morgan per track.
    error_entry, error_exit: a and b of the panel-error chain (painting.fidelity; 0 when it has no error state).
    site_correlation: r of the design effect 1 + (n - 1) r that divided each sample's bin log emission
        (PaintingConfig.emission_site_correlation; 0 = untempered); consumers of the emissions (pedigree's source
        chain) apply the same.
    """

    n_states: int
    log_same: np.ndarray
    log_one: np.ndarray
    log_two: np.ndarray
    switch_rate: float = 0.0
    error_entry: float = 0.0
    error_exit: float = 0.0
    site_correlation: float = 0.0

    def probabilities(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(same, one, two) per edge, the exponentials of the log table."""
        return np.exp(self.log_same), np.exp(self.log_one), np.exp(self.log_two)

    def switch_unit(self) -> float:
        """The median over edges of the log-probability cost of one track change against staying (nats)."""
        if len(self.log_same) == 0:
            return 0.0
        return float(np.median(self.log_same - self.log_one))


def edge_morgans(centers: np.ndarray, genetic_map) -> np.ndarray:
    """float64[B - 1]: genetic distance (Morgans, on the contig's map) between consecutive bin centres."""
    centers = np.asarray(centers, dtype=np.float64)
    return np.ascontiguousarray(genetic_map.interval_morgans(centers[:-1], centers[1:]), dtype=np.float64)


def switch_probabilities(morgans: np.ndarray, rate: float, n_states: int) -> np.ndarray:
    """theta per edge: (S - 1) / S (1 - exp(-rate d S / (S - 1))), at least MINIMUM_SWITCH_PROBABILITY."""
    morgans = np.asarray(morgans, dtype=np.float64)
    if n_states < 2:
        return np.full(len(morgans), MINIMUM_SWITCH_PROBABILITY)
    reach = (n_states - 1.0) / n_states
    return np.maximum(MINIMUM_SWITCH_PROBABILITY, -reach * np.expm1(-float(rate) * morgans / reach))


def copying_transitions(morgans: np.ndarray, n_states: int, rate: float, *, error_entry: float = 0.0,
                        error_exit: float = 0.0, site_correlation: float = 0.0) -> PaintingTransitions:
    """The label-level transition table of a component with edge distances ``morgans`` (Morgans), S = ``n_states``
    and the ancestry-change rate ``rate`` per Morgan (the other parameters are recorded with it)."""
    theta = switch_probabilities(morgans, rate, n_states)
    log_stay = np.log1p(-theta)
    log_change = np.log(theta) - (math.log(float(n_states - 1)) if n_states > 1 else 0.0)
    return PaintingTransitions(
        n_states=int(n_states),
        log_same=np.ascontiguousarray(2.0 * log_stay),
        log_one=np.ascontiguousarray(log_change + log_stay),
        log_two=np.ascontiguousarray(2.0 * log_change),
        switch_rate=float(rate), error_entry=float(error_entry), error_exit=float(error_exit),
        site_correlation=float(site_correlation))
