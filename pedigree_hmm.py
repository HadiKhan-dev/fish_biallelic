"""Shared transition primitives for pedigree-scoring HMMs.

Only the Poisson one-or-more-crossover approximation belongs here.  Haldane
map fractions and the linear or discrete-time transition models used by other
pipeline stages encode different assumptions and deliberately remain in their
owning modules.
"""

from __future__ import annotations

import numpy as np


def poisson_switch_stay_terms(
    marker_positions,
    recombination_rate,
    *,
    probability_floor=1e-15,
    probability_cap=0.5,
):
    """Return switch probabilities and their switch/stay log costs.

    Distances are measured between adjacent marker positions, with a zero
    distance for the first marker.  The switch probability is
    ``1 - exp(-distance * recombination_rate)`` clipped to the supplied
    numerical bounds.  This is the exact convention historically shared by
    painting, smart pedigree scoring, and recombination-map validation.
    """

    positions = np.asarray(marker_positions, dtype=np.float64)
    distances = np.zeros(len(positions), dtype=np.float64)
    if len(positions) > 1:
        distances[1:] = np.diff(positions)
    probabilities = np.clip(
        1.0 - np.exp(-distances * recombination_rate),
        probability_floor,
        probability_cap,
    )
    return (
        probabilities,
        np.log(probabilities),
        np.log(1.0 - probabilities),
    )
