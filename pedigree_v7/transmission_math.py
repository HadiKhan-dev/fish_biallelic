"""Shared Numba-compatible transmission-model arithmetic."""

import math

import numpy as np
from numba import njit


TINY = np.finfo(np.float64).tiny


@njit(cache=True, inline="always")
def recombination_fraction(distance_bp, recombination_rate):
    """Haldane recombination fraction for a physical interval."""
    value = 0.5 * (1.0 - math.exp(-2.0 * distance_bp * recombination_rate))
    if value < 1e-15:
        return 1e-15
    if value > 0.5:
        return 0.5
    return value


@njit(cache=True, inline="always")
def diploid_distribution(first_alt, second_alt, error_rate):
    """Error-contaminated diploid dosage distribution."""
    first_ref = 1.0 - first_alt
    second_ref = 1.0 - second_alt
    p0 = first_ref * second_ref
    p1 = first_alt * second_ref + first_ref * second_alt
    p2 = first_alt * second_alt
    background = error_rate / 3.0
    retained = 1.0 - error_rate
    return (
        retained * p0 + background,
        retained * p1 + background,
        retained * p2 + background,
    )


@njit(cache=True, inline="always")
def state_emission(child_likelihood, first_alt, second_alt, error_rate):
    """Likelihood of a child's dosage evidence under two allele marginals."""
    p0, p1, p2 = diploid_distribution(first_alt, second_alt, error_rate)
    value = (
        child_likelihood[0] * p0
        + child_likelihood[1] * p1
        + child_likelihood[2] * p2
    )
    return max(value, TINY)
