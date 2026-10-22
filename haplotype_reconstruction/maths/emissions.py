"""Genotype emissions of founder pairs with hard or unknown alleles (assembly's linker and joins).

``pair_code`` classifies an ordered founder pair at a site and ``predictive_likelihood`` gives the probability of a
sample's genotype likelihood triple under that class. Unknown founder alleles (-1 in the pre-fill calls) have a
fixed Bernoulli(1/2) predictive distribution, never a fitted or released value. Two different founder copies are
marginalised independently; two copies of the same latent allele (the same founder, or two founders copying the
same prepared-panel row) share it, so an unknown homozygote cannot produce a heterozygous genotype. This is a
per-observation predictive approximation, not an exact joint integration of unknown alleles across samples.

Seven classes (codes) suffice:

    0, 1, 2   both alleles called: genotype dosage (sum of the two alleles)
    3, 4      one allele called (0 or 1), the other unknown
    5         two unknown alleles of different latent rows
    6         two copies of one unknown latent allele

The joins score a site by the robust, floored, centred log likelihood

    e = max(log(0.99 P + 0.01 / 3), -2) - log(1 / 3)        (robust_log_likelihood(P) - CENTER)

of the sample's normalised genotype likelihoods P under the pair's class; a sample whose likelihoods sum to zero or
are uniform at a site contributes nothing there (normalised_likelihoods). The linker mixes with its own robustness
and no floor (assembly.linker.emissions).
"""
from __future__ import annotations

import math

from numba import njit

GENOTYPE_CODES = 3
"""Codes of fully called pairs (0, 1, 2): the emission categories of a block whose every founder allele at
every emission site is called."""

PREDICTIVE_CODES = 7
"""All codes: the emission categories of a block with unknown founder alleles."""

CENTER = math.log(1.0 / 3.0)
"""The log likelihood of a uniform genotype triple: robust emissions are centred on it."""


@njit(cache=True, inline="always")
def pair_code(first, second, same_source):
    """Code of an ordered founder pair at one site from its alleles (0, 1 or -1 = unknown).

    same_source: the two copies share one latent allele (the same founder, or the same prepared-panel row).
    """
    if first >= 0 and second >= 0:
        return first + second
    if first >= 0:
        return 3 + first
    if second >= 0:
        return 3 + second
    return 6 if same_source else 5


@njit(cache=True, inline="always")
def predictive_likelihood(p0, p1, p2, code):
    """P(observation | pair class ``code``) from the genotype likelihoods p0, p1, p2 of dosages 0, 1, 2."""
    if code == 0:
        return p0
    if code == 1:
        return p1
    if code == 2:
        return p2
    if code == 3:
        return .5 * (p0 + p1)
    if code == 4:
        return .5 * (p1 + p2)
    if code == 5:
        return .25 * p0 + .5 * p1 + .25 * p2
    return .5 * (p0 + p2)


@njit(cache=True, inline="always")
def normalised_likelihoods(p0, p1, p2):
    """(informative, p0, p1, p2): a sample's genotype likelihoods divided by their sum (in their own precision);
    informative is False, and the values are returned unchanged, when the sum is not positive or the three are
    equal (the site carries no evidence for the sample)."""
    total = p0 + p1 + p2
    if total <= 0.0 or (p0 == p1 and p1 == p2):
        return False, p0, p1, p2
    return True, p0 / total, p1 / total, p2 / total


@njit(cache=True, inline="always")
def robust_log_likelihood(likelihood):
    """max(log(0.99 P + 0.01 / 3), -2): the robust log likelihood of a normalised genotype likelihood P (uncentred)."""
    return max(math.log(0.99 * likelihood + 0.01 / 3.0), -2.0)
