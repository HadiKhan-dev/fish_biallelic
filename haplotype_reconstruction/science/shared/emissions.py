"""Genotype emissions of founder pairs with hard or unknown alleles (assembly's linker, joins and founder refinement).

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

Assembly scores the CENTRED view of a sample's evidence (science.assembly.evidence): the mixed likelihoods of the
observation model divided by the cohort likelihood, L~(g) = [(1 - eps) L(g) + eps B] / B, B = sum_g q(g) L(g)
(science.shared.observation, eps the run's outlier rate before the calibration barrier, q the site's cohort genotype
frequencies). For a pair class with predictive law p, P~ = sum_g p(g) L~(g) and

    c = log P~ = log[(1 - eps) P / B + eps]

is the log Bayes factor of the class (with redraws) against a random member of the cohort; a redraw from the cohort
scores exactly 0 (sum_g q(g) L~(g) = 1), and so does a sample without evidence at the site (L~ = 1 at every genotype).
The joins and founder refinement add the floor of 4.20 in these units,

    e = max(c, CENTRED_FLOOR),    CENTRED_FLOOR = -2 - log(1 / 3)        (robust_log_likelihood(P~) - CENTER)

(in 4.20 the floor was -2 on log(0.99 P + 0.01 / 3) centred by log(1 / 3); with eps = 0.01 and a uniform cohort law the
two coincide). A sample whose three values are equal at a site contributes nothing there (centred_likelihoods). The
linker scores c itself (assembly.linker.emissions).
"""

from __future__ import annotations

import math

from numba import types

from ...core.kernels import kernel

GENOTYPE_CODES = 3
"""Codes of fully called pairs (0, 1, 2): the emission categories of a block whose every founder allele at
every emission site is called."""

PREDICTIVE_CODES = 7
"""All codes: the emission categories of a block with unknown founder alleles."""

CENTER = 0.0
"""The log likelihood of the cohort law in the centred view (sum_g q(g) L~(g) = 1): site terms are read relative to
it, so a term is the log Bayes factor against a random member of the cohort."""

CENTRED_FLOOR = -2.0 - math.log(1.0 / 3.0)
"""The joins' and refinement's floor on a site term in centred units: the 4.20 floor (-2 on the uncentred term,
centred by log(1 / 3)) kept as the owner allowed, a stand-in for correlated model error (the unexplained-stretch
process). While it binds, a discordant site costs CENTRED_FLOOR rather than log eps."""


@kernel(
    overloads=tuple((a, b, types.boolean) for a in (types.int8, types.int64) for b in (types.int8, types.int64)),
    cache=True,
    inline="always",
    nogil=True,
)
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


@kernel(
    overloads=tuple((p, p, p, c) for p in (types.float32, types.float64) for c in (types.int8, types.int64)),
    cache=True,
    inline="always",
    nogil=True,
)
def predictive_likelihood(p0, p1, p2, code):
    """P(observation | pair class ``code``) from the genotype likelihoods p0, p1, p2 of dosages 0, 1, 2."""
    if code == 0:
        return p0
    if code == 1:
        return p1
    if code == 2:
        return p2
    if code == 3:
        return 0.5 * (p0 + p1)
    if code == 4:
        return 0.5 * (p1 + p2)
    if code == 5:
        return 0.25 * p0 + 0.5 * p1 + 0.25 * p2
    return 0.5 * (p0 + p2)


@kernel(overloads=tuple((p, p, p) for p in (types.float32, types.float64)), cache=True, inline="always", nogil=True)
def centred_likelihoods(p0, p1, p2):
    """(informative, p0, p1, p2): a sample's centred likelihoods at a site (module docstring), returned unchanged
    (never renormalised: their scale is the cohort likelihood); informative is False when the three are equal (the
    site carries no evidence for the sample) or not positive."""
    if not (p0 + p1 + p2 > 0.0) or (p0 == p1 and p1 == p2):
        return False, p0, p1, p2
    return True, p0, p1, p2


@kernel(overloads=((types.float32,), (types.float64,)), cache=True, inline="always", nogil=True)
def robust_log_likelihood(likelihood):
    """max(log P~, CENTRED_FLOOR): the robust site term of a pair class whose centred likelihood is P~ (module
    docstring; the outlier mixture is already in the centred view)."""
    return max(math.log(likelihood), CENTRED_FLOOR)
