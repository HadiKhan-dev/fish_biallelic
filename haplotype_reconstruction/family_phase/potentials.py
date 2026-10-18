"""Base log potentials of ordered genotypes from genotype likelihoods conditioned on the scaffold alleles."""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


# Ordered-genotype prior of a pedigree root (no observed parent): homozygotes 1/3, each ordered heterozygote 1/6.
ROOT_PRIOR = np.log(np.asarray([1 / 3, 1 / 6, 1 / 6, 1 / 3], dtype=np.float32))
# Smallest genotype probability entering a log potential.
PROBABILITY_FLOOR = 1e-30


@njit(cache=True, parallel=True)
def base_potentials(likelihoods, observed, scaffold, roots, root_prior):
    """(potentials float32 [N, L, 4], scaffold orientation int8 [N, L], root anchors int64 [N]).

    Ordered genotype g = 2 * first-track allele + second-track allele. The genotype probabilities are the
    normalised likelihoods at observed sites and 1/3 elsewhere. Scaffold alleles condition the dosage: two
    called alleles fix it, one called allele allows that allele plus 0 or 1; impossible genotypes get -inf.
    Roots add ROOT_PRIOR. The orientation is the first track's allele at scaffold heterozygous sites (-1
    elsewhere); a root's anchor is its first such site (-1 for none or for non-roots), fixing its phase gauge.
    """
    samples, sites = observed.shape
    potentials = np.empty((samples, sites, 4), dtype=np.float32)
    orientation = np.full((samples, sites), -1, dtype=np.int8)
    anchors = np.full(samples, -1, dtype=np.int64)
    for sample in prange(samples):
        anchor = -1
        for site in range(sites):
            a, b = scaffold[sample, site, 0], scaffold[sample, site, 1]
            if observed[sample, site]:
                p0 = likelihoods[sample, site, 0]
                p1 = likelihoods[sample, site, 1]
                p2 = likelihoods[sample, site, 2]
            else:
                p0, p1, p2 = 1 / 3, 1 / 3, 1 / 3
            values = (p0, p1, p1, p2)
            for state in range(4):
                dosage = (state >> 1) + (state & 1)
                if a >= 0 and b >= 0:
                    allowed = dosage == a + b
                elif a >= 0:
                    allowed = 0 <= dosage - a <= 1
                elif b >= 0:
                    allowed = 0 <= dosage - b <= 1
                else:
                    allowed = True
                value = (np.float32(math.log(max(values[state], PROBABILITY_FLOOR))) if allowed
                         else np.float32(-np.inf))
                if roots[sample]:
                    value = np.float32(value + root_prior[state])
                potentials[sample, site, state] = value
            if a >= 0 and b >= 0 and a != b:
                orientation[sample, site] = a
                if roots[sample] and anchor < 0:
                    anchor = site
        anchors[sample] = anchor
    return potentials, orientation, anchors
