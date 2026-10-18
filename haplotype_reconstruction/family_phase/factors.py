"""Copy factors and selector support of the family model.

Ordered genotypes have four states per sample and site (2 * first-track allele + second-track allele). A copy
factor joins a child to its observed parents through two transmission selectors (one bit per parent: which
parental homologue the child's track copies), each copy erring with probability ``transmission_error`` (an
unobserved slot's allele has probability 1/2). The family MAP (map_solver.py) evaluates these factors directly;
this module keeps the hard-homozygote test and each child's selector support (release 2.1: the belief-propagation
messages and sweeps of 2.0 are gone).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def fixed_nodes(base):
    """bool [N, L]: hard homozygotes of the base potentials."""
    n, l, _ = base.shape
    fixed = np.zeros((n, l), dtype=np.bool_)
    for i in prange(n):
        for j in range(l):
            fixed[i, j] = (base[i, j, 1] == -np.inf and base[i, j, 2] == -np.inf and
                           ((base[i, j, 0] > -np.inf and base[i, j, 3] == -np.inf) or
                            (base[i, j, 3] > -np.inf and base[i, j, 0] == -np.inf)))
    return fixed


@njit(cache=True, parallel=True)
def _count_selector_markers(fixed, gp):
    counts = np.zeros(len(gp), dtype=np.int64)
    for group in prange(len(gp)):
        p0, p1 = gp[group]
        for site in range(fixed.shape[1]):
            if (p0 >= 0 and not fixed[p0, site]) or (p1 >= 0 and not fixed[p1, site]):
                counts[group] += 1
    return counts


@njit(cache=True, parallel=True)
def _fill_selector_markers(fixed, gp, offsets, log_correlation, resets):
    indices = np.empty(offsets[-1], dtype=np.int32)
    theta = np.zeros(offsets[-1])
    for group in prange(len(gp)):
        p0, p1 = gp[group]
        cursor = offsets[group]
        previous = -1
        for site in range(fixed.shape[1]):
            if not ((p0 >= 0 and not fixed[p0, site]) or (p1 >= 0 and not fixed[p1, site])):
                continue
            indices[cursor] = site
            if previous >= 0:
                theta[cursor - 1] = (.5 if resets[site] != resets[previous] else
                                     -.5 * math.expm1(log_correlation[site] - log_correlation[previous]))
            previous = site
            cursor += 1
    return offsets, indices, theta


def selector_support(base, gp, theta):
    """(offsets, marker indices, transition probabilities) of every group's selector chain.

    A selector carries evidence only where some parent is not a hard homozygote; the other markers are
    integrated out exactly: two supported markers are joined by the composed switch probability
    (1 - prod(1 - 2 theta)) / 2 of the intervals between them (1/2 across a theta = 1/2 reset).
    """
    fixed = fixed_nodes(base)
    counts = _count_selector_markers(fixed, gp)
    offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
    resets = np.r_[0, np.cumsum(theta == .5)].astype(np.int64)
    logs = np.zeros(len(theta))
    np.log1p(-2 * theta, out=logs, where=theta < .5)
    log_correlation = np.r_[0., np.cumsum(logs)]
    return _fill_selector_markers(fixed, gp, offsets, log_correlation, resets)
