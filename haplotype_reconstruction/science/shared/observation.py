"""The observation model shared by every stage (4.21): genotype outliers against the cohort's genotype frequencies.

A sample's normalised genotype likelihoods L(g) at a site, under a stage's genotype law p(g) (a founder pair, a
transmission, an unknown-allele class), have the log likelihood

    e = log[ (1 - eps) P + eps B ],    P = sum_g p(g) L(g),    B = sum_g q(g) L(g)

with q the site's cohort genotype frequencies (science.shared.cohort, stored once per contig in the evidence record)
and eps the run's genotype outlier rate. Generative reading: with probability eps the sample's genotype at the site is
redrawn from the cohort's genotype frequencies (a redraw may return the same genotype), and the reads then follow the
calibrated likelihoods. A discordant site therefore costs at most -log(eps B), and needs no floor.

Because e is linear in p, every consumer gets it exactly by scoring its own genotype laws against the MIXED
likelihoods

    L'(g) = (1 - eps) L(g) + eps B,    sum_g p(g) L'(g) = (1 - eps) P + eps B for every law p (sum_g p(g) = 1),

on the same scale as L (never renormalised): mix_likelihoods. The cohort likelihood is unchanged by the mixture
(sum_g q(g) L'(g) = B), so a consumer holding L' recovers B from q.

The centred score c = e - log B = log[(1 - eps) P / B + eps] >= log eps is the log Bayes factor of the stage's law
(with redraws) against a random member of the cohort. It does not depend on the scale of L, is 0 for an uninformative
cell, and is what sums over different cell sets compare (centred_log_likelihood).

Two rates, one definition (the rate of cohort-law redraws that maximises the likelihood of the observed cells under the
current model, the model's own unexplained class kept out of it): ``outlier_rate_blocks``, estimated by the run's
observation pilot under discovery's block model and read by every stage before the calibration barrier, and
``outlier_rate``, estimated by the calibration under the final copying model and read from the painting on.
"""

from __future__ import annotations

import math

import numpy as np
from numba import prange, types

from ...core.kernels import kernel

__all__ = ["observation_log_likelihood", "centred_log_likelihood", "cohort_background", "mix_likelihoods", "refined_maximum"]

_F = types.float64


@kernel(overloads=((_F, _F, _F),), cache=True, inline="always", nogil=True)
def observation_log_likelihood(model_probability, background, epsilon):
    """e = log[(1 - eps) P + eps B]: the log likelihood of a sample's genotype likelihoods at a site under a model
    probability P = sum_g p(g) L(g), with cohort likelihood B = sum_g q(g) L(g) (module docstring)."""
    return math.log((1.0 - epsilon) * model_probability + epsilon * background)


@kernel(overloads=((_F, _F),), cache=True, inline="always", nogil=True)
def centred_log_likelihood(mixed_probability, background):
    """c = log(P' / B) with P' = sum_g p(g) L'(g) under the mixed likelihoods (= (1 - eps) P + eps B): the centred
    score of the module docstring, at least log eps."""
    return math.log(mixed_probability / background)


@kernel(overloads=((_F, _F, _F, _F, _F, _F),), cache=True, inline="always", nogil=True)
def cohort_background(l0, l1, l2, q0, q1, q2):
    """B = sum_g q(g) L(g): the likelihood of a sample's genotype likelihoods (l0, l1, l2) under the site's cohort
    genotype frequencies (q0, q1, q2)."""
    return q0 * l0 + q1 * l1 + q2 * l2


def _mix_signatures():
    out = []
    for readonly_l in (False, True):
        for readonly_q in (False, True):
            out.append((
                types.Array(_F, 3, "C", readonly=readonly_l),
                types.Array(_F, 2, "C", readonly=readonly_q),
                _F,
                types.Array(_F, 3, "C"),
            ))
    return tuple(out)


@kernel(overloads=_mix_signatures(), cache=True, parallel=True, nogil=True)
def _mix_into(likelihoods, cohort, epsilon, out):
    """out[n, s, g] = (1 - eps) L[n, s, g] + eps B[n, s], B = q[s] . L[n, s]; samples in parallel."""
    n_samples, n_sites, _ = likelihoods.shape
    keep = 1.0 - epsilon
    for n in prange(n_samples):
        for s in range(n_sites):
            l0 = likelihoods[n, s, 0]
            l1 = likelihoods[n, s, 1]
            l2 = likelihoods[n, s, 2]
            redraw = epsilon * (cohort[s, 0] * l0 + cohort[s, 1] * l1 + cohort[s, 2] * l2)
            out[n, s, 0] = keep * l0 + redraw
            out[n, s, 1] = keep * l1 + redraw
            out[n, s, 2] = keep * l2 + redraw


def mix_likelihoods(likelihoods: np.ndarray, cohort: np.ndarray, epsilon: float) -> np.ndarray:
    """float64[N, L, 3] mixed likelihoods L' = (1 - eps) L + eps B of ``likelihoods`` float64[N, L, 3] at sites whose
    cohort genotype frequencies are ``cohort`` float64[L, 3] (module docstring). A new array; the input is unchanged.
    A cell without reads (L = 1/3 each) stays at 1/3 (1 - eps + eps sum q) = 1/3."""
    likelihoods = np.ascontiguousarray(likelihoods, dtype=np.float64)
    cohort = np.ascontiguousarray(cohort, dtype=np.float64)
    if cohort.shape != (likelihoods.shape[1], 3):
        raise ValueError(f"cohort frequencies {cohort.shape} do not match likelihoods {likelihoods.shape}")
    if not 0.0 <= float(epsilon) < 1.0:
        raise ValueError(f"an outlier rate is in [0, 1): {epsilon}")
    out = np.empty(likelihoods.shape, dtype=np.float64)
    _mix_into(likelihoods, cohort, float(epsilon), out)
    return out


def refined_maximum(grid: np.ndarray, total: np.ndarray, best: int) -> float:
    """The rate maximising the pooled one-rate log likelihood, refined between the grid maximum's neighbours (ADVICE2
    14.1 item 12): the vertex in theta = log e of the parabola through the three grid points around the maximum,
    clipped to the neighbours' bracket. The function is smooth and concave in e near its maximum, so the vertex is its
    second-order refinement; a golden-section search would need the cells, which the pilot does not keep. At a grid
    bound the grid value stands (the consumer flags it)."""
    grid = np.asarray(grid, dtype=np.float64)
    total = np.asarray(total, dtype=np.float64)
    if best <= 0 or best >= len(grid) - 1:
        return float(grid[best])
    theta = np.log(grid[best - 1 : best + 2])
    f = total[best - 1 : best + 2]
    h1, h2 = theta[1] - theta[0], theta[2] - theta[1]
    slope = (h1 * h1 * f[2] - h2 * h2 * f[0] + (h2 * h2 - h1 * h1) * f[1]) / (h1 * h2 * (h1 + h2))
    curvature = 2.0 * ((f[2] - f[1]) / h2 - (f[1] - f[0]) / h1) / (h1 + h2)
    if not curvature < 0.0:
        return float(grid[best])
    vertex = theta[1] - slope / curvature
    return float(np.exp(min(max(vertex, theta[0]), theta[2])))
