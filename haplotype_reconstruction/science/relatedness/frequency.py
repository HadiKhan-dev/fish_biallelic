"""Estimate the shared Hardy-Weinberg allele frequencies.

Owns the relocated frequency units. No storage, environment or runtime policy.
Method: docs/methods/relatedness.md; WP_RL1. Legacy arithmetic retained for relocation.
"""

from __future__ import annotations

import numba
import numpy as np

from ...contracts.likelihoods import CodedLikelihoods, gather_likelihoods, likelihood_window_sites
from ...contracts.observations import PairArray
from ...core.kernels import kernel
from .parameters import AlleleFrequencyParameters

_DEFAULT_FREQUENCY = AlleleFrequencyParameters()


def allele_frequency(
    gl: np.ndarray | CodedLikelihoods,
    observed: np.ndarray | PairArray,
    parameters: AlleleFrequencyParameters = _DEFAULT_FREQUENCY,
) -> np.ndarray:
    """float64 [L]: the alternative allele frequency under Hardy-Weinberg (EM over the samples' genotype likelihoods
    gl [N, L, 3]; unobserved cells carry the prior), clipped to [clip, 1 - clip].

    One parallel pass per site (allele_frequency_sites), a mathematically equivalent per-site form of the
    array expression it replaced (allele_frequency_reference), without [N, L, 3] temporaries. It agrees bitwise on the
    production replays checked (all contigs of the validation runs); other array shapes can differ in the last bit,
    as numpy's own reductions may associate differently."""
    out = np.empty(gl.shape[1])
    if not isinstance(gl, np.ndarray) or not isinstance(observed, np.ndarray):
        size = likelihood_window_sites(gl) if isinstance(gl, CodedLikelihoods) else 2048
        for left in range(0, gl.shape[1], size):
            right = min(gl.shape[1], left + size)
            out[left:right] = allele_frequency(
                gather_likelihoods(gl, slice(left, right)), observed[:, left:right], parameters
            )
        return out
    allele_frequency_sites(
        np.ascontiguousarray(gl, dtype=np.float64),
        np.ascontiguousarray(observed, dtype=np.bool_),
        parameters.iterations,
        parameters.initial,
        parameters.clip,
        parameters.total_floor,
        out,
    )
    return out


@kernel(
    overloads=tuple(
        (
            numba.types.Array(numba.float64, 3, "C", readonly=gl_readonly),
            numba.types.Array(numba.boolean, 2, "C", readonly=obs_readonly),
            numba.int64,
            numba.float64,
            numba.float64,
            numba.float64,
            numba.float64[::1],
        )
        for gl_readonly in (False, True)
        for obs_readonly in (False, True)
    ),
    parallel=True,
    cache=True,
)
def allele_frequency_sites(gl, observed, iterations, initial, clip, total_floor, out):
    """allele_frequency per site: out[l] after ``iterations`` EM steps from 0.5 (posterior sums over the samples in
    sample order; a cell's genotype weights are summed in genotype order)."""
    N, L = gl.shape[0], gl.shape[1]
    for l in numba.prange(L):
        p = initial
        for _ in range(iterations):
            q = 1.0 - p
            prior0 = q * q
            prior1 = 2.0 * p * q
            prior2 = p * p
            het = 0.0
            hom = 0.0
            for n in range(N):
                if observed[n, l]:
                    x0 = gl[n, l, 0] * prior0
                    x1 = gl[n, l, 1] * prior1
                    x2 = gl[n, l, 2] * prior2
                    total = max((x0 + x1) + x2, total_floor)
                    het += x1 / total
                    hom += x2 / total
                else:
                    het += prior1
                    hom += prior2
            p = min(max(het / N / 2 + hom / N, clip), 1 - clip)
        out[l] = p


__all__ = ["allele_frequency"]
