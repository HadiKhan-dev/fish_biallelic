"""Build Mendelian child terms and parent laws.

Method: docs/methods/relatedness.md.
"""

from __future__ import annotations

import numpy as np

from .parameters import FamilyParameters

_DEFAULT_FAMILY = FamilyParameters()

TRANSMISSION = np.zeros((9, 3))
for _gu in range(3):
    for _gv in range(3):
        _a, _b = _gu / 2.0, _gv / 2.0
        TRANSMISSION[3 * _gu + _gv] = [(1 - _a) * (1 - _b), _a * (1 - _b) + (1 - _a) * _b, _a * _b]


def family_terms(
    gl: np.ndarray, observed: np.ndarray, p: np.ndarray, parameters: FamilyParameters = _DEFAULT_FAMILY
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(child terms float64 [N, M, 9], log prior float64 [M, 3], log posterior float64 [N, M, 3]) at the given
    sites (gl [N, M, 3], observed [N, M], p [M]).

    Child terms: a sample's log probability of its reads given each parent genotype pair (genotype likelihoods
    scaled to a maximum of 1 per cell; unobserved cells contribute 0). Log prior: Hardy-Weinberg genotype
    probabilities (a latent parent's law). Log posterior: each sample's genotype posterior from its own reads under
    that prior (its law as a sequenced parent; a family conditions on it, so its reads are counted once, in its
    own family)."""
    g = gl / np.maximum(gl.max(axis=2, keepdims=True), parameters.term_floor)
    g = np.where(observed[..., None], g, 1.0)
    terms = np.log(np.maximum(np.einsum("nmg,kg->nmk", g, TRANSMISSION), parameters.term_floor))
    prior = np.stack([(1 - p) ** 2, 2 * p * (1 - p), p**2], axis=1)
    post = g * prior[None]
    post /= post.sum(axis=2, keepdims=True)
    return (
        np.ascontiguousarray(terms),
        np.ascontiguousarray(np.log(prior)),
        np.ascontiguousarray(np.log(np.maximum(post, parameters.term_floor))),
    )


__all__ = ["family_terms"]
