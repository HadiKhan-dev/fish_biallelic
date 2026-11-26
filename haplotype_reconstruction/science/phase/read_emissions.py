"""Score each bin's reads conditional on the component prefix under a fixed pair."""

from __future__ import annotations

import math

import numpy as np
from numba import prange, types

from ...core.kernels import bounded_team, kernel
from ..painting.api import TAPE, apply_bin

_F2 = types.float64[:, ::1]
_F3 = types.float64[:, :, ::1]
_F4 = types.float64[:, :, :, ::1]
_M5 = types.float32[:, :, :, :, ::1]
_D5 = types.float64[:, :, :, :, ::1]
_I1 = types.int64[::1]


@kernel(overloads=tuple((_F4, matrix, _F3, _F2, _I1, TAPE, _F2, _I1, _I1, _I1) for matrix in (_M5, _D5)),
        cache=True, parallel=True, nogil=True)
def _predict_pairs(emissions, matrices, log_scale, initial, perm, tape, exponent, active, first, second):
    pairs, bins, classes = len(first), emissions.shape[-1], initial.shape[1]
    # Each sample and pair owns its law. The bins for that law are sequential.
    for item in prange(len(active) * pairs):
        sample, pair = active[item // pairs], np.int64(item % pairs)
        x, y = first[pair], second[pair]
        incoming = initial[sample].copy()
        moved = np.empty(classes)
        work = np.empty((4, classes))
        for b in range(bins):
            # Swapping the homologues permutes both the incoming law and the action.
            # Their conjugate action has the same mass, so score the canonical order once.
            gauge = apply_bin(
                matrices, tape, exponent, sample, pair, b, False, perm, incoming, moved, work, False
            )
            mass = moved.sum()
            score = -np.inf
            if mass > 0.0:
                # The chain scale contains the D5P correction, including the compact matrix's gauge.
                score = emissions[sample, x, y, b] + (log_scale[sample, pair, b] + gauge + math.log(mass))
                incoming[:] = moved / mass
            # As in D5P, keep the incoming law if the action has no mass to normalize.
            emissions[sample, x, y, b] = score
            emissions[sample, y, x, b] = score


def project_read_state(emissions, matrices, log_scale, initial, perm, tape, exponent, bypass):
    """Add painting's fixed-pair forward predictive increments to phase emissions in place."""
    if initial.shape[1] <= 1 or not emissions.shape[-1]:
        return
    active = np.flatnonzero(~bypass)
    if len(active):
        first, second = np.triu_indices(emissions.shape[1])
        with bounded_team(len(active) * len(first)):
            _predict_pairs(emissions, matrices, log_scale, initial, perm, tape, exponent, active, first, second)


def apply_read_matrices(emissions, read, bypass):
    """Consume the confidence-strength bin action and its log scale through the chain contract."""
    project_read_state(emissions, *read.chain_inputs(), bypass)
