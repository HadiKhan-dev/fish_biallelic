"""Compose painting read matrices into single-bin likelihoods for family phase.

The flanking bins marginalize their ordered label pairs independently with a uniform
prior. The class state is carried through those mixtures in both directions, so
an alternative pair at the focal bin is compared under the same flanking read
evidence. The two homologue orders receive equal mass. Multiplying the resulting
single-bin potentials is a composite likelihood for a multipoint label path.
"""

from __future__ import annotations

import math

import numpy as np
from numba import prange, types

from ...core.kernels import bounded_team, kernel
from ..painting.api import swap_classes

_F4 = types.Array(types.float64, 4, "C")
_M5 = types.Array(types.float32, 5, "C")
_F3 = types.Array(types.float64, 3, "C")
_F2 = types.Array(types.float64, 2, "C")
_I1 = types.int64[::1]
_BIN_BLOCK = 16


@kernel(overloads=((_F4, _M5, _F3, _I1, _I1, _F4),), cache=True, parallel=True, nogil=True)
def _mix_bins(emissions, matrices, log_scale, swap, active, mixtures):
    """Mix both orders at each bin. The uniform pair prior only changes the message scale."""
    _, states, _, bins = emissions.shape
    classes = len(swap)
    blocks = (bins + _BIN_BLOCK - 1) // _BIN_BLOCK
    for task in prange(len(active) * blocks):
        row, block = task // blocks, task % blocks
        sample = active[row]
        for b in range(block * _BIN_BLOCK, min(bins, (block + 1) * _BIN_BLOCK)):
            mixtures[row, b] = 0.0
            top = -np.inf
            pair = 0
            for x in range(states):
                for y in range(x, states):
                    top = max(top, emissions[sample, x, y, b] + log_scale[sample, pair, b])
                    pair += 1
            pair = 0
            for x in range(states):
                for y in range(x, states):
                    weight = math.exp(emissions[sample, x, y, b] + log_scale[sample, pair, b] - top)
                    matrix = matrices[sample, pair, b]
                    for i in range(classes):
                        for j in range(classes):
                            mixtures[row, b, i, j] += weight * matrix[i, j]
                            if x != y:
                                mixtures[row, b, i, j] += weight * matrix[swap[i], swap[j]]
                    pair += 1


@kernel(overloads=((_F4, _F2, _I1, _F3, _F3),), cache=True, parallel=True, nogil=True)
def _flank_messages(mixtures, initial, active, left, right):
    """Run each independent direction in its original traversal and reduction order."""
    _, bins, classes, _ = mixtures.shape
    for task in prange(2 * len(active)):
        row = task // 2
        if task % 2 == 0:
            left[row, 0] = initial[active[row]]
            next_left = np.empty(classes)
            for b in range(bins - 1):
                for j in range(classes):
                    weight = 0.0
                    for i in range(classes):
                        weight += left[row, b, i] * mixtures[row, b, i, j]
                    next_left[j] = weight
                left[row, b + 1] = next_left / next_left.sum()
        else:
            right[row, bins - 1] = 1.0
            for b in range(bins - 2, -1, -1):
                for i in range(classes):
                    weight = 0.0
                    for j in range(classes):
                        weight += mixtures[row, b + 1, i, j] * right[row, b + 1, j]
                    right[row, b, i] = weight
                right[row, b] /= right[row, b].max()


@kernel(overloads=((_F4, _M5, _F3, _I1, _I1, _F3, _F3),), cache=True, parallel=True, nogil=True)
def _project_bins(emissions, matrices, log_scale, swap, active, left, right):
    """Apply each focal correction using the read-only flanking messages."""
    _, states, _, bins = emissions.shape
    classes = len(swap)
    blocks = (bins + _BIN_BLOCK - 1) // _BIN_BLOCK
    for task in prange(len(active) * blocks):
        row, block = task // blocks, task % blocks
        sample = active[row]
        for b in range(block * _BIN_BLOCK, min(bins, (block + 1) * _BIN_BLOCK)):
            pair = 0
            for x in range(states):
                for y in range(x, states):
                    mass = 0.0
                    matrix = matrices[sample, pair, b]
                    for i in range(classes):
                        for j in range(classes):
                            factor = float(matrix[i, j])
                            if x != y:
                                factor = (factor + float(matrix[swap[i], swap[j]])) * 0.5
                            mass += left[row, b, i] * factor * right[row, b, j]
                    updated = emissions[sample, x, y, b] + log_scale[sample, pair, b] + math.log(mass)
                    emissions[sample, x, y, b] = updated
                    emissions[sample, y, x, b] = updated
                    pair += 1


def project_read_state(emissions, matrices, log_scale, initial, minor, bypass):
    """Update the phase-owned emissions in place. One-class and bypassed rows keep their exact values."""
    bins, classes = emissions.shape[-1], matrices.shape[-1]
    if classes <= 1 or not bins:
        return
    active = np.flatnonzero(~bypass)
    if not len(active):
        return
    swap = swap_classes(minor)
    mixtures = np.empty((len(active), bins, classes, classes))
    left = np.empty((len(active), bins, classes))
    right = np.empty_like(left)
    items = len(active) * ((bins + _BIN_BLOCK - 1) // _BIN_BLOCK)
    with bounded_team(items):
        _mix_bins(emissions, matrices, log_scale, swap, active, mixtures)
    with bounded_team(2 * len(active)):
        _flank_messages(mixtures, initial, active, left, right)
    del mixtures
    with bounded_team(items):
        _project_bins(emissions, matrices, log_scale, swap, active, left, right)


def apply_read_matrices(emissions, read, bypass):
    """Apply a complete painting read-emission build."""
    project_read_state(emissions, read.read, read.read_log, read.initial, read.minor, bypass)
