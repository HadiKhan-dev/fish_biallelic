"""Shared pieces of every sample's source chain (science.pedigree.read_sources).

The chain moves over ordered diplotypes (S^2 states, S = classes + BACKGROUND) with the painting's label-level Hamming
transition, which is symmetric and depends only on how many founders change: _hamming_contract applies it to an
S x S array in O(S^2) through row, column and total sums (never an S^4 transition). SourceInferenceError reports a
sample whose chain lost all supported mass.

Method: docs/methods/pedigree.md#ps05
"""

from __future__ import annotations

import copyreg

from ...core.errors import InvariantError
from ...core.kernels import kernel
from .kernels.preparation import OVERLOADS


class SourceInferenceError(InvariantError):
    """Supported source mass was lost; boundary -1 denotes the initial-bin law."""

    def __init__(self, message: str, sample: int, component: int, boundary: int):
        self.sample, self.component, self.boundary = sample, component, boundary
        super().__init__(f"{message}; sample={sample}, component={component}, boundary={boundary}")

    def __reduce__(self):
        # Raised in graph workers: unpickle without __init__ (the default calls cls(message), a TypeError).
        return copyreg.__newobj__, (type(self), *self.args), self.__dict__


@kernel(overloads=OVERLOADS["sources._hamming_contract"], cache=True, inline="always", nogil=True)
def _hamming_contract(matrix, same, one, two, output, rows, columns):
    """output = A matrix for the symmetric ordered-diplotype Hamming kernel A, in O(S^2)."""
    states = matrix.shape[0]
    rows[:] = 0.0
    columns[:] = 0.0
    total = 0.0
    for first in range(states):
        for second in range(states):
            value = matrix[first, second]
            rows[first] += value
            columns[second] += value
            total += value
    for first in range(states):
        for second in range(states):
            cell = matrix[first, second]
            output[first, second] = (
                same * cell
                + one * (rows[first] + columns[second] - 2.0 * cell)
                + two * (total - rows[first] - columns[second] + cell)
            )
