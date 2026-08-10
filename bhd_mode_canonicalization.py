"""Canonical binary founder-row operations shared by stage-1 mode paths."""

import numpy as np


def canonicalize_binary_panel(haplotypes, assignments):
    """Sort binary founder rows and remap diploid assignments consistently.

    The wildcard sentinel is the input founder count and remains unchanged.
    Returned assignments have each diploid pair ordered increasingly.
    """

    haps = np.asarray(haplotypes)
    assigned = np.asarray(assignments)
    if haps.ndim != 2:
        raise ValueError("haplotypes must be a two-dimensional matrix")
    if assigned.ndim != 2 or assigned.shape[1] != 2:
        raise ValueError("assignments must have shape (samples, 2)")
    k = len(haps)
    byte_rows = np.ascontiguousarray(haps, dtype=np.int8)
    if haps.shape[1] == 0:
        order = np.arange(k, dtype=np.int64)
    else:
        row_keys = byte_rows.view(
            np.dtype((np.void, haps.shape[1]))
        ).reshape(-1)
        order = np.argsort(row_keys, kind="stable")
    inverse = np.empty(k, dtype=np.int64)
    inverse[order] = np.arange(k, dtype=np.int64)
    remapping = np.empty(k + 1, dtype=np.int64)
    remapping[:k] = inverse
    remapping[k] = k
    canonical_assignments = remapping[assigned]
    first = canonical_assignments[:, 0].copy()
    np.minimum(
        first, canonical_assignments[:, 1], out=canonical_assignments[:, 0]
    )
    np.maximum(
        first, canonical_assignments[:, 1], out=canonical_assignments[:, 1]
    )
    canonical_haplotypes = np.ascontiguousarray(haps[order])
    canonical_key = np.ascontiguousarray(byte_rows[order]).tobytes()
    return canonical_haplotypes, canonical_assignments, order, inverse, canonical_key


def exact_unique_binary_rows(matrix):
    """Return exact distinct binary rows in NumPy lexicographic order."""

    rows = np.asarray(matrix)
    if rows.ndim != 2:
        raise ValueError("binary rows must be a two-dimensional matrix")
    if np.any((rows != 0) & (rows != 1)):
        raise ValueError("binary rows must contain only zero and one")
    if len(rows) <= 1:
        return np.array(rows, copy=True, order="C")
    packed = np.packbits(rows, axis=1, bitorder="big")
    first_index_by_key = {}
    for index, packed_row in enumerate(packed):
        first_index_by_key.setdefault(packed_row.tobytes(), index)
    indices = [first_index_by_key[key] for key in sorted(first_index_by_key)]
    return np.ascontiguousarray(rows[indices])


__all__ = ["canonicalize_binary_panel", "exact_unique_binary_rows"]
