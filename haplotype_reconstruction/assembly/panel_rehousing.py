"""Bounded count-down proposals that retain local sequence fragments.

A surplus assembled row can contain real pieces whose feasible recipients
differ along the interval. Move those pieces before proposing its deletion.
This is SEARCH, not a changed scientific score: the caller must fully rescore
every proposal. Retaining local classes here does not make that a universal
acceptance constraint or declare each local class a historical founder.
"""
from heapq import nsmallest

import numpy as np
from numba import njit


def sequence_class_maps(blocks):
    """Exact sequence classes; unknown rows are not invented called alleles."""
    result = []
    for block in blocks:
        calls = np.asarray(block.discrete_haps, dtype=np.int8)
        _, inverse = np.unique(calls, axis=0, return_inverse=True)
        inverse[~np.any(calls >= 0, axis=1)] = -1
        result.append(inverse)
    return result


@njit(cache=True)
def _receiver_path(allowed, first):
    """Minimum recipient changes, O(B K), for one prescribed first recipient.

    The previous minimum supplies every switching transition; a staying
    transition is the only exception. Ties reproduce an ascending-index
    argmin of the explicit K-by-K recurrence. Continuity ranks proposals
    only and never enters the assembly acceptance objective.
    """
    length, k = allowed.shape
    costs = np.full(k, np.inf)
    costs[first] = 0.0
    pointers = np.full((length, k), -1, dtype=np.int64)
    for t in range(1, length):
        best = np.argmin(costs)
        updated = np.full(k, np.inf)
        for recipient in range(k):
            if not allowed[t, recipient]:
                continue
            previous = best
            value = costs[best] + (1.0 if best != recipient else 0.0)
            if costs[recipient] < value or (costs[recipient] == value and recipient < best):
                previous = recipient
                value = costs[recipient]
            updated[recipient] = value
            pointers[t, recipient] = previous
        costs = updated
    path = np.empty(length, dtype=np.int64)
    path[-1] = np.argmin(costs)
    for t in range(length - 1, 0, -1):
        path[t - 1] = pointers[t, path[t]]
    return path


def rehousing_proposals(rows, class_maps, *, full_budget=16, fields=None,
                        deletion_gains=None):
    """Return bounded (proxy gain, description) pairs, O(B K**3) screening.

    At most 2K receiver paths per deleted row are screened without constructing
    whole candidate panels. Only `full_budget` full panels need evaluation.
    `fields` may rank each local edit using fixed-painting evidence. Those
    gains are not additive reoptimized likelihoods, nor an acceptance bound.
    """
    rows = np.asarray(rows, dtype=np.int64)
    k, blocks = rows.shape
    if k < 2:
        return []
    labels = np.column_stack([mapping[rows[:, b]] for b, mapping in enumerate(class_maps)])
    ranked = []
    for drop in range(k):
        keep = np.asarray([i for i in range(k) if i != drop])
        required = []
        permitted = []
        for b in range(blocks):
            donor = labels[drop, b]
            local = labels[keep, b]
            if donor < 0 or np.any(local == donor):
                continue
            required.append(b)
            permitted.append([label < 0 or np.count_nonzero(local == label) > 1
                              for label in local])
        if required:
            allowed = np.asarray(permitted, dtype=np.bool_)
            if not np.all(allowed.any(axis=1)):
                continue
            paths = []
            seen = set()
            for reverse in (False, True):
                oriented = np.ascontiguousarray(allowed[::-1] if reverse else allowed)
                for first in np.flatnonzero(oriented[0]):
                    path = _receiver_path(oriented, int(first))
                    if reverse:
                        path = path[::-1]
                    signature = tuple(map(int, path))
                    if signature not in seen:
                        seen.add(signature)
                        paths.append(signature)
        else:
            paths = [()]
        for path in paths:
            assignments = tuple((b, int(keep[receiver])) for b, receiver in zip(required, path))
            gain = 0.0 if deletion_gains is None else float(deletion_gains[drop])
            if fields is not None:
                gain += sum(float(fields[b][receiver, rows[drop, b]]) for b, receiver in assignments)
            ranked.append((gain, ('rehouse', drop, assignments)))
    return nsmallest(full_budget, ranked, key=lambda item: (-item[0], item[1]))


def materialize_rehousing(description, selected):
    """Move only observed dictionary pieces; do not fill unknown alleles."""
    _, drop, assignments = description
    result = np.asarray(selected, dtype=np.int64).copy()
    for block, receiver in assignments:
        result[receiver, block] = result[drop, block]
    result = np.delete(result, drop, axis=0)
    return list(dict.fromkeys(tuple(map(int, row)) for row in result))
