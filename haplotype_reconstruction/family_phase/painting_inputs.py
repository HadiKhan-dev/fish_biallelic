"""Inputs the family-phase stage reads from the painting product: bins, tie boundaries and coverage.

tie_boundaries finds the marker boundaries where the painting's track order was set by a tie rather than data
(release 2.2 semantics, unchanged): a sample whose order is not anchored by a linked parent or by enough children
releases its phase sets broken there. Coverage is the span of every component in which a sample was painted
(eligible); recombination decodes meioses inside coverage only. A component without painted sites uses
EMPTY_COMPONENT_BIN_SITES-site bins over its panel positions.
"""
from __future__ import annotations

import numpy as np

EMPTY_COMPONENT_BIN_SITES = 100


def component_bin_edges(painted, panel_positions):
    """Half-open bp bin edges of a component: its painting bins, or the empty-component fallback."""
    if len(painted.bins.edges) > 1:
        return np.asarray(painted.bins.edges, dtype=np.int64)
    source = np.asarray(panel_positions, dtype=np.int64)
    return np.unique(np.r_[source[::EMPTY_COMPONENT_BIN_SITES], source[-1] + 1])


def _track_changes(before, after, local_keys, bin_index, n_classes, class_of):
    """(label changes, allele-visible changes) of one track across a chunk boundary.

    ``before``/``after`` are released candidate row sets (() = unknown). A track keeps a row iff the sets
    intersect; it keeps its alleles iff some members' classes share a local key below ``n_classes`` in the bin
    before or after the boundary. O(K).
    """
    if not before or not after:
        changed = int(bool(before) != bool(after))
        return changed, changed
    if set(before).intersection(after):
        return 0, 0
    marked = set()
    for row in before:
        for shared_bin in (bin_index - 1, bin_index):
            key = int(local_keys[shared_bin, class_of[row]])
            if key < n_classes:
                marked.add((shared_bin, key))
    for row in after:
        for shared_bin in (bin_index - 1, bin_index):
            key = int(local_keys[shared_bin, class_of[row]])
            if key < n_classes and (shared_bin, key) in marked:
                return 1, 0
    return 1, 1


def tie_boundaries(painting, positions, samples):
    """{sample: int64 marker boundaries (i = between sites i and i + 1) where a painting tie set the track order}.

    Diploid painting emissions are symmetric in track order and the label transitions after a boundary are
    invariant to swapping both tracks, so the track order after a released chunk boundary (painting
    ComponentPainting.chunks) is a tie whenever the ordered and the swapped continuation change equally: tracks
    are compared by their released candidate rows and, among equal label changes, by allele-visible changes
    (local allele-class keys); a boundary is a tie unless the scaffold's continuation is lexicographically
    strictly better than the swapped one.
    """
    pos = np.asarray(positions, dtype=np.int64)
    found = {int(sample): [] for sample in samples}
    for painted in painting:
        if len(painted.bins.edges) <= 1:
            continue
        edges = np.asarray(painted.bins.edges, dtype=np.int64)
        local_keys = np.asarray(painted.local_keys)
        classes = painted.states.classes
        n_classes = len(classes)
        class_of = {int(row): index for index, members in enumerate(classes) for row in members}
        for sample, boundaries in found.items():
            chunks = painted.chunks(sample)
            for left, right in zip(chunks[:-1], chunks[1:]):
                b = int(np.searchsorted(edges, right.start, side="right")) - 1
                direct, crossed = (
                    tuple(map(sum, zip(*(_track_changes(left.members[track],
                                                        right.members[1 - track if swapped else track],
                                                        local_keys, b, n_classes, class_of)
                                         for track in (0, 1)))))
                    for swapped in (False, True))
                if direct < crossed:
                    continue
                index = int(np.searchsorted(pos, right.start, side="left"))
                if 0 < index < len(pos):
                    boundaries.append(index - 1)
    return {sample: np.unique(np.asarray(values, dtype=np.int64)) for sample, values in found.items()}


def coverage(painting, positions, samples):
    """(coverage int64 [R, 2] site ranges [start, stop), offsets int64 [N + 1]): each sample's painted spans."""
    pos = np.asarray(positions, dtype=np.int64)
    spans = [[] for _ in range(samples)]
    for painted in painting:
        edges = np.asarray(painted.bins.edges, dtype=np.int64)
        if len(edges) <= 1:
            continue
        for sample in np.flatnonzero(np.asarray(painted.eligible, dtype=bool)):
            spans[sample].append((int(edges[0]), int(edges[-1])))
    ranges, offsets = [], [0]
    for sample_spans in spans:
        merged = []
        for left, right in sorted(sample_spans):
            if merged and left <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
            else:
                merged.append((left, right))
        for left, right in merged:
            a, b = np.searchsorted(pos, [left, right], side="left")
            if b > a:
                ranges.append((int(a), int(b)))
        offsets.append(len(ranges))
    return np.asarray(ranges, dtype=np.int64).reshape(-1, 2), np.asarray(offsets, dtype=np.int64)
