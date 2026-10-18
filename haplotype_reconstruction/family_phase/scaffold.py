"""Phase scaffold of one chromosome: per-sample track alleles decoded from the painting and the founder refit.

Every sample's painting releases, per assembly component, painting bin and track, a set of candidate founder
rows (one row for a named track, the pooled rows of an equivalence class, none when the track abstains). The
scaffold allele of a track at a site is the refit allele of those rows where every member row is called and they
agree, else unknown (-1); unknown alleles are never argmaxed to REF. The scaffold covers every evidence site of
the chromosome: sites of a component's span missing from its panel and sites between components are unknown.

Phase bins are the painting bins (phase is rigid inside a bin); a component without painted sites uses
EMPTY_COMPONENT_BIN_SITES-site bins over its panel positions. A sample's coverage is the span of every component
in which it was painted (eligible), abstentions included; recombination decodes meioses inside coverage only.

``tie_boundaries`` finds the marker boundaries where the painting's track order was set by a tie rather than
data: frame repairs treat them as places where a root's own data do not carry its scaffold orientation.
Painting components are read through their release arrays (member_ids, member_sets) and released chunks
(``chunks(sample)``, the painting's chunk view).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange


EMPTY_COMPONENT_BIN_SITES = 100


@dataclass(frozen=True)
class PhaseScaffold:
    """alleles int8 [N, L, 2] (-1 unknown); bins int64 [L] (chromosome-wide phase-bin index, -1 outside the
    painting bins of a component); components int32 [L] (-1 between components); bin_edges int64 [M] with
    bin_edge_offsets int64 [C+1] (component c's half-open bp bin edges); coverage int64 [R, 2] site ranges
    [start, stop) with coverage_offsets int64 [N+1] (sorted, disjoint per sample)."""

    alleles: np.ndarray
    bins: np.ndarray
    components: np.ndarray
    bin_edges: np.ndarray
    bin_edge_offsets: np.ndarray
    coverage: np.ndarray
    coverage_offsets: np.ndarray


@njit(cache=True, parallel=True)
def _component_alleles(local_positions, columns, inference, edges, members, eligible, set_offsets, set_rows,
                       result, first_site):
    """Scaffold alleles of one component's sites (in place, rows first_site...)."""
    bins = len(edges) - 1
    for sample in prange(members.shape[0]):
        if not eligible[sample]:
            continue
        b = 0
        for index in range(len(local_positions)):
            position = local_positions[index]
            if position < edges[0] or position >= edges[bins]:
                continue
            while edges[b + 1] <= position:
                b += 1
            column = columns[index]
            if column < 0:
                continue
            for track in range(2):
                identifier = members[sample, track, b]
                first = set_offsets[identifier]
                stop = set_offsets[identifier + 1]
                if first == stop:
                    continue
                allele = inference[set_rows[first], column]
                for member in range(first + 1, stop):
                    if allele < 0 or inference[set_rows[member], column] != allele:
                        allele = -1
                        break
                result[sample, first_site + index, track] = allele


def _packed_sets(sets):
    offsets = np.zeros(len(sets) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(rows) for rows in sets])
    rows = np.asarray([row for members in sets for row in members], dtype=np.int64)
    return offsets, rows


def component_bin_edges(painted, panel_positions):
    """Half-open bp bin edges of a component: its painting bins, or the empty-component fallback."""
    if len(painted.bins.edges) > 1:
        return np.asarray(painted.bins.edges, dtype=np.int64)
    source = np.asarray(panel_positions, dtype=np.int64)
    return np.unique(np.r_[source[::EMPTY_COMPONENT_BIN_SITES], source[-1] + 1])


def build_scaffold(painting, panels, positions, samples):
    """PhaseScaffold of one chromosome from its painting components and refit panels (same order).

    A painting component (painting.ComponentPainting) provides ``bins.edges`` int64 [B+1] (B = 0: no painted
    sites), ``eligible`` bool [N], ``member_ids`` int32 [N, 2, B] and ``member_sets`` (interned row tuples, [0] =
    ()); a refit panel ``positions`` int64 [L_c] and ``inference`` int8 [K, L_c] (-1 uncalled).
    """
    pos = np.asarray(positions, dtype=np.int64)
    alleles = np.full((samples, len(pos), 2), -1, dtype=np.int8)
    bins = np.full(len(pos), -1, dtype=np.int64)
    components = np.full(len(pos), -1, dtype=np.int32)
    all_edges, spans = [], [[] for _ in range(samples)]
    offset = 0
    for index, (painted, panel) in enumerate(zip(painting, panels)):
        source = np.asarray(panel.positions, dtype=np.int64)
        lo = int(np.searchsorted(pos, source[0], side="left"))
        hi = int(np.searchsorted(pos, source[-1], side="right"))
        local = pos[lo:hi]
        columns = np.searchsorted(source, local)
        valid = columns < len(source)
        valid[valid] &= source[columns[valid]] == local[valid]
        columns = np.where(valid, columns, -1).astype(np.int64)
        edges = component_bin_edges(painted, source)
        all_edges.append(edges)
        if len(painted.bins.edges) > 1:
            eligible = np.asarray(painted.eligible, dtype=np.bool_)
            set_offsets, set_rows = _packed_sets(painted.member_sets)
            _component_alleles(local, columns, np.asarray(panel.inference, dtype=np.int8), edges,
                               np.asarray(painted.member_ids, dtype=np.int32), eligible, set_offsets, set_rows,
                               alleles, lo)
            for sample in np.flatnonzero(eligible):
                spans[sample].append((int(edges[0]), int(edges[-1])))
        local_bins = np.searchsorted(edges, local, side="right") - 1
        good = (local_bins >= 0) & (local_bins < len(edges) - 1)
        bins[lo:hi] = np.where(good, offset + local_bins, -1)
        components[lo:hi] = index
        offset += len(edges) - 1
    edge_offsets = np.r_[0, np.cumsum([len(edges) for edges in all_edges])].astype(np.int64)
    coverage, coverage_offsets = [], [0]
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
                coverage.append((int(a), int(b)))
        coverage_offsets.append(len(coverage))
    return PhaseScaffold(
        alleles, bins, components,
        np.concatenate(all_edges).astype(np.int64) if all_edges else np.zeros(0, dtype=np.int64),
        edge_offsets, np.asarray(coverage, dtype=np.int64).reshape(-1, 2),
        np.asarray(coverage_offsets, dtype=np.int64))


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
