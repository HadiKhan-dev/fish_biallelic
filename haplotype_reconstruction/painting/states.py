"""Hidden states and bins of one component's painting HMM.

A component's founder rows are pooled into trajectory classes: rows whose
inference calls (-1 / 0 / 1) agree at every selected site cannot be told apart
by any sample and form one class. The HMM state of one homologue (track) is a
class or BACKGROUND, a haplotype the panel does not represent, whose allele at
each site is drawn from the Beta(1, 1) posterior mean of the class alleles
there. A sample's state is an ordered pair of track states.

Selected sites are the component's kept sites where at least one row is called:
a site without any founder call carries no ancestry evidence (q = 0.5 diagonal
versus off-diagonal structure must not create evidence from no observation).
Bins are runs of about ``snps_per_bin`` consecutive selected sites.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange


@dataclass(frozen=True)
class PaintingStates:
    """State alphabet of one component: C trajectory classes plus BACKGROUND (state C).

    site_indices: int64[L_s] selected sites as indices into the component
        panel's positions (kept sites with at least one inference call).
    positions: int64[L_s] their bp positions (strictly increasing).
    named_alleles: int8[C, L_s] each class's allele at the selected sites,
        -1 where the class is uncalled. No two classes are equal.
    classes: tuple of C tuples of component-local founder rows (ascending);
        classes are ordered by their first row.
    background_frequency: float64[L_s] BACKGROUND's ALT probability,
        (1 + ALT calls) / (2 + calls) over the C classes (Beta(1, 1) posterior
        mean, see :func:`background_alt_frequency`).
    """

    site_indices: np.ndarray
    positions: np.ndarray
    named_alleles: np.ndarray
    classes: tuple[tuple[int, ...], ...]
    background_frequency: np.ndarray

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def n_states(self) -> int:
        """S = C + 1 (BACKGROUND is the last state)."""
        return len(self.classes) + 1

    @property
    def n_sites(self) -> int:
        return len(self.positions)

    @property
    def anchored(self) -> np.ndarray:
        """bool[C]: the class is called at some selected site (else UNANCHORED)."""
        return np.any(self.named_alleles >= 0, axis=1)

    def alt_probability(self) -> np.ndarray:
        """float64[C, L_s]: each class's ALT probability (its call, else BACKGROUND's)."""
        return state_alt_probabilities(self.named_alleles, self.background_frequency)

    def public_state_ids(self) -> np.ndarray:
        """int32[S]: anchored classes keep their index; unanchored classes and
        BACKGROUND map to the public UNKNOWN index C."""
        result = np.arange(self.n_states, dtype=np.int32)
        result[:-1][~self.anchored] = self.n_classes
        return result


@dataclass(frozen=True)
class PaintingBins:
    """Consecutive runs of selected sites (B bins, B >= 1 when L_s >= 1).

    starts, stops: int64[B] half-open ranges into the selected sites; the
        selected sites are split into max(1, L_s // snps_per_bin) near-equal
        runs (``np.array_split`` sizes: the first L_s mod B runs are one longer).
    centers: float64[B] mean bp position of each bin's sites (transition distances).
    edges: int64[B + 1] bp boundaries: bin b owns [edges[b], edges[b + 1]);
        edges[0] = first site, edges[B] = last site + 1, and an inner edge is
        floor((last site of b - 1 + first site of b + 1) / 2).
    """

    starts: np.ndarray
    stops: np.ndarray
    centers: np.ndarray
    edges: np.ndarray

    @property
    def n_bins(self) -> int:
        return len(self.starts)


def background_alt_frequency(named_alleles: np.ndarray) -> np.ndarray:
    """BACKGROUND's ALT probability per site: (1 + ALT calls) / (2 + calls).

    ``named_alleles`` int8[C, L] holds -1 / 0 / 1; the result is the Beta(1, 1)
    posterior mean of the called class alleles at each site.
    """
    called = named_alleles >= 0
    alt_calls = np.sum(named_alleles == 1, axis=0)
    calls = np.sum(called, axis=0)
    return (1.0 + alt_calls) / (2.0 + calls)


def state_alt_probabilities(named_alleles: np.ndarray, background_frequency: np.ndarray) -> np.ndarray:
    """float64[C, L]: a class's hard call where called, else the background frequency."""
    called = named_alleles >= 0
    return np.where(called, named_alleles, background_frequency[None, :]).astype(np.float64)


def build_states(positions: np.ndarray, inference: np.ndarray, kept: np.ndarray) -> PaintingStates:
    """Trajectory classes of a component's inference calls at its selected sites.

    ``positions`` int64[L], ``inference`` int8[K, L] (-1 / 0 / 1, the pre-fill
    snapshot) and ``kept`` bool[L] of one component panel. Rows with identical
    calls at every selected site share a class.
    """
    called = inference >= 0
    site_indices = np.flatnonzero(kept & np.any(called, axis=0)).astype(np.int64)
    selected = np.ascontiguousarray(inference[:, site_indices], dtype=np.int8)
    classes: list[list[int]] = []
    seen: dict[bytes, int] = {}
    for row in range(selected.shape[0]):
        signature = selected[row].tobytes()
        index = seen.get(signature)
        if index is None:
            seen[signature] = len(classes)
            classes.append([row])
        else:
            classes[index].append(row)
    representatives = np.asarray([members[0] for members in classes], dtype=np.int64)
    named_alleles = np.ascontiguousarray(selected[representatives]).reshape(len(classes), len(site_indices))
    return PaintingStates(
        site_indices=site_indices,
        positions=np.ascontiguousarray(np.asarray(positions, dtype=np.int64)[site_indices]),
        named_alleles=named_alleles,
        classes=tuple(tuple(members) for members in classes),
        background_frequency=background_alt_frequency(named_alleles),
    )


def build_bins(positions: np.ndarray, snps_per_bin: int) -> PaintingBins:
    """Split the selected sites into max(1, L // snps_per_bin) near-equal consecutive bins."""
    positions = np.asarray(positions, dtype=np.int64)
    n_sites = len(positions)
    if n_sites == 0:
        empty = np.zeros(0, dtype=np.int64)
        return PaintingBins(empty, empty.copy(), np.zeros(0, dtype=np.float64),
                            np.zeros(0, dtype=np.int64))
    n_bins = max(1, n_sites // int(snps_per_bin))
    # np.array_split sizes: the first (n_sites mod n_bins) bins hold one extra site.
    base, extra = divmod(n_sites, n_bins)
    sizes = np.full(n_bins, base, dtype=np.int64)
    sizes[:extra] += 1
    stops = np.cumsum(sizes)
    starts = stops - sizes
    # Exact integer sums (far below 2**53) divided by the counts: equal to np.mean per bin.
    sums = np.add.reduceat(positions.astype(np.float64), starts)
    centers = sums / sizes
    edges = np.empty(n_bins + 1, dtype=np.int64)
    edges[0] = positions[0]
    edges[1:-1] = (positions[stops[:-1] - 1] + positions[starts[1:]] + 1) // 2
    edges[-1] = positions[-1] + 1
    return PaintingBins(starts.astype(np.int64), stops.astype(np.int64), centers, edges)


@njit(cache=True, parallel=True)
def _local_allele_keys_kernel(named_alleles, bin_starts, bin_stops):
    n_classes, n_sites = named_alleles.shape
    n_bins = len(bin_starts)
    keys = np.empty((n_bins, n_classes + 1), dtype=np.int32)
    anchored = np.zeros(n_classes, dtype=np.bool_)
    for class_index in range(n_classes):
        for site in range(n_sites):
            if named_alleles[class_index, site] >= 0:
                anchored[class_index] = True
                break
    for bin_index in prange(n_bins):
        start = bin_starts[bin_index]
        stop = bin_stops[bin_index]
        poolable = np.zeros(n_classes, dtype=np.bool_)
        for class_index in range(n_classes):
            if anchored[class_index]:
                for site in range(start, stop):
                    if named_alleles[class_index, site] >= 0:
                        poolable[class_index] = True
                        break
        # Partition refinement: a new group is the first-seen pair of
        # (previous group, allele + 1), so groups are exact {-1,0,1} vectors.
        group = np.zeros(n_classes, dtype=np.int64)
        table = np.empty(3 * n_classes, dtype=np.int64)
        for site in range(start, stop):
            table[:] = -1
            next_group = 0
            for class_index in range(n_classes):
                if not poolable[class_index]:
                    continue
                slot = group[class_index] * 3 + named_alleles[class_index, site] + 1
                if table[slot] < 0:
                    table[slot] = next_group
                    next_group += 1
                group[class_index] = table[slot]
        smallest = np.full(n_classes + 1, -1, dtype=np.int64)
        for class_index in range(n_classes):
            if poolable[class_index] and smallest[group[class_index]] < 0:
                smallest[group[class_index]] = class_index
        for class_index in range(n_classes):
            if not anchored[class_index]:
                keys[bin_index, class_index] = n_classes
            elif poolable[class_index]:
                keys[bin_index, class_index] = smallest[group[class_index]]
            else:
                keys[bin_index, class_index] = class_index
        keys[bin_index, n_classes] = n_classes
    return keys


def local_allele_keys(named_alleles: np.ndarray, bins: PaintingBins) -> np.ndarray:
    """Per-bin allele-equivalence keys of the states, int32[B, C + 1].

    A class is poolable in a bin if it is anchored (called at some selected
    site of the component) and called at one of the bin's sites. Poolable
    classes with identical -1/0/1 vectors over the bin share the key of their
    smallest class; other anchored classes key to themselves; unanchored
    classes and BACKGROUND key to the public UNKNOWN index C. Two classes are
    indistinguishable in a bin iff they share a key below C. Exact partition
    refinement, O(C) per site.
    """
    return _local_allele_keys_kernel(
        np.ascontiguousarray(named_alleles, dtype=np.int8),
        np.ascontiguousarray(bins.starts, dtype=np.int64),
        np.ascontiguousarray(bins.stops, dtype=np.int64))


def multi_member_local_states(local_keys: np.ndarray) -> np.ndarray:
    """bool[B, S]: the state's local allele class pools at least two HMM classes in the bin."""
    n_bins, n_states = local_keys.shape
    n_classes = n_states - 1
    rows = np.broadcast_to(np.arange(n_bins)[:, None], (n_bins, n_classes))
    pooled = local_keys[:, :n_classes] < n_classes
    sizes = np.zeros((n_bins, n_states), dtype=np.int64)
    np.add.at(sizes, (rows[pooled], local_keys[:, :n_classes][pooled]), 1)
    result = np.zeros((n_bins, n_states), dtype=np.bool_)
    result[:, :n_classes] = pooled & (np.take_along_axis(
        sizes, np.minimum(local_keys[:, :n_classes], n_classes).astype(np.intp), axis=1) >= 2)
    return result
