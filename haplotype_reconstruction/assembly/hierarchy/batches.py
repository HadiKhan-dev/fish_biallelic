"""Break-aware batch partition of one hierarchy level.

At every level the current components are cut into batches of at most ``batch_size`` adjacent components,
orientation-symmetrically (release 2.2, :func:`symmetric_ranges`: mirroring the chromosome mirrors the partition).
A batch never crosses a phase break: a boundary between two adjacent components where fewer than
``min_boundary_informative_samples`` samples are informative on both sides (a sample is informative
for a component when its evidence is non-uniform at one of the component's informative linking-proxy
sites), or a boundary that an earlier level already marked. The samples informative in every
component of a batch (the intersection) are the batch's samples: inference spans the whole batch,
so a sample uninformative somewhere would have an arbitrary state there.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numba
import numpy as np
from numba import njit, prange
from numba.typed import List

from ...model.components import FounderComponent
from ..linker.proxies import proxy_from_usable

INSUFFICIENT_JOINT_SAMPLES = "insufficient_joint_informative_samples"


def component_sites(component: FounderComponent, prepared) -> tuple[np.ndarray, np.ndarray]:
    """(contig site index int64[L], usable bool[L]) of a component's sites: usable = a discovery
    site where at least one founder has a pre-fill call."""
    panels = prepared[component.start:component.stop]
    sites = np.concatenate([np.arange(panel.site_start, panel.site_start + panel.sites, dtype=np.int64)
                            for panel in panels])
    usable = np.concatenate([panel.kept & np.any(panel.inference[component.rows[:, b]] >= 0, axis=0)
                             for b, panel in enumerate(panels)])
    return sites, usable


def informative_sites(component: FounderComponent, prepared, max_sites: int) -> np.ndarray:
    """Contig site indices (int64, ascending) at which a sample counts as informative for the
    component: its usable linking-proxy sites."""
    sites, usable = component_sites(component, prepared)
    proxy = proxy_from_usable(usable, max_sites)
    return np.ascontiguousarray(sites[proxy[usable[proxy]]])


@njit(cache=True, parallel=True)
def _informative_samples(neutral, site_sets):
    """bool[n, N]: sample s is informative for component c when its (neutral) evidence is
    non-uniform and non-zero at one of c's sites. One task per component: its site indices are read
    from the typed List once (a List element read inside the parallel loop would update a reference
    count shared by all threads), and a sample's scan stops at its first informative site."""
    samples = neutral.shape[0]
    result = np.zeros((len(site_sets), samples), np.bool_)
    for index in prange(len(site_sets)):
        sites = site_sets[np.int64(index)]
        for sample in range(samples):
            for site in sites:
                a, b, c = neutral[sample, site]
                if (a + b) + c > 0.0 and (a != b or b != c):
                    result[index, sample] = True
                    break
    return result


def informative_masks(components, prepared, neutral: np.ndarray, max_sites: int) -> np.ndarray:
    """bool[n, N] informative samples of every component (float64 neutral evidence)."""
    site_sets = List.empty_list(numba.types.int64[::1])
    for component in components:
        site_sets.append(informative_sites(component, prepared, max_sites))
    return _informative_samples(neutral, site_sets)


@dataclass(frozen=True)
class LevelPartition:
    """components: the level's input components with every phase break marked (break reasons on
    both sides of a break boundary); ranges: batches as [first, end) component indices; masks:
    bool[n, N] informative samples per component; joint: int64[n - 1] samples informative on both
    sides of each boundary; batch_samples: per batch, its informative samples (the intersection)."""

    components: tuple[FounderComponent, ...]
    ranges: tuple[tuple[int, int], ...]
    masks: np.ndarray
    joint: np.ndarray
    batch_samples: tuple[np.ndarray, ...]


def symmetric_ranges(breaks, batch_size: int) -> list:
    """Orientation-symmetric batches (release 2.2; 2.1 grouped left-anchored, so mirroring the chromosome changed the
    partition): the components between phase breaks (breaks[i]: a break after component i) form segments; a segment
    of L components is cut into m = ceil(L / batch_size) consecutive batches, one more when L is odd and m even (an
    even number of batches has an even total when sizes mirror), with sizes as equal as possible and the larger ones
    placed in mirrored pairs from both ends (one in the middle when the remainder is odd). The size sequence is a
    palindrome, so the partition of the mirrored chromosome is the mirror of this one. Returns [(start, stop)]."""
    n = len(breaks)
    ranges = []
    start = 0
    while start < n:
        end = start + 1
        while end < n and not breaks[end - 1]:
            end += 1
        length = end - start
        count = -(-length // batch_size)
        if length % 2 == 1 and count % 2 == 0:
            count += 1
        base, extra = divmod(length, count)
        sizes = [base] * count
        if extra % 2 == 1:
            sizes[count // 2] += 1
            extra -= 1
        for position in range(extra // 2):
            sizes[position] += 1
            sizes[count - 1 - position] += 1
        for size in sizes:
            ranges.append((start, start + size))
            start += size
    return ranges


def partition(components, prepared, neutral, *, batch_size: int, min_informative: int,
              max_sites: int) -> LevelPartition:
    """Mark phase breaks and cut the components into batches (see the module docstring)."""
    components = list(components)
    masks = informative_masks(components, prepared, neutral, max_sites)
    joint = np.zeros(max(0, len(components) - 1), dtype=np.int64)
    for boundary in range(len(components) - 1):
        left, right = components[boundary], components[boundary + 1]
        joint[boundary] = int(np.sum(masks[boundary] & masks[boundary + 1]))
        reason = left.break_after or right.break_before
        if reason is None and joint[boundary] < min_informative:
            reason = INSUFFICIENT_JOINT_SAMPLES
        if reason is not None:
            if left.break_after != reason:
                components[boundary] = left = replace(left, break_after=reason)
            if right.break_before != reason:
                components[boundary + 1] = replace(right, break_before=reason)
    ranges = symmetric_ranges([component.break_after is not None for component in components], batch_size)
    batch_samples = tuple(np.flatnonzero(np.logical_and.reduce(masks[first:end])).astype(np.int64)
                          for first, end in ranges)
    return LevelPartition(tuple(components), tuple(ranges), masks, joint, batch_samples)
