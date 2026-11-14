"""Equal-material units of a working genetic map, shared by relatedness support and pedigree resampling.

A genome axis is the working-map coordinate (Morgans) of ordered points (painting bins or family sites), in
configured contig order, with no gap between consecutive pieces of the axis. It is cut at every whole multiple of
the unit length; a point exactly at a cut belongs to the unit on its right, and a final piece shorter than the tail
length joins the unit before it (``unit_count``). Empty nominal units are omitted. Units may straddle contigs:
chromosomes assort independently, so straddling only weakens the dependence between units.

Method: docs/methods/pedigree.md#material-units; docs/methods/relatedness.md#rl16.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["haldane_coordinates", "unit_count", "partition_coordinates"]


def haldane_coordinates(switch: np.ndarray, clip: float) -> np.ndarray:
    """float64 [B]: cumulative Haldane working-map coordinate (Morgans) of each of B bins from the B - 1 meiotic
    switch probabilities between them (each clipped at ``clip`` < 0.5); the first bin sits at 0."""
    distance = -0.5 * np.log1p(-2.0 * np.minimum(np.asarray(switch, dtype=np.float64), clip))
    return np.concatenate(([0.0], np.cumsum(distance)))


def unit_count(total: float, unit: float, tail: float) -> int:
    """Number of nominal units of an axis of ``total`` Morgans: one below one unit, else the whole units plus one
    for a final piece of at least ``tail`` (a shorter final piece joins the unit before it)."""
    whole = int(math.floor(total / unit))
    return 1 if whole == 0 else whole + int(total - whole * unit >= tail)


def partition_coordinates(
    coordinates: np.ndarray, unit: float, tail: float
) -> tuple[np.ndarray, tuple[tuple[int, int, float, int], ...]]:
    """Partition nondecreasing axis coordinates ``coordinates`` [P] (the first point at 0 or the axis start).

    Returns the dense unit index of every point (int64 [P]) and one (first, stop, morgans, nominal) per occupied
    unit: its half-open point range, its working length (the coordinate of its last point minus that of the last
    point of the unit before it, or minus 0 for the first unit, so that a gap without points is carried by the next
    occupied unit and the lengths sum to the last coordinate) and its nominal grid index."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    total = float(coordinates[-1])
    count = unit_count(total, unit, tail)
    nominal = np.minimum(np.floor(coordinates / unit).astype(np.int64), count - 1)
    occupied, dense = np.unique(nominal, return_inverse=True)
    parts = []
    for index, value in enumerate(occupied):
        points = np.flatnonzero(dense == index)
        first, stop = int(points[0]), int(points[-1]) + 1
        start = coordinates[first - 1] if first else 0.0
        parts.append((first, stop, float(coordinates[stop - 1] - start), int(value)))
    return np.ascontiguousarray(dense, dtype=np.int64), tuple(parts)
