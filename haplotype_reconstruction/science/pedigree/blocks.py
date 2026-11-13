"""Frozen component-local map partition (DESIGN_4154 A2).

A boundary belongs to its right bin. Empty nominal blocks are omitted. This
partition never restarts a score or source chain.
"""
from __future__ import annotations
from ...contracts.pedigree import MapBlock as Block
import numpy as np
from .parameters import LOW_MATERIAL_BLOCK_MORGANS, LOW_MATERIAL_TAIL_MORGANS, LOW_MATERIAL_SWITCH_CLIP



def partition(switch: np.ndarray, component: int = 0) -> tuple[np.ndarray, tuple[Block, ...]]:
    """Dense block index of each bin and occupied nominal block descriptors.

    ``switch`` has B-1 boundaries; even an empty boundary vector has one bin.
    Bin spans are half open; working lengths include the boundary entering a span.
    """
    distance = -0.5 * np.log1p(-2.0 * np.minimum(np.asarray(switch), LOW_MATERIAL_SWITCH_CLIP))
    coordinates = np.concatenate(([0.0], np.cumsum(distance)))
    return partition_coordinates(coordinates, component)


def partition_coordinates(coordinates: np.ndarray, component: int = 0) -> tuple[np.ndarray, tuple[Block, ...]]:
    """Partition the cumulative configured coordinates; useful at exact grid boundaries."""
    total = float(coordinates[-1])
    length = LOW_MATERIAL_BLOCK_MORGANS
    whole = int(total // length)
    count = 1 if whole == 0 else whole + int(total - whole * length >= LOW_MATERIAL_TAIL_MORGANS)
    nominal = np.minimum(np.floor(coordinates / length).astype(np.int64), count - 1)
    occupied, dense = np.unique(nominal, return_inverse=True)
    parts = []
    for index, value in enumerate(occupied):
        bins = np.flatnonzero(dense == index)
        first, stop = int(bins[0]), int(bins[-1]) + 1
        start_coordinate = coordinates[first - 1] if first else 0.0
        parts.append(Block(component, first, stop, float(coordinates[stop - 1] - start_coordinate), int(value)))
    return np.ascontiguousarray(dense, dtype=np.int64), tuple(parts)
