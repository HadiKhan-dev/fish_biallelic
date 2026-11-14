"""Pieces of the genome's material units on one painting component (DESIGN_417 D1, I1).

The genome axis is the working-map coordinate of every scored bin: configured contig order, then components in
map order, each component's per-bin Haldane coordinates of its selector switch probabilities (clipped,
MaterialParameters.switch_clip) added to the running offset of the components before it; gaps between components
contribute nothing. A bin belongs to the unit min(floor(x / unit_morgans), last), where last + 1 is the genome's
nominal unit count (a final piece shorter than tail_morgans joins the unit before it): a bin exactly at a cut
belongs to the unit on its right. A piece is a maximal run of one component's bins in one unit. The cuts are the
genome's (regimes.axis, from every informative prepared layout), so a contig's pieces depend on the material of
every contig; each score identity binds its contig's cut list (``cut_list``). No score or source chain restarts at
a piece boundary: every piece field is a sum of per-bin increments, and the stacking sums pieces into units
exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...contracts.pedigree import MapBlock
from ..shared.material import haldane_coordinates
from .parameters import MaterialParameters

__all__ = ["ScorePlacement", "component_length", "pieces", "cut_list"]


@dataclass(frozen=True)
class ScorePlacement:
    """One informative contig's place on the genome's material axis (regimes.axis, DESIGN_417 I1): the genome
    coordinate of its first scored bin, the genome's last nominal unit, and its cut list (``cut_list``: component
    position, first bin and unit of each piece). The cut list depends on every earlier contig's layout; the score
    identity binds it (and the screen identity binds every informative contig's preparation). Defined with the cut
    list, so a score task's arguments import nothing outside the scorer's own modules."""

    offset: float
    last: int
    cuts: np.ndarray


def component_length(switch: np.ndarray, material: MaterialParameters) -> float:
    """Working Morgans of one component: its last bin's Haldane coordinate."""
    return float(haldane_coordinates(switch, material.switch_clip)[-1])


def pieces(
    switch: np.ndarray, offset: float, last: int, material: MaterialParameters, component: int = 0
) -> tuple[np.ndarray, tuple[MapBlock, ...]]:
    """Dense piece index of each of the component's bins and the pieces' descriptors.

    ``switch`` has B - 1 boundaries (an empty vector is one bin); ``offset`` is the genome coordinate of the
    component's first bin; ``last`` the genome's last nominal unit index. A piece's working length is the
    coordinate of its last bin minus that of the last bin of the piece before it (minus the component's first bin
    for its first piece), so the lengths sum to the component's working Morgans; its nominal index is its unit's.
    """
    coordinates = offset + haldane_coordinates(switch, material.switch_clip)
    grid = np.minimum(np.floor(coordinates / material.unit_morgans).astype(np.int64), int(last))
    starts = np.concatenate(([0], np.flatnonzero(np.diff(grid)) + 1))
    stops = np.concatenate((starts[1:], [len(grid)]))
    dense = np.repeat(np.arange(len(starts), dtype=np.int64), stops - starts)
    parts = tuple(
        MapBlock(
            component,
            int(first),
            int(stop),
            float(coordinates[stop - 1] - (coordinates[first - 1] if first else coordinates[0])),
            int(grid[first]),
        )
        for first, stop in zip(starts, stops, strict=True)
    )
    return np.ascontiguousarray(dense), parts


def cut_list(
    selectors: tuple[np.ndarray, ...], offset: float, last: int, material: MaterialParameters
) -> np.ndarray:
    """int64 [P, 3]: (component position, first bin, nominal unit) of every piece of one contig whose components
    have ``selectors``, from its genome ``offset``: the contig's cuts of the genome units, bound in its score
    identity and checked by its score task."""
    rows = []
    start = float(offset)
    for position, switch in enumerate(selectors):
        _, description = pieces(switch, start, last, material)
        start += component_length(switch, material)
        rows.extend((position, piece.first_bin, piece.nominal_index) for piece in description)
    return np.asarray(rows, dtype=np.int64).reshape(-1, 3)
