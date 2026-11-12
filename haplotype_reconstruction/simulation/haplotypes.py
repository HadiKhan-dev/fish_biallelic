"""Pure breakpoint haplotypes; coordinates are bp and flips are marker indices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts.roles import freeze_record_arrays, record_field


@dataclass(frozen=True)
class Extent:
    """Inclusive physical extent on which crossovers fall, in bp."""

    start_bp: float = record_field("obs", units="bp", dtype="float")
    end_bp: float = record_field("obs", units="bp", dtype="float")


@dataclass(frozen=True)
class Mosaic:
    """Right-continuous source and cross-founder segments plus inherited allele flips.

    The final segment includes the extent end. Equal adjacent label pairs are merged
    by transmission. Arrays are immutable and independent of the source allele panel.
    """

    starts: np.ndarray = record_field("obs", axes=("segments",), units="bp", dtype="float64")
    source: np.ndarray = record_field("obs", axes=("segments",), units="index", dtype="int32")
    founder: np.ndarray = record_field("obs", axes=("segments",), units="index", dtype="int32")
    flips: np.ndarray = record_field("obs", axes=("flips",), units="index", dtype="int64")

    def __post_init__(self):
        for name, dtype in (("starts", np.float64), ("source", np.int32), ("founder", np.int32), ("flips", np.int64)):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=dtype))
        if (
            self.starts.ndim != 1
            or not len(self.starts)
            or np.any(~np.isfinite(self.starts))
            or np.any(np.diff(self.starts) <= 0)
            or self.source.shape != self.starts.shape
            or self.founder.shape != self.starts.shape
        ):
            raise ValueError("mosaic segments must have matching, increasing axes")
        if self.flips.ndim != 1 or np.any(self.flips < 0) or np.any(np.diff(self.flips) <= 0):
            raise ValueError("mosaic flips must be sorted unique marker indices")
        keep = np.r_[True, (self.source[1:] != self.source[:-1]) | (self.founder[1:] != self.founder[:-1])]
        if not np.all(keep):
            for name in ("starts", "source", "founder"):
                object.__setattr__(self, name, getattr(self, name)[keep])
        freeze_record_arrays(self)


def source_at(mosaic: Mosaic, x_bp: np.ndarray) -> np.ndarray:
    """Stable source IDs at physical positions (a boundary belongs to its right)."""
    return mosaic.source[np.searchsorted(mosaic.starts, x_bp, side="right") - 1]


def alleles(mosaic: Mosaic, source_alleles: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Gather a haplotype from a source-ID-indexed panel and apply inherited flips."""
    result = np.asarray(source_alleles[source_at(mosaic, positions), np.arange(len(positions))], dtype=np.int8)
    result[mosaic.flips] ^= 1
    return result


def transmit(parent, first: int, crossovers, new_flips, positions) -> Mosaic:
    """Copy parental intervals and their flips, then toggle unique new marker flips.

    No alleles or random values are needed. Repeated crossover events toggle twice;
    a terminal crossover can introduce a one-marker final segment.
    """
    events = np.asarray(crossovers, dtype=np.float64)
    boundaries = np.unique(np.concatenate((parent[0].starts, parent[1].starts, events)))
    owner = (first + np.searchsorted(events, boundaries, side="right")) % 2
    sources = np.empty(len(boundaries), dtype=np.int32)
    founders = np.empty(len(boundaries), dtype=np.int32)
    inherited = []
    for h, mosaic in enumerate(parent):
        keep = owner == h
        index = np.searchsorted(mosaic.starts, boundaries[keep], side="right") - 1
        sources[keep], founders[keep] = mosaic.source[index], mosaic.founder[index]
        if len(mosaic.flips):
            if positions is None:
                raise ValueError("marker positions are required to transmit flips")
            flip_owner = (first + np.searchsorted(events, positions[mosaic.flips], side="right")) % 2
            inherited.append(mosaic.flips[flip_owner == h])
    keep = np.r_[True, (sources[1:] != sources[:-1]) | (founders[1:] != founders[:-1])]
    flips = np.concatenate(inherited) if inherited else np.empty(0, dtype=np.int64)
    flips = np.setxor1d(flips, np.asarray(new_flips, dtype=np.int64))
    return Mosaic(boundaries[keep], sources[keep], founders[keep], flips)


def founder_mosaic(start_bp: float, source: int, founder: int) -> Mosaic:
    """One source chromosome entering the cross with its cross-founder ID."""
    return Mosaic([start_bp], [source], [founder], [])
