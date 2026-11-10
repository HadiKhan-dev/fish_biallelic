"""Preserve one cumulative map and its forward/inverse coordinate conventions (EC27).

Owns ChromosomeGeneticMap, GeneticMapSet and measured_coordinates; no I/O, engine
or science imports. Inverse plateaus select the leftmost knot at equality and
start strictly higher coordinates after the plateau's right edge. Tails use the
same fallback slope in both directions; zero slope refuses unattainable values.
Map identity uses SHA-256 of the declared byte encoding.
Method: docs/genetic_maps.md. Derived _morgans and _digest are rebuilt
from copied positions_bp and map_cm and never stored.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .roles import record_field

__all__ = [
    "ChromosomeGeneticMap",
    "GeneticMap",
    "GeneticMapSet",
    "measured_coordinates",
    "DEFAULT_RATE_CM_PER_MB",
    "MAP_FORMAT",
]
# Default constant recombination rate in cM/Mb.
DEFAULT_RATE_CM_PER_MB = 5.0
MAP_FORMAT = "piecewise_linear_cm_v1"


@dataclass(frozen=True, eq=False)
class ChromosomeGeneticMap:
    """One chromosome's map: knots (positions_bp float64[M] strictly increasing, map_cm float64[M]
    nondecreasing cumulative cM, M = 0 or M >= 2) and the fallback rate (cM/Mb) used without knots and beyond
    them."""

    contig: str = record_field("state", dtype="str")
    positions_bp: NDArray[np.float64] = record_field("state", axes=("M",), units="bp", dtype="float64", default=None)
    map_cm: NDArray[np.float64] = record_field("state", axes=("M",), units="cM", dtype="float64", default=None)
    default_rate_cm_per_mb: float = record_field(
        "state", units="cM/Mb", dtype="float64", default=DEFAULT_RATE_CM_PER_MB
    )
    _morgans: NDArray[np.float64] = record_field(
        "state", axes=("M",), units="Morgans", dtype="float64", stored=False, init=False, repr=False, compare=False
    )
    _digest: str = record_field("state", dtype="str", stored=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        set_(self, "contig", str(self.contig))
        rate = float(self.default_rate_cm_per_mb)
        if not np.isfinite(rate) or rate < 0:
            raise ValueError("default_rate_cm_per_mb must be finite and nonnegative")
        set_(self, "default_rate_cm_per_mb", rate)
        if (self.positions_bp is None) != (self.map_cm is None):
            raise ValueError("positions_bp and map_cm must be supplied together")
        positions = np.array([] if self.positions_bp is None else self.positions_bp, dtype=np.float64)
        cm = np.array([] if self.map_cm is None else self.map_cm, dtype=np.float64)
        if positions.ndim != 1 or cm.shape != positions.shape:
            raise ValueError(f"{self.contig}: map positions and cumulative cM must be matching vectors")
        if len(positions):
            if len(positions) < 2:
                raise ValueError(f"{self.contig}: a supplied map needs at least two knots")
            if not np.all(np.isfinite(positions)) or np.any(positions < 0):
                raise ValueError(f"{self.contig}: physical positions must be finite and nonnegative")
            if np.any(np.diff(positions) <= 0):
                raise ValueError(f"{self.contig}: physical positions must be strictly increasing (no duplicates)")
            if not np.all(np.isfinite(cm)) or np.any(np.diff(cm) < 0):
                raise ValueError(f"{self.contig}: cumulative cM must be finite and nondecreasing")
        positions.flags.writeable = False
        cm.flags.writeable = False
        set_(self, "positions_bp", positions)
        set_(self, "map_cm", cm)
        morgans = cm / 100.0
        morgans.flags.writeable = False
        set_(self, "_morgans", morgans)
        digest = hashlib.sha256()
        digest.update(positions.astype("<f8").tobytes())
        digest.update(cm.astype("<f8").tobytes())
        set_(self, "_digest", digest.hexdigest())

    @property
    def has_map(self) -> bool:
        return bool(len(self.positions_bp))

    @property
    def fallback_rate_per_bp(self) -> float:
        """Morgans per bp of the fallback rate (5 cM/Mb = 5e-8)."""
        return self.default_rate_cm_per_mb / 1e8

    def cumulative_morgans(self, positions: ArrayLike) -> NDArray[np.float64] | np.float64:
        """Continuous genetic coordinates (Morgans), keeping the supplied cM origin.

        The fallback slope extends both tails continuously (the left extension can be negative if the first knot
        is labelled zero; distances stay nonnegative for ordered intervals). Scalar input gives scalar output.
        """
        positions = np.asarray(positions, dtype=np.float64)
        if not self.has_map:
            return positions * self.fallback_rate_per_bp
        knots, morgans = self.positions_bp, self._morgans
        result = np.interp(positions, knots, morgans)
        result = np.where(positions < knots[0], morgans[0] + (positions - knots[0]) * self.fallback_rate_per_bp, result)
        return np.where(
            positions > knots[-1], morgans[-1] + (positions - knots[-1]) * self.fallback_rate_per_bp, result
        )

    def interval_morgans(self, left: ArrayLike, right: ArrayLike) -> NDArray[np.float64] | np.float64:
        """Genetic length (Morgans) of ordered intervals [left, right]; endpoints broadcast."""
        left, right = np.broadcast_arrays(np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64))
        if np.any(right < left):
            raise ValueError("interval right endpoints must not precede left endpoints")
        if not self.has_map:
            return (right - left) * self.fallback_rate_per_bp
        return self.cumulative_morgans(right) - self.cumulative_morgans(left)

    def inverse_morgans(self, values: ArrayLike) -> NDArray[np.float64] | np.float64:
        """Physical positions of genetic coordinates (continuous-position simulation).

        A finite plateau returns its leftmost knot; coordinates strictly above it start after its right edge, so
        a continuous draw never lands in a zero-rate interval. With a zero fallback rate, coordinates outside the
        map's range cannot be inverted. A wholly constant map maps its only coordinate to zero (no map) or the
        first knot.
        """
        values = np.asarray(values, dtype=np.float64)
        rate = self.fallback_rate_per_bp
        if not self.has_map:
            if rate == 0:
                if np.any(values != 0):
                    raise ValueError("cannot invert unattainable coordinates with zero recombination rate")
                return np.zeros_like(values)
            return values / rate
        knots, morgans = self.positions_bp, self._morgans
        low = values < morgans[0]
        high = values > morgans[-1]
        if rate == 0 and np.any(low | high):
            raise ValueError("cannot invert coordinates beyond map endpoints with zero fallback rate")
        right = np.clip(np.searchsorted(morgans, values, side="left"), 0, len(morgans) - 1)
        left = np.maximum(right - 1, 0)
        span = morgans[right] - morgans[left]
        fraction = np.divide(values - morgans[left], span, out=np.zeros_like(values), where=span > 0)
        result = knots[left] + fraction * (knots[right] - knots[left])
        result = np.where(values == morgans[right], knots[right], result)
        if rate > 0:
            result = np.where(low, knots[0] + (values - morgans[0]) / rate, result)
            result = np.where(high, knots[-1] + (values - morgans[-1]) / rate, result)
        return result

    def identity(self) -> dict:
        """Small semantic identity (independent of the file it came from) for stage identities."""
        return dict(
            format=MAP_FORMAT,
            contig=self.contig,
            default_rate_cm_per_mb=self.default_rate_cm_per_mb,
            knots=len(self.positions_bp),
            map_sha256=self._digest if self.has_map else None,
        )


@dataclass(frozen=True, eq=False)
class GeneticMapSet:
    """The run's genetic maps by exact chromosome name, with the fallback rate for the others."""

    maps: Mapping[str, ChromosomeGeneticMap] = record_field(
        "state", dtype="Mapping[str, ChromosomeGeneticMap]", default_factory=dict
    )
    default_rate_cm_per_mb: float = record_field(
        "state", units="cM/Mb", dtype="float64", default=DEFAULT_RATE_CM_PER_MB
    )

    def __post_init__(self) -> None:
        rate = ChromosomeGeneticMap("", default_rate_cm_per_mb=self.default_rate_cm_per_mb).default_rate_cm_per_mb
        object.__setattr__(self, "default_rate_cm_per_mb", rate)
        object.__setattr__(self, "maps", dict(self.maps))

    def for_contig(self, contig: str) -> ChromosomeGeneticMap:
        """The chromosome's map, or a map-free one with the fallback rate."""
        contig = str(contig)
        if contig in self.maps:
            return self.maps[contig]
        return ChromosomeGeneticMap(contig, default_rate_cm_per_mb=self.default_rate_cm_per_mb)

    def identity(self, contig: str | None = None) -> dict:
        if contig is not None:
            return self.for_contig(contig).identity()
        return dict(
            format=MAP_FORMAT,
            default_rate_cm_per_mb=self.default_rate_cm_per_mb,
            chromosomes={name: self.maps[name].identity() for name in sorted(self.maps)},
        )


# Both public names identify the same constant-rate map type.
GeneticMap = ChromosomeGeneticMap


def measured_coordinates(
    genetic_map: ChromosomeGeneticMap, positions: ArrayLike, rate_is_measured: bool
) -> NDArray[np.float64]:
    """Return measured Morgans at bp positions, clipping unmeasured map tails.

    With rate_is_measured, the supplied fallback measures tails and absent maps;
    otherwise an absent map measures zeros. Input vectors are never mutated.
    Carried from relatedness/stage.py:61 and pedigree/prepare.py:118 (EC27).
    """
    positions = np.asarray(positions, dtype=np.float64)
    if rate_is_measured:
        return np.asarray(genetic_map.cumulative_morgans(positions), dtype=np.float64)
    if not genetic_map.has_map:
        return np.zeros(len(positions))
    knots = genetic_map.positions_bp
    return np.asarray(genetic_map.cumulative_morgans(np.clip(positions, knots[0], knots[-1])), dtype=np.float64)
