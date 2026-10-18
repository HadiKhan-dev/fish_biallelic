"""Genetic maps: cumulative piecewise-linear maps per chromosome, map files, and genetic distance transforms.

A `ChromosomeGeneticMap` turns physical positions (bp) into genetic coordinates (Morgans) by linear interpolation
between knots; outside the knots, and for chromosomes without a supplied map, it extends with the fallback rate
(`default_rate_cm_per_mb`, 5 cM/Mb = 5e-8 Morgans/bp unless configured). `GeneticMapSet` holds the maps of a run
([genetic_map] in the run configuration) and gives every chromosome its map or the fallback.

Genetic distances d (Morgans) become the probabilities the models use through:

    haldane_fraction(d)          (1 - e^{-2d}) / 2: recombination fraction of one meiosis (a Numba function, callable
                                 from Python with float64 scalars or arrays and inlined into kernels): painting's
                                 bin-to-bin switches, the recombination decoder, screening and junction hotspots
    poisson_switch_stay_terms    1 - e^{-d} switch probabilities and their log switch/stay costs (pedigree path model)

Family phase computes the Haldane fraction of its marker intervals with NumPy's vectorised expm1 (solver.py), which
differs from the scalar expm1 of these kernels in the last bit, and local selection's copying path model uses
1 - e^{-generations x d} in NumPy (path_model.py).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib

import numba
import numpy as np

DEFAULT_RATE_CM_PER_MB = 5.0

MAP_FORMAT = "piecewise_linear_cm_v1"


@numba.njit(cache=True, inline="always")
def haldane_fraction(morgans):
    """Haldane recombination fraction of one meiosis: -expm1(-2 d) / 2."""
    return -0.5 * np.expm1(-2.0 * morgans)


@dataclass(frozen=True, eq=False)
class ChromosomeGeneticMap:
    """One chromosome's map: knots (positions_bp float64[M] strictly increasing, map_cm float64[M]
    nondecreasing cumulative cM, M = 0 or M >= 2) and the fallback rate (cM/Mb) used without knots and beyond
    them."""

    contig: str
    positions_bp: np.ndarray = None
    map_cm: np.ndarray = None
    default_rate_cm_per_mb: float = DEFAULT_RATE_CM_PER_MB
    _morgans: np.ndarray = field(init=False, repr=False, compare=False)
    _digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self):
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
        set_(self, "_morgans", cm / 100.0)
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

    def cumulative_morgans(self, positions):
        """Continuous genetic coordinates (Morgans), keeping the supplied cM origin.

        The fallback slope extends both tails continuously (the left extension can be negative if the first knot
        is labelled zero; distances stay nonnegative for ordered intervals). Scalar input gives scalar output.
        """
        positions = np.asarray(positions, dtype=np.float64)
        if not self.has_map:
            return positions * self.fallback_rate_per_bp
        knots, morgans = self.positions_bp, self._morgans
        result = np.interp(positions, knots, morgans)
        result = np.where(positions < knots[0],
                          morgans[0] + (positions - knots[0]) * self.fallback_rate_per_bp, result)
        return np.where(positions > knots[-1],
                        morgans[-1] + (positions - knots[-1]) * self.fallback_rate_per_bp, result)

    def interval_morgans(self, left, right):
        """Genetic length (Morgans) of ordered intervals [left, right]; endpoints broadcast."""
        left, right = np.broadcast_arrays(np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64))
        if np.any(right < left):
            raise ValueError("interval right endpoints must not precede left endpoints")
        if not self.has_map:
            return (right - left) * self.fallback_rate_per_bp
        return self.cumulative_morgans(right) - self.cumulative_morgans(left)

    def inverse_morgans(self, values):
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

    def record(self) -> dict:
        """Portable constructor arguments."""
        return dict(contig=self.contig,
                    positions_bp=self.positions_bp.tolist() if self.has_map else None,
                    map_cm=self.map_cm.tolist() if self.has_map else None,
                    default_rate_cm_per_mb=self.default_rate_cm_per_mb)

    def identity(self) -> dict:
        """Small semantic identity (independent of the file it came from) for stage identities."""
        return dict(format=MAP_FORMAT, contig=self.contig, default_rate_cm_per_mb=self.default_rate_cm_per_mb,
                    knots=len(self.positions_bp), map_sha256=self._digest if self.has_map else None)


def poisson_switch_stay_terms(marker_positions, recombination_rate, *, probability_floor=1e-15,
                              probability_cap=0.5, chromosome_map: ChromosomeGeneticMap | None = None):
    """Per-marker switch probabilities and their switch/stay log costs (pedigree path model).

    Distances are between adjacent markers (zero for the first). The switch probability is
    ``1 - exp(-distance x rate)`` (genetic distance from ``chromosome_map`` when given) clipped to
    [probability_floor, probability_cap]. Returns (probabilities, log probabilities, log(1 - probabilities)).
    """
    positions = np.asarray(marker_positions, dtype=np.float64)
    distances = np.zeros(len(positions), dtype=np.float64)
    if len(positions) > 1:
        distances[1:] = np.diff(positions)
    mass = distances * recombination_rate
    if chromosome_map is not None:
        if chromosome_map.has_map:
            mass[1:] = chromosome_map.interval_morgans(positions[:-1], positions[1:])
        else:
            mass = distances * chromosome_map.fallback_rate_per_bp
    probabilities = np.clip(1.0 - np.exp(-mass), probability_floor, probability_cap)
    return probabilities, np.log(probabilities), np.log(1.0 - probabilities)


@dataclass(frozen=True, eq=False)
class GeneticMapSet:
    """The run's genetic maps by exact chromosome name, with the fallback rate for the others."""

    maps: Mapping[str, ChromosomeGeneticMap] = field(default_factory=dict)
    default_rate_cm_per_mb: float = DEFAULT_RATE_CM_PER_MB

    def __post_init__(self):
        rate = ChromosomeGeneticMap("", default_rate_cm_per_mb=self.default_rate_cm_per_mb).default_rate_cm_per_mb
        object.__setattr__(self, "default_rate_cm_per_mb", rate)
        object.__setattr__(self, "maps", dict(self.maps))

    def for_contig(self, contig) -> ChromosomeGeneticMap:
        """The chromosome's map, or a map-free one with the fallback rate."""
        contig = str(contig)
        if contig in self.maps:
            return self.maps[contig]
        return ChromosomeGeneticMap(contig, default_rate_cm_per_mb=self.default_rate_cm_per_mb)

    def identity(self, contig=None) -> dict:
        if contig is not None:
            return self.for_contig(contig).identity()
        return dict(format=MAP_FORMAT, default_rate_cm_per_mb=self.default_rate_cm_per_mb,
                    chromosomes={name: self.maps[name].identity() for name in sorted(self.maps)})


_COMBINED_HEADER = ["chromosome", "position_bp", "map_cm"]

_SHAPEIT_HEADERS = (
    ["position", "combined_rate(cm/mb)", "genetic_map(cm)"],
    ["position", "rate(cm/mb)", "map(cm)"],
    ["physical_bp", "rate_cm/mb", "cumulative_cm"],
)


def _read_map(path, assigned_contig=None):
    """Read a map file without reordering rows or guessing chromosome labels: {contig: (positions, cM)}.

    Formats: headed combined TSV (chromosome, position_bp, map_cm); headed SHAPEIT single-chromosome maps (three
    header variants; need ``assigned_contig``); headerless four-column PLINK (chromosome, id, cM, bp).
    """
    rows = {}
    mode = None
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if mode is None:
                header = [item.lower() for item in fields]
                if header == _COMBINED_HEADER:
                    mode = "combined"
                    continue
                if header in _SHAPEIT_HEADERS:
                    if assigned_contig is None:
                        raise ValueError(f"{path}:{line_number}: SHAPEIT map needs a chromosome-to-path mapping")
                    mode = "shapeit"
                    continue
                if len(fields) == 4:
                    mode = "plink"
                else:
                    raise ValueError(f"{path}:{line_number}: expected four-column PLINK cM map or explicit map header")
            expected_columns = 4 if mode == "plink" else 3
            if len(fields) != expected_columns:
                raise ValueError(f"{path}:{line_number}: expected {expected_columns} columns for {mode} map")
            try:
                if mode == "plink":
                    contig, position, cm = fields[0], fields[3], fields[2]
                elif mode == "combined":
                    contig, position, cm = fields
                else:
                    contig, position, cm = assigned_contig, fields[0], fields[2]
                    local_rate = float(fields[1])
                    if not np.isfinite(local_rate) or local_rate < 0:
                        raise ValueError("SHAPEIT local rate must be finite and nonnegative")
                position = int(position)
                cm = float(cm)
                if position < 0:
                    raise ValueError("physical position must be nonnegative")
                if not np.isfinite(cm):
                    raise ValueError("cumulative cM must be finite")
                if assigned_contig is not None and contig != assigned_contig:
                    raise ValueError(f"chromosome {contig!r} differs from assigned chromosome {assigned_contig!r}")
                chromosome_rows = rows.setdefault(contig, ([], []))
                if chromosome_rows[0] and position <= chromosome_rows[0][-1]:
                    raise ValueError("physical positions must be strictly increasing (no duplicates)")
                if chromosome_rows[1] and cm < chromosome_rows[1][-1]:
                    raise ValueError("cumulative cM must be nondecreasing")
                chromosome_rows[0].append(position)
                chromosome_rows[1].append(cm)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not rows:
        raise ValueError(f"{path}: map contains no records")
    return rows


def load_genetic_maps(path_or_mapping=None, default_rate_cm_per_mb=DEFAULT_RATE_CM_PER_MB) -> GeneticMapSet:
    """Load a combined or PLINK map file, or {exact contig: single-chromosome map path} (also SHAPEIT).

    None gives a constant-rate map set. Cumulative map input is always in cM (PLINK files whose third column is
    in Morgans must be converted first).
    """
    maps = {}
    if path_or_mapping is not None:
        sources = (path_or_mapping.items() if isinstance(path_or_mapping, Mapping)
                   else [(None, path_or_mapping)])
        for assigned_contig, path in sources:
            assigned_contig = None if assigned_contig is None else str(assigned_contig)
            for contig, (positions, cm) in _read_map(path, assigned_contig).items():
                maps[contig] = ChromosomeGeneticMap(contig, positions, cm, default_rate_cm_per_mb)
    return GeneticMapSet(maps, default_rate_cm_per_mb)
