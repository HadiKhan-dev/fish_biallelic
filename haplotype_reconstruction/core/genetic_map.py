"""Cumulative genetic maps, interpolation and fallback recombination rates."""
from __future__ import annotations


from dataclasses import dataclass, field
from collections.abc import Mapping
import hashlib
import os
import numpy as np


@dataclass
class ChromosomeGeneticMap:
    contig: str
    positions_bp: object = None
    map_cm: object = None
    default_rate_cm_per_mb: float = 5.0
    _slopes: np.ndarray = field(init=False, repr=False)
    _morgans: np.ndarray = field(init=False, repr=False)
    _digest: str = field(init=False, repr=False)

    def __post_init__(self):
        self.contig = str(self.contig)
        self.default_rate_cm_per_mb = float(self.default_rate_cm_per_mb)
        if not np.isfinite(self.default_rate_cm_per_mb) or self.default_rate_cm_per_mb < 0:
            raise ValueError("default_rate_cm_per_mb must be finite and nonnegative")
        if (self.positions_bp is None) != (self.map_cm is None):
            raise ValueError("positions_bp and map_cm must be supplied together")
        self.positions_bp = np.array([] if self.positions_bp is None else self.positions_bp,
                                     dtype=np.float64, copy=True)
        self.map_cm = np.array([] if self.map_cm is None else self.map_cm,
                              dtype=np.float64, copy=True)
        if self.positions_bp.ndim != 1 or self.map_cm.shape != self.positions_bp.shape:
            raise ValueError(f"{self.contig}: map positions and cumulative cM must be matching vectors")
        if len(self.positions_bp):
            if len(self.positions_bp) < 2:
                raise ValueError(f"{self.contig}: a supplied map needs at least two knots")
            if not np.all(np.isfinite(self.positions_bp)) or np.any(self.positions_bp < 0):
                raise ValueError(f"{self.contig}: physical positions must be finite and nonnegative")
            if np.any(np.diff(self.positions_bp) <= 0):
                raise ValueError(f"{self.contig}: physical positions must be strictly increasing (no duplicates)")
            if not np.all(np.isfinite(self.map_cm)) or np.any(np.diff(self.map_cm) < 0):
                raise ValueError(f"{self.contig}: cumulative cM must be finite and nondecreasing")
        self._morgans = self.map_cm / 100.0
        self._slopes = np.diff(self._morgans) / np.diff(self.positions_bp)
        digest = hashlib.sha256()
        digest.update(self.positions_bp.astype('<f8').tobytes())
        digest.update(self.map_cm.astype('<f8').tobytes())
        self._digest = digest.hexdigest()

    @property
    def has_map(self):
        return bool(len(self.positions_bp))

    @property
    def fallback_rate_per_bp(self):
        """Morgans/bp (5 cM/Mb = 5e-8 Morgans/bp)."""
        return self.default_rate_cm_per_mb / 1e8

    def cumulative_morgans(self, positions):
        """Continuous genetic coordinates, preserving the supplied cM origin.

        The fallback slope extends both tails continuously. The left extension
        can be negative if the first knot is labelled zero; distances remain
        nonnegative for ordered intervals. Scalar input gives scalar output.
        """
        positions = np.asarray(positions, dtype=np.float64)
        if not self.has_map:
            return positions * self.fallback_rate_per_bp
        result = np.interp(positions, self.positions_bp, self._morgans)
        result = np.where(positions < self.positions_bp[0],
                          self._morgans[0] + (positions - self.positions_bp[0]) * self.fallback_rate_per_bp,
                          result)
        return np.where(positions > self.positions_bp[-1],
                        self._morgans[-1] + (positions - self.positions_bp[-1]) * self.fallback_rate_per_bp,
                        result)

    def interval_morgans(self, left, right):
        """Integrated distance for ordered intervals; endpoints can broadcast."""
        left, right = np.broadcast_arrays(np.asarray(left, dtype=np.float64),
                                         np.asarray(right, dtype=np.float64))
        if np.any(right < left):
            raise ValueError("interval right endpoints must not precede left endpoints")
        if not self.has_map:
            return (right - left) * self.fallback_rate_per_bp
        return self.cumulative_morgans(right) - self.cumulative_morgans(left)

    def rate_per_bp(self, positions):
        """Local right-hand slope in Morgans/bp, for reporting, not integration.

        At interior knots use the interval to the right. At the final knot use
        the right-tail fallback rate. The value at a single knot has zero mass.
        """
        positions = np.asarray(positions, dtype=np.float64)
        if not self.has_map:
            return np.full_like(positions, self.fallback_rate_per_bp)
        indices = np.searchsorted(self.positions_bp, positions, side='right') - 1
        inside = (indices >= 0) & (indices < len(self._slopes))
        return np.where(inside, self._slopes[np.clip(indices, 0, len(self._slopes) - 1)],
                        self.fallback_rate_per_bp)

    def inverse_morgans(self, values):
        """Invert cumulative coordinates for continuous-position simulation.

        A finite plateau tie returns its leftmost knot. Coordinates strictly
        above that plateau start after its right edge, so a continuous draw has
        zero probability of landing in a zero-rate interval. With zero fallback
        slope, values outside the map's genetic range cannot be inverted.
        For a wholly constant map, its sole attainable coordinate maps to zero
        (no supplied map) or the first knot (supplied map).
        """
        values = np.asarray(values, dtype=np.float64)
        rate = self.fallback_rate_per_bp
        if not self.has_map:
            if rate == 0:
                if np.any(values != 0):
                    raise ValueError("cannot invert unattainable coordinates with zero recombination rate")
                return np.zeros_like(values)
            return values / rate
        low = values < self._morgans[0]
        high = values > self._morgans[-1]
        if rate == 0 and np.any(low | high):
            raise ValueError("cannot invert coordinates beyond map endpoints with zero fallback rate")
        right = np.searchsorted(self._morgans, values, side='left')
        right = np.clip(right, 0, len(self._morgans) - 1)
        left = np.maximum(right - 1, 0)
        span = self._morgans[right] - self._morgans[left]
        fraction = np.divide(values - self._morgans[left], span,
                             out=np.zeros_like(values), where=span > 0)
        result = self.positions_bp[left] + fraction * (self.positions_bp[right] - self.positions_bp[left])
        result = np.where(values == self._morgans[right], self.positions_bp[right], result)
        if rate > 0:
            result = np.where(low, self.positions_bp[0] + (values - self._morgans[0]) / rate, result)
            result = np.where(high, self.positions_bp[-1] + (values - self._morgans[-1]) / rate, result)
        return result

    def record(self):
        """Portable constructor arguments for checkpointed map semantics."""
        return dict(contig=self.contig,
                    positions_bp=self.positions_bp.tolist() if self.has_map else None,
                    map_cm=self.map_cm.tolist() if self.has_map else None,
                    default_rate_cm_per_mb=self.default_rate_cm_per_mb)

    def identity(self):
        """Small semantic checkpoint/provenance record, independent of file path."""
        return dict(format='piecewise_linear_cm_v1', contig=self.contig,
                    default_rate_cm_per_mb=self.default_rate_cm_per_mb,
                    knots=len(self.positions_bp), map_sha256=self._digest if self.has_map else None)


def poisson_switch_stay_terms(
    marker_positions,
    recombination_rate,
    *,
    probability_floor=1e-15,
    probability_cap=0.5,
    chromosome_map=None,
):
    """Return switch probabilities and their switch/stay log costs.

    Distances are measured between adjacent marker positions, with a zero
    distance for the first marker.  The switch probability is
    ``1 - exp(-distance * recombination_rate)`` clipped to the supplied
    numerical bounds.  This is the exact convention historically shared by
    painting, smart pedigree scoring, and recombination-map validation.
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
    probabilities = np.clip(
        1.0 - np.exp(-mass),
        probability_floor,
        probability_cap,
    )
    return (
        probabilities,
        np.log(probabilities),
        np.log(1.0 - probabilities),
    )


@dataclass
class GeneticMapSet:
    maps: dict = field(default_factory=dict)
    default_rate_cm_per_mb: float = 5.0

    def __post_init__(self):
        # Validate the public scalar even when every requested chromosome has a map.
        self.default_rate_cm_per_mb = ChromosomeGeneticMap(
            '', default_rate_cm_per_mb=self.default_rate_cm_per_mb).default_rate_cm_per_mb

    def for_contig(self, contig):
        """Select by exact chromosome name; absent chromosomes use the fallback."""
        contig = str(contig)
        if contig in self.maps:
            return self.maps[contig]
        return ChromosomeGeneticMap(contig, default_rate_cm_per_mb=self.default_rate_cm_per_mb)

    def identity(self, contig=None):
        if contig is not None:
            return self.for_contig(contig).identity()
        return dict(format='piecewise_linear_cm_v1',
                    default_rate_cm_per_mb=self.default_rate_cm_per_mb,
                    chromosomes={name: self.maps[name].identity() for name in sorted(self.maps)})


def _read_map(path, assigned_contig=None):
    """Read external text without reordering rows or guessing chromosome labels."""
    rows = {}
    mode = None
    combined_header = ['chromosome', 'position_bp', 'map_cm']
    shapeit_headers = (
        ['position', 'combined_rate(cm/mb)', 'genetic_map(cm)'],
        ['position', 'rate(cm/mb)', 'map(cm)'],
        ['physical_bp', 'rate_cm/mb', 'cumulative_cm'],
    )
    with open(path, encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            fields = line.split()
            if mode is None:
                header = [item.lower() for item in fields]
                if header == combined_header:
                    mode = 'combined'
                    continue
                if header in shapeit_headers:
                    if assigned_contig is None:
                        raise ValueError(f"{path}:{line_number}: SHAPEIT map needs a chromosome-to-path mapping")
                    mode = 'shapeit'
                    continue
                if len(fields) == 4:
                    mode = 'plink'
                else:
                    raise ValueError(f"{path}:{line_number}: expected four-column PLINK cM map or explicit map header")
            expected_columns = 4 if mode == 'plink' else 3
            if len(fields) != expected_columns:
                raise ValueError(f"{path}:{line_number}: expected {expected_columns} columns for {mode} map")
            try:
                if mode == 'plink':
                    contig, position, cm = fields[0], fields[3], fields[2]
                elif mode == 'combined':
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


def load_genetic_maps(path_or_mapping=None, default_rate_cm_per_mb=5.0):
    """Load a combined PLINK/TSV map, or {exact_contig: single_chromosome_path}.

    None gives a constant-rate map set. Cumulative map input is always in cM;
    legacy PLINK files whose third column is in Morgans must be converted first.
    A mapping additionally supports headed SHAPEIT single-chromosome maps.
    """
    result = GeneticMapSet(default_rate_cm_per_mb=default_rate_cm_per_mb)
    if path_or_mapping is None:
        return result
    sources = path_or_mapping.items() if isinstance(path_or_mapping, Mapping) else [(None, path_or_mapping)]
    for assigned_contig, path in sources:
        assigned_contig = None if assigned_contig is None else str(assigned_contig)
        for contig, (positions, cm) in _read_map(path, assigned_contig).items():
            result.maps[contig] = ChromosomeGeneticMap(contig, positions, cm, default_rate_cm_per_mb)
    return result


def load_genetic_maps_from_environment():
    """Read BHD_RECOMBINATION_MAP and BHD_RECOMBINATION_RATE_CM_PER_MB."""
    return load_genetic_maps(os.environ.get('BHD_RECOMBINATION_MAP') or None,
                             float(os.environ.get('BHD_RECOMBINATION_RATE_CM_PER_MB', '5.0')))
