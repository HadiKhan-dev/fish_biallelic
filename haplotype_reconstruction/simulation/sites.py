"""Physical-bp site races and an independent stationary cascade profile.

Prepare a cascade with realise_profile(extent, Profile(...), profile_rng), using
stream('simulation.profile', seed, contig). Pass the result to draw_sites with
stream('simulation.sites', seed, contig); the genealogy uses its own source stream.
Positions-mode arrays are loaded by the caller. This module performs no file I/O.
Extent accepts a record with start_bp/end_bp or a pair of bounds. All genetic
coordinates subtract the map at extent.start_bp, including maps with origins/tails.
"""
from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np

from ..contracts.roles import frozen_array, record_field, schema_field
from ..core.errors import InputError
from ..core.kernels import kernel
from .parameters import (PROFILE_MASKED_FRACTION, PROFILE_MASKED_MEAN_BP, PROFILE_SCALES_BP,
                         PROFILE_SIGMAS, SITE_BISECTION_STEPS, SITE_CHUNK_BP)

__all__ = ['Profile', 'RealisedProfile', 'SiteMode', 'realise_profile', 'profile_values',
           'weight_chunks', 'draw_sites', 'draw_site_set', 'relative_morgans']


@dataclass(frozen=True, eq=False)
class Profile:
    """Cascade parameters: log-intensity SDs, stationary covered share and mean stretch bp."""

    kind: str = record_field('state', dtype='str', default='none')
    sigmas: np.ndarray = record_field('state', axes=('scale',), dtype='float64',
                                      default_factory=lambda: np.array(PROFILE_SIGMAS))
    masked_fraction: float = record_field('state', dtype='float64', default=PROFILE_MASKED_FRACTION)
    masked_mean_bp: float = record_field('state', units='bp', dtype='float64', default=PROFILE_MASKED_MEAN_BP)
    schema: int = schema_field(1)

    def __post_init__(self):
        object.__setattr__(self, 'sigmas', frozen_array(self.sigmas, np.float64))
        if self.kind not in ('none', 'cascade'):
            raise InputError('profile kind must be none or cascade')
        if self.sigmas.shape != (5,) or np.any(~np.isfinite(self.sigmas)) or np.any(self.sigmas < 0):
            raise InputError('profile sigmas must be five finite nonnegative log-intensity standard deviations')
        if not np.isfinite(self.masked_fraction) or not 0 <= self.masked_fraction < 1:
            raise InputError('masked_fraction must be finite and in [0, 1)')
        if not np.isfinite(self.masked_mean_bp) or self.masked_mean_bp <= 0:
            raise InputError('masked_mean_bp must be positive and finite (bp)')


@dataclass(frozen=True, eq=False)
class RealisedProfile:
    """Profile draws, made once on the independent profile stream; immutable across chunks."""

    specification: Profile = record_field('state', dtype='Profile')
    bounds: tuple[float, float] = record_field('state', axes=('bound',), units='bp', dtype='float64')
    first_cell: np.ndarray = record_field('state', axes=('scale',), dtype='int64')
    log_cells: tuple[np.ndarray, ...] = record_field('state', axes=('scale', 'cell'), dtype='float64')
    mask_starts: np.ndarray = record_field('state', axes=('stretch',), units='bp', dtype='float64')
    mask_ends: np.ndarray = record_field('state', axes=('stretch',), units='bp', dtype='float64')
    schema: int = schema_field(1)

    def __post_init__(self):
        for name, dtype in (('first_cell', np.int64), ('mask_starts', np.float64), ('mask_ends', np.float64)):
            object.__setattr__(self, name, frozen_array(getattr(self, name), dtype))
        object.__setattr__(self, 'log_cells', tuple(frozen_array(a, np.float64) for a in self.log_cells))


@dataclass(frozen=True, eq=False)
class SiteMode:
    """One mode; positions are the caller-loaded int64 bp vector, never a path."""

    count: int | None = record_field('state', units='sites', dtype='int', default=None)
    per_mb: float | None = record_field('state', units='sites/Mb', dtype='float64', default=None)
    positions: np.ndarray | None = record_field('state', axes=('site',), units='bp', dtype='int64', default=None)
    schema: int = schema_field(1)

    def __post_init__(self):
        if sum(v is not None for v in (self.count, self.per_mb, self.positions)) != 1:
            raise InputError('exactly one of count, per_mb or positions is required')
        if self.count is not None and (isinstance(self.count, (bool, np.bool_)) or
                                      not isinstance(self.count, (int, np.integer)) or self.count < 1):
            raise InputError('count must be a positive integer')
        if self.per_mb is not None and (not np.isfinite(self.per_mb) or self.per_mb <= 0):
            raise InputError('per_mb must be positive and finite')
        if self.positions is not None:
            raw = np.asarray(self.positions)
            if raw.ndim != 1 or raw.dtype.kind not in 'iu' or np.any(raw[1:] <= raw[:-1]):
                raise InputError('positions must be a sorted unique integer bp vector')
            object.__setattr__(self, 'positions', frozen_array(raw, np.int64))


def _bounds(extent):
    start, end = ((extent.start_bp, extent.end_bp) if hasattr(extent, 'start_bp') else extent)
    start, end = float(start), float(end)
    if not np.isfinite(start) or not np.isfinite(end) or start < 1 or end < start or end >= np.iinfo(np.int64).max:
        raise InputError('extent requires finite ordered bp bounds starting at or above 1')
    return start, end


def relative_morgans(extent, chromosome_map, positions):
    """Forward map only: equal coordinates on plateaus are retained."""
    start, _ = _bounds(extent)
    return np.asarray(chromosome_map.cumulative_morgans(positions) - chromosome_map.cumulative_morgans(start),
                      dtype=np.float64)


def realise_profile(extent, profile: Profile, rng: np.random.Generator) -> RealisedProfile:
    """Draw cell normals and exact stationary Poisson Boolean stretches, without bp arrays."""
    start, end = _bounds(extent)
    if not isinstance(rng, np.random.Generator):
        raise InputError('profile requires a numpy Generator')
    first, cells = [], []
    if profile.kind == 'cascade':
        for scale, sigma in zip(PROFILE_SCALES_BP, profile.sigmas):
            a, b = int(np.floor((start - 1) / scale)), int(np.floor((end - 1) / scale))
            first.append(a)
            cells.append(sigma * rng.standard_normal(b - a + 1) - sigma * sigma / 2)
    starts, ends = [], []
    if profile.kind == 'cascade' and profile.masked_fraction > 0:
        hazard = -np.log1p(-profile.masked_fraction)
        rate = hazard / profile.masked_mean_bp
        # Conditional residual lifetimes of all stretches already covering the left boundary.
        k = rng.poisson(hazard)
        for residual in rng.exponential(profile.masked_mean_bp, k):
            starts.append(start)
            ends.append(start + residual)
        x = start
        while True:
            x += rng.exponential(1 / rate)
            if x > end:
                break
            starts.append(x)
            ends.append(x + rng.exponential(profile.masked_mean_bp))
    # Store a union so membership takes one binary search, including overlapping stretches.
    ms, me = [], []
    for i in np.argsort(np.asarray(starts)):
        a, b = starts[i], ends[i]
        if me and a <= me[-1]:
            me[-1] = max(me[-1], b)
        else:
            ms.append(a)
            me.append(b)
    return RealisedProfile(profile, (start, end), np.array(first, np.int64), tuple(cells),
                           np.array(ms), np.array(me))


def profile_values(profile, positions):
    """m(y), with cascade cell origin floor((y-1)/scale) and half-open mask intervals."""
    y = np.asarray(positions)
    spec = profile.specification if isinstance(profile, RealisedProfile) else profile
    if spec.kind == 'none':
        return np.ones(y.shape)
    if not isinstance(profile, RealisedProfile):
        raise InputError('prepare cascade with realise_profile on its independent profile stream')
    if np.any(y < profile.bounds[0]) or np.any(y > profile.bounds[1]):
        raise InputError('profile does not cover the requested bp positions')
    logm = np.zeros(y.shape)
    for i, scale in enumerate(PROFILE_SCALES_BP):
        cell = np.floor((y - 1) / scale).astype(np.int64) - profile.first_cell[i]
        logm += profile.log_cells[i][cell]
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        m = np.exp(logm)
    if len(profile.mask_starts):
        idx = np.searchsorted(profile.mask_starts, y, side='right') - 1
        masked = (idx >= 0) & (y < profile.mask_ends[np.maximum(idx, 0)])
        m[masked] = 0
    return m


def _validate_weights(extent, chromosome_map, starts_morgans, S, profile, chunk_size):
    start, end = _bounds(extent)
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise InputError('chunk_size must be a positive integer (bp)')
    starts, sep = np.asarray(starts_morgans, dtype=np.float64), np.asarray(S, dtype=np.float64)
    length = float(chromosome_map.interval_morgans(start, end))
    if not np.isfinite(length) or length < 0:
        raise InputError('map length over the extent must be finite and nonnegative (Morgans)')
    if (starts.ndim != 1 or not len(starts) or starts[0] != 0 or sep.shape != starts.shape or
            np.any(~np.isfinite(starts)) or np.any(np.diff(starts) <= 0) or starts[-1] > length):
        raise InputError('separating intervals require increasing relative Morgan starts from 0 within the extent')
    if np.any(~np.isfinite(sep)) or np.any(sep < 0):
        raise InputError('separating lengths must be finite and nonnegative (generations)')
    if isinstance(profile, RealisedProfile) and profile.bounds != (start, end):
        raise InputError('realised profile extent does not match site extent')
    spec = profile.specification if isinstance(profile, RealisedProfile) else profile
    if not isinstance(spec, Profile):
        raise InputError('profile must be a Profile or RealisedProfile')
    if spec.kind == 'cascade' and not isinstance(profile, RealisedProfile):
        raise InputError('prepare cascade with realise_profile on its independent profile stream')
    return int(np.ceil(start)), int(np.floor(end)), starts, sep


def weight_chunks(extent, chromosome_map, starts_morgans, S, profile, *, chunk_size=SITE_CHUNK_BP):
    """Yield (integer bp, finite nonnegative S*m) in ascending physical order."""
    lo, hi, starts, sep = _validate_weights(extent, chromosome_map, starts_morgans, S, profile, chunk_size)
    for a in range(lo, hi + 1, chunk_size):
        y = np.arange(a, min(a + chunk_size, hi + 1), dtype=np.int64)
        x = relative_morgans(extent, chromosome_map, y)
        idx = np.searchsorted(starts, x, side='right') - 1
        with np.errstate(over='ignore', invalid='ignore'):
            w = sep[idx] * profile_values(profile, y)
        if np.any(~np.isfinite(w)):
            raise InputError('site weights S*m must be finite')
        yield y, w


@kernel(overloads=((nb.float64[::1], nb.float64, nb.float64, nb.float64),), cache=True)
def _probability_sum(weights, threshold, total, correction):
    """Sequential compensated sum carried across chunks: exactly the same addition order."""
    for w in weights:
        value = -np.expm1(-threshold * w)
        new = total + value
        if abs(total) >= abs(value):
            correction += (total - new) + value
        else:
            correction += (value - new) + total
        total = new
    return total, correction


def _threshold(chunks, target, maximum):
    def expected(c):
        total, correction = 0.0, 0.0
        for _, w in chunks():
            total, correction = _probability_sum(w, c, total, correction)
        return total + correction
    lo, hi = 0.0, 1.0 / maximum
    if not np.isfinite(hi):
        hi = np.finfo(float).max
    while expected(hi) < target:
        if hi >= np.finfo(float).max / 2:
            hi = np.finfo(float).max
            if expected(hi) < target:
                return _wide_threshold(chunks, target, hi)
            break
        hi *= 2
    for _ in range(SITE_BISECTION_STEPS):
        mid = lo + (hi - lo) / 2
        if expected(mid) < target:
            lo = mid
        else:
            hi = mid
    return lo + (hi - lo) / 2


def _wide_threshold(chunks, target, lower):
    """Extended-range fallback for finite weights requiring c beyond float64.

    Keep sequential compensated additions across chunks, as in the fast kernel.
    This path also keeps tiny positive weights when larger ones have saturated.
    """
    def expected(c):
        total, correction = np.longdouble(0), np.longdouble(0)
        for _, w in chunks():
            for value in -np.expm1(-c * w.astype(np.longdouble)):
                new = total + value
                if abs(total) >= abs(value):
                    correction += (total - new) + value
                else:
                    correction += (value - new) + total
                total = new
        return total + correction
    lo, hi = np.longdouble(lower), 2 * np.longdouble(lower)
    while expected(hi) < target:
        hi *= 2
    for _ in range(SITE_BISECTION_STEPS):
        mid = lo + (hi - lo) / 2
        if expected(mid) < target:
            lo = mid
        else:
            hi = mid
    return lo + (hi - lo) / 2


def _smallest(clocks, positions, count):
    if len(clocks) <= count:
        return clocks, positions
    cutoff = np.partition(clocks, count - 1)[count - 1]
    before = np.flatnonzero(clocks < cutoff)
    ties = np.flatnonzero(clocks == cutoff)
    # Exact floating ties (including infinities) resolve by physical position.
    tied = ties[np.argsort(positions[ties])[:count - len(before)]]
    chosen = np.concatenate((before, tied))
    return clocks[chosen], positions[chosen]


def draw_sites(extent, chromosome_map, starts_morgans, S, profile, mode, rng, *, chunk_size=SITE_CHUNK_BP,
               exclude_positions=()):
    """Return sorted unique positions int64[L] and dropped positions int64[D].

    Every eligible bp consumes one Exp(1)/w clock, in physical order; extended
    precision retains the range of all positive finite float64 weights. Exact machine
    ties resolve by bp order. Chunk size changes allocation only, including in the
    per_mb bisection. Count zero and per_mb endpoints are refused by the design.
    """
    if not isinstance(mode, SiteMode):
        mode = SiteMode(**mode)
    if not isinstance(rng, np.random.Generator):
        raise InputError('site clocks require a numpy Generator')
    lo, hi, starts, sep = _validate_weights(extent, chromosome_map, starts_morgans, S, profile, chunk_size)
    spec = profile.specification if isinstance(profile, RealisedProfile) else profile
    if mode.positions is not None:
        if spec.kind != 'none':
            raise InputError('positions mode requires profile none')
        y = mode.positions
        if np.any(y < lo) or np.any(y > hi):
            raise InputError('listed positions must be within the extent')
        idx = np.searchsorted(starts, relative_morgans(extent, chromosome_map, y), side='right') - 1
        keep = sep[idx] > 0
        return y[keep].copy(), y[~keep].copy()
    def chunks():
        for y, w in weight_chunks(extent, chromosome_map, starts, sep, profile, chunk_size=chunk_size):
            if len(exclude_positions):
                w[np.isin(y, exclude_positions)] = 0
            yield y, w
    # Validate every weight and both capacity refusals before drawing any site clock.
    capacity, maximum = 0, 0.0
    for _, w in chunks():
        capacity += int(np.count_nonzero(w > 0))
        maximum = max(maximum, float(w.max(initial=0)))
    threshold = None
    if mode.count is not None:
        if mode.count > capacity:
            raise InputError(f'count {mode.count} exceeds eligible-position capacity {capacity}')
    else:
        _, end = _bounds(extent)
        target = mode.per_mb * end / 1e6
        if not np.isfinite(target) or not 0 < target < capacity:
            raise InputError(f'per_mb target {target} must be strictly between zero and capacity {capacity}')
        threshold = _threshold(chunks, target, maximum)
    best_c, best_y, retained = np.empty(0, np.longdouble), np.empty(0, np.int64), []
    for y, w in chunks():
        eligible = w > 0
        y, w = y[eligible], w[eligible]
        with np.errstate(over='ignore', under='ignore'):
            clocks = rng.exponential(1.0, len(y)).astype(np.longdouble) / w
        if mode.count is not None:
            best_c, best_y = _smallest(np.concatenate((best_c, clocks)), np.concatenate((best_y, y)), mode.count)
        else:
            retained.append(y[clocks <= threshold])
    positions = np.sort(best_y) if mode.count is not None else np.concatenate(retained)
    return positions, np.empty(0, np.int64)


def draw_site_set(extent, chromosome_map, starts_morgans, S, profile, mode, fraction, rng,
                  *, chunk_size=SITE_CHUNK_BP):
    """Genealogical race followed by a disjoint profile-only race; return the truth mark.

    Count is the final number of sites. Integer allocation uses the nearest number
    of spurious sites (half rounded up). For the other modes the number added is
    nearest to genealogical_count * fraction / (1 - fraction), also half up.
    A spurious position is any unmasked profile-positive bp not already selected;
    it does not require a positive tree term. Zero fraction uses exactly the old race.
    """
    if not np.isfinite(fraction) or not 0 <= fraction < 1:
        raise InputError('spurious_fraction must be finite and in [0, 1)')
    if not isinstance(mode, SiteMode):
        mode = SiteMode(**mode)
    spurious_count = 0
    gene_mode = mode
    if fraction and mode.count is not None:
        spurious_count = int(np.floor(mode.count * fraction + 0.5))
        gene_count = mode.count - spurious_count
        gene_mode = SiteMode(count=gene_count) if gene_count else None
    if gene_mode is None:
        genealogical, dropped = np.empty(0, np.int64), np.empty(0, np.int64)
    else:
        genealogical, dropped = draw_sites(
            extent, chromosome_map, starts_morgans, S, profile, gene_mode, rng, chunk_size=chunk_size
        )
    if fraction and mode.count is None:
        spurious_count = int(np.floor(len(genealogical) * fraction / (1 - fraction) + 0.5))
    if not spurious_count:
        return genealogical, np.zeros(len(genealogical), bool), dropped
    spurious, _ = draw_sites(
        extent, chromosome_map, np.array([0.0]), np.array([1.0]), profile,
        SiteMode(count=spurious_count), rng, chunk_size=chunk_size, exclude_positions=genealogical
    )
    positions = np.sort(np.concatenate((genealogical, spurious)))
    mark = np.isin(positions, spurious)
    return positions, mark, dropped
