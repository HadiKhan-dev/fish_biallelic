"""Estimate the working map's fallback rate from painted sib frames and decide how the material regime uses it.

The late (painting) estimate er (DESIGN_MAPSCALE rev 5, release form DESIGN_RELEASE): one genome-wide rate r (cM/Mb)
maximising the composite likelihood of every frame sequence's error-run chain (map_scale_kernels.error_run_loglik;
nuisances: the misfit weight delta, fitted first on the 4-state chain, then the error-run start and end
probabilities). It is fitted on the map-free contigs only, the stretches whose distances the fallback rate sets; file
contigs give the file-map ratio diagnostic instead (the same chain on file-map coordinates).

- Family consistency (rev 5 R5.2-R5.3): z_f = d_f / sqrt(phi_f (v_f + v_rest)) with d_f = log r_f - log r_(-f), v the
  profile-curvature variances of log r, phi_f the family-within-contig deviance per degree of freedom of the other
  families (floored at 1), referred to t(df) at a Bonferroni level; the most deviant family is dropped above
  drop_margin z_crit (three or more families) and the test repeats; any remaining |z| > z_crit makes the estimate
  inconsistent. Contigs are the strata of the dispersion only; no count of contigs decides anything.
- Interval: a crossed percentile bootstrap of families and equal-material units (MaterialParameters.unit_morgans of
  the working map along the map-free contigs in configured order, the pedigree's convention), from the exact split
  of every sequence's forward log-likelihood into per-unit predictive terms. It replaces rev 5's contig bootstrap and
  its >= 5-contig rule (owner, 13 Nov 2026).
- Adopted: not at a grid edge, >= minimum_events expected crossovers on the kept families, half-width <=
  maximum_half_width, >= 2 families with frames, consistent. Usable (may veto): consistent, >= 2 families, not at a
  grid edge, >= usable_minimum_events, interval lower end above the grid floor.
- Route (mode): ignore -> working_map; veto -> both_maps when adopted or usable; adopt -> estimate when adopted and
  the interval excludes the working rate (test-agree), both_maps when adopted otherwise or usable.

Method: docs/methods/pedigree.md#map-scale.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import t as student_t

from ...contracts.genetic_map import ChromosomeGeneticMap
from ...contracts.map_scale import MapScaleEstimate
from ..shared.material import unit_count
from .map_scale_frames import Frames, GuardResult, argmax_refined
from .map_scale_kernels import error_run_loglik, frame_loglik
from .parameters import MapScaleParameters, MaterialParameters

__all__ = ["MapScaleFit", "fit", "estimate_record", "route_of", "consistency", "joint_interval", "unit_segments"]

_PER_BP = 1e-8  # cM/Mb to Morgans per bp
_FILE_SCALE = 1e6  # file-map coordinates as pseudo-bp: 1 cM -> 1e6, so a "rate" of 1 cM/Mb is the file map itself


@dataclass(frozen=True)
class MapScaleFit:
    """One fit: the estimate, its tables and the gate.

    cells: [C, F, R, N] log-likelihood of each (contig, family) on the error-run grid (N = start x end nuisances);
    unit_cells: [U, F, R, N] the same split by material unit; families: family ids (F); kept: kept family indices."""

    estimate: float
    at_edge: bool
    interval: tuple[float, float] | None
    events: float
    families: tuple[int, ...]
    kept: tuple[int, ...]
    dropped: tuple[tuple[int, float], ...]
    z_rows: tuple[tuple[int, float, float, float, float, int, float], ...]
    z_crit: float
    consistent: bool
    near: bool
    family_estimate: tuple[float, ...]
    family_events: tuple[float, ...]
    units: int
    frame_morgans: float
    frames: int
    targets: int
    contigs: tuple[int, ...]
    cells: np.ndarray
    unit_cells: np.ndarray
    grid: np.ndarray
    delta: float


def _max_refined(profile: np.ndarray) -> float:
    """Maximum of a 1-D log-likelihood profile on the log grid (parabola through the 3 points at the maximum)."""
    i = int(np.argmax(profile))
    if i == 0 or i == len(profile) - 1:
        return float(profile[i])
    y0, y1, y2 = profile[i - 1], profile[i], profile[i + 1]
    den = y0 - 2 * y1 + y2
    if den >= 0:
        return float(y1)
    s = 0.5 * (y0 - y2) / den
    return float(y1 - 0.25 * (y0 - y2) * s)


def _curvature_variance(table: np.ndarray, grid: np.ndarray) -> float:
    """var(log r) from the profile log-likelihood curvature at its maximum (nuisance profiled)."""
    profile = table.reshape(table.shape[0], -1).max(axis=1)
    i = int(np.argmax(profile))
    if i == 0 or i == len(grid) - 1:
        return math.inf
    h = math.log(grid[1]) - math.log(grid[0])
    c = (profile[i - 1] - 2 * profile[i] + profile[i + 1]) / h ** 2
    return -1.0 / c if c < 0 else math.inf


def _dispersion(cells: np.ndarray, families: Sequence[int], nuisance: int) -> tuple[float, int]:
    """Family-within-stratum deviance per degree of freedom over ``families`` at nuisance ``nuisance`` (floor 1)."""
    deviance = 0.0
    df = 0
    for c in range(cells.shape[0]):
        rows = [cells[c, k, :, nuisance] for k in families if np.any(cells[c, k] != 0)]
        if len(rows) < 2:
            continue
        deviance += 2.0 * (sum(_max_refined(x) for x in rows) - _max_refined(np.sum(rows, axis=0)))
        df += len(rows) - 1
    if df == 0:
        return 1.0, 0
    return max(1.0, deviance / df), df


def consistency(cells: np.ndarray, grid: np.ndarray, p: MapScaleParameters):
    """Rev-5 family consistency on [C, F, R, N] tables. Returns kept indices, dropped [(index, z)], the last round's
    rows (index, z, d, sd, phi, df, z_crit), z_crit, consistent, near."""
    C, F, R = cells.shape[:3]
    fam = [cells[:, f].reshape(C, R, -1) for f in range(F)]
    total = sum(fam)
    nuisance = int(np.unravel_index(int(np.argmax(total.sum(axis=0))), total.shape[1:])[1])
    flat = cells.reshape(C, F, R, -1)
    active = list(range(F))
    dropped: list[tuple[int, float]] = []
    rows: list[tuple] = []
    z_crit, near = math.nan, False
    while len(active) >= 2:
        pooled = sum(fam[f] for f in active)
        phi_all = _dispersion(flat, active, nuisance)
        rows = []
        for f in active:
            rest = pooled - fam[f]
            ef = argmax_refined(fam[f].sum(axis=0), grid)[0]
            er_ = argmax_refined(rest.sum(axis=0), grid)[0]
            d = math.log(ef) - math.log(er_)
            phi, df = _dispersion(flat, [k for k in active if k != f], nuisance) if len(active) >= 3 else phi_all
            v = _curvature_variance(fam[f].sum(axis=0), grid) + _curvature_variance(rest.sum(axis=0), grid)
            sd = math.sqrt(phi * v) if math.isfinite(v) else math.inf
            z = d / sd if sd > 0 else math.inf
            crit = float(student_t.ppf(1 - p.consistency_alpha / (2 * len(active)), max(1, df)))
            rows.append((f, z, d, sd, phi, df, crit))
        rows.sort(key=lambda x: 0.0 if math.isnan(x[1]) else -abs(x[1]) / x[6])
        worst, z_worst, z_crit = rows[0][0], rows[0][1], rows[0][6]
        if z_crit / p.drop_margin <= abs(z_worst) <= z_crit * p.drop_margin:
            near = True
        if len(active) >= 3 and abs(z_worst) > z_crit * p.drop_margin:
            active.remove(worst)
            dropped.append((int(worst), round(float(z_worst), 2)))
            continue
        return active, dropped, rows, z_crit, abs(z_worst) <= z_crit, near
    return active, dropped, rows, z_crit, len(active) >= 1, near


def joint_interval(cells: np.ndarray, grid: np.ndarray, p: MapScaleParameters) -> tuple[float, float] | None:
    """Crossed percentile bootstrap of [U, F, R, N] tables: units and families each drawn with replacement, every cell
    weighted by the product of its unit's and family's draw counts; None with fewer than minimum_interval_units
    units or two families."""
    U, F = cells.shape[:2]
    if U < max(2, p.minimum_interval_units) or F < 2:
        return None
    rng = np.random.default_rng(p.bootstrap_seed)
    flat = cells.reshape(U, F, cells.shape[2], -1)
    draws = np.empty(p.bootstrap_replicates)
    for b in range(p.bootstrap_replicates):
        wu = np.bincount(rng.integers(0, U, U), minlength=U).astype(float)
        wf = np.bincount(rng.integers(0, F, F), minlength=F).astype(float)
        draws[b] = argmax_refined(np.einsum("u,f,ufrd->rd", wu, wf, flat), grid)[0]
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def unit_segments(contig: np.ndarray, coordinate: np.ndarray, offsets: np.ndarray, material: MaterialParameters):
    """Material units of the frames and the segments of each sequence.

    ``coordinate``: each frame's working-map coordinate (Morgans) on its contig; ``contig``: each frame's contig
    index (configured order). The genome axis concatenates the contigs' frame spans in configured order (no gap
    between consecutive contigs); it is cut every unit_morgans, a final piece shorter than tail_morgans joining the
    unit before it (science.shared.material). Returns (unit of each frame, dense; segment of each frame; segment unit;
    number of units; axis Morgans). Segments number each sequence's runs of one unit, sequence by sequence."""
    axis = np.empty(len(coordinate))
    start = 0.0
    for c in np.unique(contig):
        mask = contig == c
        low, high = float(coordinate[mask].min()), float(coordinate[mask].max())
        axis[mask] = start + (coordinate[mask] - low)
        start += high - low
    count = unit_count(start, material.unit_morgans, material.tail_morgans)
    nominal = np.minimum(np.floor(axis / material.unit_morgans).astype(np.int64), count - 1)
    occupied, dense = np.unique(nominal, return_inverse=True)
    segment = np.empty(len(dense), np.int64)
    segment_unit: list[int] = []
    for s in range(len(offsets) - 1):
        lo, hi = int(offsets[s]), int(offsets[s + 1])
        for x in range(lo, hi):
            if x == lo or dense[x] != dense[x - 1]:
                segment_unit.append(int(dense[x]))
            segment[x] = len(segment_unit) - 1
    return dense.astype(np.int64), segment, np.asarray(segment_unit, np.int64), len(occupied), float(start)


def _sum_at(shape: tuple[int, ...], index: tuple[np.ndarray, ...], values: np.ndarray) -> np.ndarray:
    """An array of ``shape`` with out[index[0][i], index[1][i], ...] += values[i] for every i (what np.add.at
    computes), summed by a stable sort on the target cell and np.add.reduceat: faster than np.add.at (measured 1.0-1.8
    times at the fit's shapes); sums may differ from np.add.at's in the last bits (another summation order)."""
    head = shape[: len(index)]
    out = np.zeros(shape)
    if not len(values):
        return out
    flat = np.ravel_multi_index(index, head)
    order = np.argsort(flat, kind="stable")
    keys = flat[order]
    starts = np.flatnonzero(np.concatenate(([True], keys[1:] != keys[:-1])))
    out.reshape((-1,) + shape[len(index):])[keys[starts]] = np.add.reduceat(values[order], starts, axis=0)
    return out


def fit(frames: Frames, coordinate_maps: Mapping[int, ChromosomeGeneticMap] | None, p: MapScaleParameters,
        material: MaterialParameters, *, positions_scale: float = 1.0) -> MapScaleFit | None:
    """Fit er on ``frames`` (positions already in the chain's coordinate: bp, or file-map pseudo-bp) and apply the
    gate. ``coordinate_maps``: per contig index, the working map that places frames on the material axis (None: the
    positions are file-map pseudo-bp, whose Morgans are position / (100 positions_scale))."""
    if not len(frames):
        return None
    offsets, pos, em = frames.stacked()
    grid_fine = p.rate_grid_values()
    grid = p.error_run_grid()
    deltas = np.asarray(p.misfit_weights, dtype=np.float64)
    ll_p1 = frame_loglik(offsets, pos, em, grid_fine, deltas)
    delta = float(deltas[argmax_refined(ll_p1.sum(axis=0), grid_fine)[1]])
    del ll_p1
    contig_of_frame = np.repeat(np.asarray(frames.contig, np.int64), np.diff(offsets))
    if coordinate_maps is None:
        coordinate = pos / (100.0 * positions_scale)
    else:
        coordinate = np.empty(len(pos))
        for c in np.unique(contig_of_frame):
            mask = contig_of_frame == c
            coordinate[mask] = coordinate_maps[int(c)].cumulative_morgans(pos[mask])
    _unit, segment, segment_unit, units, axis_morgans = unit_segments(contig_of_frame, coordinate, offsets, material)
    starts = np.asarray(p.error_run_starts, dtype=np.float64)
    ends = np.asarray(p.error_run_ends, dtype=np.float64)
    seg_ll = error_run_loglik(offsets, pos, em, segment, int(len(segment_unit)), grid, delta, starts, ends)
    seg_ll = seg_ll.reshape(seg_ll.shape[0], seg_ll.shape[1], -1)
    # segment -> sequence -> (contig, family)
    seq_of_segment = np.empty(len(segment_unit), np.int64)
    for s in range(len(offsets) - 1):
        seq_of_segment[segment[offsets[s]]:segment[offsets[s + 1] - 1] + 1] = s
    families = tuple(sorted(set(frames.family)))
    contigs = tuple(sorted(set(frames.contig)))
    fpos = {f: k for k, f in enumerate(families)}
    cpos = {c: k for k, c in enumerate(contigs)}
    seq_family = np.asarray([fpos[f] for f in frames.family], np.int64)
    seq_contig = np.asarray([cpos[c] for c in frames.contig], np.int64)
    cells = _sum_at((len(contigs), len(families)) + seg_ll.shape[1:],
                    (seq_contig[seq_of_segment], seq_family[seq_of_segment]), seg_ll)
    unit_cells = _sum_at(
        (units, len(families)) + seg_ll.shape[1:], (segment_unit, seq_family[seq_of_segment]), seg_ll
    )
    kept, dropped, rows, z_crit, consistent, near = consistency(cells, grid, p)
    sub = unit_cells[:, kept]
    estimate, _n, at_edge = argmax_refined(sub.sum(axis=(0, 1)), grid)
    interval = joint_interval(sub, grid, p)
    span = np.asarray([float(x[-1] - x[0]) for x in frames.positions])
    span_family = np.zeros(len(families))
    np.add.at(span_family, seq_family, span)
    meioses = 2.0  # each sequence carries its target's two transmissions
    events = meioses * float(sum(span_family[k] for k in kept)) * _PER_BP * estimate
    family_estimate = tuple(argmax_refined(cells[:, k].sum(axis=0), grid)[0] for k in range(len(families)))
    family_events = tuple(meioses * span_family[k] * _PER_BP * family_estimate[k] for k in range(len(families)))
    return MapScaleFit(
        float(estimate), bool(at_edge), interval, float(events), families, tuple(int(k) for k in kept),
        tuple(dropped), tuple(tuple(r) for r in rows), float(z_crit), bool(consistent), bool(near), family_estimate,
        family_events, int(units), axis_morgans, int(len(pos)), int(len(set(frames.target))), contigs, cells,
        unit_cells, grid, delta,
    )


def gate(value: MapScaleFit | None, p: MapScaleParameters) -> tuple[bool, str, bool, str]:
    """(adopted, reason, usable, reason) of a fit (module docstring)."""
    if value is None:
        return False, "no frames on the map-free contigs", False, "no frames on the map-free contigs"
    half = None if value.interval is None else (value.interval[1] - value.interval[0]) / 2 / value.estimate
    families = len(value.kept)
    floor = float(value.grid[0])
    if value.at_edge:
        adopted, reason = False, "estimate at the grid edge"
    elif value.events < p.minimum_events:
        adopted, reason = False, f"{value.events:.0f} expected crossovers < {p.minimum_events:g}"
    elif half is None:
        adopted, reason = False, f"no interval (fewer than {p.minimum_interval_units} material units or one family)"
    elif half > p.maximum_half_width:
        adopted, reason = False, f"95 % half-width {half:.0%} > {p.maximum_half_width:.0%}"
    elif families < 2 or not value.consistent:
        adopted, reason = False, "inconsistent families" if families >= 2 else "fewer than two families"
    else:
        adopted, reason = True, "gate passed"
    if adopted:
        return True, reason, True, "adopted"
    if families < 2 or not value.consistent:
        usable, why = False, "inconsistent families" if families >= 2 else "fewer than two families"
    elif value.at_edge:
        usable, why = False, "estimate at the grid edge"
    elif value.events < p.usable_minimum_events:
        usable, why = False, f"{value.events:.0f} expected crossovers < {p.usable_minimum_events:g}"
    elif value.interval is None or value.interval[0] <= floor * (1.0 + 1e-9):
        usable, why = False, "interval reaches the grid floor"
    else:
        usable, why = True, "consistent estimate with enough expected crossovers"
    return False, reason, usable, why


def route_of(mode: str, adopted: bool, usable: bool, interval: tuple[float, float] | None,
             working_rate: float) -> tuple[str, str]:
    """The pedigree route of the estimate under ``mode`` (module docstring)."""
    if mode == "ignore":
        return "working_map", "rate_estimate = ignore: the estimate is a diagnostic"
    if not (adopted or usable):
        return "working_map", "no usable estimate"
    if mode == "veto":
        return "both_maps", "rate_estimate = veto: release only what both maps' regimes release"
    assert interval is not None
    if adopted and not (interval[0] <= working_rate <= interval[1]):
        return "estimate", "rate_estimate = adopt: the adopted estimate's interval excludes the working rate"
    if adopted:
        return "both_maps", "rate_estimate = adopt: the adopted estimate's interval covers the working rate"
    return "both_maps", "rate_estimate = adopt: the estimate is usable but not adopted"


def _text(value: float) -> str:
    return f"{value:.3g}"


def estimate_record(value: MapScaleFit | None, file_value: MapScaleFit | None, guard_result: GuardResult | None,
                    p: MapScaleParameters, *, working_rate: float, map_free: Sequence[str],
                    file_contigs: Sequence[str]) -> MapScaleEstimate:
    """The run record: estimate, gate, route, family rows, file-map ratio and the warning text."""
    adopted, gate_reason, usable, usable_reason = gate(value, p)
    if not map_free:
        adopted, usable = False, False
        gate_reason = usable_reason = "every contig has a map file"
    interval = None if value is None else value.interval
    route, route_reason = route_of(p.mode, adopted, usable, interval, working_rate)
    nan = float("nan")
    warnings = []
    if value is not None and interval is not None and (adopted or usable):
        if not (interval[0] <= working_rate <= interval[1]):
            warnings.append(
                f"the painting estimate of the fallback rate {_text(value.estimate)} cM/Mb, 95 % interval "
                f"[{interval[0]:.2f}, {interval[1]:.2f}], excludes the working fallback rate {_text(working_rate)} cM/Mb"
                + {"working_map": " (rate_estimate = ignore: nothing changes)",
                   "both_maps": " (the pedigree releases only what both maps' regimes release)",
                   "estimate": " (the pedigree applies the estimated map's regime)"}[route]
            )
    file_ratio = file_low = file_high = file_events = nan
    if file_value is not None:
        file_ratio, file_events = file_value.estimate, file_value.events
        if file_value.interval is not None:
            file_low, file_high = file_value.interval
            if not (file_low <= 1.0 <= file_high):
                warnings.append(
                    f"the painted crossover rate on the map-file contigs is {file_ratio:.2f} x the map file's "
                    f"(95 % interval [{file_low:.2f}, {file_high:.2f}]): the map file may not fit this cross; map "
                    "files are always followed"
                )
    rows = value.families if value is not None else ()
    status = []
    z = np.full(len(rows), nan)
    if value is not None:
        for k in range(len(rows)):
            status.append("used" if k in value.kept else "dropped")
        for r in value.z_rows:
            z[r[0]] = r[1]
        for k, zz in value.dropped:
            z[k] = zz
    top = value.z_rows[0] if value is not None and value.z_rows else None
    return MapScaleEstimate(
        mode=p.mode,
        working_rate_cm_per_mb=float(working_rate),
        map_free_contigs=tuple(map_free),
        file_contigs=tuple(file_contigs),
        estimate_cm_per_mb=nan if value is None else value.estimate,
        interval_low_cm_per_mb=nan if interval is None else interval[0],
        interval_high_cm_per_mb=nan if interval is None else interval[1],
        at_grid_edge=False if value is None else value.at_edge,
        expected_events=0.0 if value is None else value.events,
        material_units=0 if value is None else value.units,
        frame_morgans=0.0 if value is None else value.frame_morgans,
        frames=0 if value is None else value.frames,
        targets=0 if value is None else value.targets,
        consistent=False if value is None else value.consistent,
        maximum_z=nan if top is None else abs(float(top[1])),
        critical_z=nan if value is None else value.z_crit,
        dispersion=nan if top is None else float(top[4]),
        adopted=bool(adopted),
        gate_reason=gate_reason,
        usable=bool(usable),
        usable_reason=usable_reason,
        route=route,
        route_reason=route_reason,
        family=np.asarray(rows, dtype=np.int64),
        family_estimate_cm_per_mb=np.asarray(() if value is None else value.family_estimate, dtype=np.float64),
        family_events=np.asarray(() if value is None else value.family_events, dtype=np.float64),
        family_z=z,
        family_status=tuple(status),
        guard_pair_excluded=0 if guard_result is None else len(guard_result.excluded_pair),
        guard_target_removed=0 if guard_result is None else len(guard_result.excluded_target),
        file_ratio=float(file_ratio),
        file_ratio_low=float(file_low),
        file_ratio_high=float(file_high),
        file_ratio_events=float(file_events),
        warning="; ".join(warnings),
    )
