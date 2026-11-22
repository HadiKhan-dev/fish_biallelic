"""Check a fitted zero intermittent read loss using repeated nearby depth measurements."""

from __future__ import annotations

from dataclasses import replace
from functools import partial

import numpy as np
from scipy.optimize import minimize


def _row_moments(row, *, depth, short, gaps, quadruples, blocks, pair_blocks, quartet_blocks, distances):
    # Each sample contributes to its own delete-sample summary and to the same physical block partition.
    values = np.asarray(depth[row], dtype=np.float64)
    left, right = values[short], values[short + 1]
    out = np.zeros((blocks, distances * 2 + 3))
    cells = pair_blocks * distances + gaps[short] - 1
    for field, array in enumerate(((left - right) ** 2, left + right)):
        out[:, field * distances:(field + 1) * distances] = np.bincount(
            cells, weights=array, minlength=blocks * distances).reshape(blocks, distances)
    a, b, c, d = (values[quadruples + j] for j in range(4))
    vb, vc = (b - a) * (b - c), (c - b) * (c - d)
    second = vb + vc
    third = vb * (b - d) + vc * (c - a)
    for k, array in enumerate((b + c, second, 3 * second + third)):
        out[:, 2 * distances + k] = np.bincount(quartet_blocks, weights=array, minlength=blocks)
    return out


def _contrasts(moments, distances):
    """Paired zero-gap intercept, quartet loss, excess quality variance, and paired minus triplet contrast."""
    num, den = moments[..., :distances], moments[..., distances:2 * distances]
    bp = np.arange(1, distances + 1)
    w, wx, wxx = den.sum(axis=-1), den @ bp, den @ (bp ** 2)
    # Inverting a rank-two weighted line requires at least two distinct observed spacings.
    divisor = w * wxx - wx * wx
    centre, second, third = np.moveaxis(moments[..., 2 * distances:], -1, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pair = (num.sum(axis=-1) * wxx - wx * (num @ bp)) / divisor
        triplet, u = second / centre, third / (2 * centre)
        quartet = 2 * u / (1 + np.sqrt(1 + 4 * u))
    return np.stack((pair, quartet, u - triplet - triplet ** 2, pair - triplet), axis=-1)


def _cluster_se(parts, total, distances):
    parts = parts[parts[:, distances:].any(axis=1)]
    if len(parts) < 2:
        return np.full(4, np.nan)
    deleted = _contrasts(total - parts, distances)
    return np.sqrt((len(parts) - 1) * np.mean((deleted - deleted.mean(axis=0)) ** 2, axis=0))


def _endpoint_depth(sites, *, depth, rows):
    sums = np.zeros(len(sites))
    for row in rows:
        sums += depth[row, sites]
    return sums


def _intensity_parts(row, *, depth, sites, site_blocks, endpoint_depth, blocks):
    values = depth[row, sites].astype(np.float64)
    endpoints = site_blocks[sites]
    within = values * (values - 1)
    between = values * (endpoint_depth - values)
    matched = tuple(np.bincount(endpoints, weights=field, minlength=blocks)
                    for field in (within, between, values))
    all_depth = np.bincount(site_blocks, weights=depth[row], minlength=blocks)
    return np.stack((*matched, all_depth))


def _intensity_excess(within, between, exposure, total_depth, squared_depth):
    """Excess in Poisson variance units, conditional on the sample depth factors."""
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = squared_depth / (total_depth ** 2 - squared_depth)
        return (within - ratio * between) / exposure


def _intensity_se(deleted):
    if len(deleted) < 2 or not np.isfinite(deleted).all():
        return None
    return float(np.sqrt((len(deleted) - 1) * np.mean((deleted - deleted.mean()) ** 2)))


def _matched_intensity(depth, rows, positions, short, block_bp, map_work):
    """Check for sample-specific intensity variation on the endpoints of short-gap pairs.

    Both appearances of a marker shared by two adjacent pairs count. Each endpoint belongs
    to its own physical block. Between-sample products remove common site intensity;
    sample scaling uses the full contig, also when deleting a block without any endpoints.
    """
    sites = np.concatenate((short, short + 1))
    occupied, site_blocks = np.unique(positions // block_bp, return_inverse=True)
    blocks = len(occupied)
    info = dict(endpoints=len(sites), physical_blocks=blocks)
    if len(rows) < 3 or blocks < 2 or not len(sites):
        return info

    # Independent endpoint chunks are parallel. A chunk accumulates samples in fixed
    # order without allocating a samples-by-endpoints array.
    totals = partial(_endpoint_depth, depth=depth, rows=rows)
    chunks = np.array_split(sites, min(len(rows), len(sites)))
    endpoint_depth = np.concatenate(list(map_work(totals, chunks)))
    parts_for = partial(_intensity_parts, depth=depth, sites=sites, site_blocks=site_blocks,
                        endpoint_depth=endpoint_depth, blocks=blocks)
    parts = np.stack(list(map_work(parts_for, rows)))
    by_sample = parts.sum(axis=2)
    by_block = parts.sum(axis=0)
    within, between, exposure, factors = by_sample.T
    within_sum, between_sum, exposure_sum, factor_sum = by_sample.sum(axis=0)
    factor_squared = factors @ factors
    info.update(supported_blocks=int(np.count_nonzero(by_block[2])), endpoint_exposure=int(exposure_sum))
    if np.count_nonzero(factors > 0) < 3 or info["supported_blocks"] < 2:
        return info

    excess = _intensity_excess(within_sum, between_sum, exposure_sum, factor_sum, factor_squared)
    deleted_samples = _intensity_excess(
        within_sum - within, between_sum - 2 * between, exposure_sum - exposure,
        factor_sum - factors, factor_squared - factors ** 2,
    )
    # A block without short pairs still affects the whole-contig factors.
    remaining_factors = factors[:, None] - parts[:, 3, :]
    deleted_blocks = _intensity_excess(
        within_sum - by_block[0], between_sum - by_block[1], exposure_sum - by_block[2],
        remaining_factors.sum(axis=0), np.square(remaining_factors).sum(axis=0),
    )
    info.update(excess=float(excess) if np.isfinite(excess) else None,
                sample_se=_intensity_se(deleted_samples), block_se=_intensity_se(deleted_blocks))
    return info


def _shape_row(row, *, depth, site_bin, gap_index, bins, unique):
    values = np.asarray(depth[row], dtype=np.float64)
    squared = np.bincount(site_bin, weights=np.diff(values) ** 2, minlength=bins)
    exposure = np.bincount(gap_index, weights=values[:-1] + values[1:], minlength=unique)
    return squared, exposure


def _anchored_shape(depth, rows, gaps, edges, grid, loss, map_work):
    """Fit unit covering masses using the actual depth-exposure-weighted gaps in each lag bin."""
    unique, gap_index = np.unique(gaps, return_inverse=True)
    bins = len(edges) - 1
    bin_index = np.clip(np.searchsorted(edges, unique, side="right") - 1, 0, bins - 1)
    at = bin_index[gap_index]
    squares, exposure = np.zeros(bins), np.zeros(len(unique))
    work = partial(_shape_row, depth=depth, site_bin=at, gap_index=gap_index, bins=bins, unique=len(unique))
    for num, den in map_work(work, rows):
        squares += num
        exposure += den
    den = np.bincount(bin_index, weights=exposure, minlength=bins)
    active = den > 0
    if active.sum() < 2:
        return None
    # cumulated weights and weighted distances give E[(1 - gap / length)+] for each bin and grid length.
    first = np.searchsorted(bin_index, np.arange(bins))
    stop = np.searchsorted(bin_index, np.arange(bins), side="right")
    edge = np.maximum(first[:, None], np.minimum(stop[:, None], np.searchsorted(unique, grid)))
    mass = np.r_[0., np.cumsum(exposure)]
    moment = np.r_[0., np.cumsum(exposure * unique)]
    overlap = mass[edge] - mass[first, None] - (moment[edge] - moment[first, None]) / grid
    design = overlap[active] / den[active, None]
    y = (1 - squares[active] / den[active]) / (1 - loss)
    weight = den[active] / np.sqrt(squares[active] + 1)  # finite count variance at zero increments
    weight /= weight.max()
    x = design * weight[:, None]
    gram, b = x.T @ x, x.T @ (weight * y)
    count = len(grid)
    opt = minimize(lambda p: 0.5 * p @ gram @ p - b @ p, np.full(count, 1.0 / count),
                   jac=lambda p: gram @ p - b, bounds=[(0., 1.)] * count,
                   constraints=({"type": "eq", "fun": lambda p: p.sum() - 1, "jac": np.ones_like},),
                   method="SLSQP", options={"ftol": 1e-14, "maxiter": 1200})
    p = np.maximum(0, opt.x)
    if not np.isfinite(p).all() or p.sum() <= 0:
        return None
    p /= p.sum()
    gradient = gram @ p - b
    # Polish a boundary iterate by exact quadratic minimization along feasible two-coordinate transfers.
    # p·grad - min(grad) is an upper bound on remaining objective improvement for the normalized simplex.
    for _ in range(50000):
        enter = np.argmin(gradient)
        active = np.flatnonzero(p > 1e-13)
        leave = active[np.argmax(gradient[active])]
        gap = gradient[leave] - gradient[enter]
        if gap <= 1e-8:
            break
        curvature = gram[leave, leave] + gram[enter, enter] - 2 * gram[leave, enter]
        amount = min(p[leave], gap / curvature) if curvature > 0 else p[leave]
        p[leave] -= amount
        p[enter] += amount
        gradient += amount * (gram[:, enter] - gram[:, leave])
    return p if p @ gradient - gradient.min() <= 3e-7 else None


def check_zero_loss(law, depth, rows, positions, edges, parameters, z, *, map_work=map):
    """Return the effective law and the evidence for changing a fitted zero within an inferred library."""
    if law.status != "fitted" or law.loss != 0.:
        return law, None
    x = np.asarray(positions, dtype=np.int64)
    distances = parameters.loss_contrast_gap_bp
    gaps = np.diff(x)
    short = np.flatnonzero((gaps >= 1) & (gaps <= distances))
    positive_gap = gaps > 0
    quad = np.flatnonzero(
        positive_gap[:-2] & positive_gap[1:-1] & positive_gap[2:]
        & (x[3:] - x[:-3] <= parameters.loss_quartet_span_bp)
    )
    clusters = np.unique(np.r_[x[short], x[quad + 1]] // parameters.loss_cluster_bp)
    info = dict(status="unresolved", adjacent_pairs=len(short), quartets=len(quad), physical_blocks=len(clusters))
    if len(rows) < 2 or len(clusters) < 2 or not len(quad) or np.unique(gaps[short]).size < 2:
        return law, info
    work = partial(_row_moments, depth=depth, short=short, gaps=gaps, quadruples=quad, blocks=len(clusters),
                   pair_blocks=np.searchsorted(clusters, x[short] // parameters.loss_cluster_bp),
                   quartet_blocks=np.searchsorted(clusters, x[quad + 1] // parameters.loss_cluster_bp),
                   distances=distances)
    by_block = np.zeros((len(clusters), 2 * distances + 3))
    by_sample = np.empty((len(rows), by_block.shape[1]))
    for i, result in enumerate(map_work(work, rows)):
        by_block += result
        by_sample[i] = result.sum(axis=0)
    total = by_block.sum(axis=0)
    values = _contrasts(total, distances)
    se = np.maximum(_cluster_se(by_block, total, distances), _cluster_se(by_sample, total, distances))
    if not np.isfinite(values).all() or not np.isfinite(se).all():
        return law, info
    loss, quartet, quality_variance, mismatch = values
    info.update(paired_loss=float(loss), quartet_loss=float(quartet), quality_variance=float(quality_variance),
                pair_minus_triplet=float(mismatch), cluster_se=se.tolist())
    if loss <= z * se[0] or quartet <= z * se[1]:
        return law, dict(info, status="zero_supported_or_unresolved")
    if loss > parameters.maximum_loss or quality_variance < -z * se[2] or abs(mismatch) > z * se[3]:
        return law, dict(info, status="loss_model_unresolved")
    intensity = _matched_intensity(depth, rows, x, short, parameters.loss_cluster_bp, map_work)
    info["matched_intensity"] = intensity
    if any(intensity.get(key) is None for key in ("excess", "sample_se", "block_se")):
        return law, dict(info, status="intensity_unresolved")
    if intensity["excess"] <= z * max(intensity["sample_se"], intensity["block_se"]):
        return law, dict(info, status="intensity_not_supported")
    masses = _anchored_shape(depth, rows, gaps, edges, law.grid, loss, map_work)
    if masses is None:
        return law, dict(info, status="shape_unresolved")
    corrected = replace(law, weights=masses * (1 - loss) / law.grid, loss=float(loss))
    return corrected, dict(info, status="corrected")
