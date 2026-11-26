"""Block-sampled P9 profile for the run-wide foreign-read fraction.

Each sampled reference window is scored under the painting's own copy chain: an ordered label pair whose tracks
jump at the painting's starting rate (painting.transitions, PaintingParameters.initial_switch_rate), conditioned on
``PaintingParameters.share_flank_windows`` reference windows of flank on each side scored at the same share, with
the foreign occupancy renewed at its stationary law at both window edges (docs/methods/painting.md, "Estimating
the foreign-read share").
"""

from dataclasses import replace
from math import fsum, sqrt
from threading import local

import numpy as np
from numba import prange, types

from ...core.kernels import bounded_team, kernel
from ..shared import reads
from .emissions import gather_sites
from .parameters import READ_TABLE_CACHE_BYTES
from .read_factor import configuration_weights
from .run import decoding_grid
from .transitions import switch_probabilities

SHARE_GRID = (0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
# Stratify the painting calibration's reference windows along each contig.
SHARE_WINDOWS_PER_CONTIG = 48

_F1 = types.float64[::1]
_F2 = types.float64[:, ::1]
_F3 = types.float64[:, :, ::1]
_F4 = types.float64[:, :, :, ::1]
_B1 = types.boolean[::1]
_B2 = types.boolean[:, ::1]
_I = types.int64
_F = types.float64


def _midpoint(low, high):
    return sqrt(low * high) if low else high / 2


def candidates(grid=SHARE_GRID):
    """Grid knots plus one scatterable interior evaluation per bracket."""
    return tuple(sorted((*grid, *(_midpoint(lo, hi) for lo, hi in zip(grid[:-1], grid[1:], strict=True)))))


def refine_profile(grid, shares, profile):
    """Golden-section maximum of the pooled local quadratic through the sampled bracket.

    Use log share away from zero, and linear share on the interval touching zero.
    All evaluations of the read likelihood are independently scatterable. The
    enclosing grid knots and an evaluated midpoint supply each quadratic.
    """
    values = dict(zip(shares, profile, strict=True))
    index = max(range(len(grid)), key=lambda i: (values[grid[i]], -grid[i]))
    peak = grid[index]
    best = (values[peak], -peak)
    ratio = (sqrt(5.0) - 1.0) / 2.0
    for i in range(max(0, index - 1), min(index + 1, len(grid) - 1)):
        low, high = grid[i:i + 2]
        middle = _midpoint(low, high)
        # The midpoint is 1/2 on the scaled search axis.
        left, mid, right = values[low], values[middle], values[high]

        slope = 4 * mid - 3 * left - right
        curvature = 2 * left - 4 * mid + 2 * right

        def score(x, base=left, linear=slope, quadratic=curvature):
            return base + x * (linear + x * quadratic)

        a, b = 0.0, 1.0
        for _ in range(24):
            c, d = b - ratio * (b - a), a + ratio * (b - a)
            if score(c) < score(d):
                a = c
            else:
                b = d
        x = (a + b) / 2.0
        share = low * (high / low) ** x if low else high * x
        best = max(best, (score(x), -share), (mid, -middle))
    return peak, -best[1]


def pool_profiles(rows, n_samples, grid=SHARE_GRID):
    """Sum the contig shards by their fixed sample axis before selecting the share."""
    shares = candidates(grid)
    totals = tuple(
        tuple(fsum(row[i][s] for row in rows) for s in range(n_samples))
        for i in range(len(shares))
    )
    pooled = tuple(fsum(row) for row in totals)
    peak, estimate = refine_profile(grid, shares, pooled)
    return dict(estimate=estimate, grid_peak=peak, grid=grid, shares=shares, pooled=pooled, samples=totals)


@kernel(overloads=((_F3, _F3, _F3, _F2, _F2, _F2, _B1, _F2, _F, _F, _F3, _F2, _F2),),
        cache=True, inline="always", nogil=True)
def _site_tables(own, ref_removed, alt_removed, laws, genotypes, cohort, observed, alleles, error, epsilon,
                 coefficients, scalars, donor):
    """The label-pair-independent parts of one sample's P9 factors at each site.

    A pair's factor is linear in its four configuration weights (w00, w01, w10, w11). coefficients[t, j, c]: j = 0
    and 1 multiply w01 and w10 in the heterozygous term shared by every state; j = 2 and 3 multiply w00 and w11 in
    the retained own count ratio; j = 4, 5 (w00) and 6, 7 (w11) are the GL-anchored count ratios with one REF or
    one ALT read removed, which a foreign read replaces in homozygous configurations. scalars[t] = (the outlier
    term, the w00 and w11 weights of a homozygous configuration whose own count mean is zero). donor[t, k]: the
    probability that a foreign base from label k reads ALT, with the sample's base error. Unobserved sites keep no
    coefficients (their factors are one)."""
    sites, _, classes = own.shape
    labels = alleles.shape[0]
    means = np.zeros(4)
    for t in range(sites):
        for k in range(labels):
            donor[t, k] = error + (1.0 - 2.0 * error) * alleles[k, t]
        if not observed[t]:
            continue
        g0, g1, g2 = genotypes[t]
        kept = 1.0 - epsilon
        scalars[t, 0] = epsilon * (g0 * cohort[t, 0] + g1 * cohort[t, 1] + g2 * cohort[t, 2])
        means[:] = 0.0
        for g in range(4):
            for c in range(classes):
                means[g] += own[t, g, c] * laws[t, c]
        scalars[t, 1] = kept * g0 if means[0] <= 0.0 else 0.0
        scalars[t, 2] = kept * g2 if means[3] <= 0.0 else 0.0
        for c in range(classes):
            coefficients[t, 0, c] = kept * g1 * (own[t, 1, c] / means[1] if means[1] > 0.0 else 1.0)
            coefficients[t, 1, c] = kept * g1 * (own[t, 2, c] / means[2] if means[2] > 0.0 else 1.0)
            for j in range(2):
                g = 3 * j
                likelihood = g2 if j else g0
                if means[g] > 0.0:
                    relative = kept * likelihood / means[g]
                    coefficients[t, 2 + j, c] = relative * own[t, g, c]
                    coefficients[t, 4 + 2 * j, c] = relative * ref_removed[t, g, c]
                    coefficients[t, 5 + 2 * j, c] = relative * alt_removed[t, g, c]
                else:
                    coefficients[t, 2 + j, c] = 0.0
                    coefficients[t, 4 + 2 * j, c] = 0.0
                    coefficients[t, 5 + 2 * j, c] = 0.0


@kernel(overloads=((_F2, _F3),), cache=True, inline="always", nogil=True)
def _pair_weights(alleles, weights):
    """weights[t, j, p]: the configuration weights (w00, w01, w10, w11) of the ordered label pair p = x L + y at
    site t (a repeated named label shares its draw; BACKGROUND draws independently)."""
    labels, sites = alleles.shape
    for t in range(sites):
        for x in range(labels):
            for y in range(labels):
                w00, w01, w10, w11 = configuration_weights(alleles[x, t], alleles[y, t], x == y and x < labels - 1)
                p = x * labels + y
                weights[t, 0, p] = w00
                weights[t, 1, p] = w01
                weights[t, 2, p] = w10
                weights[t, 3, p] = w11


@kernel(overloads=((_F3, _B1, _F3, _F2, _I, _F, _F2, _F2, _F2),), cache=True, inline="always", nogil=True)
def _pair_factors(weights, observed, coefficients, scalars, t, loss, own, base, slope):
    """Site t's P9 factors of every ordered label pair p: own[c, p] with the own reads alone, base[c, p] + q slope[c, p]
    with a foreign read whose base reads ALT with probability q. Unobserved sites have unit factors (P9 still excludes
    a pair's labels as donors there; the callers apply that)."""
    classes, pairs = own.shape
    if not observed[t]:
        for c in range(classes):
            for p in range(pairs):
                own[c, p] = 1.0
                base[c, p] = 1.0
                slope[c, p] = 0.0
        return
    s0, u0, u3 = scalars[t, 0], scalars[t, 1], scalars[t, 2]
    for c in range(classes):
        c0, c1, c2, c3 = coefficients[t, 0, c], coefficients[t, 1, c], coefficients[t, 2, c], coefficients[t, 3, c]
        r0, a0, r3, a3 = coefficients[t, 4, c], coefficients[t, 5, c], coefficients[t, 6, c], coefficients[t, 7, c]
        for p in range(pairs):
            w00, w01, w10, w11 = weights[t, 0, p], weights[t, 1, p], weights[t, 2, p], weights[t, 3, p]
            shared = s0 + w01 * c0 + w10 * c1
            retained = w00 * c2 + w11 * c3
            with_ref = w00 * r0 + w11 * r3
            with_alt = w00 * a0 + w11 * a3
            own[c, p] = shared + retained + (w00 * u0 + w11 * u3)
            base[c, p] = shared + loss * retained + (1.0 - loss) * with_ref
            slope[c, p] = (1.0 - loss) * (with_alt - with_ref)


@kernel(overloads=((_F3, _F, _I, _F1),), cache=True, inline="always", nogil=True)
def _switch(state, redraw, active, sums):
    """Each track of every ordered label pair re-draws its label uniformly with probability ``redraw``: the
    painting's jump law (painting.transitions). It is symmetric, so the backward messages use it unchanged."""
    classes, _, pairs = state.shape
    labels = sums.shape[0]
    keep = 1.0 - redraw
    spread = redraw / labels
    for c in range(classes):
        for f in range(active):
            row = state[c, f]
            # the first track: one sum over its labels for each second-track label
            for y in range(labels):
                sums[y] = 0.0
            for x in range(labels):
                for y in range(labels):
                    sums[y] += row[x * labels + y]
            for x in range(labels):
                for y in range(labels):
                    row[x * labels + y] = keep * row[x * labels + y] + spread * sums[y]
            # the second track
            for x in range(labels):
                total = 0.0
                for y in range(labels):
                    total += row[x * labels + y]
                for y in range(labels):
                    row[x * labels + y] = keep * row[x * labels + y] + spread * total


@kernel(overloads=((_F3, _F3, _F2, _F, _F, _I, _F2, _F2, _F2, _F1, _F1, _I),),
        cache=True, inline="always", nogil=True)
def _advance(state, moved, transition, phi, depth, active, own, base, slope, donor, totals, labels):
    """One site of the share's row action on every ordered label pair: own transport, foreign survival, renewal and
    arrival, then the pair's P9 factors (the pair's labels excluded as donors). Returns the total mass."""
    classes, _, pairs = state.shape
    absent = np.exp(-depth * (1.0 - phi))
    arrival = (1.0 - absent) / labels
    for c in range(classes):
        for f in range(active):
            for p in range(pairs):
                moved[c, f, p] = 0.0
        for previous in range(classes):
            weight = transition[previous, c]
            for f in range(active):
                for p in range(pairs):
                    moved[c, f, p] += weight * state[previous, f, p]
    for p in range(pairs):
        totals[p] = 0.0
    for c in range(classes):
        carried = state[c, 0]          # reused as the carried own mass of each pair, then overwritten
        for p in range(pairs):
            carried[p] = 0.0
        for f in range(1, active):
            for p in range(pairs):
                carried[p] += moved[c, f, p]
        for p in range(pairs):
            carried[p] = moved[c, 0, p] + (1.0 - phi) * carried[p]
        for f in range(1, active):
            q = donor[f - 1]
            column = state[c, f]
            for p in range(pairs):
                column[p] = (phi * moved[c, f, p] + arrival * carried[p]) * (base[c, p] + q * slope[c, p])
            k = f - 1
            for y in range(labels):
                column[k * labels + y] = 0.0
            for x in range(labels):
                column[x * labels + k] = 0.0
            for p in range(pairs):
                totals[p] += column[p]
        for p in range(pairs):
            value = absent * carried[p] * own[c, p]
            carried[p] = value
            totals[p] += value
    total = 0.0
    for p in range(pairs):
        total += totals[p]
    return total


@kernel(overloads=((_F3, _F3, _F2, _F, _F, _I, _F2, _F2, _F2, _F1, _F1, _I),),
        cache=True, inline="always", nogil=True)
def _retreat(message, mixed, transition, phi, depth, active, own, base, slope, donor, stay, labels):
    """The column action (adjoint) of one site's operator on every ordered label pair, before the site's switch."""
    classes, _, pairs = message.shape
    absent = np.exp(-depth * (1.0 - phi))
    arrival = (1.0 - absent) / labels
    for c in range(classes):
        for p in range(pairs):
            stay[p] = 0.0
        for f in range(1, active):
            q = donor[f - 1]
            column = mixed[c, f]
            for p in range(pairs):
                column[p] = (base[c, p] + q * slope[c, p]) * message[c, f, p]
            k = f - 1
            for y in range(labels):
                column[k * labels + y] = 0.0
            for x in range(labels):
                column[x * labels + k] = 0.0
            for p in range(pairs):
                stay[p] += column[p]
        for p in range(pairs):
            stay[p] = absent * (own[c, p] * message[c, 0, p]) + arrival * stay[p]
            mixed[c, 0, p] = stay[p]
        for f in range(1, active):
            for p in range(pairs):
                mixed[c, f, p] = (1.0 - phi) * stay[p] + phi * mixed[c, f, p]
    for previous in range(classes):
        for f in range(active):
            for p in range(pairs):
                message[previous, f, p] = 0.0
        for c in range(classes):
            weight = transition[previous, c]
            for f in range(active):
                for p in range(pairs):
                    message[previous, f, p] += weight * mixed[c, f, p]


@kernel(overloads=((_F3, _F, _I),), cache=True, inline="always", nogil=True)
def _rescale(state, total, active):
    """Divide the first ``active`` columns of every pair's state by ``total``."""
    inverse = 1.0 / total
    classes, _, pairs = state.shape
    for c in range(classes):
        for f in range(active):
            for p in range(pairs):
                state[c, f, p] *= inverse


@kernel(overloads=((_F2, _F3, _B2, _F4, _F4, _F4, _F4, _F3, _F2, _F1, _F1, _F1, _F1, _F1, _F, _F, _I, _I, _F2),),
        cache=True, parallel=True, nogil=True)
def _score_window(alleles, gl, observed, own, ref_removed, alt_removed, transition, laws, cohort, error, survival,
                  redraw, covering, shares, loss, epsilon, lo, hi, scores):
    """log P(window sites [lo, hi) | flank sites) at each share minus the same without the window's data.

    The label pair of each sample is the painting's copy chain over the extended window: an ordered pair whose two
    tracks re-draw their labels at ``redraw`` per site gap (redraw[0] unused), uniform at the extended window's
    start. The flanks [0, lo) and [hi, sites) are scored at the same share as the window. The foreign occupancy is
    re-drawn from its stationary law at both window edges, so no foreign read crosses an edge. The score is
    log(alpha . chain(share) . beta) - log(alpha . blank . beta): alpha the left flank's forward message and beta
    the right flank's backward message at that share, both marginal over foreign occupancy; the blank chain has
    the same copy and own-class steps with unit factors and no foreign reads. Without flanks and switches it is
    the block likelihood of one fixed pair under the uniform ordered-pair law."""
    labels, sites = alleles.shape
    classes = own.shape[-1]
    width = labels + 1
    pairs = labels * labels
    weights = np.empty((sites, 4, pairs))
    _pair_weights(alleles, weights)
    for s in prange(len(observed)):
        coefficients = np.empty((sites, 8, classes))
        scalars = np.empty((sites, 3))
        donor = np.empty((sites, labels))
        _site_tables(own[s], ref_removed[s], alt_removed[s], laws[s], gl[s], cohort, observed[s], alleles,
                     error[s], epsilon, coefficients, scalars, donor)
        state = np.empty((classes, width, pairs))
        scratch = np.empty((classes, width, pairs))
        own_factor = np.empty((classes, pairs))
        base = np.empty((classes, pairs))
        slope = np.empty((classes, pairs))
        vector = np.empty(pairs)
        sums = np.empty(labels)
        left = np.empty((classes, pairs))
        right = np.empty((classes, pairs))
        initial = laws[s, 0]
        seen = observed[s]
        for i in range(len(shares)):
            depth = 2.0 * shares[i] * covering[s]
            active = 1 if depth == 0.0 else width
            absence = np.exp(-depth)
            supported = True
            if lo > 0:
                for c in range(classes):
                    for p in range(pairs):
                        state[c, 0, p] = initial[c] * absence / pairs
                        for f in range(1, active):
                            state[c, f, p] = initial[c] * (1.0 - absence) / labels / pairs
                for t in range(lo):
                    if t > 0 and redraw[t] > 0.0:
                        _switch(state, redraw[t], active, sums)
                    _pair_factors(weights, seen, coefficients, scalars, t, loss, own_factor, base, slope)
                    total = _advance(state, scratch, transition[s, t], survival[t], depth, active, own_factor, base,
                                     slope, donor[t], vector, labels)
                    if not total > 0.0:
                        supported = False
                        break
                    _rescale(state, total, active)
                for c in range(classes):
                    for p in range(pairs):
                        mass = 0.0
                        for f in range(active):
                            mass += state[c, f, p]
                        left[c, p] = mass
            else:
                for c in range(classes):
                    for p in range(pairs):
                        left[c, p] = initial[c] / pairs
            if supported and hi < sites:
                for c in range(classes):
                    for f in range(active):
                        for p in range(pairs):
                            state[c, f, p] = 1.0
                for t in range(sites - 1, hi - 1, -1):
                    _pair_factors(weights, seen, coefficients, scalars, t, loss, own_factor, base, slope)
                    _retreat(state, scratch, transition[s, t], survival[t], depth, active, own_factor, base, slope,
                             donor[t], vector, labels)
                    if redraw[t] > 0.0:
                        _switch(state, redraw[t], active, sums)
                    total = 0.0
                    for c in range(classes):
                        for f in range(active):
                            for p in range(pairs):
                                total += state[c, f, p]
                    if not total > 0.0:
                        supported = False
                        break
                    _rescale(state, total, active)
                for c in range(classes):
                    for p in range(pairs):
                        mass = absence * state[c, 0, p]
                        for f in range(1, active):
                            mass += (1.0 - absence) / labels * state[c, f, p]
                        right[c, p] = mass
            elif supported:
                for c in range(classes):
                    for p in range(pairs):
                        right[c, p] = 1.0
            if not supported:
                scores[i, s] = -np.inf
                continue
            numerator = 0.0
            for c in range(classes):
                for p in range(pairs):
                    state[c, 0, p] = left[c, p] * absence
                    for f in range(1, active):
                        state[c, f, p] = left[c, p] * (1.0 - absence) / labels
            for t in range(lo, hi):
                if t > 0 and redraw[t] > 0.0:
                    _switch(state, redraw[t], active, sums)
                _pair_factors(weights, seen, coefficients, scalars, t, loss, own_factor, base, slope)
                total = _advance(state, scratch, transition[s, t], survival[t], depth, active, own_factor, base,
                                 slope, donor[t], vector, labels)
                if not total > 0.0:
                    supported = False
                    break
                numerator += np.log(total)
                _rescale(state, total, active)
            close = 0.0
            if supported:
                for c in range(classes):
                    for p in range(pairs):
                        mass = 0.0
                        for f in range(active):
                            mass += state[c, f, p]
                        close += mass * right[c, p]
            if not close > 0.0:
                scores[i, s] = -np.inf
                continue
            numerator += np.log(close)
            # The same copy chain and own-class steps over the window with its data removed: unit factors and
            # no foreign reads (one column).
            denominator = 0.0
            for c in range(classes):
                for p in range(pairs):
                    state[c, 0, p] = left[c, p]
            for t in range(lo, hi):
                if t > 0 and redraw[t] > 0.0:
                    _switch(state, redraw[t], 1, sums)
                for c in range(classes):
                    for p in range(pairs):
                        scratch[c, 0, p] = 0.0
                    for previous in range(classes):
                        weight = transition[s, t, previous, c]
                        for p in range(pairs):
                            scratch[c, 0, p] += weight * state[previous, 0, p]
                for p in range(pairs):
                    vector[p] = 0.0
                for c in range(classes):
                    for p in range(pairs):
                        state[c, 0, p] = scratch[c, 0, p]
                        vector[p] += scratch[c, 0, p]
                total = 0.0
                for p in range(pairs):
                    total += vector[p]
                if not total > 0.0:
                    supported = False
                    break
                denominator += np.log(total)
                _rescale(state, total, 1)
            close = 0.0
            if supported:
                for c in range(classes):
                    for p in range(pairs):
                        close += state[c, 0, p] * right[c, p]
            if not close > 0.0:
                scores[i, s] = -np.inf
                continue
            scores[i, s] = numerator - (denominator + np.log(close))


def _parts(builder, rows, sites, survival, cache, cache_bytes):
    """The fitted read model's own transfers and original-depth count likelihoods."""
    pilot_width = builder._offset_widths[builder.library[rows[0]]]
    adaptive = pilot_width > builder.parameters.offsets
    parameters = replace(builder.parameters, offsets=pilot_width) if adaptive else builder.parameters
    first, last = int(sites[0]), int(sites[-1]) + 1
    counts = np.asarray(builder.counts[rows, first:last], dtype=np.int64)
    depth = counts.sum(axis=2)
    gaps = np.diff(builder.positions[first:last], prepend=builder.positions[first]).astype(np.float64)
    covering = builder.covering[rows]
    tables = reads.build_tables(gaps, survival, covering, reads.table_depth(depth, builder.minor, parameters),
                                builder.minor, parameters, cache=cache, cache_bytes=cache_bytes)
    if adaptive:
        posterior, operators, covering_depth = reads.adaptive_offset_operators(
            depth, covering, survival.loss, tables, builder.parameters,
        )
    else:
        posterior, operators, covering_depth = reads.offset_operators_fast(depth, covering, survival.loss, tables)
    local_sites = np.ascontiguousarray(sites - first)
    kept = np.zeros(last - first, dtype=bool)
    kept[local_sites] = True
    transition = operators if kept.all() else reads.compose_operators_fast(operators, kept)
    laws = reads.site_laws_fast(posterior, covering_depth, local_sites, tables)
    ref, alt = counts[:, kept, 0], counts[:, kept, 1]
    posterior = posterior[kept]
    error = builder.error_rate[rows]
    own = reads.class_likelihoods_fast(ref, alt, error, posterior, tables)
    ref_removed = reads.class_likelihoods_fast(np.maximum(ref - 1, 0), alt, error, posterior, tables)
    alt_removed = reads.class_likelihoods_fast(ref, np.maximum(alt - 1, 0), error, posterior, tables)
    ref_removed *= (ref > 0)[..., None, None]
    alt_removed *= (alt > 0)[..., None, None]
    return transition, laws, own, ref_removed, alt_removed


def _windows(panels, config):
    """The sampled reference windows, each with up to ``config.share_flank_windows`` reference windows of the same
    panel on each side: (contig columns of the extended window, its label ALT probabilities with BACKGROUND's as
    the last row, (window start, window stop) within it)."""
    reference = replace(config, snps_per_bin=config.reference_snps_per_bin)
    flank = int(config.share_flank_windows)
    grids = [(panel, *decoding_grid(panel, reference)) for panel in panels]
    count = sum(bins.n_bins for _, _, bins in grids)
    size = min(count, SHARE_WINDOWS_PER_CONTIG)
    sampled = iter((2 * i + 1) * count // (2 * size) for i in range(size))
    target = next(sampled, None)
    offset = 0
    for panel, states, bins in grids:
        if target is None:
            break
        if target - offset >= bins.n_bins:
            offset += bins.n_bins
            continue
        alleles = states.alt_probability()
        columns = np.asarray(panel.site_indices())[states.site_indices]
        while target is not None and target - offset < bins.n_bins:
            b = target - offset
            start, stop = int(bins.starts[b]), int(bins.stops[b])
            if start < stop:
                first = int(bins.starts[max(0, b - flank)])
                last = int(bins.stops[min(bins.n_bins - 1, b + flank)])
                yield (columns[first:last],
                       np.ascontiguousarray(np.vstack((alleles[:, first:last],
                                                       states.background_frequency[first:last]))),
                       (start - first, stop - first))
            target = next(sampled, None)
        offset += bins.n_bins


def _redraw(positions, genetic_map, rate, labels):
    """float64[L]: the probability that one track re-draws its label uniformly over the ``labels`` labels across
    each site gap (index 0: none). The painting's jump law moves a track to one of the other S - 1 labels with
    probability theta (painting.transitions.switch_probabilities); a uniform re-draw over all S labels with
    probability theta S / (S - 1) is the same law."""
    redraw = np.zeros(len(positions))
    if labels > 1 and len(positions) > 1:
        positions = np.asarray(positions, dtype=np.float64)
        morgans = genetic_map.interval_morgans(positions[:-1], positions[1:])
        redraw[1:] = switch_probabilities(morgans, rate, labels) * labels / (labels - 1.0)
    return redraw


def score_contig(panels, evidence, config, genetic_map, *, epsilon, emission_builder, batch_samples,
                 map_windows=map, window_workers=1):
    """Per-sample log-likelihood differences from share zero on sampled held-out windows, each conditioned on its
    flanks."""
    builder = emission_builder
    samples = evidence.observed.shape[0]
    shares = np.asarray(candidates())
    result = np.zeros((len(shares), samples))
    supported = 0
    material = tuple(_windows(panels, config))
    columns = np.unique(np.concatenate([window[0] for window in material])) if material else np.empty(0, dtype=int)
    # The checkpoint codec reads full underlying chunks. Gather each column once
    # so the window loop can reuse the same decoded sample rows.
    values, flags = gather_sites(evidence.likelihoods, evidence.observed, columns, slice(0, samples))
    # Each mapper thread owns a table cache. Their combined capacity stays within the shard's table allowance.
    tables = local()
    cache_bytes = READ_TABLE_CACHE_BYTES // max(1, min(len(material), window_workers))

    def score(window):
        sites, alleles, (lo, hi) = window
        cache = getattr(tables, "cache", None)
        if cache is None:
            cache = tables.cache = {}
        values_at_window = np.zeros_like(result)
        supported_at_window = 0
        lookup = np.searchsorted(columns, sites)
        size = builder.batch_samples(sites, len(alleles), 1, False, batch_samples)
        gaps = np.diff(builder.positions[sites], prepend=builder.positions[sites[0]]).astype(np.float64)
        redraw = _redraw(builder.positions[sites], genetic_map, config.initial_switch_rate, len(alleles))
        for first in range(0, samples, size):
            selection = slice(first, min(first + size, samples))
            rows = np.arange(selection.start, selection.stop)
            gl, observed = values[selection][:, lookup], flags[selection][:, lookup]
            # A sample with no observation inside the window has no data to compare shares (its score would be
            # P9's exclusion of the pair's labels as donors alone). The read state must hold over the flanks too,
            # so bypass is decided on the whole extended window.
            has_observations = observed[:, lo:hi].any(axis=1)
            for library in np.unique(builder.library[rows]):
                local = np.flatnonzero((builder.library[rows] == library) & has_observations)
                if not len(local):
                    continue
                selected = rows[local]
                use = ~builder._bypassed(selected, int(sites[0]), int(sites[-1]) + 1)
                local, selected = local[use], selected[use]
                if not len(selected):
                    continue
                law = builder.survivals[library]
                transition, pi, own, ref_removed, alt_removed = _parts(builder, selected, sites, law, cache, cache_bytes)
                survival = law.survival(gaps)
                survival[0] = 1.0
                scored = np.empty((len(shares), len(selected)))
                with bounded_team(len(selected)):
                    _score_window(
                        alleles, np.ascontiguousarray(gl[local], dtype=np.float64),
                        np.ascontiguousarray(observed[local]),
                        own, ref_removed, alt_removed, np.ascontiguousarray(transition.transpose(1, 0, 2, 3)),
                        np.ascontiguousarray(pi.transpose(1, 0, 2)), np.ascontiguousarray(builder.cohort[sites]),
                        builder.error_rate[selected], survival, redraw, builder.covering[selected], shares,
                        float(law.loss), float(epsilon), lo, hi, scored,
                    )
                # The outlier branch makes finite likelihoods typical, but unsupported
                # pairs can have zero likelihood at every share and contain no comparison.
                valid = np.isfinite(scored).all(axis=0)
                supported_at_window += int(valid.sum())
                values_at_window[:, selected[valid]] = scored[:, valid] - scored[0, valid]
        return values_at_window, supported_at_window

    for rows, count in map_windows(score, material):
        result += rows
        supported += count
    return dict(rows=tuple(tuple(float(v) for v in row) for row in result), windows=len(material), supported=supported)
