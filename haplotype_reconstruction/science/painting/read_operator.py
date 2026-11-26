"""Structured transfers for the painting read state and its bin-local projection."""

from __future__ import annotations

import math

import numpy as np
from numba import prange, types

from ...core.kernels import kernel
from .read_factor import (
    MATRIX_CHUNK_BINS,
    _untempered_ratios,
    configuration_weights,
    donor_ratio,
    own_count_ratios,
    pair_representatives,
)

_F1 = types.float64[::1]
_F2 = types.float64[:, ::1]
_F3 = types.float64[:, :, ::1]
_F4 = types.float64[:, :, :, ::1]
_I1 = types.int64[::1]
_I2 = types.int64[:, ::1]
_B1 = types.boolean[::1]
_B2 = types.boolean[:, ::1]
_M5 = types.float32[:, :, :, :, ::1]
_D5 = types.float64[:, :, :, :, ::1]
_I = types.int64
_B = types.boolean
_F = types.float64

# Bounds, unordered pairs, allele codes and probabilities, state transitions and factors.
TAPE = types.Tuple((_I1, _I1, _I1, _I1, _I2, _F2, _F4, _F3, _F2, _F2, _B1, _F4))
ALLELE_FACTOR_ROWS = 10  # Nine ordered allele-code pairs and one repeated uncalled named label.


def empty_tape():
    """The site tape of a bin product already contracted to its fixed-width own block."""
    index = np.empty(0, dtype=np.int64)
    return (index, index, index, index, np.empty((0, 0), dtype=np.int64), np.empty((0, 0)),
            np.empty((0, 0, 0, 0)), np.empty((0, 0, 0)), np.empty((0, 0)), np.empty((0, 0)),
            np.empty(0, dtype=np.bool_), np.empty((0, 0, 0, 0)))


@kernel(overloads=((_F4, _F3, _B2, _F2, _F3, _F2, _F, _I2, _I1, _I1, _F3, _F1, _F1, _B1, _B1, _F4),),
        cache=True, parallel=True, nogil=True)
def _site_factors(class_lik, law, seen, alt, gl, cohort, epsilon, codes, first, second, counts, error, loss,
                  bypass, hidden, out):
    """Build GL ratios at visible sites and normalized ALT-count probabilities at hidden sites in parallel."""
    samples, sites, rows, width = out.shape
    own = law.shape[2]
    named = len(alt) - 1
    for item in prange(samples * sites):
        s, t = item // sites, item % sites
        if not seen[s, t] or bypass[s]:
            out[s, t] = 1.0
            continue
        kappa = np.empty((4, width))
        masked = hidden.size > 0 and hidden[t]
        if masked:
            outlier = 0.0
            for c in range(own):
                outlier += law[t, s, c] * (cohort[t, 0] * class_lik[s, t, 0, c]
                                          + cohort[t, 1] * class_lik[s, t, 1, c]
                                          + cohort[t, 2] * class_lik[s, t, 3, c])
            kappa[:, :own] = class_lik[s, t]
        else:
            own_count_ratios(class_lik, law, s, t, kappa)
        if width > own:
            ref, alt_count = counts[s, t]
            depth = ref + alt_count
            e = error[s]
            for config in range(4):
                expected = e if config == 0 else 1.0 - e if config == 3 else 0.5
                pmf = 1.0
                if masked and depth > 0:
                    pmf = math.exp(math.lgamma(depth + 1.0) - math.lgamma(alt_count + 1.0)
                                   - math.lgamma(ref + 1.0) + alt_count * math.log(expected)
                                   + ref * math.log1p(-expected))
                ratio_ref = donor_ratio(config, 0, alt_count, depth, e, loss[s], expected)
                ratio_alt = donor_ratio(config, 1, alt_count, depth, e, loss[s], expected)
                kappa[config, own] = pmf * ratio_ref
                kappa[config, own + 1] = pmf * ratio_alt
                # BACKGROUND's fractional allele: the ratio is linear in the donor's ALT probability.
                kappa[config, own + 2] = pmf * (alt[named, t] * ratio_alt + (1.0 - alt[named, t]) * ratio_ref)
        g0, g1, g2 = gl[s, t]
        background = cohort[t, 0] * g0 + cohort[t, 1] * g1 + cohort[t, 2] * g2
        for row in range(rows):
            if codes.size:
                same = row == ALLELE_FACTOR_ROWS - 1
                if same:
                    p = q = alt[named, t]
                else:
                    x, y = row // 3, row % 3
                    p = alt[named, t] if x == 2 else float(x)
                    q = alt[named, t] if y == 2 else float(y)
            else:
                x, y = first[row], second[row]
                p, q = alt[x, t], alt[y, t]
                same = x == y and x < named
            w00, w01, w10, w11 = configuration_weights(p, q, bool(same))
            if masked:
                for c in range(width):
                    out[s, t, row, c] = (1.0 - epsilon) * (
                        w00 * kappa[0, c] + w01 * kappa[1, c] + w10 * kappa[2, c] + w11 * kappa[3, c]
                    ) + epsilon * outlier
                continue
            _untempered_ratios(kappa, g0, g1, g2, background, epsilon, w00, w01, w10, w11, out[s, t, row])


@kernel(overloads=((_F4, _F2, _I1, _I1, _F4),), cache=True, parallel=True, nogil=True)
def _temper(raw, weight, starts, stops, out):
    samples, _, rows, width = raw.shape
    bins = len(starts)
    for item in prange(samples * bins):
        s, b = item // bins, item % bins
        for t in range(starts[b], stops[b]):
            w = weight[s, b if weight.shape[1] == bins else t]
            if w == 1.0:
                out[s, t] = raw[s, t]
            else:
                for r in range(rows):
                    for c in range(width):
                        out[s, t, r, c] = raw[s, t, r, c] ** w


# Compile the shared site walk once. Inlining it into both directions of each recurrence
# exceeded the cold-signature suite's 600-second limit on the 76-core worker.
# Callers may pass a strided input row and scratch matrix (views into larger work arrays).
@kernel(overloads=((TAPE, _I, _I, _I, _F1, _F1, _F2, _B, _F1),
                   (TAPE, _I, _I, _I, types.float64[:], _F1, types.float64[:, :], _B, _F1)), cache=True, nogil=True)
def _walk(tape, s, p, b, v, out, work, backward, exponents):
    """Apply sites in row or column order, keeping exact support and a separate log scale."""
    starts, stops, first, second, codes, alt, transition, law, survival, absent, bypass, factors = tape
    own = law.shape[2]
    width = len(v)
    labels = width - own
    if bypass[s]:
        out[:] = 0.0
        out[0] = v[0]
        return 0.0
    x, y = first[p], second[p]
    repeated_named = x == y and x < len(alt) - 1
    current, moved = work[0], work[1]
    current[:] = v
    lg = 0.0
    start, stop = starts[b], stops[b]
    for offset in range(stop - start):
        t = stop - 1 - offset if backward else start + offset
        exponent = exponents[b] if len(exponents) == len(starts) else exponents[t]
        table = np.int64(p)
        if codes.size:
            first_code = codes[t, x]
            table = 3 * first_code + codes[t, y]
            # A called allele has the same configuration law for independent and repeated draws.
            if repeated_named and first_code == 2:
                table = ALLELE_FACTOR_ROWS - 1
        factor = factors[s, t, table]
        weighted = work[2]
        for j in range(own):
            weighted[j] = factor[j] if exponent == 1.0 else factor[j] ** exponent
        for j in range(labels):
            value = 0.0
            if j != x and j != y:
                if codes.size:
                    value = factor[own + codes[t, j]]
                else:
                    value = (1.0 - alt[j, t]) * factor[own] + alt[j, t] * factor[own + 1]
                if exponent != 1.0:
                    value = value ** exponent
            weighted[own + j] = value
        if backward:
            for j in range(own):
                weighted[j] *= current[j]
            for i in range(own):
                value = 0.0
                for j in range(own):
                    value += transition[t, s, i, j] * weighted[j]
                moved[i] = value
        else:
            for j in range(own):
                value = 0.0
                for i in range(own):
                    value += current[i] * transition[t, s, i, j]
                moved[j] = value
        if labels:
            phi = survival[s, t]
            q = absent[s, t]
            own_mass = 0.0
            foreign_mass = 0.0
            if backward:
                for j in range(own):
                    own_mass += law[t, s, j] * weighted[j]
                for j in range(own, width):
                    weighted[j] *= current[j]
                    foreign_mass += weighted[j]
                arrival = (1.0 - q) * foreign_mass / labels
                restart = (1.0 - phi) * (q * own_mass + arrival)
                for i in range(own):
                    moved[i] = q * moved[i] + arrival
                for j in range(own, width):
                    moved[j] = phi * weighted[j] + restart
            else:
                for j in range(own):
                    own_mass += current[j]
                for j in range(own, width):
                    foreign_mass += current[j]
                arrival = (1.0 - q) * (own_mass + (1.0 - phi) * foreign_mass) / labels
                restart = (1.0 - phi) * q * foreign_mass
                for j in range(own):
                    moved[j] = q * moved[j] + restart * law[t, s, j]
                for j in range(own, width):
                    moved[j] = phi * current[j] + arrival
        if not backward:
            for j in range(width):
                moved[j] *= weighted[j]
        top = 0.0
        for j in range(width):
            top = max(top, moved[j])
        if top == 0.0:
            out[:] = 0.0
            return lg
        # Keep broad headroom in both directions. The final step always contributes its gauge for cross-pair sums.
        if top < 1e-100 or top > 1e100 or offset + 1 == stop - start:
            lg += math.log(top)
            for j in range(width):
                moved[j] /= top
        current, moved = moved, current
    out[:] = current
    return lg


@kernel(overloads=tuple((matrix, TAPE, _F2, _I, _I, _I, _B, _I1, _F1, _F1, _F2, _B)
                        for matrix in (_M5, _D5)),
        cache=True, inline="always", nogil=True)
def apply_bin(matrix, tape, weight, s, p, b, swap, perm, v, out, work, backward):
    """Act in track orientation. The return value is the extra output log gauge."""
    width = len(v)
    if matrix.itemsize == 8 or not tape[0].size:
        oriented, target = v, out
        if swap:
            oriented, target = work[3], work[2]
            for j in range(width):
                oriented[j] = v[perm[j]]
        block = matrix[s, p, b]
        if backward:
            for i in range(width):
                value = 0.0
                for j in range(width):
                    value += block[i, j] * oriented[j]
                target[i] = value
        else:
            for i in range(width):
                target[i] = 0.0
            for j in range(width):
                for i in range(width):
                    target[i] += block[j, i] * oriented[j]
        if swap:
            for i in range(width):
                out[i] = target[perm[i]]
        return 0.0
    oriented = work[3]
    for i in range(width):
        oriented[i] = v[perm[i] if swap else i]
    log_scale = _walk(tape, np.int64(s), p, np.int64(b), oriented, out, work, np.bool_(backward), weight[s])
    if swap:
        oriented[:] = out
        for i in range(width):
            out[i] = oriented[perm[i]]
    return log_scale


@kernel(overloads=((_I1, _I1, _I, _I, _I, _I1),), cache=True, inline="always", nogil=True)
def _representative_orientation(first, second, pair, representative, own, permutation):
    """Exchange donor labels with identical bin alleles to match the representative's forbidden pair."""
    for c in range(len(permutation)):
        permutation[c] = c
    x, y = own + first[pair], own + second[pair]
    u, v = own + first[representative], own + second[representative]
    permutation[x], permutation[u] = u, x
    if x != y:
        # The first swap may already have moved the second representative label.
        source = u if v == x else v
        permutation[y], permutation[source] = permutation[source], permutation[y]


@kernel(overloads=((TAPE, _I2, _F4, _F3),), cache=True, parallel=True, nogil=True)
def _untempered_rows(tape, representative, rows, logs):
    """Apply each untempered adjoint to one in parallel over sample and bin chunks for the D5P carry."""
    samples, pairs, bins, width = rows.shape
    chunks = (bins + MATRIX_CHUNK_BINS - 1) // MATRIX_CHUNK_BINS
    exponents = np.ones(bins)
    for task in prange(samples * chunks):
        s, chunk = np.int64(task // chunks), np.int64(task % chunks)
        boundary = np.ones(width)
        work = np.empty((3, width))
        own = tape[7].shape[2]
        permutation = np.empty(width, dtype=np.int64)
        for b in range(chunk * MATRIX_CHUNK_BINS, min(bins, (chunk + 1) * MATRIX_CHUNK_BINS)):
            for p in range(pairs):
                r = representative[p, b] if representative.size else p
                if r == p:
                    logs[s, p, b] = _walk(tape, s, p, b, boundary, rows[s, p, b], work, np.bool_(True), exponents)
                else:
                    if width > own:
                        _representative_orientation(tape[2], tape[3], p, r, own, permutation)
                        for c in range(width):
                            rows[s, p, b, c] = rows[s, r, b, permutation[c]]
                    else:
                        rows[s, p, b] = rows[s, r, b]
                    logs[s, p, b] = logs[s, r, b]


@kernel(overloads=tuple((matrix, TAPE, _F2, _F2, _F2, _F4, _F3, _F3, _F3)
                        for matrix in (_M5, _D5)),
        cache=True, parallel=True, nogil=True)
def carry_correction(matrix, tape, exponent, weight, initial, rows, logs, scale, shift):
    """D5P's second pass, using the same bin action as the painting recurrences."""
    samples, pairs, bins, width = rows.shape
    perm = np.arange(width)
    compact = not tape[0].size
    for task in prange(samples * pairs):
        s, p = np.int64(task // pairs), np.int64(task % pairs)
        law = initial[s].copy()
        moved = np.empty(width)
        work = np.empty((4, width))
        for b in range(bins):
            if not compact and weight[s, b] == 1.0 and b == bins - 1:
                continue
            log_t = apply_bin(matrix, tape, exponent, s, p, b, False, perm, law, moved, work, False)
            tempered, untempered = 0.0, 0.0
            for c in range(width):
                tempered += moved[c]
                untempered += law[c] * rows[s, p, b, c]
            if tempered > 0.0 and untempered > 0.0 and (compact or weight[s, b] != 1.0):
                delta = weight[s, b] * (math.log(untempered) + logs[s, p, b]) - math.log(tempered) - log_t
                if compact:
                    shift[s, p, b] = delta - scale[s, p, b]
                scale[s, p, b] = delta
            if tempered > 0.0:
                for c in range(width):
                    law[c] = moved[c] / tempered


@kernel(overloads=tuple((TAPE, _F2, _I2, matrix, _F3, _B2) for matrix in (_M5, _D5)),
        cache=True, parallel=True, nogil=True)
def _compile_bins(tape, exponent, representative, matrices, scales, wide_range):
    """Multiply the basis rows together, sharing each site's factors and transition coefficients."""
    starts, stops, first, second, codes, alt, transition, law, survival, absent, bypass, factors = tape
    samples, pairs, bins, width, _ = matrices.shape
    own = law.shape[2]
    labels = width - own
    chunks = (bins + MATRIX_CHUNK_BINS - 1) // MATRIX_CHUNK_BINS
    for task in prange(samples * chunks):
        s, chunk = task // chunks, task % chunks
        current = np.empty((width, width))
        moved = np.empty_like(current)
        arrival = np.empty(width)
        restart = np.empty(width)
        own_sum = np.empty(width)
        foreign_sum = np.empty(width)
        weighted = np.empty(width)
        row_max = np.empty(width)
        row_gauge = np.empty(width)
        permutation = np.empty(width, dtype=np.int64)
        for b in range(chunk * MATRIX_CHUNK_BINS, min(bins, (chunk + 1) * MATRIX_CHUNK_BINS)):
            for p in range(pairs):
                r = representative[p, b] if representative.size else p
                if r != p:
                    if labels:
                        _representative_orientation(first, second, p, r, own, permutation)
                        for i in range(width):
                            for j in range(width):
                                matrices[s, p, b, i, j] = matrices[s, r, b, permutation[i], permutation[j]]
                    else:
                        matrices[s, p, b] = matrices[s, r, b]
                    scales[s, p, b] = scales[s, r, b]
                    continue
                if bypass[s]:
                    matrices[s, p, b] = 0.0
                    matrices[s, p, b, 0, 0] = 1.0
                    scales[s, p, b] = 0.0
                    continue
                x, y = first[p], second[p]
                repeated = x == y and x < len(alt) - 1
                current[:, :] = 0.0
                for c in range(width):
                    current[c, c] = 1.0
                row_gauge[:] = 0.0
                for t in range(starts[b], stops[b]):
                    power = exponent[s, b if exponent.shape[1] == bins else t]
                    table = p
                    if codes.size:
                        code = codes[t, x]
                        table = ALLELE_FACTOR_ROWS - 1 if repeated and code == 2 else 3 * code + codes[t, y]
                    factor = factors[s, t, table]
                    for j in range(own):
                        weighted[j] = factor[j] if power == 1.0 else factor[j] ** power
                    for j in range(labels):
                        value = 0.0
                        if j != x and j != y:
                            if codes.size:
                                value = factor[own + codes[t, j]]
                            else:
                                value = (1.0 - alt[j, t]) * factor[own] + alt[j, t] * factor[own + 1]
                            if power != 1.0:
                                value = value ** power
                        weighted[own + j] = value
                    phi, q = 0.0, 1.0
                    if labels:
                        phi, q = survival[s, t], absent[s, t]
                        arrival_factor = (1.0 - q) / labels
                        # Column sums with the basis-row axis innermost (contiguous): each column still adds its
                        # entries in ascending state order, so every sum is the per-column loop's.
                        for r in range(width):
                            own_sum[r] = 0.0
                            foreign_sum[r] = 0.0
                        for i in range(own):
                            for r in range(width):
                                own_sum[r] += current[i, r]
                        for i in range(own, width):
                            for r in range(width):
                                foreign_sum[r] += current[i, r]
                        for r in range(width):
                            arrival[r] = arrival_factor * (own_sum[r] + (1.0 - phi) * foreign_sum[r])
                            restart[r] = (1.0 - phi) * q * foreign_sum[r]
                    row_max[:] = 0.0
                    for j in range(own):
                        moved[j] = 0.0
                        for i in range(own):
                            coefficient = transition[t, s, i, j]
                            for r in range(width):
                                moved[j, r] += current[i, r] * coefficient
                        for r in range(width):
                            value = moved[j, r]
                            if labels:
                                value = q * value + restart[r] * law[t, s, j]
                            value *= weighted[j]
                            moved[j, r] = value
                            row_max[r] = max(row_max[r], value)
                    for j in range(own, width):
                        for r in range(width):
                            value = (phi * current[j, r] + arrival[r]) * weighted[j]
                            moved[j, r] = value
                            row_max[r] = max(row_max[r], value)
                    current, moved = moved, current
                    # Each basis row is a separate input. Its gauge can be carried without
                    # allowing another input row's mass to underflow it.
                    for i in range(width):
                        top = row_max[i]
                        if top == 0.0:
                            row_gauge[i] = -np.inf
                        elif top < 1e-100 or top > 1e100 or t + 1 == stops[b]:
                            row_gauge[i] += math.log(top)
                            for j in range(width):
                                current[j, i] /= top
                gauge = np.max(row_gauge)
                if not math.isfinite(gauge):
                    gauge = 0.0
                own_support = False
                for i in range(width):
                    relative = row_gauge[i] - gauge
                    if labels and math.isfinite(relative) and relative < -500.0:
                        # A common binary64 gauge is unsuitable for these rows.
                        wide_range[s, chunk] = True
                    scale = math.exp(relative)
                    for j in range(width):
                        value = current[j, i] * scale
                        matrices[s, p, b, i, j] = value
                        if i < own and j < own and value > 0.0:
                            own_support = True
                if labels and not own_support:
                    # The own projection has its own gauge, including for zero support.
                    wide_range[s, chunk] = True
                scales[s, p, b] = gauge


@kernel(overloads=((_D5, _F3, _M5, _F3),), cache=True, parallel=True, nogil=True)
def _compiled_own_block(matrices, scale, own_block, own_log):
    samples, pairs, bins, own, _ = own_block.shape
    for task in prange(samples * pairs):
        s, p = task // pairs, task % pairs
        for b in range(bins):
            top = 0.0
            for i in range(own):
                for j in range(own):
                    top = max(top, matrices[s, p, b, i, j])
            own_log[s, p, b] = scale[s, p, b] + (math.log(top) if top > 0.0 else 0.0)
            for i in range(own):
                for j in range(own):
                    own_block[s, p, b, i, j] = matrices[s, p, b, i, j] / top if top > 0.0 else 0.0


def compiled_own_block(matrices, scale, own):
    shape = matrices.shape[:3]
    block = np.empty((*shape, own, own), dtype=np.float32)
    log = np.empty(shape)
    _compiled_own_block(matrices, scale, block, log)
    return block, log


@kernel(overloads=((TAPE, _F2, _F3, _I1, _I2, _M5, _F3),), cache=True, parallel=True, nogil=True)
def _project(tape, exponent, shift, bins, representative, out, logs):
    """Compile the own block from basis row actions in parallel over samples and bins."""
    samples, pairs, count, own, _ = out.shape
    width = own + (len(tape[5]) if tape[9].size else 0)
    for task in prange(samples * count):
        s, j = np.int64(task // count), task % count
        b = bins[j]
        initial = np.zeros(width)
        row = np.empty(width)
        work = np.empty((3, width))
        block = np.empty((own, own))
        gauges = np.empty(own)
        for p in range(pairs):
            r = representative[p, b] if representative.size else p
            if r != p:
                out[s, p, j] = out[s, r, j]
                logs[s, p, j] = logs[s, r, j]
                continue
            top = -np.inf
            for i in range(own):
                initial[i] = 1.0
                lg = _walk(tape, s, p, b, initial, row, work, np.bool_(False), exponent[s])
                initial[i] = 0.0
                biggest = 0.0
                for k in range(own):
                    block[i, k] = row[k]
                    biggest = max(biggest, row[k])
                gauges[i] = lg if biggest > 0.0 else -np.inf
                if biggest > 0.0:
                    top = max(top, lg + math.log(biggest))
            if not math.isfinite(top):
                top = 0.0
            for i in range(own):
                scale = math.exp(gauges[i] - top)
                for k in range(own):
                    out[s, p, j, i, k] = block[i, k] * scale
            logs[s, p, j] = top
        for p in range(pairs):
            logs[s, p, j] += shift[s, p, b]


def site_tape(parts, observed, alt, gl, cohort, epsilon, starts, stops, counts, error, loss, survival, depth, bypass):
    """Build reusable site tables and entry laws. With no arrivals the foreign axis is empty."""
    return _make_tape(parts, observed, alt, gl, cohort, epsilon, starts, stops, counts, error, loss, survival,
                      depth, bypass, np.empty(0, dtype=np.bool_))


def _make_tape(parts, observed, alt, gl, cohort, epsilon, starts, stops, counts, error, loss, survival, depth,
               bypass, hidden):
    transition, law, class_lik = parts
    alt = np.ascontiguousarray(alt, dtype=np.float64)
    first, second = np.triu_indices(len(alt))
    codes = np.ascontiguousarray(np.where(alt.T == 0.0, 0, np.where(alt.T == 1.0, 1, 2)))
    if np.any((codes == 2) & (alt.T != alt[-1, :, None])):
        codes = np.empty((0, 0), dtype=np.int64)
    observed, gl = np.array(observed, order="C"), np.array(gl, dtype=np.float64, order="C")
    foreign = np.any(depth)
    width = law.shape[2] + (3 if foreign else 0)
    raw = np.empty((len(observed), observed.shape[1], ALLELE_FACTOR_ROWS if codes.size else len(first), width))
    _site_factors(class_lik, law, observed, alt, gl, np.ascontiguousarray(cohort), float(epsilon), codes, first,
                  second, counts, error, loss, bypass, hidden, raw)
    entry = max(0, int(starts[0]) - 1)
    own_law = law[entry].copy()
    if foreign:
        # A sample with no foreign depth (QC A's corrected rows, ReadEmissionBuilder.foreign_depth) never enters its
        # foreign classes: no entry mass, no arrival. Their factor is then free, and a unit or GL-anchored one at its
        # observed sites let their backward messages, which meet no forward mass, set the max-normalisations and the
        # bin products' shared gauge over the own classes (own maxima down to 2.6e-22 on 30 kb reads, 4.23 qca
        # review). Zero leaves them only the own classes' restart. Exact: their forward mass is zero either way.
        unreachable = (depth == 0.0) & ~np.asarray(bypass, dtype=bool)
        if unreachable.any():
            raw[..., law.shape[2]:][unreachable[:, None] & observed] = 0.0
        q = np.exp(-depth[:, None] * (1.0 - survival))
        absence = np.exp(-depth)
        initial = np.column_stack((absence[:, None] * own_law,
                                   np.broadcast_to(((1.0 - absence) / len(alt))[:, None], (len(absence), len(alt)))))
    else:
        q = np.empty((0, 0))
        initial = own_law.copy()
    initial[bypass] = 0.0
    initial[bypass, 0] = 1.0
    starts, stops = np.array(starts, dtype=np.int64), np.array(stops, dtype=np.int64)
    return (starts, stops, first, second, codes, alt, transition, law, survival, q, bypass, raw), initial, own_law


def _tempered_tape(tape, weight):
    if (tape[4].size or not tape[9].size) and np.any(weight != 1.0):
        factors = np.empty_like(tape[-1])
        _temper(tape[-1], weight, tape[0], tape[1], factors)
        return (*tape[:-1], factors), np.ones_like(weight)
    return tape, weight


def _representatives(tape, *, foreign=False):
    starts, stops, first, second, _, alt = tape[:6]
    if tape[9].size and not foreign:
        return np.empty((0, 0), dtype=np.int64)
    representative = pair_representatives(alt, starts, stops, first, second)
    if foreign:
        # A background diagonal can share its own-class factor with an off-diagonal
        # pair but excludes one donor instead of two. It is the final pair.
        representative[-1] = len(first) - 1
    return representative


def _correction_rows(tape, initial, representative, shared):
    """Untempered D5P rows depend only on the common tape."""
    if shared is not None and "rows" in shared:
        return shared["rows"]
    shape = (len(initial), len(tape[2]), len(tape[0]))
    rows = np.empty((*shape, initial.shape[1]))
    logs = np.empty(shape)
    _untempered_rows(tape, representative, rows, logs)
    if shared is not None:
        shared["rows"] = rows, logs
    return rows, logs


def strength(tape, initial, weight, *, shared=None):
    """Prepare the site walk and D5P gauge at one tempering strength."""
    starts, stops, first = tape[:3]
    weight = np.ascontiguousarray(weight)
    tempered, exponent = _tempered_tape(tape, weight)
    shape = (len(initial), len(first), len(starts))
    shift = np.zeros(shape)
    if np.any(weight != 1.0):
        rows, logs = _correction_rows(tape, initial, _representatives(tape, foreign=bool(tape[9].size)), shared)
        empty = np.empty((0, 0, 0, 0, 0), dtype=np.float32)
        discarded = np.empty((0, 0, 0))
        carry_correction(empty, tempered, exponent, weight, initial, rows, logs, shift, discarded)
    return tempered, exponent, shift


def compiled_strength(tape, initial, weight, *, shared=None):
    """Materialize a bounded-width action once for chain fits, D5P and the path search."""
    weight = np.ascontiguousarray(weight)
    tempered, exponent = _tempered_tape(tape, weight)
    shape = (len(initial), len(tape[2]), len(tape[0]))
    width = initial.shape[1]
    matrices = np.empty((*shape, width, width))
    scale = np.empty(shape)
    representative = _representatives(tape, foreign=True)
    if not _build_bin_products(tempered, exponent, representative, matrices, scale):
        return None
    shift = np.zeros(shape)
    if np.any(weight != 1.0):
        rows, logs = _correction_rows(tape, initial, representative, shared)
        carry_correction(matrices, empty_tape(), exponent, weight, initial, rows, logs, scale, shift)
    return matrices, scale, shift


def _build_bin_products(tape, exponent, representative, matrices, scale):
    samples, _, bins = scale.shape
    chunks = (bins + MATRIX_CHUNK_BINS - 1) // MATRIX_CHUNK_BINS
    wide_range = np.zeros((samples, chunks), dtype=bool)
    _compile_bins(tape, exponent, representative, matrices, scale, wide_range)
    return not wide_range.any()


def compact_strength(tape, initial, weight, bins=None, *, shared=None):
    """Compile the zero-arrival action once for repeated chain fits and decodes."""
    weight = np.ascontiguousarray(weight)
    tempered, exponent = _tempered_tape(tape, weight)
    representative = _representatives(tape)
    shape = (len(initial), len(tape[2]), len(tape[0]))
    shift = np.zeros(shape)
    width = initial.shape[1]
    matrices = np.empty((*shape, width, width), dtype=np.float32)
    scale = np.empty(shape)
    if not _build_bin_products(tempered, exponent, representative, matrices, scale):
        matrices, scale = _projection(tempered, exponent, shift, np.arange(shape[2]), representative)
    if np.any(weight != 1.0):
        rows, logs = _correction_rows(tape, initial, representative, shared)
        carry_correction(matrices, empty_tape(), exponent, weight, initial, rows, logs, scale, shift)
    if bins is not None:
        matrices = np.ascontiguousarray(matrices[:, :, bins])
        scale = np.ascontiguousarray(scale[:, :, bins])
        shift = np.ascontiguousarray(shift[:, :, bins])
    return matrices, scale, shift


def hidden_action(parts, observed, alt, gl, cohort, epsilon, starts, stops, counts, error, loss, survival,
                  depth, bypass, hidden, weight, shift, *, compile=True):
    """The visible-plus-hidden action, with hidden-site PMFs at unit weight and the visible D5P correction.
    ``compile`` False keeps a positive-share action on the site walk (ReadEmissionBuilder.compiles)."""
    tape, initial, _ = _make_tape(parts, observed, alt, gl, cohort, epsilon, starts, stops, counts, error, loss,
                                  survival, depth, bypass, np.ascontiguousarray(hidden))
    # Calibrated hidden sites use weight one independently of their bin's strength. Keep the explicit site axis
    # distinct even for a grid with exactly one site per bin.
    exponent = np.ones((len(observed), observed.shape[1] + len(starts)))
    site_bins = np.repeat(np.arange(len(starts)), stops - starts)
    exponent[:, starts[0]:stops[-1]] = np.where(
        hidden[starts[0]:stops[-1]], 1.0, weight[:, site_bins]
    )
    foreign = bool(tape[9].size)
    if compile or not foreign:
        representative = _representatives(tape, foreign=foreign)
        tempered, powers = _tempered_tape(tape, exponent)
        width = initial.shape[1]
        matrices = np.empty((*shift.shape, width, width), dtype=np.float64 if foreign else np.float32)
        logs = np.empty_like(shift)
        if _build_bin_products(tempered, powers, representative, matrices, logs):
            operator = (empty_tape(), np.empty((0, 0)), initial) if foreign else None
            return matrices, logs + shift, operator
        del matrices
    if not foreign:
        matrices, logs = own_projection(tape, exponent, shift)
        return matrices, logs, None
    return np.empty((0, 0, 0, 0, 0), dtype=np.float32), shift, (tape, exponent, initial)


def _projection(tape, exponent, shift, bins, representative):
    own = tape[7].shape[2]
    shape = (*shift.shape[:2], len(bins))
    matrices = np.empty((*shape, own, own), dtype=np.float32)
    logs = np.empty(shape)
    _project(tape, exponent, shift, np.ascontiguousarray(bins), representative, matrices, logs)
    return matrices, logs


def own_projection(tape, exponent, shift, bins=None):
    """Materialize only the fixed-width own-to-own block in its own gauge."""
    if bins is None:
        bins = np.arange(shift.shape[2])
    return _projection(tape, exponent, shift, bins, _representatives(tape))
