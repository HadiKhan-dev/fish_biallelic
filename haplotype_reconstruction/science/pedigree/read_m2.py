"""Exact M2 transfers shared by the ordered allele words observed in a child-bin."""

from __future__ import annotations

import math

import numpy as np
from numba import prange, types

from ...core.kernels import bounded_team, kernel
from .certificates import dense_bridges
from .forward import _SMALLEST_NORMAL
from .kernels.read_layout import _F32_3, _F64_1, _F64_2, _F64_3, _F64_4, _I64_1, _I64_2, _reset
from .read_transfer import move_gap

_I = types.int64
_BOOL_1 = types.Array(types.boolean, 1, "C")
_RO_GL = types.Array(types.float64, 3, "C", readonly=True)
_BRIDGES = (_F64_3,) * 5 + (_I64_2, _F64_2, types.Array(types.int8, 2, "C"))
_NULL = (_F64_2, _F64_2, _F64_3)
_DENSE_CLASSES = 16
_DENSE_STATES = 8
_WORK_BYTES = 64 << 20
_STRIPE_BINS = 8
_STRIPE_CHILDREN = 8


@kernel(overloads=((_F64_2, _I64_1, _I64_1, _I64_1, _I, _I, _I),), inline="always")
def _word_codes(alt, sites, candidates, codes, founders, start, stop):
    """Named words lead the dictionary, even when a candidate repeats one."""
    count = founders + len(candidates)
    words = np.empty(count, np.int64)
    unique, named = 0, 0
    for i in range(count):
        word = i if i < founders else founders + candidates[i - founders]
        code = 0
        while code < unique:
            same = True
            for h in range(start, stop):
                site = sites[h]
                if alt[word, site] != alt[words[code], site]:
                    same = False
                    break
            if same:
                break
            code += 1
        if code == unique:
            words[unique] = word
            unique += 1
        codes[i] = code
        if i < founders:
            named = unique
    return words, named, unique


@kernel(overloads=((_F64_1, _F64_2, types.float64, types.float64),), inline="always")
def _emission(out, corners, a, b):
    if (a == 0.0 or a == 1.0) and (b == 0.0 or b == 1.0):
        code = 2 * int(a) + int(b)
        for d in range(len(out)):
            out[d] = corners[code, d]
    else:
        w00, w01 = (1.0 - a) * (1.0 - b), (1.0 - a) * b
        w10, w11 = a * (1.0 - b), a * b
        for d in range(len(out)):
            out[d] = (w00 * corners[0, d] + w01 * corners[1, d]
                      + w10 * corners[2, d] + w11 * corners[3, d])


@kernel(overloads=((_F64_1, types.float64, types.float64),), inline="always")
def _weights(out, a, b):
    out[0] = (1.0 - a) * (1.0 - b)
    out[1] = (1.0 - a) * b
    out[2] = a * (1.0 - b)
    out[3] = a * b


@kernel(overloads=((_F64_1, _F64_2, _F64_1),), inline="always")
def _weighted_emission(out, corners, weights):
    for d in range(len(out)):
        out[d] = (weights[0] * corners[0, d] + weights[1] * corners[1, d]
                  + weights[2] * corners[2, d] + weights[3] * corners[3, d])


@kernel(overloads=((_F32_3, _I, _I, _I, _F64_3),), inline="always")
def _regular(operator, base, count, classes, corners):
    """Bound positive paths and row sums before forming an unscaled product."""
    lower, upper = 0.0, 0.0
    carried = base >= 0
    for h in range(count):
        low, high = np.inf, 0.0
        for a in range(4):
            for d in range(classes):
                value = corners[h, a, d]
                low = min(low, value)
                high = max(high, value)
        if not (low > 0.0 and high < np.inf):
            return False
        lower += min(0.0, math.log(low))
        upper += max(0.0, math.log(high))
        if carried:
            small, large = np.inf, 0.0
            gap = operator[base + h]
            rows, cols = gap.shape
            for i in range(rows):
                for j in range(cols):
                    value = np.float64(gap[i, j])
                    if value < 0.0 or not np.isfinite(value):
                        return False
                    if value > 0.0:
                        small = min(small, value)
                    large = max(large, value)
            if rows < cols:
                own = rows - 1
                persistence = np.float64(gap[own, own]) - np.float64(gap[own, own + 1])
                if persistence < 0.0:
                    return False
                if persistence > 0.0:
                    small = min(small, persistence)
            if not (0.0 < large < np.inf):
                return False
            lower += min(0.0, math.log(small))
            upper += max(0.0, math.log(large * classes))
    # Leave headroom for normalization, rounding and the addition of selector cells.
    return lower >= -640.0 and upper <= 640.0


@kernel(overloads=((_F64_1, _F64_1, _F64_1, _F32_3, _reset, _I, _I, _I),), inline="always")
def _site_step(current, moved, factor, operator, reset, site, enter, leave):
    if site >= 0:
        move_gap(operator[site], current, moved, enter, reset[site])
        if leave == 1:
            total = 0.0
            for d in range(len(current)):
                total += moved[d] * factor[d]
            current[0] = total
        else:
            for d in range(leave):
                current[d] = moved[d] * factor[d]
    else:
        current[0] *= factor[0]


@kernel(overloads=((_F64_1, _F64_1, _F32_3, _I),), inline="always")
def _transpose_gap(current, moved, operator, site):
    """Backward action for a one-column product, including shared foreign renewal."""
    if site < 0:
        return
    gap = operator[site]
    rows, cols = gap.shape
    if rows == cols:
        for i in range(rows):
            moved[i] = 0.0
            for j in range(cols):
                moved[i] += np.float64(gap[i, j]) * current[j]
    else:
        own = rows - 1
        foreign = 0.0
        for j in range(own, len(current)):
            foreign += current[j]
        for i in range(own):
            moved[i] = foreign * np.float64(gap[i, own])
            for j in range(own):
                moved[i] += np.float64(gap[i, j]) * current[j]
        off = np.float64(gap[own, own + 1])
        shared = foreign * off
        for j in range(own):
            shared += np.float64(gap[own, j]) * current[j]
        persistence = np.float64(gap[own, own]) - off
        for i in range(own, len(current)):
            moved[i] = shared + persistence * current[i]
    current[:] = moved


_BANK = (_I64_2, _I64_2, _I64_2, _F64_1, _F64_3, _BOOL_1)
_BANK_SIG = (
    _I64_2, _I64_2, _I64_1, _I64_1, _F64_2, _I64_1, _I64_1,
    _reset, _F32_3, _F32_3, *_BANK,
)


@kernel(overloads=tuple((gl, *_BANK_SIG) for gl in (_F64_3, _RO_GL)), cache=True, parallel=True, nogil=True)
def _prepare_bank(gl, jobs, candidates, sites, offset, alt, read_offset, obs_offset, reset, operator, ratio,
                  codes, dimensions, plans, values, corners, regular):
    """Build disjoint child-bin dictionaries and their ordered read products in parallel."""
    founders = alt.shape[0] - gl.shape[0]
    classes = ratio.shape[2]
    for j in prange(len(jobs)):
        child, start, stop, enter, leave = jobs[j]
        count = stop - start
        base = read_offset[child] + start - obs_offset[child] if read_offset[child + 1] > read_offset[child] else -1
        factors = corners[offset[j]:offset[j + 1]]
        for h in range(count):
            site = sites[start + h]
            for a in range(4):
                g = (a + 1) // 2
                scalar = 3.0 * gl[child, site, g]
                for d in range(classes if base >= 0 else 1):
                    factors[h, a, d] = scalar * np.float64(ratio[base + h, a, d]) if base >= 0 else scalar
        width = classes if base >= 0 else 1
        if not _regular(operator, base, count, width, factors):
            regular[j] = False
            continue
        if not count:
            continue
        words, named, unique = _word_codes(alt, sites, candidates[j], codes[j], founders, start, stop)
        dimensions[j, 0], dimensions[j, 1] = named, unique
        begin, size, mode = plans[j]
        if mode == 0:
            continue
        emission = np.empty(width)
        current, moved = np.empty(width), np.empty(width)
        # Retain one workspace across all word pairs of this child-bin.
        bank = np.empty((enter if mode == 1 and (leave > 1 or enter <= _DENSE_CLASSES) else 0, width))
        for a in range(unique):
            for b in range(unique if a < named else named):
                pair = a * unique + b if a < named else named * unique + (a - named) * named + b
                table = values[begin + pair * size:begin + (pair + 1) * size]
                if mode == 2:
                    for h in range(count):
                        site = sites[start + h]
                        _weights(table[h * 4:(h + 1) * 4], alt[words[a], site], alt[words[b], site])
                elif leave == 1 and enter > _DENSE_CLASSES:
                    current[:] = 1.0
                    for h in range(count - 1, -1, -1):
                        site = sites[start + h]
                        _emission(emission, factors[h], alt[words[a], site], alt[words[b], site])
                        for d in range(width):
                            current[d] *= emission[d]
                        _transpose_gap(current, moved, operator, base + h)
                    table[:] = current[:enter]
                else:
                    product = table.reshape((enter, leave))
                    # Columns are carried by the input rows of the product, with one shared emission per site.
                    for i in range(enter):
                        for d in range(width):
                            bank[i, d] = 1.0 if i == d else 0.0
                    for h in range(count):
                        site = sites[start + h]
                        _emission(emission, factors[h], alt[words[a], site], alt[words[b], site])
                        for d in range(enter):
                            _site_step(bank[d], moved, emission, operator, reset,
                                       base + h if base >= 0 else -1, enter if h == 0 else width,
                                       leave if h == count - 1 else width)
                    for i in range(enter):
                        for d in range(leave):
                            product[i, d] = bank[i, d]


@kernel(overloads=((_F64_3, _F64_3, _F64_2, _F64_2, _I),), inline="always")
def _move_selectors(current, work, first, second, width):
    states = current.shape[0]
    for a in range(states):
        for b in range(states):
            for d in range(width):
                work[a, b, d] = 0.0
        for source in range(states):
            weight = first[a, source]
            for b in range(states):
                for d in range(width):
                    work[a, b, d] += weight * current[source, b, d]
    for a in range(states):
        for b in range(states):
            for d in range(width):
                current[a, b, d] = 0.0
            for source in range(states):
                weight = second[b, source]
                for d in range(width):
                    current[a, b, d] += weight * work[a, source, d]


@kernel(overloads=((_F64_3, _F64_3, _F64_1, _F64_1, _F64_1, _F64_1, _F64_1, _I, _F64_2, _F64_2, _F64_1, types.int8,
                    _I),), inline="always")
def _bridge_lanes(current, output, stay, alpha, beta, row_arm, column_arm, pivot, prefix, suffix, arm_total, branch,
                  width):
    """forward.apply_bridge on the first selector axis of the classes d < ``width`` of ``current`` [S, S, C] into
    ``output``, the classes as lanes: each class performs apply_bridge's operations (its Hall-star branch included) in
    their order. The second axis is this applied to the transposed planes (_bridge_lanes_second). ``prefix``,
    ``suffix`` [S + 1, C] and ``arm_total`` [C] are workspaces."""
    states = current.shape[0]
    for outer in range(states):
        if branch == 3:
            # Hall star: prefix[0] and suffix[0] hold the sums below and above the pivot (apply_bridge's left, right).
            for d in range(width):
                prefix[0, d] = 0.0
                suffix[0, d] = 0.0
            for source in range(pivot):
                weight = alpha[source]
                for d in range(width):
                    prefix[0, d] += current[source, outer, d] * weight
            for source in range(states - 1, pivot, -1):
                weight = alpha[source]
                for d in range(width):
                    suffix[0, d] += current[source, outer, d] * weight
            for destination in range(states):
                keep, arm = stay[destination], row_arm[destination]
                for d in range(width):
                    value = keep * current[destination, outer, d]
                    if destination == pivot:
                        value += prefix[0, d] + suffix[0, d]
                    value += current[pivot, outer, d] * arm
                    output[destination, outer, d] = value
            continue
        for d in range(width):
            prefix[0, d] = 0.0
            suffix[states, d] = 0.0
        for source in range(states):
            weight = alpha[source]
            for d in range(width):
                prefix[source + 1, d] = prefix[source, d] + current[source, outer, d] * weight
        for source in range(states - 1, -1, -1):
            weight = alpha[source]
            for d in range(width):
                suffix[source, d] = suffix[source + 1, d] + current[source, outer, d] * weight
        if pivot >= 0:
            for d in range(width):
                arm_total[d] = 0.0
            for source in range(states):
                weight = column_arm[source]
                for d in range(width):
                    arm_total[d] += current[source, outer, d] * weight
        for destination in range(states):
            keep, spread, arm = stay[destination], beta[destination], row_arm[destination]
            for d in range(width):
                value = keep * current[destination, outer, d] + spread * (prefix[destination, d]
                                                                           + suffix[destination + 1, d])
                if pivot >= 0:
                    value += current[pivot, outer, d] * arm
                    if destination == pivot:
                        value += arm_total[d]
                output[destination, outer, d] = value


@kernel(overloads=((_F64_3, _F64_3, _F64_1, _F64_1, _F64_1, _F64_1, _F64_1, _I, _F64_2, _F64_2, _F64_1, types.int8,
                    _I),), inline="always")
def _bridge_lanes_second(current, output, stay, alpha, beta, row_arm, column_arm, pivot, prefix, suffix, arm_total,
                         branch, width):
    """_bridge_lanes on the second selector axis (apply_bridge with axis 1)."""
    states = current.shape[0]
    for outer in range(states):
        if branch == 3:
            for d in range(width):
                prefix[0, d] = 0.0
                suffix[0, d] = 0.0
            for source in range(pivot):
                weight = alpha[source]
                for d in range(width):
                    prefix[0, d] += current[outer, source, d] * weight
            for source in range(states - 1, pivot, -1):
                weight = alpha[source]
                for d in range(width):
                    suffix[0, d] += current[outer, source, d] * weight
            for destination in range(states):
                keep, arm = stay[destination], row_arm[destination]
                for d in range(width):
                    value = keep * current[outer, destination, d]
                    if destination == pivot:
                        value += prefix[0, d] + suffix[0, d]
                    value += current[outer, pivot, d] * arm
                    output[outer, destination, d] = value
            continue
        for d in range(width):
            prefix[0, d] = 0.0
            suffix[states, d] = 0.0
        for source in range(states):
            weight = alpha[source]
            for d in range(width):
                prefix[source + 1, d] = prefix[source, d] + current[outer, source, d] * weight
        for source in range(states - 1, -1, -1):
            weight = alpha[source]
            for d in range(width):
                suffix[source, d] = suffix[source + 1, d] + current[outer, source, d] * weight
        if pivot >= 0:
            for d in range(width):
                arm_total[d] = 0.0
            for source in range(states):
                weight = column_arm[source]
                for d in range(width):
                    arm_total[d] += current[outer, source, d] * weight
        for destination in range(states):
            keep, spread, arm = stay[destination], beta[destination], row_arm[destination]
            for d in range(width):
                value = keep * current[outer, destination, d] + spread * (prefix[destination, d]
                                                                           + suffix[destination + 1, d])
                if pivot >= 0:
                    value += current[outer, pivot, d] * arm
                    if destination == pivot:
                        value += arm_total[d]
                output[outer, destination, d] = value


@kernel(overloads=((_F64_3, _F64_3, types.float64, _F64_1, _I, _I),), inline="always")
def _excursion_lanes(current, output, probability, total, axis, width):
    """forward.apply_excursion on selector ``axis`` of the classes d < ``width``, the classes as lanes (``total`` [C]
    a workspace holding each class's total, then its share probability * total / S)."""
    states = current.shape[0]
    keep = 1.0 - probability
    for outer in range(states):
        for d in range(width):
            total[d] = 0.0
        if axis == 0:
            for source in range(states):
                for d in range(width):
                    total[d] += current[source, outer, d]
            for d in range(width):
                total[d] = probability * total[d] / states
            for destination in range(states):
                for d in range(width):
                    output[destination, outer, d] = keep * output[destination, outer, d] + total[d]
        else:
            for source in range(states):
                for d in range(width):
                    total[d] += current[outer, source, d]
            for d in range(width):
                total[d] = probability * total[d] / states
            for destination in range(states):
                for d in range(width):
                    output[outer, destination, d] = keep * output[outer, destination, d] + total[d]


@kernel(overloads=((_F64_3, _F64_3, _I, _I, _I, *_BRIDGES, _F64_2, _F64_2, _F64_1, _I),), inline="always")
def _advance_lanes(current, work, p, q, boundary, stay, alpha, beta, row_arm, column_arm, pivot, excursion, branch,
                   prefix, suffix, total, width):
    """forward.advance in mode 2 (parents ``p`` then ``q``) on the classes d < ``width`` of ``current`` [S, S, C], the
    classes as lanes; the result returns to ``current``. Each class's values are advance's on that class's plane."""
    _bridge_lanes(current, work, stay[p, boundary], alpha[p, boundary], beta[p, boundary], row_arm[p, boundary],
                  column_arm[p, boundary], pivot[p, boundary], prefix, suffix, total, branch[p, boundary], width)
    probability = excursion[p, boundary]
    if probability > 0.0:
        _excursion_lanes(current, work, probability, total, 0, width)
    _bridge_lanes_second(work, current, stay[q, boundary], alpha[q, boundary], beta[q, boundary],
                         row_arm[q, boundary], column_arm[q, boundary], pivot[q, boundary], prefix, suffix, total,
                         branch[q, boundary], width)
    probability = excursion[q, boundary]
    if probability > 0.0:
        _excursion_lanes(work, current, probability, total, 1, width)


_ROW_ARGS = (
    _I64_2, _I64_2, _I64_1, _I64_1, _I64_2, _I64_1, _I64_1, _F64_2,
    _reset, _F32_3, *_BANK, _F64_3, _F64_2, _F64_4, *_BRIDGES, *_NULL,
    _I, _I, _I64_1, _F64_4, _F64_1, _F64_2,
)


@kernel(overloads=(_ROW_ARGS,), cache=True, parallel=True, nogil=True)
def _advance_rows(rows, parents, offsets, obs_offset, jobs, sites, read_offset, alt, reset, operator,
                  codes, dimensions, plans, values, corners, regular, pi, initial, bridge, stay, alpha, beta,
                  row_arm, column_arm, pivot, excursion, branch, null_diagonal, null_off, null_frequency,
                  first_bin, bins, piece, state, totals, pieces):
    """Independent trios, with continuous state over all stripes of their component."""
    states, classes = state.shape[1], state.shape[3]
    founders = states - 1
    for r in prange(len(rows)):
        child, p, q = rows[r]
        local, slot_p, slot_q = parents[r]
        current = state[r]
        if first_bin == 0:
            for a in range(states):
                for b in range(states):
                    for d in range(classes):
                        current[a, b, d] = pi[p, 0, a] * pi[q, 0, b] * initial[child, d]
        work = np.empty_like(current)
        lane_prefix, lane_suffix = np.empty((states + 1, classes)), np.empty((states + 1, classes))
        emission, moved = np.empty(classes), np.empty(classes)
        for z in range(bins):
            if not np.isfinite(totals[r]):
                break
            j = local * bins + z
            if not regular[j]:
                totals[r] = np.nan
                break
            block = first_bin + z
            _, start, stop, enter, leave = jobs[j]
            if block > 0:
                if bridge.shape[0]:
                    _move_selectors(current, work, bridge[p, block - 1], bridge[q, block - 1], enter)
                else:
                    # forward.advance on each live class plane, the classes as lanes (the same values per class)
                    _advance_lanes(current, work, p, q, block - 1, stay, alpha, beta, row_arm, column_arm, pivot,
                                   excursion, branch, lane_prefix, lane_suffix, moved, enter)
            carried = read_offset[child + 1] > read_offset[child]
            base = read_offset[child] + start - obs_offset[child] if carried else -1
            width = classes if carried else 1
            mass = 0.0
            named, unique = dimensions[j]
            begin, size, mode = plans[j]
            for a in range(states):
                u = a if a < founders else founders + slot_p
                for b in range(states):
                    v = b if b < founders else founders + slot_q
                    cell = current[a, b]
                    if stop > start:
                        x, y = codes[j, u], codes[j, v]
                        pair = -1
                        if x < named:
                            pair = x * unique + y
                        elif y < named:
                            pair = named * unique + (x - named) * named + y
                        if mode == 1 and pair >= 0:
                            table = values[begin + pair * size:begin + (pair + 1) * size]
                            for d in range(leave):
                                moved[d] = 0.0
                            for i in range(enter):
                                weight = cell[i]
                                for d in range(leave):
                                    moved[d] += weight * table[i * leave + d]
                            cell[:leave] = moved[:leave]
                        else:
                            word_p = a if a < founders else founders + p
                            word_q = b if b < founders else founders + q
                            for h in range(stop - start):
                                if mode == 2 and pair >= 0:
                                    pos = begin + pair * size + h * 4
                                    _weighted_emission(emission[:width], corners[offsets[j] + h],
                                                       values[pos:pos + 4])
                                else:
                                    site = sites[start + h]
                                    _emission(emission[:width], corners[offsets[j] + h],
                                              alt[word_p, site], alt[word_q, site])
                                _site_step(cell, moved, emission, operator, reset,
                                           base + h if carried else -1, enter if h == 0 else width,
                                           leave if h == stop - start - 1 else width)
                    elif enter > leave:
                        for d in range(1, enter):
                            cell[0] += cell[d]
                    for d in range(leave):
                        mass += cell[d]
            if not (_SMALLEST_NORMAL <= mass < np.inf):
                totals[r] = np.nan
                break
            inverse = 1.0 / mass
            for a in range(states):
                for b in range(states):
                    for d in range(leave):
                        current[a, b, d] *= inverse
            increment = math.log(mass)
            totals[r] += increment
            if pieces.shape[1]:
                pieces[r, piece[block]] += increment


def _stripe_bank(jobs, parent_counts, founders, classes):
    """Plan transfer storage in doubles. The factored representation preserves each rounded gap.

    Two wide axes use the emission factors and the structured gaps in observation order. A one-column
    transfer is built by a backward sweep, so all builds and applications are at most cubic in K when
    the foreign axis grows with K. Oversized tables still share the per-site emission corners.
    """
    count = jobs[:, 2] - jobs[:, 1]
    enter, leave = jobs[:, 3], jobs[:, 4]
    dense = (classes <= _DENSE_CLASSES) | (enter == 1) | (leave == 1)
    cells = np.where(dense, enter * leave, count * 4)
    pairs = founders * (founders + 2 * parent_counts)
    needed = np.where(count > 0, pairs * cells, 0)
    plans = np.empty((len(jobs), 3), np.int64)
    size = 0
    for i, n in enumerate(needed):
        mode = 1 if dense[i] else 2
        if not n or size + n > _WORK_BYTES // 8:
            plans[i] = (size, 0, 0)
        else:
            plans[i] = (size, cells[i], mode)
            size += int(n)
    return plans, np.empty(size, np.float64)


def component_scores(projection, terms, trios, axis, n_pieces, claim_threads):
    """Return ordered component totals and the requested pieces, with NaN pieces on reference rows."""
    from . import read_score

    p, t = projection, terms
    totals = np.full(len(trios), np.nan)
    pieces = np.full((len(trios), n_pieces), np.nan)
    active = np.flatnonzero(np.all(p.available[trios[:, 1:]], axis=1))
    if t.mismatch != 0.0:
        active = active[:0]
    if len(active):
        active = active[np.argsort(trios[active, 0], kind="stable")]
        rows = np.ascontiguousarray(trios[active])
        totals[active], pieces[active] = _share_component(p, t, rows, axis, n_pieces, claim_threads)
    reference = np.flatnonzero(np.isnan(totals))
    if len(reference):
        totals[reference] = read_score.component_m2(p, t, trios[reference], claim_threads=claim_threads)
        pieces[reference] = np.nan
    return totals, pieces


def _share_component(p, t, rows, axis, n_pieces, claim_threads):
    states, bins, classes = p.n_states, p.n_bins, t.n_classes
    founders = states - 1
    alt = np.ascontiguousarray(np.vstack((p.state_alt[:-1], p.candidate_alt[:, :, -1])))
    bridge_args = (p.stay, p.alpha, p.beta, p.row_arm, p.column_arm, p.pivot, p.excursion, p.branch)
    claim_threads()
    with bounded_team(max(1, len(p.pi) * (bins - 1))):
        if states <= _DENSE_STATES:
            dense = dense_bridges(*bridge_args[:-2], p.branch, p.excursion)
            bridge = np.ascontiguousarray(dense.swapaxes(-1, -2))
        else:
            bridge = np.empty((0, 0, 0, 0))
    totals = np.zeros(len(rows))
    pieces = np.zeros((len(rows), n_pieces))
    children, row_cuts = np.unique(rows[:, 0], return_index=True)
    row_cuts = np.append(row_cuts, len(rows))
    read_offset = t.read_offset if classes > 1 else np.zeros_like(t.read_offset)
    obs_start = np.ascontiguousarray(t.site_start[:, 0])
    max_rows = max(1, _WORK_BYTES // (8 * states * states * classes))
    first = 0
    # Bound resident forward states. Only an oversized child's independent rows are cut mid-child.
    while first < len(rows):
        group = int(np.searchsorted(row_cuts, first, side="right")) - 1
        high = min(len(children), group + _STRIPE_CHILDREN)
        last = int(row_cuts[high])
        if last - first > max_rows:
            high = max(group + 1, int(np.searchsorted(row_cuts, first + max_rows, side="right")) - 1)
            last = min(int(row_cuts[high]), first + max_rows)
        selected = rows[first:last]
        children_here, local = np.unique(selected[:, 0], return_inverse=True)
        candidates = [np.unique(selected[local == i, 1:]) for i in range(len(children_here))]
        parent_counts = np.array([len(group) for group in candidates], dtype=np.int64)
        # Repeated padding entries map to the same word and cannot increase dictionary size.
        parents = np.empty((len(children_here), max(parent_counts)), dtype=np.int64)
        parent_slots = np.empty_like(selected)
        parent_slots[:, 0] = local
        for i, group_parents in enumerate(candidates):
            parents[i, :len(group_parents)] = group_parents
            parents[i, len(group_parents):] = group_parents[-1]
            mask = local == i
            parent_slots[mask, 1:] = np.searchsorted(group_parents, selected[mask, 1:])
        state = np.empty((len(selected), states, states, classes))
        carried = np.diff(read_offset)[children_here] > 0
        for b0 in range(0, bins, _STRIPE_BINS):
            b1 = min(bins, b0 + _STRIPE_BINS)
            width = b1 - b0
            starts = t.site_start[children_here, b0:b1]
            stops = t.site_stop[children_here, b0:b1]
            enter = np.where(carried[:, None], t.read_width[children_here, b0:b1], 1)
            leave = np.where(carried[:, None], t.read_width[children_here, b0 + 1:b1 + 1], 1)
            jobs = np.column_stack((np.repeat(children_here, width), starts.ravel(), stops.ravel(),
                                    enter.ravel(), leave.ravel()))
            offsets = np.concatenate(([0], np.cumsum(jobs[:, 2] - jobs[:, 1])))
            candidates_here = np.repeat(parents, width, axis=0)
            codes = np.empty((len(jobs), founders + parents.shape[1]), dtype=np.int64)
            dimensions = np.zeros((len(jobs), 2), dtype=np.int64)
            regular = np.ones(len(jobs), dtype=bool)
            plans, values = _stripe_bank(jobs, np.repeat(parent_counts, width), founders, classes)
            corners = np.empty((offsets[-1], 4, classes))
            claim_threads()
            with bounded_team(len(jobs)):
                _prepare_bank(t.gl, jobs, candidates_here, t.site_index, offsets, alt, read_offset, obs_start,
                              t.read_reset, t.read_operator, t.read_ratio, codes, dimensions, plans, values,
                              corners, regular)
            with bounded_team(len(selected)):
                _advance_rows(selected, parent_slots, offsets, obs_start, jobs, t.site_index, read_offset, alt,
                              t.read_reset, t.read_operator, codes, dimensions, plans, values, corners, regular,
                              p.pi, t.read_initial, bridge, *bridge_args, t.null_diagonal, t.null_off,
                              t.null_frequency, b0, width, axis, state, totals[first:last], pieces[first:last])
        first = last
    return totals, pieces
