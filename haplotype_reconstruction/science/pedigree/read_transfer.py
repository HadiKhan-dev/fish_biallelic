"""Propagate the child's read classes within a painting bin and build reusable class transfers."""

from __future__ import annotations

import numpy as np
from numba import prange, types

from ...core.kernels import kernel
from .kernels.read_transfer import OVERLOADS


@kernel(overloads=OVERLOADS["read_transfer.read_ratio"], cache=True, inline="always", nogil=True)
def read_ratio(out, ratio, slot, a, b, gl):
    """out[C] = R_c(a, b) at a read slot: the stored configuration ratio at a corner (a, b in {0, 1}), else
    sum_g W_g(a, b) L'_g R_c(g) / sum_g W_g(a, b) L'_g (W the product law, L' = ``gl`` the child's mixed genotype
    likelihoods; one where that is zero)."""
    classes = out.shape[0]
    if (a == 0.0 or a == 1.0) and (b == 0.0 or b == 1.0):
        config = 2 * int(a) + int(b)
        for c in range(classes):
            out[c] = float(ratio[slot, config, c])
        return
    w00 = (1.0 - a) * (1.0 - b) * gl[0]
    w01 = (1.0 - a) * b * gl[1]
    w10 = a * (1.0 - b) * gl[1]
    w11 = a * b * gl[2]
    total = w00 + w01 + w10 + w11
    if total <= 0.0:
        for c in range(classes):
            out[c] = 1.0
        return
    for c in range(classes):
        out[c] = (
            w00 * float(ratio[slot, 0, c])
            + w01 * float(ratio[slot, 1, c])
            + w10 * float(ratio[slot, 2, c])
            + w11 * float(ratio[slot, 3, c])
        ) / total


@kernel(overloads=((types.float32[:, ::1], types.float64[::1], types.float64[::1], types.int64,
                    types.boolean),), cache=True, inline="always", nogil=True)
def move_gap(operator, column, moved, width, reset):
    """Apply a rounded gap to one live read-class vector."""
    classes = len(column)
    rows, columns = operator.shape
    if reset:
        mass = 0.0
        for i in range(width):
            mass += column[i]
        for j in range(classes):
            moved[j] = mass * float(operator[0, min(j, rows - 1)])
    elif rows == columns:
        moved[:] = 0.0
        for i in range(width):
            for j in range(classes):
                moved[j] += column[i] * float(operator[i, j])
    elif rows == 6:
        _move_structured(operator, column, moved, width, 5)
    else:
        _move_structured(operator, column, moved, width, rows - 1)


@kernel(overloads=((types.float32[:, ::1], types.float64[::1], types.float64[::1], types.int64, types.int64),
                   (types.float32[:, ::1], types.float64[::1], types.float64[::1], types.int64, types.literal(5))),
        cache=True, inline="always", nogil=True)
def _move_structured(operator, column, moved, width, own):
    foreign_mass = 0.0
    for i in range(own, width):
        foreign_mass += column[i]
    for j in range(own):
        value = 0.0
        for i in range(own):
            value += column[i] * float(operator[i, j])
        moved[j] = value + foreign_mass * float(operator[own, j])
    off = np.float64(operator[own, own + 1])
    persistence = np.float64(operator[own, own]) - off
    renewal = foreign_mass * off
    for i in range(own):
        renewal += column[i] * float(operator[i, own])
    for j in range(own, len(column)):
        moved[j] = persistence * column[j] + renewal


@kernel(overloads=OVERLOADS["read_transfer.carry_bin"], cache=True, inline="always", nogil=True)
def carry_bin(
    current,
    mix,
    factor,
    column,
    child,
    block,
    first_source,
    second_source,
    candidate_alt,
    state_alt,
    gl,
    site_index,
    site_start,
    site_stop,
    read_offset,
    read_reset, read_width, read_operator,
    read_ratio_table,
    class_cache,
    local,
    moved,
):
    """Use one native carry body for the signed indices."""
    _carry_bin(
        current, mix, factor, column, np.int64(child), block, np.int64(first_source), np.int64(second_source),
        candidate_alt, state_alt, gl, site_index, site_start, site_stop, read_offset, read_reset, read_width,
        read_operator, read_ratio_table, class_cache, local, moved,
    )


@kernel(overloads=OVERLOADS["read_transfer._carry_bin"], cache=True, nogil=True)
def _carry_bin(
    current,
    mix,
    factor,
    column,
    child,
    block,
    first_source,
    second_source,
    candidate_alt,
    state_alt,
    gl,
    site_index,
    site_start,
    site_stop,
    read_offset,
    read_reset, read_width, read_operator,
    read_ratio_table,
    class_cache,
    local,
    moved,
):
    """In place on the chain's forward state ``current`` [C, S, S]: the class part of the child's observed markers of
    bin ``block`` (each marker's class operator from the child's previous observed marker, applied to every state pair,
    then each pair's read factor R_c(a, b)). Slot 1's
    allele comes from candidate ``first_source``'s projection (candidate_alt), the panel's (state_alt) when it is
    negative; slot 2's likewise from ``second_source``. No change with one class or for a child without read slots.
    ``class_cache`` (class_transfers of the child's batch, ``local`` its index there; None: none) holds the bin's
    product for every state pair whose alleles no candidate changes (named states, and any state of a panel slot):
    those pairs take it in one vector-matrix product. ``mix`` [C, S, S], ``factor``, ``column`` and ``moved`` [C] are
    workspaces."""
    classes = current.shape[0]
    if classes == 1:
        return
    start = read_offset[child]
    enter, leave = read_width[child, block], read_width[child, block + 1]
    first_slot, stop_slot = site_start[child, block], site_stop[child, block]
    if read_offset[child + 1] == start or first_slot == stop_slot:
        # An empty initial bin may still precede a reset (or the end of the observations).
        if enter > leave:
            for c in range(1, enter):
                current[0] += current[c]
        return
    states = current.shape[1]
    base = start - site_start[child, 0]
    last = states - 1
    structured = read_operator.shape[1] != read_operator.shape[2]
    cached = class_cache is not None
    if class_cache is not None and structured:
        cached = np.isfinite(class_cache[local, block, 0, 0, 0, 0])
    if class_cache is not None and cached:
        fixed_first = states if first_source < 0 else last
        fixed_second = states if second_source < 0 else last
        if structured:
            # Run the class contraction over contiguous state-pair planes. The BACKGROUND column is
            # copied back only when it belongs to the panel. Per cell, the input classes still add in order.
            cells = states * states
            stop = fixed_first * states
            source = current.reshape((classes, cells))
            target = mix.reshape((classes, cells))
            bank = class_cache[local, block].reshape((classes, classes, cells))
            for j in range(leave):
                for cell in range(stop):
                    target[j, cell] = 0.0
            for i in range(enter):
                for j in range(leave):
                    for cell in range(stop):
                        target[j, cell] += source[i, cell] * bank[i, j, cell]
            for j in range(leave):
                for first in range(fixed_first):
                    for second in range(fixed_second):
                        current[j, first, second] = mix[j, first, second]
        else:
            for first in range(fixed_first):
                for second in range(fixed_second):
                    transfer = class_cache[local, block, first, second]
                    if enter == 1 and leave == 1:
                        current[0, first, second] *= transfer[0, 0]
                    else:
                        for j in range(leave):
                            moved[j] = 0.0
                        for i in range(enter):
                            weight = current[i, first, second]
                            for j in range(leave):
                                moved[j] += weight * transfer[i, j]
                        for j in range(leave):
                            current[j, first, second] = moved[j]
        if fixed_first == states and fixed_second == states:
            return
    elif (read_operator.shape[1] == read_operator.shape[2] and enter == classes and leave == classes
          and not np.any(read_reset[base + first_slot:base + stop_slot])):
        for slot in range(site_start[child, block], site_stop[child, block]):
            k = base + slot
            site = site_index[slot]
            for j in range(classes):
                for first in range(states):
                    for second in range(states):
                        mix[j, first, second] = 0.0
            for i in range(classes):
                for j in range(classes):
                    weight = float(read_operator[k, i, j])
                    if weight != 0.0:
                        for first in range(states):
                            for second in range(states):
                                mix[j, first, second] += weight * current[i, first, second]
            for first in range(states):
                a = state_alt[first, site] if first_source < 0 else candidate_alt[first_source, site, first]
                for second in range(states):
                    b = state_alt[second, site] if second_source < 0 else candidate_alt[second_source, site, second]
                    for c in range(classes):
                        column[c] = mix[c, first, second]
                    read_ratio(factor, read_ratio_table, k, a, b, gl[child, site])
                    for c in range(classes):
                        current[c, first, second] = column[c] * factor[c]
        return

    first_start = last if cached and second_source < 0 else 0
    if cached and structured:
        _carry_background(current, mix, factor, child, first_source, second_source, candidate_alt,
                          state_alt, gl, site_index, read_reset, read_operator, read_ratio_table, base,
                          first_slot, stop_slot, enter, leave)
        return
    if structured:
        # Uncached structured bin: every state pair advances as a lane of the class planes (the values of the
        # per-pair recurrence below, which it replaces for structured gaps).
        _carry_lanes(current, mix, child, first_source, second_source, candidate_alt, state_alt, gl, site_index,
                     read_reset, read_operator, read_ratio_table, base, first_slot, stop_slot, enter, leave)
        return
    if not cached or enter != 1 or leave != 1:
        # Keep a coupled bin's live class vector in the workspace while its
        # markers advance. Only uncached founder pairs need the recurrence.
        for first in range(first_start, states):
            candidate_row = first == last and first_source >= 0
            second_start = 0 if not cached or candidate_row else last
            for second in range(second_start, states):
                for c in range(enter):
                    column[c] = current[c, first, second]
                width = enter
                for slot in range(first_slot, stop_slot):
                    k = base + slot
                    site = site_index[slot]
                    a = state_alt[first, site] if first_source < 0 else candidate_alt[first_source, site, first]
                    b = state_alt[second, site] if second_source < 0 else candidate_alt[second_source, site, second]
                    read_ratio(factor, read_ratio_table, k, a, b, gl[child, site])
                    move_gap(read_operator[k], column, moved, width, read_reset[k])
                    for j in range(classes):
                        moved[j] *= factor[j]
                    width = leave if slot + 1 == stop_slot else (1 if read_reset[k + 1] else classes)
                    if width == 1:
                        mass = 0.0
                        for c in range(classes):
                            mass += moved[c]
                        column[0] = mass
                    else:
                        for c in range(classes):
                            column[c] = moved[c]
                for c in range(leave):
                    current[c, first, second] = column[c]
        return

    # The remaining founder pairs advance together through each marker.
    # A candidate's BACKGROUND allele is identical across the named allele
    # pairs in its row or column. Reuse the factor for partner allele 0 or
    # 1 within each orientation (four contiguous vectors in the workspace).
    bank = mix.reshape((states * states, classes))
    width = enter
    for slot in range(first_slot, stop_slot):
        k = base + slot
        site = site_index[slot]
        reset = read_reset[k]
        next_width = leave if slot + 1 == stop_slot else (1 if read_reset[k + 1] else classes)
        cached = 0
        for first in range(first_start, states):
            candidate_row = first == last and first_source >= 0
            second_start = 0 if candidate_row else last
            a = state_alt[first, site] if first_source < 0 else candidate_alt[first_source, site, first]
            for second in range(second_start, states):
                b = state_alt[second, site] if second_source < 0 else candidate_alt[second_source, site, second]
                ratio = factor
                ready = False
                if states >= 2:
                    key = -1
                    if candidate_row:
                        if b == 0.0 or b == 1.0:
                            key = int(b)
                    elif a == 0.0 or a == 1.0:
                        key = 2 + int(a)
                    if key >= 0:
                        ratio = bank[key]
                        mask = 1 << key
                        ready = (cached & mask) != 0
                        cached |= mask
                if not ready:
                    read_ratio(ratio, read_ratio_table, k, a, b, gl[child, site])
                if reset:
                    mass = current[0, first, second]
                    for i in range(1, width):
                        mass += current[i, first, second]
                    for j in range(classes):
                        moved[j] = mass * float(read_operator[k, 0, j]) * ratio[j]
                else:
                    for j in range(classes):
                        moved[j] = 0.0
                    for i in range(width):
                        weight = current[i, first, second]
                        for j in range(classes):
                            moved[j] += weight * float(read_operator[k, i, j])
                    for j in range(classes):
                        moved[j] *= ratio[j]
                if next_width == 1:
                    mass = 0.0
                    for c in range(classes):
                        mass += moved[c]
                    current[0, first, second] = mass
                else:
                    for c in range(classes):
                        current[c, first, second] = moved[c]
        width = next_width


@kernel(overloads=((types.float32[:, ::1], types.float64[:, ::1], types.float64[:, ::1],
                    types.int64, types.boolean),), cache=True, inline="always", nogil=True)
def _move_structured_lanes(operator, source, target, width, reset):
    """The structured gap on a contiguous plane of state pairs. Each lane keeps move_gap's reduction order."""
    classes, lanes = source.shape
    own = operator.shape[0] - 1
    if reset:
        for cell in range(lanes):
            target[0, cell] = 0.0
        for i in range(width):
            for cell in range(lanes):
                target[0, cell] += source[i, cell]
        for j in range(classes - 1, 0, -1):
            weight = np.float64(operator[0, min(j, own)])
            for cell in range(lanes):
                target[j, cell] = target[0, cell] * weight
        for cell in range(lanes):
            target[0, cell] *= np.float64(operator[0, 0])
        return

    # Hold the sum of the foreign input classes in the first foreign output plane during the own movement.
    for cell in range(lanes):
        target[own, cell] = 0.0
    for i in range(own, width):
        for cell in range(lanes):
            target[own, cell] += source[i, cell]
    for j in range(own):
        for cell in range(lanes):
            target[j, cell] = 0.0
        for i in range(own):
            weight = np.float64(operator[i, j])
            for cell in range(lanes):
                target[j, cell] += source[i, cell] * weight
        weight = np.float64(operator[own, j])
        for cell in range(lanes):
            target[j, cell] += target[own, cell] * weight
    off = np.float64(operator[own, own + 1])
    persistence = np.float64(operator[own, own]) - off
    for cell in range(lanes):
        target[own, cell] *= off
    for i in range(own):
        weight = np.float64(operator[i, own])
        for cell in range(lanes):
            target[own, cell] += source[i, cell] * weight
    for j in range(own + 1, classes):
        for cell in range(lanes):
            target[j, cell] = persistence * source[j, cell] + target[own, cell]
    for cell in range(lanes):
        target[own, cell] = persistence * source[own, cell] + target[own, cell]


@kernel(overloads=OVERLOADS["read_transfer._carry_background"], cache=True, nogil=True)
def _carry_background(current, mix, factor, child, first_source, second_source, candidate_alt, state_alt, gl,
                      site_index, read_reset, read_operator, read_ratio_table, base, first_slot, stop_slot, enter,
                      leave):
    """Carry just the uncached BACKGROUND pairs in compact class planes throughout the bin.

    The first lanes present are the whole last row when the first parent supplies BACKGROUND. The remaining
    lanes are the last column when the second parent supplies BACKGROUND. Named alleles take one of three
    values at any marker (REF, ALT or the shared panel frequency), so they share the ratio by orientation.
    """
    classes, states, _ = current.shape
    last = states - 1
    row_cells = states if first_source >= 0 else 0
    col_cells = states - int(first_source >= 0) if second_source >= 0 else 0
    cells = row_cells + col_cells
    size = cells * classes
    if mix.size >= 2 * size + 6 * classes:
        workspace = mix.reshape(mix.size)
    else:
        workspace = np.empty(2 * size + 6 * classes)
    source = workspace[:size].reshape((classes, cells))
    target = workspace[size:2 * size].reshape((classes, cells))
    bank = workspace[2 * size:2 * size + 6 * classes].reshape((6, classes))
    for c in range(enter):
        for x in range(row_cells):
            source[c, x] = current[c, last, x]
        for x in range(col_cells):
            source[c, row_cells + x] = current[c, x, last]
    width = enter
    for slot in range(first_slot, stop_slot):
        k = base + slot
        site = site_index[slot]
        _move_structured_lanes(read_operator[k], source, target, width, read_reset[k])
        width = leave if slot + 1 == stop_slot else (1 if read_reset[k + 1] else classes)
        seen = 0
        for cell in range(cells):
            first = last if cell < row_cells else cell - row_cells
            second = cell if cell < row_cells else last
            row = first == last and first_source >= 0
            col = second == last and second_source >= 0
            a = candidate_alt[first_source, site, last] if row else state_alt[first, site]
            b = candidate_alt[second_source, site, last] if col else state_alt[second, site]
            ratio = factor
            key = -1
            if not col:
                key = 0 if b == 0.0 else (1 if b == 1.0 else 2)
            elif not row:
                key = 3 if a == 0.0 else (4 if a == 1.0 else 5)
            if key < 0:
                read_ratio(ratio, read_ratio_table, k, a, b, gl[child, site])
            else:
                ratio = bank[key]
                bit = 1 << key
                if not seen & bit:
                    read_ratio(ratio, read_ratio_table, k, a, b, gl[child, site])
                    seen |= bit
            if width == 1:
                mass = 0.0
                for c in range(classes):
                    mass += target[c, cell] * ratio[c]
                source[0, cell] = mass
            else:
                for c in range(classes):
                    source[c, cell] = target[c, cell] * ratio[c]
    for c in range(leave):
        for x in range(row_cells):
            current[c, last, x] = source[c, x]
        for x in range(col_cells):
            current[c, x, last] = source[c, row_cells + x]


@kernel(overloads=OVERLOADS["read_transfer._carry_lanes"], cache=True, nogil=True)
def _carry_lanes(current, mix, child, first_source, second_source, candidate_alt, state_alt, gl, site_index,
                 read_reset, read_operator, read_ratio_table, base, first_slot, stop_slot, enter, leave):
    """Carry every state pair of an uncached structured bin through its observed markers, in place on ``current``
    [C, S, S] (``mix`` a workspace of the same shape).

    The per-pair recurrence (at each marker move_gap, then the pair's read factor R(a, b); a unit width sums the class
    mass) with the S^2 state pairs as contiguous lanes of the class planes: each lane performs that recurrence's
    operations in its order (_move_structured_lanes keeps move_gap's), so every value is the per-pair route's, at
    O(S^2 (own^2 + C)) per marker and no stored transfers. A marker's factor depends on the pair's alleles only: the
    corner configurations (both alleles 0 or 1) share one factor each; a pair with a fractional allele gets its own."""
    classes, states, _ = current.shape
    cells = states * states
    source = current.reshape((classes, cells))
    target = mix.reshape((classes, cells))
    values = np.empty(5 * classes + 2 * states)
    corners = values[:4 * classes].reshape((4, classes))
    factor = values[4 * classes:5 * classes]
    first_alt = values[5 * classes:5 * classes + states]
    second_alt = values[5 * classes + states:]
    indices = np.empty(2 * cells, np.int64)
    key = indices[:cells]
    pending = indices[cells:]
    width = enter
    for slot in range(first_slot, stop_slot):
        k = base + slot
        site = site_index[slot]
        for x in range(states):
            first_alt[x] = state_alt[x, site] if first_source < 0 else candidate_alt[first_source, site, x]
            second_alt[x] = state_alt[x, site] if second_source < 0 else candidate_alt[second_source, site, x]
        seen = 0
        count = 0
        for x in range(states):
            a = first_alt[x]
            corner_a = a == 0.0 or a == 1.0
            for y in range(states):
                b = second_alt[y]
                cell = x * states + y
                if corner_a and (b == 0.0 or b == 1.0):
                    code = 2 * int(a) + int(b)
                    if not seen & (1 << code):
                        read_ratio(corners[code], read_ratio_table, k, a, b, gl[child, site])
                        seen |= 1 << code
                    key[cell] = code
                else:
                    key[cell] = -1
                    pending[count] = cell
                    count += 1
        _move_structured_lanes(read_operator[k], source, target, width, read_reset[k])
        width = leave if slot + 1 == stop_slot else (1 if read_reset[k + 1] else classes)
        if width == 1:
            for cell in range(cells):
                source[0, cell] = 0.0
            for c in range(classes):
                for cell in range(cells):
                    code = key[cell]
                    if code >= 0:
                        source[0, cell] += target[c, cell] * corners[code, c]
        else:
            for c in range(classes):
                for cell in range(cells):
                    code = key[cell]
                    if code >= 0:
                        source[c, cell] = target[c, cell] * corners[code, c]
        # Pairs with a fractional allele: their own factor, the same operations per lane.
        for i in range(count):
            cell = pending[i]
            read_ratio(factor, read_ratio_table, k, first_alt[cell // states], second_alt[cell % states],
                       gl[child, site])
            if width == 1:
                mass = 0.0
                for c in range(classes):
                    mass += target[c, cell] * factor[c]
                source[0, cell] = mass
            else:
                for c in range(classes):
                    source[c, cell] = target[c, cell] * factor[c]


@kernel(overloads=OVERLOADS["read_transfer.multiply_classes"], cache=True, nogil=True)
def multiply_classes(matrix, operator, factor, work, rows, reset):
    """Right multiply active rows by T diag(f). At an exact reset, each output row is its input mass times T[0] f."""
    classes = len(factor)
    if reset:
        for i in range(rows):
            mass = 0.0
            for m in range(classes):
                mass += matrix[i, m]
            for j in range(classes):
                matrix[i, j] = mass * float(operator[0, j]) * factor[j]
    else:
        for i in range(rows):
            for j in range(classes):
                work[i, j] = 0.0
        # Broadcast the input class over a contiguous operator row. Each
        # output retains the original input-class accumulation order.
        for m in range(classes):
            for i in range(rows):
                weight = matrix[i, m]
                for j in range(classes):
                    work[i, j] += weight * float(operator[m, j])
        for i in range(rows):
            for j in range(classes):
                matrix[i, j] = work[i, j] * factor[j]


@kernel(overloads=OVERLOADS["read_transfer.class_transfers"], cache=True, parallel=True, nogil=True)
def class_transfers(first_child, stop_child, state_alt, gl, site_index, site_start, site_stop, read_offset,
                    read_reset, read_width, read_operator, read_ratio_table):
    """Per child of [first_child, stop_child), bin and state pair (x, y) with the panel's
    alleles (state_alt) in both slots, the bin's class transfer prod_k T_k diag(f_k(x, y)) over the child's observed
    markers (f the read factor; the identity without any, or for a child without read state). States whose alleles
    agree at every observed marker of the bin (the same allele word) have the same transfers: each pair of words is
    computed once and copied to its other state pairs. Only the leading read_width[child, block] by
    read_width[child, block + 1] entries are used. A unit exit width sums class mass; a unit entry width starts with
    the equal-row operator of the next observed marker. The own cache is [n, B, S, S, C, C], and the structured
    cache is [n, B, C, C, S, S]; empty bins take the identity in the forward directly."""
    n = stop_child - first_child
    bins = site_start.shape[1]
    states = state_alt.shape[0]
    classes = read_ratio_table.shape[2]
    structured = read_operator.shape[1] != read_operator.shape[2]
    outer = classes if structured else states
    inner = states if structured else classes
    out = np.zeros((n, bins, outer, outer, inner, inner))
    for task in prange(n * bins):
        local = task // bins
        block = task - local * bins
        child = first_child + local
        # Materialize the structured bank class-major, so each update over its founder-pair plane is contiguous.
        bank = np.zeros((states, states, classes, classes)) if structured else out[local, block]
        for x in range(states):
            for y in range(states):
                for c in range(classes):
                    bank[x, y, c, c] = 1.0
        start = read_offset[child]
        if read_offset[child + 1] == start:
            continue
        base = start - site_start[child, 0]
        first_slot, stop_slot = site_start[child, block], site_stop[child, block]
        # word[x]: the first state with x's alleles at every observed marker of the bin (x itself if none before)
        word = np.empty(states, dtype=np.int64)
        for x in range(states):
            word[x] = x
            for z in range(x):
                if word[z] != z:
                    continue
                same = True
                for slot in range(first_slot, stop_slot):
                    if state_alt[x, site_index[slot]] != state_alt[z, site_index[slot]]:
                        same = False
                        break
                if same:
                    word[x] = z
                    break
        factor = np.empty(classes)
        work = np.empty((classes, classes))
        moved = np.empty(classes)
        rows = read_width[child, block]
        for slot in range(first_slot, stop_slot):
            k = base + slot
            site = site_index[slot]
            for x in range(states):
                if word[x] != x:
                    continue
                a = state_alt[x, site]
                for y in range(states):
                    if word[y] != y:
                        continue
                    b = state_alt[y, site]
                    read_ratio(factor, read_ratio_table, k, a, b, gl[child, site])
                    matrix = bank[x, y]
                    if structured:
                        width = rows if slot == first_slot else classes
                        for i in range(rows):
                            move_gap(read_operator[k], matrix[i], moved, width, read_reset[k])
                            for j in range(classes):
                                matrix[i, j] = moved[j] * factor[j]
                    else:
                        multiply_classes(matrix, read_operator[k], factor, work, rows, read_reset[k])
        if read_width[child, block + 1] == 1:
            # The next observed marker resets (or there is none). Cache the scalar exit, once per fixed pair.
            for x in range(states):
                for y in range(states):
                    if word[x] == x and word[y] == y:
                        matrix = bank[x, y]
                        for i in range(read_width[child, block]):
                            matrix[i, 0] = np.sum(matrix[i])
        for x in range(states):
            for y in range(states):
                if word[x] != x or word[y] != y:
                    bank[x, y] = bank[word[x], word[y]]
        if structured and first_slot != stop_slot:
            finite = True
            for i in range(rows):
                for j in range(read_width[child, block + 1]):
                    for x in range(states):
                        for y in range(states):
                            value = bank[x, y, i, j]
                            out[local, block, i, j, x, y] = value
                            finite &= np.isfinite(value)
            if not finite:
                # A basis vector can overflow even when the live input does not. Reapply that bin directly.
                out[local, block, 0, 0, 0, 0] = np.nan
    return out
