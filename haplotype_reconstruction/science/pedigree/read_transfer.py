"""Propagate the child's read classes within a painting bin and build reusable class transfers."""

from __future__ import annotations

import numpy as np
from numba import prange

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


@kernel(overloads=OVERLOADS["read_transfer.tempered_factor"], cache=True, inline="always", nogil=True)
def tempered_factor(out, ratio, powered, slot, a, b, gl, eta):
    """out[C]: the tempered view's per-site read factor at information exponent ``eta``, R_c(a, b) ** eta: the
    painting's tempered read factor (read_factor: R ** w per site), the one place the tempered view's read factor is
    formed. The corners come from ``powered`` (child_reads.resident_terms); R itself at eta = 1, one at eta = 0. The
    scalar part's G ** eta is applied to the bin product by the caller."""
    classes = out.shape[0]
    if eta == 0.0:
        for c in range(classes):
            out[c] = 1.0
        return
    if eta != 1.0 and (a == 0.0 or a == 1.0) and (b == 0.0 or b == 1.0):
        config = 2 * int(a) + int(b)
        for c in range(classes):
            out[c] = float(powered[slot, config, c])
        return
    read_ratio(out, ratio, slot, a, b, gl)
    if eta != 1.0:
        for c in range(classes):
            out[c] = out[c] ** eta


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
    read_powered,
    tempered,
    eta,
    class_cache,
    local,
    moved,
):
    """Use one native carry body for the signed indices and runtime view flag."""
    _carry_bin(
        current, mix, factor, column, np.int64(child), block, np.int64(first_source), np.int64(second_source),
        candidate_alt, state_alt, gl, site_index, site_start, site_stop, read_offset, read_reset, read_width,
        read_operator, read_ratio_table, read_powered, np.bool_(tempered), eta, class_cache, local, moved,
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
    read_powered,
    tempered,
    eta,
    class_cache,
    local,
    moved,
):
    """In place on one view's forward state ``current`` [C, S, S]: the class part of the child's observed markers of
    bin ``block`` (each marker's class operator from the child's previous observed marker, applied to every state pair,
    then each pair's read factor: R_c(a, b) at full strength, tempered_factor at ``eta`` when ``tempered``). Slot 1's
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
    if class_cache is not None:
        view = 0 if tempered else 1
        fixed_first = states if first_source < 0 else last
        fixed_second = states if second_source < 0 else last
        for first in range(fixed_first):
            for second in range(fixed_second):
                transfer = class_cache[local, block, view, first, second]
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
    elif enter == classes and leave == classes and not np.any(read_reset[base + first_slot:base + stop_slot]):
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
                    if tempered:
                        tempered_factor(factor, read_ratio_table, read_powered, k, a, b, gl[child, site], eta)
                    else:
                        read_ratio(factor, read_ratio_table, k, a, b, gl[child, site])
                    for c in range(classes):
                        current[c, first, second] = column[c] * factor[c]
        return

    first_start = last if class_cache is not None and second_source < 0 else 0
    if class_cache is None or enter != 1 or leave != 1:
        # Keep a coupled bin's live class vector in the workspace while its
        # markers advance. Only uncached founder pairs need the recurrence.
        for first in range(first_start, states):
            candidate_row = first == last and first_source >= 0
            second_start = 0 if class_cache is None or candidate_row else last
            for second in range(second_start, states):
                for c in range(enter):
                    column[c] = current[c, first, second]
                width = enter
                for slot in range(first_slot, stop_slot):
                    k = base + slot
                    site = site_index[slot]
                    a = state_alt[first, site] if first_source < 0 else candidate_alt[first_source, site, first]
                    b = state_alt[second, site] if second_source < 0 else candidate_alt[second_source, site, second]
                    if tempered:
                        tempered_factor(factor, read_ratio_table, read_powered, k, a, b, gl[child, site], eta)
                    else:
                        read_ratio(factor, read_ratio_table, k, a, b, gl[child, site])
                    if read_reset[k]:
                        # All rows of T are the same law. The pair's previous class mass is sufficient.
                        mass = column[0]
                        for i in range(1, width):
                            mass += column[i]
                        for j in range(classes):
                            moved[j] = mass * float(read_operator[k, 0, j]) * factor[j]
                    else:
                        for j in range(classes):
                            moved[j] = 0.0
                        for i in range(width):
                            weight = column[i]
                            for j in range(classes):
                                moved[j] += weight * float(read_operator[k, i, j])
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
                    if tempered:
                        tempered_factor(ratio, read_ratio_table, read_powered, k, a, b, gl[child, site], eta)
                    else:
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
def class_transfers(first_child, stop_child, state_alt, gl, site_index, site_start, site_stop, exponent, read_offset,
                    read_reset, read_width, read_operator, read_ratio_table, read_powered):
    """float64 [n, B, 2, S, S, C, C]: per child of [first_child, stop_child), bin, view (0 tempered, 1 full) and state
    pair (x, y) with the panel's alleles (state_alt) in both slots, the bin's class transfer prod_k T_k diag(f_k(x, y))
    over the child's observed markers (f the view's read factor; the identity without any, or for a child without
    read state). States whose alleles agree at every observed marker of the bin (the same allele word) have the same
    transfers: each pair of words is computed once and copied to its other state pairs. Only the leading
    read_width[child, block] by read_width[child, block + 1] entries are used. A unit exit width sums
    class mass; a unit entry width starts with the equal-row operator of the next observed marker."""
    n = stop_child - first_child
    bins = site_start.shape[1]
    states = state_alt.shape[0]
    classes = read_operator.shape[1]
    out = np.zeros((n, bins, 2, states, states, classes, classes))
    for task in prange(n * bins):
        local = task // bins
        block = task - local * bins
        child = first_child + local
        for view in range(2):
            for x in range(states):
                for y in range(states):
                    for c in range(classes):
                        out[local, block, view, x, y, c, c] = 1.0
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
        eta = exponent[child, block]
        for slot in range(first_slot, stop_slot):
            k = base + slot
            site = site_index[slot]
            for view in range(2):
                for x in range(states):
                    if word[x] != x:
                        continue
                    a = state_alt[x, site]
                    for y in range(states):
                        if word[y] != y:
                            continue
                        b = state_alt[y, site]
                        if view == 0:
                            tempered_factor(factor, read_ratio_table, read_powered, k, a, b, gl[child, site], eta)
                        else:
                            read_ratio(factor, read_ratio_table, k, a, b, gl[child, site])
                        matrix = out[local, block, view, x, y]
                        multiply_classes(matrix, read_operator[k], factor, work, read_width[child, block],
                                         read_reset[k])
        if read_width[child, block + 1] == 1:
            # The next observed marker resets (or there is none). Cache the scalar exit, once per fixed pair.
            for view in range(2):
                for x in range(states):
                    for y in range(states):
                        if word[x] == x and word[y] == y:
                            matrix = out[local, block, view, x, y]
                            for i in range(read_width[child, block]):
                                matrix[i, 0] = np.sum(matrix[i])
        for view in range(2):
            for x in range(states):
                for y in range(states):
                    if word[x] != x or word[y] != y:
                        out[local, block, view, x, y] = out[local, block, view, word[x], word[y]]
    return out
