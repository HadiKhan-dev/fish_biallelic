"""Map-scale frame mixtures of painting's structured read operator."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import prange, types
from numba.typed import List

from ...core.concurrency import live_threads, threaded_each
from ...core.kernels import kernel
from ..painting.api import TAPE, apply_bin
from .read_gaps import append_site, apply_gap, compose_gap, identity_gap

_F1 = types.float64[::1]
_F2 = types.float64[:, ::1]
_F3 = types.float64[:, :, ::1]
_F4 = types.float64[:, :, :, ::1]
_M5 = types.float32[:, :, :, :, ::1]
_D5 = types.float64[:, :, :, :, ::1]
_I1 = types.int64[::1]
_I2 = types.int64[:, ::1]
_I = types.int64
_F = types.float64
_B = types.boolean
_B1 = types.boolean[::1]
_SOURCE = types.Tuple((_M5, TAPE, _F2, _I1, _D5))
OPERATOR_INPUTS = types.Tuple((types.ListType(_SOURCE), _I2, _I2, _F2, _F3, _F4))
FULL_SIB_COLUMNS = (2, 3, 7, 8, 25)
FRAME_COLUMNS = 26
_CLIP = 300.0


@dataclass(frozen=True)
class FrameSource:
    """Painted bin actions. The tape is passed opaquely through painting's interface."""

    matrix: np.ndarray
    tape: tuple[np.ndarray, ...]
    exponent: np.ndarray
    permutation: np.ndarray

    def bin_inputs(self) -> tuple:
        # Checkpoint decoding gives read-only views. The painting action's native signatures use writable views.
        writable = lambda value: np.require(value, requirements=["C", "W", "A"])
        return (writable(self.matrix), tuple(writable(field) for field in self.tape),
                writable(self.exponent), writable(self.permutation))

    def native(self) -> tuple:
        # A typed list can share both compiled and matrix-free sources across one frame batch.
        matrix, tape, exponent, perm = self.bin_inputs()
        if matrix.dtype == np.float64:
            compact, compiled = np.empty((0, 0, 0, 0, 0), dtype=np.float32), matrix
        else:
            compact, compiled = matrix, np.empty((0, 0, 0, 0, 0), dtype=np.float64)
        return compact, tape, exponent, perm, compiled


@dataclass(frozen=True)
class FrameOperators:
    """Durable frame actions with component-local class coordinates.

    Index columns are source, sample, bin, and compact frame row. A negative source means
    a compact operator with an index in `compact`. The other arrays are only read by the
    structured actions. One source is shared by all of its frame requests.
    """

    sources: tuple[FrameSource, ...]
    index: np.ndarray
    labels: np.ndarray
    log_weight: np.ndarray
    gap: np.ndarray
    compact: np.ndarray

    def __getitem__(self, rows):
        return FrameOperators(self.sources, self.index[rows], self.labels[rows], self.log_weight[rows],
                              self.gap[rows], self.compact)

    def native(self):
        sources = List.empty_list(_SOURCE)
        for source in self.sources:
            sources.append(source.native())
        writable = lambda value: np.require(value, requirements=["C", "W", "A"])
        return (sources, writable(self.index), writable(self.labels), writable(self.log_weight),
                writable(self.gap), writable(self.compact))


def full_sib(emissions):
    """Select the four full-sib symbols and misfit for a compact frame."""
    if isinstance(emissions, FrameOperators):
        return emissions
    return np.take(emissions, FULL_SIB_COLUMNS, axis=1)


def _compact_operators(emissions):
    n, symbols, width, _ = emissions.shape
    if symbols == len(FULL_SIB_COLUMNS):
        expanded = np.zeros((n, FRAME_COLUMNS, width, width))
        expanded[:, FULL_SIB_COLUMNS] = emissions
        emissions = expanded
    index = np.full((n, 4), -1, dtype=np.int64)
    index[:, 3] = np.arange(n)
    return FrameOperators((), index, np.zeros((n, 4), dtype=np.int64), np.empty((n, 0)),
                          np.empty((n, 0, 0)), emissions)


def _join_compact(parts):
    """Copy large dense joins to disjoint output ranges with the live CPU allowance."""
    # Small per-family joins fit in a single copy. Keep jobs large enough that
    # thread-map setup is small beside reading and writing the matrix blocks.
    block_bytes = 16 << 20
    total_bytes = sum(part.nbytes for part in parts)
    workers = min(live_threads(), len(parts), total_bytes // block_bytes)
    if workers < 2:
        return np.ascontiguousarray(np.concatenate(parts), dtype=np.float64)

    offsets = np.r_[0, np.cumsum([len(part) for part in parts])]
    cuts = np.searchsorted(offsets, np.arange(1, workers) * (offsets[-1] // workers))
    cuts = np.unique(np.r_[0, cuts, len(parts)])
    result = np.empty((offsets[-1], *parts[0].shape[1:]), dtype=np.float64)

    def copy(bounds):
        lo, hi = bounds
        np.concatenate(parts[lo:hi], out=result[offsets[lo]:offsets[hi]])

    threaded_each(copy, zip(cuts[:-1], cuts[1:], strict=True))
    return result


def join_emissions(parts):
    """Join frames while keeping distinct local class axes and shared source tapes."""
    if all(isinstance(part, np.ndarray) for part in parts) and len({part.shape[2] for part in parts}) == 1:
        return _join_compact(parts)
    operators = [part if isinstance(part, FrameOperators) else _compact_operators(part) for part in parts]
    sources = []
    source_indices = {}
    index, labels, weights, gaps = [], [], [], []
    compact = []
    compact_indices = {}
    compact_rows = 0
    pairs = max(part.log_weight.shape[1] for part in operators)
    gap_shape = max(part.gap.shape[1:] for part in operators)
    compact_width = max(part.compact.shape[2] for part in operators)
    for part in operators:
        rows = part.index.copy()
        active = rows[:, 0] >= 0
        if np.any(active):
            for i in np.unique(rows[active, 0]):
                source = part.sources[i]
                key = id(source)
                if key not in source_indices:
                    source_indices[key] = len(sources)
                    sources.append(source)
                rows[part.index[:, 0] == i, 0] = source_indices[key]
        if np.any(~active):
            key = id(part.compact)
            if key not in compact_indices:
                compact_indices[key] = compact_rows
                compact_rows += len(part.compact)
                side = part.compact.shape[2]
                expanded = np.zeros((len(part.compact), FRAME_COLUMNS, compact_width, compact_width))
                expanded[..., :side, :side] = part.compact
                compact.append(expanded)
            rows[~active, 3] += compact_indices[key]
        index.append(rows)
        labels.append(part.labels)
        weight = np.full((len(rows), pairs), -np.inf)
        weight[:, :part.log_weight.shape[1]] = part.log_weight
        weights.append(weight)
        gap = np.zeros((len(rows), *gap_shape))
        gap[:, :part.gap.shape[1], :part.gap.shape[2]] = part.gap
        gaps.append(gap)
    return FrameOperators(tuple(sources), np.concatenate(index), np.concatenate(labels), np.concatenate(weights),
                          np.concatenate(gaps), np.concatenate(compact) if compact else np.empty((0, 26, 1, 1)))


def native_inputs(emissions):
    if isinstance(emissions, FrameOperators):
        return np.empty((0, 0, 0, 0)), emissions.native()
    return emissions, None


@kernel(overloads=((_SOURCE, _I, _I, _I, _B, _F1, _F1, _F2, _B),),
        cache=True, inline="always", nogil=True)
def _apply_source_bin(source, sample, pair, block, swap, row, moved, work, backward):
    matrix, tape, exponent, perm, compiled = source
    if compiled.size:
        return apply_bin(compiled, tape, exponent, sample, pair, block, swap, perm, row, moved, work, backward)
    return apply_bin(matrix, tape, exponent, sample, pair, block, swap, perm, row, moved, work, backward)


@kernel(overloads=tuple((types.Tuple((matrix, TAPE, _F2, _I1)), _F3, _F3, _I1, _I1, _I2, _I2, _F2)
                        for matrix in (_M5, _D5)), cache=True, parallel=True, nogil=True)
def frame_weights(source, emission, correction, rows, bins, labels, pair_index, result):
    """Clip the scalar ratio and choose a common, model-independent frame gauge."""
    matrix, tape, exponent, perm = source
    states = pair_index.shape[0]
    for request in prange(len(rows)):
        sample, block = rows[request], bins[request]
        best = -np.inf
        for i in range(2):
            for j in range(2, 4):
                pair = pair_index[labels[request, i], labels[request, j]]
                best = max(best, emission[sample, pair, block])
        top = -np.inf
        ones = np.ones(len(perm))
        moved = np.empty(len(perm))
        work = np.empty((4, len(perm)))
        for first in range(states):
            for second in range(first, states):
                pair = pair_index[first, second]
                scale = min(emission[sample, pair, block] - best, _CLIP) + correction[sample, pair, block]
                result[request, pair] = scale
                gauge = apply_bin(matrix, tape, exponent, sample, pair, block, False, perm, ones, moved, work, True)
                if np.max(moved) > 0.0:
                    top = max(top, scale + gauge + math.log(np.max(moved)))
        if np.isfinite(top):
            result[request] -= top


@kernel(overloads=((_F4, _F3, _F2, _F2, _F2, _I1, _I1, _I1, _F4, _F3),),
        cache=True, parallel=True, nogil=True)
def bin_gaps(transition, law, survival, absent, initial, rows, starts, stops, out, entering):
    """Label-free gaps and prefix marginals on every bin, including empty bins."""
    for slot in prange(len(rows)):
        sample = rows[slot]
        product = np.empty_like(out[slot, 0])
        work = np.empty_like(product)
        entering[slot, 0] = initial[sample]
        for block in range(len(starts)):
            identity_gap(product)
            for site in range(starts[block], stops[block]):
                append_site(product, transition[site, sample], law[site, sample], survival[sample, site],
                            absent[sample, site], work)
                product, work = work, product
            out[slot, block] = product
            apply_gap(product, entering[slot, block], entering[slot, block + 1], False)


@kernel(overloads=((_F4, _F3, _I1, _I1, _B1, _F3, _F2),), cache=True, parallel=True, nogil=True)
def frame_gaps(operators, entering, slot, bins, first, out, entry):
    """The skipped-bin products and entry law, both in physical homologue order."""
    for frame in prange(len(bins)):
        sample, stop = slot[frame], bins[frame]
        start = stop if first[frame] else bins[frame - 1] + 1
        entry[frame] = entering[sample, start]
        product = np.empty_like(out[frame])
        work = np.empty_like(product)
        identity_gap(product)
        for block in range(start, stop):
            compose_gap(product, operators[sample, block], work)
            product, work = work, product
        out[frame] = product


@kernel(overloads=((_SOURCE, _I, _I, _I, _I, _I, _F, _F1, _F1, _F1, _F2, _B),),
        cache=True, inline="always", nogil=True)
def _add(source, sample, block, first, second, states, log_weight, row, moved, total, scratch, backward):
    low, high = min(first, second), max(first, second)
    pair = low * states - low * (low + 1) // 2 + high
    lg = _apply_source_bin(source, sample, pair, block, first > second, row, moved, scratch, backward)
    scale = math.exp(log_weight + lg)
    for c in range(len(moved)):
        total[c] += scale * moved[c]


# frame_action is inlined with constant (misfit, backward) pairs, which reach this kernel as literal types.
@kernel(overloads=((OPERATOR_INPUTS, _I, _I, _I, _F, _F1, _F1, _F2, _F2, _B),) + tuple(
    (OPERATOR_INPUTS, _I, _I, types.literal(misfit), _F, _F1, _F1, _F2, _F2, types.literal(backward))
    for misfit, backward in ((25, False), (25, True), (4, False))),
        cache=True, nogil=True)
def _structured_action(inputs, frame, column, misfit, delta, row, out, vectors, scratch, backward):
    sources, index, labels, weight, gap, compact = inputs
    sid, sample, block, dense = index[frame]
    if misfit == 4:
        column = FULL_SIB_COLUMNS[column]
    if sid < 0:
        width = compact.shape[2]
        for j in range(width):
            value = 0.0
            for i in range(width):
                if backward:
                    expected, error = compact[dense, column, j, i], compact[dense, 25, j, i]
                else:
                    expected, error = compact[dense, column, i, j], compact[dense, 25, i, j]
                value += row[i] * ((1.0 - delta) * expected + delta * error)
            out[j] = value
        out[width:] = 0.0
        return
    source = sources[sid]
    width = len(source[3])
    if backward:
        vectors[0, :width] = row[:width]
    else:
        apply_gap(gap[frame], row[:width], vectors[0, :width], False)
    own = gap.shape[1] - 1
    states = width - own
    first_symbol, second_symbol = column // 5, column % 5
    outside = states - 4
    divisor = 1
    if first_symbol == 4:
        divisor *= outside
    if second_symbol == 4:
        divisor *= outside
    siblings = (1.0 - delta) / divisor if divisor else 0.0
    error = delta / (states * (states + 1) // 2)
    vectors[2, :width] = 0.0
    for first in range(states):
        first_match = (labels[frame, first_symbol] == first if first_symbol < 4 else
                       not np.any(labels[frame] == first))
        for second in range(states):
            second_match = (labels[frame, second_symbol] == second if second_symbol < 4 else
                            not np.any(labels[frame] == second))
            mass = error if first == second else 0.5 * error
            if first_match and second_match:
                mass += siblings
            if mass == 0.0:
                continue
            low, high = min(first, second), max(first, second)
            pair = low * states - low * (low + 1) // 2 + high
            _add(source, sample, block, first, second, states, weight[frame, pair] + math.log(mass),
                 vectors[0, :width], vectors[1, :width], vectors[2, :width], scratch[:, :width], backward)
    if backward:
        apply_gap(gap[frame], vectors[2, :width], out[:width], True)
    else:
        out[:width] = vectors[2, :width]
    out[width:] = 0.0


@kernel(overloads=tuple((_F4, native, _I, _I, _I, _F, _F1, _F1, _F2, _F2, _B)
                        for native in (types.none, OPERATOR_INPUTS)), cache=True, inline="always", nogil=True)
def frame_action(emissions, operators, frame, column, misfit, delta, row, out, vectors, scratch, backward):
    """Apply a symbol mixture and its misfit term to one class vector."""
    if operators is not None:
        _structured_action(operators, frame, column, misfit, delta, row, out, vectors, scratch, backward)
    else:
        width = emissions.shape[2]
        for j in range(width):
            value = 0.0
            for i in range(width):
                if backward:
                    symbol = emissions[frame, column, j, i]
                    noise = emissions[frame, misfit, j, i]
                else:
                    symbol = emissions[frame, column, i, j]
                    noise = emissions[frame, misfit, i, j]
                value += row[i] * ((1.0 - delta) * symbol + delta * noise)
            out[j] = value
        out[width:] = 0.0


# painting.read_operator.ALLELE_FACTOR_ROWS - 1: the factor row of a repeated called named label (a unit test pins it)
_REPEATED_CALLED_ROW = 9


@kernel(overloads=((TAPE, _F1, _I, _I, _I, _B, _I1, _F2, _F2, _F1, _F1, _F2),), cache=True, nogil=True)
def _forward_rows(tape, exponents, s, p, b, swap, perm, rows, out, logs, weighted, scratch):
    """painting.read_operator.apply_bin on its tape route, forward, for every row of ``rows`` [R, W] at once: out[r]
    and logs[r] are apply_bin's output and returned log gauge for row r (``exponents``: the sample's row of the
    source's tempering exponents). Each row's arithmetic is apply_bin's own, in its order; only the pair's site
    factors and their powers are formed once per site for all rows. ``weighted`` [W] and ``scratch`` [2 R, W] are
    scratch."""
    starts, stops, first, second, codes, alt, transition, law, survival, absent, bypass, factors = tape
    own = law.shape[2]
    width = rows.shape[1]
    labels = width - own
    R = rows.shape[0]
    current = scratch[:R]
    moved = scratch[R:]
    for r in range(R):
        for i in range(width):
            current[r, i] = rows[r, perm[i] if swap else i]
    if bypass[s]:
        for r in range(R):
            out[r, :] = 0.0
            out[r, 0] = current[r, 0]
            logs[r] = 0.0
    else:
        x, y = first[p], second[p]
        repeated_named = x == y and x < len(alt) - 1
        alive = np.ones(R, dtype=np.bool_)
        for r in range(R):
            logs[r] = 0.0
        start, stop = starts[b], stops[b]
        for offset in range(stop - start):
            t = start + offset
            exponent = exponents[b] if len(exponents) == len(starts) else exponents[t]
            table = np.int64(p)
            if codes.size:
                first_code = codes[t, x]
                table = 3 * first_code + codes[t, y]
                if repeated_named and first_code == 2:
                    table = _REPEATED_CALLED_ROW
            factor = factors[s, t, table]
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
            phi = 0.0
            q = 0.0
            if labels:
                phi = survival[s, t]
                q = absent[s, t]
            last = offset + 1 == stop - start
            for r in range(R):
                if not alive[r]:
                    continue
                cur = current[r]
                mov = moved[r]
                for j in range(own):
                    value = 0.0
                    for i in range(own):
                        value += cur[i] * transition[t, s, i, j]
                    mov[j] = value
                if labels:
                    own_mass = 0.0
                    foreign_mass = 0.0
                    for j in range(own):
                        own_mass += cur[j]
                    for j in range(own, width):
                        foreign_mass += cur[j]
                    arrival = (1.0 - q) * (own_mass + (1.0 - phi) * foreign_mass) / labels
                    restart = (1.0 - phi) * q * foreign_mass
                    for j in range(own):
                        mov[j] = q * mov[j] + restart * law[t, s, j]
                    for j in range(own, width):
                        mov[j] = phi * cur[j] + arrival
                for j in range(width):
                    mov[j] *= weighted[j]
                top = 0.0
                for j in range(width):
                    top = max(top, mov[j])
                if top == 0.0:
                    out[r, :] = 0.0
                    alive[r] = False
                    continue
                if top < 1e-100 or top > 1e100 or last:
                    logs[r] += math.log(top)
                    for j in range(width):
                        mov[j] /= top
                for j in range(width):
                    cur[j] = mov[j]
        for r in range(R):
            if alive[r]:
                for j in range(width):
                    out[r, j] = current[r, j]
    if swap:
        for r in range(R):
            for i in range(width):
                moved[r, i] = out[r, i]
            for i in range(width):
                out[r, i] = moved[r, perm[i]]


@kernel(overloads=((OPERATOR_INPUTS, _F4),), cache=True, parallel=True, nogil=True)
def cache_frames(inputs, out):
    """The dense matrices out[frame, 26, C, C] of every frame's symbol columns and misfit: the structured
    action (_structured_action) applied to each basis row, its sibling and misfit masses separated so that the
    chains mix them per misfit weight (frame_action's dense route); C = out.shape[2] >= the frame's local width,
    zero beyond it. A pair's bin action runs on all the frame's basis rows at once (_forward_rows: apply_bin's
    arithmetic row by row); every entry sums its pairs in the same order as one row at a time."""
    sources, index, labels, weight, gap, compact = inputs
    width = out.shape[2]
    for frame in prange(len(index)):
        sid, sample, block, dense = index[frame]
        if sid < 0:
            side = compact.shape[2]
            out[frame, :, :side, :side] = compact[dense]
            continue
        source = sources[sid]
        matrix, tape, exponent, perm, compiled = source
        local = len(perm)
        states = local - (gap.shape[1] - 1)
        symbol = np.full(states, 4, dtype=np.int64)
        for h in range(4):
            symbol[labels[frame, h]] = h
        error = 2.0 / (states * (states + 1))
        basis = np.zeros(local)
        rows = np.empty((local, local))
        for c in range(local):
            basis[c] = 1.0
            apply_gap(gap[frame], basis, rows[c], False)
            basis[c] = 0.0
        moved = np.empty((local, local))
        logs = np.empty(local)
        work = np.empty((4, local))
        scratch = np.empty((2 * local, local))
        # apply_bin walks the tape only for a float32 (uncompiled) source with a non-empty tape
        walk = not compiled.size and tape[0].size > 0
        for first in range(states):
            for second in range(states):
                low, high = min(first, second), max(first, second)
                pair = low * states - low * (low + 1) // 2 + high
                if walk:
                    _forward_rows(tape, exponent[sample], sample, pair, block, first > second, perm, rows, moved, logs,
                                  work[0], scratch)
                else:
                    for c in range(local):
                        logs[c] = _apply_source_bin(source, sample, pair, block, first > second, rows[c], moved[c],
                                                    work, False)
                column = 5 * symbol[first] + symbol[second]
                norm = 1.0
                if symbol[first] == 4:
                    norm *= states - 4
                if symbol[second] == 4:
                    norm *= states - 4
                mixture = error if first == second else error / 2.0
                for c in range(local):
                    scale = math.exp(weight[frame, pair] + logs[c])
                    for j in range(local):
                        value = scale * moved[c, j]
                        out[frame, column, c, j] += value / norm
                        out[frame, 25, c, j] += mixture * value
