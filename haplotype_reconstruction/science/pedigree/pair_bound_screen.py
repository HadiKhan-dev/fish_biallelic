"""Rank capped retained parent pairs on prepared chromosome components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import get_num_threads

from ...core.errors import InvariantError
from . import blocks, read_bounds, read_m2
from .forward import ChildTerms
from .numerical_parameters import PAIR_NULL_RELATIVE_GUARD
from .pair_bound import emission_bound, pair_bounds, partner_bound, scaled_emissions, slot_groups
from .projection import Projection
from .workspace_parameters import BOUND_BATCH_ROWS as BATCH_ROWS
from .workspace_parameters import BOUND_WORKSPACE_BYTES as WORKSPACE_BYTES


@dataclass(frozen=True)
class PairBatch:
    """Retained pairs of consecutive children: trios int64 [T, 3] (child, first, second) in the canonical row order
    (children ascending, then the compact profile's ascending parents, pairs i < j); the batch's (child, retained
    candidate) entries as CSR (children, offsets, candidates); each row's position in ``children`` and the entries of
    its two parents."""

    trios: np.ndarray
    children: np.ndarray
    offsets: np.ndarray
    candidates: np.ndarray
    entry_p: np.ndarray
    entry_q: np.ndarray
    row_child: np.ndarray | None = None

    def __post_init__(self):
        if self.row_child is None:  # children ascending: each row's position in the child list
            object.__setattr__(self, "row_child", np.searchsorted(self.children, self.trios[:, 0]).astype(np.int64))


def retained_pair_count(compact) -> int:
    counts = np.diff(compact.indptr)
    return int(np.sum(counts * (counts - 1) // 2))


def retained_batches(compact, bins: int, states: int, classes: int = 1):
    """PairBatch generator over every retained pair, bounded by BATCH_ROWS rows and WORKSPACE_BYTES of tables."""
    # emission tables of one listed child (never an index span; log and scaled), with the read state its class
    # transfer bounds (read_bounds.class_bound_matrices)
    per_child = (2 + (classes * classes if classes > 1 else 0)) * bins * states * states * 8
    kids: list[int]
    trios, kids, cands, ep, eq, local = [], [], [], [], [], []
    offsets = [0]
    rows = 0
    for child in range(compact.n_samples):
        a, b = compact.indptr[child : child + 2]
        parents = np.asarray(compact.parents[a:b], dtype=np.int64)
        r = len(parents)
        if r < 2:
            continue
        i, j = np.triu_indices(r, 1)
        base = offsets[-1]
        trios.append(np.column_stack([np.full(len(i), child, np.int64), parents[i], parents[j]]))
        ep.append(base + i)
        eq.append(base + j)
        local.append(np.full(len(i), len(kids), np.int64))
        kids.append(child)
        cands.append(parents)
        offsets.append(base + r)
        rows += len(i)
        # Partner log bounds plus their probability factors.
        workspace = len(kids) * per_child + offsets[-1] * bins * 16
        if rows >= BATCH_ROWS or workspace >= WORKSPACE_BYTES:
            yield PairBatch(
                np.concatenate(trios),
                np.asarray(kids, np.int64),
                np.asarray(offsets, np.int64),
                np.concatenate(cands),
                np.concatenate(ep),
                np.concatenate(eq),
                np.concatenate(local),
            )
            trios, kids, cands, ep, eq, local, offsets, rows = [], [], [], [], [], [], [0], 0
    if trios:
        yield PairBatch(
            np.concatenate(trios),
            np.asarray(kids, np.int64),
            np.asarray(offsets, np.int64),
            np.concatenate(cands),
            np.concatenate(ep),
            np.concatenate(eq),
            np.concatenate(local),
        )


@dataclass(frozen=True)
class ComponentContext:
    """Per-component inputs of the pair bound shared by every batch: the slot groups and transfers of every
    candidate, and BACKGROUND's ALT range over every allele a slot can transmit as BACKGROUND (with the candidates'
    BACKGROUND ALT [N, M])."""

    projection: Projection
    terms: ChildTerms
    group: np.ndarray
    count: np.ndarray
    initial: np.ndarray
    transfer: np.ndarray
    background: np.ndarray
    low: np.ndarray
    high: np.ndarray


def component_context(projection, terms, high_mass: float, claim_threads=lambda: None) -> ComponentContext:
    bins = projection.n_bins
    null_row = terms.null_diagonal + (projection.n_states - 1) * terms.null_off
    null_row_max = (np.max(null_row, axis=0) if null_row.size else np.zeros(max(bins - 1, 0))) * (
        1.0 + PAIR_NULL_RELATIVE_GUARD
    )
    claim_threads()
    group, count, initial, transfer = slot_groups(
        projection.pi,
        projection.stay,
        projection.alpha,
        projection.beta,
        projection.row_arm,
        projection.column_arm,
        projection.pivot,
        projection.branch,
        projection.excursion,
        projection.available,
        float(high_mass),
        np.ascontiguousarray(null_row_max),
    )
    background = np.ascontiguousarray(projection.candidate_alt[:, :, -1])
    # The range holds every BACKGROUND ALT probability a slot can transmit: each available candidate's (conditioned on
    # its reads) and, where an evidence-free candidate makes a pair fall back to M1/M0 (forward.component_m2), the null
    # chain's unconditional frequency state_alt[-1]. Without evidence-free candidates the narrower box is kept.
    null_background = np.asarray(projection.state_alt[-1], dtype=np.float64)
    if np.any(projection.available):
        low = np.min(background[projection.available], axis=0)
        high = np.max(background[projection.available], axis=0)
        if not np.all(projection.available):
            low = np.minimum(low, null_background)
            high = np.maximum(high, null_background)
    else:
        low = high = null_background
    return ComponentContext(
        projection,
        terms,
        group,
        count,
        initial,
        transfer,
        background,
        np.ascontiguousarray(low),
        np.ascontiguousarray(high),
    )


# Rows per piece of a (child, first parent) run in the class-carrying pair bound (scheduling only: a piece repeats
# the run's per-bin partial maxima, about one row's work per piece, and no value depends on the cut).
# At 19 threads famhs chr19 took 14.02 s / 232.1 CPU s (.87) with eight rows. The per-run forward
# depends on its preceding bin; the class kernel takes 12.46 s / 210.0 CPU s (.89) on the same workload.
PAIR_BOUND_RUN_ROWS = 8


def component_bounds(context: ComponentContext, batch: PairBatch, claim_threads=lambda: None) -> np.ndarray:
    """float64 [T]: the bound of each batch row on one component."""
    projection, terms = context.projection, context.terms
    claim_threads()
    log_emission = emission_bound(
        projection.state_alt,
        context.low,
        context.high,
        terms.gl,
        terms.coefficient,
        terms.site_index,
        terms.site_start,
        terms.site_stop,
        terms.mismatch,
        batch.children,
    )
    claim_threads()
    partner = partner_bound(
        projection.state_alt,
        context.low,
        context.high,
        context.background,
        terms.gl,
        terms.coefficient,
        terms.site_index,
        terms.site_start,
        terms.site_stop,
        terms.mismatch,
        batch.children,
        batch.offsets,
        batch.candidates,
    )
    claim_threads()
    scaled, offset, top = scaled_emissions(log_emission)
    claim_threads()
    if terms.n_classes > 1:
        # the read class carried (read_bounds.pair_bounds_classes): named states exact, BACKGROUND over the
        # component's range in both slots; no partner caps
        relaxed = np.zeros(projection.state_alt.shape, dtype=bool)
        relaxed[-1] = True
        lo, hi = read_bounds.slot_intervals(projection.state_alt, relaxed, context.low, context.high)
        matrices = read_bounds.class_bound_matrices(
            np.ascontiguousarray(batch.children, dtype=np.int64), 0, projection.n_bins, lo, hi, lo, hi, terms.gl,
            terms.coefficient, terms.site_index, terms.site_start, terms.site_stop, terms.read_offset,
            terms.read_reset, terms.read_operator, terms.read_ratio,
        )
        claim_threads()
        # runs of rows sharing (child, first parent): the canonical order keeps each run contiguous
        trios = batch.trios
        change = np.flatnonzero((trios[1:, 0] != trios[:-1, 0]) | (trios[1:, 1] != trios[:-1, 1])) + 1
        run_start = np.concatenate(([0], change, [len(trios)])).astype(np.int64)
        # long runs cut into pieces of at most PAIR_BOUND_RUN_ROWS rows (each piece forms its first parent's
        # partial maxima again: the same values). The kernel balances the pieces across its team.
        run_start = np.unique(np.concatenate([np.arange(a, b, PAIR_BOUND_RUN_ROWS)
                                              for a, b in zip(run_start[:-1], run_start[1:], strict=True)]
                                             + [run_start[-1:]])).astype(np.int64)
        return read_bounds.pair_bounds_classes(
            trios, batch.row_child, log_emission, scaled, top, offset, context.group, context.count,
            context.initial, context.transfer, matrices, terms.read_initial, terms.read_width, run_start,
            get_num_threads(),
        )
    return pair_bounds(
        batch.trios,
        batch.row_child,
        batch.entry_p,
        batch.entry_q,
        partner,
        log_emission,
        scaled,
        top,
        offset,
        context.group,
        context.count,
        context.initial,
        context.transfer,
        projection.available,
    )


def retained_pair_offsets(compact) -> np.ndarray:
    """The complete canonical pair axis, including children needing no ranking."""
    counts = np.diff(compact.indptr).astype(np.int64)
    return np.concatenate(([0], np.cumsum(counts * (counts - 1) // 2)))


def ranking_child_ranges(compact, cap: int, parts: int) -> tuple[tuple[int, int], ...]:
    """Contiguous ranges of children with approximately equal retained-pair work."""
    counts = np.diff(retained_pair_offsets(compact))
    work = np.concatenate(([0], np.cumsum(np.where(counts > cap, counts, 0))))
    # Keep a child together even when its own pairs exceed the mean work per part.
    cuts = np.searchsorted(work, np.linspace(0, work[-1], max(1, parts) + 1)[1:-1])
    edges = np.unique(np.concatenate(([0], cuts, [compact.n_samples])))
    return tuple((int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True))


def _ranking_rows(compact, cap, eligibility, first_child, stop_child):
    """Eligible rows of capped children, indexed relative to their range's start."""
    offset = 0
    for child in range(first_child, stop_child):
        a, b = compact.indptr[child : child + 2]
        parents = compact.parents[a:b]
        count = len(parents) * (len(parents) - 1) // 2
        if count > cap:
            i, j = np.triu_indices(len(parents), 1)
            first, second = parents[i], parents[j]
            allowed = (first != child) & (second != child)
            if eligibility is not None:
                allowed &= eligibility.pair_mask(child, first, second)
            ids = np.flatnonzero(allowed)
            if len(ids) > cap:
                for start in range(0, len(ids), BATCH_ROWS):
                    chunk = ids[start:start + BATCH_ROWS]
                    trios = np.column_stack((np.full(len(chunk), child, np.int64), first[chunk], second[chunk]))
                    yield offset + chunk, trios
        offset += count


@dataclass(frozen=True)
class RankedPieces:
    """Scores indexed by canonical ordered trios and the contig's material cuts."""

    trios: np.ndarray
    cuts: np.ndarray
    values: np.ndarray

    def for_panel(self, trios: np.ndarray, cuts: np.ndarray) -> np.ndarray | None:
        if not len(self.trios) or not np.array_equal(self.cuts, cuts):
            return None
        key = np.dtype([(label, np.int64) for label in ("child", "first", "second")])
        saved = np.ascontiguousarray(self.trios).view(key).reshape(-1)
        want = np.ascontiguousarray(trios).view(key).reshape(-1)
        positions = np.minimum(np.searchsorted(saved, want), len(saved) - 1)
        matched = saved[positions] == want
        result = np.full((len(trios), len(cuts)), np.nan)
        result[matched] = self.values[positions[matched]]
        return result


def _round_up(values: np.ndarray) -> np.ndarray:
    rounded = values.astype(np.float32)
    below = rounded.astype(np.float64) < values
    rounded[below] = np.nextafter(rounded[below], np.float32(np.inf))
    return rounded


def chromosome_ranking(
    prepared, compact, config, claim_threads=lambda: None, *, reverse: bool | None = None, eligibility=None,
    child_range: tuple[int, int] | None = None, placement: blocks.ScorePlacement | None = None,
) -> tuple[np.ndarray, RankedPieces | None]:
    """Store the full-data M2 ranking key of each relevant row, rounded up once per chromosome.

    An optional child range scores a contiguous slice for an independent task.
    Joining adjacent ranges in child order gives the full canonical array.
    """
    from . import read_score, resident

    first_child, stop_child = (0, compact.n_samples) if child_range is None else child_range
    if not 0 <= first_child <= stop_child <= compact.n_samples:
        raise ValueError("M2 ranking child range is outside the compact sample axis")
    offsets = retained_pair_offsets(compact)
    out = np.zeros(offsets[stop_child] - offsets[first_child], dtype=np.float32)
    batches = tuple(_ranking_rows(compact, int(config.panel.max_pairs_per_child), eligibility, first_child, stop_child))
    if not batches:
        return out, None
    indices, trios = (np.concatenate(parts) for parts in zip(*batches, strict=True))
    del batches
    totals = np.zeros(len(trios))
    rounding_work = np.zeros(compact.n_samples)
    parts = []
    if placement is not None:
        offset = placement.offset
        selectors = tuple(c.selector_switch_probability for c in prepared.components)
        if not np.array_equal(blocks.cut_list(selectors, offset, placement.last, config.material), placement.cuts):
            raise InvariantError(f"{prepared.contig}: prepared pieces differ from the genome cut list")
    for component in prepared.components:
        state = resident.resident(component, config, claim_threads, reverse=reverse, include_pair_masses=False)
        axis, n_pieces = np.empty(0, dtype=np.int64), 0
        if placement is not None:
            switch = component.selector_switch_probability
            axis, description = blocks.pieces(switch, offset, placement.last, config.material,
                                              component.component_index)
            offset += blocks.component_length(switch, config.material)
            n_pieces = len(description)
        full, pieces = read_m2.component_scores(state.projection, state.terms, trios, axis, n_pieces, claim_threads)
        totals += full
        terms = state.terms
        # A bin normalization sums S*S*D cells. Include per-site class movement and emission in
        # the numerical guard, and rescore near float32 boundaries using the original evaluation.
        rounding_work += (state.projection.n_bins * state.projection.n_states**2 * terms.n_classes
                          + (terms.site_stop[:, -1] - terms.site_start[:, 0]) * (terms.n_classes + 4))
        if placement is not None:
            parts.append(pieces)
    finite = np.flatnonzero(np.isfinite(totals))
    value = totals[finite]
    upper = _round_up(value)
    lower = np.nextafter(upper, np.float32(-np.inf))
    distance = np.minimum(upper.astype(np.float64) - value, value - lower.astype(np.float64))
    guard = 32.0 * np.finfo(np.float64).eps * (rounding_work[trios[finite, 0]] + np.abs(value))
    uncertain = finite[distance <= guard]
    if len(uncertain):
        totals[uncertain] = 0.0
        for component in prepared.components:
            state = resident.resident(component, config, claim_threads, reverse=reverse, include_pair_masses=False)
            totals[uncertain] += read_score.component_m2(
                state.projection, state.terms, trios[uncertain], claim_threads=claim_threads
            )
        for part in parts:
            part[uncertain] = np.nan
    out[indices] = _round_up(totals)
    ranked = (RankedPieces(trios, placement.cuts, np.concatenate(parts, axis=1))
              if placement is not None and parts else None)
    return out, ranked


def chromosome_bounds(
    prepared, compact, config, claim_threads=lambda: None, *, reverse: bool | None = None, eligibility=None,
    child_range: tuple[int, int] | None = None,
) -> np.ndarray:
    """The rounded chromosome keys alone for callers without a material placement."""
    keys, _ = chromosome_ranking(prepared, compact, config, claim_threads, reverse=reverse,
                                 eligibility=eligibility, child_range=child_range)
    return keys
