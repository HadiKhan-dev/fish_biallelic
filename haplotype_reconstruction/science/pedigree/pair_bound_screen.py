"""Batch retained parent pairs and run their bounds on prepared chromosome components."""

from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np

from . import read_bounds
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
    # emission tables of one listed child (never an index span), with the read state its class transfer bounds of
    # both views (read_bounds.class_bound_matrices)
    per_child = (3 + (2 * classes * classes if classes > 1 else 0)) * bins * states * states * 8
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
        # Partner log bounds plus the full/tempered probability factors.
        workspace = len(kids) * per_child + offsets[-1] * bins * 24
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
    full_e, tempered_e, offset_full, offset_tempered, top = scaled_emissions(
        log_emission, terms.exponent, batch.children
    )
    claim_threads()
    if terms.n_classes > 1:
        # the read class carried (read_bounds.pair_bounds_classes): named states exact, BACKGROUND over the
        # component's range in both slots; no partner caps
        relaxed = np.zeros(projection.state_alt.shape, dtype=bool)
        relaxed[-1] = True
        lo, hi = read_bounds.slot_intervals(projection.state_alt, relaxed, context.low, context.high)
        matrices = read_bounds.class_bound_matrices(
            np.ascontiguousarray(batch.children, dtype=np.int64), 0, projection.n_bins, lo, hi, lo, hi, terms.gl,
            terms.coefficient, terms.site_index, terms.site_start, terms.site_stop, terms.exponent,
            terms.read_offset, terms.read_reset, terms.read_operator, terms.read_ratio, terms.read_powered,
        )
        claim_threads()
        # runs of rows sharing (child, first parent): the canonical order keeps each run contiguous
        trios = batch.trios
        change = np.flatnonzero((trios[1:, 0] != trios[:-1, 0]) | (trios[1:, 1] != trios[:-1, 1])) + 1
        run_start = np.concatenate(([0], change, [len(trios)])).astype(np.int64)
        # long runs cut into pieces of at most PAIR_BOUND_RUN_ROWS rows (each piece forms its first parent's
        # partial maxima again: the same values) so that no run outlasts the others; then handed out one at a time
        run_start = np.unique(np.concatenate([np.arange(a, b, PAIR_BOUND_RUN_ROWS)
                                              for a, b in zip(run_start[:-1], run_start[1:], strict=True)]
                                             + [run_start[-1:]])).astype(np.int64)
        with numba.parallel_chunksize(1):
            return read_bounds.pair_bounds_classes(
                trios, batch.row_child, log_emission, full_e, tempered_e, top, offset_full, offset_tempered,
                terms.exponent, context.group, context.count, context.initial, context.transfer, matrices,
                terms.read_initial, terms.read_width, run_start,
            )
    return pair_bounds(
        batch.trios,
        batch.row_child,
        batch.entry_p,
        batch.entry_q,
        partner,
        log_emission,
        full_e,
        tempered_e,
        top,
        offset_full,
        offset_tempered,
        terms.exponent,
        context.group,
        context.count,
        context.initial,
        context.transfer,
        projection.available,
    )


def chromosome_bounds(
    prepared, compact, config, claim_threads=lambda: None, *, reverse: bool | None = None
) -> np.ndarray:
    """float32 [P]: per retained pair (canonical row order) the chromosome's M2 bound, the sum over its components,
    rounded up to float32 (never below the float64 bound)."""
    from . import resident

    high_mass = config.panel.pair_bound_high_mass
    contexts = []
    for component in prepared.components:
        state = resident.resident(component, config, claim_threads, reverse=reverse)
        contexts.append(component_context(state.projection, state.terms, high_mass, claim_threads))
        del state
    out = np.zeros(retained_pair_count(compact), dtype=np.float32)
    bins = max((component.n_bins for component in prepared.components), default=1)
    n_states = max((component.n_states for component in prepared.components), default=1)
    classes = max((context.terms.n_classes for context in contexts), default=1)
    cursor = 0
    for batch in retained_batches(compact, bins, n_states, classes):
        value = np.zeros(len(batch.trios))
        for context in contexts:
            value += component_bounds(context, batch, claim_threads)
        rounded = value.astype(np.float32)
        below = rounded.astype(np.float64) < value
        rounded[below] = np.nextafter(rounded[below], np.float32(np.inf))
        out[cursor : cursor + len(value)] = rounded
        cursor += len(value)
    return out
