"""One weighted fit: exposure gate, family-adjusted scores, marginal state, identity, acyclic graph.

A fit gives every informative chromosome a multiplicity (all ones for the full data, bootstrap
counts or a left-out chromosome in resampling). The genome-wide likelihood of a configuration is
the weighted chromosome sum, mixed once with the child's M0 likelihood. A non-M0 configuration is
testable only when its edges (and pair) are exposed in enough bins, enough of the painted genome
and enough distinct chromosomes; an untestable configuration contributes the child's M0 likelihood
to its state's mean but cannot be named. The marginal state is chosen first (log mean identity
evidence plus a fixed prior), then the identity within it; the graph step then enforces a DAG.

Method: docs/methods/pedigree.md#marginal-state-then-identity.
"""

from __future__ import annotations

import numpy as np

from ...contracts.pedigree import (
    M0,
    M1,
    M2,
    UNRESOLVED,
    DecisionInputs,
    ExposureEvidence,
    GenomeScores,
    LocalGraphChoices,
    Selection,
    edge_index,
)
from ...core.errors import InputError
from ...core.kernels import kernel
from ...core.numerics import ordered_sum
from . import alternatives as alternatives_module
from . import family, graph
from .alternatives import Alternatives, softmax_finite
from .kernels.signatures import OVERLOADS
from .orientation import Orientation
from .orientation import fit as fit_orientation
from .parameters import ExposureParameters, PedigreeParameters
from .ties import magnitude, unique_finite_winner

__all__ = ["map_units", "testable", "fit_scores", "select", "fit", "build_exposure"]


# ------------------------------------------------------------------------------- inputs
@kernel(overloads=OVERLOADS["selection._pack_presence"], cache=True, nogil=True)
def _pack_presence(values):
    """uint64 [W, items]: bit c of word c // 64 is set where values[c, item] is true."""
    n_contigs, n_items = values.shape
    packed = np.zeros(((n_contigs + 63) // 64, n_items), dtype=np.uint64)
    one = np.uint64(1)
    for contig in range(n_contigs):
        bit = one << np.uint64(contig % 64)
        for item in range(n_items):
            if values[contig, item]:
                packed[contig // 64, item] |= bit
    return packed


@kernel(overloads=OVERLOADS["selection._units_present"], cache=True, nogil=True)
def _units_present(packed, weights, units):
    """float64 [items]: map units (``units`` [C, items]) of the chromosomes with positive weight whose presence bit is
    set."""
    n_words, n_items = packed.shape
    totals = np.zeros(n_items, dtype=np.float64)
    one = np.uint64(1)
    for contig in range(weights.shape[0]):
        if weights[contig] > 0:
            bit = one << np.uint64(contig % 64)
            word = contig // 64
            for item in range(n_items):
                if packed[word, item] & bit:
                    totals[item] += units[contig, item]
    return totals


def map_units(morgans, config: ExposureParameters) -> np.ndarray:
    """float64: map units of measured map lengths ``morgans`` (Morgans; any shape), max(1, length / unit_morgans) (1
    where the length is unknown): an unlinked chromosome counts once however short, a long one by its length."""
    morgans = np.asarray(morgans, dtype=np.float64)
    units = np.where(np.isfinite(morgans), morgans / config.unit_morgans, 1.0)
    return np.ascontiguousarray(np.maximum(1.0, units))


# ------------------------------------------------------------------------------- gate and scores
def testable(weights: np.ndarray, inputs: DecisionInputs, config: ExposureParameters) -> np.ndarray:
    """bool [A]: configurations testable on the weighted chromosomes (M0 always): every edge (and an M2 pair) exposed
    in enough bins, enough of the fit's painted bins and on chromosomes holding enough map units."""
    exposure = inputs.exposure
    alternatives = inputs.alternatives
    edge_exposed = ordered_sum(exposure.edge_exposed, axis=0, weights=weights)
    pair_exposed = ordered_sum(exposure.pair_exposed, axis=0, weights=weights)
    total_bins = float(ordered_sum(exposure.total_bins, axis=0, weights=weights))
    edge_fraction = np.divide(edge_exposed, total_bins, out=np.zeros_like(edge_exposed), where=total_bins > 0.0)
    pair_fraction = np.divide(pair_exposed, total_bins, out=np.zeros_like(pair_exposed), where=total_bins > 0.0)
    # Exposed chromosomes count by their map units (a short chromosome once), not one each.
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    edge_units = _units_present(exposure.edge_present, weights, exposure.edge_units)
    pair_units = _units_present(exposure.pair_present, weights, exposure.pair_units)
    edge_ok = (
        (edge_exposed >= config.minimum_edge_exposed_bins)
        & (edge_fraction >= config.minimum_exposed_fraction)
        & (edge_units >= config.minimum_exposed_units)
    )
    pair_ok = (
        (pair_exposed >= config.minimum_pair_exposed_bins)
        & (pair_fraction >= config.minimum_exposed_fraction)
        & (pair_units >= config.minimum_exposed_units)
    )
    rows, states = alternatives.rows, alternatives.states
    first, second = exposure.row_edges[:, 0], exposure.row_edges[:, 1]
    result = np.zeros(len(rows), dtype=np.bool_)
    result[states == M0] = True
    m1 = np.flatnonzero(states == M1)
    result[m1] = edge_ok[first[m1]]
    m2 = np.flatnonzero(states == M2)
    result[m2] = edge_ok[first[m2]] & edge_ok[second[m2]] & pair_ok[alternatives.panel_row[m2]]
    return result


def fit_scores(
    weights: np.ndarray,
    inputs: DecisionInputs,
    orientation: Orientation,
    config: PedigreeParameters,
    trace: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(state scores, identity scores) [A] of one weighted fit, family terms included.

    The chromosome sum multiplies every row, including zero weights; IEEE nonfinite
    propagation differs from the panel-admission fold that omits zero-weight rows.
    """
    alternatives = inputs.alternatives
    aggregate = alternatives_module.mix_m0(
        ordered_sum(alternatives.log_likelihoods, axis=0, weights=weights),
        alternatives,
        config.selection.m0_contamination_probability,
    )
    gate = testable(weights, inputs, config.exposure)
    nonzero = alternatives.states != M0
    state_scores = aggregate.copy()
    identity_scores = aggregate.copy()
    untestable = nonzero & ~gate
    state_scores[untestable] = aggregate[alternatives.m0_row[untestable]]
    identity_scores[untestable] = -np.inf
    if trace is not None:
        trace.update(
            genetic=aggregate.copy(),
            structural_state=state_scores.copy(),
            structural_identity=identity_scores.copy(),
            testable=gate,
        )
    return family.adjust(
        state_scores,
        identity_scores,
        alternatives.rows,
        alternatives.states,
        alternatives.full_counts,
        orientation,
        config.direction,
        trace,
    )


# ------------------------------------------------------------------------------- selection


def select(
    state_scores: np.ndarray,
    identity_scores: np.ndarray,
    alternatives: Alternatives,
    priors: tuple[float, float, float],
    orientation: Orientation,
    config: PedigreeParameters,
    *,
    counter=None,
) -> Selection:
    """Marginal state, conditional identity and acyclic graph selection of one fit."""
    n_samples = alternatives.n_samples
    states = alternatives.states
    state_evidence = alternatives_module.state_log_evidence(state_scores, alternatives)
    prior = np.tile(np.asarray(priors, dtype=np.float64), (n_samples, 1))
    marginal = state_evidence + np.log(prior)
    support = np.zeros((n_samples, 3), dtype=np.float64)
    decision = np.full(len(identity_scores), -np.inf, dtype=np.float64)
    state: int | None  # a state index here; below, each child's unique marginal winner (None if tied)
    for child, child_rows in enumerate(alternatives.child_rows):
        for state in range(3):
            rows = child_rows[states[child_rows] == state]
            finite = rows[np.isfinite(identity_scores[rows])]
            if not len(finite) or not np.isfinite(marginal[child, state]):
                continue
            decision[finite] = marginal[child, state] + identity_scores[finite] - float(np.max(identity_scores[finite]))
        support[child] = softmax_finite(marginal[child])

    local_state = np.full(n_samples, UNRESOLVED, dtype=np.int8)
    local_row = np.full(n_samples, -1, dtype=np.int64)
    state_margin = np.full(n_samples, np.nan, dtype=np.float64)
    identity_margin = np.full(n_samples, np.nan, dtype=np.float64)
    # Per child, the largest |score| its two margins were computed from: their rounding scale,
    # which sets the graph step's weakest-link tie tolerance.
    score_scale = np.ones(n_samples, dtype=np.float64)
    reasons: list[str | None] = [None] * n_samples
    local_rows: dict[int, int] = {}
    state_indices = np.arange(3, dtype=np.int64)
    for child, rows in enumerate(alternatives.child_rows):
        state, margin = unique_finite_winner(state_indices, marginal[child], key=f"state/c{child}", counter=counter)
        state_margin[child] = margin
        if state is None:
            reasons[child] = "no_unique_marginal_parent_state"
            continue
        local_state[child] = state
        candidates = rows[states[rows] == state]
        row, margin = unique_finite_winner(candidates, decision, key=f"identity/c{child}", counter=counter)
        identity_margin[child] = margin
        if row is None:
            reasons[child] = "parent_state_resolved_identity_unresolved"
            continue
        local_row[child] = row
        local_rows[child] = row
        score_scale[child] = max(
            magnitude(marginal[child]), magnitude(identity_scores[candidates]), magnitude(decision[candidates])
        )

    rows = alternatives.rows
    tie_conflict = np.zeros(n_samples, dtype=np.bool_)
    if graph.is_acyclic(local_rows, rows, n_samples):
        chosen = graph.select_local(
            rows, local_rows, state_margin, identity_margin, orientation, reverse=config.reverse_conventions
        )
    else:
        conflicts = graph.tie_conflict_children(rows, local_rows, state_margin, identity_margin, score_scale, n_samples)
        tie_conflict[list(conflicts)] = True
        chosen = graph.select_acyclic(
            alternatives,
            decision,
            LocalGraphChoices(
                {child: row for child, row in local_rows.items() if child not in conflicts},
                state_margin,
                identity_margin,
            ),
            config.selection.graph_local_search_passes,
            orientation,
            reverse=config.reverse_conventions,
        )
    graph_row = np.full(n_samples, -1, dtype=np.int64)
    role_probability = np.full(n_samples, np.nan, dtype=np.float64)
    direction_resolved = np.zeros(n_samples, dtype=np.bool_)
    for child, row in chosen.rows.items():
        graph_row[child] = row
        role_probability[child] = chosen.role_probability[child]
    direction_resolved[list(chosen.direction_resolved)] = True
    return Selection(
        state_evidence,
        marginal,
        support,
        decision,
        prior,
        local_state,
        local_row,
        graph_row,
        tie_conflict,
        direction_resolved,
        role_probability,
        state_margin,
        identity_margin,
        tuple(reasons),
        graph_passes_used=chosen.graph_passes_used,
        graph_pass_budget_exhausted=chosen.graph_pass_budget_exhausted,
    )


def fit(
    weights: np.ndarray, inputs: DecisionInputs, config: PedigreeParameters, trace: dict | None = None, *, counter=None
) -> tuple[Selection, Orientation, tuple[np.ndarray, np.ndarray]]:
    """One weighted fit with the configured priors: (selection, orientation, (state, identity))."""
    orientation = fit_orientation(inputs.junctions, weights)
    scores = fit_scores(weights, inputs, orientation, config, trace)
    selection = select(
        *scores, inputs.alternatives, config.selection.state_priors, orientation, config, counter=counter
    )
    return selection, orientation, scores


def build_exposure(scores: GenomeScores, alternatives: Alternatives, config: ExposureParameters) -> ExposureEvidence:
    """Derive exposure from scored contigs; absent map lengths count one unit per chromosome."""
    edges, edge_exposed, pair_exposed = scores.edges, scores.edge_exposed, scores.pair_exposed
    total_bins = scores.total_bins
    scored_morgans = scores.scored_morgans
    edge_exposed_morgans, pair_exposed_morgans = scores.edge_exposed_morgans, scores.pair_exposed_morgans
    edge_flags = (edge_exposed > 0.0) & (edge_exposed >= config.minimum_edge_exposed_bins)
    pair_flags = (pair_exposed > 0.0) & (pair_exposed >= config.minimum_pair_exposed_bins)
    rows = alternatives.rows
    row_edges = np.full((len(rows), 2), -1, dtype=np.int64)
    for slot in (1, 2):
        named = np.flatnonzero(rows[:, slot] >= 0)
        row_edges[named, slot - 1] = edge_index(edges, alternatives.n_samples, rows[named, 0], rows[named, slot])
    if np.any(row_edges[rows[:, 1:] >= 0] < 0):
        raise InputError("a configuration row names a parent outside the eligible edges")

    def units(morgans, shape):
        return map_units(np.full(shape, np.nan) if morgans is None else morgans, config)

    return ExposureEvidence(
        edge_exposed,
        pair_exposed,
        total_bins,
        _pack_presence(np.ascontiguousarray(edge_flags)),
        _pack_presence(np.ascontiguousarray(pair_flags)),
        row_edges,
        units(scored_morgans, len(total_bins)),
        units(edge_exposed_morgans, edge_exposed.shape),
        units(pair_exposed_morgans, pair_exposed.shape),
    )
