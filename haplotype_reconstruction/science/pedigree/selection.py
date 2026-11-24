"""One weighted fit: exposure gate, birth-order posterior, marginal state, identity, joint (acyclic) read.

A fit gives every material unit (1-Morgan units of the working map, DESIGN_417) a multiplicity (all ones
for the full data, bootstrap counts or a left-out unit in resampling). The genome-wide likelihood of a
configuration is the weighted unit sum, mixed once with the child's M0 likelihood. A non-M0 configuration is
testable only when its edges (and pair) are exposed in enough bins, enough of the painted genome and enough
distinct units (each unit counts one); an untestable configuration contributes the child's M0 likelihood
to its state's mean but cannot be named. The birth-order layer (order.adjust) turns the scores into the sampled
configuration posterior. The marginal state is chosen first (log mean identity evidence plus a fixed prior), then the
identity within it; where these local choices form a cycle, the joint read gives the children on cycles their choice
among the configurations consistent with the fit's consensus birth order (select).

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
    OmissionTotals,
    Selection,
    edge_index,
)
from ...core import ties as tie_rules
from ...core.errors import InputError, InvariantError
from ...core.kernels import kernel
from ...core.numerics import ordered_sum
from . import alternatives as alternatives_module
from . import graph, order
from .alternatives import Alternatives, softmax_finite
from .decision_parameters import ExposureParameters
from .kernels.decision import OVERLOADS
from .orientation import Orientation
from .orientation import fit as fit_orientation
from .parameters import PedigreeParameters
from .ties import magnitude, unique_finite_winner

__all__ = ["testable", "fit_scores", "select", "fit", "build_exposure"]


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
def _units_present(packed, weights):
    """float64 [items]: the number of distinct units with positive weight whose presence bit is set (a unit drawn
    several times counts once)."""
    n_words, n_items = packed.shape
    totals = np.zeros(n_items, dtype=np.float64)
    one = np.uint64(1)
    for unit in range(weights.shape[0]):
        if weights[unit] > 0:
            bit = one << np.uint64(unit % 64)
            word = unit // 64
            for item in range(n_items):
                if packed[word, item] & bit:
                    totals[item] += 1.0
    return totals


# ------------------------------------------------------------------------------- gate and scores
def testable(weights: np.ndarray, inputs: DecisionInputs, config: ExposureParameters, *, omitted=None) -> np.ndarray:
    """bool [A]: configurations testable on the weighted units (M0 always): every edge (and an M2 pair) exposed
    in enough bins, enough of the fit's painted bins and on enough distinct units."""
    exposure = inputs.exposure
    alternatives = inputs.alternatives
    totals = inputs.omission_totals
    if omitted is None:
        edge_exposed = ordered_sum(exposure.edge_exposed, axis=0, weights=weights)
        pair_exposed = ordered_sum(exposure.pair_exposed, axis=0, weights=weights)
        total_bins = float(ordered_sum(exposure.total_bins, axis=0, weights=weights))
    else:
        totals = inputs.omission_totals
        assert totals is not None
        edge_exposed = totals.edge_exposed - exposure.edge_exposed[omitted]
        pair_exposed = totals.pair_exposed - exposure.pair_exposed[omitted]
        total_bins = totals.total_bins - float(exposure.total_bins[omitted])
    edge_fraction = np.divide(edge_exposed, total_bins, out=np.zeros_like(edge_exposed), where=total_bins > 0.0)
    pair_fraction = np.divide(pair_exposed, total_bins, out=np.zeros_like(pair_exposed), where=total_bins > 0.0)
    # Each exposed material unit counts one (DESIGN_417 D2).
    if omitted is None:
        weights = np.ascontiguousarray(weights, dtype=np.float64)
        edge_units = _units_present(exposure.edge_present, weights)
        pair_units = _units_present(exposure.pair_present, weights)
    else:
        assert totals is not None
        bit = np.uint64(1) << np.uint64(omitted % 64)
        edge_units = totals.edge_units - ((exposure.edge_present[omitted // 64] & bit) != 0)
        pair_units = totals.pair_units - ((exposure.pair_present[omitted // 64] & bit) != 0)
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
    *, omitted=None,
) -> order.OrderFit:
    """The birth-order layer of one weighted fit (order.OrderFit: state and identity scores [A], consensus order).

    The unit sum multiplies every row, including zero weights; IEEE nonfinite
    propagation differs from the panel-admission fold that omits zero-weight rows.
    """
    alternatives = inputs.alternatives
    with np.errstate(invalid="ignore"):
        if omitted is None:
            raw = ordered_sum(alternatives.log_likelihoods, axis=0, weights=weights)
        else:
            totals = inputs.omission_totals
            assert totals is not None
            raw = totals.genetic - alternatives.log_likelihoods[omitted]
    aggregate = alternatives_module.mix_m0(
        raw,
        alternatives,
        config.selection.m0_contamination_probability,
    )
    gate = testable(weights, inputs, config.exposure, omitted=omitted)
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
    # the fit's own random stream: its unit multiplicities (empty for a leave-one-unit-out fit) and its left-out unit,
    # under the run's bootstrap seed, drawn over the samples in sorted-identifier order
    key = order.fit_key(weights, omitted)
    return order.adjust(
        state_scores,
        identity_scores,
        alternatives.rows,
        alternatives.states,
        alternatives.full_counts,
        orientation,
        config.direction,
        config.order,
        key,
        trace,
        seed=config.resampling.bootstrap_seed,
        sample_order=inputs.sample_order,
    )


# ------------------------------------------------------------------------------- selection


def _state_choices(state_scores: np.ndarray, identity_scores: np.ndarray, alternatives: Alternatives,
                   priors: tuple[float, float, float]):
    """(state log evidence, state scores with the log prior, state support [N, 3], decision scores [A]) of one set of
    configuration scores."""
    n_samples = alternatives.n_samples
    states = alternatives.states
    state_evidence = alternatives_module.state_log_evidence(state_scores, alternatives)
    prior = np.tile(np.asarray(priors, dtype=np.float64), (n_samples, 1))
    marginal = state_evidence + np.log(prior)
    support = np.zeros((n_samples, 3), dtype=np.float64)
    decision = np.full(len(identity_scores), -np.inf, dtype=np.float64)
    for child, child_rows in enumerate(alternatives.child_rows):
        for state in range(3):
            rows = child_rows[states[child_rows] == state]
            finite = rows[np.isfinite(identity_scores[rows])]
            if not len(finite) or not np.isfinite(marginal[child, state]):
                continue
            decision[finite] = marginal[child, state] + identity_scores[finite] - float(np.max(identity_scores[finite]))
        support[child] = softmax_finite(marginal[child])
    return state_evidence, marginal, prior, support, decision


def _child_choice(child, child_rows, states, marginal, identity_scores, decision, counter, prefix=""):
    """One child's marginal state then identity: (state or None, row or None, state margin, identity margin, score
    scale, unresolved reason)."""
    state, state_margin = unique_finite_winner(np.arange(3, dtype=np.int64), marginal[child],
                                               key=f"{prefix}state/c{child}", counter=counter)
    if state is None:
        return None, None, state_margin, np.nan, 1.0, "no_unique_marginal_parent_state"
    candidates = child_rows[states[child_rows] == state]
    row, identity_margin = unique_finite_winner(candidates, decision, key=f"{prefix}identity/c{child}", counter=counter)
    if row is None:
        return state, None, state_margin, identity_margin, 1.0, "parent_state_resolved_identity_unresolved"
    scale = max(magnitude(marginal[child]), magnitude(identity_scores[candidates]), magnitude(decision[candidates]))
    return state, row, state_margin, identity_margin, scale, None


def order_consistent(
    rows: np.ndarray, constrained: np.ndarray, rank: np.ndarray, *, reverse: bool = False
) -> np.ndarray:
    """bool [A]: configurations allowed by the consensus birth order: every named parent born before the child, the
    samples ordered by ``rank`` (order.OrderFit.rank; equal ranks in the run's index convention). Rows that name
    nobody in the order factor (``constrained`` False: M0 and exposure-untestable rows) are always allowed."""
    n_samples = len(rank)
    index = np.arange(n_samples)
    ordered = np.lexsort((-index if reverse else index, np.asarray(rank, dtype=np.float64)))
    position = np.empty(n_samples, dtype=np.int64)
    position[ordered] = np.arange(n_samples)
    allowed = np.ones(len(rows), dtype=np.bool_)
    child = rows[:, 0]
    for slot in (1, 2):
        named = rows[:, slot] >= 0
        allowed[named] &= position[rows[named, slot]] < position[child[named]]
    return allowed | ~constrained


def select(
    fit: order.OrderFit,
    alternatives: Alternatives,
    priors: tuple[float, float, float],
    orientation: Orientation,
    config: PedigreeParameters,
    *,
    counter=None,
) -> Selection:
    """Marginal state, conditional identity and the joint read of one fit.

    The local choice of each child is its marginal state, then its identity, from the birth-order posterior. Where the
    local choices form a DAG they are the joint read. Otherwise the children on a cycle take their choice among the
    configurations the fit's consensus birth order allows (order_consistent: the same marginal state then identity on
    those configurations), repeated until no cycle remains: every cycle holds a child not yet restricted (restricted
    children's parents precede them in one total order), so each round restricts at least one more child."""
    n_samples = alternatives.n_samples
    states = alternatives.states
    state_scores, identity_scores = fit.state_scores, fit.identity_scores
    state_evidence, marginal, prior, support, decision = _state_choices(
        state_scores, identity_scores, alternatives, priors
    )
    local_state = np.full(n_samples, UNRESOLVED, dtype=np.int8)
    local_row = np.full(n_samples, -1, dtype=np.int64)
    state_margin = np.full(n_samples, np.nan, dtype=np.float64)
    identity_margin = np.full(n_samples, np.nan, dtype=np.float64)
    reasons: list[str | None] = [None] * n_samples
    local_rows: dict[int, int] = {}
    for child, rows in enumerate(alternatives.child_rows):
        state, row, state_margin[child], identity_margin[child], _, reasons[child] = _child_choice(
            child, rows, states, marginal, identity_scores, decision, counter
        )
        if state is not None:
            local_state[child] = state
        if row is not None:
            local_row[child] = row
            local_rows[child] = row

    rows = alternatives.rows
    graph_rows = dict(local_rows)
    graph_state_margin, graph_identity_margin = state_margin.copy(), identity_margin.copy()
    restricted = np.zeros(n_samples, dtype=np.bool_)
    if not graph.is_acyclic(graph_rows, rows, n_samples):
        allowed = order_consistent(rows, np.isfinite(identity_scores) & ((rows[:, 1] >= 0) | (rows[:, 2] >= 0)),
                                   fit.rank, reverse=config.reverse_conventions)
        ordered_state = np.where(allowed, state_scores, -np.inf)
        ordered_identity = np.where(allowed, identity_scores, -np.inf)
        _, ordered_marginal, _, _, ordered_decision = _state_choices(ordered_state, ordered_identity, alternatives,
                                                                   priors)
        while True:
            members = graph.cycle_members(graph_rows, rows, n_samples)
            if not members:
                break
            unrestricted = [child for child in members if not restricted[child]]
            if not unrestricted:
                raise InvariantError("a cycle among children restricted to one birth order")
            for child in sorted(unrestricted, key=lambda c: tie_rules.conv_key(c, reverse=config.reverse_conventions)):
                restricted[child] = True
                _, row, graph_state_margin[child], graph_identity_margin[child], _, _ = _child_choice(
                    child, alternatives.child_rows[child], states, ordered_marginal, ordered_identity,
                    ordered_decision, counter, prefix="ordered_",
                )
                graph_rows.pop(child, None)
                if row is not None:
                    graph_rows[child] = int(row)
    chosen = graph.select_local(
        rows, graph_rows, graph_state_margin, graph_identity_margin, orientation, reverse=config.reverse_conventions
    )
    graph_row = np.full(n_samples, -1, dtype=np.int64)
    role_probability = np.full(n_samples, np.nan, dtype=np.float64)
    direction_resolved = np.zeros(n_samples, dtype=np.bool_)
    for child, row in chosen.rows.items():
        graph_row[child] = row
        role_probability[child] = chosen.role_probability[child]
        # the joint read displaced the local choice by a configuration naming a parent
        direction_resolved[child] = bool(restricted[child] and row != local_row[child] and int(states[row]) != M0)
    return Selection(
        state_evidence,
        marginal,
        support,
        decision,
        prior,
        local_state,
        local_row,
        graph_row,
        restricted,
        direction_resolved,
        role_probability,
        state_margin,
        identity_margin,
        tuple(reasons),
    )


def fit(
    weights: np.ndarray, inputs: DecisionInputs, config: PedigreeParameters, trace: dict | None = None, *, counter=None,
    omitted=None,
) -> tuple[Selection, Orientation, order.OrderFit]:
    """One weighted fit with the configured priors: (selection, orientation, birth-order layer)."""
    if omitted is None:
        orientation = fit_orientation(inputs.junctions, weights)
    else:
        from .orientation import fit_omitting
        orientation = fit_omitting(inputs.junctions, inputs.omission_totals, omitted)
    scores = fit_scores(weights, inputs, orientation, config, trace, omitted=omitted)
    selection = select(scores, inputs.alternatives, config.selection.state_priors, orientation, config, counter=counter)
    return selection, orientation, scores


def build_exposure(scores: GenomeScores, alternatives: Alternatives, config: ExposureParameters) -> ExposureEvidence:
    """Derive exposure from the material units; every unit counts one."""
    edges, edge_exposed, pair_exposed = scores.edges, scores.edge_exposed, scores.pair_exposed
    total_bins = scores.total_bins
    edge_flags = (edge_exposed > 0.0) & (edge_exposed >= config.minimum_edge_exposed_bins)
    pair_flags = (pair_exposed > 0.0) & (pair_exposed >= config.minimum_pair_exposed_bins)
    rows = alternatives.rows
    row_edges = np.full((len(rows), 2), -1, dtype=np.int64)
    for slot in (1, 2):
        named = np.flatnonzero(rows[:, slot] >= 0)
        row_edges[named, slot - 1] = edge_index(edges, alternatives.n_samples, rows[named, 0], rows[named, slot])
    if np.any(row_edges[rows[:, 1:] >= 0] < 0):
        raise InputError("a configuration row names a parent outside the eligible edges")
    return ExposureEvidence(
        edge_exposed,
        pair_exposed,
        total_bins,
        _pack_presence(np.ascontiguousarray(edge_flags)),
        _pack_presence(np.ascontiguousarray(pair_flags)),
        row_edges,
    )



def omission_totals(inputs):
    """One O(H) pass over additive inputs; no H by H omission vectors or caches."""
    from .orientation import _order_statistics
    exposure, junctions = inputs.exposure, inputs.junctions
    weights = np.ones(len(exposure.total_bins))
    return OmissionTotals(
        ordered_sum(inputs.alternatives.log_likelihoods, axis=0),
        ordered_sum(exposure.edge_exposed, axis=0), ordered_sum(exposure.pair_exposed, axis=0),
        float(ordered_sum(exposure.total_bins)),
        _units_present(exposure.edge_present, weights),
        _units_present(exposure.pair_present, weights),
        junctions.junctions.sum(axis=0), junctions.callable_bins.sum(axis=0),
        junctions.pair_log_odds.sum(axis=0),
        _order_statistics(junctions.junctions, junctions.callable_bins, junctions.total_bins),
        pair_log_cosh=None if junctions.pair_log_cosh is None else junctions.pair_log_cosh.sum(axis=0))
