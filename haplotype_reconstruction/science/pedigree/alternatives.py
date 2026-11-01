"""Parent-state configurations of every eligible child and their per-chromosome likelihoods.

Each eligible child has one M0 row (no observed parent), one M1 row per eligible parent and one
M2 row per panel pair. Marginal state evidence integrates the identities of a state with the full
eligible multiplicity as the identity-prior denominator, so an incomplete M2 screen gives a lower
bound, never an inflated state.

Method: docs/methods/pedigree.md#configuration-likelihoods-and-the-exposure-gate.
"""

from __future__ import annotations

import math

import numpy as np

from ...contracts.pedigree import M0, M1, M2, NO_PARENT, Alternatives, GenomeScores
from ...core.numerics import log_sum
from .eligibility import ParentEligibility

__all__ = ["build", "mix_m0", "softmax_finite", "state_log_evidence"]


def build(scores: GenomeScores, eligibility: ParentEligibility) -> Alternatives:
    """Configuration rows of the eligible universe; M2 rows are the child's panel rows.

    ``eligibility`` must be the one ``scores`` was stacked with: a child's M1 rows are its run of ``scores.edges``.
    """
    trios = scores.panel.trios
    n_contigs = len(scores.contigs)
    n_samples = len(scores.sample_ids)
    child_starts = np.searchsorted(trios[:, 0], np.arange(n_samples + 1, dtype=np.int64))
    parent_counts = eligibility.parent_counts()
    panel_by_child = []
    for child in range(n_samples):
        rows = np.arange(child_starts[child], child_starts[child + 1], dtype=np.int64)
        if len(rows):
            rows = rows[eligibility.pair_mask(child, trios[rows, 1], trios[rows, 2])]
        panel_by_child.append(rows)
    n_rows = int(
        sum(
            1 + parent_counts[child] + len(panel_by_child[child])
            for child in range(n_samples)
            if eligibility.children[child]
        )
    )
    rows = np.empty((n_rows, 3), dtype=np.int64)
    states = np.empty(n_rows, dtype=np.int8)
    m0_row = np.empty(n_rows, dtype=np.int64)
    panel_row = np.full(n_rows, -1, dtype=np.int64)
    log_likelihoods = np.empty((n_contigs, n_rows), dtype=np.float64)
    child_rows = []
    scored_counts = np.zeros((n_samples, 3), dtype=np.int64)
    # The M1 rows of a child are its eligible edges, a contiguous run of the genome scores' edge axis.
    edge_starts = scores.edge_starts()
    offset = 0
    for child in range(n_samples):
        if not eligibility.children[child]:
            child_rows.append(np.empty(0, dtype=np.int64))
            continue
        start = offset
        rows[offset] = (child, NO_PARENT, NO_PARENT)
        states[offset] = M0
        log_likelihoods[:, offset] = scores.zero[:, child]
        offset += 1
        edges = slice(edge_starts[child], edge_starts[child + 1])
        parents = scores.edges[edges, 1]
        stop = offset + len(parents)
        rows[offset:stop, 0] = child
        rows[offset:stop, 1] = parents
        rows[offset:stop, 2] = NO_PARENT
        states[offset:stop] = M1
        log_likelihoods[:, offset:stop] = scores.one[:, edges]
        offset = stop
        panel = panel_by_child[child]
        stop = offset + len(panel)
        rows[offset:stop] = trios[panel]
        states[offset:stop] = M2
        panel_row[offset:stop] = panel
        log_likelihoods[:, offset:stop] = scores.two[:, panel]
        offset = stop
        m0_row[start:offset] = start
        child_rows.append(np.arange(start, offset, dtype=np.int64))
        scored_counts[child] = (1, len(parents), len(panel))
    full_counts = np.stack([eligibility.children.astype(np.int64), parent_counts, eligibility.pair_counts()], axis=1)
    return Alternatives(rows, states, tuple(child_rows), m0_row, panel_row, log_likelihoods, full_counts, scored_counts)


def mix_m0(aggregate: np.ndarray, alternatives: Alternatives, probability: float) -> np.ndarray:
    """Mix every aggregated M1/M2 likelihood with its child's M0 likelihood at ``probability``.

    One contamination term per genome-wide configuration (not per chromosome): a configuration
    that fails somewhere cannot be worse than "no observed parent" by more than log(probability).
    """
    if probability == 0.0:
        return aggregate
    mixed = aggregate.copy()
    nonzero = np.flatnonzero(alternatives.states != M0)
    mixed[nonzero] = np.logaddexp(
        math.log1p(-probability) + aggregate[nonzero], math.log(probability) + aggregate[alternatives.m0_row[nonzero]]
    )
    return mixed


logsumexp_finite = log_sum


def softmax_finite(values: np.ndarray) -> np.ndarray:
    """Normalised exp over the finite entries; zero elsewhere (all zero without finite entries)."""
    values = np.asarray(values, dtype=np.float64)
    output = np.zeros_like(values)
    finite = np.isfinite(values)
    if not np.any(finite):
        return output
    maximum = float(np.max(values[finite]))
    output[finite] = np.exp(values[finite] - maximum)
    output /= np.sum(output)
    return output


def state_log_evidence(aggregate: np.ndarray, alternatives: Alternatives) -> np.ndarray:
    """float64 [N, 3]: log of the mean likelihood of each state's identities (full multiplicity)."""
    evidence = np.full((alternatives.n_samples, 3), -np.inf, dtype=np.float64)
    for child, child_rows in enumerate(alternatives.child_rows):
        for state in range(3):
            rows = child_rows[alternatives.states[child_rows] == state]
            if len(rows):
                evidence[child, state] = logsumexp_finite(aggregate[rows]) - math.log(
                    float(alternatives.full_counts[child, state])
                )
    return evidence
