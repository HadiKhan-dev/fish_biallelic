"""Parent-state configurations of every eligible child and their per-chromosome likelihoods.

Each eligible child has one M0 row (no observed parent), one M1 row per eligible parent and one
M2 row per panel pair. Marginal state evidence integrates the identities of a state with the full
eligible multiplicity as the identity-prior denominator, so an incomplete M2 screen gives a lower
bound, never an inflated state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .eligibility import ParentEligibility
from .records import M0, M1, M2, NO_PARENT, GenomeScores


@dataclass(frozen=True)
class Alternatives:
    """M0/M1/M2 configuration rows (A rows) grouped by child.

    rows:            int64 [A, 3] (child, parent, parent); NO_PARENT in empty slots. Per child: the M0
                     row, M1 rows (parents ascending), then M2 rows (panel order).
    states:          int8 [A] observed-parent count of each row.
    child_rows:      per sample, int64 row indices (empty for samples that are not targets).
    m0_row:          int64 [A] row index of the same child's M0 row.
    panel_row:       int64 [A] index of an M2 row in the panel trios, -1 for M0/M1 rows.
    log_likelihoods: float64 [C, A] per-chromosome composite log likelihood of each row.
    full_counts:     int64 [N, 3] eligible universe size per state (identity-prior denominator).
    scored_counts:   int64 [N, 3] rows actually scored per state (M2 is screened).
    """

    rows: np.ndarray
    states: np.ndarray
    child_rows: tuple[np.ndarray, ...]
    m0_row: np.ndarray
    panel_row: np.ndarray
    log_likelihoods: np.ndarray
    full_counts: np.ndarray
    scored_counts: np.ndarray

    @property
    def n_samples(self) -> int:
        return len(self.child_rows)

    def scaled(self, scale: float) -> "Alternatives":
        """The same rows with every log likelihood multiplied by ``scale``."""
        return replace(self, log_likelihoods=self.log_likelihoods * scale)


def build(scores: GenomeScores, eligibility: ParentEligibility) -> Alternatives:
    """Configuration rows of the eligible universe; M2 rows are the child's panel rows."""
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
    n_rows = int(sum(1 + parent_counts[child] + len(panel_by_child[child])
                     for child in range(n_samples) if eligibility.children[child]))
    rows = np.empty((n_rows, 3), dtype=np.int64)
    states = np.empty(n_rows, dtype=np.int8)
    m0_row = np.empty(n_rows, dtype=np.int64)
    panel_row = np.full(n_rows, -1, dtype=np.int64)
    log_likelihoods = np.empty((n_contigs, n_rows), dtype=np.float64)
    child_rows = []
    scored_counts = np.zeros((n_samples, 3), dtype=np.int64)
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
        parents = np.flatnonzero(eligibility.parents[child])
        stop = offset + len(parents)
        rows[offset:stop, 0] = child
        rows[offset:stop, 1] = parents
        rows[offset:stop, 2] = NO_PARENT
        states[offset:stop] = M1
        log_likelihoods[:, offset:stop] = scores.one[:, child, parents]
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
    full_counts = np.stack([eligibility.children.astype(np.int64), parent_counts,
                            eligibility.pair_counts()], axis=1)
    return Alternatives(rows, states, tuple(child_rows), m0_row, panel_row, log_likelihoods,
                        full_counts, scored_counts)


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
        math.log1p(-probability) + aggregate[nonzero],
        math.log(probability) + aggregate[alternatives.m0_row[nonzero]])
    return mixed


def logsumexp_finite(values: np.ndarray) -> float:
    """log(sum(exp(values))) for a non-empty vector; -inf for an empty one."""
    if len(values) == 0:
        return -np.inf
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        return maximum
    return maximum + math.log(float(np.sum(np.exp(values - maximum))))


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
                evidence[child, state] = (logsumexp_finite(aggregate[rows])
                                          - math.log(float(alternatives.full_counts[child, state])))
    return evidence
