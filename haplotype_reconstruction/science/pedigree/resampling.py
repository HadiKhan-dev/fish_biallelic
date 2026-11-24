"""Material-unit bootstrap and leave-one-unit-out (LOBO) support of the full-data decisions.

Each replicate refits everything downstream of the genetic scores (orientation, exposure gate,
birth-order layer, state, identity, joint read) with unit multiplicities: seeded bootstrap draws of
H units with replacement, and H leave-one-unit-out fits (separate counts and denominator) read from the full
totals minus the left-out unit (selection.omission_totals). Counts are integer sums, so support does not
depend on worker count or order. A fit is a deterministic function of its multiplicities (the birth-order sampler's
stream is keyed by the fit's multiplicity vector, or its left-out unit, under the run's bootstrap seed: order.fit_key),
so each distinct bootstrap multiplicity vector is fitted once and counted as many times as it was drawn. Units are the
genome's 1-Morgan material units (DESIGN_417): no chromosome or contig count enters the resampling.

Method: docs/methods/pedigree.md#pd16.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

import numpy as np

from ...contracts.pedigree import Resampling, SupportCounts
from ...core.errors import InvariantError
from ...core.kernels import kernel
from ...core.random import bootstrap_multiplicities
from . import selection
from .kernels.decision import OVERLOADS
from .parameters import PedigreeParameters
from .selection import DecisionInputs

__all__ = [
    "FitTask",
    "FitArrays",
    "FitResult",
    "ResamplingPlan",
    "named_edges",
    "evaluate",
    "evaluate_task",
    "plan",
    "combine",
]

# Work and results keep the existing coalesced task order and integer draw counts.
FitTask = tuple[bool, np.ndarray, np.ndarray, PedigreeParameters]
FitArrays = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
FitResult = tuple[bool, np.ndarray, FitArrays]


# The fits are spread over workers only for at least this many (alternatives x replicates) selection scans.
_MINIMUM_POOL_WORK = 100_000


def named_edges(rows: np.ndarray, samples: int) -> np.ndarray:
    """int64 [E] sorted keys child * N + parent of the edges the configuration rows name."""
    keys = np.concatenate([rows[:, 0] * samples + rows[:, slot] for slot in (1, 2)])
    named = np.concatenate([rows[:, slot] >= 0 for slot in (1, 2)])
    return np.unique(keys[named]).astype(np.int64)


@kernel(overloads=OVERLOADS["resampling._accumulate"], cache=True, nogil=True)
def _accumulate(
    local_rows,
    graph_rows,
    local_states,
    draws,
    rows,
    edges,
    local_configuration,
    graph_configuration,
    local_state,
    local_parent,
    graph_parent,
):
    """Add each fitted multiplicity vector's selections ``draws[i]`` times (the replicates that drew it)."""
    n_replicates, n_samples = local_rows.shape
    for replicate in range(n_replicates):
        count = draws[replicate]
        for child in range(n_samples):
            state = local_states[replicate, child]
            if state >= 0:
                local_state[child, state] += count
            row = local_rows[replicate, child]
            if row >= 0:
                local_configuration[row] += count
                for slot in range(1, 3):
                    parent = rows[row, slot]
                    if parent >= 0:
                        local_parent[np.searchsorted(edges, child * n_samples + parent)] += count
            row = graph_rows[replicate, child]
            if row >= 0:
                graph_configuration[row] += count
                for slot in range(1, 3):
                    parent = rows[row, slot]
                    if parent >= 0:
                        graph_parent[np.searchsorted(edges, child * n_samples + parent)] += count


def evaluate(
    inputs: DecisionInputs, config: PedigreeParameters, multiplicities: np.ndarray, *, omissions=False
) -> FitArrays:
    """Selections of a chunk of multiplicity vectors: local rows, graph rows, local states and whether the joint read
    restricted any child to the fit's consensus birth order, per vector."""
    n_replicates = len(multiplicities)
    n_samples = inputs.alternatives.n_samples
    local_rows = np.full((n_replicates, n_samples), -1, dtype=np.int64)
    graph_rows = np.full((n_replicates, n_samples), -1, dtype=np.int64)
    local_states = np.full((n_replicates, n_samples), -1, dtype=np.int8)
    restricted = np.zeros(n_replicates, dtype=np.bool_)
    for replicate, weights in enumerate(multiplicities):
        chosen, _, _ = (selection.fit(np.empty(0), inputs, config, omitted=int(weights[0]))
                         if omissions else selection.fit(weights, inputs, config))
        local_states[replicate] = chosen.local_state
        local_rows[replicate] = chosen.local_row
        graph_rows[replicate] = chosen.graph_row
        restricted[replicate] = bool(np.any(chosen.order_restricted))
    return local_rows, graph_rows, local_states, restricted


# --------------------------------------------------------------------- driver


@dataclass(frozen=True)
class ResamplingPlan:
    """The fixed resampling work of one decision: the seeded draws and the ordered independent fit tasks.

    multiplicities: float64 [R, H] the R bootstrap draws (PCG64 stream of the configured seed).
    tasks: the fit tasks in their fixed order: the H leave-one-unit-out fits first, then the distinct bootstrap
    vectors that omit the most units (one vector per task when the work is spread over workers).
    worker_count: the workers the work is planned for (1 = one task holding every distinct vector).
    """

    multiplicities: np.ndarray
    tasks: tuple[FitTask, ...]
    worker_count: int


def plan(inputs: DecisionInputs, config: PedigreeParameters, cpus: int = 1) -> ResamplingPlan:
    """Draw the bootstrap multiplicities and order the independent fits for at most ``cpus`` workers."""
    from .regimes import effective
    config = effective(config, inputs.regime)
    alternatives = inputs.alternatives
    n_contigs = alternatives.log_likelihoods.shape[0]
    replicates = config.resampling.bootstrap_replicates
    multiplicities = bootstrap_multiplicities(n_contigs, replicates, config.resampling.bootstrap_seed)
    # Distinct vectors (each fitted once) and how many replicates drew each.
    distinct, draws = np.unique(multiplicities, axis=0, return_counts=True)
    draws = draws.astype(np.int64)
    fitted = len(distinct)
    work = len(alternatives.rows) * fitted
    worker_count = 1 if cpus <= 1 or fitted < 32 or work < _MINIMUM_POOL_WORK else min(cpus, fitted)
    # One fit per task, the leave-one-out fits first: fit costs vary, so small tasks keep the tail short. Support
    # counts are integer sums: neither order nor grouping changes them.
    chunk = 1 if worker_count > 1 else max(1, fitted)
    # Then the bootstrap vectors that omit the most units, in a fixed order.
    order = np.argsort(-np.count_nonzero(distinct == 0.0, axis=1), kind="stable")
    tasks: list[FitTask] = []
    if n_contigs > 1:
        for omitted in range(n_contigs):
            tasks.append((True, np.array([[omitted]], dtype=np.int64), np.ones(1, dtype=np.int64), config))
    tasks += [
        (
            False,
            np.ascontiguousarray(distinct[order[start : start + chunk]]),
            np.ascontiguousarray(draws[order[start : start + chunk]]),
            config,
        )
        for start in range(0, fitted, chunk)
    ]
    return ResamplingPlan(multiplicities, tuple(tasks), worker_count)


def evaluate_task(inputs: DecisionInputs, task: FitTask) -> FitResult:
    """One planned task: (is_loco, draws, selections of its multiplicity vectors)."""
    is_loco, weights, task_draws, task_config = task
    return is_loco, task_draws, evaluate(inputs, task_config, weights, omissions=is_loco)


def combine(inputs: DecisionInputs, planned: ResamplingPlan, results: Iterable[FitResult]) -> Resampling:
    """Bootstrap and LOBO support counts from every planned task's result (integer sums: order free)."""
    alternatives = inputs.alternatives
    n_samples = alternatives.n_samples
    edges = named_edges(alternatives.rows, n_samples)
    bootstrap = SupportCounts.zeros(len(alternatives.rows), n_samples, edges)
    loco = SupportCounts.zeros(len(alternatives.rows), n_samples, edges)
    fits = {False: 0, True: 0}
    restricted_fits = {False: 0, True: 0}
    received = 0
    for is_loco, task_draws, result in results:
        received += 1
        counts = loco if is_loco else bootstrap
        local_rows, graph_rows, local_states, restricted = result
        restricted_fits[is_loco] += int(np.sum(task_draws[restricted]))
        _accumulate(
            local_rows,
            graph_rows,
            local_states,
            task_draws,
            alternatives.rows,
            edges,
            counts.local_configuration,
            counts.graph_configuration,
            counts.local_state,
            counts.local_parent.values,
            counts.graph_parent.values,
        )
        fits[is_loco] += int(np.sum(task_draws))
    if received != len(planned.tasks):
        raise InvariantError(f"resampling received {received} task results for {len(planned.tasks)} planned tasks")
    bootstrap = replace(bootstrap, fits=fits[False])
    loco = replace(loco, fits=fits[True])
    return Resampling(
        bootstrap,
        loco,
        planned.multiplicities,
        planned.worker_count,
        bootstrap_order_restricted=restricted_fits[False],
        loco_order_restricted=restricted_fits[True],
    )
