"""Chromosome bootstrap and leave-one-chromosome-out support of the full-data decisions.

Each replicate refits everything downstream of the genetic scores (orientation, exposure gate,
family messages, state, identity, graph) with chromosome multiplicities: seeded bootstrap draws of
C chromosomes with replacement, and C leave-one-out weightings (separate counts and denominator).
A replicate whose final family solve does not converge abstains for every sample and stays in the
denominator. Counts are integer sums, so support does not depend on worker count or order. A fit is a
deterministic function of its multiplicities, so each distinct bootstrap multiplicity vector is fitted once and
counted as many times as it was drawn (few chromosomes repeat most draws: 10 distinct vectors among 1,000 at C = 3).

Method: docs/methods/pedigree.md#chromosome-resampling.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace

import numpy as np

from ...contracts.pedigree import Resampling, SupportCounts, TierReach
from ...core.errors import InvariantError
from ...core.kernels import kernel
from ...core.numerics import ordered_sum
from ...core.random import bootstrap_multiplicities
from . import selection
from .family import FamilyMessageNonConvergence
from .kernels.signatures import OVERLOADS
from .parameters import PedigreeParameters, TierThresholds
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
    "minimum_contigs_for_tier",
    "reach",
]

# Work and results keep the existing coalesced task order and integer draw counts.
FitTask = tuple[bool, np.ndarray, np.ndarray, PedigreeParameters]
FitArrays = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
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


def evaluate(inputs: DecisionInputs, config: PedigreeParameters, multiplicities: np.ndarray, *, omissions=False) -> FitArrays:
    """Selections of a chunk of multiplicity vectors: local rows, graph rows, local states and the
    nonconverged flag of each vector."""
    n_replicates = len(multiplicities)
    n_samples = inputs.alternatives.n_samples
    local_rows = np.full((n_replicates, n_samples), -1, dtype=np.int64)
    graph_rows = np.full((n_replicates, n_samples), -1, dtype=np.int64)
    local_states = np.full((n_replicates, n_samples), -1, dtype=np.int8)
    nonconverged = np.zeros(n_replicates, dtype=np.bool_)
    graph_passes = np.zeros(n_replicates, dtype=np.int64)
    graph_exhausted = np.zeros(n_replicates, dtype=np.bool_)
    for replicate, weights in enumerate(multiplicities):
        try:
            chosen, _, _ = (selection.fit(np.empty(0), inputs, config, omitted=int(weights[0]))
                             if omissions else selection.fit(weights, inputs, config))
        except FamilyMessageNonConvergence:
            nonconverged[replicate] = True
            continue
        local_states[replicate] = chosen.local_state
        local_rows[replicate] = chosen.local_row
        graph_rows[replicate] = chosen.graph_row
        graph_passes[replicate] = chosen.graph_passes_used
        graph_exhausted[replicate] = chosen.graph_pass_budget_exhausted
    return local_rows, graph_rows, local_states, nonconverged, graph_passes, graph_exhausted


# --------------------------------------------------------------------- driver


@dataclass(frozen=True)
class ResamplingPlan:
    """The fixed resampling work of one decision: the seeded draws and the ordered independent fit tasks.

    multiplicities: float64 [R, C] the R bootstrap draws (PCG64 stream of the configured seed).
    tasks: the fit tasks in their fixed order: the C leave-one-out fits first, then the distinct bootstrap
    vectors that omit the most chromosomes (one vector per task when the work is spread over workers).
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
    # One fit per task, the leave-one-out fits (the slowest) first: fit costs vary (slow-converging family
    # solves), so small tasks keep the tail short. Support counts are integer sums: neither order nor grouping
    # changes them.
    chunk = 1 if worker_count > 1 else max(1, fitted)
    # Then the bootstrap vectors that omit the most chromosomes (weaker evidence, typically the slowest
    # family solves), so that long fits start early rather than in the tail.
    order = np.argsort(-np.count_nonzero(distinct == 0.0, axis=1), kind="stable")
    tasks: list[FitTask] = []
    if n_contigs > 1:
        for omitted in range(n_contigs):
            if inputs.regime == "low_material":
                weights = np.array([[omitted]], dtype=np.int64)
            else:
                weights = np.ones((1, n_contigs), dtype=np.float64)
                weights[0, omitted] = 0.0
            tasks.append((True, weights, np.ones(1, dtype=np.int64), config))
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
    return is_loco, task_draws, evaluate(inputs, task_config, weights,
        omissions=is_loco and inputs.regime == "low_material")


def combine(inputs: DecisionInputs, planned: ResamplingPlan, results: Iterable[FitResult]) -> Resampling:
    """Bootstrap and LOCO support counts from every planned task's result (integer sums: order free)."""
    alternatives = inputs.alternatives
    n_samples = alternatives.n_samples
    edges = named_edges(alternatives.rows, n_samples)
    bootstrap = SupportCounts.zeros(len(alternatives.rows), n_samples, edges)
    loco = SupportCounts.zeros(len(alternatives.rows), n_samples, edges)
    tallies = {False: [0, 0], True: [0, 0]}  # fits, nonconverged
    graph_tallies = {False: 0, True: 0}
    maximum_graph_passes = 0
    received = 0
    for is_loco, task_draws, result in results:
        received += 1
        counts = loco if is_loco else bootstrap
        local_rows, graph_rows, local_states, failed, graph_passes, graph_exhausted = result
        graph_tallies[is_loco] += int(np.sum(task_draws[graph_exhausted]))
        maximum_graph_passes = max(maximum_graph_passes, int(np.max(graph_passes, initial=0)))
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
        tallies[is_loco][0] += int(np.sum(task_draws))
        tallies[is_loco][1] += int(np.sum(task_draws[failed]))
    if received != len(planned.tasks):
        raise InvariantError(f"resampling received {received} task results for {len(planned.tasks)} planned tasks")
    bootstrap = replace(bootstrap, fits=tallies[False][0], nonconverged=tallies[False][1])
    loco = replace(loco, fits=tallies[True][0], nonconverged=tallies[True][1])
    return Resampling(
        bootstrap,
        loco,
        planned.multiplicities,
        planned.worker_count,
        bootstrap_graph_budget_exhausted=graph_tallies[False],
        loco_graph_budget_exhausted=graph_tallies[True],
        maximum_graph_passes_used=maximum_graph_passes,
    )


# --------------------------------------------------------------------- chromosome-count reach
def _distinct_probability(n_contigs: int, minimum: int) -> float:
    """P(at least ``minimum`` distinct chromosomes among C draws with replacement from C)."""
    total = 0
    for distinct in range(minimum, n_contigs + 1):
        surjections = sum((-1) ** j * math.comb(distinct, j) * (distinct - j) ** n_contigs for j in range(distinct + 1))
        total += math.comb(n_contigs, distinct) * surjections
    return total / n_contigs**n_contigs


def minimum_contigs_for_tier(thresholds: TierThresholds, minimum_exposed_contigs: int) -> int:
    """Smallest count of informative chromosomes shorter than one map unit whose resampling can reach ``thresholds``
    (longer chromosomes count more units and need fewer; ``reach`` is exact for a run).

    A non-M0 configuration is testable only on exposed chromosomes holding >= m map units; with chromosomes of one
    unit each, fits with fewer than m distinct chromosomes tie every state and call nothing. So the bootstrap
    support of any call is at most P(distinct draws >= m) and LOCO support is 0 unless C - 1 >= m.
    """
    n_contigs = max(1, minimum_exposed_contigs)
    while True:
        loco = 1.0 if n_contigs - 1 >= minimum_exposed_contigs else 0.0
        if (
            loco >= thresholds.loco_fraction
            and _distinct_probability(n_contigs, minimum_exposed_contigs) >= thresholds.configuration_bootstrap
        ):
            return n_contigs
        n_contigs += 1


def reach(multiplicities: np.ndarray, units: np.ndarray, config: PedigreeParameters) -> TierReach:
    """What this run's resampling can release with its informative chromosomes of ``units`` map units each: the
    share of bootstrap draws, and of leave-one-out fits, whose distinct chromosomes hold the gate's map units."""
    n_contigs = multiplicities.shape[1]
    units = np.asarray(units, dtype=np.float64)
    minimum_units = config.exposure.minimum_exposed_units
    minimum = int(np.ceil(minimum_units))
    drawn = ordered_sum((multiplicities > 0).T, axis=0, weights=units) if len(multiplicities) else np.zeros(0)
    max_bootstrap = float(np.mean(drawn >= minimum_units)) if len(drawn) else 0.0
    left = ordered_sum(units) - units
    max_loco = float(np.mean(left >= minimum_units)) if n_contigs > 1 else 0.0

    def reachable(thresholds: TierThresholds) -> bool:
        return max_bootstrap >= thresholds.configuration_bootstrap and max_loco >= thresholds.loco_fraction

    tier_a = reachable(config.release.tier_a)
    tier_b = reachable(config.release.tier_b)
    need_a = minimum_contigs_for_tier(config.release.tier_a, minimum)
    need_b = minimum_contigs_for_tier(config.release.tier_b, minimum)
    support = f"max bootstrap support {max_bootstrap:.3f}, LOCO {max_loco:.2f}"
    size = (
        f"{n_contigs} informative chromosomes holding {float(ordered_sum(units)):.1f} map units of "
        f"{config.exposure.unit_morgans:g} Morgan"
    )
    short = f"chromosomes shorter than {config.exposure.unit_morgans:g} Morgan, fewer longer ones"
    if tier_b:
        statement = f"{size}: Tier B assessable ({support}); " + (
            "Tier A assessable" if tier_a else f"Tier A needs >= {need_a} {short}"
        )
    else:
        statement = f"{size}: not assessable ({support}); Tier B needs >= {need_b} {short}, Tier A >= {need_a}"
    return TierReach(
        n_contigs,
        minimum,
        max_bootstrap,
        max_loco,
        tier_a,
        tier_b,
        need_a,
        need_b,
        statement,
        informative_units=float(ordered_sum(units)),
        unit_morgans=float(config.exposure.unit_morgans),
    )
