"""Chromosome bootstrap and leave-one-chromosome-out support of the full-data decisions.

Each replicate refits everything downstream of the genetic scores (orientation, exposure gate,
family messages, state, identity, graph) with chromosome multiplicities: seeded bootstrap draws of
C chromosomes with replacement, and C leave-one-out weightings (separate counts and denominator).
A replicate whose final family solve does not converge abstains for every sample and stays in the
denominator. Counts are integer sums, so support does not depend on worker count or order. A fit is a
deterministic function of its multiplicities, so each distinct bootstrap multiplicity vector is fitted once and
counted as many times as it was drawn (few chromosomes repeat most draws: 10 distinct vectors among 1,000 at C = 3).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from numba import njit

from ..execution import BudgetedPool
from . import selection
from .alternatives import Alternatives
from .config import PedigreeConfig, TierThresholds
from .family import FamilyMessageNonConvergence
from .orientation import JunctionEvidence
from .records import TierReach
from .selection import DecisionInputs, ExposureEvidence

# A pool is started only for at least this many (alternatives x replicates) selection scans.
_MINIMUM_POOL_WORK = 100_000


@dataclass(frozen=True)
class EdgeCounts:
    """Integer counts per (child, parent) edge named by some configuration row, read as ``counts[child, parent]``
    (0 for an edge no row names). keys int64 [E] sorted child * N + parent; values int64 [E] (release 2.16: per
    edge, not a dense [N, N] array)."""

    keys: np.ndarray
    values: np.ndarray
    samples: int

    def __getitem__(self, edge) -> int:
        child, parent = edge
        key = int(child) * self.samples + int(parent)
        index = int(np.searchsorted(self.keys, key))
        return int(self.values[index]) if index < len(self.keys) and self.keys[index] == key else 0


def named_edges(rows: np.ndarray, samples: int) -> np.ndarray:
    """int64 [E] sorted keys child * N + parent of the edges the configuration rows name."""
    keys = np.concatenate([rows[:, 0] * samples + rows[:, slot] for slot in (1, 2)])
    named = np.concatenate([rows[:, slot] >= 0 for slot in (1, 2)])
    return np.unique(keys[named]).astype(np.int64)


@dataclass(frozen=True)
class SupportCounts:
    """Integer counts over resampled fits.

    local_configuration / graph_configuration int64 [A]: fits whose local / graph row is the row.
    local_state int64 [N, 3]: fits calling each marginal state.
    local_parent / graph_parent EdgeCounts: fits whose local / graph row names [child, parent].
    fits: number of fits (the denominator); nonconverged: fits that abstained.
    """

    local_configuration: np.ndarray
    graph_configuration: np.ndarray
    local_state: np.ndarray
    local_parent: EdgeCounts
    graph_parent: EdgeCounts
    fits: int
    nonconverged: int

    @classmethod
    def zeros(cls, n_alternatives: int, n_samples: int, edges: np.ndarray) -> "SupportCounts":
        return cls(np.zeros(n_alternatives, np.int64), np.zeros(n_alternatives, np.int64),
                   np.zeros((n_samples, 3), np.int64),
                   EdgeCounts(edges, np.zeros(len(edges), np.int64), n_samples),
                   EdgeCounts(edges, np.zeros(len(edges), np.int64), n_samples), 0, 0)

    def fraction(self, count: int) -> float:
        """count / fits; 0 without fits."""
        return float(count / self.fits) if self.fits > 0 else 0.0


@dataclass(frozen=True)
class Resampling:
    bootstrap: SupportCounts
    loco: SupportCounts
    multiplicities: np.ndarray       # float64 [R, C] bootstrap chromosome multiplicities
    worker_count: int


@njit(cache=True, nogil=True)
def _accumulate(local_rows, graph_rows, local_states, draws, rows, edges, local_configuration,
                graph_configuration, local_state, local_parent, graph_parent):
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


def _evaluate(inputs: DecisionInputs, config: PedigreeConfig, multiplicities: np.ndarray):
    """Selections of a chunk of multiplicity vectors: local rows, graph rows, local states and the
    nonconverged flag of each vector."""
    n_replicates = len(multiplicities)
    n_samples = inputs.alternatives.n_samples
    local_rows = np.full((n_replicates, n_samples), -1, dtype=np.int64)
    graph_rows = np.full((n_replicates, n_samples), -1, dtype=np.int64)
    local_states = np.full((n_replicates, n_samples), -1, dtype=np.int8)
    nonconverged = np.zeros(n_replicates, dtype=np.bool_)
    for replicate, weights in enumerate(multiplicities):
        try:
            chosen, _, _ = selection.fit(weights, inputs, config)
        except FamilyMessageNonConvergence:
            nonconverged[replicate] = True
            continue
        local_states[replicate] = chosen.local_state
        local_rows[replicate] = chosen.local_row
        graph_rows[replicate] = chosen.graph_row
    return local_rows, graph_rows, local_states, nonconverged


# --------------------------------------------------------------------- shared-memory transport
_ARRAY_FIELDS = {
    "rows": ("alternatives", "rows"), "states": ("alternatives", "states"),
    "m0_row": ("alternatives", "m0_row"), "panel_row": ("alternatives", "panel_row"),
    "log_likelihoods": ("alternatives", "log_likelihoods"),
    "full_counts": ("alternatives", "full_counts"), "scored_counts": ("alternatives", "scored_counts"),
    "edge_exposed": ("exposure", "edge_exposed"), "pair_exposed": ("exposure", "pair_exposed"),
    "total_bins": ("exposure", "total_bins"), "edge_present": ("exposure", "edge_present"),
    "pair_present": ("exposure", "pair_present"), "row_edges": ("exposure", "row_edges"),
    "junctions": ("junctions", "junctions"), "callable_bins": ("junctions", "callable_bins"),
    "junction_total_bins": ("junctions", "total_bins"),
    "pair_child": ("junctions", "pair_child"), "pair_parent": ("junctions", "pair_parent"),
    "pair_log_odds": ("junctions", "pair_log_odds"), "compared": ("junctions", "compared"),
}


def _shared_arrays(inputs: DecisionInputs) -> dict[str, np.ndarray]:
    arrays = {name: getattr(getattr(inputs, group), field)
              for name, (group, field) in _ARRAY_FIELDS.items()}
    lengths = np.asarray([len(rows) for rows in inputs.alternatives.child_rows], dtype=np.int64)
    arrays["child_offsets"] = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    arrays["chronology"] = inputs.chronology
    return arrays


def _inputs_from_shared(shared) -> DecisionInputs:
    offsets = shared["child_offsets"]
    child_rows = tuple(np.arange(offsets[i], offsets[i + 1], dtype=np.int64)
                       for i in range(len(offsets) - 1))
    alternatives = Alternatives(shared["rows"], shared["states"], child_rows, shared["m0_row"],
                                shared["panel_row"], shared["log_likelihoods"],
                                shared["full_counts"], shared["scored_counts"])
    exposure = ExposureEvidence(shared["edge_exposed"], shared["pair_exposed"],
                                shared["total_bins"], shared["edge_present"],
                                shared["pair_present"], shared["row_edges"])
    junctions = JunctionEvidence(shared["junctions"], shared["callable_bins"],
                                 shared["junction_total_bins"], shared["pair_child"], shared["pair_parent"],
                                 shared["pair_log_odds"],
                                 shared["compared"])
    return DecisionInputs(alternatives, exposure, junctions, shared["chronology"])


def _pool_task(task, shared, budget):
    """Pool worker: evaluate one tagged chunk of multiplicity vectors on the shared inputs."""
    is_loco, multiplicities, draws, config = task
    return is_loco, draws, _evaluate(_inputs_from_shared(shared), config, multiplicities)


# --------------------------------------------------------------------- driver
def bootstrap_multiplicities(n_contigs: int, replicates: int, seed: int) -> np.ndarray:
    """float64 [R, C]: chromosome draw counts of each seeded bootstrap replicate."""
    rng = np.random.default_rng(seed)
    multiplicities = np.empty((replicates, n_contigs), dtype=np.float64)
    for replicate in range(replicates):
        draws = rng.integers(0, n_contigs, size=n_contigs)
        multiplicities[replicate] = np.bincount(draws, minlength=n_contigs).astype(np.float64)
    return multiplicities


def run(inputs: DecisionInputs, config: PedigreeConfig, budget) -> Resampling:
    """Bootstrap and LOCO support counts of every configuration, state and edge."""
    alternatives = inputs.alternatives
    n_contigs = alternatives.log_likelihoods.shape[0]
    n_samples = alternatives.n_samples
    replicates = config.resampling.bootstrap_replicates
    multiplicities = bootstrap_multiplicities(n_contigs, replicates, config.resampling.bootstrap_seed)
    # Distinct vectors (each fitted once) and how many replicates drew each.
    distinct, draws = np.unique(multiplicities, axis=0, return_counts=True)
    draws = draws.astype(np.int64)
    fitted = len(distinct)
    cpus = budget.threads()
    work = len(alternatives.rows) * fitted
    worker_count = (1 if cpus <= 1 or fitted < 32 or work < _MINIMUM_POOL_WORK
                    else min(cpus, fitted))
    chunk = max(1, int(math.ceil(fitted / float(worker_count * 4))))
    tasks = [(False, np.ascontiguousarray(distinct[start:start + chunk]),
              np.ascontiguousarray(draws[start:start + chunk]), config)
             for start in range(0, fitted, chunk)]
    if n_contigs > 1:
        for omitted in range(n_contigs):
            weights = np.ones((1, n_contigs), dtype=np.float64)
            weights[0, omitted] = 0.0
            tasks.append((True, weights, np.ones(1, dtype=np.int64), config))
    edges = named_edges(alternatives.rows, n_samples)
    bootstrap = SupportCounts.zeros(len(alternatives.rows), n_samples, edges)
    loco = SupportCounts.zeros(len(alternatives.rows), n_samples, edges)
    tallies = {False: [0, 0], True: [0, 0]}     # fits, nonconverged

    def accumulate(is_loco, task_draws, result):
        counts = loco if is_loco else bootstrap
        local_rows, graph_rows, local_states, failed = result
        _accumulate(local_rows, graph_rows, local_states, task_draws, alternatives.rows, edges,
                    counts.local_configuration, counts.graph_configuration, counts.local_state,
                    counts.local_parent.values, counts.graph_parent.values)
        tallies[is_loco][0] += int(np.sum(task_draws))
        tallies[is_loco][1] += int(np.sum(task_draws[failed]))

    if worker_count == 1:
        for is_loco, weights, task_draws, task_config in tasks:
            accumulate(is_loco, task_draws, _evaluate(inputs, task_config, weights))
    else:
        with BudgetedPool(budget, processes=worker_count, shared=_shared_arrays(inputs),
                          label="pedigree resampling") as pool:
            for is_loco, task_draws, result in pool.imap_unordered(_pool_task, tasks, chunksize=1):
                accumulate(is_loco, task_draws, result)
    bootstrap = replace(bootstrap, fits=tallies[False][0], nonconverged=tallies[False][1])
    loco = replace(loco, fits=tallies[True][0], nonconverged=tallies[True][1])
    return Resampling(bootstrap, loco, multiplicities, worker_count)


# --------------------------------------------------------------------- chromosome-count reach
def _distinct_probability(n_contigs: int, minimum: int) -> float:
    """P(at least ``minimum`` distinct chromosomes among C draws with replacement from C)."""
    total = 0
    for distinct in range(minimum, n_contigs + 1):
        surjections = sum((-1) ** j * math.comb(distinct, j) * (distinct - j) ** n_contigs
                          for j in range(distinct + 1))
        total += math.comb(n_contigs, distinct) * surjections
    return total / n_contigs ** n_contigs


def minimum_contigs_for_tier(thresholds: TierThresholds, minimum_exposed_contigs: int) -> int:
    """Smallest informative-chromosome count whose resampling can reach ``thresholds``.

    A non-M0 configuration is testable only on >= ``minimum_exposed_contigs`` distinct exposed
    chromosomes; in fits with fewer, every state ties and nothing is called. So the bootstrap
    support of any call is at most P(distinct draws >= m) and LOCO support is 0 unless C - 1 >= m.
    """
    n_contigs = max(1, minimum_exposed_contigs)
    while True:
        loco = 1.0 if n_contigs - 1 >= minimum_exposed_contigs else 0.0
        if (loco >= thresholds.loco_fraction and _distinct_probability(
                n_contigs, minimum_exposed_contigs) >= thresholds.configuration_bootstrap):
            return n_contigs
        n_contigs += 1


def reach(multiplicities: np.ndarray, config: PedigreeConfig) -> TierReach:
    """What this run's resampling can release with its informative chromosomes."""
    n_contigs = multiplicities.shape[1]
    minimum = config.exposure.minimum_exposed_contigs
    distinct = np.count_nonzero(multiplicities > 0, axis=1)
    max_bootstrap = float(np.mean(distinct >= minimum)) if len(distinct) else 0.0
    max_loco = 1.0 if n_contigs > 1 and n_contigs - 1 >= minimum else 0.0

    def reachable(thresholds: TierThresholds) -> bool:
        return (max_bootstrap >= thresholds.configuration_bootstrap
                and max_loco >= thresholds.loco_fraction)

    tier_a = reachable(config.release.tier_a)
    tier_b = reachable(config.release.tier_b)
    need_a = minimum_contigs_for_tier(config.release.tier_a, minimum)
    need_b = minimum_contigs_for_tier(config.release.tier_b, minimum)
    support = f"max bootstrap support {max_bootstrap:.3f}, LOCO {max_loco:.2f}"
    if tier_b:
        statement = (f"{n_contigs} informative chromosomes: Tier B assessable ({support}); "
                     + ("Tier A assessable" if tier_a else f"Tier A needs >= {need_a}"))
    else:
        statement = (f"{n_contigs} informative chromosomes: not assessable ({support}); "
                     f"Tier B needs >= {need_b} informative chromosomes, Tier A >= {need_a}")
    return TierReach(n_contigs, minimum, max_bootstrap, max_loco, tier_a, tier_b, need_a, need_b,
                     statement)
