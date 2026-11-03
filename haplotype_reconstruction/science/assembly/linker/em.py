"""The linker: dense haploid founder transitions between the input components of one joined batch (base
linking.py).

For every gap g = 1 .. G (G = n - 1, or the beam's maximum gap), the batch's blocks split into g independent
residue-class chains (blocks i, i + g, i + 2g, ...). Each chain is a hidden Markov model of a sample's ordered
founder pair: within a block the normal/error-tract scan (``linker.scan``), between consecutive blocks of the
chain both homologues move independently through one haploid transition matrix T_i (K_i x K_{i+g}). EM fits
every T_i of the gap from the posterior homologue edge counts (``linker.kernels``) with pseudocounts, a log
floor and a robust uniform mixture. The default takes full M-steps; configured damping uses a decaying rate.
Each gap is its own EM; all gaps of a batch share the blocks' emissions and cached zero-prior scans.

Inputs are the batch's input components (their linking proxies: at most ``LinkerParameters.max_sites`` sites per
component) and the informative samples' neutral float32 genotype likelihoods. A block's emission sites are its
proxy sites that are kept and called by at least one founder in the pre-fill calls; unknown alleles are
integrated predictively (``linker.emissions``).


Owns the linker em boundary (AJ07). Must not read files, clocks or
environment, allocate execution resources or publish diagnostics.
Method: docs/methods/assembly.md#linker.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Protocol, TypeVar

import numpy as np

from haplotype_reconstruction.contracts.assembly import JoinInputs
from haplotype_reconstruction.contracts.genetic_map import ChromosomeGeneticMap
from haplotype_reconstruction.contracts.roles import record_field
from haplotype_reconstruction.core import ties
from haplotype_reconstruction.science.shared.emissions import GENOTYPE_CODES, PREDICTIVE_CODES

if TYPE_CHECKING:
    from ..parameters import LinkerParameters
from .emissions import log_emissions, pair_codes
from .kernels import edge_log_counts, entrywise_log, propagate_pair_priors, regularised_rows
from .proxies import proxy_indices
from .scan import BlockScans

T = TypeVar("T")


class OrderedMap(Protocol):
    """Caller-owned ordered map; task inputs and results remain in fixed index order."""

    def __call__(self, function: Callable[[T], object], items: Sequence[T]) -> Sequence: ...


class RunTasks(Protocol):
    """Caller executes tasks on child thread teams and returns submission-order results."""

    def __call__(self, tasks: Sequence[Callable[[LinkerWorkspace], T]], workspace: LinkerWorkspace) -> Sequence[T]: ...


@dataclass(frozen=True)
class LinkerWorkspace:
    """Explicit execution hooks and diagnostic sink; default tasks run sequentially.

    A stage may supply an ordered map for block emissions and a task runner that
    enters each child thread team. lane_threads only returns the current team's
    lane-round width. Science never sets a Numba team or reserves resources.
    """

    map_tasks: OrderedMap | None = None
    run_tasks: RunTasks | None = None
    lane_threads: Callable[[], int] | None = None
    census: Callable[[ties.TieCensus], None] | None = None


def _run_tasks(tasks, workspace):
    return (
        tuple(task(workspace) for task in tasks)
        if workspace.run_tasks is None
        else workspace.run_tasks(tasks, workspace)
    )


@dataclass(frozen=True)
class LinkerConvergence:
    """Undamped residual of the returned iterate and exact termination status."""

    gap: int = record_field("state", dtype="int64")
    iterations: int = record_field("state", dtype="int64")
    max_iterations: int = record_field("state", dtype="int64")
    iteration_cap_reached: bool = record_field("state", dtype="bool")
    cap_hit: bool = record_field("state", dtype="bool")
    converged: bool = record_field("state", dtype="bool")
    stop_rule: str = record_field("state", dtype="str")
    tolerance: float = record_field("state", dtype="float64")
    last_damped_change: float | None = record_field("state", dtype="float64 | None")
    undamped_residual: float = record_field("state", dtype="float64")
    edge_residuals: tuple[float, ...] = record_field("state", dtype="tuple[float64,...]")
    residual_evaluations: int = record_field("state", dtype="int64")


@dataclass(frozen=True)
class Transitions:
    """The fitted forward transitions of one batch: what the endpoint beam reads.

    founders: int64[n] K_j, the founders of each of the batch's n input blocks.
    log_forward: for each fitted gap g = 1 .. G (index g - 1), for each block i = 0 .. n - 1 - g, the float64
        matrix [K_i, K_{i+g}] of log P(founder v at block i + g | founder u at block i): the natural log of the
        fitted probability. Rows sum to 1 in probability; every entry is finite (the uniform mixture keeps
        probabilities positive). Keep float64 precision through the endpoint beam's score accumulation.
    diagnostics: one plain convergence record per fitted gap, in gap order. The hierarchy persists these as
        linker_em search events; old records without the events have unmeasured convergence, not zero residual.
    """

    founders: np.ndarray = record_field("state", axes=("n",), dtype="int64")
    log_forward: tuple[tuple[np.ndarray, ...], ...] = record_field(
        "state", units="nats", dtype="tuple[tuple[float64[K_i,K_j],...],...]"
    )
    diagnostics: tuple[LinkerConvergence, ...] = record_field("state", dtype="tuple[LinkerConvergence,...]", default=())

    @property
    def blocks(self) -> int:
        """n, the batch's input blocks."""
        return int(len(self.founders))

    @property
    def gaps(self) -> int:
        """G, the largest fitted gap."""
        return len(self.log_forward)

    def matrix(self, past: int, current: int) -> np.ndarray | None:
        """The log transition matrix from block ``past`` to block ``current`` > past, or None when their gap was
        not fitted."""
        gap = current - past
        if 1 <= gap <= len(self.log_forward):
            return self.log_forward[gap - 1][past]
        return None


@dataclass(frozen=True)
class _BlockData:
    """One block's linker inputs: log emissions float32[S, E, 3|7] at its E emission sites, pair codes
    uint8[E, K * K], the sites' positions int64[E], founders K."""

    log_emissions: np.ndarray
    codes: np.ndarray
    positions: np.ndarray
    founders: int


def _block_data(inputs: JoinInputs, config: LinkerParameters, index: int) -> _BlockData:
    """Emission data of block ``index``: proxy sites kept and called by some founder (pre-fill calls)."""
    block = inputs.blocks[index]
    proxy = proxy_indices(block.inference, block.kept, config.max_sites)
    called = block.inference[:, proxy] >= 0
    sites = proxy[block.kept[proxy] & np.any(called, axis=0)]
    alleles = np.ascontiguousarray(block.inference[:, sites])
    # The prepared-panel row each founder copies at each emission site (copies of one row share an allele).
    spans = np.searchsorted(block.atomic_offsets, sites, side="right") - 1
    source_rows = np.ascontiguousarray(block.rows[:, spans])
    codes = pair_codes(alleles, source_rows)
    classes = GENOTYPE_CODES if np.all(alleles >= 0) else PREDICTIVE_CODES
    likelihoods = np.ascontiguousarray(inputs.gl[:, inputs.offsets[index] + sites])
    return _BlockData(
        log_emissions(likelihoods, float(config.robustness), classes),
        codes,
        np.ascontiguousarray(block.positions[sites], dtype=np.int64),
        int(block.founders),
    )


def _block_scans(
    data: _BlockData, genetic_map: ChromosomeGeneticMap, config: LinkerParameters, workspace: LinkerWorkspace
) -> BlockScans:
    morgans = np.ascontiguousarray(
        genetic_map.interval_morgans(data.positions[:-1], data.positions[1:]), dtype=np.float64
    )
    return BlockScans(
        data.log_emissions,
        data.codes,
        data.positions,
        data.founders,
        genetic_map.fallback_rate_per_bp,
        morgans,
        config.error_fraction,
        config.error_tract_bp,
    )


def _messages(scans, founders, log_forward, gap, workspace: LinkerWorkspace):
    """Forward scores S[i] (i < n - gap) and backward scores R[i] (i >= gap) of the gap's chains under the
    current log transitions; terminal messages the M-step does not read are not computed. The two recursions
    read only the fixed transitions and run concurrently on shares of the budget (one thread runs them in
    order); concurrent scans re-read their share between rounds of lane blocks, so the recursion still running
    takes up the threads of the one that finished."""
    n = len(scans)
    samples = scans[0].samples

    def forward(child):
        scores = [None] * n
        for i in range(n - gap):
            if i >= gap:
                priors = propagate_pair_priors(scores[i - gap], np.ascontiguousarray(log_forward[i - gap]))
            else:
                # Normalised initial pair prior of each independent chain.
                k = founders[i]
                priors = np.full((samples, k * k), -2.0 * math.log(k))
            scores[i] = scans[i].scan(priors, threads=child.lane_threads)
        return scores

    def backward(child):
        scores = [None] * n
        for i in range(n - 1, gap - 1, -1):
            priors = None
            if i < n - gap:
                # The transpose for the contraction, NOT a reverse conditional.
                priors = propagate_pair_priors(scores[i + gap], np.ascontiguousarray(log_forward[i].T))
            scores[i] = scans[i].scan(priors, backward=True, threads=child.lane_threads)
        return scores

    results = _run_tasks((forward, backward), workspace)
    return results[0], results[1]


def _m_step(forward_scores, backward_scores, log_forward, gap, config):
    """Undamped regularised forward transitions from posterior homologue edge counts.

    This is the fixed-point map M(T). Damping belongs to the iteration, not its convergence residual.
    """
    log_pseudo = math.log(config.pseudo_count)
    forward = []
    for i, log_t in enumerate(log_forward):
        log_counts = edge_log_counts(
            np.ascontiguousarray(forward_scores[i]),
            np.ascontiguousarray(backward_scores[i + gap]),
            np.ascontiguousarray(log_t),
        )
        forward.append(regularised_rows(log_counts, log_pseudo, config.min_log_probability, config.uniform_mix))
    return forward


def _largest_change(old, new):
    """Largest absolute change over the entries of corresponding tables (all finite: order-free)."""
    change = 0.0
    for before, after in zip(old, new, strict=False):
        if after.size:
            value = float(np.max(np.abs(after - before)))
            if value > change:
                change = value
    return change


def _fit_gap(scans, founders, gap, config: LinkerParameters, workspace: LinkerWorkspace, label=None):
    """Fit one gap; return forward probabilities per edge and the actual returned-iterate residual.

    Start from uniform transitions. At each iterate T, evaluate the regularised target M(T) once and stop
    when max(abs(M(T) - T)) <= min_change, or after max_iterations updates. Otherwise apply
    rate max(decay^iteration, minimum), which is 1 at the default decay of 1. Evaluating the final returned
    iterate avoids reporting the residual at the previous iterate as convergence. Reverse summaries are not
    parameters of this fixed-point problem.
    label: the batch's census label; record the last two residual-minus-tolerance distances when enabled.
    """
    n = len(scans)
    forward = [np.full((founders[i], founders[i + gap]), 1 / founders[i + gap]) for i in range(n - gap)]
    residual_history = []
    last_change = None
    for updates in range(config.max_iterations + 1):
        log_forward = [entrywise_log(table) for table in forward]
        forward_scores, backward_scores = _messages(scans, founders, log_forward, gap, workspace)
        target = _m_step(forward_scores, backward_scores, log_forward, gap, config)
        residuals = [_largest_change((table,), (value,)) for table, value in zip(forward, target, strict=False)]
        residual = max(residuals, default=0.0)
        residual_history.append(residual)
        converged = residual <= config.min_change
        if converged or updates == config.max_iterations:
            break
        learning_rate = max(config.learning_rate_decay**updates, config.minimum_learning_rate)
        new_forward = [
            (1.0 - learning_rate) * table + learning_rate * value for table, value in zip(forward, target, strict=False)
        ]
        last_change = _largest_change(forward, new_forward)
        forward = new_forward
    if label is not None and workspace.census is not None:
        ties.census(
            "assembly.linker_em_stop",
            f"{label}/g{gap}",
            [value - config.min_change for value in residual_history[-2:]],
            sink=workspace.census,
        )
    diagnostic = LinkerConvergence(
        gap=int(gap),
        iterations=updates,
        max_iterations=int(config.max_iterations),
        iteration_cap_reached=updates == config.max_iterations,
        cap_hit=not converged,
        converged=converged,
        stop_rule="undamped_forward_residual",
        tolerance=float(config.min_change),
        last_damped_change=last_change,
        undamped_residual=residual,
        edge_residuals=tuple(residuals),
        residual_evaluations=len(residual_history),
    )
    return forward, diagnostic


def fit_transitions(
    inputs: JoinInputs,
    genetic_map: ChromosomeGeneticMap,
    *,
    max_gap: int | None,
    config: LinkerParameters,
    workspace: LinkerWorkspace | None = None,
    label: str | None = None,
) -> Transitions:
    """Fit all requested forward transitions from immutable float32 batch evidence.

    max_gap=None fits every gap; otherwise fit through min(n-1,max_gap). Blocks
    must have at least one usable emission site. Returned log matrices are float64
    and diagnostics describe T, the returned iterate, rather than M(T). The stage
    supplies optional task hooks; their results must retain submission order.
    """
    workspace = LinkerWorkspace() if workspace is None else workspace
    n = len(inputs.blocks)
    last_gap = n - 1 if max_gap is None else min(n - 1, int(max_gap))
    founders = [int(block.founders) for block in inputs.blocks]
    operation = partial(_block_data, inputs, config)
    blocks = tuple(
        map(operation, range(n)) if workspace.map_tasks is None else workspace.map_tasks(operation, tuple(range(n)))
    )
    scans = tuple(_run_tasks(tuple(partial(_block_scans, data, genetic_map, config) for data in blocks), workspace))
    fits = tuple(partial(_fit_gap, scans, founders, gap, config, label=label) for gap in range(1, last_gap + 1))
    fitted = _run_tasks(fits, workspace)
    return Transitions(
        founders=np.asarray(founders, dtype=np.int64),
        log_forward=tuple(
            tuple(np.ascontiguousarray(entrywise_log(table), dtype=np.float64) for table in forward)
            for forward, diagnostic in fitted
        ),
        diagnostics=tuple(diagnostic for forward, diagnostic in fitted),
    )


__all__ = ["fit_transitions", "Transitions", "LinkerConvergence", "LinkerWorkspace"]
