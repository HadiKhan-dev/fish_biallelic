"""The linker: dense haploid founder transitions between the input components of one joined batch (base
linking.py).

For every gap g = 1 .. G (G = n - 1, or the beam's maximum gap), the batch's blocks split into g independent
residue-class chains (blocks i, i + g, i + 2g, ...). Each chain is a hidden Markov model of a sample's ordered
founder pair: within a block the normal/error-tract scan (``linker.scan``), between consecutive blocks of the
chain both homologues move independently through one haploid transition matrix T_i (K_i x K_{i+g}). EM fits
every T_i of the gap from the posterior homologue edge counts (``linker.kernels``) with pseudocounts, a log
floor and a robust uniform mixture, damping each update toward the previous transitions at a decaying rate.
Each gap is its own EM; all gaps of a batch share the blocks' emissions and cached zero-prior scans.

Inputs are the batch's input components (their linking proxies: at most ``LinkerConfig.max_sites`` sites per
component) and the informative samples' neutral float32 genotype likelihoods. A block's emission sites are its
proxy sites that are kept and called by at least one founder in the pre-fill calls; unknown alleles are
integrated predictively (``linker.emissions``).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math

import numba
import numpy as np

from ...execution.memory import malloc_trim
from ...maths import ties
from ...maths.emissions import GENOTYPE_CODES, PREDICTIVE_CODES
from ..config import LinkerConfig
from ..join.inputs import JoinInputs
from .emissions import log_emissions, pair_codes
from .kernels import edge_log_counts, entrywise_log, propagate_pair_priors, regularised_rows
from .proxies import proxy_indices
from .scan import BlockScans


@dataclass(frozen=True)
class Transitions:
    """The fitted forward transitions of one batch: what the endpoint beam reads.

    founders: int64[n] K_j, the founders of each of the batch's n input blocks.
    log_forward: for each fitted gap g = 1 .. G (index g - 1), for each block i = 0 .. n - 1 - g, the float64
        matrix [K_i, K_{i+g}] of log P(founder v at block i + g | founder u at block i): the natural log of the
        fitted probability. Rows sum to 1 in probability; every entry is finite (the uniform mixture keeps
        probabilities positive). Release 2.1 keeps float64 (base FastMesh rounded to float32, so a one-ulp change
        of one entry under a tiny input perturbation reordered the endpoint beam; design 5.4, review r8).
    diagnostics: one plain convergence record per fitted gap, in gap order. The hierarchy persists these as
        linker_em search events; old records without the events have unmeasured convergence, not zero residual.
    """

    founders: np.ndarray
    log_forward: tuple[tuple[np.ndarray, ...], ...]
    diagnostics: tuple[dict, ...] = ()

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


def _apply_threads(budget) -> int:
    """Re-read the live budget and size the calling thread's Numba team (capped by Numba's pool size)."""
    threads = max(1, min(int(budget.threads()), numba.config.NUMBA_NUM_THREADS))
    numba.set_num_threads(threads)
    return threads


def _block_data(inputs: JoinInputs, config: LinkerConfig, index: int) -> _BlockData:
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
    return _BlockData(log_emissions(likelihoods, float(config.robustness), classes), codes,
                      np.ascontiguousarray(block.positions[sites], dtype=np.int64), int(block.founders))


def _block_scans(data: _BlockData, genetic_map, config: LinkerConfig, budget) -> BlockScans:
    _apply_threads(budget)
    morgans = None
    if genetic_map.has_map:
        morgans = np.ascontiguousarray(genetic_map.interval_morgans(data.positions[:-1], data.positions[1:]),
                                       dtype=np.float64)
    return BlockScans(data.log_emissions, data.codes, data.positions, data.founders,
                      genetic_map.fallback_rate_per_bp, morgans, config.error_fraction, config.error_tract_bp)


def _messages(scans, founders, log_forward, gap, budget):
    """Forward scores S[i] (i < n - gap) and backward scores R[i] (i >= gap) of the gap's chains under the
    current log transitions; terminal messages the M-step does not read are not computed. The two recursions
    read only the fixed transitions and run concurrently on shares of the budget (one thread runs them in
    order); concurrent scans re-read their share between rounds of lane blocks, so the recursion still running
    takes up the threads of the one that finished."""
    n = len(scans)
    samples = scans[0].samples

    def forward(child, rounds=None):
        scores = [None] * n
        for i in range(n - gap):
            _apply_threads(child)
            if i >= gap:
                priors = propagate_pair_priors(scores[i - gap], np.ascontiguousarray(log_forward[i - gap]))
            else:
                # Normalised initial pair prior of each independent chain.
                k = founders[i]
                priors = np.full((samples, k * k), -2.0 * math.log(k))
            scores[i] = scans[i].scan(priors, threads=rounds)
        return scores

    def backward(child, rounds=None):
        scores = [None] * n
        for i in range(n - 1, gap - 1, -1):
            _apply_threads(child)
            priors = None
            if i < n - gap:
                # The transpose for the contraction, NOT a reverse conditional.
                priors = propagate_pair_priors(scores[i + gap], np.ascontiguousarray(log_forward[i].T))
            scores[i] = scans[i].scan(priors, backward=True, threads=rounds)
        return scores

    if budget.threads() < 2:
        # One thread runs the recursions in order (no helper threads: this runs every EM iteration).
        return forward(budget), backward(budget)
    tasks = (lambda child: forward(child, partial(_apply_threads, child)),
             lambda child: backward(child, partial(_apply_threads, child)))
    # No reservation: the join batch's reservation (hierarchy.level.batch_memory_bytes) covers the linker.
    results = dict(budget.concurrent(tasks, 0, label="linker messages"))
    return results[0], results[1]


def _m_step(forward_scores, backward_scores, log_forward, gap, config, budget):
    """Undamped regularised forward transitions from posterior homologue edge counts.

    This is the fixed-point map M(T). Damping belongs to the iteration, not its convergence residual.
    """
    log_pseudo = math.log(config.pseudo_count)
    forward = []
    for i, log_t in enumerate(log_forward):
        _apply_threads(budget)
        log_counts = edge_log_counts(np.ascontiguousarray(forward_scores[i]),
                                     np.ascontiguousarray(backward_scores[i + gap]), np.ascontiguousarray(log_t))
        forward.append(regularised_rows(log_counts, log_pseudo, config.min_log_probability, config.uniform_mix))
    return forward


def _largest_change(old, new):
    """Largest absolute change over the entries of corresponding tables (all finite: order-free)."""
    change = 0.0
    for before, after in zip(old, new):
        if after.size:
            value = float(np.max(np.abs(after - before)))
            if value > change:
                change = value
    return change


def _fit_gap(scans, founders, gap, config: LinkerConfig, budget, label=None):
    """Fit one gap; return forward probabilities per edge and the actual returned-iterate residual.

    Start from uniform transitions. At each iterate T, evaluate the regularised target M(T) once and stop
    when max(abs(M(T) - T)) <= min_change, or after max_iterations damped updates. Otherwise apply the same
    rate max(decay^iteration, minimum). Evaluating the final returned iterate avoids reporting the residual
    at the previous iterate as convergence. Reverse summaries are not parameters of this fixed-point problem.
    label: the batch's census label; record the last two residual-minus-tolerance distances when enabled.
    """
    n = len(scans)
    forward = [np.full((founders[i], founders[i + gap]), 1 / founders[i + gap]) for i in range(n - gap)]
    residual_history = []
    last_change = None
    for updates in range(config.max_iterations + 1):
        _apply_threads(budget)
        log_forward = [entrywise_log(table) for table in forward]
        forward_scores, backward_scores = _messages(scans, founders, log_forward, gap, budget)
        target = _m_step(forward_scores, backward_scores, log_forward, gap, config, budget)
        residuals = [_largest_change((table,), (value,)) for table, value in zip(forward, target)]
        residual = max(residuals, default=0.0)
        residual_history.append(residual)
        converged = residual <= config.min_change
        if converged or updates == config.max_iterations:
            break
        learning_rate = max(config.learning_rate_decay ** updates, config.minimum_learning_rate)
        new_forward = [(1.0 - learning_rate) * table + learning_rate * value
                       for table, value in zip(forward, target)]
        last_change = _largest_change(forward, new_forward)
        forward = new_forward
    if label is not None and ties.census_enabled():
        ties.census("assembly.linker_em_stop", f"{label}/g{gap}",
                    [value - config.min_change for value in residual_history[-2:]])
    diagnostic = dict(gap=int(gap), iterations=updates, max_iterations=int(config.max_iterations),
                      iteration_cap_reached=updates == config.max_iterations,
                      cap_hit=not converged, converged=converged,
                      stop_rule="undamped_forward_residual", tolerance=float(config.min_change),
                      last_damped_change=last_change, undamped_residual=residual, edge_residuals=residuals,
                      residual_evaluations=len(residual_history))
    return forward, diagnostic


def fit_transitions(inputs: JoinInputs, genetic_map, *, max_gap: int | None, config: LinkerConfig,
                    budget, label: str | None = None) -> Transitions:
    """Fit the forward transitions of every gap 1 .. min(n - 1, max_gap) of one batch (max_gap None: all).

    inputs: the batch (n >= 2 input components, informative samples' float32 neutral likelihoods).
    genetic_map: the contig's ``model.genetic_map.ChromosomeGeneticMap``: within-block switch probabilities use
        its interval distances when it has a map, else max(1, bp) x ``fallback_rate_per_bp``.
    budget: the batch's CPU budget (re-read at every block, edge and EM iteration). Block emissions run one
        Numba thread per block concurrently; block scans, gap fits and each fit's forward/backward recursions
        run concurrently with shares of the live budget. Results do not depend on the thread counts.
    label: the batch's census label (None: no census).
    """
    n = len(inputs.blocks)
    last_gap = n - 1 if max_gap is None else min(n - 1, int(max_gap))
    founders = [int(block.founders) for block in inputs.blocks]
    # The linker's arrays (emission data, block scans, gap fits and their messages) are inside the join
    # batch's reservation (hierarchy.level.batch_memory_bytes), so its concurrent steps reserve nothing more.
    blocks = budget.thread_map(partial(_block_data, inputs, config), range(n), label="linker emissions")
    malloc_trim()
    scans = [None] * n
    tasks = [partial(_block_scans, data, genetic_map, config) for data in blocks]
    for index, block_scans in budget.concurrent(tasks, 0, label="linker scans"):
        scans[index] = block_scans
    del blocks, tasks
    fits = [partial(_fit_gap, scans, founders, gap, config, label=label) for gap in range(1, last_gap + 1)]
    fitted = [None] * len(fits)
    diagnostics = [None] * len(fits)
    for index, (forward, diagnostic) in budget.concurrent(fits, 0, label="linker fits"):
        fitted[index] = forward
        diagnostics[index] = diagnostic
    malloc_trim()
    return Transitions(
        founders=np.asarray(founders, dtype=np.int64),
        log_forward=tuple(tuple(np.ascontiguousarray(entrywise_log(table), dtype=np.float64) for table in forward)
                          for forward in fitted),
        diagnostics=tuple(diagnostics))
