"""Founder completion: local panels -> prepared panels (before the hierarchy).

Per block, a two-fold cross-fitted variational EM over diplotypes and missing founder alleles
gives held-out carrier profiles of the founder rows (``crossfit``); a whole-bin leave-out
("cavity") posterior proposes fills of unknown founder alleles (``cavity_fill``); per boundary,
accepted one-to-one class correspondences link rows of adjacent blocks (``links``); the occupancy
gate releases the fills of demonstrably occupied rows (``occupancy``). Released fills change only
the published calls (``PreparedPanel.released``, ALT probability = the fill posterior); every
likelihood downstream reads the frozen pre-fill snapshot (``PreparedPanel.inference``).

Execution: a forkserver pool (``execution.BudgetedPool``, one numeric thread per worker) runs the
block science in tasks of two consecutive blocks, then the boundary links in runs of eight. A task
attaches the contig's evidence view where it already lives in shared memory
(``AssemblyEvidence.shared``) and copies only its blocks' columns, so no chromosome-sized copy is
made; the coordinator computes the focal-informative mask in one parallel scan. Every task reserves
its estimated peak through the run's MemoryGovernor before it is sent (``block_memory_bytes``: the
cavity's per-bin pair emissions grow with N K^2), so tasks wait for memory instead of overcommitting
it. Results are stored by index, so nothing depends on completion order, pacing or the worker count.
The pool is released in the background while the coordinator runs the gate.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import threading
from typing import Sequence

import numpy as np
from numba import njit, prange

from ...execution.budget import as_budget, numba_threads
from ...execution.chromosome_workers import memory_worker_limit
from ...execution.memory import reserve_memory
from ...execution.pools import BudgetedPool
from ...execution.shared_arrays import COPY_THREADS, attach_shared_array, close_shared_memory
from ...model.components import PreparedPanel
from ..founder_alleles import site_posterior
from .cavity_fill import cavity_fill
from .crossfit import crossfit_block, deterministic_two_folds
from .links import link_boundary
from .observation import UNIFORM_TOLERANCE
from .occupancy import occupancy_gate

# Consecutive blocks per task. A block takes 40-120 ms, so pairs halve the coordinator's dispatch
# work per block (task pickling, result handling), while thousands of tasks keep the tail short.
_BLOCKS_PER_TASK = 2

# Consecutive boundary links per task (one link is about 2 ms of work).
_LINKS_PER_TASK = 8

# A pool task's working memory besides its blocks' arrays (unpickled inputs, pickled results, Python and
# allocator slack, the attached evidence pages it reads): measured task peaks in workers exceeded the
# in-process block peaks by up to ~40 MB at N = 116 (tropheops chr8).
_TASK_OVERHEAD_BYTES = 48 << 20


def block_memory_bytes(samples: int, founders: int, sites: int, config) -> int:
    """Upper estimate of one block's working memory in a completion worker (one numeric thread).

    The block's private evidence copy (float64 x 3 and the observed mask per sample and site); the
    cavity fill's emissions of one bin, (N, K, K, bin sites) float64, with the einsum's likelihood,
    probability and logarithm temporaries (five such arrays at the peak); its per-bin pair-state
    scores with their prefix and suffix sums (bins x N x K(K+1)/2 each); the cross-fit's pair-state
    scores and posteriors (N x K(K+1)/2). Measured in process at K = 30, N = 320, 200 sites: 591 MB
    (estimate 601 MB); on 300 tropheops chr8 / actm3 chr13 blocks (N 116): peak - estimate <= 1.2 MB.
    A pool task adds _TASK_OVERHEAD_BYTES; ``config``: CompletionConfig.
    """
    pairs = founders * (founders + 1) // 2
    width = min(sites, config.cavity.snps_per_bin)
    bins = -(-sites // config.cavity.snps_per_bin)
    return int(25 * samples * sites
               + 5 * 8 * samples * founders * founders * width
               + 3 * 8 * bins * samples * pairs
               + 8 * 8 * samples * pairs)


@dataclass(frozen=True)
class CompletionSummary:
    """What completion did (stored next to the prepared panels; one log line via ``describe``).

    candidate_fills: int64[B] cavity fills proposed per block (before the occupancy gate).
    released_fills: int64[B] fills released per block (published in ``PreparedPanel.released``).
    unknown_mass_veto: bool[B] blocks that released no fill because local selection left more
        unknown-copy mass than ``CompletionConfig.maximum_discovery_unknown_mass``.
    gate_reasons: (reason, founder rows) of every occupancy decision, sorted by reason.
    rare_without_focal_observation: candidate fills of rare-continuity rows withheld because no
        qualifying sample is focal-informative at the site.
    site_posterior: counts in ``site_posterior.SUMMARY_FIELDS`` order over both cross-fit folds
        and the cavity fills (sites above MAX_EXACT_BITS unresolved founders; zero at today's
        founder counts).
    guard_skipped: releases the Hamming-ball shell guard withheld (cross-fit and cavity).
    """

    candidate_fills: np.ndarray
    released_fills: np.ndarray
    unknown_mass_veto: np.ndarray
    gate_reasons: tuple[tuple[str, int], ...]
    rare_without_focal_observation: int
    site_posterior: tuple[int, ...]
    guard_skipped: int

    @property
    def blocks(self) -> int:
        return int(len(self.candidate_fills))

    @property
    def total_candidate_fills(self) -> int:
        return int(np.sum(self.candidate_fills))

    @property
    def total_released_fills(self) -> int:
        return int(np.sum(self.released_fills))

    @property
    def vetoed_blocks(self) -> int:
        return int(np.sum(self.unknown_mass_veto))

    def record(self) -> dict:
        """The summary as JSON-able builtins (``from_record`` inverts it)."""
        return {
            "candidate_fills": [int(value) for value in self.candidate_fills],
            "released_fills": [int(value) for value in self.released_fills],
            "unknown_mass_veto": [bool(value) for value in self.unknown_mass_veto],
            "gate_reasons": {reason: int(count) for reason, count in self.gate_reasons},
            "rare_without_focal_observation": int(self.rare_without_focal_observation),
            "site_posterior": dict(zip(site_posterior.SUMMARY_FIELDS,
                                       (int(value) for value in self.site_posterior))),
            "guard_skipped": int(self.guard_skipped),
        }

    @classmethod
    def from_record(cls, record: dict) -> "CompletionSummary":
        return cls(
            candidate_fills=np.asarray(record["candidate_fills"], dtype=np.int64),
            released_fills=np.asarray(record["released_fills"], dtype=np.int64),
            unknown_mass_veto=np.asarray(record["unknown_mass_veto"], dtype=np.bool_),
            gate_reasons=tuple(sorted((str(reason), int(count))
                                      for reason, count in record["gate_reasons"].items())),
            rare_without_focal_observation=int(record["rare_without_focal_observation"]),
            site_posterior=tuple(int(record["site_posterior"][field])
                                 for field in site_posterior.SUMMARY_FIELDS),
            guard_skipped=int(record["guard_skipped"]),
        )

    def describe(self) -> str:
        """One log line: blocks, candidate and released fills, vetoed blocks; site-posterior counts only
        when some are non-zero (base "[Preprocess] site posterior ...")."""
        text = (f"{self.blocks} blocks, {self.total_candidate_fills} candidate fills, "
                f"{self.total_released_fills} released, {self.vetoed_blocks} blocks vetoed by unknown mass")
        if any(self.site_posterior) or self.guard_skipped:
            counts = " ".join(f"{field}={value}" for field, value
                              in zip(site_posterior.SUMMARY_FIELDS, self.site_posterior))
            text += f"; site posterior {counts} guard_skipped_cells={self.guard_skipped}"
        return text


def link_memory_bytes(samples: int, founders: int) -> int:
    """Upper estimate of one boundary link's working memory: both blocks' carrier profiles and class
    summaries (a few N x K float64 arrays) and K x K cost and rank matrices."""
    return int(16 * 8 * samples * founders + 8 * 8 * founders * founders)


def _unknown_mass_veto(panel, maximum_mass: float) -> bool:
    """Local selection left unexplained copies: some sample with depth has an unknown-copy slot and
    the block's unknown-copy fraction exceeds ``maximum_mass``."""
    depth = np.asarray(panel.has_depth, dtype=np.bool_)
    slots = np.asarray(panel.unknown_slots, dtype=np.int64)
    mass = float(np.sum(slots[depth], dtype=np.float64) / max(2 * int(np.sum(depth)), 1))
    return bool(np.any(depth & (slots > 0)) and mass > maximum_mass)


@njit(cache=True, parallel=True)
def _focal_mask(neutral, observed, starts, offsets, kept, tolerance, focal):
    """Block-major focal-informative mask, in one scan over (block, sample) rows: block b occupies
    cells [N offsets[b], N offsets[b + 1]) as a (N, L_b) array; a cell is focal when the sample is
    observed at a kept site and its normalised evidence is non-uniform (divide before subtracting)."""
    samples = neutral.shape[0]
    blocks = len(starts)
    for task in prange(blocks * samples):
        block, sample = task // samples, task % samples
        start = offsets[block]
        width = offsets[block + 1] - start
        row = samples * start + sample * width
        first = starts[block]
        for site in range(width):
            seen = observed[sample, first + site] and kept[start + site]
            a = neutral[sample, first + site, 0]
            b = neutral[sample, first + site, 1]
            c = neutral[sample, first + site, 2]
            total = (a + b) + c
            if total > 0.0:
                a, b, c = a / total, b / total, c / total
                focal[row + site] = seen and max(a, b, c) - min(a, b, c) > tolerance
            else:
                focal[row + site] = False


def _block_view(flat, samples, start, stop):
    """Block [start, stop) of a block-major mask as (N, L)."""
    return flat[samples * start:samples * stop].reshape(samples, stop - start)


@contextmanager
def _attached(refs):
    """The evidence view's shared arrays, attached for one task (a task touches only the pages of its
    blocks' columns)."""
    entries = [attach_shared_array(ref) for ref in refs]
    try:
        yield tuple(array for _handle, array in entries)
    finally:
        handles = [handle for handle, _array in entries]
        del entries                 # drop the array views: a segment with live views cannot close
        close_shared_memory(handles)


def _block_task(task, shared, budget):
    """Cross-fit and cavity fill of consecutive blocks (in a pool worker, one numeric thread)."""
    refs, folds, crossfit_config, cavity_config, items = task
    settings = site_posterior.settings()
    # Private copies: both computations read only their block's evidence (observed discovery sites).
    with _attached(refs) as (neutral, observed):
        blocks = [(index, alleles, np.array(neutral[:, start:start + sites]),
                   observed[:, start:start + sites] & kept[None, :])
                  for index, start, sites, kept, alleles in items]
        del neutral, observed
    results = []
    for index, alleles, evidence, observed in blocks:
        crossfit = crossfit_block(alleles, evidence, observed, folds, crossfit_config, settings)
        called = alleles >= 0
        fill = cavity_fill(np.where(called, alleles, 0.5), called, evidence, observed, cavity_config,
                           settings)
        results.append((index, crossfit, fill))
    return tuple(results)


def _link_task(task, shared, budget):
    """Links of consecutive boundaries; each needs only its two adjacent cross-fits."""
    first, alleles, profiles, folds, link_config = task
    return first, tuple(link_boundary(alleles[offset], alleles[offset + 1], profiles[offset],
                                      profiles[offset + 1], folds, link_config)
                        for offset in range(len(alleles) - 1))


def _release(pool, errors):
    try:
        pool.close()
    except BaseException as error:  # reported by the coordinator after the gate
        errors.append(error)


def complete(panels: Sequence, evidence, config, budget) -> tuple[tuple[PreparedPanel, ...],
                                                                 CompletionSummary]:
    """Complete the local panels of one assembly (inputs are never modified).

    panels: ``model.local_panels.LocalPanel`` per assembled block, in genomic order; evidence:
    ``assembly.evidence.AssemblyEvidence`` of the contig (block b reads ``neutral[:, s:s + L]`` and
    ``observed[:, s:s + L]``, s = panel.site_start; a view not in shared memory is shared first);
    config: ``CompletionConfig``; budget: the CPU budget (``execution.CpuBudget`` or a thread
    count): at most ``budget.threads()`` workers.

    Returns one ``PreparedPanel`` per input panel (same order): inference = the calls with
    non-discovery sites -1; released = the calls plus the released fills (posterior > 0.5);
    alt_probability = local selection's, the fill posterior at released fills; support unchanged.
    """
    panels = tuple(panels)
    budget = as_budget(budget)
    blocks = len(panels)
    if blocks == 0:
        empty = np.zeros(0, dtype=np.int64)
        return (), CompletionSummary(empty, empty.copy(), np.zeros(0, dtype=np.bool_), (), 0,
                                     (0,) * len(site_posterior.SUMMARY_FIELDS), 0)
    evidence = evidence.ensure_shared()
    # The coordinator's own arrays: the focal mask (one byte per sample and site), the inference copies,
    # and the retained per-block results (carrier profiles, fills and posteriors).
    resident = sum(evidence.samples * panel.sites + 10 * panel.calls.size
                   + 4 * 8 * evidence.samples * panel.calls.shape[0] for panel in panels)
    with reserve_memory(resident, label="completion coordinator"):
        return _complete(panels, evidence, config, budget)


def _complete(panels, evidence, config, budget):
    blocks = len(panels)
    samples = evidence.samples
    folds = deterministic_two_folds(samples, config.crossfit.fold_seed)
    veto = np.array([_unknown_mass_veto(panel, config.maximum_discovery_unknown_mass)
                     for panel in panels], dtype=np.bool_)
    inference = []
    for panel in panels:
        alleles = np.array(panel.calls, dtype=np.int8)
        alleles[:, ~panel.kept] = -1
        inference.append(alleles)
    offsets = np.zeros(blocks + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([panel.sites for panel in panels])
    starts = np.array([panel.site_start for panel in panels], dtype=np.int64)
    kept = np.concatenate([panel.kept for panel in panels]).astype(np.bool_)
    focal = np.empty(samples * int(offsets[-1]), dtype=np.bool_)

    pool = None
    crossfits = [None] * blocks
    fills = [None] * blocks
    links = [None] * (blocks - 1)
    groups = [range(first, min(blocks, first + _BLOCKS_PER_TASK))
              for first in range(0, blocks, _BLOCKS_PER_TASK)]
    block_bytes = [_TASK_OVERHEAD_BYTES + max(block_memory_bytes(samples, inference[index].shape[0],
                                                                 panels[index].sites, config)
                                              for index in group) for group in groups]
    try:
        # The workers start up while the coordinator scans the focal mask.
        pool = BudgetedPool(budget, label="completion", processes=memory_worker_limit(
            block_bytes, min(budget.threads(), len(groups)), label="completion"))
        with numba_threads(min(budget.threads(), COPY_THREADS)):
            _focal_mask(evidence.neutral, evidence.observed, starts, offsets, kept, UNIFORM_TOLERANCE,
                        focal)
        refs = (evidence.shared.neutral, evidence.shared.observed)
        block_tasks = [
            (refs, folds, config.crossfit, config.cavity,
             tuple((index, int(starts[index]), int(panels[index].sites),
                    np.ascontiguousarray(panels[index].kept, dtype=np.bool_), inference[index])
                   for index in group))
            for group in groups]
        for results in pool.imap_unordered(_block_task, block_tasks, task_bytes=block_bytes):
            for index, crossfit, fill in results:
                crossfits[index] = crossfit
                fills[index] = fill
        link_tasks, link_bytes = [], []
        for first in range(0, blocks - 1, _LINKS_PER_TASK):
            last = min(blocks - 1, first + _LINKS_PER_TASK)
            link_tasks.append((first, tuple(inference[first:last + 1]),
                               tuple(crossfits[index].profiles for index in range(first, last + 1)),
                               folds, config.links))
            link_bytes.append(_TASK_OVERHEAD_BYTES + sum(link_memory_bytes(samples, inference[index].shape[0])
                                                         for index in range(first, last + 1)))
        if link_tasks:
            for first, boundary_links in pool.imap_unordered(_link_task, link_tasks, task_bytes=link_bytes):
                links[first:first + len(boundary_links)] = boundary_links
    except BaseException:
        if pool is not None:
            pool.terminate()
        raise
    errors: list = []
    releaser = threading.Thread(target=_release, args=(pool, errors),
                                name="completion-release", daemon=True)
    releaser.start()
    try:
        gate = occupancy_gate(
            tuple(fill.filled for fill in fills), tuple(crossfit.profiles for crossfit in crossfits),
            tuple(_block_view(focal, samples, int(offsets[index]), int(offsets[index + 1]))
                  for index in range(blocks)),
            tuple(links), config.occupancy, veto)
        prepared = []
        for panel, alleles, fill, released_fills in zip(panels, inference, fills, gate.released):
            released = np.array(panel.calls, dtype=np.int8)
            released[released_fills] = (fill.posterior_alt[released_fills] > 0.5).astype(np.int8)
            alt_probability = np.array(panel.alt_probability, dtype=np.float64)
            alt_probability[released_fills] = fill.posterior_alt[released_fills]
            prepared.append(PreparedPanel(
                block=int(panel.block), site_start=int(panel.site_start), positions=panel.positions,
                kept=panel.kept, released=released, inference=alleles,
                alt_probability=alt_probability, support=panel.support))
        totals = np.zeros(len(site_posterior.SUMMARY_FIELDS), dtype=np.int64)
        guard_skipped = 0
        for crossfit, fill in zip(crossfits, fills):
            totals += crossfit.site_posterior_counts
            totals += fill.site_posterior_counts
            guard_skipped += crossfit.guard_skipped + fill.guard_skipped
        summary = CompletionSummary(
            candidate_fills=np.array([np.sum(fill.filled) for fill in fills], dtype=np.int64),
            released_fills=np.array([np.sum(mask) for mask in gate.released], dtype=np.int64),
            unknown_mass_veto=veto, gate_reasons=gate.reason_counts,
            rare_without_focal_observation=gate.rare_without_focal_observation,
            site_posterior=tuple(int(value) for value in totals), guard_skipped=int(guard_skipped))
    finally:
        releaser.join()
    if errors:
        raise errors[0]
    return tuple(prepared), summary
