"""Build the three selection rounds from explicit panels, proposals and incumbent state.

Owns the pure LS23/LS25 round schedule and discovery-site projections.
Must not assemble context, read records, schedule workers or inspect runtime state.
Method: docs/methods/selection.md#ls25.
Earlier released calls remain candidates; the newest context founders retain their
multiplicity and witness mask. Memos survive exactly the initial and l1 rounds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import cast

import numpy as np

from ...contracts.discovery import DiscoveredBlock
from ...contracts.genetic_map import ChromosomeGeneticMap
from ...contracts.selection import ContextProposal
from .rounds import BlockSelection, BlockTask

__all__ = ["initial_tasks", "round_tasks"]


def initial_tasks(
    discovered: Sequence[DiscoveredBlock | None], genetic_map: ChromosomeGeneticMap, *, label: str = ""
) -> tuple[BlockTask, ...]:
    """Project informative discovered blocks onto their kept sites, in block order.

    The discovery contract provides positions int64[L], kept bool[L], latent
    int64[K,L_kept] and tied_panels. Empty latent panels supply no selection task.
    Input arrays are read without mutation. Invalid map or array shapes propagate.
    """
    tasks: list[BlockTask] = []
    for block in discovered:
        if block is None or not len(block.latent):
            continue
        index = len(tasks)
        positions = block.positions[block.kept]
        tasks.append(
            BlockTask(
                index=index,
                block=block.block,
                site_start=block.site_start,
                positions=block.positions,
                kept=block.kept,
                columns=block.site_start + np.flatnonzero(block.kept),
                intervals=cast(np.ndarray, genetic_map.interval_morgans(positions[:-1], positions[1:])),
                latent=np.ascontiguousarray(block.latent, dtype=np.int64),
                alternatives=tuple(
                    np.ascontiguousarray(panel, dtype=np.int64) for panel in block.tied_panels if len(panel)
                ),
                proposals=(),
                founders=None,
                witness=None,
                incumbent=None,
                incumbent_release=None,
                memo=None,
                keep_memo=True,
                label=f"{label}/initial/i{index}" if label else "",
            )
        )
    return tuple(tasks)


def round_tasks(
    initial: Sequence[BlockTask],
    round_index: int,
    previous: Sequence[BlockSelection] | None,
    selected_calls: Sequence[Sequence[np.ndarray]],
    proposals: Sequence[ContextProposal] | None,
    *,
    label: str = "",
) -> tuple[BlockTask, ...]:
    """Build initial/l1/l2 block tasks without changing any candidate or memo state.

    selected_calls contains all earlier released panels on their original block
    axes. A context proposal is sliced only after its duplicate founders/witnesses
    are expanded. previous is the immediate round's representative outcome.
    All outputs are immutable task records; no input is mutated.
    """
    name = ("initial", "l1", "l2")[round_index]
    tasks: list[BlockTask] = []
    for block in initial:
        index, kept = block.index, block.kept
        if round_index:
            assert proposals is not None
            proposal = proposals[index]
            candidates = tuple(np.ascontiguousarray(calls[index][:, kept]) for calls in selected_calls)
            candidates += (np.ascontiguousarray(proposal.calls[:, kept]),)
            founders = proposal.founders
            founders = None if founders is None else np.ascontiguousarray(founders[:, kept])
            witness = proposal.founder_witness
            witness = None if witness is None else np.ascontiguousarray(witness[:, kept])
        else:
            candidates, founders, witness = (), None, None
        prior = None if previous is None else previous[index]
        tasks.append(
            replace(
                block,
                proposals=candidates,
                founders=founders,
                witness=witness,
                incumbent=None if prior is None else prior.outcome,
                incumbent_release=None if prior is None else prior.release,
                memo=None if prior is None else prior.memo,
                keep_memo=round_index < 2,
                label=f"{label}/{name}/i{index}" if label else "",
            )
        )
    return tuple(tasks)
