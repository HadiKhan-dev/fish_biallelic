"""The endpoint beam: candidate chromosome paths through a batch's blocks from the linker's transitions (base
paths.run_endpoint_beam, FastMesh), and the beam's maximum transition gap (base hierarchy.compute_max_gap).

A path picks one founder per block. Its score is the sum over blocks k >= 1 of the mean of the float32 log
transitions into block k from the earlier blocks within the maximum gap (a missing gap contributes nothing).
Pass 1 builds paths left to right and keeps, at every block, at most a quota of the best paths per endpoint
founder. Pass 2 refines backwards: at each block k (n - 1 down to 0) every kept path tries every founder at k
with the rest fixed (only the affected steps are rescored), and the pooled candidates are selected with the same
endpoint quota. Every path selected at any refinement step is kept in an archive, so full paths through
interior states survive (at most n times the largest per-block sum of endpoint quotas).

Arithmetic and ties follow base exactly (the panel search's tie rules downstream depend on the order): scores
accumulate in float32 in past-block order, divide by the transition count in float32 and are stored in
float64; candidates are ranked by descending score, ties by the lexicographic path; duplicates are kept once;
the archive keeps first-selection order and is returned sorted by descending score (stable).
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from ...model.components import ComponentPanel
from ..linker.em import Transitions


def _lookup(transitions: Transitions) -> list[list[np.ndarray | None]]:
    """lookup[past][current]: the float32 log transition matrix between two blocks, or None (unfitted gap)."""
    n = transitions.blocks
    return [[transitions.matrix(past, current) if past < current else None for current in range(n)]
            for past in range(n)]


def _step_scores(path, lookup, gap_limit: int) -> np.ndarray:
    """float64[n] per-step scores of one path (step 0 is 0): float32 mean of the transitions into each block."""
    num_blocks = len(path)
    step_scores = np.zeros(num_blocks, dtype=np.float64)
    for current in range(1, num_blocks):
        earliest = max(0, current - gap_limit)
        step_total = 0.0
        count = 0
        for past in range(earliest, current):
            matrix = lookup[past][current]
            if matrix is not None:
                step_total += matrix[path[past], path[current]]
                count += 1
        if count > 0:
            step_scores[current] = step_total / count
    return step_scores


def _affected_steps(paths, step_scores, changed_block, choices, lookup, gap_limit: int):
    """float64[P, choices, n] per-step scores of every path with every founder choice at ``changed_block``.

    Only step ``changed_block`` (transitions into it) and steps changed_block + 1 .. changed_block + gap_limit
    (transitions out of it) change; each follows the scalar recursion (float32 accumulation in past-block
    order, float32 division by the transition count, float64 storage).
    """
    n_paths, num_blocks = paths.shape
    steps = np.repeat(step_scores[:, None, :], choices, axis=1)
    first = max(1, changed_block)  # step 0 is always 0
    last = min(changed_block + gap_limit, num_blocks - 1)
    for current in range(first, last + 1):
        earliest = max(0, current - gap_limit)
        step_total = np.zeros((n_paths, choices), dtype=np.float32)
        count = 0
        for past in range(earliest, current):
            matrix = lookup[past][current]
            if matrix is not None:
                if current == changed_block:
                    step_total += matrix[paths[:, past], :]
                elif past == changed_block:
                    step_total += matrix[:, paths[:, current]].T
                else:
                    step_total += matrix[paths[:, past], paths[:, current]][:, None]
                count += 1
        if count > 0:
            steps[:, :, current] = step_total / count
        else:
            steps[:, :, current] = 0.0
    return steps


def _endpoint_order(paths, scores, block, quota, overrides):
    """Indices of the candidates kept at ``block``, in selection order.

    Order: descending score (as float64), then the lexicographic path; identical paths are kept once; at most
    ``quota`` paths (or the endpoint's override) per founder at ``block``.
    """
    order = np.lexsort(tuple(paths[:, column] for column in range(paths.shape[1] - 1, -1, -1))
                       + (-np.asarray(scores, dtype=np.float64),))
    ordered = paths[order]
    keep = np.ones(len(order), dtype=bool)
    keep[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    order = order[keep]
    endpoint = paths[order, block]
    # Rank of every candidate among those with the same endpoint so far.
    by_endpoint = np.argsort(endpoint, kind='stable')
    sorted_endpoint = endpoint[by_endpoint]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_endpoint)) + 1]
    sizes = np.diff(np.r_[starts, len(sorted_endpoint)])
    rank = np.empty(len(order), dtype=np.int64)
    rank[by_endpoint] = np.arange(len(order)) - np.repeat(starts, sizes)
    limit = np.full(len(order), quota, dtype=np.int64)
    if overrides:
        for (override_block, founder), value in overrides.items():
            if override_block == block:
                limit[endpoint == founder] = value
    return order[rank < limit]


def endpoint_beam(transitions: Transitions, *, quota: int,
                  overrides: Mapping[tuple[int, int], int] | None = None,
                  max_gap: int | None = None) -> list[tuple[tuple[int, ...], float]]:
    """Candidate paths of one batch with their scores, sorted by descending score (stable).

    quota: paths kept per endpoint founder at every block. overrides: {(block, founder): quota} for endpoints
    whose quota differs (the panel search's pool extension and saturation re-runs). max_gap: the largest gap
    whose transitions enter a step (None: every fitted gap). Each item is (founder of each block, score); the
    score is the float64 sum of the path's float32 step scores.
    """
    num_blocks = transitions.blocks
    gap_limit = max_gap if max_gap is not None else num_blocks
    lookup = _lookup(transitions)

    # Pass 1 (forward): mean of the transitions into each new block.
    first_founders = int(transitions.founders[0])
    beam_paths = np.arange(first_founders, dtype=np.int64)[:, None]
    beam_scores = np.zeros(first_founders, dtype=np.float32)
    for current in range(1, num_blocks):
        choices = int(transitions.founders[current])
        earliest = max(0, current - gap_limit)
        transition_to_current = np.zeros((len(beam_paths), choices), dtype=np.float32)
        count = 0
        for past in range(earliest, current):
            matrix = lookup[past][current]
            if matrix is not None:
                transition_to_current += matrix[beam_paths[:, past], :]
                count += 1
        mean_transition = transition_to_current / count if count > 0 else transition_to_current
        scores = (beam_scores[:, None] + mean_transition).ravel()
        paths = np.empty((len(beam_paths) * choices, current + 1), dtype=np.int64)
        paths[:, :current] = np.repeat(beam_paths, choices, axis=0)
        paths[:, current] = np.tile(np.arange(choices, dtype=np.int64), len(beam_paths))
        kept = _endpoint_order(paths, scores, current, quota, overrides)
        beam_paths, beam_scores = paths[kept], scores[kept]

    # Pass 2 (backward refinement of the forward paths as scaffolds, with cached step scores).
    scaffold_paths = beam_paths
    scaffold_steps = np.stack([_step_scores(path, lookup, gap_limit) for path in beam_paths.tolist()])
    scaffold_totals = np.array([np.sum(steps) for steps in scaffold_steps])
    archive = {}
    for path, total in zip(map(tuple, scaffold_paths.tolist()), scaffold_totals):
        archive.setdefault(path, total)
    for refine_block in range(num_blocks - 1, -1, -1):
        choices = int(transitions.founders[refine_block])
        n_scaffolds = len(scaffold_paths)
        steps = _affected_steps(scaffold_paths, scaffold_steps, refine_block, choices, lookup, gap_limit)
        totals = steps.reshape(-1, num_blocks).sum(axis=1).reshape(n_scaffolds, choices)
        # An unchanged scaffold keeps its cached scores.
        original = scaffold_paths[:, refine_block]
        rows = np.arange(n_scaffolds)
        steps[rows, original] = scaffold_steps
        totals[rows, original] = scaffold_totals
        paths = np.repeat(scaffold_paths, choices, axis=0)
        paths[:, refine_block] = np.tile(np.arange(choices, dtype=np.int64), n_scaffolds)
        totals = totals.ravel()
        kept = _endpoint_order(paths, totals, refine_block, quota, overrides)
        scaffold_paths = paths[kept]
        scaffold_steps = steps.reshape(-1, num_blocks)[kept]
        scaffold_totals = totals[kept]
        for path, total in zip(map(tuple, scaffold_paths.tolist()), scaffold_totals):
            archive[path] = total

    result = [(path, float(total)) for path, total in archive.items()]
    result.sort(key=lambda item: item[1], reverse=True)
    return result


def max_gap(blocks: Sequence[ComponentPanel], genetic_map, n_generations, tolerance) -> int:
    """The beam's (and linker's) maximum transition gap of a batch at levels >= 2 (base compute_max_gap).

    recombinations per step = mean block span x n_generations (in Morgans: the map's interval distance when
    the contig has a map, else span bp x fallback rate); gap = max(1, 1 + floor(tolerance / recombinations per
    step)); n blocks when a step carries no recombination; 1 when no block spans two sites.
    """
    spans = [block.positions[-1] - block.positions[0] for block in blocks if len(block.positions) > 1]
    if not spans:
        return 1
    if genetic_map.has_map:
        morgans = [float(genetic_map.interval_morgans(block.positions[0], block.positions[-1]))
                   for block in blocks if len(block.positions) > 1]
        recombinations_per_step = np.mean(morgans) * n_generations
    else:
        recombinations_per_step = np.mean(spans) * genetic_map.fallback_rate_per_bp * n_generations
    if recombinations_per_step <= 0:
        return len(blocks)
    return max(1, 1 + int(math.floor(tolerance / recombinations_per_step)))
