"""The endpoint beam: candidate chromosome paths through a batch's blocks from the linker's transitions (base
paths.run_endpoint_beam, FastMesh), and the beam's maximum transition gap (base hierarchy.compute_max_gap).

A path picks one founder per block. Its score is the sum over blocks k >= 1 of the mean of the log transitions into
block k from the earlier blocks within the maximum gap (a missing gap contributes nothing). Pass 1 builds paths left
to right and keeps, at every block, at most a quota of the best paths per endpoint founder. Pass 2 refines
backwards: at each block k (n - 1 down to 0) every kept path tries every founder at k with the rest fixed (only the
affected steps are rescored), and the pooled candidates are selected with the same endpoint quota. Every path
selected at any refinement step is kept in an archive, so full paths through interior states survive (at most n
times the largest per-block sum of endpoint quotas, twice that when tie classes extend the quotas).

Arithmetic and ties (release 2.1, design 5.4): the log transitions are float64 (linker.em), scores accumulate in
float64 in past-block order and divide by the transition count in float64, so a one-ulp change of a stored
transition no longer reorders paths. Candidates are ranked by anchored tie classes of their scores
(maths.ties rules; quantum ``tie_rules.quantum("beam", scale of the cut)``), the lexicographic path deciding inside
a class; identical paths are kept once (their best score); an endpoint's quota keeps every path of the tie class it
cuts (at most twice the quota). The archive is returned in the same tie-class order (descending score, content
inside a class), so the panel search's pool order is a content order up to genuine score differences.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from ...maths import ties
from ...model.components import ComponentPanel
from ..linker.em import Transitions
from ..tie_rules import count_vector_classes, motion, path_class_order, quantum

# Beam-cut census: scores of the paths within this many ranks of an endpoint's quota are recorded per cut.
_CENSUS_RANKS = 3


def _lookup(transitions: Transitions) -> list[list[np.ndarray | None]]:
    """lookup[past][current]: the log transition matrix between two blocks, or None (unfitted gap)."""
    n = transitions.blocks
    return [[transitions.matrix(past, current) if past < current else None for current in range(n)]
            for past in range(n)]


def _step_scores(path, lookup, gap_limit: int) -> np.ndarray:
    """float64[n] per-step scores of one path (step 0 is 0): the mean of the transitions into each block (float64
    sums in past-block order)."""
    num_blocks = len(path)
    step_scores = np.zeros(num_blocks, dtype=np.float64)
    for current in range(1, num_blocks):
        earliest = max(0, current - gap_limit)
        step_total = 0.0
        count = 0
        for past in range(earliest, current):
            matrix = lookup[past][current]
            if matrix is not None:
                step_total += float(matrix[path[past], path[current]])
                count += 1
        if count > 0:
            step_scores[current] = step_total / count
    return step_scores


def _affected_steps(paths, step_scores, changed_block, choices, lookup, gap_limit: int):
    """float64[P, choices, n] per-step scores of every path with every founder choice at ``changed_block``.

    Only step ``changed_block`` (transitions into it) and steps changed_block + 1 .. changed_block + gap_limit
    (transitions out of it) change; each follows the scalar recursion (float64 sums in past-block order, divided
    by the transition count), so the values equal :func:`_step_scores` bit for bit.
    """
    n_paths, num_blocks = paths.shape
    steps = np.repeat(step_scores[:, None, :], choices, axis=1)
    first = max(1, changed_block)  # step 0 is always 0
    last = min(changed_block + gap_limit, num_blocks - 1)
    for current in range(first, last + 1):
        earliest = max(0, current - gap_limit)
        step_total = np.zeros((n_paths, choices), dtype=np.float64)
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


def _ranked(paths, scores):
    """Tie-class order of the candidates, identical paths kept once (their best-scoring copy, the first index among
    equal scores). Returns (kept indices in decision order, their classes in that order, wide classes, boundary
    gaps, the quantum)."""
    scores = np.asarray(scores, dtype=np.float64)
    count = len(paths)
    if count > 1:
        grouped = np.lexsort((-scores,) + tuple(paths[:, column]
                                                for column in range(paths.shape[1] - 1, -1, -1)))
        ordered = paths[grouped]
        first = np.ones(count, dtype=bool)
        first[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
        representatives = grouped[first]
    else:
        representatives = np.arange(count, dtype=np.int64)
    values = scores[representatives]
    finite = values[np.isfinite(values)]
    q = quantum("beam", float(np.max(np.abs(finite))) if finite.size else 0.0)
    order, classes, wide, _ = path_class_order(values, paths[representatives], q, descending=True)
    return representatives[order], classes[order], wide, q


def _cut_boundaries(values, classes, cut_classes, q):
    """Boundary-guard events of a beam cut: the gaps entering and leaving each class an endpoint quota cuts
    (values and classes in decision order, classes non-decreasing) that lie within BOUNDARY_GUARD noise
    motions of q. Only these gaps decide which paths a quota keeps; gaps elsewhere in the candidate list are
    not exposures of the cut."""
    guard_motion = motion("beam")
    if guard_motion is None or not cut_classes:
        return ()
    starts = np.flatnonzero(np.r_[True, classes[1:] != classes[:-1]])
    finite = np.where(np.isfinite(values), values, 0.0)
    high = np.maximum.reduceat(finite, starts)
    low = np.minimum.reduceat(finite, starts)
    guard = ties.BOUNDARY_GUARD * guard_motion
    events = set()
    for c in sorted(cut_classes):
        for gap in ((low[c - 1] - high[c]) if c > 0 else None, (low[c] - high[c + 1]) if c + 1 < len(starts) else None):
            if gap is not None and abs(gap - q) <= guard:
                events.add((int(c), float(gap)))
    return tuple(gap for _, gap in sorted(events))


def _endpoint_order(paths, scores, block, quota, overrides, *, counter=None, census_key=None):
    """Indices of the candidates kept at ``block``, in selection (decision) order.

    Order: tie classes of the scores (descending), the lexicographic path inside a class; identical paths are kept
    once; at most ``quota`` paths (or the endpoint's override) per founder at ``block``, extended by the paths of the
    tie class an endpoint's quota cuts (at most twice the quota)."""
    ranked, in_order, wide, q = _ranked(paths, scores)
    endpoint = paths[ranked, block]
    kept = np.zeros(len(ranked), dtype=bool)
    collect = census_key is not None and ties.census_enabled()
    scores = np.asarray(scores, dtype=np.float64)
    inside = False
    cut_classes = set()
    for founder in np.unique(endpoint).tolist():
        members = np.flatnonzero(endpoint == founder)
        limit = int(quota)
        if overrides:
            limit = int(overrides.get((block, founder), limit))
        if collect and len(members) > 1:
            low, high = max(0, limit - _CENSUS_RANKS), min(len(members), limit + _CENSUS_RANKS)
            if high - low > 1:
                ties.census("assembly.beam_cut", f"{census_key}/e{founder}", scores[ranked[members[low:high]]].tolist())
        if len(members) <= limit:
            kept[members] = True
            continue
        cut = in_order[members[limit - 1]]
        cut_classes.add(int(cut))
        # The whole class the quota cuts is kept, unless it would take the endpoint beyond twice its quota: then
        # the whole class is dropped (release 2.2: order-invariant; 2.1 kept its first members in content order).
        start = limit - 1
        while start > 0 and in_order[members[start - 1]] == cut:
            start -= 1
        end = limit
        while end < len(members) and in_order[members[end]] == cut:
            end += 1
        if end > limit:
            inside = True
        kept[members[:end] if (end <= 2 * limit or start == 0) else members[:start]] = True
    boundary = _cut_boundaries(scores[ranked], in_order, cut_classes, q)
    count_vector_classes(counter, "assembly.beam_cut", in_order, wide, boundary, inside)
    return ranked[kept]


def path_linkage(transitions: Transitions, max_gap: int | None = None):
    """A function path -> the path's linker log-probability score (float64 sum of its step scores, the beam's
    ranking value) for this batch's transitions; the join search's data criterion inside log-likelihood tie classes
    (release 2.2). Results are cached per path."""
    lookup = _lookup(transitions)
    gap_limit = max_gap if max_gap is not None else transitions.blocks
    cache = {}

    def score(path) -> float:
        key = tuple(int(row) for row in path)
        found = cache.get(key)
        if found is None:
            found = cache[key] = float(_step_scores(np.asarray(key, dtype=np.int64), lookup, gap_limit).sum())
        return found

    return score


def endpoint_beam(transitions: Transitions, *, quota: int,
                  overrides: Mapping[tuple[int, int], int] | None = None,
                  max_gap: int | None = None, counter: ties.TieCounter | None = None,
                  label: str | None = None) -> list[tuple[tuple[int, ...], float]]:
    """Candidate paths of one batch with their scores, in tie-class order (descending score, the lexicographic path
    inside a class).

    quota: paths kept per endpoint founder at every block. overrides: {(block, founder): quota} for endpoints
    whose quota differs (the panel search's pool extension and saturation re-runs). max_gap: the largest gap
    whose transitions enter a step (None: every fitted gap). Each item is (founder of each block, score); the
    score is the float64 sum of the path's step scores. counter: TieCounter of the batch (beam_cut decisions);
    label: the batch's census label (None: no census).
    """
    num_blocks = transitions.blocks
    gap_limit = max_gap if max_gap is not None else num_blocks
    lookup = _lookup(transitions)
    key = None
    if label is not None and ties.census_enabled():
        signature = ",".join(f"{b}:{f}={v}" for (b, f), v in sorted((overrides or {}).items()))
        key = f"{label}/q{quota}/{signature}"

    # Pass 1 (forward): mean of the transitions into each new block.
    first_founders = int(transitions.founders[0])
    beam_paths = np.arange(first_founders, dtype=np.int64)[:, None]
    beam_scores = np.zeros(first_founders, dtype=np.float64)
    for current in range(1, num_blocks):
        choices = int(transitions.founders[current])
        earliest = max(0, current - gap_limit)
        transition_to_current = np.zeros((len(beam_paths), choices), dtype=np.float64)
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
        kept = _endpoint_order(paths, scores, current, quota, overrides, counter=counter,
                               census_key=None if key is None else f"{key}/p1/b{current}")
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
        kept = _endpoint_order(paths, totals, refine_block, quota, overrides, counter=counter,
                               census_key=None if key is None else f"{key}/p2/b{refine_block}")
        scaffold_paths = paths[kept]
        scaffold_steps = steps.reshape(-1, num_blocks)[kept]
        scaffold_totals = totals[kept]
        for path, total in zip(map(tuple, scaffold_paths.tolist()), scaffold_totals):
            archive[path] = total

    if not archive:
        return []
    result_paths = np.asarray(list(archive.keys()), dtype=np.int64).reshape(len(archive), num_blocks)
    result_scores = np.asarray(list(archive.values()), dtype=np.float64)
    ranked, _, _, _ = _ranked(result_paths, result_scores)
    return [(tuple(int(row) for row in result_paths[i]), float(result_scores[i])) for i in ranked.tolist()]


def max_gap(blocks: Sequence[ComponentPanel], genetic_map, gap_horizon_scale, tolerance) -> int:
    """The beam's (and linker's) maximum transition gap of a batch at levels >= 2 (base compute_max_gap).

    scaled span = mean component span in Morgans x gap_horizon_scale (the map's interval
    distance when available, else span bp x fallback rate). The search horizon is
    max(1, 1 + floor(tolerance / scaled span)); all blocks when the scaled span is zero,
    1 when no component spans two sites. This is a proposal limit, not a copying rate.
    """
    spans = [block.positions[-1] - block.positions[0] for block in blocks if len(block.positions) > 1]
    if not spans:
        return 1
    if genetic_map.has_map:
        morgans = [float(genetic_map.interval_morgans(block.positions[0], block.positions[-1]))
                   for block in blocks if len(block.positions) > 1]
        scaled_span = np.mean(morgans) * gap_horizon_scale
    else:
        scaled_span = np.mean(spans) * genetic_map.fallback_rate_per_bp * gap_horizon_scale
    if scaled_span <= 0:
        return len(blocks)
    return max(1, 1 + int(math.floor(tolerance / scaled_span)))
