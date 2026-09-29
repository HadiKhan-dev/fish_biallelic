"""Allele-preserving L1 pruning under the existing full panel objective.

A row can be redundant as a long path yet hold indispensable local pieces.
Propose moving those pieces to ONE surviving row before deleting it. Direct
deletions preserve called allele presence; rehoming also preserves local
sequence classes. Neither condition establishes historical founder identity.

The production pass is confined to L1 of final assembly, before progressive
refinement. It does not change feedback, later-level search, or the score.
A fixed full-score budget bounds large-panel searches; ordinary small panels
receive the exhaustive single-recipient search used by the matched controls.
"""
from heapq import nlargest
import math

import numpy as np

from . import chimera_scoring as scoring
from .panel_rehousing import sequence_class_maps
from .panel_search import conditional_fields, evaluate_panel
from ..discovery.objectives import compute_outer_bic_from_log_likelihood


def _descriptions(rows, batch_blocks, class_maps):
    """O(K L + B K²) screening without materializing K² whole panels."""
    k, blocks = rows.shape
    labels = np.column_stack([
        mapping[rows[:, b]] for b, mapping in enumerate(class_maps)])
    class_counts = np.empty_like(labels)
    can_drop = np.ones(k, dtype=np.bool_)
    for b, block in enumerate(batch_blocks):
        calls = np.asarray(block.discrete_haps, dtype=np.int8)[rows[:, b]]
        for allele in (0, 1):
            present = calls == allele
            can_drop &= ~np.any(present & (present.sum(axis=0) == 1), axis=1)
        _, inverse, counts = np.unique(labels[:, b], return_inverse=True,
                                        return_counts=True)
        class_counts[:, b] = counts[inverse]
    # Preserve the reference proposal order, including stable score ties.
    descriptions = [(drop, -1, ()) for drop in range(k) if can_drop[drop]]
    for drop in range(k):
        unique = tuple(map(int, np.flatnonzero(
            (labels[drop] >= 0) & (class_counts[drop] == 1))))
        if not unique:
            # No called unique class implies the direct drop was already seen.
            continue
        for receiver in range(k):
            if receiver == drop:
                continue
            if all(labels[receiver, b] < 0 or class_counts[receiver, b] > 1
                   for b in unique):
                descriptions.append((drop, receiver, unique))
    return descriptions


def _materialize(rows, description):
    drop, receiver, moved = description
    result = rows.copy()
    if receiver >= 0:
        result[receiver, list(moved)] = rows[drop, list(moved)]
    # Keep multiplicity: collapsing another row would change the tested move.
    return np.delete(result, drop, axis=0)


def _signature(rows):
    return tuple(sorted(map(tuple, rows.tolist())))


def _rank_descriptions(rows, descriptions, paths, sub, penalty, samples,
                       full_scores, num_threads):
    """Large-panel screening only; conditional gains are NOT safe bounds."""
    if len(descriptions) <= full_scores:
        return descriptions, False
    k = len(rows)
    painting = evaluate_panel(paths, sub, penalty, samples, paint=True,
                              num_threads=num_threads)
    fields = []
    reassignment = np.zeros((k, k))
    start = 0
    for b, emission in enumerate(sub):
        stop = start + emission['n_bins']
        local = np.ascontiguousarray(painting[:, start:stop])
        unary, _ = conditional_fields(emission['bin_emissions'], local,
                                      rows[:, b], k)
        fields.append(unary)
        reassignment += unary[:, rows[:, b]]
        start = stop
    np.fill_diagonal(reassignment, -np.inf)
    deletion = reassignment.max(axis=1)

    def proxy(item):
        _, (drop, receiver, moved) = item
        value = deletion[drop]
        if receiver >= 0:
            value += sum(fields[b][receiver, rows[drop, b]] for b in moved)
        return value

    # Rank before materialization to retain cubic total work at large K.
    # Restore original ordering among selected proposals for deterministic ties.
    selected = nlargest(full_scores, enumerate(descriptions), key=proxy)
    return [description for _, description in sorted(selected)], True


def prune_rows(rows, batch_blocks, sub, penalty, samples, complexity_cost, *,
               full_scores=256, num_threads=None):
    """Greedy count-down; every accepted move passes an unchanged full score.

    At most K-1 reductions are possible. A fixed Q full-score budget gives
    O(K² L + B K³ + Q N m K³) total work for local dictionaries of O(K) rows, ignoring sorting factors, rather
    than scoring all K² alternatives at every count. No biological K cap is
    imposed. Truncated large-panel rounds are explicitly diagnostic.
    """
    if full_scores < 1:
        raise ValueError('full_scores must be positive')
    rows = np.asarray(rows, dtype=np.int64).copy()
    class_maps = sequence_class_maps(batch_blocks)
    mappings = [sorted(block.haplotypes) for block in batch_blocks]

    def paths(panel):
        return [[mappings[b][h] for b, h in enumerate(row)] for row in panel]

    def score(panel):
        likelihood = evaluate_panel(paths(panel), sub, penalty, samples,
                                     num_threads=num_threads)
        return (compute_outer_bic_from_log_likelihood(
            len(panel), likelihood, complexity_cost), likelihood)

    initial_k = len(rows)
    bic, likelihood = score(rows)
    initial_bic = bic
    history = []
    while len(rows) > 1:
        descriptions = _descriptions(rows, batch_blocks, class_maps)
        ranked, truncated = _rank_descriptions(
            rows, descriptions, paths(rows), sub, penalty, samples,
            full_scores, num_threads)
        seen = set()
        best = None
        for description in ranked:
            candidate = _materialize(rows, description)
            key = _signature(candidate)
            if key in seen:
                continue
            seen.add(key)
            candidate_bic, candidate_ll = score(candidate)
            if best is None or candidate_bic < best[0]:
                best = candidate_bic, candidate_ll, candidate, description
        history.append(dict(panel_size=len(rows), proposals=len(descriptions),
            full_scores=len(seen), truncated=truncated,
            best_score_change=None if best is None else float(best[0] - bic),
            accepted=bool(best is not None and best[0] - bic < -1e-8)))
        if best is None or best[0] - bic >= -1e-8:
            break
        bic, likelihood, rows, _ = best
    diagnostic = dict(stage='allele_preserving_l1_pruning', initial_panel_size=initial_k,
        final_panel_size=len(rows), initial_bic=float(initial_bic), bic=float(bic),
        full_scores_per_round=full_scores, history=history,
        scope='single-recipient search; called allele presence is not ancestral phase')
    return rows, likelihood, diagnostic


def prune_selected_panel(resolved_beam, batch_blocks, global_probs, global_sites, *,
                         max_bins=2000, full_scores=256, cc_scale=.5,
                         num_threads=None):
    """Hierarchy adapter; preserve local row indices and score geometry."""
    if len(resolved_beam) < 2:
        return resolved_beam, None
    spb = max(scoring.compute_spb(batch_blocks),
              math.ceil(sum(len(b.positions) for b in batch_blocks) / max_bins))
    sub = scoring.compute_subblock_emissions(
        batch_blocks, global_probs, global_sites, spb, num_threads=num_threads)
    rows, likelihood, diagnostic = prune_rows(
        [path for path, _ in resolved_beam], batch_blocks, sub,
        scoring.compute_penalty(batch_blocks), global_probs.shape[0],
        scoring.compute_cc(batch_blocks, global_probs.shape[0], cc_scale),
        full_scores=full_scores, num_threads=num_threads)
    return [(list(row), float(likelihood)) for row in rows], diagnostic
