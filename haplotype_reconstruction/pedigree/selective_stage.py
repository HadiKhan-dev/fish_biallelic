"""Default-stage adapter for the certified genome-wide M1 screen.

This migration keeps the existing dense decision/exposure API. Only the expensive
all-pairs likelihood calculation is replaced. Compact profiles and their fixed
chromosome values have separate checkpoint keys from the raw reference screen.
"""
from __future__ import annotations

import numpy as np

from .. import execution
from ..store import CheckpointStore
from . import panel, records, scoring, selective_screen
from .compact_m1 import CompactM1


def _score_handler(contig, state, phase, common, budget):
    store = CheckpointStore(common["root"], nthreads=execution.claim_threads())
    file = store.contig(contig)
    candidate_panel = common["panel"]
    store.bind_identity(file, records.FLOORED_SCORES,
                        records.floored_scores_identity(store, file, candidate_panel))
    if file.has(records.FLOORED_SCORES):
        return None, records.decode_scores(file.get(records.FLOORED_SCORES))
    saved = store.run.get(records.SELECTIVE_SCREEN)
    compact = CompactM1.decode(saved["compact"])
    chromosome = tuple(saved["contigs"]).index(contig)
    one = compact.chromosome_reference(chromosome, common["eligibility"])
    prepared = records.decode_prepared(file.get(records.PREPARED))
    value = scoring.score_chromosome_selective(
        prepared, one, candidate_panel.trios, common["eligibility"], common["config"],
        execution.claim_threads)
    file.put(records.FLOORED_SCORES, records.encode_scores(value))
    return None, value


def score_prepared(store, contigs, eligibility, config, budget, *, method="certified"):
    """Screen, freeze the G4 panel/profile, then score M2 and exposure.

    Called by stage.run after preparation; ``method='exact'`` is the same-API
    all-pairs reference. Checkpoint reuse is independent of the search method,
    since both compute the same exact fixed profile.
    """
    from .stage import _scoring_bytes

    contigs = tuple(contigs)
    store.bind_identity(store.run, records.SELECTIVE_SCREEN,
                        records.selective_screen_identity(store, contigs, config, eligibility))
    if store.run.has(records.SELECTIVE_SCREEN):
        compact = CompactM1.decode(store.run.get(records.SELECTIVE_SCREEN, items=("compact",))["compact"])
    else:
        markers = np.asarray([store.contig(c).get(records.PREPARED,
                              items=("informative_markers",))["informative_markers"] for c in contigs],
                             dtype=np.float64)
        compact, zero, work = selective_screen.run(
            store.root, contigs, markers, eligibility, config, budget, method=method)
        store.run.put(records.SELECTIVE_SCREEN,
                      dict(contigs=contigs, compact=compact.encode(), zero=zero, work=work))
    # Explicit dense migration adapter. The downstream layouts are unchanged;
    # this is not a claim that pedigree memory is already sub-quadratic.
    one = compact.dense_reference(eligibility)
    candidate_panel = panel.retained_panel(
        compact.keep_reference(), panel.robust_screen(one, compact.markers, eligibility, config), eligibility)
    del one, compact
    estimates = [_scoring_bytes(store, c) for c in contigs]
    limit = execution.memory_worker_limit(estimates, len(contigs), label="PEDIGREE SELECTIVE M2")
    weights = [store.contig(c).info(records.PREPARED).stored_bytes for c in contigs]
    common = dict(root=store.root, panel=candidate_panel, config=config, eligibility=eligibility)
    with execution.ChromosomeExecutor(contigs, _score_handler, budget=budget, weights=weights,
                                      max_workers=limit, label="PEDIGREE SELECTIVE M2",
                                      task_bytes=estimates) as pool:
        scores = tuple(pool.run("score", common))
    return candidate_panel, scores
