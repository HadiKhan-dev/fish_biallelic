"""Default-stage adapter for the certified genome-wide M1 screen.

The chromosome records keep the dense per-chromosome [N, N] API; only the expensive
all-pairs likelihood calculation is replaced. Compact profiles and their fixed
chromosome values have separate checkpoint keys from the raw reference screen.
The score workers only record each chromosome: the decision decodes the records one
at a time (``load_scores``) and keeps their eligible-edge entries (GenomeScores).
"""
from __future__ import annotations

import numpy as np

from .. import execution
from ..store import CheckpointStore
from . import panel, records, scoring, selective_screen
from .compact_m1 import CompactM1


def _score_handler(contig, state, phase, common, budget):
    store = CheckpointStore(common["root"], nthreads=budget)                # the worker's live lease
    file = store.contig(contig)
    candidate_panel = common["panel"]
    store.bind_identity(file, records.FLOORED_SCORES,
                        records.floored_scores_identity(store, file, candidate_panel))
    if file.has(records.FLOORED_SCORES):
        return None, None
    saved = store.run.get(records.SELECTIVE_SCREEN)
    compact = CompactM1.decode(saved["compact"])
    chromosome = tuple(saved["contigs"]).index(contig)
    one = compact.chromosome_reference(chromosome, common["eligibility"])
    prepared = records.decode_prepared(file.get(records.PREPARED))
    value = scoring.score_chromosome_selective(
        prepared, one, candidate_panel.trios, common["eligibility"], common["config"],
        execution.claim_threads)
    file.put(records.FLOORED_SCORES, records.encode_scores(value))
    return None, None


def score_prepared(store, contigs, eligibility, config, budget, *, method="certified"):
    """Screen, freeze the G4 panel/profile, then score and record M2 and exposure per contig; returns the panel.

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
    # Screen utilities child by child from the compact profile (no dense [C, N, N] expansion); the retained
    # parents are the compact profile's.
    utilities = panel.robust_screen_values(compact.values, compact.n_contigs, compact.n_samples, compact.markers,
                                           eligibility, config)
    candidate_panel = panel.retained_panel(compact.keep_reference(), utilities, eligibility)
    del utilities, compact
    estimates = [_scoring_bytes(store, c) for c in contigs]
    limit = execution.memory_worker_limit(estimates, len(contigs), label="PEDIGREE SELECTIVE M2")
    weights = [store.contig(c).info(records.PREPARED).stored_bytes for c in contigs]
    common = dict(root=store.root, panel=candidate_panel, config=config, eligibility=eligibility)
    with execution.ChromosomeExecutor(contigs, _score_handler, budget=budget, weights=weights,
                                      max_workers=limit, label="PEDIGREE SELECTIVE M2",
                                      task_bytes=estimates) as pool:
        pool.run("score", common)
    return candidate_panel


def load_scores(store, contigs):
    """Each contig's fixed-profile scores (``pedigree/floored_scores``), decoded one at a time."""
    for contig in contigs:
        yield records.decode_scores(store.contig(contig).get(records.FLOORED_SCORES))
