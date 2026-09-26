"""Chromosome-local preparation and scoring with a genome-wide panel barrier.

Only small summaries cross process boundaries in checkpointed runs. Prepared
arrays, source projections and reusable M0/M1 screens stay with their chromosome
worker until the common candidate panel arrives for M2 scoring.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import time

import numpy as np

from ..core import runtime
from ..core import chromosome_parallel as scheduling
from . import components, eligibility


@dataclass(frozen=True)
class StoredChromosome:
    root: str
    contig: str
    sample_ids: tuple[str, ...]
    preparation_identity: dict
    painting_stage: str
    preparation_stage: str

    def load(self):
        from . import pipeline

        store = runtime.CheckpointStore(self.root, nthreads=scheduling.current_threads())
        painting = store.load_contig(self.painting_stage, self.contig)
        store.nthreads = scheduling.current_threads()
        checkpoint = pipeline._validate_prepared_checkpoint(
            store.load_contig(self.preparation_stage, self.contig),
            expected_contig=self.contig, expected_sample_ids=self.sample_ids,
            expected_preparation_identity=self.preparation_identity,
            expected_painting=painting)
        return checkpoint.prepared_chromosome, pipeline._checkpoint_source_identity(checkpoint)

    @property
    def weight(self):
        from ..core.checkpoints import contig_path
        return Path(contig_path(self.root, self.preparation_stage, self.contig)).stat().st_size


@dataclass(frozen=True)
class StoredPedigree:
    # Layout contains run-level metadata only; workers load its chromosomes.
    layout: components.PreparedPedigree
    chromosomes: tuple[StoredChromosome, ...]


@dataclass(frozen=True)
class ScreenSummary:
    contig: str
    source_identity: object
    component_count: int
    omitted_reason: str | None
    pair_scores: np.ndarray | None
    informative_markers: int
    checkpoint_source_identity: str | None


def prepare_chromosome(task, state, phase, common):
    """Prepare/save one independently owned chromosome, returning metadata only."""
    from . import pipeline

    store = runtime.CheckpointStore(common['root'], nthreads=scheduling.current_threads())
    result = pipeline._prepare_one_contig(store, task, **common['arguments'])
    return None, result


def score_chromosome(task, state, phase, common):
    from . import likelihoods

    if phase == 'screen':
        chromosome, source = task.load() if isinstance(task, StoredChromosome) else (task, None)
        layout = common['layout']
        if chromosome.sample_ids != layout.sample_ids or chromosome.source_mode != layout.source_mode:
            raise ValueError(f'{chromosome.contig}: preparation sample order/source mode mismatch')
        screens = []
        for component, exponent in zip(chromosome.components, chromosome.information_exponents):
            scheduling.current_threads()
            screens.append(likelihoods._projected_m1_screen(
                component, exponent, common['settings'],
                common['eligible_children'], common['eligible_parents']))
        summary = ScreenSummary(
            chromosome.contig, chromosome.source_identity, chromosome.component_count,
            chromosome.omitted_reason,
            np.sum(np.stack([s.one_observed for s in screens]), axis=0) if screens else None,
            sum(c.cache.informative_markers for c in chromosome.components), source)
        return (chromosome, tuple(screens)), summary

    if phase != 'score':
        raise ValueError(f'unknown pedigree scoring phase: {phase}')
    chromosome, screens = state
    if not chromosome.components:
        return None, None
    produced = None

    def produce():
        nonlocal produced
        scheduling.current_threads()
        produced = likelihoods._score_prepared_chromosome(
            chromosome, common['trios'], common['settings'],
            common['eligible_children'], common['eligible_parents'],
            ragged_screen_scores=screens)
        return likelihoods._compact_chromosome_evidence(produced)

    request = common['requests'][chromosome.contig]
    callback = common['callback']
    score = produce() if callback is None else callback(request, produce)
    score = likelihoods._validate_scored_chromosome(
        score, contig=chromosome.contig, n_samples=len(chromosome.sample_ids),
        n_trios=len(common['trios']))
    # Production retains the lean scores, not the large runtime factor tensors.
    return None, (score, produced if common['retain_runtime'] else None)


def score_sources(layout, tasks, *, settings, parent_eligibility=None, top_k=20,
                  adaptive_initial_top_k=None, anchor_k=5, use_anchor_union=False,
                  mismatch_penalty, evidence_identity=None, callback=None,
                  n_workers=None, retain_runtime=True):
    """Screen concurrently, select one genome-wide panel, then score concurrently."""
    from . import likelihoods
    from .cache import ChromosomeScoreCache

    started = time.perf_counter()
    resolved = eligibility._resolve_parent_eligibility(parent_eligibility, layout.sample_ids)
    weights = [task.weight if isinstance(task, StoredChromosome) else
               sum(component.cache.informative_markers for component in task.components)
               for task in tasks]
    # Arbitrary user callbacks retain their caller-process semantics. The
    # production cache is a module-level, picklable chromosome-local callback.
    if callback is not None and not isinstance(callback, ChromosomeScoreCache):
        n_workers = 1
    common = dict(layout=layout, settings=settings,
                  eligible_children=resolved.eligible_children,
                  eligible_parents=resolved.eligible_parents)
    # The in-memory public API must not broadcast all chromosome tensors as
    # common metadata to every worker.
    common['layout'] = replace(layout, chromosomes=())
    with scheduling.ChromosomeExecutor(tasks, score_chromosome,
                                       n_workers=n_workers, weights=weights) as executor:
        summaries = executor.run('screen', common)
        informative = [value for value in summaries if value.pair_scores is not None]
        if not informative:
            raise ValueError('no physical chromosome contains observed nonuniform evidence')
        omissions = tuple(layout.omitted_chromosomes) + tuple(
            components.OmittedPaintingChromosome(value.contig, value.component_count,
                value.omitted_reason or 'no_component_evidence')
            for value in summaries if value.pair_scores is None)
        pair_score_array = np.stack([value.pair_scores for value in informative])
        marker_counts = np.asarray([value.informative_markers for value in informative], dtype=np.float64)
        parent_scores = likelihoods.pedigree_candidates._robust_parent_screen(
            pair_score_array, marker_counts, settings, resolved)
        trios, panel_diagnostics = likelihoods._adaptive_trio_panel(
            pair_score_array, marker_counts, parent_scores, top_k, anchor_k,
            bool(use_anchor_union), resolved, settings, adaptive_initial_top_k)
        score_identity = components._parent_state_score_identity(
            layout, settings, resolved, trios, top_k=top_k,
            adaptive_initial_top_k=adaptive_initial_top_k, anchor_k=anchor_k,
            use_anchor_union=bool(use_anchor_union), mismatch_penalty=float(mismatch_penalty),
            external_identity=evidence_identity, chromosome_sources=informative,
            omitted_chromosomes=omissions)
        score_identity['likelihood_recipe'] = 'balanced-tempered-and-full-forward-v1'
        run_digest = components._identity_digest(score_identity)
        requests = {}
        for value in informative:
            identity = dict(run_score_identity_sha256=run_digest, contig=value.contig,
                            source_identity=value.source_identity)
            requests[value.contig] = components.ChromosomeEvidenceRequest(
                value.contig, score_identity, components._identity_digest(identity), len(trios))
        common.update(trios=trios, requests=requests, callback=callback, retain_runtime=retain_runtime)
        values = [value for value in executor.run('score', common) if value is not None]

    scores, runtime_results = zip(*values)
    return components.ScoredParentStateEvidence(
        sample_ids=layout.sample_ids, contig_names=tuple(value.contig for value in informative),
        chromosomes=tuple(scores), trios=trios, parent_screen_scores=parent_scores,
        omitted_chromosomes=omissions, parent_panel_diagnostics=panel_diagnostics,
        source_mode=layout.source_mode, score_identity=score_identity,
        input_preparation_seconds=float(layout.input_preparation_seconds),
        screening_and_scoring_seconds=time.perf_counter()-started,
        runtime_chromosome_results=(tuple(runtime_results)
            if all(value is not None for value in runtime_results) else None),
    ), tuple(value.checkpoint_source_identity for value in summaries)
