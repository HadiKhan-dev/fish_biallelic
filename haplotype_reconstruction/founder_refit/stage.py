"""Founder refit stage: re-call each founder row's alleles from the samples painted with it.

Assembly releases chromosome rows as concatenations of block-local discovery
rows whose alleles are never re-called. Where block-level model selection
merged two founders that differ at a few sites, the row of the unrepresented
founder borrows the other founder's alleles there, for all of its carriers.
Given the painting, this stage performs one conditional M-step
(founder_refit/refit.py) on the released alleles (assembly calls plus
completion fills, after label continuity). Row paths, founder counts and the
painting are unchanged; painting and pedigree keep reading the frozen
inference snapshot, so nothing feeds back into carrier-state evidence. Samples
the sample-quality QC keeps out of a contig's founder construction are left out
here too (the evidence's founder view). Family phase decodes the refit
inference alleles; evaluation and export read the refit released alleles.
"""
from __future__ import annotations

from .. import execution
from ..evidence import records as evidence_records
from ..painting import records as painting_records
from ..store import CheckpointStore
from . import records
from .config import FounderRefitConfig
from .refit import refit_chromosome, summary


def _refit_contig(contig, state, phase, common, budget):
    """Chromosome-worker handler: load, refit and record one chromosome (resumable)."""
    store = CheckpointStore(common["root"], nthreads=budget)                # the worker's live lease
    file = store.contig(contig)
    # Bound before the completeness test (as painting): a finished contig is reused only under the same refit
    # settings, upstream records and code; anything else is refused ("use a new output root").
    records.bind_identity(store, file, common["config"])
    if not records.complete(file):
        painting_records.require_evidence_binding(file)
        painting = painting_records.load(file, panels=True, emissions=False)
        evidence = evidence_records.load(file)
        execution.claim_threads()
        refit = refit_chromosome(painting, evidence, common["config"])
        del painting, evidence
        records.write(file, refit)
    else:
        refit = records.load(file)
    result = summary(refit)
    excluded = result["excluded_samples"]
    print(f"[founder refit {contig}] {result['changed_alleles']} of {result['called_alleles']} "
          f"called alleles changed at {result['changed_sites']} sites"
          + (f"; sample-quality QC excludes the reads of {', '.join(excluded)}" if excluded
             else ""), flush=True)
    del refit
    execution.release_memory()
    return None, result


def run(store, contigs, config: FounderRefitConfig, budget) -> list[dict]:
    """Refit every contig (one chromosome per worker); returns the per-contig summaries.

    Memory per chromosome (each task's reservation, execution.memory): the decoded evidence (read in place),
    the founder view's copy of its likelihoods when sample QC excludes samples (counted always, about three
    quarters of the decoded evidence), twice the painting and 2 GiB of working space.
    """
    contigs = tuple(map(str, contigs))
    store.require(painting_records.KEY, contigs)
    estimates = []
    for contig in contigs:
        file = store.contig(contig)
        estimates.append(file.info(evidence_records.EVIDENCE).decoded_bytes * 2
                         + 2 * file.info(painting_records.KEY).decoded_bytes + (2 << 30))
    workers = budget.threads()
    limit = execution.memory_worker_limit(estimates, workers, label="FOUNDER REFIT")
    with execution.ChromosomeExecutor(contigs, _refit_contig, budget=budget, weights=estimates,
                                      max_workers=limit, label="FOUNDER REFIT", task_bytes=estimates) as executor:
        summaries = executor.run("refit", {"root": str(store.root), "nthreads": store.nthreads,
                                           "config": config})
    store.require(records.KEY, contigs)
    store.mark_stage_complete(records.STAGE)
    return summaries
