"""Execute the independent relatedness/families task under one runtime CPU team.

Resume is by derivation before reading evidence. The output selection is outside
scientific parameters; graph composition schedules the optional pairs task.
"""

from __future__ import annotations

from pathlib import Path

from ...engine.runtime import task_team
from ...science.relatedness.parameters import AlleleFrequencyParameters, FamilyParameters
from ...science.relatedness.run import concatenate_terms, contig_terms, fit_families
from ...store.protocol import claim
from ...store.store import CheckpointStore
from ..framework import ALL_CORES, RecordKey, ResourceRequest, Task, TaskKey
from . import families_records as records
from .evidence_pass import EvidenceInput, evidence_pass

__all__ = ["run", "tasks"]
_DEFAULT_FREQUENCY = AlleleFrequencyParameters()


def run(
    root: str | Path,
    inputs: tuple[EvidenceInput, ...],
    sample_ids: tuple[str, ...],
    parameters: FamilyParameters,
    frequency: AlleleFrequencyParameters = _DEFAULT_FREQUENCY,
    *,
    reverse: bool = False,
    team: int = 1,
) -> str:
    """Compute/publish once or reuse the same scientific record; return its derivation."""
    derivation = records.identity(inputs, sample_ids, parameters, frequency, reverse=reverse)
    file = CheckpointStore(root, nthreads=team).run
    with claim(file, records.KEY, derivation) as compute:
        if not compute:
            return derivation
        from ...engine.scheduler import claim_threads, current_budget

        def terms_of(item, evidence):
            claim_threads()  # each contig's terms on the task's current share
            # Release 4.10: the allele frequencies' site windows on that share (decoded one after another before).
            budget = current_budget()
            return contig_terms(
                evidence,
                item.genetic_map,
                parameters,
                frequency,
                map_work=lambda work, items: budget.thread_map(work, items, label="relatedness frequency windows"),
            )

        with task_team(team):
            terms = evidence_pass(root, inputs, sample_ids, terms_of, team=team)
            combined = concatenate_terms(sample_ids, tuple(item.contig for item in inputs), terms)
            # The task starts beside discovery with the CPUs it was given (often one); every search step re-reads its
            # share, so it grows into the CPUs that finished tasks free (as painting does at each batch).
            product = fit_families(combined, parameters, claim_threads=claim_threads)
        records.write(file, product, derivation)
    return derivation


def tasks(
    root: str | Path,
    inputs: tuple[EvidenceInput, ...],
    sample_ids: tuple[str, ...],
    parameters: FamilyParameters,
    frequency: AlleleFrequencyParameters = _DEFAULT_FREQUENCY,
    *,
    maximum_cores: int = ALL_CORES,
    depends_on: tuple[TaskKey, ...] = (),
    reverse: bool = False,
) -> tuple[Task, ...]:
    """Declare one independent run task."""
    return (
        Task(
            key=TaskKey("relatedness", "families"),
            function=run,
            args=(str(root), inputs, sample_ids, parameters, frequency),
            kwargs={"reverse": reverse},
            depends_on=depends_on,
            inputs=tuple(RecordKey(item.contig, "evidence") for item in inputs),
            outputs=(RecordKey("run", records.KEY),),
            resources=ResourceRequest(min_cores=1, max_cores=maximum_cores),
            thread_argument="team",
        ),
    )
