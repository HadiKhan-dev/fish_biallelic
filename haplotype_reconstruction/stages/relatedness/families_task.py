"""Execute the independent relatedness/families task under one runtime CPU team.

Resume is by derivation before reading evidence. The output selection is outside
scientific parameters; graph composition schedules the optional pairs task.
"""

from __future__ import annotations

from pathlib import Path

from ...engine.runtime import thread_team
from ...science.relatedness.parameters import AlleleFrequencyParameters, FamilyParameters
from ...science.relatedness.run import concatenate_terms, contig_terms, fit_families
from ...store.protocol import claim
from ...store.store import CheckpointStore
from ..framework import RecordKey, ResourceRequest, Task, TaskKey
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
        with thread_team(team):
            terms = evidence_pass(
                root,
                inputs,
                sample_ids,
                lambda item, evidence: contig_terms(evidence, item.genetic_map, parameters, frequency),
                team=team,
            )
            combined = concatenate_terms(sample_ids, tuple(item.contig for item in inputs), terms)
            product = fit_families(combined, parameters)
        records.write(file, product, derivation)
    return derivation


def tasks(
    root: str | Path,
    inputs: tuple[EvidenceInput, ...],
    sample_ids: tuple[str, ...],
    parameters: FamilyParameters,
    frequency: AlleleFrequencyParameters = _DEFAULT_FREQUENCY,
    *,
    maximum_cores: int = 16,
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
