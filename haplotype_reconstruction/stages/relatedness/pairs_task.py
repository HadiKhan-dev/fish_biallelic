"""Execute the independent relatedness/pairs task under one runtime CPU team.

Resume is by derivation before reading evidence. The output selection is outside
scientific parameters; graph composition schedules the optional pairs task.
"""

from __future__ import annotations

from pathlib import Path

from ...engine.runtime import task_team
from ...science.relatedness.ibd import accumulate_pairwise_ibd, finish_pairwise_ibd, new_pairwise_ibd
from ...science.relatedness.ibd_parameters import PairwiseIbdParameters
from ...science.relatedness.parameters import AlleleFrequencyParameters
from ...store.protocol import claim
from ...store.store import CheckpointStore
from ..framework import ALL_CORES, RecordKey, ResourceRequest, Task, TaskKey
from . import pairs_records as records
from .evidence_pass import EvidenceArrays, EvidenceInput, evidence_pass

__all__ = ["run", "tasks"]
_DEFAULT_FREQUENCY = AlleleFrequencyParameters()


def run(
    root: str | Path,
    inputs: tuple[EvidenceInput, ...],
    sample_ids: tuple[str, ...],
    parameters: PairwiseIbdParameters,
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
        with task_team(team):
            state = new_pairwise_ibd(len(sample_ids))

            def consume(item: EvidenceInput, evidence: EvidenceArrays) -> None:
                accumulate_pairwise_ibd(state, evidence, item.genetic_map, parameters, frequency)

            evidence_pass(root, inputs, sample_ids, consume, team=team)
            product = finish_pairwise_ibd(state)
        records.write(file, product, derivation)
    return derivation


def tasks(
    root: str | Path,
    inputs: tuple[EvidenceInput, ...],
    sample_ids: tuple[str, ...],
    parameters: PairwiseIbdParameters,
    frequency: AlleleFrequencyParameters = _DEFAULT_FREQUENCY,
    *,
    maximum_cores: int = ALL_CORES,
    depends_on: tuple[TaskKey, ...] = (),
    reverse: bool = False,
    selected: bool = False,
) -> tuple[Task, ...]:
    """Declare one independent run task only when pairs are selected."""
    if not selected:
        return ()
    return (
        Task(
            key=TaskKey("relatedness", "pairs"),
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
