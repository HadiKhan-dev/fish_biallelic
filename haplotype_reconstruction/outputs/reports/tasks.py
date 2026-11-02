"""Declare report publication after its typed products, with ties after run completion.

Owns OU11 scheduling declarations only. Engine/interface executes these tasks;
report completion never enters a scientific identity. Every invocation republishes.
Method: docs/methods/outputs.md; baseline pipeline/outputs.py and driver call sites.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from ...engine.graph import RecordKey, ResourceRequest, ResultRef, Task, TaskKey
from .painting import write_painting
from .pedigree import write_pedigree
from .quality import write_quality
from .recombination import write_recombination
from .relatedness import write_relatedness
from .tables import write_family_phase, write_founder_refit, write_truth_pedigree
from .ties import contig_ties, write_ties

if TYPE_CHECKING:
    from ...results.reader import Results

__all__ = ["make_task", "contig_tasks", "run_tasks"]


def make_task(
    name: str,
    writer: Callable,
    args: tuple,
    *,
    kwargs: Mapping | None = None,
    contig: str | None = None,
    depends_on: Sequence[TaskKey] = (),
    inputs: Sequence[RecordKey] = (),
    working_set_bytes: int = 0,
) -> Task:
    """Declare an idempotent file writer with explicit prerequisites and one CPU."""
    return Task(
        TaskKey("report", "contig" if contig else "run", scope=contig or "run", round=name),
        writer,
        args=args,
        kwargs={} if kwargs is None else dict(kwargs),
        depends_on=tuple(depends_on),
        inputs=tuple(inputs),
        resources=ResourceRequest(working_set_bytes=working_set_bytes),
    )


def contig_tasks(
    results: Results,
    output: Path,
    contig: str,
    *,
    evidence: TaskKey | None = None,
    painting: TaskKey | None = None,
    painting_bytes: int = 0,
) -> tuple[Task, ...]:
    """Write quality after evidence and painting tables after painting, including reused products."""
    tasks = []
    if evidence is not None:
        tasks.append(
            make_task(
                "quality",
                write_quality,
                (output, results, contig),
                contig=contig,
                depends_on=(evidence,),
                inputs=(RecordKey(contig, "evidence"),),
            )
        )
    if painting is not None:
        tasks.append(
            make_task(
                "painting",
                write_painting,
                (output, results, contig),
                contig=contig,
                depends_on=(painting,),
                inputs=(RecordKey(contig, "painting"),),
                working_set_bytes=painting_bytes,
            )
        )
    return tuple(tasks)


def _relatedness(output: Path, results: Results):
    product = results.relatedness()
    return [] if product is None else write_relatedness(output, product, results.input_sample_ids)


def _pedigree(output: Path, results: Results):
    product = results.pedigree()
    return [] if product is None else write_pedigree(output, product, results.input_sample_ids)


def _refit(output: Path, results: Results):
    return write_founder_refit(output, results.refit_summaries())


def _phase(output: Path, results: Results):
    return write_family_phase(output, results.phase_summaries())


def _truth(output: Path, results: Results):
    return write_truth_pedigree(output, results.simulation_manifest())


def run_tasks(
    results: Results,
    output: Path,
    dependencies: Mapping[str, Sequence[TaskKey]],
    *,
    reverse_conventions: bool = False,
    recombination_bytes: int = 0,
    recombination_seconds: Mapping[str, float] | None = None,
) -> tuple[Task, ...]:
    """Declare available gather writers in baseline order; ties require the completed-run dependency.

    Keys are truth, relatedness, pedigree, refit, phase, recombination, complete.
    The interface supplies dependencies only for available products, including reused
    ones; no reader probes or store traversal happens while constructing the graph.
    """
    writers = (
        ("truth", _truth),
        ("relatedness", _relatedness),
        ("pedigree", _pedigree),
        ("refit", _refit),
        ("phase", _phase),
        ("recombination", write_recombination),
        ("complete", write_ties),
    )
    tasks: list[Task] = []
    for name, writer in writers:
        if name not in dependencies:
            continue
        kwargs: dict[str, object] = {"reverse_conventions": reverse_conventions} if name == "complete" else {}
        if name == "complete":
            # Each contig's tie sources are read by a task of its own (they are independent; read in turn they made
            # one serial step of the whole run); the report merges them in contig order.
            readers = tuple(
                make_task("ties", contig_ties, (results, contig), contig=contig, depends_on=dependencies[name])
                for contig in results.contigs
            )
            tasks += readers
            kwargs["collected"] = tuple(ResultRef(task.key) for task in readers)
            dependencies = {**dependencies, name: (*dependencies[name], *(task.key for task in readers))}
        if name == "recombination":
            kwargs = {"elapsed_seconds": recombination_seconds}
        tasks.append(
            make_task(
                "ties" if name == "complete" else name,
                writer,
                (output, results, results.contigs) if name in ("recombination", "complete") else (output, results),
                kwargs=kwargs,
                depends_on=dependencies[name],
                working_set_bytes=recombination_bytes if name == "recombination" else 0,
            )
        )
    return tuple(tasks)
