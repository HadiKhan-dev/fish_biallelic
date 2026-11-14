"""Collect chromosome reports in parallel, then publish one ordered writer per family.

Report completion never enters a scientific identity. Private CSV parts preserve
per-contig cell formatting and are removed by the run-level atomic writers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from ...engine.graph import ResourceRequest, ResultRef, Task, TaskKey
from .painting import contig_painting, write_painting
from .pair_sharing import contig_pair_sharing, write_pair_sharing
from .pedigree import write_pedigree
from .quality import contig_quality, write_quality
from .recombination import contig_recombination, write_recombination
from .relatedness import write_relatedness
from .tables import write_family_phase, write_founder_refit, write_truth_pedigree
from .ties import contig_ties, write_ties

if TYPE_CHECKING:
    from ...contracts.genetic_map import GeneticMapSet
    from ...results.reader import Results

__all__ = ["make_task", "run_tasks"]


def make_task(
    name: str,
    writer: Callable,
    args: tuple,
    *,
    kwargs: Mapping | None = None,
    contig: str | None = None,
    depends_on: Sequence[TaskKey] = (),
    working_set_bytes: int = 0,
) -> Task:
    """Declare an idempotent collector or writer with explicit prerequisites and one CPU."""
    return Task(
        TaskKey("report", "contig" if contig else "run", scope=contig or "run", round=name),
        writer,
        args=args,
        kwargs={} if kwargs is None else dict(kwargs),
        depends_on=tuple(depends_on),
        resources=ResourceRequest(working_set_bytes=working_set_bytes),
    )


def _relatedness(output: Path, results: Results):
    product = results.relatedness()
    return [] if product is None else write_relatedness(output, product, results.input_sample_ids)


def _pedigree(output: Path, results: Results, *, map_work=None):
    product = results.pedigree()
    return [] if product is None else write_pedigree(output, product, results.input_sample_ids, map_work=map_work)


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
    pair_sharing: GeneticMapSet | None = None,
) -> tuple[Task, ...]:
    """Declare collectors and gathers without probing products or traversing the store.

    Quality, painting and recombination dependencies follow results.contigs. Other
    keys are truth, relatedness, pedigree, refit, phase, complete and optional science.
    ``pair_sharing`` (the run's configured maps, given only when the output is selected) adds the pair-sharing
    collectors, one per painted contig, and their writer; they read paintings and feed nothing back.
    """
    tasks: list[Task] = []
    writer: Callable[..., list[Path]]
    for family, collector, writer in (
        ("quality", contig_quality, write_quality),
        ("painting", contig_painting, write_painting),
        ("recombination", contig_recombination, write_recombination),
    ):
        if family not in dependencies:
            continue
        readers = tuple(
            make_task(
                family,
                collector,
                (output, results, contig),
                contig=contig,
                depends_on=(dependency,),
                kwargs={"elapsed_seconds": (recombination_seconds or {}).get(contig)}
                if family == "recombination"
                else {},
            )
            for contig, dependency in zip(results.contigs, dependencies[family], strict=True)
        )
        tasks.extend(readers)
        tasks.append(
            make_task(
                family,
                writer,
                (output, tuple(ResultRef(task.key) for task in readers)),
                depends_on=tuple(task.key for task in readers),
                working_set_bytes=recombination_bytes if family == "recombination" else 0,
            )
        )
    if pair_sharing is not None and "painting" in dependencies:
        readers = tuple(
            make_task(
                "pair_sharing",
                contig_pair_sharing,
                (output, results, contig),
                contig=contig,
                depends_on=(dependency,),
                kwargs={"genetic_maps": pair_sharing},
            )
            for contig, dependency in zip(results.contigs, dependencies["painting"], strict=True)
        )
        tasks.extend(readers)
        tasks.append(
            make_task(
                "pair_sharing",
                write_pair_sharing,
                (output, results, tuple(ResultRef(task.key) for task in readers)),
                depends_on=tuple(task.key for task in readers),
            )
        )
    for name, writer in (
        ("truth", _truth),
        ("relatedness", _relatedness),
        ("pedigree", _pedigree),
        ("refit", _refit),
        ("phase", _phase),
        ("complete", write_ties),
    ):
        if name not in dependencies:
            continue
        kwargs = {}
        after = tuple(dependencies[name])
        if name == "complete":
            # A contig's tie sources run through painting and founder refit (results.tie_sources reads both, and
            # skips a stage not yet published). Refit depends on painting, so waiting for painting alone could
            # collect before refit publishes and silently drop its tie events.
            sources = (*dependencies.get("painting", ()), *dependencies.get("refit", ()))
            sources = sources or dependencies.get("science", after)
            readers = tuple(
                make_task(
                    "ties",
                    contig_ties,
                    (results, contig),
                    contig=contig,
                    depends_on=tuple(dict.fromkeys(key for key in sources if key.scope in {contig, "run"})),
                )
                for contig in results.contigs
            )
            tasks.extend(readers)
            kwargs = {
                "collected": tuple(ResultRef(task.key) for task in readers),
                "reverse_conventions": reverse_conventions,
            }
            after += tuple(task.key for task in readers)
        tasks.append(
            make_task(
                "ties" if name == "complete" else name,
                writer,
                (output, results, results.contigs) if name == "complete" else (output, results),
                kwargs=kwargs,
                depends_on=after,
            )
        )
    return tuple(tasks)
