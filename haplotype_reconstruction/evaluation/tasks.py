"""Describe chromosome evaluation inputs independently of execution and storage.

OU31 evaluates one chromosome per task. The application schedules these plain
records and supplies results readers; this layer imports neither engine nor stages.
The memory estimate is a report, never a scientific admission threshold.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypedDict, cast

import numpy as np

from ..contracts.pedigree import Pedigree
from ..contracts.phase import PhasedSamples
from ..contracts.recombination import Recombination
from ..contracts.roles import record_field
from ..results.panels import StagePanel
from ..simulation.truth import ContigTruth, SimulationManifest
from .parameters import TASK_ALLOWANCE_BYTES

__all__ = [
    "StagePanel",
    "ChromosomeInput",
    "ChromosomeEvaluation",
    "EvaluationTask",
    "EvaluationReader",
    "evaluate_task",
    "plan_evaluation",
]


MetricValue = str | int | float | None | list[int] | dict[str, int]
MetricRow = dict[str, MetricValue]


class PhaseArrays(TypedDict):
    """Released observations on the phase product's own sample and site axes."""

    sample_ids: tuple[str, ...]
    positions: np.ndarray
    component_ids: np.ndarray
    calls: np.ndarray
    frame_undetermined: np.ndarray | None
    phase_breaks: np.ndarray | None
    phase_break_offsets: np.ndarray | None


@dataclass(frozen=True)
class ChromosomeInput:
    """Only the original-axis products selected for one contig, plus its truth.

    Founder stages preserve pipeline/request order; phase is the released product's
    named arrays, recombination supplies the interval views of its contract.
    """

    contig: str
    truth: ContigTruth
    founders: Mapping[str, Sequence[StagePanel]]
    phase: PhaseArrays | None = None
    recombination: Recombination | None = None


@dataclass(frozen=True)
class ChromosomeEvaluation:
    """Sparse rows whose absent keys remain absent in JSON and CSV rendering."""

    chromosome: MetricRow = record_field("report", units="1", dtype="json")
    founders: list[MetricRow] = record_field("report", units="1", dtype="json")
    components: list[MetricRow] = record_field("report", units="1", dtype="json")
    matches: list[MetricRow] = record_field("report", units="1", dtype="json")
    phase_groups: list[MetricRow] = record_field("report", units="1", dtype="json")
    skipped: list[str] = record_field("report", units="1", dtype="json")

    def parts(self) -> tuple[Any, ...]:
        """Return the carried collection order; callers own these fresh row objects."""
        return self.chromosome, self.founders, self.components, self.matches, self.phase_groups, self.skipped


class EvaluationReader(Protocol):
    """Read selected original-axis products without exposing stage implementations."""

    root: Path
    checkpoints: Path

    @property
    def contigs(self) -> tuple[str, ...]: ...

    def simulation_manifest(self) -> SimulationManifest | None: ...
    def simulation_truth(self, contig: str) -> ContigTruth: ...
    def stage_present(self, contig: str, stage: str) -> bool: ...
    def stage_panels(self, contig: str, stage: str) -> Sequence[StagePanel]: ...
    def pedigree(self) -> Pedigree | None: ...
    def phase(self, contig: str) -> PhasedSamples | PhaseArrays | None: ...
    def recombination(self, contig: str) -> Recombination | None: ...


@dataclass(frozen=True)
class EvaluationTask:
    """A chromosome selection in manifest order; the caller gives its thread team."""

    contig: str
    names: tuple[str, ...]
    founder_stages: tuple[str, ...]
    requested: tuple[str, ...]
    parent_counts: np.ndarray | None = None

    def working_set_bytes(self, truth_bytes: int, product_bytes: Sequence[int]) -> int:
        """Report the carried decoded-record estimate with its fixed worker allowance."""
        return (truth_bytes + max(product_bytes, default=0)) * 5 // 4 + TASK_ALLOWANCE_BYTES


def evaluate_task(task: EvaluationTask, reader: EvaluationReader) -> ChromosomeEvaluation:
    """Load only selected products and score a contig; failures propagate to the caller."""
    from .evaluate import evaluate_chromosome

    automatic = task.requested in (("available",), ("all",))
    raw_phase = reader.phase(task.contig) if automatic or "family_phase" in task.requested else None
    phase: PhaseArrays | None
    if raw_phase is not None and not isinstance(raw_phase, Mapping):
        phase = cast(
            PhaseArrays,
            {
                name: getattr(raw_phase, name)
                for name in (
                    "sample_ids",
                    "positions",
                    "component_ids",
                    "calls",
                    "frame_undetermined",
                    "phase_breaks",
                    "phase_break_offsets",
                )
            },
        )
    else:
        phase = cast(PhaseArrays | None, raw_phase)
    recombination = reader.recombination(task.contig) if automatic or "recombination" in task.requested else None
    truth = reader.simulation_truth(task.contig)
    founders = {stage: reader.stage_panels(task.contig, stage) for stage in task.founder_stages}
    if getattr(reader, "truth_sidecar", None) is not None:
        return _evaluate_projected(task, reader, truth, founders, phase, recombination)
    inputs = ChromosomeInput(
        task.contig,
        truth,
        founders,
        phase,
        recombination,
    )
    return evaluate_chromosome(inputs, task.names, task.requested, parent_counts=task.parent_counts)


class _TruthEvaluationReader:
    """Give the native planner a sidecar pedigree without inventing run truth."""

    def __init__(self, reader, truth):
        from dataclasses import replace

        from ..core.errors import InputError

        self.reader, self.truth_sidecar = reader, truth
        self.root, self.checkpoints, self.contigs = reader.root, reader.checkpoints, reader.contigs
        names = tuple(reader.manifest["sample_ids"])
        manifest = truth.manifest
        if manifest.observed is None:
            raise InputError("complete observation truth has no observed pedigree")
        by_name = {name: i for i, name in enumerate(manifest.sample_ids)}
        if len(set(names)) != len(names) or set(names) - set(by_name):
            raise InputError("run sample axis differs from observation truth")
        self._manifest = replace(
            manifest,
            sample_ids=names,
            observed=manifest.observed.take(np.asarray([by_name[n] for n in names], np.int64)),
        )

    def __getattr__(self, name):
        reader = self.__dict__.get("reader")
        if reader is None:
            raise AttributeError(name)
        return getattr(reader, name)

    def simulation_manifest(self):
        return self._manifest

    def simulation_truth(self, contig):
        return self.truth_sidecar.truth(contig)


def plan_evaluation(reader, *, truth=None, stages=("available",), contigs=None):
    """Plan native or VCF-route evaluation, returning the reader used by tasks.

    A run without a simulation manifest requires an explicitly bound truth
    sidecar. The native planner still validates all requested stage publications.
    Callers dispatch the returned tasks with the returned evaluation reader.
    """
    from .evaluate import plan_evaluation as native_plan

    if truth is None:
        report, tasks = native_plan(reader, stages=stages, contigs=contigs)
        return report, tasks, reader
    from ..results.truth import TruthSidecar

    sidecar = truth if isinstance(truth, TruthSidecar) else TruthSidecar(truth)
    sidecar.validate_run(reader)
    evaluation_reader = _TruthEvaluationReader(reader, sidecar)
    report, tasks = native_plan(evaluation_reader, stages=stages, contigs=contigs)
    report["observation_truth"] = sidecar.description()
    return report, tasks, evaluation_reader


def _evaluate_projected(task, reader, truth, founders, phase, recombination):
    """Use two independent truth views without changing any scoring arithmetic."""
    from ..results.truth import project_truth
    from .evaluate import evaluate_chromosome

    # The founder scorer sees the immutable truth positions, founder axis and
    # sampled-homologue painting, including markers and samples absent after QC.
    full = evaluate_chromosome(
        ChromosomeInput(task.contig, truth, founders), truth.sample_ids, task.founder_stages or ("pedigree",)
    )
    row, phase_groups, skipped = dict(full.chromosome), [], []
    axes = reader.observation_axes(task.contig)
    row.update(
        truth_sites=len(truth.positions),
        sites=len(axes.positions),
        qc_removed_sites=len(truth.positions) - len(axes.positions),
        qc_removed_positions=np.setdiff1d(truth.positions, axes.positions).tolist(),
    )
    automatic = task.requested in (("available",), ("all",))
    if not automatic:
        from ..core.errors import InputError

        if "family_phase" in task.requested and phase is None:
            raise InputError(f"{task.contig}: no completed family phase")
        if "recombination" in task.requested and recombination is None:
            raise InputError(f"{task.contig}: no completed recombination map")
    if phase is not None:
        names = tuple(phase["sample_ids"])
        projected = project_truth(truth, phase["positions"], names)
        counts = None
        if task.parent_counts is not None:
            by_name = dict(zip(task.names, task.parent_counts, strict=True))
            counts = np.asarray([by_name[name] for name in names], np.int64)
        scored = evaluate_chromosome(
            ChromosomeInput(task.contig, projected, {}, phase), names, ("family_phase",), parent_counts=counts
        )
        row.update(scored.chromosome)
        phase_groups.extend(scored.phase_groups)
    elif automatic:
        skipped.append(f"{task.contig}/family_phase")
    if recombination is not None:
        names = tuple(recombination.sample_ids)
        projected = project_truth(truth, axes.positions, names)
        scored = evaluate_chromosome(
            ChromosomeInput(task.contig, projected, {}, recombination=recombination), names, ("recombination",)
        )
        # The phase product owns the scored site count when it exists.
        row.update({key: value for key, value in scored.chromosome.items() if key != "sites"})
    elif automatic:
        skipped.append(f"{task.contig}/recombination")
    return ChromosomeEvaluation(row, full.founders, full.components, full.matches, phase_groups, skipped)
