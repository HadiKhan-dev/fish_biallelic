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
    inputs = ChromosomeInput(
        task.contig,
        reader.simulation_truth(task.contig),
        {stage: reader.stage_panels(task.contig, stage) for stage in task.founder_stages},
        phase,
        recombination,
    )
    return evaluate_chromosome(inputs, task.names, task.requested, parent_counts=task.parent_counts)
