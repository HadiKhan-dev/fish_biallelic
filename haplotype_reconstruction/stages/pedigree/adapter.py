"""Run the pedigree decision seam on supplied evidence with an explicitly bounded CPU team.

Owns execution setup only. Scoring outputs and exclusion evidence arrive through
PedigreeInput; the returned PedigreeResult is ready for publication by the framework.
Large calibration searches and the independent resampling fits run as engine subtasks
(stages.pedigree.execution).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numba

from ...contracts.pedigree import CalibrationRecord, PedigreeInput, PedigreeResult
from ...engine.workers import ResidentRef, release_resident, release_shared_value, resident_value, retain
from ...science.pedigree import api, resampling
from ...science.pedigree import run as science
from ...science.pedigree.parameters import PedigreeConfig
from ...science.pedigree.resampling import FitResult
from . import execution
from .execution import _NumbaConfiguration

if TYPE_CHECKING:
    from ..framework import Expansion, Task

__all__ = ["run", "task"]

_AFFINITY = "pedigree/decision"


def _cache(cache_root: str | None) -> None:
    cache_root = cache_root or cast(_NumbaConfiguration, numba.config).CACHE_DIR or None
    if cache_root:
        Path(cache_root).mkdir(parents=True, exist_ok=True)
        cast(_NumbaConfiguration, numba.config).CACHE_DIR = str(cache_root)
        from ...core.kernels import enable_caches

        enable_caches()


def decide(
    inputs: PedigreeInput, parameters: PedigreeConfig, *, workers: int = 1, cache_root: str | None = None
) -> PedigreeResult | Expansion:
    """The pilot task's action: the full fit here (its scale search in rounds when large), then its resampling
    fits as independent subtasks."""
    _cache(cache_root)
    science.precheck(len(inputs.scores.sample_ids), len(inputs.contigs), parameters, basis="configured")
    previous = numba.get_num_threads()
    numba.set_num_threads(1)
    try:
        started = science.start_full_fit(inputs.scores, parameters)
    finally:
        numba.set_num_threads(previous)
    return execution.calibrate(
        started.calibration,
        workers,
        stage="pedigree",
        affinity=_AFFINITY,
        then=_calibrated,
        args=(inputs, parameters, started),
    )


def _calibrated(
    fitted: tuple[float, CalibrationRecord],
    inputs: PedigreeInput,
    parameters: PedigreeConfig,
    started: Any,
    *,
    workers: int,
) -> PedigreeResult | Expansion:
    """Finish the full fit with its evidence scale, then resample serially or as independent subtasks."""
    from ..framework import Expansion, ResourceRequest, ResultRef, Task, TaskKey

    previous = numba.get_num_threads()
    numba.set_num_threads(1)
    try:
        state = science.finish_full_fit(started, fitted, parameters)
        planned = resampling.plan(state.inputs, parameters, workers)
        if planned.worker_count == 1:
            results = (resampling.evaluate_task(state.inputs, t) for t in planned.tasks)
            decided = science.complete_decision(state, resampling.combine(state.inputs, planned, results), parameters)
            return api.release_supplied(decided, inputs, parameters)
    finally:
        numba.set_num_threads(previous)
    shared, subtasks = execution.fit_subtasks(state.inputs, planned, "pedigree", copy_threads=workers)
    retained = retain((inputs, parameters, state, planned, shared), "fit")
    gather = Task(
        TaskKey("pedigree", "resampled"),
        _release,
        args=(retained, tuple(ResultRef(t.key) for t in subtasks)),
        depends_on=tuple(t.key for t in subtasks),
        affinity=_AFFINITY,
        resources=ResourceRequest(max_cores=workers),
    )
    return Expansion(subtasks, gather)


def _release(reference: ResidentRef, fitted: tuple[tuple[FitResult, ...], ...]) -> PedigreeResult:
    """Reduce the fits in the plan's order on the affinity worker, then release the supplied evidence.

    ``fitted`` holds one tuple of results per work-sized batch (execution.fit_batches), in plan order.
    """
    inputs, parameters, state, planned, shared = resident_value(reference)
    try:
        release_shared_value(shared)
        previous = numba.get_num_threads()
        numba.set_num_threads(1)
        try:
            results = (result for batch in fitted for result in batch)
            decided = science.complete_decision(state, resampling.combine(state.inputs, planned, results), parameters)
            return api.release_supplied(decided, inputs, parameters)
        finally:
            numba.set_num_threads(previous)
    finally:
        release_resident(reference)


def run(
    inputs: PedigreeInput, parameters: PedigreeConfig, *, workers: int = 1, cache_root: str | None = None
) -> PedigreeResult:
    """Fit and release already-scored evidence on an engine graph of at most ``workers`` processes.

    A coordinator-boundary call (never from inside an executing task). One worker runs the same composition
    serially in this process, with no engine runtime (the ordered single-worker audit route).
    """
    from ...engine.graph import run_graph

    if workers <= 1:
        _cache(cache_root)
        previous = numba.get_num_threads()
        numba.set_num_threads(1)
        try:
            return api.run(inputs, parameters)
        finally:
            numba.set_num_threads(previous)
    unit = task(inputs, parameters, workers=workers, cache_root=cache_root)
    return run_graph(
        [unit], cores=workers, in_process=False, preload=("haplotype_reconstruction.stages.pedigree.adapter",)
    )[unit.key]


def task(inputs: PedigreeInput, parameters: PedigreeConfig, *, workers: int = 1, cache_root: str | None = None) -> Task:
    """Declare the pilot's fit-to-release task for engine.graph (it expands into its resampling fits).

    The driver owns record publication and derivation identity; inputs include the
    explicit convention settings, scoring outputs and full-marker guard evidence.
    """
    from ..framework import ResourceRequest, Task, TaskKey

    return Task(
        TaskKey("pedigree", "decision"),
        decide,
        args=(inputs, parameters),
        kwargs={"cache_root": cache_root},
        affinity=_AFFINITY,
        resources=ResourceRequest(min_cores=1, max_cores=workers),
        thread_argument="workers",
    )
