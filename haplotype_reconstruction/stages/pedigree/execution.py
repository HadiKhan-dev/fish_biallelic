"""Run independent resampling fits and calibration terms as engine subtasks under the engine's CPU authority.

Resampling (PD16): one weighted fit is GIL-bound Python over the N children (selection, family
messages, ties), so threads of one process cannot run fits side by side (at N = 1,003, 16 nested
threads realised about 4.5 CPUs; ped40c PERF_FOUND). The decision task therefore returns the
planned fits as an engine Expansion: work-sized batches of planned fit tasks, run in the engine's
forkserver worker processes under its CPU and memory authority, as 3.5 ran them in a process pool.
The read-only DecisionInputs arrays go to shared memory once (engine.workers.share_value); each
subtask attaches them without copying. The gather receives the results in the plan's fixed order
and reduces them there (integer support sums), then releases the segments.

Calibration (PD03): when the objective is large enough for a term executor (calibration.uses_executor),
the scale search runs in rounds. Science's optimizer (calibration.solve: grid, Brent, refiner) runs
unchanged in a helper thread on the decision's affinity worker; each objective call it makes becomes
one engine Expansion of contiguous child-range chunks (3.5's cut: about equal terms plus a per-child
overhead, one chunk per worker) over the terms shared once, and the round's gather hands the children's
terms, concatenated in child order, back to science, which folds them serially in that order. Between
rounds the helper thread only waits. Method: docs/methods/pedigree.md#parallel-structure (WP_PE4/PE5 §8).
"""

from __future__ import annotations

import queue
import threading
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from ...contracts.pedigree import CalibrationRecord, DecisionInputs
from ...core.errors import InvariantError
from ...engine.graph import Expansion, ResourceRequest, ResultRef, Task, TaskKey
from ...engine.runtime import thread_team
from ...engine.workers import (
    ResidentRef,
    SharedValue,
    call_shared,
    release_resident,
    release_shared_value,
    resident_value,
    retain,
    share_value,
)
from ...science.pedigree import calibration
from ...science.pedigree.calibration import CalibrationProblem
from ...science.pedigree.resampling import FitResult, FitTask, ResamplingPlan, evaluate_task

__all__ = [
    "calibrate",
    "calibration_chunk",
    "calibration_round",
    "child_ranges",
    "fit_subtasks",
    "fit_task",
    "fit_batch",
    "fit_batches",
]

# Calibration chunks (3.5's pool cut; execution only, values never depend on them): contiguous child ranges of
# about equal cost, a child's cost being its term count plus this overhead, and one chunk per worker.
CALIBRATION_CHILD_OVERHEAD = 500.0
CALIBRATION_CHUNKS_PER_WORKER = 1
# How often a waiting side of the calibration rounds re-checks that the other side still exists (seconds).
_LIVENESS_SECONDS = 1.0


class _NumbaConfiguration(Protocol):
    """Numba's runtime setting is absent from its generated stubs."""

    CACHE_DIR: str


def _fit_attached(inputs: DecisionInputs, task: FitTask) -> FitResult:
    # One fit per worker. Numerical kernels never compete for that worker's CPU.
    with thread_team(1):
        return evaluate_task(inputs, task)


def fit_task(inputs: SharedValue, task: FitTask) -> FitResult:
    """Engine subtask: one planned fit task on the shared read-only DecisionInputs."""
    return call_shared(inputs, _fit_attached, task)


# Selection scans per batch; two waves per worker retain load balancing. This is
# an execution grain only: the complete seeded science plan is evaluated unchanged.
RESAMPLING_BATCH_WORK = 250_000
RESAMPLING_BATCHES_PER_WORKER = 2


def fit_batches(planned: ResamplingPlan, alternatives: int) -> tuple[tuple[FitTask, ...], ...]:
    """Group whole planned tasks by row scans, retaining their order and draw counts.

    A scan visits the alternatives once per fitted multiplicity vector. Limit the
    grain by both work and two waves per planned worker; expensive fits stay alone.
    No fit is dropped or divided, and no seed or arithmetic depends on these cuts.
    """
    costs = [max(1, alternatives) * len(task[1]) for task in planned.tasks]
    target = min(
        RESAMPLING_BATCH_WORK,
        max(
            1,
            (sum(costs) + planned.worker_count * RESAMPLING_BATCHES_PER_WORKER - 1)
            // (planned.worker_count * RESAMPLING_BATCHES_PER_WORKER),
        ),
    )
    batches: list[tuple[FitTask, ...]] = []
    batch: list[FitTask] = []
    work = 0
    for task, cost in zip(planned.tasks, costs, strict=True):
        if batch and work + cost > target:
            batches.append(tuple(batch))
            batch, work = [], 0
        batch.append(task)
        work += cost
    if batch:
        batches.append(tuple(batch))
    return tuple(batches)


def _fit_batch_attached(inputs: DecisionInputs, tasks: tuple[FitTask, ...]) -> tuple[FitResult, ...]:
    with thread_team(1):
        return tuple(evaluate_task(inputs, task) for task in tasks)


def fit_batch(inputs: SharedValue, tasks: tuple[FitTask, ...]) -> tuple[FitResult, ...]:
    """Attach the shared decision inputs once and evaluate one batch serially."""
    return call_shared(inputs, _fit_batch_attached, tasks)


def fit_subtasks(
    inputs: DecisionInputs, planned: ResamplingPlan, stage: str, *, copy_threads: int = 1
) -> tuple[SharedValue, tuple[Task, ...]]:
    """Share inputs once and declare work-sized batches in the science plan's order.

    The gather flattens batch results back to the original plan. Each worker owns
    one CPU; the caller releases the shared value in its affinity gather.
    """
    batches = fit_batches(planned, len(inputs.alternatives.rows))
    shared = share_value(inputs, "resampling-inputs", copy_threads=copy_threads)
    width = len(str(max(0, len(batches) - 1)))
    subtasks = tuple(
        Task(
            TaskKey(stage, "resampling", round=f"{index:0{width}d}"),
            fit_batch,
            args=(shared, batch),
            resources=ResourceRequest(min_cores=1, max_cores=1),
        )
        for index, batch in enumerate(batches)
    )
    return shared, subtasks


def _calibration_child(kind: str, scales: tuple[float, ...], term: tuple[np.ndarray, ...]) -> np.ndarray:
    if kind == "derivative":
        return np.asarray([calibration._derivative_terms(term, scale) for scale in scales])
    return np.asarray([calibration._prediction_term(term, scale, kind == "test") for scale in scales])


@dataclass(frozen=True)
class CalibrationTerms:
    """The children's PD03 terms in child order for SharedValue: one ragged record per field, each child's block
    in C order (the three-fold fields are [3, k] blocks, k = the child's term count)."""

    train: tuple[np.ndarray, ...]
    validation: tuple[np.ndarray, ...]
    combined: tuple[np.ndarray, ...]
    test: tuple[np.ndarray, ...]
    prior: tuple[np.ndarray, ...]
    prior_test: tuple[np.ndarray, ...]

    @classmethod
    def pack(cls, prepared: tuple[tuple[np.ndarray, ...], ...]) -> CalibrationTerms:
        return cls(*(tuple(np.ravel(block, order="C") for block in column) for column in zip(*prepared, strict=True)))

    def child(self, index: int) -> tuple[np.ndarray, ...]:
        """Child ``index``'s prepared entry as fresh C-ordered arrays of the original shapes (as 3.5 copied out)."""
        count = len(self.combined[index])
        return tuple(
            np.array(block[index].reshape((3, count) if rows else (count,)), dtype=np.float64, order="C")
            for block, rows in (
                (self.train, True),
                (self.validation, True),
                (self.combined, False),
                (self.test, False),
                (self.prior, True),
                (self.prior_test, False),
            )
        )


def child_ranges(prepared: tuple[tuple[np.ndarray, ...], ...], workers: int) -> tuple[tuple[int, int], ...]:
    """Contiguous child ranges [start, stop) of about equal cost (terms + overhead), at most one per worker."""
    cost = np.cumsum(np.asarray([len(term[2]) for term in prepared], dtype=np.float64) + CALIBRATION_CHILD_OVERHEAD)
    pieces = min(len(prepared), max(1, workers) * CALIBRATION_CHUNKS_PER_WORKER)
    cuts = np.searchsorted(cost, cost[-1] * np.arange(1, pieces) / pieces, side="right")
    bounds = np.unique(np.concatenate(([0], cuts, [len(prepared)])))
    return tuple((int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:], strict=True) if b > a)


def _chunk_attached(terms: CalibrationTerms, kind: str, scales: tuple[float, ...], start: int, stop: int) -> np.ndarray:
    return np.asarray([_calibration_child(kind, scales, terms.child(child)) for child in range(start, stop)])


def calibration_chunk(shared: SharedValue, kind: str, scales: tuple[float, ...], start: int, stop: int) -> np.ndarray:
    """Engine subtask: children start..stop-1's terms at each scale ([children, scales] or [children, scales, 4])."""
    return call_shared(shared, _chunk_attached, kind, scales, start, stop)


class _Closed(Exception):
    """The rounds were closed before the search finished; the helper thread stops quietly."""


class _Channel:
    """The two one-way queues between science's search (helper thread) and the engine rounds (task side)."""

    def __init__(self) -> None:
        self.requests: queue.SimpleQueue = queue.SimpleQueue()
        self.replies: queue.SimpleQueue = queue.SimpleQueue()
        self.closed = threading.Event()

    def close(self) -> None:
        self.closed.set()


@dataclass(frozen=True)
class _RoundExecutor:
    """calibration.CalibrationExecutor whose term requests are answered by engine rounds."""

    workers: int
    channel: _Channel

    def calibration_terms(self, prepared: Any, kind: str, scales: tuple[float, ...]) -> np.ndarray:
        self.channel.requests.put(("terms", kind, tuple(scales)))
        while True:
            try:
                return self.channel.replies.get(timeout=_LIVENESS_SECONDS)
            except queue.Empty:
                if self.channel.closed.is_set():
                    raise _Closed from None


def _search(channel: _Channel, problem: CalibrationProblem, workers: int) -> None:
    """The helper thread: science's unchanged scale search, reporting its result or error as the last request."""
    try:
        fitted = calibration.solve(problem, _RoundExecutor(workers, channel))
    except _Closed:
        return
    except BaseException as error:
        channel.requests.put(("error", error))
    else:
        channel.requests.put(("done", fitted))


@dataclass
class _Rounds:
    """Retained on the decision's affinity worker for the whole search; ``then`` continues the decision."""

    problem: CalibrationProblem
    workers: int
    stage: str
    affinity: str
    then: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: Mapping[str, Any]
    shared: SharedValue
    chunks: tuple[tuple[int, int], ...]
    channel: _Channel = field(default_factory=_Channel)
    thread: threading.Thread | None = None
    count: int = 0

    def __post_init__(self) -> None:
        # A dropped bundle (an aborted graph) stops its waiting helper thread.
        weakref.finalize(self, self.channel.close)

    def advance(self, terms: np.ndarray | None) -> tuple:
        """Answer the pending request (or start the search) and wait for the search's next request or result."""
        if self.thread is None:
            self.thread = threading.Thread(
                target=_search,
                args=(self.channel, self.problem, self.workers),
                name="pedigree-calibration",
                daemon=True,
            )
            self.thread.start()
        else:
            self.channel.replies.put(terms)
        while True:
            try:
                return self.channel.requests.get(timeout=_LIVENESS_SECONDS)
            except queue.Empty:
                if not self.thread.is_alive():
                    try:
                        return self.channel.requests.get_nowait()
                    except queue.Empty:
                        raise InvariantError("the calibration search stopped without a result") from None

    def expansion(self, reference: ResidentRef, kind: str, scales: tuple[float, ...]) -> Expansion:
        index, self.count = self.count, self.count + 1
        width = len(str(len(self.chunks) - 1))
        subtasks = tuple(
            Task(
                TaskKey(self.stage, "calibration", round=f"{index:05d}.{c:0{width}d}"),
                calibration_chunk,
                args=(self.shared, kind, scales, a, b),
                resources=ResourceRequest(min_cores=1, max_cores=1),
            )
            for c, (a, b) in enumerate(self.chunks)
        )
        gather = Task(
            TaskKey(self.stage, "calibration", round=f"{index:05d}"),
            calibration_round,
            args=(reference, tuple(ResultRef(t.key) for t in subtasks)),
            depends_on=tuple(t.key for t in subtasks),
            affinity=self.affinity,
            resources=ResourceRequest(max_cores=self.workers),
            thread_argument="workers",
        )
        return Expansion(subtasks, gather)

    def finish(self, reference: ResidentRef) -> None:
        self.channel.close()
        try:
            release_shared_value(self.shared)
        finally:
            release_resident(reference)


def calibrate(
    problem: CalibrationProblem,
    workers: int,
    *,
    stage: str,
    affinity: str,
    then: Callable[..., Any],
    args: tuple[Any, ...] = (),
    kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Run PD03's scale search, then return ``then(fitted, *args, workers=workers, **kwargs)``.

    Called by an affinity action holding ``workers`` cores. A search that uses no term executor runs here,
    serially. Otherwise it runs in engine rounds (module docstring); ``then`` must be importable, and ``args``
    and ``kwargs`` stay on this worker (retained, never pickled).
    """
    kwargs = dict(kwargs or {})
    if not calibration.uses_executor(problem.prepared, workers):
        return then(calibration.solve(problem), *args, workers=workers, **kwargs)
    assert problem.prepared is not None
    shared = share_value(CalibrationTerms.pack(problem.prepared), "calibration-terms", copy_threads=workers)
    try:
        rounds = _Rounds(
            problem, workers, stage, affinity, then, args, kwargs, shared, child_ranges(problem.prepared, workers)
        )
        reference = retain(rounds, "calibration")
    except BaseException:
        release_shared_value(shared)
        raise
    return calibration_round(reference, None)


def calibration_round(reference: ResidentRef, chunks: tuple[np.ndarray, ...] | None = None, *, workers: int = 1) -> Any:
    """One round's gather (the first call starts the search): pass the terms in child order to the search, then
    either expose its next request as the next round or continue the decision with its result."""
    rounds: _Rounds = resident_value(reference)
    try:
        event = rounds.advance(None if chunks is None else np.concatenate(chunks, axis=0))
        if event[0] == "terms":
            return rounds.expansion(reference, event[1], event[2])
        if event[0] == "error":
            raise event[1]
        fitted: tuple[float, CalibrationRecord] = event[1]
    except BaseException:
        rounds.finish(reference)
        raise
    rounds.finish(reference)
    # The decision task's own team, not this gather's claim, sizes what follows (the resampling plan).
    return rounds.then(fitted, *rounds.args, workers=rounds.workers, **rounds.kwargs)
