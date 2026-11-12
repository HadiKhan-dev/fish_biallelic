"""Run independent resampling fits as engine subtasks under the engine's CPU authority.

Resampling (PD16): one weighted fit is GIL-bound Python over the N children (selection, family
messages, ties), so threads of one process cannot run fits side by side (at N = 1,003, 16 nested
threads realised about 4.5 CPUs). The decision task therefore returns the
planned fits as an engine Expansion: work-sized batches of planned fit tasks, run in the engine's
forkserver worker processes under its CPU and memory authority.
The read-only DecisionInputs arrays go to shared memory once (engine.workers.share_value); each
subtask attaches them without copying. The gather receives the results in the plan's fixed order
and reduces them there (integer support sums), then releases the segments.

Method: docs/methods/pedigree.md#parallel-structure.
"""

from __future__ import annotations

from ...contracts.pedigree import DecisionInputs
from ...engine.cpu import thread_team
from ...engine.graph import ResourceRequest, Task, TaskKey
from ...engine.workers import (
    SharedValue,
    call_shared,
    share_value,
)
from ...science.pedigree.resampling import FitResult, FitTask, ResamplingPlan, evaluate_task

__all__ = [
    "fit_subtasks",
    "fit_batch",
    "fit_batches",
]

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
