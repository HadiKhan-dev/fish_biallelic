"""Execute dependency graphs, expansions and bounded affinity sessions."""

from __future__ import annotations

import logging
import signal
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from ..core.errors import InvariantError, ResourceError, SettingsError
from .cpu import CPU_POLL_SECONDS, FixedBudget, read_counter
from .graph import (
    RELEASED_RESULT,
    Expansion,
    PackedCall,
    ResultRef,
    Task,
    TaskKey,
    _readonly,
    _references,
    cyclic_keys,
    dependency_order,
    resolve_results,
)
from .lanes import CONTIG_LANE_CORES, ChainPlan
from .memory import (
    GIB,
    WAKE_SECONDS,
    AdmissionToken,
    MemoryGovernor,
    current_governor,
    memory_governor,
    set_governor,
    tasks_executing,
)
from .runtime import RunInterrupted, affinity_cpus, configure_worker, current_settings, signal_boundary, start

if TYPE_CHECKING:
    from .workers import WorkerGroup

_LOG = logging.getLogger(__name__)


def _resolved_task(task: Task, results: Mapping[TaskKey, Any]) -> Task:
    if len(task.args) == 1 and type(task.args[0]) is PackedCall:
        # Packed by the worker that expanded it: its references were recorded there; their values go beside it.
        packed = task.args[0]
        return (
            replace(task, args=(packed, tuple(resolve_results(ResultRef(key), results) for key in packed.refs)))
            if packed.refs
            else task
        )
    return replace(task, args=resolve_results(task.args, results), kwargs=resolve_results(task.kwargs, results))


def _precompleted() -> None:
    """Validation-only node representing an identity-checked published prerequisite."""


class TaskSession:
    """Keep affinity state across bounded graph batches under one engine authority.

    The caller owns each returned result mapping and may discard it before the next
    batch. The session retains only workers, admission tokens and completed chain
    keys; it never accumulates result arrays. Dependencies across batches must be
    passed explicitly as initial_results or as resident handles in task arguments.
    Closing the context releases every retained state, including after failure.
    Sessions belong at the coordinator boundary; executing tasks return Expansion.
    """

    def __init__(
        self,
        cores: int,
        *,
        max_workers: int | None = None,
        in_process: bool = False,
        contig_lanes: int | None = None,
        recycle_bytes: int | None = None,
        governor: MemoryGovernor | None = None,
        preload: Sequence[str] = (),
        chains: Sequence[ChainPlan] = (),
        dynamic: bool = True,
    ):
        self.cores, self.max_workers, self.in_process = cores, max_workers, in_process
        self.contig_lanes, self.recycle_bytes = contig_lanes, recycle_bytes
        self.preload, self.chains, self.dynamic = tuple(preload), tuple(chains), dynamic
        self.governor = governor
        self.group: WorkerGroup | None = None
        self.residents: dict[tuple[int, str], AdmissionToken] = {}
        self.chain_keys = {key for chain in chains for key in chain.tasks}
        self.completed: set[TaskKey] = set()
        self.active = self.running = self.closed = False

    def __enter__(self) -> TaskSession:
        if self.active or self.closed:
            raise SettingsError("task session cannot be reopened")
        self.active = True
        return self

    def run(
        self,
        tasks: Sequence[Task],
        *,
        initial_results: Mapping[TaskKey, Any] | None = None,
        priorities: Mapping[TaskKey, float] | None = None,
        serial_order: Sequence[str] = (),
        on_task: Callable[[Task, Any], None] | None = None,
        retain_intermediates: bool = False,
    ) -> dict[TaskKey, Any]:
        """Run one bounded batch; no implicit results survive from earlier batches."""
        if not self.active or self.closed or self.running:
            raise SettingsError("task session must be open and idle")
        self.running = True
        try:
            return run_graph(
                tasks,
                cores=self.cores,
                max_workers=self.max_workers,
                in_process=self.in_process,
                contig_lanes=self.contig_lanes,
                recycle_bytes=self.recycle_bytes,
                preload=self.preload,
                chains=self.chains,
                dynamic=self.dynamic,
                governor=self.governor,
                initial_results=initial_results,
                priorities=priorities,
                serial_order=serial_order,
                on_task=on_task,
                retain_intermediates=retain_intermediates,
                _session=self,
            )
        except BaseException:
            self.close(abort=True)
            raise
        finally:
            self.running = False

    def close(self, *, abort: bool = False) -> None:
        """Reap this session's workers and release its resident tokens exactly once."""
        if self.closed:
            return
        self.closed, self.active = True, False
        if self.group is not None:
            self.group.close(abort=abort)
        for token in self.residents.values():
            token.release()
        self.residents.clear()
        if self.in_process:
            from .workers import release_local_state

            release_local_state()

    def __exit__(self, kind: Any, error: Any, traceback: Any) -> None:
        self.close(abort=error is not None)


class _GraphProgress:
    """Low-volume observation of existing accounting, never a scheduling input.

    Counts are executor calls (including expansion and gather calls). Totals grow
    as expansions reveal work. CPU slots are claims, not an OS utilisation sample;
    memory is the governor's existing sampled live value. The timer also covers a
    long in-process call, where the coordinator cannot poll.
    """

    INTERVAL_SECONDS = 600.0

    def __init__(self, tasks, running, expansions, cores, governor):
        self.enabled = _LOG.isEnabledFor(logging.INFO)
        self.running, self.expansions = running, expansions
        self.cores, self.governor, self.group = cores, governor, None
        self.rows: dict[tuple[str, str], list[int]] = {}
        self.candidates: set[tuple[str, str]] = set()
        self.lock = threading.RLock()
        self.closed = False
        self.timer: threading.Timer | None = None
        self.add(tasks)

    def add(self, tasks):
        if not self.enabled:
            return
        with self.lock:
            for task in tasks:
                key = (task.key.stage, task.key.scope)
                self.rows.setdefault(key, [0, 0])[1] += 1

    def finished(self, key):
        if not self.enabled:
            return
        with self.lock:
            stage = (key.stage, key.scope)
            # A call this observer was never shown still counts; progress must never fail a run.
            row = self.rows.setdefault(stage, [0, 0])
            row[0] += 1
            row[1] = max(row[1], row[0])
            if row[0] == row[1]:
                self.candidates.add(stage)

    def start(self, group, cpu_slots):
        self.group = group
        self.cpu_slots = cpu_slots
        if self.enabled:
            self._arm()

    def _arm(self):
        self.timer = threading.Timer(self.INTERVAL_SECONDS, self._tick)
        self.timer.daemon = True
        self.timer.start()

    def _tick(self):
        with self.lock:
            if self.closed:
                return
            self.emit(periodic=True)
            self._arm()

    def emit(self, *, periodic=False):
        if not self.enabled:
            return
        try:
            self._emit(periodic)
        except Exception:  # a progress line is diagnostics: it must never fail or stall a run
            _LOG.debug("progress line failed", exc_info=True)

    def _emit(self, periodic):
        with self.lock:
            if self.closed or (not periodic and not self.candidates):
                return
            suspended = {(key.stage, key.scope) for key in self.expansions.copy()}
            done = {key for key in self.candidates - suspended if self.rows[key][0] == self.rows[key][1]}
            if not periodic and not done:
                return
            running = tuple(self.running.copy())
            parts = [
                f"{stage} {scope} {self.rows[stage, scope][0]}/{self.rows[stage, scope][1]} done"
                for stage, scope in sorted(done)
            ]
            if periodic:
                stages: dict[str, list[int]] = {}
                for (stage, _), (completed, total) in self.rows.items():
                    row = stages.setdefault(stage, [0, 0])
                    row[0] += completed
                    row[1] += total
                parts.extend(
                    f"{stage} {completed}/{total} known ({sum(k.stage == stage for k in running)} running)"
                    for stage, (completed, total) in sorted(stages.items())
                    if completed < total
                )
            self.candidates.difference_update(done)
            if not parts:
                return
            cpu = (
                sum(c.claimed.get_obj().value for c in self.group.authority.groups)
                if self.group is not None
                else self.cpu_slots()
            )
            # Aligned reads of the existing sample; no process traversal, new
            # sample, admission lock, task-log read or CPU rebalance.
            live, limit = self.governor.pressure_sample()
            _LOG.info(
                "[progress] %s; running %d tasks; CPU %d/%d slots claimed; memory %.2f/%.2f GiB",
                "; ".join(parts),
                len(running),
                cpu,
                self.cores,
                live / GIB,
                limit / GIB,
            )

    def close(self):
        with self.lock:
            self.closed = True
            if self.timer is not None:
                self.timer.cancel()


def run_graph(
    tasks: Sequence[Task],
    *,
    cores: int = 1,
    in_process: bool = True,
    max_workers: int | None = None,
    initial_results: Mapping[TaskKey, Any] | None = None,
    priorities: Mapping[TaskKey, float] | None = None,
    serial_order: Sequence[str] = (),
    contig_lanes: int | None = None,
    recycle_bytes: int | None = None,
    governor: MemoryGovernor | None = None,
    on_task: Callable[[Task, Any], None] | None = None,
    preload: Sequence[str] = (),
    chains: Sequence[ChainPlan] = (),
    dynamic: bool = True,
    _session: TaskSession | None = None,
    retain_intermediates: bool = False,
) -> dict[TaskKey, Any]:
    """Execute static and expanded tasks under one pressure/CPU authority."""
    return _GraphRun(
        tasks,
        cores=cores,
        in_process=in_process,
        max_workers=max_workers,
        initial_results=initial_results,
        priorities=priorities,
        serial_order=serial_order,
        contig_lanes=contig_lanes,
        recycle_bytes=recycle_bytes,
        governor=governor,
        on_task=on_task,
        preload=preload,
        chains=chains,
        dynamic=dynamic,
        _session=_session,
        retain_intermediates=retain_intermediates,
    ).run()


@dataclass
class _GraphRun:
    """One graph authority and the state shared by its dispatch steps."""

    tasks: Sequence[Task]
    cores: int
    in_process: bool
    max_workers: int | None
    initial_results: Mapping[TaskKey, Any] | None
    priorities: Mapping[TaskKey, float] | None
    serial_order: Sequence[str]
    contig_lanes: int | None
    recycle_bytes: int | None
    governor: MemoryGovernor | None
    on_task: Callable[[Task, Any], None] | None
    preload: Sequence[str]
    chains: Sequence[ChainPlan]
    dynamic: bool
    _session: TaskSession | None
    retain_intermediates: bool

    def prepare(self):
        """Validate the initial graph and establish this run's state."""
        # A failed in-process task can leave a native child thread unwinding. Python
        # cannot interrupt that call; refuse a second authority until it has exited.
        if tasks_executing():
            raise SettingsError(
                "an executing task must return Expansion; nested or overlapping graph executors are not allowed"
            )
        self.tasks = tuple(self.tasks)
        self.results = dict(self.initial_results or {})
        if set(self.results) & {task.key for task in self.tasks}:
            raise SettingsError("initial results and scheduled task keys overlap")
        self.public_keys = set(self.results) | {task.key for task in self.tasks}
        self.original_order = dependency_order((*self.tasks, *(Task(key, _precompleted) for key in self.results)))
        if self.cores < 1 or self.cores > affinity_cpus():
            raise ResourceError(f"requested {self.cores} CPUs; available affinity is {affinity_cpus()}")
        if self.max_workers is not None and self.max_workers < 1:
            raise SettingsError("max_workers must be positive")
        for task in self.tasks:
            if task.resources.min_cores > self.cores:
                raise ResourceError(f"{task.key}: CPU minimum exceeds allocation")
        if not self.tasks:
            return False
        configure_worker(self.cores)
        settings = current_settings()
        if settings is None:
            start(self.cores, preload=self.preload)
        elif settings.cores < self.cores:
            raise ResourceError("runtime CPU ceiling is smaller than this graph")

        self.previous_governor = current_governor()
        self.injected_governor = self.governor is not None
        if self.injected_governor:
            set_governor(self.governor)
        self.governor = memory_governor()
        if self._session is not None:
            self._session.governor = self.governor
        self.count = min(self.cores, self.max_workers or self.cores)
        self.lanes = max(
            1,
            min(
                self.count, max(1, self.cores // CONTIG_LANE_CORES) if self.contig_lanes is None else self.contig_lanes
            ),
        )
        if self.contig_lanes is not None and self.contig_lanes < 1:
            raise SettingsError("contig_lanes must be positive")
        self.indexed = {task.key: task for task in self.tasks}
        self.writers = {output: task.key for task in self.tasks for output in task.outputs}
        self.pending = set(self.indexed)
        self.pending_order = list(self.indexed)
        self.requests: dict[TaskKey, AdmissionToken] = {}
        self.gathers: dict[TaskKey, tuple[TaskKey, int]] = {}
        self.expansions: dict[TaskKey, tuple[tuple[TaskKey, ...], Task, int]] = {}
        self.parents: dict[TaskKey, TaskKey] = {}
        self.depths = {key: 0 for key in self.indexed}
        self.running: dict[TaskKey, int] = {}
        # A task's lane owner is its affinity, else the contig chain it belongs to (ENG-23 chains name every discovery,
        # selection, assembly, painting and refit task of a contig): affinity-free block tasks share their contig's lane,
        # weight and straggler floor without being pinned to one worker. Distinct contig owners keep unrelated
        # block tasks from sharing a single lane.
        self.chain_owners = {key: chain.affinity for chain in self.chains for key in chain.tasks}
        self.owners: dict[TaskKey, str | None] = {
            key: task.affinity if task.affinity is not None else self.chain_owners.get(key)
            for key, task in self.indexed.items()
        }
        self.assigned_lanes: dict[TaskKey, int] = {}
        self.expected: dict[TaskKey, int] = {}  # unsent ready tasks counted as arriving for the free workers, by lane
        self.residents: dict[tuple[int, str], AdmissionToken] = {} if self._session is None else self._session.residents
        self.rank: dict[str | None, int] = {scope: index for index, scope in enumerate(self.serial_order)}
        self.costs: dict[str | None, float] = {chain.affinity: chain.weight for chain in self.chains}
        self.priorities = (
            {key: self.costs.get(task.affinity, 0.0) for key, task in self.indexed.items()}
            if self.priorities is None
            else self.priorities
        )
        self.group: WorkerGroup | None = None if self._session is None else self._session.group
        self.stopped = False
        self.learned_cost: dict[tuple[str, str, str], tuple[float, int]] = {}

        self.progress = _GraphProgress(self.tasks, self.running, self.expansions, self.cores, self.governor)

        return True

    def cancel(self):
        self.stopped = True

    def task_order(self, key: TaskKey):
        task = self.indexed[key]
        measured, count_cost = self.learned_cost.get(
            (key.stage, task.function.__module__, task.function.__qualname__), (0.0, 0)
        )
        cost = task.resources.cost or (measured / count_cost if count_cost else 0.0)
        return (
            -self.depths.get(key, 0),
            self.rank.get(task.affinity, len(self.rank)),
            -self.priorities.get(key, 0.0),
            -cost,
            key,
        )

    def completed(self, worker: int, reply: Any):
        for message in reply.diagnostics:
            _LOG.info("%s", message)
        task = self.indexed[reply.key]
        kind = (task.key.stage, task.function.__module__, task.function.__qualname__)
        seconds, samples = self.learned_cost.get(kind, (0.0, 0))
        self.learned_cost[kind] = (seconds + reply.wall_seconds, samples + 1)
        token = self.requests.pop(reply.key)
        self.running.pop(reply.key, None)
        for identity, resident in tuple(self.residents.items()):
            if identity[0] == worker and identity[1] not in reply.resident_affinities:
                resident.release()
                del self.residents[identity]
        if isinstance(reply.value, Expansion):
            self.accept_expansion(task, reply, worker, token)
            return
        self.results[reply.key] = _readonly(reply.value)
        finished = reply.key
        while finished in self.gathers:
            parent, _ = self.gathers.pop(finished)
            self.results[parent] = reply.value
            children, continuation, _ = self.expansions.pop(parent)
            if not self.retain_intermediates:
                self.release_consumed(children, continuation)
            finished = parent
        if (
            task.affinity is not None
            and task.affinity in reply.resident_affinities
            and (worker, task.affinity) not in self.residents
        ):
            self.governor.rest(token.id)
            self.residents[worker, task.affinity] = token
        else:
            token.release(reply.peak_bytes)
        if self.on_task is not None:
            self.on_task(task, reply)
        self.progress.finished(reply.key)
        self.progress.emit()
        # Completed tasks need only their result; an expansion's parent completes through its gather.
        # Drop arguments so each round's evidence and shared-memory backing can be released.
        if task.args or task.kwargs:
            self.indexed[reply.key] = replace(task, args=(), kwargs={})

    def accept_expansion(self, task, reply, worker, token):
        """Validate and publish one expansion before exposing its children."""
        expansion = reply.value
        child_keys = tuple(child.key for child in expansion.subtasks)
        if len(set(child_keys)) != len(child_keys) or set(child_keys) & (self.indexed.keys() | self.results.keys()):
            token.release()
            raise SettingsError("expansion task keys must be new and unique")
        gather = expansion.gather
        if gather.key == task.key:
            gather = replace(gather, key=replace(task.key, unit=task.key.unit + ".gather"))
        if gather.key in self.indexed or gather.key in self.results or gather.key in child_keys:
            token.release()
            raise SettingsError("expansion gather key must be unique (or the parent key)")
        if gather.affinity is None:
            gather = replace(gather, affinity=task.affinity)
        elif gather.affinity != task.affinity:
            token.release()
            raise SettingsError("expansion gather must use its parent's affinity")
        gather = replace(gather, depends_on=tuple(dict.fromkeys((*gather.depends_on, *child_keys))))
        available = {*self.results, *self.indexed, *child_keys, gather.key} - {task.key}
        for child in (*expansion.subtasks, gather):
            if set(child.depends_on) - available:
                token.release()
                raise SettingsError("expanded child depends on an unfinished parent or unknown task")
            if child.resources.min_cores > self.cores:
                token.release()
                raise ResourceError("expanded task CPU minimum exceeds allocation")
        new = (*expansion.subtasks, gather)
        # The new tasks' own checks (CPU ranges, references, writers, cycles among them), with finished stand-ins
        # for the existing tasks they depend on; then cycles through the unfinished graph. A suspended parent
        # is completed by its gather, not a resolved leaf: earlier expansions keep that dependency, so cycles
        # through them cannot disappear behind placeholders. Finished tasks depend on nothing.
        new_keys = {child.key for child in new}
        standing = {
            dependency: Task(dependency, _precompleted)
            for child in new
            for dependency in child.depends_on
            if dependency not in new_keys
        }
        depends = {key: item.depends_on for key, item in self.indexed.items() if key not in self.results}
        for parent, (_, continuation, _) in self.expansions.items():
            depends[parent] = (continuation.key,)
            depends[continuation.key] = continuation.depends_on
        depends[task.key] = (gather.key,)
        depends.update((child.key, child.depends_on) for child in new)
        try:
            dependency_order((*new, *standing.values()))
            cycle = cyclic_keys(depends)
            if cycle:
                raise SettingsError(f"task graph contains a dependency cycle: {cycle}")
        except BaseException:
            token.release()
            raise
        for child in new:
            for output in child.outputs:
                previous = self.writers.get(output)
                if previous is not None and not (child.key == gather.key and previous == task.key):
                    token.release()
                    raise SettingsError(f"{output}: expanded task conflicts with writer {previous}")
                self.writers[output] = child.key
        self.governor.suspend(token.id)
        self.requests[task.key] = token
        self.expansions[task.key] = (child_keys, gather, worker)
        for child in expansion.subtasks:
            self.indexed[child.key] = child
            self.owners[child.key] = child.affinity if child.affinity is not None else self.owners[task.key]
            self.pending.add(child.key)
            self.pending_order.append(child.key)
            self.parents[child.key] = task.key
            self.depths[child.key] = self.depths[task.key] + 1
        self.progress.add((*expansion.subtasks, gather))
        self.progress.finished(reply.key)
        self.progress.emit()
        # Its gather completes it; its arguments (a round's evidence, a session) are never needed again.
        if task.args or task.kwargs:
            self.indexed[task.key] = replace(task, args=(), kwargs={})
        return

    def release_consumed(self, children, continuation):
        """Release expansion-local values once their gather has consumed them."""
        # Every child is consumed by this gather. Preserve any values
        # another already-declared task still reads. Child task keys are
        # private to Expansion; normal caller-supplied results stay public.
        candidates = {*children, continuation.key} - self.public_keys
        referenced = set()
        for key in self.pending:
            referenced.update(_references((self.indexed[key].args, self.indexed[key].kwargs)))
        for _, waiting_gather, _ in self.expansions.values():
            referenced.update(_references((waiting_gather.args, waiting_gather.kwargs)))
        for key in candidates - referenced:
            self.results[key] = RELEASED_RESULT

    def start_workers(self):
        from .workers import WorkerGroup

        self.group = (
            None
            if self.in_process
            else self.group
            or WorkerGroup(
                self.cores,
                self.count,
                lanes=self.lanes,
                chains=self.chains,
                recycle_bytes=self.recycle_bytes,
                static_affinities=()
                if self.dynamic
                else tuple(
                    (owner, self.costs.get(owner, 1.0))
                    for owner in dict.fromkeys(task.affinity for task in self.tasks)
                    if owner is not None
                ),
            )
        )
        if self._session is not None:
            self._session.group = self.group
        # The observer keeps accounting maps, not the graph run and its results.
        cores, indexed, running = self.cores, self.indexed, self.running
        self.progress.start(
            self.group,
            lambda: sum(min(cores, indexed[key].resources.max_cores) for key in running.copy()),
        )

    def ready_tasks(self):
        for parent, (children, gather, worker) in tuple(self.expansions.items()):
            if gather.key not in self.indexed and all(key in self.results for key in children):
                self.indexed[gather.key] = gather
                self.owners[gather.key] = self.owners[parent]
                self.pending.add(gather.key)
                self.pending_order.append(gather.key)
                self.depths[gather.key] = self.depths[parent]
                self.gathers[gather.key] = (parent, worker)
        # Retain the previous sorted run so Timsort handles an almost unchanged
        # queue in linear time. Recompute every key: learned costs and caller
        # priorities may change between passes, and new expansions append here.
        self.pending_order = [key for key in self.pending_order if key in self.pending]
        self.pending_order.sort(key=self.task_order)
        ready = [
            key
            for key in self.pending_order
            if all(dependency in self.results for dependency in self.indexed[key].depends_on)
        ]
        if not ready and not self.running:
            raise InvariantError("graph has unfinished work but no executable dependency")
        return ready

    def refresh_lanes(self, ready):
        if self.group is not None:
            needed = {self.owners[k] for k in (*ready, *self.running)}
            done = self.results.keys() if self._session is None else self.results.keys() | self._session.completed
            self.group.authority.reconcile(needed, done)
            # Demand uses declared widths of ready and running tasks, allowing a task that started
            # narrow to grow as peers release CPUs at their boundaries. Inner maps re-read the share.
            # Existing claims and holds protect in-flight work.
            demand: dict[str | None, int] = {}
            for k in (*ready, *self.running):
                owner = self.owners[k]
                demand[owner] = min(self.cores, demand.get(owner, 0) + self.indexed[k].resources.max_cores)
            self.group.authority.limit(demand)
            self.group.authority.announce(dict.fromkeys(self.owners[k] for k in ready))
            self.group.authority.refresh()  # every lane's total, once per pass

    def dispatch_pass(self, ready):
        """Admit ready tasks in order, then send the pass's counted arrivals."""
        from .workers import perform_task

        if self.group is not None and (self.governor.under_pressure() or self.governor.recovery_requested()):
            self.group.release_idle()
        active_affinities = {self.indexed[k].affinity for k in self.running if self.indexed[k].affinity is not None}
        dispatched = False
        running_cores = sum(self.indexed[k].resources.min_cores for k in self.running)
        idle = None if self.group is None else sum(1 for h in self.group.handles if h.ready and h.busy is None)
        lane_load: dict[int, int] = {}
        for k in self.running:
            if k in self.assigned_lanes:
                held = self.assigned_lanes[k]
                lane_load[held] = lane_load.get(held, 0) + self.indexed[k].resources.min_cores
        # Per owner, the smallest CPU minimum its lane could not start this pass: its later ready tasks
        # needing as many are skipped at once (a full lane stays full until the next pass's replies).
        full: dict[str | None, int] = {}
        # Prioritise gathers and pinned tasks, whose single eligible worker could otherwise keep
        # taking unpinned blocks and delay the contig's continuation. Reserve no worker until
        # its lane admits the task, preserving progress for other work in that lane.
        reserved: set[int] = set()
        if self.group is not None:
            # Workers a chain with tasks still to run or a pending gather will need take other tasks last
            # (free_worker); a chain with nothing left frees its worker like any other.
            active = {self.owners[k] for k in self.pending} | {self.owners[k] for k in self.running}
            reserved = {
                *(worker for chain, worker in self.group.affinities.items() if chain in active),
                *(worker for _, _, worker in self.expansions.values()),
            }
            pinned = [k for k in ready if k in self.gathers or self.indexed[k].affinity in self.group.affinities]
            if pinned:
                chosen = set(pinned)
                ready = pinned + [k for k in ready if k not in chosen]
        sends: list[tuple[TaskKey, int, Task, tuple[int, int], int]] = []
        refused: set[TaskKey] = set()  # held back by memory this pass
        for key in ready:
            if idle is not None and idle <= 0:
                break  # every worker is busy: nothing more can start this pass
            task = self.indexed[key]
            if key not in self.gathers and task.resources.min_cores >= full.get(self.owners[key], self.cores + 1):
                continue
            affinity = task.affinity
            if affinity is not None and affinity not in active_affinities:
                if len(active_affinities) >= (1 if self.serial_order else self.lanes):
                    continue
            if running_cores + task.resources.min_cores > self.cores:
                continue
            if self.group is not None:
                if key in self.gathers:
                    worker = self.gathers[key][1]
                    handle = self.group.handles[worker]
                    if not handle.ready or handle.busy is not None:
                        continue
                else:
                    expanding = self.parents.get(key)
                    owner = self.owners[key]
                    own = {
                        index
                        for index in (
                            None if owner is None else self.group.affinities.get(owner),
                            self.expansions[expanding][2] if expanding in self.expansions else None,
                        )
                        if index is not None
                    }
                    available_worker = self.group.free_worker(affinity, reserved, own)
                    if available_worker is None:
                        continue
                    worker = available_worker
                if task.function.__module__ == "__main__" or "<locals>" in task.function.__qualname__:
                    raise SettingsError(f"{key}: process tasks must be importable module-level functions")
            else:
                worker = 0
            if self.group is not None:
                lane = self.group.authority.acquire(self.owners[key], minimum=task.resources.min_cores)
                if lane is None:
                    if key not in self.gathers:
                        full[self.owners[key]] = min(
                            full.get(self.owners[key], self.cores + 1), task.resources.min_cores
                        )
                    continue
                counters = self.group.authority.groups[lane]
                # Hand a lane no more tasks than its CPUs admit: the rest would wait in begin_task on their
                # workers, leaving none for other lanes' ready tasks (head-of-line blocking across contigs).
                load = lane_load.get(lane, 0) + task.resources.min_cores
                if key not in self.gathers and load > max(1, read_counter(counters.total)):
                    full[self.owners[key]] = min(full.get(self.owners[key], self.cores + 1), task.resources.min_cores)
                    continue
                self.assigned_lanes[key] = lane
                if not any(self.owners[k] == self.owners[key] for k in self.running) and (
                    not read_counter(counters.active)
                ):
                    batch = [k for k in ready if self.owners[k] == self.owners[key]]
                    # The workers the batch can occupy at once: any of them for unpinned tasks, else one per
                    # distinct affinity (a contig's pinned worker; per-contig siblings are on different ones).
                    affinities = {self.indexed[k].affinity for k in batch}
                    participants = self.count if None in affinities else min(self.count, len(affinities))
                    counters.start_batch(len(batch), participants)
            token = self.requests.get(key)
            if key in self.gathers and token is None:
                parent = self.gathers[key][0]
                token = self.requests.pop(parent)
                self.requests[key] = token
                self.governor.queue(token.id)
            elif token is None:
                parent_key = self.parents.get(key)
                parent_token = None if parent_key is None else self.requests.get(parent_key)
                token = self.governor.request(
                    label=f"{key.stage}.{key.unit}",
                    queued=True,
                    outer=True,
                    parent=0 if parent_token is None else parent_token.id,
                    depth=self.depths[key],
                )
                self.requests[key] = token
            if not token.poll(fresh=True):
                refused.add(key)
                if self.group is not None:
                    self.group.authority.groups[self.assigned_lanes[key]].open_gate()
                    self.group.authority.lend_idle(self.owners[key])
                continue
            action = _resolved_task(task, self.results)
            self.pending.remove(key)
            self.running[key] = worker
            running_cores += task.resources.min_cores
            if idle is not None:
                idle -= 1
            if key in self.assigned_lanes:
                held = self.assigned_lanes[key]
                lane_load[held] = lane_load.get(held, 0) + task.resources.min_cores
            if affinity is not None:
                active_affinities.add(affinity)
            dispatched = True
            if self.group is None:
                self.completed(worker, perform_task(action, token.held, FixedBudget(self.cores)))
                break
            self.group.pin(affinity, worker)
            self.group.claim(worker, key)
            sends.append((key, worker, action, token.held, self.assigned_lanes[key]))
        if self.group is not None:
            self.count_arrivals(ready, sends, refused, running_cores)
            for _, worker, action, token_held, lane in sends:
                self.group.send(worker, action, token_held, lane=lane)
            if sends:
                # The resolved gather carries all child values. Once sent,
                # the transport's in-flight holds protect the segments;
                # keeping this local action retained the entire last wave.
                sends.clear()
                del action
        return dispatched, ready

    def count_arrivals(self, ready, sends, refused, running_cores):
        # Tasks on their way count as arriving in their lanes before any of this pass's tasks is sent: the
        # pass's sends, and as many further ready tasks as there are workers this pass left free (idle, or
        # still starting): a sibling the pass did not send because its lane's lease had not grown yet, or
        # because its worker was still starting, takes one of them next. A task pinned to a worker counts
        # only for that worker, and of the unpinned tasks sharing an affinity only the first (the others
        # follow it on the worker it will pin); a task counts only while its minimum fits beside the
        # CPUs the running, sent and already counted tasks need. This prevents the first starter
        # from borrowing CPUs its ready siblings need. Tasks awaiting a busy worker or memory
        # do not count, so their wait does not withhold useful borrowing.
        counted = {key: lane for key, *_, lane in sends}
        free = {i for i, handle in enumerate(self.group.handles) if handle.busy is None}
        committed = running_cores  # the minima of running tasks, this pass's sends included
        prospective: dict[Any, int] = {}  # unpinned affinity -> the worker its first counted task takes
        for key in ready:
            if not free:
                break
            lane = self.group.authority.owners.get(self.owners[key])
            if key in counted or key in self.gathers or key in refused or lane is None:
                continue
            candidate = self.indexed[key]
            if committed + candidate.resources.min_cores > self.cores:
                continue
            seat = None if candidate.affinity is None else self.group.affinities.get(candidate.affinity)
            if seat is None:
                if candidate.affinity is not None and candidate.affinity in prospective:
                    continue
                seat = min(free)
                if candidate.affinity is not None:
                    prospective[candidate.affinity] = seat
            elif seat not in free:
                continue
            free.discard(seat)
            committed += candidate.resources.min_cores
            counted[key] = lane
        for key, lane in self.expected.items():
            if counted.get(key) != lane:
                self.group.authority.groups[lane].withdraw_contender()
        for key, lane in counted.items():
            if self.expected.get(key) != lane:
                self.group.authority.groups[lane].dispatch_contender()
        sent = {key for key, *_ in sends}
        self.expected = {key: lane for key, lane in counted.items() if key not in sent}

    def receive(self, ready, dispatched):
        if self.group is not None:
            if ready and all(k not in self.pending for k in ready):
                for counters in self.group.authority.groups:
                    counters.open_gate()
            for worker, reply in self.group.receive(timeout=0.0 if dispatched else CPU_POLL_SECONDS):
                self.completed(worker, reply)
            for worker, released_affinities in self.group.released:
                for identity, resident in tuple(self.residents.items()):
                    if identity[0] == worker and identity[1] not in released_affinities:
                        resident.release(record=False)
                        del self.residents[identity]
            self.group.released.clear()
            self.group.reclaim_attachments()
        elif not dispatched:
            if not ready and (not self.running):
                raise InvariantError("graph has pending work but no executable dependency")
            time.sleep(WAKE_SECONDS)

    def run(self):
        if not self.prepare():
            return self.results
        try:
            with signal_boundary(self.cancel):
                self.start_workers()
                while self.pending or self.running or self.expansions:
                    if self.stopped:
                        raise RunInterrupted(signal.SIGINT)
                    ready = self.ready_tasks()
                    self.refresh_lanes(ready)
                    dispatched, ready = self.dispatch_pass(ready)
                    self.receive(ready, dispatched)
                if self._session is not None:
                    self._session.completed.update(self.results.keys() & self._session.chain_keys)
                    if self.group is not None:
                        self.group.authority.reconcile(set(), self._session.completed)
        except BaseException as error:
            from .records import record_task_failure

            try:
                failed = self.group.failed_task if self.group is not None else next(iter(self.running), None)
                if failed is not None and not isinstance(error, (KeyboardInterrupt, SystemExit)):
                    record_task_failure(failed, error)
            except Exception:  # attribution is bookkeeping: it must never replace the run's own error
                _LOG.debug("could not attribute the failed task", exc_info=True)
            if self._session is not None:
                self._session.close(abort=True)
            elif self.group is not None:
                self.group.close(abort=True)
            raise
        finally:
            self.progress.close()
            retained = self.residents.values() if self._session is None else ()
            for token in (*self.requests.values(), *retained):
                token.release()
            if self.group is not None and self._session is None:
                self.group.close()
            if self.in_process and self._session is None:
                from .workers import release_local_state

                release_local_state()
            if self.injected_governor:
                set_governor(self.previous_governor)
        order = (*self.original_order, *sorted(self.results.keys() - set(self.original_order)))
        return {key: self.results[key] for key in order if self.retain_intermediates or key in self.public_keys}
