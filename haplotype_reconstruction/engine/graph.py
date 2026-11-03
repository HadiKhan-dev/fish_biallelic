"""Declare tasks and validate their model dependencies before scheduler execution.

Tasks are plain frozen records carrying importable callables and explicit data.
The same graph runs in process or through forkserver workers. Results are placed
by task key, never completion order; worker exceptions/deaths propagate promptly.
Working-set and scratch estimates are declarations only and never block admission.
Scheduler owns expansion, admission, elastic shares and resident affinity; this
module owns frozen contracts and dependency validation only (P0 ENG-20/22).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

from ..core.errors import SettingsError

__all__ = ["TaskKey", "RecordKey", "ResourceRequest", "ResultRef", "Task", "dependency_order", "run_graph", "Expansion"]


@dataclass(frozen=True, order=True)
class TaskKey:
    """A semantic task name: stage, unit, scope (run or contig identifier) and round."""

    stage: str
    unit: str
    scope: str = "run"
    round: str = ""

    # The coordinator hashes and compares keys millions of times per run (every dictionary of tasks, results,
    # owners and dependencies is keyed by them): the hash is computed once per object and equal-by-identity keys
    # skip the field comparison. Same semantics as the generated methods; the cached hash is never pickled (string
    # hashes differ between processes).
    def __hash__(self) -> int:
        try:
            return self.__dict__["_hash"]
        except KeyError:
            value = hash((self.stage, self.unit, self.scope, self.round))
            object.__setattr__(self, "_hash", value)
            return value

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (self.stage, self.unit, self.scope, self.round) == (other.stage, other.unit, other.scope, other.round)

    def __getstate__(self) -> dict:
        return {"stage": self.stage, "unit": self.unit, "scope": self.scope, "round": self.round}


@dataclass(frozen=True, order=True)
class RecordKey:
    """A declared input/output key in a run or named contig container."""

    container: str
    key: str


@dataclass(frozen=True)
class ResourceRequest:
    """A CPU range and reported working-set/scratch bytes, never memory reservations."""

    min_cores: int = 1
    max_cores: int = 1
    working_set_bytes: int = 0
    scratch_bytes: int = 0


@dataclass(frozen=True)
class ResultRef:
    """An explicit reference to a dependency's returned value inside task args/kwargs."""

    task: TaskKey


@dataclass(frozen=True)
class Task:
    """One importable action and its semantic dependencies and resource declaration.

    ``function(*args, **kwargs)`` receives ResultRef values resolved from completed
    dependencies. Args may contain refs in tuples/lists/dicts. ``thread_argument``
    optionally names a keyword receiving the allocated core count. Science-specific
    inputs remain the stage's typed records. ``affinity`` pins a chain to its resident
    worker; it does not enter scientific identities.
    """

    key: TaskKey
    function: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    depends_on: tuple[TaskKey, ...] = ()
    inputs: tuple[RecordKey, ...] = ()
    outputs: tuple[RecordKey, ...] = ()
    resources: ResourceRequest = field(default_factory=ResourceRequest)
    affinity: str | None = None
    thread_argument: str | None = None


@dataclass(frozen=True)
class Expansion:
    """Independent outer subtasks followed by a gather in the parent context."""

    subtasks: tuple[Task, ...]
    gather: Task


def _references(value):
    if isinstance(value, ResultRef):
        yield value.task
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _references(item)
    elif is_dataclass(value) and (not isinstance(value, type)):
        for item in fields(value):
            yield from _references(getattr(value, item.name))
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _references(item)


def dependency_order(tasks: Sequence[Task]) -> tuple[TaskKey, ...]:
    """Return a stable topological order, refusing missing dependencies, cycles or duplicate writers."""
    indexed = {task.key: task for task in tasks}
    if len(indexed) != len(tasks):
        raise SettingsError("task keys must be unique")
    writers: dict[RecordKey, TaskKey] = {}
    for task in tasks:
        request = task.resources
        if request.min_cores < 1 or request.max_cores < request.min_cores:
            raise SettingsError(f"{task.key}: invalid CPU range {request.min_cores}..{request.max_cores}")
        missing = set(task.depends_on) - indexed.keys()
        if missing:
            raise SettingsError(f"{task.key}: unknown dependencies {sorted(missing)}")
        undeclared = set(_references((task.args, task.kwargs))) - set(task.depends_on)
        if undeclared:
            raise SettingsError(f"{task.key}: result references are not declared dependencies {sorted(undeclared)}")
        for output in task.outputs:
            if output in writers:
                raise SettingsError(f"{output}: multiple writers {writers[output]} and {task.key}")
            writers[output] = task.key
    # Sorted layers whose dependencies all lie in earlier layers. Kahn's algorithm visits each task and
    # dependency once; sorting orders the keys within each layer.
    waiting = {key: len(set(task.depends_on)) for key, task in indexed.items()}
    dependents: dict[TaskKey, list[TaskKey]] = {}
    for key, task in indexed.items():
        for dependency in set(task.depends_on):
            dependents.setdefault(dependency, []).append(key)
    layer = sorted(key for key, count in waiting.items() if count == 0)
    order: list[TaskKey] = []
    while layer:
        order.extend(layer)
        following = []
        for key in layer:
            for dependent in dependents.get(key, ()):
                waiting[dependent] -= 1
                if not waiting[dependent]:
                    following.append(dependent)
        layer = sorted(following)
    if len(order) != len(indexed):
        raise SettingsError(f"task graph contains a dependency cycle: {sorted(set(indexed) - set(order))}")
    return tuple(order)


def cyclic_keys(depends: Mapping[TaskKey, Sequence[TaskKey]]) -> list[TaskKey]:
    """The keys of ``depends`` (key -> its dependencies) that no topological order reaches: those on a cycle and
    those behind one, sorted. Dependencies outside the mapping count as finished."""
    waiting = {key: sum(1 for dependency in deps if dependency in depends) for key, deps in depends.items()}
    dependents: dict[TaskKey, list[TaskKey]] = {}
    for key, deps in depends.items():
        for dependency in deps:
            if dependency in depends:
                dependents.setdefault(dependency, []).append(key)
    ready = [key for key, count in waiting.items() if not count]
    while ready:
        for dependent in dependents.get(ready.pop(), ()):
            waiting[dependent] -= 1
            if not waiting[dependent]:
                ready.append(dependent)
    return sorted(key for key, count in waiting.items() if count)


_FIELD_NAMES: dict[type, tuple[str, ...] | None] = {}


def _field_names(item) -> tuple[str, ...] | None:
    """A dataclass instance's field names (cached per type); None for anything else, including classes."""
    kind = type(item)
    try:
        return _FIELD_NAMES[kind]
    except KeyError:
        names = tuple(f.name for f in fields(item)) if is_dataclass(item) and not isinstance(item, type) else None
        _FIELD_NAMES[kind] = names
        return names


def _readonly(value):
    """Task-shared arrays are read-only, including after process transport.

    Each container is visited once (a subtree shared by several fields, such as a retained decision referenced
    from two records, is not walked again) and atomic values are never queued: large typed views hold millions
    of scalar fields. Object arrays are containers too: their elements are walked.
    """
    import numpy as np

    atomic = (str, bytes, int, float, complex, bool, type(None), np.generic)
    seen: set[int] = set()
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, np.ndarray):
            item.flags.writeable = False
            if item.dtype.hasobject and id(item) not in seen:
                # The flag freezes only the object references; their arrays are frozen here too.
                seen.add(id(item))
                pending.extend(child for child in item.flat if not isinstance(child, atomic))
            continue
        if isinstance(item, atomic) or id(item) in seen:
            continue
        names = _field_names(item)
        if names is not None:
            seen.add(id(item))
            for name in names:
                child = getattr(item, name)
                if not isinstance(child, atomic):
                    pending.append(child)
        elif isinstance(item, (list, tuple)):
            seen.add(id(item))
            pending.extend(child for child in item if not isinstance(child, atomic))
        elif isinstance(item, Mapping):
            seen.add(id(item))
            pending.extend(child for child in item.values() if not isinstance(child, atomic))
    return value


def run_graph(
    tasks: Sequence[Task], *, cores: int = 1, in_process: bool = True, max_workers: int | None = None, **options: Any
) -> dict[TaskKey, Any]:
    """Run the full E1 scheduler, including expansion and affinity residency.

    Public task records stay stable. The lazy forwarding import avoids a cycle
    between these contracts and the implementation of their single authority.
    """
    from .scheduler import run_graph as execute

    return execute(tasks, cores=cores, in_process=in_process, max_workers=max_workers, **options)
