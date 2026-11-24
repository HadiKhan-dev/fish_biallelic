"""Reuse complete report families from checked task results and their published files.

Interface layer: it binds report tasks after the stage graph (stages may not import outputs, and outputs may not
import the store's code identities)."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from ..core.digest import json_digest
from ..core.files import placement_lock
from ..engine.graph import RecordKey, ResultRef, Task, TaskKey


def _identity(roots, options, inputs):
    from ..store.identity import code_dependency_identity

    receipts = []
    for key, value in inputs:
        if isinstance(value, str):
            receipts.append((key, value))
        elif isinstance(value, dict) and value and all(
            isinstance(record, RecordKey) and isinstance(identity, str) for record, identity in value.items()
        ):
            receipts.append((key, sorted(value.items())))
        else:
            return None
    code = code_dependency_identity(roots, pinned=False).digest
    return json_digest(dict(schema=1, code=code, options=options, inputs=receipts))


def _check(path, code, options, inputs):
    identity = _identity(code, options, inputs)
    if identity is None:
        return None
    try:
        saved = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if saved.get("identity") != identity:
        return None
    output = path.parent.parent
    files = tuple(output / name for name in saved["files"])
    if all(file.is_relative_to(output) and file.is_file() for file in files):
        return files
    return None


def _collect(cached, function, args, kwargs):
    if cached is None:
        return function(*args, **kwargs)
    return None


def _publish(cached, path, code, options, inputs, function, args, kwargs):
    if cached is not None:
        return list(cached)

    from ..outputs.reports.tables import write_json

    path.parent.mkdir(parents=True, exist_ok=True)
    with placement_lock(path):
        # A failed multi-file write must leave no receipt certifying the previous family.
        path.unlink(missing_ok=True)
        value = function(*args, **kwargs)
        identity = _identity(code, options, inputs)
        if identity is not None:
            paths = [value] if isinstance(value, Path) else value
            output = path.parent.parent.absolute()
            files = [str(file.absolute().relative_to(output)) for file in paths]
            write_json(path, dict(identity=identity, files=files))
    return value


def bind_report_tasks(tasks: tuple[Task, ...]) -> tuple[Task, ...]:
    """Keep fresh collectors independent and check a prior family once before reading any parts."""
    families = defaultdict(list)
    for task in tasks:
        if task.key.stage == "report":
            families[task.key.round].append(task)

    replacements = {}
    checks = []
    for family, members in families.items():
        gather = next((task for task in members if task.key.unit == "run"), None)
        if gather is None or not gather.args:
            continue
        external = tuple(dict.fromkeys(
            dependency
            for task in members
            for dependency in task.depends_on
            if dependency.stage != "report"
        ))
        if not external:
            continue
        # All family collectors are identified in their prescribed order, including excluded-contig selections.
        options = [tuple(task.key for task in members), gather.kwargs.get("reverse_conventions")]
        if family == "export_manifest":
            # the checkpoint root's path is not a cause: the export receipts in ``inputs`` bind its records, and a
            # path in the identity made equal runs at different roots disagree (interruption canary, 4.22)
            _, output, _, full_length, _ = gather.args
            options.append(full_length)
        else:
            output = gather.args[0]
        if family == "pair_sharing":
            options.append(next(task.kwargs["genetic_maps"].identity() for task in members if task != gather))
        path = Path(output) / ".report_receipts" / (family + ".json")
        roots = tuple(dict.fromkeys(task.function.__module__ for task in members))
        inputs = tuple((key, ResultRef(key)) for key in external)
        cached = None
        dependencies = tuple(dict.fromkeys((*gather.depends_on, *external)))
        if path.is_file():
            key = TaskKey("report", "check", "run", family)
            checks.append(Task(key, _check, (path, roots, options, inputs), depends_on=external))
            cached = ResultRef(key)
            dependencies += (key,)
            for task in members:
                if task.key != gather.key:
                    replacements[task.key] = replace(
                        task, function=_collect, args=(cached, task.function, task.args, task.kwargs),
                        kwargs={}, depends_on=(*task.depends_on, key),
                    )
        replacements[gather.key] = replace(
            gather, function=_publish,
            args=(cached, path, roots, options, inputs, gather.function, gather.args, gather.kwargs),
            kwargs={}, depends_on=dependencies,
        )
    return (*tuple(replacements.get(task.key, task) for task in tasks), *checks)
