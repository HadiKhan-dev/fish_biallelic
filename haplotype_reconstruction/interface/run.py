"""Compose validated settings, workflow graph cuts and output tasks at the application boundary.

Owns fail-fast ordering, run paths and user-facing command composition (ENG-02/13).
Science, task scheduling and checkpoint formats belong to their lower layers.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import traceback
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.errors import ResourceError, SettingsError
from ..core.files import atomic_path
from ..results.paths import RunPaths
from ..stages.context import default_scratch
from ..stages.export import records as export_records
from .settings.load import default_output
from .settings.schema import Settings

if TYPE_CHECKING:
    from ..contracts.genetic_map import GeneticMapSet
    from ..contracts.recombination import RecombinationMap
    from ..core.ties import TieCensus, TieSettings
    from ..engine.graph import Expansion, RecordKey, Task, TaskKey
    from ..engine.records import RunRecorder
    from ..outputs.export.records import ExportConfig
    from ..results.reader import Results
    from ..stages.configure import ResolvedConfiguration, StageConfigs
    from ..stages.graph import Cut, RunGraph
    from ..stages.inputs.sources import CanonicalOrderSource
    from ..store.store import CheckpointStore

__all__ = ["RunPaths", "RunResult", "run_paths", "resolve_cores", "error_report"]


@dataclass(frozen=True)
class RunResult:
    """A complete or cleanly stopped invocation, with its processed contig names."""

    state: str
    stopped_after: str | None
    contigs: tuple[str, ...]


def run_paths(settings: Settings) -> RunPaths:
    """Resolve output, checkpoint and log defaults without making directories."""
    output = settings.run.output or default_output(settings)
    checkpoints = settings.run.checkpoints or output / "checkpoints"
    return RunPaths(output, checkpoints, output / "logs")


def resolve_cores(settings: Settings) -> int:
    """Resolve auto to affinity and reject any explicit ceiling outside it."""
    from ..engine.runtime import affinity_cpus

    available = affinity_cpus()
    value = settings.execution.cores
    if value != "auto" and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        raise SettingsError("[execution] cores must be auto or a positive integer")
    cores = available if settings.execution.cores == "auto" else int(settings.execution.cores)
    if not 1 <= cores <= available:
        raise ResourceError(f"[execution] cores = {cores} exceeds the CPU affinity ({available})")
    return cores


def error_report(error: BaseException, args: argparse.Namespace) -> Path:
    """Atomically write an internal-error traceback and return the report path."""
    directory = getattr(args, "output", None) or getattr(args, "run_directory", None)
    directory = Path(directory) / "logs" if directory else Path(tempfile.gettempdir()) / "haplotypes-errors"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%f")
    path = directory / f"error_{stamp}_{os.getpid()}.json"
    payload: dict[str, Any] = {
        "command": args.command,
        "type": type(error).__name__,
        "message": str(error),
        "traceback": "".join(traceback.format_exception(error)),
    }
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
    return path


@dataclass(frozen=True)
class TieCensusSink:
    """Append explicit tie-census events to the per-process JSONL log."""

    directory: Path

    def __call__(self, event: TieCensus) -> None:
        import fcntl
        import json

        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"census.{os.getpid()}.jsonl"
        row = dict(kind=event.kind, key=event.key, values=list(event.values))
        with path.open("a") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class PreparedRun:
    """The fully checked invocation; construction performs no science or writes."""

    settings: Settings
    paths: RunPaths
    cores: int
    source: CanonicalOrderSource | None
    parameters: StageConfigs
    genetic_maps: GeneticMapSet
    manifest: dict[str, Any] | None


def memory_limit(settings: Settings) -> int | None:
    """Translate the execution-only auto sentinel for the runtime's memory governor."""
    value = settings.execution.memory_limit
    return None if value == "auto" else int(value)


def _existing_store(paths: RunPaths) -> CheckpointStore | None:
    """Inspect a present store without creating an absent output or checkpoint root."""
    from ..store.container import CheckpointMismatch
    from ..store.store import CheckpointStore

    if not paths.checkpoints.exists():
        return None
    if not paths.checkpoints.is_dir():
        raise CheckpointMismatch(f"{paths.checkpoints}: checkpoints must be a directory; use a new output root")
    try:
        store = CheckpointStore(paths.checkpoints)
    except RuntimeError as error:
        raise CheckpointMismatch(str(error)) from None
    store.check_layout()
    return store


def _run_checks(settings: Settings, parameters: StageConfigs, command: str) -> tuple[str, ...]:
    """Delegate the configured-contig gate to the stage graph before creating outputs."""
    from ..stages.graph import run_checks

    return run_checks(
        settings.input.contigs,
        parameters=parameters.pedigree,
        genome_wide=command in {"run", "pedigree"}
        and not settings.run.stop_after
        and not settings.execution.max_contigs,
    )


def check_run(
    settings: Settings, *, command: str = "run", tie_reverse: bool = False, contigs: tuple[str, ...] | None = None
) -> PreparedRun:
    """Apply ENG-02 fail-fast checks in order, before computation or filesystem writes.

    Paths, CPU affinity, release/layout, canonical source check, parameters, maps and
    configured-contig gates are checked in that order. Init and run may lack a bound
    manifest; run performs the same serialized initialization before its graph.
    Numerical policy is checked before importing scientific dependencies.
    """
    from ..core.errors import SettingsError
    from ..engine.runtime import configure_worker

    paths = run_paths(settings)
    cores = resolve_cores(settings)
    configure_worker(cores)
    from ..store.releases import check_directory_release

    check_directory_release(paths.output)
    store = _existing_store(paths)
    from ..stages.configure import configure, genetic_maps
    from ..stages.inputs.sources import observation_source

    try:
        source = observation_source(settings, reverse=tie_reverse)
        source.check()
    except OSError as error:
        from ..core.errors import InputError

        raise InputError(f"cannot read input source: {error}") from error
    parameters = configure(settings)
    maps = genetic_maps(settings)
    manifest = None if store is None else store.manifest("run")
    if contigs is not None and (len(set(contigs)) != len(contigs) or set(contigs) - set(settings.input.contigs)):
        raise SettingsError("selected contigs must be unique members of the initialized contig list")
    check_command = command if contigs is None or set(contigs) == set(settings.input.contigs) else "reconstruct"
    for warning in _run_checks(settings, parameters, check_command):
        print(f"WARNING: {warning}", flush=True)
    if command not in {"init", "run"} and manifest is None:
        raise SettingsError(f"{paths.output}: run has not been initialized; use haplotypes init")
    return PreparedRun(settings, paths, cores, source, parameters, maps, manifest)


def _manifest_for(
    resolved: ResolvedConfiguration, source: CanonicalOrderSource, *, tie_reverse: bool, tie_keep: tuple[str, ...]
) -> dict[str, Any]:
    """Add the source descriptor and explicit conventions to the frozen run binding."""
    from ..core.digest import jsonable
    from ..store.protocol import NumericalPolicy

    return jsonable(
        dict(
            **resolved.manifest(),
            source=dict(
                kind=source.kind,
                identity=source.identity(),
                input_identity=source.inner.identity(),
                input_sample_ids=source.input_sample_ids(),
                reference_vcf=str(source.path) if source.kind == "vcf" else None,
            ),
            ties=dict(reverse=tie_reverse, keep=tie_keep),
            numerical_policy=NumericalPolicy.current(),
        )
    )


def _match_manifest(actual: dict[str, Any], expected: dict[str, Any], *, recombination: bool = False) -> None:
    """Refuse changed upstream science; the alias binds its downstream parameters separately."""
    import copy

    from ..core.digest import canonical_json
    from ..core.errors import SettingsError

    actual, expected = copy.deepcopy(actual), copy.deepcopy(expected)
    # This is display provenance; source/payload identities are content based.
    actual.get("source", {}).pop("reference_vcf", None)
    expected.get("source", {}).pop("reference_vcf", None)
    if recombination:
        actual.get("parameters", {}).pop("recombination", None)
        expected.get("parameters", {}).pop("recombination", None)
    if canonical_json(actual) != canonical_json(expected):
        raise SettingsError("resolved configuration differs from the initialized manifest; use a new output root")


def _match_bound_configuration(
    candidate: dict[str, Any],
    manifest: dict[str, Any],
    store: CheckpointStore,
    resolved: ResolvedConfiguration,
    *,
    allow_new_recombination: bool = False,
) -> None:
    """Check immutable init science plus recombination's independently bound downstream choice."""
    from ..core.digest import canonical_json

    _match_manifest(candidate, manifest, recombination=True)
    if allow_new_recombination:
        return  # The alias checks every existing RC binding before its first write.
    changed = canonical_json(candidate["parameters"]["recombination"]) != canonical_json(
        manifest["parameters"]["recombination"]
    )
    if changed and any(not store.contig(c).has("configuration/recombination") for c in resolved.contigs):
        raise SettingsError(
            "resolved recombination configuration differs from init without a complete stage binding; "
            "use the recombination command on an independent upstream root"
        )
    _check_recombination_bindings(store, manifest, resolved)


def _output_selection(output: Path, settings: Settings) -> None:
    """Keep formatting choices beside their outputs, outside the checkpoint root."""
    payload = dict(schema=1, export_enabled=settings.export.enabled, pairwise_ibd=settings.relatedness.pairwise_ibd)
    with atomic_path(output / "output_selection.json") as temporary:
        temporary.write_text(json.dumps(payload, indent=2) + "\n")


def _initialise(prepared: PreparedRun, *, tie_reverse: bool, tie_keep: tuple[str, ...]) -> ResolvedConfiguration:
    """Serialize init, materialise a simulation once and bind its resolved configuration."""
    import fcntl

    from ..stages.configure import bind_resolved, resolve
    from ..store.protocol import mark_complete
    from ..store.store import CheckpointStore
    from .settings.load import write_settings

    assert prepared.source is not None
    store = CheckpointStore(prepared.paths.checkpoints, nthreads=prepared.cores)
    # This invocation-level lock protects random-seed resolution, including a crash
    # before manifest/run. The store's put-once operation remains the final arbiter.
    with (prepared.paths.checkpoints / ".init.lock").open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            existing = store.manifest("run")
            seeds = None if existing is None else existing.get("realised_seeds")
            if prepared.source.kind == "simulation":
                if existing is not None:
                    checked = resolve(
                        prepared.settings,
                        sample_ids=existing["sample_ids"],
                        maps=prepared.genetic_maps,
                        bound_seeds=seeds,
                    )
                    candidate = dict(
                        existing, **checked.manifest(), ties=dict(reverse=tie_reverse, keep=list(tie_keep))
                    )
                    _match_manifest(candidate, existing)
                store.bind_layout()
                simulated = prepared.source.materialise(store, prepared.settings.input.contigs, cores=prepared.cores)
                seeds = {"simulation": simulated.seed}
            resolved = resolve(
                prepared.settings,
                sample_ids=prepared.source.sample_ids(),
                maps=prepared.genetic_maps,
                bound_seeds=seeds,
            )
            manifest = _manifest_for(resolved, prepared.source, tie_reverse=tie_reverse, tie_keep=tie_keep)
            if existing is not None:
                _match_manifest(manifest, existing)
            if existing is not None:
                manifest["source"]["reference_vcf"] = existing["source"].get("reference_vcf")
            store.bind_run_manifest(manifest)
            bind_resolved(store.run, resolved)
            _output_selection(prepared.paths.output, prepared.settings)
            write_settings(prepared.settings, prepared.paths.output / "config.toml")
            mark_complete(
                store.run,
                "init",
                {"configuration/resolved": resolved.identity},
                marker_path=prepared.paths.checkpoints / "markers/init.done",
            )
            return resolved
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _resume_configuration(
    prepared: PreparedRun, *, tie_reverse: bool, tie_keep: tuple[str, ...]
) -> ResolvedConfiguration:
    """Verify the full init binding and attach an already materialised source read-only."""
    from ..core.errors import SettingsError
    from ..stages.configure import resolve
    from ..store.protocol import resume
    from ..store.store import CheckpointStore
    from ..stages.init import records as init_records

    assert prepared.source is not None
    store = CheckpointStore(prepared.paths.checkpoints, nthreads=prepared.cores)
    manifest = store.manifest("run")
    if manifest is None:
        raise SettingsError(f"{prepared.paths.output}: run has not been initialized; use haplotypes init")
    resolved = resolve(
        prepared.settings,
        sample_ids=manifest["sample_ids"],
        maps=prepared.genetic_maps,
        bound_seeds=manifest["realised_seeds"],
    )
    candidate = dict(manifest, **resolved.manifest(), ties=dict(reverse=tie_reverse, keep=list(tie_keep)))
    _match_bound_configuration(candidate, manifest, store, resolved)
    if prepared.source.kind == "simulation":
        from ..stages.inputs.simulation import COMPLETE, load_manifest

        saved = load_manifest(store)
        if saved is None or saved.state != COMPLETE:
            raise SettingsError("simulation initialization is incomplete; rerun haplotypes init")
        prepared.source.materialise(store, prepared.settings.input.contigs, cores=prepared.cores)
    _match_bound_configuration(
        _manifest_for(resolved, prepared.source, tie_reverse=tie_reverse, tie_keep=tie_keep), manifest, store, resolved
    )
    if resume(store.run, "configuration/resolved", init_records.expected(manifest, resolved)) is None:
        raise SettingsError("resolved configuration publication is missing; rerun haplotypes init")
    return resolved


def _map_output(function, items):
    """Supply formatters an ordered map bounded by this task's live CPU claim."""
    from ..engine.cpu import current_budget

    return current_budget().thread_map(function, items, label="independent output writers")


def _phase_costs(prepared: PreparedRun, resolved: ResolvedConfiguration) -> dict[str, float]:
    """Plan longest contigs first from site positions, without decoding genotypes.

    Evidence (or initial simulation reads) already carries the site-axis length.
    A fresh VCF uses declared contig lengths when available. Missing sizes get
    a neutral cost. These are scheduling hints only.
    """
    from ..store.store import CheckpointStore

    store = CheckpointStore(prepared.paths.checkpoints)
    costs = {}
    lengths = {}
    if prepared.source is not None and prepared.source.kind == "vcf":
        lengths = prepared.source.inner.header.lengths
    for contig in resolved.contigs:
        file = store.contig(contig)
        costs[contig] = float(max(1, lengths.get(contig, 1)))
        key = "evidence" if file.has("evidence") else "simulation/reads"
        if file.has(key):
            positions = file.get(key, items=("positions",), path=("payload",))["positions"]
            costs[contig] = float(max(1, len(positions)) * len(resolved.sample_ids))
    return costs


def _model_actions(prepared, resolved, cut, *, outputs=None, ties=None):
    """Bind the stage-owned factories for the selected graph cut."""
    import importlib

    from ..stages.catalog import STAGES
    from ..stages.context import StageRun

    context = StageRun(str(prepared.paths.checkpoints), str(prepared.paths.output), resolved,
                       prepared.genetic_maps, prepared.manifest, prepared.cores, ties,
                       prepared.source, str(_scratch_base(prepared)))
    actions = {} if outputs is None else {key: value for key, value in outputs.items() if key in cut.nodes}
    for key in cut.nodes:
        if key not in actions:
            module = importlib.import_module("haplotype_reconstruction." + STAGES[key.stage].tasks)
            actions[key] = module.task(key, context)
    return actions


def _complete_workflow(
    prepared: PreparedRun, cut: Cut, tasks: Sequence[Task], values: Mapping[TaskKey, object], recorder: RunRecorder
) -> None:
    """Write workflow markers only after verifying the records declared by executed tasks."""

    from ..core.errors import InvariantError
    from ..engine.graph import RecordKey
    from ..stages.graph import complete_cut
    from ..store.store import CheckpointStore

    store = CheckpointStore(prepared.paths.checkpoints)
    manifest = store.manifest("run")
    if manifest is None:
        raise InvariantError("completed workflow has no bound run manifest")
    nodes = tuple(key for key in cut.nodes if key.stage != "report")
    containers = {"run": store.run, **{c: store.contig(c) for c in prepared.settings.input.contigs}}
    outputs = {}
    for task in tasks:
        if task.key not in nodes:
            continue
        value = values[task.key]
        if task.key.stage == "pedigree":
            # The stored decision replies with its publication receipt, never the product:
            # check it against the independently bound reader and the record on disk.
            from ..results.reader import Results

            reader = Results(prepared.paths.output, checkpoints=prepared.paths.checkpoints)
            request = reader.stage_request("pedigree")
            if reader.receipt(request) is None:
                raise InvariantError("pedigree completed without its checked publication")
            checked = {RecordKey("run", request.key): request.identity}
            if value != checked:
                raise InvariantError("pedigree receipt differs from its checked publication")
            outputs[task.key] = checked
        elif (
            isinstance(value, dict)
            and value
            and all(isinstance(key, RecordKey) and isinstance(identity, str) for key, identity in value.items())
        ):
            outputs[task.key] = value
        elif isinstance(value, str) and task.outputs:
            outputs[task.key] = {key: value for key in task.outputs}
        elif task.key.stage == "discovery":
            from ..stages.discovery import records as discovery_records

            discovery_request = discovery_records.result_reader(
                store, manifest, product="discovery", contig=task.key.scope
            )
            if discovery_request is None:
                raise InvariantError("discovery completed without its checked block domain")
            outputs[task.key] = {
                RecordKey(task.key.scope, key): identity for key, identity in discovery_request.identities.items()
            }
        else:
            raise InvariantError(f"{task.key}: stage did not return its declared publication identities")
    if not nodes:
        if cut.step == "report":
            with atomic_path(prepared.paths.output / "report.done") as temporary:
                temporary.write_text("complete\n")
        return
    scopes = tuple(dict.fromkeys(key.scope for key in nodes if key.scope != "run"))
    name = cut.step + ("." + scopes[0] if cut.step in {"evidence", "reconstruct", "phase"} and len(scopes) == 1 else "")

    # the containers' envelope checks run concurrently (file reads and decodes after the graph).
    if not complete_cut(
        replace(cut, nodes=nodes),
        containers,
        outputs,
        prepared.paths.checkpoints / "markers" / (name + ".done"),
        map_work=_threaded_each,
    ):
        raise InvariantError(f"{cut.step}: declared products did not pass completion checks")
    recorder.published(
        {key.container + "/" + key.key: identity for records in outputs.values() for key, identity in records.items()}
    )
    if cut.step == "run" and not cut.clean_stop:
        with atomic_path(prepared.paths.output / "report.done") as temporary:
            temporary.write_text("complete\n")


def _task_seconds(output: Path, stage: str) -> dict[str, float]:
    """Read the last recorded contig task duration, retaining invocation order."""
    from ..engine.records import record_tasks

    seconds: dict[str, float] = {}
    for path in sorted((output / "run_records").glob("*.json")):
        for task in record_tasks(path):
            key = task["key"]
            if key.get("stage") == stage and key.get("unit") == "contig":
                value = task.get("wall_seconds")
                if isinstance(value, (float, int)) and not isinstance(value, bool):
                    seconds[key["scope"]] = float(value)
    return seconds


def _recombination_contig(
    output: Path, reader: Results, contig: str
) -> tuple[tuple[str, dict[str, Path]], RecombinationMap]:
    """Attach observed task timings from invocation records to recombination's report rows."""
    from ..outputs.reports.recombination import contig_recombination

    return contig_recombination(
        output, reader, contig, elapsed_seconds=_task_seconds(output, "recombination").get(contig)
    )


def _phase_report(output: Path, reader: Results) -> Path:
    """Format family phase's exact per-contig summaries with measured invocation task times."""
    from ..outputs.reports.tables import write_family_phase

    return write_family_phase(output, reader.phase_summaries(elapsed_seconds=_task_seconds(output, "phase")))


def _output_actions(prepared: PreparedRun, resolved: ResolvedConfiguration) -> dict[TaskKey, Task]:
    """Declare output dependencies before cuts, preserving separate scientific release records."""

    from ..engine.graph import RecordKey, ResourceRequest, ResultRef, Task, TaskKey
    from ..outputs.reports.tasks import run_tasks
    from ..results.reader import Results

    reader = Results(prepared.paths.output, checkpoints=prepared.paths.checkpoints)
    root, output = str(prepared.paths.checkpoints), str(prepared.paths.output)
    actions = {}

    def key(stage: str, c: str = "run") -> TaskKey:
        return TaskKey(stage, "run" if c == "run" else "contig", c)

    for c in resolved.contigs:
        release_inputs = tuple(key(stage, c) for stage in ("assembly", "painting", "refit", "evidence"))
        from ..stages.context import StageRun
        from ..stages.sites.tasks import task as sites_task

        context = StageRun(root, output, resolved, prepared.genetic_maps, prepared.manifest, prepared.cores,
                           source=prepared.source)
        actions[key("sites", c)] = sites_task(key("sites", c), context)
        from ..stages.release.tasks import task as release_task

        actions[key("release", c)] = release_task(key("release", c), context)
        if prepared.settings.export.enabled:
            from ..stages.export.tasks import task as export_task

            actions[key("export", c)] = export_task(key("export", c), context,
                                                   full_length=prepared.settings.export.full_length)
    dependencies = {
        "quality": tuple(key("evidence", c) for c in resolved.contigs),
        "painting": tuple(key("painting", c) for c in resolved.contigs),
        "relatedness": (key("relatedness"),),
        "pedigree": (key("pedigree"),),
        "refit": tuple(key("refit", c) for c in resolved.contigs),
        "phase": tuple(key("phase", c) for c in resolved.contigs),
        "recombination": tuple(key("recombination", c) for c in resolved.contigs),
    }
    if prepared.settings.relatedness.pairwise_ibd:
        dependencies["relatedness"] += (TaskKey("relatedness", "pairs"),)
    if reader.synthetic:
        dependencies["truth"] = (key("init"),)
    if prepared.settings.export.enabled:
        from ..stages.export.tasks import manifest_tasks

        declarations = manifest_tasks(context, full_length=prepared.settings.export.full_length)
        actions.update((task.key, task) for task in declarations)
        manifest_key = declarations[-1].key
    dependencies["complete"] = tuple(dict.fromkeys(k for values in dependencies.values() for k in values))
    dependencies["science"] = dependencies["complete"]
    if prepared.settings.export.enabled:
        dependencies["complete"] += (manifest_key,)
    for task in run_tasks(
        reader, prepared.paths.output, dependencies, reverse_conventions=reader.manifest["ties"]["reverse"]
    ):
        if task.key.round == "recombination" and task.key.unit == "contig":
            task = replace(task, function=_recombination_contig, kwargs={})
        elif task.key.round == "phase":
            task = replace(task, function=_phase_report, kwargs={})
        if task.key.round == "pedigree":
            task = replace(
                task, resources=ResourceRequest(1, prepared.cores), kwargs={**task.kwargs, "map_work": _map_output}
            )
        actions[task.key] = task
    return actions


def _attach_reports(graph: RunGraph, cut: Cut) -> Cut:
    """Attach writers whose science is in this cut, retaining the checked external prerequisites."""

    from ..engine.graph import TaskKey

    if cut.step not in {"run", "evidence", "relatedness", "reconstruct", "pedigree", "phase", "recombination"}:
        return cut
    selected = set(cut.nodes)
    for node in graph.nodes:
        if node.key.stage == "report" and set(node.dependencies) <= selected:
            selected.add(node.key)
    # A scatter cut cannot publish a run-level family until all its parts are ready.
    selected = {
        key
        for key in selected
        if not (
            key.stage == "report"
            and key.unit == "contig"
            and key.round in {"quality", "painting", "recombination", "ties", "export_manifest"}
            and TaskKey("report", "run", "run", key.round) not in selected
        )
    }
    return replace(cut, nodes=tuple(node.key for node in graph.nodes if node.key in selected))


def _prerequisites(prepared, resolved, cut):
    """Resolve the cut's prerequisites through stage-owned requests and one receipt check."""
    from ..results.reader import Results
    from ..stages.init import records as init_records
    from ..stages.init.tasks import check
    from ..outputs.export.records import SCHEMA, ExportConfig

    results = Results(prepared.paths.output, checkpoints=prepared.paths.checkpoints)
    values = {}
    for key in cut.prerequisites:
        contig = None if key.scope == "run" else key.scope
        if key.stage == "init":
            values[key] = check(str(prepared.paths.checkpoints), init_records.expected(results.manifest, resolved))
            continue
        if key.stage == "export":
            identity = export_records.expected(results, contig, ExportConfig(full_length=prepared.settings.export.full_length))
            saved = results.output_receipt("outputs/export", identity, SCHEMA, contig)
            if saved is None:
                raise SettingsError(f"{contig}: current export is not published")
            values[key] = identity
            continue
        request = results.stage_request(key.stage, contig, round_name=key.round, pairs=key.unit == "pairs")
        if results.receipt(request, contig) is None:
            raise SettingsError(f"workflow prerequisite {key.stage} for {key.scope} is not published")
        values[key] = request.identity
    return values


def execute(
    settings: Settings,
    *,
    command: str = "run",
    contig: str | None = None,
    contigs: tuple[str, ...] | None = None,
    tie_reverse: bool = False,
    tie_keep: tuple[str, ...] = (),
    tie_census: Path | None = None,
    in_process: bool = False,
) -> RunResult:
    """Execute one validated graph cut, returning a clean-stop result when requested."""
    prepared = check_run(settings, command=command, tie_reverse=tie_reverse, contigs=contigs)
    if command == "run" and prepared.manifest is None:
        from ..engine.runtime import isolated

        if prepared.source is not None and prepared.source.kind == "simulation":
            # Init materialises a simulation with its own worker pool. Its separate numerical runtime
            # keeps the subsequent graph free to declare its own complete preload set.
            isolated(
                execute,
                settings,
                cores=prepared.cores,
                command="init",
                tie_reverse=tie_reverse,
                tie_keep=tie_keep,
                tie_census=tie_census,
            )
        else:
            # An observation source's init reads its header and binds the manifest: no kernel, pool or worker, so
            # it runs here, without a second interpreter's start-up and imports before every run.
            execute(
                settings,
                command="init",
                tie_reverse=tie_reverse,
                tie_keep=tie_keep,
                tie_census=tie_census,
                in_process=in_process,
            )
        prepared = check_run(settings, command=command, tie_reverse=tie_reverse, contigs=contigs)
    from ..engine.records import record_run, run_log
    from ..engine.runtime import kernel_run
    from .settings.schema import to_dict

    resolved = (
        None if command == "init" else _resume_configuration(prepared, tie_reverse=tie_reverse, tie_keep=tie_keep)
    )
    tasks: tuple[Task, ...] = ()
    completed: dict[TaskKey, Any] = {}
    cut = None
    if command != "init":
        assert resolved is not None
        from ..engine.graph import TaskKey, run_graph
        from ..results.reader import Results
        from ..stages.graph import Node, bind_tasks, build_graph, execution_options, workflow_cut, task_preloads, with_kernel_tasks

        output_actions = _output_actions(prepared, resolved)
        output_nodes = tuple(Node(task.key, task.depends_on) for task in output_actions.values())
        if settings.relatedness.pairwise_ibd:
            output_nodes += (
                Node(
                    TaskKey("relatedness", "pairs"),
                    tuple(TaskKey("evidence", "contig", c) for c in settings.input.contigs),
                ),
            )
        graph = build_graph(settings.input.contigs, output_nodes=output_nodes)
        painted: tuple[str, ...] = ()
        if settings.execution.max_contigs:
            reader = Results(prepared.paths.output, checkpoints=prepared.paths.checkpoints)
            painted = tuple(c for c in settings.input.contigs if reader.stage_present(c, "painting"))
        cut = workflow_cut(
            graph,
            command,
            contig=contig,
            contigs=contigs,
            stop_after=(settings.run.stop_after or None) if command == "run" else None,
            serial=settings.serial if command == "run" else False,
            max_contigs=(settings.execution.max_contigs or None) if command == "run" else None,
            painted=painted,
        )
        cut = _attach_reports(graph, cut)
        completed = _prerequisites(prepared, resolved, cut)
        from ..core.ties import TieSettings

        ties = TieSettings(
            reverse=bool(prepared.manifest["ties"]["reverse"]),
            keep=tuple(prepared.manifest["ties"].get("keep", ())),
            census=None if tie_census is None else TieCensusSink(tie_census.expanduser().resolve()),
        )
        tasks = bind_tasks(
            graph, cut, _model_actions(prepared, resolved, cut, outputs=output_actions, ties=ties), completed=completed
        )
        # the contigs' size estimates (sites x samples) also order and weight the chromosome chains:
        # the largest contigs claim lanes first and a chain at least twice the median keeps a proportional floor.
        sizes = _phase_costs(prepared, resolved)
        scheduling = execution_options(graph, cut, contig_costs=sizes, phase_costs=sizes)
        preload = task_preloads(tasks)
    else:
        preload = ()
    # Init reads its source's header and binds the manifest; it runs kernels only when it simulates the source.
    # Warming kernels during observation-only init would duplicate the following run's compile-pool startup.
    warm = command != "init" or (prepared.source is not None and prepared.source.kind == "simulation")
    if warm:
        _start_runtime(prepared, preload=preload)
    # else: an observation init (no kernel, pool or worker) installs no numerical runtime, so the run that follows in
    # this interpreter starts the only one: its kernels bind to that runtime's cache directory, and a live
    # forkserver's settings are never replaced by init's.
    if command != "init" and not in_process:
        from ..engine.runtime import start_forkserver

        start_forkserver()  # its workers' imports run while the graph is planned
    prepared.paths.logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%f")
    log = prepared.paths.logs / f"{command}_{stamp}_{os.getpid()}.log"
    config_record = dict(
        to_dict(settings),
        developer=dict(
            tie_reverse=tie_reverse, tie_keep=list(tie_keep), tie_census=None if tie_census is None else str(tie_census)
        ),
    )
    # Worker-pool graphs compile missing kernels beside unrelated work. Each scientific root waits for its own
    # numerical modules before it can expand into workers; init's simulation and in-process graphs warm eagerly.
    defer = command != "init" and not in_process
    with (
        run_log(log),
        record_run(
            prepared.paths.output,
            command,
            log,
            config=config_record,
            checkpoints=prepared.paths.checkpoints,
            source_descriptor=None if prepared.manifest is None else prepared.manifest.get("source"),
        ) as recorder,
        kernel_run(store=_kernel_store(prepared), defer=defer) if warm else nullcontext() as kernels,
    ):
        if command == "init":
            _initialise(prepared, tie_reverse=tie_reverse, tie_keep=tie_keep)
            return RunResult("complete", "init", settings.input.contigs)
        if command not in {"evidence", "reconstruct", "phase"}:
            _output_selection(prepared.paths.output, settings)
        assert cut is not None
        if defer:
            tasks, scheduling = with_kernel_tasks(tasks, scheduling, kernels)
        values = run_graph(
            tasks,
            cores=prepared.cores,
            in_process=in_process,
            initial_results=completed,
            **scheduling,
            on_task=recorder.task_finished,
            contig_lanes=None if settings.execution.contig_lanes == "auto" else int(settings.execution.contig_lanes),
            preload=preload,
        )
        if defer:
            from ..engine.kernels import record_compiled

            assert kernels is not None  # a deferring graph is a warm one
            record_compiled(kernels, values)
        _complete_workflow(prepared, cut, tasks, values, recorder)
        if cut.clean_stop:
            recorder.clean_stop()
    selected = tuple(c for c in settings.input.contigs if any(key.scope == c for key in cut.nodes))
    return RunResult(
        "stopped" if cut.clean_stop else "complete",
        "painting" if settings.execution.max_contigs else settings.run.stop_after or None,
        selected,
    )


def _recorded_run(
    settings: Settings, directory: str | Path, *, recombination: bool = False
) -> tuple[PreparedRun, ResolvedConfiguration]:
    """Check settings against an initialized run without reopening its observation source."""

    from ..engine.runtime import configure_worker

    cores = resolve_cores(settings)
    configure_worker(cores)
    paths = RunPaths.from_directory(directory)
    settings = replace(settings, run=replace(settings.run, output=paths.output, checkpoints=paths.checkpoints))
    store = _existing_store(paths)
    from ..stages.configure import configure, genetic_maps, resolve

    parameters = configure(settings)
    maps = genetic_maps(settings)
    manifest = None if store is None else store.manifest("run")
    if manifest is None or store is None:
        raise SettingsError(f"{paths.output}: run has not been initialized; use haplotypes init")
    resolved = resolve(settings, sample_ids=manifest["sample_ids"], maps=maps, bound_seeds=manifest["realised_seeds"])
    from ..core.digest import jsonable
    from ..store.protocol import NumericalPolicy

    candidate = dict(manifest, **resolved.manifest(), numerical_policy=jsonable(NumericalPolicy.current()))
    _match_bound_configuration(candidate, manifest, store, resolved, allow_new_recombination=recombination)
    return PreparedRun(settings, paths, cores, None, parameters, maps, manifest), resolved


def _load_recorded_settings(
    directory: Path,
    cores: int | None = None,
    *,
    options: Mapping[str, Mapping[str, Any]] | None = None,
    overrides: Sequence[str] = (),
) -> Settings:
    """Check provenance before settings decoding, after establishing numerical policy."""
    from ..engine.runtime import configure_worker
    from .settings.load import load_settings

    # The container codec needs NumPy, but not Numba. Establish the scalar
    # library policy before reading it; select the actual Numba capacity from
    # the checked settings before returning to any scientific command imports.
    configure_worker(1)
    paths = RunPaths.from_directory(directory)
    _existing_store(paths)
    requested = {name: dict(values) for name, values in (options or {}).items()}
    if cores is not None:
        requested.setdefault("execution", {})["cores"] = cores
    settings = load_settings(directory / "config.toml", options=requested, overrides=overrides)
    configure_worker(resolve_cores(settings))
    return settings


@contextmanager
def _command_record(prepared: PreparedRun, command: str):
    """Record standalone command failures, resource usage and actual worker timings."""
    from ..engine.records import record_run, run_log
    from ..engine.runtime import kernel_run
    from .settings.schema import to_dict

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%f")
    log = prepared.paths.logs / f"{command}_{stamp}_{os.getpid()}.log"
    with (
        run_log(log),
        record_run(
            prepared.paths.output,
            command,
            log,
            config=to_dict(prepared.settings),
            checkpoints=prepared.paths.checkpoints,
            source_descriptor=None if prepared.manifest is None else prepared.manifest.get("source"),
        ) as recorder,
        kernel_run(store=_kernel_store(prepared)),
    ):
        yield recorder


def _evaluation_tasks(tasks: Any, reader: Any, *, cores: int, recorder: Any) -> list[Any]:
    """Schedule one independent chromosome evaluation per engine task in manifest order."""
    from ..engine.graph import ResourceRequest, Task, TaskKey, run_graph
    from ..evaluation.tasks import evaluate_task

    declarations = tuple(
        Task(
            TaskKey("evaluation", "contig", task.contig),
            evaluate_task,
            args=(task, reader),
            resources=ResourceRequest(
                working_set_bytes=task.working_set_bytes(
                    reader.record_bytes(task.contig, "truth"),
                    [
                        reader.record_bytes(task.contig, stage)
                        for stage in (
                            *task.founder_stages,
                            *(stage for stage in ("family_phase", "recombination") if stage in task.requested),
                        )
                    ],
                )
            ),
        )
        for task in tasks
    )
    result = run_graph(
        declarations,
        cores=cores,
        in_process=False,
        on_task=recorder.task_finished,
        preload=("haplotype_reconstruction.evaluation.tasks",),
    )
    return [result[task.key] for task in declarations]


def templates_command(args: argparse.Namespace) -> int:
    """Apply the numerical/runtime boundary before template generation imports or computes template windows."""
    from ..core.errors import InputError
    from ..engine.runtime import affinity_cpus, configure_worker, kernel_run, start_run

    available = affinity_cpus()
    cores = available if args.cores is None else args.cores
    if not isinstance(cores, int) or isinstance(cores, bool) or cores < 1:
        raise SettingsError("template cores must be a positive integer")
    if cores > available:
        raise ResourceError("template cores must fit the current CPU affinity")
    contigs = tuple(args.contigs or ())
    if not contigs or len(set(contigs)) != len(contigs):
        raise InputError("templates need the unique contigs to model (--contigs)")
    configure_worker(cores)
    from ..core.kernels import declarations
    from ..engine.kernels import cache_directory, cache_key
    from ..stages.inputs.vcf import inspect
    from .templates.builder import templates_command as build

    try:
        header = inspect(args.vcf)
    except OSError as error:
        raise InputError(f"cannot read input source: {error}") from error
    missing = [contig for contig in contigs if contig not in header.contigs]
    if missing:
        raise InputError(f"{args.vcf}: contigs not in the header: {', '.join(missing)}")
    output = args.output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise InputError("template output directory must be new or empty")
    roots = tuple(
        dict.fromkeys(
            (
                "haplotype_reconstruction.interface.templates.builder",
                "haplotype_reconstruction.interface.templates.linker",
                "haplotype_reconstruction.science.evidence.run",
                "haplotype_reconstruction.science.discovery.run",
                *(entry.module for entry in declarations()),
            )
        )
    )
    cache_base = output.parent / ".haplotypes-kernels"
    cache = cache_directory(cache_key(("haplotype_reconstruction.engine.workers", *roots)), base=cache_base / "active")
    start_run(cores, preload=roots, cache_directory=cache)
    with kernel_run(store=cache_base / "store"):
        return build(args)


def evaluate_command(args: argparse.Namespace) -> int:
    """Validate every requested product before runtime writes, then schedule evaluation."""
    settings = _load_recorded_settings(args.run_directory, args.cores)
    prepared, _ = _recorded_run(settings, args.run_directory)
    from ..evaluation.evaluate import collect, plan_evaluation, write_evaluation
    from ..results.reader import Results

    reader = Results(prepared.paths.output, checkpoints=prepared.paths.checkpoints)
    report, tasks = plan_evaluation(reader, stages=tuple(args.stages), contigs=args.contigs)
    _start_runtime(prepared, preload=("haplotype_reconstruction.evaluation.tasks",))
    with _command_record(prepared, "evaluate") as recorder:
        evaluated = _evaluation_tasks(tasks, reader, cores=prepared.cores, recorder=recorder)
        components, matches = collect(report, [result.parts() for result in evaluated])
        target = prepared.paths.output / "evaluation"
        write_evaluation(target, report, components, matches)
        print(f"Evaluation written to {target}", flush=True)
    return 0


def export_command(args: argparse.Namespace) -> int:
    """Format selected published releases into a new destination, publishing its manifest last."""
    settings = _load_recorded_settings(args.run_directory, args.cores)
    from ..stages.export.tasks import map_writers

    prepared, _ = _recorded_run(settings, args.run_directory)
    from ..core.files import is_placement_lock
    from ..engine.graph import run_graph
    from ..outputs.export.records import ExportConfig
    from ..outputs.export.run import write_founder_identity, write_manifest
    from ..outputs.export.task import make_task
    from ..outputs.export.vcf import suffix
    from ..results.reader import Results

    reader = Results(prepared.paths.output, checkpoints=prepared.paths.checkpoints)
    contigs = reader.contigs if args.contigs is None else tuple(args.contigs)
    if not contigs or len(set(contigs)) != len(contigs) or set(contigs) - set(reader.contigs):
        raise SettingsError("export contigs must be unique names in the initialized run")
    destination = args.destination.expanduser().resolve()
    # Empty placement locks left by an earlier failed export into this destination are not contents.
    if destination.exists() and (
        not destination.is_dir()
        or any(not is_placement_lock(entry, (suffix("vcf"), suffix("bcf"))) for entry in destination.iterdir())
    ):
        raise SettingsError("export destination must be new or empty")
    config = ExportConfig(
        full_length=args.full_length, file_format=args.format, products=tuple(dict.fromkeys(args.products))
    )
    tasks = [
        make_task(
            reader,
            destination,
            contig,
            config,
            export_records.expected(reader, contig, config),
            max_cores=prepared.cores,
            map_work=map_writers,
        )
        for contig in contigs
    ]
    _start_runtime(prepared, preload=("haplotype_reconstruction.outputs.export.task",))
    with _command_record(prepared, "export") as recorder:
        destination.mkdir(parents=True, exist_ok=True)
        completed = run_graph(
            tasks,
            cores=prepared.cores,
            in_process=False,
            on_task=recorder.task_finished,
            preload=("haplotype_reconstruction.outputs.export.task",),
        )
        entries = [dict(completed[task.key].entry) for task in tasks]
        write_founder_identity(destination, reader, contigs, config.products)
        write_manifest(
            destination,
            prepared.paths.checkpoints,
            synthetic=reader.synthetic,
            reference=None if reader.synthetic else str(settings.input.vcf),
            contigs=entries,
            settings=config,
        )
        print(f"Export: {len(entries)} chromosomes written to {destination}", flush=True)
    return 0


def _check_recombination_bindings(
    store: CheckpointStore, manifest: dict[str, Any], resolved: ResolvedConfiguration
) -> None:
    """Check all downstream bindings before any recombination task or report is written."""
    from ..contracts.roles import require_schema
    from ..core.errors import InvariantError
    from ..stages.recombination import records
    from ..store.protocol import read_items
    from ..results.protocol import check_receipt

    candidate = dict(manifest, **resolved.manifest())

    def check(contig: str) -> None:
        file = store.contig(contig)
        request = records.result_reader(
            store, candidate, product="recombination", contig=contig, parameters=resolved.parameters.recombination
        )
        if request is None:
            if file.has("identity/recombination") or file.has("recombination"):
                raise InvariantError(f"{contig}: recombination adapter omitted an existing binding")
            return
        if file.has("identity/recombination"):
            binding = read_items(file, "identity/recombination", request.identity, ("schema", "derivation"))
            assert binding is not None
            require_schema(binding.payload, 1)
            if binding.payload["derivation"] != request.identity:
                raise InvariantError(f"{contig}: recombination binding payload differs from its derivation")
        check_receipt(file, request)

    # each contig derives its upstream chain (recombination, phase, painting, assembly, selection,
    # discovery readers: envelope reads and decodes); contig after contig this took 20-30 s of a resumed run's start
    # and 165 s of an export of one contig. The checks are independent reads; the first error in contig order wins.
    _threaded_each(check, resolved.contigs)


def _threaded_each(function: Callable[[Any], Any], items: Sequence[Any]) -> list[Any]:
    """core.concurrency.threaded_each: bounded threads for a coordinator's independent record reads."""
    from ..core.concurrency import threaded_each

    return threaded_each(function, items)


def rerun_recombination(settings: Settings, run_directory: str | Path) -> RunResult:
    """Alias the recombination graph cut; copied completed roots keep their bindings.

    Changed downstream parameters can run only against an independent upstream
    checkpoint root with no recombination binding or product. Settings and tables
    are rewritten only after every requested contig has completed successfully.
    """
    prepared, resolved = _recorded_run(settings, run_directory, recombination=True)
    from ..engine.graph import run_graph
    from ..engine.records import record_run, run_log
    from ..engine.runtime import kernel_run
    from ..stages.graph import Node, bind_tasks, build_graph, workflow_cut, task_preloads
    from ..store.store import CheckpointStore
    from .settings.load import write_settings
    from .settings.schema import to_dict

    store = CheckpointStore(prepared.paths.checkpoints, nthreads=prepared.cores)
    assert prepared.manifest is not None
    _check_recombination_bindings(store, prepared.manifest, resolved)
    outputs = _output_actions(prepared, resolved)
    graph = build_graph(
        resolved.contigs, output_nodes=tuple(Node(task.key, task.depends_on) for task in outputs.values())
    )
    cut = _attach_reports(graph, workflow_cut(graph, "recombination"))
    completed = _prerequisites(prepared, resolved, cut)
    tasks = bind_tasks(graph, cut, _model_actions(prepared, resolved, cut, outputs=outputs), completed=completed)
    preload = task_preloads(tasks)
    _start_runtime(prepared, preload=preload)
    prepared.paths.logs.mkdir(parents=True, exist_ok=True)
    log = prepared.paths.logs / f"recombination_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S_%f')}_{os.getpid()}.log"
    with (
        run_log(log),
        record_run(
            prepared.paths.output,
            "recombination",
            log,
            config=to_dict(prepared.settings),
            checkpoints=prepared.paths.checkpoints,
            source_descriptor=None if prepared.manifest is None else prepared.manifest.get("source"),
        ) as recorder,
        kernel_run(store=_kernel_store(prepared)),
    ):
        values = run_graph(
            tasks,
            cores=prepared.cores,
            in_process=False,
            initial_results=completed,
            preload=preload,
            on_task=recorder.task_finished,
        )
        _complete_workflow(prepared, cut, tasks, values, recorder)
        write_settings(prepared.settings, prepared.paths.output / "config.toml")
    return RunResult("complete", None, resolved.contigs)


def _scratch_base(prepared: PreparedRun) -> Path:
    """Resolve the invocation's explicit scratch choice or checkpoint-adjacent default."""
    return prepared.settings.execution.scratch or default_scratch(prepared.paths.checkpoints)


def _kernel_store(prepared: PreparedRun) -> Path:
    """Keep kernel warm-up's disposable persistent cache on the invocation's selected scratch filesystem."""
    return _scratch_base(prepared) / "kernel_store"


def _start_runtime(prepared: PreparedRun, *, preload: tuple[str, ...] = ()) -> None:
    """Enter the engine with declared task roots and a code-keyed optional scratch namespace."""
    from ..engine.runtime import start_run

    # evidence forwards the registered declarations after importing these roots. Keep
    # the caller roots stable across repeated calls in the same interpreter.
    roots = tuple(dict.fromkeys(("haplotype_reconstruction.interface.run", *preload)))
    cache = None
    if prepared.settings.execution.scratch is not None:
        from ..engine.kernels import cache_directory, cache_key

        closure = tuple(dict.fromkeys(("haplotype_reconstruction.engine.workers", *roots)))
        cache = cache_directory(cache_key(closure), base=prepared.settings.execution.scratch / "kernels")
    start_run(
        prepared.cores,
        preload=roots,
        profile=prepared.settings.execution.profile,
        memory_limit=memory_limit(prepared.settings),
        memory_diagnostics=prepared.paths.logs,
        cache_directory=cache,
    )
