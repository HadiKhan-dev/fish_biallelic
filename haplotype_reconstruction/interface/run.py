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
    """Resolve the 3.5 output/checkpoint/log defaults without making directories."""
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
    """Append explicit C1 events in the baseline per-process JSONL form (maths/ties.py:435)."""

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

    Paths, CPU affinity, canonical source check, parameters, maps, layout and
    configured-contig gates are checked in that order. Init and run may lack a bound
    manifest; run performs the same serialized initialization before its graph.
    Numerical policy is checked before importing scientific dependencies.
    """
    from ..core.errors import SettingsError
    from ..engine.runtime import configure_worker

    paths = run_paths(settings)
    cores = resolve_cores(settings)
    configure_worker(cores)
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
    store = _existing_store(paths)
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
    """Add the source descriptor and explicit conventions to G1's frozen binding."""
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


def _initial_identity(manifest: dict[str, Any], resolved: ResolvedConfiguration) -> str:
    """Rebuild G1's immutable init identity from its declared manifest fields."""
    from ..core.digest import json_digest

    return json_digest({key: manifest[key] for key in resolved.manifest()})


def _match_bound_configuration(
    candidate: dict[str, Any],
    manifest: dict[str, Any],
    store: CheckpointStore,
    resolved: ResolvedConfiguration,
    *,
    allow_new_recombination: bool = False,
) -> None:
    """Check immutable init science plus RC1's independently bound downstream choice."""
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
    if resume(store.run, "configuration/resolved", _initial_identity(manifest, resolved)) is None:
        raise SettingsError("resolved configuration publication is missing; rerun haplotypes init")
    return resolved


def _check_initialized(root: str, identity: str) -> str:
    """A graph init node verifies its prerequisite publication and returns its identity."""
    from ..store.protocol import resume
    from ..store.store import CheckpointStore

    if resume(CheckpointStore(root).run, "configuration/resolved", identity) is None:
        raise SettingsError("run has not been initialized; use haplotypes init")
    return identity


def _map_output(function, items):
    """Supply formatters an ordered map bounded by this task's live CPU claim."""
    from ..engine.scheduler import current_budget

    return current_budget().thread_map(function, items, label="independent output writers")


def _map_writers(function, items):
    """Release 4.10: an ordered map giving each output writer a live share of this task's CPUs (its BGZF compression
    and index threads; core.concurrency.live_threads), one thread per writer before."""
    from ..engine.scheduler import current_budget

    items = list(items)
    results: list[Any] = [None] * len(items)
    calls = [lambda share, item=item: function(item) for item in items]
    for index, value in current_budget().concurrent(calls, label="output writers"):
        results[index] = value
    return results


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


def _model_actions(
    prepared: PreparedRun,
    resolved: ResolvedConfiguration,
    cut: Cut,
    *,
    outputs: dict[TaskKey, Task] | None = None,
    ties: TieSettings | None = None,
) -> dict[TaskKey, Task]:
    """Bind concrete stage declarations to the model; no stage is treated as a no-op."""
    from ..core.errors import InvariantError
    from ..engine.graph import RecordKey, ResourceRequest, Task, TaskKey

    function: Callable[..., Any]
    args: tuple[Any, ...]
    actions: dict[TaskKey, Task] = (
        {} if outputs is None else {key: task for key, task in outputs.items() if key in cut.nodes}
    )
    needed = {key.stage for key in cut.nodes}
    if "init" in needed:
        key = TaskKey("init", "run", "run")
        actions[key] = Task(
            key,
            _check_initialized,
            args=(str(prepared.paths.checkpoints), _initial_identity(prepared.manifest or {}, resolved)),
            outputs=(RecordKey("run", "configuration/resolved"),),
        )
    if "evidence" in needed:
        if prepared.source is None:
            raise SettingsError("evidence requires a checked observation source")
        from ..stages.evidence.tasks import tasks

        selected = tuple(key.scope for key in cut.nodes if key.stage == "evidence")
        actions.update(
            (task.key, task)
            for task in tasks(
                prepared.paths.checkpoints,
                prepared.source,
                resolved.parameters.evidence,
                contigs=selected,
                maximum_cores=prepared.cores,
            )
        )
    if "relatedness" in needed:
        for key in (key for key in cut.nodes if key.stage == "relatedness"):
            pairs = key.unit == "pairs"
            actions[key] = Task(
                key,
                _run_relatedness,
                args=(str(prepared.paths.checkpoints), resolved, prepared.genetic_maps),
                kwargs={
                    "pairs": pairs,
                    "reverse": bool((prepared.manifest or {}).get("ties", {}).get("reverse", False)),
                },
                inputs=tuple(RecordKey(c, "evidence") for c in resolved.contigs),
                outputs=(RecordKey("run", "relatedness/pairs" if pairs else "relatedness/families"),),
                resources=ResourceRequest(min_cores=1, max_cores=prepared.cores),
                thread_argument="team",
            )
    if "discovery" in needed:
        for key in (key for key in cut.nodes if key.stage == "discovery"):
            actions[key] = Task(
                key,
                _run_discovery,
                (str(prepared.paths.checkpoints), str(prepared.paths.output), key.scope, resolved, ties),
                kwargs={"cores": prepared.cores},
                inputs=(RecordKey(key.scope, "evidence"),),
                # Up to the allocation: reading the contig's inputs decompresses with the task's live share.
                resources=ResourceRequest(1, prepared.cores),
            )
    for key in (key for key in cut.nodes if key.stage == "selection"):
        actions[key] = Task(
            key,
            _run_selection,
            (
                str(prepared.paths.checkpoints),
                str(prepared.paths.output),
                key.scope,
                key.round,
                resolved,
                prepared.genetic_maps,
                ties,
            ),
            kwargs={"cores": prepared.cores},
            resources=ResourceRequest(1, prepared.cores),
        )
    for key in (key for key in cut.nodes if key.stage in {"painting", "refit"}):
        args = (str(prepared.paths.checkpoints), str(prepared.paths.output), key.scope, resolved)
        if key.stage == "painting":
            function = _run_painting
            args += (prepared.genetic_maps, ties)
            published = ("painting", "painting/summary", "identity/painting")
        else:
            function = _run_refit
            args += (ties,)
            published = ("founder_refit", "founder_refit/summary", "identity/founder_refit")
        actions[key] = Task(
            key,
            function,
            args,
            kwargs={"cores": prepared.cores} if key.stage == "painting" else {},
            outputs=tuple(RecordKey(key.scope, name) for name in published),
            resources=ResourceRequest(1, prepared.cores),
            thread_argument="team",
        )
    for key in (key for key in cut.nodes if key.stage in {"phase", "recombination"}):
        args = (str(prepared.paths.checkpoints), str(prepared.paths.output), key.scope, resolved, prepared.genetic_maps)
        if key.stage == "phase":
            function = _run_phase
            args += (str(_scratch_base(prepared)),)
            # Family phase maps its independent work over the task's live thread budget (thread_map, input order
            # kept): declared serial, a contig's phase ran on one CPU however many were idle.
            resources = ResourceRequest(1, prepared.cores)
        else:
            function = _run_recombination
            resources = ResourceRequest(1, 1)
        actions[key] = Task(key, function, args, outputs=(RecordKey(key.scope, key.stage),), resources=resources)
    for key in (key for key in cut.nodes if key.stage == "assembly"):
        actions[key] = Task(
            key,
            _run_assembly,
            (
                str(prepared.paths.checkpoints),
                str(prepared.paths.output),
                key.scope,
                key.round,
                resolved,
                prepared.genetic_maps,
                ties,
            ),
            kwargs={"cores": prepared.cores},
            resources=ResourceRequest(1, prepared.cores),
        )
    if "pedigree" in needed:
        from ..stages.pedigree.prepare import pipeline_task

        task = pipeline_task(
            str(prepared.paths.checkpoints), prepared.genetic_maps, prepared.manifest, cores=prepared.cores
        )
        actions[task.key] = task
    missing = tuple(key for key in cut.nodes if key not in actions)
    if missing:
        raise InvariantError(f"workflow cut has no concrete action for {missing}")
    return actions


def _required_product(reader: Results, adapter: Any, product: str, contig: str | None) -> tuple[Any, str]:
    """Decode a required upstream through its owner, returning its verified derivation."""
    request = adapter.result_reader(reader._store, reader.manifest, product=product, contig=contig)
    value = None if request is None else reader.read(request, contig)
    if value is None:
        raise SettingsError(f"{contig}: required {product} publication is missing")
    return value, request.identity


def _run_discovery(
    root: str, output: str, contig: str, resolved: ResolvedConfiguration, ties: TieSettings, *, cores: int = 1
) -> Expansion:
    """Expand a contig into DI2's plan, independent block tasks and checked gather."""
    from ..engine.graph import Expansion
    from ..science.discovery.run import block_evidence
    from ..stages.discovery.tasks import discovery_tasks
    from ..stages.evidence import records
    from ..store.protocol import NumericalPolicy

    reader = _task_reader(output, root)
    from ..stages.discovery import records as discovery_records
    from ..stages.discovery.tasks import complete_contig_receipt
    from ..stages.framework import RecordKey, Task, TaskKey

    # Release 4.10: a contig whose plan and every block are already published under the identities this run derives
    # (the reader checks each record's envelope; no marker is trusted) needs only its completion step. Expanding it
    # resumed every block as its own task (chr3 0.5x: 7,533 tasks, each decoding and returning its stored result).
    done = discovery_records.result_reader(reader._store, reader.manifest, product="discovery", contig=contig)
    if done is not None:
        identities = dict(done.identities)
        return Expansion(
            (),
            Task(
                TaskKey("discovery", "complete", contig),
                complete_contig_receipt,
                (reader._store.contig(contig), contig, identities),
                inputs=tuple(RecordKey(contig, key) for key in identities),
                affinity=contig,
            ),
        )
    evidence, identity = _required_product(reader, records, "evidence", contig)
    founder = evidence.founder_view()
    blocks = tuple(block_evidence(founder, index) for index in range(len(founder.block_spans)))
    tasks = discovery_tasks(
        contig,
        blocks,
        resolved.parameters.discovery,
        file=reader._store.contig(contig),
        evidence_identity=identity,
        numerical_policy=NumericalPolicy(**reader.manifest["numerical_policy"]),
        ties=ties,
        max_cores=cores,
    )
    return Expansion(tasks[:-1], tasks[-1])


@dataclass(frozen=True)
class SelectionInputs:
    """Picklable capability loading checked discovery and evidence only when a round needs them."""

    root: str
    output: str
    contig: str

    def __call__(self) -> Any:
        from ..stages.discovery import records as discovery_records

        reader = _task_reader(self.output, self.root)
        request = discovery_records.result_reader(
            reader._store, reader.manifest, product="discovery", contig=self.contig
        )
        if request is None:
            raise SettingsError(f"{self.contig}: required discovery publication is missing")
        discovered = _chain_input(f"discovery/{request.identity}", lambda: _decoded(reader, request, self.contig))
        _, _, founder, view = _chain_evidence(reader, self.contig)
        return discovered, founder, view


# A contig's selection rounds and assembly entries run as separate actions on the contig's worker (its affinity) and
# each needs the contig's discovery and evidence, decoded. They are kept there between those actions
# (engine.workers.resident_or_load) under keys naming their store identities, so a kept value equals a fresh read and
# any other worker reads the store; the final assembly's entry, their last consumer, releases them.
_CHAIN_INPUTS = "chain-inputs"
# Sample rows per evidence-count gather for the assembly depth keys (memory bound only: 32 rows of int32[L, 2]).
DEPTH_KEY_ROWS = 32


def _chain_input(key: str, load: Callable[[], Any]) -> Any:
    from ..engine.workers import resident_or_load

    return resident_or_load(f"{_CHAIN_INPUTS}/{key}", load)


def _decoded(reader: Results, request: Any, contig: str) -> Any:
    value = reader.read(request, contig)
    if value is None:
        raise SettingsError(f"{contig}: required {request.key} publication is missing")
    return value


def _chain_evidence(reader: Results, contig: str) -> tuple[Any, Any, Any, Any]:
    """Cache count-free evidence and depth keys; defer assembly arrays to their consumer."""
    import numpy as np

    from ..contracts.evidence import ContigEvidence
    from ..science.assembly.evidence import AssemblyEvidenceSource, _depth_keys
    from ..stages.evidence import records as evidence_records

    request = evidence_records.result_reader(reader._store, reader.manifest, product="evidence", contig=contig)
    if request is None:
        raise SettingsError(f"{contig}: selection and assembly need complete evidence")

    def load() -> tuple[Any, Any, Any]:
        evidence = reader.read(request, contig)
        if not isinstance(evidence, ContigEvidence):
            raise SettingsError(f"{contig}: selection and assembly need complete evidence")
        founder = evidence.founder_view()
        # The keys consume counts once. Decode DEPTH_KEY_ROWS sample rows at a time rather than retaining an
        # N x L x 2 count buffer for the rest of the chain (release 4.8: batches, hashed on spare threads; one row
        # per gather made this serial chain step ~14 s for chr1 at N = 320).
        from ..engine.scheduler import spare_threads

        columns = np.arange(evidence.n_sites, dtype=np.int64)
        keys = []
        threads = spare_threads(DEPTH_KEY_ROWS)
        with evidence_records.open_arrays(
            reader._store.contig(contig), ("counts",), expected_identity=request.identity
        ) as arrays:
            if arrays is None:
                raise SettingsError(f"{contig}: assembly depth keys need evidence counts")
            for first in range(0, evidence.n_samples, DEPTH_KEY_ROWS):
                stop = min(evidence.n_samples, first + DEPTH_KEY_ROWS)
                rows = arrays["counts"].gather(slice(first, stop), columns)
                observed = np.asarray(founder.observed[first:stop])
                keys.extend(_depth_keys(rows, observed, founder.discovery_sites, threads))
                del rows
        return evidence, founder, tuple(keys)

    items = ",".join(getattr(request, "items", None) or ())
    evidence, founder, depth_keys = _chain_input(f"evidence/{request.identity}/{items}", load)
    source = AssemblyEvidenceSource(
        founder.positions,
        founder.likelihoods,
        founder.observed,
        founder.discovery_sites,
        request.identity,
        depth_keys,
        bool(evidence.discovery_sites.all() and not evidence.founder_excluded.any()),
    )
    return request, evidence, founder, source


@dataclass(frozen=True)
class SelectionContext:
    """Read the already published context assembly without creating a second assembly task."""

    root: str
    output: str
    contig: str
    round_name: str

    def __call__(self) -> Any:
        from ..stages.assembly import records

        reader = _task_reader(self.output, self.root)
        request = records.result_reader(
            reader._store,
            reader.manifest,
            product="assembly",
            contig=self.contig,
            round_name=self.round_name,
        )
        context = None if request is None else reader.read(request, self.contig)
        if context is None:
            raise SettingsError(f"{self.contig}: selection needs published context {self.round_name}")
        return context


def _selected_result(root: str, output: str, contig: str, round_name: str, completed: object) -> dict[RecordKey, str]:
    """Verify the round publication before returning its graph completion identity."""
    from ..contracts.roles import require_schema
    from ..engine.graph import RecordKey
    from ..stages.selection import records
    from ..store.protocol import read_items

    reader = _task_reader(output, root)
    request = records.result_reader(
        reader._store,
        reader.manifest,
        product="selected",
        contig=contig,
        stage="local_selection_" + round_name,
    )
    saved = (
        None
        if request is None
        else read_items(reader._store.contig(contig), request.key, request.identity, ("schema",))
    )
    if request is None or saved is None:
        raise SettingsError(f"{contig}: selection completed without its {round_name} publication")
    require_schema(saved.payload, request.schema)
    return {RecordKey(contig, request.key): request.identity}


def _run_selection(
    root: str,
    output: str,
    contig: str,
    round_name: str,
    resolved: ResolvedConfiguration,
    maps: GeneticMapSet,
    ties: TieSettings,
    *,
    cores: int = 1,
) -> Expansion:
    """Expand precisely one selection round, consuming G1's preceding context assembly."""
    from ..contracts.roles import require_schema
    from ..engine.graph import Expansion, RecordKey, ResultRef, Task, TaskKey
    from ..stages.discovery import records as discovery_records
    from ..stages.evidence import records as evidence_records
    from ..stages.selection.tasks import SelectionSession, selection_round_task
    from ..store.protocol import NumericalPolicy, read_items

    reader = _task_reader(output, root)
    # Release 4.10: a round whose completion is published under the identities this run derives (each promised block,
    # memo and proposal record checked against its identity) is not expanded: its selected-record receipt is the
    # gather's. Expanding decoded every panel of the round in its entry task before the gather checked the same record.
    from ..stages.selection import records as selection_records

    done = selection_records.result_reader(
        reader._store, reader.manifest, product="selected", contig=contig, stage="local_selection_" + round_name
    )
    if done is not None and done.scopes[done.round_index].selected(verify=True) is not None:
        saved = read_items(reader._store.contig(contig), done.key, done.identity, ("schema",))
        if saved is not None:
            require_schema(saved.payload, done.schema)
            return {RecordKey(contig, done.key): done.identity}  # type: ignore[return-value]
    upstream = {}
    for name, adapter in (("evidence", evidence_records), ("discovery", discovery_records)):
        request = adapter.result_reader(reader._store, reader.manifest, product=name, contig=contig)
        saved = (
            None
            if request is None
            else read_items(reader._store.contig(contig), request.key, request.identity, ("schema",))
        )
        if request is None or saved is None:
            raise SettingsError(f"{contig}: selection requires published {name}")
        require_schema(saved.payload, request.schema)
        upstream[name] = request.identity
    session = SelectionSession(
        reader._store,
        contig,
        resolved.parameters.local_selection,
        resolved.parameters.discovery,
        resolved.parameters.assembly,
        maps.for_contig(contig),
        upstream,
        SelectionInputs(root, output, contig),
        reverse=ties.reverse,
        tie_keep=ties.keep,
        census_sink=ties.census,
        max_threads=cores,
        numerical_policy=NumericalPolicy(**reader.manifest["numerical_policy"]),
    )
    child = selection_round_task(
        session,
        round_name,
        load_context=None if round_name == "initial" else SelectionContext(root, output, contig, round_name),
        key=TaskKey("selection", "select", contig, round_name),
    )
    gather = Task(
        TaskKey("selection", "published", contig, round_name),
        _selected_result,
        (root, output, contig, round_name, ResultRef(child.key)),
        depends_on=(child.key,),
        affinity=contig,
    )
    return Expansion((child,), gather)


def _publication_result(
    root: str, output: str, contig: str, round_name: str | None, completed: object
) -> dict[RecordKey, str]:
    """Return only independently verified assembly publication identities after its expansion."""
    from ..contracts.roles import require_schema
    from ..engine.graph import RecordKey
    from ..stages.assembly import records
    from ..store.protocol import read_items

    reader = _task_reader(output, root)
    request = records.result_reader(
        reader._store, reader.manifest, product="assembly", contig=contig, round_name=round_name
    )
    saved = (
        None
        if request is None
        else read_items(reader._store.contig(contig), request.key, request.identity, ("schema",))
    )
    if request is None or saved is None:
        raise SettingsError(f"{contig}: assembly completed without its declared publication")
    require_schema(saved.payload, request.schema)
    return {RecordKey(contig, request.key): request.identity}


def _run_assembly(
    root: str,
    output: str,
    contig: str,
    round_name: str,
    resolved: ResolvedConfiguration,
    maps: GeneticMapSet,
    ties: TieSettings,
    *,
    cores: int = 1,
) -> Expansion:
    """Expand one AS6 context/final role from checked selected panels and founder evidence."""
    from ..engine.graph import Expansion, ResultRef, Task, TaskKey
    from ..science.assembly.run import AssemblyInputs, Role
    from ..stages.assembly.final import assembly_task, bind_session
    from ..stages.selection import records as selection_records
    from ..store.protocol import NumericalPolicy

    reader = _task_reader(output, root)
    context = None if not round_name else round_name.removeprefix("context.")
    # Release 4.10: a role already published under the identities this run derives returns the receipt its gather
    # (_publication_result) would, after the same checks, instead of decoding the selected panels, building the
    # evidence view and decoding the stored product's diagnostics again.
    from ..contracts.roles import require_schema
    from ..engine.graph import RecordKey
    from ..stages.assembly import records as assembly_records
    from ..store.protocol import read_items

    done = assembly_records.result_reader(
        reader._store, reader.manifest, product="assembly", contig=contig, round_name=context
    )
    if done is not None:
        saved = read_items(reader._store.contig(contig), done.key, done.identity, ("schema",))
        if saved is not None:
            require_schema(saved.payload, done.schema)
            return {RecordKey(contig, done.key): done.identity}  # type: ignore[return-value]
    previous = "l2" if context is None else "initial" if context == "l1" else "l1"
    request = selection_records.result_reader(
        reader._store, reader.manifest, product="selected", contig=contig, stage="local_selection_" + previous
    )
    selected = None if request is None else reader.read(request, contig)
    if request is None or selected is None:
        raise SettingsError(f"{contig}: assembly needs published selection round {previous}")
    evidence_request, evidence, _, source = _chain_evidence(reader, contig)
    from ..engine.scheduler import numba_threads, spare_threads

    # The views are built in this serial chain step, which the contig's other work waits for: the task's CPUs plus
    # its lane's idle ones (release 4.9).
    with numba_threads(spare_threads(DEPTH_KEY_ROWS)):
        view = source()  # no assembly arrays live in the cached chain tuple
    if context is None:
        from ..engine.workers import release_resident_namespace

        release_resident_namespace(_CHAIN_INPUTS)  # the final assembly is the chain inputs' last consumer
    child_key = TaskKey("assembly", "compute", contig, round_name)
    session = bind_session(
        AssemblyInputs(tuple(selected), view, maps.for_contig(contig)),
        resolved.parameters.assembly,
        file=reader._store.contig(contig),
        upstream=dict(
            panels=request.identity,
            evidence=evidence_request.identity,
            payload=evidence.payload_identity,
            map=dict(resolved.map_identities)[contig],
        ),
        role=Role.FINAL if context is None else Role.CONTEXT,
        levels=None if context is None else int(context[1:]),
        records_prefix="assembly" if context is None else f"local_selection/{context}/assembly",
        parent=child_key,
        max_cores=cores,
        ties=ties,
        numerical_policy=NumericalPolicy(**reader.manifest["numerical_policy"]),
    )
    child = assembly_task(session)
    gather = Task(
        TaskKey("assembly", "published", contig, round_name),
        _publication_result,
        (root, output, contig, context, ResultRef(child.key)),
        depends_on=(child.key,),
        affinity=contig,
    )
    return Expansion((child,), gather)


def _run_painting(
    root: str,
    output: str,
    contig: str,
    resolved: ResolvedConfiguration,
    maps: GeneticMapSet,
    ties: TieSettings,
    *,
    team: int = 1,
    cores: int = 1,
) -> str:
    """Bind AS6's checked final assembly to PA2's window-capable contig action."""
    from ..stages.assembly import records as assembly_records
    from ..stages.painting.tasks import identity, run_contig
    from ..store.protocol import NumericalPolicy

    reader = _task_reader(output, root)
    assembly, assembled = _required_product(reader, assembly_records, "assembly", contig)
    evidence_id = _evidence_identity(reader._store, resolved, contig, reverse=ties.reverse)
    file = reader._store.contig(contig)
    from ..store.protocol import bound_payload_identity

    payload = bound_payload_identity(file)
    expected = identity(
        resolved.parameters.painting,
        contig=contig,
        sample_ids=resolved.sample_ids,
        assembly_identity=assembled,
        evidence_identity=evidence_id,
        payload_identity=payload,
        map_identity=dict(resolved.map_identities)[contig],
        ties=ties,
        numerical_policy=NumericalPolicy(**reader.manifest["numerical_policy"]),
    )
    run_contig(
        file,
        expected,
        (assembly.prepared, assembly.components),
        maps.for_contig(contig),
        resolved.parameters.painting,
        evidence_identity=evidence_id,
        payload_identity=payload,
        ties=ties,
        threads=team,
        max_threads=cores,
        census=ties.census is not None,
    )
    return expected


def _required_dense_evidence(reader: Results, contig: str) -> tuple[Any, str]:
    """Read one checked dense view for a whole-contig consumer, without a source twin."""
    from ..stages.evidence import records

    request = records.result_reader(reader._store, reader.manifest, product="evidence", contig=contig)
    if request is None:
        raise SettingsError(f"{contig}: required evidence publication is missing")
    return records.load_dense(reader._store.contig(contig), expected_identity=request.identity), request.identity


def _run_refit(
    root: str, output: str, contig: str, resolved: ResolvedConfiguration, ties: TieSettings, *, team: int = 1
) -> str:
    """Pass the checked immutable painting and evidence to RF1's one contig action."""
    from ..stages.painting import records as painting_records
    from ..stages.refit.tasks import identity, run_contig
    from ..store.protocol import NumericalPolicy

    reader = _task_reader(output, root)
    painting, painted = _required_product(reader, painting_records, "painting", contig)
    evidence, evidence_id = _required_dense_evidence(reader, contig)
    expected = identity(
        resolved.parameters.founder_refit,
        contig=contig,
        sample_ids=resolved.sample_ids,
        painting_identity=painted,
        evidence_identity=evidence_id,
        payload_identity=evidence.payload_identity,
        ties=ties,
        numerical_policy=NumericalPolicy(**reader.manifest["numerical_policy"]),
    )
    run_contig(
        reader._store.contig(contig),
        expected,
        painting,
        evidence,
        resolved.parameters.founder_refit,
        payload_identity=evidence.payload_identity,
        ties=ties,
        threads=team,
    )
    return expected


def _run_phase(
    root: str, output: str, contig: str, resolved: ResolvedConfiguration, maps: GeneticMapSet, scratch: str | None
) -> str:
    """Bind the canonical typed upstream products to PH1's complete-contig publisher."""
    from ..contracts.phase import PhaseInputIdentities
    from ..core.digest import json_digest
    from ..stages.painting import records as painting_records
    from ..stages.pedigree import records as pedigree_records
    from ..stages.phase import records
    from ..stages.phase.tasks import publish_contig
    from ..stages.refit import records as refit_records
    from ..store.protocol import NumericalPolicy

    reader = _task_reader(output, root)
    evidence, evidence_id = _required_dense_evidence(reader, contig)
    painting, painting_id = _required_product(reader, painting_records, "painting", contig)
    refit, refit_id = _required_product(reader, refit_records, "refit", contig)
    pedigree, _ = _required_product(reader, pedigree_records, "pedigree", None)
    identities = PhaseInputIdentities(
        evidence_id, painting_id, refit_id, pedigree.edges.identity, dict(resolved.map_identities)[contig]
    )
    expected = records.derivation(
        contig,
        resolved.sample_ids,
        identities,
        resolved.parameters.family_phase,
        reverse=reader.manifest["ties"]["reverse"],
        policy=NumericalPolicy(**reader.manifest["numerical_policy"]),
    )
    publish_contig(
        reader._store.contig(contig),
        evidence,
        painting,
        refit,
        pedigree.edges,
        maps.for_contig(contig),
        resolved.parameters.family_phase,
        identities,
        expected,
        root=_default_scratch(Path(root)) / "phase" / json_digest(contig)[:16]
        if scratch is None
        else Path(scratch) / "phase" / json_digest(contig)[:16],
    )
    return expected.identity


def _run_recombination(
    root: str, output: str, contig: str, resolved: ResolvedConfiguration, maps: GeneticMapSet
) -> str:
    """Execute RC1 against checked phase and conditioned edges, binding before resume."""
    from ..stages.pedigree import records as pedigree_records
    from ..stages.phase import records as phase_records
    from ..stages.recombination import records
    from ..stages.recombination.tasks import publish_contig
    from ..store.protocol import NumericalPolicy

    reader = _task_reader(output, root)
    phase, phase_id = _required_product(reader, phase_records, "phase", contig)
    pedigree, _ = _required_product(reader, pedigree_records, "pedigree", None)
    reverse = reader.manifest["ties"]["reverse"]
    expected = records.derivation(
        contig,
        resolved.sample_ids,
        phase_id,
        pedigree.edges,
        resolved.parameters.recombination,
        maps.for_contig(contig),
        reverse=reverse,
        policy=NumericalPolicy(**reader.manifest["numerical_policy"]),
    )
    publish_contig(
        reader._store.contig(contig),
        phase,
        pedigree.edges,
        resolved.parameters.recombination,
        maps.for_contig(contig),
        expected,
        reverse=reverse,
    )
    return expected.identity


def _task_reader(output: str | Path, root: str | Path) -> Any:
    """The run's reader inside a graph task: its store decompresses with the task's live CPU share."""
    from ..engine.scheduler import TASK_THREADS
    from ..results.reader import Results

    return Results(output, checkpoints=root, nthreads=TASK_THREADS)


def _run_sites(root: str, output: str, contig: str, source: Any, *, team: int = 1) -> str:
    """Publish OU1's option-independent site product after checked evidence."""
    from ..stages.evidence import records
    from ..stages.inputs import sites

    reader = _task_reader(output, root)
    evidence, identity = _required_product(reader, records, "evidence", contig)
    report, _ = _required_product(reader, records, "summary", contig)
    view = sites.from_evidence(evidence, report, synthetic=reader.synthetic)
    sites.run(
        reader._store.contig(contig),
        source,
        view,
        evidence_identity=identity,
        payload_identity=evidence.payload_identity,
        team=team,
    )
    return sites.identity(identity, evidence.payload_identity, synthetic=reader.synthetic)


def _run_release(root: str, output: str, contig: str, *, team: int = 1) -> str:
    """Pass checked typed inputs to AS6's release operation without transforming them."""
    from ..stages.assembly import records as assembly_records
    from ..stages.assembly.release import ReleaseInputs, execute_release, release_identity
    from ..stages.evidence import records as evidence_records
    from ..stages.painting import records as painting_records
    from ..stages.refit import records as refit_records
    from ..store.protocol import NumericalPolicy

    reader = _task_reader(output, root)
    assembly, assembly_id = _required_product(reader, assembly_records, "assembly", contig)
    painting, painting_id = _required_product(reader, painting_records, "painting", contig)
    refit, refit_id = _required_product(reader, refit_records, "refit", contig)
    evidence, evidence_id = _required_product(reader, evidence_records, "evidence", contig)
    identity = release_identity(
        dict(
            assembly=assembly_id,
            painting=painting_id,
            refit=refit_id,
            evidence=evidence_id,
            payload=evidence.payload_identity,
        ),
        reverse=reader.manifest["ties"]["reverse"],
        numerical_policy=NumericalPolicy(**reader.manifest["numerical_policy"]),
    )
    execute_release(
        ReleaseInputs(assembly, painting, refit, evidence), reader._store.contig(contig), identity, cores=team
    )
    return identity


def _export_identity(reader: Results, contig: str, config: ExportConfig) -> str:
    """Bind formatting to its checked source products and exact selected options."""
    from ..outputs.export.records import output_identity
    from ..store.identity import code_dependency_identity

    upstream: dict[str, str] = {}
    products = set(config.products)
    names = []
    if products & {"founders", "samples"}:
        names.append("released_sites")
    if products & {"founders", "tracks"}:
        names.append("founder_release")
    if "samples" in products:
        names.append("phase")
    for name in names:
        identity = reader.product_identity(name, contig)
        if identity is None:
            raise SettingsError(f"{contig}: export requires published {name}")
        upstream[name] = identity
    code = code_dependency_identity(("outputs.export.task", "outputs.export.run")).digest
    return output_identity(upstream, config, code)


def _run_export(root: str, output: str, contig: str, full_length: bool) -> str:
    """Render checked release products and publish the output selection after its files."""
    from ..contracts.roles import require_schema
    from ..outputs.export.records import SCHEMA, ExportConfig, ExportRecord
    from ..outputs.export.task import write
    from ..store.protocol import publish_output, read_output

    reader = _task_reader(output, root)
    config = ExportConfig(full_length=full_length)
    identity = _export_identity(reader, contig, config)
    file = reader._store.contig(contig)
    previous = read_output(file, "outputs/export", identity=lambda payload: payload["identity"])
    if previous is not None:
        require_schema(previous.payload, SCHEMA)
        if previous.payload["identity"] != identity:
            # Release 4.11: the files are about to be replaced one product at a time. Until the new record is
            # published, the old record must not certify them (an interrupted replacement could leave one product
            # rerendered under it); it keeps listing its files so the replacement still retires them.
            replacing = dict(previous.payload, identity=f"replacing:{identity}")
            publish_output(file, "outputs/export", replacing["identity"], replacing, supersede=True)
            previous = read_output(file, "outputs/export", identity=lambda payload: payload["identity"])
    record = write(
        reader,
        Path(output) / "export",
        contig,
        config,
        identity,
        None if previous is None else ExportRecord(**previous.payload),
        map_work=_map_writers,
    )
    publish_output(
        file,
        "outputs/export",
        record.identity,
        dict(schema=record.schema, identity=record.identity, entry=dict(record.entry)),
        supersede=True,
    )
    return identity


def _export_manifest_part(root: str, output: str, contig: str, full_length: bool):
    """Check one published export and collect its released identity flags."""
    from ..contracts.roles import require_schema
    from ..outputs.export.records import SCHEMA, ExportConfig
    from ..store.protocol import read_output

    reader = _task_reader(output, root)
    record = read_output(reader._store.contig(contig), "outputs/export", identity=lambda p: p["identity"])
    if record is None:
        raise SettingsError(f"{contig}: export record is missing")
    expected = _export_identity(reader, contig, ExportConfig(full_length=full_length))
    if record.identity != expected:
        raise SettingsError(f"{contig}: export differs from current published products or options")
    require_schema(record.payload, SCHEMA)
    release = reader.founder_release(contig)
    if release is None:
        raise SettingsError(f"{contig}: released founder identity flags are missing")
    return dict(record.payload["entry"]), (contig, "founder_refit", release.identity_flags)


def _write_export_manifest(root: str, output: str, contigs: tuple[str, ...], full_length: bool, parts=None) -> None:
    """Publish the manifest after ordered, independently checked chromosome parts."""
    from ..outputs.export.records import ExportConfig
    from ..outputs.export.run import write_founder_identity, write_manifest

    reader = _task_reader(output, root)
    if parts is None:
        parts = tuple(_export_manifest_part(root, output, contig, full_length) for contig in contigs)
    entries, flags = zip(*parts, strict=True)
    write_founder_identity(
        Path(output) / "export", reader, contigs, ExportConfig(full_length=full_length).products, collected=flags
    )
    write_manifest(
        Path(output) / "export",
        Path(root),
        synthetic=reader.synthetic,
        reference=reader.manifest["source"].get("reference_vcf"),
        contigs=list(entries),
        settings=ExportConfig(full_length=full_length),
    )


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
            from ..contracts.roles import require_schema
            from ..stages.pedigree.records import result_reader
            from ..store.protocol import read_items

            request = result_reader(store, manifest, product="pedigree")
            saved = None if request is None else read_items(store.run, request.key, request.identity, ("schema",))
            if request is None or saved is None:
                raise InvariantError("pedigree completed without its checked publication")
            require_schema(saved.payload, request.schema)
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

    # Release 4.10: the containers' envelope checks run concurrently (file reads and decodes after the graph).
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
    """Attach observed task timings from invocation records to RC1's report rows."""
    from ..outputs.reports.recombination import contig_recombination

    return contig_recombination(
        output, reader, contig, elapsed_seconds=_task_seconds(output, "recombination").get(contig)
    )


def _phase_report(output: Path, reader: Results) -> Path:
    """Format PH1's exact per-contig summaries with measured invocation task times."""
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
        actions[key("sites", c)] = Task(
            key("sites", c),
            _run_sites,
            (root, output, c, prepared.source),
            depends_on=(key("evidence", c),),
            inputs=(RecordKey(c, "evidence"),),
            outputs=(RecordKey(c, "sites"),),
            resources=ResourceRequest(1, prepared.cores),
            thread_argument="team",
        )
        actions[key("release", c)] = Task(
            key("release", c),
            _run_release,
            (root, output, c),
            depends_on=release_inputs,
            outputs=(RecordKey(c, "release"),),
            resources=ResourceRequest(1, prepared.cores),
            thread_argument="team",
        )
        if prepared.settings.export.enabled:
            actions[key("export", c)] = Task(
                key("export", c),
                _run_export,
                (root, output, c, prepared.settings.export.full_length),
                depends_on=tuple(key(stage, c) for stage in ("sites", "release", "phase")),
                outputs=(RecordKey(c, "outputs/export"),),
                resources=ResourceRequest(1, prepared.cores),  # its writers' compression threads (release 4.10)
            )
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
        parts = tuple(
            Task(
                TaskKey("report", "contig", c, "export_manifest"),
                _export_manifest_part,
                (root, output, c, prepared.settings.export.full_length),
                depends_on=(key("export", c),),
                # Release 4.10: its identity check binds every contig's upstream chain (pedigree's inputs) on the
                # task's CPUs; at the end of a run they are idle (chr3 at 0.5x: 55 s on one CPU).
                resources=ResourceRequest(1, prepared.cores),
            )
            for c in resolved.contigs
        )
        actions.update((task.key, task) for task in parts)
        manifest_key = TaskKey("report", "run", "run", "export_manifest")
        actions[manifest_key] = Task(
            manifest_key,
            _write_export_manifest,
            (
                root,
                output,
                resolved.contigs,
                prepared.settings.export.full_length,
                tuple(ResultRef(task.key) for task in parts),
            ),
            depends_on=tuple(task.key for task in parts),
        )
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


def _prerequisites(prepared: PreparedRun, resolved: ResolvedConfiguration, cut: Cut) -> dict[TaskKey, str]:
    """Resolve workflow prerequisites through the same checked record adapters as readers."""
    import importlib

    from ..core.errors import InvariantError
    from ..results.reader import Results

    results = Results(prepared.paths.output, checkpoints=prepared.paths.checkpoints)
    values = {}
    for key in cut.prerequisites:
        if key.stage == "init":
            values[key] = _check_initialized(
                str(prepared.paths.checkpoints), _initial_identity(results.manifest, resolved)
            )
            continue
        if key.stage == "evidence":
            from ..store.store import CheckpointStore

            values[key] = _evidence_identity(
                CheckpointStore(prepared.paths.checkpoints),
                resolved,
                key.scope,
                reverse=bool(results.manifest.get("ties", {}).get("reverse", False)),
            )
            continue
        if key.stage == "export":
            from ..contracts.roles import require_schema
            from ..outputs.export.records import SCHEMA, ExportConfig
            from ..store.protocol import read_output

            expected = _export_identity(
                results, key.scope, ExportConfig(full_length=prepared.settings.export.full_length)
            )
            saved_export = read_output(results._file(key.scope), "outputs/export", identity=lambda p: p["identity"])
            if saved_export is None or saved_export.identity != expected:
                raise SettingsError(f"{key.scope}: current export is not published")
            require_schema(saved_export.payload, SCHEMA)
            values[key] = expected
            continue
        products = {
            "discovery": "discovery",
            "selection": "selected",
            "assembly": "assembly",
            "painting": "painting",
            "refit": "refit",
            "pedigree": "pedigree",
            "phase": "phase",
            "recombination": "recombination",
            "sites": "released_sites",
            "release": "founder_release",
        }
        if key.stage == "relatedness":
            module_name = (
                "stages.relatedness.pairs_records" if key.unit == "pairs" else "stages.relatedness.families_records"
            )
            product = "pairs" if key.unit == "pairs" else "families"
        else:
            module_name = {"sites": "stages.inputs.sites_records", "release": "stages.assembly.records"}.get(
                key.stage, f"stages.{key.stage}.records"
            )
            product = products[key.stage]
        module = importlib.import_module(f"haplotype_reconstruction.{module_name}")
        options: dict[str, Any] = {}
        if key.stage == "selection":
            options["stage"] = "local_selection_" + key.round
        elif key.stage == "assembly" and key.round:
            options["round_name"] = key.round.removeprefix("context.")
        elif key.stage == "painting":
            options.update(panels=False, emissions=False)
        elif key.stage == "refit":
            options["panels"] = False
        request = module.result_reader(
            results._store,
            results.manifest,
            product=product,
            contig=None if key.scope == "run" else key.scope,
            **options,
        )
        from ..contracts.roles import require_schema
        from ..store.protocol import read_items

        saved = (
            None
            if request is None
            else read_items(
                results._file(None if key.scope == "run" else key.scope), request.key, request.identity, ("schema",)
            )
        )
        if saved is None:
            raise SettingsError(f"workflow prerequisite {key.stage} for {key.scope} is not published")
        require_schema(saved.payload, request.schema)
        if key in values:
            raise InvariantError(f"duplicate workflow prerequisite: {key}")
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
        from ..stages.graph import Node, bind_tasks, build_graph, execution_options, workflow_cut

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
            reverse=tie_reverse,
            keep=tie_keep,
            census=None if tie_census is None else TieCensusSink(tie_census.expanduser().resolve()),
        )
        tasks = bind_tasks(
            graph, cut, _model_actions(prepared, resolved, cut, outputs=output_actions, ties=ties), completed=completed
        )
        scheduling = execution_options(graph, cut, phase_costs=_phase_costs(prepared, resolved))
        preload = _task_preloads(tasks)
    else:
        preload = ()
    # Init reads its source's header and binds the manifest; it runs kernels only when it simulates the source.
    # Warming them in init's own runtime first (an isolated process, before the run's warm-up on every CPU) delayed
    # every run's start by its compile pool.
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
            tasks, scheduling = _with_kernel_tasks(tasks, scheduling, kernels)
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


def _evidence_identity(store: CheckpointStore, resolved: ResolvedConfiguration, contig: str, *, reverse: bool) -> str:
    """Rebuild an evidence derivation from its source binding and initialized parameters."""
    from ..contracts.roles import require_schema
    from ..stages.evidence import records
    from ..store.protocol import bound_payload_identity, read_items

    file = store.contig(contig)
    if not file.has("input/payload"):
        raise SettingsError(f"{contig}: evidence source payload is not bound")
    consumed = bound_payload_identity(file)
    identity = records.identity(resolved.parameters.evidence, consumed, resolved.sample_ids, reverse=reverse)
    recorded = read_items(file, records.EVIDENCE, identity, ("schema",))
    if recorded is None:
        raise SettingsError(f"{contig}: evidence is not published")
    require_schema(recorded.payload, records.SCHEMA)
    return identity


def _run_relatedness(
    root: str,
    resolved: ResolvedConfiguration,
    maps: GeneticMapSet,
    *,
    pairs: bool = False,
    reverse: bool = False,
    team: int = 1,
) -> str:
    """Construct the bound-order evidence inputs after their graph dependencies finish."""
    from ..stages.relatedness.evidence_pass import EvidenceInput
    from ..store.store import CheckpointStore

    store = CheckpointStore(root, nthreads=team)
    inputs = tuple(
        EvidenceInput(c, _evidence_identity(store, resolved, c, reverse=reverse), maps.for_contig(c))
        for c in resolved.contigs
    )
    if pairs:
        from ..stages.relatedness.pairs_task import run as run_pairs

        return run_pairs(
            root, inputs, resolved.sample_ids, resolved.parameters.pairwise_ibd, reverse=reverse, team=team
        )
    from ..stages.relatedness.families_task import run as run_families

    return run_families(root, inputs, resolved.sample_ids, resolved.parameters.relatedness, reverse=reverse, team=team)


def _recorded_run(
    settings: Settings, directory: str | Path, *, recombination: bool = False
) -> tuple[PreparedRun, ResolvedConfiguration]:
    """Check settings against an initialized run without reopening its observation source."""

    from ..engine.runtime import configure_worker

    paths = RunPaths.from_directory(directory)
    settings = replace(settings, run=replace(settings.run, output=paths.output, checkpoints=paths.checkpoints))
    cores = resolve_cores(settings)
    configure_worker(cores)
    from ..stages.configure import configure, genetic_maps, resolve

    parameters = configure(settings)
    maps = genetic_maps(settings)
    store = _existing_store(paths)
    manifest = None if store is None else store.manifest("run")
    if manifest is None or store is None:
        raise SettingsError(f"{paths.output}: run has not been initialized; use haplotypes init")
    resolved = resolve(settings, sample_ids=manifest["sample_ids"], maps=maps, bound_seeds=manifest["realised_seeds"])
    from ..core.digest import jsonable
    from ..store.protocol import NumericalPolicy

    candidate = dict(manifest, **resolved.manifest(), numerical_policy=jsonable(NumericalPolicy.current()))
    _match_bound_configuration(candidate, manifest, store, resolved, allow_new_recombination=recombination)
    return PreparedRun(settings, paths, cores, None, parameters, maps, manifest), resolved


def _load_recorded_settings(directory: Path, cores: int | None = None) -> Settings:
    """Load the recorded versioned settings with an execution-only CPU override."""
    from .settings.load import load_settings

    options = {} if cores is None else {"execution": {"cores": cores}}
    return load_settings(directory / "config.toml", options=options)


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
    """Apply the numerical/runtime boundary before SI1 imports or computes template windows."""
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
    # Empty placement locks left by an earlier failed export into this destination are not contents (release 4.11).
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
            _export_identity(reader, contig, config),
            max_cores=prepared.cores,
            map_work=_map_writers,
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
        saved = read_items(file, request.key, request.identity, ("schema",))
        if saved is not None:
            require_schema(saved.payload, request.schema)

    # Release 4.10: each contig derives its upstream chain (recombination, phase, painting, assembly, selection,
    # discovery readers: envelope reads and decodes); contig after contig this took 20-30 s of a resumed run's start
    # and 165 s of an export of one contig. The checks are independent reads; the first error in contig order wins.
    _threaded_each(check, resolved.contigs)


def _threaded_each(function: Callable[[Any], Any], items: Sequence[Any]) -> list[Any]:
    """core.concurrency.threaded_each (release 4.10): bounded threads for a coordinator's independent record reads."""
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
    from ..stages.graph import Node, bind_tasks, build_graph, workflow_cut
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
    preload = _task_preloads(tasks)
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


# The stage implementations behind the application's task wrappers (a wrapper names its stage, not its code): preloaded
# for warm-up, and their import closures order the kernel compilations. Init only checks the run's bound configuration
# (_check_initialized): no stage code. Relatedness depends on the unit.
_STAGE_OWNERS: dict[str, tuple[str, ...]] = {
    "init": (),
    "evidence": ("stages.evidence.tasks",),
    "discovery": ("stages.discovery.tasks",),
    "selection": ("stages.selection.tasks",),
    "assembly": ("stages.assembly.final",),
    "painting": ("stages.painting.tasks",),
    "refit": ("stages.refit.tasks",),
    "pedigree": (
        "stages.pedigree.prepare",
        "stages.pedigree.screen",
        "stages.pedigree.scores",
        "stages.pedigree.decision",
    ),
    "phase": ("stages.phase.tasks",),
    "recombination": ("stages.recombination.tasks",),
    "sites": ("stages.inputs.sites",),
    "release": ("stages.assembly.release",),
    "export": ("outputs.export.task",),
    "report": ("outputs.reports.tasks",),
}


def _stage_modules(task: Task) -> tuple[str, ...] | None:
    """The stage implementation modules behind ``task`` (its stage's owners, with the kernel modules assembly imports
    lazily); None for a stage outside the table."""
    stage = task.key.stage
    if stage == "relatedness":
        modules: tuple[str, ...] = (
            "stages.relatedness." + ("pairs_task" if task.key.unit == "pairs" else "families_task"),
        )
    elif stage in _STAGE_OWNERS:
        modules = _STAGE_OWNERS[stage]
    else:
        return None
    result = tuple("haplotype_reconstruction." + module for module in modules)
    if stage == "assembly":
        from ..science.assembly.kernel_inventory import MODULES

        result += tuple(module.__name__ for module in MODULES)
    return result


def _task_preloads(tasks: Sequence[Task]) -> tuple[str, ...]:
    """Name the stage implementations behind application wrappers before C2 warms workers."""
    from ..stages.graph import preload_modules

    modules = list(preload_modules(tasks))
    for task in tasks:
        modules.extend(_stage_modules(task) or ())
    return tuple(dict.fromkeys(modules))


# Source identities also traverse readers, codecs and type-only imports into other stages. Those producers are
# not numerical work performed by this task. Bound first-use waits to the stage's scientific operations, plus
# shared/core/contract kernels in its source closure. General selection tasks can expand context assembly;
# application round tasks consume separate context nodes instead. Pedigree and phase can recompute painting
# emissions. Keep these real cross-stage consumers in their respective groups.
_STAGE_KERNEL_SCIENCE: dict[str, tuple[str, ...]] = {
    "evidence": ("evidence",),
    "relatedness": ("relatedness",),
    "discovery": ("discovery",),
    "selection": ("selection", "assembly"),
    "assembly": ("assembly",),
    "painting": ("painting",),
    "refit": ("refit",),
    "pedigree": ("pedigree", "painting"),
    "phase": ("phase", "painting"),
    "recombination": ("recombination",),
}


def _stage_kernel_modules(task: Task) -> tuple[str, ...]:
    """Numerical ownership within the stage's static closure, including dynamically expanded work."""
    from ..store.identity import code_dependency_identity

    roots = _stage_modules(task)
    if not roots:
        return ()
    modules = code_dependency_identity(roots).modules
    science = _STAGE_KERNEL_SCIENCE.get(task.key.stage)
    if science is None:  # output adapters retain their complete closure
        return modules
    if task.key.stage == "selection" and task.key.round in {"initial", "l1", "l2"}:
        science = ("selection", "assembly.evidence")
        if task.key.round != "initial":
            science += ("assembly.alleles",)
        else:
            modules = tuple(m for m in modules if m != "haplotype_reconstruction.science.selection.context_refit")
    if task.key.stage == "assembly" and task.key.round.startswith("context."):
        # CONTEXT returns after the join; only FINAL expands AS8 refinement.
        modules = tuple(m for m in modules if not m.startswith("haplotype_reconstruction.science.assembly.refinement."))
    prefixes = tuple(
        "haplotype_reconstruction." + name + "."
        for name in ("core", "contracts", "science.shared", *("science." + name for name in science))
    )
    return tuple(
        module
        for module in modules
        if (module + ".").startswith(prefixes) or module == "haplotype_reconstruction.science.evidence.likelihoods"
    )


def _with_kernel_tasks(
    tasks: tuple[Task, ...], scheduling: dict[str, Any], state: Any
) -> tuple[tuple[Task, ...], dict[str, Any]]:
    """Gate each scientific root on its own missing compiles, leaving unrelated work runnable.

    A root's dependency also protects its dynamic expansions: no child can use a tail signature before its module
    has been warmed in the shared run cache. Only state.missing emits tasks or edges. Compilation retains the
    earliest-consumer/learned-cost order; independent stages keep their existing priority over spare warm-up work.
    """
    from ..engine.kernels import compile_tasks

    compiles: dict[tuple[str, str], list[Task]] = {}
    for task in compile_tasks(state):  # one per missing overload, grouped by function
        compiles.setdefault(task.args[0], []).append(task)
    if not compiles:
        return tasks, scheduling
    by_module: dict[str, list[tuple[str, str]]] = {}
    for name in compiles:
        by_module.setdefault(name[0], []).append(name)
    indexed = {task.key: task for task in tasks}
    levels: dict[TaskKey, int] = {}

    def level(key: TaskKey) -> int:  # the longest dependency path to ``key`` within this graph
        if key not in levels:
            levels[key] = 1 + max((level(d) for d in indexed[key].depends_on if d in indexed), default=-1)
        return levels[key]

    first_use: dict[tuple[str, str], int] = {}
    gated = []
    for task in tasks:
        dependencies = list(task.depends_on)
        for module in _stage_kernel_modules(task):
            for name in by_module.get(module, ()):
                first_use[name] = min(first_use.get(name, level(task.key)), level(task.key))
                dependencies.extend(compile.key for compile in compiles[name])
        gated.append(replace(task, depends_on=tuple(dict.fromkeys(dependencies))))
    unused = 1 + max(levels.values(), default=0)
    order = sorted(
        compiles,
        key=lambda name: (
            first_use.get(name, unused),
            -state.seconds.get(name, 0.0),
            -len(state.missing[name]),
            name,
        ),
    )
    priorities = dict(scheduling.get("priorities") or {})
    floor = min(priorities.values(), default=0.0) - 1.0
    priorities.update({compile.key: floor - rank for rank, name in enumerate(order) for compile in compiles[name]})
    return (*gated, *(compile for name in order for compile in compiles[name])), dict(scheduling, priorities=priorities)


def _default_scratch(checkpoints: Path) -> Path:
    """Keep disposable files beside the immutable checkpoint layout on its filesystem."""
    return checkpoints.parent / ("." + checkpoints.name + ".scratch")


def _scratch_base(prepared: PreparedRun) -> Path:
    """Resolve the invocation's explicit scratch choice or checkpoint-adjacent default."""
    return prepared.settings.execution.scratch or _default_scratch(prepared.paths.checkpoints)


def _kernel_store(prepared: PreparedRun) -> Path:
    """Keep C2's disposable persistent cache on the invocation's selected scratch filesystem."""
    return _scratch_base(prepared) / "kernel_store"


def _start_runtime(prepared: PreparedRun, *, preload: tuple[str, ...] = ()) -> None:
    """Enter the engine with declared task roots and a code-keyed optional scratch namespace."""
    from ..engine.runtime import start_run

    # E1 forwards the registered declarations after importing these roots. Keep
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
