"""Bind simulation seeds at init, publish every contig, and expose observations.

OU30 orchestration owns files, resume and task declarations. Science lives in
simulation.source and receives explicit immutable inputs. Truth is published before
reads; reads mark a completed contig. A truth-only interruption keeps its existing
truth. Every contig read seed is drawn before any completed contig is skipped.

The mutable run manifest binds the realised seed before generation and changes to
complete only after all configured reads exist. Contig payloads use the checked,
immutable publication protocol. Identity includes resolved design, file contents,
realised seed and code closure, never the inference tie convention (ledger L8).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from ...contracts.roles import require_schema
from ...contracts.source import ContigReads
from ...core.digest import json_digest, jsonable
from ...core.errors import InputError, InvariantError, SettingsError
from ...core.random import realise_seed
from ...engine.graph import Expansion, RecordKey, ResourceRequest, ResultRef, Task, TaskKey, run_graph
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
from ...simulation.design import PedigreeRow, SimulationConfig, parse_pedigree_table
from ...simulation.parameters import READS_BYTES_PER_CELL, READS_WORKER_BYTES
from ...simulation.reads import ReadInputs, read_rows
from ...simulation.source import (
    ContigSimulation,
    FounderTemplate,
    contig_read_inputs,
    contig_result,
    prepare_simulation,
)
from ...simulation.truth import (
    COMPLETE,
    IN_PROGRESS,
    MANIFEST,
    READS,
    SCHEMA,
    TRUTH,
    SimulationManifest,
    truth_record,
)
from ...store.identity import code_dependency_identity
from ...store.protocol import claim, is_complete, publish, read_items, resume
from ...store.store import CheckpointStore
from .genetic_map import load_genetic_maps
from .simulation_records import load_manifest, load_reads, load_truth

__all__ = ["SimulationSource", "read_pedigree_table", "load_manifest", "load_truth", "load_reads", "reads_tasks"]


def read_pedigree_table(path: str | Path) -> tuple[PedigreeRow, ...]:
    """Read and validate the explicit table; sex/generation remain truth descriptions."""
    path = Path(path)
    return parse_pedigree_table(path.read_text(), label=str(path))


def _read_template(config: SimulationConfig, contig: str) -> FounderTemplate:
    with np.load(config.templates / f"{contig}.npz", allow_pickle=False) as data:
        return FounderTemplate(data["positions"], data["allele_probabilities"])


def _child_batches(function: Callable[..., Any], batches: Sequence[Any], *, cores: int) -> list[Any]:
    tasks = [Task(TaskKey("simulation.meiosis", str(i)), function, args=(batch,)) for i, batch in enumerate(batches)]
    results = run_graph(tasks, cores=cores, in_process=cores == 1, preload=(__name__,))
    return [results[task.key] for task in tasks]


# Blocks of samples per CPU of a contig's parallel read simulation (execution grain only: every sample draws from
# its own stream, so the counts never depend on it).
READ_BLOCKS_PER_CORE = 2


def _contig_reads(root: str, task: ContigSimulation, identity: str, cores: int) -> str | Expansion:
    """Simulate one contig's reads as blocks of samples on all CPUs, then publish TRUTH and READS (release 4.8).

    The haplotypes and shared factors are shared read-only once (ENG-24 transport); each block returns its counts;
    the gather on this contig's affinity writes the records with the run's CPUs compressing them."""
    file = CheckpointStore(root, nthreads=1).contig(task.contig)
    if is_complete(file, {READS: identity}):
        return task.contig
    inputs = contig_read_inputs(task)
    samples, sites = inputs.hap0.shape
    blocks = max(1, min(samples, READ_BLOCKS_PER_CORE * int(cores)))
    edges = [round(samples * b / blocks) for b in range(blocks + 1)]
    shared = share_value(inputs, "simulation-reads")
    width = len(str(blocks - 1))
    subtasks = tuple(
        Task(
            TaskKey("simulation", f"reads.{b:0{width}d}", scope=task.contig),
            _read_block,
            args=(shared, edges[b], edges[b + 1]),
            resources=ResourceRequest(working_set_bytes=READS_BYTES_PER_CELL * (edges[b + 1] - edges[b]) * sites),
        )
        for b in range(blocks)
        if edges[b + 1] > edges[b]
    )
    retained = retain((task, inputs, shared), "simulation-reads")
    gather = Task(
        TaskKey("simulation", "reads.gather", scope=task.contig),
        _publish_contig,
        args=(root, retained, identity, tuple(ResultRef(t.key) for t in subtasks)),
        depends_on=tuple(t.key for t in subtasks),
        affinity=task.contig,
        resources=ResourceRequest(1, max(1, int(cores)), READS_BYTES_PER_CELL * samples * sites),
        thread_argument="threads",
    )
    return Expansion(subtasks, gather)


def _rows_int32(inputs: ReadInputs, first: int, stop: int) -> np.ndarray:
    return read_rows(inputs, first, stop).astype(np.int32)


def _read_block(shared: SharedValue, first: int, stop: int) -> np.ndarray:
    """Engine subtask: counts int32[stop - first, L, 2] of one block of samples on the shared inputs."""
    return call_shared(shared, _rows_int32, first, stop)


def _publish_contig(
    root: str, reference: ResidentRef, identity: str, blocks: tuple[np.ndarray, ...], threads: int = 1
) -> str:
    """Gather: join the blocks in sample order and publish TRUTH then READS under the READS claim."""
    task, inputs, shared = resident_value(reference)
    try:
        release_shared_value(shared)
        file = CheckpointStore(root, nthreads=max(1, int(threads))).contig(task.contig)
        with claim(file, READS, identity) as needed:
            if not needed:
                return task.contig
            reads, truth = contig_result(task, inputs, np.concatenate(blocks, axis=0))
            del blocks
            if not file.has(TRUTH):
                publish(file, TRUTH, identity, truth_record(truth))
            else:
                resume(file, TRUTH, identity)
            publish(
                file,
                READS,
                identity,
                dict(schema=SCHEMA, sample_ids=reads.sample_ids, positions=reads.positions, counts=reads.counts),
            )
    finally:
        release_resident(reference)
    return task.contig


def reads_tasks(
    root: str, contigs: Sequence[ContigSimulation], identities: dict[str, str], *, cores: int = 1
) -> tuple[Task, ...]:
    """Declare whole-contig tasks largest first, each expanding into blocks of samples on ``cores`` CPUs;
    estimates report memory, never gate science."""
    order = sorted(range(len(contigs)), key=lambda i: (-len(contigs[i].sample_ids) * len(contigs[i].positions), i))
    return tuple(
        Task(
            TaskKey("simulation", "reads", scope=contigs[i].contig),
            _contig_reads,
            args=(root, contigs[i], identities[contigs[i].contig], cores),
            outputs=(RecordKey(contigs[i].contig, TRUTH), RecordKey(contigs[i].contig, READS)),
            resources=ResourceRequest(working_set_bytes=READS_WORKER_BYTES),
            affinity=contigs[i].contig,
        )
        for i in order
    )


class SimulationSource:
    """A materialised simulation observation source with truth inaccessible to inference."""

    kind = "simulation"

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self._store: CheckpointStore | None = None
        self._manifest: SimulationManifest | None = None
        self._seed: int | None = config.seed
        self._identity: dict[str, Any] | None = None

    def contigs(self) -> tuple[str, ...]:
        """Return the configured axis, whose order determines the random streams."""
        return self.config.contigs

    def identity(self) -> dict[str, Any]:
        """Return content identity with the seed realised once by materialise/init."""
        if self._seed is None:
            raise SettingsError("init must bind the simulation seed before requesting its identity")
        if self._identity is not None:
            return self._identity
        config = jsonable(self.config)
        config.pop("templates")
        config.pop("generating_map")
        config["design"].pop("pedigree", None)
        maps = load_genetic_maps(self.config.generating_map, self.config.generating_rate_cm_per_mb)
        templates = {
            c: hashlib.sha256((self.config.templates / f"{c}.npz").read_bytes()).hexdigest()
            for c in self.config.contigs
        }
        upstream = dict(templates=json_digest(templates), generating_map=json_digest(maps.identity()))
        if self.config.design.pedigree is not None:
            config.pop("cohorts")
            upstream["pedigree"] = hashlib.sha256(self.config.design.pedigree.read_bytes()).hexdigest()
        code = code_dependency_identity(
            ("haplotype_reconstruction.stages.inputs.simulation", "haplotype_reconstruction.simulation.source")
        )
        self._identity = dict(
            stage=self.kind, schema=SCHEMA, config=config, upstream=upstream, seed=self._seed, code=code.digest
        )
        return self._identity

    def payload_identity(self, contig: str) -> str:
        """Bind a contig's observation payload to design, realised seed and closure."""
        if contig not in self.config.contigs:
            raise InputError(f"contig not in the simulation: {contig}")
        return json_digest(dict(source=self.identity(), contig=contig))

    def is_materialised(self, store: CheckpointStore) -> bool:
        """Return whether every configured contig has a completed simulation."""
        manifest = load_manifest(store)
        return (
            manifest is not None
            and manifest.state == COMPLETE
            and all(store.contig(c).has(READS) for c in self.contigs())
        )

    def sample_ids(self) -> tuple[str, ...]:
        """Return the observed pedigree order; explicit tables permit this before init."""
        if self._manifest is not None and self._manifest.state == COMPLETE:
            return self._manifest.sample_ids
        if self.config.design.pedigree is not None:
            return tuple(row.sample for row in read_pedigree_table(self.config.design.pedigree) if row.observed)
        raise SettingsError("the simulation is not materialised yet")

    def reads(self, contig: str, budget: Any = None) -> ContigReads:
        """Return only stored REF/ALT counts int32[N,L,2], never simulated truth."""
        if self._store is None:
            raise SettingsError("materialise the simulation before reading it")
        payload = load_reads(self._store.contig(contig), self.payload_identity(contig))
        return ContigReads(contig, tuple(payload["sample_ids"]), payload["positions"], payload["counts"])

    def site_alleles(self, contig: str, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, None]:
        """Return A/C coding after validating stored source positions, without loading counts."""
        if self._store is None:
            raise SettingsError("materialise the simulation before reading it")
        record = read_items(self._store.contig(contig), READS, self.payload_identity(contig), ("schema", "positions"))
        if record is None:
            raise InputError(f"{contig}: no simulated reads")
        require_schema(record.payload, SCHEMA)
        available = record.payload["positions"]
        columns = np.searchsorted(available, positions)
        if np.any(columns >= len(available)) or not np.array_equal(available[columns], positions):
            raise InputError(f"{contig}: requested export positions are not simulated sites")
        return np.full(len(positions), "A"), np.full(len(positions), "C"), None

    def allele_totals(self, contig: str, positions: np.ndarray, samples: Sequence[str]) -> tuple[np.ndarray, ...]:
        """Sum stored read counts by requested names/sites with int64 accumulators."""
        reads = self.reads(contig)
        sites = np.searchsorted(reads.positions, positions)
        if np.any(sites >= len(reads.positions)) or not np.array_equal(reads.positions[sites], positions):
            raise InvariantError(f"{contig}: requested export positions are not simulated sites")
        by_name = {name: i for i, name in enumerate(reads.sample_ids)}
        rows = np.asarray([by_name[name] for name in samples], dtype=np.int64)
        totals = reads.counts[rows][:, sites].sum(axis=0, dtype=np.int64)
        return totals[:, 0], totals[:, 1], np.ones(len(sites), dtype=np.int8)

    def materialise(
        self, store: CheckpointStore, contigs: Sequence[str] | None = None, budget: Any = 1, *, cores: int | None = None
    ) -> SimulationManifest:
        """Bind the seed and publish all configured contigs, resuming interrupted writes."""
        selected = self.contigs() if contigs is None else tuple(contigs)
        if set(selected) - set(self.contigs()):
            raise InputError(f"contigs not in the simulation: {sorted(set(selected) - set(self.contigs()))}")
        self._store = store
        team = int(cores if cores is not None else budget.threads() if hasattr(budget, "threads") else budget)
        manifest = load_manifest(store)
        if manifest is not None:
            if self._seed != manifest.seed:
                self._identity = None
            self._seed = manifest.seed
            if manifest.identity != self.identity():
                raise SettingsError(
                    f"{store.root}: recorded simulation was made for another configuration; use a new output root"
                )
            if manifest.state == COMPLETE:
                for c in self.contigs():
                    load_reads(store.contig(c), self.payload_identity(c))
                self._manifest = manifest
                return manifest
        else:
            partial_contigs = [c for c in self.contigs() if store.contig(c).has(READS)]
            if partial_contigs:
                raise SettingsError(
                    f"{store.root}: simulated contigs without a simulation manifest "
                    f"({partial_contigs}); use a new output root"
                )
            self._seed = realise_seed(self.config.seed)
            self._identity = None
            manifest = SimulationManifest(IN_PROGRESS, self.config.seed, self._seed, self.contigs(), self.identity())
            store.run.put(MANIFEST, manifest.record())
        templates = tuple(_read_template(self.config, c) for c in self.contigs())
        maps = load_genetic_maps(self.config.generating_map, self.config.generating_rate_cm_per_mb)
        rows = None if self.config.design.pedigree is None else read_pedigree_table(self.config.design.pedigree)
        plan = prepare_simulation(
            self.config,
            self._seed,
            templates,
            tuple(maps.for_contig(c) for c in self.contigs()),
            pedigree_rows=rows,
            cores=team,
            runner=partial(_child_batches, cores=team),
        )
        identities = {c: self.payload_identity(c) for c in self.contigs()}
        pending = [task for task in plan.contigs if not store.contig(task.contig).has(READS)]
        run_graph(
            reads_tasks(str(store.root), pending, identities, cores=team),
            cores=team,
            in_process=team == 1,
            preload=(__name__,),
        )
        for c in self.contigs():
            load_reads(store.contig(c), identities[c])
        manifest = SimulationManifest(
            COMPLETE,
            self.config.seed,
            self._seed,
            self.contigs(),
            self.identity(),
            plan.observed.samples,
            plan.observed,
            plan.generated,
        )
        store.run.put(MANIFEST, manifest.record())
        self._manifest = manifest
        return manifest
