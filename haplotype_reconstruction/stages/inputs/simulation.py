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
from ...engine.graph import RecordKey, ResourceRequest, Task, TaskKey, run_graph
from ...simulation.design import PedigreeRow, SimulationConfig, parse_pedigree_table
from ...simulation.parameters import READS_BYTES_PER_CELL, READS_WORKER_BYTES
from ...simulation.source import ContigSimulation, FounderTemplate, generate_contig, prepare_simulation
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
from ...store.protocol import claim, publish, read_items, resume
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


def _write_contig(root: str, task: ContigSimulation, identity: str) -> str:
    store = CheckpointStore(root, nthreads=1)
    file = store.contig(task.contig)
    with claim(file, READS, identity) as needed:
        if not needed:
            return task.contig
        reads, truth = generate_contig(task)
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
    return task.contig


def reads_tasks(root: str, contigs: Sequence[ContigSimulation], identities: dict[str, str]) -> tuple[Task, ...]:
    """Declare whole-contig tasks largest first; estimates report memory, never gate science."""
    order = sorted(range(len(contigs)), key=lambda i: (-len(contigs[i].sample_ids) * len(contigs[i].positions), i))
    return tuple(
        Task(
            TaskKey("simulation", "reads", scope=contigs[i].contig),
            _write_contig,
            args=(root, contigs[i], identities[contigs[i].contig]),
            outputs=(RecordKey(contigs[i].contig, TRUTH), RecordKey(contigs[i].contig, READS)),
            resources=ResourceRequest(
                working_set_bytes=READS_BYTES_PER_CELL * len(contigs[i].sample_ids) * len(contigs[i].positions)
                + READS_WORKER_BYTES
            ),
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
            reads_tasks(str(store.root), pending, identities), cores=team, in_process=team == 1, preload=(__name__,)
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
