"""Bind simulation seeds at init, publish every contig, and expose observations.

OU30 orchestration owns files, resume and task declarations. Science lives in
simulation.source and receives explicit immutable inputs. Truth is published before
reads; reads mark a completed contig. A truth-only interruption keeps its existing
truth. Every contig read seed is drawn before any completed contig is skipped.

The mutable run manifest binds the realised seed before generation and changes to
complete only after all configured reads exist. Contig payloads use the checked,
immutable publication protocol. Identity includes resolved design, file contents,
realised seed and code closure, never the inference tie convention.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ...contracts.genetic_map import ChromosomeGeneticMap
from ...contracts.source import ContigReads
from ...core.concurrency import threaded_each
from ...core.digest import json_digest, jsonable
from ...core.errors import InputError, InvariantError, SettingsError
from ...core.random import realise_seed
from ...engine.graph import Expansion, RecordKey, ResourceRequest, ResultRef, Task, TaskKey
from ...engine.scheduler import TaskSession
from ...engine.workers import (
    ResidentRef,
    SharedArrayRef,
    SharedValue,
    attach_shared_array,
    call_shared,
    close_shared_memory,
    release_output,
    release_resident,
    release_shared_value,
    resident_value,
    retain,
    share_output,
    share_value,
)
from ...simulation.design import PedigreeRow, SimulationConfig, parse_pedigree_table
from ...simulation.parameters import (
    READ_BLOCKS_PER_CORE,
    READS_BYTES_PER_CELL,
    READS_WORKER_BYTES,
    SOURCE_BYTES_PER_SITE,
)
from ...simulation.reads import ReadInputs, read_rows
from ...simulation.source import (
    ContigSimulation,
    FounderTemplate,
    contig_read_inputs,
    contig_result,
    prepare_contig,
    prepare_stocks,
    streamed_cross,
)
from ...simulation.stocks import StockRecord
from ...simulation.truth import (
    COMPLETE,
    IN_PROGRESS,
    MANIFEST,
    READS,
    SCHEMA,
    TRUTH,
    SimulationManifest,
    require_simulation_schema,
    truth_record,
)
from ...store.identity import code_dependency_identity
from ...store.protocol import claim, is_complete, publish, read_items, resume
from ...store.store import CheckpointStore
from .genetic_map import load_genetic_maps
from .simulation_records import check_reads, load_manifest, load_reads, load_truth

__all__ = ["SimulationSource", "read_pedigree_table", "load_manifest", "load_truth", "load_reads", "reads_tasks"]


def read_pedigree_table(path: str | Path) -> tuple[PedigreeRow, ...]:
    """Read and validate the explicit table; sex/generation remain truth descriptions."""
    path = Path(path)
    return parse_pedigree_table(path.read_text(), label=str(path))


def _read_template(config: SimulationConfig, contig: str) -> FounderTemplate:
    assert config.templates is not None and config.source is not None
    with np.load(config.templates / f"{contig}.npz", allow_pickle=False) as data:
        positions = data["positions"]
        values = {name: data[name] for name in ("alleles", "allele_probabilities") if name in data}
    metadata = config.templates / "haplotypes.tsv"
    if not metadata.exists():
        if config.stocks:
            names = tuple(dict.fromkeys(stock.population for stock in config.stocks))
            populations = np.concatenate([
                np.full(2 * stock.founders, names.index(stock.population), np.int32) for stock in config.stocks
            ])
            return FounderTemplate(positions, unknown=config.source.unknown, population=populations,
                                   population_names=names, **values)
        return FounderTemplate(positions, unknown=config.source.unknown, **values)
    with metadata.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not {"haplotype", "individual", "population"}.issubset(reader.fieldnames or ()):
            raise InputError("haplotypes.tsv requires haplotype, individual and population columns")
        rows = list(reader)
    count = len(next(iter(values.values()))) if values else 0
    if len(rows) != count:
        raise InputError("haplotypes.tsv must have one row per source haplotype")
    if any(rows[h]["individual"] != rows[h + 1]["individual"] for h in range(0, len(rows) - 1, 2)):
        raise InputError("haplotypes.tsv consecutive haplotypes must name the same individual")
    names = tuple(dict.fromkeys(row["population"] for row in rows))
    populations = np.asarray([names.index(row["population"]) for row in rows], np.int32)
    return FounderTemplate(
        positions, unknown=config.source.unknown, population=populations, population_names=names, **values
    )


def _preflight(
    config: SimulationConfig,
) -> tuple[tuple[PedigreeRow, ...] | None, tuple[ChromosomeGeneticMap, ...], int, tuple[str, ...]]:
    """Read every structural input before seed realisation or any science draw.

    Panels are checked one at a time and released. They are loaded again only when
    their contig is generated, bounding resident source arrays by one contig.
    """
    assert config.source is not None
    rows = None if config.design.pedigree is None else read_pedigree_table(config.design.pedigree)
    if rows is not None:
        config.founder_stocks(rows)
    maps = load_genetic_maps(config.generating_map, config.generating_rate_cm_per_mb)
    if config.source.kind == "file":
        count = None
        populations = None
        for contig in config.contigs:
            template = _read_template(config, contig)
            assert template.population is not None
            if count is not None and count != len(template.population):
                raise InputError("simulation contigs have different founder counts")
            if populations is not None and populations != template.population_names:
                raise InputError("file source population names must agree across contigs")
            count, populations = len(template.population), template.population_names
            if config.stocks and count != 2 * sum(s.founders for s in config.stocks):
                raise InputError("file source rows must match the stock generation-zero founder layout")
            first = 0
            for stock in config.stocks:
                stop = first + 2 * stock.founders
                labels = {template.population_names[int(i)] for i in template.population[first:stop]}
                if labels != {stock.population}:
                    raise InputError(f"stock {stock.name}: file source rows disagree with its population")
                first = stop
            del template
        assert count is not None and populations is not None
        if not config.stocks and rows is not None and sum(not r.parent1 for r in rows) != count // 2:
            raise InputError("the pedigree founder count differs from the file source")
        if not config.stocks and rows is None and count < 4:
            raise InputError("the cross requires at least two founder individuals")
    else:
        count = 2 * sum(s.founders for s in config.stocks)
        populations = tuple(p.name for p in config.source.populations)
    return rows, tuple(maps.for_contig(c) for c in config.contigs), count // 2, populations


def _check_completed_contig(task: tuple[CheckpointStore, str, str]) -> None:
    """Module-scope callback for independent completion-envelope reads."""
    store, contig, identity = task
    check_reads(store.contig(contig), identity)


def _generate_contig(
    root: str,
    config: SimulationConfig,
    seed: int,
    index: int,
    maps: Sequence[ChromosomeGeneticMap],
    rows: tuple[PedigreeRow, ...] | None,
    records: tuple[StockRecord, ...],
    order: tuple[str, ...],
    identity: str,
) -> ContigSimulation | None:
    """One independent source/stock/site/cross task, before read-buffer allocation.

    No resident affinity is needed here. The engine transports the immutable panel
    to the dependent read task, whose lane owns the larger read/truth buffers.
    """
    assert config.source is not None
    contig = config.contigs[index]
    if is_complete(CheckpointStore(root).contig(contig), {READS: identity}):
        return None
    template = _read_template(config, contig) if config.source.kind == "file" else None
    return prepare_contig(config, seed, index, maps, rows, records, order, template)


def _generated_reads(
    root: str, task: ContigSimulation | None, contig: str, identity: str, cores: int
) -> str | Expansion:
    """Continue a prepared contig on an engine lane, or reuse a completed publication."""
    if task is None:
        check_reads(CheckpointStore(root).contig(contig), identity)
        return contig
    return _contig_reads(root, task, identity, cores)


def _child_batches(function: Callable[..., Any], batches: Sequence[Any], *, session: TaskSession) -> list[Any]:
    tasks = [Task(TaskKey("simulation.meiosis", str(i)), function, args=(batch,)) for i, batch in enumerate(batches)]
    results = session.run(tasks)
    return [results[task.key] for task in tasks]


def _generation_bytes(config: SimulationConfig, contig: str, samples: int, sources: int) -> int:
    """Report a contig's panel, read/truth arrays and fixed traversal allowance.

    The per-Mb mode uses its expected site count, not a scientific cap. The engine
    governs admission from measured process-tree pressure; estimates never reserve
    memory or limit the site draw. File positions are read without loading alleles.
    """
    assert config.source is not None
    if config.source.kind == "file":
        assert config.templates is not None
        with np.load(config.templates / f"{contig}.npz", allow_pickle=False) as panel:
            sites = len(panel["positions"])
    else:
        assert config.sites is not None
        mode = config.sites.mode(config.contig_length(contig))
        if mode.count is not None:
            sites = mode.count
        elif mode.positions is not None:
            sites = len(mode.positions)
        else:
            sites = int(np.ceil(mode.per_mb * config.contig_length(contig) / 1e6))
    return READS_WORKER_BYTES + sites * (READS_BYTES_PER_CELL * samples + SOURCE_BYTES_PER_SITE * (sources + 1))


def _contig_reads(root: str, task: ContigSimulation, identity: str, cores: int) -> str | Expansion:
    """Simulate one contig's reads as sample blocks under engine CPU leases, then publish TRUTH and READS.

    The haplotypes and shared factors are shared read-only once; each block writes its samples'
    counts into one shared int32 [N, L, 2] array owned by this contig's affinity (engine.workers.share_output).
    The gather publishes directly from that array with its leased CPUs compressing the records, avoiding
    coordinator copies of the counts. It then releases the array; failure releases it with the other affinity values.
    """
    file = CheckpointStore(root, nthreads=1).contig(task.contig)
    if is_complete(file, {READS: identity}):
        return task.contig
    inputs = contig_read_inputs(task)
    samples, sites = inputs.hap0.shape
    if not sites:
        # A positive per-Mb target can realise no clocks below its threshold.
        # There are no cells to sample; preserve the empty observation/truth axes.
        with claim(file, READS, identity) as needed:
            if needed:
                reads, truth = contig_result(task, inputs, np.zeros((samples, 0, 2), np.int32))
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
    blocks = max(1, min(samples, READ_BLOCKS_PER_CORE * int(cores)))
    edges = [round(samples * b / blocks) for b in range(blocks + 1)]
    shared = share_value(inputs, "simulation-reads")
    counts_ref, counts = share_output((samples, sites, 2), np.int32, "simulation-read-counts")
    width = len(str(blocks - 1))
    subtasks = tuple(
        Task(
            TaskKey("simulation", f"reads.{b:0{width}d}", scope=task.contig),
            _read_block,
            args=(shared, counts_ref, edges[b], edges[b + 1]),
            resources=ResourceRequest(working_set_bytes=READS_BYTES_PER_CELL * (edges[b + 1] - edges[b]) * sites),
        )
        for b in range(blocks)
        if edges[b + 1] > edges[b]
    )
    retained = retain((task, inputs, shared, counts), "simulation-reads")
    gather = Task(
        TaskKey("simulation", "reads.gather", scope=task.contig),
        _publish_contig,
        args=(root, retained, identity),
        depends_on=tuple(t.key for t in subtasks),
        affinity=task.contig,
        resources=ResourceRequest(1, max(1, int(cores)), READS_BYTES_PER_CELL * samples * sites),
        thread_argument="threads",
    )
    return Expansion(subtasks, gather)


def _rows_into(inputs: ReadInputs, counts_ref: SharedArrayRef, first: int, stop: int) -> int:
    handle, counts = attach_shared_array(counts_ref)
    try:
        read_rows(inputs, first, stop, out=counts[first:stop])
    finally:
        del counts
        close_shared_memory((handle,))
    return stop - first


def _read_block(shared: SharedValue, counts_ref: SharedArrayRef, first: int, stop: int) -> int:
    """Engine subtask: the counts of one block of samples on the shared inputs, written into rows first..stop-1 of
    the contig's shared int32 [N, L, 2] segment (each block owns its rows); returns its sample count."""
    return call_shared(shared, _rows_into, counts_ref, first, stop)


def _publish_contig(root: str, reference: ResidentRef, identity: str, threads: int = 1) -> str:
    """Gather: publish TRUTH then READS (the blocks' shared counts, in sample order) under the READS claim."""
    task, inputs, shared, counts = resident_value(reference)
    try:
        release_shared_value(shared)
        file = CheckpointStore(root, nthreads=max(1, int(threads))).contig(task.contig)
        with claim(file, READS, identity) as needed:
            if not needed:
                return task.contig
            reads, truth = contig_result(task, inputs, counts)
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
            del reads, truth
    finally:
        del counts
        release_resident(reference)
        release_output("simulation-read-counts")
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
        self._preflight = _preflight(config)
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
        config["source"].pop("file", None)
        upstream = dict(generating_map=json_digest(maps.identity()))
        assert self.config.source is not None
        if self.config.source.kind == "file":
            assert self.config.templates is not None
            templates = {
                c: hashlib.sha256((self.config.templates / f"{c}.npz").read_bytes()).hexdigest()
                for c in self.config.contigs
            }
            upstream["templates"] = json_digest(templates)
            metadata = self.config.templates / "haplotypes.tsv"
            if metadata.exists():
                upstream["source_metadata"] = hashlib.sha256(metadata.read_bytes()).hexdigest()
        if self.config.sites is not None and self.config.sites.positions is not None:
            config["sites"].pop("positions", None)
            upstream["positions"] = hashlib.sha256(self.config.sites.positions.read_bytes()).hexdigest()
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
        require_simulation_schema(record.payload)
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
                # Check every contig's read envelope without decoding counts; the evidence stage
                # performs the full count decode and validation.
                threaded_each(_check_completed_contig, [(store, c, self.payload_identity(c)) for c in self.contigs()])
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
        rows, maps, file_founders, populations = self._preflight
        records, order = prepare_stocks(self.config, self._seed, rows) if self.config.stocks else ((), ())
        founder_count = len(order) if records else file_founders
        generated, observed, _, _ = streamed_cross(self.config, self._seed, maps, rows, founder_count)
        identities = {c: self.payload_identity(c) for c in self.contigs()}
        # Fix stock pedigrees, taken individuals and sample order before fan-out.
        # Preparation has no resident affinity: independent single-core tasks can
        # use every available worker. The larger read expansions use engine lanes,
        # retaining only their own contig until truth and reads have been published.
        tasks = []
        for index, contig in enumerate(self.contigs()):
            if is_complete(store.contig(contig), {READS: identities[contig]}):
                continue
            generation = TaskKey("simulation", "generate", scope=contig)
            memory = _generation_bytes(self.config, contig, len(observed.samples), 2 * file_founders)
            preparation_memory = _generation_bytes(self.config, contig, 0, 2 * file_founders)
            tasks.extend((
                Task(
                    generation,
                    _generate_contig,
                    args=(
                        str(store.root), self.config, self._seed, index, maps, rows, records, order, identities[contig]
                    ),
                    resources=ResourceRequest(working_set_bytes=preparation_memory),
                ),
                Task(
                    TaskKey("simulation", "reads", scope=contig),
                    _generated_reads,
                    args=(str(store.root), ResultRef(generation), contig, identities[contig], team),
                    depends_on=(generation,),
                    outputs=(RecordKey(contig, TRUTH), RecordKey(contig, READS)),
                    resources=ResourceRequest(working_set_bytes=memory),
                    affinity=contig,
                ),
            ))
        with TaskSession(team, in_process=team == 1, preload=(__name__,)) as session:
            session.run(tasks)
        threaded_each(_check_completed_contig, [(store, c, identities[c]) for c in self.contigs()])
        manifest = SimulationManifest(
            COMPLETE,
            self.config.seed,
            self._seed,
            self.contigs(),
            self.identity(),
            observed.samples,
            observed,
            generated,
            stocks=tuple(r.record() for r in records),
            populations=populations,
        )
        store.run.put(MANIFEST, manifest.record())
        self._manifest = manifest
        return manifest
