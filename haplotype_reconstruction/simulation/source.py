"""The simulation as an observation source: templates -> founders -> pedigree -> reads + truth.

``SimulationSource`` implements the driver's ObservationSource protocol (``kind``, ``contigs()``,
``identity()``, ``materialise(store, contigs, budget)``, ``sample_ids()``, ``reads(contig, budget)``).
``materialise`` simulates every configured contig (they share one pedigree) into the store and is
resumable: the realised seed is bound first (manifest ``in_progress``), contigs whose reads are
recorded are not written again, and the manifest is completed last. For a configuration and
seed the data are identical to base (92b3daf) and do not depend on the CPU budget.
"""
from __future__ import annotations

import hashlib
import os
import resource
import time

import numpy as np

from .. import execution
from ..execution.budget import as_budget
from ..model.evidence import ContigReads
from ..model.genetic_map import load_genetic_maps
from ..store.identity import json_digest, jsonable, stage_identity
from . import truth as records
from .design import SimulationConfig, observed_indices
from .pedigree import materialise_founders, pair_founders, simulate_pedigree
from .reads import sample_reads

KIND = "simulation"
# Seed offsets of the independent random streams (base values; see pedigree.py).
_FOUNDER_SEED_KEY = 2_000_000
_READ_SEED_OFFSET = 1_000_000
# Peak bytes per (individual, site) of one contig's reads worker (int8 haplotype copies, int64 depth,
# ALT, REF and stacked read tensors; measured 48-50 at N = 320), plus 1 GiB per worker.
_READS_BYTES_PER_INDIVIDUAL_SITE = 48
_READS_WORKER_BYTES = 1 << 30


def _template_path(config: SimulationConfig, contig: str):
    return config.templates / f"{contig}.npz"


def _load_template(config: SimulationConfig, contig: str):
    with np.load(_template_path(config, contig), allow_pickle=False) as template:
        positions = np.asarray(template["positions"], dtype=np.int64)
        probabilities = template["allele_probabilities"]
    if probabilities.ndim != 3 or probabilities.shape[1:] != (len(positions), 2):
        raise ValueError(f"{contig}: founder template axes must be (haplotypes, sites, 2)")
    if len(positions) == 0 or np.any(np.diff(positions) <= 0):
        raise ValueError(f"{contig}: founder template positions must be non-empty and increasing")
    return positions, probabilities


def _reads_worker(task, shared, budget):
    """Pool task: simulate one contig's reads, convert its truth, record truth then reads."""
    started = time.monotonic()
    rng = np.random.default_rng(task["seed"])
    offspring = task["offspring"]
    hap0 = np.array([individual[0] for individual in offspring])
    hap1 = np.array([individual[1] for individual in offspring])
    counts = sample_reads(hap0, hap1, task["positions"], task["depth"], task["read_model"], rng)
    truth = records.ContigTruth(
        sample_ids=task["sample_ids"], positions=task["positions"],
        founder_haplotypes=np.asarray([haplotype for founder in task["founders"] for haplotype in founder],
                                      dtype=np.int8),
        alleles=np.ascontiguousarray(np.stack((hap0, hap1), axis=2), dtype=np.int8),
        painting=records.truth_painting(task["ancestry"]), crossovers=task["crossovers"])
    del hap0, hap1, offspring
    simulated = time.monotonic()
    from ..store import CheckpointStore
    threads = budget.threads()
    file = CheckpointStore(task["root"], nthreads=threads).contig(task["contig"])
    if not file.has(records.TRUTH):
        records.save_truth(file, truth)
    records.save_reads(file, task["sample_ids"], task["positions"], counts)
    return dict(contig=task["contig"], simulate_seconds=simulated - started,
                save_seconds=time.monotonic() - simulated, threads=threads,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


class SimulationSource:
    """A known-pedigree simulation from founder templates (see the module docstring)."""

    kind = KIND

    def __init__(self, config: SimulationConfig):
        self.config = config
        self._store = None
        self._identity = None
        self._manifest = None

    def contigs(self) -> tuple[str, ...]:
        return self.config.contigs

    def identity(self) -> dict:
        """Stage identity of the simulation: generating configuration (paths do not matter, contents do), the
        template and generating-map contents, and the simulation package code."""
        if self._identity is None:
            config = jsonable(self.config)
            config.pop("templates")
            config.pop("generating_map")
            maps = load_genetic_maps(self.config.generating_map, self.config.generating_rate_cm_per_mb)
            templates = {contig: hashlib.sha256(_template_path(self.config, contig).read_bytes()).hexdigest()
                         for contig in self.config.contigs}
            self._identity = stage_identity(
                KIND, schema=records.SCHEMA, config=config,
                upstream={"templates": json_digest(templates), "generating_map": json_digest(maps.identity())})
        return self._identity

    def is_materialised(self, store) -> bool:
        manifest = records.load_manifest(store)
        return manifest is not None and manifest.state == records.COMPLETE

    def sample_ids(self) -> tuple[str, ...]:
        if self._manifest is None or self._manifest.state != records.COMPLETE:
            raise RuntimeError("the simulation is not materialised yet")
        return self._manifest.sample_ids

    def reads(self, contig: str, budget) -> ContigReads:
        """The simulated allele depths of one contig (from its ``simulation/reads`` record)."""
        if self._store is None:
            raise RuntimeError("materialise the simulation before reading it")
        payload = records.load_reads(self._store.contig(contig), nthreads=as_budget(budget).threads())
        return ContigReads(contig=contig, sample_ids=payload["sample_ids"], positions=payload["positions"],
                           counts=payload["counts"])

    def materialise(self, store, contigs, budget) -> records.SimulationManifest:
        """Simulate every configured contig into ``store`` unless done (``contigs`` must be among them)."""
        unknown = sorted(set(contigs) - set(self.config.contigs))
        if unknown:
            raise ValueError(f"contigs not in the simulation: {unknown}")
        self._store = store
        budget = as_budget(budget)
        manifest = records.load_manifest(store)
        if manifest is not None:
            if manifest.identity != self.identity():
                raise RuntimeError(f"{store.root}: the recorded simulation was made for another configuration; "
                                   "use a new output root")
            if manifest.state == records.COMPLETE:
                store.require(records.READS, self.config.contigs)
                self._manifest = manifest
                return manifest
            seed = manifest.seed
            print(f"  [simulation] resuming with realised seed {seed}", flush=True)
        else:
            partial = [contig for contig in self.config.contigs if store.contig(contig).has(records.READS)]
            if partial:
                raise RuntimeError(f"{store.root}: simulated contigs without a simulation manifest "
                                   f"({partial}); use a new output root")
            seed = self.config.seed if self.config.seed is not None else int.from_bytes(os.urandom(8), "little")
            store.run.put(records.MANIFEST, records.SimulationManifest(
                state=records.IN_PROGRESS, requested_seed=self.config.seed, seed=seed,
                contigs=self.config.contigs, identity=self.identity()).record())
            print(f"  [simulation] realised seed {seed}", flush=True)
        manifest = self._simulate(store, seed, budget)
        store.run.put(records.MANIFEST, manifest.record())
        self._manifest = manifest
        return manifest

    def _simulate(self, store, seed: int, budget) -> records.SimulationManifest:
        config = self.config
        contigs = config.contigs
        started = time.monotonic()
        sites_by_contig, founders_by_contig = [], []
        founder_seeds = np.random.SeedSequence([seed, _FOUNDER_SEED_KEY]).spawn(len(contigs))
        for contig, founder_seed in zip(contigs, founder_seeds):
            positions, probabilities = _load_template(config, contig)
            sites_by_contig.append(positions)
            founders_by_contig.append(pair_founders(
                materialise_founders(probabilities, np.random.default_rng(founder_seed))))
            del probabilities
        maps = load_genetic_maps(config.generating_map, config.generating_rate_cm_per_mb)
        chromosome_maps = [maps.for_contig(contig) for contig in contigs]
        with execution.stage_timer("simulate_pedigree"):
            # Contigs the map lacks (or every contig without a map) use the set's one fallback rate.
            generated, simulated = simulate_pedigree(
                founders_by_contig, sites_by_contig, config.cohorts, seed=seed,
                recombination_rate=chromosome_maps[0].fallback_rate_per_bp,
                mutation_rate=config.mutation_rate_per_bp,
                chromosome_maps=[m if m.has_map else None for m in chromosome_maps],
                design=config.design, budget=budget)
        print(f"  [simulation] pedigree of {len(generated.samples)} individuals on {len(contigs)} contigs "
              f"in {time.monotonic() - started:.1f}s", flush=True)
        observed_rows = observed_indices(generated.cohorts, config.design, seed)
        observed = generated.take(observed_rows)
        sample_ids = observed.samples
        analysed = {int(row): index for index, row in enumerate(observed_rows)}
        print(f"  [simulation] {len(observed_rows)} of {len(generated.samples)} individuals sequenced", flush=True)
        # One read seed per contig, drawn in contig order, so scheduling never changes a read.
        read_rng = np.random.default_rng(seed + _READ_SEED_OFFSET)
        read_seeds = [int(read_rng.integers(0, 2 ** 63)) for _ in contigs]
        tasks = []
        for index, contig in enumerate(contigs):
            if store.contig(contig).has(records.READS):
                continue
            contig_data = simulated[index]
            tasks.append(dict(
                root=store.root, contig=contig, sample_ids=sample_ids, positions=sites_by_contig[index],
                offspring=[contig_data.haplotypes[row] for row in observed_rows],
                ancestry=[contig_data.ancestry[row] for row in observed_rows],
                founders=founders_by_contig[index],
                crossovers=records.truth_crossovers(contig_data.gametes, analysed, generated.samples,
                                                    generated.cohorts),
                depth=config.depth, read_model=config.reads, seed=read_seeds[index]))
        del simulated
        self._write_contigs(tasks, budget)
        store.require(records.READS, contigs)
        return records.SimulationManifest(
            state=records.COMPLETE, requested_seed=config.seed, seed=seed, contigs=contigs,
            identity=self.identity(), sample_ids=sample_ids, observed=observed, generated=generated)

    @staticmethod
    def _write_contigs(tasks, budget):
        """Simulate and record contigs in parallel workers, largest first, memory-bounded."""
        if not tasks:
            return
        sizes = [len(task["offspring"]) * len(task["positions"]) for task in tasks]
        order = sorted(range(len(tasks)), key=lambda index: (-sizes[index], index))
        estimates = [_READS_BYTES_PER_INDIVIDUAL_SITE * size + _READS_WORKER_BYTES for size in sizes]
        workers = execution.memory_worker_limit(estimates, min(len(tasks), budget.threads()),
                                                label="SIMULATED READS")
        with execution.BudgetedPool(budget, processes=workers, max_tasks_per_child=1,
                                    label="simulated reads") as pool:
            for row in pool.imap_unordered(_reads_worker, [tasks[index] for index in order],
                                           task_bytes=[estimates[index] for index in order]):
                execution.record_product("simulation", row["contig"], row)
                print(f"  [simulated reads] {row['contig']}: reads {row['simulate_seconds']:.1f}s, record "
                      f"{row['save_seconds']:.1f}s on {row['threads']} threads, worker peak "
                      f"{row['peak_rss_bytes'] / 2 ** 30:.1f} GiB", flush=True)
