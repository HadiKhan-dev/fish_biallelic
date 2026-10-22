"""The single driver: one stage sequence for VCF crosses, generic VCFs and simulations.

    check inputs (before any computation) -> start the run (cores, kernel cache, log, run record, config.toml)
    -> simulated inputs (simulation only) -> evidence + discovery (one persistent block pool for all contigs; the
    evidence of the next contig is prepared on discovery's helper thread) -> per contig in lanes: local selection
    -> final assembly -> painting -> pedigree -> founder refit -> family phase -> recombination -> export (the final
    founder and sample VCFs in OUTPUT/export; [export] enabled)

Serial mode runs discovery and the per-contig work one contig at a time (each recorded before the next starts);
a shard ([execution] process_contigs) runs its contigs through painting only. Every stage resumes from its records;
genome-wide completion is recorded in run.ckpt (complete/<stage>) only for full runs.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import time

from .. import execution
from ..config import RunConfig, default_output, stages, write_config
from ..store.registry import STAGE_NAMES
from .contig_work import PHASE_PROGRESS, ContigContext, contig_done, planning_bytes, reconstruct_contig
from .outputs import publish
from .samples import SampleTable, build_eligibility, read_sample_table
from .sources import observation_source

# Modules the forkserver imports once, before forking any worker (copy-on-write): every stage package and the
# modules the stages' pool workers run (named by their owners: blueprint 11.6). A module missing at import is
# skipped by the forkserver.
PRELOAD_MODULES = (
    # Third-party modules discovery workers import lazily at their first soft-cluster seed / residual birth
    # (base preloaded hdbscan too; it pulls scikit-learn).
    "hdbscan", "scipy.cluster.hierarchy", "scipy.spatial.distance",
    *(f"haplotype_reconstruction.{name}" for name in (*STAGE_NAMES, "simulation")),
    "haplotype_reconstruction.discovery.stage",
    "haplotype_reconstruction.local_selection.stage",
    "haplotype_reconstruction.assembly.hierarchy.batch",
    "haplotype_reconstruction.assembly.linker.em",
    "haplotype_reconstruction.assembly.join.search",
    "haplotype_reconstruction.assembly.join.beam",
    "haplotype_reconstruction.assembly.completion.stage",
    "haplotype_reconstruction.painting.stage",
    "haplotype_reconstruction.pedigree.stage",
    "haplotype_reconstruction.founder_refit.stage",
    "haplotype_reconstruction.reports",
    "haplotype_reconstruction.family_phase.stage",
    "haplotype_reconstruction.recombination.stage",
    "haplotype_reconstruction.simulation.source",
)


# Empirical runtime/retained-heap planning for reconstruction executed in the driver. The core term is a
# planning envelope, calibrated to 21.643 GB at 76 cores; scaling to other core counts is unverified.
# The minimum covers first use; this does not attribute the retained memory to individual threads.
DRIVER_RESIDENT_MIN_BYTES = 1536 << 20
DRIVER_RESIDENT_BASE_BYTES = 512 << 20
DRIVER_RESIDENT_BYTES_PER_CORE = 272 << 20


@dataclass(frozen=True)
class RunPaths:
    output: Path
    checkpoints: Path
    logs: Path

    @classmethod
    def from_config(cls, config: RunConfig) -> "RunPaths":
        output = config.run.output or default_output(config)
        checkpoints = config.run.checkpoints or output / "checkpoints"
        return cls(Path(output), Path(checkpoints), Path(output) / "logs")

    @classmethod
    def from_directory(cls, run) -> "RunPaths":
        """The paths of an existing run directory: its resolved config.toml's [run] checkpoints, else
        RUN/checkpoints (read without re-validating the configuration, so moved inputs do not matter)."""
        import tomllib
        output = Path(run).expanduser().resolve()
        checkpoints = output / "checkpoints"
        resolved = output / "config.toml"
        if resolved.is_file():
            value = tomllib.loads(resolved.read_text()).get("run", {}).get("checkpoints")
            if value:
                checkpoints = Path(value)
        return cls(output, checkpoints, output / "logs")


@dataclass(frozen=True)
class RunResult:
    """How an invocation ended: complete, or stopped after a stage (stop point, shard or contig limit)."""

    state: str
    stopped_after: str | None
    contigs: tuple[str, ...]


@dataclass
class PreparedRun:
    """Everything checked before any computation."""

    config: RunConfig
    paths: RunPaths
    cores: int
    source: object
    configs: stages.StageConfigs
    genetic_maps: object
    samples: SampleTable | None
    eligibility: object


def memory_limit(config: RunConfig) -> int | None:
    """[execution] memory_limit in bytes for the run's memory governor (None = auto: available at start)."""
    value = config.execution.memory_limit
    return None if value == "auto" else int(value)


def resolve_cores(config: RunConfig) -> int:
    available = execution.affinity_cpus()
    cores = available if config.execution.cores == "auto" else int(config.execution.cores)
    if not 1 <= cores <= available:
        raise ValueError(f"[execution] cores = {cores} exceeds the CPU affinity ({available})")
    return cores


def check_run(config: RunConfig) -> PreparedRun:
    """Fail before any computation: inputs, workbook and eligibility, stage configs, contig counts."""
    from ..pedigree import tier_b_minimum_contigs
    paths = RunPaths.from_config(config)
    cores = resolve_cores(config)
    # One canonical sample order for the whole run (release 2.1, design A-11): applied here, after any wrapper a test
    # harness puts around observation_source, so every record's sample axis is the samples sorted by name.
    from ..evidence.sample_order import canonical_order_source
    source = canonical_order_source(observation_source(config))
    samples = eligibility = None
    if config.input.kind == "vcf":
        source.check()
        samples = read_sample_table(config.samples, source.sample_ids())
        eligibility = build_eligibility(config, samples)
    configs = stages.stage_configs(config)
    maps = stages.genetic_maps(config)
    _check_store(paths, config, source)
    contigs = config.input.contigs
    genome_wide = not config.run.stop_after and not config.execution.process_contigs
    minimum = configs.pedigree.exposure.minimum_exposed_contigs
    if genome_wide and len(contigs) < minimum:
        raise ValueError(f"genome-wide pedigree inference needs at least {minimum} contigs ({len(contigs)} "
                         "configured); set [run] stop_after = \"painting\" for a smaller reconstruction")
    tier_b = tier_b_minimum_contigs(configs.pedigree)
    if genome_wide and len(contigs) < tier_b:
        print(f"WARNING: {len(contigs)} contigs: Tier-B pedigree calls need at least {tier_b} informative "
              "chromosomes, so this run can release no Tier-B parent; family phase and recombination then have "
              "no parent edges", flush=True)
    return PreparedRun(config, paths, cores, source, configs, maps, samples, eligibility)


def _check_store(paths: RunPaths, config: RunConfig, source) -> None:
    """Refuse, before anything is written, a checkpoint directory this release did not create, and a shard of a
    simulation whose inputs are not recorded yet (as base: concurrent shards would each simulate every contig)."""
    from ..store.store import CheckpointStore
    store = CheckpointStore(paths.checkpoints) if paths.checkpoints.is_dir() else None
    if store is not None:
        store.check_layout()
        # Completed-stage shortcuts must not reuse products from the refined pipeline.
        from ..assembly.stage import SCHEMA as ASSEMBLY_SCHEMA
        from ..store.store import CheckpointMismatch
        for contig in config.input.contigs:
            file = store.contig(contig)
            for key in file.keys("identity/"):
                if key == "identity/assembly" or key.endswith("/assembly"):
                    identity = file.get(key)
                    settings = identity["config"]
                    # The superseded completion-drop candidate also used schema 3,
                    # but still refined FINAL paths; its CONTEXT omitted completion.
                    if (identity.get("schema") != ASSEMBLY_SCHEMA or "refinement" in settings
                            or "completion" not in settings):
                        raise CheckpointMismatch(
                            f"{file.label}: assembly checkpoint is not from refinement-off assembly "
                            f"with completion (schema {ASSEMBLY_SCHEMA}); use a new output/checkpoint root")
    if config.execution.process_contigs and config.input.kind == "simulation" and (
            store is None or not source.is_materialised(store)):
        raise RuntimeError("a shard ([execution] process_contigs) needs the simulated inputs of every contig recorded "
                           "first: run the configuration once without process_contigs (the simulation of every "
                           "contig is recorded before any other stage; --max-contigs 1 stops after one contig)")


def run_pipeline(config: RunConfig, prepared: PreparedRun | None = None) -> RunResult:
    """Run (or resume) the configured pipeline; see the module docstring. ``prepared``: check_run(config), when the
    caller ran the fail-fast checks itself."""
    prepared = check_run(config) if prepared is None else prepared
    paths, cores = prepared.paths, prepared.cores
    paths.output.mkdir(parents=True, exist_ok=True)
    paths.logs.mkdir(parents=True, exist_ok=True)
    execution.start_run(cores, preload=PRELOAD_MODULES, profile=config.execution.profile,
                        memory_limit=memory_limit(config))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log = paths.logs / f"run_{stamp}_{os.getpid()}.log"
    from ..execution.kernel_cache import persistent_kernel_cache
    from ..execution.run_log import run_log
    from ..execution.run_record import record_run
    from ..config.schema import to_dict
    with run_log(log), record_run(paths.output, "run", log, config=to_dict(config),
                                  checkpoints=paths.checkpoints):
        print(f"Logging to: {log}", flush=True)
        write_config(config, paths.output / "config.toml")
        with ExitStack() as driver_memory:
            driver_resident = None

            def ensure_driver_resident():
                # Only direct reconstruction acquires this process-lifetime owner. It remains resident through
                # later contigs and genome-wide stages, without marking idle retained pages as running work.
                nonlocal driver_resident
                if driver_resident is None:
                    governor = execution.memory_governor()
                    nbytes = max(DRIVER_RESIDENT_MIN_BYTES,
                                 DRIVER_RESIDENT_BASE_BYTES + cores * DRIVER_RESIDENT_BYTES_PER_CORE)
                    driver_resident = governor.request(nbytes, label="reconstruction driver")
                    driver_memory.callback(driver_resident.release)
                    driver_resident.wait()
                    governor.rest(driver_resident.id)

            with persistent_kernel_cache(cores):
                return _run_stages(prepared, ensure_driver_resident)


def _summary_record(summary) -> dict:
    record = getattr(summary, "record", None)
    if callable(record):
        return record()
    return {key: value for key, value in vars(summary).items() if key != "prepared"}


def _run_stages(prepared: PreparedRun, ensure_driver_resident) -> RunResult:
    from ..discovery import records as discovery_records
    from ..discovery.stage import run as discover_contigs
    from ..evidence import run_contig as prepare_evidence
    from ..store.store import CheckpointStore
    config, paths, cores = prepared.config, prepared.paths, prepared.cores
    configs, source = prepared.configs, prepared.source
    contigs = config.input.contigs
    shard = config.execution.process_contigs
    todo = shard or contigs
    budget = execution.CpuBudget.fixed(cores)
    store = CheckpointStore(paths.checkpoints, nthreads=cores)
    store.bind_layout()
    started = time.monotonic()
    print(f"Run: {config.input.kind} input, {len(contigs)} contigs"
          + (f" (shard: {', '.join(shard)})" if shard else "") + f", {cores}-core ceiling", flush=True)

    with execution.stage_timer("simulation" if config.input.kind == "simulation" else "input"):
        source.materialise(store, contigs, budget)
    if config.input.kind == "simulation":
        from ..simulation.truth import load_manifest
        publish("write_truth_pedigree", paths.output, load_manifest(store))
    sample_ids = tuple(source.sample_ids())
    store.bind_manifest("run", dict(kind=config.input.kind, contigs=list(contigs), sample_ids=list(sample_ids),
                                    source=source.identity()))
    queue = config.execution.batch_queue
    context = ContigContext(configs, prepared.genetic_maps, queue, paths.output)

    def prepare(contig):
        """The evidence stage for one contig (discovery calls it one contig ahead on its helper thread); returns the
        contig's ContigEvidence in memory, so discovery need not decode the record again (as base)."""
        summary = prepare_evidence(store, source, contig, configs.evidence, budget, keep=True)
        execution.record_product("evidence", contig, _summary_record(summary))
        publish("write_quality", paths.output, store, contig, sample_ids)
        return summary.prepared.evidence

    def discover(selected):
        pending = [contig for contig in selected if not discovery_records.complete(store.contig(contig))]
        if pending:
            with execution.stage_timer("discovery"):
                discover_contigs(store, tuple(pending), configs.discovery, budget, prepare=prepare)

    if config.serial:
        if not _run_serially(store, todo, discover, context, cores, config.execution.max_contigs,
                             len(sample_ids), ensure_driver_resident):
            return RunResult("stopped", "max_contigs", todo)
    discover(todo)
    if not shard:
        _mark_when(store, "discovery", contigs, lambda contig: discovery_records.complete(store.contig(contig)))
        if config.reports.reference_consistency:
            publish("write_reference_consistency", paths.output, store, contigs, sample_ids,
                    prepared.samples.reference_rows, label=config.samples.reference_generation)
    if config.run.stop_after == "discovery":
        print("[STOP] discovery recorded; local selection, assembly and painting not started", flush=True)
        return RunResult("stopped", "discovery", todo)

    _reconstruct(store, todo, context, cores, config, len(sample_ids), paths, ensure_driver_resident)
    if not shard:
        _mark_when(store, "painting", contigs, lambda contig: contig_done(store, contig))
    if config.run.stop_after == "painting" or shard:
        print("[STOP] painting recorded; genome-wide stages not started", flush=True)
        return RunResult("stopped", "painting", todo)

    _genome_wide(store, contigs, sample_ids, prepared, budget, paths.output)
    print(f"Pipeline complete in {time.monotonic() - started:.1f} s", flush=True)
    return RunResult("complete", None, contigs)


def _mark_when(store, stage, contigs, done) -> None:
    missing = [contig for contig in contigs if not done(contig)]
    if missing:
        raise RuntimeError(f"{stage} is missing for: {', '.join(missing)}")
    store.mark_stage_complete(stage)


def _run_serially(store, contigs, discover, context, cores, max_contigs, n_samples,
                  ensure_driver_resident) -> bool:
    """Discover, assemble and paint one contig at a time, each recorded before the next; True when all are done.

    ``max_contigs`` (> 0) caps the contigs finished in this invocation, a resumed partial one included.
    """
    pending = [contig for contig in contigs if not contig_done(store, contig)]
    selected = pending if not max_contigs else pending[:max_contigs]
    print(f"SERIAL CONTIGS: {len(contigs) - len(pending)} of {len(contigs)} painted"
          + (f"; now {', '.join(selected)}, one at a time with {cores} cores" if selected else ""), flush=True)
    budget = execution.CpuBudget.fixed(cores)
    for done, contig in enumerate(selected, start=len(contigs) - len(pending) + 1):
        started = time.monotonic()
        discover([contig])
        ensure_driver_resident()
        with execution.reserve_memory(planning_bytes(store, contig, n_samples), label="reconstruction arrays"):
            reconstruct_contig(store, contig, context, budget)
        execution.release_memory()   # return the contig's freed heap before the next one (driver RSS ratchet)
        print(f"  [SERIAL] {contig} painted in {time.monotonic() - started:.1f} s; {done} of {len(contigs)}",
              flush=True)
    remaining = pending[len(selected):]
    if remaining:
        print(f"[STOP] max_contigs = {max_contigs} reached; remaining: {', '.join(remaining)}. Records are kept; "
              "rerun the same command to continue.", flush=True)
        return False
    return True


def _reconstruct(store, contigs, context, cores, config, n_samples, paths, ensure_driver_resident) -> None:
    from ..execution.contig_lanes import plan_lanes, run_lanes, run_sequentially, scan_concurrently
    pending = [not contig_done(store, contig) for contig in contigs]
    if not any(pending):
        print("[RESUME] every contig is painted", flush=True)
        return
    requested = None if config.execution.contig_lanes == "auto" else int(config.execution.contig_lanes)
    lanes, weights = 1, None
    if requested != 1 and len(contigs) > 1 and cores > 1:
        sizes = scan_concurrently(lambda contig: planning_bytes(store, contig, n_samples), contigs)
        lanes, weights = plan_lanes(sizes, pending, cores=cores, requested=requested)
    if lanes > 1:
        run_lanes(store, contigs, reconstruct_contig, context, cores=cores, lanes=lanes, weights=weights,
                  log_directory=paths.logs / "contigs", phase_progress=PHASE_PROGRESS,
                  done=lambda contig: contig_done(store, contig))
    else:
        ensure_driver_resident()
        run_sequentially(store, [c for c, flag in zip(contigs, pending) if flag], reconstruct_contig, context,
                         cores=cores, task_bytes=lambda contig: planning_bytes(store, contig, n_samples))


def _genome_wide(store, contigs, sample_ids, prepared: PreparedRun, budget, output) -> None:
    """Pedigree, founder refit, family phase, recombination and the export (each stage resumes and marks itself
    complete), then the run's tie report."""
    from .. import founder_refit, pedigree
    from ..family_phase.stage import run as phase_families
    from ..recombination.stage import run as map_recombination
    configs, maps = prepared.configs, prepared.genetic_maps
    with execution.stage_timer("pedigree"):
        product = pedigree.run(store, contigs, sample_ids, prepared.eligibility, configs.pedigree, maps, budget)
    from ..evidence.sample_order import store_input_order
    publish("write_pedigree", output, product, input_ids=store_input_order(store))
    with execution.stage_timer("founder_refit"):
        summaries = founder_refit.run(store, contigs, configs.founder_refit, budget)
    publish("write_founder_refit", output, summaries)
    with execution.stage_timer("family_phase"):
        summaries = phase_families(store, contigs, product.edges, configs.family_phase, maps, budget)
    publish("write_family_phase", output, summaries)
    with execution.stage_timer("recombination"):
        map_recombination(store, contigs, product.edges, configs.recombination, maps, budget)
    publish("write_recombination", output, store, contigs)
    if prepared.config.export.enabled:
        from ..export import stage as export_stage
        from ..export.run import reference_vcf
        synthetic = prepared.config.input.kind == "simulation"
        with execution.stage_timer("export"):
            export_stage.run(store, contigs, configs.export, output, budget, synthetic=synthetic,
                             reference=None if synthetic else reference_vcf(store, output, None))
    # Decisions taken inside a tie and the recorded ambiguity flags (docs/methods/ties.md), from the records.
    publish("write_ties", output, store, contigs, product.tie_events)


__all__ = ["PRELOAD_MODULES", "PreparedRun", "RunPaths", "RunResult", "check_run", "run_pipeline"]


def rerun_recombination(config: RunConfig, run_directory) -> None:
    """The recombination stage of the run in ``run_directory`` (whose family phase is recorded): runs or resumes it
    with ``config`` (the run's config.toml plus recombination overrides) and republishes its tables. The run is
    located from the directory itself (its checkpoints: config.toml's [run] checkpoints, else RUN/checkpoints), not
    from paths in the configuration. A recorded stage is bound to the settings it ran with, so changed recombination
    settings are refused (use a copy of the run directory); the settings of a stage recorded here are written to the
    run's config.toml."""
    from dataclasses import replace
    from ..execution.kernel_cache import persistent_kernel_cache
    from ..recombination.stage import run as map_recombination
    from ..execution.run_log import run_log
    from ..execution.run_record import record_run
    from ..config.schema import to_dict
    from ..pedigree import records as pedigree_records
    from ..store.store import CheckpointStore
    paths = RunPaths.from_directory(run_directory)
    if not paths.checkpoints.is_dir():
        raise FileNotFoundError(f"{paths.output}: no checkpoints at {paths.checkpoints}")
    CheckpointStore(paths.checkpoints).check_layout()
    config = replace(config, run=replace(config.run, output=paths.output))
    cores = resolve_cores(config)
    configs = stages.stage_configs(config)
    execution.start_run(cores, preload=PRELOAD_MODULES, profile=config.execution.profile,
                        memory_limit=memory_limit(config))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log = paths.logs / f"recombination_{stamp}_{os.getpid()}.log"
    paths.logs.mkdir(parents=True, exist_ok=True)
    with run_log(log), record_run(paths.output, "recombination", log, config=to_dict(config),
                                  checkpoints=paths.checkpoints), persistent_kernel_cache(cores):
        store = CheckpointStore(paths.checkpoints, nthreads=cores)
        store.bind_layout()
        edges = pedigree_records.load_edges(store)
        with execution.stage_timer("recombination"):
            map_recombination(store, config.input.contigs, edges, configs.recombination,
                              stages.genetic_maps(config), execution.CpuBudget.fixed(cores))
        publish("write_recombination", paths.output, store, config.input.contigs)
        write_config(config, paths.output / "config.toml")   # the settings the recorded stage ran with
