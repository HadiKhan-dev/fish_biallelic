"""Checkpointed family inference and stable, genotype-preserving final phase.

family phase publishes one conditional phase product. Phase stability is not a claim of
marginal-posterior convergence; full family probability tensors are not built.
"""
from __future__ import annotations
from ..core.run_record import timed_stage

from dataclasses import asdict, replace
import gc
import os
from pathlib import Path
import time
import numpy as np

from haplotype_reconstruction.core import parallel, products, runtime
from haplotype_reconstruction.core.chromosome_parallel import (
    ChromosomeExecutor, current_threads, memory_worker_limit,
)
from haplotype_reconstruction.pedigree import components as pedigree_components
from haplotype_reconstruction.painting import allele_refit as painting_allele_refit
from haplotype_reconstruction.painting import checkpoints as painting_checkpoints
from haplotype_reconstruction.workflows.reconstruction import PAINTING_STAGE
from . import conditioning, frames, model, polish


FINAL_PHASE_STAGE = "family_phase"
CANONICAL_PHASE_CONFIG = polish.PhasePolishConfig(
    copy_error=.01, adaptive_window_sites=100, minimum_biological_gain=0.,
    polish_from_scaffold=True)


class PhaseStabilityError(RuntimeError):
    """The supported solve reached its limit without a stable called phase."""

    def __init__(self, checks):
        super().__init__("final called phase did not stabilize; iteration state retained, no final product released")
        self.checks = tuple(checks)


def _fit_phase(gl, observed, scaffold, positions, relationships, names, *,
               config, phase_config, identity, work=None, checkpoint_threads=1,
               checkpoint_min_seconds=120., progress_callback=None, chromosome_map=None,
               frame_inputs=None):
    """Resume family messages and the actual consecutive called-phase checks.

The prepared workspace survives one-iteration continuations but is rebuilt on
restart. Each polisher starts independently from the current family context
(with ``polish_from_scaffold``, from the scaffold); the preceding polished
path is only a comparison, never a warm start.

After the stable release, root frames are repaired once on the final family
state (refinement/frames.py) and the polish is rerun on the repaired scaffold.
This step is a deterministic function of the final messages, so a resumed
stage repeats it exactly. It needs the scaffold polish, which it edits.
"""
    if config.max_iterations < config.minimum_iterations:
        raise ValueError("maximum family iterations must cover the initial phase window")
    if config.required_unchanged < 2:
        raise ValueError("phase release needs at least two consecutive unchanged checks")
    if not np.isfinite(checkpoint_min_seconds) or checkpoint_min_seconds < 0:
        raise ValueError("checkpoint_min_seconds must be finite and nonnegative")
    state = point = None
    unchanged = 0
    checks = []

    def work_threads():
        # Encode and decode with the live CPU lease, like the chromosome's other records.
        work.nthreads = current_threads() if checkpoint_threads is None else checkpoint_threads
        return work

    saved = None if work is None else work_threads().load()
    if saved is not None:
        if saved["identity"] != identity:
            raise ValueError("Family phase iteration state inputs/configuration differ")
        state, point, unchanged = saved["messages"], saved["phase"], saved["unchanged"]
        checks = list(saved["checks"])
        del saved
    converged = bool(state is not None and state.deltas and state.deltas[-1] < config.tolerance)
    # Progress is saved every checkpoint_min_seconds of work, never sooner: a
    # resumed solve repeats any later iterations and checks exactly, and the
    # final product follows the stable release within seconds.
    last_saved = time.perf_counter()

    def save(messages, phase, stable_count, *, force=False):
        nonlocal last_saved
        if work is not None and (force or time.perf_counter() - last_saved >= checkpoint_min_seconds):
            # Emissions are recomputed before the next chain/MAP update.
            # Shallow copies preserve the live solver, cavities and phase checks.
            saved_messages = replace(messages, branch_clusters=tuple(
                replace(cluster, emissions=None) for cluster in messages.branch_clusters))
            work_threads().save({
                "identity": identity, "messages": saved_messages, "phase": phase,
                "unchanged": stable_count, "checks": tuple(checks),
            })
            last_saved = time.perf_counter()

    def save_initial(messages):
        # Before the first phase assessment there is no preceding phase whose
        # iteration could become inconsistent with these checkpointed messages.
        save(messages, None, 0)

    workspace = family = None
    # Only the immediately preceding independent solve is reusable. Fixed
    # chromosome axes, pedigree, map and configuration belong to this call;
    # restarting from messages intentionally starts with an empty cache.
    # With polish_from_scaffold the solve reads only these fixed inputs (the
    # scaffold alleles and a zero start), never the family state, so the first
    # solve of the call serves every later check.
    polished = False
    polish_context = polish_initial = None
    while point is None or (not converged and unchanged < config.required_unchanged):
        current_threads()
        iteration = 0 if state is None else state.iteration
        if point is not None and iteration >= config.max_iterations:
            save(state, point, unchanged, force=True)
            raise PhaseStabilityError(checks)
        stop = (max(config.minimum_iterations, iteration) if point is None else iteration + 1)
        if workspace is None:
            workspace = model.prepare_family_workspace(
                gl, observed, scaffold.reference_alleles, positions, scaffold.phase_bins,
                scaffold.component_ids, relationships, names, config=config,
                chromosome_map=chromosome_map)
        family = model.refine_family(
            gl, observed, scaffold.reference_alleles, positions, scaffold.phase_bins,
            scaffold.component_ids, relationships, names,
            config=replace(config, max_iterations=stop), resume=state,
            checkpoint_callback=save_initial if point is None else None,
            workspace=workspace, chromosome_map=chromosome_map)
        state = family.messages
        try:
            same_inputs = polished and (phase_config.polish_from_scaffold or (
                np.array_equal(polish_context, family.phase_context)
                and np.array_equal(polish_initial, family.inferred_phase_map)))
            following = polish.finish_family_phase(
                scaffold.reference_alleles, family, positions, scaffold.phase_bins,
                scaffold.component_ids, workspace.parents, workspace.children, workspace.slots,
                config=phase_config, chromosome_map=chromosome_map,
                conditional_result=point.conditional_result if same_inputs else None)
            if not same_inputs and not phase_config.polish_from_scaffold:
                polish_context = family.phase_context.copy()
                polish_initial = family.inferred_phase_map.copy()
            polished = True
        except Exception:
            # These messages have advanced; do not pair them with an older
            # phase check. Retry this exact iteration after an interruption.
            save(state, None, 0, force=True)
            raise
        changed = None if point is None else int(
            np.count_nonzero(point.allele_calls != following.allele_calls)
        )
        unchanged = unchanged + 1 if changed == 0 else 0
        point, converged = following, bool(family.converged)
        item = {"iteration": state.iteration, "latent_delta": state.deltas[-1],
                "latent_converged": converged, "changed_called_alleles": changed,
                "consecutive_unchanged": unchanged}
        checks.append(item)
        save(state, point, unchanged)
        print(f"  [family phase] iteration={state.iteration} delta={state.deltas[-1]:.6g} "
              f"changed_alleles={changed} unchanged_checks={unchanged}", flush=True)
        if progress_callback is not None:
            progress_callback(item)
        del following
    repair = None
    if frame_inputs is not None and config.root_frame_repair and phase_config.polish_from_scaffold:
        if workspace is None:
            workspace = model.prepare_family_workspace(
                gl, observed, scaffold.reference_alleles, positions, scaffold.phase_bins,
                scaffold.component_ids, relationships, names, config=config,
                chromosome_map=chromosome_map)
        if family is None:
            # Resumed after the release: rebuild the final family view, no iteration.
            family = model.refine_family(
                gl, observed, scaffold.reference_alleles, positions, scaffold.phase_bins,
                scaffold.component_ids, relationships, names,
                config=replace(config, max_iterations=state.iteration), resume=state,
                workspace=workspace, chromosome_map=chromosome_map)
        repair = frames.repair_root_frames(workspace, state, family.inferred_phase_map, config,
                                           **frame_inputs)
        if repair["display"] is not None:
            point = frames.finish_repaired_phase(
                scaffold.reference_alleles, family, repair, positions, scaffold.phase_bins,
                scaffold.component_ids, workspace.parents, workspace.children, workspace.slots,
                phase_config=phase_config, chromosome_map=chromosome_map,
                finish=polish.finish_family_phase)
        print(f"  [family phase] root frame repair: moves={len(repair['moves'])} "
              f"reverts={len(repair['reverts'])} transfers={len(repair['transfers'])} "
              f"rejected={repair['rejected']}", flush=True)
    return point, tuple(checks), converged, repair


def work_name(attempt):
    """Name of an attempt's transient iteration state (CheckpointStore.work)."""
    return FINAL_PHASE_STAGE if attempt == 0 else f"{FINAL_PHASE_STAGE}.retry{attempt}"


def refine_chromosome(painting_checkpoint, gl, positions, observed, relationships, sample_ids, *,
                      contig, config=model.FamilyRefinementConfig(), phase_config=None,
                      identity=None, work_store=None, checkpoint_threads=1,
                      checkpoint_min_seconds=120., progress_callback=None, chromosome_map=None,
                      refit_panel=None, founder_evidence_excluded=None, protected_intervals=None,
                      recorded_digests=None):
    """Refine frozen painting/pedigree; ``checkpoint_threads=None`` uses the live CPU lease.

    ``work_store`` (a CheckpointStore) keeps each attempt's iteration state
    for resume, under ``work_name(attempt)``; None disables it.
    ``refit_panel`` (the allele refit of this painting) supplies the scaffold
    founder alleles; without it the frozen assembly snapshot is decoded.
    ``founder_evidence_excluded`` loads the contig's sample-quality flags and
    ``recorded_digests`` are the release's recorded identities of these arrays
    (core.products.evidence_checks), both for the release-identity check only;
    every sample's evidence is decoded.
    ``protected_intervals`` (closed bp intervals, shape (n, 2)) are reference
    order breakpoints where no root frame repair is made.
    """
    names = tuple(map(str, sample_ids))
    painting_checkpoint = painting_checkpoints.validate_painting_checkpoint(painting_checkpoint, expected_sample_ids=names)
    pedigree_components._validate_release_array_identity(
        painting_checkpoint, np.asarray(gl), np.asarray(positions), np.asarray(observed),
        founder_evidence_excluded, recorded_digests)
    if chromosome_map is not None:
        if chromosome_map.contig != str(contig):
            raise ValueError("Family phase chromosome map name differs from input contig")
        config = replace(config, recombination_rate=chromosome_map.fallback_rate_per_bp)
    config = config.validated()
    phase_config = replace(phase_config or CANONICAL_PHASE_CONFIG,
                           recombination_rate=config.recombination_rate)
    protected = (None if protected_intervals is None
                 else np.asarray(protected_intervals, dtype=np.int64).reshape((-1, 2)))
    family_identity = {
        "model": model.MODEL_VERSION, "config": asdict(config), "sample_ids": names,
        "contig": str(contig), "assembly": painting_checkpoint.release_identity.record(),
        "tpainting": painting_checkpoint.painting_product_identity.record(),
        "pedigree_sha256": conditioning.relationship_identity(relationships),
        "recombination_map": None if chromosome_map is None else chromosome_map.identity(),
        "allele_refit": None if refit_panel is None else refit_panel.identity,
        "protected_intervals": None if protected is None else protected.tolist(),
        "external": identity,
    }
    product_identity = {"schema": conditioning.SCHEMA, "family": family_identity,
                        "phase_config": asdict(phase_config),
                        "code": conditioning.refinement_code_identity()}
    scaffold = conditioning.prepare_phase_scaffold(painting_checkpoint, positions, refit_panel)
    frame_inputs = None
    if config.root_frame_repair:
        edge_parents = model.pedigree_edges(relationships, names)[0]
        counts = np.bincount(edge_parents, minlength=len(names))
        frame_inputs = {
            "tie_boundaries": frames.painting_tie_boundaries(
                painting_checkpoint, positions,
                np.flatnonzero(counts >= config.root_frame_minimum_children)),
            "protected": protected}
    attempts = []
    for retry in range(config.phase_retry_count + 1):
        # Restart only after a genuine stability failure. Scale tolerance with
        # damping so the undamped fixed-point residual criterion is unchanged.
        scale = .5 ** retry
        attempt_config = replace(config, damping=config.damping * scale,
                                 tolerance=config.tolerance * scale)
        product_identity = {**product_identity, "family": {
            **family_identity, "config": asdict(attempt_config)}}
        work = None if work_store is None else work_store.work(contig, work_name(retry))
        record = {"attempt": retry, "damping": attempt_config.damping,
                  "tolerance": attempt_config.tolerance,
                  "work_state": None if work is None else work_name(retry)}
        try:
            phase, checks, converged, repair = _fit_phase(
                gl, observed, scaffold, positions, relationships, names, config=attempt_config,
                phase_config=phase_config, identity=product_identity, work=work,
                checkpoint_threads=checkpoint_threads, checkpoint_min_seconds=checkpoint_min_seconds,
                progress_callback=progress_callback, chromosome_map=chromosome_map,
                frame_inputs=frame_inputs)
        except PhaseStabilityError as error:
            record.update(phase_stable=False, iterations=error.checks[-1]["iteration"])
            attempts.append(record)
            if retry == config.phase_retry_count:
                raise
            print(f"[family phase {contig}] Phase not stable at damping={attempt_config.damping:g}; "
                  "retrying from the scaffold with half damping.", flush=True)
        else:
            record.update(phase_stable=True, iterations=checks[-1]["iteration"])
            attempts.append(record)
            break
    return {
        "identity": product_identity, "contig": str(contig), "sample_ids": names,
        "positions": np.asarray(positions), "component_ids": scaffold.component_ids,
        "phase": phase,
        "corrected_component_paintings": conditioning.paint_final_phase(painting_checkpoint, positions, phase.phase_map),
        "phase_stable": True, "phase_stability_checks": checks,
        "phase_solver_attempts": attempts,
        "root_frame_repairs": None if repair is None else {
            key: repair.get(key, []) for key in
            ("moves", "reverts", "transfers", "child_frames", "rejected", "guard_details",
             "flagged_edges")},
        "source_posterior_converged": converged,
        "posterior_and_recombination_products_included": False,
        "confidence_policy": "conditional phase point path; no marginal posterior probabilities are published or transferred",
    }


def _summary(result, file, elapsed):
    phase = result["phase"].summary
    checks = result["phase_stability_checks"]
    repairs = result.get("root_frame_repairs") or {}
    return {
        "contig": result["contig"], "samples": len(result["sample_ids"]),
        "sites": len(result["positions"]), "phase_stable": result["phase_stable"],
        "source_posterior_converged": result["source_posterior_converged"],
        "iterations": checks[-1]["iteration"], "phase_stability_checks": len(checks),
        "solver_attempts": len(result["phase_solver_attempts"]),
        "total_solver_iterations": sum(a["iterations"] for a in result["phase_solver_attempts"]),
        "final_damping": result["phase_solver_attempts"][-1]["damping"],
        "consecutive_unchanged": checks[-1]["consecutive_unchanged"],
        "called_alleles": phase["called_alleles"],
        "changed_observable_phase_sites": phase["changed_observable_phase_sites"],
        "known_genotypes_changed": phase["known_genotypes_changed"],
        "imputation_enabled": False, "posterior_and_recombination_products_included": False,
        "confidence_policy": result["confidence_policy"], "elapsed_seconds": elapsed,
        "final_checkpoint": str(Path(file.path).resolve()),
        "root_frame_moves": len(repairs.get("moves", ())),
        "root_frame_reverts": len(repairs.get("reverts", ())),
        "root_frame_transfers": len(repairs.get("transfers", ())),
    }


def _refine_contig(contig, state, phase_name, common):
    """Own one chromosome's evidence, iteration state and final product."""
    parallel.malloc_trim()
    started = time.perf_counter()
    store = runtime.CheckpointStore(common["checkpoint_root"])
    file = store.contig(contig)
    identity = common["identity"]
    if file.has(FINAL_PHASE_STAGE):
        result = file.get(FINAL_PHASE_STAGE, nthreads=current_threads())
        if result["identity"]["family"]["external"] != identity:
            raise ValueError("Family phase final chromosome identity differs")
    else:
        painting = file.get(PAINTING_STAGE, nthreads=current_threads())
        refit = file.get(painting_allele_refit.ALLELE_REFIT_STAGE, nthreads=current_threads())
        gl, pos, observed = products.load_evidence(file, nthreads=current_threads())
        flags, digests = products.evidence_checks(file)
        maps = common["genetic_maps"]
        chromosome_map = None if maps is None else maps.for_contig(contig)
        protected = common["protected_intervals"]
        current_threads()
        result = refine_chromosome(
            painting, gl, pos, observed, common["relationships"], common["names"],
            contig=contig, config=common["config"], identity=identity, work_store=store,
            checkpoint_threads=None, chromosome_map=chromosome_map, refit_panel=refit,
            founder_evidence_excluded=flags, recorded_digests=digests,
            protected_intervals=None if protected is None else protected.get(contig, ()))
        file.put(FINAL_PHASE_STAGE, result, nthreads=current_threads())
        del painting, refit, gl, pos, observed
    # The released product supersedes every attempt's resume state.
    for attempt in range(common["config"].phase_retry_count + 1):
        store.work(contig, work_name(attempt)).clear()
    summary = _summary(result, file, time.perf_counter() - started)
    print(f"[family phase {contig}] Stable final phase released; latent converged="
          f"{result['source_posterior_converged']}; threads={current_threads()}", flush=True)
    del result
    gc.collect()
    parallel.malloc_trim()
    return None, summary


def _refinement_resources(store, contigs, workers):
    """Bound concurrent chromosome workspaces using decoded source sizes.

    Float64 GLs, four-state family messages, selector caches, polishing and
    iteration state peaked at 7.7 decoded-evidence sizes plus 3.2 GiB for
    Python/JIT, plus up to 3.6 GiB of heap a worker retains from earlier
    chromosomes (N = 320, independent of the thread count: 94 GiB for 11.2
    GiB of evidence, 19-38 GiB for 1.9-4.1 GiB). Nine evidence-sized copies,
    two painting copies and 5 GiB per process stay above every measured peak
    by 12-21%. This is a scheduling estimate, not an RSS guarantee. Already
    completed chromosomes need only a decoded product and read buffers.
    """
    estimates = []
    for contig in contigs:
        file = store.contig(contig)
        if file.has(FINAL_PHASE_STAGE):
            estimate = 2 * file.info(FINAL_PHASE_STAGE).decoded_bytes
        else:
            estimate = (9 * file.info(products.EVIDENCE).decoded_bytes
                        + 2 * file.info(PAINTING_STAGE).decoded_bytes)
        estimates.append(estimate + (5 << 30))
    return estimates, memory_worker_limit(
        estimates, workers, runtime.available_memory_bytes(), label="FAMILY PHASE")


@timed_stage("family_phase")
def run_refinement(store, contigs, sample_ids, *, pedigree_payload, output_dir,
                   n_workers=None, config=None, genetic_maps=None, protected_intervals=None):
    """Run independent chromosomes with a shared, dynamically reassigned CPU budget.

    Only final phase is a release product. Work-in-progress messages and
    consecutive phase checks are transient resume state (CheckpointStore.work),
    removed once the chromosome's final phase is stored.
    ``protected_intervals`` ({contig: (n, 2) bp intervals}, or a breakpoint
    TSV path in BHD_FAMILY_PHASE_PROTECTED_INTERVALS) blocks root frame
    repairs at reference-order breakpoints; default none.
    """
    if pedigree_payload is None:
        print("[family phase] Waiting for the complete genome-wide pedigree; no shard inference.")
        return None
    if protected_intervals is None and os.environ.get(frames.PROTECTED_INTERVALS_ENV):
        protected_intervals = frames.read_protected_intervals(os.environ[frames.PROTECTED_INTERVALS_ENV])
    config = (config or model.FamilyRefinementConfig()).validated()
    if genetic_maps is not None:
        config = replace(config, recombination_rate=genetic_maps.default_rate_cm_per_mb / 1e8).validated()
    workers = runtime.available_cpu_count() if n_workers is None else int(n_workers)
    if not 1 <= workers <= runtime.available_cpu_count():
        raise ValueError("Family phase CPU budget exceeds allocation")
    names = tuple(map(str, sample_ids))
    contigs = tuple(map(str, contigs))
    if (tuple(map(str, pedigree_payload["ordered_sample_ids"])) != names or
            tuple(map(str, pedigree_payload["ordered_contigs"])) != contigs):
        raise ValueError("Family phase axes differ from the complete pedigree")
    relationships = pedigree_payload["tier_b_partial_relationships"]
    identity = {
        "schema": conditioning.SCHEMA, "config": asdict(config),
        "phase_config": asdict(replace(CANONICAL_PHASE_CONFIG, recombination_rate=config.recombination_rate)),
        "sample_ids": names, "contigs": contigs,
        "source_records": products.record_identities(
            store, (PAINTING_STAGE, painting_allele_refit.ALLELE_REFIT_STAGE, products.EVIDENCE), contigs),
        "pedigree_sha256": conditioning.relationship_identity(relationships),
        "code": conditioning.refinement_code_identity(),
        "recombination_maps": None if genetic_maps is None else {c: genetic_maps.identity(c) for c in contigs},
        "protected_intervals": None if protected_intervals is None else {
            c: np.asarray(protected_intervals.get(c, ()), dtype=np.int64).reshape((-1, 2)).tolist()
            for c in contigs},
    }
    store.bind_identity(store.run, FINAL_PHASE_STAGE, identity)
    output = Path(output_dir) / "phase_correction"
    output.mkdir(parents=True, exist_ok=True)
    if store.stage_complete(FINAL_PHASE_STAGE) and store.run.has(FINAL_PHASE_STAGE):
        runtime.require_records(store, FINAL_PHASE_STAGE, contigs)
        saved = store.run.get(FINAL_PHASE_STAGE)
        if saved["identity"] != identity:
            raise ValueError("Family phase final summary identity differs")
        conditioning._write_summary_table(output, FINAL_PHASE_STAGE, saved["summaries"])
        print("[family phase] Resumed complete final phase output from its compact summary.")
        return saved["summaries"]
    weights, process_limit = _refinement_resources(store, contigs, workers)
    common = dict(checkpoint_root=store.root, identity=identity,
                  names=names, relationships=relationships, config=config,
                  genetic_maps=genetic_maps, protected_intervals=protected_intervals)
    with ChromosomeExecutor(contigs, _refine_contig, n_workers=workers,
                            weights=weights, max_workers=process_limit,
                            label="FAMILY PHASE") as executor:
        summaries = executor.run("refine", common)
    runtime.require_records(store, FINAL_PHASE_STAGE, contigs)
    store.run.put(FINAL_PHASE_STAGE, {"identity": identity, "summaries": summaries})
    conditioning._write_summary_table(output, FINAL_PHASE_STAGE, summaries)
    store.mark_stage_complete(FINAL_PHASE_STAGE)
    print(f"[COMPLETE] Final phase ready for recombination maps: {output / (FINAL_PHASE_STAGE + '.csv')}")
    return summaries
