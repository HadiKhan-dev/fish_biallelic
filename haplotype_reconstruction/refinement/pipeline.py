"""Checkpointed family inference and stable, genotype-preserving final phase.

T11 publishes one conditional phase product. Phase stability is not a claim of
marginal-posterior convergence; full family probability tensors are not built.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import gc
from pathlib import Path
import time
import numpy as np

from haplotype_reconstruction.core import checkpoints, parallel, runtime
from haplotype_reconstruction.pedigree import components as pedigree_components
from haplotype_reconstruction.pedigree import pipeline as pedigree_pipeline
from haplotype_reconstruction.painting import checkpoints as painting_checkpoints
from haplotype_reconstruction.workflows.reconstruction import PAINTING_STAGE
from . import conditioning, model, polish


FINAL_PHASE_STAGE = "11_phase_correction"
CANONICAL_PHASE_CONFIG = polish.PhasePolishConfig(
    copy_error=.01, adaptive_window_sites=100, minimum_biological_gain=0.)


def _fit_phase(gl, observed, scaffold, positions, relationships, names, *,
               config, phase_config, identity, work_path=None, checkpoint_threads=1,
               checkpoint_min_seconds=120., progress_callback=None, chromosome_map=None):
    """Resume family messages and the actual consecutive called-phase checks.

The prepared workspace survives one-iteration continuations but is rebuilt on
restart. Each polisher starts independently from the current family context;
the preceding polished path is only a comparison, never a warm start.
"""
    if config.max_iterations < config.minimum_iterations:
        raise ValueError("maximum family iterations must cover the initial phase window")
    if config.required_unchanged < 2:
        raise ValueError("phase release needs at least two consecutive unchanged checks")
    if not np.isfinite(checkpoint_min_seconds) or checkpoint_min_seconds < 0:
        raise ValueError("checkpoint_min_seconds must be finite and nonnegative")
    work = None if work_path is None else Path(work_path)
    state = point = None
    unchanged = 0
    checks = []
    if work is not None and work.is_file():
        saved = checkpoints.read(str(work), nthreads=checkpoint_threads)
        if saved["identity"] != identity:
            raise ValueError("Stage11 iteration checkpoint inputs/configuration differ")
        state, point, unchanged = saved["messages"], saved["phase"], saved["unchanged"]
        checks = list(saved["checks"])
        del saved
    converged = bool(state is not None and state.deltas and state.deltas[-1] < config.tolerance)
    last_saved = -np.inf

    def save(messages, phase, stable_count, *, force=False):
        nonlocal last_saved
        if work is not None and (force or time.perf_counter()-last_saved >= checkpoint_min_seconds):
            checkpoints.write(str(work), {
                "identity": identity, "messages": messages, "phase": phase,
                "unchanged": stable_count, "checks": tuple(checks),
            }, nthreads=checkpoint_threads)
            last_saved = time.perf_counter()

    def save_initial(messages):
        # Before the first phase assessment there is no preceding phase whose
        # iteration could become inconsistent with these checkpointed messages.
        save(messages, None, 0)

    workspace = None
    # Only the immediately preceding independent solve is reusable. Fixed
    # chromosome axes, pedigree, map and configuration belong to this call;
    # restarting from messages intentionally starts with an empty cache.
    polish_context = polish_initial = None
    while point is None or (not converged and unchanged < config.required_unchanged):
        iteration = 0 if state is None else state.iteration
        if point is not None and iteration >= config.max_iterations:
            save(state, point, unchanged, force=True)
            raise RuntimeError("final called phase did not stabilize; iteration checkpoint retained, no final product released")
        stop = (max(config.minimum_iterations, iteration) if point is None else iteration+1)
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
            same_inputs = (polish_context is not None
                and np.array_equal(polish_context, family.phase_context)
                and np.array_equal(polish_initial, family.inferred_phase_map))
            following = polish.finish_family_phase(
                scaffold.reference_alleles, family, positions, scaffold.phase_bins,
                scaffold.component_ids, workspace.parents, workspace.children, workspace.slots,
                config=phase_config, chromosome_map=chromosome_map,
                conditional_result=point.conditional_result if same_inputs else None)
            if not same_inputs:
                polish_context = family.phase_context.copy()
                polish_initial = family.inferred_phase_map.copy()
        except Exception:
            # These messages have advanced; do not pair them with an older
            # phase check. Retry this exact iteration after an interruption.
            save(state, None, 0, force=True)
            raise
        changed = None if point is None else int(np.count_nonzero(point.allele_calls != following.allele_calls))
        unchanged = unchanged+1 if changed == 0 else 0
        point, converged = following, bool(family.converged)
        item = {"iteration": state.iteration, "latent_delta": state.deltas[-1],
                "latent_converged": converged, "changed_called_alleles": changed,
                "consecutive_unchanged": unchanged}
        checks.append(item)
        stable = converged or unchanged >= config.required_unchanged
        save(state, point, unchanged, force=stable or state.iteration >= config.max_iterations)
        print(f"  [T11] iteration={state.iteration} delta={state.deltas[-1]:.6g} "
              f"changed_alleles={changed} unchanged_checks={unchanged}", flush=True)
        if progress_callback is not None:
            progress_callback(item)
        del family, following
    return point, tuple(checks), converged


def refine_chromosome(t09, gl, positions, observed, relationships, sample_ids, *,
                      contig, config=model.FamilyRefinementConfig(), phase_config=None,
                      identity=None, work_path=None, checkpoint_threads=1,
                      checkpoint_min_seconds=120., progress_callback=None, chromosome_map=None):
    """Produce one independently resumable T11 chromosome from frozen T09/T10."""
    names = tuple(map(str, sample_ids))
    t09 = painting_checkpoints.validate_t09_component_checkpoint(t09, expected_sample_ids=names)
    pedigree_components._validate_release_array_identity(
        t09, np.asarray(gl), np.asarray(positions), np.asarray(observed))
    if chromosome_map is not None:
        if chromosome_map.contig != str(contig):
            raise ValueError("Stage11 chromosome map name differs from input contig")
        config = replace(config, recombination_rate=chromosome_map.fallback_rate_per_bp)
    config = config.validated()
    phase_config = replace(phase_config or CANONICAL_PHASE_CONFIG,
                           recombination_rate=config.recombination_rate)
    family_identity = {
        "model": model.MODEL_VERSION, "config": asdict(config), "sample_ids": names,
        "contig": str(contig), "t09_release": t09.release_identity.record(),
        "t09_painting": t09.painting_product_identity.record(),
        "pedigree_sha256": conditioning.relationship_identity(relationships),
        "recombination_map": None if chromosome_map is None else chromosome_map.identity(),
        "external": identity,
    }
    product_identity = {"schema": conditioning.SCHEMA, "family": family_identity,
                        "phase_config": asdict(phase_config),
                        "code": conditioning.refinement_code_identity()}
    scaffold = conditioning.prepare_phase_scaffold(t09, positions)
    phase, checks, converged = _fit_phase(
        gl, observed, scaffold, positions, relationships, names, config=config,
        phase_config=phase_config, identity=product_identity, work_path=work_path,
        checkpoint_threads=checkpoint_threads, checkpoint_min_seconds=checkpoint_min_seconds,
        progress_callback=progress_callback, chromosome_map=chromosome_map)
    return {
        "identity": product_identity, "contig": str(contig), "sample_ids": names,
        "positions": np.asarray(positions), "component_ids": scaffold.component_ids,
        "phase": phase,
        "corrected_component_paintings": conditioning.paint_final_phase(t09, positions, phase.phase_map),
        "phase_stable": True, "phase_stability_checks": checks,
        "source_posterior_converged": converged,
        "posterior_and_recombination_products_included": False,
        "confidence_policy": "conditional phase point path; no marginal posterior probabilities are published or transferred",
    }


def _summary(result, store, elapsed):
    phase = result["phase"].summary
    checks = result["phase_stability_checks"]
    return {
        "contig": result["contig"], "samples": len(result["sample_ids"]),
        "sites": len(result["positions"]), "phase_stable": result["phase_stable"],
        "source_posterior_converged": result["source_posterior_converged"],
        "iterations": checks[-1]["iteration"], "phase_stability_checks": len(checks),
        "consecutive_unchanged": checks[-1]["consecutive_unchanged"],
        "called_alleles": phase["called_alleles"],
        "changed_observable_phase_sites": phase["changed_observable_phase_sites"],
        "known_genotypes_changed": phase["known_genotypes_changed"],
        "imputation_enabled": False, "posterior_and_recombination_products_included": False,
        "confidence_policy": result["confidence_policy"], "elapsed_seconds": elapsed,
        "final_checkpoint": str(Path(checkpoints.contig_path(
            store.root, FINAL_PHASE_STAGE, result["contig"])).resolve()),
    }


def run_refinement(checkpoint_store, contigs, sample_ids, *, pedigree_payload,
                   output_dir, raw_gl_stage, raw_sites_stage, raw_gl_key="global_probs",
                   n_workers=None, config=None, genetic_maps=None):
    """Run canonical T11 sequentially by chromosome with the full CPU budget.

Only final phase is a release product. Work-in-progress messages and consecutive
phase checks live beside it as *.iterations.p5.b2, never as completed contigs.
"""
    if pedigree_payload is None:
        print("[T11] Waiting for the complete genome-wide T10 pedigree; no shard inference.")
        return None
    config = (config or model.FamilyRefinementConfig()).validated()
    if genetic_maps is not None:
        config = replace(config, recombination_rate=genetic_maps.default_rate_cm_per_mb / 1e8).validated()
    workers = runtime.available_cpu_count() if n_workers is None else int(n_workers)
    if not 1 <= workers <= runtime.available_cpu_count():
        raise ValueError("Stage11 CPU budget exceeds allocation")
    names = tuple(map(str, sample_ids)); contigs = tuple(map(str, contigs))
    if (tuple(map(str, pedigree_payload["ordered_sample_ids"])) != names or
            tuple(map(str, pedigree_payload["ordered_contigs"])) != contigs):
        raise ValueError("Stage11 axes differ from the complete T10 pedigree")
    relationships = pedigree_payload["tier_b_relationships"]
    source_files = []
    for source in dict.fromkeys((PAINTING_STAGE, raw_gl_stage, raw_sites_stage)):
        runtime.require_contig_checkpoints(checkpoint_store, source, contigs)
        for contig in contigs:
            path = Path(checkpoints.contig_path(checkpoint_store.root, source, contig)).resolve()
            stat = path.stat()
            source_files.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    identity = {
        "schema": conditioning.SCHEMA, "config": asdict(config),
        "phase_config": asdict(replace(CANONICAL_PHASE_CONFIG, recombination_rate=config.recombination_rate)),
        "sample_ids": names, "contigs": contigs, "source_files": source_files,
        "pedigree_sha256": conditioning.relationship_identity(relationships),
        "code": conditioning.refinement_code_identity(),
        "raw_gl_key": raw_gl_key,
        "recombination_maps": None if genetic_maps is None else {c: genetic_maps.identity(c) for c in contigs},
    }
    checkpoint_store.bind_stage_identity(FINAL_PHASE_STAGE, identity)
    output = Path(output_dir) / "phase_correction"
    output.mkdir(parents=True, exist_ok=True)
    if checkpoint_store.stage_complete(FINAL_PHASE_STAGE) and checkpoint_store.global_done(FINAL_PHASE_STAGE):
        runtime.require_contig_checkpoints(checkpoint_store, FINAL_PHASE_STAGE, contigs)
        saved = checkpoint_store.load_global(FINAL_PHASE_STAGE)
        if saved["identity"] != identity:
            raise ValueError("Stage11 final summary identity differs")
        conditioning._write_summary_table(output, FINAL_PHASE_STAGE, saved["summaries"])
        print("[T11] Resumed complete final phase output from its compact summary.")
        return saved["summaries"]
    summaries = []
    evidence_store = runtime.CheckpointStore(checkpoint_store.root, nthreads=workers)
    print(f"[T11] Stable final phase; one chromosome, {workers} Numba threads.", flush=True)
    with parallel.numba_thread_scope(workers):
        for contig in contigs:
            started = time.perf_counter()
            if checkpoint_store.contig_done(FINAL_PHASE_STAGE, contig):
                result = checkpoint_store.load_contig(FINAL_PHASE_STAGE, contig, nthreads=workers)
                if result["identity"]["family"]["external"] != identity:
                    raise ValueError("Stage11 final chromosome identity differs")
            else:
                t09 = checkpoint_store.load_contig(PAINTING_STAGE, contig, nthreads=workers)
                gl, pos, observed, gl_payload, sites_payload = pedigree_pipeline._load_raw_evidence(
                    evidence_store, contig, raw_gl_stage=raw_gl_stage,
                    raw_sites_stage=raw_sites_stage, raw_gl_key=raw_gl_key,
                    raw_sites_key="global_sites", raw_observed_mask_key="global_observed_mask")
                del gl_payload, sites_payload
                chromosome_map = None if genetic_maps is None else genetic_maps.for_contig(contig)
                result = refine_chromosome(
                    t09, gl, pos, observed, relationships, names, contig=contig, config=config,
                    work_path=Path(checkpoint_store.stage_dir(FINAL_PHASE_STAGE)) / f"{contig}.iterations.p5.b2",
                    identity=identity, checkpoint_threads=workers, chromosome_map=chromosome_map)
                checkpoints.write(checkpoints.contig_path(checkpoint_store.root, FINAL_PHASE_STAGE, contig),
                                  result, nthreads=workers)
                del t09, gl, pos, observed
            summaries.append(_summary(result, checkpoint_store, time.perf_counter()-started))
            print(f"[T11 {contig}] Stable final phase released; latent converged="
                  f"{result['source_posterior_converged']}", flush=True)
            del result
            gc.collect(); parallel.malloc_trim()
    runtime.require_contig_checkpoints(checkpoint_store, FINAL_PHASE_STAGE, contigs)
    checkpoint_store.save_global(FINAL_PHASE_STAGE, {"identity": identity, "summaries": summaries})
    conditioning._write_summary_table(output, FINAL_PHASE_STAGE, summaries)
    checkpoint_store.mark_stage_complete(FINAL_PHASE_STAGE)
    print(f"[COMPLETE] Final phase ready for recombination maps: {output / (FINAL_PHASE_STAGE + '.csv')}")
    return summaries
