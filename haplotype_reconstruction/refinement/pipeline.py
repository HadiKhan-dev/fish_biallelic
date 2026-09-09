"""refinement / pipeline for the canonical reconstruction pipeline."""
from __future__ import annotations
from haplotype_reconstruction import PACKAGE_ROOT

from dataclasses import asdict, replace
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import gc
import haplotype_reconstruction.refinement.polish as refinement_polish

CANONICAL_PHASE_CONFIG=refinement_polish.PhasePolishConfig(copy_error=.01,adaptive_window_sites=100,minimum_biological_gain=0.)


FINAL_PHASE_STAGE = "11_phase_correction"


def finalize_stage11_product(product, t09, *, checkpoint_path=None, checkpoint_threads=1,
                             genotype_likelihoods=None, relationships=None, messages=None,
                             phase_config=None, chromosome_map=None):
    """Return coherent allele/painting views; never attach stale confidence.

For a converged source, the typed Stage11 and T09 products suffice. An
unconverged source additionally needs its saved messages, original GLs and
fixed pedigree for consecutive phase-stability assessment. It does not release
unconverged posterior probabilities or recombination estimates.
"""
    if tuple(product.sample_ids)!=tuple(t09.sample_ids):
        raise ValueError('Stage11 and painting sample axes differ')
    if phase_config is None:
        phase_config = replace(CANONICAL_PHASE_CONFIG,
            recombination_rate=product.identity['config']['recombination_rate'])
    if chromosome_map is not None:
        phase_config = replace(phase_config, recombination_rate=chromosome_map.fallback_rate_per_bp)
    source_map = product.identity.get('recombination_map')
    supplied_map = None if chromosome_map is None else chromosome_map.identity()
    if source_map != supplied_map:
        raise ValueError('final phase requires the same chromosome recombination map as its source family fit')
    if chromosome_map is not None and chromosome_map.contig != str(product.contig):
        raise ValueError('final phase chromosome map name differs from input contig')
    if (product.identity['t09_release']!=t09.release_identity.record() or
            product.identity['t09_painting']!=t09.painting_product_identity.record()):
        raise ValueError('family fit belongs to another painting')
    files=('refinement/pipeline.py', 'refinement/polish.py', 'refinement/conditioning.py', 'core/genetic_map.py')
    identity={'schema':'stage11-final-phase-only-v1','family':product.identity,
              'phase_config':asdict(phase_config),
              'code':{f:hashlib.sha256((PACKAGE_ROOT / f).read_bytes()).hexdigest() for f in files}}
    path=Path(checkpoint_path) if checkpoint_path is not None else None
    if path is not None and path.exists():
        cached=core_checkpoints.read(str(path),nthreads=checkpoint_threads)
        if cached['identity']!=identity:raise ValueError('final phase checkpoint inputs/configuration differ')
        return cached
    scaffold=refinement_conditioning.prepare_phase_scaffold(t09,product.positions)
    if not np.array_equal(scaffold.reference_alleles,product.reference_alleles):
        raise ValueError('source family and frozen painting alleles differ')
    checks=();latent_converged=bool(product.summary['converged'])
    if latent_converged:
        phase=refinement_polish.finish_family_phase(product.reference_alleles,product,product.positions,scaffold.phase_bins,
            product.component_ids,product.edge_parent,product.edge_child,product.edge_child_slot,config=phase_config,
            chromosome_map=chromosome_map)
    else:
        if genotype_likelihoods is None or relationships is None or messages is None:
            raise ValueError('phase-only assessment requires the unconverged fit messages and original evidence')
        family=SimpleNamespace(converged=False,messages=messages,
            ordered_genotype_probability=product.ordered_genotype_probability,
            phase_map=product.phase_map,inferred_phase_map=product.inferred_phase_map,phase_flip=product.phase_flip)
        assessed=refinement_polish.assess_phase_release(genotype_likelihoods,product.raw_observed_mask,product.reference_alleles,
            product.positions,scaffold.phase_bins,product.component_ids,relationships,product.sample_ids,
            product.edge_parent,product.edge_child,product.edge_child_slot,family,
            phase_config=phase_config,family_config=refinement_model.FamilyRefinementConfig(**product.identity['config']),
            chromosome_map=chromosome_map)
        if not assessed.phase_stable:raise RuntimeError('final called phase did not stabilize; source checkpoints are unchanged')
        phase,checks,latent_converged=assessed.phase,assessed.checks,assessed.latent_converged
    paintings=refinement_conditioning.paint_final_phase(t09,product.positions,phase.phase_map)
    result={'identity':identity,'contig':product.contig,'sample_ids':product.sample_ids,
            'positions':product.positions,'component_ids':product.component_ids,
            'phase':phase,'corrected_component_paintings':paintings,'phase_stable':True,
            'source_posterior_converged':bool(product.summary['converged']),
            'continuation_posterior_converged':latent_converged,'phase_stability_checks':checks,
            'confidence_policy':'conditional phase point path; source family posterior is separate and is not transferred to this path',
            'posterior_and_recombination_products_included':False}
    if path is not None:core_checkpoints.write(str(path),result,nthreads=checkpoint_threads)
    return result


def _finalizer_code_identity():
    files = ('refinement/pipeline.py', 'refinement/polish.py', 'refinement/conditioning.py', 'core/genetic_map.py')
    return {name: hashlib.sha256((PACKAGE_ROOT / name).read_bytes()).hexdigest()
            for name in files}


def _source_path(store, stage, contig, converged):
    if converged:
        return Path(core_checkpoints.contig_path(store.root, stage, contig))
    # Existing family-stage readers interpret a contig checkpoint as a
    # converged fit. Keep unconverged diagnostics out of that namespace.
    return Path(store.stage_dir(stage)) / f"{contig}.nonconverged.p5.b2"


def _summary(result, store, source_stage):
    phase = result["phase"].summary
    contig = result["contig"]
    return {
        "contig": contig, "samples": len(result["sample_ids"]),
        "sites": len(result["positions"]), "phase_stable": result["phase_stable"],
        "source_posterior_converged": result["source_posterior_converged"],
        "continuation_posterior_converged": result["continuation_posterior_converged"],
        "phase_stability_checks": len(result["phase_stability_checks"]),
        "called_alleles": phase["called_alleles"],
        "changed_observable_phase_sites": phase["changed_observable_phase_sites"],
        "known_genotypes_changed": phase["known_genotypes_changed"],
        "imputation_enabled": phase["imputation_enabled"],
        "posterior_and_recombination_products_included": False,
        "confidence_policy": result["confidence_policy"],
        "final_checkpoint": str(Path(core_checkpoints.contig_path(
            store.root, FINAL_PHASE_STAGE, contig)).resolve()),
        "source_family_checkpoint": str(_source_path(
            store, source_stage, contig, result["source_posterior_converged"]).resolve()),
    }


def run_refinement(checkpoint_store, contigs, sample_ids, *, pedigree_payload,
                        output_dir, raw_gl_stage, raw_sites_stage, raw_gl_key="global_probs",
                        n_workers=None, config=None, genetic_maps=None):
    """Publish the validated phase-only product, reusing completed family fits.

Chromosomes run sequentially with the full supplied Numba budget. Existing
family iteration/output checkpoints are preserved; final chromosomes are
atomic and a compact final summary permits a no-array-load completed resume.
Optional family imputation remains in the separate source family product.
"""
    if pedigree_payload is None:
        print("[T11] Waiting for the complete genome-wide T10 pedigree; no shard inference.")
        return None
    config = (config or refinement_conditioning.config_from_environment()).validated()
    if genetic_maps is not None:
        config = replace(config, recombination_rate=genetic_maps.default_rate_cm_per_mb / 1e8).validated()
    workers = core_runtime.available_cpu_count() if n_workers is None else int(n_workers)
    if not 1 <= workers <= core_runtime.available_cpu_count():
        raise ValueError("Stage11 CPU budget exceeds allocation")
    names = tuple(map(str, sample_ids)); contigs = tuple(map(str, contigs))
    if (tuple(map(str, pedigree_payload["ordered_sample_ids"])) != names or
            tuple(map(str, pedigree_payload["ordered_contigs"])) != contigs):
        raise ValueError("Stage11 axes differ from the complete T10 pedigree")
    relationships = pedigree_payload["tier_b_relationships"]
    source_stage = "11_family_imputation" if config.impute_missing else "11_family_refinement"
    source_files = []
    for source in dict.fromkeys((workflows_reconstruction.PAINTING_STAGE, raw_gl_stage, raw_sites_stage)):
        core_runtime.require_contig_checkpoints(checkpoint_store, source, contigs)
        for contig in contigs:
            path = Path(core_checkpoints.contig_path(checkpoint_store.root, source, contig)).resolve()
            stat = path.stat()
            source_files.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    # Exactly the existing family-stage identity: finalizer changes must not
    # cause an expensive family refit or discard upstream assembly checkpoints.
    source_identity = {"schema": refinement_conditioning.SCHEMA, "config": asdict(config),
        "sample_ids": names, "contigs": contigs,
        "pedigree_sha256": refinement_conditioning.relationship_identity(relationships),
        "code": refinement_conditioning.refinement_code_identity(), "source_files": source_files}
    if genetic_maps is not None:
        source_identity['recombination_maps'] = {name: genetic_maps.identity(name) for name in contigs}
    checkpoint_store.bind_stage_identity(source_stage, source_identity)
    if checkpoint_store.stage_complete(source_stage):
        core_runtime.require_contig_checkpoints(checkpoint_store, source_stage, contigs)
    phase_code = _finalizer_code_identity()
    phase_settings = replace(CANONICAL_PHASE_CONFIG, recombination_rate=config.recombination_rate)
    phase_config = asdict(phase_settings)
    identity = {"schema": "stage11-canonical-final-phase-v1", "family": source_identity,
        "phase_config": phase_config, "finalizer_code": phase_code,
        "driver_code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    checkpoint_store.bind_stage_identity(FINAL_PHASE_STAGE, identity)
    output = Path(output_dir) / "phase_correction"
    output.mkdir(parents=True, exist_ok=True)
    if checkpoint_store.stage_complete(FINAL_PHASE_STAGE) and checkpoint_store.global_done(FINAL_PHASE_STAGE):
        core_runtime.require_contig_checkpoints(checkpoint_store, FINAL_PHASE_STAGE, contigs)
        saved = checkpoint_store.load_global(FINAL_PHASE_STAGE)
        if saved["identity"] != identity:
            raise ValueError("Stage11 final summary identity differs")
        refinement_conditioning._write_summary_table(output, FINAL_PHASE_STAGE, saved["summaries"])
        print("[T11] Resumed complete final phase output from its compact summary.")
        return saved["summaries"]
    summaries = []
    with core_parallel.numba_thread_scope(workers):
        for contig in contigs:
            chromosome_map = None if genetic_maps is None else genetic_maps.for_contig(contig)
            contig_phase_settings = phase_settings if chromosome_map is None else replace(
                phase_settings, recombination_rate=chromosome_map.fallback_rate_per_bp)
            contig_phase_config = asdict(contig_phase_settings)
            if checkpoint_store.contig_done(FINAL_PHASE_STAGE, contig):
                result = checkpoint_store.load_contig(FINAL_PHASE_STAGE, contig, nthreads=workers)
                saved = result["identity"]
                if (saved["family"]["external"] != source_identity or
                        saved["phase_config"] != contig_phase_config or saved["code"] != phase_code):
                    raise ValueError("Stage11 final chromosome identity differs")
            else:
                t09 = checkpoint_store.load_contig(workflows_reconstruction.PAINTING_STAGE, contig, nthreads=workers)
                work = Path(checkpoint_store.stage_dir(source_stage)) / f"{contig}.iterations.p5.b2"
                gl = pos = observed = gl_payload = sites_payload = messages = None
                diagnostic = output / f"{contig}.nonconverged.p5.b2"
                unconverged = _source_path(checkpoint_store, source_stage, contig, False)
                if checkpoint_store.contig_done(source_stage, contig):
                    product = checkpoint_store.load_contig(source_stage, contig, nthreads=workers)
                elif unconverged.is_file():
                    product = core_checkpoints.read(str(unconverged), nthreads=workers)
                elif diagnostic.is_file():
                    # The earlier runner saved unconverged fits here before
                    # stopping. Reuse that fit without overwriting its record.
                    product = core_checkpoints.read(str(diagnostic), nthreads=workers)
                else:
                    gl, pos, observed, gl_payload, sites_payload = pedigree_pipeline._load_raw_evidence(
                        checkpoint_store, contig, raw_gl_stage=raw_gl_stage,
                        raw_sites_stage=raw_sites_stage, raw_gl_key=raw_gl_key,
                        raw_sites_key="global_sites", raw_observed_mask_key="global_observed_mask")
                    product = refinement_conditioning.refine_t09_chromosome(t09, gl, pos, observed, relationships,
                        names, contig=contig, config=config, work_path=work,
                        identity=source_identity, checkpoint_threads=None,
                        chromosome_map=chromosome_map)
                if product.identity["external"] != source_identity:
                    raise ValueError("Stage11 source family identity differs")
                source_path = _source_path(checkpoint_store, source_stage, contig,
                                           product.summary["converged"])
                if not source_path.is_file():
                    core_checkpoints.write(str(source_path), product, nthreads=workers)
                if not product.summary["converged"]:
                    if gl is None:
                        gl, pos, observed, gl_payload, sites_payload = pedigree_pipeline._load_raw_evidence(
                            checkpoint_store, contig, raw_gl_stage=raw_gl_stage,
                            raw_sites_stage=raw_sites_stage, raw_gl_key=raw_gl_key,
                            raw_sites_key="global_sites", raw_observed_mask_key="global_observed_mask")
                    saved = core_checkpoints.read(str(work), nthreads=workers)
                    if saved["identity"] != product.identity:
                        raise ValueError("Stage11 phase assessment messages belong to another family fit")
                    messages = saved["messages"]
                    del saved
                result = finalize_stage11_product(product, t09,
                    checkpoint_path=core_checkpoints.contig_path(checkpoint_store.root, FINAL_PHASE_STAGE, contig),
                    checkpoint_threads=workers, genotype_likelihoods=gl,
                    relationships=relationships, messages=messages,
                    phase_config=contig_phase_settings, chromosome_map=chromosome_map)
                del t09, product, gl, pos, observed, gl_payload, sites_payload, messages
            summaries.append(_summary(result, checkpoint_store, source_stage))
            print(f"[T11 {contig}] Final phase released; source posterior converged="
                  f"{result['source_posterior_converged']}; phase stable={result['phase_stable']}", flush=True)
            del result
            gc.collect()
    core_runtime.require_contig_checkpoints(checkpoint_store, FINAL_PHASE_STAGE, contigs)
    checkpoint_store.save_global(FINAL_PHASE_STAGE, {"identity": identity, "summaries": summaries})
    refinement_conditioning._write_summary_table(output, FINAL_PHASE_STAGE, summaries)
    checkpoint_store.mark_stage_complete(FINAL_PHASE_STAGE)
    print(f"[COMPLETE] Final phase ready for recombination-map estimation: "
          f"{output / (FINAL_PHASE_STAGE + '.csv')}")
    return summaries

import haplotype_reconstruction.core.checkpoints as core_checkpoints
import haplotype_reconstruction.core.parallel as core_parallel
import haplotype_reconstruction.core.runtime as core_runtime
import haplotype_reconstruction.pedigree.pipeline as pedigree_pipeline
import haplotype_reconstruction.refinement.conditioning as refinement_conditioning
import haplotype_reconstruction.refinement.model as refinement_model
import haplotype_reconstruction.workflows.reconstruction as workflows_reconstruction
