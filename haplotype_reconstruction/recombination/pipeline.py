"""recombination / pipeline for the canonical reconstruction pipeline."""
from __future__ import annotations
from haplotype_reconstruction import PACKAGE_ROOT

from dataclasses import asdict, replace
import gc
import hashlib
import os
from pathlib import Path
import warnings

import pandas as pd
import haplotype_reconstruction.recombination.model as module_recombination_model

RECOMBINATION_STAGE = "12_recombination"


def resolve_shared_family_evidence(value=None):
    """Explicit API/CLI setting wins over the default-on pipeline environment."""
    if value is not None:
        if not isinstance(value, bool):
            raise ValueError("shared_family_evidence must be boolean or None")
        return value
    setting = os.environ.get("BHD_RECOMBINATION_SHARED_FAMILY", "1").strip().lower()
    if setting in ("1", "true", "yes", "on"):
        return True
    if setting in ("0", "false", "no", "off"):
        return False
    raise ValueError("BHD_RECOMBINATION_SHARED_FAMILY must be 1/0 or true/false")


def _csv(frame, path):
    temporary = path.with_name("." + path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_chromosome_outputs(result, output):
    contig = result["contig"]
    curve = result["map"]
    output.mkdir(parents=True, exist_ok=True)
    bins = curve["edges_bp"]
    frame = {"left_bp": bins[:-1], "right_bp": bins[1:]}
    frame.update({key: value for key, value in curve.items()
                  if key not in ("edges_bp", "cumulative_observed_cM")})
    frame["cumulative_observed_cM_at_right"] = curve["cumulative_observed_cM"][1:]
    _csv(pd.DataFrame(frame),output/f"{contig}.map.csv")
    events = []
    coverage = []
    called_coverage = []
    meioses = []
    for edge, (parent, child, slot) in enumerate(zip(result["edge_parent"], result["edge_child"], result["edge_child_slot"])):
        labels = {
            "parent": result["sample_ids"][parent],
            "child": result["sample_ids"][child],
            "parent_slot": int(slot)
        }
        for left, right, ps, cs, component in result["crossover_intervals"][edge]:
            events.append({**labels, "left_bp": int(left), "right_bp": int(right),
                "left_origin_probability": ps, "right_origin_probability": cs,
                "joint_endpoint_probability_lower_bound": max(0., ps + cs - 1),
                "component_id": int(component)})
        for left, right, component in result["informative_spans"][edge]:
            coverage.append(
                {**labels, "left_bp": int(left), "right_bp": int(right), "component_id": int(component)}
            )
        for left, right, component in result["callable_spans"][edge]:
            called_coverage.append(
                {**labels, "left_bp": int(left), "right_bp": int(right), "component_id": int(component)}
            )
        counts = result["marker_counts"][edge]
        meioses.append({**labels, "informative_markers": int(counts[0]),
            "supported_origin_markers": int(counts[1]), "orientation_artifact_uncertain_markers": int(counts[2]),
            "called_crossovers": len(result["crossover_intervals"][edge]),
            "expected_crossovers": float(result["expected_crossovers_by_edge_bin"][edge].sum()),
            "exposure_meiosis_bp": float(result["exposure_by_edge_bin"][edge].sum())})
    _csv(pd.DataFrame(meioses),output/f"{contig}.meioses.csv")
    _csv(pd.DataFrame(events, columns=("parent", "child", "parent_slot", "left_bp", "right_bp",
        "left_origin_probability", "right_origin_probability", "joint_endpoint_probability_lower_bound",
        "component_id")),output/f"{contig}.crossovers.csv")
    _csv(pd.DataFrame(coverage, columns=("parent", "child", "parent_slot", "left_bp", "right_bp", "component_id")),
         output/f"{contig}.coverage.csv")
    _csv(pd.DataFrame(called_coverage, columns=("parent", "child", "parent_slot", "left_bp", "right_bp", "component_id")),
         output/f"{contig}.called_coverage.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    x = (bins[:-1] + bins[1:]) / 2e6
    axes[0].plot(x, curve["rate_cM_per_Mb"], color="#24557b", label="Posterior mean")
    axes[0].plot(x, curve["called_rate_cM_per_Mb"], color="#777777", linestyle="--", label="High-confidence calls")
    axes[0].legend()
    axes[0].set_ylabel("Conditional rate (cM/Mb)")
    axes[1].stairs(curve["effective_meioses"], bins / 1e6, color="#3d7e55")
    axes[1].set_ylabel("Effective meioses")
    axes[1].set_xlabel("Position (Mb)")
    axes[0].set_title(f"{contig}: missing-aware recombination and exposure")
    figure.tight_layout()
    path=output/f"{contig}.png"
    temporary=output/f".{contig}.png.tmp"
    figure.savefig(temporary, format="png", dpi=150)
    plt.close(figure)
    temporary.replace(path)


def run_recombination(checkpoint_store, contigs, sample_ids, *, pedigree_payload, output_dir,
                        n_workers=None, config=module_recombination_model.RecombinationMapConfig(), genetic_maps=None,
                        shared_family_evidence=None, shared_config=module_recombination_model.SharedOrientationConfig()):
    if pedigree_payload is None:
        print("[T12] Awaiting the complete T10 pedigree and final phase products.")
        return None
    shared_family_evidence = resolve_shared_family_evidence(shared_family_evidence)
    if shared_family_evidence:
        shared_config = shared_config.validated()
    if genetic_maps is not None:
        config = replace(config, recombination_rate=genetic_maps.default_rate_cm_per_mb / 1e8)
    config = config.validated()
    workers = core_runtime.available_cpu_count() if n_workers is None else int(n_workers)
    if not 1 <= workers <= core_runtime.available_cpu_count():
        raise ValueError("Stage12 CPU budget exceeds allocation")
    names = tuple(map(str, sample_ids))
    contigs = tuple(map(str, contigs))
    if (tuple(map(str, pedigree_payload["ordered_sample_ids"])) != names or
            tuple(map(str, pedigree_payload["ordered_contigs"])) != contigs):
        raise ValueError("Stage12 axes differ from the complete T10 pedigree")
    if not checkpoint_store.stage_complete(refinement_pipeline.FINAL_PHASE_STAGE):
        raise ValueError("Stage12 requires the completed canonical final phase stage")
    core_runtime.require_contig_checkpoints(checkpoint_store, refinement_pipeline.FINAL_PHASE_STAGE, contigs)
    relationships = pedigree_payload["tier_b_relationships"]
    pedigree_hash = refinement_conditioning.relationship_identity(relationships)
    files = []
    for contig in contigs:
        path = Path(core_checkpoints.contig_path(checkpoint_store.root, refinement_pipeline.FINAL_PHASE_STAGE, contig)).resolve()
        stat = path.stat()
        files.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    identity = {"schema": "missing-aware-recombination-map-v1", "config": asdict(config),
        "shared_family_evidence": shared_family_evidence,
        "shared_orientation_config": asdict(shared_config) if shared_family_evidence else None,
        "sample_ids": names, "contigs": contigs, "pedigree_sha256": pedigree_hash, "final_phase_files": files,
        "code": {name: hashlib.sha256((PACKAGE_ROOT / name).read_bytes()).hexdigest()
                for name in ('recombination/pipeline.py', 'recombination/model.py',
                             'recombination/intervals.py', 'recombination/orientation_prior.py',
                             'refinement/model.py')}}
    if shared_family_evidence:
        identity["code"]['recombination/model.py'] = hashlib.sha256(
            (PACKAGE_ROOT / 'recombination/model.py').read_bytes()).hexdigest()
    if genetic_maps is not None and any(genetic_maps.for_contig(c).has_map for c in contigs):
        identity["input_recombination_maps"] = {c: genetic_maps.identity(c) for c in contigs}
        identity["code"]['core/genetic_map.py'] = hashlib.sha256((PACKAGE_ROOT / 'core/genetic_map.py').read_bytes()).hexdigest()
    checkpoint_store.bind_stage_identity(RECOMBINATION_STAGE, identity)
    output = Path(output_dir) / "recombination_map"
    output.mkdir(parents=True, exist_ok=True)
    if checkpoint_store.stage_complete(RECOMBINATION_STAGE) and checkpoint_store.global_done(RECOMBINATION_STAGE):
        core_runtime.require_contig_checkpoints(checkpoint_store, RECOMBINATION_STAGE, contigs)
        saved = checkpoint_store.load_global(RECOMBINATION_STAGE)
        if saved["identity"] != identity:
            raise ValueError("Stage12 completed summary identity differs")
        _csv(pd.DataFrame(saved["summaries"]), output / "summary.csv")
        print("[T12] Resumed complete missing-aware recombination maps.")
        return saved["summaries"]
    summaries = []
    print(f"STAGE 12: missing-aware recombination maps; one chromosome, {workers} Numba threads",flush=True)
    print(f"[T12] Shared-family orientation evidence: {'on' if shared_family_evidence else 'off'}",flush=True)
    with core_parallel.numba_thread_scope(workers):
        for contig in contigs:
            if checkpoint_store.contig_done(RECOMBINATION_STAGE, contig):
                result = checkpoint_store.load_contig(RECOMBINATION_STAGE, contig, nthreads=workers)
                if result["identity"] != identity:
                    raise ValueError("Stage12 chromosome identity differs")
            else:
                phase = checkpoint_store.load_contig(
                    refinement_pipeline.FINAL_PHASE_STAGE,
                    contig,
                    nthreads=workers
                )
                if (tuple(phase["sample_ids"]) != names or phase["contig"] != contig or
                        phase["identity"]["family"]["pedigree_sha256"] != pedigree_hash):
                    raise ValueError("final phase and recombination pedigree/axes differ")
                chromosome_map = None if genetic_maps is None else genetic_maps.for_contig(contig)
                if shared_family_evidence:
                    result = module_recombination_model.build_shared_family_map(phase, relationships, config=config,
                        shared_config=shared_config, chromosome_map=chromosome_map)
                else:
                    result = module_recombination_model.build_missing_aware_map(phase, relationships, config=config,
                        chromosome_map=chromosome_map)
                result["summary"]["shared_family_evidence"] = shared_family_evidence
                if shared_family_evidence:
                    diagnostics = result["shared_orientation"]
                    result["summary"].update(
                        shared_orientation_converged=diagnostics["converged_on_screened_candidates"],
                        shared_orientation_corrected_individuals=diagnostics["individuals_with_corrections"],
                        shared_orientation_accepted_moves=len(diagnostics["accepted_moves"]))
                result["identity"] = identity
                core_checkpoints.write(core_checkpoints.contig_path(checkpoint_store.root, RECOMBINATION_STAGE, contig),
                                    result, nthreads=workers)
                del phase
            if shared_family_evidence and not result["summary"]["shared_orientation_converged"]:
                warnings.warn(f"{contig}: shared-family orientation fit reached its sweep limit; "
                              "map remains conditional on the selected, nonconverged shared paths.",
                              RuntimeWarning)
            write_chromosome_outputs(result, output)
            summaries.append(result["summary"])
            print(f"[T12 {contig}] {result['summary']}",flush=True)
            del result
            gc.collect()
    checkpoint_store.save_global(RECOMBINATION_STAGE, {"identity": identity, "summaries": summaries})
    _csv(pd.DataFrame(summaries), output / "summary.csv")
    checkpoint_store.mark_stage_complete(RECOMBINATION_STAGE)
    print(f"[COMPLETE] Missing-aware pipeline through recombination maps: {output}")
    return summaries


def run_from_checkpoints(root, output_dir, *, n_workers=None, config=module_recombination_model.RecombinationMapConfig(), genetic_maps=None,
                         shared_family_evidence=None, shared_config=module_recombination_model.SharedOrientationConfig()):
    """Standalone entrypoint uses the exact accepted T10 pedigree, never truth."""

    store = core_runtime.CheckpointStore(root)
    pedigree = store.load_global(pedigree_pipeline.PEDIGREE_STAGE)
    return run_recombination(store, pedigree["ordered_contigs"], pedigree["ordered_sample_ids"],
        pedigree_payload=pedigree, output_dir=output_dir, n_workers=n_workers, config=config, genetic_maps=genetic_maps,
        shared_family_evidence=shared_family_evidence, shared_config=shared_config)

import haplotype_reconstruction.core.checkpoints as core_checkpoints
import haplotype_reconstruction.core.parallel as core_parallel
import haplotype_reconstruction.core.runtime as core_runtime
import haplotype_reconstruction.pedigree.pipeline as pedigree_pipeline
import haplotype_reconstruction.refinement.conditioning as refinement_conditioning
import haplotype_reconstruction.refinement.pipeline as refinement_pipeline
