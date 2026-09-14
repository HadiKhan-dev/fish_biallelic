"""workflows / astcal for the canonical reconstruction pipeline."""
from __future__ import annotations

import json


def configured_regions(default, *, template_regions=False):
    requested = os.environ.get("HAPLOTYPES_CONTIGS")
    if requested is None:
        return default
    names = json.loads(requested)
    if not names or len(names) != len(set(names)):
        raise ValueError("contigs must be a nonempty unique ordered list")
    return [dict(contig=str(name), **({'start': 0, 'end': 3000} if template_regions else {}))
            for name in names]


import os
import haplotype_reconstruction.assembly.pipeline as assembly_pipeline
import haplotype_reconstruction.core.environment as core_environment
import haplotype_reconstruction.core.genetic_map as core_genetic_map
import haplotype_reconstruction.core.haplotypes as core_haplotypes
import haplotype_reconstruction.core.numerics as core_numerics
import haplotype_reconstruction.core.parallel as core_parallel
import haplotype_reconstruction.core.runtime as core_runtime
import haplotype_reconstruction.core.variants as core_variants
import haplotype_reconstruction.discovery.blocks as discovery_blocks
import haplotype_reconstruction.discovery.search as discovery_search
import haplotype_reconstruction.pedigree.pipeline as pedigree_pipeline
import haplotype_reconstruction.recombination.model as module_recombination_model
import haplotype_reconstruction.recombination.pipeline as recombination_pipeline
import haplotype_reconstruction.refinement.pipeline as refinement_pipeline
import haplotype_reconstruction.refinement.model as refinement_model
import haplotype_reconstruction.workflows.design as workflows_design
import haplotype_reconstruction.workflows.reconstruction as workflows_reconstruction

CHECKPOINT_DIR = (
    os.environ.get("HAPLOTYPES_CHECKPOINT_DIR", "work/runs/astcal/checkpoints")
)


ASAC_METADATA_PATH = os.environ.get("HAPLOTYPES_METADATA", "work/data/fish_vcf_restriped/X_AsAc_metafile.xlsx")


ASAC_METADATA_SHEET = os.environ.get("HAPLOTYPES_METADATA_SHEET", "main_data")


def run():
    """Execute or resume the configured, chromosome-checkpointed workflow."""
    import os
    import sys
    from datetime import datetime

    # FORCE NUMPY/BLAS TO USE 1 THREAD PER PROCESS
    core_environment.force_single_threaded_numeric_libraries()

    # =============================================================================
    # DUAL LOGGING: Console + File
    # =============================================================================

    os.makedirs(os.environ.get("HAPLOTYPES_LOG_DIR", "work/logs"), exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(os.environ.get("HAPLOTYPES_LOG_DIR", "work/logs"), f"run_{run_timestamp}.log")
    sys.stdout = core_runtime.TeeOutput(log_path, sys.stdout)
    print(f"Logging to: {log_path}")
    print(f"Run started: {run_timestamp}")

    import numpy as np
    import pandas as pd
    import time
    import warnings
    import platform
    import gc
    from dataclasses import asdict
    from cyvcf2 import VCF

    np.seterr(divide='ignore', invalid='ignore')


    stage1_config = discovery_search.ReversibleCavitySearchConfig()
    stage1_config_record = asdict(stage1_config)
    stage1_identity_record = {
        "backend": discovery_blocks.STAGE1_BACKEND,
        "config": stage1_config_record,
    }


    from dataclasses import replace

    rate_maps = core_genetic_map.load_genetic_maps_from_environment()
    inference_recombination_rate = rate_maps.default_rate_cm_per_mb / 1e8
    inference_genetic_maps = rate_maps if rate_maps.maps else None

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    if platform.system() != "Windows":
        print(f"Main process ({os.getpid()}) niceness set to: {os.nice(0)}")

    n_processes = int(os.environ.get("BHD_NUM_PROCESSES", str(core_runtime.available_cpu_count())))
    # Reuse native code across four batches; trim between batches and retain
    # periodic process recycling to bound long-lived allocator fragmentation.
    WORKER_MAXTASKS = 4

    # Start forkserver before data loading
    _warmup_pool = core_parallel.NonDaemonicForkserverPool(1)
    _warmup_pool.terminate()
    _warmup_pool.join()
    del _warmup_pool
    print("Forkserver started (lightweight, pre-data).")
    print(f"Numba threading layer: {os.environ.get('NUMBA_THREADING_LAYER', 'not set')}")

    # =========================================================================
    # Configuration
    # =========================================================================
    vcf_path = os.environ.get("HAPLOTYPES_VCF", "work/data/fish_vcf_restriped/AsAc.AulStuGenome.biallelic.bcf.gz")

    regions_config = configured_regions([
        {"contig": "chr1"},  {"contig": "chr2"},  {"contig": "chr3"},
        {"contig": "chr4"},  {"contig": "chr5"},  {"contig": "chr6"},
        {"contig": "chr7"},  {"contig": "chr8"},  {"contig": "chr9"},
        {"contig": "chr10"}, {"contig": "chr11"}, {"contig": "chr12"},
        {"contig": "chr13"}, {"contig": "chr14"}, {"contig": "chr15"},
        {"contig": "chr16"}, {"contig": "chr17"}, {"contig": "chr18"},
        {"contig": "chr19"}, {"contig": "chr20"}, {"contig": "chr22"},
        {"contig": "chr23"},
    ], template_regions=False)

    output_dir = (
        os.environ.get("HAPLOTYPES_OUTPUT_DIR", "work/runs/astcal")
    )

    # =========================================================================
    # Checkpoint Infrastructure
    # =========================================================================
    checkpoint_store = core_runtime.CheckpointStore(
        CHECKPOINT_DIR, nthreads=n_processes, global_log_indent="    "
    )
    os.makedirs(output_dir, exist_ok=True)
    stage_complete = checkpoint_store.stage_complete
    mark_stage_complete = checkpoint_store.mark_stage_complete
    contig_done = checkpoint_store.contig_done
    save_contig = checkpoint_store.save_contig
    load_contig = checkpoint_store.load_contig
    save_global = checkpoint_store.save_global
    load_global = checkpoint_store.load_global

    region_keys = [r['contig'] for r in regions_config]

    # Get sample names from VCF header
    _vcf_tmp = VCF(vcf_path)
    sample_names = list(_vcf_tmp.samples)
    _vcf_tmp.close()
    n_samples = len(sample_names)
    print(f"VCF samples: {n_samples}")
    print(f"Regions: {len(region_keys)}")

    total_pipeline_start = time.time()

    # =========================================================================
    # STAGE R01: VCF Loading + Block Discovery + Global Probabilities
    # =========================================================================
    STAGE_R1 = "R00_founder_templates"
    checkpoint_store.bind_stage_identity(
        STAGE_R1, stage1_identity_record
    )

    if stage_complete(STAGE_R1):
        print(f"\n[RESUME] Skipping VCF loading + discovery (checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("STAGE R01: VCF Loading + Block Haplotype Discovery")
        print(f"{'='*60}")
        start = time.time()

        with discovery_blocks.BlockDiscoveryPool(n_processes) as block_pool:
            for r_name in region_keys:
                if contig_done(STAGE_R1, r_name):
                    print(f"  [RESUME] {r_name} already done")
                    continue
                print(f"\n  Processing {r_name}...")

                t0 = time.time()
                genomic_data = core_variants.cleanup_block_reads_list(
                    vcf_path, r_name,
                    use_snp_count=True, snps_per_block=200, snp_shift=200,
                    num_processes=n_processes
                )
                print(f"    [Loader] {len(genomic_data)} blocks in {time.time()-t0:.1f}s")

                global_sites, global_reads = (
                    core_variants.concatenate_unique_block_reads(genomic_data)
                )
                if global_sites is None:
                    print(f"    WARNING: No data for {r_name}, skipping")
                    continue

                # Preserve the observation event before releasing read counts.
                global_observed_mask = (
                    workflows_reconstruction.observed_call_mask_from_read_counts(
                        global_reads
                    )
                )

                # Keep cohort-frequency regularization inside block discovery;
                # linkage and assembly consume the raw genotype likelihoods.
                (site_priors, global_probs) = core_numerics.reads_to_probabilities(
                    global_reads,
                    use_hwe_prior=False,
                )
                avg_depth = np.mean(np.sum(global_reads, axis=-1))
                print(f"    Sites: {len(global_sites)}, Samples: {global_probs.shape[0]}, "
                      f"Depth: {avg_depth:.1f}x")
                del global_reads, site_priors

                t0 = time.time()
                block_results = discovery_blocks.generate_all_block_haplotypes(
                    genomic_data,
                    num_processes=n_processes,
                    block_pool=block_pool,
                    discovery_config=stage1_config,
                )
                valid_blocks = [b for b in block_results if len(b.positions) > 0]
                block_results = core_haplotypes.BlockResults(valid_blocks)

                hap_counts = [len(b.haplotypes) for b in valid_blocks]
                print(f"    [Discovery] {len(valid_blocks)} blocks, haps/block: "
                      f"min={min(hap_counts)}, max={max(hap_counts)}, "
                      f"mean={np.mean(hap_counts):.1f} in {time.time()-t0:.1f}s")

                save_contig(STAGE_R1, r_name, {
                    'global_probs': global_probs,
                    'global_sites': global_sites,
                    'global_observed_mask': global_observed_mask,
                    'observed_call_mask_mode': (
                        workflows_reconstruction.EXACT_OBSERVED_MASK_MODE
                    ),
                    'genotype_evidence_mode': (
                        workflows_reconstruction.SUPPORTED_GENOTYPE_EVIDENCE_MODE
                    ),
                    'block_results': block_results,
                    'avg_depth': avg_depth,
                    'stage1_backend': discovery_blocks.STAGE1_BACKEND,
                    'stage1_config': stage1_config_record,
                })
                del genomic_data, block_results, global_probs, global_sites
                del global_observed_mask
                gc.collect()

        save_global(STAGE_R1, {
            'sample_ids': sample_names,
            'contigs': region_keys,
            'genotype_evidence_mode': (
                workflows_reconstruction.SUPPORTED_GENOTYPE_EVIDENCE_MODE
            ),
            'observed_call_mask_mode': (
                workflows_reconstruction.EXACT_OBSERVED_MASK_MODE
            ),
            'stage1_backend': discovery_blocks.STAGE1_BACKEND,
            'stage1_config': stage1_config_record,
        })
        print(f"\nVCF loading + discovery complete in {time.time()-start:.1f}s")
        mark_stage_complete(STAGE_R1)

    # =========================================================================
    # STAGE 2: COMPONENT ASSEMBLY AND PAINTING
    # =========================================================================
    stage2_config = workflows_reconstruction.ReconstructionConfig(
        release_config=assembly_pipeline.AssemblyConfig(
            num_processes=n_processes,
            maxtasksperchild=WORKER_MAXTASKS,
            recombination_rate=inference_recombination_rate,
        ),
        paint_cores=n_processes,
        paint_recombination_rate=inference_recombination_rate,
    )
    print()
    print("=" * 60)
    print("STAGE 2: R01 -> COMPONENT T09")
    print(f"{'='*60}")
    print(
        "  Sequential contigs; release and painting are non-overlapping "
        f"phases with a {n_processes}-core ceiling"
    )
    summaries = workflows_reconstruction.run_reconstruction(
        checkpoint_store,
        region_keys,
        sample_names,
        stage1_identity=stage1_identity_record,
        config=stage2_config,
        genetic_maps=inference_genetic_maps,
        source_stage=STAGE_R1,
    )
    for summary in summaries:
        status = "resumed" if summary.resumed else "completed"
        print(
            f"  [{status}] {summary.contig}: "
            f"components={summary.component_count}, "
            "evidence-eligible component-sample pairs="
            f"{summary.evidence_eligible_component_sample_pairs}/"
            f"{summary.total_component_sample_pairs}, "
            f"observation mask={summary.observed_mask_mode}"
        )

    # Preserve the established AsAc design policy; cohort labels do not
    # establish individual parentage and G0 reference fish are excluded.
    asac_metadata = pd.read_excel(ASAC_METADATA_PATH, sheet_name=ASAC_METADATA_SHEET)
    parent_eligibility = workflows_design.build_asac_parent_eligibility(
        asac_metadata, sample_names, require_opposite_sex_pair=True,
    )
    _stage10_summaries, stage10_payload = pedigree_pipeline.run_pedigree(
        checkpoint_store, region_keys, sample_names,
        output_dir=output_dir, n_workers=n_processes,
        raw_gl_stage=STAGE_R1, raw_sites_stage=STAGE_R1,
        parent_eligibility=parent_eligibility,
        genetic_maps=inference_genetic_maps, recombination_rate=inference_recombination_rate,
    )
    print("STAGE 11: pedigree-conditioned refinement and canonical final phase polishing")
    refinement_pipeline.run_refinement(
        checkpoint_store, region_keys, sample_names,
        pedigree_payload=stage10_payload, output_dir=output_dir,
        raw_gl_stage=STAGE_R1, raw_sites_stage=STAGE_R1,
        n_workers=n_processes,
        genetic_maps=inference_genetic_maps,
        config=refinement_model.FamilyRefinementConfig(recombination_rate=inference_recombination_rate),
    )
    recombination_pipeline.run_recombination(
        checkpoint_store, region_keys, sample_names,
        pedigree_payload=stage10_payload, output_dir=output_dir,
        n_workers=n_processes,
        genetic_maps=inference_genetic_maps,
        config=module_recombination_model.RecombinationMapConfig(recombination_rate=inference_recombination_rate),
    )


if __name__ == "__main__":
    run()
