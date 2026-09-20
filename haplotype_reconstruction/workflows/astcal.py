"""AstCal cross driver from variant input through final phase and maps."""
from __future__ import annotations

import os

from ..core.environment import configured_regions
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
import haplotype_reconstruction.workflows.design as workflows_design
import haplotype_reconstruction.workflows.reconstruction as workflows_reconstruction
from .downstream import run_downstream

CHECKPOINT_DIR = (
    os.environ.get("HAPLOTYPES_CHECKPOINT_DIR", "work/runs/astcal/checkpoints")
)


ASAC_METADATA_PATH = os.environ.get(
    "HAPLOTYPES_METADATA",
    "work/data/fish_vcf_restriped/X_AsAc_metafile.xlsx"
)


ASAC_METADATA_SHEET = os.environ.get("HAPLOTYPES_METADATA_SHEET", "main_data")


@core_runtime.logged_workflow("astcal")
def run():
    """Execute or resume the configured, chromosome-checkpointed workflow."""
    import os

    # FORCE NUMPY/BLAS TO USE 1 THREAD PER PROCESS
    core_environment.force_single_threaded_numeric_libraries()

    # =============================================================================
    # DUAL LOGGING: Console + File
    # =============================================================================



    import numpy as np
    import pandas as pd
    import time
    import platform
    import gc
    from dataclasses import asdict
    from cyvcf2 import VCF

    np.seterr(divide='ignore', invalid='ignore')


    discovery_config = discovery_search.ReversibleCavitySearchConfig()
    discovery_config_record = asdict(discovery_config)
    discovery_identity_record = {
        "backend": discovery_blocks.DISCOVERY_BACKEND,
        "config": discovery_config_record,
    }



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
    vcf_path = os.environ.get(
        "HAPLOTYPES_VCF",
        "work/data/fish_vcf_restriped/AsAc.AulStuGenome.biallelic.bcf.gz"
    )

    regions_config = configured_regions([
        {"contig": "chr1"}, {"contig": "chr2"}, {"contig": "chr3"},
        {"contig": "chr4"}, {"contig": "chr5"}, {"contig": "chr6"},
        {"contig": "chr7"}, {"contig": "chr8"}, {"contig": "chr9"},
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
    # STAGE block discovery: VCF Loading + Block Discovery + Global Probabilities
    # =========================================================================
    DISCOVERY_STAGE = "block_discovery"
    checkpoint_store.bind_stage_identity(
        DISCOVERY_STAGE, discovery_identity_record
    )

    if stage_complete(DISCOVERY_STAGE):
        print(f"\n[RESUME] Skipping VCF loading + discovery (checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("STAGE block discovery: VCF Loading + Block Haplotype Discovery")
        print(f"{'='*60}")
        start = time.time()

        with discovery_blocks.BlockDiscoveryPool(n_processes) as block_pool:
            for r_name in region_keys:
                if contig_done(DISCOVERY_STAGE, r_name):
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
                    discovery_config=discovery_config,
                )
                valid_blocks = [b for b in block_results if len(b.positions) > 0]
                block_results = core_haplotypes.BlockResults(valid_blocks)

                hap_counts = [len(b.haplotypes) for b in valid_blocks]
                print(f"    [Discovery] {len(valid_blocks)} blocks, haps/block: "
                      f"min={min(hap_counts)}, max={max(hap_counts)}, "
                      f"mean={np.mean(hap_counts):.1f} in {time.time()-t0:.1f}s")

                save_contig(DISCOVERY_STAGE, r_name, {
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
                    'discovery_backend': discovery_blocks.DISCOVERY_BACKEND,
                    'discovery_config': discovery_config_record,
                })
                del genomic_data, block_results, global_probs, global_sites
                del global_observed_mask
                gc.collect()

        save_global(DISCOVERY_STAGE, {
            'sample_ids': sample_names,
            'contigs': region_keys,
            'genotype_evidence_mode': (
                workflows_reconstruction.SUPPORTED_GENOTYPE_EVIDENCE_MODE
            ),
            'observed_call_mask_mode': (
                workflows_reconstruction.EXACT_OBSERVED_MASK_MODE
            ),
            'discovery_backend': discovery_blocks.DISCOVERY_BACKEND,
            'discovery_config': discovery_config_record,
        })
        print(f"\nVCF loading + discovery complete in {time.time()-start:.1f}s")
        mark_stage_complete(DISCOVERY_STAGE)

    # =========================================================================
    # ASSEMBLY: COMPONENT ASSEMBLY AND PAINTING
    # =========================================================================
    assembly_config = workflows_reconstruction.ReconstructionConfig(
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
    print("ASSEMBLY: block discovery -> COMPONENT painting")
    print(f"{'='*60}")
    print(
        "  Sequential contigs; release and painting are non-overlapping "
        f"phases with a {n_processes}-core ceiling"
    )
    summaries = workflows_reconstruction.run_reconstruction(
        checkpoint_store,
        region_keys,
        sample_names,
        discovery_identity=discovery_identity_record,
        config=assembly_config,
        genetic_maps=inference_genetic_maps,
        source_stage=DISCOVERY_STAGE,
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
    run_downstream(
        checkpoint_store, region_keys, sample_names,
        output_dir=output_dir, n_workers=n_processes,
        raw_gl_stage=DISCOVERY_STAGE, raw_sites_stage=DISCOVERY_STAGE,
        parent_eligibility=parent_eligibility,
        genetic_maps=inference_genetic_maps, recombination_rate=inference_recombination_rate,
    )



if __name__ == "__main__":
    run()
