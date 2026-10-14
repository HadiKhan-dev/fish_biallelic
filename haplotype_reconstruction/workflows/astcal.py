"""AstCal cross driver from variant input through final phase and maps."""
from __future__ import annotations

import os

from ..core.environment import configured_regions
import haplotype_reconstruction.assembly.pipeline as assembly_pipeline
import haplotype_reconstruction.core.config as core_config
import haplotype_reconstruction.core.environment as core_environment
import haplotype_reconstruction.core.genetic_map as core_genetic_map
import haplotype_reconstruction.core.read_calibration as read_calibration
import haplotype_reconstruction.core.parallel as core_parallel
import haplotype_reconstruction.core.runtime as core_runtime
import haplotype_reconstruction.core.sample_quality as core_sample_quality
import haplotype_reconstruction.core.site_quality as core_site_quality
import haplotype_reconstruction.core.variants as core_variants
import haplotype_reconstruction.discovery.blocks as discovery_blocks
import haplotype_reconstruction.discovery.checkpoints as discovery_checkpoints
import haplotype_reconstruction.discovery.search as discovery_search
import haplotype_reconstruction.workflows.design as workflows_design
import haplotype_reconstruction.workflows.contig_parallel as workflows_contig_parallel
import haplotype_reconstruction.workflows.reconstruction as workflows_reconstruction
from . import CICHLID_AUTOSOMES
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

    # FORCE NUMPY/BLAS TO USE 1 THREAD PER PROCESS
    core_environment.force_single_threaded_numeric_libraries()


    import numpy as np
    import pandas as pd
    import time
    import platform
    from dataclasses import asdict
    from cyvcf2 import VCF

    np.seterr(divide='ignore', invalid='ignore')


    discovery_config = discovery_search.ReversibleCavitySearchConfig()
    discovery_config_record = asdict(discovery_config)
    site_quality_config = core_site_quality.configured(core_config.SiteQualityConfig())
    sample_quality_config = core_sample_quality.configured(core_config.SampleQualityConfig())
    discovery_identity_record = read_calibration.discovery_identity(
        discovery_blocks.DISCOVERY_BACKEND, discovery_config,
        core_site_quality.identity(site_quality_config),
        core_sample_quality.identity(sample_quality_config, site_quality_config))


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

    regions_config = configured_regions(
        [{"contig": contig} for contig in CICHLID_AUTOSOMES])

    output_dir = (
        os.environ.get("HAPLOTYPES_OUTPUT_DIR", "work/runs/astcal")
    )

    # =========================================================================
    # Checkpoint Infrastructure
    # =========================================================================
    checkpoint_store = core_runtime.CheckpointStore(CHECKPOINT_DIR, nthreads=n_processes)
    os.makedirs(output_dir, exist_ok=True)
    stage_complete = checkpoint_store.stage_complete
    mark_stage_complete = checkpoint_store.mark_stage_complete

    region_keys = [r['contig'] for r in regions_config]

    # Get sample names from VCF header
    _vcf_tmp = VCF(vcf_path)
    sample_names = list(_vcf_tmp.samples)
    _vcf_tmp.close()
    n_samples = len(sample_names)
    print(f"VCF samples: {n_samples}")
    print(f"Regions: {len(region_keys)}")


    # =========================================================================
    # STAGE block discovery: VCF Loading + Block Discovery + Global Probabilities
    # =========================================================================
    DISCOVERY_STAGE = workflows_reconstruction.DISCOVERY_STAGE
    # Bind the input file, loader, sample axis and contigs as well, so a
    # different input never resumes these sites (as the general route does);
    # downstream stages still receive the narrower discovery identity.
    checkpoint_store.bind_identity(checkpoint_store.run, DISCOVERY_STAGE, dict(
        schema="astcal-vcf-ad-v1", **core_variants.input_identity(vcf_path),
        sample_ids=sample_names, contigs=region_keys,
        discovery=discovery_identity_record, block_sites=200))

    # Sample/contig manifest checked by reconstruction; recorded before any contig.
    checkpoint_store.bind_manifest(DISCOVERY_STAGE, {
        'sample_ids': sample_names,
        'contigs': region_keys,
        'genotype_evidence_mode': (
            workflows_reconstruction.SUPPORTED_GENOTYPE_EVIDENCE_MODE
        ),
        'observed_call_mask_mode': (
            workflows_reconstruction.EXACT_OBSERVED_MASK_MODE
        ),
        'discovery_backend': discovery_identity_record['backend'],
        'discovery_config': discovery_config_record,
    })

    record_provenance = workflows_reconstruction.discovery_record_provenance(
        discovery_identity_record)

    def prepare_discovery(r_name):
        """Load a contig, filter its sites and fit its read model."""
        print(f"\n  Preparing {r_name}...", flush=True)
        t0 = time.time()
        genomic_data = core_variants.cleanup_block_reads_list(
            vcf_path, r_name,
            use_snp_count=True, snps_per_block=200, snp_shift=200,
            num_processes=n_processes
        )
        print(f"    [Loader] {r_name}: {len(genomic_data)} blocks in {time.time()-t0:.1f}s")
        genomic_data, site_quality = core_site_quality.filter_loaded_sites(
            genomic_data, contig=r_name, config=site_quality_config,
            threads=n_processes, snps_per_block=200,
            sample_config=sample_quality_config)
        # Samples flagged by the per-sample QC on this contig (None if off).
        founder_excluded = core_sample_quality.contig_flags(
            site_quality, contig=r_name, sample_ids=sample_names, output_dir=output_dir)

        global_sites, global_reads = (
            core_variants.concatenate_unique_block_reads(genomic_data)
        )
        if global_sites is None:
            print(f"    WARNING: No data for {r_name}, skipping")
            return None

        # Preserve the observation event before releasing read counts.
        global_observed_mask = (
            workflows_reconstruction.observed_call_mask_from_read_counts(
                global_reads
            )
        )

        # Fit once on observed AD; every stage consumes these same raw GLs.
        global_probs, calibration = read_calibration.prepare_likelihoods(
            global_reads, global_sites, threads=n_processes)
        avg_depth = np.mean(np.sum(global_reads, axis=-1))
        print(f"    {r_name} sites: {len(global_sites)}, Samples: {global_probs.shape[0]}, "
              f"Depth: {avg_depth:.1f}x")
        del global_reads
        # Flagged libraries enter discovery without reads (unobserved).
        genomic_data = core_sample_quality.exclude_from_discovery(
            genomic_data, founder_excluded)
        return discovery_blocks.PreparedContig(
            genomic_data=genomic_data,
            genotype_likelihoods_by_block=read_calibration.block_likelihoods(
                genomic_data, global_sites, global_probs),
            fields=dict(
                global_probs=global_probs, read_calibration=calibration,
                site_quality=site_quality, global_sites=global_sites,
                global_observed_mask=global_observed_mask, **record_provenance,
                avg_depth=avg_depth,
                **({} if founder_excluded is None
                   else dict(founder_evidence_excluded=founder_excluded))),
            products=core_site_quality.run_record_products(site_quality, sample_names),
        )

    def discover(contigs):
        """Discover these contigs with one block pool, closed on return."""
        with discovery_blocks.BlockDiscoveryPool(n_processes) as block_pool:
            discovery_blocks.discover_contigs(
                checkpoint_store, contigs, prepare_discovery,
                block_pool=block_pool, discovery_config=discovery_config)

    assembly_config = workflows_reconstruction.ReconstructionConfig(
        release_config=assembly_pipeline.AssemblyConfig(
            num_processes=n_processes,
            maxtasksperchild=WORKER_MAXTASKS,
            recombination_rate=inference_recombination_rate,
        ),
        paint_cores=n_processes,
        paint_recombination_rate=inference_recombination_rate,
    )

    # Serial contigs: each contig is discovered, assembled and painted before
    # the next; the stages below then publish completion as in a normal run.
    serial_contigs, max_contigs = (
        workflows_contig_parallel.configured_serial_contigs())
    if serial_contigs and not workflows_contig_parallel.run_serially(
            checkpoint_store, region_keys, sample_names, discover=discover,
            discovery_identity=discovery_identity_record, config=assembly_config,
            output_dir=output_dir, max_contigs=max_contigs,
            genetic_maps=inference_genetic_maps):
        return

    if stage_complete(DISCOVERY_STAGE):
        print(f"\n[RESUME] Skipping VCF loading + discovery (checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("STAGE block discovery: VCF Loading + Block Haplotype Discovery")
        print(f"{'='*60}")
        start = time.time()

        pending_contigs = []
        for r_name in region_keys:
            if discovery_checkpoints.complete(checkpoint_store.contig(r_name)):
                print(f"  [RESUME] {r_name} already done")
            else:
                pending_contigs.append(r_name)
        if pending_contigs:
            discover(pending_contigs)
        print(f"\nVCF loading + discovery complete in {time.time()-start:.1f}s")
        discovery_checkpoints.require_complete(checkpoint_store, region_keys)
        mark_stage_complete(DISCOVERY_STAGE)

    if os.environ.get("HAPLOTYPES_STOP_AFTER_STAGE") == "block_discovery":
        print("[STOP] Block discovery checkpointed; feedback and assembly not started.")
        return

    # =========================================================================
    # ASSEMBLY: COMPONENT ASSEMBLY AND PAINTING
    # =========================================================================
    print()
    print("=" * 60)
    print("ASSEMBLY: block discovery -> COMPONENT painting")
    print(f"{'='*60}")
    print(
        "  Release and painting are non-overlapping phases within each contig; "
        f"concurrent contigs share a {n_processes}-core ceiling"
    )
    summaries = workflows_contig_parallel.run_reconstruction(
        checkpoint_store,
        region_keys,
        sample_names,
        discovery_identity=discovery_identity_record,
        config=assembly_config,
        output_dir=output_dir,
        genetic_maps=inference_genetic_maps,
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
        parent_eligibility=parent_eligibility,
        genetic_maps=inference_genetic_maps, recombination_rate=inference_recombination_rate,
    )


if __name__ == "__main__":
    run()
