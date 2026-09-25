"""Tropheops cross driver from variant input through final phase and maps."""
from __future__ import annotations

import os

from ..core.environment import configured_regions
import haplotype_reconstruction.assembly.pipeline as assembly_pipeline
import haplotype_reconstruction.core.environment as core_environment
import haplotype_reconstruction.core.genetic_map as core_genetic_map
import haplotype_reconstruction.core.haplotypes as core_haplotypes
import haplotype_reconstruction.core.numerics as core_numerics
import haplotype_reconstruction.core.parallel as core_parallel
import haplotype_reconstruction.core.read_calibration as read_calibration
import haplotype_reconstruction.core.runtime as core_runtime
import haplotype_reconstruction.core.variants as core_variants
import haplotype_reconstruction.discovery.blocks as discovery_blocks
import haplotype_reconstruction.discovery.search as discovery_search
import haplotype_reconstruction.workflows.design as workflows_design
import haplotype_reconstruction.workflows.reconstruction as workflows_reconstruction
from . import CICHLID_AUTOSOMES
from .downstream import run_downstream
from .reports import report_discovery, write_reference_comparison

INCLUDE_REFERENCE_SAMPLES = True


DISCOVERY_BACKEND = "reversible_cavity_depth_observation_v1"


_mode_label = "references_included" if INCLUDE_REFERENCE_SAMPLES else "references_excluded"


CHECKPOINT_DIR = os.environ.get("HAPLOTYPES_CHECKPOINT_DIR", "work/runs/tropheops/checkpoints")


output_dir = os.environ.get("HAPLOTYPES_OUTPUT_DIR", "work/runs/tropheops")


@core_runtime.logged_workflow("tropheops")
def run():
    """Execute or resume the configured, chromosome-checkpointed workflow."""

    # Enable faulthandler FIRST — catches C-level segfaults in numba-compiled
    # code, numpy, BLAS, etc. and prints a Python traceback to stderr before
    # the process dies.  Without this, such faults leave no trail (silent
    # worker death). Diagnostics go to stderr, separate from the stdout log.
    import faulthandler
    faulthandler.enable()

    # FORCE NUMPY/BLAS TO USE 1 THREAD PER PROCESS
    core_environment.force_single_threaded_numeric_libraries()

    # =============================================================================
    # CONFIGURATION
    # =============================================================================
    # INCLUDE_REFERENCE_SAMPLES, _mode_label, CHECKPOINT_DIR and output_dir are defined
    # at module top level. Edit the comparison flag there, not here.


    print(f"INCLUDE_REFERENCE_SAMPLES = {INCLUDE_REFERENCE_SAMPLES}  (mode: {_mode_label})")
    print(f"DISCOVERY_BACKEND = {DISCOVERY_BACKEND}")
    print("ASSEMBLY_ROUTE = component-local block discovery -> painting")

    import numpy as np
    import pandas as pd
    import time
    import platform
    import gc
    from dataclasses import asdict
    from cyvcf2 import VCF

    np.seterr(divide='ignore', invalid='ignore')


    if DISCOVERY_BACKEND != discovery_blocks.DISCOVERY_BACKEND:
        raise RuntimeError("block discovery checkpoint identity mismatch")


    rate_maps = core_genetic_map.load_genetic_maps_from_environment()
    inference_recombination_rate = rate_maps.default_rate_cm_per_mb / 1e8
    inference_genetic_maps = rate_maps if rate_maps.maps else None

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    if platform.system() != "Windows":
        print(f"Main process ({os.getpid()}) niceness set to: {os.nice(0)}")

    n_processes = int(os.environ.get("BHD_NUM_PROCESSES", str(core_runtime.available_cpu_count())))
    # block discovery uses the complete allocation: one block worker per Numba thread.
    # Dynamic reallocation gives the full budget to remaining stragglers.
    block_discovery_processes = n_processes
    block_discovery_numba_threads = n_processes
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
    # Paths & Regions (AcTm tropheops cross)
    # =========================================================================
    vcf_path = os.environ.get("HAPLOTYPES_VCF", "work/data/fish_vcf_restriped/AcTm.biallelic.bcf.gz")
    meta_path = os.environ.get("HAPLOTYPES_METADATA", "work/data/fish_vcf_restriped/X_AcTm_metadata.xlsx")

    # AcTm BCF covers the same reference as the AsAc files: chr1-chr20, chr22,
    # chr23 autosomes, plus chrM and U_scaffolds.  We only run the pipeline on
    # the 22 autosomes (chrM has no recombination; U_scaffolds are short/unplaced
    # and not useful for pedigree-scale linkage).
    regions_config = configured_regions(
        [{"contig": contig} for contig in CICHLID_AUTOSOMES])

    # CHECKPOINT_DIR and output_dir are defined at module top level (above).
    # =========================================================================
    # Checkpoint infrastructure (atomic core/checkpoints records)
    # =========================================================================
    checkpoint_store = core_runtime.CheckpointStore(
        CHECKPOINT_DIR, nthreads=n_processes, global_log_indent="    "
    )
    os.makedirs(output_dir, exist_ok=True)
    stage_complete = checkpoint_store.stage_complete
    mark_stage_complete = checkpoint_store.mark_stage_complete
    contig_done = checkpoint_store.contig_done
    save_contig = checkpoint_store.save_contig
    save_global = checkpoint_store.save_global


    region_keys = [r['contig'] for r in regions_config]

    # =========================================================================
    # SAMPLE IDENTIFICATION — match VCF samples to metafile, find G0 indices
    # =========================================================================
    # This runs before any stage so we always know:
    #   g0_vcf_indices      : positions of the 4 G0 samples in the VCF header
    #   active_vcf_indices  : positions of the samples the pipeline will see
    #                         (all 116 if INCLUDE_REFERENCE_SAMPLES, else 112 = no G0s)
    #   sample_names_active : VCF sample names the pipeline will see
    #                         (the ordered block discovery and component-painting sample axis)
    #   g0_sample_names     : the 4 G0 primary_IDs (for post-hoc reference comparison, not trio truth)
    print(f"\n{'='*60}")
    print("Sample Identification (VCF <-> metafile)")
    print(f"{'='*60}")

    _vcf_tmp = VCF(vcf_path)
    sample_names = list(_vcf_tmp.samples)
    _vcf_tmp.close()
    n_samples_total = len(sample_names)
    print(f"VCF samples: {n_samples_total}")

    # Load metafile main_data sheet — contains generation column
    meta_df = pd.read_excel(meta_path, sheet_name=os.environ.get('HAPLOTYPES_METADATA_SHEET', 'main_data'))
    print(f"Metafile main_data rows: {len(meta_df)}")

    # Match BCF samples to metafile by primary_ID (user verified this is the
    # ID column with 116/116 matches).
    bcf_set = set(sample_names)
    matched_meta = meta_df[meta_df['primary_ID'].astype(str).isin(bcf_set)].copy()
    print(f"Matched {len(matched_meta)}/{n_samples_total} VCF samples via primary_ID")

    unmatched = bcf_set - set(matched_meta['primary_ID'].astype(str))
    if unmatched:
        print(f"WARNING: {len(unmatched)} VCF samples not in metafile:")
        for s in sorted(unmatched)[:5]:
            print(f"  {s}")
        # Do not hard-fail: unmatched samples remain active but cannot be
        # identified as G0 reference rows from metadata.

    # Build a primary_ID -> generation lookup
    id_to_gen = dict(zip(matched_meta['primary_ID'].astype(str),
                         matched_meta['generation'].astype(str)))

    # Identify G0 indices in the VCF sample list
    g0_vcf_indices = []
    g0_sample_names = []
    for i, s in enumerate(sample_names):
        if id_to_gen.get(s) == 'G0':
            g0_vcf_indices.append(i)
            g0_sample_names.append(s)

    if len(g0_vcf_indices) != 4:
        print(f"WARNING: Expected 4 G0 samples, found {len(g0_vcf_indices)}: "
              f"{g0_sample_names}")
    else:
        print(f"Identified 4 G0 samples at VCF indices {g0_vcf_indices}:")
        for idx, name in zip(g0_vcf_indices, g0_sample_names):
            print(f"  [{idx}] {name}")

    # Decide which samples the pipeline will see
    if INCLUDE_REFERENCE_SAMPLES:
        active_vcf_indices = np.arange(n_samples_total, dtype=np.int64)
        print(f"\nINCLUDE_REFERENCE_SAMPLES=True -> pipeline sees ALL {n_samples_total} samples "
              f"(G0 included)")
    else:
        g0_set = set(g0_vcf_indices)
        active_vcf_indices = np.array(
            [i for i in range(n_samples_total) if i not in g0_set],
            dtype=np.int64
        )
        print(f"\nINCLUDE_REFERENCE_SAMPLES=False -> pipeline sees {len(active_vcf_indices)} "
              f"samples (G0 removed)")

    sample_names_active = [sample_names[i] for i in active_vcf_indices]

    # Sanity-log generation composition of active samples
    gen_counts_active = pd.Series(
        [id_to_gen.get(s, '?') for s in sample_names_active]
    ).value_counts()
    print(f"Active sample generation breakdown:")
    for gen, count in gen_counts_active.items():
        print(f"  {gen}: {count}")

    print(f"Regions: {len(region_keys)}")

    # =========================================================================
    # STAGE block discovery: VCF Loading + Block Discovery + Global Probabilities
    # =========================================================================
    # Retain observed G0 genotypes separately from the active sample axis. We
    # split out the G0 reads into a separate `g0_slice` that's stashed in the
    # checkpoint for block discovery validation and missing-aware assembly.  When
    # INCLUDE_REFERENCE_SAMPLES=False, the main global_probs/global_sites/block_results
    # are computed from the 112 non-G0 samples only (the reads array is sliced
    # along the sample axis before reads_to_probabilities / block discovery).
    DISCOVERY_STAGE = "block_discovery"
    discovery_config = discovery_search.ReversibleCavitySearchConfig()
    discovery_config_record = asdict(discovery_config)
    discovery_identity_record = read_calibration.discovery_identity(
        DISCOVERY_BACKEND, discovery_config)
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
        print(
            "  Cap-free reversible cavity discovery: no explicit K grid or "
            "scientific K cap; "
            f"beam_width={discovery_config.beam_width}, "
            f"max_expansions={discovery_config.max_expansions}, "
            f"max_exact_scores={discovery_config.max_exact_scores}, "
            "max_proposals_per_expansion="
            f"{discovery_config.max_proposals_per_expansion}"
        )
        print(
            "  Block discovery parallelism: "
            f"workers={block_discovery_processes}, "
            f"Numba budget={block_discovery_numba_threads}"
        )

        with discovery_blocks.BlockDiscoveryPool(
            block_discovery_processes,
            block_discovery_numba_threads,
        ) as block_pool:
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

                # Full reads: (n_samples_total, n_sites, 2) — all 116 samples
                global_sites, global_reads_full = (
                    core_variants.concatenate_unique_block_reads(genomic_data)
                )
                if global_sites is None:
                    print(f"    WARNING: No data for {r_name}, skipping")
                    continue

                # ALWAYS extract G0 reads separately for post-hoc validation.
                # Retain observed reference genotypes under either discovery
                # policy. They are not independent biological truth.
                g0_reads = global_reads_full[g0_vcf_indices,:,:]
                (_, g0_probs) = core_numerics.reads_to_probabilities(
                    g0_reads,
                    use_hwe_prior=False,
                )
                # Downcast G0 probs to float32 — we only use argmax for validation,
                # so float64 precision is wasted.
                if g0_probs.dtype == np.float64:
                    g0_probs = g0_probs.astype(np.float32)

                # Select which samples the pipeline will see (116 or 112).
                # IMPORTANT: we also need to slice genomic_data.reads along the
                # sample axis so generate_all_block_haplotypes
                # operates on the filtered sample set.  The GenomicData container
                # stores per-block (samples, sites, 2) arrays.
                if INCLUDE_REFERENCE_SAMPLES:
                    active_reads_full = global_reads_full
                else:
                    active_reads_full = global_reads_full[active_vcf_indices,:,:]
                    # Also filter genomic_data in place so block discovery sees 112 samples
                    for bi in range(len(genomic_data.reads)):
                        if genomic_data.reads[bi].shape[0] == n_samples_total:
                            genomic_data.reads[bi] = genomic_data.reads[bi][active_vcf_indices,:,:]

                # Preserve the exact observation event before read counts are
                # released. Zero-depth cells are scientifically distinct from
                # uncertain observed genotypes and must remain state-neutral in
                # missing-aware assembly.
                global_observed_mask = (
                    workflows_reconstruction.observed_call_mask_from_read_counts(
                        active_reads_full
                    )
                )

                # Calibrate only the selected analysis cohort, not excluded rows.
                global_probs, calibration = read_calibration.prepare_likelihoods(
                    active_reads_full, global_sites, threads=n_processes)
                avg_depth = np.mean(np.sum(active_reads_full, axis=-1))
                print(f"    Sites: {len(global_sites)}, Samples (active): {global_probs.shape[0]}, "
                      f"Depth: {avg_depth:.1f}x")
                del global_reads_full, active_reads_full, g0_reads

                t0 = time.time()
                block_results = discovery_blocks.generate_all_block_haplotypes(
                    genomic_data,
                    num_processes=block_discovery_processes,
                    discovery_config=discovery_config,
                    genotype_likelihoods_by_block=read_calibration.block_likelihoods(
                        genomic_data, global_sites, global_probs),
                    total_numba_threads=block_discovery_numba_threads,
                    block_pool=block_pool,
                )
                valid_blocks = [b for b in block_results if len(b.positions) > 0]
                block_results = core_haplotypes.BlockResults(valid_blocks)

                hap_counts = [len(b.haplotypes) for b in valid_blocks]
                print(f"    [Discovery] {len(valid_blocks)} blocks, haps/block: "
                      f"min={min(hap_counts)}, max={max(hap_counts)}, "
                      f"mean={np.mean(hap_counts):.1f} in {time.time()-t0:.1f}s")

                report_discovery(valid_blocks)

                # G0 probabilities are retained as an explicit reference; they are
                # non-independent when G0 rows entered reconstruction above.
                save_contig(DISCOVERY_STAGE, r_name, {
                    'global_probs': global_probs, 'global_sites': global_sites,
                    'read_calibration': calibration,
                    'global_observed_mask': global_observed_mask,
                    'observed_call_mask_mode': (
                        workflows_reconstruction.EXACT_OBSERVED_MASK_MODE),
                    'genotype_evidence_mode': (
                        'normalized_raw_linear_likelihood_v1'),
                    'block_results': block_results, 'avg_depth': avg_depth,
                    'g0_probs': g0_probs, 'g0_sample_names': g0_sample_names,
                    'active_vcf_indices': active_vcf_indices,
                    'discovery_backend': discovery_identity_record['backend'],
                    'discovery_config': discovery_config_record,
                })
                del genomic_data, block_results, global_probs, global_sites
                del global_observed_mask, g0_probs
                gc.collect()

        save_global(DISCOVERY_STAGE, {
            'sample_ids': sample_names_active,
            'sample_names_full': sample_names,
            'contigs': region_keys,
            'g0_vcf_indices': g0_vcf_indices,
            'g0_sample_names': g0_sample_names,
            'active_vcf_indices': active_vcf_indices,
            'include_reference_samples': INCLUDE_REFERENCE_SAMPLES,
            'genotype_evidence_mode': 'normalized_raw_linear_likelihood_v1',
            'observed_call_mask_mode': (
                workflows_reconstruction.EXACT_OBSERVED_MASK_MODE
            ),
            'discovery_backend': discovery_identity_record['backend'],
            'discovery_config': discovery_config_record,
        })
        print(f"\nVCF loading + discovery complete in {time.time()-start:.1f}s")
        mark_stage_complete(DISCOVERY_STAGE)

    write_reference_comparison(
        checkpoint_store, region_keys, output_dir,
        include_reference_samples=INCLUDE_REFERENCE_SAMPLES,
        source_stage=DISCOVERY_STAGE,
    )

    # =====================================================================
    # ASSEMBLY THROUGH TYPED COMPONENT painting
    # =====================================================================
    # The canonical release preprocesses the raw block discovery blocks, assembles
    # rectangular-K phase components through L1-L4, and paints each component
    # in an independent founder namespace using the exact block discovery observation mask.
    assembly_config = workflows_reconstruction.ReconstructionConfig(
        release_config=assembly_pipeline.AssemblyConfig(
            num_processes=n_processes,
            maxtasksperchild=WORKER_MAXTASKS,
            recombination_rate=inference_recombination_rate,
        ),
        paint_cores=n_processes,
        paint_recombination_rate=inference_recombination_rate,
    )
    print(f"\n{'='*60}")
    print("ASSEMBLY: block discovery -> COMPONENT painting")
    print(f"{'='*60}")
    print(
        "  Sequential contigs; release and painting are non-overlapping "
        f"phases with a {n_processes}-core ceiling"
    )
    assembly_summaries = workflows_reconstruction.run_reconstruction(
        checkpoint_store,
        region_keys,
        sample_names_active,
        discovery_identity=discovery_identity_record,
        config=assembly_config,
        genetic_maps=inference_genetic_maps,
        source_stage=DISCOVERY_STAGE,
    )
    for summary in assembly_summaries:
        status = "resumed" if summary.resumed else "completed"
        print(
            f"  [{status}] {summary.contig}: "
            f"components={summary.component_count}, "
            "evidence-eligible component-sample pairs="
            f"{summary.evidence_eligible_component_sample_pairs}/"
            f"{summary.total_component_sample_pairs}, "
            f"observation mask={summary.observed_mask_mode}"
        )
    # Existing F2 <- F1 design eligibility excludes G0 and outside-pedigree
    # reference samples; it does not assume an individual parental pair.
    parent_eligibility = workflows_design.build_tropheops_parent_eligibility(
        meta_df, sample_names_active, require_opposite_sex_pair=True,
    )
    run_downstream(
        checkpoint_store, region_keys, sample_names_active,
        output_dir=output_dir, n_workers=n_processes,
        raw_gl_stage=DISCOVERY_STAGE, raw_sites_stage=DISCOVERY_STAGE,
        parent_eligibility=parent_eligibility,
        genetic_maps=inference_genetic_maps, recombination_rate=inference_recombination_rate,
    )


if __name__ == "__main__":
    run()
