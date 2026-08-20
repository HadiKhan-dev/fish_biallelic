#%%
# =============================================================================
# Module-level definitions (PICKLABLE by forkserver workers)
# =============================================================================
# Functions defined inside `if __name__ == '__main__':` are closures that
# cannot be pickled by multiprocessing.  Forkserver workers receive their
# initargs (including callback functions) via pickle, so any function that
# crosses the worker boundary MUST live at module top level here.  Keep this
# section small -- imports here run in every forkserver worker at startup.

CHECKPOINT_DIR = ".pipeline_checkpoints_real"

import os
import pipeline_runtime

from thread_env import force_single_threaded_numeric_libraries

def _load_contig_for_phase_correction(r_name):
    """Load the atomic final-panel painting bundle for phase correction."""
    return pipeline_runtime.load_phase_correction_inputs(
        CHECKPOINT_DIR,
        r_name,
        tolerance_stage="R09_viterbi_painting",
        strip_founder_probs=True,
    )


#%%
if __name__ == '__main__':
    import os
    import sys
    from datetime import datetime

    # FORCE NUMPY/BLAS TO USE 1 THREAD PER PROCESS
    force_single_threaded_numeric_libraries()

    # =============================================================================
    # DUAL LOGGING: Console + File
    # =============================================================================

    os.makedirs("logs", exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join("logs", f"run_real_{run_timestamp}.log")
    sys.stdout = pipeline_runtime.TeeOutput(log_path, sys.stdout)
    print(f"Logging to: {log_path}")
    print(f"Run started: {run_timestamp}")

    import numpy as np
    import pandas as pd
    import time
    import warnings
    import platform
    import gc
    from cyvcf2 import VCF

    warnings.filterwarnings("ignore")
    np.seterr(divide='ignore', invalid='ignore')

    import thread_config
    import vcf_data_loader
    import block_haplotypes
    import small_block_refine
    import residual_discovery
    import hierarchical_assembly
    import paint_samples
    import pedigree_inference
    import phase_correction
    import analysis_utils
    import terminal_cavity_refinement

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    if platform.system() != "Windows":
        print(f"Main process ({os.getpid()}) niceness set to: {os.nice(0)}")

    n_processes = 112
    # Recycle workers after each batch to prevent memory accumulation
    # from glibc malloc fragmentation (Python doesn't return freed pages to OS).
    WORKER_MAXTASKS = 1

    # Start forkserver before data loading
    _warmup_pool = hierarchical_assembly.NoDaemonPool(1)
    _warmup_pool.terminate()
    _warmup_pool.join()
    del _warmup_pool
    print("Forkserver started (lightweight, pre-data).")
    print(f"Numba threading layer: {os.environ.get('NUMBA_THREADING_LAYER', 'not set')}")

    # =========================================================================
    # Configuration
    # =========================================================================
    vcf_path = "./fish_vcf_restriped/AsAc.AulStuGenome.biallelic.bcf.gz"

    regions_config = [
        {"contig": "chr1"},  {"contig": "chr2"},  {"contig": "chr3"},
        {"contig": "chr4"},  {"contig": "chr5"},  {"contig": "chr6"},
        {"contig": "chr7"},  {"contig": "chr8"},  {"contig": "chr9"},
        {"contig": "chr10"}, {"contig": "chr11"}, {"contig": "chr12"},
        {"contig": "chr13"}, {"contig": "chr14"}, {"contig": "chr15"},
        {"contig": "chr16"}, {"contig": "chr17"}, {"contig": "chr18"},
        {"contig": "chr19"}, {"contig": "chr20"}, {"contig": "chr22"},
        {"contig": "chr23"},
    ]

    output_dir = "results_real"

    # =========================================================================
    # Checkpoint Infrastructure
    # =========================================================================
    checkpoint_store = pipeline_runtime.CheckpointStore(
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

    strip_block_probs = pipeline_runtime.strip_block_probs

    def load_global_arrays(r_name):
        return pipeline_runtime.load_global_arrays(
            checkpoint_store, STAGE_R1, r_name
        )

    region_keys = [r['contig'] for r in regions_config]

    # Get sample names from VCF header
    _vcf_tmp = VCF(vcf_path)
    sample_names = list(_vcf_tmp.samples)
    _vcf_tmp.close()
    n_samples = len(sample_names)
    print(f"VCF samples: {n_samples}")
    print(f"Regions: {len(region_keys)}")

    total_pipeline_start = time.time()

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE R01: VCF Loading + Block Discovery + Global Probabilities
    # =========================================================================
    STAGE_R1 = "R01_vcf_discovery"

    if stage_complete(STAGE_R1):
        print(f"\n[RESUME] Skipping VCF loading + discovery (checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("STAGE R01: VCF Loading + Block Haplotype Discovery")
        print(f"{'='*60}")
        start = time.time()

        for r_name in region_keys:
            if contig_done(STAGE_R1, r_name):
                print(f"  [RESUME] {r_name} already done")
                continue
            print(f"\n  Processing {r_name}...")

            t0 = time.time()
            genomic_data = vcf_data_loader.cleanup_block_reads_list(
                vcf_path, r_name,
                use_snp_count=True, snps_per_block=200, snp_shift=200,
                num_processes=16
            )
            print(f"    [Loader] {len(genomic_data)} blocks in {time.time()-t0:.1f}s")

            global_sites, global_reads = (
                vcf_data_loader.concatenate_unique_block_reads(genomic_data)
            )
            if global_sites is None:
                print(f"    WARNING: No data for {r_name}, skipping")
                continue

            # Keep cohort-frequency regularization inside block discovery;
            # linkage and assembly consume the raw genotype likelihoods.
            (site_priors, global_probs) = analysis_utils.reads_to_probabilities(
                global_reads,
                use_hwe_prior=False,
            )
            avg_depth = np.mean(np.sum(global_reads, axis=-1))
            print(f"    Sites: {len(global_sites)}, Samples: {global_probs.shape[0]}, "
                  f"Depth: {avg_depth:.1f}x")
            del global_reads, site_priors

            t0 = time.time()
            block_results = block_haplotypes.generate_all_block_haplotypes(
                genomic_data,
                uniqueness_threshold_percent=1.0,
                diff_threshold_percent=0.5,
                wrongness_threshold=1.0,
                num_processes=n_processes
            )
            valid_blocks = [b for b in block_results if len(b.positions) > 0]
            block_results = block_haplotypes.BlockResults(valid_blocks)

            hap_counts = [len(b.haplotypes) for b in valid_blocks]
            print(f"    [Discovery] {len(valid_blocks)} blocks, haps/block: "
                  f"min={min(hap_counts)}, max={max(hap_counts)}, "
                  f"mean={np.mean(hap_counts):.1f} in {time.time()-t0:.1f}s")

            save_contig(STAGE_R1, r_name, {
                'global_probs': global_probs, 'global_sites': global_sites,
                'block_results': block_results, 'avg_depth': avg_depth,
            })
            del genomic_data, block_results, global_probs, global_sites
            gc.collect()

        save_global(STAGE_R1, {'sample_names': sample_names, 'region_keys': region_keys})
        print(f"\nVCF loading + discovery complete in {time.time()-start:.1f}s")
        mark_stage_complete(STAGE_R1)

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE R02: Refinement (if avg depth < 10x)
    # =========================================================================
    STAGE_R2 = "R02_refinement"

    if stage_complete(STAGE_R2):
        print(f"\n[RESUME] Skipping refinement (checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("STAGE R02: Checking Read Depth for Refinement")
        print(f"{'='*60}")

        REFINEMENT_DEPTH_THRESHOLD = 100.0
        REFINEMENT_BATCH_SIZE = 10
        REFINEMENT_PENALTY_SCALE = 20.0
        RECOMB_RATE = 5e-8
        N_GENERATIONS = 3

        import chimera_resolution
        start = time.time()

        for r_name in region_keys:
            if contig_done(STAGE_R2, r_name):
                print(f"  [RESUME] {r_name} already done")
                continue

            r1 = load_contig(STAGE_R1, r_name)
            avg_depth = r1['avg_depth']
            global_probs = r1['global_probs']
            global_sites = r1['global_sites']
            block_results = strip_block_probs(r1['block_results'])
            del r1
            # Downcast: float64 only needed for HDBSCAN (R01)
            if global_probs.dtype == np.float64:
                global_probs = global_probs.astype(np.float32)

            print(f"\n{'='*60}")
            print(f"{r_name}: average read depth = {avg_depth:.1f}x")
            print(f"{'='*60}")

            if avg_depth < REFINEMENT_DEPTH_THRESHOLD:
                print(f"  Depth < {REFINEMENT_DEPTH_THRESHOLD}x -> Running L1+L2 refinement")
                num_samples = global_probs.shape[0]
                chimera_resolution.warmup_jit(num_samples)

                l1_fn, l2_fn = pipeline_runtime.make_refinement_assembly_functions(
                    hierarchical_assembly.run_hierarchical_step,
                    global_probs,
                    global_sites,
                    batch_size=REFINEMENT_BATCH_SIZE,
                    recomb_rate=RECOMB_RATE,
                    n_generations=N_GENERATIONS,
                    beam_width=200,
                    max_founders=12,
                    cc_scale=0.5,
                    num_processes=n_processes,
                    maxtasksperchild=WORKER_MAXTASKS,
                )

                t0 = time.time()
                refinement_results = small_block_refine.run_refinement_pipeline(
                    raw_blocks=block_results, global_probs=global_probs,
                    global_sites=global_sites, num_samples=num_samples,
                    run_l1_assembly_fn=l1_fn,
                    run_l2_assembly_fn=l2_fn,
                    batch_size=REFINEMENT_BATCH_SIZE, penalty_scale=REFINEMENT_PENALTY_SCALE,
                    recomb_rate=RECOMB_RATE, n_generations=N_GENERATIONS, verbose=True)
                print(f"\n  Refinement complete in {time.time()-t0:.0f}s")

                l2_refined = refinement_results['l2_refined']
                l2_refined_dd = small_block_refine.dedup_blocks(l2_refined, verbose=True)
                save_contig(STAGE_R2, r_name, {'block_results': l2_refined_dd})
                del refinement_results, l2_refined, l2_refined_dd
            else:
                print(f"  Depth >= {REFINEMENT_DEPTH_THRESHOLD}x -> Skipping refinement")
                save_contig(STAGE_R2, r_name, {'block_results': block_results})

            del block_results, global_probs, global_sites
            gc.collect()

        print(f"\nRefinement stage complete in {time.time()-start:.1f}s")
        mark_stage_complete(STAGE_R2)

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE R03: Residual Discovery (Missing Founder Recovery)
    # =========================================================================
    STAGE_R3 = "R03_residual_discovery"

    if stage_complete(STAGE_R3):
        print(f"\n[RESUME] Skipping residual discovery (checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("STAGE R03: Residual Discovery (Missing Founder Recovery)")
        print(f"{'='*60}")
        start = time.time()

        for r_name in region_keys:
            if contig_done(STAGE_R3, r_name):
                print(f"  [RESUME] {r_name} already done")
                continue
            print(f"\n  Processing {r_name}...")

            r2 = load_contig(STAGE_R2, r_name)
            blocks = strip_block_probs(r2['block_results'])
            del r2

            global_probs, global_sites = load_global_arrays(r_name)

            print(f"    Input: {len(blocks)} blocks, "
                  f"avg haps: {np.mean([len(b.haplotypes) for b in blocks]):.1f}")

            blocks_out = residual_discovery.discover_missing_haplotypes(
                blocks, global_probs, global_sites,
                min_residual_reduction=0.10,
                num_processes=n_processes,
                verbose=True
            )

            print(f"    Output: {len(blocks_out)} blocks, "
                  f"avg haps: {np.mean([len(b.haplotypes) for b in blocks_out]):.1f}")

            pipeline_runtime.strip_block_evidence(blocks_out)
            save_contig(STAGE_R3, r_name, {'block_results': blocks_out})
            del blocks, blocks_out, global_probs, global_sites
            gc.collect()

        print(f"\nResidual discovery complete in {time.time()-start:.1f}s")
        mark_stage_complete(STAGE_R3)

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE R04: L1 Assembly
    # =========================================================================
    STAGE_R4 = "R04_assembly_L1"

    if stage_complete(STAGE_R4):
        print(f"\n[RESUME] Skipping L1 assembly (checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("STAGE R04: Level 1 Hierarchical Assembly")
        print(f"{'='*60}")
        start = time.time()

        for r_name in region_keys:
            if contig_done(STAGE_R4, r_name):
                print(f"  [RESUME] {r_name} already done")
                continue
            print(f"\n  Processing {r_name}...")

            r3 = load_contig(STAGE_R3, r_name)
            block_results = strip_block_probs(r3['block_results'])
            del r3

            global_probs, global_sites = load_global_arrays(r_name)

            print(f"    Input: {len(block_results)} blocks")

            super_blocks = hierarchical_assembly.run_hierarchical_step(
                block_results, global_probs, global_sites,
                batch_size=10, use_hmm_linking=False, beam_width=200,
                max_founders=12, max_sites_for_linking=2000, cc_scale=0.5,
                num_processes=n_processes, maxtasksperchild=WORKER_MAXTASKS,
                verbose=False)

            hap_counts = [len(b.haplotypes) for b in super_blocks]
            print(f"    Output: {len(super_blocks)} L1 super-blocks, "
                  f"haps: min={min(hap_counts)}, max={max(hap_counts)}, "
                  f"mean={np.mean(hap_counts):.1f}")

            pipeline_runtime.strip_block_evidence(super_blocks)
            save_contig(STAGE_R4, r_name, {'super_blocks_L1': super_blocks})
            del block_results, global_probs, super_blocks
            gc.collect()

        print(f"\nL1 assembly complete in {time.time()-start:.1f}s")
        mark_stage_complete(STAGE_R4)

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE R05: L2 Assembly
    # =========================================================================
    STAGE_R5 = "R05_assembly_L2"

    if stage_complete(STAGE_R5):
        print(f"\n[RESUME] Skipping L2 assembly (checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("STAGE R05: Level 2 Hierarchical Assembly")
        print(f"{'='*60}")
        start = time.time()

        for r_name in region_keys:
            if contig_done(STAGE_R5, r_name):
                print(f"  [RESUME] {r_name} already done")
                continue
            print(f"\n  Processing {r_name}...")

            r4 = load_contig(STAGE_R4, r_name)
            l1_blocks = strip_block_probs(r4['super_blocks_L1'])
            del r4

            global_probs, global_sites = load_global_arrays(r_name)

            print(f"    Input: {len(l1_blocks)} L1 super-blocks")

            l2_blocks = hierarchical_assembly.run_hierarchical_step(
                l1_blocks, global_probs, global_sites,
                batch_size=10, use_hmm_linking=True, recomb_rate=5e-8,
                beam_width=200, max_founders=12, cc_scale=0.5,
                num_processes=n_processes, maxtasksperchild=WORKER_MAXTASKS,
                n_generations=3, verbose=False)

            hap_counts = [len(b.haplotypes) for b in l2_blocks]
            print(f"    Output: {len(l2_blocks)} L2 super-blocks, haps: {hap_counts}")

            pipeline_runtime.strip_block_evidence(l2_blocks)
            save_contig(STAGE_R5, r_name, {'super_blocks_L2': l2_blocks})
            del l1_blocks, global_probs, l2_blocks
            gc.collect()

        print(f"\nL2 assembly complete in {time.time()-start:.1f}s")
        mark_stage_complete(STAGE_R5)

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE R06: L3 Assembly
    # =========================================================================
    STAGE_R6 = "R06_assembly_L3"

    if stage_complete(STAGE_R6):
        print(f"\n[RESUME] Skipping L3 assembly (checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("STAGE R06: Level 3 Hierarchical Assembly")
        print(f"{'='*60}")
        start = time.time()

        for r_name in region_keys:
            if contig_done(STAGE_R6, r_name):
                print(f"  [RESUME] {r_name} already done")
                continue
            print(f"\n  Processing {r_name}...")

            r5 = load_contig(STAGE_R5, r_name)
            l2_blocks = strip_block_probs(r5['super_blocks_L2'])
            del r5

            global_probs, global_sites = load_global_arrays(r_name)

            print(f"    Input: {len(l2_blocks)} L2 super-blocks")

            l3_blocks = hierarchical_assembly.run_hierarchical_step(
                l2_blocks, global_probs, global_sites,
                batch_size=10, use_hmm_linking=True, recomb_rate=5e-8,
                beam_width=200, max_founders=12, cc_scale=0.5,
                num_processes=n_processes, maxtasksperchild=WORKER_MAXTASKS,
                n_generations=3, verbose=False)

            hap_counts = [len(b.haplotypes) for b in l3_blocks]
            print(f"    Output: {len(l3_blocks)} L3 super-blocks, haps: {hap_counts}")

            pipeline_runtime.strip_block_evidence(l3_blocks)
            save_contig(STAGE_R6, r_name, {'super_blocks_L3': l3_blocks})
            del l2_blocks, global_probs, l3_blocks
            gc.collect()

        print(f"\nL3 assembly complete in {time.time()-start:.1f}s")
        mark_stage_complete(STAGE_R6)

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE R07: L4 Assembly
    # =========================================================================
    STAGE_R7 = "R07_assembly_L4"

    if stage_complete(STAGE_R7):
        print(f"\n[RESUME] Skipping L4 assembly (checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("STAGE R07: Level 4 Hierarchical Assembly")
        print(f"{'='*60}")
        start = time.time()

        for r_name in region_keys:
            if contig_done(STAGE_R7, r_name):
                print(f"  [RESUME] {r_name} already done")
                continue
            print(f"\n  Processing {r_name}...")

            r6 = load_contig(STAGE_R6, r_name)
            l3_blocks = strip_block_probs(r6['super_blocks_L3'])
            del r6

            if len(l3_blocks) < 2:
                print("    Only 1 L3 block -- no L4 needed.")
                l4_blocks = l3_blocks
            else:
                global_probs, global_sites = load_global_arrays(r_name)

                print(f"    Input: {len(l3_blocks)} L3 super-blocks")

                l4_blocks = hierarchical_assembly.run_hierarchical_step(
                    l3_blocks, global_probs, global_sites,
                    batch_size=10, use_hmm_linking=True, recomb_rate=5e-8,
                    beam_width=200, max_founders=12, cc_scale=0.5,
                    num_processes=n_processes, maxtasksperchild=WORKER_MAXTASKS,
                    n_generations=3, verbose=False)
                del global_probs

            hap_counts = [len(b.haplotypes) for b in l4_blocks]
            print(f"    Output: {len(l4_blocks)} L4 super-blocks, haps: {hap_counts}")

            pipeline_runtime.strip_block_evidence(l4_blocks)
            save_contig(STAGE_R7, r_name, {'super_blocks_L4': l4_blocks})
            del l3_blocks, l4_blocks
            gc.collect()

        print(f"\nL4 assembly complete in {time.time()-start:.1f}s")
        mark_stage_complete(STAGE_R7)
#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE R08: Terminal whole-bin cavity refinement (canonical final panel)
    # =========================================================================
    # R07 is the raw L4 intermediate; R08 publishes the only downstream panel.
    STAGE_R8 = "R08_terminal_cavity"

    missing_terminal = [r for r in region_keys if not contig_done(STAGE_R8, r)]
    if stage_complete(STAGE_R8) and missing_terminal:
        raise RuntimeError(
            f"{STAGE_R8} is marked complete but lacks: {missing_terminal}"
        )
    if stage_complete(STAGE_R8):
        print("\n[RESUME] Skipping terminal cavity refinement "
              "(checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("STAGE R08: Terminal Cavity Refinement (canonical final panel)")
        print(f"{'='*60}")
        start = time.time()
        terminal_threads = min(
            n_processes,
            pipeline_runtime.available_cpu_count(),
        )
        print(f"  Sequential contigs; {terminal_threads} Numba threads/contig")

        for r_name in region_keys:
            if contig_done(STAGE_R8, r_name):
                print(f"  [RESUME] {r_name} already done")
                continue
            print(f"\n  [Terminal] Processing {r_name}...")

            r7 = load_contig(STAGE_R7, r_name)
            l4_blocks = strip_block_probs(r7['super_blocks_L4'])
            del r7
            if len(l4_blocks) != 1:
                raise RuntimeError(
                    f"{r_name}: terminal refinement requires exactly one "
                    f"chromosome-length L4 block; found {len(l4_blocks)}"
                )
            global_probs, global_sites = load_global_arrays(r_name)

            final_blocks, diagnostics = (
                terminal_cavity_refinement.refine_terminal_cavity_blocks(
                    l4_blocks,
                    global_sites,
                    global_probs,
                    return_diagnostics=True,
                    num_threads=terminal_threads,
                )
            )
            strip_block_probs(final_blocks)
            summary = (
                terminal_cavity_refinement.summarize_terminal_cavity_results(
                    diagnostics
                )
            )
            pipeline_runtime.strip_block_evidence(final_blocks)
            save_contig(STAGE_R8, r_name, {
                'super_blocks_L4': final_blocks,
                'terminal_cavity_summary': summary,
            })
            if not contig_done(STAGE_R8, r_name):
                raise OSError(f"Failed to checkpoint {STAGE_R8}/{r_name}")
            print(
                f"    Changed {summary['changed_founder_cells']} founder "
                f"cells at {summary['changed_sites']} sites"
            )
            del l4_blocks, global_probs, global_sites, final_blocks, diagnostics
            gc.collect()

        mark_stage_complete(STAGE_R8)
        print(f"Terminal refinement complete in {time.time()-start:.1f}s")


#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE R09: Viterbi Painting
    # =========================================================================
    STAGE_R9 = "R09_viterbi_painting"

    missing_painting = [
        r for r in region_keys if not contig_done(STAGE_R9, r)
    ]
    if stage_complete(STAGE_R9) and missing_painting:
        raise RuntimeError(
            f"{STAGE_R9} is marked complete but lacks: "
            f"{missing_painting}"
        )

    if stage_complete(STAGE_R9):
        print(f"\n[RESUME] Skipping Viterbi painting (checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("STAGE R09: Viterbi Painting (Real Data)")
        print(f"{'='*60}")
        start = time.time()

        with paint_samples.PaintingPoolManager(num_processes=n_processes) as painter:
            for r_name in region_keys:
                if contig_done(STAGE_R9, r_name):
                    print(f"  [RESUME] {r_name} already done")
                    continue

                print(f"\n  [Viterbi Painting] Processing Region: {r_name}")

                terminal_payload = load_contig(STAGE_R8, r_name)
                final_blocks = terminal_payload['super_blocks_L4']
                if len(final_blocks) != 1:
                    raise RuntimeError(
                        f"{r_name}: painting requires exactly one final L4 "
                        f"block; found {len(final_blocks)}"
                    )
                discovered_block = final_blocks[0]
                del terminal_payload, final_blocks

                global_probs, global_sites = load_global_arrays(r_name)

                painting_result = painter.paint_chromosome(
                    discovered_block, global_probs, global_sites,
                    recomb_rate=5e-8, switch_penalty_per_snp=1.0, batch_size=1)

                # Population painting visualization
                print(f"  Generating Population Painting Plot...")
                plot_filename = os.path.join(output_dir, f"{r_name}_viterbi_population.png")
                paint_samples.plot_population_painting(
                    painting_result, output_file=plot_filename,
                    title=f"Viterbi Painting - {r_name}",
                    sample_names=sample_names, figsize_width=20,
                    row_height_per_sample=0.25)

                founder_block = pipeline_runtime.compact_founder_block(
                    discovered_block
                )
                save_contig(STAGE_R9, r_name, {
                    'tolerance_result': painting_result,
                    pipeline_runtime.FOUNDER_BLOCK_KEY: founder_block,
                    pipeline_runtime.SAMPLE_IDS_KEY: tuple(
                        str(value) for value in sample_names
                    ),
                })
                del discovered_block, founder_block, global_probs, painting_result
                gc.collect()

        missing_painting = [
            r for r in region_keys if not contig_done(STAGE_R9, r)
        ]
        if missing_painting:
            raise OSError(f"Failed to checkpoint {STAGE_R9}: {missing_painting}")
        print(f"\nViterbi painting complete in {time.time()-start:.1f}s")
        mark_stage_complete(STAGE_R9)

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE R10: Pedigree Inference
    # =========================================================================
    STAGE_R10 = "R10_pedigree_inference"

    if stage_complete(STAGE_R10) and not checkpoint_store.global_done(STAGE_R10):
        raise RuntimeError(f"{STAGE_R10} is complete but lacks _global")
    if stage_complete(STAGE_R10):
        print(f"\n[RESUME] Skipping pedigree inference (checkpoint found)")
        pedigree_df = load_global(STAGE_R10)['pedigree_df']
    else:
        print(f"\n{'='*60}")
        print("STAGE R10: Multi-Contig Pedigree Inference (Real Data)")
        print(f"{'='*60}")

        contig_inputs = []
        for r_name in region_keys:
            painting_payload = load_contig(STAGE_R9, r_name)
            pipeline_runtime.validate_painting_bundle(
                painting_payload,
                expected_sample_ids=sample_names,
                context=f"{STAGE_R9}/{r_name}",
            )
            founder_block = pipeline_runtime.compact_founder_block(
                painting_payload[pipeline_runtime.FOUNDER_BLOCK_KEY]
            )
            entry = {
                'tolerance_painting': painting_payload['tolerance_result'],
                'founder_block': founder_block
            }
            contig_inputs.append(entry)
            del painting_payload

        start = time.time()
        pedigree_result = pedigree_inference.infer_pedigree_multi_contig_tolerance(
            contig_inputs, sample_ids=sample_names, top_k=20, n_workers=n_processes)
        print(f"\nPedigree inference time: {time.time()-start:.1f}s")

        pedigree_df = pedigree_result.relationships

        gen_counts = pedigree_df['Generation'].value_counts()
        print(f"\n--- Pedigree Summary ---")
        print(f"Generations: {gen_counts.to_dict()}")
        n_with_parents = pedigree_df['Parent1'].notna().sum()
        print(f"Individuals with parents: {n_with_parents}/{len(pedigree_df)}")

        output_csv = os.path.join(output_dir, "pedigree_inference_real.csv")
        pedigree_df.to_csv(output_csv, index=False)
        print(f"Pedigree saved to: {output_csv}")

        output_tree = os.path.join(output_dir, "pedigree_tree_real.png")
        pedigree_inference.draw_pedigree_tree(pedigree_df, output_file=output_tree)

        save_global(STAGE_R10, {'pedigree_df': pedigree_df})
        if not checkpoint_store.global_done(STAGE_R10):
            raise OSError(f"Failed to checkpoint {STAGE_R10}/_global")
        del contig_inputs
        gc.collect()
        mark_stage_complete(STAGE_R10)

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE R11: Phase Correction + Greedy Refinement + F1 Recoloring + Propagation
    # =========================================================================
    STAGE_R11 = "R11_phase_correction"

    missing_phase = [
        r for r in region_keys if not contig_done(STAGE_R11, r)
    ]
    if stage_complete(STAGE_R11) and missing_phase:
        raise RuntimeError(
            f"{STAGE_R11} is marked complete but lacks: "
            f"{missing_phase}"
        )

    if stage_complete(STAGE_R11):
        print(f"\n[RESUME] Skipping phase correction (checkpoint found)")
    else:
        print("\n" + "="*60)
        print("STAGE R11: Phase Correction (Real Data)")
        print("="*60)

        if 'pedigree_df' not in dir():
            pedigree_df = load_global(STAGE_R10)['pedigree_df']

        # _load_contig_for_phase_correction is defined at MODULE top level
        # (above `if __name__`) so forkserver workers can pickle a reference
        # to it; a closure defined here would not survive pickling.

        # Lightweight dict — just contig names, workers load their own data
        mcr = {r_name: {} for r_name in region_keys}

        # Step 1: Viterbi phase correction (6 rounds, workers load via load_fn)
        start = time.time()
        mcr = phase_correction.correct_phase_all_contigs(
            mcr, pedigree_df, sample_names, num_rounds=6, verbose=True,
            max_workers=n_processes, load_fn=_load_contig_for_phase_correction)
        print(f"Phase correction time: {time.time()-start:.1f}s")

        # Step 2: Greedy phase refinement
        print("\n" + "="*60)
        print("Greedy Phase Refinement (HOM->HET boundary flips)")
        print("="*60)

        start_refine = time.time()
        mcr = phase_correction.post_process_phase_greedy_all_contigs(
            mcr, pedigree_df, sample_names,
            snps_per_bin=100, recomb_rate=5e-8, mismatch_cost=4.6,
            max_workers=n_processes, load_fn=_load_contig_for_phase_correction,
            verbose=True)
        print(f"Greedy refinement time: {time.time()-start_refine:.1f}s")

        # Pre-load founder_blocks for the (main-process) F1 recoloring +
        # propagation steps below.  The greedy workers no longer return
        # founder_block (IPC-cost fix), so load it here into mcr in parallel
        # via threads (parallel disk I/O across contigs).
        _t0 = time.time()
        founder_blocks = pipeline_runtime.load_founder_blocks_parallel(
            checkpoint_store,
            region_keys,
            ((STAGE_R9, pipeline_runtime.FOUNDER_BLOCK_KEY),),
            max_workers=n_processes,
            require_all=True,
        )
        for r_name, founder_block in founder_blocks.items():
            mcr.setdefault(r_name, {})['founder_block'] = founder_block
        del founder_blocks
        print(f"  Founder block parallel load: {time.time()-_t0:.1f}s")

        # Step 3: Parsimonious F1 recoloring
        print("\n" + "="*60)
        print("Parsimonious F1 Recoloring")
        print("="*60)

        for r_name in region_keys:
            if r_name not in mcr:
                continue
            data = mcr[r_name]
            painting_key = 'refined_painting' if 'refined_painting' in data else 'corrected_painting'
            if painting_key not in data or 'founder_block' not in data:
                continue

            recolored = phase_correction.apply_parsimonious_f1_recoloring(
                data[painting_key], data['founder_block'],
                pedigree_df, sample_names,
                max_workers=n_processes, max_mismatch_rate=0.02, verbose=True)
            data['final_painting'] = recolored

        # Step 4: Propagate recoloring to offspring
        print("\n" + "="*60)
        print("Propagate Recoloring to Offspring")
        print("="*60)

        for r_name in region_keys:
            if r_name not in mcr:
                continue
            data = mcr[r_name]
            if 'final_painting' not in data or 'founder_block' not in data:
                continue

            propagated = phase_correction.propagate_recoloring_to_offspring(
                data['final_painting'], data['founder_block'],
                pedigree_df, sample_names,
                max_workers=n_processes, max_mismatch_rate=0.02, verbose=True)
            data['final_painting'] = propagated

        # Save per-contig results
        for r_name in region_keys:
            if r_name in mcr:
                d = {k: mcr[r_name][k]
                     for k in ('corrected_painting', 'refined_painting',
                               'final_painting', 'founder_block')
                     if k in mcr[r_name]}
                save_contig(STAGE_R11, r_name, d)

        missing_phase = [
            r for r in region_keys if not contig_done(STAGE_R11, r)
        ]
        if missing_phase:
            raise OSError(f"Failed to checkpoint {STAGE_R11}: {missing_phase}")

        del mcr
        gc.collect()
        mark_stage_complete(STAGE_R11)

#%%
if __name__ == '__main__':
    elapsed = time.time() - total_pipeline_start
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)

    print(f"\n{'='*60}")
    print("REAL DATA PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {hours}h {minutes}m ({elapsed:.0f}s)")
    print(f"Checkpoints: {CHECKPOINT_DIR}/")
    print(f"Results: {output_dir}/")
    print(f"Regions processed: {len(region_keys)}")