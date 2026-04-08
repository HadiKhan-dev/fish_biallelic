#%%
if __name__ == '__main__':
    import os
    import sys
    from datetime import datetime

    # FORCE NUMPY/BLAS TO USE 1 THREAD PER PROCESS
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # =============================================================================
    # DUAL LOGGING: Console + File
    # =============================================================================
    class TeeOutput:
        """Writes to both the original stdout and a log file."""
        def __init__(self, log_path, original_stdout):
            object.__setattr__(self, '_log_file', open(log_path, 'a', buffering=1))
            object.__setattr__(self, '_original', original_stdout)
        def write(self, message):
            self._original.write(message)
            try: self._log_file.write(message)
            except (ValueError, OSError): pass
            return None
        def flush(self):
            self._original.flush()
            try: self._log_file.flush()
            except (ValueError, OSError): pass
        def close(self):
            self._log_file.close()
        def __getattr__(self, name):
            return getattr(self._original, name)

    os.makedirs("logs", exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join("logs", f"run_real_{run_timestamp}.log")
    sys.stdout = TeeOutput(log_path, sys.stdout)
    print(f"Logging to: {log_path}")
    print(f"Run started: {run_timestamp}")

    import numpy as np
    import pandas as pd
    import time
    import warnings
    import platform
    import pickle
    import gc
    from cyvcf2 import VCF

    warnings.filterwarnings("ignore")
    np.seterr(divide='ignore', invalid='ignore')

    import thread_config
    import vcf_data_loader
    import block_haplotypes
    import block_haplotype_refinement
    import residual_discovery
    import hierarchical_assembly
    import paint_samples
    import pedigree_inference
    import phase_correction
    import analysis_utils

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    if platform.system() != "Windows":
        print(f"Main process ({os.getpid()}) niceness set to: {os.nice(0)}")

    n_processes = 112

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
    vcf_path = "./fish_vcf/AsAc.AulStuGenome.biallelic.bcf.gz"

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

    CHECKPOINT_DIR = ".pipeline_checkpoints_real"
    output_dir = "results_real"

    # =========================================================================
    # Checkpoint Infrastructure
    # =========================================================================
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    def _stage_dir(stage):
        d = os.path.join(CHECKPOINT_DIR, stage)
        os.makedirs(d, exist_ok=True)
        return d

    def stage_complete(stage):
        return os.path.exists(os.path.join(_stage_dir(stage), '_done'))

    def mark_stage_complete(stage):
        with open(os.path.join(_stage_dir(stage), '_done'), 'w') as f:
            f.write(datetime.now().isoformat())
        print(f"  [Checkpoint] Stage '{stage}' marked complete")

    def contig_done(stage, r_name):
        return os.path.exists(os.path.join(_stage_dir(stage), f'{r_name}.pkl'))

    def save_contig(stage, r_name, data_dict):
        path = os.path.join(_stage_dir(stage), f'{r_name}.pkl')
        tmp = path + ".tmp"
        try:
            with open(tmp, 'wb') as f:
                pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
            sz = os.path.getsize(path) / (1024*1024)
            print(f"    [Checkpoint] {stage}/{r_name} ({sz:.1f} MB)")
        except OSError as e:
            print(f"    [Checkpoint] WARNING: {stage}/{r_name}: {e}")
            try: os.unlink(tmp)
            except OSError: pass

    def load_contig(stage, r_name):
        path = os.path.join(_stage_dir(stage), f'{r_name}.pkl')
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save_global(stage, data_dict):
        path = os.path.join(_stage_dir(stage), '_global.pkl')
        tmp = path + ".tmp"
        try:
            with open(tmp, 'wb') as f:
                pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
            sz = os.path.getsize(path) / (1024*1024)
            print(f"    [Checkpoint] {stage}/_global ({sz:.1f} MB)")
        except OSError as e:
            print(f"    [Checkpoint] WARNING: {stage}/_global: {e}")
            try: os.unlink(tmp)
            except OSError: pass

    def load_global(stage):
        path = os.path.join(_stage_dir(stage), '_global.pkl')
        with open(path, 'rb') as f:
            return pickle.load(f)

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

            all_positions, all_reads = [], []
            for i in range(len(genomic_data)):
                pos_i = genomic_data.positions[i]
                reads_i = genomic_data.reads[i]
                if len(pos_i) > 0:
                    all_positions.append(pos_i)
                    all_reads.append(reads_i)

            if not all_positions:
                print(f"    WARNING: No data for {r_name}, skipping")
                continue

            global_sites = np.concatenate(all_positions)
            global_reads = np.concatenate(all_reads, axis=1)

            _, unique_idx = np.unique(global_sites, return_index=True)
            unique_idx = np.sort(unique_idx)
            global_sites = global_sites[unique_idx]
            global_reads = global_reads[:, unique_idx, :]

            (site_priors, global_probs) = analysis_utils.reads_to_probabilities(global_reads)
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

        REFINEMENT_DEPTH_THRESHOLD = 10.0
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
            block_results = r1['block_results']
            del r1

            print(f"\n  {r_name}: average read depth = {avg_depth:.1f}x")

            if avg_depth < REFINEMENT_DEPTH_THRESHOLD:
                print(f"  Depth < {REFINEMENT_DEPTH_THRESHOLD}x -> Running L1+L2 refinement")
                num_samples = global_probs.shape[0]
                chimera_resolution.warmup_jit(num_samples)

                def make_l1_fn(gp, gs):
                    def l1_fn(input_blocks):
                        return hierarchical_assembly.run_hierarchical_step(
                            input_blocks=input_blocks, global_probs=gp, global_sites=gs,
                            batch_size=REFINEMENT_BATCH_SIZE, use_hmm_linking=False,
                            beam_width=200, max_founders=12, max_sites_for_linking=2000,
                            cc_scale=0.2, num_processes=n_processes)
                    return l1_fn

                def make_l2_fn(gp, gs):
                    def l2_fn(input_blocks):
                        return hierarchical_assembly.run_hierarchical_step(
                            input_blocks=input_blocks, global_probs=gp, global_sites=gs,
                            batch_size=REFINEMENT_BATCH_SIZE, use_hmm_linking=True,
                            recomb_rate=RECOMB_RATE, beam_width=200, max_founders=12,
                            cc_scale=0.2, num_processes=n_processes,
                            n_generations=N_GENERATIONS, verbose=False)
                    return l2_fn

                t0 = time.time()
                refinement_results = block_haplotype_refinement.run_refinement_pipeline(
                    raw_blocks=block_results, global_probs=global_probs,
                    global_sites=global_sites, num_samples=num_samples,
                    run_l1_assembly_fn=make_l1_fn(global_probs, global_sites),
                    run_l2_assembly_fn=make_l2_fn(global_probs, global_sites),
                    batch_size=REFINEMENT_BATCH_SIZE, penalty_scale=REFINEMENT_PENALTY_SCALE,
                    recomb_rate=RECOMB_RATE, n_generations=N_GENERATIONS, verbose=True)
                print(f"\n  Refinement complete in {time.time()-t0:.0f}s")

                l2_refined = refinement_results['l2_refined']
                l2_refined_dd = block_haplotype_refinement.dedup_blocks(l2_refined, verbose=True)
                save_contig(STAGE_R2, r_name, {'block_results': l2_refined_dd})
            else:
                print(f"  Depth >= {REFINEMENT_DEPTH_THRESHOLD}x -> Skipping refinement")
                save_contig(STAGE_R2, r_name, {'block_results': block_results})

            del block_results, global_probs
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
            blocks = r2['block_results']
            del r2

            r1 = load_contig(STAGE_R1, r_name)
            global_probs = r1['global_probs']
            global_sites = r1['global_sites']
            del r1

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
            block_results = r3['block_results']
            del r3

            r1 = load_contig(STAGE_R1, r_name)
            global_probs = r1['global_probs']
            global_sites = r1['global_sites']
            del r1

            print(f"    Input: {len(block_results)} blocks")

            super_blocks = hierarchical_assembly.run_hierarchical_step(
                block_results, global_probs, global_sites,
                batch_size=10, use_hmm_linking=False, beam_width=200,
                max_founders=12, max_sites_for_linking=2000, cc_scale=0.2,
                num_processes=n_processes, verbose=False)

            hap_counts = [len(b.haplotypes) for b in super_blocks]
            print(f"    Output: {len(super_blocks)} L1 super-blocks, "
                  f"haps: min={min(hap_counts)}, max={max(hap_counts)}, "
                  f"mean={np.mean(hap_counts):.1f}")

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
            l1_blocks = r4['super_blocks_L1']
            del r4

            r1 = load_contig(STAGE_R1, r_name)
            global_probs = r1['global_probs']
            global_sites = r1['global_sites']
            del r1

            print(f"    Input: {len(l1_blocks)} L1 super-blocks")

            l2_blocks = hierarchical_assembly.run_hierarchical_step(
                l1_blocks, global_probs, global_sites,
                batch_size=10, use_hmm_linking=True, recomb_rate=5e-8,
                beam_width=200, max_founders=12, cc_scale=0.2,
                num_processes=n_processes, n_generations=3, verbose=False)

            hap_counts = [len(b.haplotypes) for b in l2_blocks]
            print(f"    Output: {len(l2_blocks)} L2 super-blocks, haps: {hap_counts}")

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
            l2_blocks = r5['super_blocks_L2']
            del r5

            r1 = load_contig(STAGE_R1, r_name)
            global_probs = r1['global_probs']
            global_sites = r1['global_sites']
            del r1

            print(f"    Input: {len(l2_blocks)} L2 super-blocks")

            l3_blocks = hierarchical_assembly.run_hierarchical_step(
                l2_blocks, global_probs, global_sites,
                batch_size=10, use_hmm_linking=True, recomb_rate=5e-8,
                beam_width=200, max_founders=12, cc_scale=0.2,
                num_processes=n_processes, n_generations=3, verbose=False)

            hap_counts = [len(b.haplotypes) for b in l3_blocks]
            print(f"    Output: {len(l3_blocks)} L3 super-blocks, haps: {hap_counts}")

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
            l3_blocks = r6['super_blocks_L3']
            del r6

            if len(l3_blocks) < 2:
                print("    Only 1 L3 block -- no L4 needed.")
                l4_blocks = l3_blocks
            else:
                r1 = load_contig(STAGE_R1, r_name)
                global_probs = r1['global_probs']
                global_sites = r1['global_sites']
                del r1

                print(f"    Input: {len(l3_blocks)} L3 super-blocks")

                l4_blocks = hierarchical_assembly.run_hierarchical_step(
                    l3_blocks, global_probs, global_sites,
                    batch_size=10, use_hmm_linking=True, recomb_rate=5e-8,
                    beam_width=200, max_founders=12, cc_scale=0.2,
                    num_processes=n_processes, n_generations=3, verbose=False)
                del global_probs

            hap_counts = [len(b.haplotypes) for b in l4_blocks]
            print(f"    Output: {len(l4_blocks)} L4 super-blocks, haps: {hap_counts}")

            save_contig(STAGE_R7, r_name, {'super_blocks_L4': l4_blocks})
            del l3_blocks, l4_blocks
            gc.collect()

        print(f"\nL4 assembly complete in {time.time()-start:.1f}s")
        mark_stage_complete(STAGE_R7)

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE R08: Viterbi Painting
    # =========================================================================
    STAGE_R8 = "R08_viterbi_painting"

    if stage_complete(STAGE_R8):
        print(f"\n[RESUME] Skipping Viterbi painting (checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("STAGE R08: Viterbi Painting (Real Data)")
        print(f"{'='*60}")
        start = time.time()

        with paint_samples.PaintingPoolManager(num_processes=n_processes) as painter:
            for r_name in region_keys:
                if contig_done(STAGE_R8, r_name):
                    print(f"  [RESUME] {r_name} already done")
                    continue

                print(f"\n  [Viterbi Painting] Processing Region: {r_name}")

                r7 = load_contig(STAGE_R7, r_name)
                discovered_block = r7['super_blocks_L4'][0]
                del r7

                r1 = load_contig(STAGE_R1, r_name)
                global_probs = r1['global_probs']
                global_sites = r1['global_sites']
                del r1

                painting_result = painter.paint_chromosome(
                    discovered_block, global_probs, global_sites,
                    recomb_rate=5e-8, switch_penalty=10.0, batch_size=1)

                # Population painting visualization
                print(f"  Generating Population Painting Plot...")
                plot_filename = os.path.join(output_dir, f"{r_name}_viterbi_population.png")
                paint_samples.plot_population_painting(
                    painting_result, output_file=plot_filename,
                    title=f"Viterbi Painting - {r_name}",
                    sample_names=sample_names, figsize_width=20,
                    row_height_per_sample=0.25)

                save_contig(STAGE_R8, r_name, {'tolerance_result': painting_result})
                del discovered_block, global_probs, painting_result
                gc.collect()

        print(f"\nViterbi painting complete in {time.time()-start:.1f}s")
        mark_stage_complete(STAGE_R8)

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE R09: Pedigree Inference
    # =========================================================================
    STAGE_R9 = "R09_pedigree_inference"

    if stage_complete(STAGE_R9):
        print(f"\n[RESUME] Skipping pedigree inference (checkpoint found)")
        pedigree_df = load_global(STAGE_R9)['pedigree_df']
    else:
        print(f"\n{'='*60}")
        print("STAGE R09: Multi-Contig Pedigree Inference (Real Data)")
        print(f"{'='*60}")

        contig_inputs = []
        for r_name in region_keys:
            r8 = load_contig(STAGE_R8, r_name)
            r7 = load_contig(STAGE_R7, r_name)
            entry = {
                'tolerance_painting': r8['tolerance_result'],
                'founder_block': r7['super_blocks_L4'][0]
            }
            contig_inputs.append(entry)
            del r8, r7

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

        save_global(STAGE_R9, {'pedigree_df': pedigree_df})
        del contig_inputs
        gc.collect()
        mark_stage_complete(STAGE_R9)

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE R10: Phase Correction + Greedy Refinement + F1 Recoloring
    # =========================================================================
    STAGE_R10 = "R10_phase_correction"

    if stage_complete(STAGE_R10):
        print(f"\n[RESUME] Skipping phase correction (checkpoint found)")
    else:
        print("\n" + "="*60)
        print("STAGE R10: Phase Correction (Real Data)")
        print("="*60)

        if 'pedigree_df' not in dir():
            pedigree_df = load_global(STAGE_R9)['pedigree_df']

        mcr = {}
        for r_name in region_keys:
            r7 = load_contig(STAGE_R7, r_name)
            r8 = load_contig(STAGE_R8, r_name)
            mcr[r_name] = {
                'tolerance_result': r8['tolerance_result'],
                'founder_block': r7['super_blocks_L4'][0],
            }
            del r7, r8

        # Step 1: Viterbi phase correction (3 rounds)
        start = time.time()
        mcr = phase_correction.correct_phase_all_contigs(
            mcr, pedigree_df, sample_names, num_rounds=3, verbose=True)
        print(f"Phase correction time: {time.time()-start:.1f}s")

        # Step 2: Greedy phase refinement
        print("\n" + "="*60)
        print("Greedy Phase Refinement (HOM->HET boundary flips)")
        print("="*60)

        start_refine = time.time()
        mcr = phase_correction.post_process_phase_greedy_all_contigs(
            mcr, pedigree_df, sample_names,
            snps_per_bin=100, recomb_rate=5e-8, mismatch_cost=4.6, verbose=True)
        print(f"Greedy refinement time: {time.time()-start_refine:.1f}s")

        # Step 3: Parsimonious F1 recoloring
        print("\n" + "="*60)
        print("Parsimonious F1 Recoloring")
        print("="*60)

        for r_name in region_keys:
            if r_name not in mcr:
                continue
            data = mcr[r_name]
            painting_key = 'refined_painting' if 'refined_painting' in data else 'corrected_painting'
            if painting_key not in data:
                continue

            recolored = phase_correction.apply_parsimonious_f1_recoloring(
                data[painting_key], data['founder_block'],
                pedigree_df, sample_names,
                max_mismatch_rate=0.02, verbose=True)
            data['final_painting'] = recolored

        # Save per-contig results
        for r_name in region_keys:
            if r_name in mcr:
                d = {k: mcr[r_name][k]
                     for k in ('corrected_painting', 'refined_painting',
                               'final_painting', 'founder_block')
                     if k in mcr[r_name]}
                save_contig(STAGE_R10, r_name, d)

        del mcr
        gc.collect()
        mark_stage_complete(STAGE_R10)

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