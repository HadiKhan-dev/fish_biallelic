#%%
if __name__ == '__main__':
    import os
    import sys
    from datetime import datetime

    # FORCE NUMPY/BLAS TO USE 1 THREAD PER PROCESS
    # (Numba threading is now managed by thread_config.py — do NOT set
    #  NUMBA_NUM_THREADS or NUMBA_THREADING_LAYER here)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # =============================================================================
    # DUAL LOGGING: Console + File
    # =============================================================================
    # All print() output goes to both the terminal and a timestamped log file.
    # tqdm progress bars still display on the terminal only (they use stderr).
    # If the SSH connection drops, the log file preserves all output.

    class TeeOutput:
        """Writes to both the original stdout and a log file.
        
        Proxies all attributes from the original stdout so that VS Code's
        IPython kernel still recognises the object as a valid output stream.
        """
        def __init__(self, log_path, original_stdout):
            # Use object.__setattr__ to avoid triggering our __getattr__
            object.__setattr__(self, '_log_file', open(log_path, 'a', buffering=1))
            object.__setattr__(self, '_original', original_stdout)
        
        def write(self, message):
            self._original.write(message)
            try:
                self._log_file.write(message)
            except (ValueError, OSError):
                pass  # log file closed or disk error — don't break output
            return getattr(self._original, 'write', lambda m: len(m))(message) if False else None
        
        def flush(self):
            self._original.flush()
            try:
                self._log_file.flush()
            except (ValueError, OSError):
                pass
        
        def close(self):
            self._log_file.close()
        
        def __getattr__(self, name):
            # Proxy everything else (encoding, fileno, isatty, etc.) to original
            return getattr(self._original, name)

    os.makedirs("logs", exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join("logs", f"run_{run_timestamp}.log")
    sys.stdout = TeeOutput(log_path, sys.stdout)
    print(f"Logging to: {log_path}")
    print(f"Run started: {run_timestamp}")


    import numpy as np
    import pandas as pd
    import time
    import warnings
    import platform
    import importlib
    import math
    import pickle
    import gc
    from tqdm import tqdm
    from dataclasses import dataclass
    from typing import Dict
    from multiprocess import Pool


    import vcf_data_loader
    import analysis_utils
    import hap_statistics
    import block_haplotypes
    import block_linking_naive
    import block_linking
    import simulate_sequences
    import hmm_matching
    import viterbi_likelihood_calculator
    import beam_search_core
    import chimera_resolution
    import hierarchical_assembly
    import block_haplotype_refinement
    import paint_samples
    import pedigree_inference
    import phase_correction
    import residual_discovery


    warnings.filterwarnings("ignore")
    np.seterr(divide='ignore', invalid="ignore")

    if platform.system() != "Windows":
        #os.nice(15)
        print(f"Main process ({os.getpid()}) niceness set to: {os.nice(0)}")


    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    n_processes = 112
    # Recycle workers after each batch to prevent memory accumulation
    # from glibc malloc fragmentation (Python doesn't return freed pages to OS).
    WORKER_MAXTASKS = 1

    # -------------------------------------------------------------------------
    # REPRODUCIBILITY: Set a master seed for the simulation.
    # All random processes (pedigree structure, meiosis, read sampling) derive
    # deterministic sub-seeds from this value.  Set to None for non-reproducible
    # runs using system entropy.
    # -------------------------------------------------------------------------
    SIMULATION_SEED = 42

    # Start the forkserver NOW, before any data is loaded.
    # The forkserver process inherits only the current ~500 MB footprint
    # (imported modules), not the ~200 GB that will exist after data loading.
    # All future pools fork workers from this lightweight forkserver.
    # thread_config.py already called set_forkserver_preload().
    _warmup_pool = hierarchical_assembly.NoDaemonPool(1)
    _warmup_pool.terminate()
    _warmup_pool.join()
    del _warmup_pool
    print("Forkserver started (lightweight, pre-data).")
    print(f"Numba threading layer: {os.environ.get('NUMBA_THREADING_LAYER', 'not set')}")
    
    # =============================================================================
    # PER-CONTIG CHECKPOINTING
    # =============================================================================
    # Each stage gets a subdirectory.  Each contig gets its own .pkl file.
    # A _done marker indicates the stage completed for ALL contigs.
    #
    # On resume, _ensure_key loads ONLY the keys a stage needs from checkpoints,
    # avoiding the monolithic pickle that caused OOM.
    #
    # Memory pruning after safe points:
    #   After stage 3 -> drop simd_genomic_data
    #   After stage 4 -> drop simulated_reads
    #   After validations (before stage 9) -> drop simd_block_results, L1, L2, L3
    #   After stage 9 -> drop simd_probs, simd_priors
    #
    # To force a full re-run:  rm -rf .pipeline_checkpoints/
    # To re-run from stage N:  delete that stage's dir and all later ones.

    CHECKPOINT_DIR = ".pipeline_checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    def _stage_dir(stage):
        d = os.path.join(CHECKPOINT_DIR, stage)
        os.makedirs(d, exist_ok=True)
        return d

    def stage_complete(stage):
        return os.path.exists(os.path.join(_stage_dir(stage), "_done"))

    def mark_stage_complete(stage):
        with open(os.path.join(_stage_dir(stage), "_done"), 'w') as f:
            f.write(datetime.now().isoformat())
        print(f"  [Checkpoint] Stage '{stage}' marked complete")

    def contig_done(stage, r_name):
        return os.path.exists(os.path.join(_stage_dir(stage), f"{r_name}.pkl"))

    def save_contig(stage, r_name, data):
        path = os.path.join(_stage_dir(stage), f"{r_name}.pkl")
        tmp = path + ".tmp"
        try:
            with open(tmp, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"    [Checkpoint] {stage}/{r_name} ({size_mb:.1f} MB)")
        except OSError as e:
            print(f"    [Checkpoint] WARNING: {stage}/{r_name}: {e}")
            try: os.unlink(tmp)
            except OSError: pass

    def load_contig(stage, r_name):
        path = os.path.join(_stage_dir(stage), f"{r_name}.pkl")
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save_global(stage, data):
        path = os.path.join(_stage_dir(stage), "_global.pkl")
        tmp = path + ".tmp"
        try:
            with open(tmp, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  [Checkpoint] {stage}/_global ({size_mb:.1f} MB)")
        except OSError as e:
            print(f"  [Checkpoint] WARNING: {stage}/_global: {e}")
            try: os.unlink(tmp)
            except OSError: pass

    def load_global(stage):
        path = os.path.join(_stage_dir(stage), "_global.pkl")
        with open(path, 'rb') as f:
            return pickle.load(f)

    # Which stage checkpoint holds each per-contig key.
    # Values can be a single stage string or a list (tried in order, first hit wins).
    # simd_block_results lives in 03 before refinement, 04 after refinement,
    # and 04b after residual discovery.
    _KEY_SOURCE = {
        'naive_long_haps':    '01_vcf_discovery',
        'simulated_reads':    '02_simulation',
        'simd_genomic_data':  '02_simulation',
        'simd_probs':         '02_simulation',
        'simd_priors':        '02_simulation',
        'truth_painting':     '02_simulation',
        'simd_block_results': ['05_residual_discovery', '04_refinement', '03_block_haplotypes'],
        'super_blocks_L1':    '06_assembly_L1',
        'super_blocks_L2':    '07_assembly_L2',
        'super_blocks_L3':    '08_assembly_L3',
        'super_blocks_L4':    '09_assembly_L4',
        'tolerance_result':   '10_viterbi_painting',
    }

    def _ensure_key(r_name, key):
        """Load a key from its checkpoint into multi_contig_results if not present."""
        mcr = multi_contig_results.setdefault(r_name, {})
        if key not in mcr:
            sources = _KEY_SOURCE[key]
            if isinstance(sources, str):
                sources = [sources]
            for src in sources:
                if contig_done(src, r_name):
                    ckpt = load_contig(src, r_name)
                    if key in ckpt:
                        mcr[key] = ckpt[key]
                        del ckpt
                        return
                    del ckpt
            raise FileNotFoundError(
                f"Cannot find '{key}' for {r_name} in any of {sources}"
            )

    def _prune_key(key):
        """Remove a key from all contigs to free RAM."""
        n = 0
        for r_name in list(multi_contig_results.keys()):
            if key in multi_contig_results.get(r_name, {}):
                del multi_contig_results[r_name][key]; n += 1
        if n > 0:
            gc.collect()
            print(f"  [Prune] Dropped '{key}' from {n} contigs")

#%%
if __name__ == '__main__':
    vcf_path = "./fish_vcf/AsAc.AulStuGenome.biallelic.bcf.gz"

    # Define the regions you want to use for inference.
    regions_config = [
        {"contig": "chr1", "start": 0, "end": 3000},
        {"contig": "chr2", "start": 0, "end": 3000},
        {"contig": "chr3", "start": 0, "end": 3000},
        {"contig": "chr4", "start": 0, "end": 3000},
        {"contig": "chr5", "start": 0, "end": 3000},
        {"contig": "chr6", "start": 0, "end": 3000},
        {"contig": "chr7", "start": 0, "end": 3000},
        {"contig": "chr8", "start": 0, "end": 3000},
        {"contig": "chr9", "start": 0, "end": 3000},
        {"contig": "chr10", "start": 0, "end": 3000},
        {"contig": "chr11", "start": 0, "end": 3000},
        {"contig": "chr12", "start": 0, "end": 3000},
        {"contig": "chr13", "start": 0, "end": 3000},
        {"contig": "chr14", "start": 0, "end": 3000},
        {"contig": "chr15", "start": 0, "end": 3000},
        {"contig": "chr16", "start": 0, "end": 3000},
        {"contig": "chr17", "start": 0, "end": 3000},
        {"contig": "chr18", "start": 0, "end": 3000},
        {"contig": "chr19", "start": 0, "end": 3000},
        {"contig": "chr20", "start": 0, "end": 3000},
        {"contig": "chr22", "start": 0, "end": 3000},
        {"contig": "chr23", "start": 0, "end": 3000},
        ]

    block_size = 100000
    shift_size = 50000

    multi_contig_results = {}

    total_start = time.time()

    # =========================================================================
    # STAGE 1: VCF Loading + Haplotype Discovery + Naive Linking
    # =========================================================================
    STAGE_1 = "01_vcf_discovery"

    if stage_complete(STAGE_1):
        print(f"\n[RESUME] Skipping VCF loading + discovery (checkpoint found)")
        # naive_long_haps loaded on-demand via _ensure_key
    else:
        for region in regions_config:
            r_name = region['contig']
            if contig_done(STAGE_1, r_name):
                print(f"  [RESUME] {r_name} already done")
                continue

            print(f"\n" + "="*60)
            print(f"PROCESSING REGION: ({region['contig']} blocks {region['start']}-{region['end']})")
            print("="*60)
            
            # 1. Load Data
            start = time.time()
            genomic_data = vcf_data_loader.cleanup_block_reads_list(
                vcf_path, 
                region['contig'],
                start_block_idx=region['start'],
                end_block_idx=region['end'],
                block_size=block_size,
                shift_size=shift_size,
                num_processes=16
            )
            print(f"  [Loader] Loaded {len(genomic_data)} blocks in {time.time() - start:.2f}s")

            # 2. Run Haplotype Discovery
            start = time.time()
            block_results = block_haplotypes.generate_all_block_haplotypes(genomic_data,
                                                                           num_processes=n_processes)

            valid_blocks = [b for b in block_results if len(b.positions) > 0]
            block_results = block_haplotypes.BlockResults(valid_blocks)
            
            print(f"  [Discovery] Haplotypes generated in {time.time() - start:.2f}s")

            # 3. Run Naive Linker (to get long templates for simulation)
            start = time.time()
            (naive_blocks, naive_long_haps) = block_linking_naive.generate_long_haplotypes_naive(
                block_results, 
                num_long_haps=6
            )
            print(f"  [Naive Linker] Chained {len(naive_long_haps[1])} haps in {time.time() - start:.2f}s")
            
            # Store only naive_long_haps (genomic_data + block_results are huge, never needed again)
            multi_contig_results[region['contig']] = {
                "naive_long_haps": naive_long_haps
            }
            save_contig(STAGE_1, r_name, {'naive_long_haps': naive_long_haps})
            del genomic_data, block_results, naive_blocks, naive_long_haps
            gc.collect()

        print(f"\nAll regions processed in {time.time() - total_start:.2f}s")
        mark_stage_complete(STAGE_1)

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE 2: Simulation + Post-processing
    # =========================================================================
    STAGE_2 = "02_simulation"

    if stage_complete(STAGE_2):
        print(f"\n[RESUME] Skipping simulation (checkpoint found)")
        g = load_global(STAGE_2)
        truth_pedigree = g['truth_pedigree']
        sample_names = g['sample_names']
        region_keys = g['region_keys']
        del g
        # Per-contig data loaded on-demand via _ensure_key
    else:
        start = time.time()
        output_dir = "results_simulation"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError:
            pass

        # 1. Prepare Founders and Sites for ALL regions
        founders_list = []
        sites_list = []
        region_keys = []

        for r_name in [r['contig'] for r in regions_config]:
            _ensure_key(r_name, 'naive_long_haps')
            data = multi_contig_results[r_name]
            sites, haps_data = data['naive_long_haps']
            concrete_haps = simulate_sequences.concretify_haps(haps_data)
            parents = simulate_sequences.pairup_haps(concrete_haps)
            founders_list.append(parents)
            sites_list.append(sites)
            region_keys.append(r_name)

        # 2. Run Multi-Contig Simulation
        generation_sizes = [20, 100, 200]
        print(f"Running Multi-Contig Simulation for {len(region_keys)} regions...")

        STRESS_TEST_MUTATIONS = False
        if STRESS_TEST_MUTATIONS:
            mutate_rate = 1e-5
            print(f"STRESS TEST MODE: Using mutation rate {mutate_rate} (~1% per generation)")
        else:
            mutate_rate = 1e-10
            print(f"Normal mode: Using mutation rate {mutate_rate} (minimal mutations)")

        t0 = time.time()
        all_offspring_lists, truth_pedigree, truth_paintings_lists = simulate_sequences.simulate_pedigree(
            founders_list, 
            sites_list, 
            generation_sizes, 
            recomb_rate=5e-8, 
            mutate_rate=mutate_rate,
            output_plot=None,
            parallel=True,
            num_processes=n_processes,
            seed=SIMULATION_SEED
        )
        print(f"Pedigree simulation: {time.time()-t0:.1f}s")

        # 3. Save Truth
        try:
            truth_csv_path = os.path.join(output_dir, "ground_truth_pedigree.csv")
            truth_pedigree.to_csv(truth_csv_path, index=False)
            print(f"Ground Truth Pedigree data saved to '{truth_csv_path}'")
        except OSError:
            print("WARNING: Could not save truth CSV (disk full)")

        sample_names = truth_pedigree['Sample'].tolist()

        # 4. Process All Contigs in Parallel (read sampling, chunking, probs)
        t0 = time.time()
        contig_results = simulate_sequences.process_all_contigs_parallel(
            region_keys, all_offspring_lists, truth_paintings_lists, sites_list,
            read_depth=5, error_rate=0.02,
            snps_per_block=200, snp_shift=200,
            num_processes=n_processes,
            seed=(SIMULATION_SEED + 1_000_000) if SIMULATION_SEED is not None else None
        )
        
        for r_name in region_keys:
            result = contig_results[r_name]
            multi_contig_results[r_name]['simulated_reads'] = result['simulated_reads']
            multi_contig_results[r_name]['simd_genomic_data'] = result['simd_genomic_data']
            multi_contig_results[r_name]['simd_probs'] = result['simd_probs']
            multi_contig_results[r_name]['simd_priors'] = result['simd_priors']
            multi_contig_results[r_name]['truth_painting'] = result['truth_painting']
            save_contig(STAGE_2, r_name, {
                'simulated_reads': result['simulated_reads'],
                'simd_genomic_data': result['simd_genomic_data'],
                'simd_probs': result['simd_probs'],
                'simd_priors': result['simd_priors'],
                'truth_painting': result['truth_painting'],
            })
        
        print(f"Post-processing ({len(region_keys)} contigs parallel): {time.time()-t0:.1f}s")

        print("\nSimulation, Sequencing, and Chunking complete for all regions.")
        print(f"Total time: {time.time()-start:.1f}s")

        save_global(STAGE_2, {
            'truth_pedigree': truth_pedigree,
            'sample_names': sample_names,
            'region_keys': region_keys,
        })
        # Free heavy simulation data — all checkpointed, will reload on demand
        for r_name in region_keys:
            for _k in ('simulated_reads', 'simd_genomic_data', 'simd_probs', 'simd_priors', 'truth_painting'):
                multi_contig_results[r_name].pop(_k, None)
        del contig_results; gc.collect()
        mark_stage_complete(STAGE_2)
    
#%%
if __name__ == '__main__':
    # Ensure output_dir and globals exist for all subsequent stages
    output_dir = "results_simulation"
    os.makedirs(output_dir, exist_ok=True)

    if 'region_keys' not in dir() or region_keys is None:
        g = load_global('02_simulation')
        region_keys = g['region_keys']
        sample_names = g['sample_names']
        truth_pedigree = g['truth_pedigree']
        del g

    # =========================================================================
    # STAGE 3: Discover Block Haplotypes from Simulated Reads
    # =========================================================================
    STAGE_3 = "03_block_haplotypes"

    if stage_complete(STAGE_3):
        print(f"\n[RESUME] Skipping block haplotype discovery (checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("Discovering Block Haplotypes from Simulated Reads")
        print(f"{'='*60}")

        start = time.time()

        for r_name in region_keys:
            if contig_done(STAGE_3, r_name):
                print(f"  [RESUME] {r_name} already done")
                continue
            print(f"\n  Processing {r_name}...")
            
            _ensure_key(r_name, 'simd_genomic_data')
            simd_genomic_data = multi_contig_results[r_name]['simd_genomic_data']
            
            simd_block_results = block_haplotypes.generate_all_block_haplotypes(
                simd_genomic_data,
                uniqueness_threshold_percent=1.0,
                diff_threshold_percent=0.5,
                wrongness_threshold=1.0,
                num_processes=n_processes
            )
            
            valid_blocks = [b for b in simd_block_results if len(b.positions) > 0]
            simd_block_results = block_haplotypes.BlockResults(valid_blocks)
            
            multi_contig_results[r_name]['simd_block_results'] = simd_block_results
            save_contig(STAGE_3, r_name, {'simd_block_results': simd_block_results})
            
            hap_counts = [len(b.haplotypes) for b in valid_blocks]
            print(f"    {len(valid_blocks)} blocks, haps/block: "
                  f"min={min(hap_counts)}, max={max(hap_counts)}, mean={np.mean(hap_counts):.1f}")

            # Free this contig's data immediately (don't accumulate across contigs)
            for _k in ('simd_genomic_data', 'simd_block_results'):
                multi_contig_results[r_name].pop(_k, None)

        print(f"\nBlock haplotype discovery complete in {time.time()-start:.1f}s")
        _prune_key('simd_genomic_data')
        mark_stage_complete(STAGE_3)

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE 4: Conditional Refinement (if average read depth < 10)
    # =========================================================================
    STAGE_4 = "04_refinement"

    if stage_complete(STAGE_4):
        print(f"\n[RESUME] Skipping refinement (checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("Checking Read Depth for Refinement")
        print(f"{'='*60}")

        REFINEMENT_DEPTH_THRESHOLD = 10.0
        REFINEMENT_BATCH_SIZE = 10
        REFINEMENT_PENALTY_SCALE = 20.0
        RECOMB_RATE = 5e-8
        N_GENERATIONS = 3

        for r_name in region_keys:
            if contig_done(STAGE_4, r_name):
                print(f"  [RESUME] {r_name} already done")
                continue
            _ensure_key(r_name, 'simulated_reads')
            _ensure_key(r_name, 'simd_block_results')
            _ensure_key(r_name, 'simd_probs')
            _ensure_key(r_name, 'naive_long_haps')
            reads = multi_contig_results[r_name]['simulated_reads']
            avg_depth = np.mean(np.sum(reads, axis=-1))
            print(f"\n  {r_name}: average read depth = {avg_depth:.1f}x")
            
            if avg_depth < REFINEMENT_DEPTH_THRESHOLD:
                print(f"  Depth < {REFINEMENT_DEPTH_THRESHOLD}x → Running L1+L2 refinement")
                
                raw_blocks = multi_contig_results[r_name]['simd_block_results']
                global_probs = multi_contig_results[r_name]['simd_probs']
                global_sites = multi_contig_results[r_name]['naive_long_haps'][0]
                num_samples = global_probs.shape[0]
                
                # Warmup numba JIT
                chimera_resolution.warmup_jit(num_samples)
                
                # Define assembly functions as callables
                def make_l1_fn(gp, gs):
                    def l1_fn(input_blocks):
                        return hierarchical_assembly.run_hierarchical_step(
                            input_blocks=input_blocks,
                            global_probs=gp,
                            global_sites=gs,
                            batch_size=REFINEMENT_BATCH_SIZE,
                            use_hmm_linking=False,
                            beam_width=200,
                            max_founders=12,
                            max_sites_for_linking=2000,
                            cc_scale=0.5,
                            num_processes=n_processes,
                            maxtasksperchild=WORKER_MAXTASKS
                        )
                    return l1_fn
                
                def make_l2_fn(gp, gs):
                    def l2_fn(input_blocks):
                        return hierarchical_assembly.run_hierarchical_step(
                            input_blocks=input_blocks,
                            global_probs=gp,
                            global_sites=gs,
                            batch_size=REFINEMENT_BATCH_SIZE,
                            use_hmm_linking=True,
                            recomb_rate=RECOMB_RATE,
                            beam_width=200,
                            max_founders=12,
                            cc_scale=0.5,
                            num_processes=n_processes,
                            n_generations=N_GENERATIONS,
                            verbose=False,
                            maxtasksperchild=WORKER_MAXTASKS
                        )
                    return l2_fn
                
                l1_fn = make_l1_fn(global_probs, global_sites)
                l2_fn = make_l2_fn(global_probs, global_sites)
                
                # Run full refinement pipeline
                t0 = time.time()
                refinement_results = block_haplotype_refinement.run_refinement_pipeline(
                    raw_blocks=raw_blocks,
                    global_probs=global_probs,
                    global_sites=global_sites,
                    num_samples=num_samples,
                    run_l1_assembly_fn=l1_fn,
                    run_l2_assembly_fn=l2_fn,
                    batch_size=REFINEMENT_BATCH_SIZE,
                    penalty_scale=REFINEMENT_PENALTY_SCALE,
                    recomb_rate=RECOMB_RATE,
                    n_generations=N_GENERATIONS,
                    verbose=True
                )
                print(f"\n  Refinement complete in {time.time()-t0:.0f}s")
                
                # Replace raw blocks with L2-refined blocks
                l2_refined = refinement_results['l2_refined']
                
                # Dedup before feeding into main assembly
                l2_refined_dd = block_haplotype_refinement.dedup_blocks(l2_refined, verbose=True)
                
                # Store refined blocks as the new starting point
                multi_contig_results[r_name]['simd_block_results'] = l2_refined_dd
                multi_contig_results[r_name]['refinement_results'] = refinement_results
                
                print(f"  Raw blocks updated with L2-refined blocks")
            else:
                print(f"  Depth >= {REFINEMENT_DEPTH_THRESHOLD}x → Skipping refinement")

            save_contig(STAGE_4, r_name, {
                'simd_block_results': multi_contig_results[r_name]['simd_block_results']
            })

            # Free this contig's heavy input data immediately
            for _k in ('simulated_reads', 'simd_probs', 'simd_block_results', 'refinement_results'):
                multi_contig_results[r_name].pop(_k, None)

        _prune_key('simulated_reads')
        _prune_key('simd_probs')
        mark_stage_complete(STAGE_4)

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE 5: Residual Discovery (find missing founders HDBSCAN missed)
    # =========================================================================
    STAGE_5 = "05_residual_discovery"

    if stage_complete(STAGE_5):
        print(f"\n[RESUME] Skipping residual discovery (checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("Residual Discovery (Missing Founder Recovery)")
        print(f"{'='*60}")

        start = time.time()

        for r_name in region_keys:
            if contig_done(STAGE_5, r_name):
                print(f"  [RESUME] {r_name} already done")
                continue
            print(f"\n  Processing {r_name}...")

            _ensure_key(r_name, 'simd_block_results')
            _ensure_key(r_name, 'simd_probs')
            _ensure_key(r_name, 'naive_long_haps')
            blocks = multi_contig_results[r_name]['simd_block_results']
            global_probs = multi_contig_results[r_name]['simd_probs']
            global_sites = multi_contig_results[r_name]['naive_long_haps'][0]

            print(f"    Input: {len(blocks)} blocks, "
                  f"avg haps: {np.mean([len(b.haplotypes) for b in blocks]):.1f}")

            blocks_out = residual_discovery.discover_missing_haplotypes(
                blocks, global_probs, global_sites,
                min_residual_reduction=0.10,
                num_processes=n_processes,
                verbose=True
            )

            multi_contig_results[r_name]['simd_block_results'] = blocks_out
            save_contig(STAGE_5, r_name, {'simd_block_results': blocks_out})

            print(f"    Output: {len(blocks_out)} blocks, "
                  f"avg haps: {np.mean([len(b.haplotypes) for b in blocks_out]):.1f}")

            # Free this contig's data immediately
            for _k in ('simd_probs', 'simd_block_results'):
                multi_contig_results[r_name].pop(_k, None)

        print(f"\nResidual discovery complete in {time.time()-start:.1f}s")
        mark_stage_complete(STAGE_5)

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE 6: Hierarchical Assembly (Level 1)
    # =========================================================================
    STAGE_6 = "06_assembly_L1"

    if stage_complete(STAGE_6):
        print(f"\n[RESUME] Skipping L1 assembly (checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("Level 1 Hierarchical Assembly")
        print(f"{'='*60}")

        start = time.time()

        for r_name in region_keys:
            if contig_done(STAGE_6, r_name):
                print(f"  [RESUME] {r_name} already done")
                continue
            print(f"\n  Processing {r_name}...")
            
            _ensure_key(r_name, 'simd_block_results')
            _ensure_key(r_name, 'simd_probs')
            _ensure_key(r_name, 'naive_long_haps')
            simd_block_results = multi_contig_results[r_name]['simd_block_results']
            global_probs = multi_contig_results[r_name]['simd_probs']
            global_sites = multi_contig_results[r_name]['naive_long_haps'][0]

            print(f"    Input: {len(simd_block_results)} blocks")
            
            super_blocks = hierarchical_assembly.run_hierarchical_step(
                input_blocks=simd_block_results,
                global_probs=global_probs,
                global_sites=global_sites,
                batch_size=10,
                use_hmm_linking=False,
                beam_width=200,
                max_founders=12,
                max_sites_for_linking=2000,
                cc_scale=0.5,
                num_processes=n_processes,
                maxtasksperchild=WORKER_MAXTASKS
            )
            
            multi_contig_results[r_name]['super_blocks_L1'] = super_blocks
            save_contig(STAGE_6, r_name, {'super_blocks_L1': super_blocks})
            
            hap_counts = [len(b.haplotypes) for b in super_blocks]
            total_sites = sum(len(b.positions) for b in super_blocks)
            print(f"\n    Output: {len(super_blocks)} super-blocks")
            print(f"    Total sites: {total_sites}")
            print(f"    Haps per super-block: min={min(hap_counts)}, max={max(hap_counts)}, "
                  f"mean={np.mean(hap_counts):.1f}")

            # Free this contig's input data (will reload from checkpoint if needed)
            for _k in ('simd_block_results', 'simd_probs', 'super_blocks_L1'):
                multi_contig_results[r_name].pop(_k, None)

        print(f"\nHierarchical Assembly (Level 1) complete in {time.time()-start:.1f}s")
        _prune_key('simd_block_results')
        mark_stage_complete(STAGE_6)

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE 7: Hierarchical Assembly (Level 2)
    # =========================================================================
    STAGE_7 = "07_assembly_L2"

    if stage_complete(STAGE_7):
        print(f"\n[RESUME] Skipping L2 assembly (checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("Level 2 Hierarchical Assembly")
        print(f"{'='*60}")

        start_time = time.time()

        for r_name in region_keys:
            if contig_done(STAGE_7, r_name):
                print(f"  [RESUME] {r_name} already done")
                continue
            print(f"\n  Processing {r_name}...")
            
            _ensure_key(r_name, 'super_blocks_L1')
            _ensure_key(r_name, 'simd_probs')
            _ensure_key(r_name, 'naive_long_haps')
            super_blocks_L1 = multi_contig_results[r_name]['super_blocks_L1']
            global_probs = multi_contig_results[r_name]['simd_probs']
            global_sites = multi_contig_results[r_name]['naive_long_haps'][0]
            
            print(f"    Input: {len(super_blocks_L1)} L1 super-blocks")
            
            super_blocks_L2 = hierarchical_assembly.run_hierarchical_step(
                super_blocks_L1,
                global_probs,
                global_sites,
                batch_size=10,
                use_hmm_linking=True,
                recomb_rate=5e-8,
                beam_width=200,
                max_founders=12,
                cc_scale=0.5,
                num_processes=n_processes,
                maxtasksperchild=WORKER_MAXTASKS,
                n_generations=3,
                verbose=False
            )
            
            multi_contig_results[r_name]['super_blocks_L2'] = super_blocks_L2
            save_contig(STAGE_7, r_name, {'super_blocks_L2': super_blocks_L2})
            
            haps_per_block = [len(b.haplotypes) for b in super_blocks_L2]
            total_sites = sum(len(b.positions) for b in super_blocks_L2)
            print(f"\n    Output: {len(super_blocks_L2)} L2 super-blocks")
            print(f"    Total sites: {total_sites}")
            print(f"    Haps per super-block: min={min(haps_per_block)}, max={max(haps_per_block)}, "
                  f"mean={np.mean(haps_per_block):.1f}")

            # Free this contig's input data
            for _k in ('super_blocks_L1', 'simd_probs', 'super_blocks_L2'):
                multi_contig_results[r_name].pop(_k, None)

        print(f"\nHierarchical Assembly (Level 2) complete in {time.time()-start_time:.1f}s")
        _prune_key('super_blocks_L1')
        mark_stage_complete(STAGE_7)

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE 8: Hierarchical Assembly (Level 3)
    # =========================================================================
    STAGE_8 = "08_assembly_L3"

    if stage_complete(STAGE_8):
        print(f"\n[RESUME] Skipping L3 assembly (checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("Level 3 Hierarchical Assembly")
        print(f"{'='*60}")

        start_time = time.time()

        for r_name in region_keys:
            if contig_done(STAGE_8, r_name):
                print(f"  [RESUME] {r_name} already done")
                continue
            print(f"\n  Processing {r_name}...")
            
            _ensure_key(r_name, 'super_blocks_L2')
            _ensure_key(r_name, 'simd_probs')
            _ensure_key(r_name, 'naive_long_haps')
            super_blocks_L2 = multi_contig_results[r_name]['super_blocks_L2']
            global_probs = multi_contig_results[r_name]['simd_probs']
            global_sites = multi_contig_results[r_name]['naive_long_haps'][0]
            
            print(f"    Input: {len(super_blocks_L2)} L2 super-blocks")
            
            super_blocks_L3 = hierarchical_assembly.run_hierarchical_step(
                super_blocks_L2,
                global_probs,
                global_sites,
                batch_size=10,
                use_hmm_linking=True,
                recomb_rate=5e-8,
                beam_width=200,
                max_founders=12,
                cc_scale=0.5,
                num_processes=n_processes,
                maxtasksperchild=WORKER_MAXTASKS,
                n_generations=3,
                verbose=False
            )
            
            multi_contig_results[r_name]['super_blocks_L3'] = super_blocks_L3
            save_contig(STAGE_8, r_name, {'super_blocks_L3': super_blocks_L3})
            
            haps_per_block = [len(b.haplotypes) for b in super_blocks_L3]
            print(f"\n    Output: {len(super_blocks_L3)} L3 super-blocks")
            print(f"    Sites per block: {[len(b.positions) for b in super_blocks_L3]}")
            print(f"    Haps per super-block: {haps_per_block}")

            # Free this contig's input data
            for _k in ('super_blocks_L2', 'simd_probs', 'super_blocks_L3'):
                multi_contig_results[r_name].pop(_k, None)

        print(f"\nHierarchical Assembly (Level 3) complete in {time.time()-start_time:.1f}s")
        _prune_key('super_blocks_L2')
        mark_stage_complete(STAGE_8)

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE 9: Hierarchical Assembly (Level 4)
    # =========================================================================
    STAGE_9 = "09_assembly_L4"

    if stage_complete(STAGE_9):
        print(f"\n[RESUME] Skipping L4 assembly (checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("Level 4 Hierarchical Assembly")
        print(f"{'='*60}")

        start_time = time.time()

        for r_name in region_keys:
            if contig_done(STAGE_9, r_name):
                print(f"  [RESUME] {r_name} already done")
                continue
            print(f"\n  Processing {r_name}...")
            
            _ensure_key(r_name, 'super_blocks_L3')
            super_blocks_L3 = multi_contig_results[r_name]['super_blocks_L3']
            
            if len(super_blocks_L3) < 2:
                print("    Only 1 L3 block — no L4 needed.")
                multi_contig_results[r_name]['super_blocks_L4'] = super_blocks_L3
            else:
                print(f"    Input: {len(super_blocks_L3)} L3 super-blocks")
                
                _ensure_key(r_name, 'simd_probs')
                _ensure_key(r_name, 'naive_long_haps')
                global_probs = multi_contig_results[r_name]['simd_probs']
                global_sites = multi_contig_results[r_name]['naive_long_haps'][0]
                
                super_blocks_L4 = hierarchical_assembly.run_hierarchical_step(
                    super_blocks_L3,
                    global_probs,
                    global_sites,
                    batch_size=10,
                    use_hmm_linking=True,
                    recomb_rate=5e-8,
                    beam_width=200,
                    max_founders=12,
                    cc_scale=0.5,
                    num_processes=n_processes,
                    maxtasksperchild=WORKER_MAXTASKS,
                    n_generations=3,
                    verbose=False
                )
                
                multi_contig_results[r_name]['super_blocks_L4'] = super_blocks_L4
                
                haps_per_block = [len(b.haplotypes) for b in super_blocks_L4]
                print(f"\n    Output: {len(super_blocks_L4)} L4 super-blocks")
                print(f"    Sites per block: {[len(b.positions) for b in super_blocks_L4]}")
                print(f"    Haps per super-block: {haps_per_block}")

            save_contig(STAGE_9, r_name, {
                'super_blocks_L4': multi_contig_results[r_name]['super_blocks_L4']
            })

            # Free this contig's input data
            for _k in ('super_blocks_L3', 'simd_probs', 'super_blocks_L4'):
                multi_contig_results[r_name].pop(_k, None)

        print(f"\nHierarchical Assembly (Level 4) complete in {time.time()-start_time:.1f}s")
        _prune_key('super_blocks_L3')
        mark_stage_complete(STAGE_9)

#%%
if __name__ == '__main__':
    # ==========================================================================
    # VALIDATE: Block Haplotypes Against Ground Truth
    # ==========================================================================
    # Validation stages are fast and read-only — no checkpointing needed.
    print(f"\n{'='*60}")
    print("Validating Discovered Block Haplotypes Against Ground Truth")
    print(f"{'='*60}")

    def validate_block_haplotypes(simd_block_results, orig_sites, orig_haps_concrete):
        """
        Compare discovered block haplotypes against true founder haplotypes.
        """
        orig_site_to_idx = {s: i for i, s in enumerate(orig_sites)}
        block_stats = []
        
        for block in simd_block_results:
            block_positions = block.positions
            block_haps = block.haplotypes
            
            if len(block_positions) == 0:
                continue
            
            common_indices = []
            block_indices = []
            for bi, pos in enumerate(block_positions):
                if pos in orig_site_to_idx:
                    common_indices.append(orig_site_to_idx[pos])
                    block_indices.append(bi)
            
            if len(common_indices) == 0:
                continue
            
            true_at_block = [h[common_indices] for h in orig_haps_concrete]
            num_true_founders = len(true_at_block)
            
            discovered_at_block = []
            for hap_idx, hap_arr in block_haps.items():
                concrete = np.argmax(hap_arr, axis=1)
                discovered_at_block.append(concrete[block_indices])
            
            num_discovered = len(discovered_at_block)
            
            true_to_best_discovered = []
            for ti, true_h in enumerate(true_at_block):
                best_diff = 100.0
                best_idx = -1
                for di, disc_h in enumerate(discovered_at_block):
                    diff = np.mean(true_h != disc_h) * 100
                    if diff < best_diff:
                        best_diff = diff
                        best_idx = di
                true_to_best_discovered.append((ti, best_idx, best_diff))
            
            discovered_to_best_true = []
            for di, disc_h in enumerate(discovered_at_block):
                best_diff = 100.0
                best_idx = -1
                for ti, true_h in enumerate(true_at_block):
                    diff = np.mean(true_h != disc_h) * 100
                    if diff < best_diff:
                        best_diff = diff
                        best_idx = ti
                discovered_to_best_true.append((di, best_idx, best_diff))
            
            founders_found = sum(1 for _, _, diff in true_to_best_discovered if diff < 2.0)
            avg_true_match_error = np.mean([diff for _, _, diff in true_to_best_discovered])
            avg_disc_match_error = np.mean([diff for _, _, diff in discovered_to_best_true])
            
            block_stats.append({
                'start_pos': block_positions[0],
                'n_sites': len(common_indices),
                'n_true': num_true_founders,
                'n_discovered': num_discovered,
                'founders_found': founders_found,
                'avg_true_match_err': avg_true_match_error,
                'avg_disc_match_err': avg_disc_match_error,
                'true_matches': true_to_best_discovered,
                'disc_matches': discovered_to_best_true
            })
        
        return block_stats


    for r_name in region_keys:
        print(f"\n{r_name}:")
        
        _ensure_key(r_name, 'simd_block_results')
        _ensure_key(r_name, 'naive_long_haps')
        simd_block_results = multi_contig_results[r_name]['simd_block_results']
        orig_sites, orig_haps = multi_contig_results[r_name]['naive_long_haps']
        orig_haps_concrete = simulate_sequences.concretify_haps(orig_haps)
        
        block_stats = validate_block_haplotypes(simd_block_results, orig_sites, orig_haps_concrete)
        multi_contig_results[r_name]['block_validation_stats'] = block_stats
        
        n_blocks = len(block_stats)
        avg_discovered = np.mean([bs['n_discovered'] for bs in block_stats])
        avg_founders_found = np.mean([bs['founders_found'] for bs in block_stats])
        avg_true_err = np.mean([bs['avg_true_match_err'] for bs in block_stats])
        avg_disc_err = np.mean([bs['avg_disc_match_err'] for bs in block_stats])
        all_found_count = sum(1 for bs in block_stats if bs['founders_found'] == bs['n_true'])
        
        print(f"  Blocks analyzed: {n_blocks}")
        print(f"  True founders per block: {block_stats[0]['n_true']}")
        print(f"  Avg discovered haps per block: {avg_discovered:.1f}")
        print(f"  Avg founders found per block (<2% diff): {avg_founders_found:.1f} / {block_stats[0]['n_true']}")
        print(f"  Blocks with ALL founders found: {all_found_count} / {n_blocks} ({100*all_found_count/n_blocks:.1f}%)")
        print(f"  Avg best-match error (true->discovered): {avg_true_err:.2f}%")
        print(f"  Avg best-match error (discovered->true): {avg_disc_err:.2f}%")
        
        founders_found_dist = {}
        for bs in block_stats:
            ff = bs['founders_found']
            founders_found_dist[ff] = founders_found_dist.get(ff, 0) + 1
        
        print(f"  Founders found distribution:")
        for k in sorted(founders_found_dist.keys()):
            print(f"    {k} founders: {founders_found_dist[k]} blocks ({100*founders_found_dist[k]/n_blocks:.1f}%)")

        # Free — already checkpointed, validation is read-only
        for _k in ('simd_block_results', 'block_validation_stats'):
            multi_contig_results[r_name].pop(_k, None)

    print(f"\n{'='*60}")
    print("Block Haplotype Validation Complete")
    print(f"{'='*60}")

#%%
if __name__ == '__main__':
    # ==========================================================================
    # VALIDATE: Level 1 Super Blocks
    # ==========================================================================
    print(f"\n{'='*60}")
    print("Validating Level 1 Super Blocks against Ground Truth")
    print(f"{'='*60}")

    for r_name in region_keys:
        print(f"\n{r_name}:")
        
        _ensure_key(r_name, 'super_blocks_L1')
        _ensure_key(r_name, 'naive_long_haps')
        super_blocks = multi_contig_results[r_name]['super_blocks_L1']
        orig_sites, orig_haps = multi_contig_results[r_name]['naive_long_haps']
        orig_haps_concrete = simulate_sequences.concretify_haps(orig_haps)
        orig_site_to_idx = {s: i for i, s in enumerate(orig_sites)}
        num_true_founders = len(orig_haps_concrete)
        
        total_discovered = 0
        total_good = 0
        total_chimeras = 0
        blocks_with_all_founders = 0
        chimera_details = []
        
        for block_idx, block in enumerate(super_blocks):
            positions = block.positions
            true_at_block = [np.array([orig_haps_concrete[f_idx][orig_site_to_idx[pos]] 
                                        for pos in positions])
                             for f_idx in range(num_true_founders)]
            
            founders_found = 0
            for tf in true_at_block:
                best_error = min(np.mean((np.argmax(hap, axis=1) if hap.ndim > 1 else hap) != tf) * 100
                               for hap in block.haplotypes.values())
                if best_error < 2.0:
                    founders_found += 1
            
            for h_idx, hap in block.haplotypes.items():
                if hap.ndim > 1:
                    hap = np.argmax(hap, axis=1)
                errors = [np.mean(hap != tf) * 100 for tf in true_at_block]
                best_f = np.argmin(errors)
                best_error = errors[best_f]
                total_discovered += 1
                if best_error < 2.0:
                    total_good += 1
                else:
                    total_chimeras += 1
                    chimera_details.append({
                        'block': block_idx, 'hap': h_idx,
                        'best_f': best_f, 'error': best_error,
                        'n_sites': len(positions)
                    })
            
            if founders_found == num_true_founders:
                blocks_with_all_founders += 1
        
        print(f"  L1 super-blocks: {len(super_blocks)}")
        print(f"  Blocks with ALL founders: {blocks_with_all_founders} / {len(super_blocks)} "
              f"({100*blocks_with_all_founders/len(super_blocks):.1f}%)")
        print(f"  Total haplotypes: {total_discovered}")
        print(f"  Good haplotypes (<2% error): {total_good}")
        print(f"  Chimeras (>2% error): {total_chimeras}")
        
        if chimera_details:
            print(f"  Chimera details:")
            for c in sorted(chimera_details, key=lambda x: x['error'], reverse=True):
                print(f"    Block {c['block']}, H{c['hap']}: F{c['best_f']} @ {c['error']:.2f}%")

        multi_contig_results[r_name].pop('super_blocks_L1', None)

#%%
if __name__ == '__main__':
    # ==========================================================================
    # VALIDATE: Level 2 Super Blocks
    # ==========================================================================
    print(f"\n{'='*60}")
    print("Validating Level 2 Super Blocks against Ground Truth")
    print(f"{'='*60}")

    for r_name in region_keys:
        print(f"\n{r_name}:")
        
        _ensure_key(r_name, 'super_blocks_L2')
        _ensure_key(r_name, 'naive_long_haps')
        super_blocks_L2 = multi_contig_results[r_name]['super_blocks_L2']
        orig_sites, orig_haps = multi_contig_results[r_name]['naive_long_haps']
        orig_haps_concrete = simulate_sequences.concretify_haps(orig_haps)
        orig_site_to_idx = {s: idx for idx, s in enumerate(orig_sites)}
        num_true_founders = len(orig_haps_concrete)
        
        total_discovered = 0
        total_good = 0
        total_chimeras = 0
        blocks_with_all_founders = 0
        
        for block_idx, block in enumerate(super_blocks_L2):
            positions = block.positions
            true_at_block = [np.array([orig_haps_concrete[f_idx][orig_site_to_idx[pos]] 
                                        for pos in positions])
                             for f_idx in range(num_true_founders)]
            
            founders_found = 0
            for tf in true_at_block:
                best_error = min(np.mean((np.argmax(hap, axis=1) if hap.ndim > 1 else hap) != tf) * 100
                               for hap in block.haplotypes.values())
                if best_error < 2.0:
                    founders_found += 1
            
            for h_idx, hap in block.haplotypes.items():
                if hap.ndim > 1:
                    hap = np.argmax(hap, axis=1)
                errors = [np.mean(hap != tf) * 100 for tf in true_at_block]
                best_error = min(errors)
                total_discovered += 1
                if best_error < 2.0:
                    total_good += 1
                else:
                    total_chimeras += 1
            
            if founders_found == num_true_founders:
                blocks_with_all_founders += 1
        
        print(f"  L2 super-blocks: {len(super_blocks_L2)}")
        print(f"  Blocks with ALL founders: {blocks_with_all_founders} / {len(super_blocks_L2)} "
              f"({100*blocks_with_all_founders/len(super_blocks_L2):.1f}%)")
        print(f"  Total haplotypes: {total_discovered}")
        print(f"  Good haplotypes (<2% error): {total_good}")
        print(f"  Chimeras (>2% error): {total_chimeras}")

        multi_contig_results[r_name].pop('super_blocks_L2', None)

#%%
if __name__ == '__main__':
    # ==========================================================================
    # VALIDATE: Level 3 Super Blocks
    # ==========================================================================
    print(f"\n{'='*60}")
    print("Validating Level 3 Super Blocks against Ground Truth")
    print(f"{'='*60}")

    for r_name in region_keys:
        print(f"\n{r_name}:")
        
        _ensure_key(r_name, 'super_blocks_L3')
        _ensure_key(r_name, 'naive_long_haps')
        super_blocks_L3 = multi_contig_results[r_name]['super_blocks_L3']
        orig_sites, orig_haps = multi_contig_results[r_name]['naive_long_haps']
        orig_haps_concrete = simulate_sequences.concretify_haps(orig_haps)
        orig_site_to_idx = {s: idx for idx, s in enumerate(orig_sites)}
        num_true_founders = len(orig_haps_concrete)
        
        total_discovered = 0
        total_good = 0
        total_chimeras = 0
        blocks_with_all_founders = 0
        chimera_details = []
        
        for block_idx, block in enumerate(super_blocks_L3):
            positions = block.positions
            true_at_block = [np.array([orig_haps_concrete[f_idx][orig_site_to_idx[pos]] 
                                        for pos in positions])
                             for f_idx in range(num_true_founders)]
            
            founders_found = 0
            for tf in true_at_block:
                best_error = min(np.mean((np.argmax(hap, axis=1) if hap.ndim > 1 else hap) != tf) * 100
                               for hap in block.haplotypes.values())
                if best_error < 2.0:
                    founders_found += 1
            
            for h_idx, hap in block.haplotypes.items():
                if hap.ndim > 1:
                    hap = np.argmax(hap, axis=1)
                errors = [np.mean(hap != tf) * 100 for tf in true_at_block]
                best_f = np.argmin(errors)
                best_error = errors[best_f]
                total_discovered += 1
                if best_error < 2.0:
                    total_good += 1
                else:
                    total_chimeras += 1
                    chimera_details.append({
                        'block': block_idx, 'hap': h_idx,
                        'best_f': best_f, 'error': best_error,
                        'n_sites': len(positions)
                    })
            
            if founders_found == num_true_founders:
                blocks_with_all_founders += 1
            
            print(f"  Block {block_idx}: {len(positions)} sites, {len(block.haplotypes)} haps, "
                  f"{founders_found}/{num_true_founders} founders")
        
        print(f"\n  Results:")
        print(f"    L3 super-blocks: {len(super_blocks_L3)}")
        print(f"    Blocks with ALL founders: {blocks_with_all_founders} / {len(super_blocks_L3)}")
        print(f"    Total haplotypes: {total_discovered}")
        print(f"    Good haplotypes (<2% error): {total_good}")
        print(f"    Chimeras (>2% error): {total_chimeras}")
        
        if chimera_details:
            print(f"    Chimera details:")
            for c in sorted(chimera_details, key=lambda x: x['error'], reverse=True):
                print(f"      Block {c['block']}, H{c['hap']}: F{c['best_f']} @ {c['error']:.2f}%")

        multi_contig_results[r_name].pop('super_blocks_L3', None)

#%%
if __name__ == '__main__':
    # ==========================================================================
    # VALIDATE: Final (Level 4) Super Blocks
    # ==========================================================================
    print(f"\n{'='*60}")
    print("Validating Final Super Blocks against Ground Truth")
    print(f"{'='*60}")

    for r_name in region_keys:
        print(f"\n{r_name}:")
        
        _ensure_key(r_name, 'super_blocks_L4')
        _ensure_key(r_name, 'naive_long_haps')
        if 'super_blocks_L4' in multi_contig_results[r_name]:
            final_blocks = multi_contig_results[r_name]['super_blocks_L4']
            level_name = "L4"
        else:
            _ensure_key(r_name, 'super_blocks_L3')
            final_blocks = multi_contig_results[r_name]['super_blocks_L3']
            level_name = "L3 (final)"
        
        orig_sites, orig_haps = multi_contig_results[r_name]['naive_long_haps']
        orig_haps_concrete = simulate_sequences.concretify_haps(orig_haps)
        orig_site_to_idx = {s: idx for idx, s in enumerate(orig_sites)}
        num_true_founders = len(orig_haps_concrete)
        
        total_discovered = 0
        total_good = 0
        total_chimeras = 0
        blocks_with_all_founders = 0
        chimera_details = []
        
        for block_idx, block in enumerate(final_blocks):
            positions = block.positions
            true_at_block = [np.array([orig_haps_concrete[f_idx][orig_site_to_idx[pos]] 
                                        for pos in positions])
                             for f_idx in range(num_true_founders)]
            
            founders_found = 0
            for tf in true_at_block:
                best_error = min(np.mean((np.argmax(hap, axis=1) if hap.ndim > 1 else hap) != tf) * 100
                               for hap in block.haplotypes.values())
                if best_error < 2.0:
                    founders_found += 1
            
            for h_idx, hap in block.haplotypes.items():
                if hap.ndim > 1:
                    hap = np.argmax(hap, axis=1)
                errors = [np.mean(hap != tf) * 100 for tf in true_at_block]
                best_f = np.argmin(errors)
                best_error = errors[best_f]
                total_discovered += 1
                if best_error < 2.0:
                    total_good += 1
                else:
                    total_chimeras += 1
                    chimera_details.append({
                        'block': block_idx, 'hap': h_idx,
                        'best_f': best_f, 'error': best_error,
                        'n_sites': len(positions)
                    })
            
            if founders_found == num_true_founders:
                blocks_with_all_founders += 1
            
            print(f"  Block {block_idx}: {len(positions)} sites, {len(block.haplotypes)} haps, "
                  f"{founders_found}/{num_true_founders} founders")
        
        print(f"\n  Final Results ({level_name}):")
        print(f"    Super-blocks: {len(final_blocks)}")
        print(f"    Blocks with ALL founders: {blocks_with_all_founders} / {len(final_blocks)}")
        print(f"    Total haplotypes: {total_discovered}")
        print(f"    Good haplotypes (<2% error): {total_good}")
        print(f"    Chimeras (>2% error): {total_chimeras}")
        
        if chimera_details:
            print(f"    Chimera details:")
            for c in sorted(chimera_details, key=lambda x: x['error'], reverse=True):
                print(f"      Block {c['block']}, H{c['hap']}: F{c['best_f']} @ {c['error']:.2f}%")

        multi_contig_results[r_name].pop('super_blocks_L4', None)


#%%




#%%
if __name__ == '__main__':
    # Safety-net prune — most keys already freed per-contig in loops above
    for _k in ('simd_block_results', 'super_blocks_L1', 'super_blocks_L2',
               'super_blocks_L3', 'super_blocks_L4', 'block_validation_stats',
               'refinement_results'):
        _prune_key(_k)
    gc.collect()

    # =============================================================================
    # STAGE 10: VITERBI PAINTING (using DISCOVERED haplotypes from L4 assembly)
    # =============================================================================
    STAGE_10 = "10_viterbi_painting"

    if stage_complete(STAGE_10):
        print(f"\n[RESUME] Skipping Viterbi painting (checkpoint found)")
    else:
        print("\n" + "="*60)
        print("RUNNING: Viterbi Painting (Discovered Haplotypes)")
        print("="*60)

        with paint_samples.PaintingPoolManager(num_processes=n_processes) as painter:
            for r_name in region_keys:
                if contig_done(STAGE_10, r_name):
                    print(f"  [RESUME] {r_name} already done")
                    continue
                print(f"\n[Viterbi Painting] Processing Region: {r_name}")

                # Retrieve Data — use L4 discovered super-block
                _ensure_key(r_name, 'super_blocks_L4')
                _ensure_key(r_name, 'simd_probs')
                _ensure_key(r_name, 'naive_long_haps')
                if 'super_blocks_L4' in multi_contig_results[r_name]:
                    discovered_block = multi_contig_results[r_name]['super_blocks_L4'][0]
                else:
                    _ensure_key(r_name, 'super_blocks_L3')
                    discovered_block = multi_contig_results[r_name]['super_blocks_L3'][0]
                
                global_probs = multi_contig_results[r_name]['simd_probs']
                sites, _ = multi_contig_results[r_name]['naive_long_haps']

                # Run Viterbi Painting (single best path, no tolerance margin)
                painting_result = painter.paint_chromosome(
                    discovered_block,
                    global_probs,
                    sites,
                    recomb_rate=5e-8,
                    switch_penalty=10.0,
                    batch_size=1
                )

                multi_contig_results[r_name]['tolerance_result'] = painting_result

                # Population painting visualization
                print(f"  Generating Population Painting Plot...")
                plot_filename = os.path.join(output_dir, f"{r_name}_viterbi_population.png")
                paint_samples.plot_population_painting(
                    painting_result,
                    output_file=plot_filename,
                    title=f"Viterbi Painting (Discovered Haplotypes) - {r_name}",
                    sample_names=sample_names,
                    figsize_width=20,
                    row_height_per_sample=0.25
                )

                save_contig(STAGE_10, r_name, {'tolerance_result': painting_result})

                # Free this contig's data immediately
                for _k in ('simd_probs', 'tolerance_result', 'super_blocks_L4'):
                    multi_contig_results[r_name].pop(_k, None)

        print("\nViterbi Painting complete.")
        _prune_key('simd_probs')
        _prune_key('simd_priors')
        mark_stage_complete(STAGE_10)

#%%
if __name__ == '__main__':
    # =============================================================================
    # STAGE 11: MULTI-CONTIG PEDIGREE INFERENCE (using DISCOVERED haplotypes)
    # =============================================================================
    STAGE_11 = "11_pedigree_inference"

    if stage_complete(STAGE_11):
        print(f"\n[RESUME] Skipping pedigree inference (checkpoint found)")
        pedigree_df = load_global(STAGE_11)['pedigree_df']
    else:
        print("\n" + "="*60)
        print("RUNNING: Multi-Contig Pedigree Inference (Discovered Haplotypes)")
        print("="*60)

        # 1. Gather Data from all regions
        contig_inputs = []
        for r_name in region_keys:
            _ensure_key(r_name, 'tolerance_result')
            _ensure_key(r_name, 'super_blocks_L4')
            if 'tolerance_result' in multi_contig_results[r_name]:
                # Use discovered L4 block instead of ground truth
                if 'super_blocks_L4' in multi_contig_results[r_name]:
                    discovered_block = multi_contig_results[r_name]['super_blocks_L4'][0]
                else:
                    _ensure_key(r_name, 'super_blocks_L3')
                    discovered_block = multi_contig_results[r_name]['super_blocks_L3'][0]
                
                entry = {
                    'tolerance_painting': multi_contig_results[r_name]['tolerance_result'],
                    'founder_block': discovered_block
                }
                contig_inputs.append(entry)
            else:
                print(f"Warning: Tolerance painting missing for {r_name}")

        # 2. Run Inference (16-State HMM with tolerance-aware scoring)
        #    n_workers uses all available cores
        #    perform_consistency_cutoff + resolve_cycles called internally
        pedigree_result = pedigree_inference.infer_pedigree_multi_contig_tolerance(
            contig_inputs, 
            sample_ids=sample_names,
            top_k=20,
            n_workers=n_processes
        )

        # 3. Save & Visualize
        pedigree_df = pedigree_result.relationships
        output_csv = os.path.join(output_dir, "pedigree_inference_discovered.csv")
        pedigree_df.to_csv(output_csv, index=False)
        print(f"Pedigree saved to: {output_csv}")

        output_tree = os.path.join(output_dir, "pedigree_tree_discovered.png")
        pedigree_inference.draw_pedigree_tree(pedigree_df, output_file=output_tree)

        # 4. Validate against Truth (if available)
        if 'truth_pedigree' in dir():
            print("\n--- Pedigree Validation ---")
            validation_df = pd.merge(
                truth_pedigree[['Sample', 'Generation', 'Parent1', 'Parent2']],
                pedigree_df[['Sample', 'Generation', 'Parent1', 'Parent2']],
                on='Sample',
                suffixes=('_True', '_Inf')
            )

            def check_parent_match(row):
                true_p = {row['Parent1_True'], row['Parent2_True']}
                true_p = {x for x in true_p if pd.notna(x)}
                inf_p = {row['Parent1_Inf'], row['Parent2_Inf']}
                inf_p = {x for x in inf_p if pd.notna(x)}
                
                # F1 check (Truth has Founders, Inf has None)
                if any("Founder" in str(x) for x in true_p):
                    return len(inf_p) == 0
                return true_p == inf_p

            validation_df['Gen_Match'] = validation_df['Generation_True'] == validation_df['Generation_Inf']
            validation_df['Parents_Match'] = validation_df.apply(check_parent_match, axis=1)

            gen_acc = validation_df['Gen_Match'].mean() * 100
            descendant_mask = validation_df['Generation_True'].isin(['F2', 'F3'])
            parent_acc = validation_df[descendant_mask]['Parents_Match'].mean() * 100

            print(f"Generation Accuracy: {gen_acc:.2f}%")
            print(f"Parentage Accuracy (F2+F3): {parent_acc:.2f}%")

        save_global(STAGE_11, {'pedigree_df': pedigree_df})
        mark_stage_complete(STAGE_11)

#%%
if __name__ == '__main__':
    # =============================================================================
    # STAGE 12: PHASE CORRECTION (using DISCOVERED haplotypes)
    # =============================================================================
    STAGE_12 = "12_phase_correction"

    if stage_complete(STAGE_12):
        print(f"\n[RESUME] Skipping phase correction (checkpoint found)")
        # Load per-contig phase correction results for validation
        for r_name in region_keys:
            s12 = load_contig(STAGE_12, r_name)
            for k, v in s12.items():
                multi_contig_results.setdefault(r_name, {})[k] = v
            del s12
    else:
        print("\n" + "="*60)
        print("RUNNING: Phase Correction (Discovered Haplotypes)")
        print("="*60)

        # Define loader callback — workers call this to load their own contig data
        # (parallelizes I/O across all worker processes)
        def _load_contig_for_phase_correction(r_name):
            """Load tolerance_result and founder_block for one contig from checkpoints."""
            data = {}
            _ensure_key(r_name, 'super_blocks_L4')
            _ensure_key(r_name, 'tolerance_result')
            data['tolerance_result'] = multi_contig_results[r_name]['tolerance_result']
            if 'super_blocks_L4' in multi_contig_results[r_name]:
                data['founder_block'] = multi_contig_results[r_name]['super_blocks_L4'][0]
            elif 'super_blocks_L3' in multi_contig_results[r_name]:
                data['founder_block'] = multi_contig_results[r_name]['super_blocks_L3'][0]
            return data

        # Ensure contig names exist in multi_contig_results (load_fn needs keys)
        for r_name in region_keys:
            multi_contig_results.setdefault(r_name, {})

        start = time.time()
        # Run phase correction — workers load their own data via load_fn
        multi_contig_results = phase_correction.correct_phase_all_contigs(
            multi_contig_results,
            pedigree_df,
            sample_names,
            num_rounds=3,
            verbose=True,
            load_fn=_load_contig_for_phase_correction
        )
        print(f"Phase correction time: {time.time()-start:.1f}s")

        # =============================================================================
        # GREEDY PHASE REFINEMENT POST-PROCESSING
        # =============================================================================
        print("\n" + "="*60)
        print("RUNNING: Greedy Phase Refinement (HOM→HET boundary flips)")
        print("="*60)

        start_refine = time.time()
        multi_contig_results = phase_correction.post_process_phase_greedy_all_contigs(
            multi_contig_results,
            pedigree_df,
            sample_names,
            snps_per_bin=100,
            recomb_rate=5e-8,
            mismatch_cost=4.6,
            verbose=True
        )
        print(f"Greedy refinement time: {time.time()-start_refine:.1f}s")

        # =============================================================================
        # PARSIMONIOUS F1 RECOLORING
        # =============================================================================
        print("\n" + "="*60)
        print("RUNNING: Parsimonious F1 Recoloring")
        print("="*60)

        for r_name in region_keys:
            if r_name not in multi_contig_results:
                continue
            data = multi_contig_results[r_name]
            painting_key = 'refined_painting' if 'refined_painting' in data else 'corrected_painting'
            if painting_key not in data or 'founder_block' not in data:
                continue

            recolored = phase_correction.apply_parsimonious_f1_recoloring(
                data[painting_key],
                data['founder_block'],
                pedigree_df,
                sample_names,
                max_mismatch_rate=0.02,
                verbose=True
            )
            data['final_painting'] = recolored

        # =============================================================================
        # PROPAGATE RECOLORING TO OFFSPRING
        # =============================================================================
        print("\n" + "="*60)
        print("RUNNING: Propagate Recoloring to Offspring")
        print("="*60)

        for r_name in region_keys:
            if r_name not in multi_contig_results:
                continue
            data = multi_contig_results[r_name]
            if 'final_painting' not in data or 'founder_block' not in data:
                continue

            propagated = phase_correction.propagate_recoloring_to_offspring(
                data['final_painting'],
                data['founder_block'],
                pedigree_df,
                sample_names,
                max_mismatch_rate=0.02,
                verbose=True
            )
            data['final_painting'] = propagated

        # Save per-contig phase correction results
        for r_name in region_keys:
            d = {k: multi_contig_results[r_name][k]
                 for k in ('corrected_painting', 'refined_painting',
                           'final_painting', 'founder_block')
                 if k in multi_contig_results[r_name]}
            save_contig(STAGE_12, r_name, d)

        # Free everything — phase validation reloads from checkpoints
        multi_contig_results = {r: {'naive_long_haps': multi_contig_results[r].get('naive_long_haps')}
                                for r in region_keys if 'naive_long_haps' in multi_contig_results.get(r, {})}
        gc.collect()
        mark_stage_complete(STAGE_12)

#%%
if __name__ == '__main__':
    # =============================================================================
    # VALIDATE PHASE CORRECTION AGAINST GROUND TRUTH (ALLELE-LEVEL)
    # =============================================================================
    print("\n" + "="*60)
    print("VALIDATING: Phase Correction vs Ground Truth (Allele-Level)")
    print("  (Using DISCOVERED haplotypes for painting, TRUE founders for validation)")
    print("="*60)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Reload paintings from stage 12 checkpoints (freed during save)
    def _load_stage12(r_name):
        if contig_done(STAGE_12, r_name):
            s12 = load_contig(STAGE_12, r_name)
            for k, v in s12.items():
                multi_contig_results.setdefault(r_name, {})[k] = v
            del s12
        return r_name

    print("\nLoading phase correction + validation data from checkpoints...")
    load_start = time.time()
    with ThreadPoolExecutor(max_workers=min(8, len(region_keys))) as executor:
        list(executor.map(_load_stage12, region_keys))
    print(f"Stage 12 reload: {time.time()-load_start:.1f}s")

    def extract_founder_ids_at_positions(painting, positions):
        n_pos = len(positions)
        hap1_ids = np.full(n_pos, -1, dtype=np.int32)
        hap2_ids = np.full(n_pos, -1, dtype=np.int32)
        chunks = painting.chunks if hasattr(painting, 'chunks') else []
        if not chunks:
            return hap1_ids, hap2_ids
        n_chunks = len(chunks)
        chunk_starts = np.array([c.start for c in chunks], dtype=np.int64)
        chunk_ends = np.array([c.end for c in chunks], dtype=np.int64)
        chunk_hap1 = np.array([c.hap1 for c in chunks], dtype=np.int32)
        chunk_hap2 = np.array([c.hap2 for c in chunks], dtype=np.int32)
        chunk_indices = np.searchsorted(chunk_ends, positions, side='right')
        chunk_indices = np.clip(chunk_indices, 0, n_chunks - 1)
        valid_mask = (positions >= chunk_starts[chunk_indices]) & (positions < chunk_ends[chunk_indices])
        hap1_ids[valid_mask] = chunk_hap1[chunk_indices[valid_mask]]
        hap2_ids[valid_mask] = chunk_hap2[chunk_indices[valid_mask]]
        return hap1_ids, hap2_ids

    def evaluate_contig_dual_founders(args):
        r_name, painting, truth, positions, disc_dense_haps, true_dense_haps, sample_names = args
        results = []
        for i, name in enumerate(sample_names):
            corrected_sample = painting[i]
            truth_sample = truth[i]
            corr_hap1, corr_hap2 = extract_founder_ids_at_positions(corrected_sample, positions)
            true_hap1, true_hap2 = extract_founder_ids_at_positions(truth_sample, positions)
            n_pos = len(positions)
            pos_indices = np.arange(n_pos)
            max_disc = disc_dense_haps.shape[0]
            corr_allele1 = np.full(n_pos, -1, dtype=np.int8)
            corr_allele2 = np.full(n_pos, -1, dtype=np.int8)
            v1 = (corr_hap1 >= 0) & (corr_hap1 < max_disc)
            v2 = (corr_hap2 >= 0) & (corr_hap2 < max_disc)
            corr_allele1[v1] = disc_dense_haps[corr_hap1[v1], pos_indices[v1]]
            corr_allele2[v2] = disc_dense_haps[corr_hap2[v2], pos_indices[v2]]
            max_true = true_dense_haps.shape[0]
            true_allele1 = np.full(n_pos, -1, dtype=np.int8)
            true_allele2 = np.full(n_pos, -1, dtype=np.int8)
            v3 = (true_hap1 >= 0) & (true_hap1 < max_true)
            v4 = (true_hap2 >= 0) & (true_hap2 < max_true)
            true_allele1[v3] = true_dense_haps[true_hap1[v3], pos_indices[v3]]
            true_allele2[v4] = true_dense_haps[true_hap2[v4], pos_indices[v4]]
            direct_match = (corr_allele1 == true_allele1) & (corr_allele2 == true_allele2)
            flipped_match = (corr_allele1 == true_allele2) & (corr_allele2 == true_allele1)
            correct_either = direct_match | flipped_match
            n_direct = np.sum(direct_match)
            n_flipped = np.sum(flipped_match)
            if n_direct >= n_flipped:
                track1_correct = (corr_allele1 == true_allele1)
                track2_correct = (corr_allele2 == true_allele2)
                dominant_phase = "Direct"
            else:
                track1_correct = (corr_allele1 == true_allele2)
                track2_correct = (corr_allele2 == true_allele1)
                dominant_phase = "Flipped"
            valid_mask = (corr_allele1 != -1) & (corr_allele2 != -1) & (true_allele1 != -1) & (true_allele2 != -1)
            n_valid = np.sum(valid_mask)
            if n_valid > 0:
                accuracy = np.sum(correct_either & valid_mask) / n_valid
                track1_acc = np.sum(track1_correct & valid_mask) / n_valid
                track2_acc = np.sum(track2_correct & valid_mask) / n_valid
            else:
                accuracy = 0.0
                track1_acc = 0.0
                track2_acc = 0.0
            results.append({
                'Sample': name, 'Total_sites': n_pos, 'Valid_sites': int(n_valid),
                'Correct_sites': int(np.sum(correct_either & valid_mask)),
                'Accuracy': accuracy, 'Track1_accuracy': track1_acc,
                'Track2_accuracy': track2_acc, 'Direct_matches': int(n_direct),
                'Flipped_matches': int(n_flipped), 'Dominant_phase': dominant_phase
            })
        contig_eval = pd.DataFrame(results)
        contig_eval['Contig'] = r_name
        return r_name, contig_eval

    print("Evaluating phase correction accuracy (allele-level)...")
    print("  Paintings use DISCOVERED haplotypes")
    print("  Validation converts to alleles using TRUE founders")

    # Build shared data for each contig (truth, positions, dense haps)
    # Pre-load any missing keys in parallel
    def _load_validation_keys(r_name):
        _ensure_key(r_name, 'truth_painting')
        _ensure_key(r_name, 'super_blocks_L4')
        _ensure_key(r_name, 'naive_long_haps')
        _ensure_key(r_name, 'tolerance_result')
        return r_name
    
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=min(8, len(region_keys))) as executor:
        list(executor.map(_load_validation_keys, region_keys))
    print(f"  Validation data loaded in {time.time()-t0:.1f}s")
    
    contig_shared = {}
    for r_name in region_keys:
        if 'truth_painting' not in multi_contig_results[r_name]:
            continue
        truth = multi_contig_results[r_name]['truth_painting']
        if 'super_blocks_L4' in multi_contig_results[r_name]:
            discovered_block = multi_contig_results[r_name]['super_blocks_L4'][0]
        else:
            discovered_block = multi_contig_results[r_name]['super_blocks_L3'][0]
        positions = discovered_block.positions
        dense_haps, _ = phase_correction.founder_block_to_dense(discovered_block)
        
        orig_sites, orig_haps = multi_contig_results[r_name]['naive_long_haps']
        orig_haps_concrete = simulate_sequences.concretify_haps(orig_haps)
        site_indices = np.searchsorted(orig_sites, positions)
        site_indices = np.clip(site_indices, 0, len(orig_sites) - 1)
        true_dense_haps = np.array([h[site_indices] for h in orig_haps_concrete], dtype=np.int8)
        
        contig_shared[r_name] = (truth, positions, dense_haps, true_dense_haps)

    # --- BEFORE: evaluate uncorrected painting (raw Viterbi output) ---
    print("\n" + "-"*60)
    print("BEFORE phase correction (raw Viterbi painting):")
    print("-"*60)

    before_args = []
    for r_name in region_keys:
        if r_name not in contig_shared:
            continue
        truth, positions, dense_haps, true_dense_haps = contig_shared[r_name]
        raw_painting = multi_contig_results[r_name].get('tolerance_result')
        if raw_painting is None:
            continue
        before_args.append((r_name, raw_painting, truth, positions, dense_haps,
                            true_dense_haps, sample_names))

    before_contig_results = []
    with ThreadPoolExecutor(max_workers=len(region_keys)) as executor:
        for r_name, contig_eval in executor.map(evaluate_contig_dual_founders, before_args):
            mean_acc = contig_eval['Accuracy'].mean()*100
            mean_t1 = contig_eval['Track1_accuracy'].mean()*100
            mean_t2 = contig_eval['Track2_accuracy'].mean()*100
            print(f"  {r_name}: Allele={mean_acc:.2f}%, Track1={mean_t1:.2f}%, Track2={mean_t2:.2f}%")
            before_contig_results.append(contig_eval)

    if before_contig_results:
        before_df = pd.concat(before_contig_results, ignore_index=True)
        before_df['Generation'] = before_df['Sample'].apply(
            lambda x: 'F1' if x.startswith('F1') else ('F2' if x.startswith('F2') else 'F3'))
    else:
        before_df = pd.DataFrame()

    # --- AFTER: evaluate corrected painting ---
    print("\n" + "-"*60)
    print("AFTER phase correction (corrected + greedy + F1 recoloring):")
    print("-"*60)

    eval_args = []
    for r_name in region_keys:
        if r_name not in contig_shared:
            continue
        truth, positions, dense_haps, true_dense_haps = contig_shared[r_name]
        if 'final_painting' in multi_contig_results[r_name]:
            painting = multi_contig_results[r_name]['final_painting']
        elif 'refined_painting' in multi_contig_results[r_name]:
            painting = multi_contig_results[r_name]['refined_painting']
        elif 'corrected_painting' in multi_contig_results[r_name]:
            painting = multi_contig_results[r_name]['corrected_painting']
        else:
            continue
        
        eval_args.append((r_name, painting, truth, positions, dense_haps, 
                          true_dense_haps, sample_names))

    all_contig_results = []
    with ThreadPoolExecutor(max_workers=len(region_keys)) as executor:
        for r_name, contig_eval in executor.map(evaluate_contig_dual_founders, eval_args):
            mean_acc = contig_eval['Accuracy'].mean()*100
            mean_t1 = contig_eval['Track1_accuracy'].mean()*100
            mean_t2 = contig_eval['Track2_accuracy'].mean()*100
            print(f"  {r_name}: Allele={mean_acc:.2f}%, Track1={mean_t1:.2f}%, Track2={mean_t2:.2f}%")
            all_contig_results.append(contig_eval)

    if all_contig_results:
        full_eval_df = pd.concat(all_contig_results, ignore_index=True)
        eval_output = os.path.join(output_dir, "phase_correction_evaluation_discovered.csv")
        try:
            full_eval_df.to_csv(eval_output, index=False)
            print(f"\nDetailed evaluation saved to: {eval_output}")
        except OSError:
            print("WARNING: Could not save evaluation CSV (disk full)")
        
        full_eval_df['Generation'] = full_eval_df['Sample'].apply(
            lambda x: 'F1' if x.startswith('F1') else ('F2' if x.startswith('F2') else 'F3')
        )
        
        # ============================================================
        # BEFORE vs AFTER COMPARISON
        # ============================================================
        print("\n" + "="*60)
        print("PHASE CORRECTION: BEFORE vs AFTER COMPARISON")
        print("="*60)
        
        if len(before_df) > 0:
            print("\nBy Generation:")
            print(f"  {'Gen':<4s}  {'Before Allele':>14s}  {'After Allele':>13s}  {'Before Track1':>14s}  {'After Track1':>13s}  {'Improvement':>12s}")
            for gen in ['F1', 'F2', 'F3']:
                b_gen = before_df[before_df['Generation'] == gen]
                a_gen = full_eval_df[full_eval_df['Generation'] == gen]
                if len(b_gen) > 0 and len(a_gen) > 0:
                    b_acc = b_gen['Accuracy'].mean()*100
                    a_acc = a_gen['Accuracy'].mean()*100
                    b_t1 = b_gen['Track1_accuracy'].mean()*100
                    a_t1 = a_gen['Track1_accuracy'].mean()*100
                    diff = a_t1 - b_t1
                    print(f"  {gen:<4s}  {b_acc:>13.2f}%  {a_acc:>12.2f}%  {b_t1:>13.2f}%  {a_t1:>12.2f}%  {diff:>+11.2f}%")
            
            b_overall_acc = before_df['Accuracy'].mean()*100
            a_overall_acc = full_eval_df['Accuracy'].mean()*100
            b_overall_t1 = before_df['Track1_accuracy'].mean()*100
            a_overall_t1 = full_eval_df['Track1_accuracy'].mean()*100
            diff_overall = a_overall_t1 - b_overall_t1
            print(f"  {'All':<4s}  {b_overall_acc:>13.2f}%  {a_overall_acc:>12.2f}%  {b_overall_t1:>13.2f}%  {a_overall_t1:>12.2f}%  {diff_overall:>+11.2f}%")
            
            # Perfect phasing comparison
            perfect_threshold = 0.999
            b_perfect = len(before_df[before_df['Track1_accuracy'] >= perfect_threshold])
            a_perfect = len(full_eval_df[full_eval_df['Track1_accuracy'] >= perfect_threshold])
            n_total = len(full_eval_df)
            print(f"\n  Perfect phasing (>=99.9% Track1):")
            print(f"    Before: {b_perfect}/{n_total} ({100*b_perfect/n_total:.1f}%)")
            print(f"    After:  {a_perfect}/{n_total} ({100*a_perfect/n_total:.1f}%)")
        
        # ============================================================
        # DETAILED AFTER RESULTS
        # ============================================================
        print("\n" + "="*60)
        print("PHASE CORRECTION RESULTS (AFTER)")
        print("="*60)
        
        print("\nAccuracy by Generation:")
        for gen in ['F1', 'F2', 'F3']:
            gen_df = full_eval_df[full_eval_df['Generation'] == gen]
            if len(gen_df) > 0:
                print(f"  {gen}: Accuracy={gen_df['Accuracy'].mean()*100:.2f}%, "
                      f"Track1={gen_df['Track1_accuracy'].mean()*100:.2f}%, "
                      f"Track2={gen_df['Track2_accuracy'].mean()*100:.2f}%, "
                      f"N={len(gen_df)}")
        
        print(f"\nOverall Accuracy:  {full_eval_df['Accuracy'].mean()*100:.2f}%")
        print(f"Overall Track1:    {full_eval_df['Track1_accuracy'].mean()*100:.2f}%")
        print(f"Overall Track2:    {full_eval_df['Track2_accuracy'].mean()*100:.2f}%")
        
        n_direct = (full_eval_df['Dominant_phase'] == 'Direct').sum()
        n_flipped = (full_eval_df['Dominant_phase'] == 'Flipped').sum()
        print(f"\nPhase assignment: {n_direct} samples Direct, {n_flipped} samples Flipped")
        
        print("\nWorst 10 samples by accuracy:")
        worst = full_eval_df.nsmallest(10, 'Accuracy')[['Sample', 'Contig', 'Accuracy', 'Track1_accuracy', 'Track2_accuracy', 'Dominant_phase']]
        worst_display = worst.copy()
        worst_display['Accuracy'] = worst_display['Accuracy'] * 100
        worst_display['Track1_accuracy'] = worst_display['Track1_accuracy'] * 100
        worst_display['Track2_accuracy'] = worst_display['Track2_accuracy'] * 100
        print(worst_display.to_string(index=False, float_format='%.2f'))
        
        print("\n" + "="*60)
        print("PERFECT PHASING SUMMARY")
        print("="*60)
        
        perfect_threshold = 0.999
        perfect_samples = full_eval_df[full_eval_df['Track1_accuracy'] >= perfect_threshold]
        n_perfect = len(perfect_samples)
        n_total = len(full_eval_df)
        
        print(f"\nSamples with >=99.9% Track1 accuracy: {n_perfect}/{n_total} ({100*n_perfect/n_total:.1f}%)")
        
        for gen in ['F1', 'F2', 'F3']:
            gen_df = full_eval_df[full_eval_df['Generation'] == gen]
            gen_perfect = gen_df[gen_df['Track1_accuracy'] >= perfect_threshold]
            if len(gen_df) > 0:
                print(f"  {gen}: {len(gen_perfect)}/{len(gen_df)} ({100*len(gen_perfect)/len(gen_df):.1f}%)")
        
        internal_switch = full_eval_df[
            (full_eval_df['Track1_accuracy'] < perfect_threshold) & 
            (full_eval_df['Track1_accuracy'] > 0.5)
        ]
        print(f"\nSamples with internal phase switches: {len(internal_switch)}")
        if len(internal_switch) > 0:
            print(internal_switch[['Sample', 'Contig', 'Track1_accuracy', 'Track2_accuracy']].head(20).to_string(index=False))

    print(f"\nPhase correction validation complete.")
    print(f"Total time: {time.time()-start:.1f}s")

#%%
if __name__ == '__main__':
    # =============================================================================
    # FINAL SUMMARY
    # =============================================================================
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"COMPLETE RUN FINISHED in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"Log saved to: {log_path}")
    print(f"{'='*60}")
# %%