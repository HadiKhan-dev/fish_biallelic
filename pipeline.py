#%%
# =============================================================================
# Module-level definitions (PICKLABLE by forkserver workers)
# =============================================================================
# Functions defined inside `if __name__ == '__main__':` are closures
# that cannot be pickled by multiprocessing.  Forkserver workers receive
# their initargs (including any callback functions) via pickle, so any
# function that needs to cross the worker boundary MUST live at module
# top level here.  Keep this section small -- imports here run in every
# forkserver worker at startup.

CHECKPOINT_DIR = ".pipeline_checkpoints"


def _load_contig_for_phase_correction(r_name):
    """
    Load tolerance_result and founder_block for one contig from
    checkpoint files on disk.

    Top-level (picklable) version used by forkserver workers in
    phase_correction.  Reads checkpoints DIRECTLY rather than
    going through main's `multi_contig_results` cache, because
    forkserver workers do not inherit main's process state.

    The stage->key mapping mirrors `_KEY_SOURCE` inside the
    `__main__` block:
        tolerance_result -> 10_viterbi_painting
        super_blocks_L4  -> 09_assembly_L4    (preferred)
        super_blocks_L3  -> 08_assembly_L3    (fallback)
    If a checkpoint file is missing the corresponding data key is
    simply omitted from the returned dict; the worker handles
    missing keys by falling back to an identity equivalence matrix.
    """
    import os
    import pickle as _pickle

    data = {}

    # tolerance_result lives in stage 10 (Viterbi painting)
    tol_path = os.path.join(CHECKPOINT_DIR, "10_viterbi_painting", f"{r_name}.pkl")
    if os.path.exists(tol_path):
        with open(tol_path, 'rb') as f:
            ckpt = _pickle.load(f)
        if 'tolerance_result' in ckpt:
            data['tolerance_result'] = ckpt['tolerance_result']
        del ckpt

    # founder_block lives in stage 9 (L4 assembly), falling back to
    # stage 8 (L3 assembly) when L4 is absent.  We take element [0]
    # because super_blocks is a list of BlockResults; in the
    # phase-correction flow there is exactly one block per contig
    # (the whole-chromosome merged block).
    for stage, list_key in [("09_assembly_L4", "super_blocks_L4"),
                             ("08_assembly_L3", "super_blocks_L3")]:
        if 'founder_block' in data:
            break
        path = os.path.join(CHECKPOINT_DIR, stage, f"{r_name}.pkl")
        if not os.path.exists(path):
            continue
        with open(path, 'rb') as f:
            ckpt = _pickle.load(f)
        if list_key in ckpt and ckpt[list_key]:
            data['founder_block'] = ckpt[list_key][0]
        del ckpt

    return data


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

    # =============================================================================
    # RUN-TIME TOGGLES
    # =============================================================================
    # Validation toggles -- each gates a group of ground-truth diagnostic cells
    # that are read-only (they do not affect any downstream stage) but take
    # non-trivial wall time when reloading checkpoints.  Set any flag to True
    # to skip its group; set to False to re-enable, e.g. when investigating
    # a regression in the corresponding upstream stage.
    #
    #   SKIP_VALIDATIONS_BLOCK_HAPS         -- 5 cells:
    #       * Block Haplotypes vs Ground Truth
    #       * Level 1 / 2 / 3 / 4 Super Blocks vs Ground Truth
    #     These all compare DISCOVERED haplotypes against the simulation's
    #     true founder haplotypes at increasingly aggregated granularities.
    #     Combined runtime: ~5-6 min when reloading checkpoints.
    #
    #   SKIP_VALIDATIONS_PAINTING           -- 1 cell:
    #       * Painted Samples Output vs Ground Truth (topology-based)
    #     Per-sample, per-contig assessment of the Stage 10 Viterbi painting
    #     before any phase correction.  Includes the disc->true founder
    #     relabelling bijection search; the slowest single validation.
    #
    #   SKIP_VALIDATIONS_PHASE_CORRECTION   -- 1 cell:
    #       * Phase Correction vs Ground Truth (allele-level)
    #     The final BEFORE/AFTER comparison run after Stage 12.  Reports
    #     Track1/Track2 accuracy by generation and the perfect-phasing rate.
    #
    # All three default to False (run all validations).
    SKIP_VALIDATIONS_BLOCK_HAPS = False
    SKIP_VALIDATIONS_PAINTING = True
    SKIP_VALIDATIONS_PHASE_CORRECTION = False


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
    import block_haplotypes  # Discrete coordinate descent w/ wildcard founder (drop-in for block_haplotypes)
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
    SIMULATION_SEED = 72

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
    vcf_path = "./fish_vcf_restriped/AsAc.AulStuGenome.biallelic.bcf.gz"

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
            
            t_chr = time.time()
            simd_block_results = block_haplotypes.generate_all_block_haplotypes(
                simd_genomic_data,
                uniqueness_threshold_percent=1.0,
                diff_threshold_percent=0.5,
                wrongness_threshold=1.0,
                num_processes=n_processes
            )
            disc_time = time.time() - t_chr
            
            valid_blocks = [b for b in simd_block_results if len(b.positions) > 0]
            simd_block_results = block_haplotypes.BlockResults(valid_blocks)
            
            multi_contig_results[r_name]['simd_block_results'] = simd_block_results
            save_contig(STAGE_3, r_name, {'simd_block_results': simd_block_results})
            
            hap_counts = [len(b.haplotypes) for b in valid_blocks]
            print(f"    {len(valid_blocks)} blocks, haps/block: "
                  f"min={min(hap_counts)}, max={max(hap_counts)}, mean={np.mean(hap_counts):.1f} "
                  f"[discovery: {disc_time:.1f}s]")

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
                            maxtasksperchild=WORKER_MAXTASKS,
                            refine_after_stitch=False  # STAGE_4 runs its own refinement pipeline; opt out
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
                            maxtasksperchild=WORKER_MAXTASKS,
                            refine_after_stitch=False  # STAGE_4 runs its own refinement pipeline; opt out
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
if __name__ == '__main__' and not SKIP_VALIDATIONS_BLOCK_HAPS:
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
if __name__ == '__main__' and not SKIP_VALIDATIONS_BLOCK_HAPS:
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
if __name__ == '__main__' and not SKIP_VALIDATIONS_BLOCK_HAPS:
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
if __name__ == '__main__' and not SKIP_VALIDATIONS_BLOCK_HAPS:
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
if __name__ == '__main__' and not SKIP_VALIDATIONS_BLOCK_HAPS:
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
                    switch_penalty_per_snp=1.0,
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
if __name__ == '__main__' and not SKIP_VALIDATIONS_PAINTING:
    # ==========================================================================
    # VALIDATE: Painted Samples Output Against Ground Truth (topology-based)
    # ==========================================================================
    # Per-sample, per-contig assessment of paint_samples (Viterbi painting,
    # Stage 10) output before any downstream correction is applied.
    #
    # The painting at this stage is an UNORDERED pair of founder IDs at each
    # position (no phase information; phase is introduced in Stage 12).  The
    # discovered founder ID space differs from the truth founder ID space by
    # a relabelling -- a bijection M : disc -> true that is constant across
    # the chromosome (i.e. disc-hap k is "the same biological haplotype" as
    # true-hap M[k]).  Two paintings are therefore equivalent up to:
    #   (1) unordered tuples per chunk: (a, b) ~ (b, a)
    #   (2) global founder bijection M: disc-tuple (a, b) ~ truth-tuple
    #       (M[a], M[b]) (sorted as unordered).
    #
    # We measure TOPOLOGY (the sequence of unordered tuples after applying M
    # and collapsing adjacent duplicates), not per-site allele accuracy.
    # An F1 with 107 raw chunks but topology [(2,3), (2,5), (0,5)] is doing
    # the right thing at the topology level even if it's heavily over-
    # segmented due to chimera noise within chunks.  Conversely, a painting
    # whose mapped topology contains a tuple absent from the truth has a
    # SPURIOUS segment regardless of how many bases it covers.
    #
    # Reports per (sample, contig):
    #   - raw chunk counts (painted, truth) and chunk-size distribution
    #   - mapped+collapsed topology lengths (painted, truth)
    #   - exact topology match (bool)
    #   - n spurious tuples (tuples in disc-mapped not in truth) as a SET
    #   - n missing tuples (tuples in truth not in disc-mapped) as a SET
    #   - extra transitions (topology length diff: signed)
    #
    # Writes a single CSV: paint_samples_topology_evaluation.csv, with one
    # row per (sample, contig).  No allele-level CSV is produced -- per-site
    # allele accuracy is reported by Stage 12 BEFORE/AFTER on the same
    # painting and would be redundant here.
    #
    # Validation is read-only -- no checkpointing.
    print(f"\n{'='*60}")
    print("Validating Painted Samples (paint_samples / Stage 10) Topology")
    print(f"{'='*60}")

    def _compute_disc_to_true_mapping(disc_dense_haps, true_dense_haps):
        """Greedy bijection: pair each discovered founder with its best-
        matching true founder, with each true founder used at most once.

        Returns M of shape (n_disc,) of dtype int32: M[d] = best true
        founder index for disc founder d, or -1 if no available match.

        Assumes both inputs have the same number of SNP sites (the second
        axis) and have entries in {0, 1, -1=missing}.  Agreement counts
        positions where neither side is missing AND they match.
        """
        n_disc = disc_dense_haps.shape[0]
        n_true = true_dense_haps.shape[0]
        # Pairwise agreement (fraction of sites matching where both are
        # non-missing; sites missing in either are excluded from numerator
        # and denominator).
        agreement = np.zeros((n_disc, n_true), dtype=np.float64)
        for d in range(n_disc):
            d_row = disc_dense_haps[d]
            for t in range(n_true):
                t_row = true_dense_haps[t]
                valid = (d_row != -1) & (t_row != -1)
                n_v = int(np.sum(valid))
                if n_v == 0:
                    agreement[d, t] = 0.0
                else:
                    agreement[d, t] = float(np.sum((d_row == t_row) & valid)) / n_v
        # Greedy bijection: pick max remaining, assign, mask row+col, repeat
        M = np.full(n_disc, -1, dtype=np.int32)
        assigned_disc = np.zeros(n_disc, dtype=bool)
        assigned_true = np.zeros(n_true, dtype=bool)
        for _ in range(min(n_disc, n_true)):
            masked = agreement.copy()
            masked[assigned_disc, :] = -np.inf
            masked[:, assigned_true] = -np.inf
            if not np.isfinite(masked).any():
                break
            idx = np.unravel_index(np.argmax(masked), masked.shape)
            d, t = int(idx[0]), int(idx[1])
            M[d] = t
            assigned_disc[d] = True
            assigned_true[t] = True
        return M

    def _topology_from_chunks(chunks, M=None):
        """Walk a painting's chunks and return the sequence of unordered
        founder-ID tuples in TRUTH SPACE, with adjacent duplicates collapsed.

        Each tuple is a canonical (min, max) pair so that (a, b) and (b, a)
        produce the same tuple object.  If M is provided, each hap id is
        translated through M; ids outside M's range or with M[id] == -1 are
        recorded as -1 (which collides only with itself: a (-1, -1) tuple
        marks a wholly-unmappable chunk).

        A chunk's mapped tuple is appended only when it differs from the
        previous one already on the list, so the returned list IS the
        topology (sequence of unique consecutive tuples).
        """
        topology = []
        if M is not None:
            M_len = len(M)
        for c in (chunks or []):
            h1, h2 = c.hap1, c.hap2
            if M is not None:
                h1 = int(M[h1]) if (0 <= h1 < M_len) else -1
                h2 = int(M[h2]) if (0 <= h2 < M_len) else -1
            # Canonicalize as ordered (min, max) -- this is the unordered
            # tuple as a hashable, comparable object.
            t = (h1, h2) if h1 <= h2 else (h2, h1)
            if not topology or topology[-1] != t:
                topology.append(t)
        return topology

    def validate_paint_samples_contig(r_name, painting, truth, M, sample_names):
        """Per-contig paint_samples topology assessment.

        Parameters
        ----------
        r_name : str
            Contig name.
        painting : BlockPainting
            Discovered paint_samples output (one SamplePainting per sample).
        truth : BlockPainting
            Ground-truth painting (already in truth founder ID space).
        M : np.ndarray of shape (n_disc,), dtype int32
            Disc-to-true bijection (see _compute_disc_to_true_mapping).
            Used to map discovered hap IDs into truth space before topology
            comparison.
        sample_names : list[str]

        Returns
        -------
        DataFrame with one row per sample, columns described in the cell
        docstring.
        """
        rows = []

        for i, name in enumerate(sample_names):
            painted_sample = painting[i]
            truth_sample = truth[i]

            painted_chunks = painted_sample.chunks if hasattr(painted_sample, 'chunks') else []
            truth_chunks = truth_sample.chunks if hasattr(truth_sample, 'chunks') else []
            n_painted = len(painted_chunks)
            n_truth = len(truth_chunks)

            # ---- Raw chunk-size distribution (no mapping/collapsing) ----
            if n_painted > 0:
                chunk_widths = np.array(
                    [c.end - c.start for c in painted_chunks], dtype=np.int64)
            else:
                chunk_widths = np.array([], dtype=np.int64)
            n_lt_10kb = int(np.sum(chunk_widths < 10_000))
            n_lt_100kb = int(np.sum(chunk_widths < 100_000))
            n_lt_1Mb = int(np.sum(chunk_widths < 1_000_000))
            median_width = int(np.median(chunk_widths)) if n_painted else 0
            min_width = int(chunk_widths.min()) if n_painted else 0
            max_width = int(chunk_widths.max()) if n_painted else 0
            total_painted_bp = int(chunk_widths.sum()) if n_painted else 0

            # ---- Topology in truth space (M-mapped, collapsed) ----
            # disc topology: chunks remapped to truth-space via M, then
            # adjacent duplicates collapsed.
            disc_topo = _topology_from_chunks(painted_chunks, M=M)
            # truth topology: chunks are already in truth space; collapse
            # any adjacent duplicates (truth_disc usually has none, but
            # defend against degenerate cases).
            truth_topo = _topology_from_chunks(truth_chunks, M=None)

            n_topo_disc = len(disc_topo)
            n_topo_truth = len(truth_topo)
            topology_exact_match = (disc_topo == truth_topo)

            # Set-level diffs: which TUPLES (regardless of position or
            # repetition count) are unique to one side?
            disc_set = set(disc_topo)
            truth_set = set(truth_topo)
            spurious_tuples = disc_set - truth_set
            missing_tuples = truth_set - disc_set
            n_spurious_tuples = len(spurious_tuples)
            n_missing_tuples = len(missing_tuples)

            # Sequence-level extra transitions (signed; positive = disc
            # over-segments at the topology level, negative = disc under-
            # segments).
            extra_transitions = n_topo_disc - n_topo_truth

            # Did the painting have any unmappable founder IDs?  These show
            # up as -1 in the mapped topology and indicate a discovered
            # founder that failed to find a matching truth founder under
            # the greedy bijection M.
            unmappable_in_topology = any(
                (a == -1 or b == -1) for a, b in disc_topo)

            rows.append({
                'Sample': name,
                'Contig': r_name,
                'N_chunks_painted': n_painted,
                'N_chunks_truth': n_truth,
                'N_topology_painted_mapped': n_topo_disc,
                'N_topology_truth': n_topo_truth,
                'Topology_exact_match': topology_exact_match,
                'N_spurious_tuples': n_spurious_tuples,
                'N_missing_tuples': n_missing_tuples,
                'Extra_transitions': extra_transitions,
                'Has_unmappable_founder': unmappable_in_topology,
                'Spurious_tuples': sorted(spurious_tuples),
                'Missing_tuples': sorted(missing_tuples),
                'Topology_painted_mapped': disc_topo,
                'Topology_truth': truth_topo,
                # Raw-chunk-size diagnostics (independent of topology):
                'N_chunks_lt_10kb': n_lt_10kb,
                'N_chunks_lt_100kb': n_lt_100kb,
                'N_chunks_lt_1Mb': n_lt_1Mb,
                'Median_chunk_width_bp': median_width,
                'Min_chunk_width_bp': min_width,
                'Max_chunk_width_bp': max_width,
                'Total_painted_bp': total_painted_bp,
            })

        return pd.DataFrame(rows)

    # ---- Loop over contigs, load checkpoint data lazily, evaluate ----
    print(f"\nPer-contig paint_samples topology:")
    all_dfs = []
    t_eval_start = time.time()
    for r_name in region_keys:
        # Load the inputs needed for this validation from their stage
        # checkpoints if not already in memory.  paint_samples' output is
        # registered as 'tolerance_result' in _KEY_SOURCE, so _ensure_key
        # will pull it from 10_viterbi_painting whether we just ran Stage 10
        # or are resuming from a completed checkpoint.
        try:
            _ensure_key(r_name, 'tolerance_result')
            _ensure_key(r_name, 'truth_painting')
            _ensure_key(r_name, 'naive_long_haps')
        except FileNotFoundError as e:
            print(f"  {r_name}: SKIP -- {e}")
            continue

        # Choose the discovered super-block used by paint_samples: Stage 10
        # used super_blocks_L4 (or L3 fallback) -- match its choice exactly
        # so positions/dense_haps align with what was painted.
        try:
            _ensure_key(r_name, 'super_blocks_L4')
            discovered_block = multi_contig_results[r_name]['super_blocks_L4'][0]
        except FileNotFoundError:
            try:
                _ensure_key(r_name, 'super_blocks_L3')
                discovered_block = multi_contig_results[r_name]['super_blocks_L3'][0]
            except FileNotFoundError as e:
                print(f"  {r_name}: SKIP -- no L4/L3 super_blocks ({e})")
                continue

        painting = multi_contig_results[r_name]['tolerance_result']
        truth = multi_contig_results[r_name]['truth_painting']
        positions = discovered_block.positions
        dense_haps, _ = phase_correction.founder_block_to_dense(discovered_block)

        orig_sites, orig_haps = multi_contig_results[r_name]['naive_long_haps']
        orig_haps_concrete = simulate_sequences.concretify_haps(orig_haps)
        site_indices = np.searchsorted(orig_sites, positions)
        site_indices = np.clip(site_indices, 0, len(orig_sites) - 1)
        true_dense_haps = np.array(
            [h[site_indices] for h in orig_haps_concrete], dtype=np.int8)

        # ---- Compute the disc-to-true founder bijection M for this contig ----
        # Per the project's assumption, this is a CLEAN BIJECTION when the
        # discovered haplotypes are well-recovered; if it isn't (e.g. some
        # disc founders are chimeric), per-sample diagnostics will flag
        # cases via Has_unmappable_founder or via spurious topology tuples.
        M = _compute_disc_to_true_mapping(dense_haps, true_dense_haps)

        contig_df = validate_paint_samples_contig(
            r_name, painting, truth, M, sample_names)

        # Per-contig one-line summary
        mean_n_painted = contig_df['N_chunks_painted'].mean()
        mean_n_truth = contig_df['N_chunks_truth'].mean()
        mean_n_topo_disc = contig_df['N_topology_painted_mapped'].mean()
        mean_n_topo_truth = contig_df['N_topology_truth'].mean()
        match_rate = 100.0 * contig_df['Topology_exact_match'].mean()
        mean_spurious = contig_df['N_spurious_tuples'].mean()
        mean_extra = contig_df['Extra_transitions'].mean()
        print(f"  {r_name}: raw chunks {mean_n_painted:5.1f} (truth {mean_n_truth:4.1f}), "
              f"topology {mean_n_topo_disc:4.1f} (truth {mean_n_topo_truth:4.1f}), "
              f"exact_match={match_rate:5.1f}%, "
              f"spurious={mean_spurious:.2f}, "
              f"extra_trans={mean_extra:+.2f}")

        all_dfs.append(contig_df)

        # Free this contig's tolerance_result and truth_painting after
        # evaluation to keep RAM in check; they'll be reloaded by later
        # stages (e.g. Stage 12) if needed.
        multi_contig_results[r_name].pop('tolerance_result', None)
        # NOTE: truth_painting is also used by Stage 12 evaluation; leave
        # it in place rather than re-loading (cheaper to keep).

    print(f"\nPaint_samples validation finished in "
          f"{time.time()-t_eval_start:.1f}s")

    # ---- Aggregate, summarize, save ----
    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)

        # Generation label from sample name prefix (F1_*/F2_*/F3_*).  Falls
        # back to 'F0' for any unexpected prefix so the column is never NaN.
        def _gen_of(sample_name):
            if sample_name.startswith('F1'):
                return 'F1'
            if sample_name.startswith('F2'):
                return 'F2'
            if sample_name.startswith('F3'):
                return 'F3'
            return 'F0'
        full_df['Generation'] = full_df['Sample'].apply(_gen_of)

        # ---- Per-generation summary ----
        # The headline diagnostic is the topology-match rate (what fraction
        # of (sample, contig) pairs have the discovered topology equal to
        # the truth topology, after M-mapping and collapsing) and the mean
        # extra-transitions count (how many extra topology segments does
        # the discovered painting introduce on average).
        print(f"\n{'-'*92}")
        print(f"Paint_samples topology summary by generation:")
        print(f"{'-'*92}")
        hdr = (f"  {'Gen':>4s}  {'N':>5s}  "
               f"{'raw_pt':>6s}  {'raw_th':>6s}  "
               f"{'topo_pt':>7s}  {'topo_th':>7s}  "
               f"{'match%':>7s}  {'spur':>5s}  "
               f"{'miss':>5s}  "
               f"{'xtra_trans':>10s}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for gen in sorted(full_df['Generation'].unique()):
            sub = full_df[full_df['Generation'] == gen]
            print(f"  {gen:>4s}  {len(sub):>5d}  "
                  f"{sub['N_chunks_painted'].mean():>6.1f}  "
                  f"{sub['N_chunks_truth'].mean():>6.1f}  "
                  f"{sub['N_topology_painted_mapped'].mean():>7.2f}  "
                  f"{sub['N_topology_truth'].mean():>7.2f}  "
                  f"{100.0*sub['Topology_exact_match'].mean():>6.2f}%  "
                  f"{sub['N_spurious_tuples'].mean():>5.2f}  "
                  f"{sub['N_missing_tuples'].mean():>5.2f}  "
                  f"{sub['Extra_transitions'].mean():>+10.2f}")

        # ---- Save CSV to output_dir ----
        # Note: Spurious_tuples / Missing_tuples / Topology_* columns are
        # stored as their Python repr (list of tuples) so the CSV preserves
        # full diagnostic detail at the cost of slightly awkward parsing.
        # Reread with: ast.literal_eval(row['Topology_painted_mapped']).
        topo_out = os.path.join(output_dir,
                                   "paint_samples_topology_evaluation.csv")
        try:
            full_df.to_csv(topo_out, index=False)
            print(f"\nTopology evaluation saved to: {topo_out}")
        except OSError:
            print("WARNING: Could not save paint_samples topology CSV "
                  "(disk full)")

        # ---- Top-N worst over-segmentation cases (diagnostic spotlight) ----
        # Sorted by extra_transitions (topology-level over-segmentation),
        # since absolute chunk count alone can be misleading after M-mapping
        # and collapsing (a 107-chunk painting may collapse to a clean
        # truth-matching topology if the chunks are M-equivalent).
        n_top = min(20, len(full_df))
        worst_topo = full_df.sort_values(
            'Extra_transitions', ascending=False).head(n_top)
        print(f"\nTop {n_top} cases by EXTRA TOPOLOGY TRANSITIONS "
              f"(topology over-segmentation vs truth):")
        print(worst_topo[['Sample', 'Contig', 'Generation',
                            'N_chunks_painted', 'N_chunks_truth',
                            'N_topology_painted_mapped',
                            'N_topology_truth',
                            'Topology_exact_match',
                            'N_spurious_tuples',
                            'Extra_transitions']].to_string(index=False))

        # Also show cases with spurious tuples (tuples present in discovered
        # that don't appear anywhere in truth's topology), since these are
        # the clearest signal of WRONG painting (vs simply over-segmented).
        has_spurious = full_df[full_df['N_spurious_tuples'] > 0]
        if len(has_spurious) > 0:
            n_top2 = min(20, len(has_spurious))
            spur = has_spurious.sort_values(
                'N_spurious_tuples', ascending=False).head(n_top2)
            print(f"\nTop {n_top2} cases with SPURIOUS TUPLES "
                  f"(wrong-founder painting not in truth):")
            print(spur[['Sample', 'Contig', 'Generation',
                         'N_chunks_painted', 'N_topology_painted_mapped',
                         'N_spurious_tuples',
                         'Spurious_tuples']].to_string(index=False))
        else:
            print(f"\nNo (sample, contig) cases have spurious tuples.")
    else:
        print("\nNo paint_samples evaluation data available -- "
              "tolerance_result missing for all contigs?")

    gc.collect()

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

        # Founder-block eviction (pairs with the SharedMemory hand-off inside
        # infer_pedigree_multi_contig_tolerance).  contig_inputs now holds the
        # only reference to each founder block, so drop the checkpoint-backed
        # super_blocks_L4 from multi_contig_results: pedigree inference copies
        # each block into SharedMemory and releases its own reference once
        # Phase 1 is dispatched, so the large founder set is freed for the
        # duration of Phase 2/3 + the consistency cutoff instead of sitting
        # idle in main.  super_blocks_L4 is re-loaded (via _ensure_key) further
        # below for F1 recoloring / propagation.
        for r_name in region_keys:
            multi_contig_results[r_name].pop('super_blocks_L4', None)

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

        # `_load_contig_for_phase_correction` is now defined at MODULE
        # top level (above the `if __name__ == '__main__':` block) so
        # the forkserver workers can pickle a reference to it.  The
        # previous closure here -- defined inside __main__ and using
        # main's `_ensure_key` + `multi_contig_results` -- would not
        # have survived pickling under the forkserver start method.
        # See the docstring of the top-level function for the stage
        # mapping it uses.

        # Ensure contig names exist in multi_contig_results (load_fn needs keys)
        for r_name in region_keys:
            multi_contig_results.setdefault(r_name, {})

        start = time.time()
        # Run phase correction — workers load their own data via load_fn
        # num_rounds=6 (was 3): under Jacobi iteration (introduced with
        # the May 2026 within-contig per-sample threading in
        # phase_correction.run_correction_round), the round count to
        # reach `corrections == 0` is typically 1-2 higher than under
        # Gauss-Seidel.  Setting num_rounds=3 left every contig
        # hitting the limit without full convergence, which caused a
        # measurable accuracy regression (perfect-phasing rate dropped
        # ~11pp).  Bumping the ceiling to 6 gives Jacobi room to
        # actually finish; the early-exit on `corrections == 0` means
        # genuinely-converged contigs incur no extra work.  Look for
        # "HIT MAX ROUNDS" in the worker output if even 6 is too low.
        multi_contig_results = phase_correction.correct_phase_all_contigs(
            multi_contig_results,
            pedigree_df,
            sample_names,
            num_rounds=6,
            verbose=True,
            max_workers=n_processes,
            load_fn=_load_contig_for_phase_correction
        )
        print(f"Phase correction time: {time.time()-start:.1f}s")

        # =============================================================================
        # GREEDY PHASE REFINEMENT POST-PROCESSING
        # =============================================================================
        print("\n" + "="*60)
        print("RUNNING: Greedy Phase Refinement (HOM→HET boundary flips)")
        print("="*60)

        # Same load_fn that phase correction used.  Greedy workers
        # now load founder_block themselves rather than expecting it
        # in main's multi_contig_results -- this is the
        # complement of the May 2026 IPC-cost fix in
        # phase_correction.py (`_process_contig_worker` no longer
        # returns founder_block).  Parallel disk I/O across 22
        # worker processes ~ same speed as the previous mechanism
        # (each worker was also loading founder_block) without the
        # ~40s pickle/pipe overhead of returning it.
        start_refine = time.time()
        multi_contig_results = phase_correction.post_process_phase_greedy_all_contigs(
            multi_contig_results,
            pedigree_df,
            sample_names,
            snps_per_bin=100,
            recomb_rate=5e-8,
            mismatch_cost=4.6,
            max_workers=n_processes,
            load_fn=_load_contig_for_phase_correction,
            verbose=True
        )
        print(f"Greedy refinement time: {time.time()-start_refine:.1f}s")

        # =============================================================================
        # PARSIMONIOUS F1 RECOLORING
        # =============================================================================
        # F1 recoloring and the propagation step that follows both run
        # in the MAIN process (sequentially across contigs).  They
        # access founder_block from main's multi_contig_results, which
        # no longer has it after the IPC-cost fix removed founder_block
        # from the phase correction worker return tuple.  Pre-load it
        # here in parallel via threads (parallel disk I/O across 22
        # contigs, typical ~5-15s) so the F1 + propagation loops see
        # the data they expect.  RAM cost is ~30-40 GB for the full
        # set of founder_blocks; HPC nodes typically have plenty.
        print("\n" + "="*60)
        print("LOADING founder_blocks for F1 recoloring + propagation")
        print("="*60)
        from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
        _t0 = time.time()
        def _load_founder_block_for_main(r_name):
            _ensure_key(r_name, 'super_blocks_L4')
            if 'super_blocks_L4' in multi_contig_results[r_name]:
                multi_contig_results[r_name]['founder_block'] = \
                    multi_contig_results[r_name]['super_blocks_L4'][0]
            elif 'super_blocks_L3' in multi_contig_results[r_name]:
                multi_contig_results[r_name]['founder_block'] = \
                    multi_contig_results[r_name]['super_blocks_L3'][0]
        with _ThreadPoolExecutor(max_workers=min(n_processes, len(region_keys))) as _ex:
            list(_ex.map(_load_founder_block_for_main, region_keys))
        print(f"Founder block parallel load: {time.time()-_t0:.1f}s")

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
                max_workers=n_processes,
                max_mismatch_rate=0.02,
                verbose=False
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
                max_workers=n_processes,
                max_mismatch_rate=0.02,
                verbose=False
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
if __name__ == '__main__' and not SKIP_VALIDATIONS_PHASE_CORRECTION:
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
        """
        Per-contig evaluator -- kept as a thin wrapper around the
        new per-sample evaluator `_evaluate_one_sample` so that any
        external code paths still see this name.  The actual BEFORE /
        AFTER evaluation loops below bypass this wrapper and submit
        per-sample tasks directly to a ThreadPoolExecutor with
        max_workers=n_processes, which gives ~5x more parallelism
        on a 22-contig / 320-sample workload (22 contigs only used
        22 cores; per-sample dispatch uses all 112).
        """
        r_name, painting, truth, positions, disc_dense_haps, true_dense_haps, sample_names_local = args
        results = []
        for i, name in enumerate(sample_names_local):
            _r, _name, sample_result = _evaluate_one_sample(
                (r_name, i, name, painting[i], truth[i],
                 positions, disc_dense_haps, true_dense_haps)
            )
            results.append(sample_result)
        contig_eval = pd.DataFrame(results)
        contig_eval['Contig'] = r_name
        return r_name, contig_eval


    def _evaluate_one_sample(args):
        """
        Per-sample dual-founders evaluator.  Safe to call concurrently
        across threads -- it only reads from its arg tuple and the
        shared `disc_dense_haps` / `true_dense_haps` arrays (which are
        read-only after construction in the calling site).  Returns
        (r_name, sample_name, result_dict) so the caller can group
        results by contig for the per-contig summary print.

        The numerical body is byte-identical to the original inner
        loop of evaluate_contig_dual_founders -- only the wrapping
        changed (single sample per call instead of looping over all
        samples in a contig).
        """
        (r_name, sample_idx, sample_name, corrected_sample, truth_sample,
         positions, disc_dense_haps, true_dense_haps) = args
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
        return (r_name, sample_name, {
            'Sample': sample_name, 'Total_sites': n_pos, 'Valid_sites': int(n_valid),
            'Correct_sites': int(np.sum(correct_either & valid_mask)),
            'Accuracy': accuracy, 'Track1_accuracy': track1_acc,
            'Track2_accuracy': track2_acc, 'Direct_matches': int(n_direct),
            'Flipped_matches': int(n_flipped), 'Dominant_phase': dominant_phase
        })


    def _evaluate_paintings_per_sample(painting_by_contig, contig_shared_local,
                                       sample_names_local, region_keys_local,
                                       max_workers):
        """
        Driver: flatten all (contig, sample) pairs into a single task
        list, dispatch to a ThreadPoolExecutor with `max_workers`
        workers, and group the per-sample results back by contig.

        Per-contig summary lines are printed in region_keys order
        after ALL samples complete (the previous per-contig dispatch
        printed each contig's summary as the contig finished; the new
        scheme has to wait for all samples to finish first because
        the workload is interleaved across contigs).  This is a minor
        UX trade-off; the per-sample work is so fast that the print
        block lands within a fraction of a second of the last sample.

        Returns a list of per-contig DataFrames in region_keys order.
        """
        # Build flat per-sample arg list.  All large arrays
        # (disc_dense_haps, true_dense_haps, positions) are shared
        # by reference across all sample tasks within a contig --
        # ThreadPoolExecutor preserves shared memory semantics so
        # there's no pickling cost.
        all_args = []
        for r_name in region_keys_local:
            painting = painting_by_contig.get(r_name)
            if painting is None:
                continue
            if r_name not in contig_shared_local:
                continue
            truth, positions, disc_dense_haps, true_dense_haps = contig_shared_local[r_name]
            for i, name in enumerate(sample_names_local):
                all_args.append((r_name, i, name, painting[i], truth[i],
                                  positions, disc_dense_haps, true_dense_haps))

        if not all_args:
            return []

        # Group results by contig as they come back.  Using a dict
        # keyed by r_name so we can iterate region_keys at the end
        # for ordered printing.
        results_by_contig = {r: [] for r in region_keys_local}
        effective_workers = max(1, min(len(all_args), max_workers))

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            for r_name, sample_name, sample_result in executor.map(_evaluate_one_sample, all_args):
                results_by_contig[r_name].append(sample_result)

        # Build per-contig DataFrames + print summary in region_keys order
        contig_dfs = []
        for r_name in region_keys_local:
            sample_results = results_by_contig.get(r_name, [])
            if not sample_results:
                continue
            contig_eval = pd.DataFrame(sample_results)
            contig_eval['Contig'] = r_name
            mean_acc = contig_eval['Accuracy'].mean() * 100
            mean_t1 = contig_eval['Track1_accuracy'].mean() * 100
            mean_t2 = contig_eval['Track2_accuracy'].mean() * 100
            print(f"  {r_name}: Allele={mean_acc:.2f}%, Track1={mean_t1:.2f}%, Track2={mean_t2:.2f}%")
            contig_dfs.append(contig_eval)

        return contig_dfs

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
    # Per-sample parallelism: flatten the 22-contig x 320-sample
    # workload into 7040 independent sample tasks and dispatch via a
    # single ThreadPoolExecutor with max_workers=n_processes.  See
    # the docstring of `_evaluate_paintings_per_sample` for details.
    print("\n" + "-"*60)
    print("BEFORE phase correction (raw Viterbi painting):")
    print("-"*60)

    before_painting_by_contig = {}
    for r_name in region_keys:
        if r_name not in contig_shared:
            continue
        raw_painting = multi_contig_results[r_name].get('tolerance_result')
        if raw_painting is None:
            continue
        before_painting_by_contig[r_name] = raw_painting

    before_contig_results = _evaluate_paintings_per_sample(
        before_painting_by_contig, contig_shared, sample_names,
        region_keys, max_workers=n_processes
    )

    if before_contig_results:
        before_df = pd.concat(before_contig_results, ignore_index=True)
        before_df['Generation'] = before_df['Sample'].apply(
            lambda x: 'F1' if x.startswith('F1') else ('F2' if x.startswith('F2') else 'F3'))
    else:
        before_df = pd.DataFrame()

    # --- AFTER: evaluate corrected painting ---
    # Per-sample parallelism: same flatten-and-dispatch pattern as
    # the BEFORE block above.  See `_evaluate_paintings_per_sample`
    # for the rationale.
    print("\n" + "-"*60)
    print("AFTER phase correction (corrected + greedy + F1 recoloring):")
    print("-"*60)

    after_painting_by_contig = {}
    for r_name in region_keys:
        if r_name not in contig_shared:
            continue
        if 'final_painting' in multi_contig_results[r_name]:
            painting = multi_contig_results[r_name]['final_painting']
        elif 'refined_painting' in multi_contig_results[r_name]:
            painting = multi_contig_results[r_name]['refined_painting']
        elif 'corrected_painting' in multi_contig_results[r_name]:
            painting = multi_contig_results[r_name]['corrected_painting']
        else:
            continue
        after_painting_by_contig[r_name] = painting

    all_contig_results = _evaluate_paintings_per_sample(
        after_painting_by_contig, contig_shared, sample_names,
        region_keys, max_workers=n_processes
    )

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
    print(f"Total time: {time.time()-total_start:.1f}s")

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