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

import os
import hashlib

CHECKPOINT_DIR = os.environ.get(
    "BHD_SIM_CHECKPOINT_DIR", ".pipeline_checkpoints"
)
SIMULATION_OUTPUT_DIR = os.environ.get(
    "BHD_SIM_OUTPUT_DIR", "results_simulation"
)
_SIMULATION_SEED_TEXT = os.environ.get(
    "BHD_SIMULATION_SEED", "72"
).strip()
SIMULATION_SEED = (
    None
    if _SIMULATION_SEED_TEXT.lower() in {"none", "random"}
    else int(_SIMULATION_SEED_TEXT)
)
_SIMULATION_CONTIGS_TEXT = os.environ.get("BHD_SIM_CONTIGS")
_PER_CONTIG_STAGE_NAMES = (
    "03_block_haplotypes",
    "04_refinement",
    "05_residual_discovery",
    "06_assembly_L1",
    "07_assembly_L2",
    "08_assembly_L3",
    "09_assembly_L4",
    "10_terminal_cavity",
    "11_viterbi_painting",
)
SIMULATION_STOP_AFTER_STAGE = os.environ.get("BHD_SIM_STOP_AFTER_STAGE")
if (SIMULATION_STOP_AFTER_STAGE is not None
        and SIMULATION_STOP_AFTER_STAGE not in _PER_CONTIG_STAGE_NAMES):
    raise ValueError(
        "BHD_SIM_STOP_AFTER_STAGE must be one of: "
        + ", ".join(_PER_CONTIG_STAGE_NAMES)
    )


def _parse_simulation_contig_shard(raw_value):
    """Parse an explicit comma-separated shard without assigning its order."""
    if raw_value is None:
        return None
    requested = raw_value.split(",")
    if not requested or any(not name or name != name.strip()
                            for name in requested):
        raise ValueError(
            "BHD_SIM_CONTIGS must be a comma-separated list of exact, "
            "non-empty contig names without surrounding whitespace"
        )
    duplicates = []
    seen = set()
    for name in requested:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise ValueError(
            f"BHD_SIM_CONTIGS contains duplicate contigs: {duplicates}"
        )
    return tuple(requested)


def _select_simulation_contigs(all_contigs, requested_contigs):
    """Validate requested names and return them in the Stage-2 manifest order."""
    if requested_contigs is None:
        return list(all_contigs)
    known = set(all_contigs)
    unknown = [name for name in requested_contigs if name not in known]
    if unknown:
        raise ValueError(
            f"BHD_SIM_CONTIGS contains unknown contigs: {unknown}; "
            f"available contigs: {list(all_contigs)}"
        )
    requested = set(requested_contigs)
    return [name for name in all_contigs if name in requested]


SIMULATION_CONTIG_SHARD = _parse_simulation_contig_shard(
    _SIMULATION_CONTIGS_TEXT
)
SIMULATION_SHARD_MODE = SIMULATION_CONTIG_SHARD is not None
SIMULATION_SHARD_LOG_ID = (
    hashlib.sha256(",".join(SIMULATION_CONTIG_SHARD).encode()).hexdigest()[:10]
    if SIMULATION_SHARD_MODE else None
)

# Run identity overrides: BHD_SIMULATION_SEED, BHD_SIM_CHECKPOINT_DIR,
# BHD_SIM_OUTPUT_DIR, BHD_NUM_PROCESSES, BHD_SIM_CONTIGS, and
# BHD_SIM_STOP_AFTER_STAGE. Defaults preserve interactive use.

import checkpoint_io
from thread_env import force_single_threaded_numeric_libraries
import pipeline_runtime
def _finish_simulation_stage_checkpoints(
        checkpoint_store, stage, contigs, shard_mode):
    """Require every contig output and publish a marker only for full runs."""
    pipeline_runtime.require_contig_checkpoints(
        checkpoint_store, stage, contigs
    )
    if not shard_mode and not checkpoint_store.stage_complete(stage):
        checkpoint_store.mark_stage_complete(stage)




def _load_contig_for_phase_correction(r_name):
    """Load the atomic final-panel painting bundle for phase correction."""
    return pipeline_runtime.load_phase_correction_inputs(
        CHECKPOINT_DIR,
        r_name,
        tolerance_stage="11_viterbi_painting",
    )


#%%
if __name__ == '__main__':
    import os
    import sys
    from datetime import datetime

    # FORCE NUMPY/BLAS TO USE 1 THREAD PER PROCESS
    # (Numba threading is now managed by thread_config.py — do NOT set
    #  NUMBA_NUM_THREADS or NUMBA_THREADING_LAYER here)
    force_single_threaded_numeric_libraries()

    # =============================================================================
    # DUAL LOGGING: Console + File
    # =============================================================================
    # All print() output goes to both the terminal and a timestamped log file.
    # tqdm progress bars still display on the terminal only (they use stderr).
    # If the SSH connection drops, the log file preserves all output.


    os.makedirs("logs", exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    seed_label = "random" if SIMULATION_SEED is None else str(SIMULATION_SEED)
    shard_log_suffix = (
        f"_shard{SIMULATION_SHARD_LOG_ID}" if SIMULATION_SHARD_MODE else ""
    )
    log_path = os.path.join(
        "logs", f"run_sim_seed{seed_label}{shard_log_suffix}_{run_timestamp}.log"
    )
    sys.stdout = pipeline_runtime.TeeOutput(log_path, sys.stdout)
    print(f"Logging to: {log_path}")
    print(f"Run started: {run_timestamp}")
    print(
        f"Simulation run: seed={SIMULATION_SEED}, checkpoints={CHECKPOINT_DIR}, "
        f"output={SIMULATION_OUTPUT_DIR}"
    )
    if SIMULATION_SHARD_MODE:
        print(
            f"Simulation chromosome shard {SIMULATION_SHARD_LOG_ID}: "
            + ", ".join(SIMULATION_CONTIG_SHARD)
        )
    if SIMULATION_STOP_AFTER_STAGE is not None:
        print(
            "Simulation stop point: "
            f"after {SIMULATION_STOP_AFTER_STAGE}"
        )

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
    #     Per-sample, per-contig assessment of the Stage 11 Viterbi painting
    #     before any phase correction.  Includes the disc->true founder
    #     relabelling bijection search; the slowest single validation.
    #
    #   SKIP_VALIDATIONS_PHASE_CORRECTION   -- 1 cell:
    #       * Phase Correction vs Ground Truth (allele-level)
    #     The final BEFORE/AFTER comparison run after Stage 13.  Reports
    #     Track1/Track2 accuracy by generation and the perfect-phasing rate.
    #
    # All three default to False (run all validations).
    SKIP_VALIDATIONS_BLOCK_HAPS = SIMULATION_SHARD_MODE
    SKIP_VALIDATIONS_PAINTING = SIMULATION_SHARD_MODE
    SKIP_VALIDATIONS_PHASE_CORRECTION = SIMULATION_SHARD_MODE


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
    import beam_search_core
    import chimera_resolution
    import hierarchical_assembly
    import small_block_refine
    import paint_samples
    import pedigree_inference
    import pedigree_pipeline
    from pedigree_evaluation import compare_relationships_to_truth
    import phase_correction
    import residual_discovery
    import terminal_cavity_refinement


    warnings.filterwarnings("ignore")
    np.seterr(divide='ignore', invalid="ignore")

    if platform.system() != "Windows":
        #os.nice(15)
        print(f"Main process ({os.getpid()}) niceness set to: {os.nice(0)}")


    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    n_processes = int(os.environ.get("BHD_NUM_PROCESSES", "112"))
    available_cpus = pipeline_runtime.available_cpu_count()
    if not 1 <= n_processes <= available_cpus:
        raise ValueError(
            f"BHD_NUM_PROCESSES must lie in [1, {available_cpus}]; "
            f"received {n_processes}"
        )
    print(f"CPU budget: {n_processes} of {available_cpus} available CPUs")
    # Recycle workers after each batch to prevent memory accumulation
    # from glibc malloc fragmentation (Python doesn't return freed pages to OS).
    WORKER_MAXTASKS = 1

    # -------------------------------------------------------------------------
    # REPRODUCIBILITY: Set BHD_SIMULATION_SEED for the simulation.
    # All random processes (pedigree structure, meiosis, read sampling) derive
    # deterministic sub-seeds from this value. Use "none" or "random" for
    # non-reproducible runs using system entropy.
    # -------------------------------------------------------------------------
    # SIMULATION_SEED is parsed at module import so forkserver workers and the
    # main process share one exact run identity.

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
    # Each stage gets a subdirectory.  Each contig gets its own checkpoint
    # file (a protocol-5/Blosc frame, suffix ".p5.b2"; see checkpoint_io).
    # The format-qualified done marker means all contigs completed.
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
    # To force a full re-run, remove the exact configured checkpoint directory
    # after verifying its resolved path. To resume, keep completed stage files.

    checkpoint_store = pipeline_runtime.CheckpointStore(
        CHECKPOINT_DIR, nthreads=n_processes
    )
    stage_complete = checkpoint_store.stage_complete
    mark_stage_complete = checkpoint_store.mark_stage_complete
    contig_done = checkpoint_store.contig_done
    save_contig = checkpoint_store.save_contig
    load_contig = checkpoint_store.load_contig
    save_global = checkpoint_store.save_global
    load_global = checkpoint_store.load_global
    if SIMULATION_SHARD_MODE:
        missing_shared_stages = [
            stage for stage in ("01_vcf_discovery", "02_simulation")
            if not stage_complete(stage)
        ]
        if missing_shared_stages:
            raise RuntimeError(
                "BHD_SIM_CONTIGS requires globally completed shared "
                "Stages 1-2; missing format-qualified completion markers for: "
                f"{missing_shared_stages}"
            )
        print(
            "[SHARD] Verified globally completed shared Stages 1-2; "
            "diagnostic validations and per-contig plots are disabled"
        )

    def _finish_per_contig_stage(stage):
        """Verify a shard/stop boundary and publish only normal-run markers."""
        _finish_simulation_stage_checkpoints(
            checkpoint_store, stage, region_keys,
            shard_mode=SIMULATION_SHARD_MODE,
        )
        if SIMULATION_SHARD_MODE:
            print(
                f"[SHARD] Verified {stage} outputs for "
                f"{', '.join(region_keys)}; global completion marker not written"
            )
        if SIMULATION_STOP_AFTER_STAGE == stage:
            print(f"[STOP] Completed and verified {stage}; exiting cleanly")
            raise SystemExit(0)


    # Which stage checkpoint holds each per-contig key.
    # Values can be a single stage string or a list (tried in order, first hit wins).
    # simd_block_results lives in 03 before refinement, 04 after refinement,
    # and 05 after residual discovery.
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
        'super_blocks_L4':    '10_terminal_cavity',
        'tolerance_result':   '11_viterbi_painting',
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
        pipeline_runtime.require_contig_checkpoints(
            checkpoint_store,
            STAGE_1,
            [region['contig'] for region in regions_config],
        )
        mark_stage_complete(STAGE_1)

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE 2: Simulation + Post-processing
    # =========================================================================
    STAGE_2 = "02_simulation"
    generation_sizes = (20, 100, 200)
    STRESS_TEST_MUTATIONS = False
    mutate_rate = 1e-5 if STRESS_TEST_MUTATIONS else 1e-10
    current_stage2_region_keys = [
        region['contig'] for region in regions_config
    ]
    stage2_run_spec = {
        'ordered_regions': tuple(
            (region['contig'], int(region['start']), int(region['end']))
            for region in regions_config
        ),
        'generation_sizes': generation_sizes,
        'recombination_rate_per_bp': 5e-8,
        'mutation_rate_per_bp': mutate_rate,
        'read_depth': 5.0,
        'read_error_rate': 0.02,
        'snps_per_block': 200,
        'snp_shift': 200,
    }
    STAGE_2_REQUIRED_KEYS = frozenset({
        'truth_pedigree',
        'sample_names',
        'region_keys',
        'simulation_seed',
        'requested_simulation_seed',
        'run_spec',
        'simulation_state',
    })

    def require_completed_stage2_payload(payload, context):
        missing_keys = STAGE_2_REQUIRED_KEYS.difference(payload)
        if missing_keys:
            raise RuntimeError(
                f"{context} lacks required keys: {sorted(missing_keys)!r}"
            )
        if payload['simulation_state'] != 'complete':
            raise RuntimeError(
                f"{context} has simulation_state "
                f"{payload['simulation_state']!r}, not 'complete'"
            )

    def require_current_stage2_identity(payload, context):
        if payload['requested_simulation_seed'] != SIMULATION_SEED:
            raise RuntimeError(
                f"{context} was generated for requested seed "
                f"{payload['requested_simulation_seed']!r}, not "
                f"{SIMULATION_SEED!r}"
            )
        if payload['simulation_seed'] is None:
            raise RuntimeError(f"{context} has no realized simulation seed")
        if (SIMULATION_SEED is not None
                and payload['simulation_seed'] != SIMULATION_SEED):
            raise RuntimeError(
                f"{context} has realized seed "
                f"{payload['simulation_seed']!r}, not requested fixed seed "
                f"{SIMULATION_SEED!r}"
            )
        if payload['run_spec'] != stage2_run_spec:
            raise RuntimeError(
                f"{context} run specification does not match this run"
            )
        if list(payload['region_keys']) != current_stage2_region_keys:
            raise RuntimeError(
                f"{context} contig order does not match this run"
            )

    # An allocation can end after the complete global payload is durable but
    # before its tiny marker is published. Validate it, then finish atomically.
    if (not stage_complete(STAGE_2)
            and checkpoint_store.global_done(STAGE_2)):
        durable_stage2_payload = load_global(STAGE_2)
        if 'simulation_state' not in durable_stage2_payload:
            raise RuntimeError(
                f"{STAGE_2} global payload lacks simulation_state"
            )
        durable_state = durable_stage2_payload['simulation_state']
        if durable_state == 'complete':
            require_completed_stage2_payload(
                durable_stage2_payload, f"Durable {STAGE_2}"
            )
            require_current_stage2_identity(
                durable_stage2_payload, f"Durable {STAGE_2}"
            )
            pipeline_runtime.require_contig_checkpoints(
                checkpoint_store, STAGE_2,
                durable_stage2_payload['region_keys'],
            )
            mark_stage_complete(STAGE_2)
            print(f"  [RECOVER] Published completion marker for {STAGE_2}")
        elif durable_state != 'in_progress':
            raise RuntimeError(
                f"{STAGE_2} global payload has invalid simulation_state "
                f"{durable_state!r}"
            )
        del durable_stage2_payload

    if stage_complete(STAGE_2):
        print(f"\n[RESUME] Skipping simulation (checkpoint found)")
        g = load_global(STAGE_2)
        require_completed_stage2_payload(g, STAGE_2)
        require_current_stage2_identity(g, STAGE_2)
        pipeline_runtime.require_contig_checkpoints(
            checkpoint_store, STAGE_2, g['region_keys']
        )
        realized_simulation_seed = g["simulation_seed"]
        requested_simulation_seed = g["requested_simulation_seed"]
        truth_pedigree = g['truth_pedigree']
        sample_names = g['sample_names']
        region_keys = g['region_keys']
        del g
        # Per-contig data loaded on-demand via _ensure_key
    else:
        start = time.time()
        output_dir = SIMULATION_OUTPUT_DIR
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

        # Bind the checkpoint root to one realized seed before simulation or
        # any per-contig Stage-2 writes. This makes an entropy-seeded run
        # reproducible on restart and prevents old partial contigs from being
        # silently mixed with a newly generated pedigree.
        if checkpoint_store.global_done(STAGE_2):
            stage2_provenance = load_global(STAGE_2)
            partial_required_keys = {
                'simulation_seed', 'requested_simulation_seed',
                'region_keys', 'run_spec', 'simulation_state',
            }
            missing_keys = partial_required_keys.difference(stage2_provenance)
            if missing_keys:
                raise RuntimeError(
                    f"Partial {STAGE_2} checkpoint lacks required keys: "
                    f"{sorted(missing_keys)!r}"
                )
            if stage2_provenance['simulation_state'] != 'in_progress':
                raise RuntimeError(
                    f"Partial {STAGE_2} checkpoint has simulation_state "
                    f"{stage2_provenance['simulation_state']!r}, not "
                    "'in_progress'"
                )
            require_current_stage2_identity(
                stage2_provenance, f"Partial {STAGE_2} checkpoint"
            )
            realized_simulation_seed = stage2_provenance["simulation_seed"]
            requested_simulation_seed = (
                stage2_provenance["requested_simulation_seed"]
            )
            del stage2_provenance
            print(
                f"  [RESUME] {STAGE_2} realized seed "
                f"{realized_simulation_seed}"
            )
        else:
            partial_contigs = [
                r_name for r_name in region_keys
                if contig_done(STAGE_2, r_name)
            ]
            if partial_contigs:
                raise RuntimeError(
                    f"Partial {STAGE_2} contigs lack run provenance "
                    f"({partial_contigs}); use a fresh checkpoint directory"
                )
            realized_simulation_seed = (
                SIMULATION_SEED
                if SIMULATION_SEED is not None
                else int.from_bytes(os.urandom(8), "little")
            )
            save_global(STAGE_2, {
                'simulation_seed': realized_simulation_seed,
                'requested_simulation_seed': SIMULATION_SEED,
                'region_keys': list(region_keys),
                'run_spec': stage2_run_spec,
                'simulation_state': 'in_progress',
            })
            if not checkpoint_store.global_done(STAGE_2):
                raise OSError(
                    f"Failed to checkpoint early {STAGE_2} provenance"
                )
            print(
                f"  {STAGE_2} realized seed: {realized_simulation_seed}"
            )

        # 2. Run Multi-Contig Simulation
        print(f"Running Multi-Contig Simulation for {len(region_keys)} regions...")

        if STRESS_TEST_MUTATIONS:
            print(f"STRESS TEST MODE: Using mutation rate {mutate_rate} (~1% per generation)")
        else:
            print(f"Normal mode: Using mutation rate {mutate_rate} (minimal mutations)")

        t0 = time.time()
        all_offspring_lists, truth_pedigree, truth_paintings_lists = simulate_sequences.simulate_pedigree(
            founders_list, 
            sites_list, 
            generation_sizes, 
            recomb_rate=stage2_run_spec['recombination_rate_per_bp'],
            mutate_rate=stage2_run_spec['mutation_rate_per_bp'],
            output_plot=None,
            parallel=True,
            num_processes=n_processes,
            seed=realized_simulation_seed
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

        # 4. Process and checkpoint one contig at a time. Holding all 22
        # processed payloads while compression workers make pickle copies can
        # exceed the intended memory budget. These per-contig seeds exactly
        # match process_all_contigs_parallel for a fixed master seed.
        t0 = time.time()
        read_seed = realized_simulation_seed + 1_000_000
        read_seed_rng = np.random.default_rng(read_seed)
        contig_read_seeds = [
            int(read_seed_rng.integers(0, 2**63))
            for _ in region_keys
        ]
        print(
            "Post-processing one contig at a time for bounded peak memory"
            + (f" (seed={read_seed})" if read_seed is not None else "")
        )

        stage2_payload_keys = (
            'simulated_reads', 'simd_genomic_data', 'simd_probs',
            'simd_priors', 'truth_painting',
        )
        for contig_index, r_name in enumerate(region_keys):
            if contig_done(STAGE_2, r_name):
                print(f"  [RESUME] {r_name} post-processing already done")
            else:
                result = (
                    simulate_sequences._process_single_contig_postprocessing((
                        r_name,
                        all_offspring_lists[contig_index],
                        truth_paintings_lists[contig_index],
                        sites_list[contig_index],
                        stage2_run_spec['read_depth'],
                        stage2_run_spec['read_error_rate'],
                        stage2_run_spec['snps_per_block'],
                        stage2_run_spec['snp_shift'],
                        contig_read_seeds[contig_index],
                    ))
                )
                payload = {
                    key: result[key] for key in stage2_payload_keys
                }
                save_contig(STAGE_2, r_name, payload)
                if not contig_done(STAGE_2, r_name):
                    raise OSError(
                        f"Failed to checkpoint {STAGE_2}/{r_name}"
                    )
                del payload, result

            # Release each contig as soon as its checkpoint is durable.
            all_offspring_lists[contig_index] = None
            truth_paintings_lists[contig_index] = None
            founders_list[contig_index] = None
            sites_list[contig_index] = None
            gc.collect()

        pipeline_runtime.require_contig_checkpoints(
            checkpoint_store, STAGE_2, region_keys
        )
        print(
            f"Post-processing ({len(region_keys)} contigs, bounded): "
            f"{time.time()-t0:.1f}s"
        )

        print("\nSimulation, Sequencing, and Chunking complete for all regions.")
        print(f"Total time: {time.time()-start:.1f}s")

        completed_stage2_payload = {
            'truth_pedigree': truth_pedigree,
            'sample_names': sample_names,
            'region_keys': region_keys,
            'simulation_seed': realized_simulation_seed,
            'requested_simulation_seed': SIMULATION_SEED,
            'run_spec': stage2_run_spec,
            'simulation_state': 'complete',
        }
        save_global(STAGE_2, completed_stage2_payload)
        if not checkpoint_store.global_done(STAGE_2):
            raise OSError(f"Failed to checkpoint {STAGE_2}/_global")
        persisted_stage2_payload = load_global(STAGE_2)
        require_completed_stage2_payload(
            persisted_stage2_payload, f"Persisted {STAGE_2}"
        )
        for key in (
            'sample_names', 'region_keys', 'simulation_seed',
            'requested_simulation_seed', 'run_spec',
        ):
            if persisted_stage2_payload[key] != completed_stage2_payload[key]:
                raise RuntimeError(
                    f"Persisted {STAGE_2} changed {key!r} during checkpointing"
                )
        del persisted_stage2_payload, completed_stage2_payload
        mark_stage_complete(STAGE_2)
        # Free heavy simulation data — all checkpointed, will reload on demand
        for r_name in region_keys:
            for _k in ('simulated_reads', 'simd_genomic_data', 'simd_probs', 'simd_priors', 'truth_painting'):
                multi_contig_results[r_name].pop(_k, None)
        del all_offspring_lists, truth_paintings_lists
        del founders_list, sites_list
        gc.collect()
    
#%%
if __name__ == '__main__':
    # Ensure output_dir and globals exist for all subsequent stages
    output_dir = SIMULATION_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    if 'region_keys' not in dir() or region_keys is None:
        g = load_global('02_simulation')
        region_keys = g['region_keys']
        sample_names = g['sample_names']
        truth_pedigree = g['truth_pedigree']
        del g
    if SIMULATION_SHARD_MODE:
        all_region_keys = list(region_keys)
        if not checkpoint_store.global_done("02_simulation"):
            raise RuntimeError(
                "BHD_SIM_CONTIGS requires the global Stage-2 manifest"
            )
        pipeline_runtime.require_contig_checkpoints(
            checkpoint_store, "01_vcf_discovery", all_region_keys
        )
        pipeline_runtime.require_contig_checkpoints(
            checkpoint_store, "02_simulation", all_region_keys
        )
        region_keys = _select_simulation_contigs(
            all_region_keys, SIMULATION_CONTIG_SHARD
        )
        print(
            f"[SHARD] Processing {len(region_keys)} of "
            f"{len(all_region_keys)} contigs in Stage-2 manifest order: "
            f"{', '.join(region_keys)}"
        )


    # =========================================================================
    # STAGE 3: Discover Block Haplotypes from Simulated Reads
    # =========================================================================
    STAGE_3 = "03_block_haplotypes"

    if stage_complete(STAGE_3) and not SIMULATION_SHARD_MODE:
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
        pipeline_runtime.require_contig_checkpoints(
            checkpoint_store, STAGE_3, region_keys
        )
    _finish_per_contig_stage(STAGE_3)

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE 4: Conditional Refinement (if average read depth < 10)
    # =========================================================================
    STAGE_4 = "04_refinement"

    if stage_complete(STAGE_4) and not SIMULATION_SHARD_MODE:
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
            print(f"\n{'='*60}")
            print(f"{r_name}: average read depth = {avg_depth:.1f}x")
            print(f"{'='*60}")
            
            if avg_depth < REFINEMENT_DEPTH_THRESHOLD:
                print(f"  Depth < {REFINEMENT_DEPTH_THRESHOLD}x → Running L1+L2 refinement")
                
                raw_blocks = multi_contig_results[r_name]['simd_block_results']
                global_probs = multi_contig_results[r_name]['simd_probs']
                global_sites = multi_contig_results[r_name]['naive_long_haps'][0]
                num_samples = global_probs.shape[0]
                
                # Warmup numba JIT
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
                
                # Run full refinement pipeline
                t0 = time.time()
                refinement_results = small_block_refine.run_refinement_pipeline(
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
                l2_refined_dd = small_block_refine.dedup_blocks(l2_refined, verbose=True)
                
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
        pipeline_runtime.require_contig_checkpoints(
            checkpoint_store, STAGE_4, region_keys
        )
    _finish_per_contig_stage(STAGE_4)

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE 5: Residual Discovery (find missing founders HDBSCAN missed)
    # =========================================================================
    STAGE_5 = "05_residual_discovery"

    if stage_complete(STAGE_5) and not SIMULATION_SHARD_MODE:
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
            pipeline_runtime.strip_block_evidence(blocks_out)
            save_contig(STAGE_5, r_name, {'simd_block_results': blocks_out})

            print(f"    Output: {len(blocks_out)} blocks, "
                  f"avg haps: {np.mean([len(b.haplotypes) for b in blocks_out]):.1f}")

            # Free this contig's data immediately
            for _k in ('simd_probs', 'simd_block_results'):
                multi_contig_results[r_name].pop(_k, None)

        print(f"\nResidual discovery complete in {time.time()-start:.1f}s")
        pipeline_runtime.require_contig_checkpoints(
            checkpoint_store, STAGE_5, region_keys
        )
    _finish_per_contig_stage(STAGE_5)

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE 6: Hierarchical Assembly (Level 1)
    # =========================================================================
    STAGE_6 = "06_assembly_L1"

    if stage_complete(STAGE_6) and not SIMULATION_SHARD_MODE:
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
            pipeline_runtime.strip_block_evidence(super_blocks)
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
        pipeline_runtime.require_contig_checkpoints(
            checkpoint_store, STAGE_6, region_keys
        )
    _finish_per_contig_stage(STAGE_6)

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE 7: Hierarchical Assembly (Level 2)
    # =========================================================================
    STAGE_7 = "07_assembly_L2"

    if stage_complete(STAGE_7) and not SIMULATION_SHARD_MODE:
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
            pipeline_runtime.strip_block_evidence(super_blocks_L2)
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
        pipeline_runtime.require_contig_checkpoints(
            checkpoint_store, STAGE_7, region_keys
        )
    _finish_per_contig_stage(STAGE_7)

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE 8: Hierarchical Assembly (Level 3)
    # =========================================================================
    STAGE_8 = "08_assembly_L3"

    if stage_complete(STAGE_8) and not SIMULATION_SHARD_MODE:
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
            pipeline_runtime.strip_block_evidence(super_blocks_L3)
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
        pipeline_runtime.require_contig_checkpoints(
            checkpoint_store, STAGE_8, region_keys
        )
    _finish_per_contig_stage(STAGE_8)

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE 9: Hierarchical Assembly (Level 4)
    # =========================================================================
    STAGE_9 = "09_assembly_L4"

    if stage_complete(STAGE_9) and not SIMULATION_SHARD_MODE:
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

            pipeline_runtime.strip_block_evidence(
                multi_contig_results[r_name]['super_blocks_L4']
            )
            save_contig(STAGE_9, r_name, {
                'super_blocks_L4': multi_contig_results[r_name]['super_blocks_L4']
            })

            # Free this contig's input data
            for _k in ('super_blocks_L3', 'simd_probs', 'super_blocks_L4'):
                multi_contig_results[r_name].pop(_k, None)

        print(f"\nHierarchical Assembly (Level 4) complete in {time.time()-start_time:.1f}s")
        _prune_key('super_blocks_L3')
        pipeline_runtime.require_contig_checkpoints(
            checkpoint_store, STAGE_9, region_keys
        )
    _finish_per_contig_stage(STAGE_9)
#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE 10: Terminal whole-bin cavity refinement (canonical final panel)
    # =========================================================================
    # Stage 9 is the raw L4 assembly intermediate. This stage publishes the
    # only founder panel consumed by painting, pedigree and phase correction.
    STAGE_10 = "10_terminal_cavity"

    missing_terminal = [r for r in region_keys if not contig_done(STAGE_10, r)]
    if (stage_complete(STAGE_10) and not SIMULATION_SHARD_MODE
            and missing_terminal):
        raise RuntimeError(
            f"{STAGE_10} is marked complete but lacks: {missing_terminal}"
        )
    if stage_complete(STAGE_10) and not SIMULATION_SHARD_MODE:
        print("\n[RESUME] Skipping terminal cavity refinement "
              "(checkpoint found)")
    else:
        print("\n" + "="*60)
        print("RUNNING: Terminal Cavity Refinement (canonical final panel)")
        print("="*60)
        start_time = time.time()
        terminal_threads = min(
            n_processes,
            pipeline_runtime.available_cpu_count(),
        )
        print(f"  Sequential contigs; {terminal_threads} Numba threads/contig")

        for r_name in region_keys:
            if contig_done(STAGE_10, r_name):
                print(f"  [RESUME] {r_name} already done")
                continue
            print(f"\n  [Terminal] Processing {r_name}...")

            stage9 = load_contig(STAGE_9, r_name)
            l4_blocks = pipeline_runtime.strip_block_probs(
                stage9['super_blocks_L4']
            )
            del stage9
            if len(l4_blocks) != 1:
                raise RuntimeError(
                    f"{r_name}: terminal refinement requires exactly one "
                    f"chromosome-length L4 block; found {len(l4_blocks)}"
                )

            stage2 = load_contig(STAGE_2, r_name)
            source_probs = stage2['simd_probs']
            del stage2
            global_probs = np.ascontiguousarray(source_probs, dtype=np.float32)
            del source_probs
            stage1 = load_contig(STAGE_1, r_name)
            global_sites = np.asarray(stage1['naive_long_haps'][0])
            del stage1

            final_blocks, diagnostics = (
                terminal_cavity_refinement.refine_terminal_cavity_blocks(
                    l4_blocks,
                    global_sites,
                    global_probs,
                    return_diagnostics=True,
                    num_threads=terminal_threads,
                )
            )
            pipeline_runtime.strip_block_probs(final_blocks)
            summary = (
                terminal_cavity_refinement.summarize_terminal_cavity_results(
                    diagnostics
                )
            )
            pipeline_runtime.strip_block_evidence(final_blocks)
            save_contig(STAGE_10, r_name, {
                'super_blocks_L4': final_blocks,
                'terminal_cavity_summary': summary,
            })
            if not contig_done(STAGE_10, r_name):
                raise OSError(f"Failed to checkpoint {STAGE_10}/{r_name}")
            print(
                f"    Changed {summary['changed_founder_cells']} founder "
                f"cells at {summary['changed_sites']} sites"
            )
            del l4_blocks, global_probs, global_sites, final_blocks, diagnostics
            gc.collect()

        print(f"Terminal refinement complete in {time.time()-start_time:.1f}s")
    _finish_per_contig_stage(STAGE_10)


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
        
        _ensure_key(r_name, 'naive_long_haps')
        # Validate the canonical Stage-10 terminal panel.  It may have been
        # evicted from memory after refinement, so require an explicit reload;
        # falling back to L3 here would silently validate a different panel
        # from the one used by every downstream stage.
        _ensure_key(r_name, 'super_blocks_L4')
        final_blocks = multi_contig_results[r_name]['super_blocks_L4']
        level_name = "L4 + terminal cavity"
        
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
    # STAGE 11: VITERBI PAINTING (using DISCOVERED haplotypes from L4 assembly)
    # =============================================================================
    STAGE_11 = "11_viterbi_painting"

    missing_painting = [
        r for r in region_keys if not contig_done(STAGE_11, r)
    ]
    if (stage_complete(STAGE_11) and not SIMULATION_SHARD_MODE
            and missing_painting):
        raise RuntimeError(
            f"{STAGE_11} is marked complete but lacks: "
            f"{missing_painting}"
        )

    if stage_complete(STAGE_11) and not SIMULATION_SHARD_MODE:
        print(f"\n[RESUME] Skipping Viterbi painting (checkpoint found)")
    else:
        print("\n" + "="*60)
        print("RUNNING: Viterbi Painting (Discovered Haplotypes)")
        print("="*60)

        with paint_samples.PaintingPoolManager(num_processes=n_processes) as painter:
            for r_name in region_keys:
                if contig_done(STAGE_11, r_name):
                    print(f"  [RESUME] {r_name} already done")
                    continue
                print(f"\n[Viterbi Painting] Processing Region: {r_name}")

                # Strictly paint the canonical final panel from Stage 10.
                terminal_payload = load_contig(STAGE_10, r_name)
                final_blocks = terminal_payload['super_blocks_L4']
                if len(final_blocks) != 1:
                    raise RuntimeError(
                        f"{r_name}: painting requires exactly one final L4 "
                        f"block; found {len(final_blocks)}"
                    )
                discovered_block = final_blocks[0]
                del terminal_payload, final_blocks

                _ensure_key(r_name, 'simd_probs')
                _ensure_key(r_name, 'naive_long_haps')
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
                if not SIMULATION_SHARD_MODE:
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

                founder_block = pipeline_runtime.compact_founder_block(
                    discovered_block
                )
                save_contig(STAGE_11, r_name, {
                    'tolerance_result': painting_result,
                    pipeline_runtime.FOUNDER_BLOCK_KEY: founder_block,
                    pipeline_runtime.SAMPLE_IDS_KEY: tuple(
                        str(value) for value in sample_names
                    ),
                })
                del founder_block

                # Free this contig's data immediately
                for _k in ('simd_probs', 'tolerance_result', 'super_blocks_L4'):
                    multi_contig_results[r_name].pop(_k, None)

        missing_painting = [
            r for r in region_keys if not contig_done(STAGE_11, r)
        ]
        if missing_painting:
            raise OSError(f"Failed to checkpoint {STAGE_11}: {missing_painting}")
        print("\nViterbi Painting complete.")
        _prune_key('simd_probs')
        _prune_key('simd_priors')
    _finish_per_contig_stage(STAGE_11)
    if SIMULATION_SHARD_MODE:
        print(
            "[SHARD] Selected contigs completed through Stage 11; "
            "exiting before global Stage 12"
        )
        raise SystemExit(0)

#%%
if __name__ == '__main__' and not SKIP_VALIDATIONS_PAINTING:
    # ==========================================================================
    # VALIDATE: Painted Samples Output Against Ground Truth (topology-based)
    # ==========================================================================
    # Per-sample, per-contig assessment of paint_samples (Viterbi painting,
    # Stage 11) output before any downstream correction is applied.
    #
    # The painting at this stage is an UNORDERED pair of founder IDs at each
    # position (no phase information; phase is introduced in Stage 13).  The
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
    # allele accuracy is reported by Stage 13 BEFORE/AFTER on the same
    # painting and would be redundant here.
    #
    # Validation is read-only -- no checkpointing.
    print(f"\n{'='*60}")
    print("Validating Painted Samples (paint_samples / Stage 11) Topology")
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
        # will pull it from 11_viterbi_painting whether we just ran Stage 11
        # or are resuming from a completed checkpoint.
        try:
            _ensure_key(r_name, 'tolerance_result')
            _ensure_key(r_name, 'truth_painting')
            _ensure_key(r_name, 'naive_long_haps')
        except FileNotFoundError as e:
            print(f"  {r_name}: SKIP -- {e}")
            continue

        # Use the same canonical final panel that Stage 11 painted.
        try:
            _ensure_key(r_name, 'super_blocks_L4')
            discovered_block = multi_contig_results[r_name]['super_blocks_L4'][0]
        except FileNotFoundError as e:
            print(f"  {r_name}: SKIP -- no final L4 panel ({e})")
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
        # stages (e.g. Stage 13) if needed.
        multi_contig_results[r_name].pop('tolerance_result', None)
        # NOTE: truth_painting is also used by Stage 13 evaluation; leave
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
    # STAGE 12: CALIBRATED COMBINED-V1 B1 PEDIGREE INFERENCE
    # =============================================================================
    STAGE_12 = "12_pedigree_inference_current_b1_combined_v1_calibrated_v1"
    pedigree_config = pedigree_pipeline.build_current_pedigree_config()

    if stage_complete(STAGE_12) and not checkpoint_store.global_done(STAGE_12):
        raise RuntimeError(f"{STAGE_12} is complete but lacks _global")
    if stage_complete(STAGE_12):
        print(f"\n[RESUME] Skipping current B1 pedigree inference (checkpoint found)")
        pedigree_payload = load_global(STAGE_12)
        pedigree_df = pedigree_payload['pipeline_control_relationships']
        scientific_pedigree_df = pedigree_payload['scientific_relationships']
        complete_pedigree_df = pedigree_payload['complete_relationships']
        tier_b_pedigree_df = pedigree_payload['tier_b_relationships']
        smart_diagnostics = pedigree_payload['smart_diagnostics']
        if pedigree_payload['smart_config'] != pedigree_config:
            raise RuntimeError("current B1 pedigree checkpoint config changed")
        for label, frame in (
            ('pipeline_control', pedigree_df),
            ('scientific', scientific_pedigree_df),
            ('complete', complete_pedigree_df),
            ('tier_b', tier_b_pedigree_df),
        ):
            if frame['Sample'].tolist() != list(sample_names):
                raise RuntimeError(f"{label} checkpoint sample order changed")
        del pedigree_payload
    else:
        print("\n" + "="*60)
        print("RUNNING: Current B1 Multi-Contig Pedigree Inference")
        print("="*60)

        contig_inputs = []
        for r_name in region_keys:
            painting_payload = load_contig(STAGE_11, r_name)
            pipeline_runtime.validate_painting_bundle(
                painting_payload,
                expected_sample_ids=sample_names,
                context=f"{STAGE_11}/{r_name}",
            )
            discovered_block = pipeline_runtime.compact_founder_block(
                painting_payload[pipeline_runtime.FOUNDER_BLOCK_KEY]
            )
            contig_inputs.append({
                'tolerance_painting': painting_payload['tolerance_result'],
                'founder_block': discovered_block,
            })
            del painting_payload


        # Release redundant final-panel references before shared-memory handoff.
        for r_name in region_keys:
            multi_contig_results[r_name].pop('super_blocks_L4', None)

        pedigree_result = pedigree_inference.infer_pedigree_for_pipeline(
            contig_inputs,
            sample_ids=sample_names,
            config=pedigree_config,
            top_k=20,
            n_workers=n_processes,
        )
        if pedigree_result.smart_config != pedigree_config:
            raise RuntimeError("current pedigree engine changed the requested B1 config")

        # Only this adapter is passed to downstream phase correction.
        pedigree_df = pedigree_result.pipeline_control_relationships
        scientific_pedigree_df = pedigree_result.relationships
        complete_pedigree_df = pedigree_result.complete_relationships
        tier_b_pedigree_df = pedigree_result.tier_b_relationships
        smart_diagnostics = pedigree_result.smart_diagnostics
        for label, frame in (
            ('pipeline_control', pedigree_df),
            ('scientific', scientific_pedigree_df),
            ('complete', complete_pedigree_df),
            ('tier_b', tier_b_pedigree_df),
        ):
            if frame['Sample'].tolist() != list(sample_names):
                raise RuntimeError(f"{label} pedigree sample order changed")

        exports = {
            "pedigree_inference_current_scientific.csv": scientific_pedigree_df,
            "pedigree_inference_current_complete.csv": complete_pedigree_df,
            "pedigree_inference_current_tier_b.csv": tier_b_pedigree_df,
            "pedigree_inference_current_diagnostics.csv": smart_diagnostics,
        }
        for filename, frame in exports.items():
            output_csv = os.path.join(output_dir, filename)
            frame.to_csv(output_csv, index=False)
            print(f"Pedigree output saved to: {output_csv}")
        pedigree_inference.draw_pedigree_tree(
            scientific_pedigree_df,
            output_file=os.path.join(
                output_dir, "pedigree_tree_current_scientific.png"
            ),
        )

        if 'truth_pedigree' in dir():
            print("\n--- Current Pedigree Validation ---")
            for view_name, inferred_frame in (
                ('scientific', scientific_pedigree_df),
                ('complete', complete_pedigree_df),
                ('tier_b', tier_b_pedigree_df),
            ):
                validation_df = compare_relationships_to_truth(
                    truth_pedigree, inferred_frame
                )
                descendants = validation_df["TruthGeneration"].isin(
                    ["F2", "F3"]
                )
                state_acc = (
                    validation_df.loc[
                        descendants, "ParentState_Match"
                    ].mean() * 100
                )
                parent_acc = (
                    validation_df.loc[
                        descendants, "Parents_Match"
                    ].mean() * 100
                )
                validation_df.to_csv(os.path.join(
                    output_dir, f"pedigree_validation_current_{view_name}.csv"
                ), index=False)
                print(
                    f"{view_name}: observed-parent-state accuracy "
                    f"(F2+F3)={state_acc:.2f}%, parentage accuracy "
                    f"(F2+F3)={parent_acc:.2f}%"
                )

        save_global(STAGE_12, {
            'pipeline_control_relationships': pedigree_df,
            'scientific_relationships': scientific_pedigree_df,
            'complete_relationships': complete_pedigree_df,
            'tier_b_relationships': tier_b_pedigree_df,
            'smart_diagnostics': smart_diagnostics,
            'smart_config': pedigree_config,
        })
        if not checkpoint_store.global_done(STAGE_12):
            raise OSError(f"Failed to checkpoint {STAGE_12}/_global")
        del contig_inputs
        gc.collect()
        mark_stage_complete(STAGE_12)

#%%
if __name__ == '__main__':
    # =============================================================================
    # STAGE 13: PHASE CORRECTION (using DISCOVERED haplotypes)
    # =============================================================================
    STAGE_13 = "13_phase_correction_current_b1_combined_v1_calibrated_v1"

    missing_phase = [
        r for r in region_keys if not contig_done(STAGE_13, r)
    ]
    if stage_complete(STAGE_13) and missing_phase:
        raise RuntimeError(
            f"{STAGE_13} is marked complete but lacks: "
            f"{missing_phase}"
        )

    if stage_complete(STAGE_13):
        print(f"\n[RESUME] Skipping phase correction (checkpoint found)")
        # Load per-contig phase correction results for validation
        for r_name in region_keys:
            s13 = load_contig(STAGE_13, r_name)
            for k, v in s13.items():
                multi_contig_results.setdefault(r_name, {})[k] = v
            del s13
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
        # F1 recoloring and propagation use the same final panel bundled with
        # the Stage-11 painting.
        print("\n" + "="*60)
        print("LOADING final founder_blocks for F1 recoloring + propagation")
        print("="*60)
        _t0 = time.time()
        founder_blocks = pipeline_runtime.load_founder_blocks_parallel(
            checkpoint_store,
            region_keys,
            ((STAGE_11, pipeline_runtime.FOUNDER_BLOCK_KEY),),
            max_workers=n_processes,
            require_all=True,
        )
        for r_name, founder_block in founder_blocks.items():
            multi_contig_results.setdefault(r_name, {})[
                'founder_block'
            ] = founder_block
        del founder_blocks
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
            save_contig(STAGE_13, r_name, d)

        missing_phase = [
            r for r in region_keys if not contig_done(STAGE_13, r)
        ]
        if missing_phase:
            raise OSError(f"Failed to checkpoint {STAGE_13}: {missing_phase}")

        # Free everything — phase validation reloads from checkpoints
        multi_contig_results = {r: {'naive_long_haps': multi_contig_results[r].get('naive_long_haps')}
                                for r in region_keys if 'naive_long_haps' in multi_contig_results.get(r, {})}
        gc.collect()
        mark_stage_complete(STAGE_13)

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

    # Reload paintings from Stage 13 checkpoints (freed during save)
    def _load_stage13(r_name):
        if contig_done(STAGE_13, r_name):
            s13 = load_contig(STAGE_13, r_name)
            for k, v in s13.items():
                multi_contig_results.setdefault(r_name, {})[k] = v
            del s13
        return r_name

    print("\nLoading phase correction + validation data from checkpoints...")
    load_start = time.time()
    with ThreadPoolExecutor(max_workers=min(8, len(region_keys))) as executor:
        list(executor.map(_load_stage13, region_keys))
    print(f"Stage 13 reload: {time.time()-load_start:.1f}s")

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
        valid_mask = ((corr_allele1 != -1) & (corr_allele2 != -1)
                      & (true_allele1 != -1) & (true_allele2 != -1))
        direct_match = valid_mask & (corr_allele1 == true_allele1) & (corr_allele2 == true_allele2)
        flipped_match = valid_mask & (corr_allele1 == true_allele2) & (corr_allele2 == true_allele1)
        correct_either = direct_match | flipped_match
        n_direct = np.sum(direct_match)
        n_flipped = np.sum(flipped_match)
        if n_direct >= n_flipped:
            track1_correct = valid_mask & (corr_allele1 == true_allele1)
            track2_correct = valid_mask & (corr_allele2 == true_allele2)
            dominant_phase = "Direct"
        else:
            track1_correct = valid_mask & (corr_allele1 == true_allele2)
            track2_correct = valid_mask & (corr_allele2 == true_allele1)
            dominant_phase = "Flipped"
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

    # Decode both BEFORE and AFTER paintings against the same canonical
    # final panel so the comparison measures phase correction alone.
    def _load_validation_keys(r_name):
        _ensure_key(r_name, 'truth_painting')
        _ensure_key(r_name, 'naive_long_haps')
        _ensure_key(r_name, 'tolerance_result')
        return r_name

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=min(8, len(region_keys))) as executor:
        list(executor.map(_load_validation_keys, region_keys))
    print(f"  Validation data loaded in {time.time()-t0:.1f}s")

    contig_shared = {}
    for r_name in region_keys:
        data = multi_contig_results[r_name]
        if 'truth_painting' not in data or 'founder_block' not in data:
            continue
        truth = data['truth_painting']
        final_block = data['founder_block']
        positions = np.asarray(final_block.positions)
        dense_haps, _ = phase_correction.founder_block_to_dense(final_block)

        orig_sites, orig_haps = data['naive_long_haps']
        orig_haps_concrete = simulate_sequences.concretify_haps(orig_haps)
        site_indices = np.searchsorted(orig_sites, positions)
        site_indices = np.clip(site_indices, 0, len(orig_sites) - 1)
        true_dense_haps = np.array(
            [h[site_indices] for h in orig_haps_concrete], dtype=np.int8
        )
        contig_shared[r_name] = (
            truth, positions, dense_haps, true_dense_haps
        )

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