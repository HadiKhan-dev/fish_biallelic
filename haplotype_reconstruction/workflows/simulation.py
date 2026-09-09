"""workflows / simulation for the canonical reconstruction pipeline."""
from __future__ import annotations

import json
from pathlib import Path


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
import hashlib
import math
import haplotype_reconstruction.assembly.pipeline as assembly_pipeline
import haplotype_reconstruction.core.environment as core_environment
import haplotype_reconstruction.core.genetic_map as core_genetic_map
import haplotype_reconstruction.core.haplotypes as core_haplotypes
import haplotype_reconstruction.core.parallel as core_parallel
import haplotype_reconstruction.core.runtime as core_runtime
import haplotype_reconstruction.core.variants as core_variants
import haplotype_reconstruction.discovery.blocks as discovery_blocks
import haplotype_reconstruction.discovery.search as discovery_search
import haplotype_reconstruction.pedigree.pipeline as pedigree_pipeline
import haplotype_reconstruction.recombination.model as module_recombination_model
import haplotype_reconstruction.recombination.pipeline as recombination_pipeline
import haplotype_reconstruction.refinement.pipeline as refinement_pipeline
import haplotype_reconstruction.refinement.conditioning as refinement_conditioning
import haplotype_reconstruction.simulation.pedigree as simulation_pedigree
import haplotype_reconstruction.simulation.templates as simulation_templates
import haplotype_reconstruction.workflows.reconstruction as workflows_reconstruction

CHECKPOINT_DIR = os.environ.get(
    "BHD_SIM_CHECKPOINT_DIR",
    "work/runs/seed_400/checkpoints"
)


SIMULATION_OUTPUT_DIR = os.environ.get(
    "BHD_SIM_OUTPUT_DIR",
    "work/runs/seed_400"
)


_SIMULATION_SEED_TEXT = os.environ.get(
    "BHD_SIMULATION_SEED", "400"
).strip()


SIMULATION_SEED = (
    None
    if _SIMULATION_SEED_TEXT.lower() in {"none", "random"}
    else int(_SIMULATION_SEED_TEXT)
)


_SIMULATION_CONTIGS_TEXT = os.environ.get("BHD_SIM_CONTIGS")


SIMULATION_READ_DEPTH = float(os.environ.get("BHD_SIM_READ_DEPTH", "5.0"))


if not math.isfinite(SIMULATION_READ_DEPTH) or SIMULATION_READ_DEPTH <= 0.0:
    raise ValueError("BHD_SIM_READ_DEPTH must be finite and positive")


_PER_CONTIG_STAGE_NAMES = (
    "01_blocks",
    "09_painting",
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


def _finish_simulation_stage_checkpoints(
        checkpoint_store, stage, contigs, shard_mode):
    """Require every contig output and publish a marker only for full runs."""
    core_runtime.require_contig_checkpoints(
        checkpoint_store, stage, contigs
    )
    if not shard_mode and not checkpoint_store.stage_complete(stage):
        checkpoint_store.mark_stage_complete(stage)


def run():
    """Execute or resume the configured, chromosome-checkpointed workflow."""
    import os
    import sys
    from datetime import datetime

    # FORCE NUMPY/BLAS TO USE 1 THREAD PER PROCESS
    # (Numba threading is now managed by core/parallel.py — do NOT set
    #  NUMBA_NUM_THREADS or NUMBA_THREADING_LAYER here)
    core_environment.force_single_threaded_numeric_libraries()

    # =============================================================================
    # DUAL LOGGING: Console + File
    # =============================================================================
    # All print() output goes to both the terminal and a timestamped log file.
    # tqdm progress bars still display on the terminal only (they use stderr).
    # If the SSH connection drops, the log file preserves all output.


    os.makedirs(os.environ.get("HAPLOTYPES_LOG_DIR", "work/logs"), exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    seed_label = "random" if SIMULATION_SEED is None else str(SIMULATION_SEED)
    shard_log_suffix = (
        f"_shard{SIMULATION_SHARD_LOG_ID}" if SIMULATION_SHARD_MODE else ""
    )
    log_path = os.path.join(os.environ.get("HAPLOTYPES_LOG_DIR", "work/logs"), f"run_{run_timestamp}.log")
    sys.stdout = core_runtime.TeeOutput(log_path, sys.stdout)
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

    import numpy as np
    import time
    import warnings
    import platform
    import gc
    from dataclasses import asdict


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
    # Generating truth and the inference prior are deliberately independent.
    simulation_rate_maps = core_genetic_map.load_genetic_maps(
        os.environ.get('BHD_SIMULATION_RECOMBINATION_MAP'),
        float(os.environ.get('BHD_SIMULATION_RECOMBINATION_RATE_CM_PER_MB', '5.0')))
    simulation_genetic_maps = simulation_rate_maps if simulation_rate_maps.maps else None


    np.seterr(divide='ignore', invalid="ignore")

    if platform.system() != "Windows":
        #os.nice(15)
        print(f"Main process ({os.getpid()}) niceness set to: {os.nice(0)}")


    n_processes = int(os.environ.get("BHD_NUM_PROCESSES", str(core_runtime.available_cpu_count())))
    available_cpus = core_runtime.available_cpu_count()
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
    # core/parallel.py already called set_forkserver_preload().
    _warmup_pool = core_parallel.NonDaemonicForkserverPool(1)
    _warmup_pool.terminate()
    _warmup_pool.join()
    del _warmup_pool
    print("Forkserver started (lightweight, pre-data).")
    print(f"Numba threading layer: {os.environ.get('NUMBA_THREADING_LAYER', 'not set')}")

    # =============================================================================
    # PER-CONTIG CHECKPOINTING
    # =============================================================================
    # Each stage gets a subdirectory.  Each contig gets its own checkpoint
    # file (a protocol-5/Blosc frame, suffix ".p5.b2"; see core/checkpoints).
    # The format-qualified done marker means all contigs completed.
    #
    # On resume, _ensure_key loads ONLY the keys a stage needs from checkpoints,
    # avoiding the monolithic pickle that caused OOM.
    #
    # Heavy arrays are loaded and released one contig at a time. Stage 2 has
    # atomic preprocessing, L1-L4, and final component-painting checkpoints.
    #
    # To force a full re-run, remove the exact configured checkpoint directory
    # after verifying its resolved path. To resume, keep completed stage files.

    checkpoint_store = core_runtime.CheckpointStore(
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
            stage for stage in ("00_founder_templates", "00_simulated_reads")
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


    # Source checkpoint for each value needed before the canonical Stage 2.
    _KEY_SOURCE = {
        'naive_long_haps': '00_founder_templates',
        'simulated_reads': '00_simulated_reads',
        'simd_genomic_data': '00_simulated_reads',
        'simd_probs': '00_simulated_reads',
        'simd_priors': '00_simulated_reads',
        'truth_painting': '00_simulated_reads',
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

    vcf_path = os.environ.get("HAPLOTYPES_VCF", "work/data/fish_vcf_restriped/AsAc.AulStuGenome.biallelic.bcf.gz")

    # Define the regions you want to use for inference.
    regions_config = configured_regions([
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
        ], template_regions=True)

    block_size = 100000
    shift_size = 50000

    multi_contig_results = {}

    total_start = time.time()

    # =========================================================================
    # STAGE 1: VCF Loading + Haplotype Discovery + Naive Linking
    # =========================================================================
    STAGE_1 = "00_founder_templates"
    template_directory = os.environ.get("HAPLOTYPES_TEMPLATES")
    if template_directory:
        template_identity = {
            region['contig']: hashlib.sha256(
                (Path(template_directory) / (region['contig'] + '.npz')).read_bytes()
            ).hexdigest() for region in regions_config
        }
    else:
        input_path = Path(vcf_path).resolve()
        template_identity = {'vcf': str(input_path), 'size': input_path.stat().st_size,
                             'mtime_ns': input_path.stat().st_mtime_ns,
                             'regions': regions_config, 'block_size': block_size, 'shift_size': shift_size}
    checkpoint_store.bind_stage_identity(STAGE_1, {
        'discovery': stage1_identity_record, 'inputs': template_identity,
    })
    if template_directory and not stage_complete(STAGE_1):
        for region in regions_config:
            contig = region['contig']
            if contig_done(STAGE_1, contig):
                continue
            template_path = Path(template_directory) / f"{contig}.npz"
            with np.load(template_path, allow_pickle=False) as template:
                positions = template['positions']
                probabilities = template['allele_probabilities']
            if probabilities.ndim != 3 or probabilities.shape[1:] != (len(positions), 2):
                raise ValueError(f"{contig}: invalid founder-template axes")
            save_contig(STAGE_1, contig, {'naive_long_haps': [positions, list(probabilities)],
                'stage1_backend': discovery_blocks.STAGE1_BACKEND,
                'stage1_config': stage1_config_record})
        mark_stage_complete(STAGE_1)

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
            genomic_data = core_variants.cleanup_block_reads_list(
                vcf_path,
                region['contig'],
                start_block_idx=region['start'],
                end_block_idx=region['end'],
                block_size=block_size,
                shift_size=shift_size,
                num_processes=n_processes
            )
            print(f"  [Loader] Loaded {len(genomic_data)} blocks in {time.time() - start:.2f}s")

            # 2. Run Haplotype Discovery
            start = time.time()
            block_results = discovery_blocks.generate_all_block_haplotypes(
                genomic_data,
                num_processes=n_processes,
                discovery_config=stage1_config,
            )

            valid_blocks = [b for b in block_results if len(b.positions) > 0]
            block_results = core_haplotypes.BlockResults(valid_blocks)

            print(f"  [Discovery] Haplotypes generated in {time.time() - start:.2f}s")

            # 3. Run Naive Linker (to get long templates for simulation)
            start = time.time()
            (naive_blocks, naive_long_haps) = simulation_templates.build_founder_templates(
                block_results,
                num_long_haps=6
            )
            print(f"  [Naive Linker] Chained {len(naive_long_haps[1])} haps in {time.time() - start:.2f}s")

            # Store only naive_long_haps (genomic_data + block_results are huge, never needed again)
            multi_contig_results[region['contig']] = {
                "naive_long_haps": naive_long_haps
            }
            save_contig(STAGE_1, r_name, {
                'naive_long_haps': naive_long_haps,
                'stage1_backend': discovery_blocks.STAGE1_BACKEND,
                'stage1_config': stage1_config_record,
            })
            del genomic_data, block_results, naive_blocks, naive_long_haps
            gc.collect()

        print(f"\nAll regions processed in {time.time() - total_start:.2f}s")
        core_runtime.require_contig_checkpoints(
            checkpoint_store,
            STAGE_1,
            [region['contig'] for region in regions_config],
        )
        mark_stage_complete(STAGE_1)

    # =========================================================================
    # STAGE 2: Simulation + Post-processing
    # =========================================================================
    STAGE_2 = "00_simulated_reads"
    generation_sizes = tuple(json.loads(os.environ.get("HAPLOTYPES_GENERATIONS", "[20, 100, 200]")))
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
        'template_identity': template_identity,
        'truth_payload': 'alleles_and_raw_crossovers_v1',
        'recombination_rate_per_bp': simulation_rate_maps.default_rate_cm_per_mb / 1e8,
        **({'genetic_maps': simulation_genetic_maps.identity()}
           if simulation_genetic_maps is not None else {}),
        'mutation_rate_per_bp': mutate_rate,
        'read_depth': SIMULATION_READ_DEPTH,
        'read_error_rate': 0.02,
        'snps_per_block': 200,
        'snp_shift': 200,
        'stage1_backend': discovery_blocks.STAGE1_BACKEND,
        'stage1_config': stage1_config_record,
        'truth_haplotype_completion': 'seeded_unbiased_tie_resolution_v1',
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
            core_runtime.require_contig_checkpoints(
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
        core_runtime.require_contig_checkpoints(
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

        # 1. Load probabilistic founder templates. Explicit probability ties
        # are resolved only after the realized seed is durably bound below.
        founder_templates = []
        sites_list = []
        region_keys = []

        for r_name in [r['contig'] for r in regions_config]:
            _ensure_key(r_name, 'naive_long_haps')
            data = multi_contig_results[r_name]
            sites, haplotypes = data['naive_long_haps']
            founder_templates.append(haplotypes)
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

        # Materialize complete truth without assigning every unknown tie to
        # the reference allele. One deterministic child seed is used per contig.
        founder_seed_sequence = np.random.SeedSequence(
            [realized_simulation_seed, 2_000_000]
        )
        founder_seeds = founder_seed_sequence.spawn(len(region_keys))
        founders_list = []
        for haplotypes, child_seed in zip(founder_templates, founder_seeds):
            concrete_haplotypes = (
                simulation_pedigree.materialize_simulation_haplotypes(
                    haplotypes, np.random.default_rng(child_seed)
                )
            )
            founders_list.append(
                simulation_pedigree.pairup_haps(concrete_haplotypes)
            )
        del founder_templates, founder_seeds

        # 2. Run Multi-Contig Simulation
        print(f"Running Multi-Contig Simulation for {len(region_keys)} regions...")

        if STRESS_TEST_MUTATIONS:
            print(f"STRESS TEST MODE: Using mutation rate {mutate_rate} (~1% per generation)")
        else:
            print(f"Normal mode: Using mutation rate {mutate_rate} (minimal mutations)")

        t0 = time.time()
        (all_offspring_lists, truth_pedigree, truth_paintings_lists,
         truth_crossovers) = simulation_pedigree.simulate_pedigree(
            founders_list,
            sites_list,
            generation_sizes,
            recomb_rate=stage2_run_spec['recombination_rate_per_bp'],
            mutate_rate=stage2_run_spec['mutation_rate_per_bp'],
            output_plot=None,
            parallel=True,
            num_processes=n_processes,
            seed=realized_simulation_seed,
            genetic_maps=simulation_genetic_maps, contig_names=region_keys,
            return_crossover_events=True,
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
                    simulation_pedigree._process_single_contig_postprocessing((
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
                payload['truth_founder_haplotypes'] = tuple(
                    np.asarray(haplotype, dtype=np.int8)
                    for parent in founders_list[contig_index]
                    for haplotype in parent
                )
                payload['truth_alleles'] = np.asarray(all_offspring_lists[contig_index], dtype=np.int8).transpose(0, 2, 1)
                payload['truth_crossovers'] = truth_crossovers[contig_index]
                save_contig(STAGE_2, r_name, payload)
                if not contig_done(STAGE_2, r_name):
                    raise OSError(
                        f"Failed to checkpoint {STAGE_2}/{r_name}"
                    )
                del payload, result

            # Release each contig as soon as its checkpoint is durable.
            truth_crossovers[contig_index] = None
            all_offspring_lists[contig_index] = None
            truth_paintings_lists[contig_index] = None
            founders_list[contig_index] = None
            sites_list[contig_index] = None
            gc.collect()

        core_runtime.require_contig_checkpoints(
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

    # Ensure output_dir and globals exist for all subsequent stages
    output_dir = SIMULATION_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    if 'region_keys' not in dir() or region_keys is None:
        g = load_global('00_simulated_reads')
        region_keys = g['region_keys']
        sample_names = g['sample_names']
        truth_pedigree = g['truth_pedigree']
        del g
    all_region_keys = list(region_keys)
    if SIMULATION_SHARD_MODE:
        if not checkpoint_store.global_done("00_simulated_reads"):
            raise RuntimeError(
                "BHD_SIM_CONTIGS requires the global Stage-2 manifest"
            )
        core_runtime.require_contig_checkpoints(
            checkpoint_store, "00_founder_templates", all_region_keys
        )
        core_runtime.require_contig_checkpoints(
            checkpoint_store, "00_simulated_reads", all_region_keys
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
    STAGE_3 = "01_blocks"
    checkpoint_store.bind_stage_identity(
        STAGE_3, stage1_identity_record
    )
    stage1_source_manifest = {
        'sample_ids': sample_names,
        'contigs': all_region_keys,
        'genotype_evidence_mode': (
            workflows_reconstruction.SUPPORTED_GENOTYPE_EVIDENCE_MODE
        ),
        'observed_call_mask_mode': (
            workflows_reconstruction.EXACT_OBSERVED_MASK_MODE
        ),
        'stage1_backend': discovery_blocks.STAGE1_BACKEND,
        'stage1_config': stage1_config_record,
    }
    checkpoint_store.bind_global_manifest(
        STAGE_3, stage1_source_manifest
    )

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
            _ensure_key(r_name, 'simulated_reads')
            _ensure_key(r_name, 'naive_long_haps')
            simd_genomic_data = multi_contig_results[r_name]['simd_genomic_data']
            simulated_reads = multi_contig_results[r_name]['simulated_reads']
            global_sites = np.asarray(
                multi_contig_results[r_name]['naive_long_haps'][0]
            )

            t_chr = time.time()
            simd_block_results = discovery_blocks.generate_all_block_haplotypes(
                simd_genomic_data,
                num_processes=n_processes,
                discovery_config=stage1_config,
            )
            disc_time = time.time() - t_chr

            valid_blocks = [b for b in simd_block_results if len(b.positions) > 0]
            simd_block_results = core_haplotypes.BlockResults(valid_blocks)

            global_observed_mask = (
                workflows_reconstruction.observed_call_mask_from_read_counts(
                    simulated_reads
                )
            )
            save_contig(STAGE_3, r_name, {
                'block_results': simd_block_results,
                'global_sites': global_sites,
                'global_observed_mask': global_observed_mask,
                'observed_call_mask_mode': (
                    workflows_reconstruction.EXACT_OBSERVED_MASK_MODE
                ),
                'genotype_evidence_mode': (
                    workflows_reconstruction.SUPPORTED_GENOTYPE_EVIDENCE_MODE
                ),
                'stage1_backend': discovery_blocks.STAGE1_BACKEND,
                'stage1_config': stage1_config_record,
            })

            hap_counts = [len(b.haplotypes) for b in valid_blocks]
            print(f"    {len(valid_blocks)} blocks, haps/block: "
                  f"min={min(hap_counts)}, max={max(hap_counts)}, mean={np.mean(hap_counts):.1f} "
                  f"[discovery: {disc_time:.1f}s]")

            # Free this contig's data immediately (don't accumulate across contigs)
            for _k in (
                    'simd_genomic_data', 'simulated_reads', 'naive_long_haps',
            ):
                multi_contig_results[r_name].pop(_k, None)

        print(f"\nBlock haplotype discovery complete in {time.time()-start:.1f}s")
        _prune_key('simd_genomic_data')
        core_runtime.require_contig_checkpoints(
            checkpoint_store, STAGE_3, region_keys
        )
    _finish_per_contig_stage(STAGE_3)

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
    print("STAGE 2: DISCOVERED BLOCKS -> COMPONENT T09")
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
        source_stage=STAGE_3,
        probabilities_stage=STAGE_2,
        probabilities_key='simd_probs',
        all_contigs=all_region_keys,
        publish_completion=not SIMULATION_SHARD_MODE,
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

    if SIMULATION_STOP_AFTER_STAGE == workflows_reconstruction.PAINTING_STAGE:
        print("[STOP] Requested stop after typed component T09.")
        raise SystemExit(0)
    _stage10_summaries, stage10_payload = pedigree_pipeline.run_pedigree(
        checkpoint_store, region_keys, sample_names,
        output_dir=output_dir, n_workers=n_processes,
        raw_gl_stage=STAGE_2, raw_sites_stage=STAGE_3,
        raw_gl_key='simd_probs', parent_eligibility=None,
        all_contigs=all_region_keys, publish_global=not SIMULATION_SHARD_MODE,
        genetic_maps=inference_genetic_maps, recombination_rate=inference_recombination_rate,
    )
    print("STAGE 11: pedigree-conditioned refinement and canonical final phase polishing")
    refinement_pipeline.run_refinement(
        checkpoint_store, all_region_keys, sample_names,
        pedigree_payload=stage10_payload, output_dir=output_dir,
        raw_gl_stage=STAGE_2, raw_sites_stage=STAGE_3,
        raw_gl_key='simd_probs', n_workers=n_processes,
        genetic_maps=inference_genetic_maps,
        config=replace(refinement_conditioning.config_from_environment(),
                       recombination_rate=inference_recombination_rate),
    )
    recombination_pipeline.run_recombination(
        checkpoint_store, all_region_keys, sample_names,
        pedigree_payload=stage10_payload, output_dir=output_dir,
        n_workers=n_processes,
        genetic_maps=inference_genetic_maps,
        config=module_recombination_model.RecombinationMapConfig(recombination_rate=inference_recombination_rate),
    )


if __name__ == "__main__":
    run()
