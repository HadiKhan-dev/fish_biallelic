"""Checkpointed known-pedigree simulation and end-to-end reconstruction driver."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from ..core.environment import configured_regions
import haplotype_reconstruction.assembly.pipeline as assembly_pipeline
import haplotype_reconstruction.core.environment as core_environment
import haplotype_reconstruction.core.genetic_map as core_genetic_map
import haplotype_reconstruction.core.haplotypes as core_haplotypes
import haplotype_reconstruction.core.parallel as core_parallel
import haplotype_reconstruction.core.runtime as core_runtime
import haplotype_reconstruction.core.variants as core_variants
import haplotype_reconstruction.discovery.blocks as discovery_blocks
import haplotype_reconstruction.discovery.search as discovery_search
import haplotype_reconstruction.simulation.pedigree as simulation_pedigree
import haplotype_reconstruction.simulation.templates as simulation_templates
import haplotype_reconstruction.workflows.reconstruction as workflows_reconstruction
from .downstream import run_downstream

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
    "block_discovery",
    "painting",
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
    """Validate requested names and return them in the simulation manifest order."""
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


@core_runtime.logged_workflow("simulation")
def run():
    """Execute or resume the configured, chromosome-checkpointed workflow."""
    import os

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
    import platform
    import gc
    from dataclasses import asdict


    discovery_config = discovery_search.ReversibleCavitySearchConfig()
    discovery_config_record = asdict(discovery_config)
    discovery_identity_record = {
        "backend": discovery_blocks.DISCOVERY_BACKEND,
        "config": discovery_config_record,
    }




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
    # Reuse native code across four batches; trim between batches and retain
    # periodic process recycling to bound long-lived allocator fragmentation.
    WORKER_MAXTASKS = 4

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
    # Heavy arrays are loaded and released one contig at a time. assembly has
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
            stage for stage in ("founder_templates", "simulated_reads")
            if not stage_complete(stage)
        ]
        if missing_shared_stages:
            raise RuntimeError(
                "BHD_SIM_CONTIGS requires globally completed shared "
                "founder templates and simulated reads; missing format-qualified completion markers for: "
                f"{missing_shared_stages}"
            )
        print(
            "[SHARD] Verified globally completed shared founder templates and simulated reads; "
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


    # Source checkpoint for each value needed before the canonical assembly.
    _KEY_SOURCE = {
        'naive_long_haps': 'founder_templates',
        'simulated_reads': 'simulated_reads',
        'simd_genomic_data': 'simulated_reads',
        'simd_probs': 'simulated_reads',
        'simd_priors': 'simulated_reads',
        'truth_painting': 'simulated_reads',
    }

    def _ensure_key(r_name, key, *additional_keys):
        """Load requested fields together, reading each source checkpoint once."""
        mcr = multi_contig_results.setdefault(r_name, {})
        missing = [name for name in (key, *additional_keys) if name not in mcr]
        for src in dict.fromkeys(_KEY_SOURCE[name] for name in missing):
            if not contig_done(src, r_name):
                raise FileNotFoundError(f"Cannot find {src} for {r_name}")
            ckpt = load_contig(src, r_name)
            for name in missing:
                if _KEY_SOURCE[name] == src:
                    mcr[name] = ckpt[name]
            del ckpt

    def _prune_key(key):
        """Remove a key from all contigs to free RAM."""
        n = 0
        for r_name in list(multi_contig_results.keys()):
            if key in multi_contig_results.get(r_name, {}):
                del multi_contig_results[r_name][key]
                n += 1
        if n > 0:
            gc.collect()
            print(f"  [Prune] Dropped '{key}' from {n} contigs")

    vcf_path = os.environ.get(
        "HAPLOTYPES_VCF",
        "work/data/fish_vcf_restriped/AsAc.AulStuGenome.biallelic.bcf.gz"
    )

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
    # FOUNDER TEMPLATES: empirical sequence construction
    # =========================================================================
    TEMPLATE_STAGE = "founder_templates"
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
    checkpoint_store.bind_stage_identity(TEMPLATE_STAGE, {
        'discovery': discovery_identity_record, 'inputs': template_identity,
    })
    if template_directory and not stage_complete(TEMPLATE_STAGE):
        for region in regions_config:
            contig = region['contig']
            if contig_done(TEMPLATE_STAGE, contig):
                continue
            template_path = Path(template_directory) / f"{contig}.npz"
            with np.load(template_path, allow_pickle=False) as template:
                positions = template['positions']
                probabilities = template['allele_probabilities']
            if probabilities.ndim != 3 or probabilities.shape[1:] != (len(positions), 2):
                raise ValueError(f"{contig}: invalid founder-template axes")
            save_contig(TEMPLATE_STAGE, contig, {'naive_long_haps': [positions, list(probabilities)],
                'discovery_backend': discovery_blocks.DISCOVERY_BACKEND,
                'discovery_config': discovery_config_record})
        mark_stage_complete(TEMPLATE_STAGE)

    if stage_complete(TEMPLATE_STAGE):
        print(f"\n[RESUME] Skipping VCF loading + discovery (checkpoint found)")
        # naive_long_haps loaded on-demand via _ensure_key
    else:
        with discovery_blocks.BlockDiscoveryPool(n_processes) as block_pool:
            for region in regions_config:
                r_name = region['contig']
                if contig_done(TEMPLATE_STAGE, r_name):
                    print(f"  [RESUME] {r_name} already done")
                    continue

                print(f"\n" + "="*60)
                print(f"PROCESSING REGION: ({region['contig']} blocks {region['start']}-{region['end']})")
                print("=" * 60)

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
                    block_pool=block_pool,
                    discovery_config=discovery_config,
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
                save_contig(TEMPLATE_STAGE, r_name, {
                    'naive_long_haps': naive_long_haps,
                    'discovery_backend': discovery_blocks.DISCOVERY_BACKEND,
                    'discovery_config': discovery_config_record,
                })
                del genomic_data, block_results, naive_blocks, naive_long_haps
                gc.collect()

        print(f"\nAll regions processed in {time.time() - total_start:.2f}s")
        core_runtime.require_contig_checkpoints(
            checkpoint_store,
            TEMPLATE_STAGE,
            [region['contig'] for region in regions_config],
        )
        mark_stage_complete(TEMPLATE_STAGE)

    # =========================================================================
    # SIMULATED READS: pedigree, meiosis and observations
    # =========================================================================
    SIMULATION_STAGE = "simulated_reads"
    generation_sizes = tuple(json.loads(os.environ.get("HAPLOTYPES_GENERATIONS", "[20, 100, 200]")))
    from ..simulation.designs import SimulationDesign, ReadModel, observed_indices
    if not generation_sizes or any(size < 2 for size in generation_sizes):
        raise ValueError("each generated cohort must have at least two individuals")
    simulation_design = SimulationDesign(**json.loads(os.environ.get("HAPLOTYPES_SIMULATION_DESIGN", "{}")))
    read_model = ReadModel(**json.loads(os.environ.get("HAPLOTYPES_READ_MODEL", "{}")))
    STRESS_TEST_MUTATIONS = False
    mutate_rate = 1e-5 if STRESS_TEST_MUTATIONS else 1e-10
    current_simulation_contigs = [
        region['contig'] for region in regions_config
    ]
    simulation_run_spec = {
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
        'read_error_rate': read_model.error_rate,
        **({'observation_design': simulation_design.record()}
           if simulation_design.record() != SimulationDesign().record() else {}),
        **({'read_model': read_model.record()}
           if read_model != ReadModel() else {}),
        'snps_per_block': 200,
        'snp_shift': 200,
        'discovery_backend': discovery_blocks.DISCOVERY_BACKEND,
        'discovery_config': discovery_config_record,
        'truth_haplotype_completion': 'seeded_unbiased_tie_resolution_v1',
    }
    SIMULATION_REQUIRED_KEYS = frozenset({
        'truth_pedigree',
        'sample_names',
        'region_keys',
        'simulation_seed',
        'requested_simulation_seed',
        'run_spec',
        'simulation_state',
    })

    def require_completed_simulation_payload(payload, context):
        missing_keys = SIMULATION_REQUIRED_KEYS.difference(payload)
        if missing_keys:
            raise RuntimeError(
                f"{context} lacks required keys: {sorted(missing_keys)!r}"
            )
        if payload['simulation_state'] != 'complete':
            raise RuntimeError(
                f"{context} has simulation_state "
                f"{payload['simulation_state']!r}, not 'complete'"
            )

    def require_current_simulation_identity(payload, context):
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
        if payload['run_spec'] != simulation_run_spec:
            raise RuntimeError(
                f"{context} run specification does not match this run"
            )
        if list(payload['region_keys']) != current_simulation_contigs:
            raise RuntimeError(
                f"{context} contig order does not match this run"
            )

    # An allocation can end after the complete global payload is durable but
    # before its tiny marker is published. Validate it, then finish atomically.
    if (not stage_complete(SIMULATION_STAGE)
            and checkpoint_store.global_done(SIMULATION_STAGE)):
        durable_simulation_payload = load_global(SIMULATION_STAGE)
        if 'simulation_state' not in durable_simulation_payload:
            raise RuntimeError(
                f"{SIMULATION_STAGE} global payload lacks simulation_state"
            )
        durable_state = durable_simulation_payload['simulation_state']
        if durable_state == 'complete':
            require_completed_simulation_payload(
                durable_simulation_payload, f"Durable {SIMULATION_STAGE}"
            )
            require_current_simulation_identity(
                durable_simulation_payload, f"Durable {SIMULATION_STAGE}"
            )
            core_runtime.require_contig_checkpoints(
                checkpoint_store, SIMULATION_STAGE,
                durable_simulation_payload['region_keys'],
            )
            mark_stage_complete(SIMULATION_STAGE)
            print(f"  [RECOVER] Published completion marker for {SIMULATION_STAGE}")
        elif durable_state != 'in_progress':
            raise RuntimeError(
                f"{SIMULATION_STAGE} global payload has invalid simulation_state "
                f"{durable_state!r}"
            )
        del durable_simulation_payload

    if stage_complete(SIMULATION_STAGE):
        print(f"\n[RESUME] Skipping simulation (checkpoint found)")
        g = load_global(SIMULATION_STAGE)
        require_completed_simulation_payload(g, SIMULATION_STAGE)
        require_current_simulation_identity(g, SIMULATION_STAGE)
        core_runtime.require_contig_checkpoints(
            checkpoint_store, SIMULATION_STAGE, g['region_keys']
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
        # any per-contig assembly writes. This makes an entropy-seeded run
        # reproducible on restart and prevents old partial contigs from being
        # silently mixed with a newly generated pedigree.
        if checkpoint_store.global_done(SIMULATION_STAGE):
            simulation_provenance = load_global(SIMULATION_STAGE)
            partial_required_keys = {
                'simulation_seed', 'requested_simulation_seed',
                'region_keys', 'run_spec', 'simulation_state',
            }
            missing_keys = partial_required_keys.difference(simulation_provenance)
            if missing_keys:
                raise RuntimeError(
                    f"Partial {SIMULATION_STAGE} checkpoint lacks required keys: "
                    f"{sorted(missing_keys)!r}"
                )
            if simulation_provenance['simulation_state'] != 'in_progress':
                raise RuntimeError(
                    f"Partial {SIMULATION_STAGE} checkpoint has simulation_state "
                    f"{simulation_provenance['simulation_state']!r}, not "
                    "'in_progress'"
                )
            require_current_simulation_identity(
                simulation_provenance, f"Partial {SIMULATION_STAGE} checkpoint"
            )
            realized_simulation_seed = simulation_provenance["simulation_seed"]
            requested_simulation_seed = (
                simulation_provenance["requested_simulation_seed"]
            )
            del simulation_provenance
            print(
                f"  [RESUME] {SIMULATION_STAGE} realized seed "
                f"{realized_simulation_seed}"
            )
        else:
            partial_contigs = [
                r_name for r_name in region_keys
                if contig_done(SIMULATION_STAGE, r_name)
            ]
            if partial_contigs:
                raise RuntimeError(
                    f"Partial {SIMULATION_STAGE} contigs lack run provenance "
                    f"({partial_contigs}); use a fresh checkpoint directory"
                )
            realized_simulation_seed = (
                SIMULATION_SEED
                if SIMULATION_SEED is not None
                else int.from_bytes(os.urandom(8), "little")
            )
            save_global(SIMULATION_STAGE, {
                'simulation_seed': realized_simulation_seed,
                'requested_simulation_seed': SIMULATION_SEED,
                'region_keys': list(region_keys),
                'run_spec': simulation_run_spec,
                'simulation_state': 'in_progress',
            })
            if not checkpoint_store.global_done(SIMULATION_STAGE):
                raise OSError(
                    f"Failed to checkpoint early {SIMULATION_STAGE} provenance"
                )
            print(
                f"  {SIMULATION_STAGE} realized seed: {realized_simulation_seed}"
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
            recomb_rate=simulation_run_spec['recombination_rate_per_bp'],
            mutate_rate=simulation_run_spec['mutation_rate_per_bp'],
            output_plot=None,
            parallel=True,
            num_processes=n_processes,
            seed=realized_simulation_seed,
            genetic_maps=simulation_genetic_maps, contig_names=region_keys,
            return_crossover_events=True,
            design=simulation_design,
        )
        print(f"Pedigree simulation: {time.time()-t0:.1f}s")

        # Only observed individuals reach discovery, painting or pedigree inference.
        # Parent names are intentionally retained in truth even when absent from the VCF.
        observed = observed_indices(truth_pedigree, simulation_design, realized_simulation_seed)
        generated_pedigree = truth_pedigree
        truth_pedigree = truth_pedigree.iloc[observed].reset_index(drop=True)
        if len(observed) != len(generated_pedigree):
            all_offspring_lists = [[rows[i] for i in observed] for rows in all_offspring_lists]
            truth_paintings_lists = [[rows[i] for i in observed] for rows in truth_paintings_lists]
            observed_axis = {name: i for i, name in enumerate(truth_pedigree.Sample)}
            truth_crossovers = [[dict(event, child_index=observed_axis[event['child']])
                                 for event in events if event['child'] in observed_axis]
                                for events in truth_crossovers]
        print(f"Observed {len(observed)} / {len(generated_pedigree)} generated individuals")

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
            int(read_seed_rng.integers(0, 2 ** 63))
            for _ in region_keys
        ]
        print(
            "Post-processing one contig at a time for bounded peak memory"
            + (f" (seed={read_seed})" if read_seed is not None else "")
        )

        simulation_payload_keys = (
            'simulated_reads', 'simd_genomic_data', 'simd_probs',
            'simd_priors', 'truth_painting',
        )
        for contig_index, r_name in enumerate(region_keys):
            if contig_done(SIMULATION_STAGE, r_name):
                print(f"  [RESUME] {r_name} post-processing already done")
            else:
                result = (
                    simulation_pedigree._process_single_contig_postprocessing((
                        r_name,
                        all_offspring_lists[contig_index],
                        truth_paintings_lists[contig_index],
                        sites_list[contig_index],
                        simulation_run_spec['read_depth'],
                        simulation_run_spec['read_error_rate'],
                        simulation_run_spec['snps_per_block'],
                        simulation_run_spec['snp_shift'],
                        contig_read_seeds[contig_index],
                        {key: value for key, value in read_model.record().items() if key != 'error_rate'},
                    ))
                )
                payload = {
                    key: result[key] for key in simulation_payload_keys
                }
                payload['truth_founder_haplotypes'] = tuple(
                    np.asarray(haplotype, dtype=np.int8)
                    for parent in founders_list[contig_index]
                    for haplotype in parent
                )
                payload['truth_alleles'] = np.asarray(all_offspring_lists[contig_index], dtype=np.int8).transpose(0, 2, 1)
                payload['truth_crossovers'] = truth_crossovers[contig_index]
                save_contig(SIMULATION_STAGE, r_name, payload)
                if not contig_done(SIMULATION_STAGE, r_name):
                    raise OSError(
                        f"Failed to checkpoint {SIMULATION_STAGE}/{r_name}"
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
            checkpoint_store, SIMULATION_STAGE, region_keys
        )
        print(
            f"Post-processing ({len(region_keys)} contigs, bounded): "
            f"{time.time()-t0:.1f}s"
        )

        print("\nSimulation, Sequencing, and Chunking complete for all regions.")
        print(f"Total time: {time.time()-start:.1f}s")

        completed_simulation_payload = {
            'truth_pedigree': truth_pedigree,
            'generated_pedigree': generated_pedigree,
            'sample_names': sample_names,
            'region_keys': region_keys,
            'simulation_seed': realized_simulation_seed,
            'requested_simulation_seed': SIMULATION_SEED,
            'run_spec': simulation_run_spec,
            'simulation_state': 'complete',
        }
        save_global(SIMULATION_STAGE, completed_simulation_payload)
        if not checkpoint_store.global_done(SIMULATION_STAGE):
            raise OSError(f"Failed to checkpoint {SIMULATION_STAGE}/_global")
        persisted_simulation_payload = load_global(SIMULATION_STAGE)
        require_completed_simulation_payload(
            persisted_simulation_payload, f"Persisted {SIMULATION_STAGE}"
        )
        for key in (
            'sample_names', 'region_keys', 'simulation_seed',
            'requested_simulation_seed', 'run_spec',
        ):
            if persisted_simulation_payload[key] != completed_simulation_payload[key]:
                raise RuntimeError(
                    f"Persisted {SIMULATION_STAGE} changed {key!r} during checkpointing"
                )
        del persisted_simulation_payload, completed_simulation_payload
        mark_stage_complete(SIMULATION_STAGE)
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
        g = load_global('simulated_reads')
        region_keys = g['region_keys']
        sample_names = g['sample_names']
        truth_pedigree = g['truth_pedigree']
        del g
    all_region_keys = list(region_keys)
    if SIMULATION_SHARD_MODE:
        if not checkpoint_store.global_done("simulated_reads"):
            raise RuntimeError(
                "BHD_SIM_CONTIGS requires the global simulation manifest"
            )
        core_runtime.require_contig_checkpoints(
            checkpoint_store, "founder_templates", all_region_keys
        )
        core_runtime.require_contig_checkpoints(
            checkpoint_store, "simulated_reads", all_region_keys
        )
        region_keys = _select_simulation_contigs(
            all_region_keys, SIMULATION_CONTIG_SHARD
        )
        print(
            f"[SHARD] Processing {len(region_keys)} of "
            f"{len(all_region_keys)} contigs in simulation manifest order: "
            f"{', '.join(region_keys)}"
        )


    # =========================================================================
    # BLOCK DISCOVERY: simulated observations
    # =========================================================================
    DISCOVERY_STAGE = "block_discovery"
    checkpoint_store.bind_stage_identity(
        DISCOVERY_STAGE, discovery_identity_record
    )
    discovery_source_manifest = {
        'sample_ids': sample_names,
        'contigs': all_region_keys,
        'genotype_evidence_mode': (
            workflows_reconstruction.SUPPORTED_GENOTYPE_EVIDENCE_MODE
        ),
        'observed_call_mask_mode': (
            workflows_reconstruction.EXACT_OBSERVED_MASK_MODE
        ),
        'discovery_backend': discovery_blocks.DISCOVERY_BACKEND,
        'discovery_config': discovery_config_record,
    }
    checkpoint_store.bind_global_manifest(
        DISCOVERY_STAGE, discovery_source_manifest
    )

    if stage_complete(DISCOVERY_STAGE) and not SIMULATION_SHARD_MODE:
        print(f"\n[RESUME] Skipping block haplotype discovery (checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("Discovering Block Haplotypes from Simulated Reads")
        print(f"{'='*60}")

        start = time.time()

        with discovery_blocks.BlockDiscoveryPool(n_processes) as block_pool:
            for r_name in region_keys:
                if contig_done(DISCOVERY_STAGE, r_name):
                    print(f"  [RESUME] {r_name} already done")
                    continue
                print(f"\n  Processing {r_name}...")

                _ensure_key(r_name, 'simd_genomic_data', 'simulated_reads')
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
                    block_pool=block_pool,
                    discovery_config=discovery_config,
                )
                disc_time = time.time() - t_chr

                valid_blocks = [b for b in simd_block_results if len(b.positions) > 0]
                simd_block_results = core_haplotypes.BlockResults(valid_blocks)

                global_observed_mask = (
                    workflows_reconstruction.observed_call_mask_from_read_counts(
                        simulated_reads
                    )
                )
                save_contig(DISCOVERY_STAGE, r_name, {
                    'block_results': simd_block_results,
                    'global_sites': global_sites,
                    'global_observed_mask': global_observed_mask,
                    'observed_call_mask_mode': (
                        workflows_reconstruction.EXACT_OBSERVED_MASK_MODE
                    ),
                    'genotype_evidence_mode': (
                        workflows_reconstruction.SUPPORTED_GENOTYPE_EVIDENCE_MODE
                    ),
                    'discovery_backend': discovery_blocks.DISCOVERY_BACKEND,
                    'discovery_config': discovery_config_record,
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
            checkpoint_store, DISCOVERY_STAGE, region_keys
        )
    _finish_per_contig_stage(DISCOVERY_STAGE)

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
    print("ASSEMBLY: DISCOVERED BLOCKS -> COMPONENT painting")
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
        probabilities_stage=SIMULATION_STAGE,
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
        print("[STOP] Requested stop after typed component painting.")
        raise SystemExit(0)
    run_downstream(
        checkpoint_store, region_keys, sample_names,
        output_dir=output_dir, n_workers=n_processes,
        raw_gl_stage=SIMULATION_STAGE, raw_sites_stage=DISCOVERY_STAGE,
        raw_gl_key='simd_probs', parent_eligibility=None,
        all_contigs=all_region_keys, publish_global=not SIMULATION_SHARD_MODE,
        genetic_maps=inference_genetic_maps, recombination_rate=inference_recombination_rate,
    )



if __name__ == "__main__":
    run()
