"""Checkpointed known-pedigree simulation and end-to-end reconstruction driver."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from ..core.environment import configured_regions
import haplotype_reconstruction.assembly.pipeline as assembly_pipeline
import haplotype_reconstruction.core.config as core_config
import haplotype_reconstruction.core.environment as core_environment
import haplotype_reconstruction.core.genetic_map as core_genetic_map
import haplotype_reconstruction.core.haplotypes as core_haplotypes
import haplotype_reconstruction.core.parallel as core_parallel
import haplotype_reconstruction.core.read_calibration as read_calibration
import haplotype_reconstruction.core.run_record as core_run_record
import haplotype_reconstruction.core.runtime as core_runtime
import haplotype_reconstruction.core.sample_quality as core_sample_quality
import haplotype_reconstruction.core.site_quality as core_site_quality
import haplotype_reconstruction.core.variants as core_variants
import haplotype_reconstruction.discovery.blocks as discovery_blocks
import haplotype_reconstruction.discovery.checkpoints as discovery_checkpoints
import haplotype_reconstruction.discovery.search as discovery_search
import haplotype_reconstruction.simulation.pedigree as simulation_pedigree
import haplotype_reconstruction.simulation.templates as simulation_templates
import haplotype_reconstruction.simulation.truth as simulation_truth
import haplotype_reconstruction.workflows.contig_parallel as workflows_contig_parallel
import haplotype_reconstruction.workflows.reconstruction as workflows_reconstruction
from . import CICHLID_AUTOSOMES
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


# Run-level stage names (complete/<stage>, identity/<stage> in run.ckpt) and
# the contig records of the simulation stages.
TEMPLATE_STAGE = "simulation/templates"


SIMULATION_STAGE = "simulation/reads"


DISCOVERY_STAGE = "discovery"


TEMPLATES = "simulation/templates"


# --stop-after-stage values.
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


@core_runtime.logged_workflow("simulation")
def run():
    """Execute or resume the configured, chromosome-checkpointed workflow."""

    # FORCE NUMPY/BLAS TO USE 1 THREAD PER PROCESS
    # (Numba threading is now managed by core/parallel.py — do NOT set
    #  NUMBA_NUM_THREADS or NUMBA_THREADING_LAYER here)
    core_environment.force_single_threaded_numeric_libraries()

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
    # CHECKPOINTS
    # =============================================================================
    # One growing file per contig (contigs/<contig>.ckpt) holds every record of
    # that contig, from its simulated reads to its painting; run.ckpt holds the
    # genome-wide records (core/checkpoints, core/runtime). A stage loads only
    # the records it needs, one contig at a time.
    #
    # Resume with the same settings; use a new output root for a fresh run.
    # Preserve existing checkpoints and their original scientific identities.

    checkpoint_store = core_runtime.CheckpointStore(
        CHECKPOINT_DIR, nthreads=n_processes
    )
    run_file = checkpoint_store.run
    stage_complete = checkpoint_store.stage_complete
    mark_stage_complete = checkpoint_store.mark_stage_complete
    if SIMULATION_SHARD_MODE:
        missing_shared_stages = [
            stage for stage in (TEMPLATE_STAGE, SIMULATION_STAGE)
            if not stage_complete(stage)
        ]
        if missing_shared_stages:
            raise RuntimeError(
                "BHD_SIM_CONTIGS requires globally completed shared "
                "founder templates and simulated reads; missing completion records for: "
                f"{missing_shared_stages}"
            )
        print(
            "[SHARD] Verified globally completed shared founder templates and simulated reads; "
            "diagnostic validations and per-contig plots are disabled"
        )

    def _finish_per_contig_stage(stage, stop_name):
        """Verify a shard/stop boundary and publish only normal-run completion."""
        if not SIMULATION_SHARD_MODE:
            mark_stage_complete(stage)
        else:
            print(
                f"[SHARD] Verified {stage} records for "
                f"{', '.join(region_keys)}; genome-wide completion not recorded"
            )
        if SIMULATION_STOP_AFTER_STAGE == stop_name:
            print(f"[STOP] Completed and verified {stage}; exiting cleanly")
            raise SystemExit(0)

    def template_sites_and_haplotypes(r_name):
        return checkpoint_store.contig(r_name).get(TEMPLATES)['naive_long_haps']

    vcf_path = os.environ.get(
        "HAPLOTYPES_VCF",
        "work/data/fish_vcf_restriped/AsAc.AulStuGenome.biallelic.bcf.gz"
    )

    # Define the regions you want to use for inference.
    regions_config = configured_regions(
        [{"contig": contig, "start": 0, "end": 3000} for contig in CICHLID_AUTOSOMES],
        template_regions=True)

    block_size = 100000
    shift_size = 50000

    total_start = time.time()

    # =========================================================================
    # FOUNDER TEMPLATES: empirical sequence construction
    # =========================================================================
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
    checkpoint_store.bind_identity(run_file, TEMPLATE_STAGE, {
        'discovery': discovery_identity_record, 'inputs': template_identity,
    })
    if template_directory and not stage_complete(TEMPLATE_STAGE):
        for region in regions_config:
            contig = region['contig']
            template_file = checkpoint_store.contig(contig)
            if template_file.has(TEMPLATES):
                continue
            template_path = Path(template_directory) / f"{contig}.npz"
            with np.load(template_path, allow_pickle=False) as template:
                positions = template['positions']
                probabilities = template['allele_probabilities']
            if probabilities.ndim != 3 or probabilities.shape[1:] != (len(positions), 2):
                raise ValueError(f"{contig}: invalid founder-template axes")
            template_file.put(TEMPLATES, {'naive_long_haps': [positions, list(probabilities)],
                'discovery_backend': discovery_blocks.DISCOVERY_BACKEND,
                'discovery_config': discovery_config_record})
        mark_stage_complete(TEMPLATE_STAGE)

    if stage_complete(TEMPLATE_STAGE):
        print(f"\n[RESUME] Skipping VCF loading + discovery (checkpoint found)")
    else:
        with discovery_blocks.BlockDiscoveryPool(n_processes) as block_pool:
            for region in regions_config:
                r_name = region['contig']
                template_file = checkpoint_store.contig(r_name)
                if template_file.has(TEMPLATES):
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

                # 3. Construct generating templates, not inferred chromosomes.
                start = time.time()
                (naive_blocks, naive_long_haps) = simulation_templates.build_founder_templates(
                    block_results,
                    num_long_haps=6
                )
                print(f"  [Template construction] Chained {len(naive_long_haps[1])} haps in {time.time() - start:.2f}s")

                template_file.put(TEMPLATES, {
                    'naive_long_haps': naive_long_haps,
                    'discovery_backend': discovery_blocks.DISCOVERY_BACKEND,
                    'discovery_config': discovery_config_record,
                })
                del genomic_data, block_results, naive_blocks, naive_long_haps
                gc.collect()

        print(f"\nAll regions processed in {time.time() - total_start:.2f}s")
        core_runtime.require_records(
            checkpoint_store, TEMPLATES, [region['contig'] for region in regions_config])
        mark_stage_complete(TEMPLATE_STAGE)

    # =========================================================================
    # SIMULATED READS: pedigree, meiosis and observations
    # =========================================================================
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

    # An allocation can end after the complete simulation record is durable
    # but before the stage's completion record. Validate it, then complete.
    if (not stage_complete(SIMULATION_STAGE)
            and run_file.has(simulation_truth.SIMULATION)):
        durable_simulation_payload = run_file.get(simulation_truth.SIMULATION)
        if 'simulation_state' not in durable_simulation_payload:
            raise RuntimeError(
                f"{simulation_truth.SIMULATION} lacks simulation_state"
            )
        durable_state = durable_simulation_payload['simulation_state']
        if durable_state == 'complete':
            require_completed_simulation_payload(
                durable_simulation_payload, f"Durable {simulation_truth.SIMULATION}"
            )
            require_current_simulation_identity(
                durable_simulation_payload, f"Durable {simulation_truth.SIMULATION}"
            )
            core_runtime.require_records(
                checkpoint_store, simulation_pedigree.READS,
                durable_simulation_payload['region_keys'],
            )
            mark_stage_complete(SIMULATION_STAGE)
            print(f"  [RECOVER] Recorded completion of {SIMULATION_STAGE}")
        elif durable_state != 'in_progress':
            raise RuntimeError(
                f"{simulation_truth.SIMULATION} has invalid simulation_state "
                f"{durable_state!r}"
            )
        del durable_simulation_payload

    if stage_complete(SIMULATION_STAGE):
        print(f"\n[RESUME] Skipping simulation (checkpoint found)")
        g = run_file.get(simulation_truth.SIMULATION)
        require_completed_simulation_payload(g, SIMULATION_STAGE)
        require_current_simulation_identity(g, SIMULATION_STAGE)
        core_runtime.require_records(
            checkpoint_store, simulation_pedigree.READS, g['region_keys']
        )
        realized_simulation_seed = g["simulation_seed"]
        truth_pedigree = g['truth_pedigree']
        sample_names = g['sample_names']
        region_keys = g['region_keys']
        del g
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
            sites, haplotypes = template_sites_and_haplotypes(r_name)
            founder_templates.append(haplotypes)
            sites_list.append(sites)
            region_keys.append(r_name)

        # Bind the checkpoint root to one realized seed before simulation or
        # any per-contig assembly writes. This makes an entropy-seeded run
        # reproducible on restart and prevents old partial contigs from being
        # silently mixed with a newly generated pedigree.
        if run_file.has(simulation_truth.SIMULATION):
            simulation_provenance = run_file.get(simulation_truth.SIMULATION)
            partial_required_keys = {
                'simulation_seed', 'requested_simulation_seed',
                'region_keys', 'run_spec', 'simulation_state',
            }
            missing_keys = partial_required_keys.difference(simulation_provenance)
            if missing_keys:
                raise RuntimeError(
                    f"Partial {simulation_truth.SIMULATION} lacks required keys: "
                    f"{sorted(missing_keys)!r}"
                )
            if simulation_provenance['simulation_state'] != 'in_progress':
                raise RuntimeError(
                    f"Partial {simulation_truth.SIMULATION} has simulation_state "
                    f"{simulation_provenance['simulation_state']!r}, not "
                    "'in_progress'"
                )
            require_current_simulation_identity(
                simulation_provenance, f"Partial {simulation_truth.SIMULATION}"
            )
            realized_simulation_seed = simulation_provenance["simulation_seed"]
            del simulation_provenance
            print(
                f"  [RESUME] {SIMULATION_STAGE} realized seed "
                f"{realized_simulation_seed}"
            )
        else:
            partial_contigs = [
                r_name for r_name in region_keys
                if checkpoint_store.contig(r_name).has(simulation_pedigree.READS)
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
            run_file.put(simulation_truth.SIMULATION, {
                'simulation_seed': realized_simulation_seed,
                'requested_simulation_seed': SIMULATION_SEED,
                'region_keys': list(region_keys),
                'run_spec': simulation_run_spec,
                'simulation_state': 'in_progress',
            })
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

        # 4. Simulate reads and checkpoint each contig in parallel workers.
        # Every contig keeps its own read seed, drawn here in contig order,
        # so worker scheduling cannot change a simulated read.
        t0 = time.time()
        read_seed = realized_simulation_seed + 1_000_000
        read_seed_rng = np.random.default_rng(read_seed)
        contig_read_seeds = [
            int(read_seed_rng.integers(0, 2 ** 63))
            for _ in region_keys
        ]
        print(
            "Post-processing contigs in parallel, memory-bounded"
            + (f" (seed={read_seed})" if read_seed is not None else "")
        )
        read_model_settings = {
            key: value for key, value in read_model.record().items()
            if key != 'error_rate'
        }
        postprocessing_tasks = []
        for contig_index, r_name in enumerate(region_keys):
            if checkpoint_store.contig(r_name).has(simulation_pedigree.READS):
                print(f"  [RESUME] {r_name} post-processing already done")
                continue
            postprocessing_tasks.append(dict(
                root=checkpoint_store.root,
                contig=r_name,
                sample_names=sample_names,
                offspring=all_offspring_lists[contig_index],
                paintings=truth_paintings_lists[contig_index],
                sites=sites_list[contig_index],
                founders=founders_list[contig_index],
                crossovers=truth_crossovers[contig_index],
                read_depth=simulation_run_spec['read_depth'],
                error_rate=simulation_run_spec['read_error_rate'],
                seed=contig_read_seeds[contig_index],
                read_model=read_model_settings,
            ))
        for record in simulation_pedigree.checkpoint_simulated_contigs(
                postprocessing_tasks, cpu_budget=n_processes):
            core_run_record.record_product(
                SIMULATION_STAGE, record.pop('contig'), record)
        del postprocessing_tasks

        core_runtime.require_records(
            checkpoint_store, simulation_pedigree.READS, region_keys
        )
        print(
            f"Post-processing ({len(region_keys)} contigs, parallel): "
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
        run_file.put(simulation_truth.SIMULATION, completed_simulation_payload)
        persisted_simulation_payload = run_file.get(simulation_truth.SIMULATION)
        require_completed_simulation_payload(
            persisted_simulation_payload, f"Persisted {simulation_truth.SIMULATION}"
        )
        for key in (
            'sample_names', 'region_keys', 'simulation_seed',
            'requested_simulation_seed', 'run_spec',
        ):
            if persisted_simulation_payload[key] != completed_simulation_payload[key]:
                raise RuntimeError(
                    f"Persisted {simulation_truth.SIMULATION} changed {key!r} during checkpointing"
                )
        del persisted_simulation_payload, completed_simulation_payload
        mark_stage_complete(SIMULATION_STAGE)
        # Free heavy simulation data — all checkpointed, will reload on demand
        del all_offspring_lists, truth_paintings_lists, truth_crossovers
        del founders_list, sites_list
    gc.collect()

    # Ensure output_dir and globals exist for all subsequent stages
    output_dir = SIMULATION_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    all_region_keys = list(region_keys)
    if SIMULATION_SHARD_MODE:
        core_runtime.require_records(checkpoint_store, TEMPLATES, all_region_keys)
        core_runtime.require_records(
            checkpoint_store, simulation_pedigree.READS, all_region_keys)
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
    # The generating/template identities above deliberately remain independent
    # of the inference read model. Fit only the observed simulated allele depths.
    # Off by default here (core.config.SIMULATION_SITE_QUALITY); when switched
    # on, the kept simulated sites are re-blocked like the real loader's.
    site_quality_config = core_site_quality.configured(
        core_config.SIMULATION_SITE_QUALITY)
    # The per-sample QC (core.config.SIMULATION_SAMPLE_QUALITY, off) runs only
    # together with the site filter, whose quality-passing sites it uses.
    sample_quality_config = core_sample_quality.configured(
        core_config.SIMULATION_SAMPLE_QUALITY)
    discovery_identity_record = read_calibration.discovery_identity(
        discovery_blocks.DISCOVERY_BACKEND, discovery_config,
        core_site_quality.identity(site_quality_config),
        core_sample_quality.identity(sample_quality_config, site_quality_config))
    checkpoint_store.bind_identity(
        run_file, DISCOVERY_STAGE, discovery_identity_record
    )
    checkpoint_store.bind_manifest(DISCOVERY_STAGE, {
        'sample_ids': sample_names,
        'contigs': all_region_keys,
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
        """Load a contig's simulated reads and fit its read model."""
        print(f"\n  Preparing {r_name}...", flush=True)
        simulated_reads = checkpoint_store.contig(r_name).get(
            simulation_pedigree.READS)['simulated_reads']
        global_sites = np.asarray(template_sites_and_haplotypes(r_name)[0])
        simd_genomic_data = simulation_pedigree.simulated_genomic_data(
            global_sites, simulated_reads,
            simulation_run_spec['snps_per_block'],
            simulation_run_spec['snp_shift'])
        simd_genomic_data, site_quality = core_site_quality.filter_loaded_sites(
            simd_genomic_data, contig=r_name, config=site_quality_config,
            threads=n_processes, snps_per_block=simulation_run_spec['snps_per_block'],
            sample_config=sample_quality_config)
        founder_excluded = core_sample_quality.contig_flags(
            site_quality, contig=r_name, sample_ids=sample_names, output_dir=output_dir)
        if site_quality.get('applied'):
            global_sites, simulated_reads = (
                core_variants.concatenate_unique_block_reads(simd_genomic_data))
        global_observed_mask = (
            workflows_reconstruction.observed_call_mask_from_read_counts(
                simulated_reads
            )
        )
        global_probs, calibration = read_calibration.prepare_likelihoods(
            simulated_reads, global_sites, threads=n_processes)
        simd_genomic_data = core_sample_quality.exclude_from_discovery(
            simd_genomic_data, founder_excluded)
        return discovery_blocks.PreparedContig(
            genomic_data=simd_genomic_data,
            genotype_likelihoods_by_block=read_calibration.block_likelihoods(
                simd_genomic_data, global_sites, global_probs),
            fields={
                'global_sites': global_sites,
                'global_probs': global_probs,
                'read_calibration': calibration,
                'site_quality': site_quality,
                'global_observed_mask': global_observed_mask,
                **record_provenance,
                **({} if founder_excluded is None
                   else {'founder_evidence_excluded': founder_excluded}),
            },
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
    # the next; the stages below then record completion as in a normal run.
    serial_contigs, max_contigs = (
        workflows_contig_parallel.configured_serial_contigs())
    if serial_contigs and not workflows_contig_parallel.run_serially(
            checkpoint_store, region_keys, sample_names, discover=discover,
            discovery_identity=discovery_identity_record, config=assembly_config,
            output_dir=output_dir, max_contigs=max_contigs, all_contigs=all_region_keys,
            genetic_maps=inference_genetic_maps):
        return

    if stage_complete(DISCOVERY_STAGE) and not SIMULATION_SHARD_MODE:
        print(f"\n[RESUME] Skipping block haplotype discovery (checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("Discovering Block Haplotypes from Simulated Reads")
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

        print(f"\nBlock haplotype discovery complete in {time.time()-start:.1f}s")
        discovery_checkpoints.require_complete(checkpoint_store, region_keys)
    _finish_per_contig_stage(DISCOVERY_STAGE, "block_discovery")

    # =========================================================================
    # ASSEMBLY: COMPONENT ASSEMBLY AND PAINTING
    # =========================================================================
    print()
    print("=" * 60)
    print("ASSEMBLY: DISCOVERED BLOCKS -> COMPONENT painting")
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
        parent_eligibility=None,
        all_contigs=all_region_keys, publish_global=not SIMULATION_SHARD_MODE,
        genetic_maps=inference_genetic_maps, recombination_rate=inference_recombination_rate,
    )


if __name__ == "__main__":
    run()
