"""General indexed-VCF/BCF workflow using the canonical scientific stages."""
from __future__ import annotations

from ..core import environment
environment.force_single_threaded_numeric_libraries()

from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from cyvcf2 import VCF

from ..core import haplotypes, numerics, runtime, variants
from ..discovery import blocks, search
from ..assembly.pipeline import AssemblyConfig
from ..core.genetic_map import load_genetic_maps_from_environment
from . import reconstruction
from .downstream import run_downstream
from .design import build_current_pedigree_config
from .eligibility import load_constraints


@runtime.logged_workflow("reconstruct")
def run():
    """All input samples are in scope unless explicit eligibility excludes them.

    Input must be indexed, sorted, biallelic SNP data with FORMAT/AD. Raw AD
    likelihoods, not hard GT calls, drive the canonical pipeline. Missing AD
    is unobserved. No population, generation, sex or sample-name inference is
    used to construct eligibility. Filter unrelated/outgroup samples from the
    VCF if they should also be excluded from founder discovery.
    """
    path = Path(os.environ["HAPLOTYPES_VCF"]).resolve()
    output = Path(os.environ["HAPLOTYPES_OUTPUT_DIR"])
    workers = int(os.environ["BHD_NUM_PROCESSES"])
    if not 1 <= workers <= runtime.available_cpu_count():
        raise ValueError("requested cores exceed CPU affinity")
    stop = os.environ.get("HAPLOTYPES_STOP_AFTER_STAGE")
    reader = VCF(str(path))
    try:
        names = tuple(reader.samples)
        contigs = tuple(json.loads(os.environ.get("HAPLOTYPES_CONTIGS", json.dumps(reader.seqnames))))
        if not contigs or len(contigs) != len(set(contigs)) or set(contigs) - set(reader.seqnames):
            raise ValueError("contigs must be unique names present in the input header")
        if len(names) < 3 or len(set(names)) != len(names):
            raise ValueError("input requires at least three uniquely named samples")
        if "##FORMAT=<ID=AD," not in reader.raw_header:
            raise ValueError("reconstruct requires FORMAT/AD allele depths; GT-only or PL-only input is not supported")
        # Exercise indexed queries before any expensive discovery.
        for contig in contigs:
            if next(iter(reader(contig)), None) is None:
                raise ValueError(f"{contig}: no indexed records; select populated contigs with --contigs")
    finally:
        reader.close()
    eligibility = load_constraints(os.environ.get("HAPLOTYPES_ELIGIBILITY"), names)
    if stop is None and len(contigs) < build_current_pedigree_config().parent_state_minimum_exposed_contigs:
        raise ValueError("genome-wide pedigree inference requires at least three contigs; "
                         "use --stop-after-stage painting for a smaller reconstruction")
    output.mkdir(parents=True, exist_ok=True)
    store = runtime.CheckpointStore(os.environ["HAPLOTYPES_CHECKPOINT_DIR"], nthreads=workers)
    maps = load_genetic_maps_from_environment()
    rate = maps.default_rate_cm_per_mb / 1e8
    genetic_maps = maps if maps.maps else None
    discovery_config = search.ReversibleCavitySearchConfig()
    identity = dict(backend=blocks.DISCOVERY_BACKEND, config=asdict(discovery_config))
    stat = path.stat()
    run_identity = dict(schema="general-vcf-ad-v1", vcf=str(path), size=stat.st_size,
                        mtime_ns=stat.st_mtime_ns, sample_ids=names, contigs=contigs,
                        discovery=identity, block_sites=200,
                        loader_sha256=hashlib.sha256(Path(variants.__file__).read_bytes()).hexdigest())
    source_stage = "block_discovery"
    store.bind_stage_identity(source_stage, run_identity)
    print(f"General reconstruction: {len(names)} samples, {len(contigs)} contigs; {workers}-core ceiling")
    print("Sequential scientific stages: pooled discovery/assembly, threaded painting/scoring, pooled resampling.")
    started = time.perf_counter()
    if not store.stage_complete(source_stage):
        with blocks.BlockDiscoveryPool(workers) as pool:
            for contig in contigs:
                if store.contig_done(source_stage, contig):
                    continue
                reads = variants.cleanup_block_reads_list(
                    str(path), contig, use_snp_count=True, snps_per_block=200, snp_shift=200,
                    num_processes=workers, strict_input=True)
                sites, counts = variants.concatenate_unique_block_reads(reads)
                if sites is None:
                    raise ValueError(f"{contig}: no usable SNPs")
                observed = reconstruction.observed_call_mask_from_read_counts(counts)
                _, probabilities = numerics.reads_to_probabilities(counts, use_hwe_prior=False)
                discovered = blocks.generate_all_block_haplotypes(
                    reads, num_processes=workers, block_pool=pool, discovery_config=discovery_config)
                discovered = haplotypes.BlockResults([b for b in discovered if len(b.positions)])
                store.save_contig(source_stage, contig, dict(
                    block_results=discovered, global_sites=sites, global_probs=probabilities,
                    global_observed_mask=observed, discovery_backend=identity["backend"],
                    discovery_config=identity["config"],
                    genotype_evidence_mode=reconstruction.SUPPORTED_GENOTYPE_EVIDENCE_MODE,
                    observed_call_mask_mode=reconstruction.EXACT_OBSERVED_MASK_MODE))
                del reads, counts, probabilities, observed, discovered
                gc.collect()
        store.save_global(source_stage, dict(
            sample_ids=names, contigs=contigs, discovery_backend=identity["backend"],
            discovery_config=identity["config"],
            genotype_evidence_mode=reconstruction.SUPPORTED_GENOTYPE_EVIDENCE_MODE,
            observed_call_mask_mode=reconstruction.EXACT_OBSERVED_MASK_MODE))
        runtime.require_contig_checkpoints(store, source_stage, contigs)
        store.mark_stage_complete(source_stage)
    if stop == "block_discovery":
        return
    reconstruction.run_reconstruction(
        store, contigs, names, discovery_identity=identity, source_stage=source_stage,
        config=reconstruction.ReconstructionConfig(
            release_config=AssemblyConfig(num_processes=workers, recombination_rate=rate),
            paint_cores=workers, paint_recombination_rate=rate), genetic_maps=genetic_maps)
    if stop == "painting":
        return
    run_downstream(
        store, contigs, names, output_dir=output, n_workers=workers,
        raw_gl_stage=source_stage, raw_sites_stage=source_stage,
        parent_eligibility=eligibility, genetic_maps=genetic_maps, recombination_rate=rate)
    print(f"Reconstruction through final phase and recombination completed in {time.perf_counter()-started:.1f}s")


if __name__ == "__main__":
    run()
