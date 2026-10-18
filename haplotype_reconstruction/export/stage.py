"""The export stage: the pipeline's final products under OUTPUT/export, after recombination.

Per contig (``[export]`` settings, ExportConfig): ``<contig>.founders.vcf.gz`` (founder VCF, export.founders) and
``<contig>.samples.vcf.gz`` (final sample phase, export.samples), each with its tabix index, full length by default;
OUTPUT/export/manifest.json describes every file. Each contig's files are recorded complete by its ``export``
record (records.py), so a rerun skips exported contigs; changed settings or export code re-export them. Nothing
here changes another stage's records or results.
"""
from __future__ import annotations

from pathlib import Path

from . import records
from .config import ExportConfig
from .run import contig_bytes, run_tasks, write_manifest
from .sites import evidence_sites

STAGE = records.STAGE


def run(store, contigs, config: ExportConfig, output, budget, *, synthetic: bool, reference: str | None) -> dict:
    """Export every contig not yet exported under ``config`` to OUTPUT/export; returns the manifest.

    ``synthetic``: a simulation (alleles A/C); ``reference``: the input VCF (REF/ALT, AD) otherwise.
    """
    destination = Path(output) / STAGE
    destination.mkdir(parents=True, exist_ok=True)
    entries, tasks, estimates = {}, [], []
    for contig in contigs:
        file = store.contig(contig)
        identity = records.identity(file, config)
        recorded = records.load(file)
        if (recorded is not None and recorded["identity"] == identity
                and all((destination / name).is_file() for name in recorded["entry"]["files"])):
            entries[contig] = recorded["entry"]
            continue
        tasks.append(dict(checkpoints=str(store.root), destination=str(destination), contig=contig,
                          products=tuple(config.products), file_format=config.file_format, reference=reference,
                          synthetic=synthetic, full_length=config.full_length, record=identity,
                          previous_files=() if recorded is None else tuple(recorded["entry"]["files"])))
        estimates.append(contig_bytes(file, config.products, full_length=config.full_length, synthetic=synthetic,
                                      evidence=evidence_sites(file)))
    if not tasks:
        print(f"[RESUME] export: every contig is exported ({destination})", flush=True)
    for task, entry in zip(tasks, run_tasks(tasks, estimates, max(1, int(budget.threads())))):
        entries[task["contig"]] = entry
    manifest = write_manifest(destination, store.root, synthetic=synthetic, reference=reference,
                              contigs=[entries[contig] for contig in contigs], settings=config)
    store.mark_stage_complete(STAGE)
    print(f"[export] {len(contigs)} contigs: founder and sample VCFs in {destination}", flush=True)
    return manifest
