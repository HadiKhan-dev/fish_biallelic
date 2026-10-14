"""Read checkpointed evidence and founder products without rerunning inference."""
from __future__ import annotations

import numpy as np

from ..discovery import checkpoints as discovery_checkpoints


FOUNDER_STAGES = ("block_discovery", "feedback_initial", "feedback_l1", "feedback_l2",
                  "assembly_l1", "assembly_l2", "assembly_l3", "assembly_l4",
                  "founder_refinement", "painting", "allele_refit")

# The chromosome's raw observations: the only stored copy of its genotype likelihoods.
EVIDENCE = discovery_checkpoints.PREPARED
# SHA-256 identities of the evidence arrays as the assembly release hashed them.
EVIDENCE_DIGESTS = discovery_checkpoints.EVIDENCE_SHA256
EVIDENCE_ARRAYS = ("global_probs", "global_sites", "global_observed_mask")
# Run-file record naming the published genome-wide pedigree decision.
PEDIGREE = "pedigree"


def load_evidence(file, arrays=EVIDENCE_ARRAYS, *, nthreads=None):
    """Selected raw evidence arrays of one chromosome, decoding only those arrays.

    Returns them in the order requested. The record's genotype-likelihood and
    observation-mask provenance must be the supported ones.
    """
    payload = file.get(EVIDENCE, items=(*arrays, "genotype_evidence_mode", "observed_call_mask_mode"),
                       nthreads=nthreads)
    if payload.get("genotype_evidence_mode") != workflows_reconstruction.SUPPORTED_GENOTYPE_EVIDENCE_MODE:
        raise RuntimeError(f"{file.label}: evidence lacks normalized raw-likelihood provenance")
    if payload.get("observed_call_mask_mode") != workflows_reconstruction.EXACT_OBSERVED_MASK_MODE:
        raise RuntimeError(f"{file.label}: evidence lacks exact observation-mask provenance")
    missing = [name for name in arrays if name not in payload]
    if missing:
        raise KeyError(f"{file.label}: {EVIDENCE} lacks {missing}")
    return tuple(payload[name] for name in arrays)


def evidence_checks(file):
    """Inputs of the painting release-identity check (pedigree.components).

    Returns a loader of the sample-quality flags (called only when the raw
    arrays differ from the release input) and the recorded release digests of
    exactly these arrays, or None when the release hashed QC-filtered evidence.
    """
    def flags():
        value = file.get(EVIDENCE, items=("founder_evidence_excluded",)).get("founder_evidence_excluded")
        return None if value is None else np.array(value, dtype=np.bool_)

    digests = file.get(EVIDENCE_DIGESTS) if file.has(EVIDENCE_DIGESTS) else None
    return flags, None if digests is None else {name: str(digests[name]) for name in EVIDENCE_ARRAYS}


def record_identities(store, keys, contigs):
    """Identities of the chromosome records a product is computed from."""
    for key in keys:
        core_runtime.require_records(store, key, contigs)
    return [dict(contig=contig, key=key, record=store.contig(contig).info(key).identity)
            for contig in contigs for key in keys]


def published_pedigree(store):
    """The published genome-wide pedigree decision, or None before one exists."""
    if not store.run.has(PEDIGREE):
        return None
    return store.run.get(store.run.get(PEDIGREE)["decision"], nthreads=store.nthreads)


def panel_keys(file, stage):
    """Record keys holding a founder stage's panels, or None while the product is absent."""
    if stage not in FOUNDER_STAGES:
        raise ValueError(f"unknown founder stage: {stage}")
    if stage in ("painting", "allele_refit"):
        return (stage,) if file.has(stage) else None
    if stage == "block_discovery":
        return discovery_checkpoints.block_keys(file)
    return assembly_checkpoints.stage_keys(file, stage)


def panel_bytes(file, stage):
    """Decoded bytes of a founder stage's records, from their metadata alone."""
    return sum(file.info(key).decoded_bytes for key in panel_keys(file, stage))


def load_panels(file, stage="painting"):
    """One chromosome's founder panels of a stage, as the stage released them.

    Discovery blocks are returned without their genotype-likelihood slices.
    """
    if panel_keys(file, stage) is None:
        raise FileNotFoundError(f"{file.label}: no completed {stage} product")
    if stage == "painting":
        return core_runtime.validate_phase_component_manifest(file.get(stage).component_manifest)
    if stage == "allele_refit":
        return tuple(file.get(stage).components)
    if stage == "block_discovery":
        return tuple(discovery_checkpoints.load_blocks(file, probs=False))
    return tuple(assembly_checkpoints.stage_blocks(file, stage))


def panel_alleles(block):
    """Use published hard calls only; never turn probabilistic ties into alleles."""
    if isinstance(block, painting_allele_refit.RefitComponentPanel):
        keys = tuple(block.founder_keys)
        calls = np.asarray(block.released_alleles, dtype=np.int8)
    else:
        keys = tuple(sorted(block.haplotypes))
        calls = np.asarray(block.discrete_haps, dtype=np.int8)
    positions = np.asarray(block.positions, dtype=np.int64)
    if calls.shape != (len(keys), len(positions)) or np.any(~np.isin(calls, (-1, 0, 1))):
        raise ValueError("founder call/position/key axes are inconsistent")
    return positions, keys, calls

import haplotype_reconstruction.assembly.checkpoints as assembly_checkpoints
import haplotype_reconstruction.core.runtime as core_runtime
import haplotype_reconstruction.painting.allele_refit as painting_allele_refit
import haplotype_reconstruction.workflows.reconstruction as workflows_reconstruction
