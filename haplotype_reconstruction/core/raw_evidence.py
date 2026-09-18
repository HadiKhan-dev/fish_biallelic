"""Compact downstream evidence alongside the richer discovery checkpoints.

These are lossless derived caches, not new scientific inputs. Source file
size/mtime records detect ordinary checkpoint replacement; the consuming
painting/pedigree code still verifies the exact evidence array identities.
"""
from __future__ import annotations

import os
from.import checkpoints


STAGE = "00_genotype_evidence"
SCHEMA = "compact-raw-genotype-evidence-v1"


def source_identity(store, contig, *, raw_gl_stage, raw_sites_stage,
                    raw_gl_key, raw_sites_key="global_sites",
                    raw_observed_mask_key="global_observed_mask"):
    sources = {}
    for stage in dict.fromkeys((raw_gl_stage, raw_sites_stage)):
        stat = os.stat(checkpoints.contig_path(store.root, stage, contig))
        sources[stage] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return {"contig": str(contig), "sources": sources,
            "gl": (raw_gl_stage, raw_gl_key), "sites": (raw_sites_stage, raw_sites_key),
            "observed": (raw_sites_stage, raw_observed_mask_key)}


def load(store, contig, **source):
    """Read current compact evidence, or let older inputs use their raw files."""
    if not store.contig_done(STAGE, contig):
        return None
    expected = source_identity(store, contig, **source)
    payload = store.load_contig(STAGE, contig)
    if payload.get("schema") != SCHEMA or payload.get("source_identity") != expected:
        return None
    return payload


def save(store, contig, probabilities, positions, observed, source_payload, **source):
    """Persist the already validated arrays; never modify their source files."""
    if load(store, contig, **source) is not None:
        return
    store.save_contig(STAGE, contig, {
        "schema": SCHEMA, "source_identity": source_identity(store, contig, **source),
        "global_probs": probabilities, "global_sites": positions,
        "global_observed_mask": observed,
        "genotype_evidence_mode": source_payload["genotype_evidence_mode"],
        "observed_call_mask_mode": source_payload["observed_call_mask_mode"],
    })
