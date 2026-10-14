"""Compact downstream evidence alongside the richer discovery checkpoints.

These are lossless derived caches, not new scientific inputs. Source file
size/mtime records detect ordinary checkpoint replacement; the consuming
painting/pedigree code still verifies the exact evidence array identities,
by comparing the SHA-256 identities recorded with the arrays when the assembly
release computed them from exactly these arrays, else by hashing them.
"""
from __future__ import annotations

import os
from . import checkpoints


STAGE = "genotype_evidence"
SCHEMA = "compact-raw-genotype-evidence-v1"
ARRAYS = ("global_probs", "global_sites", "global_observed_mask")
# Optional entry: {array name: SHA-256 identity} (pedigree.components._array_digest).
DIGESTS = "input_array_sha256"


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


def load(store, contig, *, keys=None, **source):
    """Read current compact evidence, or let older inputs use their raw files.

    ``keys`` reads only those entries (plus the schema and source identity),
    decoding only their arrays (checkpoints.read_items).
    """
    if not store.contig_done(STAGE, contig):
        return None
    expected = source_identity(store, contig, **source)
    if keys is None:
        payload = store.load_contig(STAGE, contig)
    else:
        payload = checkpoints.read_items(
            checkpoints.contig_path(store.root, STAGE, contig),
            ("schema", "source_identity", *keys), nthreads=store.nthreads)
    if payload.get("schema") != SCHEMA or payload.get("source_identity") != expected:
        return None
    return payload


def recorded_digests(payload):
    """The array identities recorded with compact evidence, else None.

    None for raw source payloads and for caches written without them.
    """
    if payload is None or payload.get("schema") != SCHEMA:
        return None
    digests = payload.get(DIGESTS)
    return dict(digests) if isinstance(digests, dict) else None


def save(store, contig, probabilities, positions, observed, source_payload, digests=None,
         **source):
    """Persist the already validated arrays; never modify their source files.

    ``digests`` are the SHA-256 identities that the assembly release computed
    from exactly these three arrays, recorded for later consumers.
    """
    if load(store, contig, keys=(), **source) is not None:
        return
    payload = {
        "schema": SCHEMA, "source_identity": source_identity(store, contig, **source),
        "global_probs": probabilities, "global_sites": positions,
        "global_observed_mask": observed,
        "genotype_evidence_mode": source_payload["genotype_evidence_mode"],
        "observed_call_mask_mode": source_payload["observed_call_mask_mode"],
    }
    if digests is not None:
        payload[DIGESTS] = {name: str(digests[name]) for name in ARRAYS}
    store.save_contig(STAGE, contig, payload)
