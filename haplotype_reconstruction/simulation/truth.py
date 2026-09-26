"""Compact simulation truth for downstream-only evaluation.

Inference never consumes this product. New simulations write it while their
truth arrays are already resident; old runs can derive it once into the
evaluation output's cache without changing the original checkpoints.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..core import checkpoints
from ..core.runtime import CheckpointStore


_SCHEMA = "evaluation-truth-v1"
_FIELDS = ("truth_founder_haplotypes", "truth_painting", "truth_alleles", "truth_crossovers")


def truth_identity(store, contig, sample_ids):
    """Bind the derived product to ordinary atomic input replacement and axes."""
    inputs = []
    for stage in ("simulated_reads", "founder_templates"):
        path = Path(checkpoints.contig_path(store.root, stage, contig)).resolve()
        stat = path.stat()
        inputs.append(dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns))
    return dict(schema=_SCHEMA, contig=contig, sample_ids=list(sample_ids), inputs=inputs)


def _truth_stage(identity):
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return "evaluation_truth/" + hashlib.sha256(encoded).hexdigest()


def truth_location(store, contig, sample_ids, cache_root=None):
    """Find a current compact product using only source/file metadata."""
    identity = truth_identity(store, contig, sample_ids)
    stage = _truth_stage(identity)
    roots = (store.root,) if cache_root is None else (store.root, cache_root)
    for root in roots:
        path = Path(checkpoints.contig_path(root, stage, contig))
        if path.is_file():
            return path, identity
    return None, identity


def save_truth(store, contig, sample_ids, positions, source, *, destination=None):
    """Save a compact view after its simulated-read input is durable."""
    identity = truth_identity(store, contig, sample_ids)
    payload = dict(identity=identity, positions=np.asarray(positions),
                   source={key: source[key] for key in _FIELDS if key in source})
    target = store if destination is None else destination
    target.save_contig(_truth_stage(identity), contig, payload)
    return payload


def load_truth(store, contig, sample_ids, *, cache_root):
    """Load compact truth or derive it once, leaving input checkpoints intact."""
    path, identity = truth_location(store, contig, sample_ids, cache_root)
    if path is not None:
        payload = checkpoints.read(str(path), nthreads=store.nthreads)
        if payload["identity"] != identity:
            raise ValueError(f"{contig}: evaluation truth source or sample axes differ")
        return payload["source"], np.asarray(payload["positions"])

    source = store.load_contig("simulated_reads", contig)
    # Release unused read/likelihood arrays before another substantial load.
    source = {key: source[key] for key in _FIELDS if key in source}
    positions = np.asarray(store.load_contig("founder_templates", contig)["naive_long_haps"][0])
    destination = CheckpointStore(cache_root, nthreads=store.nthreads)
    save_truth(store, contig, sample_ids, positions, source, destination=destination)
    return source, positions
