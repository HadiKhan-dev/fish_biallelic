"""Canonical cache provenance for safe V7 scoring reuse and resume."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import checkpoint_io

from .design import _strict_boolean
from .io import atomic_json


CACHE_MANIFEST_SCHEMA_VERSION = 1
CACHE_MANIFEST_NAME = "cache_manifest.json"


def _json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json(value):
    """Return deterministic compact JSON for hashing provenance structures."""
    return json.dumps(
        value,
        default=_json_scalar,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_json(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalized_seed_records(seed_audit):
    """Canonicalise the scientifically relevant seed mapping and provenance."""
    required = {
        "child_index", "child", "father_index", "father",
        "mother_index", "mother", "seed_basis",
        "seed_status", "report_parent_edges",
    }
    missing = required - set(seed_audit.columns)
    if missing:
        raise ValueError(f"seed audit lacks canonical columns: {sorted(missing)}")
    records = []
    ordered = seed_audit.sort_values("child_index", kind="stable")
    for row in ordered.itertuples(index=False):
        basis = " ".join(str(row.seed_basis).split())
        if not basis:
            raise ValueError("seed_basis must remain non-empty after normalization")
        records.append({
            "child_index": int(row.child_index),
            "child": str(row.child),
            "father_index": int(row.father_index),
            "father": str(row.father),
            "mother_index": int(row.mother_index),
            "mother": str(row.mother),
            "seed_basis": basis,
            "seed_status": str(row.seed_status),
            "report_parent_edges": _strict_boolean(
                row.report_parent_edges, "report_parent_edges"
            ),
        })
    if len({record["child_index"] for record in records}) != len(records):
        raise ValueError("normalized G0 seed children must be unique")
    return records


def normalized_metadata_records(metadata):
    required = (
        "sample_index", "Sample", "Alias", "SantosID", "Generation", "Sex"
    )
    missing = set(required) - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata lacks manifest columns: {sorted(missing)}")
    ordered = metadata.sort_values("sample_index", kind="stable")
    expected = np.arange(len(ordered), dtype=np.int64)
    if not np.array_equal(ordered["sample_index"].to_numpy(np.int64), expected):
        raise ValueError("sample_index must be contiguous BCF sample order")
    return [
        {
            "sample_index": int(row.sample_index),
            "sample": str(row.Sample),
            "alias": str(row.Alias),
            "santos_id": str(row.SantosID),
            "generation": str(row.Generation),
            "sex": str(row.Sex),
        }
        for row in ordered.itertuples(index=False)
    ]


def file_identity(path):
    """Fingerprint a source cheaply enough for routine cache validation."""
    path = Path(path).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def source_identities(bcf_path, metadata_path, checkpoint_dir, contigs):
    bcf = Path(bcf_path).resolve()
    indices = [
        candidate for candidate in (
            Path(str(bcf) + ".csi"),
            Path(str(bcf) + ".tbi"),
        )
        if candidate.exists()
    ]
    if not indices:
        raise FileNotFoundError(f"no BCF/VCF index beside {bcf}")
    checkpoints = []
    stage = "T09_viterbi_painting"
    for contig in contigs:
        path = checkpoint_io.contig_path(checkpoint_dir, stage, contig)
        identity = file_identity(path)
        identity["stage"] = stage
        identity["contig"] = str(contig)
        checkpoints.append(identity)
    return {
        "bcf": file_identity(bcf),
        "bcf_indices": [file_identity(path) for path in indices],
        "metadata": file_identity(metadata_path),
        "checkpoint_dir": str(Path(checkpoint_dir).resolve()),
        "checkpoint_files": checkpoints,
    }


def normalized_candidate_arrays(candidates):
    keys = (
        "g0_parents", "g0_pairs", "g0_pairs_local",
        "f1_parents", "f1_pairs", "f1_pairs_local",
        "f1_children", "f2_children",
    )
    missing = set(keys) - set(candidates)
    if missing:
        raise ValueError(f"candidate sets lack manifest arrays: {sorted(missing)}")
    return {
        key: np.asarray(candidates[key], dtype=np.int64).tolist()
        for key in keys
    }


def build_cache_manifest(
    *,
    scoring_model_revision,
    contigs,
    bcf_path,
    metadata_path,
    checkpoint_dir,
    metadata,
    candidates,
    selected_g0_pairs_local,
    seed_audit,
    scientific_parameters,
):
    """Build the complete identity of a set of V7 contig score caches."""
    metadata_records = normalized_metadata_records(metadata)
    candidate_arrays = normalized_candidate_arrays(candidates)
    seed_records = normalized_seed_records(seed_audit)
    selected_local = np.asarray(selected_g0_pairs_local, dtype=np.int64)
    if selected_local.shape != (len(seed_records), 2):
        raise ValueError("selected G0 local pairs do not match normalized seeds")
    return {
        "cache_manifest_schema_version": CACHE_MANIFEST_SCHEMA_VERSION,
        "scoring_model_revision": str(scoring_model_revision),
        "contigs": [str(contig) for contig in contigs],
        "sources": source_identities(
            bcf_path, metadata_path, checkpoint_dir, contigs
        ),
        "ordered_bcf_sample_ids": [
            record["sample"] for record in metadata_records
        ],
        "metadata_design_records": metadata_records,
        "metadata_design_sha256": sha256_json(metadata_records),
        "candidate_arrays": candidate_arrays,
        "candidate_arrays_sha256": sha256_json(candidate_arrays),
        "normalized_g0_seed_mapping": seed_records,
        "g0_seed_sha256": sha256_json(seed_records),
        "selected_g0_pairs_local": selected_local.tolist(),
        "selected_g0_pairs_local_sha256": sha256_json(selected_local.tolist()),
        "scientific_parameters": scientific_parameters,
        "scientific_parameters_sha256": sha256_json(scientific_parameters),
    }


def manifest_sha256(manifest):
    return sha256_json(manifest)


def _read_manifest(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def validate_cache_manifest(
    cache_dir,
    expected,
    *,
    allow_legacy_cache_without_manifest=False,
):
    """Require exact cache provenance, or a conspicuous legacy override."""
    path = Path(cache_dir) / CACHE_MANIFEST_NAME
    if not path.exists():
        if allow_legacy_cache_without_manifest:
            return "unsafe_legacy_cache_without_manifest"
        raise RuntimeError(
            f"cache provenance manifest is absent: {path}; refusing reuse. "
            "Only an explicit --allow-legacy-cache-without-manifest override "
            "may read exploratory legacy caches."
        )
    observed = _read_manifest(path)
    if observed != expected:
        differing = sorted(
            key for key in set(observed) | set(expected)
            if observed.get(key) != expected.get(key)
        )
        raise RuntimeError(
            "cache provenance does not match current BCF/sample order/design/"
            f"seeds/settings; differing manifest fields: {differing}"
        )
    return "validated"


def prepare_internal_cache(cache_dir, expected, resume):
    """Create or validate a package-owned cache before any contig is reused."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / CACHE_MANIFEST_NAME
    existing_scores = list(cache_dir.glob("v7_*.npz"))
    if path.exists():
        validate_cache_manifest(cache_dir, expected)
        if existing_scores and not resume:
            raise RuntimeError("contig cache exists; pass --resume to reuse it")
    elif existing_scores:
        raise RuntimeError(
            "package cache contains score files without provenance manifest"
        )
    else:
        atomic_json(path, expected)
    return cache_dir
