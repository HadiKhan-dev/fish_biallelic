"""Decode simulation records for results without importing stage orchestration.

Run-level manifest schema 2 binds the realised seed and ordered truth tables.
Contig reads and truth require independently supplied expected payload identity;
store.protocol.resume validates the immutable envelope before these decoders check
or reconstruct the schema-2 values. Publication, seed binding and resume scheduling
remain in stages.inputs.simulation (P3 ruling 8).
"""

from __future__ import annotations

from typing import Any

from ...core.errors import InputError
from ...simulation.truth import (
    MANIFEST,
    READS,
    TRUTH,
    ContigTruth,
    SimulationManifest,
    require_simulation_schema,
    truth_from_record,
)
from ...store.container import CheckpointFile
from ...store.protocol import read_items, resume
from ...store.store import CheckpointStore

__all__ = ["check_reads", "load_manifest", "load_reads", "load_truth"]


def load_manifest(store: CheckpointStore) -> SimulationManifest | None:
    """Return the bound run-level simulation manifest, checking its schema."""
    payload = store.control(MANIFEST)
    if payload is None:
        return None
    require_simulation_schema(payload)
    return SimulationManifest.from_record(payload)


def load_reads(file: CheckpointFile, identity: str) -> dict[str, Any]:
    """Read an identity-checked observation payload; refuse a missing contig."""
    record = resume(file, READS, identity)
    if record is None:
        raise InputError(f"{file.label}: no simulated reads")
    require_simulation_schema(record.payload)
    return record.payload


def check_reads(file: CheckpointFile, identity: str) -> None:
    """Refuse a missing contig, another identity or schema, as load_reads does, without decoding the counts (release
    4.10: the simulation's completion and resume checks; the evidence stage decodes and checks them in full)."""
    record = read_items(file, READS, identity, ("schema",))
    if record is None:
        raise InputError(f"{file.label}: no simulated reads")
    require_simulation_schema(record.payload)


def load_truth(file: CheckpointFile, identity: str) -> ContigTruth:
    """Read identity-checked truth for evaluation, never for inference."""
    record = resume(file, TRUTH, identity)
    if record is None:
        raise InputError(f"{file.label}: no simulated truth")
    return truth_from_record(record.payload)
