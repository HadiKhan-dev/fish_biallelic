"""Read content-bound observation truth without changing native release admission.

The sidecar stores plain truth-v2 records, a simulation manifest and independent
BCF, index, sample and site bindings. Inference never consumes these files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..contracts.payload import to_payload
from ..contracts.source import DROPPED_REASONS, ContigReads
from ..core.digest import array_digest, json_digest, jsonable
from ..core.errors import InputError
from ..stages.inputs.simulation_records import ContigTruth, SimulationManifest, truth_from_record
from ..store.protocol import ContigPayload, payload_identity

SIDECAR_SCHEMA = 1


def file_digest(path: Path) -> str:
    """SHA-256 of a file, with bounded read buffers."""
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_truth(path: Path, truth: ContigTruth) -> None:
    """Write a plain, pickle-free truth-v2 record to an NPZ archive."""
    arrays: dict[str, np.ndarray] = {}

    def separate(value):
        if isinstance(value, np.ndarray):
            key = f"array_{len(arrays)}"
            arrays[key] = value
            return {"array": key}
        if isinstance(value, dict):
            return {key: separate(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [separate(item) for item in value]
        return value

    metadata = separate(to_payload(truth))
    with Path(path).open("wb") as handle:
        np.savez_compressed(handle, metadata=np.asarray(json.dumps(metadata)), allow_pickle=False, **arrays)


def read_truth(path: Path) -> ContigTruth:
    """Decode the shared truth-v2 schema; old schemas retain their refusal."""
    with np.load(path, allow_pickle=False) as data:

        def restore(value):
            if isinstance(value, dict):
                if set(value) == {"array"}:
                    return data[value["array"]]
                return {key: restore(item) for key, item in value.items()}
            if isinstance(value, list):
                return [restore(item) for item in value]
            return value

        payload = restore(json.loads(str(data["metadata"])))
    return truth_from_record(payload)


def observation_binding(reads: ContigReads, length_bp: int) -> dict[str, str]:
    """Consumed-payload identities for both supported canonical sample orders."""
    names = tuple(reads.sample_ids)
    result = {}
    for reverse in (False, True):
        ordered = tuple(sorted(names, reverse=reverse))
        order = np.asarray([names.index(name) for name in ordered], np.int64)
        sites = len(reads.positions)
        payload = ContigPayload(
            reads.contig,
            ordered,
            reads.positions,
            ("A",) * sites,
            ("C",) * sites,
            reads.counts[order],
            np.ones(sites, bool),
            None,
            int(length_bp),
            {reason: 0 for reason in DROPPED_REASONS},
        )
        result[json_digest(ordered)] = payload_identity(payload)
    return result


def _project_intervals(record, rows, columns, *, paired):
    """Compress selected labels back to shared marker-index interval records."""
    offsets, starts, ends, first_labels, second_labels = [0], [], [], [], []
    for row in rows:
        first, stop = record.offsets[row : row + 2]
        if len(columns):
            segment = np.searchsorted(record.start[first:stop], columns, side="right") - 1 + first
            if paired:
                first_values, second_values = record.hap1[segment], record.hap2[segment]
                changes = (first_values[1:] != first_values[:-1]) | (second_values[1:] != second_values[:-1])
            else:
                first_values = record.source[segment]
                changes = first_values[1:] != first_values[:-1]
            boundaries = np.r_[0, np.flatnonzero(changes) + 1]
            starts.extend(boundaries.tolist())
            ends.extend(np.r_[boundaries[1:], len(columns)].tolist())
            first_labels.extend(first_values[boundaries].tolist())
            if paired:
                second_labels.extend(second_values[boundaries].tolist())
        offsets.append(len(starts))
    result = dict(offsets=np.asarray(offsets, np.int64), start=np.asarray(starts, np.int64))
    if paired:
        result.update(
            end=np.asarray(ends, np.int64),
            hap1=np.asarray(first_labels, np.int32),
            hap2=np.asarray(second_labels, np.int32),
        )
    else:
        result["source"] = np.asarray(first_labels, np.int32)
    return result


def _projection_indices(truth, positions, sample_ids):
    """Validate identifier subsets without gathering dense truth arrays."""
    names = tuple(map(str, sample_ids))
    sites = np.asarray(positions)
    if sites.ndim != 1 or sites.dtype.kind not in "iu" or np.any(sites[1:] <= sites[:-1]):
        raise InputError("projected truth sites must be strictly increasing integer positions")
    if len(set(names)) != len(names) or len(set(truth.sample_ids)) != len(truth.sample_ids):
        raise InputError("projected truth requires unique sample identifiers")
    by_name = {name: i for i, name in enumerate(truth.sample_ids)}
    missing = set(names) - set(by_name)
    if missing:
        raise InputError(f"samples absent from truth: {', '.join(sorted(missing))}")
    columns = np.searchsorted(truth.positions, sites)
    if np.any(columns >= len(truth.positions)) or not np.array_equal(truth.positions[columns], sites):
        raise InputError("run sites are absent from truth")
    rows = np.asarray([by_name[name] for name in names], np.int64)
    return names, sites, rows, columns


def project_truth(truth: ContigTruth, positions, sample_ids) -> ContigTruth:
    """Project shared truth by identifiers, including its marker-index records.

    Founder evaluation keeps the original record and its immutable truth grid.
    Phase/genotype evaluation receives this separate record on the product axes.
    """
    names, sites, rows, columns = _projection_indices(truth, positions, sample_ids)
    payload = to_payload(truth)
    payload.update(
        sample_ids=names,
        positions=sites.astype(np.int64),
        alleles=truth.alleles[rows][:, columns],
        source_haplotypes=truth.source_haplotypes[:, columns],
        reference_derived=truth.reference_derived[columns],
        spurious=truth.spurious[columns],
        founder_mosaics=_project_intervals(
            truth.founder_mosaics, range(len(truth.founder_mosaics.offsets) - 1), columns, paired=False
        ),
    )
    for key in ("painting", "source_painting"):
        payload[key] = _project_intervals(getattr(truth, key), rows, columns, paired=True)
    offsets, markers = [0], []
    for row in rows:
        for slot in (0, 1):
            first, stop = truth.flips.offsets[2 * row + slot : 2 * row + slot + 2]
            original = truth.flips.markers[first:stop]
            projected = np.searchsorted(columns, original)
            valid = projected < len(columns)
            projected, original = projected[valid], original[valid]
            markers.extend(projected[columns[projected] == original].tolist())
            offsets.append(len(markers))
    payload["flips"] = dict(offsets=np.asarray(offsets, np.int64), markers=np.asarray(markers, np.int64))
    crossovers = truth.crossovers
    chosen = np.asarray([i for i, child in enumerate(crossovers.child) if child in names], np.int64)
    offsets, events = [0], []
    for i in chosen:
        first, stop = crossovers.offsets[i : i + 2]
        events.extend(crossovers.positions[first:stop].tolist())
        offsets.append(len(events))
    payload["crossovers"] = dict(
        child_index=np.asarray([names.index(crossovers.child[i]) for i in chosen], np.int64),
        child=tuple(crossovers.child[i] for i in chosen),
        parent=tuple(crossovers.parent[i] for i in chosen),
        parent_slot=crossovers.parent_slot[chosen],
        cohort_index=crossovers.cohort_index[chosen],
        offsets=np.asarray(offsets, np.int64),
        positions=np.asarray(events, np.float64),
    )
    for key in ("missed", "block"):
        value = getattr(truth, key)
        payload[key] = None if value is None else value[rows][:, columns]
    if truth.systematic_site is not None:
        payload["systematic_site"] = truth.systematic_site[columns]
    return truth_from_record(payload)


class TruthSidecar:
    """A checked frozen package; run bindings are checked separately from file digests."""

    def __init__(self, directory: str | Path, *, bcf: str | Path | None = None):
        self.root = Path(directory).expanduser().resolve()
        self.document = json.loads((self.root / "manifest.json").read_text())
        if self.document.get("schema") != SIDECAR_SCHEMA:
            raise InputError("unsupported observation truth manifest schema")
        self.manifest = SimulationManifest.from_record(self.document["simulation"])
        self.sample_ids = tuple(self.document["sample_ids"])
        self.contigs = tuple(entry["contig"] for entry in self.document["contigs"])
        if (
            self.sample_ids != self.manifest.sample_ids
            or self.contigs != self.manifest.contigs
            or len(set(self.contigs)) != len(self.contigs)
            or len(self.sample_ids) < 3
            or len(set(self.sample_ids)) != len(self.sample_ids)
        ):
            raise InputError("truth manifest sample or contig axes differ")
        self.bcf = (self.root / self.document["bcf"]["path"]).resolve() if bcf is None else Path(bcf).resolve()
        if file_digest(self.bcf) != self.document["bcf"]["sha256"]:
            raise InputError("observation BCF digest differs from truth binding")
        if file_digest(Path(str(self.bcf) + ".csi")) != self.document["bcf"]["index_sha256"]:
            raise InputError("observation BCF index digest differs from truth binding")
        for entry in self.document.get("settings_files", ()):
            path = self.root / entry["path"]
            if file_digest(path) != entry["sha256"]:
                raise InputError("observation settings file digest differs from truth binding")
        self._entries = {entry["contig"]: entry for entry in self.document["contigs"]}

    def truth(self, contig: str) -> ContigTruth:
        """Verify bytes and axes before returning a frozen shared truth record."""
        if contig not in self._entries:
            raise InputError(f"{contig}: absent from observation truth")
        entry = self._entries[contig]
        path = self.root / entry["file"]
        if file_digest(path) != entry["sha256"]:
            raise InputError(f"{contig}: truth record digest differs")
        truth = read_truth(path)
        if (
            tuple(truth.sample_ids) != self.sample_ids
            or array_digest(truth.positions) != entry["positions"]
            or len(truth.positions) != entry["sites"]
        ):
            raise InputError(f"{contig}: truth sample or site axis differs from manifest")
        return truth

    def validate_run(self, reader) -> None:
        """Bind checked run inputs to the package before any score is computed."""
        if reader.manifest.get("source", {}).get("kind") != "vcf":
            raise InputError("truth sidecars require a VCF-route run")
        if tuple(reader.manifest.get("source", {}).get("input_sample_ids", ())) != self.sample_ids:
            raise InputError("run input sample order differs from observation truth binding")
        if set(reader.contigs) - set(self.contigs):
            raise InputError("run contigs are absent from observation truth")
        for contig in reader.contigs:
            axes = reader.observation_axes(contig)
            entry = self._entries[contig]
            expected = entry["payloads"].get(json_digest(axes.sample_ids))
            if axes.payload_identity != expected:
                raise InputError(f"{contig}: run consumed a different observation payload")
            _projection_indices(self.truth(contig), axes.positions, axes.sample_ids)

    def description(self) -> dict:
        """Portable provenance for an evaluation report."""
        return jsonable(
            dict(schema=SIDECAR_SCHEMA, generator=self.document["generator"], bcf_sha256=self.document["bcf"]["sha256"])
        )
