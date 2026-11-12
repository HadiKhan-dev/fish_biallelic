"""Publish one immutable observation package, with truth outside the BCF header."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts.roles import record_field
from ..contracts.source import ContigReads
from ..core.digest import array_digest, jsonable
from ..core.errors import InputError
from ..results.truth import (
    SIDECAR_SCHEMA,
    ContigTruth,
    file_digest,
    observation_binding,
    write_truth,
)
from .export.vcf import FILEFORMAT, RecordWriter, build_index, contig_line


@dataclass(frozen=True)
class ObservationContig:
    """One complete observation axis and its corresponding full truth."""

    reads: ContigReads = record_field("obs", dtype="ContigReads")
    truth: ContigTruth = record_field("obs", dtype="ContigTruth")
    length_bp: int = record_field("obs", units="bp", dtype="int64")


def _check_names(names, *, samples=False) -> None:
    if samples and len(names) < 3:
        raise InputError("observation package requires at least three uniquely named observed samples")
    bad = "\t\r\n " if samples else "\t\r\n ,=<>:"
    if not names or len(set(names)) != len(names) or any(not name or any(c in name for c in bad) for name in names):
        raise InputError("observation package needs unique VCF-compatible identifiers")


def write_package(
    output: str | Path,
    manifest: Any,
    load_contig: Callable[[str], ObservationContig],
    inference_text: str,
    *,
    generator: Mapping[str, Any],
    settings_files: Mapping[str, Path] | None = None,
) -> Path:
    """Write BCF/index, inference settings and truth, then publish the directory.

    ``load_contig`` is called once per contig, so counts are retained for at most
    one contig. Refuse malformed/empty axes without drawing replacements. A failed
    write leaves the destination unmodified; an existing nonempty package is refused.
    """
    output = Path(output).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise InputError("observation output directory must be new or empty")
    names, contigs = tuple(manifest.sample_ids), tuple(manifest.contigs)
    _check_names(names, samples=True)
    _check_names(contigs)
    if manifest.state != "complete":
        raise InputError("observation package requires complete simulated truth")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".observations-", dir=output.parent) as temporary:
        stage = Path(temporary) / "package"
        truth_dir = stage / "truth"
        truth_dir.mkdir(parents=True)
        # Header lengths must be known before the first BCF record. The caller
        # supplies them without retaining every contig's counts at once.
        entries = []
        # Stage a small header-free text spool while discovering contig lengths.
        # This bounds memory independently of the number of selected contigs.
        spool = stage / ".records.vcf"
        with spool.open("w") as records:
            for index, contig in enumerate(contigs):
                item = load_contig(contig)
                reads, truth = item.reads, item.truth
                positions, counts = np.asarray(reads.positions), np.asarray(reads.counts)
                if (
                    reads.contig != contig
                    or reads.sample_ids != names
                    or tuple(truth.sample_ids) != names
                    or not np.array_equal(positions, truth.positions)
                ):
                    raise InputError(f"{contig}: observation and truth axes differ")
                if (
                    positions.ndim != 1
                    or positions.dtype != np.int64
                    or not len(positions)
                    or positions[0] < 1
                    or np.any(positions[1:] <= positions[:-1])
                ):
                    raise InputError(f"{contig}: observation package needs populated, positive, increasing sites")
                if counts.dtype != np.int32 or counts.shape != (len(names), len(positions), 2) or np.any(counts < 0):
                    raise InputError(f"{contig}: FORMAT/AD must be nonnegative int32 REF, ALT counts")
                if (
                    not isinstance(item.length_bp, (int, np.integer))
                    or not positions[-1] <= item.length_bp <= 2147483647
                ):
                    raise InputError(f"{contig}: invalid BCF contig length")
                for column, position in enumerate(positions):
                    ad = "\t".join(f"{int(ref)},{int(alt)}" for ref, alt in counts[:, column])
                    records.write(f"{contig}\t{int(position)}\t.\tA\tC\t.\tPASS\t.\tAD\t{ad}\n")
                filename = f"contig_{index:06d}.npz"
                write_truth(truth_dir / filename, truth)
                entries.append(
                    dict(
                        contig=contig,
                        file=filename,
                        sha256=file_digest(truth_dir / filename),
                        sites=len(positions),
                        positions=array_digest(positions),
                        length_bp=int(item.length_bp),
                        payloads=observation_binding(reads, item.length_bp),
                    )
                )
                del item, reads, truth, positions, counts
        header = (
            FILEFORMAT
            + "##source=synthetic-observations\n"
            + "".join(contig_line(entry["contig"], entry["length_bp"]) for entry in entries)
            + '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="REF and ALT read counts">\n'
            + "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            + "\t".join(names)
            + "\n"
        )
        bcf = stage / "observations.bcf"
        writer = RecordWriter(bcf, header, "bcf")
        try:
            with spool.open() as records:
                for line in records:
                    writer.write(line.rstrip("\n"))
        finally:
            writer.close()
        spool.unlink()
        bcf_index = Path(str(bcf) + ".csi")
        build_index(bcf, bcf_index, csi=True)
        (stage / "inference.toml").write_text(inference_text)
        copied = []
        for name, source in (settings_files or {}).items():
            if Path(name).name != name or name in {
                "inference.toml",
                "observations.bcf",
                "observations.bcf.csi",
                "truth",
            }:
                raise InputError("observation settings files need distinct simple filenames")
            shutil.copyfile(source, stage / name)
            copied.append(dict(path=f"../{name}", sha256=file_digest(stage / name)))
        simulation = manifest.record()
        simulation["stocks"] = [
            {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in stock.items()}
            for stock in simulation["stocks"]
        ]
        document = dict(
            schema=SIDECAR_SCHEMA,
            generator=jsonable(generator),
            simulation=jsonable(simulation),
            sample_ids=names,
            contigs=entries,
            bcf=dict(path="../observations.bcf", sha256=file_digest(bcf), index_sha256=file_digest(bcf_index)),
            settings_files=[dict(path="../inference.toml", sha256=file_digest(stage / "inference.toml")), *copied],
        )
        (truth_dir / "manifest.json").write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
        stage.replace(output)
    return output
