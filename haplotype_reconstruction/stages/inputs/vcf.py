"""Read biallelic allele depths with the production admission and duplicate policy."""

from __future__ import annotations

from array import array
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from cyvcf2 import VCF

from haplotype_reconstruction.contracts.source import (
    DROPPED_DUPLICATE_POSITION,
    DROPPED_MULTIALLELIC,
    DROPPED_NOT_SNP,
    ContigReads,
)
from haplotype_reconstruction.core.digest import json_digest
from haplotype_reconstruction.core.errors import InputError
from haplotype_reconstruction.store.protocol import ContigPayload, payload_identity

from . import vcf_alleles

_BASES = frozenset("ACGTacgt")
_WINDOWS_PER_WORKER = 4
_MINIMUM_WINDOW_BP = 100_000


@dataclass(frozen=True)
class VcfHeader:
    """What the header of an input file declares."""

    sample_ids: tuple[str, ...]
    contigs: tuple[str, ...]
    lengths: dict  # contig -> length in bp (empty when the header gives none)
    has_allele_depths: bool


def inspect(path: str | Path) -> VcfHeader:
    """Read the header of an indexed VCF/BCF."""
    reader = VCF(str(path))
    try:
        from ...store.releases import CURRENT_RELEASE, unsupported_release

        lines = reader.raw_header.splitlines()
        if any(line.startswith("##source=haplotype-reconstruction") for line in lines):
            prefix = "##haplotype_reconstruction_release="
            releases = [line[len(prefix) :] for line in lines if line.startswith(prefix)]
            if releases != [CURRENT_RELEASE]:
                raise unsupported_release(path, releases[0] if releases else None)
        try:
            lengths = dict(zip(reader.seqnames, map(int, reader.seqlens), strict=False))
        except AttributeError:  # cyvcf2: "no sequence lengths found in header"
            lengths = {}
        return VcfHeader(
            sample_ids=tuple(reader.samples),
            contigs=tuple(reader.seqnames),
            lengths=lengths,
            has_allele_depths="##FORMAT=<ID=AD," in reader.raw_header,
        )
    finally:
        reader.close()


def _decode_window(
    path: str | Path, contig: str, start: int, stop: int | None, samples: Sequence[str] | None, with_payload: bool
):
    """One pass over the records starting in [start, stop): admit biallelic SNPs, decode and check their AD, then
    drop every member of each duplicate-position run. Returns (positions, counts int32[N, L, 2], metadata or None,
    not_snp, multiallelic, duplicated). Malformed AD raises even at duplicate positions, before filtering."""
    reader = VCF(str(path), lazy=True) if samples is None else VCF(str(path), samples=list(samples), lazy=True)
    try:
        n_samples = len(reader.samples)
        if stop is None:
            region = contig if start <= 1 else f"{contig}:{start}-"
        else:
            region = f"{contig}:{start}-{stop - 1}"
        admitted = array("q")
        rows: list[np.ndarray | None] = []
        refs: list[str] = []
        alts: list[str] = []
        fallback: list[np.ndarray | None] = []
        not_snp = multiallelic = 0
        previous = 0
        for record in reader(region):
            position = record.POS
            if position < start or (stop is not None and position >= stop):
                continue  # starts outside this window (overlap query)
            if position < previous:
                raise InputError(f"{path}: {contig}:{position}: records are not sorted by position")
            previous = position
            if len(record.ALT) > 1:
                multiallelic += 1
                continue
            if not vcf_alleles.biallelic_snp(record):
                not_snp += 1
                continue
            ad = record.format("AD")
            if ad is None or (ad.shape[1] == 1 and np.all(ad < 0)):
                # No AD, or HTSlib's single all-missing sentinel column: zero observed reads.
                ad = None
            elif ad.shape != (n_samples, 2):
                raise InputError(
                    f"{path}: {contig}:{position}: AD must hold REF and ALT depths "
                    f"for every sample (found shape {ad.shape})"
                )
            admitted.append(position)
            # Copies: cyvcf2 may reuse a record's FORMAT buffers once the iterator advances.
            rows.append(None if ad is None else np.maximum(ad, 0).astype(np.int32, copy=False))
            if with_payload:
                refs.append(record.REF)
                alts.append(record.ALT[0])
                fallback.append(
                    np.array(record.gt_types, dtype=np.int8) if ad is None and "GT" in record.FORMAT else None
                )
    finally:
        reader.close()
    positions = np.asarray(admitted, np.int64)
    # Equal positions always belong to the same physical window. Drop every member of each
    # duplicate run, including the first copy, so no duplicate position enters the source axis.
    keep = np.ones(len(positions), bool)
    repeated = positions[1:] == positions[:-1]
    keep[1:] &= ~repeated
    keep[:-1] &= ~repeated
    duplicated = int(len(positions) - np.count_nonzero(keep))
    kept = np.flatnonzero(keep)
    counts = np.zeros((n_samples, len(kept), 2), np.int32)
    for column, index in enumerate(kept):
        ad = rows[index]
        if ad is not None:
            counts[:, column, :] = ad
    metadata = None
    if with_payload:
        ref = np.asarray([refs[i] for i in kept], dtype="<U1")
        alt = np.asarray([alts[i] for i in kept], dtype="<U1")
        usable = np.asarray([rows[i] is not None for i in kept], dtype=bool)
        gt_types = np.full((n_samples, len(kept)), 2, dtype=np.int8)
        for column, index in enumerate(kept):
            if rows[index] is None and fallback[index] is not None:
                gt_types[:, column] = fallback[index]
        metadata = (ref, alt, usable, gt_types)
    return positions[keep], counts, metadata, not_snp, multiallelic, duplicated


def _windows(length: int | None, workers: int) -> list[tuple[int, int | None]]:
    """Disjoint [start, stop) windows covering 1..length (one open window when length is unknown)."""
    if length is None or workers <= 1:
        return [(1, None)]
    count = max(1, min(workers * _WINDOWS_PER_WORKER, length // _MINIMUM_WINDOW_BP))
    bounds = np.linspace(1, length + 1, count + 1).astype(np.int64)
    windows: list[tuple[int, int | None]] = [
        (int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:], strict=False) if b > a
    ]
    # Records beyond the declared length (a malformed header) still belong to the last window.
    windows[-1] = (windows[-1][0], None)
    return windows


def _read(
    path: str | Path, contig: str, workers: int, samples: Sequence[str] | None, *, with_payload: bool
) -> ContigReads | tuple[ContigReads, ContigPayload]:
    """Decode admitted SNPs in header sample order, without changing the source."""
    header = inspect(path)
    if contig not in header.contigs:
        raise InputError(f"{path}: contig {contig!r} is not in the header")
    if not header.has_allele_depths:
        raise InputError(f"{path}: FORMAT/AD allele depths are required (GT- or PL-only input is not supported)")
    sample_ids = header.sample_ids
    if samples is not None:
        unknown = sorted(set(samples) - set(sample_ids))
        if unknown:
            raise InputError(f"{path}: samples not in the header: {unknown}")
        sample_ids = tuple(name for name in sample_ids if name in set(samples))
    tasks = [
        (str(path), contig, start, stop, None if samples is None else sample_ids, with_payload)
        for start, stop in _windows(header.lengths.get(contig), workers)
    ]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda task: _decode_window(*task), tasks))
    positions = np.concatenate([result[0] for result in results])
    counts = np.concatenate([result[1] for result in results], axis=1) if len(results) > 1 else results[0][1]
    if with_payload:
        ref = np.concatenate([result[2][0] for result in results])
        alt = np.concatenate([result[2][1] for result in results])
        usable = np.concatenate([result[2][2] for result in results])
        gt_types = (
            np.concatenate([result[2][3] for result in results], axis=1) if len(results) > 1 else results[0][2][3]
        )
    dropped = {
        DROPPED_NOT_SNP: sum(r[3] for r in results),
        DROPPED_MULTIALLELIC: sum(r[4] for r in results),
        DROPPED_DUPLICATE_POSITION: sum(r[5] for r in results),
    }
    reads = ContigReads(contig, sample_ids, positions, counts, dropped)
    if not with_payload:
        return reads
    usable.flags.writeable = False
    gt_types.flags.writeable = False
    payload = ContigPayload(
        contig,
        sample_ids,
        reads.positions,
        tuple(ref.tolist()),
        tuple(alt.tolist()),
        reads.counts,
        usable,
        gt_types if not usable.all() else None,
        header.lengths.get(contig),
        dropped,
    )
    return reads, payload


def read_contig(
    path: str | Path, contig: str, workers: int = 1, *, samples: Sequence[str] | None = None
) -> ContigReads:
    """Return int32[N,L,2] depths after the exact EC01 admission policy.

    One pass per window: admit, decode, drop duplicates.

    workers bounds independent physical windows, gathered in position order.
    Malformed AD raises InputError even at duplicate positions; inputs are unchanged.
    """
    return cast(ContigReads, _read(path, contig, workers, samples, with_payload=False))


def read_with_payload(
    path: str | Path, contig: str, workers: int = 1, *, samples: Sequence[str] | None = None
) -> tuple[ContigReads, ContigPayload]:
    """Decode observations and consumed-source identity fields in the same single pass.

    AD availability is per record; fallback GT is retained only where AD is unusable.
    Input FILTER and GT at usable AD records have no effect on this payload.
    """
    return cast(tuple[ContigReads, ContigPayload], _read(path, contig, workers, samples, with_payload=True))


class VcfSource:
    """Indexed observations with explicit header checks and per-contig payload identity.

    Owns no checkpoint or digest cache: the evidence task binds those through ST1.
    Sample order here is the header order; CanonicalOrderSource permutes it once.
    """

    kind = "vcf"

    def __init__(self, path: str | Path, contigs: Sequence[str], *, workers: int = 1) -> None:
        self.path = Path(path)
        self._contigs = tuple(contigs)
        self.workers = workers
        self.header = inspect(self.path)

    def check(self) -> None:
        """Require AD, three unique samples, and declared populated contigs (EC01)."""
        header = self.header
        if not header.has_allele_depths:
            raise InputError(
                f"{self.path}: FORMAT/AD allele depths are required (GT- or PL-only input is not supported)"
            )
        if len(header.sample_ids) < 3 or len(set(header.sample_ids)) != len(header.sample_ids):
            raise InputError(f"{self.path}: at least three uniquely named samples are required")
        missing = [contig for contig in self._contigs if contig not in header.contigs]
        if missing:
            raise InputError(f"{self.path}: contigs not in the header: {', '.join(missing)}")
        reader = VCF(str(self.path))
        try:
            empty = [contig for contig in self._contigs if next(iter(reader(contig)), None) is None]
        finally:
            reader.close()
        if empty:
            raise InputError(f"{self.path}: no indexed records for {', '.join(empty)}; select populated contigs")

    def contigs(self) -> tuple[str, ...]:
        """Configured contigs in their declared run order."""
        return self._contigs

    def sample_ids(self) -> tuple[str, ...]:
        """Header sample identifiers, before canonicalisation."""
        return self.header.sample_ids

    def identity(self) -> str:
        """Source descriptor independent of path, timestamp and header sample permutation."""
        return json_digest(dict(kind=self.kind, contigs=self._contigs, sample_ids=sorted(self.header.sample_ids)))

    def reads(self, contig: str, team: int | None = None) -> ContigReads:
        """Read this contig's observations without retaining identity scratch."""
        return read_contig(self.path, contig, self.workers if team is None else team)

    def read_with_payload(self, contig: str, team: int | None = None) -> tuple[ContigReads, ContigPayload]:
        """Read both products together; no checkpoint binding or cache side effect."""
        return read_with_payload(self.path, contig, self.workers if team is None else team)

    def payload_identity(self, contig: str) -> str:
        """Digest freshly decoded consumed fields in this source's sample order."""
        return payload_identity(self.read_with_payload(contig)[1])

    def site_alleles(self, contig: str, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, int | None]:
        """Return REF/ALT and header length at requested positions; missing sites raise InputError."""
        payload = self.read_with_payload(contig)[1]
        return vcf_alleles.site_alleles(payload, positions, expected_identity=payload_identity(payload))

    def allele_totals(self, contig: str, positions: np.ndarray, samples: Sequence[str]) -> tuple[np.ndarray, ...]:
        """Return AD totals, or unphased GT totals only at unusable AD records (OU01)."""
        payload = self.read_with_payload(contig)[1]
        return vcf_alleles.allele_totals(payload, positions, samples, expected_identity=payload_identity(payload))


__all__ = ["VcfHeader", "VcfSource", "inspect", "read_contig", "read_with_payload"]


@dataclass(frozen=True)
class VerifiedVcfView:
    """One fresh payload verified against evidence, shared by both export protocol calls."""

    payload: ContigPayload
    expected_identity: str

    def __post_init__(self) -> None:
        vcf_alleles.verify(self.payload, self.expected_identity)

    def site_alleles(self, contig: str, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, int | None]:
        """Return alleles from this verified payload; a different contig is an input error."""
        if contig != self.payload.contig:
            raise InputError(f"{contig}: verified payload belongs to {self.payload.contig}")
        return vcf_alleles.site_alleles(self.payload, positions, expected_identity=self.expected_identity)

    def allele_totals(self, contig: str, positions: np.ndarray, samples: Sequence[str]) -> tuple[np.ndarray, ...]:
        """Return source totals using the same fresh payload as site_alleles."""
        if contig != self.payload.contig:
            raise InputError(f"{contig}: verified payload belongs to {self.payload.contig}")
        return vcf_alleles.allele_totals(self.payload, positions, samples, expected_identity=self.expected_identity)
