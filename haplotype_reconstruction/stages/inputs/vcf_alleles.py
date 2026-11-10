"""Verify consumed VCF payloads before exposing export alleles and exact totals.

Owns the shared biallelic-SNP predicate and OU01/OU02 protocol views. The VCF source
supplies its freshly read ContigPayload; this module never repeats the loader policy.
Method: docs/methods/export.md#sites. AD availability is
record-level, including usable zero-depth AD; GT contributes only where AD is absent.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from ...core.errors import InputError
from ...store.protocol import ContigPayload, payload_identity

__all__ = ["biallelic_snp", "verify", "site_alleles", "allele_totals"]
_BASES = frozenset("ACGTacgt")


class VariantAlleles(Protocol):
    """The two allele properties read from a cyvcf2 variant."""

    REF: str
    ALT: list[str]


def biallelic_snp(record: VariantAlleles) -> bool:
    """Whether the record has exactly one REF and ALT nucleotide; mutates nothing."""
    return len(record.ALT) == 1 and record.REF in _BASES and record.ALT[0] in _BASES


def verify(payload: ContigPayload, expected_identity: str) -> None:
    """Refuse a freshly read payload differing from the evidence-bound identity.

    Raises InputError for changed consumed source values; filenames and input FILTER
    are deliberately outside this comparison. No array is mutated.
    """
    actual = payload_identity(payload)
    if actual != expected_identity:
        raise InputError(f"{payload.contig}: input payload {actual} differs from recorded {expected_identity}")


def _columns(payload: ContigPayload, positions: np.ndarray) -> np.ndarray:
    requested = np.asarray(positions, dtype=np.int64)
    columns = np.asarray(np.searchsorted(payload.positions, requested), dtype=np.int64)
    if np.any(columns >= len(payload.positions)) or not np.array_equal(payload.positions[columns], requested):
        raise InputError(f"{payload.contig}: input VCF lacks an exported biallelic SNP")
    return columns


def site_alleles(
    payload: ContigPayload, positions: np.ndarray, *, expected_identity: str
) -> tuple[np.ndarray, np.ndarray, int | None]:
    """Return REF/ALT <U1[M] and header length for requested int64[M] positions.

    The source must freshly read the payload before constructing its per-contig
    export view. Raises InputError on a payload mismatch or missing position.
    """
    verify(payload, expected_identity)
    columns = _columns(payload, positions)
    return (
        np.asarray(payload.ref, dtype="<U1")[columns],
        np.asarray(payload.alt, dtype="<U1")[columns],
        payload.contig_length,
    )


def allele_totals(
    payload: ContigPayload, positions: np.ndarray, samples: Sequence[str], *, expected_identity: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact int64[M] REF/ALT totals and int8[M] AD=1/GT=2 provenance.

    Samples are counted once in their recorded axis. Missing/negative AD is already
    zero in the loader's payload. Fallback GT classes are cyvcf2's 0/1/2/3. InputError
    reports a missing sample/site or identity mismatch; input arrays remain unchanged.
    """
    verify(payload, expected_identity)
    columns = _columns(payload, positions)
    chosen = set(samples)
    if chosen - set(payload.sample_ids):
        raise InputError(f"{payload.contig}: export samples are absent from the original VCF")
    rows = tuple(index for index, name in enumerate(payload.sample_ids) if name in chosen)
    # Sum one sample at a time: exact int64 addition with O(M') temporary storage.
    # Avoid the N x M copy made by indexing the sample axis before requested sites.
    ref: np.ndarray = np.zeros(len(columns), dtype=np.int64)
    alt: np.ndarray = np.zeros(len(columns), dtype=np.int64)
    fallback = ~np.asarray(payload.ad_usable, dtype=bool)[columns]
    source = np.where(fallback, 2, 1).astype(np.int8)
    for row in rows:
        ref += payload.counts[row, columns, 0]
        alt += payload.counts[row, columns, 1]
    if np.any(fallback):
        ref[fallback] = 0
        alt[fallback] = 0
        for row in rows:
            types = np.asarray(payload.gt_types)[row, columns[fallback]]
            ref[fallback] += 2 * (types == 0).astype(np.int64) + (types == 1)
            alt[fallback] += 2 * (types == 3).astype(np.int64) + (types == 1)
    return ref, alt, source
