"""Declare observations and consumed-source fields independently of their reader (EC01/03).

Owns ContigReads and ObservationSource, including allele and provenance methods.
No file, storage, engine or science calls belong here. Sample identifiers name rows
only. Positions are 1-based bp; missing AD contributes zero counts. Payload identity
binds positions, REF/ALT, AD, record-level AD availability, fallback GT only where AD
is unusable, header length and loader drops (schema 2); it never binds input FILTER.
Method: docs/methods/evidence.md.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .roles import frozen_array, record_field

__all__ = [
    "ContigReads",
    "ObservationSource",
    "DROPPED_NOT_SNP",
    "DROPPED_MULTIALLELIC",
    "DROPPED_DUPLICATE_POSITION",
    "DROPPED_REASONS",
]

# Loader omissions, counted before site-quality filtering.
DROPPED_NOT_SNP = "not_snp"
DROPPED_MULTIALLELIC = "multiallelic"
DROPPED_DUPLICATE_POSITION = "duplicate_position"
DROPPED_REASONS = (DROPPED_NOT_SNP, DROPPED_MULTIALLELIC, DROPPED_DUPLICATE_POSITION)


@dataclass(frozen=True)
class ContigReads:
    """Read counts int32[N,L,2] at strictly increasing int64[L] input SNP positions.

    sample_ids gives the counts' current sample order (canonical after the source
    adapter). dropped counts loader omissions, never site-quality decisions.
    Arrays are read-only, sharing already compatible buffers; caller flags are unchanged.
    """

    contig: str = record_field("obs", dtype="str")
    sample_ids: tuple[str, ...] = record_field("obs", axes=("N",), dtype="str")
    positions: NDArray[np.int64] = record_field("obs", axes=("L",), units="bp", dtype="int64")
    counts: NDArray[np.int32] = record_field("obs", axes=("N", "L", "2"), units="reads", dtype="int32")
    dropped: Mapping[str, int] = record_field("obs", units="count", dtype="Mapping[str, int]", default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_ids", tuple(map(str, self.sample_ids)))
        object.__setattr__(self, "positions", frozen_array(self.positions, np.int64))
        object.__setattr__(self, "counts", frozen_array(self.counts, np.int32))
        object.__setattr__(self, "dropped", MappingProxyType({str(k): int(v) for k, v in self.dropped.items()}))

    @property
    def n_samples(self) -> int:
        """Number N of observed sample identifiers."""
        return len(self.sample_ids)

    @property
    def n_sites(self) -> int:
        """Number L of retained input SNP records."""
        return len(self.positions)

    def select_sites(self, keep: NDArray[np.bool_]) -> ContigReads:
        """Copy only sites where keep bool[L] is true, retaining sample/drop provenance."""
        keep = np.asarray(keep, bool)
        return replace(self, positions=self.positions[keep], counts=self.counts[:, keep])


@runtime_checkable
class ObservationSource(Protocol):
    """Observation-only source implemented by VCF and simulation adapters.

    The adapter owns I/O and canonical ordering. No implementation may read truth
    through these methods. Allele-total samples are identifiers, not row indices.
    """

    def identity(self) -> str | Mapping[str, object]:
        """Return the bound source descriptor identity (simulation: design and realised seed)."""
        ...

    def payload_identity(self, contig: str) -> str:
        """Digest every consumed field of this contig, including GT only where AD is unusable."""
        ...

    def sample_ids(self) -> tuple[str, ...]:
        """Return the source's sample order; names carry no pedigree metadata."""
        ...

    def contigs(self) -> tuple[str, ...]:
        """Return configured contigs in their bound run order."""
        ...

    def reads(self, contig: str) -> ContigReads:
        """Return observed depths and loader drops at all retained biallelic SNP sites."""
        ...

    def site_alleles(
        self,
        contig: str,
        positions: NDArray[np.int64],
    ) -> tuple[NDArray[np.str_], NDArray[np.str_], int | None]:
        """Return REF/ALT <U1[M] and header length in bp for positions int64[M].

        Simulation returns A/C and None. Missing source positions raise InputError.
        """
        ...

    def allele_totals(
        self,
        contig: str,
        positions: NDArray[np.int64],
        samples: Sequence[str],
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int8]]:
        """Return REF/ALT int64[M'] totals over sample names and source int8[M'].

        VCF uses AD where record-level ad_usable (source 1), including zero depth,
        and GT alleles otherwise (source 2). Simulation uses reads (source 1).
        No genotype fallback is used in reads(), and no truth is read here.
        """
        ...
