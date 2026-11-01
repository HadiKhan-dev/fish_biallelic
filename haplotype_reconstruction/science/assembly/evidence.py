"""Build the exact founder-visible evidence used throughout assembly.

Owns masking, float32 casting and depth-row content keys (P0 U01).
Memory allocation/attachment and record decoding belong to the stage adapter.
Method: docs/methods/assembly.md#evidence. Inputs are borrowed; outputs are owned.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace

import numpy as np
from numba import prange, types
from numpy.typing import NDArray

from haplotype_reconstruction.contracts.roles import freeze_record_arrays
from haplotype_reconstruction.core.immutable import declare_immutable
from haplotype_reconstruction.core.kernels import kernel

_READ_F64 = tuple(types.Array(types.float64, 3, "C", readonly=r) for r in (False, True))
_READ_BOOL2 = tuple(types.Array(types.boolean, 2, "C", readonly=r) for r in (False, True))
_READ_BOOL1 = tuple(types.Array(types.boolean, 1, "C", readonly=r) for r in (False, True))
a64 = types.float64[:, :, ::1]
a32 = types.float32[:, :, ::1]
bo = types.boolean[:, ::1]


@kernel(
    overloads=tuple((f64, b2, b1, a64, a32, bo) for f64 in _READ_F64 for b2 in _READ_BOOL2 for b1 in _READ_BOOL1),
    parallel=True,
    cache=True,
)
def _fill_views(likelihoods, observed, kept, neutral, scoring, seen):
    """Write the float64 neutral view, its float32 copy and observed & kept in one pass over the cells."""
    samples, sites, _ = likelihoods.shape
    for sample in prange(samples):
        for site in range(sites):
            used = observed[sample, site] and kept[site]
            seen[sample, site] = used
            for genotype in range(3):
                value = likelihoods[sample, site, genotype] if used else 1.0 / 3.0
                neutral[sample, site, genotype] = value
                scoring[sample, site, genotype] = value


def _depth_keys(counts, observed, kept):
    """Unseeded SHA256 of each founder-visible int64 total-depth vector; one row of scratch.

    Non-discovery and unobserved cells are zero. Canonical little-endian bytes make the exact
    integer content independent of machine byte order. Neither likelihoods nor sample labels enter.
    """
    keys = []
    for sample in range(observed.shape[0]):
        depth = counts[sample].sum(axis=1, dtype=np.int64).astype("<i8", copy=False)
        depth[~observed[sample]] = 0
        depth[~kept] = 0
        keys.append(hashlib.sha256(depth).digest())
    return tuple(keys)


@dataclass(frozen=True)
class SelectionEvidence:
    """Borrowed founder evidence: positions[L], float64 GL[N,L,3], observed[N,L]."""

    positions: NDArray[np.int64]
    likelihoods: NDArray[np.float64]
    observed: NDArray[np.bool_]

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class AssemblyEvidence:
    """Exact neutral float64 and scoring float32 views on one contig's site axis.

    Neutral may be released after the final join. Scoring and observation masks
    remain valid. No shared-memory handles or record readers enter this record.
    """

    positions: NDArray[np.int64]
    neutral: NDArray[np.float64] | None
    scoring: NDArray[np.float32]
    observed: NDArray[np.bool_]
    kept: NDArray[np.bool_]
    identity: str
    depth_keys: tuple[bytes, ...] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        freeze_record_arrays(self)

    @property
    def samples(self) -> int:
        return int(self.scoring.shape[0])

    @property
    def nbytes(self) -> int:
        keys = 0 if self.depth_keys is None else sum(map(len, self.depth_keys))
        neutral = 0 if self.neutral is None else self.neutral.nbytes
        return int(neutral + self.scoring.nbytes + self.observed.nbytes + keys)

    def founder_evidence(self) -> SelectionEvidence:
        """Borrow the neutral view before its final-assembly lifetime ends."""
        if self.neutral is None:
            raise ValueError("the founder view's neutral evidence was released by a final assembly")
        return SelectionEvidence(self.positions, self.neutral, self.observed)

    def release_neutral(self) -> None:
        """Drop this view's neutral reference after the final join, before refinement."""
        object.__setattr__(self, "neutral", None)

    @classmethod
    def from_likelihoods(
        cls, positions, likelihoods, observed, kept, identity: str, *, counts=None, depth_keys=None
    ) -> AssemblyEvidence:
        """Build owned arrays; observed discovery cells retain the exact input GL."""
        likelihoods = np.ascontiguousarray(likelihoods, dtype=np.float64)
        kept = np.ascontiguousarray(kept, dtype=np.bool_)
        observed = np.ascontiguousarray(observed, dtype=np.bool_)
        if depth_keys is None and counts is not None:
            depth_keys = _depth_keys(counts, observed, kept)
        neutral = np.empty(likelihoods.shape, dtype=np.float64)
        scoring = np.empty(likelihoods.shape, dtype=np.float32)
        seen = np.empty(observed.shape, dtype=np.bool_)
        _fill_views(likelihoods, observed, kept, neutral, scoring, seen)
        for array in (neutral, scoring, seen):
            declare_immutable(array)  # owned, filled once and never changed: tasks may share it (core.immutable)
        return cls(np.asarray(positions), neutral, scoring, seen, kept, str(identity), depth_keys)

    @classmethod
    def from_contig(cls, evidence, identity: str, *, counts=None) -> AssemblyEvidence:
        """Mask founder-excluded samples before hashing exact observed total depths."""
        counts = evidence.counts if counts is None else counts
        source = replace(evidence, counts=None) if evidence.counts is not None else evidence
        founder = source.founder_view()
        return cls.from_likelihoods(
            founder.positions, founder.likelihoods, founder.observed, founder.discovery_sites, identity, counts=counts
        )
