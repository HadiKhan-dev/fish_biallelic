"""Build the exact founder-visible evidence used throughout assembly.

Owns masking, float32 casting and depth-row content keys (P0 U01).
Memory allocation/attachment and record decoding belong to the stage adapter.
Method: docs/methods/assembly.md#evidence. Exact neutral inputs may be borrowed; scoring is owned.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace

import numpy as np
from numba import prange, types
from numpy.typing import NDArray

from haplotype_reconstruction.contracts.likelihoods import CodedLikelihoods
from haplotype_reconstruction.contracts.observations import PairArray, dense_observed
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
    nogil=True,
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


def _depth_keys(counts, observed, kept, threads=1):
    """Unseeded SHA256 of each founder-visible int64 total-depth vector, for the rows of ``counts`` int[n, L, 2].

    Non-discovery and unobserved cells are zero. Canonical little-endian bytes make the exact
    integer content independent of machine byte order. Neither likelihoods nor sample labels enter.
    Release 4.8: depths of up to _DEPTH_ROWS rows at once (bounded scratch), hashed on up to ``threads`` threads
    (hashing releases the GIL).
    """
    excluded = ~np.asarray(kept, dtype=bool)
    keys: list[bytes] = []
    pool = None
    if threads > 1 and observed.shape[0] > 1:
        from concurrent.futures import ThreadPoolExecutor

        pool = ThreadPoolExecutor(max_workers=min(int(threads), _DEPTH_ROWS, observed.shape[0]))
    try:
        for first in range(0, observed.shape[0], _DEPTH_ROWS):
            stop = min(observed.shape[0], first + _DEPTH_ROWS)
            depth = np.asarray(counts[first:stop]).sum(axis=2, dtype=np.int64).astype("<i8", copy=False)
            depth[~np.asarray(observed[first:stop], dtype=bool)] = 0
            depth[:, excluded] = 0
            rows = [np.ascontiguousarray(row) for row in depth]
            hashed = map(_sha256, rows) if pool is None else pool.map(_sha256, rows)
            keys.extend(hashed)
            del depth, rows
    finally:
        if pool is not None:
            pool.shutdown()
    return tuple(keys)


_DEPTH_ROWS = 32  # rows per depth batch in _depth_keys (scratch bound only)


def _sha256(row):
    return hashlib.sha256(row).digest()


@dataclass(frozen=True)
class SelectionEvidence:
    """Borrowed founder evidence: positions[L], float64 GL[N,L,3], observed[N,L]."""

    positions: NDArray[np.int64]
    likelihoods: CodedLikelihoods | NDArray[np.float64]
    observed: NDArray[np.bool_] | PairArray

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
        cls,
        positions,
        likelihoods,
        observed,
        kept,
        identity: str,
        *,
        counts=None,
        depth_keys=None,
        borrow_neutral: bool = False,
    ) -> AssemblyEvidence:
        """Build the exact views; borrow_neutral certifies already neutral raw evidence.

        Only the ContigEvidence producer can establish that missing raw cells are
        exactly 1/3, every site is kept, and no founder row was excluded. Generic
        likelihood callers retain the owned/masked route by default.
        """
        if isinstance(likelihoods, CodedLikelihoods):
            neutral = np.empty(likelihoods.shape, dtype=np.float64)
            scoring = np.empty(likelihoods.shape, dtype=np.float32)
            borrow_mask = borrow_neutral and isinstance(observed, np.ndarray)
            seen = observed.view() if borrow_mask else np.empty(observed.shape, dtype=np.bool_)
            if borrow_mask:
                seen.flags.writeable = False
            if depth_keys is None and counts is not None:
                depth_keys = _depth_keys(counts, observed, kept)
            for sample in range(likelihoods.shape[0]):
                values = likelihoods.gather(slice(sample, sample + 1), slice(None))
                flags = (
                    np.empty((1, likelihoods.shape[1]), dtype=np.bool_) if borrow_mask else seen[sample : sample + 1]
                )
                _fill_views(
                    values,
                    np.ascontiguousarray(observed[sample : sample + 1]),
                    np.ascontiguousarray(kept),
                    neutral[sample : sample + 1],
                    scoring[sample : sample + 1],
                    flags,
                )
                del values
            for array in (neutral, scoring):
                declare_immutable(array)
            if not borrow_mask:
                declare_immutable(seen)
            return cls(np.asarray(positions), neutral, scoring, seen, np.asarray(kept), str(identity), depth_keys)
        observed = dense_observed(observed)
        likelihoods = np.ascontiguousarray(likelihoods, dtype=np.float64)
        kept = np.ascontiguousarray(kept, dtype=np.bool_)
        observed = np.ascontiguousarray(observed, dtype=np.bool_)
        if depth_keys is None and counts is not None:
            depth_keys = _depth_keys(counts, observed, kept)
        if borrow_neutral:
            neutral, seen = likelihoods.view(), observed.view()
            neutral.flags.writeable = seen.flags.writeable = False
            scoring = np.array(likelihoods, dtype=np.float32, order="C", copy=True)
            declare_immutable(scoring)
        else:
            neutral = np.empty(likelihoods.shape, dtype=np.float64)
            scoring = np.empty(likelihoods.shape, dtype=np.float32)
            seen = np.empty(observed.shape, dtype=np.bool_)
            _fill_views(likelihoods, observed, kept, neutral, scoring, seen)
            for array in (neutral, scoring, seen):
                declare_immutable(array)  # owned and filled once; borrowed views are never declared owned
        return cls(np.asarray(positions), neutral, scoring, seen, kept, str(identity), depth_keys)

    @classmethod
    def from_contig(cls, evidence, identity: str, *, counts=None, depth_keys=None) -> AssemblyEvidence:
        """Mask founder-excluded samples before hashing exact observed total depths."""
        counts = evidence.counts if counts is None else counts
        source = replace(evidence, counts=None) if evidence.counts is not None else evidence
        founder = source.founder_view()
        return cls.from_likelihoods(
            founder.positions,
            founder.likelihoods,
            founder.observed,
            founder.discovery_sites,
            identity,
            counts=counts,
            depth_keys=depth_keys,
            borrow_neutral=bool(evidence.discovery_sites.all() and not evidence.founder_excluded.any()),
        )


@dataclass(frozen=True)
class AssemblyEvidenceSource:
    """Borrowed inputs for an assembly consumer; constructing this does not build its views.

    Each call owns its scoring/masked arrays and release_neutral lifetime. The
    source never caches a materialised view across selection/assembly actions.
    """

    positions: NDArray[np.int64]
    likelihoods: CodedLikelihoods | NDArray[np.float64]
    observed: NDArray[np.bool_] | PairArray
    kept: NDArray[np.bool_]
    identity: str
    depth_keys: tuple[bytes, ...] | None
    borrow_neutral: bool

    def __post_init__(self) -> None:
        freeze_record_arrays(self)

    def __call__(self) -> AssemblyEvidence:
        return AssemblyEvidence.from_likelihoods(
            self.positions,
            self.likelihoods,
            self.observed,
            self.kept,
            self.identity,
            depth_keys=self.depth_keys,
            borrow_neutral=self.borrow_neutral,
        )
