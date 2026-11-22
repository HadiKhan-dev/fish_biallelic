"""Build the exact founder-visible evidence used throughout assembly.

Owns masking, the centred observation view, float32 casting and depth-row content keys (U01).
Memory allocation/attachment and record decoding belong to the stage adapter.
Method: docs/methods/assembly.md#evidence.

Assembly scores the CENTRED view of each observed cell: L~(g) = [(1 - eps) L(g) + eps B] / B, B = sum_g q(g) L(g), the
mixed likelihoods of the observation model (science.shared.observation; eps the run's outlier rate before the
calibration barrier, q the site's cohort genotype frequencies) divided by the cohort likelihood. A genotype law p then
scores log sum_g p(g) L~(g), the log Bayes factor against a random member of the cohort (science.shared.emissions); a
cell without evidence (unobserved, excluded or outside the kept sites) is exactly 1 at every genotype and scores 0.
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
_READ_F64_2 = tuple(types.Array(types.float64, 2, "C", readonly=r) for r in (False, True))
a64 = types.float64[:, :, ::1]
a32 = types.float32[:, :, ::1]
bo = types.boolean[:, ::1]


@kernel(
    overloads=tuple(
        (f64, b2, b1, q, types.float64, a64, a32, bo)
        for f64 in _READ_F64
        for b2 in _READ_BOOL2
        for b1 in _READ_BOOL1
        for q in _READ_F64_2
    ),
    parallel=True,
    cache=True,
    nogil=True,
)
def _fill_views(likelihoods, observed, kept, cohort, epsilon, neutral, scoring, seen):
    """Write the float64 centred view (module docstring), its float32 copy and observed & kept in one pass over the
    cells; cohort float64[L, 3] the sites' cohort genotype frequencies, epsilon the outlier rate."""
    samples, sites, _ = likelihoods.shape
    keep = 1.0 - epsilon
    for sample in prange(samples):
        for site in range(sites):
            used = observed[sample, site] and kept[site]
            seen[sample, site] = used
            if not used:
                for genotype in range(3):
                    neutral[sample, site, genotype] = 1.0
                    scoring[sample, site, genotype] = 1.0
                continue
            l0 = likelihoods[sample, site, 0]
            l1 = likelihoods[sample, site, 1]
            l2 = likelihoods[sample, site, 2]
            background = cohort[site, 0] * l0 + cohort[site, 1] * l1 + cohort[site, 2] * l2
            for genotype in range(3):
                value = keep * likelihoods[sample, site, genotype] / background + epsilon
                neutral[sample, site, genotype] = value
                scoring[sample, site, genotype] = value


def _borrowed(observed: np.ndarray) -> np.ndarray:
    """A read-only view of the input observed flags (the views' flags when every site is kept)."""
    view = observed.view()
    view.flags.writeable = False
    return view


def depth_row_keys(counts, observed, kept, *, map_work=map):
    """Unseeded SHA256 of each founder-visible int64 total-depth vector, for the rows of ``counts`` int[n, L, 2].

    Non-discovery and unobserved cells are zero. Canonical little-endian bytes make the exact
    integer content independent of machine byte order. Neither likelihoods nor sample labels enter.
    Depths are summed DEPTH_ROWS rows at a time to bound scratch storage; the caller
    supplies the ordered map over independent row hashes.
    """
    excluded = ~np.asarray(kept, dtype=bool)
    keys: list[bytes] = []
    for first in range(0, observed.shape[0], DEPTH_ROWS):
        stop = min(observed.shape[0], first + DEPTH_ROWS)
        depth = np.asarray(counts[first:stop]).sum(axis=2, dtype=np.int64).astype("<i8", copy=False)
        depth[~np.asarray(observed[first:stop], dtype=bool)] = 0
        depth[:, excluded] = 0
        rows = [np.ascontiguousarray(row) for row in depth]
        keys.extend(map_work(_sha256, rows))
        del depth, rows
    return tuple(keys)


DEPTH_ROWS = 32  # rows per evidence batch (scratch bound only)


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
    cohort: NDArray[np.float64] | None = field(default=None, repr=False)
    """float64 [L, 3] the sites' cohort genotype frequencies (ContigEvidence.cohort_frequencies): the outlier law of
    the observation model (science.shared.observation)."""

    def __post_init__(self) -> None:
        freeze_record_arrays(self)

    @property
    def samples(self) -> int:
        return int(self.scoring.shape[0])

    @property
    def nbytes(self) -> int:
        keys = 0 if self.depth_keys is None else sum(map(len, self.depth_keys))
        neutral = 0 if self.neutral is None else self.neutral.nbytes
        cohort = 0 if self.cohort is None else self.cohort.nbytes
        return int(neutral + self.scoring.nbytes + self.observed.nbytes + keys + cohort)

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
        cohort,
        outlier_rate: float,
    ) -> AssemblyEvidence:
        """Build the exact centred views (module docstring) from normalised likelihoods, their observed flags, the
        kept sites, the sites' cohort genotype frequencies ``cohort`` float64[L, 3] and the run's ``outlier_rate``."""
        cohort = np.ascontiguousarray(cohort, dtype=np.float64)
        epsilon = float(outlier_rate)
        if cohort.shape != (likelihoods.shape[1], 3) or not 0.0 <= epsilon < 1.0:
            raise ValueError("assembly evidence needs every site's cohort genotype frequencies and a rate in [0, 1)")
        # The observed flags of the views are observed & kept: with every site kept they ARE the input flags, which
        # are then borrowed read-only instead of copied ([N, L] bool; 4.20 borrowed them on the same condition).
        borrow = isinstance(observed, np.ndarray) and bool(np.all(kept))
        if isinstance(likelihoods, CodedLikelihoods):
            neutral = np.empty(likelihoods.shape, dtype=np.float64)
            scoring = np.empty(likelihoods.shape, dtype=np.float32)
            seen = _borrowed(observed) if borrow else np.empty(observed.shape, dtype=np.bool_)
            if depth_keys is None and counts is not None:
                depth_keys = depth_row_keys(counts, observed, kept)
            # Gather and fill DEPTH_ROWS samples at a time so the sample-parallel kernel has
            # work for its team while the decoded slice stays small.
            kept_c = np.ascontiguousarray(kept)
            for first in range(0, likelihoods.shape[0], DEPTH_ROWS):
                stop = min(likelihoods.shape[0], first + DEPTH_ROWS)
                values = likelihoods.gather(slice(first, stop), slice(None))
                _fill_views(
                    values,
                    np.ascontiguousarray(observed[first:stop]),
                    kept_c,
                    cohort,
                    epsilon,
                    neutral[first:stop],
                    scoring[first:stop],
                    np.empty((stop - first, likelihoods.shape[1]), dtype=np.bool_) if borrow else seen[first:stop],
                )
                del values
            for array in (neutral, scoring) if borrow else (neutral, scoring, seen):
                declare_immutable(array)
            return cls(
                np.asarray(positions), neutral, scoring, seen, np.asarray(kept), str(identity), depth_keys, cohort=cohort
            )
        observed = dense_observed(observed)
        likelihoods = np.ascontiguousarray(likelihoods, dtype=np.float64)
        kept = np.ascontiguousarray(kept, dtype=np.bool_)
        observed = np.ascontiguousarray(observed, dtype=np.bool_)
        if depth_keys is None and counts is not None:
            depth_keys = depth_row_keys(counts, observed, kept)
        neutral = np.empty(likelihoods.shape, dtype=np.float64)
        scoring = np.empty(likelihoods.shape, dtype=np.float32)
        if borrow:
            # one sample-parallel call; its flags output is transient (the views keep the borrowed input flags)
            _fill_views(likelihoods, observed, kept, cohort, epsilon, neutral, scoring, np.empty(observed.shape, np.bool_))
            seen = _borrowed(observed)
        else:
            seen = np.empty(observed.shape, dtype=np.bool_)
            _fill_views(likelihoods, observed, kept, cohort, epsilon, neutral, scoring, seen)
        for array in (neutral, scoring) if borrow else (neutral, scoring, seen):
            declare_immutable(array)  # owned and filled once
        return cls(np.asarray(positions), neutral, scoring, seen, kept, str(identity), depth_keys, cohort=cohort)

    @classmethod
    def from_contig(
        cls, evidence, identity: str, *, outlier_rate: float, counts=None, depth_keys=None
    ) -> AssemblyEvidence:
        """Mask founder-excluded samples before hashing exact observed total depths; ``outlier_rate`` the run's rate
        before the calibration barrier."""
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
            cohort=evidence.cohort_frequencies,
            outlier_rate=outlier_rate,
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
    cohort: NDArray[np.float64]
    outlier_rate: float

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
            cohort=self.cohort,
            outlier_rate=self.outlier_rate,
        )
