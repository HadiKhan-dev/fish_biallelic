"""Exact read counts and observation flags backed by the GL pair dictionary.

Only fallback rows own dense arrays. Views borrow the same GL codes/pairs; no
implicit ndarray conversion can accidentally expand a whole contig.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from .likelihoods import (
    PARALLEL_GATHER_CELLS,
    CodedLikelihoods,
    _coded_flags,
    _columns,
    _read_rows,
    coded_arguments,
    readonly,
    resident_coded,
    table_rows,
)
from .roles import freeze_record_arrays, record_field

ObservationKind = Literal["counts", "observed"]


def gather_pairs(source: Any, dense: Any, kind: ObservationKind, samples: slice, columns: Any) -> np.ndarray:
    selected = range(*samples.indices(len(source.widths)))
    columns = _columns(columns, source.sites)
    shape = (len(selected), len(columns)) + ((2,) if kind == "counts" else ())
    result: np.ndarray = np.empty(shape, dtype=np.int32 if kind == "counts" else np.bool_)
    if (
        kind == "observed"
        and len(selected) * len(columns) >= PARALLEL_GATHER_CELLS
        and isinstance(dense, np.ndarray)
        and isinstance(source.pairs, np.ndarray)
        and resident_coded(source)
    ):
        _coded_flags(
            *coded_arguments(source, "pairs"),
            readonly(dense, np.bool_),
            readonly(np.asarray(selected), np.int64),
            readonly(columns, np.int64),
            result,
        )
        return result
    for target, sample in enumerate(selected):
        width, row = int(source.widths[sample]), int(source.rows[sample])
        if width:
            codes = getattr(source, {1: "codes8", 2: "codes16", 4: "codes32"}[width])
            indices = _read_rows(codes, slice(row, row + 1), columns)[0]
            first, last = source.offsets[sample : sample + 2]
            if kind == "observed" and isinstance(source.pairs, np.ndarray) and last - first <= len(columns):
                table = source.pairs[first:last]
                flags = (table[:, 0] != 0) | (table[:, 1] != 0)
                result[target] = flags[indices]
            else:
                pairs = table_rows(source.pairs, int(first), int(last), indices)
                result[target] = pairs if kind == "counts" else (pairs[:, 0] != 0) | (pairs[:, 1] != 0)
        else:
            result[target] = _read_rows(dense, slice(row, row + 1), columns)[0]
    return result


@dataclass(frozen=True)
class PairArray:
    """Read-only logical counts or mask, with optional founder exclusions/site view.

    source borrows the GL dictionary/codes. dense owns only fallback sample rows;
    a mask view never retains fallback counts. Founder exclusions affect mask
    gathers only, without changing raw GL/counts or copying N by L flags.
    """

    source: CodedLikelihoods = record_field("evidence", dtype="CodedLikelihoods")
    dense: np.ndarray = record_field("obs", dtype="ndarray")
    kind: ObservationKind = record_field("state", dtype="str")
    excluded: np.ndarray | None = record_field("state", dtype="bool", default=None)
    start: int = record_field("state", dtype="int", default=0)
    stop: int | None = record_field("state", dtype="int | None", default=None)

    def __post_init__(self) -> None:
        freeze_record_arrays(self)

    @property
    def shape(self) -> tuple[int, ...]:
        length = (self.source.sites if self.stop is None else self.stop) - self.start
        return (len(self.source.widths), length) + ((2,) if self.kind == "counts" else ())

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.int32 if self.kind == "counts" else np.bool_)

    @property
    def nbytes(self) -> int:
        return self.dense.nbytes  # dictionary/codes are borrowed from likelihoods

    def gather(self, samples: slice, columns: Any) -> np.ndarray:
        columns = self.start + _columns(columns, self.shape[1])
        result = gather_pairs(self.source, self.dense, self.kind, samples, columns)
        if self.excluded is not None and self.kind == "observed":
            result[self.excluded[samples]] = False
        return result

    def __getitem__(self, selection: Any) -> np.ndarray:
        # Existing sample/site projections stay explicit dense kernel inputs.
        samples, columns = selection if isinstance(selection, tuple) else (selection, slice(None))
        sample_scalar, site_scalar = isinstance(samples, (int, np.integer)), isinstance(columns, (int, np.integer))
        if sample_scalar:
            sample = int(samples)
            if sample < 0:
                sample += self.shape[0]
            if not 0 <= sample < self.shape[0]:
                raise IndexError(sample)
            samples = slice(sample, sample + 1)
        if site_scalar:
            site = int(columns)
            if site < 0:
                site += self.shape[1]
            if not 0 <= site < self.shape[1]:
                raise IndexError(site)
            columns = np.asarray([site], dtype=np.int64)
        result = self.gather(samples, columns)
        if site_scalar:
            result = result[:, 0]
        return result[0] if sample_scalar else result

    def materialise(self) -> np.ndarray:
        return self.gather(slice(None), slice(None))

    def mean(self, axis: int) -> np.ndarray:
        if self.kind != "observed" or axis not in (0, 1):
            raise ValueError("only observation-mask axis means are supported")
        counts = observation_counts(self, axis)
        return counts / self.shape[axis]


def observation_counts(observed: Any, axis: int = 1, *, rows: tuple[int, int] | None = None) -> np.ndarray:
    """Exact int64 positive-depth counts without a whole-contig mask projection.

    ``rows`` (first, stop; axis 1 only) restricts totals to samples first..stop-1.
    Independent sample ranges can be read concurrently."""
    if rows is not None and axis != 1:
        raise ValueError("a sample range applies to per-sample totals (axis 1) only")
    first, stop = (0, observed.shape[0]) if rows is None else (int(rows[0]), int(rows[1]))
    if isinstance(observed, np.ndarray):
        return np.count_nonzero(observed if rows is None else observed[first:stop], axis=axis)
    result = np.zeros(stop - first if axis == 1 else observed.shape[1], dtype=np.int64)
    if (
        isinstance(observed, PairArray)
        and observed.kind == "observed"  # Counts-kind arrays use the row-at-a-time route below.
        and isinstance(observed.dense, np.ndarray)
        and isinstance(observed.source.pairs, np.ndarray)
        and resident_coded(observed.source)
    ):
        # Gather all samples of a site window in one native call for exact integer counts.
        # Target at most 2^24 flags of scratch, allowing one site when N exceeds that target.
        window = max(1, min(65536, (1 << 24) // max(1, stop - first)))
        for left in range(0, observed.shape[1], window):
            right = min(observed.shape[1], left + window)
            block = observed.gather(slice(first, stop), np.arange(left, right, dtype=np.int64))
            if axis == 0:
                result[left:right] = np.count_nonzero(block, axis=0)
            else:
                result += np.count_nonzero(block, axis=1)
        return result
    # Count one contiguous sample row at a time. Checkpoint code arrays are
    # sample-major; cycling through every sample in each small site tile would
    # repeatedly evict/decompress the same large physical chunks.
    for sample in range(first, stop):
        for left in range(0, observed.shape[1], 65536):
            right = min(observed.shape[1], left + 65536)
            block = observed.gather(slice(sample, sample + 1), np.arange(left, right, dtype=np.int64))
            if axis == 0:
                result[left:right] += block[0]
            else:
                result[sample - first] += np.count_nonzero(block)

    return result


def gather_observed(observed: Any, columns: Any, samples: slice = slice(None)) -> np.ndarray:
    if isinstance(observed, np.ndarray):
        return np.ascontiguousarray(observed[samples][:, columns], dtype=np.bool_)
    return observed.gather(samples, columns)


def slice_observed(observed: Any, start: int, stop: int) -> Any:
    if isinstance(observed, np.ndarray):
        return observed[:, start:stop]
    return replace(observed, start=observed.start + start, stop=observed.start + stop)


def pair_views(likelihoods: CodedLikelihoods, counts: np.ndarray) -> tuple[PairArray, PairArray]:
    """Build only fallback count/mask rows; raw coded observations live in pairs."""
    fallback = np.ascontiguousarray(counts[likelihoods.widths == 0], dtype=np.int32)
    observed = (fallback[:, :, 0] != 0) | (fallback[:, :, 1] != 0)
    fallback.flags.writeable = observed.flags.writeable = False
    return PairArray(likelihoods, fallback, "counts"), PairArray(likelihoods, observed, "observed")


def dense_observed(observed: Any) -> np.ndarray:
    return observed if isinstance(observed, np.ndarray) else observed.materialise()
