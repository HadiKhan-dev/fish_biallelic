"""Lossless genotype-likelihood storage and explicit dense projections.

A coded sample has its own exact float64 triples keyed by int32 REF/ALT depth.
Widths are grouped so one wider sample never widens another sample's indices.
There is deliberately no implicit ndarray conversion: kernels gather their inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .roles import freeze_record_arrays, record_field


@dataclass(frozen=True)
class CodedLikelihoods:
    """Mixed per-sample representation; dense rows remain float64 without quantisation.

    widths[N]: 0=dense, 1/2/4=bytes per code; rows[N] selects the corresponding
    dense/codes array. offsets[N+1] bounds each sample's pairs/values table.
    All fields are ordinary numeric arrays, including during worker transport.
    """

    sites: int = record_field("state", dtype="int")
    widths: np.ndarray = record_field("state", axes=("N",), dtype="uint8")
    rows: np.ndarray = record_field("state", axes=("N",), dtype="int64")
    offsets: np.ndarray = record_field("state", axes=("N+1",), dtype="int64")
    pairs: np.ndarray = record_field("obs", axes=("D", "2"), dtype="int32", units="reads")
    values: np.ndarray = record_field("evidence", axes=("D", "3"), dtype="float64", units="probability")
    codes8: np.ndarray = record_field("state", axes=("N8", "L"), dtype="uint8")
    codes16: np.ndarray = record_field("state", axes=("N16", "L"), dtype="uint16")
    codes32: np.ndarray = record_field("state", axes=("N32", "L"), dtype="uint32")
    dense: np.ndarray = record_field("evidence", axes=("Ndense", "L", "3"), dtype="float64", units="probability")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)

    @property
    def shape(self) -> tuple[int, int, int]:
        return len(self.widths), self.sites, 3

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.float64)

    @property
    def nbytes(self) -> int:
        return sum(value.nbytes for value in vars(self).values() if isinstance(value, np.ndarray))

    def gather(self, samples: slice, columns: Any) -> np.ndarray:
        return gather_coded(self, samples, columns)

    def materialise(self, out: np.ndarray | None = None) -> np.ndarray:
        return gather_coded(self, slice(None), slice(None), out=out)


def _columns(columns: Any, length: int) -> np.ndarray:
    if isinstance(columns, slice):
        return np.arange(*columns.indices(length), dtype=np.int64)
    columns = np.asarray(columns)
    return np.flatnonzero(columns) if columns.dtype == np.bool_ else np.asarray(columns, dtype=np.int64)


def _read_rows(array: Any, samples: slice, columns: np.ndarray) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array[samples][:, columns]
    if len(columns) > 1 and np.any(columns[1:] <= columns[:-1]):
        selected, inverse = np.unique(columns, return_inverse=True)
        return array.gather(samples, selected)[:, inverse]
    return array.gather(samples, columns)


class TableProjection(Protocol):
    """Checkpoint dictionaries read only the requested local rows."""

    def gather_rows(self, first: int, last: int, indices: np.ndarray) -> np.ndarray: ...


def table_rows(table: np.ndarray | TableProjection, first: int, last: int, indices: np.ndarray) -> np.ndarray:
    return table[first:last][indices] if isinstance(table, np.ndarray) else table.gather_rows(first, last, indices)


def gather_coded(source: Any, samples: slice, columns: Any, *, out: np.ndarray | None = None) -> np.ndarray:
    """Used by resident sources and evidence-owned checkpoint windows alike."""
    selected = range(*samples.indices(source.shape[0]))
    columns = _columns(columns, source.shape[1])
    shape = (len(selected), len(columns), 3)
    result = np.empty(shape, dtype=np.float64) if out is None else out
    if result.shape != shape or result.dtype != np.float64:
        raise ValueError("GL projection requires a matching float64 destination")
    for target, sample in enumerate(selected):
        width, row = int(source.widths[sample]), int(source.rows[sample])
        if width:
            codes = getattr(source, {1: "codes8", 2: "codes16", 4: "codes32"}[width])
            indices = _read_rows(codes, slice(row, row + 1), columns)[0]
            first, last = source.offsets[sample : sample + 2]
            result[target] = table_rows(source.values, int(first), int(last), indices)
        else:
            result[target] = _read_rows(source.dense, slice(row, row + 1), columns)[0]
    return result


def gather_likelihoods(source: Any, columns: Any, samples: slice = slice(None)) -> np.ndarray:
    """The same ordered float64 projection for dense, coded and window sources."""
    if isinstance(source, np.ndarray):
        return np.ascontiguousarray(source[samples][:, columns], dtype=np.float64)
    return source.gather(samples, columns)


@dataclass(frozen=True)
class LikelihoodSlice:
    """A cheap site interval, kept lazy when discovery declares all contig tasks."""

    source: CodedLikelihoods
    start: int
    stop: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.source.shape[0], self.stop - self.start, 3

    @property
    def nbytes(self) -> int:
        return 0  # borrowed source; no interval payload has been materialised

    def gather(self, samples: slice, columns: Any) -> np.ndarray:
        columns = _columns(columns, self.stop - self.start)
        return self.source.gather(samples, self.start + columns)


def slice_likelihoods(source: Any, start: int, stop: int) -> np.ndarray | LikelihoodSlice:
    if isinstance(source, np.ndarray):
        return source[:, start:stop]
    return LikelihoodSlice(source, start, stop)


# Operational projection allowance, independent of evidence storage admission.
LIKELIHOOD_WORKSPACE_BYTES = 8 * 1024 * 1024


def likelihood_window_sites(source: CodedLikelihoods, maximum: int = 2048) -> int:
    """Bound GL/mask tiles and per-row lookup scratch by an 8 MiB workspace.

    Joint codebooks can save storage while exceeding GL-only dense bytes. Tile
    size therefore depends on the projection's dimensions, never storage savings.
    One site is always processed, even if its sample axis exceeds the allowance;
    this is a workspace target, not a cap on samples, sites or scientific work.
    """
    samples = source.shape[0]
    return max(1, min(maximum, LIKELIHOOD_WORKSPACE_BYTES // (25 * samples + 128)))
