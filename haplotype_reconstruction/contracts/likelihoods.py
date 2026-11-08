"""Lossless genotype-likelihood storage and explicit dense projections.

A coded sample has its own exact float64 triples keyed by int32 REF/ALT depth.
Widths are grouped so one wider sample never widens another sample's indices.
There is deliberately no implicit ndarray conversion: kernels gather their inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numba import prange, types

from ..core.kernels import kernel
from .roles import freeze_record_arrays, record_field

# A resident coded gather of at least this many sample-site cells runs as one parallel kernel (release 4.7.1); smaller
# ones keep the per-sample loop, whose cost is below a parallel region's start-up.
PARALLEL_GATHER_CELLS = 1 << 14


def _array(dtype: Any, ndim: int, readonly: bool) -> Any:
    return types.Array(dtype, ndim, "C", readonly=readonly, aligned=True)


def _coded_overloads(table: Any, table_ndim: int, dense: Any, dense_ndim: int, out: Any, out_ndim: int) -> tuple:
    """Argument types of a resident coded gather: every input read-only (the wrappers pass read-only views), the
    output writable."""
    return (
        (
            _array(types.uint8, 1, True),
            _array(types.int64, 1, True),
            _array(types.int64, 1, True),
            _array(table, table_ndim, True),
            _array(types.uint8, 2, True),
            _array(types.uint16, 2, True),
            _array(types.uint32, 2, True),
            _array(dense, dense_ndim, True),
            _array(types.int64, 1, True),
            _array(types.int64, 1, True),
            _array(out, out_ndim, False),
        ),
    )


def readonly(array: np.ndarray, dtype: Any) -> np.ndarray:
    """A read-only, C-contiguous view of ``array`` as ``dtype`` (a copy only when the layout or dtype differs), so that
    a kernel sees one argument type whatever its caller's arrays (release 4.7.1)."""
    view = np.ascontiguousarray(array, dtype=dtype).view()
    view.flags.writeable = False
    return view


@kernel(
    overloads=_coded_overloads(types.float64, 2, types.float64, 3, types.float64, 3),
    cache=True,
    parallel=True,
    nogil=True,
)
def _coded_rows(widths, rows, offsets, values, codes8, codes16, codes32, dense, samples, columns, out):
    """out[t, j] = the GL row of sample ``samples[t]`` at site ``columns[j]``: its dictionary row ``values[offsets[s] +
    code]`` for a coded sample (width 1, 2 or 4 bytes per code), its own row of ``dense`` otherwise. A copy, sample-
    parallel; the same values as the per-sample loop of gather_coded."""
    for target in prange(samples.shape[0]):
        sample = samples[target]
        width = widths[sample]
        row = rows[sample]
        base = offsets[sample]
        for index in range(columns.shape[0]):
            column = columns[index]
            if width == 0:
                for genotype in range(3):
                    out[target, index, genotype] = dense[row, column, genotype]
                continue
            if width == 1:
                code = np.int64(codes8[row, column])
            elif width == 2:
                code = np.int64(codes16[row, column])
            else:
                code = np.int64(codes32[row, column])
            for genotype in range(3):
                out[target, index, genotype] = values[base + code, genotype]


@kernel(
    overloads=_coded_overloads(types.int32, 2, types.boolean, 2, types.boolean, 2),
    cache=True,
    parallel=True,
    nogil=True,
)
def _coded_flags(widths, rows, offsets, pairs, codes8, codes16, codes32, dense, samples, columns, out):
    """out[t, j] = sample ``samples[t]`` has a read at site ``columns[j]``: its dictionary pair (REF, ALT depth) by code
    is not (0, 0) for a coded sample, its own row of the dense mask otherwise. Sample-parallel; the same flags as the
    per-sample loop of observations.gather_pairs."""
    for target in prange(samples.shape[0]):
        sample = samples[target]
        width = widths[sample]
        row = rows[sample]
        base = offsets[sample]
        for index in range(columns.shape[0]):
            column = columns[index]
            if width == 0:
                out[target, index] = dense[row, column]
                continue
            if width == 1:
                code = np.int64(codes8[row, column])
            elif width == 2:
                code = np.int64(codes16[row, column])
            else:
                code = np.int64(codes32[row, column])
            out[target, index] = pairs[base + code, 0] != 0 or pairs[base + code, 1] != 0


def resident_coded(source: Any) -> bool:
    """Whether every table, code and dense array of a coded source is an in-memory ndarray."""
    names = ("values", "codes8", "codes16", "codes32", "dense")
    return all(isinstance(getattr(source, name), np.ndarray) for name in names)


def coded_arguments(source: Any, table: str) -> tuple[np.ndarray, ...]:
    """(widths, rows, offsets, ``table`` = values or pairs, codes8, codes16, codes32) of a resident coded source as
    the kernels' read-only arguments."""
    return (
        readonly(source.widths, np.uint8),
        readonly(source.rows, np.int64),
        readonly(source.offsets, np.int64),
        readonly(getattr(source, table), np.float64 if table == "values" else np.int32),
        readonly(source.codes8, np.uint8),
        readonly(source.codes16, np.uint16),
        readonly(source.codes32, np.uint32),
    )


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
    if len(selected) * len(columns) >= PARALLEL_GATHER_CELLS and resident_coded(source) and result.flags.c_contiguous:
        _coded_rows(
            *coded_arguments(source, "values"),
            readonly(source.dense, np.float64),
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
