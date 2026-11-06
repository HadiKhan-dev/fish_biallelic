"""Choose exact joint evidence storage from actual byte costs, without a setting.

The existing dense GL kernels also fill bounded sample/site projections. Packing
copies their bits. A repeated depth pair with different GL bits forces dense.
"""

from __future__ import annotations

import numpy as np

from ...contracts.likelihoods import CodedLikelihoods
from ...core.immutable import declare_immutable
from .likelihoods import _ROW_SUM_TOLERANCE, _invalid_likelihood_cells, likelihoods_by_sample, likelihoods_into

# Storage policy only. GL + two int32 depths + bool mask cost 33 bytes/cell.
ROW_METADATA_BYTES = 17
CHECK_SITES = 8192


def index_width(distinct):
    """Check D before casting codes 0..D-1; no depth limit or wraparound."""
    for width in (1, 2, 4):
        if distinct <= 1 << (8 * width):
            return width
    return 0


def storage_width(distinct, sites):
    """Code only when exact triples/pairs, codes and maps beat joint dense bytes."""
    width = index_width(distinct)
    return width if width and 32 * distinct + width * sites + ROW_METADATA_BYTES < 33 * sites else 0


def _keys(counts):
    """One unsigned row; in-place conversion avoids signed/unsigned ufunc promotion."""
    keys = np.empty(len(counts), np.uint64)
    np.copyto(keys, counts[:, 0], casting="unsafe")
    keys <<= np.uint64(32)
    np.bitwise_or(keys, counts[:, 1], out=keys, casting="unsafe", dtype=np.uint64)
    return keys


def _sample_plan(counts, size):
    """One sample's sorted pair keys and exact first-site indices, never N plans."""
    keys = _keys(counts)
    keys.sort()
    keep = np.empty(len(keys), np.bool_)
    if len(keys):
        keep[0] = True
        keep[1:] = keys[1:] != keys[:-1]
    table_keys = keys[keep]
    del keys, keep
    first = np.full(len(table_keys), len(counts), np.int64)
    for left in range(0, len(counts), size):
        right = min(len(counts), left + size)
        indices = np.searchsorted(table_keys, _keys(counts[left:right]))
        np.minimum.at(first, indices, np.arange(left, right, dtype=np.int64))
    return table_keys, first


def encode_likelihoods(counts, fill, *, statistics=None, dense_result=None):
    """Losslessly encode dense projections supplied by fill(samples, out, columns).

    Choose each sample before allocating final storage. Retain only cardinalities
    and widths across samples, then rebuild one sample plan at a time while filling
    final arrays. No N-sample plans or duplicate float64 tables survive allocation.
    The exact-value safety check uses bounded site windows. Dense fallback uses
    the original output, after planning scratch dies.
    """
    n, sites = counts.shape[:2]
    widths = np.zeros(n, np.uint8)
    sizes = np.zeros(n, np.int64)
    nonconstant = []
    size = CHECK_SITES
    for sample in range(n):
        keys = _keys(counts[sample])
        keys.sort()
        keep = np.empty(sites, np.bool_)
        if sites:
            keep[0] = True
            keep[1:] = keys[1:] != keys[:-1]
        distinct = int(np.count_nonzero(keep))
        sizes[sample] = distinct
        width = storage_width(distinct, sites)
        if not width:
            del keys, keep
            continue
        del keys, keep
        table_keys, first = _sample_plan(counts[sample], size)
        table = np.empty((1, distinct, 3), np.float64)
        fill(slice(sample, sample + 1), table, first)
        consistent = True
        for left in range(0, sites, size):
            right = min(sites, left + size)
            truth = np.empty((1, right - left, 3), np.float64)
            fill(slice(sample, sample + 1), truth, slice(left, right))
            indices = np.searchsorted(table_keys, _keys(counts[sample, left:right]))
            if not np.array_equal(table[0, indices].view(np.uint64), truth[0].view(np.uint64)):
                consistent = False
                break
        del table, truth, indices
        if consistent:
            widths[sample] = width
        else:
            nonconstant.append(sample)
        del table_keys, first
    # A mixed container's maps also describe dense rows. Do not keep a container
    # whose overhead would outweigh its total saving, even if one row could code.
    encoded = int(np.sum(np.where(widths, 32 * sizes + widths.astype(np.int64) * sites, 33 * sites)))
    if widths.any() and encoded + 17 * n + 8 >= 33 * n * sites:
        widths.fill(0)
    if statistics is not None:
        statistics.update(distinct_pairs=sizes.tolist(), index_bytes=widths.tolist(), nonconstant_fallback=nonconstant)
    if not widths.any():
        if dense_result is not None:
            return dense_result()
        out = np.empty((n, sites, 3), np.float64)
        fill(slice(None), out, slice(None))
        declare_immutable(out)
        return out
    rows = np.empty(n, np.int64)
    offsets = np.zeros(n + 1, np.int64)
    offsets[1:] = np.cumsum(np.where(widths, sizes, 0))
    arrays = {
        name: np.empty((int(np.count_nonzero(widths == width)), sites), dtype)
        for width, name, dtype in ((1, "codes8", np.uint8), (2, "codes16", np.uint16), (4, "codes32", np.uint32))
    }
    arrays.update(
        widths=widths,
        rows=rows,
        offsets=offsets,
        pairs=np.empty((int(offsets[-1]), 2), np.int32),
        values=np.empty((int(offsets[-1]), 3), np.float64),
        dense=np.empty((int(np.count_nonzero(widths == 0)), sites, 3), np.float64),
    )
    next_row = {width: 0 for width in (0, 1, 2, 4)}
    for sample in range(n):
        width = int(widths[sample])
        row = next_row[width]
        rows[sample] = row
        next_row[width] += 1
        if width:
            keys, first = _sample_plan(counts[sample], size)
            begin, end = offsets[sample : sample + 2]
            arrays["pairs"][begin:end] = counts[sample, first]
            fill(slice(sample, sample + 1), arrays["values"][begin:end].reshape(1, -1, 3), first)
            target = arrays[{1: "codes8", 2: "codes16", 4: "codes32"}[width]][row]
            for left in range(0, sites, size):
                right = min(sites, left + size)
                target[left:right] = np.searchsorted(keys, _keys(counts[sample, left:right]))
            del keys, first, target
        else:
            fill(slice(sample, sample + 1), arrays["dense"][row : row + 1], slice(None))
    for array in arrays.values():
        declare_immutable(array)
    return CodedLikelihoods(sites, **arrays)


def build_likelihoods(counts, error, heterozygote, het_inverse, lookup_depth, *, statistics=None):
    """Use the existing lookup/direct arithmetic on explicit dense projections.

    The table's first-occurrence sites include every depth pair, hence the same
    maximum depth and prefix/lookup bounds as the original sample. Safety windows
    reproduce original site values independently before a row is admitted.
    """

    def fill(samples, out, columns):
        selected = np.ascontiguousarray(counts[samples][:, columns])
        likelihoods_into(selected, error[samples], heterozygote[samples], het_inverse, lookup_depth, out)
        if _invalid_likelihood_cells(out, selected, _ROW_SUM_TOLERANCE):
            raise ValueError("invalid genotype likelihoods")

    def dense_result():
        # Keep 4.3's original Numba output allocator as well as its arithmetic.
        # This is the same dense path at high depth, not a second GL allocation.
        out = likelihoods_by_sample(counts, error, heterozygote, het_inverse, lookup_depth)
        if _invalid_likelihood_cells(out, counts, _ROW_SUM_TOLERANCE):
            raise ValueError("invalid genotype likelihoods")
        out.flags.writeable = False
        return out

    return encode_likelihoods(counts, fill, statistics=statistics, dense_result=dense_result)
