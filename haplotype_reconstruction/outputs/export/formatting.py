"""Release 4.10: native text of the sample-phase VCF and its BGZF block boundaries.

samples.write_sample_phase built every cell as a Python string and joined each record's cells: on chr3 of the N = 320
simulation at 0.5x (1.5 million records) that formatting was most of the export, on one CPU. ``sample_lines`` writes a
chunk of records as bytes with the same text: each record's prefix (CHROM .. FORMAT, formatted by the caller), then one
GT:PS cell per sample, tab-separated. A cell is the genotype of allele codes (".", "0", "1"): phased "a|b:PS" with the
set's first position ("." beyond int32, as htslib writes it), unphased (no set) "lo/hi:." with the alleles in
ascending text order, missing first, as the previous genotype tables; a removed (filtered) record is "./.:." in
every sample. ``block_cuts`` gives the positions where htslib's record writer starts a new BGZF block (vcf_write's
bgzf_flush_try), so the compressed file is the same.
"""

from __future__ import annotations

from numba import types

from ...core.kernels import kernel

_BYTES = types.Array(types.uint8, 1, "C", False, aligned=True)
_PREFIX = types.Array(types.uint8, 1, "C", True, aligned=True)  # np.frombuffer of the joined prefixes: read-only
_INDEX = types.Array(types.int64, 1, "C", False, aligned=True)
_FLAGS = types.Array(types.boolean, 1, "C", False, aligned=True)
_TABLE = types.Array(types.int64, 2, "C", False, aligned=True)

_DOT, _ZERO, _ONE, _TAB, _NEWLINE, _COLON, _SLASH, _BAR = 46, 48, 49, 9, 10, 58, 47, 124
_INT32_MAX = 2147483647  # largest FORMAT integer htslib keeps (BCF_MAX_BT_INT32)


@kernel(overloads=((_PREFIX, _INDEX, _FLAGS, _TABLE, _TABLE, _BYTES, _INDEX),), cache=True, nogil=True)
def sample_lines(prefix, prefix_ends, kept, codes, phase_sets, out, line_ends):
    """Write the chunk's records into ``out`` (large enough; see sample_line_bytes) and their end offsets into
    ``line_ends``; return the bytes written. prefix uint8 (read-only): the records' prefixes concatenated, prefix_ends
    int64 [R] their end offsets; kept bool [R] (writable: the caller's own copy); codes, phase_sets int64
    [N, kept records]: 3 * (a + 1) + (b + 1) of each cell's allele codes a, b in {-1, 0, 1} and its phase set's first
    position (-1: unphased), in written column order."""
    samples = codes.shape[0]
    position = 0
    start = 0
    column = 0
    for row in range(prefix_ends.shape[0]):
        stop = prefix_ends[row]
        for i in range(start, stop):
            out[position] = prefix[i]
            position += 1
        start = stop
        for n in range(samples):
            if n:
                out[position] = _TAB
                position += 1
            first = 0
            second = 0
            phase_set = -1
            if kept[row]:
                code = codes[n, column]
                first = code // 3
                second = code - 3 * first
                phase_set = phase_sets[n, column]
            if phase_set < 0:
                low = min(first, second)
                high = max(first, second)
                out[position] = _DOT if low == 0 else (_ZERO if low == 1 else _ONE)
                out[position + 1] = _SLASH
                out[position + 2] = _DOT if high == 0 else (_ZERO if high == 1 else _ONE)
                out[position + 3] = _COLON
                out[position + 4] = _DOT
                position += 5
            else:
                out[position] = _DOT if first == 0 else (_ZERO if first == 1 else _ONE)
                out[position + 1] = _BAR
                out[position + 2] = _DOT if second == 0 else (_ZERO if second == 1 else _ONE)
                out[position + 3] = _COLON
                position += 4
                if phase_set > _INT32_MAX:
                    # htslib's VCF writer (4.9.0, and the BCF route) stores FORMAT integers as int32: a larger PS is
                    # written missing, the genotype staying phased.
                    out[position] = _DOT
                    position += 1
                    continue
                digits = 1
                scale = 10
                while phase_set >= scale:
                    digits += 1
                    scale *= 10
                value = phase_set
                for d in range(digits - 1, -1, -1):
                    out[position + d] = _ZERO + value % 10
                    value //= 10
                position += digits
        if kept[row]:
            column += 1
        out[position] = _NEWLINE
        position += 1
        line_ends[row] = position
    return position


def sample_line_bytes(prefix_bytes: int, records: int, samples: int) -> int:
    """An upper bound of sample_lines' output: the prefixes, then per record and sample a cell of at most 5 + 19
    characters (an int64 phase set) and its separator, and a newline."""
    return int(prefix_bytes) + int(records) * (int(samples) * 25 + 1)


@kernel(overloads=((_INDEX, types.int64, types.int64, _INDEX),), cache=True, nogil=True)
def block_cuts(line_ends, offset, block, cuts):
    """The record starts (offsets into the chunk) before which htslib's writer flushes the current BGZF block, into
    ``cuts``; returns (their count, the block offset after the chunk). ``offset``: bytes already in the current block;
    ``block``: BGZF_BLOCK_SIZE. A record that does not fit in the rest of a non-empty block starts a new one
    (bgzf_flush_try); a write that fills a block flushes it (bgzf_write)."""
    count = 0
    previous = 0
    current = offset
    for row in range(line_ends.shape[0]):
        size = line_ends[row] - previous
        if current > 0 and current + size > block:
            cuts[count] = previous
            count += 1
            current = 0
        current += size
        if current >= block:
            current = current % block
        previous = line_ends[row]
    return count, current


__all__ = ["block_cuts", "sample_line_bytes", "sample_lines"]
