"""Pieces shared by the VCF writers: header lines, the htslib index and atomic placement of a file and its index."""

from __future__ import annotations

import ctypes
from pathlib import Path

from ...core.files import atomic_path
from .records import COMMON_ALLELE_RULE, RULE_DESCRIPTIONS, SiteTable

# Tabix (TBI) indexes address positions below 2^29; longer contigs get a CSI index.
TBI_LIMIT = 1 << 29
_CSI_MIN_SHIFT = 14


def suffix(file_format: str) -> str:
    return ".bcf" if file_format == "bcf" else ".vcf.gz"


# htslib's writer header already declares VCFv4.2: a header text declaring another version keeps that line as a
# second fileformat line (as in kept-site sample files, unchanged for byte identity with earlier exports).
FILEFORMAT = "##fileformat=VCFv4.2\n"
LEGACY_FILEFORMAT = "##fileformat=VCFv4.3\n"


def contig_line(contig: str, length) -> str:
    return f"##contig=<ID={contig}" + (f",length={length}" if length and length > 0 else "") + ">\n"


def site_header_lines(sites: SiteTable) -> str:
    """FILTER definitions and the site-set notes of a full-length file (nothing for a kept-site file)."""
    if not sites.full_length:
        return ""
    lines = [f'##FILTER=<ID={rule},Description="{text}">\n' for rule, text in RULE_DESCRIPTIONS.items()]
    lines.append(
        f'##site_set="full length: every input biallelic SNP that reached the site-quality filter; FILTER '
        f"PASS = kept, else the applied rules that removed the site; site filter on this contig: "
        f'{sites.site_filter}"\n'
    )
    dropped = ", ".join(f"{count} {reason}" for reason, count in sites.dropped.items()) or "none"
    lines.append(
        f'##omitted_records="records the loader drops never reach the site filter and are not written: '
        f"non-biallelic records (multiallelic, not a SNP) and biallelic SNPs sharing a position "
        f'({dropped})"\n'
    )
    return "".join(lines)


def common_allele_line() -> str:
    return f'##common_allele="{COMMON_ALLELE_RULE}"\n'


def _builder():
    """htslib's bcf_index_build3, from the htslib linked into cyvcf2 (a dependency; no extra package)."""
    from cyvcf2 import cyvcf2 as extension

    function = ctypes.CDLL(extension.__file__).bcf_index_build3
    function.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int)
    function.restype = ctypes.c_int
    return function


_BGZF_BLOCK = 0xFF00  # htslib's BGZF_BLOCK_SIZE: uncompressed bytes per block


def _bgzf():
    """htslib's BGZF writer functions, from the htslib linked into cyvcf2 (as the index builder)."""
    from cyvcf2 import cyvcf2 as extension

    library = ctypes.CDLL(extension.__file__)
    library.bgzf_open.restype = ctypes.c_void_p
    library.bgzf_open.argtypes = (ctypes.c_char_p, ctypes.c_char_p)
    library.bgzf_write.restype = ctypes.c_ssize_t
    library.bgzf_write.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
    library.bgzf_flush.restype = ctypes.c_int
    library.bgzf_flush.argtypes = (ctypes.c_void_p,)
    library.bgzf_close.restype = ctypes.c_int
    library.bgzf_close.argtypes = (ctypes.c_void_p,)
    library.bgzf_mt.restype = ctypes.c_int
    library.bgzf_mt.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
    library.bgzf_flush_try.restype = ctypes.c_int
    library.bgzf_flush_try.argtypes = (ctypes.c_void_p, ctypes.c_ssize_t)
    return library


# BGZF blocks per job of htslib's compression thread pool (its own default in bgzip and samtools).
_BGZF_SUB_BLOCKS = 256


def htslib_header(path: Path, header: str) -> str:
    """The header text htslib writes for ``header`` (it normalises a header it parses), from a header-only write of
    an uncompressed VCF next to ``path``."""
    import os

    from cyvcf2 import Writer

    scratch = path.with_name(f".{path.name}.{os.getpid()}.header.vcf")
    try:
        writer = Writer.from_string(str(scratch), header, mode="w")
        writer.close()
        return scratch.read_text()
    finally:
        scratch.unlink(missing_ok=True)


class RecordWriter:
    """Write VCF/BCF records given as text lines (no trailing newline) to ``path``.

    BCF: cyvcf2's record writer, each line parsed into a record. VCF (release 4.10): htslib's own header text, then
    the lines as text through htslib's bgzf_write, with the BGZF blocks htslib's record writer makes: its header
    write ends with a flush, and before each record it starts a new block when the record does not fit in the current
    one (vcf_write's bgzf_flush_try). The same htslib compresses the same blocks, so the file, and its index, are the
    same bytes. The record writer parsed every line back into a record and formatted it again, which took most of a
    large export (chr3 at 0.5x: 84 of 139 s). Pending text is at most one block (or one longer record).
    ``threads``: the CPUs this writer may use; above one, the caller's thread formats and writes while threads - 1
    htslib workers compress the VCF's blocks (bgzf_mt; each block is deflated alone, so the bytes are the same; chr3's
    samples file: 21 s of single-thread compression).
    """

    def __init__(self, path: Path, header: str, file_format: str, threads: int = 1):
        self.path = Path(path)
        self.pending: list[str] = []
        self.size = 0  # pending bytes
        self.offset = 0  # bytes in htslib's current block
        self.handle = None
        self.writer = None
        if file_format == "bcf":
            from cyvcf2 import Writer

            self.writer = Writer.from_string(str(self.path), header, mode="wb")
            return
        self.library = _bgzf()
        text = htslib_header(self.path, header)
        self.handle = self.library.bgzf_open(str(self.path).encode(), b"wz")
        if not self.handle:
            raise OSError(f"htslib could not open {self.path} for writing")
        try:
            if threads > 1 and self.library.bgzf_mt(self.handle, int(threads) - 1, _BGZF_SUB_BLOCKS) != 0:
                raise OSError(f"htslib could not start {threads - 1} compression threads for {self.path}")
            self._write(text.encode())
            self._flush_block()
        except BaseException:
            handle, self.handle = self.handle, None
            self.library.bgzf_close(handle)  # the header write failed: release the handle; its error is the one raised
            raise

    def _write(self, data) -> None:
        """bytes or a contiguous uint8 array."""
        import numpy as np

        buffer = np.frombuffer(data, dtype=np.uint8) if isinstance(data, (bytes, bytearray)) else data
        if len(buffer) and self.library.bgzf_write(self.handle, buffer.ctypes.data, len(buffer)) != len(buffer):
            raise OSError(f"htslib could not write {self.path}")

    def _flush_block(self) -> None:
        """End the current block (after the header: htslib's header write ends with a flush)."""
        if self.library.bgzf_flush(self.handle) != 0:
            raise OSError(f"htslib could not write {self.path}")
        self.offset = 0

    def _start_block(self) -> None:
        """Start a new block before a record that does not fit in the current, non-empty one, as vcf_write's
        bgzf_flush_try does: with compression threads it queues the block (bgzf_flush would wait for every queued
        block), otherwise it is bgzf_flush. Asking room for a whole block flushes exactly a non-empty block."""
        if self.library.bgzf_flush_try(self.handle, _BGZF_BLOCK) != 0:
            raise OSError(f"htslib could not write {self.path}")
        self.offset = 0

    def _emit(self) -> None:
        if self.pending:
            self._write(("\n".join(self.pending) + "\n").encode())
            self.offset = (self.offset + self.size) % _BGZF_BLOCK  # bgzf_write flushes each block it fills
            self.pending, self.size = [], 0

    def write(self, line: str) -> None:
        if self.writer is not None:
            self.writer.write_record(self.writer.variant_from_string(line))
            return
        n = (len(line) if line.isascii() else len(line.encode())) + 1
        if self.offset + self.size + n > _BGZF_BLOCK and self.offset + self.size > 0:
            self._emit()
            self._start_block()
        self.pending.append(line)
        self.size += n
        if self.offset + self.size >= _BGZF_BLOCK:  # a record that fills a block alone
            self._emit()

    def write_chunk(self, data, line_ends) -> None:
        """Records as bytes: ``data`` uint8 (contiguous), record i ending with its newline at ``line_ends[i]``
        (int64), written with htslib's record-writer blocks (formatting.block_cuts). BCF: parsed record by record."""
        import numpy as np

        from .formatting import block_cuts

        if self.writer is not None:
            for line in bytes(data).decode().split("\n")[:-1]:
                self.write(line)
            return
        self._emit()
        cuts = np.empty(len(line_ends), dtype=np.int64)
        count, current = block_cuts(np.ascontiguousarray(line_ends, dtype=np.int64), self.offset, _BGZF_BLOCK, cuts)
        start = 0
        for cut in cuts[:count].tolist():
            self._write(data[start:cut])
            self._start_block()
            start = cut
        self._write(data[start:])
        self.offset = int(current)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            return
        if self.handle is None:
            return
        try:
            self._emit()
        finally:
            handle, self.handle = self.handle, None
            if self.library.bgzf_close(handle) != 0:
                raise OSError(f"htslib could not close {self.path}")


def index_path(path: Path, file_format: str, last_position: int) -> Path:
    """The index written next to ``path``: CSI for BCF and for VCF beyond TBI_LIMIT, else tabix (TBI)."""
    return Path(str(path) + "." + index_kind(file_format, last_position))


def build_index(data: Path, index: Path, *, csi: bool, threads: int = 1) -> None:
    """Write the CSI (``csi``) or tabix index of the BGZF-compressed ``data`` (BCF: CSI only) to ``index``;
    ``threads`` (the CPUs it may use) > 1: the caller indexes while threads - 1 workers decompress (the same index)."""
    status = _builder()(
        str(data).encode(), str(index).encode(), _CSI_MIN_SHIFT if csi else 0, int(threads) - 1 if threads > 1 else 0
    )
    if status != 0:
        raise RuntimeError(f"htslib could not index {data} (bcf_index_build3 returned {status})")


def place(
    temporary: Path, final: Path, *, file_format: str, last_position: int, index: bool, threads: int = 1
) -> list[str]:
    """Index ``temporary`` when asked (on ``threads``), then move it (and its index) to ``final``; returns the file
    names placed."""
    names = [final.name]
    # Own a caller's staged data even when indexing or placement fails.
    try:
        if index:
            target = index_path(final, file_format, last_position)
            with atomic_path(target) as staged:
                build_index(temporary, staged, csi=target.suffix == ".csi", threads=threads)
                for stale in (Path(str(final) + ".csi"), Path(str(final) + ".tbi")):
                    stale.unlink(missing_ok=True)
                temporary.replace(final)
            names.append(target.name)
        else:
            temporary.replace(final)
    finally:
        temporary.unlink(missing_ok=True)
    return names


def index_kind(file_format: str, last_position: int) -> str:
    """Choose CSI for BCF or last written position at/above the TBI address limit."""
    return "csi" if file_format == "bcf" or last_position >= TBI_LIMIT else "tbi"
