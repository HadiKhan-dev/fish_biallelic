"""Pieces shared by the VCF writers: header lines, the htslib index and atomic placement of a file and its index."""
from __future__ import annotations

import ctypes
from pathlib import Path

from .sites import COMMON_ALLELE_RULE, RULE_DESCRIPTIONS, SiteTable

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
    lines.append(f'##site_set="full length: every input biallelic SNP that reached the site-quality filter; FILTER '
                 f'PASS = kept, else the applied rules that removed the site; site filter on this contig: '
                 f'{sites.site_filter}"\n')
    dropped = ", ".join(f"{count} {reason}" for reason, count in sites.dropped.items()) or "none"
    lines.append(f'##omitted_records="records the loader drops never reach the site filter and are not written: '
                 f'non-biallelic records (multiallelic, not a SNP) and biallelic SNPs sharing a position '
                 f'({dropped})"\n')
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


def index_path(path: Path, file_format: str, last_position: int) -> Path:
    """The index written next to ``path``: CSI for BCF and for VCF beyond TBI_LIMIT, else tabix (TBI)."""
    csi = file_format == "bcf" or last_position >= TBI_LIMIT
    return Path(str(path) + (".csi" if csi else ".tbi"))


def build_index(data: Path, index: Path, *, csi: bool) -> None:
    """Write the CSI (``csi``) or tabix index of the BGZF-compressed ``data`` (BCF: CSI only) to ``index``."""
    status = _builder()(str(data).encode(), str(index).encode(), _CSI_MIN_SHIFT if csi else 0, 0)
    if status != 0:
        raise RuntimeError(f"htslib could not index {data} (bcf_index_build3 returned {status})")


def place(temporary: Path, final: Path, *, file_format: str, last_position: int, index: bool) -> list[str]:
    """Index ``temporary`` when asked, then move it (and its index) to ``final``; returns the file names placed."""
    names = [final.name]
    if index:
        target = index_path(final, file_format, last_position)
        staged = target.with_name(f".{target.name}.tmp")
        build_index(temporary, staged, csi=target.suffix == ".csi")
        for stale in (Path(str(final) + ".csi"), Path(str(final) + ".tbi")):
            stale.unlink(missing_ok=True)       # never leave an index of earlier contents beside the new file
        temporary.replace(final)
        staged.replace(target)
        names.append(target.name)
    else:
        temporary.replace(final)
    return names
