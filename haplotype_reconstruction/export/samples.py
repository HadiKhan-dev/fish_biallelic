"""Final sample phase of one chromosome as a BCF/VCF.gz with FORMAT GT:PS.

Released calls only: an unknown allele stays `.` (no imputation). Sites inside a supported component are phased
(`|`) with PS = the component's first site position, so phase sets never cross a component boundary; sites
between components are unphased (`/`, PS `.`). REF/ALT are the input VCF's alleles at each position (which must
be a biallelic SNP record), or A/C standing for the abstract alleles 0/1 of a simulation (``synthetic``). In a
full-length file (export.sites) every site the filter removed is `./.` in every sample, with FILTER = the rules.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .sites import SiteTable
from .vcf import FILEFORMAT, LEGACY_FILEFORMAT, contig_line, site_header_lines

_ALLELES = (".", "0", "1")
# Sites formatted at once (genotype codes int64 N x CHUNK_SITES).
CHUNK_SITES = 4096


def _genotype_table(separator: str, phase_set: str) -> list[str]:
    """The 9 genotype strings, indexed by 3 * (first allele + 1) + (second allele + 1)."""
    return [a + separator + b + ":" + phase_set for a in _ALLELES for b in _ALLELES]


def write_sample_phase(path: Path, sites: SiteTable, positions: np.ndarray, calls: np.ndarray,
                       components: np.ndarray, names, *, synthetic: bool, file_format: str,
                       sample_rows: np.ndarray | None = None) -> None:
    """Write one chromosome's phased calls int8 [N, L, 2] (-1 / 0 / 1) at ``positions`` (the kept rows of
    ``sites``) with the rows of ``sites`` to ``path`` (format "bcf" or "vcf").

    ``names`` are the written sample columns in order; ``sample_rows`` (int64 [N], optional) the row of ``calls``
    written in each column (the run's canonical sample order mapped back to the input order, release 2.1), applied
    chunk by chunk (no reordered copy of the calls)."""
    from cyvcf2 import Writer
    positions = np.asarray(positions)
    calls = np.asarray(calls)
    components = np.asarray(components)
    names = tuple(names)
    contig = sites.contig
    if calls.shape != (len(names), len(positions), 2):
        raise ValueError("final phase sample/site axes do not match")
    if sample_rows is not None:
        sample_rows = np.asarray(sample_rows, dtype=np.int64)
        if sorted(sample_rows.tolist()) != list(range(len(names))):
            raise ValueError("sample_rows must be a permutation of the call rows")
    if not np.array_equal(sites.positions[sites.kept], positions):
        raise ValueError(f"{contig}: final phase sites differ from the kept sites of the export")
    length = int(sites.positions[-1]) if synthetic else sites.contig_length
    header = ((FILEFORMAT if sites.full_length else LEGACY_FILEFORMAT) +
              "##source=haplotype-reconstruction\n"
              + contig_line(contig, length) +
              '##FORMAT=<ID=GT,Number=1,Type=String,Description="Released genotype; no imputation">\n'
              '##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Component-local phase set">\n'
              + site_header_lines(sites))
    if synthetic:
        header += '##synthetic_alleles="A/C encode abstract simulated 0/1; not reference-genome bases"\n'
    header += "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(names) + "\n"
    first = {}
    for index, component in enumerate(components.tolist()):
        if component >= 0:
            first.setdefault(int(component), int(positions[index]))
    # Missing alleles remain dots; phase sets do not cross components.
    tables = {component: _genotype_table("|", str(start)) for component, start in first.items()}
    unphased = _genotype_table("/", ".")
    filtered = "\t".join([unphased[0]] * len(names))       # "./.:." in every sample
    filters = sites.filters()
    kept_index = np.cumsum(sites.kept) - 1                  # row -> index among the kept sites
    rows = sites.positions.tolist()
    writer = Writer.from_string(str(path), header, mode="wb" if file_format == "bcf" else "wz")
    try:
        for start in range(0, len(rows), CHUNK_SITES):
            stop = min(start + CHUNK_SITES, len(rows))
            kept = sites.kept[start:stop]
            selected = kept_index[start:stop][kept]
            # Genotype codes 3 * (a + 1) + (b + 1) of the chunk's kept sites, one list per site.
            chunk = calls[:, selected] if sample_rows is None else calls[:, selected][sample_rows]
            codes = (3 * (chunk[:, :, 0].astype(np.int64) + 1)
                     + chunk[:, :, 1].astype(np.int64) + 1).T.tolist()
            site_components = components[selected].tolist()
            k = 0
            for offset, row in enumerate(range(start, stop)):
                if kept[offset]:
                    component = int(site_components[k])
                    table = tables[component] if component >= 0 else unphased
                    genotypes = "\t".join([table[code] for code in codes[k]])
                    k += 1
                else:
                    genotypes = filtered
                line = (f"{contig}\t{rows[row]}\t.\t{sites.ref[row]}\t{sites.alt[row]}\t.\t{filters[row]}\t.\t"
                        "GT:PS\t" + genotypes)
                writer.write_record(writer.variant_from_string(line))
    finally:
        writer.close()
