"""Final sample phase of one chromosome as a BCF/VCF.gz with FORMAT GT:PS.

Released calls only: an unknown allele stays `.` (no imputation). Sites inside a supported component are phased
(`|`) with PS = the position of the first site of the sample's phase set, so phase sets never cross a component
boundary; sites between components are unphased (`/`, PS `.`). Release 2.2 (family phase's released ambiguity,
when its record carries it): a sample's phase set also restarts at the first site after each of its phase breaks
(marker boundaries where the displayed phase of the right side relative to the left is not determined by the data),
and a site whose displayed frame is undetermined (an unplaceable switch inside an uninformative stretch) is written
unphased (`/`, PS `.`) without breaking the phase set around it. REF/ALT are the input VCF's alleles at each
position (which must be a biallelic SNP record), or A/C standing for the abstract alleles 0/1 of a simulation
(``synthetic``). In a full-length file (export.sites) every site the filter removed is `./.` in every sample, with
FILTER = the rules.
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


# Bit of family phase's frame_undetermined: the displayed frame at the site is undetermined (release 2.2).
DISPLAYED_UNDETERMINED = 2


def phase_set_starts(components: np.ndarray, breaks) -> list[np.ndarray]:
    """Per call row, the sorted site indices where a phase set starts: every component's first site and the site
    after each of the row's phase breaks (``breaks``: per row int64 marker boundaries i, between sites i and i + 1;
    None = component starts only)."""
    components = np.asarray(components)
    n_sites = len(components)
    starts = np.flatnonzero((components >= 0) & np.r_[True, components[1:] != components[:-1]]).astype(np.int64)
    if breaks is None:
        return None if not len(starts) else [starts]
    result = []
    for row_breaks in breaks:
        extra = np.asarray(row_breaks, dtype=np.int64) + 1
        extra = extra[(extra > 0) & (extra < n_sites)]
        result.append(np.unique(np.r_[starts, extra]))
    return result


def write_sample_phase(path: Path, sites: SiteTable, positions: np.ndarray, calls: np.ndarray,
                       components: np.ndarray, names, *, synthetic: bool, file_format: str,
                       sample_rows: np.ndarray | None = None, phase_breaks=None,
                       frame_undetermined: np.ndarray | None = None) -> None:
    """Write one chromosome's phased calls int8 [N, L, 2] (-1 / 0 / 1) at ``positions`` (the kept rows of
    ``sites``) with the rows of ``sites`` to ``path`` (format "bcf" or "vcf").

    ``names`` are the written sample columns in order; ``sample_rows`` (int64 [N], optional) the row of ``calls``
    written in each column (the run's canonical sample order mapped back to the input order, release 2.1), applied
    chunk by chunk (no reordered copy of the calls). ``phase_breaks`` (per call row, int64 marker boundaries) and
    ``frame_undetermined`` (int8 [N, L], bit DISPLAYED_UNDETERMINED) are family phase's released phase ambiguity
    (release 2.2; None = none)."""
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
              '##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Phase set: the position of its first site; it '
              'restarts at every component boundary and after each of the sample\'s undetermined phase boundaries">\n'
              + site_header_lines(sites))
    if synthetic:
        header += '##synthetic_alleles="A/C encode abstract simulated 0/1; not reference-genome bases"\n'
    header += "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(names) + "\n"
    # Phase sets per call row: start indices (component starts, plus the site after each phase break); PS = the
    # position of the set's first site. Missing alleles remain dots; phase sets never cross components.
    if phase_breaks is not None and len(phase_breaks) != len(names):
        raise ValueError("phase breaks need one entry per sample")
    if frame_undetermined is not None and np.asarray(frame_undetermined).shape != (len(names), len(positions)):
        raise ValueError("frame_undetermined sample/site axes do not match")
    set_starts = phase_set_starts(components, phase_breaks)
    tables: dict[int, list[str]] = {}
    unphased = _genotype_table("/", ".")

    def table_of(phase_set: int) -> list[str]:
        if phase_set < 0:
            return unphased
        found = tables.get(phase_set)
        if found is None:
            found = tables[phase_set] = _genotype_table("|", str(phase_set))
        return found
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
            # PS per call row and selected site (-1 = unphased), then in the written column order.
            phase_sets = np.full((calls.shape[0], len(selected)), -1, dtype=np.int64)
            if set_starts is not None and len(selected):
                for row_index in range(calls.shape[0]):
                    row_starts = set_starts[row_index if len(set_starts) > 1 else 0]
                    block = np.searchsorted(row_starts, selected, side="right") - 1
                    inside = block >= 0
                    phase_sets[row_index, inside] = positions[row_starts[block[inside]]]
            phase_sets[:, components[selected] < 0] = -1
            if frame_undetermined is not None and len(selected):
                phase_sets[(np.asarray(frame_undetermined)[:, selected] & DISPLAYED_UNDETERMINED) != 0] = -1
            if sample_rows is not None:
                phase_sets = phase_sets[sample_rows]
            changed = np.ones(len(selected), dtype=np.bool_)
            changed[1:] = np.any(phase_sets[:, 1:] != phase_sets[:, :-1], axis=0)
            phase_sets = phase_sets.T.tolist()
            column_tables = None
            k = 0
            for offset, row in enumerate(range(start, stop)):
                if kept[offset]:
                    if changed[k] or column_tables is None:
                        column_tables = [table_of(value) for value in phase_sets[k]]
                    genotypes = "\t".join([table[code] for table, code in zip(column_tables, codes[k])])
                    k += 1
                else:
                    genotypes = filtered
                line = (f"{contig}\t{rows[row]}\t.\t{sites.ref[row]}\t{sites.alt[row]}\t.\t{filters[row]}\t.\t"
                        "GT:PS\t" + genotypes)
                writer.write_record(writer.variant_from_string(line))
    finally:
        writer.close()
