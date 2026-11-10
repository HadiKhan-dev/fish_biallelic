"""Final sample phase of one chromosome as a BCF/VCF.gz with FORMAT GT:PS.

Released calls only: an unknown allele stays `.` (no imputation). Sites inside a supported component are phased
(`|`) with PS = the position of the first site of the sample's phase set, so phase sets never cross a component
boundary; sites between components are unphased (`/`, PS `.`). With family phase's released ambiguity,
a sample's phase set also restarts at the first site after each of its phase breaks
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

from ...contracts.phase import DISPLAYED_UNDETERMINED
from ...core.errors import InvariantError
from ...core.release import CURRENT_RELEASE
from .formatting import sample_line_bytes, sample_lines
from .records import SiteTable
from .vcf import FILEFORMAT, RecordWriter, contig_line, site_header_lines

# Sites formatted at once: CHUNK_SITES, fewer with many samples so that a chunk holds at most CHUNK_CELLS cells (its
# int64 codes, phase sets and their unique-inverse, and its cell texts are each 8 bytes per cell: 8 MiB at the limit).
CHUNK_SITES = 4096
CHUNK_CELLS = 1 << 20


def phase_set_starts(components: np.ndarray, breaks) -> list[np.ndarray] | None:
    """Per call row, the sorted site indices where a phase set starts: every component's first site and the site
    after each of the row's phase breaks (``breaks``: per row int64 marker boundaries i, between sites i and i + 1;
    None = component starts only)."""
    components = np.asarray(components)
    n_sites = len(components)
    starts: np.ndarray = np.flatnonzero((components >= 0) & np.r_[True, components[1:] != components[:-1]]).astype(
        np.int64
    )
    if breaks is None:
        return None if not len(starts) else [starts]
    result = []
    for row_breaks in breaks:
        extra = np.asarray(row_breaks, dtype=np.int64) + 1
        extra = extra[(extra > 0) & (extra < n_sites)]
        result.append(np.unique(np.r_[starts, extra]))
    return result


def write_sample_phase(
    path: Path,
    sites: SiteTable,
    positions: np.ndarray,
    calls: np.ndarray,
    components: np.ndarray,
    names,
    *,
    synthetic: bool,
    file_format: str,
    sample_rows: np.ndarray | None = None,
    phase_breaks=None,
    frame_undetermined: np.ndarray | None = None,
    threads: int = 1,
) -> None:
    """Write one chromosome's phased calls int8 [N, L, 2] (-1 / 0 / 1) at ``positions`` (the kept rows of
    ``sites``) with the rows of ``sites`` to ``path`` (format "bcf" or "vcf").

    ``names`` are the written sample columns in order; ``sample_rows`` (int64 [N], optional) the row of ``calls``
    written in each column (the run's canonical sample order mapped back to the input order), applied
    chunk by chunk (no reordered copy of the calls). ``phase_breaks`` (per call row, int64 marker boundaries) and
    ``frame_undetermined`` (int8 [N, L], bit DISPLAYED_UNDETERMINED) are family phase's released phase ambiguity
    (None = none)."""
    positions = np.asarray(positions)
    calls = np.asarray(calls)
    components = np.asarray(components)
    names = tuple(names)
    contig = sites.contig
    if calls.shape != (len(names), len(positions), 2):
        raise InvariantError("final phase sample/site axes do not match")
    if sample_rows is not None:
        sample_rows = np.asarray(sample_rows, dtype=np.int64)
        if sorted(sample_rows.tolist()) != list(range(len(names))):
            raise InvariantError("sample_rows must be a permutation of the call rows")
    if not np.array_equal(sites.positions[sites.kept], positions):
        raise InvariantError(f"{contig}: final phase sites differ from the kept sites of the export")
    length = int(sites.positions[-1]) if synthetic else sites.contig_length
    header = (
        FILEFORMAT
        + "##source=haplotype-reconstruction\n"
        + f"##haplotype_reconstruction_release={CURRENT_RELEASE}\n"
        + contig_line(contig, length)
        + '##FORMAT=<ID=GT,Number=1,Type=String,Description="Released genotype; no imputation">\n'
        '##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Phase set: the position of its first site; it '
        "restarts at every component boundary and after each of the sample's undetermined phase boundaries\">\n"
        + site_header_lines(sites)
    )
    if synthetic:
        header += '##synthetic_alleles="A/C encode abstract simulated 0/1; not reference-genome bases"\n'
    header += "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(names) + "\n"
    # Phase sets per call row: start indices (component starts, plus the site after each phase break); PS = the
    # position of the set's first site. Missing alleles remain dots; phase sets never cross components.
    if phase_breaks is not None and len(phase_breaks) != len(names):
        raise InvariantError("phase breaks need one entry per sample")
    if frame_undetermined is not None and np.asarray(frame_undetermined).shape != (len(names), len(positions)):
        raise InvariantError("frame_undetermined sample/site axes do not match")
    set_starts = phase_set_starts(components, phase_breaks)
    filters = sites.filters()
    kept_index: np.ndarray = np.cumsum(sites.kept) - 1  # row -> index among the kept sites
    rows, refs, alts = sites.positions.tolist(), sites.ref.tolist(), sites.alt.tolist()
    writer = RecordWriter(path, header, file_format, threads=threads)
    try:
        step = max(1, min(CHUNK_SITES, CHUNK_CELLS // max(1, len(names))))
        for start in range(0, len(rows), step):
            stop = min(start + step, len(rows))
            kept = sites.kept[start:stop]
            selected = kept_index[start:stop][kept]
            # Genotype codes 3 * (a + 1) + (b + 1) of the chunk's kept sites, int64 [N, sites] in written column order.
            chunk = calls[:, selected] if sample_rows is None else calls[:, selected][sample_rows]
            codes = 3 * (chunk[:, :, 0].astype(np.int64) + 1) + chunk[:, :, 1].astype(np.int64) + 1
            # PS per call row and selected site (-1 = unphased), then in the written column order.
            phase_sets = np.full((calls.shape[0], len(selected)), -1, dtype=np.int64)
            if set_starts is not None and len(selected):
                for row_index in range(calls.shape[0]):
                    row_starts = set_starts[row_index if len(set_starts) > 1 else 0]
                    block = np.asarray(np.searchsorted(row_starts, selected, side="right"), dtype=np.int64) - 1
                    inside = block >= 0
                    phase_sets[row_index, inside] = positions[row_starts[block[inside]]]
            phase_sets[:, components[selected] < 0] = -1
            if frame_undetermined is not None and len(selected):
                phase_sets[(np.asarray(frame_undetermined)[:, selected] & DISPLAYED_UNDETERMINED) != 0] = -1
            if sample_rows is not None:
                phase_sets = phase_sets[sample_rows]
            # the records' text as bytes (formatting.sample_lines: each cell the genotype table's text of
            # its phase set, the unphased table for -1; "./.:." throughout a removed record), not Python cell strings.
            prefixes = [
                f"{contig}\t{rows[row]}\t.\t{refs[row]}\t{alts[row]}\t.\t{filters[row]}\t.\tGT:PS\t".encode()
                for row in range(start, stop)
            ]
            prefix = np.frombuffer(b"".join(prefixes), dtype=np.uint8)
            prefix_ends = np.cumsum([len(item) for item in prefixes], dtype=np.int64)
            out = np.empty(sample_line_bytes(len(prefix), stop - start, len(names)), dtype=np.uint8)
            line_ends = np.empty(stop - start, dtype=np.int64)
            written = sample_lines(
                prefix,
                prefix_ends,
                np.array(kept, dtype=np.bool_),  # a writable copy: the declared signature (sites.kept may be frozen)
                np.ascontiguousarray(codes, dtype=np.int64),
                np.ascontiguousarray(phase_sets, dtype=np.int64),
                out,
                line_ends,
            )
            writer.write_chunk(out[:written], line_ends)
    finally:
        writer.close()
