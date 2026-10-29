"""Founder products of one chromosome.

Founder tracks (product ``tracks``): one gzip TSV per release segment, `CHROM POS H0 H1 ...`, one row per panel site.
Cells are the released calls 0 / 1, or `.` where the panel withholds the allele (no imputation). Column Hk is
founder row k of this segment: founder labels are segment-local (H0 of two segments is not one founder).

Founder VCF (product ``founders``): one BCF/VCF.gz per chromosome at the rows of an export.sites table, FORMAT GT
haploid, one column ``c<C>_H<k>`` per founder row k of every release segment C. A column has calls only inside its
segment's span (first to last panel position): the released call at the component's panel sites, `.` at a span
site without a panel allele, and in a full-length file the common allele at sites removed by r5 alone (`.` at sites
removed by r1-r4). Outside its span a column is `.`, so no column implies an identity across release segments.
Source inference components and segment indices are retained in the manifest and VCF header.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

from .sites import SiteTable
from .vcf import FILEFORMAT, common_allele_line, contig_line, site_header_lines

_TEXT = np.array([".", "0", "1"])
# Sites formatted at once: the text temporaries stay below ~20 bytes x founder rows x CHUNK_SITES.
CHUNK_SITES = 1 << 16
# Founder VCF rows formatted at once per component (text lists of founder rows x VCF_CHUNK_SITES).
VCF_CHUNK_SITES = 4096


def write_founder_tracks(path: Path, contig: str, positions: np.ndarray, calls: np.ndarray) -> None:
    """Write ``calls`` int8 [K, L] (-1 / 0 / 1) at ``positions`` int64 [L] atomically to ``path``."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    calls = np.asarray(calls)
    positions = np.asarray(positions)
    with gzip.open(temporary, "wt", compresslevel=6) as handle:
        handle.write("CHROM\tPOS\t" + "\t".join(f"H{row}" for row in range(len(calls))) + "\n")
        for start in range(0, len(positions), CHUNK_SITES):
            text = _TEXT[calls[:, start:start + CHUNK_SITES].astype(np.int64) + 1]
            for site, position in enumerate(positions[start:start + CHUNK_SITES].tolist()):
                handle.write(f"{contig}\t{int(position)}\t" + "\t".join(text[:, site]) + "\n")
    temporary.replace(path)


def vcf_column(component: int, row: int) -> str:
    """The founder VCF column of founder row ``row`` of component ``component``."""
    return f"c{component}_H{row}"


def component_record(contig: str, component: int, positions: np.ndarray, calls: np.ndarray, stage: str,
                     filename: str | None = None, vcf_file: str | None = None,
                     withheld: int | None = None, release_segment: dict | None = None) -> dict:
    """The manifest entry of one exported component (``filename``: its track TSV; ``vcf_file``: the founder VCF
    holding its columns; ``withheld``: alleles released as '.' because no painted carrier observes them)."""
    calls = np.asarray(calls)
    entry = dict(contig=contig, component=component)
    if release_segment is not None:
        entry.update(release_segment)
    if filename is not None:
        entry["local_columns"] = {f"H{row}": str(row) for row in range(len(calls))}
    entry.update(first_position=int(positions[0]), last_position=int(positions[-1]), markers=len(positions),
                 called_alleles=int((calls >= 0).sum()), missing_alleles=int((calls < 0).sum()),
                 founder_stage=stage)
    if withheld is not None:
        entry["withheld_unobserved_alleles"] = int(withheld)
    if filename is not None:
        entry["file"] = filename
    if vcf_file is not None:
        entry["vcf_file"] = vcf_file
        entry["vcf_columns"] = [vcf_column(component, row) for row in range(len(calls))]
    return entry


def founder_values(sites: SiteTable, panels, release_segments=None) -> list[tuple[int, int, np.ndarray]]:
    """Per component (row range [lo, hi) of ``sites`` covered by its span, int8 [K, hi - lo] alleles): the panel
    calls at its sites, the common allele at sites removed by r5 alone, -1 elsewhere."""
    rows = sites.positions
    blocks = []
    for component, (positions, calls) in enumerate(panels):
        positions = np.asarray(positions, dtype=np.int64)
        lo = int(np.searchsorted(rows, positions[0], side="left"))
        hi = int(np.searchsorted(rows, positions[-1], side="right"))
        columns = np.searchsorted(rows, positions)
        if (np.any(columns >= len(rows)) or not np.array_equal(rows[np.minimum(columns, len(rows) - 1)], positions)
                or not sites.kept[columns].all()):
            raise ValueError(f"{sites.contig}: component {component} has panel sites that are not kept sites")
        values = np.full((len(calls), hi - lo), -1, dtype=np.int8)
        values[:, columns - lo] = calls
        common = sites.common[lo:hi]
        fill = common >= 0            # set only at sites removed by r5 alone (never panel sites)
        values[:, fill] = common[fill]
        if release_segments is not None:
            no_fill = release_segments[component].get("r5_withheld_rows", ())
            if no_fill:
                values[np.ix_(no_fill, np.flatnonzero(fill))] = -1
        blocks.append((lo, hi, values))
    return blocks


def write_founder_vcf(path: Path, sites: SiteTable, panels, stage: str, *, synthetic: bool,
                      file_format: str, withheld=None, release_segments=None) -> None:
    """Write the founder VCF of one chromosome (module docstring) to ``path`` (format "bcf" or "vcf").

    ``panels``: [(positions int64 [L_c], released int8 [K_c, L_c])] in component order; ``withheld``: alleles
    per component released as '.' because no painted carrier observes them (header ##component lines).
    """
    from cyvcf2 import Writer
    contig = sites.contig
    panels = [(np.asarray(positions, dtype=np.int64), np.asarray(calls, dtype=np.int8))
              for positions, calls in panels]
    blocks = founder_values(sites, panels, release_segments)
    columns = [vcf_column(component, row) for component, (_, calls) in enumerate(panels)
               for row in range(len(calls))]
    length = int(sites.positions[-1]) if synthetic else sites.contig_length
    header = (FILEFORMAT +
              "##source=haplotype-reconstruction\n"
              + contig_line(contig, length) +
              '##FORMAT=<ID=GT,Number=1,Type=String,Description="Founder allele (haploid): 0 REF, 1 ALT, . unknown '
              '(withheld, or no painted carrier observes it) or outside the column\'s component; no imputation">\n'
              + site_header_lines(sites) +
              '##founder_columns="cC_Hk is founder row k of release segment C of this chromosome. Founder labels '
              'are segment-local: columns of different segments (c0_H1, c1_H1) are not implied to be one '
              'founder. A column has calls only inside its segment\'s span (first to last panel position; '
              '##component lines) and . outside it; inside the span . marks an allele the panel withholds"\n'
              f"##founder_stage={stage}\n")
    for component, (positions, calls) in enumerate(panels):
        unobserved = "" if withheld is None else f",UnobservedWithheld={int(withheld[component])}"
        release_info = ""
        if release_segments is not None:
            segment = release_segments[component]
            release_info = (f",SourceComponent={segment['source_component']},"
                            f"SegmentIndex={segment['segment_index']},"
                            f"AmbiguousWithheld={segment['withheld_ambiguous_alleles']}")
        header += (f"##component=<ID=c{component},Start={int(positions[0])},End={int(positions[-1])},"
                   f"Founders={len(calls)},Sites={len(positions)}{unobserved}{release_info}>\n")
    if release_segments is not None:
        for component, segment in enumerate(release_segments):
            for number, detail in enumerate(segment.get("count_unverified", ())):
                first, last = detail["uncertainty_interval"]
                header += (f"##founder_count_verification=<ID=c{component}u{number},Status=count_unverified,"
                           f"SourceComponent={segment['source_component']},SegmentIndex={segment['segment_index']},"
                           f"Start={first},End={last},Cause={detail['cause']},"
                           f"DomainSize={detail['domain_size']},Evaluated={detail['evaluated']}>\n")
            for number, detail in enumerate(segment.get("continuation_unverified", ())):
                first, last = detail["uncertainty_interval"]
                header += (f"##founder_continuation_verification=<ID=c{component}v{number},"
                           f"Status=continuation_unverified,SourceComponent={segment['source_component']},"
                           f"SegmentIndex={segment['segment_index']},Start={first},End={last},"
                           f"Cause={detail['cause']},Alternatives={detail['alternative_count']},"
                           f"ObservedIntervals={len(detail['local_disagreement_intervals'])}>\n")
            for number, detail in enumerate(segment.get("ambiguity", ())):
                first, last = detail["uncertainty_interval"]
                header += (f"##founder_ambiguity=<ID=c{component}a{number},SourceComponent="
                           f"{segment['source_component']},SegmentIndex={segment['segment_index']},"
                           f"Start={first},End={last},Reason={detail['kind']},"
                           f"Alternatives={detail['alternative_count']}>\n")
    if withheld is not None:
        header += (f'##unobserved_founder_alleles={int(sum(withheld))}\n'
                   '##unobserved_founder_alleles_rule="a founder allele is written as . when no sample whose '
                   'released painting names that founder at the site has read depth there (release 2.2)"\n')
    if sites.full_length:
        header += common_allele_line()
        if release_segments is not None:
            header += ('##assembly_ambiguity_r5="r5 filling cannot bridge release segments or restore a row '
                       'withheld for evaluated unresolved assembly content"\n')
    if synthetic:
        header += '##synthetic_alleles="A/C encode abstract simulated 0/1; not reference-genome bases"\n'
    header += "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO" + ("\tFORMAT\t" + "\t".join(columns)
                                                                    if columns else "") + "\n"
    rows = sites.positions.tolist()
    filters = sites.filters()
    dots = ["\t".join(["."] * len(calls)) for _, calls in panels]
    # Row ranges with one set of covering components (spans are disjoint in practice; overlaps are handled).
    boundaries = sorted({0, len(rows), *(lo for lo, _, _ in blocks), *(hi for _, hi, _ in blocks)})
    writer = Writer.from_string(str(path), header, mode="wb" if file_format == "bcf" else "wz")

    def emit(row: int, genotypes: str | None) -> None:
        line = f"{contig}\t{rows[row]}\t.\t{sites.ref[row]}\t{sites.alt[row]}\t.\t{filters[row]}\t."
        writer.write_record(writer.variant_from_string(line if genotypes is None else line + "\tGT\t" + genotypes))

    try:
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            cover = [index for index, (lo, hi, _) in enumerate(blocks) if lo <= a and b <= hi]
            if not columns or not cover:
                text = "\t".join(dots) if columns else None
                for row in range(a, b):
                    emit(row, text)
                continue
            template = list(dots)
            for start in range(a, b, VCF_CHUNK_SITES):
                stop = min(b, start + VCF_CHUNK_SITES)
                texts = {index: _TEXT[blocks[index][2][:, start - blocks[index][0]:stop - blocks[index][0]]
                                      .astype(np.int64) + 1].T.tolist() for index in cover}
                for row in range(start, stop):
                    for index in cover:
                        template[index] = "\t".join(texts[index][row - start])
                    emit(row, "\t".join(template))
    finally:
        writer.close()


IDENTITY_COLUMNS = ("contig", "column", "export_component", "founder_row", "source_stage", "source_component",
                    "source_row", "block", "left_position", "right_position", "left_local_row", "right_local_row",
                    "left_partners", "right_partners", "at_release_break", "reason", "method", "status")


def write_identity_flags(path: Path, contig: str, stage: str, identity_paths, segments=None) -> int:
    """Write the founder identity flags of one chromosome (export.identity.partner_changes on the exported panel's
    row paths) atomically to ``path``; returns their number.

    ``identity_paths``: per source component (first prepared block, positions int64 [L], atomic_offsets int64 [B],
    row paths [K, B]); ``segments``: the assembly release's segment metadata in export-component order
    (source_component, global block_start/block_stop, source_founder_rows), None when every source component is
    exported whole with its own row order. A flag is listed under the export component holding its right block, its
    rows renamed to that component's columns through source_founder_rows; a flag at a release segment boundary
    (at_release_break 1) names its left partners in the left segment's columns. Positions are the last marker of
    the left block and the first of the right one.
    """
    from .identity import METHOD, REASON, partner_changes
    path = Path(path)
    lines = []
    for source, (first_block, positions, offsets, rows) in enumerate(identity_paths):
        positions = np.asarray(positions, dtype=np.int64)
        offsets = np.asarray(offsets, dtype=np.int64)
        if segments is None:
            owners = [(source, dict(block_start=first_block, block_stop=first_block + len(offsets),
                                    source_founder_rows=list(range(len(rows)))))]
        else:
            owners = [(index, segment) for index, segment in enumerate(segments)
                      if segment["source_component"] == source]

        def owner(block):
            return next(((index, segment) for index, segment in owners
                         if segment["block_start"] <= first_block + block < segment["block_stop"]), (None, None))

        def columns(index, segment):
            order = list(segment["source_founder_rows"])
            return {row: (order.index(row), vcf_column(index, order.index(row))) for row in range(len(rows))}

        for flag in partner_changes(rows):
            index, segment = owner(flag.block)
            if segment is None:
                continue
            right = columns(index, segment)
            at_break = int(first_block + flag.block == segment["block_start"])
            left_owner = owner(flag.block - 1) if at_break else (index, segment)
            left = right if left_owner[1] is None or not at_break else columns(*left_owner)
            lines.append((contig, right[flag.row][1], index, right[flag.row][0], stage, source, flag.row,
                          first_block + flag.block, int(positions[offsets[flag.block] - 1]),
                          int(positions[offsets[flag.block]]), flag.left_local_row, flag.right_local_row,
                          ",".join(left[r][1] for r in flag.left_partners),
                          ",".join(right[r][1] for r in flag.right_partners), at_break, REASON, METHOD,
                          "diagnostic_only"))
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("\t".join(IDENTITY_COLUMNS) + "\n"
                         + "".join("\t".join(map(str, line)) + "\n" for line in lines))
    temporary.replace(path)
    return len(lines)
