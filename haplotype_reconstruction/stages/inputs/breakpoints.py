"""Read the shared reference-order breakpoint TSV at the stage boundary (RC01).

Coordinates are 1-based inclusive. File order is scientific: recombination widens
ordinary intervals first, then the two endpoints of each reversed segment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...contracts.roles import record_field
from ...core.errors import SettingsError


@dataclass(frozen=True)
class BreakpointTable:
    """Parsed input rows, preserving native order; applied record types belong to record codec."""

    rows: tuple[tuple[str, int, int, str], ...] = record_field(
        "obs", axes=("R", "4"), units="contig,bp,bp,reason", dtype="tuple[str,int,int,str]"
    )

    def protected(self) -> tuple[tuple[str, int, int], ...]:
        """Every ordinary interval and both endpoints of a reversed segment."""
        out: list[tuple[str, int, int]] = []
        for contig, start, end, reason in self.rows:
            if reason.startswith("reversed_segment"):
                out.extend(((contig, start, start), (contig, end, end)))
            else:
                out.append((contig, start, end))
        return tuple(out)


HEADER = ("chrom", "start", "end", "reason")


def read_breakpoint_table(path: str | Path) -> BreakpointTable:
    """Read native TSV semantics; malformed rows raise SettingsError with the line number."""
    rows = []
    header = None
    try:
        handle = open(path, encoding="utf-8")
    except OSError as error:
        raise SettingsError(f"{path}: {error}") from error
    with handle:
        for number, line in enumerate(handle, 1):
            text = line.split("#", 1)[0].rstrip("\r\n")
            if not text.strip():
                continue
            fields = [field.strip() for field in text.split("\t")]
            if header is None:
                header = tuple(fields[:4])
                if header != HEADER:
                    raise SettingsError(f"{path}:{number}: breakpoint file needs the header chrom, start, end, reason")
                continue
            if len(fields) < 3:
                raise SettingsError(f"{path}:{number}: breakpoint row needs chrom, start, end")
            try:
                start, end = int(fields[1]), int(fields[2])
            except ValueError as error:
                raise SettingsError(f"{path}:{number}: breakpoint start/end must be integers") from error
            if start < 1 or end < start:
                raise SettingsError(f"{path}:{number}: breakpoint intervals need 1 <= start <= end")
            rows.append((fields[0], start, end, fields[3] if len(fields) > 3 else ""))
    if header is None:
        raise SettingsError(f"{path}: breakpoint file has no header")
    return BreakpointTable(tuple(rows))
