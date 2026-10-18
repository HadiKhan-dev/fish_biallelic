"""Reference-order breakpoint table (one reader for family phase and recombination).

File format (docs): tab-separated, a header line ``chrom start end reason``, ``#`` comments, 1-based inclusive
bp. The order discontinuity lies somewhere inside [start, end]. A row whose reason starts with
``reversed_segment`` declares an inverted reference segment instead (both of its ends are breakpoints).
Family phase protects the ordinary breakpoint rows from root-frame repairs; recombination masks every
breakpoint and flags calls inside reversed segments.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


REVERSED_SEGMENT = "reversed_segment"
HEADER = ("chrom", "start", "end", "reason")


@dataclass(frozen=True)
class BreakpointTable:
    """Rows (contig, start, end, reason) in file order."""

    rows: tuple[tuple[str, int, int, str], ...]

    def protected(self):
        """(contig, start, end) of the ordinary breakpoint rows: family phase's protected intervals."""
        return tuple((contig, start, end) for contig, start, end, reason in self.rows
                     if not reason.startswith(REVERSED_SEGMENT))


def read_breakpoint_table(path):
    """Read a breakpoint TSV (module docstring) into a BreakpointTable; format errors raise ValueError."""
    rows = []
    header = None
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            text = line.split("#", 1)[0].rstrip("\r\n")
            if not text.strip():
                continue
            fields = [field.strip() for field in text.split("\t")]
            if header is None:
                header = tuple(fields[:4])
                if header != HEADER:
                    raise ValueError(f"{path}:{number}: breakpoint file needs the header chrom, start, end, reason")
                continue
            if len(fields) < 3:
                raise ValueError(f"{path}:{number}: breakpoint row needs chrom, start, end")
            try:
                start, end = int(fields[1]), int(fields[2])
            except ValueError as error:
                raise ValueError(f"{path}:{number}: breakpoint start/end must be integers") from error
            if start < 1 or end < start:
                raise ValueError(f"{path}:{number}: breakpoint intervals need 1 <= start <= end")
            rows.append((fields[0], start, end, fields[3] if len(fields) > 3 else ""))
    if header is None:
        raise ValueError(f"{path}: breakpoint file has no header")
    return BreakpointTable(tuple(rows))


def contig_breakpoints(rows, contig):
    """(breakpoints int64 (k, 2), reversed segments int64 (j, 2), reasons) of one contig, or None without rows."""
    selected = [row for row in rows if row[0] == contig]
    if not selected:
        return None
    order = [(start, end) for _, start, end, reason in selected if not reason.startswith(REVERSED_SEGMENT)]
    reversed_segments = [(start, end) for _, start, end, reason in selected if reason.startswith(REVERSED_SEGMENT)]
    return (np.asarray(order, dtype=np.int64).reshape(-1, 2),
            np.asarray(reversed_segments, dtype=np.int64).reshape(-1, 2),
            tuple(reason for *_, reason in selected))
