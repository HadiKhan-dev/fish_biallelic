"""Read genetic-map files without sorting or repairing their coordinates (EC27)."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from typing import TypeAlias, cast

import numpy as np

from ...contracts.genetic_map import DEFAULT_RATE_CM_PER_MB, ChromosomeGeneticMap, GeneticMapSet

__all__ = ["load_genetic_maps"]
MapPath: TypeAlias = str | PathLike[str]

_COMBINED_HEADER = ["chromosome", "position_bp", "map_cm"]

_SHAPEIT_HEADERS = (
    ["position", "combined_rate(cm/mb)", "genetic_map(cm)"],
    ["position", "rate(cm/mb)", "map(cm)"],
    ["physical_bp", "rate_cm/mb", "cumulative_cm"],
)


def _read_map(path: MapPath, assigned_contig: str | None = None) -> dict[str, tuple[list[int], list[float]]]:
    """Read a map file without reordering rows or guessing chromosome labels: {contig: (positions, cM)}.

    Formats: headed combined TSV (chromosome, position_bp, map_cm); headed SHAPEIT single-chromosome maps (three
    header variants; need ``assigned_contig``); headerless four-column PLINK (chromosome, id, cM, bp).
    """
    rows: dict[str, tuple[list[int], list[float]]] = {}
    mode = None
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if mode is None:
                header = [item.lower() for item in fields]
                if header == _COMBINED_HEADER:
                    mode = "combined"
                    continue
                if header in _SHAPEIT_HEADERS:
                    if assigned_contig is None:
                        raise ValueError(f"{path}:{line_number}: SHAPEIT map needs a chromosome-to-path mapping")
                    mode = "shapeit"
                    continue
                if len(fields) == 4:
                    mode = "plink"
                else:
                    raise ValueError(f"{path}:{line_number}: expected four-column PLINK cM map or explicit map header")
            expected_columns = 4 if mode == "plink" else 3
            if len(fields) != expected_columns:
                raise ValueError(f"{path}:{line_number}: expected {expected_columns} columns for {mode} map")
            try:
                if mode == "plink":
                    contig, position_text, cm_text = fields[0], fields[3], fields[2]
                elif mode == "combined":
                    contig, position_text, cm_text = fields
                else:
                    contig, position_text, cm_text = cast(str, assigned_contig), fields[0], fields[2]
                    local_rate = float(fields[1])
                    if not np.isfinite(local_rate) or local_rate < 0:
                        raise ValueError("SHAPEIT local rate must be finite and nonnegative")
                position = int(position_text)
                cm = float(cm_text)
                if position < 0:
                    raise ValueError("physical position must be nonnegative")
                if not np.isfinite(cm):
                    raise ValueError("cumulative cM must be finite")
                if assigned_contig is not None and contig != assigned_contig:
                    raise ValueError(f"chromosome {contig!r} differs from assigned chromosome {assigned_contig!r}")
                chromosome_rows = rows.setdefault(contig, ([], []))
                if chromosome_rows[0] and position <= chromosome_rows[0][-1]:
                    raise ValueError("physical positions must be strictly increasing (no duplicates)")
                if chromosome_rows[1] and cm < chromosome_rows[1][-1]:
                    raise ValueError("cumulative cM must be nondecreasing")
                chromosome_rows[0].append(position)
                chromosome_rows[1].append(cm)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not rows:
        raise ValueError(f"{path}: map contains no records")
    return rows


def load_genetic_maps(
    path_or_mapping: MapPath | Mapping[str, MapPath] | None = None,
    default_rate_cm_per_mb: float = DEFAULT_RATE_CM_PER_MB,
) -> GeneticMapSet:
    """Load a combined or PLINK map file, or {exact contig: single-chromosome map path} (also SHAPEIT).

    None gives a constant-rate map set. Cumulative map input is always in cM (PLINK files whose third column is
    in Morgans must be converted first).
    """
    maps = {}
    if path_or_mapping is not None:
        sources = path_or_mapping.items() if isinstance(path_or_mapping, Mapping) else [(None, path_or_mapping)]
        for assigned_contig, path in sources:
            assigned_contig = None if assigned_contig is None else str(assigned_contig)
            for contig, (positions, cm) in _read_map(path, assigned_contig).items():
                maps[contig] = ChromosomeGeneticMap(
                    contig,
                    np.asarray(positions, dtype=np.float64),
                    np.asarray(cm, dtype=np.float64),
                    default_rate_cm_per_mb,
                )
    return GeneticMapSet(maps, default_rate_cm_per_mb)
