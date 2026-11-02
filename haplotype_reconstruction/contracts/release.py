"""Publish the source-site release independently of export options (OU01/02).

ReleasedSites contains every loader-retained site, including removed sites and
r5-only common alleles. The sites stage computes it; output writers select rows.
This contract performs no source I/O, filtering decisions or output formatting.
FounderRelease is owned by contracts.assembly (AS6). Method: WP_OU1 §5–§6.1.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .roles import record_field, schema_field

__all__ = ["ReleasedSites"]


@dataclass(frozen=True)
class ReleasedSites:
    """All M source sites, their QC decisions and r5-only common alleles.

    common int8[M] is -1 outside r5-only sites or on tied totals. common_counts
    holds ref, alt, tie, from_reads and from_gt integer counts. ref/alt are <U1.
    Arrays are read-only views and retain exact dtype/value; construction does not
    change caller flags. synthetic identifies abstract A/C simulated alleles.
    """

    schema: int = schema_field(1)
    contig: str = record_field("obs", dtype="str")
    positions: NDArray[np.int64] = record_field("obs", axes=("M",), units="bp", dtype="int64")
    kept: NDArray[np.bool_] = record_field("call", axes=("M",), dtype="bool")
    removed_bits: NDArray[np.uint8] = record_field("call", axes=("M",), dtype="uint8")
    ref: NDArray[np.str_] = record_field("obs", axes=("M",), dtype="<U1")
    alt: NDArray[np.str_] = record_field("obs", axes=("M",), dtype="<U1")
    common: NDArray[np.int8] = record_field("call", axes=("M",), dtype="int8")
    common_counts: Mapping[str, int] = record_field(
        "report", units="count", dtype="Mapping[str, int]", default_factory=dict
    )
    contig_length: int | None = record_field("obs", units="bp", dtype="int64 | None", default=None)
    site_filter: str = record_field("call", dtype="str", default="unknown")
    applied_rules: tuple[str, ...] = record_field("call", dtype="str", default=())
    dropped: Mapping[str, int] = record_field("obs", units="count", dtype="Mapping[str, int]", default_factory=dict)
    synthetic: bool = record_field("obs", dtype="bool", default=False)

    def __post_init__(self) -> None:
        for name in ("positions", "kept", "removed_bits", "ref", "alt", "common"):
            array = getattr(self, name).view()
            array.flags.writeable = False
            object.__setattr__(self, name, array)
