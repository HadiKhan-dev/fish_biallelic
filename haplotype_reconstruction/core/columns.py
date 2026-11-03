"""Canonical CSV column names shared by reports, evaluation and data exports."""

from __future__ import annotations

import re

_SHARED = {
    "SampleID": "sample",
    "CHROM": "contig",
    "POS": "pos",
    "component_id": "component",
    "export_component": "component",
    "founder_row": "founder",
    "founder_key": "founder",
}


def column_name(name: str) -> str:
    """Split case boundaries, retaining ordinal coordinate suffixes such as 1based."""
    if name in _SHARED:
        return _SHARED[name]
    name = re.sub(r"(\d+)Based\b", r"_\1based", name)
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()
