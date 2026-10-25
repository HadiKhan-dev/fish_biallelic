"""Discovery's switch for its arbitrary conventions (release 2.2 convention-reversal test, design/V22_BRIEF.md).

``rev(name)`` is True when the run reverses its conventions (maths.ties.reverse_conventions, environment
HAPLOTYPES_TIE_REVERSE=1). Diagnostics only: the environment variable HAPLOTYPES_TIE_REVERSE_KEEP (comma-separated names, or "all") keeps
the listed conventions at their default while the others are reversed, to find which one a reversal difference comes
from. Names: d1 (A-step row order in the band), d3 (Gauss-Seidel founder order on usage ties), d8a (dosage argmax
ties), d8s (the binary default state at exactly 1/2), d9 (gauge max-cut code order and heuristic vertex order), d9p
(proposal order before the expansion budget), d10p (dominant-partner ties), d12 (mean-field start order),
d13 (greedy residual edge-cover content order). The central
reversals of maths.ties (content order inside tie classes, sample-name order) are not covered by HAPLOTYPES_TIE_REVERSE_KEEP.
"""
from __future__ import annotations

import os

from ..maths.ties import reverse_conventions


def rev(name: str) -> bool:
    """Whether discovery's convention ``name`` runs reversed."""
    if not reverse_conventions():
        return False
    keep = {item.strip() for item in os.environ.get("HAPLOTYPES_TIE_REVERSE_KEEP", "").split(",") if item.strip()}
    return not ("all" in keep or name in keep)
