"""Import-time numerical-library limits and supported environment choices."""
from __future__ import annotations


import os
import json


NUMERIC_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def assembly_transition_model():
    """Assembly choice is independent of discovery; dense is the default."""
    value = os.environ.get("HAPLOTYPES_ASSEMBLY_MODEL", "dense")
    if value not in ("dense", "structured"):
        raise ValueError("HAPLOTYPES_ASSEMBLY_MODEL must be dense or structured")
    return value


def assembly_panel_search():
    """Panel-search breadth is independent of the assembly transition model."""
    value = os.environ.get("HAPLOTYPES_ASSEMBLY_SEARCH", "bounded")
    if value not in ("bounded", "broad"):
        raise ValueError("HAPLOTYPES_ASSEMBLY_SEARCH must be bounded or broad")
    return value


def assembly_founder_refinement():
    """Reopen local row choices after each final L1-L4 level by default."""
    value = os.environ.get("HAPLOTYPES_FOUNDER_REFINEMENT", "on")
    if value not in ("on", "off"):
        raise ValueError("HAPLOTYPES_FOUNDER_REFINEMENT must be on or off")
    return value == "on"


def block_feedback_selection():
    """Balanced local feedback rescue is the default; strict protects calls."""
    value = os.environ.get("HAPLOTYPES_FEEDBACK_SELECTION", "balanced")
    if value not in ("balanced", "strict"):
        raise ValueError("HAPLOTYPES_FEEDBACK_SELECTION must be balanced or strict")
    return value


def batched_discovery_enabled():
    """The experimental discovery search requires its own explicit choice."""
    value = os.environ.get("HAPLOTYPES_DISCOVERY_SEARCH", "standard")
    if value not in ("standard", "batched"):
        raise ValueError("HAPLOTYPES_DISCOVERY_SEARCH must be standard or batched")
    return value == "batched"


def configured_regions(default, *, template_regions=False):
    requested = os.environ.get("HAPLOTYPES_CONTIGS")
    if requested is None:
        return default
    names = json.loads(requested)
    if not names or len(names) != len(set(names)):
        raise ValueError("contigs must be a nonempty unique ordered list")
    return [dict(contig=str(name), **({'start': 0, 'end': 3000} if template_regions else {}))
            for name in names]



def force_single_threaded_numeric_libraries():
    """Force BLAS/OpenMP libraries to one thread before importing them."""

    for variable in NUMERIC_THREAD_ENV_VARS:
        os.environ[variable] = "1"
