"""Read released founder products without rerunning inference or changing identities."""
from __future__ import annotations

import numpy as np

from . import runtime

FOUNDER_STAGES = ("block_discovery", "feedback_initial", "feedback_l1", "feedback_l2",
                  "assembly_l1", "assembly_l2", "assembly_l3", "assembly_l4",
                  "founder_refinement", "painting", "allele_refit")


def panel_location(store, contig, stage):
    """Return the existing file location, including the refined hierarchy when present."""
    if stage in ("painting", "block_discovery", "allele_refit"):
        options = [(stage, contig)]
    elif stage == "feedback_initial":
        options = [("feedback_path_initial", f"{contig}.__assembly_release__.selected")]
    elif stage.startswith("feedback_l"):
        level = int(stage[-1])
        options = [(f"feedback_path_l{level}", f"{contig}.__assembly_release__.selected")]
    elif stage == "founder_refinement":
        # Final count-up can change the panel after the last hierarchy-level
        # snapshot. Expose it separately, without relabelling an earlier level.
        options = [("assembly", f"{contig}.__assembly_release__.{phase}")
                   for phase in ("final_count_increase", "founder_refinement")]
    elif stage.startswith("assembly_l"):
        level = int(stage[-1])
        options = [("assembly", f"{contig}.__assembly_release__.{phase}_l{level}")
                   for phase in ("refinement", "hierarchy")]
    else:
        raise ValueError(f"unknown founder stage: {stage}")
    for directory, name in options:
        if store.contig_done(directory, name):
            return directory, name
    return None


def load_panels(store, contig, stage="painting"):
    location = panel_location(store, contig, stage)
    if location is None:
        raise FileNotFoundError(f"{contig}: no completed {stage} product")
    payload = store.load_contig(*location)
    if stage == "painting":
        return runtime.validate_phase_component_manifest(payload.component_manifest)
    if stage == "block_discovery":
        return tuple(payload["block_results"])
    if stage == "allele_refit":
        return tuple(payload.components)
    return tuple(payload.payload["blocks"])


def panel_alleles(block):
    """Use published hard calls only; never turn probabilistic ties into alleles."""
    if isinstance(block, painting_allele_refit.RefitComponentPanel):
        keys = tuple(block.founder_keys)
        calls = np.asarray(block.released_alleles, dtype=np.int8)
    else:
        keys = tuple(sorted(block.haplotypes))
        calls = np.asarray(block.discrete_haps, dtype=np.int8)
    positions = np.asarray(block.positions, dtype=np.int64)
    if calls.shape != (len(keys), len(positions)) or np.any(~np.isin(calls, (-1, 0, 1))):
        raise ValueError("founder call/position/key axes are inconsistent")
    return positions, keys, calls

import haplotype_reconstruction.painting.allele_refit as painting_allele_refit
