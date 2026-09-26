"""Read released founder products without rerunning inference or changing identities."""
from __future__ import annotations

import numpy as np

from . import runtime

FOUNDER_STAGES = ("block_discovery", "feedback_initial", "feedback_l1", "feedback_l2",
                  "feedback_exchange",
                  "assembly_l1", "assembly_l2", "assembly_l3", "assembly_l4",
                  "founder_refinement", "painting")


def panel_location(store, contig, stage, feedback_selection="path"):
    """Return the existing file location, including the refined hierarchy when present."""
    if stage in ("painting", "block_discovery"):
        options = [(stage, contig)]
    elif stage in ("feedback_initial", "feedback_exchange"):
        if feedback_selection != "path":
            return None
        directory = "feedback_path_initial" if stage == "feedback_initial" else "feedback_path_exchange"
        options = [(directory, f"{contig}.__assembly_release__.selected")]
    elif stage.startswith("feedback_l"):
        level = int(stage[-1])
        options = [(f"feedback_{feedback_selection}_l{level}",
                    f"{contig}.__assembly_release__.selected")]
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


def load_panels(store, contig, stage="painting", *, feedback_selection="path"):
    location = panel_location(store, contig, stage, feedback_selection)
    if location is None:
        raise FileNotFoundError(f"{contig}: no completed {stage} product")
    payload = store.load_contig(*location)
    if stage == "painting":
        return runtime.validate_phase_component_manifest(payload.component_manifest)
    if stage == "block_discovery":
        return tuple(payload["block_results"])
    return tuple(payload.payload["blocks"])


def panel_alleles(block):
    """Use published hard calls only; never turn probabilistic ties into alleles."""
    keys = tuple(sorted(block.haplotypes))
    calls = np.asarray(block.discrete_haps, dtype=np.int8)
    positions = np.asarray(block.positions, dtype=np.int64)
    if calls.shape != (len(keys), len(positions)) or np.any(~np.isin(calls, (-1, 0, 1))):
        raise ValueError("founder call/position/key axes are inconsistent")
    return positions, keys, calls
