"""Optional, sample-name-based constraints for the general VCF workflow."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from ..pedigree.eligibility import _resolve_parent_eligibility


def load_constraints(path, sample_ids):
    """Unlisted children retain all non-self candidates; [] explicitly allows none.

    Excluded samples are outside pedigree inference (both children and parents).
    Direction edges are [parent, child] and require independently known temporal
    order; candidate lists alone do not imply direction support or true parentage.
    """
    if path is None:
        return None
    data = json.loads(Path(path).read_text())
    allowed = {"candidate_parents", "excluded_samples", "ineligible_children",
               "direction_supported_edges"}
    if not isinstance(data, dict) or set(data) - allowed:
        raise ValueError("eligibility JSON must contain only " + ", ".join(sorted(allowed)))
    ids = {name: i for i, name in enumerate(sample_ids)}

    def index(name):
        if name not in ids:
            raise ValueError(f"unknown eligibility sample {name!r}")
        return ids[name]

    n = len(ids)
    children = np.ones(n, dtype=bool)
    parents = ~np.eye(n, dtype=bool)
    for child, candidates in data.get("candidate_parents", {}).items():
        if not isinstance(candidates, list):
            raise ValueError("candidate_parents values must be lists of sample IDs")
        c = index(child)
        selected = [index(name) for name in candidates]
        if c in selected:
            raise ValueError("a sample cannot be its own parent")
        parents[c] = False
        parents[c, selected] = True
    for name in data.get("excluded_samples", []):
        i = index(name)
        children[i] = False
        parents[i] = parents[:, i] = False
    for name in data.get("ineligible_children", []):
        i = index(name)
        children[i] = False
        parents[i] = False
    direction = np.zeros_like(parents)
    for edge in data.get("direction_supported_edges", []):
        if not isinstance(edge, list) or len(edge) != 2:
            raise ValueError("direction_supported_edges entries must be [parent, child]")
        p, c = map(index, edge)
        if not parents[c, p]:
            raise ValueError("direction support names an ineligible parent/child edge")
        direction[c, p] = True
    record = dict(format_version=1, sample_ids=tuple(sample_ids), eligible_children=children,
                  eligible_parents=parents, eligible_parent_pairs=None,
                  direction_supported_parents=direction,
                  policy_name="explicit_sample_constraints_v1", source_fields=tuple(sorted(data)),
                  assumptions=("Caller constraints are eligibility/chronology, not known individual parentage.",),
                  individual_parentage_ground_truth=False)
    _resolve_parent_eligibility(record, sample_ids)
    return record
