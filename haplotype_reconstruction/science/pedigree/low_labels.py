"""Apply low-material release provenance only after final view composition (A4)."""
from dataclasses import replace


def label_views(views, released):
    def evidence(row):
        child = row.sample
        count = bool(released.count_release.released[child]) and released.calls.state[child] >= 0
        # Every row of a low-material run rests on block resampling, released or not: an unresolved sample is
        # labelled like a released one, as chromosome resampling labels every row of a standard run.
        return "low_material_exclusion" if count else "low_material_resampling"
    changes = dict(tier_a_relationships=(), tier_a_partial_relationships=())
    for table in ("tier_b_relationships", "tier_b_partial_relationships"):
        changes[table] = tuple(replace(row, release_evidence=evidence(row))
                               for row in getattr(views, table))
    for table in ("tier_b_candidate_sets", "parent_state_calls", "diagnostics", "call_explanations"):
        changes[table] = tuple(replace(row, tier_b_release_evidence=evidence(row))
                               for row in getattr(views, table))
    return replace(views, **changes)
