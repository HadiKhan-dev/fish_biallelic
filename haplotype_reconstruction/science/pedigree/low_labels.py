"""Apply the regime's release provenance only after final view composition (DESIGN_4154 A4, DESIGN_417 G10).

Below Tier A's material (U < MaterialParameters.tier_a_morgans) the Tier A tables are empty: Tier A is neither
published nor used to order parent slots. A low-material run's Tier B rows name their low-material resampling.
"""
from dataclasses import replace


def label_views(views, released, *, low_material: bool, tier_a: bool):
    changes = {} if tier_a else dict(tier_a_relationships=(), tier_a_partial_relationships=())
    if not low_material:
        return replace(views, **changes)

    def evidence(row):
        child = row.sample
        count = bool(released.count_release.released[child]) and released.calls.state[child] >= 0
        # Every row of a low-material run rests on unit resampling with the low-material thresholds, released or
        # not: an unresolved sample is labelled like a released one, as unit resampling labels every row of a
        # standard run.
        return "low_material_exclusion" if count else "low_material_resampling"
    for table in ("tier_b_relationships", "tier_b_partial_relationships"):
        changes[table] = tuple(replace(row, release_evidence=evidence(row))
                               for row in getattr(views, table))
    for table in ("tier_b_candidate_sets", "parent_state_calls", "diagnostics", "call_explanations"):
        changes[table] = tuple(replace(row, tier_b_release_evidence=evidence(row))
                               for row in getattr(views, table))
    return replace(views, **changes)
