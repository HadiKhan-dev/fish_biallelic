"""Bounded complete component choices for larger-context comparisons.

Only terminal, scored panels with an established component correspondence can
be input alternatives. An unexplored search history remains explicitly unknown.
The bound limits computation, never the recorded biological family.
"""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import product
from typing import Any

import numpy as np

from haplotype_reconstruction.contracts.assembly import ReleaseFamily
from haplotype_reconstruction.core import ties

from .orders import panel_key
from .parameters import BRANCH_LIMIT, WIDE_CLASS_QUANTA

# Contract strings from 3.5 assembly-core U02/U13/U16; no tuning or normalization.
AMBIGUOUS_K = "ambiguous_k"
COMMON_COHORT_PARENT_JOIN = "common_cohort_parent_join"
COMPONENT_COUNT_NOT_REDUCED = "component_count_not_reduced"
CONTEXT_LEVELS_REACHED = "context_levels_reached"
ONLY_PHASE_BREAKS_REMAIN = "only_phase_breaks_remain"
PARENT_JOIN_ALTERNATIVES = "parent_join_alternatives"
SAME_K_EQUAL_LL = "same_k_equal_LL"
SINGLE_COMPONENT = "single_component"
SMALLER_K_THEN_EXACT_MINIMUM_SWITCHES = "smaller_K_then_exact_minimum_switches"
UNRESOLVED_COMPONENT_CORRESPONDENCE = "unresolved_component_correspondence"
UNRESOLVED_CONTINUATION = "unresolved_continuation"


def component_choices(components, families):
    """(complete choices per component, consumable families, unexamined families)."""
    if not families:
        return tuple((component,) for component in components), (), ()
    options = [{panel_key(component.rows): component} for component in components]
    spans = {(component.start, component.stop): i for i, component in enumerate(components)}
    consumed, pending = [], []
    for family in families:
        family = ReleaseFamily.from_mapping(family)
        index = spans.get((int(family["start"]), int(family["stop"])))
        if index is None or family.get("unresolved_search", False):
            pending.append(family.updated(unresolved_search=True, status=family.get("status", UNRESOLVED_CONTINUATION)))
            continue
        component = components[index]
        panels = {
            panel_key(rows): np.ascontiguousarray(panel_key(rows), dtype=np.int32) for rows in family.get("panels", ())
        }
        if panel_key(component.rows) not in panels:
            pending.append(family.updated(unresolved_search=True, status=UNRESOLVED_COMPONENT_CORRESPONDENCE))
            continue
        options[index].update((key, replace(component, rows=rows)) for key, rows in panels.items())
        consumed.append(family)
    choices = tuple(tuple(values[key] for key in sorted(values)) for values in options)
    return choices, tuple(consumed), tuple(pending)


def joint_choices(choices, limit=BRANCH_LIMIT):
    """All configurations, or no configurations when the complete product is too large."""
    size = math.prod(len(values) for values in choices)
    if size > limit:
        return (), size
    return tuple(product(*choices)), size


def common_site_masks(choices, prepared):
    """Common usable sites and union-minus-intersection losses per input component.

    Every branch keeps the same position/bin/count-cost axes, but only sites usable
    under every alternative may contribute to an across-input comparison.
    """
    from .hierarchy.batches import component_sites

    shared, lost = [], []
    for values in choices:
        masks = [component_sites(component, prepared)[1] for component in values]
        common = np.logical_and.reduce(masks)
        union = np.logical_or.reduce(masks)
        shared.append(common)
        lost.append(int(np.count_nonzero(union & ~common)))
    return tuple(shared), tuple(lost)


def common_masks(choices, prepared, neutral, max_sites, *, usable_masks=None):
    """One informative cohort from the shared usable-site proxy of each component.

    Intersecting cohorts chosen on different site sets is insufficient: branch
    likelihoods must also read the same genomic evidence domain. Shared proxies
    must additionally preserve eligibility under every original alternative.
    """
    from .hierarchy.batches import informative_masks

    if usable_masks is None:
        usable_masks, _ = common_site_masks(choices, prepared)
    shared = informative_masks(
        tuple(values[0] for values in choices), prepared, neutral, max_sites, usable_masks=usable_masks
    )
    # Restriding the shared site set can expose a sample absent from an original
    # proxy cohort. Preserve eligibility under every original alternative too.
    flat = tuple(component for values in choices for component in values)
    original = informative_masks(flat, prepared, neutral, max_sites)
    cursor = 0
    for index, values in enumerate(choices):
        shared[index] &= np.logical_and.reduce(original[cursor : cursor + len(values)])
        cursor += len(values)
    return shared


def terminal_winners(
    results,
    *,
    objective="join_ll",
    value=lambda result: result.log_likelihood,
    rows=lambda result: result.rows,
    reverse: bool = False,
):
    """Keep the best common-objective class separately for each supported count.

    Counts reached through different complete continuations are not compared
    by unpenalized likelihood. All surviving counts remain explicit.
    """
    by_count: dict[int, list[Any]] = {}
    for result in results:
        by_count.setdefault(len(rows(result)), []).append(result)
    winners: list[Any] = []
    for group in by_count.values():
        values = [float(value(result)) for result in group]
        scale = max(abs(v) for v in values)
        classes = ties.tie_classes(
            values,
            ties.quantum(objective, scale),
            motion=ties.motion(objective, scale),
            content=[panel_key(rows(result)) for result in group],
            descending=True,
            reverse=reverse,
        )
        members: np.ndarray | list[int] = classes.members(int(classes.classes[classes.winner]))
        if float(classes.spans[int(classes.classes[classes.winner])]) > WIDE_CLASS_QUANTA * classes.quantum:
            members = [classes.winner]
        winners.extend(group[int(index)] for index in members)
    distinct: dict[tuple, Any] = {}
    for result in winners:
        key = panel_key(rows(result))
        if key not in distinct or value(result) > value(distinct[key]):
            distinct[key] = result
    return tuple(distinct[key] for key in sorted(distinct))
