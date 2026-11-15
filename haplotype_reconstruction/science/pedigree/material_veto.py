"""Release only what two material regimes both release: the both-maps veto (DESIGN_RELEASE section 6, review3 MINOR 7).

The working map and the estimated map can put the run's material in different regimes. On the both-maps route the
lower regime's rule runs the whole pedigree and the higher regime's rule runs to its released calls (``VetoSet``);
the product then releases a Tier B call only when the veto regime releases the same call (same observed-parent count
and the same parents), and an edge only when the veto regime releases it too. The intersection is computed
explicitly: low-material Tier B is not nested in standard Tier B (minimum units 1 against 3, bootstrap cut-offs
0.95 against the released ones), so neither regime's rule alone is the intersection. A withheld call becomes
unresolved with status ``tier_b_material_veto``; edges released by both survive as marginal edges, as after the
sibling guard. Tier A: the applied (lower) regime is never standard when a veto runs, so no Tier A call exists.

The report counts what the product withholds (and adds) relative to the working map alone (``compare``, ``report``):
under both_maps the working map's release is the veto chain's (estimate lower) or the applied chain's before the
veto (estimate higher); on the estimate route a reference chain of the working regime runs to its released calls
only to be counted (``VetoSet.role`` "reference").

Method: docs/methods/pedigree.md#map-scale.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ...contracts.pedigree import NO_PARENT, UNRESOLVED, ParentEdges, PedigreeCalls, ReleasedDecision

__all__ = ["VetoSet", "VetoOutcome", "Comparison", "veto_set", "empty_set", "apply", "apply_views", "compare",
           "report", "STATUS_B", "STATUS_A", "EVIDENCE"]

STATUS_B = "tier_b_material_veto"
STATUS_A = "tier_a_material_veto"
EVIDENCE = "material_veto"


@dataclass(frozen=True)
class VetoSet:
    """A regime's released Tier B calls (state int8 [N], parents int64 [N, 2]), Tier B edges (int64 [E, 2] of
    parent, child) and Tier A calls (tier_a_state, tier_a_parents). role "veto": the both-maps veto regime, applied as
    a veto; "reference": the working map's regime, only counted against the product."""

    regime: str
    state: np.ndarray
    parents: np.ndarray
    edges: np.ndarray
    tier_a_state: np.ndarray
    tier_a_parents: np.ndarray
    role: str = "veto"


@dataclass(frozen=True)
class VetoOutcome:
    """Children whose Tier B call the veto withheld and the (parent, child) edges it withheld."""

    regime: str
    children: frozenset[int]
    edges: frozenset[tuple[int, int]]
    tier_a: int


def veto_set(released: ReleasedDecision, regime: str) -> VetoSet:
    """The released Tier B calls and edges and the Tier A calls of a regime's chain (small: one row per sample and
    edge)."""
    calls, edges = released.calls, released.edges
    pairs = np.stack([np.asarray(edges.parents, np.int64), np.asarray(edges.children, np.int64)], axis=1)
    return VetoSet(regime, np.array(calls.state, dtype=np.int8), np.array(calls.parents, dtype=np.int64), pairs,
                   np.array(released.tier_a.state, dtype=np.int8), np.array(released.tier_a.parents, dtype=np.int64))


def empty_set(samples: int, regime: str) -> VetoSet:
    """A regime that releases nothing (not assessable)."""
    state, parents = np.full(samples, UNRESOLVED, np.int8), np.full((samples, 2), NO_PARENT, np.int64)
    return VetoSet(regime, state, parents, np.empty((0, 2), np.int64), state.copy(), parents.copy(), "reference")


def _same_call(state_a, parents_a, state_b, parents_b) -> bool:
    return int(state_a) == int(state_b) and sorted(int(p) for p in parents_a) == sorted(int(p) for p in parents_b)


def _unresolve(calls: PedigreeCalls, children, status: str) -> PedigreeCalls:
    state, parents, statuses = calls.state.copy(), calls.parents.copy(), list(calls.status)
    for child in children:
        state[child] = UNRESOLVED
        parents[child] = NO_PARENT
        statuses[child] = status
    return PedigreeCalls(state, parents, tuple(statuses))


def apply(released: ReleasedDecision, veto: VetoSet) -> tuple[ReleasedDecision, VetoOutcome]:
    """Withhold every Tier B call and edge of ``released`` that the veto regime does not release identically."""
    calls, edges = released.calls, released.edges
    if veto.state.shape != calls.state.shape:
        raise ValueError("the veto regime's calls follow another sample axis")
    allowed = set(zip(veto.edges[:, 0].tolist(), veto.edges[:, 1].tolist(), strict=True))
    children = set()
    for child in np.flatnonzero(calls.state != UNRESOLVED):
        if not _same_call(calls.state[child], calls.parents[child], veto.state[child], veto.parents[child]):
            children.add(int(child))
    keep = np.asarray([(int(p), int(c)) in allowed for p, c in zip(edges.parents, edges.children, strict=True)],
                      dtype=np.bool_)
    withheld = frozenset((int(p), int(c)) for p, c, k in zip(edges.parents, edges.children, keep, strict=True)
                         if not k)
    children.update(c for _, c in withheld)
    minimum, maximum = edges.observed_parent_minimum.copy(), edges.observed_parent_maximum.copy()
    retained = np.bincount(np.asarray(edges.children)[keep], minlength=len(minimum))
    for child in children:
        minimum[child] = retained[child]
        maximum[child] = 2
    vetoed = ParentEdges(edges.sample_ids, edges.parents[keep], edges.children[keep], edges.slots[keep], minimum,
                         maximum)
    tier_a_released = [c for c in children if released.tier_a.state[c] != UNRESOLVED]
    outcome = VetoOutcome(veto.regime, frozenset(children), withheld, len(tier_a_released))
    if not children:
        return released, outcome
    return replace(
        released,
        calls=_unresolve(calls, sorted(children), STATUS_B),
        tier_a=_unresolve(released.tier_a, sorted(tier_a_released), STATUS_A),
        edges=vetoed,
    ), outcome


@dataclass(frozen=True)
class Comparison:
    """Tier B calls, Tier B edges and Tier A calls the reference regime releases and the product does not release
    identically (withheld), and the reverse (added)."""

    withheld_calls: int
    withheld_edges: int
    withheld_tier_a: int
    added_calls: int
    added_edges: int
    added_tier_a: int


def _differing(state_a, parents_a, state_b, parents_b) -> int:
    """Calls released in a (state not unresolved) that b does not release identically."""
    return sum(1 for child in np.flatnonzero(np.asarray(state_a) != UNRESOLVED)
               if not _same_call(state_a[child], parents_a[child], state_b[child], parents_b[child]))


def compare(reference: VetoSet, product: VetoSet) -> Comparison:
    """What ``product`` withholds and adds relative to ``reference`` (both on one sample axis)."""
    if reference.state.shape != product.state.shape:
        raise ValueError("the compared regimes follow different sample axes")
    ref_edges = set(map(tuple, reference.edges.tolist()))
    out_edges = set(map(tuple, product.edges.tolist()))
    return Comparison(
        _differing(reference.state, reference.parents, product.state, product.parents),
        len(ref_edges - out_edges),
        _differing(reference.tier_a_state, reference.tier_a_parents, product.tier_a_state, product.tier_a_parents),
        _differing(product.state, product.parents, reference.state, reference.parents),
        len(out_edges - ref_edges),
        _differing(product.tier_a_state, product.tier_a_parents, reference.tier_a_state, reference.tier_a_parents),
    )


NOT_COMPUTED = Comparison(-1, -1, -1, -1, -1, -1)


def report(choice, comparison: Comparison, outcome: VetoOutcome | None = None):
    """The material choice with its counts relative to the working map alone and the statement completed."""
    vetoed = (0, 0, 0) if outcome is None else (len(outcome.children), len(outcome.edges), outcome.tier_a)
    c = comparison
    if c.withheld_calls < 0:
        words = (f"; the working map alone applies the {choice.working_regime} regime, whose releases are not computed "
                 f"(the product is not assessable and releases no call)")
    else:
        words = (f"; relative to the working map alone ({choice.working_regime}) the product withholds "
                 f"{c.withheld_calls} Tier B calls, {c.withheld_edges} Tier B edges and {c.withheld_tier_a} Tier A "
                 f"calls" + (f" ({vetoed[0]} Tier B calls by the veto itself)" if vetoed[0] else ""))
        if c.added_calls or c.added_edges or c.added_tier_a:
            words += (f", and adds {c.added_calls} Tier B calls, {c.added_edges} Tier B edges and {c.added_tier_a} "
                      f"Tier A calls")
    head, sep, warning = choice.statement.partition("; WARNING:")
    return replace(
        choice,
        withheld_calls=c.withheld_calls, withheld_edges=c.withheld_edges, withheld_tier_a=c.withheld_tier_a,
        added_calls=c.added_calls, added_edges=c.added_edges, added_tier_a=c.added_tier_a,
        vetoed_calls=vetoed[0], vetoed_edges=vetoed[1], vetoed_tier_a=vetoed[2],
        statement=head + words + sep + warning,
    )


def apply_views(views, outcome: VetoOutcome):
    """Withdraw the vetoed Tier B calls and edges from the report views, keeping candidate lists and pre-veto parents
    (``parents_before_guard`` records the parents before the veto, as before the sibling guard)."""
    if not outcome.children:
        return views
    children, withheld = outcome.children, outcome.edges

    def relationship(row):
        if row.sample not in children:
            return row
        return replace(row, parent1=NO_PARENT, parent2=NO_PARENT, relationship_basis="",
                       parents_before_guard=(row.parent1, row.parent2), parent_state=UNRESOLVED,
                       observed_parent_count=float("nan"), inference_status=STATUS_B, parent_slot_basis="none",
                       release_evidence=EVIDENCE if row.release_evidence is not None else None)

    def partial(row, status=STATUS_B):
        if row.sample not in children:
            return row
        first = (row.parent1, row.sample) in withheld
        second = (row.parent2, row.sample) in withheld
        parent1, parent2 = NO_PARENT if first else row.parent1, NO_PARENT if second else row.parent2
        supported1, supported2 = row.parent1_supported and not first, row.parent2_supported and not second
        changes = dict(parent1=parent1, parent2=parent2, parents_before_guard=(row.parent1, row.parent2),
                       parent1_supported=supported1, parent2_supported=supported2,
                       observed_parent_count_minimum=int(supported1 and parent1 >= 0) + int(supported2 and parent2 >= 0),
                       observed_parent_count_maximum=2, exact_configuration_resolved=False, parent_state=UNRESOLVED,
                       observed_parent_count=float("nan"), inference_status=status)
        if hasattr(row, "parent_slot_basis"):
            changes["parent_slot_basis"] = "none" if parent1 < 0 and parent2 < 0 else row.parent_slot_basis
        return replace(row, **changes)

    return replace(
        views,
        tier_b_relationships=tuple(relationship(r) for r in views.tier_b_relationships),
        tier_b_partial_relationships=tuple(partial(r) for r in views.tier_b_partial_relationships),
        tier_b_candidate_sets=tuple(partial(r) for r in views.tier_b_candidate_sets),
        parent_state_calls=tuple(
            replace(r, tier_b_final_state_call=False, tier_b_release_evidence=EVIDENCE) if r.sample in children else r
            for r in views.parent_state_calls
        ),
    )
