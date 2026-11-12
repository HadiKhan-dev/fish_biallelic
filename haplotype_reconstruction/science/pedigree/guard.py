"""Painting relationship evidence on every pre-guard Tier-B edge (PD23).

The fixed point withdraws irreversibly, recomputing linked-family cleaning against
surviving edges. Likelihoods are summed over contigs before taking process margins;
all possible witnesses are scored once. Count release and call application retain
those withdrawals.
Method: docs/methods/pedigree.md#sibling-as-parent-guard.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from ...contracts.pedigree import (
    NO_PARENT,
    UNRESOLVED,
    EdgeFamily,
    GuardRecord,
    GuardVote,
    ParentEdges,
    PedigreeCalls,
    RelationshipRequest,
)
from ...core.errors import InputError, InvariantError
from .parameters import PedigreeParameters
from .relationship import evidence

__all__ = ["edge_families", "relationship_request", "decide", "withheld_edges", "fixed_point", "apply_calls"]
WITHHELD, KEPT, UNTESTED = "withheld", "kept", "untested"
LINKED = "linked"
STATUS = "tier_b_sibling_guard_withheld"
STATUS_A = "tier_a_sibling_guard_withheld"


def edge_families(sample_ids, parents, children, relatedness, config: PedigreeParameters, *, cleaning_edges=None):
    """Whole relatedness families, without the proposed parent or linked members."""
    if tuple(relatedness.sample_ids) != tuple(sample_ids):
        raise InputError("the relatedness product's sample axis differs from the pedigree's")
    released = (
        set(zip(np.asarray(parents).tolist(), np.asarray(children).tolist(), strict=True))
        if cleaning_edges is None
        else set(cleaning_edges)
    )
    adjacent: dict[int, set[int]] = {}
    for parent, child in released:
        adjacent.setdefault(parent, set()).add(child)
        adjacent.setdefault(child, set()).add(parent)
    out = []
    for parent, child in zip(np.asarray(parents).tolist(), np.asarray(children).tolist(), strict=True):
        members = (
            np.flatnonzero(relatedness.family == relatedness.family[child])
            if relatedness.family[child] >= 0
            else [child]
        )
        family = sorted((int(m) for m in members if int(m) != parent), key=lambda m: sample_ids[m])
        family_set = set(family)
        linked = {m for m in family if m != child and adjacent.get(m, set()) & family_set}
        supported = tuple(m for m in family if m != child)
        out.append(
            EdgeFamily(
                parent,
                child,
                supported,
                tuple((m, LINKED) for m in supported if m in linked),
                tuple(m for m in supported if m not in linked),
                tuple(sorted((int(m) for m in members if int(m) != child), key=lambda m: sample_ids[m])),
            )
        )
    return out


def check_partner_map(family: EdgeFamily, limit: int, *, current=True):
    """Next P on the fixed family ring, then filter by the current witness set.

    A proposed parent inside the family keeps its ring position. Filtering never
    advances the window to replace absent members. Build indices once per edge.
    """
    ring = family.ring or family.supported
    positions = {member: i for i, member in enumerate(ring)}
    members = family.tested if current else family.supported
    allowed = set(members)
    return {
        w: tuple(
            ring[(positions[w] + offset) % len(ring)]
            for offset in range(1, min(limit, len(ring) - 1) + 1)
            if ring[(positions[w] + offset) % len(ring)] in allowed
        )
        for w in members
    }


def relationship_request(families, sample_ids, contigs, config: PedigreeParameters | None = None):
    """Score the original pair and every original family member, including linked ones.

    Cleaning can admit a linked member after a premise is withdrawn. Requesting
    the whole original family covers both returning and removed witnesses.
    """
    parameters = (config or PedigreeParameters()).relationship
    dyads, trios = set(), set()
    for f in families:
        dyads.add(tuple(sorted((f.child, f.parent))))
        partners = check_partner_map(f, parameters.sibship_check_partners, current=False)
        for w in f.supported:
            dyads.add(tuple(sorted((w, f.parent))))
            trios.add(tuple(sorted((f.child, f.parent, w))))
            for other in partners[w]:
                trios.add(tuple(sorted((f.child, w, other))))
    return RelationshipRequest(
        tuple(sample_ids),
        tuple(contigs),
        np.array(sorted(dyads), dtype=np.int64).reshape(-1, 2),
        np.array(sorted(trios), dtype=np.int64).reshape(-1, 3),
    )


def summed_margins(request, records):
    """Sum each process over chromosomes before forming evidence against feasibility."""
    pair = np.zeros((len(request.dyads), 4))
    trio = np.zeros((len(request.trios), 4))
    if len(records) != (len(request.contigs) if len(request.dyads) else 0):
        raise InputError("relationship evidence must cover configured contigs in order")
    for c, record in zip(request.contigs, records, strict=bool(records)):
        if (
            record.contig != c
            or record.sample_ids != request.sample_ids
            or not np.array_equal(record.dyads, request.dyads)
            or not np.array_equal(record.trios, request.trios)
        ):
            raise InputError("relationship evidence differs from its frozen request")
        pair += record.pair_log_likelihoods
        trio += record.trio_log_likelihoods
    return (
        dict(zip(map(tuple, request.dyads.tolist()), evidence(pair), strict=True)),
        dict(zip(map(tuple, request.trios.tolist()), evidence(trio), strict=True)),
    )


def decide(families, pair_margins, trio_margins, config: PedigreeParameters, member_support):
    """Profile likelihood score = pair margin plus checked witness contributions, in nats."""
    p = config.relationship
    threshold = p.withholding_score
    out = []
    supports = np.nan_to_num(member_support, nan=0.0)
    for f in families:
        margin = float(pair_margins[tuple(sorted((f.child, f.parent)))])
        contribution = 0.0
        set_aside = against = supporting = 0
        partners = check_partner_map(f, p.sibship_check_partners)
        for w in f.tested:
            dpar = float(pair_margins[tuple(sorted((w, f.parent)))])
            dsib = float(trio_margins[tuple(sorted((f.child, f.parent, w)))])
            support = min(supports[f.child], supports[w])
            band = int(support >= p.support_bounds[0]) + int(support >= p.support_bounds[1])
            prior = p.sibship_prior[band]
            price = max(0.0, math.log(prior / (1.0 - prior)))
            others = partners[w]
            family_contradicts = (
                bool(others)
                and float(np.median([trio_margins[tuple(sorted((f.child, w, other)))] for other in others])) <= -price
            )
            v = 0.0 if family_contradicts else max(dpar, -price) - max(dsib, -price)
            contribution += v
            set_aside += int(family_contradicts or (dpar <= -price and dsib <= -price))
            against += int(v < -1.0)
            supporting += int(v > 1.0)
        total = margin + contribution
        call = WITHHELD if total < -threshold else (UNTESTED if not f.tested and margin == 0.0 else KEPT)
        basis = "family" if total > threshold else "pair_only"
        withholding = ("pair" if margin < -threshold else "family") if call == WITHHELD else "none"
        out.append(
            GuardVote(
                f,
                set_aside,
                against,
                supporting,
                margin,
                contribution,
                total,
                "" if call == WITHHELD else basis,
                call,
                call,
                0,
                withholding,
            )
        )
    return out


def withheld_edges(families, decisions):
    """The (parent, child) edges the relationship evidence withholds."""
    return frozenset((f.parent, f.child) for f, v in zip(families, decisions, strict=True) if v.call == WITHHELD)


def _unresolve(calls: PedigreeCalls, children, status_name: str = STATUS) -> PedigreeCalls:
    """The calls of ``children`` (sample indices with a withheld edge) become unresolved, named or count-only: the
    evidence that released the withheld edge also decided the child's count."""
    state, parents, status = calls.state.copy(), calls.parents.copy(), list(calls.status)
    for child in children:
        state[child] = UNRESOLVED
        parents[child] = NO_PARENT
        status[child] = status_name
    return PedigreeCalls(state, parents, tuple(status))


def fixed_point(edges, relatedness, request, records, config: PedigreeParameters):
    """Contract retained edges in at most E+1 passes, with irreversible withdrawals."""
    pair, trio = summed_margins(request, records)
    initial = set(zip(edges.parents.tolist(), edges.children.tolist(), strict=True))
    retained = initial.copy()
    withdrawn = {}
    for iteration in range(1, len(initial) + 2):
        families = edge_families(
            edges.sample_ids, edges.parents, edges.children, relatedness, config, cleaning_edges=retained
        )
        decisions = decide(families, pair, trio, config, relatedness.member_support)
        removed = retained & withheld_edges(families, decisions)
        if not removed:
            break
        for vote in decisions:
            key = (vote.family.parent, vote.family.child)
            if key in removed:
                withdrawn[key] = (iteration, vote.withholding_basis)
        retained.difference_update(removed)
    else:
        raise InvariantError("sibling guard did not converge within its E+1 pass bound")
    votes = []
    for vote in decisions:
        key = (vote.family.parent, vote.family.child)
        if key in withdrawn:
            number, basis = withdrawn[key]
            vote = replace(
                vote, call=WITHHELD, relationship_basis="", withdrawal_iteration=number, withholding_basis=basis
            )
        votes.append(vote)
    return GuardRecord(tuple(votes), iteration, True, len(initial) + 1)


def apply_calls(
    calls: PedigreeCalls, tier_a: PedigreeCalls, edges: ParentEdges, guard: GuardRecord
) -> tuple[PedigreeCalls, PedigreeCalls, ParentEdges]:
    """Withdraw guard-vetoed Tier-B edges and the affected Tier-A/B count calls.

    Surviving marginal edges remain; affected count bounds become [survivors, 2].
    The complete view is not a released view and is not changed (PD23).
    """
    keep = np.asarray([vote.call != WITHHELD for vote in guard.votes], dtype=np.bool_)
    if np.all(keep):
        return calls, tier_a, edges
    children = sorted({int(child) for child in edges.children[~keep]})
    minimum, maximum = edges.observed_parent_minimum.copy(), edges.observed_parent_maximum.copy()
    retained = np.bincount(edges.children[keep], minlength=len(minimum))
    for child in children:
        minimum[child] = retained[child]
        maximum[child] = 2
    guarded = ParentEdges(
        edges.sample_ids, edges.parents[keep], edges.children[keep], edges.slots[keep], minimum, maximum
    )
    return _unresolve(calls, children), _unresolve(tier_a, children, STATUS_A), guarded
