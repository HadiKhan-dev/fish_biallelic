"""The sibling-as-parent guard: released edges whose parent is Mendelian-excluded
from most of the child's full sibs.

A parent of c is a parent of every full sib of c, so it is Mendelian-compatible with each of them. A relative released
as c's parent (a sib, grandparent or half sib that happens to be compatible with c itself) is excluded from most of
c's other sibs. For a released Tier-B edge p -> c (``ParentEdges`` before the guard), the members tested are:

* c's full-sib family in the relatedness product: the members with support at least ``member_support`` (c itself must
  reach it, else the family is c alone), without p;
* without any member released as a parent or child of c or of another member (no full sib is anyone's parent or child:
  ``linked``);
* without any member with a released parent that is neither p nor a released parent of c (the ``half_sib``
  heuristic: incomplete or wrong parent lists can also remove genuine full sibs);
* without c itself.

The edge is tested when at least ``minimum_family`` - 1 members remain (|F| >= minimum_family counting c), and withheld
when a strict majority of them are Mendelian-excluded from p: full-marker exclusion log E(p, s) >= log(2N/alpha), the
evidence and threshold of the stage's count release (pedigree.exclusion, ExclusionConfig.alpha). Otherwise it is kept
(tested) or untested. The guard reads the relatedness product (genotype likelihoods), the released edges and the
exclusion evidence of the member dyads (genotype likelihoods); no painting, label or metadata.

Cleaning uses only edges that remain. ``fixed_point`` repeatedly withdraws majority-vetoed edges, then reports
all votes against the final edge set. Withdrawals are irreversible: a withdrawn edge whose final vote no longer
vetoes is an explicit abstention, not a certified majority. With E initial edges, at most E changing passes and one
unchanged pass are possible. Family support does not guarantee that most witnesses are actual offspring of p;
an unsequenced shared co-parent can produce a wrong-member majority and veto a true edge. Absence of a released
edge is not absence of a biological parent. The per-dyad exclusion threshold does not bound this family error.

Method: docs/methods/pedigree.md#sibling-as-parent-guard.
"""

from __future__ import annotations

import numpy as np

from ...contracts.pedigree import NO_PARENT, UNRESOLVED, EdgeFamily, GuardRecord, GuardVote, ParentEdges, PedigreeCalls
from ...core.errors import InputError, InvariantError
from . import exclusion
from .parameters import PedigreeConfig

__all__ = ["edge_families", "guard_dyads", "decide", "withheld_edges", "fixed_point", "apply_calls"]

WITHHELD, KEPT, UNTESTED = "withheld", "kept", "untested"
LINKED, HALF_SIB = "linked", "half_sib"
STATUS = "tier_b_sibling_guard_withheld"
STATUS_A = "tier_a_sibling_guard_withheld"
COLUMNS = [
    "Parent",
    "Child",
    "SupportedMembers",
    "DroppedLinked",
    "DroppedHalfSib",
    "DroppedMembers",
    "TestedMembers",
    "ExcludedMembers",
    "Call",
]


def edge_families(
    sample_ids, parents, children, relatedness, config: PedigreeConfig, *, cleaning_edges=None
) -> list[EdgeFamily]:
    """The members of every released edge p -> c (``parents``/``children``: the released edges before the guard, on
    the axis ``sample_ids``). ``cleaning_edges`` supplies the surviving premises, independently of which edges
    are being reported; omitted only for a single-pass diagnostic."""
    if tuple(relatedness.sample_ids) != tuple(sample_ids):
        raise InputError("the relatedness product's sample axis differs from the pedigree's")
    guard = config.guard
    released = (
        set(zip(np.asarray(parents).tolist(), np.asarray(children).tolist(), strict=False))
        if cleaning_edges is None
        else set(cleaning_edges)
    )
    released_parents: dict[int, set[int]] = {}
    for parent, child in released:
        released_parents.setdefault(child, set()).add(parent)
    out = []
    for parent, child in zip(np.asarray(parents).tolist(), np.asarray(children).tolist(), strict=False):
        family = [int(m) for m in relatedness.supported_family(child, guard.member_support) if int(m) != parent]
        # A member released as a parent or child of the child or of another member is no full sib of the family.
        linked = {
            m for m in family if m != child and any((m, x) in released or (x, m) in released for x in family if x != m)
        }
        # A different released parent is a half-sib heuristic, not proof: parent sets can be incomplete or wrong.
        allowed = released_parents.get(child, set()) | {parent}
        half_sib = {
            m for m in family if m != child and m not in linked and not released_parents.get(m, set()) <= allowed
        }
        dropped = tuple((m, LINKED) for m in family if m in linked) + tuple(
            (m, HALF_SIB) for m in family if m in half_sib
        )
        tested = tuple(m for m in family if m != child and m not in linked and m not in half_sib)
        out.append(EdgeFamily(parent, child, tuple(m for m in family if m != child), dropped, tested))
    return out


def guard_dyads(families: list[EdgeFamily], config: PedigreeConfig, *, all_supported=False) -> np.ndarray:
    """int64 [D, 2] (member, parent) dyads the guard needs exclusion evidence for (sorted, unique): the tested members
    of every edge with enough of them. ``all_supported`` scores the superset needed by fixed-point cleaning:
    a dropped member can return after its cleaning edge is withdrawn."""
    dyads = {
        (s, f.parent)
        for f in families
        if len(f.supported if all_supported else f.tested) >= config.guard.minimum_family - 1
        for s in (f.supported if all_supported else f.tested)
    }
    return np.asarray(sorted(dyads), dtype=np.int64).reshape(-1, 2)


def decide(
    families: list[EdgeFamily],
    dyads: np.ndarray,
    log_e: np.ndarray,
    log_e_upper: np.ndarray | None,
    config: PedigreeConfig,
    sample_count: int,
) -> list[tuple[float, str]]:
    """(excluded members, NaN when untested; call) per family. ``dyads``/``log_e``/``log_e_upper``: the exclusion
    evidence of (at least) guard_dyads(families) as (child, candidate) rows of any order; a member is excluded from the
    parent when log E >= log(2N/alpha), the threshold of the count release (N = ``sample_count``)."""
    threshold = exclusion.log_threshold(sample_count, config.exclusion.alpha)
    evidence = {}
    for k, (a, b) in enumerate(np.asarray(dyads, dtype=np.int64).reshape(-1, 2).tolist()):
        lower = float(log_e[k])
        upper = lower if log_e_upper is None else float(log_e_upper[k])
        if lower < threshold <= upper and lower != upper:
            raise InputError("unresolved exclusion interval requires exact refinement")
        evidence[(a, b)] = evidence[(b, a)] = lower >= threshold
    out = []
    for f in families:
        if len(f.tested) < config.guard.minimum_family - 1:
            out.append((np.nan, UNTESTED))
            continue
        excluded = sum(evidence[(s, f.parent)] for s in f.tested)
        out.append((excluded, WITHHELD if 2 * excluded > len(f.tested) else KEPT))
    return out


def withheld_edges(families: list[EdgeFamily], decisions: list[tuple[float, str]]) -> frozenset:
    """The (parent, child) edges the guard withholds."""
    return frozenset((f.parent, f.child) for f, (_, call) in zip(families, decisions, strict=False) if call == WITHHELD)


def _unresolve(calls: PedigreeCalls, children, status_name: str = STATUS) -> PedigreeCalls:
    """The calls of ``children`` (sample indices with a withheld edge) become unresolved, named or count-only: the
    evidence that released the withheld edge also decided the child's count."""
    state, parents, status = calls.state.copy(), calls.parents.copy(), list(calls.status)
    for child in children:
        state[child] = UNRESOLVED
        parents[child] = NO_PARENT
        status[child] = status_name
    return PedigreeCalls(state, parents, tuple(status))


def fixed_point(
    edges: ParentEdges,
    relatedness,
    dyads: np.ndarray,
    log_e: np.ndarray,
    log_e_upper: np.ndarray | None,
    config: PedigreeConfig,
) -> GuardRecord:
    """Contract the retained edge set in at most E+1 passes; withdrawals never reverse.

    Final votes describe the final premises. A prior veto that no longer has a
    majority remains an explicit abstention. Inputs are not mutated; a straddling
    exclusion interval raises instead of inventing evidence (P0 PD23).
    """
    initial = set(zip(edges.parents.tolist(), edges.children.tolist(), strict=False))
    retained = initial.copy()
    withdrawn: dict[tuple[int, int], int] = {}
    for iteration in range(1, len(initial) + 2):
        families = edge_families(
            edges.sample_ids, edges.parents, edges.children, relatedness, config, cleaning_edges=retained
        )
        decisions = decide(families, dyads, log_e, log_e_upper, config, len(edges.sample_ids))
        removed = retained & withheld_edges(families, decisions)
        if not removed:
            break
        withdrawn.update((edge, iteration) for edge in removed)
        retained.difference_update(removed)
    else:
        raise InvariantError("sibling guard did not converge within its E+1 pass bound")
    votes = tuple(
        GuardVote(
            family,
            excluded,
            WITHHELD if (family.parent, family.child) in withdrawn else call,
            call,
            withdrawn.get((family.parent, family.child), 0),
            ("final_majority" if call == WITHHELD else "prior_veto_abstention")
            if (family.parent, family.child) in withdrawn
            else "none",
        )
        for family, (excluded, call) in zip(families, decisions, strict=False)
    )
    return GuardRecord(votes, iteration, True, len(initial) + 1)


def apply_calls(
    calls: PedigreeCalls, tier_a: PedigreeCalls, edges: ParentEdges, guard: GuardRecord
) -> tuple[PedigreeCalls, PedigreeCalls, ParentEdges]:
    """Withdraw guard-vetoed Tier-B edges and the affected Tier-A/B count calls.

    Surviving marginal edges remain; affected count bounds become [survivors, 2].
    The complete view is not a released view and is not changed (P0 PD23).
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
