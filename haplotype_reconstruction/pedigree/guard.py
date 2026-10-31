"""The sibling-as-parent guard (release 3.4: family consistency): released edges whose parent is Mendelian-excluded
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
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .config import PedigreeConfig
from .records import NO_PARENT, UNRESOLVED, ParentEdges, PedigreeCalls, PedigreeProduct

WITHHELD, KEPT, UNTESTED = "withheld", "kept", "untested"
LINKED, HALF_SIB = "linked", "half_sib"
STATUS = "tier_b_sibling_guard_withheld"
STATUS_A = "tier_a_sibling_guard_withheld"
COLUMNS = ["Parent", "Child", "SupportedMembers", "DroppedLinked", "DroppedHalfSib", "DroppedMembers",
           "TestedMembers", "ExcludedMembers", "Call"]


@dataclass(frozen=True)
class EdgeFamily:
    """The members of one released edge's test (sample indices): ``supported`` = c's supported family without c and
    p; ``dropped`` = (member, reason) pairs; ``tested`` = the remaining members."""

    parent: int
    child: int
    supported: tuple[int, ...]
    dropped: tuple[tuple[int, str], ...]
    tested: tuple[int, ...]


def edge_families(sample_ids, parents, children, relatedness, config: PedigreeConfig, *,
                  cleaning_edges=None) -> list[EdgeFamily]:
    """The members of every released edge p -> c (``parents``/``children``: the released edges before the guard, on
    the axis ``sample_ids``). ``cleaning_edges`` supplies the surviving premises, independently of which edges
    are being reported; omitted only for a single-pass diagnostic."""
    if tuple(relatedness.sample_ids) != tuple(sample_ids):
        raise ValueError("the relatedness product's sample axis differs from the pedigree's")
    guard = config.guard
    released = (set(zip(np.asarray(parents).tolist(), np.asarray(children).tolist()))
                if cleaning_edges is None else set(cleaning_edges))
    released_parents: dict[int, set[int]] = {}
    for parent, child in released:
        released_parents.setdefault(child, set()).add(parent)
    out = []
    for parent, child in zip(np.asarray(parents).tolist(), np.asarray(children).tolist()):
        family = [int(m) for m in relatedness.supported_family(child, guard.member_support) if int(m) != parent]
        # A member released as a parent or child of the child or of another member is no full sib of the family.
        linked = {m for m in family if m != child and any(
            (m, x) in released or (x, m) in released for x in family if x != m)}
        # A different released parent is a half-sib heuristic, not proof: parent sets can be incomplete or wrong.
        allowed = released_parents.get(child, set()) | {parent}
        half_sib = {m for m in family if m != child and m not in linked
                    and not released_parents.get(m, set()) <= allowed}
        dropped = tuple((m, LINKED) for m in family if m in linked) + \
            tuple((m, HALF_SIB) for m in family if m in half_sib)
        tested = tuple(m for m in family if m != child and m not in linked and m not in half_sib)
        out.append(EdgeFamily(parent, child, tuple(m for m in family if m != child), dropped, tested))
    return out


def guard_dyads(families: list[EdgeFamily], config: PedigreeConfig, *, all_supported=False) -> np.ndarray:
    """int64 [D, 2] (member, parent) dyads the guard needs exclusion evidence for (sorted, unique): the tested members
    of every edge with enough of them. ``all_supported`` scores the superset needed by fixed-point cleaning:
    a dropped member can return after its cleaning edge is withdrawn."""
    dyads = {(s, f.parent) for f in families
             if len(f.supported if all_supported else f.tested) >= config.guard.minimum_family - 1
             for s in (f.supported if all_supported else f.tested)}
    return np.asarray(sorted(dyads), dtype=np.int64).reshape(-1, 2)


def decide(families: list[EdgeFamily], dyads: np.ndarray, log_e: np.ndarray, log_e_upper: np.ndarray | None,
           config: PedigreeConfig, sample_count: int) -> list[tuple[float, str]]:
    """(excluded members, NaN when untested; call) per family. ``dyads``/``log_e``/``log_e_upper``: the exclusion
    evidence of (at least) guard_dyads(families) as (child, candidate) rows of any order; a member is excluded from the
    parent when log E >= log(2N/alpha), the threshold of the count release (N = ``sample_count``)."""
    threshold = math.log(2 * sample_count / config.exclusion.alpha)
    evidence = {}
    for k, (a, b) in enumerate(np.asarray(dyads, dtype=np.int64).reshape(-1, 2).tolist()):
        lower = float(log_e[k])
        upper = lower if log_e_upper is None else float(log_e_upper[k])
        if lower < threshold <= upper and lower != upper:
            raise ValueError("unresolved exclusion interval requires exact refinement")
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
    return frozenset((f.parent, f.child) for f, (_, call) in zip(families, decisions) if call == WITHHELD)


def sibling_guard(edges: ParentEdges, families: list[EdgeFamily], dyads: np.ndarray, log_e: np.ndarray,
                  log_e_upper: np.ndarray | None, config: PedigreeConfig) -> pd.DataFrame:
    """One row per released edge (in edge order): Parent, Child, SupportedMembers, DroppedLinked, DroppedHalfSib,
    DroppedMembers ("name:reason;..."), TestedMembers, ExcludedMembers (NaN when untested) and Call (withheld / kept /
    untested). ``families`` = edge_families of these edges; the evidence as for ``decide``."""
    names = edges.sample_ids
    if [(f.parent, f.child) for f in families] != list(zip(edges.parents.tolist(), edges.children.tolist())):
        raise ValueError("the guard's families do not match the released edges")
    rows = []
    for f, (excluded, call) in zip(families, decide(families, dyads, log_e, log_e_upper, config, len(names))):
        rows.append(dict(Parent=names[f.parent], Child=names[f.child], SupportedMembers=len(f.supported),
                         DroppedLinked=sum(reason == LINKED for _, reason in f.dropped),
                         DroppedHalfSib=sum(reason == HALF_SIB for _, reason in f.dropped),
                         DroppedMembers=";".join(f"{names[m]}:{reason}" for m, reason in f.dropped),
                         TestedMembers=len(f.tested), ExcludedMembers=excluded, Call=call))
    return pd.DataFrame(rows, columns=COLUMNS)


def fixed_point(edges: ParentEdges, relatedness, dyads: np.ndarray, log_e: np.ndarray,
                log_e_upper: np.ndarray | None, config: PedigreeConfig) -> pd.DataFrame:
    """Remove vetoed edges until the retained set is stable, cleaning only against that set each pass.

    The update R <- R minus veto(R) is contracting, even though the majority vote need not be monotone.
    Therefore it terminates in <= E+1 passes. Previously withdrawn edges stay withheld as abstentions if their
    final vote changes; they are never presented as current majority vetoes. All member/count columns and
    EvidenceCall describe the final premises. WithdrawalIteration records the first veto (0 when retained).
    """
    initial = set(zip(edges.parents.tolist(), edges.children.tolist()))
    retained = initial.copy()
    withdrawn = {}
    for iteration in range(1, len(initial) + 2):
        families = edge_families(edges.sample_ids, edges.parents, edges.children, relatedness, config,
                                 cleaning_edges=retained)
        decisions = decide(families, dyads, log_e, log_e_upper, config, len(edges.sample_ids))
        removed = retained & withheld_edges(families, decisions)
        if not removed:
            break
        withdrawn.update((edge, iteration) for edge in removed)
        retained.difference_update(removed)
    else:
        raise RuntimeError("sibling guard did not converge within its E+1 pass bound")
    table = sibling_guard(edges, families, dyads, log_e, log_e_upper, config)
    table["EvidenceCall"] = table["Call"]
    table["WithdrawalIteration"] = [withdrawn.get((f.parent, f.child), 0) for f in families]
    table["WithholdingBasis"] = [
        ("final_majority" if call == WITHHELD else "prior_veto_abstention")
        if (f.parent, f.child) in withdrawn else "none"
        for f, (_, call) in zip(families, decisions)]
    table.loc[table["WithdrawalIteration"] > 0, "Call"] = WITHHELD
    table.attrs.update(cleaning_iterations=iteration, cleaning_converged=True,
                       cleaning_iteration_bound=len(initial) + 1)
    return table


def _unresolve(calls: PedigreeCalls, children, status_name: str = STATUS) -> PedigreeCalls:
    """The calls of ``children`` (sample indices with a withheld edge) become unresolved, named or count-only: the
    evidence that released the withheld edge also decided the child's count."""
    state, parents, status = calls.state.copy(), calls.parents.copy(), list(calls.status)
    for child in children:
        state[child] = UNRESOLVED
        parents[child] = NO_PARENT
        status[child] = status_name
    return PedigreeCalls(state, parents, tuple(status))


_PARENT_COLUMNS = ("Parent1", "Parent2")


def _named(value) -> bool:
    return isinstance(value, str) and bool(value)


def _guard_frame(frame, withheld_names, *, partial: bool, status: str = STATUS):
    """A published relationship table after the guard; ``withheld_names`` maps each child with a withheld edge to
    its withheld parents.

    The child's row becomes unresolved (state "unresolved", ``status``, no count), whether it names the withheld
    parent or states a count only. In a full view (``partial`` False: the call) it names no parent. A partial view
    (edges with their own support) drops only the withheld parents from their slots (no longer supported); a
    supported other parent stays, the configuration is no longer resolved, and the observed-parent bounds become
    [remaining supported parents, 2]. Release-evidence and Tier-B final-call columns follow when the table has
    them."""
    if frame is None or not withheld_names:
        return frame
    frame = frame.copy()
    for index in frame.index[frame["Sample"].isin(list(withheld_names))]:
        parents = withheld_names[frame.at[index, "Sample"]]
        slots = [column for column in _PARENT_COLUMNS if frame.at[index, column] in parents]
        for column in (slots if partial else _PARENT_COLUMNS):
            frame.at[index, column] = None
        if partial:
            for column in slots:
                frame.at[index, f"{column}Supported"] = False
            frame.at[index, "ObservedParentCountMinimum"] = sum(
                bool(frame.at[index, f"{column}Supported"]) and _named(frame.at[index, column])
                for column in _PARENT_COLUMNS)
            frame.at[index, "ObservedParentCountMaximum"] = 2
            frame.at[index, "ExactConfigurationResolved"] = False
        if not any(_named(frame.at[index, column]) for column in _PARENT_COLUMNS):
            frame.at[index, "ParentSlotBasis"] = "none"
        frame.at[index, "ParentState"] = "unresolved"
        frame.at[index, "ObservedParentCount"] = np.nan
        frame.at[index, "InferenceStatus"] = status
        for column in ("ReleaseEvidence", "TierBReleaseEvidence"):
            if column in frame:
                frame.at[index, column] = "sibling_guard"
        if "TierBFinalStateCall" in frame:
            frame.at[index, "TierBFinalStateCall"] = False
    return frame


def _guard_state_calls(frame, children):
    """parent_state_calls after the guard: the Tier-B final state call of a child with a withheld edge is withdrawn
    (the selection diagnostics stay)."""
    if frame is None or not children:
        return frame
    frame = frame.copy()
    rows = frame["Sample"].isin(list(children))
    frame.loc[rows, "TierBFinalStateCall"] = False
    frame.loc[rows, "TierBReleaseEvidence"] = "sibling_guard"
    return frame


def apply(product: PedigreeProduct, table: pd.DataFrame) -> PedigreeProduct:
    """The product without the withheld edges. They leave ``edges`` (the child's observed-parent minimum becomes its
    remaining released edges and its maximum returns to 2: its count becomes unresolved), and the child's Tier-B and
    Tier-A calls become
    unresolved, named or count-only. Every released view changes together: the calls and edges, the published
    Tier-B and Tier-A relationship tables (full and partial), the Tier-B candidate sets and the Tier-B final state
    calls. The complete (unreleased) view and the selection diagnostics keep the pre-guard decisions. Records the
    guard table in the report and a summary in ``sibling_guard``."""
    calls = table["Call"].to_numpy()
    summary = dict(model="family-exclusion-v2", edges=int(len(table)),
                   tested=int(np.count_nonzero(calls != UNTESTED)),
                   withheld=int(np.count_nonzero(calls == WITHHELD)))
    summary.update(table.attrs)
    if "WithholdingBasis" in table:
        summary["prior_veto_abstentions"] = int(np.count_nonzero(
            table["WithholdingBasis"].to_numpy() == "prior_veto_abstention"))
    report = replace(product.report, sibling_guard=table)
    if not summary["withheld"]:
        return replace(product, report=report, sibling_guard=summary)
    edges = product.edges
    keep = calls != WITHHELD
    withheld = list(zip(edges.parents[~keep].tolist(), edges.children[~keep].tolist()))
    minimum, maximum = edges.observed_parent_minimum.copy(), edges.observed_parent_maximum.copy()
    retained = np.bincount(edges.children[keep], minlength=len(minimum))
    for _, child in withheld:
        minimum[child] = retained[child]     # the remaining released edges; a count-only minimum is withdrawn
        maximum[child] = 2
    guarded = ParentEdges(edges.sample_ids, edges.parents[keep], edges.children[keep], edges.slots[keep],
                          minimum, maximum)
    names = edges.sample_ids
    withheld_names = {}
    for parent, child in withheld:
        withheld_names.setdefault(names[child], set()).add(names[parent])
    children = sorted({child for _, child in withheld})
    report = replace(
        report,
        tier_b_relationships=_guard_frame(report.tier_b_relationships, withheld_names, partial=False),
        tier_b_partial_relationships=_guard_frame(report.tier_b_partial_relationships, withheld_names, partial=True),
        tier_a_relationships=_guard_frame(report.tier_a_relationships, withheld_names, partial=False,
                                          status=STATUS_A),
        tier_a_partial_relationships=_guard_frame(report.tier_a_partial_relationships, withheld_names, partial=True,
                                                  status=STATUS_A),
        tier_b_candidate_sets=_guard_frame(report.tier_b_candidate_sets, withheld_names, partial=True),
        parent_state_calls=_guard_state_calls(report.parent_state_calls, set(withheld_names)))
    return replace(product, edges=guarded, calls=_unresolve(product.calls, children),
                   tier_a=_unresolve(product.tier_a, children, STATUS_A), report=report, sibling_guard=summary)
