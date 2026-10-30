"""The sibling-as-parent guard (release 3.2): released edges whose parent looks like a full sib of the child's family.

For a released Tier-B edge p -> c, F = c's full-sib family in the relatedness product (members with support at least
``member_support``; c itself must reach it) without p and without any member released as a parent or child of c or of
another member (no full sib is anyone's parent or child). With |F| >= ``minimum_family``,

    r = median_{s in F} LR0(p, s) / median_{s < t in F} LR0(s, t),

LR0 being a pair's genome-wide evidence for stretches without identity by descent (relatedness.records). A parent of
c is a parent of all c's full sibs and shares no IBD0 stretch with any of them (r near 0); a full sib of c is a full
sib of them all and exchangeable with them (r near 1); an unrelated or distant candidate has more IBD0 than the sibs
(r above 1). The edge is withheld when lower <= r <= upper; otherwise it is kept (tested) or untested (family too
small or unsupported). The guard reads only the relatedness product (genotype likelihoods) and the released edges.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .config import SiblingGuardConfig
from .records import NO_PARENT, UNRESOLVED, ParentEdges, PedigreeCalls, PedigreeProduct

WITHHELD, KEPT, UNTESTED = "withheld", "kept", "untested"
STATUS = "tier_b_sibling_guard_withheld"
STATUS_A = "tier_a_sibling_guard_withheld"


def sibling_guard(edges: ParentEdges, relatedness, config: SiblingGuardConfig) -> pd.DataFrame:
    """One row per released edge (in edge order): Parent, Child, FamilySize (|F|), Ratio (NaN when untested),
    Call (withheld / kept / untested)."""
    if tuple(relatedness.sample_ids) != tuple(edges.sample_ids):
        raise ValueError("the relatedness product's sample axis differs from the pedigree's")
    lr0 = relatedness.pair_matrix(relatedness.lr_ibd0)
    released = set(zip(edges.parents.tolist(), edges.children.tolist()))
    rows = []
    for parent, child in zip(edges.parents.tolist(), edges.children.tolist()):
        family = relatedness.supported_family(child, config.member_support)
        family = family[family != parent]
        # A member released as a parent or child of the child or of another member is no full sib of the family:
        # it leaves F (a parent-offspring pair inside F would lower the family's own level and inflate r).
        linked = {int(m) for m in family if m != child and any(
            (int(m), int(x)) in released or (int(x), int(m)) in released for x in family if x != m)}
        family = np.asarray([m for m in family if int(m) not in linked], dtype=np.int64)
        ratio, call = np.nan, UNTESTED
        if len(family) >= config.minimum_family:
            outer = float(np.nanmedian(lr0[parent, family]))
            inner = float(np.nanmedian(lr0[np.ix_(family, family)][np.triu_indices(len(family), k=1)]))
            if inner > 0:
                ratio = outer / inner
                call = WITHHELD if config.lower <= ratio <= config.upper else KEPT
        rows.append(dict(Parent=edges.sample_ids[parent], Child=edges.sample_ids[child], FamilySize=len(family),
                         Ratio=ratio, Call=call))
    return pd.DataFrame(rows, columns=["Parent", "Child", "FamilySize", "Ratio", "Call"])


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
    summary = dict(model="ibd0-ratio-v1", edges=int(len(table)), tested=int(np.count_nonzero(calls != UNTESTED)),
                   withheld=int(np.count_nonzero(calls == WITHHELD)))
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
