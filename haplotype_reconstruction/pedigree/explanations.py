"""Compact, descriptive explanations of the actual full-data pedigree fit.

Scores are composite log evidence, not calibrated probabilities. Identity
scores must not be mistaken for marginal M0/M1/M2 state evidence or DAG utility.
No extra inference, thresholds, or release decisions are introduced here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def explain_calls(samples, alternatives, states, by_child, full_counts, scored_counts,
                  selection, terms, adjusted, diagnostics, candidate_sets):
    records = []
    summaries = []
    for child, rows in enumerate(by_child):
        if not len(rows):
            summaries.append(dict(candidate_sets.iloc[child], Interpretation="excluded as an inference target by caller eligibility"))
            continue
        local = selection.local_rows.get(child)
        graph = selection.graph_rows.get(child)
        chosen = set(row for row in (local, graph) if row is not None)
        ranks = {}
        for state in range(3):
            group = rows[(states[rows] == state) & np.isfinite(adjusted[1][rows])]
            order = group[np.argsort(-adjusted[1][group], kind="stable")]
            chosen.update(map(int, order[:3]))
            ranks.update({int(row): rank + 1 for rank, row in enumerate(order)})
        for row in sorted(chosen):
            state = int(states[row])
            records.append(dict(
                Sample=samples[child], ParentState=f"M{state}",
                Parent1=None if alternatives[row, 1] < 0 else samples[alternatives[row, 1]],
                Parent2=None if alternatives[row, 2] < 0 else samples[alternatives[row, 2]],
                LocalWinner=row == local, GraphWinner=row == graph,
                WithinStateRank=ranks.get(row),
                GeneticLogEvidence=terms["genetic"][row],
                StructuralStateLogEvidence=terms["structural_state"][row],
                StructuralIdentityEligible=bool(np.isfinite(terms["structural_identity"][row])),
                DirectionLogContribution=terms["direction"][row],
                ReciprocalFamilyLogContribution=terms["reciprocal_family"][row],
                AncestryPathLogContribution=terms["ancestry_paths"][row],
                AdjustedStateLogEvidence=adjusted[0][row],
                AdjustedIdentityLogEvidence=adjusted[1][row],
                ParentStateLogPrior=np.log(selection.loo_state_priors[child, state]),
                LogEligibleIdentityMultiplicity=np.log(max(1, full_counts[child, state])),
                MarginalStateLogEvidence=selection.state_log_evidence[child, state],
                GraphDecisionUtility=selection.decision_scores[row]))
        summary = dict(candidate_sets.iloc[child])
        diag = diagnostics.iloc[child]
        for name in ("LocalWinnerParentState", "SelectedParentState", "LocalParent1",
                     "LocalParent2", "GraphConflict", "StateWinnerMargin", "ConditionalIdentityMargin",
                     "LocalStateBootstrapFraction", "LocalStateLOCOFraction"):
            if name in diag:
                summary[name] = diag[name]
        summary.update(
            EligibleM2Pairs=int(full_counts[child, 2]), ScoredM2Pairs=int(scored_counts[child, 2]),
            IncompleteM2Search=bool(scored_counts[child, 2] < full_counts[child, 2]),
            GraphChangedLocalConfiguration=graph != local,
            Interpretation="composite model support; M0 means no observed parent, not founder generation")
        summaries.append(summary)
    return pd.DataFrame(summaries), pd.DataFrame(records)


def parent_search_diagnostics(scored, parent_eligibility, relationships):
    """Report the screened M2 boundary; M1 still scores all eligible parents."""
    from .eligibility import _resolve_parent_eligibility
    eligible = _resolve_parent_eligibility(parent_eligibility, scored.sample_ids)
    rows = []
    ids = {name: i for i, name in enumerate(scored.sample_ids)}
    for child, sample in enumerate(scored.sample_ids):
        parents = np.flatnonzero(eligible.eligible_parents[child])
        order = parents[np.lexsort((parents, -scored.parent_screen_scores[child, parents]))]
        rank = {int(parent): i + 1 for i, parent in enumerate(order)}
        trios = scored.trios[scored.trios[:, 0] == child]
        included = set(map(int, trios[:, 1:].ravel()))
        omitted = [p for p in order if p not in included]
        weakest = min((scored.parent_screen_scores[child, p] for p in included), default=np.nan)
        strongest_omitted = scored.parent_screen_scores[child, omitted[0]] if omitted else np.nan
        boundary = max((rank[p] for p in included), default=0)
        called = relationships.iloc[child]
        selected_ranks = [rank.get(ids.get(called.get(f"Parent{slot}"))) for slot in (1, 2)]
        rows.append(dict(Sample=sample, EligibleParents=len(parents), ScoredM2Pairs=len(trios),
                         M2PanelParents=len(included), ScreenBoundaryRank=boundary,
                         ScreenBoundaryGap=weakest - strongest_omitted,
                         OmittedParentCount=len(omitted),
                         StrongestOmittedParent=None if not omitted else scored.sample_ids[omitted[0]],
                         Parent1ScreenRank=selected_ranks[0], Parent2ScreenRank=selected_ranks[1],
                         SelectedParentAtScreenBoundary=bool(omitted and boundary in selected_ranks),
                         Note="screen ranks/gaps are diagnostics, not a certificate of search adequacy"))
    return pd.DataFrame(rows)
