"""Release supported parent edges without asserting an uncertain parent count.

Chromosome-resampling criteria remain unchanged. A separate, explicitly labelled
full-marker count release can resolve M0/M1 without adding edges. Partial tables
marginalize edge support over M1 and M2, and remain subsets of the full-data
acyclic graph. Downstream family conditioning uses only explicitly supported
edges, not the unresolved co-parent or a guessed M0/M1/M2 state.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


def partial_relationships(exact, diagnostics, settings, tier):
    """Keep edge-level stability separate from count/configuration stability."""
    partial = exact.copy(deep=True)
    bootstrap_cutoff = getattr(settings, f"{tier}_parent_bootstrap")
    loco_cutoff = getattr(settings, f"{tier}_loco_fraction")
    for slot in (1, 2):
        partial[f"Parent{slot}Supported"] = partial[f"Parent{slot}"].notna()
    partial["ExactConfigurationResolved"] = partial["InferenceStatus"].str.startswith(
        f"{tier}_supported_")
    partial["ObservedParentCountMinimum"] = partial["ObservedParentCount"].fillna(0).astype(int)
    partial["ObservedParentCountMaximum"] = partial["ObservedParentCount"].fillna(2).astype(int)

    for child, diagnostic in enumerate(diagnostics):
        if bool(partial.at[child, "ExactConfigurationResolved"]):
            continue
        # Accept only a subset of the same supported full-data DAG: marginal
        # edges from mutually incompatible graph solutions cannot form a union.
        if (diagnostic["GraphConflict"] or diagnostic["GraphTieConflict"]
                or diagnostic["InformativeContigCount"] < settings.minimum_informative_contigs):
            continue
        retained = 0
        for slot in (1, 2):
            parent = diagnostic[f"LocalParent{slot}"]
            if parent is None or parent != diagnostic[f"CompleteParent{slot}"]:
                continue
            support = np.asarray([
                diagnostic[f"Parent{slot}BootstrapFraction"],
                diagnostic[f"Parent{slot}LOCOFraction"],
                diagnostic[f"SelectedGraphParent{slot}BootstrapFraction"],
                diagnostic[f"SelectedGraphParent{slot}LOCOFraction"],
            ], dtype=float)
            if (np.all(np.isfinite(support))
                    and np.all(support >= (bootstrap_cutoff, loco_cutoff,
                                           bootstrap_cutoff, loco_cutoff))):
                partial.at[child, f"Parent{slot}"] = parent
                partial.at[child, f"Parent{slot}Supported"] = True
                retained += 1
        if retained:
            partial.at[child, "ObservedParentCountMinimum"] = max(
                int(partial.at[child, "ObservedParentCountMinimum"]), retained)
            partial.at[child, "InferenceStatus"] = f"{tier}_marginal_parent_support"
    return partial


def add_partial_support(candidate_sets, partial):
    """Expose retained edges without inventing a complete configuration set."""
    result = candidate_sets.copy(deep=True)
    for name in ("Parent1Supported", "Parent2Supported",
                 "ObservedParentCountMinimum", "ObservedParentCountMaximum"):
        result[name] = partial[name].to_numpy()
    for child, row in partial.iterrows():
        if row["ExactConfigurationResolved"]:
            continue
        retained = False
        for slot in (1, 2):
            if row[f"Parent{slot}Supported"]:
                parent = row[f"Parent{slot}"]
                result.at[child, f"Parent{slot}"] = parent
                result.at[child, f"Parent{slot}Candidates"] = (parent,)
                retained = True
        if retained:
            result.at[child, "InferenceStatus"] = row["InferenceStatus"]
    return result


def exclusion_candidate_pairs(frame, eligible):
    """All eligible candidate dyads for unresolved/non-exact Tier-B rows."""
    counts = frame[['Parent1', 'Parent2']].notna().sum(axis=1).to_numpy()
    unresolved = ((frame.ParentState.to_numpy() == 'unresolved')
                  | (counts != frame.ObservedParentCount.to_numpy()))
    children = np.flatnonzero(unresolved & eligible.eligible_children)
    return np.asarray([(int(c), int(p)) for c in children
                       for p in np.flatnonzero(eligible.eligible_parents[c])],
                      dtype=np.int64).reshape(-1, 2)


def apply_full_marker_release(result, rows, log_e, eligible):
    """Release M0/M1 with exclusion bounds and existing directed family edges.

    This clarifies the count of observed parents; it cannot create new edges.
    The exclusion error bound is conditional on calibrated read likelihoods.
    Using an outgoing edge to reject its reverse additionally assumes that
    supported direction is correct; bootstrap fractions are not probabilities
    guaranteeing this. No same-data evidence is multiplied into HMM scores.
    """
    frame = result.tier_b_relationships.copy(deep=True)
    partial = result.tier_b_partial_relationships.copy(deep=True)
    complete = result.complete_relationships
    diagnostics = result.diagnostics
    settings = result.config
    names = tuple(frame.Sample)
    if any(tuple(table.Sample) != names for table in (partial, complete, diagnostics)):
        raise ValueError("full-marker release requires matching pedigree sample order")
    evidence = {(int(c), int(p)): float(e) for (c, p), e in zip(rows, log_e)}
    targets = sorted({int(c) for c, _ in rows})
    outgoing = {name: set() for name in names}
    for row in partial.itertuples():
        for slot in (1, 2):
            if getattr(row, f'Parent{slot}Supported'):
                outgoing[getattr(row, f'Parent{slot}')].add(row.Sample)
    frame['ReleaseEvidence'] = 'chromosome_resampling'
    partial['ReleaseEvidence'] = 'chromosome_resampling'
    threshold = math.log(2*len(names)/settings.mendelian_exclusion_alpha)
    records = []
    for child in targets:
        parents = list(map(int, np.flatnonzero(eligible.eligible_parents[child])))
        covered = all((child, p) in evidence for p in parents)
        compatible = {names[p] for p in parents if not covered or evidence[child, p] < threshold}
        reverse = compatible & outgoing[names[child]]
        survivors = compatible - reverse
        supported = {partial.at[child, f'Parent{k}'] for k in (1, 2)
                     if partial.at[child, f'Parent{k}Supported']}
        selected = {complete.at[child, f'Parent{k}'] for k in (1, 2)
                    if pd.notna(complete.at[child, f'Parent{k}'])}
        diag = diagnostics.iloc[child]
        safe = (eligible.eligible_children[child]
                and diag.InformativeContigCount >= settings.minimum_informative_contigs
                and not diag.GraphConflict and not diag.GraphTieConflict)
        concordant = (survivors == supported == selected
                      and complete.at[child, 'ObservedParentCount'] == len(survivors))
        released = bool(covered and safe and len(survivors) <= 1 and concordant)
        excluded = [evidence[child, p] for p in parents
                    if covered and names[p] not in compatible]
        records.append(dict(Sample=names[child], EligibleCandidates=len(parents),
            CompatibleCandidates=tuple(sorted(compatible)),
            SupportedReverseCandidates=tuple(sorted(reverse)),
            RemainingCandidates=tuple(sorted(survivors)),
            WeakestExclusionLogE=min(excluded, default=np.nan),
            ThresholdLogE=threshold, CompleteCandidateEvidence=covered, Released=released,
            ReleaseConditionalOnSupportedDirections=bool(reverse)))
        if not released:
            continue
        kept = sorted(survivors)
        for table in (frame, partial):
            table.at[child, 'ParentState'] = ('zero_observed_parents', 'one_observed_parent')[len(kept)]
            table.at[child, 'ObservedParentCount'] = len(kept)
            table.at[child, 'Parent1'] = kept[0] if kept else None
            table.at[child, 'Parent2'] = None
            table.at[child, 'InferenceStatus'] = 'tier_b_fixed_genome_directed_exclusion'
            table.at[child, 'ReleaseEvidence'] = 'mendelian_exclusion_and_supported_direction'
        partial.at[child, 'ExactConfigurationResolved'] = True
        for slot in (1, 2):
            partial.at[child, f'Parent{slot}Supported'] = pd.notna(frame.at[child, f'Parent{slot}'])
        for bound in ('Minimum', 'Maximum'):
            partial.at[child, f'ObservedParentCount{bound}'] = len(kept)

        # Keep the primary, candidate-set and explanation views consistent,
        # without relabelling their original bootstrap/LOCO measurements.
        for table in (result.tier_b_candidate_sets, result.call_explanations):
            for column in ('ParentState', 'ObservedParentCount', 'Parent1', 'Parent2', 'InferenceStatus'):
                table.at[child, column] = frame.at[child, column]
            table.at[child, 'ExactConfigurationResolved'] = True
            table.at[child, 'Parent1Candidates'] = tuple(kept)
            table.at[child, 'Parent2Candidates'] = ()
            table.at[child, 'ConfigurationCandidates'] = (tuple(kept),)
            for column in ('Parent1Supported', 'Parent2Supported',
                           'ObservedParentCountMinimum', 'ObservedParentCountMaximum'):
                table.at[child, column] = partial.at[child, column]
        result.call_explanations.at[child, 'Interpretation'] = (
            'fixed-observed-genome exclusion conditional on supported family directions; '
            'original resampling stability is unchanged; M0 is not founder generation')

    final_state = (frame.ParentState.ne('unresolved')
                   & frame.ObservedParentCount.eq(frame[['Parent1', 'Parent2']].notna().sum(axis=1)))
    for table in (result.diagnostics, result.parent_state_calls,
                  result.tier_b_candidate_sets, result.call_explanations):
        table['TierBFinalStateCall'] = final_state.to_numpy()
        table['TierBReleaseEvidence'] = frame.ReleaseEvidence.to_numpy()
    result.tier_b_relationships = frame
    result.tier_b_partial_relationships = partial
    if settings.primary_view == 'tier_b':
        result.relationships = frame.copy(deep=True)
    result.exclusion_diagnostics = pd.DataFrame(records, columns=[
        'Sample', 'EligibleCandidates', 'CompatibleCandidates', 'SupportedReverseCandidates',
        'RemainingCandidates', 'WeakestExclusionLogE', 'ThresholdLogE',
        'CompleteCandidateEvidence', 'Released', 'ReleaseConditionalOnSupportedDirections'])
