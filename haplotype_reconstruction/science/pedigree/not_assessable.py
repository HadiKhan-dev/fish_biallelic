"""Completed unresolved pedigree without a fitted decision (DESIGN_4154 A3)."""
from __future__ import annotations
import numpy as np
from ...contracts import pedigree as records
from .eligibility import ParentEligibility


def product(sample_ids, contigs, summary, reach, parameters):
    n = len(sample_ids)
    calls = records.PedigreeCalls(np.full(n, -1, np.int8), np.full((n, 2), -1, np.int64),
                                 ("unresolved",) * n)
    tier_a = records.PedigreeCalls(calls.state, calls.parents, ("not_assessed",) * n)
    edges = records.ParentEdges(sample_ids, np.empty(0, np.int64), np.empty(0, np.int64),
                                np.empty(0, np.int64), np.zeros(n, np.int64), np.full(n, 2, np.int64))
    rows = tuple(records.RelationshipView(i, -1, -1, -1, float("nan"), "unresolved") for i in range(n))
    partial = tuple(records.PartialRelationshipView(i, -1, -1, -1, float("nan"), "unresolved",
                    False, False, False, 0, 2) for i in range(n))
    empty = records.ExclusionEvidenceView(np.empty(0, np.int64), np.empty(0, np.int64),
                                          np.empty(0), np.empty(0, np.int64))
    views = records.PedigreeViews(
        rows, partial, (), (), rows, (), (), (), (), (), (), empty, (), (), tuple(summary), None, None, regime="not_assessable")
    return records.Pedigree(
        sample_ids, tuple(contigs), tuple(s.contig for s in summary if s.omitted_reason is None),
        {s.contig: s.omitted_reason for s in summary if s.omitted_reason is not None},
        calls, edges, tier_a, calls, reach, ParentEligibility.unrestricted(sample_ids).summary(),
        None, None, None, views=views, family_solve=None, guard_summary=None,
        exclusion=records.ExclusionSummary("full-marker-mendelian-directed-v1", parameters.exclusion.alpha,
            parameters.exclusion.genotype_replacement_probability, True, 0, len(contigs), "not_assessed", 0),
        retained=None, reports_complete=True, regime="not_assessable")
