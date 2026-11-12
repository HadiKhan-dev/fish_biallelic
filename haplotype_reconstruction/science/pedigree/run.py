"""Compose the pedigree decision, exclusion request and release without executing tasks.

Owns PD27's pure transitions. Exclusion scoring supplies the requested dyads; relative scoring supplies reports.
No engine, store, filesystem or logging belongs here.
Method: docs/methods/pedigree.md#products-and-published-tables (PD17, PD22–PD27).
Identity: guard before count; target prefix only; original edges and both guard passes
must agree; slots are filtered rather than reassigned after withdrawal.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ...contracts.pedigree import (
    NO_PARENT,
    CalibrationRecord,
    CountExclusion,
    CountReleaseBasis,
    Decision,
    ExclusionEvidence,
    ExclusionRequest,
    ExclusionSummary,
    GenomeScores,
    GuardRelatedness,
    GuardSummary,
    Pedigree,
    PedigreePrecheck,
    PedigreeReportInputs,
    PedigreeResult,
    PedigreeRetained,
    RelativeRequest,
    ReleasedDecision,
    Resampling,
    SupportCounts,
    SupportFractions,
    SupportRecord,
)
from ...core.errors import InputError, InvariantError
from . import api, count, decision, exclusion, guard
from .eligibility import ParentEligibility
from .parameters import PedigreeParameters
from .resampling import minimum_contigs_for_tier

__all__ = [
    "precheck",
    "start_full_fit",
    "finish_full_fit",
    "complete_decision",
    "exclusion_request",
    "release_decision",
    "assemble",
]


def precheck(
    n_samples: int, contigs: int, parameters: PedigreeParameters, *, basis: str = "informative"
) -> PedigreePrecheck:
    """Reject N<3 or C<ceil(exposure units); report the configured tier-reach warning.

    Workflow callers apply the configured check only when they will reach pedigree;
    start_full_fit applies the informative check after preparation omits uninformative contigs.
    """
    if n_samples < 3:
        raise InputError(f"pedigree requires at least 3 samples; received {n_samples}")
    minimum = int(math.ceil(parameters.exposure.minimum_exposed_units))
    if contigs < minimum:
        raise InputError(f"pedigree requires at least {minimum} {basis} contigs; received {contigs}")
    tier_minimum = minimum_contigs_for_tier(parameters.release.tier_b, minimum)
    return PedigreePrecheck(n_samples, contigs, minimum, tier_minimum, contigs < tier_minimum, basis)


def start_full_fit(scores: GenomeScores, parameters: PedigreeParameters) -> decision.FitStart:
    """Prepare the full fit and its evidence-scale search (calibration.solve on ``.calibration``)."""
    precheck(len(scores.sample_ids), len(scores.contigs), parameters)
    return decision.start_fit(scores, ParentEligibility.unrestricted(scores.sample_ids), parameters)


def finish_full_fit(
    started: decision.FitStart, fitted: tuple[float, CalibrationRecord], parameters: PedigreeParameters
) -> decision.FullFit:
    """Finish the full fit after its evidence-scale search."""
    return decision.finish_fit(started, fitted, parameters)


def complete_decision(state: decision.FullFit, resampled: Resampling, parameters: PedigreeParameters) -> Decision:
    """Complete the decision after resampling: support, tiers, views and candidate sets."""
    return decision.decide_resampled(state, resampled, parameters)


def exclusion_request(
    decided: Decision, contigs: tuple[str, ...], relatedness: GuardRelatedness, parameters: PedigreeParameters
) -> ExclusionRequest:
    """Freeze independent count-exclusion and painting-relationship queries."""
    sample_ids, rows = decided.scores.sample_ids, decided.alternatives.rows
    dyads = count.target_dyads(decided.tier_b, rows, decided.eligibility).reshape(-1, 2)
    before = api.edges(decided.partial_b, sample_ids)
    families = guard.edge_families(sample_ids, before.parents, before.children, relatedness, parameters)
    relationship = guard.relationship_request(families, sample_ids, contigs, parameters)
    dyads.setflags(write=False)
    return ExclusionRequest(sample_ids, tuple(contigs), dyads, len(dyads), before, relationship)


def release_decision(
    decided: Decision,
    request: ExclusionRequest,
    evidence: Sequence[ExclusionEvidence],
    relatedness: GuardRelatedness,
    observed_counts: np.ndarray,
    parameters: PedigreeParameters,
    *,
    relationship_evidence=(),
) -> ReleasedDecision:
    """Combine configured evidence, guard/count, orient and check the second guard; no scoring."""
    sample_ids, rows = decided.scores.sample_ids, decided.alternatives.rows
    if request.sample_ids != sample_ids or relatedness.sample_ids != sample_ids:
        raise InvariantError("exclusion/relatedness sample axes differ from the decision")
    if observed_counts.shape != (len(request.contigs), len(sample_ids)):
        raise InvariantError("observed counts do not follow configured contig and sample axes")
    scored, before = request.dyads, request.before_edges
    targets = request.target_count
    dyads = scored[:targets]
    edge_set = set(zip(before.parents.tolist(), before.children.tolist(), strict=True))
    chromosomes: list[dict[str, str | int]] | None = None
    exclusion_status = "no_unresolved_candidates"
    lower, upper, observed = np.empty(0), None, np.empty(0, dtype=np.int64)
    if len(scored):
        if len(evidence) != len(request.contigs):
            raise InvariantError(f"decision requires {len(scored)} exclusion dyads on every configured contig")
        for contig, value in zip(request.contigs, evidence, strict=True):
            if value.contig != contig or not np.array_equal(value.dyads, scored):
                raise InvariantError(f"decision exclusion dyads differ from supplied evidence on {contig}")
            if value.log_e.shape != (len(scored),) or value.observed_sites.shape != (len(scored),):
                raise InvariantError(f"exclusion evidence axes differ on {contig}")
            if value.log_e_upper is not None and value.log_e_upper.shape != (len(scored),):
                raise InvariantError(f"exclusion upper endpoint axis differs on {contig}")
        lower, upper, observed = exclusion.combine(evidence)
        chromosomes = [
            dict(contig=value.contig, markers=value.markers, pattern_fallback_samples=value.pattern_fallback_samples)
            for value in evidence
        ]
        if targets:
            exclusion_status = "evaluated"
    elif evidence:
        raise InvariantError("supplied exclusion evidence is outside the decision's path")
    preliminary_guard = guard.fixed_point(before, relatedness, request.relationship, relationship_evidence, parameters)
    released = count.release_counts(
        CountReleaseBasis(
            decided.tier_b,
            decided.partial_b,
            decided.complete,
            decided.evidence.graph_conflict,
            decided.informative_units,
            rows,
            decided.eligibility,
        ),
        CountExclusion(
            dyads, lower[:targets], observed[:targets], None if upper is None or not targets else upper[:targets]
        ),
        parameters,
        withheld=preliminary_guard.withheld,
    )
    for record in released.records:
        child = record.child
        if record.released and (
            any((child, candidate) in preliminary_guard.withheld for candidate in record.reverse)
            or any(edge[1] == child for edge in preliminary_guard.withheld)
        ):
            raise InvariantError(f"count release of {sample_ids[child]} rests on a withdrawn edge")
    orientation = api.released_order(
        observed_counts,
        sample_ids,
        [
            api.calls(released.tier_b, rows).parents,
            api.calls(decided.tier_a, rows).parents,
            api.calls(decided.complete, rows).parents,
            decided.candidates.parents,
            np.where(released.partial_b.supported, released.partial_b.parents, NO_PARENT),
            np.where(decided.partial_b.supported, decided.partial_b.parents, NO_PARENT),
            np.where(decided.partial_a.supported, decided.partial_a.parents, NO_PARENT),
        ],
        reverse=parameters.reverse_conventions,
    )
    ordered_edges = api.edges(released.partial_b, sample_ids, orientation)
    if set(zip(ordered_edges.parents.tolist(), ordered_edges.children.tolist(), strict=True)) != edge_set:
        raise InvariantError("count release changed the edges the guard was scored for")
    final_guard = guard.fixed_point(ordered_edges, relatedness, request.relationship, relationship_evidence, parameters)
    if final_guard.withheld != preliminary_guard.withheld:
        raise InvariantError("guard withdrawals differ after released slot ordering")
    pre_guard_calls = api.calls(released.tier_b, rows, orientation)
    final_calls, tier_a, final_edges = guard.apply_calls(
        pre_guard_calls, api.calls(decided.tier_a, rows, orientation), ordered_edges, final_guard
    )
    summary = ExclusionSummary(
        model="full-marker-mendelian-directed-v1",
        alpha=parameters.exclusion.alpha,
        genotype_replacement_probability=parameters.exclusion.genotype_replacement_probability,
        conditional_on_supported_edge_directions=True,
        candidate_dyads=targets,
        contigs=len(request.contigs),
        status=exclusion_status,
        released=int(np.count_nonzero(released.released)),
        chromosomes=chromosomes,
    )
    return ReleasedDecision(
        decided,
        request,
        released,
        orientation.ranks,
        pre_guard_calls,
        final_calls,
        tier_a,
        final_edges,
        api.calls(decided.complete, rows, orientation),
        final_guard,
        summary,
    )


def assemble(
    released: ReleasedDecision,
    omitted: dict[str, str],
    parameters: PedigreeParameters,
    reports: PedigreeReportInputs,
) -> PedigreeResult:
    """Assemble retained state and complete numeric report views for publication."""
    decided = released.decision
    numeric = api.report_views(
        decided,
        released.count_release,
        released.orientation_ranks,
        released.guard,
        decided.scores.sample_ids,
        released.request.contigs,
        reports,
        parameters,
    )
    product = Pedigree(
        decided.scores.sample_ids,
        released.request.contigs,
        decided.scores.contigs,
        omitted,
        released.calls,
        released.edges,
        released.tier_a,
        released.complete,
        decided.reach,
        decided.calibration,
        decided.eligibility.summary(),
        decided.scores.panel,
        support_record(decided),
        released.guard,
        views=numeric,
        family_solve=decided.family_solve,
        guard_summary=guard_summary(released.guard),
        exclusion=released.exclusion,
        retained=PedigreeRetained(decided, released.count_release, released.orientation_ranks, released.request),
        reports_complete=True,
    )
    return PedigreeResult(product, decided, released.count_release, released.orientation_ranks, released.exclusion)


def relative_request(released: ReleasedDecision) -> RelativeRequest:
    """PD26 request: sorted pairs from post-count Tier B, partial B and complete, before guard.

    View membership follows insertion order; duplicate memberships are
    retained. Slot orientation cannot change the pair set or membership order.
    """
    rows = released.decision.inputs.alternatives.rows
    memberships: dict[tuple[int, int], list[str]] = {}
    for label, parents in (
        ("tier_b", api.calls(released.count_release.tier_b, rows).parents),
        (
            "tier_b_partial",
            np.where(released.count_release.partial_b.supported, released.count_release.partial_b.parents, NO_PARENT),
        ),
        ("complete", api.calls(released.decision.complete, rows).parents),
    ):
        for child, candidates in enumerate(parents.tolist()):
            for parent in candidates:
                if parent >= 0:
                    memberships.setdefault((child, int(parent)), []).append(label)
    pairs = np.asarray(sorted(memberships), dtype=np.int64).reshape(-1, 2)
    return RelativeRequest(pairs, tuple(tuple(memberships[(int(c), int(p))]) for c, p in pairs))


def guard_summary(value) -> GuardSummary:
    """Summarize the final typed guard votes."""
    return GuardSummary(
        "painting-relationship-v1",
        len(value.votes),
        sum(v.call != "untested" for v in value.votes),
        sum(v.call == "withheld" for v in value.votes),
        value.cleaning_iterations,
        value.cleaning_converged,
        value.cleaning_iteration_bound,
        sum(v.call == "withheld" and v.evidence_call != "withheld" for v in value.votes),
    )


def support_record(decided: Decision) -> SupportRecord:
    """Persist exact count/fits fractions once; counts remain the comparison authority."""

    def fractions(value: SupportCounts) -> SupportFractions:
        arrays = (
            value.local_configuration,
            value.graph_configuration,
            value.local_state,
            value.local_parent.values,
            value.graph_parent.values,
        )
        divided = tuple(a / value.fits if value.fits else np.zeros(a.shape, dtype=np.float64) for a in arrays)
        return SupportFractions(*divided)

    support = decided.resampling
    # By keyword: the record's kw_only diagnostics sit between its positional fields in declaration order.
    return SupportRecord(
        bootstrap=support.bootstrap,
        loco=support.loco,
        multiplicities=support.multiplicities,
        worker_count=support.worker_count,
        bootstrap_fractions=fractions(support.bootstrap),
        loco_fractions=fractions(support.loco),
        bootstrap_graph_budget_exhausted=support.bootstrap_graph_budget_exhausted,
        loco_graph_budget_exhausted=support.loco_graph_budget_exhausted,
        maximum_graph_passes_used=support.maximum_graph_passes_used,
    )
