"""Compose the pedigree decision, exclusion request and release without executing tasks.

Owns PD27's pure transitions. Exclusion scoring supplies the requested dyads; relative scoring supplies reports.
No engine, store, filesystem or logging belongs here.
Method: docs/methods/pedigree.md#products-and-published-tables (PD17, PD22–PD27).
Identity: guard before count; target prefix only; original edges and both guard passes
must agree; slots are filtered rather than reassigned after withdrawal.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ...contracts.pedigree import (
    NO_PARENT,
    CountExclusion,
    CountReleaseBasis,
    Decision,
    DecisionInputs,
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
from . import api, count, decision, exclusion, family_transmission, guard, regimes
from .eligibility import ParentEligibility
from .parameters import PedigreeParameters

__all__ = [
    "exclusion_contigs",
    "precheck",
    "start_full_fit",
    "start_inputs",
    "full_selection",
    "full_fit_state",
    "complete_decision",
    "exclusion_request",
    "release_decision",
    "assemble",
]


def precheck(n_samples: int, *, basis: str = "informative") -> PedigreePrecheck:
    """Reject N < 3. Material (never a count of contigs) is stated by the run's TierReach."""
    if n_samples < 3:
        raise InputError(f"pedigree requires at least 3 samples; received {n_samples}")
    return PedigreePrecheck(n_samples, basis)


def start_full_fit(scores: GenomeScores, parameters: PedigreeParameters) -> decision.FullFit | Pedigree:
    """Fit the supplied unit evidence; a run below the assessable material gets the completed unresolved product."""
    start = start_inputs(scores, parameters)
    if isinstance(start, Pedigree):
        return start
    return full_fit_state(start, full_selection(start.inputs, parameters))


def full_selection(inputs: DecisionInputs, parameters: PedigreeParameters) -> decision.FullSelection:
    """The decision's full-data fit from its inputs alone (it reads nothing of the resampled fits, which read nothing
    of it), under the inputs' regime."""
    return decision.full_selection(inputs, regimes.effective(parameters, inputs.regime))


def full_fit_state(start: decision.FitStart, chosen: decision.FullSelection) -> decision.FullFit:
    """The decision state before resampling: the fit inputs joined with their full-data fit."""
    return decision.assemble(start, chosen)


def start_inputs(scores: GenomeScores, parameters: PedigreeParameters) -> decision.FitStart | Pedigree:
    """The decision's fit inputs (the direction model's neutral fit included) from the supplied unit evidence; a run
    below the assessable material gets the completed unresolved product."""
    precheck(len(scores.sample_ids))
    classified = regimes.classify(scores.regime_basis_morgans, len(scores.contigs), parameters)
    if classified.name != scores.regime:
        raise InputError("unit scores' regime differs from their material")
    if classified.name == "not_assessable":
        from . import not_assessable
        if not scores.chromosome_summary and scores.informative_contigs:
            raise InputError("not-assessable scored inputs require their prepared chromosome summaries")
        return not_assessable.product(scores.sample_ids, scores.informative_contigs,
                                      scores.chromosome_summary, classified.reach, parameters)
    return decision.start_inputs(scores, ParentEligibility.unrestricted(scores.sample_ids),
                                 regimes.effective(parameters, scores.regime))


def complete_decision(state: decision.FullFit, resampled: Resampling, parameters: PedigreeParameters) -> Decision:
    """Complete the decision after resampling: support, tiers, views and candidate sets."""
    return decision.decide_resampled(state, resampled, regimes.effective(parameters, state.scores.regime))


def exclusion_request(
    decided: Decision, contigs: tuple[str, ...], relatedness: GuardRelatedness, parameters: PedigreeParameters,
    configured_morgans: Sequence[float],
) -> ExclusionRequest:
    """Freeze independent count-exclusion and painting-relationship queries; ``configured_morgans``: each
    configured contig's material (the fixed exclusion weights, DESIGN_417 D9)."""
    parameters = regimes.effective(parameters, decided.scores.regime)
    sample_ids, rows = decided.scores.sample_ids, decided.alternatives.rows
    dyads = count.target_dyads(decided.tier_b, rows, decided.eligibility).reshape(-1, 2)
    before = api.edges(decided.partial_b, sample_ids)
    families = guard.edge_families(sample_ids, before.parents, before.children, relatedness, parameters)
    relationship = guard.relationship_request(families, sample_ids, contigs, parameters)
    dyads.setflags(write=False)
    if len(configured_morgans) != len(contigs):
        raise InputError("exclusion weights must follow the configured contig axis")
    log_weights = exclusion.contig_log_weights(configured_morgans) if len(dyads) else ()
    return ExclusionRequest(sample_ids, tuple(contigs), dyads, len(dyads), before, relationship,
                            log_weights=log_weights)


def transmission_request(
    decided: Decision,
    request: ExclusionRequest,
    relatedness: GuardRelatedness,
    observed_counts: np.ndarray,
    parameters: PedigreeParameters,
    *,
    relationship_evidence=(),
):
    """Freeze the PD29 family transmission request after the guard's fixed point: (request, witnesses per edge)."""
    parameters = regimes.effective(parameters, decided.scores.regime)
    preliminary = guard.fixed_point(request.before_edges, relatedness, request.relationship, relationship_evidence,
                                    parameters)
    return family_transmission.request(decided.scores.sample_ids, request.contigs, request.before_edges, preliminary,
                                       relatedness, request.relationship, relationship_evidence, observed_counts,
                                       parameters)


def release_decision(
    decided: Decision,
    request: ExclusionRequest,
    evidence: Sequence[ExclusionEvidence],
    relatedness: GuardRelatedness,
    observed_counts: np.ndarray,
    parameters: PedigreeParameters,
    *,
    relationship_evidence=(),
    transmission_evidence=(),
    unresolved_rate_samples: int = 0,
) -> ReleasedDecision:
    """Combine configured evidence, guard/count, orient and check the second guard; no scoring.
    ``unresolved_rate_samples``: samples whose outlier rate sits at the calibration grid's top (their exclusion dyads
    were withheld from certification), reported in the exclusion summary."""
    parameters = regimes.effective(parameters, decided.scores.regime)
    sample_ids, rows = decided.scores.sample_ids, decided.alternatives.rows
    if request.sample_ids != sample_ids or relatedness.sample_ids != sample_ids:
        raise InvariantError("exclusion/relatedness sample axes differ from the decision")
    if observed_counts.shape != (len(request.contigs), len(sample_ids)):
        raise InvariantError("observed counts do not follow configured contig and sample axes")
    scored, before = request.dyads, request.before_edges
    targets = request.target_count
    dyads = scored[:targets]
    edge_set = set(zip(before.parents.tolist(), before.children.tolist(), strict=True))
    chromosomes: list[dict[str, str | int | float]] | None = None
    exclusion_status = "no_unresolved_candidates"
    lower, upper, observed = np.empty(0), None, np.empty(0, dtype=np.int64)
    certifiable = None
    if len(scored):
        weighted = exclusion_contigs(request)
        if len(evidence) != len(weighted):
            raise InvariantError(f"decision requires {len(scored)} exclusion dyads on every contig with material")
        for (contig, log_weight), value in zip(weighted, evidence, strict=True):
            if value.contig != contig or not np.array_equal(value.dyads, scored):
                raise InvariantError(f"decision exclusion dyads differ from supplied evidence on {contig}")
            if value.log_weight != log_weight:
                raise InvariantError(f"exclusion evidence on {contig} carries another weight than the request")
            if value.log_e.shape != (len(scored),) or value.observed_sites.shape != (len(scored),):
                raise InvariantError(f"exclusion evidence axes differ on {contig}")
            if value.log_e_upper is not None and value.log_e_upper.shape != (len(scored),):
                raise InvariantError(f"exclusion upper endpoint axis differs on {contig}")
        lower, upper, observed = exclusion.combine(evidence)
        # A fitted library on any contributing contig makes the genome score a working score.
        certifiable = np.ones(len(scored), dtype=np.bool_)
        for value in evidence:
            if value.certifiable is not None:
                if value.certifiable.shape != certifiable.shape:
                    raise InvariantError(f"exclusion certification axis differs on {value.contig}")
                certifiable &= value.certifiable
        chromosomes = [
            dict(contig=value.contig, markers=value.markers, weight=float(np.exp(value.log_weight)))
            for value in evidence
        ]
        if targets:
            exclusion_status = "evaluated"
    elif evidence:
        raise InvariantError("supplied exclusion evidence is outside the decision's path")
    preliminary_guard = guard.fixed_point(before, relatedness, request.relationship, relationship_evidence, parameters)
    # PD29: one family transmission pass on the edges the guard retained (no iteration with the guard's cleaning)
    transmitted, witnesses = family_transmission.request(sample_ids, request.contigs, before, preliminary_guard,
                                                         relatedness, request.relationship, relationship_evidence,
                                                         observed_counts, parameters)
    if len(transmitted.parents) and len(transmission_evidence) == 0:
        raise InvariantError("the family transmission request needs its per-contig evidence")
    preliminary_guard = family_transmission.apply(preliminary_guard, transmitted, witnesses, transmission_evidence,
                                                  parameters)
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
            dyads, lower[:targets], observed[:targets], None if upper is None or not targets else upper[:targets],
            None if certifiable is None else certifiable[:targets],
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
            *([api.calls(decided.tier_a, rows).parents] if decided.tier_a_assessed else []),
            api.calls(decided.complete, rows).parents,
            decided.candidates.parents,
            np.where(released.partial_b.supported, released.partial_b.parents, NO_PARENT),
            np.where(decided.partial_b.supported, decided.partial_b.parents, NO_PARENT),
            *([np.where(decided.partial_a.supported, decided.partial_a.parents, NO_PARENT)]
              if decided.tier_a_assessed else []),
        ],
        reverse=parameters.reverse_conventions,
    )
    ordered_edges = api.edges(released.partial_b, sample_ids, orientation)
    if set(zip(ordered_edges.parents.tolist(), ordered_edges.children.tolist(), strict=True)) != edge_set:
        raise InvariantError("count release changed the edges the guard was scored for")
    final_guard = guard.fixed_point(ordered_edges, relatedness, request.relationship, relationship_evidence, parameters)
    final_transmitted, final_witnesses = family_transmission.request(
        sample_ids, request.contigs, ordered_edges, final_guard, relatedness, request.relationship,
        relationship_evidence, observed_counts, parameters)
    if final_transmitted.digest != transmitted.digest:
        raise InvariantError("family transmission request differs after released slot ordering")
    final_guard = family_transmission.apply(final_guard, final_transmitted, final_witnesses, transmission_evidence,
                                            parameters)
    if final_guard.withheld != preliminary_guard.withheld:
        raise InvariantError("guard withdrawals differ after released slot ordering")
    pre_guard_calls = api.calls(released.tier_b, rows, orientation)
    final_calls, tier_a, final_edges = guard.apply_calls(
        pre_guard_calls, api.calls(decided.tier_a, rows, orientation), ordered_edges, final_guard
    )
    summary = ExclusionSummary(
        model="full-marker-mendelian-directed-observation-v1",
        alpha=parameters.exclusion.alpha,
        unresolved_rate_samples=int(unresolved_rate_samples),
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
    *,
    veto_outcome=None,
) -> PedigreeResult:
    """Assemble retained state and complete numeric report views for publication; ``veto_outcome``: the both-maps
    veto's withheld calls (material_veto), already applied to ``released``, withdrawn from the views too."""
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
    from dataclasses import replace

    from ...contracts.pedigree import BlockSummaryView
    from .low_labels import label_views
    numeric = label_views(numeric, released, low_material=decided.scores.regime == "low_material",
                          tier_a=decided.tier_a_assessed)
    if veto_outcome is not None:
        from .material_veto import apply_views
        numeric = apply_views(numeric, veto_outcome)
    scores = decided.scores
    if scores.piece_contig is not None and scores.piece_unit is not None:
        assert scores.piece_bins is not None and scores.piece_markers is not None
        numeric = replace(numeric, regime=scores.regime, block_summary=tuple(
            BlockSummaryView(scores.real_contigs[int(scores.piece_contig[i])], int(scores.piece_unit[i]),
                d.component, d.first_bin, d.stop_bin - 1, d.nominal_index, d.working_morgans,
                int(scores.piece_bins[i]), int(scores.piece_markers[i]))
            for i, d in enumerate(scores.pieces)))
    else:
        numeric = replace(numeric, regime=scores.regime)
    product = Pedigree(
        decided.scores.sample_ids,
        released.request.contigs,
        decided.scores.informative_contigs,
        omitted,
        released.calls,
        released.edges,
        released.tier_a,
        released.complete,
        decided.reach,
        decided.eligibility.summary(),
        decided.scores.panel,
        support_record(decided),
        released.guard,
        views=numeric,
        birth_order=decided.birth_order,
        guard_summary=guard_summary(released.guard),
        exclusion=released.exclusion,
        retained=PedigreeRetained(decided, released.count_release, released.orientation_ranks, released.request),
        reports_complete=True,
        regime=decided.scores.regime,
    )
    return PedigreeResult(product, decided, released.count_release, released.orientation_ranks, released.exclusion)


def exclusion_contigs(request: ExclusionRequest) -> tuple[tuple[str, float], ...]:
    """(contig, log weight) of the configured contigs that carry material and so get an exclusion task."""
    if not len(request.dyads):
        return ()
    if len(request.log_weights) != len(request.contigs):
        raise InvariantError("exclusion request weights do not follow its contigs")
    return tuple((c, w) for c, w in zip(request.contigs, request.log_weights, strict=True) if np.isfinite(w))


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
        bootstrap_order_restricted=support.bootstrap_order_restricted,
        loco_order_restricted=support.loco_order_restricted,
    )
