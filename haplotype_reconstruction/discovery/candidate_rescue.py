"""Read-supported rescue of variation missing from feedback panels.

Inputs are two read-derived local panels, never simulation truth. Strict rescue
protects feedback calls; balanced rescue also permits final same-K refinement.
Competing BIC rows can add missing variation. The private
allele branch deliberately targets variation not explainable by a mosaic of
the called backbone. It cannot recover all absent haplotype combinations and
is not asserted to distinguish every founder from an inherited recombinant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from numba import njit

from . import cavity, fitting, modes, objectives, search
from ..core import haplotypes, parallel


@dataclass(frozen=True)
class CandidateRescueConfig:
    discovery: search.ReversibleCavitySearchConfig = field(
        default_factory=search.ReversibleCavitySearchConfig)


def private_allele_mask(backbone, candidate):
    """Called candidate alleles absent from every called backbone homologue.

An unknown backbone allele prevents claiming absence at that site. Candidate
calls have already passed the ordinary read support rule; source agreement
is not counted as another observation.
    """
    return ((candidate >= 0) & np.all(backbone >= 0, axis=0)
            & np.all(backbone != candidate, axis=0))


@njit(cache=True)
def minimum_mosaic_joins(backbone, candidate):
    """Fewest donor changes explaining a candidate, O(K L), missing neutral.

Unknown cells in either panel allow a match; thus this is a lower bound on
required joins, not invented evidence. L+1 means no mosaic can explain the
candidate's called alleles. Balanced rescue requires more than one join;
this is a novelty heuristic, not proof that a rejected founder is redundant.
    """
    k, length = backbone.shape
    impossible = length + 1
    costs = np.zeros(k, dtype=np.int64)
    for site in range(length):
        best = np.min(costs)
        for row in range(k):
            if candidate[site] < 0 or backbone[row,site] < 0 or backbone[row,site] == candidate[site]:
                costs[row] = min(costs[row], best + 1, impossible)
            else:
                costs[row] = impossible
    return int(np.min(costs))


def combine_supported_calls(backbone, additions):
    """Append exact-distinct rows without merging near neighbours."""
    rows = [row.copy() for row in backbone]
    seen = {row.tobytes() for row in rows}
    for row in additions:
        row = np.asarray(row, dtype=np.int8)
        if np.any(row >= 0) and row.tobytes() not in seen:
            rows.append(row.copy())
            seen.add(row.tobytes())
    return np.ascontiguousarray(rows, dtype=np.int8)


def rescue_candidate_panel(evidence, *, allele_depths=None, observed_mask=None,
                           feedback, competing, config=None, selection="balanced"):
    """Rescue read-supported variation from a clean feedback scaffold.

``feedback`` and ``competing`` contain ``discrete_haps`` and ``latent_haps``.
Latent values only initialize unknown sites; releases never copy those values
as evidence. BIC uses the existing missing-aware, switch-penalized score and
complexity convention. The add-only search holds source backbone alleles fixed.
Balanced rescue then permits same-K cavity-ranked refinement; strict rescue
rechecks only added rows and protects backbone calls. Return calls with aligned
probability, support and assignment metadata. Neither rule uses truth.
    """
    if selection not in ("balanced", "strict"):
        raise ValueError("selection must be balanced or strict")
    settings = CandidateRescueConfig() if config is None else config
    base = settings.discovery
    likelihood = np.array(evidence, dtype=np.float64, order="C", copy=True)
    if observed_mask is None:
        observed_mask = np.asarray(allele_depths).sum(axis=2) > 0
    observed = np.ascontiguousarray(observed_mask, dtype=np.bool_)
    if likelihood.shape != (*observed.shape, 3):
        raise ValueError("evidence and depth shape mismatch")
    active = observed.any(axis=1)
    if not active.any():
        raise ValueError("rescue needs an observed sample")
    likelihood[~observed] = 1.0 / 3.0
    scaffold = np.array(feedback["discrete_haps"], dtype=np.int8, copy=True)
    backbone = np.where(scaffold >= 0, scaffold, feedback["latent_haps"]).astype(np.int64)
    proposed = np.asarray(competing["discrete_haps"], dtype=np.int8)
    proposed_latent = np.where(proposed >= 0, proposed, competing["latent_haps"]).astype(np.int64)
    if scaffold.shape[1] != likelihood.shape[1] or proposed.shape != proposed_latent.shape:
        raise ValueError("candidate markers must match read markers")
    for panel in (backbone, proposed_latent):
        if np.any((panel != 0) & (panel != 1)):
            raise ValueError("latent starts must be binary")

    bank = []
    present = {row.tobytes() for row in backbone}
    for calls, latent in sorted(zip(proposed, proposed_latent), key=lambda x: x[1].tobytes()):
        key = latent.tobytes()
        if key in present or not np.any(calls >= 0):
            continue
        present.add(key)
        bank.append((calls.copy(), latent.copy()))
    private = [row for row, _ in bank if private_allele_mask(scaffold, row).any()]
    diagnostic = dict(bank_size=len(bank), private_candidates=len(private), scored_panels=0,
                      accepted={})
    if not bank:
        return dict(feedback, diagnostic=diagnostic)

    workspace = fitting._prepare_fixed_k_fit_workspace(
        likelihood, base.lambda_wildcard_penalty, observed_mask=observed)
    complexity = objectives.compute_founder_complexity_cost(.5, int(active.sum()), likelihood.shape[1])
    cache = {}
    polish_cache = {}
    score_config = search._stage_config(base.cavity, "mean_field")
    score_workspace = None

    def polish(panel):
        nonlocal score_workspace
        # Completing partial scaffold rows can produce exact duplicates. Use
        # the fitter's distinct-row K for both caching and same-K selection.
        panel = objectives.exact_unique_binary_rows(panel)
        key = (panel.shape, panel.tobytes())
        if key not in polish_cache:
            if score_workspace is None:
                score_evidence = likelihood if active.all() else np.ascontiguousarray(likelihood[active])
                score_workspace = cavity._prepare_cavity_scoring_workspace(score_evidence,score_config)
            raw, fitted = modes._fit_starts_with_synchronized_endpoints(
                likelihood, [panel], search._internal_move_config(base), workspace)
            candidates = [mode for mode in modes._deduplicate_modes((*raw,*fitted)) if mode.k == len(panel)]
            scores = search._score_stage(likelihood,candidates,score_config,score_workspace,active)
            best = min(scores,key=lambda s:(-s.log_score,s.mode.total_nll,s.mode.canonical_key))
            polished = fit(best.mode.haplotypes)
            polish_cache[key] = polished
        return polish_cache[key]

    def fit(panel):
        panel = np.ascontiguousarray(panel, dtype=np.int64)
        key = (panel.shape, panel.tobytes())
        if key not in cache:
            parallel.apply_dynamic_threads()
            assignment = workspace.compute_assignment_scalar(panel)
            q, support, _, _, calls = haplotypes._materialize_founder_site_pseudo_evidence(
                likelihood, panel, assignment[0], observed, base.lambda_wildcard_penalty,
                base.min_directional_supporters, base.min_hard_call_pseudo_probability)
            score = -assignment[4] - .5 * len(panel) * complexity
            cache[key] = dict(score=float(score), calls=calls, q=q, support=support,
                              latent_haps=panel, assignments=assignment[0])
        return cache[key]

    def novel(panel, row):
        if selection == "strict":
            return private_allele_mask(panel, row).any()
        return minimum_mosaic_joins(panel, row) > 1

    current = backbone.copy()
    released = scaffold.copy()
    current_score = fit(current)["score"]
    remaining = list(bank)
    accepted = []
    while remaining:
        choices = []
        for index, (calls, latent) in enumerate(remaining):
            if not novel(released, calls):
                continue
            candidate = np.vstack((current, latent))
            fitted = fit(candidate)
            if fitted["score"] > current_score + base.score_tolerance:
                choices.append((fitted["score"], candidate[-1].tobytes(), index, candidate, calls))
        if not choices:
            break
        winner = min(choices, key=lambda x: (-x[0], x[1], x[2]))
        current_score, _, index, current, calls = winner
        released = combine_supported_calls(released, [calls])
        accepted.append(dict(gain=float(current_score-fit(current[:-1])["score"]),
                             private_sites=int(private_allele_mask(released[:-1], calls).sum())))
        remaining.pop(index)
        remaining = [(r,h) for r,h in remaining if not any(np.array_equal(h,x) for x in current)]
    label = "nonmosaic1_bic_add" if selection == "balanced" else "private_bic_add"
    diagnostic["accepted"][label] = accepted
    diagnostic["scored_panels"] = len(cache)
    if not accepted:
        return dict(feedback, diagnostic=diagnostic)
    if selection == "balanced":
        result = dict(polish(current))
        result["discrete_haps"] = result.pop("calls")
    else:
        fitted = fit(current)
        # Keep the old rows/calls, and only release supported,
        # exact-distinct additions. Preserve metadata alignment.
        calls = np.vstack((scaffold, fitted["calls"][len(backbone):]))
        indices = list(range(len(backbone)))
        seen = {row.tobytes() for row in scaffold}
        for i in range(len(backbone), len(calls)):
            key = calls[i].tobytes()
            if np.any(calls[i] >= 0) and key not in seen:
                seen.add(key)
                indices.append(i)
        remap = np.full(len(current)+1, len(indices), dtype=np.int64)
        remap[indices] = np.arange(len(indices))
        result = dict(discrete_haps=calls[indices],
            latent_haps=current[indices],
            q=np.vstack((feedback["q"], fitted["q"][len(backbone):]))[indices],
            support=np.vstack((feedback["support"], fitted["support"][len(backbone):]))[indices],
            assignments=remap[fitted["assignments"]])
    result["diagnostic"] = diagnostic
    return result


def reduce_empty_rows(evidence, observed_mask, result, *, config=None,
                      selection="balanced"):
    """Test smaller panels only when a released row is wholly unsupported.

    Deletion is a model proposal, not an array filter. Refit assignments (and,
    for balanced selection, alleles), accept under the existing BIC-like
    objective, and rebuild all call/support metadata. A losing deletion keeps
    the uncertainty explicit; partially called rows never initiate deletion.
    Strict selection also requires preservation of surviving backbone calls.
    """
    if selection not in ("balanced", "strict"):
        raise ValueError("selection must be balanced or strict")
    calls = np.asarray(result["discrete_haps"])
    if len(calls) < 2 or not np.any(~np.any(calls >= 0, axis=1)):
        return result
    base = search.ReversibleCavitySearchConfig() if config is None else config
    observed = np.ascontiguousarray(observed_mask, dtype=np.bool_)
    likelihood = np.array(evidence, dtype=np.float64, order="C", copy=True)
    likelihood[~observed] = 1.0 / 3.0
    workspace = fitting._prepare_fixed_k_fit_workspace(
        likelihood, base.lambda_wildcard_penalty, observed_mask=observed)
    complexity = objectives.compute_founder_complexity_cost(
        .5, int(observed.any(axis=1).sum()), likelihood.shape[1])
    current = dict(result)
    diagnostic = dict(initial_k=len(calls), initial_empty_rows=int(
        (~np.any(calls >= 0, axis=1)).sum()), reductions=[], tested_modes=0)

    def release(panel):
        assignment = workspace.compute_assignment_scalar(panel)
        q, support, _, _, values = haplotypes._materialize_founder_site_pseudo_evidence(
            likelihood, panel, assignment[0], observed, base.lambda_wildcard_penalty,
            base.min_directional_supporters, base.min_hard_call_pseudo_probability)
        return dict(discrete_haps=values, latent_haps=panel, q=q, support=support,
                    assignments=assignment[0],
                    score=float(-assignment[4] - .5 * len(panel) * complexity))

    while len(current["discrete_haps"]) > 1:
        panel = np.ascontiguousarray(current["latent_haps"], dtype=np.int64)
        empty = np.flatnonzero(~np.any(current["discrete_haps"] >= 0, axis=1))
        if not len(empty):
            break
        baseline_score = float(
            -workspace.compute_assignment_scalar(panel)[4] - .5 * len(panel) * complexity)
        starts = [np.delete(panel, int(row), axis=0) for row in empty]
        if 1 < len(empty) < len(panel):
            starts.append(np.delete(panel, empty, axis=0))
        parallel.apply_dynamic_threads()
        raw, fitted = modes._fit_starts_with_synchronized_endpoints(
            likelihood, starts, search._internal_move_config(base), workspace,
            max_iter=(0 if selection == "strict" else None))
        options = []
        for mode in modes._deduplicate_modes((*raw, *fitted)):
            if mode.k >= len(panel):
                continue
            diagnostic["tested_modes"] += 1
            candidate_score = -mode.total_nll - .5 * mode.k * complexity
            if candidate_score < baseline_score - base.score_tolerance:
                continue
            candidate = release(mode.haplotypes)
            if selection == "strict":
                lookup = {row.tobytes(): i for i, row in enumerate(mode.haplotypes)}
                protects_calls = True
                for old, row in enumerate(panel):
                    known = current["discrete_haps"][old] >= 0
                    if not known.any():
                        continue
                    new = lookup.get(row.tobytes())
                    if new is None or not np.array_equal(
                            candidate["discrete_haps"][new, known],
                            current["discrete_haps"][old, known]):
                        protects_calls = False
                        break
                if not protects_calls:
                    continue
            options.append((candidate, mode.canonical_key))
        if not options:
            break
        best, _ = min(options, key=lambda item: (
            -item[0]["score"], len(item[0]["discrete_haps"]), item[1]))
        diagnostic["reductions"].append(dict(
            before_k=len(panel), after_k=len(best["discrete_haps"]),
            score_gain=best["score"] - baseline_score))
        current = best
    diagnostic["final_k"] = len(current["discrete_haps"])
    diagnostic["remaining_empty_rows"] = int(
        (~np.any(current["discrete_haps"] >= 0, axis=1)).sum())
    # Selection diagnostics remain diagnostic: do not reuse pre-reduction
    # scores or assignments as if they described the reduced fit.
    current["diagnostic"] = dict(result.get("diagnostic", {}),
                                 empty_row_reduction=diagnostic)
    return current
