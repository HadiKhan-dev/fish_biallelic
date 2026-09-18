"""Local selection from independent reconstruction proposals.

Original and feedback panels propose haplotypes, not extra observations. The
read likelihood, depth mask and fixed-K fitter are the existing discovery model.
Multiple same-K panels compete under either the regularized mean-field cavity
score or the existing switch-penalized BIC convention. These are distinct
experimental selection criteria, not claimed mathematically equivalent.
Neither truth nor downstream assembly/painting scores enter selection.

The feedback workflow uses the BIC bank and cavity-ranked source endpoints
together; the two criteria are not interchangeable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from.import cavity, fitting, modes, objectives, search
from..core import haplotypes, parallel


@dataclass(frozen=True)
class CandidateSelectionConfig:
    discovery: search.ReversibleCavitySearchConfig = field(
        default_factory=search.ReversibleCavitySearchConfig)
    max_rounds: int = 8
    max_scores: int = 256
    max_starts_per_round: int = 128
    criterion: str = "cavity"

    def __post_init__(self):
        if self.criterion not in ("cavity", "bic"):
            raise ValueError("criterion must be cavity or bic")
        for value in (self.max_rounds, self.max_scores, self.max_starts_per_round):
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError("candidate search budgets must be positive integers")


@dataclass
class CandidateSelectionResult:
    selected_mode: modes.FactorizationMode
    discrete_haps: np.ndarray
    founder_alt_pseudo_probability: np.ndarray
    n_directional_site_supporters: np.ndarray
    source_results: tuple[dict | None, ...]
    diagnostic: dict
    cavity_source_results: tuple = ()


def prepare_candidate_panels(original_latent, proposal_panels):
    """Exact-dedup panels; missing proposal alleles remain initialization only.

The original latent panel supplies binary optimization starts, NOT confident
calls. Partial feedback rows get completions from nearest original latent
rows on their observed overlap; all tied distinct completions enter the bank.
Known feedback alleles are never overwritten. An entirely unknown row adds no
candidate. No near-duplicate row is automatically merged or discarded.
"""
    original = np.asarray(original_latent)
    if original.ndim != 2 or not original.size:
        raise ValueError("original latent panel must be nonempty and two-dimensional")
    original = objectives.exact_unique_binary_rows(original).astype(np.int64)
    starts = [original]
    bank = [original]
    missing = 0
    for panel in proposal_panels:
        panel = np.asarray(panel)
        if panel.ndim != 2 or panel.shape[1] != original.shape[1]:
            raise ValueError("candidate panels must have the same marker order/length")
        if np.any((panel != -1) & (panel != 0) & (panel != 1)):
            raise ValueError("candidate calls must be -1, 0 or 1")
        completed = []
        alternatives = []
        for row in panel:
            known = row >= 0
            missing += int((~known).sum())
            if not known.any():
                continue
            if known.all():
                completed.append(row.astype(np.int64))
                continue
            distance = np.sum(original[:, known] != row[known], axis=1)
            donors = original[distance == distance.min()].copy()
            donors[:, known] = row[known]
            donors = objectives.exact_unique_binary_rows(donors)
            completed.append(donors[0])
            alternatives.extend(donors)
        if completed:
            complete = objectives.exact_unique_binary_rows(np.asarray(completed))
            starts.append(complete)
            bank.append(complete)
        else:
            # Keep source order in the diagnostic even for an empty proposal.
            starts.append(np.empty((0, original.shape[1]), np.int64))
        if alternatives:
            bank.append(np.asarray(alternatives))
    combined = objectives.exact_unique_binary_rows(np.vstack(bank)).astype(np.int64)
    return tuple(starts), combined, missing


def _neighbours(panel, bank):
    """Drop, add and replace proposals; joint refitting also tests mergers.

Closest replacements are examined first if the explicit work budget binds.
Deleting either near-duplicate and refitting its carriers tests retaining one
representative. Distance never determines acceptance. All candidate distances
are eligible, including quite distinct missing founder haplotypes.
    """
    k = len(panel)
    proposals = []
    present = {row.tobytes() for row in panel}
    extra = [row for row in bank if row.tobytes() not in present]
    if k > 1:
        for i in range(k):
            proposals.append(np.delete(panel, i, axis=0))
    for row in extra:
        proposals.append(np.vstack((panel, row)))
    replacements = []
    for row in extra:
        for i in range(k):
            replacements.append((int(np.sum(row != panel[i])), i, row.tobytes(), row))
    for _, i, _, row in sorted(replacements, key=lambda value: value[:3]):
        new = panel.copy()
        new[i] = row
        proposals.append(new)
    return proposals


def select_candidate_panel(evidence, *, allele_depths=None, observed_mask=None,
                           original_latent, proposal_panels, config=None):
    """Refit competing local panels under the explicitly configured criterion.

Both assignment-only endpoints (preserving a proposed H) and full allele
refits compete, using the existing synchronized fitter. Final calls use the
canonical missing-aware fixed-assignment release rule, not the latent seed.
Scores remain selection-leakage-affected pseudo-scores, not calibrated LOO.
    """
    settings = CandidateSelectionConfig() if config is None else config
    base = settings.discovery
    likelihood = np.array(modes._validate_evidence(evidence), copy=True, order="C")
    if observed_mask is None:
        reads = np.asarray(allele_depths)
        if reads.shape != (*likelihood.shape[:2], 2):
            raise ValueError("read counts and evidence must have matching samples/sites")
        if np.any(~np.isfinite(reads)) or np.any(reads < 0):
            raise ValueError("read counts must be finite and nonnegative")
        observed_mask = reads.sum(axis=2) > 0
    observed = np.ascontiguousarray(observed_mask, dtype=np.bool_)
    if observed.shape != likelihood.shape[:2]:
        raise ValueError("observed mask and evidence must have matching samples/sites")
    active = observed.any(axis=1)
    if not active.any():
        raise ValueError("a local panel needs at least one observed sample")
    likelihood[~observed] = 1.0 / 3.0
    starts, bank, missing = prepare_candidate_panels(original_latent, proposal_panels)
    if bank.shape[1] != likelihood.shape[1]:
        raise ValueError("panel markers must match evidence")
    workspace = fitting._prepare_fixed_k_fit_workspace(
        likelihood, base.lambda_wildcard_penalty, observed_mask=observed)
    score_config = search._stage_config(base.cavity, "mean_field")
    score_evidence = likelihood if np.all(active) else np.ascontiguousarray(likelihood[active])
    score_workspace = cavity._prepare_cavity_scoring_workspace(score_evidence, score_config)
    fit_config = search._internal_move_config(base)
    scored = {}
    fitted_starts = set()
    endpoint_keys = {}
    mode_kinds = {}
    budget_bound = False
    complexity = objectives.compute_founder_complexity_cost(
        0.5, int(active.sum()), likelihood.shape[1])

    def selection_score(score):
        if settings.criterion == "bic":
            # The canonical fixed-K fitter reports -Viterbi score with its
            # existing binned switching/wildcard model and depth mask. This
            # is the existing BIC convention, not a calibrated marginal LL.
            return -0.5 * objectives.compute_outer_bic_from_log_likelihood(
                score.mode.k, -score.mode.total_nll, complexity)
        return score.log_score

    def rank(score):
        return (-selection_score(score), score.mode.total_nll, score.mode.k,
                score.mode.canonical_key)

    def fit_and_score(panels):
        nonlocal budget_bound
        novel = []
        for panel in panels:
            if not len(panel):
                continue
            panel = objectives.exact_unique_binary_rows(panel).astype(np.int64)
            key = panel.astype(np.int8).tobytes()
            if key not in fitted_starts:
                fitted_starts.add(key)
                novel.append(panel)
        if not novel:
            keys = set()
            for panel in panels:
                if len(panel):
                    keys.update(endpoint_keys.get(modes._canonical_haplotype_key(panel), ()))
            return tuple(scored[key] for key in keys if key in scored)
        if len(novel) > settings.max_starts_per_round:
            budget_bound = True
            for panel in novel[settings.max_starts_per_round:]:
                fitted_starts.remove(panel.astype(np.int8).tobytes())
            novel = novel[:settings.max_starts_per_round]
        parallel.apply_dynamic_threads()
        raw, refitted = modes._fit_starts_with_synchronized_endpoints(
            likelihood, novel, fit_config, workspace)
        for label, collection in (("assignment_refit", raw), ("allele_refit", refitted)):
            for mode in collection:
                mode_kinds.setdefault(mode.canonical_key, set()).add(label)
        unique = modes._deduplicate_modes((*raw, *refitted))
        # Keep both endpoints for identical source-only refit controls.
        if len(novel) == 1:
            endpoint_keys[modes._canonical_haplotype_key(novel[0])] = tuple(
                mode.canonical_key for mode in unique)
        pending = [mode for mode in unique if mode.canonical_key not in scored]
        room = max(0, settings.max_scores - len(scored))
        if len(pending) > room:
            budget_bound = True
            pending = sorted(pending, key=search._mode_order)[:room]
        for score in search._score_stage(
                likelihood, pending, score_config, score_workspace, active) if pending else ():
            scored[score.mode.canonical_key] = score
        return tuple(scored[mode.canonical_key] for mode in unique
                     if mode.canonical_key in scored)

    # Score each source separately before union/neighbour proposals. This
    # provides fair source-only refit controls and protects all input basins.
    source_scores = []
    cavity_source_scores = []
    for panel in starts:
        values = fit_and_score([panel])
        if not values and len(panel):
            key = modes._canonical_haplotype_key(panel)
            values = (scored[key],) if key in scored else ()
        source_scores.append(min(values, key=rank) if values else None)
        cavity_source_scores.append(min(values, key=lambda s: (
            -s.log_score, s.mode.total_nll, s.mode.k, s.mode.canonical_key))
            if values else None)
    if not scored:
        raise RuntimeError("no candidate panel could be scored")
    fit_and_score([bank])
    rounds = []
    selected = min(scored.values(), key=rank)
    for iteration in range(settings.max_rounds):
        before = selected
        old_count = len(scored)
        fit_and_score(_neighbours(selected.mode.haplotypes, bank))
        selected = min(scored.values(), key=rank)
        rounds.append(dict(round=iteration + 1, scores=len(scored) - old_count,
                           k=selected.mode.k, score=float(selection_score(selected))))
        if selection_score(selected) <= selection_score(before) + base.score_tolerance:
            break
        if len(scored) >= settings.max_scores:
            budget_bound = True
            break

    def release(score):
        if score is None:
            return None
        mode = score.mode
        parallel.apply_dynamic_threads()
        q, support, _, _, calls = haplotypes._materialize_founder_site_pseudo_evidence(
            likelihood, mode.haplotypes, mode.assignments, observed,
            base.lambda_wildcard_penalty, base.min_directional_supporters,
            base.min_hard_call_pseudo_probability)
        return dict(discrete_haps=calls, q=q, support=support,
                    latent_haps=mode.haplotypes, score=float(selection_score(score)),
                    cavity_score=float(score.log_score), nll=mode.total_nll,
                    assignments=mode.assignments)

    released = release(selected)
    controls = tuple(release(score) for score in source_scores)
    cavity_controls = tuple(release(score) for score in cavity_source_scores)
    ordered = sorted(scored.values(), key=rank)
    return CandidateSelectionResult(
        selected.mode, released["discrete_haps"], released["q"], released["support"],
        controls, dict(
            bank_size=len(bank), source_sizes=[len(s) for s in starts],
            partial_seed_cells=missing, fitted_starts=len(fitted_starts),
            scored_modes=len(scored), score=float(selection_score(selected)),
            criterion=settings.criterion, cavity_score=float(selected.log_score),
            selected_k=selected.mode.k, selected_kind=sorted(mode_kinds[selected.mode.canonical_key]),
            n_active_samples=int(active.sum()), search_budget_bound=budget_bound,
            rounds=rounds, n_cavity_nonconverged=sum(
                score.diagnostic.n_mean_field_not_converged for score in ordered),
            selected_cavity_nonconverged=selected.diagnostic.n_mean_field_not_converged,
            top_scores=[dict(k=s.mode.k, score=float(selection_score(s)),
                             cavity_score=float(s.log_score), nll=s.mode.total_nll,
                             digest=s.mode_digest if hasattr(s, "mode_digest") else s.digest)
                        for s in ordered[:8]],
            interpretation=(f"multiple same-K modes; {settings.criterion} selection; "
                            "uncalibrated; no truth/context likelihood")), cavity_controls)
