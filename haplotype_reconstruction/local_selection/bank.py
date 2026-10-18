"""The candidate bank of a context round: which rows the path search may add, and where it starts.

Context proposals are candidates, never observations. Each proposal panel's partial rows are
completed from the nearest raw-discovery latent rows on their called sites (every tied
completion enters the bank; called alleles are never overwritten). Every source (the latent
panel, then each proposal panel) is refitted with discovery's hard fixed-K model and its
endpoints are cavity-scored; then the union bank and its drop / add / replace neighbourhoods are
searched under the switch-penalised BIC (cavity scores deferred: neither ranking nor stopping
reads them). The path search starts from the latest source's cavity-best endpoint released
with the raw-discovery rule (released alleles, latent rows where withheld); the bank adds the
raw latent rows, that start and the BIC-selected panel. Neither score replaces the path
objective.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..discovery.cavity import CavityWorkspace, score_solutions
from ..discovery.fixed_k.fit import fit_panels, prepare_workspace
from ..discovery.panel import deduplicate, panel_key, unique_rows
from ..discovery.release import release_alleles
from ..execution import claim_threads
from .fit import unique_rows as unique_integer_rows


@dataclass(frozen=True)
class BankSummary:
    """bank_rows: distinct candidate rows; fitted_starts / scored: novel starts fitted and
    panels admitted; selected_k: the BIC-selected K; rounds: neighbourhood rounds; budget_bound:
    a reported backstop was hit (start, score or round backstop)."""

    bank_rows: int
    fitted_starts: int
    scored: int
    selected_k: int
    rounds: int
    budget_bound: bool


def complete_proposals(original_latent, proposal_panels):
    """(sources, bank rows, missing cells): the latent panel and each proposal panel's completed
    rows as search sources, and every distinct completion as a bank row."""
    original = unique_rows(original_latent).astype(np.int64)
    sources = [original]
    bank = [original]
    missing = 0
    for panel in proposal_panels:
        panel = np.asarray(panel)
        completed, alternatives = [], []
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
            donors = unique_rows(donors)
            completed.append(donors[0])
            alternatives.extend(donors)
        if completed:
            complete = unique_rows(np.asarray(completed))
            sources.append(complete)
            bank.append(complete)
        else:
            sources.append(np.empty((0, original.shape[1]), np.int64))
        if alternatives:
            bank.append(np.asarray(alternatives))
    return tuple(sources), unique_rows(np.vstack(bank)).astype(np.int64), missing


def _neighbours(panel, bank):
    """Drops, adds of every bank row not in the panel, then replacements (closest first)."""
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


@dataclass(frozen=True)
class _Deferred:
    """An admitted panel whose cavity score is deferred (BIC bank search)."""

    solution: object


def _bic_search(likelihoods, observed, original_latent, proposal_panels, discovery_config,
                bank_config):
    """(BIC-selected solution, released cavity-best endpoint per source (None for an empty
    source), summary)."""
    model = discovery_config.model
    likelihood = np.array(likelihoods, dtype=np.float64, order="C", copy=True)
    observed = np.ascontiguousarray(observed, dtype=np.bool_)
    active = observed.any(axis=1)
    likelihood[~observed] = 1.0 / 3.0
    sources, bank, _missing = complete_proposals(original_latent, proposal_panels)
    workspace = prepare_workspace(likelihood, observed, model)
    cavity = CavityWorkspace(likelihood if np.all(active) else likelihood[active],
                             discovery_config.cavity)
    scored: dict = {}
    fitted_starts: set = set()
    endpoint_keys: dict = {}
    state = dict(budget_bound=False, selected=None)
    complexity = bank_config.complexity_scale * (likelihood.shape[1] / 200.0) * int(active.sum())

    def selection_score(entry) -> float:
        # The fixed-K fitter's binned switching-Viterbi NLL in the outer BIC convention
        # K * complexity - 2 log L (lower is better), negated and halved.
        solution = entry.solution
        return -0.5 * (solution.k * complexity - 2.0 * (-solution.nll))

    def rank(entry):
        solution = entry.solution
        return (-selection_score(entry), solution.nll, solution.k, solution.key)

    def fit_and_score(panels, require_cavity=False):
        selected = state["selected"]
        k = (selected.solution.k if selected is not None
             else max((len(panel) for panel in panels), default=0))
        novel = []
        for panel in panels:
            if not len(panel):
                continue
            panel = unique_rows(panel).astype(np.int64)
            key = panel.astype(np.int8).tobytes()
            if key not in fitted_starts:
                fitted_starts.add(key)
                novel.append(panel)
        if not novel:
            keys = set()
            for panel in panels:
                if len(panel):
                    keys.update(endpoint_keys.get(panel_key(panel), ()))
            return tuple(scored[key] for key in keys if key in scored)
        start_limit = bank_config.start_backstop(k)
        if len(novel) > start_limit:
            state["budget_bound"] = True
            for panel in novel[start_limit:]:
                fitted_starts.remove(panel.astype(np.int8).tobytes())
            novel = novel[:start_limit]
        claim_threads()
        synchronized, refitted = fit_panels(workspace, novel, model.max_iterations)
        unique = deduplicate((*synchronized, *refitted))
        if len(novel) == 1:
            endpoint_keys[panel_key(novel[0])] = tuple(solution.key for solution in unique)
        pending = [solution for solution in unique if solution.key not in scored]
        room = max(0, bank_config.score_backstop(k) - len(scored))
        if len(pending) > room:
            state["budget_bound"] = True
            pending = sorted(pending, key=lambda solution: solution.order())[:room]
        if require_cavity:
            for score in (score_solutions(cavity, pending, active) if pending else ()):
                scored[score.solution.key] = score
        else:
            for solution in pending:
                scored[solution.key] = _Deferred(solution)
        return tuple(scored[solution.key] for solution in unique if solution.key in scored)

    # Each source separately first: fair source-only refits that protect every input basin.
    cavity_best = []
    for panel in sources:
        values = fit_and_score([panel], require_cavity=True)
        if not values and len(panel):
            key = panel_key(panel)
            values = (scored[key],) if key in scored else ()
        cavity_best.append(min(values, key=lambda score: (
            -score.log_score, score.solution.nll, score.solution.k, score.solution.key))
            if values else None)
    fit_and_score([bank])
    selected = state["selected"] = min(scored.values(), key=rank)
    rounds = 0
    while True:
        if rounds >= bank_config.round_backstop(selected.solution.k):
            state["budget_bound"] = True
            break
        before = selected
        fit_and_score(_neighbours(selected.solution.haplotypes, bank))
        selected = state["selected"] = min(scored.values(), key=rank)
        rounds += 1
        if selection_score(selected) <= selection_score(before) + bank_config.improvement_tolerance:
            break
        if len(scored) >= bank_config.score_backstop(selected.solution.k):
            state["budget_bound"] = True
            break
    releases = {}

    def released(score):
        """(released calls, latent rows) of a cavity-scored endpoint (raw-discovery rule)."""
        if score is None:
            return None
        solution = score.solution
        key = (solution.key, solution.assignments.tobytes())
        if key not in releases:
            claim_threads()
            _, _, calls = release_alleles(likelihood, solution.haplotypes, solution.assignments,
                                          observed, model.unknown_copy_penalty,
                                          discovery_config.release)
            releases[key] = calls
        return releases[key], solution.haplotypes

    summary = BankSummary(bank_rows=len(bank), fitted_starts=len(fitted_starts), scored=len(scored),
                          selected_k=selected.solution.k, rounds=rounds,
                          budget_bound=bool(state["budget_bound"]))
    return selected.solution, tuple(released(score) for score in cavity_best), summary


def candidate_bank(likelihoods, observed, original_latent, proposal_panels, discovery_config,
                   bank_config, incumbent_panel=None):
    """(start panel, bank rows, summary or None) of a block's path search.

    Without proposals (the initial round) the start is the raw latent panel and the bank its
    rows. An incumbent (the previous round's selected panel) contributes all its rows.
    """
    original_latent = np.asarray(original_latent, dtype=np.int64)
    proposal_panels = list(proposal_panels)
    summary = None
    if proposal_panels:
        selected, endpoints, summary = _bic_search(
            likelihoods, observed, original_latent, proposal_panels, discovery_config, bank_config)
        calls, latent = next(value for value in reversed(endpoints) if value is not None)
        start = np.where(calls >= 0, calls, latent)
        bank = unique_integer_rows(np.vstack((original_latent, start, selected.haplotypes)))
    else:
        start = original_latent
        bank = unique_integer_rows(start)
    if incumbent_panel is not None:
        bank = unique_integer_rows(np.vstack((bank, incumbent_panel)))
    return start, bank, summary
