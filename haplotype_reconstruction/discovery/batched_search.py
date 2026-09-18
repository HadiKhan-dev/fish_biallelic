"""Opt-in bounded-panel search with the existing reversible cavity objective.

Growth and pruning propose whole batches; fixed-K fitting and held-out
mean-field scoring are unchanged. This is a heuristic neighbourhood, not a
certificate for every K or every founder panel. Missing evidence is normalized
by the public discovery entry point before this module is called.
"""
from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace
import math
import numpy as np
from numba import njit


@dataclass(frozen=True)
class BatchedSearchConfig:
    max_rounds: int = 20
    fits_per_round: int = 24
    birth_starts: int = 8
    max_lookahead_rounds: int = 2
    cut_starts: int = 8
    cut_sweeps: int = 20
    residual_samples: int = 8

    def __post_init__(self):
        for value in (
            self.max_rounds,
            self.fits_per_round,
            self.birth_starts,
            self.max_lookahead_rounds,
            self.cut_starts,
            self.cut_sweeps,
            self.residual_samples
        ):
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError('batched discovery budgets must be positive integers')


def configured_search():
    from..core.environment import batched_discovery_enabled
    return BatchedSearchConfig() if batched_discovery_enabled() else None


@njit(cache=True)
def _improve_cut(weights, initial, order, sweeps):
    side = initial.copy()
    for _ in range(sweeps):
        changed = False
        for i in order:
            delta = 0
            for j in range(len(side)):
                delta += weights[i, j] if side[i] == side[j] else -weights[i, j]
            if delta > 0:
                side[i] = not side[i]
                changed = True
        if not changed:
            break
    return side


def bounded_cuts(weights, config, max_ties):
    """Fixed starts/passes; O(K²), including deterministic degree ordering."""
    from.import modes
    k = len(weights)
    order = np.argsort(-weights.sum(axis=1), kind='stable')
    rng = np.random.default_rng(719)
    seen = {}
    for start in range(config.cut_starts):
        initial = np.arange(k) % 2 == 0 if start == 0 else rng.integers(0, 2, k).astype(bool)
        side = _improve_cut(weights, initial, order, config.cut_sweeps)
        if side[0]:
            side = ~side
        seen[side.tobytes()] = side
    return tuple(
        sorted(seen.values(), key=lambda side: (-modes._cut_score(weights, side)[0], side.tobytes()))[:max_ties]
    )


def residual_rows(mode, evidence, reads, settings, growth, residual_workspace):
    from.import candidates, search
    adapter = SimpleNamespace(discrete_haps=mode.haplotypes, pair_assignments=mode.assignments,
        K_final=mode.k, keep_flags=np.ones(mode.n_sites, np.int8),
        precleanup_candidate_discrete_haps=mode.haplotypes, precleanup_candidate_k=mode.k, haplotypes={})
    augmented = candidates.augment_combined_soft_candidates(adapter, reads,
        base_candidates=mode.haplotypes.astype(float), read_error_probability=settings.read_error_probability,
        minimum_soft_unique_sample_support=settings.min_soft_unique_sample_support,
        residual_input_workspace=residual_workspace, binary_panel_fast_path=True)
    rows = list(search._ordered_binary_candidate_rows(
        augmented.candidates[augmented.n_base_candidates:], evidence))
    excess = np.maximum(mode.per_sample_cost - growth.oracle_nll, 0.)
    active = np.flatnonzero(growth.active_sample_mask)
    order = active[np.lexsort((active, -growth.decisiveness[active], -excess[active]))]
    for sample in order[:settings.batched_search_config.residual_samples]:
        dosage = growth.dosage_by_sample[sample]
        rows.extend(np.clip(dosage[None,:] - mode.haplotypes, 0, 1))
        rows.append(growth.seed_haplotypes_by_sample[sample])
    existing = {np.asarray(row, dtype=np.int8).tobytes() for row in mode.haplotypes}
    unique = {}
    for row in rows:
        key = np.asarray(row, dtype=np.int8).tobytes()
        if key not in existing:
            unique.setdefault(key, np.asarray(row, dtype=np.int64))
    return list(unique.values())


def run(evidence, reads, seeds, candidate_rows, settings, workspace, growth, residual_workspace):
    from.import search, modes, cavity
    cfg = settings.batched_search_config
    active = growth.active_sample_mask
    n_sites = evidence.shape[1]
    ceiling = search._natural_k_ceiling(int(active.sum()), n_sites)
    fit_config = search._internal_move_config(settings)
    score_config = search._stage_config(settings.cavity, 'mean_field')
    score_evidence = evidence if np.all(active) else np.ascontiguousarray(evidence[active])
    score_workspace = cavity._prepare_cavity_scoring_workspace(score_evidence, score_config)
    archive = {}
    cache = {}
    scores = {}
    fitted_count = 0
    steps = []
    exhausted = False
    def fit(starts):
        nonlocal fitted_count, exhausted
        unique = {}
        for start in starts:
            panel = np.ascontiguousarray(start, dtype=np.int64)
            if panel.ndim != 2 or panel.shape[1] != n_sites or len(panel) < 1 or len(panel) > ceiling:
                raise ValueError('invalid complete-panel seed')
            if np.any((panel != 0) & (panel != 1)):
                raise ValueError('panel seeds must be binary')
            unique.setdefault(modes._canonical_haplotype_key(panel), panel)
        retained = list(unique.values())[:cfg.fits_per_round]
        fitted_count += len(retained)
        for mode in search._fit_panel_starts(evidence, retained, fit_config, workspace):
            previous = archive.get(mode.canonical_key)
            if previous is None or search._mode_order(mode) < search._mode_order(previous):
                archive[mode.canonical_key] = mode
        grouped = {}
        for mode in archive.values():
            old = grouped.get(mode.k)
            if old is None or search._mode_order(mode) < search._mode_order(old):
                grouped[mode.k] = mode
        novel = [mode for mode in grouped.values() if mode.canonical_key not in cache]
        room = max(0, settings.max_exact_scores - len(cache))
        exhausted |= len(novel) > room
        for stage in search._score_stage(evidence, novel[:room], score_config, score_workspace, active):
            cache[stage.mode.canonical_key] = search.ReversibleModeScore(stage.mode, stage.digest,
                math.nan, math.nan, stage.cavity_log_predictive, stage.log_k_prior,
                stage.log_haplotype_set_prior, stage.log_score, stage.diagnostic, stage.diagnostic)
        for k, mode in grouped.items():
            if mode.canonical_key in cache:
                scores[k] = cache[mode.canonical_key]

    initial = modes._initial_complete_modes(evidence, settings.data_start_beam_width,
        settings.n_data_seed_modes, settings.soft_seed_min_cluster_size,
        settings.lambda_wildcard_penalty, settings.coordinate_descent_max_iter,
        fit_workspace=workspace, seed_sample_mask=active)
    starts = [mode.haplotypes for mode in initial]
    starts.extend(seed.haplotypes if isinstance(seed, modes.FactorizationMode) else seed for seed in seeds)
    starts.extend(row[None,:] for row in candidate_rows[:settings.max_candidate_start_rows])
    fit(starts)
    stop = 'round_budget_exhausted'
    exploration_parent = None
    expanded = set()
    non_improving_rounds = 0
    for iteration in range(cfg.max_rounds):
        selected = min(scores.values(), key=search._score_sort_key)
        parent = selected.mode if exploration_parent is None else exploration_parent
        k = parent.k
        expanded.add(parent.canonical_key)
        rows = residual_rows(parent, evidence, reads, settings, growth, residual_workspace)
        existing = {np.asarray(row, dtype=np.int8).tobytes() for row in parent.haplotypes}
        rows.extend(row for row in candidate_rows if np.asarray(row, dtype=np.int8).tobytes() not in existing)
        # Large moves first: O(1) refits can grow from K to 2K, followed by
        # joint pruning. A single-row candidate remains available as a rescue.
        starts = []
        if rows and k < ceiling:
            batch = min(k, len(rows), ceiling - k)
            starts.append(np.vstack((parent.haplotypes, np.asarray(rows[:batch]))))
            # A single failed seed is not evidence against a new founder.
            # Keep a fixed number of distinct residual basins, independent K.
            for row in rows[:cfg.birth_starts]:
                starts.append(np.vstack((parent.haplotypes, row)))
            if len(rows) > batch:
                shifted = rows[batch:2 * batch]
                starts.append(np.vstack((parent.haplotypes, np.asarray(shifted))))
        occupancy = np.bincount(parent.assignments.ravel(), minlength=k + 1)[:k]
        order = np.argsort(occupancy, kind='stable')
        if k > 1:
            for count in dict.fromkeys((int(np.sum(occupancy == 0)), max(1, k // 4), max(1, k // 2), 1)):
                if 0 < count < k:
                    starts.append(np.delete(parent.haplotypes, order[:count], axis=0))
        if rows:
            count = min(k, len(rows))
            replacement = parent.haplotypes.copy()
            replacement[order[:count]] = rows[:count]
            starts.append(replacement)
        if settings.apply_gauge_rewire and k > 1:
            weights = modes.assignment_graph(parent)
            cuts = bounded_cuts(weights, cfg, settings.max_cut_ties)
            starts.extend(proposal.haplotypes for proposal in modes._propose_bipartite_gauge_starts(
                parent, evidence, exact_cut_max_k=settings.exact_cut_max_k, max_cut_ties=settings.max_cut_ties,
                assignment_weights=weights, cut_partitions=cuts))
        before = len(cache)
        before_fits = fitted_count
        fit(starts)
        new = min(scores.values(), key=search._score_sort_key)
        steps.append(dict(round=iteration + 1, k=k, selected_k=new.k, fits=fitted_count - before_fits,
            exact_scores=len(cache) - before, log_score=new.log_score))
        exploration_parent = None
        if new.log_score <= selected.log_score + settings.score_tolerance:
            non_improving_rounds += 1
            if non_improving_rounds >= cfg.max_lookahead_rounds:
                stop = 'no_improving_batched_lookahead'
                break
            # A temporary lower-scoring intermediate can open a useful basin.
            # Explore bounded lookahead and alternative same-K basins; never
            # replace the released incumbent without its cavity-score win.
            limit = min(ceiling, max(4, 2 * new.k))
            alternatives = [mode for mode in archive.values()
                if mode.canonical_key not in expanded and new.k <= mode.k <= limit]
            if alternatives:
                exploration_parent = min(
                    alternatives,
                    key=lambda mode: (-mode.k, mode.total_nll, mode.canonical_key)
                )
            else:
                stop = 'no_improving_batched_neighbour'
                break
        else:
            non_improving_rounds = 0
        if exhausted:
            stop = 'exact_score_budget'
            break
    ordered = sorted(scores.values(), key=search._score_sort_key)
    selected = ordered[0]
    limits = ('batched_candidate_neighbourhood',) + (('exact_score_budget',) if exhausted else ())
    if any(item.mean_field_diagnostic.n_mean_field_not_converged for item in ordered):
        limits += ('mean_field_nonconvergence',)
    cert = search.LocalSearchCertificate(selected.k, selected.mode_digest, selected.log_score,
        False, (), (), (), False, False, None, False, limits,
        'Bounded batch births/deaths, replacements and gauge starts; no exhaustive-neighbourhood certificate.')
    logs = np.asarray([item.log_score for item in ordered])
    weights = np.exp(logs - logs.max())
    weights /= weights.sum()
    grouped = {}
    for mode in archive.values():
        grouped.setdefault(mode.k, []).append(mode)
    result = search.ReversibleCavitySearchResult(
        selected=selected, runner_up=ordered[1] if len(ordered) > 1 else None,
        visited_scores=tuple(sorted(ordered, key=lambda item: item.k)), anchored_scores=(),
        visited_modes_by_k=tuple((k, tuple(sorted(values, key=search._mode_order))) for k, values in sorted(grouped.items())),
        best_score_by_k=tuple(sorted((item.k, item.log_score) for item in ordered)),
        pseudo_probability_by_k=tuple(sorted((item.k, float(w)) for item, w in zip(ordered, weights))),
        search_steps=(), local_certificate=cert, natural_k_ceiling=ceiling,
        n_input_samples=len(evidence), n_scored_samples=int(active.sum()), search_limited=True,
        search_limit_reasons=limits, boundary_limited=selected.k == ceiling, stop_reason=stop,
        data_start_count=len(initial), overcomplete_data_start_count=0, supplied_panel_start_count=len(seeds),
        candidate_row_count=len(candidate_rows), exact_score_evaluations=len(cache), exact_score_cache_hits=0,
        anchored_score_evaluations=0, high_k_tail_mass_upper_bound=math.nan,
        high_k_tail_log_mass_upper_bound=math.nan, high_k_tail_pseudo_probability_upper_bound=math.nan,
        high_k_tail_bound_interpretation='No tail bound for adaptively visited K.',
        objective_interpretation='Same minimum-NLL representative per K and mean-field cavity objective; weights are uncalibrated.',
        search_interpretation=f'Bounded batched search: {fitted_count} starts fitted; {steps!r}')
    return result
