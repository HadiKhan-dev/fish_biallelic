"""Bounded count-up search after the final hierarchy refinement.

The incumbent and each K+1 proposal receive the same primal/dual and paired
fixed-count repair. They compete under K * complexity_cost - 2 * full_site_score,
using only paths through the existing local panels. No expected founder count,
invented allele, pedigree or truth enters the search. Stop at the first rejected
addition, with at most two additions by default.
"""
import time
import numpy as np

from .. import chimera_scoring, founder_refinement as refiner
from . import count, count_bound, dual_search, exchanges
from .checkpoints import FounderCheckpointStore
from .workspace import component_workspace, resolve_threads
from ...core import haplotypes, parallel


def _repair(panel, label, *, workspace, batch, original, neutral, sites,
            config, threads, checkpoints, workspaces):
    scope = count._ScopedCheckpoints(checkpoints, label)
    cached = scope.load("done")
    if cached is not None:
        return cached["panel"], cached["score"], cached["block"]
    score = workspace.canonical(panel)
    initial = count._reconstruct(panel, score, batch, original)
    result = count._fixed_count_refit(
        batch, initial, neutral, sites, config, threads,
        count._ScopedCheckpoints(scope, "initial"), None, workspaces)
    result, paired = exchanges.refine_components(
        batch, result, neutral, sites, count._ScopedCheckpoints(scope, "paired"),
        iterations=config.max_iterations, quota=config.branch_cap,
        workspaces=workspaces)
    if any(item["changed"] for item in paired["components"]):
        result = count._fixed_count_refit(
            batch, result[0], neutral, sites, config, threads,
            count._ScopedCheckpoints(scope, "after_pair"), None, workspaces)
    rows = refiner._local_selection(result[0], batch)
    final = workspace.canonical(rows)
    if len(rows) != len(panel) or final < score - 1e-7 * max(1., abs(score)):
        raise RuntimeError("count-up fixed-count repair changed count or reduced its objective")
    scope.save("done", dict(panel=rows, score=final, block=result[0]))
    return rows, final, result[0]


def refine_components(prepared, components, neutral, sites, *, config,
                      num_threads=1, checkpoints=None, cc_scale=0.5):
    """Refit count-up proposals independently within unchanged phase components."""
    if not config.enabled or config.count_max_additions == 0:
        return components, dict(enabled=False, components=[])
    if checkpoints is not None:
        checkpoints = FounderCheckpointStore(checkpoints, prepared)
    starts = {int(block.positions[0]): i for i, block in enumerate(prepared)}
    ends = {int(block.positions[-1]): i + 1 for i, block in enumerate(prepared)}
    outputs, diagnostics = [], []
    with parallel.numba_thread_scope(resolve_threads(num_threads)):
        for number, original in enumerate(components):
            scope = count._ScopedCheckpoints(checkpoints, f"count_up.component{number}")
            cached = scope.load("result")
            if cached is not None:
                outputs.append(cached["block"])
                diagnostics.append(cached["diagnostic"])
                continue
            started = time.perf_counter()
            batch = prepared[starts[int(original.positions[0])]:ends[int(original.positions[-1])]]
            if not np.array_equal(np.concatenate([b.positions for b in batch]), original.positions):
                raise ValueError("count-up refinement cannot split or reorder prepared blocks")
            selected = refiner._local_selection(original, batch)
            if len(batch) < 2 or len(selected) == 0:
                outputs.append(original)
                diagnostics.append(dict(component=number, changed=False, reason="single_local_block_or_empty"))
                continue
            # Component-sized buffers are released before the next component.
            workspaces = {}
            workspace = component_workspace(
                workspaces, batch, neutral, sites, config.proposal_max_bins, num_threads,
                minimum_bin_size=config.proposal_min_sites_per_bin)
            cost = float(chimera_scoring.compute_cc(batch, len(neutral), cc_scale))
            original_rows = selected.copy()
            initial_score = workspace.canonical(selected)
            options = dict(workspace=workspace, batch=batch, original=original,
                           neutral=neutral, sites=sites, config=config, threads=num_threads,
                           checkpoints=scope, workspaces=workspaces)
            selected, score, current = _repair(selected, "baseline", **options)
            proposals = []
            for step in range(config.count_max_additions):
                label = f"addition{step}"
                candidate = scope.load(label)
                if candidate is None:
                    # Free block boundaries and unrestricted local panels are
                    # optimistic for every admissible K+1 chromosome panel.
                    # Skip only when even that relaxation loses, with the same
                    # floating-point safety margin as reduced-count search.
                    bounds, constrained = count_bound.upper_bound(
                        workspace.leaves, workspace.offsets, workspace.logs,
                        workspace.penalty, len(selected) + 1)
                    optimistic = float(bounds.sum())
                    maximum_gain = 2 * (optimistic - score) - cost
                    margin = 1e-8 * max(1., abs(score), abs(optimistic),
                                        abs(len(selected) * cost - 2 * score))
                    if maximum_gain < -margin:
                        candidate = dict(diagnostic=dict(
                            step=step+1, rows=len(selected)+1, accepted=False,
                            reason="optimistic_local_bound_excludes_increased_count",
                            maximum_bic_gain=maximum_gain,
                            optimistic_score=optimistic, numerical_margin=margin,
                            count_constrained_blocks=int(constrained.sum())))
                        scope.save(label, candidate)
                        proposals.append(candidate["diagnostic"])
                        break
                    models = workspace.models()
                    unused = np.array([
                        next((j for j in range(len(block.haplotypes)) if j not in selected[:, i]),
                             int(selected[0, i])) for i, block in enumerate(batch)], np.int64)
                    candidates = []
                    for start in (selected[0].copy(), unused):
                        for reverse in (False, True):
                            row, _, _ = dual_search.solve(
                                models, selected, start, workspace.penalty,
                                branch_cap=config.branch_cap, reverse=reverse,
                                sweeps=config.dual_search_sweeps)
                            panel = np.vstack((selected, row))
                            candidates.append((workspace.canonical(panel), panel))
                    seed_score, panel = max(candidates, key=lambda item: item[0])
                    panel, found, block = _repair(panel, label + ".repair", **options)
                    gain = 2 * (found-score) - (len(panel)-len(selected)) * cost
                    candidate = dict(panel=panel, score=found, block=block,
                        diagnostic=dict(step=step+1, rows=len(panel), bic_gain=float(gain),
                                        accepted=bool(gain > 0), before_refit_score=float(seed_score),
                                        after_refit_score=float(found)))
                    scope.save(label, candidate)
                proposals.append(candidate["diagnostic"])
                if not candidate["diagnostic"]["accepted"]:
                    break
                selected, score, current = candidate["panel"], candidate["score"], candidate["block"]
            diagnostic = dict(
                component=number, changed=not np.array_equal(selected, original_rows),
                founders_before=len(original_rows), founders_after=len(selected),
                initial_likelihood=initial_score, final_likelihood=score,
                initial_bic=len(original_rows)*cost-2*initial_score,
                final_bic=len(selected)*cost-2*score, complexity_cost=cost,
                proposals=proposals, elapsed_seconds=time.perf_counter()-started)
            scope.save("result", dict(block=current, diagnostic=diagnostic))
            outputs.append(current)
            diagnostics.append(diagnostic)
    return haplotypes.BlockResults(outputs), dict(
        enabled=True, model="symmetric_count_up_refit_v1", components=diagnostics)
