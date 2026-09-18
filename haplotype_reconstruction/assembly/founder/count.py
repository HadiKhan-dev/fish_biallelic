"""All-deletion repair screening, followed by one deep fixed-count refit.

This pass asks whether one fewer assembled founder explains the cohort better
after the remaining paths have been refitted. It never assumes a true count,
uses pedigree metadata, changes local candidate alleles, or joins components.
Every deletion gets bounded conditional-row repairs; only the best repaired
panel enters the expensive path search. Singleton local blocks and one/two-row
panels remain the responsibility of discovery and ordinary hierarchy selection.

The objective is the existing K * complexity_cost - 2 * full_site_score. The
data mask and per-site likelihood model do not depend on the proposed count.
One deletion round is deliberately bounded; this is not exhaustive count search.
"""
import time
import numpy as np

from..import founder_refinement as refiner
from.import exchanges as founder_exchanges, scoring as founder_scoring
from..import chimera_scoring, paths, hierarchy
from.import count_bound as founder_count_bound
from.workspace import component_workspace, resolve_threads
from ...core import haplotypes, parallel
from ...discovery.objectives import compute_outer_bic_from_log_likelihood as bic


class _ScopedCheckpoints:
    """Keep each competing fit's ordinary checkpoint names separate."""
    def __init__(self, checkpoints, prefix):
        self.checkpoints = checkpoints
        self.prefix = prefix

    def load(self, phase):
        return refiner._load(self.checkpoints, self.prefix + "." + phase)

    def save(self, phase, value):
        refiner._save(self.checkpoints, self.prefix + "." + phase, value)


def _reconstruct(selected, score, batch, original):
    reconstruction = paths.reconstruct_haplotypes_from_beam(
        [(list(row), score) for row in selected], refiner._LeafKeyMap(batch), batch)
    result = hierarchy.convert_reconstruction_to_superblock(reconstruction, batch)
    for side in ("before", "after"):
        for stem in ("missing_aware_break", "missing_aware_break_reason",
                     "missing_aware_joint_informative_samples"):
            name = f"{stem}_{side}"
            default = False if stem == "missing_aware_break" else None
            setattr(result, name, getattr(original, name, default))
    return result


def _fixed_count_refit(prepared, block, neutral, sites, config, num_threads,
                       checkpoints, l1_blocks, workspaces=None):
    workspaces = {} if workspaces is None else workspaces
    options = dict(config=config, num_threads=num_threads,
                   checkpoints=checkpoints, l1_blocks=l1_blocks, workspaces=workspaces)
    first, _ = refiner._refine_components(
        prepared, haplotypes.BlockResults([block]), neutral, sites, **options)
    second, _ = refiner._refine_components(
        prepared, first, neutral, sites, dual=True, **options)
    return second


def refine_components(prepared, components, neutral, sites, *, config,
                      num_threads=1, checkpoints=None, l1_blocks=None, cc_scale=0.5,
                      workspaces=None):
    """Compare one-deletion panels using the hierarchy's complexity scale."""
    workspaces = {} if workspaces is None else workspaces
    results, diagnostics = [], []
    starts = {int(block.positions[0]): i for i, block in enumerate(prepared)}
    ends = {int(block.positions[-1]): i + 1 for i, block in enumerate(prepared)}
    with parallel.numba_thread_scope(resolve_threads(num_threads)):
        for number, original in enumerate(components):
            token = f"founder_count.component{number}"
            cached = refiner._load(checkpoints, token)
            if cached is not None:
                results.append(cached["block"])
                diagnostics.append(cached["diagnostic"])
                continue
            started = time.perf_counter()
            batch = prepared[starts[int(original.positions[0])]:ends[int(original.positions[-1])]]
            positions = np.concatenate([block.positions for block in batch])
            if not np.array_equal(positions, original.positions):
                raise ValueError("founder count search cannot split or reorder prepared blocks")
            selected = refiner._local_selection(original, batch)
            founders = len(selected)
            if len(batch) < 2 or founders <= 2:
                results.append(original)
                diagnostics.append(dict(component=number, changed=False,
                    reason="single_local_block_or_at_most_two_founders"))
                continue
            workspace = component_workspace(workspaces, batch, neutral, sites,
                config.proposal_max_bins, num_threads,
                minimum_bin_size=config.proposal_min_sites_per_bin)
            fitting, leaves, offsets = workspace.evidence, workspace.leaves, workspace.offsets
            complete, penalty, logs = workspace.complete, workspace.penalty, workspace.logs
            cost = float(chimera_scoring.compute_cc(batch, len(neutral), cc_scale))
            score = workspace.evaluate

            initial_score = score(selected)
            initial_bic = float(bic(founders, initial_score, cost))
            bounds, constrained = founder_count_bound.upper_bound(
                leaves, offsets, logs, penalty, founders - 1)
            optimistic = float(bounds.sum())
            optimistic_bic = float(bic(founders - 1, optimistic, cost))
            bound_tolerance = 1e-8 * max(1., abs(initial_score), abs(optimistic), abs(initial_bic))
            bound_record = dict(optimistic_score=optimistic, optimistic_bic=optimistic_bic,
                                count_constrained_blocks=int(constrained.sum()),
                                numerical_margin=bound_tolerance)
            if optimistic_bic > initial_bic + bound_tolerance:
                diagnostic = dict(component=number, changed=False,
                    reason="optimistic_local_bound_excludes_reduced_count",
                    founders_before=founders, founders_after=founders,
                    initial_likelihood=initial_score, final_likelihood=initial_score,
                    initial_bic=initial_bic, final_bic=initial_bic,
                    complexity_cost=cost, selected_deletion=None, proposals=[],
                    bound=bound_record, elapsed_seconds=time.perf_counter() - started)
                refiner._save(checkpoints, token, dict(block=original, diagnostic=diagnostic))
                results.append(original)
                diagnostics.append(diagnostic)
                continue
            order = np.arange(founders)
            best, best_bic, best_score, best_drop = original, initial_bic, initial_score, None
            from.import count_workers as founder_count_workers
            found_by_drop, pending = {}, []
            for dropped in order:
                dropped = int(dropped)
                scope = _ScopedCheckpoints(checkpoints, f"{token}.drop{dropped}")
                found = scope.load("result")
                if found is None:
                    pending.append((dropped, scope))
                else:
                    found_by_drop[dropped] = found
            if pending:
                fitted = founder_count_workers.run(pending, batch=batch,
                    selected=selected,
                    neutral=neutral, sites=sites, config=config,
                    threads=resolve_threads(num_threads),
                    cost=cost, workspaces=workspaces)
                found_by_drop.update((task[0], value) for task, value in zip(pending, fitted))
            proposals = []
            best_rows = None
            # All K lightweight repairs compete; deeply search one winner only.
            for dropped in order:
                dropped = int(dropped)
                found = found_by_drop[dropped]
                proposals.append(dict(dropped=dropped, bic=float(found["bic"]),
                    bic_gain=initial_bic - float(found["bic"]),
                    paired_changed=found["paired_changed"], runtime=found.get("runtime")))
                if found["bic"] < best_bic - 1e-8:
                    best_rows, best_bic, best_score, best_drop = (
                        found["selected"], float(found["bic"]), float(found["score"]), dropped)
            if best_rows is not None:
                best = _reconstruct(best_rows, best_score, batch, original)
            winner = min((int(i) for i in order), key=lambda i: found_by_drop[i]["bic"])
            deficit = float(found_by_drop[winner]["bic"]) - initial_bic
            # A deliberately generous search-budget heuristic, not a bound.
            # Every deletion has already had conditional repairs. Never screen
            # an improving repaired panel; None requests the unscreened search.
            multiple = config.count_refit_deficit_multiple
            if best_drop is None and multiple is not None and deficit > multiple * cost:
                diagnostic = dict(component=number, changed=False,
                    reason="cheap_repair_deficit_screen",
                    founders_before=founders, founders_after=founders,
                    initial_likelihood=initial_score, final_likelihood=initial_score,
                    initial_bic=initial_bic, final_bic=initial_bic,
                    complexity_cost=cost, selected_deletion=None,
                    proposals=proposals, bound=bound_record,
                    screen_multiple=multiple, deficit_multiple=deficit / cost,
                    elapsed_seconds=time.perf_counter() - started)
                refiner._save(checkpoints, token, dict(block=original, diagnostic=diagnostic))
                results.append(original)
                diagnostics.append(diagnostic)
                continue
            scope = _ScopedCheckpoints(checkpoints, f"{token}.deep_drop{winner}")
            deep = scope.load("result")
            if deep is None:
                found = found_by_drop[winner]
                seed = (best if winner == best_drop else
                        _reconstruct(found["selected"], found["score"], batch, original))
                result = _fixed_count_refit(batch, seed, neutral, sites, config,
                    num_threads, _ScopedCheckpoints(scope, "initial"), l1_blocks, workspaces)
                result, paired = founder_exchanges.refine_components(
                    batch, result, neutral, sites, _ScopedCheckpoints(scope, "paired"),
                    iterations=config.max_iterations, quota=config.branch_cap,
                    workspaces=workspaces)
                if any(item["changed"] for item in paired["components"]):
                    result = _fixed_count_refit(batch, result[0], neutral, sites, config,
                        num_threads, _ScopedCheckpoints(scope, "after_pair"), l1_blocks, workspaces)
                rows = refiner._local_selection(result[0], batch)
                likelihood = score(rows)
                deep = dict(block=result[0], score=likelihood,
                            bic=float(bic(len(rows), likelihood, cost)))
                scope.save("result", deep)
            if deep["bic"] < best_bic - 1e-8:
                best, best_bic, best_score, best_drop = (
                    deep["block"], deep["bic"], deep["score"], winner)
            diagnostic = dict(component=number, changed=best_drop is not None,
                founders_before=founders, founders_after=len(best.haplotypes),
                deep_refit_deletion=winner, deep_refit_bic=float(deep["bic"]),
                complexity_cost=cost, initial_likelihood=initial_score,
                final_likelihood=best_score, initial_bic=initial_bic, final_bic=best_bic,
                selected_deletion=best_drop, proposals=proposals, bound=bound_record,
                elapsed_seconds=time.perf_counter() - started)
            refiner._save(checkpoints, token, dict(block=best, diagnostic=diagnostic))
            results.append(best)
            diagnostics.append(diagnostic)
    return haplotypes.BlockResults(results), dict(
        model="all_deletion_conditional_repairs_single_deep_refit", components=diagnostics)
