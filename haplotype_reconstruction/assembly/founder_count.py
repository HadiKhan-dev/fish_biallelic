"""Bounded deletion-and-refit search under the existing founder-count penalty.

This pass asks whether one fewer assembled founder explains the cohort better
after the remaining paths have been refitted. It never assumes a true count,
uses pedigree metadata, changes local candidate alleles, or joins components.
At most ``branch_cap`` complete refits are attempted per component; larger
panels prioritize low-occupancy rows. Singleton local blocks and one/two-row
panels remain the responsibility of discovery and ordinary hierarchy selection.

The objective is the existing K * complexity_cost - 2 * full_site_score. The
data mask and per-site likelihood model do not depend on the proposed count.
One deletion round is deliberately bounded; this is not exhaustive count search.
"""
import time
import numpy as np
from numba.typed import List

from . import founder_refinement as refiner, founder_exchanges, founder_scoring
from . import chimera_scoring, observations, paths, hierarchy, founder_count_bound
from ..core import haplotypes, parallel
from ..discovery.objectives import compute_outer_bic_from_log_likelihood as bic


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
                       checkpoints, l1_blocks):
    options = dict(config=config, num_threads=num_threads,
                   checkpoints=checkpoints, l1_blocks=l1_blocks)
    first, _ = refiner._refine_components(
        prepared, haplotypes.BlockResults([block]), neutral, sites, **options)
    second, _ = refiner._refine_components(
        prepared, first, neutral, sites, dual=True, **options)
    return second


def refine_components(prepared, components, neutral, sites, *, config,
                      num_threads=1, checkpoints=None, l1_blocks=None, cc_scale=0.5):
    """Compare one-deletion panels using the hierarchy's complexity scale."""
    results, diagnostics = [], []
    starts = {int(block.positions[0]): i for i, block in enumerate(prepared)}
    ends = {int(block.positions[-1]): i + 1 for i, block in enumerate(prepared)}
    with parallel.numba_thread_scope(num_threads):
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
            indices = np.searchsorted(sites, positions)
            if not np.array_equal(sites[indices], positions):
                raise ValueError("founder count evidence positions do not match")
            fitting = np.ascontiguousarray(neutral[:, indices], np.float32)
            leaves = List([np.ascontiguousarray(
                getattr(block, "missing_aware_inference_discrete_haps", block.discrete_haps), np.int8)
                for block in batch])
            offsets = np.asarray([0, *np.cumsum([len(block.positions) for block in batch])], np.int64)
            complete = np.concatenate([
                (np.ones(len(block.positions), np.bool_) if block.keep_flags is None
                 else np.asarray(block.keep_flags, np.bool_))
                & np.all(observations.founder_inference_panel_from_block_result(block).called, axis=0)
                for block in batch])
            penalty = chimera_scoring.compute_penalty(batch)
            cost = float(chimera_scoring.compute_cc(batch, len(neutral), cc_scale))
            logs = founder_scoring.prepare_log_evidence(fitting, complete)

            def score(rows):
                alleles = founder_scoring.selected_alleles(leaves, offsets, rows)
                return float(founder_scoring.score_panel(
                    alleles, fitting, complete, penalty, logs).sum())

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
                    bound=bound_record, elapsed_seconds=time.perf_counter()-started)
                refiner._save(checkpoints, token, dict(block=original, diagnostic=diagnostic))
                results.append(original)
                diagnostics.append(diagnostic)
                continue
            order = np.arange(founders)
            if founders > config.branch_cap:
                alleles = founder_scoring.selected_alleles(leaves, offsets, selected)
                painting = founder_scoring.paint_panel(alleles, fitting, complete, penalty, logs)[0]
                occupancy = founder_scoring.painting_occupancy(
                    painting, fitting, complete, founders, fitting.dtype.type(1e-10))
                order = np.argsort(occupancy, kind="stable")[:config.branch_cap]
                del alleles, painting, occupancy
            best, best_bic, best_score, best_drop = original, initial_bic, initial_score, None
            proposals = []
            for dropped in order:
                dropped = int(dropped)
                scope = _ScopedCheckpoints(checkpoints, f"{token}.drop{dropped}")
                found = scope.load("result")
                if found is None:
                    remaining = np.delete(selected, dropped, axis=0)
                    reduced = _reconstruct(remaining, initial_score, batch, original)
                    result = _fixed_count_refit(prepared, reduced, neutral, sites,
                        config, num_threads, _ScopedCheckpoints(scope, "initial"), l1_blocks)
                    result, paired = founder_exchanges.refine_components(
                        prepared, result, neutral, sites, _ScopedCheckpoints(scope, "paired"),
                        iterations=config.max_iterations, quota=config.branch_cap)
                    if any(item["changed"] for item in paired["components"]):
                        result = _fixed_count_refit(prepared, result[0], neutral, sites,
                            config, num_threads, _ScopedCheckpoints(scope, "after_pair"), l1_blocks)
                    rows = refiner._local_selection(result[0], batch)
                    likelihood = score(rows)
                    value = float(bic(len(rows), likelihood, cost))
                    found = dict(block=result[0], score=likelihood, bic=value,
                        paired_changed=any(item["changed"] for item in paired["components"]))
                    scope.save("result", found)
                proposals.append(dict(dropped=dropped, bic=float(found["bic"]),
                    bic_gain=initial_bic-float(found["bic"]),
                    paired_changed=found["paired_changed"]))
                if found["bic"] < best_bic - 1e-8:
                    best, best_bic, best_score, best_drop = (
                        found["block"], float(found["bic"]), float(found["score"]), dropped)
            diagnostic = dict(component=number, changed=best_drop is not None,
                founders_before=founders, founders_after=len(best.haplotypes),
                complexity_cost=cost, initial_likelihood=initial_score,
                final_likelihood=best_score, initial_bic=initial_bic, final_bic=best_bic,
                selected_deletion=best_drop, proposals=proposals, bound=bound_record,
                elapsed_seconds=time.perf_counter()-started)
            refiner._save(checkpoints, token, dict(block=best, diagnostic=diagnostic))
            results.append(best)
            diagnostics.append(diagnostic)
    return haplotypes.BlockResults(results), dict(
        model="bounded_delete_refit_pair_existing_complexity", components=diagnostics)
