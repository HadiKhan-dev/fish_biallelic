"""Refine final founder chromosomes against the original prepared local panels.

Hierarchy supplies the initial founder count and fixed phase-component boundaries.
Final refinement reopens local row choices and compares bounded count reductions
without inventing alleles, merging components or using pedigree information.
The sample-level fitting HMM is internal to assembly; it does not replace T09
painting or publish sample ancestry. Path edits improve the same full-site
cohort objective; count changes compete under the existing complexity penalty.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import time
import numpy as np
from numba import set_num_threads

from.import chimera_scoring
from.founder import dual_search as founder_dual_search, path_search as founder_path_search, scoring as founder_scoring
from.import hierarchy, panel_search, paths
from.founder.workspace import component_workspace, resolve_threads
from.founder.packing import score_rows
from..core import haplotypes, parallel


@dataclass(frozen=True)
class FounderRefinementConfig:
    enabled: bool = True
    beam_width: int = 64
    maximum_beam_width: int = 1024
    focal_quota: int = 1
    max_iterations: int = 20
    branch_cap: int = 16
    proposal_max_bins: int = 2000
    proposal_min_sites_per_bin: int = 1
    dual_search_sweeps: int = 20
    window_blocks: int = 100
    count_repair_sweeps: int = 3
    # Heuristic deep-refit screen in units of the existing per-founder cost.
    # None retains unscreened deep count search; this is not an upper bound.
    count_refit_deficit_multiple: float | None = 8.0
    interval_partners: int = 3

    def __post_init__(self):
        for name in ("beam_width", "maximum_beam_width", "focal_quota",
                     "max_iterations", "branch_cap", "proposal_max_bins", "proposal_min_sites_per_bin", "dual_search_sweeps",
                     "window_blocks", "count_repair_sweeps", "interval_partners"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        multiple = self.count_refit_deficit_multiple
        if multiple is not None and (not np.isfinite(multiple) or multiple <= 0):
            raise ValueError("count_refit_deficit_multiple must be positive or None")
        if self.maximum_beam_width < self.beam_width:
            raise ValueError("maximum_beam_width must be at least beam_width")


class _LeafKeyMap:
    def __init__(self, blocks):
        self.keys = [tuple(sorted(block.haplotypes)) for block in blocks]

    def get_key_from_dense(self, block, row):
        return self.keys[block][row]


def _load(checkpoints, phase):
    return None if checkpoints is None else checkpoints.load(phase)


def _save(checkpoints, phase, payload):
    if checkpoints is not None:
        checkpoints.save(phase, payload)


def _local_selection(component, blocks):
    """Decode composed source provenance back to rows of the prepared inputs."""
    atomic, _, atomic_lengths, _ = paths.missing_aware_atomic_source_provenance(component)
    selected = np.empty((len(component.haplotypes), len(blocks)), np.int64)
    offset = 0
    for column, block in enumerate(blocks):
        local, _, lengths, _ = paths.missing_aware_atomic_source_provenance(block)
        end = offset + len(lengths)
        if not np.array_equal(atomic_lengths[offset:end], lengths):
            raise ValueError("founder refinement source spans do not match prepared blocks")
        lookup = {}
        for row, source in enumerate(local):
            lookup.setdefault(tuple(source), row)
        for founder, source in enumerate(atomic[:, offset:end]):
            selected[founder, column] = lookup[tuple(source)]
        offset = end
    if offset != atomic.shape[1]:
        raise ValueError("founder refinement did not cover the component's source rows")
    original = np.concatenate([
        block.discrete_haps[selected[:, column]]
        for column, block in enumerate(blocks)
    ], axis=1)
    if not np.array_equal(original, component.discrete_haps):
        raise ValueError("founder refinement provenance does not reproduce input alleles")
    return selected


def _macro_context(batch, l1_blocks):
    """Combine alternative assembled rows inside each unchanged L1 span."""
    if l1_blocks is None:
        return None
    starts = {int(block.positions[0]): index for index, block in enumerate(batch)}
    ends = {int(block.positions[-1]): index + 1 for index, block in enumerate(batch)}
    by_span = {}
    for block in l1_blocks:
        if int(block.positions[0]) not in starts or int(block.positions[-1]) not in ends:
            continue
        start, end = starts[int(block.positions[0])], ends[int(block.positions[-1])]
        rows = by_span.setdefault((start, end), {})
        # Stable deduplication preserves the preferred refined-path ordering.
        # Only complete prepared-row paths are offered, never inferred alleles.
        for row in _local_selection(block, batch[start:end]):
            rows.setdefault(tuple(row), row)
    groups = sorted(by_span)
    if not groups or all(end - start == 1 for start, end in groups):
        return None
    if (groups[0][0] != 0 or groups[-1][1] != len(batch)
            or any(left[1] != right[0] for left, right in zip(groups, groups[1:]))):
        raise ValueError("L1 founder context does not partition the final component")
    context = [np.asarray(list(by_span[group].values()), np.int64) for group in groups]
    return groups, context


def _compute_path_proposal(models, known, incumbent, penalty, config,
                           width, reverse, dual, window, thread_budget=None, background=None,
                           candidate_choices=None):
    if window:
        from.founder import windows as founder_windows
        path, score, diagnostic = founder_windows.solve(
            models, known, incumbent, penalty, branch_cap=config.branch_cap,
            reverse=reverse, width=config.beam_width, window_blocks=config.window_blocks,
            ranking=("upper" if window == "upper" else
                     "incumbent" if window == "incumbent" else "tie"),
            thread_budget=thread_budget, background=background)
    elif dual:
        path, score, diagnostic = founder_dual_search.solve(
            models, known, incumbent, penalty, branch_cap=config.branch_cap,
            reverse=reverse, sweeps=config.dual_search_sweeps,
            thread_budget=thread_budget, background=background,
            candidate_choices=candidate_choices)
    else:
        path, score = founder_path_search.conditional_path(
            models, known, incumbent, penalty, width=width,
            branch_cap=config.branch_cap, reverse=reverse, thread_budget=thread_budget)
        diagnostic = None
    return {"path": path, "score": score, "search_diagnostic": diagnostic}


def _path_proposals(requests, models, penalty, config, checkpoints, width,
                    dual, window, num_threads, proposal_cache=None, candidate_choices=None):
    from.founder.candidates import completed_candidates
    answers, missing = {}, []
    for index, (_, _, known, incumbent, phase) in enumerate(requests):
        candidate = _load(checkpoints, phase)
        signature = (known.shape, known.tobytes(), incumbent.tobytes())
        if candidate is None and window == "lookahead" and ".macro." not in phase:
            ordinary_phase = phase.replace("founder_window_lookahead", "founder_window_escape")
            ordinary = _load(checkpoints, ordinary_phase)
            if ordinary is None and proposal_cache is not None:
                ordinary = proposal_cache.get(ordinary_phase)
            if (ordinary is not None and ordinary.get("input_rows") == signature
                    and ordinary.get("search_diagnostic", {}).get("equivalent_incumbent_tie", False)):
                candidate = dict(ordinary, reused_from=ordinary_phase)
                _save(checkpoints, phase, candidate)
        if candidate is None:
            missing.append(index)
        else:
            answers[index] = candidate

    shared = None
    if missing and (dual or window) and any(
            model['bin_emissions'].shape[3] > 2 for model in models):
        from.founder.background import SharedBackground
        focal, _, known, incumbent, _ = requests[0]
        panel = np.insert(known, focal, incumbent, axis=0)
        with parallel.numba_thread_scope(resolve_threads(num_threads)):
            shared = SharedBackground(models, panel)

    def compute(index, thread_budget):
        _, reverse, known, incumbent, _ = requests[index]
        return _compute_path_proposal(models, known, incumbent, penalty, config,
            width, reverse, dual, window, thread_budget,
            None if shared is None else shared.view(requests[index][0]),
            None if candidate_choices is None else candidate_choices[requests[index][0]])

    if missing:
        samples = models[0]["bin_emissions"].shape[0]
        founders = len(requests[0][2]) + 1
        states = founders * (founders + 1) // 2
        branches = min(config.branch_cap, max(m["bin_emissions"].shape[1] for m in models))
        workspace_bytes = (8 * samples * (
            3 * (len(models) + 1) * states + len(models) * branches
            + 2 * width * (states + branches)))
        # Shared-background scans allocate a t² segment table per active sample.
        largest_bins = max(m["bin_emissions"].shape[3] for m in models)
        workspace_bytes += 8 * samples * (largest_bins + 1) ** 2
        if window:
            workspace_bytes += sum(m["bin_emissions"].nbytes for m in models)
        functions = [partial(compute, index) for index in missing]
        for position, candidate in completed_candidates(functions, num_threads, workspace_bytes):
            index = missing[position]
            known, incumbent = requests[index][2:4]
            candidate["input_rows"] = (known.shape, known.tobytes(), incumbent.tobytes())
            answers[index] = candidate
            # Save each finished search immediately, on the controller thread.
            _save(checkpoints, requests[index][4], candidate)
            if (proposal_cache is not None and window == "incumbent"
                    and ".macro." not in requests[index][4]):
                proposal_cache[requests[index][4]] = candidate
    return [(request[0], request[1], answers[index]["path"], answers[index]["score"])
            for index, request in enumerate(requests)]


def _refine_panel(selected, leaves, offsets, evidence, complete, submodels,
                  penalty, evaluate, config, checkpoints, token, num_threads,
                  prepared_evidence=None, macro_context=None, *, dual=False, window=False, workspace=None):
    search_name = "window" if window else ("dual" if dual else "beam")

    def score_proposal(trial, threshold):
        value = evaluate(trial)
        # Local flank summation can round differently. Canonicalize every
        # possible winner (including near ties), not just the final winner.
        margin = max(1e-6, 1e-10 * max(abs(value), abs(threshold)))
        if workspace is not None and value + margin >= threshold:
            value = workspace.canonical(trial)
        return value

    def better(score, trial, reference_score, reference, primary_margin=1e-6):
        if score > reference_score + primary_margin:
            return True
        # Resolve genuine primary ties, not near ties or small primary losses.
        # Both panels are evaluated under the same fixed evidence masks.
        return (workspace is not None and score == reference_score
                and not np.array_equal(trial, reference)
                and workspace.predictive(trial) > workspace.predictive(reference) + 1e-6)

    selected = selected.copy()
    proposal_cache = None if workspace is None else workspace.path_proposals
    likelihood = evaluate(selected)
    history = []
    founders = len(selected)
    occupancy_tolerance = evidence.dtype.type(1e-10)
    working_width = config.beam_width
    for iteration in range(config.max_iterations):
        phase = f"{token}.iteration{iteration}"
        cached = _load(checkpoints, phase)
        if cached is not None:
            selected, likelihood = cached["selected"], cached["likelihood"]
            history, working_width = cached["history"], cached["working_width"]
            if cached["converged"]:
                break
            continue
        if callable(num_threads):
            # Count-refit workers grow at safe phase boundaries as peers finish.
            set_num_threads(resolve_threads(num_threads))
        initial_likelihood = likelihood
        if workspace is not None:
            workspace.set_reference(selected)
        painting = evaluate(selected, paint=True)
        replacement, gains = founder_scoring.fixed_path_proposals(
            leaves, offsets, selected, evidence, complete, painting, prepared_evidence)
        occupancy = founder_scoring.painting_occupancy(
            painting, evidence, complete, founders, occupancy_tolerance)
        quota = founders if dual else config.focal_quota
        order = np.lexsort((np.arange(founders), occupancy, -gains))[:quota]
        record = {
            "iteration": iteration, "search": search_name,
            "conditional_gains": gains.tolist(),
            "focal_paths": order.tolist(), "width_before": working_width,
            "proposals": [],
        }
        best = None
        proposals = [("all", replacement)]
        for founder in order:
            trial = selected.copy()
            trial[founder] = replacement[founder]
            proposals.append((int(founder), trial))
        for label, trial in proposals:
            if np.array_equal(trial, selected):
                continue
            score = score_proposal(trial, likelihood if best is None else best[0])
            record["proposals"].append({
                "kind": "fixed_painting", "focal": label, "gain": score - likelihood})
            if better(score, trial, likelihood, selected) and (best is None or
                    better(score, trial, best[0], best[1], primary_margin=0.0)):
                best = score, trial.copy()
        # The first beam must see the original assembly, not a warm start
        # that has already discarded a potentially better search basin.
        if iteration == 0 or best is None:
            width = working_width
            while True:
                cheaper_score = likelihood if best is None else best[0]
                requests = []
                for founder in order:
                    known = np.ascontiguousarray(np.delete(selected, founder, axis=0))
                    for reverse in (False, True):
                        beam_phase = f"{phase}.path{founder}.width{width}.reverse{int(reverse)}"
                        requests.append((founder, reverse, known, selected[founder], beam_phase))
                for founder, reverse, proposed_path, predicted in _path_proposals(
                        requests, submodels, penalty, config, checkpoints, width,
                        dual, window, num_threads, proposal_cache):
                    trial = selected.copy()
                    trial[founder] = proposed_path
                    checked = score_rows(submodels, trial, penalty)
                    if abs(checked - predicted) > 1e-5:
                        raise RuntimeError("conditional founder-path score does not match its full panel")
                    score = score_proposal(trial, likelihood if best is None else best[0])
                    record["proposals"].append({
                        "kind": "conditional_" + search_name,
                        "focal": int(founder),
                        "reverse": reverse, "width": width, "gain": score - likelihood,
                    })
                    if better(score, trial, likelihood, selected) and (best is None or
                            better(score, trial, best[0], best[1], primary_margin=0.0)):
                        best = score, trial
                improvement = (likelihood if best is None else best[0]) - cheaper_score
                # A computational budget rule, not a confidence/calling rule.
                # Dual search has a sweep budget, not a beam width. Repeating
                # it at a wider nominal beam would return the same candidate.
                if dual or improvement <= penalty or width >= config.maximum_beam_width:
                    break
                width = min(width * 4, config.maximum_beam_width)
                working_width = max(working_width, width)
        # Whole L1 pieces can cross a local-row search barrier even when
        # a much wider 200-SNP beam stalls. They compete under the same
        # full-site objective, and only reopen original prepared rows.
        if best is None and macro_context is not None:
            groups, context_paths = macro_context
            models, alphabets, macro_selected = founder_path_search.coarsen_submodels(
                submodels, selected, groups, context_paths)
            requests = []
            for founder in order:
                known = np.ascontiguousarray(np.delete(macro_selected, founder, axis=0))
                for reverse in (False, True):
                    macro_phase = (f"{phase}.macro.path{founder}."
                                   f"width{config.beam_width}.reverse{int(reverse)}")
                    requests.append((founder, reverse, known, macro_selected[founder], macro_phase))
            for founder, reverse, proposed_path, predicted in _path_proposals(
                    requests, models, penalty, config, checkpoints, config.beam_width,
                    dual, window, num_threads, proposal_cache):
                trial = selected.copy()
                trial[founder] = founder_path_search.expand_macro_path(
                    proposed_path, alphabets, groups)
                checked = score_rows(submodels, trial, penalty)
                if abs(checked - predicted) > 1e-5:
                    raise RuntimeError("macro founder-path score does not match its full panel")
                score = score_proposal(trial, likelihood if best is None else best[0])
                if workspace is not None and score > likelihood + 1e-6:
                    score = workspace.canonical(trial)
                proposal = {
                    "kind": "l1_macro_" + search_name,
                    "focal": int(founder),
                    "reverse": reverse, "width": config.beam_width,
                    "groups": len(groups), "gain": score - likelihood,
                }
                # A long founder edit must not buy fewer sample switches
                # by worsening genotype fit. Otherwise a shared descendant
                # crossover can be absorbed into an artificial founder.
                # score = centered genotype fit - penalty * switch count.
                if better(score, trial, likelihood, selected):
                    alleles = founder_scoring.selected_alleles(leaves, offsets, trial)
                    _, proposed_switches = founder_scoring.score_and_switch_count(
                        alleles, evidence, complete, penalty, prepared_evidence)
                    switch_delta = (int(proposed_switches.sum())
                        - founder_path_search.count_diplotype_switches(painting))
                    emission_gain = score - likelihood + penalty * switch_delta
                    proposal.update(
                        sample_switch_delta=int(switch_delta),
                        genotype_fit_gain=float(emission_gain),
                        genotype_fit_guard_passed=emission_gain >= -1e-6)
                    if emission_gain >= -1e-6 and (best is None or
                            better(score, trial, best[0], best[1], primary_margin=0.0)):
                        best = score, trial
                record["proposals"].append(proposal)
            del models, alphabets, macro_selected
        if best is None and dual and not window and workspace is not None:
            # The unrestricted partial-evidence optimum can sacrifice primary
            # evidence elsewhere and be rejected. Reopen only complete-site
            # equivalent local rows to avoid that search barrier.
            choices = workspace.primary_preserving_choices(selected, config.branch_cap)
            requests = []
            for founder in choices:
                known = np.ascontiguousarray(np.delete(selected, founder, axis=0))
                for reverse in (False, True):
                    restricted_phase = (f"{phase}.primary_preserving.path{founder}."
                                        f"reverse{int(reverse)}")
                    requests.append((founder, reverse, known, selected[founder],
                                     restricted_phase))
            for founder, reverse, proposed_path, predicted in _path_proposals(
                    requests, submodels, penalty, config, checkpoints, config.beam_width,
                    True, False, num_threads, candidate_choices=choices):
                trial = selected.copy()
                trial[founder] = proposed_path
                checked = score_rows(submodels, trial, penalty)
                if abs(checked - predicted) > 1e-5:
                    raise RuntimeError("primary-preserving proposal has an inconsistent score")
                score = workspace.canonical(trial)
                if score != workspace.canonical(selected):
                    raise RuntimeError("primary-preserving proposal changed primary evidence")
                record["proposals"].append({
                    "kind": "primary_preserving_dual", "focal": int(founder),
                    "reverse": reverse, "gain": score - likelihood})
                if better(score, trial, likelihood, selected) and (best is None or
                        better(score, trial, best[0], best[1], primary_margin=0.0)):
                    best = score, trial
        if best is not None and workspace is not None:
            checked = workspace.canonical(best[1])
            if abs(checked - best[0]) > max(1e-6, 1e-10 * abs(checked)):
                raise RuntimeError("localized founder score disagrees with complete panel")
            best = ((checked, best[1]) if better(
                checked, best[1], likelihood, selected) else None)
        if best is not None:
            if best[0] == likelihood and workspace is not None:
                record["accepted_predictive_tie_gain"] = (
                    workspace.predictive(best[1]) - workspace.predictive(selected))
            likelihood, selected = best
        record["accepted_gain"] = likelihood - initial_likelihood
        record["width_after"] = working_width
        history.append(record)
        _save(checkpoints, phase, {
            "selected": selected, "likelihood": likelihood, "history": history,
            "working_width": working_width, "converged": best is None,
        })
        if best is None:
            break
    return selected, likelihood, history


def _select_window_trajectories(ordinary, ordinary_diagnostics,
                                optimistic, optimistic_diagnostics):
    """Select by primary score, then partial-founder evidence at exact ties."""
    results, diagnostics = [], []
    for left, a, right, b in zip(
            ordinary, ordinary_diagnostics["components"],
            optimistic, optimistic_diagnostics["components"]):
        # Singleton components bypass fitting in both searches.
        choose_optimistic = (
            "final_likelihood" in b and
            (b["final_likelihood"] > a["final_likelihood"] + 1e-6 or (
                b["final_likelihood"] == a["final_likelihood"]
                and b["final_predictive_score"] > a["final_predictive_score"] + 1e-6)))
        selected = b if choose_optimistic else a
        results.append(right if choose_optimistic else left)
        diagnostics.append({
            "component": selected["component"], "changed": selected["changed"],
            "selected_search": "lookahead" if choose_optimistic else "incumbent",
            "initial_likelihood": selected.get("initial_likelihood"),
            "final_likelihood": selected.get("final_likelihood"),
            "incumbent_search": a, "lookahead_search": b,
        })
    return haplotypes.BlockResults(results), {
        "enabled": True, "search": "two_completed_window_trajectories",
        "components": diagnostics,
    }


def refine_components(prepared_blocks, components, neutral_probs, global_sites, *,
                      config=FounderRefinementConfig(), num_threads=1,
                      checkpoints=None, l1_blocks=None, cc_scale=0.5):
    """Run every refinement pass independently inside each phase component."""
    if config.enabled and len(components) > 1:
        from.founder.components import refine_independent_components
        return refine_independent_components(
            prepared_blocks, components, neutral_probs, global_sites,
            config=config, num_threads=num_threads, checkpoints=checkpoints,
            l1_blocks=l1_blocks, cc_scale=cc_scale)
    return _refine_serial_components(
        prepared_blocks, components, neutral_probs, global_sites,
        config=config, num_threads=num_threads, checkpoints=checkpoints,
        l1_blocks=l1_blocks, cc_scale=cc_scale)


def _refine_serial_components(prepared_blocks, components, neutral_probs, global_sites, *,
                      config=FounderRefinementConfig(), num_threads=1,
                      checkpoints=None, l1_blocks=None, cc_scale=0.5):
    """Refine paths and bounded count proposals before exact-flank polishing.

    Path passes retain the full-site objective and macro genotype-fit guard.
    Count proposals reuse the existing complexity cost and data mask; a local
    optimistic bound skips provably losing reductions. Upstream passes start
    from their completed predecessor. Two final window trajectories share
    that same starting panel and compete by their completed full-site score.
    An upper-bound-ranked polish reopens paths from the selected fit, followed
    by bounded paired intervals at local and staggered L1 boundary grids.
    Components and each pass's proposals/iterations have separate checkpoints.
    Pre-count paired suffix exchanges preserve the local called/missing allele
    multiset and require full-objective improvement. The extra genotype-fit
    guard remains for count-reduction refits and bounded interval polishing.
    No truth or pedigree enters these passes.
    """
    if checkpoints is not None and config.enabled:
        from.founder.checkpoints import FounderCheckpointStore
        checkpoints = FounderCheckpointStore(checkpoints, prepared_blocks)
    options = dict(config=config, num_threads=num_threads,
                   checkpoints=checkpoints, l1_blocks=l1_blocks, workspaces={})
    refined, first = _refine_components(
        prepared_blocks, components, neutral_probs, global_sites, **options)
    if not config.enabled:
        return refined, first
    output, second = _refine_components(
        prepared_blocks, refined, neutral_probs, global_sites, dual=True, **options)
    from.founder import exchanges as founder_exchanges, count as founder_count
    with parallel.numba_thread_scope(resolve_threads(num_threads)):
        output, exchanges = founder_exchanges.refine_components(
            prepared_blocks, output, neutral_probs, global_sites,
            founder_count._ScopedCheckpoints(checkpoints, "before_count"),
            iterations=config.max_iterations, quota=config.branch_cap,
            preserve_genotype_fit=False, workspaces=options["workspaces"])
    output, counts = founder_count.refine_components(
        prepared_blocks, output, neutral_probs, global_sites, cc_scale=cc_scale, **options)
    # Two bounded search trajectories share the same starting panel. An early
    # lookahead gain can enter a worse basin, so compare completed full-site
    # fits rather than dropping the stable-order trajectory greedily.
    ordinary, ordinary_windows = _refine_components(
        prepared_blocks, output, neutral_probs, global_sites, dual=True,
        window="incumbent", **options)
    optimistic, optimistic_windows = _refine_components(
        prepared_blocks, output, neutral_probs, global_sites, dual=True,
        window="lookahead", **options)
    output, windows = _select_window_trajectories(
        ordinary, ordinary_windows, optimistic, optimistic_windows)
    # Feasible-prefix ranking can discard a beneficial tract before its far
    # flank is reached. An upper-bound-ranked pass explores such alternatives
    # from the selected completed fit; acceptance remains the exact objective.
    output, upper_windows = _refine_components(
        prepared_blocks, output, neutral_probs, global_sites, dual=True,
        window="upper", **options)
    # Bounded paired intervals can repair a coordinated two-founder barrier.
    # Retain exact flank scores and the genotype-fit guard; do not reopen the
    # unbounded chromosome-wide suffix permutations after local polishing.
    from.founder import intervals as founder_intervals
    with parallel.numba_thread_scope(resolve_threads(num_threads)):
        output, interval_windows = founder_intervals.refine_components(
            prepared_blocks, output, neutral_probs, global_sites,
            config=config, checkpoints=checkpoints, l1_blocks=l1_blocks,
            workspaces=options["workspaces"])
    diagnostics = [
        {"component": before["component"],
         "changed": any(item["changed"] for item in (before, after, exchange, count, local, upper, interval)),
         "beam_refinement": before, "dual_escape": after,
         "paired_exchange": exchange, "count_refinement": count,
         "window_escape": local, "optimistic_window_polish": upper,
         "paired_interval_polish": interval}
        for before, after, exchange, count, local, upper, interval in zip(
            first["components"], second["components"], exchanges["components"],
            counts["components"], windows["components"], upper_windows["components"],
            interval_windows["components"])
    ]
    return output, {
        "enabled": True, "model": "full_site_potts_predictive_ties_progressive_v12",
        "candidate_rows": "original_prepared_inference_panels",
        "components": diagnostics,
    }


def _refine_components(prepared_blocks, components, neutral_probs, global_sites, *,
                       config=FounderRefinementConfig(), num_threads=1,
                       checkpoints=None, l1_blocks=None, dual=False, window=False,
                       workspaces=None):
    """Refine final component paths while preserving their geometry and count.

    ``checkpoints`` is the already-bound assembly checkpoint callback. Every
    beam, iteration and completed component can be resumed independently.
    This function must run only for the final release, not the L1/L2 feedback
    rounds. It never modifies the original prepared panels. The final
    hierarchy supplies its L1 blocks as larger search moves after a fine-scale
    stall. These moves must also preserve or improve the cohort's genotype fit
    under the optimized internal sample paths, not just save switch penalties.
    """
    if not config.enabled:
        return components, {"enabled": False, "components": []}
    workspaces = {} if workspaces is None else workspaces
    prepared = list(prepared_blocks)
    starts = {int(block.positions[0]): index for index, block in enumerate(prepared)}
    ends = {int(block.positions[-1]): index + 1 for index, block in enumerate(prepared)}
    results, diagnostics = [], []
    with parallel.numba_thread_scope(resolve_threads(num_threads)):
        for number, component in enumerate(components):
            phase = ("founder_window_lookahead" if window == "lookahead" else
                     "founder_window_upper" if window == "upper" else
                     ("founder_window_escape" if window else
                      ("founder_dual_escape" if dual else "founder_refinement")))
            token = f"{phase}.component{number}"
            cached = _load(checkpoints, token)
            if cached is not None:
                results.append(cached["block"])
                diagnostics.append(cached["diagnostic"])
                continue
            started = time.perf_counter()
            batch = prepared[starts[int(component.positions[0])]:ends[int(component.positions[-1])]]
            positions = np.concatenate([block.positions for block in batch])
            if not np.array_equal(positions, component.positions):
                raise ValueError("founder refinement cannot split or reorder a prepared block")
            selected = _local_selection(component, batch)
            if len(batch) < 2 or len(selected) < 2:
                results.append(component)
                diagnostics.append({"component": number, "changed": False,
                                    "reason": "single_block_or_founder", "iterations": []})
                continue
            original = selected.copy()
            workspace = component_workspace(workspaces, batch, neutral_probs,
                global_sites, config.proposal_max_bins, num_threads,
                minimum_bin_size=config.proposal_min_sites_per_bin)
            evidence, leaves, offsets = workspace.evidence, workspace.leaves, workspace.offsets
            complete, penalty = workspace.complete, workspace.penalty
            bin_size, submodels = workspace.bin_size, workspace.models()
            prepared_evidence = workspace.logs
            evaluate = workspace.evaluate

            initial_likelihood = evaluate(selected)
            selected, likelihood, history = _refine_panel(
                selected, leaves, offsets, evidence, complete, submodels, penalty,
                evaluate, config, checkpoints, token, num_threads, prepared_evidence,
                _macro_context(batch, l1_blocks), dual=dual, window=window, workspace=workspace)
            changed = not np.array_equal(selected, original)
            result = component
            if changed:
                reconstructed = paths.reconstruct_haplotypes_from_beam(
                    [(list(row), likelihood) for row in selected], _LeafKeyMap(batch), batch)
                result = hierarchy.convert_reconstruction_to_superblock(reconstructed, batch)
                # Cached prepared leaves predate boundaries introduced by the
                # hierarchy. The existing component is authoritative for its
                # unchanged outer phase boundaries, including on resume.
                for side in ("before", "after"):
                    for stem in ("missing_aware_break", "missing_aware_break_reason",
                                 "missing_aware_joint_informative_samples"):
                        name = f"{stem}_{side}"
                        default = False if stem == "missing_aware_break" else None
                        setattr(result, name, getattr(component, name, default))
            diagnostic = {
                "component": number, "changed": changed,
                "founders": len(selected), "original_blocks": len(batch),
                "changed_local_rows": int(np.count_nonzero(selected != original)),
                "called_before": int(np.count_nonzero(component.discrete_haps >= 0)),
                "called_after": int(np.count_nonzero(result.discrete_haps >= 0)),
                "initial_likelihood": initial_likelihood, "final_likelihood": likelihood,
                "final_predictive_score": workspace.predictive(selected),
                "proposal_bin_size": bin_size, "switch_penalty": penalty,
                "local_score_calls": workspace.local_scores,
                "iterations": history, "elapsed_seconds": time.perf_counter() - started,
            }
            _save(checkpoints, token, {"block": result, "diagnostic": diagnostic})
            results.append(result)
            diagnostics.append(diagnostic)
    return haplotypes.BlockResults(results), {
        "enabled": True, "model": "fixed_count_full_site_potts_v1",
        "search": "window" if window else ("dual" if dual else "beam"),
        "candidate_rows": "original_prepared_inference_panels",
        "components": diagnostics,
    }
