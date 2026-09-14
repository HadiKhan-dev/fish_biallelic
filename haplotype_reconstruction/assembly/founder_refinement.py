"""Refine final founder chromosomes against the original prepared local panels.

Hierarchy remains responsible for founder count and phase-component boundaries.
This final pass reopens local row choices discarded by earlier hierarchy levels
without inventing alleles, merging components or using pedigree information.
The sample-level fitting HMM is internal to assembly; it does not replace T09
painting or publish sample ancestry. Every accepted edit improves the same
full-site, fixed-count cohort objective.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
import numpy as np
from numba.typed import List

from . import chimera_scoring, founder_path_search, founder_scoring
from . import hierarchy, observations, panel_search, paths
from ..core import haplotypes, parallel


@dataclass(frozen=True)
class FounderRefinementConfig:
    enabled: bool = True
    beam_width: int = 64
    maximum_beam_width: int = 1024
    focal_quota: int = 1
    max_iterations: int = 20
    branch_cap: int = 16
    proposal_max_bins: int = 2000

    def __post_init__(self):
        for name in ("beam_width", "maximum_beam_width", "focal_quota",
                     "max_iterations", "branch_cap", "proposal_max_bins"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
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
    """Restrict L1 contexts to complete pieces inside this phase component."""
    if l1_blocks is None:
        return None
    starts = {int(block.positions[0]): index for index, block in enumerate(batch)}
    ends = {int(block.positions[-1]): index + 1 for index, block in enumerate(batch)}
    groups, context = [], []
    for block in l1_blocks:
        if int(block.positions[0]) not in starts or int(block.positions[-1]) not in ends:
            continue
        start, end = starts[int(block.positions[0])], ends[int(block.positions[-1])]
        groups.append((start, end))
        context.append(_local_selection(block, batch[start:end]))
    if not groups or all(end - start == 1 for start, end in groups):
        return None
    if (groups[0][0] != 0 or groups[-1][1] != len(batch)
            or any(left[1] != right[0] for left, right in zip(groups, groups[1:]))):
        raise ValueError("L1 founder context does not partition the final component")
    return groups, context


def _refine_panel(selected, leaves, offsets, evidence, complete, submodels,
                  penalty, evaluate, config, checkpoints, token, num_threads,
                  prepared_evidence=None, macro_context=None):
    selected = selected.copy()
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
        initial_likelihood = likelihood
        painting = evaluate(selected, paint=True)
        replacement, gains = founder_scoring.fixed_path_proposals(
            leaves, offsets, selected, evidence, complete, painting, prepared_evidence)
        occupancy = founder_scoring.painting_occupancy(
            painting, evidence, complete, founders, occupancy_tolerance)
        order = np.lexsort((np.arange(founders), occupancy, -gains))[:config.focal_quota]
        record = {
            "iteration": iteration, "conditional_gains": gains.tolist(),
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
            score = evaluate(trial)
            record["proposals"].append({
                "kind": "fixed_painting", "focal": label, "gain": score - likelihood})
            if score > likelihood + 1e-6 and (best is None or score > best[0]):
                best = score, trial.copy()
        # The first beam must see the original assembly, not a warm start
        # that has already discarded a potentially better search basin.
        if iteration == 0 or best is None:
            width = working_width
            while True:
                cheaper_score = likelihood if best is None else best[0]
                for founder in order:
                    known = np.ascontiguousarray(np.delete(selected, founder, axis=0))
                    for reverse in (False, True):
                        beam_phase = f"{phase}.path{founder}.width{width}.reverse{int(reverse)}"
                        candidate = _load(checkpoints, beam_phase)
                        if candidate is None:
                            candidate = founder_path_search.conditional_path(
                                submodels, known, selected[founder], penalty,
                                width=width, branch_cap=config.branch_cap, reverse=reverse)
                            _save(checkpoints, beam_phase, candidate)
                        proposed_path, predicted = candidate
                        trial = selected.copy()
                        trial[founder] = proposed_path
                        keys = [[entry["hap_keys"][row] for entry, row in zip(submodels, path)]
                                for path in trial]
                        checked = panel_search.evaluate_panel(
                            keys, submodels, penalty, len(painting), num_threads=num_threads)
                        if abs(checked - predicted) > 1e-5:
                            raise RuntimeError("conditional founder-path score does not match its full panel")
                        score = evaluate(trial)
                        record["proposals"].append({
                            "kind": "conditional_beam", "focal": int(founder),
                            "reverse": reverse, "width": width, "gain": score - likelihood,
                        })
                        if score > likelihood + 1e-6 and (best is None or score > best[0]):
                            best = score, trial
                improvement = (likelihood if best is None else best[0]) - cheaper_score
                # A computational budget rule, not a confidence/calling rule.
                if improvement <= penalty or width >= config.maximum_beam_width:
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
            for founder in order:
                known = np.ascontiguousarray(np.delete(macro_selected, founder, axis=0))
                for reverse in (False, True):
                    macro_phase = (f"{phase}.macro.path{founder}."
                                   f"width{config.beam_width}.reverse{int(reverse)}")
                    candidate = _load(checkpoints, macro_phase)
                    if candidate is None:
                        candidate = founder_path_search.conditional_path(
                            models, known, macro_selected[founder], penalty,
                            width=config.beam_width, branch_cap=config.branch_cap,
                            reverse=reverse)
                        _save(checkpoints, macro_phase, candidate)
                    proposed_path, predicted = candidate
                    trial = selected.copy()
                    trial[founder] = founder_path_search.expand_macro_path(
                        proposed_path, alphabets, groups)
                    keys = [[entry["hap_keys"][row]
                             for entry, row in zip(submodels, path)]
                            for path in trial]
                    checked = panel_search.evaluate_panel(
                        keys, submodels, penalty, len(painting), num_threads=num_threads)
                    if abs(checked - predicted) > 1e-5:
                        raise RuntimeError("macro founder-path score does not match its full panel")
                    score = evaluate(trial)
                    proposal = {
                        "kind": "l1_macro_beam", "focal": int(founder),
                        "reverse": reverse, "width": config.beam_width,
                        "groups": len(groups), "gain": score - likelihood,
                    }
                    # A long founder edit must not buy fewer sample switches
                    # by worsening genotype fit. Otherwise a shared descendant
                    # crossover can be absorbed into an artificial founder.
                    # score = centered genotype fit - penalty * switch count.
                    if score > likelihood + 1e-6:
                        proposed_painting = evaluate(trial, paint=True)
                        switch_delta = (
                            founder_path_search.count_diplotype_switches(proposed_painting)
                            - founder_path_search.count_diplotype_switches(painting))
                        del proposed_painting
                        emission_gain = score - likelihood + penalty * switch_delta
                        proposal.update(
                            sample_switch_delta=int(switch_delta),
                            genotype_fit_gain=float(emission_gain),
                            genotype_fit_guard_passed=emission_gain >= -1e-6)
                        if emission_gain >= -1e-6 and (best is None or score > best[0]):
                            best = score, trial
                    record["proposals"].append(proposal)
            del models, alphabets, macro_selected
        if best is not None:
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


def refine_components(prepared_blocks, components, neutral_probs, global_sites, *,
                      config=FounderRefinementConfig(), num_threads=1,
                      checkpoints=None, l1_blocks=None):
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
    prepared = list(prepared_blocks)
    starts = {int(block.positions[0]): index for index, block in enumerate(prepared)}
    ends = {int(block.positions[-1]): index + 1 for index, block in enumerate(prepared)}
    results, diagnostics = [], []
    with parallel.numba_thread_scope(num_threads):
        for number, component in enumerate(components):
            token = f"founder_refinement.component{number}"
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
            indices = np.searchsorted(global_sites, positions)
            if not np.array_equal(global_sites[indices], positions):
                raise ValueError("founder refinement evidence positions do not match")
            evidence = np.ascontiguousarray(neutral_probs[:, indices], np.float32)
            leaves = List([np.ascontiguousarray(
                getattr(block, "missing_aware_inference_discrete_haps", block.discrete_haps), np.int8)
                for block in batch])
            complete = np.concatenate([
                (np.ones(len(block.positions), np.bool_) if block.keep_flags is None
                 else np.asarray(block.keep_flags, np.bool_))
                & np.all(observations.founder_inference_panel_from_block_result(block).called, axis=0)
                for block in batch
            ])
            offsets = np.asarray([0, *np.cumsum([len(block.positions) for block in batch])], np.int64)
            penalty = chimera_scoring.compute_penalty(batch)
            bin_size = max(chimera_scoring.compute_spb(batch),
                           math.ceil(len(positions) / config.proposal_max_bins))
            submodels = chimera_scoring.compute_subblock_emissions(
                batch, evidence, positions, bin_size, num_threads=num_threads)

            prepared_evidence = founder_scoring.prepare_log_evidence(evidence, complete)

            def evaluate(panel, paint=False):
                alleles = founder_scoring.selected_alleles(leaves, offsets, panel)
                if paint:
                    return founder_scoring.paint_panel(
                        alleles, evidence, complete, penalty, prepared_evidence)[0]
                return float(founder_scoring.score_panel(
                    alleles, evidence, complete, penalty, prepared_evidence).sum())

            initial_likelihood = evaluate(selected)
            selected, likelihood, history = _refine_panel(
                selected, leaves, offsets, evidence, complete, submodels, penalty,
                evaluate, config, checkpoints, token, num_threads, prepared_evidence,
                _macro_context(batch, l1_blocks))
            changed = not np.array_equal(selected, original)
            result = component
            if changed:
                reconstructed = paths.reconstruct_haplotypes_from_beam(
                    [(list(row), likelihood) for row in selected], _LeafKeyMap(batch), batch)
                result = hierarchy.convert_reconstruction_to_superblock(reconstructed, batch)
            diagnostic = {
                "component": number, "changed": changed,
                "founders": len(selected), "original_blocks": len(batch),
                "changed_local_rows": int(np.count_nonzero(selected != original)),
                "called_before": int(np.count_nonzero(component.discrete_haps >= 0)),
                "called_after": int(np.count_nonzero(result.discrete_haps >= 0)),
                "initial_likelihood": initial_likelihood, "final_likelihood": likelihood,
                "proposal_bin_size": bin_size, "switch_penalty": penalty,
                "iterations": history, "elapsed_seconds": time.perf_counter() - started,
            }
            _save(checkpoints, token, {"block": result, "diagnostic": diagnostic})
            results.append(result)
            diagnostics.append(diagnostic)
    return haplotypes.BlockResults(results), {
        "enabled": True, "model": "fixed_count_full_site_potts_v1",
        "candidate_rows": "original_prepared_inference_panels",
        "components": diagnostics,
    }
