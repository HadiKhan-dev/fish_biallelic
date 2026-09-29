"""Global founder escape competing under the canonical full-site objective.

The sparse coordinator allocates one global step per round. Original fine-row
and assembled-context proposals compete; only exact primary ties consult the
partial-founder score, and long macro edits retain the genotype-fit guard.
"""
from dataclasses import dataclass
import numpy as np
from numba import set_num_threads
from .. import founder_refinement as refiner
from . import scoring as founder_scoring, path_search as founder_path_search
from .packing import score_rows
from .workspace import resolve_threads
from .search_ranking import Ranking

_load, _save, _path_proposals = refiner._load, refiner._save, refiner._path_proposals

@dataclass
class PrimaryFirstState:
    calls: int = 0
    prior_rows: object = None
    prior_score: object = None
    focus: bool = False

    def snapshot(self):
        return dict(calls=self.calls, prior_rows=self.prior_rows,
                    prior_score=self.prior_score, focus=self.focus)

    @classmethod
    def restore(cls, saved):
        return cls(**saved)

    def proposals(self, requests, models, penalty, config, checkpoints, width,
                  dual, window, threads, proposal_cache, panel, workspace,
                  iteration, schedule):
        arguments = (requests, models, penalty, config, checkpoints, width,
                     dual, window, threads, proposal_cache)
        if not schedule.primary_first or not dual or window or workspace is None or not requests:
            return _path_proposals(*arguments), False
        first = self.calls == 0
        final = iteration >= config.max_iterations - 1
        self.calls += 1
        primary = workspace.canonical(panel)
        if self.prior_rows is not None and not np.array_equal(panel, self.prior_rows):
            self.focus = (primary == self.prior_score and workspace.predictive(panel)
                          > workspace.predictive(self.prior_rows) + 1e-6)
        self.prior_rows, self.prior_score = panel.copy(), primary
        if self.focus and not first and not final:
            choices = workspace.primary_preserving_choices(panel, config.branch_cap)
            restricted = [(f, r, known, incumbent, phase + ".primary_first")
                          for f, r, known, incumbent, phase in requests if f in choices]
            if restricted:
                found = _path_proposals(restricted, models, penalty, config,
                    checkpoints, config.beam_width, True, False, threads,
                    candidate_choices=choices)
                schedule.add("primary_first_batches")
                success, seen = False, set()
                baseline_predictive = None
                for focal, reverse, path, predicted in found:
                    if np.array_equal(path, panel[focal]):
                        continue
                    trial = panel.copy(); trial[focal] = path
                    key = (trial.shape, trial.tobytes())
                    if key in seen:
                        continue
                    seen.add(key)
                    score = workspace.canonical(trial)
                    if score != primary:
                        raise RuntimeError("primary-first proposal changed primary evidence")
                    if baseline_predictive is None:
                        baseline_predictive = workspace.predictive(panel)
                    if workspace.predictive(trial) > baseline_predictive + 1e-6:
                        success = True
                        break
                if success:
                    paths = {(f, bool(r)): path for f, r, path, _ in found}
                    answers, scores = [], {}
                    for focal, reverse, known, incumbent, phase in requests:
                        path = paths.get((focal, bool(reverse)), incumbent)
                        trial = panel.copy(); trial[focal] = path
                        key = (trial.shape, trial.tobytes())
                        if key not in scores:
                            scores[key] = score_rows(models, trial, penalty)
                        answers.append((focal, reverse, path.copy(), scores[key]))
                    schedule.add("primary_first_successes")
                    return answers, True
                schedule.add("primary_first_stall_refreshes")
        schedule.add("primary_first_full_batches")
        if final:
            schedule.add("primary_first_final_refreshes")
        return _path_proposals(*arguments), False

def refine_panel(selected, leaves, offsets, evidence, complete, submodels,
                  penalty, evaluate, config, checkpoints, token, num_threads,
                  prepared_evidence=None, macro_context=None, *, dual=False, window=False, workspace=None,
                  _schedule=None):
    search_name = "window" if window else ("dual" if dual else "beam")
    ranker = Ranking(_schedule, workspace)
    primary_state = PrimaryFirstState()

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
            primary_state = PrimaryFirstState.restore(cached["primary_first_state"])
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
        canonical_painting = None
        canonical_switches = None
        if _schedule.proxy_paint:
            painting = ranker.proxy().paint_compact(selected)[0]
        else:
            painting = evaluate(selected, paint=True)
            canonical_painting = painting
        if _schedule.proxy_paint:
            replacement, gains = ranker.proxy().fixed(selected, painting)
            occupancy = ranker.proxy().occupancy(painting, founders)
        else:
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
        for label, trial in ranker.ordered(
                proposals, 1, lambda: best, record, "fixed_painting"):
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
                candidates = []
                answers, restricted_first = primary_state.proposals(
                    requests, submodels, penalty, config, checkpoints, width,
                    dual, window, num_threads, proposal_cache, selected, workspace,
                    iteration, _schedule)
                for founder, reverse, proposed_path, predicted in answers:
                    trial = selected.copy()
                    trial[founder] = proposed_path
                    candidates.append((founder, reverse, trial, predicted))
                # Primary-equivalent alternatives need the complete predictive
                # comparison. Their tied coarse primary scores cannot rank them.
                checked_candidates = (candidates if restricted_first else ranker.ordered(
                    candidates, 2, lambda: best, record,
                    "conditional_" + search_name + f"_width{width}"))
                if restricted_first:
                    record["primary_first_success"] = True
                for founder, reverse, trial, predicted in checked_candidates:
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
            candidates = []
            for founder, reverse, proposed_path, predicted in _path_proposals(
                    requests, models, penalty, config, checkpoints, config.beam_width,
                    dual, window, num_threads, proposal_cache):
                trial = selected.copy()
                trial[founder] = founder_path_search.expand_macro_path(
                    proposed_path, alphabets, groups)
                candidates.append((founder, reverse, trial, predicted))
            for founder, reverse, trial, predicted in ranker.ordered(
                    candidates, 2, lambda: best, record, "l1_macro_" + search_name):
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
                    # Coarse-paint proposals never define the canonical
                    # incumbent switch count used in this full-score guard.
                    if canonical_switches is None:
                        if canonical_painting is not None:
                            canonical_switches = founder_path_search.count_diplotype_switches(
                                canonical_painting)
                        else:
                            incumbent = founder_scoring.selected_alleles(leaves, offsets, selected)
                            counted_score, counts = founder_scoring.score_and_switch_count(
                                incumbent, evidence, complete, penalty, prepared_evidence)
                            counted_score = float(counted_score.sum())
                            if abs(counted_score - likelihood) > max(1e-6, 1e-10 * abs(likelihood)):
                                raise RuntimeError("canonical counted incumbent score changed")
                            canonical_switches = int(counts.sum())
                            _schedule.add("macro_full_count_refreshes")
                    switch_delta = int(proposed_switches.sum()) - canonical_switches
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
            "primary_first_state": primary_state.snapshot(),
        })
        if best is None:
            break
    return selected, likelihood, history
