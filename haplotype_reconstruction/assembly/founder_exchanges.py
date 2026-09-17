"""Exact Potts scores for paired founder suffix exchanges.

A suffix permutation preserves every local allele and genotype alphabet.
Inside the suffix it is only a relabeling of diploid states; the only changed
transition connects the prefix to the suffix. Thus forward/backward messages
score every exchange without refitting the chromosome for every proposal.

For K founders, N samples, L sites, B boundaries: message preparation O(NLK²),
all exchange queries O(NB(K³ + K² log K)), memory O(NBK²). No truth is used.
All accepted changes are rescored by the canonical full-site objective.
"""
import numpy as np
from numba import njit, prange
from haplotype_reconstruction.assembly import founder_scoring


@njit(cache=True, parallel=True)
def messages(haps, logs, cuts, penalty):
    first, second = founder_scoring._unordered_pairs(len(haps))
    samples, sites, _ = logs.shape
    forward = np.empty((samples, len(cuts), len(first)), np.float64)
    backward = np.empty_like(forward)
    center = np.log(1. / 3.)
    for sample in prange(samples):
        row = np.zeros(len(first))
        cut = 0
        for site in range(sites):
            if cut < len(cuts) and site == cuts[cut]:
                forward[sample, cut] = row
                cut += 1
            switched = np.max(row) - penalty
            for state in range(len(first)):
                dosage = haps[first[state], site] + haps[second[state], site]
                value = logs[sample, site, dosage] - center
                row[state] = max(row[state], switched) + value
        row[:] = 0.
        cut = len(cuts) - 1
        for site in range(sites-1, -1, -1):
            switched = np.max(row) - penalty
            for state in range(len(first)):
                dosage = haps[first[state], site] + haps[second[state], site]
                value = logs[sample, site, dosage] - center
                row[state] = max(row[state], switched) + value
            if cut >= 0 and site == cuts[cut]:
                backward[sample, cut] = row
                cut -= 1
    return forward, backward


@njit(cache=True, parallel=True)
def exchange_scores(forward, backward, founders, penalty):
    first, second = founder_scoring._unordered_pairs(founders)
    index = np.empty((founders, founders), np.int64)
    for s in range(len(first)):
        index[first[s], second[s]] = index[second[s], first[s]] = s
    pair_a, pair_b = np.triu_indices(founders, 1)
    samples, boundaries, states = forward.shape
    values = np.empty((samples, boundaries, len(pair_a)), np.float64)
    baseline = np.empty((samples, boundaries), np.float64)
    for sample in prange(samples):
        terms = np.empty(states, np.float64)
        for boundary in range(boundaries):
            left, right = forward[sample, boundary], backward[sample, boundary]
            switched = np.max(left) + np.max(right) - penalty
            for state in range(states):
                terms[state] = left[state] + right[state]
            # At most 2K-2 states change under any transposition. A sorted
            # scan therefore visits at most 2K-1 entries for the unaffected max.
            order = np.argsort(-terms)
            baseline[sample, boundary] = max(switched, terms[order[0]])
            for p in range(len(pair_a)):
                a, b = pair_a[p], pair_b[p]
                best = switched
                for state in order:
                    i, j = first[state], second[state]
                    if ((i != a and i != b and j != a and j != b)
                            or (i == a and j == b)):
                        best = max(best, terms[state])
                        break
                best = max(best, left[index[a,a]] + right[index[b,b]])
                best = max(best, left[index[b,b]] + right[index[a,a]])
                for c in range(founders):
                    if c != a and c != b:
                        best = max(best, left[index[a,c]] + right[index[b,c]])
                        best = max(best, left[index[b,c]] + right[index[a,c]])
                values[sample, boundary, p] = best
    return values.sum(axis=0), baseline.sum(axis=0), pair_a, pair_b


def score_exchanges(haps, evidence, complete, cuts, penalty, prepared=None):
    """Use neutral, called-only arrays, matching canonical full-site scoring."""
    logs = (founder_scoring.prepare_log_evidence(evidence, complete)
            if prepared is None else prepared)
    dosages = founder_scoring._dosage_table(haps)
    if dosages is not None:
        from .founder_site_kernels import exchange_messages
        forward, backward = exchange_messages(dosages, logs, cuts, float(penalty))
        return exchange_scores(forward, backward, len(haps), float(penalty))
    # The direct kernel avoids a large dosage table when workspace RAM is tight.
    # Neutral rows in canonical evidence have zero uncentered emission. Here
    # add the centering constant so neutral sites contribute exactly zero.
    usable = np.any(evidence != evidence[:,:,:1], axis=2) & (evidence.sum(axis=2) > 0)
    usable &= complete[None,:]
    safe_logs = logs.copy()
    safe_logs[~usable] = np.log(1./3.)
    safe_haps = np.where(complete[None,:], haps, 0).astype(np.int8)
    assert np.all(safe_haps >= 0)
    forward, backward = messages(safe_haps, safe_logs, cuts, float(penalty))
    return exchange_scores(forward, backward, len(haps), float(penalty))


from numba.typed import List
from ..core import haplotypes
from . import founder_refinement, founder_path_search, chimera_scoring
from . import observations, paths, hierarchy


def refine_components(prepared, components, neutral, sites, checkpoints=None, *,
                      iterations=20, quota=16, preserve_genotype_fit=True, workspaces=None):
    """Refine phase while preserving every local called/missing allele multiset.

    The pre-count phase pass uses the full objective: its optimized mosaic may
    trade a small genotype-fit loss for fewer switches. Count-reduction refits
    retain the extra genotype-fit veto, since those also change representation.
    """
    results, diagnostics = [], []
    for number, component in enumerate(components):
        token = f"founder_paired_exchange.component{number}"
        cached = founder_refinement._load(checkpoints, token)
        if cached is not None:
            results.append(cached["block"])
            diagnostics.append(cached["diagnostic"])
            continue
        batch = [b for b in prepared if b.positions[0] >= component.positions[0]
                 and b.positions[-1] <= component.positions[-1]]
        positions = np.concatenate([b.positions for b in batch])
        assert np.array_equal(positions, component.positions)
        selected = founder_refinement._local_selection(component, batch)
        original = selected.copy()
        if len(batch) < 2 or len(selected) < 2:
            results.append(component)
            diagnostics.append(dict(component=number, changed=False, history=[]))
            continue
        indices = np.searchsorted(sites, positions)
        if not np.array_equal(sites[indices], positions):
            raise ValueError("paired founder refinement evidence positions do not match")
        workspace = (None if workspaces is None else
                     workspaces.get((int(positions[0]), int(positions[-1]))))
        if workspace is None:
            fitting = np.ascontiguousarray(neutral[:, indices], np.float32)
            leaves = List([np.ascontiguousarray(getattr(b, "missing_aware_inference_discrete_haps", b.discrete_haps), np.int8)
                           for b in batch])
            offsets = np.asarray([0, *np.cumsum([len(b.positions) for b in batch])], np.int64)
            complete = np.concatenate([(np.ones(len(b.positions), np.bool_) if b.keep_flags is None
                else np.asarray(b.keep_flags, np.bool_)) & np.all(
                    observations.founder_inference_panel_from_block_result(b).called, axis=0)
                for b in batch])
            penalty = chimera_scoring.compute_penalty(batch)
            logs = founder_scoring.prepare_log_evidence(fitting, complete)
        else:
            fitting, leaves, offsets = workspace.evidence, workspace.leaves, workspace.offsets
            complete, penalty, logs = workspace.complete, workspace.penalty, workspace.logs
        def evaluate(panel, paint=False):
            alleles = founder_scoring.selected_alleles(leaves, offsets, panel)
            if paint:
                values, switches = founder_scoring.score_and_switch_count(
                    alleles, fitting, complete, penalty, logs)
                return float(values.sum()), int(switches.sum())
            return float(founder_scoring.score_panel(alleles, fitting, complete, penalty, logs).sum())
        current, switches = evaluate(selected, paint=True)
        initial = current
        history = []
        # Bound accumulated float64 addition error in both canonical and flank
        # evaluations. Max is non-expansive. Include three operations/site,
        # sample reduction, path magnitude and a factor for both evaluations.
        operations = 3 * len(positions) + len(fitting) + 16
        epsilon = np.finfo(np.float64).eps * operations
        magnitude = max(abs(float(logs.min())), abs(float(logs.max())))
        rounding_bound = (4 * epsilon / (1 - epsilon) * len(fitting)
            * len(positions) * (magnitude + abs(np.log(1. / 3.)) + abs(penalty)))
        for iteration in range(iterations):
            phase = f"{token}.iteration{iteration}"
            cached = founder_refinement._load(checkpoints, phase)
            if cached is not None:
                selected, current, switches, history = (cached[k] for k in
                    ("selected", "score", "switches", "history"))
                if cached["converged"]:
                    break
                continue
            alleles = founder_scoring.selected_alleles(leaves, offsets, selected)
            exact, reference, first, second = score_exchanges(
                alleles, fitting, complete, offsets[1:-1], penalty, logs)
            max_deviation = float(np.max(np.abs(reference - current)))
            # Forward/backward and whole-chromosome accumulation have different
            # float64 summation orders. Use a scale-aware diagnostic tolerance;
            # candidate acceptance below still uses full canonical rescoring.
            score_tolerance = max(1e-6, 1e-10 * max(1., abs(current)))
            assert max_deviation <= score_tolerance, (number, iteration,
                                                       max_deviation, score_tolerance)
            gains = exact - current
            candidates = np.argsort(-gains.ravel(), kind="stable")[:quota]
            record = dict(iteration=iteration, reference_deviation=max_deviation,
                          proposals=[], accepted=None, pruned_dominated=0)
            best = None
            for rank, flat in enumerate(candidates):
                boundary, pair = divmod(int(flat), len(first))
                if gains[boundary, pair] < .001:
                    break
                if best is not None and exact[boundary, pair] + rounding_bound < best[0]:
                    # Sorted exact scores: none of the remaining proposals can
                    # beat this guard-passing winner. Near ties still rescore.
                    record["pruned_dominated"] = len(candidates) - rank
                    break
                a, b = int(first[pair]), int(second[pair])
                trial = selected.copy()
                trial[[a,b], boundary+1:] = selected[[b,a], boundary+1:]
                score, next_switches = evaluate(trial, paint=True)
                assert abs(score - exact[boundary,pair]) <= score_tolerance
                emission_gain = score - current + penalty * (next_switches - switches)
                proposal = dict(first=a, second=b, prepared_boundary=boundary+1,
                    site_index=int(indices[offsets[boundary+1]]),
                    gain=score-current, genotype_fit_gain=emission_gain,
                    switch_delta=next_switches-switches,
                    genotype_fit_guard_passed=emission_gain >= -1e-6)
                record["proposals"].append(proposal)
                if (score > current+1e-6
                        and (not preserve_genotype_fit or emission_gain >= -1e-6)
                        and (best is None or score > best[0])):
                    best = score, next_switches, trial, proposal
            if best is not None:
                current, switches, selected, record["accepted"] = best
            history.append(record)
            founder_refinement._save(checkpoints, phase, dict(selected=selected, score=current, switches=switches,
                                     history=history, converged=best is None))
            if best is None:
                break
        changed = not np.array_equal(original, selected)
        result = component
        if changed:
            reconstructed = paths.reconstruct_haplotypes_from_beam(
                [(list(row), current) for row in selected], founder_refinement._LeafKeyMap(batch), batch)
            result = hierarchy.convert_reconstruction_to_superblock(reconstructed, batch)
            for side in ("before", "after"):
                for stem in ("missing_aware_break", "missing_aware_break_reason",
                             "missing_aware_joint_informative_samples"):
                    name = f"{stem}_{side}"
                    default = False if stem == "missing_aware_break" else None
                    setattr(result, name, getattr(component, name, default))
            # Every SNP retains the exact multiset of called AND missing alleles.
            assert np.array_equal(np.sort(result.discrete_haps, axis=0),
                                  np.sort(component.discrete_haps, axis=0))
        diagnostic = dict(component=number, changed=changed, initial_likelihood=initial,
                          final_likelihood=current, history=history,
                          changed_local_rows=int(np.count_nonzero(selected != original)),
                          acceptance=("full_objective_and_genotype_fit" if preserve_genotype_fit
                                      else "full_objective_allele_preserving_phase"))
        founder_refinement._save(checkpoints, token, dict(block=result, diagnostic=diagnostic))
        results.append(result)
        diagnostics.append(diagnostic)
    return haplotypes.BlockResults(results), dict(model="exact_paired_suffix_exchange_potts",
                                                 components=diagnostics)
