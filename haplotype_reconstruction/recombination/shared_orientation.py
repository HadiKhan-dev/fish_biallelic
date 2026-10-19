"""Shared-family orientation corrections: screened collapsed coordinate ascent over sample orientation tracts.

A sample's decoder-frame orientation can be wrong over a tract (phase or painting errors); every meiosis it
takes part in then shows the same artifact. Candidate tracts come from a two-state screening posterior of each
meiosis (runs of supported homologue states); for samples with at least ``minimum_incident_edges`` meioses each
candidate flip is scored exactly by the selector x artifact marginal likelihood of all incident edges plus the
orientation-error CTMC prior of the sample, and accepted greedily (descending degree order) when it gains more
than ``minimum_objective_gain`` nats. The objective is conditional (margins, not Bayes factors). Segment trees
(trees.py) make each trial O(log J); maps with flat stretches or no artifact process (non-mixing products) use
the streaming recurrences instead. Under the convention reversal (HAPLOTYPES_TIE_REVERSE,
maths.ties.reverse_conventions) candidates of equal degree are visited by descending sample, start and stop; the
degree order and the acceptance threshold are kept.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numba
import numpy as np
from numba import njit, prange

from ..maths.chains import binary_posterior, chain_segments, contracted_emissions
from ..maths.ties import reverse_conventions
from ..model.genetic_map import haldane_fraction
from .decoder import edge_log_evidence
from .trees import EdgeProductTree, OrientationPriorTree


@dataclass(frozen=True)
class OrientationMove:
    """An accepted orientation flip of ``sample`` on sites [start_index, stop_index) (bp half-open)."""

    sample_index: int
    start_index: int
    stop_index: int
    start_bp: int
    stop_bp: int
    sweep: int
    objective_gain: float
    log_likelihood_gain: float
    log_prior_change: float
    incident_edges: tuple[int, ...]
    per_edge_log_likelihood_gain: tuple[float, ...]


@dataclass(frozen=True)
class SharedOrientationFit:
    """Diagnostics of the fit (the corrected orientation itself is applied before decoding)."""

    candidates: int
    evaluations: int
    accepted: tuple[OrientationMove, ...]
    sweeps: int
    converged: bool
    initial_objective: float
    final_objective: float
    corrected_samples: int
    flipped_markers: int
    elapsed_seconds: float


@njit(cache=True)
def edge_observations(calls, orientation, positions, runs, parent, child, slot,
                      maximum_gap, flip_sample=-1, flip_left=0, flip_right=0):
    """(site indices, observations, resets) of one edge under ``orientation`` (optionally a trial flip)."""
    indices = np.empty(len(positions), dtype=np.int64)
    obs = np.empty(len(positions), dtype=np.int8)
    resets = np.empty(len(positions), dtype=np.bool_)
    count = 0
    for j in range(len(positions)):
        if runs[parent, j] < 0 or runs[child, j] < 0:
            continue
        p_phase = orientation[parent, j]
        c_phase = orientation[child, j]
        if flip_left <= j < flip_right:
            if parent == flip_sample:
                p_phase ^= 1
            if child == flip_sample:
                c_phase ^= 1
        p0 = calls[parent, j, p_phase]
        p1 = calls[parent, j, 1 ^ p_phase]
        c = calls[child, j, slot ^ c_phase]
        if p0 < 0 or p1 < 0 or p0 == p1 or c < 0:
            continue
        reset = count == 0
        if count:
            previous = indices[count - 1]
            reset = (runs[parent, j] != runs[parent, previous]
                     or runs[child, j] != runs[child, previous]
                     or positions[j] - positions[previous] > maximum_gap)
        indices[count], obs[count], resets[count] = j, int(c != p0), reset
        count += 1
    return indices[:count], obs[:count], resets[:count]


@njit(cache=True)
def orientation_log_prior(path, positions, runs, entry_rate, mean_bp, flip_left=0, flip_right=0):
    """Log prior of a sample's orientation path (optionally with [flip_left, flip_right) flipped).

    The asymmetric CTMC prior changes inside a tract, not only at its ends; uncovered markers reset it.
    """
    stationary = entry_rate / (entry_rate + 1 / mean_bp)
    value = 0.
    previous = -1
    for j in range(len(positions)):
        if runs[j] < 0:
            previous = -1
            continue
        current_phase = path[j] ^ (flip_left <= j < flip_right)
        if previous < 0 or runs[j] != runs[previous]:
            probability = stationary if current_phase else 1 - stationary
        else:
            previous_phase = path[previous] ^ (flip_left <= previous < flip_right)
            changed = -math.expm1(-(entry_rate + 1 / mean_bp) * (positions[j] - positions[previous]))
            enter, leave = stationary * changed, (1 - stationary) * changed
            probability = ((leave if not current_phase else 1 - leave) if previous_phase
                           else (enter if current_phase else 1 - enter))
        value += math.log(probability)
        previous = j
    return value


@njit(cache=True)
def screening_posterior(observations, genetic, resets, error):
    """P(homologue 0) of a two-state Haldane chain on genetic coordinates, restarting at resets.

    Markers joined by a zero-probability transition (a flat stretch of a supplied map) cannot switch: they are
    contracted into one chain step whose evidence is combined in log space (chains.contracted_emissions), so
    contradicting evidence on the two sides of a flat stretch cannot underflow both states (base divided 0 by 0
    there; fixes.md A5). Without such transitions every step is one marker.
    """
    n = len(observations)
    emission = np.empty((n, 2))
    theta = np.empty(max(0, n - 1))
    for i in range(n):
        emission[i, 0] = 1 - error if observations[i] == 0 else error
        emission[i, 1] = 1 - emission[i, 0]
        if i:
            theta[i - 1] = .5 if resets[i] else haldane_fraction(genetic[i] - genetic[i - 1])
    starts, trans = chain_segments(theta, n)
    if len(starts) - 1 >= n:
        return binary_posterior(emission, theta, -1)[:, 0]
    compact, _ = contracted_emissions(emission, starts, -1)
    runs = binary_posterior(compact, trans, -1)
    posterior = np.empty(n)
    for run in range(len(starts) - 1):
        posterior[starts[run]:starts[run + 1]] = runs[run, 0]
    return posterior


@njit(cache=True)
def supported_tracts(indices, posterior, resets, threshold):
    """Candidate tracts (first marker, stop marker, kind) from runs of supported homologue states.

    Kind 0 lies between two opposite supported runs; kinds 1 and 2 are the leading and trailing runs of a reset
    segment (one state change and the segment's first or last informative marker bound them).
    """
    output = []
    previous_state = -1
    previous_start = 0
    segment_start = 0
    run_count = 0
    for i, probability in enumerate(posterior):
        if resets[i]:
            if run_count >= 2:
                output.append((int(indices[previous_start]), int(indices[i - 1]) + 1, 2))
            run_count = 0
            previous_state = -1
            segment_start = i
        if max(probability, 1 - probability) < threshold:
            continue
        state = int(probability < .5)
        if state != previous_state:
            if run_count >= 2:
                output.append((int(indices[previous_start]), int(indices[i]), 0))
            elif run_count == 1:
                output.append((int(indices[segment_start]), int(indices[i]), 1))
            previous_state = state
            previous_start = i
            run_count += 1
    if run_count >= 2:
        output.append((int(indices[previous_start]), int(indices[len(indices) - 1]) + 1, 2))
    return output


@njit(cache=True, parallel=True)
def _screen_edge_tracts(calls, orientation, positions, runs, parents, children, slots,
                        genetic, maximum_gap, error, support):
    results = [np.empty((0, 3), dtype=np.int64) for _ in range(len(parents))]
    for edge in prange(len(parents)):
        indices, obs, resets = edge_observations(
            calls, orientation, positions, runs, parents[edge], children[edge], slots[edge], maximum_gap)
        posterior = screening_posterior(obs, genetic[indices], resets, error)
        tracts = supported_tracts(indices, posterior, resets, support)
        result = np.empty((len(tracts), 3), dtype=np.int64)
        for i in range(len(tracts)):
            result[i, 0], result[i, 1], result[i, 2] = tracts[i]
        results[edge] = result
    return results


@njit(cache=True, parallel=True)
def _score_edges(calls, orientation, positions, runs, parents, children, slots,
                 genetic, edges, maximum_gap, error, artifact_rate, artifact_mean):
    scores = np.empty(len(edges))
    for i in prange(len(edges)):
        edge = edges[i]
        indices, obs, resets = edge_observations(calls, orientation, positions, runs,
                                                 parents[edge], children[edge], slots[edge], maximum_gap)
        scores[i] = edge_log_evidence(obs, positions[indices], resets, genetic[indices],
                                      error, artifact_rate, artifact_mean)
    return scores


@njit(cache=True, parallel=True)
def _score_candidate_edges(calls, orientation, positions, runs, parents, children,
                           slots, genetic, trials, edges, maximum_gap, error, artifact_rate, artifact_mean):
    """Read-only trial flips: each job evaluates one candidate/incident edge."""
    scores = np.empty(len(edges))
    for i in prange(len(edges)):
        edge = edges[i]
        sample, left, right = trials[i]
        indices, obs, resets = edge_observations(
            calls, orientation, positions, runs, parents[edge], children[edge], slots[edge], maximum_gap,
            sample, left, right)
        scores[i] = edge_log_evidence(obs, positions[indices], resets, genetic[indices],
                                      error, artifact_rate, artifact_mean)
    return scores


@njit(cache=True, parallel=True)
def _orientation_priors(orientation, positions, runs, entry_rate, mean_bp):
    priors = np.empty(len(orientation))
    for sample in prange(len(orientation)):
        priors[sample] = orientation_log_prior(orientation[sample], positions, runs[sample], entry_rate, mean_bp)
    return priors


@njit(cache=True, parallel=True)
def _candidate_orientation_priors(orientation, positions, runs, trials, entry_rate, mean_bp):
    priors = np.empty(len(trials))
    for i in prange(len(trials)):
        sample, left, right = trials[i]
        priors[i] = orientation_log_prior(orientation[sample], positions, runs[sample], entry_rate, mean_bp,
                                          left, right)
    return priors


def screen_candidates(data, orientation, config):
    """Sorted candidate tracts (sample, first marker, stop marker) and the incident edges of every sample."""
    calls, positions, runs, parents, children, slots, genetic = data
    shared = config.shared_orientation
    adjacent = [[] for _ in range(len(calls))]
    for edge, (parent, child) in enumerate(zip(parents, children)):
        adjacent[parent].append(edge)
        adjacent[child].append(edge)
    proposals = _screen_edge_tracts(calls, orientation, positions, runs, parents, children, slots, genetic,
                                    config.maximum_gap_bp, config.copy_error, shared.candidate_support)
    candidates = set()
    run_limits = {}
    for edge, tracts in enumerate(proposals):
        for left, right, kind in tracts:
            for sample in (int(parents[edge]), int(children[edge])):
                if len(adjacent[sample]) < shared.minimum_incident_edges:
                    continue
                start, stop = int(left), int(right)
                # Run identifiers are contiguous, so both ends locate the tract.
                if runs[sample, start] < 0 or runs[sample, start] != runs[sample, stop - 1]:
                    continue
                if kind:
                    # A boundary tract reaches this individual's own run end, which can lie beyond the edge's
                    # informative markers.
                    if sample not in run_limits:
                        run_limits[sample] = np.r_[0, np.flatnonzero(np.diff(runs[sample])) + 1, len(positions)]
                    limits = run_limits[sample]
                    cut = np.searchsorted(limits, start, side="right")
                    if kind == 1:
                        start = int(limits[cut - 1])
                    else:
                        stop = int(limits[cut])
                # No marker of the same run follows a trailing tract.
                end = positions[stop - 1] if kind == 2 else positions[stop]
                if end - positions[start] > shared.maximum_candidate_bp:
                    continue
                candidates.add((sample, start, stop))
    return sorted(candidates), adjacent


def _candidate_scores(data, orientation, candidates, adjacent, config, epochs, orientation_epochs,
                      edge_products, prior_sums):
    """Batch independent trial work; consume it in the greedy order.

    An accepted flip invalidates its incident-edge scores and its sample's prior (separate epochs: neighbouring
    flips do not change that prior); stale work is recomputed immediately before its greedy decision, so the
    batch width (the thread count) never changes a decision.
    """
    shared = config.shared_orientation
    offset = 0
    while offset < len(candidates):
        width = numba.get_num_threads()
        stop = offset
        jobs, edges, cuts = [], [], [0]
        while stop < len(candidates) and (len(edges) < width or stop == offset):
            trial = candidates[stop]
            incident = adjacent[trial[0]]
            jobs.extend([trial] * len(incident))
            edges.extend(incident)
            cuts.append(len(edges))
            stop += 1
        edge_array = np.asarray(edges, dtype=np.int64)
        trial_array = np.asarray(jobs, dtype=np.int64).reshape((-1, 3))
        candidate_array = np.asarray(candidates[offset:stop], dtype=np.int64).reshape((-1, 3))
        versions = epochs[edge_array].copy()
        prior_versions = orientation_epochs[candidate_array[:, 0]].copy()
        if edge_products is None:
            values = _score_candidate_edges(
                data[0], orientation, *data[1:], trial_array, edge_array, config.maximum_gap_bp,
                config.copy_error, config.artifact_rate, config.artifact_mean_bp)
            priors = _candidate_orientation_priors(
                orientation, data[1], data[2], candidate_array, shared.error_rate, shared.error_mean_bp)
        else:
            indexed_jobs = np.concatenate([
                edge_products.jobs(*trial, adjacent[trial[0]]) for trial in candidate_array])
            prior_jobs = prior_sums.jobs(candidate_array)
            values = edge_products.score(indexed_jobs)
            priors = prior_sums.score(prior_jobs)
        for i in range(offset, stop):
            lo, hi = cuts[i - offset:i - offset + 2]
            incident = edge_array[lo:hi]
            result = values[lo:hi]
            if np.any(epochs[incident] != versions[lo:hi]):
                if edge_products is None:
                    result = _score_candidate_edges(
                        data[0], orientation, *data[1:], trial_array[lo:hi], incident, config.maximum_gap_bp,
                        config.copy_error, config.artifact_rate, config.artifact_mean_bp)
                else:
                    result = edge_products.score(indexed_jobs[lo:hi])
            sample, left, right = candidates[i]
            prior = priors[i - offset]
            if orientation_epochs[sample] != prior_versions[i - offset]:
                if prior_sums is None:
                    prior = orientation_log_prior(orientation[sample], data[1], data[2][sample],
                                                  shared.error_rate, shared.error_mean_bp, left, right)
                else:
                    prior = prior_sums.score(prior_jobs[i - offset:i - offset + 1])[0]
            yield sample, left, right, incident, result, prior
        offset = stop


def fit_shared_orientations(data, config):
    """(orientation int8 [N, L], SharedOrientationFit) for decoder-frame data (stage.shared_data). Equal-degree
    candidates go by descending sample, start and stop under the convention reversal (module docstring)."""
    started = time.monotonic()
    shared = config.shared_orientation
    calls, positions, runs, parents, children, slots, genetic = data
    orientation = np.zeros(calls.shape[:2], dtype=np.int8)
    candidates, adjacent = screen_candidates(data, orientation, config)
    edge_products = prior_sums = None
    # Nonmixing products can lose initial-state mass (no artifact process, or flat stretches of the map): keep
    # the scaled streaming recurrences there.
    mixing = config.artifact_rate > 0 and np.all(np.diff(genetic) > 0)
    if candidates and mixing:
        edge_products = EdgeProductTree(data, candidates, adjacent, config)
        prior_sums = OrientationPriorTree(positions, runs, candidates, shared)
        scores = np.empty(len(parents))
        scores[edge_products.edges] = edge_products.current_scores()
        remaining = np.setdiff1d(np.arange(len(parents)), edge_products.edges)
        scores[remaining] = _score_edges(calls, orientation, positions, runs, parents, children, slots, genetic,
                                         remaining, config.maximum_gap_bp, config.copy_error,
                                         config.artifact_rate, config.artifact_mean_bp)
    else:
        scores = _score_edges(calls, orientation, positions, runs, parents, children, slots, genetic,
                              np.arange(len(parents), dtype=np.int64), config.maximum_gap_bp, config.copy_error,
                              config.artifact_rate, config.artifact_mean_bp)
    prior = _orientation_priors(orientation, positions, runs, shared.error_rate, shared.error_mean_bp)
    if prior_sums is not None:
        prior[prior_sums.samples] = prior_sums.current_scores()
    initial_objective = float(scores.sum() + prior.sum())
    accepted = []
    tested = 0
    converged = False
    # Deterministic descending degree order lets well-supported shared parents resolve before the more
    # ambiguous degree-two child cases. Equal degrees: sample, start, stop (all descending under the reversal).
    if reverse_conventions():
        candidates.sort(key=lambda c: (-len(adjacent[c[0]]), -c[0], -c[1], -c[2]))
    else:
        candidates.sort(key=lambda c: (-len(adjacent[c[0]]), c[0], c[1], c[2]))
    epochs = np.zeros(len(parents), dtype=np.int64)
    orientation_epochs = np.zeros(len(calls), dtype=np.int64)
    sweep = 0
    for sweep in range(shared.maximum_sweeps):
        changes = 0
        for sample, left, right, incident, new_scores, new_prior in _candidate_scores(
                data, orientation, candidates, adjacent, config, epochs, orientation_epochs,
                edge_products, prior_sums):
            per_edge = new_scores - scores[incident]
            prior_change = float(new_prior - prior[sample])
            gain = float(per_edge.sum() + prior_change)
            tested += 1
            if gain > shared.minimum_objective_gain:
                if edge_products is None:
                    orientation[sample, left:right] ^= 1
                else:
                    # Difference encoding makes accepted range flips O(1); the trees own the current paths
                    # until the final materialisation.
                    orientation[sample, left] ^= 1
                    if right < len(positions):
                        orientation[sample, right] ^= 1
                    edge_products.accept(edge_products.jobs(sample, left, right, incident))
                    prior_sums.accept(prior_sums.jobs([(sample, left, right)])[0])
                scores[incident] = new_scores
                epochs[incident] += 1
                orientation_epochs[sample] += 1
                prior[sample] = new_prior
                # Half-open: a tract reaching the last marker stops one bp beyond it.
                stop_bp = int(positions[right]) if right < len(positions) else int(positions[-1]) + 1
                accepted.append(OrientationMove(
                    int(sample), int(left), int(right), int(positions[left]), stop_bp, sweep + 1, gain,
                    float(per_edge.sum()), prior_change, tuple(map(int, incident)), tuple(map(float, per_edge))))
                changes += 1
        if not changes:
            converged = True
            break
    if edge_products is not None:
        np.bitwise_xor.accumulate(orientation, axis=1, out=orientation)
    fit = SharedOrientationFit(
        len(candidates), tested, tuple(accepted), sweep + 1, converged, initial_objective,
        float(scores.sum() + prior.sum()), int(np.sum(np.any(orientation, axis=1))), int(orientation.sum()),
        time.monotonic() - started)
    return orientation, fit
