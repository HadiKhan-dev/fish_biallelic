"""Painting's copying chain with panel errors (release 2.14): class x fidelity track states.

Model (docs/methods/painting.md). A track state is (label, fidelity): the label a trajectory class or BACKGROUND
(state S - 1 of the S labels), the fidelity faithful (0) or panel error (1; anchored classes only). A faithful track
emits its class's calls (two faithful tracks of one class share their latent allele, as in the robust bin emission);
an error track emits as BACKGROUND (alleles from the population frequencies): its row is locally wrong for this
sample, or the sample's haplotype is locally absent from the panel, while the founder label continues.

Per track and bin edge, independently of the other track: the label follows the jump process of painting.transitions
(it stays with probability 1 - theta, else moves to one of the other S - 1 labels uniformly); a label that stays keeps
or changes its fidelity by the two-state chain faithful -> error with probability a, error -> faithful with
probability b (panel-error runs along the panel; 1 / b bins is the mean length of a run that keeps its label); a
label that moves draws its fidelity from the chain's stationary law (error with probability a / (a + b)); BACKGROUND
and unanchored classes are always faithful. A panel-error run thus contributes log(1/a) at its entry and log(1/b) at
its exit (besides its duration and label-stay factors) and keeps the founder label, where a label-level model pays two
ancestry changes (an excursion) to escape a locally wrong row. Painted switches are label
changes only.

Emissions are the robust bin emissions, each sample's bin divided by its design effect 1 + (n - 1) r
(painting.hmm.design_effect, PaintingConfig.emission_site_correlation), the composite-likelihood adjustment of the
per-site product.

The first bin's prior is the chain's stationary law (labels uniform; an anchored label's fidelity error with
probability a / (a + b)); the mirrored traceback convention decodes the same model backwards (transposed transition,
that prior at its last step).

Track states are ordered: the S faithful states (index = label), then one error state per anchored class in label
order. The pair transition is the product of the two tracks' transitions, so every recursion runs in two stages
(track 1, then track 2), each O(S') per state from per-label sums: O(S'^2) per bin and sample with S' <= 2 S - 1
track states (K = 30: S' <= 61).

Parameters (rho, a, b) are estimated per component by EM over all its eligible samples (estimate_chain).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numba import njit, prange

from ..maths import ties
from . import hmm
from .transitions import MINIMUM_SWITCH_PROBABILITY, switch_probabilities

# Bounds of the estimates: rho per Morgan per track (from almost none, a founder's own haplotype, to far beyond any
# cross); a and b per bin edge. A bound reached is reported in the trajectory, not an error: a = 1e-6 is a boundary
# fit (no support for error runs under this model and schedule), not a test of their absence.
RATE_BOUNDS = (1e-2, 1e3)
ENTRY_BOUNDS = (1e-6, 0.5)
EXIT_BOUNDS = (1e-3, 0.999)


@dataclass(frozen=True)
class TrackStates:
    """The track states of a component: label, fidelity and emission label int64[S'] (an error track emits as
    BACKGROUND), and error_of int64[S] (the error state of each label, -1 when it has none)."""

    labels: np.ndarray
    fidelity: np.ndarray
    emission: np.ndarray
    error_of: np.ndarray

    @property
    def n_track(self) -> int:
        return len(self.labels)


def track_states(capable: np.ndarray) -> TrackStates:
    """The track states for ``capable`` int[S] (1 = anchored class with an error state; BACKGROUND last, 0)."""
    capable = np.asarray(capable, dtype=np.int64)
    n_labels = len(capable)
    errors = np.flatnonzero(capable)
    labels = np.concatenate((np.arange(n_labels), errors)).astype(np.int64)
    fids = np.concatenate((np.zeros(n_labels), np.ones(len(errors)))).astype(np.int64)
    emission = np.where(fids == 1, n_labels - 1, labels).astype(np.int64)
    error_of = np.full(n_labels, -1, dtype=np.int64)
    error_of[errors] = n_labels + np.arange(len(errors))
    return TrackStates(labels, fids, emission, error_of)


@njit(cache=True, parallel=True)
def _structured_emissions(raw, emission_label, n_labels):
    n_samples, _, n_bins = raw.shape
    n_track = len(emission_label)
    out = np.empty((n_samples, n_track * n_track, n_bins))
    for task in prange(n_samples * n_track):
        sample, first = task // n_track, task % n_track
        for second in range(n_track):
            source = emission_label[first] * n_labels + emission_label[second]
            for bin_index in range(n_bins):
                out[sample, first * n_track + second, bin_index] = raw[sample, source, bin_index]
    return out


def structured_emissions(raw: np.ndarray, tracks: TrackStates) -> np.ndarray:
    """float64[N, S'*S', B] pair emissions of the track states from the (tempered) label emissions raw [N, S*S, B]."""
    return _structured_emissions(np.ascontiguousarray(raw), tracks.emission, len(tracks.error_of))


@njit(cache=True, inline="always")
def _log_terms(theta, n_labels, a, b):
    """(log stay, log move to one particular other label, log F[2, 2], log pi[2])."""
    log_stay = math.log1p(-theta)
    log_move = math.log(theta) - math.log(n_labels - 1.0) if n_labels > 1 else -np.inf
    fidelity = np.empty((2, 2))
    fidelity[0, 0] = math.log1p(-a)
    fidelity[0, 1] = math.log(a)
    fidelity[1, 0] = math.log(b)
    fidelity[1, 1] = math.log1p(-b)
    error = a / (a + b)
    stationary = np.empty(2)
    stationary[0] = math.log1p(-error)
    stationary[1] = math.log(error)
    return log_stay, log_move, fidelity, stationary


@njit(cache=True, inline="always")
def _target_log(label_from, fid_from, label_to, fid_to, capable, log_stay, log_move, fidelity, stationary):
    """log T(track state -> track state)."""
    if label_from == label_to:
        if capable[label_to]:
            return log_stay + fidelity[fid_from, fid_to]
        return log_stay if fid_to == 0 else -np.inf
    if capable[label_to]:
        return log_move + stationary[fid_to]
    return log_move if fid_to == 0 else -np.inf


def initial_log(tracks: TrackStates, capable: np.ndarray, a: float, b: float) -> np.ndarray:
    """float64[S']: log of the chain's stationary law of a track's fidelity given its label (labels uniform, a
    constant left out): log(1 - a/(a + b)) and log(a/(a + b)) for an anchored label's faithful and error states, 0 for
    the other labels. The first bin's prior (the chain is stationary, so its marginal at either end
    of a component is this law; a uniform law over track states would give the error states as much prior mass as the
    faithful ones). The mirrored Viterbi convention applies it at the original first bin, its last step."""
    capable = np.asarray(capable, dtype=np.int64)
    error = a / (a + b)
    result = np.zeros(tracks.n_track)
    anchored = capable[tracks.labels] == 1
    result[anchored & (tracks.fidelity == 0)] = math.log1p(-error)
    result[anchored & (tracks.fidelity == 1)] = math.log(error)
    return result


@njit(cache=True, inline="always")
def _first_within(values, labels, threshold, excluded, reverse):
    """The smallest index y (the largest under ``reverse``) with values[y] >= threshold and labels[y] != excluded
    (-1 when there is none)."""
    n = len(values)
    for rank in range(n):
        y = n - 1 - rank if reverse else rank
        if labels[y] != excluded and values[y] >= threshold:
            return y
    return -1


@njit(cache=True)
def _max_stage(values, labels, fids, error_of, capable, log_stay, log_move, fidelity, stationary, tie, reverse,
               transpose, adjusted, out_value, out_index):
    """out[z] = max_y values[y] + log T(y -> z) over track states (exact maxima; with ``transpose``, log T(z -> y): the
    mirrored chain of the other traceback convention, since the chain is not time-symmetric), with the argmax under
    the stated tie rule (VITERBI_TIE): stay in z when within ``tie`` of the maximum, else the label's other fidelity
    state when within ``tie`` of a label change, else a label change from the smallest index (the largest under
    ``reverse``) whose value is within ``tie`` of the best other label's. A label change has one weight for every
    source of another label up to a term of the target's fidelity (forward: its stationary probability) or of the
    source's (transposed), so the best change comes from the largest ``adjusted`` value (values plus the source term)
    among the other labels: the global best M1 (label L1), or, for z of label L1, the best M2 of the labels other than
    L1. ``adjusted`` is a work array of length S'. O(S')."""
    n = len(values)
    for y in range(n):
        adjusted[y] = values[y]
        if transpose and capable[labels[y]]:
            adjusted[y] += stationary[fids[y]]
    best = -np.inf
    best_label = -1
    for y in range(n):
        if adjusted[y] > best:
            best = adjusted[y]
            best_label = labels[y]
    second = -np.inf
    for y in range(n):
        if labels[y] != best_label and adjusted[y] > second:
            second = adjusted[y]
    # Tie-chosen sources of a label change: for targets not of label L1 (threshold M1 - tie), the first index within
    # the tie, or the first within it among labels other than that index's; for targets of label L1 (M2 - tie).
    first = _first_within(adjusted, labels, best - tie, -1, reverse)
    first_other = _first_within(adjusted, labels, best - tie, labels[first], reverse) if first >= 0 else -1
    second_first = _first_within(adjusted, labels, second - tie, best_label, reverse) if second > -np.inf else -1
    for z in range(n):
        label = labels[z]
        fid = fids[z]
        # Same-label predecessors: the label's faithful state (index = label) and its error state.
        top = -np.inf
        top_index = -1
        stay_value = -np.inf
        for y_fid in range(2):
            y = label if y_fid == 0 else error_of[label]
            if y < 0:
                continue
            if capable[label]:
                value = values[y] + log_stay + (fidelity[fid, y_fid] if transpose else fidelity[y_fid, fid])
            elif fid == 0:
                value = values[y] + log_stay
            else:
                value = -np.inf
            if y == z:
                stay_value = value
            if value > top:
                top = value
                top_index = y
        # A label change: the exact best value among the other labels, from the tie-chosen source.
        if label != best_label:
            change = best
            source = first if labels[first] != label else first_other
        else:
            change = second
            source = second_first
        if source >= 0 and change > -np.inf:
            target = 0.0
            if not transpose:
                if capable[label]:
                    target = stationary[fid]
                elif fid != 0:
                    target = -np.inf
            value = change + log_move + target
            if value > top + tie:
                top = value
                top_index = source
            elif value > top:
                top = value      # within the tie: keep the same-label predecessor
        if stay_value >= top - tie and stay_value > -np.inf:
            top_index = z
        out_value[z] = top
        out_index[z] = top_index


@njit(cache=True, parallel=True)
def _viterbi_kernel(emissions, labels, fids, error_of, capable, initial, theta, a, b, tie, reverse):
    """Max-product recursion over the bins in the given order; ``reverse`` (the mirrored convention, emissions and
    theta reversed by the caller) uses the transposed transition and the reversed index ties, and takes the initial
    law at its last step (the original first bin), so both conventions maximise the same path score."""
    n_samples, n_pairs, n_bins = emissions.shape
    n = len(labels)
    n_labels = len(capable)
    result = np.empty((n_samples, 2, n_bins), dtype=np.int32)
    for sample in prange(n_samples):
        back = np.empty((n_bins, n_pairs), dtype=np.int32)
        previous = np.empty(n_pairs)
        for first in range(n):
            for second in range(n):
                state = first * n + second
                previous[state] = emissions[sample, state, 0]
                if not reverse:
                    previous[state] += initial[first] + initial[second]
        previous -= np.max(previous)
        column = np.empty(n)
        adjusted = np.empty(n)
        stage_value = np.empty((n, n))
        stage_index = np.empty((n, n), dtype=np.int64)
        out_value = np.empty(n)
        out_index = np.empty(n, dtype=np.int64)
        current = np.empty(n_pairs)
        for bin_index in range(1, n_bins):
            log_stay, log_move, fidelity, stationary = _log_terms(theta[bin_index - 1], n_labels, a, b)
            # Stage 1: track 1, for every predecessor state of track 2.
            for y2 in range(n):
                for y1 in range(n):
                    column[y1] = previous[y1 * n + y2]
                _max_stage(column, labels, fids, error_of, capable, log_stay, log_move, fidelity, stationary,
                           tie, reverse, reverse, adjusted, out_value, out_index)
                for z1 in range(n):
                    stage_value[z1, y2] = out_value[z1]
                    stage_index[z1, y2] = out_index[z1]
            # Stage 2: track 2.
            for z1 in range(n):
                for y2 in range(n):
                    column[y2] = stage_value[z1, y2]
                _max_stage(column, labels, fids, error_of, capable, log_stay, log_move, fidelity, stationary,
                           tie, reverse, reverse, adjusted, out_value, out_index)
                for z2 in range(n):
                    y2 = out_index[z2]
                    current[z1 * n + z2] = out_value[z2] + emissions[sample, z1 * n + z2, bin_index]
                    back[bin_index, z1 * n + z2] = stage_index[z1, y2] * n + y2
            current -= np.max(current)
            previous[:] = current
        if reverse:
            for first in range(n):
                for second in range(n):
                    previous[first * n + second] += initial[first] + initial[second]
        top = np.max(previous)
        state = 0
        for rank in range(n_pairs):
            candidate = n_pairs - 1 - rank if reverse else rank
            if previous[candidate] >= top - tie:
                state = candidate
                break
        result[sample, 0, n_bins - 1] = state // n
        result[sample, 1, n_bins - 1] = state % n
        for bin_index in range(n_bins - 1, 0, -1):
            state = back[bin_index, state]
            result[sample, 0, bin_index - 1] = state // n
            result[sample, 1, bin_index - 1] = state % n
    return result


def viterbi_track_states(emissions: np.ndarray, tracks: TrackStates, capable: np.ndarray, theta: np.ndarray,
                         a: float, b: float, *, reverse: bool = False) -> np.ndarray:
    """int32[N, 2, B] Viterbi track states (indices into ``tracks``) of the structured emissions [N, S'*S', B].
    ``reverse`` decodes the other traceback convention (release 2.2 convention reversal): the bins in reverse order
    under the transposed transition (the same paths and scores, the chain being not time-symmetric) with the reversed
    index ties, so co-optimal switches land on the rightmost tied edge."""
    capable = np.ascontiguousarray(capable, dtype=np.int64)
    arguments = (tracks.labels, tracks.fidelity, tracks.error_of, capable, initial_log(tracks, capable, a, b))
    if not reverse:
        return _viterbi_kernel(np.ascontiguousarray(emissions), *arguments, np.ascontiguousarray(theta),
                               float(a), float(b), float(hmm.VITERBI_TIE), False)
    mirrored = _viterbi_kernel(np.ascontiguousarray(emissions[:, :, ::-1]), *arguments,
                               np.ascontiguousarray(theta[::-1]), float(a), float(b), float(hmm.VITERBI_TIE), True)
    return np.ascontiguousarray(mirrored[:, :, ::-1])


@njit(cache=True, inline="always")
def _sum_stage(values, labels, fids, error_of, capable, n_labels, log_stay, log_move, fidelity, stationary, out):
    """out[z] = sum_y values[y] T(y -> z) (probabilities, values scaled), O(S')."""
    n = len(values)
    total = 0.0
    per_label = np.zeros(n_labels)
    for y in range(n):
        total += values[y]
        per_label[labels[y]] += values[y]
    stay = math.exp(log_stay)
    move = math.exp(log_move)
    for z in range(n):
        label = labels[z]
        fid = fids[z]
        if capable[label]:
            faithful = values[label]
            error = values[error_of[label]]
            same = stay * (faithful * math.exp(fidelity[0, fid]) + error * math.exp(fidelity[1, fid]))
            moved = move * (total - per_label[label]) * math.exp(stationary[fid])
        else:
            same = stay * values[label] if fid == 0 else 0.0
            moved = move * (total - per_label[label]) if fid == 0 else 0.0
        out[z] = same + moved


@njit(cache=True, inline="always")
def _transposed_parts(values, labels, fids, error_of, capable, n_labels, log_stay, log_move, fidelity, stationary,
                      total, moved, started, ended, kept, moved_error, moved_faithful):
    """For every track state y: total[y] = sum_z T(y -> z) values[z], and its parts moved (label changes), started
    (faithful -> error, same label), ended (error -> faithful, same label), kept (same label and fidelity, for an
    anchored label), and of moved the parts into an anchored label's error state (moved_error) and faithful state
    (moved_faithful). O(S')."""
    n = len(values)
    stay = math.exp(log_stay)
    move = math.exp(log_move)
    weighted = 0.0                      # sum_z pi(z) values[z]
    weighted_error = 0.0                # the same over anchored labels' error states
    weighted_faithful = 0.0             # and over anchored labels' faithful states
    per_label = np.zeros(n_labels)
    per_label_error = np.zeros(n_labels)
    per_label_faithful = np.zeros(n_labels)
    for z in range(n):
        label = labels[z]
        if capable[label]:
            value = math.exp(stationary[fids[z]]) * values[z]
            if fids[z] == 1:
                weighted_error += value
                per_label_error[label] += value
            else:
                weighted_faithful += value
                per_label_faithful[label] += value
        else:
            value = values[z] if fids[z] == 0 else 0.0
        weighted += value
        per_label[label] += value
    for y in range(n):
        label = labels[y]
        fid = fids[y]
        moved[y] = move * (weighted - per_label[label])
        moved_error[y] = move * (weighted_error - per_label_error[label])
        moved_faithful[y] = move * (weighted_faithful - per_label_faithful[label])
        started[y] = 0.0
        ended[y] = 0.0
        kept[y] = 0.0
        if capable[label]:
            faithful = values[label]
            error = values[error_of[label]]
            keep = stay * math.exp(fidelity[fid, fid]) * (faithful if fid == 0 else error)
            flip = stay * math.exp(fidelity[fid, 1 - fid]) * (error if fid == 0 else faithful)
            if fid == 0:
                started[y] = flip
            else:
                ended[y] = flip
            kept[y] = keep
            total[y] = moved[y] + keep + flip
        else:
            total[y] = moved[y] + stay * values[label]


@njit(cache=True, parallel=True)
def _chain_kernel(emissions, labels, fids, error_of, capable, initial, theta, a, b, eligible, want_masses):
    """Exact scaled forward-backward of the chain, per eligible sample: expected label changes per edge (both
    tracks); expected error-run starts and ends and kept faithful and kept error transitions of anchored labels that
    stay, and label moves into an anchored label's error and faithful states (summed over edges); the first bin's
    expected anchored tracks in error and faithful; the log-likelihood; and (``want_masses``) the posterior masses of
    every label pair and of every label pair with both tracks faithful, float64[N, S*S, B]."""
    n_samples, n_pairs, n_bins = emissions.shape
    n = len(labels)
    n_labels = len(capable)
    label_changes = np.zeros((n_samples, max(n_bins - 1, 0)))
    starts = np.zeros(n_samples)
    ends = np.zeros(n_samples)
    kept_faithful = np.zeros(n_samples)
    kept_error = np.zeros(n_samples)
    into_error = np.zeros(n_samples)
    into_faithful = np.zeros(n_samples)
    first_error = np.zeros(n_samples)
    first_faithful = np.zeros(n_samples)
    log_likelihood = np.zeros(n_samples)
    pair_mass = np.zeros((n_samples if want_masses else 0, n_labels * n_labels, n_bins))
    faithful_mass = np.zeros((n_samples if want_masses else 0, n_labels * n_labels, n_bins))
    for sample in prange(n_samples):
        if not eligible[sample]:
            continue
        forward = np.empty((n_bins, n_pairs))
        column = np.empty(n)
        out = np.empty(n)
        moved = np.empty(n)
        started = np.empty(n)
        ended = np.empty(n)
        kept = np.empty(n)
        moved_error = np.empty(n)
        moved_faithful = np.empty(n)
        stage = np.empty((n, n))
        top = -np.inf
        for p in range(n_pairs):
            value = emissions[sample, p, 0] + initial[p // n] + initial[p % n]
            forward[0, p] = value
            if value > top:
                top = value
        for p in range(n_pairs):
            forward[0, p] = math.exp(forward[0, p] - top)
        scale = np.sum(forward[0])
        log_likelihood[sample] += top + math.log(scale)
        forward[0] /= scale
        for bin_index in range(1, n_bins):
            log_stay, log_move, fidelity, stationary = _log_terms(theta[bin_index - 1], n_labels, a, b)
            for y2 in range(n):
                for y1 in range(n):
                    column[y1] = forward[bin_index - 1, y1 * n + y2]
                _sum_stage(column, labels, fids, error_of, capable, n_labels, log_stay, log_move, fidelity,
                           stationary, out)
                for z1 in range(n):
                    stage[z1, y2] = out[z1]
            top = np.max(emissions[sample, :, bin_index])
            for z1 in range(n):
                for y2 in range(n):
                    column[y2] = stage[z1, y2]
                _sum_stage(column, labels, fids, error_of, capable, n_labels, log_stay, log_move, fidelity,
                           stationary, out)
                for z2 in range(n):
                    forward[bin_index, z1 * n + z2] = out[z2] * math.exp(
                        emissions[sample, z1 * n + z2, bin_index] - top)
            scale = np.sum(forward[bin_index])
            log_likelihood[sample] += top + math.log(scale)
            forward[bin_index] /= scale
        backward = np.ones(n_pairs)
        weight = np.empty(n_pairs)
        new_backward = np.empty(n_pairs)
        for bin_index in range(n_bins - 1, -1, -1):
            if bin_index == 0:
                total = 0.0
                for p in range(n_pairs):
                    weight[p] = forward[0, p] * backward[p]
                    total += weight[p]
                for p in range(n_pairs):
                    mass = weight[p] / total
                    for track_state in (p // n, p % n):
                        if capable[labels[track_state]]:
                            if fids[track_state] == 1:
                                first_error[sample] += mass
                            else:
                                first_faithful[sample] += mass
            if want_masses:
                total = 0.0
                for p in range(n_pairs):
                    weight[p] = forward[bin_index, p] * backward[p]
                    total += weight[p]
                for p in range(n_pairs):
                    mass = weight[p] / total
                    first, second = p // n, p % n
                    pair = labels[first] * n_labels + labels[second]
                    pair_mass[sample, pair, bin_index] += mass
                    if fids[first] == 0 and fids[second] == 0:
                        faithful_mass[sample, pair, bin_index] += mass
            if bin_index == 0:
                continue
            edge = bin_index - 1
            log_stay, log_move, fidelity, stationary = _log_terms(theta[edge], n_labels, a, b)
            top = np.max(emissions[sample, :, bin_index])
            for p in range(n_pairs):
                weight[p] = math.exp(emissions[sample, p, bin_index] - top) * backward[p]
            # Track 1's parts: contract track 2 first, stage[z1, y2] = sum_z2 T(y2 -> z2) w(z1, z2).
            for z1 in range(n):
                for z2 in range(n):
                    column[z2] = weight[z1 * n + z2]
                _transposed_parts(column, labels, fids, error_of, capable, n_labels, log_stay, log_move, fidelity,
                                  stationary, out, moved, started, ended, kept, moved_error, moved_faithful)
                for y2 in range(n):
                    stage[z1, y2] = out[y2]
            changes = 0.0
            run_starts = 0.0
            run_ends = 0.0
            keep_faithful = 0.0
            keep_error = 0.0
            enter_error = 0.0
            enter_faithful = 0.0
            for y2 in range(n):
                for z1 in range(n):
                    column[z1] = stage[z1, y2]
                _transposed_parts(column, labels, fids, error_of, capable, n_labels, log_stay, log_move, fidelity,
                                  stationary, out, moved, started, ended, kept, moved_error, moved_faithful)
                for y1 in range(n):
                    alpha = forward[edge, y1 * n + y2]
                    new_backward[y1 * n + y2] = out[y1]
                    changes += alpha * moved[y1]
                    enter_error += alpha * moved_error[y1]
                    enter_faithful += alpha * moved_faithful[y1]
                    run_starts += alpha * started[y1]
                    run_ends += alpha * ended[y1]
                    if fids[y1] == 0:
                        keep_faithful += alpha * kept[y1]
                    else:
                        keep_error += alpha * kept[y1]
            # Track 2's parts: contract track 1 first, stage[y1, z2] = sum_z1 T(y1 -> z1) w(z1, z2).
            for z2 in range(n):
                for z1 in range(n):
                    column[z1] = weight[z1 * n + z2]
                _transposed_parts(column, labels, fids, error_of, capable, n_labels, log_stay, log_move, fidelity,
                                  stationary, out, moved, started, ended, kept, moved_error, moved_faithful)
                for y1 in range(n):
                    stage[y1, z2] = out[y1]
            for y1 in range(n):
                for z2 in range(n):
                    column[z2] = stage[y1, z2]
                _transposed_parts(column, labels, fids, error_of, capable, n_labels, log_stay, log_move, fidelity,
                                  stationary, out, moved, started, ended, kept, moved_error, moved_faithful)
                for y2 in range(n):
                    alpha = forward[edge, y1 * n + y2]
                    changes += alpha * moved[y2]
                    enter_error += alpha * moved_error[y2]
                    enter_faithful += alpha * moved_faithful[y2]
                    run_starts += alpha * started[y2]
                    run_ends += alpha * ended[y2]
                    if fids[y2] == 0:
                        keep_faithful += alpha * kept[y2]
                    else:
                        keep_error += alpha * kept[y2]
            norm = 0.0
            for p in range(n_pairs):
                norm += forward[edge, p] * new_backward[p]
            label_changes[sample, edge] = changes / norm
            starts[sample] += run_starts / norm
            ends[sample] += run_ends / norm
            kept_faithful[sample] += keep_faithful / norm
            kept_error[sample] += keep_error / norm
            into_error[sample] += enter_error / norm
            into_faithful[sample] += enter_faithful / norm
            scale = np.sum(new_backward)
            for p in range(n_pairs):
                backward[p] = new_backward[p] / scale
    return (label_changes, starts, ends, kept_faithful, kept_error, into_error, into_faithful, first_error,
            first_faithful, log_likelihood, pair_mass, faithful_mass)


def chain_statistics(emissions: np.ndarray, tracks: TrackStates, capable: np.ndarray, theta: np.ndarray, a: float,
                     b: float, eligible: np.ndarray, *, want_masses: bool = False):
    """(label changes float64[N, B - 1], error-run starts [N], ends [N], kept faithful [N], kept error [N], moves into
    an anchored error state [N] and faithful state [N], first-bin anchored tracks in error [N] and faithful [N],
    log-likelihood [N], label-pair masses [N, S*S, B], faithful label-pair masses [N, S*S, B]) of the structured
    emissions (see _chain_kernel; the masses are empty unless ``want_masses``)."""
    capable = np.ascontiguousarray(capable, dtype=np.int64)
    return _chain_kernel(np.ascontiguousarray(emissions), tracks.labels, tracks.fidelity, tracks.error_of, capable,
                         initial_log(tracks, capable, a, b), np.ascontiguousarray(theta, dtype=np.float64),
                         float(a), float(b), np.ascontiguousarray(eligible, dtype=np.bool_), bool(want_masses))


@njit(cache=True, parallel=True)
def masses_from_pairs(pair_mass, faithful_mass, masks, other_masks, public_ids, local_keys):
    """(qV, qA, qF, qAF) float64[N, B] from label-pair posterior masses [N, S*S, B]: the mass of the unordered public
    (local-key) pairs consistent with the two tracks' candidate masks (as painting.hmm.release_posteriors), all and
    with both tracks faithful; each the smaller over the two decodings' masks."""
    n_samples, n_pairs, n_bins = pair_mass.shape
    n_states = public_ids.shape[0]
    qv = np.zeros((n_samples, n_bins))
    qa = np.zeros((n_samples, n_bins))
    qf = np.zeros((n_samples, n_bins))
    qaf = np.zeros((n_samples, n_bins))
    for task in prange(n_samples * n_bins):
        sample, bin_index = task // n_bins, task % n_bins
        for decoding in range(2):
            source = masks if decoding == 0 else other_masks
            mask_first = source[sample, 0, bin_index]
            mask_second = source[sample, 1, bin_index]
            pf = np.int64(0)
            ps = np.int64(0)
            lf = np.int64(0)
            ls = np.int64(0)
            for state in range(n_states):
                if (mask_first >> state) & 1:
                    pf |= np.int64(1) << public_ids[state]
                    lf |= np.int64(1) << local_keys[bin_index, state]
                if (mask_second >> state) & 1:
                    ps |= np.int64(1) << public_ids[state]
                    ls |= np.int64(1) << local_keys[bin_index, state]
            v = 0.0
            av = 0.0
            f = 0.0
            af = 0.0
            for first in range(n_states):
                for second in range(n_states):
                    mass = pair_mass[sample, first * n_states + second, bin_index]
                    faithful = faithful_mass[sample, first * n_states + second, bin_index]
                    a1 = public_ids[first]
                    a2 = public_ids[second]
                    if (((pf >> a1) & 1) and ((ps >> a2) & 1)) or (((ps >> a1) & 1) and ((pf >> a2) & 1)):
                        v += mass
                        f += faithful
                    k1 = local_keys[bin_index, first]
                    k2 = local_keys[bin_index, second]
                    if (((lf >> k1) & 1) and ((ls >> k2) & 1)) or (((ls >> k1) & 1) and ((lf >> k2) & 1)):
                        av += mass
                        af += faithful
            v = min(1.0, max(0.0, v))
            av = min(1.0, max(0.0, av))
            f = min(1.0, max(0.0, f))
            af = min(1.0, max(0.0, af))
            if decoding == 0:
                qv[sample, bin_index] = v
                qa[sample, bin_index] = av
                qf[sample, bin_index] = f
                qaf[sample, bin_index] = af
            else:
                qv[sample, bin_index] = min(qv[sample, bin_index], v)
                qa[sample, bin_index] = min(qa[sample, bin_index], av)
                qf[sample, bin_index] = min(qf[sample, bin_index], f)
                qaf[sample, bin_index] = min(qaf[sample, bin_index], af)
    return qv, qa, qf, qaf


# ---------------------------------------------------------------------------------------------- the copying model


@dataclass(frozen=True)
class ChainModel:
    """A component's copying model: rate rho (per Morgan per track), a, b, the emissions' site correlation r (the
    design effect's), the per-edge label change probabilities theta float64[B - 1] and the EM trajectory ((rho, a,
    b, log-likelihood) after each update, unrounded; the log-likelihood is of the tempered emissions under the
    values before that update)."""

    rate: float
    entry: float
    exit: float
    site_correlation: float
    theta: np.ndarray
    trajectory: tuple[tuple[float, float, float, float], ...] = ()


def _round_log(value: float, step: float, low: float, high: float) -> float:
    value = min(high, max(low, float(value)))
    return float(min(high, max(low, math.exp(round(math.log(value) / step) * step))))


def rate_objective(rate: float, changes: np.ndarray, tracks: float, morgans: np.ndarray, n_labels: int) -> float:
    """sum_e [n_e log theta_e + (tracks - n_e) log(1 - theta_e)] over the edges of positive map length (the rate's part
    of the expected complete-data log-likelihood), theta_e = (S - 1)/S (1 - exp(-rho d_e S/(S - 1)))."""
    positive = np.asarray(morgans, dtype=np.float64) > 0.0
    n = np.minimum(np.asarray(changes, dtype=np.float64)[positive], tracks)
    d = np.asarray(morgans, dtype=np.float64)[positive]
    theta = np.clip(switch_probabilities(d, rate, n_labels), MINIMUM_SWITCH_PROBABILITY, 1.0 - 1e-15)
    return float(np.sum(n * np.log(theta) + (tracks - n) * np.log1p(-theta)))


def rate_update(changes: np.ndarray, tracks: float, morgans: np.ndarray, n_labels: int,
                previous: float | None = None) -> float:
    """The rho in RATE_BOUNDS maximising rate_objective: a log-spaced scan of 241 points, then a golden-section
    refinement around the best. The objective can have more than one local maximum (each edge's term is unimodal in
    rho, their sum need not be), so this is a numerical search, not a certificate; ``previous`` is kept when the
    search does not improve on it (the update never lowers the objective)."""
    low, high = RATE_BOUNDS
    positive = np.asarray(morgans, dtype=np.float64) > 0.0
    n = np.minimum(np.asarray(changes, dtype=np.float64)[positive], tracks)
    if tracks <= 0.0 or not np.any(positive) or n_labels < 2 or float(np.sum(n)) <= 0.0:
        return low if previous is None else float(previous)

    def objective(log_rate: float) -> float:
        return rate_objective(math.exp(log_rate), changes, tracks, morgans, n_labels)

    grid = np.linspace(math.log(low), math.log(high), 241)
    values = [objective(x) for x in grid]
    best = int(np.argmax(values))
    lo, hi = grid[max(0, best - 1)], grid[min(len(grid) - 1, best + 1)]
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    f1, f2 = objective(x1), objective(x2)
    for _ in range(60):
        if f1 >= f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - ratio * (hi - lo)
            f1 = objective(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + ratio * (hi - lo)
            f2 = objective(x2)
    found = math.exp(0.5 * (lo + hi))
    if previous is not None and objective(math.log(found)) < objective(math.log(previous)):
        return float(previous)
    return found


def fidelity_objective(a: float, b: float, counts) -> float:
    """The (a, b) part of the expected complete-data log-likelihood: with s, e the expected error-run starts and ends,
    kf, ke the expected kept faithful and kept error transitions of anchored labels that stay, and ie, if_ the
    expected anchored tracks drawn into error or faithful fidelity from the stationary law (at the first bin and after
    a label move: pi_error = a / (a + b)),
        s log a + kf log(1 - a) + e log b + ke log(1 - b) + ie log a + if_ log b - (ie + if_) log(a + b)."""
    s, kf, e, ke, ie, if_ = counts
    return float((s + ie) * math.log(a) + kf * math.log1p(-a) + (e + if_) * math.log(b) + ke * math.log1p(-b)
                 - (ie + if_) * math.log(a + b))


def fidelity_update(counts, previous: tuple[float, float]) -> tuple[float, float]:
    """(a, b) in ENTRY_BOUNDS x EXIT_BOUNDS maximising fidelity_objective: bounded quasi-Newton (L-BFGS-B) in
    (log a, log b) from the previous values and from the stay-only ratios s / (s + kf), e / (e + ke); the previous
    values are kept unless the best start improves on them."""
    from scipy.optimize import minimize
    s, kf, e, ke, _, _ = counts
    bounds = [(math.log(ENTRY_BOUNDS[0]), math.log(ENTRY_BOUNDS[1])),
              (math.log(EXIT_BOUNDS[0]), math.log(EXIT_BOUNDS[1]))]

    def clip(value, low_high):
        return min(low_high[1], max(low_high[0], value))

    starts = [(math.log(previous[0]), math.log(previous[1]))]
    if s + kf > 0.0 and e + ke > 0.0 and s > 0.0 and e > 0.0:
        starts.append((clip(math.log(s / (s + kf)), bounds[0]), clip(math.log(e / (e + ke)), bounds[1])))
    best = (fidelity_objective(*previous, counts), tuple(previous))
    for x0 in starts:
        result = minimize(lambda x: -fidelity_objective(math.exp(x[0]), math.exp(x[1]), counts), np.asarray(x0),
                          method="L-BFGS-B", bounds=bounds)
        candidate = (float(math.exp(result.x[0])), float(math.exp(result.x[1])))
        value = fidelity_objective(*candidate, counts)
        if value > best[0]:
            best = (value, candidate)
    return best[1]


def model_for(morgans: np.ndarray, n_labels: int, rate: float, entry: float, exit_: float, site_correlation: float,
              trajectory=()) -> ChainModel:
    return ChainModel(rate=float(rate), entry=float(entry), exit=float(exit_), site_correlation=float(site_correlation),
                      theta=np.ascontiguousarray(switch_probabilities(morgans, rate, n_labels)),
                      trajectory=tuple(trajectory))


def estimate_chain(raw_batches, capable: np.ndarray, morgans: np.ndarray, config) -> ChainModel:
    """EM estimate of (rho, a, b) over all eligible samples of a component. ``raw_batches()`` yields (tempered label
    emissions float64[n, S*S, B], eligible bool[n]) batch by batch, for every sample; it is called once per update.

    From the fixed starting values (PaintingConfig.initial_*), ``estimation_iterations`` updates, each an exact
    forward-backward pass under the current values (chain_statistics) followed by the M-step of the expected
    complete-data log-likelihood: rho by rate_update, (a, b) by fidelity_update (both include every term of the model
    in them: the label changes, the fidelity chain of labels that stay, and the stationary fidelity draws at the first
    bin and after label moves); neither update lowers its part of the objective. A fixed number of updates from fixed
    starts: a deterministic, truncated estimator, not a convergence test. The results are clipped to their bounds and
    rounded to ``estimate_grid`` in natural-log space. The trajectory records, per update, the new values and the
    log-likelihood of the tempered emissions under the values before it (up to the constant 2 log S per sample of
    the uniform label law).
    """
    capable = np.ascontiguousarray(capable, dtype=np.int64)
    n_labels = len(capable)
    tracks_states = track_states(capable)
    rate = float(config.initial_switch_rate)
    entry = float(config.initial_error_entry)
    exit_ = float(config.initial_error_exit)
    if not np.any(capable):     # no anchored class, no error state: a and b unused, reported as no runs
        entry, exit_ = ENTRY_BOUNDS[0], EXIT_BOUNDS[1]
    site_correlation = float(config.emission_site_correlation)
    trajectory = []
    for _ in range(int(config.estimation_iterations)):
        theta = switch_probabilities(morgans, rate, n_labels)
        changes = np.zeros(len(morgans))
        counts = np.zeros(6)        # s, kf, e, ke, ie, if_ (fidelity_objective)
        tracks = log_likelihood = 0.0
        for raw, eligible in raw_batches():
            emissions = structured_emissions(raw, tracks_states)
            (c, s, e, kf, ke, into_error, into_faithful, first_error, first_faithful, ll,
             _, _) = chain_statistics(emissions, tracks_states, capable, theta, entry, exit_, eligible)
            del emissions
            changes += c.sum(axis=0)
            counts += (float(s.sum()), float(kf.sum()), float(e.sum()), float(ke.sum()),
                       float(into_error.sum() + first_error.sum()), float(into_faithful.sum() + first_faithful.sum()))
            log_likelihood += float(ll.sum())
            tracks += 2.0 * float(np.sum(eligible))
        if tracks <= 0.0:
            break
        rate = rate_update(changes, tracks, morgans, n_labels, previous=rate)
        if np.any(capable):
            entry, exit_ = fidelity_update(tuple(counts), (entry, exit_))
        trajectory.append((rate, entry, exit_, log_likelihood))
    step = float(config.estimate_grid)
    return model_for(morgans, n_labels, _round_log(rate, step, *RATE_BOUNDS), _round_log(entry, step, *ENTRY_BOUNDS),
                     _round_log(exit_, step, *EXIT_BOUNDS), site_correlation, trajectory)


# ------------------------------------------------------------------------------------------------------- decoding


def label_paths(raw: np.ndarray, model: ChainModel, capable: np.ndarray, *, reverse: bool) -> np.ndarray:
    """int32[N, 2, B] Viterbi labels of the (tempered) label emissions ``raw`` [N, S*S, B] under ``model``. The mirrored
    convention expands the reversed label emissions directly, so one expanded table (8 S'^2 B bytes per sample) is
    alive at a time (viterbi_track_states would copy the expanded table)."""
    tracks = track_states(capable)
    capable = np.ascontiguousarray(capable, dtype=np.int64)
    if not reverse:
        states = viterbi_track_states(structured_emissions(raw, tracks), tracks, capable, model.theta, model.entry,
                                      model.exit, reverse=False)
    else:
        mirrored = _viterbi_kernel(structured_emissions(np.ascontiguousarray(raw[:, :, ::-1]), tracks), tracks.labels,
                                   tracks.fidelity, tracks.error_of, capable,
                                   initial_log(tracks, capable, model.entry, model.exit),
                                   np.ascontiguousarray(model.theta[::-1]), float(model.entry), float(model.exit),
                                   float(hmm.VITERBI_TIE), True)
        states = mirrored[:, :, ::-1]
    return np.ascontiguousarray(tracks.labels[states].astype(np.int32))


def painted_switches(raw: np.ndarray, model: ChainModel, capable: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """int64[N] painted switches exactly as the painting reports them: the minimum unordered label changes of the two
    traceback conventions' Viterbi paths, the fewer of the two (0 for ineligible samples)."""
    switches = np.minimum(hmm.minimum_switch_counts(label_paths(raw, model, capable, reverse=False)),
                          hmm.minimum_switch_counts(label_paths(raw, model, capable, reverse=True)))
    switches[~np.asarray(eligible, dtype=np.bool_)] = 0
    return switches


def decode_candidates(raw, model: ChainModel, capable, local_keys, named_alleles, observed, bin_starts, bin_stops,
                      n_classes, *, reverse=None):
    """(paths int32[N, 2, B], candidate masks int64[N, 2, B], switch counts int64[N], the mirrored convention's
    paths int32[N, 2, B]) of one batch: painting.hmm.decode_candidates for the chain's Viterbi labels (the run's
    convention, with the mirrored one; switch intervals, combined masks and the canonical track order as there)."""
    if reverse is None:
        reverse = ties.reverse_conventions()
    decoded = []
    for convention in (bool(reverse), not bool(reverse)):
        paths = hmm.parsimonious_track_order(label_paths(raw, model, capable, reverse=convention), local_keys,
                                             n_classes)
        masks = hmm.switch_interval_masks(paths, named_alleles, observed, bin_starts, bin_stops, n_classes)
        decoded.append((paths, masks))
    (paths, primary), (other_paths, secondary) = decoded
    counts = np.minimum(hmm.minimum_switch_counts(paths), hmm.minimum_switch_counts(other_paths))
    masks = hmm.combine_candidate_masks(primary, secondary)
    hmm.apply_track_order(hmm.canonical_track_order(masks, local_keys, n_classes), paths, masks)
    return paths, masks, counts, other_paths


def release_posteriors(raw, model: ChainModel, capable, paths, other_paths, public_ids, local_keys):
    """(qV, qA, qF, qAF) float64[N, B]: posterior masses of the two conventions' Viterbi pairs (public classes, local
    keys), the smaller of the two, all and with both tracks faithful (the release gate is qF, qAF)."""
    tracks = track_states(capable)
    emissions = structured_emissions(raw, tracks)
    *_, pair_mass, faithful_mass = chain_statistics(emissions, tracks, capable, model.theta, model.entry, model.exit,
                                                    np.ones(raw.shape[0], dtype=np.bool_), want_masses=True)
    del emissions
    return masses_from_pairs(pair_mass, faithful_mass, np.left_shift(np.int64(1), paths.astype(np.int64)),
                             np.left_shift(np.int64(1), other_paths.astype(np.int64)),
                             np.ascontiguousarray(public_ids), np.ascontiguousarray(local_keys))
