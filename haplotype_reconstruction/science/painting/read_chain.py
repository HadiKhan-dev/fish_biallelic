"""The painting chain with the read state carried across bins (PaintingParameters.read_classes > 1).

Same model as painting.fidelity's chain (track states, per-track jump transitions, panel-error fidelity), with each
track-pair state carrying a vector over the read classes (science.shared.reads): the track transitions act on every
class component alike (they do not depend on the reads), then the bin multiplies by today's tempered scalar emission,
exp(the read matrix's log scale) and the pair's read matrix (painting.read_factor; the mirror label pair reads the
matrix with the homologues' classes swapped). The read state is a nuisance: the forward-backward sums it exactly;
the Viterbi path maximises over track-pair paths with the read state SUMMED along each path (a joint max over
(pair, read state) fails at low depth): each state keeps its log total and a normalised class vector,
the predecessor is chosen by scoring those retained histories through the destination matrix. Retaining one
history per state is a greedy approximation of the marginal-MAP path. A bounded shortlist further
approximates predecessor selection when the exact scan reaches its work limit."""

from __future__ import annotations

import math

import numpy as np
from numba import prange, types

from ...core.kernels import kernel
from .fidelity import _log_terms, _max_stage

__all__ = ["read_chain_kernel", "read_viterbi_kernel", "track_pair_maps"]  # _block_scores: read_calibration

# Exact argument types of the kernels' call sites (core.kernels audit): the track arrays are TrackStates' read-only
# fields; theta is writable during the EM updates (switch_probabilities) and read-only as a stored ChainModel's.
_I1 = types.Array(types.int64, 1, "C")
_I1R = types.Array(types.int64, 1, "C", readonly=True)
_B1 = types.Array(types.boolean, 1, "C")
_F1 = types.Array(types.float64, 1, "C")
_F1R = types.Array(types.float64, 1, "C", readonly=True)
_F2 = types.Array(types.float64, 2, "C")
_F3 = types.Array(types.float64, 3, "C")
_M5 = types.Array(types.float32, 5, "C")
_READ = (_F3, _M5, _F3, _I1, _B1, _I1, _F2, _I1R, _I1R, _I1R, _I1, _F1)    # emissions ... initial
_OVERLOADS = {
    "_matrix_entry": ((_M5, types.int64, types.int64, types.boolean, _I1, types.int64, types.int64, types.int64),),
    "read_chain_kernel": tuple(
        (*_READ, theta, types.float64, types.float64, _B1, types.boolean, types.boolean, types.boolean, types.boolean)
        for theta in (_F1, _F1R)
    ),
    "_log_transition": (
        (types.int64, types.int64, _I1R, _I1R, _I1, types.float64, types.float64, _F2, _F1, types.boolean),
    ),
    "_unit_read_row": ((_M5, types.int64, _F2),),
    "read_viterbi_kernel": tuple(
        (*_READ, theta, types.float64, types.float64, types.float64, types.boolean) for theta in (_F1, _F1R)
    ),
    "_block_scores": ((_F3, _M5, _F3, _F3, _M5, _F3, _I1, _B1, _I1, _F2, _I1R, _I1R, _I1R, _I1, _F1, _F1,
                       types.float64, types.float64, _F2),),
    "_exp_terms": ((types.float64, types.float64, _F2, _F1, _F2, _F1),),
    "_sum_stage_x": ((_F1, _I1R, _I1R, _I1R, _I1, types.float64, types.float64, _F2, _F1, _F1, _F1),),
    "_transposed_parts_x": ((_F1, _I1R, _I1R, _I1R, _I1, types.float64, types.float64, _F2, _F1, _F2, _F1, _F1, _F1,
                             _F1, _F1, _F1, _F1),),
}


@kernel(overloads=_OVERLOADS["_exp_terms"], cache=True, inline="always", nogil=True)
def _exp_terms(log_stay, log_move, fidelity, stationary, efid, estat):
    """(stay, move) = exp of fidelity._log_terms' log stay and log move, and efid[2, 2] = exp(fidelity), estat[2] =
    exp(stationary): the values fidelity._sum_stage and fidelity._transposed_parts exponentiate at every call, formed
    once per bin (the same exp of the same arguments)."""
    for i in range(2):
        estat[i] = math.exp(stationary[i])
        for j in range(2):
            efid[i, j] = math.exp(fidelity[i, j])
    return math.exp(log_stay), math.exp(log_move)


@kernel(overloads=_OVERLOADS["_sum_stage_x"], cache=True, inline="always", nogil=True)
def _sum_stage_x(values, labels, fids, error_of, capable, stay, move, efid, estat, per_label, out):
    """fidelity._sum_stage with its exponentials formed per bin (_exp_terms) and its per-label work array supplied:
    the same arithmetic in the same order (identical values)."""
    n = len(values)
    total = 0.0
    per_label[:] = 0.0
    for y in range(n):
        total += values[y]
        per_label[labels[y]] += values[y]
    for z in range(n):
        label = labels[z]
        fid = fids[z]
        if capable[label]:
            faithful = values[label]
            error = values[error_of[label]]
            same = stay * (faithful * efid[0, fid] + error * efid[1, fid])
            moved = move * (total - per_label[label]) * estat[fid]
        else:
            same = stay * values[label] if fid == 0 else 0.0
            moved = move * (total - per_label[label]) if fid == 0 else 0.0
        out[z] = same + moved


@kernel(overloads=_OVERLOADS["_transposed_parts_x"], cache=True, inline="always", nogil=True)
def _transposed_parts_x(values, labels, fids, error_of, capable, stay, move, efid, estat, scratch, total, moved,
                        started, ended, kept, moved_error, moved_faithful):
    """fidelity._transposed_parts with its exponentials formed per bin (_exp_terms) and its three per-label work
    arrays supplied (``scratch`` [3, labels]): the same arithmetic in the same order (identical values)."""
    n = len(values)
    weighted = 0.0  # sum_z pi(z) values[z]
    weighted_error = 0.0  # the same over anchored labels' error states
    weighted_faithful = 0.0  # and over anchored labels' faithful states
    scratch[:, :] = 0.0
    for z in range(n):
        label = labels[z]
        if capable[label]:
            value = estat[fids[z]] * values[z]
            if fids[z] == 1:
                weighted_error += value
                scratch[1, label] += value
            else:
                weighted_faithful += value
                scratch[2, label] += value
        else:
            value = values[z] if fids[z] == 0 else 0.0
        weighted += value
        scratch[0, label] += value
    for y in range(n):
        label = labels[y]
        fid = fids[y]
        moved[y] = move * (weighted - scratch[0, label])
        moved_error[y] = move * (weighted_error - scratch[1, label])
        moved_faithful[y] = move * (weighted_faithful - scratch[2, label])
        started[y] = 0.0
        ended[y] = 0.0
        kept[y] = 0.0
        if capable[label]:
            faithful = values[label]
            error = values[error_of[label]]
            keep = stay * efid[fid, fid] * (faithful if fid == 0 else error)
            flip = stay * efid[fid, 1 - fid] * (error if fid == 0 else faithful)
            if fid == 0:
                started[y] = flip
            else:
                ended[y] = flip
            kept[y] = keep
            total[y] = moved[y] + keep + flip
        else:
            total[y] = moved[y] + stay * values[label]


@kernel(overloads=_OVERLOADS["_matrix_entry"], cache=True, inline="always", nogil=True)
def _matrix_entry(read, sample, pair_index, swap, perm, b, i, j):
    if swap:
        return read[sample, pair_index, b, perm[i], perm[j]]
    return read[sample, pair_index, b, i, j]


# Serial evidence (AGENTS 4.6; lab421/lr/cost, 18 Nov): the kernel is parallel over samples only. A sample's
# forward-backward is sequential over bins (each bin's message needs the previous one), so a batch takes as long as
# its slowest sample and the last, partly filled batch idles the rest of the team: painting core use 0.70-0.77 at
# 19 threads on bc8, seed and TroMau (task records; the kernel alone 0.97 on a full 19-sample batch, 0.10 on a
# 2-sample tail). Splitting a sample into forward and backward work items would need the backward messages stored
# and the transposed parts recomputed in a third pass: +30-50 % chain CPU for at most ~0.17 more core use on the
# tail, so it is not done.
@kernel(overloads=_OVERLOADS["read_chain_kernel"], cache=True, parallel=True, nogil=True)
def read_chain_kernel(emissions, read, read_log, pair_unordered, pair_swap, perm, read_init, labels, fids, error_of,
                      capable, initial, theta, a, b, eligible, want_masses, emission_masses, label_masses,
                      bin_likelihood):
    """As fidelity._chain_kernel (same outputs), with the read classes carried. emissions [N, P, B] log scalar bin
    emissions of the track pairs (fidelity.structured_emissions of the tempered emissions); read [N, U, B, C, C] and
    read_log [N, U, B] the read matrices and their log scales per unordered LABEL pair; pair_unordered [P] / pair_swap
    [P] the unordered label pair of each track pair and whether its labels are in descending order; perm [C] the
    homologue swap of the classes; read_init [N, C] the class law at the component's first selected site."""
    n_samples, n_pairs, n_bins = emissions.shape
    n = len(labels)
    n_labels = len(capable)
    n_classes = read.shape[3]
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
    faithful_mass = np.zeros(
        (n_samples if want_masses or emission_masses else 0, n_labels * n_labels * (2 if label_masses else 1), n_bins)
    )
    bin_log_likelihood = np.zeros((n_samples if bin_likelihood else 0, n_bins))
    for sample in prange(n_samples):
        if not eligible[sample]:
            continue
        forward = np.empty((n_bins, n_pairs, n_classes))
        column = np.empty(n)
        out = np.empty(n)
        moved = np.empty(n)
        started = np.empty(n)
        ended = np.empty(n)
        kept = np.empty(n)
        moved_error = np.empty(n)
        moved_faithful = np.empty(n)
        stage = np.empty((n, n))
        pre = np.empty((n_pairs, n_classes))
        logs = np.empty(n_pairs)
        efid = np.empty((2, 2))
        estat = np.empty(2)
        per_label = np.empty(n_labels)
        scratch = np.empty((3, n_labels))
        # bin 0: initial labels, first class law, the bin's matrix
        top = -np.inf
        for p in range(n_pairs):
            u = pair_unordered[p]
            logs[p] = emissions[sample, p, 0] + read_log[sample, u, 0] + initial[p // n] + initial[p % n]
            if logs[p] > top:
                top = logs[p]
        scale = 0.0
        for p in range(n_pairs):
            u = pair_unordered[p]
            sw = pair_swap[p]
            weight = math.exp(logs[p] - top)
            for j in range(n_classes):
                acc = 0.0
                for i in range(n_classes):
                    acc += read_init[sample, i] * _matrix_entry(read, sample, u, sw, perm, 0, i, j)
                forward[0, p, j] = weight * acc
                scale += forward[0, p, j]
        term = top + math.log(scale)
        log_likelihood[sample] += term
        if bin_likelihood:
            bin_log_likelihood[sample, 0] = term
        for p in range(n_pairs):
            for j in range(n_classes):
                forward[0, p, j] /= scale
        for bin_index in range(1, n_bins):
            log_stay, log_move, fidelity, stationary = _log_terms(theta[bin_index - 1], n_labels, a, b)
            stay, move = _exp_terms(log_stay, log_move, fidelity, stationary, efid, estat)
            for c in range(n_classes):
                for y2 in range(n):
                    for y1 in range(n):
                        column[y1] = forward[bin_index - 1, y1 * n + y2, c]
                    _sum_stage_x(column, labels, fids, error_of, capable, stay, move, efid, estat, per_label, out)
                    for z1 in range(n):
                        stage[z1, y2] = out[z1]
                for z1 in range(n):
                    for y2 in range(n):
                        column[y2] = stage[z1, y2]
                    _sum_stage_x(column, labels, fids, error_of, capable, stay, move, efid, estat, per_label, out)
                    for z2 in range(n):
                        pre[z1 * n + z2, c] = out[z2]
            top = -np.inf
            for p in range(n_pairs):
                logs[p] = emissions[sample, p, bin_index] + read_log[sample, pair_unordered[p], bin_index]
                if logs[p] > top:
                    top = logs[p]
            scale = 0.0
            for p in range(n_pairs):
                u = pair_unordered[p]
                sw = pair_swap[p]
                weight = math.exp(logs[p] - top)
                for j in range(n_classes):
                    acc = 0.0
                    for i in range(n_classes):
                        acc += pre[p, i] * _matrix_entry(read, sample, u, sw, perm, bin_index, i, j)
                    forward[bin_index, p, j] = weight * acc
                    scale += forward[bin_index, p, j]
            term = top + math.log(scale)
            log_likelihood[sample] += term
            if bin_likelihood:
                bin_log_likelihood[sample, bin_index] = term
            for p in range(n_pairs):
                for j in range(n_classes):
                    forward[bin_index, p, j] /= scale
        backward = np.ones((n_pairs, n_classes))
        new_backward = np.empty((n_pairs, n_classes))
        weight_c = np.empty((n_pairs, n_classes))
        mass_of = np.empty(n_pairs)
        for bin_index in range(n_bins - 1, -1, -1):
            if bin_index == 0 or want_masses or emission_masses:
                total = 0.0
                for p in range(n_pairs):
                    acc = 0.0
                    for c in range(n_classes):
                        acc += forward[bin_index, p, c] * backward[p, c]
                    mass_of[p] = acc
                    total += acc
                if bin_index == 0:
                    for p in range(n_pairs):
                        mass = mass_of[p] / total
                        for track_state in (p // n, p % n):
                            if capable[labels[track_state]]:
                                if fids[track_state] == 1:
                                    first_error[sample] += mass
                                else:
                                    first_faithful[sample] += mass
                if want_masses or emission_masses:
                    for p in range(n_pairs):
                        mass = mass_of[p] / total
                        first, second = p // n, p % n
                        pair = labels[first] * n_labels + labels[second]
                        if emission_masses:
                            i = n_labels - 1 if fids[first] else labels[first]
                            j = n_labels - 1 if fids[second] else labels[second]
                            faithful_mass[sample, i * n_labels + j, bin_index] += mass
                        elif fids[first] == 0 and fids[second] == 0:
                            faithful_mass[sample, pair, bin_index] += mass
                        if label_masses:
                            faithful_mass[sample, n_labels * n_labels + pair, bin_index] += mass
            if bin_index == 0:
                continue
            edge = bin_index - 1
            log_stay, log_move, fidelity, stationary = _log_terms(theta[edge], n_labels, a, b)
            stay, move = _exp_terms(log_stay, log_move, fidelity, stationary, efid, estat)
            top = -np.inf
            for p in range(n_pairs):
                logs[p] = emissions[sample, p, bin_index] + read_log[sample, pair_unordered[p], bin_index]
                if logs[p] > top:
                    top = logs[p]
            # class-level backward message before bin b's matrix: w_c(p) = e(p) sum_c' M_b(p)[c, c'] beta(p, c')
            for p in range(n_pairs):
                u = pair_unordered[p]
                sw = pair_swap[p]
                e = math.exp(logs[p] - top)
                for c in range(n_classes):
                    acc = 0.0
                    for j in range(n_classes):
                        acc += _matrix_entry(read, sample, u, sw, perm, bin_index, c, j) * backward[p, j]
                    weight_c[p, c] = e * acc
            changes = 0.0
            run_starts = 0.0
            run_ends = 0.0
            keep_faithful = 0.0
            keep_error = 0.0
            enter_error = 0.0
            enter_faithful = 0.0
            for c in range(n_classes):
                # track 1's parts: contract track 2 first
                for z1 in range(n):
                    for z2 in range(n):
                        column[z2] = weight_c[z1 * n + z2, c]
                    _transposed_parts_x(column, labels, fids, error_of, capable, stay, move, efid, estat, scratch, out,
                                        moved, started, ended, kept, moved_error, moved_faithful)
                    for y2 in range(n):
                        stage[z1, y2] = out[y2]
                for y2 in range(n):
                    for z1 in range(n):
                        column[z1] = stage[z1, y2]
                    _transposed_parts_x(column, labels, fids, error_of, capable, stay, move, efid, estat, scratch, out,
                                        moved, started, ended, kept, moved_error, moved_faithful)
                    for y1 in range(n):
                        alpha = forward[edge, y1 * n + y2, c]
                        new_backward[y1 * n + y2, c] = out[y1]
                        changes += alpha * moved[y1]
                        enter_error += alpha * moved_error[y1]
                        enter_faithful += alpha * moved_faithful[y1]
                        run_starts += alpha * started[y1]
                        run_ends += alpha * ended[y1]
                        if fids[y1] == 0:
                            keep_faithful += alpha * kept[y1]
                        else:
                            keep_error += alpha * kept[y1]
                # track 2's parts: contract track 1 first
                for z2 in range(n):
                    for z1 in range(n):
                        column[z1] = weight_c[z1 * n + z2, c]
                    _transposed_parts_x(column, labels, fids, error_of, capable, stay, move, efid, estat, scratch, out,
                                        moved, started, ended, kept, moved_error, moved_faithful)
                    for y1 in range(n):
                        stage[y1, z2] = out[y1]
                for y1 in range(n):
                    for z2 in range(n):
                        column[z2] = stage[y1, z2]
                    _transposed_parts_x(column, labels, fids, error_of, capable, stay, move, efid, estat, scratch, out,
                                        moved, started, ended, kept, moved_error, moved_faithful)
                    for y2 in range(n):
                        alpha = forward[edge, y1 * n + y2, c]
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
                for c in range(n_classes):
                    norm += forward[edge, p, c] * new_backward[p, c]
            label_changes[sample, edge] = changes / norm
            starts[sample] += run_starts / norm
            ends[sample] += run_ends / norm
            kept_faithful[sample] += keep_faithful / norm
            kept_error[sample] += keep_error / norm
            into_error[sample] += enter_error / norm
            into_faithful[sample] += enter_faithful / norm
            scale = 0.0
            for p in range(n_pairs):
                for c in range(n_classes):
                    scale += new_backward[p, c]
            for p in range(n_pairs):
                for c in range(n_classes):
                    backward[p, c] = new_backward[p, c] / scale
    return (label_changes, starts, ends, kept_faithful, kept_error, into_error, into_faithful, first_error,
            first_faithful, log_likelihood, faithful_mass, bin_log_likelihood)


@kernel(overloads=_OVERLOADS["_log_transition"], cache=True, inline="always", nogil=True)
def _log_transition(y, z, labels, fids, capable, log_stay, log_move, fidelity, stationary, transpose):
    """log T(y -> z) of one track (log T(z -> y) with ``transpose``), as fidelity._max_stage scores it."""
    if labels[y] == labels[z]:
        if capable[labels[z]]:
            return log_stay + (fidelity[fids[z], fids[y]] if transpose else fidelity[fids[y], fids[z]])
        return log_stay if (fids[z] == 0 and fids[y] == 0) else -np.inf
    if transpose:
        source = y
        if capable[labels[source]]:
            return log_move + stationary[fids[source]]
        return log_move if fids[source] == 0 else -np.inf
    if capable[labels[z]]:
        return log_move + stationary[fids[z]]
    return log_move if fids[z] == 0 else -np.inf


@kernel(overloads=_OVERLOADS["_unit_read_row"], cache=True, inline="always", nogil=True)
def _unit_read_row(read, sample, read_init):
    """A bypassed row in a mixed batch has only its balanced-to-balanced entry set."""
    if read_init[sample, 0] != 1.0:
        return False
    n_classes = read.shape[3]
    for u in range(read.shape[1]):
        for b in range(read.shape[2]):
            if read[sample, u, b, 0, 0] != 1.0:
                return False
            for i in range(n_classes):
                for j in range(n_classes):
                    if (i != 0 or j != 0) and read[sample, u, b, i, j] != 0.0:
                        return False
    return True


@kernel(overloads=_OVERLOADS["read_viterbi_kernel"], cache=True, parallel=True, nogil=True)
def read_viterbi_kernel(emissions, read, read_log, pair_unordered, pair_swap, perm, read_init, labels, fids, error_of,
                        capable, initial, theta, a, b, tie, reverse):
    """Greedy marginal-MAP search with one class vector per ordered track pair.

    Compare retained parents after the copying transition and the destination's class contraction.
    When exhaustive search exceeds the number of class-conditioned alternatives, keep its visited parents
    and supplement them with same-label parents and class-conditioned alternatives.
    In the mirrored bin order, transpose the transitions and matrices and apply the first-site laws
    at the original first bin. The matrices and their scales are passed in the original bin order.
    Traceback returns int32[N, 2, B] track states in the order of the supplied scalar emissions.
    """
    n_samples, n_pairs, n_bins = emissions.shape
    n = len(labels)
    n_labels = len(capable)
    n_classes = read.shape[3]
    n_unordered = read.shape[1]
    first_bin = n_bins - 1 if reverse else 0
    result = np.empty((n_samples, 2, n_bins), dtype=np.int32)
    preference = np.empty((n, n), dtype=np.int64)
    for dest in range(n):
        preference[dest, 0] = dest
        other = error_of[labels[dest]] if fids[dest] == 0 else labels[dest]
        length = 1
        if other >= 0 and other != dest:
            preference[dest, length] = other
            length += 1
        for rank in range(n):
            source = n - rank - 1 if reverse else rank
            if source != dest and source != other:
                preference[dest, length] = source
                length += 1

    preference_rank = np.empty((n, n), dtype=np.int64)
    for dest in range(n):
        for rank in range(n):
            preference_rank[dest, preference[dest, rank]] = rank

    # Each row's bins are dependent. Independent rows use the caller's Numba team.
    for sample in prange(n_samples):
        scalar = n_classes == 1 or _unit_read_row(read, sample, read_init)
        back = np.empty((n_bins, n_pairs), dtype=np.int32)
        previous = np.empty(n_pairs)
        current = np.empty_like(previous)
        vector = np.empty((n_pairs, n_classes))
        new_vector = np.empty_like(vector)
        for p in range(n_pairs):
            u = pair_unordered[p]
            sw = pair_swap[p]
            total = 0.0
            for j in range(n_classes):
                acc = 0.0
                for i in range(n_classes):
                    if reverse:
                        acc += _matrix_entry(read, sample, u, sw, perm, first_bin, j, i)
                    else:
                        acc += read_init[sample, i] * _matrix_entry(read, sample, u, sw, perm, first_bin, i, j)
                vector[p, j] = acc
                total += acc
            for j in range(n_classes):
                vector[p, j] = vector[p, j] / total if total > 0.0 else 1.0 / n_classes
            previous[p] = emissions[sample, p, 0] + read_log[sample, u, first_bin] + (
                math.log(total) if total > 0.0 else -1e300)
            if not reverse:
                previous[p] += initial[p // n] + initial[p % n]
        previous -= np.max(previous)

        column = np.empty(n)
        adjusted = np.empty(n)
        first_bound = np.empty((n, n))
        first_index = np.empty((n, n), dtype=np.int64)
        out_value = np.empty(n)
        out_index = np.empty(n, dtype=np.int64)
        transition = np.empty((n, n))
        contracted = np.empty((2, n_unordered, n_classes))
        upper = np.empty((2, n_unordered))
        # A fallback may rescore a previously visited source. It includes at most four track rows or
        # columns with a matching founder label, C class leaders per column and scalar leaders in both directions.
        offer_slots = n_pairs if scalar else n_pairs + (n_classes + 6) * n
        candidates = np.empty(offer_slots, dtype=np.int64)
        scores = np.empty(offer_slots)
        class_index = np.empty((0, 0, 0), dtype=np.int64)
        class_column_ready = np.empty(n, dtype=np.bool_)
        row_index = np.empty((0, 0), dtype=np.int64)
        for bin_index in range(1, n_bins):
            read_bin = n_bins - 1 - bin_index if reverse else bin_index
            last = bin_index == n_bins - 1
            log_stay, log_move, fidelity, stationary = _log_terms(theta[bin_index - 1], n_labels, a, b)
            for y2 in range(n):
                for y1 in range(n):
                    column[y1] = previous[y1 * n + y2]
                _max_stage(column, labels, fids, error_of, capable, log_stay, log_move, fidelity, stationary, tie,
                           reverse, reverse, adjusted, out_value, out_index)
                for z1 in range(n):
                    transition[z1, y2] = _log_transition(y2, z1, labels, fids, capable, log_stay, log_move,
                                                         fidelity, stationary, reverse)
                    first_bound[z1, y2] = out_value[z1]
                    first_index[z1, y2] = out_index[z1]
            if scalar:
                # The destination factor is constant over parents. Preserve the scalar two-stage tie rule.
                for z1 in range(n):
                    for y2 in range(n):
                        y1 = first_index[z1, y2]
                        column[y2] = previous[y1 * n + y2] + transition[z1, y1]
                    _max_stage(column, labels, fids, error_of, capable, log_stay, log_move, fidelity, stationary, tie,
                               reverse, reverse, adjusted, out_value, out_index)
                    for z2 in range(n):
                        y2 = out_index[z2]
                        p = z1 * n + z2
                        back[bin_index, p] = first_index[z1, y2] * n + y2
                        u = pair_unordered[p]
                        mass = read[sample, u, read_bin, 0, 0]
                        current[p] = (column[y2] + transition[z2, y2] + emissions[sample, p, bin_index]
                                      + read_log[sample, u, read_bin]
                                      + (math.log(np.float64(mass)) if mass > 0.0 else -1e300))
            else:
                # Contract once per emission pair. Homologue swap just permutes the result, except when the
                # reverse terminal contraction weights the columns by the first site's class law.
                terminal = reverse and last
                for u in range(n_unordered):
                    for sw in range(2 if terminal else 1):
                        largest = 0.0
                        for i in range(n_classes):
                            mass = 0.0
                            for j in range(n_classes):
                                if reverse:
                                    value = _matrix_entry(read, sample, u, sw != 0, perm, read_bin, j, i)
                                    mass += value * read_init[sample, j] if terminal else np.float64(value)
                                else:
                                    mass += np.float64(read[sample, u, read_bin, i, j])
                            contracted[sw, u, i] = mass
                            largest = max(largest, mass)
                        # The dot product is bounded by its largest coefficient. Allow for class-vector rounding.
                        upper[sw, u] = (math.log(largest) if largest > 0.0 else -1e300) + 1e-12
                    if not terminal:
                        upper[1, u] = upper[0, u]
                        for i in range(n_classes):
                            contracted[1, u, i] = contracted[0, u, perm[i]]

                class_indices_ready = False
                for p in range(n_pairs):
                    z1, z2 = p // n, p % n
                    u = pair_unordered[p]
                    sw = int(pair_swap[p])
                    best = -np.inf
                    floor = -np.inf
                    count = 0
                    # Exact search until its visit count reaches the size of the structured alternative.
                    visits_left = (n_classes + 1) * n
                    use_class_candidates = False
                    first_unvisited = n_pairs
                    for rank2 in range(n):
                        y2 = preference[z2, rank2]
                        t2 = transition[z2, y2]
                        if first_bound[z1, y2] + t2 + upper[sw, u] < floor:
                            continue
                        for rank1 in range(n):
                            if visits_left == 0:
                                use_class_candidates = True
                                first_unvisited = rank2 * n + rank1
                                break
                            visits_left -= 1
                            y1 = preference[z1, rank1]
                            source = y1 * n + y2
                            prefix = previous[source] + transition[z1, y1] + t2
                            if prefix == -np.inf or prefix + upper[sw, u] < floor:
                                continue
                            mass = 0.0
                            for c in range(n_classes):
                                mass += vector[source, c] * contracted[sw, u, c]
                            value = prefix + (math.log(mass) if mass > 0.0 else -1e300)
                            if value >= floor:
                                scores[count] = value
                                candidates[count] = source
                                count += 1
                                best = max(best, value)
                                floor = best - tie
                        if use_class_candidates:
                            break
                    if use_class_candidates:
                        # For a fixed second source track, retain the first-source maximizer of the
                        # prefix mass in each class, with the copying transition included in that mass.
                        # The destination's class coefficients and second transition are fixed within
                        # the column. Select using full, destination-aware scores among these leaders,
                        # the scalar leader, all same-label sources and every exact candidate visited.
                        if not class_indices_ready:
                            if class_index.size == 0:
                                class_index = np.empty((n, n, n_classes), dtype=np.int64)
                            if row_index.size == 0:
                                row_index = np.empty((n, n), dtype=np.int64)
                            for y1 in range(n):
                                for y2 in range(n):
                                    column[y2] = previous[y1 * n + y2]
                                _max_stage(column, labels, fids, error_of, capable, log_stay, log_move,
                                           fidelity, stationary, 0.0, reverse, reverse,
                                           adjusted, out_value, out_index)
                                for z in range(n):
                                    row_index[y1, z] = out_index[z]
                            class_column_ready[:] = False
                            class_indices_ready = True
                        # Keep the scalar copying leader for every first source as well.
                        # A pair already considered in the exact pass cannot improve the current floor.
                        for y1 in range(n):
                            y2 = row_index[y1, z2]
                            if y2 < 0:
                                continue
                            rank = preference_rank[z2, y2] * n + preference_rank[z1, y1]
                            if rank < first_unvisited:
                                continue
                            source = y1 * n + y2
                            prefix = previous[source] + transition[z1, y1] + transition[z2, y2]
                            if prefix == -np.inf or prefix + upper[sw, u] < floor:
                                continue
                            mass = 0.0
                            for c in range(n_classes):
                                mass += vector[source, c] * contracted[sw, u, c]
                            value = prefix + (math.log(mass) if mass > 0.0 else -1e300)
                            if value >= floor:
                                scores[count] = value
                                candidates[count] = source
                                count += 1
                                best = max(best, value)
                                floor = best - tie
                        label = labels[z1]
                        other = error_of[label] if fids[z1] == 0 else label
                        for rank2 in range(first_unvisited // n, n):
                            y2 = preference[z2, rank2]
                            t2 = transition[z2, y2]
                            if first_bound[z1, y2] + t2 + upper[sw, u] < floor:
                                continue
                            same_second = labels[y2] == labels[z2]
                            offer_count = n if same_second else n_classes + 3
                            for option in range(offer_count):
                                if same_second:
                                    y1 = preference[z1, option]
                                elif option == 0:
                                    y1 = z1
                                elif option == 1:
                                    y1 = other
                                elif option == 2:
                                    y1 = first_index[z1, y2]
                                else:
                                    if option == 3 and not class_column_ready[y2]:
                                        # Build a column only if a same-label or scalar candidate
                                        # has not already made its upper bound irrelevant.
                                        if first_bound[z1, y2] + t2 + upper[sw, u] < floor:
                                            break
                                        for c in range(n_classes):
                                            for source_y1 in range(n):
                                                source = source_y1 * n + y2
                                                mass = vector[source, c]
                                                column[source_y1] = previous[source] + (
                                                    math.log(mass) if mass > 0.0 else -np.inf)
                                            _max_stage(column, labels, fids, error_of, capable, log_stay, log_move,
                                                       fidelity, stationary, 0.0, reverse, reverse,
                                                       adjusted, out_value, out_index)
                                            for z in range(n):
                                                class_index[z, y2, c] = out_index[z]
                                        class_column_ready[y2] = True
                                    y1 = class_index[z1, y2, option - 3]
                                if y1 < 0 or rank2 * n + preference_rank[z1, y1] < first_unvisited:
                                    continue
                                source = y1 * n + y2
                                prefix = previous[source] + transition[z1, y1] + t2
                                if prefix == -np.inf or prefix + upper[sw, u] < floor:
                                    continue
                                mass = 0.0
                                for c in range(n_classes):
                                    mass += vector[source, c] * contracted[sw, u, c]
                                value = prefix + (math.log(mass) if mass > 0.0 else -1e300)
                                if value >= floor:
                                    scores[count] = value
                                    candidates[count] = source
                                    count += 1
                                    best = max(best, value)
                                    floor = best - tie
                    if not count:
                        # This destination has no reachable parent (e.g. a zero copying transition).
                        current[p] = -np.inf
                        back[bin_index, p] = p
                        continue
                    # Keep the first retained parent in the final tie band, with its own score.
                    if use_class_candidates:
                        first = n_pairs
                        chosen = 0
                        for index in range(count):
                            if scores[index] < floor:
                                continue
                            source = candidates[index]
                            rank = preference_rank[z2, source % n] * n + preference_rank[z1, source // n]
                            if rank < first:
                                first = rank
                                chosen = index
                    else:
                        chosen = 0
                        while scores[chosen] < floor:
                            chosen += 1
                    parent = candidates[chosen]
                    back[bin_index, p] = parent
                    current[p] = scores[chosen] + emissions[sample, p, bin_index] + read_log[sample, u, read_bin]
                    if not last:
                        total = 0.0
                        for j in range(n_classes):
                            acc = 0.0
                            for i in range(n_classes):
                                if reverse:
                                    acc += vector[parent, i] * _matrix_entry(read, sample, u, sw != 0,
                                                                             perm, read_bin, j, i)
                                else:
                                    acc += vector[parent, i] * _matrix_entry(read, sample, u, sw != 0,
                                                                             perm, read_bin, i, j)
                            new_vector[p, j] = acc
                            total += acc
                        for j in range(n_classes):
                            new_vector[p, j] = new_vector[p, j] / total if total > 0.0 else 1.0 / n_classes
            current -= np.max(current)
            previous, current = current, previous
            vector, new_vector = new_vector, vector

        if reverse:
            for p in range(n_pairs):
                previous[p] += initial[p // n] + initial[p % n]
                if n_bins == 1 and not scalar:
                    mass = 0.0
                    for c in range(n_classes):
                        mass += read_init[sample, c] * vector[p, c]
                    previous[p] += math.log(mass) if mass > 0.0 else -1e300
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


def track_pair_maps(track_labels: np.ndarray, emission_label: np.ndarray, n_labels: int):
    """(pair_unordered [P], pair_swap [P]) for the track pairs (t1, t2) = (p // n, p % n): the unordered label pair of
    their EMISSION labels (an error track emits as BACKGROUND, fidelity.structured_emissions) in triu order, and
    whether the first emission label is the larger."""
    n = len(track_labels)
    first, second = np.triu_indices(n_labels)
    index = {(int(x), int(y)): k for k, (x, y) in enumerate(zip(first, second, strict=True))}
    unordered = np.empty(n * n, dtype=np.int64)
    swap = np.empty(n * n, dtype=np.bool_)
    for p in range(n * n):
        e1, e2 = int(emission_label[p // n]), int(emission_label[p % n])
        unordered[p] = index[(min(e1, e2), max(e1, e2))]
        swap[p] = e1 > e2
    return unordered, swap


# the read-aware calibration's hidden-block scores (painting.read_calibration): kept with the chain kernels so the
# painting's keyed kernel closure covers them
@kernel(overloads=_OVERLOADS["_block_scores"], cache=True, parallel=True, nogil=True)
def _block_scores(emissions, read, read_log, hidden_scalar, read_hidden, read_log_hidden, pair_unordered, pair_swap,
                  perm, read_init, labels, fids, error_of, capable, initial, theta, a, b, out):
    """out[N, B] = log P(hidden block of bin | visible data) of the read-state chain (read_chain.read_chain_kernel's
    model): emissions [N, P, B] the visible log scalar bin emissions of the track pairs, read/read_log the visible read
    matrices per unordered label pair, hidden_scalar [N, P, B] scalar hidden log probabilities (samples bypassing the
    read state; zero otherwise), read_hidden/read_log_hidden the read matrices of the visible plus hidden data. With
    pre_b the prior over (track pair, class) of bin b given the visible data before it and beta_b the backward message
    after it: out[b] = log sum pre_b E^H_b beta_b - log sum pre_b E^V_b beta_b."""
    n_samples, n_pairs, n_bins = emissions.shape
    n = len(labels)
    n_labels = len(capable)
    n_classes = read.shape[3]
    for sample in prange(n_samples):
        pre = np.empty((n_bins, n_pairs, n_classes))
        fwd = np.empty((n_pairs, n_classes))
        column = np.empty(n)
        out_stage = np.empty(n)
        moved = np.empty(n)
        started = np.empty(n)
        ended = np.empty(n)
        kept = np.empty(n)
        moved_error = np.empty(n)
        moved_faithful = np.empty(n)
        stage = np.empty((n, n))
        logs = np.empty(n_pairs)
        logs_hidden = np.empty(n_pairs)
        efid = np.empty((2, 2))
        estat = np.empty(2)
        per_label = np.empty(n_labels)
        scratch = np.empty((3, n_labels))
        top = -np.inf
        for p in range(n_pairs):
            logs[p] = initial[p // n] + initial[p % n]
            if logs[p] > top:
                top = logs[p]
        total = 0.0
        for p in range(n_pairs):
            w = np.exp(logs[p] - top)
            for c in range(n_classes):
                pre[0, p, c] = w * read_init[sample, c]
                total += pre[0, p, c]
        for p in range(n_pairs):
            for c in range(n_classes):
                pre[0, p, c] /= total
        for bin_index in range(n_bins):
            if bin_index > 0:
                log_stay, log_move, fidelity, stationary = _log_terms(theta[bin_index - 1], n_labels, a, b)
                stay, move = _exp_terms(log_stay, log_move, fidelity, stationary, efid, estat)
                for c in range(n_classes):
                    for y2 in range(n):
                        for y1 in range(n):
                            column[y1] = fwd[y1 * n + y2, c]
                        _sum_stage_x(column, labels, fids, error_of, capable, stay, move, efid, estat, per_label,
                                     out_stage)
                        for z1 in range(n):
                            stage[z1, y2] = out_stage[z1]
                    for z1 in range(n):
                        for y2 in range(n):
                            column[y2] = stage[z1, y2]
                        _sum_stage_x(column, labels, fids, error_of, capable, stay, move, efid, estat, per_label,
                                     out_stage)
                        for z2 in range(n):
                            pre[bin_index, z1 * n + z2, c] = out_stage[z2]
                total = 0.0
                for p in range(n_pairs):
                    for c in range(n_classes):
                        total += pre[bin_index, p, c]
                for p in range(n_pairs):
                    for c in range(n_classes):
                        pre[bin_index, p, c] /= total
            top = -np.inf
            for p in range(n_pairs):
                logs[p] = emissions[sample, p, bin_index] + read_log[sample, pair_unordered[p], bin_index]
                if logs[p] > top:
                    top = logs[p]
            total = 0.0
            for p in range(n_pairs):
                u = pair_unordered[p]
                sw = pair_swap[p]
                w = np.exp(logs[p] - top)
                for j in range(n_classes):
                    acc = 0.0
                    for i in range(n_classes):
                        acc += pre[bin_index, p, i] * _matrix_entry(read, sample, u, sw, perm, bin_index, i, j)
                    fwd[p, j] = w * acc
                    total += fwd[p, j]
            for p in range(n_pairs):
                for j in range(n_classes):
                    fwd[p, j] /= total
        beta = np.ones((n_pairs, n_classes))
        new_beta = np.empty((n_pairs, n_classes))
        weight_c = np.empty((n_pairs, n_classes))
        for bin_index in range(n_bins - 1, -1, -1):
            top = -np.inf
            top_hidden = -np.inf
            for p in range(n_pairs):
                u = pair_unordered[p]
                logs[p] = emissions[sample, p, bin_index] + read_log[sample, u, bin_index]
                logs_hidden[p] = (emissions[sample, p, bin_index] + hidden_scalar[sample, p, bin_index]
                                  + read_log_hidden[sample, u, bin_index])
                if logs[p] > top:
                    top = logs[p]
                if logs_hidden[p] > top_hidden:
                    top_hidden = logs_hidden[p]
            den = 0.0
            num = 0.0
            for p in range(n_pairs):
                u = pair_unordered[p]
                sw = pair_swap[p]
                ev = np.exp(logs[p] - top)
                eh = np.exp(logs_hidden[p] - top_hidden)
                for i in range(n_classes):
                    av = 0.0
                    ah = 0.0
                    for j in range(n_classes):
                        av += _matrix_entry(read, sample, u, sw, perm, bin_index, i, j) * beta[p, j]
                        ah += _matrix_entry(read_hidden, sample, u, sw, perm, bin_index, i, j) * beta[p, j]
                    den += pre[bin_index, p, i] * ev * av
                    num += pre[bin_index, p, i] * eh * ah
                    if bin_index > 0:
                        weight_c[p, i] = ev * av
            out[sample, bin_index] = (np.log(num) + top_hidden) - (np.log(den) + top)
            if bin_index == 0:
                break
            log_stay, log_move, fidelity, stationary = _log_terms(theta[bin_index - 1], n_labels, a, b)
            stay, move = _exp_terms(log_stay, log_move, fidelity, stationary, efid, estat)
            for c in range(n_classes):
                for z1 in range(n):
                    for z2 in range(n):
                        column[z2] = weight_c[z1 * n + z2, c]
                    _transposed_parts_x(column, labels, fids, error_of, capable, stay, move, efid, estat, scratch,
                                        out_stage, moved, started, ended, kept, moved_error, moved_faithful)
                    for y2 in range(n):
                        stage[z1, y2] = out_stage[y2]
                for y2 in range(n):
                    for z1 in range(n):
                        column[z1] = stage[z1, y2]
                    _transposed_parts_x(column, labels, fids, error_of, capable, stay, move, efid, estat, scratch,
                                        out_stage, moved, started, ended, kept, moved_error, moved_faithful)
                    for y1 in range(n):
                        new_beta[y1 * n + y2, c] = out_stage[y1]
            total = 0.0
            for p in range(n_pairs):
                for c in range(n_classes):
                    total += new_beta[p, c]
            for p in range(n_pairs):
                for c in range(n_classes):
                    beta[p, c] = new_beta[p, c] / total
