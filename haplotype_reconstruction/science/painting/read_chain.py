"""Copying-chain statistics and path search with the read class carried across bins.

Copying and fidelity contract the two track axes separately for each read class. The read operator acts on the
class axis of each destination track pair. Its gauge is restored before comparisons or summation across pairs.
"""

from __future__ import annotations

import math

import numpy as np
from numba import prange, types

from ...core.kernels import kernel
from .fidelity import _log_terms, _max_stage
from .read_operator import TAPE, _D5, apply_bin, empty_tape

__all__ = ["read_chain_kernel", "read_viterbi_kernel", "track_pair_maps"]  # _block_scores: read_calibration

# Exact argument types of the kernels' call sites (core.kernels audit). TrackStates arrays are read-only.
# Read decoding passes a writable contiguous view of theta, including for stored copying models.

_I1 = types.Array(types.int64, 1, "C")
_I1R = types.Array(types.int64, 1, "C", readonly=True)
_B1 = types.Array(types.boolean, 1, "C")
_F1 = types.Array(types.float64, 1, "C")
_F2 = types.Array(types.float64, 2, "C")
_F3 = types.Array(types.float64, 3, "C")
_M5 = types.Array(types.float32, 5, "C")
_READ = (_F3, _I1, _B1, _I1, _F2, _I1R, _I1R, _I1R, _I1, _F1)    # emissions ... initial, without matrix and scale
_OVERLOADS = {
    "_read_chain_kernel": tuple(
        (_READ[0], matrix, _F3, *_READ[1:], _F1, types.float64, types.float64, _B1, types.boolean, types.boolean, types.boolean, types.boolean, TAPE, _F2)
        for matrix in (_M5, _D5)
    ),
    "_log_transition": (
        (types.int64, types.int64, _I1R, _I1R, _I1, types.float64, types.float64, _F2, _F1, types.boolean),
    ),
    "_read_viterbi_kernel": tuple(
        (_READ[0], matrix, _F3, *_READ[1:], _F1, types.float64, types.float64, types.float64, types.int64, TAPE, _F2, types.int64)
        for matrix in (_M5, _D5)
    ),
    "_block_scores_kernel": tuple((_F3, matrix, _F3, _F3, hidden_matrix, _F3, _I1, _B1, _I1, _F2, _I1R, _I1R, _I1R, _I1, _F1, _F1,
                       types.float64, types.float64, _F2, TAPE, _F2, TAPE, _F2)
                                  for matrix, hidden_matrix in ((_M5, _M5), (_M5, _D5), (_D5, _M5), (_D5, _D5))),
    "_exp_terms": ((types.float64, types.float64, _F2, _F1, _F2, _F1),),
    "_sum_stage_x": ((_F1, _I1R, _I1R, _I1R, _I1, types.float64, types.float64, _F2, _F1, _F1, _F1),),
    "_transposed_sum_x": ((_F1, _I1R, _I1R, _I1R, _I1, types.float64, types.float64, _F2, _F1, _F1, _F1),),
    "_transposed_parts_x": ((_F1, _I1R, _I1R, _I1R, _I1, types.float64, types.float64, _F2, _F1, _F2, _F1, _F1, _F1,
                             _F1, _F1, _F1, _F1),),
    # Every class at once (flat row views with offsets and strides): _read_chain_kernel's copying contractions.
    "_sum_stage_classes": ((_F1, types.int64, types.int64, _F1, types.int64, _I1R, _I1R, _I1R, _I1, types.float64,
                            types.float64, _F2, _F1, _F1, _F2),),
    "_transposed_sum_classes": ((_F1, types.int64, types.int64, _F1, types.int64, _I1R, _I1R, _I1R, _I1,
                                 types.float64, types.float64, _F2, _F1, _F1, _F2),),
    "_transposed_parts_classes": ((_F1, types.int64, types.int64, _F1, _F1, types.int64, _I1R, _I1R, _I1R, _I1,
                                   types.float64, types.float64, _F2, _F1, _F2, _F3),),
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


@kernel(overloads=_OVERLOADS["_transposed_sum_x"], cache=True, inline="always", nogil=True)
def _transposed_sum_x(values, labels, fids, error_of, capable, stay, move, efid, estat, per_label, out):
    """Adjoint copying contraction when only the total message is consumed."""
    weighted = 0.0
    per_label[:] = 0.0
    for z in range(len(values)):
        label, fid = labels[z], fids[z]
        value = estat[fid] * values[z] if capable[label] else values[z]
        weighted += value
        per_label[label] += value
    for y in range(len(values)):
        label, fid = labels[y], fids[y]
        moved = move * (weighted - per_label[label])
        if capable[label]:
            faithful, error = values[label], values[error_of[label]]
            keep = stay * efid[fid, fid] * (faithful if fid == 0 else error)
            flip = stay * efid[fid, 1 - fid] * (error if fid == 0 else faithful)
            out[y] = moved + keep + flip
        else:
            out[y] = moved + stay * values[label]


@kernel(overloads=_OVERLOADS["_sum_stage_classes"], cache=True, nogil=True)
def _sum_stage_classes(values, offset, stride, out, out_offset, labels, fids, error_of, capable, stay, move, efid,
                       estat, total, per_label):
    """_sum_stage_x for every class at once. Row y of the input is values[offset + y * stride:][:C], row z of the
    output out[out_offset + z * C:][:C] (C = len(total) classes). Each class column gets _sum_stage_x's arithmetic in
    its order (identical values); the classes are the innermost, contiguous axis."""
    n = len(labels)
    classes = len(total)
    for c in range(classes):
        total[c] = 0.0
    per_label[:, :] = 0.0
    for y in range(n):
        base = offset + y * stride
        label = labels[y]
        for c in range(classes):
            value = values[base + c]
            total[c] += value
            per_label[label, c] += value
    for z in range(n):
        label = labels[z]
        fid = fids[z]
        row = offset + label * stride
        target = out_offset + z * classes
        if capable[label]:
            error_row = offset + error_of[label] * stride
            for c in range(classes):
                same = stay * (values[row + c] * efid[0, fid] + values[error_row + c] * efid[1, fid])
                moved = move * (total[c] - per_label[label, c]) * estat[fid]
                out[target + c] = same + moved
        elif fid == 0:
            for c in range(classes):
                out[target + c] = stay * values[row + c] + move * (total[c] - per_label[label, c])
        else:
            for c in range(classes):
                out[target + c] = 0.0


@kernel(overloads=_OVERLOADS["_transposed_sum_classes"], cache=True, nogil=True)
def _transposed_sum_classes(values, offset, stride, out, out_offset, labels, fids, error_of, capable, stay, move,
                            efid, estat, weighted, per_label):
    """_transposed_sum_x for every class at once (rows as _sum_stage_classes): the same arithmetic in the same order
    for each class."""
    n = len(labels)
    classes = len(weighted)
    for c in range(classes):
        weighted[c] = 0.0
    per_label[:, :] = 0.0
    for z in range(n):
        label, fid = labels[z], fids[z]
        base = offset + z * stride
        if capable[label]:
            factor = estat[fid]
            for c in range(classes):
                value = factor * values[base + c]
                weighted[c] += value
                per_label[label, c] += value
        else:
            for c in range(classes):
                value = values[base + c]
                weighted[c] += value
                per_label[label, c] += value
    for y in range(n):
        label, fid = labels[y], fids[y]
        row = offset + label * stride
        target = out_offset + y * classes
        if capable[label]:
            error_row = offset + error_of[label] * stride
            keep_factor = stay * efid[fid, fid]
            flip_factor = stay * efid[fid, 1 - fid]
            for c in range(classes):
                moved = move * (weighted[c] - per_label[label, c])
                faithful, error = values[row + c], values[error_row + c]
                keep = keep_factor * (faithful if fid == 0 else error)
                flip = flip_factor * (error if fid == 0 else faithful)
                out[target + c] = moved + keep + flip
        else:
            for c in range(classes):
                moved = move * (weighted[c] - per_label[label, c])
                out[target + c] = moved + stay * values[row + c]


@kernel(overloads=_OVERLOADS["_transposed_parts_classes"], cache=True, nogil=True)
def _transposed_parts_classes(values, offset, stride, total, parts, slot, labels, fids, error_of, capable, stay,
                              move, efid, estat, sums, scratch):
    """_transposed_parts_x for every class at once: total[y * C + c] its total message and parts[slot + (q * C + c) *
    n + y] its parts q = moved, moved_error, moved_faithful, started, ended, kept, each class's values as
    _transposed_parts_x computes them (sums [3, C] and scratch [3, labels, C] are its accumulators)."""
    n = len(labels)
    classes = sums.shape[1]
    sums[:, :] = 0.0
    scratch[:, :, :] = 0.0
    for z in range(n):
        label = labels[z]
        base = offset + z * stride
        if capable[label]:
            factor = estat[fids[z]]
            part = 1 if fids[z] == 1 else 2
            for c in range(classes):
                value = factor * values[base + c]
                sums[part, c] += value
                scratch[part, label, c] += value
                sums[0, c] += value
                scratch[0, label, c] += value
        elif fids[z] == 0:
            for c in range(classes):
                value = values[base + c]
                sums[0, c] += value
                scratch[0, label, c] += value
        else:
            for c in range(classes):
                sums[0, c] += 0.0
                scratch[0, label, c] += 0.0
    plane = classes * n
    for y in range(n):
        label = labels[y]
        fid = fids[y]
        row = offset + label * stride
        for c in range(classes):
            cell = slot + c * n + y
            parts[cell] = move * (sums[0, c] - scratch[0, label, c])
            parts[cell + plane] = move * (sums[1, c] - scratch[1, label, c])
            parts[cell + 2 * plane] = move * (sums[2, c] - scratch[2, label, c])
            parts[cell + 3 * plane] = 0.0
            parts[cell + 4 * plane] = 0.0
            parts[cell + 5 * plane] = 0.0
        if capable[label]:
            error_row = offset + error_of[label] * stride
            keep_factor = stay * efid[fid, fid]
            flip_factor = stay * efid[fid, 1 - fid]
            for c in range(classes):
                cell = slot + c * n + y
                faithful = values[row + c]
                error = values[error_row + c]
                keep = keep_factor * (faithful if fid == 0 else error)
                flip = flip_factor * (error if fid == 0 else faithful)
                if fid == 0:
                    parts[cell + 3 * plane] = flip
                else:
                    parts[cell + 4 * plane] = flip
                parts[cell + 5 * plane] = keep
                total[y * classes + c] = parts[cell] + keep + flip
        else:
            for c in range(classes):
                total[y * classes + c] = parts[slot + c * n + y] + stay * values[row + c]


# Bins in one direction share a recurrent message across track pairs. A wider retained batch allows independent
# sample recurrences to fill the team without per-bin barriers or an additional traversal of the history.
def read_chain_kernel(*args, tape=None, exponent=None):
    if tape is None:
        tape, exponent = empty_tape(), np.empty((0, 0))
    return _read_chain_kernel(*args, tape, exponent)


@kernel(overloads=_OVERLOADS["_read_chain_kernel"], cache=True, parallel=True, nogil=True)
def _read_chain_kernel(emissions, read, read_log, pair_unordered, pair_swap, perm, read_init, labels, fids, error_of,
                      capable, initial, theta, a, b, eligible, want_masses, emission_masses, label_masses,
                      bin_likelihood, tape, exponent):
    """As fidelity._chain_kernel (same outputs), with the read classes carried. emissions [N, P, B] log scalar bin
    emissions of the track pairs (fidelity.structured_emissions of the tempered emissions); read [N, U, B, C, C] and
    read_log [N, U, B] the read matrices and their log scales per unordered LABEL pair; pair_unordered [P] / pair_swap
    [P] the unordered label pair of each track pair and whether its labels are in descending order; perm [C] the
    homologue swap of the classes; read_init [N, C] the class law at the component's first selected site."""
    n_samples, n_pairs, n_bins = emissions.shape
    n = len(labels)
    n_labels = len(capable)
    n_classes = read_init.shape[1]
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
    # All samples and both sweeps use the same copying coefficients for an edge.
    _, _, fidelity, stationary = _log_terms(0.5, n_labels, a, b)
    efid, estat = np.empty((2, 2)), np.empty(2)
    _exp_terms(0.0, 0.0, fidelity, stationary, efid, estat)
    stays, moves = np.empty(len(theta)), np.empty(len(theta))
    log_labels = math.log(n_labels - 1.0) if n_labels > 1 else 0.0
    for edge in prange(len(theta)):
        stays[edge] = math.exp(math.log1p(-theta[edge]))
        moves[edge] = math.exp(math.log(theta[edge]) - log_labels) if n_labels > 1 else 0.0
    for sample in prange(n_samples):
        if not eligible[sample]:
            continue
        forward = np.empty((n_bins, n_pairs, n_classes))
        pre = np.empty((n_pairs, n_classes))
        logs = np.empty(n_pairs)
        transfer_work = np.empty((4, n_classes))
        # The copying contractions treat every class at once on flat row views (_sum_stage_classes and the
        # transposed pair below): each class's arithmetic and its order are the per-class loops'.
        forward_flat = forward.reshape(forward.size)
        pre_flat = pre.reshape(pre.size)
        stage_flat = np.empty(n * n * n_classes)
        class_total = np.empty(n_classes)
        class_label = np.empty((n_labels, n_classes))
        class_sums = np.empty((3, n_classes))
        class_scratch = np.empty((3, n_labels, n_classes))
        class_out = np.empty(n * n_classes)
        first_parts = np.empty(n * 6 * n_classes * n)
        second_parts = np.empty(n * 6 * n_classes * n)
        # Bin 0 applies the initial class law. Restore each pair's log gauge before summing.
        top = -np.inf
        for p in range(n_pairs):
            u = pair_unordered[p]
            gauge = apply_bin(read, tape, exponent, sample, u, 0, pair_swap[p], perm, read_init[sample],
                              forward[0, p], transfer_work, False)
            logs[p] = emissions[sample, p, 0] + read_log[sample, u, 0] + initial[p // n] + initial[p % n] + gauge
            top = max(top, logs[p])
        scale = 0.0
        for p in range(n_pairs):
            weight = math.exp(logs[p] - top)
            for j in range(n_classes):
                forward[0, p, j] *= weight
                scale += forward[0, p, j]
        term = top + math.log(scale)
        log_likelihood[sample] += term
        if bin_likelihood:
            bin_log_likelihood[sample, 0] = term
        for p in range(n_pairs):
            for j in range(n_classes):
                forward[0, p, j] /= scale
        for bin_index in range(1, n_bins):
            stay, move = stays[bin_index - 1], moves[bin_index - 1]
            previous = (bin_index - 1) * n_pairs * n_classes
            # Track 1 first: stage row (y2, z1) from the column of forward pairs (y1, y2); then track 2: pre row
            # (z1, z2) from the stage rows (y2, z1).
            for y2 in range(n):
                _sum_stage_classes(forward_flat, previous + y2 * n_classes, n * n_classes, stage_flat,
                                   y2 * n * n_classes, labels, fids, error_of, capable, stay, move, efid, estat,
                                   class_total, class_label)
            for z1 in range(n):
                _sum_stage_classes(stage_flat, z1 * n_classes, n * n_classes, pre_flat, z1 * n * n_classes, labels,
                                   fids, error_of, capable, stay, move, efid, estat, class_total, class_label)
            top = -np.inf
            for p in range(n_pairs):
                u = pair_unordered[p]
                gauge = apply_bin(read, tape, exponent, sample, u, bin_index, pair_swap[p], perm, pre[p],
                                  forward[bin_index, p], transfer_work, False)
                logs[p] = emissions[sample, p, bin_index] + read_log[sample, u, bin_index] + gauge
                top = max(top, logs[p])
            scale = 0.0
            for p in range(n_pairs):
                weight = math.exp(logs[p] - top)
                for j in range(n_classes):
                    forward[bin_index, p, j] *= weight
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
            stay, move = stays[edge], moves[edge]
            top = -np.inf
            for p in range(n_pairs):
                u = pair_unordered[p]
                gauge = apply_bin(read, tape, exponent, sample, u, bin_index, pair_swap[p], perm, backward[p],
                                  weight_c[p], transfer_work, True)
                logs[p] = emissions[sample, p, bin_index] + read_log[sample, u, bin_index] + gauge
                top = max(top, logs[p])
            for p in range(n_pairs):
                e = math.exp(logs[p] - top)
                for c in range(n_classes):
                    weight_c[p, c] *= e
            changes = 0.0
            run_starts = 0.0
            run_ends = 0.0
            keep_faithful = 0.0
            keep_error = 0.0
            enter_error = 0.0
            enter_faithful = 0.0
            # Both tracks' parts for every class at once, then the statistics added in the per-class loops' order
            # (class; its track 1 parts by (y2, y1); its track 2 parts by (y1, y2)), so every sum is the same.
            weight_flat = weight_c.reshape(weight_c.size)
            parts_size = 6 * n_classes * n
            plane = n_classes * n
            for z1 in range(n):
                # track 1's parts: contract track 2 first (stage row (z1, y2))
                _transposed_sum_classes(weight_flat, z1 * n * n_classes, n_classes, stage_flat, z1 * n * n_classes,
                                        labels, fids, error_of, capable, stay, move, efid, estat, class_total,
                                        class_label)
            for y2 in range(n):
                _transposed_parts_classes(stage_flat, y2 * n_classes, n * n_classes, class_out, first_parts,
                                          y2 * parts_size, labels, fids, error_of, capable, stay, move, efid, estat,
                                          class_sums, class_scratch)
                for y1 in range(n):
                    for c in range(n_classes):
                        new_backward[y1 * n + y2, c] = class_out[y1 * n_classes + c]
            for z2 in range(n):
                # track 2's parts: contract track 1 first (stage row (z2, y1))
                _transposed_sum_classes(weight_flat, z2 * n_classes, n * n_classes, stage_flat, z2 * n * n_classes,
                                        labels, fids, error_of, capable, stay, move, efid, estat, class_total,
                                        class_label)
            for y1 in range(n):
                _transposed_parts_classes(stage_flat, y1 * n_classes, n * n_classes, class_out, second_parts,
                                          y1 * parts_size, labels, fids, error_of, capable, stay, move, efid, estat,
                                          class_sums, class_scratch)
            for c in range(n_classes):
                for y2 in range(n):
                    cell = y2 * parts_size + c * n
                    for y1 in range(n):
                        alpha = forward[edge, y1 * n + y2, c]
                        changes += alpha * first_parts[cell + y1]
                        enter_error += alpha * first_parts[cell + plane + y1]
                        enter_faithful += alpha * first_parts[cell + 2 * plane + y1]
                        run_starts += alpha * first_parts[cell + 3 * plane + y1]
                        run_ends += alpha * first_parts[cell + 4 * plane + y1]
                        if fids[y1] == 0:
                            keep_faithful += alpha * first_parts[cell + 5 * plane + y1]
                        else:
                            keep_error += alpha * first_parts[cell + 5 * plane + y1]
                for y1 in range(n):
                    cell = y1 * parts_size + c * n
                    for y2 in range(n):
                        alpha = forward[edge, y1 * n + y2, c]
                        changes += alpha * second_parts[cell + y2]
                        enter_error += alpha * second_parts[cell + plane + y2]
                        enter_faithful += alpha * second_parts[cell + 2 * plane + y2]
                        run_starts += alpha * second_parts[cell + 3 * plane + y2]
                        run_ends += alpha * second_parts[cell + 4 * plane + y2]
                        if fids[y2] == 0:
                            keep_faithful += alpha * second_parts[cell + 5 * plane + y2]
                        else:
                            keep_error += alpha * second_parts[cell + 5 * plane + y2]
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


def read_viterbi_kernel(*args, tape=None, exponent=None, own_width=None, both=False, bypass=None):
    if tape is None:
        tape, exponent = empty_tape(), np.empty((0, 0))
    if own_width is None:
        own_width = tape[7].shape[2] if tape[0].size else args[6].shape[1]
    direction = 2 if both else int(args[-1])
    if bypass is not None and not tape[0].size:
        # Compiled actions still need the tape's row metadata for decoding.
        tape = (*tape[:10], bypass, tape[11])
    paths = _read_viterbi_kernel(*args[:-1], direction, tape, exponent, own_width)
    return paths if both else paths[0]


@kernel(overloads=((types.float64,),), cache=True, inline="always", nogil=True)
def _log_read_mass(mass):
    return math.log(mass) if mass > 0.0 else -1e300


@kernel(overloads=((_F1, _F1),), cache=True, inline="always", nogil=True)
def _dot(a, b):
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total


@kernel(overloads=((_F1,),), cache=True, inline="always", nogil=True)
def _normalize_history(row):
    total = 0.0
    for c in range(len(row)):
        total += row[c]
    for c in range(len(row)):
        row[c] = row[c] / total if total > 0.0 else 1.0 / len(row)
    return total


@kernel(overloads=_OVERLOADS["_read_viterbi_kernel"], cache=True, parallel=True, nogil=True)
def _read_viterbi_kernel(emissions, read, read_log, pair_unordered, pair_swap, perm, read_init, labels, fids,
                         error_of, capable, initial, theta, a, b, tie, direction, tape, exponent, own_width):
    """Build independent edge/pair transition rows, then search over sample and direction in one team."""
    samples, pairs, bins = emissions.shape
    n = len(labels)
    n_labels = len(capable)
    directions = 2 if direction == 2 else 1
    result = np.empty((directions, samples, 2, bins), dtype=np.int32)
    preferences = np.empty((directions, n, n), dtype=np.int64)
    rank_tables = np.empty_like(preferences)
    for d in range(directions):
        reverse = bool(d if directions == 2 else direction)
        preference, ranks = preferences[d], rank_tables[d]
        for z in range(n):
            other = error_of[labels[z]] if fids[z] == 0 else labels[z]
            preference[z, 0] = z
            count = 1
            if other >= 0 and other != z:
                preference[z, count] = other
                count += 1
            for rank in range(n):
                y = n - 1 - rank if reverse else rank
                if y != z and y != other:
                    preference[z, count] = y
                    count += 1
            for rank in range(n):
                ranks[z, preference[z, rank]] = rank

    # The transition row is shared by every sample, the forward search, and the suffix splice.
    _, _, fidelity, stationary = _log_terms(0.5, n_labels, a, b)
    edges = len(theta)
    log_stays, log_moves = np.empty(edges), np.empty(edges)
    log_labels = math.log(n_labels - 1.0) if n_labels > 1 else 0.0
    for edge in prange(edges):
        log_stays[edge] = math.log1p(-theta[edge])
        log_moves[edge] = math.log(theta[edge]) - log_labels if n_labels > 1 else -np.inf
    transitions = np.empty((directions, edges, n, n))
    for task in prange(directions * edges * n):
        d, edge, z = task // (edges * n), (task // n) % edges, task % n
        reverse = bool(d if directions == 2 else direction)
        for y in range(n):
            transitions[d, edge, z, y] = _log_transition(y, z, labels, fids, capable, log_stays[edge],
                                                         log_moves[edge], fidelity, stationary, reverse)

    bypass = tape[10]
    for task in prange(samples * directions):
        d, sample = task // samples, task % samples
        reverse = bool(d if directions == 2 else direction)
        original_order = directions == 2 and reverse
        # A bypassed row's action and entry law are supported only on the fixed class zero.
        # Decode it with the scalar tie policy even when the batch stores a wider class axis.
        bypassed = bypass.size > 0 and bypass[sample]
        width = 1 if bypassed else read_init.shape[1]
        own = 1 if bypassed else own_width
        preference, ranks = preferences[d], rank_tables[d]
        first_bin = bins - 1 if reverse else 0
        first_emission = first_bin if original_order else 0
        previous = np.empty(pairs)
        current = np.empty(pairs)
        vector = np.empty((pairs, width))
        new_vector = np.empty_like(vector)
        work = np.empty((4, width))
        boundary = np.ones(width)
        entry = read_init[sample, :width]
        if not reverse:
            boundary[:] = entry
        for p in range(pairs):
            u = pair_unordered[p]
            log_scale = apply_bin(read, tape, exponent, sample, u, first_bin, pair_swap[p], perm, boundary,
                                  vector[p], work, reverse)
            mass = _normalize_history(vector[p])
            previous[p] = emissions[sample, p, first_emission] + read_log[sample, u, first_bin] + log_scale + _log_read_mass(mass)
            if not reverse:
                previous[p] += initial[p // n] + initial[p % n]
            if reverse and bins == 1:
                previous[p] += _log_read_mass(_dot(vector[p], entry))

        # At width one the suffix cannot affect the predecessor. Larger states retain every prefix needed by
        # the splice, and a separate support bit keeps the finite search sentinel out of its score.
        history = np.empty((bins - 1 if width > 1 else 0, pairs, width))
        prefixes = np.empty((len(history), pairs))
        live = np.empty(pairs, dtype=np.bool_)
        new_live = np.empty_like(live)
        for p in range(pairs):
            live[p] = previous[p] > -1e299
        parents = np.empty((bins, pairs), dtype=np.int32)
        column = np.empty(n)
        adjusted = np.empty(n)
        best_values = np.empty(n)
        best_sources = np.empty(n, dtype=np.int64)
        first = np.empty((n, n))
        first_source = np.empty((n, n), dtype=np.int64)
        scalar_columns = np.empty((n, n))
        leaders = np.empty((width + 3, pairs), dtype=np.int64)
        peaks = np.empty((width + 3, pairs))
        feature = np.empty((width + 3, pairs))
        candidates = np.empty(own + 6, dtype=np.int64)
        scores = np.empty(own + 6)
        h = np.empty(width)
        boundary[:] = 1.0

        for step in range(1, bins):
            previous -= np.max(previous)
            if width > 1:
                history[step - 1] = vector
                for p in range(pairs):
                    prefixes[step - 1, p] = previous[p] if live[p] else -np.inf
            read_bin = bins - 1 - step if reverse else step
            last = step == bins - 1
            if reverse and last:
                boundary[:] = entry
            edge = bins - 1 - step if original_order else step - 1
            log_stay, log_move = log_stays[edge], log_moves[edge]
            transition = transitions[d, edge]

            for p in range(pairs):
                own_mass, foreign_mass = 0.0, 0.0
                for c in range(own):
                    own_mass += vector[p, c]
                for c in range(own, width):
                    foreign_mass += vector[p, c]
                feature[0, p] = previous[p]
                feature[1, p] = previous[p] + (math.log(own_mass) if own_mass > 0.0 else -np.inf)
                feature[2, p] = previous[p] + (math.log(foreign_mass) if foreign_mass > 0.0 else -np.inf)
                for c in range(width):
                    mass = vector[p, c]
                    feature[c + 3, p] = previous[p] + (math.log(mass) if mass > 0.0 else -np.inf)

            # Two linear one-track maxima for each feature. Keep the full source pair attaining each maximum.
            for f in range(width + 3 if width > 1 else 1):
                for y2 in range(n):
                    for y1 in range(n):
                        column[y1] = feature[f, y1 * n + y2]
                    _max_stage(column, labels, fids, error_of, capable, log_stay, log_move, fidelity, stationary,
                               0.0, reverse, reverse, adjusted, best_values, best_sources)
                    for z1 in range(n):
                        first[z1, y2] = best_values[z1]
                        first_source[z1, y2] = best_sources[z1]
                        if f == 0:
                            scalar_columns[z1, y2] = best_values[z1]
                for z1 in range(n):
                    _max_stage(first[z1], labels, fids, error_of, capable, log_stay, log_move, fidelity, stationary,
                               0.0, reverse, reverse, adjusted, best_values, best_sources)
                    for z2 in range(n):
                        p = z1 * n + z2
                        y2 = best_sources[z2]
                        peaks[f, p] = best_values[z2]
                        leaders[f, p] = first_source[z1, y2] * n + y2 if y2 >= 0 else -1

            for p in range(pairs):
                z1, z2 = p // n, p % n
                u = pair_unordered[p]
                gauge = apply_bin(read, tape, exponent, sample, u, read_bin, pair_swap[p], perm, boundary, h,
                                  work, not reverse)
                scalar = p
                floor = peaks[0, p] - tie
                if math.isfinite(floor):
                    for j in range(n):
                        y2 = preference[z2, j]
                        if scalar_columns[z1, y2] + transition[z2, y2] >= floor:
                            for i in range(n):
                                y1 = preference[z1, i]
                                y = y1 * n + y2
                                if previous[y] + (transition[z1, y1] + transition[z2, y2]) >= floor:
                                    scalar = y
                                    break
                            break

                candidates[0], candidates[1] = p, scalar
                count = 2
                if width > 1:
                    for f in range(own + 3):
                        y = leaders[f, p]
                        if y >= 0:
                            candidates[count] = y
                            count += 1
                    if own < width:
                        minimum = np.min(h[own:])
                        best = -np.inf
                        donor = -1
                        for c in range(own, width):
                            coeff = h[c] - minimum
                            if coeff > 0.0:
                                value = math.log(coeff) + peaks[c + 3, p]
                                if value > best:
                                    best, donor = value, leaders[c + 3, p]
                        if donor >= 0:
                            candidates[count] = donor
                            count += 1

                top = -np.inf
                for i in range(count):
                    y = candidates[i]
                    mass = _dot(vector[y], h)
                    scores[i] = (previous[y] + transition[z1, y // n] + transition[z2, y % n]
                                 + _log_read_mass(mass))
                    top = max(top, scores[i])
                chosen, best_rank = 0, pairs
                for i in range(count):
                    y = candidates[i]
                    rank = ranks[z2, y % n] * n + ranks[z1, y // n]
                    if scores[i] >= top - tie and rank < best_rank:
                        chosen, best_rank = i, rank
                parent = candidates[chosen]
                parents[step, p] = parent
                current[p] = scores[chosen] + gauge + emissions[sample, p, read_bin if original_order else step] + read_log[sample, u, read_bin]
                new_live[p] = live[parent] and _dot(vector[parent], h) > 0.0
                if not last:
                    apply_bin(read, tape, exponent, sample, u, read_bin, pair_swap[p], perm, vector[parent],
                              new_vector[p], work, reverse)
                    _normalize_history(new_vector[p])
            previous, current = current, previous
            vector, new_vector = new_vector, vector
            live, new_live = new_live, live

        if reverse:
            for p in range(pairs):
                previous[p] += initial[p // n] + initial[p % n]
        top = np.max(previous)
        state = 0
        for rank in range(pairs):
            p = pairs - 1 - rank if reverse else rank
            if previous[p] >= top - tie:
                state = p
                break

        suffix = entry.copy() if reverse else np.ones(width)
        splice_scores = np.empty(pairs)
        for step in range(bins - 1, -1, -1):
            z1, z2 = state // n, state % n
            result[d, sample, 0, step] = z1
            result[d, sample, 1, step] = z2
            if step == 0:
                break
            incumbent = parents[step, state]
            if width > 1:
                u = pair_unordered[state]
                read_bin = bins - 1 - step if reverse else step
                apply_bin(read, tape, exponent, sample, u, read_bin, pair_swap[state], perm, suffix, h, work,
                          not reverse)
                largest = np.max(h)
                if largest > 0.0:
                    suffix[:] = h / largest
                else:
                    suffix[:] = 0.0
                edge = bins - 1 - step if original_order else step - 1
                transition = transitions[d, edge]
                peak = -np.inf
                for y in range(pairs):
                    mass = _dot(history[step - 1, y], suffix)
                    score = -np.inf
                    if mass > 0.0:
                        score = (prefixes[step - 1, y] + transition[z1, y // n] + transition[z2, y % n]
                                 + math.log(mass))
                    splice_scores[y] = score
                    peak = max(peak, score)
                if splice_scores[incumbent] < peak - tie:
                    rank = 0
                    while rank < pairs:
                        y = preference[z1, rank % n] * n + preference[z2, rank // n]
                        if splice_scores[y] >= peak - tie:
                            incumbent = y
                            break
                        rank += 1
            state = incumbent
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
def _block_scores(*args, tape=None, exponent=None, hidden_tape=None, hidden_exponent=None):
    if tape is None:
        tape, exponent = empty_tape(), np.empty((0, 0))
    if hidden_tape is None:
        hidden_tape, hidden_exponent = empty_tape(), np.empty((0, 0))
    return _block_scores_kernel(*args, tape, exponent, hidden_tape, hidden_exponent)


@kernel(overloads=_OVERLOADS["_block_scores_kernel"], cache=True, parallel=True, nogil=True)
def _block_scores_kernel(emissions, read, read_log, hidden_scalar, read_hidden, read_log_hidden,
                         pair_unordered, pair_swap, perm, read_init, labels, fids, error_of, capable,
                         initial, theta, a, b, out, tape, exponent, hidden_tape, hidden_exponent):
    """out[N, B] = log P(hidden block of bin | visible data) of the read-state chain (read_chain.read_chain_kernel's
    model): emissions [N, P, B] the visible log scalar bin emissions of the track pairs, read/read_log the visible read
    actions and their log scales per unordered label pair, hidden_scalar [N, P, B] scalar hidden log probabilities
    for samples bypassing the read state, read_hidden/read_log_hidden the actions and scales with hidden data. With
    pre_b the prior over (track pair, class) of bin b given the visible data before it and beta_b the backward message
    after it: out[b] = log sum pre_b E^H_b beta_b - log sum pre_b E^V_b beta_b."""
    n_samples, n_pairs, n_bins = emissions.shape
    n = len(labels)
    n_labels = len(capable)
    n_classes = read_init.shape[1]
    for sample in prange(n_samples):
        pre = np.empty((n_bins, n_pairs, n_classes))
        fwd = np.empty((n_pairs, n_classes))
        transfer_work = np.empty((4, n_classes))
        hidden_vector = np.empty(n_classes)
        column = np.empty(n)
        out_stage = np.empty(n)
        stage = np.empty((n, n))
        logs = np.empty(n_pairs)
        logs_hidden = np.empty(n_pairs)
        efid = np.empty((2, 2))
        estat = np.empty(2)
        per_label = np.empty(n_labels)
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
                u = pair_unordered[p]
                gauge = apply_bin(read, tape, exponent, sample, u, bin_index, pair_swap[p], perm,
                                  pre[bin_index, p], fwd[p], transfer_work, False)
                logs[p] = emissions[sample, p, bin_index] + read_log[sample, u, bin_index] + gauge
                top = max(top, logs[p])
            total = 0.0
            for p in range(n_pairs):
                w = np.exp(logs[p] - top)
                for j in range(n_classes):
                    fwd[p, j] *= w
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
                sw = pair_swap[p]
                gauge = apply_bin(read, tape, exponent, sample, u, bin_index, sw, perm, beta[p], weight_c[p],
                                  transfer_work, True)
                hidden_gauge = apply_bin(read_hidden, hidden_tape, hidden_exponent, sample, u, bin_index, sw,
                                         perm, beta[p], hidden_vector, transfer_work, True)
                # The hidden action is used only in the local contraction. The adjoint advances on visible data.
                hidden_mass = 0.0
                for i in range(n_classes):
                    hidden_mass += pre[bin_index, p, i] * hidden_vector[i]
                logs[p] = emissions[sample, p, bin_index] + read_log[sample, u, bin_index] + gauge
                log_mass = math.log(hidden_mass) if hidden_mass > 0.0 else -np.inf
                logs_hidden[p] = (emissions[sample, p, bin_index] + hidden_scalar[sample, p, bin_index]
                                  + read_log_hidden[sample, u, bin_index] + hidden_gauge + log_mass)
                top = max(top, logs[p])
                top_hidden = max(top_hidden, logs_hidden[p])
            den = 0.0
            num = 0.0
            for p in range(n_pairs):
                ev = np.exp(logs[p] - top)
                if top_hidden > -np.inf:
                    num += np.exp(logs_hidden[p] - top_hidden)
                for i in range(n_classes):
                    weight_c[p, i] *= ev
                    den += pre[bin_index, p, i] * weight_c[p, i]
            out[sample, bin_index] = (np.log(num) + top_hidden) - (np.log(den) + top)
            if bin_index == 0:
                break
            log_stay, log_move, fidelity, stationary = _log_terms(theta[bin_index - 1], n_labels, a, b)
            stay, move = _exp_terms(log_stay, log_move, fidelity, stationary, efid, estat)
            for c in range(n_classes):
                for z1 in range(n):
                    for z2 in range(n):
                        column[z2] = weight_c[z1 * n + z2, c]
                    _transposed_sum_x(column, labels, fids, error_of, capable, stay, move, efid, estat,
                                      per_label, out_stage)
                    for y2 in range(n):
                        stage[z1, y2] = out_stage[y2]
                for y2 in range(n):
                    for z1 in range(n):
                        column[z1] = stage[z1, y2]
                    _transposed_sum_x(column, labels, fids, error_of, capable, stay, move, efid, estat,
                                      per_label, out_stage)
                    for y1 in range(n):
                        new_beta[y1 * n + y2, c] = out_stage[y1]
            total = 0.0
            for p in range(n_pairs):
                for c in range(n_classes):
                    total += new_beta[p, c]
            for p in range(n_pairs):
                for c in range(n_classes):
                    beta[p, c] = new_beta[p, c] / total
