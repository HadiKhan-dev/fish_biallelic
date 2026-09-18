"""Conditional founder-path proposals by dual decomposition.

Each sample has a relaxed copy of the focal founder's block choices. Zero-sum
messages enforce consensus by equalizing max-marginals across sample chains.
We decode a single common path and retain the incumbent unless that feasible
path scores better. The caller still checks the unchanged full-site objective
and, for L1-sized moves, the genotype-fit guard.

The dual bound applies only to this restricted, binned conditional problem.
It is neither a global assembly certificate nor a biological confidence score.
"""
import time
import numpy as np
from numba import njit, prange, get_num_threads, set_num_threads
from.import sparse as founder_sparse, dual_short as founder_dual_short
from.packing import emission_arrays, score_rows, packed_emissions, candidate_alphabet, short_models
_COORDINATE_THREADS = 8


@njit(cache=True, parallel=True, nogil=True)
def suffix_bound(emissions, known, choices, offsets, messages, penalty,
                 reverse, first, second, prepared=None, mapping=None, focal_index=0):
    blocks, samples, states = len(emissions), messages.shape[0], len(first)
    result = np.zeros((blocks + 1, samples, states))
    scales = np.zeros(samples)
    for sample in prange(samples):
        for step in range(blocks - 1, -1, -1):
            block = blocks - 1 - step if reverse else step
            begin, end = offsets[block], offsets[block + 1]
            if prepared is None:
                row = founder_sparse.backward_sample(result[step + 1, sample],
                    emissions[block][sample], known[:, block], choices[begin:end],
                    messages[sample, begin:end], penalty, not reverse, first, second)
            else:
                row = founder_sparse.backward_sample(result[step + 1, sample],
                    emissions[block][sample], known[:, block], choices[begin:end],
                    messages[sample, begin:end], penalty, not reverse, first, second,
                    prepared[block], mapping, focal_index, sample)
            normalizer = np.max(row)
            result[step, sample] = row - normalizer
            scales[sample] += normalizer
    return result, scales.sum()


@njit(cache=True, nogil=True)
def coordinate_sweep(emissions, known, incumbent, choices, offsets, messages,
                     suffix, penalty, reverse, first, second, prepared=None, mapping=None, focal_index=0):
    blocks, samples, states = len(emissions), messages.shape[0], len(first)
    prefix = np.zeros((samples, states))
    scales = np.zeros(samples)
    decoded = np.empty(blocks, np.int64)
    predicted_change = 0.0
    normalizers = np.empty(samples)
    for step in range(blocks):
        block = blocks - 1 - step if reverse else step
        start, stop = offsets[block], offsets[block + 1]
        count = stop - start
        local_known = np.ascontiguousarray(known[:, block])
        if prepared is None:
            rows, entries, marginals = founder_sparse.forward_details(prefix,
                emissions[block], local_known, choices[start:stop], penalty,
                reverse, first, second, suffix[step + 1], weights=messages[:, start:stop])
        else:
            rows, entries, marginals = founder_sparse.forward_details(prefix,
                emissions[block], local_known, choices[start:stop], penalty,
                reverse, first, second, suffix[step + 1], prepared[block], mapping, focal_index,
                messages[:, start:stop])
        common = np.zeros(count)
        for local in range(count):
            for sample in range(samples):
                common[local] += marginals[sample, local]
            common[local] /= samples
        best = np.argmax(common)
        for local in range(count):
            if choices[start + local] == incumbent[block] and common[local] == common[best]:
                best = local
        decoded[block] = choices[start + best]
        predicted_change += samples * common[best]
        if prepared is None:
            prefix = founder_sparse.combine_forward(prefix, emissions[block], local_known,
                reverse, first, second, rows, entries, messages[:, start:stop],
                marginals=marginals, common=common, normalizers=normalizers)
        else:
            prefix = founder_sparse.combine_forward(prefix, emissions[block], local_known,
                reverse, first, second, rows, entries, messages[:, start:stop],
                prepared[block], mapping, focal_index, marginals, common, normalizers)
        for sample in range(samples):
            scales[sample] += normalizers[sample]
    return decoded, scales.sum(), predicted_change


@njit(cache=True)
def residual_correction(residual, offsets):
    result = 0.0
    for block in range(len(offsets) - 1):
        result += np.min(residual[offsets[block]:offsets[block + 1]])
    return result


def canonical_score(submodels, known, path, penalty):
    return score_rows(submodels, np.vstack((known, path)), penalty)


def solve(submodels, known, incumbent, penalty, *, branch_cap=16,
          reverse=False, sweeps=20, thread_budget=None, background=None,
          candidate_choices=None):
    """Return a feasible path, its binned score and numerical search diagnostics.

    Alternating forward/reverse sweeps reuse quadratic diploid-state kernels.
    Candidate alleles, missing observations and the uniform switch cost are
    identical to the beam proposal model; only the search strategy differs.
    """
    started = time.monotonic()
    emissions = emission_arrays(submodels)
    short = short_models(submodels)
    packed = packed_emissions(submodels) if short else None
    coordinate_packed = (founder_dual_short.pack_coordinate(submodels, packed)
                         if short else None)
    known = np.ascontiguousarray(known, dtype=np.int64)
    incumbent = np.asarray(incumbent, np.int64)
    first, second = (np.ascontiguousarray(x, np.int64)
                     for x in np.triu_indices(len(known) + 1))
    choices, offsets = (candidate_alphabet(submodels, incumbent, branch_cap)
                        if candidate_choices is None else candidate_choices)
    messages = np.zeros((emissions[0].shape[0], len(choices)))
    best_path = incumbent.copy()
    best_score = canonical_score(submodels, known, best_path, penalty)
    initial_score, history = best_score, []
    # Messages change every sweep, but the feasible score only depends on the
    # decoded path and the fixed evidence/known founders. Repeated decodes are
    # common during convergence; never rescore those identical paths.
    path_scores = {best_path.tobytes(): best_score}
    mapping = None if background is None else background.mapping
    focal_index = 0 if background is None else background.focal
    early_bound = None
    for sweep in range(sweeps):
        if thread_budget is not None:
            set_num_threads(thread_budget())
        backwards = bool(reverse) ^ bool(sweep % 2)
        cached_back = None if background is None else background.get(not backwards, True)
        cached_forward = None if background is None else background.get(backwards, False)
        if short:
            suffix, before = founder_dual_short.dual_suffix(*packed, known, choices, offsets,
                messages, float(penalty), backwards, first, second)
        else:
            suffix, before = suffix_bound(emissions, known, choices, offsets,
                messages, float(penalty), backwards, first, second, cached_back, mapping, focal_index)
        residual = messages.sum(axis=0)
        correction = residual_correction(residual, offsets)
        upper = float(before - correction)
        margin = 2e-8 * max(1.0, abs(upper), abs(best_score))
        if upper + margin <= best_score + 1e-7:
            early_bound = upper
            break
        # Short blocks use sample-wise SIMD within each independent query.
        # Generic blocks synchronize per block: bound that small inner team,
        # while suffix scans retain the dynamically assigned query budget.
        team = get_num_threads()
        try:
            set_num_threads(min(team, _COORDINATE_THREADS))
            if short:
                path, after, change = founder_dual_short.coordinate(*coordinate_packed, known, incumbent,
                    choices, offsets, messages, suffix, float(penalty), backwards, first, second)
            else:
                path, after, change = coordinate_sweep(emissions, known, incumbent,
                    choices, offsets, messages, suffix, float(penalty), backwards, first, second,
                    cached_forward, mapping, focal_index)
        finally:
            set_num_threads(team)
        key = path.tobytes()
        score = path_scores.get(key)
        if score is None:
            score = canonical_score(submodels, known, path, penalty)
            path_scores[key] = score
        if score > best_score:
            best_path, best_score = path.copy(), score
        # Correct a floating residual in the nominal zero-sum constraints.
        residual = messages.sum(axis=0)
        correction = residual_correction(residual, offsets)
        bound = float(after - correction)
        tolerance = 2e-8 * max(1.0, abs(before), abs(after), abs(best_score))
        if after > before + tolerance or abs(after - before - change) > tolerance:
            raise RuntimeError("dual coordinate sweep did not reproduce its predicted descent")
        if best_score > bound + tolerance:
            raise RuntimeError("feasible common founder path exceeds the dual upper bound")
        history.append(dict(sweep=sweep, reverse=backwards, dual_before=float(before),
            dual_after=float(after), corrected_bound=bound, predicted_change=float(change),
            decoded_score=float(score), best_score=float(best_score),
            message_zero_sum_residual=float(np.max(np.abs(residual)))))
        # This certifies only the binned conditional restricted-alphabet fit.
        if bound - best_score <= 1e-7:
            break
    diagnostics = dict(blocks=len(emissions), samples=messages.shape[0],
        founders=len(known) + 1, alphabet_max=int(np.max(np.diff(offsets))),
        initial_score=float(initial_score), best_score=float(best_score),
        changed_rows=int(np.count_nonzero(best_path != incumbent)), history=history,
        early_stop_bound=early_bound, seconds=time.monotonic() - started)
    return best_path, float(best_score), diagnostics
