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
from numba import njit, prange, get_num_threads
from numba.typed import List
from haplotype_reconstruction.assembly.founder_path_search import _advance_row
from haplotype_reconstruction.assembly import panel_search


@njit(cache=True, parallel=True)
def suffix_bound(emissions, known, choices, offsets, messages, penalty,
                 reverse, first, second):
    blocks, samples, states = len(emissions), messages.shape[0], len(first)
    result = np.zeros((blocks + 1, samples, states))
    scales = np.zeros(samples)
    founders = len(known) + 1
    for sample in prange(samples):
        value = np.empty(states)
        row = np.empty(states)
        aggregate = np.empty(states)
        for step in range(blocks - 1, -1, -1):
            block = blocks - 1 - step if reverse else step
            emission = emissions[block]
            aggregate[:] = -np.inf
            for index in range(offsets[block], offsets[block + 1]):
                choice = choices[index]
                row[:] = result[step + 1, sample]
                for offset in range(emission.shape[3]):
                    site = offset if reverse else emission.shape[3] - 1 - offset
                    for state in range(states):
                        a, b = first[state], second[state]
                        aa = known[a, block] if a < founders - 1 else choice
                        bb = known[b, block] if b < founders - 1 else choice
                        value[state] = row[state] + emission[sample, aa, bb, site]
                    switched = np.max(value) - penalty
                    for state in range(states):
                        row[state] = max(value[state], switched)
                for state in range(states):
                    aggregate[state] = max(aggregate[state], row[state] + messages[sample, index])
            normalizer = np.max(aggregate)
            result[step, sample] = aggregate - normalizer
            scales[sample] += normalizer
    return result, scales.sum()


@njit(cache=True, parallel=True)
def coordinate_sweep(emissions, known, incumbent, choices, offsets, messages,
                     suffix, penalty, reverse, first, second):
    blocks, samples, states = len(emissions), messages.shape[0], len(first)
    max_choices = np.max(np.diff(offsets))
    prefix = np.zeros((samples, states))
    scales = np.zeros(samples)
    terminal = np.empty((samples, max_choices, states))
    marginals = np.empty((samples, max_choices))
    common = np.empty(max_choices)
    decoded = np.empty(blocks, np.int64)
    predicted_change = 0.0
    for step in range(blocks):
        block = blocks - 1 - step if reverse else step
        emission = emissions[block]
        start, stop = offsets[block], offsets[block + 1]
        count = stop - start
        local_known = np.ascontiguousarray(known[:, block])
        for sample in prange(samples):
            for local in range(count):
                index = start + local
                _advance_row(prefix[sample], terminal[sample, local], emission,
                             sample, local_known, choices[index], penalty,
                             reverse, first, second)
                marginal = -np.inf
                for state in range(states):
                    terminal[sample, local, state] += messages[sample, index]
                    marginal = max(marginal, terminal[sample, local, state] + suffix[step + 1, sample, state])
                marginals[sample, local] = marginal
            normalizer = np.max(marginals[sample, :count])
            for local in range(count):
                marginals[sample, local] -= normalizer
        for local in range(count):
            total = 0.0
            for sample in range(samples):
                total += marginals[sample, local]
            common[local] = total / samples
        best = np.argmax(common[:count])
        for local in range(count):
            if choices[start + local] == incumbent[block] and common[local] == common[best]:
                best = local
        decoded[block] = choices[start + best]
        predicted_change += samples * common[best]
        for sample in prange(samples):
            prefix[sample, :] = -np.inf
            for local in range(count):
                delta = common[local] - marginals[sample, local]
                messages[sample, start + local] += delta
                for state in range(states):
                    prefix[sample, state] = max(prefix[sample, state], terminal[sample, local, state] + delta)
            normalizer = np.max(prefix[sample])
            prefix[sample] -= normalizer
            scales[sample] += normalizer
    return decoded, scales.sum(), predicted_change


def canonical_score(submodels, known, path, penalty):
    rows = [*known, path]
    keys = [[entry["hap_keys"][row] for entry, row in zip(submodels, values)]
            for values in rows]
    return panel_search.evaluate_panel(keys, submodels, penalty,
        submodels[0]["bin_emissions"].shape[0], num_threads=get_num_threads())


def solve(submodels, known, incumbent, penalty, *, branch_cap=16,
          reverse=False, sweeps=20):
    """Return a feasible path, its binned score and numerical search diagnostics.

    Alternating forward/reverse sweeps reuse quadratic diploid-state kernels.
    Candidate alleles, missing observations and the uniform switch cost are
    identical to the beam proposal model; only the search strategy differs.
    """
    started = time.monotonic()
    emissions = List([entry["bin_emissions"] for entry in submodels])
    known = np.ascontiguousarray(known, dtype=np.int64)
    incumbent = np.asarray(incumbent, np.int64)
    first, second = (np.ascontiguousarray(x, np.int64)
                     for x in np.triu_indices(len(known) + 1))
    alphabet, offsets = [], [0]
    for block, emission in enumerate(emissions):
        choices = np.arange(emission.shape[1], dtype=np.int64)
        if len(choices) > branch_cap:
            rank = emission.max(axis=2).sum(axis=(0, 2))
            choices = np.argsort(-rank, kind="stable")[:branch_cap].astype(np.int64)
            if incumbent[block] not in choices:
                choices[-1] = incumbent[block]
        alphabet.extend(choices)
        offsets.append(len(alphabet))
    choices, offsets = np.asarray(alphabet, np.int64), np.asarray(offsets, np.int64)
    messages = np.zeros((emissions[0].shape[0], len(choices)))
    best_path = incumbent.copy()
    best_score = canonical_score(submodels, known, best_path, penalty)
    initial_score, history = best_score, []
    for sweep in range(sweeps):
        backwards = bool(reverse) ^ bool(sweep % 2)
        suffix, before = suffix_bound(emissions, known, choices, offsets,
            messages, float(penalty), backwards, first, second)
        path, after, change = coordinate_sweep(emissions, known, incumbent,
            choices, offsets, messages, suffix, float(penalty), backwards, first, second)
        score = canonical_score(submodels, known, path, penalty)
        if score > best_score:
            best_path, best_score = path.copy(), score
        # Correct a floating residual in the nominal zero-sum constraints.
        residual = messages.sum(axis=0)
        correction = sum(float(residual[offsets[b]:offsets[b+1]].min())
                         for b in range(len(emissions)))
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
        seconds=time.monotonic() - started)
    return best_path, float(best_score), diagnostics
