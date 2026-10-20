"""Kernels of the fixed-assignment leave-one-sample-out cavity score (see cavity.py).

At a fixed site, held-out samples with equal complete tied-state memberships and likelihood triples
have equal held-out potentials, support and starts: each such case is solved once (exact equality,
no quantisation). Every multi-state membership is compared in full before grouping. Further exact reuse: a later mean-field start that reaches a state of the converged
first start replays that start's remaining sweeps; a predictive value equal to a floored evidence
value reuses that value's logarithm; the site's complete-data potentials are filled once and a
held-out solve patches every weighted unary and edge of its sample (restored afterwards); sigmoid and
entropy terms are memoised per founder. Every sample accumulates its own predictions in site and
founder order and samples are reduced in sample order, so results do not depend on threads.

Release 2.3: the three mean-field starts of a held-out solve are compared by training ELBO with
its unchanged ``nll`` quantum. All distinct winning converged endpoints (all starts when none
converged) form equivalence classes at the existing mean-field convergence tolerance, after
exact deduplication. Classes contribute equally to each state's predictive likelihood; their
numerically distinct members are averaged internally. The floor and log follow both averages.
Exact identity alone made a near-duplicate mode's weight depend on GL noise. The retained scalar _solve_cavity_site wrapper still returns the old directional representative
for reference tests; production predictions use the full endpoint family. Release 2.1 chose the
first winning start in the fixed order (q = H, q = 1 - H, q = 1/2). 2.0 compared with an
absolute 1e-12 band and broke ties by the endpoints' values, so rounding- or perturbation-level ELBO
differences between different endpoints (e.g. mirror-image modes) chose the endpoint. A founder
update's logit within the same quantum of zero is zero (q = 1/2 exactly): symmetric evidence (equal
REF and ALT reads) makes such logits exactly zero, and an exactly symmetric start then stays on its
symmetric fixed point; GL noise broke that symmetry and let the start escape to another mode
(TroMau chr18:375: a non-selected representative's score moved by 3.9 nats under 1e-12 noise).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from .residual_births import _accurate_sum

# Workspace budget of one site tile, not a statistical parameter: one site is the minimum, so
# memory is O(N K^2) per tile, never O(N L K^2).
PREDICTION_TILE_BYTES = 8 * 1024 * 1024


@njit(cache=True)
def _stable_sigmoid(logit: float) -> float:
    # exp(-37) < 2**-53, so 1 + exp(-logit) already rounds to exactly one:
    # returning 1.0 skips the exponential without changing any result.
    if logit >= 37.0:
        return 1.0
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


@njit(cache=True)
def _site_workspace(k, n_edges, max_iter):
    q = np.empty(k, dtype=np.float64)
    candidate_q = np.empty((3, k), dtype=np.float64)
    candidate_elbo = np.empty(3, dtype=np.float64)
    candidate_converged = np.empty(3, dtype=np.uint8)
    cavity_unary = np.empty((k, 2), dtype=np.float64)
    cavity_unary_logit = np.empty(k, dtype=np.float64)
    directed_difference_zero = np.empty((k, k), dtype=np.float64)
    directed_difference_one = np.empty((k, k), dtype=np.float64)
    cavity_edge_potential = np.empty((n_edges, 4), dtype=np.float64)
    remaining_support_by_founder = np.empty(k, dtype=np.int64)
    supported_founders = np.empty(k, dtype=np.int64)
    trajectory = np.empty((max_iter + 1, k), dtype=np.float64)
    # Per-founder memos of pure functions, shared by the site's held-out
    # solves. Sigmoid: row 0 keeps the last first-start logit and its value,
    # row 1 the last later-start one. Entropy terms: keyed by the endpoint
    # probability. NaN keys never compare equal, so empty entries miss.
    memo_logit = np.full((2, k), np.nan, dtype=np.float64)
    memo_sigmoid = np.empty((2, k), dtype=np.float64)
    memo_probability = np.full(k, np.nan, dtype=np.float64)
    memo_q_log_q = np.empty(k, dtype=np.float64)
    memo_complement_log_complement = np.empty(k, dtype=np.float64)
    patched_unaries = np.empty(k, dtype=np.int64)
    saved_unaries = np.empty((k, 3), dtype=np.float64)
    patched_edges = np.empty(n_edges, dtype=np.int64)
    saved_edges = np.empty((n_edges, 8), dtype=np.float64)
    return (
        q,
        candidate_q,
        candidate_elbo,
        candidate_converged,
        cavity_unary,
        cavity_unary_logit,
        directed_difference_zero,
        directed_difference_one,
        cavity_edge_potential,
        remaining_support_by_founder,
        supported_founders,
        trajectory,
        memo_logit,
        memo_sigmoid,
        memo_probability,
        memo_q_log_q,
        memo_complement_log_complement,
        patched_unaries, saved_unaries, patched_edges, saved_edges,
    )


@njit(cache=True)
def _set_site_potentials(site, k, unary, pair, edge_first, edge_second, work):
    """Fill the site's complete-data potentials, shared by its held-outs.

    Each held-out solve patches all unaries and edges to which its sample
    contributes and restores their saved complete-data values afterwards.
    """
    cavity_unary = work[4]
    cavity_unary_logit = work[5]
    directed_difference_zero = work[6]
    directed_difference_one = work[7]
    cavity_edge_potential = work[8]
    for founder in range(k):
        u0 = unary[founder, site, 0]
        u1 = unary[founder, site, 1]
        cavity_unary[founder, 0] = u0
        cavity_unary[founder, 1] = u1
        cavity_unary_logit[founder] = u1 - u0
    for edge_index in range(pair.shape[0]):
        first_founder = edge_first[edge_index]
        second_founder = edge_second[edge_index]
        v00 = pair[edge_index, site, 0]
        v01 = pair[edge_index, site, 1]
        v10 = pair[edge_index, site, 2]
        v11 = pair[edge_index, site, 3]
        cavity_edge_potential[edge_index, 0] = v00
        cavity_edge_potential[edge_index, 1] = v01
        cavity_edge_potential[edge_index, 2] = v10
        cavity_edge_potential[edge_index, 3] = v11
        directed_difference_zero[first_founder, second_founder] = v10 - v00
        directed_difference_one[first_founder, second_founder] = v11 - v01
        directed_difference_zero[second_founder, first_founder] = v01 - v00
        directed_difference_one[second_founder, first_founder] = v11 - v10


@njit(cache=True)
def _solve_cavity_site(
        heldout, site, k, log_likelihood, rw_log_likelihood, haplotypes,
        assignments, founder_support, unary, pair, edge_first, edge_second,
        neighbours, neighbour_counts, mean_field_max_iter,
        mean_field_tolerance, work, edge_lookup, elbo_tie, reverse,
        member_offsets, member_states, state_first, state_second, sample_support):
    (
        q,
        candidate_q,
        candidate_elbo,
        candidate_converged,
        cavity_unary,
        cavity_unary_logit,
        directed_difference_zero,
        directed_difference_one,
        cavity_edge_potential,
        remaining_support_by_founder,
        supported_founders,
        trajectory,
        memo_logit,
        memo_sigmoid,
        memo_probability,
        memo_q_log_q,
        memo_complement_log_complement,
        patched_unaries, saved_unaries, patched_edges, saved_edges,
    ) = work
    wildcard = k
    n_edges = pair.shape[0]
    heldout_not_converged = 0
    n_supported = 0
    for founder in range(k):
        remaining_support = founder_support[founder] - sample_support[heldout, founder]
        remaining_support_by_founder[founder] = remaining_support
        if remaining_support > 0:
            supported_founders[n_supported] = founder
            n_supported += 1

    # Every state in the winning category has the same unknown-copy count.
    # Its homozygote/RW unary or heterozygote edge is unique within that set.
    # Save all affected factors before subtraction; restoring saved values
    # avoids roundoff accumulating across successive held-out samples.
    begin = member_offsets[heldout]
    end = member_offsets[heldout + 1]
    share = 1.0 / (end - begin)
    n_patched_unaries = 0
    n_patched_edges = 0
    for member in range(begin, end):
        state = member_states[member]
        a = state_first[state]
        b = state_second[state]
        if a == wildcard:
            continue
        if a == b or b == wildcard:
            index = n_patched_unaries
            n_patched_unaries += 1
            patched_unaries[index] = a
            saved_unaries[index, 0] = cavity_unary[a, 0]
            saved_unaries[index, 1] = cavity_unary[a, 1]
            saved_unaries[index, 2] = cavity_unary_logit[a]
            if a == b:
                u0 = cavity_unary[a, 0] - share * log_likelihood[heldout, site, 0]
                u1 = cavity_unary[a, 1] - share * log_likelihood[heldout, site, 2]
            else:
                u0 = cavity_unary[a, 0] - share * rw_log_likelihood[heldout, site, 0]
                u1 = cavity_unary[a, 1] - share * rw_log_likelihood[heldout, site, 1]
            cavity_unary[a, 0] = u0
            cavity_unary[a, 1] = u1
            cavity_unary_logit[a] = u1 - u0
        else:
            edge = edge_lookup[a, b]
            index = n_patched_edges
            n_patched_edges += 1
            patched_edges[index] = edge
            for value in range(4):
                saved_edges[index, value] = cavity_edge_potential[edge, value]
            saved_edges[index, 4] = directed_difference_zero[a, b]
            saved_edges[index, 5] = directed_difference_one[a, b]
            saved_edges[index, 6] = directed_difference_zero[b, a]
            saved_edges[index, 7] = directed_difference_one[b, a]
            v00 = cavity_edge_potential[edge, 0] - share * log_likelihood[heldout, site, 0]
            v01 = cavity_edge_potential[edge, 1] - share * log_likelihood[heldout, site, 1]
            v10 = cavity_edge_potential[edge, 2] - share * log_likelihood[heldout, site, 1]
            v11 = cavity_edge_potential[edge, 3] - share * log_likelihood[heldout, site, 2]
            cavity_edge_potential[edge, 0] = v00
            cavity_edge_potential[edge, 1] = v01
            cavity_edge_potential[edge, 2] = v10
            cavity_edge_potential[edge, 3] = v11
            directed_difference_zero[a, b] = v10 - v00
            directed_difference_one[a, b] = v11 - v01
            directed_difference_zero[b, a] = v01 - v00
            directed_difference_one[b, a] = v11 - v10

    candidate_converged[:] = 0
    # Sweeps used by the first start; with ``trajectory`` (its state after
    # every sweep) this lets later starts reuse its exact continuation.
    first_length = 0
    for start_index in range(3):
        for founder in range(k):
            if (
                remaining_support_by_founder[founder] == 0
                or start_index == 2
            ):
                q[founder] = 0.5
            elif start_index == 0:
                q[founder] = float(haplotypes[founder, site])
            else:
                q[founder] = 1.0 - float(
                    haplotypes[founder, site]
                )

        converged = False
        joined = False
        used_iterations = 0
        if start_index == 0:
            for founder in range(k):
                trajectory[0, founder] = q[founder]
        for iteration in range(mean_field_max_iter):
            maximum_change = 0.0
            for supported_index in range(n_supported):
                founder = supported_founders[supported_index]
                # An isolated founder reaches its exact unary update on
                # the first sweep; subsequent sweeps would repeat the
                # same sigmoid and contribute an exact zero change.
                if neighbour_counts[founder] == 0 and iteration > 0:
                    continue
                logit = cavity_unary_logit[founder]

                for neighbour_index in range(neighbour_counts[founder]):
                    other = neighbours[founder, neighbour_index]
                    qo = q[other]
                    logit += (1.0 - qo) * (
                        directed_difference_zero[founder, other]
                    )
                    logit += qo * (
                        directed_difference_one[founder, other]
                    )

                # Release 2.1: an exact zero by symmetry stays exactly zero.
                if logit <= elbo_tie and logit >= -elbo_tie:
                    logit = 0.0
                # The sigmoid is a pure function of the logit: an equal
                # logit reuses the earlier value bit for bit.
                if logit == memo_logit[0, founder]:
                    updated = memo_sigmoid[0, founder]
                elif logit == memo_logit[1, founder]:
                    updated = memo_sigmoid[1, founder]
                else:
                    updated = _stable_sigmoid(logit)
                    memo_row = 0 if start_index == 0 else 1
                    memo_logit[memo_row, founder] = logit
                    memo_sigmoid[memo_row, founder] = updated
                change = abs(updated - q[founder])
                if change > maximum_change:
                    maximum_change = change
                q[founder] = updated
            used_iterations = iteration + 1
            if start_index == 0:
                for founder in range(k):
                    trajectory[used_iterations, founder] = q[founder]
            if maximum_change <= mean_field_tolerance:
                converged = True
                break
            # A sweep is a deterministic function of the state alone (an
            # isolated founder held at its exact unary value contributes a
            # zero change whether or not it is revisited). A later start that
            # reaches state s of the converged first start therefore repeats
            # that start's remaining sweeps, convergence test and endpoint bit
            # for bit; count those sweeps instead of recomputing them. A
            # continuation that would pass the sweep limit is not taken.
            if start_index > 0 and candidate_converged[0] == 1:
                for state in range(first_length):
                    if (used_iterations + first_length - state
                            > mean_field_max_iter):
                        continue
                    same_state = True
                    for founder in range(k):
                        if q[founder] != trajectory[state, founder]:
                            same_state = False
                            break
                    if same_state:
                        used_iterations += first_length - state
                        converged = True
                        joined = True
                        break
                if joined:
                    break
        if start_index == 0:
            first_length = used_iterations
        if converged:
            candidate_converged[start_index] = 1
        if start_index > 0:
            same_endpoint = joined
            if not joined:
                same_endpoint = True
                for founder in range(k):
                    if q[founder] != candidate_q[0, founder]:
                        same_endpoint = False
                        break
            if same_endpoint:
                # The ELBO is a function of the endpoint alone, so an equal
                # endpoint reuses it exactly.
                candidate_elbo[start_index] = candidate_elbo[0]
                for founder in range(k):
                    candidate_q[start_index, founder] = candidate_q[0, founder]
                continue

        elbo = 0.0
        for founder in range(k):
            u0 = cavity_unary[founder, 0]
            u1 = cavity_unary[founder, 1]
            probability = q[founder]
            elbo += (1.0 - probability) * u0 + probability * u1
            q_log_q = 0.0
            one_minus_q_log_one_minus_q = 0.0
            if probability > 0.0 and probability < 1.0:
                # Both terms are pure functions of the probability.
                if probability == memo_probability[founder]:
                    q_log_q = memo_q_log_q[founder]
                    one_minus_q_log_one_minus_q = (
                        memo_complement_log_complement[founder]
                    )
                else:
                    q_log_q = probability * math.log(probability)
                    complement = 1.0 - probability
                    # log(1) is exactly zero: a complement that rounds to one
                    # contributes the same zero term without the logarithm.
                    if complement != 1.0:
                        one_minus_q_log_one_minus_q = (
                            complement * math.log(complement)
                        )
                    memo_probability[founder] = probability
                    memo_q_log_q[founder] = q_log_q
                    memo_complement_log_complement[founder] = (
                        one_minus_q_log_one_minus_q
                    )
                elbo -= q_log_q
                elbo -= one_minus_q_log_one_minus_q
        for edge_index in range(n_edges):
            first_founder = edge_first[edge_index]
            second_founder = edge_second[edge_index]
            v00 = cavity_edge_potential[edge_index, 0]
            v01 = cavity_edge_potential[edge_index, 1]
            v10 = cavity_edge_potential[edge_index, 2]
            v11 = cavity_edge_potential[edge_index, 3]
            q_first = q[first_founder]
            q_second = q[second_founder]
            elbo += (1.0 - q_first) * (1.0 - q_second) * v00
            elbo += (1.0 - q_first) * q_second * v01
            elbo += q_first * (1.0 - q_second) * v10
            elbo += q_first * q_second * v11
        candidate_elbo[start_index] = elbo
        candidate_q[start_index,:] = q

    # The best ELBO among the converged starts (all starts when none converged), then the first
    # start in the fixed order (H, 1 - H, 1/2; reversed conventions: the last) whose ELBO is within elbo_tie of it.
    any_converged = False
    for start_index in range(3):
        if candidate_converged[start_index] == 1:
            any_converged = True
    if not any_converged:
        heldout_not_converged += 1
    best_elbo = -math.inf
    for start_index in range(3):
        if (not any_converged or candidate_converged[start_index] == 1) \
                and candidate_elbo[start_index] > best_elbo:
            best_elbo = candidate_elbo[start_index]
    best_start = 0
    for position in range(3):
        start_index = 2 - position if reverse else position
        if (not any_converged or candidate_converged[start_index] == 1) \
                and candidate_elbo[start_index] >= best_elbo - elbo_tie:
            best_start = start_index
            break
    q[:] = candidate_q[best_start,:]
    # Restore the complete-data potentials for the site's next held-out.
    for index in range(n_patched_unaries):
        founder = patched_unaries[index]
        cavity_unary[founder, 0] = saved_unaries[index, 0]
        cavity_unary[founder, 1] = saved_unaries[index, 1]
        cavity_unary_logit[founder] = saved_unaries[index, 2]
    for index in range(n_patched_edges):
        edge = patched_edges[index]
        a, b = edge_first[edge], edge_second[edge]
        for value in range(4):
            cavity_edge_potential[edge, value] = saved_edges[index, value]
        directed_difference_zero[a, b] = saved_edges[index, 4]
        directed_difference_one[a, b] = saved_edges[index, 5]
        directed_difference_zero[b, a] = saved_edges[index, 6]
        directed_difference_one[b, a] = saved_edges[index, 7]

    return q, heldout_not_converged


@njit(cache=True)
def _eligible_endpoint_indices(endpoints, elbo, converged, elbo_tie):
    """Distinct winning training endpoints, with the existing convergence/ELBO rules.

    Exact duplicates carry one share regardless of how many starts reached them.
    This uses no held-out evidence. The returned order is immaterial to the mixture.
    """
    any_converged = False
    for index in range(len(elbo)):
        if converged[index] == 1:
            any_converged = True
    best = -math.inf
    for index in range(len(elbo)):
        if (not any_converged or converged[index] == 1) and elbo[index] > best:
            best = elbo[index]
    indices = np.empty(len(elbo), dtype=np.int64)
    count = 0
    for index in range(len(elbo)):
        if (any_converged and converged[index] != 1) or elbo[index] < best - elbo_tie:
            continue
        duplicate = False
        for previous in range(count):
            equal = True
            for founder in range(endpoints.shape[1]):
                if endpoints[index, founder] != endpoints[indices[previous], founder]:
                    equal = False
                    break
            if equal:
                duplicate = True
                break
        if not duplicate:
            indices[count] = index
            count += 1
    return indices[:count]


@njit(cache=True)
def _endpoint_class_labels(endpoints, indices, equivalence_scale=0.0):
    """Connected components at the stated endpoint equivalence scale.

    The helper's zero default exposes the exact-identity reference. Production
    uses the solver's existing convergence tolerance: tiny endpoint differences
    otherwise change mode multiplicity under GL noise. This scale is never chosen
    from held-out evidence or endpoint predictions. Connected components avoid
    any dependence on the order in which close endpoints are encountered.
    """
    count = len(indices)
    labels = np.arange(count, dtype=np.int64)
    for first in range(count):
        for second in range(first + 1, count):
            same = True
            for founder in range(endpoints.shape[1]):
                if abs(endpoints[indices[first], founder] -
                       endpoints[indices[second], founder]) > equivalence_scale:
                    same = False
                    break
            if same:
                old = labels[second]
                new = labels[first]
                for index in range(count):
                    if labels[index] == old:
                        labels[index] = new
    return labels


@njit(cache=True)
def _predictive_family_log_states(endpoints, indices, g0, g1, g2,
                                  likelihood_floor, ww_log_emission,
                                  equivalence_scale=0.0):
    """Uniform training-endpoint mixture of state likelihoods, then floor and log.

    Founder marginals are never averaged: heterozygote states contain products.
    Endpoint-equivalence classes, when requested, have equal weight; exact-distinct
    vectors within a class have equal weight. Expansion sums make both reductions
    independent of start order. The WW state is endpoint-independent.
    """
    k = endpoints.shape[1]
    n_states = (k + 1) * (k + 2) // 2
    result = np.empty(n_states, dtype=np.float64)
    count = len(indices)
    labels = _endpoint_class_labels(endpoints, indices, equivalence_scale)
    values = np.empty(count, dtype=np.float64)
    class_values = np.empty(count, dtype=np.float64)
    partials = np.empty(count, dtype=np.float64)
    # State geometry is the unchanged RR upper triangle, RW, WW.
    state = 0
    for first in range(k):
        for second in range(first, k + 1):
            if second == k:
                # RW states follow all RR states in the output.
                output_state = k * (k + 1) // 2 + first
            else:
                output_state = state
                state += 1
            classes = 0
            for root in range(count):
                if labels[root] != root:
                    continue
                members = 0
                for index in range(count):
                    if labels[index] != root:
                        continue
                    qi = endpoints[indices[index], first]
                    if second == k:
                        predictive = 0.5 * (1.0 - qi) * g0 + 0.5 * g1 + 0.5 * qi * g2
                    elif first == second:
                        predictive = (1.0 - qi) * g0 + qi * g2
                    else:
                        qj = endpoints[indices[index], second]
                        p0 = (1.0 - qi) * (1.0 - qj)
                        p2 = qi * qj
                        p1 = 1.0 - p0 - p2
                        predictive = p0 * g0 + p1 * g1 + p2 * g2
                    values[members] = predictive
                    members += 1
                class_values[classes] = _accurate_sum(values[:members], partials) / members
                classes += 1
            predictive = _accurate_sum(class_values[:classes], partials) / classes
            if predictive < likelihood_floor:
                predictive = likelihood_floor
            result[output_state] = math.log(predictive)
    result[-1] = ww_log_emission
    return result


@njit(cache=True)
def _membership_groups(offsets, members):
    """Exact labels of complete membership lists; candidate keys are verified in full.

    Length and endpoint states only choose the short collision chain. They never
    establish equality, and no ordering or statistical tie tolerance is used.
    """
    n = len(offsets) - 1
    labels = np.empty(n, dtype=np.int64)
    previous = np.full(n, -1, dtype=np.int64)
    heads = {}
    for sample in range(n):
        begin, end = offsets[sample], offsets[sample + 1]
        key = (end - begin, members[begin], members[end - 1])
        head = -1
        if key in heads:
            head = heads[key]
        candidate = head
        found = -1
        while candidate >= 0:
            other = offsets[candidate]
            equal = True
            for index in range(end - begin):
                if members[begin + index] != members[other + index]:
                    equal = False
                    break
            if equal:
                found = candidate
                break
            candidate = previous[candidate]
        if found < 0:
            previous[sample] = head
            heads[key] = sample
            labels[sample] = sample
        else:
            labels[sample] = found
    return labels


@njit(cache=True, parallel=True)
def grouped_cavity_predictions(
        likelihood, log_likelihood, rw_log_likelihood, ww_log_emission,
        haplotypes, assignments, founder_support, unary, pair, edge_first,
        edge_second, edge_lookup, neighbours, neighbour_counts,
        mean_field_max_iter, mean_field_tolerance, likelihood_floor,
        site_start, site_stop, shared_triple, elbo_tie, reverse,
        member_offsets, member_states, state_first, state_second, sample_support,
        endpoint_counts=None, endpoint_equivalence_scale=None):
    """Solve and predict exact cases once in a bounded, site-parallel tile.

    ``edge_lookup[a, b]`` is the edge index of the pair factor of founders
    a < b, or -1 when no real-real assignment uses that pair.
    """
    if endpoint_equivalence_scale is None:
        endpoint_equivalence_scale = mean_field_tolerance
    n_samples = likelihood.shape[0]
    k = haplotypes.shape[0]
    n_states = (k + 1) * (k + 2) // 2
    tile_sites = site_stop - site_start
    membership_groups = _membership_groups(member_offsets, member_states)
    representatives = np.empty((tile_sites, n_samples), dtype=np.int64)
    predictions = np.empty((tile_sites, n_samples, n_states), dtype=np.float64)
    not_converged = np.empty((tile_sites, n_samples), dtype=np.int64)
    for offset in prange(tile_sites):
        site = site_start + offset
        work = _site_workspace(k, pair.shape[0], mean_field_max_iter)
        _set_site_potentials(
            site, k, unary, pair, edge_first, edge_second, work
        )
        seen = {}
        for heldout in range(n_samples):
            # Supported inputs are finite normalized likelihoods; their log
            # and wildcard caches are deterministic functions of this triple.
            # Equality is exact, with no rounding or likelihood bucketing.
            # A sample whose triple no other sample shares at this site can
            # never join a group, so only shared triples are looked up.
            if shared_triple[heldout, site]:
                key = (membership_groups[heldout], likelihood[heldout, site, 0],
                       likelihood[heldout, site, 1], likelihood[heldout, site, 2])
                if key in seen:
                    representatives[offset, heldout] = seen[key]
                    continue
                seen[key] = heldout
            representatives[offset, heldout] = heldout
            q, failed = _solve_cavity_site(
                heldout, site, k, log_likelihood, rw_log_likelihood, haplotypes,
                assignments, founder_support, unary, pair, edge_first,
                edge_second, neighbours, neighbour_counts, mean_field_max_iter,
                mean_field_tolerance, work, edge_lookup, elbo_tie, reverse,
                member_offsets, member_states, state_first, state_second, sample_support)
            not_converged[offset, heldout] = failed

            g0 = likelihood[heldout, site, 0]
            g1 = likelihood[heldout, site, 1]
            g2 = likelihood[heldout, site, 2]
            endpoints = _eligible_endpoint_indices(work[1], work[2], work[3], elbo_tie)
            if endpoint_counts is not None:
                classes = 1
                if len(endpoints) > 1:
                    labels = _endpoint_class_labels(work[1], endpoints, endpoint_equivalence_scale)
                    classes = 0
                    for endpoint in range(len(endpoints)):
                        if labels[endpoint] == endpoint:
                            classes += 1
                endpoint_counts[offset, heldout, 0] = len(endpoints)
                endpoint_counts[offset, heldout, 1] = classes
            if len(endpoints) > 1:
                predictions[offset, heldout, :] = _predictive_family_log_states(
                    work[1], endpoints, g0, g1, g2, likelihood_floor,
                    ww_log_emission[heldout, site], endpoint_equivalence_scale)
                continue
            # A unique endpoint retains the literal scalar arithmetic and cache path.
            # log() is a pure function: a floored predictive value equal to
            # a floored evidence value reuses that value's cached logarithm
            # (the floors below mirror the cache construction exactly).
            c0 = g0
            if c0 < likelihood_floor:
                c0 = likelihood_floor
            c1 = g1
            if c1 < likelihood_floor:
                c1 = likelihood_floor
            c2 = g2
            if c2 < likelihood_floor:
                c2 = likelihood_floor
            l0 = log_likelihood[heldout, site, 0]
            l1 = log_likelihood[heldout, site, 1]
            l2 = log_likelihood[heldout, site, 2]
            # Geometry is RR (row-major upper triangle), RW, WW. Retain
            # exactly the scalar arithmetic and floors within each state.
            state = 0
            for first in range(k):
                qi = q[first]
                predictive = (1.0 - qi) * g0 + qi * g2
                if predictive < likelihood_floor:
                    predictive = likelihood_floor
                if predictive == c0:
                    predictions[offset, heldout, state] = l0
                elif predictive == c2:
                    predictions[offset, heldout, state] = l2
                else:
                    predictions[offset, heldout, state] = math.log(predictive)
                state += 1
                for second in range(first + 1, k):
                    qj = q[second]
                    p0 = (1.0 - qi) * (1.0 - qj)
                    p2 = qi * qj
                    p1 = 1.0 - p0 - p2
                    predictive = p0 * g0 + p1 * g1 + p2 * g2
                    if predictive < likelihood_floor:
                        predictive = likelihood_floor
                    if predictive == c0:
                        predictions[offset, heldout, state] = l0
                    elif predictive == c1:
                        predictions[offset, heldout, state] = l1
                    elif predictive == c2:
                        predictions[offset, heldout, state] = l2
                    else:
                        predictions[offset, heldout, state] = math.log(predictive)
                    state += 1
            m0 = 0.5 * (g0 + g1)
            m1 = 0.5 * (g1 + g2)
            if m0 < likelihood_floor:
                m0 = likelihood_floor
            if m1 < likelihood_floor:
                m1 = likelihood_floor
            r0 = rw_log_likelihood[heldout, site, 0]
            r1 = rw_log_likelihood[heldout, site, 1]
            for first in range(k):
                qi = q[first]
                predictive = (
                    0.5 * (1.0 - qi) * g0
                    + 0.5 * g1
                    + 0.5 * qi * g2
                )
                if predictive < likelihood_floor:
                    predictive = likelihood_floor
                if predictive == m0:
                    predictions[offset, heldout, state] = r0
                elif predictive == m1:
                    predictions[offset, heldout, state] = r1
                else:
                    predictions[offset, heldout, state] = math.log(predictive)
                state += 1
            predictions[offset, heldout, state] = ww_log_emission[heldout, site]
    return representatives, predictions, not_converged


@njit(cache=True)
def _logaddexp_values(values: np.ndarray) -> float:
    maximum = -math.inf
    for index in range(values.shape[0]):
        if values[index] > maximum:
            maximum = values[index]
    total = 0.0
    for index in range(values.shape[0]):
        total += math.exp(values[index] - maximum)
    return maximum + math.log(total)


@njit(cache=True)
def _floor_log_likelihood_kernel(
    likelihood: np.ndarray,
    likelihood_floor: float,
) -> np.ndarray:
    """log max(likelihood, floor) of every cell, once per block."""

    n_samples, n_sites, _ = likelihood.shape
    result = np.empty_like(likelihood)
    for sample in range(n_samples):
        for site in range(n_sites):
            for genotype in range(3):
                value = likelihood[sample, site, genotype]
                if value < likelihood_floor:
                    value = likelihood_floor
                result[sample, site, genotype] = math.log(value)
    return result


@njit(cache=True)
def _build_evidence_emission_cache_kernel(
    likelihood: np.ndarray,
    likelihood_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evidence-only log emissions of a known-allele copy paired with an unknown copy (per allele)
    and of two unknown copies, shared by every scored panel."""

    n_samples, n_sites, _ = likelihood.shape
    rw_log_likelihood = np.empty(
        (n_samples, n_sites, 2), dtype=np.float64
    )
    ww_log_emission = np.empty((n_samples, n_sites), dtype=np.float64)
    for sample in range(n_samples):
        for site in range(n_sites):
            g0 = likelihood[sample, site, 0]
            g1 = likelihood[sample, site, 1]
            g2 = likelihood[sample, site, 2]
            mix0 = 0.5 * (g0 + g1)
            mix1 = 0.5 * (g1 + g2)
            ww = 0.25 * g0 + 0.5 * g1 + 0.25 * g2
            if mix0 < likelihood_floor:
                mix0 = likelihood_floor
            if mix1 < likelihood_floor:
                mix1 = likelihood_floor
            if ww < likelihood_floor:
                ww = likelihood_floor
            rw_log_likelihood[sample, site, 0] = math.log(mix0)
            rw_log_likelihood[sample, site, 1] = math.log(mix1)
            ww_log_emission[sample, site] = math.log(ww)
    return rw_log_likelihood, ww_log_emission


@njit(cache=True)
def _state_membership(assignments, k, tie_mask, state_kind, state_first, state_second):
    """Compact full membership and actual founder pairs (K denotes the unknown copy)."""
    n_samples = assignments.shape[0]
    n_states = state_kind.shape[0]
    first = np.empty(n_states, dtype=np.int64)
    second = np.empty(n_states, dtype=np.int64)
    lookup = np.full((k + 1, k + 1), -1, dtype=np.int64)
    for state in range(n_states):
        kind = state_kind[state]
        a = state_first[state] if kind < 2 else k
        b = state_second[state] if kind == 0 else k
        first[state], second[state] = a, b
        lookup[a, b] = state
    offsets = np.zeros(n_samples + 1, dtype=np.int64)
    with_ties = tie_mask.shape[0] == n_samples
    for sample in range(n_samples):
        size = 0
        if with_ties:
            for state in range(n_states):
                size += int(tie_mask[sample, state] != 0)
        offsets[sample + 1] = offsets[sample] + max(1, size)
    members = np.empty(offsets[-1], dtype=np.int64)
    for sample in range(n_samples):
        offset = offsets[sample]
        if with_ties:
            for state in range(n_states):
                if tie_mask[sample, state]:
                    members[offset] = state
                    offset += 1
        if offset == offsets[sample]:
            members[offset] = lookup[assignments[sample, 0], assignments[sample, 1]]
    return offsets, members, first, second


@njit(cache=True)
def _build_cavity_potentials(log_likelihood, rw_log_likelihood, k,
                             member_offsets, member_states, first, second):
    """Weighted founder fields and distinct-sample support, before any held-out deletion."""
    n_samples, n_sites, _ = log_likelihood.shape
    pair_used = np.zeros((k, k), dtype=np.uint8)
    founder_support = np.zeros(k, dtype=np.int64)
    sample_support = np.zeros((n_samples, k), dtype=np.uint8)
    state_counts = np.zeros(first.shape[0], dtype=np.float64)
    for sample in range(n_samples):
        begin, end = member_offsets[sample], member_offsets[sample + 1]
        share = 1.0 / (end - begin)
        for member in range(begin, end):
            state = member_states[member]
            a, b = first[state], second[state]
            state_counts[state] += share
            if a < k:
                sample_support[sample, a] = 1
            if b < k:
                sample_support[sample, b] = 1
            if a < b and b < k:
                pair_used[a, b] = 1
        for founder in range(k):
            founder_support[founder] += sample_support[sample, founder]

    neighbours = np.empty((k, k), dtype=np.int64)
    neighbour_counts = np.zeros(k, dtype=np.int64)
    edge_first = np.empty(k * (k - 1) // 2, dtype=np.int64)
    edge_second = np.empty(k * (k - 1) // 2, dtype=np.int64)
    edge_lookup = np.full((k, k), -1, dtype=np.int64)
    n_edges = 0
    for a in range(k):
        for b in range(a + 1, k):
            if pair_used[a, b]:
                neighbours[a, neighbour_counts[a]] = b
                neighbours[b, neighbour_counts[b]] = a
                neighbour_counts[a] += 1
                neighbour_counts[b] += 1
                edge_first[n_edges], edge_second[n_edges] = a, b
                edge_lookup[a, b] = n_edges
                n_edges += 1

    unary = np.full((k, n_sites, 2), -math.log(2.0), dtype=np.float64)
    pair = np.zeros((n_edges, n_sites, 4), dtype=np.float64)
    for sample in range(n_samples):
        begin, end = member_offsets[sample], member_offsets[sample + 1]
        share = 1.0 / (end - begin)
        for member in range(begin, end):
            state = member_states[member]
            a, b = first[state], second[state]
            if a == b and a < k:
                for site in range(n_sites):
                    unary[a, site, 0] += share * log_likelihood[sample, site, 0]
                    unary[a, site, 1] += share * log_likelihood[sample, site, 2]
            elif a < b and b < k:
                edge = edge_lookup[a, b]
                for site in range(n_sites):
                    pair[edge, site, 0] += share * log_likelihood[sample, site, 0]
                    pair[edge, site, 1] += share * log_likelihood[sample, site, 1]
                    pair[edge, site, 2] += share * log_likelihood[sample, site, 1]
                    pair[edge, site, 3] += share * log_likelihood[sample, site, 2]
            elif a < k:
                for site in range(n_sites):
                    unary[a, site, 0] += share * rw_log_likelihood[sample, site, 0]
                    unary[a, site, 1] += share * rw_log_likelihood[sample, site, 1]
    return (unary, pair, founder_support, sample_support, state_counts,
            edge_first[:n_edges], edge_second[:n_edges], edge_lookup, neighbours, neighbour_counts)


@njit(cache=True, parallel=True)
def _score_mode_cavity_kernel(
    likelihood: np.ndarray,
    log_likelihood: np.ndarray,
    rw_log_likelihood: np.ndarray,
    ww_log_emission: np.ndarray,
    haplotypes: np.ndarray,
    assignments: np.ndarray,
    state_kind: np.ndarray,
    state_first: np.ndarray,
    state_second: np.ndarray,
    prior_base: np.ndarray,
    state_concentration: float,
    mean_field_max_iter: int,
    mean_field_tolerance: float,
    likelihood_floor: float,
    shared_triple: np.ndarray,
    elbo_tie: float,
    reverse: bool,
    tie_mask: np.ndarray,
    endpoint_counts=None,
    endpoint_equivalence_scale=None,
) -> tuple[np.ndarray, int]:
    """Fixed-assignment cavity log predictive of every sample of one complete panel, and the
    number of held-out site solves whose three mean-field starts all failed to converge
    (``elbo_tie``: the start-choice tie quantum, see the module docstring).

    Exact equal assignment/evidence cases share per-site posterior solves. Each sample
    accumulates its own predictions in site/founder order.
    """

    if endpoint_equivalence_scale is None:
        endpoint_equivalence_scale = mean_field_tolerance
    n_samples, n_sites, _ = likelihood.shape
    k = haplotypes.shape[0]
    n_states = state_kind.shape[0]
    member_offsets, member_states, first, second = _state_membership(
        assignments, k, tie_mask, state_kind, state_first, state_second)
    (unary, pair, founder_support, sample_support, state_counts,
     edge_first, edge_second, edge_lookup, neighbours, neighbour_counts) = _build_cavity_potentials(
        log_likelihood, rw_log_likelihood, k, member_offsets, member_states, first, second)

    sample_scores = np.zeros(n_samples, dtype=np.float64)
    not_converged_by_heldout = np.zeros(n_samples, dtype=np.int64)
    # The common denominator N-1+alpha is shared by held-out samples/states.
    log_prior_denominator = math.log(
        n_samples - 1 + state_concentration
    )
    # Cache the full masses and single-state deletions once per panel. Multi-state
    # held-outs below instead subtract their share at every member of their tied set.
    full_log_state_mass = np.empty(n_states, dtype=np.float64)
    deleted_log_state_mass = np.empty(n_states, dtype=np.float64)
    for state in range(n_states):
        prior_mass = state_concentration * prior_base[state]
        full_log_state_mass[state] = math.log(
            state_counts[state] + prior_mass
        )
        if state_counts[state] >= 1.0:
            deleted_log_state_mass[state] = math.log(
                state_counts[state] - 1 + prior_mass
            )
        else:
            deleted_log_state_mass[state] = 0.0

    emissions = np.zeros((n_samples, n_states), dtype=np.float64)
    bytes_per_site = 8 * max(1, n_samples) * (n_states + 4)
    tile_width = max(1, PREDICTION_TILE_BYTES // bytes_per_site)
    for site_start in range(0, n_sites, tile_width):
        site_stop = min(n_sites, site_start + tile_width)
        tile_endpoint_counts = np.empty((site_stop - site_start, n_samples, 2), dtype=np.int64)
        representatives, predictions, tile_not_converged = grouped_cavity_predictions(
            likelihood, log_likelihood, rw_log_likelihood, ww_log_emission,
            haplotypes, assignments, founder_support, unary, pair, edge_first,
            edge_second, edge_lookup, neighbours, neighbour_counts,
            mean_field_max_iter, mean_field_tolerance, likelihood_floor,
            site_start, site_stop, shared_triple, elbo_tie, reverse,
            member_offsets, member_states, first, second, sample_support,
            tile_endpoint_counts, endpoint_equivalence_scale,
        )
        if endpoint_counts is not None:
            # Counts refer to every held-out sample/site, including shared solves.
            # Row zero: exact-distinct endpoints; row one: equivalence classes.
            for offset in range(site_stop - site_start):
                for heldout in range(n_samples):
                    representative = representatives[offset, heldout]
                    for identity in range(2):
                        count = tile_endpoint_counts[offset, representative, identity]
                        endpoint_counts[identity, count - 1] += 1
        for heldout in prange(n_samples):
            heldout_not_converged = not_converged_by_heldout[heldout]
            # Continue each sample's site-ordered accumulation across tiles.
            # A group is computation reuse, not a change in its weight.
            for offset in range(site_stop - site_start):
                representative = representatives[offset, heldout]
                heldout_not_converged += tile_not_converged[offset, representative]
                for state in range(n_states):
                    emissions[heldout, state] += predictions[
                        offset, representative, state
                    ]
            not_converged_by_heldout[heldout] = heldout_not_converged

    for heldout in prange(n_samples):
        log_state_predictive = np.empty(n_states, dtype=np.float64)
        for state in range(n_states):
            log_state_predictive[state] = full_log_state_mass[state]
        begin = member_offsets[heldout]
        end = member_offsets[heldout + 1]
        if end - begin == 1:
            held_state = member_states[begin]
            log_state_predictive[held_state] = deleted_log_state_mass[held_state]
        else:
            share = 1.0 / (end - begin)
            for member in range(begin, end):
                state = member_states[member]
                log_state_predictive[state] = math.log(
                    state_counts[state] - share + state_concentration * prior_base[state])
        for state in range(n_states):
            log_state_predictive[state] += emissions[heldout, state]
            log_state_predictive[state] -= log_prior_denominator
        sample_scores[heldout] = _logaddexp_values(log_state_predictive)

    failed_solves = 0
    for heldout in range(n_samples):
        failed_solves += not_converged_by_heldout[heldout]
    return sample_scores, failed_solves
