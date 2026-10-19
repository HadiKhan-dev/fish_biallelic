"""Kernels of the fixed-assignment leave-one-sample-out cavity score (see cavity.py).

At a fixed site, held-out samples with equal assigned pairs and equal likelihood triples have
equal held-out potentials, support and starts: each such case is solved once (exact equality, no
quantisation). Further exact reuse: a later mean-field start that reaches a state of the converged
first start replays that start's remaining sweeps; a predictive value equal to a floored evidence
value reuses that value's logarithm; the site's complete-data potentials are filled once and a
held-out solve patches only its own sample's unary or edge term (restored afterwards); sigmoid and
entropy terms are memoised per founder. Every sample accumulates its own predictions in site and
founder order and samples are reduced in sample order, so results do not depend on threads.

Release 2.1: the three mean-field starts of a held-out solve are compared by ELBO with a tie
quantum (the block's ``nll`` quantum, ``elbo_tie``): converged starts within it of the best ELBO are
tied and the first in the fixed start order (q = H, q = 1 - H, q = 1/2) wins. 2.0 compared with an
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
    )


@njit(cache=True)
def _set_site_potentials(site, k, unary, pair, edge_first, edge_second, work):
    """Fill the site's complete-data potentials, shared by its held-outs.

    Each held-out solve then patches only the one unary or edge to which its
    own sample contributes and restores it afterwards.
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
        mean_field_tolerance, work, edge_lookup, elbo_tie, reverse):
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
    ) = work
    wildcard = k
    n_edges = pair.shape[0]
    heldout_not_converged = 0
    held_a = assignments[heldout, 0]
    held_b = assignments[heldout, 1]
    held_support_a = held_a if held_a < wildcard else -1
    held_support_b = (
        held_b if held_b < wildcard and held_b != held_a else -1
    )
    n_supported = 0
    for founder in range(k):
        remaining_support = founder_support[founder]
        if founder == held_support_a or founder == held_support_b:
            remaining_support -= 1
        remaining_support_by_founder[founder] = remaining_support
        if remaining_support > 0:
            supported_founders[n_supported] = founder
            n_supported += 1
    held_log_mix0 = 0.0
    held_log_mix1 = 0.0
    if held_a < wildcard and held_b == wildcard:
        held_log_mix0 = rw_log_likelihood[heldout, site, 0]
        held_log_mix1 = rw_log_likelihood[heldout, site, 1]

    # Remove this sample's own term from the site potentials: the same
    # subtraction of the same values as rebuilding every potential, at the
    # one unary or edge it contributes to (each is restored below).
    patched_founder = -1
    saved_u0 = 0.0
    saved_u1 = 0.0
    saved_logit = 0.0
    if held_a == held_b and held_a < wildcard:
        patched_founder = held_a
    elif held_a < wildcard and held_b == wildcard:
        patched_founder = held_a
    if patched_founder >= 0:
        saved_u0 = cavity_unary[patched_founder, 0]
        saved_u1 = cavity_unary[patched_founder, 1]
        saved_logit = cavity_unary_logit[patched_founder]
        u0 = saved_u0
        u1 = saved_u1
        if held_a == held_b:
            u0 -= log_likelihood[heldout, site, 0]
            u1 -= log_likelihood[heldout, site, 2]
        else:
            u0 -= held_log_mix0
            u1 -= held_log_mix1
        cavity_unary[patched_founder, 0] = u0
        cavity_unary[patched_founder, 1] = u1
        cavity_unary_logit[patched_founder] = u1 - u0
    patched_edge = -1
    saved_v00 = 0.0
    saved_v01 = 0.0
    saved_v10 = 0.0
    saved_v11 = 0.0
    saved_d0_ab = 0.0
    saved_d1_ab = 0.0
    saved_d0_ba = 0.0
    saved_d1_ba = 0.0
    if held_a < wildcard and held_b < wildcard:
        patched_edge = edge_lookup[held_a, held_b]
    if patched_edge >= 0:
        saved_v00 = cavity_edge_potential[patched_edge, 0]
        saved_v01 = cavity_edge_potential[patched_edge, 1]
        saved_v10 = cavity_edge_potential[patched_edge, 2]
        saved_v11 = cavity_edge_potential[patched_edge, 3]
        saved_d0_ab = directed_difference_zero[held_a, held_b]
        saved_d1_ab = directed_difference_one[held_a, held_b]
        saved_d0_ba = directed_difference_zero[held_b, held_a]
        saved_d1_ba = directed_difference_one[held_b, held_a]
        v00 = saved_v00 - log_likelihood[heldout, site, 0]
        v01 = saved_v01 - log_likelihood[heldout, site, 1]
        v10 = saved_v10 - log_likelihood[heldout, site, 1]
        v11 = saved_v11 - log_likelihood[heldout, site, 2]
        cavity_edge_potential[patched_edge, 0] = v00
        cavity_edge_potential[patched_edge, 1] = v01
        cavity_edge_potential[patched_edge, 2] = v10
        cavity_edge_potential[patched_edge, 3] = v11
        directed_difference_zero[held_a, held_b] = v10 - v00
        directed_difference_one[held_a, held_b] = v11 - v01
        directed_difference_zero[held_b, held_a] = v01 - v00
        directed_difference_one[held_b, held_a] = v11 - v10

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
    if patched_founder >= 0:
        cavity_unary[patched_founder, 0] = saved_u0
        cavity_unary[patched_founder, 1] = saved_u1
        cavity_unary_logit[patched_founder] = saved_logit
    if patched_edge >= 0:
        cavity_edge_potential[patched_edge, 0] = saved_v00
        cavity_edge_potential[patched_edge, 1] = saved_v01
        cavity_edge_potential[patched_edge, 2] = saved_v10
        cavity_edge_potential[patched_edge, 3] = saved_v11
        directed_difference_zero[held_a, held_b] = saved_d0_ab
        directed_difference_one[held_a, held_b] = saved_d1_ab
        directed_difference_zero[held_b, held_a] = saved_d0_ba
        directed_difference_one[held_b, held_a] = saved_d1_ba

    return q, heldout_not_converged


@njit(cache=True, parallel=True)
def grouped_cavity_predictions(
        likelihood, log_likelihood, rw_log_likelihood, ww_log_emission,
        haplotypes, assignments, founder_support, unary, pair, edge_first,
        edge_second, edge_lookup, neighbours, neighbour_counts,
        mean_field_max_iter, mean_field_tolerance, likelihood_floor,
        site_start, site_stop, shared_triple, elbo_tie, reverse):
    """Solve and predict exact cases once in a bounded, site-parallel tile.

    ``edge_lookup[a, b]`` is the edge index of the pair factor of founders
    a < b, or -1 when no real-real assignment uses that pair.
    """
    n_samples = likelihood.shape[0]
    k = haplotypes.shape[0]
    n_states = (k + 1) * (k + 2) // 2
    tile_sites = site_stop - site_start
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
                key = (assignments[heldout, 0], assignments[heldout, 1],
                       likelihood[heldout, site, 0], likelihood[heldout, site, 1],
                       likelihood[heldout, site, 2])
                if key in seen:
                    representatives[offset, heldout] = seen[key]
                    continue
                seen[key] = heldout
            representatives[offset, heldout] = heldout
            q, failed = _solve_cavity_site(
                heldout, site, k, log_likelihood, rw_log_likelihood, haplotypes,
                assignments, founder_support, unary, pair, edge_first,
                edge_second, neighbours, neighbour_counts, mean_field_max_iter,
                mean_field_tolerance, work, edge_lookup, elbo_tie, reverse)
            not_converged[offset, heldout] = failed

            g0 = likelihood[heldout, site, 0]
            g1 = likelihood[heldout, site, 1]
            g2 = likelihood[heldout, site, 2]
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
) -> tuple[np.ndarray, int]:
    """Fixed-assignment cavity log predictive of every sample of one complete panel, and the
    number of held-out site solves whose three mean-field starts all failed to converge
    (``elbo_tie``: the start-choice tie quantum, see the module docstring).

    Exact equal assignment/evidence cases share per-site posterior solves. Each sample
    accumulates its own predictions in site/founder order.
    """

    n_samples, n_sites, _ = likelihood.shape
    k = haplotypes.shape[0]
    n_states = state_kind.shape[0]
    wildcard = k
    log_half = -math.log(2.0)

    # Count support and active row-pair edges first; the potentials are then built in a
    # sample-ordered pass storing only the edges some row-pair assignment uses.
    pair_counts = np.zeros((k, k), dtype=np.int64)
    founder_support = np.zeros(k, dtype=np.int64)
    state_counts = np.zeros(n_states, dtype=np.float64)
    # Release 2.2: a sample whose A step held several tied states T (tie_mask rows, discovery) counts 1/|T| at each
    # of them; every other sample one count at its assigned state (the same values as integer counts).
    with_ties = tie_mask.shape[0] == n_samples
    tie_size = np.ones(n_samples, dtype=np.int64)
    if with_ties:
        for sample in range(n_samples):
            size = 0
            for state in range(n_states):
                size += tie_mask[sample, state]
            tie_size[sample] = size if size > 0 else 1

    # Dense (K+1)^2 state lookup (K <= 30: under 8 KB).
    state_lookup = np.empty((k + 1, k + 1), dtype=np.int64)
    state_lookup[:,:] = -1
    for state in range(n_states):
        kind = state_kind[state]
        if kind == 0:
            a = state_first[state]
            b = state_second[state]
        elif kind == 1:
            a = state_first[state]
            b = wildcard
        else:
            a = wildcard
            b = wildcard
        state_lookup[a, b] = state

    for sample in range(n_samples):
        a = assignments[sample, 0]
        b = assignments[sample, 1]
        if tie_size[sample] > 1:
            share = 1.0 / tie_size[sample]
            for state in range(n_states):
                if tie_mask[sample, state]:
                    state_counts[state] += share
        else:
            state_counts[state_lookup[a, b]] += 1.0
        if a < wildcard:
            founder_support[a] += 1
        if b < wildcard and b != a:
            founder_support[b] += 1
        if a < b and b < wildcard:
            pair_counts[a, b] += 1

    # Neighbours in ascending order and edges in ascending pair order.
    neighbours = np.empty((k, k), dtype=np.int64)
    neighbour_counts = np.zeros(k, dtype=np.int64)
    edge_first = np.empty(k * (k - 1) // 2, dtype=np.int64)
    edge_second = np.empty(k * (k - 1) // 2, dtype=np.int64)
    edge_lookup = np.empty((k, k), dtype=np.int64)
    edge_lookup[:,:] = -1
    n_edges = 0
    for first_founder in range(k):
        for second_founder in range(first_founder + 1, k):
            if pair_counts[first_founder, second_founder] == 0:
                continue
            neighbours[first_founder, neighbour_counts[first_founder]] = (
                second_founder
            )
            neighbour_counts[first_founder] += 1
            neighbours[second_founder, neighbour_counts[second_founder]] = (
                first_founder
            )
            neighbour_counts[second_founder] += 1
            edge_first[n_edges] = first_founder
            edge_second[n_edges] = second_founder
            edge_lookup[first_founder, second_founder] = n_edges
            n_edges += 1

    unary = np.empty((k, n_sites, 2), dtype=np.float64)
    pair = np.zeros((n_edges, n_sites, 4), dtype=np.float64)
    for founder in range(k):
        for site in range(n_sites):
            unary[founder, site, 0] = log_half
            unary[founder, site, 1] = log_half
    for sample in range(n_samples):
        a = assignments[sample, 0]
        b = assignments[sample, 1]
        if a == b and a < wildcard:
            for site in range(n_sites):
                unary[a, site, 0] += log_likelihood[sample, site, 0]
                unary[a, site, 1] += log_likelihood[sample, site, 2]
        elif a < b and b < wildcard:
            edge = edge_lookup[a, b]
            for site in range(n_sites):
                pair[edge, site, 0] += log_likelihood[sample, site, 0]
                pair[edge, site, 1] += log_likelihood[sample, site, 1]
                pair[edge, site, 2] += log_likelihood[sample, site, 1]
                pair[edge, site, 3] += log_likelihood[sample, site, 2]
        elif a < wildcard and b == wildcard:
            for site in range(n_sites):
                unary[a, site, 0] += rw_log_likelihood[sample, site, 0]
                unary[a, site, 1] += rw_log_likelihood[sample, site, 1]

    sample_scores = np.zeros(n_samples, dtype=np.float64)
    not_converged_by_heldout = np.zeros(n_samples, dtype=np.int64)
    # The common denominator N-1+alpha is shared by held-out samples/states.
    log_prior_denominator = math.log(
        n_samples - 1 + state_concentration
    )
    # Every held-out prior differs from the full count vector at one state.
    # Cache both alternatives once per mode rather than taking the same logs
    # for every sample.
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
    bytes_per_site = 8 * max(1, n_samples) * (n_states + 2)
    tile_width = max(1, PREDICTION_TILE_BYTES // bytes_per_site)
    for site_start in range(0, n_sites, tile_width):
        site_stop = min(n_sites, site_start + tile_width)
        representatives, predictions, tile_not_converged = grouped_cavity_predictions(
            likelihood, log_likelihood, rw_log_likelihood, ww_log_emission,
            haplotypes, assignments, founder_support, unary, pair, edge_first,
            edge_second, edge_lookup, neighbours, neighbour_counts,
            mean_field_max_iter, mean_field_tolerance, likelihood_floor,
            site_start, site_stop, shared_triple, elbo_tie, reverse,
        )
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
        if tie_size[heldout] > 1:
            share = 1.0 / tie_size[heldout]
            for state in range(n_states):
                if tie_mask[heldout, state]:
                    log_state_predictive[state] = math.log(
                        state_counts[state] - share + state_concentration * prior_base[state])
        else:
            held_state = state_lookup[assignments[heldout, 0], assignments[heldout, 1]]
            log_state_predictive[held_state] = deleted_log_state_mass[held_state]
        for state in range(n_states):
            log_state_predictive[state] += emissions[heldout, state]
            log_state_predictive[state] -= log_prior_denominator
        sample_scores[heldout] = _logaddexp_values(log_state_predictive)

    failed_solves = 0
    for heldout in range(n_samples):
        failed_solves += not_converged_by_heldout[heldout]
    return sample_scores, failed_solves
