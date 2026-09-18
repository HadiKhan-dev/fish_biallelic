"""Exact per-site leave-one-out cavity reuse; no evidence quantization.

At a fixed site, equal assigned pairs and equal likelihood triples imply equal
held-out potentials, support and deterministic starts. Solve each such case
once, including their predictive emissions and entropy terms. The caller
bounds the site tile so storage never grows as an N L K-squared tensor.
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange


@njit(cache=True)
def _stable_sigmoid(logit: float) -> float:
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


@njit(cache=True)
def _site_workspace(k, n_edges):
    q = np.empty(k, dtype=np.float64)
    candidate_q = np.empty((3, k), dtype=np.float64)
    candidate_elbo = np.empty(3, dtype=np.float64)
    candidate_converged = np.empty(3, dtype=np.uint8)
    candidate_q_log_q = np.empty((3, k), dtype=np.float64)
    candidate_one_minus_q_log_one_minus_q = np.empty(
        (3, k), dtype=np.float64
    )
    cavity_unary = np.empty((k, 2), dtype=np.float64)
    cavity_unary_logit = np.empty(k, dtype=np.float64)
    directed_difference_zero = np.empty((k, k), dtype=np.float64)
    directed_difference_one = np.empty((k, k), dtype=np.float64)
    cavity_edge_potential = np.empty((n_edges, 4), dtype=np.float64)
    remaining_support_by_founder = np.empty(k, dtype=np.int64)
    supported_founders = np.empty(k, dtype=np.int64)
    return (
        q,
        candidate_q,
        candidate_elbo,
        candidate_converged,
        candidate_q_log_q,
        candidate_one_minus_q_log_one_minus_q,
        cavity_unary,
        cavity_unary_logit,
        directed_difference_zero,
        directed_difference_one,
        cavity_edge_potential,
        remaining_support_by_founder,
        supported_founders,
    )


@njit(cache=True)
def _solve_cavity_site(
        heldout, site, k, log_likelihood, rw_log_likelihood, haplotypes,
        assignments, founder_support, unary, pair, edge_first, edge_second,
        neighbours, neighbour_counts, mean_field_max_iter,
        mean_field_tolerance, work):
    (
        q,
        candidate_q,
        candidate_elbo,
        candidate_converged,
        candidate_q_log_q,
        candidate_one_minus_q_log_one_minus_q,
        cavity_unary,
        cavity_unary_logit,
        directed_difference_zero,
        directed_difference_one,
        cavity_edge_potential,
        remaining_support_by_founder,
        supported_founders,
    ) = work
    wildcard = k
    n_edges = pair.shape[0]
    heldout_iterations = 0
    heldout_not_converged = 0
    heldout_alternate_wins = 0
    heldout_initialization_spread = 0.0
    held_a = assignments[heldout, 0]
    held_b = assignments[heldout, 1]
    held_support_a = held_a if held_a < wildcard else -1
    held_support_b = (
        held_b if held_b < wildcard and held_b != held_a else -1
    )
    n_supported = 0
    n_zero_support = 0
    for founder in range(k):
        remaining_support = founder_support[founder]
        if founder == held_support_a or founder == held_support_b:
            remaining_support -= 1
        remaining_support_by_founder[founder] = remaining_support
        if remaining_support > 0:
            supported_founders[n_supported] = founder
            n_supported += 1
        else:
            n_zero_support += 1
    held_log_mix0 = 0.0
    held_log_mix1 = 0.0
    if held_a < wildcard and held_b == wildcard:
        held_log_mix0 = rw_log_likelihood[heldout, site, 0]
        held_log_mix1 = rw_log_likelihood[heldout, site, 1]

    for founder in range(k):
        u0 = unary[founder, site, 0]
        u1 = unary[founder, site, 1]
        if held_a == held_b and held_a == founder:
            u0 -= log_likelihood[heldout, site, 0]
            u1 -= log_likelihood[heldout, site, 2]
        elif held_a == founder and held_b == wildcard:
            u0 -= held_log_mix0
            u1 -= held_log_mix1
        cavity_unary[founder, 0] = u0
        cavity_unary[founder, 1] = u1
        cavity_unary_logit[founder] = u1 - u0

    for edge_index in range(n_edges):
        first_founder = edge_first[edge_index]
        second_founder = edge_second[edge_index]
        v00 = pair[edge_index, site, 0]
        v01 = pair[edge_index, site, 1]
        v10 = pair[edge_index, site, 2]
        v11 = pair[edge_index, site, 3]
        if (
            held_a == first_founder
            and held_b == second_founder
        ):
            v00 -= log_likelihood[heldout, site, 0]
            v01 -= log_likelihood[heldout, site, 1]
            v10 -= log_likelihood[heldout, site, 1]
            v11 -= log_likelihood[heldout, site, 2]
        cavity_edge_potential[edge_index, 0] = v00
        cavity_edge_potential[edge_index, 1] = v01
        cavity_edge_potential[edge_index, 2] = v10
        cavity_edge_potential[edge_index, 3] = v11
        directed_difference_zero[
            first_founder, second_founder
        ] = v10 - v00
        directed_difference_one[
            first_founder, second_founder
        ] = v11 - v01
        directed_difference_zero[
            second_founder, first_founder
        ] = v01 - v00
        directed_difference_one[
            second_founder, first_founder
        ] = v11 - v10

    candidate_converged[:] = 0
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
        used_iterations = 0
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

                updated = _stable_sigmoid(logit)
                change = abs(updated - q[founder])
                if change > maximum_change:
                    maximum_change = change
                q[founder] = updated
            used_iterations = iteration + 1
            if maximum_change <= mean_field_tolerance:
                converged = True
                break
        heldout_iterations += used_iterations
        if converged:
            candidate_converged[start_index] = 1

        elbo = 0.0
        for founder in range(k):
            u0 = cavity_unary[founder, 0]
            u1 = cavity_unary[founder, 1]
            probability = q[founder]
            elbo += (1.0 - probability) * u0 + probability * u1
            q_log_q = 0.0
            one_minus_q_log_one_minus_q = 0.0
            if probability > 0.0 and probability < 1.0:
                q_log_q = probability * math.log(probability)
                one_minus_q_log_one_minus_q = (
                    (1.0 - probability) * math.log(
                        1.0 - probability
                    )
                )
                elbo -= q_log_q
                elbo -= one_minus_q_log_one_minus_q
            candidate_q_log_q[start_index, founder] = q_log_q
            candidate_one_minus_q_log_one_minus_q[
                start_index, founder
            ] = one_minus_q_log_one_minus_q
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

    any_converged = False
    best_start = 0
    best_elbo = -math.inf
    minimum_converged_elbo = math.inf
    for start_index in range(3):
        if candidate_converged[start_index] == 1:
            any_converged = True
            value = candidate_elbo[start_index]
            better = value > best_elbo + 1e-12
            if abs(value - best_elbo) <= 1e-12:
                for founder in range(k):
                    difference = (
                        candidate_q[start_index, founder]
                        - candidate_q[best_start, founder]
                    )
                    if difference < -1e-12:
                        better = True
                        break
                    if difference > 1e-12:
                        break
            if better:
                best_elbo = value
                best_start = start_index
            if value < minimum_converged_elbo:
                minimum_converged_elbo = value
    if not any_converged:
        heldout_not_converged += 1
        best_start = 0
        best_elbo = candidate_elbo[0]
        for start_index in range(1, 3):
            value = candidate_elbo[start_index]
            better = value > best_elbo + 1e-12
            if abs(value - best_elbo) <= 1e-12:
                for founder in range(k):
                    difference = (
                        candidate_q[start_index, founder]
                        - candidate_q[best_start, founder]
                    )
                    if difference < -1e-12:
                        better = True
                        break
                    if difference > 1e-12:
                        break
            if better:
                best_elbo = value
                best_start = start_index
    else:
        heldout_initialization_spread += (
            best_elbo - minimum_converged_elbo
        )
    if best_start != 0:
        heldout_alternate_wins += 1
    q[:] = candidate_q[best_start,:]

    return (q, heldout_iterations, heldout_not_converged, n_zero_support,
            heldout_alternate_wins, heldout_initialization_spread,
            candidate_q_log_q[best_start],
            candidate_one_minus_q_log_one_minus_q[best_start])


@njit(cache=True, parallel=True)
def grouped_cavity_predictions(
        likelihood, log_likelihood, rw_log_likelihood, ww_log_emission,
        haplotypes, assignments, founder_support, unary, pair, edge_first,
        edge_second, neighbours, neighbour_counts, mean_field_max_iter,
        mean_field_tolerance, likelihood_floor, site_start, site_stop):
    """Solve and predict exact cases once in a bounded, site-parallel tile."""
    n_samples = likelihood.shape[0]
    k = haplotypes.shape[0]
    n_states = (k + 1) * (k + 2) // 2
    tile_sites = site_stop - site_start
    representatives = np.empty((tile_sites, n_samples), dtype=np.int64)
    predictions = np.empty((tile_sites, n_samples, n_states), dtype=np.float64)
    entropy_terms = np.empty((tile_sites, n_samples, 2 * k), dtype=np.float64)
    diagnostics = np.empty((tile_sites, n_samples, 4), dtype=np.int64)
    spreads = np.empty((tile_sites, n_samples), dtype=np.float64)
    for offset in prange(tile_sites):
        site = site_start + offset
        work = _site_workspace(k, pair.shape[0])
        seen = {}
        for heldout in range(n_samples):
            # Supported inputs are finite normalized likelihoods; their log
            # and wildcard caches are deterministic functions of this triple.
            # Equality is exact, with no rounding or likelihood bucketing.
            key = (assignments[heldout, 0], assignments[heldout, 1],
                   likelihood[heldout, site, 0], likelihood[heldout, site, 1],
                   likelihood[heldout, site, 2])
            if key in seen:
                representatives[offset, heldout] = seen[key]
                continue
            seen[key] = heldout
            representatives[offset, heldout] = heldout
            (q, iterations, failed, zero, alternate, spread,
             q_log_q, complement_log_complement) = _solve_cavity_site(
                heldout, site, k, log_likelihood, rw_log_likelihood, haplotypes,
                assignments, founder_support, unary, pair, edge_first,
                edge_second, neighbours, neighbour_counts, mean_field_max_iter,
                mean_field_tolerance, work)
            diagnostics[offset, heldout, 0] = iterations
            diagnostics[offset, heldout, 1] = failed
            diagnostics[offset, heldout, 2] = zero
            diagnostics[offset, heldout, 3] = alternate
            spreads[offset, heldout] = spread
            for founder in range(k):
                # Reuse the chosen start's ELBO terms, retaining the two
                # separate subtractions and founder order during replay.
                entropy_terms[offset, heldout, 2 * founder] = q_log_q[founder]
                entropy_terms[offset, heldout, 2 * founder + 1] = (
                    complement_log_complement[founder]
                )

            g0 = likelihood[heldout, site, 0]
            g1 = likelihood[heldout, site, 1]
            g2 = likelihood[heldout, site, 2]
            # Geometry is RR (row-major upper triangle), RW, WW. Retain
            # exactly the scalar arithmetic and floors within each state.
            state = 0
            for first in range(k):
                qi = q[first]
                predictive = (1.0 - qi) * g0 + qi * g2
                if predictive < likelihood_floor:
                    predictive = likelihood_floor
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
                    predictions[offset, heldout, state] = math.log(predictive)
                    state += 1
            for first in range(k):
                qi = q[first]
                predictive = (
                    0.5 * (1.0 - qi) * g0
                    + 0.5 * g1
                    + 0.5 * qi * g2
                )
                if predictive < likelihood_floor:
                    predictive = likelihood_floor
                predictions[offset, heldout, state] = math.log(predictive)
                state += 1
            predictions[offset, heldout, state] = ww_log_emission[heldout, site]
    return representatives, predictions, entropy_terms, diagnostics, spreads
