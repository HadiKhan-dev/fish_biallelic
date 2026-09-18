"""Site-level likelihoods for fixed-count founder-path refinement.

The observation model matches the assembly panel scorer: normalized genotype
likelihoods, a one-percent uniform mixture and a -2 log-likelihood floor.
Unlike its binned approximation, sample diplotypes may switch at every SNP.
Only a candidate-independent complete-site mask contributes evidence. Missing
founder alleles and unobserved sample calls are not invented or imputed here.

Symmetric genotype emissions and a uniform penalty for any diplotype change
allow exact K(K+1)/2 unordered states. This reduction is specific to this
scoring model, not the pipeline's homologue-specific ancestry HMMs.
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange


@njit(inline="always")
def _log_genotype_evidence(values):
    """Uncentered robust emissions; equal likelihoods are exactly neutral."""
    p0, p1, p2 = values[0], values[1], values[2]
    total = p0 + p1 + p2
    if total <= 0.0 or (p0 == p1 and p1 == p2):
        return 0.0, 0.0, 0.0
    p0 /= total
    p1 /= total
    p2 /= total
    return (
        max(math.log(p0 * .99 + .01 / 3.), -2.),
        max(math.log(p1 * .99 + .01 / 3.), -2.),
        max(math.log(p2 * .99 + .01 / 3.), -2.),
    )


@njit(parallel=True, cache=True, nogil=True)
def selected_alleles(leaves, offsets, selected):
    founders, blocks = selected.shape
    result = np.empty((founders, offsets[-1]), np.int8)
    for task in prange(founders * blocks):
        founder, block = task // blocks, task % blocks
        result[founder, offsets[block]:offsets[block + 1]] = leaves[block][selected[founder, block]]
    return result


@njit(parallel=True, cache=True, nogil=True)
def painting_occupancy(painting, evidence, complete, founders, tolerance):
    counts = np.zeros((evidence.shape[0], founders), np.int64)
    for sample in prange(evidence.shape[0]):
        for site in range(evidence.shape[1]):
            if complete[site]:
                p = evidence[sample, site]
                if max(p[0], p[1], p[2]) - min(p[0], p[1], p[2]) > tolerance:
                    state = painting[sample, site]
                    counts[sample, state // founders] += 1
                    counts[sample, state % founders] += 1
    return counts.sum(axis=0)


@njit(parallel=True, cache=True, nogil=True)
def prepare_log_evidence(evidence, complete):
    """Compute the unchanged robust float64 emissions once per component."""
    result = np.zeros(evidence.shape, np.float64)
    for sample in prange(evidence.shape[0]):
        for site in range(evidence.shape[1]):
            if complete[site]:
                value = _log_genotype_evidence(evidence[sample, site])
                for genotype in range(3):
                    result[sample, site, genotype] = value[genotype]
    return result


@njit(inline="always")
def _emission_at(evidence, prepared, sample, site):
    if prepared is None:
        return _log_genotype_evidence(evidence[sample, site])
    return (prepared[sample, site, 0], prepared[sample, site, 1],
            prepared[sample, site, 2])


@njit(inline="always")
def _unordered_pairs(founders):
    first = np.empty(founders * (founders + 1) // 2, np.int64)
    second = np.empty_like(first)
    state = 0
    for a in range(founders):
        for b in range(a, founders):
            first[state], second[state] = a, b
            state += 1
    return first, second


@njit(parallel=True, cache=True, nogil=True)
def _score_panel_direct(haplotypes, evidence, complete, penalty, prepared=None):
    """Per-sample Viterbi scores in O(N*L*K²) time and O(threads*K²) work RAM."""
    samples, sites, _ = evidence.shape
    first, second = _unordered_pairs(len(haplotypes))
    answer = np.zeros(samples)
    center = math.log(1. / 3.)
    for sample in prange(samples):
        scores = np.zeros(len(first))
        for site in range(sites):
            if not complete[site]:
                continue
            p = evidence[sample, site]
            if p.sum() <= 0.0 or (p[0] == p[1] and p[1] == p[2]):
                continue
            emission = _emission_at(evidence, prepared, sample, site)
            switched = np.max(scores) - penalty
            for state in range(len(first)):
                dosage = haplotypes[first[state], site] + haplotypes[second[state], site]
                scores[state] = max(scores[state], switched) + (emission[dosage] - center)
        answer[sample] = np.max(scores)
    return answer


@njit(parallel=True, cache=True, nogil=True)
def _paint_panel_direct(haplotypes, evidence, complete, penalty, prepared=None):
    """Exact site-level traceback, encoded as first_founder*K+second_founder."""
    samples, sites, _ = evidence.shape
    founders = len(haplotypes)
    first, second = _unordered_pairs(founders)
    answer = np.empty((samples, sites), np.int32)
    likelihood = np.empty(samples)
    center = math.log(1. / 3.)
    for sample in prange(samples):
        scores = np.zeros(len(first))
        switched_from = np.empty(sites, np.int32)
        did_switch = np.empty((sites, len(first)), np.bool_)
        for site in range(sites):
            p = evidence[sample, site]
            neutral = (not complete[site] or p.sum() <= 0.0
                       or (p[0] == p[1] and p[1] == p[2]))
            emission = (0.0, 0.0, 0.0) if neutral else _emission_at(evidence, prepared, sample, site)
            previous = int(np.argmax(scores))
            switched_from[site] = previous
            switched = scores[previous] - penalty
            for state in range(len(first)):
                did_switch[site, state] = scores[state] < switched
                dosage = 0 if neutral else (
                    haplotypes[first[state], site] + haplotypes[second[state], site])
                value = 0.0 if neutral else emission[dosage] - center
                scores[state] = max(scores[state], switched) + value
        state = int(np.argmax(scores))
        likelihood[sample] = scores[state]
        for site in range(sites - 1, -1, -1):
            answer[sample, site] = first[state] * founders + second[state]
            if did_switch[site, state]:
                state = switched_from[site]
    return answer, likelihood


@njit(parallel=True, cache=True, nogil=True)
def _count_panel_direct(haplotypes, evidence, complete, penalty, prepared=None):
    """Canonical Viterbi score and switch count without a site traceback.

    Carry the count of the chosen predecessor with each state. Strict switch
    comparisons and first-argmax ties match paint_panel, including neutral
    sites. The initial scores are equal and the switch penalty is positive,
    so the first site never contributes a counted transition.
    """
    samples, sites, _ = evidence.shape
    first, second = _unordered_pairs(len(haplotypes))
    likelihood = np.empty(samples)
    counts = np.empty(samples, np.int64)
    center = math.log(1. / 3.)
    for sample in prange(samples):
        scores = np.zeros(len(first))
        switches = np.zeros(len(first), np.int64)
        for site in range(sites):
            p = evidence[sample, site]
            neutral = (not complete[site] or p.sum() <= 0.0
                       or (p[0] == p[1] and p[1] == p[2]))
            emission = (0.0, 0.0, 0.0) if neutral else _emission_at(
                evidence, prepared, sample, site)
            previous = int(np.argmax(scores))
            switched = scores[previous] - penalty
            next_count = switches[previous] + 1
            for state in range(len(first)):
                if scores[state] < switched:
                    switches[state] = next_count
                dosage = 0 if neutral else (
                    haplotypes[first[state], site] + haplotypes[second[state], site])
                value = 0.0 if neutral else emission[dosage] - center
                scores[state] = max(scores[state], switched) + value
        state = int(np.argmax(scores))
        likelihood[sample], counts[sample] = scores[state], switches[state]
    return likelihood, counts


def _dosage_table(haplotypes):
    """Prepare shared dosages only within the existing workspace RAM allowance."""
    from.site_kernels import prepare_dosages
    from ...painting.model import available_process_memory_bytes
    founders, sites = haplotypes.shape
    states = founders * (founders + 1) // 2
    available = available_process_memory_bytes()
    if available is not None and sites * states > available // 8:
        return None
    first, second = _unordered_pairs(founders)
    return prepare_dosages(haplotypes, first, second)


def score_panel(haplotypes, evidence, complete, penalty, prepared=None):
    """Canonical scores; share pair dosages without changing the observation model."""
    from.site_kernels import score_dosages
    dosages = _dosage_table(haplotypes)
    if dosages is None:
        return _score_panel_direct(haplotypes, evidence, complete, penalty, prepared)
    logs = prepare_log_evidence(evidence, complete) if prepared is None else prepared
    return score_dosages(dosages, logs, float(penalty))


def paint_panel(haplotypes, evidence, complete, penalty, prepared=None):
    """Canonical full-site traceback with compact switch storage when possible."""
    from.site_kernels import paint_dosages
    founders = len(haplotypes)
    if founders * (founders + 1) // 2 > 64:
        return _paint_panel_direct(haplotypes, evidence, complete, penalty, prepared)
    dosages = _dosage_table(haplotypes)
    if dosages is None:
        return _paint_panel_direct(haplotypes, evidence, complete, penalty, prepared)
    logs = prepare_log_evidence(evidence, complete) if prepared is None else prepared
    first, second = _unordered_pairs(founders)
    return paint_dosages(dosages, logs, float(penalty), first, second, founders)


def score_and_switch_count(haplotypes, evidence, complete, penalty, prepared=None):
    """Canonical scores and switch counts, with unchanged tie handling."""
    from.site_kernels import count_switches
    dosages = _dosage_table(haplotypes)
    if dosages is None:
        return _count_panel_direct(haplotypes, evidence, complete, penalty, prepared)
    logs = prepare_log_evidence(evidence, complete) if prepared is None else prepared
    return count_switches(dosages, logs, float(penalty))

@njit(parallel=True, cache=True, nogil=True)
def fixed_path_proposals(leaves, offsets, selected, evidence, complete, painting, prepared=None):
    """Best original row per leaf under a fixed painting, including homozygotes.

    Proposals are not accepted on these conditional gains alone. Changing
    several paths together or repainting requires a full score comparison.
    Work is parallel across original blocks, independent of founder count.
    """
    founders, blocks = selected.shape
    proposed = selected.copy()
    gains = np.zeros((founders, blocks))
    for block_index in prange(blocks):
        block = np.int64(block_index)
        local = leaves[block]
        unary = np.zeros((founders, len(local)))
        begin = offsets[block]
        for sample in range(evidence.shape[0]):
            for local_site in range(local.shape[1]):
                site = begin + local_site
                if not complete[site]:
                    continue
                p = evidence[sample, site]
                if p.sum() <= 0.0 or (p[0] == p[1] and p[1] == p[2]):
                    continue
                emission = _emission_at(evidence, prepared, sample, site)
                state = painting[sample, site]
                first, second = state // founders, state % founders
                old_first = local[selected[first, block], local_site]
                old_second = local[selected[second, block], local_site]
                current = emission[old_first + old_second]
                for side in range(1 if first == second else 2):
                    focal = first if side == 0 else second
                    for candidate in range(len(local)):
                        a = local[candidate, local_site] if first == focal else old_first
                        b = local[candidate, local_site] if second == focal else old_second
                        unary[focal, candidate] += emission[a + b] - current
        for focal in range(founders):
            best = int(np.argmax(unary[focal]))
            if unary[focal, best] > 1e-8:
                proposed[focal, block] = best
                gains[focal, block] = unary[focal, best]
    return proposed, gains.sum(axis=1)
