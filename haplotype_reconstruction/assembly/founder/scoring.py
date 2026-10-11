"""Site-level likelihoods for fixed-count founder-path refinement.

The observation model matches the assembly panel scorer: normalized genotype
likelihoods, a one-percent uniform mixture and a -2 log-likelihood floor.
Unlike its binned approximation, sample diplotypes may switch at every SNP.
Only a candidate-independent complete-site mask contributes evidence. Missing
founder alleles and unobserved sample calls are not invented or imputed here.

Symmetric genotype emissions and a uniform penalty for any diplotype change
allow exact K(K+1)/2 unordered states. This reduction is specific to this
scoring model, not the pipeline's homologue-specific ancestry HMMs.
The refinement penalty is length-independent; hierarchy panel selection keeps
its separate length-scaled cost. Neither cost is an inferred crossover rate.
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange


# Shared by refinement workspaces and standalone paired-exchange refits.
# Growing this cost with component length can prefer a poorer founder phase
# merely because it saves sample switches; keep the same cost at every level.
SWITCH_PENALTY = 20.0


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


def _dosage_table(haplotypes):
    """Prepare shared dosages only within the existing workspace RAM allowance."""
    from .site_kernels import prepare_dosages
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
    from .site_kernels import score_dosages
    dosages = _dosage_table(haplotypes)
    if dosages is None:
        return _score_panel_direct(haplotypes, evidence, complete, penalty, prepared)
    logs = prepare_log_evidence(evidence, complete) if prepared is None else prepared
    return score_dosages(dosages, logs, float(penalty))

