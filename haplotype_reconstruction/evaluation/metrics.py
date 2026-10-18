"""Phase metrics of the final calls against the simulated truth (evaluation only).

Both kernels take, for one chromosome, the final phased calls int8 [N, L, 2] (-1 unknown, 0 REF, 1 ALT), the
true alleles int8 [N, L, 2] (maternal, paternal) and the supported component of every site int [L] (-1 between
components). Samples are counted in parallel and summed in sample order: integer counts, independent of the
thread count.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def phase_counts(calls, truth, components):
    """Count called coverage and phase errors without crossing unsupported links.

    Returns int64 [7]: called_alleles, called_genotypes, genotype_errors, eligible_heterozygotes,
    phase_comparisons, phase_switch_errors, component_aligned_allele_errors.

    Compare consecutive true heterozygotes only when both calls have the right
    genotype and belong to the same supported component. An uncalled/incorrect
    intervening heterozygote breaks the comparison. Allele mismatch counts are
    aligned by one arbitrary strand swap per sample and component.
    """
    samples, sites, _ = calls.shape
    result = np.zeros((samples, 7), dtype=np.int64)
    component_count = max(0, int(components.max()) + 1)
    for sample in prange(samples):
        direct = np.zeros(component_count, dtype=np.int64)
        swapped = np.zeros(component_count, dtype=np.int64)
        previous_valid = False
        previous_component = -1
        previous_flip = False
        for site in range(sites):
            a, b = calls[sample, site]
            x, y = truth[sample, site]
            component = components[site]
            result[sample, 0] += int(a >= 0) + int(b >= 0)
            complete = a >= 0 and b >= 0
            result[sample, 1] += int(complete)
            correct = complete and a + b == x + y
            result[sample, 2] += int(complete and not correct)
            if component >= 0:
                direct[component] += int(a >= 0 and a != x) + int(b >= 0 and b != y)
                swapped[component] += int(a >= 0 and a != y) + int(b >= 0 and b != x)
            if x != y:
                valid = correct and component >= 0
                result[sample, 3] += int(valid)
                flip = a != x
                if valid and previous_valid and component == previous_component:
                    result[sample, 4] += 1
                    result[sample, 5] += int(flip != previous_flip)
                previous_valid = valid
                previous_component = component
                previous_flip = flip
        for component in range(component_count):
            result[sample, 6] += min(direct[component], swapped[component])
    return result.sum(axis=0)


@njit(cache=True, parallel=True)
def long_switch_counts(calls, truth, components, minimum_runs):
    """Count long-range orientation changes of true heterozygotes, per run threshold.

    Returns int64 [R] for the R thresholds of ``minimum_runs`` (int64).

    Orientation runs are formed over correct-genotype true heterozygotes of a
    supported component, skipping (bridging) uncalled and wrong-genotype ones,
    so one bad call at a switch point does not hide the switch. Runs break at
    orientation and component changes. A change within one component counts
    for threshold t when both adjacent runs hold at least t heterozygotes;
    isolated flips and short flipped stretches are not long switches.
    """
    samples, sites, _ = calls.shape
    result = np.zeros((samples, len(minimum_runs)), dtype=np.int64)
    for sample in prange(samples):
        # left: length of the run before the current one when an orientation
        # change (not a component change) separates them; 0 otherwise.
        left = length = 0
        run_flip = False
        run_component = -1
        for site in range(sites):
            a, b = calls[sample, site]
            x, y = truth[sample, site]
            component = components[site]
            if x == y or component < 0 or a < 0 or b < 0 or a + b != x + y:
                continue
            flip = a != x
            if length and component == run_component and flip == run_flip:
                length += 1
                continue
            if left:
                shorter = min(left, length)
                for index in range(len(minimum_runs)):
                    result[sample, index] += int(shorter >= minimum_runs[index])
            left = length if length and component == run_component else 0
            length = 1
            run_flip = flip
            run_component = component
        if left:
            shorter = min(left, length)
            for index in range(len(minimum_runs)):
                result[sample, index] += int(shorter >= minimum_runs[index])
    return result.sum(axis=0)
