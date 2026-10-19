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


@njit(cache=True)
def _phase_sets_of_sample(calls, truth, components, undetermined, positions, spans):
    """One sample's released phase sets (release 2.2; evaluation only).

    calls, truth int8 [L, 2]; components int [L] (-1 between components); undetermined bool [L] (the displayed phase
    of the site is undetermined: family phase's frame_undetermined bit 1); positions int64 [L]; spans int64 [L]
    (output buffer). A phase set is a maximal run of sites of one component whose displayed phase is determined:
    a component change, a site outside components or an undetermined site ends the set, and the next determined
    site starts a new one (as the export breaks the VCF phase set there). Returns int64 [4]: allele errors aligned
    by one strand swap per phase set (undetermined called sites add their per-site minimum, i.e. genotype-level
    errors only), phase sets holding at least two called heterozygotes, called alleles at undetermined sites, and the
    number of spans written (bp from the first to the last called heterozygote of each such set)."""
    out = np.zeros(4, dtype=np.int64)
    direct = swapped = 0
    hets = 0
    first = last = 0
    current = -2               # component of the open set (-2: no open set)
    for site in range(calls.shape[0]):
        a, b = calls[site, 0], calls[site, 1]
        x, y = truth[site, 0], truth[site, 1]
        component = components[site]
        d = int(a >= 0 and a != x) + int(b >= 0 and b != y)
        w = int(a >= 0 and a != y) + int(b >= 0 and b != x)
        closes = component < 0 or undetermined[site] or (current != -2 and component != current)
        if closes and current != -2:
            out[0] += min(direct, swapped)
            if hets >= 2:
                out[1] += 1
                spans[out[3]] = last - first
                out[3] += 1
            direct = swapped = hets = 0
            current = -2
        if component < 0:
            continue
        if undetermined[site]:
            out[0] += min(d, w)
            out[2] += int(a >= 0) + int(b >= 0)
            continue
        if current == -2:
            current = component
        direct += d
        swapped += w
        if a >= 0 and b >= 0 and a != b:
            if hets == 0:
                first = positions[site]
            last = positions[site]
            hets += 1
    if current != -2:
        out[0] += min(direct, swapped)
        if hets >= 2:
            out[1] += 1
            spans[out[3]] = last - first
            out[3] += 1
    return out


def phase_set_counts(calls, truth, components, undetermined, positions):
    """Released-phase-set metrics of one chromosome (release 2.2): (ps_aligned_allele_errors, phase_sets,
    undetermined_called_alleles, spans int64[...]). ``undetermined`` bool [N, L] or None (2.1 records: phase sets are
    the components, and ps_aligned_allele_errors equals component_aligned_allele_errors)."""
    samples, sites, _ = calls.shape
    if undetermined is None:
        undetermined = np.zeros((samples, sites), dtype=np.bool_)
    positions = np.ascontiguousarray(positions, dtype=np.int64)
    components = np.ascontiguousarray(components)
    buffer = np.empty(sites + 1, dtype=np.int64)
    errors = sets = loose = 0
    spans = []
    for sample in range(samples):
        result = _phase_sets_of_sample(np.ascontiguousarray(calls[sample]), np.ascontiguousarray(truth[sample]),
                                       components, np.ascontiguousarray(undetermined[sample], dtype=np.bool_),
                                       positions, buffer)
        errors += int(result[0])
        sets += int(result[1])
        loose += int(result[2])
        spans.append(buffer[:int(result[3])].copy())
    return errors, sets, loose, (np.concatenate(spans) if spans else np.zeros(0, dtype=np.int64))


def n50(lengths) -> int:
    """The N50 of a set of lengths (the length at which the longest ones reach half the total); 0 if empty."""
    values = np.sort(np.asarray(lengths, dtype=np.int64))[::-1]
    if len(values) == 0 or values.sum() == 0:
        return 0
    cumulative = np.cumsum(values)
    return int(values[int(np.searchsorted(cumulative, cumulative[-1] / 2.0))])
