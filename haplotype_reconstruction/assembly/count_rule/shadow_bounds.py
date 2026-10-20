"""Independent-block shadow envelopes and proof-only count certificates.

Refinement uses this proof after its enumeration budget is exceeded. A global
shadow chooses one allowed local row per
(founder, block); choices cannot change inside a block. Maximizing block transfer
messages over its LOCAL products gives the exact per-sample maximum over all global
shadows. Different samples may select different maximizing shadows. This is a bound
on the shared-panel criterion, never a representative shared panel.

Kernels preserve the ordinary forward operation order, skipped-site mask and switch
penalty, with no fastmath. Callers provide the original scoring frame and manage the
Numba thread budget. Local products, unlike the global product, are materialized:
inspect factor_sizes and bound that work before constructing a plan. No production
budget or altered count rule is selected here.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import numpy as np
from numba import njit, prange

from . import criterion
from ..join.objective import max_range


@dataclass(frozen=True)
class FactorPlan:
    choices: np.ndarray       # sum_b C_b x K local row selections
    starts: np.ndarray        # B+1 offsets into choices
    domain_size: int          # product_b C_b, Python integer
    local_products: tuple[int, ...]

    @property
    def bytes(self):
        return self.choices.nbytes + self.starts.nbytes


def factor_sizes(options, shape):
    """Cheap local/global work inspection, without enumerating any choices."""
    founders, blocks = shape
    sizes = tuple(math.prod(len(options[f * blocks + b]) for f in range(founders))
                  for b in range(blocks))
    return sizes, math.prod(sizes)


def factor_plan(options, shape):
    """Materialize only local products after the caller accepts their work size."""
    founders, blocks = shape
    sizes, domain = factor_sizes(options, shape)
    starts = np.concatenate(([0], np.cumsum(sizes, dtype=np.int64)))
    choices = np.empty((int(starts[-1]), founders), np.int64)
    for block in range(blocks):
        factors = [options[f * blocks + block] for f in range(founders)]
        for index, local in enumerate(itertools.product(*factors), int(starts[block])):
            choices[index] = local
    return FactorPlan(choices, starts, domain, sizes)


def bounded_factor_plan(options, shape, site_counts, *, max_local_product, max_work_ratio):
    """Decline before allocation if a local product or weighted work exceeds caller caps.

    Caps restrict this optional proof computation only; None leaves the ordinary
    incumbent/unknown path intact. Site counts are the existing full-site counts
    (refinement) or bin counts (join), without changing their scoring layout.
    """
    sizes, _ = factor_sizes(options, shape)
    work = sum(size * int(sites) for size, sites in zip(sizes, site_counts))
    ordinary = sum(int(sites) for sites in site_counts)
    if max(sizes) > max_local_product or work > max_work_ratio * ordinary:
        return None
    return factor_plan(options, shape)


@njit(cache=True, nogil=True)
def _dosages(leaves, site_offsets, choices, starts):
    founders = choices.shape[1]
    states = founders * (founders + 1) // 2
    offsets = np.zeros(len(choices) + 1, np.int64)
    for block in range(len(starts) - 1):
        sites = site_offsets[block + 1] - site_offsets[block]
        for choice in range(starts[block], starts[block + 1]):
            offsets[choice + 1] = offsets[choice] + sites
    dosages = np.empty((offsets[-1], states), np.int8)
    for block in range(len(starts) - 1):
        leaf = leaves[block]
        for choice in range(starts[block], starts[block + 1]):
            for site in range(leaf.shape[1]):
                state = 0
                for first in range(founders):
                    a = leaf[choices[choice, first], site]
                    for second in range(first, founders):
                        dosages[offsets[choice] + site, state] = a + leaf[choices[choice, second], site]
                        state += 1
    return dosages, offsets


@njit(cache=True, parallel=True, nogil=True)
def _full_site(logs, site_offsets, dosages, dose_offsets, starts, penalty, center):
    samples, states = len(logs), dosages.shape[1]
    result = np.empty(samples, np.float64)
    for sample in prange(samples):
        row = np.zeros(states)
        trial = np.empty(states)
        target = np.empty(states)
        for block in range(len(starts) - 1):
            target[:] = -np.inf
            lo, hi = site_offsets[block], site_offsets[block + 1]
            for choice in range(starts[block], starts[block + 1]):
                trial[:] = row
                for site in range(lo, hi):
                    p0, p1, p2 = logs[sample, site]
                    if p0 == 0. and p1 == 0. and p2 == 0.:
                        continue
                    e0, e1, e2 = p0 - center, p1 - center, p2 - center
                    switch = max_range(trial, 0, states, -np.inf) - penalty
                    index = dose_offsets[choice] + site - lo
                    for state in range(states):
                        dosage = dosages[index, state]
                        emit = e0 if dosage == 0 else (e1 if dosage == 1 else e2)
                        trial[state] = max(trial[state], switch) + emit
                for state in range(states):
                    target[state] = max(target[state], trial[state])
            row[:] = target
        result[sample] = max_range(row, 0, states, -np.inf)
    return result


def full_site_envelope(leaves, site_offsets, logs, plan, penalty, center):
    """Exact per-sample envelope; transient dosage table size is local site work x Q."""
    dosages, dose_offsets = _dosages(leaves, site_offsets, plan.choices, plan.starts)
    return _full_site(logs, site_offsets, dosages, dose_offsets, plan.starts, penalty, center)


@njit(cache=True, parallel=True, nogil=True)
def _join(flat, block_start, block_haps, block_bins, choices, starts, penalty, sample_ids):
    founders = choices.shape[1]
    states = founders * (founders + 1) // 2
    result = np.empty(len(sample_ids), np.float64)
    for index in prange(len(sample_ids)):
        sample = sample_ids[index]
        row = np.zeros(states)
        trial = np.empty(states)
        target = np.empty(states)
        for block in range(len(starts) - 1):
            target[:] = -np.inf
            local_haps, bins = block_haps[block], block_bins[block]
            base = block_start[block] + sample * local_haps * local_haps * bins
            for choice in range(starts[block], starts[block + 1]):
                trial[:] = row
                for column in range(bins):
                    switch = max_range(trial, 0, states, -np.inf) - penalty
                    state = 0
                    for first in range(founders):
                        a = choices[choice, first]
                        for second in range(first, founders):
                            b = choices[choice, second]
                            emit = flat[base + (a * local_haps + b) * bins + column]
                            trial[state] = max(trial[state], switch) + emit
                            state += 1
                for state in range(states):
                    target[state] = max(target[state], trial[state])
            row[:] = target
        result[index] = max_range(row, 0, states, -np.inf)
    return result


def join_envelope(layout, plan, penalty, sample_ids):
    """Exact join envelope on its unchanged binned layout and sample cohort."""
    return _join(*layout.arrays(), plan.choices, plan.starts, penalty,
                 np.asarray(sample_ids, np.int64))



def _carrier_ceiling(count):
    """Outward floating upper bound for the ordinary participation ratio.

    Use the standard gamma_n error bound for positive reductions and basic
    arithmetic, deliberately enlarged for squaring/division. A normal positive
    squared-gain anchor (checked below) absorbs the absolute subnormal error.
    This guards the proof only: ordinary float n_eff can exceed n by a few ulps.
    The pipeline's finite log-score gains are far below overflow magnitudes.
    """
    eps = np.finfo(np.float64).eps
    gamma = (count + 8.) * eps / (1. - (count + 8.) * eps)
    return float(np.nextafter(count / (1. - gamma) ** 8, np.inf))


def _gain_bounds(lower, cost):
    samples = len(lower)
    anchor = np.sqrt(np.finfo(np.float64).tiny * samples)
    if samples < 2 or np.partition(lower, -2)[-2] < anchor:
        return None
    point = lower.sum() / cost - _carrier_ceiling(samples) / (2. * samples)
    # The actual omitted index can differ from argmax(lower). Every remaining
    # vector dominates lower with that same index omitted, including NumPy's
    # reduction order. Taking all indices avoids cancellation/order assumptions.
    loso_net = min(float(np.sum(np.delete(lower, i))) for i in range(samples)) / cost
    loso = loso_net - _carrier_ceiling(samples - 1) / (2. * samples)
    return float(np.nextafter(point, -np.inf)), float(np.nextafter(loso, -np.inf))


def _sum_interval(values, terms=None):
    """Enclose every ordinary floating reduction order of these finite terms.

    fsum fixes the proof's own reduction order; the enlarged gamma bound covers
    a caller's sum, products/division nearby and subnormal absolute error. This
    is a proof bound only, never a tolerance in the ordinary count criterion.
    """
    values = tuple(float(x) for x in values)
    count = len(values) if terms is None else int(terms)
    eps = np.finfo(np.float64).eps
    gamma = (count + 8.) * eps / (1. - (count + 8.) * eps)
    value = math.fsum(values)
    error = gamma * math.fsum(abs(x) for x in values) + count * np.finfo(np.float64).tiny
    return (float(np.nextafter(value - error, -np.inf)),
            float(np.nextafter(value + error, np.inf)))


def _permutation_dominance(lower, total_gain, cost, index):
    """Sufficient dominance for every ordering of the original bootstrap cohort.

    A bootstrap draw fixes sample multiplicities, not their pairing with the
    biological samples. Rearrangement bounds the worst pairing over the complete
    permutation family. No resample, sample, score or count test is changed.
    """
    samples = len(lower)
    q = np.sort(np.asarray(lower, np.float64))
    d = np.sort(np.asarray(total_gain, np.float64))
    anchor = np.sqrt(np.finfo(np.float64).tiny * samples)
    if samples < 2 or q[-2] < anchor:
        return None
    penalty = float(np.nextafter(_carrier_ceiling(samples) / (2. * samples), np.inf))
    rest_penalty = float(np.nextafter(_carrier_ceiling(samples - 1) / (2. * samples), np.inf))
    # Lowest sequence sum removes its largest entry; the greatest possible
    # omitted total sum removes its smallest. Cover every reduction order.
    point = float(np.nextafter(_sum_interval(q)[0] / cost - penalty, -np.inf))
    loso = float(np.nextafter(_sum_interval(q[:-1])[0] / cost - rest_penalty, -np.inf))
    total_upper = float(np.nextafter(_sum_interval(d)[1] / cost, np.inf))
    total_loso_upper = float(np.nextafter(_sum_interval(d[1:])[1] / cost, np.inf))
    if not (point > max(0., total_upper) and loso > max(0., total_loso_upper)):
        return None
    # Lower q-d in the original paired sample frame, then sort only for the
    # complete-permutation bound. Sorting raw GLs never enters a model decision.
    delta = np.sort(np.nextafter(lower - total_gain, -np.inf))[::-1]
    absolute = np.sort(np.abs(lower) + np.abs(total_gain))[::-1]
    patterns = {tuple(sorted(np.bincount(draw, minlength=samples).tolist()))
                for draw in index}
    minimum_gap = np.inf
    for pattern in sorted(patterns):
        distinct = sum(weight > 0 for weight in pattern)
        # Any distinct samples can be selected. Even the smallest such subset
        # must have a normal positive gain to justify the carrier ceiling.
        if distinct == 0 or q[distinct - 1] < anchor:
            return None
        weights = np.asarray(pattern, np.float64)
        # The product error is covered together with both ordinary q/d sums.
        net = _sum_interval(weights * delta, terms=samples)[0]
        magnitude = _sum_interval(weights[::-1] * absolute, terms=samples)[1]
        eps = np.finfo(np.float64).eps
        gamma = (2. * samples + 16.) * eps / (1. - (2. * samples + 16.) * eps)
        net = float(np.nextafter(net - gamma * magnitude
                                 - 2. * samples * np.finfo(np.float64).tiny, -np.inf))
        gap = float(np.nextafter(net / cost - penalty, -np.inf))
        if not gap > 0.:
            return None
        minimum_gap = min(minimum_gap, gap)
    return dict(sequence_margin_lower=point, sequence_loso_lower=loso,
                minimum_permutation_dominance_gap=float(minimum_gap),
                bootstrap_multiplicity_patterns=len(patterns))


def total_dominance_certificate(larger, smaller, upper_shadow, cost, *, incumbent,
                                config, seed_key, domain_size, cause):
    """A separate proof record, or None; never a substitute CountDecision.

    U bounds every shadow score, so q >= larger-U. n_eff(q)<=S bounds the
    sequence margin by sum(q_lower)/c-1/2; dropping its largest entry similarly
    bounds LOSO. Pairing the lower gains with every possible permutation of each original
    bootstrap draw proves total dominance independently of input sample order.
    Require all combined fields to equal the evaluated incumbent. Sequence
    MarginTest fields and worst-shadow identity remain explicitly uncomputed.
    Failure to prove anything returns None and is never evidence for a count.
    """
    larger = np.asarray(larger, np.float64)
    smaller = np.asarray(smaller, np.float64)
    lower = larger - np.asarray(upper_shadow, np.float64)
    samples = len(larger)
    family = criterion.ShadowDecisionFamily(larger, smaller, cost, kind=incumbent.kind,
                                           config=config, seed_key=seed_key)
    total = family.total
    universal = _permutation_dominance(lower, larger - smaller, cost, family.index)
    if universal is None:
        return None
    point_lower = universal['sequence_margin_lower']
    loso_lower = universal['sequence_loso_lower']
    # This stronger sufficient proof may decline a valid single-order proof.
    # Refusal changes verification metadata only, never the incumbent decision.
    # Universal dominance proves the original total test controls each combined
    # field. The original stream is retained to report its exact interval.
    draws = lower[family.index]
    sequence_lower = np.nextafter(draws.sum(axis=1) / cost
                                  - _carrier_ceiling(samples) / (2. * samples), -np.inf)
    tail = (1. - float(config.bootstrap_level)) / 2.
    band = tuple(float(x) for x in np.quantile(family.total_replicates, [tail, 1. - tail]))
    ambiguous = bool(band[0] < 0. < band[1])
    if not (incumbent.total == total and incumbent.samples == samples
            and incumbent.cost == float(cost) and incumbent.seed == family.seed
            and incumbent.accept_larger == total.passes and incumbent.margin == total.margin
            and min(incumbent.total.loso_margin, incumbent.sequence.loso_margin) == total.loso_margin
            and incumbent.band == band and incumbent.ambiguous == ambiguous):
        return None
    return dict(evaluation_kind='block_factor_total_dominance_proof', cause=cause,
                domain_size=int(domain_size), enumerated=0, evaluated_witnesses=1,
                computed_fields=['accept_larger', 'margin', 'combined_loso_margin', 'band', 'ambiguous'],
                uncomputed_fields=['sequence_test', 'worst_sequence_loso_shadow'],
                kind=incumbent.kind, samples=samples, cost=float(cost), seed=family.seed,
                accept_larger=bool(total.passes), margin=total.margin,
                combined_loso_margin=total.loso_margin, band=list(band), ambiguous=ambiguous,
                sequence_margin_lower=point_lower, sequence_loso_lower=loso_lower,
                floating_guard='gamma_n_carrier_ceiling_and_all_sample_permutations',
                permutation_scope='every_ordering_of_original_bootstrap_cohort',
                minimum_permutation_dominance_gap=universal['minimum_permutation_dominance_gap'],
                bootstrap_multiplicity_patterns=universal['bootstrap_multiplicity_patterns'],
                sequence_diagnostics='evaluated_incumbent_only',
                minimum_replicate_dominance_gap=float(np.min(sequence_lower-family.total_replicates)))


def attempt_certificate(options, shape, site_counts, larger, smaller, cost, *, incumbent,
                        config, seed_key, upper_scores, scorer_samples, threads,
                        full_site=False, max_table_bytes=None):
    """Bounded optional proof after an overflow's unchanged incumbent evaluation.

    upper_scores(plan) reads the caller's original evidence and returns the
    envelope on its original informative cohort. Memory admission never waits;
    failure is computational unknown, never a different count decision.
    """
    import time
    from ... import execution
    begun = time.perf_counter()
    sizes, domain = factor_sizes(options, shape)
    ordinary = sum(int(n) for n in site_counts)
    work = sum(int(n) * size for n, size in zip(site_counts, sizes))
    audit = dict(attempted=False, domain_size=domain, maximum_local_product=max(sizes),
                 local_products=sum(sizes), ordinary_units=ordinary, local_units=work,
                 work_ratio=work / max(1, ordinary))
    def result(proof, reason):
        audit.update(reason=reason, seconds=time.perf_counter() - begun)
        return proof, audit
    if max(sizes) > int(config.shadow_combinations) or work > 4 * ordinary:
        return result(None, 'local_work_budget')
    founders, blocks = shape
    states = founders * (founders + 1) // 2
    dosage_bytes = work * states if full_site else 0
    if max_table_bytes is not None and dosage_bytes > max_table_bytes:
        return result(None, 'existing_pair_table_budget')
    plan_bytes = 8 * ((blocks + 1) + sum(sizes) * founders)
    scoring_bytes = dosage_bytes + (sum(sizes) + 1) * 8 + 24 * states * max(1, threads) + 8 * scorer_samples
    bootstrap_bytes = 64 * int(config.bootstrap_resamples) * len(larger) + 128 * len(larger)
    memory_bytes = plan_bytes + max(scoring_bytes, bootstrap_bytes) + (1 << 16)
    audit.update(transient_bytes=memory_bytes, dosage_bytes=dosage_bytes)
    governor = execution.memory_governor()
    live = execution.available_memory_bytes()
    if (memory_bytes > governor.limit - governor.reserved
            or (live is not None and memory_bytes > live - governor.headroom)):
        # Optional work must never invoke the governor's run-alone progress
        # exception; declining is safe even if a parent envelope could lend.
        return result(None, 'memory_budget')
    request = governor.request(memory_bytes, label='assembly shadow proof')
    try:
        if not request.poll():
            return result(None, 'memory_admission')
        plan = factor_plan(options, shape)
        audit['attempted'] = True
        upper = upper_scores(plan)
        proof = total_dominance_certificate(larger, smaller, upper, cost, incumbent=incumbent,
                                            config=config, seed_key=seed_key,
                                            domain_size=domain, cause='nearest_shadow_budget')
        return result(proof, 'certified' if proof is not None else 'criterion_not_dominated')
    finally:
        request.release()
