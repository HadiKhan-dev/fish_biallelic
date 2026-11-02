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

Owns: dimension-bounded proof-only count certificates.
Must not: read files, environment, live memory or clocks, or emit logs/warnings.
Method: WP_AS5 §4.11; AJ35.
Identity: L5 removes live-memory admission only; a proof never substitutes a different incumbent CountDecision.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from haplotype_reconstruction.contracts.roles import record_field

from . import criterion
from .parameters import PROOF_SLACK_BYTES, SHADOW_WORK_RATIO


@dataclass(frozen=True)
class FactorPlan:
    choices: np.ndarray = record_field("state", axes=("sum_C_b", "K"), units="1", dtype="int64")
    starts: np.ndarray = record_field("state", axes=("B+1",), units="1", dtype="int64")
    domain_size: int = record_field("state", axes=(), units="1", dtype="int64")
    local_products: tuple[int, ...] = record_field("state", axes=(), units="1", dtype="tuple[int, ...]")

    @property
    def bytes(self):
        return self.choices.nbytes + self.starts.nbytes


def factor_sizes(options, shape):
    """Cheap local/global work inspection, without enumerating any choices."""
    founders, blocks = shape
    sizes = tuple(math.prod(len(options[f * blocks + b]) for f in range(founders)) for b in range(blocks))
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


def _carrier_ceiling(count):
    """Outward floating upper bound for the ordinary participation ratio.

    Use the standard gamma_n error bound for positive reductions and basic
    arithmetic, deliberately enlarged for squaring/division. A normal positive
    squared-gain anchor (checked below) absorbs the absolute subnormal error.
    This guards the proof only: ordinary float n_eff can exceed n by a few ulps.
    The pipeline's finite log-score gains are far below overflow magnitudes.
    """
    eps = np.finfo(np.float64).eps
    gamma = (count + 8.0) * eps / (1.0 - (count + 8.0) * eps)
    return float(np.nextafter(count / (1.0 - gamma) ** 8, np.inf))


def _sum_interval(values, terms=None):
    """Enclose every ordinary floating reduction order of these finite terms.

    fsum fixes the proof's own reduction order; the enlarged gamma bound covers
    a caller's sum, products/division nearby and subnormal absolute error. This
    is a proof bound only, never a tolerance in the ordinary count criterion.
    """
    values = tuple(float(x) for x in values)
    count = len(values) if terms is None else int(terms)
    eps = np.finfo(np.float64).eps
    gamma = (count + 8.0) * eps / (1.0 - (count + 8.0) * eps)
    value = math.fsum(values)
    error = gamma * math.fsum(abs(x) for x in values) + count * np.finfo(np.float64).tiny
    return (float(np.nextafter(value - error, -np.inf)), float(np.nextafter(value + error, np.inf)))


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
    penalty = float(np.nextafter(_carrier_ceiling(samples) / (2.0 * samples), np.inf))
    rest_penalty = float(np.nextafter(_carrier_ceiling(samples - 1) / (2.0 * samples), np.inf))
    # Lowest sequence sum removes its largest entry; the greatest possible
    # omitted total sum removes its smallest. Cover every reduction order.
    point = float(np.nextafter(_sum_interval(q)[0] / cost - penalty, -np.inf))
    loso = float(np.nextafter(_sum_interval(q[:-1])[0] / cost - rest_penalty, -np.inf))
    total_upper = float(np.nextafter(_sum_interval(d)[1] / cost, np.inf))
    total_loso_upper = float(np.nextafter(_sum_interval(d[1:])[1] / cost, np.inf))
    if not (point > max(0.0, total_upper) and loso > max(0.0, total_loso_upper)):
        return None
    # Lower q-d in the original paired sample frame, then sort only for the
    # complete-permutation bound. Sorting raw GLs never enters a model decision.
    delta = np.sort(np.nextafter(lower - total_gain, -np.inf))[::-1]
    absolute = np.sort(np.abs(lower) + np.abs(total_gain))[::-1]
    patterns = {tuple(sorted(np.bincount(draw, minlength=samples).tolist())) for draw in index}
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
        gamma = (2.0 * samples + 16.0) * eps / (1.0 - (2.0 * samples + 16.0) * eps)
        net = float(np.nextafter(net - gamma * magnitude - 2.0 * samples * np.finfo(np.float64).tiny, -np.inf))
        gap = float(np.nextafter(net / cost - penalty, -np.inf))
        if not gap > 0.0:
            return None
        minimum_gap = min(minimum_gap, gap)
    return dict(
        sequence_margin_lower=point,
        sequence_loso_lower=loso,
        minimum_permutation_dominance_gap=float(minimum_gap),
        bootstrap_multiplicity_patterns=len(patterns),
    )


def total_dominance_certificate(
    larger, smaller, upper_shadow, cost, *, incumbent, config, seed_key, domain_size, cause
):
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
    family = criterion.ShadowDecisionFamily(
        larger, smaller, cost, kind=incumbent.kind, config=config, seed_key=seed_key
    )
    total = family.total
    universal = _permutation_dominance(lower, larger - smaller, cost, family.index)
    if universal is None:
        return None
    point_lower = universal["sequence_margin_lower"]
    loso_lower = universal["sequence_loso_lower"]
    # This stronger sufficient proof may decline a valid single-order proof.
    # Refusal changes verification metadata only, never the incumbent decision.
    # Universal dominance proves the original total test controls each combined
    # field. The original stream is retained to report its exact interval.
    draws = lower[family.index]
    sequence_lower = np.nextafter(draws.sum(axis=1) / cost - _carrier_ceiling(samples) / (2.0 * samples), -np.inf)
    tail = (1.0 - float(config.bootstrap_level)) / 2.0
    band = tuple(float(x) for x in np.quantile(family.total_replicates, [tail, 1.0 - tail]))
    ambiguous = bool(band[0] < 0.0 < band[1])
    if not (
        incumbent.total == total
        and incumbent.samples == samples
        and incumbent.cost == float(cost)
        and incumbent.seed == family.seed
        and incumbent.accept_larger == total.passes
        and incumbent.margin == total.margin
        and min(incumbent.total.loso_margin, incumbent.sequence.loso_margin) == total.loso_margin
        and incumbent.band == band
        and incumbent.ambiguous == ambiguous
    ):
        return None
    return dict(
        evaluation_kind="block_factor_total_dominance_proof",
        cause=cause,
        domain_size=int(domain_size),
        enumerated=0,
        evaluated_witnesses=1,
        computed_fields=["accept_larger", "margin", "combined_loso_margin", "band", "ambiguous"],
        uncomputed_fields=["sequence_test", "worst_sequence_loso_shadow"],
        kind=incumbent.kind,
        samples=samples,
        cost=float(cost),
        seed=family.seed,
        accept_larger=bool(total.passes),
        margin=total.margin,
        combined_loso_margin=total.loso_margin,
        band=list(band),
        ambiguous=ambiguous,
        sequence_margin_lower=point_lower,
        sequence_loso_lower=loso_lower,
        floating_guard="gamma_n_carrier_ceiling_and_all_sample_permutations",
        permutation_scope="every_ordering_of_original_bootstrap_cohort",
        minimum_permutation_dominance_gap=universal["minimum_permutation_dominance_gap"],
        bootstrap_multiplicity_patterns=universal["bootstrap_multiplicity_patterns"],
        sequence_diagnostics="evaluated_incumbent_only",
        minimum_replicate_dominance_gap=float(np.min(sequence_lower - family.total_replicates)),
    )


def attempt_certificate(
    options,
    shape,
    site_counts,
    larger,
    smaller,
    cost,
    *,
    incumbent,
    config,
    seed_key,
    upper_scores,
    scorer_samples,
    threads,
    full_site=False,
    max_table_bytes=None,
):
    """Bounded optional proof after an overflow's unchanged incumbent evaluation.

    upper_scores(plan) reads the caller's original evidence and returns the
    envelope on its original informative cohort. L5 attempts the proof exactly when the dimension budgets permit it.
    This annotation never changes the incumbent count decision.
    """
    sizes, domain = factor_sizes(options, shape)
    ordinary = sum(int(n) for n in site_counts)
    work = sum(int(n) * size for n, size in zip(site_counts, sizes, strict=False))
    audit = dict(
        attempted=False,
        domain_size=domain,
        maximum_local_product=max(sizes),
        local_products=sum(sizes),
        ordinary_units=ordinary,
        local_units=work,
        work_ratio=work / max(1, ordinary),
    )

    def result(proof, reason):
        audit.update(reason=reason, seconds=0.0)
        return proof, audit

    if max(sizes) > int(config.shadow_combinations) or work > SHADOW_WORK_RATIO * ordinary:
        return result(None, "local_work_budget")
    founders, blocks = shape
    states = founders * (founders + 1) // 2
    dosage_bytes = work * states if full_site else 0
    if max_table_bytes is not None and dosage_bytes > max_table_bytes:
        return result(None, "existing_pair_table_budget")
    plan_bytes = 8 * ((blocks + 1) + sum(sizes) * founders)
    scoring_bytes = dosage_bytes + (sum(sizes) + 1) * 8 + 24 * states * max(1, threads) + 8 * scorer_samples
    bootstrap_bytes = 64 * int(config.bootstrap_resamples) * len(larger) + 128 * len(larger)
    memory_bytes = plan_bytes + max(scoring_bytes, bootstrap_bytes) + PROOF_SLACK_BYTES
    audit.update(transient_bytes=memory_bytes, dosage_bytes=dosage_bytes)
    plan = factor_plan(options, shape)
    audit["attempted"] = True
    upper = upper_scores(plan)
    proof = total_dominance_certificate(
        larger,
        smaller,
        upper,
        cost,
        incumbent=incumbent,
        config=config,
        seed_key=seed_key,
        domain_size=domain,
        cause="nearest_shadow_budget",
    )
    return result(proof, "certified" if proof is not None else "criterion_not_dominated")


__all__ = ["FactorPlan", "factor_sizes", "factor_plan", "total_dominance_certificate", "attempt_certificate"]
