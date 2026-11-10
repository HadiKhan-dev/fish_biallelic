"""Founder-count decisions between two panels that differ by one founder row.

The criterion compares a larger panel X+ with a smaller panel X- (one row fewer) through
per-sample scores (founder refinement: each sample's robust max-path score F_s, refinement/objective.py; the
assembly join: its per-sample panel likelihood) and the per-founder cost c of
:func:`founder_cost`, c = scale x (mean sites per block / 200) x S x B. S counts the informative
samples only, the samples whose score can depend on the panel (the join's batch samples; in
founder refinement the samples with evidence at one of the component's objective sites): a sample whose
score is identically zero carries no evidence for any count decision, so it neither scales the
cost nor enters the bootstrap population. Every per-sample vector passed here has length S. Let
d_s = F_s(X+) - F_s(X-) and L = sum_s d_s / c (in units of c).

The effective-carrier criterion charges the per-sample cost c/S only to the founder's effective
carriers (a BIC-style rule would charge every founder c/2 for every sample, although a
founder's evidence accrues only on the samples that carry it): n_eff = (sum_s d_s+)^2 /
sum_s (d_s+)^2 with d_s+ = max(d_s, 0) (the participation ratio of the positive part). A
vector of per-sample differences passes iff m = L - n_eff / (2S) > 0 and the same margin with
the sample of largest d_s removed (S unchanged), m', is > 0. With n_eff = S the margin equals
BIC's L - 1/2.

:func:`decide` is the pipeline's one count rule: the larger panel is kept (deletion rejected) or
accepted (addition) iff the test passes on the total per-sample gain d AND on its sequence part
q_s = F_s(X+) - F_s(X+^sigma), where the shadow X+^sigma replaces the decided founder's private
sequence by the nearest rows of the panel it is compared with (:mod:`.geometry`), so switch
savings alone never justify a founder. The rule is the same for deletions and additions.

Ambiguity: individuals are resampled with replacement (a deterministic stream derived from
``seed_key``) and min(m_d, m_q) is recomputed; a call whose percentile interval contains zero
in its interior is ambiguous. The point decision stands.

Count cycles (:func:`pre_cycle_count`): when a search deletes a row that one of its own
additions created, the deletion stands (births then stop). A cycle whose closing deletion is
ambiguous and that began with a decided deletion is undecided: the data could not tell the two
counts apart, and the call is flagged.

The bound test (:func:`deletion_bound_excludes`) lets founder refinement skip a round whose decision AND
non-ambiguity optimistic scores already prove for every candidate, so a skipped round's
``ambiguous = False`` is a proven statement, not a default.

Owns: effective-carrier tests, bootstrap bands, cycle origin and safe count bounds.
Must not: read files, environment, live memory or clocks, or emit logs/warnings.
Method: docs/methods/assembly.md#aj30.
Invariant: strict zero tests use input sample order; one keyed PCG64 draw matrix is shared within a decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from haplotype_reconstruction.contracts.roles import record_field
from haplotype_reconstruction.core.random import bootstrap_seed, resampled_individuals

from .parameters import REFERENCE_SITES, TOP_SAMPLES, CountRuleParameters

CRITERION = "effective_carrier+sequence"


@dataclass(frozen=True)
class MarginTest:
    """The effective-carrier test on one per-sample difference vector (d or q).

    sum: sum of the differences (nats); n_eff: effective carriers; margin: m (units of c);
    loso_margin: m' (the largest sample removed, S unchanged); loso_sample: that sample;
    passes: m > 0 and m' > 0; positive_samples: samples with a positive difference;
    top_samples / top_differences / top_shares: the TOP_SAMPLES largest differences, their
    samples and their shares of the positive mass (diagnostic).
    """

    sum: float = record_field("state", axes=(), units="nats", dtype="float64")
    n_eff: float = record_field("state", axes=(), units="1", dtype="float64")
    margin: float = record_field("state", axes=(), units="1", dtype="float64")
    loso_margin: float = record_field("state", axes=(), units="1", dtype="float64")
    loso_sample: int = record_field("state", axes=(), units="1", dtype="int64")
    passes: bool = record_field("state", axes=(), units="1", dtype="bool")
    positive_samples: int = record_field("state", axes=(), units="1", dtype="int64")
    top_samples: tuple[int, ...] = record_field("diagnostic", axes=(), units="1", dtype="tuple[int, ...]")
    top_differences: tuple[float, ...] = record_field("diagnostic", axes=(), units="nats", dtype="tuple[float, ...]")
    top_shares: tuple[float, ...] = record_field("diagnostic", axes=(), units="1", dtype="tuple[float, ...]")


@dataclass(frozen=True)
class CountDecision:
    """Outcome of :func:`decide`.

    accept_larger: keep the larger panel (deletion rejected / addition accepted); this is the
        point decision (both tests pass).
    margin: min(m_d, m_q) in units of c (> 0 favours the larger panel).
    band: bootstrap percentile interval of that margin; ambiguous: band[0] < 0 < band[1].
    total, sequence: the tests on d and on q.
    samples, cost, seed: S, c and the bootstrap seed.
    """

    kind: str = record_field("state", axes=(), units="1", dtype="str")
    accept_larger: bool = record_field("state", axes=(), units="1", dtype="bool")
    margin: float = record_field("state", axes=(), units="1", dtype="float64")
    band: tuple[float, float] = record_field("state", axes=("2",), units="1", dtype="float64")
    ambiguous: bool = record_field("state", axes=(), units="1", dtype="bool")
    total: MarginTest = record_field("state", axes=(), units="1", dtype="MarginTest")
    sequence: MarginTest = record_field("state", axes=(), units="1", dtype="MarginTest")
    samples: int = record_field("state", axes=(), units="1", dtype="int64")
    cost: float = record_field("state", axes=(), units="nats", dtype="float64")
    seed: int = record_field("state", axes=(), units="1", dtype="int64")

    def record(self) -> dict:
        """Plain-JSON record of the decision (search records and reports)."""
        return dict(
            criterion=CRITERION,
            kind=self.kind,
            samples=self.samples,
            cost=self.cost,
            threshold_per_carrier=self.cost / (2.0 * self.samples),
            margin=self.margin,
            accept_larger=self.accept_larger,
            band=list(self.band),
            ambiguous=self.ambiguous,
            seed=self.seed,
            total=asdict(self.total),
            sequence=asdict(self.sequence),
        )


def founder_cost(site_counts, samples: int, scale: float) -> float:
    """Per-founder cost c = scale x (mean sites per block / 200) x S x B (nats).

    ``site_counts``: sites of each block the panel spans (B blocks); ``samples``: S, the
    informative samples (module docstring), the length of the per-sample vectors the decision
    reads. The per-carrier charge c / (2S) does not depend on S.
    """
    mean_sites = float(np.mean(np.asarray(site_counts, dtype=np.float64)))
    return scale * (mean_sites / REFERENCE_SITES) * samples * len(site_counts)


def effective_carriers(differences) -> float:
    """Participation ratio of the positive per-sample differences (0 if none is positive)."""
    positive = np.maximum(np.asarray(differences, np.float64), 0.0)
    square = float(np.sum(positive * positive))
    return float(np.sum(positive)) ** 2 / square if square > 0.0 else 0.0


def margin(differences, cost: float, samples: int) -> float:
    """The effective-carrier margin m = sum d / c - n_eff / (2S) of one difference vector."""
    return float(np.sum(differences)) / cost - effective_carriers(differences) / (2.0 * samples)


def passes(differences, cost: float, samples: int) -> bool:
    """m > 0 and m' > 0 (the largest difference removed, S unchanged)."""
    differences = np.asarray(differences, np.float64)
    top = int(np.argmax(differences))
    return margin(differences, cost, samples) > 0.0 and margin(np.delete(differences, top), cost, samples) > 0.0


_resampled_individuals = resampled_individuals


def _joint_bootstrap(total, sequence, cost, samples, resamples, level, seed):
    """Percentile interval of min(m_total, m_sequence) over the same resampled individuals."""
    index = _resampled_individuals(seed, resamples, len(total))
    values = None
    for differences in (total, sequence):
        draws = differences[index]
        net = draws.sum(axis=1) / cost
        positive = np.maximum(draws, 0.0)
        square = (positive * positive).sum(axis=1)
        mass = positive.sum(axis=1)
        carriers = np.divide(mass * mass, square, out=np.zeros_like(mass), where=square > 0.0)
        test = net - carriers / (2.0 * samples)
        values = test if values is None else np.minimum(values, test)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return float(low), float(high)


def _test(differences, cost, samples) -> MarginTest:
    top = int(np.argmax(differences))
    positive = np.maximum(differences, 0.0)
    positive_total = float(positive.sum())
    order = np.argsort(-differences, kind="stable")[:TOP_SAMPLES]
    value = margin(differences, cost, samples)
    loso = margin(np.delete(differences, top), cost, samples)
    return MarginTest(
        sum=float(differences.sum()),
        n_eff=effective_carriers(differences),
        margin=value,
        loso_margin=loso,
        loso_sample=top,
        passes=bool(value > 0.0 and loso > 0.0),
        positive_samples=int(np.count_nonzero(differences > 0.0)),
        top_samples=tuple(int(i) for i in order),
        top_differences=tuple(float(differences[i]) for i in order),
        top_shares=tuple(float(positive[i] / positive_total) if positive_total > 0.0 else 0.0 for i in order),
    )


def decide(
    per_sample_larger,
    per_sample_smaller,
    per_sample_shadow,
    cost: float,
    *,
    kind: str,
    config: CountRuleParameters,
    seed_key,
) -> CountDecision:
    """The pipeline's count decision: the effective-carrier test on the total gain AND on its
    sequence part.

    ``per_sample_shadow`` scores the larger panel with the decided path's cells that the smaller
    panel lacks replaced by the smaller panel's nearest rows; q = larger - shadow is the sequence
    part of the per-sample gain (zero for a path whose every cell another path uses). The larger
    panel is kept iff both tests pass (m > 0 and m' > 0 for d and for q); the band is that of
    min(m_d, m_q). ``kind`` is "deletion" (larger = incumbent) or "addition" (smaller =
    incumbent). Scores are float64 vectors of equal length S over the informative samples (the
    S of ``cost``); ``cost`` is c > 0.
    """
    larger = np.asarray(per_sample_larger, np.float64)
    smaller = np.asarray(per_sample_smaller, np.float64)
    shadow = np.asarray(per_sample_shadow, np.float64)
    samples = len(larger)
    total = _test(larger - smaller, cost, samples)
    sequence = _test(larger - shadow, cost, samples)
    seed = bootstrap_seed(seed_key)
    band = _joint_bootstrap(
        larger - smaller,
        larger - shadow,
        cost,
        samples,
        int(config.bootstrap_resamples),
        float(config.bootstrap_level),
        seed,
    )
    return CountDecision(
        kind=kind,
        accept_larger=bool(total.passes and sequence.passes),
        margin=float(min(total.margin, sequence.margin)),
        band=band,
        ambiguous=bool(band[0] < 0.0 < band[1]),
        total=total,
        sequence=sequence,
        samples=samples,
        cost=float(cost),
        seed=seed,
    )


class ShadowDecisionFamily:
    """Reuse one count comparison's total test and bootstrap draws across shadows.

    Construct only for a multi-shadow family sharing larger/smaller scores, cost,
    kind, configuration and seed. The caller still scores every represented shadow
    before calling ``decide``. Identical float64 score vectors reuse the immutable
    CountDecision; no approximation, alternative pruning or cross-family cache is
    involved. Drop this object after the family so its draw matrix and small score
    keys are released. ``decide`` above remains the ordinary/reference path.
    """

    def __init__(
        self, per_sample_larger, per_sample_smaller, cost: float, *, kind: str, config: CountRuleParameters, seed_key
    ):
        self.larger = np.asarray(per_sample_larger, np.float64)
        differences = self.larger - np.asarray(per_sample_smaller, np.float64)
        self.samples = len(self.larger)
        self.cost = float(cost)
        self.kind = kind
        self.seed = bootstrap_seed(seed_key)
        self.level = float(config.bootstrap_level)
        self.total = _test(differences, cost, self.samples)
        self.index = _resampled_individuals(self.seed, int(config.bootstrap_resamples), self.samples)
        self.total_replicates = self._replicates(differences)
        self._decisions: dict[bytes, CountDecision] = {}

    def _replicates(self, differences):
        # Keep the exact array shapes and arithmetic order of _joint_bootstrap.
        draws = differences[self.index]
        net = draws.sum(axis=1) / self.cost
        positive = np.maximum(draws, 0.0)
        square = (positive * positive).sum(axis=1)
        mass = positive.sum(axis=1)
        carriers = np.divide(mass * mass, square, out=np.zeros_like(mass), where=square > 0.0)
        return net - carriers / (2.0 * self.samples)

    def decide(self, per_sample_shadow) -> CountDecision:
        shadow = np.asarray(per_sample_shadow, np.float64)
        key = shadow.tobytes()
        found = self._decisions.get(key)
        if found is not None:
            return found
        differences = self.larger - shadow
        sequence = _test(differences, self.cost, self.samples)
        values = np.minimum(self.total_replicates, self._replicates(differences))
        tail = (1.0 - self.level) / 2.0
        low, high = np.quantile(values, [tail, 1.0 - tail])
        band = float(low), float(high)
        decision = CountDecision(
            kind=self.kind,
            accept_larger=bool(self.total.passes and sequence.passes),
            margin=float(min(self.total.margin, sequence.margin)),
            band=band,
            ambiguous=bool(band[0] < 0.0 < band[1]),
            total=self.total,
            sequence=sequence,
            samples=self.samples,
            cost=self.cost,
            seed=self.seed,
        )
        self._decisions[key] = decision
        return decision


def pre_cycle_count(history, larger: int, structural=frozenset()) -> int:
    """The founder count a count cycle started from (the join and founder refinement).

    ``history`` lists one search's founder count at its start and after every accepted count
    change, ending with the current count ``larger``; ``structural`` holds the indices j of the
    changes history[j - 1] -> history[j] that deleted a row without sequence of its own (always
    deleted: not a count decision). A deletion from ``larger`` of a row that an addition of the
    same search created closes a count cycle: rows were added and deleted again, each call made
    against a different panel, so the rule does not separate ``larger`` from ``larger - 1``
    there. The cycle began where the count last stood without falling below ``larger - 1``
    afterwards. If the count there is ``larger`` or more and then first fell to ``larger - 1`` by
    a count decision, the cycle began by deleting: that count is returned, and the callers flag
    the cycle as undecided when the closing deletion is ambiguous (its band contains zero).
    Otherwise (it began by adding, or the first fall removed a row without sequence, which never
    counted) ``larger - 1`` is returned. The deletion stands either way.
    """
    index = len(history) - 1
    while index > 0 and history[index - 1] >= larger - 1:
        index -= 1
    origin = int(history[index])
    if origin >= larger:
        for step in range(index + 1, len(history)):
            if history[step - 1] >= larger and history[step] == larger - 1:
                if step in structural:
                    return larger - 1
                break
    return origin


def _block_chunks(option_starts, options, width):
    """Consecutive block ranges [b0, b1) whose options times width stay near _CHUNK_ENTRIES (at least one block)."""
    blocks = len(option_starts)
    bounds = np.append(option_starts, options)
    b0 = 0
    while b0 < blocks:
        b1 = b0 + 1
        while b1 < blocks and (bounds[b1 + 1] - bounds[b0]) * width <= _CHUNK_ENTRIES:
            b1 += 1
        yield b0, b1, int(bounds[b0]), int(bounds[b1])
        b0 = b1


_CHUNK_ENTRIES = 1 << 21


def _summed_optimum(option_scores, option_starts, weights):
    """U(w) - sum_s w_s n_s for every row w of ``weights``: sum over blocks of the best option's weighted total."""
    total = np.zeros(len(weights))
    for b0, b1, o0, o1 in _block_chunks(option_starts, len(option_scores), len(weights)):
        weighted = option_scores[o0:o1] @ weights.T
        total += np.maximum.reduceat(weighted, option_starts[b0:b1] - o0, axis=0).sum(axis=0)
    return total


def deletion_bound_excludes(
    per_sample_score, option_scores, option_starts, normalisers, cost,
    *, tolerance, config: CountRuleParameters, seed_key
):
    """Whether optimistic K-1 scores prove that every deletion keeps the larger panel and that the
    call is not ambiguous. Returns (excluded, band_bound).

    Optimistic scores (founder refinement's count.option_scores). P_{K-1} is the set of panels with
    at most K-1 distinct prepared rows per block. Every Y in P_{K-1} fixes at every block b an
    option o_b(Y): a row Y lacks at a block with K rows, otherwise the block's only option, which
    is all its rows. The option is the same for every sample, and F_s(Y) <= sum_b sigma_s(b, o_b(Y)) + n_s
    for every sample s (count.option_scores' relaxation: each block is scored as a free panel of the
    option's rows, with every sample's chain started afresh). ``option_scores`` holds sigma for the
    informative samples, one row per option; block b owns rows option_starts[b] up to
    option_starts[b+1] (or the end); ``normalisers`` holds n_s. For any weights w >= 0:

        sum_s w_s F_s(Y) <= U(w) = sum_b max_o sum_s w_s sigma_s(b, o) + sum_s w_s n_s   for every Y.

    U(w) never exceeds sum_s w_s U_s, where U_s is the per-sample bound (each sample at its own best
    option). Every margin below needs only a weighted sum of d_s = F_s(X_K) - F_s(Y).

    Point decision. The test keeps a founder only if m > 0 and the leave-one-sample-out margin
    m' > 0. With n_eff <= S, m >= (sum_s F_s(X_K) - U(1)) / c - 1/2. In m' the removed sample t
    is the largest d_s, so it depends on Y; for every t,
    sum_{s != t} d_s >= sum_{s != t} F_s(X_K) - U(1 - e_t) and n_eff' <= S - 1, hence
    m' >= min_t [sum_{s != t} F_s(X_K) - U(1 - e_t)] / c - (S - 1) / (2S).

    The same proof covers the sequence part in founder refinement, whose q is an exact shadow
    difference. The shadow of a deletion of f uses at every block only rows that X_K minus f uses,
    so it is in P_{K-1}, with duplicated founders adding no emission. So the bounds on d also bound
    q for every candidate. A founder without a private cell has q = 0 = F_s(X_K) - F_s(X_K), and
    X_K is then itself in P_{K-1}, so the margins cannot pass and no round is skipped. (The join's
    near-identity shortcut is not a score, and no score bound sees it; refinement does not use it.)

    Band. Replicate r of the decision's bootstrap (``seed_key``: the key :func:`decide` would use,
    so the same individuals) draws sample s w_rs times (sum_s w_rs = S). For d and for q alike its
    margin is >= (sum_s w_rs F_s(X_K) - U(w_r)) / c - 1/2 = B_r, because its n_eff is at most its
    S entries. So min(m_d, m_q) >= B_r replicate by replicate, and the band's lower percentile is
    at least the same percentile of B (order statistics and their interpolation are monotone).
    band_bound is that percentile. The round is skipped only when it also exceeds the tolerance,
    so band[0] > 0: the call is proven not ambiguous for every candidate. Otherwise the round runs.

    ``per_sample_score`` holds F_s(X_K) (length S); ``tolerance`` (nats) covers the rounding of
    the computed scores; ``config`` gives the bootstrap's resamples and level. band_bound is None
    when the point bounds fail. The weighted totals are taken block chunk by block chunk, keeping
    memory near _CHUNK_ENTRIES floats for any K, sample count or block count.
    """
    score = np.asarray(per_sample_score, np.float64)
    sigma = np.asarray(option_scores, np.float64)
    starts = np.asarray(option_starts, np.int64)
    norm = np.asarray(normalisers, np.float64)
    samples = len(score)
    relative = tolerance / cost
    ones = np.ones((1, samples))
    whole = float(score.sum()) - float(_summed_optimum(sigma, starts, ones)[0] + norm.sum())
    if not whole / cost - 0.5 > relative:
        return False, None
    # U(1 - e_t) = sum_b max_o (A_b(o) - sigma_t(b, o)) + sum_{s != t} n_s, with A_b(o) the option's total.
    rest = np.zeros(samples)
    by_option = sigma.sum(axis=1)
    for b0, b1, o0, o1 in _block_chunks(starts, len(sigma), samples):
        reduced = by_option[o0:o1, None] - sigma[o0:o1]
        rest += np.maximum.reduceat(reduced, starts[b0:b1] - o0, axis=0).sum(axis=0)
    rest += norm.sum() - norm
    loso = float(((score.sum() - score) - rest).min())
    if not loso / cost - (samples - 1) / (2.0 * samples) > relative:
        return False, None
    index = _resampled_individuals(bootstrap_seed(seed_key), int(config.bootstrap_resamples), samples)
    replicates = len(index)
    offsets = np.repeat(np.arange(replicates) * samples, index.shape[1])
    weights = np.bincount(offsets + index.ravel(), minlength=replicates * samples).reshape(replicates, samples)
    weights = weights.astype(np.float64)
    replicate_bounds = (weights @ score - _summed_optimum(sigma, starts, weights) - weights @ norm) / cost - 0.5
    band_bound = float(np.quantile(replicate_bounds, (1.0 - float(config.bootstrap_level)) / 2.0))
    return bool(band_bound > relative), band_bound


__all__ = [
    "CRITERION",
    "MarginTest",
    "CountDecision",
    "founder_cost",
    "effective_carriers",
    "margin",
    "passes",
    "decide",
    "ShadowDecisionFamily",
    "pre_cycle_count",
    "deletion_bound_excludes",
]
