"""Founder-count decisions between two panels that differ by one founder row.

The criterion compares a larger panel X+ with a smaller panel X- (one row fewer) through
per-sample scores (founder refinement: each sample's robust max-path score F_s, founder_refinement/objective.py; the
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

The bound tests (:func:`deletion_bound_excludes`, :func:`addition_bound_excludes`,
:func:`addition_per_sample_bound_excludes`) let founder refinement skip a round whose decision AND
non-ambiguity optimistic scores already prove for every candidate, so a skipped round's
``ambiguous = False`` is a proven statement, not a default.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from .config import CountRuleConfig

CRITERION = "effective_carrier+sequence"
KINDS = ("deletion", "addition")
TOP_SAMPLES = 5   # samples listed per test in a decision record (diagnostic)


@dataclass(frozen=True)
class MarginTest:
    """The effective-carrier test on one per-sample difference vector (d or q).

    sum: sum of the differences (nats); n_eff: effective carriers; margin: m (units of c);
    loso_margin: m' (the largest sample removed, S unchanged); loso_sample: that sample;
    passes: m > 0 and m' > 0; positive_samples: samples with a positive difference;
    top_samples / top_differences / top_shares: the TOP_SAMPLES largest differences, their
    samples and their shares of the positive mass (diagnostic).
    """

    sum: float
    n_eff: float
    margin: float
    loso_margin: float
    loso_sample: int
    passes: bool
    positive_samples: int
    top_samples: tuple[int, ...]
    top_differences: tuple[float, ...]
    top_shares: tuple[float, ...]


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

    kind: str
    accept_larger: bool
    margin: float
    band: tuple[float, float]
    ambiguous: bool
    total: MarginTest
    sequence: MarginTest
    samples: int
    cost: float
    seed: int

    def record(self) -> dict:
        """Plain-JSON record of the decision (search records and reports)."""
        return dict(criterion=CRITERION, kind=self.kind, samples=self.samples, cost=self.cost,
                    threshold_per_carrier=self.cost / (2. * self.samples), margin=self.margin,
                    accept_larger=self.accept_larger, band=list(self.band),
                    ambiguous=self.ambiguous, seed=self.seed, total=asdict(self.total),
                    sequence=asdict(self.sequence))


def founder_cost(site_counts, samples: int, scale: float) -> float:
    """Per-founder cost c = scale x (mean sites per block / 200) x S x B (nats).

    ``site_counts``: sites of each block the panel spans (B blocks); ``samples``: S, the
    informative samples (module docstring), the length of the per-sample vectors the decision
    reads. The per-carrier charge c / (2S) does not depend on S.
    """
    mean_sites = float(np.mean(np.asarray(site_counts, dtype=np.float64)))
    return scale * (mean_sites / 200.0) * samples * len(site_counts)


def effective_carriers(differences) -> float:
    """Participation ratio of the positive per-sample differences (0 if none is positive)."""
    positive = np.maximum(np.asarray(differences, np.float64), 0.)
    square = float(np.sum(positive * positive))
    return float(np.sum(positive)) ** 2 / square if square > 0. else 0.


def margin(differences, cost: float, samples: int) -> float:
    """The effective-carrier margin m = sum d / c - n_eff / (2S) of one difference vector."""
    return float(np.sum(differences)) / cost - effective_carriers(differences) / (2. * samples)


def passes(differences, cost: float, samples: int) -> bool:
    """m > 0 and m' > 0 (the largest difference removed, S unchanged)."""
    differences = np.asarray(differences, np.float64)
    top = int(np.argmax(differences))
    return (margin(differences, cost, samples) > 0.
            and margin(np.delete(differences, top), cost, samples) > 0.)


def bootstrap_seed(seed_key) -> int:
    """Deterministic bootstrap seed of a decision from its JSON seed key."""
    text = json.dumps(seed_key, sort_keys=True, default=str)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def _resampled_individuals(seed, resamples, samples):
    """int64[resamples, samples]: the individuals each bootstrap replicate draws (with replacement)
    from the stream ``seed``. A decision's band and its replicate-wise bound
    (:func:`deletion_bound_excludes`) draw the same replicates from the same seed."""
    return np.random.default_rng(seed).integers(0, samples, size=(resamples, samples))


def _joint_bootstrap(total, sequence, cost, samples, resamples, level, seed):
    """Percentile interval of min(m_total, m_sequence) over the same resampled individuals."""
    index = _resampled_individuals(seed, resamples, len(total))
    values = None
    for differences in (total, sequence):
        draws = differences[index]
        net = draws.sum(axis=1) / cost
        positive = np.maximum(draws, 0.)
        square = (positive * positive).sum(axis=1)
        mass = positive.sum(axis=1)
        carriers = np.divide(mass * mass, square, out=np.zeros_like(mass), where=square > 0.)
        test = net - carriers / (2. * samples)
        values = test if values is None else np.minimum(values, test)
    tail = (1. - level) / 2.
    low, high = np.quantile(values, [tail, 1. - tail])
    return float(low), float(high)


def _test(differences, cost, samples) -> MarginTest:
    top = int(np.argmax(differences))
    positive = np.maximum(differences, 0.)
    positive_total = float(positive.sum())
    order = np.argsort(-differences, kind="stable")[:TOP_SAMPLES]
    value = margin(differences, cost, samples)
    loso = margin(np.delete(differences, top), cost, samples)
    return MarginTest(
        sum=float(differences.sum()), n_eff=effective_carriers(differences), margin=value,
        loso_margin=loso, loso_sample=top, passes=bool(value > 0. and loso > 0.),
        positive_samples=int(np.count_nonzero(differences > 0.)),
        top_samples=tuple(int(i) for i in order),
        top_differences=tuple(float(differences[i]) for i in order),
        top_shares=tuple(float(positive[i] / positive_total) if positive_total > 0. else 0.
                         for i in order))


def decide(per_sample_larger, per_sample_smaller, per_sample_shadow, cost: float, *, kind: str,
           config: CountRuleConfig, seed_key) -> CountDecision:
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
    band = _joint_bootstrap(larger - smaller, larger - shadow, cost, samples,
                            int(config.bootstrap_resamples), float(config.bootstrap_level), seed)
    return CountDecision(
        kind=kind, accept_larger=bool(total.passes and sequence.passes),
        margin=float(min(total.margin, sequence.margin)), band=band,
        ambiguous=bool(band[0] < 0. < band[1]), total=total, sequence=sequence,
        samples=samples, cost=float(cost), seed=seed)


class ShadowDecisionFamily:
    """Reuse one count comparison's total test and bootstrap draws across shadows.

    Construct only for a multi-shadow family sharing larger/smaller scores, cost,
    kind, configuration and seed. The caller still scores every represented shadow
    before calling ``decide``. Identical float64 score vectors reuse the immutable
    CountDecision; no approximation, alternative pruning or cross-family cache is
    involved. Drop this object after the family so its draw matrix and small score
    keys are released. ``decide`` above remains the ordinary/reference path.
    """

    def __init__(self, per_sample_larger, per_sample_smaller, cost: float, *, kind: str,
                 config: CountRuleConfig, seed_key):
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
        self._decisions = {}

    def _replicates(self, differences):
        # Keep the exact array shapes and arithmetic order of _joint_bootstrap.
        draws = differences[self.index]
        net = draws.sum(axis=1) / self.cost
        positive = np.maximum(draws, 0.)
        square = (positive * positive).sum(axis=1)
        mass = positive.sum(axis=1)
        carriers = np.divide(mass * mass, square, out=np.zeros_like(mass), where=square > 0.)
        return net - carriers / (2. * self.samples)

    def decide(self, per_sample_shadow) -> CountDecision:
        shadow = np.asarray(per_sample_shadow, np.float64)
        key = shadow.tobytes()
        found = self._decisions.get(key)
        if found is not None:
            return found
        differences = self.larger - shadow
        sequence = _test(differences, self.cost, self.samples)
        values = np.minimum(self.total_replicates, self._replicates(differences))
        tail = (1. - self.level) / 2.
        low, high = np.quantile(values, [tail, 1. - tail])
        band = float(low), float(high)
        decision = CountDecision(
            kind=self.kind, accept_larger=bool(self.total.passes and sequence.passes),
            margin=float(min(self.total.margin, sequence.margin)), band=band,
            ambiguous=bool(band[0] < 0. < band[1]), total=self.total, sequence=sequence,
            samples=self.samples, cost=self.cost, seed=self.seed)
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


def smallest_threshold(samples: int) -> float:
    """Smallest charge (units of c) that any one-row decision can require of its margin."""
    # n_eff >= 1 whenever any sample gains; with no gain the margin is <= 0 anyway.
    return 1. / (2. * samples)


def addition_bound_excludes(optimistic_score, score, cost, samples, *, tolerance) -> bool:
    """True iff an optimistic K+1 score proves that no addition passes the criterion.

    An accepted addition needs a gain G with G / c > n_eff / (2S) >= 1 / (2S), so the bound must
    exclude that smallest threshold: (U - F) - c / (2S) < -tolerance / 2. (Base wrote it as
    2 (U - F) - 2 c / (2S) < -tolerance; the same inequality.)
    """
    gain = 2. * (optimistic_score - score) - 2. * smallest_threshold(samples) * cost
    return gain < -tolerance


def addition_per_sample_bound_excludes(per_sample_optimistic, per_sample_score, cost, *,
                                       tolerance):
    """True iff per-sample optimistic scores prove every addition rejected and not ambiguous.

    For every larger panel X+ the per-sample gain d_s = F_s(X+) - F_s(X) is at most
    u_s = U_s - F_s(X) (``per_sample_optimistic`` U bounds F_s of every panel with one row more per
    block; ``per_sample_score`` is F_s(X)). Let M = max_s d_s and P = sum_s d_s+. If P = 0,
    m = sum_s d_s / c <= 0; otherwise sum_s (d_s+)^2 <= M P, so n_eff >= P / M and
    m <= P (1/c - 1/(2 S M)) < 0 whenever M < c / (2S). A bootstrap replicate is a length-S vector
    of entries of d (maximum <= M, divided by the same S), so its margin is <= 0 as well: the
    band's upper end is <= 0, the call is not ambiguous and the addition is rejected, whatever
    the sequence part. Hence max_s u_s + tolerance < c / (2S) proves the decision; the tolerance
    covers the rounding of the computed scores, and a NaN never passes this form. Only the
    maximum over all samples is exact: the leave-one-sample-out margin m' is not what the
    bootstrap resamples, so a bound on it cannot fix the band.
    Returns (excluded, max_s u_s, its sample).
    """
    gains = np.asarray(per_sample_optimistic, np.float64) - np.asarray(per_sample_score,
                                                                        np.float64)
    bound = float(np.max(gains))
    return bool(bound + tolerance < cost / (2. * len(gains))), bound, int(np.argmax(gains))


def deletion_bound_excludes(per_sample_score, per_sample_optimistic, cost, *, tolerance,
                            config: CountRuleConfig, seed_key):
    """Whether per-sample optimistic K-1 scores prove that every deletion keeps the larger panel
    and that the call is not ambiguous. Returns (excluded, band_bound).

    Point decision. The test keeps a founder only if both m and the leave-one-sample-out margin
    m' are positive; a chromosome total cannot bound m', so it needs per-sample upper bounds
    U_s >= F_s(any K-1 panel). With lower bounds l_s = F_s(X_K) - U_s of d_s, m >= sum l / c - 1/2
    (n_eff <= S) and m' >= (sum l - max l) / c - (S - 1) / (2S) (the S - 1 smallest d_s sum to at
    least the S - 1 smallest l_s).

    The same proof covers the sequence part in founder refinement, whose q is an exact shadow
    difference: the shadow of a deletion of f uses at every block only rows that X_K minus f
    uses, at most K - 1 distinct rows, and the count bound bounds every panel with at most K - 1
    distinct rows per block, duplicated founders adding no emission; so U_s >= F_s(shadow),
    l_s <= q_s and both tests pass for every candidate. A founder without a private cell has
    q = 0 = F_s(X_K) - F_s(X_K), and X_K then has at most K - 1 distinct rows per block, so
    l_s <= 0 and no round is skipped. (The join's near-identity shortcut is not a score and no
    score bound sees it; refinement does not use it.)

    Band. The point bounds say nothing about the bootstrap band (lower bounds can be negative and
    a replicate can miss the few carriers), so a skipped round would otherwise report
    ``ambiguous = False`` unproven. Replicate r of the decision's bootstrap (``seed_key``: the key
    :func:`decide` would use, so the same individuals i_r1 .. i_rS) has, for d and for q alike,
    margin >= sum_j l_(i_rj) / c - 1/2 = B_r (its n_eff is at most its S entries), hence
    min(m_d, m_q) >= B_r replicate by replicate, and the band's lower percentile is at least the
    same percentile of B (order statistics and their interpolation are monotone). band_bound is
    that percentile; the round is skipped only when it also exceeds the tolerance, so band[0] > 0:
    the call is proven not ambiguous for every candidate. Otherwise the round runs.

    ``per_sample_score`` (F_s(X_K)) and ``per_sample_optimistic`` (U_s) cover the informative
    samples (length S); ``tolerance`` (nats) covers the rounding of the computed scores; ``config``
    gives the bootstrap's resamples and level. band_bound is None when the point bounds fail.
    """
    lower = np.asarray(per_sample_score, np.float64) - np.asarray(per_sample_optimistic,
                                                                   np.float64)
    samples = len(lower)
    total = float(lower.sum())
    relative = tolerance / cost
    if not (total / cost - .5 > relative
            and (total - float(lower.max())) / cost - (samples - 1) / (2. * samples) > relative):
        return False, None
    index = _resampled_individuals(bootstrap_seed(seed_key), int(config.bootstrap_resamples),
                                   samples)
    replicate_bounds = lower[index].sum(axis=1) / cost - .5
    band_bound = float(np.quantile(replicate_bounds, (1. - float(config.bootstrap_level)) / 2.))
    return bool(band_bound > relative), band_bound
