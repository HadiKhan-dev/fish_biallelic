"""Founder-count decisions between two panels that differ by one founder row.

The criterion compares a larger panel X+ with a smaller panel X- (one row fewer) through
per-sample scores (founder refinement: the canonical per-sample scores F_s of
``founder/scoring.score_panel``, each sample's full-site Potts Viterbi score; the assembly
join: its per-sample panel likelihood) and the per-founder cost c of
``chimera_scoring.compute_cc``, c = cc_scale x (mean sites per block / 200) x S x B.
Let d_s = F_s(X+) - F_s(X-) and L = sum_s d_s / c (in units of c).

The effective-carrier criterion (ECB) charges the per-sample cost c/S only to the
founder's effective carriers (a BIC-style K c - 2F rule would charge every founder c/2
for every sample, although a founder's evidence accrues only on the samples that carry
it), n_eff = (sum_s d_s+)^2 / sum_s (d_s+)^2 with d_s+ = max(d_s, 0)
(the participation ratio of the positive part). A vector of per-sample differences passes
iff m = L - n_eff / (2S) > 0 and the same margin with the sample of largest d_s removed
(S unchanged), m', is > 0. With n_eff = S the margin equals BIC's L - 1/2.

The pipeline's one count rule, ``decide_with_sequence`` (the assembly join and founder
refinement): the larger panel is kept (deletion rejected) or accepted (addition) iff the
test passes on the total per-sample gain d AND on its sequence part q_s = F_s(X+) -
F_s(X+^sigma), where the shadow X+^sigma replaces the decided founder's private sequence
by the nearest rows of the panel it is compared with (``assembly/panel_count``), so switch
savings alone never justify a founder. The rule is the same for deletions and additions.

Ambiguity: individuals are resampled with replacement (a deterministic stream derived
from ``seed_key``) and min(m_d, m_q) is recomputed; a call whose percentile interval
contains zero in its interior is ambiguous. Ambiguous calls keep the point-estimate
decision unless ``ambiguous_count_action`` is "keep_larger".

Count cycles (``pre_cycle_count``): when a search deletes a row that one of its own
additions created, the deletion stands (births then stop). A cycle whose closing deletion
is ambiguous and that began with a decided deletion is undecided: the data could not tell
the two counts apart, and the call is flagged (founder_counts.tsv).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

CRITERION = "effective_carrier"
AMBIGUOUS_ACTIONS = ("point_estimate", "keep_larger")
KINDS = ("deletion", "addition")
_TOP_SAMPLES = 5


@dataclass(frozen=True)
class CountDecision:
    accept_larger: bool        # True: keep the founder (deletion rejected) / accept the addition
    margin: float              # criterion margin in units of c (> 0 favours the larger panel)
    band: tuple[float, float]  # sample-bootstrap percentile interval of the margin
    ambiguous: bool
    diagnostics: dict


def effective_carriers(differences):
    """Participation ratio of the positive per-sample differences (0 if none is positive)."""
    positive = np.maximum(np.asarray(differences, np.float64), 0.)
    square = float(np.sum(positive * positive))
    return float(np.sum(positive)) ** 2 / square if square > 0. else 0.


def _ecb_margin(differences, cost, samples):
    return float(np.sum(differences)) / cost - effective_carriers(differences) / (2. * samples)


def _seed(seed_key):
    text = json.dumps(seed_key, sort_keys=True, default=str)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def _joint_bootstrap(total, sequence, cost, samples, resamples, level, seed):
    """Percentile interval of min(m_total, m_sequence) over the same resampled individuals."""
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(total), size=(resamples, len(total)))
    values = None
    for differences in (total, sequence):
        draws = differences[index]
        net = draws.sum(axis=1) / cost
        positive = np.maximum(draws, 0.)
        square = (positive * positive).sum(axis=1)
        mass = positive.sum(axis=1)
        carriers = np.divide(mass * mass, square, out=np.zeros_like(mass), where=square > 0.)
        margin = net - carriers / (2. * samples)
        values = margin if values is None else np.minimum(values, margin)
    tail = (1. - level) / 2.
    low, high = np.quantile(values, [tail, 1. - tail])
    return float(low), float(high)


def decide_with_sequence(per_sample_larger, per_sample_smaller, per_sample_shadow, cost, *, kind, config,
                         seed_key) -> CountDecision:
    """The pipeline's count decision: the effective-carrier test on the total gain AND on its sequence part.

    ``per_sample_shadow`` scores the larger panel with the decided path's cells that the smaller panel
    lacks replaced by the smaller panel's nearest rows; q = larger - shadow is the sequence part of the
    per-sample gain (zero for a path whose every cell another path uses). The larger panel is kept iff
    both tests pass (m > 0 and m' > 0 for d and for q); the band is that of min(m_d, m_q). ``kind`` is
    "deletion" (larger = incumbent) or "addition" (smaller = incumbent).
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    larger = np.asarray(per_sample_larger, np.float64)
    smaller = np.asarray(per_sample_smaller, np.float64)
    shadow = np.asarray(per_sample_shadow, np.float64)
    if larger.ndim != 1 or larger.shape != smaller.shape or larger.shape != shadow.shape or not len(larger):
        raise ValueError("per-sample scores must be equal-length vectors")
    cost = float(cost)
    if not np.isfinite(cost) or cost <= 0.:
        raise ValueError("count decisions need a positive per-founder cost")
    samples = len(larger)
    parts = {}
    for name, differences in (("total", larger - smaller), ("sequence", larger - shadow)):
        top = int(np.argmax(differences))
        rest = np.delete(differences, top)
        carriers = effective_carriers(differences)
        margin = _ecb_margin(differences, cost, samples)
        loso = _ecb_margin(rest, cost, samples)
        positive = np.maximum(differences, 0.)
        positive_total = float(positive.sum())
        order = np.argsort(-differences, kind="stable")[:_TOP_SAMPLES]
        parts[name] = dict(sum=float(differences.sum()), n_eff=carriers, margin=margin, loso_margin=loso,
                           loso_sample=top, passes=bool(margin > 0. and loso > 0.),
                           positive_samples=int(np.count_nonzero(differences > 0.)),
                           top_samples=[int(i) for i in order],
                           top_differences=[float(differences[i]) for i in order],
                           top_shares=[float(positive[i] / positive_total) if positive_total > 0. else 0.
                                       for i in order])
    point = parts["total"]["passes"] and parts["sequence"]["passes"]
    margin = min(parts["total"]["margin"], parts["sequence"]["margin"])
    seed = _seed(seed_key)
    band = _joint_bootstrap(larger - smaller, larger - shadow, cost, samples,
                            int(config.count_bootstrap_resamples), float(config.count_bootstrap_level), seed)
    ambiguous = bool(band[0] < 0. < band[1])
    accept = True if ambiguous and config.ambiguous_count_action == "keep_larger" else bool(point)
    diagnostics = {"criterion": CRITERION + "+sequence", "kind": kind, "samples": samples, "cost": cost,
                   "threshold_per_carrier": cost / (2. * samples), "margin": margin,
                   "point_decision_accept_larger": bool(point), "accept_larger": accept,
                   "band": [band[0], band[1]], "ambiguous": ambiguous,
                   "ambiguous_count_action": config.ambiguous_count_action,
                   "bootstrap": {"resamples": int(config.count_bootstrap_resamples),
                                 "level": float(config.count_bootstrap_level), "seed": seed},
                   "total": parts["total"], "sequence": parts["sequence"]}
    return CountDecision(accept, float(margin), (band[0], band[1]), ambiguous, diagnostics)


def pre_cycle_count(history, larger, structural=frozenset()):
    """The founder count a count cycle started from (the join and founder refinement).

    ``history`` lists one search's founder count at its start and after every accepted count
    change, ending with the current count ``larger``; ``structural`` holds the indices j of
    the changes history[j - 1] -> history[j] that deleted a row without sequence of its own
    (always deleted: not a count decision). A deletion from ``larger`` of a row that an
    addition of the same search created closes a count cycle: rows were added and deleted
    again, each call made against a different panel, so the rule does not separate
    ``larger`` from ``larger - 1`` there. The cycle began where the count last stood without
    falling below ``larger - 1`` afterwards. If the count there is ``larger`` or more and
    then first fell to ``larger - 1`` by a count decision, the cycle began by deleting: that
    count is returned, and the callers flag the cycle as undecided when the closing deletion
    is ambiguous (its band contains zero). Otherwise (it began by adding, or the first fall
    removed a row without sequence, which never counted) ``larger - 1`` is returned. The
    deletion stands either way.
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


def smallest_threshold(samples):
    """Smallest charge (units of c) that any one-row decision can require of its margin."""
    # n_eff >= 1 whenever any sample gains; with no gain the margin is <= 0 anyway.
    return 1. / (2. * samples)


def addition_bound_excludes(optimistic_score, score, cost, samples, *, margin):
    """True iff an optimistic K+1 score proves that no addition passes the criterion.

    An accepted addition needs G > n_eff / (2S) >= 1 / (2S), so the bound must
    exclude that smallest threshold.
    """
    gain = 2. * (optimistic_score - score) - 2. * smallest_threshold(samples) * cost
    return gain < -margin


def addition_per_sample_bound_excludes(per_sample_optimistic, per_sample_score, cost, *, tolerance):
    """True iff per-sample optimistic scores prove every addition rejected and not ambiguous.

    For every larger panel X+ the per-sample gain d_s = F_s(X+) - F_s(X) is at most
    u_s = U_s - F_s(X) (``per_sample_optimistic`` U bounds F_s of every panel with one row
    more per block, ``count_bound``; ``per_sample_score`` is F_s(X)). Let M = max_s d_s and
    P = sum_s d_s+. If P = 0, m = sum_s d_s / c <= 0; otherwise sum_s (d_s+)^2 <= M P, so
    n_eff >= P / M and m <= P (1/c - 1/(2 S M)) < 0 whenever M < c / (2S). A bootstrap
    replicate is a length-S vector of entries of d (maximum <= M, divided by the same S), so
    its margin is <= 0 as well: the band's upper end is <= 0, the call is not ambiguous and
    the addition is rejected under either ``ambiguous_count_action``, whatever the sequence
    part. Hence max_s u_s + tolerance < c / (2S) proves the decision; the tolerance covers
    the rounding of the computed scores, and a NaN never passes this form. Only the maximum
    over all samples is exact: the leave-one-sample-out margin m' is not what the bootstrap
    resamples, so a bound on it cannot fix the band. Returns (excluded, max_s u_s, its sample).
    """
    gains = np.asarray(per_sample_optimistic, np.float64) - np.asarray(per_sample_score, np.float64)
    bound = float(np.max(gains))
    return bool(bound + tolerance < cost / (2. * len(gains))), bound, int(np.argmax(gains))


def deletion_bound_excludes(score, optimistic_score, cost, *, margin,
                            per_sample_score=None, per_sample_optimistic=None):
    """True iff optimistic K-1 scores prove that every deletion keeps the larger panel.

    ECB keeps a founder only if both m and the leave-one-sample-out margin m' are
    positive; a chromosome total cannot bound m', so ECB needs per-sample upper
    bounds U_s >= F_s(any K-1 panel). With lower bounds l_s = F_s(X_K) - U_s of
    d_s, m >= sum l / c - 1/2 (n_eff <= S) and m' >= (sum l - max l) / c - (S - 1) / (2S)
    (the S - 1 smallest d_s sum to at least the S - 1 smallest l_s).
    Without per-sample bounds the round is not excluded.

    The same proof covers the sequence part of ``decide_with_sequence`` in founder
    refinement, whose q is an exact shadow difference: the shadow of a deletion of f uses at
    every block only rows that X_K minus f uses, at most K - 1 distinct rows, and
    ``count_bound`` bounds every panel with at most K - 1 distinct rows per block (per block,
    any K - 1 of a block's K rows, or all rows; free resets between blocks), duplicated
    founders adding no emission; so U_s >= F_s(shadow), l_s <= q_s and both tests pass for
    every candidate. A founder without a private cell has q = 0 = F_s(X_K) - F_s(X_K), and
    X_K then has at most K - 1 distinct rows per block, so l_s <= 0 and no round is skipped.
    (The join's near-identity shortcut, q = 0 for a path within 1% of another, is not a
    score and no score bound sees it; refinement does not use it.)
    """
    if per_sample_score is None or per_sample_optimistic is None:
        return False
    lower = np.asarray(per_sample_score, np.float64) - np.asarray(per_sample_optimistic, np.float64)
    samples = len(lower)
    total = float(lower.sum())
    relative = margin / cost
    return bool(total / cost - .5 > relative
                and (total - float(lower.max())) / cost - (samples - 1) / (2. * samples) > relative)


def check_config(config):
    """Validate the criterion fields of a FounderRefinementConfig."""
    if config.ambiguous_count_action not in AMBIGUOUS_ACTIONS:
        raise ValueError(f"ambiguous_count_action must be one of {AMBIGUOUS_ACTIONS}")
    resamples = config.count_bootstrap_resamples
    if isinstance(resamples, bool) or int(resamples) != resamples or resamples < 1:
        raise ValueError("count_bootstrap_resamples must be a positive integer")
    if not 0. < float(config.count_bootstrap_level) < 1.:
        raise ValueError("count_bootstrap_level must lie in (0, 1)")
