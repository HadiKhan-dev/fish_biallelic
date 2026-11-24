"""Orientation evidence from the joint transmission likelihood of painted homologue schedules.

For each material unit compare the ordered dyads' joint log likelihoods
(``transmission.log_likelihood``). A unit's contrast is decisive through the transmitting meiosis or through
painting noise: its robust log odds mix the contrast with the unit's uninformative probability
1 - (1 - eps) rho, eps the relationship model's failure probability of the unit (``unit_failure``, DESIGN_417 D8)
and rho the run's own reliability. A linked pair's units are either all informative or all fair coins, the latter
with the run's own coin share eta. Both are estimated once from the run's own linked pairs (``direction_model``),
without labels. Two impossible directions are uninformative. A fit sums the unit log odds and their log cosh,
weighted by resampling multiplicity, and mixes the two orientations' likelihoods with the coin component
(``effective_log_odds``); with the optional window cap (DirectionParameters.window_morgans > 0) each window's
summed log odds is capped first. A paired junction-density contrast only orders the acyclic graph search and is
reported.

Scope of D8 (AGENTS 2.2): it protects against short painting failures (the relationship model's regions of about
5 cM at 0.1 per Morgan). It does not cap chromosome-wide failures. The window cap, off by default, bounds the
direction evidence of any window of W working Morgans.

Method: docs/methods/pedigree.md#finite-continuous-direction
"""

from __future__ import annotations

import math

import numpy as np

from ...contracts.pedigree import JunctionEvidence, Orientation
from ...core.kernels import kernel
from .kernels.decision import OVERLOADS
from .numerical_parameters import (
    DIRECTION_COIN_SHARE_GRID,
    DIRECTION_RELIABILITY_GRID,
    ORDER_MINIMUM_UNIT_WEIGHT,
    ORDER_VARIANCE_OFFSET,
)

__all__ = [
    "unit_failure",
    "direction_windows",
    "direction_model",
    "structural_prior",
    "effective_log_odds",
    "junction_evidence",
    "fit",
    "parent_role_probability",
]


@kernel(overloads=OVERLOADS["orientation._order_probability"], cache=True, nogil=True)
def _order_probability(junctions, callable_bins, total_bins, multiplicity):
    """Paired unit contrast softened by missingness and count variance.

    A summary-level MAR approximation, not a common-interval recount. Inclusion-exclusion bounds
    the common exposure; absent overlap is neutral.
    """
    chromosomes, samples = junctions.shape
    result = np.full((samples, samples), 0.5)
    for child in range(samples):
        for parent in range(child):
            count = mean_sum = square_sum = missing_variance = 0.0
            for chrom in range(chromosomes):
                if multiplicity[chrom] <= 0 or total_bins[chrom] <= 0:
                    continue
                fc = min(1.0, callable_bins[chrom, child] / (2.0 * total_bins[chrom]))
                fp = min(1.0, callable_bins[chrom, parent] / (2.0 * total_bins[chrom]))
                common = max(0.0, fc + fp - 1.0)
                if common <= 0:
                    continue
                weight = multiplicity[chrom] * common
                dc = junctions[chrom, child] / fc
                dp = junctions[chrom, parent] / fp
                delta = dc - dp
                count += weight
                mean_sum += weight * delta
                square_sum += weight * delta * delta
                missing_variance += weight * ((1.0 - fc) * dc / fc + (1.0 - fp) * dp / fp)
            if count <= ORDER_MINIMUM_UNIT_WEIGHT:
                continue
            variance = max(0.0, square_sum - mean_sum * mean_sum / count)
            variance *= count / (count - 1.0)
            variance += missing_variance + ORDER_VARIANCE_OFFSET
            probability = 0.5 * (1.0 + math.erf(mean_sum / math.sqrt(2.0 * variance)))
            result[child, parent] = probability
            result[parent, child] = 1.0 - probability
    return result


@kernel(overloads=OVERLOADS["orientation._pair_log_odds"], cache=True, nogil=True)
def _pair_log_odds(
    log_likelihood, forward_index, reverse_index, pair_child, pair_parent, callable_bins, total_bins, error_probability
):
    """values[unit, u] for the unordered dyads u (pair_parent -> pair_child against the reverse) where
    ``compared``; log_likelihood[unit, d] over the ordered dyads (pedigree.transmission.log_likelihood);
    error_probability[unit] the unit's painting-failure probability (a row is capped at ln((2 - eps) / eps))."""
    chromosomes = log_likelihood.shape[0]
    pairs = len(pair_child)
    values = np.zeros((chromosomes, pairs))
    compared = np.zeros((chromosomes, pairs), dtype=np.bool_)
    for chrom in range(chromosomes):
        if total_bins[chrom] <= 0:
            continue
        keep = 1.0 - error_probability[chrom]
        floor = 0.5 * error_probability[chrom]
        for u in range(pairs):
            child = pair_child[u]
            parent = pair_parent[u]
            fc = min(1.0, callable_bins[chrom, child] / (2.0 * total_bins[chrom]))
            fp = min(1.0, callable_bins[chrom, parent] / (2.0 * total_bins[chrom]))
            if fc <= 0.0 or fp <= 0.0:
                continue
            forward = log_likelihood[chrom, forward_index[u]]
            reverse = log_likelihood[chrom, reverse_index[u]]
            if forward == -np.inf and reverse == -np.inf:
                continue  # impossible both ways: uninformative
            difference = forward - reverse
            if difference == 0.0:
                continue
            scaled = math.exp(-abs(difference))
            larger = keep / (1.0 + scaled) + floor
            smaller = keep * scaled / (1.0 + scaled) + floor
            value = math.log(larger) - math.log(smaller)
            if difference < 0.0:
                value = -value
            values[chrom, u] = value
            compared[chrom, u] = True
    return values, compared


@kernel(overloads=OVERLOADS["orientation._weighted_log_odds"], cache=True, nogil=True)
def _weighted_log_odds(values, compared, pair_child, pair_parent, total_bins, multiplicity, samples):
    """Unit-weighted sum of the pair log odds, ``result[child, parent]``, in unit order."""
    chromosomes = values.shape[0]
    result = np.zeros((samples, samples))
    for chrom in range(chromosomes):
        if multiplicity[chrom] <= 0 or total_bins[chrom] <= 0:
            continue
        for u in range(len(pair_child)):
            if compared[chrom, u]:
                value = values[chrom, u]
                result[pair_child[u], pair_parent[u]] += multiplicity[chrom] * value
                result[pair_parent[u], pair_child[u]] -= multiplicity[chrom] * value
    return result


def _unordered_dyads(dyads: np.ndarray, samples: int):
    """(child, parent, forward index, reverse index) of the unordered dyads whose two directions are both in
    the ordered ``dyads`` [D, 2], child > parent, in row-major (child, parent) order."""
    dyads = np.asarray(dyads, dtype=np.int64).reshape(-1, 2)
    keys = dyads[:, 0] * samples + dyads[:, 1]
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    forward = np.flatnonzero(dyads[:, 0] > dyads[:, 1])
    reverse_keys = dyads[forward, 1] * samples + dyads[forward, 0]
    where = np.searchsorted(sorted_keys, reverse_keys)
    found = (where < len(sorted_keys)) & (sorted_keys[np.minimum(where, len(sorted_keys) - 1)] == reverse_keys)
    forward = forward[found]
    reverse = order[where[found]]
    rank = np.lexsort((dyads[forward, 1], dyads[forward, 0]))
    forward, reverse = forward[rank], reverse[rank]
    return (
        np.ascontiguousarray(dyads[forward, 0]),
        np.ascontiguousarray(dyads[forward, 1]),
        np.ascontiguousarray(forward),
        np.ascontiguousarray(reverse),
    )


def unit_failure(morgans: np.ndarray, rate: float, length: float) -> np.ndarray:
    """float64 [H]: probability that a painting failure region touches a unit of ``morgans`` working Morgans.

    The relationship model's two-state process on the working map (RelationshipParameters): feasible -> failure at
    lambda = ``rate`` per Morgan, back at mu = 1 / ``length`` per Morgan, from its stationary law, so
    P(no failure touches [0, L]) = mu / (lambda + mu) * exp(-lambda L) and eps(L) = 1 - that (DESIGN_417 D8; a 1-M
    unit: eps 0.0997, row cap ln((2 - eps) / eps) = 2.95 nats)."""
    mu = 1.0 / float(length)
    lam = float(rate)
    return np.ascontiguousarray(1.0 - mu / (lam + mu) * np.exp(-lam * np.asarray(morgans, dtype=np.float64)))


def direction_windows(morgans: np.ndarray, window: float) -> np.ndarray | None:
    """int64 [H]: consecutive windows of ``window`` working Morgans along the unit axis (a unit belongs to the
    window holding its start), or None when ``window`` is 0 (cap off)."""
    if window <= 0:
        return None
    starts = np.concatenate(([0.0], np.cumsum(np.asarray(morgans, dtype=np.float64))[:-1]))
    return np.ascontiguousarray(np.floor(starts / window).astype(np.int64))


def _log_cosh_half(values: np.ndarray) -> np.ndarray:
    """log cosh(values / 2), stable for large |values|."""
    half = 0.5 * np.abs(values)
    return half + np.log1p(np.exp(-2.0 * half)) - math.log(2.0)


def structural_prior(log_odds: np.ndarray, pair_failure: float) -> np.ndarray:
    """P(forward orientation | the pair's structural orientation log odds), bounded by the pair-level failure:
    (1 - e) sigmoid(S) + e/2; 1/2 where S is undefined (NaN)."""
    log_odds = np.asarray(log_odds, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        value = (1.0 - pair_failure) / (1.0 + np.exp(-log_odds)) + 0.5 * pair_failure
    return np.where(np.isnan(log_odds), 0.5, value)


def direction_model(
    contrasts: np.ndarray, compared: np.ndarray, failure: np.ndarray, forward: np.ndarray
) -> tuple[float, float, int, int]:
    """The run's own direction model from its reference pairs: (reliability rho, coin share eta, pairs, units).

    ``contrasts`` [H, P]: the raw unit contrasts d of the reference pairs (forward minus reverse joint transmission
    log likelihood); ``compared`` [H, P]: units with evidence; ``failure`` [H]: each unit's painting-failure
    probability (``unit_failure``); ``forward`` [P]: prior probability of each pair's forward orientation
    (``structural_prior`` of the direction-free solve).

    Model: an informative pair's unit favours its true orientation with probability
    f = pi sigmoid(|d|) + (1 - pi) / 2, pi = (1 - failure) rho; a coin pair's units are fair coins. Marginal
    likelihood of the unit signs given their magnitudes,
    sum_i log[(1 - eta)(w_i Pf_i + (1 - w_i) Pr_i) + eta 2^-n_i], with uniform priors on rho and eta; the estimates
    are the posterior means on fixed midpoint grids (deterministic; no optimiser). Without reference pairs the
    posterior is the prior: (1/2, 1/2). Label-free: no sample name, cohort, relatedness family or released edge
    enters.
    """
    compared = np.asarray(compared, dtype=np.bool_)
    keep = compared.any(axis=0) if compared.size else np.zeros(compared.shape[1], dtype=np.bool_)
    pairs, units = int(np.count_nonzero(keep)), int(np.count_nonzero(compared[:, keep]))
    if not pairs:
        return 0.5, 0.5, 0, 0
    compared = compared[:, keep]
    contrast = np.where(compared, np.asarray(contrasts, dtype=np.float64)[:, keep], 0.0)
    prior = np.clip(np.asarray(forward, dtype=np.float64)[keep], np.finfo(float).tiny, 1.0)
    with np.errstate(over="ignore"):
        sigmoid = 1.0 / (1.0 + np.exp(-contrast))
    rho_grid = (np.arange(DIRECTION_RELIABILITY_GRID) + 0.5) / DIRECTION_RELIABILITY_GRID
    eta_grid = (np.arange(DIRECTION_COIN_SHARE_GRID) + 0.5) / DIRECTION_COIN_SHARE_GRID
    coin = -np.count_nonzero(compared, axis=0) * math.log(2.0)
    keep_eta, with_eta = np.log1p(-eta_grid), np.log(eta_grid)
    surface = np.empty((len(rho_grid), len(eta_grid)))
    informative = (1.0 - np.asarray(failure, dtype=np.float64))[:, None]
    with np.errstate(divide="ignore"):
        log_forward, log_reverse = np.log(prior), np.log1p(-prior)
    for index, rho in enumerate(rho_grid):
        pi = informative * rho
        right = np.where(compared, np.log(pi * sigmoid + 0.5 * (1.0 - pi)), 0.0).sum(axis=0)
        wrong = np.where(compared, np.log(pi * (1.0 - sigmoid) + 0.5 * (1.0 - pi)), 0.0).sum(axis=0)
        pair = np.logaddexp(log_forward + right, log_reverse + wrong)
        surface[index] = np.logaddexp(keep_eta[None, :] + pair[:, None], with_eta[None, :] + coin[:, None]).sum(axis=0)
    posterior = np.exp(surface - surface.max())
    posterior /= posterior.sum()
    return (float(posterior.sum(axis=1) @ rho_grid), float(posterior.sum(axis=0) @ eta_grid), pairs, units)


def effective_log_odds(
    log_odds: np.ndarray, log_cosh: np.ndarray, coin_share: float, pair_failure: float
) -> np.ndarray:
    """A pair's orientation log odds with the coin component: ``log_odds`` L = sum w v and ``log_cosh``
    G = sum w log cosh(v / 2) of a fit (elementwise over pairs).

    The two orientations' log likelihoods against a coin are A = L/2 - G and B = -L/2 - G. With the pair law
    P(D | forward) = (1 - e - eta) Pf + (e/2)(Pf + Pr) + eta C (e the pair-level failure, eta the coin share),
    order.direction_terms' factor ln 2 + ln((1 - e) sigmoid(L_eff) + e/2) equals log[P(D | forward) / P_ref(D)],
    P_ref the two orientations' mean, exactly when
    L_eff = log[(1 - e - eta) e^A + k] - log[(1 - e - eta) e^B + k], k = eta (e (e^A + e^B)/2 + 1 - e).
    eta = 0 returns L. Coin data drive A and B down linearly in the units, so L_eff -> 0: a symmetric relative's
    random walk does not accumulate."""
    if coin_share <= 0.0:
        return log_odds
    eta = min(float(coin_share), 1.0 - pair_failure - np.finfo(float).eps)
    failure = max(float(pair_failure), np.finfo(float).tiny)
    forward = 0.5 * log_odds - log_cosh
    reverse = -0.5 * log_odds - log_cosh
    mixed = np.logaddexp(forward, reverse) - math.log(2.0)
    constant = math.log(eta) + np.logaddexp(math.log(failure) + mixed, math.log1p(-failure))
    head = math.log(1.0 - failure - eta)
    return np.logaddexp(head + forward, constant) - np.logaddexp(head + reverse, constant)


def junction_evidence(
    junctions: np.ndarray,
    callable_bins: np.ndarray,
    total_bins: np.ndarray,
    error_probability: np.ndarray,
    dyads: np.ndarray,
    log_likelihood: np.ndarray,
    *,
    window: np.ndarray | None = None,
    window_cap: float = float("inf"),
    reliability: float = 1.0,
    coin_share: float = 0.0,
    pair_failure: float = 0.0,
    reference_pairs: int = 0,
    reference_units: int = 0,
) -> JunctionEvidence:
    """Weight-independent junction evidence of the material units (``dyads`` [D, 2] and ``log_likelihood``
    [H, D]: the joint transmission log likelihoods of the scores; ``error_probability`` [H] each unit's
    painting-failure probability, ``unit_failure``; ``reliability`` and ``coin_share`` the run's direction
    model, ``direction_model``). A unit's uninformative probability is 1 - (1 - error_probability) reliability."""
    junctions = np.ascontiguousarray(junctions, dtype=np.float64)
    callable_bins = np.ascontiguousarray(callable_bins, dtype=np.float64)
    total_bins = np.ascontiguousarray(total_bins, dtype=np.float64)
    child, parent, forward, reverse = _unordered_dyads(dyads, junctions.shape[1])
    failure = np.broadcast_to(np.asarray(error_probability, dtype=np.float64), total_bins.shape)
    values, compared = _pair_log_odds(
        np.ascontiguousarray(log_likelihood, dtype=np.float64).reshape(len(total_bins), -1),
        forward,
        reverse,
        child,
        parent,
        callable_bins,
        total_bins,
        np.ascontiguousarray(1.0 - (1.0 - failure) * float(reliability)).copy(),
    )
    return JunctionEvidence(junctions, callable_bins, total_bins, child, parent, values, compared,
                            window, float(window_cap), float(reliability), float(coin_share), float(pair_failure),
                            int(reference_pairs), int(reference_units),
                            pair_log_cosh=_pair_log_cosh(values, compared) if coin_share > 0.0 else None)


def _pair_log_cosh(values: np.ndarray, compared: np.ndarray) -> np.ndarray:
    """float64 [H, U]: log cosh(v / 2) of the compared unit log odds, zero elsewhere."""
    return np.ascontiguousarray(np.where(compared, _log_cosh_half(values), 0.0))


def reference_contrasts(dyads: np.ndarray, log_likelihood: np.ndarray, samples: int, units: int,
                        pairs: np.ndarray) -> np.ndarray:
    """float64 [H, P]: raw unit contrasts (forward minus reverse joint transmission log likelihood) of the unordered
    dyads ``pairs`` (indices into the evidence's pair axis); a unit impossible both ways gives 0."""
    _, _, forward, reverse = _unordered_dyads(dyads, samples)
    values = np.asarray(log_likelihood, dtype=np.float64).reshape(units, -1)
    with np.errstate(invalid="ignore"):
        contrast = values[:, forward[pairs]] - values[:, reverse[pairs]]
    return np.where(np.isnan(contrast), 0.0, contrast)


def _with_coin_share(evidence: JunctionEvidence, log_odds: np.ndarray, log_cosh: np.ndarray) -> np.ndarray:
    """log_odds[child, parent] of a fit after the coin mixture; ``log_cosh`` [U] the fit's summed log cosh."""
    totals = effective_log_odds(log_odds[evidence.pair_child, evidence.pair_parent], log_cosh,
                                evidence.coin_share, evidence.pair_failure)
    result = np.array(log_odds, dtype=np.float64, copy=True)
    result[evidence.pair_child, evidence.pair_parent] = totals
    result[evidence.pair_parent, evidence.pair_child] = -totals
    return result


def _windowed_log_odds(evidence: JunctionEvidence, weights: np.ndarray) -> np.ndarray:
    """log_odds[child, parent] with each direction window's weighted sum capped at +-window_cap."""
    assert evidence.window is not None
    samples = evidence.junctions.shape[1]
    weights = np.asarray(weights, dtype=np.float64)
    active = (weights > 0) & (evidence.total_bins > 0)
    values = np.where(evidence.compared, evidence.pair_log_odds, 0.0) * np.where(active, weights, 0.0)[:, None]
    totals = np.zeros(evidence.pair_log_odds.shape[1])
    for window in np.unique(evidence.window):
        rows = evidence.window == window
        totals += np.clip(values[rows].sum(axis=0), -evidence.window_cap, evidence.window_cap)
    result = np.zeros((samples, samples))
    result[evidence.pair_child, evidence.pair_parent] = totals
    result[evidence.pair_parent, evidence.pair_child] = -totals
    return result


def fit(evidence: JunctionEvidence, weights: np.ndarray) -> Orientation:
    """Orientation of a fit whose units carry multiplicities ``weights`` [H]."""
    # Burden and callability are report-only; direction decisions use the kernels below.
    counts = np.sum(weights[:, None] * evidence.junctions, axis=0)
    exposure = np.sum(weights[:, None] * evidence.callable_bins, axis=0)
    full = float(np.sum(weights * (2.0 * evidence.total_bins)))
    fractions = np.divide(exposure, full, out=np.zeros_like(exposure), where=full > 0)
    adjusted = np.divide(counts, fractions, out=np.full_like(counts, np.nan), where=fractions > 0)
    samples = evidence.junctions.shape[1]
    multiplicity = np.ascontiguousarray(weights, dtype=np.float64)
    log_odds = _weighted_log_odds(
        evidence.pair_log_odds,
        evidence.compared,
        evidence.pair_child,
        evidence.pair_parent,
        evidence.total_bins,
        multiplicity,
        samples,
    ) if evidence.window is None else _windowed_log_odds(evidence, weights)
    if evidence.coin_share > 0.0:
        # The same unit-ordered weighted sum over log cosh(v / 2); the window cap never applies to it.
        cosh = evidence.pair_log_cosh
        if cosh is None:
            cosh = _pair_log_cosh(evidence.pair_log_odds, evidence.compared)
        log_cosh = _weighted_log_odds(
            cosh, evidence.compared, evidence.pair_child, evidence.pair_parent, evidence.total_bins, multiplicity,
            samples,
        )[evidence.pair_child, evidence.pair_parent]
        log_odds = _with_coin_share(evidence, log_odds, log_cosh)
    return Orientation(
        order_probability=_order_probability(evidence.junctions, evidence.callable_bins, evidence.total_bins, weights),
        log_odds=log_odds,
        adjusted_burden=adjusted,
        callability=fractions,
    )


def parent_role_probability(row: int, rows: np.ndarray, orientation: Orientation) -> float:
    """Graph-ordering support of a configuration: product of its edges' order probabilities
    (1 for M0); orientation-ordering support, not a parenthood probability."""
    parents = [int(parent) for parent in rows[row, 1:] if parent >= 0]
    if not parents:
        return 1.0
    return float(np.prod(orientation.order_probability[int(rows[row, 0]), parents]))


@kernel(overloads=OVERLOADS["orientation._order_statistics"], cache=True, nogil=True)
def _order_statistics(junctions, callable_bins, total_bins):
    """Additive four-moment statistics of the block-level graph-order model."""
    h, n = junctions.shape
    result = np.zeros((n, n, 4))
    for child in range(n):
        for parent in range(child):
            for block in range(h):
                if total_bins[block] <= 0:
                    continue
                fc = min(1.0, callable_bins[block, child] / (2 * total_bins[block]))
                fp = min(1.0, callable_bins[block, parent] / (2 * total_bins[block]))
                common = max(0.0, fc + fp - 1.0)
                if common <= 0:
                    continue
                dc, dp = junctions[block, child] / fc, junctions[block, parent] / fp
                delta = dc - dp
                result[child, parent, 0] += common
                result[child, parent, 1] += common * delta
                result[child, parent, 2] += common * delta * delta
                result[child, parent, 3] += common * ((1-fc)*dc/fc + (1-fp)*dp/fp)
    return result


@kernel(overloads=OVERLOADS["orientation._omit_order_probability"], cache=True, nogil=True)
def _omit_order_probability(totals, junctions, callable_bins, total_bins):
    """O(N²) order probabilities after removing one block's moment contributions."""
    n = len(junctions)
    result = np.full((n, n), .5)
    for child in range(n):
        for parent in range(child):
            count, mean, square, missing = totals[child, parent]
            if total_bins > 0:
                fc = min(1.0, callable_bins[child] / (2*total_bins))
                fp = min(1.0, callable_bins[parent] / (2*total_bins))
                common = max(0.0, fc+fp-1.0)
                if common > 0:
                    dc, dp = junctions[child]/fc, junctions[parent]/fp
                    delta = dc-dp
                    count -= common
                    mean -= common*delta
                    square -= common*delta*delta
                    missing -= common*((1-fc)*dc/fc+(1-fp)*dp/fp)
            if count <= ORDER_MINIMUM_UNIT_WEIGHT:
                continue
            variance = max(0.0, square-mean*mean/count) * count/(count-1.0)
            variance += missing + ORDER_VARIANCE_OFFSET
            probability = .5 * (1+math.erf(mean/math.sqrt(2*variance)))
            result[child,parent] = probability
            result[parent,child] = 1-probability
    return result


def fit_omitting(evidence, totals, omitted):
    """The same unit-level model, using totals minus the omitted unit."""
    counts = totals.junctions - evidence.junctions[omitted]
    exposure = totals.callable_bins - evidence.callable_bins[omitted]
    full = 2.0 * (totals.total_bins - evidence.total_bins[omitted])
    fractions = np.divide(exposure, full, out=np.zeros_like(exposure), where=full > 0)
    adjusted = np.divide(counts, fractions, out=np.full_like(counts, np.nan), where=fractions > 0)
    if evidence.window is None:
        values = totals.pair_log_odds - evidence.pair_log_odds[omitted]
        log_odds = np.zeros((len(counts), len(counts)))
        log_odds[evidence.pair_child, evidence.pair_parent] = values
        log_odds[evidence.pair_parent, evidence.pair_child] = -values
    else:
        weights = np.ones(len(evidence.total_bins))
        weights[omitted] = 0.0
        log_odds = _windowed_log_odds(evidence, weights)
    if evidence.coin_share > 0.0:
        cosh = evidence.pair_log_cosh
        if cosh is None:
            cosh = _pair_log_cosh(evidence.pair_log_odds, evidence.compared)
        total = totals.pair_log_cosh if totals.pair_log_cosh is not None else cosh.sum(axis=0)
        log_odds = _with_coin_share(evidence, log_odds, total - cosh[omitted])
    return Orientation(_omit_order_probability(totals.order_statistics, evidence.junctions[omitted],
        evidence.callable_bins[omitted], float(evidence.total_bins[omitted])), log_odds, adjusted, fractions)
