"""Contig spike-and-slab fit of wrong-read excess from painted homozygotes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logsumexp

from .parameters import SITE_ERROR_BATCH, SITE_ERROR_BISECTIONS, SITE_ERROR_PREVALENCE_BOUNDARY


@dataclass(frozen=True)
class SiteErrorFit:
    direction: np.ndarray
    rate: np.ndarray
    probability: np.ndarray
    prevalence: float
    gain: float


def _project(painting, positions):
    """Project the two named tracks through the original panel at the physical bin."""
    shape = (len(painting.sample_ids), len(positions))
    dosage = np.full(shape, -1, dtype=np.int8)
    pair_error = np.zeros(shape)
    for component in painting.components:
        if not component.bins.n_bins:
            continue
        panel = component.panel
        left = np.searchsorted(panel.positions, positions[0], side="left")
        right = np.searchsorted(panel.positions, positions[-1], side="right")
        _, sites, columns = np.intersect1d(
            positions, panel.positions[left:right], assume_unique=True, return_indices=True,
        )
        if not len(sites):
            continue
        bins = np.searchsorted(component.bins.edges, positions[sites], side="right") - 1
        inside = (bins >= 0) & (bins < component.bins.n_bins)
        bins, sites, columns = bins[inside], sites[inside], columns[inside] + left
        labels = component.labels[:, :, bins]
        alleles = panel.released[np.maximum(labels, 0), columns]
        valid = ((labels >= 0) & (alleles >= 0)).all(axis=1)
        dosage[:, sites] = np.where(valid, alleles.sum(axis=1), -1)
        if component.label_error is not None:
            error = component.label_error[:, bins, 0].astype(np.float64)
            pair_error[:, sites] = np.where(np.isfinite(error), error, 0)
    return dosage, 1 - (1 - painting.robustness_epsilon) * (1 - pair_error)


def _hom_log_likelihood(minor, major, error):
    return minor * np.log(error) + major * np.log1p(-error)


def _log_likelihoods(ref, alt, error, balance, inverse):
    """Native genotype likelihoods in log space, omitting the shared binomial coefficient."""
    ref, alt = np.asarray(ref, dtype=np.int64), np.asarray(alt, dtype=np.int64)
    error, balance = error[:, None], balance[:, None]
    if inverse:
        # The evidence model's rising products, accumulated at integer read depths.
        depth = ref + alt
        shifts = np.arange(depth.max()) * inverse

        def prefix(mean):
            return np.pad(np.cumsum(np.log(mean + shifts), axis=1), ((0, 0), (1, 0)))

        rows = np.arange(len(error))[:, None]
        heterozygote = (
            prefix(balance)[rows, alt] + prefix(1 - balance)[rows, ref]
            - np.r_[0.0, np.cumsum(np.log1p(shifts))][depth]
        )
    else:
        heterozygote = _hom_log_likelihood(alt, ref, balance)
    return np.stack(
        (_hom_log_likelihood(alt, ref, error), heterozygote, _hom_log_likelihood(ref, alt, error)),
        axis=-1,
    )


def _score(ref, alt, error, balance, inverse, painted, redraw, cohort, eligible, rates):
    """Sum each direction/rate's marginal log likelihood ratio in sample order."""
    raw = _log_likelihoods(ref, alt, error, balance, inverse)
    prior = redraw[:, :, None] * cohort[None, :, :] + (1 - redraw[:, :, None]) * (
        painted[:, :, None] == np.arange(3)
    )
    used = ((painted == 0) | (painted == 2)) & eligible[:, None] & ((ref > 0) | (alt > 0))
    prior[~used] = 1 / 3
    with np.errstate(divide="ignore"):
        log_prior = np.log(prior)
    joint = raw + log_prior
    baseline = logsumexp(joint, axis=2)
    scores = np.empty((ref.shape[1], 2, len(rates)))
    for side, (minor, major) in enumerate(((alt, ref), (ref, alt))):
        genotype = 2 * side
        other = np.logaddexp(joint[:, :, 1], joint[:, :, 2 - genotype])
        for step, rate in enumerate(rates):
            excess = (error + (1 - 2 * error) * rate)[:, None]
            adjusted = _hom_log_likelihood(minor, major, excess)
            ratio = np.logaddexp(other, log_prior[:, :, genotype] + adjusted) - baseline
            scores[:, side, step] = np.where(used, ratio, 0.0).sum(axis=0)
    return scores


def _prevalence(log_bayes):
    # The next derivative depends on the bracket from the previous step. Scaling by
    # max(1, BF) keeps each full-contig reduction finite for very strong alternatives.
    slab = np.exp(np.minimum(log_bayes, 0))
    spike = np.exp(np.minimum(-log_bayes, 0))
    difference = slab - spike
    lo, hi = 0.0, 1.0
    for _ in range(SITE_ERROR_BISECTIONS):
        mid = (lo + hi) / 2
        if np.sum(difference / (mid * slab + (1 - mid) * spike)) > 0:
            lo = mid
        else:
            hi = mid
    prevalence = (lo + hi) / 2
    if prevalence < SITE_ERROR_PREVALENCE_BOUNDARY:
        prevalence = 0.0
    if prevalence > 1 - SITE_ERROR_PREVALENCE_BOUNDARY:
        prevalence = 1.0
    with np.errstate(divide="ignore"):
        gain = float(np.logaddexp(np.log1p(-prevalence), np.log(prevalence) + log_bayes).sum())
    return prevalence, gain


def fit_site_error(painting, positions, counts, cohort, eligible, read_model, parameters, map_work):
    """Score independent site blocks, then fit one prevalence over all supplied kept sites."""
    error, balance = read_model.error_and_heterozygote()
    inverse = read_model.heterozygote_inverse_concentration
    rates = np.asarray(parameters.site_error_rates)
    n_sites = len(positions)
    log_bayes = np.empty(n_sites)
    probability = np.empty((n_sites, 2))
    mean_rate = np.empty((n_sites, 2))
    blocks = tuple(slice(left, min(left + SITE_ERROR_BATCH, n_sites)) for left in range(0, n_sites, SITE_ERROR_BATCH))

    def score(block):
        painted, redraw = _project(painting, positions[block])
        ref, alt = counts[:, block].transpose(2, 0, 1)
        scores = _score(ref, alt, error, balance, inverse, painted, redraw, cohort[block], eligible, rates)
        norm = logsumexp(scores, axis=(1, 2))
        log_bayes[block] = norm - np.log(2 * len(rates))
        conditional = np.exp(scores - norm[:, None, None])
        mass = conditional.sum(axis=2)
        probability[block] = mass
        mean_rate[block] = np.divide(
            (conditional * rates).sum(axis=2), mass, out=np.zeros_like(mass), where=mass > 0,
        )

    list(map_work(score, blocks))
    prevalence, gain = _prevalence(log_bayes)
    if gain <= parameters.site_error_bic_penalty * np.log(max(1, n_sites)):
        prevalence = 0.0
    direction = np.full(n_sites, -1, dtype=np.int8)
    rate = np.zeros(n_sites)
    confidence = np.zeros(n_sites)
    if prevalence:
        with np.errstate(divide="ignore"):
            odds = np.log(prevalence) - np.log1p(-prevalence)

        def select(block):
            mass = probability[block] * expit(log_bayes[block] + odds)[:, None]
            winner = np.argmax(mass, axis=1)
            winning_mass = np.take_along_axis(mass, winner[:, None], axis=1)[:, 0]
            winning_rate = np.take_along_axis(mean_rate[block], winner[:, None], axis=1)[:, 0]
            selected = (winning_mass > parameters.site_error_direction_posterior) & (mass[:, 0] != mass[:, 1])
            direction[block] = np.where(selected, 2 * winner, -1)
            rate[block] = np.where(selected, winning_rate, 0.0)
            confidence[block] = winning_mass

        list(map_work(select, blocks))
    return SiteErrorFit(direction, rate, confidence, prevalence, gain)


def adjust_likelihoods(output, counts, columns, direction, rate, error, balance, inverse):
    """In-place update of one sample's selected, covered release sites."""
    covered = (counts > 0).any(axis=1)
    ref, alt = counts[covered].T
    if not len(ref):
        return
    logs = _log_likelihoods(ref[None], alt[None], np.array([error]), np.array([balance]), inverse)[0]
    effective = error + (1 - 2 * error) * rate[covered]
    homozygote = direction[covered]
    minor = np.where(homozygote == 0, alt, ref)
    major = np.where(homozygote == 0, ref, alt)
    logs[np.arange(len(logs)), homozygote] = _hom_log_likelihood(minor, major, effective)
    output[columns[covered]] = np.exp(logs - logsumexp(logs, axis=1, keepdims=True))
