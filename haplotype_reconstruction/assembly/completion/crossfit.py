"""Cross-fitted variational EM over diplotypes and missing founder alleles, conservative release, and
held-out carrier profiles of one block.

Model (one block, K founder rows with hard calls or -1 = unresolved): each sample copies an unordered
pair of founder rows (uniform prior over the K(K+1)/2 pairs); each unresolved founder allele is a
shared binary variable (uniform prior 2^-U over the U unresolved alleles of a site); a sample's
genotype evidence at a site enters through the robust log emission max(log((1 - mix) P(g) + mix/3),
floor) of the dosage of its pair. The variational family factorises pair states per sample and
configurations per site; up to MAX_EXACT_BITS unresolved founders a site's configurations are all
assignments (a cube), above that the uncertain-core support of the shared site posterior under the
first iteration's pair posteriors (fixed for the whole EM, so the ELBO stays a lower bound).

Cross-fitting: the samples are split into two deterministic folds; a model fitted on one fold
releases calls (posterior >= minimum_call_probability, carrier support, non-exchangeable founders,
no multimodal site, shell guard), and the other fold's samples are painted on complete sites of the
released panel only. Each sample's carrier profile therefore never uses its own evidence in the
release that defines it.

The EM runs from one start (uniform configurations, complete-site pair posteriors): base ran five
starts, but with minimum_iterations >= 2 starts 1-4 shared start 0's trajectory after the initial
ELBO, so start 0 was always selected (survey MAP 4.2).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..founder_alleles import site_posterior
from . import dosage_statistics
from .observation import (UNIFORM_TOLERANCE, carrier_probability, normalised_evidence, row_classes,
                          unordered_pairs)


@dataclass(frozen=True)
class CarrierProfiles:
    """Held-out (cross-fitted) carrier profiles of one block's founder rows.

    expected_copies: float64[N, K] expected number of copies of each founder row a sample carries
        (0 .. 2; a homozygous pair counts twice).
    carrier_probability: float64[N, K] probability that the sample carries at least one copy.
    informative_samples: bool[N] the sample has at least one usable complete site.
    informative_site_count: int64[N] observed sites where the released panel calls every founder.
    """

    expected_copies: np.ndarray
    carrier_probability: np.ndarray
    informative_samples: np.ndarray
    informative_site_count: np.ndarray


@dataclass(frozen=True)
class CrossFit:
    """One block's cross-fit: carrier profiles plus counts for the completion summary.

    site_posterior_counts: the two fold fits' site-posterior counts (``site_posterior.SUMMARY_FIELDS``
        order; sites above MAX_EXACT_BITS unresolved founders).
    guard_skipped: releases the Hamming-ball shell guard withheld in the two folds.
    """

    profiles: CarrierProfiles
    site_posterior_counts: tuple[int, ...]
    guard_skipped: int


def deterministic_two_folds(n_samples: int, seed: int) -> np.ndarray:
    """int8[N] fold label (0/1) of each sample: a seeded permutation alternated between the folds."""
    if n_samples < 2:
        raise ValueError("two-fold cross-fitting requires at least two samples")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_samples)
    folds = np.empty(n_samples, dtype=np.int8)
    folds[order] = np.arange(n_samples, dtype=np.int64) % 2
    return folds


def _log_emission(evidence, observed, config):
    """float64[N, L, 3] robust log emissions max(log((1 - mix) P + mix/3), floor), 0 where unobserved."""
    robust = (1.0 - config.uniform_mix) * normalised_evidence(evidence) + config.uniform_mix / 3.0
    with np.errstate(divide="ignore"):
        log_emission = np.log(robust)
    np.maximum(log_emission, config.log_floor, out=log_emission)
    log_emission[~observed] = 0.0
    return log_emission


def _softmax(log_values: np.ndarray, axis: int = -1) -> np.ndarray:
    maximum = np.max(log_values, axis=axis, keepdims=True)
    values = np.exp(log_values - maximum)
    values /= np.sum(values, axis=axis, keepdims=True)
    return values


def _entropy_term(probability: np.ndarray, log_prior) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        log_probability = np.log(probability)
    terms = np.where(probability > 0, probability * (log_prior - log_probability), 0.0)
    return float(np.sum(terms))


def _complete_site_state_scores(alleles, log_emission, observed, pairs):
    """Per-sample log likelihood of every pair state on the sites where every founder is called,
    and the number of such observed sites per sample."""
    called = alleles >= 0
    complete = np.all(called, axis=0)
    usable = complete[None, :] & observed
    scores = np.zeros((log_emission.shape[0], pairs.shape[0]), dtype=np.float64)
    for state, (first, second) in enumerate(pairs):
        # Dosages at incomplete sites (negative) index arbitrary genotypes; usable zeroes them.
        dosage = alleles[first] + alleles[second]
        chosen = np.take_along_axis(log_emission, dosage[None, :, None], axis=2)[..., 0]
        scores[:, state] = np.sum(chosen * usable, axis=1)
    return scores, np.sum(usable, axis=1, dtype=np.int64)


def _configurations_for_site(alleles, site, unknown, pairs, state_probability, log_emission,
                             observed, settings):
    """Allowed assignments of a site's unresolved alleles, with their uniform prior.

    Up to MAX_EXACT_BITS unresolved founders: every assignment (a cube). Above that, the
    uncertain-core support of the shared site posterior under the first EM iteration's pair
    posteriors (``site_posterior.completion_site``): certain founders at the mode and an exact
    core or a Hamming ball. Returns (configurations int8[C, K], log prior float64[C], ball radius
    (0 for a cube), shell flags or None, site-posterior diagnostics or None).
    """
    fixed = alleles[:, site]
    log_prior = -unknown.size * math.log(2.0)
    if unknown.size <= site_posterior.MAX_EXACT_BITS:
        n_configurations = 1 << int(unknown.size)
        configurations = np.empty((n_configurations, alleles.shape[0]), dtype=np.int8)
        for index in range(n_configurations):
            configurations[index] = fixed
            for offset, founder in enumerate(unknown):
                configurations[index, founder] = (index >> offset) & 1
        return configurations, np.full(n_configurations, log_prior), 0, None, None
    evidence = np.ascontiguousarray(np.exp(log_emission[:, site, :]))
    informative = observed[:, site] & (np.ptp(evidence, axis=1) > UNIFORM_TOLERANCE)
    rows, shell, _, info, _ = site_posterior.completion_site(
        evidence, informative, state_probability, pairs, np.ascontiguousarray(fixed),
        unknown, np.zeros(unknown.size), -1.0, settings)
    configurations = np.repeat(fixed[None, :], len(rows), axis=0)
    configurations[:, unknown] = rows
    return (configurations, np.full(len(rows), log_prior), int(info[site_posterior.INFO_RADIUS]),
            shell, info)


def _site_probabilities(probabilities, offsets):
    return tuple(probabilities[start:stop] for start, stop in zip(offsets[:-1], offsets[1:]))


def _elbo(expected_scores, log_priors, site_probabilities, state_probability):
    # Reuse the same expected scores that updated z; q has not changed.
    value = float(np.sum(state_probability * expected_scores))
    value += _entropy_term(state_probability, -math.log(state_probability.shape[1]))
    for log_prior, probability in zip(log_priors, site_probabilities):
        # A fully known site has q=[1] and log_prior=[0], hence zero entropy.
        if len(probability) > 1:
            value += _entropy_term(probability, log_prior)
    return value


def _variational_em(pairs, configurations, log_priors, workspace, complete_scores, config):
    """Coordinate ascent from uniform site configurations and complete-site pair posteriors.

    Returns the pair-state posteriors float64[N, S] and each site's configuration probabilities.
    Raises when the ELBO decreases by more than ``monotonic_tolerance`` (a numerical failure).
    """
    evidence, alleles, offsets, packed_prior, cube = workspace
    state_probability = _softmax(complete_scores - math.log(pairs.shape[0]), axis=1)
    initial = [np.full(values.shape[0], 1.0 / values.shape[0]) for values in configurations]
    probability = np.concatenate(initial) if initial else np.empty(0)
    state_scores = evidence @ dosage_statistics.dosage_marginals(
        alleles, pairs, offsets, probability, cube)
    site_probability = _site_probabilities(probability, offsets)
    previous = _elbo(state_scores, log_priors, site_probability, state_probability)
    for iteration in range(1, config.max_iterations + 1):
        # All site-q updates condition on the same z. Their three-dosage
        # statistics can therefore be contracted together without changing
        # the coordinate-update schedule. BLAS remains worker-local/one thread.
        statistics = state_probability.T @ evidence
        probability = dosage_statistics.configuration_probabilities(
            statistics, alleles, pairs, offsets, packed_prior, cube)
        state_scores = evidence @ dosage_statistics.dosage_marginals(
            alleles, pairs, offsets, probability, cube)
        state_probability = _softmax(state_scores - math.log(pairs.shape[0]), axis=1)
        site_probability = _site_probabilities(probability, offsets)
        value = _elbo(state_scores, log_priors, site_probability, state_probability)
        increment = value - previous
        if increment < -config.monotonic_tolerance:
            raise RuntimeError(f"completion ELBO decreased by {increment:.6g} at iteration {iteration}")
        threshold = config.elbo_absolute_tolerance + config.elbo_relative_tolerance * abs(previous)
        previous = value
        if iteration >= config.minimum_iterations and abs(increment) <= threshold:
            break
    return state_probability, site_probability


def _released_alleles(alleles, evidence, observed, config, settings):
    """Fit the joint model on these samples and release conservative hard calls.

    Returns (released alleles int8[K, L], guard-withheld releases, site-posterior diagnostics
    int64[INFO_FIELDS, L]).
    """
    founders, sites = alleles.shape
    called = alleles >= 0
    log_emission = _log_emission(evidence, observed, config)
    pairs = unordered_pairs(founders)
    complete_scores, _ = _complete_site_state_scores(alleles, log_emission, observed, pairs)
    first_state = None
    configurations, log_priors, unknown_by_site, shell_flags = [], [], [], []
    support_radius = np.zeros(sites, dtype=np.int64)
    site_info = np.zeros((site_posterior.INFO_FIELDS, sites), dtype=np.int64)
    site_info[site_posterior.INFO_CORE] = -1
    for site in range(sites):
        unknown = np.flatnonzero(alleles[:, site] < 0)
        if unknown.size > site_posterior.MAX_EXACT_BITS and first_state is None:
            # Pair posteriors of the first EM iteration.
            first_state = _softmax(complete_scores - math.log(pairs.shape[0]), axis=1)
        site_configurations, log_prior, radius, flags, info = _configurations_for_site(
            alleles, site, unknown, pairs, first_state, log_emission, observed, settings)
        shell_flags.append(flags)
        configurations.append(site_configurations)
        log_priors.append(log_prior)
        unknown_by_site.append(unknown)
        support_radius[site] = radius
        if info is not None:
            site_info[:, site] = info
    workspace = dosage_statistics.prepare(
        configurations, log_priors, log_emission, founders, support_radius)
    state, site_probability = _variational_em(
        pairs, configurations, log_priors, workspace, complete_scores, config)

    marginal = np.full(alleles.shape, 0.5, dtype=np.float64)
    marginal[called] = alleles[called]
    shell = np.zeros(alleles.shape, dtype=np.float64)
    for site, (site_configurations, probability, unknown) in enumerate(
            zip(configurations, site_probability, unknown_by_site)):
        for founder in unknown:
            marginal[founder, site] = float(np.sum(probability * site_configurations[:, founder]))
        if support_radius[site]:
            shell[unknown, site] = site_posterior.shell_weights(probability, shell_flags[site])

    # Carrier support: the larger of the number of samples more likely than not
    # carrying the founder in this block and their summed carrier probability
    # (a homozygous pair counts once).
    carriers_by_sample = carrier_probability(pairs, state, founders)
    carriers = np.maximum(np.count_nonzero(carriers_by_sample > 0.5, axis=0).astype(np.float64),
                          carriers_by_sample.sum(axis=0))
    exchangeable = np.zeros(founders, dtype=np.bool_)
    for rows in row_classes(alleles):
        if len(rows) > 1:
            exchangeable[np.asarray(rows, dtype=np.int64)] = True

    # Release: confident, supported, non-exchangeable founders; above MAX_EXACT_BITS unresolved
    # founders only, multimodal sites release nothing and the ball's shell guard withholds
    # near-threshold calls.
    released = alleles.copy()
    initially_unknown = alleles < 0
    confidence = np.maximum(marginal, 1.0 - marginal)
    eligible = (initially_unknown & ~exchangeable[:, None]
                & (carriers[:, None] >= config.minimum_carriers)
                & (confidence >= config.minimum_call_probability))
    eligible &= ~(site_info[site_posterior.INFO_MULTIMODAL] > 0)[None, :]
    guard_skipped = eligible & site_posterior.guard_mask(
        marginal, shell, config.minimum_call_probability, settings)
    eligible &= ~guard_skipped
    released[eligible] = (marginal[eligible] > 0.5).astype(np.int8)
    return released, int(np.sum(guard_skipped)), site_info


def founder_profiles(pairs, probabilities, informative_site_count, n_founders) -> CarrierProfiles:
    """Expected copies and carrier probabilities of every founder row from pair-state posteriors."""
    copies = np.zeros((probabilities.shape[0], n_founders), dtype=np.float64)
    for state, (first, second) in enumerate(pairs):
        mass = probabilities[:, state]
        if first == second:
            copies[:, first] += 2.0 * mass
        else:
            copies[:, first] += mass
            copies[:, second] += mass
    carriers = carrier_probability(pairs, probabilities, n_founders)
    # Exact expectations/probabilities mathematically; summing normalised state masses can leave a
    # one-ULP excursion such as 1.0000000000000002.
    np.clip(copies, 0.0, 2.0, out=copies)
    np.clip(carriers, 0.0, 1.0, out=carriers)
    count = np.asarray(informative_site_count, dtype=np.int64)
    return CarrierProfiles(expected_copies=copies, carrier_probability=carriers,
                           informative_samples=count > 0, informative_site_count=count)


def crossfit_block(alleles, evidence, observed, folds, config, settings) -> CrossFit:
    """Learn calls on one fold and paint only the complementary held-out fold.

    alleles: int8[K, L] the block's pre-fill calls (-1 unresolved, and at every non-discovery site);
    evidence: float64[N, L, 3] genotype likelihoods (1/3 where unobserved); observed: bool[N, L];
    folds: int8[N] (``deterministic_two_folds``); config: CompletionConfig.crossfit; settings:
    ``site_posterior.settings()``.
    """
    founders = alleles.shape[0]
    pairs = unordered_pairs(founders)
    heldout_probability = np.empty((evidence.shape[0], len(pairs)), dtype=np.float64)
    informative_count = np.zeros(evidence.shape[0], dtype=np.int64)
    counts = np.zeros(len(site_posterior.SUMMARY_FIELDS), dtype=np.int64)
    guard_skipped = 0
    for fold in (0, 1):
        heldout = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        released, guarded, site_info = _released_alleles(
            alleles, evidence[training], observed[training], config, settings)
        guard_skipped += guarded
        counts += site_posterior.summary_counts(site_info)
        heldout_observed = observed[heldout]
        log_emission = _log_emission(evidence[heldout], heldout_observed, config)
        scores, count = _complete_site_state_scores(released, log_emission, heldout_observed, pairs)
        heldout_probability[heldout] = _softmax(scores - math.log(pairs.shape[0]), axis=1)
        informative_count[heldout] = count
    return CrossFit(profiles=founder_profiles(pairs, heldout_probability, informative_count, founders),
                    site_posterior_counts=tuple(int(value) for value in counts),
                    guard_skipped=guard_skipped)
