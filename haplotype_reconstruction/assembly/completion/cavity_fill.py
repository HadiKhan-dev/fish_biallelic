"""Whole-bin leave-out ("cavity") fills of unknown founder alleles in one block.

Pair-state weights are inferred from observed, nonuniform sites outside the
focal bin for which *all* founders are called. At each focal site, every
unresolved founder allele is jointly enumerated once. Conditional on one
assignment, each sample sums its pair-state mixture before sample
likelihoods are multiplied. Independent Bernoulli priors from the panel's q
(0.5 for an unresolved allele) are included for all unresolved alleles,
including the target.

Quadratic dosage coefficients reduce the likelihood table to
``O(N*K**2 + N*2**U)``; exact carrier-sensitivity diagnostics retain
``O(U*N*2**U)`` worst-case work. Near-zero polynomial values use the
direct positive mixture to avoid cancellation, giving a worst-case
``O(N*2**U*K**2)`` cost at U <= MAX_EXACT_BITS. Working memory is
``O(N*K**2 + (N+U)*2**U)``. Above MAX_EXACT_BITS unresolved founders the
uncertain-core support replaces the enumeration (O(N U**3) per site); in
its Hamming-ball regime the shell guard withholds releases near the
threshold.

Carrier support is taken over the focal-informative samples whose
likelihood is sensitive to the target allele: the larger of how many have a
leave-bin-out probability of carrying the target above 1/2 and the sum of
those probabilities. Release needs ``minimum_supporters``; carrier support
gates release but never scales an LLR. A target founder is ineligible when
another founder has the same q vector on eligible outside-bin inference
sites (within ``founder_identifiability_tolerance``), or differs from some
founder at fewer than ``minimum_observable_founder_separation`` observable
outside-bin sites. All candidate fills are computed against the immutable
input panel; the occupancy gate decides which are released.

Evidence normalisation follows base exactly: the block's likelihoods are normalised once for the
focal-informative mask and the site posteriors, and the bin statistics and site posteriors
normalise their (already normalised) input again. Compact support changes summation order only;
scientific identity, rather than bit identity, is the acceptance criterion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from ..founder_alleles import site_posterior
from ..founder_alleles.polynomials import shared_sample_likelihoods
from . import support
from .observation import UNIFORM_TOLERANCE, carrier_probability, normalised_evidence, unordered_pairs


@dataclass(frozen=True)
class CavityFill:
    """One block's candidate fills (before the occupancy gate).

    filled: bool[K, L] unresolved founder alleles the cavity posterior would release.
    posterior_alt: float64[K, L] the cavity posterior ALT probability at unresolved cells (the
        panel's q, i.e. the call or 0.5, elsewhere); a released fill publishes posterior > 0.5.
    site_posterior_counts: site-posterior counts (``site_posterior.SUMMARY_FIELDS`` order) of its
        sites above MAX_EXACT_BITS unresolved founders.
    guard_skipped: releases the Hamming-ball shell guard withheld.
    """

    filled: np.ndarray
    posterior_alt: np.ndarray
    site_posterior_counts: tuple[int, ...]
    guard_skipped: int


@dataclass(frozen=True)
class _SitePosterior:
    """Shared-latent posterior of one site's unresolved founder alleles (float64[K] arrays)."""

    posterior_alt: np.ndarray
    log_posterior_odds: np.ndarray
    carrier_supporters: np.ndarray
    finite_joint_mass: bool
    shell_weight: np.ndarray
    info: np.ndarray


def _informative_cells(evidence, observed, tolerance):
    """(normalised evidence, observed & non-uniform cells)."""
    evidence = normalised_evidence(evidence)
    nonuniform = np.ptp(evidence, axis=2) > tolerance
    return evidence, observed & nonuniform


def diploid_genotype_distributions(q: np.ndarray) -> np.ndarray:
    """P(genotype dosage | founder pair), float64[K, K, L, 3], from founder ALT probabilities q[K, L].

    Distinct founders use their mean-field Bernoulli marginals. The diagonal
    is different: two homologs labelled with the same founder refer to one
    shared latent allele, not two independent draws. Consequently (k, k) has
    only homozygous-reference and homozygous-alternate mass.
    """
    q_first = q[:, None, :]
    q_second = q[None, :, :]
    distribution = np.empty((q.shape[0], q.shape[0], q.shape[1], 3), dtype=np.float64)
    distribution[..., 0] = (1.0 - q_first) * (1.0 - q_second)
    distribution[..., 1] = q_first * (1.0 - q_second) + (1.0 - q_first) * q_second
    distribution[..., 2] = q_first * q_second
    for founder in range(q.shape[0]):
        distribution[founder, founder, :, 0] = 1.0 - q[founder]
        distribution[founder, founder, :, 1] = 0.0
        distribution[founder, founder, :, 2] = q[founder]
    return distribution


def site_log_emissions(evidence, distribution, *, uniform_mix, log_floor=-2.0):
    """float64[N, K, K, L]: robust log emissions relative to uninformative evidence.

    The robust likelihood (1 - mix) sum_g P(g) E(g) + mix/3 (evidence normalised here) is floored
    on the log scale, then log(1/3) is subtracted: an additive state-independent centring that makes
    every uniform sample/site row exactly zero.
    """
    likelihood = np.einsum("nlg,ijlg->nijl", normalised_evidence(evidence), distribution,
                           optimize=True)
    probability = (1.0 - uniform_mix) * likelihood + uniform_mix / 3.0
    with np.errstate(divide="ignore"):
        log_emission = np.log(probability)
    np.maximum(log_emission, log_floor, out=log_emission)
    uniform_log = max(float(np.log(1.0 / 3.0)), float(log_floor))
    return log_emission - uniform_log


def _whole_bin_statistics(q, called, evidence, observed, rule, uniform_tolerance, informative_focal):
    """Per bin: the leave-bin-out pair-state log likelihood of every sample, and the observable
    separation of every founder pair inside the bin.

    Returns (pairs int64[S, 2], outside scores float64[bins, N, S] (sum over the other bins),
    separations int64[bins, K, K]).
    """
    evidence, informative = _informative_cells(evidence, observed, uniform_tolerance)
    informative &= np.all(called, axis=0)[None, :]
    samples, sites = informative.shape
    k = q.shape[0]
    bins = (sites + rule.snps_per_bin - 1) // rule.snps_per_bin
    pairs = unordered_pairs(k)
    scores = np.zeros((bins, samples, len(pairs)))
    separations = np.zeros((bins, k, k), dtype=np.int64)
    for block in range(bins):
        start = block * rule.snps_per_bin
        stop = min(sites, start + rule.snps_per_bin)
        bin_q = q[:, start:stop]
        bin_called = called[:, start:stop]
        # (N, K, K, bin sites) float64 at once: N K^2 x 50 x 8 bytes per bin (bounded by the bin width;
        # chunking over samples would change the einsum's kernel choice and so the last bits).
        emissions = site_log_emissions(evidence[:, start:stop], diploid_genotype_distributions(bin_q),
                                       uniform_mix=rule.uniform_mix)
        mask = informative[:, start:stop]
        emissions *= mask[:, None, None, :]
        for state, (first, second) in enumerate(pairs):
            scores[block, :, state] = np.sum(emissions[:, first, second, :], axis=1)
        observable = np.all(bin_called, axis=0) & np.any(informative_focal[:, start:stop], axis=0)
        for first in range(k):
            for second in range(first + 1, k):
                count = np.count_nonzero(observable & (
                    np.abs(bin_q[first] - bin_q[second]) > rule.founder_identifiability_tolerance))
                separations[block, first, second] = count
                separations[block, second, first] = count
    # Positive/negative bin totals are combined without subtracting a large
    # held-out contribution from a nearly equal whole-block score.
    prefix = np.zeros_like(scores)
    suffix = np.zeros_like(scores)
    for block in range(1, bins):
        prefix[block] = prefix[block - 1] + scores[block - 1]
    for block in range(bins - 2, -1, -1):
        suffix[block] = scores[block + 1] + suffix[block + 1]
    return pairs, prefix + suffix, separations


def _site_posterior(q_site, called_site, focal, observed, pairs, weights, *, uniform_mix,
                    uniform_tolerance, tie_tolerance, settings) -> _SitePosterior:
    """Marginalise the shared unresolved alleles at one founder site.

    For each binary assignment ``a`` to the unresolved founder alleles, each
    sample first marginalises its leave-bin-out pair-state weights,
    ``L_n(a) = sum_s w[n,s] E_n(dosage(s, a))``; the assignment likelihood is
    ``prior(a) * product_n L_n(a)``, so every founder-site allele is drawn once
    and shared by every sample. Up to MAX_EXACT_BITS unresolved founders every
    assignment is enumerated; above that the uncertain-core support replaces
    the enumeration (``_core_site_posterior``).
    """
    n_founders = len(q_site)
    evidence = normalised_evidence(focal[:, None, :])[:, 0, :]
    informative = observed & (np.ptp(evidence, axis=1) > uniform_tolerance)
    unresolved = np.flatnonzero(~called_site)
    n_unresolved = int(unresolved.size)
    if n_unresolved > site_posterior.MAX_EXACT_BITS:
        return _core_site_posterior(q_site, evidence, informative, pairs, weights, unresolved,
                                    uniform_mix=uniform_mix, tie_tolerance=tie_tolerance,
                                    settings=settings)
    assignment_codes = np.arange(1 << n_unresolved, dtype=np.uint64)
    bit_positions = np.arange(n_unresolved, dtype=np.uint64)
    assignments = ((assignment_codes[:, None] >> bit_positions[None, :]) & 1).astype(np.int8)

    robust_evidence = (1.0 - uniform_mix) * evidence + uniform_mix / 3.0
    sample_likelihood = shared_sample_likelihoods(
        robust_evidence, weights, pairs, (q_site > 0.5).astype(np.int8), unresolved)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_sample = np.log(sample_likelihood)
    log_likelihood = np.sum(log_sample[:, informative], axis=1)

    if n_unresolved:
        unresolved_q = q_site[unresolved]
        with np.errstate(divide="ignore"):
            log_alt_prior = np.log(unresolved_q)
            log_ref_prior = np.log1p(-unresolved_q)
        log_prior = np.sum(np.where(assignments == 1, log_alt_prior, log_ref_prior), axis=1)
    else:
        log_prior = np.zeros(1, dtype=np.float64)
    log_joint = log_prior + log_likelihood
    log_marginal = float(logsumexp(log_joint))
    finite_joint_mass = bool(np.isfinite(log_marginal))
    if finite_joint_mass:
        assignment_probability = np.exp(log_joint - log_marginal)
    else:
        # A zero robust mixture can make every assignment impossible.
        # Report the normalized prior and let the caller abstain.
        assignment_probability = np.exp(log_prior - float(logsumexp(log_prior)))

    posterior_alt = q_site.copy()
    log_odds = np.full(n_founders, np.nan, dtype=np.float64)
    supporters = np.zeros(n_founders, dtype=np.float64)
    carriers = carrier_probability(pairs, weights, n_founders)
    for bit, founder in enumerate(unresolved):
        is_alt = assignments[:, bit] == 1
        posterior_alt[founder] = float(np.clip(np.sum(assignment_probability[is_alt]), 0.0, 1.0))
        if finite_joint_mass:
            log_odds[founder] = float(logsumexp(log_joint[is_alt])) - float(logsumexp(log_joint[~is_alt]))
        ref_assignments = np.flatnonzero(~is_alt)
        alt_assignments = ref_assignments + (1 << bit)
        target_sensitive = np.any(~np.isclose(log_sample[ref_assignments], log_sample[alt_assignments],
                                              rtol=0.0, atol=tie_tolerance, equal_nan=True), axis=0)
        selected = carriers[informative & target_sensitive, founder]
        supporters[founder] = max(float(np.count_nonzero(selected > 0.5)), float(np.sum(selected)))
    info = np.zeros(site_posterior.INFO_FIELDS, dtype=np.int64)
    info[site_posterior.INFO_CORE] = -1
    return _SitePosterior(posterior_alt, log_odds, supporters, finite_joint_mass,
                          np.zeros(n_founders, dtype=np.float64), info)


def _core_site_posterior(q_site, evidence, informative, pairs, weights, unresolved, *, uniform_mix,
                         tie_tolerance, settings) -> _SitePosterior:
    """``_site_posterior`` above MAX_EXACT_BITS unresolved founders.

    The same likelihood and Bernoulli(q) priors over the support of the
    uncertain-core rule. Carrier support is taken over the informative samples
    whose likelihood changes with the target allele somewhere on that support.
    """
    n_founders = len(q_site)
    robust_evidence = (1.0 - uniform_mix) * evidence + uniform_mix / 3.0
    unresolved_q = q_site[unresolved]
    with np.errstate(divide="ignore"):
        log_alt_prior = np.log(unresolved_q)
        log_ref_prior = np.log1p(-unresolved_q)
    with np.errstate(invalid="ignore"):
        prior_log_odds = log_alt_prior - log_ref_prior
    b, core, flips, sizes, log_likelihood, info, sensitive = site_posterior.completion_site(
        np.ascontiguousarray(robust_evidence), np.ascontiguousarray(informative), weights, pairs,
        (q_site > 0.5).astype(np.int8), unresolved.astype(np.int64), prior_log_odds,
        float(tie_tolerance), settings)
    configurations = support.from_core(
        b, np.arange(len(unresolved)), b, core, flips, sizes,
        int(info[site_posterior.INFO_RADIUS]))
    log_prior = support.log_priors(configurations, log_ref_prior, log_alt_prior)
    log_joint = log_prior + log_likelihood
    log_marginal = float(logsumexp(log_joint))
    finite_joint_mass = bool(np.isfinite(log_marginal))
    if finite_joint_mass:
        assignment_probability = np.exp(log_joint - log_marginal)
    else:
        assignment_probability = np.exp(log_prior - float(logsumexp(log_prior)))
    posterior_alt = q_site.copy()
    log_odds = np.full(n_founders, np.nan, dtype=np.float64)
    supporters = np.zeros(n_founders, dtype=np.float64)
    carriers = carrier_probability(pairs, weights, n_founders)
    sums, outer = support.allele_sums(configurations, assignment_probability)
    log_sums, _ = support.allele_sums(configurations, log_joint, logarithmic=True)
    for bit, founder in enumerate(unresolved):
        posterior_alt[founder] = float(np.clip(sums[bit, 1], 0.0, 1.0))
        if finite_joint_mass:
            log_odds[founder] = log_sums[bit, 1] - log_sums[bit, 0]
        selected = carriers[informative & sensitive[bit], founder]
        supporters[founder] = max(float(np.count_nonzero(selected > 0.5)), float(np.sum(selected)))
    shell = np.zeros(n_founders, dtype=np.float64)
    shell[unresolved] = outer
    return _SitePosterior(posterior_alt, log_odds, supporters, finite_joint_mass, shell, info)


def cavity_fill(q, called, evidence, observed, rule, settings) -> CavityFill:
    """Candidate fills of one block's unknown founder alleles from leave-whole-bin-out evidence.

    q: float64[K, L] founder ALT probability (the call where called, 0.5 elsewhere); called:
    bool[K, L] (pre-fill calls at discovery sites); evidence: float64[N, L, 3] genotype likelihoods
    (1/3 where unobserved); observed: bool[N, L] (observed discovery sites); rule:
    CompletionConfig.cavity; settings: ``site_posterior.settings()``.
    """
    uniform_tolerance = UNIFORM_TOLERANCE
    evidence, informative_focal = _informative_cells(evidence, observed, uniform_tolerance)
    k, n_sites = q.shape
    filled = np.zeros(called.shape, dtype=np.bool_)
    posterior_alt = q.copy()
    guard_skipped = 0
    site_info = np.zeros((site_posterior.INFO_FIELDS, n_sites), dtype=np.int64)
    site_info[site_posterior.INFO_CORE] = -1

    n_bins = (n_sites + rule.snps_per_bin - 1) // rule.snps_per_bin
    pairs, outside_scores, bin_separations = _whole_bin_statistics(
        q, called, evidence, observed, rule, uniform_tolerance, informative_focal)
    all_separations = np.sum(bin_separations, axis=0)
    for bin_index in range(n_bins):
        start = bin_index * rule.snps_per_bin
        stop = min(start + rule.snps_per_bin, n_sites)
        exchangeable_founder = np.zeros(k, dtype=np.bool_)
        minimum_separation = np.full(k, -1, dtype=np.int64)
        if k > 1:
            separation = all_separations - bin_separations[bin_index]
            for founder in range(k):
                other = np.arange(k) != founder
                minimum_separation[founder] = int(np.min(separation[founder, other]))
                exchangeable_founder[founder] = minimum_separation[founder] == 0

        log_likelihood = outside_scores[bin_index]
        maximum = np.max(log_likelihood, axis=1, keepdims=True)
        weights = np.exp(log_likelihood - maximum)
        weights /= np.sum(weights, axis=1, keepdims=True)

        for site in range(start, stop):
            unresolved = np.flatnonzero(~called[:, site])
            if unresolved.size == 0:
                continue
            joint = _site_posterior(
                q[:, site], called[:, site], evidence[:, site, :], observed[:, site], pairs, weights,
                uniform_mix=rule.uniform_mix, uniform_tolerance=uniform_tolerance,
                tie_tolerance=rule.tie_tolerance, settings=settings)
            posterior_alt[unresolved, site] = joint.posterior_alt[unresolved]
            site_info[:, site] = joint.info
            if not joint.finite_joint_mass:
                continue
            for founder in unresolved:
                if exchangeable_founder[founder]:
                    continue
                if k > 1 and minimum_separation[founder] < rule.minimum_observable_founder_separation:
                    continue
                probability_alt = joint.posterior_alt[founder]
                confidence = max(probability_alt, 1.0 - probability_alt)
                if (abs(joint.log_posterior_odds[founder]) > rule.tie_tolerance
                        and joint.carrier_supporters[founder] >= rule.minimum_supporters
                        and confidence >= rule.minimum_posterior_probability):
                    if joint.info[site_posterior.INFO_MULTIMODAL]:
                        # Two comparable, unconnected posterior basins: no release.
                        continue
                    if joint.shell_weight[founder] > 0.0 and site_posterior.guard_withholds(
                            probability_alt, joint.shell_weight[founder],
                            rule.minimum_posterior_probability, settings):
                        guard_skipped += 1
                        continue
                    # A released founder allele is a hard biological call; its calibrated
                    # marginal stays in posterior_alt.
                    filled[founder, site] = True
    return CavityFill(filled=filled, posterior_alt=posterior_alt,
                      site_posterior_counts=tuple(site_posterior.summary_counts(site_info)),
                      guard_skipped=guard_skipped)
