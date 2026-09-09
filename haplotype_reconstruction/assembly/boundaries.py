"""assembly / boundaries for the canonical reconstruction pipeline."""
from __future__ import annotations


from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
import haplotype_reconstruction.assembly.observations as assembly_observations

@dataclass(frozen=True)
class UnorderedDiplotypePosterior:
    """Posterior mass over lexicographically ordered pairs ``i <= j``."""

    pairs: np.ndarray
    probabilities: np.ndarray
    informative_site_count: np.ndarray


@dataclass(frozen=True)
class SharedLatentSitePosterior:
    """Exact small-K posterior over shared unresolved founder-site alleles."""

    unresolved_founders: np.ndarray
    assignments: np.ndarray
    log_joint: np.ndarray
    assignment_probabilities: np.ndarray
    posterior_alt: np.ndarray
    log_posterior_odds: np.ndarray
    effective_carrier_support: np.ndarray
    informative_samples: np.ndarray
    log_marginal_likelihood: float
    finite_joint_mass: bool


@dataclass(frozen=True)
class CavityFillRule:
    """Conservative release rule for whole-bin cavity allele filling.

    ``minimum_observable_founder_separation`` requires each founder to differ
    observably from every other founder outside the focal bin.
    ``maximum_enumerated_unresolved_founders`` bounds the exponential exact
    focal-site sum; sites above it are retained as unknown.
    """

    snps_per_bin: int = 10
    minimum_effective_supporters: float = 2.0
    minimum_posterior_probability: float = 0.85
    uniform_mix: float = 0.01
    tie_tolerance: float = 1e-12
    founder_identifiability_tolerance: float = 1e-12
    minimum_observable_founder_separation: int = 1
    maximum_enumerated_unresolved_founders: int = 6

    def __post_init__(self) -> None:
        if self.snps_per_bin < 1:
            raise ValueError("snps_per_bin must be positive")
        if self.minimum_effective_supporters < 0.0:
            raise ValueError("minimum_effective_supporters must be non-negative")
        if not 0.5 < self.minimum_posterior_probability <= 1.0:
            raise ValueError(
                "minimum_posterior_probability must lie in (0.5, 1]"
            )
        if not 0.0 <= self.uniform_mix <= 1.0:
            raise ValueError("uniform_mix must lie in [0, 1]")
        if self.tie_tolerance < 0.0:
            raise ValueError("tie_tolerance must be non-negative")
        if (
            isinstance(self.minimum_observable_founder_separation, bool)
            or int(self.minimum_observable_founder_separation)
            != self.minimum_observable_founder_separation
            or self.minimum_observable_founder_separation < 1
        ):
            raise ValueError(
                "minimum_observable_founder_separation must be a positive integer"
            )
        if self.founder_identifiability_tolerance < 0.0:
            raise ValueError(
                "founder_identifiability_tolerance must be non-negative"
            )
        if (
            isinstance(self.maximum_enumerated_unresolved_founders, bool)
            or int(self.maximum_enumerated_unresolved_founders)
            != self.maximum_enumerated_unresolved_founders
            or self.maximum_enumerated_unresolved_founders < 0
        ):
            raise ValueError(
                "maximum_enumerated_unresolved_founders must be a "
                "non-negative integer"
            )


@dataclass(frozen=True)
class CavityFillResult:
    """Filled copy plus distinct exchangeability/separation diagnostics."""

    panel: assembly_observations.FounderPanel
    filled: np.ndarray
    posterior_alt: np.ndarray
    effective_supporters: np.ndarray
    exchangeable_skipped: np.ndarray
    minimum_observable_separation: np.ndarray
    insufficient_separation_skipped: np.ndarray
    enumeration_limit_skipped: np.ndarray
    n_exchangeable_skipped: int
    n_insufficient_separation_skipped: int
    n_enumeration_limit_skipped: int


def _normalised_evidence(genotype_likelihoods: np.ndarray) -> np.ndarray:
    evidence = np.asarray(genotype_likelihoods, dtype=np.float64)
    if evidence.ndim != 3 or evidence.shape[2] != 3:
        raise ValueError("genotype_likelihoods must have shape (samples, sites, 3)")
    if np.any(~np.isfinite(evidence)) or np.any(evidence < 0.0):
        raise ValueError("genotype_likelihoods must be finite and non-negative")
    total = np.sum(evidence, axis=2, keepdims=True)
    result = np.full(evidence.shape, 1.0 / 3.0, dtype=np.float64)
    np.divide(evidence, total, out=result, where=total > 0.0)
    return result


def _informative_cells(
    genotype_likelihoods: np.ndarray,
    observed: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    evidence = _normalised_evidence(genotype_likelihoods)
    observed = np.asarray(observed, dtype=np.bool_)
    if observed.shape != evidence.shape[:2]:
        raise ValueError("observed must have shape (samples, sites)")
    nonuniform = np.ptp(evidence, axis=2) > tolerance
    return evidence, observed & nonuniform


def _unordered_pairs(k: int) -> np.ndarray:
    return np.asarray(
        [(first, second) for first in range(k) for second in range(first, k)],
        dtype=np.int64,
    )


def exact_shared_latent_site_posterior(
    panel: assembly_observations.FounderPanel,
    site: int,
    genotype_likelihoods: np.ndarray,
    observed: np.ndarray,
    pairs: np.ndarray,
    pair_state_probabilities: np.ndarray,
    *,
    uniform_mix: float = 0.01,
    uniform_tolerance: float = 1e-12,
    tie_tolerance: float = 1e-12,
) -> SharedLatentSitePosterior:
    """Exactly marginalize shared unresolved alleles at one founder site.

    For each binary assignment ``a`` to the unresolved founder alleles, each
    sample first marginalizes its leave-bin-out pair-state weights,

    ``L_n(a) = sum_s w[n,s] E_n(dosage(s, a))``.

    The assignment likelihood is then ``prior(a) * product_n L_n(a)``.  Thus
    every founder-site allele is drawn once and shared by every sample, rather
    than independently re-marginalized in each per-sample emission.  This is an
    exponential small-K oracle and callers are responsible for applying a
    conservative enumeration cap.
    """

    if isinstance(site, bool) or int(site) != site:
        raise ValueError("site must be an integer index")
    site = int(site)
    n_founders, n_sites = panel.q.shape
    if not 0 <= site < n_sites:
        raise ValueError("site is outside the panel")
    if not 0.0 <= uniform_mix <= 1.0:
        raise ValueError("uniform_mix must lie in [0, 1]")
    if uniform_tolerance < 0.0 or tie_tolerance < 0.0:
        raise ValueError("tolerances must be non-negative")

    focal = np.asarray(genotype_likelihoods, dtype=np.float64)
    if focal.ndim != 2 or focal.shape[1] != 3:
        raise ValueError("focal genotype likelihoods must have shape (samples, 3)")
    evidence = _normalised_evidence(focal[:, None, :])[:, 0, :]
    observed = np.asarray(observed, dtype=np.bool_)
    if observed.shape != (evidence.shape[0],):
        raise ValueError("focal observed must have shape (samples,)")
    informative = observed & (np.ptp(evidence, axis=1) > uniform_tolerance)

    pairs = np.asarray(pairs, dtype=np.int64)
    weights = np.asarray(pair_state_probabilities, dtype=np.float64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape (states, 2)")
    if np.any((pairs < 0) | (pairs >= n_founders)):
        raise ValueError("pairs contain an invalid founder index")
    if weights.shape != (evidence.shape[0], pairs.shape[0]):
        raise ValueError("pair_state_probabilities must have shape (samples, states)")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("pair-state probabilities must be finite and non-negative")
    if not np.allclose(np.sum(weights, axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("each sample pair-state distribution must sum to one")

    unresolved = np.flatnonzero(~panel.called[:, site])
    n_unresolved = int(unresolved.size)
    assignment_codes = np.arange(1 << n_unresolved, dtype=np.uint64)
    bit_positions = np.arange(n_unresolved, dtype=np.uint64)
    assignments = (
        (assignment_codes[:, None] >> bit_positions[None, :]) & 1
    ).astype(np.int8)

    allele_assignments = np.broadcast_to(
        (panel.q[:, site] > 0.5).astype(np.int8),
        (assignment_codes.size, n_founders),
    ).copy()
    if n_unresolved:
        allele_assignments[:, unresolved] = assignments
    dosage = (
        allele_assignments[:, pairs[:, 0]]
        + allele_assignments[:, pairs[:, 1]]
    )

    robust_evidence = (1.0 - uniform_mix) * evidence + uniform_mix / 3.0
    sample_likelihood = np.empty(
        (assignment_codes.size, evidence.shape[0]), dtype=np.float64
    )
    for assignment_index in range(assignment_codes.size):
        state_likelihood = robust_evidence[:, dosage[assignment_index]]
        sample_likelihood[assignment_index] = np.sum(
            weights * state_likelihood, axis=1
        )
    with np.errstate(divide="ignore"):
        log_sample_likelihood = np.log(sample_likelihood)
    log_likelihood = np.sum(
        log_sample_likelihood[:, informative], axis=1
    )

    if n_unresolved:
        unresolved_q = panel.q[unresolved, site]
        with np.errstate(divide="ignore"):
            log_alt_prior = np.log(unresolved_q)
            log_ref_prior = np.log1p(-unresolved_q)
        log_prior = np.sum(
            np.where(assignments == 1, log_alt_prior, log_ref_prior),
            axis=1,
        )
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
        prior_normalizer = float(logsumexp(log_prior))
        assignment_probability = np.exp(log_prior - prior_normalizer)

    posterior_alt = panel.q[:, site].copy()
    log_odds = np.full(n_founders, np.nan, dtype=np.float64)
    effective_support = np.zeros(n_founders, dtype=np.float64)
    carrier_state = (
        (pairs[:, :, None] == np.arange(n_founders)[None, None, :])
        .any(axis=1)
        .astype(np.float64)
    )
    carrier_probability = weights @ carrier_state
    with np.errstate(divide="ignore", invalid="ignore"):
        log_sample = np.log(sample_likelihood)

    for bit, founder in enumerate(unresolved):
        is_alt = assignments[:, bit] == 1
        posterior_alt[founder] = float(np.clip(
            np.sum(assignment_probability[is_alt]), 0.0, 1.0
        ))
        if finite_joint_mass:
            log_alt = float(logsumexp(log_joint[is_alt]))
            log_ref = float(logsumexp(log_joint[~is_alt]))
            log_odds[founder] = log_alt - log_ref

        ref_assignments = np.flatnonzero(~is_alt)
        alt_assignments = ref_assignments + (1 << bit)
        target_sensitive = np.any(
            ~np.isclose(
                log_sample[ref_assignments],
                log_sample[alt_assignments],
                rtol=0.0,
                atol=tie_tolerance,
                equal_nan=True,
            ),
            axis=0,
        )
        effective_support[founder] = float(np.sum(
            carrier_probability[informative & target_sensitive, founder]
        ))

    return SharedLatentSitePosterior(
        unresolved_founders=unresolved,
        assignments=assignments,
        log_joint=log_joint,
        assignment_probabilities=assignment_probability,
        posterior_alt=posterior_alt,
        log_posterior_odds=log_odds,
        effective_carrier_support=effective_support,
        informative_samples=informative,
        log_marginal_likelihood=log_marginal,
        finite_joint_mass=finite_joint_mass,
    )


def _posterior_with_site_mask(
    panel: assembly_observations.FounderPanel,
    genotype_likelihoods: np.ndarray,
    observed: np.ndarray,
    included_sites: np.ndarray,
    *,
    uniform_mix: float,
    uniform_tolerance: float,
) -> UnorderedDiplotypePosterior:
    evidence, informative = _informative_cells(
        genotype_likelihoods, observed, uniform_tolerance
    )
    included = np.asarray(included_sites, dtype=np.bool_)
    if included.shape != (panel.q.shape[1],):
        raise ValueError("included_sites must match the panel site dimension")
    # A site may enter factorized per-sample state inference only when every
    # founder allele is authoritative.  Otherwise its shared latent allele
    # belongs in the joint focal-site model below, never in independent sample
    # emissions.
    inference_sites = included & np.all(panel.called, axis=0)
    informative &= inference_sites[None, :]

    distributions = assembly_observations.diploid_genotype_distributions(panel)
    site_scores = assembly_observations.site_log_emissions(
        evidence, distributions, uniform_mix=uniform_mix
    )
    site_scores *= informative[:, None, None, :]
    pairs = _unordered_pairs(panel.q.shape[0])
    log_likelihood = np.empty((evidence.shape[0], pairs.shape[0]), np.float64)
    for state, (first, second) in enumerate(pairs):
        log_likelihood[:, state] = np.sum(
            site_scores[:, first, second, :], axis=1
        )

    # A uniform prior over unordered states avoids giving heterozygous states
    # twice the prior mass merely because they have two ordered representations.
    maximum = np.max(log_likelihood, axis=1, keepdims=True)
    probability = np.exp(log_likelihood - maximum)
    probability /= np.sum(probability, axis=1, keepdims=True)
    return UnorderedDiplotypePosterior(
        pairs=pairs,
        probabilities=probability,
        informative_site_count=np.sum(informative, axis=1, dtype=np.int64),
    )


def cavity_fill_unknown_alleles(
    panel: assembly_observations.FounderPanel,
    genotype_likelihoods: np.ndarray,
    observed: np.ndarray,
    rule: CavityFillRule = CavityFillRule(),
    *,
    uniform_tolerance: float = 1e-12,
) -> CavityFillResult:
    """Fill unknown cells with coherent leave-whole-bin-out evidence.

    Pair-state weights are inferred from observed, nonuniform sites outside the
    focal bin for which *all* founders are called.  At each focal site, every
    unresolved founder allele is jointly enumerated once.  Conditional on one
    assignment, each sample sums its pair-state mixture before sample
    likelihoods are multiplied.  Independent Bernoulli priors from ``panel.q``
    are included for all unresolved alleles, including the target.

    Exact enumeration costs ``O(2**U * N * K**2)`` time and
    ``O(2**U * (N + K**2))`` working memory at a site with ``U`` unresolved
    founders.  Sites over the configured cap conservatively remain unresolved.
    Effective carrier support is the sum of leave-bin-out probabilities of
    carrying the target over focal-informative samples whose likelihood is
    sensitive to that target allele; it gates release but never scales an LLR.

    A target founder is ineligible when another founder has the same q vector
    on eligible outside-bin inference sites, within
    ``founder_identifiability_tolerance``.  All releases are computed against
    the immutable input panel.  Initially released calls and their q/support
    values are copied exactly.
    """

    evidence, informative_focal = _informative_cells(
        genotype_likelihoods, observed, uniform_tolerance
    )
    if evidence.shape[1] != panel.q.shape[1]:
        raise ValueError("genotype likelihood sites must match the panel")
    observed = np.asarray(observed, dtype=np.bool_)
    k, n_sites = panel.q.shape
    q = panel.q.copy()
    called = panel.called.copy()
    filled = np.zeros_like(called)
    posterior_alt = panel.q.copy()
    effective = np.zeros_like(q)
    exchangeable_skipped = np.zeros_like(called)
    insufficient_separation_skipped = np.zeros_like(called)
    enumeration_limit_skipped = np.zeros_like(called)
    minimum_observable_separation = np.full(q.shape, -1, dtype=np.int64)

    n_bins = (n_sites + rule.snps_per_bin - 1) // rule.snps_per_bin
    for bin_index in range(n_bins):
        start = bin_index * rule.snps_per_bin
        stop = min(start + rule.snps_per_bin, n_sites)
        outside = np.ones(n_sites, dtype=np.bool_)
        outside[start:stop] = False
        inference_outside = outside & np.all(panel.called, axis=0)
        observable_outside = inference_outside & np.any(
            informative_focal, axis=0
        )
        exchangeable_founder = np.zeros(k, dtype=np.bool_)
        minimum_separation = np.full(k, -1, dtype=np.int64)
        if k > 1:
            separation = np.zeros((k, k), dtype=np.int64)
            for first in range(k):
                for second in range(first + 1, k):
                    count = int(np.sum(
                        np.abs(
                            panel.q[first, observable_outside]
                            - panel.q[second, observable_outside]
                        ) > rule.founder_identifiability_tolerance
                    ))
                    separation[first, second] = count
                    separation[second, first] = count
            for founder in range(k):
                other = np.arange(k) != founder
                minimum_separation[founder] = int(
                    np.min(separation[founder, other])
                )
                exchangeable_founder[founder] = (
                    minimum_separation[founder] == 0
                )
        minimum_observable_separation[:, start:stop] = minimum_separation[:, None]

        cavity = _posterior_with_site_mask(
            panel,
            evidence,
            observed,
            outside,
            uniform_mix=rule.uniform_mix,
            uniform_tolerance=uniform_tolerance,
        )

        for site in range(start, stop):
            unresolved = np.flatnonzero(~panel.called[:, site])
            if unresolved.size == 0:
                continue
            if (
                unresolved.size
                > rule.maximum_enumerated_unresolved_founders
            ):
                enumeration_limit_skipped[unresolved, site] = True
                continue

            joint = exact_shared_latent_site_posterior(
                panel,
                site,
                evidence[:, site, :],
                observed[:, site],
                cavity.pairs,
                cavity.probabilities,
                uniform_mix=rule.uniform_mix,
                uniform_tolerance=uniform_tolerance,
                tie_tolerance=rule.tie_tolerance,
            )
            posterior_alt[unresolved, site] = joint.posterior_alt[unresolved]
            effective[unresolved, site] = (
                joint.effective_carrier_support[unresolved]
            )
            if not joint.finite_joint_mass:
                continue

            for founder in unresolved:
                if exchangeable_founder[founder]:
                    exchangeable_skipped[founder, site] = True
                    continue
                if (
                    k > 1
                    and minimum_separation[founder]
                    < rule.minimum_observable_founder_separation
                ):
                    insufficient_separation_skipped[founder, site] = True
                    continue

                probability_alt = joint.posterior_alt[founder]
                target_log_odds = joint.log_posterior_odds[founder]
                total_support = joint.effective_carrier_support[founder]
                confidence = max(probability_alt, 1.0 - probability_alt)
                if (
                    abs(target_log_odds) > rule.tie_tolerance
                    and total_support >= rule.minimum_effective_supporters
                    and confidence >= rule.minimum_posterior_probability
                ):
                    # A released founder allele is a hard biological call.
                    # Its calibrated marginal remains in posterior_alt; a soft
                    # q here would reintroduce the shared-latent factorization
                    # problem this cavity step resolves.
                    q[founder, site] = float(probability_alt > 0.5)
                    called[founder, site] = True
                    filled[founder, site] = True

    filled_panel = assembly_observations.FounderPanel(
        positions=panel.positions,
        keys=panel.keys,
        q=q,
        called=called,
        support=panel.support,
    )
    return CavityFillResult(
        panel=filled_panel,
        filled=filled,
        posterior_alt=posterior_alt,
        effective_supporters=effective,
        exchangeable_skipped=exchangeable_skipped,
        minimum_observable_separation=minimum_observable_separation,
        insufficient_separation_skipped=insufficient_separation_skipped,
        enumeration_limit_skipped=enumeration_limit_skipped,
        n_exchangeable_skipped=int(np.sum(exchangeable_skipped)),
        n_insufficient_separation_skipped=int(
            np.sum(insufficient_separation_skipped)
        ),
        n_enumeration_limit_skipped=int(np.sum(enumeration_limit_skipped)),
    )


