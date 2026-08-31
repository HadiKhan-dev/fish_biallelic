"""Founder-source posterior for candidate parents from normalized raw GLs.

This module is an integration-neutral building block for pedigree scoring.  It
uses a candidate's reads exactly once, here, to infer a linked founder-source
distribution.  Exact downstream pedigree likelihoods must consume the linked
posterior-chain factors, not products of ``track_alt_probability``: per-site
marginals do not retain the correlation induced by a shared founder state.
No separate candidate likelihood or HMM normalizer may be added to downstream
evidence.

At bin ``b`` the hidden state is the ordered founder diplotype ``(I_b, J_b)``.
The two tracks independently follow the same founder-switch transition, so an
ordered-state transition is ``T1 (x) T1``.  Emissions are symmetric in the two
tracks and use the normalized three-genotype raw likelihood vector ``L``::

    eta * sum_s log(3 * dot(L_s, Q(I_b, J_b, s)))

where ``Q`` is the genotype distribution implied by the two founder alleles.
Two copies of the same missing founder allele share one fixed latent allele;
they therefore form only a homozygous reference/alternate mixture, never a
spurious heterozygote.  Distinct missing founders use independent founder-only
allele priors.  The factor three makes a uniform GL vector neutral.  Missing
founder alleles
are integrated with a frequency estimated only from called founders at that
site; a site with no called founder is unavailable.

Unphased GLs cannot identify the global homolog labels.  An optional painted
track label can therefore orient, but never reweight, each unordered founder
pair at the first informative bin.  With no anchor a deterministic
founder-sequence ordering fixes an arbitrary gauge.  In either case the two
output tracks are posterior summaries in that gauge, not independently
identified biological phase.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional

import thread_config  # configure numerical thread pools before NumPy/Numba

import numba
from numba import prange

thread_config.ensure_numba_registry_warmup()
njit = thread_config.original_njit

import numpy as np


@dataclass(frozen=True)
class CandidateSourcePosterior:
    """Posterior summaries for one contig.

    ``track_alt_probability`` has shape ``(samples, bins, 2, snp_slots)`` and
    is a diagnostic marginal only; multiplying it across sites is not an exact
    conditional child likelihood.  The three ``linked_*`` arrays, when
    requested, retain the exact posterior path law in compact factor form.
    ``source_posterior``, when requested, has shape
    ``(samples, bins, founders, founders)`` and retains the ordered state axes.
    It is ``None`` by default because that debug tensor scales as
    ``8 * samples * bins * founders**2`` bytes.  The production-facing track
    array is float32 and scales as ``8 * samples * bins * snp_slots`` bytes.
    Unavailable samples and unavailable SNP slots are represented by NaN, with
    the corresponding Boolean flags making that absence explicit.

    No candidate-source log normalizer is returned: the candidate reads have
    already been used to form this posterior and are not an additional piece
    of downstream pedigree evidence.
    """

    track_alt_probability: np.ndarray
    source_posterior: Optional[np.ndarray]
    linked_initial_log_probability: Optional[np.ndarray]
    linked_next_log_weight: Optional[np.ndarray]
    linked_transition_probability: Optional[np.ndarray]
    lumped_initial_log_probability: Optional[np.ndarray]
    lumped_next_log_weight: Optional[np.ndarray]
    lumped_transition_probability: Optional[np.ndarray]
    lumped_available: np.ndarray
    lumped_site_available: np.ndarray
    lumped_informative_site_count: np.ndarray
    lumped_informative_bins: np.ndarray
    lumped_anchor_bin: np.ndarray
    available: np.ndarray
    informative_bins: np.ndarray
    informative_sites: np.ndarray
    informative_site_count: np.ndarray
    founder_site_available: np.ndarray
    founder_alt_frequency: np.ndarray
    posterior_entropy: np.ndarray
    max_state_posterior: np.ndarray
    gauge_anchored: np.ndarray
    gauge_anchor_bin: np.ndarray
    gauge_canonical_swap: np.ndarray
    canonical_founder_order: np.ndarray
    inconsistent: np.ndarray
    lumped_root_prior_mode: str


def _normalise_linear_gl(genotype_likelihoods: np.ndarray) -> np.ndarray:
    gl = np.asarray(genotype_likelihoods, dtype=np.float64)
    if gl.ndim != 4 or gl.shape[-1] != 3:
        raise ValueError(
            "genotype_likelihoods must have shape "
            "(samples, bins, snp_slots, 3)"
        )
    if np.any(~np.isfinite(gl)) or np.any(gl < 0.0):
        raise ValueError("genotype likelihoods must be finite and nonnegative")
    total = np.sum(gl, axis=-1, keepdims=True)
    if np.any(total <= 0.0):
        raise ValueError("every genotype-likelihood vector must contain mass")
    return gl / total


def _marker_counts(
    selected_markers_per_bin: np.ndarray, n_bins: int, n_slots: int
) -> np.ndarray:
    counts = np.asarray(selected_markers_per_bin, dtype=np.int64)
    if counts.shape != (n_bins,):
        raise ValueError("selected_markers_per_bin must have shape (bins,)")
    if np.any(counts < 0) or np.any(counts > n_slots):
        raise ValueError("selected marker counts must be between 0 and snp_slots")
    return counts


def _switch_probabilities(switch_probability: np.ndarray, n_bins: int) -> np.ndarray:
    value = np.asarray(switch_probability, dtype=np.float64)
    if value.ndim == 0:
        boundary = np.full(max(n_bins - 1, 0), float(value), dtype=np.float64)
    elif value.shape == (max(n_bins - 1, 0),):
        boundary = value.copy()
    elif value.shape == (n_bins,):
        boundary = value[1:].copy()
    else:
        raise ValueError(
            "switch_probability must be scalar, shape (bins - 1,), "
            "or shape (bins,) with element zero unused"
        )
    if np.any(~np.isfinite(boundary)) or np.any(
        (boundary < 0.0) | (boundary > 1.0)
    ):
        raise ValueError("switch probabilities must be finite and in [0, 1]")
    return boundary


def _eta_matrix(eta: np.ndarray, n_samples: int, n_bins: int) -> np.ndarray:
    value = np.asarray(eta, dtype=np.float64)
    if value.ndim == 0:
        result = np.full((n_samples, n_bins), float(value), dtype=np.float64)
    elif value.shape == (n_bins,):
        result = np.broadcast_to(value[None, :], (n_samples, n_bins)).copy()
    elif value.shape == (n_samples, n_bins):
        result = value.copy()
    else:
        raise ValueError("eta must be scalar, shape (bins,), or shape (samples, bins)")
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("eta must be finite and strictly positive")
    return result


def _founder_probabilities(
    founder_alleles: np.ndarray, marker_counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    founders = np.asarray(founder_alleles)
    if founders.ndim != 3:
        raise ValueError("founder_alleles must have shape (founders, bins, snp_slots)")
    if np.any(~np.isin(founders, (-1, 0, 1))):
        raise ValueError("founder alleles must be -1 (missing), 0, or 1")

    n_founders, n_bins, n_slots = founders.shape
    frequency = np.full((n_bins, n_slots), np.nan, dtype=np.float64)
    available = np.zeros((n_bins, n_slots), dtype=bool)
    probability = np.full((n_founders, n_bins, n_slots), np.nan, dtype=np.float64)
    for block in range(n_bins):
        for snp in range(int(marker_counts[block])):
            called = founders[:, block, snp] >= 0
            if not np.any(called):
                continue
            site_frequency = float(np.mean(founders[called, block, snp]))
            frequency[block, snp] = site_frequency
            available[block, snp] = True
            probability[:, block, snp] = np.where(
                called, founders[:, block, snp], site_frequency
            )
    return probability, frequency, available


def _canonical_founder_keys(
    founder_alleles: np.ndarray, marker_counts: np.ndarray
) -> list[tuple[int, ...]]:
    """Return founder-content keys independent of founder array order."""
    keys: list[tuple[int, ...]] = []
    for founder in range(founder_alleles.shape[0]):
        values: list[int] = []
        for block, count in enumerate(marker_counts):
            values.extend(
                int(x) for x in founder_alleles[founder, block, : int(count)]
            )
        keys.append(tuple(values))
    return keys


def _root_prior(
    n_founders: int,
    founder_keys: list[tuple[int, ...]],
    anchor: Optional[np.ndarray],
) -> np.ndarray:
    """Gauge-fix a prior that is uniform over unordered diplotypes."""
    prior = np.zeros((n_founders, n_founders), dtype=np.float64)
    unordered_states = n_founders * (n_founders + 1) // 2
    state_mass = 1.0 / float(unordered_states)
    first_anchor = second_anchor = -1
    if anchor is not None:
        first_anchor, second_anchor = (int(anchor[0]), int(anchor[1]))

    for first in range(n_founders):
        prior[first, first] = state_mass
        for second in range(first + 1, n_founders):
            if anchor is not None:
                direct = int(first == first_anchor) + int(second == second_anchor)
                swapped = int(second == first_anchor) + int(first == second_anchor)
            else:
                direct = int(founder_keys[first] < founder_keys[second])
                swapped = int(founder_keys[second] < founder_keys[first])
            if direct > swapped:
                prior[first, second] = state_mass
            elif swapped > direct:
                prior[second, first] = state_mass
            else:
                prior[first, second] = 0.5 * state_mass
                prior[second, first] = 0.5 * state_mass
    return prior


def _physical_root_prior(n_founders: int) -> np.ndarray:
    """Swap-symmetric prior with uniform mass per unordered diplotype."""
    prior = np.full(
        (n_founders, n_founders),
        0.5 / float(n_founders * (n_founders + 1) // 2),
        dtype=np.float64,
    )
    np.fill_diagonal(
        prior, 1.0 / float(n_founders * (n_founders + 1) // 2)
    )
    return prior


def _ordered_independent_uniform_root_prior(
    n_founders: int,
) -> np.ndarray:
    """Ordered diploid root from two independent uniform homolog draws."""
    return np.full(
        (n_founders, n_founders),
        1.0 / float(n_founders * n_founders),
        dtype=np.float64,
    )


def _transition_matrix(n_founders: int, switch_probability: float) -> np.ndarray:
    if n_founders == 1:
        return np.ones((1, 1), dtype=np.float64)
    transition = np.full(
        (n_founders, n_founders),
        float(switch_probability) / float(n_founders - 1),
        dtype=np.float64,
    )
    np.fill_diagonal(transition, 1.0 - float(switch_probability))
    return transition


def _candidate_emissions(
    gl: np.ndarray,
    founder_alleles: np.ndarray,
    founder_probability: np.ndarray,
    founder_site_available: np.ndarray,
    marker_counts: np.ndarray,
    eta: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Return emissions with shared latent alleles for repeated founders."""
    n_bins = gl.shape[0]
    n_founders = founder_probability.shape[0]
    emissions = np.ones((n_bins, n_founders, n_founders), dtype=np.float64)
    for block in range(n_bins):
        count = int(marker_counts[block])
        sites = np.flatnonzero(founder_site_available[block, :count])
        if len(sites) == 0:
            continue
        first_alt = founder_probability[:, block, sites].T[:, :, None]
        second_alt = founder_probability[:, block, sites].T[:, None, :]
        dosage = np.empty((len(sites), n_founders, n_founders, 3), dtype=np.float64)
        dosage[..., 0] = (1.0 - first_alt) * (1.0 - second_alt)
        dosage[..., 1] = (
            first_alt * (1.0 - second_alt)
            + (1.0 - first_alt) * second_alt
        )
        dosage[..., 2] = first_alt * second_alt
        for site_index, snp in enumerate(sites):
            missing = np.flatnonzero(founder_alleles[:, block, snp] < 0)
            for founder in missing:
                frequency = founder_probability[founder, block, snp]
                dosage[site_index, founder, founder, 0] = 1.0 - frequency
                dosage[site_index, founder, founder, 1] = 0.0
                dosage[site_index, founder, founder, 2] = frequency
        overlap = 3.0 * np.einsum(
            "sg,sijg->sij", gl[block, sites], dosage, optimize=True
        )
        with np.errstate(divide="ignore"):
            log_emission = float(eta[block]) * np.sum(np.log(overlap), axis=0)
        maximum = float(np.max(log_emission))
        if not np.isfinite(maximum):
            return emissions, True
        emissions[block] = np.exp(log_emission - maximum)
    return emissions, False


def _separable_forward_backward(
    emissions: np.ndarray,
    transitions: list[np.ndarray],
    root_prior: np.ndarray,
    root: int,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return state marginals and compact exact posterior-chain factors."""
    node_weight = emissions.copy()
    node_weight[root] *= root_prior
    n_bins = emissions.shape[0]
    alpha = np.empty_like(emissions)
    alpha[0] = node_weight[0]
    total = float(np.sum(alpha[0]))
    if total <= 0.0:
        return None
    alpha[0] /= total
    for block in range(1, n_bins):
        transition = transitions[block - 1]
        predicted = transition.T @ alpha[block - 1] @ transition
        alpha[block] = predicted * node_weight[block]
        total = float(np.sum(alpha[block]))
        if total <= 0.0:
            return None
        alpha[block] /= total

    beta = np.ones_like(emissions)
    for block in range(n_bins - 2, -1, -1):
        transition = transitions[block]
        following = node_weight[block + 1] * beta[block + 1]
        beta[block] = transition @ following @ transition.T
        scale = float(np.sum(beta[block]))
        if scale <= 0.0:
            return None
        beta[block] /= scale

    posterior = alpha * beta
    totals = np.sum(posterior, axis=(1, 2), keepdims=True)
    if np.any(totals <= 0.0):
        return None
    posterior /= totals

    next_log_weight = np.empty(
        (n_bins - 1,) + emissions.shape[1:], dtype=np.float64
    )
    for block in range(1, n_bins):
        right_weight = node_weight[block] * beta[block]
        maximum = float(np.max(right_weight))
        if maximum <= 0.0:
            return None
        with np.errstate(divide="ignore"):
            next_log_weight[block - 1] = np.log(right_weight / maximum)
    return posterior, next_log_weight


def _conditional_track_alt_by_state(
    gl: np.ndarray,
    founder_alleles: np.ndarray,
    founder_probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior ALT probability per ordered state for one candidate site."""
    first_prior = founder_probability[:, None]
    second_prior = founder_probability[None, :]
    q0 = (1.0 - first_prior) * (1.0 - second_prior)
    q1 = (
        first_prior * (1.0 - second_prior)
        + (1.0 - first_prior) * second_prior
    )
    q2 = first_prior * second_prior
    denominator = gl[0] * q0 + gl[1] * q1 + gl[2] * q2
    first_numerator = first_prior * (
        (1.0 - second_prior) * gl[1] + second_prior * gl[2]
    )
    second_numerator = second_prior * (
        (1.0 - first_prior) * gl[1] + first_prior * gl[2]
    )
    first = np.divide(
        first_numerator,
        denominator,
        out=np.broadcast_to(first_prior, denominator.shape).copy(),
        where=denominator > 0.0,
    )
    second = np.divide(
        second_numerator,
        denominator,
        out=np.broadcast_to(second_prior, denominator.shape).copy(),
        where=denominator > 0.0,
    )
    for founder in np.flatnonzero(founder_alleles < 0):
        frequency = founder_probability[founder]
        shared_denominator = (1.0 - frequency) * gl[0] + frequency * gl[2]
        shared_alt = frequency
        if shared_denominator > 0.0:
            shared_alt = frequency * gl[2] / shared_denominator
        first[founder, founder] = shared_alt
        second[founder, founder] = shared_alt
    return first, second


def infer_candidate_source_posterior(
    genotype_likelihoods: np.ndarray,
    founder_alleles: np.ndarray,
    selected_markers_per_bin: np.ndarray,
    switch_probability: np.ndarray,
    *,
    eta: np.ndarray = 1.0,
    painted_track_labels: Optional[np.ndarray] = None,
    uniform_tolerance: float = 1e-12,
    return_state_posterior: bool = False,
    return_linked_posterior: bool = False,
    return_lumped_posterior: bool = False,
    lumped_root_prior_mode: str = "uniform_unordered",
    posterior_factor_dtype: np.dtype = np.float32,
) -> CandidateSourcePosterior:
    """Infer candidate founder-source posteriors once for one contig.

    Parameters
    ----------
    genotype_likelihoods
        Nonnegative linear raw GLs with shape ``(samples, bins, slots, 3)``.
        Each vector is normalized internally, making arbitrary per-vector GL
        scale irrelevant.  A uniform vector is missing/uninformative.
    founder_alleles
        Hard founder panel with shape ``(founders, bins, slots)`` and values
        ``{-1, 0, 1}``, where -1 is missing.
    selected_markers_per_bin
        Number of real (non-padding) SNP slots in each bin.
    switch_probability
        The per-homolog founder-switch probability, scalar or one value per
        boundary.  A switch chooses each other founder with probability
        ``rho / (F - 1)``.
    eta
        Positive raw-GL tempering exponent, scalar, per-bin, or per-sample/bin.
    painted_track_labels
        Optional integer array ``(samples, bins, 2)``.  At each sample's first
        informative bin only, these labels orient the homolog gauge.  They do
        not affect unordered diplotype prior mass or any emission.
    return_state_posterior
        Retain the full float64 ordered-state posterior for exact/debug use.
        False by default to avoid ``O(samples * bins * founders**2)`` retained
        memory.  Candidate working memory still uses one ``(bins, F, F)``
        lattice at a time.
    return_linked_posterior
        Retain a float32 exact-path representation: the initial ordered-state
        log probabilities, one backward-conditioned log-weight matrix per
        boundary, and the shared one-track transition matrices.  This costs
        ``O(samples * bins * founders**2)`` storage but preserves linkage.
    return_lumped_posterior
        Retain a separate swap-symmetric physical chain for exact strong
        lumping with the transmitted homolog.  Gauge-oriented linked factors
        remain available for FFBS validation.
    lumped_root_prior_mode
        ``"uniform_unordered"`` preserves the B4 prior with equal mass per
        unordered founder diplotype. ``"ordered_independent_uniform"`` uses
        two independent uniform founder draws, the matched-null B5a prior.
    posterior_factor_dtype
        Float32 (default storage seam) or float64 (sensitivity reference) for
        retained linked/lumped factors.

    Notes
    -----
    The returned probabilities are intended as a precomputed source object at
    the integration seam of a parent-state GL HMM.  Do not add a likelihood
    normalizer from this inference to downstream pedigree evidence.  Default
    retained and peak working memory scale as ``O(S * B * K + B * F**2)``:
    candidates are processed sequentially, while the separable transition uses
    ``O(B * F**3)`` arithmetic instead of a dense ``O(B * F**4)`` update.
    """
    try:
        factor_dtype = np.dtype(posterior_factor_dtype)
    except TypeError as error:
        raise ValueError("posterior_factor_dtype must be float32 or float64") from error
    if factor_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("posterior_factor_dtype must be float32 or float64")
    if lumped_root_prior_mode not in {
        "uniform_unordered", "ordered_independent_uniform"
    }:
        raise ValueError(
            "lumped_root_prior_mode must be 'uniform_unordered' or "
            "'ordered_independent_uniform'"
        )
    if not np.isfinite(uniform_tolerance) or uniform_tolerance < 0.0:
        raise ValueError("uniform_tolerance must be finite and nonnegative")
    gl = _normalise_linear_gl(genotype_likelihoods)
    founders = np.asarray(founder_alleles)
    if founders.ndim != 3:
        raise ValueError("founder_alleles must have shape (founders, bins, snp_slots)")
    n_samples, n_bins, n_slots, _ = gl.shape
    n_founders, founder_bins, founder_slots = founders.shape
    if n_bins < 1 or n_founders < 1:
        raise ValueError("at least one bin and one founder are required")
    if (founder_bins, founder_slots) != (n_bins, n_slots):
        raise ValueError("founder and genotype-likelihood bin/slot shapes must match")

    marker_counts = _marker_counts(selected_markers_per_bin, n_bins, n_slots)
    boundary_switch = _switch_probabilities(switch_probability, n_bins)
    eta_values = _eta_matrix(eta, n_samples, n_bins)
    founder_probability, founder_frequency, founder_available = (
        _founder_probabilities(founders, marker_counts)
    )

    anchor_labels: Optional[np.ndarray]
    if painted_track_labels is None:
        anchor_labels = None
    else:
        anchor_labels = np.asarray(painted_track_labels, dtype=np.int64)
        if anchor_labels.shape != (n_samples, n_bins, 2):
            raise ValueError("painted_track_labels must have shape (samples, bins, 2)")
        if np.any((anchor_labels < -1) | (anchor_labels >= n_founders)):
            raise ValueError("painted track labels must be -1 or valid founder indices")

    real_site = np.arange(n_slots)[None, :] < marker_counts[:, None]
    candidate_nonuniform = np.ptp(gl, axis=-1) > float(uniform_tolerance)
    informative_sites = (
        candidate_nonuniform
        & founder_available[None, :, :]
        & real_site[None, :, :]
    )
    informative_bins = np.any(informative_sites, axis=2)
    informative_count = np.sum(informative_sites, axis=(1, 2)).astype(np.int64)
    available = informative_count > 0
    lumped_site_available = (
        np.all(founders >= 0, axis=0) & real_site
    )
    lumped_informative_sites = (
        candidate_nonuniform
        & lumped_site_available[None, :, :]
        & real_site[None, :, :]
    )
    lumped_informative_bins = np.any(lumped_informative_sites, axis=2)
    lumped_informative_count = np.sum(
        lumped_informative_sites, axis=(1, 2)
    ).astype(np.int64)
    lumped_available = lumped_informative_count > 0

    source_posterior = None
    if return_state_posterior:
        source_posterior = np.full(
            (n_samples, n_bins, n_founders, n_founders),
            np.nan,
            dtype=np.float64,
        )
    linked_initial = None
    linked_next = None
    linked_transition = None
    lumped_initial = None
    lumped_next = None
    lumped_transition = None
    if return_linked_posterior:
        linked_initial = np.full(
            (n_samples, n_founders, n_founders), np.nan, dtype=factor_dtype
        )
        linked_next = np.full(
            (n_samples, n_bins - 1, n_founders, n_founders),
            np.nan,
            dtype=factor_dtype,
        )
    if return_lumped_posterior:
        lumped_initial = np.full(
            (n_samples, n_founders, n_founders), np.nan, dtype=factor_dtype
        )
        lumped_next = np.full(
            (n_samples, n_bins - 1, n_founders, n_founders),
            np.nan,
            dtype=factor_dtype,
        )
    track_alt = np.full(
        (n_samples, n_bins, 2, n_slots), np.nan, dtype=np.float32
    )
    entropy = np.full((n_samples, n_bins), np.nan, dtype=np.float64)
    maximum = np.full((n_samples, n_bins), np.nan, dtype=np.float64)
    gauge_anchored = np.zeros(n_samples, dtype=bool)
    gauge_anchor_bin = np.full(n_samples, -1, dtype=np.int64)
    gauge_canonical_swap = np.zeros(n_samples, dtype=bool)
    lumped_anchor_bin = np.full(n_samples, -1, dtype=np.int64)
    inconsistent = np.zeros(n_samples, dtype=bool)

    transitions = [
        _transition_matrix(n_founders, float(rho)) for rho in boundary_switch
    ]
    if return_linked_posterior:
        linked_transition = (
            np.stack(transitions).astype(factor_dtype, copy=False)
            if transitions
            else np.empty((0, n_founders, n_founders), dtype=factor_dtype)
        )
    if return_lumped_posterior:
        lumped_transition = (
            np.stack(transitions).astype(factor_dtype, copy=False)
            if transitions
            else np.empty((0, n_founders, n_founders), dtype=factor_dtype)
        )
    founder_keys = _canonical_founder_keys(founders, marker_counts)
    canonical_founder_order = np.asarray(
        sorted(range(n_founders), key=lambda founder: founder_keys[founder]),
        dtype=np.int64,
    )

    for sample in range(n_samples):
        if not available[sample]:
            continue
        root = int(np.flatnonzero(informative_bins[sample])[0])
        gauge_anchor_bin[sample] = root
        anchor = None
        if anchor_labels is not None and np.any(anchor_labels[sample, root] >= 0):
            anchor = anchor_labels[sample, root]
            gauge_anchored[sample] = True
            if anchor[0] >= 0 and anchor[1] >= 0 and anchor[0] != anchor[1]:
                gauge_canonical_swap[sample] = (
                    founder_keys[int(anchor[0])] > founder_keys[int(anchor[1])]
                )
        prior = _root_prior(n_founders, founder_keys, anchor)
        emissions, impossible = _candidate_emissions(
            gl[sample],
            founders,
            founder_probability,
            founder_available,
            marker_counts,
            eta_values[sample],
        )
        if impossible:
            available[sample] = False
            inconsistent[sample] = True
            continue
        posterior_result = _separable_forward_backward(
            emissions, transitions, prior, root
        )
        if posterior_result is None:
            available[sample] = False
            lumped_available[sample] = False
            inconsistent[sample] = True
            continue
        posterior, next_log_weight = posterior_result
        if lumped_initial is not None and lumped_available[sample]:
            physical_root = int(
                np.flatnonzero(lumped_informative_bins[sample])[0]
            )
            lumped_anchor_bin[sample] = physical_root
            physical_emissions, physical_impossible = _candidate_emissions(
                gl[sample],
                founders,
                founder_probability,
                lumped_site_available,
                marker_counts,
                eta_values[sample],
            )
            if physical_impossible:
                lumped_available[sample] = False
                continue
            physical_root_prior = (
                _physical_root_prior(n_founders)
                if lumped_root_prior_mode == "uniform_unordered"
                else _ordered_independent_uniform_root_prior(n_founders)
            )
            physical_result = _separable_forward_backward(
                physical_emissions,
                transitions,
                physical_root_prior,
                physical_root,
            )
            if physical_result is None:
                lumped_available[sample] = False
                continue
            physical_posterior, physical_next_log_weight = physical_result
            with np.errstate(divide="ignore"):
                lumped_initial[sample] = np.log(
                    physical_posterior[0]
                ).astype(factor_dtype)
            lumped_next[sample] = physical_next_log_weight.astype(factor_dtype)
        if linked_initial is not None:
            with np.errstate(divide="ignore"):
                linked_initial[sample] = np.log(posterior[0]).astype(factor_dtype)
            linked_next[sample] = next_log_weight.astype(factor_dtype)
        if source_posterior is not None:
            source_posterior[sample] = posterior
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy[sample] = -np.sum(
                np.where(posterior > 0.0, posterior * np.log(posterior), 0.0),
                axis=(1, 2),
            )
        maximum[sample] = np.max(posterior, axis=(1, 2))
        for block in range(n_bins):
            sites = np.flatnonzero(founder_available[block])
            if len(sites) == 0:
                continue
            for snp in sites:
                first_alt, second_alt = _conditional_track_alt_by_state(
                    gl[sample, block, snp],
                    founders[:, block, snp],
                    founder_probability[:, block, snp],
                )
                track_alt[sample, block, 0, snp] = np.sum(
                    posterior[block] * first_alt
                )
                track_alt[sample, block, 1, snp] = np.sum(
                    posterior[block] * second_alt
                )

    return CandidateSourcePosterior(
        track_alt_probability=track_alt,
        source_posterior=source_posterior,
        linked_initial_log_probability=linked_initial,
        linked_next_log_weight=linked_next,
        linked_transition_probability=linked_transition,
        lumped_initial_log_probability=lumped_initial,
        lumped_next_log_weight=lumped_next,
        lumped_transition_probability=lumped_transition,
        lumped_available=lumped_available,
        lumped_site_available=lumped_site_available,
        lumped_informative_site_count=lumped_informative_count,
        lumped_informative_bins=lumped_informative_bins,
        lumped_anchor_bin=lumped_anchor_bin,
        available=available,
        informative_bins=informative_bins,
        informative_sites=informative_sites,
        informative_site_count=informative_count,
        founder_site_available=founder_available,
        founder_alt_frequency=founder_frequency,
        posterior_entropy=entropy,
        max_state_posterior=maximum,
        gauge_anchored=gauge_anchored,
        gauge_anchor_bin=gauge_anchor_bin,
        gauge_canonical_swap=gauge_canonical_swap,
        canonical_founder_order=canonical_founder_order,
        inconsistent=inconsistent,
        lumped_root_prior_mode=lumped_root_prior_mode,
    )



@dataclass(frozen=True)
class CandidateSourceTrajectoryDraws:
    """Deterministic coherent founder-ID trajectory draws.

    ``founder_tracks`` has shape ``(samples, draws, bins, 2)`` and contains
    founder IDs only.  Missing founder alleles are generated lazily from a
    counter keyed by ``seed, draw, founder-content rank, bin, SNP`` so the same
    founder-site allele is shared wherever it is reused without materializing
    expanded allele tracks.
    """

    founder_tracks: np.ndarray
    available: np.ndarray
    n_draws: int
    seed: int
    canonical_founder_order: np.ndarray


@dataclass(frozen=True)
class MonteCarloChildLikelihood:
    """Reference coherent-path likelihood and Monte Carlo diagnostics."""

    log_likelihood: float
    log_likelihood_standard_error: float
    doubling_delta_log_likelihood: float
    n_draws: int
    n_available_parents: int
    missing_founder_draws_used: int


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if integer != value or integer < 1:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _sample_log_categorical(
    log_probability: np.ndarray,
    founder_order: np.ndarray,
    generator: np.random.Generator,
) -> tuple[int, int]:
    """Sample an ordered founder pair in founder-content order."""
    ordered = np.asarray(
        [
            log_probability[first, second]
            for first in founder_order
            for second in founder_order
        ],
        dtype=np.float64,
    )
    maximum = float(np.max(ordered))
    if not np.isfinite(maximum):
        raise ValueError("posterior trajectory has no reachable source state")
    probability = np.exp(ordered - maximum)
    probability /= np.sum(probability)
    selected = int(generator.choice(len(probability), p=probability))
    n_founders = len(founder_order)
    return (
        int(founder_order[selected // n_founders]),
        int(founder_order[selected % n_founders]),
    )


def sample_candidate_source_trajectories(
    posterior: CandidateSourcePosterior,
    *,
    n_draws: int,
    seed: int = 0,
) -> CandidateSourceTrajectoryDraws:
    """Draw coherent paths from the compact posterior-chain factorization.

    The chain factors must have been requested with
    ``return_linked_posterior=True``.  Sampling first maps an anchored result
    into the content-canonical homolog gauge and maps each draw back afterward.
    Consequently swapping the optional anchor swaps the returned tracks under
    common random numbers, while founder-ID permutations only relabel IDs.
    """
    draw_count = _positive_integer(n_draws, "n_draws")
    if isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be a non-negative integer")
    try:
        seed_value = int(seed)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("seed must be a non-negative integer") from error
    if seed_value != seed or seed_value < 0:
        raise ValueError("seed must be a non-negative integer")
    if (
        posterior.linked_initial_log_probability is None
        or posterior.linked_next_log_weight is None
        or posterior.linked_transition_probability is None
    ):
        raise ValueError(
            "posterior must be inferred with return_linked_posterior=True"
        )

    initial = posterior.linked_initial_log_probability
    next_weight = posterior.linked_next_log_weight
    transition = posterior.linked_transition_probability
    n_samples, n_founders, second_founders = initial.shape
    if n_founders != second_founders:
        raise ValueError("linked initial state must have two equal founder axes")
    n_bins = next_weight.shape[1] + 1
    tracks = np.full(
        (n_samples, draw_count, n_bins, 2), -1, dtype=np.int16
    )
    founder_order = np.asarray(posterior.canonical_founder_order, dtype=np.int64)

    for sample in range(n_samples):
        if not posterior.available[sample]:
            continue
        generator = np.random.default_rng(
            np.random.SeedSequence((seed_value, sample))
        )
        canonical_swap = bool(posterior.gauge_canonical_swap[sample])
        initial_log = initial[sample].astype(np.float64)
        if canonical_swap:
            initial_log = initial_log.T
        for draw in range(draw_count):
            first, second = _sample_log_categorical(
                initial_log, founder_order, generator
            )
            tracks[sample, draw, 0] = (first, second)
            for block in range(1, n_bins):
                right_log_weight = next_weight[
                    sample, block - 1
                ].astype(np.float64)
                if canonical_swap:
                    right_log_weight = right_log_weight.T
                one_track = transition[block - 1].astype(np.float64)
                with np.errstate(divide="ignore"):
                    next_log_probability = (
                        np.log(one_track[first])[:, None]
                        + np.log(one_track[second])[None, :]
                        + right_log_weight
                    )
                first, second = _sample_log_categorical(
                    next_log_probability, founder_order, generator
                )
                tracks[sample, draw, block] = (first, second)
        if canonical_swap:
            tracks[sample] = tracks[sample, :, :, ::-1]

    return CandidateSourceTrajectoryDraws(
        founder_tracks=tracks,
        available=np.asarray(posterior.available, dtype=bool).copy(),
        n_draws=draw_count,
        seed=seed_value,
        canonical_founder_order=founder_order.copy(),
    )


_UINT64_MASK = (1 << 64) - 1


def _mix_uint64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return value ^ (value >> 31)


def _counter_uniform(seed: int, *counters: int) -> float:
    value = _mix_uint64(seed & _UINT64_MASK)
    for counter in counters:
        value = _mix_uint64(value ^ _mix_uint64(int(counter) & _UINT64_MASK))
    return ((value >> 11) + 0.5) / float(1 << 53)


def _reference_eta(eta: np.ndarray, n_bins: int) -> np.ndarray:
    value = np.asarray(eta, dtype=np.float64)
    if value.ndim == 0:
        result = np.full(n_bins, float(value), dtype=np.float64)
    elif value.shape == (n_bins,):
        result = value.copy()
    else:
        raise ValueError("eta must be scalar or shape (bins,)")
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("eta must be finite and positive")
    return result


def _reference_track_switch(
    track_switch_probability: np.ndarray, n_bins: int
) -> np.ndarray:
    value = np.asarray(track_switch_probability, dtype=np.float64)
    if value.ndim == 0:
        result = np.full(max(n_bins - 1, 0), float(value), dtype=np.float64)
    elif value.shape == (max(n_bins - 1, 0),):
        result = value.copy()
    elif value.shape == (n_bins,):
        result = value[1:].copy()
    else:
        raise ValueError(
            "track_switch_probability must be scalar, shape (bins - 1,), "
            "or shape (bins,)"
        )
    if np.any(~np.isfinite(result)) or np.any((result < 0.0) | (result > 0.5)):
        raise ValueError("track switch probabilities must lie in [0, 0.5]")
    return result


def _reference_founder_frequency(founders: np.ndarray) -> np.ndarray:
    n_bins, n_slots = founders.shape[1:]
    frequency = np.full((n_bins, n_slots), np.nan, dtype=np.float64)
    for block in range(n_bins):
        for snp in range(n_slots):
            called = founders[:, block, snp] >= 0
            if np.any(called):
                frequency[block, snp] = np.mean(founders[called, block, snp])
    return frequency


def _reference_external_alt(
    value: Optional[np.ndarray], frequency: np.ndarray
) -> np.ndarray:
    if value is None:
        return np.where(np.isfinite(frequency), frequency, 0.5)
    result = np.asarray(value, dtype=np.float64)
    if result.shape != frequency.shape or np.any(~np.isfinite(result)) or np.any(
        (result < 0.0) | (result > 1.0)
    ):
        raise ValueError(
            "external ALT probabilities must be finite in [0, 1] with "
            "shape (bins, snp_slots)"
        )
    return result


def _reference_source_alt(
    tracks: Optional[np.ndarray],
    track: int,
    draw: int,
    block: int,
    snp: int,
    founders: np.ndarray,
    founder_frequency: np.ndarray,
    founder_rank: np.ndarray,
    missing_seed: int,
    external_alt: np.ndarray,
) -> tuple[float, int]:
    if tracks is None:
        return float(external_alt[block, snp]), 0
    founder = int(tracks[draw, block, track])
    allele = int(founders[founder, block, snp])
    if allele >= 0:
        return float(allele), 0
    frequency = founder_frequency[block, snp]
    if not np.isfinite(frequency):
        return float(external_alt[block, snp]), 0
    uniform = _counter_uniform(
        missing_seed, draw, int(founder_rank[founder]), block, snp
    )
    return float(uniform < frequency), 1


def _reference_draw_log_likelihood(
    child_gl: np.ndarray,
    founders: np.ndarray,
    marker_counts: np.ndarray,
    first_tracks: Optional[np.ndarray],
    second_tracks: Optional[np.ndarray],
    draw: int,
    first_external: np.ndarray,
    second_external: np.ndarray,
    founder_frequency: np.ndarray,
    founder_rank: np.ndarray,
    missing_seed: int,
    mismatch_probability: float,
    track_switch: np.ndarray,
    eta: np.ndarray,
) -> tuple[float, int]:
    n_bins = child_gl.shape[0]
    first_states = 1 if first_tracks is None else 2
    second_states = 1 if second_tracks is None else 2
    forward = np.full(
        (first_states, second_states),
        1.0 / float(first_states * second_states),
        dtype=np.float64,
    )
    total = 0.0
    missing_used = 0
    for block in range(n_bins):
        if block > 0:
            theta = track_switch[block - 1]
            first_transition = (
                np.ones((1, 1), dtype=np.float64)
                if first_states == 1
                else np.asarray(((1.0 - theta, theta), (theta, 1.0 - theta)))
            )
            second_transition = (
                np.ones((1, 1), dtype=np.float64)
                if second_states == 1
                else np.asarray(((1.0 - theta, theta), (theta, 1.0 - theta)))
            )
            forward = first_transition.T @ forward @ second_transition

        log_emission = np.zeros_like(forward)
        for first_track in range(first_states):
            for second_track in range(second_states):
                value = 0.0
                for snp in range(int(marker_counts[block])):
                    first_alt, first_missing = _reference_source_alt(
                        first_tracks,
                        first_track,
                        draw,
                        block,
                        snp,
                        founders,
                        founder_frequency,
                        founder_rank,
                        missing_seed,
                        first_external,
                    )
                    second_alt, second_missing = _reference_source_alt(
                        second_tracks,
                        second_track,
                        draw,
                        block,
                        snp,
                        founders,
                        founder_frequency,
                        founder_rank,
                        missing_seed,
                        second_external,
                    )
                    missing_used += first_missing + second_missing
                    first_alt = (
                        mismatch_probability
                        + (1.0 - 2.0 * mismatch_probability) * first_alt
                    )
                    second_alt = (
                        mismatch_probability
                        + (1.0 - 2.0 * mismatch_probability) * second_alt
                    )
                    q0 = (1.0 - first_alt) * (1.0 - second_alt)
                    q1 = (
                        first_alt * (1.0 - second_alt)
                        + (1.0 - first_alt) * second_alt
                    )
                    q2 = first_alt * second_alt
                    likelihood = float(
                        np.dot(child_gl[block, snp], (q0, q1, q2))
                    )
                    value += np.log(max(3.0 * likelihood, np.finfo(float).tiny))
                log_emission[first_track, second_track] = eta[block] * value
        maximum = float(np.max(log_emission))
        forward *= np.exp(log_emission - maximum)
        normalizer = float(np.sum(forward))
        if normalizer <= 0.0:
            return -np.inf, missing_used
        total += maximum + np.log(normalizer)
        forward /= normalizer
    return total, missing_used


def reference_conditional_child_likelihood_mc(
    child_genotype_likelihoods: np.ndarray,
    founder_alleles: np.ndarray,
    selected_markers_per_bin: np.ndarray,
    *,
    first_draws: Optional[CandidateSourceTrajectoryDraws] = None,
    first_sample: int = 0,
    second_draws: Optional[CandidateSourceTrajectoryDraws] = None,
    second_sample: int = 0,
    external_first_alt_probability: Optional[np.ndarray] = None,
    external_second_alt_probability: Optional[np.ndarray] = None,
    mismatch_probability: float = 0.01,
    track_switch_probability: np.ndarray = 0.01,
    eta: np.ndarray = 1.0,
    n_draws: Optional[int] = None,
) -> MonteCarloChildLikelihood:
    """Reference conditional M0/M1/M2 likelihood over coherent source draws.

    Candidate paths are paired by equal draw index, giving a symmetric common-
    random-number M2 estimator.  Likelihoods, not log likelihoods, are averaged.
    A missing founder allele is a deterministic founder-site/draw latent shared
    across both candidates.  This is a controlled founder-prior approximation:
    candidate-specific reads do not jointly update that shared missing allele.
    Sites with no called founder fall back to the supplied external probability,
    never ALT=0.
    """
    child = np.asarray(child_genotype_likelihoods, dtype=np.float64)
    if child.ndim != 3 or child.shape[2] != 3:
        raise ValueError("child genotype likelihoods must have shape (bins, slots, 3)")
    if np.any(~np.isfinite(child)) or np.any(child < 0.0):
        raise ValueError("child genotype likelihoods must be finite and nonnegative")
    total = np.sum(child, axis=2, keepdims=True)
    if np.any(total <= 0.0):
        raise ValueError("every child genotype-likelihood vector must contain mass")
    child = child / total
    founders = np.asarray(founder_alleles, dtype=np.int8)
    if founders.ndim != 3 or founders.shape[1:] != child.shape[:2]:
        raise ValueError("founder alleles must have shape (founders, bins, slots)")
    if np.any(~np.isin(founders, (-1, 0, 1))):
        raise ValueError("founder alleles must be -1, 0, or 1")
    marker_counts = _marker_counts(
        selected_markers_per_bin, child.shape[0], child.shape[1]
    )
    if not np.isfinite(mismatch_probability) or not 0.0 <= mismatch_probability < 0.5:
        raise ValueError("mismatch_probability must lie in [0, 0.5)")
    exponent = _reference_eta(eta, child.shape[0])
    track_switch = _reference_track_switch(
        track_switch_probability, child.shape[0]
    )
    frequency = _reference_founder_frequency(founders)
    first_external = _reference_external_alt(
        external_first_alt_probability, frequency
    )
    second_external = _reference_external_alt(
        external_second_alt_probability, frequency
    )
    founder_keys = _canonical_founder_keys(founders, marker_counts)
    order = sorted(range(founders.shape[0]), key=lambda index: founder_keys[index])
    rank = np.empty(founders.shape[0], dtype=np.int64)
    rank[np.asarray(order, dtype=np.int64)] = np.arange(founders.shape[0])

    first_available = (
        first_draws is not None and bool(first_draws.available[first_sample])
    )
    second_available = (
        second_draws is not None and bool(second_draws.available[second_sample])
    )
    first_tracks = (
        first_draws.founder_tracks[first_sample] if first_available else None
    )
    second_tracks = (
        second_draws.founder_tracks[second_sample] if second_available else None
    )
    active = [
        draws
        for draws, available in (
            (first_draws, first_available),
            (second_draws, second_available),
        )
        if available
    ]
    if active:
        if len(active) == 2 and active[0].seed != active[1].seed:
            raise ValueError(
                "M2 common-random-number draws must use the same seed"
            )
        maximum_draws = min(draws.n_draws for draws in active)
        missing_seed = active[0].seed
    else:
        maximum_draws = 1
        missing_seed = 0
    draw_count = maximum_draws if n_draws is None else _positive_integer(
        n_draws, "n_draws"
    )
    if draw_count > maximum_draws:
        raise ValueError("n_draws exceeds the available coherent trajectories")

    log_likelihoods = np.empty(draw_count, dtype=np.float64)
    missing_used = 0
    for draw in range(draw_count):
        value, count = _reference_draw_log_likelihood(
            child,
            founders,
            marker_counts,
            first_tracks,
            second_tracks,
            draw,
            first_external,
            second_external,
            frequency,
            rank,
            missing_seed,
            float(mismatch_probability),
            track_switch,
            exponent,
        )
        log_likelihoods[draw] = value
        missing_used += count

    maximum = float(np.max(log_likelihoods))
    scaled = np.exp(log_likelihoods - maximum)
    mean_scaled = float(np.mean(scaled))
    log_likelihood = maximum + np.log(mean_scaled)
    if draw_count > 1 and mean_scaled > 0.0:
        standard_error = float(
            np.std(scaled, ddof=1) / np.sqrt(draw_count) / mean_scaled
        )
    else:
        standard_error = 0.0
    if draw_count >= 2:
        half = draw_count // 2
        first_half = log_likelihoods[:half]
        half_maximum = float(np.max(first_half))
        half_log_likelihood = half_maximum + np.log(
            np.mean(np.exp(first_half - half_maximum))
        )
        doubling_delta = log_likelihood - half_log_likelihood
    else:
        doubling_delta = np.nan
    return MonteCarloChildLikelihood(
        log_likelihood=float(log_likelihood),
        log_likelihood_standard_error=standard_error,
        doubling_delta_log_likelihood=float(doubling_delta),
        n_draws=draw_count,
        n_available_parents=int(first_available) + int(second_available),
        missing_founder_draws_used=missing_used,
    )

@dataclass(frozen=True)
class ExactTensorChildLikelihood:
    """Deterministic exact conditional child likelihood diagnostics."""

    log_likelihood: float
    mode: str
    hidden_state_count: int
    peak_forward_bytes: int
    excluded_marker_count: int


def _compound_transition_axis(
    values: np.ndarray, transition: np.ndarray, axis: int
) -> np.ndarray:
    """Apply a compound-symmetry transition along one tensor axis."""
    matrix = np.asarray(transition, dtype=np.float64)
    n_states = matrix.shape[0]
    if matrix.shape != (n_states, n_states):
        raise ValueError("transition must be square")
    if n_states == 1:
        return values.copy()
    off_diagonal = float(matrix[0, 1])
    diagonal = float(matrix[0, 0])
    expected = np.full_like(matrix, off_diagonal)
    np.fill_diagonal(expected, diagonal)
    if not np.allclose(matrix, expected, rtol=0.0, atol=2e-7):
        raise ValueError("founder transition must have compound symmetry")
    coefficient = diagonal - off_diagonal
    return coefficient * values + off_diagonal * np.sum(
        values, axis=axis, keepdims=True
    )


def _dense_transition_axis(
    values: np.ndarray, transition: np.ndarray, axis: int
) -> np.ndarray:
    """Dense reference update ``out[j] = sum_i values[i] T[i,j]``."""
    moved = np.moveaxis(values, axis, 0)
    updated = np.tensordot(
        np.asarray(transition, dtype=np.float64).T, moved, axes=(1, 0)
    )
    return np.moveaxis(updated, 0, axis)


def _broadcast_pair(
    matrix: np.ndarray, ndim: int, first_axis: int, second_axis: int
) -> np.ndarray:
    shape = [1] * ndim
    shape[first_axis] = matrix.shape[0]
    shape[second_axis] = matrix.shape[1]
    return matrix.reshape(shape)


def _lumped_candidate_update(
    values: np.ndarray,
    posterior: CandidateSourcePosterior,
    sample: int,
    boundary: int,
    first_axis: int,
    second_axis: int,
    track_switch_probability: float,
) -> np.ndarray:
    """Exact strong-lumped candidate transition in axis-sum form."""
    transition = posterior.lumped_transition_probability[boundary].astype(
        np.float64
    )
    right = np.exp(
        posterior.lumped_next_log_weight[sample, boundary].astype(np.float64)
    )
    denominator = transition @ right @ transition.T
    broadcast_denominator = _broadcast_pair(
        denominator, values.ndim, first_axis, second_axis
    )
    scaled = np.divide(
        values,
        broadcast_denominator,
        out=np.zeros_like(values),
        where=broadcast_denominator > 0.0,
    )
    direct = _compound_transition_axis(scaled, transition, first_axis)
    direct = _compound_transition_axis(direct, transition, second_axis)
    direct *= _broadcast_pair(right, values.ndim, first_axis, second_axis)
    swapped = np.swapaxes(direct, first_axis, second_axis)
    return (
        (1.0 - track_switch_probability) * direct
        + track_switch_probability * swapped
    )


def _external_transition_update(
    values: np.ndarray, transition: np.ndarray, axis: int
) -> np.ndarray:
    matrix = np.asarray(transition, dtype=np.float64)
    if matrix.shape[0] > 1:
        off_diagonal = matrix[0, 1]
        diagonal = matrix[0, 0]
        expected = np.full_like(matrix, off_diagonal)
        np.fill_diagonal(expected, diagonal)
        if np.allclose(matrix, expected, rtol=0.0, atol=2e-7):
            return _compound_transition_axis(values, matrix, axis)
    return _dense_transition_axis(values, matrix, axis)


def _founder_pair_genotype_probability(
    first_founder: int,
    second_founder: int,
    block: int,
    snp: int,
    founder_alleles: np.ndarray,
    founder_frequency: np.ndarray,
    mismatch_probability: float,
) -> tuple[float, float, float]:
    """Genotype probabilities with one shared latent for a repeated founder."""
    first_hard = int(founder_alleles[first_founder, block, snp])
    second_hard = int(founder_alleles[second_founder, block, snp])
    frequency = founder_frequency[block, snp]
    if not np.isfinite(frequency):
        frequency = 0.5

    if first_founder == second_founder and first_hard < 0:
        ref_alt = mismatch_probability
        alt_alt = 1.0 - mismatch_probability
        q_ref = (
            (1.0 - ref_alt) ** 2,
            2.0 * ref_alt * (1.0 - ref_alt),
            ref_alt**2,
        )
        q_alt = (
            (1.0 - alt_alt) ** 2,
            2.0 * alt_alt * (1.0 - alt_alt),
            alt_alt**2,
        )
        return tuple(
            (1.0 - frequency) * q_ref[index] + frequency * q_alt[index]
            for index in range(3)
        )

    first_prior = frequency if first_hard < 0 else float(first_hard)
    second_prior = frequency if second_hard < 0 else float(second_hard)
    first_alt = (
        mismatch_probability
        + (1.0 - 2.0 * mismatch_probability) * first_prior
    )
    second_alt = (
        mismatch_probability
        + (1.0 - 2.0 * mismatch_probability) * second_prior
    )
    return (
        (1.0 - first_alt) * (1.0 - second_alt),
        first_alt * (1.0 - second_alt)
        + (1.0 - first_alt) * second_alt,
        first_alt * second_alt,
    )


def _founder_pair_bin_emission(
    child_gl: np.ndarray,
    founder_alleles: np.ndarray,
    founder_frequency: np.ndarray,
    marker_count: int,
    block: int,
    mismatch_probability: float,
    exponent: float,
    site_available: Optional[np.ndarray] = None,
) -> np.ndarray:
    n_founders = founder_alleles.shape[0]
    emission = np.ones((n_founders, n_founders), dtype=np.float64)
    for first in range(n_founders):
        for second in range(n_founders):
            value = 0.0
            for snp in range(marker_count):
                if site_available is not None and not site_available[snp]:
                    continue
                q = _founder_pair_genotype_probability(
                    first,
                    second,
                    block,
                    snp,
                    founder_alleles,
                    founder_frequency,
                    mismatch_probability,
                )
                likelihood = float(np.dot(child_gl[block, snp], q))
                value += np.log(max(3.0 * likelihood, np.finfo(float).tiny))
            emission[first, second] = np.exp(exponent * value)
    return emission


def _normalise_tensor_forward(
    values: np.ndarray,
) -> tuple[np.ndarray, float]:
    normalizer = float(np.sum(values))
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        return values, -np.inf
    return values / normalizer, np.log(normalizer)



@njit(cache=True, parallel=True, fastmath=False)
def _m2_compound_forward_kernel(
    initial,
    right_weight,
    denominator,
    diagonal,
    off_diagonal,
    track_switch,
    emission,
):
    """Compiled target-pair M2 forward kernel with O(B F^4) updates."""
    n_pairs = initial.shape[0]
    n_founders = initial.shape[1]
    n_bins = emission.shape[0]
    output = np.empty(n_pairs, dtype=np.float64)
    for pair in prange(n_pairs):
        current = initial[pair].copy()
        work1 = np.empty_like(current)
        work2 = np.empty_like(current)
        total = 0.0

        normalizer = 0.0
        for h1 in range(n_founders):
            for o1 in range(n_founders):
                for h2 in range(n_founders):
                    for o2 in range(n_founders):
                        current[h1, o1, h2, o2] *= emission[0, h1, h2]
                        normalizer += current[h1, o1, h2, o2]
        total += np.log(normalizer)
        current /= normalizer

        for boundary in range(n_bins - 1):
            coefficient = diagonal[boundary] - off_diagonal[boundary]
            if track_switch.ndim == 1:
                theta1 = track_switch[boundary]
                theta2 = theta1
            else:
                theta1 = track_switch[pair, boundary, 0]
                theta2 = track_switch[pair, boundary, 1]
            for h1 in range(n_founders):
                for o1 in range(n_founders):
                    for h2 in range(n_founders):
                        for o2 in range(n_founders):
                            combined_denominator = (
                                denominator[pair, boundary, 0, h1, o1]
                                * denominator[pair, boundary, 1, h2, o2]
                            )
                            if combined_denominator > 0.0:
                                current[h1, o1, h2, o2] /= combined_denominator
                            else:
                                current[h1, o1, h2, o2] = 0.0

            for o1 in range(n_founders):
                for h2 in range(n_founders):
                    for o2 in range(n_founders):
                        axis_sum = 0.0
                        for h1 in range(n_founders):
                            axis_sum += current[h1, o1, h2, o2]
                        for h1 in range(n_founders):
                            work1[h1, o1, h2, o2] = (
                                coefficient * current[h1, o1, h2, o2]
                                + off_diagonal[boundary] * axis_sum
                            )
            for h1 in range(n_founders):
                for h2 in range(n_founders):
                    for o2 in range(n_founders):
                        axis_sum = 0.0
                        for o1 in range(n_founders):
                            axis_sum += work1[h1, o1, h2, o2]
                        for o1 in range(n_founders):
                            work2[h1, o1, h2, o2] = (
                                coefficient * work1[h1, o1, h2, o2]
                                + off_diagonal[boundary] * axis_sum
                            )
            for h1 in range(n_founders):
                for o1 in range(n_founders):
                    for h2 in range(n_founders):
                        for o2 in range(n_founders):
                            current[h1, o1, h2, o2] = (
                                (1.0 - theta1)
                                * right_weight[pair, boundary, 0, h1, o1]
                                * work2[h1, o1, h2, o2]
                                + theta1
                                * right_weight[pair, boundary, 0, o1, h1]
                                * work2[o1, h1, h2, o2]
                            )

            for h1 in range(n_founders):
                for o1 in range(n_founders):
                    for o2 in range(n_founders):
                        axis_sum = 0.0
                        for h2 in range(n_founders):
                            axis_sum += current[h1, o1, h2, o2]
                        for h2 in range(n_founders):
                            work1[h1, o1, h2, o2] = (
                                coefficient * current[h1, o1, h2, o2]
                                + off_diagonal[boundary] * axis_sum
                            )
            for h1 in range(n_founders):
                for o1 in range(n_founders):
                    for h2 in range(n_founders):
                        axis_sum = 0.0
                        for o2 in range(n_founders):
                            axis_sum += work1[h1, o1, h2, o2]
                        for o2 in range(n_founders):
                            work2[h1, o1, h2, o2] = (
                                coefficient * work1[h1, o1, h2, o2]
                                + off_diagonal[boundary] * axis_sum
                            )
            for h1 in range(n_founders):
                for o1 in range(n_founders):
                    for h2 in range(n_founders):
                        for o2 in range(n_founders):
                            current[h1, o1, h2, o2] = (
                                (1.0 - theta2)
                                * right_weight[pair, boundary, 1, h2, o2]
                                * work2[h1, o1, h2, o2]
                                + theta2
                                * right_weight[pair, boundary, 1, o2, h2]
                                * work2[h1, o1, o2, h2]
                            )

            normalizer = 0.0
            for h1 in range(n_founders):
                for o1 in range(n_founders):
                    for h2 in range(n_founders):
                        for o2 in range(n_founders):
                            current[h1, o1, h2, o2] *= emission[
                                boundary + 1, h1, h2
                            ]
                            normalizer += current[h1, o1, h2, o2]
            total += np.log(normalizer)
            current /= normalizer
        output[pair] = total
    return output
def score_conditional_child_tensor_exact(
    child_genotype_likelihoods: np.ndarray,
    founder_alleles: np.ndarray,
    selected_markers_per_bin: np.ndarray,
    posterior: CandidateSourcePosterior,
    *,
    first_sample: Optional[int] = None,
    second_sample: Optional[int] = None,
    mismatch_probability: float = 0.01,
    track_switch_probability: np.ndarray = 0.01,
    eta: np.ndarray = 1.0,
    external_initial_probability: Optional[np.ndarray] = None,
    external_transition_probability: Optional[np.ndarray] = None,
) -> ExactTensorChildLikelihood:
    """Exact deterministic M0/M1/M2 score using strong-lumped source chains.

    Only sites at which every founder is called enter this exact-v1 score;
    founder-missing sites are neutral here and must use the unchanged linked
    hard/founder fallback at integration.  The candidate posterior normalizer
    is absent: only normalized physical
    initial probabilities and conditional transition factors are consumed.
    Candidate-unavailable states reduce before tensor allocation, so M1=M0,
    one-unavailable M2=M1, and two-unavailable M2=M0 exactly.
    """
    if (
        posterior.lumped_initial_log_probability is None
        or posterior.lumped_next_log_weight is None
        or posterior.lumped_transition_probability is None
    ):
        raise ValueError(
            "posterior must be inferred with return_lumped_posterior=True"
        )
    child = np.asarray(child_genotype_likelihoods, dtype=np.float64)
    if child.ndim != 3 or child.shape[2] != 3:
        raise ValueError("child genotype likelihoods must have shape (bins, slots, 3)")
    if np.any(~np.isfinite(child)) or np.any(child < 0.0):
        raise ValueError("child genotype likelihoods must be finite and nonnegative")
    totals = np.sum(child, axis=2, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("every child genotype-likelihood vector must contain mass")
    child = child / totals
    founders = np.asarray(founder_alleles, dtype=np.int8)
    if founders.ndim != 3 or founders.shape[1:] != child.shape[:2]:
        raise ValueError("founder alleles must have shape (founders, bins, slots)")
    marker_counts = _marker_counts(
        selected_markers_per_bin, child.shape[0], child.shape[1]
    )
    if not np.isfinite(mismatch_probability) or not 0.0 <= mismatch_probability < 0.5:
        raise ValueError("mismatch_probability must lie in [0, 0.5)")
    exponent = _reference_eta(eta, child.shape[0])
    track_switch = _reference_track_switch(
        track_switch_probability, child.shape[0]
    )
    founder_frequency = _reference_founder_frequency(founders)
    n_founders = founders.shape[0]
    excluded_marker_count = int(
        np.sum(
            (np.arange(child.shape[1])[None, :] < marker_counts[:, None])
            & ~posterior.lumped_site_available
        )
    )

    available_samples = []
    for sample in (first_sample, second_sample):
        if sample is not None and bool(posterior.lumped_available[sample]):
            available_samples.append(int(sample))
    if len(available_samples) == 2 and available_samples[0] == available_samples[1]:
        raise ValueError("M2 requires two distinct candidate samples")

    if external_initial_probability is None:
        external_initial = np.full(
            n_founders, 1.0 / n_founders, dtype=np.float64
        )
    else:
        external_initial = np.asarray(
            external_initial_probability, dtype=np.float64
        )
        if (
            external_initial.shape != (n_founders,)
            or np.any(~np.isfinite(external_initial))
            or np.any(external_initial < 0.0)
            or np.sum(external_initial) <= 0.0
        ):
            raise ValueError("external initial probability must have shape (founders,)")
        external_initial = external_initial / np.sum(external_initial)
    if external_transition_probability is None:
        external_transition = posterior.lumped_transition_probability.astype(
            np.float64
        )
    else:
        external_transition = np.asarray(
            external_transition_probability, dtype=np.float64
        )
        expected = (child.shape[0] - 1, n_founders, n_founders)
        if external_transition.shape != expected:
            raise ValueError(
                "external transition probability must have shape "
                "(bins - 1, founders, founders)"
            )
        if np.any(~np.isfinite(external_transition)) or np.any(
            external_transition < 0.0
        ):
            raise ValueError("external transitions must be finite and nonnegative")
        row_sum = np.sum(external_transition, axis=2, keepdims=True)
        if np.any(row_sum <= 0.0):
            raise ValueError("external transition rows must contain mass")
        external_transition = external_transition / row_sum

    if len(available_samples) == 0:
        mode = "m0"
        forward = external_initial[:, None] * external_initial[None, :]
    elif len(available_samples) == 1:
        mode = "m1"
        sample = available_samples[0]
        initial = np.exp(
            posterior.lumped_initial_log_probability[sample].astype(np.float64)
        )
        initial /= np.sum(initial)
        initial = 0.5 * (initial + initial.T)
        forward = initial[:, :, None] * external_initial[None, None, :]
    else:
        mode = "m2"
        first, second = available_samples
        first_initial = np.exp(
            posterior.lumped_initial_log_probability[first].astype(np.float64)
        )
        second_initial = np.exp(
            posterior.lumped_initial_log_probability[second].astype(np.float64)
        )
        first_initial /= np.sum(first_initial)
        second_initial /= np.sum(second_initial)
        first_initial = 0.5 * (first_initial + first_initial.T)
        second_initial = 0.5 * (second_initial + second_initial.T)
        forward = (
            first_initial[:, :, None, None]
            * second_initial[None, None, :, :]
        )

    total_log_likelihood = 0.0
    peak_forward_bytes = int(forward.nbytes)
    for block in range(child.shape[0]):
        if block > 0:
            theta = float(track_switch[block - 1])
            if mode == "m0":
                forward = _external_transition_update(
                    forward, external_transition[block - 1], 0
                )
                forward = _external_transition_update(
                    forward, external_transition[block - 1], 1
                )
            elif mode == "m1":
                forward = _lumped_candidate_update(
                    forward,
                    posterior,
                    sample,
                    block - 1,
                    0,
                    1,
                    theta,
                )
                forward = _external_transition_update(
                    forward, external_transition[block - 1], 2
                )
            else:
                forward = _lumped_candidate_update(
                    forward,
                    posterior,
                    first,
                    block - 1,
                    0,
                    1,
                    theta,
                )
                forward = _lumped_candidate_update(
                    forward,
                    posterior,
                    second,
                    block - 1,
                    2,
                    3,
                    theta,
                )
            transition_mass = float(np.sum(forward))
            if transition_mass <= 0.0 or not np.isfinite(transition_mass):
                return ExactTensorChildLikelihood(
                    -np.inf, mode, int(forward.size), peak_forward_bytes,
                    excluded_marker_count,
                )
            forward /= transition_mass
        emission = _founder_pair_bin_emission(
            child,
            founders,
            founder_frequency,
            int(marker_counts[block]),
            block,
            float(mismatch_probability),
            float(exponent[block]),
            posterior.lumped_site_available[block],
        )
        if mode == "m0":
            forward *= emission
        elif mode == "m1":
            forward *= emission[:, None, :]
        else:
            forward *= emission[:, None, :, None]
        forward, increment = _normalise_tensor_forward(forward)
        if not np.isfinite(increment):
            return ExactTensorChildLikelihood(
                -np.inf, mode, int(forward.size), peak_forward_bytes,
                excluded_marker_count,
            )
        total_log_likelihood += increment
        peak_forward_bytes = max(peak_forward_bytes, int(forward.nbytes))
    return ExactTensorChildLikelihood(
        log_likelihood=float(total_log_likelihood),
        mode=mode,
        hidden_state_count=int(forward.size),
        peak_forward_bytes=peak_forward_bytes,
        excluded_marker_count=excluded_marker_count,
    )



@njit(cache=True, parallel=True, fastmath=False)
def _m1_compound_forward_kernel(
    initial,
    right_weight,
    denominator,
    diagonal,
    off_diagonal,
    track_switch,
    emission,
    external_initial,
    external_transition,
    available,
):
    """Compiled M1 scores for every candidate for one child."""
    n_candidates = initial.shape[0]
    n_founders = initial.shape[1]
    n_bins = emission.shape[0]
    output = np.empty(n_candidates, dtype=np.float64)
    for parent in prange(n_candidates):
        if not available[parent]:
            output[parent] = np.nan
            continue
        current = np.empty((n_founders, n_founders, n_founders))
        work1 = np.empty_like(current)
        work2 = np.empty_like(current)
        total = 0.0
        normalizer = 0.0
        for h in range(n_founders):
            for o in range(n_founders):
                for external in range(n_founders):
                    current[h, o, external] = (
                        initial[parent, h, o]
                        * external_initial[external]
                        * emission[0, h, external]
                    )
                    normalizer += current[h, o, external]
        total += np.log(normalizer)
        current /= normalizer
        for boundary in range(n_bins - 1):
            coefficient = diagonal[boundary] - off_diagonal[boundary]
            theta = track_switch[parent, boundary]
            for h in range(n_founders):
                for o in range(n_founders):
                    for external in range(n_founders):
                        d = denominator[parent, boundary, h, o]
                        current[h, o, external] = (
                            current[h, o, external] / d if d > 0.0 else 0.0
                        )
            for o in range(n_founders):
                for external in range(n_founders):
                    axis_sum = 0.0
                    for h in range(n_founders):
                        axis_sum += current[h, o, external]
                    for h in range(n_founders):
                        work1[h, o, external] = (
                            coefficient * current[h, o, external]
                            + off_diagonal[boundary] * axis_sum
                        )
            for h in range(n_founders):
                for external in range(n_founders):
                    axis_sum = 0.0
                    for o in range(n_founders):
                        axis_sum += work1[h, o, external]
                    for o in range(n_founders):
                        work2[h, o, external] = (
                            coefficient * work1[h, o, external]
                            + off_diagonal[boundary] * axis_sum
                        )
            for h in range(n_founders):
                for o in range(n_founders):
                    for external in range(n_founders):
                        current[h, o, external] = (
                            (1.0 - theta)
                            * right_weight[parent, boundary, h, o]
                            * work2[h, o, external]
                            + theta
                            * right_weight[parent, boundary, o, h]
                            * work2[o, h, external]
                        )
            for h in range(n_founders):
                for o in range(n_founders):
                    for destination in range(n_founders):
                        value = 0.0
                        for source in range(n_founders):
                            value += (
                                current[h, o, source]
                                * external_transition[boundary, source, destination]
                            )
                        work1[h, o, destination] = value
            transition_mass = 0.0
            for h in range(n_founders):
                for o in range(n_founders):
                    for external in range(n_founders):
                        current[h, o, external] = work1[h, o, external]
                        transition_mass += current[h, o, external]
            if transition_mass <= 0.0:
                output[parent] = -np.inf
                break
            current /= transition_mass
            normalizer = 0.0
            for h in range(n_founders):
                for o in range(n_founders):
                    for external in range(n_founders):
                        current[h, o, external] *= emission[
                            boundary + 1, h, external
                        ]
                        normalizer += current[h, o, external]
            if normalizer <= 0.0:
                output[parent] = -np.inf
                break
            total += np.log(normalizer)
            current /= normalizer
        else:
            output[parent] = total
    return output


@dataclass(frozen=True)
class CandidateSourceBatchScores:
    """Production-neutral batch M0/M1/M2 scores and B4 diagnostics."""

    zero_observed: np.ndarray
    one_observed: np.ndarray
    two_observed: np.ndarray
    one_parent_identity_information: np.ndarray
    two_parent_edge_information: np.ndarray
    candidate_source_available: np.ndarray
    candidate_source_informative_marker_count: np.ndarray
    child_complete_informative_marker_count: np.ndarray
    complete_founder_marker_count: int
    excluded_founder_marker_count: int
    excluded_founder_marker_count_per_child: np.ndarray
    candidate_initial_max_probability: np.ndarray
    candidate_initial_point_mass: np.ndarray
    peak_streamed_tensor_bytes: int
    factor_preparation_seconds: float
    emission_preparation_seconds: float
    m0_m1_scoring_seconds: float
    m2_scoring_seconds: float


def _m0_from_precomputed_emission(
    emission: np.ndarray,
    external_initial: np.ndarray,
    external_transition: np.ndarray,
) -> float:
    forward = external_initial[:, None] * external_initial[None, :]
    total = 0.0
    for block in range(emission.shape[0]):
        if block > 0:
            transition = external_transition[block - 1]
            forward = transition.T @ forward @ transition
            mass = float(np.sum(forward))
            if mass <= 0.0:
                return -np.inf
            forward /= mass
        forward *= emission[block]
        normalizer = float(np.sum(forward))
        if normalizer <= 0.0:
            return -np.inf
        total += np.log(normalizer)
        forward /= normalizer
    return total


def score_candidate_source_batch_exact(
    posterior: CandidateSourcePosterior,
    child_genotype_likelihoods: np.ndarray,
    founder_alleles: np.ndarray,
    selected_markers_per_bin: np.ndarray,
    child_information_exponent: np.ndarray,
    candidate_track_switch_probability: np.ndarray,
    external_initial_probability: np.ndarray,
    external_transition_probability: np.ndarray,
    trios: np.ndarray,
    *,
    mismatch_probability: float = 0.01,
    uniform_tolerance: float = 1e-12,
) -> CandidateSourceBatchScores:
    """Batch exact-v1 conditional scores with child and candidate axes separate.

    Complete-founder sites alone enter B4.  Excluded sites are neutral and are
    counted explicitly for the unchanged fallback seam.  M2 tensors are packed
    and released one child at a time rather than retained for all trio rows.
    """
    if (
        posterior.lumped_initial_log_probability is None
        or posterior.lumped_next_log_weight is None
        or posterior.lumped_transition_probability is None
    ):
        raise ValueError("posterior needs return_lumped_posterior=True")
    child = np.asarray(child_genotype_likelihoods, dtype=np.float64)
    if child.ndim != 4 or child.shape[3] != 3:
        raise ValueError("child GL must have shape (children, bins, slots, 3)")
    if np.any(~np.isfinite(child)) or np.any(child < 0.0):
        raise ValueError("child GL must be finite and nonnegative")
    child_total = np.sum(child, axis=3, keepdims=True)
    if np.any(child_total <= 0.0):
        raise ValueError("every child GL vector must contain mass")
    child = child / child_total
    founders = np.asarray(founder_alleles, dtype=np.int8)
    n_children, n_bins, n_slots, _ = child.shape
    n_candidates = posterior.lumped_initial_log_probability.shape[0]
    n_founders = founders.shape[0]
    if founders.shape[1:] != (n_bins, n_slots):
        raise ValueError("founder and child bin/slot shapes must match")
    marker_counts = _marker_counts(
        selected_markers_per_bin, n_bins, n_slots
    )
    exponent = np.asarray(child_information_exponent, dtype=np.float64)
    if exponent.shape != (n_children, n_bins) or np.any(~np.isfinite(exponent)) or np.any(exponent < 0.0):
        raise ValueError("child information exponent must be nonnegative (children, bins)")
    candidate_switch = np.asarray(
        candidate_track_switch_probability, dtype=np.float64
    )
    if candidate_switch.shape != (n_candidates, n_bins - 1) or np.any(
        ~np.isfinite(candidate_switch)
    ) or np.any((candidate_switch < 0.0) | (candidate_switch > 0.5)):
        raise ValueError("candidate track switch must have shape (candidates, bins - 1)")
    external_initial = np.asarray(external_initial_probability, dtype=np.float64)
    if external_initial.shape != (n_children, n_founders):
        raise ValueError("external initial must have shape (children, founders)")
    external_initial = external_initial / np.sum(
        external_initial, axis=1, keepdims=True
    )
    external_transition = np.asarray(
        external_transition_probability, dtype=np.float64
    )
    if external_transition.shape != (
        n_children, n_bins - 1, n_founders, n_founders
    ):
        raise ValueError(
            "external transitions need shape (children, bins - 1, founders, founders)"
        )
    external_transition = external_transition / np.sum(
        external_transition, axis=3, keepdims=True
    )
    trio_array = np.asarray(trios, dtype=np.int64)
    if trio_array.ndim != 2 or trio_array.shape[1] != 3:
        raise ValueError("trios must have shape (rows, 3)")
    if len(trio_array) and (
        np.any(trio_array[:, 0] < 0)
        or np.any(trio_array[:, 0] >= n_children)
        or np.any(trio_array[:, 1:] < 0)
        or np.any(trio_array[:, 1:] >= n_candidates)
        or np.any(trio_array[:, 1] == trio_array[:, 2])
    ):
        raise ValueError("trio row contains an invalid child or parent")
    if not np.isfinite(mismatch_probability) or not 0.0 <= mismatch_probability < 0.5:
        raise ValueError("mismatch_probability must lie in [0, 0.5)")

    factor_start = time.perf_counter()
    transition = posterior.lumped_transition_probability.astype(np.float64)
    initial = np.exp(
        posterior.lumped_initial_log_probability.astype(np.float64)
    )
    initial_sum = np.sum(initial, axis=(1, 2), keepdims=True)
    initial = np.divide(
        initial,
        initial_sum,
        out=np.zeros_like(initial),
        where=initial_sum > 0.0,
    )
    initial = 0.5 * (initial + np.swapaxes(initial, 1, 2))
    right = np.exp(posterior.lumped_next_log_weight.astype(np.float64))
    denominator = np.empty_like(right)
    for candidate in range(n_candidates):
        if not posterior.lumped_available[candidate]:
            denominator[candidate] = 0.0
            continue
        for boundary in range(n_bins - 1):
            denominator[candidate, boundary] = (
                transition[boundary]
                @ right[candidate, boundary]
                @ transition[boundary].T
            )
    diagonal = transition[:, 0, 0]
    off_diagonal = (
        transition[:, 0, 1]
        if n_founders > 1
        else np.zeros(n_bins - 1)
    )
    factor_seconds = time.perf_counter() - factor_start

    emission_start = time.perf_counter()
    frequency = _reference_founder_frequency(founders)
    emission = np.empty(
        (n_children, n_bins, n_founders, n_founders), dtype=np.float64
    )
    for child_index in range(n_children):
        for block in range(n_bins):
            emission[child_index, block] = _founder_pair_bin_emission(
                child[child_index],
                founders,
                frequency,
                int(marker_counts[block]),
                block,
                float(mismatch_probability),
                float(exponent[child_index, block]),
                posterior.lumped_site_available[block],
            )
    emission_seconds = time.perf_counter() - emission_start

    m0_m1_start = time.perf_counter()
    zero = np.empty(n_children, dtype=np.float64)
    one = np.empty((n_children, n_candidates), dtype=np.float64)
    peak_streamed = 0
    for child_index in range(n_children):
        zero[child_index] = _m0_from_precomputed_emission(
            emission[child_index],
            external_initial[child_index],
            external_transition[child_index],
        )
        one_child = _m1_compound_forward_kernel(
            initial,
            right,
            denominator,
            diagonal,
            off_diagonal,
            candidate_switch,
            emission[child_index],
            external_initial[child_index],
            external_transition[child_index],
            posterior.lumped_available,
        )
        one[child_index] = np.where(
            posterior.lumped_available, one_child, zero[child_index]
        )
        peak_streamed = max(
            peak_streamed,
            int(3 * n_candidates * n_founders**3 * 8),
        )
    m0_m1_seconds = time.perf_counter() - m0_m1_start

    m2_start = time.perf_counter()
    two = np.empty(len(trio_array), dtype=np.float64)
    for child_index in range(n_children):
        rows = np.flatnonzero(trio_array[:, 0] == child_index)
        if len(rows) == 0:
            continue
        first_parent = np.minimum(
            trio_array[rows, 1], trio_array[rows, 2]
        )
        second_parent = np.maximum(
            trio_array[rows, 1], trio_array[rows, 2]
        )
        first_available = posterior.lumped_available[first_parent]
        second_available = posterior.lumped_available[second_parent]
        neither = ~first_available & ~second_available
        first_only = first_available & ~second_available
        second_only = ~first_available & second_available
        both = first_available & second_available
        two[rows[neither]] = zero[child_index]
        two[rows[first_only]] = one[child_index, first_parent[first_only]]
        two[rows[second_only]] = one[child_index, second_parent[second_only]]
        if np.any(both):
            active_rows = rows[both]
            first = first_parent[both]
            second = second_parent[both]
            pair_count = len(active_rows)
            pair_initial = (
                initial[first, :, :, None, None]
                * initial[second, None, None, :, :]
            )
            pair_right = np.empty(
                (pair_count, n_bins - 1, 2, n_founders, n_founders),
                dtype=np.float64,
            )
            pair_denominator = np.empty_like(pair_right)
            pair_right[:, :, 0] = right[first]
            pair_right[:, :, 1] = right[second]
            pair_denominator[:, :, 0] = denominator[first]
            pair_denominator[:, :, 1] = denominator[second]
            pair_switch = np.empty((pair_count, n_bins - 1, 2))
            pair_switch[:, :, 0] = candidate_switch[first]
            pair_switch[:, :, 1] = candidate_switch[second]
            two[active_rows] = _m2_compound_forward_kernel(
                pair_initial,
                pair_right,
                pair_denominator,
                diagonal,
                off_diagonal,
                pair_switch,
                emission[child_index],
            )
            peak_streamed = max(
                peak_streamed,
                int(
                    pair_initial.nbytes
                    + pair_right.nbytes
                    + pair_denominator.nbytes
                    + pair_switch.nbytes
                ),
            )
    m2_seconds = time.perf_counter() - m2_start

    real_site = np.arange(n_slots)[None, :] < marker_counts[:, None]
    child_informative = (
        np.ptp(child, axis=3) > uniform_tolerance
    ) & posterior.lumped_site_available[None, :, :] & real_site[None, :, :]
    child_count = np.sum(child_informative, axis=(1, 2)).astype(np.int64)
    child_information = np.sum(
        exponent * np.sum(child_informative, axis=2), axis=1
    )
    identity = child_information[:, None] * posterior.lumped_available[None, :]
    edge_information = np.empty((len(trio_array), 2), dtype=np.float64)
    if len(trio_array):
        edge_information[:, 0] = identity[
            trio_array[:, 0], trio_array[:, 1]
        ]
        edge_information[:, 1] = identity[
            trio_array[:, 0], trio_array[:, 2]
        ]
    complete_count = int(np.sum(posterior.lumped_site_available & real_site))
    excluded_count = int(np.sum(real_site) - complete_count)
    initial_max = np.zeros(n_candidates, dtype=np.float64)
    for candidate in range(n_candidates):
        for first in range(n_founders):
            initial_max[candidate] = max(
                initial_max[candidate], initial[candidate, first, first]
            )
            for second in range(first + 1, n_founders):
                initial_max[candidate] = max(
                    initial_max[candidate],
                    initial[candidate, first, second]
                    + initial[candidate, second, first],
                )
    return CandidateSourceBatchScores(
        zero_observed=zero,
        one_observed=one,
        two_observed=two,
        one_parent_identity_information=identity,
        two_parent_edge_information=edge_information,
        candidate_source_available=posterior.lumped_available.copy(),
        candidate_source_informative_marker_count=(
            posterior.lumped_informative_site_count.copy()
        ),
        child_complete_informative_marker_count=child_count,
        complete_founder_marker_count=complete_count,
        excluded_founder_marker_count=excluded_count,
        excluded_founder_marker_count_per_child=np.full(
            n_children, excluded_count, dtype=np.int64
        ),
        candidate_initial_max_probability=initial_max,
        candidate_initial_point_mass=initial_max == 1.0,
        peak_streamed_tensor_bytes=peak_streamed,
        factor_preparation_seconds=factor_seconds,
        emission_preparation_seconds=emission_seconds,
        m0_m1_scoring_seconds=m0_m1_seconds,
        m2_scoring_seconds=m2_seconds,
    )


@dataclass(frozen=True)
class MatchedNullCandidateSourceBatchScores:
    """B5a scores from candidates and two matched synthetic null parents."""

    zero_observed: np.ndarray
    one_observed: np.ndarray
    two_observed: np.ndarray
    one_parent_identity_information: np.ndarray
    two_parent_edge_information: np.ndarray
    candidate_source_available: np.ndarray
    candidate_source_informative_marker_count: np.ndarray
    child_complete_informative_marker_count: np.ndarray
    complete_founder_marker_count: int
    excluded_founder_marker_count: int
    excluded_founder_marker_count_per_child: np.ndarray
    candidate_initial_max_probability: np.ndarray
    candidate_initial_point_mass: np.ndarray
    null_homolog_root_probability: np.ndarray
    null_diplotype_initial_max_probability: float
    source_path_switch_probability: np.ndarray
    transmission_switch_probability: np.ndarray
    null_parent_count: int
    matched_pair_evaluation_count: int
    peak_streamed_tensor_bytes: int
    factor_preparation_seconds: float
    emission_preparation_seconds: float
    m0_scoring_seconds: float
    m1_scoring_seconds: float
    m2_scoring_seconds: float


def _matched_compound_transition_terms(
    transition: np.ndarray,
    n_bins: int,
    n_founders: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(transition, dtype=np.float64)
    if value.shape != (max(n_bins - 1, 0), n_founders, n_founders):
        raise ValueError(
            "lumped transition must have shape (bins - 1, founders, founders)"
        )
    if np.any(~np.isfinite(value)) or np.any(value < 0.0):
        raise ValueError("lumped transition must be finite and nonnegative")
    if len(value) and not np.allclose(
        np.sum(value, axis=2), 1.0, rtol=0.0, atol=2e-7
    ):
        raise ValueError("lumped transition rows must sum to one")
    diagonal = np.empty(len(value), dtype=np.float64)
    off_diagonal = np.empty(len(value), dtype=np.float64)
    source_switch = np.empty(len(value), dtype=np.float64)
    for boundary, matrix in enumerate(value):
        diagonal[boundary] = matrix[0, 0]
        off_diagonal[boundary] = (
            matrix[0, 1] if n_founders > 1 else 0.0
        )
        expected = np.full_like(matrix, off_diagonal[boundary])
        np.fill_diagonal(expected, diagonal[boundary])
        if not np.allclose(matrix, expected, rtol=0.0, atol=2e-7):
            raise ValueError(
                "matched-null v2 requires a shared compound source transition"
            )
        source_switch[boundary] = (
            1.0 - diagonal[boundary] if n_founders > 1 else 0.0
        )
    return diagonal, off_diagonal, source_switch


def _matched_pair_score(
    first: np.ndarray,
    second: np.ndarray,
    initial: np.ndarray,
    right: np.ndarray,
    denominator: np.ndarray,
    diagonal: np.ndarray,
    off_diagonal: np.ndarray,
    transmission_switch: np.ndarray,
    emission: np.ndarray,
) -> tuple[np.ndarray, int]:
    pair_initial = (
        initial[first, :, :, None, None]
        * initial[second, None, None, :, :]
    )
    n_pairs = len(first)
    n_boundaries = right.shape[1]
    n_founders = initial.shape[1]
    pair_right = np.empty(
        (n_pairs, n_boundaries, 2, n_founders, n_founders),
        dtype=np.float64,
    )
    pair_denominator = np.empty_like(pair_right)
    pair_right[:, :, 0] = right[first]
    pair_right[:, :, 1] = right[second]
    pair_denominator[:, :, 0] = denominator[first]
    pair_denominator[:, :, 1] = denominator[second]
    scores = _m2_compound_forward_kernel(
        pair_initial,
        pair_right,
        pair_denominator,
        diagonal,
        off_diagonal,
        transmission_switch,
        emission,
    )
    peak_bytes = int(
        pair_initial.nbytes
        + pair_right.nbytes
        + pair_denominator.nbytes
        + transmission_switch.nbytes
    )
    return scores, peak_bytes


def score_candidate_source_batch_matched_null_exact(
    posterior: CandidateSourcePosterior,
    child_genotype_likelihoods: np.ndarray,
    founder_alleles: np.ndarray,
    selected_markers_per_bin: np.ndarray,
    child_information_exponent: np.ndarray,
    transmission_switch_probability: np.ndarray,
    trios: np.ndarray,
    *,
    mismatch_probability: float = 0.01,
    uniform_tolerance: float = 1e-12,
) -> MatchedNullCandidateSourceBatchScores:
    """Score B5a with two deterministic null-parent source factor rows.

    Candidate and null parents share one compound physical source transition,
    an ordered-independent uniform diploid root, and one biological
    transmission-selector law. The two identical null rows denote independent
    latent parent draws because their factor laws enter the pair forward as a
    product. M0 is null-null, M1 is candidate-null, and M2 is
    candidate-candidate. An unavailable candidate is replaced by the null
    factor, giving exact nested reductions without an availability threshold
    discontinuity.
    """
    if posterior.lumped_root_prior_mode != "ordered_independent_uniform":
        raise ValueError(
            "matched-null v2 requires lumped_root_prior_mode="
            "'ordered_independent_uniform'"
        )
    if (
        posterior.lumped_initial_log_probability is None
        or posterior.lumped_next_log_weight is None
        or posterior.lumped_transition_probability is None
    ):
        raise ValueError("posterior needs return_lumped_posterior=True")

    child = np.asarray(child_genotype_likelihoods, dtype=np.float64)
    if child.ndim != 4 or child.shape[3] != 3:
        raise ValueError("child GL must have shape (children, bins, slots, 3)")
    if np.any(~np.isfinite(child)) or np.any(child < 0.0):
        raise ValueError("child GL must be finite and nonnegative")
    child_total = np.sum(child, axis=3, keepdims=True)
    if np.any(child_total <= 0.0):
        raise ValueError("every child GL vector must contain mass")
    child = child / child_total

    founders = np.asarray(founder_alleles, dtype=np.int8)
    n_children, n_bins, n_slots, _ = child.shape
    n_candidates = posterior.lumped_initial_log_probability.shape[0]
    n_founders = founders.shape[0]
    if founders.ndim != 3 or founders.shape[1:] != (n_bins, n_slots):
        raise ValueError("founder alleles must have shape (founders, bins, slots)")
    marker_counts = _marker_counts(
        selected_markers_per_bin, n_bins, n_slots
    )
    exponent = np.asarray(child_information_exponent, dtype=np.float64)
    if (
        exponent.shape != (n_children, n_bins)
        or np.any(~np.isfinite(exponent))
        or np.any(exponent < 0.0)
    ):
        raise ValueError(
            "child information exponent must be nonnegative (children, bins)"
        )
    transmission_switch = _reference_track_switch(
        transmission_switch_probability, n_bins
    )
    trio_array = np.asarray(trios, dtype=np.int64)
    if trio_array.ndim != 2 or trio_array.shape[1] != 3:
        raise ValueError("trios must have shape (rows, 3)")
    if len(trio_array) and (
        np.any(trio_array[:, 0] < 0)
        or np.any(trio_array[:, 0] >= n_children)
        or np.any(trio_array[:, 1:] < 0)
        or np.any(trio_array[:, 1:] >= n_candidates)
        or np.any(trio_array[:, 1] == trio_array[:, 2])
    ):
        raise ValueError("trio row contains an invalid child or parent")
    if (
        not np.isfinite(mismatch_probability)
        or not 0.0 <= mismatch_probability < 0.5
    ):
        raise ValueError("mismatch_probability must lie in [0, 0.5)")
    if not np.isfinite(uniform_tolerance) or uniform_tolerance < 0.0:
        raise ValueError("uniform_tolerance must be finite and nonnegative")

    factor_start = time.perf_counter()
    transition = np.asarray(
        posterior.lumped_transition_probability, dtype=np.float64
    )
    diagonal, off_diagonal, source_switch = (
        _matched_compound_transition_terms(
            transition, n_bins, n_founders
        )
    )
    available = np.asarray(posterior.lumped_available, dtype=np.bool_)
    if available.shape != (n_candidates,):
        raise ValueError("lumped candidate availability has the wrong shape")
    candidate_initial = np.exp(
        posterior.lumped_initial_log_probability.astype(np.float64)
    )
    candidate_right = np.exp(
        posterior.lumped_next_log_weight.astype(np.float64)
    )
    if candidate_right.shape != (
        n_candidates, max(n_bins - 1, 0), n_founders, n_founders
    ):
        raise ValueError("lumped next weights have incompatible dimensions")
    null_initial = _ordered_independent_uniform_root_prior(n_founders)
    null_right = np.ones(
        (max(n_bins - 1, 0), n_founders, n_founders),
        dtype=np.float64,
    )
    source_initial = np.empty(
        (n_candidates + 2, n_founders, n_founders), dtype=np.float64
    )
    source_right = np.empty(
        (
            n_candidates + 2,
            max(n_bins - 1, 0),
            n_founders,
            n_founders,
        ),
        dtype=np.float64,
    )
    for candidate in range(n_candidates):
        if available[candidate]:
            mass = float(np.sum(candidate_initial[candidate]))
            if not np.isfinite(mass) or mass <= 0.0:
                raise ValueError("available candidate initial factor has no mass")
            value = candidate_initial[candidate] / mass
            source_initial[candidate] = 0.5 * (value + value.T)
            if np.any(~np.isfinite(candidate_right[candidate])):
                raise ValueError("available candidate next weights must be finite")
            source_right[candidate] = candidate_right[candidate]
        else:
            source_initial[candidate] = null_initial
            source_right[candidate] = null_right
    source_initial[n_candidates:] = null_initial
    source_right[n_candidates:] = null_right

    source_denominator = np.empty_like(source_right)
    for source in range(n_candidates + 2):
        for boundary in range(n_bins - 1):
            source_denominator[source, boundary] = (
                transition[boundary]
                @ source_right[source, boundary]
                @ transition[boundary].T
            )
    factor_seconds = time.perf_counter() - factor_start

    emission_start = time.perf_counter()
    frequency = _reference_founder_frequency(founders)
    emission = np.empty(
        (n_children, n_bins, n_founders, n_founders), dtype=np.float64
    )
    for child_index in range(n_children):
        for block in range(n_bins):
            emission[child_index, block] = _founder_pair_bin_emission(
                child[child_index],
                founders,
                frequency,
                int(marker_counts[block]),
                block,
                float(mismatch_probability),
                float(exponent[child_index, block]),
                posterior.lumped_site_available[block],
            )
    emission_seconds = time.perf_counter() - emission_start

    null0 = n_candidates
    null1 = n_candidates + 1
    zero = np.empty(n_children, dtype=np.float64)
    one = np.empty((n_children, n_candidates), dtype=np.float64)
    two = np.empty(len(trio_array), dtype=np.float64)
    peak_streamed = 0

    m0_start = time.perf_counter()
    for child_index in range(n_children):
        values, peak = _matched_pair_score(
            np.asarray([null0]),
            np.asarray([null1]),
            source_initial,
            source_right,
            source_denominator,
            diagonal,
            off_diagonal,
            transmission_switch,
            emission[child_index],
        )
        zero[child_index] = values[0]
        peak_streamed = max(peak_streamed, peak)
    m0_seconds = time.perf_counter() - m0_start

    m1_start = time.perf_counter()
    candidate_indices = np.arange(n_candidates, dtype=np.int64)
    null_indices = np.full(n_candidates, null0, dtype=np.int64)
    for child_index in range(n_children):
        values, peak = _matched_pair_score(
            candidate_indices,
            null_indices,
            source_initial,
            source_right,
            source_denominator,
            diagonal,
            off_diagonal,
            transmission_switch,
            emission[child_index],
        )
        one[child_index] = values
        peak_streamed = max(peak_streamed, peak)
    m1_seconds = time.perf_counter() - m1_start

    m2_start = time.perf_counter()
    for child_index in range(n_children):
        rows = np.flatnonzero(trio_array[:, 0] == child_index)
        if not len(rows):
            continue
        values, peak = _matched_pair_score(
            trio_array[rows, 1],
            trio_array[rows, 2],
            source_initial,
            source_right,
            source_denominator,
            diagonal,
            off_diagonal,
            transmission_switch,
            emission[child_index],
        )
        two[rows] = values
        peak_streamed = max(peak_streamed, peak)
    m2_seconds = time.perf_counter() - m2_start

    real_site = np.arange(n_slots)[None, :] < marker_counts[:, None]
    child_informative = (
        np.ptp(child, axis=3) > uniform_tolerance
    ) & posterior.lumped_site_available[None, :, :] & real_site[None, :, :]
    child_count = np.sum(child_informative, axis=(1, 2)).astype(np.int64)
    child_information = np.sum(
        exponent * np.sum(child_informative, axis=2), axis=1
    )
    identity = child_information[:, None] * available[None, :]
    edge_information = np.empty((len(trio_array), 2), dtype=np.float64)
    if len(trio_array):
        edge_information[:, 0] = identity[
            trio_array[:, 0], trio_array[:, 1]
        ]
        edge_information[:, 1] = identity[
            trio_array[:, 0], trio_array[:, 2]
        ]
    complete_count = int(np.sum(posterior.lumped_site_available & real_site))
    excluded_count = int(np.sum(real_site) - complete_count)
    initial_max = np.zeros(n_candidates, dtype=np.float64)
    for candidate in range(n_candidates):
        value = source_initial[candidate]
        for first in range(n_founders):
            initial_max[candidate] = max(
                initial_max[candidate], value[first, first]
            )
            for second in range(first + 1, n_founders):
                initial_max[candidate] = max(
                    initial_max[candidate],
                    value[first, second] + value[second, first],
                )
    null_max = (
        1.0 if n_founders == 1 else 2.0 / float(n_founders * n_founders)
    )
    pair_count = int(
        n_children * (1 + n_candidates) + len(trio_array)
    )
    return MatchedNullCandidateSourceBatchScores(
        zero_observed=zero,
        one_observed=one,
        two_observed=two,
        one_parent_identity_information=identity,
        two_parent_edge_information=edge_information,
        candidate_source_available=available.copy(),
        candidate_source_informative_marker_count=(
            posterior.lumped_informative_site_count.copy()
        ),
        child_complete_informative_marker_count=child_count,
        complete_founder_marker_count=complete_count,
        excluded_founder_marker_count=excluded_count,
        excluded_founder_marker_count_per_child=np.full(
            n_children, excluded_count, dtype=np.int64
        ),
        candidate_initial_max_probability=initial_max,
        candidate_initial_point_mass=initial_max == 1.0,
        null_homolog_root_probability=np.full(
            n_founders, 1.0 / float(n_founders), dtype=np.float64
        ),
        null_diplotype_initial_max_probability=null_max,
        source_path_switch_probability=source_switch,
        transmission_switch_probability=transmission_switch.copy(),
        null_parent_count=2,
        matched_pair_evaluation_count=pair_count,
        peak_streamed_tensor_bytes=peak_streamed,
        factor_preparation_seconds=factor_seconds,
        emission_preparation_seconds=emission_seconds,
        m0_scoring_seconds=m0_seconds,
        m1_scoring_seconds=m1_seconds,
        m2_scoring_seconds=m2_seconds,
    )


__all__ = [
    "CandidateSourcePosterior",
    "CandidateSourceTrajectoryDraws",
    "MonteCarloChildLikelihood",
    "ExactTensorChildLikelihood",
    "CandidateSourceBatchScores",
    "MatchedNullCandidateSourceBatchScores",
    "infer_candidate_source_posterior",
    "sample_candidate_source_trajectories",
    "reference_conditional_child_likelihood_mc",
    "score_conditional_child_tensor_exact",
    "score_candidate_source_batch_exact",
    "score_candidate_source_batch_matched_null_exact",
]
