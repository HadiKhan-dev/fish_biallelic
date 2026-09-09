"""assembly / observations for the canonical reconstruction pipeline."""
from __future__ import annotations


from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class FounderPanel:
    """Canonical founder alleles and explicit missingness for one block."""

    positions: np.ndarray
    keys: tuple[Any, ...]
    q: np.ndarray
    called: np.ndarray
    support: np.ndarray | None = None

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions)
        q = np.asarray(self.q, dtype=np.float64)
        called = np.asarray(self.called, dtype=np.bool_)
        if q.ndim != 2:
            raise ValueError("q must have shape (founders, sites)")
        if called.shape != q.shape:
            raise ValueError("called must have the same shape as q")
        if positions.shape != (q.shape[1],):
            raise ValueError("positions must match the site dimension")
        if len(self.keys) != q.shape[0]:
            raise ValueError("keys must match the founder dimension")
        if not np.all(np.isfinite(q)) or np.any((q < 0.0) | (q > 1.0)):
            raise ValueError("q must contain finite probabilities in [0, 1]")

        support = self.support
        if support is not None:
            support = np.asarray(support, dtype=np.float64)
            if support.shape != q.shape:
                raise ValueError("support must have the same shape as q")
            if not np.all(np.isfinite(support)) or np.any(support < 0.0):
                raise ValueError("support must be finite and non-negative")

        object.__setattr__(self, "positions", positions.copy())
        object.__setattr__(self, "keys", tuple(self.keys))
        object.__setattr__(self, "q", q.copy())
        object.__setattr__(self, "called", called.copy())
        object.__setattr__(self, "support", None if support is None else support.copy())


def founder_panel_from_block_result(block: Any) -> FounderPanel:
    """Build the canonical panel from a ``BlockResult``-like object.

    Stage-1 results expose discrete_haps; its non-negative cells are the
    authoritative release mask. Released values become hard q=0/1 and all
    other cells become q=0.5. Directional supporter counts are retained when
    present. A block without this explicit mask is not a valid Stage-2 input.
    """

    positions = np.asarray(block.positions)
    keys = tuple(sorted(block.haplotypes))
    n_founders = len(keys)
    n_sites = len(positions)

    discrete = getattr(block, "discrete_haps", None)
    if discrete is None:
        raise ValueError("Stage-2 founder blocks require discrete_haps")
    discrete = np.asarray(discrete)
    if discrete.shape != (n_founders, n_sites):
        raise ValueError("block.discrete_haps has the wrong shape")
    called = discrete >= 0
    if np.any(called & ~np.isin(discrete, (0, 1))):
        raise ValueError("released discrete founder alleles must be 0 or 1")
    q = np.full((n_founders, n_sites), 0.5, dtype=np.float64)
    q[called] = discrete[called]

    support = getattr(block, "n_directional_site_supporters", None)
    if support is not None:
        support = np.asarray(support, dtype=np.float64)
        if support.shape != q.shape:
            raise ValueError(
                "block.n_directional_site_supporters has the wrong shape"
            )
    return FounderPanel(positions, keys, q, called, support)


def founder_inference_panel_from_block_result(block: Any) -> FounderPanel:
    """Build the immutable pre-fill panel used for Stage-2 inference.

    Missing-aware postprocessing may publish filled alleles in
    ``discrete_haps`` while retaining its pre-cavity evidence snapshot in the
    stable ``missing_aware_inference_discrete_haps`` attribute.  Linkage and
    painting must consume that snapshot so an output fill cannot feed back
    into its own carrier-state evidence.  Blocks without a snapshot fall back
    to their authoritative ``discrete_haps`` exactly as
    :func:`founder_panel_from_block_result` does.
    """

    inference_discrete = getattr(
        block, "missing_aware_inference_discrete_haps", None
    )
    if inference_discrete is None:
        return founder_panel_from_block_result(block)

    positions = np.asarray(block.positions)
    keys = tuple(sorted(block.haplotypes))
    discrete = np.asarray(inference_discrete)
    expected_shape = (len(keys), len(positions))
    if discrete.shape != expected_shape:
        raise ValueError(
            "block.missing_aware_inference_discrete_haps has the wrong shape"
        )
    called = discrete >= 0
    if np.any(called & ~np.isin(discrete, (0, 1))):
        raise ValueError("released discrete founder alleles must be 0 or 1")
    q = np.full(expected_shape, 0.5, dtype=np.float64)
    q[called] = discrete[called]

    support = getattr(block, "n_directional_site_supporters", None)
    if support is not None:
        support = np.asarray(support, dtype=np.float64)
        if support.shape != expected_shape:
            raise ValueError(
                "block.n_directional_site_supporters has the wrong shape"
            )
    return FounderPanel(positions, keys, q, called, support)


def diploid_genotype_distributions(panel: FounderPanel) -> np.ndarray:
    """Return P(genotype dosage | founder pair), shape ``(K,K,L,3)``.

    Distinct founders use their mean-field Bernoulli marginals.  The diagonal
    is different: two homologs labelled with the same founder refer to one
    shared latent allele, not two independent draws.  Consequently (k, k) has
    only homozygous-reference and homozygous-alternate mass.
    """

    q_first = panel.q[:, None, :]
    q_second = panel.q[None, :, :]
    distribution = np.empty(
        (panel.q.shape[0], panel.q.shape[0], panel.q.shape[1], 3),
        dtype=np.float64,
    )
    distribution[..., 0] = (1.0 - q_first) * (1.0 - q_second)
    distribution[..., 1] = (
        q_first * (1.0 - q_second) + (1.0 - q_first) * q_second
    )
    distribution[..., 2] = q_first * q_second
    for founder in range(panel.q.shape[0]):
        distribution[founder, founder, :, 0] = 1.0 - panel.q[founder]
        distribution[founder, founder, :, 1] = 0.0
        distribution[founder, founder, :, 2] = panel.q[founder]
    return distribution


def _normalised_genotype_evidence(evidence: np.ndarray) -> np.ndarray:
    evidence = np.asarray(evidence, dtype=np.float64)
    if evidence.ndim != 3 or evidence.shape[2] != 3:
        raise ValueError("genotype evidence must have shape (samples, sites, 3)")
    if np.any(~np.isfinite(evidence)) or np.any(evidence < 0.0):
        raise ValueError("genotype evidence must be finite and non-negative")
    totals = np.sum(evidence, axis=2, keepdims=True)
    normalised = np.full(evidence.shape, 1.0 / 3.0, dtype=np.float64)
    np.divide(evidence, totals, out=normalised, where=totals > 0.0)
    return normalised


def site_emission_probabilities(
    genotype_evidence: np.ndarray,
    genotype_distributions: np.ndarray,
    *,
    uniform_mix: float = 0.01,
) -> np.ndarray:
    """Return robust per-site diplotype likelihoods, shape ``(N,K,K,L)``.

    The robust mixture is ``(1-uniform_mix) * likelihood + uniform_mix/3``.
    It is the existing one-hot Stage-2 observation model generalized to a
    genotype distribution.  Therefore fully called one-hot panels retain the
    established emission probabilities exactly.
    """

    if not 0.0 <= uniform_mix <= 1.0:
        raise ValueError("uniform_mix must lie in [0, 1]")
    evidence = _normalised_genotype_evidence(genotype_evidence)
    distribution = np.asarray(genotype_distributions, dtype=np.float64)
    if distribution.ndim != 4 or distribution.shape[-1] != 3:
        raise ValueError(
            "genotype_distributions must have shape (founders, founders, sites, 3)"
        )
    if distribution.shape[0] != distribution.shape[1]:
        raise ValueError("the two founder dimensions must be equal")
    if distribution.shape[2] != evidence.shape[1]:
        raise ValueError("genotype distributions and evidence disagree on sites")
    if np.any(~np.isfinite(distribution)) or np.any(distribution < 0.0):
        raise ValueError("genotype distributions must be finite and non-negative")
    if not np.allclose(np.sum(distribution, axis=3), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("each genotype distribution must sum to one")

    likelihood = np.einsum("nlg,ijlg->nijl", evidence, distribution, optimize=True)
    return (1.0 - uniform_mix) * likelihood + uniform_mix / 3.0


def site_log_emissions(
    genotype_evidence: np.ndarray,
    genotype_distributions: np.ndarray,
    *,
    uniform_mix: float = 0.01,
    log_floor: float = -2.0,
) -> np.ndarray:
    """Return robust log emissions relative to uninformative evidence.

    Clipping is applied on the ordinary likelihood scale exactly as in the
    current one-hot Stage-2 kernel, then ``log(1/3)`` is subtracted.  This
    additive state-independent centering makes every uniform sample/site row
    exactly zero while preserving all diplotype comparisons and Viterbi paths.
    """

    if log_floor > 0.0:
        raise ValueError("log_floor must be non-positive")
    probability = site_emission_probabilities(
        genotype_evidence, genotype_distributions, uniform_mix=uniform_mix
    )
    with np.errstate(divide="ignore"):
        log_emission = np.log(probability)
    np.maximum(log_emission, log_floor, out=log_emission)
    uniform_log = max(float(np.log(1.0 / 3.0)), float(log_floor))
    return log_emission - uniform_log


