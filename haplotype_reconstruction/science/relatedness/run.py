"""Build bound-order family terms and fit the full-sib partition.

Owns family entries. No storage or runtime policy.
Method: docs/methods/relatedness.md.
The site axis and every reduction follow the supplied contig order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ...contracts.genetic_map import ChromosomeGeneticMap
from ...contracts.likelihoods import CodedLikelihoods, gather_likelihoods
from ...contracts.observations import PairArray
from ...contracts.relatedness import RelatednessFamilies
from ...core.concurrency import ThreadSource
from ...core.errors import InputError
from ..shared.material import partition_coordinates
from .frequency import allele_frequency
from .parameters import AlleleFrequencyParameters, FamilyParameters
from .search import FamilyFit, fit
from .terms import family_terms, scaled_likelihoods

_DEFAULT_FAMILY = FamilyParameters()
_DEFAULT_FREQUENCY = AlleleFrequencyParameters()

__all__ = [
    "ContigTerms",
    "FamilyTerms",
    "contig_terms",
    "concatenate_terms",
    "fit_families",
    "RelatednessFamilies",
    "publish_families",
]


class Evidence(Protocol):
    """The observed evidence fields consumed by the family model."""

    @property
    def sample_ids(self) -> tuple[str, ...]: ...

    @property
    def positions(self) -> np.ndarray: ...

    @property
    def likelihoods(self) -> np.ndarray | CodedLikelihoods: ...

    @property
    def observed(self) -> np.ndarray | PairArray: ...


@dataclass(frozen=True)
class ContigTerms:
    """One contig's float64 child/parent terms and its family sites' working-map coordinates (Morgans,
    nondecreasing: the configured map, or the configured rate where it applies); the family sites' bp positions
    (float64[M]), float32 scaled genotype likelihoods (unobserved cells 1) and observation mask feed the rate
    estimate (rl20)."""

    child: np.ndarray
    log_prior: np.ndarray
    log_posterior: np.ndarray
    coordinates: np.ndarray
    positions: np.ndarray | None = None
    scaled: np.ndarray | None = None
    observed: np.ndarray | None = None


@dataclass(frozen=True)
class FamilyTerms:
    """Family input arrays on the bound concatenated site axis; all inputs read-only. ``ranges``: the support
    units' site ranges, ``weights`` their working Morgans (DESIGN_417 D5). Positions (bp), float32 scaled
    likelihoods and the observation mask serve only the rate estimate (rl20)."""

    sample_ids: tuple[str, ...]
    contigs: tuple[str, ...]
    child: np.ndarray
    log_prior: np.ndarray
    log_posterior: np.ndarray
    ranges: tuple[tuple[int, int], ...]
    chromosome_sites: tuple[int, ...]
    weights: tuple[float, ...] = ()
    positions: np.ndarray | None = None
    scaled: np.ndarray | None = None
    observed: np.ndarray | None = None


def contig_terms(
    evidence: Evidence,
    genetic_map: ChromosomeGeneticMap,
    family_parameters: FamilyParameters,
    frequency_parameters: AlleleFrequencyParameters = _DEFAULT_FREQUENCY,
    *,
    map_work=map,
) -> ContigTerms:
    """Return C[N,M,9], logpi[M,3], logpost[N,M,3] and the family sites' working-map coordinates; mutate nothing.
    ``map_work`` (ordered, optional): the allele frequencies' site windows concurrently."""
    gl, observed, positions = evidence.likelihoods, evidence.observed, evidence.positions
    if not len(positions):
        n = len(evidence.sample_ids)
        return ContigTerms(
            np.empty((n, 0, 9)), np.empty((0, 3)), np.empty((n, 0, 3)), np.empty(0), np.empty(0),
            np.empty((n, 0, 3), dtype=np.float32), np.empty((n, 0), dtype=np.bool_),
        )
    p = allele_frequency(gl, observed, frequency_parameters, map_work=map_work)
    keep = (np.minimum(p, 1 - p) >= family_parameters.minor_allele_frequency) & (
        observed.mean(axis=0) >= family_parameters.minimum_observed
    )
    sites = np.flatnonzero(keep)[:: family_parameters.family_site_step]
    site_likelihoods, site_observed = gather_likelihoods(gl, sites), np.asarray(observed[:, sites], dtype=np.bool_)
    child, prior, post = family_terms(site_likelihoods, site_observed, p[sites], family_parameters)
    site_positions = np.asarray(positions[sites], dtype=np.float64)
    coordinates = np.asarray(genetic_map.cumulative_morgans(site_positions), dtype=np.float64).reshape(-1)
    scaled = scaled_likelihoods(site_likelihoods, site_observed, family_parameters).astype(np.float32)
    return ContigTerms(child, prior, post, coordinates, site_positions, scaled, site_observed)


def support_axis(
    terms: tuple[ContigTerms, ...], unit_morgans: float, tail_morgans: float
) -> tuple[tuple[tuple[int, int], ...], tuple[float, ...]]:
    """Support units of the pooled family-site axis (DESIGN_417 D5, RELCAL 1.1): each contig's first family site at
    the running offset (no gap between contigs, configured order), cut into consecutive ``unit_morgans`` pieces
    (a final piece shorter than ``tail_morgans`` joins the one before it; a site exactly at a cut belongs to the
    unit on its right); empty units omitted. Returns each unit's site range on the concatenated axis and its
    working Morgans (the coordinate of its last site minus that of the previous unit's last site; a gap without
    sites is carried by the next unit); a degenerate axis of 0 Morgans is one unit of weight 1."""
    parts, offset = [], 0.0
    for value in terms:
        if not len(value.coordinates):
            continue
        parts.append(offset + (value.coordinates - value.coordinates[0]))
        offset += float(value.coordinates[-1] - value.coordinates[0])
    if not parts:
        return (), ()
    coordinates = np.concatenate(parts)
    _, units = partition_coordinates(coordinates, unit_morgans, tail_morgans)
    ranges = tuple((first, stop) for first, stop, _, _ in units)
    weights = tuple(morgans for _, _, morgans, _ in units)
    if sum(weights) <= 0:
        return ((0, len(coordinates)),), (1.0,)
    return ranges, weights


def concatenate_terms(
    sample_ids: tuple[str, ...], contigs: tuple[str, ...], terms: tuple[ContigTerms, ...],
    parameters: FamilyParameters = _DEFAULT_FAMILY,
) -> FamilyTerms:
    """Concatenate float64 family terms in supplied contig order with the pooled Morgan-weighted support units."""
    if len(contigs) != len(terms):
        raise InputError("relatedness contigs and term batches must have equal lengths")
    if not terms:
        raise InputError("relatedness requires at least one bound contig")
    chromosome_sites = [value.child.shape[1] for value in terms]
    ranges, weights = support_axis(terms, parameters.support_unit_morgans, parameters.support_tail_morgans)
    return FamilyTerms(
        sample_ids,
        contigs,
        np.ascontiguousarray(np.concatenate([t.child for t in terms], axis=1)),
        np.ascontiguousarray(np.concatenate([t.log_prior for t in terms], axis=0)),
        np.ascontiguousarray(np.concatenate([t.log_posterior for t in terms], axis=1)),
        tuple(ranges),
        tuple(chromosome_sites),
        tuple(weights),
        *(
            (
                np.concatenate([t.positions for t in terms]),
                np.ascontiguousarray(np.concatenate([t.scaled for t in terms], axis=1)),
                np.ascontiguousarray(np.concatenate([t.observed for t in terms], axis=1)),
            )
            if all(t.positions is not None and t.scaled is not None and t.observed is not None for t in terms)
            else ()  # terms built for the family fit alone (no rate estimate)
        ),
    )


def fit_families(
    terms: FamilyTerms, parameters: FamilyParameters = _DEFAULT_FAMILY, *, threads: ThreadSource = 1
) -> RelatednessFamilies:
    """Fit both bounded searches from input terms; no report selection or input mutation. ``threads`` is a
    fixed count or caller-owned live source, read at every batch/search/sweep boundary.
    A live source may refresh the ambient kernel team; values never depend on its size."""
    fitted = fit(
        terms.child,
        terms.log_prior,
        terms.log_posterior,
        terms.ranges,
        terms.weights,
        max_sweeps=parameters.max_sweeps,
        parameters=parameters,
        threads=threads,
    )

    return publish_families(terms, fitted)


def publish_families(terms: FamilyTerms, fitted: FamilyFit) -> RelatednessFamilies:
    """L6: remap family rows and the complete aligned half-sib table at publication only."""
    order = np.asarray(sorted(range(len(fitted.members)), key=lambda f: int(fitted.members[f][0])), dtype=np.int64)
    remap = np.empty(len(order), dtype=np.int64)
    remap[order] = np.arange(len(order), dtype=np.int64)
    family = np.full(len(terms.sample_ids), -1, dtype=np.int64)
    for old, members in enumerate(fitted.members):
        family[members] = remap[old]
    links = np.sort(remap[fitted.half_sib], axis=1)
    rows = np.lexsort((links[:, 1], links[:, 0]))
    return RelatednessFamilies(
        sample_ids=terms.sample_ids,
        contigs=terms.contigs,
        family=family,
        parents=fitted.parents[order],
        member_support=fitted.member_support,
        parent_support=fitted.parent_support[order],
        half_sib=links[rows],
        half_sib_gain=fitted.half_sib_gain[rows],
        converged=fitted.converged,
        search=fitted.search,
        sweeps=fitted.sweeps,
        loglik=fitted.loglik,
        search_margin=fitted.search_margin,
        family_sites=terms.child.shape[1],
        chromosome_sites=terms.chromosome_sites,
        support_units=len(terms.ranges),
    )
