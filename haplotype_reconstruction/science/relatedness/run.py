"""Build bound-order family terms and fit the full-sib partition.

Owns family entries. No storage or runtime policy.
Method: docs/methods/relatedness.md.
The site axis and every reduction follow the supplied contig order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ...contracts.genetic_map import ChromosomeGeneticMap, measured_coordinates
from ...contracts.likelihoods import CodedLikelihoods, gather_likelihoods
from ...contracts.observations import PairArray
from ...contracts.relatedness import RelatednessFamilies
from ...core.concurrency import ThreadSource
from ...core.errors import InputError
from .frequency import allele_frequency
from .parameters import AlleleFrequencyParameters, FamilyParameters
from .search import FamilyFit, fit
from .support import support_units
from .terms import family_terms

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
    """One contig's float64 child/parent terms and int64 unit boundaries."""

    child: np.ndarray
    log_prior: np.ndarray
    log_posterior: np.ndarray
    units: np.ndarray


@dataclass(frozen=True)
class FamilyTerms:
    """Family input arrays on the bound concatenated site axis; all inputs read-only."""

    sample_ids: tuple[str, ...]
    contigs: tuple[str, ...]
    child: np.ndarray
    log_prior: np.ndarray
    log_posterior: np.ndarray
    ranges: tuple[tuple[int, int], ...]
    chromosome_sites: tuple[int, ...]


def contig_terms(
    evidence: Evidence,
    genetic_map: ChromosomeGeneticMap,
    family_parameters: FamilyParameters,
    frequency_parameters: AlleleFrequencyParameters = _DEFAULT_FREQUENCY,
    *,
    map_work=map,
) -> ContigTerms:
    """Return C[N,M,9], logpi[M,3], logpost[N,M,3] and support boundaries; mutate nothing. ``map_work`` (ordered,
    optional): the allele frequencies' site windows concurrently."""
    gl, observed, positions = evidence.likelihoods, evidence.observed, evidence.positions
    if not len(positions):
        n = len(evidence.sample_ids)
        return ContigTerms(
            np.empty((n, 0, 9)), np.empty((0, 3)), np.empty((n, 0, 3)), np.asarray([0, 0], dtype=np.int64)
        )
    p = allele_frequency(gl, observed, frequency_parameters, map_work=map_work)
    keep = (np.minimum(p, 1 - p) >= family_parameters.minor_allele_frequency) & (
        observed.mean(axis=0) >= family_parameters.minimum_observed
    )
    sites = np.flatnonzero(keep)[:: family_parameters.family_site_step]
    child, prior, post = family_terms(gather_likelihoods(gl, sites), observed[:, sites], p[sites], family_parameters)
    units = support_units(
        measured_coordinates(genetic_map, positions[sites], family_parameters.rate_is_measured),
        family_parameters.support_unit_morgans,
    )
    return ContigTerms(child, prior, post, units)


def concatenate_terms(
    sample_ids: tuple[str, ...], contigs: tuple[str, ...], terms: tuple[ContigTerms, ...]
) -> FamilyTerms:
    """Concatenate float64 family terms in supplied contig order; drop empty support units."""
    if len(contigs) != len(terms):
        raise InputError("relatedness contigs and term batches must have equal lengths")
    if not terms:
        raise InputError("relatedness requires at least one bound contig")
    ranges: list[tuple[int, int]] = []
    chromosome_sites: list[int] = []
    lo = 0
    for value in terms:
        chromosome_sites.append(value.child.shape[1])
        ranges.extend(
            (lo + int(a), lo + int(b)) for a, b in zip(value.units[:-1], value.units[1:], strict=False) if b > a
        )
        lo += value.child.shape[1]
    return FamilyTerms(
        sample_ids,
        contigs,
        np.ascontiguousarray(np.concatenate([t.child for t in terms], axis=1)),
        np.ascontiguousarray(np.concatenate([t.log_prior for t in terms], axis=0)),
        np.ascontiguousarray(np.concatenate([t.log_posterior for t in terms], axis=1)),
        tuple(ranges),
        tuple(chromosome_sites),
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
