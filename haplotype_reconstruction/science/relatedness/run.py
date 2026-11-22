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
from ...contracts.relatedness import RelatednessFamilies, SourceStructure
from ...core.concurrency import ThreadSource
from ...core.errors import InputError
from ..shared.material import partition_coordinates
from .ancestry import apply_sources, composition, fit_sources, source_warnings
from ..shared.observation import mix_likelihoods
from .parameters import FamilyParameters
from .search import FamilyFit, fit
from .terms import family_terms, scaled_likelihoods

_DEFAULT_FAMILY = FamilyParameters()

__all__ = [
    "ContigTerms",
    "FamilyTerms",
    "contig_terms",
    "concatenate_terms",
    "with_sources",
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

    @property
    def cohort_frequencies(self) -> np.ndarray | None: ...

    @property
    def read_sharing_range(self) -> int: ...

    @property
    def read_sharing_status(self) -> str: ...

    @property
    def read_sharing_reach(self) -> int: ...


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
    # rl21, two source populations only (ancestry.apply_sources): the classes' log genotype laws [K, M, 3], their
    # log laws over the parental configurations [K, M, G], the child terms over configuration pairs [N, M, G x G]
    # and each sample's log configuration posterior [N, M, G]
    class_prior: np.ndarray | None = None
    ancestry_prior: np.ndarray | None = None
    ancestry_child: np.ndarray | None = None
    ancestry_posterior: np.ndarray | None = None
    ancestry_emission: np.ndarray | None = None  # [N, M, 3]: each ancestry state's probability of the reads
    # B3's record (ADVICE2 17.3): the evidence's read-sharing status and the reach (bp) of the curve its consumers read
    read_sharing_status: str = ""
    read_sharing_reach_bp: int = 0


@dataclass(frozen=True)
class FamilyTerms:
    """Family input arrays on the bound concatenated site axis; all inputs read-only. ``ranges``: the support
    units' site ranges, ``weights`` their working Morgans (DESIGN_417 D5). Positions (bp), float32 scaled
    likelihoods and the observation mask serve only the rate estimate (rl20). ``class_prior`` (rl21, two source
    populations only): the ancestry classes' log genotype laws float64 [K, M, 3]; ``ancestry_prior`` [K, M, G],
    ``ancestry_child`` [N, M, G x G] and ``ancestry_posterior`` [N, M, G] are the family fit's terms over the
    parental configurations (ancestry.CONFIGURATIONS), and ``log_posterior`` is the genotype marginal of the
    configuration posterior. ``child`` stays the genotype-level child terms (the rate estimate reads them)."""

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
    class_prior: np.ndarray | None = None
    ancestry_prior: np.ndarray | None = None
    ancestry_child: np.ndarray | None = None
    ancestry_posterior: np.ndarray | None = None
    # rl21: (log-likelihood ratios [1 + U, N, T, T] of each sample's ancestry along the map under parents of
    # transmitted shares (a, b) against its own chain, over the genome (row 0) and each support unit; the classes'
    # share index [K]; the samples' [N]): ancestry.composition
    ancestry_composition: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


def spaced_sites(sites: np.ndarray, positions: np.ndarray, reach_bp: int) -> np.ndarray:
    """``sites`` (ascending indices) without every site closer than ``reach_bp`` to the previous kept one."""
    if reach_bp <= 0 or len(sites) < 2:
        return sites
    keep = np.ones(len(sites), dtype=bool)
    last = positions[sites[0]]
    for i in range(1, len(sites)):
        if positions[sites[i]] - last < reach_bp:
            keep[i] = False
        else:
            last = positions[sites[i]]
    return sites[keep]


def contig_terms(
    evidence: Evidence,
    genetic_map: ChromosomeGeneticMap,
    family_parameters: FamilyParameters,
    *,
    outlier_rate: float,
) -> ContigTerms:
    """Return C[N,M,9], logpi[M,3], logpost[N,M,3] and the family sites' working-map coordinates; mutate nothing.

    The ALT allele frequency of a site is that of its cohort genotype frequencies q (stored with the evidence),
    p = q1 / 2 + q2: the EM's pseudo-counts keep it inside (0, 1), so no clip is needed. A latent parent's genotype
    prior stays Hardy-Weinberg at p (a genetic prior, distinct from the observation model's outlier law q, which has no
    Hardy-Weinberg assumption). The family sites' likelihoods
    are the mixed likelihoods of the observation model at ``outlier_rate`` (the run's rate before the calibration
    barrier, science.shared.observation): every transmission law of a child term scores (1 - eps) P + eps B."""
    gl, observed, positions = evidence.likelihoods, evidence.observed, evidence.positions
    if not len(positions):
        n = len(evidence.sample_ids)
        return ContigTerms(
            np.empty((n, 0, 9)), np.empty((0, 3)), np.empty((n, 0, 3)), np.empty(0), np.empty(0),
            np.empty((n, 0, 3), dtype=np.float32), np.empty((n, 0), dtype=np.bool_),
        )
    cohort = evidence.cohort_frequencies
    if cohort is None:
        raise InputError("relatedness needs the evidence's cohort genotype frequencies")
    p = 0.5 * cohort[:, 1] + cohort[:, 2]
    keep = (np.minimum(p, 1 - p) >= family_parameters.minor_allele_frequency) & (
        observed.mean(axis=0) >= family_parameters.minimum_observed
    )
    # The stride is a compute budget, not a dependence estimate (a named exception). Kept sites closer than the contig's
    # resolved read-sharing range would share reads, so of two such sites the later is dropped (4.21, B3); an
    # unresolved range arrives as 0 and the stride stays alone (ADVICE2 14.1 item 4).
    sites = spaced_sites(np.flatnonzero(keep)[:: family_parameters.family_site_step], positions,
                         int(evidence.read_sharing_range))
    site_likelihoods = mix_likelihoods(gather_likelihoods(gl, sites), cohort[sites], outlier_rate)
    site_observed = np.asarray(observed[:, sites], dtype=np.bool_)
    child, prior, post = family_terms(site_likelihoods, site_observed, p[sites], family_parameters)
    site_positions = np.asarray(positions[sites], dtype=np.float64)
    coordinates = np.asarray(genetic_map.cumulative_morgans(site_positions), dtype=np.float64).reshape(-1)
    scaled = scaled_likelihoods(site_likelihoods, site_observed, family_parameters).astype(np.float32)
    return ContigTerms(child, prior, post, coordinates, site_positions, scaled, site_observed,
                       read_sharing_status=str(evidence.read_sharing_status),
                       read_sharing_reach_bp=int(evidence.read_sharing_reach))


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
    parameters: FamilyParameters = _DEFAULT_FAMILY, sources=None,
) -> FamilyTerms:
    """Concatenate float64 family terms in supplied contig order with the pooled Morgan-weighted support units."""
    if len(contigs) != len(terms):
        raise InputError("relatedness contigs and term batches must have equal lengths")
    if not terms:
        raise InputError("relatedness requires at least one bound contig")
    chromosome_sites = [value.child.shape[1] for value in terms]
    ranges, weights = support_axis(terms, parameters.support_unit_morgans, parameters.support_tail_morgans)
    rated = all(t.positions is not None and t.scaled is not None and t.observed is not None for t in terms)
    classed = all(t.class_prior is not None for t in terms)
    return FamilyTerms(
        sample_ids,
        contigs,
        np.ascontiguousarray(np.concatenate([t.child for t in terms], axis=1)),
        np.ascontiguousarray(np.concatenate([t.log_prior for t in terms], axis=0)),
        np.ascontiguousarray(np.concatenate([t.log_posterior for t in terms], axis=1)),
        tuple(ranges),
        tuple(chromosome_sites),
        tuple(weights),
        # None: terms built for the family fit alone (no rate estimate)
        np.concatenate([t.positions for t in terms]) if rated else None,
        np.ascontiguousarray(np.concatenate([t.scaled for t in terms], axis=1)) if rated else None,
        np.ascontiguousarray(np.concatenate([t.observed for t in terms], axis=1)) if rated else None,
        np.ascontiguousarray(np.concatenate([t.class_prior for t in terms], axis=1)) if classed else None,
        np.ascontiguousarray(np.concatenate([t.ancestry_prior for t in terms], axis=1)) if classed else None,
        np.ascontiguousarray(np.concatenate([t.ancestry_child for t in terms], axis=1)) if classed else None,
        np.ascontiguousarray(np.concatenate([t.ancestry_posterior for t in terms], axis=1)) if classed else None,
        _composition(terms, ranges, sources, parameters)
        if classed and all(t.ancestry_emission is not None for t in terms) else None,
    )


def _composition(terms, ranges, sources, parameters):
    """rl21: ancestry.composition over the genome and each support unit of the concatenated family-site axis."""
    if sources is None:
        raise InputError("terms with ancestry classes need their fitted source model")
    emission = np.ascontiguousarray(np.concatenate([t.ancestry_emission for t in terms], axis=1))
    gaps = np.concatenate(
        [np.concatenate(([np.inf], np.diff(t.coordinates))) for t in terms if len(t.coordinates)] or [np.empty(0)]
    )
    spans = np.asarray([(0, emission.shape[1]), *ranges], dtype=np.int64)
    table, class_type, sample_type, _ = composition(
        emission, np.ascontiguousarray(sources.sample_shares), np.ascontiguousarray(gaps), spans, sources.shares,
        sources.classes, float(parameters.source_switch_rate),
    )
    return table, class_type, sample_type


def fit_families(
    terms: FamilyTerms, parameters: FamilyParameters = _DEFAULT_FAMILY, *, threads: ThreadSource = 1,
    sources=None,
) -> RelatednessFamilies:
    """Fit both bounded searches from input terms; no report selection or input mutation. ``threads`` is a
    fixed count or caller-owned live source, read at every batch/search/sweep boundary.
    A live source may refresh the ambient kernel team; values never depend on its size. ``sources``: the fitted
    ancestry model (ancestry.SourceModel) of terms that carry class laws (rl21), recorded with the families."""
    classed = terms.ancestry_prior is not None
    fitted = fit(
        terms.ancestry_child if classed else terms.child,
        terms.log_prior,
        terms.ancestry_posterior if classed else terms.log_posterior,
        terms.ranges,
        terms.weights,
        max_sweeps=parameters.max_sweeps,
        parameters=parameters,
        threads=threads,
        class_prior=terms.ancestry_prior,
        composition=terms.ancestry_composition,
    )

    return publish_families(terms, fitted, sources, parameters)


def with_sources(terms: tuple[ContigTerms, ...], parameters: FamilyParameters = _DEFAULT_FAMILY):
    """(terms, model): with ``source_populations`` = 2, the contigs' terms with the ancestry classes' laws and the
    class-aware sequenced-parent posteriors, and the fitted model (rl21); with 1, the terms themselves and None (no
    fit runs)."""
    if parameters.source_populations < 2 or not terms or terms[0].child.shape[0] < 2:
        return terms, None
    if any(t.scaled is None or t.observed is None for t in terms):
        raise InputError("two source populations need the terms' scaled likelihoods and observation masks")
    model = fit_sources(terms, parameters)
    return apply_sources(terms, model, parameters), model


def publish_families(
    terms: FamilyTerms, fitted: FamilyFit, sources=None, parameters: FamilyParameters = _DEFAULT_FAMILY
) -> RelatednessFamilies:
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
        sources=None if sources is None else _source_structure(sources, fitted, order, parameters),
    )


def _source_structure(model, fitted: FamilyFit, order: np.ndarray, parameters: FamilyParameters) -> SourceStructure:
    """The ancestry model's record: sample shares, the fitted families' slot classes in published family order."""
    frequencies = np.concatenate(model.frequencies, axis=1)
    difference = np.abs(frequencies[0] - frequencies[1])
    differentiated = difference > parameters.source_differentiation
    strong = float(np.mean(difference > parameters.source_strong_differentiation)) if difference.size else 0.0
    return SourceStructure(
        populations=2,
        shares=tuple(float(s) for s in model.shares),
        classes=np.asarray(model.classes, dtype=np.int64),
        sample_shares=np.asarray(model.sample_shares, dtype=np.float64),
        slot_class=np.asarray(fitted.slot_class, dtype=np.int64)[order],
        slot_class_probability=np.asarray(fitted.slot_class_probability, dtype=np.float64)[order],
        differentiated_fraction=float(np.mean(differentiated)) if differentiated.size else 0.0,
        fit_sites=int(model.fit_sites),
        loglik=float(model.loglik),
        start_spread=float(max(model.start_logliks) - min(model.start_logliks)),
        iterations=int(model.iterations),
        strong_fraction=strong,
        depth=float(model.depth),
        warnings=source_warnings(strong, float(model.depth), parameters),
    )
