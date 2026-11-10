"""Painting + raw evidence -> transmission-scoring inputs of one chromosome (PreparedChromosome).

Per painted component: the exact source factors of every sample (from the painting's emissions and
its own normalised transition table), at most ``markers_per_bin`` evenly spaced selected markers per
painting bin with their genotype likelihoods, the painted state classes (junction counts), the
meiotic switch probability and the measured map length between bins, and the information exponent of the
tempered view.

Method: docs/methods/pedigree.md#preparation-markers-source-factors-and-information-blocks
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import cast

import numpy as np
from numba import prange

from haplotype_reconstruction.science.painting.parameters import PaintingParameters

from ...contracts.evidence import ContigEvidence
from ...contracts.genetic_map import ChromosomeGeneticMap, measured_coordinates
from ...contracts.likelihoods import gather_likelihoods, likelihood_window_sites
from ...contracts.painting import ComponentPainting, Painting, TrackStatus
from ...contracts.pedigree import HammingTransition, PreparedChromosome, PreparedComponent, SourceFactors
from ...core.errors import InputError
from ...core.kernels import kernel
from ..painting import api as painting_hmm
from ..shared.recombination import haldane_fraction
from . import sources
from .kernels.signatures import OVERLOADS
from .parameters import EMISSION_SAMPLE_BLOCK, INFORMATIVE_GL_SPREAD, SELECTOR_SWITCH_FLOOR, PreparationParameters

# Painting tracks whose class is not an anchored founder identity (painting TrackStatus values).
_BACKGROUND_STATUS = TrackStatus.BACKGROUND
_UNANCHORED_STATUS = TrackStatus.UNANCHORED_TRAJECTORY
# Samples per block of a recomputed emission array: each block boundary lets this chromosome
# worker's CPU lease grow (no value changes).
_EMISSION_SAMPLE_BLOCK = EMISSION_SAMPLE_BLOCK
OMITTED = "no_observed_nonuniform_component_evidence"
# Smallest selector switch probability between bins (bins at equal map positions keep a finite switch cost).


def selected_markers(
    positions: np.ndarray, bin_edges: np.ndarray, markers_per_bin: int
) -> tuple[np.ndarray, np.ndarray]:
    """(indices into ``positions`` of the selected markers, bin-major; markers per bin [B]).

    A bin with more than ``markers_per_bin`` painting sites keeps the evenly spaced sites
    first + floor(slot * count / markers_per_bin).
    """
    n_bins = len(bin_edges) - 1
    chosen = []
    counts = np.zeros(n_bins, dtype=np.int64)
    for block in range(n_bins):
        first = int(np.searchsorted(positions, bin_edges[block]))
        last = int(np.searchsorted(positions, bin_edges[block + 1]))
        count = last - first
        if count < 1:
            continue
        if count <= markers_per_bin:
            indices = np.arange(first, last, dtype=np.int64)
        else:
            step = count / float(markers_per_bin)
            indices = np.asarray([first + int(slot * step) for slot in range(markers_per_bin)], dtype=np.int64)
        counts[block] = len(indices)
        chosen.append(indices)
    return (np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)), counts


@kernel(overloads=OVERLOADS["prepare._informative_counts"], cache=True, parallel=True, nogil=True)
def _informative_counts(likelihoods, observed, sites, spread):
    """int64 [N]: observed sites among ``sites`` whose GL row spread exceeds ``spread``."""
    counts = np.zeros(likelihoods.shape[0], dtype=np.int64)
    for sample in prange(likelihoods.shape[0]):
        count = 0
        for site in sites:
            a, b, c = likelihoods[sample, site, 0], likelihoods[sample, site, 1], likelihoods[sample, site, 2]
            if observed[sample, site] and max(a, b, c) - min(a, b, c) > spread:
                count += 1
        counts[sample] = count
    return counts


@kernel(overloads=OVERLOADS["prepare._information_exponent"], cache=True, parallel=True, nogil=True)
def _information_exponent(likelihoods, bin_sizes, markers_per_block, effective_markers):
    """float64 [N, B]: per child and bin min(effective, n) / n for the n non-uniform GL markers of the
    bin's information group (consecutive bins of >= ``markers_per_block`` selected markers)."""
    n_samples = likelihoods.shape[0]
    n_bins = len(bin_sizes)
    group_of_bin = np.empty(n_bins, dtype=np.int64)
    group = 0
    markers_in_group = 0
    for block in range(n_bins):
        if block > 0 and markers_in_group >= markers_per_block:
            group += 1
            markers_in_group = 0
        group_of_bin[block] = group
        markers_in_group += max(int(bin_sizes[block]), 1)
    starts = np.zeros(n_bins, dtype=np.int64)
    for block in range(1, n_bins):
        starts[block] = starts[block - 1] + bin_sizes[block - 1]
    exponent = np.zeros((n_samples, n_bins), dtype=np.float64)
    for child in prange(n_samples):
        informative = np.zeros(group + 1, dtype=np.int64)
        for block in range(n_bins):
            for site in range(starts[block], starts[block] + bin_sizes[block]):
                gl0 = likelihoods[child, site, 0]
                gl1 = likelihoods[child, site, 1]
                gl2 = likelihoods[child, site, 2]
                if gl0 != gl1 or gl1 != gl2:
                    informative[group_of_bin[block]] += 1
        for block in range(n_bins):
            count = informative[group_of_bin[block]]
            if count > 0:
                exponent[child, block] = min(effective_markers, float(count)) / float(count)
    return exponent


def _evidence_sites(evidence_positions: np.ndarray, positions: np.ndarray, *, contig: str = "unknown") -> np.ndarray:
    indices = np.searchsorted(evidence_positions, positions)
    present = indices < len(evidence_positions)
    if len(evidence_positions):
        present &= evidence_positions[np.minimum(indices, len(evidence_positions) - 1)] == positions
    if not np.all(present):
        missing = int(positions[np.flatnonzero(~present)[0]])
        raise InputError(f"painted sites are missing from the evidence record; contig={contig}, position={missing}")
    return indices.astype(np.int64)


def _source_factors(
    painted: ComponentPainting,
    evidence: ContigEvidence,
    sites: np.ndarray,
    painting_config: PaintingParameters,
    transition: HammingTransition,
    available: np.ndarray,
    claim_threads: Callable[[], None],
    component_index: int,
) -> SourceFactors:
    """Fill the dense factors while keeping only one sample batch's emissions/GL live.

    Every candidate's backward recurrence is independent. Cached emissions are unpacked
    by the same batch boundary as uncached emissions; neither route retains an all-sample
    emission array or its bin-major transpose beside the final right weights.
    """
    states, bins = painted.states, painted.bins
    samples, n_states = evidence.n_samples, states.n_states
    initial = np.empty((samples, n_states, n_states), dtype=np.float64)
    right = np.empty((samples, max(0, bins.n_bins - 1), n_states, n_states), dtype=np.float64)
    alt_probability = states.alt_probability() if painted.packed_emissions is None else None
    for first in range(0, samples, _EMISSION_SAMPLE_BLOCK):
        last = min(samples, first + _EMISSION_SAMPLE_BLOCK)
        claim_threads()
        if painted.packed_emissions is not None:
            emissions = painting_hmm.unpack_emissions(painted.packed_emissions[first:last], n_states)
        else:
            selected_gl, selected_observed = painting_hmm.gather_sites(
                evidence.likelihoods, evidence.observed, sites, slice(first, last)
            )
            emissions = painting_hmm.binned_emissions(
                selected_gl,
                selected_observed,
                cast(np.ndarray, alt_probability),
                states.background_frequency,
                bins.starts,
                bins.stops,
                robustness_epsilon=painting_config.robustness_epsilon,
                log_floor=painting_config.emission_log_floor,
            )
            emissions /= painting_hmm.design_effect(
                selected_gl, selected_observed, bins.starts, painted.transitions.site_correlation
            )[:, None, :]
            del selected_gl, selected_observed
        factors = sources.infer_factors(
            emissions, transition, available[first:last], component_index=component_index, sample_offset=first
        )
        initial[first:last] = factors.initial
        right[first:last] = factors.right_weight
        del factors, emissions
    return SourceFactors(initial, right, available)


def _prepare_component(
    index: int,
    painted: ComponentPainting,
    evidence: ContigEvidence,
    painting_config: PaintingParameters,
    chromosome_map: ChromosomeGeneticMap,
    config: PreparationParameters,
    claim_threads: Callable[[], None],
    rate_is_measured: bool = False,
    *,
    contig: str = "unknown",
) -> PreparedComponent | None:
    """One component's scoring inputs, or None without observed non-uniform selected evidence."""
    states, bins = painted.states, painted.bins
    sites = _evidence_sites(evidence.positions, states.positions, contig=contig)
    chosen, bin_sizes = selected_markers(states.positions, bins.edges, config.markers_per_bin)
    if not len(chosen):
        return None
    marker_sites = sites[chosen]
    gl = gather_likelihoods(evidence.likelihoods, marker_sites)
    observed = np.ascontiguousarray(evidence.observed[:, marker_sites])
    if not np.any((gl[..., 0] != gl[..., 1]) | (gl[..., 1] != gl[..., 2])):
        return None
    claim_threads()
    if isinstance(evidence.likelihoods, np.ndarray) and isinstance(evidence.observed, np.ndarray):
        available = _informative_counts(evidence.likelihoods, evidence.observed, sites, INFORMATIVE_GL_SPREAD) >= 1
    else:
        # Availability is an integer count; sum bounded projections in site order.
        counts = np.zeros(evidence.n_samples, np.int64)
        size = (
            likelihood_window_sites(evidence.likelihoods) if not isinstance(evidence.likelihoods, np.ndarray) else 2048
        )
        for left in range(0, len(sites), size):
            selected = sites[left : left + size]
            values = gather_likelihoods(evidence.likelihoods, selected)
            seen = np.ascontiguousarray(evidence.observed[:, selected])
            counts += _informative_counts(values, seen, np.arange(len(selected), dtype=np.int64), INFORMATIVE_GL_SPREAD)
            del values, seen
        available = counts >= 1
    transition = HammingTransition(*painted.transitions.probabilities())
    factors = _source_factors(painted, evidence, sites, painting_config, transition, available, claim_threads, index)
    unknown = (painted.status == _BACKGROUND_STATUS) | (painted.status == _UNANCHORED_STATUS)
    state_classes = np.where(unknown, -1, painted.classes).astype(np.int16)
    # Which parental homologue is transmitted changes between adjacent bin centres when an odd number of
    # crossovers falls between them: one meiosis's Haldane recombination fraction, kept within [floor, 1/2].
    theta = np.clip(
        haldane_fraction(chromosome_map.interval_morgans(bins.centers[:-1], bins.centers[1:])),
        SELECTOR_SWITCH_FLOOR,
        0.5,
    )
    return PreparedComponent(
        component_index=index,
        named_alleles=np.ascontiguousarray(states.named_alleles[:, chosen]),
        bin_sizes=bin_sizes,
        # Classes called anywhere in the whole component (before marker subsampling) are anchored.
        anchored_states=np.concatenate((np.any(states.named_alleles >= 0, axis=1), [False])),
        transition=transition,
        factors=factors,
        robustness_epsilon=float(painting_config.robustness_epsilon),
        selector_switch_probability=np.ascontiguousarray(theta),
        measured_morgans=np.ascontiguousarray(
            np.diff(measured_coordinates(chromosome_map, bins.centers, rate_is_measured))
        ),
        genotype_likelihoods=gl,
        observed=observed,
        state_classes=np.ascontiguousarray(np.transpose(state_classes, (0, 2, 1))),
        information_exponent=np.empty((gl.shape[0], 0)),
    )


def _information_exponents(
    prepared: list[PreparedComponent], config: PreparationParameters, claim_threads: Callable[[], None]
) -> np.ndarray:
    """Chromosome-wide marker groups with only one sample batch's concatenated GL live."""
    samples = prepared[0].genotype_likelihoods.shape[0]
    bin_sizes = np.concatenate([c.bin_sizes for c in prepared])
    exponent = np.empty((samples, len(bin_sizes)), dtype=np.float64)
    # Information groups cross component boundaries. Keep that complete bin/marker axis,
    # but concatenate only one independent sample batch rather than all N marker GL rows.
    for first in range(0, samples, _EMISSION_SAMPLE_BLOCK):
        last = min(samples, first + _EMISSION_SAMPLE_BLOCK)
        claim_threads()
        selected_gl = np.concatenate([c.genotype_likelihoods[first:last] for c in prepared], axis=1)
        exponent[first:last] = _information_exponent(
            selected_gl,
            bin_sizes,
            config.markers_per_information_block,
            float(config.effective_markers_per_information_block),
        )
        del selected_gl
    return exponent


def prepare_chromosome(
    painting: Painting,
    evidence: ContigEvidence,
    chromosome_map: ChromosomeGeneticMap,
    config: PreparationParameters,
    claim_threads: Callable[[], None] = lambda: None,
    rate_is_measured: bool = False,
) -> PreparedChromosome:
    """Prepared components of one contig from its painting (ChromosomePainting) and evidence
    (ContigEvidence, all samples), with the chromosome-wide information exponents. ``rate_is_measured``: see
    measured_coordinates."""
    prepared = []
    for index, painted in enumerate(painting.components):
        if painted.states.n_sites == 0 or painted.bins.n_bins == 0:
            continue
        claim_threads()
        component = _prepare_component(
            index,
            painted,
            evidence,
            painting.parameters,
            chromosome_map,
            config,
            claim_threads,
            rate_is_measured,
            contig=painting.contig,
        )
        if component is not None:
            prepared.append(component)
    if not prepared:
        return PreparedChromosome(painting.contig, len(painting.components), (), OMITTED)
    claim_threads()
    exponent = _information_exponents(prepared, config, claim_threads)
    components = []
    offset = 0
    for component in prepared:
        components.append(
            replace(
                component, information_exponent=np.ascontiguousarray(exponent[:, offset : offset + component.n_bins])
            )
        )
        offset += component.n_bins
    return PreparedChromosome(painting.contig, len(painting.components), tuple(components))
