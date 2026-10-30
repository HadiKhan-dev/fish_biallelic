"""Painting + raw evidence -> transmission-scoring inputs of one chromosome (PreparedChromosome).

Per painted component: the exact source factors of every sample (from the painting's emissions and
its own normalised transition table), at most ``markers_per_bin`` evenly spaced selected markers per
painting bin with their genotype likelihoods, the painted state classes (junction counts), the
meiotic switch probability and the measured map length between bins, and the information exponent of the
tempered view.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from numba import njit, prange

from ..model.genetic_map import haldane_fraction
from ..painting import hmm as painting_hmm
from . import sources
from .config import PreparationConfig
from .projection import INFORMATIVE_GL_SPREAD
from .records import HammingTransition, PreparedChromosome, PreparedComponent

# Painting tracks whose class is not an anchored founder identity (painting TrackStatus values).
_BACKGROUND_STATUS = 4
_UNANCHORED_STATUS = 5
# Samples per block of a recomputed emission array: each block boundary lets this chromosome
# worker's CPU lease grow (no value changes).
_EMISSION_SAMPLE_BLOCK = 16
OMITTED = "no_observed_nonuniform_component_evidence"
# Smallest selector switch probability between bins (bins at equal map positions keep a finite switch cost).
SELECTOR_SWITCH_FLOOR = 1e-15


def selected_markers(positions: np.ndarray, bin_edges: np.ndarray,
                     markers_per_bin: int) -> tuple[np.ndarray, np.ndarray]:
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
            indices = np.asarray([first + int(slot * step) for slot in range(markers_per_bin)],
                                 dtype=np.int64)
        counts[block] = len(indices)
        chosen.append(indices)
    return (np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)), counts


@njit(cache=True, parallel=True)
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


@njit(cache=True, parallel=True)
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


def _evidence_sites(evidence_positions: np.ndarray, positions: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(evidence_positions, positions)
    if (np.any(indices >= len(evidence_positions))
            or np.any(evidence_positions[np.minimum(indices, len(evidence_positions) - 1)] != positions)):
        raise ValueError("painted sites are missing from the evidence record")
    return indices.astype(np.int64)


def measured_coordinates(chromosome_map, positions, rate_is_measured: bool = False) -> np.ndarray:
    """float64 [n]: map coordinates (Morgans) of ``positions`` (bp) as far as a measurement supports them (release
    3.2). The difference of two coordinates is the measured map length of the interval between them, that is, of its
    intersection with the measured span:
    - ``rate_is_measured`` ([genetic_map] rate_is_measured: the fallback rate is the organism's stated genome-wide
      average): the run's map everywhere;
    - a supplied map: its coordinates with the positions clipped to its knots (the fallback rate extends a map's
      tails, but a tail is a placeholder, not a measurement);
    - otherwise zeros: the fallback rate measures no recombination."""
    positions = np.asarray(positions, dtype=np.float64)
    if rate_is_measured:
        return np.asarray(chromosome_map.cumulative_morgans(positions), dtype=np.float64)
    if not chromosome_map.has_map:
        return np.zeros(len(positions))
    knots = chromosome_map.positions_bp
    return np.asarray(chromosome_map.cumulative_morgans(np.clip(positions, knots[0], knots[-1])),
                      dtype=np.float64)


def _prepare_component(index: int, painted, evidence, painting_config, chromosome_map,
                       config: PreparationConfig, claim_threads,
                       rate_is_measured: bool = False) -> PreparedComponent | None:
    """One component's scoring inputs, or None without observed non-uniform selected evidence."""
    states, bins = painted.states, painted.bins
    sites = _evidence_sites(evidence.positions, states.positions)
    chosen, bin_sizes = selected_markers(states.positions, bins.edges, config.markers_per_bin)
    if not len(chosen):
        return None
    marker_sites = sites[chosen]
    gl = np.ascontiguousarray(evidence.likelihoods[:, marker_sites])
    observed = np.ascontiguousarray(evidence.observed[:, marker_sites])
    if not np.any((gl[..., 0] != gl[..., 1]) | (gl[..., 1] != gl[..., 2])):
        return None
    claim_threads()
    available = _informative_counts(evidence.likelihoods, evidence.observed, sites,
                                    INFORMATIVE_GL_SPREAD) >= 1
    n_states = states.n_states
    if painted.packed_emissions is not None:
        emissions = painting_hmm.unpack_emissions(painted.packed_emissions, n_states)
    else:
        # Components beyond the painting's emission cache: recompute in sample blocks so the
        # lease can grow (values are identical).
        selected_gl, selected_observed = painting_hmm.gather_sites(evidence.likelihoods,
                                                                   evidence.observed, sites)
        emissions = painting_hmm.binned_emissions(
            selected_gl, selected_observed, states.alt_probability(), states.background_frequency,
            bins.starts, bins.stops, robustness_epsilon=painting_config.robustness_epsilon,
            log_floor=painting_config.emission_log_floor, sample_block=_EMISSION_SAMPLE_BLOCK,
            before_block=claim_threads)
        # Tempered as the painting's decoder and its packed emissions (release 2.14): each sample's bin divided by its
        # design effect.
        emissions /= painting_hmm.design_effect(selected_gl, selected_observed, bins.starts,
                                                painted.transitions.site_correlation)[:, None, :]
        del selected_gl, selected_observed
    transition = HammingTransition(*painted.transitions.probabilities())
    claim_threads()
    factors = sources.infer_factors(emissions, transition, available)
    del emissions
    unknown = (painted.status == _BACKGROUND_STATUS) | (painted.status == _UNANCHORED_STATUS)
    state_classes = np.where(unknown, -1, painted.classes).astype(np.int16)
    # Which parental homologue is transmitted changes between adjacent bin centres when an odd number of
    # crossovers falls between them: one meiosis's Haldane recombination fraction, kept within [floor, 1/2].
    theta = np.clip(haldane_fraction(chromosome_map.interval_morgans(bins.centers[:-1], bins.centers[1:])),
                    SELECTOR_SWITCH_FLOOR, 0.5)
    return PreparedComponent(
        component_index=index,
        named_alleles=np.ascontiguousarray(states.named_alleles[:, chosen]),
        bin_sizes=bin_sizes,
        # Classes called anywhere in the whole component (before marker subsampling) are anchored.
        anchored_states=np.concatenate((np.any(states.named_alleles >= 0, axis=1), [False])),
        transition=transition, factors=factors,
        robustness_epsilon=float(painting_config.robustness_epsilon),
        selector_switch_probability=np.ascontiguousarray(theta),
        measured_morgans=np.ascontiguousarray(
            np.diff(measured_coordinates(chromosome_map, bins.centers, rate_is_measured))),
        genotype_likelihoods=gl, observed=observed,
        state_classes=np.ascontiguousarray(np.transpose(state_classes, (0, 2, 1))),
        information_exponent=np.empty((gl.shape[0], 0)))


def prepare_chromosome(painting, evidence, chromosome_map, config: PreparationConfig,
                       claim_threads=lambda: None, rate_is_measured: bool = False) -> PreparedChromosome:
    """Prepared components of one contig from its painting (ChromosomePainting) and evidence
    (ContigEvidence, all samples), with the chromosome-wide information exponents. ``rate_is_measured``: see
    measured_coordinates."""
    prepared = []
    for index, painted in enumerate(painting.components):
        if painted.states.n_sites == 0 or painted.bins.n_bins == 0:
            continue
        claim_threads()
        component = _prepare_component(index, painted, evidence, painting.config, chromosome_map,
                                       config, claim_threads, rate_is_measured)
        if component is not None:
            prepared.append(component)
    if not prepared:
        return PreparedChromosome(painting.contig, len(painting.components), (), OMITTED)
    claim_threads()
    exponent = _information_exponent(
        np.ascontiguousarray(np.concatenate([c.genotype_likelihoods for c in prepared], axis=1)),
        np.concatenate([c.bin_sizes for c in prepared]), config.markers_per_information_block,
        float(config.effective_markers_per_information_block))
    components = []
    offset = 0
    for component in prepared:
        components.append(replace(component, information_exponent=np.ascontiguousarray(
            exponent[:, offset:offset + component.n_bins])))
        offset += component.n_bins
    return PreparedChromosome(painting.contig, len(painting.components), tuple(components))
