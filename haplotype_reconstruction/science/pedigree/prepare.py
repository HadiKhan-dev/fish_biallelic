"""Painting + raw evidence -> transmission-scoring inputs of one chromosome (PreparedChromosome).

Per painted component: every sample's source laws as a candidate parent (science.pedigree.read_sources: the
painting's emissions at its path strength with its read state, and its own normalised transition table), at most
``markers_per_bin`` evenly spaced selected markers per painting bin with their genotype likelihoods, the painted state
classes (junction counts), and the meiotic switch probability and the measured map length between bins.

Method: docs/methods/pedigree.md#preparation-markers-and-source-factors
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numba
import numpy as np
from numba import prange

from haplotype_reconstruction.science.painting.parameters import READ_STATE_BATCH_BYTES, PaintingParameters

from ...contracts.evidence import ContigEvidence
from ...contracts.genetic_map import ChromosomeGeneticMap
from ...contracts.likelihoods import gather_likelihoods, likelihood_window_sites
from ...contracts.painting import ComponentPainting, Painting, TrackStatus
from ...contracts.pedigree import ChildReads, HammingTransition, PreparedChromosome, PreparedComponent, SourceLaws
from ...core.errors import InputError
from ...core.kernels import kernel
from ..painting import api as painting_hmm
from ..shared.observation import mix_likelihoods
from ..shared.recombination import haldane_fraction
from . import child_reads, projection, read_sources
from .kernels.preparation import OVERLOADS
from .read_gaps import foreign_reads
from .numerical_parameters import INFORMATIVE_GL_SPREAD, SELECTOR_SWITCH_FLOOR
from .scoring_parameters import PreparationParameters
from .workspace_parameters import EMISSION_SAMPLE_BLOCK

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


def _evidence_sites(evidence_positions: np.ndarray, positions: np.ndarray, *, contig: str = "unknown") -> np.ndarray:
    indices = np.searchsorted(evidence_positions, positions)
    present = indices < len(evidence_positions)
    if len(evidence_positions):
        present &= evidence_positions[np.minimum(indices, len(evidence_positions) - 1)] == positions
    if not np.all(present):
        missing = int(positions[np.flatnonzero(~present)[0]])
        raise InputError(f"painted sites are missing from the evidence record; contig={contig}, position={missing}")
    return indices.astype(np.int64)


def path_correlation(painting: Painting) -> float:
    """r of the painting's PATH strength, (path_divisor - 1) / (reference_snps_per_bin - 1), the strength at which the
    pedigree reads the painting's evidence (painting.calibration.correlation). The pedigree asks which painted labels
    a sample shares with a candidate relative, a question about the decoded path; the confidence strength (the
    published site_correlation) weighs the painting's release posteriors and family phase. Read at the confidence
    strength (up to 128 on crosses without sequenced parents), the evidence that tells a full sib from a parent is
    divided away: the release matrix of 4.20 released two full sibs as parent and child."""
    return (float(painting.path_divisor) - 1.0) / (int(painting.parameters.reference_snps_per_bin) - 1)


def scalar_emissions(
    likelihoods,
    observed,
    sites: np.ndarray,
    samples: slice,
    alt_probability: np.ndarray,
    background_frequency: np.ndarray,
    bin_starts: np.ndarray,
    bin_stops: np.ndarray,
    config: PaintingParameters,
    *,
    cohort: np.ndarray,
    robustness_epsilon: float,
    site_correlation: float,
    parts=None,
    columns=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The source chain's tempered label emissions of one sample batch, an emission builder (painting.emissions.
    EmissionBuilder; ``cohort`` float64 [L, 3] the contig's cohort genotype law, bound by the caller): the painting's
    binned log emissions of the mixed window divided by the reference window's design effect at ``site_correlation``,
    with the observed flags and informative sites. The pedigree stage wraps it in the painting's read-state builder
    (stages.painting.read_state.stage_builder) when the painting carries the read state."""
    if parts is not None or columns is not None:
        raise InputError("the pedigree's source emissions take whole bins of contig columns")
    values, seen = painting_hmm.gather_sites(likelihoods, observed, sites, samples)
    values = painting_hmm.mix_window(values, np.asarray(cohort)[sites], robustness_epsilon)
    emissions = painting_hmm.binned_emissions(
        values, seen, alt_probability, background_frequency, bin_starts, bin_stops, log_floor=config.emission_log_floor
    )
    emissions /= painting_hmm.design_effect(
        values, seen, bin_starts, site_correlation, config.reference_snps_per_bin // config.snps_per_bin
    )[:, None, :]
    return emissions, seen, painting_hmm.informative_sites(values, seen)


def _site_parts(
    read_builder, evidence: ContigEvidence, sites: np.ndarray, samples: slice, informative: np.ndarray, flags,
    painted: ComponentPainting, painting_config: PaintingParameters, robustness_epsilon: float, site_correlation: float,
) -> read_sources.SiteParts:
    """The batch's per-site read-state parts at the painting's selected sites (the within-bin class posteriors of
    BACKGROUND's transmitted allele): the builder's label-independent parts, the raw genotype likelihoods and observed
    flags, and the bins' tempering weights at the path strength (the read builder's own)."""
    deep, composed, laws, class_lik = read_builder.site_parts(sites, samples)
    gl, seen = painting_hmm.gather_sites(evidence.likelihoods, evidence.observed, sites, samples)
    states, bins = painted.states, painted.bins
    group = max(1, int(painting_config.reference_snps_per_bin) // int(painting_config.snps_per_bin))
    return read_sources.SiteParts(
        deep=deep, composed=composed, laws=laws, class_lik=class_lik, gl=gl, seen=seen & flags,
        cohort=np.asarray(evidence.cohort_frequencies)[sites],
        alt=np.vstack([states.alt_probability(), np.asarray(states.background_frequency)[None, :]]),
        weight=painting_hmm.design_weights(informative, bins.starts, site_correlation, group),
        site_start=np.asarray(bins.starts, dtype=np.int64), site_stop=np.asarray(bins.stops, dtype=np.int64),
        epsilon=float(robustness_epsilon),
        foreign=foreign_reads(read_builder, sites, samples),
    )


def _source_laws(
    painted: ComponentPainting,
    evidence: ContigEvidence,
    sites: np.ndarray,
    markers: read_sources.MarkerInputs,
    painting_config: PaintingParameters,
    robustness_epsilon: float,
    transition: HammingTransition,
    selector: np.ndarray,
    available: np.ndarray,
    claim_threads: Callable[[], None],
    component_index: int,
    site_correlation: float,
    read_builder: Callable | None = None,
) -> tuple[SourceLaws, ChildReads | None]:
    """Every sample's source laws and, with the read state, its read state as a child (child_reads), one sample
    batch's emissions, read state and GL live at a time.

    Each candidate's chain is independent. Without a read builder (the painting has no read state) the emissions are
    the painting's packed cache when it is at the requested strength, else rebuilt (scalar_emissions); with one, the
    read builder serves the emissions, the read matrices and the per-site parts of the batch, from which the batch's
    child read terms come too (the same parts at the selected markers)."""
    states, bins = painted.states, painted.bins
    samples, n_states, n_bins = evidence.n_samples, states.n_states, bins.n_bins
    features = n_states * (n_states + 1) // 2
    pair_mass = np.empty((samples, n_bins, features), dtype=np.float64)
    pi = np.empty((samples, n_bins, n_states), dtype=np.float64)
    flows = {name: np.empty((samples, max(0, n_bins - 1), n_states), dtype=np.float64)
             for name in ("diagonal_flow", "outgoing_flow", "incoming_flow")}
    background_alt = np.empty((samples, len(markers.site)), dtype=np.float64)
    reads = []
    # the painting's packed emissions are at its published (confidence) strength: used only when that is the
    # requested strength and no read state is carried, otherwise rebuilt at ``site_correlation`` (path_correlation)
    packed = (
        painted.packed_emissions
        if read_builder is None and site_correlation == painted.transitions.site_correlation else None
    )
    alt_probability = states.alt_probability()
    step = _EMISSION_SAMPLE_BLOCK
    split_bypass = (read_builder is not None and not read_builder.foreign_read_share
                    and read_builder.unit_batches and bool(read_builder.unshared.any()))
    if read_builder is not None:
        # one batch's working set: the builder's (read matrices, parts and their transient inputs) plus the source
        # recurrence's backward vectors and entering vectors and the per-site parts it copies
        classes = int(painting_config.read_classes)
        width = classes + n_states if read_builder.foreign_read_share else classes
        recurrence = 8 * n_bins * width * (n_states * n_states + n_states) + 8 * len(sites) * (
            classes * classes + 5 * classes + 4)
        per_sample = read_builder.sample_bytes(sites, n_states, n_bins, False) + recurrence
        # at least one sample per thread: the read parts' per-batch passes over the sites (operator composition) and
        # the chain kernels (parallel over samples) amortise over the batch; values do not depend on the batch
        requested = max(_EMISSION_SAMPLE_BLOCK, numba.get_num_threads())
        step = max(1, min(requested, READ_STATE_BATCH_BYTES // per_sample))
    for first in range(0, samples, step):
        last = min(samples, first + step)
        batch = slice(first, last)
        claim_threads()
        read = site_parts = None
        if packed is not None:
            emissions = painting_hmm.unpack_emissions(packed[first:last], n_states)
        elif read_builder is None:
            emissions = scalar_emissions(
                evidence.likelihoods, evidence.observed, sites, batch, alt_probability, states.background_frequency,
                bins.starts, bins.stops, painting_config, cohort=evidence.cohort_frequencies,
                robustness_epsilon=robustness_epsilon, site_correlation=site_correlation,
            )[0]
        else:
            built, flags, informative = read_builder(
                evidence.likelihoods, evidence.observed, sites, batch, alt_probability, states.background_frequency,
                bins.starts, bins.stops, painting_config, robustness_epsilon=robustness_epsilon,
                site_correlation=site_correlation,
            )
            emissions = built.raw
            if built.initial.shape[1] == 1:
                reads.append(child_reads.scalar_batch(last - first))
            else:
                read = built
                site_parts = _site_parts(
                    read_builder, evidence, sites, batch, informative, flags, painted, painting_config,
                    robustness_epsilon, site_correlation,
                )
                reads.append(child_reads.batch_reads(site_parts, markers.site, markers.observed[first:last]))
            del built, flags, informative
        compute = read_sources.mixed_batch_laws if split_bypass and read is not None else read_sources.batch_laws
        laws = compute(
            emissions, transition, selector, available[first:last],
            replace(markers, gl=markers.gl[first:last], observed=markers.observed[first:last]),
            read=read, sites=site_parts, component_index=component_index, sample_offset=first,
        )
        pair_mass[first:last] = laws["pair_mass"]
        pi[first:last] = laws["pi"]
        for name, array in flows.items():
            array[first:last] = laws[name]
        background_alt[first:last] = laws["background_alt"]
        del laws, emissions, read, site_parts
    laws = SourceLaws(pair_mass, pi, flows["diagonal_flow"], flows["outgoing_flow"], flows["incoming_flow"],
                      background_alt, available)
    if not reads:
        return laws, None
    width = 2 * int(read_builder.minor) + 3
    if read_builder.foreign_read_share:
        width += n_states
    return laws, child_reads.join_reads(reads, width)


def _prepare_component(
    index: int,
    painted: ComponentPainting,
    evidence: ContigEvidence,
    painting_config: PaintingParameters,
    robustness_epsilon: float,
    chromosome_map: ChromosomeGeneticMap,
    config: PreparationParameters,
    claim_threads: Callable[[], None],
    *,
    contig: str = "unknown",
    site_correlation: float,
    read_builder: Callable | None = None,
) -> PreparedComponent | None:
    """One component's scoring inputs, or None without observed non-uniform selected evidence."""
    states, bins = painted.states, painted.bins
    sites = _evidence_sites(evidence.positions, states.positions, contig=contig)
    chosen, bin_sizes = selected_markers(states.positions, bins.edges, config.markers_per_bin)
    if not len(chosen):
        return None
    marker_sites = sites[chosen]
    # The children's and candidates' mixed likelihoods at the painting's outlier rate (science.shared.observation).
    gl = mix_likelihoods(
        gather_likelihoods(evidence.likelihoods, marker_sites), evidence.cohort_frequencies[marker_sites],
        robustness_epsilon,
    )
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
    # Which parental homologue is transmitted changes between adjacent bin centres when an odd number of
    # crossovers falls between them: one meiosis's Haldane recombination fraction, kept within [floor, 1/2].
    theta = np.clip(
        haldane_fraction(chromosome_map.interval_morgans(bins.centers[:-1], bins.centers[1:])),
        SELECTOR_SWITCH_FLOOR,
        0.5,
    )
    named_alleles = np.ascontiguousarray(states.named_alleles[:, chosen])
    markers = read_sources.MarkerInputs(
        state_alt=projection.marker_state_alt(named_alleles), gl=gl, observed=observed,
        site=np.asarray(chosen, dtype=np.int64), bin_sizes=bin_sizes,
    )
    laws, reads = _source_laws(
        painted, evidence, sites, markers, painting_config, robustness_epsilon, transition, theta, available,
        claim_threads, index, site_correlation, read_builder,
    )
    unknown = (painted.status == _BACKGROUND_STATUS) | (painted.status == _UNANCHORED_STATUS)
    state_classes = np.where(unknown, -1, painted.classes).astype(np.int16)
    return PreparedComponent(
        component_index=index,
        named_alleles=named_alleles,
        bin_sizes=bin_sizes,
        # Classes called anywhere in the whole component (before marker subsampling) are anchored.
        anchored_states=np.concatenate((np.any(states.named_alleles >= 0, axis=1), [False])),
        transition=transition,
        sources=laws,
        robustness_epsilon=float(robustness_epsilon),
        selector_switch_probability=np.ascontiguousarray(theta),
        genotype_likelihoods=gl,
        observed=observed,
        state_classes=np.ascontiguousarray(np.transpose(state_classes, (0, 2, 1))),
        selector_fallback_bp=np.ascontiguousarray(
            chromosome_map.fallback_bp(bins.centers[:-1], bins.centers[1:]), dtype=np.float64
        ),
        # Per-bin label-error allowances of A9 and A10 (ADVICE2 11.1), tracks in the order of state_classes.
        label_error=(
            None if painted.label_error is None else np.ascontiguousarray(painted.label_error, dtype=np.float32)
        ),
        child_reads=reads,
    )


def prepare_chromosome(
    painting: Painting,
    evidence: ContigEvidence,
    chromosome_map: ChromosomeGeneticMap,
    config: PreparationParameters,
    claim_threads: Callable[[], None] = lambda: None,
    *,
    read_builder: Callable | None = None,
) -> PreparedChromosome:
    """Prepared components of one contig from its painting (ChromosomePainting) and evidence
    (ContigEvidence, all samples), and the contig's configured
    material (configured_morgans: m(L) - m(0) of ChromosomeGeneticMap.cumulative_morgans, L its last evidence
    position: the configured map and its fallback-rate tails, fixed before any genotype is read). ``read_builder``:
    the painting's read-state emission builder over scalar_emissions (painting.read_inputs.ReadEmissionBuilder, made
    by the stage when the painting carries the read state; None without it: one read class)."""
    if read_builder is None and int(painting.parameters.read_classes) > 1:
        # never a silent single class: the painting's read state needs its builder (a scalar control states
        # read_classes = 1 on the painting it passes)
        raise InputError(f"{painting.contig}: the painting carries the read state (read_classes "
                         f"{painting.parameters.read_classes}); the source chain needs its read builder")
    configured = configured_morgans(chromosome_map, evidence.positions)
    site_correlation = path_correlation(painting)
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
            painting.robustness_epsilon,
            chromosome_map,
            config,
            claim_threads,
            contig=painting.contig,
            site_correlation=site_correlation,
            read_builder=read_builder,
        )
        if component is not None:
            prepared.append(component)
    return PreparedChromosome(
        painting.contig, len(painting.components), tuple(prepared), None if prepared else OMITTED, configured,
        search_alleles_reversed=evidence.search_alleles_reversed,
    )


def configured_morgans(chromosome_map: ChromosomeGeneticMap, positions: np.ndarray) -> float:
    """Working-map Morgans of [0, L] for a contig whose last evidence position is L (0 without evidence):
    m(L) - m(0), m = ChromosomeGeneticMap.cumulative_morgans (DESIGN_417 rev 4, S10). The VCF header length is
    not recorded in any checkpoint the pedigree stage reads, so L is always the last evidence position (the
    rule's fallback); it depends on the site set, not on any dyad's genotypes."""
    if not len(positions):
        return 0.0
    last = float(np.max(positions))
    return float(chromosome_map.cumulative_morgans(last) - chromosome_map.cumulative_morgans(0.0))
