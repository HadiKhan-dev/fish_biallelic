"""Painting stage: label continuity, then the released painting of every component.

For one chromosome, each assembled component's copying model is first estimated from all its samples (the
ancestry-change rate and the panel-error chain, painting.fidelity.estimate_chain); the component is
then relabelled by label continuity (painting/label_continuity.py) under that model, and painted: every sample is
decoded as a diploid mosaic of the component's trajectory classes plus BACKGROUND, each track copying faithfully or
through a panel-error run, with exact Viterbi and forward-backward recurrences, and the release rule
(painting/release.py) turns the decoding into the typed product. Components are independent HMMs (an unconditional
reset at every assembly break). Painting reads every sample's evidence, including samples the sample-quality QC
keeps out of founder construction.

Explicit sample batches bound the chain workspace (about 40 S'^2 B + 16 S^2 B + 24 L bytes per sample).
The stage supplies whole-bin evidence windows and the thread team. Batches do not change a sample's arithmetic.
Estimator reductions follow sample order across batches. The stage controls emission-cache residency.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ...contracts.genetic_map import ChromosomeGeneticMap
from ...contracts.painting import ComponentPainting
from ...contracts.panels import ComponentPanel
from ...contracts.roles import freeze_record_arrays, record_field
from ...core import ties
from ...core.errors import InvariantError
from . import emissions, fidelity, hmm
from .emissions import ArraySource, EmissionBuilder
from .parameters import MAXIMUM_MASK_STATES, PaintingParameters
from .release import release_tracks, track_members
from .states import PaintingBins, PaintingStates, build_bins, build_states, local_allele_keys
from .transitions import PaintingTransitions, copying_transitions, edge_morgans


class PaintingEvidence(Protocol):
    """All-sample evidence boundary; the stage owns any borrowed source lifetime."""

    @property
    def contig(self) -> str: ...
    @property
    def sample_ids(self) -> tuple[str, ...]: ...
    @property
    def likelihoods(self) -> ArraySource: ...
    @property
    def observed(self) -> ArraySource: ...


@dataclass(frozen=True)
class ComponentDecoding:
    """The HMM decoding of one component before the release rule.

    paths: int32[N, 2, B] gauge-fixed Viterbi state per track and bin;
    masks: int64[N, 2, B] candidate states per track and bin as bits (the Viterbi
        state, plus the other side inside a switch interval, over both traceback
        conventions);
    switch_counts: int64[N] minimum track changes (the fewer of the two conventions' paths);
    qf, qaf: float64[N, B] the same masses with both tracks copying faithfully (no panel error): the release gate;
    model: the component's copying model (painting.fidelity.ChainModel; None without bins);
    support: bool[N, 2, B] direct read support of each Viterbi track
        (False for ineligible samples);
    batch_size, working_memory_bytes: the sample batching used (execution only).
    Other fields as in :class:`painting.release.ComponentPainting`.
    """

    states: PaintingStates = record_field("state", axes=(), dtype="PaintingStates", units="1")
    bins: PaintingBins = record_field("state", axes=(), dtype="PaintingBins", units="1")
    transitions: PaintingTransitions = record_field("state", axes=(), dtype="PaintingTransitions", units="1")
    local_keys: np.ndarray = record_field("state", axes=("B", "S"), dtype="int32", units="1")
    eligible: np.ndarray = record_field("state", axes=("N",), dtype="bool", units="1")
    paths: np.ndarray = record_field("call", axes=("N", "2", "B"), dtype="int32", units="1")
    masks: np.ndarray = record_field("call", axes=("N", "2", "B"), dtype="int64", units="1")
    switch_counts: np.ndarray = record_field("call", axes=("N",), dtype="int64", units="1")
    support: np.ndarray = record_field("call", axes=("N", "2", "B"), dtype="bool", units="1")
    packed_emissions: np.ndarray | None = record_field(
        "diagnostic", axes=("N", "packed_pairs", "B"), dtype="float64", units="nats"
    )
    batch_size: int = record_field("diagnostic", axes=(), dtype="int", units="1")
    working_memory_bytes: int = record_field("diagnostic", axes=(), dtype="int", units="1")
    qf: np.ndarray = record_field("state", axes=("N", "B"), dtype="float64", units="probability")
    qaf: np.ndarray = record_field("state", axes=("N", "B"), dtype="float64", units="probability")
    model: fidelity.ChainModel | None = record_field("state", axes=(), dtype="fidelity.ChainModel | None", units="1")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


def _raw_batches(states, bins, likelihoods, observed, sites, config, size, emission_builder, skip=()):
    """Yield emissions, eligibility and observed masks for the requested contiguous sample batches."""
    alt_probability = states.alt_probability()
    n_samples = likelihoods.shape[0]

    def batches():
        for start in range(0, n_samples, size):
            if start in skip:
                continue
            stop = min(start + size, n_samples)
            raw, batch_observed, informative = emission_builder(
                likelihoods,
                observed,
                sites,
                slice(start, stop),
                alt_probability,
                states.background_frequency,
                bins.starts,
                bins.stops,
                config,
            )
            eligible = np.any(informative, axis=1)
            yield raw, eligible, start, stop, informative, batch_observed

    return batches


def error_capable(states: PaintingStates) -> np.ndarray:
    """int64[S]: 1 for the labels with a panel-error state (anchored classes), 0 for the others and BACKGROUND."""
    return np.concatenate((np.asarray(states.anchored, dtype=np.int64), np.zeros(1, dtype=np.int64)))


def component_model(
    panel: ComponentPanel,
    likelihoods: ArraySource,
    observed: ArraySource,
    evidence_sites: np.ndarray,
    config: PaintingParameters,
    genetic_map: ChromosomeGeneticMap,
    *,
    batch_samples: int = 1,
    cache_emissions: bool = True,
    emission_builder: EmissionBuilder = emissions.tempered_emissions,
) -> fidelity.ChainModel | None:
    """The component's copying model (rho, a, b; painting.fidelity.estimate_chain), estimated from all its samples on
    the assembled panel, or None without bins. Label continuity then uses it unchanged: a relabelling renames states
    but leaves their emissions and error capabilities.

    The tempered emissions do not depend on (rho, a, b), so every EM update reads the same batches. When the stage requests cached
    packed upper triangles they are built once and unpacked for each update (the same
    values: the ordered-pair emissions are symmetric), otherwise rebuilt for each update. The cache is released
    before label continuity, so it never coexists with the decoding cache."""
    n_samples = likelihoods.shape[0]
    states = build_states(panel.positions, panel.inference, panel.kept)
    bins = build_bins(states.positions, config.snps_per_bin)
    if bins.n_bins == 0:
        return None
    sites = evidence_sites[states.site_indices]
    size = max(1, min(n_samples, int(batch_samples)))
    batches = _raw_batches(states, bins, likelihoods, observed, sites, config, size, emission_builder)
    n_states = states.n_states
    n_packed = n_states * (n_states + 1) // 2
    if cache_emissions:
        packed = np.empty((n_samples, n_packed, bins.n_bins))
        spans = []
        for raw, eligible, start, stop, *_ in batches():
            emissions.pack_emissions(raw, packed, start, n_states)
            spans.append((eligible, start, stop))
            del raw

        def source():
            for eligible, start, stop in spans:
                yield emissions.unpack_emissions(packed[start:stop], n_states), eligible

    else:

        def source():
            return ((raw, eligible) for raw, eligible, *_ in batches())

    return fidelity.estimate_chain(
        source,
        error_capable(states),
        edge_morgans(bins.centers, genetic_map),
        config,
    )


def release_component(
    panel: ComponentPanel,
    decoding: ComponentDecoding,
    config: PaintingParameters,
    *,
    census_sink: Callable[[ties.TieCensus], None] | None = None,
) -> ComponentPainting:
    """Apply the release rule once and build the component's product."""
    states = decoding.states
    if ties.census_enabled(census_sink) and decoding.bins.n_bins:
        # Continuous release thresholds are not banded. Posteriors within 1e-5 of tau enter the valley census
        # so a crossing is reported with its distance.
        tau = config.minimum_release_posterior
        first = int(panel.positions[0])
        for name, values in (("painting.release.qf", decoding.qf), ("painting.release.qaf", decoding.qaf)):
            for sample, bin_index in zip(*np.nonzero(np.abs(values - tau) < 1e-5), strict=True):
                ties.census(
                    name,
                    f"p{first}/s{int(sample)}/b{int(bin_index)}",
                    [float(values[sample, bin_index]) - tau],
                    sink=census_sink,
                )
    # The release gate is the faithful-copy mass: the pair's rows carry the sample's alleles here.
    labels, classes, status = release_tracks(
        states,
        decoding.paths,
        decoding.eligible,
        decoding.qf,
        decoding.qaf,
        decoding.local_keys,
        decoding.support,
        config.minimum_release_posterior,
        masks=decoding.masks,
    )
    # A released interval track of a bin below the stage-1 threshold was released at stage 2 (local keys).
    member_ids, member_sets = track_members(
        status,
        classes,
        decoding.local_keys,
        states.classes,
        masks=decoding.masks,
        stage_two=decoding.qf < config.minimum_release_posterior,
        anchored=states.anchored,
    )
    switch_counts = decoding.switch_counts.copy()
    switch_counts[~decoding.eligible] = 0
    return ComponentPainting(
        panel=panel,
        states=states,
        bins=decoding.bins,
        transitions=decoding.transitions,
        local_keys=decoding.local_keys,
        eligible=decoding.eligible,
        labels=labels,
        classes=classes,
        status=status,
        direct=decoding.support & (classes >= 0),
        member_ids=member_ids,
        member_sets=member_sets,
        switch_counts=switch_counts,
        packed_emissions=decoding.packed_emissions,
    )


def decoding_grid(panel, config):
    """Construct the component's state and bin axes before decoding."""
    states = build_states(panel.positions, panel.inference, panel.kept)
    bins = build_bins(states.positions, config.snps_per_bin)
    if states.n_states > MAXIMUM_MASK_STATES:
        raise InvariantError(f"painting interval masks require S <= {MAXIMUM_MASK_STATES}; got {states.n_states}")
    return states, bins


def decoding_model(states, bins, evidence_sites, genetic_map, model):
    """Prepare the fixed copying transitions, local labels and sample-independent inputs."""
    n_states = states.n_states
    capable = error_capable(states)
    transitions = copying_transitions(
        edge_morgans(bins.centers, genetic_map),
        n_states,
        model.rate,
        error_entry=model.entry,
        error_exit=model.exit,
        site_correlation=model.site_correlation,
    )
    local_keys = local_allele_keys(states.named_alleles, bins)
    public_ids = states.public_state_ids()
    sites = evidence_sites[states.site_indices]

    return capable, transitions, local_keys, public_ids, sites


def decode_batch(raw, batch_eligible, batch_observed, informative, states, bins, model, capable,
                 local_keys, public_ids, config, *, reverse):
    """Compute one batch's candidate paths, release masses and direct support."""
    # Combine switch intervals from both traceback conventions, canonical track order and faithful-copy
    # release masses of the chain's Viterbi labels.
    batch_paths, batch_masks, batch_switches, other_paths = fidelity.decode_candidates(
        raw,
        model,
        capable,
        local_keys,
        states.named_alleles,
        batch_observed,
        bins.starts,
        bins.stops,
        states.n_classes,
        reverse=reverse,
        tie=config.viterbi_tie,
    )
    batch_qf, batch_qaf = fidelity.release_posteriors(
        raw, model, capable, batch_paths, other_paths, public_ids, local_keys
    )
    del raw
    batch_support = hmm.direct_support(
        batch_paths, states.named_alleles, informative, bins.starts, bins.stops, states.n_classes
    )
    batch_support[~batch_eligible] = False
    return batch_paths, batch_masks, batch_switches, batch_qf, batch_qaf, batch_support
