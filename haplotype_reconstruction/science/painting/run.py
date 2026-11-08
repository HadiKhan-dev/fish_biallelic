"""Painting stage: label continuity, then the released painting of every component.

For one chromosome, each assembled component's copying model is first estimated from all its samples (the
ancestry-change rate and the panel-error chain, painting.fidelity.estimate_chain; release 2.14); the component is
then relabelled by label continuity (painting/label_continuity.py) under that model, and painted: every sample is
decoded as a diploid mosaic of the component's trajectory classes plus BACKGROUND, each track copying faithfully or
through a panel-error run, with exact Viterbi and forward-backward recurrences, and the release rule
(painting/release.py) turns the decoding into the typed product. Components are independent HMMs (an unconditional
reset at every assembly break). Painting reads every sample's evidence, including samples the sample-quality QC
keeps out of founder construction.

Explicit sample batches bound the chain workspace (about 40 S'^2 B + 16 S^2 B + 24 L bytes per sample).
The stage supplies whole-bin evidence windows and the thread team. Batches do not change a sample's arithmetic;
L2a fixes estimator reductions across samples. A dimension-only 1 GiB cache is consumed in component order.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np

from ...contracts.genetic_map import ChromosomeGeneticMap
from ...contracts.painting import ComponentPainting, LabelContinuity, Painting, PaintingSettings
from ...contracts.panels import ComponentPanel, relabel_rows
from ...contracts.payload import from_payload, to_payload
from ...contracts.roles import freeze_record_arrays, record_field
from ...core import ties
from ...core.errors import InvariantError
from . import emissions, fidelity, hmm
from .emissions import ArraySource, EmissionBuilder
from .label_continuity import relabel_component
from .parameters import EMISSION_CACHE_MAX_BYTES, MAXIMUM_MASK_STATES, PaintingParameters
from .release import release_tracks, track_members
from .states import PaintingBins, PaintingStates, build_bins, build_states, local_allele_keys
from .transitions import PaintingTransitions, copying_transitions, edge_morgans


class PaintingProgress(Protocol):
    """Checkpoint records of one painting derivation (the stage binds them to the store): ``load`` returns a saved
    payload or None, ``save`` publishes one atomically. Keys are relative to the painting."""

    def load(self, key: str) -> Any | None: ...

    def save(self, key: str, payload: Any) -> None: ...


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
        conventions; release 2.2);
    switch_counts: int64[N] minimum track changes (the fewer of the two conventions' paths);
    qf, qaf: float64[N, B] the same masses with both tracks copying faithfully (no panel error): the release gate
        (release 2.14, see painting/release.py);
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


# Samples per decode progress record (fixed: progress records must not depend on the CPU count).
PROGRESS_BLOCK_SAMPLES = 16


def _raw_batches(states, bins, likelihoods, observed, sites, config, size, emission_builder, skip=()):
    """Generator factory: for each contiguous batch of ``size`` samples, (tempered label emissions float64[n, S*S, B],
    eligible bool[n], start, stop, informative mask, gathered observed). Batches whose start is in ``skip`` (restored
    from progress records) are not built."""
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


def _decoder_bytes(states: PaintingStates, n_bins: int) -> int:
    """Per-sample working bytes of the chain's estimation and decoding (module docstring)."""
    n_track = states.n_states + int(np.sum(states.anchored))
    return max(
        1, 40 * n_track * n_track * n_bins + 16 * states.n_states * states.n_states * n_bins + 24 * states.n_sites
    )


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
    emission_cache_bytes: int = EMISSION_CACHE_MAX_BYTES,
    emission_builder: EmissionBuilder = emissions.tempered_emissions,
) -> fidelity.ChainModel | None:
    """The component's copying model (rho, a, b; painting.fidelity.estimate_chain), estimated from all its samples on
    the assembled panel, or None without bins. Label continuity then uses it unchanged: a relabelling renames states
    but leaves their emissions and error capabilities.

    The tempered emissions do not depend on (rho, a, b), so every EM update reads the same batches. When their
    packed upper triangles fit ``emission_cache_bytes`` they are built once and unpacked for each update (the same
    values: the ordered-pair emissions are symmetric), otherwise rebuilt for each update. The cache is released
    before label continuity, so it never coexists with decode_component's packed emissions."""
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
    if 8 * n_samples * n_packed * bins.n_bins <= emission_cache_bytes:
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


def decode_component(
    panel: ComponentPanel,
    likelihoods: ArraySource,
    observed: ArraySource,
    evidence_sites: np.ndarray,
    config: PaintingParameters,
    genetic_map: ChromosomeGeneticMap,
    *,
    model: fidelity.ChainModel | None,
    emission_cache_bytes: int = EMISSION_CACHE_MAX_BYTES,
    batch_samples: int = 1,
    reverse: bool,
    emission_builder: EmissionBuilder = emissions.tempered_emissions,
    progress: PaintingProgress | None = None,
    progress_key: str = "",
) -> ComponentDecoding:
    """Decode every sample of one component (all samples of ``likelihoods``).

    ``progress``: optional records of finished sample batches under ``progress_key`` (the stage's checkpoint store,
    bound to this painting's derivation). A restored batch is not decoded again; the packed-emission cache is then
    left out (a cache, never a result), so later stages rebuild those emissions.

    ``evidence_sites`` int64[L] maps the panel's sites to the evidence site axis (``panel.site_indices()``).
    ``model``: the component's copying model (component_model; required for nonempty bins). The packed tempered label
    emissions are kept when they fit ``emission_cache_bytes``. ``reverse``: the run's tie-convention reversal
    (TieSettings.reverse), required so that no caller falls back to the forward convention by omission.
    """
    n_samples = likelihoods.shape[0]
    states = build_states(panel.positions, panel.inference, panel.kept)
    bins = build_bins(states.positions, config.snps_per_bin)
    n_states = states.n_states
    if n_states > MAXIMUM_MASK_STATES:
        raise InvariantError(f"painting interval masks require S <= {MAXIMUM_MASK_STATES}; got {n_states}")
    n_bins = bins.n_bins
    if n_bins == 0:
        empty_grid = np.zeros((n_samples, 2, 0), dtype=np.int32)
        no_edges = np.zeros(0, dtype=np.float64)
        return ComponentDecoding(
            states=states,
            bins=bins,
            transitions=PaintingTransitions(
                n_states,
                no_edges,
                no_edges.copy(),
                no_edges.copy(),
                site_correlation=float(config.emission_site_correlation),
            ),
            local_keys=np.zeros((0, n_states), dtype=np.int32),
            eligible=np.zeros(n_samples, dtype=np.bool_),
            paths=empty_grid,
            masks=np.zeros((n_samples, 2, 0), dtype=np.int64),
            switch_counts=np.zeros(n_samples, dtype=np.int64),
            support=np.zeros((n_samples, 2, 0), dtype=np.bool_),
            packed_emissions=None,
            batch_size=0,
            working_memory_bytes=0,
            qf=np.zeros((n_samples, 0)),
            qaf=np.zeros((n_samples, 0)),
            model=None,
        )
    if model is None:
        raise InvariantError("a nonempty painting component requires its assembled-panel model")
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

    n_packed = n_states * (n_states + 1) // 2
    packed_bytes = 8 * n_samples * n_packed * n_bins
    fixed_bytes = packed_bytes if packed_bytes <= emission_cache_bytes else 0
    bytes_per_sample = _decoder_bytes(states, n_bins)
    packed = np.empty((n_samples, n_packed, n_bins)) if fixed_bytes else None
    size = max(1, min(n_samples, int(batch_samples)))
    budget = fixed_bytes + size * bytes_per_sample

    paths = np.empty((n_samples, 2, n_bins), dtype=np.int32)
    masks = np.empty((n_samples, 2, n_bins), dtype=np.int64)
    switch_counts = np.zeros(n_samples, dtype=np.int64)
    qf = np.empty((n_samples, n_bins))
    qaf = np.empty((n_samples, n_bins))
    support = np.empty((n_samples, 2, n_bins), dtype=np.bool_)
    eligible = np.empty(n_samples, dtype=np.bool_)
    # Progress records hold fixed blocks of PROGRESS_BLOCK_SAMPLES samples, so their keys and contents do not depend on
    # the CPU-sized decode batch: a run and its shell workflow publish the same records, and a resume on another node
    # reuses every block it finds. A batch is skipped when all its samples were restored.
    done = np.zeros(n_samples, dtype=np.bool_)
    saved_blocks: set[int] = set()
    restored = set()
    if progress is not None:
        for block in range(0, n_samples, PROGRESS_BLOCK_SAMPLES):
            saved = progress.load(f"{progress_key}/decode/{block}")
            if saved is None:
                continue
            end = min(block + PROGRESS_BLOCK_SAMPLES, n_samples)
            paths[block:end], masks[block:end] = saved["paths"], saved["masks"]
            switch_counts[block:end], qf[block:end], qaf[block:end] = saved["switches"], saved["qf"], saved["qaf"]
            support[block:end], eligible[block:end] = saved["support"], saved["eligible"]
            done[block:end] = True
            saved_blocks.add(block)
        restored = {start for start in range(0, n_samples, size) if done[start : min(start + size, n_samples)].all()}
    if done.any():
        packed = None
    for raw, batch_eligible, start, stop, informative, batch_observed in _raw_batches(
        states, bins, likelihoods, observed, sites, config, size, emission_builder, skip=restored
    )():
        if packed is not None:
            emissions.pack_emissions(raw, packed, start, n_states)
        # Switch intervals (release 2.2) over both traceback conventions, the canonical track order (release
        # 2.14) and the release masses as before, of the chain's Viterbi labels.
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
        qf[start:stop], qaf[start:stop] = fidelity.release_posteriors(
            raw, model, capable, batch_paths, other_paths, public_ids, local_keys
        )
        del raw
        batch_support = hmm.direct_support(
            batch_paths, states.named_alleles, informative, bins.starts, bins.stops, states.n_classes
        )
        batch_support[~batch_eligible] = False
        paths[start:stop] = batch_paths
        masks[start:stop] = batch_masks
        switch_counts[start:stop] = batch_switches
        support[start:stop] = batch_support
        eligible[start:stop] = batch_eligible
        done[start:stop] = True
        if progress is not None:
            first_block = start - start % PROGRESS_BLOCK_SAMPLES
            for block in range(first_block, stop, PROGRESS_BLOCK_SAMPLES):
                end = min(block + PROGRESS_BLOCK_SAMPLES, n_samples)
                if block in saved_blocks or not done[block:end].all():
                    continue
                progress.save(
                    f"{progress_key}/decode/{block}",
                    dict(
                        paths=paths[block:end],
                        masks=masks[block:end],
                        switches=switch_counts[block:end],
                        qf=qf[block:end],
                        qaf=qaf[block:end],
                        support=support[block:end],
                        eligible=eligible[block:end],
                    ),
                )
                saved_blocks.add(block)
    return ComponentDecoding(
        states=states,
        bins=bins,
        transitions=transitions,
        local_keys=local_keys,
        eligible=eligible,
        paths=paths,
        masks=masks,
        switch_counts=switch_counts,
        support=support,
        packed_emissions=packed,
        batch_size=size,
        working_memory_bytes=budget,
        qf=qf,
        qaf=qaf,
        model=model,
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
        # Continuous release thresholds are not banded (design A-7, owner 15:50): posteriors within 1e-5 of tau go
        # to the valley census so a crossing is reported with its distance.
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
    # The release gate is the faithful-copy mass (release 2.14): the pair's rows carry the sample's alleles here.
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


def paint_chromosome(
    panels: tuple[ComponentPanel, ...],
    evidence: PaintingEvidence,
    config: PaintingParameters,
    genetic_map: ChromosomeGeneticMap,
    *,
    batch_samples: int = 1,
    reverse: bool,
    emission_cache_bytes: int = EMISSION_CACHE_MAX_BYTES,
    census_sink: Callable[[ties.TieCensus], None] | None = None,
    emission_builder: EmissionBuilder = emissions.tempered_emissions,
    model_sink: Callable[[int, fidelity.ChainModel | None], None] | None = None,
    progress: PaintingProgress | None = None,
) -> Painting:
    """Label continuity, then the painting of every component of one chromosome.

    ``panels``: the final assembly's ComponentPanels in genomic order;
    ``evidence``: the contig's evidence (all samples); ``genetic_map``: the
    contig's ChromosomeGeneticMap; ``reverse``: the run's tie-convention reversal (TieSettings.reverse), required.
    Numba kernels use the caller's thread team.
    """
    likelihoods = evidence.likelihoods
    observed = evidence.observed
    cache_remaining = max(0, int(emission_cache_bytes))
    components: list[ComponentPainting] = []
    continuity: list[LabelContinuity] = []
    for component_index, panel in enumerate(panels):
        sites = panel.site_indices()
        # Progress records (resume after an interruption): the copying model, the label-continuity outcome and each
        # decoded sample batch, each published as soon as it is final and bound to this painting's derivation.
        saved = None if progress is None else progress.load(f"c{component_index}/model")
        if saved is not None:
            model = None if saved["model"] is None else from_payload(fidelity.ChainModel, saved["model"])
        else:
            model = component_model(
                panel,
                likelihoods,
                observed,
                sites,
                config,
                genetic_map,
                batch_samples=batch_samples,
                emission_cache_bytes=cache_remaining,
                emission_builder=emission_builder,
            )
            if progress is not None:
                progress.save(f"c{component_index}/model", dict(model=None if model is None else to_payload(model)))
        if model_sink is not None:
            model_sink(component_index, model)
        saved = None if progress is None else progress.load(f"c{component_index}/continuity")
        if saved is not None:
            outcome = from_payload(LabelContinuity, saved["outcome"])
            painted = relabel_rows(panel, outcome.row_maps) if outcome.accepted else panel
        else:
            painted, outcome = relabel_component(
                panel,
                likelihoods,
                observed,
                sites,
                config,
                genetic_map,
                model=model,
                batch_samples=batch_samples,
                reverse=reverse,
                census_sink=census_sink,
                emission_builder=emission_builder,
                emission_cache_bytes=cache_remaining,
            )
            if progress is not None:
                progress.save(f"c{component_index}/continuity", dict(outcome=to_payload(outcome)))
        decoding = decode_component(
            painted,
            likelihoods,
            observed,
            sites,
            config,
            genetic_map,
            model=model,
            emission_cache_bytes=cache_remaining,
            batch_samples=batch_samples,
            reverse=reverse,
            emission_builder=emission_builder,
            progress=progress,
            progress_key=f"c{component_index}",
        )
        if decoding.packed_emissions is not None:
            cache_remaining -= decoding.packed_emissions.nbytes
        components.append(release_component(painted, decoding, config, census_sink=census_sink))
        continuity.append(outcome)
        # Release-only paths, masks and posterior gates end here, before the next component.
        del decoding
    return Painting(
        contig=evidence.contig,
        sample_ids=tuple(evidence.sample_ids),
        components=tuple(components),
        label_continuity=tuple(continuity),
        parameters=PaintingSettings(**asdict(config)),
    )
