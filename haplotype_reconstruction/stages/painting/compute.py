"""Orchestrate painting progress, sample blocks and component cache residency."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import Any, Protocol

import numpy as np

from ...contracts.genetic_map import ChromosomeGeneticMap
from ...contracts.painting import ComponentPainting, LabelContinuity, Painting
from ...contracts.panels import ComponentPanel, relabel_rows
from ...contracts.payload import from_payload, to_payload
from ...core import ties
from ...core.errors import InvariantError
from ...science.painting import emissions, fidelity
from ...science.painting.emissions import ArraySource, EmissionBuilder
from ...science.painting.label_continuity import relabel_component
from ...science.painting.parameters import EMISSION_CACHE_MAX_BYTES, LABEL_ERROR_CORRELATION, PaintingParameters
from ...science.painting.run import (ComponentDecoding, PaintingEvidence, _raw_batches,
                                      component_model, decoding_grid, decoding_model, decode_batch, release_component)
from ...science.painting.states import PaintingStates, build_states, build_bins
from ...science.painting.transitions import PaintingTransitions

class PaintingProgress(Protocol):
    """Checkpoint records of one painting derivation (the stage binds them to the store): ``load`` returns a saved
    payload or None, ``save`` publishes one atomically. Keys are relative to the painting."""

    def load(self, key: str) -> Any | None: ...

    def save(self, key: str, payload: Any) -> None: ...


# Sample blocks are fixed across CPU-sized batches so another node can reuse every complete block.
PROGRESS_BLOCK_SAMPLES = 16


def _packed_bytes(panel, samples, config):
    states, bins = decoding_grid(panel, config)
    return 8 * samples * (states.n_states * (states.n_states + 1) // 2) * bins.n_bins


def _read_state(builder: Any) -> tuple[Any, np.ndarray | None]:
    """(the read-state builder behind ``builder`` (science.painting.read_inputs.ReadEmissionBuilder; None for the
    emissions without read state), the ``columns`` keyword bound around it)."""
    columns = None
    while isinstance(builder, partial):
        columns = builder.keywords.get("columns", columns)
        builder = builder.func
    return (builder if hasattr(builder, "sample_bytes") else None), columns


def _decoder_bytes(states: PaintingStates, n_bins: int) -> int:
    """Per-sample working bytes of the chain's estimation and decoding (module docstring)."""
    n_track = states.n_states + int(np.sum(states.anchored))
    return max(
        1, 40 * n_track * n_track * n_bins + 16 * states.n_states * states.n_states * n_bins + 24 * states.n_sites
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
    robustness_epsilon: float,
    site_correlation: float,
    confidence_model: fidelity.ChainModel | None = None,
    label_model: fidelity.ChainModel | None = None,
    emission_cache_bytes: int = EMISSION_CACHE_MAX_BYTES,
    batch_samples: int = 1,
    reverse: bool,
    emission_builder: EmissionBuilder | None = None,
    progress: PaintingProgress | None = None,
    progress_key: str = "",
    label_errors: bool = False,
) -> ComponentDecoding:
    """Decode every sample of one component (all samples of ``likelihoods``).

    ``progress``: optional records of finished sample batches under ``progress_key`` (the stage's checkpoint store,
    bound to this painting's derivation). A restored batch is not decoded again; the packed-emission cache is then
    left out (a cache, never a result), so later stages rebuild those emissions.

    ``evidence_sites`` int64[L] maps the panel's sites to the evidence site axis (``panel.site_indices()``).
    ``model``: the component's copying model at the path strength ``site_correlation`` (component_model; required for
    nonempty bins): it decodes. ``confidence_model``: None when the run's two strengths are equal; else the copying
    model fitted at the confidence strength (its ``site_correlation``): the release posteriors, the published
    ``PaintingTransitions.site_correlation`` and the packed emission cache (what later stages read) are then at the
    confidence strength. The packed tempered label
    emissions are kept when they fit ``emission_cache_bytes``. ``reverse``: the run's tie-convention reversal
    (TieSettings.reverse), required so that no caller falls back to the forward convention by omission.
    ``label_errors``: also the label-error statistic of the decoded labels (fidelity.label_errors; the reference
    painting, ADVICE2 11.1), read from the chain at the label-error strength (parameters.LABEL_ERROR_CORRELATION, the
    untempered chain): from the release pass when the release chain is at that strength, else from a second pass
    over that strength's emissions (the path emissions rescaled by emissions.confidence_scale, with the read state's
    matrices at its weights) and ``label_model``, the copying model fitted at it (required when the strength differs
    from both run strengths). The labels, their release and the packed emissions do not depend on it.
    """
    n_samples = likelihoods.shape[0]
    states, bins = decoding_grid(panel, config)
    n_states = states.n_states
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
                site_correlation=float(site_correlation),
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
            label_error=np.zeros((n_samples, 0, 3), dtype=np.float32) if label_errors else None,
        )
    if model is None:
        raise InvariantError("a nonempty painting component requires its assembled-panel model")
    capable, transitions, local_keys, public_ids, sites = decoding_model(
        states, bins, evidence_sites, genetic_map, model
    )
    group = config.reference_snps_per_bin // config.snps_per_bin
    if confidence_model is not None:
        transitions = replace(transitions, site_correlation=float(confidence_model.site_correlation))
    # The label-error statistic's chain: a pass of its own when its strength differs from the release chain's, with
    # emissions of its own when it differs from the path strength too.
    release_correlation = float(site_correlation if confidence_model is None else confidence_model.site_correlation)
    label_pass = bool(label_errors) and float(LABEL_ERROR_CORRELATION) != release_correlation
    label_scaled = label_pass and float(LABEL_ERROR_CORRELATION) != float(site_correlation)
    if label_scaled and label_model is None:
        raise InvariantError("the label-error statistic needs the copying model fitted at the label-error strength")

    n_packed = n_states * (n_states + 1) // 2
    packed_bytes = 8 * n_samples * n_packed * n_bins
    fixed_bytes = packed_bytes if packed_bytes <= emission_cache_bytes else 0
    bytes_per_sample = _decoder_bytes(states, n_bins) + (8 * n_states * n_states * n_bins if label_scaled else 0)
    read_state, columns = _read_state(emission_builder)
    if read_state is not None:      # the batch's read matrices, label-independent parts and their transient inputs
        selected = emissions.contig_columns(np.asarray(evidence_sites, dtype=np.int64)[states.site_indices], columns)
        bytes_per_sample += read_state.sample_bytes(
            selected, n_states, n_bins, int(confidence_model is not None) + int(label_scaled)
        )
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
    label_error = np.full((n_samples, n_bins, 3), np.nan, dtype=np.float32) if label_errors else None
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
            if label_error is not None:
                label_error[block:end] = saved["label_error"]
            done[block:end] = True
            saved_blocks.add(block)
        restored = {start for start in range(0, n_samples, size) if done[start : min(start + size, n_samples)].all()}
    if done.any():
        packed = None
    for raw, batch_eligible, start, stop, informative, batch_observed in _raw_batches(
        states, bins, likelihoods, observed, sites, config, size, emission_builder, robustness_epsilon, site_correlation, skip=restored
    )():
        confident = None
        scalar = raw.raw if hasattr(raw, "read") else raw          # the label emissions (read state: the scalar part)
        if confidence_model is not None:
            scaled = scalar * emissions.confidence_scale(
                informative, bins.starts, site_correlation, confidence_model.site_correlation, group
            )[:, None, :]
            confident = raw.at_confidence(scaled) if hasattr(raw, "read") else scaled
        label = None
        if label_pass and not label_scaled:
            label = (model, raw)                    # the label-error strength is the path strength
        elif label_scaled:
            labelled = scalar * emissions.confidence_scale(
                informative, bins.starts, site_correlation, LABEL_ERROR_CORRELATION, group
            )[:, None, :]
            if hasattr(raw, "read"):
                if raw.label is None and raw.read.shape[-1] > 1:
                    raise InvariantError("the emission builder carries no label-error strength (stage_builder)")
                labelled = raw.at_label(labelled)
            label = (label_model, labelled)
        if packed is not None:
            # later stages (pedigree, family phase) read the emissions without read state (the scalar part)
            emissions.pack_emissions(scalar if confident is None else (confident.raw if hasattr(confident, "read")
                                     else confident), packed, start, n_states)
        decoded = decode_batch(
            raw, batch_eligible, batch_observed, informative, states, bins, model, capable,
            local_keys, public_ids, config, reverse=reverse,
            confidence=None if confident is None else (confidence_model, confident), errors=label_errors,
            label=label,
        )
        batch_paths, batch_masks, batch_switches, batch_qf, batch_qaf, batch_support = decoded[:6]
        if label_error is not None:
            label_error[start:stop, :, 0] = decoded[6]
            label_error[start:stop, :, 1:] = np.transpose(decoded[7], (0, 2, 1))
        del confident, label
        qf[start:stop], qaf[start:stop] = batch_qf, batch_qaf
        del raw
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
                        **({} if label_error is None else dict(label_error=label_error[block:end])),
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
        label_error=label_error,
    )


def _component_sources(
    panel: ComponentPanel,
    likelihoods: ArraySource,
    observed: ArraySource,
    sites: np.ndarray,
    config: PaintingParameters,
    budget: int,
) -> tuple[tuple[Any, Any, np.ndarray, np.ndarray | None], int]:
    """((likelihoods, observed, panel-site map, contig columns) to build the component's emissions from, the emission
    budget left).

    When the evidence offers resident copies (resident_columns), the component's evidence at its panel
    sites (each sample's genotype-likelihood codes and its likelihood dictionary, far smaller than the emissions) is
    read once into memory, charged to the budget, whenever the copying-model estimate and the decode keep the packed
    emissions they would keep without it (or could not keep them anyway). Every emission build of the component, the
    estimate's, label continuity's and the decode's, is then a parallel in-memory gather and emission kernel instead
    of a serial read of the stored evidence. The values are the same. The panel-site map then indexes the copy, and
    the contig columns (the copy's column -> contig evidence column; None without a copy) are what contig-wide
    per-site arrays such as the cohort frequencies are read at (emissions.contig_columns)."""
    unchanged = (likelihoods, observed, sites, None), budget
    if not (hasattr(likelihoods, "resident_columns") and hasattr(observed, "resident_columns")):
        return unchanged
    states = build_states(panel.positions, panel.inference, panel.kept)
    bins = build_bins(states.positions, config.snps_per_bin, config.reference_snps_per_bin)
    if bins.n_bins == 0:
        return unchanged
    packed = 8 * likelihoods.shape[0] * (states.n_states * (states.n_states + 1) // 2) * bins.n_bins
    copy_bytes = likelihoods.resident_bytes(len(sites)) + observed.resident_bytes(len(sites))
    if copy_bytes + (packed if packed <= budget else 0) > budget:
        return unchanged
    copy = likelihoods.resident_columns(sites)
    resident = (
        copy,
        observed.resident_columns(sites, copy),
        np.arange(len(sites), dtype=np.int64),
        np.asarray(sites, dtype=np.int64),
    )
    return resident, budget - copy_bytes


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
    emission_builder: EmissionBuilder | None = None,
    model_sink: Callable[[int, fidelity.ChainModel | None], None] | None = None,
    progress: PaintingProgress | None = None,
    label_continuity: tuple[LabelContinuity, ...] | None = None,
    robustness_epsilon: float,
    site_correlation: float,
    confidence_correlation: float | None = None,
    label_errors: bool = False,
) -> Painting:
    """Label continuity, then the painting of every component of one chromosome.

    ``panels``: the final assembly's ComponentPanels in genomic order;
    ``evidence``: the contig's evidence (all samples); ``genetic_map``: the
    contig's ChromosomeGeneticMap; ``reverse``: the run's tie-convention reversal (TieSettings.reverse), required.
    Numba kernels use the caller's thread team. ``site_correlation`` is the run's path strength (model fit, label
    continuity and decoding); ``confidence_correlation`` (None = the same) is the run's confidence strength: a second
    fit of the copying model at that strength gives the release posteriors, and later stages read emissions at it.
    A supplied label_continuity fixes the fine painting's
    row outcome: apply it before fitting this resolution's own chain and do not search again. ``emission_builder``
    (None: resident tempered emissions under the evidence's own cohort law, ``evidence.cohort_frequencies``).
    """
    if emission_builder is None:
        emission_builder = partial(emissions.tempered_emissions, cohort=evidence.cohort_frequencies)
    likelihoods = evidence.likelihoods
    observed = evidence.observed
    epsilon = robustness_epsilon
    cache_remaining = max(0, int(emission_cache_bytes))
    components: list[ComponentPainting] = []
    continuity: list[LabelContinuity] = []
    # The label-error statistic's own copying model (parameters.LABEL_ERROR_CORRELATION, the untempered chain): fitted
    # when the reference painting reads the statistic and its strength differs from both run strengths.
    release_correlation = float(site_correlation if confidence_correlation is None else confidence_correlation)
    label_strength = bool(label_errors) and float(LABEL_ERROR_CORRELATION) not in (
        float(site_correlation), release_correlation
    )
    for component_index, panel in enumerate(panels):
        if label_continuity is not None:
            outcome = label_continuity[component_index]
            panel = relabel_rows(panel, outcome.row_maps) if outcome.accepted else panel
        (component_likelihoods, component_observed, sites, columns), budget = _component_sources(
            panel, likelihoods, observed, panel.site_indices(), config, cache_remaining
        )
        # From a resident copy the builder reads contig-wide per-site arrays at the copy's contig columns.
        builder = emission_builder if columns is None else partial(emission_builder, columns=columns)
        # With the read state, the sample batch holds the read matrices of every label pair and bin: batches are
        # capped so that their read-state working set fits READ_STATE_BATCH_BYTES (read_inputs.ReadEmissionBuilder.
        # batch_samples; results do not depend on the batch), and a copying-model fit holds its batches only when
        # their read matrices fit the emission budget beside the packed emissions.
        component_batch, held_read_bytes = batch_samples, 0
        read_state, _ = _read_state(emission_builder)
        if read_state is not None:
            states, bins = decoding_grid(panel, config)
            if bins.n_bins:
                selected = emissions.contig_columns(np.asarray(sites, dtype=np.int64)[states.site_indices], columns)
                confidence = confidence_correlation is not None and float(confidence_correlation) != float(
                    site_correlation
                )
                component_batch = read_state.batch_samples(
                    selected, states.n_states, bins.n_bins, int(confidence) + int(label_strength), batch_samples
                )
                held_read_bytes = component_likelihoods.shape[0] * read_state.matrix_bytes(
                    states.n_states, bins.n_bins
                )
        # Progress records (resume after an interruption): the copying model, the label-continuity outcome and each
        # decoded sample batch, each published as soon as it is final and bound to this painting's derivation.
        saved = None if progress is None else progress.load(f"c{component_index}/model")
        if saved is not None:
            model = None if saved["model"] is None else from_payload(fidelity.ChainModel, saved["model"])
        else:
            model = component_model(
                panel,
                component_likelihoods,
                component_observed,
                sites,
                config,
                genetic_map,
                robustness_epsilon=epsilon, site_correlation=site_correlation,
                batch_samples=component_batch,
                cache_emissions=_packed_bytes(panel, component_likelihoods.shape[0], config) + held_read_bytes
                <= budget,
                emission_builder=builder,
            )
            if progress is not None:
                progress.save(f"c{component_index}/model", dict(model=None if model is None else to_payload(model)))
        if model_sink is not None:
            model_sink(component_index, model)
        saved = None if progress is None else progress.load(f"c{component_index}/continuity")
        if label_continuity is not None:
            painted = panel
        elif saved is not None:
            outcome = from_payload(LabelContinuity, saved["outcome"])
            painted = relabel_rows(panel, outcome.row_maps) if outcome.accepted else panel
        else:
            painted, outcome = relabel_component(
                panel,
                component_likelihoods,
                component_observed,
                sites,
                config,
                genetic_map,
                model=model,
                robustness_epsilon=epsilon, site_correlation=site_correlation,
                batch_samples=component_batch,
                reverse=reverse,
                census_sink=census_sink,
                emission_builder=builder,
                emission_cache_bytes=budget,
            )
            if progress is not None:
                progress.save(f"c{component_index}/continuity", dict(outcome=to_payload(outcome)))
        confidence_model = None
        if confidence_correlation is not None and float(confidence_correlation) != float(site_correlation):
            saved = None if progress is None else progress.load(f"c{component_index}/confidence_model")
            if saved is not None:
                confidence_model = (
                    None if saved["model"] is None else from_payload(fidelity.ChainModel, saved["model"])
                )
            else:
                confidence_model = component_model(
                    painted,
                    component_likelihoods,
                    component_observed,
                    sites,
                    config,
                    genetic_map,
                    robustness_epsilon=epsilon, site_correlation=float(confidence_correlation),
                    batch_samples=component_batch,
                    cache_emissions=_packed_bytes(painted, component_likelihoods.shape[0], config) + held_read_bytes
                    <= budget,
                    emission_builder=builder,
                )
                if progress is not None:
                    progress.save(
                        f"c{component_index}/confidence_model",
                        dict(model=None if confidence_model is None else to_payload(confidence_model)),
                    )
        label_model = None
        if label_strength:
            saved = None if progress is None else progress.load(f"c{component_index}/label_model")
            if saved is not None:
                label_model = None if saved["model"] is None else from_payload(fidelity.ChainModel, saved["model"])
            else:
                label_model = component_model(
                    painted,
                    component_likelihoods,
                    component_observed,
                    sites,
                    config,
                    genetic_map,
                    robustness_epsilon=epsilon, site_correlation=float(LABEL_ERROR_CORRELATION),
                    batch_samples=component_batch,
                    cache_emissions=_packed_bytes(painted, component_likelihoods.shape[0], config) + held_read_bytes
                    <= budget,
                    emission_builder=builder,
                )
                if progress is not None:
                    progress.save(
                        f"c{component_index}/label_model",
                        dict(model=None if label_model is None else to_payload(label_model)),
                    )
        decoding = decode_component(
            painted,
            component_likelihoods,
            component_observed,
            sites,
            config,
            genetic_map,
            model=model,
            confidence_model=confidence_model,
            label_model=label_model,
            robustness_epsilon=epsilon, site_correlation=site_correlation,
            emission_cache_bytes=budget,
            batch_samples=component_batch,
            reverse=reverse,
            emission_builder=builder,
            progress=progress,
            progress_key=f"c{component_index}",
            label_errors=label_errors,
        )
        if decoding.packed_emissions is not None:
            cache_remaining -= decoding.packed_emissions.nbytes
        components.append(release_component(painted, decoding, config, census_sink=census_sink))
        continuity.append(outcome)
        # Release-only paths, masks and posterior gates end here, before the next component, as does its resident
        # evidence copy.
        del decoding, component_likelihoods, component_observed
    return Painting(
        contig=evidence.contig,
        sample_ids=tuple(evidence.sample_ids),
        components=tuple(components),
        label_continuity=tuple(continuity),
        parameters=config,
        robustness_epsilon=epsilon,
        emission_divisor=1.0
        + (site_correlation if confidence_correlation is None else confidence_correlation)
        * (config.reference_snps_per_bin - 1),
        path_divisor=1.0 + site_correlation * (config.reference_snps_per_bin - 1),
    )
