"""Orchestrate painting progress, sample blocks and component cache residency."""

from __future__ import annotations

from collections.abc import Callable
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
from ...science.painting.parameters import EMISSION_CACHE_MAX_BYTES, PaintingParameters
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
    capable, transitions, local_keys, public_ids, sites = decoding_model(
        states, bins, evidence_sites, genetic_map, model
    )

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
        batch_paths, batch_masks, batch_switches, batch_qf, batch_qaf, batch_support = decode_batch(
            raw, batch_eligible, batch_observed, informative, states, bins, model, capable,
            local_keys, public_ids, config, reverse=reverse,
        )
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


def _component_sources(
    panel: ComponentPanel,
    likelihoods: ArraySource,
    observed: ArraySource,
    sites: np.ndarray,
    config: PaintingParameters,
    budget: int,
) -> tuple[tuple[Any, Any, np.ndarray], int]:
    """((likelihoods, observed, panel-site map) to build the component's emissions from, the emission budget left).

    When the evidence offers resident copies (resident_columns), the component's evidence at its panel
    sites (each sample's genotype-likelihood codes and its likelihood dictionary, far smaller than the emissions) is
    read once into memory, charged to the budget, whenever the copying-model estimate and the decode keep the packed
    emissions they would keep without it (or could not keep them anyway). Every emission build of the component, the
    estimate's, label continuity's and the decode's, is then a parallel in-memory gather and emission kernel instead
    of a serial read of the stored evidence. The values are the same."""
    unchanged = (likelihoods, observed, sites), budget
    if not (hasattr(likelihoods, "resident_columns") and hasattr(observed, "resident_columns")):
        return unchanged
    states = build_states(panel.positions, panel.inference, panel.kept)
    bins = build_bins(states.positions, config.snps_per_bin)
    if bins.n_bins == 0:
        return unchanged
    packed = 8 * likelihoods.shape[0] * (states.n_states * (states.n_states + 1) // 2) * bins.n_bins
    copy_bytes = likelihoods.resident_bytes(len(sites)) + observed.resident_bytes(len(sites))
    if copy_bytes + (packed if packed <= budget else 0) > budget:
        return unchanged
    copy = likelihoods.resident_columns(sites)
    resident = (copy, observed.resident_columns(sites, copy), np.arange(len(sites), dtype=np.int64))
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
        (component_likelihoods, component_observed, sites), budget = _component_sources(
            panel, likelihoods, observed, panel.site_indices(), config, cache_remaining
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
                batch_samples=batch_samples,
                cache_emissions=_packed_bytes(panel, component_likelihoods.shape[0], config) <= budget,
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
                component_likelihoods,
                component_observed,
                sites,
                config,
                genetic_map,
                model=model,
                batch_samples=batch_samples,
                reverse=reverse,
                census_sink=census_sink,
                emission_builder=emission_builder,
                emission_cache_bytes=budget,
            )
            if progress is not None:
                progress.save(f"c{component_index}/continuity", dict(outcome=to_payload(outcome)))
        decoding = decode_component(
            painted,
            component_likelihoods,
            component_observed,
            sites,
            config,
            genetic_map,
            model=model,
            emission_cache_bytes=budget,
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
        # Release-only paths, masks and posterior gates end here, before the next component, as does its resident
        # evidence copy.
        del decoding, component_likelihoods, component_observed
    return Painting(
        contig=evidence.contig,
        sample_ids=tuple(evidence.sample_ids),
        components=tuple(components),
        label_continuity=tuple(continuity),
        parameters=config,
    )
