"""Painting stage: label continuity, then the released painting of every component.

For one chromosome, each assembled component's copying model is first estimated from all its samples (the
ancestry-change rate and the panel-error chain, painting.fidelity.estimate_chain; release 2.14); the component is
then relabelled by label continuity (painting/label_continuity.py) under that model, and painted: every sample is
decoded as a diploid mosaic of the component's trajectory classes plus BACKGROUND, each track copying faithfully or
through a panel-error run, with exact Viterbi and forward-backward recurrences, and the release rule
(painting/release.py) turns the decoding into the typed product. Components are independent HMMs (an unconditional
reset at every assembly break). Painting reads every sample's evidence, including samples the sample-quality QC
keeps out of founder construction.

Work runs in bounded sample batches: per sample the decoder holds about 40 S'^2 B + 16 S^2 B + 24 L bytes (S' <= 2 S
- 1 track states: structured emissions, Viterbi back-pointers, one forward table, the label-pair masses), so a batch
holds one sample per Numba thread when memory allows: the decoder reserves its working memory from the run's memory
governor (execution.memory), as much as fits up to one sample per thread, at least the emission cache plus one
sample. Batches are independent samples, so the batch size never changes results. The packed (tempered) label
emissions are kept for pedigree preparation while they fit a 1 GiB budget per chromosome.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numba
import numpy as np

from .. import execution
from ..assembly import records as assembly_records
from ..maths import ties
from ..evidence import records as evidence_records
from ..model.components import ComponentPanel
from ..model.evidence import ContigEvidence
from . import fidelity, hmm, records
from .config import PaintingConfig
from .label_continuity import relabel_component
from .release import (ChromosomePainting, ComponentPainting, LabelContinuity, release_tracks,
                      track_members)
from .states import PaintingBins, PaintingStates, build_bins, build_states, local_allele_keys
from .transitions import PaintingTransitions, copying_transitions, edge_morgans

# Packed emissions kept for pedigree preparation, per chromosome.
EMISSION_CACHE_MAX_BYTES = 1024 ** 3
# Working-memory budget: at least this much (when it fits), and, for an explicit
# amount of available memory, a reserve of 10% of it (at least 4 GiB, at most half)
# left to the rest of the process.
MINIMUM_WORKING_MEMORY_BYTES = 512 * 1024 ** 2
MEMORY_RESERVE_BYTES = 4 * 1024 ** 3


@dataclass(frozen=True)
class ComponentDecoding:
    """The HMM decoding of one component before the release rule.

    paths: int32[N, 2, B] gauge-fixed Viterbi state per track and bin;
    masks: int64[N, 2, B] candidate states per track and bin as bits (the Viterbi
        state, plus the other side inside a switch interval, over both traceback
        conventions; release 2.2); qv/qa are the 2.1 Viterbi-pair masses, the smaller
        of the two conventions' pairs;
    switch_counts: int64[N] minimum track changes (the fewer of the two conventions' paths);
    qv, qa: float64[N, B] posterior masses of the Viterbi pairs (public classes, local keys), the smaller of the
        two conventions' pairs;
    qf, qaf: float64[N, B] the same masses with both tracks copying faithfully (no panel error): the release gate
        (release 2.14, see painting/release.py);
    model: the component's copying model (painting.fidelity.ChainModel; None without bins);
    support: bool[N, 2, B] direct read support of each Viterbi track
        (False for ineligible samples);
    batch_size, working_memory_bytes: the sample batching used (execution only).
    Other fields as in :class:`painting.release.ComponentPainting`.
    """

    states: PaintingStates
    bins: PaintingBins
    transitions: PaintingTransitions
    local_keys: np.ndarray
    eligible: np.ndarray
    paths: np.ndarray
    masks: np.ndarray
    switch_counts: np.ndarray
    qv: np.ndarray
    qa: np.ndarray
    support: np.ndarray
    packed_emissions: np.ndarray | None
    batch_size: int
    working_memory_bytes: int
    qf: np.ndarray
    qaf: np.ndarray
    model: fidelity.ChainModel | None


def working_memory_budget(bytes_per_sample: int, fixed_bytes: int, threads: int,
                          available_bytes: int | None) -> int:
    """Bytes the decoder may hold: one sample per thread plus fixed arrays, within the
    available memory after the reserve."""
    if available_bytes is None:
        return max(MINIMUM_WORKING_MEMORY_BYTES, fixed_bytes + bytes_per_sample)
    reserve = min(available_bytes // 2, max(MEMORY_RESERVE_BYTES, available_bytes // 10))
    usable = max(1, available_bytes - reserve)
    return min(usable, max(MINIMUM_WORKING_MEMORY_BYTES, fixed_bytes + threads * bytes_per_sample))


@contextmanager
def working_memory(bytes_per_sample: int, fixed_bytes: int, threads: int, available_bytes: int | None):
    """The decoder's working-memory budget for the block: computed from ``available_bytes`` when given
    (working_memory_budget), otherwise reserved from the run's memory governor: up to one sample per thread plus
    the emission cache, as much as fits now, at least the cache plus one sample (the cache is dropped only when
    even that exceeds the governor's limit)."""
    if available_bytes is not None:
        yield working_memory_budget(bytes_per_sample, fixed_bytes, threads, available_bytes)
        return
    governor = execution.memory_governor()
    if fixed_bytes + bytes_per_sample > governor.limit:
        fixed_bytes = 0
    desired = max(MINIMUM_WORKING_MEMORY_BYTES, fixed_bytes + threads * bytes_per_sample)
    with governor.reserve(desired, minimum=fixed_bytes + bytes_per_sample, label="painting decoder") as held:
        yield held.nbytes


def batch_size(n_samples: int, threads: int, bytes_per_sample: int, fixed_bytes: int,
               budget_bytes: int) -> int:
    """Samples per batch: one per thread, at most what the budget admits (at least one)."""
    memory_limited = max(1, max(0, budget_bytes - fixed_bytes) // bytes_per_sample)
    return min(n_samples, max(1, min(threads, n_samples)), memory_limited)


def _raw_batches(states, bins, likelihoods, observed, sites, config, size):
    """Generator factory: for each contiguous batch of ``size`` samples, (tempered label emissions float64[n, S*S, B],
    eligible bool[n], start, stop, gathered likelihoods, gathered observed)."""
    alt_probability = states.alt_probability()
    n_samples = likelihoods.shape[0]
    correlation = float(config.emission_site_correlation)

    def batches():
        for start in range(0, n_samples, size):
            stop = min(start + size, n_samples)
            # A boundary: the team follows the live share (a lane's lease grows as other lanes finish); the batch
            # size was fixed by the caller, so results and memory do not depend on it.
            execution.claim_threads()
            batch_likelihoods, batch_observed = hmm.gather_sites(likelihoods, observed, sites, slice(start, stop))
            raw = hmm.binned_emissions(batch_likelihoods, batch_observed, alt_probability,
                                       states.background_frequency, bins.starts, bins.stops,
                                       robustness_epsilon=config.robustness_epsilon,
                                       log_floor=config.emission_log_floor)
            raw /= hmm.design_effect(batch_likelihoods, batch_observed, bins.starts, correlation)[:, None, :]
            eligible = np.any(hmm.informative_sites(batch_likelihoods, batch_observed), axis=1)
            yield raw, eligible, start, stop, batch_likelihoods, batch_observed

    return batches


def _decoder_bytes(states: PaintingStates, n_bins: int) -> int:
    """Per-sample working bytes of the chain's estimation and decoding (module docstring)."""
    n_track = states.n_states + int(np.sum(states.anchored))
    return max(1, 40 * n_track * n_track * n_bins + 16 * states.n_states * states.n_states * n_bins
               + 24 * states.n_sites)


def error_capable(states: PaintingStates) -> np.ndarray:
    """int64[S]: 1 for the labels with a panel-error state (anchored classes), 0 for the others and BACKGROUND."""
    return np.concatenate((np.asarray(states.anchored, dtype=np.int64), np.zeros(1, dtype=np.int64)))


def component_model(panel: ComponentPanel, likelihoods: np.ndarray, observed: np.ndarray,
                    evidence_sites: np.ndarray, config: PaintingConfig, genetic_map, *,
                    available_memory: int | None = None) -> fidelity.ChainModel | None:
    """The component's copying model (rho, a, b; painting.fidelity.estimate_chain), estimated from all its samples on
    the assembled panel, or None without bins. Label continuity then uses it unchanged: a relabelling renames states
    but leaves their emissions and error capabilities."""
    n_samples = likelihoods.shape[0]
    states = build_states(panel.positions, panel.inference, panel.kept)
    bins = build_bins(states.positions, config.snps_per_bin)
    if bins.n_bins == 0:
        return None
    sites = evidence_sites[states.site_indices]
    threads = max(1, int(numba.get_num_threads()))
    bytes_per_sample = _decoder_bytes(states, bins.n_bins)
    with working_memory(bytes_per_sample, 0, threads, available_memory) as budget:
        size = batch_size(n_samples, threads, bytes_per_sample, 0, budget)
        batches = _raw_batches(states, bins, likelihoods, observed, sites, config, size)
        return fidelity.estimate_chain(lambda: ((raw, eligible) for raw, eligible, *_ in batches()),
                                       error_capable(states), edge_morgans(bins.centers, genetic_map), config)


def decode_component(panel: ComponentPanel, likelihoods: np.ndarray, observed: np.ndarray,
                     evidence_sites: np.ndarray, config: PaintingConfig, genetic_map, *,
                     model: fidelity.ChainModel | None = None,
                     emission_cache_bytes: int = EMISSION_CACHE_MAX_BYTES,
                     available_memory: int | None = None) -> ComponentDecoding:
    """Decode every sample of one component (all samples of ``likelihoods``).

    ``evidence_sites`` int64[L] maps the panel's sites to the evidence site axis (``panel.site_indices()``).
    ``model``: the component's copying model (component_model; estimated here when None). The packed tempered label
    emissions are kept when they fit ``emission_cache_bytes``.
    """
    n_samples = likelihoods.shape[0]
    states = build_states(panel.positions, panel.inference, panel.kept)
    bins = build_bins(states.positions, config.snps_per_bin)
    n_states = states.n_states
    n_bins = bins.n_bins
    if n_bins == 0:
        empty_grid = np.zeros((n_samples, 2, 0), dtype=np.int32)
        no_edges = np.zeros(0, dtype=np.float64)
        return ComponentDecoding(
            states=states, bins=bins,
            transitions=PaintingTransitions(n_states, no_edges, no_edges.copy(), no_edges.copy(),
                                            site_correlation=float(config.emission_site_correlation)),
            local_keys=np.zeros((0, n_states), dtype=np.int32),
            eligible=np.zeros(n_samples, dtype=np.bool_), paths=empty_grid,
            masks=np.zeros((n_samples, 2, 0), dtype=np.int64),
            switch_counts=np.zeros(n_samples, dtype=np.int64),
            qv=np.zeros((n_samples, 0)), qa=np.zeros((n_samples, 0)),
            support=np.zeros((n_samples, 2, 0), dtype=np.bool_), packed_emissions=None,
            batch_size=0, working_memory_bytes=0, qf=np.zeros((n_samples, 0)), qaf=np.zeros((n_samples, 0)),
            model=None)
    if model is None:
        model = component_model(panel, likelihoods, observed, evidence_sites, config, genetic_map,
                                available_memory=available_memory)
    capable = error_capable(states)
    transitions = copying_transitions(edge_morgans(bins.centers, genetic_map), n_states, model.rate,
                                      error_entry=model.entry, error_exit=model.exit,
                                      site_correlation=model.site_correlation)
    local_keys = local_allele_keys(states.named_alleles, bins)
    public_ids = states.public_state_ids()
    sites = evidence_sites[states.site_indices]

    threads = max(1, int(numba.get_num_threads()))
    n_packed = n_states * (n_states + 1) // 2
    packed_bytes = 8 * n_samples * n_packed * n_bins
    fixed_bytes = packed_bytes if packed_bytes <= emission_cache_bytes else 0
    bytes_per_sample = _decoder_bytes(states, n_bins)
    with working_memory(bytes_per_sample, fixed_bytes, threads, available_memory) as budget:
        if fixed_bytes + bytes_per_sample > budget:
            fixed_bytes = 0
        packed = np.empty((n_samples, n_packed, n_bins)) if fixed_bytes else None
        size = batch_size(n_samples, threads, bytes_per_sample, fixed_bytes, budget)

        paths = np.empty((n_samples, 2, n_bins), dtype=np.int32)
        masks = np.empty((n_samples, 2, n_bins), dtype=np.int64)
        switch_counts = np.zeros(n_samples, dtype=np.int64)
        qv = np.empty((n_samples, n_bins))
        qa = np.empty((n_samples, n_bins))
        qf = np.empty((n_samples, n_bins))
        qaf = np.empty((n_samples, n_bins))
        support = np.empty((n_samples, 2, n_bins), dtype=np.bool_)
        eligible = np.empty(n_samples, dtype=np.bool_)
        for raw, batch_eligible, start, stop, batch_likelihoods, batch_observed in _raw_batches(
                states, bins, likelihoods, observed, sites, config, size)():
            if packed is not None:
                hmm.pack_emissions(raw, packed, start, n_states)
            # Switch intervals (release 2.2) over both traceback conventions, the canonical track order (release
            # 2.14) and the release masses as before, of the chain's Viterbi labels.
            batch_paths, batch_masks, batch_switches, other_paths = fidelity.decode_candidates(
                raw, model, capable, local_keys, states.named_alleles, batch_observed, bins.starts, bins.stops,
                states.n_classes)
            (qv[start:stop], qa[start:stop], qf[start:stop],
             qaf[start:stop]) = fidelity.release_posteriors(raw, model, capable, batch_paths, other_paths,
                                                            public_ids, local_keys)
            del raw
            informative = hmm.informative_sites(batch_likelihoods, batch_observed)
            batch_support = hmm.direct_support(batch_paths, states.named_alleles, informative,
                                               bins.starts, bins.stops, states.n_classes)
            batch_support[~batch_eligible] = False
            paths[start:stop] = batch_paths
            masks[start:stop] = batch_masks
            switch_counts[start:stop] = batch_switches
            support[start:stop] = batch_support
            eligible[start:stop] = batch_eligible
    return ComponentDecoding(
        states=states, bins=bins, transitions=transitions, local_keys=local_keys,
        eligible=eligible, paths=paths, masks=masks, switch_counts=switch_counts, qv=qv, qa=qa, support=support,
        packed_emissions=packed, batch_size=size, working_memory_bytes=budget, qf=qf, qaf=qaf, model=model)


def release_component(panel: ComponentPanel, decoding: ComponentDecoding,
                      config: PaintingConfig) -> ComponentPainting:
    """Apply the release rule once and build the component's product."""
    states = decoding.states
    if ties.census_enabled() and decoding.bins.n_bins:
        # Continuous release thresholds are not banded (design A-7, owner 15:50): posteriors within 1e-5 of tau go
        # to the valley census so a crossing is reported with its distance.
        tau = config.minimum_release_posterior
        first = int(panel.positions[0])
        for name, values in (("painting.release.qf", decoding.qf), ("painting.release.qaf", decoding.qaf)):
            for sample, bin_index in zip(*np.nonzero(np.abs(values - tau) < 1e-5)):
                ties.census(name, f"p{first}/s{int(sample)}/b{int(bin_index)}",
                            [float(values[sample, bin_index]) - tau])
    # The release gate is the faithful-copy mass (release 2.14): the pair's rows carry the sample's alleles here.
    gate_v, gate_a = decoding.qf, decoding.qaf
    labels, classes, status = release_tracks(
        states, decoding.paths, decoding.eligible, gate_v, gate_a,
        decoding.local_keys, decoding.support, config.minimum_release_posterior, masks=decoding.masks)
    # A released interval track of a bin below the stage-1 threshold was released at stage 2 (local keys).
    member_ids, member_sets = track_members(status, classes, decoding.local_keys, states.classes,
                                            masks=decoding.masks,
                                            stage_two=gate_v < config.minimum_release_posterior,
                                            anchored=states.anchored)
    switch_counts = decoding.switch_counts.copy()
    switch_counts[~decoding.eligible] = 0
    return ComponentPainting(
        panel=panel, states=states, bins=decoding.bins, transitions=decoding.transitions,
        local_keys=decoding.local_keys, eligible=decoding.eligible, labels=labels,
        classes=classes, status=status, direct=decoding.support & (classes >= 0),
        member_ids=member_ids, member_sets=member_sets, switch_counts=switch_counts,
        packed_emissions=decoding.packed_emissions)


def paint_chromosome(panels: tuple[ComponentPanel, ...], evidence: ContigEvidence,
                     config: PaintingConfig, genetic_map, *,
                     available_memory: int | None = None) -> ChromosomePainting:
    """Label continuity, then the painting of every component of one chromosome.

    ``panels``: the final assembly's ComponentPanels in genomic order;
    ``evidence``: the contig's evidence (all samples); ``genetic_map``: the
    contig's ChromosomeGeneticMap. Numba kernels use the caller's thread team.
    ``available_memory`` (bytes) sizes the batches by formula; None reserves the
    decoders' working memory from the run's memory governor.
    """
    likelihoods = evidence.likelihoods
    observed = evidence.observed
    cache_remaining = EMISSION_CACHE_MAX_BYTES
    components: list[ComponentPainting] = []
    continuity: list[LabelContinuity] = []
    for panel in panels:
        sites = panel.site_indices()
        model = component_model(panel, likelihoods, observed, sites, config, genetic_map,
                                available_memory=available_memory)
        painted, outcome = relabel_component(panel, likelihoods, observed, sites, config,
                                             genetic_map, model=model)
        decoding = decode_component(painted, likelihoods, observed, sites, config, genetic_map, model=model,
                                    emission_cache_bytes=cache_remaining,
                                    available_memory=available_memory)
        if decoding.packed_emissions is not None:
            cache_remaining -= decoding.packed_emissions.nbytes
        components.append(release_component(painted, decoding, config))
        continuity.append(outcome)
    return ChromosomePainting(contig=evidence.contig, sample_ids=tuple(evidence.sample_ids),
                              components=tuple(components), label_continuity=tuple(continuity),
                              config=config)


def run_contig(store, contig: str, config: PaintingConfig, genetic_map, budget) -> dict:
    """Paint one contig from its final assembly and evidence records (resumable).

    Binds ``identity/painting`` (painting config + assembly and evidence record
    identities + the contig's genetic map) and writes ``painting/summary`` and
    ``painting``; returns the small progress summary.
    """
    file = store.contig(contig)
    records.bind_identity(store, file, config, genetic_map)
    if records.complete(file):
        return records.progress(file)
    panels = assembly_records.founder_panels(file)
    evidence = evidence_records.load(file)
    threads = budget.phase("painting").threads()
    with execution.numba_threads(threads):
        painting = paint_chromosome(panels, evidence, config, genetic_map)
    del panels, evidence
    moves = sum(len(outcome.moves) for outcome in painting.label_continuity if outcome.accepted)
    ambiguous = sum(not move.identified for outcome in painting.label_continuity if outcome.accepted
                    for move in outcome.moves)
    before = sum(outcome.switches_before or 0 for outcome in painting.label_continuity)
    after = sum(outcome.switches_after or 0 for outcome in painting.label_continuity)
    print(f"  [label continuity] {contig}: {moves} row-suffix relabellings ({ambiguous} identity-ambiguous); "
          f"painted switches {before} -> {after}", flush=True)
    models = [component.transitions for component in painting.components if component.bins.n_bins]
    if models:
        rates = [model.switch_rate for model in models]
        with_errors = sum(model.error_entry > fidelity.ENTRY_BOUNDS[0] for model in models)
        print(f"  [painting] {contig}: copying model rho {min(rates):.3g}-{max(rates):.3g} per Morgan, panel-error "
              f"runs in {with_errors} of {len(models)} components, site correlation {models[0].site_correlation:g} "
              f"(painting_model.tsv)", flush=True)
    counts = [records.interval_counts(component) for component in painting.components]
    print(f"  [painting] {contig}: {sum(count[0] for count in counts)} switch intervals "
          f"({sum(count[1] for count in counts)} track bins) released", flush=True)
    records.write(file, painting)
    return records.progress(file)
