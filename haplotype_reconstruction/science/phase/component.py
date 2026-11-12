"""Compose a single family-phase component through both solves and their release.

Owns the FP01-FP21 transition sequence; must not read files, clocks, environment,
choose a worker count, or alter search budgets. Method: docs/methods/family_phase.md.
Declared phase -> painting dependency: binned_emissions, design_effect, gather_sites,
unpack_emissions and state_alt_probabilities, through science.painting.api.
The two solves retain their distinct search orders; the release is asymmetric in
orientation and uses the default solve's calls wherever both solves agree.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

import numpy as np

from ...contracts.phase import (
    AlleleUncertainty,
    MeiosisChain,
    PhaseComponentInputs,
    PhaseComponentResult,
    PhaseComponentState,
    PhaseSolve,
    SampleRelease,
)
from ...contracts.roles import record_field

if TYPE_CHECKING:
    from ...contracts.pedigree import ParentEdges
    from ...contracts.phase import PhasedSamples
from ..painting.api import (
    binned_emissions,
    design_effect,
    gather_sites,
    state_alt_probabilities,
    unpack_emissions,
)
from .agreement import agreement, edge_chains, set_start_bins
from .allele_evidence import allele_evidence, allele_prior, flip_probabilities
from .inputs import _state_alleles, edge_rates
from .junctions import JunctionInputs, junction_candidates, withheld_joins
from .model import LabelModel
from .painting_inputs import component_bin_edges, tie_break_bins
from .parameters import SITE_WINDOW, PhaseParameters
from .release import ReleaseInputs, release_sample, shared_windows

__all__ = ["PhaseWorkspace", "phase_component", "component_emissions"]


_MemoValue = TypeVar("_MemoValue")


class PhaseMemo(Protocol):
    """Exact unit reuse, with noncomputing lookup and independent publication."""

    def __call__(self, unit: tuple, compute: Callable[[], _MemoValue]) -> _MemoValue: ...

    def lookup(self, unit: tuple) -> Any | None: ...

    def save(self, unit: tuple, value: Any) -> None: ...


@dataclass(frozen=True)
class PhaseWorkspace:
    """Explicit release storage capability: whole arrays or complete-bin projections.

    Site evidence is a supplied ndarray (possibly memory mapped); the stage owns
    its lifetime. Carrier evidence always uses its fixed EVIDENCE_CHUNK boundaries. The optional
    map capability returns results in input order; it cannot choose arithmetic order.
    The optional memo returns an exact unit value or calls compute once. Units are
    (component, "solve"), (component, "release", batch), and (component, "component").
    A supplied memo requires the stage's fixed positive release_batch in samples;
    batches are storage units only. All pending releases share the same map and
    CPU authority; completed batches are published in completion order.
    """

    windowed: bool = record_field("diagnostic", units="1", dtype="bool", stored=False, default=False)
    scores: bool = record_field("diagnostic", units="1", dtype="bool", stored=False, default=True)
    prepare_likelihoods: Callable[..., np.ndarray] | None = record_field(
        "diagnostic", units="1", dtype="Callable[..., np.ndarray] | None", stored=False, default=None
    )
    map_work: Callable[..., list[Any]] | None = record_field(
        "diagnostic", units="1", dtype="Callable[..., list[Any]] | None", stored=False, default=None
    )
    # ``concurrent(functions)`` runs each function(map_work) at once, each with a map over its own CPU share, and
    # returns their results in order: independent solves map their block waves.
    concurrent: Callable[..., list[Any]] | None = record_field(
        "diagnostic", units="1", dtype="Callable[..., list[Any]] | None", stored=False, default=None
    )
    # The stage supplies this capability where elastic release intervals repay
    # per-sample admission. Otherwise releases share reusable map workers.
    concurrent_releases: Callable[..., list[Any]] | None = record_field(
        "diagnostic", units="1", dtype="Callable[..., list[Any]] | None", stored=False, default=None
    )
    memo: PhaseMemo | None = record_field("diagnostic", units="1", dtype="PhaseMemo | None", stored=False, default=None)
    release_batch: int | None = record_field(
        "diagnostic", units="samples", dtype="int | None", stored=False, default=None
    )


def component_emissions(inputs: PhaseComponentInputs, *, recompute: bool = False, windowed: bool = False) -> np.ndarray:
    """Return float64[N,S,S,M] tempered emissions; mutate no input (FP01).

    The painting's packed cache is optional. Without it (or with recompute) the same complete bins are recomputed
    with the painting's own kernel and design effect. Resident: all samples and sites at once. Windowed: one sample
    at a time, in runs of whole bins spanning at most SITE_WINDOW sites (a bin
    longer than that is read whole), so only one sample's window of evidence is ever copied. Every bin's emission
    and design effect depend only on that sample's sites in that bin, so both routes give the same values.
    """
    painted = inputs.painting
    samples = len(inputs.observations.sample_ids)
    states = painted.states.n_states
    if painted.packed_emissions is not None and not recompute:
        return unpack_emissions(painted.packed_emissions, states).reshape(samples, states, states, -1)
    selected = np.searchsorted(inputs.observations.positions, painted.states.positions)
    probabilities = state_alt_probabilities(painted.states.named_alleles, painted.states.background_frequency)
    if windowed:
        return _windowed_emissions(inputs, selected, probabilities).reshape(samples, states, states, -1)
    gl, observed = gather_sites(inputs.observations.likelihoods, inputs.observations.observed, selected)
    emissions = binned_emissions(
        gl,
        observed,
        probabilities,
        painted.states.background_frequency,
        painted.bins.starts,
        painted.bins.stops,
        robustness_epsilon=painted.robustness_epsilon,
        log_floor=painted.emission_log_floor,
    )
    emissions /= design_effect(gl, observed, painted.bins.starts, painted.transitions.site_correlation)[:, None, :]
    return emissions.reshape(samples, states, states, -1)


def _windowed_emissions(inputs: PhaseComponentInputs, selected: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """float64[N, S*S, M] emissions recomputed one sample and one run of whole bins at a time."""
    painted = inputs.painting
    samples = len(inputs.observations.sample_ids)
    starts, stops = painted.bins.starts, painted.bins.stops
    background = painted.states.background_frequency
    emissions = np.empty((samples, painted.states.n_states**2, len(starts)), dtype=np.float64)
    for n in range(samples):
        first = 0
        while first < len(starts):
            last = first + 1
            while last < len(starts) and stops[last] - starts[first] <= SITE_WINDOW:
                last += 1
            left, right = int(starts[first]), int(stops[last - 1])
            gl, observed = gather_sites(
                inputs.observations.likelihoods, inputs.observations.observed, selected[left:right], slice(n, n + 1)
            )
            raw = binned_emissions(
                gl,
                observed,
                probabilities[:, left:right],
                background[left:right],
                starts[first:last] - left,
                stops[first:last] - left,
                robustness_epsilon=painted.robustness_epsilon,
                log_floor=painted.emission_log_floor,
            )
            raw /= design_effect(gl, observed, starts[first:last] - left, painted.transitions.site_correlation)[
                :, None, :
            ]
            emissions[n : n + 1, :, first:last] = raw
            first = last
    return emissions


def _solve_one(model, map_work=None):
    model.solve(map_work=map_work)
    return model


def _solve_models(inputs, parameters, emissions, rates, map_work=None, concurrent=None):
    """Default then mirror, with the specified sample and pairing search orders (FP05-FP09)."""
    parents = np.full((len(inputs.observations.sample_ids), 2), -1, dtype=np.int64)
    edges = inputs.pedigree
    for parent, child, slot in zip(edges.parents, edges.children, edges.slots, strict=False):
        parents[int(child), int(slot)] = int(parent)
    models = tuple(
        LabelModel(
            emissions,
            parents,
            rates.theta_l,
            rates.theta_m,
            rates.alpha,
            rates.beta,
            parameters,
            reverse=reverse,
        )
        for reverse in (False, True)
    )
    if map_work is None:
        for model in models:
            model.solve()
    elif concurrent is not None:
        # Both solves at once, each running its blocks in conflict-free waves on its CPU share.
        models = tuple(concurrent([partial(_solve_one, model) for model in models]))
    else:
        models = tuple(map_work(_solve_one, models))
    return models, parents


def _forced_breaks(inputs, models, parents, bin_edges):
    """Union junctions, parental-pairing ambiguity and painting ties; only isolated ties may bridge (FP10-FP12)."""
    model, mirror = models
    final = [JunctionInputs(m.X, m.V, m.E, m.par, m.config) for m in models]
    junctions = [junction_candidates(state) for state in final]
    withheld = withheld_joins(final[0], junctions[0])
    for sample, bins in withheld_joins(final[1], junctions[1]).items():
        withheld[sample] = sorted(set(withheld.get(sample, ())) | set(bins))
    pairing = {}
    for solved in models:
        for parent, edge in solved.ambiguous_pairings():
            for child in [parent] + solved.children_of(parent):
                pairing.setdefault(child, set()).add(int(edge))
    positions = inputs.observations.positions
    ties = tie_break_bins((inputs.painting,), positions, parents, bin_edges)
    break_bins, tie_cuts = [], []
    for sample in range(len(parents)):
        tie_bins = ties[sample]
        joins = set(withheld.get(sample, ()))
        pairs = {b for b in pairing.get(sample, ()) if 0 < b < model.B}
        break_bins.append(tuple(sorted(tie_bins | joins | pairs)))
        tie_cuts.append(tuple(sorted(tie_bins - joins - pairs)))
    return junctions, withheld, pairing, tuple(break_bins), tuple(tie_cuts)


def _release_sites(inputs, bin_edges, bins):
    """Project refit alleles only onto evidence sites inside the painted bins (FP13)."""
    positions, source = inputs.observations.positions, inputs.refit.positions
    columns = np.searchsorted(source, positions)
    valid = columns < len(source)
    valid[valid] &= source[columns[valid]] == positions[valid]
    site_bins = np.searchsorted(bin_edges, positions, side="right") - 1
    indices = np.flatnonzero(valid & (site_bins >= 0) & (site_bins < bins))
    alleles = _state_alleles(inputs.painting.states, inputs.refit.inference, inputs.painting.states.n_states)
    return indices, site_bins[indices], alleles[:, columns[indices]]


def _uncertainty(model, inputs, site_bins, site_alleles, site_gl, map_work=None):
    """Allele uncertainty with fixed chunk and carrier addition order (FP14).

    ``map_work`` runs the fixed chunks concurrently.
    """
    epsilon = inputs.painting.robustness_epsilon
    excluded = inputs.observations.excluded
    evidence = allele_evidence(model.X, site_bins, site_alleles, site_gl, excluded, epsilon, map_work=map_work)
    prior = allele_prior(inputs.refit.prior_ones, inputs.refit.prior_called)
    return AlleleUncertainty(evidence, flip_probabilities(evidence, site_alleles, prior), prior)


def _release_one(arguments, map_work=None):
    return release_sample(*arguments, map_work=map_work)


def _sample_releases(solves, inputs, sites, breaks, uncertainties, parameters, workspace):
    """Release independent samples from the fixed solves (default, then mirror), each solve's samples in run order
    (FP15-FP19); one release tuple per solve. The releases of both solves are independent and form one map, so the
    mirror's releases do not wait for the default solve's slowest sample. Each
    release may map its hard-cut intervals over its live share."""
    jobs = [
        _release_arguments(solved, inputs, sites, breaks, uncertainty, parameters, workspace)
        for solved, uncertainty in zip(solves, uncertainties, strict=True)
    ]
    if workspace.memo is None:
        return _release_work(jobs, workspace)
    batch = workspace.release_batch
    if type(batch) is not int or batch <= 0:
        from ...core.errors import InvariantError

        raise InvariantError("phase memo requires a positive release batch size")
    from threading import Lock

    samples = len(jobs[0])
    releases: list[list[Any]] = [[None] * samples for _ in jobs]
    remaining = {}
    for number, first in enumerate(range(0, samples, batch)):
        end = min(first + batch, samples)
        saved = workspace.memo.lookup((inputs.component, "release", number))
        if saved is None:
            remaining[number] = len(jobs) * (end - first)
        else:
            for output, values in zip(releases, saved, strict=True):
                output[first:end] = values
    guard = Lock()

    def completed(solve, sample, value):
        number = sample // batch
        ready = None
        with guard:
            releases[solve][sample] = value
            remaining[number] -= 1
            if remaining[number] == 0:
                first = number * batch
                ready = tuple(tuple(values[first : first + batch]) for values in releases)
        if ready is not None:
            workspace.memo.save((inputs.component, "release", number), ready)

    work = [
        (solve, sample, arguments, completed)
        for solve, job in enumerate(jobs)
        for sample, arguments in enumerate(job)
        if sample // batch in remaining
    ]
    if work:
        _map_releases(_release_pending, work, workspace)
    return tuple(tuple(values) for values in releases)


def _release_pending(job, map_work=None):
    """Compute one pending sample on its existing share; publish its batch when complete."""
    solve, sample, arguments, completed = job
    value = _release_one(arguments, map_work=map_work)
    completed(solve, sample, value)


def _map_releases(function, work, workspace):
    """One shared map for all pending releases, preserving elastic interval authority."""
    if workspace.concurrent_releases is not None:
        return workspace.concurrent_releases([partial(function, arguments) for arguments in work])
    if workspace.map_work is not None:
        return workspace.map_work(function, work)
    return [function(arguments) for arguments in work]


def _release_work(jobs, workspace):
    """Map independent releases in solve order, retaining each sample's arithmetic order."""
    work = [arguments for job in jobs for arguments in job]
    released = _map_releases(_release_one, work, workspace)
    ends = np.cumsum([len(job) for job in jobs])
    return tuple(released[end - len(job) : end] for job, end in zip(jobs, ends.tolist(), strict=True))


def _release_arguments(model, inputs, sites, breaks, uncertainty, parameters, workspace):
    """Every sample's release inputs from one fixed solve, in run order."""
    site_bins, alleles, site_gl = sites
    break_bins, tie_cuts = breaks
    # The solve's window copies of its alleles and flips, made once and shared by all its samples.
    shared = shared_windows(site_bins, alleles, uncertainty.flips) if workspace.windowed else None
    values = []
    for sample in range(model.N):
        context = ReleaseInputs(
            parameters,
            site_bins,
            alleles,
            site_gl[sample],
            break_bins[sample],
            tie_cuts[sample],
            uncertainty.evidence,
            uncertainty.flips,
            uncertainty.logit_prior,
            inputs.painting.robustness_epsilon,
            bool(inputs.observations.excluded[sample]),
            workspace.windowed,
            workspace.scores,
            True,
            shared_windows=shared,
        )
        values.append((model, sample, context))
    return values


def phase_component(
    inputs: PhaseComponentInputs,
    parameters: PhaseParameters,
    workspace: PhaseWorkspace | None = None,
) -> PhaseComponentState:
    """Solve and release one painted component with at least one bin (FP01-FP21).

    Inputs and output axes are documented in contracts.phase. Inputs are borrowed
    and never mutated; solver work arrays and outputs are newly allocated. Empty
    unpainted components are outside this component task. Errors from invalid
    evidence/parameters propagate to the task boundary. No randomness is used.
    """
    workspace = PhaseWorkspace() if workspace is None else workspace
    if workspace.memo is None:
        return _phase_component(inputs, parameters, workspace)
    return workspace.memo((inputs.component, "component"), lambda: _phase_component(inputs, parameters, workspace))


def _phase_component(inputs, parameters, workspace):
    if workspace.memo is None:
        emissions = component_emissions(inputs, windowed=workspace.windowed)
        rates = edge_rates(inputs.painting, inputs.genetic_map, parameters)
        models, parents = _solve_models(inputs, parameters, emissions, rates, workspace.map_work, workspace.concurrent)
    else:
        rates = edge_rates(inputs.painting, inputs.genetic_map, parameters)

        def solve():
            emissions = component_emissions(inputs, windowed=workspace.windowed)
            return _solve_models(inputs, parameters, emissions, rates, workspace.map_work, workspace.concurrent)

        models, parents = workspace.memo((inputs.component, "solve"), solve)
        emissions = models[0].E
    model, mirror = models
    bin_edges = component_bin_edges(inputs.painting, inputs.refit.positions)
    junctions, withheld, pairing, break_bins, tie_cuts = _forced_breaks(inputs, models, parents, bin_edges)
    site_indices, site_bins, site_alleles = _release_sites(inputs, bin_edges, model.B)
    site_gl = (
        np.ascontiguousarray(inputs.observations.likelihoods[:, site_indices], dtype=np.float32)
        if workspace.prepare_likelihoods is None
        else workspace.prepare_likelihoods(inputs.observations, site_indices, site_bins)
    )
    # Each solve's fixed evidence chunks run concurrently; chunk boundaries, not the worker count, fix the sums.
    chunks = workspace.map_work
    uncertainty = tuple(
        _uncertainty(solved, inputs, site_bins, site_alleles, site_gl, map_work=chunks) for solved in models
    )
    releases = _sample_releases(
        models,
        inputs,
        (site_bins, site_alleles, site_gl),
        (break_bins, tie_cuts),
        uncertainty,
        parameters,
        workspace,
    )
    pairs = list(zip(*releases, strict=False))
    if workspace.map_work is not None:
        # Samples' agreements are independent of one another.
        agreed = workspace.map_work(lambda pair: agreement(pair[0], pair[1], site_bins), pairs)
    else:
        agreed = [agreement(first, second, site_bins) for first, second in pairs]
    if workspace.map_work is not None:
        starts = tuple(workspace.map_work(lambda value: set_start_bins(value.set_starts, site_bins, model.B), agreed))
    else:
        starts = tuple(set_start_bins(value.set_starts, site_bins, model.B) for value in agreed)
    chains = edge_chains(
        model,
        mirror,
        [value.chain for value in releases[0]],
        [value.chain for value in releases[1]],
        rates.theta_m,
        starts,
        parameters.phase_posterior,
    )
    result = PhaseComponentResult(
        cast(
            tuple[PhaseSolve, PhaseSolve],
            tuple(
                PhaseSolve(
                    solved.X,
                    solved.V,
                    tuple(solved.history),
                    solved.settled,
                    tuple(solved.pairings),
                    tuple(solved.pairing_moves),
                )
                for solved in models
            ),
        ),
        site_indices,
        site_bins,
        site_alleles,
        uncertainty,
        cast(
            tuple[tuple[SampleRelease, ...], tuple[SampleRelease, ...]],
            tuple(tuple(per_solve) for per_solve in releases),
        ),
        tuple(agreed),
        tuple(
            MeiosisChain(*chains[(int(p), int(c), int(k))])
            for p, c, k in zip(inputs.pedigree.parents, inputs.pedigree.children, inputs.pedigree.slots, strict=False)
        ),
    )
    return PhaseComponentState(
        emissions,
        rates,
        parents,
        tuple(tuple(j) for j in junctions),
        {n: tuple(v) for n, v in withheld.items()},
        {n: tuple(sorted(v)) for n, v in pairing.items()},
        break_bins,
        tie_cuts,
        starts,
        result,
    )


def phase_contig(
    contig: str,
    sample_ids: tuple[str, ...],
    positions: np.ndarray,
    inputs: tuple[PhaseComponentInputs, ...],
    parameters: PhaseParameters,
    edges: ParentEdges,
    workspace: PhaseWorkspace | None = None,
    *,
    allocate: Callable[..., np.ndarray] | None = None,
) -> PhasedSamples:
    """Assemble every component in order into one complete PhasedSamples (FP22-FP24).

    Empty paintings keep their fallback geometry and chain columns. The optional
    allocation capability owns buffers only; it cannot choose model or reduction
    order. Inputs already passed the canonical stage boundary. No I/O occurs here.
    """
    from ...contracts.phase import MirroredSolveDiagnostics, PairingDiagnostics, PhaseDiagnostics, PhasedSamples
    from ...core.errors import InvariantError
    from .diagnostics import phase_set_count, solve_diagnostics
    from .painting_inputs import coverage

    workspace = PhaseWorkspace() if workspace is None else workspace
    allocate = np.full if allocate is None else allocate
    n, length = len(sample_ids), len(positions)
    calls = allocate((n, length, 2), -1, dtype=np.int8)
    undetermined = allocate((n, length), 0, dtype=np.int8)
    component_ids = np.full(length, -1, dtype=np.int32)
    breaks: list[list[int]] = [[] for _ in range(n)]
    all_edges, all_centers, histories, settled, candidates = [], [], [], [], []
    objectives: list[tuple[float, ...]] = []
    mirrored: list[MirroredSolveDiagnostics | None] = []
    pairing_rows: list[PairingDiagnostics | None] = []
    chain_parts = []
    withheld_joins = tie_breaks = 0
    for index, component in enumerate(inputs):
        if component.contig != contig or component.component != index:
            raise InvariantError("phase component order differs from the contig boundary")
        painted, source = component.painting, component.refit.positions
        bin_edges = component_bin_edges(painted, source)
        all_edges.append(bin_edges)
        all_centers.append(
            np.asarray(painted.bins.centers, dtype=np.float64)
            if len(painted.bins.edges) > 1
            else (bin_edges[:-1] + bin_edges[1:]) / 2.0
        )
        lo = int(np.searchsorted(positions, source[0], side="left"))
        hi = int(np.searchsorted(positions, source[-1], side="right"))
        if not np.array_equal(positions[lo:hi], component.observations.positions):
            raise InvariantError("phase component evidence sites differ from the contig span")
        component_ids[lo:hi] = index
        if len(painted.bins.edges) <= 1:
            histories.append(0)
            settled.append(True)
            objectives.append(())
            candidates.append(0)
            mirrored.append(None)
            pairing_rows.append(None)
            continue
        state = phase_component(component, parameters, workspace)
        first, second = state.result.solves
        histories.append(len(first.history) - 1)
        settled.append(first.settled and second.settled)
        objectives.append(first.history)
        candidates.append(len({j[1:] for js in state.junctions for j in js}))
        mirror_row, pairing_row = solve_diagnostics(state, parameters)
        mirrored.append(mirror_row)
        pairing_rows.append(pairing_row)
        ties = tie_break_bins((painted,), component.observations.positions, state.parents, bin_edges)
        indices = lo + state.result.site_indices
        for sample, agreed in enumerate(state.result.agreed):
            tie_bins = ties[sample]
            joins = set(state.withheld.get(sample, ()))
            tie_breaks += len(tie_bins)
            withheld_joins += len(joins - tie_bins)
            a, b = state.result.releases[0][sample], state.result.releases[1][sample]
            tie_breaks -= len(set(state.tie_cuts[sample]) - set(a.set_breaks) - set(b.set_breaks))
            calls[sample, indices] = agreed.calls
            undetermined[sample, indices] = np.where(agreed.undetermined, 3, 0)
            breaks[sample].extend(lo + int(state.result.site_indices[i]) - 1 for i in agreed.set_starts)
        chain_parts.append((index, state.result.chains))
    sizes = np.asarray([len(e) - 1 for e in all_edges], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(sizes)].astype(np.int64)
    chain_source = allocate((len(edges.parents), int(offsets[-1])), -1, dtype=np.int8)
    chain_support = allocate(chain_source.shape, 0, dtype=np.float32)
    chain_reset = allocate(chain_source.shape, False, dtype=bool)
    chain_reset[:, offsets[:-1][sizes > 0]] = True
    for index, chains in chain_parts:
        left, right = offsets[index : index + 2]
        for row, chain in enumerate(chains):
            chain_source[row, left:right] = chain.source
            chain_support[row, left:right] = chain.support
            chain_reset[row, left:right] = chain.reset
    called = phased = unphased = sets = 0
    for sample in range(n):
        row = calls[sample]
        het = (row[:, 0] >= 0) & (row[:, 1] >= 0) & (row[:, 0] != row[:, 1])
        und = het & (undetermined[sample] > 0)
        called += int(np.sum(row >= 0))
        phased += int(np.sum(het & ~und))
        unphased += int(np.sum(und))
        sets += phase_set_count((het & ~und)[None, :], component_ids, [breaks[sample]])
    cover, cover_offsets = coverage(tuple(c.painting for c in inputs), positions, n)
    diagnostics = PhaseDiagnostics(
        tuple(histories),
        tuple(settled),
        tuple(objectives),
        tuple(candidates),
        withheld_joins,
        tie_breaks,
        called,
        phased,
        unphased,
        sets,
        tuple(mirrored),
        tuple(pairing_rows),
    )
    return PhasedSamples(
        contig,
        sample_ids,
        positions,
        component_ids,
        np.concatenate(all_edges).astype(np.int64) if all_edges else np.zeros(0, dtype=np.int64),
        np.r_[0, np.cumsum([len(e) for e in all_edges])].astype(np.int64),
        np.concatenate(all_centers).astype(np.float64) if all_centers else np.zeros(0),
        calls,
        cover,
        cover_offsets,
        edges.identity,
        undetermined,
        np.asarray([b for sample in breaks for b in sorted(set(sample))], dtype=np.int64),
        np.r_[0, np.cumsum([len(set(sample)) for sample in breaks])].astype(np.int64),
        chain_source,
        chain_support,
        chain_reset,
        diagnostics=diagnostics,
    )
