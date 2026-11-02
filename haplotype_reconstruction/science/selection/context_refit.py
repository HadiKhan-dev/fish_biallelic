"""Refit assembled context founders into evidence-based proposals for covered blocks.

Owns LS26-LS31: complete-coverage tasks, ordered-pair cavity HMM, founder-site
posterior, witness masks and duplicate local-row aggregation. Must not assemble,
schedule workers, read storage or inspect process state.
Method: docs/methods/local_selection.md#context-refit (WP_SE2 section 4.13).
Distinct inference rows retain consensus released templates. Local duplicates
sum support and use its weighted posterior, with a plain mean at zero support.
Background count is the minimum label among individual pairs within
BACKGROUND_COUNT_BAND of the pair maximum; pair weights are never aggregated by
label. L11 forms interval Morgans once before the redraw expm1 transition.
Accretion (LS29): the ordered-pair background rule is preserved as scientific state.
The component's N x B x S x S workspace remains the approved 0c memory exemption.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numba import prange, types

from ...contracts.genetic_map import ChromosomeGeneticMap
from ...contracts.panels import ComponentPanel, LocalPanel
from ...contracts.roles import freeze_record_arrays, record_field
from ...contracts.selection import ContextProposal
from ...core.kernels import kernel
from ...core.ties import census, census_enabled
from ..assembly.api import (
    INFO_FIELDS,
    INFO_MULTIMODAL,
    SETTING_HALF_BAND,
    SitePosteriorParameters,
    guard_withholds,
    site_release_marginals,
    summarize_sites,
)
from .diagnostics import CensusSink
from .fit import unique_rows
from .parameters import BACKGROUND_COUNT_BAND, HALF_CENSUS_WINDOW, ContextRefitConfig, SelectionParameters

__all__ = ["refit_task_bytes", "refit_component", "ContextRefitTask", "finish_component", "refit_tasks"]


# Per site, a copy state's allele probability is 0 or 1 (a called founder) or the site
# background (an uncalled founder or the background state): codes 0, 1, 2. The emission term of
# a state pair depends only on its unordered code pair, terms 0-5 for (0, 0), (0, 1), (0, 2),
# (1, 1), (1, 2), (2, 2), or, for a founder paired with itself (both copies one founder), on its
# code: terms 6, 7, 8.
_EMISSION_TERMS = 9
_TERM_LOW = (0, 0, 0, 1, 1, 2)
_TERM_HIGH = (0, 1, 2, 1, 2, 2)
_CODE_PAIR_TERM = (0, 1, 2, 1, 3, 4, 2, 4, 5)


@kernel(overloads=((types.int64, types.float64),), cache=True, inline="always")
def _code_probability(code, background):
    return background if code == 2 else float(code)


@kernel(
    overloads=tuple(
        (
            types.Array(types.int8, 2, "C", readonly=frozen),
            types.Array(types.float64, 3, "C", readonly=False),
            types.Array(types.boolean, 2, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=frozen),
            types.Array(types.int64, 1, "C", readonly=frozen),
            types.float64,
        )
        for frozen in (False, True)
    ),
    cache=True,
    parallel=True,
)
def _emission_grid(h, gl, observed, starts, stops, mix):
    """Per sample and bin, the summed log emission of every copy-state pair (sites added in
    ascending order); the per-site background ALT frequency (1 + called ALT) / (2 + called)."""
    n, length, _ = gl.shape
    k = len(h)
    s = k + 1
    emission = np.zeros((n, len(starts), s, s))
    background = np.empty(length)
    informative = np.empty(length, dtype=np.bool_)
    code = np.empty((s, length), dtype=np.int64)
    for site in range(length):
        called = 0
        alt = 0
        for i in range(k):
            if h[i, site] >= 0:
                called += 1
                alt += h[i, site]
                code[i, site] = h[i, site]
            else:
                code[i, site] = 2
        code[k, site] = 2
        background[site] = (1.0 + alt) / (2.0 + called)
        informative[site] = called > 0
    width = 0
    for b in range(len(starts)):
        width = max(width, stops[b] - starts[b])
    pair_term = np.array(_CODE_PAIR_TERM)
    for sample in prange(n):
        terms = np.empty((width, _EMISSION_TERMS))
        for b in range(len(starts)):
            start, stop = starts[b], stops[b]
            for site in range(start, stop):
                if not observed[sample, site] or not informative[site]:
                    continue
                e = gl[sample, site]
                total = e.sum()
                for term in range(_EMISSION_TERMS):
                    if term < 6:
                        # Symmetric in the two copies (IEEE products commute).
                        qi = _code_probability(_TERM_LOW[term], background[site])
                        qj = _code_probability(_TERM_HIGH[term], background[site])
                        p0 = (1 - qi) * (1 - qj)
                        p2 = qi * qj
                        p1 = 1 - p0 - p2
                    else:
                        qi = _code_probability(term - 6, background[site])
                        p0 = 1 - qi
                        p1 = 0.0
                        p2 = qi
                    likelihood = (e[0] * p0 + e[1] * p1 + e[2] * p2) / total if total > 0 else 1 / 3
                    terms[site - start, term] = np.log((1 - mix) * likelihood + mix / 3)
            for i in range(s):
                for j in range(i, s):
                    value = 0.0
                    for site in range(start, stop):
                        if not observed[sample, site] or not informative[site]:
                            continue
                        term = 6 + code[i, site] if i == j and i < k else pair_term[3 * code[i, site] + code[j, site]]
                        value += terms[site - start, term]
                    emission[sample, b, i, j] = value
                    emission[sample, b, j, i] = value
    return emission, background


@kernel(
    overloads=(
        (
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.float64,
        ),
    ),
    cache=True,
)
def _redraw_forward(x, pi, r):
    s = len(pi)
    out = np.empty_like(x)
    row = x.sum(axis=1)
    col = x.sum(axis=0)
    total = x.sum()
    for i in range(s):
        for j in range(s):
            out[i, j] = (
                (1 - r) ** 2 * x[i, j] + r * (1 - r) * (row[i] * pi[j] + pi[i] * col[j]) + r * r * pi[i] * pi[j] * total
            )
    return out


@kernel(
    overloads=(
        (
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.float64,
        ),
    ),
    cache=True,
)
def _redraw_backward(x, pi, r):
    s = len(pi)
    out = np.empty_like(x)
    row = x @ pi
    col = pi @ x
    total = pi @ row
    for i in range(s):
        for j in range(s):
            out[i, j] = (1 - r) ** 2 * x[i, j] + r * (1 - r) * (row[i] + col[j]) + r * r * total
    return out


@kernel(
    overloads=tuple(
        (
            types.Array(types.float64, 4, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=frozen),
            types.float64,
            types.float64,
            types.float64,
        )
        for frozen in (False, True)
    ),
    cache=True,
    parallel=True,
)
def _carrier_weights(weights, intervals_morgans, rate, background_mass, count_band):
    """Per sample and bin, the normalised cavity weights (bin emission excluded) of the ordered
    state pairs (forward-backward over bins, each step normalised), written over ``weights``,
    which holds the summed log emissions on entry (a sample's are read before its cavity weights
    replace them, so one (N, B, S, S) array serves both). Returns per sample and bin the MAP
    number of background copies (0, 1, 2): release 2.2, the smallest count among the state pairs
    whose posterior weight is within ``count_band`` x the largest of it (the conservative side,
    as the release's unknown-copy count; 2.1 took the first maximum in row-major pair order, a
    convention)."""
    n, bins, s, _ = weights.shape
    pi = np.full(s, (1 - background_mass) / (s - 1))
    pi[-1] = background_mass
    slots = np.empty((n, bins), dtype=np.int8)
    for sample in prange(n):
        predicted = np.empty((bins, s, s))
        emit = np.empty((bins, s, s))
        x = np.outer(pi, pi)
        for b in range(bins):
            if b:
                x = _redraw_forward(x, pi, -np.expm1(-(rate * intervals_morgans[b - 1])))
            predicted[b] = x
            emit[b] = np.exp(weights[sample, b] - np.max(weights[sample, b]))
            x = x * emit[b]
            x /= x.sum()
        beta = np.ones((s, s))
        for b in range(bins - 1, -1, -1):
            w = predicted[b] * beta
            w /= w.sum()
            weights[sample, b] = w
            w = w * emit[b]
            w /= w.sum()
            best = w[0, 0]
            for i in range(s):
                for j in range(s):
                    if w[i, j] > best:
                        best = w[i, j]
            floor = best - count_band * best
            count = 2
            for i in range(s):
                for j in range(s):
                    if w[i, j] >= floor:
                        value = (1 if i == s - 1 else 0) + (1 if j == s - 1 else 0)
                        if value < count:
                            count = value
            slots[sample, b] = count
            if b:
                beta = _redraw_backward(beta * emit[b], pi, -np.expm1(-(rate * intervals_morgans[b - 1])))
                beta /= beta.sum()
    return slots


@kernel(
    overloads=(
        (
            types.Array(types.int8, 2, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 3, "C", readonly=False),
            types.Array(types.boolean, 2, "C", readonly=False),
            types.Array(types.float64, 3, "A", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.int64,
            types.int64,
            types.float64,
            types.float64,
            types.Array(types.int64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.int64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
        ),
    ),
    cache=True,
    parallel=True,
)
def _refit_sites(
    result,
    probabilities,
    supporters,
    gl,
    observed,
    cavity,
    background,
    start,
    stop,
    threshold,
    min_support,
    site_info,
    site_boundary,
    site_withheld,
    site_bands,
    settings,
):
    """Refit and release the alleles of sites [start, stop) of one bin. Release 2.1: the support
    test is ``count >= min_support`` or ``mass > min_support + band`` (a carrier mass sitting
    exactly at the threshold, a sum of exact halves by the refit's symmetric founder prior, is
    withheld inside the band, design A-7); site_bands[0, site] counts the carrier decisions inside
    the 1/2 band, site_bands[1, site] the founders whose support mass lies inside the threshold
    band (count below the threshold)."""
    band = settings[SETTING_HALF_BAND]
    for site in prange(start, stop):
        q, count, mass, shell, info, boundary, inside = site_release_marginals(
            gl[:, site], observed[:, site], cavity, background[site], settings
        )
        probabilities[:, site] = q
        supporters[:, site] = np.maximum(count, mass)
        site_info[:, site] = info
        site_boundary[site] = boundary
        site_bands[0, site] = inside
        mass_band = 0
        for i in range(len(result)):
            if count[i] < min_support and abs(mass[i] - min_support) <= band:
                mass_band += 1
        site_bands[1, site] = mass_band
        if info[INFO_MULTIMODAL]:
            # Two comparable, unconnected posterior basins: release nothing here.
            site_withheld[site] = 0
            continue
        withheld = 0
        for i in range(len(result)):
            if count[i] >= min_support or mass[i] > min_support + band:
                if q[i] >= threshold or 1 - q[i] >= threshold:
                    if guard_withholds(q[i], shell[i], threshold, settings):
                        withheld += 1
                    elif q[i] >= threshold:
                        result[i, site] = 1
                    else:
                        result[i, site] = 0
        site_withheld[site] = withheld


def _consensus_rows(released, inverse, count) -> np.ndarray:
    """int8[count, L]: per distinct row (``inverse`` maps each original row to it) the call on which
    all its original rows' released calls agree (uncalled rows abstain), -1 where called rows differ
    or none is called."""
    released = np.asarray(released, dtype=np.int8)
    alt = np.zeros((count, released.shape[1]), dtype=np.int64)
    ref = np.zeros((count, released.shape[1]), dtype=np.int64)
    np.add.at(alt, inverse, (released == 1).astype(np.int64))
    np.add.at(ref, inverse, (released == 0).astype(np.int64))
    out = np.full((count, released.shape[1]), -1, dtype=np.int8)
    out[(alt > 0) & (ref == 0)] = 1
    out[(ref > 0) & (alt == 0)] = 0
    return np.ascontiguousarray(out)


def _combine_duplicates(posterior, support, inverse, count):
    """(posterior, support) per distinct local row: summed support, support-weighted mean posterior
    (the plain mean where the support is zero)."""
    total = np.zeros((count, posterior.shape[1]))
    weighted = np.zeros((count, posterior.shape[1]))
    plain = np.zeros((count, posterior.shape[1]))
    members = np.bincount(inverse, minlength=count).astype(np.float64)
    for row, target in enumerate(inverse):
        total[target] += support[row]
        weighted[target] += support[row] * posterior[row]
        plain[target] += posterior[row]
    mean = plain / members[:, None]
    return np.where(total > 0, weighted / np.where(total > 0, total, 1.0), mean), total


def refit_task_bytes(samples: int, sites: int, blocks: int, founders: int) -> int:
    """Planned peak additional bytes of one component refit: its sites' GL and mask slices,
    the (N, B, S, S) emission / cavity array, each thread's (B, S, S) forward and emission
    buffers, the refitted arrays and the site-posterior work (S = founders + 1). Worker startup and
    retained private memory are held separately for the pool lifetime."""
    s = int(founders) + 1
    slices = int(samples) * int(sites) * (3 * 8 + 1)
    grid = 8 * int(samples) * int(blocks) * s * s
    threads = 2 * 8 * int(blocks) * s * s * 4
    outputs = 3 * 8 * int(founders) * int(sites)
    return slices + grid + threads + outputs


def refit_component(
    h: np.ndarray,
    gl: np.ndarray,
    observed: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    intervals_morgans: np.ndarray,
    rate: float,
    config: ContextRefitConfig,
    unknown_mass: float,
    census_key: str | None = None,
    *,
    reverse: bool,
    census_sink: CensusSink | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int | float | dict[int, int]], np.ndarray]:
    """(refitted calls, ALT posterior, carrier support, per-bin MAP background-copy counts
    (N, B), site-posterior report, released) of one component's founders ``h`` over its blocks' sites; the refitted
    calls are the refit's releases where it released (``released``, bool) and ``h`` elsewhere.
    ``census_key``: the component's key for the valley census of its carrier probabilities near
    1/2 (diagnostics only)."""
    cavity, background = _emission_grid(h, gl, observed, starts, stops, float(config.emission_mix))
    slots = _carrier_weights(cavity, intervals_morgans, float(rate), float(unknown_mass), BACKGROUND_COUNT_BAND)
    if census_key is not None and census_enabled(census_sink):
        k = len(h)
        carried = (
            cavity[:, :, :k, :].sum(axis=3)
            + cavity[:, :, :, :k].sum(axis=2)
            - np.diagonal(cavity, axis1=2, axis2=3)[:, :, :k]
        )
        for sample, bin_index, founder in np.argwhere(np.abs(carried - 0.5) <= HALF_CENSUS_WINDOW):
            census(
                "ls.refit.half.band",
                f"{census_key}/b{bin_index}/n{sample}/f{founder}",
                (0.0, float(carried[sample, bin_index, founder] - 0.5)),
                sink=census_sink,
            )
    # The kernel writes only the calls it releases; the rest keep h below (release 2.12: the release mask is kept).
    result = np.full(h.shape, -1, dtype=h.dtype)
    probabilities = np.full(h.shape, 0.5)
    supporters = np.zeros(h.shape)
    site_info = np.zeros((INFO_FIELDS, h.shape[1]), np.int64)
    site_boundary = np.zeros(h.shape[1])
    site_withheld = np.zeros(h.shape[1], np.int64)
    site_bands = np.zeros((2, h.shape[1]), np.int64)
    settings = SitePosteriorParameters().vector(reverse)
    for b in range(len(starts)):
        _refit_sites(
            result,
            probabilities,
            supporters,
            gl,
            observed,
            cavity[:, b],
            background,
            starts[b],
            stops[b],
            float(config.posterior_threshold),
            float(config.min_support),
            site_info,
            site_boundary,
            site_withheld,
            site_bands,
            settings,
        )
    released = result >= 0
    result = np.where(released, result, h).astype(h.dtype)
    report = summarize_sites(site_info, site_boundary, site_withheld)
    report["half_band_sites"] = int(np.count_nonzero(site_bands[0]))
    report["support_band_sites"] = int(np.count_nonzero(site_bands[1]))
    return result, probabilities, supporters, slots, report, released


@dataclass(frozen=True)
class ContextRefitTask:
    """One component's immutable context-refit boundary (LS26); arrays are read-only inputs.

    h/template int8[K,L], columns int64[L], starts/stops int64[B], intervals float64[B-1]
    in Morgans, carriers float64[K]. members indexes the input LocalPanels.
    """

    index: int = record_field("state", dtype="int", units="component")
    members: tuple[int, ...] = record_field("state", axes=("B",), dtype="tuple[int]", units="block")
    h: np.ndarray = record_field("state", axes=("K", "L_component"), dtype="int8", units="allele")
    columns: np.ndarray = record_field("state", axes=("L_component",), dtype="int64", units="site")
    starts: np.ndarray = record_field("state", axes=("B",), dtype="int64", units="site")
    stops: np.ndarray = record_field("state", axes=("B",), dtype="int64", units="site")
    intervals_morgans: np.ndarray = record_field("state", axes=("B_minus_one",), dtype="float64", units="Morgans")
    rate: float = record_field("state", dtype="float64", units="1/Morgan")
    config: ContextRefitConfig = record_field("state", dtype="ContextRefitConfig")
    unknown_mass: float = record_field("state", dtype="float64", units="probability")
    census_key: str | None = record_field("diagnostic", dtype="str | None")
    template: np.ndarray = record_field("state", axes=("K", "L_component"), dtype="int8", units="allele")
    block_ids: tuple[int, ...] = record_field("state", axes=("B",), dtype="tuple[int]", units="block")
    carriers: np.ndarray = record_field("state", axes=("K",), dtype="float64", units="count")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


def finish_component(task: ContextRefitTask, outputs, observed: np.ndarray):
    """Finish one component into its member blocks without scheduling or evidence I/O.

    observed is bool[N,L_component]. The refit owns outputs and this function fills
    its withheld calls from the released template. Input task arrays are unchanged.
    Returns the component index, block proposals, changed/fill/site/founder counts,
    and the exact AS2 report. Duplicate inference/local rows retain the 3.5 rules.
    """
    result, q, support, slots, report, released = outputs
    template = task.template
    block_ids, starts, stops, carriers = task.block_ids, task.starts, task.stops, task.carriers
    index = task.index
    # A withheld allele keeps the component's published (completion-filled) call.
    fallback = (result < 0) & (template >= 0)
    result[fallback] = template[fallback]
    changes = int(((result >= 0) & (template >= 0) & (result != template)).sum())
    fills = int(((result >= 0) & (template < 0)).sum())
    proposals = []
    for block, lo, hi, block_slots in zip(block_ids, starts, stops, slots.T, strict=True):
        rows = result[:, lo:hi]
        # Distinct local rows in content order. Founders whose local rows are equal carry the same local
        # haplotype: its support is their summed support and its ALT posterior their support-weighted mean
        # (release 2.2: order-free; 2.1 kept the first founder's, a convention).
        local, inverse = unique_rows(rows, return_inverse=True)
        posterior_alt, local_support = _combine_duplicates(q[:, lo:hi], support[:, lo:hi], inverse, len(local))
        # A distinct local row's call is witnessed when the refit released it for one of its founders (equal rows
        # carry equal calls).
        witnessed = np.zeros(local.shape, dtype=np.bool_)
        np.logical_or.at(witnessed, inverse, released[:, lo:hi])
        witnessed &= local >= 0
        depth = np.any(observed[:, lo:hi], axis=1)
        # Assembled founders per distinct local row (``carriers``: per refitted row, its duplicate inference rows).
        multiplicity = np.bincount(inverse, weights=carriers, minlength=len(local)).astype(np.int64)
        proposals.append(
            ContextProposal(
                block=block,
                covered=True,
                calls=np.ascontiguousarray(local),
                posterior_alt=posterior_alt,
                support=local_support,
                unknown_slots=np.where(depth, block_slots, 2).astype(np.int8),
                multiplicity=multiplicity,
                witnessed=witnessed,
            )
        )
    return index, proposals, changes, fills, int(result.shape[1]), len(result), report


def refit_tasks(
    panels: Sequence[LocalPanel],
    components: Sequence[ComponentPanel],
    observed: np.ndarray,
    positions: np.ndarray,
    genetic_map: ChromosomeGeneticMap,
    parameters: SelectionParameters,
    *,
    census_key: str | None = None,
) -> tuple[tuple[ContextRefitTask, ...], int]:
    """Describe the completely covered context components in original component order.

    positions is int64[L_contig] in bp; observed is bool[N,L_contig]. No arrays are
    mutated. Coverage compares concatenated complete block positions exactly;
    partial components increment skipped. Centres remain mean bp coordinates and
    L11 forms their interval Morgans once. Returns immutable task records and skips.
    """
    evidence_positions = positions
    refit_config = parameters.context_refit
    unknown_mass = parameters.model.unknown_mass
    redraw_rate = parameters.model.redraw_rate_per_morgan
    first_positions = np.array([panel.positions[0] for panel in panels])
    last_positions = np.array([panel.positions[-1] for panel in panels])
    tasks, skipped = [], 0
    for index, component in enumerate(components):
        positions = np.asarray(component.positions)
        members = np.flatnonzero((first_positions >= positions[0]) & (last_positions <= positions[-1]))
        if not len(members):
            continue
        if not np.array_equal(np.concatenate([panels[int(b)].positions for b in members]), positions):
            skipped += 1
            continue
        inference = np.asarray(component.inference, np.int8)
        # Distinct founder rows in content order; the released template of duplicate inference
        # rows is their consensus (release 2.2: an allele where every duplicate's released call
        # agrees, -1 elsewhere; 2.1 kept the first occurrence's, a convention).
        h, inverse = unique_rows(inference, return_inverse=True)
        h = np.ascontiguousarray(h)
        template = _consensus_rows(np.asarray(component.released), inverse, len(h))
        lengths = [panels[int(b)].sites for b in members]
        starts = np.r_[0, np.cumsum(lengths)[:-1]].astype(np.int64)
        stops = np.r_[starts[1:], len(positions)].astype(np.int64)
        centers = np.array([np.mean(panels[int(b)].positions) for b in members])
        # L11: one interval-Morgans form; centres remain in base pairs.
        # Do not subtract separately scaled bp coordinates: that changes cancellation.
        intervals_morgans = np.asarray(genetic_map.interval_morgans(centers[:-1], centers[1:]), dtype=np.float64)
        rate = float(redraw_rate)
        columns = np.searchsorted(evidence_positions, positions)
        tasks.append(
            ContextRefitTask(
                index=index,
                members=tuple(int(b) for b in members),
                h=h,
                columns=columns,
                starts=starts,
                stops=stops,
                intervals_morgans=intervals_morgans,
                rate=rate,
                config=refit_config,
                unknown_mass=unknown_mass,
                census_key=census_key,
                template=template,
                block_ids=tuple(panels[int(b)].block for b in members),
                carriers=np.bincount(inverse, minlength=len(h)).astype(np.float64),
            )
        )
    return tuple(tasks), skipped
