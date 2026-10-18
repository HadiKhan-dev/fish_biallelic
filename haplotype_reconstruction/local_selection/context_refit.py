"""Context refit: an assembled context component's founders, refitted from each covered block's
own evidence, become that block's proposals (candidates, never observations).

For a component covering consecutive blocks, the distinct pre-fill rows (its founders; -1 =
uncalled) define a bin-level diploid HMM per sample: states are ordered pairs of founders plus a
background state (alleles Bernoulli(background frequency) per copy), one bin per block, redraw
transitions toward pi = ((1 - u)/K, ..., u) at rate 1 - exp(-rate x distance between block
centres) (Morgans with a genetic map file, else bp x fallback rate). A pair's bin emission sums,
over the bin's observed discovery sites, the log of a mix-robust genotype likelihood
((1 - mix) GL(pair's allele probabilities) + mix / 3). The *cavity* weights of block b exclude
block b's own emission; with them, the shared founder-allele site posterior of the assembly
library (exact for at most ten founders, polynomial core above) refits every allele from the
block's own reads, released at ``posterior_threshold`` with carrier support ``min_support`` (none
where the posterior is multimodal or the ball guard withholds it). A withheld allele keeps the
component's published (completion-filled) call when it has one.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from numba import njit, prange

from ..assembly.founder_alleles import site_posterior
from ..execution import BudgetedPool, claim_threads
from .fit import unique_rows

# Per site, a copy state's allele probability is 0 or 1 (a called founder) or the site
# background (an uncalled founder or the background state): codes 0, 1, 2. The emission term of
# a state pair depends only on its unordered code pair, terms 0-5 for (0, 0), (0, 1), (0, 2),
# (1, 1), (1, 2), (2, 2), or, for a founder paired with itself (both copies one founder), on its
# code: terms 6, 7, 8.
_EMISSION_TERMS = 9
_TERM_LOW = (0, 0, 0, 1, 1, 2)
_TERM_HIGH = (0, 1, 2, 1, 2, 2)
_CODE_PAIR_TERM = (0, 1, 2, 1, 3, 4, 2, 4, 5)


@njit(cache=True, inline="always")
def _code_probability(code, background):
    return background if code == 2 else float(code)


@njit(cache=True, parallel=True)
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
                        term = (6 + code[i, site] if i == j and i < k
                                else pair_term[3 * code[i, site] + code[j, site]])
                        value += terms[site - start, term]
                    emission[sample, b, i, j] = value
                    emission[sample, b, j, i] = value
    return emission, background


@njit(cache=True)
def _redraw_forward(x, pi, r):
    s = len(pi)
    out = np.empty_like(x)
    row = x.sum(axis=1)
    col = x.sum(axis=0)
    total = x.sum()
    for i in range(s):
        for j in range(s):
            out[i, j] = ((1 - r) ** 2 * x[i, j] + r * (1 - r) * (row[i] * pi[j] + pi[i] * col[j])
                         + r * r * pi[i] * pi[j] * total)
    return out


@njit(cache=True)
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


@njit(cache=True, parallel=True)
def _carrier_weights(weights, centers, rate, background_mass):
    """Per sample and bin, the normalised cavity weights (bin emission excluded) of the ordered
    state pairs (forward-backward over bins, each step normalised), written over ``weights``,
    which holds the summed log emissions on entry (a sample's are read before its cavity weights
    replace them, so one (N, B, S, S) array serves both). Returns per sample and bin the MAP
    pair's number of background copies (0, 1, 2; the first maximum of the posterior weights in
    row-major pair order)."""
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
                x = _redraw_forward(x, pi, -np.expm1(-rate * (centers[b] - centers[b - 1])))
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
            best, best_i, best_j = w[0, 0], 0, 0
            for i in range(s):
                for j in range(s):
                    if w[i, j] > best:
                        best, best_i, best_j = w[i, j], i, j
            slots[sample, b] = (1 if best_i == s - 1 else 0) + (1 if best_j == s - 1 else 0)
            if b:
                beta = _redraw_backward(beta * emit[b], pi, -np.expm1(-rate * (centers[b] - centers[b - 1])))
                beta /= beta.sum()
    return slots


@njit(cache=True, parallel=True)
def _refit_sites(result, probabilities, supporters, gl, observed, cavity, background, start, stop,
                 threshold, min_support, site_info, site_boundary, site_withheld, settings):
    for site in prange(start, stop):
        q, support, shell, info, boundary = site_posterior.site_marginals(
            gl[:, site], observed[:, site], cavity, background[site], settings)
        probabilities[:, site] = q
        supporters[:, site] = support
        site_info[:, site] = info
        site_boundary[site] = boundary
        if info[site_posterior.INFO_MULTIMODAL]:
            # Two comparable, unconnected posterior basins: release nothing here.
            site_withheld[site] = 0
            continue
        withheld = 0
        for i in range(len(result)):
            if support[i] >= min_support:
                if q[i] >= threshold or 1 - q[i] >= threshold:
                    if site_posterior.guard_withholds(q[i], shell[i], threshold, settings):
                        withheld += 1
                    elif q[i] >= threshold:
                        result[i, site] = 1
                    else:
                        result[i, site] = 0
        site_withheld[site] = withheld


def refit_task_bytes(samples: int, sites: int, blocks: int, founders: int) -> int:
    """Planned peak additional bytes of one component refit: its sites' GL and mask slices,
    the (N, B, S, S) emission / cavity array, each thread's (B, S, S) forward and emission
    buffers, the refitted arrays and the site-posterior work (S = founders + 1), plus a fixed
    0.25 GiB for a worker's first task, which also loads the kernels (measured on actm3 chr8:
    first tasks up to 0.31 GB, later ones up to 0.07 GB)."""
    s = int(founders) + 1
    slices = int(samples) * int(sites) * (3 * 8 + 1)
    grid = 8 * int(samples) * int(blocks) * s * s
    threads = 2 * 8 * int(blocks) * s * s * 4
    outputs = 3 * 8 * int(founders) * int(sites)
    return slices + grid + threads + outputs + (256 << 20)


def refit_component(h, gl, observed, starts, stops, centers, rate, config, unknown_mass):
    """(refitted calls, ALT posterior, carrier support, per-bin MAP background-copy counts
    (N, B), site-posterior report) of one component's founders ``h`` over its blocks' sites."""
    claim_threads()
    cavity, background = _emission_grid(h, gl, observed, starts, stops,
                                        float(config.emission_mix))
    claim_threads()
    slots = _carrier_weights(cavity, centers, float(rate), float(unknown_mass))
    result = h.copy()
    probabilities = np.full(h.shape, .5)
    supporters = np.zeros(h.shape)
    site_info = np.zeros((site_posterior.INFO_FIELDS, h.shape[1]), np.int64)
    site_boundary = np.zeros(h.shape[1])
    site_withheld = np.zeros(h.shape[1], np.int64)
    settings = site_posterior.settings()
    for b in range(len(starts)):
        claim_threads()
        _refit_sites(result, probabilities, supporters, gl, observed, cavity[:, b], background,
                     starts[b], stops[b], float(config.posterior_threshold),
                     float(config.min_support), site_info, site_boundary, site_withheld, settings)
    return (result, probabilities, supporters, slots,
            site_posterior.summarize_sites(site_info, site_boundary, site_withheld))


@dataclass(frozen=True)
class ContextProposal:
    """One block's proposal of a context round.

    covered: a context component covered the block; then ``calls`` are the distinct local rows
    of the component's refitted founders (first-occurrence order; -1 uncalled), which are also
    the context founders of the path objective's code, ``posterior_alt``/``support`` the refit's
    ALT posterior and carrier support, ``unknown_slots`` the MAP number of background copies per
    sample (2 without depth). Otherwise ``calls`` are the block's current selected calls and the
    other fields None.
    """

    block: int
    covered: bool
    calls: np.ndarray
    posterior_alt: np.ndarray | None = None
    support: np.ndarray | None = None
    unknown_slots: np.ndarray | None = None

    @property
    def founders(self):
        return self.calls if self.covered else None


def _refit_task(task, shared, budget):
    """Pool task: one component (worker side)."""
    index, h, columns, starts, stops, centers, rate, config, unknown_mass = task
    gl = np.ascontiguousarray(shared["likelihoods"][:, columns], dtype=np.float64)
    observed = np.ascontiguousarray(shared["observed"][:, columns])
    started = time.monotonic()
    result, q, support, slots, report = refit_component(
        h, gl, observed, starts, stops, centers, rate, config, unknown_mass)
    return index, result, q, support, np.ascontiguousarray(slots.T), time.monotonic() - started, report


@dataclass(frozen=True)
class RefitSummary:
    """components refitted, components skipped (block coverage incomplete), released alleles
    that changed a published call, alleles newly called, site-posterior totals, largest K."""

    components: int
    skipped: int
    allele_changes: int
    filled_alleles: int
    site_posterior: tuple[tuple[str, int], ...]
    max_founders: int


_TOTALS = ("cube_sites", "core_cube_sites", "core_ball_sites", "bp_nonconverged",
           "ascent_backstop_hits", "recentre_backstop_hits", "complement_mode_sites",
           "multimodal_abstained", "guard_withheld")


def refine(panels, components, evidence, genetic_map, config, generations, budget):
    """Context proposals of every panel (one per LocalPanel, same order) and a summary.

    panels: the current round's LocalPanels; components: the context assembly's materialised
    ComponentPanels (first/stop index ``panels``); evidence: the founder-view ContigEvidence.
    """
    refit_config = config.context_refit
    unknown_mass = config.model.unknown_mass
    observed = np.array(evidence.observed, copy=True)
    for panel in panels:
        observed[:, panel.site_start:panel.site_start + panel.sites] &= panel.kept
    first_positions = np.array([panel.positions[0] for panel in panels])
    last_positions = np.array([panel.positions[-1] for panel in panels])
    specs, tasks, skipped = {}, [], 0
    for index, component in enumerate(components):
        positions = np.asarray(component.positions)
        members = np.flatnonzero((first_positions >= positions[0]) & (last_positions <= positions[-1]))
        if not len(members):
            continue
        if not np.array_equal(np.concatenate([panels[b].positions for b in members]), positions):
            skipped += 1
            continue
        inference = np.asarray(component.inference, np.int8)
        _, representatives = unique_rows(inference, return_index=True)
        representatives = np.sort(representatives)
        h = np.ascontiguousarray(inference[representatives])
        template = np.ascontiguousarray(np.asarray(component.released)[representatives])
        lengths = [panels[b].sites for b in members]
        starts = np.r_[0, np.cumsum(lengths)[:-1]].astype(np.int64)
        stops = np.r_[starts[1:], len(positions)].astype(np.int64)
        centers = np.array([np.mean(panels[b].positions) for b in members])
        if genetic_map.has_map:
            # Rate per Morgan; the map is integrated between block centres (with its tails).
            centers = genetic_map.cumulative_morgans(centers)
            rate = float(generations)
        else:
            rate = genetic_map.fallback_rate_per_bp * generations
        columns = np.searchsorted(evidence.positions, positions)
        specs[index] = (members, starts, stops, template)
        tasks.append((index, h, columns, starts, stops, centers, rate, refit_config, unknown_mass))
    proposals = [ContextProposal(block=panel.block, covered=False, calls=panel.calls)
                 for panel in panels]
    changes = fills = 0
    totals = dict.fromkeys(_TOTALS, 0)
    max_founders = 0
    if tasks:
        samples = evidence.likelihoods.shape[0]
        sizes = [refit_task_bytes(samples, len(task[2]), len(task[3]), len(task[1])) for task in tasks]
        with BudgetedPool(budget, processes=min(int(budget.threads()), len(tasks)),
                          shared=dict(likelihoods=evidence.likelihoods, observed=observed),
                          label="context-refit") as pool:
            for index, result, q, support, slots, _seconds, report in pool.imap_unordered(
                    _refit_task, tasks, task_bytes=sizes):
                members, starts, stops, template = specs[index]
                # A withheld allele keeps the component's published (completion-filled) call.
                fallback = (result < 0) & (template >= 0)
                result[fallback] = template[fallback]
                changes += int(((result >= 0) & (template >= 0) & (result != template)).sum())
                fills += int(((result >= 0) & (template < 0)).sum())
                for key in _TOTALS:
                    totals[key] += int(report[key])
                max_founders = max(max_founders, len(result))
                for b, lo, hi, block_slots in zip(members, starts, stops, slots):
                    rows = result[:, lo:hi]
                    _, unique = unique_rows(rows, return_index=True)
                    unique = np.sort(unique)
                    panel = panels[b]
                    depth = np.any(observed[:, panel.site_start:panel.site_start + panel.sites],
                                   axis=1)
                    proposals[b] = ContextProposal(
                        block=panel.block, covered=True,
                        calls=np.ascontiguousarray(rows[unique]),
                        posterior_alt=q[unique, lo:hi].copy(), support=support[unique, lo:hi].copy(),
                        unknown_slots=np.where(depth, block_slots, 2).astype(np.int8))
    summary = RefitSummary(components=len(tasks), skipped=skipped, allele_changes=changes,
                           filled_alleles=fills, site_posterior=tuple(totals.items()),
                           max_founders=max_founders)
    print(f"[Context refit] {len(tasks)} components, {skipped} skipped; allele changes {changes}, "
          f"fills {fills}; site posterior " + " ".join(f"{k}={v}" for k, v in totals.items())
          + f" max_founders {max_founders}", flush=True)
    return proposals, summary
