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

Release 2.1 (noise stability): the component's distinct founder rows enter the refit, and each
block's proposal rows leave it, in canonical content (lexicographic) order, so neither the
refit's arithmetic nor the recorded proposals depend on the component's founder order.

Release 2.2 (no conventions): duplicate founder rows share the consensus of their released
templates (-1 where called templates differ); duplicate local rows of a block carry their summed support and
support-weighted ALT posterior; a bin's MAP background-copy count is the smallest count among the
state pairs within BACKGROUND_COUNT_BAND of the largest weight. 2.1 took the first occurrence and
the first maximum.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from numba import njit, prange

from ..assembly.founder_alleles import site_posterior
from ..execution import BudgetedPool, claim_threads, memory_worker_limit, reserve_memory
from ..maths.ties import census, census_enabled
from .fit import unique_rows

# MAP background-copy count band of the carrier weights (release 2.2): the probability tie quantum
# of maths.ties (2^-20, as the release's unknown_count_band); a larger count must win by more than
# this relative weight.
BACKGROUND_COUNT_BAND = 2.0 ** -20
# Census window of the carrier-probability band at 1/2 (diagnostics only).
HALF_CENSUS_WINDOW = 1e-3

# Private process/startup allowance: historical first-task incremental RSS reached 362 MB. Unlike the old
# 256 MiB per-task kernel allowance, this remains reserved for the complete lifetime of each pool worker.
# Sequential K30 post-trim private memory reached 116 MB; this also covers the supported cold path.
WORKER_RESIDENT_BYTES = 512 << 20

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
def _carrier_weights(weights, centers, rate, background_mass, count_band):
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
                beta = _redraw_backward(beta * emit[b], pi, -np.expm1(-rate * (centers[b] - centers[b - 1])))
                beta /= beta.sum()
    return slots


@njit(cache=True, parallel=True)
def _refit_sites(result, probabilities, supporters, gl, observed, cavity, background, start, stop,
                 threshold, min_support, site_info, site_boundary, site_withheld, site_bands, settings):
    """Refit and release the alleles of sites [start, stop) of one bin. Release 2.1: the support
    test is ``count >= min_support`` or ``mass > min_support + band`` (a carrier mass sitting
    exactly at the threshold, a sum of exact halves by the refit's symmetric founder prior, is
    withheld inside the band, design A-7); site_bands[0, site] counts the carrier decisions inside
    the 1/2 band, site_bands[1, site] the founders whose support mass lies inside the threshold
    band (count below the threshold)."""
    band = settings[site_posterior.SETTING_HALF_BAND]
    for site in prange(start, stop):
        q, count, mass, shell, info, boundary, inside = site_posterior.site_release_marginals(
            gl[:, site], observed[:, site], cavity, background[site], settings)
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
        if info[site_posterior.INFO_MULTIMODAL]:
            # Two comparable, unconnected posterior basins: release nothing here.
            site_withheld[site] = 0
            continue
        withheld = 0
        for i in range(len(result)):
            if count[i] >= min_support or mass[i] > min_support + band:
                if q[i] >= threshold or 1 - q[i] >= threshold:
                    if site_posterior.guard_withholds(q[i], shell[i], threshold, settings):
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


def refit_component(h, gl, observed, starts, stops, centers, rate, config, unknown_mass, census_key=None):
    """(refitted calls, ALT posterior, carrier support, per-bin MAP background-copy counts
    (N, B), site-posterior report) of one component's founders ``h`` over its blocks' sites.
    ``census_key``: the component's key for the valley census of its carrier probabilities near
    1/2 (diagnostics only)."""
    claim_threads()
    cavity, background = _emission_grid(h, gl, observed, starts, stops,
                                        float(config.emission_mix))
    claim_threads()
    slots = _carrier_weights(cavity, centers, float(rate), float(unknown_mass), BACKGROUND_COUNT_BAND)
    if census_key is not None and census_enabled():
        k = len(h)
        carried = (cavity[:, :, :k, :].sum(axis=3) + cavity[:, :, :, :k].sum(axis=2)
                   - np.diagonal(cavity, axis1=2, axis2=3)[:, :, :k])
        for sample, bin_index, founder in np.argwhere(np.abs(carried - .5) <= HALF_CENSUS_WINDOW):
            census("ls.refit.half.band", f"{census_key}/b{bin_index}/n{sample}/f{founder}",
                   (0.0, float(carried[sample, bin_index, founder] - .5)))
    result = h.copy()
    probabilities = np.full(h.shape, .5)
    supporters = np.zeros(h.shape)
    site_info = np.zeros((site_posterior.INFO_FIELDS, h.shape[1]), np.int64)
    site_boundary = np.zeros(h.shape[1])
    site_withheld = np.zeros(h.shape[1], np.int64)
    site_bands = np.zeros((2, h.shape[1]), np.int64)
    settings = site_posterior.settings()
    for b in range(len(starts)):
        claim_threads()
        _refit_sites(result, probabilities, supporters, gl, observed, cavity[:, b], background,
                     starts[b], stops[b], float(config.posterior_threshold),
                     float(config.min_support), site_info, site_boundary, site_withheld, site_bands,
                     settings)
    report = site_posterior.summarize_sites(site_info, site_boundary, site_withheld)
    report["half_band_sites"] = int(np.count_nonzero(site_bands[0]))
    report["support_band_sites"] = int(np.count_nonzero(site_bands[1]))
    return result, probabilities, supporters, slots, report


@dataclass(frozen=True)
class ContextProposal:
    """One block's proposal of a context round.

    covered: a context component covered the block; then ``calls`` are the distinct local rows
    of the component's refitted founders (lexicographic order; -1 uncalled), which are also
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
    """Pool task: one component, refitted and finished into its blocks' proposals (worker side): returns
    (index, proposals of its member blocks in order, allele changes, fills, sites, founders, report)."""
    (index, h, columns, starts, stops, centers, rate, config, unknown_mass, census_key, template,
     block_ids) = task
    gl = np.ascontiguousarray(shared["likelihoods"][:, columns], dtype=np.float64)
    observed = np.ascontiguousarray(shared["observed"][:, columns])
    result, q, support, slots, report = refit_component(
        h, gl, observed, starts, stops, centers, rate, config, unknown_mass,
        census_key=None if census_key is None else f"{census_key}/c{index}")
    # A withheld allele keeps the component's published (completion-filled) call.
    fallback = (result < 0) & (template >= 0)
    result[fallback] = template[fallback]
    changes = int(((result >= 0) & (template >= 0) & (result != template)).sum())
    fills = int(((result >= 0) & (template < 0)).sum())
    proposals = []
    for block, lo, hi, block_slots in zip(block_ids, starts, stops, slots.T):
        rows = result[:, lo:hi]
        # Distinct local rows in content order. Founders whose local rows are equal carry the same local
        # haplotype: its support is their summed support and its ALT posterior their support-weighted mean
        # (release 2.2: order-free; 2.1 kept the first founder's, a convention).
        local, inverse = unique_rows(rows, return_inverse=True)
        posterior_alt, local_support = _combine_duplicates(q[:, lo:hi], support[:, lo:hi], inverse, len(local))
        depth = np.any(observed[:, lo:hi], axis=1)
        proposals.append(ContextProposal(
            block=block, covered=True, calls=np.ascontiguousarray(local), posterior_alt=posterior_alt,
            support=local_support, unknown_slots=np.where(depth, block_slots, 2).astype(np.int8)))
    return index, proposals, changes, fills, int(result.shape[1]), len(result), report


@dataclass(frozen=True)
class RefitSummary:
    """components refitted, components skipped (block coverage incomplete), released alleles
    that changed a published call, alleles newly called, site-posterior totals, largest K;
    tie_events (release 2.1, maths.ties TieCounter record; one decision per refitted site): sites
    with a band decision of the release ("ls.refit.half.band": a carrier probability within the
    band of an exact 1/2; "ls.refit.support.band": a founder's support mass within the band of the
    threshold, its count below it)."""

    components: int
    skipped: int
    allele_changes: int
    filled_alleles: int
    site_posterior: tuple[tuple[str, int], ...]
    max_founders: int
    tie_events: dict | None = None


_TOTALS = ("cube_sites", "core_cube_sites", "core_ball_sites", "bp_nonconverged",
           "ascent_backstop_hits", "recentre_backstop_hits", "complement_mode_sites",
           "multimodal_abstained", "guard_withheld")


def refine(panels, components, evidence, genetic_map, config, redraw_rate, budget, census_key=None):
    """Context proposals of every panel (one per LocalPanel, same order) and a summary.

    panels: the current round's LocalPanels; components: the context assembly's materialised
    ComponentPanels (first/stop index ``panels``); evidence: the founder-view ContigEvidence;
    census_key: the round's key for the valley census (diagnostics only).
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
        # Distinct founder rows in content order; the released template of duplicate inference
        # rows is their consensus (release 2.2: an allele where every duplicate's released call
        # agrees, -1 elsewhere; 2.1 kept the first occurrence's, a convention).
        h, inverse = unique_rows(inference, return_inverse=True)
        h = np.ascontiguousarray(h)
        template = _consensus_rows(np.asarray(component.released), inverse, len(h))
        lengths = [panels[b].sites for b in members]
        starts = np.r_[0, np.cumsum(lengths)[:-1]].astype(np.int64)
        stops = np.r_[starts[1:], len(positions)].astype(np.int64)
        centers = np.array([np.mean(panels[b].positions) for b in members])
        if genetic_map.has_map:
            # Rate per Morgan; the map is integrated between block centres (with its tails).
            centers = genetic_map.cumulative_morgans(centers)
            rate = float(redraw_rate)
        else:
            rate = genetic_map.fallback_rate_per_bp * redraw_rate
        columns = np.searchsorted(evidence.positions, positions)
        specs[index] = members
        tasks.append((index, h, columns, starts, stops, centers, rate, refit_config, unknown_mass, census_key,
                      template, tuple(panels[b].block for b in members)))
    proposals = [ContextProposal(block=panel.block, covered=False, calls=panel.calls)
                 for panel in panels]
    changes = fills = 0
    totals = dict.fromkeys(_TOTALS, 0)
    bands = dict(half=0, support=0, sites=0)
    max_founders = 0
    if tasks:
        samples = evidence.likelihoods.shape[0]
        sizes = [refit_task_bytes(samples, len(task[2]), len(task[3]), len(task[1])) for task in tasks]
        planned = [size + WORKER_RESIDENT_BYTES for size in sizes]
        # One shared evidence copy for the pool, counted once in the largest concurrent group.
        largest = max(range(len(planned)), key=planned.__getitem__)
        planned[largest] += evidence.likelihoods.nbytes + observed.nbytes
        workers = memory_worker_limit(planned, min(int(budget.threads()), len(tasks)), label="context-refit")
        with BudgetedPool(budget, processes=workers,
                          shared=dict(likelihoods=evidence.likelihoods, observed=observed),
                          label="context-refit") as pool:
            # Each task returns its blocks' finished proposals: the coordinator only places them and adds counts.
            for index, finished, changed, filled, sites, founders, report in pool.imap_unordered(
                    _refit_task, tasks, task_bytes=sizes):
                changes += changed
                fills += filled
                for key in _TOTALS:
                    totals[key] += int(report[key])
                bands["half"] += int(report["half_band_sites"])
                bands["support"] += int(report["support_band_sites"])
                bands["sites"] += sites
                max_founders = max(max_founders, founders)
                for b, proposal in zip(specs[index], finished):
                    proposals[b] = proposal
    tie_events = {"ls.refit.half.band": dict(decisions=bands["sites"], inside_tie=bands["half"], wide=0, boundary=0),
                  "ls.refit.support.band": dict(decisions=bands["sites"], inside_tie=bands["support"], wide=0,
                                                boundary=0)}
    summary = RefitSummary(components=len(tasks), skipped=skipped, allele_changes=changes,
                           filled_alleles=fills, site_posterior=tuple(totals.items()),
                           max_founders=max_founders, tie_events=tie_events)
    print(f"[Context refit] {len(tasks)} components, {skipped} skipped; allele changes {changes}, "
          f"fills {fills}; site posterior " + " ".join(f"{k}={v}" for k, v in totals.items())
          + f" max_founders {max_founders}", flush=True)
    return proposals, summary
