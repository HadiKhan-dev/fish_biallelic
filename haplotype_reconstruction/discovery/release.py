"""Raw-discovery allele release and the stage product of one block.

The selected panel's hard rows are latent: they are what local selection fits from. Its
*released* alleles carry a capped fixed-assignment pseudo-probability and are called only where
that probability reaches ``ReleaseRuleConfig.min_call_probability`` with at least
``min_supporters`` informative carriers. The same rule releases the candidate bank's source
endpoints in local selection (local_selection/bank.py), whose calls form the path-search start.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
from numba import njit, prange

from ..model.local_panels import unknown_copy_mass
from .panel import known_content_key, unknown_alleles, solution_tie_mask, assignment_tie_mask
from .cavity_kernels import _state_membership
from .search import BlockSearch, SearchSummary


@dataclass(frozen=True)
class DiscoveredBlock:
    """The raw-discovery result of one informative block (record ``discovery/blocks``).

    block: index into ``ContigEvidence.block_spans``; site_start: its first contig site.
    positions: int64[L] the block's site positions; kept: bool[L] its discovery sites.
    latent: int8[K, L_kept] the selected hard panel at the discovery sites, rows in canonical
        (lexicographic) order; local selection also fits its complete tied alternatives. K=0
        only when a search limit omitted every candidate before an incumbent existed.
    calls: int8[K, L] released alleles (-1 = withheld or not a discovery site).
    alt_probability: float64[K, L] capped fixed-assignment pseudo-probability of ALT
        (0.5 at sites that are not discovery sites).
    support: int32[K, L] informative carriers of either direction (0 off the discovery sites); release 2.1: a
        carrier counts when its contribution exceeds the block's ``cell`` tie quantum in absolute value (2.0: when it
        was not exactly zero), so symmetric evidence no longer becomes a supporter under rounding or noise.
    unknown_mass: float, the selected assignment's unknown-copy fraction over samples with depth.
    nll: float, the selected panel's binned switching-Viterbi negative log likelihood (nats).
    search: small typed summary of the search (scores per represented K, limits, counts).
    latent_unknown: bool[K, L_kept], True where no sample with positive tied-state
        membership of the founder observes the site: the founder's allele there is unknown; ``latent`` keeps a binary value only as the fit's
        initial value (no evidence depends on it) and ``calls`` withholds it. None for records before 2.2.
    tied_panels: tuple of int8[K_i, L_kept] (release 2.2), the tied alternatives of the released panel (the other
        panels of its K's first NLL tie class and other-K representatives in its score class; rows canonical like
        ``latent``), without those that equal the released panel, or another alternative, at every allele some
        assigned carrier observes (the same science); the order carries no meaning. tied_panel_nll: their NLLs;
        tied_unknown: tuple of bool[K_i, L_kept], their unknown alleles. Empty tuples without alternatives.
    """

    block: int
    site_start: int
    positions: np.ndarray
    kept: np.ndarray
    latent: np.ndarray
    calls: np.ndarray
    alt_probability: np.ndarray
    support: np.ndarray
    unknown_mass: float
    nll: float
    search: SearchSummary
    latent_unknown: np.ndarray | None = None
    tied_panels: tuple = ()
    tied_panel_nll: tuple = ()
    tied_unknown: tuple = ()

    @property
    def founders(self) -> int:
        return int(self.latent.shape[0])


def release_alleles(likelihoods, haplotypes, assignments, observed, unknown_copy_penalty,
                    rule, quanta=None, tie_mask=None):
    """Fixed-assignment founder/site evidence of a hard panel and its released alleles.

    likelihoods: float64[N, L, 3] normalised GLs (unobserved cells are ignored); haplotypes:
    int[K, L] binary rows; assignments: int[N, 2] sorted row pairs, K = unknown copy; observed:
    bool[N, L]. Costs follow the H step's buckets (homozygous carriers, carriers with a known
    partner, carriers with an unknown partner) capped per cell at the unknown-unknown cost; a
    carrier whose capped costs for allele 0 and 1 differ by more than the ``cell`` tie quantum
    (quanta: the block's BlockQuanta; default the floors) is an informative supporter.
    Tied states supplied by ``tie_mask`` share evidence uniformly. The cell band applies
    before weighting and a sample counts at most once per founder/site. Returns
    (alt_probability float64[K, L], supporters int64[K, L], calls int8[K, L]); log pseudo-odds
    NLL(0) - NLL(1) > 0 favour ALT.
    """
    return _release(likelihoods, haplotypes, assignments, observed, unknown_copy_penalty, rule, quanta, tie_mask)[:3]


def _release(likelihoods, haplotypes, assignments, observed, unknown_copy_penalty, rule, quanta=None, tie_mask=None):
    """release_alleles plus (contributions evaluated, contributions strictly inside the cell band)."""
    from ..maths.ties import BlockQuanta
    cell = (quanta if quanta is not None else BlockQuanta.floors()).cell
    k = haplotypes.shape[0]
    assignments = np.ascontiguousarray(assignments, dtype=np.int64)
    mask = assignment_tie_mask(assignments, k) if tie_mask is None else np.ascontiguousarray(tie_mask, dtype=np.uint8)
    pair_i, pair_j = np.triu_indices(k)
    kind = np.concatenate((np.zeros(len(pair_i), dtype=np.int64), np.ones(k, dtype=np.int64), [2]))
    first = np.concatenate((pair_i, np.arange(k), [-1])).astype(np.int64)
    second = np.concatenate((pair_j, np.full(k + 1, -1))).astype(np.int64)
    offsets, members, first, second = _state_membership(assignments, k, mask, kind, first, second)
    carriers = _release_carriers(k, offsets, members, first, second)
    probability, supporters, calls, evaluated, banded = _release_kernel(
        np.ascontiguousarray(likelihoods, dtype=np.float64),
        np.ascontiguousarray(haplotypes, dtype=np.int64),
        np.ascontiguousarray(observed, dtype=np.bool_),
        float(unknown_copy_penalty), int(rule.min_supporters),
        float(rule.min_call_probability), float(cell), *carriers)
    return probability, supporters, calls, int(evaluated.sum()), int(banded.sum())


@njit(cache=True)
def _release_carriers(k, offsets, members, first, second):
    """Founder-indexed weighted state terms, with each sample's terms adjacent."""
    counts = np.zeros(k, dtype=np.int64)
    for member in members:
        a, b = first[member], second[member]
        if a < k:
            counts[a] += 1
        if b < k and b != a:
            counts[b] += 1
    founder_offsets = np.zeros(k + 1, dtype=np.int64)
    for founder in range(k):
        founder_offsets[founder + 1] = founder_offsets[founder] + counts[founder]
    size = founder_offsets[-1]
    samples = np.empty(size, dtype=np.int64)
    partners = np.empty(size, dtype=np.int64)
    weights = np.empty(size, dtype=np.float64)
    cursor = founder_offsets[:-1].copy()
    for sample in range(len(offsets) - 1):
        share = 1.0 / (offsets[sample + 1] - offsets[sample])
        for index in range(offsets[sample], offsets[sample + 1]):
            member = members[index]
            a, b = first[member], second[member]
            if a < k:
                position = cursor[a]
                samples[position], partners[position], weights[position] = sample, b, share
                cursor[a] += 1
            if b < k and b != a:
                position = cursor[b]
                samples[position], partners[position], weights[position] = sample, a, share
                cursor[b] += 1
    return founder_offsets, samples, partners, weights


@njit(cache=True, parallel=True)
def _release_kernel(likelihoods, haplotypes, observed, penalty, min_supporters,
                    min_call_probability, cell_tie, offsets, samples, partners, weights):
    floor = 1e-12
    k, length = haplotypes.shape
    probability = np.full((k, length), 0.5, dtype=np.float64)
    supporters = np.zeros((k, length), dtype=np.int64)
    calls = np.full((k, length), -1, dtype=np.int8)
    evaluated = np.zeros(k, dtype=np.int64)
    banded = np.zeros(k, dtype=np.int64)
    for founder in prange(k):
        for site in range(length):
            support = 0
            evidence = 0.0
            last_supporter = -1
            for member in range(offsets[founder], offsets[founder + 1]):
                sample = samples[member]
                if not observed[sample, site]:
                    continue
                partner = partners[member]
                g0 = likelihoods[sample, site, 0]
                g1 = likelihoods[sample, site, 1]
                g2 = likelihoods[sample, site, 2]
                cost_cap = -math.log(max(g0, g1, g2, floor)) + 2.0 * penalty
                if partner == founder:
                    cost0 = -math.log(max(g0, floor))
                    cost1 = -math.log(max(g2, floor))
                elif partner < k:
                    partner_allele = haplotypes[partner, site]
                    cost0 = -math.log(max(likelihoods[sample, site, partner_allele], floor))
                    cost1 = -math.log(max(likelihoods[sample, site, partner_allele + 1], floor))
                else:
                    cost0 = -math.log(max(g0, g1, floor)) + penalty
                    cost1 = -math.log(max(g1, g2, floor)) + penalty
                contribution = min(cost0, cost_cap) - min(cost1, cost_cap)
                evaluated[founder] += 1
                if abs(contribution) > cell_tie:
                    evidence += weights[member] * contribution
                    if sample != last_supporter:
                        support += 1
                        last_supporter = sample
                elif contribution != 0.0:
                    banded[founder] += 1
            supporters[founder, site] = support
            if evidence >= 0.0:
                value = 1.0 / (1.0 + math.exp(-evidence))
            else:
                exponential = math.exp(evidence)
                value = exponential / (1.0 + exponential)
            probability[founder, site] = value
            if support >= min_supporters:
                if value >= min_call_probability:
                    calls[founder, site] = 1
                elif 1.0 - value >= min_call_probability:
                    calls[founder, site] = 0
    return probability, supporters, calls, evaluated, banded


def _shared_alternative_calls(haplotypes, unknown, probability, support, calls, alternatives):
    """Conservative calls for exact shared known rows; never choose a founder matching.

    A row absent from any complete alternative is withheld. If several rows share its known
    content, every one must support the same call. Probabilities and support use the least
    confident matching row, so a representative cannot strengthen the released evidence.
    """
    rows = np.where(unknown, -1, haplotypes).astype(np.int8)
    for panel, panel_unknown, panel_probability, panel_support, panel_calls in alternatives:
        other_rows = np.where(panel_unknown, -1, panel).astype(np.int8)
        for founder, row in enumerate(rows):
            matches = np.flatnonzero(np.all(other_rows == row, axis=1))
            if len(matches) == 0:
                calls[founder] = -1
                probability[founder] = 0.5
                support[founder] = 0
                continue
            agreed = (calls[founder] >= 0) & np.all(panel_calls[matches] == calls[founder], axis=0)
            alt = calls[founder] == 1
            conservative = np.where(alt, np.min(panel_probability[matches], axis=0),
                                     np.max(panel_probability[matches], axis=0))
            probability[founder] = np.where(
                agreed, np.where(alt, np.minimum(probability[founder], conservative),
                                 np.maximum(probability[founder], conservative)), 0.5)
            support[founder] = np.where(agreed, np.minimum(support[founder],
                                                         np.min(panel_support[matches], axis=0)), 0)
            calls[founder, ~agreed] = -1


def materialise(block: int, site_start: int, positions, kept, likelihoods, observed,
                search: BlockSearch, config) -> DiscoveredBlock:
    """The DiscoveredBlock of a searched block: the selected panel released by the raw rule.

    likelihoods/observed: the block's founder-view evidence at all its sites (L); the release
    uses the discovery sites only (unobserved cells are ignored by the release).
    """
    solution = search.selected
    length = len(positions)
    if solution is None:
        # A whole seed class was omitted before any incumbent existed.
        # No fitted biological answer is available to release or pass as a start.
        return DiscoveredBlock(
            block=int(block), site_start=int(site_start),
            positions=np.ascontiguousarray(positions, dtype=np.int64),
            kept=np.ascontiguousarray(kept, dtype=np.bool_),
            latent=np.empty((0, int(np.count_nonzero(kept))), dtype=np.int8),
            calls=np.empty((0, length), dtype=np.int8),
            alt_probability=np.full((0, length), 0.5),
            support=np.zeros((0, length), dtype=np.int32),
            unknown_mass=1.0, nll=math.inf, search=search.summary,
            latent_unknown=np.empty((0, int(np.count_nonzero(kept))), dtype=np.bool_))
    k = solution.k
    observed_kept = np.ascontiguousarray(observed[:, kept])
    mask = solution_tie_mask(solution)
    probability_kept, supporters_kept, calls_kept, evaluated, banded = _release(
        np.ascontiguousarray(likelihoods[:, kept]), solution.haplotypes, solution.assignments,
        observed_kept, config.model.unknown_copy_penalty, config.release, solution.quanta, mask)
    events = dict(search.summary.tie_events)
    events["discovery.release.cell_band"] = dict(decisions=evaluated, inside_tie=banded, wide=0, boundary=0)
    summary = replace(search.summary, tie_events=events)
    alt_probability = np.full((k, length), 0.5, dtype=np.float64)
    alt_probability[:, kept] = probability_kept
    support = np.zeros((k, length), dtype=np.int32)
    support[:, kept] = supporters_kept
    calls = np.full((k, length), -1, dtype=np.int8)
    calls[:, kept] = calls_kept
    latent_unknown = unknown_alleles(solution.assignments, observed_kept, k, tie_mask=mask)
    seen = {known_content_key(solution.haplotypes, latent_unknown)}
    tied, tied_nll, tied_unknown = [], [], []
    alternatives = []
    for panel in search.alternatives:
        panel_mask = solution_tie_mask(panel)
        unknown = unknown_alleles(panel.assignments, observed_kept, panel.k, tie_mask=panel_mask)
        key = known_content_key(panel.haplotypes, unknown)
        if key in seen:
            continue
        seen.add(key)
        tied.append(np.ascontiguousarray(panel.haplotypes, dtype=np.int8))
        tied_nll.append(float(panel.nll))
        tied_unknown.append(np.ascontiguousarray(unknown, dtype=np.bool_))
        panel_probability, panel_support, panel_calls = release_alleles(
            np.ascontiguousarray(likelihoods[:, kept]), panel.haplotypes, panel.assignments,
            observed_kept, config.model.unknown_copy_penalty, config.release, panel.quanta, panel_mask)
        alternatives.append((panel.haplotypes, unknown, panel_probability, panel_support, panel_calls))
    if alternatives:
        _shared_alternative_calls(solution.haplotypes, latent_unknown, probability_kept,
                                  supporters_kept, calls_kept, alternatives)
        alt_probability[:, kept] = probability_kept
        support[:, kept] = supporters_kept
        calls[:, kept] = calls_kept
    return DiscoveredBlock(
        block=int(block), site_start=int(site_start),
        positions=np.ascontiguousarray(positions, dtype=np.int64),
        kept=np.ascontiguousarray(kept, dtype=np.bool_),
        latent=np.ascontiguousarray(solution.haplotypes, dtype=np.int8), calls=calls,
        alt_probability=alt_probability, support=support,
        unknown_mass=unknown_copy_mass(solution.unknown_slots, observed_kept.any(axis=1)),
        nll=float(solution.nll), search=summary,
        latent_unknown=np.ascontiguousarray(latent_unknown, dtype=np.bool_),
        tied_panels=tuple(tied), tied_panel_nll=tuple(tied_nll), tied_unknown=tuple(tied_unknown))
