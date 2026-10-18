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
from .search import BlockSearch, SearchSummary


@dataclass(frozen=True)
class DiscoveredBlock:
    """The raw-discovery result of one informative block (record ``discovery/blocks``).

    block: index into ``ContigEvidence.block_spans``; site_start: its first contig site.
    positions: int64[L] the block's site positions; kept: bool[L] its discovery sites.
    latent: int8[K, L_kept] the selected hard panel at the discovery sites, rows in canonical
        (lexicographic) order; the only discovery output that local selection fits from.
    calls: int8[K, L] released alleles (-1 = withheld or not a discovery site).
    alt_probability: float64[K, L] capped fixed-assignment pseudo-probability of ALT
        (0.5 at sites that are not discovery sites).
    support: int32[K, L] informative carriers of either direction (0 off the discovery sites); release 2.1: a
        carrier counts when its contribution exceeds the block's ``cell`` tie quantum in absolute value (2.0: when it
        was not exactly zero), so symmetric evidence no longer becomes a supporter under rounding or noise.
    unknown_mass: float, the selected assignment's unknown-copy fraction over samples with depth.
    nll: float, the selected panel's binned switching-Viterbi negative log likelihood (nats).
    search: small typed summary of the search (scores per represented K, limits, counts).
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

    @property
    def founders(self) -> int:
        return int(self.latent.shape[0])


def release_alleles(likelihoods, haplotypes, assignments, observed, unknown_copy_penalty,
                    rule, quanta=None):
    """Fixed-assignment founder/site evidence of a hard panel and its released alleles.

    likelihoods: float64[N, L, 3] normalised GLs (unobserved cells are ignored); haplotypes:
    int[K, L] binary rows; assignments: int[N, 2] sorted row pairs, K = unknown copy; observed:
    bool[N, L]. Costs follow the H step's buckets (homozygous carriers, carriers with a known
    partner, carriers with an unknown partner) capped per cell at the unknown-unknown cost; a
    carrier whose capped costs for allele 0 and 1 differ by more than the ``cell`` tie quantum
    (quanta: the block's BlockQuanta; default the floors) is an informative supporter. Returns
    (alt_probability float64[K, L], supporters int64[K, L], calls int8[K, L]); log pseudo-odds
    NLL(0) - NLL(1) > 0 favour ALT.
    """
    return _release(likelihoods, haplotypes, assignments, observed, unknown_copy_penalty, rule, quanta)[:3]


def _release(likelihoods, haplotypes, assignments, observed, unknown_copy_penalty, rule, quanta=None):
    """release_alleles plus (contributions evaluated, contributions strictly inside the cell band)."""
    from ..maths.ties import BlockQuanta
    cell = (quanta if quanta is not None else BlockQuanta.floors()).cell
    probability, supporters, calls, evaluated, banded = _release_kernel(
        np.ascontiguousarray(likelihoods, dtype=np.float64),
        np.ascontiguousarray(haplotypes, dtype=np.int64),
        np.ascontiguousarray(assignments, dtype=np.int64),
        np.ascontiguousarray(observed, dtype=np.bool_),
        float(unknown_copy_penalty), int(rule.min_supporters),
        float(rule.min_call_probability), float(cell))
    return probability, supporters, calls, int(evaluated.sum()), int(banded.sum())


@njit(cache=True, parallel=True)
def _release_kernel(likelihoods, haplotypes, assignments, observed, penalty, min_supporters,
                    min_call_probability, cell_tie):
    floor = 1e-12
    n = likelihoods.shape[0]
    k, length = haplotypes.shape
    unknown = k
    probability = np.full((k, length), 0.5, dtype=np.float64)
    supporters = np.zeros((k, length), dtype=np.int64)
    calls = np.full((k, length), -1, dtype=np.int8)
    evaluated = np.zeros(k, dtype=np.int64)
    banded = np.zeros(k, dtype=np.int64)
    for founder in prange(k):
        for site in range(length):
            support = 0
            evidence = 0.0
            for sample in range(n):
                if not observed[sample, site]:
                    continue
                a0 = assignments[sample, 0]
                a1 = assignments[sample, 1]
                bucket = 0
                partner_allele = 0
                if a0 == founder and a1 == founder:
                    bucket = 1
                elif a0 == founder and a1 == unknown:
                    bucket = 3
                elif ((a0 == founder or a1 == founder) and a0 != a1
                      and a0 != unknown and a1 != unknown):
                    bucket = 2
                    partner = a1 if a0 == founder else a0
                    partner_allele = haplotypes[partner, site]
                else:
                    continue
                g0 = likelihoods[sample, site, 0]
                g1 = likelihoods[sample, site, 1]
                g2 = likelihoods[sample, site, 2]
                cost_cap = -math.log(max(g0, g1, g2, floor)) + 2.0 * penalty
                if bucket == 1:
                    cost0 = -math.log(max(g0, floor))
                    cost1 = -math.log(max(g2, floor))
                elif bucket == 2:
                    cost0 = -math.log(max(likelihoods[sample, site, partner_allele], floor))
                    cost1 = -math.log(max(likelihoods[sample, site, partner_allele + 1], floor))
                else:
                    cost0 = -math.log(max(g0, g1, floor)) + penalty
                    cost1 = -math.log(max(g1, g2, floor)) + penalty
                contribution = min(cost0, cost_cap) - min(cost1, cost_cap)
                evaluated[founder] += 1
                if abs(contribution) > cell_tie:
                    evidence += contribution
                    support += 1
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


def materialise(block: int, site_start: int, positions, kept, likelihoods, observed,
                search: BlockSearch, config) -> DiscoveredBlock:
    """The DiscoveredBlock of a searched block: the selected panel released by the raw rule.

    likelihoods/observed: the block's founder-view evidence at all its sites (L); the release
    uses the discovery sites only (unobserved cells are ignored by the release).
    """
    solution = search.selected
    k = solution.k
    length = len(positions)
    observed_kept = np.ascontiguousarray(observed[:, kept])
    probability_kept, supporters_kept, calls_kept, evaluated, banded = _release(
        np.ascontiguousarray(likelihoods[:, kept]), solution.haplotypes, solution.assignments,
        observed_kept, config.model.unknown_copy_penalty, config.release, solution.quanta)
    events = dict(search.summary.tie_events)
    events["discovery.release.cell_band"] = dict(decisions=evaluated, inside_tie=banded, wide=0, boundary=0)
    summary = replace(search.summary, tie_events=events)
    alt_probability = np.full((k, length), 0.5, dtype=np.float64)
    alt_probability[:, kept] = probability_kept
    support = np.zeros((k, length), dtype=np.int32)
    support[:, kept] = supporters_kept
    calls = np.full((k, length), -1, dtype=np.int8)
    calls[:, kept] = calls_kept
    return DiscoveredBlock(
        block=int(block), site_start=int(site_start),
        positions=np.ascontiguousarray(positions, dtype=np.int64),
        kept=np.ascontiguousarray(kept, dtype=np.bool_),
        latent=np.ascontiguousarray(solution.haplotypes, dtype=np.int8), calls=calls,
        alt_probability=alt_probability, support=support,
        unknown_mass=unknown_copy_mass(solution.unknown_slots, observed_kept.any(axis=1)),
        nll=float(solution.nll), search=summary)
