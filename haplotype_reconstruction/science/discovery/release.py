"""Raw-discovery allele release and the stage product of one block.

The selected panel's hard rows are latent: they are what local selection fits from. Its
*released* alleles carry a capped fixed-assignment pseudo-probability and are called only where
that probability reaches ``ReleaseRuleConfig.min_call_probability`` with at least
``min_supporters`` informative carriers. The same rule releases the candidate bank's source
endpoints in local selection (local_selection/bank.py), whose calls form the path-search start.
"""

from __future__ import annotations

import math

import numba
import numpy as np
from numba import prange

from ...contracts.discovery import DiscoveredBlock, DiscoveryOmission, DiscoverySearchStatus
from ...contracts.panels import unknown_copy_mass
from ...core.kernels import kernel
from ...core.ties import BlockQuanta, fixed_policy
from .diagnostics import BlockDiagnostics
from .panel import assignment_tie_mask, known_content_key, solution_tie_mask, state_layout, unknown_alleles
from .panel import state_membership as _state_membership
from .parameters import EMISSION_FLOOR, DiscoveryConfig, ReleaseRuleConfig
from .search import BlockSearch, SearchSummary

# The cell band is a symmetric |NLL(0) - NLL(1)| <= cell test that orders nothing, so no convention reverses it
# (3.5 discovery/release.py:93 reads quanta.cell with no rev()); only the quantum is taken from the policy.
_CELL_BAND_REVERSE = False


def release_alleles(
    likelihoods: np.ndarray,
    haplotypes: np.ndarray,
    assignments: np.ndarray,
    observed: np.ndarray,
    unknown_copy_penalty: float,
    rule: ReleaseRuleConfig,
    quanta: BlockQuanta | None = None,
    tie_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    cell = fixed_policy(
        "discovery.release.cell",
        (quanta if quanta is not None else BlockQuanta.floors()).cell,
        reverse=_CELL_BAND_REVERSE,
    ).quantum()
    k = haplotypes.shape[0]
    assignments = np.ascontiguousarray(assignments, dtype=np.int64)
    mask = assignment_tie_mask(assignments, k) if tie_mask is None else np.ascontiguousarray(tie_mask, dtype=np.uint8)
    kind, first, second = state_layout(k)
    offsets, members, first, second = _state_membership(assignments, k, mask, kind, first, second)
    carriers = _release_carriers(k, offsets, members, first, second)
    probability, supporters, calls, evaluated, banded = _release_kernel(
        np.ascontiguousarray(likelihoods, dtype=np.float64),
        np.ascontiguousarray(haplotypes, dtype=np.int64),
        np.ascontiguousarray(observed, dtype=np.bool_),
        float(unknown_copy_penalty),
        int(rule.min_supporters),
        float(rule.min_call_probability),
        float(cell),
        *carriers,
    )
    return probability, supporters, calls, int(evaluated.sum()), int(banded.sum())


@kernel(
    overloads=(
        (
            numba.int64,
            numba.types.Array(numba.int64, 1, "C", readonly=False),
            numba.types.Array(numba.int64, 1, "C", readonly=False),
            numba.types.Array(numba.int64, 1, "C", readonly=False),
            numba.types.Array(numba.int64, 1, "C", readonly=False),
        ),
    ),
    cache=True,
)
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


@kernel(
    overloads=(
        (
            numba.types.Array(numba.float64, 3, "C", readonly=False),
            numba.types.Array(numba.int64, 2, "C", readonly=False),
            numba.types.Array(numba.boolean, 2, "C", readonly=False),
            numba.float64,
            numba.int64,
            numba.float64,
            numba.float64,
            numba.types.Array(numba.int64, 1, "C", readonly=False),
            numba.types.Array(numba.int64, 1, "C", readonly=False),
            numba.types.Array(numba.int64, 1, "C", readonly=False),
            numba.types.Array(numba.float64, 1, "C", readonly=False),
        ),
        (
            numba.types.Array(numba.float64, 3, "C", readonly=False),
            numba.types.Array(numba.int64, 2, "C", readonly=True),
            numba.types.Array(numba.boolean, 2, "C", readonly=False),
            numba.float64,
            numba.int64,
            numba.float64,
            numba.float64,
            numba.types.Array(numba.int64, 1, "C", readonly=False),
            numba.types.Array(numba.int64, 1, "C", readonly=False),
            numba.types.Array(numba.int64, 1, "C", readonly=False),
            numba.types.Array(numba.float64, 1, "C", readonly=False),
        ),
    ),
    cache=True,
    parallel=True,
)
def _release_kernel(
    likelihoods,
    haplotypes,
    observed,
    penalty,
    min_supporters,
    min_call_probability,
    cell_tie,
    offsets,
    samples,
    partners,
    weights,
):
    floor = EMISSION_FLOOR
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
            conservative = np.where(
                alt, np.min(panel_probability[matches], axis=0), np.max(panel_probability[matches], axis=0)
            )
            probability[founder] = np.where(
                agreed,
                np.where(
                    alt, np.minimum(probability[founder], conservative), np.maximum(probability[founder], conservative)
                ),
                0.5,
            )
            support[founder] = np.where(agreed, np.minimum(support[founder], np.min(panel_support[matches], axis=0)), 0)
            calls[founder, ~agreed] = -1


def _search_status(summary: SearchSummary) -> DiscoverySearchStatus:
    """Freeze the exact computational status, separating only declared diagnostics."""
    return DiscoverySearchStatus(
        k_values=summary.k_values,
        k_log_scores=summary.k_log_scores,
        natural_k_ceiling=summary.natural_k_ceiling,
        expansions=summary.expansions,
        data_starts=summary.data_starts,
        exact_scores=summary.exact_scores,
        exact_score_cache_hits=summary.exact_score_cache_hits,
        residual_calls=summary.residual_calls,
        residual_rows=summary.residual_rows,
        limits=summary.limits,
        heuristic_gauge_cuts=summary.heuristic_gauge_cuts,
        stop_reason=summary.stop_reason,
        certified=summary.certified,
        tie_quantum=summary.tie_quantum,
        ambiguous=summary.ambiguous,
        tied_alternatives=summary.tied_alternatives,
        score_margin=summary.score_margin,
        search_incomplete=summary.search_incomplete,
        omissions=tuple(DiscoveryOmission(**item) for item in summary.omissions),
        unresolved=summary.unresolved,
        archive_max_k=summary.archive_max_k,
        selected_mean_field_failures=summary.selected_mean_field_failures,
        other_mean_field_failures=summary.other_mean_field_failures,
    )


def materialise(
    block: int,
    site_start: int,
    positions: np.ndarray,
    kept: np.ndarray,
    likelihoods: np.ndarray,
    observed: np.ndarray,
    search: BlockSearch,
    config: DiscoveryConfig,
    diagnostics: BlockDiagnostics | None = None,
) -> DiscoveredBlock:
    """The DiscoveredBlock of a searched block: the selected panel released by the raw rule.

    likelihoods/observed: the block's founder-view evidence at all its sites (L); the release
    uses the discovery sites only (unobserved cells are ignored by the release).
    """
    solution = search.selected
    if diagnostics is not None:
        diagnostics.capture_search(search.summary)
    length = len(positions)
    if solution is None:
        # A whole seed class was omitted before any incumbent existed.
        # No fitted biological answer is available to release or pass as a start.
        return DiscoveredBlock(
            block=int(block),
            site_start=int(site_start),
            positions=np.ascontiguousarray(positions, dtype=np.int64),
            kept=np.ascontiguousarray(kept, dtype=np.bool_),
            latent=np.empty((0, int(np.count_nonzero(kept))), dtype=np.int8),
            calls=np.empty((0, length), dtype=np.int8),
            alt_probability=np.full((0, length), 0.5),
            support=np.zeros((0, length), dtype=np.int32),
            unknown_mass=1.0,
            nll=math.inf,
            search=_search_status(search.summary),
            latent_unknown=np.empty((0, int(np.count_nonzero(kept))), dtype=np.bool_),
        )
    k = solution.k
    observed_kept = np.ascontiguousarray(observed[:, kept])
    mask = solution_tie_mask(solution)
    probability_kept, supporters_kept, calls_kept, evaluated, banded = _release(
        np.ascontiguousarray(likelihoods[:, kept]),
        solution.haplotypes,
        solution.assignments,
        observed_kept,
        config.model.unknown_copy_penalty,
        config.release,
        solution.quanta,
        mask,
    )
    events = dict(search.summary.tie_events)
    events["discovery.release.cell_band"] = dict(decisions=evaluated, inside_tie=banded, wide=0, boundary=0)
    if diagnostics is not None:
        diagnostics.capture_search(search.summary, tie_events=events)
    summary = _search_status(search.summary)
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
            np.ascontiguousarray(likelihoods[:, kept]),
            panel.haplotypes,
            panel.assignments,
            observed_kept,
            config.model.unknown_copy_penalty,
            config.release,
            panel.quanta,
            panel_mask,
        )
        alternatives.append((panel.haplotypes, unknown, panel_probability, panel_support, panel_calls))
    if alternatives:
        _shared_alternative_calls(
            solution.haplotypes, latent_unknown, probability_kept, supporters_kept, calls_kept, alternatives
        )
        alt_probability[:, kept] = probability_kept
        support[:, kept] = supporters_kept
        calls[:, kept] = calls_kept
    return DiscoveredBlock(
        block=int(block),
        site_start=int(site_start),
        positions=np.ascontiguousarray(positions, dtype=np.int64),
        kept=np.ascontiguousarray(kept, dtype=np.bool_),
        latent=np.ascontiguousarray(solution.haplotypes, dtype=np.int8),
        calls=calls,
        alt_probability=alt_probability,
        support=support,
        unknown_mass=unknown_copy_mass(solution.unknown_slots, observed_kept.any(axis=1)),
        nll=float(solution.nll),
        search=summary,
        latent_unknown=np.ascontiguousarray(latent_unknown, dtype=np.bool_),
        tied_panels=tuple(tied),
        tied_panel_nll=np.asarray(tied_nll, dtype=np.float64),
        tied_unknown=tuple(tied_unknown),
    )


__all__ = ["DiscoveredBlock", "release_alleles", "materialise"]
