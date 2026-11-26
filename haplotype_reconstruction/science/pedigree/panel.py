"""Genome-retained parent identities define the fixed M2 panel.

Every eligible pair of a child's genome-retained M1 parents enters the panel: there is no relevance screen, so for a
child within ``PanelParameters.max_pairs_per_child`` the M2 evidence covers every retained pair in every fit (the
full data, each bootstrap draw and each leave-one-unit-out fit). A child with more pairs keeps those with the largest
full-data M2 likelihood key (``pair_bound``, summed over contigs; no chromosome or unit structure enters the ranking), every
pair tied with the last kept one and every pair whose key is indeterminate; it is flagged, because the cap can drop
a competitive pair. Pairs outside the panel contribute nothing to M2 evidence, which is then a lower bound. The
robust M1 utility keeps all eligible identities, floor multiplicities and contamination; it is reported for
diagnostics and does not select an arbitrary tied subset.

Method: docs/methods/pedigree.md#candidate-panel
"""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np

from ...contracts.pedigree import CandidatePanel, SearchDiagnosticView
from ...core import ties as tie_rules
from ...core.errors import InvariantError
from . import m1
from .eligibility import ParentEligibility
from .parameters import PedigreeParameters
from .ties import tied_rank_probabilities


class BoundsSource(Protocol):
    """Bounded child-column access in configured contig and canonical pair order."""

    @property
    def shape(self) -> tuple[int, int]: ...

    def __getitem__(self, key: tuple[slice, slice]) -> np.ndarray: ...


def chromosome_weights(markers: np.ndarray, config: PedigreeParameters) -> np.ndarray:
    """float64 [C]: sqrt(markers) / median, clipped to [1 / ratio, ratio], normalised to sum 1."""
    weights = np.sqrt(np.asarray(markers, dtype=np.float64))
    ratio = config.panel.maximum_contig_weight_ratio
    weights = np.clip(weights / float(np.median(weights)), 1.0 / ratio, ratio)
    return weights / np.sum(weights)


def robust_screen_values(
    values_of,
    n_contigs: int,
    n_samples: int,
    markers: np.ndarray,
    eligibility: ParentEligibility,
    config: PedigreeParameters,
    units: np.ndarray,
) -> np.ndarray:
    """robust_screen with each child's per-chromosome M1 scores [C, P] of its eligible parents from
    ``values_of(child, parents)`` (e.g. CompactM1.values), never a dense [C, N, N] array. ``units``: the retention
    band's units per chromosome (m1.retention_units)."""
    panel = config.panel
    output = np.full((n_samples, n_samples), -np.inf, dtype=np.float64)
    weights = chromosome_weights(markers, config)
    tempering = m1.temperature(units, config)
    # Caller supplies the fixed genome-floor values; retained entries stay raw.
    for child in np.flatnonzero(eligibility.children):
        parents = np.flatnonzero(eligibility.parents[child])
        if not len(parents):
            continue
        utility = np.empty((n_contigs, len(parents)), dtype=np.float64)
        scores = values_of(child, parents)
        for contig in range(n_contigs):
            values = scores[contig]
            centered = (values - np.max(values)) / tempering[contig]
            soft = np.exp(centered)
            soft /= np.sum(soft)
            soft = (1.0 - panel.chromosome_contamination) * soft + panel.chromosome_contamination / len(parents)
            utility[contig] = panel.rank_weight * tied_rank_probabilities(values) + (1.0 - panel.rank_weight) * soft
        output[child, parents] = np.sum(utility * weights[:, None], axis=0)
    return output


def _genome_sums(values: np.ndarray) -> np.ndarray:
    """float64 [P]: the full-data sums of the chromosome rows of values [C, P] (rows are the per-contig storage of the
    M2 keys), added in configured contig order. A sum meeting +inf and -inf is NaN."""
    out = np.zeros(values.shape[1])
    with np.errstate(invalid="ignore"):
        for row in values:
            out += row
    return out


def bounded_panel(
    compact,
    bounds: np.ndarray | BoundsSource,
    screen_scores: np.ndarray,
    eligibility: ParentEligibility,
    config: PedigreeParameters,
) -> tuple[CandidatePanel, dict]:
    """The capped retained-parent pairs, and counts for the log.

    ``compact`` the certified M1 profile (the retained parents of each child), ``bounds`` [C, P] each chromosome's
    M2 keys in the canonical retained-pair order (``pair_bound_screen.retained_batches``). Every eligible pair of a
    child's retained parents enters. When a child has more than the nominal per-child cap, it keeps the pairs with
    the largest sum of chromosome M2 keys, including every pair tied with
    the last kept one and every pair whose sum is not a number (a numerical failure is never evidence for exclusion).
    A capped child's panel is incomplete: the cap has no competitive-pair exclusion guarantee. Membership depends
    only on scores and eligibility; rows are serialized by child, then by ascending parent indices.
    """
    cap = int(config.panel.max_pairs_per_child)
    capped = []
    counts = np.diff(compact.indptr)
    offsets = np.concatenate([[0], np.cumsum(counts * (counts - 1) // 2)])
    if bounds.shape != (compact.n_contigs, int(offsets[-1])):
        raise InvariantError("pair bounds do not match the retained pairs of the screen")
    rows = []
    for child in np.flatnonzero(eligibility.children):
        start, stop = compact.indptr[child : child + 2]
        if stop - start < 2:
            continue
        parents = compact.parents[start:stop]
        first, second = np.triu_indices(stop - start, 1)
        allowed = eligibility.pair_mask(int(child), parents[first], parents[second])
        trio = np.column_stack(
            [np.full(int(np.sum(allowed)), child, dtype=np.int64), parents[first[allowed]], parents[second[allowed]]]
        )
        if len(trio) > cap:
            # cast: NumPy 2.5's stubs type signedinteger[32 | 64] + 1 as float64; it is an intp.
            bound = np.asarray(bounds[:, offsets[child] : offsets[cast(np.intp, child + 1)]], dtype=np.float64)
            # A NaN key ranks first. Keep every pair tied at the cap.
            total = _genome_sums(bound)
            best = np.where(np.isnan(total), np.inf, total)[allowed]
            trio = trio[best >= np.sort(best)[len(best) - cap]]
            capped.append(int(child))
        rows.append(trio[np.lexsort((trio[:, 2], trio[:, 1]))])
    trios = np.concatenate(rows) if rows else np.empty((0, 3), dtype=np.int64)
    log = dict(
        retained_pairs=int(offsets[-1]),
        panel_rows=int(len(trios)),
        capped_children=len(capped),
    )
    return CandidatePanel(np.ascontiguousarray(trios, dtype=np.int64), screen_scores, tuple(capped)), log


def search_diagnostics(
    panel: CandidatePanel,
    eligibility: ParentEligibility,
    called_parents: np.ndarray,
    *,
    reverse: bool = False,
) -> tuple[SearchDiagnosticView, ...]:
    """Screen ranks of the panel boundary and of the called parents (diagnostics, not a certificate
    of search adequacy): ``called_parents`` int64 [N, 2] of the final Tier-B calls."""
    records = []
    scores = panel.screen_scores
    capped = set(panel.capped_children)
    convention = tie_rules.policy("pedigree.screen.rank", kind="probability", reverse=reverse)
    for child in range(len(eligibility.children)):
        parents = np.flatnonzero(eligibility.parents[child])
        # Exact score ties by sample index (reversed under the convention reversal).
        index_key = -parents if convention.reverse else parents
        order = parents[np.lexsort((index_key, -scores[child, parents]))]
        rank = {int(parent): position + 1 for position, parent in enumerate(order)}
        trios = panel.trios[panel.trios[:, 0] == child]
        included = set(map(int, trios[:, 1:].ravel()))
        omitted = [int(parent) for parent in order if int(parent) not in included]
        weakest = min((scores[child, parent] for parent in included), default=np.nan)
        strongest_omitted = scores[child, omitted[0]] if omitted else np.nan
        boundary = max((rank[parent] for parent in included), default=0)
        selected_ranks = [rank.get(int(parent)) if parent >= 0 else None for parent in called_parents[child]]
        records.append(
            SearchDiagnosticView(
                sample=child,
                eligible_parents=len(parents),
                scored_m2_pairs=len(trios),
                m2_panel_parents=len(included),
                screen_boundary_rank=int(boundary),
                screen_boundary_gap=float(weakest - strongest_omitted),
                omitted_parent_count=len(omitted),
                strongest_omitted_parent=-1 if not omitted else omitted[0],
                parent1_screen_rank=float(selected_ranks[0]) if selected_ranks[0] is not None else np.nan,
                parent2_screen_rank=float(selected_ranks[1]) if selected_ranks[1] is not None else np.nan,
                selected_parent_at_screen_boundary=bool(omitted and boundary in selected_ranks),
                m2_panel_capped=child in capped,
            )
        )
    return tuple(records)
