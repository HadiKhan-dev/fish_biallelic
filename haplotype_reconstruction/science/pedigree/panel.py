"""Genome-retained parent identities define the fixed M2 panel.

The panel holds the bounded pairs of retained identities (``bounded_panel``): a pair enters when a valid
upper bound on its joint M2 likelihood (``pedigree.pair_bound``) comes within ``PanelConfig.pair_relevance_margin`` of
the child's best known configuration in the full-data fit or in a leave-one-chromosome-out fit. Pairs outside the
panel contribute nothing to the M2 evidence (a lower bound), as pairs of non-retained identities already did. The
robust M1 utility keeps all eligible identities, floor multiplicities and contamination; it is reported for
diagnostics and does not select an arbitrary tied subset.

Method: docs/methods/pedigree.md#candidate-panel
PE1–PE3 preserve this 3.5 method.
"""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np

from ...contracts.pedigree import CandidatePanel, SearchDiagnosticView
from ...core import ties as tie_rules
from ...core.errors import InvariantError
from . import m1
from .eligibility import ParentEligibility
from .parameters import PedigreeConfig
from .ties import tied_rank_probabilities


class BoundsSource(Protocol):
    """Bounded child-column access in configured contig and canonical pair order."""

    @property
    def shape(self) -> tuple[int, int]: ...

    def __getitem__(self, key: tuple[slice, slice]) -> np.ndarray: ...


def chromosome_weights(markers: np.ndarray, config: PedigreeConfig) -> np.ndarray:
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
    config: PedigreeConfig,
) -> np.ndarray:
    """robust_screen with each child's per-chromosome M1 scores [C, P] of its eligible parents from
    ``values_of(child, parents)`` (e.g. CompactM1.values), never a dense [C, N, N] array."""
    panel = config.panel
    output = np.full((n_samples, n_samples), -np.inf, dtype=np.float64)
    weights = chromosome_weights(markers, config)
    tempering = m1.temperature(markers, config)
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


def relevance_fits(n_contigs: int) -> np.ndarray:
    """float64 [F, C]: the chromosome multiplicities the panel is screened for: the full-data fit (all ones), then
    each leave-one-chromosome-out refit (one zero), as the decision weights them (pedigree.resampling)."""
    full = np.ones((1, n_contigs))
    if n_contigs < 2:
        return full
    return np.vstack([full, np.ones((n_contigs, n_contigs)) - np.eye(n_contigs)])


def _fit_sums(fits: np.ndarray, values: np.ndarray) -> np.ndarray:
    """float64 [F, P]: for each fit, the weighted sum of the chromosome rows of values [C, P] with positive weight.
    A zero-weight chromosome is left out, never multiplied (0 x inf would be NaN); a sum meeting +inf and -inf is
    NaN, which the caller treats as no evidence for exclusion."""
    out = np.zeros((len(fits), values.shape[1]))
    with np.errstate(invalid="ignore"):
        for fit, weights in enumerate(fits):
            for contig in np.flatnonzero(weights > 0):
                out[fit] += weights[contig] * values[contig]
    return out


def bounded_panel(
    compact,
    zero: np.ndarray,
    bounds: np.ndarray | BoundsSource,
    screen_scores: np.ndarray,
    eligibility: ParentEligibility,
    config: PedigreeConfig,
) -> tuple[CandidatePanel, dict]:
    """The eligible retained pairs whose joint M2 bound reaches the child's best known configuration,
    and counts for the log.

    ``compact`` the certified M1 profile (raw chromosome M1 of each child's retained parents), ``zero`` float64 [C, N]
    the chromosome M0 scores of the same screen, ``bounds`` [C, P] each chromosome's pair bounds in the canonical
    retained-pair order (``pair_bound.retained_batches``). A pair enters when, for some fit w of ``relevance_fits``,
    w . bound >= max(w . M0, max_r w . M1_r) - margin, r over the child's retained parents: a pair left out has an M2
    below the best configuration by more than the margin in every such fit (the bound is >= M2 on every chromosome
    and w >= 0). Chromosomes a fit leaves out are omitted from its sums, and a pair whose bound sum is not a number is
    kept (a numerical failure is never evidence for exclusion). Membership is a function of the scores only; rows are
    serialized by child, then by ascending parent indices.
    """
    margin = float(config.panel.pair_relevance_margin)
    fits = relevance_fits(compact.n_contigs)
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
        known = np.maximum(
            _fit_sums(fits, zero[:, child][:, None])[:, 0], np.max(_fit_sums(fits, compact.raw[:, start:stop]), axis=1)
        )
        # cast: NumPy 2.5's stubs type signedinteger[32 | 64] + 1 as float64; it is an intp.
        bound = np.asarray(bounds[:, offsets[child] : offsets[cast(np.intp, child + 1)]], dtype=np.float64)
        with np.errstate(invalid="ignore"):
            gap = _fit_sums(fits, bound) - known[:, None]
        admitted = np.any(gap >= -margin, axis=0) | np.any(np.isnan(gap), axis=0)
        first, second = np.triu_indices(stop - start, 1)
        first, second = parents[first[admitted]], parents[second[admitted]]
        allowed = eligibility.pair_mask(int(child), first, second)
        trio = np.column_stack([np.full(int(np.sum(allowed)), child, dtype=np.int64), first[allowed], second[allowed]])
        rows.append(trio[np.lexsort((trio[:, 2], trio[:, 1]))])
    trios = np.concatenate(rows) if rows else np.empty((0, 3), dtype=np.int64)
    log = dict(retained_pairs=int(offsets[-1]), panel_rows=int(len(trios)), margin=margin, fits=int(len(fits)))
    return CandidatePanel(np.ascontiguousarray(trios, dtype=np.int64), screen_scores), log


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
    convention = tie_rules.policy("pedigree.screen.rank", kind="probability", reverse=reverse)
    for child in range(len(eligibility.children)):
        parents = np.flatnonzero(eligibility.parents[child])
        # Exact score ties by sample index (reversed under the convention reversal, release 2.2).
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
            )
        )
    return tuple(records)
