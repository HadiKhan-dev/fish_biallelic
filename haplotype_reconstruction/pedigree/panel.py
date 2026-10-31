"""Genome-retained parent identities define the fixed M2 panel.

Release 3.4: the panel holds the bounded pairs of retained identities (``bounded_panel``): a pair enters when a valid
upper bound on its joint M2 likelihood (``pedigree.pair_bound``) comes within ``PanelConfig.pair_relevance_margin`` of
the child's best known configuration in the full-data fit or in a leave-one-chromosome-out fit. Pairs outside the
panel contribute nothing to the M2 evidence (a lower bound), as pairs of non-retained identities already did. The
robust M1 utility keeps all eligible identities, floor multiplicities and contamination; it is reported for
diagnostics and does not select an arbitrary tied subset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..maths import ties as tie_rules
from . import m1
from .config import PedigreeConfig
from .eligibility import ParentEligibility
from .records import CandidatePanel
from .ties import tied_rank_probabilities

# Screen utilities (probability mixtures in [0, 1]) closer than this are tied (release 2.1): the top-parent cut keeps
# every parent tied with the last one admitted (at most twice ``parents_per_child``), ordered by sample index.
# Census (std_n80, work/IE/probes/std_t2_*): utilities move by <= 5.4e-12 under iid/value-consistent 1e-9 GL noise;
# consecutive genuine gaps among the top 25 parents reach down to 1e-8 (1 in [1e-8, 1e-7), 9 in [1e-7, 1e-6)), so the
# quantum sits between: 2^-32 = 2.3e-10, ~40x from both.
SCREEN_TIE = 2.0 ** -32

# Centred tempered scores below this many nats contribute exp(-60) (no underflow).
_SOFTMAX_CLIP = m1.LOG_FLOOR


def chromosome_weights(markers: np.ndarray, config: PedigreeConfig) -> np.ndarray:
    """float64 [C]: sqrt(markers) / median, clipped to [1 / ratio, ratio], normalised to sum 1."""
    weights = np.sqrt(np.asarray(markers, dtype=np.float64))
    ratio = config.panel.maximum_contig_weight_ratio
    weights = np.clip(weights / float(np.median(weights)), 1.0 / ratio, ratio)
    return weights / np.sum(weights)


def robust_screen(pair_scores: np.ndarray, markers: np.ndarray, eligibility: ParentEligibility,
                  config: PedigreeConfig) -> np.ndarray:
    """float64 [N, N] screen utility [child, parent] from the per-chromosome M1 scores [C, N, N]."""
    return robust_screen_values(lambda child, parents: pair_scores[:, child, parents], len(markers),
                                pair_scores.shape[1], markers, eligibility, config)


def robust_screen_values(values_of, n_contigs: int, n_samples: int, markers: np.ndarray,
                         eligibility: ParentEligibility, config: PedigreeConfig) -> np.ndarray:
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
            soft = ((1.0 - panel.chromosome_contamination) * soft
                    + panel.chromosome_contamination / len(parents))
            utility[contig] = (panel.rank_weight * tied_rank_probabilities(values)
                               + (1.0 - panel.rank_weight) * soft)
        output[child, parents] = np.sum(utility * weights[:, None], axis=0)
    return output


def fixed_panel(screen_scores: np.ndarray, eligibility: ParentEligibility,
                config: PedigreeConfig) -> CandidatePanel:
    """Legacy reference only. Every eligible pair among top screen parents: utilities in tie classes, ties by
    sample index, and the parents tied with the last admitted one are admitted too (at most twice the budget)."""
    rows = []
    top = config.panel.parents_per_child
    for child in np.flatnonzero(eligibility.children):
        parents = np.flatnonzero(eligibility.parents[child])
        if not len(parents):
            continue
        values = screen_scores[child, parents]
        ranked = tie_rules.tie_classes(np.where(np.isfinite(values), values, -np.inf), SCREEN_TIE,
                                       content=[int(parent) for parent in parents], descending=True)
        count = min(top, len(parents))
        if count < len(parents):
            last_class = ranked.classes[ranked.order[count - 1]]
            while count < min(2 * top, len(parents)) and ranked.classes[ranked.order[count]] == last_class:
                count += 1
        leading = parents[ranked.order[:count]].tolist()
        pairs = {tuple(sorted((int(leading[i]), int(leading[j]))))
                 for i in range(len(leading)) for j in range(i + 1, len(leading))
                 if eligibility.pair_allowed(int(child), leading[i], leading[j])}
        rows.extend((int(child), first, second) for first, second in sorted(pairs))
    trios = np.asarray(rows, dtype=np.int64).reshape((-1, 3))
    return CandidatePanel(trios, screen_scores)


def retained_panel(keep: np.ndarray, screen_scores: np.ndarray,
                   eligibility: ParentEligibility) -> CandidatePanel:
    """Every eligible pair of genome-retained parents, including all cutoff ties.

    Ascending indices only serialize the complete unordered set; they never
    choose membership. There is no parent budget or tie-class cap.
    """
    rows = []
    for child in np.flatnonzero(eligibility.children):
        parents = np.flatnonzero(keep[child] & eligibility.parents[child])
        for i, first in enumerate(parents):
            for second in parents[i + 1:]:
                if eligibility.pair_allowed(int(child), int(first), int(second)):
                    rows.append((int(child), int(first), int(second)))
    return CandidatePanel(np.asarray(rows, dtype=np.int64).reshape(-1, 3), screen_scores)


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


def bounded_panel(compact, zero: np.ndarray, bounds: np.ndarray, screen_scores: np.ndarray,
                  eligibility: ParentEligibility, config: PedigreeConfig) -> tuple[CandidatePanel, dict]:
    """The eligible retained pairs whose joint M2 bound reaches the child's best known configuration (release 3.4),
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
        raise ValueError("pair bounds do not match the retained pairs of the screen")
    rows = []
    for child in np.flatnonzero(eligibility.children):
        start, stop = compact.indptr[child:child + 2]
        if stop - start < 2:
            continue
        parents = compact.parents[start:stop]
        known = np.maximum(_fit_sums(fits, zero[:, child][:, None])[:, 0],
                           np.max(_fit_sums(fits, compact.raw[:, start:stop]), axis=1))
        bound = np.asarray(bounds[:, offsets[child]:offsets[child + 1]], dtype=np.float64)
        with np.errstate(invalid="ignore"):
            gap = _fit_sums(fits, bound) - known[:, None]
        admitted = np.any(gap >= -margin, axis=0) | np.any(np.isnan(gap), axis=0)
        first, second = np.triu_indices(stop - start, 1)
        first, second = parents[first[admitted]], parents[second[admitted]]
        allowed = eligibility.pair_mask(int(child), first, second)
        trio = np.column_stack([np.full(int(np.sum(allowed)), child, dtype=np.int64), first[allowed],
                                second[allowed]])
        rows.append(trio[np.lexsort((trio[:, 2], trio[:, 1]))])
    trios = np.concatenate(rows) if rows else np.empty((0, 3), dtype=np.int64)
    log = dict(retained_pairs=int(offsets[-1]), panel_rows=int(len(trios)), margin=margin, fits=int(len(fits)))
    return CandidatePanel(np.ascontiguousarray(trios, dtype=np.int64), screen_scores), log


def genome_panel(raw_one: np.ndarray, markers: np.ndarray,
                 eligibility: ParentEligibility, config: PedigreeConfig) -> CandidatePanel:
    """Build the fixed panel from RAW summed chromosome M1, before resampling."""
    one, keep, _, _, _ = m1.genome_profile(raw_one, markers, eligibility, config)
    return retained_panel(keep, robust_screen(one, markers, eligibility, config), eligibility)


def search_diagnostics(panel: CandidatePanel, eligibility: ParentEligibility,
                       called_parents: np.ndarray, sample_ids: tuple[str, ...]) -> pd.DataFrame:
    """Screen ranks of the panel boundary and of the called parents (diagnostics, not a certificate
    of search adequacy): ``called_parents`` int64 [N, 2] of the final Tier-B calls."""
    records = []
    scores = panel.screen_scores
    for child, sample in enumerate(sample_ids):
        parents = np.flatnonzero(eligibility.parents[child])
        # Exact score ties by sample index (reversed under the convention reversal, release 2.2).
        index_key = -parents if tie_rules.reverse_conventions() else parents
        order = parents[np.lexsort((index_key, -scores[child, parents]))]
        rank = {int(parent): position + 1 for position, parent in enumerate(order)}
        trios = panel.trios[panel.trios[:, 0] == child]
        included = set(map(int, trios[:, 1:].ravel()))
        omitted = [int(parent) for parent in order if int(parent) not in included]
        weakest = min((scores[child, parent] for parent in included), default=np.nan)
        strongest_omitted = scores[child, omitted[0]] if omitted else np.nan
        boundary = max((rank[parent] for parent in included), default=0)
        selected_ranks = [rank.get(int(parent)) if parent >= 0 else None
                          for parent in called_parents[child]]
        records.append(dict(
            Sample=sample, EligibleParents=len(parents), ScoredM2Pairs=len(trios),
            M2PanelParents=len(included), ScreenBoundaryRank=boundary,
            ScreenBoundaryGap=weakest - strongest_omitted, OmittedParentCount=len(omitted),
            StrongestOmittedParent=None if not omitted else sample_ids[omitted[0]],
            Parent1ScreenRank=selected_ranks[0], Parent2ScreenRank=selected_ranks[1],
            SelectedParentAtScreenBoundary=bool(omitted and boundary in selected_ranks)))
    return pd.DataFrame(records)
