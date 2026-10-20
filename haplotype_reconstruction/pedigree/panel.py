"""Genome-retained parent identities define the fixed M2 panel.

Every eligible pair of retained identities is included. The robust M1 utility
keeps all eligible identities, floor multiplicities and contamination; it is
reported for diagnostics and does not select an arbitrary tied subset.
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
    n_contigs, n_samples, _ = pair_scores.shape
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
        for contig in range(n_contigs):
            values = pair_scores[contig, child, parents]
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
