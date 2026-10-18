"""Genome-wide robust parent screen and the fixed M2 candidate panel.

M0 and every eligible M1 identity are scored on every chromosome. M2 is scored for the eligible
pairs among each child's ``parents_per_child`` best screen parents: per chromosome, a tempered
softmax of the M1 scores (contaminated with a uniform share) is mixed with tied-rank evidence,
then weighted over chromosomes by sqrt(markers) clipped around the median.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PedigreeConfig
from .eligibility import ParentEligibility
from .records import CandidatePanel
from .ties import tied_rank_probabilities

# Centred tempered scores below this many nats contribute exp(-60) (no underflow).
_SOFTMAX_CLIP = -60.0


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
    blocks = np.maximum(np.ceil(markers / config.preparation.markers_per_information_block), 1.0)
    tempering = blocks ** panel.information_tempering_power
    for child in np.flatnonzero(eligibility.children):
        parents = np.flatnonzero(eligibility.parents[child])
        if not len(parents):
            continue
        utility = np.empty((n_contigs, len(parents)), dtype=np.float64)
        for contig in range(n_contigs):
            values = pair_scores[contig, child, parents]
            centered = (values - np.max(values)) / tempering[contig]
            soft = np.exp(np.clip(centered, _SOFTMAX_CLIP, 0.0))
            soft /= np.sum(soft)
            soft = ((1.0 - panel.chromosome_contamination) * soft
                    + panel.chromosome_contamination / len(parents))
            utility[contig] = (panel.rank_weight * tied_rank_probabilities(values)
                               + (1.0 - panel.rank_weight) * soft)
        output[child, parents] = np.sum(utility * weights[:, None], axis=0)
    return output


def fixed_panel(screen_scores: np.ndarray, eligibility: ParentEligibility,
                config: PedigreeConfig) -> CandidatePanel:
    """Every eligible pair among each child's top screen parents (ties by sample index)."""
    rows = []
    top = config.panel.parents_per_child
    for child in np.flatnonzero(eligibility.children):
        parents = np.flatnonzero(eligibility.parents[child])
        order = np.lexsort((parents, -screen_scores[child, parents]))
        leading = parents[order[:min(top, len(parents))]].tolist()
        pairs = {tuple(sorted((int(leading[i]), int(leading[j]))))
                 for i in range(len(leading)) for j in range(i + 1, len(leading))
                 if eligibility.pair_allowed(int(child), leading[i], leading[j])}
        rows.extend((int(child), first, second) for first, second in sorted(pairs))
    trios = np.asarray(rows, dtype=np.int64).reshape((-1, 3))
    return CandidatePanel(trios, screen_scores)


def search_diagnostics(panel: CandidatePanel, eligibility: ParentEligibility,
                       called_parents: np.ndarray, sample_ids: tuple[str, ...]) -> pd.DataFrame:
    """Screen ranks of the panel boundary and of the called parents (diagnostics, not a certificate
    of search adequacy): ``called_parents`` int64 [N, 2] of the final Tier-B calls."""
    records = []
    scores = panel.screen_scores
    for child, sample in enumerate(sample_ids):
        parents = np.flatnonzero(eligibility.parents[child])
        order = parents[np.lexsort((parents, -scores[child, parents]))]
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
