"""Held-out read-model calibration of one contig and the application of the selected model.

Every ``marker_stride``-th site enters per-sample histograms of (depth, ALT count) cells with
depth 1..``maximum_fit_depth``, split into four physical folds by ``fold_window_bp`` windows. The
nested candidates of ``fit.py`` are fitted on folds 0/1; fold 2 selects the candidate with the
highest predictive log likelihood (candidates within ``_SELECTION_TIE_NATS`` of the best are tied
and the simplest of them wins; a fit that fails with an error cannot displace the shared model;
optimiser convergence flags are diagnostics and decide nothing); fold 3 is a diagnostic only.
The selected fit is applied to every cell of the contig without genotype-mixture priors. When
calibration is disabled, a fold is empty, the shared model cannot be fitted or no candidate has a
finite selection score, the fixed binomial model applies (``EvidenceConfig.fixed_error``).
No truth, pedigree or founder panel enters fitting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from numba import njit, prange

from ..config import ReadModelConfig
from . import fit as fits
from .emission import cell_probabilities, gather_likelihoods, genotype_probabilities, prefix

MODEL_VERSION = "heldout-read-model-v6"
_TRAIN_FOLDS, _SELECTION_FOLD, _DIAGNOSTIC_FOLD = (0, 1), 2, 3
# Selection-fold log likelihoods (~1e6-2e7 nats) within this many nats of the best are ties and
# the simplest tied candidate wins: a more complex model must improve the held-out log likelihood
# by more than one nat (a likelihood ratio of e). The joint fit's score moves by up to ~0.3 nats
# under rounding-level arithmetic changes (flat in log kappa) and the gaps between candidates are
# spread continuously from 0.002 nats up, so no smaller tolerance keeps the selection
# reproducible (measurements: tickets/B-RM/design.md).
_SELECTION_TIE_NATS = 1.0


@dataclass(frozen=True)
class CalibrationReport:
    """Outcome of one contig's calibration.

    applied: a fitted model was applied (False: the fixed model, see fallback_reason).
    selected: name of the applied candidate (None when not applied). fits: every candidate that
    was fitted, by name in candidate order. scores: {name: (selection fold, diagnostic fold)}
    predictive log likelihoods (nats). fold_observations: training/selection/diagnostic cell
    counts per fold. Failures of the complex fits are kept as messages.
    """

    applied: bool
    selected: str | None
    fallback_reason: str | None
    fold_observations: tuple[int, ...]
    observations_used: int
    high_depth_excluded: int
    fits: Mapping[str, fits.ReadModelFit] = field(default_factory=dict)
    scores: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    complex_fit_failure: str | None = None

    @property
    def selection_gain_over_shared(self) -> float | None:
        if self.selected is None:
            return None
        return self.scores[self.selected][0] - self.scores[fits.BINOMIAL][0]

    @property
    def diagnostic_gain_over_shared(self) -> float | None:
        if self.selected is None:
            return None
        return self.scores[self.selected][1] - self.scores[fits.BINOMIAL][1]

    def record(self) -> dict:
        return dict(model=MODEL_VERSION, applied=self.applied, selected=self.selected,
                    fallback_reason=self.fallback_reason,
                    fold_observations=list(self.fold_observations),
                    observations_used=self.observations_used,
                    high_depth_excluded=self.high_depth_excluded,
                    fits={name: value.record() for name, value in self.fits.items()},
                    scores={name: list(value) for name, value in self.scores.items()},
                    complex_fit_failure=self.complex_fit_failure)

    @classmethod
    def from_record(cls, record: Mapping) -> "CalibrationReport":
        return cls(applied=record["applied"], selected=record["selected"],
                   fallback_reason=record["fallback_reason"],
                   fold_observations=tuple(record["fold_observations"]),
                   observations_used=record["observations_used"],
                   high_depth_excluded=record["high_depth_excluded"],
                   fits={name: fits.ReadModelFit.from_record(value)
                         for name, value in record["fits"].items()},
                   scores={name: tuple(value) for name, value in record["scores"].items()},
                   complex_fit_failure=record["complex_fit_failure"])


def histograms(counts, positions, config: ReadModelConfig):
    """Per-fold, per-sample histograms over the observed (depth, ALT) cells of the stride sites.

    Returns (hist float64[4, N, C], depth int64[C], alt int64[C], observations_used,
    high_depth_excluded); C = cells with depth 1..maximum_fit_depth seen in any fold.
    """
    indices = np.arange(0, len(positions), config.marker_stride)
    sampled = np.asarray(counts[:, indices], np.int64)
    depth = sampled.sum(axis=2)
    alt = sampled[:, :, 1]
    folds = (positions[indices] // config.fold_window_bp) % 4
    width = config.maximum_fit_depth + 1
    result = np.zeros((4, len(sampled), width * width), np.float64)
    admitted = (depth > 0) & (depth <= config.maximum_fit_depth)
    for fold in range(4):
        for sample in range(len(sampled)):
            mask = admitted[sample] & (folds == fold)
            result[fold, sample] = np.bincount(depth[sample, mask] * width + alt[sample, mask],
                                               minlength=width * width)
    d, a = np.divmod(np.arange(width * width), width)
    valid = (d > 0) & (a <= d) & (result.sum(axis=(0, 1)) > 0)
    return (result[:, :, valid], d[valid], a[valid], int(admitted.sum()),
            int((depth > config.maximum_fit_depth).sum()))


def calibrate(counts, positions, config: ReadModelConfig, *, threads: int):
    """Fit and select the contig's read model. Returns (ReadModelFit | None, CalibrationReport).

    Kernels run on the caller's Numba team; per-sample fits use ``threads`` Python threads.
    """
    if not config.enabled:
        return None, CalibrationReport(applied=False, selected=None,
                                       fallback_reason="calibration disabled",
                                       fold_observations=(), observations_used=0,
                                       high_depth_excluded=0)
    hist, depth, alt, used, high = histograms(counts, positions, config)
    fold_observations = tuple(int(fold.sum()) for fold in hist)
    report = dict(fold_observations=fold_observations, observations_used=used,
                  high_depth_excluded=high)
    if any(fold.sum() == 0 for fold in hist):
        return None, CalibrationReport(
            applied=False, selected=None,
            fallback_reason="all four physical folds need calibration observations", **report)
    training = hist[_TRAIN_FOLDS[0]] + hist[_TRAIN_FOLDS[1]]
    try:
        shared = fits.fit_shared(training, depth, alt, config)
    except fits.SharedStationarityError:
        # Distinct stationary modes need an explicit observation-model policy.
        # They must not silently select the uncalibrated fallback.
        raise
    except (ValueError, RuntimeError) as error:
        return None, CalibrationReport(applied=False, selected=None, fallback_reason=str(error),
                                       **report)
    candidates = {fits.BINOMIAL: shared}
    complex_failure = None
    try:
        pooled = fits.fit_pooled_samples(training, depth, alt, shared, threads)
        joint = fits.fit_joint(training, depth, alt, pooled, config)
        candidates.update({fits.SAMPLE_BINOMIAL: pooled, fits.SAMPLE_JOINT: joint})
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        complex_failure = str(error)
    scores = {name: (fits.score(hist[_SELECTION_FOLD], depth, alt, model),
                     fits.score(hist[_DIAGNOSTIC_FOLD], depth, alt, model))
              for name, model in candidates.items()}
    common = dict(fits=candidates, scores=scores, complex_fit_failure=complex_failure, **report)
    selected = select(scores)
    if selected is None:
        return None, CalibrationReport(applied=False, selected=None,
                                       fallback_reason="no finite predictive fit", **common)
    return candidates[selected], CalibrationReport(applied=True, selected=selected,
                                                   fallback_reason=None, **common)


def select(scores) -> str | None:
    """The selected candidate: the simplest within ``_SELECTION_TIE_NATS`` of the best selection score.

    ``scores``: {name: (selection fold, diagnostic fold)} in candidate order (simplest first); a
    candidate with a non-finite selection score is not eligible. None when nothing is eligible.
    """
    eligible = [name for name, value in scores.items() if np.isfinite(value[0])]
    if not eligible:
        return None
    best = max(scores[name][0] for name in eligible)
    return next(name for name in eligible if scores[name][0] >= best - _SELECTION_TIE_NATS)


@njit(cache=True, parallel=True)
def _fill_deep_cells(counts, out, cap, error, heterozygote, het_inverse):
    """Evaluate the cells deeper than the lookup table directly (samples in parallel)."""
    n, length = counts.shape[:2]
    for sample in prange(n):
        deepest = 0
        for site in range(length):
            depth = counts[sample, site, 0] + counts[sample, site, 1]
            if depth > cap and depth > deepest:
                deepest = depth
        if deepest == 0:
            continue
        hom = prefix(error[sample], 0., deepest)
        het = prefix(heterozygote[sample], het_inverse, deepest)
        for site in range(length):
            a = int(counts[sample, site, 1])
            r = int(counts[sample, site, 0])
            d = a + r
            if d <= cap:
                continue
            out[sample, site, 0], out[sample, site, 1], out[sample, site, 2] = cell_probabilities(hom, het, a, r, d)


@njit(cache=True, parallel=True)
def _maximum_depth(counts):
    n, length = counts.shape[:2]
    per_sample = np.zeros(n, np.int64)
    for sample in prange(n):
        deepest = 0
        for site in range(length):
            depth = counts[sample, site, 0] + counts[sample, site, 1]
            if depth > deepest:
                deepest = depth
        per_sample[sample] = deepest
    return per_sample.max() if n else 0


def likelihoods(counts, model: fits.ReadModelFit, lookup_depth: int) -> np.ndarray:
    """Normalised P(reads | g) float64[N, L, 3] of every cell under ``model`` (no prior).

    Cells up to ``lookup_depth`` come from a per-sample (depth, ALT) table, deeper cells are
    evaluated directly (a performance split, not an approximation). Zero depth is exactly 1/3.
    Runs on the caller's Numba team.
    """
    counts = np.ascontiguousarray(counts)
    error, heterozygote = model.error_and_heterozygote()
    het_inverse = model.heterozygote_inverse_concentration
    cap = min(int(lookup_depth), int(_maximum_depth(counts)))
    d, a = np.tril_indices(cap + 1)
    table = np.empty((len(counts), cap + 1, cap + 1, 3), np.float64)
    for sample in range(len(counts)):
        table[sample, d, a] = genotype_probabilities(error[sample], heterozygote[sample],
                                                     het_inverse, d, a)
    output = gather_likelihoods(counts, table, cap)
    _fill_deep_cells(counts, output, cap, error, heterozygote, het_inverse)
    return output
