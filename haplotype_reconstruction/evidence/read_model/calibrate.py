"""Held-out read-model calibration of one contig and the application of the selected model.

Every kept site enters per-sample histograms of (depth, ALT count) cells with depth 1..``maximum_fit_depth``,
split into four physical folds by ``fold_window_bp`` windows: folds 0/1 train, folds 2/3 are held out. The nested
candidates of ``fit.py`` are fitted on the training folds. Each more complex candidate in turn replaces the currently
accepted one when its held-out log-likelihood gain over it is at least ``selection_z`` standard errors, the error
estimated from the gains of the held-out stretches with admitted cells (the folds 2/3 of each four-window cycle),
because sites linked through ancestry or sharing reads are not independent observations; with fewer than
``selection_minimum_blocks`` such stretches the shared model is kept. A fit that fails with an error cannot displace the
shared model; optimiser convergence flags are diagnostics and decide nothing. The selected fit is applied to every
cell of the contig without genotype-mixture priors. When calibration is disabled, the training or held-out folds are
empty, the shared model cannot be fitted or it has no finite held-out score, the fixed binomial model applies
(``EvidenceConfig.fixed_error``).
No truth, pedigree or founder panel enters fitting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Mapping

import numpy as np
from numba import njit, prange

from ..config import ReadModelConfig
from . import fit as fits
from .emission import cell_probabilities, gather_likelihoods, genotype_probabilities, prefix

MODEL_VERSION = "heldout-read-model-v7"
# Folds 0/1 (of every four fold windows) train, folds 2/3 are held out.
_TRAIN_FOLDS = 2
# A selection whose z lies within this distance of selection_z is counted as a near-tie in the ties report.
_SELECTION_TIE_Z = 0.5
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibrationReport:
    """Outcome of one contig's calibration.

    applied: a fitted model was applied (False: the fixed model, see fallback_reason).
    selected: name of the applied candidate (None when not applied). fits: every candidate that
    was fitted, by name in candidate order. scores: {name: (held-out log likelihood, None)} (nats;
    the second entry, the former diagnostic fold, is always None). comparisons: the sequential
    selection steps ({candidate, against, gain, standard_error, blocks, z, accepted}).
    fold_observations: training and held-out cell counts. Failures of the complex fits are kept
    as messages.
    """

    applied: bool
    selected: str | None
    fallback_reason: str | None
    fold_observations: tuple[int, ...]
    observations_used: int
    high_depth_excluded: int
    fits: Mapping[str, fits.ReadModelFit] = field(default_factory=dict)
    scores: Mapping[str, tuple[float, float | None]] = field(default_factory=dict)
    complex_fit_failure: str | None = None
    shared_fit_warning: str | None = None
    shared_fit_diagnostics: Mapping | None = None
    # maths.ties.TieCounter record: ev.read_model.select.band, one decision per selection step, inside_tie when
    # its z lay within _SELECTION_TIE_Z of selection_z.
    tie_events: Mapping = field(default_factory=dict)
    comparisons: tuple = ()

    @property
    def selection_gain_over_shared(self) -> float | None:
        if self.selected is None:
            return None
        return self.scores[self.selected][0] - self.scores[fits.BINOMIAL][0]

    @property
    def diagnostic_gain_over_shared(self) -> float | None:
        if self.selected is None or self.scores[self.selected][1] is None:
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
                    complex_fit_failure=self.complex_fit_failure, tie_events=dict(self.tie_events),
                    comparisons=[dict(step) for step in self.comparisons],
                    **({} if self.shared_fit_warning is None else dict(
                        shared_fit_warning=self.shared_fit_warning,
                        shared_fit_diagnostics=self.shared_fit_diagnostics)))

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
                   complex_fit_failure=record["complex_fit_failure"],
                   shared_fit_warning=record.get("shared_fit_warning"),
                   shared_fit_diagnostics=record.get("shared_fit_diagnostics"),
                   tie_events=record.get("tie_events", {}),
                   comparisons=tuple(dict(step) for step in record.get("comparisons", ())))


def histograms(counts, positions, config: ReadModelConfig):
    """Per-sample histograms over the observed (depth, ALT) cells of every kept site, built one sample at a time.

    Returns (training float64[N, C], held-out float64[N, C], depth int64[C], alt int64[C],
    observations_used, high_depth_excluded); C = cells with depth 1..maximum_fit_depth seen anywhere.
    """
    width = config.maximum_fit_depth + 1
    training_sites = (np.asarray(positions) // config.fold_window_bp) % 4 < _TRAIN_FOLDS
    result = np.zeros((2, len(counts), width * width), np.float64)
    used = high = 0
    for sample in range(len(counts)):
        depth = counts[sample].sum(axis=1, dtype=np.int64)
        admitted = (depth > 0) & (depth <= config.maximum_fit_depth)
        used += int(admitted.sum())
        high += int((depth > config.maximum_fit_depth).sum())
        key = depth * width + counts[sample, :, 1]
        result[0, sample] = np.bincount(key[admitted & training_sites], minlength=width * width)
        result[1, sample] = np.bincount(key[admitted & ~training_sites], minlength=width * width)
    d, a = np.divmod(np.arange(width * width), width)
    valid = (d > 0) & (a <= d) & (result.sum(axis=(0, 1)) > 0)
    return result[0][:, valid], result[1][:, valid], d[valid], a[valid], used, high


def block_scores(counts, positions, config: ReadModelConfig, depth, alt, models):
    """(float64[B, len(models)], int64[B]): held-out log likelihood of each held-out stretch (folds 2/3 of one
    four-window cycle) under each model, and the stretch's admitted held-out cells (its exposure; 0 for a stretch whose
    sites have no admitted cell); rows in position order. Built one sample at a time from per-cell tables."""
    positions = np.asarray(positions)
    held = np.flatnonzero((positions // config.fold_window_bp) % 4 >= _TRAIN_FOLDS)
    if not len(held):
        return np.zeros((0, len(models))), np.zeros(0, np.int64)
    _, block = np.unique(positions[held] // (4 * config.fold_window_bp), return_inverse=True)
    blocks = int(block.max()) + 1
    width = config.maximum_fit_depth + 1
    lookup = np.full(width * width, -1, np.int64)
    lookup[np.asarray(depth) * width + np.asarray(alt)] = np.arange(len(depth))
    tables = [fits.log_predictive(model, depth, alt) for model in models]
    out = np.zeros((blocks, len(models)))
    exposure = np.zeros(blocks, np.int64)
    for sample in range(len(counts)):
        sample_depth = counts[sample, held].sum(axis=1, dtype=np.int64)
        admitted = (sample_depth > 0) & (sample_depth <= config.maximum_fit_depth)
        cell = lookup[sample_depth[admitted] * width + counts[sample, held, 1][admitted]]
        exposure += np.bincount(block[admitted], minlength=blocks)
        for column, table in enumerate(tables):
            out[:, column] += np.bincount(block[admitted], weights=table[sample, cell], minlength=blocks)
    return out, exposure


def calibrate(counts, positions, config: ReadModelConfig, *, threads: int):
    """Fit and select the contig's read model. Returns (ReadModelFit | None, CalibrationReport).

    Kernels run on the caller's Numba team; per-sample fits use ``threads`` Python threads.
    """
    if not config.enabled:
        return None, CalibrationReport(applied=False, selected=None,
                                       fallback_reason="calibration disabled",
                                       fold_observations=(), observations_used=0,
                                       high_depth_excluded=0)
    training, heldout, depth, alt, used, high = histograms(counts, positions, config)
    report = dict(fold_observations=(int(training.sum()), int(heldout.sum())), observations_used=used,
                  high_depth_excluded=high)
    if training.sum() == 0 or heldout.sum() == 0:
        return None, CalibrationReport(
            applied=False, selected=None,
            fallback_reason="the training and held-out folds need calibration observations", **report)
    try:
        shared = fits.fit_shared(training, depth, alt, config)
    except fits.SharedStationarityError as error:
        _LOG.warning("Read calibration uses fixed binomial fallback: %s "
                     "(%d shared starts; endpoints and distances recorded in the evidence report)",
                     error, len(error.diagnostics["starts"]))
        return None, CalibrationReport(applied=False, selected=None, fallback_reason=str(error),
                                       shared_fit_warning=error.diagnostics["reason"],
                                       shared_fit_diagnostics=error.diagnostics, **report)
    except (ValueError, RuntimeError) as error:
        _LOG.warning("Read calibration uses fixed binomial fallback: %s", error)
        return None, CalibrationReport(applied=False, selected=None, fallback_reason=str(error),
                                       shared_fit_warning=str(error), **report)
    warning = shared.details.get("shared_fit_warning")
    if warning is not None:
        diagnostics = shared.details["shared_fit_diagnostics"]
        _LOG.warning("Read calibration shared fit: %s "
                     "(%d starts; endpoints and distances recorded in the evidence report)",
                     warning, len(diagnostics["starts"]))
        report.update(shared_fit_warning=warning, shared_fit_diagnostics=diagnostics)
    candidates = {fits.BINOMIAL: shared}
    complex_failure = None
    try:
        pooled = fits.fit_pooled_samples(training, depth, alt, shared, threads)
        joint = fits.fit_joint(training, depth, alt, pooled, config)
        candidates.update({fits.SAMPLE_BINOMIAL: pooled, fits.SAMPLE_JOINT: joint})
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        complex_failure = str(error)
    blocks, exposure = block_scores(counts, positions, config, depth, alt, list(candidates.values()))
    scores = {name: (float(blocks[:, column].sum()), None) for column, name in enumerate(candidates)}
    selected, comparisons = select(blocks, list(candidates), config, exposure)
    near = sum(abs(step["z"] - config.selection_z) < _SELECTION_TIE_Z for step in comparisons
               if np.isfinite(step["z"]))
    tie_events = {"ev.read_model.select.band": dict(decisions=len(comparisons), inside_tie=int(near),
                                                    wide=0, boundary=0)}
    common = dict(fits=candidates, scores=scores, complex_fit_failure=complex_failure, tie_events=tie_events,
                  comparisons=tuple(comparisons), **report)
    if selected is None:
        return None, CalibrationReport(applied=False, selected=None,
                                       fallback_reason="no finite predictive fit", **common)
    return candidates[selected], CalibrationReport(applied=True, selected=selected,
                                                   fallback_reason=None, **common)


def _step(gains, exposure, config: ReadModelConfig):
    """(gain, SE, z, accepted) of one comparison over the stretches with admitted cells."""
    count = len(gains)
    gain = float(gains.sum())
    error = float(np.sqrt(count * gains.var(ddof=1))) if count > 1 else float("nan")
    z = gain / error if error > 0 else (float("inf") if gain > 0 else 0.)
    return gain, error, z, bool(count >= config.selection_minimum_blocks and gain > 0 and z >= config.selection_z)


def select(blocks, names, config: ReadModelConfig, exposure=None):
    """(selected name, comparison steps) from per-stretch held-out scores ``blocks`` [B, len(names)] of the nested
    candidates ``names`` (simplest first) and the stretches' admitted cells ``exposure`` (stretches without any are
    left out). Each more complex candidate in turn is compared with the currently accepted one and replaces it when its
    total held-out gain G > 0 and G >= selection_z SE, SE = sqrt(B var(stretch gains)) over B >=
    selection_minimum_blocks stretches (a rejected candidate does not stop the later comparisons). Each step also
    records an exposure-centred SE, sqrt(B/(B-1) sum((d_b - n_b g)^2)) with g = G / sum n_b, and how many of the
    leave-one-stretch-out decisions differ (sensitivity diagnostics; they decide nothing). None when the simplest
    candidate has no finite held-out score."""
    exposure = np.ones(len(blocks), np.int64) if exposure is None else np.asarray(exposure)
    used = exposure > 0
    blocks, exposure = blocks[used], exposure[used]
    totals = blocks.sum(axis=0)
    if not len(names) or not np.isfinite(totals[0]):
        return None, []
    current, steps = 0, []
    for nxt in range(1, len(names)):
        if not np.isfinite(totals[nxt]):
            continue
        gains = blocks[:, nxt] - blocks[:, current]
        gain, error, z, accepted = _step(gains, exposure, config)
        count = len(gains)
        if count > 1:
            per_cell = gain / max(1, int(exposure.sum()))
            centred = float(np.sqrt(count / (count - 1) * np.sum((gains - exposure * per_cell) ** 2)))
        else:
            centred = float("nan")
        flips = sum(_step(np.delete(gains, b), np.delete(exposure, b), config)[3] != accepted for b in range(count))
        steps.append(dict(candidate=names[nxt], against=names[current], gain=gain, standard_error=error,
                          blocks=count, z=z, accepted=accepted, exposure=int(exposure.sum()),
                          largest_block_share=float(exposure.max() / exposure.sum()) if count else 0.,
                          exposure_centred_error=centred, leave_one_out_flips=int(flips)))
        if accepted:
            current = nxt
    return names[current], steps


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
