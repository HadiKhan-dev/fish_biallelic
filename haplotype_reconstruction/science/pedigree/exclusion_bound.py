"""Completed-window bounds on the chromosome-averaged Mendelian exclusion e-value.

One FINISHED window/bet contributes P/(chromosomes * windows * bets) to
that average. Unfinished products never justify stopping; uncertified rows
finish the exact sum in its fixed reduction order. This is not a
maximum-window model.

Method: docs/methods/pedigree.md#full-marker-mendelian-exclusion-of-m0-and-m1-counts
"""

from __future__ import annotations

from collections.abc import Callable

import numba
import numpy as np
from numba import prange

from ...contracts.pedigree import ExclusionEvidence
from ...core.errors import InputError
from ...core.kernels import kernel
from . import exclusion
from .exclusion_parameters import ExclusionParameters
from .kernels.exclusion import OVERLOADS

MODEL = "completed-window-observation-exclusion-v1"  # 4.22 A11: cells mixed against the cohort law


def observation_words(observed):
    packed = np.packbits(observed, axis=1, bitorder="little")
    words = np.zeros((len(observed), (packed.shape[1] + 7) // 8), np.uint64)
    words.view(np.uint8)[:, : packed.shape[1]] = packed
    return words


@kernel(overloads=OVERLOADS["exclusion_bound._popcount"], cache=True, inline="always", nogil=True)
def _popcount(value):
    value -= (value >> np.uint64(1)) & np.uint64(0x5555555555555555)
    value = (value & np.uint64(0x3333333333333333)) + ((value >> np.uint64(2)) & np.uint64(0x3333333333333333))
    value = (value + (value >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return int((value * np.uint64(0x0101010101010101)) >> np.uint64(56))


@kernel(overloads=OVERLOADS["exclusion_bound._observed_counts"], cache=True, parallel=True, nogil=True)
def _observed_counts(words, rows):
    result = np.zeros(len(rows), np.int64)
    for row in prange(len(rows)):
        a, b = rows[row]
        count = 0
        for word in range(words.shape[1]):
            count += _popcount(words[a, word] & words[b, word])
        result[row] = count
    return result


@kernel(overloads=OVERLOADS["exclusion_bound._completed_windows"], cache=True, parallel=True, nogil=True)
def _completed_windows(
    gl,
    observed,
    cohort,
    rates,
    rows,
    starts,
    stops,
    start_index,
    original_window,
    bets,
    log_denominator,
    log_chromosomes,
    threshold,
    certifiable,
):
    """Window/bet log products of each dyad (rows [R, 2]) under the observation model: every site's ratio from the
    two samples' likelihoods gl [N, S, 3] mixed against cohort [S, 3] at their own rates [N] (exclusion._ratio).
    The ratio depends on the site's cohort law, so there is no cache over repeated likelihood patterns; dyads are
    independent and run in parallel."""
    lower = np.full(len(rows), -np.inf)
    visited = np.zeros(len(rows), np.int64)
    is_certificate = np.zeros(len(rows), np.bool_)
    eps = np.finfo(np.float64).eps
    for row in prange(len(rows)):
        a, b = rows[row]
        ea = rates[a]
        eb = rates[b]
        prefix = np.zeros(len(bets))
        absolute = np.zeros(len(bets))
        saved = np.empty((len(starts), len(bets)))
        completed = np.empty((len(bets), len(stops)))
        first = 0
        last = 0
        certified = False
        can_certify = certifiable[row]
        for site in range(gl.shape[1] + 1):
            while first < len(starts) and starts[first] == site:
                saved[first] = prefix
                first += 1
            while last < len(stops) and stops[last] == site:
                for h in range(len(bets)):
                    value = prefix[h] - saved[start_index[last], h]
                    completed[h, original_window[last]] = value
                    if not can_certify:
                        continue
                    # Conservative allowance for prefix summation, subtraction,
                    # logarithms and the fixed positive-term reduction.
                    guard = (
                        64
                        * eps
                        * (site + len(stops) * len(bets) + 1)
                        * (absolute[h] + abs(value) + abs(log_denominator) + abs(log_chromosomes) + abs(threshold) + 1)
                    )
                    candidate = value - log_denominator - guard
                    if candidate - log_chromosomes - guard >= threshold:
                        lower[row] = candidate
                        certified = True
                        break
                last += 1
                if certified:
                    break
            if certified or site == gl.shape[1]:
                visited[row] = site
                break
            if not observed[a, site] or not observed[b, site]:
                continue
            ratio = exclusion._ratio(
                gl[a, site, 0],
                gl[a, site, 1],
                gl[a, site, 2],
                gl[b, site, 0],
                gl[b, site, 1],
                gl[b, site, 2],
                cohort[site, 0],
                cohort[site, 1],
                cohort[site, 2],
                ea,
                eb,
            )
            for h in range(len(bets)):
                value = exclusion._log_bet(ratio, bets[h])
                prefix[h] += value
                if can_certify:
                    absolute[h] += abs(value)
        is_certificate[row] = certified
        if not certified:
            # Every window has completed. Reduce in the declared bet/window
            # order, reproducing the reference float64 value without a rescan.
            combined = -np.inf
            for h in range(len(bets)):
                for window in range(len(stops)):
                    combined = np.logaddexp(combined, completed[h, window])
            lower[row] = combined - log_denominator
    return lower, visited, is_certificate


def score_contig(
    contig: str,
    likelihoods: np.ndarray,
    observed: np.ndarray,
    positions: np.ndarray,
    dyads: np.ndarray,
    config: ExclusionParameters,
    log_weight: float,
    sample_count: int,
    claim_threads: Callable[[], int | None] = lambda: None,
    *,
    cohort: np.ndarray,
    sample_rates: np.ndarray,
    certifiable: np.ndarray | None = None,
) -> tuple[ExclusionEvidence, dict[str, int | float]]:
    """Bounded scorer with exact fallback; a certificate proves log w_k + log E_k >= threshold (``log_weight`` =
    log w_k, the contig's fixed weight in the genome e-value, DESIGN_417 D9).

    4.22 A11 (owner, 18 Nov; lab422/r4a): every site is scored under the observation model (exclusion._ratio), each
    cell mixed against the site's cohort genotype frequencies ``cohort`` [S, 3] at its sample's own outlier rate
    ``sample_rates`` [N] (the painting calibration's per-sample point estimates), with no replacement term and no
    floor. On identical inputs every site ratio is at least (1 - e_x)(1 - e_y) times 4.21's (its allowance
    max(0.01, e_x + e_y) bounded the redraws from above), so the statistic can add exclusions relative to 4.21; it
    claims no error control for products over dependent sites beyond 4.20's test. Shared reads break its per-cell
    model: the caller withholds certification there (``certifiable``)."""
    if not (np.isfinite(log_weight) and log_weight <= 0.0) or sample_count < 1:
        raise InputError("a finite log weight <= 0 and a positive sample count are required")
    rates = np.ascontiguousarray(sample_rates, dtype=np.float64)
    if rates.ndim != 1 or rates.shape[0] != likelihoods.shape[0]:
        raise InputError("per-sample outlier rates must cover every sample of the likelihoods")
    if not np.all((rates >= 0.0) & (rates < 1.0)):
        raise InputError("per-sample outlier rates are probabilities in [0, 1)")
    cohort = np.array(cohort, dtype=np.float64, order="C")  # small [S, 3]: a writable contiguous copy
    if cohort.shape != (likelihoods.shape[1], 3):
        raise InputError("cohort genotype frequencies must cover every site of the likelihoods")
    rates = rates.copy()
    certifiable = (np.ones(len(dyads), dtype=np.bool_) if certifiable is None
                   else np.array(certifiable, dtype=np.bool_, order="C"))
    threshold = exclusion.log_threshold(sample_count, config.alpha)
    spans = exclusion.physical_windows(positions, config.window_widths_bp)
    starts = np.unique(spans[:, 0])
    order = np.argsort(spans[:, 1], kind="stable")
    stops = spans[order, 1]
    start_index = np.searchsorted(starts, spans[order, 0])
    bets = np.asarray(config.bet_sizes, dtype=np.float64)
    log_denominator = np.log(len(spans) * len(bets))
    lower = np.empty(len(dyads))
    visited = np.empty(len(dyads), np.int64)
    certified = np.empty(len(dyads), np.bool_)
    claim_threads()
    words = observation_words(observed)
    jointly_observed = _observed_counts(words, np.array(dyads, dtype=np.int64, order="C"))
    del words
    for start in range(0, len(dyads), exclusion.DYAD_BATCH):
        claim_threads()
        stop = min(start + exclusion.DYAD_BATCH, len(dyads))
        # Deal the batch's dyads round-robin over the team's contiguous prange blocks: neighbouring dyads (one child's
        # candidates) cost alike, and certificates make costs very unequal. Dyads are independent, so the values
        # do not depend on the dealing (scatter back by the same permutation).
        team = max(1, numba.get_num_threads())
        dealt = np.concatenate([np.arange(start + k, stop, team) for k in range(min(team, stop - start))])
        lower[dealt], visited[dealt], certified[dealt] = _completed_windows(
            likelihoods,
            observed,
            cohort,
            rates,
            np.array(dyads[dealt], dtype=np.int64, order="C"),
            starts,
            stops,
            start_index,
            order,
            bets,
            log_denominator,
            -float(log_weight),
            threshold,
            np.ascontiguousarray(certifiable[dealt]),
        )
    upper = np.where(certified, np.inf, lower)
    work = dict(
        dyads=len(dyads),
        certified=int(certified.sum()),
        exact_fallback=int((~certified).sum()),
        certificate_visited_sites=int(visited.sum()),
        full_dyad_sites=int(len(dyads) * len(positions)),
        packed_words=int(len(dyads) * ((len(positions) + 63) // 64)),
        final_log_weight=float(log_weight),
        threshold=threshold,
    )
    return ExclusionEvidence(contig, dyads, lower, jointly_observed, len(positions), upper,
                             log_weight=float(log_weight), certifiable=certifiable), work
