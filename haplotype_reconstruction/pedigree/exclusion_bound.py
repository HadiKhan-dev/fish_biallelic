"""Completed-window bounds on the original averaged Mendelian exclusion e-value.

One FINISHED window/bet contributes P/(chromosomes * windows * bets) to
that average. Unfinished products never justify stopping; uncertified rows
finish the original exact sum in its original reduction order. The reference
scorer stays available. This is not a maximum-window model.
"""
from __future__ import annotations
import math
import numpy as np
from numba import njit, prange
from . import exclusion
from .records import ExclusionEvidence

MODEL = 'completed-window-exclusion-v1'


def observation_words(observed):
    packed = np.packbits(observed, axis=1, bitorder='little')
    words = np.zeros((len(observed), (packed.shape[1] + 7) // 8), np.uint64)
    words.view(np.uint8)[:, :packed.shape[1]] = packed
    return words


@njit(cache=True, inline='always')
def _popcount(value):
    value -= (value >> np.uint64(1)) & np.uint64(0x5555555555555555)
    value = ((value & np.uint64(0x3333333333333333))
             + ((value >> np.uint64(2)) & np.uint64(0x3333333333333333)))
    value = (value + (value >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return int((value * np.uint64(0x0101010101010101)) >> np.uint64(56))


@njit(cache=True, parallel=True)
def _observed_counts(words, rows):
    result = np.zeros(len(rows), np.int64)
    for row in prange(len(rows)):
        a, b = rows[row]
        count = 0
        for word in range(words.shape[1]):
            count += _popcount(words[a, word] & words[b, word])
        result[row] = count
    return result


@njit(cache=True, parallel=True)
def _completed_windows(gl, observed, codes, catalog, counts, rows, starts, stops, start_index, original_window,
                       delta, bets, log_denominator, log_chromosomes, threshold):
    lower = np.full(len(rows), -np.inf)
    visited = np.zeros(len(rows), np.int64)
    is_certificate = np.zeros(len(rows), np.bool_)
    eps = np.finfo(np.float64).eps
    for row in prange(len(rows)):
        a, b = rows[row]
        encoded = counts[a] >= 0 and counts[b] >= 0
        width = counts[b] if encoded else 0
        entries = counts[a] * width if encoded else 0
        table = np.empty((entries, len(bets)))
        seen = np.zeros(entries, np.bool_)
        prefix = np.zeros(len(bets))
        absolute = np.zeros(len(bets))
        saved = np.empty((len(starts), len(bets)))
        completed = np.empty((len(bets), len(stops)))
        first = 0
        last = 0
        certified = False
        for site in range(gl.shape[1] + 1):
            while first < len(starts) and starts[first] == site:
                saved[first] = prefix
                first += 1
            while last < len(stops) and stops[last] == site:
                for h in range(len(bets)):
                    value = prefix[h] - saved[start_index[last], h]
                    completed[h, original_window[last]] = value
                    # Conservative allowance for prefix summation, subtraction,
                    # logarithms and the original positive-term reduction.
                    guard = (64 * eps * (site + len(stops) * len(bets) + 1)
                             * (absolute[h] + abs(value) + abs(log_denominator)
                                + abs(log_chromosomes) + abs(threshold) + 1))
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
            if encoded:
                ca, cb = codes[a, site], codes[b, site]
                pair = ca * width + cb
                if not seen[pair]:
                    ratio = exclusion._ratio(
                        catalog[a, ca, 0], catalog[a, ca, 1], catalog[a, ca, 2],
                        catalog[b, cb, 0], catalog[b, cb, 1], catalog[b, cb, 2], delta)
                    for h in range(len(bets)):
                        table[pair, h] = exclusion._log_bet(ratio, bets[h])
                    seen[pair] = True
                for h in range(len(bets)):
                    value = table[pair, h]
                    prefix[h] += value
                    absolute[h] += abs(value)
            else:
                ratio = exclusion._ratio(
                    gl[a, site, 0], gl[a, site, 1], gl[a, site, 2],
                    gl[b, site, 0], gl[b, site, 1], gl[b, site, 2], delta)
                for h in range(len(bets)):
                    value = exclusion._log_bet(ratio, bets[h])
                    prefix[h] += value
                    absolute[h] += abs(value)
        is_certificate[row] = certified
        if not certified:
            # Every window has completed. Reduce in the original bet/window
            # order, reproducing the reference float64 value without a rescan.
            combined = -np.inf
            for h in range(len(bets)):
                for window in range(len(stops)):
                    combined = np.logaddexp(combined, completed[h, window])
            lower[row] = combined - log_denominator
    return lower, visited, is_certificate


def score_contig(contig, likelihoods, observed, positions, dyads, config,
                 genome_contigs, sample_count, claim_threads=lambda: None):
    """Bounded scorer with exact fallback and the required FINAL average divisor."""
    if genome_contigs < 1 or sample_count < 1:
        raise ValueError('positive chromosome and sample counts required')
    threshold = math.log(2 * sample_count / config.alpha)
    spans = exclusion.physical_windows(positions, config.window_widths_bp)
    starts = np.unique(spans[:, 0])
    order = np.argsort(spans[:, 1], kind='stable')
    stops = spans[order, 1]
    start_index = np.searchsorted(starts, spans[order, 0])
    bets = np.asarray(config.bet_sizes, dtype=np.float64)
    log_denominator = np.log(len(spans) * len(bets))
    lower = np.empty(len(dyads))
    visited = np.empty(len(dyads), np.int64)
    certified = np.empty(len(dyads), np.bool_)
    claim_threads()
    codes, catalog, counts = exclusion._encode_patterns(likelihoods, observed, exclusion.MAX_PATTERNS)
    words = observation_words(observed)
    jointly_observed = _observed_counts(words, dyads)
    del words
    for start in range(0, len(dyads), exclusion.DYAD_BATCH):
        claim_threads()
        stop = start + exclusion.DYAD_BATCH
        lower[start:stop], visited[start:stop], certified[start:stop] = _completed_windows(
            likelihoods, observed, codes, catalog, counts, dyads[start:stop], starts, stops, start_index, order,
            float(config.genotype_replacement_probability), bets, log_denominator,
            np.log(genome_contigs), threshold)
    upper = np.where(certified, np.inf, lower)
    fallback_samples = int(np.count_nonzero(counts < 0))
    work = dict(dyads=len(dyads), certified=int(certified.sum()),
                exact_fallback=int((~certified).sum()),
                certificate_visited_sites=int(visited.sum()),
                full_dyad_sites=int(len(dyads) * len(positions)),
                packed_words=int(len(dyads) * ((len(positions) + 63) // 64)),
                final_chromosome_divisor=genome_contigs, threshold=threshold)
    return ExclusionEvidence(contig, dyads, lower, jointly_observed, len(positions),
                             fallback_samples, upper), work


def combine(evidence):
    """Original chromosome average with lower and upper endpoints."""
    lower, observed = exclusion.combine(evidence)
    upper = (np.logaddexp.reduce(np.stack([
        value.log_e if value.log_e_upper is None else value.log_e_upper
        for value in evidence]), axis=0) - np.log(len(evidence)))
    return lower, upper, observed
