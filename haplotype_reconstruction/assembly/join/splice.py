"""Exact scores of all two-founder suffix exchanges from boundary messages.

The unchanged panel supplies a forward prefix F_b and a backward suffix G_b at every
block boundary b. G_b includes the switch into the suffix. Swapping founders a and c
only relabels suffix states, so the exchanged panel scores max_q F_b(q) + G_b(pi(q)),
where pi transposes a and c in the unordered state q.

Only 2K-2 states move under a transposition. Sorting the unchanged F+G terms once
per boundary finds the best fixed state after at most 2K-2 rejected terms, and all
moved terms take O(K). All exchanges therefore cost O(S bins K^2 + S blocks K^3)
and return O(S blocks K^2) values. Prefix and suffix sums use different addition
orders from a full forward scan; they represent the same objective in float64.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from .objective import _advance, _gather_states, _start, max_range, pair_lists, widest_block


def pair_column(founders: int, first: int, second: int) -> int:
    """Column of the pair first < second, ordered (0,1), (0,2), ..., (K-2,K-1)."""
    return first * (2 * founders - first - 1) // 2 + second - first - 1


@njit(cache=True)
def _junction_scores(prefix, suffix, first, second, index, output):
    """Score every founder transposition at one sample's boundary in O(K^3)."""
    founders = len(index)
    terms = prefix + suffix
    order = np.argsort(-terms)
    column = 0
    for a in range(founders):
        for b in range(a + 1, founders):
            best = -np.inf
            for position in range(len(order)):
                state = order[position]
                x, y = first[state], second[state]
                if ((x != a and x != b and y != a and y != b)
                        or (x == a and y == b)):
                    best = terms[state]
                    break
            best = max(best, prefix[index[a, a]] + suffix[index[b, b]])
            best = max(best, prefix[index[b, b]] + suffix[index[a, a]])
            for other in range(founders):
                if other != a and other != b:
                    best = max(best, prefix[index[a, other]] + suffix[index[b, other]])
                    best = max(best, prefix[index[b, other]] + suffix[index[a, other]])
            output[column] = best
            column += 1


@njit(cache=True, parallel=True, nogil=True)
def score_splices(flat, block_start, block_haps, block_bins, rows, penalty, samples):
    """Per-sample score of every suffix exchange: (S, B-1, K(K-1)/2).

    Boundary index b-1 exchanges rows on blocks b..B-1; pair columns follow
    pair_column. Rows retain their slots, including duplicates during seed refits.
    The caller performs the ordinary admissibility checks before using a move.
    """
    founders, blocks = rows.shape
    first, second = pair_lists(np.ones(founders, np.bool_))
    states = len(first)
    index = np.empty((founders, founders), np.int64)
    for state in range(states):
        index[first[state], second[state]] = state
        index[second[state], first[state]] = state
    output = np.empty((samples, max(0, blocks - 1), founders * (founders - 1) // 2))
    widest = widest_block(block_bins)
    for sample in prange(samples):
        # Per-thread boundary prefixes; suffixes need only one running state vector.
        prefixes = np.empty((max(0, blocks - 1), states))
        current = np.zeros(states)
        gathered = np.empty((widest, states))
        started = False
        best = 0.0
        for block in range(blocks - 1):
            haps, bins = block_haps[block], block_bins[block]
            base = block_start[block] + sample * (haps * (haps + 1) // 2) * bins
            _gather_states(flat, base, haps, bins, rows, first, second, block, gathered)
            for marker in range(bins):
                if started:
                    best = _advance(current, states, gathered, marker, best - penalty)
                else:
                    best = _start(current, states, gathered, marker)
                    started = True
            prefixes[block] = current
        suffix = np.zeros(states)
        for block in range(blocks - 1, 0, -1):
            haps, bins = block_haps[block], block_bins[block]
            base = block_start[block] + sample * (haps * (haps + 1) // 2) * bins
            _gather_states(flat, base, haps, bins, rows, first, second, block, gathered)
            for marker in range(bins - 1, -1, -1):
                for state in range(states):
                    suffix[state] += gathered[marker, state]
                switch = max_range(suffix, 0, states, -np.inf) - penalty
                for state in range(states):
                    suffix[state] = max(suffix[state], switch)
            _junction_scores(prefixes[block - 1], suffix, first, second, index,
                             output[sample, block - 1])
    return output


def splice_scores(layout, rows, penalty):
    """All suffix-exchange scores for one panel; build lazily once per certification pass."""
    return score_splices(*layout.arrays(), np.ascontiguousarray(rows, np.int64),
                         float(penalty), int(layout.samples))


def workspace_bytes(samples: int, blocks: int, founders: int, widest: int, threads: int) -> int:
    """Retained sample scores plus per-thread boundary prefixes and state scratch."""
    states = founders * (founders + 1) // 2
    pairs = founders * (founders - 1) // 2
    active = min(samples, max(1, threads))
    return int(8 * (samples * max(0, blocks - 1) * pairs
                    + active * (max(0, blocks - 1) + widest + 6) * states
                    + founders * founders + 2 * states))
