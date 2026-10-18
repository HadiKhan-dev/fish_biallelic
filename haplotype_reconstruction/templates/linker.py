"""The naive linker: chain overlapping window haplotypes into chromosome-length founder templates.

Input: the discovered haplotypes of consecutive, overlapping physical windows of one chromosome (in order), each a
float64 [K, L, 2] array of (P(REF), P(ALT)) per row and site (0.5 / 0.5 where the window withholds the allele),
with the window's discovery-site mask. Steps:

1. Junction scores: for every row pair of two consecutive windows, the probabilistic mismatch percentage d of
   their alleles over the shared discovery sites gives similarity s = 0 if d > 50 else 2 (50 - d), scored
   100 (s / 100)^2; windows that do not overlap get score 1 for every pair. Edge weights are a third of these
   scores (base weighted overlap 1 : sample usage 2, and the sample-usage term never contributed).
2. Within-window similarity of rows (1 - min(1, 2 d / L) over the discovery sites, normalised by the diagonal):
   the usage penalty spreads to rows similar to a used one.
3. Up to ``count`` best paths through the window layers (dynamic programming over node + edge scores), each
   found path lowering the scores of its nodes (by node_penalty x similarity) and edges (by edge_penalty).
4. Each path is stitched into one haplotype: unique parts of every window in order, overlaps merged by adding
   the two windows' signed information (uniform prior).

Templates generate simulated founders; they are not inferred chromosomes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numba import njit

NODE_PENALTY = 10.0
EDGE_PENALTY = 10.0
GAP_SCORE = 1.0
# Junction edge weight: base combined the overlap score (weight 1) with a sample-usage score (weight 2) that never
# contributed (its inputs were stripped before matching), so every edge weighs a third of its overlap score.
EDGE_DIVISOR = 3.0
_START, _END = "I", "S"


@dataclass(frozen=True)
class WindowHaplotypes:
    """One window's discovered rows: positions int64 [L], kept bool [L] (discovery sites), haplotypes float64
    [K, L, 2] = (P(REF), P(ALT)) per site (0.5 / 0.5 where uncalled)."""

    positions: np.ndarray
    kept: np.ndarray
    haplotypes: np.ndarray


def haploid_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Expected allele mismatches of two [L, 2] probability rows: sum_i P1(0) P2(1) + P1(1) P2(0)."""
    return float(np.sum(np.einsum("ij,ik->ijk", first, second) * np.array([[0., 1.], [1., 0.]])))


def junction_scores(index: int, current: WindowHaplotypes, following: WindowHaplotypes) -> dict:
    """{((index, a), (index + 1, b)): score} of every row pair of two consecutive windows."""
    start = np.searchsorted(current.positions, following.positions[0])
    overlap = len(current.positions) - start
    rows_current, rows_following = range(len(current.haplotypes)), range(len(following.haplotypes))
    if overlap <= 0:
        return {((index, a), (index + 1, b)): GAP_SCORE for a in rows_current for b in rows_following}
    keep_current = current.kept[start:]
    keep_following = following.kept[:overlap]
    shared = min(len(keep_current), len(keep_following))
    keep_current, keep_following = keep_current[:shared], keep_following[:shared]
    scores = {}
    for a in rows_current:
        first = current.haplotypes[a, start:][:shared][keep_current]
        for b in rows_following:
            second = following.haplotypes[b, :overlap][:shared][keep_following]
            distance = 100 * haploid_distance(first, second) / len(first) if len(first) else 0
            similarity = 0 if distance > 50 else 2 * (50 - distance)
            scores[((index, a), (index + 1, b))] = 100 * (similarity / 100.0) ** 2
    return scores


def row_similarity(window: WindowHaplotypes) -> np.ndarray:
    """float64 [K, K]: 1 - min(1, 2 d(i, j) / L) over the discovery sites, normalised by sqrt of the diagonal."""
    rows = len(window.haplotypes)
    scores = np.zeros((rows, rows))
    for i in range(rows):
        for j in range(i, rows):
            first, second = window.haplotypes[i][window.kept], window.haplotypes[j][window.kept]
            if len(first) == 0:
                scores[i, j] = 0
            else:
                scores[i, j] = 1.0 - min(1.0, 2.0 * haploid_distance(first, second) / len(first))
    scores = scores + scores.T - np.diag(scores.diagonal())
    diagonal = np.sqrt(scores.diagonal())
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = scores / diagonal
        scores = scores / diagonal.reshape(1, -1).T
    return scores


def _best_scores(layers, node_scores, edge_scores):
    """Per layer {node: best total score of a path from it to the end} (reverse dynamic programming)."""
    scorings = [{_END: 0}]
    for layer in range(len(layers) - 2, -1, -1):
        following = scorings[-1]
        current = {}
        for node in layers[layer]:
            best = -np.inf
            for other in layers[layer + 1]:
                if (node, other) in edge_scores:
                    score = node_scores[node] + edge_scores[(node, other)] + following[other]
                    if score > best:
                        best = score
            current[node] = best
        scorings.append(current)
    return scorings[::-1]


def _optimal_path(scorings, layers, node_scores, edge_scores):
    """The path the scorings imply, from the start node; stops early where none continues."""
    path, node = [_START], _START
    for layer in range(len(layers) - 1):
        score = scorings[layer][node]
        if score == -np.inf:
            break
        for other in layers[layer + 1]:
            if (node, other) in edge_scores:
                remaining = score - (node_scores[node] + edge_scores[(node, other)])
                if abs(remaining - scorings[layer + 1][other]) < 1e-9:
                    path.append(other)
                    node = other
                    break
    return path


def chain_paths(windows, scores, similarities, count: int) -> list[list[tuple[int, int]]]:
    """Up to ``count`` paths (one (window, row) node per window) through the junction-scored windows."""
    layers = [[(index, row) for row in range(len(window.haplotypes))] for index, window in enumerate(windows)]
    edge_scores = dict(scores)
    node_scores = {_START: 0, _END: 0}
    for layer in layers:
        for node in layer:
            node_scores[node] = 0
    for row in range(len(layers[0])):
        edge_scores[(_START, (0, row))] = 0
    for row in range(len(layers[-1])):
        edge_scores[((len(layers) - 1, row), _END)] = 0
    padded = [[_START], *layers, [_END]]
    paths = []
    for _ in range(count):
        path = _optimal_path(_best_scores(padded, node_scores, edge_scores), padded, node_scores, edge_scores)
        if len(path) < len(padded):
            break
        for layer, used in path[1:-1]:
            reductions = NODE_PENALTY * similarities[layer][used, :]
            for row in range(len(reductions)):
                node_scores[(layer, row)] -= reductions[row]
        for edge in zip(path[1:-2], path[2:-1]):
            if edge in edge_scores:
                edge_scores[edge] -= EDGE_PENALTY
        paths.append(path[1:-1])
    return paths


@njit(cache=True, fastmath=True)
def _information(probability):
    """Signed information of P(ALT) = p: sign(p - 1/2) (1 - binary entropy), +-1 at certainty."""
    if probability >= 1.0:
        return 1.0
    if probability <= 0.0:
        return -1.0
    sign = -1.0 if probability < 0.5 else 1.0
    entropy = -(1 - probability) * math.log2(1 - probability) - probability * math.log2(probability)
    return sign * (1.0 - entropy)


@njit(cache=True, fastmath=True)
def _add_information(first, second):
    """Relativistic addition of signed informations (0 when they cancel exactly)."""
    if abs(first * second + 1.0) < 1e-9:
        return 0.0
    return (first + second) / (1.0 + first * second)


@njit(cache=True, fastmath=True)
def merge_overlap(first, second):
    """[n, 2] (P(REF), P(ALT)) of two windows' rows merged site by site: their informations added and the sum
    inverted by bisection (60 halvings)."""
    merged = np.empty_like(first)
    for site in range(first.shape[0]):
        target = _add_information(_information(first[site, 1]), _information(second[site, 1]))
        low, high = 0.0, 1.0
        for _ in range(60):
            midpoint = (low + high) * 0.5
            if _information(midpoint) < target:
                low = midpoint
            else:
                high = midpoint
        probability = (low + high) * 0.5
        merged[site, 0] = 1.0 - probability
        merged[site, 1] = probability
    return merged


def stitch(windows, path) -> tuple[np.ndarray, np.ndarray]:
    """(positions int64 [L], haplotype float64 [L, 2]) of one path: unique parts in order, overlaps merged."""
    pieces, locations = [], []
    following_start = [int(np.searchsorted(windows[index].positions, windows[index + 1].positions[0]))
                       for index in range(len(windows) - 1)]
    for index, window in enumerate(windows):
        data = window.haplotypes[path[index][1]]
        positions = window.positions
        trim_start = 0
        if index:
            previous = windows[index - 1].positions
            trim_start = max(0, len(previous) - int(np.searchsorted(previous, positions[0])))
        trim_end = len(data)
        overlap_next = 0
        if index < len(windows) - 1:
            overlap_next = len(positions) - following_start[index]
            if overlap_next > 0:
                trim_end = following_start[index]
        if trim_start < len(data):
            pieces.append(data[trim_start:trim_end])
            locations.append(positions[trim_start:trim_end])
        if overlap_next > 0:
            current = data[trim_end:]
            following = windows[index + 1].haplotypes[path[index + 1][1]][:overlap_next]
            shared = min(len(current), len(following))
            pieces.append(merge_overlap(np.ascontiguousarray(current[:shared]),
                                        np.ascontiguousarray(following[:shared])))
            locations.append(positions[trim_end:trim_end + shared])
    return (np.concatenate(locations).astype(np.int64),
            np.concatenate(pieces) if pieces else np.zeros((0, 2)))


def link_windows(windows, count: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """(positions, [haplotype float64 [L, 2]] x <= count) chained from consecutive windows (module docstring);
    empty when fewer than two windows hold haplotypes."""
    windows = [window for window in windows if len(window.positions) and len(window.haplotypes)
               and window.kept.any()]
    if len(windows) < 2:
        return np.zeros(0, np.int64), []
    scores = {}
    for index in range(len(windows) - 1):
        for edge, score in junction_scores(index, windows[index], windows[index + 1]).items():
            scores[edge] = score / EDGE_DIVISOR
    similarities = [row_similarity(window) for window in windows]
    stitched = [stitch(windows, path) for path in chain_paths(windows, scores, similarities, count)]
    stitched = [item for item in stitched if len(item[0])]
    if not stitched:
        return np.zeros(0, np.int64), []
    longest = int(np.argmax([len(positions) for positions, _ in stitched]))
    return stitched[longest][0], [haplotype for _, haplotype in stitched]
