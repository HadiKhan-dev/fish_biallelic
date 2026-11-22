"""The frames' emissions of the map-scale estimate, with the target's read state (LR phase 2, Tier 3).

A frame (map_scale_frames) is a bin b where target member i's four parental homologue classes are labelled: parent
0's (lab 0, lab 1), parent 1's (lab 2, lab 3). Its emission is a transfer matrix [C, C] of the target's read class
per emitted state, in the TRANSMISSION frame: homologue 1 carries parent 0's transmitted copy, homologue 2 parent 1's
(a physical homologue keeps its parent through every crossover, so the class carries across frames without a
permutation). The ordered pair (x on homologue 1, y on homologue 2) has

    W_b(x, y) = exp(min(e_b(x, y) - m, 300) + l_b(x, y) - t) M~_b(x, y),

e_b the painting's tempered scalar bin emission of the target at the PATH strength (the frames' scalar emission
before the read state; science.pedigree.prepare.path_correlation), M_b with log scale l_b the bin's read matrix of the
unordered pair (painting.read_factor: the class at the previous bin's last selected site carried through the bin's
sites), M~_b(x, y) = M_b with both class indices permuted by J (painting.read_factor.swap_classes) when x > y (Tier 1's
convention), m the largest scalar e of the four full-sib configurations (today's ratios to the frame's maximum, with
today's 300-nat overflow guard on the scalar ratio, applied before the read factor) and t the largest e - m + l of the
four (one common scale for every column of the frame: it cancels from every likelihood ratio). The 26 columns
(frame_matrices):

- 5 s1 + s2, s1, s2 < 4: W(lab s1, lab s2) (track 1 on homologue 1, track 2 on homologue 2); the four full-sib
  configurations are columns 2, 3, 7, 8 (map_scale_frames.FULL_SIB_COLUMNS);
- 'other' (symbol 4): the mean over the labels outside the four, (lab s1, y), (y, lab s2) or (y, y'); zero when
  every label is one of the four;
- misfit (25): the mean over all unordered pairs U of W with both homologue orientations equally weighted,
  exp(min(e_U - m, 300) + l_U - t) (M_U + J M_U J) / 2. The misfit hypothesis replaces the configuration by a pair drawn
  from today's law (uniform over unordered pairs) that carries no parent assignment; the target's reads still follow
  its read state, so the class crosses the frame by the law's average transfer (the chains' mixture with the
  configurations is linear in the transfer).

Between consecutive frames at bins b' < b of one sequence the class crosses the undetermined bins b' + 1 .. b - 1 by
the label-free operator G = prod of the composed class operators of their sites (their alleles are not read, as the
scalar frames skip those bins: the read factor is 1 at an unread site), folded into the later frame's matrices:
G W_b. A sequence starts from the class law at its first frame's gap start (entry: the law at the previous frame's
bin end, or at the frame's own bin start for a first frame); the read state's site laws satisfy laws[l] =
laws[l - 1] T_l, so a sequence cut at its leading frames (map_scale_frames.file_frames) starts from the exact
marginal.

Without read state (read_classes 1) or for a sample that bypasses it (painting.read_inputs.ReadEmissionBuilder:
deep at every site, or a library without detected read sharing) the matrices are 1 x 1, l = 0, entry 1: today's
scalar frames exactly (C = 1 of the same kernels). Bypassed samples among read-state samples are embedded in class 0
(the builder's own identity read state).

Sources: ScalarFrames serves the painting's packed scalar emissions; ReadFrames rebuilds a component's emissions with
the painting's read builder for the requested (target, bin, labels) only, and caches them by the canonical split of
the four labels (a relabelling of the symbols; the homologue frame is the ordered pair's), so the guard's later builds
reuse them: no all-pair read matrices are held between contigs.

Method: docs/methods/pedigree.md#map-scale.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numba
import numpy as np
from numba import prange, types

from haplotype_reconstruction.science.painting.parameters import READ_STATE_BATCH_BYTES

from ...core.errors import InputError
from ...core.kernels import kernel
from ..painting import api as painting_hmm

__all__ = ["FrameSites", "ContigReads", "ScalarFrames", "ReadFrames", "frame_matrices", "bin_operators",
           "link_frames", "canonical_split", "FRAME_COLUMNS"]

FRAME_COLUMNS = 26     # 25 track-symbol pairs, then misfit
_CLIP = 300.0          # today's overflow guard on a frame's scalar ratios (nats above the configurations' best)


def _a(dtype, ndim):
    return types.Array(dtype, ndim, "C", readonly=False, aligned=True)


_b1 = _a(types.boolean, 1)
_i64_1, _i64_2 = _a(types.int64, 1), _a(types.int64, 2)
_f2, _f3, _f4 = _a(types.float64, 2), _a(types.float64, 3), _a(types.float64, 4)
_m5 = _a(types.float32, 5)
_ro_bins = types.Array(types.int64, 1, "C", readonly=True)


@kernel(overloads=((_f3, _f3, _m5, _i64_1, _i64_2, types.boolean, types.int64, types.int64, types.int64, types.int64,
                    types.int64, types.int64, types.float64, types.float64, _f4),), cache=True, inline="always",
        nogil=True)
def _add_pair(emission, read_log, read, perm, pair_index, scalar, row, block, x, y, request, column, best, top, out):
    """out[request, column] += exp(min(e - best, 300) + l - top) M~(x, y) of the ordered pair (x on homologue 1, y on
    homologue 2) at bin ``block`` of batch row ``row``; C = 1 (``scalar``): exp(min(e - best, 300)) alone."""
    u = pair_index[x, y]
    if scalar:
        out[request, column, 0, 0] += math.exp(min(emission[row, u, block] - best, _CLIP))
        return
    scale = math.exp(min(emission[row, u, block] - best, _CLIP) + read_log[row, u, block] - top)
    C = out.shape[2]
    if x <= y:
        for i in range(C):
            for j in range(C):
                out[request, column, i, j] += scale * float(read[row, u, block, i, j])
    else:
        for i in range(C):
            for j in range(C):
                out[request, column, i, j] += scale * float(read[row, u, block, perm[i], perm[j]])


@kernel(overloads=((_f3, _f3, _m5, _i64_1, _i64_1, _i64_2, _i64_1, types.boolean, _i64_1, _i64_1, _i64_2, _f4),),
        cache=True, parallel=True, nogil=True)
def frame_matrices(emission, read_log, read, pair_first, pair_second, pair_index, perm, scalar, request_row,
                   request_bin, request_labels, out):
    """out[Q, 26, C, C] (zeroed by the caller): every request's 26 frame matrices (module docstring), without the
    gap operator. ``emission`` / ``read_log`` float64 [n, U, B] and ``read`` float32 [n, U, B, C, C] per unordered
    pair (upper-triangle order: pair_first <= pair_second, pair_index [S, S]); request_row indexes their first axis,
    request_labels [Q, 4] the four distinct labels (parent 0's two homologues, then parent 1's). ``scalar``: C = 1
    without read state (read_log and read unused)."""
    n_requests = len(request_row)
    n_states = pair_index.shape[0]
    n_pairs = len(pair_first)
    C = out.shape[2]
    for request in prange(n_requests):
        row = request_row[request]
        block = request_bin[request]
        # the scale: the configurations' best scalar emission (best), then their best e - best + l (top)
        best = -np.inf
        for h0 in range(2):
            for h1 in range(2, 4):
                u = pair_index[request_labels[request, h0], request_labels[request, h1]]
                if emission[row, u, block] > best:
                    best = emission[row, u, block]
        top = -np.inf
        if scalar:
            top = 0.0
        else:
            for h0 in range(2):
                for h1 in range(2, 4):
                    u = pair_index[request_labels[request, h0], request_labels[request, h1]]
                    value = emission[row, u, block] - best + read_log[row, u, block]
                    if value > top:
                        top = value
        inside = np.zeros(n_states, dtype=np.bool_)
        for h in range(4):
            inside[request_labels[request, h]] = True
        outside = n_states - 4
        for s1 in range(5):
            for s2 in range(5):
                column = 5 * s1 + s2
                if s1 < 4 and s2 < 4:
                    _add_pair(emission, read_log, read, perm, pair_index, scalar, row, block,
                              request_labels[request, s1], request_labels[request, s2], request, column, best, top, out)
                    continue
                if outside == 0:
                    continue
                if s1 < 4:
                    for y in range(n_states):
                        if not inside[y]:
                            _add_pair(emission, read_log, read, perm, pair_index, scalar, row, block,
                                      request_labels[request, s1], y, request, column, best, top, out)
                    norm = float(outside)
                elif s2 < 4:
                    for y in range(n_states):
                        if not inside[y]:
                            _add_pair(emission, read_log, read, perm, pair_index, scalar, row, block, y,
                                      request_labels[request, s2], request, column, best, top, out)
                    norm = float(outside)
                else:
                    for x in range(n_states):
                        if inside[x]:
                            continue
                        for y in range(n_states):
                            if not inside[y]:
                                _add_pair(emission, read_log, read, perm, pair_index, scalar, row, block, x, y,
                                          request, column, best, top, out)
                    norm = float(outside) * float(outside)
                for i in range(C):
                    for j in range(C):
                        out[request, column, i, j] /= norm
        # misfit: every unordered pair, both homologue orientations
        for u in range(n_pairs):
            if scalar:
                out[request, FRAME_COLUMNS - 1, 0, 0] += math.exp(min(emission[row, u, block] - best, _CLIP))
                continue
            scale = 0.5 * math.exp(min(emission[row, u, block] - best, _CLIP) + read_log[row, u, block] - top)
            for i in range(C):
                for j in range(C):
                    # each float32 matrix entry enters a float64 product (exact), never a float32 sum
                    out[request, FRAME_COLUMNS - 1, i, j] += (scale * float(read[row, u, block, i, j])
                                                              + scale * float(read[row, u, block, perm[i], perm[j]]))
        for i in range(C):
            for j in range(C):
                out[request, FRAME_COLUMNS - 1, i, j] /= n_pairs


@kernel(overloads=((_f4, _i64_1, _i64_1, _i64_1, _f4),
                   (_f4, _i64_1, _ro_bins, _ro_bins, _f4)), cache=True, parallel=True, nogil=True)
def bin_operators(composed, rows, starts, stops, out):
    """out[n, B, C, C]: per requested batch row (``rows`` into composed's second axis) and bin, the label-free class
    operator of the bin, the product in site order of the composed operators of its sites (composed [L, N, C, C],
    painting.read_inputs.ReadEmissionBuilder.site_parts): the bin's read matrix with every read factor 1."""
    n = len(rows)
    n_bins = len(starts)
    C = composed.shape[2]
    for task in prange(n * n_bins):
        slot = task // n_bins
        block = task % n_bins
        row = rows[slot]
        work = np.empty((C, C))
        for i in range(C):
            for j in range(C):
                out[slot, block, i, j] = 1.0 if i == j else 0.0
        for site in range(starts[block], stops[block]):
            for i in range(C):
                for j in range(C):
                    value = 0.0
                    for k in range(C):
                        value += out[slot, block, i, k] * composed[site, row, k, j]
                    work[i, j] = value
            for i in range(C):
                for j in range(C):
                    out[slot, block, i, j] = work[i, j]


@kernel(overloads=((_f4, _f3, _i64_1, _i64_1, _b1, _f4, _f2),), cache=True, parallel=True, nogil=True)
def link_frames(operators, entering, slot, bins, first, em, entry):
    """Fold each frame's gap operator into its matrices and set its entry law, in place: em[X, K, C, C], entry[X, C].
    operators [n, B, C, C] the bins' label-free operators and entering [n, B + 1, C] the class law entering each bin
    (index B: after the last bin) of the frames' rows (``slot``); frames in sequence order (``first`` marks a
    sequence's first frame). A later frame at bin b after one at b' gets G = operators[b' + 1] ... operators[b - 1]
    (the identity for adjacent bins) and entry = entering[b' + 1]; a first frame entering[b]."""
    X = em.shape[0]
    K = em.shape[1]
    C = em.shape[2]
    for x in prange(X):
        s = slot[x]
        b = bins[x]
        if first[x]:
            for i in range(C):
                entry[x, i] = entering[s, b, i]
            continue
        previous = bins[x - 1]
        for i in range(C):
            entry[x, i] = entering[s, previous + 1, i]
        if b <= previous + 1:
            continue
        gap = np.empty((C, C))
        work = np.empty((C, C))
        for i in range(C):
            for j in range(C):
                gap[i, j] = operators[s, previous + 1, i, j]
        for block in range(previous + 2, b):
            for i in range(C):
                for j in range(C):
                    value = 0.0
                    for k in range(C):
                        value += gap[i, k] * operators[s, block, k, j]
                    work[i, j] = value
            for i in range(C):
                for j in range(C):
                    gap[i, j] = work[i, j]
        for column in range(K):
            for i in range(C):
                for j in range(C):
                    value = 0.0
                    for k in range(C):
                        value += gap[i, k] * em[x, column, k, j]
                    work[i, j] = value
            for i in range(C):
                for j in range(C):
                    em[x, column, i, j] = work[i, j]


def canonical_split(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(canonical labels [n, 4], column map [n, 26]) of frames' labels [n, 4] (parent 0's two, parent 1's two): each
    parent's pair sorted, the parent holding the smaller label first. A frame's 26 matrices are the canonical
    frame's columns relabelled, em[:, k] = em_canonical[:, map[:, k]] (symbol s -> its label's canonical position,
    'other' and misfit fixed): the ordered pair of a symbol pair reads the same labels on the same homologues."""
    labels = np.asarray(labels, dtype=np.int64)
    first = np.sort(labels[:, :2], axis=1)
    second = np.sort(labels[:, 2:], axis=1)
    swap = second[:, 0] < first[:, 0]
    canonical = np.where(swap[:, None], np.concatenate([second, first], axis=1),
                         np.concatenate([first, second], axis=1))
    position = np.argmax(canonical[:, None, :] == labels[:, :, None], axis=2)          # [n, 4]
    symbol = np.concatenate([position, np.full((len(labels), 1), 4)], axis=1)        # symbol 4 fixed
    columns = (5 * symbol[:, :, None] + symbol[:, None, :]).reshape(len(labels), 25)
    return canonical, np.concatenate([columns, np.full((len(labels), 1), FRAME_COLUMNS - 1)], axis=1)


def _pair_tables(n_states: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first, second = np.triu_indices(n_states)
    index = np.empty((n_states, n_states), dtype=np.int64)
    index[first, second] = np.arange(len(first))
    index[second, first] = np.arange(len(first))
    return first.astype(np.int64), second.astype(np.int64), index


_NO_LOG = np.zeros((1, 1, 1))
_NO_READ = np.zeros((1, 1, 1, 1, 1), dtype=np.float32)
_NO_PERM = np.zeros(1, dtype=np.int64)


def _scalar_matrices(emission: np.ndarray, rows: np.ndarray, bins: np.ndarray, labels: np.ndarray,
                     n_states: int) -> np.ndarray:
    first, second, index = _pair_tables(n_states)
    out = np.zeros((len(rows), FRAME_COLUMNS, 1, 1))
    frame_matrices(np.ascontiguousarray(emission, dtype=np.float64), _NO_LOG, _NO_READ, first, second, index,
                   _NO_PERM, True, np.asarray(rows, np.int64), np.asarray(bins, np.int64),
                   np.ascontiguousarray(labels, dtype=np.int64), out)
    return out


class ScalarFrames:
    """Frame emissions without read state (C = 1): each component's packed scalar emissions (ComponentInput.packed,
    log [R, U, B] of the relevant rows at the path strength)."""

    def __init__(self, components: Sequence[Any]):
        self.packed = [c.packed for c in components]
        self.n_states = [int(c.n_states) for c in components]

    def prepare(self, requests) -> int:
        return 1

    def emissions(self, index: int, rows, bins, labels, first, classes: int) -> tuple[np.ndarray, np.ndarray]:
        em = _scalar_matrices(self.packed[index], rows, bins, labels, self.n_states[index])
        return em, np.ones((len(rows), 1))

    def statistics(self) -> dict:
        return dict(classes=1, loads=0, requests=0, computed=0, rows_with_reads=0, rows_bypassed=0)


@dataclass(frozen=True)
class FrameSites:
    """What a component's frame emissions are rebuilt from: its contig (index in the run's order), the painted
    (selected) sites' positions, the labels' ALT probabilities [S - 1, L] and BACKGROUND's frequency [L], the bins'
    site ranges (starts, stops [B]) and the label count S (BACKGROUND included)."""

    contig: int
    positions: np.ndarray
    alt: np.ndarray
    background: np.ndarray
    starts: np.ndarray
    stops: np.ndarray
    n_states: int


@dataclass(frozen=True)
class ContigReads:
    """One contig's read-state inputs (the stage's loader): ``evidence`` (likelihoods, observed, cohort_frequencies,
    positions of every sample), ``builder`` the painting's read-state emission builder over the pedigree's scalar
    emissions (painting.read_inputs.ReadEmissionBuilder), the painting's parameters and outlier rate, and the path
    strength's site correlation."""

    evidence: Any
    builder: Callable
    parameters: Any
    robustness_epsilon: float
    site_correlation: float


class ReadFrames:
    """Frame emissions with the read state, rebuilt on request (module docstring).

    ``sites`` per component (FrameSites); ``samples`` int64 [R] the sample index of each relevant row; ``load(contig)``
    the contig's ContigReads (evidence and builder are read again for every load and dropped after it). ``prepare``
    computes every request not yet cached, one work item per contig with misses (the requested rows in builder
    batches), then files the items' frames in the cache; ``emissions`` assembles a component's frames from the cache.
    ``concurrent(functions)``: the caller's concurrent map of the contigs' items (the engine's CPU-seated children;
    None: one after another): an item's values do not depend on the others, on its batches or on the order, and the
    items' batches together stay within ``batch_bytes`` (each item gets an equal share when they run concurrently)."""

    def __init__(self, sites: Sequence[FrameSites], samples: np.ndarray, load: Callable[[int], ContigReads], *,
                 batch_bytes: int = READ_STATE_BATCH_BYTES,
                 concurrent: Callable[[list[Callable[[], Any]]], list[Any]] | None = None):
        self.sites = list(sites)
        self.samples = np.asarray(samples, dtype=np.int64)
        self.load = load
        self.batch_bytes = int(batch_bytes)
        self.concurrent = concurrent
        self._index: dict[tuple, tuple[int, int]] = {}      # (component, row, bin, canonical labels) -> block, offset
        self._blocks: list[np.ndarray] = []                   # [n, 26, C_b, C_b]
        self._links: dict[tuple[int, int], tuple] = {}        # (component, row) -> (operators | None, entering)
        self._classes = 1
        self._statistics = dict(loads=0, requests=0, computed=0)

    def statistics(self) -> dict:
        bypassed = sum(1 for v in self._links.values() if v[0] is None)
        return dict(self._statistics, classes=self._classes, rows_with_reads=len(self._links) - bypassed,
                    rows_bypassed=bypassed)

    def class_count(self, index, rows):
        """Width of these rows alone, so a retained family can be embedded in the next guard pass's width."""
        return max((self._links[index, int(r)][1].shape[1] for r in np.unique(rows)), default=1)

    def _keys(self, index, rows, bins, canonical):
        return list(zip([index] * len(rows), rows.tolist(), bins.tolist(), *canonical.T.tolist()))

    def prepare(self, requests) -> int:
        """Compute the uncached frames of ``requests`` [(component, rows, bins, labels)]; returns the class count of
        these frames (C when any of their rows carries the read state, else 1)."""
        # a component's requests (one per family) are joined: every miss and every row lacking its links, once
        missing: dict[int, dict[int, list]] = {}
        used: set[tuple[int, int]] = set()
        for index, rows, bins, labels in requests:
            rows = np.asarray(rows, dtype=np.int64)
            bins = np.asarray(bins, dtype=np.int64)
            canonical, _ = canonical_split(labels)
            keys = self._keys(index, rows, bins, canonical)
            self._statistics["requests"] += len(keys)
            used.update((index, int(r)) for r in np.unique(rows))
            need = np.fromiter((k not in self._index for k in keys), dtype=bool, count=len(keys))
            need_links = {int(r) for r in np.unique(rows) if (index, int(r)) not in self._links}
            if need.any() or need_links:
                parts = missing.setdefault(self.sites[index].contig, {}).setdefault(index, [[], [], [], set(), set()])
                for k in np.flatnonzero(need):
                    if keys[k] not in parts[4]:
                        parts[4].add(keys[k])
                        parts[0].append(rows[k])
                        parts[1].append(bins[k])
                        parts[2].append(canonical[k])
                parts[3].update(need_links)
        contigs = sorted(missing)
        share = self.batch_bytes // max(1, len(contigs)) if self.concurrent is not None else self.batch_bytes

        def item(contig):
            def run():
                data = self.load(contig)
                out = [self._compute(index, data, np.asarray(rows, dtype=np.int64).reshape(-1),
                                     np.asarray(bins, dtype=np.int64).reshape(-1),
                                     np.asarray(canonical, dtype=np.int64).reshape(-1, 4), sorted(link_rows), share)
                       for index, (rows, bins, canonical, link_rows, _) in missing[contig].items()]
                del data
                return out
            return run

        work = [item(contig) for contig in contigs]
        results = [run() for run in work] if self.concurrent is None or len(work) <= 1 else self.concurrent(work)
        self._statistics["loads"] += len(contigs)
        for parts in results:
            for blocks, links in parts:
                for keys, values in blocks:
                    block = len(self._blocks)
                    self._blocks.append(values)
                    for offset, key in enumerate(keys):
                        self._index[key] = (block, offset)
                    self._statistics["computed"] += len(keys)
                self._links.update(links)
        classes = 1
        for key in used:
            operators = self._links[key][0]
            if operators is not None:
                classes = max(classes, operators.shape[-1])
        self._classes = max(self._classes, classes)
        return classes

    def _compute(self, index, data: ContigReads, rows, bins, canonical, link_rows, batch_bytes):
        """One component's missing frames and links: ([(keys, matrices [n, 26, C_b, C_b])], {(component, row):
        (operators | None, entering)}); reads no shared state but the source's fixed inputs."""
        blocks: list[tuple[list, np.ndarray]] = []
        links: dict[tuple[int, int], tuple] = {}
        site = self.sites[index]
        evidence, builder = data.evidence, data.builder
        positions = np.asarray(evidence.positions)
        columns = np.searchsorted(positions, site.positions)
        if np.any(columns >= len(positions)) or np.any(positions[np.minimum(columns, len(positions) - 1)] !=
                                                        site.positions):
            raise InputError("painted sites are missing from the evidence record")
        columns = columns.astype(np.int64)
        S, n_bins = int(site.n_states), len(site.starts)
        first, second, pair_index = _pair_tables(S)
        upper = first * S + second
        wanted = np.unique(np.concatenate([rows, np.asarray(link_rows, dtype=np.int64)]))
        samples = self.samples[wanted]
        # one batch's working set: the builder's (read matrices, per-site parts, transient inputs) plus this module's
        # frame matrices and bin operators of the batch's rows
        frames_per_row = len(rows) / max(1, len(wanted))
        classes = 2 * int(builder.minor) + 3
        own = 8 * classes * classes * (FRAME_COLUMNS * frames_per_row + n_bins) + 8 * S * S * n_bins
        per_sample = int(builder.sample_bytes(columns, S, n_bins, False)) + int(own)
        step = max(1, min(max(16, numba.get_num_threads()), batch_bytes // max(1, per_sample)))
        stops = np.asarray(site.stops, dtype=np.int64)
        starts = np.asarray(site.starts, dtype=np.int64)
        link_set = set(int(r) for r in link_rows)
        k = 0
        while k < len(wanted):
            lo = int(samples[k])
            z = k
            while z < len(wanted) and int(samples[z]) - lo < step:
                z += 1
            batch = slice(lo, int(samples[z - 1]) + 1)
            local = wanted[k:z]
            k = z
            bypassed, composed, laws, _ = builder.site_parts(columns, batch)
            batch_row = {int(r): int(s) - lo for r, s in zip(local, self.samples[local])}
            in_batch = np.flatnonzero(np.isin(rows, local))
            if len(in_batch):
                selected_bins = np.unique(bins[in_batch])
                built, _flags, _informative = builder.build_at(
                    evidence.likelihoods, evidence.observed, columns, batch, site.alt, site.background, starts, stops,
                    data.parameters, bins=selected_bins, robustness_epsilon=data.robustness_epsilon, site_correlation=data.site_correlation,
                )
                emission = np.ascontiguousarray(np.asarray(built.raw)[:, upper, :], dtype=np.float64)
                read = np.require(built.read, dtype=np.float32, requirements=["C", "W", "A"])
                read_log = np.require(built.read_log, dtype=np.float64, requirements=["C", "W", "A"])
                perm = painting_hmm.swap_classes(int(built.minor)).astype(np.int64)
                C = read.shape[3]
                row_bypassed = np.asarray([bool(bypassed[batch_row[int(r)]]) for r in rows[in_batch]], dtype=bool)
                for deep in (False, True):
                    pick = in_batch[row_bypassed == deep]
                    if not len(pick):
                        continue
                    request_row = np.asarray([batch_row[int(r)] for r in rows[pick]], dtype=np.int64)
                    width = 1 if deep else C
                    out = np.zeros((len(pick), FRAME_COLUMNS, width, width))
                    frame_matrices(emission, read_log, read, first, second, pair_index, perm, bool(deep), request_row,
                                   np.searchsorted(selected_bins, bins[pick]), np.ascontiguousarray(canonical[pick]), out)
                    blocks.append((self._keys(index, rows[pick], bins[pick], canonical[pick]), out))
                del built, emission, read, read_log
            C = composed.shape[2]
            link_local = [int(r) for r in local if int(r) in link_set]
            shared = [r for r in link_local if not bypassed[batch_row[r]]]
            for r in link_local:
                if bypassed[batch_row[r]]:
                    links[(index, r)] = (None, np.ones((n_bins + 1, 1)))
            if shared:
                batch_rows = np.asarray([batch_row[r] for r in shared], dtype=np.int64)
                operators = np.empty((len(shared), n_bins, C, C))
                bin_operators(np.require(composed, dtype=np.float64, requirements=["C", "W", "A"]), batch_rows,
                              starts, stops, operators)
                enter_site = np.concatenate([np.maximum(starts - 1, 0), stops[-1:] - 1])
                laws = np.asarray(laws)
                for slot, r in enumerate(shared):
                    entering = np.ascontiguousarray(laws[enter_site, batch_row[r]], dtype=np.float64)
                    links[(index, r)] = (operators[slot], entering)
            del composed, laws
        return blocks, links

    def emissions(self, index: int, rows, bins, labels, first, classes: int) -> tuple[np.ndarray, np.ndarray]:
        """(em [n, 26, classes, classes], entry [n, classes]) of one component's frames in sequence order."""
        rows = np.asarray(rows, dtype=np.int64)
        bins = np.asarray(bins, dtype=np.int64)
        canonical, column_map = canonical_split(labels)
        located = np.asarray([self._index[k] for k in self._keys(index, rows, bins, canonical)], dtype=np.int64)
        em = np.zeros((len(rows), FRAME_COLUMNS, classes, classes))
        for block in np.unique(located[:, 0]):
            pick = np.flatnonzero(located[:, 0] == block)
            values = self._blocks[block][located[pick, 1]]
            if values.shape[-1] == classes:
                em[pick] = values
            elif values.shape[-1] == 1:
                em[pick, :, 0, 0] = values[:, :, 0, 0]
            else:
                raise InputError("frame emissions with different read-class counts")
        em = np.ascontiguousarray(np.take_along_axis(em, column_map[:, :, None, None], axis=1))
        if classes == 1:
            return em, np.ones((len(rows), 1))
        unique, slot = np.unique(rows, return_inverse=True)
        n_bins = len(self.sites[index].starts)
        operators = np.empty((len(unique), n_bins, classes, classes))
        entering = np.zeros((len(unique), n_bins + 1, classes))
        for k, r in enumerate(unique):
            ops, enter = self._links[(index, int(r))]
            if ops is None:
                # a bypassed row: the identity read state in class 0
                operators[k] = np.eye(classes)
                entering[k, :, 0] = 1.0
            else:
                operators[k] = ops
                entering[k] = enter
        entry = np.empty((len(rows), classes))
        link_frames(operators, entering, slot.astype(np.int64), bins, np.asarray(first, dtype=np.bool_), em, entry)
        return em, entry
