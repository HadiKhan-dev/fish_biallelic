"""Release rule and the painting product.

Per sample, bin and track the decoder gives a Viterbi state (gauge-fixed track
order) and two posterior masses: qV, the mass of the Viterbi state's unordered
public class pair, and qA, the mass of its unordered pair of per-bin local
allele-equivalence keys (qA >= qV). Release, with tau = minimum_release_posterior:

1. qV >= tau: both tracks are released as their classes: a singleton anchored
   class names its row (SINGLETON_NAMED), a pooled class its rows
   (POOLED_EQUIVALENCE), an unanchored class or BACKGROUND names none
   (UNANCHORED_TRAJECTORY, BACKGROUND).
2. else qA >= tau and every track whose state belongs to a multi-class local
   key in the bin is directly read-supported there: such tracks are released
   as LOCAL_ALLELE_EQUIVALENCE (class = the local key, rows = all rows of the
   classes with that key); the other track keeps its stage-1 release.
3. else both tracks abstain (LOW_POSTERIOR_ABSTENTION).

Switch intervals (release 2.2). Where a track's switch between classes a and b
cannot be placed by the sample's data (a run of bins in which a and b have the
same calls at every site the sample observes; painting.hmm.switch_interval_masks),
the track's candidates in those bins are {a, b}. The gate is unchanged from 2.1
(qV / qA of the Viterbi pair, taken as the smaller of the two traceback
conventions' pairs, so no bin 2.1 rejects is released). A released interval track has status
SWITCH_INTERVAL, no label or class, and the union of the candidates' founder rows
as members (stage 2: of every class sharing a candidate's local key); () when a
candidate is BACKGROUND or unanchored. The other track is released as usual.

Samples without any observed informative site abstain everywhere
(INELIGIBLE_NO_EVIDENCE). The rule runs once, when the product is made.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import NamedTuple

import numpy as np
from numba import njit, prange

from ..model.components import ComponentPanel
from .config import PaintingConfig
from .states import PaintingBins, PaintingStates, multi_member_local_states
from .transitions import PaintingTransitions


class TrackStatus(IntEnum):
    """Scientific meaning of one released track in one bin (stored as uint8)."""

    INELIGIBLE_NO_EVIDENCE = 0
    LOW_POSTERIOR_ABSTENTION = 1
    SINGLETON_NAMED = 2
    POOLED_EQUIVALENCE = 3
    BACKGROUND = 4
    UNANCHORED_TRAJECTORY = 5
    # Founder label ambiguous among rows identical over the bin's selected
    # sites; the alleles there are known and the sample's reads observe them.
    LOCAL_ALLELE_EQUIVALENCE = 6
    # The track switches between candidate classes somewhere in a run of bins
    # where the candidates have the same calls at every site the sample observes
    # (release 2.2): the members are the candidates' rows; neither is named.
    SWITCH_INTERVAL = 7


class PaintedChunk(NamedTuple):
    """A run of bins with one release on both tracks; [start, end) in bp.

    labels: per track the released founder row of a SINGLETON_NAMED track,
        else -1 (component-local row index).
    members: per track the candidate founder rows (ascending): the row of a
        named track, the pooled rows of POOLED_EQUIVALENCE and
        LOCAL_ALLELE_EQUIVALENCE tracks, the candidates' rows of a
        SWITCH_INTERVAL track, () otherwise.
    """

    start: int
    end: int
    labels: tuple[int, int]
    members: tuple[tuple[int, ...], tuple[int, ...]]


class SwitchInterval(NamedTuple):
    """One released switch interval of a sample's track (release 2.2); [start, end) in bp.

    track: 0 or 1; start, end: the bp span of the interval's bins (the switch lies inside);
    rows: the candidate founder rows released there (() if a candidate is not a panel row);
    before, after: the released member rows of the track in the bins just before and after
    (() at a component end or where the track abstains).
    """

    track: int
    start: int
    end: int
    rows: tuple[int, ...]
    before: tuple[int, ...]
    after: tuple[int, ...]


@dataclass(frozen=True)
class ComponentPainting:
    """The released painting of one component (N samples, B bins, S states).

    panel: the founder panel painted (assembly's component after label
        continuity); rows of every grid below are its component-local rows.
        None when the product was loaded without panels (records.load(panels=False)).
    states, bins, transitions: the HMM's state alphabet, bins and transition
        table (pedigree rebuilds its source model from exactly these).
    local_keys: int32[B, S] per-bin allele-equivalence keys of the states.
    eligible: bool[N] the sample has an observed informative GL row at a
        selected site of the component (else it abstains everywhere).
    labels: int32[N, 2, B] released founder row of a SINGLETON_NAMED track, else -1.
    classes: int32[N, 2, B] released class (stage 1) or local key
        (LOCAL_ALLELE_EQUIVALENCE), -1 when abstaining or SWITCH_INTERVAL.
    status: uint8[N, 2, B] :class:`TrackStatus`.
    direct: bool[N, 2, B] the released class is called and read-observed at an
        informative selected site of the bin (direct callability).
    member_ids: int32[N, 2, B] indices into ``member_sets`` (0 = empty set).
    member_sets: interned ascending tuples of candidate founder rows; [0] = ().
    switch_counts: int64[N] minimum unordered track changes of the Viterbi
        path (a pure track swap is gauge, not a switch); 0 if ineligible.
    packed_emissions: float64[N, S (S + 1) / 2, B] upper-triangle binned log
        emissions (the ordered-pair emissions are symmetric), kept for
        pedigree preparation when they fit the chromosome's cache budget,
        else None.
    """

    panel: ComponentPanel | None
    states: PaintingStates
    bins: PaintingBins
    transitions: PaintingTransitions
    local_keys: np.ndarray
    eligible: np.ndarray
    labels: np.ndarray
    classes: np.ndarray
    status: np.ndarray
    direct: np.ndarray
    member_ids: np.ndarray
    member_sets: tuple[tuple[int, ...], ...]
    switch_counts: np.ndarray
    packed_emissions: np.ndarray | None

    @property
    def interval(self) -> tuple[int, int]:
        """First and last bp position of the component panel (needs the panel)."""
        positions = self.panel.positions
        return int(positions[0]), int(positions[-1])

    def chunks(self, sample: int) -> list[PaintedChunk]:
        """Released chunks of one sample: consecutive bins with equal (label, class,
        status, members) on both tracks coalesce; ineligible samples have none."""
        if not self.eligible[sample] or self.bins.n_bins == 0:
            return []
        labels = self.labels[sample]
        classes = self.classes[sample]
        status = self.status[sample]
        members = self.member_ids[sample]
        n_bins = labels.shape[1]
        changed = np.zeros(n_bins, dtype=np.bool_)
        changed[0] = True
        for grid in (labels, classes, status, members):
            changed[1:] |= (grid[0, 1:] != grid[0, :-1]) | (grid[1, 1:] != grid[1, :-1])
        starts = np.flatnonzero(changed)
        stops = np.r_[starts[1:], n_bins]
        edges = self.bins.edges
        return [PaintedChunk(
            int(edges[start]), int(edges[stop]),
            (int(labels[0, start]), int(labels[1, start])),
            (self.member_sets[members[0, start]], self.member_sets[members[1, start]]))
            for start, stop in zip(starts, stops)]

    def switch_intervals(self, sample: int) -> list[SwitchInterval]:
        """The released switch intervals of one sample (runs of SWITCH_INTERVAL bins per track)."""
        if not self.eligible[sample] or self.bins.n_bins == 0:
            return []
        edges = self.bins.edges
        n_bins = self.bins.n_bins
        result = []
        for track in (0, 1):
            flagged = self.status[sample, track] == int(TrackStatus.SWITCH_INTERVAL)
            if not np.any(flagged):
                continue
            ids = self.member_ids[sample, track]
            changed = np.ones(n_bins, dtype=np.bool_)
            changed[1:] = (flagged[1:] != flagged[:-1]) | (ids[1:] != ids[:-1])
            starts = np.flatnonzero(changed)
            stops = np.r_[starts[1:], n_bins]
            for start, stop in zip(starts, stops):
                if not flagged[start]:
                    continue
                before = self.member_sets[ids[start - 1]] if start > 0 else ()
                after = self.member_sets[ids[stop]] if stop < n_bins else ()
                result.append(SwitchInterval(track, int(edges[start]), int(edges[stop]),
                                             self.member_sets[ids[start]], before, after))
        return result

    def carrier_observed(self, positions: np.ndarray, columns: np.ndarray, observed: np.ndarray,
                         n_rows: int) -> np.ndarray:
        """bool[K, L]: founder row k at bp ``positions[j]`` is observed by a painted carrier (release 2.2): some
        eligible sample whose released track at that site names k in its member set (named, pooled, local
        equivalence or switch interval) has read depth there (``observed`` bool [N, L_e] at evidence columns
        ``columns`` int64 [L]). False outside the painting bins."""
        positions = np.asarray(positions, dtype=np.int64)
        if self.bins.n_bins == 0:
            return np.zeros((n_rows, len(positions)), dtype=np.bool_)
        bins = np.searchsorted(self.bins.edges, positions, side="right") - 1
        bins[(bins < 0) | (bins >= self.bins.n_bins)] = -1
        offsets = np.zeros(len(self.member_sets) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(rows) for rows in self.member_sets])
        rows = np.asarray([row for members in self.member_sets for row in members], dtype=np.int64)
        return _carrier_observed_kernel(np.ascontiguousarray(self.member_ids, dtype=np.int32), offsets, rows,
                                        np.ascontiguousarray(self.eligible, dtype=np.bool_), bins.astype(np.int64),
                                        np.asarray(columns, dtype=np.int64),
                                        np.ascontiguousarray(observed, dtype=np.bool_), int(n_rows))

    def site_labels(self, positions: np.ndarray) -> np.ndarray:
        """int16[N, 2, L] released labels at bp ``positions`` (sorted); -1 outside the bins."""
        positions = np.asarray(positions, dtype=np.int64)
        n_samples = len(self.eligible)
        result = np.full((n_samples, 2, len(positions)), -1, dtype=np.int16)
        if self.bins.n_bins == 0:
            return result
        bins = np.searchsorted(self.bins.edges, positions, side="right") - 1
        inside = (bins >= 0) & (bins < self.bins.n_bins)
        result[:, :, inside] = self.labels[:, :, bins[inside]]
        return result


@dataclass(frozen=True)
class RelabelMove:
    """One accepted row-suffix relabelling of label continuity.

    atomic_span: index of the atomic span from which rows exchange labels;
    position: bp position of its first site; rows: the two founder rows;
    score_gain_switches: summed Viterbi score gain in units of one track change;
    painted_switch_change: change of the painted switch total (negative);
    winners, losers: samples whose painted switches fall / rise.
    """

    atomic_span: int
    position: int
    rows: tuple[int, int]
    score_gain_switches: float
    painted_switch_change: int
    winners: tuple[int, ...]
    losers: tuple[int, ...]


@dataclass(frozen=True)
class LabelContinuity:
    """Outcome of label continuity for one component.

    accepted: the painted panel is the relabelled one.
    reason: 'relabelled', 'no_link_or_movable_row', 'no_evidence', 'no_move',
        'trajectory_classes_changed' or 'painted_switches_not_lowered'.
    link_points: atomic-span boundaries inside the selected sites;
    movable_rows: singleton anchored classes (only these exchange labels);
    switch_unit_nats: median cost of one track change over the bin edges;
    switches_before / switches_after: painted switch totals of the eligible
        samples before and after (after = before unless accepted);
    candidate_checks: candidates re-decoded during the search;
    moves: accepted moves in search order (also when the final check rejects them);
    row_maps: int64[A, K] applied relabelling (span a, new row k carries old
        row row_maps[a, k]), None unless accepted.
    tie_events: candidate rankings decided inside a tie class (``maths.ties.TieCounter``
        record; release 2.1).
    """

    accepted: bool
    reason: str
    link_points: int
    movable_rows: int
    switch_unit_nats: float | None
    switches_before: int | None
    switches_after: int | None
    candidate_checks: int
    moves: tuple[RelabelMove, ...]
    row_maps: np.ndarray | None
    tie_events: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ChromosomePainting:
    """Painting product of one chromosome (record ``painting``).

    contig, sample_ids: chromosome and the ordered samples of every grid (all
        samples of the evidence, including those the sample QC keeps out of
        founder construction).
    components: one :class:`ComponentPainting` per assembled component, in
        chromosome order; component HMMs are independent (reset at breaks).
    label_continuity: one :class:`LabelContinuity` per component.
    config: the painting settings used (refit and pedigree reuse the emission model).
    Provenance (the evidence and assembly records painted) is bound in the
    store identity ``identity/painting`` (painting.records.require_evidence_binding).
    """

    contig: str
    sample_ids: tuple[str, ...]
    components: tuple[ComponentPainting, ...]
    label_continuity: tuple[LabelContinuity, ...]
    config: PaintingConfig


def interval_tracks(paths: np.ndarray, masks: np.ndarray | None) -> np.ndarray:
    """bool[N, 2, B]: the track has candidates besides its Viterbi state (a switch interval)."""
    if masks is None:
        return np.zeros(paths.shape, dtype=np.bool_)
    return masks != np.left_shift(np.int64(1), paths.astype(np.int64))


def release_tracks(states: PaintingStates, paths: np.ndarray, eligible: np.ndarray,
                   qv: np.ndarray, qa: np.ndarray, local_keys: np.ndarray,
                   support: np.ndarray, threshold: float, masks: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Staged release of the Viterbi tracks: (labels, classes, status), each [N, 2, B].

    ``paths`` int32[N, 2, B] gauge-fixed Viterbi states; ``qv``/``qa``
    float64[N, B] (masses over the candidate sets, painting.hmm.release_posteriors);
    ``support`` bool[N, 2, B] direct read support of each Viterbi track;
    ``threshold`` = tau; ``masks`` int64[N, 2, B] candidate states (None = the
    Viterbi state alone): an interval track of a released bin gets SWITCH_INTERVAL.
    The gate itself is 2.1's (stages 1 and 2 on the Viterbi tracks), so a switch
    interval never releases a bin that 2.1 rejects.
    """
    anchored = states.anchored
    class_labels = np.asarray([
        members[0] if len(members) == 1 and anchored[index] else -1
        for index, members in enumerate(states.classes)] + [-1], dtype=np.int32)
    classes = paths.astype(np.int32, copy=True)
    labels = class_labels[paths]
    status_by_class = np.full(states.n_states, int(TrackStatus.BACKGROUND), dtype=np.uint8)
    for index, members in enumerate(states.classes):
        if not anchored[index]:
            status_by_class[index] = int(TrackStatus.UNANCHORED_TRAJECTORY)
        elif len(members) > 1:
            status_by_class[index] = int(TrackStatus.POOLED_EQUIVALENCE)
        else:
            status_by_class[index] = int(TrackStatus.SINGLETON_NAMED)
    status = status_by_class[paths]
    stage_one = qv >= threshold
    bins = np.arange(paths.shape[2])[None, None, :]
    interval = interval_tracks(paths, masks)
    multi = multi_member_local_states(local_keys)[bins, paths]
    # Stage 2 exactly as 2.1 (the Viterbi tracks' direct support); an interval track is then released as
    # SWITCH_INTERVAL instead of LOCAL_ALLELE_EQUIVALENCE.
    stage_two = (~stage_one) & (qa >= threshold) & ~np.any(multi & ~support, axis=1)
    local = stage_two[:, None, :] & multi & ~interval
    classes[local] = local_keys[bins, paths][local]
    labels[local] = -1
    status[local] = int(TrackStatus.LOCAL_ALLELE_EQUIVALENCE)
    classes[interval] = -1
    labels[interval] = -1
    status[interval] = int(TrackStatus.SWITCH_INTERVAL)
    low = np.broadcast_to(~(stage_one | stage_two)[:, None, :], paths.shape)
    classes[low] = -1
    labels[low] = -1
    status[low] = int(TrackStatus.LOW_POSTERIOR_ABSTENTION)
    classes[~eligible] = -1
    labels[~eligible] = -1
    status[~eligible] = int(TrackStatus.INELIGIBLE_NO_EVIDENCE)
    return labels, classes, status


def track_members(status: np.ndarray, classes: np.ndarray, local_keys: np.ndarray,
                  state_classes: tuple[tuple[int, ...], ...], masks: np.ndarray | None = None,
                  stage_two: np.ndarray | None = None, anchored: np.ndarray | None = None
                  ) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Candidate founder rows of every released track: (member_ids int32[N, 2, B], member_sets).

    SINGLETON_NAMED and POOLED_EQUIVALENCE name the rows of their class;
    LOCAL_ALLELE_EQUIVALENCE names the rows of every class whose key in that
    bin is the released key; SWITCH_INTERVAL names the rows of its candidate
    classes (``masks``; in a stage-2 bin, ``stage_two`` bool[N, B], of every class
    sharing a candidate's local key), none if a candidate is BACKGROUND or not
    ``anchored``; other statuses name none (id 0, the empty set).
    """
    n_classes = len(state_classes)
    sets: list[tuple[int, ...]] = [()]
    index: dict[tuple[int, ...], int] = {(): 0}

    def intern(rows) -> int:
        rows = tuple(sorted(int(row) for row in rows))
        found = index.get(rows)
        if found is None:
            found = index[rows] = len(sets)
            sets.append(rows)
        return found

    ids = np.zeros(status.shape, dtype=np.int32)
    whole = np.asarray([intern(members) for members in state_classes] + [0], dtype=np.int32)
    named = ((status == int(TrackStatus.SINGLETON_NAMED))
             | (status == int(TrackStatus.POOLED_EQUIVALENCE)))
    ids[named] = whole[classes[named]]
    local = status == int(TrackStatus.LOCAL_ALLELE_EQUIVALENCE)
    if np.any(local):
        # One member set per distinct (bin, key): O(N B + B C) overall.
        bins = np.broadcast_to(np.arange(status.shape[2]), status.shape)[local]
        codes, inverse = np.unique(
            bins.astype(np.int64) * (n_classes + 1) + classes[local], return_inverse=True)
        code_ids = np.empty(len(codes), dtype=np.int32)
        for position, code in enumerate(codes):
            bin_index, key = divmod(int(code), n_classes + 1)
            code_ids[position] = intern(
                row for class_index in np.flatnonzero(local_keys[bin_index, :n_classes] == key)
                for row in state_classes[class_index])
        ids[local] = code_ids[inverse.reshape(-1)]
    interval = status == int(TrackStatus.SWITCH_INTERVAL)
    if np.any(interval):
        if masks is None:
            raise ValueError("switch-interval tracks need their candidate masks")
        anchored = (np.ones(n_classes, dtype=np.bool_) if anchored is None
                    else np.asarray(anchored, dtype=np.bool_))
        bins = np.broadcast_to(np.arange(status.shape[2]), status.shape)[interval]
        local_stage = (np.zeros(status.shape, dtype=np.bool_) if stage_two is None
                       else np.broadcast_to(np.asarray(stage_two)[:, None, :], status.shape))[interval]
        # One member set per distinct (bin if stage 2 else -1, mask).
        keyed = np.stack([np.where(local_stage, bins, -1).astype(np.int64),
                          np.asarray(masks)[interval].astype(np.int64)], axis=1)
        codes, inverse = np.unique(keyed, axis=0, return_inverse=True)
        code_ids = np.empty(len(codes), dtype=np.int32)
        for position, (bin_index, mask) in enumerate(codes.tolist()):
            candidates = [state for state in range(n_classes + 1) if (mask >> state) & 1]
            if any(state >= n_classes or not anchored[state] for state in candidates):
                code_ids[position] = 0
                continue
            if bin_index >= 0:
                keys = {int(local_keys[bin_index, state]) for state in candidates}
                candidates = [index for index in range(n_classes) if int(local_keys[bin_index, index]) in keys]
            code_ids[position] = intern(row for class_index in candidates for row in state_classes[class_index])
        ids[interval] = code_ids[inverse.reshape(-1)]
    return ids, tuple(sets)


def segment_rows(painting: "ChromosomePainting", input_ids=None) -> list[dict]:
    """The released painting as segments (release 2.2): one row per sample, component, track and run of bins with
    one release (status, label and candidate rows), in sample, component, track and position order.

    Columns: SampleID, Component, Track (0/1, the painting's gauge: which homologue is track 0 carries no meaning),
    Start, End (bp, [Start, End)), Status (TrackStatus name), Label (the founder row of a SINGLETON_NAMED segment,
    else empty), Rows (candidate founder rows, ';'-joined; component-local). A SWITCH_INTERVAL segment is the bp
    interval where the track switches between its candidate rows; the segments before and after it name the two
    sides. Ineligible samples have no rows. ``input_ids`` (the run's input sample order) orders the samples; default
    the records' order."""
    names = {int(status.value): status.name for status in TrackStatus}
    order = list(range(len(painting.sample_ids)))
    if input_ids is not None:
        position = {str(name): index for index, name in enumerate(input_ids)}
        order.sort(key=lambda sample: position.get(str(painting.sample_ids[sample]), len(position)))
    rows = []
    for sample in order:
        sample_id = painting.sample_ids[sample]
        for index, component in enumerate(painting.components):
            if not component.eligible[sample] or component.bins.n_bins == 0:
                continue
            edges = component.bins.edges
            n_bins = component.bins.n_bins
            for track in (0, 1):
                status = component.status[sample, track]
                labels = component.labels[sample, track]
                ids = component.member_ids[sample, track]
                changed = np.ones(n_bins, dtype=np.bool_)
                changed[1:] = (status[1:] != status[:-1]) | (labels[1:] != labels[:-1]) | (ids[1:] != ids[:-1])
                starts = np.flatnonzero(changed)
                stops = np.r_[starts[1:], n_bins]
                for start, stop in zip(starts.tolist(), stops.tolist()):
                    label = int(labels[start])
                    rows.append(dict(SampleID=str(sample_id), Component=index, Track=track,
                                     Start=int(edges[start]), End=int(edges[stop]),
                                     Status=names[int(status[start])], Label="" if label < 0 else str(label),
                                     Rows=";".join(map(str, component.member_sets[ids[start]]))))
    return rows


@njit(cache=True, parallel=True)
def _carrier_observed_kernel(member_ids, set_offsets, set_rows, eligible, site_bins, site_columns, observed,
                             n_rows):
    n_samples = member_ids.shape[0]
    n_sites = len(site_bins)
    result = np.zeros((n_rows, n_sites), dtype=np.bool_)
    for site in prange(n_sites):
        bin_index = site_bins[site]
        if bin_index < 0:
            continue
        column = site_columns[site]
        for sample in range(n_samples):
            if not eligible[sample] or not observed[sample, column]:
                continue
            for track in range(2):
                identifier = member_ids[sample, track, bin_index]
                for member in range(set_offsets[identifier], set_offsets[identifier + 1]):
                    row = set_rows[member]
                    if row < n_rows:
                        result[row, site] = True
    return result
