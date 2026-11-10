"""Release rule and the painting product.

Per sample, bin and track the decoder gives a Viterbi state (gauge-fixed track
order) and the gate's two posterior masses: the mass of the Viterbi state's unordered public
class pair, and that of its unordered pair of per-bin local allele-equivalence
keys (the second >= the first). Release, with
tau = minimum_release_posterior:

1. qF >= tau: both tracks are released as their classes: a singleton anchored
   class names its row (SINGLETON_NAMED), a pooled class its rows
   (POOLED_EQUIVALENCE), an unanchored class or BACKGROUND names none
   (UNANCHORED_TRAJECTORY, BACKGROUND).
2. else qAF >= tau and every track whose state belongs to a multi-class local
   key in the bin is directly read-supported there: such tracks are released
   as LOCAL_ALLELE_EQUIVALENCE (class = the local key, rows = all rows of the
   classes with that key); the other track keeps its stage-1 release.
3. else both tracks abstain (LOW_POSTERIOR_ABSTENTION).

Switch intervals. When a sample's data cannot place a switch between classes a and b (a run of bins with
identical calls at every observed site; painting.hmm.switch_interval_masks), the track's candidates there are
{a, b}. The release gate takes the smaller qF / qAF mass of the two traceback conventions' Viterbi pairs. A
released interval track has status SWITCH_INTERVAL, no label or class, and the union of candidate founder rows
as members (stage 2 includes every class sharing a candidate's local key); members are empty when a candidate
is BACKGROUND or unanchored. The other track is released as usual.

Samples without any observed informative site abstain everywhere
(INELIGIBLE_NO_EVIDENCE). The rule runs once, when the product is made.
"""

from __future__ import annotations

import numpy as np

from ...contracts.painting import TrackStatus
from .states import PaintingStates, multi_member_local_states


def interval_tracks(paths: np.ndarray, masks: np.ndarray | None) -> np.ndarray:
    """bool[N, 2, B]: the track has candidates besides its Viterbi state (a switch interval)."""
    if masks is None:
        return np.zeros(paths.shape, dtype=np.bool_)
    return masks != np.left_shift(np.int64(1), paths.astype(np.int64))


def release_tracks(
    states: PaintingStates,
    paths: np.ndarray,
    eligible: np.ndarray,
    qf: np.ndarray,
    qaf: np.ndarray,
    local_keys: np.ndarray,
    support: np.ndarray,
    threshold: float,
    masks: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Staged release of the Viterbi tracks: (labels, classes, status), each [N, 2, B].

    ``paths`` int32[N, 2, B] gauge-fixed Viterbi states; ``qf``/``qaf`` float64[N, B] faithful-copy masses
    (painting.fidelity.release_posteriors); ``support`` bool[N, 2, B] direct read support; ``threshold`` =
    tau; ``masks`` int64[N, 2, B] candidates (None = the Viterbi state alone). The posterior and
    direct-support gates apply to Viterbi tracks; an interval track of a released bin gets SWITCH_INTERVAL.
    """
    anchored = states.anchored
    class_labels = np.asarray(
        [members[0] if len(members) == 1 and anchored[index] else -1 for index, members in enumerate(states.classes)]
        + [-1],
        dtype=np.int32,
    )
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
    stage_one = qf >= threshold
    bins = np.arange(paths.shape[2])[None, None, :]
    interval = interval_tracks(paths, masks)
    multi = multi_member_local_states(local_keys)[bins, paths]
    # Stage 2 requires direct support of the Viterbi tracks; a released interval track gets SWITCH_INTERVAL
    # instead of LOCAL_ALLELE_EQUIVALENCE.
    stage_two = (~stage_one) & (qaf >= threshold) & ~np.any(multi & ~support, axis=1)
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


def track_members(
    status: np.ndarray,
    classes: np.ndarray,
    local_keys: np.ndarray,
    state_classes: tuple[tuple[int, ...], ...],
    masks: np.ndarray | None = None,
    stage_two: np.ndarray | None = None,
    anchored: np.ndarray | None = None,
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
    named = (status == int(TrackStatus.SINGLETON_NAMED)) | (status == int(TrackStatus.POOLED_EQUIVALENCE))
    ids[named] = whole[classes[named]]
    local = status == int(TrackStatus.LOCAL_ALLELE_EQUIVALENCE)
    if np.any(local):
        # One member set per distinct (bin, key): O(N B + B C) overall.
        bins = np.broadcast_to(np.arange(status.shape[2]), status.shape)[local]
        codes, inverse = np.unique(bins.astype(np.int64) * (n_classes + 1) + classes[local], return_inverse=True)
        code_ids = np.empty(len(codes), dtype=np.int32)
        for position, code in enumerate(codes):
            bin_index, key = divmod(int(code), n_classes + 1)
            code_ids[position] = intern(
                row
                for class_index in np.flatnonzero(local_keys[bin_index, :n_classes] == key)
                for row in state_classes[class_index]
            )
        ids[local] = code_ids[inverse.reshape(-1)]
    interval = status == int(TrackStatus.SWITCH_INTERVAL)
    if np.any(interval):
        if masks is None:
            raise ValueError("switch-interval tracks need their candidate masks")
        anchored = np.ones(n_classes, dtype=np.bool_) if anchored is None else np.asarray(anchored, dtype=np.bool_)
        bins = np.broadcast_to(np.arange(status.shape[2]), status.shape)[interval]
        local_stage = (
            np.zeros(status.shape, dtype=np.bool_)
            if stage_two is None
            else np.broadcast_to(np.asarray(stage_two)[:, None, :], status.shape)
        )[interval]
        # One member set per distinct (bin if stage 2 else -1, mask).
        keyed = np.stack(
            [np.where(local_stage, bins, -1).astype(np.int64), np.asarray(masks)[interval].astype(np.int64)], axis=1
        )
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
