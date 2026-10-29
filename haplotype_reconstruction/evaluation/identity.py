"""Truth-only founder identity evaluation (release 2.17): copy-count strata and continuity assertions.

Copy counts. c[f, l] is the number of sampled homologues that carry true founder haplotype f at marker l (both
homologues of a sample count). A truth allele falls in the stratum absent (c = 0), single (c = 1) or multi
(c >= 2); under the founder evaluation's one-to-one component matching each stratum reports its correct, wrong
and missing alleles (missing: uncalled, an unmatched truth founder or an uncovered marker). Absent ancestry has no
sampled copy; single-copy ancestry has one, which the data may or may not resolve; neither is a recoverability
claim.

Continuity. The truth markers are cut into a fixed grid of windows of IDENTITY_WINDOW_MARKERS consecutive markers
(independent of every panel). A panel row's label in a window is the truth founder its calls match: at least
LABEL_MIN_CALLS calls, the best founder's mismatches at most 5% of the calls or one (whichever is larger), and every
other founder at least LABEL_MARGIN mismatches worse; otherwise the window is unlabelled for that row (no call,
too few calls, or founders locally indistinguishable there). Within a component, a row ASSERTS continuity between
each two consecutive windows it labels: correct when the labels agree, wrong (an identity switch) otherwise; direct
when the windows are adjacent, gap-spanning when unlabelled windows lie between them. An assertion's interval is
(centre marker of the first window, centre marker of the second] in bp on the truth grid. A correct assertion's
stratum is the founder's copy class in the poorer of its two windows; a wrong assertion's is the departing
founder's class in the second window (absent there: the row could not have continued with it). A window's copy
class is that of the floor of its median copy count.

Coverage. A truth founder's eligible windows are those where it is represented (median copy count >= 1) and at least
COVERAGE_MIN_DISTANCE markers from every other founder; two consecutive eligible windows at most COVERAGE_MAX_GAP
windows apart are covered when one row labels the founder in both and no other founder between them.

Flags (identity diagnostics of the panel's row paths, export.identity): an assertion is flagged when a flag of its
row lies in its interval (the flag's position: the first marker of the block right of its junction). Flagged
assertions are counted separately, never as errors avoided; covered pairs are also reported without flagged
assertions (a consumer that drops flagged continuity). Splitting rows or dropping assertions can raise the share of
correct assertions without raising coverage, so the counts are read together, not as one score.
"""
from __future__ import annotations

import numpy as np

IDENTITY_WINDOW_MARKERS = 100
LABEL_MIN_CALLS = 16
LABEL_MARGIN = 2
COVERAGE_MIN_DISTANCE = 3
COVERAGE_MAX_GAP = 3
STRATA = ("absent", "single", "multi")


def copy_counts(painting, positions, founders: int) -> np.ndarray:
    """int32 [F, L]: sampled homologues carrying founder haplotype f at each marker (chunk bounds as
    founders.represented_ancestry: [start, end) bp, the chunk ending at the last marker includes it)."""
    positions = np.asarray(positions)
    starts = np.searchsorted(positions, np.asarray(painting.start))
    ends_bp = np.asarray(painting.end)
    ends = np.searchsorted(positions, ends_bp)
    ends[ends_bp == positions[-1]] = len(positions)
    change = np.zeros((founders, len(positions) + 1), np.int64)
    for haplotype in (np.asarray(painting.hap1), np.asarray(painting.hap2)):
        keep = (haplotype >= 0) & (haplotype < founders)
        np.add.at(change, (haplotype[keep], starts[keep]), 1)
        np.add.at(change, (haplotype[keep], ends[keep]), -1)
    return np.cumsum(change, axis=1)[:, :-1].astype(np.int32)


def stratum_counts(panels, matches, truth, counts) -> dict:
    """Per copy stratum: truth alleles, correct and wrong calls of the one-to-one matched rows, and missing."""
    positions = np.asarray(truth["positions"])
    haplotypes = truth["haplotypes"]
    classes = np.minimum(counts, 2)
    result = {name: dict(truth_alleles=int(np.count_nonzero(classes == i)), correct=0, wrong=0)
              for i, name in enumerate(STRATA)}
    matched = {}
    for match in matches:
        matched.setdefault(match["component"], []).append((match["predicted_row"], match["truth_founder"]))
    for component, (sites, calls) in enumerate(panels):
        index = np.searchsorted(positions, sites)
        for row, founder in matched.get(component, ()):
            call = np.asarray(calls[row])
            correct = call == haplotypes[founder, index]
            wrong = (call >= 0) & ~correct
            stratum = classes[founder, index]
            for i, name in enumerate(STRATA):
                result[name]["correct"] += int(np.count_nonzero(correct & (stratum == i)))
                result[name]["wrong"] += int(np.count_nonzero(wrong & (stratum == i)))
    for values in result.values():
        values["missing"] = values["truth_alleles"] - values["correct"] - values["wrong"]
    return result


def _window_labels(calls, index, window, haplotypes, windows) -> np.ndarray:
    """int32 [K, W]: each row's truth label per window, -1 where unlabelled."""
    labels = np.full((len(calls), windows), -1, np.int32)
    for row in range(len(calls)):
        called = calls[row] >= 0
        where = window[called]
        n = np.bincount(where, minlength=windows)
        mismatches = np.stack([np.bincount(where, weights=calls[row, called] != haplotypes[f, index[called]],
                                           minlength=windows) for f in range(len(haplotypes))])
        ordered = np.sort(mismatches, axis=0)
        close = ordered[0] <= np.maximum(1, np.floor(0.05 * n))
        if len(haplotypes) > 1:
            close &= ordered[1] - ordered[0] >= LABEL_MARGIN
        labels[row] = np.where(close & (n >= LABEL_MIN_CALLS), np.argmin(mismatches, axis=0), -1)
    return labels


def identity_counts(panels, truth, counts, flags=None) -> dict:
    """Continuity assertions and coverage of one stage's panels on the truth grid (module docstring).

    ``panels``: [(positions, calls [K, L])]; ``truth``: dict(haplotypes int8 [F, L], positions int64 [L]);
    ``counts``: copy_counts; ``flags``: per component {row: sorted int64 flag positions} or None."""
    haplotypes = np.asarray(truth["haplotypes"])
    positions = np.asarray(truth["positions"])
    founders, markers = haplotypes.shape
    windows = (markers + IDENTITY_WINDOW_MARKERS - 1) // IDENTITY_WINDOW_MARKERS
    marker_window = np.arange(markers) // IDENTITY_WINDOW_MARKERS
    # The middle marker of each window [start, stop); the final window may be partial.
    starts = np.arange(windows) * IDENTITY_WINDOW_MARKERS
    stops = np.minimum(starts + IDENTITY_WINDOW_MARKERS, markers)
    centre = positions[starts + (stops - starts) // 2]
    median = np.zeros((founders, windows))
    distance = np.full((founders, windows), np.inf)
    for w in range(windows):
        part = slice(w * IDENTITY_WINDOW_MARKERS, min(markers, (w + 1) * IDENTITY_WINDOW_MARKERS))
        median[:, w] = np.median(counts[:, part], axis=1)
        block = haplotypes[:, part]
        pairwise = (block[:, None, :] != block[None, :, :]).sum(axis=2).astype(float)
        np.fill_diagonal(pairwise, np.inf)
        distance[:, w] = pairwise.min(axis=1)
    window_class = np.minimum(np.floor(median), 2).astype(np.int64)
    eligible = (median >= 1) & (distance >= COVERAGE_MIN_DISTANCE)
    keys = [f"{kind}_{verdict}_{name}" for kind in ("direct", "gap") for verdict in ("correct", "wrong")
            for name in STRATA]
    result = {key: 0 for key in keys}
    result.update(flagged_correct=0, flagged_wrong=0, labelled_cells=0)

    def flagged(component, row, left, right):
        marks = None if flags is None else flags.get(component, {}).get(row)
        if marks is None or not len(marks):
            return False
        i = np.searchsorted(marks, left, side="right")
        return i < len(marks) and marks[i] <= right

    holders = {}
    rows_labels = {}
    for component, (sites, calls) in enumerate(panels):
        index = np.searchsorted(positions, np.asarray(sites))
        labels = _window_labels(np.asarray(calls), index, marker_window[index], haplotypes, windows)
        rows_labels[component] = labels
        result["labelled_cells"] += int(np.count_nonzero(labels >= 0))
        for row in range(len(labels)):
            labelled = np.flatnonzero(labels[row] >= 0)
            for w in labelled:
                holders.setdefault((int(labels[row, w]), int(w)), []).append((component, row))
            for w1, w2 in zip(labelled[:-1], labelled[1:]):
                f1, f2 = int(labels[row, w1]), int(labels[row, w2])
                good = f1 == f2
                if flagged(component, row, int(centre[w1]), int(centre[w2])):
                    result["flagged_correct" if good else "flagged_wrong"] += 1
                stratum = (min(window_class[f1, w1], window_class[f1, w2]) if good else window_class[f1, w2])
                kind = "direct" if w2 == w1 + 1 else "gap"
                result[f"{kind}_{'correct' if good else 'wrong'}_{STRATA[stratum]}"] += 1
    for name in ("single", "multi"):
        result[f"coverage_pairs_{name}"] = result[f"coverage_covered_{name}"] = 0
        result[f"coverage_covered_unflagged_{name}"] = 0
    for founder in range(founders):
        eligible_windows = np.flatnonzero(eligible[founder])
        for w1, w2 in zip(eligible_windows[:-1], eligible_windows[1:]):
            if w2 - w1 > COVERAGE_MAX_GAP:
                continue
            name = "multi" if min(window_class[founder, w1], window_class[founder, w2]) >= 2 else "single"
            result[f"coverage_pairs_{name}"] += 1
            covered = unflagged = False
            for component, row in set(holders.get((founder, int(w1)), ())) & set(holders.get((founder, int(w2)), ())):
                between = rows_labels[component][row, w1 + 1:w2]
                if np.all((between < 0) | (between == founder)):
                    covered = True
                    if not flagged(component, row, int(centre[w1]), int(centre[w2])):
                        unflagged = True
                        break
            result[f"coverage_covered_{name}"] += covered
            result[f"coverage_covered_unflagged_{name}"] += unflagged
    result["windows"] = int(windows)
    return result
