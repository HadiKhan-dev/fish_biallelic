"""Release founder intervals and calls from evaluated assembly alternatives.

Owns carrier masking, continuity cuts, ambiguity masks and identity flag mapping.
Does not read records, clocks or formatting options. Evaluated source-row paths
are the only evidence for cuts and content withholding (P0 U16; OU03/OU06).
Method: docs/methods/assembly.md#release; WP_AS6 §4.7 and WP_OU1 §5.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from functools import cache
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from haplotype_reconstruction.contracts.assembly import (
    FounderIdentityFlag,
    FounderRelease,
    ReleaseAxes,
    ReleaseFamily,
    ReleaseSegment,
)
from haplotype_reconstruction.contracts.panels import FounderComponent, partner_changes
from haplotype_reconstruction.core.errors import InvariantError

if TYPE_CHECKING:
    from typing import TypedDict

    class _ReleaseSegmentMetadata(TypedDict):
        source_component: int
        segment_index: int
        block_start: int
        block_stop: int
        source_founder_rows: tuple[int, ...]
        ambiguity: tuple[Mapping[str, Any], ...]
        continuation_unverified: tuple[Mapping[str, Any], ...]
        count_unverified: tuple[Mapping[str, Any], ...]
        withheld_ambiguous_alleles: int
        r5_withheld_rows: tuple[int, ...]


# Contract statuses, P0 U16 assembly/release.py. These are not tuning parameters.
COUNT_UNVERIFIED = "count_unverified"
UNRESOLVED_SEARCH = "unresolved_search"
RECORDED = "recorded"
UNRESOLVED_JOINT_BUDGET = "unresolved_joint_budget"
UNRESOLVED_BRANCH_BUDGET = "unresolved_branch_budget"
UNRESOLVED_REPLAY_BUDGET = "unresolved_replay_budget"

# Contract strings from 3.5 assembly-core U02/U13/U16; no tuning or normalization.
ALLELE_CONTENT_AMBIGUITY = "allele_content_ambiguity"
CONTINUATION_NOT_PERFORMED = "continuation_not_performed"
CONTINUATION_UNVERIFIED = "continuation_unverified"
CONTINUITY_AMBIGUITY = "continuity_ambiguity"
COUNT_PRIOR_CORRESPONDENCE_UNVERIFIED = "count_prior_correspondence_unverified"
EQUAL_F = "equal_F"
EVALUATED_OVERFLOW_CONTENT_AMBIGUITY = "evaluated_overflow_content_ambiguity"
EVALUATED_OVERFLOW_CONTINUITY_AMBIGUITY = "evaluated_overflow_continuity_ambiguity"
EVALUATED_PREPARED_ROW_MULTIPLICITY = "evaluated_prepared_row_multiplicity"
EVALUATED_PRIOR_TIE = "evaluated_prior_tie"
EXACT_ANCHORS_WITH_UNMATCHED_MULTIPLICITY_FALLBACK = "exact_anchors_with_unmatched_multiplicity_fallback"
EXACT_COMMON_PATH_ANCHOR = "exact_common_path_anchor"
FINAL_CORRESPONDENCE_NOT_ESTABLISHED = "final_correspondence_not_established"
LEGACY_UNKNOWN = "legacy_unknown"
MINIMUM_SWITCH_PRIOR_UNVERIFIED = "minimum_switch_prior_unverified"
OBSERVED_CONTENT_MEET_WITH_UNMAPPED_RESIDUAL = "observed_content_meet_with_unmapped_residual"
OBSERVED_SOURCE_CONTENT_MEET = "observed_source_content_meet"
PREPARED_ALLELES_UNAVAILABLE = "prepared_alleles_unavailable"
SMALLER_K = "smaller_k"
SOURCE_CONTINUITY_DISAGREEMENT = "source_continuity_disagreement"
SOURCE_LOCAL_CONTENT_DISAGREEMENT = "source_local_content_disagreement"
UNIQUE_EVALUATED_PRIOR_WINNER = "unique_evaluated_prior_winner"
UNPERFORMED_COUNT_VERIFICATION = "unperformed_count_verification"
UNRESOLVED_FINAL_CORRESPONDENCE = "unresolved_final_correspondence"


def _same_multiset(values) -> bool:
    first = sorted(values[0])
    return all(sorted(rows) == first for rows in values[1:])


def _scan_cuts(panels, start, stop, reverse=False):
    """Greedy maximal common intervals, with exact incremental path codes (O(B A K log K))."""
    codes = [[0] * len(panel) for panel in panels]
    cuts = []
    previous = None
    for block in range(stop - 1, start - 1, -1) if reverse else range(start, stop):
        mapping = {}
        extended = []
        for panel, row_codes in zip(panels, codes, strict=False):
            extended.append(
                [
                    mapping.setdefault((code, int(row)), len(mapping))
                    for code, row in zip(row_codes, panel[:, block], strict=False)
                ]
            )
        if previous is not None and not _same_multiset(extended):
            cuts.append(block + 1 if reverse else block)
            # Every single-block multiset agrees in this run; reset after the cut.
            mapping = {}
            extended = [[mapping.setdefault(int(row), len(mapping)) for row in panel[:, block]] for panel in panels]
        codes = extended
        previous = block
    return sorted(cuts)


def continuity(panels):
    """(cuts, uncertainty block-boundary intervals, local-content-ambiguous blocks).

    Cuts are relative to the source span. Content-disagreeing blocks are isolated; other
    intervals agree up to a single whole-row permutation, including duplicate rows.
    """
    panels = tuple(np.asarray(panel, dtype=np.int32) for panel in panels)
    if not panels:
        return (), (), ()
    blocks = panels[0].shape[1]
    bad = [block for block in range(blocks) if not _same_multiset([panel[:, block].tolist() for panel in panels])]
    cuts = {edge for block in bad for edge in (block, block + 1) if 0 < edge < blocks}
    intervals = []
    start = 0
    for stop in [*bad, blocks]:
        if stop > start:
            left = _scan_cuts(panels, start, stop)
            right = _scan_cuts(panels, start, stop, reverse=True)
            cuts.update(left)
            cuts.update(right)
            # Both greedy scans give minimum-cardinality partitions of hereditary
            # common intervals; pair corresponding extremal boundaries.
            intervals.extend((min(a, b), max(a, b)) for a, b in zip(left, right, strict=False))
        start = stop + 1
    return tuple(sorted(cuts)), tuple(intervals), tuple(bad)


def _continuation_record(family, common, lo, hi, panel, positions, offsets, final_mismatch):
    """Observed source-family differences, distinct from unverified final correspondence.

    Prepared-row paths share the assembly's atomic catalogue. Their local multisets
    and continuity intervals describe the evaluated source choices, not final allele
    errors. Never infer a final row mapping from an approximate alignment.
    """
    start = int(family["start"])
    values = [np.asarray(value, dtype=np.int32) for value in family.get("panels", ())]
    canonical = sorted(tuple(sorted(tuple(map(int, row)) for row in value)) for value in values)
    observed = []

    def interval(a, b, kind, boundary=False):
        first, last = int(offsets[a - panel.start]), int(offsets[b - panel.start])
        if boundary:
            first, last = max(0, first - 1), min(len(positions) - 1, last)
        else:
            last = max(first, last - 1)
        return dict(
            kind=kind,
            block_start=int(a),
            block_stop=int(b),
            uncertainty_interval=[int(positions[first]), int(positions[last])],
        )

    if values:
        clipped = [value[:, lo - start : hi - start] for value in values]
        _, boundaries, bad = continuity(clipped)
        for a, b in boundaries:
            observed.append(interval(lo + a, lo + b, SOURCE_CONTINUITY_DISAGREEMENT, True))
        # Adjacent content-disagreeing blocks describe one observed source interval.
        runs = []
        for block in bad:
            if runs and block == runs[-1][1]:
                runs[-1][1] += 1
            else:
                runs.append([block, block + 1])
        observed.extend(interval(lo + a, lo + b, SOURCE_LOCAL_CONTENT_DISAGREEMENT) for a, b in runs)
    observed.sort(key=lambda value: (value["block_start"], value["block_stop"], value["kind"]))
    source_status = family.get("status")
    cause = (
        FINAL_CORRESPONDENCE_NOT_ESTABLISHED
        if final_mismatch
        else family.get("cause") or family.get("reason") or CONTINUATION_NOT_PERFORMED
    )
    return dict(
        common,
        status=CONTINUATION_UNVERIFIED,
        unresolved_search=True,
        cause=cause,
        source_status=source_status,
        source_reason=family.get("source_reason", family.get("reason")),
        source_fit=family.get("source_fit"),
        block_start=int(lo),
        block_stop=int(hi),
        uncertainty_interval=interval(lo, hi, "scope")["uncertainty_interval"],
        scope=dict(
            source_level=family.get("source_level"),
            source_start=int(family["start"]),
            source_stop=int(family["stop"]),
            source_decision=family.get("source_decision"),
        ),
        **(
            {
                "window_domains": family["window_domains"],
                "domain_basis": family.get("domain_basis"),
                "window_domains_scope": family.get("window_domains_scope"),
            }
            if "window_domains" in family
            else {}
        ),
        evaluated_source_panels=canonical,
        evaluated_source_founder_counts=[len(value) for value in canonical],
        local_disagreement_intervals=observed,
        disagreement_basis="evaluated source prepared-row paths; final correspondence unverified",
    )


def budget_overflow(family):
    """Only evaluated biological families stopped by a joint/search budget qualify."""
    if family.get(COUNT_UNVERIFIED) or family.get("status") == COUNT_UNVERIFIED:
        return False
    reasons = {family.get(key) for key in ("reason", "status", "cause", "source_reason")}
    return bool(reasons & {UNRESOLVED_JOINT_BUDGET, UNRESOLVED_BRANCH_BUDGET, UNRESOLVED_REPLAY_BUDGET})


def _overflow_prior_family(family):
    """The complete evaluated class in which a convention may choose a representative.

    A unique smaller-count or exact-switch prior winner cannot flip. Missing switch
    evidence is not an invitation to invent that prior or withhold its competitors.
    All original alternatives remain in the surrounding continuation sidecar.
    """
    panels = tuple(family.get("panels", ()))
    prior = str(family.get("representative_prior", "")).lower()
    indices = list(range(len(panels)))
    if indices and SMALLER_K in prior:
        minimum = min(len(panels[index]) for index in indices)
        indices = [index for index in indices if len(panels[index]) == minimum]
    diagnostic = dict(
        prior=family.get("representative_prior"), evaluated_panels=len(panels), eligible_panels=len(indices)
    )
    if len(indices) <= 1:
        return (), dict(diagnostic, status=UNIQUE_EVALUATED_PRIOR_WINNER)
    if len({len(panels[index]) for index in indices}) != 1:
        return (), dict(diagnostic, status=COUNT_PRIOR_CORRESPONDENCE_UNVERIFIED)
    if "switch" in prior:
        switches = family.get("minimum_switches")
        if switches is None or len(switches) != len(panels) or any(switches[i] is None for i in indices):
            return (), dict(diagnostic, status=MINIMUM_SWITCH_PRIOR_UNVERIFIED)
        minimum = min(int(switches[index]) for index in indices)
        indices = [index for index in indices if int(switches[index]) == minimum]
        diagnostic.update(eligible_panels=len(indices), exact_minimum_switches=minimum)
    if len(indices) <= 1:
        return (), dict(diagnostic, status=UNIQUE_EVALUATED_PRIOR_WINNER)
    return tuple(panels[index] for index in indices), dict(diagnostic, status=EVALUATED_PRIOR_TIE)


def _overflow_content(family, lo, hi, panel, prepared_alleles, nominal_component):
    """Observed local-content meet, with exact prepared-row provenance.

    Equal path anchors outside content-disagreeing blocks separate independent source
    founders when possible. If a later path no longer has such an anchor, all source
    groups containing its exact local prepared ID remain possible. Their union is
    deliberately conservative; it is not an inferred final row correspondence.
    Newly introduced local IDs have no evaluated source correspondence and stay flagged.
    """
    start = int(family["start"])
    eligible, prior = _overflow_prior_family(family)
    canonical = sorted(tuple(sorted(tuple(map(int, row)) for row in value)) for value in eligible)
    paths = tuple(np.asarray(value, dtype=np.int32)[:, lo - start : hi - start] for value in canonical)
    diagnostic = dict(
        status=OBSERVED_SOURCE_CONTENT_MEET,
        prior_evaluation=prior,
        basis="exact immutable prepared-row IDs; agreeing source-path anchors when available",
        residual_scope="unperformed continuations outside observed local disagreements remain unverified",
        local_groups=[],
        unmapped_local_ids=[],
    )
    if prepared_alleles is None:
        diagnostic["status"] = PREPARED_ALLELES_UNAVAILABLE
        return (), diagnostic
    if len(paths) < 2:
        diagnostic["status"] = prior["status"]
        return (), diagnostic
    source_cuts, intervals, bad = continuity(paths)
    nominal_paths = np.asarray(nominal_component.rows)[:, lo - panel.start : hi - panel.start]
    final_key = sorted(map(tuple, nominal_paths.tolist()))
    direct = any(final_key == sorted(map(tuple, path.tolist())) for path in paths)
    # A whole-row permutation gives an exact final correspondence for continuity.
    # Without it, local ID content remains measurable but long-range rows do not.
    diagnostic["continuity_cuts"] = [lo + cut for cut in source_cuts] if direct else []
    diagnostic["continuity_intervals"] = [[lo + a, lo + b] for a, b in intervals] if direct else []
    bad_set = set(bad)
    anchors = [block for block in range(hi - lo) if block not in bad_set]
    groups = {}
    for index, path in enumerate(paths):
        for row in path:
            key = tuple(map(int, row[anchors]))
            groups.setdefault(key, [[] for _ in paths])[index].append(row)
    nominal = np.asarray(panel.rows)[:, lo - panel.start : hi - panel.start]
    updates = []
    for local in bad:
        block = lo + local
        observed_ids = {int(row) for path in paths for row in path[:, local]}
        variants = []
        alleles = np.asarray(prepared_alleles[block], dtype=np.int8)
        for anchor, values in sorted(groups.items()):
            counts = [Counter(int(row[local]) for row in value) for value in values]
            ids = sorted({row for count in counts for row in count if len({other[row] for other in counts}) > 1})
            if not ids:
                continue
            # A missing member is observed biological count/content ambiguity,
            # not a missing computation. Equal-size groups compare actual alleles.
            missing = len({len(value) for value in values}) > 1
            data = alleles[ids]
            disagree = np.ones(alleles.shape[1], dtype=bool) if missing else np.any(data != data[0], axis=0)
            variants.append((anchor, frozenset(ids), disagree))
            diagnostic["local_groups"].append(
                dict(
                    block=block,
                    source_anchor=list(anchor),
                    varying_prepared_ids=ids,
                    source_multiplicities=[len(value) for value in values],
                    evaluated_absence=missing,
                    disagreeing_sites=int(np.count_nonzero(disagree)),
                )
            )
        mask = np.zeros((len(nominal), alleles.shape[1]), dtype=bool)
        for row, path in enumerate(nominal):
            identity, anchor = int(path[local]), tuple(map(int, path[anchors]))
            if identity not in observed_ids:
                diagnostic["unmapped_local_ids"].append(dict(block=block, prepared_id=identity))
                continue
            exact = [entry for entry in variants if entry[0] == anchor and identity in entry[1]]
            # An exact source anchor that contains this ID establishes its own
            # group, even when that group is invariant and another group is not.
            anchored = anchor in groups and any(
                int(source[local]) == identity for value in groups[anchor] for source in value
            )
            eligible = exact if anchored else [entry for entry in variants if identity in entry[1]]
            for _, _, disagree in eligible:
                mask[row] |= disagree
        if np.any(mask):
            updates.append((block, mask))
    diagnostic["unmapped_local_ids"] = [
        dict(block=block, prepared_id=identity)
        for block, identity in sorted(
            {(item["block"], item["prepared_id"]) for item in diagnostic["unmapped_local_ids"]}
        )
    ]
    if diagnostic["unmapped_local_ids"]:
        diagnostic["status"] = OBSERVED_CONTENT_MEET_WITH_UNMAPPED_RESIDUAL
    return tuple(updates), diagnostic


def _terminal_anchor_masks(paths, panel, lo, hi, bad, prepared_alleles):
    """Exact same-count terminal correspondence, or the existing conservative fallback.

    Removing all locally disputed blocks leaves invariant source path anchors. Every
    anchor must identify exactly one row in every evaluated panel, and the anchor sets
    must agree. This is an equality proof, not a distance or assignment optimization.
    Displayed rows are located by their own exact anchors and immutable local IDs;
    painting relabelling that destroys this correspondence keeps the fallback mask.
    """
    if prepared_alleles is None or not bad or len({len(path) for path in paths}) != 1:
        return {}
    bad_set = set(bad)
    good = [block for block in range(hi - lo) if block not in bad_set]
    if not good:
        return {}
    by_anchor = []
    for path in paths:
        keys = [tuple(map(int, row[good])) for row in path]
        if len(set(keys)) != len(keys):
            return {}
        by_anchor.append(dict(zip(keys, path, strict=False)))
    if any(set(value) != set(by_anchor[0]) for value in by_anchor[1:]):
        return {}
    displayed = np.asarray(panel.rows)[:, lo - panel.start : hi - panel.start]
    result = {}
    for local in bad:
        alleles = np.asarray(prepared_alleles[lo + local], dtype=np.int8)
        matched = np.zeros(len(displayed), dtype=bool)
        mask = np.zeros((len(displayed), alleles.shape[1]), dtype=bool)
        for index, row in enumerate(displayed):
            anchor = tuple(map(int, row[good]))
            if anchor not in by_anchor[0]:
                continue
            identities = [int(value[anchor][local]) for value in by_anchor]
            if int(row[local]) not in identities:
                continue
            data = alleles[identities]
            matched[index] = True
            # LOCAL1 requires every evaluated alternative to supply the retained
            # allele. Even unanimous source missingness cannot justify a refit fill.
            mask[index] = np.any(data < 0, axis=0) | np.any(data != data[0], axis=0)
        result[local] = (matched, mask)
    return result


def _terminal_shared_rows(paths, panel, lo, hi):
    """Unique full paths shared by every unequal-count terminal member (LOCAL2).

    The caller supplies the entire evaluated source span, never a clipped fragment.
    Exact immutable prepared-row sequences establish partial correspondence even
    when other paths have different local multiplicities. Duplicates do not qualify.
    """
    counts = [Counter(tuple(map(int, row)) for row in path) for path in paths]
    shared = {row for row, count in counts[0].items() if count == 1 and all(other[row] == 1 for other in counts[1:])}
    displayed = np.asarray(panel.rows)[:, lo - panel.start : hi - panel.start]
    return np.asarray([tuple(map(int, row)) in shared for row in displayed], dtype=bool)


def release_founders(
    component_panels: Sequence[ReleaseAxes],
    panels: Sequence[tuple[NDArray[np.int64], NDArray[np.int8]]],
    alternatives: Sequence[ReleaseFamily | Mapping[str, Any]] | None,
    *,
    unobserved_masks: Sequence[NDArray[np.bool_]] | None = None,
    nominal_components: Sequence[FounderComponent] | None = None,
    prepared_alleles: Sequence[NDArray[np.int8]] | None = None,
    identity_paths: Sequence[tuple[int, NDArray[np.int64], NDArray[np.int64], NDArray[np.int32]]] = (),
    refit_ambiguous_alleles: int = 0,
) -> FounderRelease:
    """Slice already carrier-masked painted/refit panels at final assembly ambiguity.

    ``component_panels`` supplies painted rows and atomic offsets; ``panels`` supplies the
    same axes after existing export masks. Withheld counts use the original carrier masks.
    The returned segment metadata includes rows on which r5 filling must remain disabled.
    """
    metadata: list[dict[str, Any]]
    nominal: ReleaseAxes | FounderComponent | NDArray[np.int32]
    paths: tuple[np.ndarray, ...] | np.ndarray
    released, counts, metadata = [], [], []
    total_breaks = total_hidden = 0
    unverified_sources = set()
    known = alternatives is not None
    alternatives = () if alternatives is None else tuple(ReleaseFamily.from_mapping(f) for f in alternatives)
    for source, (panel, (positions, calls)) in enumerate(zip(component_panels, panels, strict=False)):
        positions, calls = np.asarray(positions), np.asarray(calls)
        offsets = np.append(np.asarray(panel.atomic_offsets, dtype=np.int64), len(positions))
        cuts = {panel.start, panel.stop}
        hidden = np.zeros(calls.shape, dtype=bool)
        details: list[dict[str, Any]]
        details, verification, continuation = [], [], []
        # r5 rows are disabled where a local row's content/correspondence was withheld.
        blocked = np.zeros((len(calls), panel.stop - panel.start), dtype=bool)
        for family_index, family in enumerate(alternatives):
            source_start, source_stop = int(family["start"]), int(family["stop"])
            lo, hi = max(panel.start, source_start), min(panel.stop, source_stop)
            if lo >= hi:
                continue
            if family.get(COUNT_UNVERIFIED, False) or family.get("status") == COUNT_UNVERIFIED:
                a, b = offsets[lo - panel.start], offsets[hi - panel.start]
                verification.append(
                    dict(
                        status=COUNT_UNVERIFIED,
                        cause=family.get("cause", UNPERFORMED_COUNT_VERIFICATION),
                        source_level=family.get("source_level"),
                        source_decision=family.get("source_decision"),
                        source_start=source_start,
                        source_stop=source_stop,
                        block_start=int(lo),
                        block_stop=int(hi),
                        scope=family.get("scope"),
                        domain_size=family.get("domain_size"),
                        evaluated=family.get("evaluated"),
                        uncertainty_interval=[int(positions[a]), int(positions[b - 1])],
                    )
                )
                continue
            unknown = bool(family.get("unresolved_search", False))
            final_mismatch = False
            if not unknown and family.get("panels"):
                nominal = panel if nominal_components is None else nominal_components[source]
                final_rows = np.asarray(nominal.rows, dtype=np.int32)[:, lo - panel.start : hi - panel.start]
                final_key = sorted(row.tobytes() for row in final_rows)
                final_mismatch = not any(
                    final_key
                    == sorted(
                        row.tobytes()
                        for row in np.asarray(value, dtype=np.int32)[:, lo - source_start : hi - source_start]
                    )
                    for value in family["panels"]
                )
                unknown = final_mismatch
            common = dict(
                reason=family["reason"],
                source_level=family.get("source_level"),
                source_decision=family.get("source_decision"),
                source_start=source_start,
                source_stop=source_stop,
                alternative_count=len(family.get("panels", ())),
                representative_prior=family.get("representative_prior"),
                unresolved_search=unknown,
                status=(
                    UNRESOLVED_FINAL_CORRESPONDENCE
                    if final_mismatch
                    else family.get("status", UNRESOLVED_SEARCH if unknown else EQUAL_F)
                ),
            )

            def annotate(
                a,
                b,
                kind,
                *,
                boundary=False,
                offsets=offsets,
                panel=panel,
                positions=positions,
                details=details,
                common=common,
            ):
                # Boundary coordinates bracket all possible cuts using observed SNPs.
                first = offsets[max(0, a - panel.start)]
                last = offsets[min(panel.stop, b) - panel.start]
                if boundary:
                    first = max(0, first - 1)
                    last = min(len(positions) - 1, last)
                else:
                    last = max(first, last - 1)
                details.append(
                    dict(
                        common,
                        kind=kind,
                        block_start=int(a),
                        block_stop=int(b),
                        uncertainty_interval=[int(positions[first]), int(positions[last])],
                    )
                )

            if unknown or not family.get("panels"):
                unverified_sources.add(family_index)
                source_record = _continuation_record(family, common, lo, hi, panel, positions, offsets, final_mismatch)
                if budget_overflow(family):
                    updates, observation = _overflow_content(
                        family,
                        lo,
                        hi,
                        panel,
                        prepared_alleles,
                        panel if nominal_components is None else nominal_components[source],
                    )
                    source_record["overflow_release"] = observation
                    cuts.update(observation.get("continuity_cuts", ()))
                    for a, b in observation.get("continuity_intervals", ()):
                        annotate(a, b, EVALUATED_OVERFLOW_CONTINUITY_AMBIGUITY, boundary=True)
                    for block, mask in updates:
                        a, b = offsets[block - panel.start : block - panel.start + 2]
                        hidden[:, a:b] |= mask
                        blocked[:, block - panel.start] |= np.any(mask, axis=1)
                        cuts.update((block, block + 1))
                        annotate(block, block + 1, EVALUATED_OVERFLOW_CONTENT_AMBIGUITY)
                continuation.append(source_record)
                continue
            paths = tuple(
                np.asarray(value, dtype=np.int32)[:, lo - source_start : hi - source_start]
                for value in family["panels"]
            )
            family_cuts, intervals, bad = continuity(paths)
            exact_masks = _terminal_anchor_masks(paths, panel, lo, hi, bad, prepared_alleles)
            shared_rows = None
            if (
                prepared_alleles is not None
                and len({len(path) for path in paths}) > 1
                and lo == source_start
                and hi == source_stop
            ):
                shared_rows = _terminal_shared_rows(paths, panel, lo, hi)
            cuts.update(lo + cut for cut in family_cuts)
            for a, b in intervals:
                annotate(lo + a, lo + b, CONTINUITY_AMBIGUITY, boundary=True)
            for local in bad:
                block = lo + local
                cuts.update((block, block + 1))
                memberships = [Counter(path[:, local].tolist()) for path in paths]
                nominal = np.asarray(panel.rows)[:, block - panel.start]
                # Varying multiplicity leaves duplicate ownership undetermined in
                # every representative, including one with only the minimum count.
                # Conditioning this test on the nominal count is asymmetric.
                unmatched = np.asarray([len({m[int(row)] for m in memberships}) > 1 for row in nominal], dtype=bool)
                a, b = offsets[block - panel.start : block - panel.start + 2]
                mask = np.broadcast_to(unmatched[:, None], (len(unmatched), b - a)).copy()
                matched, exact = exact_masks.get(local, (None, None))
                if matched is not None:
                    mask[matched] = exact[matched]
                if shared_rows is not None:
                    # Family-local protection precedes the union: no independent
                    # family, carrier mask or r5 veto can be cleared. Missing source
                    # alleles still cannot authorize a downstream refit fill.
                    source_calls = np.asarray(cast(Sequence[NDArray[np.int8]], prepared_alleles)[block], dtype=np.int8)
                    mask[shared_rows] = source_calls[nominal[shared_rows]] < 0
                blocked[:, block - panel.start] |= np.any(mask, axis=1)
                hidden[:, a:b] |= mask
                annotate(block, block + 1, ALLELE_CONTENT_AMBIGUITY)
                details[-1]["allele_mask_basis"] = (
                    EXACT_COMMON_PATH_ANCHOR
                    if matched is not None and np.all(matched)
                    else EXACT_ANCHORS_WITH_UNMATCHED_MULTIPLICITY_FALLBACK
                    if matched is not None and np.any(matched)
                    else EVALUATED_PREPARED_ROW_MULTIPLICITY
                )
        details.sort(
            key=lambda entry: (
                entry["uncertainty_interval"],
                entry["kind"],
                entry["source_start"],
                entry["source_stop"],
                str(entry["source_level"]),
                entry["reason"],
                entry["status"],
                str(entry["representative_prior"]),
                entry["alternative_count"],
                str(entry["source_decision"]),
            )
        )
        verification.sort(
            key=lambda entry: (
                entry["source_start"],
                entry["source_stop"],
                str(entry["source_level"]),
                str(entry["source_decision"]),
                entry["cause"],
                str(entry["scope"]),
                str(entry["domain_size"]),
                str(entry["evaluated"]),
            )
        )
        continuation.sort(
            key=lambda entry: (
                entry["source_start"],
                entry["source_stop"],
                str(entry["source_level"]),
                str(entry["source_decision"]),
                entry["cause"],
                str(entry["source_fit"]),
                str(entry["representative_prior"]),
                repr(entry["evaluated_source_panels"]),
            )
        )
        ordered = sorted(cuts)
        total_breaks += len(ordered) - 2
        for segment, (start, stop) in enumerate(zip(ordered[:-1], ordered[1:], strict=False)):
            a, b = offsets[start - panel.start], offsets[stop - panel.start]
            segment_calls = calls[:, a:b]
            mask = hidden[:, a:b]
            newly_hidden = int(np.count_nonzero(mask & (segment_calls >= 0)))
            total_hidden += newly_hidden
            if np.any(mask):
                segment_calls = segment_calls.copy()
                segment_calls[mask] = -1
            unobserved = 0 if unobserved_masks is None else int(np.count_nonzero(unobserved_masks[source][:, a:b]))
            # A demonstrated content-ambiguous block is a separate segment. Never
            # impute r5 calls into a row whose evaluated local content was withheld.
            no_fill = np.any(blocked[:, start - panel.start : stop - panel.start], axis=1)
            row_order = list(range(len(calls)))
            if len(ordered) > 2 or np.any(mask):
                paths = np.asarray(panel.rows)[:, start - panel.start : stop - panel.start]
                # Segment labels name released local content. This removes a suffix
                # relabelling of the point representative from the displayed columns.
                row_order.sort(
                    key=lambda row: (segment_calls[row].tobytes(), bool(no_fill[row]), tuple(paths[row].tolist()))
                )
                segment_calls = segment_calls[row_order]
                no_fill = no_fill[row_order]
            relevant = [entry for entry in details if entry["block_start"] <= stop and entry["block_stop"] >= start]
            released.append((positions[a:b], segment_calls))
            counts.append(unobserved)
            metadata.append(
                dict(
                    source_component=source,
                    segment_index=segment,
                    block_start=int(start),
                    block_stop=int(stop),
                    source_founder_rows=row_order,
                    ambiguity=relevant,
                    continuation_unverified=[
                        entry for entry in continuation if entry["block_start"] < stop and entry["block_stop"] > start
                    ],
                    count_unverified=[
                        entry for entry in verification if entry["block_start"] < stop and entry["block_stop"] > start
                    ],
                    withheld_ambiguous_alleles=newly_hidden,
                    r5_withheld_rows=np.flatnonzero(no_fill).tolist(),
                )
            )
    segments = tuple(
        ReleaseSegment(
            positions=position,
            calls=call,
            **cast(
                "_ReleaseSegmentMetadata",
                {
                    key: tuple(value)
                    if key
                    in (
                        "source_founder_rows",
                        "ambiguity",
                        CONTINUATION_UNVERIFIED,
                        COUNT_UNVERIFIED,
                        "r5_withheld_rows",
                    )
                    else value
                    for key, value in meta.items()
                },
            ),
            first_position=int(position[0]),
            last_position=int(position[-1]),
            markers=len(position),
            called_alleles=int(np.count_nonzero(call >= 0)),
            missing_alleles=int(np.count_nonzero(call < 0)),
            withheld_unobserved_alleles=count,
        )
        for (position, call), count, meta in zip(released, counts, metadata, strict=False)
    )
    carrier_counts = tuple(
        0 if unobserved_masks is None else int(np.count_nonzero(unobserved_masks[index]))
        for index in range(len(component_panels))
    )
    return FounderRelease(
        segments=segments,
        carrier_withheld_by_component=carrier_counts,
        unobserved_founder_alleles=sum(carrier_counts),
        founder_continuity_breaks=total_breaks,
        assembly_ambiguous_alleles=total_hidden,
        assembly_ambiguity_metadata=RECORDED if known else LEGACY_UNKNOWN,
        assembly_continuation_unverified_records=len(unverified_sources),
        assembly_count_unverified_records=sum(
            bool(f.get(COUNT_UNVERIFIED, False) or f.get("status") == COUNT_UNVERIFIED) for f in alternatives
        ),
        identity_flags=identity_flags(identity_paths, segments),
        refit_ambiguous_alleles=int(refit_ambiguous_alleles),
        work_counters=dict(
            components=len(component_panels),
            families=len(alternatives),
            segments=len(released),
            continuity_breaks=total_breaks,
            withheld_alleles=total_hidden,
        ),
    )


def withhold_unobserved(
    panels: Sequence[tuple[NDArray[np.int64], NDArray[np.int8]]],
    carrier_seen: Sequence[NDArray[np.bool_]],
) -> tuple[tuple[tuple[NDArray[np.int64], NDArray[np.int8]], ...], tuple[int, ...], tuple[NDArray[np.bool_], ...]]:
    """Apply OU03 to already aligned PA2 carrier-observation masks.

    panels contain (int64 positions[L], int8 calls[K,L]); carrier_seen contains
    bool[K,L] with the same component order. Return panels, counts and original
    hidden masks. A missing call never increases the withholding count. Inputs
    are not changed; only affected call arrays are copied (3.5 export/run.py).
    """
    if len(panels) != len(carrier_seen):
        raise InvariantError("founder panels and carrier masks differ")
    result, withheld, masks = [], [], []
    for (positions, calls), seen in zip(panels, carrier_seen, strict=False):
        positions = np.asarray(positions, dtype=np.int64)
        calls = np.asarray(calls, dtype=np.int8)
        seen = np.asarray(seen, dtype=bool)
        if calls.shape != seen.shape:
            raise InvariantError("founder calls and carrier mask axes differ")
        hidden = (calls >= 0) & ~seen
        withheld.append(int(np.count_nonzero(hidden)))
        masks.append(hidden)
        if withheld[-1]:
            calls = calls.copy()
            calls[hidden] = -1
        result.append((positions, calls))
    return tuple(result), tuple(withheld), tuple(masks)


def identity_flags(identity_paths, segments=None) -> tuple[FounderIdentityFlag, ...]:
    """Map duplicate-partner changes through released segment row permutations.

    Paths are (first global block, positions, local atomic offsets, source rows).
    Left/right partners retain their source enumeration. Missing right owners
    skip a flag; a missing left owner falls back to the right owner (A40-16).
    A segment with an incomplete source row permutation is an invariant error.
    """
    output = []
    for source, (first_block, positions, offsets, rows) in enumerate(identity_paths):
        positions = np.asarray(positions, dtype=np.int64)
        offsets = np.asarray(offsets, dtype=np.int64)
        if segments is None:
            owners = [(source, first_block, first_block + len(offsets), tuple(range(len(rows))))]
        else:
            owners = [
                (index, s.block_start, s.block_stop, tuple(s.source_founder_rows))
                for index, s in enumerate(segments)
                if s.source_component == source
            ]

        @cache
        def owner(block, *, owners=owners, first_block=first_block):
            return next((item for item in owners if item[1] <= first_block + block < item[2]), None)

        @cache
        def columns(item, *, rows=rows):
            order = list(item[3])
            if sorted(order) != list(range(len(rows))):
                raise InvariantError("release segment does not contain every source founder row once")
            return {row: column for column, row in enumerate(order)}

        for flag in partner_changes(rows):
            right_owner = owner(flag.block)
            if right_owner is None:
                continue
            right = columns(right_owner)
            at_break = first_block + flag.block == right_owner[1]
            left_owner = (owner(flag.block - 1) if at_break else right_owner) or right_owner
            left = columns(left_owner)
            output.append(
                FounderIdentityFlag(
                    export_component=right_owner[0],
                    founder_row=right[flag.row],
                    source_component=source,
                    source_row=flag.row,
                    block=first_block + flag.block,
                    left_position=int(positions[offsets[flag.block] - 1]),
                    right_position=int(positions[offsets[flag.block]]),
                    left_local_row=flag.left_local_row,
                    right_local_row=flag.right_local_row,
                    left_partner_component=left_owner[0],
                    left_partners=tuple(left[row] for row in flag.left_partners),
                    right_partners=tuple(right[row] for row in flag.right_partners),
                    at_release_break=bool(at_break),
                )
            )
    return tuple(output)
