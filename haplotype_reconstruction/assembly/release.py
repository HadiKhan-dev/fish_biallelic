"""Founder release boundaries implied by live assembly alternatives (release 2.3).

The inference components remain unchanged. A released founder identity extends only over
an interval whose row-path multiset agrees in every represented terminal alternative.
Unperformed continuations retain their evaluated source families and continuation flags.
For budget-overflow families, immutable prepared-row IDs permit a conservative meet of
observed differing local alleles; unperformed continuations themselves supply no evidence.
Computational count verification flags only annotate the output. Legacy records without the sidecar have unknown ambiguity coverage.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from . import records


@dataclass(frozen=True)
class ReleaseAxes:
    """Small component axes needed after materialised allele panels have been released."""
    start: int
    stop: int
    rows: np.ndarray
    atomic_offsets: np.ndarray


@dataclass(frozen=True)
class FounderRelease:
    panels: tuple
    withheld_unobserved: tuple[int, ...]
    segments: tuple[dict, ...]
    breaks: int
    withheld_ambiguous: int
    metadata_status: str
    continuation_unverified_records: int = 0


def load_alternatives(file):
    """Final source families, or None for an older product without this release metadata."""
    key = f"{records.FINAL_PREFIX}/{records.FINAL}"
    if not file.has(key):
        return None
    payload = file.get(key, items=("release_alternatives",))
    value = payload.get("release_alternatives")
    return None if value is None else tuple(value)



def load_release_input(file):
    """(final source families or legacy None, nominal final components), without allele panels."""
    key = f"{records.FINAL_PREFIX}/{records.FINAL}"
    payload = file.get(key, items=("release_alternatives", "components"))
    value = payload.get("release_alternatives")
    return (None if value is None else tuple(value), records.decode_components(payload["components"]))

def _same_multiset(values) -> bool:
    first = sorted(values[0])
    return all(sorted(rows) == first for rows in values[1:])


def _scan_cuts(panels, start, stop, reverse=False):
    """Greedy maximal common intervals, with exact incremental path codes (O(B A K log K))."""
    codes = [[0] * len(panel) for panel in panels]
    cuts = []
    previous = None
    for block in (range(stop - 1, start - 1, -1) if reverse else range(start, stop)):
        mapping = {}
        extended = []
        for panel, row_codes in zip(panels, codes):
            extended.append([mapping.setdefault((code, int(row)), len(mapping))
                             for code, row in zip(row_codes, panel[:, block])])
        if previous is not None and not _same_multiset(extended):
            cuts.append(block + 1 if reverse else block)
            # Every single-block multiset agrees in this run; reset after the cut.
            mapping = {}
            extended = [[mapping.setdefault(int(row), len(mapping)) for row in panel[:, block]]
                        for panel in panels]
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
    bad = [block for block in range(blocks)
           if not _same_multiset([panel[:, block].tolist() for panel in panels])]
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
            intervals.extend((min(a, b), max(a, b)) for a, b in zip(left, right))
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
        return dict(kind=kind, block_start=int(a), block_stop=int(b),
                    uncertainty_interval=[int(positions[first]), int(positions[last])])

    if values:
        clipped = [value[:, lo - start:hi - start] for value in values]
        _, boundaries, bad = continuity(clipped)
        for a, b in boundaries:
            observed.append(interval(lo + a, lo + b, "source_continuity_disagreement", True))
        # Adjacent content-disagreeing blocks describe one observed source interval.
        runs = []
        for block in bad:
            if runs and block == runs[-1][1]:
                runs[-1][1] += 1
            else:
                runs.append([block, block + 1])
        observed.extend(interval(lo + a, lo + b, "source_local_content_disagreement")
                        for a, b in runs)
    observed.sort(key=lambda value: (value["block_start"], value["block_stop"], value["kind"]))
    source_status = family.get("status")
    cause = ("final_correspondence_not_established" if final_mismatch else
             family.get("cause") or family.get("reason") or "continuation_not_performed")
    return dict(common, status="continuation_unverified", unresolved_search=True, cause=cause,
                source_status=source_status, source_reason=family.get("source_reason", family.get("reason")),
                source_fit=family.get("source_fit"), block_start=int(lo), block_stop=int(hi),
                uncertainty_interval=interval(lo, hi, "scope")["uncertainty_interval"],
                scope=dict(source_level=family.get("source_level"),
                           source_start=int(family["start"]), source_stop=int(family["stop"]),
                           source_decision=family.get("source_decision")),
                evaluated_source_panels=canonical,
                evaluated_source_founder_counts=[len(value) for value in canonical],
                local_disagreement_intervals=observed,
                disagreement_basis="evaluated source prepared-row paths; final correspondence unverified")


def budget_overflow(family):
    """Only evaluated biological families stopped by a joint/search budget qualify."""
    if family.get("count_unverified") or family.get("status") == "count_unverified":
        return False
    reasons = {family.get(key) for key in ("reason", "status", "cause", "source_reason")}
    return bool(reasons & {"unresolved_joint_budget", "unresolved_branch_budget",
                           "unresolved_replay_budget"})


def load_prepared_alleles(file):
    """Immutable prepared released alleles; retain no evidence or probability arrays."""
    data = file.get(f"{records.FINAL_PREFIX}/{records.COMPLETION}", items=("prepared",))["prepared"]
    output, first = [], 0
    for founders, sites in zip(data["rows"], data["sites"]):
        stop = first + int(founders) * int(sites)
        output.append(data["released"][first:stop].reshape(int(founders), int(sites)))
        first = stop
    return tuple(output)


def _overflow_prior_family(family):
    """The complete evaluated class in which a convention may choose a representative.

    A unique smaller-count or exact-switch prior winner cannot flip. Missing switch
    evidence is not an invitation to invent that prior or withhold its competitors.
    All original alternatives remain in the surrounding continuation sidecar.
    """
    panels = tuple(family.get("panels", ()))
    prior = str(family.get("representative_prior", "")).lower()
    indices = list(range(len(panels)))
    if indices and "smaller_k" in prior:
        minimum = min(len(panels[index]) for index in indices)
        indices = [index for index in indices if len(panels[index]) == minimum]
    diagnostic = dict(prior=family.get("representative_prior"),
                      evaluated_panels=len(panels), eligible_panels=len(indices))
    if len(indices) <= 1:
        return (), dict(diagnostic, status="unique_evaluated_prior_winner")
    if len({len(panels[index]) for index in indices}) != 1:
        return (), dict(diagnostic, status="count_prior_correspondence_unverified")
    if "switch" in prior:
        switches = family.get("minimum_switches")
        if switches is None or len(switches) != len(panels) or any(switches[i] is None for i in indices):
            return (), dict(diagnostic, status="minimum_switch_prior_unverified")
        minimum = min(int(switches[index]) for index in indices)
        indices = [index for index in indices if int(switches[index]) == minimum]
        diagnostic.update(eligible_panels=len(indices), exact_minimum_switches=minimum)
    if len(indices) <= 1:
        return (), dict(diagnostic, status="unique_evaluated_prior_winner")
    return tuple(panels[index] for index in indices), dict(diagnostic, status="evaluated_prior_tie")


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
    paths = tuple(np.asarray(value, dtype=np.int32)[:, lo - start:hi - start]
                  for value in canonical)
    diagnostic = dict(status="observed_source_content_meet", prior_evaluation=prior,
                      basis="exact immutable prepared-row IDs; agreeing source-path anchors when available",
                      residual_scope="unperformed continuations outside observed local disagreements remain unverified",
                      local_groups=[], unmapped_local_ids=[])
    if prepared_alleles is None:
        diagnostic["status"] = "prepared_alleles_unavailable"
        return (), diagnostic
    if len(paths) < 2:
        diagnostic["status"] = prior["status"]
        return (), diagnostic
    source_cuts, intervals, bad = continuity(paths)
    nominal_paths = np.asarray(nominal_component.rows)[:, lo - panel.start:hi - panel.start]
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
    nominal = np.asarray(panel.rows)[:, lo - panel.start:hi - panel.start]
    updates = []
    for local in bad:
        block = lo + local
        observed_ids = {int(row) for path in paths for row in path[:, local]}
        variants = []
        alleles = np.asarray(prepared_alleles[block], dtype=np.int8)
        for anchor, values in sorted(groups.items()):
            counts = [Counter(int(row[local]) for row in value) for value in values]
            ids = sorted({row for count in counts for row in count
                          if len({other[row] for other in counts}) > 1})
            if not ids:
                continue
            # A missing member is observed biological count/content ambiguity,
            # not a missing computation. Equal-size groups compare actual alleles.
            missing = len({len(value) for value in values}) > 1
            data = alleles[ids]
            disagree = np.ones(alleles.shape[1], dtype=bool) if missing else np.any(data != data[0], axis=0)
            variants.append((anchor, frozenset(ids), disagree))
            diagnostic["local_groups"].append(dict(
                block=block, source_anchor=list(anchor), varying_prepared_ids=ids,
                source_multiplicities=[len(value) for value in values],
                evaluated_absence=missing, disagreeing_sites=int(np.count_nonzero(disagree))))
        mask = np.zeros((len(nominal), alleles.shape[1]), dtype=bool)
        for row, path in enumerate(nominal):
            identity, anchor = int(path[local]), tuple(map(int, path[anchors]))
            if identity not in observed_ids:
                diagnostic["unmapped_local_ids"].append(dict(block=block, prepared_id=identity))
                continue
            exact = [entry for entry in variants if entry[0] == anchor and identity in entry[1]]
            # An exact source anchor that contains this ID establishes its own
            # group, even when that group is invariant and another group is not.
            anchored = anchor in groups and any(int(source[local]) == identity
                                               for value in groups[anchor] for source in value)
            eligible = exact if anchored else [entry for entry in variants if identity in entry[1]]
            for _, _, disagree in eligible:
                mask[row] |= disagree
        if np.any(mask):
            updates.append((block, mask))
    diagnostic["unmapped_local_ids"] = [dict(block=block, prepared_id=identity)
        for block, identity in sorted({(item["block"], item["prepared_id"])
                                      for item in diagnostic["unmapped_local_ids"]})]
    if diagnostic["unmapped_local_ids"]:
        diagnostic["status"] = "observed_content_meet_with_unmapped_residual"
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
        by_anchor.append(dict(zip(keys, path)))
    if any(set(value) != set(by_anchor[0]) for value in by_anchor[1:]):
        return {}
    displayed = np.asarray(panel.rows)[:, lo - panel.start:hi - panel.start]
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
    shared = {row for row, count in counts[0].items()
              if count == 1 and all(other[row] == 1 for other in counts[1:])}
    displayed = np.asarray(panel.rows)[:, lo - panel.start:hi - panel.start]
    return np.asarray([tuple(map(int, row)) in shared for row in displayed], dtype=bool)


def release_founders(component_panels, panels, alternatives, *, unobserved_masks=None, nominal_components=None,
                     prepared_alleles=None):
    """Slice already carrier-masked painted/refit panels at final assembly ambiguity.

    ``component_panels`` supplies painted rows and atomic offsets; ``panels`` supplies the
    same axes after existing export masks. Withheld counts use the original carrier masks.
    The returned segment metadata includes rows on which r5 filling must remain disabled.
    """
    released, counts, metadata = [], [], []
    total_breaks = total_hidden = 0
    unverified_sources = set()
    known = alternatives is not None
    alternatives = () if alternatives is None else alternatives
    for source, (panel, (positions, calls)) in enumerate(zip(component_panels, panels)):
        positions, calls = np.asarray(positions), np.asarray(calls)
        offsets = np.append(np.asarray(panel.atomic_offsets, dtype=np.int64), len(positions))
        cuts = {panel.start, panel.stop}
        hidden = np.zeros(calls.shape, dtype=bool)
        details, verification, continuation = [], [], []
        # r5 rows are disabled where a local row's content/correspondence was withheld.
        blocked = np.zeros((len(calls), panel.stop - panel.start), dtype=bool)
        for family_index, family in enumerate(alternatives):
            source_start, source_stop = int(family["start"]), int(family["stop"])
            lo, hi = max(panel.start, source_start), min(panel.stop, source_stop)
            if lo >= hi:
                continue
            if family.get("count_unverified", False) or family.get("status") == "count_unverified":
                a, b = offsets[lo - panel.start], offsets[hi - panel.start]
                verification.append(dict(
                    status="count_unverified", cause=family.get("cause", "unperformed_count_verification"),
                    source_level=family.get("source_level"), source_decision=family.get("source_decision"),
                    source_start=source_start, source_stop=source_stop,
                    block_start=int(lo), block_stop=int(hi), scope=family.get("scope"),
                    domain_size=family.get("domain_size"), evaluated=family.get("evaluated"),
                    uncertainty_interval=[int(positions[a]), int(positions[b - 1])]))
                continue
            unknown = bool(family.get("unresolved_search", False))
            final_mismatch = False
            if not unknown and family.get("panels"):
                nominal = panel if nominal_components is None else nominal_components[source]
                final_rows = np.asarray(nominal.rows, dtype=np.int32)[:, lo - panel.start:hi - panel.start]
                final_key = sorted(row.tobytes() for row in final_rows)
                final_mismatch = not any(final_key == sorted(row.tobytes() for row in
                    np.asarray(value, dtype=np.int32)[:, lo - source_start:hi - source_start])
                    for value in family["panels"])
                unknown = final_mismatch
            common = dict(reason=family["reason"], source_level=family.get("source_level"),
                          source_decision=family.get("source_decision"),
                          source_start=source_start, source_stop=source_stop,
                          alternative_count=len(family.get("panels", ())),
                          representative_prior=family.get("representative_prior"),
                          unresolved_search=unknown,
                          status=("unresolved_final_correspondence" if final_mismatch else
                                  family.get("status", "unresolved_search" if unknown else "equal_F")))

            def annotate(a, b, kind, *, boundary=False):
                # Boundary coordinates bracket all possible cuts using observed SNPs.
                first = offsets[max(0, a - panel.start)]
                last = offsets[min(panel.stop, b) - panel.start]
                if boundary:
                    first = max(0, first - 1)
                    last = min(len(positions) - 1, last)
                else:
                    last = max(first, last - 1)
                details.append(dict(common, kind=kind, block_start=int(a), block_stop=int(b),
                                    uncertainty_interval=[int(positions[first]), int(positions[last])]))

            if unknown or not family.get("panels"):
                unverified_sources.add(family_index)
                source_record = _continuation_record(
                    family, common, lo, hi, panel, positions, offsets, final_mismatch)
                if budget_overflow(family):
                    updates, observation = _overflow_content(
                        family, lo, hi, panel, prepared_alleles,
                        panel if nominal_components is None else nominal_components[source])
                    source_record["overflow_release"] = observation
                    cuts.update(observation.get("continuity_cuts", ()))
                    for a, b in observation.get("continuity_intervals", ()):
                        annotate(a, b, "evaluated_overflow_continuity_ambiguity", boundary=True)
                    for block, mask in updates:
                        a, b = offsets[block - panel.start:block - panel.start + 2]
                        hidden[:, a:b] |= mask
                        blocked[:, block - panel.start] |= np.any(mask, axis=1)
                        cuts.update((block, block + 1))
                        annotate(block, block + 1, "evaluated_overflow_content_ambiguity")
                continuation.append(source_record)
                continue
            paths = tuple(np.asarray(value, dtype=np.int32)[:, lo - source_start:hi - source_start]
                          for value in family["panels"])
            family_cuts, intervals, bad = continuity(paths)
            exact_masks = _terminal_anchor_masks(paths, panel, lo, hi, bad, prepared_alleles)
            shared_rows = None
            if (prepared_alleles is not None and len({len(path) for path in paths}) > 1
                    and lo == source_start and hi == source_stop):
                shared_rows = _terminal_shared_rows(paths, panel, lo, hi)
            cuts.update(lo + cut for cut in family_cuts)
            for a, b in intervals:
                annotate(lo + a, lo + b, "continuity_ambiguity", boundary=True)
            for local in bad:
                block = lo + local
                cuts.update((block, block + 1))
                memberships = [Counter(path[:, local].tolist()) for path in paths]
                nominal = np.asarray(panel.rows)[:, block - panel.start]
                # Varying multiplicity leaves duplicate ownership undetermined in
                # every representative, including one with only the minimum count.
                # Conditioning this test on the nominal count is asymmetric.
                unmatched = np.asarray([len({m[int(row)] for m in memberships}) > 1
                                        for row in nominal], dtype=bool)
                a, b = offsets[block - panel.start:block - panel.start + 2]
                mask = np.broadcast_to(unmatched[:, None], (len(unmatched), b - a)).copy()
                matched, exact = exact_masks.get(local, (None, None))
                if matched is not None:
                    mask[matched] = exact[matched]
                if shared_rows is not None:
                    # Family-local protection precedes the union: no independent
                    # family, carrier mask or r5 veto can be cleared. Missing source
                    # alleles still cannot authorize a downstream refit fill.
                    source_calls = np.asarray(prepared_alleles[block], dtype=np.int8)
                    mask[shared_rows] = source_calls[nominal[shared_rows]] < 0
                blocked[:, block - panel.start] |= np.any(mask, axis=1)
                hidden[:, a:b] |= mask
                annotate(block, block + 1, "allele_content_ambiguity")
                details[-1]["allele_mask_basis"] = (
                    "exact_common_path_anchor" if matched is not None and np.all(matched) else
                    "exact_anchors_with_unmatched_multiplicity_fallback" if matched is not None and np.any(matched)
                    else "evaluated_prepared_row_multiplicity")
        details.sort(key=lambda entry: (entry["uncertainty_interval"], entry["kind"],
                                        entry["source_start"], entry["source_stop"],
                                        str(entry["source_level"]), entry["reason"], entry["status"],
                                        str(entry["representative_prior"]), entry["alternative_count"],
                                        str(entry["source_decision"])))
        verification.sort(key=lambda entry: (entry["source_start"], entry["source_stop"],
                          str(entry["source_level"]), str(entry["source_decision"]), entry["cause"],
                          str(entry["scope"]), str(entry["domain_size"]), str(entry["evaluated"])))
        continuation.sort(key=lambda entry: (entry["source_start"], entry["source_stop"],
                          str(entry["source_level"]), str(entry["source_decision"]), entry["cause"],
                          str(entry["source_fit"]), str(entry["representative_prior"]),
                          repr(entry["evaluated_source_panels"])))
        ordered = sorted(cuts)
        total_breaks += len(ordered) - 2
        for segment, (start, stop) in enumerate(zip(ordered[:-1], ordered[1:])):
            a, b = offsets[start - panel.start], offsets[stop - panel.start]
            segment_calls = calls[:, a:b]
            mask = hidden[:, a:b]
            newly_hidden = int(np.count_nonzero(mask & (segment_calls >= 0)))
            total_hidden += newly_hidden
            if np.any(mask):
                segment_calls = segment_calls.copy()
                segment_calls[mask] = -1
            unobserved = (0 if unobserved_masks is None else
                          int(np.count_nonzero(unobserved_masks[source][:, a:b])))
            # A demonstrated content-ambiguous block is a separate segment. Never
            # impute r5 calls into a row whose evaluated local content was withheld.
            no_fill = np.any(blocked[:, start - panel.start:stop - panel.start], axis=1)
            row_order = list(range(len(calls)))
            if len(ordered) > 2 or np.any(mask):
                paths = np.asarray(panel.rows)[:, start - panel.start:stop - panel.start]
                # Segment labels name released local content. This removes a suffix
                # relabelling of the point representative from the displayed columns.
                row_order.sort(key=lambda row: (segment_calls[row].tobytes(), bool(no_fill[row]),
                                                tuple(paths[row].tolist())))
                segment_calls = segment_calls[row_order]
                no_fill = no_fill[row_order]
            relevant = [entry for entry in details
                        if entry["block_start"] <= stop and entry["block_stop"] >= start]
            released.append((positions[a:b], segment_calls))
            counts.append(unobserved)
            metadata.append(dict(source_component=source, segment_index=segment,
                                 block_start=int(start), block_stop=int(stop),
                                 source_founder_rows=row_order,
                                 ambiguity=relevant,
                                 continuation_unverified=[entry for entry in continuation
                                     if entry["block_start"] < stop and entry["block_stop"] > start],
                                 count_unverified=[entry for entry in verification
                                                   if entry["block_start"] < stop and entry["block_stop"] > start],
                                 withheld_ambiguous_alleles=newly_hidden,
                                 r5_withheld_rows=np.flatnonzero(no_fill).tolist()))
    return FounderRelease(tuple(released), tuple(counts), tuple(metadata), total_breaks,
                          total_hidden, "recorded" if known else "legacy_unknown", len(unverified_sources))
