"""Canonicalize comparable chromosome-level M0/M1/M2 evidence."""
from __future__ import annotations


from typing import Any, Mapping, Optional, Sequence


import numpy as np


def _as_parent_state_evidence(
    value: Any,
    n_samples: int,
    eligibility: Optional[pedigree_eligibility._ResolvedParentEligibility]=None,
) -> pedigree_models.ParentStateEvidence:
    """Validate one comparable parent-state evidence object."""
    if isinstance(value, pedigree_models.ParentStateEvidence):
        item = value
    elif isinstance(value, Mapping):
        try:
            item = pedigree_models.ParentStateEvidence(
                contig=str(value["contig"]),
                trios=np.asarray(value["trios"], dtype=np.int64),
                zero_parent_log_likelihoods=np.asarray(
                    value["zero_parent_log_likelihoods"], dtype=np.float64
                ),
                one_parent_log_likelihoods=np.asarray(
                    value["one_parent_log_likelihoods"], dtype=np.float64
                ),
                two_parent_log_likelihoods=np.asarray(
                    value["two_parent_log_likelihoods"], dtype=np.float64
                ),
                informative_markers=int(value["informative_markers"]),
                edge_matched_bins=value.get("edge_matched_bins"),
                edge_exposed_bins=value.get("edge_exposed_bins"),
                pair_explained_bins=value.get("pair_explained_bins"),
                pair_exposed_bins=value.get("pair_exposed_bins"),
                structure_total_bins=value.get("structure_total_bins"),
            )
        except KeyError as error:
            raise pedigree_models.PedigreeEvidenceError(
                f"parent-state evidence is missing {error.args[0]!r}"
            ) from error
    else:
        raise pedigree_models.PedigreeEvidenceError(
            "parent-state evidence must be SmartParentStateEvidence or a mapping"
        )

    trios = np.asarray(item.trios, dtype=np.int64)
    zero = np.asarray(item.zero_parent_log_likelihoods, dtype=np.float64)
    one = np.asarray(item.one_parent_log_likelihoods, dtype=np.float64)
    two = np.asarray(item.two_parent_log_likelihoods, dtype=np.float64)
    if eligibility is None:
        required_children = np.ones(n_samples, dtype=np.bool_)
        required_parents = ~np.eye(n_samples, dtype=np.bool_)
    else:
        required_children = eligibility.eligible_children
        required_parents = eligibility.eligible_parents
    if trios.ndim != 2 or trios.shape[1] != 3:
        raise pedigree_models.PedigreeEvidenceError("parent-state trios must have shape (rows, 3)")
    if (
        zero.shape != (n_samples,)
        or np.any(~np.isfinite(zero[required_children]))
        or np.any(np.isnan(zero) | np.isposinf(zero))
    ):
        raise pedigree_models.PedigreeEvidenceError(
            "zero-parent evidence must be one finite log likelihood per child"
        )
    if one.shape != (n_samples, n_samples):
        raise pedigree_models.PedigreeEvidenceError(
            "one-parent evidence must have shape (samples, samples)"
        )
    if (
        np.any(~np.isfinite(one[required_parents]))
        or np.any(np.isnan(one) | np.isposinf(one))
    ):
        raise pedigree_models.PedigreeEvidenceError(
            "off-diagonal one-parent log likelihoods must be finite"
        )
    if not np.all(np.isneginf(np.diag(one))):
        raise pedigree_models.PedigreeEvidenceError(
            "the one-parent self-parent diagonal must be negative infinity"
        )
    if two.shape != (len(trios),) or np.any(~np.isfinite(two)):
        raise pedigree_models.PedigreeEvidenceError(
            "two-parent evidence must be one finite log likelihood per trio"
        )
    if item.informative_markers < 1:
        raise pedigree_models.PedigreeEvidenceError("informative_markers must be positive")

    structure_values = (
        getattr(item, "edge_matched_bins", None),
        getattr(item, "edge_exposed_bins", None),
        getattr(item, "pair_explained_bins", None),
        getattr(item, "pair_exposed_bins", None),
        getattr(item, "structure_total_bins", None),
    )
    if any(raw is None for raw in structure_values):
        raise pedigree_models.PedigreeEvidenceError(
            "combined_v1 requires all parenthood structure arrays and "
            "structure_total_bins on every contig"
        )
    else:
        edge_matched = np.asarray(structure_values[0], dtype=np.float64)
        edge_exposed = np.asarray(structure_values[1], dtype=np.float64)
        pair_explained = np.asarray(structure_values[2], dtype=np.float64)
        pair_exposed = np.asarray(structure_values[3], dtype=np.float64)
        structure_total_bins = float(structure_values[4])
        arrays = (edge_matched, edge_exposed, pair_explained, pair_exposed)
        if (
            edge_matched.shape != (n_samples, n_samples)
            or edge_exposed.shape != edge_matched.shape
            or pair_explained.shape != (len(trios),)
            or pair_exposed.shape != pair_explained.shape
            or any(np.any(~np.isfinite(array)) for array in arrays)
            or any(np.any(array < 0.0) for array in arrays)
            or np.any(edge_matched > edge_exposed)
            or np.any(pair_explained > pair_exposed)
            or not np.isfinite(structure_total_bins)
            or structure_total_bins <= 0.0
        ):
            raise pedigree_models.PedigreeEvidenceError(
                "parenthood structure counts must be finite non-negative arrays "
                "with valid totals and matched counts no larger than exposed"
            )
        if not (
            np.array_equal(edge_matched, edge_matched.T)
            and np.array_equal(edge_exposed, edge_exposed.T)
        ):
            raise pedigree_models.PedigreeEvidenceError("edge structure counts must be symmetric")

    return pedigree_models.ParentStateEvidence(
        contig=str(item.contig),
        trios=trios,
        zero_parent_log_likelihoods=zero,
        one_parent_log_likelihoods=one,
        two_parent_log_likelihoods=two,
        informative_markers=int(item.informative_markers),
        edge_matched_bins=edge_matched,
        edge_exposed_bins=edge_exposed,
        pair_explained_bins=(
            pair_explained
        ),
        pair_exposed_bins=pair_exposed,
        structure_total_bins=structure_total_bins,
    )


def _canonical_parent_state_evidence(
    evidence: Sequence[pedigree_models.ParentStateEvidence],
    n_samples: int,
    eligibility: Optional[pedigree_eligibility._ResolvedParentEligibility]=None,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],
    Optional[np.ndarray], Optional[np.ndarray], list[str],
]:
    """Align a fixed two-parent screen while retaining full M0/M1 evidence.

    Internally generated panels are already canonical and identically ordered.
    Validate that common case with vector operations and append its score array
    directly. Permuted caller-supplied panels are canonically sorted once,
    preserving the historical support for arbitrary external row order without
    rebuilding a 539,200-entry Python dictionary for every contig.
    """
    reference_trios: Optional[np.ndarray] = None
    zero_rows = []
    one_rows = []
    two_rows = []
    edge_matched_rows = []
    edge_exposed_rows = []
    pair_explained_rows = []
    pair_exposed_rows = []
    structure_total_rows = []
    structure_presence: Optional[bool] = None
    markers = []
    names = []
    seen_contigs = set()
    for raw in evidence:
        item = _as_parent_state_evidence(raw, n_samples, eligibility)
        if item.contig in seen_contigs:
            raise pedigree_models.PedigreeEvidenceError(f"duplicate contig identifier {item.contig!r}")
        seen_contigs.add(item.contig)
        edge_matched = item.edge_matched_bins
        edge_exposed = item.edge_exposed_bins
        pair_explained = item.pair_explained_bins
        structure_total = item.structure_total_bins
        pair_exposed = item.pair_exposed_bins
        item_has_structure = edge_matched is not None
        if structure_presence is None:
            structure_presence = item_has_structure
        elif structure_presence != item_has_structure:
            raise pedigree_models.PedigreeEvidenceError(
                "parenthood structure evidence must be present on every contig "
                "or absent from every contig"
            )

        canonical = np.asarray(item.trios, dtype=np.int64)
        scores = np.asarray(
            item.two_parent_log_likelihoods, dtype=np.float64
        )
        if len(canonical):
            child = canonical[:, 0]
            first = canonical[:, 1]
            second = canonical[:, 2]
            if (
                np.any(child < 0)
                or np.any(child >= n_samples)
                or np.any(first < 0)
                or np.any(first >= n_samples)
                or np.any(second < 0)
                or np.any(second >= n_samples)
            ):
                raise pedigree_models.PedigreeEvidenceError("trio index outside sample array")
            if np.any(child == first) or np.any(child == second):
                raise pedigree_models.PedigreeEvidenceError(
                    "invalid self-parent or duplicate-parent trio"
                )
            swap = second < first
            if np.any(swap):
                canonical = canonical.copy()
                temporary = canonical[swap, 1].copy()
                canonical[swap, 1] = canonical[swap, 2]
                canonical[swap, 2] = temporary
            if np.any(canonical[:, 1] == canonical[:, 2]):
                raise pedigree_models.PedigreeEvidenceError(
                    "invalid self-parent or duplicate-parent trio"
                )

            previous = canonical[:-1]
            current = canonical[1:]
            ordered = np.all(
                (current[:, 0] > previous[:, 0])
                | (
                    (current[:, 0] == previous[:, 0])
                    & (
                        (current[:, 1] > previous[:, 1])
                        | (
                            (current[:, 1] == previous[:, 1])
                            & (current[:, 2] >= previous[:, 2])
                        )
                    )
                )
            )
            if not ordered:
                order = np.lexsort((
                    canonical[:, 2], canonical[:, 1], canonical[:, 0]
                ))
                canonical = np.ascontiguousarray(canonical[order])
                scores = np.ascontiguousarray(scores[order])
                if pair_explained is not None:
                    pair_explained = np.ascontiguousarray(pair_explained[order])
                    pair_exposed = np.ascontiguousarray(pair_exposed[order])
            duplicate = np.all(canonical[1:] == canonical[:-1], axis=1)
            if np.any(duplicate):
                key = tuple(
                    int(value) for value in canonical[1:][duplicate][0]
                )
                raise pedigree_models.PedigreeEvidenceError(
                    f"duplicate trio key {key} on {item.contig}"
                )

        if reference_trios is None:
            reference_trios = canonical
        elif not np.array_equal(canonical, reference_trios):
            raise pedigree_models.PedigreeEvidenceError(
                "every contig must score the same canonical two-parent panel"
            )
        zero_rows.append(item.zero_parent_log_likelihoods)
        one_rows.append(item.one_parent_log_likelihoods)
        two_rows.append(scores)
        markers.append(item.informative_markers)
        names.append(item.contig)
        if item_has_structure:
            edge_matched_rows.append(edge_matched)
            edge_exposed_rows.append(edge_exposed)
            structure_total_rows.append(structure_total)
            pair_explained_rows.append(pair_explained)
            pair_exposed_rows.append(pair_exposed)
    if not zero_rows or reference_trios is None:
        raise pedigree_models.PedigreeEvidenceError("at least one parent-state contig is required")
    return (
        np.asarray(reference_trios, dtype=np.int64).reshape((-1, 3)),
        np.asarray(zero_rows, dtype=np.float64),
        np.asarray(one_rows, dtype=np.float64),
        np.asarray(two_rows, dtype=np.float64),
        np.asarray(markers, dtype=np.float64),
        (
            None if not structure_presence
            else np.asarray(edge_matched_rows, dtype=np.float64)
        ),
        (
            None if not structure_presence
            else np.asarray(edge_exposed_rows, dtype=np.float64)
        ),
        (
            None if not structure_presence
            else np.asarray(pair_explained_rows, dtype=np.float64)
        ),
        (
            None if not structure_presence
            else np.asarray(pair_exposed_rows, dtype=np.float64)
        ),
        (
            None if not structure_presence
            else np.asarray(structure_total_rows, dtype=np.float64)
        ),
        names,
    )

import haplotype_reconstruction.pedigree.eligibility as pedigree_eligibility
import haplotype_reconstruction.pedigree.models as pedigree_models
