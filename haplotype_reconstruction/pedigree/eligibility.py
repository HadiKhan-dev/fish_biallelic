"""Candidate universe of observed-parent inference: a parent mask plus a pair rule.

Eligibility encodes design assumptions supplied by the caller: which samples are inference targets,
which samples are legitimate observed-parent candidates of each target, which unordered parent pairs
are admissible together (for example one recorded female and one recorded male), and which
child <- parent edges have a known design chronology (exempt from the inferred-direction terms).
It never asserts individual parentage. The stage reads no metadata: dataset adapters build the
record (``from_cohorts``, ``from_constraints``); simulations use ``unrestricted``.

The M2 universe of a child c is {a < b : parents[c, a] and parents[c, b] and compatible_pairs[a, b]}
(``compatible_pairs`` None admits every pair), so no N x N x N cube is ever formed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

UNRESTRICTED_POLICY = "unrestricted_no_metadata"
CONSTRAINTS_POLICY = "explicit_sample_constraints"
CONSTRAINT_KEYS = frozenset(
    ("candidate_parents", "excluded_samples", "ineligible_children", "direction_supported_edges"))


def _bool_matrix(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean array with shape {shape}")
    array = np.array(array, dtype=np.bool_, order="C", copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ParentEligibility:
    """Caller design assumptions over the ordered sample axis (never parentage truth).

    sample_ids:       ordered sample IDs, identical to the run's sample axis (N).
    children:         bool [N]; samples whose observed parents are inferred.
    parents:          bool [N, N]; [child, parent] admissible M1 edges. False on the diagonal and on
                      rows of non-target samples.
    compatible_pairs: bool [N, N] symmetric with a False diagonal, or None (every pair); an unordered
                      M2 pair {a, b} of child c needs parents[c, a], parents[c, b] and this entry.
    chronology:       bool [N, N]; [child, parent] edges whose direction the design fixes (subset of
                      ``parents``). Such edges are exempt from inferred orientation and family terms.
    policy:           short policy name recorded in reports.
    assumptions:      plain-language statements of what the policy assumes.
    source_fields:    metadata fields the adapter read to build the policy.
    sexes:            recorded sex label per sample (upper-cased; "F", "M", or anything else = unknown), or
                      None without metadata. Never inferred. Release 2.2: it orders released parent pairs
                      (dam = F first, sire = M second; pedigree.slots) and enters the identity.
    """

    sample_ids: tuple[str, ...]
    children: np.ndarray
    parents: np.ndarray
    compatible_pairs: np.ndarray | None
    chronology: np.ndarray
    policy: str
    assumptions: tuple[str, ...] = ()
    source_fields: tuple[str, ...] = ()
    sexes: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        # Real boundary: adapters build this record from user metadata or JSON constraints.
        samples = tuple(str(value) for value in self.sample_ids)
        if len(samples) != len(set(samples)):
            raise ValueError("parent eligibility sample IDs must be unique")
        n = len(samples)
        children = _bool_matrix(self.children, (n,), "eligible children")
        parents = _bool_matrix(self.parents, (n, n), "eligible parents")
        chronology = _bool_matrix(self.chronology, (n, n), "chronology")
        diagonal = np.arange(n)
        if np.any(parents[diagonal, diagonal]) or np.any(chronology[diagonal, diagonal]):
            raise ValueError("eligibility cannot admit self-parenting")
        if np.any(parents[~children]):
            raise ValueError("non-target samples cannot have eligible parents")
        if np.any(chronology & ~parents):
            raise ValueError("chronology edges must be eligible parent edges")
        pairs = None
        if self.compatible_pairs is not None:
            pairs = _bool_matrix(self.compatible_pairs, (n, n), "compatible parent pairs")
            if not np.array_equal(pairs, pairs.T) or np.any(pairs[diagonal, diagonal]):
                raise ValueError("compatible parent pairs must be symmetric with a False diagonal")
        if not isinstance(self.policy, str) or not self.policy:
            raise ValueError("eligibility policy must be a non-empty string")
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "children", children)
        object.__setattr__(self, "parents", parents)
        object.__setattr__(self, "compatible_pairs", pairs)
        object.__setattr__(self, "chronology", chronology)
        object.__setattr__(self, "assumptions", tuple(str(value) for value in self.assumptions))
        object.__setattr__(self, "source_fields", tuple(str(value) for value in self.source_fields))
        if self.sexes is not None:
            sexes = tuple(str(value).strip().upper() for value in self.sexes)
            if len(sexes) != n:
                raise ValueError("one recorded sex label is required per sample")
            object.__setattr__(self, "sexes", sexes)

    # ---------------------------------------------------------------- construction
    @classmethod
    def unrestricted(cls, sample_ids: Sequence[str]) -> "ParentEligibility":
        """Every sample is a target and every other sample a candidate; no chronology."""
        n = len(sample_ids)
        parents = ~np.eye(n, dtype=np.bool_)
        return cls(tuple(sample_ids), np.ones(n, dtype=np.bool_), parents, None,
                   np.zeros((n, n), dtype=np.bool_), UNRESTRICTED_POLICY,
                   ("No metadata eligibility or simulated parentage truth enters inference.",))

    @classmethod
    def from_cohorts(
            cls, sample_ids: Sequence[str], cohorts: Sequence[str],
            parent_cohort_of: Mapping[str, str], *, sexes: Sequence[str] | None = None,
            require_opposite_sex_pairs: bool = False, policy: str,
            assumptions: Sequence[str] = (), source_fields: Sequence[str] = (),
    ) -> "ParentEligibility":
        """Generation steps, e.g. {"F2": "F1", "F3": "F2"}: cohort labels constrain, never assign.

        Targets are samples whose cohort is a key; a target's candidates are the samples of its
        parent cohort. Every eligible edge is design-ordered (chronology). With
        ``require_opposite_sex_pairs`` an M2 pair needs one recorded "F" and one recorded "M"
        sample (other sex labels pair with nobody); the parent order stays arbitrary.
        """
        labels = np.asarray([str(value) for value in cohorts])
        n = len(sample_ids)
        if labels.shape != (n,):
            raise ValueError("one cohort label is required per sample")
        children = np.isin(labels, list(parent_cohort_of))
        parents = np.zeros((n, n), dtype=np.bool_)
        for child_cohort, parent_cohort in parent_cohort_of.items():
            parents[labels == child_cohort, :] = labels == parent_cohort
        np.fill_diagonal(parents, False)
        pairs = None
        if require_opposite_sex_pairs:
            if sexes is None:
                raise ValueError("opposite-sex pairs need one recorded sex per sample")
            sex = np.asarray([str(value).upper() for value in sexes])
            if sex.shape != (n,):
                raise ValueError("one sex label is required per sample")
            female, male = sex == "F", sex == "M"
            pairs = (female[:, None] & male[None, :]) | (male[:, None] & female[None, :])
        return cls(tuple(sample_ids), children, parents, pairs, parents.copy(), policy,
                   tuple(assumptions), tuple(source_fields),
                   sexes=None if sexes is None else tuple(str(value) for value in sexes))

    @classmethod
    def from_constraints(cls, sample_ids: Sequence[str],
                         constraints: Mapping[str, Any]) -> "ParentEligibility":
        """Sample-name constraints (the parsed eligibility JSON of a generic VCF run).

        ``candidate_parents`` {child: [parents]} replaces that child's candidates ([] allows none;
        unlisted children keep every other sample); ``excluded_samples`` are neither targets nor
        candidates; ``ineligible_children`` are candidates only; ``direction_supported_edges``
        [[parent, child], ...] mark independently known design order, not parentage.
        """
        if not isinstance(constraints, Mapping) or set(constraints) - CONSTRAINT_KEYS:
            raise ValueError("eligibility constraints may contain only "
                             + ", ".join(sorted(CONSTRAINT_KEYS)))
        index = {name: position for position, name in enumerate(sample_ids)}

        def position(name: str) -> int:
            if name not in index:
                raise ValueError(f"unknown eligibility sample {name!r}")
            return index[name]

        n = len(sample_ids)
        children = np.ones(n, dtype=np.bool_)
        parents = ~np.eye(n, dtype=np.bool_)
        for child, candidates in constraints.get("candidate_parents", {}).items():
            if not isinstance(candidates, list):
                raise ValueError("candidate_parents values must be lists of sample IDs")
            child_index = position(child)
            selected = [position(name) for name in candidates]
            if child_index in selected:
                raise ValueError("a sample cannot be its own parent")
            parents[child_index] = False
            parents[child_index, selected] = True
        for name in constraints.get("excluded_samples", []):
            excluded = position(name)
            children[excluded] = False
            parents[excluded] = False
            parents[:, excluded] = False
        for name in constraints.get("ineligible_children", []):
            target = position(name)
            children[target] = False
            parents[target] = False
        chronology = np.zeros((n, n), dtype=np.bool_)
        for edge in constraints.get("direction_supported_edges", []):
            if not isinstance(edge, list) or len(edge) != 2:
                raise ValueError("direction_supported_edges entries must be [parent, child]")
            parent, child = (position(name) for name in edge)
            if not parents[child, parent]:
                raise ValueError("direction support names an ineligible parent/child edge")
            chronology[child, parent] = True
        return cls(tuple(sample_ids), children, parents, None, chronology, CONSTRAINTS_POLICY,
                   ("Caller constraints are eligibility/chronology, not known individual parentage.",),
                   tuple(sorted(constraints)))

    # ---------------------------------------------------------------- queries
    @property
    def supplied(self) -> bool:
        """Whether the caller restricted the universe (False for ``unrestricted``)."""
        return self.policy != UNRESTRICTED_POLICY

    def pair_allowed(self, child: int, first: int, second: int) -> bool:
        """Exact M2 membership of one unordered pair."""
        if first == second or not (self.parents[child, first] and self.parents[child, second]):
            return False
        return self.compatible_pairs is None or bool(self.compatible_pairs[first, second])

    def pair_mask(self, child: int | np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        """Vectorised ``pair_allowed``."""
        allowed = (first != second) & self.parents[child, first] & self.parents[child, second]
        if self.compatible_pairs is not None:
            allowed &= self.compatible_pairs[first, second]
        return allowed

    def parent_counts(self) -> np.ndarray:
        """int64 [N]: eligible M1 candidates per sample."""
        return np.count_nonzero(self.parents, axis=1).astype(np.int64)

    def pair_counts(self) -> np.ndarray:
        """int64 [N]: size of each sample's unordered M2 universe (the identity-prior denominator)."""
        counts = self.parent_counts()
        if self.compatible_pairs is None:
            return counts * (counts - 1) // 2
        candidates = self.parents.astype(np.int64)
        within = (candidates @ self.compatible_pairs.astype(np.int64)) * candidates
        return np.sum(within, axis=1) // 2

    def identity(self) -> str:
        """SHA-256 of the policy text and every mask (a cache identity, not an integrity check)."""
        # DEDUP(T0): use store/identity's canonical digest once it exists.
        fields = {"sample_ids": list(self.sample_ids), "policy": self.policy,
                  "assumptions": list(self.assumptions), "source_fields": list(self.source_fields)}
        if self.sexes is not None:          # release 2.2: recorded sexes order released parent pairs
            fields["sexes"] = list(self.sexes)
        digest = hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode())
        arrays = (("children", self.children), ("parents", self.parents),
                  ("compatible_pairs", self.compatible_pairs), ("chronology", self.chronology))
        for name, array in arrays:
            digest.update(name.encode())
            if array is None:
                digest.update(b"any")
            else:
                digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
                digest.update(array.tobytes())
        return digest.hexdigest()

    def summary(self) -> dict[str, Any]:
        """Small JSON-safe counts for the product and reports."""
        targets = np.flatnonzero(self.children)
        parent_counts = self.parent_counts()[targets]
        pair_counts = self.pair_counts()[targets]
        return {
            "policy": self.policy,
            "supplied": self.supplied,
            "sample_count": len(self.sample_ids),
            "eligible_child_count": int(len(targets)),
            "candidate_parent_sample_count": int(np.count_nonzero(np.any(self.parents, axis=0))),
            "chronology_edge_count": int(np.count_nonzero(self.chronology)),
            "minimum_parent_candidates_per_target": int(parent_counts.min()) if len(targets) else 0,
            "maximum_parent_candidates_per_target": int(parent_counts.max(initial=0)),
            "minimum_parent_pairs_per_target": int(pair_counts.min()) if len(targets) else 0,
            "maximum_parent_pairs_per_target": int(pair_counts.max(initial=0)),
            "individual_parentage_ground_truth": False,
            "assumptions": list(self.assumptions),
            "source_fields": list(self.source_fields),
            "recorded_sexes": (None if self.sexes is None else
                               {label: int(sum(sex == label for sex in self.sexes))
                                for label in sorted(set(self.sexes))}),
            "parent_slot_basis": ("dam_sire where both parents have recorded sexes F and M, else unordered "
                                  "(canonical order by observed sites); sexes are never inferred"),
        }
