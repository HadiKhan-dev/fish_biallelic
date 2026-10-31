"""Candidate universe of observed-parent inference: every sequenced sample (release 3.4).

The pipeline reads no sample metadata, so nothing narrows the universe: every sample of the run is an inference
target, every other sample is a candidate parent of it, and every unordered pair of its candidates is an admissible
two-parent configuration. Which sample is the parent of which is inferred from the data (orientation, family terms,
the sibling-as-parent guard and the release tiers); the universe never asserts parentage or direction.

The record keeps the universe as boolean masks over the run's sample axis, the form the scoring and decision
kernels are written over. The M2 universe of a child c is {a < b : parents[c, a] and parents[c, b]}, so no
N x N x N cube is ever formed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

POLICY = "every_sequenced_sample"


@dataclass(frozen=True)
class ParentEligibility:
    """The candidate universe over the ordered sample axis (never parentage truth).

    sample_ids: ordered sample IDs, identical to the run's sample axis (N).
    children:   bool [N]; samples whose observed parents are inferred (every sample).
    parents:    bool [N, N]; [child, parent] admissible M1 edges (every pair of distinct samples).
    """

    sample_ids: tuple[str, ...]
    children: np.ndarray
    parents: np.ndarray

    @classmethod
    def unrestricted(cls, sample_ids: Sequence[str]) -> "ParentEligibility":
        """Every sample is a target and every other sample a candidate."""
        samples = tuple(str(value) for value in sample_ids)
        if len(samples) != len(set(samples)):
            raise ValueError("sample IDs must be unique")
        n = len(samples)
        children = np.ones(n, dtype=np.bool_)
        parents = ~np.eye(n, dtype=np.bool_)
        children.setflags(write=False)
        parents.setflags(write=False)
        return cls(samples, children, parents)

    # ---------------------------------------------------------------- queries
    def pair_allowed(self, child: int, first: int, second: int) -> bool:
        """Exact M2 membership of one unordered pair."""
        return first != second and bool(self.parents[child, first] and self.parents[child, second])

    def pair_mask(self, child: int | np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        """Vectorised ``pair_allowed``."""
        return (first != second) & self.parents[child, first] & self.parents[child, second]

    def parent_counts(self) -> np.ndarray:
        """int64 [N]: eligible M1 candidates per sample."""
        return np.count_nonzero(self.parents, axis=1).astype(np.int64)

    def pair_counts(self) -> np.ndarray:
        """int64 [N]: size of each sample's unordered M2 universe (the identity-prior denominator)."""
        counts = self.parent_counts()
        return counts * (counts - 1) // 2

    def identity(self) -> str:
        """SHA-256 of the universe and its masks (a cache identity, not an integrity check)."""
        fields = {"sample_ids": list(self.sample_ids), "policy": POLICY}
        digest = hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode())
        for name, array in (("children", self.children), ("parents", self.parents)):
            digest.update(name.encode())
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        return digest.hexdigest()

    def summary(self) -> dict[str, Any]:
        """Small JSON-safe counts for the product and reports."""
        targets = np.flatnonzero(self.children)
        parent_counts = self.parent_counts()[targets]
        pair_counts = self.pair_counts()[targets]
        return {
            "policy": POLICY,
            "sample_count": len(self.sample_ids),
            "eligible_child_count": int(len(targets)),
            "candidate_parent_sample_count": int(np.count_nonzero(np.any(self.parents, axis=0))),
            "minimum_parent_candidates_per_target": int(parent_counts.min()) if len(targets) else 0,
            "maximum_parent_candidates_per_target": int(parent_counts.max(initial=0)),
            "minimum_parent_pairs_per_target": int(pair_counts.min()) if len(targets) else 0,
            "maximum_parent_pairs_per_target": int(pair_counts.max(initial=0)),
            "individual_parentage_ground_truth": False,
            "sample_metadata": False,
            "parent_slot_basis": ("unordered: canonical order by observed sites per contig, then sample name; "
                                  "no sex is recorded or inferred"),
        }
