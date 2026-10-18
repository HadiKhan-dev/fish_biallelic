"""The chromosome evidence assembly reads, built once per contig from the evidence record.

Every assembly of a contig (the local-selection context assemblies and the final assembly) and
founder refinement read the same view of the founder evidence (``ContigEvidence.founder_view``):
genotype likelihoods where a sample is observed at a discovery site, exactly 1/3 elsewhere.
Completion and the linker read it in float64; the joins and founder refinement read one float32
copy (made once, shared by every level and every component).

The view's arrays live in POSIX shared memory from the start, so the completion and hierarchy pool
workers attach them (``AssemblyEvidence.shared``) instead of receiving a chromosome-sized copy per
pool; the segments are unlinked when the view is garbage collected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import weakref

import numpy as np
from numba import njit, prange

from ..execution.shared_arrays import SharedArrayRef, close_shared_memory, empty_shared_array


@njit(parallel=True, cache=True)
def _fill_views(likelihoods, observed, kept, neutral, scoring, seen):
    """Write the float64 neutral view, its float32 copy and observed & kept in one pass over the cells."""
    samples, sites, _ = likelihoods.shape
    for sample in prange(samples):
        for site in range(sites):
            used = observed[sample, site] and kept[site]
            seen[sample, site] = used
            for genotype in range(3):
                value = likelihoods[sample, site, genotype] if used else 1.0 / 3.0
                neutral[sample, site, genotype] = value
                scoring[sample, site, genotype] = value


@dataclass(frozen=True)
class EvidenceRefs:
    """Shared-memory references of a view's neutral (float64[N, L, 3]), scoring (float32[N, L, 3])
    and observed (bool[N, L]) arrays, for pool workers (``execution.shared_arrays.attach_shared_array``)."""

    neutral: SharedArrayRef
    scoring: SharedArrayRef
    observed: SharedArrayRef


@dataclass(frozen=True)
class AssemblyEvidence:
    """One contig's evidence as assembly reads it (build it with :meth:`from_contig` or
    :meth:`from_likelihoods`).

    positions: int64[L] the contig's site positions (evidence site axis).
    neutral: float64[N, L, 3] founder-view genotype likelihoods (normalised) at observed discovery
        sites, exactly 1/3 at every other cell (unobserved, not a discovery site, or a sample the
        contamination QC excluded from founder construction).
    scoring: float32[N, L, 3] the same values in float32 (joins and founder refinement).
    observed: bool[N, L] observed at a discovery site (False for excluded samples).
    kept: bool[L] the discovery sites.
    identity: identity of the evidence record the view was built from (join reuse keys).
    shared: where neutral, scoring and observed live in shared memory (None for a copy made by
        unpickling: pools then copy the arrays they need).
    """

    positions: np.ndarray
    neutral: np.ndarray
    scoring: np.ndarray
    observed: np.ndarray
    kept: np.ndarray
    identity: str
    shared: EvidenceRefs | None = field(default=None, repr=False, compare=False)

    @property
    def samples(self) -> int:
        return int(self.neutral.shape[0])

    @property
    def nbytes(self) -> int:
        """Bytes of the view's chromosome-sized arrays (neutral, scoring, observed)."""
        return int(self.neutral.nbytes + self.scoring.nbytes + self.observed.nbytes)

    @classmethod
    def from_likelihoods(cls, positions, likelihoods, observed, kept, identity: str) -> "AssemblyEvidence":
        """Build the view from founder-view arrays: positions int64[L], likelihoods float64[N, L, 3]
        (normalised), observed bool[N, L], kept bool[L] (discovery sites)."""
        likelihoods = np.ascontiguousarray(likelihoods, dtype=np.float64)
        samples, sites, _ = likelihoods.shape
        kept = np.ascontiguousarray(kept, dtype=np.bool_)
        segments = []
        try:
            for shape, dtype in (((samples, sites, 3), np.float64), ((samples, sites, 3), np.float32),
                                 ((samples, sites), np.bool_)):
                segments.append(empty_shared_array(shape, dtype))
            (_, neutral_ref, neutral), (_, scoring_ref, scoring), (_, observed_ref, seen) = segments
            _fill_views(likelihoods, np.ascontiguousarray(observed, dtype=np.bool_), kept,
                        neutral, scoring, seen)
        except BaseException:
            close_shared_memory([segment[0] for segment in segments], unlink=True)
            raise
        for array in (neutral, scoring, seen):
            array.setflags(write=False)
        view = cls(positions=np.asarray(positions), neutral=neutral, scoring=scoring, observed=seen,
                   kept=kept, identity=str(identity),
                   shared=EvidenceRefs(neutral=neutral_ref, scoring=scoring_ref, observed=observed_ref))
        # The segments live as long as the view: unlinked (and unmapped once no array uses them)
        # when the view is collected; the resource tracker unlinks them if the process dies first.
        weakref.finalize(view, close_shared_memory, [segment[0] for segment in segments], unlink=True)
        return view

    @classmethod
    def from_contig(cls, evidence, identity: str) -> "AssemblyEvidence":
        """Build the view from a ``model.evidence.ContigEvidence`` (its founder view is taken here)."""
        founder = evidence.founder_view()
        return cls.from_likelihoods(founder.positions, founder.likelihoods, founder.observed,
                                    founder.discovery_sites, identity)

    def ensure_shared(self) -> "AssemblyEvidence":
        """This view when its arrays live in shared memory, else a shared copy with identical values
        (a view rebuilt from pickled arrays or constructed directly)."""
        if self.shared is not None:
            return self
        return AssemblyEvidence.from_likelihoods(self.positions, self.neutral, self.observed, self.kept,
                                                 self.identity)

    def __reduce__(self):
        # A pickled view carries its arrays (copied), not the shared segments it was built in.
        return (AssemblyEvidence, (self.positions, np.array(self.neutral), np.array(self.scoring),
                                   np.array(self.observed), self.kept, self.identity))
