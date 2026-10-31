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

from dataclasses import dataclass, field, replace
import hashlib
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


@njit(parallel=True, cache=True)
def _fill_neutral(likelihoods, observed, kept, neutral, seen):
    """_fill_views without the float32 copy: the neutral view and observed & kept."""
    samples, sites, _ = likelihoods.shape
    for sample in prange(samples):
        for site in range(sites):
            used = observed[sample, site] and kept[site]
            seen[sample, site] = used
            for genotype in range(3):
                neutral[sample, site, genotype] = likelihoods[sample, site, genotype] if used else 1.0 / 3.0


@njit(parallel=True, cache=True)
def _fill_scoring(neutral, scoring):
    """The float32 copy of the neutral view (the same cast _fill_views makes of each value)."""
    samples, sites, _ = neutral.shape
    for sample in prange(samples):
        for site in range(sites):
            for genotype in range(3):
                scoring[sample, site, genotype] = neutral[sample, site, genotype]


def _depth_keys(counts, observed, kept):
    """Unseeded SHA256 of each founder-visible int64 total-depth vector; one row of scratch.

    Non-discovery and unobserved cells are zero. Canonical little-endian bytes make the exact
    integer content independent of machine byte order. Neither likelihoods nor sample labels enter.
    """
    keys = []
    for sample in range(observed.shape[0]):
        depth = counts[sample].sum(axis=1, dtype=np.int64).astype("<i8", copy=False)
        depth[~observed[sample]] = 0
        depth[~kept] = 0
        keys.append(hashlib.sha256(depth).digest())
    return tuple(keys)


@dataclass(frozen=True)
class EvidenceRefs:
    """Shared-memory references of a view's neutral (float64[N, L, 3]), scoring (float32[N, L, 3])
    and observed (bool[N, L]) arrays, for pool workers (``execution.shared_arrays.attach_shared_array``)."""

    neutral: SharedArrayRef
    scoring: SharedArrayRef
    observed: SharedArrayRef


@dataclass(frozen=True)
class SelectionEvidence:
    """Local selection's founder evidence, read from the assembly view (release 3.4).

    likelihoods: the view's neutral float64[N, L, 3]; observed: the view's bool[N, L] (observed at a discovery
    site, excluded samples unobserved); positions: int64[L]. likelihoods_ref / observed_ref: their shared-memory
    references (pools attach them instead of copying), or None. Local selection reads discovery-site columns only
    and treats unobserved cells as uniform (block fits set them to 1/3; the context refit and its site posteriors
    skip them), so it computes exactly what it computed from ``ContigEvidence.founder_view()``, whose likelihoods
    equal the view's at every observed discovery-site cell. Holding the view alone drops the decoded float64
    likelihoods and the pools' copies of them (about 50 bytes per sample-site) from the local-selection phase.
    """

    positions: np.ndarray
    likelihoods: np.ndarray
    observed: np.ndarray
    likelihoods_ref: SharedArrayRef | None = None
    observed_ref: SharedArrayRef | None = None

    def pool_arrays(self, observed=None) -> dict:
        """The ``shared`` arrays of a pool reading this evidence: shared-memory references where they exist
        (attached, not copied). ``observed``: a caller's own mask in place of the view's."""
        return dict(likelihoods=self.likelihoods if self.likelihoods_ref is None else self.likelihoods_ref,
                    observed=(observed if observed is not None
                              else self.observed if self.observed_ref is None else self.observed_ref))

    def pool_copy_bytes(self, observed=None) -> int:
        """Bytes a pool copies into new shared memory for pool_arrays(observed)."""
        likelihoods = self.likelihoods.nbytes if self.likelihoods_ref is None else 0
        mask = (observed.nbytes if observed is not None
                else self.observed.nbytes if self.observed_ref is None else 0)
        return int(likelihoods + mask)


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
    depth_keys: optional unseeded SHA256 keys of exact founder-visible total read-depth rows,
        for completion folds. Only these small keys are retained, never the read-count array.
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
    depth_keys: tuple[bytes, ...] | None = field(default=None, repr=False)

    @property
    def samples(self) -> int:
        return int(self.scoring.shape[0])

    @property
    def nbytes(self) -> int:
        """Bytes of the view arrays and retained depth-key payloads."""
        keys = 0 if self.depth_keys is None else sum(map(len, self.depth_keys))
        neutral = 0 if self.neutral is None else self.neutral.nbytes
        return int(neutral + self.scoring.nbytes + self.observed.nbytes + keys)

    def founder_evidence(self) -> SelectionEvidence:
        """This view as local selection's founder evidence (SelectionEvidence; before ``release_neutral``)."""
        if self.neutral is None:
            raise ValueError("the founder view's neutral evidence was released by a final assembly")
        return SelectionEvidence(positions=self.positions, likelihoods=self.neutral, observed=self.observed,
                                 likelihoods_ref=None if self.shared is None else self.shared.neutral,
                                 observed_ref=None if self.shared is None else self.shared.observed)

    def release_neutral(self) -> None:
        """Free the float64 founder view (``neutral`` becomes None) once nothing reads it again: the final
        assembly after its last level's join (completion and the joins read it; founder refinement reads
        ``scoring`` only). Its shared-memory segment is closed and unlinked, so its 24 N L bytes leave before
        the last level's refinement, the assembly's memory peak (release 3.4; memory only, no value changes). The
        view cannot serve another assembly afterwards: build a new one."""
        if self.neutral is None:
            return
        object.__setattr__(self, "neutral", None)
        if self.shared is not None:
            object.__setattr__(self, "shared", replace(self.shared, neutral=None))
        segments = getattr(self, "_segments", None)
        if segments:
            close_shared_memory([segments[0]], unlink=True)

    @classmethod
    def from_likelihoods(cls, positions, likelihoods, observed, kept, identity: str, *,
                         counts=None, depth_keys=None) -> "AssemblyEvidence":
        """Build the view from founder-view arrays: positions int64[L], likelihoods float64[N, L, 3]
        (normalised), observed bool[N, L], kept bool[L] (discovery sites). Optional counts are
        int32[N, L, 2] REF/ALT reads; only exact total-depth keys survive construction. Supplied
        depth_keys preserve an existing view when its shared arrays must be rebuilt."""
        likelihoods = np.ascontiguousarray(likelihoods, dtype=np.float64)
        samples, sites, _ = likelihoods.shape
        kept = np.ascontiguousarray(kept, dtype=np.bool_)
        observed = np.ascontiguousarray(observed, dtype=np.bool_)
        if depth_keys is None and counts is not None:
            depth_keys = _depth_keys(counts, observed, kept)
        segments = []
        try:
            for shape, dtype in (((samples, sites, 3), np.float64), ((samples, sites, 3), np.float32),
                                 ((samples, sites), np.bool_)):
                segments.append(empty_shared_array(shape, dtype))
            (_, neutral_ref, neutral), (_, scoring_ref, scoring), (_, observed_ref, seen) = segments
            _fill_views(likelihoods, observed, kept,
                        neutral, scoring, seen)
        except BaseException:
            close_shared_memory([segment[0] for segment in segments], unlink=True)
            raise
        for array in (neutral, scoring, seen):
            array.setflags(write=False)
        view = cls(positions=np.asarray(positions), neutral=neutral, scoring=scoring, observed=seen,
                   kept=kept, identity=str(identity), depth_keys=depth_keys,
                   shared=EvidenceRefs(neutral=neutral_ref, scoring=scoring_ref, observed=observed_ref))
        # The segments live as long as the view: unlinked (and unmapped once no array uses them)
        # when the view is collected; the resource tracker unlinks them if the process dies first.
        weakref.finalize(view, close_shared_memory, [segment[0] for segment in segments], unlink=True)
        object.__setattr__(view, "_segments", tuple(segment[0] for segment in segments))
        return view

    @classmethod
    def from_contig(cls, evidence, identity: str, *, counts=None) -> "AssemblyEvidence":
        """Build the founder view, optionally using separately decoded aligned REF/ALT counts."""
        counts = evidence.counts if counts is None else counts
        # Founder exclusion masks the keys via observed. Do not copy a chromosome's counts
        # merely to zero excluded rows: the hashing loop needs only one depth row at a time.
        source = replace(evidence, counts=None) if evidence.counts is not None else evidence
        founder = source.founder_view()
        return cls.from_likelihoods(founder.positions, founder.likelihoods, founder.observed,
                                    founder.discovery_sites, identity, counts=counts)

    @classmethod
    def from_record(cls, file, evidence=None) -> "AssemblyEvidence":
        """Build a production view with transient count decoding and memory admission.

        A caller may reuse decoded evidence. Counts absent from it are loaded only while the
        depth keys are built, then freed before context or final assembly starts.

        Release 3.4: build the depth keys first and free the counts, decode likelihoods directly into the shared
        neutral buffer, mask it in place, then make its float32 scoring copy. For separately stored likelihoods
        the transient arrays take about 38 bytes per sample-site (including the source observation mask), plus
        bounded codec scratch. A caller's already decoded evidence uses the staged copy path (about 50 bytes per
        sample-site). Both paths produce from_contig's values.
        """
        from ..evidence import records
        from ..execution.memory import reserve_memory
        identity = file.info(records.EVIDENCE).identity
        if evidence is None:
            arrays = records.load_arrays(file, ("positions", "observed", "discovery_sites", "founder_excluded"))
            observed = arrays["observed"]
            if arrays["founder_excluded"].any():
                observed = observed.copy()
                observed[arrays["founder_excluded"]] = False
            kept = arrays["discovery_sites"]
            positions = arrays["positions"]
            samples, sites = observed.shape
            del arrays
            with reserve_memory(8 * samples * sites + 9 * sites + 32 * samples,
                                label="assembly depth keys"):
                counts = records.load_arrays(file, ("counts",))["counts"]
                depth_keys = _depth_keys(counts, observed, kept)
                del counts
            return cls._staged(positions, None, observed, kept, identity, depth_keys, file=file)
        samples, sites = evidence.n_samples, evidence.n_sites
        decoded_bytes = 8 * samples * sites if evidence.counts is None else 0
        # Founder exclusion masks the keys via observed (as from_contig).
        founder = (replace(evidence, counts=None) if evidence.counts is not None else evidence).founder_view()
        kept = np.ascontiguousarray(founder.discovery_sites, dtype=np.bool_)
        observed = np.ascontiguousarray(founder.observed, dtype=np.bool_)
        with reserve_memory(decoded_bytes + 9 * sites + 32 * samples,
                            label="assembly depth keys"):
            counts = (records.load_arrays(file, ("counts",))["counts"]
                      if evidence.counts is None else evidence.counts)
            depth_keys = _depth_keys(counts, observed, kept)
            del counts
        positions = np.asarray(founder.positions)
        holder = [np.ascontiguousarray(founder.likelihoods, dtype=np.float64)]
        del evidence, founder
        return cls._staged(positions, holder, observed, kept, identity, depth_keys)

    @classmethod
    def _staged(cls, positions, holder, observed, kept, identity, depth_keys, *, file=None) -> "AssemblyEvidence":
        """Decode into neutral (or copy holder's likelihoods), mask, then cast once into scoring."""
        samples, sites = observed.shape
        segments = []
        try:
            for shape, dtype in (((samples, sites, 3), np.float64), ((samples, sites), np.bool_)):
                segments.append(empty_shared_array(shape, dtype))
            (_, neutral_ref, neutral), (_, observed_ref, seen) = segments
            if holder is None:
                from ..evidence import records
                records.load_arrays(file, ("likelihoods",), out={"likelihoods": neutral})
                likelihoods = neutral
            else:
                likelihoods = holder.pop()
            _fill_neutral(likelihoods, observed, kept, neutral, seen)
            del likelihoods
            segments.append(empty_shared_array((samples, sites, 3), np.float32))
            _, scoring_ref, scoring = segments[2]
            _fill_scoring(neutral, scoring)
        except BaseException:
            close_shared_memory([segment[0] for segment in segments], unlink=True)
            raise
        for array in (neutral, scoring, seen):
            array.setflags(write=False)
        view = cls(positions=positions, neutral=neutral, scoring=scoring, observed=seen, kept=kept,
                   identity=str(identity), depth_keys=depth_keys,
                   shared=EvidenceRefs(neutral=neutral_ref, scoring=scoring_ref, observed=observed_ref))
        handles = (segments[0][0], segments[2][0], segments[1][0])      # neutral, scoring, observed
        weakref.finalize(view, close_shared_memory, list(handles), unlink=True)
        object.__setattr__(view, "_segments", handles)
        return view

    def ensure_shared(self) -> "AssemblyEvidence":
        """This view when its arrays live in shared memory, else a shared copy with identical values
        (a view rebuilt from pickled arrays or constructed directly)."""
        if self.shared is not None:
            return self
        return AssemblyEvidence.from_likelihoods(self.positions, self.neutral, self.observed, self.kept,
                                                 self.identity, depth_keys=self.depth_keys)

    def __reduce__(self):
        # A pickled view carries its arrays (copied), not the shared segments it was built in.
        return (AssemblyEvidence, (self.positions, np.array(self.neutral), np.array(self.scoring),
                                   np.array(self.observed), self.kept, self.identity, None, self.depth_keys))
