"""Publish option-free founder release after assembly, painting, refit and evidence.

Owns checked input alignment, PA2 carrier views and durable FounderRelease output.
Science computes withholding, segmentation and partner flags; outputs only formats
this product. Method: WP_AS6 §4.7; WP_OU1 OU03/OU06.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from haplotype_reconstruction.contracts.assembly import AssemblyProduct, FounderRelease, ReleaseAxes
from haplotype_reconstruction.contracts.evidence import ContigEvidence
from haplotype_reconstruction.contracts.painting import Painting
from haplotype_reconstruction.contracts.payload import from_payload, to_payload
from haplotype_reconstruction.contracts.refit import FounderRefit
from haplotype_reconstruction.core.errors import InvariantError
from haplotype_reconstruction.engine.graph import RecordKey, ResourceRequest, Task, TaskKey
from haplotype_reconstruction.engine.runtime import task_team
from haplotype_reconstruction.science.assembly.release import budget_overflow, release_founders, withhold_unobserved
from haplotype_reconstruction.store.container import CheckpointFile
from haplotype_reconstruction.store.protocol import NumericalPolicy, claim, publish, resume

from .records import operation_derivation

RELEASE = "release"
RELEASE_TIMINGS = "release/timings"


@dataclass(frozen=True)
class ReleaseInputs:
    """Identity-checked upstream products with refit/painting panel views reconstructed."""

    assembly: AssemblyProduct
    painting: Painting
    refit: FounderRefit
    evidence: ContigEvidence


def release_identity(
    upstream: Mapping[str, str], *, reverse: bool = False, numerical_policy: NumericalPolicy | None = None
) -> str:
    """Bind all four upstream derivations and the common evidence payload identity."""
    required = {"assembly", "painting", "refit", "evidence", "payload"}
    if set(upstream) != required:
        raise InvariantError(f"release identity needs {sorted(required)}")
    return operation_derivation(
        {},
        upstream,
        reverse=reverse,
        roots=("haplotype_reconstruction.stages.assembly.release",),
        numerical_policy=numerical_policy,
    ).identity


def compute_release(
    inputs: ReleaseInputs, *, cores: int = 1, phase_seconds: MutableMapping[str, float] | None = None
) -> FounderRelease:
    """Align PA2 carrier observation before applying either release mask."""
    started = perf_counter() if phase_seconds is not None else 0.0
    panels = inputs.refit.panels
    painting = inputs.painting.components
    if len(panels) != len(painting):
        raise InvariantError("founder panels and painting components differ")
    sites = np.asarray(inputs.evidence.positions)
    observed = np.asarray(inputs.evidence.observed, dtype=bool)
    original, seen = [], []
    with task_team(cores):
        for component, panel in zip(painting, panels, strict=False):
            positions = np.asarray(panel.positions, dtype=np.int64)
            calls = np.asarray(panel.released, dtype=np.int8)
            columns = np.searchsorted(sites, positions)
            if np.any(columns >= len(sites)) or not np.array_equal(
                sites[np.minimum(columns, len(sites) - 1)], positions
            ):
                raise InvariantError("founder panel sites are missing from the evidence record")
            original.append((positions, calls))
            seen.append(component.carrier_observed(positions, columns, observed, len(calls)))
    if phase_seconds is not None:
        phase_seconds["carrier_observation"] = perf_counter() - started
        started = perf_counter()
    masked, _counts, masks = withhold_unobserved(original, seen)
    if phase_seconds is not None:
        phase_seconds["carrier_withholding"] = perf_counter() - started
    alternatives = inputs.assembly.release_alternatives
    need_alleles = any(
        budget_overflow(f) or (not f.get("unresolved_search", False) and len(f.get("panels", ())) > 1)
        for f in alternatives
    )
    prepared = tuple(p.released for p in inputs.assembly.prepared) if need_alleles else None
    axes = tuple(ReleaseAxes(p.start, p.stop, p.rows, p.atomic_offsets) for p in panels)
    paths = tuple((p.start, p.positions, p.atomic_offsets, p.rows) for p in panels)
    ambiguous = sum(int(np.sum(c.released[c.changed_rows, c.changed_sites] < 0)) for c in inputs.refit.components)
    started = perf_counter() if phase_seconds is not None else 0.0
    result = release_founders(
        axes,
        masked,
        alternatives,
        unobserved_masks=masks,
        nominal_components=inputs.assembly.components,
        prepared_alleles=prepared,
        identity_paths=paths,
        refit_ambiguous_alleles=ambiguous,
    )
    if phase_seconds is not None:
        # The native release phase times this call; OU06 flag construction now
        # belongs to this call as well. Preparation and serialization are separate.
        phase_seconds["release"] = perf_counter() - started
    return result


def execute_release(inputs: ReleaseInputs, file: CheckpointFile, identity: str, *, cores: int = 1) -> FounderRelease:
    """Publish release once, checking its derivation before every resumed read."""
    with claim(file, RELEASE, identity) as needed:
        if not needed:
            stored = resume(file, RELEASE, identity)
            assert stored is not None  # claim() yields False only once the record is complete
            return from_payload(FounderRelease, stored.payload)
        phase_seconds: dict[str, float] = {}
        result = compute_release(inputs, cores=cores, phase_seconds=phase_seconds)
        # Publish measured diagnostics first: a visible release always has its
        # measurements. Reuse keeps the first completed computation's durations.
        publish(file, RELEASE_TIMINGS, identity, dict(schema=1, phase_seconds=phase_seconds))
        publish(file, RELEASE, identity, to_payload(result))
        return result


def release_task(
    contig: str,
    inputs: ReleaseInputs,
    file: CheckpointFile,
    identity: str,
    *,
    depends_on: Sequence[TaskKey] = (),
    upstream_records: Sequence[RecordKey] = (),
    max_cores: int = 1,
) -> Task:
    """Declare release[c], downstream of refit, painting, assembly and evidence."""
    return Task(
        TaskKey("release", "contig", contig),
        execute_release,
        args=(inputs, file, identity),
        depends_on=tuple(depends_on),
        inputs=tuple(upstream_records),
        outputs=(RecordKey(contig, RELEASE), RecordKey(contig, RELEASE_TIMINGS)),
        resources=ResourceRequest(1, max_cores),
        affinity=contig,
        thread_argument="cores",
    )
