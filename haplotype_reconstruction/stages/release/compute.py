"""Publish option-free founder release after assembly, painting, refit and evidence.

Owns checked input alignment, painting carrier views and durable FounderRelease output.
Science computes withholding, segmentation and partner flags; outputs only formats
this product. Method: docs/methods/export.md (founder release).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from haplotype_reconstruction.contracts.assembly import AssemblyProduct, FounderRelease, ReleaseAxes
from haplotype_reconstruction.contracts.evidence import ContigEvidence
from haplotype_reconstruction.contracts.observations import dense_observed
from haplotype_reconstruction.contracts.painting import Painting
from haplotype_reconstruction.contracts.payload import from_payload, to_payload
from haplotype_reconstruction.contracts.refit import FounderRefit
from haplotype_reconstruction.core.errors import InvariantError
from haplotype_reconstruction.engine.cpu import thread_team
from haplotype_reconstruction.science.assembly.release import budget_overflow, release_founders, withhold_unobserved
from haplotype_reconstruction.store.container import CheckpointFile
from haplotype_reconstruction.store.protocol import claim, publish, resume

from .records import RELEASE


@dataclass(frozen=True)
class ReleaseInputs:
    """Identity-checked upstream products with refit/painting panel views reconstructed."""

    assembly: AssemblyProduct
    painting: Painting
    refit: FounderRefit
    evidence: ContigEvidence


def compute_release(inputs: ReleaseInputs, *, cores: int = 1) -> FounderRelease:
    """Align painting carrier observation before applying either release mask."""
    panels = inputs.refit.panels
    painting = inputs.painting.components
    if len(panels) != len(painting):
        raise InvariantError("founder panels and painting components differ")
    sites = np.asarray(inputs.evidence.positions)
    observed = dense_observed(inputs.evidence.observed)
    original, seen = [], []
    with thread_team(default=cores):
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
    masked, _counts, masks = withhold_unobserved(original, seen)
    alternatives = inputs.assembly.release_alternatives
    need_alleles = any(
        budget_overflow(f) or (not f.get("unresolved_search", False) and len(f.get("panels", ())) > 1)
        for f in alternatives
    )
    prepared = tuple(p.released for p in inputs.assembly.prepared) if need_alleles else None
    axes = tuple(ReleaseAxes(p.start, p.stop, p.rows, p.atomic_offsets) for p in panels)
    paths = tuple((p.start, p.positions, p.atomic_offsets, p.rows) for p in panels)
    ambiguous = sum(int(np.sum(c.released[c.changed_rows, c.changed_sites] < 0)) for c in inputs.refit.components)
    return release_founders(
        axes,
        masked,
        alternatives,
        unobserved_masks=masks,
        nominal_components=inputs.assembly.components,
        prepared_alleles=prepared,
        identity_paths=paths,
        refit_ambiguous_alleles=ambiguous,
    )


def execute_release(inputs: ReleaseInputs, file: CheckpointFile, identity: str, *, cores: int = 1) -> FounderRelease:
    """Publish release once, checking its derivation before every resumed read."""
    with claim(file, RELEASE, identity) as needed:
        if not needed:
            stored = resume(file, RELEASE, identity)
            assert stored is not None  # claim() yields False only once the record is complete
            return from_payload(FounderRelease, stored.payload)
        result = compute_release(inputs, cores=cores)
        publish(file, RELEASE, identity, to_payload(result))
        return result
