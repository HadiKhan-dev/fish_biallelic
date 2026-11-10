"""Apply the canonical sample permutation once at the observation boundary (EC03).

Owns source construction and ordering; science never sees input-header order or
reads the convention environment. VCF payloads are permuted with their counts;
simulation keeps its realised-seed identity and never constructs a VCF payload.
Method: docs/methods/evidence.md#sample-order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from ...contracts.source import ContigReads
from ...core.digest import json_digest
from ...core.errors import InvariantError, SettingsError
from ...core.ties import run_sample_order
from ...store.protocol import ContigPayload, payload_identity
from .vcf import VcfSource, VerifiedVcfView

if TYPE_CHECKING:
    from ..configure import SettingsView
    from .simulation import SimulationSource

__all__ = ["CanonicalOrderSource", "canonical_order_source", "observation_source"]
SAMPLE_ORDER_RULE = "sample-name-v1"


class InputSettings(Protocol):
    """The resolved source selectors supplied by the settings layer."""

    @property
    def kind(self) -> str: ...

    @property
    def vcf(self) -> Path | None: ...

    @property
    def contigs(self) -> tuple[str, ...]: ...


class SourceSettings(Protocol):
    """The source factory consumes only the declared input settings."""

    @property
    def input(self) -> InputSettings: ...


class CanonicalOrderSource:
    """An observation source in explicit ascending or reversed sample-name order.

    Lazy resolution permits simulations to materialise their sample identifiers in
    init. Repeated wrapping is idempotent; a conflicting convention is rejected.
    """

    def __init__(self, inner: VcfSource | SimulationSource, *, reverse: bool = False) -> None:
        self.inner = inner
        self.reverse = reverse
        self._input: tuple[str, ...] | None = None
        self._order: np.ndarray | None = None

    @property
    def kind(self) -> str:
        """Source kind for the stage's payload-binding branch."""
        return self.inner.kind

    @property
    def path(self) -> Path:
        """VCF source path, used only by the stage digest cache."""
        if not isinstance(self.inner, VcfSource):
            raise SettingsError("only VCF sources have an input path")
        return self.inner.path

    def _resolve(self) -> tuple[tuple[str, ...], np.ndarray]:
        if self._order is None:
            self._input = tuple(map(str, self.inner.sample_ids()))
            self._order = run_sample_order(self._input, reverse=self.reverse)
        assert self._input is not None
        return self._input, self._order

    def input_sample_ids(self) -> tuple[str, ...]:
        """Original header order, for export restoration and run provenance."""
        return self._resolve()[0]

    def order(self) -> np.ndarray:
        """int64[N] mapping canonical rows to input rows; caller owns the copy."""
        return self._resolve()[1].copy()

    def sample_ids(self) -> tuple[str, ...]:
        """Canonical sample names in the order of every returned observation array."""
        ids, order = self._resolve()
        return tuple(ids[index] for index in order)

    def contigs(self) -> tuple[str, ...]:
        """Forward the source's declared contig order."""
        return self.inner.contigs()

    def identity(self) -> str:
        """Bind the canonical convention while preserving the underlying source identity."""
        return json_digest(
            dict(source=self.inner.identity(), sample_order=SAMPLE_ORDER_RULE, sample_ids=self.sample_ids())
        )

    def _reads(self, reads: ContigReads) -> ContigReads:
        ids, order = self._resolve()
        if reads.sample_ids != ids:
            raise InvariantError(f"{reads.contig}: the reads' sample axis differs from the source's sample ids")
        if np.array_equal(order, np.arange(len(order))):
            return reads
        return replace(reads, sample_ids=self.sample_ids(), counts=reads.counts[order])

    def reads(self, contig: str, team: int = 1) -> ContigReads:
        """Return canonical int32[N,L,2] counts; input arrays remain unchanged."""
        reads = self.inner.reads(contig, team) if isinstance(self.inner, VcfSource) else self.inner.reads(contig)
        return self._reads(reads)

    def read_with_payload(self, contig: str, team: int = 1) -> tuple[ContigReads, ContigPayload]:
        """Permute the VCF reads and payload together, sharing the final counts buffer."""
        if not isinstance(self.inner, VcfSource):
            raise SettingsError("only VCF sources have decoded input payloads")
        reads, payload = self.inner.read_with_payload(contig, team)
        canonical = self._reads(reads)
        _, order = self._resolve()
        gt = payload.gt_types
        if canonical is not reads and gt is not None:
            gt = gt[order]
            gt.flags.writeable = False
        return canonical, replace(payload, sample_ids=canonical.sample_ids, counts=canonical.counts, gt_types=gt)

    def payload_identity(self, contig: str) -> str:
        """Canonical consumed VCF fields, or the simulation's resolved design and seed."""
        if self.kind == "vcf":
            return payload_identity(self.read_with_payload(contig)[1])
        return self.inner.payload_identity(contig)

    def verified_view(
        self, contig: str, expected_identity: str, *, team: int = 1
    ) -> VerifiedVcfView | CanonicalOrderSource:
        """Re-read VCF once for export and verify against evidence's consumed digest.

        The returned view shares a fresh immutable payload across both export calls.
        Simulation verifies its bound design/seed identity and returns its own view.
        """
        if self.kind == "vcf":
            return VerifiedVcfView(self.read_with_payload(contig, team)[1], expected_identity)
        if self.payload_identity(contig) != expected_identity:
            raise InvariantError(f"{contig}: simulation payload identity differs from evidence")
        return self

    def site_alleles(self, contig: str, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, int | None]:
        """Source allele labels and header length at the requested input positions."""
        return self.inner.site_alleles(contig, positions)

    def allele_totals(self, contig: str, positions: np.ndarray, samples: Sequence[str]) -> tuple[np.ndarray, ...]:
        """Source totals over sample identifiers; their requested order is immaterial."""
        return self.inner.allele_totals(contig, positions, samples)

    def check(self) -> None:
        """Run the VCF's fail-fast header checks; simulation validates in init."""
        if isinstance(self.inner, VcfSource):
            self.inner.check()

    def materialise(self, *args: Any, **kwargs: Any) -> Any:
        """Forward simulation init and leave the canonical axis unresolved until ready."""
        if not isinstance(self.inner, VcfSource):
            return self.inner.materialise(*args, **kwargs)
        return None


def canonical_order_source(
    source: VcfSource | SimulationSource | CanonicalOrderSource, *, reverse: bool = False
) -> CanonicalOrderSource:
    """Wrap once, refusing a second conflicting convention instead of permuting twice."""
    if isinstance(source, CanonicalOrderSource):
        if source.reverse != reverse:
            raise SettingsError("canonical sample-order convention differs from the existing source")
        return source
    return CanonicalOrderSource(source, reverse=reverse)


def observation_source(
    settings: SourceSettings, *, reverse: bool = False, simulation_parameters: Any = None
) -> CanonicalOrderSource:
    """Construct a canonical source from resolved settings; simulation parameters are stage-resolved.

    The simulation argument is the typed SimulationConfig owned by template generation, kept out
    of this module's imports so importing a VCF reader does not initialise simulation.
    """
    if settings.input.kind == "vcf":
        if settings.input.vcf is None:
            raise SettingsError("input.vcf is required for a VCF source")
        source: VcfSource | SimulationSource = VcfSource(settings.input.vcf, settings.input.contigs)
    else:
        from .simulation import SimulationSource

        if simulation_parameters is None:
            from ..configure import simulation

            simulation_parameters = simulation(cast("SettingsView", settings))
        source = SimulationSource(simulation_parameters)
    return canonical_order_source(source, reverse=reverse)
