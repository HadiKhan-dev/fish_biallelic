"""Read partial or finished runs through schema- and derivation-checked stage adapters.

Owns the public Results object. Adapters recompute their expected derivation from
the bound manifest and recorded upstream products; the reader checks it with the
store protocol before invoking a decoder. There is no science or original-input
I/O here (WP_I1 section 4.5; WP_OU1 section 6.1; WP_EVA1 section 4.5).
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from ..contracts.roles import require_schema
from ..core.digest import json_digest
from ..core.errors import InputError, InvariantError, SettingsError
from ..store.container import CheckpointMismatch
from ..store.protocol import read_items, resume
from ..store.store import CheckpointStore
from .order import recorded_input_order
from .panels import FOUNDER_STAGES, StagePanel
from .paths import RunPaths

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ..contracts.assembly import FounderRelease
    from ..contracts.evidence import EvidenceDecisions
    from ..contracts.painting import Painting
    from ..contracts.panels import ComponentPanels
    from ..contracts.pedigree import Pedigree
    from ..contracts.phase import PhasedSamples, PhaseSummary
    from ..contracts.recombination import Recombination
    from ..contracts.relatedness import RelatednessFamilies, RelatednessPairs
    from ..contracts.release import ReleasedSites
    from ..core.ties import TieCounts
    from ..stages.inputs.simulation_records import ContigTruth, SimulationManifest

__all__ = ["Results", "ProductReader", "RelatednessView"]

Product = TypeVar("Product", covariant=True)


class ProductReader(Protocol[Product]):
    """A stage-supplied read request; constructing it verifies manifest dependencies.

    The expected identity must be computed from the bound configuration and inputs,
    never copied from the product's own publication envelope. Stage adapters may
    implement this structural protocol with their own ordinary frozen dataclasses.
    """

    @property
    def key(self) -> str: ...

    @property
    def identity(self) -> str: ...

    @property
    def schema(self) -> int: ...

    def decode(self, payload: dict[str, Any]) -> Product: ...


@dataclass(frozen=True)
class PanelReader:
    """Project an owner's decoded product onto original panels without changing axes or calls."""

    request: ProductReader[Any]
    stage: str

    @property
    def key(self) -> str:
        return self.request.key

    @property
    def identity(self) -> str:
        return self.request.identity

    @property
    def schema(self) -> int:
        return self.request.schema

    @property
    def items(self) -> tuple[str, ...] | None:
        return getattr(self.request, "items", None)

    def decode(self, payload: dict[str, Any]) -> tuple[Any, ...]:
        product = self.request.decode(payload)
        if self.stage == "founder_refit":
            return tuple(product.panels)
        panels = tuple(component.panel for component in product.components)
        if any(panel is None for panel in panels):
            raise InvariantError("painting reader did not reconstruct its original founder panels")
        return panels


@dataclass(frozen=True)
class RelatednessView:
    """Family product with optional selected pair fields; retained unselected pairs stay hidden."""

    families: RelatednessFamilies
    pair_product: RelatednessPairs | None = None

    @property
    def pairs(self) -> NDArray[np.int64] | None:
        return None if self.pair_product is None else self.pair_product.pairs

    @property
    def ibd(self) -> NDArray[np.float64] | None:
        return None if self.pair_product is None else self.pair_product.ibd

    @property
    def ibd_sites(self) -> NDArray[np.float64] | None:
        return None if self.pair_product is None else self.pair_product.ibd_sites

    @property
    def lr_ibd0(self) -> NDArray[np.float64] | None:
        return None if self.pair_product is None else self.pair_product.lr_ibd0

    @property
    def lr_ibd2(self) -> NDArray[np.float64] | None:
        return None if self.pair_product is None else self.pair_product.lr_ibd2

    def __getattr__(self, name: str) -> Any:
        # Reports consume the unchanged family fields; no copying, renumbering or filtering.
        return getattr(self.families, name)


@dataclass(frozen=True)
class TieSource:
    """An explicit record group and its unchanged recorded tie diagnostics for reporting."""

    record_group: str
    counters: tuple[TieCounts, ...]
    discovery: tuple[object, ...] = ()
    refinement_agreement: tuple[bool | None, ...] = ()
    count_calls: tuple[object, ...] = ()


class Results:
    """Lazy, read-only access to an initialized run, including unfinished contigs.

    Construction reads layout and the small manifest only. Product methods return
    None for absent products; unsupported schemas and changed derivations raise.
    The original input files are never opened. Each call refreshes publications,
    including outputs/selection, so an off/on/off change is visible immediately.
    """

    def __init__(self, path: str | Path, *, checkpoints: str | Path | None = None, nthreads: Any = 1) -> None:
        paths = RunPaths.from_directory(path)
        self.root = paths.output
        self.checkpoints = paths.checkpoints if checkpoints is None else Path(checkpoints).expanduser().resolve()
        if not self.checkpoints.is_dir():
            raise SettingsError(f"{self.root}: run has not been initialized; use haplotypes init")
        self._store = CheckpointStore(self.checkpoints, nthreads=nthreads)
        self._store.check_layout()
        manifest = self._store.manifest("run")
        if manifest is None:
            raise SettingsError(f"{self.root}: run has not been initialized; use haplotypes init")
        require_schema(manifest, 1)
        self._manifest = manifest

    @property
    def manifest(self) -> dict[str, Any]:
        """Return a private copy of the bound run manifest; callers cannot mutate reader state."""
        return copy.deepcopy(self._manifest)

    @property
    def contigs(self) -> tuple[str, ...]:
        """The manifest's bound processing order, including contigs with no products yet."""
        return tuple(self._manifest.get("contigs", ()))

    @property
    def input_sample_ids(self) -> tuple[str, ...] | None:
        """Original source columns, for rendering only; scientific records keep canonical order."""
        return recorded_input_order(self._manifest)

    def _file(self, contig: str | None) -> Any:
        if contig is None:
            return self._store.run
        if contig not in self.contigs:
            raise SettingsError(f"contig {contig!r} is not in the run manifest: {', '.join(self.contigs)}")
        return self._store.contig(contig)

    def read(self, request: ProductReader[Product], contig: str | None = None) -> Product | None:
        """Check a stage adapter's expected identity and schema, then decode its immutable product."""
        items = getattr(request, "items", None)
        record = (
            resume(self._file(contig), request.key, request.identity)
            if items is None
            else read_items(self._file(contig), request.key, request.identity, tuple(dict.fromkeys(("schema", *items))))
        )
        if record is None:
            return None
        require_schema(record.payload, request.schema)
        return request.decode(record.payload)

    def _has(self, contig: str | None, *prefixes: str) -> bool:
        file = self._file(contig)
        return any(file.has(prefix) or file.keys(prefix + "/") for prefix in prefixes)

    def _read(self, adapter: Any, product: str, contig: str | None = None, **options: Any) -> Any | None:
        request = adapter.result_reader(self._store, self.manifest, product=product, contig=contig, **options)
        return None if request is None else self.read(request, contig)

    def evidence_summary(self, contig: str) -> EvidenceDecisions | None:
        """Read the evidence adapter's typed quality and calibration summary without its large arrays."""
        if not self._has(contig, "evidence"):
            return None
        from ..stages.evidence import records

        return self._read(records, "summary", contig)

    def founders(self, contig: str) -> ComponentPanels | None:
        """Read final assembly founder panels before output release transformations."""
        if not self._has(contig, "assembly"):
            return None
        from ..stages.assembly import records

        return self._read(records, "founders", contig)

    def painting(self, contig: str) -> Painting | None:
        """Read typed chromosome painting with its canonical sample and component axes."""
        if not self._has(contig, "painting"):
            return None
        from ..stages.painting import records

        return self._read(records, "painting", contig)

    def pedigree(self) -> Pedigree | None:
        """Read the genome-wide pedigree calls, edges and report views when recorded."""
        if not self._has(None, "pedigree"):
            return None
        from ..stages.pedigree import records

        return self._read(records, "pedigree")

    def phase(self, contig: str) -> PhasedSamples | None:
        """Read PhasedSamples on its recorded site and sample axes."""
        if not self._has(contig, "phase", "family_phase"):
            return None
        from ..stages.phase import records

        return self._read(records, "phase", contig)

    def recombination(self, contig: str) -> Recombination | None:
        """Read the contig's recombination product with its immutable derivation binding."""
        if not self._has(contig, "recombination"):
            return None
        from ..stages.recombination import records

        return self._read(records, "recombination", contig)

    def released_sites(self, contig: str) -> ReleasedSites | None:
        """Read the option-independent ReleasedSites product for output rendering."""
        if not self._has(contig, "sites", "released_sites"):
            return None
        from ..stages.inputs import sites_records as sites

        return self._read(sites, "released_sites", contig)

    def founder_release(self, contig: str) -> FounderRelease | None:
        """Read AS6's FounderRelease segments and identity flags, including left-partner ownership."""
        if not self._has(contig, "release", "founder_release"):
            return None
        from ..stages.assembly import records

        return self._read(records, "founder_release", contig)

    def release_timings(self, contig: str) -> dict[str, float] | None:
        """Read measured AS6 release subphase seconds through the checked release derivation."""
        if not self._has(contig, "release/timings"):
            return None
        from ..stages.assembly import records

        return self._read(records, "release_timings", contig)

    def relatedness(self) -> RelatednessView | None:
        """Apply the current output selection and expose pairs only when selected and recorded."""
        if not self._has(None, "relatedness"):
            return None
        from ..stages.relatedness import families_records

        families = self._read(families_records, "families")
        if families is None:
            return None
        selection = self.output_selection()
        enabled = selection is not None and (
            selection.get("pairwise_ibd", False) if isinstance(selection, Mapping) else selection.pairwise_ibd
        )
        pairs = None
        if enabled:
            from ..stages.relatedness import pairs_records

            pairs = self._read(pairs_records, "pairs")
        return RelatednessView(families, pairs)

    def _panel_request(self, contig: str, stage: str) -> ProductReader[Any] | None:
        """Ask the owning adapter for the exact founder stage, retaining its original axes."""
        adapter: Any
        request: ProductReader[Any] | None
        if stage not in FOUNDER_STAGES:
            raise SettingsError(f"unknown founder stage {stage!r}; valid: {', '.join(FOUNDER_STAGES)}")
        if stage == "discovery":
            if not self._has(contig, "discovery"):
                return None
            from ..stages.discovery import records as discovery_records

            adapter = discovery_records
        elif stage.startswith("local_selection_"):
            if not self._has(contig, "selection", "local_selection"):
                return None
            from ..stages.selection import records as selection_records

            adapter = selection_records
        elif stage.startswith("assembly_l") or stage == "founder_refinement":
            if not self._has(contig, "assembly"):
                return None
            from ..stages.assembly import records as assembly_records

            adapter = assembly_records
        elif stage == "painting":
            if not self._has(contig, "painting"):
                return None
            from ..stages.painting import records as painting_records

            request = painting_records.result_reader(
                self._store, self.manifest, product="painting", contig=contig, panels=True, emissions=False
            )
            return None if request is None else PanelReader(request, stage)
        else:
            if not self._has(contig, "refit", "founder_refit"):
                return None
            from ..stages.refit import records as refit_records

            request = refit_records.result_reader(
                self._store, self.manifest, product="refit", contig=contig, panels=True
            )
            return None if request is None else PanelReader(request, stage)
        return adapter.result_reader(self._store, self.manifest, product="stage_panels", contig=contig, stage=stage)

    def stage_panels(self, contig: str, stage: str) -> tuple[StagePanel, ...]:
        """Read original component/row axes before withholding, segmentation or export names."""
        request = self._panel_request(contig, stage)
        panels = None if request is None else self.read(request, contig)
        if panels is None:
            return ()
        return tuple(
            p
            if isinstance(p, StagePanel)
            else StagePanel(
                p.positions,
                p.calls if hasattr(p, "calls") else p.released,
                getattr(p, "atomic_offsets", None),
                getattr(p, "rows", None),
            )
            for p in panels
        )

    def stage_present(self, contig: str, stage: str) -> bool:
        """Check a stage through its publication reader, never through workflow markers."""
        if stage in FOUNDER_STAGES:
            request = self._panel_request(contig, stage)
            if request is None:
                return False
            record = read_items(self._file(contig), request.key, request.identity, ("schema",))
            if record is None:
                return False
            require_schema(record.payload, request.schema)
            return True
        if stage == "family_phase":
            return self.phase(contig) is not None
        if stage == "recombination":
            return self.recombination(contig) is not None
        raise SettingsError(f"unknown evaluation stage {stage!r}")

    @property
    def synthetic(self) -> bool:
        """Whether the bound observation source is explicitly simulated."""
        return self._manifest.get("source", {}).get("kind") == "simulation"

    def output_selection(self) -> Mapping[str, Any] | None:
        """Refresh the latest output-only choice; it is never a scientific binding."""
        from ..store.protocol import read_output

        record = read_output(self._store.run, "outputs/selection", identity=json_digest)
        if record is None:
            return None
        require_schema(record.payload, 1)
        if record.identity != json_digest(record.payload):
            raise CheckpointMismatch("outputs/selection content differs from its selection identity")
        return record.payload

    def simulation_manifest(self) -> SimulationManifest | None:
        """Read SI1's seed/truth metadata and check it against init's source binding."""
        if not self.synthetic:
            return None
        from ..stages.inputs import simulation_records as simulation

        saved = simulation.load_manifest(self._store)
        if saved is None:
            return None
        source = self._manifest["source"]
        if (
            saved.identity != source["input_identity"]
            or saved.seed != self._manifest["realised_seeds"]["simulation"]
            or tuple(saved.contigs) != self.contigs
        ):
            raise CheckpointMismatch("simulation manifest differs from the initialized source; use a new output root")
        return saved

    def simulation_truth(self, contig: str) -> ContigTruth:
        """Decode only the requested truth contig under its initialized source identity."""
        file = self._file(contig)
        saved = self.simulation_manifest()
        if saved is None:
            raise InputError(f"{self.root}: no simulation manifest")
        from ..stages.inputs import simulation_records as simulation

        identity = json_digest(dict(source=saved.identity, contig=contig))
        return simulation.load_truth(file, identity)

    def painting_report(self, contig: str) -> Any | None:
        """Expose recorded painting and ordered search summaries for OU2 formatting."""
        if not self._has(contig, "painting"):
            return None
        from ..stages.painting import records

        return self._read(records, "painting_report", contig)

    def tie_sources(self, contig: str) -> tuple[Any, ...]:
        """Collect only explicitly declared stage tie views in the baseline report order."""
        sources: list[Any] = []
        if self._has(contig, "evidence"):
            from ..stages.evidence import records as evidence_records

            diagnostics = self._read(evidence_records, "diagnostics", contig)
            if diagnostics is not None:
                sources.append(TieSource("evidence", (diagnostics.tie_events,)))
        if self._has(contig, "discovery"):
            from ..stages.discovery import records as discovery_records

            sources.extend(self._read(discovery_records, "tie_sources", contig) or ())
        if self._has(contig, "selection", "local_selection"):
            from ..stages.selection import records as selection_records

            sources.extend(self._read(selection_records, "tie_sources", contig) or ())
        if self._has(contig, "assembly"):
            from ..stages.assembly import records as assembly_records

            sources.extend(self._read(assembly_records, "tie_sources", contig) or ())
        if self._has(contig, "painting"):
            from ..stages.painting import records as painting_records

            sources.extend(self._read(painting_records, "tie_sources", contig) or ())
        if self._has(contig, "refit", "founder_refit"):
            from ..stages.refit import records as refit_records

            sources.extend(self._read(refit_records, "tie_sources", contig) or ())
        return tuple(sources)

    def pedigree_ties(self) -> Any | None:
        """Return the pedigree's recorded tie summary without recomputing its decisions."""
        if not self._has(None, "pedigree"):
            return None
        from ..stages.pedigree import records

        return self._read(records, "pedigree_ties")

    def refit_summaries(self) -> tuple[Any, ...]:
        """Read available contig summaries in the bound contig order."""
        from ..stages.refit import records

        return tuple(
            row
            for c in self.contigs
            if self._has(c, "refit", "founder_refit")
            if (row := self._read(records, "summary", c)) is not None
        )

    def phase_summaries(self, *, elapsed_seconds: Mapping[str, float] | None = None) -> tuple[PhaseSummary, ...]:
        """Read available phase summaries without inferring any absent report field."""
        from ..stages.phase import records

        return tuple(
            row
            for c in self.contigs
            if self._has(c, "phase", "family_phase")
            if (row := self._read(records, "phase_summary", c, elapsed_seconds=(elapsed_seconds or {}).get(c)))
            is not None
        )

    def record_bytes(self, contig: str, product: str) -> int:
        """Return checked decoded bytes for EVA1's worker estimate.

        The target read is header-only. Owner request construction is responsible
        for checking its upstream identity without materialising large products.
        """
        file = self._file(contig)
        adapter: Any
        request: ProductReader[Any] | None
        if product == "truth":
            from ..stages.inputs import simulation_records as simulation

            saved = self.simulation_manifest()
            if saved is None:
                raise InputError(f"{contig}: simulation truth is unavailable")
            key = simulation.TRUTH
            identity = json_digest(dict(source=saved.identity, contig=contig))
            record = read_items(file, key, identity, ())
            return 0 if record is None else record.info.decoded_bytes
        if product in FOUNDER_STAGES:
            request = self._panel_request(contig, product)
        else:
            if product == "family_phase":
                from ..stages.phase import records as phase_records

                adapter = phase_records
                name = "phase"
            elif product == "recombination":
                from ..stages.recombination import records as recombination_records

                adapter = recombination_records
                name = "recombination"
            else:
                raise SettingsError(f"unknown evaluation product {product!r}")
            request = adapter.result_reader(self._store, self.manifest, product=name, contig=contig)
        if request is None:
            return 0
        record = read_items(file, request.key, request.identity, ("schema",))
        if record is None:
            return 0
        require_schema(record.payload, request.schema)
        return record.info.decoded_bytes

    def product_identity(self, product: str, contig: str | None = None) -> str | None:
        """Return a manifest-derived publication identity after checking its schema/header.

        This exposes scientific derivation identities for output-task dependencies,
        never physical record tokens. Missing products return None.
        """
        adapter: Any
        if contig is None:
            raise SettingsError(f"{product}: a contig is required")
        if product == "released_sites":
            if not self._has(contig, "sites", "released_sites"):
                return None
            from ..stages.inputs import sites_records as sites

            adapter = sites
        elif product == "founder_release":
            if not self._has(contig, "release", "founder_release"):
                return None
            from ..stages.assembly import records as assembly_records

            adapter = assembly_records
        elif product == "phase":
            if not self._has(contig, "phase", "family_phase"):
                return None
            from ..stages.phase import records as phase_records

            adapter = phase_records
        else:
            raise SettingsError(f"unsupported output dependency product {product!r}")
        request = adapter.result_reader(self._store, self.manifest, product=product, contig=contig)
        if request is None:
            return None
        record = read_items(self._file(contig), request.key, request.identity, ("schema",))
        if record is None:
            return None
        require_schema(record.payload, request.schema)
        return request.identity
