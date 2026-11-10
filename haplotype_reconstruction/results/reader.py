"""Read partial or finished runs through schema- and derivation-checked stage adapters.

Owns the public Results object. Adapters recompute their expected derivation from
the bound manifest and recorded upstream products; the reader checks it with the
store protocol before invoking a decoder. There is no science or original-input
I/O here. Each stage owns its product decoder and expected derivation.
"""

from __future__ import annotations

import json
import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..contracts.roles import require_schema
from ..core.digest import json_digest
from ..core.errors import InputError, InvariantError, SettingsError
from ..store.container import CheckpointMismatch
from ..store.protocol import read_items
from .order import recorded_input_order
from .protocol import ProductInputs, ProductReader
from ..stages.catalog import STAGES
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


class Results(ProductInputs):
    """Lazy, read-only access to an initialized run, including unfinished contigs.

    Construction reads layout and the small manifest only. Product methods return
    None for absent products; unsupported schemas and changed derivations raise.
    The original input files are never opened. Each call refreshes publications,
    including output_selection.json, so an off/on/off change is visible immediately.
    """

    def __init__(self, path: str | Path, *, checkpoints: str | Path | None = None, nthreads: Any = 1) -> None:
        paths = RunPaths.from_directory(path)
        self.root = paths.output
        self.checkpoints = paths.checkpoints if checkpoints is None else Path(checkpoints).expanduser().resolve()
        if not self.checkpoints.is_dir():
            raise SettingsError(f"{self.root}: run has not been initialized; use haplotypes init")
        super().__init__(self.checkpoints, nthreads=nthreads)

    @property
    def input_sample_ids(self) -> tuple[str, ...] | None:
        """Original source columns, for rendering only; scientific records keep canonical order."""
        return recorded_input_order(self._manifest)


    def _has(self, contig: str | None, prefix: str) -> bool:
        file = self._file(contig)
        return bool(file.has(prefix) or file.keys(prefix + "/"))

    def _read(self, adapter: Any, product: str, contig: str | None = None, **options: Any) -> Any | None:
        request = self.request(adapter, product, contig, **options)
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
        if not self._has(contig, "phase"):
            return None
        from ..stages.phase import records

        return self._read(records, "phase", contig)

    def recombination(self, contig: str) -> Recombination | None:
        """Read the contig's recombination product with its immutable derivation binding."""
        if not self._has(contig, "recombination"):
            return None
        from ..stages.recombination import records

        return self._read(records, "recombination", contig)

    def sites(self, contig: str) -> ReleasedSites | None:
        """Read the option-independent ReleasedSites product for output rendering."""
        if not self._has(contig, "sites"):
            return None
        from ..stages.sites import records as sites

        return self._read(sites, "sites", contig)

    def release(self, contig: str) -> FounderRelease | None:
        """Read assembly's FounderRelease segments and identity flags, including left-partner ownership."""
        if not self._has(contig, "release"):
            return None
        from ..stages.release import records

        return self._read(records, "release", contig)

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

    def stage_request(self, stage, contig=None, *, round_name="", pairs=False, **options):
        """Ask the data table's record owner for a header-only workflow request."""
        spec = STAGES[stage]
        adapter, product = (spec.pair_records, "pairs") if pairs else (spec.records, spec.product)
        if stage == "selection" and round_name:
            options["stage"] = "local_selection_" + round_name
        elif stage == "assembly" and round_name:
            options["round_name"] = round_name.removeprefix("context.")
        elif stage == "painting":
            options.update(panels=False, emissions=False)
        elif stage == "refit":
            options["panels"] = False
        return self.request(importlib.import_module("haplotype_reconstruction." + adapter), product, contig, **options)

    def _panel_request(self, contig: str, stage: str) -> ProductReader[Any] | None:
        """Read a published evaluation panel label through its one owning stage."""
        if stage not in FOUNDER_STAGES:
            raise SettingsError(f"unknown founder stage {stage!r}; valid: {', '.join(FOUNDER_STAGES)}")
        name, spec = next((name, spec) for name, spec in STAGES.items() if stage in spec.panels)
        if not self._has(contig, spec.prefix):
            return None
        adapter = importlib.import_module("haplotype_reconstruction." + spec.records)
        if name in ("painting", "refit"):
            options = {"panels": True, **({"emissions": False} if name == "painting" else {})}
            request = self.request(adapter, spec.product, contig, **options)
            return None if request is None else PanelReader(request, stage)
        return self.request(adapter, "stage_panels", contig, stage=stage)

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
            return self.receipt(request, contig) is not None
        name = next((name for name, spec in STAGES.items() if spec.status and stage in spec.evaluation), None)
        if name is not None:
            return self.receipt(self.stage_request(name, contig), contig) is not None
        raise SettingsError(f"unknown evaluation stage {stage!r}")


    def output_selection(self) -> Mapping[str, Any] | None:
        """Refresh the latest output-only choice; it is never a scientific binding."""
        path = self.root / "output_selection.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text())
        require_schema(payload, 1)
        return payload

    def simulation_manifest(self) -> SimulationManifest | None:
        """Read template generation's seed/truth metadata and check it against init's source binding."""
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
        """Expose recorded painting and ordered search summaries for painting reports formatting."""
        if not self._has(contig, "painting"):
            return None
        from ..stages.painting import records

        return self._read(records, "painting_report", contig)

    def tie_sources(self, contig: str) -> tuple[Any, ...]:
        """Collect only explicitly declared stage tie views in the published report order."""
        sources: list[Any] = []
        if self._has(contig, "evidence"):
            from ..stages.evidence import records as evidence_records

            diagnostics = self._read(evidence_records, "diagnostics", contig)
            if diagnostics is not None:
                sources.append(TieSource("evidence", (diagnostics.tie_events,)))
        if self._has(contig, "discovery"):
            from ..stages.discovery import records as discovery_records

            sources.extend(self._read(discovery_records, "tie_sources", contig) or ())
        if self._has(contig, "selection"):
            from ..stages.selection import records as selection_records

            sources.extend(self._read(selection_records, "tie_sources", contig) or ())
        if self._has(contig, "assembly"):
            from ..stages.assembly import records as assembly_records

            sources.extend(self._read(assembly_records, "tie_sources", contig) or ())
        if self._has(contig, "painting"):
            from ..stages.painting import records as painting_records

            sources.extend(self._read(painting_records, "tie_sources", contig) or ())
        if self._has(contig, "founder_refit"):
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
            if self._has(c, "founder_refit")
            if (row := self._read(records, "summary", c)) is not None
        )

    def phase_summaries(self, *, elapsed_seconds: Mapping[str, float] | None = None) -> tuple[PhaseSummary, ...]:
        """Read available phase summaries without inferring any absent report field."""
        from ..stages.phase import records

        return tuple(
            row
            for c in self.contigs
            if self._has(c, "phase")
            if (row := self._read(records, "phase_summary", c, elapsed_seconds=(elapsed_seconds or {}).get(c)))
            is not None
        )

    def record_bytes(self, contig: str, product: str) -> int:
        """Return checked decoded bytes for evaluation's worker estimate.

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
            name = next(name for name, spec in STAGES.items() if product in spec.panels)
            request = self.stage_request(name, contig) if name in ("painting", "refit") else self._panel_request(contig, product)
        else:
            name = next((name for name, spec in STAGES.items() if spec.status and product in spec.evaluation), None)
            if name is None:
                raise SettingsError(f"unknown evaluation product {product!r}")
            request = self.stage_request(name, contig)
        record = self.receipt(request, contig)
        return 0 if record is None else record.info.decoded_bytes

    def product_identity(self, product: str, contig: str | None = None) -> str | None:
        """Return an output dependency's manifest-derived identity after its shared receipt check."""
        if contig is None:
            raise SettingsError(f"{product}: a contig is required")
        if product not in ("sites", "release", "phase"):
            raise SettingsError(f"unsupported output dependency product {product!r}")
        spec = STAGES[product]
        if not self._has(contig, spec.prefix):
            return None
        request = self.stage_request(product, contig)
        return None if self.receipt(request, contig) is None else request.identity
