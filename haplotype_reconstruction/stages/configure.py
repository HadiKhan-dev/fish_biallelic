"""Resolve settings into stage-owned parameters, preserving ENG-10 field mappings.

Auto quality switches depend only on source kind. Parsed breakpoint content and
map identities are bound; execution and output selections are invocation-local.
Random simulation seeds are realized once by init and reused from its manifest.
No science parameter is invented or adjusted according to available resources.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol

from ..core.digest import canonical_json, json_digest, jsonable
from ..core.errors import SettingsError
from ..core.random import realise_seed

if TYPE_CHECKING:
    from ..contracts.genetic_map import GeneticMapSet
    from ..science.assembly.parameters import AssemblyParameters
    from ..science.discovery.parameters import DiscoveryConfig
    from ..science.evidence.parameters import EvidenceParameters
    from ..science.painting.parameters import PaintingParameters
    from ..science.pedigree.parameters import PedigreeConfig
    from ..science.phase.parameters import PhaseParameters
    from ..science.recombination.parameters import RecombinationParameters
    from ..science.refit.parameters import RefitParameters
    from ..science.relatedness.ibd_parameters import PairwiseIbdParameters
    from ..science.relatedness.parameters import FamilyParameters
    from ..science.selection.parameters import SelectionParameters
    from ..simulation.design import SimulationConfig

__all__ = [
    "SettingsView",
    "StageConfigs",
    "ExportParameters",
    "OutputSelection",
    "ResolvedConfiguration",
    "configure",
    "stage_configs",
    "simulation",
    "genetic_maps",
    "resolve",
    "resolve_and_bind",
    "RETIRED_BASELINE_FIELDS",
    "bind_resolved",
    "FIELD_MAP",
    "baseline_projection",
]


class SettingsView(Protocol):
    """Structural boundary to I1 settings, without an upward interface import."""

    @property
    def input(self) -> Any: ...
    @property
    def quality(self) -> Any: ...
    @property
    def genetic_map(self) -> Any: ...
    @property
    def pedigree(self) -> Any: ...
    @property
    def family_phase(self) -> Any: ...
    @property
    def recombination(self) -> Any: ...
    @property
    def export(self) -> Any: ...
    @property
    def simulation(self) -> Any: ...


@dataclass(frozen=True)
class ExportParameters:
    """ENG-10 output-only setting; I1 passes this value into OU1's own config."""

    full_length: bool = field(default=True, metadata={"provenance": "3.5 export/config.py:ExportConfig"})
    file_format: str = field(default="vcf", metadata={"provenance": "3.5 export/config.py:ExportConfig"})
    products: tuple[str, ...] = field(
        default=("founders", "samples"), metadata={"provenance": "3.5 export/config.py:ExportConfig"}
    )


@dataclass(frozen=True)
class OutputSelection:
    """Invocation-local output selection; never part of the init binding."""

    export_enabled: bool
    pairwise_ibd: bool


@dataclass(frozen=True)
class StageConfigs:
    """Typed stage-owned parameters; legacy names retained in the field map."""

    evidence: EvidenceParameters
    discovery: DiscoveryConfig
    local_selection: SelectionParameters
    assembly: AssemblyParameters
    painting: PaintingParameters
    relatedness: FamilyParameters
    pairwise_ibd: PairwiseIbdParameters
    pedigree: PedigreeConfig
    founder_refit: RefitParameters
    family_phase: PhaseParameters
    recombination: RecombinationParameters
    export: ExportParameters


def _quality(value: bool | str, kind: str) -> bool:
    if value == "auto":
        return kind == "vcf"
    if value in ("on", "off"):
        return value == "on"
    if not isinstance(value, bool):
        raise SettingsError(f"quality switch must be on, off, or auto; got {value!r}")
    return value


def stage_configs(settings: SettingsView) -> StageConfigs:
    """The exact 3.5 config/stages mapping, now to each stage's parameter record."""
    from ..science.assembly.parameters import AssemblyParameters
    from ..science.discovery.parameters import DiscoveryConfig
    from ..science.evidence.parameters import EvidenceParameters
    from ..science.painting.parameters import PaintingParameters
    from ..science.pedigree.parameters import ExposureConfig, PedigreeConfig
    from ..science.phase.parameters import PhaseParameters
    from ..science.recombination.parameters import RecombinationParameters
    from ..science.refit.parameters import RefitParameters
    from ..science.relatedness.ibd_parameters import PairwiseIbdParameters
    from ..science.relatedness.parameters import FamilyParameters
    from ..science.selection.parameters import SelectionParameters

    ev = EvidenceParameters()
    quality = settings.quality
    evidence = replace(
        ev,
        site_quality=replace(ev.site_quality, enabled=_quality(quality.site_filter, settings.input.kind)),
        sample_quality=replace(ev.sample_quality, enabled=_quality(quality.sample_filter, settings.input.kind)),
        read_model=replace(ev.read_model, enabled=quality.read_calibration),
    )
    protected, ordered = settings.family_phase.protected_intervals, settings.recombination.order_breakpoints
    if protected is not None or ordered is not None:
        from .inputs.breakpoints import read_breakpoint_table

        protected = () if protected is None else read_breakpoint_table(protected).protected()
        ordered = () if ordered is None else read_breakpoint_table(ordered).rows
    else:
        protected = ordered = ()
    return StageConfigs(
        evidence,
        DiscoveryConfig(),
        SelectionParameters(),
        AssemblyParameters(),
        PaintingParameters(),
        FamilyParameters(rate_is_measured=settings.genetic_map.rate_is_measured),
        PairwiseIbdParameters(),
        PedigreeConfig(
            evidence_calibration=settings.pedigree.evidence_calibration,
            exposure=ExposureConfig(rate_is_measured=settings.genetic_map.rate_is_measured),
        ),
        RefitParameters(),
        PhaseParameters(protected_intervals=tuple(protected)),
        RecombinationParameters(order_breakpoints=tuple(ordered)),
        ExportParameters(full_length=settings.export.full_length),
    )


def configure(settings: SettingsView) -> StageConfigs:
    """Resolve the parameter bundle; init additionally binds orders/maps/seeds via resolve."""
    return stage_configs(settings)


def simulation(settings: SettingsView, *, realised_seed: int | None = None) -> SimulationConfig:
    """Carry every simulation setting without drawing a seed until init resolves it."""
    from ..simulation.design import ReadModel, SimulationConfig, SimulationDesign

    section = settings.simulation
    reads = section.reads
    seed = None if section.seed == "random" else section.seed
    if realised_seed is not None:
        if seed is not None and seed != realised_seed:
            raise SettingsError("configured simulation seed differs from the bound seed")
        seed = realised_seed
    return SimulationConfig(
        templates=section.templates,
        contigs=tuple(settings.input.contigs),
        seed=seed,
        cohorts=tuple(section.cohorts),
        depth=section.depth,
        generating_rate_cm_per_mb=section.generating_rate_cm_per_mb,
        generating_map=section.generating_map,
        design=SimulationDesign(
            backcross_fraction=section.backcross_fraction,
            backcross_start_cohort=section.backcross_start_cohort,
            observe_cohorts=tuple(section.observe_cohorts),
            observed_fraction=section.observed_fraction,
            pedigree=section.pedigree,
        ),
        reads=ReadModel(
            error_rate=reads.error_rate,
            depth_cv=reads.depth_cv,
            heterozygote_alt_probability=reads.heterozygote_alt_probability,
            dropout_fraction=reads.dropout_fraction,
            read_length_bp=reads.read_length_bp,
            fragment_mean_bp=reads.fragment_mean_bp,
            fragment_sd_bp=reads.fragment_sd_bp,
        ),
    )


def genetic_maps(settings: SettingsView) -> GeneticMapSet:
    from .inputs.genetic_map import load_genetic_maps

    return load_genetic_maps(settings.genetic_map.file, settings.genetic_map.rate_cm_per_mb)


@dataclass(frozen=True)
class ResolvedConfiguration:
    """Frozen canonical scientific binding, plus output selections kept outside it.

    The canonical string freezes every nested dataclass value at resolution time,
    including arrays' precomputed map identities. Execution settings never enter.
    """

    parameters: StageConfigs
    contigs: tuple[str, ...]
    sample_ids: tuple[str, ...]
    realised_seeds: tuple[tuple[str, int], ...]
    map_identities: tuple[tuple[str, str], ...]
    simulation_config: SimulationConfig | None
    scientific_json: str
    outputs: OutputSelection

    @property
    def identity(self) -> str:
        import json

        return json_digest(json.loads(self.scientific_json))

    def manifest(self) -> dict[str, Any]:
        import json

        return json.loads(self.scientific_json)


def resolve(
    settings: SettingsView,
    *,
    contigs: Sequence[str] | None = None,
    sample_ids: Sequence[str] = (),
    maps: GeneticMapSet | None = None,
    bound_seeds: Mapping[str, int] | None = None,
) -> ResolvedConfiguration:
    """Resolve once under init's claim; on resume pass seeds from its manifest."""
    parameters = configure(settings)
    order = tuple(settings.input.contigs if contigs is None else contigs)
    samples = tuple(sample_ids)
    if len(set(order)) != len(order) or len(set(samples)) != len(samples):
        raise SettingsError("bound sample and contig orders must be unique")
    maps = genetic_maps(settings) if maps is None else maps
    map_ids = tuple((c, json_digest(maps.identity(c))) for c in order)
    sim = None
    seeds = {}
    if settings.input.kind == "simulation":
        configured = None if settings.simulation.seed == "random" else settings.simulation.seed
        stored = None if bound_seeds is None else bound_seeds.get("simulation")
        if stored is not None and configured is not None and stored != configured:
            raise SettingsError("simulation seed differs from the bound run")
        seed = realise_seed(configured) if stored is None else stored
        seeds["simulation"] = seed
        sim = replace(simulation(settings, realised_seed=seed), contigs=order)
    # full_length is an output parameter too: changing it re-exports, never
    # invalidates scientific records. Pairwise settings bind their own subrecord.
    scientific = jsonable(parameters)
    scientific.pop("export")
    document = dict(
        schema=1,
        parameters=scientific,
        contigs=order,
        sample_ids=samples,
        realised_seeds=seeds,
        map_identities=dict(map_ids),
        simulation=jsonable(sim),
    )
    related = getattr(settings, "relatedness", None)
    outputs = OutputSelection(bool(settings.export.enabled), bool(getattr(related, "pairwise_ibd", False)))
    return ResolvedConfiguration(
        parameters, order, samples, tuple(seeds.items()), map_ids, sim, canonical_json(jsonable(document)), outputs
    )


def bind_resolved(file: Any, resolved: ResolvedConfiguration, *, key: str = "configuration/resolved") -> Any:
    """Publish the immutable init binding through ST1; conflicting bindings refuse."""
    from ..store.protocol import publish

    return publish(file, key, resolved.identity, resolved.manifest())


# Field-by-field migration map. Every unlisted leaf has the identical dotted
# field name. RL1 split the two optional pairwise constants out of family identity.
# The full expanded leaf map is written by the G1 corpus extractor and checked in
# tests; unknown/missing baseline leaves raise, never disappear from comparison.
RETIRED_BASELINE_FIELDS = {
    "pedigree.panel.parents_per_child": "WP_PE2 section 6 / WP_PE4 section 7 (PS25): dead fixed-panel parameter",
    "discovery.residual_births.usable_founder_known_fraction": "WP_DI3 section 7 (DI30): unused field; retired",
}

FIELD_MAP = {
    "pedigree.scoring.bridge_tolerance": "pedigree.scoring.bridge_postcheck_tolerance",
    "evidence.fixed_error": "evidence.read_model.fixed_error",
    "relatedness.ibd_site_step": "pairwise_ibd.ibd_site_step",
    "relatedness.ibd_switch_rate": "pairwise_ibd.ibd_switch_rate",
}


def baseline_projection(parameters: StageConfigs, baseline_fields: Sequence[str]) -> dict[str, Any]:
    """Project named baseline leaves exactly; extra 4.0 constants are not silently compared."""
    data = jsonable(parameters)
    projected = {}
    missing = []
    for source in baseline_fields:
        if source in RETIRED_BASELINE_FIELDS:
            continue
        value = data
        for part in FIELD_MAP.get(source, source).split("."):
            if not isinstance(value, dict) or part not in value:
                missing.append(source)
                break
            value = value[part]
        projected[source] = value
    if missing:
        raise SettingsError(f"unmapped 3.5 configuration fields: {missing}")
    return projected


def resolve_and_bind(
    settings: SettingsView,
    store: Any,
    *,
    contigs: Sequence[str] | None = None,
    sample_ids: Sequence[str] = (),
    maps: GeneticMapSet | None = None,
    source_descriptor: Any = None,
) -> ResolvedConfiguration:
    """Initialize atomically, realizing a random seed exactly once across contenders.

    A fixed request identity excludes invocation-only choices and the unrealized
    entropy value. Its published payload holds the realized seeds. A crash after
    this record but before manifest publication therefore reuses the same seed.
    """
    from ..store.protocol import claim, publish, resume

    store.bind_layout()
    maps = genetic_maps(settings) if maps is None else maps
    order = tuple(settings.input.contigs if contigs is None else contigs)
    parameters = jsonable(configure(settings))
    parameters.pop("export")
    sim = simulation(settings) if settings.input.kind == "simulation" else None
    if sim is not None:
        sim = replace(sim, contigs=order)
    request = dict(
        parameters=parameters,
        contigs=order,
        sample_ids=tuple(sample_ids),
        maps={c: maps.identity(c) for c in order},
        simulation=jsonable(sim),
        source=jsonable(source_descriptor),
    )
    identity = json_digest(request)
    key = "configuration/request"
    with claim(store.run, key, identity) as needed:
        if needed:
            resolved = resolve(settings, contigs=order, sample_ids=sample_ids, maps=maps)
            publish(store.run, key, identity, {"realised_seeds": dict(resolved.realised_seeds)})
        else:
            bound = resume(store.run, key, identity)
            if bound is None:
                raise SettingsError("init claim completed without its configuration record")
            resolved = resolve(
                settings, contigs=order, sample_ids=sample_ids, maps=maps, bound_seeds=bound.payload["realised_seeds"]
            )
    manifest = resolved.manifest()
    manifest["source"] = jsonable(source_descriptor)
    store.bind_run_manifest(manifest)
    return resolved
