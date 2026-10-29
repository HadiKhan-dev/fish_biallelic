"""RunConfig -> the stage configs (one function per stage) and the run's genetic maps.

Stage configs are frozen dataclasses owned by their stages; their defaults are base production values and are
internal model constants. Only settings base exposed are mapped from the run configuration:
[quality] -> evidence; [pedigree] evidence_calibration -> pedigree; [family_phase] protected_intervals and
[recombination] order_breakpoints (breakpoint TSVs, parsed here so identities hold their content) and
shared_family_evidence -> family phase / recombination; [export] full_length -> the export stage (its
enabled switch is read by the driver); [simulation] -> the simulation source; [genetic_map] -> the
GeneticMapSet every stage receives. Stage modules are imported inside the functions: the configuration layer
does not depend on the stages at import time.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .schema import RunConfig


def evidence(config: RunConfig):
    from ..evidence.config import EvidenceConfig
    base = EvidenceConfig()
    return replace(base,
                   site_quality=replace(base.site_quality, enabled=config.site_filter()),
                   sample_quality=replace(base.sample_quality, enabled=config.sample_filter()),
                   read_model=replace(base.read_model, enabled=config.quality.read_calibration))


def discovery(config: RunConfig):
    from ..discovery.config import DiscoveryConfig
    return DiscoveryConfig()


def local_selection(config: RunConfig):
    from ..local_selection.config import LocalSelectionConfig
    return LocalSelectionConfig()


def assembly(config: RunConfig):
    from ..assembly.config import AssemblyConfig
    return AssemblyConfig()


def painting(config: RunConfig):
    from ..painting.config import PaintingConfig
    return PaintingConfig()


def pedigree(config: RunConfig):
    from ..pedigree.config import PedigreeConfig
    return PedigreeConfig(evidence_calibration=config.pedigree.evidence_calibration)


def founder_refit(config: RunConfig):
    from ..founder_refit.config import FounderRefitConfig
    return FounderRefitConfig()


def family_phase(config: RunConfig):
    from ..family_phase.breakpoints import read_breakpoint_table
    from ..family_phase.config import FamilyPhaseConfig
    path = config.family_phase.protected_intervals
    return FamilyPhaseConfig(protected_intervals=() if path is None else read_breakpoint_table(path).protected())


def recombination(config: RunConfig):
    from ..family_phase.breakpoints import read_breakpoint_table
    from ..recombination.config import RecombinationConfig
    path = config.recombination.order_breakpoints
    return RecombinationConfig(shared_family_evidence=config.recombination.shared_family_evidence,
                               order_breakpoints=() if path is None else read_breakpoint_table(path).rows)


def export(config: RunConfig):
    from ..export.config import ExportConfig
    return ExportConfig(full_length=config.export.full_length)


def simulation(config: RunConfig):
    from ..simulation.design import ReadModel, SimulationConfig, SimulationDesign
    section = config.simulation
    reads = section.reads
    return SimulationConfig(
        templates=section.templates, contigs=config.input.contigs,
        seed=None if section.seed == "random" else section.seed, cohorts=section.cohorts, depth=section.depth,
        generating_rate_cm_per_mb=section.generating_rate_cm_per_mb, generating_map=section.generating_map,
        design=SimulationDesign(backcross_fraction=section.backcross_fraction,
                                backcross_start_cohort=section.backcross_start_cohort,
                                observe_cohorts=section.observe_cohorts,
                                observed_fraction=section.observed_fraction, pedigree=section.pedigree),
        reads=ReadModel(error_rate=reads.error_rate, depth_cv=reads.depth_cv,
                        heterozygote_alt_probability=reads.heterozygote_alt_probability,
                        dropout_fraction=reads.dropout_fraction, read_length_bp=reads.read_length_bp,
                        fragment_mean_bp=reads.fragment_mean_bp, fragment_sd_bp=reads.fragment_sd_bp))


def genetic_maps(config: RunConfig):
    """The inference genetic maps ([genetic_map]); contigs without a map get the fallback rate."""
    from ..model.genetic_map import load_genetic_maps
    return load_genetic_maps(config.genetic_map.file, config.genetic_map.rate_cm_per_mb)


@dataclass(frozen=True)
class StageConfigs:
    """Every stage's config for one run."""

    evidence: Any
    discovery: Any
    local_selection: Any
    assembly: Any
    painting: Any
    pedigree: Any
    founder_refit: Any
    family_phase: Any
    recombination: Any
    export: Any


def stage_configs(config: RunConfig) -> StageConfigs:
    return StageConfigs(
        evidence=evidence(config), discovery=discovery(config), local_selection=local_selection(config),
        assembly=assembly(config), painting=painting(config), pedigree=pedigree(config),
        founder_refit=founder_refit(config), family_phase=family_phase(config),
        recombination=recombination(config), export=export(config))
