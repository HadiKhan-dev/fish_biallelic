"""The run configuration: one typed tree of user settings (TOML sections), strictly validated.

Sections: run (where results go, where to stop), execution (CPUs, lanes, shards; never part of any scientific
identity), input (VCF or simulation, contigs), samples (optional metadata workbook), quality (evidence QC and read
calibration), genetic_map (inference prior), pedigree (+ eligibility), family_phase, recombination, export (the
final founder and sample VCFs), reports, simulation (+ reads; generating parameters only). Stage-internal
constants are not settings: they are the defaults of each stage's own config (config/stages.py maps these sections
into them).

Every field carries its kind (how TOML values are parsed and checked) and a one-line description (docs are
generated from it). Paths are resolved against the working directory when the configuration is loaded.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

CICHLID_AUTOSOMES = tuple(f"chr{index}" for index in (*range(1, 21), 22, 23))


def setting(default, kind: str, doc: str, choices: tuple = ()):
    """A configuration field: ``kind`` is one of bool, int, float, str, path, strings, ints, table, cores, seed,
    memory."""
    if isinstance(default, (dict, list)):
        raise TypeError("mutable defaults are not allowed")
    return field(default=default, metadata=dict(kind=kind, doc=doc, choices=tuple(choices)))


def section(factory, doc: str):
    return field(default_factory=factory, metadata=dict(kind="section", doc=doc, choices=()))


@dataclass(frozen=True)
class RunSection:
    output: Path | None = setting(None, "path", "Run directory (outputs, logs, run records); default "
                                  "work/runs/<preset or input kind>[_seed<seed>].")
    checkpoints: Path | None = setting(None, "path", "Checkpoint directory; default OUTPUT/checkpoints.")
    stop_after: str = setting("", "str", "Stop cleanly after this stage: discovery, or painting (per-contig work "
                              "done, genome-wide stages not started); empty runs everything.",
                              ("", "discovery", "painting"))


@dataclass(frozen=True)
class ExecutionSection:
    cores: int | str = setting("auto", "cores", "Total CPU ceiling (processes x threads); auto = CPU affinity.")
    contig_lanes: int | str = setting("auto", "cores", "Contigs reconstructed concurrently; auto = one lane per 8 "
                                      "cores, paced by measured memory pressure; 1 = one contig at a time with all cores.")
    memory_limit: int | str = setting("auto", "memory", "Capacity for measured job memory pressure: a size such as "
                                      "\"200G\" (binary units), or auto = job cgroup capacity. SOFT/HARD "
                                      "thresholds and sampling interval are in execution/config.py; affects scheduling only.")
    serial: bool = setting(False, "bool", "Discover, assemble and paint one contig at a time, each recorded "
                           "before the next (short allocations).")
    max_contigs: int = setting(0, "int", "Finish at most this many more contigs in this invocation, then stop "
                               "cleanly (0 = no limit; > 0 implies serial).")
    process_contigs: tuple[str, ...] = setting((), "strings", "Shard: process only these contigs (of "
                                               "[input] contigs) through painting; no genome-wide stages.")
    batch_queue: Path | None = setting(None, "path", "Shared directory through which batch-worker helper "
                                       "allocations take local-selection fits and hierarchy join batches.")
    profile: bool = setting(False, "bool", "Print per-phase timings and thread-allocation traces.")


@dataclass(frozen=True)
class InputSection:
    kind: str = setting("vcf", "str", "Observations: an indexed VCF/BCF with FORMAT/AD, or a simulation from "
                        "founder templates.", ("vcf", "simulation"))
    vcf: Path | None = setting(None, "path", "Input VCF/BCF (kind = vcf).")
    contigs: tuple[str, ...] = setting((), "strings", "Contigs to analyse, in order (required unless the preset "
                                       "lists them).")


@dataclass(frozen=True)
class SamplesSection:
    metadata: Path | None = setting(None, "path", "Metadata workbook (.xlsx), or a .tsv/.csv table, describing "
                                    "the samples.")
    sheet: str = setting("main_data", "str", "Workbook sheet.")
    id_column: str = setting("primary_ID", "str", "Column holding the VCF sample IDs.")
    generation_column: str = setting("generation", "str", "Column holding generation/cohort labels.")
    sex_column: str = setting("sex", "str", "Column holding recorded sex (F/M).")
    reference_generation: str = setting("", "str", "Generation whose rows the reference-consistency report "
                                        "compares with discovered blocks (e.g. G0).")
    generation_prefixes: tuple[str, ...] = setting((), "strings", "Labels starting with one of these are read "
                                                   "as that prefix (e.g. G0 for G0_AulStu).")
    generations: tuple[str, ...] = setting((), "strings", "Allowed generation labels (empty: any).")
    sexes: tuple[str, ...] = setting((), "strings", "Allowed sex labels (empty: any).")


@dataclass(frozen=True)
class QualitySection:
    site_filter: str = setting("auto", "str", "Self-calibrated per-contig site filter; auto = on for VCF input, "
                               "off for simulations.", ("auto", "on", "off"))
    sample_filter: str = setting("auto", "str", "Per-contig library-contamination QC (flagged samples leave "
                                 "founder construction); auto = on for VCF input, off for simulations; it runs "
                                 "only where the site filter is applied.", ("auto", "on", "off"))
    read_calibration: bool = setting(True, "bool", "Fit read error, allelic balance and overdispersion from the "
                                     "observed AD; off = fixed binomial model (error 0.02).")


@dataclass(frozen=True)
class GeneticMapSection:
    rate_cm_per_mb: float = setting(5.0, "float", "Fallback recombination rate (cM/Mb) without a map or beyond "
                                    "its knots.")
    file: Path | None = setting(None, "path", "Cumulative cM map (PLINK, or chromosome/position_bp/map_cm TSV).")


@dataclass(frozen=True)
class EligibilitySection:
    policy: str = setting("none", "str", "Parent candidates: none (every sample), constraints (JSON file), "
                          "cohorts (workbook generations).", ("none", "constraints", "cohorts"))
    constraints: Path | None = setting(None, "path", "JSON constraints (policy = constraints): candidate_parents, "
                                       "excluded_samples, ineligible_children, direction_supported_edges.")
    parent_generations: tuple[tuple[str, str], ...] = setting((), "table", "Cohorts: child generation -> parent "
                                                              "generation, e.g. {F2 = \"F1\", F3 = \"F2\"}.")
    opposite_sex_pairs: bool = setting(True, "bool", "Cohorts: two-parent configurations need one recorded "
                                       "female and one recorded male.")


@dataclass(frozen=True)
class PedigreeSection:
    evidence_calibration: str = setting("predictive", "str", "Cross-chromosome pedigree evidence scaling: "
                                        "predictive, or off for unscaled evidence.", ("predictive", "off"))
    eligibility: EligibilitySection = section(EligibilitySection, "Who may be whose parent.")


@dataclass(frozen=True)
class FamilyPhaseSection:
    protected_intervals: Path | None = setting(None, "path", "TSV of intervals (chrom, start, end, reason) where "
                                               "family phase keeps its frames.")


@dataclass(frozen=True)
class RecombinationSection:
    shared_family_evidence: bool = setting(True, "bool", "Use shared-family phase-error evidence in the maps.")
    order_breakpoints: Path | None = setting(None, "path", "TSV of reference-order breakpoints (chrom, start, "
                                             "end, reason).")


@dataclass(frozen=True)
class ExportSection:
    enabled: bool = setting(True, "bool", "Write the final products at the end of a complete run (OUTPUT/export): per "
                            "contig a founder VCF (haploid GT, one column c<component>_H<row> per founder row) and "
                            "the phased sample VCF (GT:PS), bgzipped with tabix indexes.")
    full_length: bool = setting(True, "bool", "Write every input biallelic SNP that reached the site filter (FILTER "
                                "PASS for kept sites, the rule codes r1-r5 for removed ones); false writes only "
                                "the kept sites.")


@dataclass(frozen=True)
class ReportsSection:
    reference_consistency: bool = setting(False, "bool", "Compare discovered blocks with the reference-generation "
                                          "rows ([samples] reference_generation).")


@dataclass(frozen=True)
class SimulationReadsSection:
    error_rate: float = setting(0.02, "float", "Generating sequencing error rate.")
    depth_cv: float = setting(0.0, "float", "Between-sample depth coefficient of variation.")
    heterozygote_alt_probability: float = setting(0.5, "float", "Alternate-read probability in heterozygotes.")
    dropout_fraction: float = setting(0.0, "float", "Contiguous missing-marker fraction per sample.")
    read_length_bp: int = setting(0, "int", "Read length; > 0 samples paired-end fragments shared by "
                                  "neighbouring sites.")
    fragment_mean_bp: float = setting(375.0, "float", "Mean fragment length (read_length_bp > 0).")
    fragment_sd_bp: float = setting(150.0, "float", "Fragment length standard deviation.")


@dataclass(frozen=True)
class SimulationSection:
    seed: int | str = setting(400, "seed", "Simulation seed, or \"random\" (the realised seed is recorded).")
    templates: Path | None = setting(None, "path", "Directory of chromosome NPZ founder templates.")
    cohorts: tuple[int, ...] = setting((20, 100, 200), "ints", "Individuals in successive cohorts.")
    depth: float = setting(5.0, "float", "Mean read depth.")
    generating_rate_cm_per_mb: float = setting(5.0, "float", "Generating fallback recombination rate.")
    generating_map: Path | None = setting(None, "path", "Generating genetic map (independent of inference).")
    backcross_fraction: float = setting(0.0, "float", "Fraction of later matings with an actual parent.")
    backcross_start_cohort: int = setting(3, "int", "First cohort with backcrosses.")
    observe_cohorts: tuple[int, ...] = setting((), "ints", "Only sequence these cohorts (empty: all).")
    observed_fraction: float = setting(1.0, "float", "Random fraction of eligible individuals sequenced.")
    pedigree: Path | None = setting(None, "path", "Explicit pedigree table (TSV: sample, parent1, parent2, sex, "
                                    "generation, observed); replaces cohorts and the random matings.")
    reads: SimulationReadsSection = section(SimulationReadsSection, "Generating read model.")


@dataclass(frozen=True)
class RunConfig:
    """The resolved configuration of one run (see the module docstring)."""

    preset: str = setting("", "str", "Built-in base configuration: tromau or aulstu.", ("", "tromau", "aulstu"))
    run: RunSection = section(RunSection, "Where results go and where to stop.")
    execution: ExecutionSection = section(ExecutionSection, "CPUs, lanes and shards (never part of results).")
    input: InputSection = section(InputSection, "Observations and contigs.")
    samples: SamplesSection = section(SamplesSection, "Optional sample metadata.")
    quality: QualitySection = section(QualitySection, "Evidence quality control and read model.")
    genetic_map: GeneticMapSection = section(GeneticMapSection, "Inference genetic map.")
    pedigree: PedigreeSection = section(PedigreeSection, "Pedigree inference.")
    family_phase: FamilyPhaseSection = section(FamilyPhaseSection, "Family phase.")
    recombination: RecombinationSection = section(RecombinationSection, "Recombination maps.")
    export: ExportSection = section(ExportSection, "Final products: founder and sample VCFs.")
    reports: ReportsSection = section(ReportsSection, "Optional reports.")
    simulation: SimulationSection = section(SimulationSection, "Simulation (input kind = simulation).")

    # Sections that never influence scientific results.
    EXECUTION_SECTIONS = ("run", "execution", "export", "reports")

    def scientific(self) -> dict:
        """The settings that can change results (all sections but run, execution, export and reports), as plain
        data."""
        data = to_dict(self)
        for name in self.EXECUTION_SECTIONS:
            data.pop(name, None)
        return data

    @property
    def serial(self) -> bool:
        """Contigs one at a time ([execution] serial, implied by max_contigs > 0)."""
        return self.execution.serial or self.execution.max_contigs > 0

    def site_filter(self) -> bool:
        """The site filter after resolving auto (on for VCF input, off for simulations)."""
        value = self.quality.site_filter
        return self.input.kind == "vcf" if value == "auto" else value == "on"

    def sample_filter(self) -> bool:
        value = self.quality.sample_filter
        return self.input.kind == "vcf" if value == "auto" else value == "on"


def schema_fields(cls) -> list:
    """The configuration fields of a section class, in declaration order."""
    return [item for item in fields(cls) if "kind" in item.metadata]


def to_dict(value) -> Any:
    """Plain data (paths as strings, tables as dicts, None as "") of a config object."""
    if hasattr(value, "__dataclass_fields__"):
        result = {}
        for item in schema_fields(type(value)):
            entry = getattr(value, item.name)
            if item.metadata["kind"] == "table":
                result[item.name] = dict(entry)
            else:
                result[item.name] = to_dict(entry)
        return result
    if isinstance(value, tuple):
        return [to_dict(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None:
        return ""
    return value


def reference_rows(cls=RunConfig, prefix: tuple = ()) -> list[dict]:
    """Every configuration key with its section, kind, default, choices and description (docs/configuration.md is
    generated from these rows, so the documentation cannot drift from the schema)."""
    rows = []
    for item in schema_fields(cls):
        kind = item.metadata["kind"]
        if kind == "section":
            rows.extend(reference_rows(type(item.default_factory()), (*prefix, item.name)))
            continue
        default = item.default
        rows.append(dict(section=".".join(prefix), key=item.name, kind=kind,
                         default=to_dict(default) if kind != "table" else dict(default),
                         choices=[choice for choice in item.metadata["choices"] if choice != ""],
                         doc=item.metadata["doc"]))
    return rows
