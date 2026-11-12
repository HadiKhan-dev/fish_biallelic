"""Command functions for frozen observations and explicit truth-sidecar evaluation.

CLI registration is owned by the application integrator. These functions can also
be called directly with resolved Settings or a configuration filename.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ..core.errors import InputError

# Only keys accepted by unmodified release 4.14.0 belong in inference.toml.
# New simulation settings can evolve independently of this compatibility surface.
_RELEASE_414_KEYS = {
    "run": ("output", "checkpoints", "exclude_contigs", "stop_after"),
    "execution": ("cores", "contig_lanes", "memory_limit", "serial", "max_contigs", "profile", "scratch"),
    "input": ("kind", "vcf", "contigs"),
    "quality": ("site_filter", "sample_filter", "read_calibration"),
    "genetic_map": ("rate_cm_per_mb", "file", "rate_is_measured"),
    "relatedness": ("pairwise_ibd",),
    "pedigree": ("evidence_calibration",),
    "family_phase": ("protected_intervals",),
    "recombination": ("order_breakpoints",),
    "export": ("enabled", "full_length"),
}


def inference_configuration(settings, output):
    """Expand the simulation's effective inference choices using the 4.14 keys.

    Auto filters are resolved for the simulation source before switching the input
    kind. External inference maps/intervals are copied into the frozen package.
    Returns rendered TOML and the files the output writer must copy.
    """
    from .settings.load import _toml_value
    from .settings.schema import to_dict

    output = Path(output).expanduser().resolve()
    mapping = to_dict(settings)
    sections = {section: {key: mapping[section][key] for key in keys} for section, keys in _RELEASE_414_KEYS.items()}
    sections["input"].update(kind="vcf", vcf=str(output / "observations.bcf"))
    sections["run"].update(output="", checkpoints="", stop_after="")
    sections["execution"]["scratch"] = ""
    for key in ("site_filter", "sample_filter"):
        if sections["quality"][key] == "auto":
            sections["quality"][key] = "off"
    copies = {}
    for section, key, filename in (
        ("genetic_map", "file", "genetic_map.tsv"),
        ("family_phase", "protected_intervals", "protected_intervals.tsv"),
        ("recombination", "order_breakpoints", "order_breakpoints.tsv"),
    ):
        source = getattr(getattr(settings, section), key)
        if source is not None:
            copies[filename] = Path(source)
            sections[section][key] = str(output / filename)
    lines = [
        "version = 4",
        'preset = ""',
        "# After moving, use --set input.vcf=PATH/observations.bcf; paths resolve against the working directory.",
    ]
    for section, values in sections.items():
        lines.extend(("", f"[{section}]"))
        lines.extend(f"{key} = {_toml_value(value)}" for key, value in values.items())
    return "\n".join(lines) + "\n", copies


def write_observations(config, output):
    """Materialise one simulation configuration and publish its frozen package."""
    from .. import __version__
    from ..engine.runtime import start_run
    from ..outputs.observations import ObservationContig, write_package
    from ..stages.configure import simulation
    from ..stages.inputs.simulation import SimulationSource, load_truth
    from ..store.store import CheckpointStore
    from .run import resolve_cores
    from .settings.load import load_settings, validate
    from .settings.schema import Settings

    settings = config if isinstance(config, Settings) else load_settings(config)
    validate(settings)
    if settings.input.kind != "simulation":
        raise InputError("writing observations requires a simulation configuration")
    output = Path(output).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise InputError("observation output directory must be new or empty")
    text, files = inference_configuration(settings, output)
    cores = resolve_cores(settings)
    parameters = simulation(settings)
    source = SimulationSource(parameters)
    if parameters.design.pedigree is not None and len(source.sample_ids()) < 3:
        raise InputError("observation package requires at least three uniquely named observed samples")
    start_run(cores, preload=("haplotype_reconstruction.stages.inputs.simulation",))
    with tempfile.TemporaryDirectory(prefix="observation-simulation-") as temporary:
        store = CheckpointStore(Path(temporary) / "checkpoints", nthreads=cores)
        manifest = source.materialise(store, cores=cores)

        def load_contig(contig):
            reads = source.reads(contig)
            truth = load_truth(store.contig(contig), source.payload_identity(contig))
            length = (
                parameters.contig_length(contig)
                if parameters.source.kind == "coalescent"
                else int(reads.positions[-1])
                if len(reads.positions)
                else 0
            )
            return ObservationContig(reads, truth, length)

        return write_package(
            output,
            manifest,
            load_contig,
            text,
            generator=dict(version=__version__, truth_schema=2),
            settings_files=files,
        )


def evaluate_with_truth(
    run, truth_dir, *, output=None, stages=("available",), contigs=None, release_tree=None, python=None, timeout=600
):
    """Score a VCF-route run with bound truth, using its release's checked reader.

    Supply ``release_tree`` to decode a 4.14.0 run in a separate interpreter. The
    current-release path uses ordinary Results. Reports go to RUN/evaluation unless
    an explicit output is given. Truth never enters an inference call.
    """
    from ..evaluation.evaluate import collect, write_evaluation
    from ..evaluation.tasks import evaluate_task, plan_evaluation
    from ..results.old_products import OldProducts, RunProducts

    reader = (
        RunProducts(run) if release_tree is None else OldProducts(run, release_tree, python=python, timeout=timeout)
    )
    report, tasks, evaluation_reader = plan_evaluation(reader, truth=truth_dir, stages=stages, contigs=contigs)
    evaluated = [evaluate_task(task, evaluation_reader) for task in tasks]
    components, matches = collect(report, [result.parts() for result in evaluated])
    from ..contracts.contigs import AnalysisContigs
    from ..stages.analysis import output_directory

    analysis = AnalysisContigs.from_manifest(reader.manifest)
    if analysis.excluded:
        report["analysis"] = analysis.document()
    target = (
        Path(output).expanduser().resolve()
        if output is not None
        else output_directory(reader.root, analysis) / "evaluation"
    )
    write_evaluation(target, report, components, matches)
    return report
