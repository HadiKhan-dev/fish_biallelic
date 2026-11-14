"""Describe stage owners, published spellings and workflow membership as plain data."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    """Static ownership; task factories and codecs retain all executable policy."""

    tasks: str
    records: str | None
    product: str
    owners: tuple[str, ...] = ()
    kernel_roots: tuple[str, ...] = ()
    kernels: tuple[str, ...] = ()
    steps: tuple[str, ...] = ()
    prefix: str = ""
    status: bool = False
    panels: tuple[str, ...] = ()
    evaluation: tuple[str, ...] = ()


# Module paths keep catalog reads independent of scientific imports.
# Numerical roots scope compile waits; task owners retain the complete input/record identity and preload closure.
# Record prefixes and evaluation labels are serialized public spellings. Keeping them here preserves existing
# record keys and filenames while graph/task stage names stay unambiguous.
STAGES = {
    "init": Stage(
        "stages.init.tasks",
        None,
        "configuration/resolved",
        steps=("init",),
    ),
    "evidence": Stage(
        "stages.evidence.tasks",
        "stages.evidence.records",
        "evidence",
        owners=("stages.evidence.tasks",),
        kernel_roots=("science.evidence.run", "science.evidence.likelihoods"),
        kernels=("evidence",),
        steps=("evidence",),
        prefix="evidence",
        status=True,
    ),
    "relatedness": Stage(
        "stages.relatedness.tasks",
        "stages.relatedness.families_records",
        "families",
        owners=("stages.relatedness.families_task",),
        kernel_roots=("science.relatedness.run", "science.evidence.likelihoods"),
        kernels=("relatedness",),
        steps=("relatedness",),
        prefix="relatedness",
    ),
    "discovery": Stage(
        "stages.discovery.tasks",
        "stages.discovery.records",
        "discovery",
        owners=("stages.discovery.tasks",),
        kernel_roots=("science.discovery.run", "science.evidence.likelihoods"),
        kernels=("discovery",),
        steps=("reconstruct",),
        prefix="discovery",
        status=True,
        panels=("discovery",),
    ),
    "selection": Stage(
        "stages.selection.tasks",
        "stages.selection.records",
        "selected",
        owners=("stages.selection.tasks",),
        kernel_roots=(
            "science.selection.run",
            "science.assembly.run",
            "science.evidence.likelihoods",
            "contracts.painting",
            "science.selection.context_refit",
            "science.assembly.alleles",
            "science.shared.emissions",
        ),
        kernels=("selection", "assembly", "shared.emissions"),
        steps=("reconstruct",),
        prefix="selection",
        status=True,
        panels=("local_selection_initial", "local_selection_l1", "local_selection_l2"),
    ),
    "assembly": Stage(
        "stages.assembly.tasks",
        "stages.assembly.records",
        "assembly",
        owners=("stages.assembly.tasks", "stages.assembly.final"),
        kernel_roots=(
            "stages.assembly.final",
            "science.assembly.kernel_inventory",
            "science.evidence.likelihoods",
            "contracts.painting",
        ),
        kernels=("assembly", "shared.emissions"),
        steps=("reconstruct",),
        prefix="assembly",
        status=True,
        panels=("assembly_l1", "assembly_l2", "assembly_l3", "assembly_l4", "founder_refinement"),
    ),
    "painting": Stage(
        "stages.painting.tasks",
        "stages.painting.records",
        "painting",
        owners=("stages.painting.tasks",),
        kernel_roots=("stages.painting.compute", "science.evidence.likelihoods"),
        kernels=("painting",),
        steps=("reconstruct",),
        prefix="painting",
        status=True,
        panels=("painting",),
    ),
    "refit": Stage(
        "stages.refit.tasks",
        "stages.refit.records",
        "refit",
        owners=("stages.refit.tasks",),
        kernel_roots=("science.refit.run", "science.evidence.likelihoods", "core.numerics"),
        kernels=("refit",),
        steps=("reconstruct",),
        prefix="founder_refit",
        status=True,
        panels=("founder_refit",),
    ),
    "pedigree": Stage(
        "stages.pedigree.tasks",
        "stages.pedigree.records",
        "pedigree",
        owners=("stages.pedigree.prepare", "stages.pedigree.screen", "stages.pedigree.scores", "stages.pedigree.decision"),
        kernel_roots=("science.pedigree", "science.painting", "science.evidence.likelihoods"),
        kernels=("pedigree", "painting", "shared.recombination"),
        steps=("pedigree",),
        prefix="pedigree",
        evaluation=("pedigree",),
    ),
    "phase": Stage(
        "stages.phase.tasks",
        "stages.phase.records",
        "phase",
        owners=("stages.phase.tasks",),
        kernel_roots=("science.phase", "science.painting", "science.evidence.likelihoods"),
        kernels=("phase", "painting", "shared.recombination"),
        steps=("phase",),
        prefix="phase",
        status=True,
        evaluation=("family_phase",),
    ),
    "recombination": Stage(
        "stages.recombination.tasks",
        "stages.recombination.records",
        "recombination",
        owners=("stages.recombination.tasks",),
        kernel_roots=(
            "science.recombination",
            "science.evidence.likelihoods",
            "contracts.painting",
            "core.numerics",
            "contracts.likelihoods",
        ),
        kernels=("recombination", "shared.recombination"),
        steps=("phase", "recombination"),
        prefix="recombination",
        status=True,
        evaluation=("recombination",),
    ),
    "sites": Stage(
        "stages.sites.tasks",
        "stages.sites.records",
        "sites",
        owners=("stages.sites.tasks",),
        kernel_roots=("science.evidence.run", "engine.workers", "science.evidence.likelihoods"),
        kernels=("evidence",),
        steps=("phase",),
        prefix="sites",
    ),
    "release": Stage(
        "stages.release.tasks",
        "stages.release.records",
        "release",
        owners=("stages.release.tasks", "stages.release.compute"),
        kernel_roots=(
            "stages.release.compute",
            "science.painting",
            "science.evidence.likelihoods",
            "engine.workers",
        ),
        kernels=("discovery", "evidence", "painting", "refit", "selection"),
        steps=("phase",),
        prefix="release",
    ),
    "export": Stage(
        "stages.export.tasks",
        "stages.export.records",
        "export",
        owners=("stages.export.tasks", "outputs.export.task"),
        kernel_roots=("outputs.export.formatting", "engine.workers"),
        kernels=(),
        steps=("phase",),
        prefix="outputs/export",
    ),
    "report": Stage(
        "",
        None,
        "report",
        owners=("outputs.reports.tasks",),
        kernel_roots=(
            "outputs.reports.tasks",
            "science.painting",
            "science.evidence.likelihoods",
            "engine.workers",
        ),
        kernels=("discovery", "evidence", "painting", "pedigree", "refit", "selection", "shared.recombination"),
        steps=("report",),
    ),
}
