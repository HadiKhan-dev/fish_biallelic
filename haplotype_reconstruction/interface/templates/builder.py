"""Build simulation founder templates with fixed-error evidence and window discovery.

Accretion OU20: windows overlap by the carried physical stride; evidence disables
site/sample QC and read calibration. Default discovery searches informative windows,
then the naive linker chains its released probabilities. No simulation or inference
stage consumes builder intermediates. Method: docs/methods/simulation.md#templates.
The application owns files and scheduling here; the linker performs only arithmetic.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ...contracts.likelihoods import slice_likelihoods
from ...contracts.observations import slice_observed
from ...contracts.source import ContigReads
from ...core.errors import InputError, ResourceError
from ...core.files import atomic_path
from ...engine.graph import ResourceRequest, Task, TaskKey, run_graph
from ...engine.runtime import affinity_cpus
from .parameters import (
    TEMPLATE_HAPLOTYPES,
    UNDECLARED_LENGTH_BP,
    WINDOW_ALLOWANCE_BYTES,
    WINDOW_BP,
    WINDOW_EVIDENCE_FACTOR,
    WINDOW_SHIFT_BP,
)

__all__ = [
    "fixed_error_evidence",
    "window_ranges",
    "window_bytes",
    "discover_windows",
    "window_haplotypes",
    "contig_templates",
    "build_templates",
    "templates_command",
]


def fixed_error_evidence(reads: ContigReads, budget: int = 1, *, reverse: bool = False) -> Any:
    """Return unchanged fixed-error evidence with every QC/calibration switch off."""
    from ...science.evidence.parameters import EvidenceParameters
    from ...science.evidence.run import prepare_contig

    config = EvidenceParameters()
    config = replace(
        config,
        site_quality=replace(config.site_quality, enabled=False),
        sample_quality=replace(config.sample_quality, enabled=False),
        read_model=replace(config.read_model, enabled=False),
    )
    return prepare_contig(reads, config, int(budget), reverse=reverse).evidence


def window_ranges(positions: np.ndarray, length_bp: int) -> list[tuple[int, int]]:
    """Return left-searchsorted site ranges of all physical windows, empty ones included."""
    starts = np.arange(0, length_bp, WINDOW_SHIFT_BP, dtype=np.int64)
    first = np.searchsorted(positions, starts, side="left")
    stop = np.searchsorted(positions, starts + WINDOW_BP, side="left")
    return list(zip(first.tolist(), stop.tolist(), strict=False))


def _discover_window(index: int, evidence: Any, config: Any, reverse: bool) -> tuple[int, Any]:
    """Search one informative window and return its original index."""
    from ...science.discovery.run import discover_block

    return index, discover_block(evidence, config, reverse=reverse)


def window_bytes(window: Any) -> int:
    """Report the carried discovery working-set estimate; it never changes search budgets."""
    # Kernels materialise these dense logical windows, even when declarations
    # borrow codes/pairs. Preserve the original dense working-set estimate.
    dense = int(np.prod(window.likelihoods.shape)) * 8 + int(np.prod(window.observed.shape))
    return WINDOW_EVIDENCE_FACTOR * (window.positions.nbytes + window.kept.nbytes + dense) + WINDOW_ALLOWANCE_BYTES


def discover_windows(
    evidence: Any, ranges: list[tuple[int, int]], budget: int = 1, *, reverse: bool = False
) -> list[Any]:
    """Discover informative windows independently; restore window order after execution."""
    from ...science.discovery.parameters import DiscoveryConfig
    from ...science.discovery.run import BlockEvidence

    config = DiscoveryConfig()
    tasks = []
    for index, (first, stop) in enumerate(ranges):
        if stop <= first:
            continue
        window = BlockEvidence(
            block=index,
            site_start=first,
            positions=evidence.positions[first:stop],
            kept=evidence.discovery_sites[first:stop],
            likelihoods=slice_likelihoods(evidence.likelihoods, first, stop),
            observed=slice_observed(evidence.observed, first, stop),
        )
        if window.informative:
            tasks.append(
                Task(
                    TaskKey("templates", str(index)),
                    _discover_window,
                    args=(index, window, config, reverse),
                    resources=ResourceRequest(working_set_bytes=window_bytes(window)),
                )
            )
    results = [None] * len(ranges)
    completed = run_graph(tasks, cores=int(budget), in_process=int(budget) == 1, preload=(__name__,))
    for task in tasks:
        index, block = completed[task.key]
        results[index] = block
    return results


def window_haplotypes(block: Any) -> Any:
    """Return linker probabilities with exactly 0.5 at uncalled sites."""
    from .linker import WindowHaplotypes

    alt = np.where(block.calls >= 0, block.alt_probability, 0.5)
    return WindowHaplotypes(
        np.asarray(block.positions, np.int64), np.asarray(block.kept, bool), np.stack([1.0 - alt, alt], axis=-1)
    )


def contig_templates(
    reads: ContigReads, length_bp: int, budget: int = 1, *, count: int = TEMPLATE_HAPLOTYPES, reverse: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Return positions int64[L] and probabilities float64[H,L,2]; fewer than two windows give H=0."""
    from .linker import link_windows

    evidence = fixed_error_evidence(reads, budget, reverse=reverse)
    blocks = discover_windows(evidence, window_ranges(evidence.positions, length_bp), budget, reverse=reverse)
    windows = [window_haplotypes(block) for block in blocks if block is not None]
    positions, haplotypes = link_windows(windows, count)
    return positions, np.stack(haplotypes) if haplotypes else np.zeros((0, len(positions), 2))


def build_templates(
    vcf: str | Path, output: str | Path, *, contigs: tuple[str, ...], cores: int | None = None, reverse: bool = False
) -> Path:
    """Write each informative contig's NPZ atomically to a new or empty directory."""
    from ...stages.inputs import vcf as vcf_reader

    available = affinity_cpus()
    team = available if cores is None else int(cores)
    if not 1 <= team <= available:
        raise ResourceError("template cores must fit the current CPU affinity")
    contigs = tuple(contigs or ())
    if not contigs or len(set(contigs)) != len(contigs):
        raise InputError("templates need the unique contigs to model (--contigs)")
    header = vcf_reader.inspect(vcf)
    missing = [c for c in contigs if c not in header.contigs]
    if missing:
        raise InputError(f"{vcf}: contigs not in the header: {', '.join(missing)}")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise InputError("template output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    for contig in contigs:
        reads = vcf_reader.read_contig(vcf, contig, workers=team)
        positions, probabilities = contig_templates(
            reads, header.lengths.get(contig) or UNDECLARED_LENGTH_BP, team, reverse=reverse
        )
        if len(probabilities) == 0:
            continue
        with atomic_path(output / f"{contig}.npz") as temporary:
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, positions=positions, allele_probabilities=probabilities)
    return output


def templates_command(args: Any) -> int:
    """Run the templates command after the interface has configured the numerical runtime."""
    build_templates(
        args.vcf, args.output, contigs=tuple(args.contigs), cores=args.cores, reverse=getattr(args, "reverse", False)
    )
    return 0
