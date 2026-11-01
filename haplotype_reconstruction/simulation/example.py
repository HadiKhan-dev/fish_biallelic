"""Generate the explicitly synthetic onboarding example without writing files.

OU22 (docs/methods/simulation.md#example): six independent Bernoulli haplotypes
per contig, from SeedSequence(seed).spawn(C)[c]. One int8 integers(0,2,(H,L))
draw per contig; positions use the original integer linspace call. This is a
software wiring example, not a reference genome or calibrated diversity model.
The interface owns destination checks, NPZ/config/README writing and validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts.roles import record_field
from ..core.errors import SettingsError
from ..core.random import stream
from .parameters import EXAMPLE_COHORTS, EXAMPLE_DEPTH, EXAMPLE_HAPLOTYPES
from .source import FounderTemplate

__all__ = ["ExampleDesign", "create_example"]


@dataclass(frozen=True)
class ExampleDesign:
    """Generated templates and the original example's generating settings."""

    seed: int = record_field("obs", axes=(), units="1", dtype="int")
    contigs: tuple[str, ...] = record_field("obs", axes=(), units="1", dtype="tuple[str, ...]")
    templates: tuple[FounderTemplate, ...] = record_field(
        "obs", axes=(), units="1", dtype="tuple[FounderTemplate, ...]"
    )
    cohorts: tuple[int, ...] = record_field("obs", axes=(), units="1", dtype="tuple[int, ...]", default=EXAMPLE_COHORTS)
    depth: float = record_field("obs", axes=(), units="1", dtype="float", default=EXAMPLE_DEPTH)


def create_example(
    *, seed: int = 400, contigs: int = 3, sites: int = 1200, length_bp: int = 12_000_000
) -> ExampleDesign:
    """Return independent templates; reject fewer than 400 sites or length < sites.

    Inputs are scalar settings; outputs own int64[L] positions and float64[6,L,2]
    probabilities. No input, environment, file or global stream is modified.
    """
    if contigs < 1 or sites < 400 or length_bp < sites:
        raise SettingsError("example needs positive contigs, at least 400 sites and length >= sites")
    templates = []
    for index in range(contigs):
        rng = stream("templates.example", seed=seed, index=index)
        alleles = rng.integers(0, 2, (EXAMPLE_HAPLOTYPES, sites), dtype=np.int8)
        positions = np.linspace(1, length_bp, sites, dtype=np.int64)
        templates.append(FounderTemplate(positions, np.eye(2, dtype=float)[alleles]))
    return ExampleDesign(seed, tuple(f"chr{i + 1}" for i in range(contigs)), tuple(templates))
