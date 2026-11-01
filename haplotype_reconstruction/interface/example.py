"""`haplotypes example DEST`: a portable, explicitly synthetic end-to-end example.

Writes DEST/templates/<contig>.npz (six founder haplotypes of independent Bernoulli(1/2) SNPs at evenly spaced
positions: a wiring example, not a biological benchmark), DEST/simulation.toml (a complete configuration that
simulates three small cohorts from them; checked by loading it) and DEST/README.txt.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def create_example(
    destination, *, seed: int = 400, contigs: int = 3, sites: int = 1200, length_bp: int = 12_000_000
) -> Path:
    """Write the example into ``destination`` (new or empty); returns the configuration path."""
    from ..core.errors import SettingsError
    from ..simulation.example import create_example as generate_example
    from .settings.load import load_settings, write_settings

    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise SettingsError("example destination must be new or empty")
    design = generate_example(seed=seed, contigs=contigs, sites=sites, length_bp=length_bp)
    templates = destination / "templates"
    templates.mkdir(parents=True, exist_ok=True)
    for name, template in zip(design.contigs, design.templates, strict=True):
        np.savez_compressed(
            templates / f"{name}.npz",
            positions=template.positions,
            allele_probabilities=template.allele_probabilities,
        )
    config = load_settings(
        options={
            "run": {"output": str(destination / "run")},
            "input": {"kind": "simulation", "contigs": list(design.contigs)},
            "simulation": {
                "seed": seed,
                "templates": str(templates),
                "depth": design.depth,
                "cohorts": list(design.cohorts),
            },
        }
    )
    path = write_settings(config, destination / "simulation.toml")
    load_settings(path)
    (destination / "README.txt").write_text(
        f"Synthetic {design.templates[0].allele_probabilities.shape[0]}-founder Bernoulli marker templates. "
        "Not a reference genome, "
        "empirical diversity model, or calibrated pedigree accuracy benchmark.\n"
        f"The {contigs} short chromosomes are an end-to-end software example.\n"
        f"Initialize: haplotypes init {path}\n"
        f"Run: haplotypes run {path}\n"
        f"Then: haplotypes evaluate {destination / 'run'}\n"
    )
    return path


def example_command(args) -> int:
    """`haplotypes example DEST [--seed N] [--contigs N] [--sites N] [--length-bp N]`."""
    path = create_example(
        args.destination, seed=args.seed, contigs=args.contigs, sites=args.sites, length_bp=args.length_bp
    )
    print(f"Example configuration: {path}", flush=True)
    return 0


__all__ = ["create_example", "example_command"]
