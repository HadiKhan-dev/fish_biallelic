"""`haplotypes example DEST`: a portable, explicitly synthetic end-to-end example.

Writes DEST/templates/<contig>.npz (six founder haplotypes of independent Bernoulli(1/2) SNPs at evenly spaced
positions: a wiring example, not a biological benchmark), DEST/simulation.toml (a complete configuration that
simulates three small cohorts from them; checked by loading it) and DEST/README.txt.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FOUNDER_HAPLOTYPES = 6
COHORTS = (20, 30, 30)
DEPTH = 5.0


def create_example(destination, *, seed: int = 400, contigs: int = 3, sites: int = 1200,
                   length_bp: int = 12_000_000) -> Path:
    """Write the example into ``destination`` (new or empty); returns the configuration path."""
    from ..config import load_config, write_config
    if contigs < 1 or sites < 400 or length_bp < sites:
        raise ValueError("example needs positive contigs, at least 400 sites and length >= sites")
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("example destination must be new or empty")
    templates = destination / "templates"
    templates.mkdir(parents=True, exist_ok=True)
    names = [f"chr{index + 1}" for index in range(contigs)]
    for name, child in zip(names, np.random.SeedSequence(seed).spawn(contigs)):
        rng = np.random.default_rng(child)
        alleles = rng.integers(0, 2, (FOUNDER_HAPLOTYPES, sites), dtype=np.int8)
        positions = np.linspace(1, length_bp, sites, dtype=np.int64)
        np.savez_compressed(templates / f"{name}.npz", positions=positions,
                            allele_probabilities=np.eye(2, dtype=float)[alleles])
    config = load_config(options={
        "run": {"output": str(destination / "run")},
        "input": {"kind": "simulation", "contigs": names},
        "simulation": {"seed": seed, "templates": str(templates), "depth": DEPTH, "cohorts": list(COHORTS)},
    })
    path = write_config(config, destination / "simulation.toml")
    load_config(path)
    (destination / "README.txt").write_text(
        f"Synthetic {FOUNDER_HAPLOTYPES}-founder Bernoulli marker templates. Not a reference genome, "
        "empirical diversity model, or calibrated pedigree accuracy benchmark.\n"
        f"The {contigs} short chromosomes are an end-to-end software example.\n"
        f"Run: haplotypes run {path}\n"
        f"Then: haplotypes evaluate {destination / 'run'}\n")
    return path


def example_command(args) -> int:
    """`haplotypes example DEST [--seed N] [--contigs N] [--sites N] [--length-bp N]`."""
    path = create_example(args.destination, seed=args.seed, contigs=args.contigs, sites=args.sites,
                          length_bp=args.length_bp)
    print(f"Example configuration: {path}", flush=True)
    return 0
