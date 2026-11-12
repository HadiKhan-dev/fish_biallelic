"""Write a portable coalescent configuration for the end-to-end example."""

from __future__ import annotations

from pathlib import Path


def create_example(
    destination, *, seed: int = 400, contigs: int = 3, sites: int = 1200, length_bp: int = 12_000_000
) -> Path:
    """Write a checked count-mode configuration into a new or empty directory."""
    import tomllib
    from importlib import resources

    from ..core.errors import SettingsError
    from ..simulation.parameters import EXAMPLE_COHORTS, EXAMPLE_DEPTH
    from .settings.load import load_settings, write_settings

    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in (contigs, sites, length_bp)):
        raise SettingsError("contigs, sites and length_bp must be positive integers")
    if sites > length_bp:
        raise SettingsError("sites exceeds the number of integer genome positions")
    destination = Path(destination).resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise SettingsError("example destination must be new or empty")
    names = [f"chr{i + 1}" for i in range(contigs)]
    mapping = tomllib.loads(
        resources.files("haplotype_reconstruction.interface.settings")
        .joinpath("presets", "example_coalescent.toml")
        .read_text()
    )
    mapping.update(run={"output": str(destination / "run")}, input={"kind": "simulation", "contigs": names})
    mapping["simulation"].update(
        seed=seed,
        depth=EXAMPLE_DEPTH,
        cohorts=list(EXAMPLE_COHORTS),
        genome={"contigs": names, "length_bp": length_bp},
        sites={"count": sites},
    )
    config = load_settings(options=mapping)
    path = write_settings(config, destination / "simulation.toml")
    load_settings(path)
    (destination / "README.txt").write_text(
        "Synthetic coalescent source with three diploid cross founders.\n"
        "The configuration is a software example, with an explicit effective size and site count.\n"
        f"Initialize: haplotypes init {path}\nRun: haplotypes run {path}\n"
        f"Then: haplotypes evaluate {destination / 'run'}\n"
    )
    return path


def example_command(args) -> int:
    """Create the example while retaining the seed, contigs, sites and length flags."""
    path = create_example(
        args.destination, seed=args.seed, contigs=args.contigs, sites=args.sites, length_bp=args.length_bp
    )
    print(f"Example configuration: {path}", flush=True)
    return 0


__all__ = ["create_example", "example_command"]
