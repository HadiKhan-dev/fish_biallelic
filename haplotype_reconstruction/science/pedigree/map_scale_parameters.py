"""Parameters for the painting-based map scale estimate."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...core.errors import SettingsError

# [genetic_map] rate_estimate: what the estimated recombination rates may change (DESIGN_RELEASE section 1).
RATE_ESTIMATE_MODES = ("veto", "ignore", "adopt")


@dataclass(frozen=True)
class MapScaleParameters:
    """The painting-based genome-wide recombination rate and its use by the material rule (4.19, DESIGN_MAPSCALE
    rev 5, DESIGN_RELEASE).

    One rate in cM/Mb (Morgans per bp = rate 1e-8) for the fallback stretches of the working map (contigs without a
    map file, stretches beyond a file's knots), estimated after painting from per-sib transmission frames of the
    relatedness families (error-run chain on the painting's own emissions). It never changes the working map of any
    stage; it decides only which material regime the pedigree applies (``mode``). Grids and guard constants are the
    rev-5 frozen estimator's (md5 404a44b6); the gate replaces rev 5's contig rule by information and material
    criteria (owner, 13 Nov: no count of chromosomes decides).
    """

    mode: str = field(
        default="veto",
        metadata={
            "provenance": (
                "Owner, 14 Nov 2026 ([genetic_map] rate_estimate): veto (default) releases only what both the working "
                "map's and the estimated map's regimes release; ignore reports the estimate only; adopt uses the "
                "estimated map's regime alone when its interval excludes the working rate (test-agree, review3 "
                "MUST-FIX 2), else both."
            ),
        },
    )
    guard_nats: float = field(
        default=10.0,
        metadata={
            "units": "nats",
            "provenance": (
                "DESIGN_MAPSCALE rev 4 G4: both relationship tests (pair, target) need a log-likelihood ratio above "
                "10 nats (p about 1e-5 on the chi-square(1) scale) to keep a pair or remove a target."
            ),
        },
    )
    target_share: float = field(
        default=0.2,
        metadata={
            "units": "fraction",
            "provenance": (
                "Rev 4 G4 (ii): a target is removed only when at least 20 % of its frames are posterior non-full-sib; "
                "the share criterion prevents selection on crossover counts (review R4 note 1)."
            ),
        },
    )
    target_iterations: int = field(
        default=12,
        metadata={"provenance": "Rev 4: at most 12 target-test rounds (one removal per family and round)."},
    )
    rate_grid: tuple[float, float, int] = field(
        default=(0.1, 50.0, 141),
        metadata={
            "units": "cM/Mb",
            "provenance": (
                "Rev 2-5: log-spaced rate grid (low, high, points) of the misfit-weight fit; the error-run chain "
                "uses every second point (71, log step 0.089). An estimate at either end is at the grid edge."
            ),
        },
    )
    coarse_rate_grid_spec: tuple[float, float, int] = field(
        default=(0.2, 50.0, 29),
        metadata={
            "units": "cM/Mb",
            "provenance": "Rev 4: the coarse log grid (low, high, points) of the relationship tests' nuisance rate.",
        },
    )
    misfit_weights: tuple[float, ...] = field(
        default=(0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 0.03, 0.1, 0.3),
        metadata={
            "units": "probability",
            "provenance": (
                "Rev 3: weight delta of the misfit column (mean over all state pairs: panel or painting misfit) "
                "mixed into each frame emission; profiled by maximum likelihood."
            ),
        },
    )
    pair_error_rates: tuple[float, ...] = field(
        default=(0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
        metadata={"units": "probability", "provenance": "Rev 2: observation error grid of the pair-test chains."},
    )
    target_entry_rates: tuple[float, ...] = field(
        default=(0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1),
        metadata={
            "units": "probability per frame",
            "provenance": "Rev 4 G4 (ii): entry grid of a track leaving its parents' classes (0 = the full-sib chain).",
        },
    )
    target_exit_rates: tuple[float, ...] = field(
        default=(3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3),
        metadata={"units": "probability per frame", "provenance": "Rev 4 G4 (ii): exit grid of such an excursion."},
    )
    error_run_starts: tuple[float, ...] = field(
        default=(0.0, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2),
        metadata={
            "units": "probability per frame",
            "provenance": (
                "Rev 4 er: start probability eps of an error run (the shown state leaves the true transmission "
                "state); eps = 0 is the 4-state chain; profiled by maximum likelihood."
            ),
        },
    )
    error_run_ends: tuple[float, ...] = field(
        default=(0.02, 0.1, 0.4),
        metadata={"units": "probability per frame", "provenance": "Rev 4 er: end probability xi of an error run."},
    )
    consistency_alpha: float = field(
        default=0.01,
        metadata={
            "units": "probability",
            "provenance": (
                "Rev 5 R5.2: family-wise level of the family-consistency test (Bonferroni over the active families, "
                "t(df) reference of the quasi-likelihood z, df from the family-within-unit deviance on the material "
                "units, 4.21 C11; rev 5 used contigs). Null calibration on units: 1 of 65 development genomes refused "
                "(one-sided 95 % upper bound 7 %; the genome contig strata also refuse), 1 of 126 simulated genomes; "
                "null families beyond the t(df) 0.995 quantile 0.010."
            ),
        },
    )
    drop_margin: float = field(
        default=1.25,
        metadata={
            "provenance": (
                "Rev 4/5: with three or more families the most deviant family is dropped when |z| exceeds 1.25 "
                "z_crit (an identifiable outlier, e.g. a guard failure), and the test repeats."
            ),
        },
    )
    bootstrap_replicates: int = field(
        default=2000,
        metadata={"provenance": "Rev 5 B_J: percentile-bootstrap replicates of the joint interval (count)."},
    )
    minimum_interval_units: int = field(
        default=5,
        metadata={
            "provenance": (
                "The joint interval needs at least five equal-material units (working Morgans of frame axis) and two "
                "families: a percentile bootstrap over fewer units is degenerate. The same minimum as the relatedness "
                "estimate's (RateParameters.minimum_interval_units) and the pedigree's assessable material; it "
                "replaces rev 5's >= 5 contigs (owner, 13 Nov 2026). On the 195 lab genomes no adopted or usable "
                "estimate had fewer than 5 units (matrix f3: 3 units, refused by R2 and the usable floor)."
            ),
        },
    )
    bootstrap_seed: int = field(
        default=20261114,
        metadata={"provenance": "Fixed seed of the joint bootstrap (integer, rev 5's); the interval is reproducible."},
    )
    minimum_events: float = field(
        default=60.0,
        metadata={
            "units": "expected crossovers",
            "provenance": (
                "Rev 4/5 R2: adoption needs 60 expected crossovers on the kept families (2 x frame span x rate): "
                "about a 25 % half-width (R3) at Poisson precision; the information criterion that replaces rev 5's "
                ">= 5 contigs together with the material-unit interval (DESIGN_RELEASE section 4)."
            ),
        },
    )
    maximum_half_width: float = field(
        default=0.25,
        metadata={
            "units": "fraction",
            "provenance": "Rev 4/5 R3: adoption needs a joint 95 % interval half-width of at most 25 % of the estimate.",
        },
    )
    usable_minimum_events: float = field(
        default=20.0,
        metadata={
            "units": "expected crossovers",
            "provenance": (
                "Review3 SHOULD-FIX 4: a non-adopted estimate may still take part in the veto only with 20 expected "
                "crossovers and an interval whose lower end lies above the grid floor; every dev and held-out "
                "fallback had >= 46, the two V5 scaffold cells that decided on 19 and 4 events (interval lower end on "
                "the floor) are refused."
            ),
        },
    )

    def __post_init__(self) -> None:
        if self.mode not in RATE_ESTIMATE_MODES:
            raise SettingsError(f"rate_estimate must be one of {', '.join(RATE_ESTIMATE_MODES)}")

    def rate_grid_values(self) -> np.ndarray:
        """The fine log grid (cM/Mb)."""
        low, high, points = self.rate_grid
        return np.exp(np.linspace(np.log(low), np.log(high), int(points)))

    def error_run_grid(self) -> np.ndarray:
        """The error-run chain's rate grid: every second fine point (cM/Mb), contiguous."""
        return np.ascontiguousarray(self.rate_grid_values()[::2])

    def coarse_rate_grid(self) -> np.ndarray:
        """The relationship tests' coarse log grid (cM/Mb)."""
        low, high, points = self.coarse_rate_grid_spec
        return np.exp(np.linspace(np.log(low), np.log(high), int(points)))
