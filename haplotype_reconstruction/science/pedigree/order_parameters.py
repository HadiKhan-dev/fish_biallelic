"""Parameters of the birth-order prior's sampler (R4, 4.22): sweeps, burn-in and the sampler's numerical guards.

The model has no tunable constant: every sample has an unknown birth order, all orders are equally likely, and parents
are born before their children (science/pedigree/order.py). The values below set only how long the exact Gibbs chain
runs and two numerical safeguards of its incremental moves; none changes the target distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...core.errors import SettingsError


@dataclass(frozen=True)
class OrderParameters:
    """Monte Carlo length and numerical guards of the birth-order chain run inside every pedigree fit."""

    burn_in_sweeps: int = field(
        default=100,
        metadata={
            "units": "sweeps (one sweep reinserts every sample once)",
            "provenance": (
                "R4 step 2 (lab422/r4/STEP2.md) and the sampler check (lab422/r4/sampler/SAMPLER.md section 6): from "
                "random initial orders the log target reaches its post-burn median within 1-7 sweeps on the six "
                "captured roots (16 chains each, split R-hat 0.9998-1.0004, integrated autocorrelation 1.0-2.5 "
                "sweeps); 100 leaves a wide margin. Discarded sweeps contribute nothing to the estimates."
            ),
        },
    )
    resample_sweeps: int = field(
        default=400,
        metadata={
            "units": "sweeps after burn-in",
            "provenance": (
                "R4 step 2's value for every bootstrap, leave-one-unit-out and sensitivity fit, with which the "
                "birth-order prior was gated; its Monte Carlo sd of a near-tie state-evidence margin is 0.12 nats "
                "(SAMPLER.md section 3), so a child whose support sits within one bootstrap draw of a release cut "
                "can differ between seeds. Cost is linear in sweeps (0.45-0.68 s per fit at N = 80-116, 3.0 s at "
                "N = 320 for 500 sweeps)."
            ),
        },
    )
    full_sweeps: int = field(
        default=2000,
        metadata={
            "units": "sweeps after burn-in",
            "provenance": (
                "R4 step 2's value for the full-data fits (the decision fit and the direction model's neutral fit, "
                "both of which record a trace); five times the resample length because their row posteriors and "
                "orientation log odds enter the release directly, not through bootstrap counts."
            ),
        },
    )
    cancellation_guard: float = field(
        default=1e-3,
        metadata={
            "units": "share of Z_c",
            "provenance": (
                "Numerical, not statistical. A move of sample s needs Z_c without the rows of c that name s; it is "
                "taken as the maintained Z_c minus those rows unless that keeps less than this share of Z_c (s is "
                "c's dominant parent), when it is recomputed exactly from c's rows. The subtraction's relative "
                "error is then at most about 2 machine-epsilon / guard = 4e-13. Exactness against enumeration: "
                "SAMPLER.md section 2. About 1.5 recomputes per child per sweep at every N measured."
            ),
        },
    )
    max_weight_span: float = field(
        default=600.0,
        metadata={
            "units": "nats",
            "provenance": (
                "Numerical, not statistical. Row weights are held in linear space relative to each child's best "
                "always-allowed row (its M0 or exposure-untestable row); a row more than this many nats above it "
                "raises the child's reference to keep every weight and the sum of a scan over N insertion positions "
                "finite in float64 (exp(709) is the largest finite value). Captured roots span at most 256 nats; "
                "a raised reference is counted in the chain diagnostics."
            ),
        },
    )

    def __post_init__(self) -> None:
        for name in ("burn_in_sweeps", "resample_sweeps", "full_sweeps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise SettingsError(f"order.{name} must be a non-negative integer, got {value!r}")
        if self.resample_sweeps < 1 or self.full_sweeps < 1:
            raise SettingsError("order.resample_sweeps and order.full_sweeps must be at least 1")
        if not 0.0 < self.cancellation_guard < 1.0:
            raise SettingsError(f"order.cancellation_guard must lie in (0, 1), got {self.cancellation_guard!r}")
        if not 1.0 <= self.max_weight_span <= 700.0:
            raise SettingsError(f"order.max_weight_span must lie in [1, 700] nats, got {self.max_weight_span!r}")
