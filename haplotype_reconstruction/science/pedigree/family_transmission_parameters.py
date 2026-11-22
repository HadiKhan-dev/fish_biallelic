"""Parameters for the family transmission vote."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...core.errors import SettingsError


@dataclass(frozen=True)
class FamilyTransmissionParameters:
    """PD29 family transmission test of retained Tier B parent edges (after the sibling guard's fixed point)."""

    role_prior: float = field(
        default=0.01,
        metadata={
            "units": "probability",
            "provenance": (
                "PD29; prior of each non-child role of a witness (a half child through either latent parent, a "
                "direct observation of a latent parent, or an outsider explained by its own mosaic); refitted per "
                "1-Morgan segment, so the prior alone charges at most ln((1 - 5 rho) / rho) = 4.55 nats per segment "
                "for a witness that is not a full sib (the decision rests on PAR - ALT). Fixed before the held-out "
                "runs (sibpar_tx VALIDATION_PLAN; half-child role: VALIDATION_PLAN_419b amendment 3)."
            ),
        },
    )
    minimum_witnesses: int = field(
        default=2,
        metadata={
            "units": "count",
            "provenance": (
                "PD29; edges with fewer retained witnesses (families of fewer than three children) are untested: "
                "with one witness the evidence is near-pairwise and a single non-sib witness dominates it (sibpar_tx "
                "R1 withheld 2 of 38,601 correct Tier A instances there; released sibs scored +10 to +18)."
            ),
        },
    )
    maximum_witnesses: int = field(
        default=12,
        metadata={
            "units": "count",
            "provenance": (
                "PD29; witness cap bounding the per-edge cost (fits are linear to quadratic in the witness count); "
                "witnesses are taken by relatedness member support, then observed-site counts in configured contig "
                "order (the PD24 convention), names last."
            ),
        },
    )
    maximum_iterations: int = field(
        default=30,
        metadata={"units": "count", "provenance": "PD29; MAP-EM iteration bound per fit (each step is monotone)."},
    )
    start_rounds: int = field(
        default=3,
        metadata={
            "units": "count",
            "provenance": "PD29; rounds of cross-initialisation between the three models (stops when none improves).",
        },
    )
    random_starts: int = field(
        default=3,
        metadata={
            "units": "count",
            "provenance": (
                "PD29; data-seeded block perturbations of each model's best paths per round (the review's restarts "
                "raised segment likelihoods by up to 32 nats without them)."
            ),
        },
    )

    def __post_init__(self) -> None:
        import math

        if not math.isfinite(self.role_prior) or not 0 < self.role_prior < 1 / 5:
            raise SettingsError("pedigree family_transmission role_prior must lie in (0, 1/5)")
        for name in ("minimum_witnesses", "maximum_witnesses", "maximum_iterations", "start_rounds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise SettingsError(f"pedigree family_transmission {name} must be a positive integer")
        if isinstance(self.random_starts, bool) or not isinstance(self.random_starts, int) or self.random_starts < 0:
            raise SettingsError("pedigree family_transmission random_starts must be a nonnegative integer")
        if self.maximum_witnesses < self.minimum_witnesses:
            raise SettingsError("pedigree family_transmission maximum_witnesses must be at least minimum_witnesses")
