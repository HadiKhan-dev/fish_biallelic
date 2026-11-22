"""Parameters for painting-based pedigree relationship evidence."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...core.errors import SettingsError


@dataclass(frozen=True)
class RelationshipParameters:
    """Painting relationship process and score (PD28).

    Rates are per Morgan, lengths in Morgans, isolated_error scales chance-infeasibility per known bin,
    withholding_score in nats, sibship_prior probabilities in increasing
    support bands, support_bounds dimensionless shares, reference_triples and
    sibship_check_partners counts.
    """

    failure_region_rate: float = field(
        default=0.1,
        metadata={
            "units": "1/Morgan",
            "provenance": "PD28; one observed painting failure in fourteen Morgans motivates 0.1 regions per Morgan.",
        },
    )
    failure_region_length: float = field(
        default=0.05,
        metadata={
            "units": "Morgan",
            "provenance": "PD28; observed painting failures span 4–8 cM, shorter than full-sib IBD0 segments.",
        },
    )
    isolated_error: float = field(
        default=1e-3,
        metadata={
            "units": "probability/bin",
            "provenance": (
                "PD28; scale 1-q per bin; at most 0.03% wrong bins; no uninformative-bin penalty. 4.21 (ADVICE2 "
                "11.1): the FLOOR of the per-unit, per-bin error min(1/2, max(0.001, sum of the members' u)), u the "
                "reference painting's probability that a member's released pair is wrong."
            ),
        },
    )
    withholding_score: float = field(
        default=float(np.log(1000.0)),
        metadata={
            "units": "nats",
            "provenance": (
                "PD23; log(1000) survives higher-prior sensitivity; with the 4.18 priors one top-band witness "
                "withholds only with a pair margin below -2.31 nats."
            ),
        },
    )
    sibship_prior: tuple[float, ...] = field(
        default=(0.67, 0.96, 0.99),
        metadata={
            "units": "probability",
            "provenance": (
                "PD28 / DESIGN_417 D5; full-sib shares of the guard's witness occurrences in three support bands, "
                "support on pooled Morgan-weighted 1-M units, pooled over three known-truth sets (4,159 "
                "occurrences, 73 runs; run-bootstrap intervals 0.54-0.79, 0.91-0.98, 0.97-1.00); every held-out "
                "fit changed no guard decision (RELCAL)."
            ),
        },
    )
    support_bounds: tuple[float, ...] = field(
        default=(0.5, 0.75),
        metadata={
            "units": "share",
            "provenance": "PD28; support bands of the sibship calibration, unchanged in 4.18 (DESIGN_417 D5).",
        },
    )
    sibship_check_partners: int = field(
        default=16,
        metadata={
            "units": "count",
            "provenance": "PD28; bounded ring check; 8 and 16 partners match all-partner validation scores.",
        },
    )
    reference_triples: int = field(
        default=400,
        metadata={
            "units": "count",
            "provenance": "PD28; 400 reference triples give about 0.02 sampling error in bin infeasibility.",
        },
    )

    def __post_init__(self) -> None:
        import math

        for name in ("failure_region_rate", "failure_region_length"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise SettingsError(f"pedigree relationship {name} must be finite and positive")
        if not 0 < self.isolated_error < 1:
            raise SettingsError("pedigree relationship isolated_error must lie in (0, 1)")
        if not math.isfinite(self.withholding_score) or self.withholding_score <= 0:
            raise SettingsError("pedigree relationship withholding_score must be finite and positive")
        if (
            len(self.sibship_prior) != 3
            or any(not 0 < p < 1 for p in self.sibship_prior)
            or tuple(sorted(self.sibship_prior)) != self.sibship_prior
        ):
            raise SettingsError("pedigree relationship sibship_prior must contain three increasing probabilities")
        if len(self.support_bounds) != 2 or not 0 <= self.support_bounds[0] < self.support_bounds[1] <= 1:
            raise SettingsError("pedigree relationship support_bounds must contain two increasing shares")
        if (
            isinstance(self.sibship_check_partners, bool)
            or not isinstance(self.sibship_check_partners, int)
            or self.sibship_check_partners < 1
        ):
            raise SettingsError("pedigree relationship sibship_check_partners must be a positive integer")
        if (
            isinstance(self.reference_triples, bool)
            or not isinstance(self.reference_triples, int)
            or self.reference_triples < 1
        ):
            raise SettingsError("pedigree relationship reference_triples must be a positive integer")
