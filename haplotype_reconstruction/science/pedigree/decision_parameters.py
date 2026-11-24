"""Pedigree exposure, direction, selection, resampling and evidence tier parameters."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ExposureParameters:
    """Hard exposure gate: a configuration is testable only on enough exposed evidence."""

    minimum_edge_exposed_bins: float = field(
        default=1.0,
        metadata={
            "provenance": "PD05; require at least 1 posterior-expected exposed painting bin for an edge.",
        },
    )
    minimum_pair_exposed_bins: float = field(
        default=1.0,
        metadata={
            "provenance": "PD05; require at least 1 posterior-expected exposed painting bin for a parent pair.",
        },
    )
    # Exposed bins / total painting bins of the fit's units.
    minimum_exposed_fraction: float = field(
        default=0.1,
        metadata={
            "provenance": "PD05; require an exposed-bin fraction of at least 0.1 across the fit's units.",
        },
    )
    # Material units (MaterialParameters: 1-Morgan units of the working map) a fit's exposed units must number
    # (bootstrap draws count once). Every unit counts one, whatever chromosome or component it lies on; the
    # low-material band (U < MaterialParameters.standard_morgans) needs one.
    minimum_exposed_units: float = field(
        default=3.0,
        metadata={
            "provenance": (
                "PD05 / DESIGN_417 D2; require 3 exposed material units (1-Morgan units of the working map) in the "
                "standard regime; 1 in the low-material band."
            ),
        },
    )
    # Report-only thresholds of the C/X compatibility diagnostics (never gate a decision).
    report_edge_coverage: float = field(
        default=0.95,
        metadata={
            "provenance": "PD05; fraction 0.95 labels the edge-coverage diagnostic; does not gate decisions.",
        },
    )
    report_pair_explainability: float = field(
        default=0.95,
        metadata={
            "provenance": "PD05; fraction 0.95 labels the pair-explainability diagnostic; does not gate decisions.",
        },
    )


@dataclass(frozen=True)
class DirectionParameters:
    """Joint-transmission orientation and its pair-level failure mixture (the birth-order layer reads both)."""

    mismatch_probability: float = field(
        default=0.01,
        metadata={
            "provenance": (
                "PS19/PD06; the FLOOR (4.21, ADVICE2 11.1) of the per-bin probability that a meiotic copy has a "
                "different painted local class: the direction kernels use min(1/2, max(0.01, v_child + v_parent)), "
                "v the reference painting's per-track label error; the family transmission vote (4.22 R4b) bounds "
                "each error pattern of a member's two labels below by its independence value at this floor (one "
                "wrong label 0.01 (1 - 0.01), both 0.01^2) and gives a wrong label's whole mass to whichever class a "
                "hypothesis needs, so an over-confident posterior is never more decisive than 4.20 in either. "
                "Decision replays at 0.001, 0.01 and 0.1, all with the 4.21 vote algebra (before R4b), agreed at "
                "Tier B for a real multi-generation cross, the TroMau cross and large known-truth simulations; 0.1 "
                "added false full-fit edges in a 30-founder known-truth simulation."
            ),
        },
    )

    # The probability that one unit's junction comparison of a pair is painting failure is not a parameter here:
    # it follows from the relationship model's two-state failure process (RelationshipParameters
    # failure_region_rate and failure_region_length) and the unit's working length (orientation.unit_failure,
    # DESIGN_417 D8). It protects against short failures only; the window cap below bounds long ones.
    # The reliability of a decisive unit contrast and the share of linked pairs whose units are fair coins are not
    # parameters either: each run estimates them from its own linked pairs (orientation.direction_model: posterior
    # means under uniform priors; no constant of this record enters except pair_failure_probability, which bounds
    # the structural orientation used as the orientation prior).
    window_morgans: float = field(
        default=0.0,
        metadata={
            "units": "Morgan",
            "provenance": (
                "DESIGN_417 rev 4; width W of the direction windows (consecutive windows of W working Morgans of "
                "the unit axis) whose summed direction log odds are capped at window_cap; 0 (default) turns the "
                "cap off. A count-free bound on a long (chromosome-wide) painting failure's votes, measured in V5 "
                "with a positive control before the owner chooses W."
            ),
        },
    )
    window_cap: float = field(
        default=float(np.log(39.0)),
        metadata={
            "units": "nats",
            "provenance": (
                "DESIGN_417 rev 4; ln 39 = 3.66 nats, 4.16.1's per-chromosome cap (failure probability 0.05), "
                "below the release cut-off ln 49, so one window cannot orient an edge alone."
            ),
        },
    )
    # Pair-level painting-failure mixture bounding the orientation correction to
    # [log(epsilon), log(2 - epsilon)]; the same probability bounds the structural orientation that serves as the
    # orientation prior of the run's direction model (orientation.structural_prior). Independent of
    # SelectionParameters.m0_contamination_probability, which has the same default but a different role.
    pair_failure_probability: float = field(
        default=0.02,
        metadata={
            "provenance": (
                "PD06; pair-level failure probability 0.02 bounds the orientation correction, and the orientation "
                "prior that the run's direction model reads from the direction-free fit (within [0.01, 0.99])."
            ),
        },
    )


@dataclass(frozen=True)
class SelectionParameters:
    """Marginal parent-state priors and M0 contamination."""

    state_priors: tuple[float, float, float] = field(
        default=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        metadata={
            "provenance": (
                "PD04/PD12/PD13/PD14; dimensionless M0/M1/M2 probabilities of one third each before evidence."
            ),
        },
    )
    # Alternative fixed priors; an M0 call must survive all of them to be released.
    prior_sensitivity: tuple[tuple[float, float, float], ...] = field(
        default=((0.5, 0.3, 0.2), (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), (0.2, 0.3, 0.5)),
        metadata={
            "provenance": (
                "PD04/PD12/PD13/PD14; three fixed dimensionless M0/M1/M2 prior vectors; released M0 must survive "
                "all."
            ),
        },
    )
    # Each aggregated M1/M2 likelihood is mixed with the child's M0 likelihood at this weight.
    m0_contamination_probability: float = field(
        default=0.02,
        metadata={
            "provenance": (
                "PD04/PD12/PD13/PD14; probability weight 0.02 mixes child M0 into aggregated M1/M2 likelihoods."
            ),
        },
    )


@dataclass(frozen=True)
class ResamplingParameters:
    """Material-unit bootstrap (seeded) and leave-one-unit-out (LOBO) fits."""

    bootstrap_replicates: int = field(
        default=1000,
        metadata={
            "provenance": "PD16; 1000 unit-bootstrap draws for configuration and parent support fractions.",
        },
    )
    bootstrap_seed: int = field(
        default=20260725,
        metadata={
            "provenance": "PD16; integer seed 20260725 fixes the unit-bootstrap random stream.",
        },
    )


@dataclass(frozen=True)
class TierThresholds:
    """Release cut-offs of one evidence tier (fractions of resampled fits)."""

    configuration_bootstrap: float = field(
        metadata={
            "provenance": "PD18; minimum fraction of bootstrap fits selecting the configuration for this release tier.",
        },
    )
    parent_bootstrap: float = field(
        metadata={
            "provenance": (
                "PD18; minimum fraction of bootstrap fits selecting the parent identity for this release tier."
            ),
        },
    )
    lobo_fraction: float = field(
        metadata={
            "provenance": "PD18; minimum fraction of leave-one-unit-out (LOBO) fits supporting this release tier.",
        },
    )


@dataclass(frozen=True)
class ReleaseParameters:
    """Tier A / Tier B release rules (Tier B is the primary product)."""

    # A tier call needs the child's own per-unit state evidence to pick a unique winner on at least this many
    # material units (1 in the low-material band).
    minimum_informative_units: float = field(
        default=3.0,
        metadata={
            "provenance": (
                "PD18/PD20/PD21 / DESIGN_417 D2; require 3 material units with a unique child-evidence state winner."
            ),
        },
    )
    tier_a: TierThresholds = field(
        default_factory=lambda: TierThresholds(0.95, 0.95, 0.95),
        metadata={
            "provenance": (
                "PD18/PD20/PD21; dimensionless configuration-bootstrap, parent-bootstrap and leave-one-out "
                "cutoffs (0.95, 0.95, 0.95)."
            ),
        },
    )
    tier_b: TierThresholds = field(
        default_factory=lambda: TierThresholds(0.7, 0.7, 0.8),
        metadata={
            "provenance": (
                "PD18/PD20/PD21; primary dimensionless configuration-bootstrap, parent-bootstrap and "
                "leave-one-out cutoffs (0.7, 0.7, 0.8)."
            ),
        },
    )
    # A released edge needs full-data orientation log odds >= log((1 - p) / p) (3.9 nats).
    orientation_reversal_probability: float = field(
        default=0.02,
        metadata={
            "provenance": (
                "PD18/PD20/PD21; reversal probability 0.02 gives the edge orientation cutoff log((1-p)/p) in "
                "nats."
            ),
        },
    )
    # Candidate sets carry this much configuration probability mass.
    support_set_coverage: float = field(
        default=0.95,
        metadata={
            "provenance": (
                "PD18/PD20/PD21; probability mass 0.95 retained by configuration candidate sets, including "
                "boundary ties."
            ),
        },
    )
