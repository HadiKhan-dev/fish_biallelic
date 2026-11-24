"""Bind the pedigree decision rules to explicit, immutable parameters.

The groups below are internal model constants, documented
here and in docs/configuration.md. Each group is bound into the identity of exactly the records it
affects: ``preparation`` into prepared factors, ``scoring`` into screens and scores, ``panel`` into the
M2 candidate panel, direction mismatch probability into scores, and the remaining decision parameters
into the genome-wide decision.

Thresholds label evidence tiers; they are not biological acceptance probabilities.

Method: docs/methods/pedigree.md#inputs-and-outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .decision_parameters import (
    DirectionParameters,
    ExposureParameters,
    ReleaseParameters,
    ResamplingParameters,
    SelectionParameters,
)
from .exclusion_parameters import ExclusionParameters
from .family_transmission_parameters import FamilyTransmissionParameters
from .map_scale_parameters import MapScaleParameters
from .material_parameters import MaterialParameters
from .order_parameters import OrderParameters
from .relationship_parameters import RelationshipParameters
from .scoring_parameters import PanelParameters, PreparationParameters, ScoringParameters


@dataclass(frozen=True)
class PedigreeParameters:
    """Internal pedigree model and execution parameters."""

    # The run's explicit tie convention (reverses arbitrary orders); never read from the environment.
    reverse_conventions: bool = field(
        default=False,
        metadata={
            "provenance": "PD27; boolean false selects the default explicit arbitrary tie convention.",
        },
    )
    preparation: PreparationParameters = field(
        default_factory=PreparationParameters,
        metadata={
            "provenance": (
                "PD27; internal marker-selection parameter group; units are declared by its fields."
            ),
        },
    )
    scoring: ScoringParameters = field(
        default_factory=ScoringParameters,
        metadata={
            "provenance": "PD27; internal projected-transmission parameter group; units are declared by its fields.",
        },
    )
    panel: PanelParameters = field(
        default_factory=PanelParameters,
        metadata={
            "provenance": "PD27; internal bounded-M2-search parameter group; units are declared by its fields.",
        },
    )
    exposure: ExposureParameters = field(
        default_factory=ExposureParameters,
        metadata={
            "provenance": "PD27; internal exposure-gate parameter group; units are declared by its fields.",
        },
    )
    direction: DirectionParameters = field(
        default_factory=DirectionParameters,
        metadata={
            "provenance": (
                "PD27; internal orientation parameter group; units are declared by its fields."
            ),
        },
    )
    order: OrderParameters = field(
        default_factory=OrderParameters,
        metadata={
            "provenance": (
                "PD27; internal birth-order sampler parameter group (sweeps, burn-in, numerical guards); units are "
                "declared by its fields."
            ),
        },
    )
    selection: SelectionParameters = field(
        default_factory=SelectionParameters,
        metadata={
            "provenance": "PD27; internal state-prior parameter group; units are declared by its fields.",
        },
    )
    resampling: ResamplingParameters = field(
        default_factory=ResamplingParameters,
        metadata={
            "provenance": "PD27; internal unit-bootstrap parameter group; units are declared by its fields.",
        },
    )
    release: ReleaseParameters = field(
        default_factory=ReleaseParameters,
        metadata={
            "provenance": "PD27; internal evidence-tier parameter group; units are declared by its fields.",
        },
    )
    exclusion: ExclusionParameters = field(
        default_factory=ExclusionParameters,
        metadata={
            "provenance": "PD27; internal full-marker exclusion parameter group; units are declared by its fields.",
        },
    )
    relationship: RelationshipParameters = field(
        default_factory=RelationshipParameters,
        metadata={"provenance": "PD27; painting relationship evidence parameters, with units declared by each field."},
    )
    family_transmission: FamilyTransmissionParameters = field(
        default_factory=FamilyTransmissionParameters,
        metadata={"provenance": "PD29; family transmission test parameters, with units declared by each field."},
    )
    material: MaterialParameters = field(
        default_factory=MaterialParameters,
        metadata={"provenance": "DESIGN_417; material units and regime cut-offs, with units declared by each field."},
    )
    map_scale: MapScaleParameters = field(
        default_factory=MapScaleParameters,
        metadata={
            "provenance": (
                "4.19 (DESIGN_RELEASE); the painting-based rate estimate and the mode of its use by the material "
                "regime, with units declared by each field."
            ),
        },
    )
