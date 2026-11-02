"""Share typed boundary records and storage conventions across the 4.0 stages.

Axes: N samples, L sites, B blocks, K founders, R local rows, C components,
M bins (or source sites where explicitly stated), E edges, P pairs, F families.
Positions are int64 1-based bp; genetic distances float64 Morgans; log values
float64 nats. Alleles are int8 (0 REF, 1 ALT, -1 withheld); absent indices are -1.
Missing-read likelihoods are exactly thirds. Arrays shared between tasks are
read-only. Scientific fallback/search status is state or call, never diagnostic.

Ordinary frozen dataclasses declare role/axes/units/dtype/stored with record_field.
Products declare schema_field(1); nested records need no schema. Derived fields
and views never enter payloads; adapters supply their documented reconstruction
inputs. Source and stage modules define their own records; this initializer
exposes only the shared boundaries and never imports science or storage.
"""

from .genetic_map import ChromosomeGeneticMap, GeneticMap, GeneticMapSet, measured_coordinates
from .panels import (
    ComponentPanel,
    CountCall,
    CountStepDecision,
    FounderComponent,
    LocalPanel,
    PartnerChange,
    PreparedPanel,
    materialise,
    partner_changes,
    relabel_rows,
)
from .payload import from_payload, to_payload
from .release import ReleasedSites
from .roles import FieldRole, FieldSpec, field_specs, record_field, require_schema, schema_field
from .source import ContigReads, ObservationSource

__all__ = [
    "ChromosomeGeneticMap",
    "GeneticMap",
    "GeneticMapSet",
    "measured_coordinates",
    "ComponentPanel",
    "CountCall",
    "CountStepDecision",
    "FounderComponent",
    "LocalPanel",
    "PartnerChange",
    "PreparedPanel",
    "materialise",
    "partner_changes",
    "relabel_rows",
    "from_payload",
    "to_payload",
    "ReleasedSites",
    "FieldRole",
    "FieldSpec",
    "field_specs",
    "record_field",
    "require_schema",
    "schema_field",
    "ContigReads",
    "ObservationSource",
]
