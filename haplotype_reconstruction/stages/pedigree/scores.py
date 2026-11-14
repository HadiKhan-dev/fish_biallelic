"""Publish each informative contig's pieces of the genome's material units (PS16-PS19; DESIGN_417 I1-I3)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import Any

from ...contracts.payload import from_payload, to_payload
from ...contracts.pedigree import (
    CandidatePanel,
    ChromosomeScores,
    ChromosomeSummaryView,
    CompactM1,
    ParentEligibility,
    PreparedChromosome,
)
from ...contracts.roles import require_schema
from ...core.digest import array_digest
from ...core.errors import InputError
from ...engine.cpu import claim_threads, thread_team
from ...science.pedigree.block_scoring import score
from ...science.pedigree.parameters import PedigreeParameters
from ...store.container import CheckpointFile
from ...store.identity import code_dependency_identity
from ...store.protocol import Derivation, NumericalPolicy
from ..framework import load_product, publish_product

KEY = "pedigree/floored_scores"


class ScoresAdapter:
    """Schema-5 piece record (one form for every run); 4.16.x chromosome-form and block-form records (schema 4)
    are refused, so a resumed run re-scores its contigs."""

    schema = 5

    def encode(self, value: ChromosomeScores) -> Mapping[str, Any]:
        return {"schema": self.schema, **to_payload(value)}

    def decode(self, payload: Mapping[str, Any]) -> ChromosomeScores:
        require_schema(payload, self.schema)
        names = {f.name for f in fields(ChromosomeScores)}
        return from_payload(ChromosomeScores, {k: v for k, v in payload.items() if k in names})


def derivation(
    parameters: PedigreeParameters,
    prepared_identity: str,
    screen_identity: str,
    panel: CandidatePanel,
    placement: Any,
    *,
    reverse: bool | None = None,
    numerical_policy: NumericalPolicy | None = None,
) -> str:
    """Bind what scoring reads: the scoring parameters, the partition rule and this contig's placement on the
    genome axis (offset, last unit and cut list), the fixed screen and the canonical panel rows. The regime and the
    release thresholds are not bound (a change of cut-off re-scores nothing). The cut list depends on every earlier
    informative contig's layout; the screen identity binds every informative contig's preparation and order."""
    reverse = parameters.reverse_conventions if reverse is None else reverse
    material = parameters.material
    return Derivation(
        code_dependency_identity((__name__,)),
        dict(
            scoring=parameters.scoring,
            direction_mismatch_probability=parameters.direction.mismatch_probability,
            score_schema=ScoresAdapter.schema,
            m1_values="fixed-genome-profile",
            partition=(material.unit_morgans, material.tail_morgans, material.switch_clip,
                       "genome-axis-pieces-of-material-units"),
            placement=dict(offset=float(placement.offset), last=int(placement.last),
                           cuts=array_digest(placement.cuts)),
        ),
        numerical_policy or NumericalPolicy.current(),
        dict(prepared=prepared_identity, screen=screen_identity, trios=array_digest(panel.trios)),
        reverse,
    ).identity


def run(
    prepared: PreparedChromosome,
    compact: CompactM1,
    chromosome: int,
    panel: CandidatePanel,
    eligibility: ParentEligibility,
    parameters: PedigreeParameters,
    placement: Any,
    *,
    threads: int = 1,
    reverse: bool | None = None,
    file: CheckpointFile | None = None,
    identity: str | None = None,
) -> ChromosomeScores:
    """Score this contig's pieces of the genome units from its placement (offset, last unit, cut list)."""
    if (file is None) != (identity is None):
        raise InputError("publication requires both file and identity")
    adapter = ScoresAdapter()
    if file is not None:
        assert identity is not None
        previous = load_product(file, KEY, identity, adapter)
        if previous is not None:
            return previous
    with thread_team(default=threads):
        # claim_threads: the science's kernel boundaries re-read this task's share (it grows as peers finish).
        product = score(prepared, compact, chromosome, panel, eligibility, parameters, placement.offset,
                        placement.last, claim_threads, reverse=reverse, cuts=placement.cuts)
    if file is not None:
        assert identity is not None
        publish_product(file, KEY, identity, product, adapter)
    return adapter.decode(adapter.encode(product))


def chromosome_summary(prepared: tuple[PreparedChromosome, ...]) -> tuple[ChromosomeSummaryView, ...]:
    """Supply numeric summary rows on the configured contig axis, including omissions."""
    return tuple(
        ChromosomeSummaryView(p.contig, p.component_count, len(p.components), p.informative_markers, p.omitted_reason,
                              float(p.configured_morgans))
        for p in prepared
    )
