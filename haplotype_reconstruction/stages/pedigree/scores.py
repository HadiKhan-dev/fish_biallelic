"""Publish once-floored chromosome M0/M1/M2, exposure and junction evidence (PS16–PS19)."""

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
from ...science.pedigree.parameters import PedigreeParameters
from ...science.pedigree.scoring import score_chromosome_selective
from ...store.container import CheckpointFile
from ...store.identity import code_dependency_identity
from ...store.protocol import Derivation, NumericalPolicy
from ..framework import load_product, publish_product

KEY = "pedigree/floored_scores"


class ScoresAdapter:
    """Schema-2 score product: the M1 field is already the fixed genome profile."""

    schema = 2

    def encode(self, value: ChromosomeScores) -> Mapping[str, Any]:
        return {"schema": self.schema, **to_payload(value)}

    def decode(self, payload: Mapping[str, Any]) -> ChromosomeScores:
        require_schema(payload, self.schema)
        names = {f.name for f in fields(ChromosomeScores)}
        return from_payload(ChromosomeScores, {k: v for k, v in payload.items() if k != "schema" or k in names})


def derivation(
    parameters: PedigreeParameters,
    prepared_identity: str,
    screen_identity: str,
    panel: CandidatePanel,
    *,
    reverse: bool | None = None,
    numerical_policy: NumericalPolicy | None = None,
) -> str:
    """No second flooring: bind the fixed screen and canonical panel row digest."""
    reverse = parameters.reverse_conventions if reverse is None else reverse
    return Derivation(
        code_dependency_identity((__name__,)),
        dict(
            scoring=parameters.scoring,
            exposure=parameters.exposure,
            direction_mismatch_probability=parameters.direction.mismatch_probability,
            score_schema=ScoresAdapter.schema,
            m1_values="fixed-genome-profile",
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
    *,
    threads: int = 1,
    reverse: bool | None = None,
    file: CheckpointFile | None = None,
    identity: str | None = None,
) -> ChromosomeScores:
    """Expand only this chromosome's M1 profile and evaluate the fixed panel."""
    if (file is None) != (identity is None):
        raise InputError("publication requires both file and identity")
    adapter = ScoresAdapter()
    if file is not None:
        assert identity is not None
        previous = load_product(file, KEY, identity, adapter)
        if previous is not None:
            return previous
    one = compact.chromosome_reference(chromosome, eligibility)
    with thread_team(default=threads):
        # claim_threads: the science's kernel boundaries re-read this task's share (it grows as peers finish).
        product = score_chromosome_selective(
            prepared, one, panel.trios, eligibility, parameters, claim_threads, reverse=reverse
        )
    if file is not None:
        assert identity is not None
        publish_product(file, KEY, identity, product, adapter)
    return adapter.decode(adapter.encode(product))


def chromosome_summary(prepared: tuple[PreparedChromosome, ...]) -> tuple[ChromosomeSummaryView, ...]:
    """Supply numeric summary rows on the configured contig axis, including omissions."""
    return tuple(
        ChromosomeSummaryView(p.contig, p.component_count, len(p.components), p.informative_markers, p.omitted_reason)
        for p in prepared
    )
