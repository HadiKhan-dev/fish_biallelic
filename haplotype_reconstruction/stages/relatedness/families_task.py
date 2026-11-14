"""Execute the independent relatedness/families task under one runtime CPU team.

Resume is by derivation before reading evidence.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ...engine.cpu import thread_team
from ...science.relatedness.parameters import AlleleFrequencyParameters, FamilyParameters, RateParameters
from ...science.relatedness.rate import decide, estimate_rate
from ...science.relatedness.run import concatenate_terms, contig_terms, fit_families
from ...store.protocol import claim
from ...store.store import CheckpointStore
from . import families_records as records
from .evidence_pass import upstream_identities, bound_inputs, EvidenceInput, evidence_pass

__all__ = ["run", "run_stage"]
_DEFAULT_FREQUENCY = AlleleFrequencyParameters()
_DEFAULT_RATE = RateParameters()


def run(
    root: str | Path,
    inputs: tuple[EvidenceInput, ...],
    sample_ids: tuple[str, ...],
    parameters: FamilyParameters,
    frequency: AlleleFrequencyParameters = _DEFAULT_FREQUENCY,
    rate: RateParameters = _DEFAULT_RATE,
    *,
    reverse: bool = False,
    team: int = 1,
) -> str:
    """Compute/publish once or reuse the same scientific record; return its derivation."""
    from ...contracts.payload import to_payload
    from ...store.protocol import NumericalPolicy

    store = CheckpointStore(root, nthreads=team)
    manifest = dict(
        contigs=tuple(item.contig for item in inputs), sample_ids=sample_ids,
        parameters={
            "relatedness": to_payload(parameters),
            "allele_frequency": to_payload(frequency),
            "rate": to_payload(rate),
        },
        ties={"reverse": reverse}, numerical_policy=to_payload(NumericalPolicy.current()),
    )
    # These inputs have already been checked by evidence_pass; the reader derives the same named map/evidence set.
    derivation = records.expected(
        store, manifest, product="families", upstream=upstream_identities(inputs)
    ).identity
    file = CheckpointStore(root, nthreads=team).run
    with claim(file, records.KEY, derivation) as compute:
        if not compute:
            return derivation
        from ...engine.cpu import claim_threads, current_budget

        def terms_of(item, evidence):
            claim_threads()  # each contig's terms on the task's current share
            # Decode independent allele-frequency site windows on that live share.
            budget = current_budget()
            return contig_terms(
                evidence,
                item.genetic_map,
                parameters,
                frequency,
                map_work=lambda work, items: budget.thread_map(work, items, label="relatedness frequency windows"),
            )

        with thread_team(default=team):
            terms = evidence_pass(root, inputs, sample_ids, terms_of, team=team)
            combined = concatenate_terms(sample_ids, tuple(item.contig for item in inputs), terms, parameters)
            del terms  # the concatenated copy alone is read from here on
            # The task starts beside discovery with the CPUs it was given (often one); every search step re-reads its
            # share, so it grows into the CPUs that finished tasks free (as painting does at each batch).
            product = fit_families(combined, parameters, threads=claim_threads)
            # The genome-wide rate from these families (rl20), on the same family-site terms.
            fitted = estimate_rate(combined, product.family, product.parents, rate, threads=claim_threads)
        del combined
        estimate = decide(
            fitted,
            tuple(item.contig for item in inputs),
            supplied_rate=inputs[0].genetic_map.default_rate_cm_per_mb,
            map_free_contigs=tuple(item.contig for item in inputs if not item.genetic_map.has_map),
            infer_rate=parameters.infer_rate,
            parameters=rate,
        )
        records.write(file, replace(product, rate=estimate), derivation)
    return derivation


def run_stage(root, resolved, maps, *, reverse=False, team=1):
    """Construct bound evidence inputs for the production graph entry."""
    inputs = bound_inputs(root, resolved, maps, team=team)
    return run(root, inputs, resolved.sample_ids, resolved.parameters.relatedness, reverse=reverse, team=team)
