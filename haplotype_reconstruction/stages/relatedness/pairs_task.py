"""Execute the independent relatedness/pairs task under one runtime CPU team.

Resume is by derivation before reading evidence. The output selection is outside
scientific parameters; graph composition schedules the optional pairs task.
"""

from __future__ import annotations

from pathlib import Path

from ...engine.cpu import thread_team
from ...science.relatedness.ibd import accumulate_pairwise_ibd, finish_pairwise_ibd, new_pairwise_ibd
from ...science.relatedness.ibd_parameters import PairwiseIbdParameters
from ...science.relatedness.parameters import AlleleFrequencyParameters
from ...store.protocol import claim
from ...store.store import CheckpointStore
from . import pairs_records as records
from .evidence_pass import upstream_identities, bound_inputs, EvidenceArrays, EvidenceInput, evidence_pass

__all__ = ["run", "run_stage"]
_DEFAULT_FREQUENCY = AlleleFrequencyParameters()


def run(
    root: str | Path,
    inputs: tuple[EvidenceInput, ...],
    sample_ids: tuple[str, ...],
    parameters: PairwiseIbdParameters,
    frequency: AlleleFrequencyParameters = _DEFAULT_FREQUENCY,
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
        parameters={"pairwise_ibd": to_payload(parameters), "allele_frequency": to_payload(frequency)},
        ties={"reverse": reverse}, numerical_policy=to_payload(NumericalPolicy.current()),
    )
    # These inputs have already been checked by evidence_pass; the reader derives the same named map/evidence set.
    derivation = records.expected(
        store, manifest, product="pairs", upstream=upstream_identities(inputs)
    ).identity
    file = CheckpointStore(root, nthreads=team).run
    with claim(file, records.KEY, derivation) as compute:
        if not compute:
            return derivation
        with thread_team(default=team):
            state = new_pairwise_ibd(len(sample_ids))

            def consume(item: EvidenceInput, evidence: EvidenceArrays) -> None:
                accumulate_pairwise_ibd(state, evidence, item.genetic_map, parameters, frequency)

            evidence_pass(root, inputs, sample_ids, consume, team=team)
            product = finish_pairwise_ibd(state)
        records.write(file, product, derivation)
    return derivation


def run_stage(root, resolved, maps, *, reverse=False, team=1):
    """Construct bound evidence inputs for the production graph entry."""
    inputs = bound_inputs(root, resolved, maps, team=team)
    return run(root, inputs, resolved.sample_ids, resolved.parameters.pairwise_ibd, reverse=reverse, team=team)
