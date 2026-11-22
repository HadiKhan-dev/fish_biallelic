"""Execute the independent relatedness/families task under one runtime CPU team.

Resume is by derivation before reading evidence.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from ...engine.cpu import thread_team
from ...science.relatedness.parameters import FamilyParameters, RateParameters
from ...science.relatedness.rate import decide, estimate_rate
from ...science.relatedness.run import concatenate_terms, contig_terms, fit_families, with_sources
from ...store.protocol import claim
from ...store.store import CheckpointStore
from . import families_records as records
from .evidence_pass import upstream_identities, bound_inputs, EvidenceInput, evidence_pass

__all__ = ["run", "run_stage"]


class _ManifestReader:
    """The run manifest as the observation records read it."""

    def __init__(self, manifest):
        self.manifest = manifest
_DEFAULT_RATE = RateParameters()


def run(
    root: str | Path,
    inputs: tuple[EvidenceInput, ...],
    sample_ids: tuple[str, ...],
    parameters: FamilyParameters,
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
            "rate": to_payload(rate),
        },
        ties={"reverse": reverse}, numerical_policy=to_payload(NumericalPolicy.current()),
    )
    # These inputs have already been checked by evidence_pass; the reader derives the same named map/evidence set.
    from ..observation import records as observation_records

    run_manifest = store.manifest("run")
    observation_identity = observation_records.bound_identity(store, run_manifest)
    outlier_rate = observation_records.outlier_rate_blocks(_ManifestReader(run_manifest), store)
    derivation = records.expected(
        store, manifest, product="families",
        upstream=dict(upstream_identities(inputs), observation=observation_identity),
    ).identity
    file = CheckpointStore(root, nthreads=team).run
    with claim(file, records.KEY, derivation) as compute:
        if not compute:
            return derivation
        from ...engine.cpu import claim_threads, current_budget

        def terms_of(item, evidence):
            claim_threads()  # each contig's terms on the task's current share
            return contig_terms(evidence, item.genetic_map, parameters, outlier_rate=outlier_rate)

        with thread_team(default=team):
            terms = evidence_pass(root, inputs, sample_ids, terms_of, team=team)
            # B3's record per contig (ADVICE2 17.3): read-sharing status, the consumers' reach, the median spacing
            sharing_record = dict(
                read_sharing_status=tuple(t.read_sharing_status for t in terms),
                read_sharing_reach_bp=tuple(int(t.read_sharing_reach_bp) for t in terms),
                site_spacing_bp=tuple(
                    float(np.median(np.diff(t.positions))) if t.positions is not None and len(t.positions) > 1
                    else float("nan") for t in terms
                ),
            )
            # rl21: with two source populations, the ancestry classes' laws on every contig's terms (else unchanged).
            terms, sources = with_sources(terms, parameters)
            combined = concatenate_terms(
                sample_ids, tuple(item.contig for item in inputs), terms, parameters, sources
            )
            del terms  # the concatenated copy alone is read from here on
            # The task starts beside discovery with the CPUs it was given (often one); every search step re-reads its
            # share, so it grows into the CPUs that finished tasks free (as painting does at each batch).
            product = fit_families(combined, parameters, threads=claim_threads, sources=sources)
            # The genome-wide rate from these families (rl20), on the same family-site terms.
            # Fitted on the map-free contigs when some contigs have map files (the estimate stands in for the
            # fallback rate only); a fully map-free or fully file-covered genome uses every contig.
            free = np.asarray([not item.genetic_map.has_map for item in inputs], dtype=np.bool_)
            fitted = estimate_rate(combined, product.family, product.parents, rate, threads=claim_threads,
                                   contig_mask=free if free.any() and not free.all() else None)
        del combined
        estimate = decide(
            fitted,
            tuple(item.contig for item in inputs),
            supplied_rate=inputs[0].genetic_map.default_rate_cm_per_mb,
            map_free_contigs=tuple(item.contig for item in inputs if not item.genetic_map.has_map),
            rate_estimate=parameters.rate_estimate,
            parameters=rate,
        )
        records.write(file, replace(product, rate=estimate, **sharing_record), derivation)
    return derivation


def run_stage(root, resolved, maps, *, reverse=False, team=1):
    """Construct bound evidence inputs for the production graph entry."""
    inputs = bound_inputs(root, resolved, maps, team=team)
    return run(root, inputs, resolved.sample_ids, resolved.parameters.relatedness, reverse=reverse, team=team)
