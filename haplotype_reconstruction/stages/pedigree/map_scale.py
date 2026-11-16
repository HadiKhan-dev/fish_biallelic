"""Publish the run's painting-based fallback-rate estimate (record ``pedigree/map_scale``) for the pedigree material rule.

One genome-wide task of the pedigree stage: after every included contig's painting and the relatedness families (the
pedigree node's own prerequisites), before the screen barrier, which reads the record to choose the material regime
(science.pedigree.regimes.material_choice). Paintings are read one contig at a time and kept only for the relevant
samples (members and sequenced slots of the families of >= 2); a component without its packed emission cache is
recomputed from the contig's evidence for those samples. The derivation binds every painting, the families product,
the working map identities and the map-scale parameters, so resume follows each cause.

Method: docs/methods/pedigree.md#map-scale.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ...contracts.contigs import AnalysisContigs
from ...contracts.genetic_map import GeneticMapSet
from ...contracts.map_scale import MapScaleEstimate
from ...contracts.payload import from_payload, to_payload
from ...contracts.roles import require_schema
from ...core.concurrency import threaded_each
from ...core.errors import InputError
from ...engine.cpu import thread_team
from ...science.pedigree import map_scale as science
from ...science.pedigree import map_scale_frames as frames_science
from ...science.pedigree.parameters import PedigreeParameters
from ...store.identity import code_dependency_identity
from ...store.protocol import Derivation, NumericalPolicy
from ...store.store import CheckpointStore
from ..framework import load_product, publish_product
from ..relatedness.working_map import map_identity

__all__ = ["KEY", "MapScaleAdapter", "derivation", "expected", "load", "run_stored"]

KEY = "pedigree/map_scale"


class MapScaleAdapter:
    """Schema-1 plain payload of MapScaleEstimate; decoded arrays are read-only."""

    schema = 1

    def encode(self, value: MapScaleEstimate) -> Mapping[str, Any]:
        return {"schema": self.schema, **to_payload(value)}

    def decode(self, payload: Mapping[str, Any]) -> MapScaleEstimate:
        require_schema(payload, self.schema)
        return from_payload(MapScaleEstimate, {k: v for k, v in payload.items() if k != "schema"})


def derivation(
    parameters: PedigreeParameters, upstream: Mapping[str, str], *, numerical_policy: NumericalPolicy | None = None
) -> str:
    """Bind the estimator's code, its parameters and material unit, and every upstream record identity."""
    material = parameters.material
    return Derivation(
        code_dependency_identity((__name__,)),
        dict(map_scale=parameters.map_scale, unit=(material.unit_morgans, material.tail_morgans),
             schema=MapScaleAdapter.schema),
        numerical_policy or NumericalPolicy.current(),
        dict(upstream),
        parameters.reverse_conventions,
    ).identity


def expected(store: CheckpointStore, manifest: Mapping[str, Any], parameters: PedigreeParameters,
             numerical_policy: NumericalPolicy | None = None) -> str:
    """The record identity from the bound manifest: paintings and evidence of the included contigs, the families
    product and the working map identities (header reads only)."""
    from ..evidence import records as evidence_records
    from ..painting import records as painting_records
    from ..relatedness import families_records

    def bind(c: str) -> tuple[str, str, str]:
        painted = painting_records.result_reader(store, manifest, product="reference", contig=c, panels=False)
        observed = evidence_records.result_reader(store, manifest, product="evidence", contig=c)
        if painted is None or observed is None:
            raise InputError(f"the map-scale estimate requires painting and evidence for {c}")
        return painted.identity, observed.identity, map_identity(store, manifest, c)

    contigs = AnalysisContigs.from_manifest(manifest).included
    upstream: dict[str, str] = {}
    # each contig's upstream chain is independent of the others (records.bound_inputs binds the same way)
    for c, (painted, observed, working_map) in zip(contigs, threaded_each(bind, contigs), strict=True):
        upstream[f"painting:{c}"] = painted
        upstream[f"evidence:{c}"] = observed
        upstream[f"map:{c}"] = working_map
    family = families_records.result_reader(store, manifest, product="families")
    if family is None:
        raise InputError("the map-scale estimate requires relatedness/families")
    upstream["relatedness/families"] = family.identity
    return derivation(parameters, upstream, numerical_policy=numerical_policy)


def load(store: CheckpointStore, identity: str) -> MapScaleEstimate:
    value = load_product(store.run, KEY, identity, MapScaleAdapter())
    if value is None:
        raise InputError("the pedigree material rule requires the published map-scale estimate")
    return value


def _component_inputs(store, manifest, contig_index, contig, rows, threads) -> tuple[list, int]:
    """Restricted component inputs of one contig (``rows``: relevant samples, ascending), recomputing missing
    packed emissions from the contig's evidence."""
    from ..evidence import records as evidence_records
    from ..painting import records as painting_records

    from .prepare import _read_request

    file = store.contig(contig)
    reader = painting_records.result_reader(store, manifest, product="reference", contig=contig, panels=False)
    painting = _read_request(file, reader)
    if tuple(painting.sample_ids) != tuple(manifest["sample_ids"]):
        raise InputError("painting sample axis differs from the run's")
    evidence = None
    out, recomputed = [], 0
    # the pedigree reads the painting at its path strength (science.pedigree.prepare.path_correlation)
    from ...science.pedigree.prepare import path_correlation

    site_correlation = path_correlation(painting)
    for comp in painting.components:
        if comp.bins.n_bins < 2:
            continue
        packed = comp.packed_emissions if comp.transitions.site_correlation == site_correlation else None
        if packed is None:
            if evidence is None:
                observed = evidence_records.result_reader(store, manifest, product="evidence", contig=contig)
                evidence = evidence_records.load_dense(file, expected_identity=observed.identity)
            packed = frames_science.packed_emissions(
                comp, painting.parameters, painting.robustness_epsilon, evidence, rows, site_correlation
            )
            recomputed += 1
            local = np.arange(len(rows))
        else:
            local = rows
        out.append(frames_science.component_input(
            contig_index, comp.bins.centers, comp.local_keys, comp.eligible, comp.classes, comp.status,
            int(comp.states.n_classes), int(comp.states.n_states), packed, rows, packed_rows=local))
    del painting, evidence
    return out, recomputed


def run_stored(
    root: str,
    manifest: Mapping[str, Any],
    maps: GeneticMapSet,
    parameters: PedigreeParameters,
    numerical_policy: NumericalPolicy,
    *,
    threads: int = 1,
) -> str:
    """Resume or compute the record; returns its identity (the barrier reads the record by it)."""
    from ..relatedness import families_records

    store = CheckpointStore(root, nthreads=threads)
    identity = expected(store, manifest, parameters, numerical_policy)
    adapter = MapScaleAdapter()
    if load_product(store.run, KEY, identity, adapter) is not None:
        return identity
    family_reader = families_records.result_reader(store, manifest, product="families")
    families_product = families_records.load(store.run, family_reader.identity)
    if families_product is None or tuple(families_product.sample_ids) != tuple(manifest["sample_ids"]):
        raise InputError("relatedness/families sample axis differs from the map-scale estimate's")
    family = np.asarray(families_product.family, dtype=np.int64)
    parents = np.asarray(families_product.parents, dtype=np.int64)
    rows = frames_science.relevant_samples(family, parents)
    families = frames_science.families_of(family, parents, rows)
    contigs = AnalysisContigs.from_manifest(manifest).included
    with thread_team(default=threads):
        components = []
        for k, c in enumerate(contigs):
            part, _recomputed = _component_inputs(store, manifest, k, c, rows, threads)
            components.extend(part)
        p = parameters.map_scale
        guard = frames_science.guard(components, families, p)
        final, _ = frames_science.build_frames(components, families, guard.active, False)
        del components
        chromosome_maps = {k: maps.for_contig(c) for k, c in enumerate(contigs)}
        free = tuple(c for c in contigs if not maps.for_contig(c).has_map)
        filed = tuple(c for c in contigs if maps.for_contig(c).has_map)
        free_index = {k for k, c in enumerate(contigs) if c in free}
        decision_frames = final.subset(np.asarray([c in free_index for c in final.contig], dtype=np.bool_))
        value = science.fit(decision_frames, chromosome_maps, p, parameters.material) if free else None
        file_value = None
        if filed:
            file_frames = frames_science.file_frames(final, chromosome_maps, set(free_index))
            file_value = science.fit(file_frames, None, p, parameters.material, positions_scale=1e6)
    record = science.estimate_record(value, file_value, guard, p, working_rate=float(maps.default_rate_cm_per_mb),
                                     map_free=free, file_contigs=filed)
    publish_product(store.run, KEY, identity, record, adapter)
    return identity
