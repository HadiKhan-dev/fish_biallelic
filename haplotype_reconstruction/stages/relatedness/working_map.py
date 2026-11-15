"""Resolve the run's working genetic map for the stages that read a map after relatedness (rl20).

Unless ``[genetic_map] rate_estimate = "adopt"`` the working map is the configured map: every function here returns
the configured maps and the manifest's map identities without reading any record, and the workflow graph has no edge
from relatedness to the map consumers. With adopt, the relatedness product decides: when it adopted the estimated
rate, it becomes the fallback rate of the working map (contigs without a map file and the stretches beyond every
file's knots; a file's own distances never change), and every consumer binds that working map's digest, so resume
and record reuse follow the decision. The painting-based estimate never changes the working map (it decides only the
pedigree's material regime, stages.pedigree.map_scale).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...contracts.genetic_map import GeneticMapSet
from ...contracts.relatedness import RecombinationRateEstimate
from ...core.digest import json_digest
from ...core.errors import InvariantError

__all__ = ["infers_rate", "rate_decision", "working_maps", "map_identity"]


def infers_rate(manifest: Mapping[str, Any]) -> bool:
    """Whether the bound run lets relatedness replace the fallback rate (rate_estimate = adopt)."""
    return manifest.get("parameters", {}).get("relatedness", {}).get("rate_estimate", "veto") == "adopt"


def rate_decision(store: Any, manifest: Mapping[str, Any]) -> RecombinationRateEstimate:
    """The published relatedness rate decision of this run (its record must exist)."""
    from . import families_records

    reader = families_records.result_reader(store, manifest, product="families")
    product = None if reader is None else families_records.load(store.run, reader.identity)
    if product is None or product.rate is None:
        raise InvariantError("the working map of a run with rate_estimate = adopt needs the published relatedness record")
    return product.rate


def working_maps(store: Any, manifest: Mapping[str, Any], maps: GeneticMapSet) -> GeneticMapSet:
    """The configured maps, or with an adopted rate the same map files and the estimate as fallback rate."""
    if not infers_rate(manifest):
        return maps
    decision = rate_decision(store, manifest)
    if decision.decision != "adopted":
        return maps
    return maps.with_fallback_rate(decision.working_rate_cm_per_mb)


def map_identity(store: Any, manifest: Mapping[str, Any], contig: str) -> str:
    """The digest of ``contig``'s working map, as a consumer's derivation binds it."""
    if not infers_rate(manifest):
        return manifest["map_identities"][contig]
    decision = rate_decision(store, manifest)
    if decision.decision != "adopted":
        return manifest["map_identities"][contig]
    document = manifest["map_documents"][contig]
    return json_digest({**document, "default_rate_cm_per_mb": float(decision.working_rate_cm_per_mb)})
