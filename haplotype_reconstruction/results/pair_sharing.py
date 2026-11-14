"""Founder sharing of every sample pair (PR20), computed from a run's published paintings.

A view over the public Results methods: the painting is read through its stage request (its released calls, without
panels or the emission cache) and counted by contracts.pair_sharing. Results never opens input files, so the caller
supplies the run's configured maps; each used contig's map must have the identity the run bound. Lengths are measured
on the run's working map: the configured maps, or, when ``[genetic_map] infer_rate`` adopted the relatedness stage's
rate estimate (rl20), the same map files with the estimate as fallback rate, as every map consumer of the run reads
them. Nothing is stored and nothing feeds back. This module is kept out of the Results class so that the readers
inside every stage's code identity are unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..contracts.genetic_map import GeneticMapSet
from ..contracts.pair_sharing import PairSharing, contig_pair_lengths
from ..core.digest import json_digest
from ..core.errors import InvariantError, SettingsError

if TYPE_CHECKING:
    from ..contracts.painting import Painting
    from .reader import Results

__all__ = ["painting_calls", "pair_sharing", "working_maps"]


def painting_calls(results: Results, contig: str) -> Painting | None:
    """The contig's released painting calls (no panels, no emission cache); None before painting publishes."""
    request = results.stage_request("painting", contig)
    return None if request is None else results.read(request, contig)


def working_maps(results: Results, genetic_maps: GeneticMapSet) -> GeneticMapSet:
    """The run's working maps from its configured ``genetic_maps``: unchanged without ``infer_rate`` or when the
    relatedness stage kept the supplied rate; with an adopted rate, contigs without a map file use the estimate
    (the rule of stages.relatedness.working_map.working_maps, read here through the public Results surface)."""
    if not results.manifest.get("parameters", {}).get("relatedness", {}).get("infer_rate", False):
        return genetic_maps
    families = results.relatedness()
    if families is None or families.rate is None:
        raise InvariantError("pair sharing of a run with infer_rate needs the published relatedness record")
    if families.rate.decision != "adopted":
        return genetic_maps
    return GeneticMapSet(genetic_maps.maps, default_rate_cm_per_mb=families.rate.working_rate_cm_per_mb)


def pair_sharing(results: Results, genetic_maps: GeneticMapSet, contig: str | None = None) -> PairSharing | None:
    """One included contig's pair sharing, or (``contig`` None) the sum over the included contigs that have a
    painting, in their bound order; None when none has one. ``genetic_maps`` are the run's configured maps."""
    if contig is not None and contig not in results.contigs:
        raise SettingsError(f"{contig}: not an included contig of this run")
    bound = results.manifest["map_identities"]
    working = None

    def part(name: str) -> PairSharing | None:
        nonlocal working
        painting = painting_calls(results, name)
        if painting is None:
            return None
        if json_digest(genetic_maps.identity(name)) != bound[name]:
            raise SettingsError(f"{name}: the supplied genetic map is not the run's configured map")
        working = working_maps(results, genetic_maps) if working is None else working
        return PairSharing(painting.sample_ids, (name,), contig_pair_lengths(painting, working.for_contig(name)))

    return PairSharing.total(
        item for name in (results.contigs if contig is None else (contig,)) if (item := part(name)) is not None
    )
