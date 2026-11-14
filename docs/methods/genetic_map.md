# Genetic coordinates

A map is one working map: a supplied map's knots, extended at the fallback rate beyond them and on map-free contigs. Positions are bp; internal cumulative distance is Morgans, and files use cM. The pedigree's material and the relatedness support units are measured on this working map.

The sections below identify semantic units, not separately configurable pipeline steps. [Outputs](../user/outputs.md), [record contracts](../reference/records.md) and [costs](../developer/costs.md) describe their public boundaries.

<a id="ec27"></a>
## ec27 Genetic map: piecewise-linear cumulative map, fallback rate, files, identity, Haldane fraction

Validate strictly increasing map positions and nondecreasing cumulative cM without sorting or repair. Interpolate cumulative Morgans between knots and extend both tails continuously at the fallback rate; use direct interval distance when map-free. The common Haldane function uses expm1 to evaluate the small-distance recombination fraction without cancellation.


Implementation: [contracts/genetic_map.py](../../haplotype_reconstruction/contracts/genetic_map.py), [stages/inputs/genetic_map.py](../../haplotype_reconstruction/stages/inputs/genetic_map.py), [stages/configure.py](../../haplotype_reconstruction/stages/configure.py).

Parameters by name: `rate_cm_per_mb`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(log M). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.
