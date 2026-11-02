# Genetic coordinates

A map has two views: model distances include declared fallback tails, while measured coordinates identify only measured support. Positions are bp; internal cumulative distance is Morgans, and files use cM. These views must not be substituted for each other.

The sections below identify semantic units, not separately configurable pipeline steps. [Outputs](../user/outputs.md), [record contracts](../reference/records.md) and [costs](../developer/costs.md) describe their public boundaries.

<a id="ec27"></a>
## ec27 Genetic map: piecewise-linear cumulative map, fallback rate, files, identity, Haldane fraction

Validate strictly increasing map positions and nondecreasing cumulative cM without sorting or repair. Interpolate cumulative Morgans between knots and extend both tails continuously at the fallback rate; use direct interval distance when map-free. Measured-coordinate views omit invented fallback length. The common Haldane function uses expm1 (L1).

Implementation: [contracts/genetic_map.py](../../haplotype_reconstruction/contracts/genetic_map.py), [stages/inputs/genetic_map.py](../../haplotype_reconstruction/stages/inputs/genetic_map.py), [stages/configure.py](../../haplotype_reconstruction/stages/configure.py).

Parameters by name: `rate_is_measured`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(log M). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.
