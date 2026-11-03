# Conditional recombination maps

Recombination follows released meiosis chains under the fixed inferred pedigree. Called parity determines conditional Poisson/Haldane interpolation between markers; it is not a posterior computed from the chain model. E is parent edges, B chain bins, L sites and J admitted junction loci. Missing exposure remains missing. Reference-order junctions are diagnostics and do not mask or refit the map.

The sections below identify semantic units, not separately configurable pipeline steps. [Outputs](../user/outputs.md), [record contracts](../reference/records.md) and [costs](../developer/costs.md) describe their public boundaries.

<a id="rc01"></a>
## rc01 Breakpoint table (shared reader; protected intervals and order breakpoints)

Parse the breakpoint table in file order with one-based inclusive coordinates, comments and validated bounds. Ordinary intervals protect their whole span; a reversed_segment reason identifies an inversion whose two endpoints protect phase. Bind parsed content rather than the file path. The reader is an input adapter; scientific functions consume the immutable table.

Implementation: [stages/inputs/breakpoints.py](../../haplotype_reconstruction/stages/inputs/breakpoints.py), [contracts/recombination.py](../../haplotype_reconstruction/contracts/recombination.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(rows). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rc02"></a>
## rc02 Breakpoint axis split, chain bins and segments, reversed-segment flags

Enumerate ordinary intervals then inversion endpoint points. Convert each to half-open coordinates and widen cumulatively to overlapping component bins in component order. Mask those widened regions, cut at their right ends and rebuild segment IDs along valid sites. Inversion interiors are not automatically masked, although widened endpoints can cover them. Derive chain-bin segment/reset flags and remap reported spans to assembly components.

Implementation: [science/recombination/junctions.py](../../haplotype_reconstruction/science/recombination/junctions.py), [stages/recombination/tasks.py](../../haplotype_reconstruction/stages/recombination/tasks.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(intervals x components x log B + L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rc03"></a>
## rc03 Chain following: called crossovers, reported intervals, callable spans, marker counts

Walk released, valid chain bins and follow only consecutive released markers within the same reset stretch. Merge contiguous callable spans and call a crossover when source changes, reporting the inclusive outer-bin interval and a joint support lower bound. An isolated released bin contributes no followed span. Missing interior bins remain explicitly unresolved; even numbers of changes between markers cannot be identified.

Implementation: [stages/recombination/tasks.py](../../haplotype_reconstruction/stages/recombination/tasks.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(M); O(E M). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rc04"></a>
## rc04 M2 conditional map: map bins, genetic coordinates, expected crossovers and exposure per meiosis

Build map bins from the first/last site and segment changes. For each nonreset marker interval, use called parity with the Poisson/Haldane conditional count: even and odd intervals have distinct expectations. Spread expected counts by supplied genetic mass, or uniformly in physical distance when no positive mass exists. An odd zero-mass interval still contributes a crossover. Exposure is one meiosis per covered base. These are conditional bridge interpolations, not chain-posterior means.

Implementation: [science/recombination/map.py](../../haplotype_reconstruction/science/recombination/map.py).

Parameters by name: `bin_bp`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(markers x bins touched); O(E bins). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rc05"></a>
## rc05 Map aggregation and map units

Sum expected counts and exposure over edges in the declared array reduction order. Derive rate from expected crossovers per meiosis-base and retain NaN where exposure is absent. Integrate only covered physical length for observed cumulative cM; distribute called events separately for the called-rate table. The rate_is_measured flag does not alter this conditional-map prior calculation.

Implementation: [science/recombination/map.py](../../haplotype_reconstruction/science/recombination/map.py).

Parameters by name: `bin_bp`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(E bins + calls). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rc06"></a>
## rc06 Reference-order junction detector (report only)

Group touching/overlapping reported event intervals into loci and require support from enough distinct parents. Compute odd parity and decidability, retaining zero-odd parents. Compare mixture and paired-locus grid scores, with paired scores restricted to edges decidable at both loci and correction over all admitted locus pairs. Apply named report tie conventions to grid maxima, partners and nearest supplied breakpoints. Flag by the inclusive likelihood-ratio threshold; these junction diagnostics never mask or refit the map.

Implementation: [science/recombination/junctions.py](../../haplotype_reconstruction/science/recombination/junctions.py).

Parameters by name: `minimum_parents`, `hotspot_switch_probability`, `flag_log_likelihood_ratio`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(loci^2 x E); O(loci^2 x parents x grid). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rc07"></a>
## rc07 Recombination stage orchestration, record, identity, rerun and summary

Require released chains conditioned on exactly the supplied parent-edge digest. Bind recombination settings and breakpoint content before work, publish the contig result and format summaries from checked records. Reuse a matching publication and refuse an incompatible binding, including in a copied completed run. Alternative settings require an independent checkpoint root. Report phase convergence and missing exposure explicitly; no coverage field or diagnostic junction call substitutes for chain evidence.

Implementation: [stages/recombination/records.py](../../haplotype_reconstruction/stages/recombination/records.py), [stages/recombination/tasks.py](../../haplotype_reconstruction/stages/recombination/tasks.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(E M + L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.
