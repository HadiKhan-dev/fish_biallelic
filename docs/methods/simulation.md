# Simulation and templates

Simulation keeps its generating model and truth separate from inference. Explicit named streams preserve generator type, call shape, draw order and contig order. N is generated or observed individuals as stated, L sites, F founder haplotypes and C contigs. The independent-site and fragment read models have different dependence assumptions. Stored pedigree descriptions are truth annotations, never inference metadata.

The sections below identify semantic units, not separately configurable pipeline steps. [Outputs](../user/outputs.md), [record contracts](../reference/records.md) and [costs](../developer/costs.md) describe their public boundaries.

<a id="ou20"></a>
## ou20 Template builder (windowed discovery on fixed-error evidence)

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Build templates by overlapping physical windows of fixed-error read evidence, discovering only informative windows and collecting results by window index. Convert called discovered alleles to probabilities and uncalled cells to neutral allele mass. Use the declared missing-length fallback when needed. The template-building model is separate from inference on simulated observations.

Implementation: [interface/templates/builder.py](../../haplotype_reconstruction/interface/templates/builder.py).

Parameters by name: `WINDOW_BP`, `WINDOW_SHIFT_BP`, `TEMPLATE_HAPLOTYPES`, `WINDOW_EVIDENCE_FACTOR`, `WINDOW_ALLOWANCE_BYTES`, `CONTIG_CELL_BYTES`, `UNDECLARED_LENGTH_BP`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(windows · K²). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou21"></a>
## ou21 Naive window linker

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Link admitted windows by repeated best-path dynamic programming with first-candidate exact ties, then penalise used nodes and edges before selecting another path. Reconstruct within the named equality tolerance and merge overlaps through signed information with the declared cancellation and inversion rules. This naive linker is a retained approximation. L14 disables compiler arithmetic reassociation; any moved template draw is attributable only when templates are rebuilt.

Implementation: [interface/templates/linker.py](../../haplotype_reconstruction/interface/templates/linker.py).

Parameters by name: `NODE_PENALTY`, `EDGE_PENALTY`, `GAP_SCORE`, `EDGE_DIVISOR`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(W K² overlap); O(W K² L_w); O(count · W · K²). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou22"></a>
## ou22 Synthetic example generator

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Generate each example contig from its named spawned stream, with independent binary founder haplotypes and evenly spaced positions. Write certain template allele probabilities and a complete validated simulation configuration. Check contig/site/length bounds. The default example contains three diploid founders and three contigs; it demonstrates software wiring rather than biological realism or benchmark accuracy.

Implementation: [simulation/example.py](../../haplotype_reconstruction/simulation/example.py), [interface/example.py](../../haplotype_reconstruction/interface/example.py).

Parameters by name: `EXAMPLE_HAPLOTYPES`, `EXAMPLE_COHORTS`, `EXAMPLE_DEPTH`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(F L) generated template cells per contig.

<a id="ou23"></a>
## ou23 Founder materialisation from templates

Validate template axes, finite nonnegative allele masses, increasing positions and a consistent even founder-haplotype count across contigs. Choose the maximum allele probability; for exact ties make one vector draw per haplotype only when ties exist, in site order. Pair consecutive haplotypes into diploid founders. Tie-free rows consume no random draw.

Implementation: [simulation/source.py](../../haplotype_reconstruction/simulation/source.py), [simulation/pedigree.py](../../haplotype_reconstruction/simulation/pedigree.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(H L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou24"></a>
## ou24 Random-cohort pedigree, backcrosses and observation selection

Use the first master draw to seed the structural generator. Build each cohort’s whole pedigree before drawing its contig-by-child seed block. In a backcross-eligible cohort consume the structural probability draw for every child, then select the appropriate ordered parent pair. Select observed samples with the separate observation stream and restore pedigree order. Contigs share structure and scheduling cannot change it; cohort labels remain truth only.

Implementation: [simulation/pedigree.py](../../haplotype_reconstruction/simulation/pedigree.py), [simulation/design.py](../../haplotype_reconstruction/simulation/design.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(total individuals). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou25"></a>
## ou25 Explicit pedigree table

Validate explicit pedigree rows with founders first and distinct earlier parents for each descendant. Simulate by ancestry depth, preserving table order within depth and drawing each contig-by-member seed block before scheduling; this route has no structure-seed draw. Sort recorded gametes by child and slot after all depths. Observation follows the table flag. Sex and generation are stored descriptions read by no inference model; bind table content, not its path.

Implementation: [simulation/pedigree.py](../../haplotype_reconstruction/simulation/pedigree.py), [simulation/design.py](../../haplotype_reconstruction/simulation/design.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: One task set per ancestry depth, with generated haplotypes retained per contig; pedigree validation is linear in rows/edges.

<a id="ou26"></a>
## ou26 Meiosis: gametes, crossovers, mutations, ancestry segments

Draw the starting homologue, then crossover events by the selected map/no-map branch, then mutations. Zero genetic distance or zero fallback rate consumes no crossover draws. Include the final marker with the declared terminal boundary, and assign a marker exactly on a crossover to the new homologue. Simulate maternal then paternal gametes from one child stream. Duplicate mutation indices flip once. Ancestry intervals and fractional map events retain their distinct coordinate forms.

Implementation: [simulation/pedigree.py](../../haplotype_reconstruction/simulation/pedigree.py).

Parameters by name: `MUTATION_RATE_PER_BP`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(L); O(crossovers); O(generated individuals · L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou27"></a>
## ou27 Independent-site read sampler

Draw optional normalised depth factors, then the full sample-by-site Poisson depth array, optional dropout starts and genotype-class binomial ALT counts in class/row-major order. Skip a genotype-class draw when its mask is empty. Emit REF/ALT counts after dropout. Splitting a contig into new sample chunks changes draw layout and is not a valid execution-only optimisation.

Implementation: [simulation/reads.py](../../haplotype_reconstruction/simulation/reads.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N L) draws and storage; full-contig array draw shapes prevent sample-chunk substitution.

<a id="ou28"></a>
## ou28 Paired-end fragment read sampler

Draw depth factors once, then each sample’s fragment count, starts, lengths and chosen homologues. Pack first-mate bases before second-mate bases, avoiding double-counted overlap. A sample with no covered base skips later base draws. Apply optional heterozygote bias, then the declared homozygous error mask, accumulating counts by site. Draw dropout starts only after all samples. Fragment correlation is part of the generating model.

Implementation: [simulation/reads.py](../../haplotype_reconstruction/simulation/reads.py).

Parameters by name: `FRAGMENT_QUANTILES`, `MINIMUM_FRAGMENT_BP`, `FRAGMENT_SD_LIMIT`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N · fragments); O(bases per sample). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou29"></a>
## ou29 Truth records: alleles, painting chunks, crossovers, manifest, truth pedigree

Publish observations and independent truth with aligned sample/site axes, true founder alleles, observed-sample alleles, ancestry chunks and analysed-child gametes. Truth chunks use the union of homologue segment bounds and require whole-chunk containment. Stored integer bounds truncate fractional events, a known truth-surface limitation that can disagree at a boundary marker; crossover positions retain float precision. Truth metadata never enters inference.

Implementation: [simulation/source.py](../../haplotype_reconstruction/simulation/source.py), [simulation/truth.py](../../haplotype_reconstruction/simulation/truth.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N · segments). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou30"></a>
## ou30 Simulation source: streams, identity, resume, scheduling and memory

Bind the realised seed once and preserve named founder, pedigree, observation, gamete and read stream constructors and draw order. Materialise the whole configured contig set, reusing only compatible completed source products. Publish truth before its read-completion product. Keep generated and analysed sample axes distinct and reject partial products without their manifest. Task order may change execution, never per-contig seeds or data.

Implementation: [simulation/source.py](../../haplotype_reconstruction/simulation/source.py), [stages/inputs/simulation.py](../../haplotype_reconstruction/stages/inputs/simulation.py).

Parameters by name: `READS_BYTES_PER_CELL`, `READS_WORKER_BYTES`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: Sum of generation and read-sampling costs; whole-contig read arrays often dominate memory.

<!-- named mechanism anchors -->

## Named mechanisms

<a id="example"></a>

**Example:** see [OU22](#ou22).

<a id="templates"></a>

**Templates:** see [OU20](#ou20).
