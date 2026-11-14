# Relatedness

Relatedness fits a composite full-sib/half-sib family model from genotype likelihoods. Its latent or sequenced parent slots carry no sex, generation or sample-name prior. N is samples, M selected family sites and U nonempty support units. The two bounded searches are alternatives, not an exhaustive pedigree search. The stage estimates no pairwise identity-by-descent shares ([below](#pairwise-identity-by-descent)).

The sections below identify semantic units, not separately configurable pipeline steps. [Outputs](../user/outputs.md), [record contracts](../reference/records.md) and [costs](../developer/costs.md) describe their public boundaries.

## Pairwise identity by descent

The stage reports no pairwise IBD0/IBD1/IBD2 shares. A pairwise IBD chain on genotype likelihoods treats the alleles two samples do not share by descent as independent Hardy–Weinberg draws at the sample allele frequency, with sites independent given the IBD state. In a cross every sample carries mosaics of a few founder haplotypes: alleles not shared by descent come from the same few haplotypes, in complete linkage disequilibrium over long blocks, and the sample frequency is the frequency of no base population. Identity by state at sites where a parent is homozygous, or where the founder groups differ, then reads as identity by descent; only the founder haplotypes or the parents' genotypes show which parental homologue a child received at such a site, and this stage runs before the founder haplotypes exist. Measured on 27 simulated crosses at depths 1, 5 and 10.5 against the simulated founder-haplotype transmission, such a chain gave parent–offspring pairs mean IBD0 shares of 0.22–0.59 per cross (the truth is 0) and full sibs 0.36–0.66 (the truth is 0.09–0.26); it was close to the truth only in a cross of 60 sequenced founders and their children. On the TroMau cross it gave the 69 child–parent pairs that the pedigree stage calls IBD0 shares of 0.25–0.84, where a parent and child share at least one homologue throughout.

Sharing of founder haplotypes is defined once the founder haplotypes exist. The released paintings name, bin by bin, the founder rows each sample carries ([PR13](painting.md#pr13)), and the pedigree stage scores pairs of samples on the same painting classes ([PD28](pedigree.md#pd28)).

<a id="rl01"></a>
## rl01 Genome-wide evidence pass and the concatenated site axis

Read evidence from every configured contig, validate the canonical sample axis, compute its summaries and concatenate family-site evidence in configuration order. Overlap decoding only as an execution choice. Contig order is scientifically bound because it fixes ranks and reduction order.

Implementation: [science/relatedness/run.py](../../haplotype_reconstruction/science/relatedness/run.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N L) decoded evidence per active contig and O(N M) retained selected-site tables; concatenation can temporarily retain both source and combined arrays.

<a id="rl02"></a>
## rl02 Hardy-Weinberg EM allele frequency per site

Iterate the Hardy–Weinberg allele-frequency EM for its fixed schedule, using the declared floors/clips and sample-order sums. Count every sample in the denominator, including neutral missing observations. Preserve the left-to-right arithmetic of genotype contributions and frequency updates.

Implementation: [science/relatedness/frequency.py](../../haplotype_reconstruction/science/relatedness/frequency.py).

Parameters by name: `iterations`, `clip`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(6 · N · L_c). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rl03"></a>
## rl03 Site selection and thinning (family sites)

For family sites, apply the inclusive minor-frequency gate and compare the floating mean of the Boolean observation mask with the inclusive observed-share threshold, then thin the survivors.

Implementation: [science/relatedness/run.py](../../haplotype_reconstruction/science/relatedness/run.py).

Parameters by name: `minor_allele_frequency`, `minimum_observed`, `family_site_step`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L_c). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rl05"></a>
## rl05 Family-model site terms (child terms, latent prior, sequenced-parent posterior)

Construct child-genotype terms, the latent Hardy–Weinberg parent prior and sequenced-parent genotype posterior on family sites. Floor logs before using them in the composite model; normalise posterior states in the declared order. A slot sample is a sequenced sample whose genotypes serve the family model as a parent's would. It may be a parent, a child (for a one-member family the two directions tie), a sib or another close relative. Parent calls, direction and their support belong to the pedigree stage.

Implementation: [science/relatedness/run.py](../../haplotype_reconstruction/science/relatedness/run.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(27 N M_c). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rl06"></a>
## rl06 Composite full-sib family likelihood, half-sib link likelihood and the objective

For each family/site, add the parent terms to its member-sum state, log-sum over parent genotypes, then sum sites in order. Use one family-likelihood function for predicted and realised member updates. Half-sib scores integrate the shared latent parent under their own composite rule; the objective is not a joint pedigree likelihood.


Implementation: [science/relatedness/family_kernels.py](../../haplotype_reconstruction/science/relatedness/family_kernels.py).

Parameters by name: `initial`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(9 M); O(18 M). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rl07"></a>
## rl07 Evidence ranks, content order, family and pair keys, tie rules

Rank complete child-evidence vectors lexicographically by value, assigning dense equal ranks. Define family and pair keys from ordered member/parent ranks, then use those keys within the declared gain ties. The earliest differing site can determine rank; sample and contig axes must therefore remain bound.

Implementation: [science/relatedness/keys.py](../../haplotype_reconstruction/science/relatedness/keys.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N log N); O(9 M); O(|F| log |F|). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rl08"></a>
## rl08 Partition state: slot allocation, member order, family sums, snapshot and restore

Maintain family members in declared evidence-rank order and copy shared singleton sums before mutation. Apply join/leave/merge changes with the same arithmetic that predicts their likelihood. Snapshot and restore complete family state; recomputing member sums follows their declared order, never arbitrary set iteration.

Implementation: [science/relatedness/partition.py](../../haplotype_reconstruction/science/relatedness/partition.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N · depth). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rl09"></a>
## rl09 Agglomeration of latent full-sib families

Screen singleton-family pairs, evaluate unresolved gains exactly, and offer positive full-sib gains only when they strictly exceed the half-sib alternative. Choose the largest pair key within the gain band, merge, and update affected scores. The positive-gain and full-sib/half-sib comparisons have their stated strict boundaries.

Implementation: [science/relatedness/search.py](../../haplotype_reconstruction/science/relatedness/search.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(9 N² M); O(N M); O(9 M). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rl10"></a>
## rl10 Parent-slot assignment (one per open family per sweep)

For each family’s first free parent slot, offer its best non-member and eligible member candidates. Rank offers by gain and content, then enforce the log-sample-count penalty, per-sweep use rules and acyclicity at application. Only one non-member is offered per family; failure does not trigger a second-best search. A slot sample is a sequenced sample whose genotypes serve the family model as a parent's would. It may be a parent, a child (for a one-member family the two directions tie), a sib or another close relative. Parent calls, direction and their support belong to the pedigree stage.

Implementation: [science/relatedness/search.py](../../haplotype_reconstruction/science/relatedness/search.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(3 N M); O(9 M). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rl11"></a>
## rl11 Best-improvement single-sample moves

For each sample, choose its best admissible target family or new singleton under gain/key ties. Rank eligible positive moves and apply at most one per iteration, within the sample-count iteration limit. Exclude ancestry-cycle targets and rescore the selected move against current state.

Implementation: [science/relatedness/search.py](../../haplotype_reconstruction/science/relatedness/search.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N F); O(N M); O(N² F). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rl12"></a>
## rl12 Content-order moves and leaving with one kept parent slot

Visit samples in fixed content order. Compare the new-singleton target first and valid live families thereafter, using the declared key tie rule. Apply a sufficient gain immediately and refresh changed-family leave gains. This first-improvement schedule is intentionally distinct from the best-improvement search.

In both searches, after single-sample moves and before the merge, a member of a family with at least two members and one sequenced slot may leave for a new one-member family. Its first slot keeps one of the source family's sequenced samples and its second slot is latent. The gain is `leave_gain(c) + loglik({c}; u, latent)`. Apply the best gain strictly above log N (one additional sequenced slot), one move at a time, until none qualifies or N moves have been applied in the sweep. Gains within the tolerance tie by the member's evidence rank, then the kept sample's evidence rank. Rescore the selected leave on the current partition before application. The edge already existed, so no new ancestry or cycle arises. The second slot is offered only by ordinary assignment in the next sweep.

Keep each one-member likelihood for the fit, across both searches: it depends only on the member and kept sample, never on family versions. Up to two such likelihoods per eligible member are needed for its source family's slots.

Implementation: [science/relatedness/search.py](../../haplotype_reconstruction/science/relatedness/search.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Content-order moves use O(N M); O(N). Kept-slot likelihood work is O(N M), cached for the fit; candidate scans are bounded by N moves per sweep. There is no dependence on founder count K.

<a id="rl13"></a>
## rl13 One merge per sweep

Enumerate compatible live-family pairs, compare full-sib and applicable half-sib gains, and rank sufficient merge offers. Apply the first acyclic merge; undo a rejected cyclic trial without changing the remaining offer order. At most one merge is accepted per sweep.

Implementation: [science/relatedness/search.py](../../haplotype_reconstruction/science/relatedness/search.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(F²); O(9 M); O(F² M). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rl14"></a>
## rl14 Sweep loop, the two searches and the kept fit

Agglomerate, save the starting partition, then run parent-slot assignment, best-improvement moves, kept-slot splits and one-merge sweeps. Restore the start and repeat with content-order moves. Stop each search at no change or its sweep cap. Keep the second only when it clears the total-objective tolerance; retain explicit convergence and search-margin diagnostics.

Implementation: [science/relatedness/run.py](../../haplotype_reconstruction/science/relatedness/run.py).

Parameters by name: `max_sweeps`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: Two bounded searches, each up to the named sweep cap, with member/parent/slot scans per sweep.

<a id="rl15"></a>
## rl15 Gain screens and kept-gain caches (decision-neutral acceleration)

Use conservative float32 screens with declared error bounds, promoting every near-best or undecided comparison to the exact family score. Cache only gains whose dependency state is unchanged. The screen changes work, never the candidate universe or the acceptance cut-off.

Implementation: [science/relatedness/family_kernels.py](../../haplotype_reconstruction/science/relatedness/family_kernels.py), [science/relatedness/partition.py](../../haplotype_reconstruction/science/relatedness/partition.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N M) screen workspace and O(N²) proposal-cache storage; exact fallback preserves decisions.

<a id="rl16"></a>
## rl16 Support units on the measured map

Divide measured map length into support units and map cuts to the first site at or above each cut. Drop empty units from the denominator. Without positive measured length, use one nonempty unit; fallback genetic distance is not invented measured support.

Implementation: [science/relatedness/run.py](../../haplotype_reconstruction/science/relatedness/run.py).

Parameters by name: `support_unit_morgans`, `rate_is_measured`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(M_c log M_c). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rl17"></a>
## rl17 Member support and parent-slot support

For each nonempty support unit, compare a member’s stay likelihood with other families/new singleton and compare each named parent slot with its latent replacement. Count strict supported gains and divide by nonempty units. Latent-slot support is missing. A slot sample is a sequenced sample whose genotypes serve the family model as a parent's would. It may be a parent, a child (for a one-member family the two directions tie), a sib or another close relative. Parent calls, direction and their support belong to the pedigree stage. These support comparisons have their own target exclusions, distinct from search-cycle rules.

Implementation: [science/relatedness/support.py](../../haplotype_reconstruction/science/relatedness/support.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(9 F N M). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rl18"></a>
## rl18 Final half-sib links

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Enumerate pairs of entirely latent families in recorded order and report half-sib links strictly above the log-sample-count threshold. Do not require the same pair to win a full-sib comparison. Links are a diagnostic after the retained family fit, including fits stopped by a sweep budget.

Implementation: [science/relatedness/support.py](../../haplotype_reconstruction/science/relatedness/support.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(18 M); O(F_lat² M). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rl19"></a>
## rl19 Product, identity, barrier placement, reuse, shard consumption and published tables

Publish the family fit. Canonicalise family IDs as dense ranks by smallest member; keep membership and parent/support axes aligned. Schedule the family gather after all evidence, while independent reconstruction may overlap it.


Implementation: [science/relatedness/run.py](../../haplotype_reconstruction/science/relatedness/run.py), [stages/graph.py](../../haplotype_reconstruction/stages/graph.py), [interface/run.py](../../haplotype_reconstruction/interface/run.py).

Parameters: supplied typed inputs and the stage's [parameter record](../reference/parameters.md).

Cost and scope: Family products scale with N.

<!-- named mechanism anchors -->

## Named mechanisms

<a id="pairwise-ibd"></a>

**Pairwise ibd:** not estimated; see [Pairwise identity by descent](#pairwise-identity-by-descent).
