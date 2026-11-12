# Relatedness

Relatedness fits a composite full-sib/half-sib family model from genotype likelihoods. Its latent or sequenced parent slots carry no sex, generation or sample-name prior. N is samples, M selected family sites, L pair-chain sites and U nonempty support units. The two bounded searches are alternatives, not an exhaustive pedigree search. Pairwise IBD is an optional output and is off by default.

The sections below identify semantic units, not separately configurable pipeline steps. [Outputs](../user/outputs.md), [record contracts](../reference/records.md) and [costs](../developer/costs.md) describe their public boundaries.

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
## rl03 Site selection and thinning (family sites, IBD sites)

For family sites, apply the inclusive minor-frequency gate and compare the floating mean of the Boolean observation mask with the inclusive observed-share threshold, then thin the survivors. The optional pair chain takes the direct `::ibd_site_step` slice without those family gates; it handles joint observation inside the chain. Keep the two evidence axes separate.

Implementation: [science/relatedness/run.py](../../haplotype_reconstruction/science/relatedness/run.py), [science/relatedness/ibd.py](../../haplotype_reconstruction/science/relatedness/ibd.py).

Parameters by name: `minor_allele_frequency`, `minimum_observed`, `family_site_step`, `ibd_site_step`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L_c). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rl04"></a>
## rl04 Pairwise three-state IBD chain and its genome-wide summaries

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

For each selected sample pair, run the three-state IBD chain along each contig with the stated allele-frequency emissions and map transition law. Aggregate posterior state shares over jointly observed sites and compute the recorded IBD0/IBD2 stretch evidence. This optional output costs quadratically in sample count and does not feed family fitting.

Implementation: [science/relatedness/ibd.py](../../haplotype_reconstruction/science/relatedness/ibd.py).

Parameters by name: `ibd_switch_rate`, `ibd_site_step`, `initial`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(P · L/ibd_site_step · c). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rl05"></a>
## rl05 Family-model site terms (child terms, latent prior, sequenced-parent posterior)

Construct child-genotype terms, the latent Hardy–Weinberg parent prior and sequenced-parent genotype posterior on family sites. Floor logs before using them in the composite model; normalise posterior states in the declared order. A sequenced sample’s posterior is not a true parent label.

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
## rl10 Parent assignment (one per open family per sweep)

For each family’s first free parent slot, offer its best non-member and eligible member candidates. Rank offers by gain and content, then enforce the log-sample-count penalty, per-sweep use rules and acyclicity at application. Only one non-member is offered per family; failure does not trigger a second-best search.

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
## rl12 Content-order first-improvement moves

Visit samples in fixed content order. Compare the new-singleton target first and valid live families thereafter, using the declared key tie rule. Apply a sufficient gain immediately and refresh changed-family leave gains. This first-improvement schedule is intentionally distinct from the best-improvement search.

Implementation: [science/relatedness/search.py](../../haplotype_reconstruction/science/relatedness/search.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N M); O(N). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rl13"></a>
## rl13 One merge per sweep

Enumerate compatible live-family pairs, compare full-sib and applicable half-sib gains, and rank sufficient merge offers. Apply the first acyclic merge; undo a rejected cyclic trial without changing the remaining offer order. At most one merge is accepted per sweep.

Implementation: [science/relatedness/search.py](../../haplotype_reconstruction/science/relatedness/search.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(F²); O(9 M); O(F² M). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rl14"></a>
## rl14 Sweep loop, the two searches and the kept fit

Agglomerate, save the starting partition, then run parent assignment, best-improvement moves and one-merge sweeps. Restore the start and repeat with content-order moves. Stop each search at no change or its sweep cap. Keep the second only when it clears the total-objective tolerance; retain explicit convergence and search-margin diagnostics.

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
## rl17 Member support and parent support

For each nonempty support unit, compare a member’s stay likelihood with other families/new singleton and compare each named parent slot with its latent replacement. Count strict supported gains and divide by nonempty units. Latent-parent support is missing. These support comparisons have their own target exclusions, distinct from search-cycle rules.

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

Publish family fitting separately from the optional pair chain. Canonicalise family IDs as dense ranks by smallest member; keep membership and parent/support axes aligned. Schedule the family gather after all evidence, while independent reconstruction may overlap it. Persist pair selection outside scientific identities and render only selected, checked pair products.


Implementation: [science/relatedness/run.py](../../haplotype_reconstruction/science/relatedness/run.py), [stages/graph.py](../../haplotype_reconstruction/stages/graph.py), [interface/run.py](../../haplotype_reconstruction/interface/run.py).

Parameters: supplied typed inputs and the stage's [parameter record](../reference/parameters.md).

Cost and scope: Family products scale with N; optional pair products have N(N−1)/2 rows and O(N² L) chain work.

<!-- named mechanism anchors -->

## Named mechanisms

<a id="pairwise-ibd"></a>

**Pairwise ibd:** see [RL04](#rl04).
