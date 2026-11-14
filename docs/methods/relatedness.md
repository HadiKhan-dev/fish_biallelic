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
## rl16 Support units on the working map

Pool the family sites of every contig into one axis in configured contig order: each contig's sites at their
working-map coordinates (the configured map, or its rate where it applies), its first family site at the running
offset, with no gap between contigs. Cut the axis into consecutive units of `support_unit_morgans`; a final piece
shorter than `support_tail_morgans` joins the unit before it; a site exactly at a cut belongs to the unit on its
right; empty units are omitted. A unit may straddle contigs. Each unit is weighted by its working Morgans: the
coordinate of its last site minus that of the previous unit's last site, so a gap without family sites is carried by
the next unit and the weights sum to the axis length. An axis of no length is one unit of weight 1. Unit composition
depends on the configured contig order, a deterministic convention; shifting the grid by half a unit moves
individual supports but no measured guard decision.

Implementation: [science/relatedness/run.py](../../haplotype_reconstruction/science/relatedness/run.py),
[science/shared/material.py](../../haplotype_reconstruction/science/shared/material.py).

Parameters by name: `support_unit_morgans`, `support_tail_morgans`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(M log M). These terms describe the unit's work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rl17"></a>
## rl17 Member support and parent-slot support

For each nonempty support unit, compare a member’s stay likelihood with other families/new singleton and compare each named parent slot with its latent replacement. Weight each strict supported gain by its unit's working Morgans and divide by the total: member and parent-slot support (`FamilySupport`, `ParentSlot*Support`) are Morgan-weighted shares. Latent-slot support is missing. A slot sample is a sequenced sample whose genotypes serve the family model as a parent's would. It may be a parent, a child (for a one-member family the two directions tie), a sib or another close relative. Parent calls, direction and their support belong to the pedigree stage. These support comparisons have their own target exclusions, distinct from search-cycle rules.

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

Publish the family fit, with the recombination-rate decision of [rl20](#rl20). Canonicalise family IDs as dense ranks by smallest member; keep membership and parent/support axes aligned. Schedule the family gather after all evidence, while independent reconstruction may overlap it. With `[genetic_map] infer_rate`, each contig's initial local selection, its chain's first map consumer, waits for the gather; discovery still overlaps it, and the `reconstruct` step needs the `relatedness` step first.


Implementation: [science/relatedness/run.py](../../haplotype_reconstruction/science/relatedness/run.py), [stages/graph.py](../../haplotype_reconstruction/stages/graph.py), [interface/run.py](../../haplotype_reconstruction/interface/run.py).

Parameters: supplied typed inputs and the stage's [parameter record](../reference/parameters.md).

Cost and scope: Family products scale with N.

<a id="rl20"></a>
## rl20 Genome-wide recombination rate from full-sib families and the working map

The configured fallback rate (`[genetic_map] rate_cm_per_mb`, default 5 cM/Mb) sets the genetic distance of every contig without a map file, and every stage after relatedness reads it: local selection's context refit, assembly joins, linker and refinement, painting transitions, pedigree preparation and scoring, family phase and the recombination priors. The stage estimates one genome-wide rate from its families, as a check of that rate and, on request, as its replacement.

**Unit.** A pair (i, j) of one family f with parent slots u and v. Per family site the law of the parents' genotype pair leaves the pair out: `w(G_u, G_v) ∝ slot_u(G_u) slot_v(G_v) Π_{k ∈ f, k ≠ i, j} C_k(G_u, G_v)`, where a slot is a sequenced slot's genotype posterior or the latent Hardy–Weinberg prior and C are the child terms of [rl05](#rl05). Site selection therefore uses no read of the pair, whose meioses are independent of the other members'.

**Chains.** With at least one sequenced slot, the pair follows the two indicators (I_u, I_v): whether it received the same copy of each parent. Each flips at 2c per bp, a crossover in either sib's meiosis from that parent. It is read on sites whose most likely parental genotype pair has leave-pair-out posterior at least `site_posterior` with a heterozygous parent, and that the other members alone, with both slots latent, support with posterior at least `segregation_posterior` (a sequenced slot's confident false heterozygous calls cannot select sites its children do not segregate at). With two latent slots the slots are exchangeable, so a one-heterozygous site cannot say which parent's indicator it informs; only sites whose both-heterozygous posterior is at least `site_posterior` are used, with the XOR chain Z = I_u xor I_v (one shared copy against none or two), which flips at 4c.

**Emission.** The selection only chooses the sites. At each selected site the pair's likelihood of each chain state is averaged over the leave-pair-out law of the parents' genotype pair (all nine pairs), the exact marginal. Conditioning on the most likely pair instead treats its residual posterior as zero: in a simulated sib-ship whose two parents were sequenced at depth 10.5, the parents' genotype miscalls (which parent is heterozygous, invisible to the unordered segregation check) then read as inheritance changes, +17 % against the realised rate; with the parents' true genotypes the estimate was right. Genotype errors the likelihoods do not represent are a second nuisance: each site's state emissions are mixed with their state mean, e' = (1 − ε) e + ε ē, with ε fitted by maximum likelihood on the `error_rates` grid (simulations select ε of 0–0.001; TroMau selects 0.03, a gain of 712 nats). Both chains also carry an artefact-stretch process (stretches start at rate rho per bp and last ell bp on average; inside one the emission is the state average). (ε, rho, ell) are nuisances of the data set, fitted with c by maximum likelihood.

**Estimate.** The composite log-likelihood weights each pair 1/(n_f − 1), so a family's about 2 n_f meioses count about once; families below `minimum_information` (members − 2 + 2 × sequenced slots) are not used. The estimate maximises it on a log-spaced grid of `rate_points` rates from `rate_low` to `rate_high` with parabolic refinement on log rate. Families are dropped one at a time while a family's own composite maximum exceeds its value at the leave-one-family-out maximiser by more than `inconsistency_nats` (a family that merged two maternal half-sib groups, or took in a grandchild). The 95 % interval is a two-level percentile bootstrap (`bootstrap_replicates`, fixed `bootstrap_seed`, nuisances refitted): families with replacement, then each drawn family's chromosomes with replacement, when there are at least `minimum_interval_units` contigs; otherwise families only, when there are that many. Meioses cluster in families (shared parents, sex and background), so a bootstrap over chromosomes alone conditions on the families and ignores their variation; on 15 simulations and pearly the two-level interval covered the realised rate in every case, the chromosome bootstrap in 12 of 15.

**Gate.** The estimate is usable only when an informative family remains, the estimate lies inside the grid (`edge_low`, `edge_high`), the meiosis-weighted mean of the families' median informative sites per pair is at least `minimum_determined_sites` (a depth and design guard: 1x runs score 2–200, 5x runs 3,000–20,000), an interval exists and the two-level interval's half-width is at most `maximum_half_width` of the estimate. The estimate needs large full-sib families sequenced at about 5x or more: in the 4.18 rate study all depth-1x runs and small-family designs fell back.

**Decision.** One fixed rate for the genome; the per-contig values are reported as diagnostics only and never enter a map.

- Default (`infer_rate` off): the estimate is a diagnostic. When the gate passes and the supplied rate lies outside the interval, a warning is printed once in the run log and recorded in `relatedness/summary.json`. Nothing downstream reads the estimate, so released results do not depend on it.
- `[genetic_map] infer_rate = true` (command line `--infer-rate`): when the gate passes, the supplied rate lies outside the interval and at least `minimum_adoption_families` informative families were used, the estimate, rounded to `working_digits` significant digits, becomes the fallback rate of every contig without a map file (the working map). Every later stage reads it; the pedigree's material (U, its units and the exclusion weights) is measured on this working map, so an adopted rate enters it. The relatedness support units are computed before the decision, on the supplied map.
- The family rule is structural, not a tuned threshold: the rate can differ between families (sex of the parents, genetic background), and with fewer than five families that variation cannot be measured, whatever the interval says. Such a run keeps its supplied rate and, when that rate lies outside the interval, warns; a real cichlid cross with two informative families is one.
- Always: a contig's map file is never changed. To force the supplied rate, leave `infer_rate` off (the default).

**Identity.** The working map is a relatedness product. Without `infer_rate` every consumer binds the configured map's identity from the run manifest and no consumer waits for relatedness. With it, each consumer's derivation binds the working map's digest read from the published relatedness record, and each contig's chain waits for that record, so resume and record reuse follow the decision.

Implementation: [science/relatedness/rate.py](../../haplotype_reconstruction/science/relatedness/rate.py), [stages/relatedness/families_task.py](../../haplotype_reconstruction/stages/relatedness/families_task.py), [stages/relatedness/working_map.py](../../haplotype_reconstruction/stages/relatedness/working_map.py).

Parameters by name: `site_posterior`, `segregation_posterior`, `minimum_information`, `rate_low`, `rate_high`, `rate_points`, `artefacts`, `error_rates`, `inconsistency_nats`, `bootstrap_replicates`, `bootstrap_seed`, `minimum_interval_units`, `minimum_determined_sites`, `maximum_half_width`, `minimum_adoption_families`, `edge_low`, `edge_high`, `working_digits`, `infer_rate`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(Σ_f n_f² · M · rate_points · artefacts · error_rates) forward steps over each pair's selected sites (pairs are independent and run in parallel), plus `bootstrap_replicates` two-level grid fits. No dependence on founder count K.

<!-- named mechanism anchors -->

## Named mechanisms

<a id="pairwise-ibd"></a>

**Pairwise ibd:** not estimated; see [Pairwise identity by descent](#pairwise-identity-by-descent).
