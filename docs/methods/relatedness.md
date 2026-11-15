# Relatedness

Relatedness fits a composite full-sib/half-sib family model from genotype likelihoods. Its latent or sequenced parent slots carry no sex, generation or sample-name prior. N is samples, M selected family sites and U nonempty support units. The two bounded searches are alternatives, not an exhaustive pedigree search. The stage estimates no pairwise identity-by-descent shares ([below](#pairwise-identity-by-descent)).

The sections below identify semantic units, not separately configurable pipeline steps. [Outputs](../user/outputs.md), [record contracts](../reference/records.md) and [costs](../developer/costs.md) describe their public boundaries.

## Pairwise identity by descent

The stage reports no pairwise IBD0/IBD1/IBD2 shares. A pairwise IBD chain on genotype likelihoods treats the alleles two samples do not share by descent as independent Hardy–Weinberg draws at the sample allele frequency, with sites independent given the IBD state. In a cross every sample carries mosaics of a few founder haplotypes: alleles not shared by descent come from the same few haplotypes, in complete linkage disequilibrium over long blocks, and the sample frequency is the frequency of no base population. Identity by state at sites where a parent is homozygous, or where the founder groups differ, then reads as identity by descent; only the founder haplotypes or the parents' genotypes show which parental homologue a child received at such a site, and this stage runs before the founder haplotypes exist. Measured on 27 simulated crosses at depths 1, 5 and 10.5 against the simulated founder-haplotype transmission, such a chain gave parent–offspring pairs mean IBD0 shares of 0.22–0.59 per cross (the truth is 0) and full sibs 0.36–0.66 (the truth is 0.09–0.26); it was close to the truth only in a cross of 60 sequenced founders and their children. On the TroMau cross it gave the 69 child–parent pairs that the pedigree stage calls IBD0 shares of 0.25–0.84, where a parent and child share at least one homologue throughout.

Sharing of founder haplotypes is defined once the founder haplotypes exist. The released paintings name, bin by bin, the founder rows each sample carries ([PR13](painting.md#pr13)), and the pedigree stage scores pairs of samples on the same painting classes ([PD28](pedigree.md#pd28)). The optional pair-sharing report counts, for every pair of samples, the founder rows their named tracks share ([PR20](painting.md#pr20)).

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

Construct child-genotype terms, the latent Hardy–Weinberg parent prior and sequenced-parent genotype posterior on family sites. (With two source populations these become terms over parental ancestry configurations and ancestry classes: [rl21](#rl21).) Floor logs before using them in the composite model; normalise posterior states in the declared order. A slot sample is a sequenced sample whose genotypes serve the family model as a parent's would. It may be a parent, a child (for a one-member family the two directions tie), a sib or another close relative. Parent calls, direction and their support belong to the pedigree stage.

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

Publish the family fit, with the recombination-rate decision of [rl20](#rl20). Canonicalise family IDs as dense ranks by smallest member; keep membership and parent/support axes aligned. Schedule the family gather after all evidence, while independent reconstruction may overlap it. With `[genetic_map] rate_estimate = "adopt"`, each contig's initial local selection, its chain's first map consumer, waits for the gather; discovery still overlaps it, and the `reconstruct` step needs the `relatedness` step first.


Implementation: [science/relatedness/run.py](../../haplotype_reconstruction/science/relatedness/run.py), [stages/graph.py](../../haplotype_reconstruction/stages/graph.py), [interface/run.py](../../haplotype_reconstruction/interface/run.py).

Parameters: supplied typed inputs and the stage's [parameter record](../reference/parameters.md).

Cost and scope: Family products scale with N.

<a id="rl20"></a>
## rl20 Genome-wide recombination rate from full-sib families and the working map

The configured fallback rate (`[genetic_map] rate_cm_per_mb`, default 5 cM/Mb) sets the genetic distance of every contig without a map file, and every stage after relatedness reads it: local selection's context refit, assembly joins, linker and refinement, painting transitions, pedigree preparation and scoring, family phase and the recombination priors. The stage estimates one genome-wide rate from its families, as a check of that rate and, on request, as its replacement. It is the early of the run's two rate estimates: the only one available before discovery, so the only one that can set the working map of every stage. The late estimate, fitted on the paintings (pedigree [map scale](pedigree.md#map-scale)), never changes the working map; it checks its scale for the pedigree's material regime only. A run therefore has one working map.

**Unit.** A pair (i, j) of one family f with parent slots u and v. Per family site the law of the parents' genotype pair leaves the pair out: `w(G_u, G_v) ∝ slot_u(G_u) slot_v(G_v) Π_{k ∈ f, k ≠ i, j} C_k(G_u, G_v)`, where a slot is a sequenced slot's genotype posterior or the latent Hardy–Weinberg prior and C are the child terms of [rl05](#rl05). Site selection therefore uses no read of the pair, whose meioses are independent of the other members'.

**Chains.** With at least one sequenced slot, the pair follows the two indicators (I_u, I_v): whether it received the same copy of each parent. Each flips at 2c per bp, a crossover in either sib's meiosis from that parent. It is read on sites whose most likely parental genotype pair has leave-pair-out posterior at least `site_posterior` with a heterozygous parent, and that the other members alone, with both slots latent, support with posterior at least `segregation_posterior` (a sequenced slot's confident false heterozygous calls cannot select sites its children do not segregate at). With two latent slots the slots are exchangeable, so a one-heterozygous site cannot say which parent's indicator it informs; only sites whose both-heterozygous posterior is at least `site_posterior` are used, with the XOR chain Z = I_u xor I_v (one shared copy against none or two), which flips at 4c.

**Emission.** The selection only chooses the sites. At each selected site the pair's likelihood of each chain state is averaged over the leave-pair-out law of the parents' genotype pair (all nine pairs), the exact marginal. Conditioning on the most likely pair instead treats its residual posterior as zero: in a simulated sib-ship whose two parents were sequenced at depth 10.5, the parents' genotype miscalls (which parent is heterozygous, invisible to the unordered segregation check) then read as inheritance changes, +17 % against the realised rate; with the parents' true genotypes the estimate was right. Genotype errors the likelihoods do not represent are a second nuisance: each site's state emissions are mixed with their state mean, e' = (1 − ε) e + ε ē, with ε fitted by maximum likelihood on the `error_rates` grid (simulations select ε of 0–0.001; TroMau selects 0.03, a gain of 712 nats). Both chains also carry an artefact-stretch process (stretches start at rate rho per bp and last ell bp on average; inside one the emission is the state average). (ε, rho, ell) are nuisances of the data set, fitted with c by maximum likelihood.

**Estimate.** The composite log-likelihood weights each pair 1/(n_f − 1), so a family's about 2 n_f meioses count about once; families below `minimum_information` (members − 2 + 2 × sequenced slots) are not used. The estimate maximises it on a log-spaced grid of `rate_points` rates from `rate_low` to `rate_high` with parabolic refinement on log rate. Families are dropped one at a time while a family's own composite maximum exceeds its value at the leave-one-family-out maximiser by more than `inconsistency_nats` (a family that merged two maternal half-sib groups, or took in a grandchild). The 95 % interval is a two-level percentile bootstrap (`bootstrap_replicates`, fixed `bootstrap_seed`, nuisances refitted): families with replacement, then each drawn family's equal-material units with replacement, when the fit's material holds at least `minimum_interval_units` units; otherwise families only, when there are that many families. The units are the 1-Morgan units of the support axis ([rl16](#rl16), DESIGN_417 D5: the working map's family-site coordinates, contigs in configured order, a final piece shorter than half a unit joining the one before it); each pair's forward log-likelihood is split exactly into per-site predictive terms, summed per unit. No count of chromosomes decides anything (owner, 13 Nov 2026; until 4.18.2 the units were whole chromosomes and at least five contigs were needed). Meioses cluster in families (shared parents, sex and background), so a bootstrap over units alone conditions on the families and ignores their variation; on 15 simulations and a second cross the two-level family x chromosome interval covered the realised rate in every case, the chromosome bootstrap in 12 of 15. On the 16 simulations with truth the family x unit interval also covered the realised rate in every case, with the same gate decisions and a median half-width of 0.109 (0.105 for family x chromosome); single chromosomes of 5 to 10 Morgans now get an interval, and on the tested ones the determined-sites gate still decides.

**Gate.** The estimate is usable only when an informative family remains, the estimate lies inside the grid (`edge_low`, `edge_high`), the meiosis-weighted mean of the families' median informative sites per pair is at least `minimum_determined_sites` (a depth and design guard: 1x runs score 2–200, 5x runs 3,000–20,000), an interval exists and the two-level interval's half-width is at most `maximum_half_width` of the estimate. The estimate needs large full-sib families sequenced at about 5x or more: in the 4.18 rate study all depth-1x runs and small-family designs fell back.

**Map files.** The estimate stands in only for the fallback rate: contigs without a map file and the stretches beyond a map file's knots. A map file's own distances, with their varying local rate, are never rescaled. In a genome that mixes file and map-free contigs the estimate is fitted on the map-free contigs only, so that file contigs (another, varying rate) cannot contaminate the quantity it may replace; their per-chromosome values are reported as NaN. A genome fully covered by map files is fitted on every contig and the estimate is a diagnostic only (nothing to replace).

**Decision.** One fixed rate for the genome; the per-contig values are reported as diagnostics only and never enter a map. `[genetic_map] rate_estimate` (command line `--rate-estimate`) chooses what the run's two estimates may change:

- `veto` (default) and `ignore`: this estimate is a diagnostic. When the gate passes and the supplied rate lies outside the interval, a warning is printed once in the run log and recorded in `relatedness/summary.json`. Nothing downstream reads it, so released results do not depend on it (with `veto` the painting estimate may still withhold pedigree calls: [map scale](pedigree.md#map-scale); with `ignore` nothing reads either estimate: the forced configured map).
- `adopt`: when the gate passes, the supplied rate lies outside the interval and at least `minimum_adoption_families` informative families were used, the estimate, rounded to `working_digits` significant digits, becomes the fallback rate of the working map: every contig without a map file and every stretch beyond a map file's knots. Every later stage reads it; the pedigree's material (U, its units and the exclusion weights) is measured on this working map, and the painting estimate is then compared with it. The relatedness support units are computed before the decision, on the supplied map.
- The family rule is structural, not a tuned threshold: the rate can differ between families (sex of the parents, genetic background), and with fewer than five families that variation cannot be measured, whatever the interval says. Such a run keeps its supplied rate and, when that rate lies outside the interval, warns; a real cichlid cross with two informative families is one. It counts families, not chromosomes, and is kept.
- Always: a map file's own distances never change.

**Identity.** The working map is a relatedness product. Unless `rate_estimate = "adopt"` every consumer binds the configured map's identity from the run manifest and no consumer waits for relatedness. With adopt, each consumer's derivation binds the working map's digest (computed from the manifest's map documents and the adopted rate) read from the published relatedness record, and each contig's chain waits for that record, so resume and record reuse follow the decision.

Implementation: [science/relatedness/rate.py](../../haplotype_reconstruction/science/relatedness/rate.py), [stages/relatedness/families_task.py](../../haplotype_reconstruction/stages/relatedness/families_task.py), [stages/relatedness/working_map.py](../../haplotype_reconstruction/stages/relatedness/working_map.py).

Parameters by name: `site_posterior`, `segregation_posterior`, `minimum_information`, `rate_low`, `rate_high`, `rate_points`, `artefacts`, `error_rates`, `inconsistency_nats`, `bootstrap_replicates`, `bootstrap_seed`, `minimum_interval_units`, `minimum_determined_sites`, `maximum_half_width`, `minimum_adoption_families`, `edge_low`, `edge_high`, `working_digits`, `rate_estimate`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(Σ_f n_f² · M · rate_points · artefacts · error_rates) forward steps over each pair's selected sites (pairs are independent and run in parallel, in chunks of 256 pairs whose per-unit tables hold pairs x error_rates x artefacts x rate_points x units values), plus `bootstrap_replicates` two-level grid fits. No dependence on founder count K.

<a id="rl21"></a>
## rl21 Two source populations: local ancestry and ancestry classes of latent parents

The family model of [rl05](#rl05) gives a latent parent a Hardy–Weinberg genotype law at the cohort frequency of [rl02](#rl02). In a cross between two diverged populations or species that law is false wherever the sources differ. Parents from different sources are homozygous for different alleles at a fixed difference, a first-generation hybrid is heterozygous there with probability 1, and neither is what Hardy–Weinberg at a cohort frequency near 1/2 says. The family fit then merges sibships that share ancestry (one latent pair pays the improbable prior once) and breaks up F2 sibships (their parents explain nothing a child alone does not, and cost ln 4 per fixed difference). With `[input] source_populations = 2` the stage replaces that law as follows; with 1 (the default) none of this runs and rl01–rl20 are unchanged.

**Sources and shares.** Each site has two source frequencies (a, b) of the ALT allele, and each position of a genome is in one of three ancestry states: both homologues from source A, one from each, both from B. In a state the genotype follows the two-homologue law at the homologues' source frequencies. An EM over each sample's genome-wide state shares z and the sites' source frequencies runs on every s-th family site in genome order (at most `source_fit_sites`), from `source_starts` seeded starts (best log likelihood kept); every family site's frequencies are then fitted at those shares (`source_site_iterations` steps, kept in [`source_clip`, 1 − `source_clip`]). Only genotype likelihoods enter: no sample name, order or label. Which source is called A is arbitrary and changes no output.

**Local ancestry.** In a cross the states run along the genome in blocks of many Morgans, so a sample's composition is a few dozen draws, not one per site. Each sample's state is therefore read from its own reads by a three-state chain along the working map (the state is kept over a distance d with probability exp(−`source_switch_rate` · d), else redrawn from z), and at every site the state probabilities are those given the sample's *other* sites.

**Family terms.** A parent's configuration at a site is its state with the alleles on its homologues: A/A and B/B by genotype (three each), A/B by the allele on the A homologue and the allele on the B homologue (four). A child's term for a pair of parental configurations is the probability of its reads given the homologues its own state says it received: an A/A child received an A homologue from each parent, an A/B child an A homologue from one and a B homologue from the other. A state the pair cannot produce gets `source_state_error`. The term is conditional on the child's state, so:

- a child's ancestry composition is not scored site by site as evidence about its parents (alone, it has the same likelihood under every pair of parental classes that can produce its states);
- a child a pair of parents cannot have produced is still excluded (an F2 is not a child of two pure parents);
- a fixed difference is exactly neutral for an F2 sibship, whose parents are heterozygous there and whose children's genotypes are their states; the sibship evidence is at the sites polymorphic inside a source, sharpened by knowing which homologue each child received.

What a sample's ancestry along the map does say about its parents is counted once, at the scale of its blocks: the likelihood of its reads when its two homologues come from parents of given transmitted shares (each received homologue's source a two-state chain along the map, redrawn at `source_switch_rate` / 2 per Morgan; a first-generation hybrid's gamete changes source at each crossover, a pure parent's never) against its own chain.

**Latent parents.** A latent parent has an ancestry class that is constant over its family's sites: the unordered pair of its two homologues' types, a type being a share of source B (`source_shares`: 0, 1/2 for a homologue transmitted by a first-generation hybrid, 1). The six classes are the two pure ones, the first-generation hybrid (always in the A/B state: represented exactly), the two backcrosses and the F2-or-later class. A class is a law over the configurations: the state drawn at the class's shares, the alleles at the source frequencies. A family's term ([rl06](#rl06)) is the marginal over the classes of its latent slots with equal weights; the shared parent of a half-sib link has its own class. A sequenced slot is its sample's configuration posterior (its state from its other sites, its alleles from its reads). Every search step, support unit ([rl17](#rl17), each unit scored on its own sites) and half-sib link uses these terms; gains are exact (the float32 screens of [rl15](#rl15) do not run).

**Rate estimate.** [rl20](#rl20) reads a latent slot's genotype law as the classes' laws mixed at the family's class posterior, and a sequenced slot's as the genotype marginal of its configuration posterior. For an F2 family with unsequenced first-generation parents every fixed difference is then a both-heterozygous site.

**Scope (measured on simulations with known pedigrees).** Counts are pairs of sequenced samples: true full sibs placed in one family / other pairs placed in one family / true full sibs split; a control is the same pedigree drawn from one population and fitted with one source.

- *Validated: about 5 reads per site, parents pure or first-generation hybrids.* Held-out interspecies F2 crosses (three females of one species, one male of the other, 116 samples, three runs): one source 224 / 445 / 829, two sources 1,034 / 26 / 19, controls 1,037 / 13 / 16. Slot samples that are true parents of the member rose from 116 to 279 (member, slot) pairs and non-parents fell from 131 to 17. The rate estimate of [rl20](#rl20) went from one or two families (twice with a passing gate and an interval that excluded the simulated 10 cM/Mb) to 14–15 families with intervals that covered it in all three runs. Two-species cohorts with pure, hybrid and backcross-type parents reached their controls exactly at depth 5 (192 / 26 / 12; 280 / 0 / 0 twice) and at depth 1 gave 166 / 4 / 38 against 132 / 91 / 72 with one source.
- *About 1 read per site: not shown to help on the interspecies F2 design.* True-sib pairs together rose from 20 to 889 of 1,053, but sibships whose parents are sibs (double first cousins) merged around a sequenced aunt or uncle in a parent slot: other pairs together rose from 81 to 1,113 (controls: 785 / 119 / 268). With three founders of one source and one of the other, cousins carry the same founder haplotypes wherever the grandparents are homozygous, and one read per site rarely shows the mismatch.
- *Unsequenced later-generation parents.* The local ancestry of a latent F2 or backcross parent is drawn independently per site, not followed along the map. F3 sibships with unsequenced F2 parents: at depth 5 about as with one source (two sources 423 / 19 / 21 over two runs, one source 398 / 5 / 46, controls 438 / 0 / 6); at depth 1 worse (383 / 288 / 61 against 351 / 59 / 93).
- *Weakly diverged sources at 1 read per site are not identified* (one run: 7 / 30 / 344 against 60 / 15 / 291 with one source); at depth 5 the same sources are (346 / 11 / 5 against 266 / 6 / 85).
- *Declaring two sources for a one-population cohort splits sibships.* The fit reads families as sources and the child terms trust the resulting states: controls of the interspecies F2 design at depth 5 went from 335 / 13 / 16, 351 / 0 / 0 and 351 / 0 / 0 to 118 / 15 / 233, 173 / 1 / 178 and 166 / 11 / 185. An automatic choice of the number of sources was tested and rejected: a cross from few founders is itself an admixture of founder haplotypes, and no admixture statistic separated the two cases with a margin.
- *A real cross, without truth.* On a real interspecies F2 cross at about 10 reads per site, declaring two sources reduced the families from 72 to 40 and the one-member families from 47 to 13, but the members best placed in their family on at least 75 % of the support units fell from 15 to 4. The cause is not established, and neither warning below fires there. That cross is run with one source.
- Two sources only (three-way crosses are not modelled); later backcross generations are represented by their nearest class; sources are identified by sequenced samples that are not all F2-like.

**Warnings.** Two texts are printed in the run log and recorded in `relatedness/summary.json` (`source_structure.warnings`); no result depends on them.

- *Low depth:* the cohort's estimated reads per family site and sample (the median over samples of −ln(1 − the sample's share of family sites with a read)) is below `source_validated_depth`, the midpoint of the depth that met the criteria (5) and the depth that did not (1).
- *Not a two-source cross, or sources not identified:* fewer than `source_minimum_strong` of the family sites differ by more than `source_strong_differentiation` in allele frequency between the fitted sources. In the study 16 one-population cohorts declared as two sources had 0.000–0.001 of such sites, the cohorts whose ancestry was read correctly 0.04–0.50, and the weakly diverged depth-1 cohort whose sources were not identified 0.000.

Implementation: [science/relatedness/ancestry.py](../../haplotype_reconstruction/science/relatedness/ancestry.py) (shares, frequencies, local ancestry, terms), [science/relatedness/class_kernels.py](../../haplotype_reconstruction/science/relatedness/class_kernels.py) (class-marginal family and half-sib terms), [science/relatedness/partition.py](../../haplotype_reconstruction/science/relatedness/partition.py); recorded in the families product (`sources`) and, with two sources, in `relatedness/summary.json` under `source_structure` (each sample's state shares, each family's latent slot classes, the estimated depth, the share of strongly differentiated sites and the warnings).

Parameters by name: `source_populations`, `source_shares`, `source_clip`, `source_switch_rate`, `source_state_error`, `source_fit_sites`, `source_starts`, `source_iterations`, `source_tolerance`, `source_seed`, `source_site_iterations`, `source_chunk`, `source_differentiation`, `source_strong_differentiation`, `source_minimum_strong`, `source_validated_depth`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: nothing with one source. With two: O(N · `source_fit_sites`) per EM iteration and start, O(N · M · `source_site_iterations`) for the site frequencies, O(N · M) for the chains; child terms over 100 configuration pairs per sample and site (O(100 · N · M) memory), and every family term O(100 · M) exponentials with 36 class pairs in the linear domain, without screens: O(N² · M) for the agglomeration and O(F · N · M) per sweep. No dependence on founder count K.

<!-- named mechanism anchors -->

## Named mechanisms

<a id="pairwise-ibd"></a>

**Pairwise ibd:** not estimated; see [Pairwise identity by descent](#pairwise-identity-by-descent).
