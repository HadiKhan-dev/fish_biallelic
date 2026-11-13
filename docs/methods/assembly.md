# Assembly and founder release

Assembly joins local panels into component paths, compares founder counts and releases conservative allele/identity claims. Its bounded seed, join and continuation searches are distinct from proof of a global optimum. N is evidence samples, S the selected scoring samples, L sites, B blocks, K founder rows, R prepared rows and Q unordered founder-pair states. Context and final assemblies have different stopping roles. Search-dependent costs include natural-domain count refits and repeated count rounds; a complete pass is not guaranteed cubic.

The sections below identify semantic units, not separately configurable pipeline steps. [Outputs](../user/outputs.md), [record contracts](../reference/records.md) and [costs](../developer/costs.md) describe their public boundaries.

<a id="u01"></a>
## u01 Assembly evidence view (neutralised founder evidence, float64/float32 copies, depth keys, float64 release)

Construct neutral founder-visible evidence in float64 and the declared scoring view in float32, preserving positions, observation masks and exact depth-content keys. Decoding into shared storage and copying a supplied evidence record must agree. Release the larger neutral view only at its declared lifetime boundary.

When every discovery site is kept and no founder sample is excluded, the evidence contract already makes every unobserved raw GL triple exactly 1/3. For dense raw inputs the contig adapter borrows read-only GL and observation views in this case, and owns only the float32 scoring copy. Coded inputs expand directly to the active consumer's neutral, scoring and observation arrays. Other inputs use the original masking/copy route; generic likelihood callers keep that route by default. A consumer's release of its neutral field cannot invalidate another view or the raw evidence.

The application chain caches count-free raw/founder evidence and exact depth keys. It hashes identity-checked count windows in bounded sample batches, discarding each decoded batch after use. An `AssemblyEvidenceSource` carries only immutable inputs and keys through selection; the assembly consumer constructs its own view on demand. Scoring and masked-neutral arrays are not kept in the chain cache between consumers. The paired evidence-v2 checkpoint reconstructs coded counts and masks from its GL pair table/code stream; only fallback rows retain those dense fields. Exact reconstructed count rows and founder-mask rows enter the depth-key hash in canonical little-endian order.

Implementation: [science/assembly/evidence.py](../../haplotype_reconstruction/science/assembly/evidence.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="u02"></a>
## u02 Roles, identity binding, level loop, stop rule and refinement hand-off

Complete panels, initialise one component per block, then run hierarchy levels. Evaluate stop reasons on the unrefined hierarchy output in precedence order: single component, no count reduction after the initial level, requested context depth, then final minimum-depth/all-breaks condition. Final-role refinement follows every level and never increases K; its count rounds only remove founders. Carry refined and unrefined first-level context to later refinement.

Implementation: [science/assembly/run.py](../../haplotype_reconstruction/science/assembly/run.py) (stop rules and context transitions) and [stages/assembly/final.py](../../haplotype_reconstruction/stages/assembly/final.py) (level orchestration for final and context roles).

Parameters by name: `minimum_levels`, `batch_size`, `min_boundary_informative_samples`, `selection`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(levels). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="u03"></a>
## u03 Completion: content-keyed two-fold split

Hash the fold seed with each sample’s observation-mask content and depth key, group equal keys, and visit groups in byte order. Assign each whole group to the lighter fold, resolving equal load and inactive groups to fold zero. This fixed content split is not reversed; no random draw or sample label chooses a fold.

Implementation: [science/assembly/completion/crossfit.py](../../haplotype_reconstruction/science/assembly/completion/crossfit.py), [science/assembly/completion/parameters.py](../../haplotype_reconstruction/science/assembly/completion/parameters.py), [science/assembly/completion/run.py](../../haplotype_reconstruction/science/assembly/completion/run.py).

Parameters by name: `fold_seed`, `completion`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L) deterministic fold work with the named sampling stride.

<a id="u04"></a>
## u04 Completion: cross-fitted variational EM, conservative release, held-out carrier profiles

For each held-out fold, fit the other fold with the declared variational update order and monotonic ELBO guard. Without informative training data, retain the pre-fill panel and record absence. Release only conservative site-posterior calls and keep held-out carrier profiles separate from training evidence. The iteration cap is part of the model schedule.

Implementation: [science/assembly/completion/crossfit.py](../../haplotype_reconstruction/science/assembly/completion/crossfit.py), [science/assembly/completion/dosage_statistics.py](../../haplotype_reconstruction/science/assembly/completion/dosage_statistics.py), [science/assembly/completion/observation.py](../../haplotype_reconstruction/science/assembly/completion/observation.py), [science/assembly/completion/support.py](../../haplotype_reconstruction/science/assembly/completion/support.py).

Parameters by name: `max_iterations`, `minimum_iterations`, `elbo_absolute_tolerance`, `elbo_relative_tolerance`, `monotonic_tolerance`, `uniform_mix`, `log_floor`, `minimum_call_probability`, `minimum_carriers`, `UNIFORM_TOLERANCE`, `completion`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L Q + sum_l (Q + C_l r_l^2)); O(N L K^2 + ...); O(N U^3). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="u05"></a>
## u05 Completion: leave-bin-out cavity fills (candidate fills)

Compute outside-bin statistics as prefix plus suffix, assess founder separability outside the focal bin, then consider unknown-allele fills with the site-posterior model. An exchangeable founder cannot acquire a confident fill from that bin alone. A single-bin multi-founder block consequently has no independent outside-bin separation.

Implementation: [science/assembly/completion/cavity_fill.py](../../haplotype_reconstruction/science/assembly/completion/cavity_fill.py), [science/assembly/completion/support.py](../../haplotype_reconstruction/science/assembly/completion/support.py).

Parameters by name: `snps_per_bin`, `minimum_supporters`, `minimum_posterior_probability`, `uniform_mix`, `tie_tolerance`, `founder_identifiability_tolerance`, `minimum_observable_founder_separation`, `log_floor`, `completion`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N K² L) scoring, with transient site tiles sized by the named storage parameter.

<a id="u06"></a>
## u06 Completion: boundary class links

Require informative evidence on both sides and sufficient pooled samples/classes. Form stable reciprocal best matches, withdraw insufficient margins, and apply pooled copy/cost/separation and fold checks in order. Retain only mutually identifiable class links; an exact best/runner-up ambiguity is not resolved by row number.

Implementation: [science/assembly/completion/links.py](../../haplotype_reconstruction/science/assembly/completion/links.py).

Parameters by name: `minimum_pooled_informative_samples`, `minimum_fold_informative_samples`, `minimum_informative_sites_per_sample`, `minimum_effective_copies`, `minimum_fold_effective_copies`, `maximum_mean_cost`, `minimum_pooled_margin`, `minimum_fold_margin`, `minimum_original_hard_anchor_sites`, `tolerance`, `completion`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N K^2); O(K^2 log K); O(K^2 L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="u07"></a>
## u07 Completion: unknown-mass veto, occupancy gate and prepared-panel construction

First veto unexplained-copy underfit and unverifiable singleton panels. Then test the abundant occupancy route; otherwise rank atomic-neighbour rare-continuity witnesses by sample count and probability classes. Conflicting tied witness sample sets abstain. Apply the recorded reason precedence and fill only kept sites. Occupancy uses the named probability quantum, not a copied literal.

Implementation: [science/assembly/completion/observation.py](../../haplotype_reconstruction/science/assembly/completion/observation.py), [science/assembly/completion/occupancy.py](../../haplotype_reconstruction/science/assembly/completion/occupancy.py), [science/assembly/completion/parameters.py](../../haplotype_reconstruction/science/assembly/completion/parameters.py), [science/assembly/completion/run.py](../../haplotype_reconstruction/science/assembly/completion/run.py).

Parameters by name: `minimum_abundant_sum_squared_probability`, `abundant_high_confidence_probability`, `minimum_abundant_high_confidence_carriers`, `minimum_rare_local_carrier_probability`, `minimum_rare_neighbor_carrier_probability`, `minimum_same_sample_continuity`, `minimum_rare_continuity_samples`, `maximum_discovery_unknown_mass`, `completion`, `occupancy`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N K L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="u08"></a>
## u08 Founder alleles: exact binary-dosage polynomials

Enumerate the binary-dosage polynomial exactly on the admitted cube, accumulating genotype-state contributions in declared pair order. When subtraction threatens a small likelihood, recompute it as a direct positive sum. This is an arithmetic evaluation rule; it neither changes the allele prior nor invents evidence.

Implementation: [science/assembly/alleles/parameters.py](../../haplotype_reconstruction/science/assembly/alleles/parameters.py), [science/assembly/alleles/polynomials.py](../../haplotype_reconstruction/science/assembly/alleles/polynomials.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(Q); O(K^2); O(2^U). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="u09"></a>
## u09 Founder alleles: uncertain-core site posterior above the exact-enumeration ceiling

Above the exact-enumeration ceiling, run damped loopy BP, then deterministic ascent from its mode and complement. Carry the declared single/pair moves, uncertain-core cube or bounded Hamming ball, multimodality and shell guards. Nonconverged BP is flagged. This bounded approximation can withhold a release even when a mode exists.

Implementation: [science/assembly/alleles/site_posterior.py](../../haplotype_reconstruction/science/assembly/alleles/site_posterior.py).

Parameters by name: `MAX_EXACT_BITS`, `SITE_CERTAIN_NATS`, `SITE_BALL_CONFIGURATIONS`, `SITE_BALL_MINIMUM_RADIUS`, `SITE_SHELL_GUARD`, `SITE_BP_DAMPING`, `SITE_BP_TOLERANCE`, `SITE_BP_ITERATIONS`, `SITE_ASCENT_BACKSTOP`, `SITE_RECENTRE_BACKSTOP`, `SITE_IMPROVEMENT`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: m is the number of free founder alleles. BP costs O(N m²) per iteration and ascent O(N m²) per step, under their named iteration/backstop limits. An exact uncertain-core cube costs O(N (m² + 2^c)) with c <= `MAX_EXACT_BITS` (default 10). A radius-r core ball has O(c^r) support; radius three costs O(N m³) per support evaluation and O(c³) support storage, with recentring bounded by `SITE_RECENTRE_BACKSTOP`.

<a id="u10"></a>
## u10 Founder alleles: context-refit site posterior with background state

Use exact site enumeration within its ceiling and the uncertain-core route beyond it, with the context model’s background state and prior. Return count and posterior carrier mass separately. The caller applies its conservative support band; a mass exactly at the threshold is not silently rounded into a supporter.

Implementation: [science/assembly/alleles/parameters.py](../../haplotype_reconstruction/science/assembly/alleles/parameters.py), [science/assembly/alleles/site_posterior.py](../../haplotype_reconstruction/science/assembly/alleles/site_posterior.py).

Parameters by name: `SITE_HALF_BAND`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N (K² + 2^K)) only on the exact route, K <= `MAX_EXACT_BITS` (default 10). Above that ceiling, BP/ascent cost O(N K²) per iteration/step, followed by a bounded core cube or sparse ball as in u09. Support size, BP iterations, ascent steps and recentring limits determine the approximate route’s work; it does not enumerate an unrestricted founder cube.

<a id="u11"></a>
## u11 Hierarchy partition: informative samples, phase breaks, symmetric batches, batch cohorts

Find informative overlap with the exact evidence test, preserve existing break reasons, then introduce insufficient-joint-evidence breaks. Partition each unbroken span into symmetric size-limited batches and derive its cohort. Both sides of a break carry the reason. Batch grouping is deterministic; the completed founder rows remain immutable inputs.

Implementation: [science/assembly/hierarchy/level.py](../../haplotype_reconstruction/science/assembly/hierarchy/level.py), [science/assembly/hierarchy/batches.py](../../haplotype_reconstruction/science/assembly/hierarchy/batches.py).

Parameters by name: `batch_size`, `max_sites`, `linker`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N x proxy sites); O(n N); O(K L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="u12"></a>
## u12 Batch join sequencing, digests and reuse, linker->beam->search, canonical founder order, level output

Create join work only for multi-component batches with a usable cohort. Bind complete evidence, prepared-panel, map, parameter and alternative-branch content in the reuse digest. Run linker, endpoint beam and certified panel search, then canonicalise the resulting founder rows. Pass-through components retain their existing scientific content.

Implementation: [science/assembly/hierarchy/batch.py](../../haplotype_reconstruction/science/assembly/hierarchy/batch.py), [science/assembly/hierarchy/level.py](../../haplotype_reconstruction/science/assembly/hierarchy/level.py).

Parameters by name: `paths_per_endpoint`, `full_scores_per_kind`, `max_bins`, `tensor_budget_mb`, `gap_horizon_scale`, `recombination_tolerance`, `cost_scale`, `join`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N K^3); O(n q K^2 G^2); O(N m^2 K^3). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="u13"></a>
## u13 Alternatives carried between levels (component choices, joint branches bounded, common sites and cohort, parent-join comparison)

Match incoming alternative families to exact component spans and nominal panels before consuming them. Compute common usable sites and cohort across all admitted choices, then enumerate joint branches within the complete-family budget. Preserve unresolved correspondence/budget states and compare parent-join outcomes; never choose a convenient subset of a tied family.

Branches of one batch share their prepared panels, cohort, evidence and settings, so a branch's join search is a function of its join view alone: per input component, in local founder order, the positions, the kept mask and inference calls after the common usable-site masks, the atomic spans and which founders copy the same prepared row. Branches with equal views run one search; each maps the local outcome (row paths and the panels of its families) through its own input rows. This is exact: every branch's record equals the record of searching it alone. It merges no branches and compares no alternatives, so members that are equal only up to a relabelling of founders are still searched apart. The common case is tied input members that differ only in which of two inference-identical prepared rows a founder copies. The batch's work counters count each search once, and its `parent_shadow_join` diagnostic records the number of searches and the search each branch maps. With a tie census attached, every branch is searched, so each branch records its decisions under its own label.

Implementation: [science/assembly/hierarchy/level.py](../../haplotype_reconstruction/science/assembly/hierarchy/level.py), [science/assembly/alternatives.py](../../haplotype_reconstruction/science/assembly/alternatives.py), [science/assembly/hierarchy/batch.py](../../haplotype_reconstruction/science/assembly/hierarchy/batch.py).

Parameters by name: `BRANCH_LIMIT`, `WIDE_CLASS_QUANTA`, `refinement`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N x proxy sites). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="u14"></a>
## u14 Assembly tie rules (per-kind quanta and motions, tie-class orders, truncation, counters, canonical order)

Use named scale-aware assembly policies for likelihoods, proxy gains, beams, refinement and count decisions. Build anchored classes with explicit motion/width guards, then apply linkage, count or content secondary rules as that decision declares. Scientific priors remain fixed under reversal; stable exact-duplicate ordering applies where specified.

Implementation: [core/ties.py](../../haplotype_reconstruction/core/ties.py), [science/assembly/orders.py](../../haplotype_reconstruction/science/assembly/orders.py).

Parameters by name: `WIDE_CLASS_QUANTA`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(n log n); O(n B). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="u15"></a>
## u15 Records, identities, resume and readers

Read completion, joined levels, refined levels and final products through checked typed adapters. Derive expected identities from the complete upstream closure; never trust a completion flag alone. Founder-panel readers prefer a recorded refined level over its join-only level. Report join searches once at the level where they actually ran.

Implementation: [stages/assembly/records.py](../../haplotype_reconstruction/stages/assembly/records.py), [contracts/assembly.py](../../haplotype_reconstruction/contracts/assembly.py), [science/assembly/run.py](../../haplotype_reconstruction/science/assembly/run.py), [science/assembly/hierarchy/batch.py](../../haplotype_reconstruction/science/assembly/hierarchy/batch.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(sum R_b L_b). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="u16"></a>
## u16 Founder release at export (continuity cuts, content withholding, continuation records)

Apply carrier withholding before assembly-family release. Resolve correspondence, distinguish diagnostic count verification from unresolved continuation, and use complete alternative families to cut unsupported continuity and withhold disagreeing alleles. Overflow follows the declared smaller-count/minimum-switch prior and anchor rules. Reorder cut/withheld segments by content, block r5 fill on withheld rows, and retain source mappings.

Implementation: [science/assembly/release.py](../../haplotype_reconstruction/science/assembly/release.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(B A K log K); O(K L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="u17"></a>
## u17 Memory and scheduling estimates of completion tasks and batches (decision-inert)

Calculate completion, link and batch workspace/operation estimates from dimensions and report them to the engine. Shared evidence and whole-branch maxima determine storage plans. Estimates never gate a scientific decision or proof; actual memory pressure governs execution admission without changing candidate budgets.

Implementation: [science/assembly/hierarchy/costs.py](../../haplotype_reconstruction/science/assembly/hierarchy/costs.py), [science/assembly/completion/run.py](../../haplotype_reconstruction/science/assembly/completion/run.py), [science/assembly/hierarchy/level.py](../../haplotype_reconstruction/science/assembly/hierarchy/level.py).

Parameters: supplied dimensions and stage parameter records determine the execution estimates; this unit has no independent bundle setting.

Cost and scope: O(1). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="aj01"></a>
## aj01 Join binned emissions (robust centred site logs summed in per-block bins)

Centre robust site emissions and accumulate each sample/pair/bin in site order. Preserve the bin layout and model uniform penalty. Parallelism is across independent samples; task chunking cannot reorder a bin’s arithmetic.

Implementation: [science/assembly/join/__init__.py](../../haplotype_reconstruction/science/assembly/join/__init__.py), [science/assembly/join/emissions.py](../../haplotype_reconstruction/science/assembly/join/emissions.py), [science/assembly/join/inputs.py](../../haplotype_reconstruction/science/assembly/join/inputs.py), [science/assembly/pairs.py](../../haplotype_reconstruction/science/assembly/pairs.py).

Parameters by name: `max_bins`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(S L K^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="aj02"></a>
## aj02 Join objective V_s and LL (Viterbi over unordered path pairs; uniform penalty; free start)

Score unordered path pairs with a free-start Viterbi recurrence and the uniform copying penalty, then combine per-sample values in the declared reduction. This exact objective drives all same-count comparisons. Bound and removal sums retain their specifically declared forms; an approximation is not substituted at acceptance.

Implementation: [science/assembly/join/objective.py](../../haplotype_reconstruction/science/assembly/join/objective.py), [science/assembly/pairs.py](../../haplotype_reconstruction/science/assembly/pairs.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(S m K^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="aj03"></a>
## aj03 Exact minimum-switch prior

Count the exact minimum changes of unordered founder-pair state among maximum-score paths (one state change can replace both homologues) and use it as a secondary representative prior. It never changes the primary likelihood or turns a lower-likelihood panel into an equal one. Count is computed on the same complete path definition.

Implementation: [science/assembly/pairs.py](../../haplotype_reconstruction/science/assembly/pairs.py), [science/assembly/join/objective.py](../../haplotype_reconstruction/science/assembly/join/objective.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(S m K^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="aj04"></a>
## aj04 Join painting traceback with tie band

Trace the join painting with the declared tie band under both conventions. Generate painting-dependent proposals from both traces and merge their union. A traceback convention therefore supplies candidates symmetrically; it does not select a single released ancestry claim.

Implementation: [science/assembly/join/objective.py](../../haplotype_reconstruction/science/assembly/join/objective.py), [science/assembly/pairs.py](../../haplotype_reconstruction/science/assembly/pairs.py).

Parameters by name: `EXPLORATORY_PENALTY`, `tensor_budget_mb`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(S m K^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="aj05"></a>
## aj05 Linker observation model (proxies, emission sites, pair codes, robust float32 logs)

Build linkage proxies, selected emission sites and founder-pair codes, then robust float32 log emissions. Keep the model inputs separate from the join objective and from released hard calls. Proxy reduction is a proposal device, not replacement scientific evidence.

Implementation: [science/assembly/linker/emissions.py](../../haplotype_reconstruction/science/assembly/linker/emissions.py), [science/assembly/linker/proxies.py](../../haplotype_reconstruction/science/assembly/linker/proxies.py), [science/assembly/linker/em.py](../../haplotype_reconstruction/science/assembly/linker/em.py).

Parameters by name: `max_sites`, `robustness`, `linker`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(S E C + E K^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="aj06"></a>
## aj06 Linker within-block scan (one-switch ordered-pair chain x error-tract CTMC)

Scan the ordered-pair copying model with its one-switch and error-tract process. Use the scaled route when its numerical guards hold, otherwise rescore the same model in log space. A constant finite prior may use the declared offset shortcut. Numerical fallback does not truncate candidate states.

Implementation: [science/assembly/linker/scan.py](../../haplotype_reconstruction/science/assembly/linker/scan.py).

Parameters by name: `error_fraction`, `error_tract_bp`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(S E K^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="aj07"></a>
## aj07 Linker EM per gap (chains, propagation, edge counts, regularised M-step, stop rule)

For each gap, alternate the specified forward/backward expectations with regularised transition updates until the stop rule or update cap. Preserve propagation and edge-count route guards and zero-probability transitions. Concurrent gap fits merge in fixed order; the fitted linker is a proposal model, not a pedigree likelihood.

Implementation: [science/assembly/linker/em.py](../../haplotype_reconstruction/science/assembly/linker/em.py), [science/assembly/linker/kernels.py](../../haplotype_reconstruction/science/assembly/linker/kernels.py).

Parameters by name: `max_iterations`, `min_change`, `pseudo_count`, `min_log_probability`, `uniform_mix`, `learning_rate_decay`, `minimum_learning_rate`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(S sum_b E_b K_b^2); O(S (K_l^2 K_r + K_l K_r^2)). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="aj08"></a>
## aj08 Maximum transition gap (proposal horizon)

Compute the maximum allowed transition gap from the declared geometric rule and use it to bound linker/beam proposals. The horizon is a search restriction; it does not censor a term of the exact join likelihood.

Implementation: [science/assembly/join/beam.py](../../haplotype_reconstruction/science/assembly/join/beam.py).

Parameters by name: `gap_horizon_scale`, `recombination_tolerance`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(n). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="aj09"></a>
## aj09 Endpoint beam (forward pass, backward refinement, endpoint quotas, archive order)

Extend paths forward in path-major/row-minor order, deduplicate by best score, and retain endpoint score classes in content order up to the quota, splitting the cut class in that order. Apply backward refinement and the declared archive order. Reversal affects allowed content ties; endpoint quotas mean the beam is not exhaustive path enumeration.

Implementation: [science/assembly/join/beam.py](../../haplotype_reconstruction/science/assembly/join/beam.py).

Parameters by name: `paths_per_endpoint`, `WIDE_CLASS_QUANTA`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(q K^2); O(n q K^2 G); O(q K^2 log). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="aj10"></a>
## aj10 Constant-transition Cartesian family (first-covering proposals)

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

When transitions are constant, construct the first-covering Cartesian paths with the order specified for initial pools, anchors and beam reruns. Deduplicate without replacing original-pool positions. The shortcut preserves its proposal order, which subsequent bounded search can observe.

Implementation: [science/assembly/join/beam.py](../../haplotype_reconstruction/science/assembly/join/beam.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(B G K^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="aj11"></a>
## aj11 Path linkage score (linker data criterion inside LL tie classes)

Within an ordinary likelihood tie class, compare linker scores in their own classes before content. Use panel linkage for panel choices, added-path linkage for births and negative removed-path linkage for deaths. Compute links only when needed; linkage never overrides a decisive likelihood difference.

Implementation: [science/assembly/join/beam.py](../../haplotype_reconstruction/science/assembly/join/beam.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(n G). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="aj12"></a>
## aj12 Search start (anchor panels)

Score all anchor starts, rank by the start likelihood policy at the trial scale, then linkage and panel content. A fully symmetric class invokes the count/minimum-switch representative prior and records its replayable choice. The first enumerated anchor has no unconditional preference.

Implementation: [science/assembly/join/search.py](../../haplotype_reconstruction/science/assembly/join/search.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(n S m K^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="aj13"></a>
## aj13 Panel-search main loop (phase order, certification flags)

Run tier one, then death/birth count work at a stall. Extend the pool and repeat count work only if its content changes, then run tier two and duplicate-seed count work. Escalate the saturation ladder, then terminal plateau continuation. Preserve the sweep/count/quota state and exact stop/certification branches; event counters advance only for work that runs.

Implementation: [science/assembly/join/search.py](../../haplotype_reconstruction/science/assembly/join/search.py).

Parameters by name: `paths_per_endpoint`, `POOL_QUOTA_DOUBLINGS`, `POOL_EXTENDED_DOUBLINGS`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: Data-dependent sweeps, count changes and rungs; local backstops do not bound the whole search.

<a id="aj14"></a>
## aj14 Same-count admissibility and improvement threshold

Test same-count admissibility on the slot-level edit witness before deduplicating target panels. Seed refits use their explicit exemption. Require a gain strictly above the named improvement tolerance; equality is not improvement. A canonical target alone cannot prove that its generating edit was admissible.

Implementation: [science/assembly/join/diagnostics.py](../../haplotype_reconstruction/science/assembly/join/diagnostics.py), [science/assembly/join/moves.py](../../haplotype_reconstruction/science/assembly/join/moves.py), [science/assembly/join/search.py](../../haplotype_reconstruction/science/assembly/join/search.py).

Parameters by name: `near_identity`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(K^2 B); O(K B). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="aj15"></a>
## aj15 Tier-1 proposal generators (fixed-painting fields, novelty, static gains, votes, permutations)

Generate replacement, novelty, local, one-sided, splice, cycle and switch-based edits from both paintings and fixed-field evidence. Merge duplicate edit scores by the declared maximum rule. Preserve permutation-construction order and pass caps; these bounded generators do not enumerate every possible panel edit.

Implementation: [science/assembly/join/proposals.py](../../haplotype_reconstruction/science/assembly/join/proposals.py).

Parameters by name: `EXPLORATORY_PENALTY`, `full_scores_per_kind`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(S m K^2); O(S m K K_b); O(S m K |C|). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="aj16"></a>
## aj16 Tier-1 sweep (per-kind truncation, witness choice, exact scoring, winner)

Visit proposal kinds in the fixed order, rank proxy gains and admit complete cut classes within each kind’s allowance. Materialise edits, skip no-ops, reject inadmissible witnesses, deduplicate targets and score exact candidates. Select the winner with the likelihood/linkage/content policy and preserve alternative-family evidence.

Implementation: [science/assembly/join/proposals.py](../../haplotype_reconstruction/science/assembly/join/proposals.py), [science/assembly/join/search.py](../../haplotype_reconstruction/science/assembly/join/search.py).

Parameters by name: `full_scores_per_kind`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(S m K^2 + K^2 B log K). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="aj17"></a>
## aj17 Exclusion forwards and two-region bounds (incl. pair-segment programme)

Compute exclusion forwards and two-region upper bounds, including pair-segment bounds. Apply the named exclusion margin when deciding which moves or samples need exact work. A bound must certify its skip; unresolved margins continue to exact scoring rather than become implicit zero gains.

Implementation: [science/assembly/join/certify.py](../../haplotype_reconstruction/science/assembly/join/certify.py), [science/assembly/join/moves.py](../../haplotype_reconstruction/science/assembly/join/moves.py), [science/assembly/join/pair_segments.py](../../haplotype_reconstruction/science/assembly/join/pair_segments.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(S m K² + K² B log K) proposal work plus admitted exact scores. The direct finite-grid route can be quartic in K; this unresolved cost is documented separately from the natural-domain refit cost.

<a id="aj18"></a>
## aj18 Sparse exact replacement scores

Use bulk bounds to discard clearly inactive samples, then tighter move bounds and exact sparse trial scores for unresolved samples. Other samples retain their exact lower/base value under the declared relative band. Reuse a removal bound only for its matching removed slot, including duplicate-path cases.

Implementation: [science/assembly/join/certify.py](../../haplotype_reconstruction/science/assembly/join/certify.py).

Parameters by name: `tolerance`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(S' m K^2); O(S m K). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="aj19"></a>
## aj19 Suffix-exchange (splice) table and the selected splice's forward rescore

Build the suffix-exchange score table lazily for the tier-two pass and use its total for admission and flat-neighbour observation. Rescore the selected splice by the ordinary sorted-panel forward to establish its continuing baseline. Preserve the declared acceptance point; do not add a second threshold test after selection.

Implementation: [science/assembly/join/splice.py](../../haplotype_reconstruction/science/assembly/join/splice.py), [science/assembly/pairs.py](../../haplotype_reconstruction/science/assembly/pairs.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(S m K^2 + S B K^3). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="aj20"></a>
## aj20 Tier-2 certification pass (bounded neighbourhood, best-first exact scoring, choice, flat neighbours)

Enumerate pool and structural moves in canonical kind/slot/block order, compute bounds, then score promising moves best-first. Preserve seed-refit slot restrictions and the bounded neighbourhood. Rank exact candidates, record flat neighbours and accept the declared winner; certification applies to this neighbourhood, not all conceivable edits.

Implementation: [science/assembly/join/moves.py](../../haplotype_reconstruction/science/assembly/join/moves.py), [science/assembly/join/search.py](../../haplotype_reconstruction/science/assembly/join/search.py).

Parameters by name: `WIDE_CLASS_QUANTA`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N m K (|C| + K^2)); O(N m^2 K^3). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="aj21"></a>
## aj21 Count step: deaths (pure-path cleanup, best death candidate, death test)

Compute all removal scores, prefer paths lacking their own sequence when present, and select the least-cost death using likelihood, negative linkage and content ties. Pure cleanup uses its explicit point rule; otherwise evaluate the sequence-shadow family. Record count ambiguity and update cycle state on deletion.

Implementation: [science/assembly/join/count_step.py](../../haplotype_reconstruction/science/assembly/join/count_step.py), [science/assembly/join/diagnostics.py](../../haplotype_reconstruction/science/assembly/join/diagnostics.py).

Parameters by name: `near_identity`, `cost_scale`, `shadow_combinations`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(shadow_combinations S m K²). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="aj22"></a>
## aj22 Count step: births (novelty, bound screen, best-first exact, winner, trap indicators, birth memo)

Enumerate distinct novel pool paths, bound per-sample birth gains and require both total and leave-one-out support. Retain pure-mosaic trap indicators separately. Score passing novel paths best-first, select a winner under the count/likelihood rules and lazily resolve its reported bootstrap band.

Implementation: [science/assembly/join/count_step.py](../../haplotype_reconstruction/science/assembly/join/count_step.py).

Parameters by name: `WIDE_CLASS_QUANTA`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(shadow_combinations S m K²) when the complete shadow family is evaluated.

<a id="aj23"></a>
## aj23 Duplicate-seed births (seed choice, concurrent refits, settle, acceptance)

When the ordinary birth search has no accepted birth and supplies a chimera indicator, duplicate every selected founder. Refit every seed copy with restricted slot moves, settle pure-path cleanup/death tests, then compare surviving improvements. Preserve survivor alternatives and revisit checks; acceptance can be a count birth or same-count seed swap. Seed admission is not restricted by indicator gains or coverer ranks. Endpoint selection, tie census kinds, symmetric nulls, bootstrap streams and shadow families retain their existing rules.

Implementation: [science/assembly/count_rule/parameters.py](../../haplotype_reconstruction/science/assembly/count_rule/parameters.py), [science/assembly/join/search.py](../../haplotype_reconstruction/science/assembly/join/search.py).

Parameters by name: `BIRTH_SETS`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(K) fitted seeds, O(N L K³) per count round under the quadratic fixed-K work model. Repeated count rounds can make a pass quartic; this is not a cubic bound for the complete join search. Fixed-K search iterations and shadow-family work are additional factors. Candidate scratch grows with samples times sites times K. Refinement round counts are reported in [costs](../developer/costs.md#natural-domain-count-refits).

<a id="aj24"></a>
## aj24 Count cycles, born-path freeze and count-state revisits

Propagate born-path tags through same-count edits, record deletion cycles and freeze births when the declared cycle rule applies. Detect revisited count states and retain the associated ambiguity status. Tags, cleanup history and count-state visits are decision-bearing state, not disposable diagnostics.

Implementation: [science/assembly/count_rule/criterion.py](../../haplotype_reconstruction/science/assembly/count_rule/criterion.py), [science/assembly/join/search.py](../../haplotype_reconstruction/science/assembly/join/search.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Linear in the represented panel/family memberships, plus canonical set ordering.

<a id="aj25"></a>
## aj25 Backstops (sweep/count/pool) and final death-test census

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Apply sweep, count and pool backstops at their declared points; a count backstop freezes births/seed swaps while deaths continue. Run the final death-test census only on the specified terminal branches. A backstop stop is uncertified and must not be labelled exhaustive convergence.

Implementation: [science/assembly/count_rule/parameters.py](../../haplotype_reconstruction/science/assembly/count_rule/parameters.py), [science/assembly/join/count_step.py](../../haplotype_reconstruction/science/assembly/join/count_step.py), [science/assembly/join/diagnostics.py](../../haplotype_reconstruction/science/assembly/join/diagnostics.py), [science/assembly/join/search.py](../../haplotype_reconstruction/science/assembly/join/search.py).

Parameters by name: `SWEEP_BACKSTOP_BASE`, `SWEEP_BACKSTOP_PER_ROW`, `COUNT_BACKSTOP_BASE`, `COUNT_BACKSTOP_PER_ROW`, `POOL_EXTENDED_DOUBLINGS`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: One symmetric shadow-family evaluation per final path; its combination budget limits evaluated alternatives.

<a id="aj26"></a>
## aj26 Pool extension and the saturation ladder

Extend endpoint quotas using current multiplicities, retaining original-pool order before new beam paths. Test only newly added paths on each saturation rung and preserve quota/ladder progression. Compare ordered pool content for repeated work, so cache identity and eviction cannot create extra count decisions.


Implementation: [science/assembly/join/search.py](../../haplotype_reconstruction/science/assembly/join/search.py).

Parameters by name: `paths_per_endpoint`, `POOL_QUOTA_DOUBLINGS`, `POOL_EXTENDED_DOUBLINGS`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(n q K²); O(paths_per_endpoint · 2^POOL_EXTENDED_DOUBLINGS · B K). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="aj27"></a>
## aj27 Search tie handling and the representative prior (release families)

Form likelihood classes, then linkage classes within ordinary ties, then content order. For fully symmetric panels, prefer smaller count and exact minimum switches, carrying the full replayable release family. Wide classes keep their declared value priority. Reversal changes only the remaining arbitrary convention.

Implementation: [science/assembly/join/search.py](../../haplotype_reconstruction/science/assembly/join/search.py).

Parameters by name: `WIDE_CLASS_QUANTA`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(n log n). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="aj28"></a>
## aj28 Terminal plateau continuation

Collect observed terminal plateau panels with the nominal panel and select the symmetric winner family. Continue only through not-yet-certified eligible members, preserving counters and quota; stop when the episode is covered. A nominal panel outside the selected family does not acquire a fabricated continuation choice.

Implementation: [science/assembly/join/search.py](../../haplotype_reconstruction/science/assembly/join/search.py).

Parameters by name: `WIDE_CLASS_QUANTA`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: One full score per continued panel and minimum-switch scoring per retained family member.

<a id="aj29"></a>
## aj29 Bounded original-input replays and terminal resolution

Replay bounded original-input branches in FIFO prefix order, verifying the expected decision kind/source/panel set at each forced choice. Queue complete novel alternatives; stop on mismatch or whole-family budget overflow. Exclude the declared seed choices from forcing and record incomplete resolution rather than selecting a convenient subset.

Implementation: [science/assembly/join/__init__.py](../../haplotype_reconstruction/science/assembly/join/__init__.py), [science/assembly/join/alternatives.py](../../haplotype_reconstruction/science/assembly/join/alternatives.py), [science/assembly/join/search.py](../../haplotype_reconstruction/science/assembly/join/search.py).

Parameters by name: `MAX_BRANCHES`, `join`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: A bounded number of complete join searches under the continuation-leaf budget; each retains its own search cost.

<a id="aj30"></a>
## aj30 Effective-carrier criterion and founder cost (shared count rule)

Compare larger and smaller panels using effective-carrier support, leave-one-out support and the shared founder-cost criterion. Apply strict inequalities and the declared first-maximum sample rule. The total improvement and sequence-specific improvement are separate requirements; one large carrier cannot replace independent support.

Implementation: [science/assembly/count_rule/__init__.py](../../haplotype_reconstruction/science/assembly/count_rule/__init__.py), [science/assembly/count_rule/criterion.py](../../haplotype_reconstruction/science/assembly/count_rule/criterion.py), [science/assembly/count_rule/parameters.py](../../haplotype_reconstruction/science/assembly/count_rule/parameters.py).

Parameters by name: `cost_scale`, `TOP_SAMPLES`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(S); O(S log S). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="aj31"></a>
## aj31 Bootstrap ambiguity band (seed key, draws, shadow family, lazy resolution)

Derive the bootstrap stream from its named content key and retain the exact generator/draw schedule. Share draws across a shadow family and reuse identical shadow vectors. Resolve bands lazily for reported count decisions only; cache timing cannot change which scientific test is reported.

Implementation: [science/assembly/count_rule/criterion.py](../../haplotype_reconstruction/science/assembly/count_rule/criterion.py).

Parameters by name: `bootstrap_resamples`, `bootstrap_level`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(R S). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="aj32"></a>
## aj32 Row geometry (private cells, nearest rows, near identity, novelty)

Compute integer row distances, private cells, nearest-row choices, near-identity classes and novelty with declared content ties. Cache only exact geometry keyed by its complete dependencies. Geometry constrains proposals and shadows; it does not itself release an allele.

Implementation: [science/assembly/count_rule/geometry.py](../../haplotype_reconstruction/science/assembly/count_rule/geometry.py), [science/assembly/join/count_step.py](../../haplotype_reconstruction/science/assembly/join/count_step.py).

Parameters by name: `near_identity`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(L_b); O(B); O(K^2 B). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="aj33"></a>
## aj33 Shared sequence-shadow family decision

Construct the complete sequence-shadow choice product when it fits the budget and the total test passes; otherwise evaluate only the declared incumbent shadow and mark completeness honestly. Reduce decisions in the specified reject/margin order. Both join and refinement use this rule. The conventional incumbent follows RowGeometry.reverse (lowest nearest row normally, highest under reversal). Only a passing total with budget overflow is count_unverified. Opposed evaluated verdicts withhold without forcing bootstrap ambiguity. The published shadow prior is `smaller_k+nearest_shadow`. Scores and diagnostic arrays follow canonical Cartesian order; exact ties in (accept_larger, margin, sequence LOSO margin) select the first canonical member normally and the last under reversal. Strictly smaller decision keys always take precedence. Verification depends on enumeration admission alone.

Implementation: [science/assembly/count_rule/shadow_family.py](../../haplotype_reconstruction/science/assembly/count_rule/shadow_family.py), [science/assembly/count_rule/geometry.py](../../haplotype_reconstruction/science/assembly/count_rule/geometry.py), [science/assembly/join/count_step.py](../../haplotype_reconstruction/science/assembly/join/count_step.py), [science/assembly/refinement/count.py](../../haplotype_reconstruction/science/assembly/refinement/count.py).

Parameters by name: `shadow_combinations`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(shadow_combinations S m K²). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="aj34"></a>
## aj34 Count-rule bound tests used by founder refinement

Apply the shared strict total and sequence count bounds to refinement candidates before expensive refits. A bound can certify acceptance/rejection only under its stated operands; unresolved intervals continue to the exact criterion. Use the same effective-carrier definitions as the join count rule.

Implementation: [science/assembly/count_rule/criterion.py](../../haplotype_reconstruction/science/assembly/count_rule/criterion.py).

Parameters by name: `tolerance`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(shadow_combinations S m K²) for the bounded family scoring route.

<a id="aj36"></a>
## aj36 Join record key and schema (joins/<digest>)

Bind the complete join input and parameter closure into a content digest, publish one checked result and reuse it only on exact derivation agreement. Keep level-plan spans and diagnostic census labels separate from scientific identity. Physical record location and cache object identity do not choose the search path.

Implementation: [stages/assembly/records.py](../../haplotype_reconstruction/stages/assembly/records.py), [science/assembly/hierarchy/batch.py](../../haplotype_reconstruction/science/assembly/hierarchy/batch.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(sites x K). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<!-- named mechanism anchors -->

## Named mechanisms

<a id="alleles"></a>

**Alleles:** see [U08](#u08).

<a id="completion"></a>

**Completion:** see [U03](#u03).

<a id="evidence"></a>

**Evidence:** see [U01](#u01).

<a id="founder-allele-posteriors"></a>

**Founder allele posteriors:** see [U08](#u08).

<a id="hierarchy"></a>

**Hierarchy:** see [U11](#u11).

<a id="linker"></a>

**Linker:** see [AJ05](#aj05).

<a id="release"></a>

**Release:** see [U16](#u16).


Evidence-v2 inputs remain compact in the chain cache. `AssemblyEvidenceSource` expands a coded source directly into its final neutral float64 and scoring float32 arrays one sample at a time; it never creates another whole-contig raw float64 buffer. The coded mask is reconstructed one sample at a time into the active assembly mask; no full mask is held in its cached source. Mask values, the literal missing-cell value and the float32 cast point are unchanged. Fully dense, already neutral inputs retain the read-only borrowing route. The compact raw source can remain cached for subsequent selection without retaining assembly's dense views.

### Execution order and CPU lending

Independent joined batches and refinement components carry dimension-only work
estimates (samples, sites and founder-state/refit work). The engine starts larger
ready units first; dependencies still separate hierarchy levels and result gathers
still visit genomic component order. Synchronous context joins use the same batch
cost ordering and keyed composition.

Stage expansion preserves the enclosing graph task's declared CPU maximum,
rather than freezing the share passed when its parent first started. A join's
CPU ceiling covers its independent count refits. Each sample kernel is
separately capped at the useful sample width, so that width does not restrict the
entire refit family. The team is sized so that each thread would take a grain of at most
`SAMPLES_PER_THREAD` (8) samples; in refinement the grain is only as many samples as
carry `THREAD_WORK_FLOOR` site-diplotype cells (a scheduling estimate: component sites x
K² per sample), so a 40-sample component of 300,000 sites can use 40 threads rather than
5. A limited CPU share can still give a thread more samples than the grain. Team size changes scheduling only. Engine child budgets split the current
task allowance and lend finished children's CPUs at later kernel boundaries.
Refit results return to their original indices before the existing count, tie and
acceptance rules run. Neither completion order nor team size chooses a founder
count, search history or released panel.

### Kernel registration and warm-up

Preloading the assembly stage registers its numerical kernel closure, including
join, linker, refinement-proposal and polish modules that ordinary execution may
import only on first use. Registration declares argument types and compiles
nothing. It lets the engine's initial warm-up plan include these kernels before
an assembly worker reaches them. Supported storage signatures include immutable,
noncontiguous shared GL and immutable local row codes; storage layout does not
select a different objective or search.

Parallel warm-up and scientific workers must use the same explicit run cache,
code identity and compilation target. A separate consumer process verifies disk
cache use; same-process reuse alone cannot establish it. Completing the relevant
compile tasks before first scientific use is an application scheduling policy.
A run that allows scientific tasks to race those tasks can still pay compilation
cost in a worker, even with complete declarations. Init and scientific wall time
are reported separately so a moved cost remains visible.


Cooperative reconstruction claims the existing completion product on the direct
route. Expanded completion claims each deterministic 16-block range under
`<assembly-prefix>/completion/range<first-block>` and stores its complete typed
reply. The original final gather takes the completion-product claim before
linking/release and publication. Plans, evidence views and resident continuations
remain local to each invocation; no claim spans an asynchronous expansion.
