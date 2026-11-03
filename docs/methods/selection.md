# Local panel selection

Selection refits local copying panels with a regularised dictionary objective, then repeats the declared context rounds using assembly-derived proposals. Context can rescue a release only through its explicit witness. N is samples, L block sites, K panel rows, F context rows and S copying states. The bounded search and fixed round schedule do not establish global optimality.

The sections below identify semantic units, not separately configurable pipeline steps. [Outputs](../user/outputs.md), [record contracts](../reference/records.md) and [costs](../developer/costs.md) describe their public boundaries.

<a id="ls01"></a>
## ls01 Copying model and block log likelihood

Evaluate the diploid copying model for a binary panel on the block’s discovery sites. Preserve ordered recurrence products and the direct-division fallback for subnormal normalisers. Missing evidence is neutral. The copying likelihood is distinct from the regularised panel-selection objective.

Implementation: [science/selection/__init__.py](../../haplotype_reconstruction/science/selection/__init__.py), [science/selection/path_model.py](../../haplotype_reconstruction/science/selection/path_model.py).

Parameters by name: `unknown_mass`, `redraw_rate_per_morgan`, `model`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L S^2); O(N L K^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ls02"></a>
## ls02 E-step statistics levels (COUNTS, FLIPS, EXACT) and order of summation

Compute COUNTS, FLIPS or EXACT statistics with identical arithmetic for common quantities. Combine lane contributions in ascending sample order independently of worker count. Skip zero-posterior ratio terms. Reuse a cached level only for the same panel/frequencies and when it contains all requested statistics.

Implementation: [science/selection/path_model.py](../../haplotype_reconstruction/science/selection/path_model.py).

Parameters by name: `LANE_WIDTH`, `LANE_ALIGN`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L K^2); O(K S). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ls03"></a>
## ls03 Regularised objective J (dictionary code, K prior, frequency shrinkage)

Add the dictionary code, founder-count prior and frequency shrinkage to the likelihood. At fixed count without context, the dictionary term is constant; different counts or claimable context rows must be compared with the full selection objective. A likelihood gain alone is insufficient.

Implementation: [science/selection/objective.py](../../haplotype_reconstruction/science/selection/objective.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(K). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ls04"></a>
## ls04 Context dictionary code: shortest description (edit-ball code, matching of every size)

Compute edit-ball costs for context founders against panel rows, solve minimum-cost matchings for every matching size, then choose the largest size within the description tie tolerance. Canonicalise content for released matching pairs. Matching order and the selection of size are separate rules.

Implementation: [science/selection/objective.py](../../haplotype_reconstruction/science/selection/objective.py).

Parameters by name: `model`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(F K); O(F L); O(min(F, K) (F + K)^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ls05"></a>
## ls05 Fixed-K GEM fit (frequency step, GEM flip, exact one-bit escape, ties, stops, canonical output)

Canonicalise the start and alternate a frequency step with allele updates. Accept frequency trials within the allowed objective tolerance; choose the declared statistics level for the current update. Apply positive GEM flips, then exact single-bit escape when required. Preserve the update backstop, stationary test and canonical endpoint.

Implementation: [science/selection/fit.py](../../haplotype_reconstruction/science/selection/fit.py).

Parameters by name: `tolerance`, `FLIP_TIE`, `EXACT_FIRST_UPDATE_MAX_K`, `search`, `update_backstop_base`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L K^2); O(N L K^2 (K + L)). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ls06"></a>
## ls06 Trajectory-best selection with frequency-only refit (context rounds)

Track the earliest strictly best selection state along the fit trajectory. Adopt it only if it clears the endpoint’s decision tolerance. Refit frequencies with the panel fixed, stopping at the first insufficient gain or update cap. The returned model objective and context selection objective remain separate.

Implementation: [science/selection/fit.py](../../haplotype_reconstruction/science/selection/fit.py).

Parameters by name: `decision_tolerance`, `tolerance`, `search`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(K^3). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ls07"></a>
## ls07 Context snaps (selection-aware one-bit steps after the GEM)

Build exact one-bit gains, exclude flips that would collide rows, and examine matched context pairs in order. Stop if the largest bound does not clear the decision tolerance. Choose content within the top bound band, score the actual trial, and accept only the required selection improvement.

Implementation: [science/selection/fit.py](../../haplotype_reconstruction/science/selection/fit.py).

Parameters by name: `decision_tolerance`, `tolerance`, `search`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L K^2); O(K^3); O(K^2 L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ls08"></a>
## ls08 Proposal generation: drop, merge consensus, add; start de-duplication

Register the incumbent, then generate drops, lexicographic merges and bank-order additions. Skip existing bank rows and deduplicate near-equal frequency starts by first occurrence under the proposal tolerance. A merge may collapse more than one row; a singleton panel has no death/merge proposal.

Implementation: [science/selection/diagnostics.py](../../haplotype_reconstruction/science/selection/diagnostics.py), [science/selection/search.py](../../haplotype_reconstruction/science/selection/search.py).

Parameters by name: `CONSENSUS_TIE`, `PROPOSAL_FREQUENCY_TIE`, `CONSENSUS_CENSUS_WINDOW`, `search`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(K^2); O(L); O(K L log K). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ls09"></a>
## ls09 Proposal screening: exact scores, merge EM lower bounds, double-removal representatives

Memoise bounds by full panel/frequency content. Select the declared nearest-row and double-removal merge representatives in bound tie classes, score those exactly, and retain bounds for the others. Exact and bounded values enter the same admission ranking; equal bounds do not imply equal exact scores.

Implementation: [science/selection/search.py](../../haplotype_reconstruction/science/selection/search.py).

Parameters by name: `decision_tolerance`, `DECISION_MOTION`, `search`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L K^2); O(N L K^3); O(K^2 (L + K)). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ls10"></a>
## ls10 Refit admission cut (whole tie classes, whole-class limits per kind)

Form chained decision classes for each proposal kind and admit complete classes up to the target with the hard cap. A leading oversized class can leave a kind with no refits. Refit every admitted start with its own frequencies; no trial means the climb stops.

Implementation: [science/selection/search.py](../../haplotype_reconstruction/science/selection/search.py).

Parameters by name: `refits_per_kind`, `refits_per_kind_cap`, `search`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L K^2 (K + L)). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ls11"></a>
## ls11 Round winner, same-panel representative, climb stop and round backstop

Check the count-dependent round backstop, rank trial objectives in tie classes, and apply count/content/trial-order secondary rules. Choose a stable same-panel proposal representative. Stop unless the winner improves the incumbent by the declared band; a class representative need not be its numerically highest member.

Implementation: [science/selection/search.py](../../haplotype_reconstruction/science/selection/search.py).

Parameters by name: `round_backstop_base`, `round_backstop_per_founder`, `decision_tolerance`, `KIND_ORDER`, `search`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L K^3 (K + L)); O(K); O(K + L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ls12"></a>
## ls12 Search start: start fit versus incumbent continuation

Fit each start and compare it with the incumbent under the current context. Keep the incumbent fit when an equal panel does not improve sufficiently. If the incumbent is decisively better, try a continuation from its own frequencies, retaining it only for the required improvement.

Implementation: [science/selection/search.py](../../haplotype_reconstruction/science/selection/search.py).

Parameters by name: `decision_tolerance`, `search`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: One fitted continuation, and at most one additional incumbent continuation under the specified comparison.

<a id="ls13"></a>
## ls13 Whole-search reuse and block-scoped memoisation

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Reuse a whole search when the retained key for the start, candidate bank, genotype likelihoods, observation mask, rates, founders, unknown mass and search settings agrees. This key excludes the incumbent and memo history: a hit returns the incumbent before continuation. A dependency-complete reuse predicate remains an open design item in the [accretion register](../developer/accretions.md). Context-dependent fits bind context rows and called masks; omitted posterior arrays are reconstructed by the same exact E-step, preserving packed-memo hit/completion precedence. Completed blocks require their persisted initial/L1 memos. That recorded memo route is part of the retained execution contract; replacing it with cold fits or changing memo capacity is not guaranteed to preserve selected endpoints or release.

Implementation: [science/selection/search.py](../../haplotype_reconstruction/science/selection/search.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Memory proportional to retained fit/memo contents; reuse saves otherwise identical fit work.

<a id="ls14"></a>
## ls14 Multi-start selection and outcome comparison (tied alternatives)

Fit and deduplicate multiple starts, select their best decision class and preferred count, then search distinct retained starts in content order. Compare resulting outcomes by the same objective/count policy. Carry distinct tied panels as alternatives; a wide class follows its declared value-first rule.

Implementation: [science/selection/rounds.py](../../haplotype_reconstruction/science/selection/rounds.py), [science/selection/search.py](../../haplotype_reconstruction/science/selection/search.py).

Parameters by name: `decision_tolerance`, `search`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: One fit per start and one search per distinct retained best-class endpoint.

<a id="ls15"></a>
## ls15 Keep-incumbent hysteresis between rounds

Rescore the incumbent in the new context and retain it unless the attempted outcome clears the improvement tolerance. Retention also discards the attempted tied alternatives. Pass this retained outcome to the next round; release reuse has its own stricter conditions.

Implementation: [science/selection/rounds.py](../../haplotype_reconstruction/science/selection/rounds.py).

Parameters by name: `decision_tolerance`, `search`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: One context-description evaluation and the existing objective comparison.

<a id="ls16"></a>
## ls16 Initial-round starts and bank

Form canonical distinct starts from discovery’s latent panel and tied alternatives, including their mass vectors. Sort complete start identities and use the union of their rows as the bank. The initial round has no incumbent and introduces no population-based parent assumption.

Implementation: [science/selection/rounds.py](../../haplotype_reconstruction/science/selection/rounds.py), [science/selection/bank.py](../../haplotype_reconstruction/science/selection/bank.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Linear in discovery starts and their row/mass content, plus canonical sorting.

<a id="ls17"></a>
## ls17 Context-round proposal completion (tied donors, variant cap)

Drop fully uncalled proposal rows, retain fully called ones, and complete partial rows from every minimum-distance latent donor. Deduplicate donor options and enumerate whole combinations within the variant cap. Overflow discards the ambiguous option family instead of choosing an arbitrary completion.

Implementation: [science/selection/bank.py](../../haplotype_reconstruction/science/selection/bank.py).

Parameters by name: `completion_variant_cap`, `bank`, `search`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(K_p K L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ls18"></a>
## ls18 Bank sources: discovery fixed-K refits, cavity-best start sets, released starts

Fit discovery-derived sources, reuse complete endpoint keys, and select cavity-best families with nested score/NLL classes and the preferred count. Release each start, filling only its uncalled candidate positions from its own latent panel. Deduplicate complete row/mass starts before multi-start search.

Implementation: [science/selection/bank.py](../../haplotype_reconstruction/science/selection/bank.py).

Parameters by name: `max_iterations`, `unknown_copy_penalty`, `cavity`, `release`, `improvement`, `WIDE_CLASS_QUANTA`, `model`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: Fixed-count discovery fits and cavity scores for every admitted source; cost is bounded by source/start budgets.

<a id="ls19"></a>
## ls19 Bank BIC neighbourhood search and the add-row bank

Fit the completed bank as an initial source, select its best admitted entries, then visit deletion, addition and replacement neighbours in the declared order. Reselect after each round. Stop at insufficient improvement or the count-dependent round/score budget, retaining explicit budget diagnostics.

Implementation: [science/selection/bank.py](../../haplotype_reconstruction/science/selection/bank.py).

Parameters by name: `complexity_scale`, `improvement`, `bank`, `model`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(K x |bank|); O(K). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ls20"></a>
## ls20 Bank budgets: start backstop, family admission at the score cap, round and score backstops

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Apply the start backstop in declared enumeration order. Group novel fitted endpoint keys into families and admit complete NLL classes within remaining score capacity. Preserve first-key retention, including the explicitly reversed keep convention. These scientific backstops cannot depend on memory or cache eviction.

Implementation: [science/selection/bank.py](../../haplotype_reconstruction/science/selection/bank.py).

Parameters by name: `start_backstop_base`, `start_backstop_per_founder`, `score_backstop_base`, `score_backstop_per_founder`, `round_backstop_base`, `bank`, `search`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(K). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ls21"></a>
## ls21 Release rule (one-bit probability, favourable carriers, support) and release reuse

Compute exact per-founder/site evidence from the copying posterior. Count favourable observed carriers with the strict gain rule, then apply inclusive probability/support thresholds. Invoke context rescue only afterwards. Reuse an incumbent release only when panel/frequencies and the absence of current/prior context rescue satisfy the complete predicate.

Implementation: [science/selection/release.py](../../haplotype_reconstruction/science/selection/release.py).

Parameters by name: `min_call_probability`, `min_supporters`, `favourable_margin`, `RELEASE_SAMPLE_BATCH`, `FAVOURABLE_CENSUS_WINDOW`, `release`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L K^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ls22"></a>
## ls22 Context-witness rescue

Find cells meeting the probability gate but lacking independent confidence. Rescue only cells witnessed by a matched context founder with the same called allele. Record rescued-cell count so the next round cannot incorrectly reuse an evidence-only release.

Implementation: [science/selection/release.py](../../haplotype_reconstruction/science/selection/release.py).

Parameters by name: `posterior_threshold`, `min_support`, `context_refit`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: One context matching over panel/context rows, plus witnessed-site comparisons.

<a id="ls23"></a>
## ls23 Unknown-copy summary

At each observed site aggregate posterior mass into zero, one and two unknown-copy categories. Choose the smallest count whose mass is at least `(1 - unknown_count_band)` times the largest mass: a banded MAP choice among those three categories. Take each sample’s maximum selected count over observed sites and over tied released panels. A sample without depth receives two unknown slots.

Implementation: [science/selection/path_model.py](../../haplotype_reconstruction/science/selection/path_model.py), [science/selection/rounds.py](../../haplotype_reconstruction/science/selection/rounds.py), [science/selection/run.py](../../haplotype_reconstruction/science/selection/run.py), [stages/selection/tasks.py](../../haplotype_reconstruction/stages/selection/tasks.py), [contracts/panels.py](../../haplotype_reconstruction/contracts/panels.py).

Parameters by name: `unknown_count_band`, `unknown_mass`, `release`, `model`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: Additional reductions within the release E-step; no independent model fit.

<a id="ls24"></a>
## ls24 Tied-alternative consensus release and the next round's representative

Intersect panel row content across tied outcomes, withhold cells where released alleles disagree, average ALT probabilities and take minimum support. Maximise unknown-slot uncertainty over alternatives, order rows canonically and recompute unknown mass. The next round sees this conservative release, not every latent alternative.

Implementation: [science/selection/rounds.py](../../haplotype_reconstruction/science/selection/rounds.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: One release per retained alternative and O(K L) consensus work per alternative.

<a id="ls25"></a>
## ls25 Three-round unrolled schedule and context assembly

Execute the initial round and the two declared context rounds, assembling context at the configured fixed levels. Each informative block is selected once per round. An uncovered block gets its current calls as a proposal without context witnesses, claims, trajectory rescue or snaps.

Implementation: [science/selection/run.py](../../haplotype_reconstruction/science/selection/run.py), [stages/selection/tasks.py](../../haplotype_reconstruction/stages/selection/tasks.py).

Parameters by name: `context_levels`, `search`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: Three block-selection rounds and two context assembly/refit passes; rounds depend sequentially, blocks within a round do not.

<a id="ls26"></a>
## ls26 Context coverage and component founder rows (de-duplication, template consensus)

Restrict evidence to kept discovery sites, match complete component coverage, and collapse duplicate context rows into consensus templates. Use the map object’s interval distances for the context chain (L11). Skip incompatible coverage explicitly; disagreeing called duplicate alleles yield unknown template cells.

Implementation: [science/selection/context_refit.py](../../haplotype_reconstruction/science/selection/context_refit.py).

Parameters by name: `redraw_rate_per_morgan`, `unknown_mass`, `model`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(K_c L_c log K_c). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ls27"></a>
## ls27 Context-refit bin HMM: emission grid and cavity weights

Construct pair emissions in ascending site order, centre/exponentiate each bin and normalise forward/backward recurrences. Compute cavity weights on this same grid. Background-count summaries are diagnostic; no memory route may change the recurrence or candidate set.

Implementation: [science/selection/context_refit.py](../../haplotype_reconstruction/science/selection/context_refit.py).

Parameters by name: `emission_mix`, `unknown_mass`, `redraw_rate_per_morgan`, `context_refit`, `model`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L_c S^2); O(N L_c); O(N B S^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ls28"></a>
## ls28 Context-refit site release, fallback and proposal construction (multiplicity, witness)

Compute site-release marginals; reject multimodal or guard-withheld cells before applying named support and probability gates. Prefer released refit calls, then retained calls, then eligible template fallback. Deduplicate local rows, combine witness masks and carry multiplicity into context proposals.

Implementation: [science/selection/context_refit.py](../../haplotype_reconstruction/science/selection/context_refit.py).

Parameters by name: `posterior_threshold`, `min_support`, `MAX_EXACT_BITS`, `SITE_CERTAIN_NATS`, `SITE_BALL_CONFIGURATIONS`, `SITE_BALL_MINIMUM_RADIUS`, `SITE_SHELL_GUARD`, `SITE_IMPROVEMENT`, `SITE_HALF_BAND`, `context_refit`, `release`, `tolerance`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N 2^K_c); O(N K^2); O(N (m² + 2^MAX_EXACT_BITS)). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ls29"></a>
## ls29 Recorded proposal summaries (per-bin background counts, combined posterior and support)

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Record per-bin background counts, combined posterior and support alongside the proposal. Downstream selection consumes the declared calls, founders and witness fields; these additional summaries do not alter admission or release.

Implementation: [science/selection/context_refit.py](../../haplotype_reconstruction/science/selection/context_refit.py).

Parameters by name: `BACKGROUND_COUNT_BAND`, `HALF_CENSUS_WINDOW`, `context_refit`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: Linear in generated context-bin summaries.

<a id="ls30"></a>
## ls30 Round execution, scheduling, records, identities and resume

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Schedule independent block work from checked inputs, storing the round’s proposals and outcomes through typed adapters. Reassemble by block index and resume only compatible publications. Resource estimates and task completion order do not select proposals or change scientific budgets.

Implementation: [stages/selection/records.py](../../haplotype_reconstruction/stages/selection/records.py), [science/selection/rounds.py](../../haplotype_reconstruction/science/selection/rounds.py).

Parameters by name: `TASK_WORKSPACE_BASE_BYTES`, `SEARCH_GROWTH`, `context_refit`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: Storage proportional to fits and retained round memos; block results assemble in index order.

<a id="ls31"></a>
## ls31 Diagnostics: tie counters, valley census, summaries and reports

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Return typed tie counters, search summaries and optional census information to the stage layer. Render reports from those records. Keep diagnostics separate from the incumbent, input key and context-rescue state that really do affect later decisions.

Implementation: [science/selection/__init__.py](../../haplotype_reconstruction/science/selection/__init__.py), [science/selection/context_refit.py](../../haplotype_reconstruction/science/selection/context_refit.py), [science/selection/diagnostics.py](../../haplotype_reconstruction/science/selection/diagnostics.py), [stages/selection/tasks.py](../../haplotype_reconstruction/stages/selection/tasks.py), [science/selection/rounds.py](../../haplotype_reconstruction/science/selection/rounds.py).

Parameters by name: `DECISION_MOTION`, `search`, `release`, `context_refit`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: Counters are small; optional gain census can require O(N K L) values.
