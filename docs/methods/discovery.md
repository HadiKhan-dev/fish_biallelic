# Local founder discovery

Discovery searches binary local founder panels under a known/unknown copying model. Fixed-K fitting and cavity scoring are separate objectives. The bounded proposal search carries ties and omissions, so a selected panel is not a proof of a global optimum. N is samples, L block sites, K founder rows, B bins and S copying states. All candidate budgets are scientific parameters, independent of execution resources.

The sections below identify semantic units, not separately configurable pipeline steps. [Outputs](../user/outputs.md), [record contracts](../reference/records.md) and [costs](../developer/costs.md) describe their public boundaries.

<a id="di01"></a>
## di01 Block inputs and preparation (informative test, neutral missing cells, canonical sample order, active samples, K ceiling)

Restrict to the block’s informative evidence, neutralise unobserved cells, and use the canonical sample axis. Determine active samples and the founder-count ceiling before proposing panels. Blocks without informative evidence produce no inferred panel; sample labels supply no biological information.

Implementation: [science/discovery/__init__.py](../../haplotype_reconstruction/science/discovery/__init__.py), [stages/discovery/tasks.py](../../haplotype_reconstruction/stages/discovery/tasks.py), [science/discovery/run.py](../../haplotype_reconstruction/science/discovery/run.py), [science/discovery/search.py](../../haplotype_reconstruction/science/discovery/search.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above. The sample-oriented capped-cost tables are retained for frontier and fractional sweeps. The equivalent site-oriented tables are allocated only if a site-parallel singleton sweep uses them; a one-thread block avoids an additional 40 N L bytes. This changes storage lifetime only, preserving every float64 cost and reduction.

<a id="di02"></a>
## di02 Block tie quanta and class/budget primitives (BlockQuanta, tie_classes, whole_class_cut, improves)

Compute the block scale once and keep its tie quanta fixed throughout search. NLL and sample quanta scale with evidence magnitude; allele-step, improvement and probability policies have their own scales. Apply the same NLL band to cavity logits. Admit complete tie classes at budgets, recording omitted classes.

Implementation: [science/discovery/search.py](../../haplotype_reconstruction/science/discovery/search.py), [core/ties.py](../../haplotype_reconstruction/core/ties.py), [science/discovery/panel.py](../../haplotype_reconstruction/science/discovery/panel.py).

Parameters by name: `WIDE_CLASS_QUANTA`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L); O(n log n). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="di03"></a>
## di03 Fixed-K observation model and evidence-only tables

Enumerate unordered known-row pairs, a known row with one unknown copy, then two unknown copies. Known alleles use genotype likelihoods; an unknown copy chooses its site allele subject to the named penalty. Sum centred terms into consecutive bins. Missing cells contribute zero, and the H-step costs are capped by the all-unknown cost.

Implementation: [science/discovery/api.py](../../haplotype_reconstruction/science/discovery/api.py), [science/discovery/fixed_k/workspace.py](../../haplotype_reconstruction/science/discovery/fixed_k/workspace.py).

Parameters by name: `unknown_copy_penalty`, `snps_per_bin`, `release`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="di04"></a>
## di04 A step (tie band, category preference, meet summary, membership, switching-Viterbi NLL)

Find the minimum whole-block state cost and retain every state within the inclusive sample band. Apply the fixed category preference before forming the tied-state meet, membership and switching score. Category preference is a model rule and is not reversed with arbitrary content conventions.

Implementation: [science/discovery/fixed_k/assign.py](../../haplotype_reconstruction/science/discovery/fixed_k/assign.py), [science/discovery/fixed_k/fit.py](../../haplotype_reconstruction/science/discovery/fixed_k/fit.py).

Parameters by name: `switch_penalty`, `snps_per_bin`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N K² B) state comparisons on binned evidence; B is determined by the named binning parameter.

<a id="di05"></a>
## di05 H step (usage order, buckets, fractional evidence, h_step decision, certified sums)

Choose singleton or fractional assignment evidence according to the complete tied-state set. Visit founders in usage order and samples in declared bucket order; update sites independently. Retain the current allele inside the allele-step band. When an ordinary sum cannot certify a decision, recompute with the exactly rounded expansion sum before testing.

Implementation: [science/discovery/fixed_k/update.py](../../haplotype_reconstruction/science/discovery/fixed_k/update.py), [science/discovery/fixed_k/fit.py](../../haplotype_reconstruction/science/discovery/fixed_k/fit.py).

Parameters by name: `H_SUM_ERROR_FACTOR`, `H_SUM_ERROR_CUTOFF`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(L x total memberships + K L); O(N L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="di06"></a>
## di06 Fit trajectory and complete usage-schedule families

For each ordered start, look up or compute the next transition, enforce the trajectory cap, record usage ties, and stop at a fixed point or a repeated state. A cycle returns its best recorded state. Complete usage-schedule families are carried; changing row order can change the trajectory and is not a harmless cache key change.

Implementation: [science/discovery/fixed_k/fit.py](../../haplotype_reconstruction/science/discovery/fixed_k/fit.py).

Parameters by name: `max_iterations`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(iterations N K² B) per trajectory, with named iteration and usage-schedule budgets.

<a id="di07"></a>
## di07 Fit memo, frontier batching and fixed-point certificates

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Memoise only quantities determined by their complete content key. Recompute an evicted transition exactly; batching and cache capacity cannot alter a value. A certificate skips a fit only when the same canonical panel is certified under both usage schedules. Certificate availability changes work, not the fitted endpoint.

Implementation: [science/discovery/fixed_k/workspace.py](../../haplotype_reconstruction/science/discovery/fixed_k/workspace.py), [science/discovery/fixed_k/fit.py](../../haplotype_reconstruction/science/discovery/fixed_k/fit.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Entry-bounded cache storage includes O(N) per-sample arrays per transition plus membership; eviction changes recomputation only.

<a id="di08"></a>
## di08 fit_panels (unique-row starts, synchronized solutions, collapse recursion, families)

Canonicalise distinct-row starts, obtain synchronised and refitted endpoints, and recursively handle row collapse where the entry contract permits it. Deduplicate by panel key, retaining the first result unless another clears the NLL improvement band. The fixed-point flag records an existing certificate, not a new scientific preference.

Implementation: [science/discovery/fixed_k/fit.py](../../haplotype_reconstruction/science/discovery/fixed_k/fit.py).

Parameters by name: `max_iterations`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: One fixed-count trajectory family per distinct admitted start, plus permitted collapse refits.

<a id="di09"></a>
## di09 Panel canonical form, keys, deduplication and NLL ordering

Canonicalise panels by content, deduplicate exact keys, then order NLL tie classes with founder count and content as secondary keys. Incumbent retention applies to duplicate keys. Reversal affects only the declared arbitrary content convention, not the NLL objective or founder-count preference.

Implementation: [science/discovery/panel.py](../../haplotype_reconstruction/science/discovery/panel.py).

Parameters by name: `SET_COUNT_ROUNDING_FLOOR`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(K^2 L); O(n); O(n log n). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="di10"></a>
## di10 Cavity score (state prior, founder MRF mean field, endpoint classes, predictive, K prior, set code)

Combine the state prior, founder mean-field model, endpoint classes, predictive score, count prior and set code. Evaluate the declared starts and retain eligible endpoints under the NLL policy. A failed mean-field solve remains represented and is counted; it prevents a certification claim rather than silently removing a candidate.

Implementation: [science/discovery/api.py](../../haplotype_reconstruction/science/discovery/api.py), [science/discovery/cavity.py](../../haplotype_reconstruction/science/discovery/cavity.py), [science/discovery/cavity_kernels.py](../../haplotype_reconstruction/science/discovery/cavity_kernels.py).

Parameters by name: `state_concentration`, `unknown_prior_mass`, `double_unknown_fraction`, `mean_field_max_iterations`, `mean_field_tolerance`, `likelihood_floor`, `PREDICTION_TILE_BYTES`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L K²) per bounded mean-field sweep/start budget, plus endpoint/cavity scoring.

<a id="di11"></a>
## di11 Level-wise HDBSCAN*

Build the level-wise density hierarchy with equal edge weights merged simultaneously. Accumulate stability with exactly rounded sums and apply the excess-of-mass selection. Internal cluster numbering has no biological meaning; too few points produce noise rather than a fabricated cluster.

Implementation: [science/discovery/residual_births.py](../../haplotype_reconstruction/science/discovery/residual_births.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(n^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="di12"></a>
## di12 Bounded landmark geometry (bounded landmarks)

Select landmark classes under the declared whole-class limit, build bounded proposal geometry, and lift the resulting groups. Omitted landmark classes are recorded; other starts remain available. Fitting, scoring and release still use all observations. The bounded geometry can miss a proposal that dense geometry would find.

Implementation: [science/discovery/landmarks.py](../../haplotype_reconstruction/science/discovery/landmarks.py).

Parameters by name: `landmark_budget`, `seed_landmark_budget`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(R L log R + A R log R + R A ceil(L/64)); O(R A). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="di13"></a>
## di13 K = 1 initial starts

Rank initial clusters by size, admit whole size classes, then use the decisive-sample fallback when its complete class fits. Fit the admitted starts and retain whole NLL classes. If every start class overflows, report an unresolved block with no incumbent; do not substitute an unscored population row.

Implementation: [science/discovery/moves.py](../../haplotype_reconstruction/science/discovery/moves.py).

Parameters by name: `data_seed_count`, `seed_min_cluster_size`, `data_start_beam_width`, `seed_landmark_budget`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N_a^2 L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="di14"></a>
## di14 Death moves

Generate deletion starts from the current panel and put the complete death bucket before gauge proposals. Fit admitted starts with the fixed-K machinery and carry tied endpoint families. Death ordering and whole-bucket admission are part of the search.

Implementation: [science/discovery/moves.py](../../haplotype_reconstruction/science/discovery/moves.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Up to K deletion starts per expanded parent, each with the fixed-count fit cost.

<a id="di15"></a>
## di15 Gauge rewires and death/gauge family admission

Generate gauge rewires, deduplicate each death/gauge bucket by content, and admit deaths first under the complete-class proposal budget. Use the exact cut only within its named size ceiling; larger components use the heuristic cut and record that limitation. Fit admitted starts before considering births.

Implementation: [science/discovery/moves.py](../../haplotype_reconstruction/science/discovery/moves.py).

Parameters by name: `exact_cut_max_k`, `max_cut_ties`, `max_proposals_per_expansion`, `MAX_CUT_CACHE_ENTRIES`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(2^(m-1) m); O(K); O(K^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="di16"></a>
## di16 Data births

Construct observed-cell dosage and population seed rows, rank complete seed classes, and apply the founder-count-dependent target, budget and hard cap. Fit admitted new-row starts. Keep complete endpoints with the requested larger count; collapsed endpoints on this route do not recurse into smaller counts.

Implementation: [science/discovery/moves.py](../../haplotype_reconstruction/science/discovery/moves.py).

Parameters by name: `data_birth_attempts_factor`, `data_birth_children_factor`, `data_birth_tie_budget_factor`, `unknown_copy_penalty`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(K); O(K · max_iterations · N K² L / 10); O(N L K³). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="di17"></a>
## di17 Soft-residual records and dominant-partner routes

Create soft-residual records in sample/row order and retain complete dominant-partner route classes using the probability band. Route membership is set-valued; no convention selects one member of an equal band. Missing cells do not supply residual support.

Implementation: [science/discovery/residual_births.py](../../haplotype_reconstruction/science/discovery/residual_births.py).

Parameters by name: `min_soft_responsibility`, `residual_call_probability`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N P L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="di18"></a>
## di18 Residual grouping (density groups, edge clique cover, per-component size-class budget, lifting)

Group residuals with density geometry and an edge-clique cover, then apply the alternative-size-class budget separately to each connected component. Lift retained groups to sample evidence. The cover represents bounded pairs, not every larger clique; crowding one component cannot consume another’s allowance.

Implementation: [science/discovery/residual_births.py](../../haplotype_reconstruction/science/discovery/residual_births.py), [science/discovery/_cliques.py](../../haplotype_reconstruction/science/discovery/_cliques.py).

Parameters by name: `max_cluster_hamming`, `min_residual_joint_known_fraction`, `max_alternative_groups`, `landmark_budget`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(R^2 ceil(L/64)); O(E A ceil(A/64)); O(A^3 ceil(A/64)). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="di19"></a>
## di19 Soft candidate consensus, emission order and soft births

Form each soft candidate’s consensus from its supporting residual records, order candidates by the declared support classes, then fit soft births. Integer support classes and probability bands preserve complete alternatives. Equal support does not authorise choosing a single arbitrary candidate.

Implementation: [science/discovery/residual_births.py](../../haplotype_reconstruction/science/discovery/residual_births.py), [science/discovery/moves.py](../../haplotype_reconstruction/science/discovery/moves.py).

Parameters by name: `candidate_call_probability`, `min_candidate_known_fraction`, `min_dedup_joint_known_fraction`, `min_dedup_bidirectional_coverage`, `dedup_hamming_fraction`, `min_unique_sample_support`, `min_effective_sample_support`, `max_proposals_per_expansion`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(members x L); O(C (K + C) L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="di20"></a>
## di20 Birth-delete replacements

Use ranked birth classes to generate birth-delete replacements, fitting and deduplicating them in the declared order. Later classes are reached only while the replacement budget has room. This is a bounded neighbourhood, not exhaustive fixed-count search.

Implementation: [science/discovery/moves.py](../../haplotype_reconstruction/science/discovery/moves.py).

Parameters by name: `max_replacement_children`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: The replacement budget bounds fitted starts per call; the offered neighbourhood grows with births times K.

<a id="di21"></a>
## di21 Search archive per K

Maintain a permanent fitted-key archive and bounded per-count beams of whole NLL classes. A key removed from the beam is not fitted again merely because it is encountered again. If a leading class exceeds the beam allowance, that count’s beam can be empty while its last scored class remains eligible for selection.

Implementation: [science/discovery/search.py](../../haplotype_reconstruction/science/discovery/search.py).

Parameters by name: `beam_width`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(n K L); O(n log n). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="di22"></a>
## di22 Cavity refresh and exact-score backstop

Refresh cavity scores in the declared count-distance order and count exact-score cache misses against the backstop. Keep the score cache unbounded within the search: eviction would change the scientific budget. Record mean-field failures and preserve the selection classes scored before the backstop stopped refresh.

Implementation: [science/discovery/search.py](../../haplotype_reconstruction/science/discovery/search.py).

Parameters by name: `exact_score_backstop_base`, `exact_score_backstop_per_founder`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(K); O(N L K^2); O(N L K^3). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="di23"></a>
## di23 Selection, winning set and score margin

Form chained score classes across represented founder counts, select the preferred count, and carry the winning panels of that count. A panel from another count can bridge a score class but is never emitted as a winning alternative. Record score margin and ambiguity before known-content deduplication.

Implementation: [contracts/discovery.py](../../haplotype_reconstruction/contracts/discovery.py), [science/discovery/search.py](../../haplotype_reconstruction/science/discovery/search.py).

Parameters by name: `WIDE_CLASS_QUANTA`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(n log n). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="di24"></a>
## di24 Expansion loop (phases, improvement, replacements, bridge, expansion backstop, stop)

Expand every parent in a selected class before testing improvement. Run the declared death/gauge, birth and replacement phases, refresh scores after each phase, and compare with the pre-expansion value. Bridge and backstop rules retain whole straddling classes within their hard allowance; accumulate every omission and heuristic-cut flag.

Implementation: [science/discovery/search.py](../../haplotype_reconstruction/science/discovery/search.py).

Parameters by name: `expansion_backstop_base`, `expansion_backstop_per_founder`, `beam_width`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(K); O(max_iterations · N K² L / 10); O(N L K⁴). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="di25"></a>
## di25 Search status: limits, omissions, search_incomplete, unresolved, certification

Derive incomplete, unresolved, omission and certification fields after search. These diagnostics do not steer another search step. An unresolved block has no usable panel; a bounded but resolved result remains explicitly distinguishable from a certified search.

Implementation: [science/discovery/search.py](../../haplotype_reconstruction/science/discovery/search.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Linear in accumulated search diagnostics; omission lists can grow throughout archive expansion.

<a id="di26"></a>
## di26 Raw allele release rule

Evaluate raw allele evidence and informative carrier support in either direction, then apply the named pseudo-probability and minimum-support gates. Each sample counts at most once per founder/site when its absolute capped contribution exceeds the cell band, whether it favours REF or ALT; its signed contribution still enters the pseudo-odds. Selection’s favourable-carrier statistic is a separate rule. The zero-contribution boundary uses its declared tie band; the release probability gate remains its continuous comparison. Withheld alleles remain unknown rather than adopting the latent fit.

Implementation: [science/discovery/api.py](../../haplotype_reconstruction/science/discovery/api.py), [science/discovery/release.py](../../haplotype_reconstruction/science/discovery/release.py).

Parameters by name: `min_call_probability`, `min_supporters`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(K L x memberships per founder); O(N L); O(N K L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="di27"></a>
## di27 Block product: latent, latent_unknown, tied alternatives, shared calls, unknown_mass

Keep the latent representative separately from released calls and unknown-copy mass. Form shared calls across the winning same-count alternatives, retaining known-content alternatives only where distinct. A tied fit can therefore have no distinct released alternative while its search ambiguity flag remains true.

Implementation: [science/discovery/release.py](../../haplotype_reconstruction/science/discovery/release.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: One allele release per retained alternative plus intersection/deduplication over K L cells.

<a id="di28"></a>
## di28 Records, resume and stage execution

Declare independent block tasks from checked evidence and parameters, publish each typed product under its derivation, and assemble results by block index. Reuse requires checked publications, not arrival order. Worker count and storage route change scheduling only.

Implementation: [stages/discovery/tasks.py](../../haplotype_reconstruction/stages/discovery/tasks.py), [science/discovery/run.py](../../haplotype_reconstruction/science/discovery/run.py), [stages/discovery/records.py](../../haplotype_reconstruction/stages/discovery/records.py), [contracts/discovery.py](../../haplotype_reconstruction/contracts/discovery.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Sum of independent block costs; resident transition and pattern workspaces are per active block worker.

<a id="di29"></a>
## di29 Convention-reversal switch and diagnostics

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Pass the resolved reversal and diagnostic settings explicitly into named tie policies. Reverse declared arbitrary conventions while retaining model priors and fixed category preferences. Emit typed counters and optional census data; diagnostic collection must not feed decisions.

Implementation: [science/discovery/__init__.py](../../haplotype_reconstruction/science/discovery/__init__.py), [science/discovery/diagnostics.py](../../haplotype_reconstruction/science/discovery/diagnostics.py), [core/ties.py](../../haplotype_reconstruction/core/ties.py), [science/discovery/search.py](../../haplotype_reconstruction/science/discovery/search.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Constant work for a disabled event; enabled recording grows with emitted counters/events.

<!-- named mechanism anchors -->

## Named mechanisms

<a id="search-status"></a>

**Search status:** see [DI25](#di25).


Pair-backed evidence-v2 observation masks use the same lazy block intervals as GL declarations. A task gathers its original bool block before the unchanged discovery kernels; declaring all blocks does not recreate a resident whole-contig mask.
