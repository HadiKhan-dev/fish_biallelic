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

Generate one deletion start for every founder row. Rank the fixed-row assignment-only NLLs within K and admit the leading whole classes under `death_screen_width`, extending a straddling class through twice that width. Refit admitted starts with both usage schedules. Return admitted starts in their original order before the death/gauge family budget. This predictive screen defines the searched neighbourhood; it is not a bound on a converged endpoint.

Implementation: [science/discovery/moves.py](../../haplotype_reconstruction/science/discovery/moves.py).

Parameters by name: `death_screen_width`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: Up to K assignment-only scores, O(N B K³) per parent, followed by a K-independent number of admitted full-fit starts, O(I N L K²) with fixed screen width and iteration bound I. Row-collapse descendants in the shared refitter are additional work.

<a id="di15"></a>
## di15 Gauge rewires and death/gauge family admission

Generate gauge rewires, deduplicate each death/gauge start bucket by content, and admit deaths first under the complete-class proposal budget. Use one heuristic cut search at every K and record its use. Keep at most eight times `max_cut_ties` cuts (32 by default): the canonical prefix of the complete found family at its best found weight. Record any truncation as `tie_class_truncated`, cause `gauge_tie_class_truncated`, so it sets search limits and `search_incomplete` and prevents a generated-neighbourhood certificate. Its count is exact for the found family and a lower bound on all cuts at that weight; it does not count certified global maximum cuts. The scope states the weight and retained count. This cut budget is separate from the later whole-class death/gauge start budget. Before the death/gauge family budget, screen deduplicated gauge starts by fixed-row NLL at `gauge_screen_width`, independently of deaths and independently within each K. Keep whole NLL classes through twice the screen width and restore input order for the refit. Record exclusions as `gauge_screen_ranked_truncation`; corresponding death exclusions use `death_screen_ranked_truncation`. These are deliberate ranked-neighbourhood decisions, including omission of an overlarge straddling class, and do not claim an exhaustive fitted neighbourhood. Fit admitted starts before considering data births, which bypass this screen.

The cut search pins vertex zero on side False and visits the other vertices by descending weighted degree, with index ties following convention d9. Run strictly improving single-vertex searches from every unpinned singleton, an alternating start and `GAUGE_CUT_GW_PLANES` Goemans-Williamson starts; retain the distinct endpoints at the best integer (cross, within) weight. Move gains are initialized once and updated exactly after each move. The endpoints are locally optimal with respect to unpinned vertices; the search does not certify a global maximum.

For the GW starts, approximate the max-cut semidefinite relaxation by the Burer-Monteiro mixing method at rank ⌈√(2K)⌉ + 1. Stop after a sweep whose summed squared unit-vector movement is below `GAUGE_CUT_SDP_TOLERANCE`, or after `GAUGE_CUT_SDP_SWEEPS` sweeps. The tolerance is a step-size criterion, not an objective-error bound. Round with 16 random hyperplanes. The private generator uses `GAUGE_CUT_GW_SEED` and CRC-32 of the int64 weights in founder-content order; solve and round in that order, then map back. Production panels have distinct rows sorted by content and remapped assignments. Together with sample-count edge weights, this makes the ordered retained cuts and truncation records invariant to founder/sample input order for distinct founder rows. Equal rows keep input order and are excluded by the distinct-row production contract. Convention d9 reverses both local-search index ties and cut ordering; keeping d9 restores normal behaviour.

The SDP only supplies extra valid cuts. Integer local search scores every endpoint exactly; finite or unconverged mixing affects proposal coverage, not the validity of a cut score. Keeping the singleton and alternating starts ensures that adding GW starts cannot lower the best weight found. The tied family and retained prefix may change, so released discovery panels may change. No production SDP certificate or approximation ratio is promised.

Implementation: [science/discovery/moves.py](../../haplotype_reconstruction/science/discovery/moves.py).

Parameters by name: `gauge_screen_width`, `death_screen_width`, `max_cut_ties`, `max_proposals_per_expansion`, `MAX_CUT_CACHE_ENTRIES`, `GAUGE_CUT_GW_PLANES`, `GAUGE_CUT_GW_SEED`, `GAUGE_CUT_SDP_SWEEPS`, `GAUGE_CUT_SDP_TOLERANCE`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: For total integer edge weight W ≤ N, local search costs O(K² + K W) per start, and the K + P starts cost O((K + P)(K² + K W)), P = 16. Mixing costs O(T K² √K), T ≤ 2000; rounding costs O(P K √K). With fixed P and T this is O(K³ + K² W + T K² √K), polynomial with at most cubic K dependence. The complete found family has at most K + P members. Cache eviction changes recomputation only.

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

Form each soft candidate’s consensus from its supporting residual records and order candidates by the declared support classes. Harden and deduplicate novel rows, then screen the resulting soft-birth starts by their assignment-only NLL within K under `soft_birth_screen_width`. Only leading whole NLL classes are fully refitted; straddling classes fit through twice the width, and larger classes are omitted whole. Record all screen exclusions as `soft_birth_screen_ranked_truncation`. The emitted-row count precedes screening; the omitted-start count includes the screen. Integer support classes and probability bands preserve complete alternatives during generation. Data births are not screened.

Implementation: [science/discovery/residual_births.py](../../haplotype_reconstruction/science/discovery/residual_births.py), [science/discovery/moves.py](../../haplotype_reconstruction/science/discovery/moves.py).

Parameters by name: `soft_birth_screen_width`, `candidate_call_probability`, `min_candidate_known_fraction`, `min_dedup_joint_known_fraction`, `min_dedup_bidirectional_coverage`, `dedup_hamming_fraction`, `min_unique_sample_support`, `min_effective_sample_support`, `max_proposals_per_expansion`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(members x L); O(C (K + C) L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="di20"></a>
## di20 Birth-delete replacements

Use ranked birth classes to generate birth-delete replacements under the existing complete-class replacement budget. Rank the admitted starts by assignment-only NLL within each K, keep leading whole classes under `replacement_screen_width` with a twice-width straddling allowance, and refit in their original order. Record screen exclusions as `replacement_screen_ranked_truncation`, in addition to generation omissions. This is a screened neighbourhood, not exhaustive fixed-count search.

Implementation: [science/discovery/moves.py](../../haplotype_reconstruction/science/discovery/moves.py).

Parameters by name: `max_replacement_children`, `replacement_screen_width`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

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

Expand every parent in a selected class before testing improvement. Deaths, gauge rewires, soft births and replacements use their separate fixed-row NLL screens before complete refits; data births, including bridge data births, retain their full existing route. A small family already within its screen width needs no preliminary score. Assignment-only scores reuse the ordinary fit caches when the ordered panel agrees with the subsequent refit. Refresh cavity scores and compare with the pre-expansion value. Bridge and backstop rules retain whole straddling classes within their hard allowance; accumulate every omission and heuristic-cut flag. Fixed-row ranking is a predictive search assumption, not an admissible convergence bound.

Implementation: [science/discovery/search.py](../../haplotype_reconstruction/science/discovery/search.py).

Parameters by name: `expansion_backstop_base`, `expansion_backstop_per_founder`, `beam_width`, `death_screen_width`, `gauge_screen_width`, `soft_birth_screen_width`, `replacement_screen_width`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: The screens bound admitted initial full-fit starts per expansion independently of K for these four phases. With O(R) expansions through highest searched count R, their full-fit component is O(I N L R³) for fixed widths and before row-collapse descendants. This does not make the whole block cubic: scoring all K deletions with full assignment evaluations costs O(N B K³) per expansion and O(N B R⁴) over the ladder. Unscreened data births still admit O(K) starts per expansion, leaving O(I N L R⁴) full-fit work. Replacement enumeration and gauge generation can also contribute quartic work across expansions. Residual geometry and collapse descendants require their own bounds. Frontier partitioning improves a numerical constant, not these powers. The combined candidate still exceeds the cubic design requirement.

<a id="di25"></a>
## di25 Search status: limits, omissions, search_incomplete, unresolved, certification

Derive incomplete, unresolved, omission and certification fields after search. These diagnostics do not steer another search step. An unresolved block has no usable panel; a bounded but resolved result remains explicitly distinguishable from a certified search. The four `*_screen_ranked_truncation` causes describe starts deliberately excluded by fixed-row ranking. They do not set `search_incomplete` or computational-limit flags. Certification, when otherwise available, covers the generated and screened neighbourhood, not the excluded starts or all possible refitted neighbours. Computational omissions elsewhere, including gauge-cut truncations, still prevent certification.

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


### Emission frontier partitioning

Fixed-K starts share emission contractions within a frontier. A frontier is split
recursively when its shared emission tile cannot retain the target sample-vector
width within the numerical worker's scratch allowance. Each part evaluates every
state and sample of its panels; concatenation restores the original start order.
A single panel may use a narrower sample tile. This partition controls scratch
and locality only: it imposes no founder, start, iteration or state budget, and
leaves each state's ascending-bin summation and switching recurrence unchanged.
