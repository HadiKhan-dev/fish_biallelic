# Founder refinement

Refinement optimises a maximum-path copying objective at fixed founder count, then compares adjacent counts with the symmetric sequence null and bounded continuation. It preserves search order, exact local gains, numerical message conventions and scientific work accounts. N is all evidence samples, S informative samples, L sites, B blocks, K founders, Q unordered founder pairs, W units per proposal window, b bins and R alphabet size. Surrogate and bounded-search convergence is not global optimality. Retained biological alternatives and unperformed searches have separate meanings.

The sections below identify semantic units, not separately configurable pipeline steps. [Outputs](../user/outputs.md), [record contracts](../reference/records.md) and [costs](../developer/costs.md) describe their public boundaries.

Every deletion is decided by the ordinary count rule; there is no separate zero-gain cleanup, and addition seeds are not ranked before fitting. Count-rule settings and count-decision records are schema 2.

<a id="rc01"></a>
## rc01 Component workspace: objective site mask, informative samples, coded leaves, map hazards, normalisers

Restrict objective sites to kept sites with at least one called inference row; mark partial sites separately. Code uncalled leaves by row identity, identify informative evidence samples, and construct map hazards and count-dependent normalisers. Restrict only count-rule vectors to informative samples; total F still sums all evidence rows. An uninformative sample contributes an exact zero. A map is required and the fixed genotype-error model is checked.

Implementation: [science/assembly/refinement/capabilities.py](../../haplotype_reconstruction/science/assembly/refinement/capabilities.py), [science/assembly/refinement/costs.py](../../haplotype_reconstruction/science/assembly/refinement/costs.py), [science/assembly/refinement/component_workspace.py](../../haplotype_reconstruction/science/assembly/refinement/component_workspace.py).

Parameters by name: `reset_rate`, `map_probability_floor`, `genotype_error`, `cost_scale`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rc02"></a>
## rc02 Site log evidence and map-derived switch costs (float32 costs; reversed-orientation convention; boundary costs)

Form the floored site log evidence and map-derived switch cost in the declared arithmetic order. Store the cost as float32 and promote it inside recurrences. First informative and skipped sites use positive infinity; reversed cost is the forward cost of the next informative site. Sum hazards in forward site order. Sentinel cells and exact fill memoisation must not introduce evidence or alter rounding.

Implementation: [contracts/assembly.py](../../haplotype_reconstruction/contracts/assembly.py), [science/assembly/refinement/objective.py](../../haplotype_reconstruction/science/assembly/refinement/objective.py), [science/assembly/refinement/window.py](../../haplotype_reconstruction/science/assembly/refinement/window.py).

Parameters by name: `genotype_error`, `reset_rate`, `map_probability_floor`, `panel`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(1); O(gap). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rc03"></a>
## rc03 Surrogate bin size and binned surrogate emissions

Choose the surrogate bin size from the component length, contig-level floor where applicable and named proposal limits. Restart bins within each block. Sum emissions in site order, using hard dosages for fully called blocks and distinct uncalled-row pair codes elsewhere. This binning changes offered moves and therefore belongs to the scientific parameters, not runtime tuning.

Implementation: [science/assembly/refinement/component_workspace.py](../../haplotype_reconstruction/science/assembly/refinement/component_workspace.py), [science/assembly/refinement/run.py](../../haplotype_reconstruction/science/assembly/refinement/run.py).

Parameters by name: `proposal_max_bins`, `proposal_min_sites_per_bin`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L R_b^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rc04"></a>
## rc04 Evidence windows: resident or streamed route (3.4), leases, cover, exact fallback

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Choose resident or streamed evidence from the fixed dimension rule. Resident logs are immutable and cost arrays are cached by count/orientation. Streamed leases are private to a thread and founder count; nested leases reuse buffers, and reverse views share the same storage. Cover a requested stretch before a kernel reads it and compute an exact fallback outside the held window. Buffer capacity changes storage only; retained views obey their lease lifetime.

Implementation: [contracts/assembly.py](../../haplotype_reconstruction/contracts/assembly.py), [science/assembly/refinement/capabilities.py](../../haplotype_reconstruction/science/assembly/refinement/capabilities.py), [science/assembly/refinement/messages.py](../../haplotype_reconstruction/science/assembly/refinement/messages.py), [science/assembly/refinement/window.py](../../haplotype_reconstruction/science/assembly/refinement/window.py), [science/assembly/refinement/workspace.py](../../haplotype_reconstruction/science/assembly/refinement/workspace.py).

Parameters by name: `WINDOW_BYTES`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: Resident storage grows as N times sites/partial sites; streamed storage grows with N times the fixed window plus partial-site codes. Refills recompute exact evidence.

<a id="rc05"></a>
## rc05 Exact normalised block messages (full, in-place update, chains, lookahead kernels)

Construct normalised forward and backward block messages with identical site-step arithmetic on full, update and streaming routes. Recompute a block when its content or entering message changed, stopping propagation only at bitwise equal boundary messages. Relabel stored backward states after exchanges. Choose the first state attaining a diagnostic block maximum. Message equality is a cache certificate, not an approximate convergence test.

Implementation: [science/assembly/refinement/messages.py](../../haplotype_reconstruction/science/assembly/refinement/messages.py).

Parameters: `genotype_error` validates the fixed model. The shared emission in [science/shared/emissions.py](../../haplotype_reconstruction/science/shared/emissions.py) evaluates `max(log(0.99 * P + 0.01 / 3), -2.0)`. Refinement has no independent emission constants.

Cost and scope: O(N L (Q + K)). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rc06"></a>
## rc06 Reversed-orientation stored messages by clamping the forward recursion

Construct reversed stored boundaries by clamping the forward recursion with the entering transition cost and its original normaliser. Apply the same clamp on dense and reconstructed boundaries, including checkpoint boundaries. Carry the lexicographic secondary state through that transition. A reverse search discards its backward half before the next sweep; replacing the clamp with a mathematically equivalent backward recursion changes rounding.

Implementation: [science/assembly/refinement/checkpoints.py](../../haplotype_reconstruction/science/assembly/refinement/checkpoints.py), [science/assembly/refinement/messages.py](../../haplotype_reconstruction/science/assembly/refinement/messages.py), [science/assembly/refinement/search.py](../../haplotype_reconstruction/science/assembly/refinement/search.py).

Parameters by name: `order_breakpoints`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N B Q). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rc07"></a>
## rc07 SW5 message storage route: dense or block checkpoints; reconstruction reader and view lifetimes

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Select dense or checkpointed messages once per fixed-count search from declared dimensions, reserving the guarded layout even during a plain sweep. Reconstruct checkpoint segments by the same normalised block recurrences. Each reconstruction slot has a bounded view lifetime; indexed access copies, and parallel flank requests receive private copies. Reverse readers share underlying immutable boundaries. The routes must preserve all gains and decisions.

Implementation: [science/assembly/refinement/capabilities.py](../../haplotype_reconstruction/science/assembly/refinement/capabilities.py), [science/assembly/refinement/checkpoints.py](../../haplotype_reconstruction/science/assembly/refinement/checkpoints.py), [science/assembly/refinement/workspace.py](../../haplotype_reconstruction/science/assembly/refinement/workspace.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Retained boundary storage is O(A N Q B / stride), plus O(N B) normalisers and O(A N Q stride) per-reader reconstruction cache; A accounts for guarded messages.

<a id="rc08"></a>
## rc08 Shared-junction guard: C_shared from traced switch events

Trace old/new lexicographic paths for affected samples in a window that doubles until boundary states agree or the whole component is covered. Canonicalise switch events to their leftmost evidence interval, remove exact shared events, exclude paired excursions, then compare matching transitions across overlapping open intervals of different samples. Charge shared single/double transitions with the declared pairing rule, cap each sample by its saved cost, and sum in settling order. The result feeds the coarse-move guard only.

Implementation: [contracts/assembly.py](../../haplotype_reconstruction/contracts/assembly.py), [science/assembly/refinement/guard.py](../../haplotype_reconstruction/science/assembly/refinement/guard.py), [science/assembly/refinement/window.py](../../haplotype_reconstruction/science/assembly/refinement/window.py), [science/assembly/refinement/acceptance.py](../../haplotype_reconstruction/science/assembly/refinement/acceptance.py).

Parameters by name: `GUARD_TOLERANCE`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(width_sites x affected samples x Q); O(width x K); O(E^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rc09"></a>
## rc09 Count pass skeleton: start fit, deletion rounds, addition rounds, stops and count limit

Converge the start, then run ordinary count-deletion rounds until a round retains the larger panel or the minimum count is reached. Only at the addition-enabled level (the final level, when `final_additions` is set), test successive additions until rejection, a bound skip or the count limit; do not rerun deletion after an accepted addition. Since 4.9 `final_additions` is off by default: across 42 validation runs (418 chromosomes) no final-level addition was accepted, while an addition round's refits and decision took about two thirds of a whole-chromosome component's refinement (chromosome 1 at 0.5x: 251 s -> 80 s with the same panel). Refit every eligible candidate, expand tied endpoints, evaluate the symmetric null and record ancestry in that order. Restore the best same-count fit before constructing terminal alternatives.

Implementation: [science/assembly/refinement/count.py](../../haplotype_reconstruction/science/assembly/refinement/count.py).

Parameters by name: `COUNT_LIMIT_MULTIPLE`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L K³) per round for O(K) quadratic candidate fits; at most (accepted deletions + 1) deletion rounds and O(N L K⁴) worst-case count-refit work per pass. Additional fixed-K search, shadows and continuation histories are separate factors. See [costs](../developer/costs.md#natural-domain-count-refits).

<a id="rc10"></a>
## rc10 Optimistic count bounds and round skipping

Build blockwise free-panel bounds with independent cohort-best and per-sample-best deletion options. Test deletion only when all founders have private cells; test addition by the total bound before its per-sample bound. The deletion test sums its bound over samples at one option per block, because a K − 1 panel lacks the same row at a block for all its samples. For weights w ≥ 0, sum_s w_s F_s(Y) ≤ sum_b max_o sum_s w_s σ_s(b, o) + sum_s w_s n_s for every K − 1 panel and shadow Y. This bounds the point margin (w = 1), the leave-one-sample-out margin (w = 1 − e_t, minimised over t) and every bootstrap replicate (w = its draw counts from the decision’s own stream). The form is never looser than the per-sample one it replaces, which let each sample drop a different row. If a best-F restore later reads the adjacent K − 1 fit, a skipped round’s refits run before that comparison, under the round’s label. A certified skip is a recorded rejection, never unverified evidence for changing the count.

Implementation: [science/assembly/refinement/count.py](../../haplotype_reconstruction/science/assembly/refinement/count.py).

Parameters by name: `bootstrap_resamples`, `bootstrap_level`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L R_b^2 x (R_b if constrained)). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rc11"></a>
## rc11 Natural-domain admission and refit choice

Refit every eligible deletion candidate: founders without a private cell when present, otherwise every founder. For additions, refit every duplicate seed unless the existing optimistic bound proves the entire round unneeded. Compare converged endpoint F with the incumbent’s scale and origin content, taking the content-smallest member of the best tie class before the existing tied-verdict resolution. Direction endpoints, symmetric nulls, bootstrap streams, shadow families and decision tie census kinds are unchanged. No unrefitted margin or copy-response ranking limits admission; worker completion order cannot select a candidate.

Implementation: [science/assembly/refinement/count.py](../../haplotype_reconstruction/science/assembly/refinement/count.py).

Parameters by name: `COUNT_FANOUT_BYTES`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(K) candidate fits, O(N L K³) per count round under the quadratic fixed-K work model. A pass has at most (accepted deletions + 1) deletion rounds, hence a quartic worst case; fit iterations, canonical programmes and continuation histories are additional factors. The 4.0 saved-output census found 25 deletion rounds in 25 k30 passes (one each), 1,268 in 1,157 TroMau passes (1,066 with one, 77 with two, 11 with three and three with five), and 1,359 in 1,347 std passes (1,335 with one and 12 with two). These are recorded component passes, not independent biological replicates or a bound on future inputs. See [costs](../developer/costs.md#natural-domain-count-refits).

<a id="rc12"></a>
## rc12 Symmetric sequence null: death class, nearest-row shadow families, budget overflow and certificate

Build the larger panel’s death class, preferring founders without private cells. For each member, evaluate the complete nearest-row shadow product when it fits the scientific budget; otherwise evaluate only the conventional shadow and mark the count unverified. Share one round stream across members, refits and shadows. Opposed evaluated verdicts favour the smaller count; overflow alone supplies no count evidence. L5’s dimension-admitted proof may clear an unverified flag without changing the decision.

Implementation: [science/assembly/refinement/count.py](../../haplotype_reconstruction/science/assembly/refinement/count.py).

Parameters by name: `shadow_combinations`, `bootstrap_resamples`, `bootstrap_level`, `FAMILY_BYTES`, `PAIR_TABLE_BYTES`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L K^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rc13"></a>
## rc13 Tied-refit resolution, endpoint expansion and count-choice selection

Expand direction-tied endpoints before selecting refits. Evaluate every selected-class member with the same stream; opposing count verdicts retain smaller-panel outcomes, then rank margins and minimise switches among tied resulting panels. Preserve both observed verdict classes of an unresolved shadow family for continuation. Deduplicate option panels while preserving the nominal aggregate outcome, bind the continuation token and adopt its chosen panel. Do not re-add eliminated direction families when an endpoint evaluation table exists.

Implementation: [science/assembly/refinement/count.py](../../haplotype_reconstruction/science/assembly/refinement/count.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Up to one extra symmetric decision per tied class member and a minimum-switch score per distinct tied panel.

<a id="rc15"></a>
## rc15 Best-F panel restoration and restored adjacent-K comparisons

Remember a fit at each count only when it clears the named F improvement band. Restore a decisively better fit at the final count, then inspect its recorded comparisons and compare adjacent remembered counts with dedicated restored-comparison streams. A restored panel failing its own deletion test marks the call ambiguous. Retain unresolved shadows and unverified comparisons; terminal-family selection can subsequently replace the representative within its F class.

Implementation: [science/assembly/refinement/count.py](../../haplotype_reconstruction/science/assembly/refinement/count.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: At most one restored comparison for each adjacent remembered count, plus recorded-decision inspection.

<a id="rc16"></a>
## rc16 Terminal same-K equal-F family and its representative

Collect eligible ancestry fits, the best fit at the returned count and allowed ancestry-family panels. Find the returned panel’s F class, rescore its other members exactly and recompute that class. If multiple distinct panels remain, choose the convention-first minimum-switch member among those with actual fits. Release the full live same-count family. Unperformed continuation sources are not automatically terminal alternatives.

Implementation: [science/assembly/refinement/count.py](../../haplotype_reconstruction/science/assembly/refinement/count.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Exact full rescoring and minimum-switch scoring for retained class members.

<a id="rc17"></a>
## rc17 Bounded count continuation (incoming roots, fit ties, count choices; leaf budget) and terminal release

Run the nominal pass, then alternative input roots in sorted order. Expand the first unhandled continuation event breadth-first, retaining the nominal state and replaying other options from the original input with the same stream. Enforce the leaf budget before expansion and keep overflow unresolved. Compare winning F classes per count, prefer the smallest count and then fewest switches. Verification requires complete source coverage and finished matching histories; merely starting every option is insufficient. Preserve unverified histories separately from biological terminal alternatives.

Implementation: [science/assembly/refinement/count.py](../../haplotype_reconstruction/science/assembly/refinement/count.py).

Parameters by name: `alternative_branches`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: At most the alternative_branches leaf budget of complete count histories, with exact fit reuse; overflow remains unresolved.

<a id="rc18"></a>
## rc18 Component release families, pass-through and the window-work stop

Use whole-span incoming families containing the nominal panel as alternative starts; retain unresolved or incompatible families with their original span and source level. Pass through undersized components unchanged. A work-account stop anywhere in count continuation discards the attempt and restores input rows/families, adding a non-biological unperformed-search family. Stamp only newly generated families. Coarse moves and their guard require usable multi-block macro context, not merely a supplied context object.

Implementation: [science/assembly/refinement/run.py](../../haplotype_reconstruction/science/assembly/refinement/run.py).

Parameters by name: `window_work_coefficient`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: Linear in incoming/new release-family metadata, plus the count/refinement work being orchestrated.

<a id="rc19"></a>
## rc19 Polish orchestration (decided panel and equal-F alternatives)

Polish the decided panel and every distinct live-family panel, in start order, without changing count. Compare polished F classes and reconstruct the terminal minimum-switch family. Replace calls only when the selected panel or live family changes; retain fit summaries either way. If polish exhausts its work account, keep the completed count-pass result and record an unpolished status.

Implementation: [science/assembly/refinement/run.py](../../haplotype_reconstruction/science/assembly/refinement/run.py).

Parameters by name: `polish`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: One single-direction polish fit per distinct live start and terminal-family scoring.

<a id="rc20"></a>
## rc20 Level orchestration: scheduling, worker pools, memory admission

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Construct component tasks from shared level settings and checked records. Execute independent components through the supplied team and storage capabilities, then assemble by component number. Runtime memory pressure governs task admission and cancellation; it does not alter candidate budgets, work accounts, component results or output order. Scientific code receives no worker-pool or memory-governor decision.

Implementation: [science/assembly/refinement/costs.py](../../haplotype_reconstruction/science/assembly/refinement/costs.py), [stages/assembly/refinement.py](../../haplotype_reconstruction/stages/assembly/refinement.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Independent component costs plus shared-evidence and task-boundary overhead.

<a id="rc21"></a>
## rc21 Records, resume and census

Publish typed component and level records under their derivations, reusing only compatible complete products. Resume fits from complete done records or ordered saved sweeps, rebuilding transient messages and proposals. Branch records bind forced fit panels and count overrides as well as their input. Aggregate typed census fields after assembling all components; timings and report formatting remain outside science.

Implementation: [stages/assembly/records.py](../../haplotype_reconstruction/stages/assembly/records.py), [stages/assembly/refinement.py](../../haplotype_reconstruction/stages/assembly/refinement.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: One product per component plus declared saved fit/sweep records; resume validates records before use.

<a id="rs01"></a>
## rs01 Refinement objective F (per-homologue founder-switch max-path score)

Score every evidence sample under a per-homologue founder-switch maximum-path chain, reduced to unordered founder pairs. Promote the stored switch costs for the recurrence, retain the no-transition first informative site, and add the count-dependent normalisers only when scoring a whole panel. Fused and unfused steps preserve the same arithmetic. Missing evidence contributes zero; this objective is not a posterior probability.

Implementation: [contracts/assembly.py](../../haplotype_reconstruction/contracts/assembly.py), [science/assembly/refinement/__init__.py](../../haplotype_reconstruction/science/assembly/refinement/__init__.py), [science/assembly/refinement/objective.py](../../haplotype_reconstruction/science/assembly/refinement/objective.py), [science/assembly/refinement/parameters.py](../../haplotype_reconstruction/science/assembly/refinement/parameters.py), [science/assembly/refinement/reductions.py](../../haplotype_reconstruction/science/assembly/refinement/reductions.py), [science/assembly/refinement/search.py](../../haplotype_reconstruction/science/assembly/refinement/search.py).

Parameters by name: `reset_rate`, `map_probability_floor`, `genotype_error`, `PAIR_TABLE_BYTES`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L (Q + K)); O(N L K^2); O(threads x Q). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rs02"></a>
## rs02 Secondary keys of maximum-score paths (lexicographic switch cost E; integer switch counts)

Among exactly equal path scores, minimise accumulated switch cost, retaining the stay route before the first-founder route. A separate recursion minimises the integer switch count. These secondary keys use exact score equality, not a score quantum. Plain and lexicographic recursions must return bitwise equal primary scores.

Implementation: [science/assembly/refinement/proposals.py](../../haplotype_reconstruction/science/assembly/refinement/proposals.py), [science/assembly/refinement/objective.py](../../haplotype_reconstruction/science/assembly/refinement/objective.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N L Q). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rs03"></a>
## rs03 Exact move gains (content moves and suffix exchanges)

Compute a content move from the carried forward boundary, changed interval and unchanged backward boundary. Compute suffix-exchange gains by permuting the affected pair states and retaining the best unmoved state. Accumulate totals sequentially in sample order. The gains are exact for the supplied panel and messages; proposal surrogate scores never substitute for them.

Implementation: [science/assembly/refinement/gains.py](../../haplotype_reconstruction/science/assembly/refinement/gains.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(S K^3); O(S l K^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rs04"></a>
## rs04 Window surrogate H, window templates and coarse alphabets

Enumerate fine templates before coarse templates, then focal founders in index order. Fine alphabets contain prepared block rows; coarse alphabets contain distinct founder/context paths. Close the flank messages under the fixed Potts penalty and retain the incumbent alphabet row first. The surrogate selects offers only; template and focal order remain acceptance tie keys.

Implementation: [science/assembly/refinement/proposals.py](../../haplotype_reconstruction/science/assembly/refinement/proposals.py).

Parameters by name: `proposal_max_bins`, `proposal_min_sites_per_bin`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: Fine/coarse template counts scale with B and context spans divided by named strides, each with K focal problems; alphabet pair tables are quadratic in alphabet size.

<a id="rs05"></a>
## rs05 Canonical window programme (exact surrogate maximiser per founder and window)

Try the box certificate, then the main programme with its ordered children, bound pruning and dominance tests. Restart only after the initial backstop and when the final frontier permits it; the restart uses feasible beam/dual bounds and the declared beam cap. A final frontier or truncated-beam fallback keeps the incumbent; an unexplained lost answer raises. Retain an incumbent within tolerance, otherwise fix canonical rows by restricted programmes in order. This bounded solver does not certify an optimum when a fallback occurs.

Implementation: [science/assembly/refinement/parameters.py](../../haplotype_reconstruction/science/assembly/refinement/parameters.py), [science/assembly/refinement/proposals.py](../../haplotype_reconstruction/science/assembly/refinement/proposals.py).

Parameters by name: `CANONICAL_TOL`, `DOMINANCE_TOLERANCE`, `BOX_ARITHMETIC_MARGIN`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(W S R (b K + b^2)); O(S (b K + b^2)); O(children x kept x S x Q(b + 1)). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rs06"></a>
## rs06 Window-programme frontier cap and memory-budget retry schedule

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Start at the dimension-derived first programme budget and increase it by the fixed growth schedule up to the scientific frontier cap. Retry only pending focal problems, preserving their order and the same template work account. The final frontier fallback retains the incumbent. Available machine memory may delay execution but cannot change this schedule or its charged retries.

Implementation: [science/assembly/refinement/costs.py](../../haplotype_reconstruction/science/assembly/refinement/costs.py), [science/assembly/refinement/parameters.py](../../haplotype_reconstruction/science/assembly/refinement/parameters.py), [science/assembly/refinement/proposals.py](../../haplotype_reconstruction/science/assembly/refinement/proposals.py).

Parameters by name: `frontier_bytes`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: Retries repeat pending programme work; the frontier-byte cap fixes the allowed state count, while the programme beam may bind earlier.

<a id="rs07"></a>
## rs07 Per-template window work account and whole-component stop

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Allocate one work account per template, scaled from all evidence rows, window dimensions and the founder/alphabet size. Debit focal solves, retries, restarts, dominance, dual and canonical-fixing work in the declared integer order. Exhaustion discards every focal result of the template and propagates the component stop. L9 records deterministic logical-prefix work witnesses; physical work performed concurrently is separate. The bound is per template, not a whole-component complexity guarantee.

Implementation: [science/assembly/refinement/parameters.py](../../haplotype_reconstruction/science/assembly/refinement/parameters.py), [science/assembly/refinement/proposals.py](../../haplotype_reconstruction/science/assembly/refinement/proposals.py), [science/assembly/refinement/work_limit.py](../../haplotype_reconstruction/science/assembly/refinement/work_limit.py).

Parameters by name: `window_work_coefficient`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: The charged allowance is window_work_coefficient times N W b² max(K,R)³ per template; it does not bound the number of templates, sweeps or fits.

<a id="rs08"></a>
## rs08 Gauss-Seidel acceptance pass (junction walk, exchanges then content, caps)

Walk junctions with an exact carried forward message. Apply improving suffix exchanges before content moves, respecting the exchange cap and relabelling messages/proposal founders after each exchange. In full content rounds rank all improving active offers and accept the first guard-passing item, then recompute the ranking. Later rounds rescore untried items in their last full ranking. Update changed spans and advance the boundary. An optimistic pass aborts before reading a coarse-move guard and is discarded.

Implementation: [science/assembly/refinement/acceptance.py](../../haplotype_reconstruction/science/assembly/refinement/acceptance.py), [science/assembly/refinement/gains.py](../../haplotype_reconstruction/science/assembly/refinement/gains.py).

Parameters by name: `EXCHANGE_ROUNDS`, `CONTENT_FULL_ROUNDS`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(B N K^3); O(N l K^2); O(B). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rs09"></a>
## rs09 Acceptance tie rules (tie classes of exact gains, content order, convention)

Filter gains strictly above the acceptance threshold, then form chained gain classes. A wide class orders unequal gains by value; exact equalities still use content. Exchange keys compare the resulting pair of paths; content keys compare resulting path, founder slot and proposal index in pass order. Reverse only declared content conventions. L3 resolves identical-content, identical-gain exchanges by the smallest enumerated pair under both conventions.

Implementation: [science/assembly/orders.py](../../haplotype_reconstruction/science/assembly/orders.py), [science/assembly/refinement/gains.py](../../haplotype_reconstruction/science/assembly/refinement/gains.py), [science/assembly/refinement/acceptance.py](../../haplotype_reconstruction/science/assembly/refinement/acceptance.py).

Parameters by name: `WIDE_QUANTA`, `WIDE_CLASS_QUANTA`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(n^2); O(B); O(K^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rs10"></a>
## rs10 Coarse-move guard decision and lazy lexicographic rescoring

For a coarse improving offer, lazily recompute the lexicographic gain and assert bitwise agreement with the plain gain. Only when gain plus switch-cost change crosses the negative guard tolerance, compute shared switch savings and require the residual gain to exceed the acceptance threshold. A rejection continues the ranked walk. Fine moves bypass this guard; the lexicographic-item counter controls safe message-mode reuse, not acceptance.

Implementation: [science/assembly/refinement/acceptance.py](../../haplotype_reconstruction/science/assembly/refinement/acceptance.py), [science/assembly/refinement/guard.py](../../haplotype_reconstruction/science/assembly/refinement/guard.py).

Parameters by name: `GUARD_TOLERANCE`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(width x affected samples x K^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rs11"></a>
## rs11 Fixed-K search: sweeps, direction alternation, convergence, sweep limit, exactness check, sweep mode, resume

Restore compatible saved rows, recompute messages/proposals, and alternate pass direction while holding the content convention fixed. An optimistic abort reruns the entire pass from its original panel with lexicographic messages. Rescore after accepted moves or missing resumed scores; check the predicted gain against the full score every sweep, including zero-move sweeps. Polish only at ordinary fixed points. Zero accepted moves means convergence; reaching the sweep limit records nonconvergence.

Implementation: [science/assembly/refinement/parameters.py](../../haplotype_reconstruction/science/assembly/refinement/parameters.py), [science/assembly/refinement/reductions.py](../../haplotype_reconstruction/science/assembly/refinement/reductions.py), [science/assembly/refinement/fit.py](../../haplotype_reconstruction/science/assembly/refinement/fit.py).

Parameters by name: `sweep_limit`, `WINDOW_MEMO_BYTES`, `tolerance`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L K^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rs12"></a>
## rs12 Best-of-two fit selection, minimum-switch prior and canonical fit

Reuse fits only for their full start identity. Run both direction starts, except the single-direction polish route, and compare their F values in the named tie policy. When distinct panels share the winning class, minimise switch count within that class and retain the other tied variant. Canonicalise founder rows after choosing. Branch replay can explicitly force a retained tied panel; it cannot invent an unrecorded alternative. The named tie policy receives the run's reversal convention explicitly (L8).

Implementation: [contracts/assembly.py](../../haplotype_reconstruction/contracts/assembly.py), [science/assembly/orders.py](../../haplotype_reconstruction/science/assembly/orders.py), [science/assembly/refinement/__init__.py](../../haplotype_reconstruction/science/assembly/refinement/__init__.py), [science/assembly/refinement/fit.py](../../haplotype_reconstruction/science/assembly/refinement/fit.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N L Q). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rs14"></a>
## rs14 Polish schedule at ordinary fixed points (escape, long repair, bridge; verified)

At a polish fixed point try beam escape, then long repair if nothing was accepted, then a bridge if still unchanged and fine offers exist. Bridges use the beam escape’s fine offers. Verify each accepted family by full rescoring and its named exactness tolerances, requiring positive improvement. Polish never changes founder count and never reruns the count decision; a work stop keeps the unpolished count-pass panel.

Implementation: [science/assembly/refinement/fit.py](../../haplotype_reconstruction/science/assembly/refinement/fit.py).

Parameters by name: `polish`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: One single-direction fit per polished start; each ordinary fixed point can add escape, long-repair and bridge searches.

<a id="rs15"></a>
## rs15 Window-beam proposer and beam escape

Canonicalise founder order, construct the map-aware binned surrogate and apply the box gate. Rank beam children by anchored quantised bounds, retain the bounded cut cohort, remove dominated children, and restore flat child order. Retain only a changed final path clearing the beam gain rule. Apply one guarded acceptance pass over the offers and verify its exact gain. Beam width and cohort truncation limit the searched paths.

Implementation: [science/assembly/refinement/beam.py](../../haplotype_reconstruction/science/assembly/refinement/beam.py).

Parameters by name: `POLISH_BEAM_WIDTH`. Values and provenance are in the [generated parameter reference](../reference/parameters.md); conventions are explicit [tie policies](../developer/ties.md).

Cost and scope: O(W N (b^2 Q + h b (Q + R(K + b)) + h^2 Q)). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

A width hit counts a window/founder problem in which the beam discarded at least
one child. It does not count an accepted move or establish that a better panel
was lost. The beam also retains the largest discarded upper bound and its gap
above the best retained surrogate value. A gap within the offer quantum certifies
that surrogate comparison despite a width hit. A positive gap leaves that
comparison uncertified; it does not measure a loss in the exact refinement
objective. The surrogate uses binned evidence and selects proposals, while
acceptance and the verification of accepted gains use every objective site.

Measured evidence: saved refinement inputs were replayed with both polish widths at 1x (64 and 16),
2x and 4x. The replays covered terminal components and sampled level-1 and level-2 components of chr1, chr3 and
chr8 of the N = 320 simulation at depths 5, 1 and 0.5, and three matrix designs, including K = 30. Founder rows
were identical to the control in every component of every case except one: in the std design's terminal
component, both wider settings reached a different endpoint that was worse on the refinement objective (11.2
nats) and on truth (562 and 567 more one-to-one founder-allele errors in 1.97 million). Wider beams used 0.8x
to 1.4x the CPU. Width hits are therefore not evidence of lost accuracy, and the widths stay unchanged. The
largest replays (chr3 at depth 0.5 at all widths, and depth 1 at 2x) did not finish within their lab
time limit and are not part of this evidence.

<a id="rs16"></a>
## rs16 Long repair (long one-founder moves with one private correction)

Generate long-window beam offers, choose the incumbent and agreement-ranked donors, then enumerate deduplicated private corrections. Score every variant exactly; filter by the strict acceptance threshold before forming the maximum gain band. Select the content-smallest rows in that band, offer them as coarse moves and verify accepted gains. A rejected chosen variant does not trigger a search through the offer’s discarded variants.

Implementation: [science/assembly/refinement/fit.py](../../haplotype_reconstruction/science/assembly/refinement/fit.py).

Parameters by name: `LONG_WINDOW`, `LONG_STRIDE`, `LONG_CORRECTION`, `LONG_DONORS`, `LONG_BEAM_WIDTH`, `BRIDGE_BATCH_BYTES`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(B K / LONG_STRIDE); O(N · LONG_WINDOW blocks · K²). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rs17"></a>
## rs17 Bridges (fine-window precursor plus suffix exchange)

Choose each fine-window precursor in the maximum gain band, apply it privately, then examine suffix exchanges across its cuts. Choose the exchange within its band before testing the combined gain; a tied representative can therefore leave no offer. Compare surviving bridge offers by band and the declared tuple key, then verify the selected combined gain. Bridges bypass the coarse-move guard.

Implementation: [science/assembly/refinement/fit.py](../../haplotype_reconstruction/science/assembly/refinement/fit.py).

Parameters by name: `BRIDGE_BATCH_BYTES`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(#windows x N x 8 blocks x K^2); O(N K^3). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rs18"></a>
## rs18 Weak label junctions (diagnostic)

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Report junctions whose best exchange gain is strictly above the negative weak-junction margin. Keep this diagnostic separate from the acceptance threshold and from released panel state. No downstream scientific decision reads the report.

Implementation: [science/assembly/refinement/fit.py](../../haplotype_reconstruction/science/assembly/refinement/fit.py).

Parameters by name: `WEAK_JUNCTION_MARGIN`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(B). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

### Parallel evaluation with ordered selection

Independent count refits retain their logical fit ordinals and are evaluated on
child CPU shares before the original ordered count selection. The polish beam
bank likewise evaluates each unchanged `(window, founder)` beam problem through
the existing template team. Dimension-based cost ordering affects dispatch only;
proposal collection and all beam-bound receipts return to original task order.
Dynamic dispatch lets expensive windows use CPUs released by other work and avoids
waiting for a statically assigned chunk of unusually costly beam problems.

Fine-window/suffix bridge trials each start from the same current panel. Their
private panels, forward/backward chains and streaming evidence windows are
independent. Checkpoint flanks are copied by the coordinating thread before
parallel evaluation; byte-bounded waves limit simultaneous scratch. Offers are
collected in the original window/cut/pair order before the existing global gain
and content selection. The accepted panel is still rescored by the original path.

Beam-escape width hits count window/founder problems whose approximate surrogate
search discarded children at the existing width. They measure repeated bounded
proposal work, rather than accepted edits or failed exact acceptance. Parallel
execution preserves every width, work account, offer rule and acceptance bound.

### Initial-fit execution and compilation

The initial fixed-K fit keeps the same best-of-two directional searches and
Gauss–Seidel acceptance schedule described above. Its searches run concurrently;
independent proposal templates and per-sample recursions use their existing
teams. An accepted update still feeds the next update in the original order.
Sample kernels retain their useful-width policy and acquire the current child
share at their boundaries. Compilation is not an optimisation step and does not
change convergence, the objective, fit ordinals or any count decision.

The assembly preload declares the refinement entry points before runtime warm-up,
including the acceptance pass, template solver, lexicographic messages and exact
score kernels. Init compiles their supported signatures in the engine's process
pool. A fresh scientific worker then loads the compiled implementations from the
same run cache. Cached implementations retain their exact argument-type checks,
including layout and read-only flags. First-use compilation and warm numerical
execution are measured separately; compiler-lock waits among proposal workers
must not be interpreted as serial Gauss–Seidel work.

The unordered-state pair constructor is used both as an inlined numerical helper
and directly from Python. Its declared Python entry point is cacheable too;
inlining its native callers does not require recompiling that Python entry point
in each worker. This changes cache persistence only, not state ordering.
