# Founder refinement

Refinement optimises a maximum-path copying objective at fixed founder count, then compares adjacent counts with the symmetric sequence null and bounded continuation. It preserves search order, exact local gains, numerical message conventions and scientific work accounts. N is all evidence samples, S informative samples, L sites, B blocks, K founders, Q unordered founder pairs, W units per proposal window, b bins and R alphabet size. Surrogate and bounded-search convergence is not global optimality. Retained biological alternatives and unperformed searches have separate meanings.

The sections below identify semantic units, not separately configurable pipeline steps. [Outputs](../user/outputs.md), [record contracts](../reference/records.md) and [costs](../developer/costs.md) describe their public boundaries.

Every deletion is decided by the ordinary count rule; there is no separate zero-gain cleanup. Count-rule settings and count-decision records are schema 2.

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
## rc04 Evidence windows: resident or streamed route, leases, cover, exact fallback

**Accretion:** Windows bound evidence storage while preserving the site recurrence.

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

Implementation: [science/assembly/refinement/checkpoints.py](../../haplotype_reconstruction/science/assembly/refinement/checkpoints.py), [science/assembly/refinement/messages.py](../../haplotype_reconstruction/science/assembly/refinement/messages.py), [science/assembly/refinement/fit.py](../../haplotype_reconstruction/science/assembly/refinement/fit.py).

Parameters by name: `order_breakpoints`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N B Q). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rc07"></a>
## rc07 Message storage route: dense or block checkpoints; reconstruction reader and view lifetimes

**Accretion:** Sparse boundary messages trade storage for reconstruction work.

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
## rc09 Count pass skeleton: start fit, deletion rounds and stops

Converge the start, then run ordinary count-deletion rounds until a round retains the larger panel or the minimum count is reached. **Refinement never increases K.** Fixed-K fitting preserves the founder count, and count rounds only remove founders; there is no final-level addition route or setting. Refit every eligible candidate, expand tied endpoints, evaluate the symmetric null and record ancestry in that order. Restore the best same-count fit before constructing terminal alternatives. Restored-panel diagnostics can compare an already-computed adjacent larger fit from an earlier deletion round; that comparison preserves count ambiguity and does not propose or accept a founder addition.

Implementation: [science/assembly/refinement/count.py](../../haplotype_reconstruction/science/assembly/refinement/count.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

The minimum count is two founders; there is no addition count limit.

Cost and scope: O(N L K³) per round for O(K) quadratic candidate fits; at most (accepted deletions + 1) deletion rounds and O(N L K⁴) worst-case count-refit work per pass. Additional fixed-K search, shadows and continuation histories are separate factors. See [costs](../developer/costs.md#natural-domain-count-refits).

<a id="rc10"></a>
## rc10 Optimistic count bounds and round skipping

Build blockwise free-panel scores for every deletion option. Test the deletion bound only when all founders have private cells. The deletion test sums its bound over samples at one option per block, because a K − 1 panel lacks the same row at a block for all its samples. For weights w ≥ 0, sum_s w_s F_s(Y) ≤ sum_b max_o sum_s w_s σ_s(b, o) + sum_s w_s n_s for every K − 1 panel and shadow Y. This bounds the point margin (w = 1), the leave-one-sample-out margin (w = 1 − e_t, minimised over t) and every bootstrap replicate (w = its draw counts from the decision’s own stream). The form is never looser than the per-sample one it replaces, which let each sample drop a different row. If a best-F restore later reads the adjacent K − 1 fit, a skipped round’s refits run before that comparison, under the round’s label. A certified skip is a recorded rejection, never unverified evidence for changing the count.

Implementation: [science/assembly/refinement/count.py](../../haplotype_reconstruction/science/assembly/refinement/count.py).

Parameters by name: `bootstrap_resamples`, `bootstrap_level`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L R_b^2 x (R_b if constrained)). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rc11"></a>
## rc11 Natural-domain admission and refit choice

Refit every eligible deletion candidate: founders without a private cell when present, otherwise every founder. Compare converged endpoint F with the incumbent’s scale and origin content, taking the content-smallest member of the best tie class before the existing tied-verdict resolution. Direction endpoints, symmetric nulls, bootstrap streams, shadow families and decision tie census kinds are unchanged. No unrefitted margin or copy-response ranking limits admission; worker completion order cannot select a candidate.

Implementation: [science/assembly/refinement/count.py](../../haplotype_reconstruction/science/assembly/refinement/count.py).

Parameters by name: `COUNT_FANOUT_BYTES`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(K) candidate fits, O(N L K³) per count round under the quadratic fixed-K work model. A pass has at most (accepted deletions + 1) deletion rounds, hence a quartic worst case; fit iterations, canonical programmes and continuation histories are additional factors. See [costs](../developer/costs.md#natural-domain-count-refits).

<a id="rc12"></a>
## rc12 Symmetric sequence null and shared nearest-shadow family rule

Build the larger panel’s death class, preferring founders without private cells. For each member use the [shared shadow-family rule](assembly.md#aj33): evaluate the complete nearest-row product only when the total test passes and the product fits the budget; otherwise evaluate the conventional incumbent. A passing total with budget overflow records count_unverified. Share one round stream across members, refits and shadows. Opposed shadow verdicts withhold the larger count while preserving the selected bootstrap band; disagreement between death-class references retains its independent ambiguity rule. Prefix-sharing scores admitted complete families in canonical order. Exact per-sample misses and admitted family vectors enter the component's shared score ledger under the canonical full-panel key, preserving duplicate founders. Later endpoints reuse vectors while recomputing their own total differences, bootstrap bands and decisions. The shared family rule still selects complete evaluation, a single conventional shadow or overflow; caching never expands that domain. Exact decision-key ties use the shared convention when reducing those scores. The shadow-family prior is `smaller_k+nearest_shadow`.

Implementation: [science/assembly/refinement/count.py](../../haplotype_reconstruction/science/assembly/refinement/count.py).

Parameters by name: `shadow_combinations`, `bootstrap_resamples`, `bootstrap_level`, `FAMILY_BYTES`, `SCORE_LEDGER_ENTRIES`, `SCORE_LEDGER_BYTES`, `SCORE_LEDGER_ENTRY_OVERHEAD`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

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

Run the nominal pass, then alternative input roots in sorted order. An input root of the nominal count follows the nominal pass's adopted deletions instead of deciding the count again: it is converged from its own start, the corresponding founder (unique mutual nearest row) is deleted at each adopted round, and it is re-converged after each, so it ends at its own panel of the nominal final count; a root whose panels meet the nominal root's is the nominal history from there on. A root of another count, a root without an unambiguous founder correspondence, a root one of whose fits ends in tied endpoints, and every root when the nominal pass restored a best-F panel run their own complete pass (in a branch of their own, so no fit record of the abandoned replay is read). Expand the first unhandled continuation event breadth-first, retaining the nominal state and replaying other options from the original input with the same stream. Enforce the leaf budget before expansion and keep overflow unresolved. Compare winning F classes per count, prefer the smallest count and then fewest switches. Verification requires complete source coverage and finished matching histories; merely starting every option is insufficient. Preserve unverified histories separately from biological terminal alternatives. Independent histories run concurrently, with outcomes exactly those of this sequential order ([RC22](#rc22)).

Implementation: [science/assembly/refinement/count.py](../../haplotype_reconstruction/science/assembly/refinement/count.py).

Parameters by name: `alternative_branches`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: At most the alternative_branches leaf budget of complete count histories, with exact fit reuse; overflow remains unresolved. A root that follows the nominal decisions costs one fit plus one per adopted deletion. Its own count decisions are not evaluated: a root whose decisions would have differed from the nominal root's is not detected.

<a id="rc18"></a>
## rc18 Component release families, pass-through and the window-work stop

Use whole-span incoming families containing the nominal panel as alternative starts; retain unresolved or incompatible families with their original span and source level. Pass through undersized components unchanged. A work-account stop anywhere in count continuation discards the attempt and restores input rows/families, adding a non-biological unperformed-search family. Stamp only newly generated families. Coarse moves and their guard require usable multi-block macro context, not merely a supplied context object.

Implementation: [science/assembly/refinement/run.py](../../haplotype_reconstruction/science/assembly/refinement/run.py).

Parameters by name: `window_work_coefficient`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: Linear in incoming/new release-family metadata, plus the count/refinement work being orchestrated.

<a id="rc19"></a>
## rc19 Polish orchestration (decided panel and equal-F alternatives)

Polish the decided panel and every distinct live-family panel, in start order, without changing count. Fit ordinals are bound in start order before any polish fit runs; the decided panel's polish runs first and the alternatives' polishes then run concurrently, so that an alternative meeting the decided panel's problems takes their answers from the component's stores ([RC22](#rc22)). Compare polished F classes and reconstruct the terminal minimum-switch family. Replace calls only when the selected panel or live family changes; retain fit summaries either way. If polish exhausts its work account, keep the completed count-pass result and record an unpolished status.

Implementation: [science/assembly/refinement/run.py](../../haplotype_reconstruction/science/assembly/refinement/run.py).

Parameters by name: `polish`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: One single-direction polish fit per distinct live start and terminal-family scoring.

<a id="rc20"></a>
## rc20 Level orchestration: scheduling, worker pools, memory admission

**Accretion:** Dimension-based admission governs resources independently of scientific search limits.

Construct component tasks from shared level settings and checked records. Execute independent components through the supplied team and storage capabilities, then assemble by component number. Runtime memory pressure governs task admission and cancellation; it does not alter candidate budgets, work accounts, component results or output order. The component reserves `SCORE_LEDGER_BYTES` once for additional exact score misses shared by all fits and continuation branches. Locked admission enforces both `SCORE_LEDGER_ENTRIES` and the byte allowance, charging owned ndarray storage, panel/row tuples, every referenced Python integer and conservative dictionary/metadata overhead. Saturation recomputes the same score. Admission remains charged for the component lifetime, including misses later published by fits, so retained keys and dictionary capacity cannot reopen the allowance. Completed-fit vectors retain their separate lifetime and unconditional publication precedence; these limits do not bound the entire fitted-result ledger. Family scoring scratch and score-result transients remain separately reserved. The component also reserves its window-result store and its shared polish answers ([RC22](#rc22)), both dimension-only budgets. Scientific code receives no worker-pool or memory-governor decision.

Implementation: [science/assembly/refinement/costs.py](../../haplotype_reconstruction/science/assembly/refinement/costs.py), [stages/assembly/refinement.py](../../haplotype_reconstruction/stages/assembly/refinement.py).

Parameters by name: `SCORE_LEDGER_ENTRIES`, `SCORE_LEDGER_BYTES`, `SCORE_LEDGER_ENTRY_OVERHEAD`, `WINDOW_STORE_MIN_BYTES`, `WINDOW_STORE_MAX_BYTES`, `WINDOW_STORE_TEMPLATE_BYTES`, `POLISH_SHARE_MIN_BYTES`, `POLISH_SHARE_MAX_BYTES`, `POLISH_SHARE_TEMPLATE_BYTES`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: Independent component costs plus shared-evidence and task-boundary overhead.

<a id="rc21"></a>
## rc21 Records, resume and census

Publish typed component and level records under their derivations, reusing only compatible complete products. A component whose work (samples × sites × K³, the same estimate that orders component tasks) reaches `FIT_RECORD_WORK` also publishes every completed sweep and fit under its own scope before its component record, so that a rerun of an interrupted component resumes its fits from complete done records or ordered saved sweeps, rebuilding transient messages and proposals; smaller components take about a minute or less and are recomputed. Branch records bind forced fit panels and count overrides as well as their input. Aggregate typed census fields after assembling all components; timings and report formatting remain outside science.

Implementation: [stages/assembly/records.py](../../haplotype_reconstruction/stages/assembly/records.py), [stages/assembly/refinement.py](../../haplotype_reconstruction/stages/assembly/refinement.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: One product per component plus declared saved fit/sweep records; resume validates records before use.

<a id="rc22"></a>
## rc22 Component window store, shared polish answers and concurrent count histories

**Accretion:** Identical sub-problems of tied alternatives are solved once per component, and independent count histories run at once; every output is the sequential driver's.

One window-result store serves every fit of a component attempt: the nominal count pass, every incoming root (replayed or complete), every continuation branch and the polish fits. A window solve is a pure function of its template, the content convention, the panel's rows in the window and the flank messages at its two ends, given the component's immutable solver (emissions, row domains, template list, coarse context alphabets, penalty, frontier and seeded-search controls). The store is bound to that solver object and refuses any other, since a template index means nothing outside its solver's list. Its key is the content digest used by every refinement store: a 128-bit BLAKE2b digest of the template index, the convention and the dtype, shape and bytes of the window rows and both flanks. Keys are digests, not inputs, so two different problems share a result only on a 128-bit collision, with probability below n²/2¹²⁹ for n stored problems (about 1e-28 at n = 2×10⁵); the polish stores carry the same qualification ([RS15](#rs15)). Values are serialised; every hit returns private arrays. A hit restores the template's logical work receipt and does no physical programme work, so work accounts, fit summaries and sweep histories are those of a solve. A template that exhausts its account raises before it is stored. A requester of a problem that another fit is solving at that moment waits for that result; if that solve fails, a waiter solves it.

Retention is bounded by a dimension-only budget, 8 KiB per window template and K² clamped to [256 MiB, 8 GiB] (`WINDOW_STORE_*`; K is the largest root count). Replays of captured components with tied roots, continuations and polish alternatives stored 3.5 to 9.4 KiB per template and K² for every distinct solve. Entries are evicted least recently used first, except the adopted path: the keys each count fit used are tracked for its last `WINDOW_STORE_TRACKED_FITS` fits, and when a count history adopts a fit (its start fit, an accepted deletion, a restored panel) that fit's entries are pinned, within half the budget. Replays of tied roots and continuation branches meet the adopted path again after the pass has moved on. Tracked keys are charged to the budget. Eviction, pinning and the budget change runtime only: a miss solves the identical problem again.

The polish searches keep their own last-problem slots ([RS14](#rs14)); a slot miss also consults the answers that every polish fit of the component shared under the same complete keys (beam answers in their window founder frame, scalar-gain tables, plain bridge trials), and every slot write is offered to that share. A gain table is shared as one object and completed by whichever polish fit meets a new candidate; its values are pure functions of the framed problem and candidate. The share is bounded by 2 KiB per template and K², clamped to [64 MiB, 2 GiB] (`POLISH_SHARE_*`), charged by held arrays and objects as tables grow, least recently used first.

Count histories that do not read each other run concurrently as children of the component's CPU share: the complete passes of roots of another count start with the nominal pass; the replays of roots of the nominal count advance with the nominal pass; the complete passes of roots that cannot be replayed start together as soon as the nominal pass ends; and all histories created by one breadth-first generation of continuation events run together, after that generation is expanded.

A replay's fit after the nominal pass's k-th adopted deletion depends only on the replay's previous fit, the nominal panel that deletion was taken from and the deleted founder ([RC17](#rc17)), so its start fit runs in the nominal start fit's batch and its k-th deletion fit in the batch of the nominal round that follows the k-th adopted deletion: one more child of that batch, whose window problems are mostly the adopted fit's, already in the window store. Every check and fit is the one the replay makes after the nominal pass, in the same order. Until the nominal pass is known to be replayable (no restored best-F panel, its returned panel the end of the adopted deletions), the replay's fit records, tie counts, tie-census events and work counters are held privately; they are then handed on as the replay would have produced them after the nominal pass, or dropped, with the fits only the replay requested removed from the shared fit ledger, so that the root's complete pass computes and records them itself. A long chain of lone fits (one per adopted deletion) thus leaves the end of the count pass. Expansion reads only histories of earlier generations, in the sequential FIFO order and with the sequential leaf budget, and results are collected and selected in the sequential order ([RC17](#rc17)). Concurrent near-duplicate histories lose no store hits: a problem being solved by one is waited for by the other. A fit requested by two histories, completed or in progress, is computed once (the shared fit ledger); the reported reuse is counted in the sequential driver order from each branch's requests. Each history is a pure function of its root, its forced choices and the immutable component, so rows, F, count decisions with margins and bands, families, fit lists, work counters and tie counts do not depend on the schedule or the CPU count.

Count decisions of concurrent histories take turns through one component gate (death classes, shadow families, scoring fan-outs, deletion bounds and terminal families), so their scratch stays within one `COUNT_FANOUT_BYTES` allowance and the shared decision caches have one writer; decisions cost well under 1 % of a count pass. Every fit of every history is a child holding at least one CPU of the component's share, so at most one fit per declared core runs, each within the per-core child reservation. Between its fits a history holds one round's candidate panels and score vectors, and histories at different counts can keep up to two more switch-cost arrays of a resident component referenced ([RC04](#rc04)); the component reserves both for `alternative_branches` concurrent histories. All of this runs in the component task's process: one process, and Numba threads per running child equal to its live share, never more than the task's declared cores in total. The first failure of a history abandons the attempt: every other history stops at its next window solve, and the first failure is raised once all have joined.

A window-work stop ([RS07](#rs07)) names fit ordinals in the sequential driver order, which interleaved histories do not keep. A stop in an attempt that ran any histories concurrently therefore reruns the component sequentially from a fresh work account, fit ordinals, ledger and tie counter; the stores keep their exact results, records the abandoned attempt saved are neither read nor saved again (records of an earlier process are read as usual), and its diagnostic tie-census events are discarded. The stop, its domains and its logical work are then those of the sequential order; physical work, which includes concurrent speculation, remains a diagnostic. Fit records are kept per branch; a fit shared by two histories is recorded under the branch that computed it, and a resumed history that finds no record of its own recomputes the fit or takes it from the ledger.

Implementation: [science/assembly/refinement/cache.py](../../haplotype_reconstruction/science/assembly/refinement/cache.py), [science/assembly/refinement/polish_cache.py](../../haplotype_reconstruction/science/assembly/refinement/polish_cache.py), [science/assembly/refinement/count.py](../../haplotype_reconstruction/science/assembly/refinement/count.py), [science/assembly/refinement/fit.py](../../haplotype_reconstruction/science/assembly/refinement/fit.py), [science/assembly/refinement/run.py](../../haplotype_reconstruction/science/assembly/refinement/run.py).

Parameters by name: `WINDOW_STORE_MIN_BYTES`, `WINDOW_STORE_MAX_BYTES`, `WINDOW_STORE_TEMPLATE_BYTES`, `WINDOW_STORE_ENTRY_OVERHEAD`, `WINDOW_STORE_TRACK_BYTES`, `WINDOW_STORE_TRACKED_FITS`, `POLISH_SHARE_MIN_BYTES`, `POLISH_SHARE_MAX_BYTES`, `POLISH_SHARE_TEMPLATE_BYTES`, `COUNT_FANOUT_BYTES`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: Hashing and serialising each window solve (linear in the window rows and two flanks); identical problems of tied alternatives cost a lookup. The wall time of a component with tied histories approaches its longest history instead of their sum; the CPU of messages, exchange scans and acceptance passes of every history remains.

<a id="rs01"></a>
## rs01 Refinement objective F (per-homologue founder-switch max-path score)

Score every evidence sample under a per-homologue founder-switch maximum-path chain, reduced to unordered founder pairs. Promote the stored switch costs for the recurrence, retain the no-transition first informative site, and add the count-dependent normalisers only when scoring a whole panel. Fused and unfused steps preserve the same arithmetic. Missing evidence contributes zero; this objective is not a posterior probability.

Implementation: [contracts/assembly.py](../../haplotype_reconstruction/contracts/assembly.py), [science/assembly/refinement/__init__.py](../../haplotype_reconstruction/science/assembly/refinement/__init__.py), [science/assembly/refinement/objective.py](../../haplotype_reconstruction/science/assembly/refinement/objective.py), [science/assembly/refinement/parameters.py](../../haplotype_reconstruction/science/assembly/refinement/parameters.py), [science/assembly/refinement/reductions.py](../../haplotype_reconstruction/science/assembly/refinement/reductions.py), [science/assembly/refinement/fit.py](../../haplotype_reconstruction/science/assembly/refinement/fit.py).

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
## rs05 Hybrid window search

The surrogate H sums per-sample Viterbi maxima under the fixed Potts switch penalty. First try the relaxed box certificate. Otherwise run the exact frontier programme, retaining at most `HYBRID_FRONTIER` (64) partial paths, with its ordered children, bound pruning, dominance tests and canonical restricted programmes. A completed programme supplies its answer: retain an incumbent within tolerance, otherwise take the first qualifying path in incumbent-first content order.

If that frontier is exceeded, use seeded search. Each relevant sample supplies its preferred whole-unit row path; score each distinct wish and all its contiguous restrictions to the incumbent. Ascend from the incumbent, the best six whole wishes and the best six distinct restrictions. Each ascent takes the best improving single-unit move, with canonical content ties, for at most 32 iterations. Samples whose relaxed upper bound cannot exceed their background-only score within the relative arithmetic margin contribute a constant and are omitted from the search. Rescore the chosen path with the ordinary H recurrence; that score alone determines whether it is an improving offer. Site-level F scoring and the acceptance guards decide acceptance. The seeded answer is a single-unit local optimum when ascent converges; the iteration limit can stop it before stationarity. Exact rescoring certifies the returned offer's score, not global optimality.

Implementation: [science/assembly/refinement/parameters.py](../../haplotype_reconstruction/science/assembly/refinement/parameters.py), [science/assembly/refinement/proposals.py](../../haplotype_reconstruction/science/assembly/refinement/proposals.py), [science/assembly/refinement/seeded_search.py](../../haplotype_reconstruction/science/assembly/refinement/seeded_search.py).

Parameters by name: `HYBRID_FRONTIER`, `SEED_STARTS`, `INTERVAL_SEED_STARTS`, `LOCAL_ITERATIONS`, `RELEVANCE_MARGIN`, `CANONICAL_TOL`, `DOMINANCE_TOLERANCE`, `BOX_ARITHMETIC_MARGIN`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: The exact attempt has a fixed frontier. Seeded background preparation costs O(N B² Q), where B is total window bins and Q=K(K+1)/2. Wishes number at most N; ascent starts and iterations have fixed limits. Every kernel states its loop-domain complexity.

<a id="rs06"></a>
## rs06 Shared messages and bounded search storage

**Accretion:** Shared max-plus messages reuse unchanged path segments, and storage admission follows the actual problem dimensions and fixed frontier.

After bin t the background-eliminated state consists of K focal values and the maxima of earlier bins. A fixed suffix is max-plus linear in this state, with covector `(beta, gamma, const)`: its value is `max(const, max_c focal[c]+beta[c], max_j maxima[j]+gamma[j])`. One forward and one backward pass of the incumbent produce every unit-boundary state and suffix covector. Advancing each seed from each starting unit scores all its restrictions; advancing each candidate row only across its unit scores all single-unit moves. These scores equal individual path scores, including units containing several bins.

Storage follows the actual sample, founder, row, unit and bin counts. The exact attempt uses 64 kept paths; admission reserves its frontier, gathered emissions, background runs, restrictions and boundary covectors. Live memory may delay execution but never selects a different route or order.

Implementation: [science/assembly/refinement/costs.py](../../haplotype_reconstruction/science/assembly/refinement/costs.py), [science/assembly/refinement/parameters.py](../../haplotype_reconstruction/science/assembly/refinement/parameters.py), [science/assembly/refinement/proposals.py](../../haplotype_reconstruction/science/assembly/refinement/proposals.py), [science/assembly/refinement/seeded_search.py](../../haplotype_reconstruction/science/assembly/refinement/seeded_search.py).

Parameters by name: `HYBRID_FRONTIER`, `SEED_STARTS`, `INTERVAL_SEED_STARTS`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N (B R K+B Q+B²)+N W²+W(B+K)) seeded storage for W units and largest alphabet R, plus the fixed frontier and window-shared tables. Storage admission changes scheduling only.

<a id="rs07"></a>
## rs07 Per-template window work account and whole-component stop

**Accretion:** A deterministic work account bounds each template and propagates exhaustion to the component.

Allocate one work account per template, scaled from all evidence rows, window dimensions and founder/alphabet size. Debit the box, exact programme and canonical fixing in their declared order. Before seeded allocation, precharge its bounded preparation, wishes, restriction scores, ranking, ascent and ordinary rescore domains. Exhaustion discards every focal result of the template and propagates the component stop. The ledger records deterministic logical-prefix work; physical work performed concurrently is separate. Focal chunks run through the engine's thread map, and merge into the one-call account with window-shared debits counted once. If that account would be exhausted, one call supplies the deterministic stop receipt.

Implementation: [science/assembly/refinement/parameters.py](../../haplotype_reconstruction/science/assembly/refinement/parameters.py), [science/assembly/refinement/proposals.py](../../haplotype_reconstruction/science/assembly/refinement/proposals.py), [science/assembly/refinement/work_limit.py](../../haplotype_reconstruction/science/assembly/refinement/work_limit.py).

Parameters by name: `window_work_coefficient`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: The charged allowance is window_work_coefficient times N W b² max(K,R)³ per template, for largest unit bin count b. It bounds admitted work, not the number of templates, sweeps or fits.

<a id="rs08"></a>
## rs08 Gauss-Seidel acceptance pass (junction walk, exchanges then content, caps)

Walk junctions with an exact carried forward message. Apply improving suffix exchanges before content moves, respecting the exchange cap and relabelling messages/proposal founders after each exchange. In full content rounds rank all improving active offers and accept the first guard-passing item, then recompute the ranking. Later rounds rescore untried items in their last full ranking. Update changed spans and advance the boundary. An optimistic pass aborts before reading a coarse-move guard and is discarded.

Implementation: [science/assembly/refinement/acceptance.py](../../haplotype_reconstruction/science/assembly/refinement/acceptance.py), [science/assembly/refinement/gains.py](../../haplotype_reconstruction/science/assembly/refinement/gains.py).

Parameters by name: `EXCHANGE_ROUNDS`, `CONTENT_FULL_ROUNDS`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(B N K^3); O(N l K^2); O(B). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rs09"></a>
## rs09 Acceptance tie rules (tie classes of exact gains, content order, convention)

Filter gains strictly above the acceptance threshold, then form chained gain classes. A wide class orders unequal gains by value; exact equalities still use content. Exchange keys compare the resulting pair of paths; content keys compare resulting path, founder slot and proposal index in pass order. Reverse only declared content conventions. Identical-content, identical-gain exchanges use the smallest enumerated pair under both conventions.


Implementation: [science/assembly/orders.py](../../haplotype_reconstruction/science/assembly/orders.py), [science/assembly/refinement/gains.py](../../haplotype_reconstruction/science/assembly/refinement/gains.py), [science/assembly/refinement/acceptance.py](../../haplotype_reconstruction/science/assembly/refinement/acceptance.py).

Parameters by name: `WIDE_CLASS_QUANTA`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

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

Parameters by name: `sweep_limit`, `WINDOW_STORE_TEMPLATE_BYTES`, `tolerance`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L K^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rs12"></a>
## rs12 Best-of-two fit selection, minimum-switch prior and canonical fit

Reuse fits only for their full start identity. Run both direction starts, except the single-direction polish route, and compare their F values in the named tie policy. When distinct panels share the winning class, minimise switch count within that class and retain the other tied variant. Canonicalise founder rows after choosing. Branch replay can explicitly force a retained tied panel; it cannot invent an unrecorded alternative. The named tie policy receives the run's reversal convention explicitly.


Implementation: [contracts/assembly.py](../../haplotype_reconstruction/contracts/assembly.py), [science/assembly/orders.py](../../haplotype_reconstruction/science/assembly/orders.py), [science/assembly/refinement/__init__.py](../../haplotype_reconstruction/science/assembly/refinement/__init__.py), [science/assembly/refinement/fit.py](../../haplotype_reconstruction/science/assembly/refinement/fit.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N L Q). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rs14"></a>
## rs14 Polish schedule at ordinary fixed points (escape, long repair, bridge; verified)

At a polish fixed point try beam escape, then long repair if nothing was accepted, then a bridge if still unchanged and fine offers exist. Bridges use the beam escape’s fine offers. Verify each accepted family by full rescoring and its named exactness tolerances, requiring positive improvement. Polish never changes founder count and never reruns the count decision; a work stop keeps the unpolished count-pass panel.

Implementation: [science/assembly/refinement/fit.py](../../haplotype_reconstruction/science/assembly/refinement/fit.py).

Parameters by name: `polish`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: One single-direction fit per polished start; each ordinary fixed point can add escape, long-repair and bridge searches.

The search owns one last-problem entry per beam template or gain/bridge window. Entries are
replaced when their keyed inputs change, discarded at fit end and rebuilt after interruption;
no store enters a checkpoint. A slot miss consults the answers the component's polish fits
share under the same keys ([RC22](#rc22)); count refits never use them. Fixed
ownership binds evidence and sample order, masks, prepared-row coding, emissions, map costs,
coarse context, K and search controls. Keys compare numerical array dtype, shape and content
using content digests. The dimension-only residence bound, including the union of possible tied
donors, is stated in `PolishStores` in
[polish_cache.py](../../haplotype_reconstruction/science/assembly/refinement/polish_cache.py)
and included in fit and component memory admission. Store size does not grow with sweep count.
Only physical polish work is avoided; proposal order, accepted moves and exact scores are preserved.


<a id="rs15"></a>
## rs15 Window-beam proposer and beam escape

Canonicalise founder order, construct the map-aware binned surrogate and apply the box gate. Rank beam children by anchored quantised bounds, retain the bounded cut cohort, remove dominated children, and restore flat child order. Retain only a changed final path clearing the beam gain rule. Apply one guarded acceptance pass over the offers and verify its exact gain. Beam width and cohort truncation limit the searched paths.

Implementation: [science/assembly/refinement/beam.py](../../haplotype_reconstruction/science/assembly/refinement/beam.py).

A beam entry binds template kind and unit span, width, content convention, window rows and
both surrogate flanks. Order founders by their window rows, breaking duplicate-row ties by
current index, and permute both flanks' unordered-pair states into that single frame. Store all
focal answers in the frame; restore current focal order before collecting proposals or accepting
moves. Duplicate local rows can cause a miss when their framed flanks differ; a hit still binds
the complete framed problem. Candidate alphabets and their content order are reconstructible
from window rows and fixed context. The max-plus recursions take maxima over founder states
and sum along sites/bins and samples in their existing order. Founder renaming permutes state
values without changing arithmetic or path ties, which are over content-ordered candidates.


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

Generate long-window beam offers, choose the incumbent and agreement-ranked donors, then enumerate deduplicated private corrections. Obtain each variant's exact scalar gain from its last-problem window table or score it freshly; filter by the strict acceptance threshold before forming the maximum gain band. Select the content-smallest rows in that band, offer them as coarse moves and verify accepted gains. A rejected chosen variant does not trigger a search through the offer’s discarded variants.

Implementation: [science/assembly/refinement/fit.py](../../haplotype_reconstruction/science/assembly/refinement/fit.py).

Parameters by name: `LONG_WINDOW`, `LONG_STRIDE`, `LONG_CORRECTION`, `LONG_DONORS`, `LONG_BEAM_WIDTH`, `BRIDGE_BATCH_BYTES`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Regenerate variants at every fixed point in current labels: three equally agreeing donors can
have different correction rows, and their index tie determines which two survive. Neither the
variant set nor its selected best member is reused up to founder labels. A scalar-gain table
binds the block span, framed current rows, exact forward flank at the start and exact backward
flank at the end. Each scalar names the mapped focal and actual candidate row bytes; only that
same mapped candidate can hit. The current rows and end flank reconstruct the start backward
flank and block normalisers deterministically, so those redundant inputs need no separate key.
Gain recursions are max-plus over founder states, with sample sums in fixed order. All retained
gains use the forward orientation, even after a reverse acceptance sweep; reverse-oriented gains
are excluded because their arithmetic can differ in the last bits.

Cost and scope: O(B K / LONG_STRIDE); O(N · LONG_WINDOW blocks · K²). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rs17"></a>
## rs17 Bridges (fine-window precursor plus suffix exchange)

Choose each fine-window precursor in the maximum gain band, apply it privately, then examine suffix exchanges across its cuts. Choose the exchange within its band before testing the combined gain; a tied representative can therefore leave no offer. Compare surviving bridge offers by band and the declared tuple key, then verify the selected combined gain. Bridges bypass the coarse-move guard.

Implementation: [science/assembly/refinement/fit.py](../../haplotype_reconstruction/science/assembly/refinement/fit.py).

Parameters by name: `BRIDGE_BATCH_BYTES`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Precursor scalar gains may use the framed gain tables. Local bridge trials instead bind plain
current window rows and exact flanks, the selected precursor's focal, rows and scalar gain.
Precursor ties, pair ties and final offer ties read founder indices: a bridge-window result is
therefore never reused up to relabelling. Store only its local cut/pair/gain trials, reconstruct
the current precursor index, and repeat grouping and global selection in their original order.
A whole acceptance pass or converged panel is not a reusable answer under these keys. Ordinary
window proposal caches, initial sweeps and solver tables keep their own execution paths.

Cost and scope: O(#windows x N x 8 blocks x K^2); O(N K^3). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rs18"></a>
## rs18 Weak label junctions (diagnostic)

**Accretion:** Weak junctions are diagnostics and do not select moves.

Report junctions whose best exchange gain is strictly above the negative weak-junction margin. Keep this diagnostic separate from the acceptance threshold and from released panel state. No downstream scientific decision reads the report.

Implementation: [science/assembly/refinement/fit.py](../../haplotype_reconstruction/science/assembly/refinement/fit.py).

Parameters by name: `SWITCH_PENALTY`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(B). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="rs19"></a>
## rs19 Exact reuse inside a fixed-K search (restated windows, relabelled messages, shared sweep 0, standing-offer gains)

**Accretion:** Work a search would repeat on a renamed or unchanged problem is reused exactly; every decision, record and logical work account is the one computing it afresh gives.

The window solver, the block messages and the objective take maxima over pair states and add per-state terms; none breaks a tie by founder index except the seeded search's latent trace ([RS05](#rs05)), which follows the first tied pair state. Renaming founders therefore permutes every state-indexed array bit for bit and leaves every value and every content-ordered choice unchanged, except where that trace meets an exact tie.

Restated window results. A pass's accepted suffix exchanges ([RS08](#rs08)) rename founder paths from their junctions on. After the pass, the search derives, per block, the label each founder's path held when the pass started, and restates each held window problem whose first and last blocks carry one such renaming: its rows, its focal results (moved to the renamed focals, with the template's work receipt on the first focal) and the flank messages it read. A held problem whose ends carry different renamings is dropped; one the renaming maps to itself bit for bit stays as held. The proposal cache ([RS11](#rs11)) accepts a restated problem only when its rows and both flanks equal the new panel's bitwise; the renamed problem is then the problem the search would solve, and its solution is the restated one. The seeded search reports, per focal, whether some sample's trace took an exactly tied maximum that could change its wishes: the latent trace is followed backwards through every optimal resolution of every tie (O(B Q) per tied sample), and the focal is order-sensitive when one such resolution gives another per-unit verdict on whether the sample's path uses a focal state, which is all the wishes read. The order-sensitive focals of a restated problem are solved afresh, and the rest are restated: a focal's result does not depend on which other focals share its solve, and their work debits are at most the full solve's, which stayed within the account. A restated problem is bound and receipted like the solve it replaces ([RS07](#rs07)), so the logical work prefix, any stop record and the solved/reused counts are those of the released search; only physical work is saved. It is also kept in the component's window store ([RC22](#rc22)) as that solve would be, so the store holds the problems the search meets, with their solved results, for every other fit of the component; an order-sensitive completion takes the stored solve instead when the store already holds the problem.

Relabelled messages. The search's held messages are permuted into the new labels before their in-place update ([RC05](#rc05)), in the dense and the checkpointed layouts ([RC07](#rc07)); the update then recomputes the blocks where the panel differs from the renamed held panel and both blocks at every junction where the renaming changes. On every other block the renamed held messages are the max-plus recursion of the renamed inputs, so the update equals the full computation bit for bit for any renaming.

Shared sweep 0. When neither search of a fit resumes from a record, the start panel's messages are computed once (the reverse-first search takes a private copy) and so is every sweep-0 window solve up to the point where the content convention first enters: the window tables, the box certificate and the frontier programme of every focal, which order children by value and content whatever the alphabet's listing order. Each search then finishes every template in its own convention (canonical fixing, or the seeded search on frontier overflow). A template the component's window store already holds in both conventions gets no shared stage; its searches take the stored solves. The work account continues across the two stages, so a template exhausts it exactly when one call would; an exhausted template is solved again in one call for its stop receipt (phase and units), as exhausted focal chunks are. Polish fits run one search and share nothing.

Standing-offer gains. An acceptance pass reuses the exact per-sample gains of a content item whose founder, block range, rows, panel rows on that range and messages at its ends (forward at its start, backward at both ends, backward normalisers) equal those of an earlier evaluation in the same search and pass orientation; the gains are a pure function of these inputs. The item is keyed by two independent 64-bit hashes and each sample's messages by a 64-bit digest, all of which must match: a stale hit needs a 128-bit item collision or, for the same item, a 64-bit collision of every sample whose messages changed, below 2⁻⁶⁴ per lookup. The two pass orientations keep separate tables, since their gains differ in the last bits, and the lexicographic rescoring is never reused. Tables hold `GAIN_CACHE_TEMPLATE_SLOTS` entries per window template (at least `GAIN_CACHE_MIN_SLOTS`, a power of two), eight probes per item, least recently used first; an evicted entry is recomputed.

A resumed search starts with no held messages, results or gains. None of this state crosses a fit or a process. A reference path (`SweepEngine.verify_reuse`, for tests) computes each sweep 0 per search and every reused value afresh, compares them and keeps the fresh value.

Implementation: [science/assembly/refinement/cache.py](../../haplotype_reconstruction/science/assembly/refinement/cache.py), [science/assembly/refinement/fit.py](../../haplotype_reconstruction/science/assembly/refinement/fit.py), [science/assembly/refinement/acceptance.py](../../haplotype_reconstruction/science/assembly/refinement/acceptance.py), [science/assembly/refinement/proposals.py](../../haplotype_reconstruction/science/assembly/refinement/proposals.py), [science/assembly/refinement/seeded_search.py](../../haplotype_reconstruction/science/assembly/refinement/seeded_search.py).

Parameters by name: `GAIN_CACHE_TEMPLATE_SLOTS`, `GAIN_CACHE_MIN_SLOTS`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: Restating a held problem costs its rows and two flanks (O(W K + N Q)); the order code costs O(B Q) per sample with a used tie. Sharing sweep 0 holds one more message set and every template's stage-1 state (K (W + 3) integers) until both searches have finished their sweep 0. The gain tables take 2 (24 + 16 N) bytes per slot per search, declared in the fit's memory estimate and reserved per CPU with the component ([RC20](#rc20)); a lookup hashes the item's rows and the panel on its range and digests 3 Q values per sample.

### Parallel evaluation with ordered selection

Independent count refits retain their logical fit ordinals and are evaluated on
child CPU shares before the original ordered count selection. A sweep's window
solves run one template per thread; a template whose estimated work exceeds an
even share of the whole map on the search's threads would set the map's critical
path on its own thread, so it is split into focal chunks (at most its
`focal_chunks` count), which run on its child share as that share grows and merge
into the one-call result and work account. A junction's exchange gains open one
parallel region on the pass's team whenever their work (samples x K^3) reaches
`EXCHANGE_PARALLEL_WORK` (refinement gains module); the gains are summed in sample order either
way. Dominance tests of the frontier programme read the background helpers' rows
in place (`proposals._background_at`): forming a row view per sample and background
state costs an atomic reference-count update around a few operations, which on some
processors (Sapphire Rapids) took half of a coarse window's solve. The polish beam
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
