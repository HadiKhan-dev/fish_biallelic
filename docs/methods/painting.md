# Painting and label continuity

Painting uses a diploid copying chain over founder trajectory classes with a background law and panel-error fidelity states. Run calibration selects the outlier mixture and evidence divisor. Named track release, local allele equivalence and switch intervals express different claims. N is samples, L sites, B bins and S track states. Label continuity precedes the final decode and never reads pedigree truth.

The sections below identify semantic units, not separately configurable pipeline steps. [Outputs](../user/outputs.md), [record contracts](../reference/records.md) and [costs](../developer/costs.md) describe their public boundaries.

<a id="pr01"></a>
## pr01 State alphabet: selected sites, trajectory classes, BACKGROUND law, anchoring, public ids

Form trajectory-equivalent row classes on selected sites, preserve first-appearance class order, and introduce the declared background state. Determine which classes are anchored and error-capable before constructing public state IDs. Class IDs are local indices, not founder identities across components.

Implementation: [science/painting/states.py](../../haplotype_reconstruction/science/painting/states.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(K L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pr02"></a>
## pr02 Painting bins: layout, centres, bp edges

First split selected sites into max(1, L // reference_snps_per_bin) near-equal reference windows. Split each full reference window into reference_snps_per_bin // snps_per_bin near-equal fine bins; a component shorter than one reference target uses max(1, L // snps_per_bin) fine bins. Consecutive fine-bin groups therefore exactly match the reference decode, including unequal site counts. Compute centres and half-open bp edges from the declared coordinates. Binning is deterministic and does not follow task windows or sample batches.

Implementation: [science/painting/states.py](../../haplotype_reconstruction/science/painting/states.py).

Parameters by name: `snps_per_bin`, `reference_snps_per_bin`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(L_s). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pr03"></a>
## pr03 Local allele-equivalence keys and multi-member states

Partition trajectory classes by local allele equivalence within each bin and assign the smallest class index as the equivalence key. Preserve multi-member states for release. Local equality does not establish chromosome-wide row identity.

Implementation: [science/painting/states.py](../../haplotype_reconstruction/science/painting/states.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(C L_s); O(B S). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pr04"></a>
## pr04 Tempered binned emissions (robust site emission, informative sites and eligibility, design effect, windowed construction, packing)

For each sample/pair/bin, sum robust centred site logs in site order and divide by 1 + max(n_w - 1, 0) r, where n_w is that sample’s informative-site count over the whole reference window. A reference window consists of g = reference_snps_per_bin / snps_per_bin consecutive fine bins, starting at the component’s first bin; the final group may be shorter. The fine target must divide the reference target. Whole-reference storage windows and sample tiles preserve the values, including label-continuity parts of bins. Defaults are 25 selected sites per fine bin and 100 per reference window. Informative-site counts accumulate in int64. NumPy’s internal int64 input conversion belongs to the sample/window workspace and is bounded by that window. The run-calibrated divisor d gives r = (d - 1) / (reference_snps_per_bin - 1).

Implementation: [science/painting/emissions.py](../../haplotype_reconstruction/science/painting/emissions.py), [science/painting/hmm.py](../../haplotype_reconstruction/science/painting/hmm.py), [stages/painting/tasks.py](../../haplotype_reconstruction/stages/painting/tasks.py).

Parameters by name: `robustness_grid`, `reference_snps_per_bin`, `emission_log_floor`, `divisor_grid`, `calibration_folds`, `INFORMATIVE_SPREAD`, `WINDOW_BYTES`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L_s S(S + 1)/2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pr05"></a>
## pr05 Label-level copying process (painting/transitions.py)

Construct the label copying process from map distance, ancestry-change rate and the declared stationary label law. Use the same transition expression for fitting and decoding. A zero-length interval and a long interval retain their explicit boundary behaviour.

Implementation: [science/painting/transitions.py](../../haplotype_reconstruction/science/painting/transitions.py).

Parameters by name: `MINIMUM_SWITCH_PROBABILITY`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(B). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pr06"></a>
## pr06 Class x fidelity copying chain (track states, transitions, initial law, structured emissions)

Augment each copying label with its permitted faithful/panel-error states. Label changes follow the copying process; within-label error entry/exit follows the fidelity chain. Combine two ordered tracks into the diploid state and initial law. Error states describe panel mismatch, not extra inferred founders.

Implementation: [science/painting/fidelity.py](../../haplotype_reconstruction/science/painting/fidelity.py).

Parameters by name: `ENTRY_BOUNDS`, `EXIT_BOUNDS`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(n B S'^2); O(K^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pr07"></a>
## pr07 Exact scaled forward-backward of the chain (expected statistics, label-pair posterior masses)

Run scaled forward/backward recurrences in fixed within-sample order, computing expected transition statistics and label-pair posterior masses. Scale every step consistently. Samples may run independently, but reductions across samples use the declared estimator order.

Implementation: [science/painting/fidelity.py](../../haplotype_reconstruction/science/painting/fidelity.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(n B S'^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pr08"></a>
## pr08 Copying-model estimator: fixed EM updates of (rho, a, b), M-step searches, named log-grid rounding; estimation batches

Start from the named copying/error parameters and execute the fixed number of EM updates. Each E-step supplies both the rate and error updates; use the fixed grid/refinement and bounded searches with their stated tie preferences. Round final estimates to the named natural-log grid. Sample reductions follow sample order independently of memory batches. Run-level calibration, described below, supplies the outlier mixture and divisor before each component's final fit.


Implementation: [science/painting/fidelity.py](../../haplotype_reconstruction/science/painting/fidelity.py), [stages/painting/tasks.py](../../haplotype_reconstruction/stages/painting/tasks.py), [science/painting/run.py](../../haplotype_reconstruction/science/painting/run.py).

Parameters by name: `initial_switch_rate`, `initial_error_entry`, `initial_error_exit`, `estimation_iterations`, `estimate_grid`, `RATE_BOUNDS`, `ENTRY_BOUNDS`, `EXIT_BOUNDS`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L_s S^2/2); O(N B S'^2); O(B). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pr09"></a>
## pr09 Chain Viterbi with the stated tie rule under both traceback conventions

Decode both conventions with the structured two-track Viterbi recurrence. Apply the same-label/fidelity and label-change candidate rules before centring scores. Ties use the declared Viterbi policy and traceback convention; the decoder does not replace them with a generic argmax.

Implementation: [science/painting/fidelity.py](../../haplotype_reconstruction/science/painting/fidelity.py).

Parameters by name: `VITERBI_TIE`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(n B S'^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pr10"></a>
## pr10 Parsimonious track order of decoded paths and painted switch counts

Walk decoded track pairs left to right and choose the parsimonious ordering by homologue changes, then visible changes under the fixed tie rule. Accumulate the gauge swap and count biological label switches. Pure track swaps are gauge and must not inflate switch counts.

Implementation: [science/painting/hmm.py](../../haplotype_reconstruction/science/painting/hmm.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(n B). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pr11"></a>
## pr11 Switch intervals, combined candidate masks of both conventions, canonical track order

Combine candidate masks from both traceback conventions, identify switch intervals, and assign a canonical track order from unordered masks. The smaller mask supplies the declared gauge representative. Interval ambiguity is preserved, not converted to a confident single-row track.

Implementation: [science/painting/fidelity.py](../../haplotype_reconstruction/science/painting/fidelity.py).

Parameters by name: `MAXIMUM_MASK_STATES`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(n L_s); O(n B); O(n B S^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pr12"></a>
## pr12 Release posteriors qF and qAF (min over the two conventions' Viterbi pairs)

Compute the faithful Viterbi-pair posterior qF and its local allele-equivalence posterior qAF, and take the conservative value over the two conventions. These values feed staged release; they do not independently create a new acceptance threshold.

Implementation: [science/painting/fidelity.py](../../haplotype_reconstruction/science/painting/fidelity.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(n B S'^2); O(n B S^2 S). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pr13"></a>
## pr13 Release rule: staged gate, statuses, member sets, direct support, released switch counts

Apply the faithful unordered-pair gate first, then the locally allele-equivalent direct-support gate. Assign local, interval, low-posterior and ineligible statuses in the declared precedence. Use inclusive continuous posterior thresholds without a tie band. An interval cannot release a bin that failed the staged rule.

Implementation: [science/painting/release.py](../../haplotype_reconstruction/science/painting/release.py), [stages/painting/tasks.py](../../haplotype_reconstruction/stages/painting/tasks.py), [science/painting/run.py](../../haplotype_reconstruction/science/painting/run.py).

Parameters by name: `minimum_release_posterior`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N B); O(N B + B C). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pr14"></a>
## pr14 Decoder execution: memory-granted sample batches and the packed-emission cache

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Choose resident packed emissions or whole-bin windows using the supplied storage capability and dimension-based workspace plan. When the evidence offers resident copies, each component's evidence at its panel sites (codes and likelihood dictionaries) is read once into memory under the emission budget, whenever the estimate and the decode keep the packed emissions they would keep without it, so that every emission build of the component is an in-memory parallel gather. The copying-model estimate (pr08) reads the same tempered emissions in each of its updates, since they do not depend on (rho, a, b). When their packed form fits the remaining cache budget, they are built once and unpacked for each update. The values are the same: the ordered-pair emissions are written symmetrically. The estimate releases them before label continuity. Label continuity's part emissions are kept for the whole relabelling search under the same budget, because a candidate relabelling only remaps states; each candidate check then re-decodes without rebuilding emissions. When every sample batch's part emissions do not fit, the batches that fit are kept, and the component's selected evidence (each sample's genotype-likelihood codes at the selected sites and its likelihood dictionary, far smaller than the emissions) is read once into memory under the same budget (the leading batches that fit, when the whole copy does not), so that rebuilding a batch is a parallel in-memory gather and emission kernel; one rebuild serves several candidates, the next ones in ranked order being decoded from it after a candidate fails (1, 2, 4, ... per iteration, reserving allowance for decoded results still waiting), and they count as checks only when the search reaches them. The search, its checks and its outcome are the same. Progress records make an interrupted painting resumable: each component's copying model, its label-continuity outcome and every decoded sample batch are published as soon as they are final, under the painting's derivation, and a resumed painting reads them instead of recomputing (a restored decode leaves the packed-emission cache out, which later stages rebuild). Decode each sample with identical kernels and site order. Cache presence and batch size affect work only; packed arrays are reconstructible from the recorded model/evidence. Release-only decoding arrays end at the component release boundary; the next component retains only the published product and its original cache entry.

Implementation: [stages/painting/tasks.py](../../haplotype_reconstruction/stages/painting/tasks.py), [stages/painting/compute.py](../../haplotype_reconstruction/stages/painting/compute.py), [science/painting/run.py](../../haplotype_reconstruction/science/painting/run.py).

Parameters by name: `EMISSION_CACHE_MAX_BYTES`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L_s S^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pr15"></a>
## pr15 Label continuity set-up: link points, movable classes, bin parts, capability table, sample batches

Construct eligible link points, movable classes, split-bin emissions and the transposition capability table. Split-bin parts retain the whole-bin design-effect denominator. Process sample batches in index order; caching a complete batch is a performance choice, not an admission decision.

Implementation: [science/painting/label_continuity.py](../../haplotype_reconstruction/science/painting/label_continuity.py).

Parameters: supplied typed inputs and the painting parameter record; batching uses the supplied execution capability. This unit has no independent byte allowance.

Cost and scope: Per-sample label-continuity memory is O(S²(B + links)) plus switch-count/message/scoring scratch; batching bounds the simultaneous sample set.

<a id="pr16"></a>
## pr16 Label continuity screen: max-product link messages and exact transposition gains (label-level process)

Use left/right max-product messages to score exact label transpositions at each link. Separate unaffected states from affected rows/columns, then sum per-sample gains in canonical order. The screen ranks repair candidates but makes no released ancestry call.

Implementation: [science/painting/label_continuity.py](../../haplotype_reconstruction/science/painting/label_continuity.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N B S^2); O(links N (S^2 log S + C^2 S)); O(links N S^3). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pr17"></a>
## pr17 Label continuity decision: ranking in tie classes, leftmost placement, painted-switch acceptance, competing repairs, iteration

Admit candidates meeting the named switch-gain gate, rank gain tie classes, and move equivalent repairs to their leftmost placement. Test actual painted switch gains and the loser/winner criterion before acceptance. Group the ranked tie peers once per iteration without changing their order or ambiguity rule. Ranked candidates and tie peers share `label_continuity_check_budget` charged checks per iteration. Capability-changing peers are rejected without a decode or charge, even when that budget is spent; cached verdicts also need no charge. An acceptable peer makes its class ambiguous. If a distinct, unchecked, capability-preserving peer remains when the budget is spent, withhold the undecidable repair, end the search, flag `check_budget_reached` (ContinuityBudgetReached in the painting model table), and record `painting.relabel_class_budget`. This conservative outcome can withhold a correct repair whose class needs more checks than the budget. Recompute scores and candidate checks after each accepted repair until none qualifies; there is no fixed iteration cap.

Implementation: [science/painting/label_continuity.py](../../haplotype_reconstruction/science/painting/label_continuity.py), [core/ties.py](../../haplotype_reconstruction/core/ties.py).

Parameters by name: `label_continuity_minimum_switch_gain`, `label_continuity_maximum_loser_ratio`, `label_continuity_check_budget`, `WIDE_CLASS_QUANTA`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: With J links, Q <= J K(K-1)/2 candidates per iteration. Tie-peer lookup is O(Q) after ranking. For a check budget b, charged checks are at most b, including peers. Decoded look-ahead results count as checks only when consumed; reserving their allowance bounds waiting trials at b - 1. Physical trial decodes are therefore at most 2b - 1 for b >= 1 and zero for b = 0. This excludes the baseline decode, winner-direction diagnostics and final verification. Each trial costs O(N (B + J) K^2), so trial work per iteration is O(b N (B + J) K^2), in addition to the cubic gain screen and ranking. Accepted repairs strictly decrease the integer switch total; their number is bounded by that decrease, not by a founder-sized cap.

<a id="pr18"></a>
## pr18 Label continuity verification, row maps, outcome record, winner directions

Verify the accepted row maps preserve class/capability structure and strictly improve the required painted-switch total. Record winner directions and identity status. A repair whose verification fails cannot be accepted solely because its screen looked favourable.

Implementation: [science/painting/label_continuity.py](../../haplotype_reconstruction/science/painting/label_continuity.py), [contracts/panels.py](../../haplotype_reconstruction/contracts/panels.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N B S'^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pr19"></a>
## pr19 Painting stage order, product, records and views

The stage orchestrates model fitting, label continuity, decoding and release, in that order. It owns checkpoint progress and cache budgets; science computes models and batch results. Publish typed painting and model diagnostics with declared source identities. When the fine and reference targets differ, publish a second painting at reference_snps_per_bin under painting/reference, bound to the fine painting’s derivation. Apply exactly the fine painting’s label-continuity row maps, then fit the reference resolution’s own copying chain at the same estimated genotype-outlier mass and decode it. Pedigree preparation, map-scale estimation and projections read this reference painting; its per-bin rules retain their reference units. When the targets coincide, both readers use the fine product with no extra decode or publication. Founder refit, family phase, recombination, exports and evaluation use the fine painting; refit does not feed back into pedigree scoring.

Implementation: [contracts/painting.py](../../haplotype_reconstruction/contracts/painting.py), [science/painting/diagnostics.py](../../haplotype_reconstruction/science/painting/diagnostics.py), [stages/painting/tasks.py](../../haplotype_reconstruction/stages/painting/tasks.py), [science/painting/run.py](../../haplotype_reconstruction/science/painting/run.py), [stages/painting/records.py](../../haplotype_reconstruction/stages/painting/records.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Sum of the component estimator/continuity/decode costs, with contig-affine state and validated evidence routes.

<a id="pr20"></a>
## pr20 Pairwise founder sharing (optional report)

For every pair of samples, count the founder rows the two samples' released tracks share, bin by bin: where all four tracks are `SINGLETON_NAMED` ([PR13](#pr13)), the count is the multiset intersection of the two samples' row pairs (0, 1 or 2; a sample whose two tracks name one row shares both copies with another such sample). Rows are component-local and both samples of a pair share the component's bins and panel, so the count needs no row identity across components or label-continuity links. Every other bin of the pair is classified by its worst track, in this order: a track abstains (no evidence or low posterior); a track is unidentified (background, or a class with no called allele); a track names candidate rows (pooled or local allele equivalence, or a switch interval inside the bin: rows identical by state, alias-uncertain). Each bin weighs its length on the working map, M(edge b+1) − M(edge b) in Morgans ([PR02](#pr02) edges; fallback rate off the map), because the identity-by-descent state of a pair changes at a constant rate per Morgan: equal Morgans carry equal information about the relationship, and physical length would overweight low-recombination stretches. With a constant-rate map the shares equal physical-length shares. Sums over components and contigs give genome-wide (included contigs) and per-contig IBD0, IBD1 and IBD2 shares of the jointly named length, with the shares of the painted length that each reason leaves uncounted.

These are identity-by-descent shares relative to the panel's founder rows. Founders that assembly cannot separate are one row, and two samples carrying such a row count as sharing it. Counting only jointly named bins conditions on all four tracks naming a row. A shared founder copy is one haplotype, so where founders that are never named (pooled or background) are common, the named subset slightly over-represents sharing; the coverage columns state how much of the painted length the shares rest on. The report reads published paintings and feeds nothing back; it has no checkpoint record and changes no other product. `[painting] pair_sharing` selects it (off by default; its tables have N(N − 1)/2 rows per scope).

Implementation: [contracts/pair_sharing.py](../../haplotype_reconstruction/contracts/pair_sharing.py), [outputs/reports/pair_sharing.py](../../haplotype_reconstruction/outputs/reports/pair_sharing.py), [results/pair_sharing.py](../../haplotype_reconstruction/results/pair_sharing.py).

Parameters by name: `pair_sharing`. Values, units and current rationale are in the [generated settings reference](../reference/settings.md).

Cost and scope: O(N² B) per contig in NumPy row passes over pairs, O(P + N B) memory with P = N(N − 1)/2 (no N² × B array); output N(N − 1)/2 rows per table and contig.

## Run-level painting calibration

After the final assembly of every analysed contig is published, reference-resolution
calibration fits each component for each `robustness_grid` value with r = 0.
The first run reduction sums the last recorded trajectory log likelihood across
components and contigs, selecting the largest sum, with ties to the smaller value.
The trajectory records the likelihood before its final EM update, as in the copying
estimator. No emission cache crosses candidate values.

With this outlier rate fixed, each contig evaluates every `divisor_grid` value and
`calibration_folds` fold. Fold f hides the f-th of `calibration_folds` contiguous near-equal
parts of every reference window: those selected sites are unobserved for every sample during
fitting and label continuity, and every window keeps the rest of its own sites. The evidence
divisor corrects for dependence among the sites within a window, so it is tested where that
dependence acts: by predicting part of a window from the rest of it (hiding whole windows
scores a window predicted from its neighbours only, and prefers no tempering at all). Component preparation, continuity,
and decoding-grid construction otherwise follow painting. Forward-backward then
returns every ordered emission-label pair's posterior mass, mapping each error track
independently to BACKGROUND. All originally observed hidden sample-site cells contribute
log sum_ij m_ij [(1 - eps) sum_g G(g | p_i, p_j) L_g + eps/3], where L is normalized,
p is the called allele or background frequency, and G is the independent-allele genotype
law. This is the confidence objective: the held-out log posterior predictive density.
It uses neither a hard decode nor a release threshold or an eligibility filter.

The same fits also give the path objective. The hidden-window component is decoded as
painting decodes it (the label pair of each bin, before any release gate), and every
originally observed hidden cell contributes
log [(1 - eps) sum_g G(g | p_a, p_b) L_g + eps/3] for the decoded pair (a, b) of its bin:
the model's own site emission at the decoded labels.

The final reduction pools both objectives over all analysed contigs and folds and chooses
two evidence strengths from the one grid, with ties to the smaller divisor in both: the
PATH divisor has the largest path objective, the CONFIDENCE divisor the largest predictive
density. The two answer different questions. A weak divisor lets the decoded path follow
the data closely (sharp switches); a strong divisor keeps the chain's probabilities honest
when panel rows carry errors and sites of a window do not err independently. Both
objectives, cell counts, estimates and per-contig durations are published in
`painting/calibration` and its report (`calibration_divisor.csv`: `log_predictive_density`,
`path_score`, `selected`, `selected_path`).

Painting then fits the copying model twice per component when the two divisors differ. The
path fit (emissions tempered by the path divisor) serves label continuity, decoding,
candidate masks, switch counts and direct support. The confidence fit (the same estimator
on emissions tempered by the confidence divisor) serves the release posteriors qF and qAF
of the decoded labels. Confidence emissions are the path emissions times
(1 + (n_w - 1) r_path) / (1 + (n_w - 1) r_confidence) per sample and reference window: the
site floor is applied before tempering, so no second pass over the evidence is needed. The
published `PaintingTransitions.site_correlation` and the packed emission cache are at the
confidence strength, which family phase reads. The pedigree (preparation and map-scale frames)
reads the painting at the path strength (`science.pedigree.prepare.path_correlation`): its
questions are which painted labels a sample shares with a candidate relative, questions about
the decoded path, and at the confidence strength (up to 128 on crosses without sequenced
parents) the evidence that tells a full sib from a parent is divided away. It uses the packed
cache only when the two strengths are equal and otherwise rebuilds the emissions.
`Painting.emission_divisor` records the confidence divisor and `Painting.path_divisor` the
path divisor. With equal divisors there is one fit and one set of emissions.

Single-value grids skip their corresponding calibration fits. The setting
robustness_grid=(0.01,), divisor_grid=(10.9,), and both bin sizes 100 gives r=0.1 for both
strengths and reproduces the 4.19 painting.
Every fine and reference painting depends on the final run calibration identity and
uses the same estimates; the reference painting has its own copying-chain fit after
the fine painting's label-continuity outcome. Pedigree consumes reference paintings;
founder refit, phase, recombination, export and evaluation consume fine paintings.
