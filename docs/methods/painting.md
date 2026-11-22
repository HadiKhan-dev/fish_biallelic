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

The read-state search keeps one history per ordered track-pair state, with its log score and normalized
read-class vector $v$. For each destination it ranks the retained predecessors by their prefix score plus
the copying transition and $\log(v M \mathbf{1})$, where $M$ is the destination bin's read matrix for that
pair. The destination's scalar emission and matrix log scale are common to those contenders. The reverse
convention uses transposed copying transitions and read matrices. Its last contraction weights the original
first bin by the first-site class law, and the track initial law also enters at that bin. Copying bounds and
the largest coefficient of the contraction skip contenders outside the tie band. The search is exact over the
current one-history candidates when it finishes within $(C+1)S'$ inspected predecessors per destination. This
is the number of class-specific leaders plus a scalar leader for each second source track. When the exact
search needs more work, keep its visited candidates and all predecessors sharing either founder label with
the destination, subject to the same safe pruning. For each second source track, also consider the
scalar copying leader under the usual tie preference and the first source track maximizing each of the
$C$ prefix class masses after copying. Also consider the scalar copying leader along each first source track.
Rank their union by the original full marginal score and apply the predecessor preference within the tie band.
The shortlist can omit a source that changes both founder labels. The highest score before tie selection
corresponds to at least $1/C$ of the unrestricted peak mass, including the prefix and copying weights.
This compares the same one-history input for a single step. The complete traceback has no such bound.

Implementation: [science/painting/fidelity.py](../../haplotype_reconstruction/science/painting/fidelity.py) and [science/painting/read_chain.py](../../haplotype_reconstruction/science/painting/read_chain.py).

Parameters by name: `VITERBI_TIE`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: The scalar path search is O(n B S′²). The read-state search contracts each destination's C-by-C
matrix in O(n B S′² C²). It computes each needed column of class leaders once per sample and bin,
for O(C S′²) preprocessing work. At each of O(S′²) destinations the bounded scan, shared-label pairs and class leaders
require O(C S′) candidate visits. The O(C) marginal score therefore bounds traceback work by
O(n B C² S′³), cubic in the founder count for fixed class width. Backpointers and active class vectors
use O(B S′² + S′² C) space per sample apart from the input matrices.

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

With the read state enabled, the screen and provisional switch counts use scalar bin-part emissions. The
final verification decodes the proposed row order with the full read-state builder and compares its switches
with the scalar baseline count. The candidate check budget below bounds the search. In paired replays across
famhs, longbc, seed, bc8, std and TroMau (57 contigs), screening with path-strength read matrices and checking
both panels with the read-state decoder produced identical ordered row maps, genotype calls and painting
switch counts. The read-aware arm's selected held-out log score on TroMau was 526 nats lower at the same
divisor.

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
calibration fits each component for each `robustness_grid` value with r = 0 (independent one-CPU tasks).
Each fit's score is its log-likelihood at the returned (rounded) values, one forward-backward pass after its last
update: the support curve of the eight-update fit (4.21, ADVICE2 14.1 item 6). Once every contig's grid fits exist,
each contig also fits the two rates at the geometric midpoints around the pooled grid maximum (a refinement step of
its own in the graph). The run reduction sums the scores across components and contigs over the grid and its
midpoints and selects the largest sum, with ties to the smaller rate; the estimate, its plain support set and the
curvature read these sums only. The uncertainty's clusters are the pedigree's equal-material units
([material units](pedigree.md#material-units), the run's `MaterialParameters`; 4.21 C10). Each fit also keeps, per
informative component (one with an eligible sample), the per-bin terms of its log-likelihood summed over the samples
(bin b's term is log P(x_b | x_1..x_b-1), the log normaliser of the scaled forward step, so a stretch of bins carries
its share of the likelihood) and each bin's working-map coordinate, built as the pedigree builds its axis (the
cumulative Haldane distance of the per-edge switch probabilities of the working map between bin centres, clipped at
`switch_clip`). The genome axis places these components, analysed contigs in configured order and components in map
order, at the running sum of the lengths before them (gaps add nothing) and is cut into units of `unit_morgans`
working Morgans: a bin at a cut belongs to the unit on its right, a final piece shorter than `tail_morgans` joins the
unit before it, empty units are omitted, units straddle components and contigs, and no chain restarts at a cut. A
unit's curve is the sum of its bins' terms. With theta = log rate, the refined maximum's unequal-gap three-point
derivatives give each unit's score g_u and the sum's curvature H; the variance inflation is d = max(1, J / H),
J = U / (U - 1) sum_u (g_u - mean g)^2 over the occupied units, and the inflated support set is
{rate : 2 [l(best) - l(rate)] <= 3.84 d} (a log-likelihood loss of at most 1.92 d nats): a support set, never an
interval; no consumer receives an uncertainty. At fixed true parameters, conditional score increments are martingale
differences under the model; here J / H is a working dispersion adjustment from finite differences of the
nuisance-refitted eight-update support curve, and independent unit scores and coverage are not established. The floor d >= 1 means
duplicated evidence keeps the set only where J / H >= 1; at the floor duplication narrows it. It is unresolved below
`assessable_morgans` working Morgans of informative material (the axis length; no count of chromosomes, contigs or
components enters), with one occupied unit, a boundary maximum, a non-positive curvature, two competing modes (a
support set that is not contiguous), or an inflated set that fills the grid or touches a bound. A flat plain set is
"not identified" and the run uses the observation pilot's rate. Each sample's own rate, the maximum of its per-sample
terms summed over contigs (flagged at the grid's top), is published for the pedigree exclusion's per-dyad allowance
(A11). Every rate also keeps the regime of its returned fits, so that a change of regime between rates is visible:
the expected label changes and the error-track occupancy (the share of anchored track-bins in an error state), per
contig and pooled. The exported `painting/calibration.json` carries the profile (estimate, boundary, sets, d, curvature H and
score variance J, occupied units and working Morgans, uncertainty state and reason), the material units (each unit's
working Morgans and curve over the grid), the pooled regime, every contig's rows, per-sample rows and regime rows over
the grid and its midpoints, and the sample identifiers that label the per-sample rates. No emission cache
crosses candidate values.

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

With the read state (`read_classes` > 1) the divisor step differs (`science/painting/read_calibration.py`).
The path divisor is fixed at `path_divisor`. The confidence divisor is chosen from `confidence_grid`
(1 to 16) by each sample's held-out log probability of the hidden part of every reference window
given the rest of its data under the read-state chain, summed over folds and contigs; the choice is
the smallest divisor whose paired difference to the best, summed over samples, is within one
standard error. These fits run on a declared amount of genetic map, `calibration_morgans`
(10 Morgans): the analysed contigs in configured order, whole, accumulated until their working-map
length (between each contig's first and last evidence position) reaches the budget, or every
contig when the run has less. The other contigs publish empty divisor records. The
`read_confidence` record lists the material's contigs, their Morgans and the budget, and whether
the best divisor is the grid's largest. The outlier-rate step uses every analysed contig.

An inferred library whose fitted covariance curve gives zero intermittent loss gets a second check on its own
read depths. At each adjacent spacing from 1 to 5 bp, the sum of squared depth differences divided by the sum
of the two depths estimates one minus called-read overlap. A depth-weighted line estimates its limit at zero
spacing. Four distinct nearby markers spanning at most 10 bp supply a separate moment check on the same
observations. For an inner
depth Y, adjacent depths L and R, and fourth depth F, let V=(Y−L)(Y−R) and T=V(Y−F). The ratio of the sum of
V to the sum of Y estimates the same conditional loss as the paired contrast. The root of q(1+q)=u, for
u=sum(3V+T)/(2 sum Y), estimates the ordinary loss under homogeneous independent thinning. Both inner sites
contribute. Heterogeneous read qualities separate these targets slightly and require u−v−v² ≥ 0 for v=sum V/sum Y.

The paired and four-marker estimates must each exceed zero by the read-sharing detection multiplier times
their uncertainty. The variance condition and the agreement of paired and three-marker estimates must hold
within that multiplier. The loss check borrows this conservative multiplier. The read-sharing family-wise
test count covers sharing detection and library splitting. Each loss contrast uses the larger delete-sample
or delete-1-Mb-block standard error.
The same short-gap endpoints must also show extra sample-specific depth variance under a separable Poisson
model. At each endpoint the check subtracts scaled cross-sample depth products from the within-sample
factorial second moment. The scale uses each sample's whole-contig depth. Cross-sample products remove
common site intensity in expectation under the model. A site shared by two pairs contributes twice.
The check recomputes the scale after each deletion, using the same sample and physical-block uncertainty
rule. Blocks without short pairs still contribute to the whole-contig depth. The excess must exceed zero
by the same multiplier times its uncertainty. Sample-specific effects other than regional intensity can
also contribute to this excess. Sparse or contradictory observations use the covariance fit.
A supported correction sets loss to the measured paired intercept and fits unit nonnegative covering masses on the existing
read-length grid using squared adjacent depth differences at real, depth-weighted gaps. The offset and
allele-evidence chains then consume that library's corrected loss and shape, including the implied covering
depth. The per-library calibration report records the decision, contrasts, support and uncertainty.
The working estimate assumes local continuity of the underlying coverage intensity and a contiguous common
footprint at very short gaps. Independent-site observations and whole-cell dropout can violate those assumptions.

Whether the run uses the read state is decided from the data first (`stages/painting/read_state.py`).
Every analysed contig's read-sharing estimate (evidence stage) is tested, and the samples are grouped
into sequencing libraries by divisive splits tested the same way. All these tests share one
family-wise level: a library shares reads when some lag bin's excess exceeds z* standard errors,
z* = Φ⁻¹(1 − α / M), with α = `sharing_detection_alpha` (1 %) and M every one-sided test the
procedure can make, fixed before any test is read: each contig's lag bins that hold site pairs, once
for the all-sample estimate and four times for each of at most 2^`library_levels` − 1 candidate
splits (the parts' difference in both directions and each part's own estimate). Counting only the
all-sample bins accepts a split of std's chr18 on noise. The count does not correct for the split
being chosen from the same depths: two-means partitions of one library give contrasts above z*, and
the material survival difference (`library_survival_tolerance`) is what rejects them. Samples in
libraries without sharing bypass the read state; when no library of the run shares reads, the whole
run takes `read_classes` = 1 (the calibration above and the paintings the pedigree reads). The
calibration records α, M, z*, the route, each library's largest excess / SE with its signed margin
to z*, winning lag bin and usable bins, and each candidate split's sizes, contrast and margin. The
run log warns when a library's largest excess / SE, or a split's deciding contrast, lies within a
factor of two of z*: the decision is close to the detection edge (a simulated library with
independent reads and regional depth variation reaches z* + 0.005 on one contig).

Mixed-technology painting can have higher genotype error for long-read samples on some chromosomes.
With the earlier decoder, the same 30 ONT samples had 0.05280% genotype error in the 5× ONT/PE150
simulation and 0.00579% in the pure ONT control. Chromosomes 8 and 20 accounted for 87.1% of the mixed ONT
errors in that comparison. On the saved mixed inputs, replay with the destination-aware decoder gave
0.09390% genotype error in fine painting and 0.10685% in the reference repaint for the ONT samples.
The mixed replay released 92 of 98 true Tier B parent edges and had zero false parent edges, including
partial calls.

At clean calls with qAF ≥ 0.999, the saved simulations recorded 623 label errors in `seed`
and 2,012 in `std`, compared with posterior expectations of 78 and 429, respectively.

### Adaptive loss offsets

The depth-only chain infers the number of covering reads lost to deletions or third-base calls at
each evidence site. For positive fitted loss, its pilot width starts from the upper Poisson quantile
(tail probability `loss_pilot_tail`) of stationary lost depth at the library's highest mean covering
depth. The pilot adds `loss_pilot_extra_offsets` for depth jumps and bounds the result to 13–33
offsets by default. All observed depths in the span, including those between selected sites,
condition the pilot. Each site keeps at least `offsets` (three by default) and discards only an
upper tail with pilot posterior mass at most `loss_span_tail` divided by the number of evidence
sites in the span. The union bound limits the discarded path probability within the finite pilot
to `loss_span_tail` (0.0001 by default).

A second forward–backward pass conditions the depth chain on these supports. The site marginals
weight the allele likelihoods and the adjacent joint laws weight the class operators. The builder
uses one joint-law scratch matrix per sample, independent of span length, and computes the retained
site and edge laws in parallel across samples. Zero fitted loss bypasses the pilot because all
posterior mass sits at offset zero. The wider-support checks include observed depth jumps and
high-loss cases and compare both site and edge laws.

The read model uses the fitted overlap for each adjacent pair of evidence sites and averages over read
age within the retained state. Between selected sites it composes the intervening depth-conditioned
operators. This approximation can propagate allele evidence across a span that no single molecule could
cover. With fixed 30 kb reads, one called read at each of three sites, mean covering depth one, zero loss
and a 1% base-error probability, the model gives final REF probability 0.713 after an initial REF at
positions 0, 15 and 30 kb. The molecule reference gives 0.500. The calculation uses the middle site's
depth and skips its allele evidence. At positions 0, 7.5 and 15 kb, the endpoints have positive overlap
and the probabilities are 0.909 and 0.884, respectively.

A zero-overlap span reset would replace the composed operator with the target class law when consecutive
selected sites skip an evidence site and their separation reaches the longest positively weighted length
in the library's fitted distribution. The fine painting had zero such spans across famhs, longbc, seed,
bc8, std and TroMau. The six paired reset replays agreed with their controls on released genotypes and
switches.

Single-value grids skip their corresponding calibration fits. The setting
robustness_grid=(0.01,), divisor_grid=(10.9,), and both bin sizes 100 gives r=0.1 for both
strengths and reproduces the 4.19 painting.
Every fine and reference painting depends on the final run calibration identity and
uses the same estimates; the reference painting has its own copying-chain fit after
the fine painting's label-continuity outcome. After the founder refit, the reference
decode is repeated on the refit's founders with the same parameters, calibration,
evidence and map, and with the fine painting's label-continuity outcome passed without
row maps (the refit's panels already carry them): painting/refit_reference. Pedigree
consumes that second reference painting; founder refit, phase, recombination, export
and evaluation consume fine paintings.
