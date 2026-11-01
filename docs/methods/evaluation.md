# Evaluation against recorded truth

Evaluation runs after inference on simulated products and independent recorded truth. Founder recovery, genotype/site-phase accuracy, switch events, directed pedigree edges and map intervals measure different properties. N is observed samples, L truth sites, K inferred founder rows, F truth founders and E inferred edges. Partial-stage reports have conditional columns. Metrics retain known truth-surface and assignment conventions; they are not interchangeable accuracy scores.

The sections below identify semantic units, not separately configurable pipeline steps. [Outputs](../user/outputs.md), [record contracts](../reference/records.md) and [costs](../developer/costs.md) describe their public boundaries.

<a id="ou31"></a>
## ou31 Evaluation orchestration: stage vocabulary, selection, rows, summary and tables

Evaluate available, all or explicitly requested products using the declared stage vocabulary. Score founder stages on original component/row axes before export withholding or segmentation. Preserve manifest contig order and build CSV columns from the first-seen union of actually present row keys; empty rows produce a newline-only file. Strict summary JSON uses null for undefined values. Recompute aggregate ratios and N50 from pooled counts/spans, not averages of contig metrics.

Implementation: [evaluation/tasks.py](../../haplotype_reconstruction/evaluation/tasks.py), [evaluation/evaluate.py](../../haplotype_reconstruction/evaluation/evaluate.py).

Parameters by name: `PHASE_METRIC`, `TASK_ALLOWANCE_BYTES`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: Independent contig evaluation; decoded truth, selected products and any reordered copies determine peak storage.

<a id="ou32"></a>
## ou32 Founder panel metrics: nearest-founder, one-to-one, truth-to-panel, represented/absent

Report nearest-founder errors separately from content-ordered one-to-one assignment under error-plus-missing cost. Charge unmatched truth founders and uncovered markers as missing, and retain unmatched predicted rows. Assignment ties follow the pinned solver on stably content-sorted rows, so some assignment-derived errors/strata can differ between equally optimal matches. Duplicate predicted rows can score well by nearest-founder error but do not establish recovery of all founders.

Implementation: [evaluation/founders.py](../../haplotype_reconstruction/evaluation/founders.py).

Parameters by name: `SITE_TILE`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(Σ K_c L_c F); O(K³). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou33"></a>
## ou33 Segment-wise founder score and the split-charged cost

Find each predicted row’s minimum-cost truth-founder path with mismatch and named switch charges. Keep the current label on equal switch/stay cost and choose the first terminal minimum. Count segment errors and label switches separately, charging a split at every later component boundary per row. This segment metric permits identity changes deliberately; it complements whole-component founder matching.

Implementation: [evaluation/founders.py](../../haplotype_reconstruction/evaluation/founders.py).

Parameters by name: `SEGMENT_SWITCH_COST`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(Σ K_c L_c F). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou34"></a>
## ou34 Truth ancestry masks: represented ancestry, copy counts and strata

Project half-open truth ancestry chunks onto marker indices, including the documented final-marker endpoint exception. Count founder presence once per sample but homologue copies separately, ignoring unknown ancestry. Build absent/single/multiple-copy strata and score them through the same one-to-one founder assignment. Missing strata include uncalled, unmatched and uncovered truth alleles.

Implementation: [evaluation/founders.py](../../haplotype_reconstruction/evaluation/founders.py), [evaluation/identity.py](../../haplotype_reconstruction/evaluation/identity.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(chunks); O(F L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou35"></a>
## ou35 Identity continuity assertions and coverage

Label each predicted row independently within fixed truth-marker windows only when call count, error and separation gates pass. Compare consecutive labelled windows and classify continuity by ancestry-copy strata. Coverage uses consecutive eligible truth windows within the maximum gap, requiring one consistent covering row and checking identity flags on the intervening physical interval. It does not enumerate every nearby window pair or relabel founders site by site.

Implementation: [evaluation/identity.py](../../haplotype_reconstruction/evaluation/identity.py).

Parameters by name: `IDENTITY_WINDOW_MARKERS`, `LABEL_MIN_CALLS`, `LABEL_MARGIN`, `COVERAGE_MIN_DISTANCE`, `COVERAGE_MAX_GAP`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(K F L); O(F² · IDENTITY_WINDOW_MARKERS); O(windows · F² · IDENTITY_WINDOW_MARKERS). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou36"></a>
## ou36 Released-phase metrics: sites, switch events, long switches, sets, N50 and parent groups

Score phase only at genotype-correct, called true heterozygotes released phased within a set. Count site-aligned errors by the minority orientation within each set, switch events between consecutive scored sites, and long switches between retained long orientation runs. Unscored sites do not split sets. Separately count truth-free released sets, including wrongly called phased heterozygotes. Pool spans for N50 and group samples by released post-guard parent edges. Site errors and switch events have different denominators.

Implementation: [evaluation/metrics.py](../../haplotype_reconstruction/evaluation/metrics.py).

Parameters by name: `LONG_SWITCH_MIN_RUN`, `PHASE_METRIC`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N · L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou37"></a>
## ou37 Pedigree metrics

Match inferred Tier-B calls to observed-parent truth by sample name, ignoring arbitrary slot order. Require both correct parent state and exact parent set for an exact configuration; an unresolved empty call is not exact. Count directed correct, missing and extra edges independently. Cohort labels are not substituted for individual truth.

Implementation: [evaluation/evaluate.py](../../haplotype_reconstruction/evaluation/evaluate.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou38"></a>
## ou38 Map metrics

Evaluate map calls only on inferred parent-child edges present in the true gamete mapping and within inferred exposure. A true edge with no events still contributes expected counts, exposure, calls and interval widths; an absent truth key is skipped and counted. Use open-left/closed-right event intervals for both call-centre and reported-interval hits. Report hit rates alongside widths because wider intervals hit more often by construction.

Implementation: [evaluation/evaluate.py](../../haplotype_reconstruction/evaluation/evaluate.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(calls · log events). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.
