# Output dictionary

Published files are views of checked records. They contain released calls and
recorded diagnostics; missing alleles are never filled by general imputation.
All paths below are relative to the run's output directory. A partial run writes
only the products it has. Standalone export writes into the destination supplied
to `haplotypes export RUN DEST`.

[Exact report column tables](../reference/output-columns.md) are generated from
the writers' column constants and field descriptions. [Record schemas](../reference/records.md)
describe the typed products behind the files. This page defines the file-level
semantics and encodings that those tables cannot express.

## Coordinates, ordering and text

Positions are 1-based input SNP coordinates. Painting intervals labelled
`Start, End` are half-open `[Start, End)`. Recombination call intervals use
`(left, right]`; reported uncertainty intervals have the additional meaning below.
A founder **row** is a panel haplotype; a VCF **row** is a genomic site.

Records use canonical sample-name order. Per-sample published tables and sample
VCF columns use the input sample order. Contigs follow configuration order.
Founder labels are local to an export segment. Export filenames percent-encode
contig names; report directories use contig names directly.

CSV/TSV files have no pandas index column. Ordinary floats use shortest round-trip
text; `NaN` and `None` become empty cells, and booleans become `True` or `False`.
Row-dictionary and painting writers encode list, tuple and dictionary cells as
JSON with sorted dictionary keys. **Pedigree tables use Python tuple reprs**:
`('sample_a', 'sample_b')`, `()`, and nested tuples for `ConfigurationCandidates`.
Do not parse these cells as JSON. Quality tables have their own fixed decimal
formats and `NA` spelling, described below.

An integer column containing a missing floating value can contain `1.0`; an
integer-only column contains `1`. This is intentional column-wise text inference.
No rounding or integer cast is applied to make unresolved pedigree rows look like
resolved rows. Each output file is written through a neighbouring temporary file
and atomically placed; the export manifest is written last.

## D1: founder VCF

`export/CONTIG.founders.vcf.gz` (or `export/CONTIG.founders.bcf` from standalone
`haplotypes export RUN DEST --format bcf`) contains one haploid column per row of
each released segment, named `cC_Hk`. `C` is the segment's export index in genomic
order, and `k` is its released row index. These are not assembly component IDs.
`c0_H1` and `c1_H1` do not assert the same founder. Columns list segments in order,
then rows in ascending order. A cut or withheld segment uses the release's
content ordering; the file does not infer identity across a cut.

The default full-length site set includes every input biallelic SNP that reached
the site-quality filter, in increasing position. Loader-dropped multiallelic,
non-SNP and duplicate-position records are absent. A kept-site export includes
only retained sites. Fixed fields, in order, are `CHROM, POS, ID, REF, ALT, QUAL,
FILTER, INFO`, followed by `FORMAT, c0_H0, ...` when founder columns exist.
`ID`, `QUAL` and `INFO` are `.`; REF/ALT preserve input bases (simulations use A/C).
FORMAT is `GT`. A full-length kept site has `PASS`; a removed site lists its applied
removing rules in r1 through r5 order. A kept-site file uses `.` for FILTER.

Within a segment's first-to-last panel position, a panel allele is `0`, `1` or `.`.
Unknown calls, unsupported carrier alleles, assembly alternatives and refit
ambiguity stay `.`. At sites removed **only by r5**, a decided common allele can be
written to unblocked rows of the segment. It is the strict REF/ALT majority among
counted samples: usable AD totals first, fallback GT only when AD is unusable.
An exact tie remains unknown. Rows identified by `r5_withheld_rows` stay missing.
Sites removed by r1–r4, unrepresented sites and positions outside a segment's span
stay `.`. A panel site absent from the kept axis is an error.

Header order is:

1. File format, htslib's PASS definition, source, contig, haploid GT definition.
2. For full-length files: r1–r5 FILTER definitions, `site_set`, `omitted_records`.
3. `founder_columns`, `founder_stage`.
4. Each `component` declaration, followed by its count-verification,
   continuation-verification and ambiguity declarations when present.
5. `unobserved_founder_alleles` and its rule.
6. For full-length files: `common_allele` and `assembly_ambiguity_r5`.
7. For simulations: `synthetic_alleles`.
8. The tab-delimited column header.

A component declaration gives its ID, Start/End, founder and site counts,
withheld-carrier count, source component, segment index and ambiguity-withheld
count. Verification/ambiguity declarations give their uncertainty interval,
cause, domain or alternative count and evaluation status; they do not supply
missing alleles. The header's exact descriptions are visible in the captured
[full founder header](../examples/full.founders.header.vcf) and
[kept founder header](../examples/kept.founders.header.vcf).

## D2: sample phase VCF

`export/CONTIG.samples.vcf.gz` (or `export/CONTIG.samples.bcf`) uses the same site
rows and fixed fields.
Columns are input sample names; FORMAT is `GT:PS`. A cell is `a|b:POSITION` inside
a released phase set, or `a/b:.` when unphased. Alleles are `0`, `1` or `.`;
partial calls such as `0|.:123` and wholly missing `.|.:123` are possible. Removed
full-length sites are `./.:.`. There is no r5 common-allele fill in sample calls.
Unphased genotypes are written in ascending allele order, missing first (`0/1:.`, `./1:.`);
their order carries no phase.

A phase set starts at a component's first kept site and after each released
sample-specific phase break. PS is that start site's bp position, which need not
be heterozygous or itself phased. Sites outside a component are unphased. Displayed
frame-undetermined bit 2 makes a site unphased **without ending its set**: later
phased sites reuse its PS. Each set has an independent orientation; relative phase,
slot assignment and parental origin are different claims. Parent slots carry no
sex labels.

Header order is file format, htslib PASS, source, contig, GT, PS, then full-length
FILTER/site-set/omission lines, the simulation note when appropriate, and columns.
Observed [full sample](../examples/full.samples.header.vcf) and
[kept sample](../examples/kept.samples.header.vcf) headers are captured from real
produced files. In the kept sample file, htslib's VCFv4.2 line precedes the writer's
VCFv4.3 line; both occur. The [header receipt](../examples/vcf_headers.json) hashes
these captures and their produced source files. htslib's inserted line is
`##FILTER=<ID=PASS,Description="All filters passed">`.

## VCF and BCF indexes

A VCF gets a tabix index (`CONTIG.founders.vcf.gz.tbi`, `CONTIG.samples.vcf.gz.tbi`)
when its **last written position** is below 2^29; otherwise it gets a CSI index
(`CONTIG.founders.vcf.gz.csi`). A BCF always gets `.csi`. The rule does not use the contig header length.
Data and index are created before placement; obsolete index types are removed.

## D3: export manifest

`export/manifest.json` has schema `released-allele-export-v3`. Its ordered top-level
keys are `schema`, `checkpoint_root`, `coordinate_system`, `unknown_allele`,
`founder_identity`, `unobserved_founder_alleles`, `sample_phase`, `synthetic_alleles`,
`reference_vcf`, `contigs`, `products`, `format`, `site_set`, `omitted_records`,
`founder_columns`, `founder_identity_flags`, `founder_continuity`; full-length
exports append `filters` and `common_allele`. `checkpoint_root` is the resolved
record location, and `reference_vcf` is the original display path or null, not a
requirement to re-open the source during standalone export.

Each contig entry lists `contig`, `files` in write order, `founder_components`,
`unobserved_founder_alleles` and the site summary. The site summary records written,
kept and removed counts, removal counts by applied rule, r5-only count, common
REF/ALT/tie and read/GT provenance counts, filter state, applied rules and loader
drops. Component entries retain the source component/row axes, segment index and
block span, ambiguity and verification records, withheld counts, first/last
positions, marker/called/missing counts and founder stage. Track selection adds
local column/file mappings; VCF selection adds VCF filename/column mappings.
Release diagnostics add identity-flag, continuity-break, ambiguous-allele and
verification counts, work counters and elapsed release time. A refit record adds
its ambiguous-allele count. Product selection determines which entries are present;
an omitted product is not represented by invented empty data.

## D4: founder identity flags

`export/CONTIG.founder_identity.tsv` is header-only when no flag is present. Its
columns, in order, are `contig, column, export_component, founder_row, source_stage,
source_component, source_row, block, left_position, right_position, left_local_row,
right_local_row, left_partners, right_partners, at_release_break, reason, method,
status`.

Each row marks a change in a row's duplicate-partner set between adjacent atomic
blocks. The export segment owning the right block supplies `column`; source
indices identify the underlying row. Positions are the left block's last and right
block's first marker, in bp. Partner fields are comma-joined column names.
`at_release_break` is 0/1; reason is `local_duplicate_partner_change`, method is
`duplicate-partner-v1`, status is `diagnostic_only`. A flag is not a crossover call
or an instruction to merge columns.

## D5: founder tracks

Selecting `tracks` with standalone export writes
`export/CONTIG.componentNNNN.founders.tsv.gz`. The gzip-compressed header is
`CHROM, POS, H0, H1, ...` with tabs. Rows are panel sites, not full-length filler
sites. `NNNN` is the four-digit export segment index, and `Hk` is local to that
segment. Cells are `.`, `0`, `1`. The ordinary run's export defaults select founder
and sample VCFs; standalone product selection controls tracks.

## D6: report tables

The generated [column dictionary](../reference/output-columns.md) lists every
base column, position, meaning and unit, including all 17 pedigree tables.

| Files | Rows and ordering |
|---|---|
| `quality/CONTIG.samples.tsv`, `.segments.tsv` | Samples in input order; inherited-segment rows in recorded order; written only where sample QC ran |
| `painting/CONTIG/founder_support.tsv`, `founder_support_windows.tsv` | Component, founder row, then window in genomic order |
| `painting/CONTIG/assembly_search.tsv`, `founder_counts.tsv` | Recorded final assembly join searches; components in order |
| `painting/CONTIG/sample_switches.tsv` | Input sample order |
| `painting/CONTIG/sample_segments.tsv` | Input sample, component, track, maximal run of one status/label/member set |
| `painting/CONTIG/label_continuity.tsv`, `painting_model.tsv` | Accepted relabellings in recorded order; model rows by component |
| `relatedness/families.csv` | Input sample order; each member repeats its family's parents and support |
| `relatedness/half_sib_links.csv` | Recorded sorted family-pair order |
| `relatedness/pairs.csv` | Recorded sample-pair order; **optional and off by default** |
| `pedigree/*.csv` | Stable input order where a `Sample` column exists; otherwise recorded order |
| `founder_refit/summary.csv`, `family_phase/summary.csv` | One recorded summary per contig |
| `recombination/CONTIG.*.csv` | Bins, edges, calls or loci in recorded order; genome-wide tables concatenate contigs in configuration order |
| `ties/decisions.csv` | Contigs, stage order, record-group string, kind string; run-level pedigree last |
| `simulation/truth_pedigree.csv` | Analysed samples in pedigree order; parent names may identify unsequenced individuals |

### Quality and painting details

Quality missing/non-finite numbers are `NA`; absent partners are `-`. The sample
formats are: eligible windows one decimal, LRT two, foreign fraction four, foreign
and artefact shares five, artefact peak four, heterozygote fit three, dispersion
four, error fit six. Segment events have one decimal and minor-read fractions
three. `flagged` is 0/1.

Painting support counts each sample once per called founder allele if released
named ancestry and positive read depth both support it. Pooled candidates are
counted separately, never as named carriers. Summary means/minima use called
alleles; without calls they are empty. These tables read the painted panel, while
export withholding can use the refit panel. They are diagnostics of correlated
samples, not independent lineage evidence. Support windows use
`outputs.reports.painting.WINDOW_SITES` panel sites.

Painting segment status is one of `INELIGIBLE_NO_EVIDENCE`,
`LOW_POSTERIOR_ABSTENTION`, `SINGLETON_NAMED`, `POOLED_EQUIVALENCE`, `BACKGROUND`,
`UNANCHORED_TRAJECTORY`, `LOCAL_ALLELE_EQUIVALENCE`, `SWITCH_INTERVAL`. `Label` is
populated only for a singleton named run; `Rows` joins row indices with semicolons.
Switch ratios use the median over samples painted in at least one component,
without generation groups; a nonpositive median gives an empty ratio. Model
`ErrorRunBinsKeepingLabel` is reciprocal error-exit probability, empty at zero.
Assembly-search rows describe each recorded final-assembly join search: its event
records in order, then one closing record. Every row starts with `Level`,
`FirstVariant1Based`, `LastVariant1Based` (the joined batch's level and its first
and last variant, 1-based). Columns are the first-seen union of the records' fields,
so a field absent from a record is blank. An event record has a `stage` field naming
its kind (`beam_covering_proposals`, `linker_em`, `join_replay`, `parent_shadow_join`,
`count_death`, `count_birth`, `count_seed`, `count_revisit`, `count_backstop`,
`count_shadow_ambiguity`, `pool_saturation`, `symmetric_tie`, `terminal_choice`,
`terminal_cycle_covered`) and carries that event's recorded operands. The closing
record holds `total_full_scores`, `final_panel_size`, `final_likelihood` (join log
likelihood, nats), `initialization` (the anchor panels scored at the start),
`selected_initial_anchor`, `sweeps`, `tier1_sweeps`, `sweep_backstop`,
`tier2_passes`, `tier2_accepted`, `saturation_steps`, `saturation_accepted`, `quota`,
`ladder_top`, `ladder_extended`, `quota_backstop`, `beam_seconds`, `count` (the
count step's tallies), `count_exact_scores`, `count_exact_sample_scores`,
`pool_sizes` (original, final, largest), `certified`, `backstop_reached`,
`pool_backstop`, `tie_events`, `phase_seconds`, `work_counters`. `sweeps` counts
tier-1 sweeps plus accepted tier-2 moves; `certified` means that no move improves
the panel and the count rule is quiescent at every rung; `backstop_reached` and
`pool_backstop` mean that the search stopped at a backstop and is not certified.
The search and count rules are described from [AJ14](../methods/assembly.md#aj14)
to [AJ25](../methods/assembly.md#aj25). List and dictionary values are JSON. Times and work counters are diagnostics. Feedback
searches belong to selection's context assemblies.

### Relatedness selection and JSON

`[relatedness] pairwise_ibd = true` enables the optional O(N² L) pair calculation.
It is stored per invocation in `outputs/selection`, outside every scientific
identity. Families are unaffected. The results reader exposes pair fields only
when selection is on **and** the checked `relatedness/pairs` publication exists.
When off, report publication removes a stale pairs CSV while retaining the stored
pair product for later use.

Family IDs are canonical dense ranks ordered by the family's smallest canonical
member index. They are not that minimum index itself. Empty parents are latent
slots. `relatedness/summary.json` contains exactly these keys, in order; there is
no pair-selection key.

| Key | Meaning |
|---|---|
| `families` | Number of fitted full-sib families |
| `sequenced_parent_slots` | Parent slots filled by a sequenced sample |
| `half_sib_links` | Released half-sib links between families |
| `loglik` | The composite family model's final log-likelihood, nats |
| `sweeps` | Move sweeps of the family search |
| `converged` | Whether the family search converged within its sweep limit |
| `search` | Which of the two bounded searches supplied the released partition |
| `search_margin` | That search's objective minus the other search's, nats; null when the other search was not measured |
| `family_sites` | Sites on the family evidence axis |
| `chromosome_sites` | Object: family sites per contig, in configuration order |

### Pedigree tables and JSON

The 17 CSV names, in publication order, are `tier_b_relationships`,
`tier_b_partial_relationships`, `tier_a_relationships`,
`tier_a_partial_relationships`, `complete_relationships`, `tier_b_candidate_sets`,
`parent_state_calls`, `diagnostics`, `call_explanations`, `alternative_explanations`,
`search_diagnostics`, `exclusion_evidence`, `exclusion_diagnostics`,
`evidence_summary`, `chromosome_summary`, `relative_contrasts`, `sibling_guard`.
Optional views can be absent; absence is distinct from an empty available table.

The first six relationship/candidate views insert `ParentSlotBasis` after `Parent2`
when slot assignment is available: `unordered`, `single`, `none`. Slots do not
encode maternal/paternal labels. Diagnostic configurations retain their internal
order. `ParentState` is `zero_observed_parents`, `one_observed_parent`,
`two_observed_parents` or `unresolved`; an unresolved count is empty.
`ConfigurationSupportSet` is `state:parents:prob` text, with semicolon-separated
configurations and three-decimal probability. M0 uses `external+external`.
Nested `ConfigurationCandidates` retain tuple reprs. Alternatives list the retained
local/graph and finite identity-score alternatives in ascending configuration row.

Interval-bearing exclusion tables append the lower, upper and exactness columns
shown in the generated dictionary. A non-exact interval has an empty point `LogE`.
Exclusion evidence can exist for guard dyads even when there are no count-release
targets. Released relationship views are post-guard; complete relationships,
selection/search diagnostics and relative contrasts describe pre-guard evidence.

`pedigree/evidence_calibration.json` serialises the typed calibration report
([`CalibrationRecord`](../reference/records.md#pedigreecalibrationrecord)).
`pedigree/summary.json` has these keys, in order. The nested objects carry the
fields of the linked records, in declaration order.

| Key | Meaning |
|---|---|
| `informative_contigs` | Contigs that supplied pedigree evidence, in configuration order |
| `omitted` | Object: each omitted contig and the reason |
| `reach` | What chromosome resampling can release with this many informative chromosomes ([`TierReach`](../reference/records.md#pedigreetierreach)) |
| `family_solve` | The final reciprocal solve and resampling convergence diagnostics ([`FamilySolveRecord`](../reference/records.md#pedigreefamilysolverecord)) |
| `exclusion` | The count evidence's model and coverage: `model`, `alpha`, `genotype_replacement_probability`, `conditional_on_supported_edge_directions`, `candidate_dyads`, `guard_dyads`, `contigs`, `status` and the further fields of [`ExclusionSummary`](../reference/records.md#pedigreeexclusionsummary) that are present |
| `eligibility` | The unrestricted candidate universe; it states that no sample metadata or parentage truth was used ([`EligibilityRecord`](../reference/records.md#pedigreeeligibilityrecord)) |
| `sibling_guard` | Edges tested and withheld by the sibling-as-parent guard ([`GuardSummary`](../reference/records.md#pedigreeguardsummary)) |

These report JSON files permit NaN where recorded; evaluation JSON below requires
null for undefined values.

### Recombination and ties

Map CSVs give bp bin bounds, expected/called crossovers and exposure, rates in
cM/Mb, effective meioses and cumulative observed cM at the right edge. Coverage and
called-coverage tables distinguish informative from callable exposure. A called
crossover lies between centres of consecutive released chain bins; its reported
uncertainty interval includes both bins' bp bounds. Evaluation reads a true event
in `(low, high]`, and its boundary rule is stated explicitly in the metrics.
The joint endpoint lower bound is `max(0, left_probability + right_probability - 1)`.
`CONTIG.junctions.csv` and run-level `junctions.csv` use the 27 `JunctionLocus`
fields in order. `CONTIG.png` plots expected and called rates against Mb and
an effective-meiosis staircase (9 × 6 inches, 150 dpi).

`ties/summary.json` has these keys, in order:

| Key | Meaning |
|---|---|
| `conventions` | The documentation path of the [tie rules](../methods/ties.md) |
| `reverse_conventions` | Whether the run used the mirrored arbitrary conventions |
| `contigs` | Contigs counted, in configuration order |
| `stages` | Per stage: `tie_classes` (`decisions`, `inside_tie`, `wide`, `boundary`, summed over class kinds) and `bands` (`comparisons`, `inside_band`, summed over threshold-band kinds) |
| `kinds` | Per stage and kind: the four counts of `ties/decisions.csv`, summed over contigs and record groups |
| `ambiguity` | Recorded ambiguity flags: discovery blocks whose selection was taken inside a tie (`ambiguous`, `tied_alternatives`) and the margin of each block's founder-count choice (`score_margin`); final founder-count calls marked ambiguous or closing a count cycle; refinement count decisions whose tied refits disagree |
Counters come only from the declared evidence, discovery, selection, assembly,
painting and pedigree sources, without duplicate final-assembly or context-refit
counters. Band kinds are summarised separately. Finite discovery score margins use
the **upper middle** element as median: margins 0.2 and 0.8 yield 0.8. This differs
from the arithmetic median used by other reports. Timing columns and work counts
are diagnostics, not evidence or release probabilities.

### Empty tables

A table constructed from no row dictionaries and no fixed columns is a single
newline with no header: recombination meioses/summary, refit/phase summaries and
painting support/window/search/count tables. Fixed-schema empty tables retain
headers: crossover, junction, coverage, called coverage, map, tie decisions,
truth pedigree, painting segments/continuity/model and quality tables, plus the
relatedness tables. Pedigree emptiness follows each constructor: many tables are
headerless, while exclusion evidence and evidence summary keep their columns and
slot/guard additions follow their view. Consumers must handle both an absent file
and an available empty file; neither implies a zero-valued scientific estimate.

## D7: evaluation outputs

`evaluation/summary.json` and five CSVs are written only by evaluation of simulated
truth. The CSVs are `chromosomes.csv`, `phase_groups.csv`, `founders.csv`,
`founder_components.csv`, `founder_matches.csv`. Columns are the first-seen union
of keys from actual rows: unavailable stages contribute no columns; a missing key
in one row is blank; an empty row list is a single newline. Rows follow manifest
contig order, then stage/group/component/match order. Truth is used only here,
never by inference. See the [evaluation method](../methods/evaluation.md) for the
founder, ancestry and map metrics.

### summary.json

Keys, in order. Undefined values are null; NaN is an error.

| Key | Meaning |
|---|---|
| `seed` | The simulation's realised seed |
| `samples` | Number of analysed samples |
| `pedigree` | Null, or the genome-wide Tier-B pedigree comparison below; it stays genome-wide when contigs are restricted |
| `chromosomes` | The `chromosomes.csv` rows |
| `phase_groups` | The `phase_groups.csv` rows |
| `founders` | The `founders.csv` rows |
| `requested_stages` | The `--stages` values as given |
| `checkpoint_root` | The evaluated run's resolved checkpoint root |
| `phase_set_metric` | `released_phase_v3`, the version of the phase definitions below |
| `skipped` | `CONTIG/stage` entries that automatic selection skipped because the product was not complete |
| `phase_totals` | Per parent group: `samples` (the group's size, kept once), the nine phase counts summed over contigs, `phased_share` recomputed from the sums, and `phase_set_n50_bp`, `released_phase_set_n50_bp` over the concatenated spans of all contigs |
| `totals` | `total_alleles`, `called_alleles` and the nine phase counts summed over contigs, `phased_share` recomputed, and the `all` group's genome-wide N50s (0 without phase) |
| `interpretation`, `segment_and_long_range_interpretation` | Fixed text that states how the counts are to be read |

The `pedigree` object compares released Tier-B calls with the true pedigree
restricted to analysed samples; a parent outside them is not an observable edge.
Edges are matched by sample name, ignoring slot order.

| Key | Meaning (counts) |
|---|---|
| `exact_configurations` | Samples whose called parent state and exact parent set equal the truth |
| `correct_parent_states` | Samples whose called state (`zero_observed_parents`, `one_observed_parent`, `two_observed_parents`) equals the true one; an unresolved call is never correct |
| `expected_states` | Object: the number of samples in each true state |
| `true_edges` | True parent-to-child edges between analysed samples |
| `correct_edges` | Called edges that are true |
| `extra_edges` | Called edges that are not true |
| `missing_edges` | True edges that were not called |

### chromosomes.csv

One row per evaluated contig. Phase columns are present when family phase was
evaluated, map columns when recombination was; the two N50 columns come last
because the summary step adds them.

| Column | Present | Meaning and unit |
|---|---|---|
| `contig` | always | Contig name |
| `sites` | always | Simulated truth markers on the contig |
| `total_alleles` | phase | Sample-site allele cells, 2 × samples × sites |
| `called_alleles` | phase | Released allele calls that are not missing |
| `called_fraction` | phase | `called_alleles` / `total_alleles` |
| `called_genotypes` … `released_phase_sets` | phase | The nine phase counts defined [below](#phase-sites-and-switch-events-are-separate-metrics), summed over samples |
| `phased_share` | phase | `phased_true_heterozygotes` / `correct_true_heterozygotes`; empty when the denominator is zero |
| `true_crossovers_in_inferred_exposure` | map | True crossovers of called edges that are true edges, inside those edges' informative spans |
| `expected_crossovers_on_correct_edges` | map | Sum of expected crossovers over called edges that are true edges |
| `correct_edge_exposure_meiosis_bp` | map | Summed exposure of those edges, bp × meioses |
| `inferred_edges_absent_from_truth` | map | Called meiosis edges that are not true edges; they are not scored |
| `called_crossovers_on_correct_edges` | map | Called crossover intervals on true edges |
| `called_intervals_containing_true_crossover` | map | Of those, intervals `(left, right]` containing a true crossover |
| `called_interval_hit_rate` | map | The ratio of the two previous counts; empty without calls |
| `reported_intervals_containing_true_crossover` | map | The same count for the reported intervals `(low, high]` |
| `reported_interval_hit_rate` | map | The reported-interval ratio; empty without calls |
| `called_interval_median_width_bp` | map | Median called-interval width, bp; empty without calls |
| `reported_interval_median_width_bp` | map | Median reported-interval width, bp; empty without calls |
| `phase_set_n50_bp` | phase | N50 of scored phase-set spans (first to last scored site), bp; 0 without a positive span |
| `released_phase_set_n50_bp` | phase | N50 of released phase-set spans, bp |

### phase_groups.csv

One row per contig and parent group, groups in the order `all`, `parents0`,
`parents1`, `parents2`; the parent groups exist only when the run has a pedigree
product, and group samples by their number of released Tier-B parent edges.
Columns: `contig`, `parent_group`, `samples` (samples in the group), the nine phase
counts, `phased_share`, `phase_set_n50_bp`, `released_phase_set_n50_bp`, with the
meanings above restricted to the group's samples.

### founders.csv

One row per contig and evaluated founder stage. Columns, in order:

1. `contig`, `stage`, `components`, `markers`, `reconstructed_markers`,
   `uncovered_markers`, `min_founder_rows`, `max_founder_rows`, `truth_founders`,
   `truth_alleles`, `represented_truth_alleles`, `absent_truth_alleles`;
2. the component sums `called_alleles`, `missing_alleles`, `called_allele_errors`,
   `one_to_one_called_allele_errors`, `matched_called_alleles`,
   `unmatched_founder_rows`, `unmatched_truth_founders`, `unmatched_called_alleles`,
   `truth_to_panel_errors_missing`, `represented_matched_called_alleles`,
   `represented_matched_called_errors`, `absent_matched_called_alleles`,
   `absent_matched_called_errors`, `represented_truth_to_panel_errors_missing`,
   `absent_truth_to_panel_errors_missing`;
3. `called_errors_per_million`, `one_to_one_errors_per_million`;
4. `segment_allele_errors`, `segment_label_switches`, `chimeric_segment_charges`,
   `split_charged_segment_cost`;
5. copy strata, `STRATUM_copy_COUNT` for each stratum `absent`, `single`, `multi`
   and count `truth_alleles`, `correct`, `wrong`, `missing` (stratum-major);
6. identity continuity, `identity_KIND_VERDICT_STRATUM` for kind `direct`, `gap`,
   verdict `correct`, `wrong` and stratum `absent`, `single`, `multi`, then
   `identity_flagged_correct`, `identity_flagged_wrong`, `identity_labelled_cells`,
   `identity_coverage_pairs_single`, `identity_coverage_covered_single`,
   `identity_coverage_covered_unflagged_single`, the same three for `multi`, and
   `identity_windows`.

At chromosome level, uncovered markers add every truth founder's alleles there to
`truth_to_panel_errors_missing` and its represented/absent parts.

### founder_components.csv

One row per contig, stage and component. Columns, in order: `contig`, `stage`,
`component`, `first_position`, `last_position`, `founder_rows`, `truth_founders`,
`markers`, `called_alleles`, `missing_alleles`, `called_allele_errors`,
`one_to_one_called_allele_errors`, `matched_called_alleles`,
`unmatched_founder_rows`, `unmatched_truth_founders`, `unmatched_called_alleles`,
`truth_alleles`, `truth_to_panel_errors_missing`, `represented_truth_alleles`,
`absent_truth_alleles`, `represented_matched_called_alleles`,
`represented_matched_called_errors`, `represented_truth_to_panel_errors_missing`,
`absent_matched_called_alleles`, `absent_matched_called_errors`,
`absent_truth_to_panel_errors_missing`, `segment_allele_errors`,
`segment_label_switches`, `chimeric_segment_charges`, `split_charged_segment_cost`.

### founder_matches.csv

One row per one-to-one match, in panel-row order: `contig`, `stage`, `component`,
`founder_key` (the panel row as text), `predicted_row`, `truth_founder`,
`called_alleles`, `errors`, `missing` (the row's calls, mismatches against the
matched truth founder, and missing calls).

### Founder metric definitions

All values are allele counts unless stated. A panel row is compared with the true
founder haplotypes at the component's markers.

| Column | Meaning |
|---|---|
| `components`, `founder_rows`, `min_founder_rows`, `max_founder_rows` | Evaluated components; rows of one component; the fewest and most rows of any component |
| `markers`, `reconstructed_markers`, `uncovered_markers` | Truth markers (per component: the component's markers); those inside a component; those outside every component |
| `first_position`, `last_position` | The component's first and last marker, bp |
| `truth_founders`, `truth_alleles` | True founder haplotypes F; F × markers |
| `represented_truth_alleles`, `absent_truth_alleles` | True founder alleles at markers where the founder is, or is not, carried by a sampled homologue |
| `called_alleles`, `missing_alleles` | Called and missing panel cells |
| `called_allele_errors` | Mismatches of each row against its nearest truth founder (nearest-row errors) |
| `one_to_one_called_allele_errors` | Mismatches under a one-to-one row-to-founder assignment minimising errors plus missing calls; exact ties are broken by row content |
| `matched_called_alleles`, `unmatched_called_alleles` | Called alleles in matched and unmatched rows |
| `unmatched_founder_rows`, `unmatched_truth_founders` | Rows without a founder, and founders without a row, under that assignment |
| `truth_to_panel_errors_missing` | One-to-one errors plus missing calls of matched rows plus every allele of unmatched founders |
| `represented_…`, `absent_…` | The same matched counts split by whether the true allele is represented |
| `called_errors_per_million`, `one_to_one_errors_per_million` | 10⁶ × errors / called alleles (nearest-row) or / matched called alleles (one-to-one); empty when the denominator is zero |
| `segment_allele_errors`, `segment_label_switches` | Errors and label switches when each row is labelled segment-wise, a switch costing `evaluation.parameters.SEGMENT_SWITCH_COST` mismatches |
| `chimeric_segment_charges` | `called_allele_errors` − `segment_allele_errors` |
| `split_charged_segment_cost` | Segment errors plus the switch cost for every switch and for one switch per row at each component start after the contig's first; it never falls when a component is cut |
| `STRATUM_copy_COUNT` | One-to-one correct, wrong and missing alleles (and the truth total) of alleles whose true founder has no (`absent`), one (`single`) or several (`multi`) sampled copies |
| `identity_*` | Continuity assertions on a fixed truth-marker window grid: a row labelled with founders in consecutive windows asserts continuity, correct when the labels agree; `direct` for adjacent windows, `gap` across unlabelled windows; flagged assertions lie at a row's identity flag; coverage pairs are eligible window pairs of a founder, covered when one row holds it in both |

Represented ancestry means present in a sampled homologue, not independently
identifiable. The [evaluation method](../methods/evaluation.md) states the window
labelling and coverage rules with their named parameters.

### Phase sites and switch events are separate metrics

A scored site has both alleles called, the correct unordered genotype, true
heterozygosity, membership in a phase set, and released displayed orientation.
Genotype errors and unphased heterozygotes are not scored phase sites and do not
split a set. A released set uses called heterozygosity and phase availability
without checking truth: a call 0|1 against truth 0/0 gives one genotype error, one
released set and zero scored sets.

| Metric | Unit and definition |
|---|---|
| `called_genotypes` | Sample-site cells with two called alleles |
| `genotype_errors` | Called genotypes with incorrect allele sum |
| `correct_true_heterozygotes` | Correctly called true heterozygotes |
| `phased_true_heterozygotes` | Scored phase sites |
| `set_aligned_errors` | Sites on the minority orientation within each scored set, summed |
| `switch_errors` | Orientation changes between consecutive scored sites within a set |
| `long_switches` | Orientation changes between consecutive retained runs; runs shorter than `evaluation.parameters.LONG_SWITCH_MIN_RUN` scored sites are ignored |
| `phase_sets`, `released_phase_sets` | Sets with a scored site, or a released called heterozygote, respectively |
| `phase_set_n50_bp`, `released_phase_set_n50_bp` | N50 of first-to-last scored/phased-heterozygote spans; zero with no positive span |
| `phased_share` | Scored sites divided by correct true heterozygotes |

Worked examples of orientation relative to truth, all in one set:

| Orientations | Scored sites | Site errors | Switch events | Long switches |
|---|---|---|---|---|
| `0 0 0 1 1 0 0 0 0 1` | 10 | 3 | 3 | 0 |
| 30 zeros, 2 ones, 25 zeros, 21 ones | 78 | 23 | 3 | 1 |
| Alternating at every site, length n | n | floor(n/2) | n − 1 | 0 for runs below the long-run threshold |

Parent groups `parents0`, `parents1`, `parents2` count released post-guard incoming
edges in the inferred pedigree; `all` includes every sample. They do not use true
or cohort-derived parent labels.

## D8: simulation records

These are checkpoint products, not loose output tables: run-level
`simulation/manifest`, and per-contig `simulation/reads`, `simulation/truth`.
Their schema is 1. The manifest binds state, requested/realised seed, contigs,
source identity, analysed sample IDs and observed/generated pedigree tables.
Reads store positions int64[L] and AD counts int32[N,L,2]. Truth stores founder
haplotypes, diploid sample alleles, ancestry chunks and true crossovers. Chunk
bounds are `[start,end)`, with the final chunk including the last marker.
The generated [record reference](../reference/records.md) lists their fields.
Explicit-pedigree sex and generation descriptions are stored as truth metadata;
no inference step reads them. `simulation/truth_pedigree.csv` is the public summary.

## D9: founder templates

`CONTIG.npz` contains `positions` int64[L], strictly increasing 1-based positions,
and `allele_probabilities` float64[H,L,2] for REF/ALT probabilities. Consecutive
haplotype rows form diploid founders, so H is even. Templates supply founder
material to simulations; they do not contain inferred family labels. The synthetic
example has six haplotype rows, giving three diploid founders.

## Results readers

Use `haplotype_reconstruction.results.open_results` to obtain a `Results` reader.
Its methods expose checked products such as evidence, relatedness, pedigree,
painting, refit, phase, recombination and released sites. Readers derive expected
identities from the manifest/upstream bindings and decode only requested items;
they do not trust a product's own claimed identity or completion marker.
Absent optional products return the documented optional result; incompatible
schema or derivation is an error. `stage_panels` supplies original component/row
axes for evaluation, including founder refit; the export release is a distinct
view with carrier/ambiguity withholding and segment-local row names.
