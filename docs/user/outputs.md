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
`start, end` are half-open `[start, end)`. Recombination call intervals use
`(left, right]`; reported uncertainty intervals have the additional meaning below.
A founder **row** is a panel haplotype; a VCF **row** is a genomic site.

Records use canonical sample-name order. Per-sample published tables and sample
VCF columns use the input sample order. Contigs follow configuration order.
Founder labels are local to an export segment. Export filenames percent-encode
contig names; every run-level table uses the original contig name in its leading `contig` column.
Rows concatenate contigs in configuration order, then retain each contig's existing order.
All table columns are snake_case; shared keys are `contig`, `sample`,
`component`, `founder`, `parent1` and `parent2`. VCF fields retain their standard spelling.
CSV uses standard quoting, including for JSON cells containing commas or quotes.

CSV files have no pandas index column. Ordinary floats use shortest round-trip
text; `NaN` and `None` become empty cells, and booleans become `True` or `False`.
Row-dictionary and painting writers encode list, tuple and dictionary cells as
JSON with sorted dictionary keys. Cohort-sized pedigree collections use counts
and ordered JSON values in `pedigree/cell_items.csv`; see [collection tables](#collection-tables).
Quality tables have their own fixed decimal formats and `NA` spelling, described below.

An integer column containing a missing floating value can contain `1.0`; an
integer-only column contains `1`. This is intentional column-wise text inference.
No rounding or integer cast is applied to make unresolved pedigree rows look like
resolved rows. Each output file is written through a neighbouring temporary file
and atomically placed; the export manifest is written last. Failed writes remove
their private temporary files and preserve any existing complete destination.

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
[full founder header](../examples/full.founders.header.vcf.txt) and
[kept founder header](../examples/kept.founders.header.vcf.txt).

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
Observed [full sample](../examples/full.samples.header.vcf.txt) and
[kept sample](../examples/kept.samples.header.vcf.txt) headers are captured from real
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

`export/manifest.json` has schema `released-allele-export-v4`. Its ordered top-level
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

`export/founder_identity.csv` is header-only when no flag is present. Its
columns, in order, are `contig, column, component, founder, source_stage,
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

Selecting `tracks` with standalone export writes `export/CONTIG.tracks.csv.gz`.
The gzip CSV header is `contig, component, pos, h0, h1, ...`. Rows list export
segments in order, then panel sites within each segment; they exclude full-length
filler sites. `component` is the export segment index and `hk` is row k local to
that segment. The header extends to the largest segment's founder count; cells
beyond a shorter segment's row count are empty, while an existing unknown allele
is `.`. Called alleles are `0` or `1`. Manifest `local_columns` retains its `Hk`
keys for the segment-local founder labels, which correspond to track columns `hk`.
The ordinary run selects founder and sample VCFs; standalone selection controls tracks.

## D6: report tables

The generated [column dictionary](../reference/output-columns.md) lists every
base column, position, meaning and unit, including all 17 pedigree tables.

| Files | rows and ordering |
|---|---|
| `quality/samples.csv`, `quality/segments.csv` | Samples in input order; inherited-segment rows in recorded order; written only where sample QC ran |
| `painting/founder_support.csv`, `founder_support_windows.csv` | component, founder row, then window in genomic order |
| `painting/assembly_search.csv`, `founder_counts.csv`, `assembly_search_shadow_alternatives.csv` | Recorded final assembly join searches; components in order |
| `painting/sample_switches.csv` | Input sample order |
| `painting/sample_segments.csv` | Input sample, component, track, maximal run of one status/label/member set |
| `painting/label_continuity.csv`, `painting_model.csv` | Accepted relabellings in recorded order; model rows by component |
| `relatedness/families.csv` | Input sample order; each member repeats its family's parents and support |
| `relatedness/half_sib_links.csv` | Recorded sorted family-pair order |
| `relatedness/pairs.csv` | Recorded sample-pair order; **optional and off by default** |
| `pedigree/*.csv` | Stable input order where a `sample` column exists; otherwise recorded order |
| `founder_refit/summary.csv`, `family_phase/summary.csv` | One recorded summary per contig |
| `recombination/map.csv`, `recombination/crossovers.csv`, `recombination/meioses.csv`, `recombination/coverage.csv`, `recombination/junctions.csv`, `recombination/junction_members.csv`, `recombination/summary.csv` | bins, edges, calls or loci in recorded order; genome-wide tables concatenate contigs in configuration order |
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
`UNANCHORED_TRAJECTORY`, `LOCAL_ALLELE_EQUIVALENCE`, `SWITCH_INTERVAL`. `label` is
populated only for a singleton named run; `rows` joins row indices with semicolons.
Switch ratios use the median over samples painted in at least one component,
without generation groups; a nonpositive median gives an empty ratio. Model
`error_run_bins_keeping_label` is reciprocal error-exit probability, empty at zero.
Assembly-search rows describe each recorded final-assembly join search: its event
records in order, then one closing record. Every row starts with `level`,
`first_variant_1based`, `last_variant_1based` (the joined batch's level and its first
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

Task resource detail is retained in `run_records/*.tasks.jsonl`, including after
success or failure. The matching invocation JSON holds `task_count` and
`task_summaries`, one aggregate per stage: task outcomes, summed measured task
wall and CPU seconds, maximum measured task peak bytes, maximum starting and
peak thread counts, and known task failures. Missing measurements for failed
tasks are not estimated. Nested stage timers retain inclusive and exclusive
wall time separately; summed task wall time can exceed invocation wall time.
Readers also accept legacy invocation JSON containing an embedded `tasks` list.
Status does not read legacy live task logs without aggregates; it reports their
published task count and stage timers.

### Collection tables

`painting/assembly_search_shadow_alternatives.csv` contains one row per shadow
alternative. The parent key is `contig, assembly_search_row`; the latter is a
zero-based ordinal across all search events and closing records within a contig.
Level and span can repeat and are not a unique key. `alternative` is the
zero-based list position; `value` is the original alternative dictionary as
sorted-key JSON, preserving nulls and absent keys. The parent's
`shadow_alternatives` cell is the number of alternatives: zero for an empty list,
empty when the field is absent. Group on the parent key, order by `alternative`
and JSON-decode `value` to recover the recorded list exactly.

`painting/assembly_search_items.csv` separates symmetric and plateau families:
`panels`, `alternatives`, `observed_panels`, `replay_panels`,
`cycle_excluded_panels`, `values`, `minimum_switches` and `source_passes`.
Each parent list becomes the bounded JSON descriptor `{"items": COUNT}`.
Scalar census values in the same column remain scalar. Child rows carry
`contig, assembly_search_row, column, item, value`; `value` is a JSON element.
Order items by index and JSON-decode them to reconstruct each list, including
empty lists. The replay budget bounds executed continuations, not all recorded
alternatives, which is why these lists also use child rows.

`pedigree/cell_items.csv` separates the following cohort-sized collections. Its
key is `source_table, sample, column, item` for sample tables and
`source_table, parent, child, column, item` for `sibling_guard`. Unused sample or
parent/child keys are empty. `source_table` omits `.csv`, `column` is snake_case,
`item` is zero-based and `value` is JSON. Parent cells give item counts. Empty
collections have count zero and no item rows. Items retain record order; they
are grouped by table and column, then parent-row input order (recorded order for
the sibling guard).

| Parent table | Collection columns | Decoded item |
|---|---|---|
| `tier_b_candidate_sets`, `call_explanations` | `parent1_candidates`, `parent2_candidates` | Sample name |
| `tier_b_candidate_sets`, `call_explanations` | `configuration_candidates` | List of zero, one or two parent names |
| `diagnostics` | `parent1_candidate_set`, `parent2_candidate_set` | Sample name |
| `diagnostics` | `configuration_support_set`, `evidence_configuration_set` | Original formatted configuration item, including its recorded decimal text |
| `exclusion_diagnostics` | `compatible_candidates`, `supported_reverse_candidates`, `remaining_candidates` | Sample name |
| `sibling_guard` | `dropped_members` | Original sample/reason item |

To reconstruct tuple text for a candidate or exclusion cell, form a tuple of
decoded items and use Python `repr`; configuration items become inner tuples.
Brace-set cells join decoded names with commas inside braces. Support,
evidence-configuration and dropped-member cells join decoded items with
semicolons. This recovers their complete former cell text, including item order
and formatting, without parsing delimiters inside sample identifiers.

`recombination/junction_members.csv` has `contig, locus, column, item, value`.
`column` is `parents` or `children`; values are the original semicolon-delimited
text entries. Joining them with semicolons in item order recovers the original
junction cell exactly. The corresponding cells in `junctions.csv` contain item
counts. Repeated child names remain repeated.

`founder_refit/excluded_samples.csv` contains `contig, item, sample`: one row
per excluded sample, with a zero-based item index in recorded order. Its parent
is the contig row in `founder_refit/summary.csv`, where `excluded_samples` is a
count. Group child rows by contig and order by item to recover the complete
original JSON list. An empty list has count zero and no child rows.

Other collection cells have explicit producer bounds: assembly joins use at
most `AssemblyParameters.batch_size` input components (default 10); nearest
shadow arrays use `CountRuleParameters.shadow_combinations` evaluations (default
2048); refit histories use `100 + 20 K` sweeps and count decisions use the
incoming-count domain up to twice K. Join replay work arrays use at most four executions; refinement continuations
use 16 leaves. Their unbudgeted source families use child rows.
Painting candidate rows and export partner sets use at most K founder labels.
Proposal, census, timer and work-counter dictionaries have fixed key vocabularies;
leading evaluated alternatives contain at most two entries. These are bounds on
collection cardinality, not a universal character limit for arbitrary sample
names or custom parameter values. JSON summary documents may grow in rows or
keys; they are documents rather than table cells.

### Relatedness selection and JSON

`[relatedness] pairwise_ibd = true` enables the optional O(N² L) pair calculation.
It is stored per invocation in `output_selection.json` in the output directory, outside every scientific
identity. Families are unaffected. The results reader exposes pair fields only
when selection is on **and** the checked `relatedness/pairs` publication exists.
When off, report publication removes a stale pairs CSV while retaining the stored
pair product for later use.

family IDs are canonical dense ranks ordered by the family's smallest canonical
member index. They are not that minimum index itself. Empty parents are latent
slots. `relatedness/summary.json` contains exactly these keys, in order; there is
no pair-selection key.

| Key | Meaning |
|---|---|
| `families` | Number of fitted full-sib families |
| `sequenced_parent_slots` | parent slots filled by a sequenced sample |
| `half_sib_links` | released half-sib links between families |
| `loglik` | The composite family model's final log-likelihood, nats |
| `sweeps` | Move sweeps of the family search |
| `converged` | Whether the family search converged within its sweep limit |
| `search` | Which of the two bounded searches supplied the released partition |
| `search_margin` | That search's objective minus the other search's, nats; null when the other search was not measured |
| `family_sites` | sites on the family evidence axis |
| `chromosome_sites` | Object: family sites per contig, in configuration order |

### Pedigree tables and JSON

The 17 parent CSV names, in publication order, are `tier_b_relationships`,
`tier_b_partial_relationships`, `tier_a_relationships`,
`tier_a_partial_relationships`, `complete_relationships`, `tier_b_candidate_sets`,
`parent_state_calls`, `diagnostics`, `call_explanations`, `alternative_explanations`,
`search_diagnostics`, `exclusion_evidence`, `exclusion_diagnostics`,
`evidence_summary`, `chromosome_summary`, `relative_contrasts`, `sibling_guard`.
`cell_items.csv` follows these parent tables. Optional views can be absent;
absence is distinct from an empty available table.

The first six relationship/candidate views insert `parent_slot_basis` after `parent2`
when slot assignment is available: `unordered`, `single`, `none`. Slots do not
encode maternal/paternal labels. Diagnostic configurations retain their internal
order. `parent_state` is `zero_observed_parents`, `one_observed_parent`,
`two_observed_parents` or `unresolved`; an unresolved count is empty.
`configuration_support_set` items use `state:parents:prob` text with
three-decimal probability. M0 uses `external+external`. These items and the
parent-name configuration lists are stored in `cell_items.csv`, with counts in
the parent cells. Alternatives list the retained
local/graph and finite identity-score alternatives in ascending configuration row.

Interval-bearing exclusion tables append the lower, upper and exactness columns
shown in the generated dictionary. A non-exact interval has an empty point `log_e`.
Exclusion evidence can exist for guard dyads even when there are no count-release
targets. released relationship views are post-guard; complete relationships,
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
cM/Mb, effective meioses and cumulative observed cM at the right edge. `recombination/coverage.csv` lists spans between consecutive released markers of
a followed stretch. These same callable spans define map coverage, called exposure,
junction decidability and evaluation; an unresolved or interrupted stretch contributes no span. A called
crossover lies between centres of consecutive released chain bins; its reported
uncertainty interval includes both bins' bp bounds. Evaluation reads a true event
in `(low, high]`, and its boundary rule is stated explicitly in the metrics.
The joint endpoint lower bound is `max(0, left_probability + right_probability - 1)`.
`recombination/junctions.csv` uses the 27 `JunctionLocus` fields in order.
`recombination/maps.pdf` has one 9 × 6 inch page per contig in configuration
order, plotting expected and called rates against Mb and an effective-meiosis staircase.

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

The run-level quality, painting and recombination tables retain their headers
when empty. A contig with no rows contributes no rows; a contig where sample QC
did not run contributes no quality table part. Optional unavailable products
remain absent. Refit/phase summaries constructed without rows remain a single
newline with no header. Pedigree emptiness follows each view: many tables are
headerless, while exclusion evidence and evidence summary retain their columns;
slot/guard additions follow their view. An empty available table does not imply
a zero-valued scientific estimate.

<!-- output-columns:start -->
### Complete report column inventory

Columns below are in publication order. Optional pedigree suffixes and dynamic
assembly-search columns follow the rules immediately below; evaluation, tracks
and identity columns are listed in their own sections.

| Table | Columns |
|---|---|
| `quality/samples.csv` | `contig`, `sample`, `observed_cells`, `eligible_cells`, `eligible_windows`, `minor_cells`, `near_private_cells`, `lrt`, `foreign_fraction`, `foreign_share`, `artefact_share`, `artefact_peak`, `heterozygote_fit`, `heterozygote_dispersion`, `error_fit`, `flagged` |
| `quality/segments.csv` | `contig`, `sample`, `partner`, `start`, `end`, `events`, `cells`, `minor_read_fraction`, `partner_minor_read_fraction` |
| `painting/sample_segments.csv` | `contig`, `sample`, `component`, `track`, `start`, `end`, `status`, `label`, `rows` |
| `painting/label_continuity.csv` | `contig`, `component`, `atomic_span`, `position`, `row_a`, `row_b`, `score_gain_switches`, `painted_switch_change`, `winners`, `losers`, `winners_ato_b`, `winners_bto_a`, `identity` |
| `painting/painting_model.csv` | `contig`, `component`, `first_variant_1based`, `last_variant_1based`, `bins`, `states`, `error_capable_classes`, `switch_rate_per_morgan`, `error_entry`, `error_exit`, `error_run_bins_keeping_label`, `site_correlation`, `eligible_samples` |
| `painting/founder_support.csv` | `contig`, `component`, `founder`, `first_variant_1based`, `last_variant_1based`, `break_before`, `break_after`, `marker_count`, `called_alleles`, `missing_alleles`, `observed_carrier_samples`, `called_sites_without_observed_named_carrier`, `minimum_observed_carriers`, `mean_observed_carriers`, `observed_pooled_candidate_samples`, `mean_observed_pooled_candidates`, `called_sites_without_observed_named_or_pooled_candidate`, `support_basis` |
| `painting/founder_support_windows.csv` | `contig`, `component`, `founder`, `first_variant_1based`, `last_variant_1based`, `marker_count`, `called_alleles`, `missing_alleles`, `called_sites_without_observed_named_carrier`, `mean_observed_carriers`, `mean_observed_pooled_candidates`, `called_sites_without_observed_named_or_pooled_candidate` |
| `painting/assembly_search.csv (prefix)` | `contig`, `level`, `first_variant_1based`, `last_variant_1based` |
| `painting/founder_counts.csv` | `contig`, `component`, `first_variant_1based`, `last_variant_1based`, `founder_rows`, `founder_count_ambiguous`, `undecided_count_cycles`, `count_step_founders_before`, `count_step_cycle`, `count_step_cycle_undecided`, `count_step_ambiguous_decisions`, `count_step_decisions`, `join_count_cycles`, `join_undecided_cycles`, `join_ambiguous_count_calls`, `feedback_join_count_cycles`, `feedback_join_undecided_cycles`, `feedback_join_ambiguous_count_calls` |
| `painting/sample_switches.csv` | `contig`, `sample_index`, `sample`, `painted_components`, `painted_switches`, `cohort_median_switches`, `switches_over_cohort_median` |
| `pedigree/tier_b_relationships.csv` | `sample`, `parent1`, `parent2`, `parent_state`, `observed_parent_count`, `inference_status`, `release_evidence` |
| `pedigree/tier_b_partial_relationships.csv` | `sample`, `parent1`, `parent2`, `parent_state`, `observed_parent_count`, `inference_status`, `parent1_supported`, `parent2_supported`, `exact_configuration_resolved`, `observed_parent_count_minimum`, `observed_parent_count_maximum`, `release_evidence` |
| `pedigree/tier_a_relationships.csv` | `sample`, `parent1`, `parent2`, `parent_state`, `observed_parent_count`, `inference_status` |
| `pedigree/tier_a_partial_relationships.csv` | `sample`, `parent1`, `parent2`, `parent_state`, `observed_parent_count`, `inference_status`, `parent1_supported`, `parent2_supported`, `exact_configuration_resolved`, `observed_parent_count_minimum`, `observed_parent_count_maximum` |
| `pedigree/complete_relationships.csv` | `sample`, `parent1`, `parent2`, `parent_state`, `observed_parent_count`, `inference_status` |
| `pedigree/tier_b_candidate_sets.csv` | `sample`, `parent_state`, `observed_parent_count`, `parent1`, `parent2`, `parent1_candidates`, `parent2_candidates`, `configuration_candidates`, `exact_configuration_resolved`, `inference_status`, `parent1_supported`, `parent2_supported`, `observed_parent_count_minimum`, `observed_parent_count_maximum`, `tier_b_final_state_call`, `tier_b_release_evidence` |
| `pedigree/parent_state_calls.csv` | `sample`, `selected_parent_state`, `local_winner_parent_state`, `observed_parent_count`, `local_observed_parent_count`, `inference_status`, `dag_displaced`, `graph_conflict`, `graph_tie_conflict`, `graph_direction_resolved_alternative`, `selected_parent_role_probability`, `tier_a_state_call`, `tier_b_state_call`, `tier_b_final_state_call`, `tier_b_release_evidence` |
| `pedigree/diagnostics.csv` | `sample`, `m0_prior_sensitivity_state_agreement`, `m0_prior_sensitivity_graph_agreement`, `m0_prior_sensitivity_tier_veto`, `structure_child_evaluable`, `structure_selected_row_exposure_testable`, `structure_selected_row_cx_compatible`, `structure_parent1_coverage`, `structure_parent2_coverage`, `structure_pair_explainability`, `structure_parent1_direction_probability`, `structure_parent2_direction_probability`, `best_m1_parent`, `best_m1_conditional_identity_margin`, `best_m2_parent1`, `best_m2_parent2`, `best_m2_conditional_identity_margin`, `local_parent1`, `local_parent2`, `eligible_child`, `eligible_parent_count`, `eligible_parent_pair_count`, `complete_parent1`, `complete_parent2`, `selected_parent_state`, `local_winner_parent_state`, `observed_parent_count`, `local_observed_parent_count`, `identifiable`, `dag_displaced`, `graph_conflict`, `graph_tie_conflict`, `graph_direction_resolved_alternative`, `local_parent_role_probability`, `selected_parent_role_probability`, `raw_ancestry_junction_count`, `callable_ancestry_haplotype_bin_count`, `adjusted_ancestry_junction_burden`, `ancestry_painting_callability_fraction`, `informative_contig_count`, `informative_map_units`, `local_state_bootstrap_fraction`, `local_configuration_bootstrap_fraction`, `graph_configuration_bootstrap_fraction`, `selected_graph_configuration_bootstrap_fraction`, `selected_graph_parent1_bootstrap_fraction`, `selected_graph_parent2_bootstrap_fraction`, `parent1_bootstrap_fraction`, `parent2_bootstrap_fraction`, `local_state_loco_fraction`, `local_configuration_loco_fraction`, `graph_configuration_loco_fraction`, `selected_graph_configuration_loco_fraction`, `selected_graph_parent1_loco_fraction`, `selected_graph_parent2_loco_fraction`, `parent1_loco_fraction`, `parent2_loco_fraction`, `state_winner_margin`, `conditional_identity_margin`, `unconstrained_winner_margin`, `selected_aggregate_utility`, `selected_minus_unconstrained_best`, `state_log_evidence0`, `state_log_evidence1`, `state_log_evidence2`, `state_support0`, `state_support1`, `state_support2`, `scored_candidate_count2`, `m2_candidate_screen_incomplete`, `prior_sensitivity_local_state_agreement_fraction`, `prior_sensitivity_local_identity_agreement_fraction`, `prior_sensitivity_graph_agreement_fraction`, `prior_sensitivity_selected_state_minimum_support`, `inference_status`, `tier_a_state_call`, `tier_b_state_call`, `tier_a_exact_configuration`, `tier_b_exact_configuration`, `tier_a_exact_pair`, `tier_b_exact_pair`, `tier_a_parent1`, `tier_a_parent2`, `tier_b_parent1`, `tier_b_parent2`, `configuration_support_set`, `parent1_candidate_set`, `parent2_candidate_set`, `evidence_configuration_set`, `tier_b_final_state_call`, `tier_b_release_evidence` |
| `pedigree/call_explanations.csv` | `sample`, `parent_state`, `observed_parent_count`, `parent1`, `parent2`, `parent1_candidates`, `parent2_candidates`, `configuration_candidates`, `exact_configuration_resolved`, `inference_status`, `parent1_supported`, `parent2_supported`, `observed_parent_count_minimum`, `observed_parent_count_maximum`, `local_winner_parent_state`, `selected_parent_state`, `local_parent1`, `local_parent2`, `graph_conflict`, `state_winner_margin`, `conditional_identity_margin`, `local_state_bootstrap_fraction`, `local_state_loco_fraction`, `eligible_m2_pairs`, `scored_m2_pairs`, `incomplete_m2_search`, `graph_changed_local_configuration`, `interpretation`, `tier_b_final_state_call`, `tier_b_release_evidence` |
| `pedigree/alternative_explanations.csv` | `sample`, `parent_state`, `parent1`, `parent2`, `local_winner`, `graph_winner`, `within_state_rank`, `genetic_log_evidence`, `structural_state_log_evidence`, `structural_identity_eligible`, `direction_log_contribution`, `reciprocal_family_log_contribution`, `ancestry_path_log_contribution`, `adjusted_state_log_evidence`, `adjusted_identity_log_evidence`, `parent_state_log_prior`, `log_eligible_identity_multiplicity`, `marginal_state_log_evidence`, `graph_decision_utility` |
| `pedigree/search_diagnostics.csv` | `sample`, `eligible_parents`, `scored_m2_pairs`, `m2_panel_parents`, `screen_boundary_rank`, `screen_boundary_gap`, `omitted_parent_count`, `strongest_omitted_parent`, `parent1_screen_rank`, `parent2_screen_rank`, `selected_parent_at_screen_boundary` |
| `pedigree/exclusion_evidence.csv` | `sample`, `candidate_parent`, `log_e`, `observed_sites` |
| `pedigree/exclusion_diagnostics.csv` | `sample`, `eligible_candidates`, `compatible_candidates`, `supported_reverse_candidates`, `remaining_candidates`, `weakest_exclusion_log_e`, `threshold_log_e`, `complete_candidate_evidence`, `released`, `release_conditional_on_supported_directions` |
| `pedigree/evidence_summary.csv` | `contig`, `informative_markers`, `mean_minimum_ancestry_junctions`, `mean_callable_haplotype_bins` |
| `pedigree/chromosome_summary.csv` | `contig`, `component_count`, `scored_component_count`, `informative_markers`, `omitted_reason` |
| `pedigree/relative_contrasts.csv` | `sample`, `candidate_parent`, `views`, `proxy_model`, `m1_minus_second_degree_proxy_nats`, `m1_minus_full_sib_proxy_nats`, `usable_chromosomes`, `negative_chromosomes_second_degree`, `negative_chromosomes_full_sib`, `min_chromosome_contrast_second_degree` |
| `pedigree/sibling_guard.csv` | `parent`, `child`, `supported_members`, `dropped_linked`, `dropped_half_sib`, `dropped_members`, `tested_members`, `excluded_members`, `call`, `evidence_call`, `withdrawal_iteration`, `withholding_basis` |
| `recombination/coverage.csv` | `contig`, `parent`, `child`, `parent_slot`, `left_bp`, `right_bp`, `component` |
| `recombination/crossovers.csv` | `contig`, `parent`, `child`, `parent_slot`, `left_bp`, `right_bp`, `left_origin_probability`, `right_origin_probability`, `joint_endpoint_probability_lower_bound`, `component`, `reported_left_bp`, `reported_right_bp`, `in_reversed_segment` |
| `recombination/junctions.csv` | `contig`, `locus`, `left_bp`, `right_bp`, `calls_left_bp`, `calls_right_bp`, `calls`, `parents_with_calls`, `odd_meioses`, `decidable_meioses`, `switch_fraction`, `flip_adjusted_switch_fraction`, `hotspot_ceiling`, `mixed_parents`, `all_or_none_parents`, `parents`, `children`, `mixing_log_lr`, `paired_locus`, `paired_odd_meioses`, `pairing_log_lr`, `pairing_tests`, `junction_log_lr`, `flagged`, `interpretation`, `nearest_supplied_breakpoint`, `breakpoint_distance_bp` |
| `recombination/map.csv` | `contig`, `left_bp`, `right_bp`, `crossover_count`, `exposure_meiosis_bp`, `expected_crossovers`, `called_exposure_meiosis_bp`, `called_rate_c_m_per_mb`, `covered_bp`, `effective_meioses`, `rate_c_m_per_mb`, `c_m_on_covered_bp`, `cumulative_observed_c_m_at_right` |
| `recombination/meioses.csv` | `contig`, `parent`, `child`, `parent_slot`, `released_chain_bins`, `followed_chain_bins`, `unresolved_chain_bins`, `called_crossovers`, `expected_crossovers`, `exposure_meiosis_bp` |
| `recombination/summary.csv` | `contig`, `meioses`, `observable_meioses`, `crossovers`, `expected_crossovers`, `unresolved_bins`, `exposure_meiosis_bp`, `covered_bp`, `cumulative_observed_c_m`, `uncovered_bins`, `order_breakpoints`, `order_breakpoint_masked_sites`, `junction_loci`, `junction_loci_flagged`, `family_phase_converged`, `elapsed_seconds` |
| `relatedness/families.csv` | `sample`, `family`, `family_support`, `parent1`, `parent1_support`, `parent2`, `parent2_support` |
| `relatedness/pairs.csv (optional; off by default)` | `sample_a`, `sample_b`, `ibd0`, `ibd1`, `ibd2`, `sites`, `ibd0_evidence`, `ibd2_evidence` |
| `relatedness/half_sib_links.csv` | `family_a`, `family_b`, `gain` |
| `simulation/truth_pedigree.csv` | `sample`, `cohort`, `parent1`, `parent2` |
| `founder_refit/summary.csv` | `contig`, `components`, `founder_rows`, `sites`, `called_alleles`, `changed_alleles`, `ambiguous_alleles`, `filled_alleles`, `changed_inference_alleles`, `changed_sites`, `median_changed_gain_nats`, `excluded_samples` |
| `family_phase/summary.csv` | `contig`, `samples`, `sites`, `components`, `sweeps`, `settled`, `objective`, `junction_candidates`, `withheld_joins`, `tie_breaks`, `called_alleles`, `phased_heterozygotes`, `unphased_heterozygotes`, `phase_sets`, `elapsed_seconds` |
| `ties/decisions.csv` | `contig`, `stage`, `records`, `kind`, `decisions`, `inside_tie`, `wide`, `boundary` |

Relationship/candidate tables insert `parent_slot_basis` after `parent2` when
available. Exclusion evidence appends `log_e_lower`, `log_e_upper`, `log_e_exact`;
exclusion diagnostics appends `weakest_exclusion_log_e_lower`,
`weakest_exclusion_log_e_upper`, `weakest_exclusion_log_e_exact`. These suffixes
retain their lower/upper/exact meaning.

Assembly-search columns are the first-seen union across contigs as well as
records. Its prefix is always present. The closing-record fields and their
meaning are listed above; event-specific columns are:

`accepted`, `additional_seconds`, `alternatives`, `branch_k`, `branch_ll`, `certified`, `change`, `changes`, `complete`, `continued`, `cycle_excluded`, `cycle_excluded_panels`, `exact`, `kind`, `minimum_switches`, `objective`, `observed_panels`, `panel`, `panel_size`, `panels`, `proposed_paths`, `reason`, `replay_choices`, `representative_prior`, `represented_paths`, `resolution_seconds`, `resolved`, `source`, `source_choices`, `source_decision`, `source_replay_complete`, `source_search_certified`, `stage`, `status`, `survivors`, `unresolved_search`, `values`.

Each event keeps its recorded operand value; a field absent from a record is an
empty cell. Nested object keys inside these CSV cells retain their original names.

<!-- output-columns:end -->

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
`founder` (the panel row as text), `predicted_row`, `truth_founder`,
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


Collection-file schemas are generated alongside the parent columns:
`painting/assembly_search_shadow_alternatives.csv`, `painting/assembly_search_items.csv`,
`pedigree/cell_items.csv`
and `recombination/junction_members.csv`. Each has its header even with no items.
