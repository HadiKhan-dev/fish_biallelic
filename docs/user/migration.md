# Migration guides

## Migrating from 4.6 to 4.7.0

Start a new output and checkpoint root with the same inputs, configuration and
seed. **4.7.0 adds a kernel and a pedigree setting, so its checkpoint identities
differ from 4.6's and it requires a new root.** Keep existing runs with their
matching code; this release does not convert their checkpoints.

Released outputs are scientifically identical to 4.6 except where the new per-child
cap on scored parent pairs binds (very low depth only; see below):
- founder refinement skips a count-deletion round when a sample-summed optimistic
  bound proves its decision (keep the larger panel, not ambiguous) for every
  candidate, so it runs fewer refits;
- painting builds a component's emissions once for its copying-model estimate;
- family phase computes its allele-evidence terms with fewer array passes.

One table changes: `search_diagnostics` has a new last column, `M2PanelCapped`. It is
true for a child whose admitted parent pairs exceeded `max_pairs_per_child` (256),
so its two-parent evidence covers an incomplete panel (an internal pedigree
parameter, listed in the parameter reference). Count-decision records name the
deletion bound `deletion_summed`.

## Migrating from 4.5 to 4.6.0

Start a new output and checkpoint root with the same inputs, configuration and
seed. **4.6.0 changes selection code and requires a new root.** Keep existing runs
with their matching code; this release does not convert their checkpoints.
Configurations, commands, record schemas and table layouts are those of 4.5.

Local selection is faster, mostly at low depth, and takes a different optimisation
path to its panels. Its fixed points, objective and decisions are unchanged:
- one likelihood pass per GEM update instead of two (frequencies and allele flips from
  one E-step), and several improving one-bit flips at once at stationary points and in
  context snaps;
- accelerated frequency refits at a fixed panel, and short probes that rank many
  candidate-bank starts before the full fits;
- exact one-bit gains computed with one logarithm per founder and site.

Selections are therefore not bit-identical to 4.5. In replays of recorded blocks,
they are identical at 1x and above. Below 1x, a growing share of block rounds changes
(3 of 84 at 0.5x, most at 0.1x), more often for the better under the selection
objective. Downstream stages consume the new selections as before.

## Migrating from 4.4 to 4.5.0

Start a new output and checkpoint root with the same inputs, configuration and
seed. **4.5.0 changes code and requires a new root.** Keep existing runs with their
matching code; this release does not convert their checkpoints.

From 4.5 onward, a release-label-only change preserves stage identities. The
release value lives in a lazily loaded module outside scientific dependency
closures. Identities still bind scientific code and its transitive dependencies,
consumed inputs, parameters, tie settings, codecs and numerical policy. A changed
scientific dependency still invalidates its products and their downstream
consumers. The full source digest remains in invocation provenance;
distribution metadata still exposes the release label. A version label alone is neither a compatibility test nor a reason
to discard checkpoints.

Output choices now live in `output_selection.json` beside published tables.
`report` reads checkpoint roots without writing into them; its completion marker
is `OUTPUT/report.done`. See the [workflow guide](workflow.md) for republishing
from an archive and the shared-root multi-node recipe.

Run records (`run_records/*.json`, schema `run-record-v5`) keep per-stage task aggregates; the per-task
detail stays in the `*.tasks.jsonl` log beside each record, which is now retained after completion.
`status` reads the aggregates. Readers of the old embedded `tasks` list should read that log.
A run interrupted by SIGTERM or SIGINT is recorded as `interrupted` (exit status 130), not `failed`.

Simulation configurations simulate exactly as before. `[simulation.reads]` gains opt-in
`site_depth_cv`, `site_depth_window_bp`, `count_dispersion` and `preset = "realistic"`.

The published table layout changes explicitly (4.5.0 includes the table changes prepared as 4.4.2,
which was not released separately):

- `painting/assembly_search.csv` gains `assembly_search_row`, a zero-based ordinal
  within each contig, and `shadow_alternatives` contains a count instead of a JSON
  list. Join `painting/assembly_search_shadow_alternatives.csv` on `contig` and
  `assembly_search_row`, sort by `alternative`, and JSON-decode `value` to recover
  every original alternative dictionary in order. Empty lists have count zero;
  absent fields keep empty cells. Level/span alone do not uniquely identify rows.
- Symmetric and plateau alternative lists in assembly search move to
  `painting/assembly_search_items.csv`; parent list cells become `{"items": COUNT}`.
  Scalar census cells in the same columns remain scalar. Reconstruct lists by
  parent key and column, ordering by `item` and JSON-decoding `value`.
- Cohort-sized collection cells in `tier_b_candidate_sets`, `call_explanations`,
  `diagnostics`, `exclusion_diagnostics` and `sibling_guard` become counts.
  `pedigree/cell_items.csv` carries the parent table, sample (or parent/child),
  column, item index and JSON value. The [output guide](outputs.md#collection-tables)
  lists the fields and their reconstruction rules.
- `recombination/junctions.csv` parent/child lists become counts.
  `recombination/junction_members.csv` preserves every ordered text item, keyed
  by contig, locus and column. Join values with semicolons to recover the cell.

- `founder_refit/summary.csv:excluded_samples` becomes a count.
  `founder_refit/excluded_samples.csv` carries each sample, keyed by contig and
  zero-based item index, preserving the original list order.

All other published cells keep their existing values. These are report-layout
changes; the scientific records retain their full collections. The CSV field-limit
fix remains for trusted per-contig parts. Failed writes remove their private
staging files without replacing an existing complete destination.


## Migrating from 4.4.0 to 4.4.1

Start a new run with a new output and checkpoint root using the same inputs,
configuration and seed, as for any release: checkpoint identities bind the
package version. The models, settings, record schemas and table layouts are
those of 4.4.0, and the published values are the same. A 4.4.0 run that stopped
with `field larger than field limit (131072)` while publishing tables has to be
rerun under 4.4.1; its 4.4.0 checkpoints cannot be resumed by 4.4.1.

## Migrating from 4.3 to 4.4

Start a new run with a new output and checkpoint root using the same inputs,
configuration and seed. Keep the original 4.3 run and its outputs; finish or
inspect that run with its matching 4.3 code. Do not point a 4.4 resume at a 4.3
checkpoint root or rename old records into the new root. There is no automatic
in-place derivation upgrade.

Evidence publications now use schema 2 and may store exact joint read-pair/GL
codebooks. Low-level evidence adapters can still decode schema 1 when explicitly
supplied its original expected derivation. The public results reader instead
rebuilds the current derivation from code, schema and bound inputs, so format
support does not make an old 4.3 derivation compatible with 4.4. Downstream
identities also bind their evidence inputs and must be regenerated in the new run.

Scientific models, thresholds and released table layouts retain 4.3's meaning.
No new memory or evidence-encoding setting is required: storage admission uses
actual joint bytes and projections use bounded workspaces. Full-pipeline release
validation remains subject to the 4.4 release gate recorded in the changelog.

## Migrating from 4.2 to 4.3

Start a new run and checkpoint root. Recombination records are schema 2 and
export completion records are schema 4; scientific reconstruction and release
rules are unchanged. Re-export a 4.3 run with `haplotypes export RUN DEST`; it
uses the same CSV/track layout as an ordinary run. A 4.2 checkpoint is not upgraded
by renaming output files. Keep the original 4.2 outputs for comparison.

All tables use CSV quoting and snake_case columns. Former per-contig tables
gain a leading `contig`; rows follow configured contig order and then their
existing order. Only large VCF/BCF exports, indexes and optional tracks remain
per contig. CamelCase splits mechanically at case boundaries, with
`FirstVariant1Based` → `first_variant_1based` and `SampleID` → `sample`. Shared
identifier aliases are `component_id`/`export_component` → `component` and
`founder_row`/`founder_key` → `founder`; qualified source identifiers keep their names.
JSON keys and nested object keys inside CSV cells keep their names.

The complete mapping follows. A lone column name is unchanged; arrows name
every renamed column. Unavailable optional views remain absent.

| 4.2 path | 4.3 path | Column mapping |
|---|---|---|
| `quality/CONTIG.samples.tsv` | `quality/samples.csv` | add leading `contig` from the source contig; `sample`; `observed_cells`; `eligible_cells`; `eligible_windows`; `minor_cells`; `near_private_cells`; `lrt`; `foreign_fraction`; `foreign_share`; `artefact_share`; `artefact_peak`; `heterozygote_fit`; `heterozygote_dispersion`; `error_fit`; `flagged` |
| `quality/CONTIG.segments.tsv` | `quality/segments.csv` | add leading `contig` from the source contig; `sample`; `partner`; `start`; `end`; `events`; `cells`; `minor_read_fraction`; `partner_minor_read_fraction` |
| `painting/CONTIG/sample_segments.tsv` | `painting/sample_segments.csv` | add leading `contig` from the source contig; `SampleID` → `sample`; `Component` → `component`; `Track` → `track`; `Start` → `start`; `End` → `end`; `Status` → `status`; `Label` → `label`; `Rows` → `rows` |
| `painting/CONTIG/label_continuity.tsv` | `painting/label_continuity.csv` | add leading `contig` from the source contig; `Component` → `component`; `AtomicSpan` → `atomic_span`; `Position` → `position`; `RowA` → `row_a`; `RowB` → `row_b`; `ScoreGainSwitches` → `score_gain_switches`; `PaintedSwitchChange` → `painted_switch_change`; `Winners` → `winners`; `Losers` → `losers`; `WinnersAtoB` → `winners_ato_b`; `WinnersBtoA` → `winners_bto_a`; `Identity` → `identity` |
| `painting/CONTIG/painting_model.tsv` | `painting/painting_model.csv` | add leading `contig` from the source contig; `Component` → `component`; `FirstVariant1Based` → `first_variant_1based`; `LastVariant1Based` → `last_variant_1based`; `Bins` → `bins`; `States` → `states`; `ErrorCapableClasses` → `error_capable_classes`; `SwitchRatePerMorgan` → `switch_rate_per_morgan`; `ErrorEntry` → `error_entry`; `ErrorExit` → `error_exit`; `ErrorRunBinsKeepingLabel` → `error_run_bins_keeping_label`; `SiteCorrelation` → `site_correlation`; `EligibleSamples` → `eligible_samples` |
| `painting/CONTIG/founder_support.tsv` | `painting/founder_support.csv` | add leading `contig` from the source contig; `Component` → `component`; `Founder` → `founder`; `FirstVariant1Based` → `first_variant_1based`; `LastVariant1Based` → `last_variant_1based`; `BreakBefore` → `break_before`; `BreakAfter` → `break_after`; `MarkerCount` → `marker_count`; `CalledAlleles` → `called_alleles`; `MissingAlleles` → `missing_alleles`; `ObservedCarrierSamples` → `observed_carrier_samples`; `CalledSitesWithoutObservedNamedCarrier` → `called_sites_without_observed_named_carrier`; `MinimumObservedCarriers` → `minimum_observed_carriers`; `MeanObservedCarriers` → `mean_observed_carriers`; `ObservedPooledCandidateSamples` → `observed_pooled_candidate_samples`; `MeanObservedPooledCandidates` → `mean_observed_pooled_candidates`; `CalledSitesWithoutObservedNamedOrPooledCandidate` → `called_sites_without_observed_named_or_pooled_candidate`; `SupportBasis` → `support_basis` |
| `painting/CONTIG/founder_support_windows.tsv` | `painting/founder_support_windows.csv` | add leading `contig` from the source contig; `Component` → `component`; `Founder` → `founder`; `FirstVariant1Based` → `first_variant_1based`; `LastVariant1Based` → `last_variant_1based`; `MarkerCount` → `marker_count`; `CalledAlleles` → `called_alleles`; `MissingAlleles` → `missing_alleles`; `CalledSitesWithoutObservedNamedCarrier` → `called_sites_without_observed_named_carrier`; `MeanObservedCarriers` → `mean_observed_carriers`; `MeanObservedPooledCandidates` → `mean_observed_pooled_candidates`; `CalledSitesWithoutObservedNamedOrPooledCandidate` → `called_sites_without_observed_named_or_pooled_candidate` |
| `painting/CONTIG/assembly_search.tsv` | `painting/assembly_search.csv` | add leading `contig` from the source contig; `Level` → `level`; `FirstVariant1Based` → `first_variant_1based`; `LastVariant1Based` → `last_variant_1based` |
| `painting/CONTIG/founder_counts.tsv` | `painting/founder_counts.csv` | add leading `contig` from the source contig; `Component` → `component`; `FirstVariant1Based` → `first_variant_1based`; `LastVariant1Based` → `last_variant_1based`; `FounderRows` → `founder_rows`; `FounderCountAmbiguous` → `founder_count_ambiguous`; `UndecidedCountCycles` → `undecided_count_cycles`; `CountStepFoundersBefore` → `count_step_founders_before`; `CountStepCycle` → `count_step_cycle`; `CountStepCycleUndecided` → `count_step_cycle_undecided`; `CountStepAmbiguousDecisions` → `count_step_ambiguous_decisions`; `CountStepDecisions` → `count_step_decisions`; `JoinCountCycles` → `join_count_cycles`; `JoinUndecidedCycles` → `join_undecided_cycles`; `JoinAmbiguousCountCalls` → `join_ambiguous_count_calls`; `FeedbackJoinCountCycles` → `feedback_join_count_cycles`; `FeedbackJoinUndecidedCycles` → `feedback_join_undecided_cycles`; `FeedbackJoinAmbiguousCountCalls` → `feedback_join_ambiguous_count_calls` |
| `painting/CONTIG/sample_switches.tsv` | `painting/sample_switches.csv` | add leading `contig` from the source contig; `SampleIndex` → `sample_index`; `SampleID` → `sample`; `PaintedComponents` → `painted_components`; `PaintedSwitches` → `painted_switches`; `CohortMedianSwitches` → `cohort_median_switches`; `SwitchesOverCohortMedian` → `switches_over_cohort_median` |
| `pedigree/tier_b_relationships.csv` | `pedigree/tier_b_relationships.csv` | `Sample` → `sample`; `Parent1` → `parent1`; `Parent2` → `parent2`; `ParentState` → `parent_state`; `ObservedParentCount` → `observed_parent_count`; `InferenceStatus` → `inference_status`; `ReleaseEvidence` → `release_evidence` |
| `pedigree/tier_b_partial_relationships.csv` | `pedigree/tier_b_partial_relationships.csv` | `Sample` → `sample`; `Parent1` → `parent1`; `Parent2` → `parent2`; `ParentState` → `parent_state`; `ObservedParentCount` → `observed_parent_count`; `InferenceStatus` → `inference_status`; `Parent1Supported` → `parent1_supported`; `Parent2Supported` → `parent2_supported`; `ExactConfigurationResolved` → `exact_configuration_resolved`; `ObservedParentCountMinimum` → `observed_parent_count_minimum`; `ObservedParentCountMaximum` → `observed_parent_count_maximum`; `ReleaseEvidence` → `release_evidence` |
| `pedigree/tier_a_relationships.csv` | `pedigree/tier_a_relationships.csv` | `Sample` → `sample`; `Parent1` → `parent1`; `Parent2` → `parent2`; `ParentState` → `parent_state`; `ObservedParentCount` → `observed_parent_count`; `InferenceStatus` → `inference_status` |
| `pedigree/tier_a_partial_relationships.csv` | `pedigree/tier_a_partial_relationships.csv` | `Sample` → `sample`; `Parent1` → `parent1`; `Parent2` → `parent2`; `ParentState` → `parent_state`; `ObservedParentCount` → `observed_parent_count`; `InferenceStatus` → `inference_status`; `Parent1Supported` → `parent1_supported`; `Parent2Supported` → `parent2_supported`; `ExactConfigurationResolved` → `exact_configuration_resolved`; `ObservedParentCountMinimum` → `observed_parent_count_minimum`; `ObservedParentCountMaximum` → `observed_parent_count_maximum` |
| `pedigree/complete_relationships.csv` | `pedigree/complete_relationships.csv` | `Sample` → `sample`; `Parent1` → `parent1`; `Parent2` → `parent2`; `ParentState` → `parent_state`; `ObservedParentCount` → `observed_parent_count`; `InferenceStatus` → `inference_status` |
| `pedigree/tier_b_candidate_sets.csv` | `pedigree/tier_b_candidate_sets.csv` | `Sample` → `sample`; `ParentState` → `parent_state`; `ObservedParentCount` → `observed_parent_count`; `Parent1` → `parent1`; `Parent2` → `parent2`; `Parent1Candidates` → `parent1_candidates`; `Parent2Candidates` → `parent2_candidates`; `ConfigurationCandidates` → `configuration_candidates`; `ExactConfigurationResolved` → `exact_configuration_resolved`; `InferenceStatus` → `inference_status`; `Parent1Supported` → `parent1_supported`; `Parent2Supported` → `parent2_supported`; `ObservedParentCountMinimum` → `observed_parent_count_minimum`; `ObservedParentCountMaximum` → `observed_parent_count_maximum`; `TierBFinalStateCall` → `tier_b_final_state_call`; `TierBReleaseEvidence` → `tier_b_release_evidence` |
| `pedigree/parent_state_calls.csv` | `pedigree/parent_state_calls.csv` | `Sample` → `sample`; `SelectedParentState` → `selected_parent_state`; `LocalWinnerParentState` → `local_winner_parent_state`; `ObservedParentCount` → `observed_parent_count`; `LocalObservedParentCount` → `local_observed_parent_count`; `InferenceStatus` → `inference_status`; `DAGDisplaced` → `dag_displaced`; `GraphConflict` → `graph_conflict`; `GraphTieConflict` → `graph_tie_conflict`; `GraphDirectionResolvedAlternative` → `graph_direction_resolved_alternative`; `SelectedParentRoleProbability` → `selected_parent_role_probability`; `TierAStateCall` → `tier_a_state_call`; `TierBStateCall` → `tier_b_state_call`; `TierBFinalStateCall` → `tier_b_final_state_call`; `TierBReleaseEvidence` → `tier_b_release_evidence` |
| `pedigree/diagnostics.csv` | `pedigree/diagnostics.csv` | `Sample` → `sample`; `M0PriorSensitivityStateAgreement` → `m0_prior_sensitivity_state_agreement`; `M0PriorSensitivityGraphAgreement` → `m0_prior_sensitivity_graph_agreement`; `M0PriorSensitivityTierVeto` → `m0_prior_sensitivity_tier_veto`; `StructureChildEvaluable` → `structure_child_evaluable`; `StructureSelectedRowExposureTestable` → `structure_selected_row_exposure_testable`; `StructureSelectedRowCXCompatible` → `structure_selected_row_cx_compatible`; `StructureParent1Coverage` → `structure_parent1_coverage`; `StructureParent2Coverage` → `structure_parent2_coverage`; `StructurePairExplainability` → `structure_pair_explainability`; `StructureParent1DirectionProbability` → `structure_parent1_direction_probability`; `StructureParent2DirectionProbability` → `structure_parent2_direction_probability`; `BestM1Parent` → `best_m1_parent`; `BestM1ConditionalIdentityMargin` → `best_m1_conditional_identity_margin`; `BestM2Parent1` → `best_m2_parent1`; `BestM2Parent2` → `best_m2_parent2`; `BestM2ConditionalIdentityMargin` → `best_m2_conditional_identity_margin`; `LocalParent1` → `local_parent1`; `LocalParent2` → `local_parent2`; `EligibleChild` → `eligible_child`; `EligibleParentCount` → `eligible_parent_count`; `EligibleParentPairCount` → `eligible_parent_pair_count`; `CompleteParent1` → `complete_parent1`; `CompleteParent2` → `complete_parent2`; `SelectedParentState` → `selected_parent_state`; `LocalWinnerParentState` → `local_winner_parent_state`; `ObservedParentCount` → `observed_parent_count`; `LocalObservedParentCount` → `local_observed_parent_count`; `Identifiable` → `identifiable`; `DAGDisplaced` → `dag_displaced`; `GraphConflict` → `graph_conflict`; `GraphTieConflict` → `graph_tie_conflict`; `GraphDirectionResolvedAlternative` → `graph_direction_resolved_alternative`; `LocalParentRoleProbability` → `local_parent_role_probability`; `SelectedParentRoleProbability` → `selected_parent_role_probability`; `RawAncestryJunctionCount` → `raw_ancestry_junction_count`; `CallableAncestryHaplotypeBinCount` → `callable_ancestry_haplotype_bin_count`; `AdjustedAncestryJunctionBurden` → `adjusted_ancestry_junction_burden`; `AncestryPaintingCallabilityFraction` → `ancestry_painting_callability_fraction`; `InformativeContigCount` → `informative_contig_count`; `InformativeMapUnits` → `informative_map_units`; `LocalStateBootstrapFraction` → `local_state_bootstrap_fraction`; `LocalConfigurationBootstrapFraction` → `local_configuration_bootstrap_fraction`; `GraphConfigurationBootstrapFraction` → `graph_configuration_bootstrap_fraction`; `SelectedGraphConfigurationBootstrapFraction` → `selected_graph_configuration_bootstrap_fraction`; `SelectedGraphParent1BootstrapFraction` → `selected_graph_parent1_bootstrap_fraction`; `SelectedGraphParent2BootstrapFraction` → `selected_graph_parent2_bootstrap_fraction`; `Parent1BootstrapFraction` → `parent1_bootstrap_fraction`; `Parent2BootstrapFraction` → `parent2_bootstrap_fraction`; `LocalStateLOCOFraction` → `local_state_loco_fraction`; `LocalConfigurationLOCOFraction` → `local_configuration_loco_fraction`; `GraphConfigurationLOCOFraction` → `graph_configuration_loco_fraction`; `SelectedGraphConfigurationLOCOFraction` → `selected_graph_configuration_loco_fraction`; `SelectedGraphParent1LOCOFraction` → `selected_graph_parent1_loco_fraction`; `SelectedGraphParent2LOCOFraction` → `selected_graph_parent2_loco_fraction`; `Parent1LOCOFraction` → `parent1_loco_fraction`; `Parent2LOCOFraction` → `parent2_loco_fraction`; `StateWinnerMargin` → `state_winner_margin`; `ConditionalIdentityMargin` → `conditional_identity_margin`; `UnconstrainedWinnerMargin` → `unconstrained_winner_margin`; `SelectedAggregateUtility` → `selected_aggregate_utility`; `SelectedMinusUnconstrainedBest` → `selected_minus_unconstrained_best`; `StateLogEvidence0` → `state_log_evidence0`; `StateLogEvidence1` → `state_log_evidence1`; `StateLogEvidence2` → `state_log_evidence2`; `StateSupport0` → `state_support0`; `StateSupport1` → `state_support1`; `StateSupport2` → `state_support2`; `ScoredCandidateCount2` → `scored_candidate_count2`; `M2CandidateScreenIncomplete` → `m2_candidate_screen_incomplete`; `PriorSensitivityLocalStateAgreementFraction` → `prior_sensitivity_local_state_agreement_fraction`; `PriorSensitivityLocalIdentityAgreementFraction` → `prior_sensitivity_local_identity_agreement_fraction`; `PriorSensitivityGraphAgreementFraction` → `prior_sensitivity_graph_agreement_fraction`; `PriorSensitivitySelectedStateMinimumSupport` → `prior_sensitivity_selected_state_minimum_support`; `InferenceStatus` → `inference_status`; `TierAStateCall` → `tier_a_state_call`; `TierBStateCall` → `tier_b_state_call`; `TierAExactConfiguration` → `tier_a_exact_configuration`; `TierBExactConfiguration` → `tier_b_exact_configuration`; `TierAExactPair` → `tier_a_exact_pair`; `TierBExactPair` → `tier_b_exact_pair`; `TierAParent1` → `tier_a_parent1`; `TierAParent2` → `tier_a_parent2`; `TierBParent1` → `tier_b_parent1`; `TierBParent2` → `tier_b_parent2`; `ConfigurationSupportSet` → `configuration_support_set`; `Parent1CandidateSet` → `parent1_candidate_set`; `Parent2CandidateSet` → `parent2_candidate_set`; `EvidenceConfigurationSet` → `evidence_configuration_set`; `TierBFinalStateCall` → `tier_b_final_state_call`; `TierBReleaseEvidence` → `tier_b_release_evidence` |
| `pedigree/call_explanations.csv` | `pedigree/call_explanations.csv` | `Sample` → `sample`; `ParentState` → `parent_state`; `ObservedParentCount` → `observed_parent_count`; `Parent1` → `parent1`; `Parent2` → `parent2`; `Parent1Candidates` → `parent1_candidates`; `Parent2Candidates` → `parent2_candidates`; `ConfigurationCandidates` → `configuration_candidates`; `ExactConfigurationResolved` → `exact_configuration_resolved`; `InferenceStatus` → `inference_status`; `Parent1Supported` → `parent1_supported`; `Parent2Supported` → `parent2_supported`; `ObservedParentCountMinimum` → `observed_parent_count_minimum`; `ObservedParentCountMaximum` → `observed_parent_count_maximum`; `LocalWinnerParentState` → `local_winner_parent_state`; `SelectedParentState` → `selected_parent_state`; `LocalParent1` → `local_parent1`; `LocalParent2` → `local_parent2`; `GraphConflict` → `graph_conflict`; `StateWinnerMargin` → `state_winner_margin`; `ConditionalIdentityMargin` → `conditional_identity_margin`; `LocalStateBootstrapFraction` → `local_state_bootstrap_fraction`; `LocalStateLOCOFraction` → `local_state_loco_fraction`; `EligibleM2Pairs` → `eligible_m2_pairs`; `ScoredM2Pairs` → `scored_m2_pairs`; `IncompleteM2Search` → `incomplete_m2_search`; `GraphChangedLocalConfiguration` → `graph_changed_local_configuration`; `Interpretation` → `interpretation`; `TierBFinalStateCall` → `tier_b_final_state_call`; `TierBReleaseEvidence` → `tier_b_release_evidence` |
| `pedigree/alternative_explanations.csv` | `pedigree/alternative_explanations.csv` | `Sample` → `sample`; `ParentState` → `parent_state`; `Parent1` → `parent1`; `Parent2` → `parent2`; `LocalWinner` → `local_winner`; `GraphWinner` → `graph_winner`; `WithinStateRank` → `within_state_rank`; `GeneticLogEvidence` → `genetic_log_evidence`; `StructuralStateLogEvidence` → `structural_state_log_evidence`; `StructuralIdentityEligible` → `structural_identity_eligible`; `DirectionLogContribution` → `direction_log_contribution`; `ReciprocalFamilyLogContribution` → `reciprocal_family_log_contribution`; `AncestryPathLogContribution` → `ancestry_path_log_contribution`; `AdjustedStateLogEvidence` → `adjusted_state_log_evidence`; `AdjustedIdentityLogEvidence` → `adjusted_identity_log_evidence`; `ParentStateLogPrior` → `parent_state_log_prior`; `LogEligibleIdentityMultiplicity` → `log_eligible_identity_multiplicity`; `MarginalStateLogEvidence` → `marginal_state_log_evidence`; `GraphDecisionUtility` → `graph_decision_utility` |
| `pedigree/search_diagnostics.csv` | `pedigree/search_diagnostics.csv` | `Sample` → `sample`; `EligibleParents` → `eligible_parents`; `ScoredM2Pairs` → `scored_m2_pairs`; `M2PanelParents` → `m2_panel_parents`; `ScreenBoundaryRank` → `screen_boundary_rank`; `ScreenBoundaryGap` → `screen_boundary_gap`; `OmittedParentCount` → `omitted_parent_count`; `StrongestOmittedParent` → `strongest_omitted_parent`; `Parent1ScreenRank` → `parent1_screen_rank`; `Parent2ScreenRank` → `parent2_screen_rank`; `SelectedParentAtScreenBoundary` → `selected_parent_at_screen_boundary` |
| `pedigree/exclusion_evidence.csv` | `pedigree/exclusion_evidence.csv` | `Sample` → `sample`; `CandidateParent` → `candidate_parent`; `LogE` → `log_e`; `ObservedSites` → `observed_sites` |
| `pedigree/exclusion_diagnostics.csv` | `pedigree/exclusion_diagnostics.csv` | `Sample` → `sample`; `EligibleCandidates` → `eligible_candidates`; `CompatibleCandidates` → `compatible_candidates`; `SupportedReverseCandidates` → `supported_reverse_candidates`; `RemainingCandidates` → `remaining_candidates`; `WeakestExclusionLogE` → `weakest_exclusion_log_e`; `ThresholdLogE` → `threshold_log_e`; `CompleteCandidateEvidence` → `complete_candidate_evidence`; `Released` → `released`; `ReleaseConditionalOnSupportedDirections` → `release_conditional_on_supported_directions` |
| `pedigree/evidence_summary.csv` | `pedigree/evidence_summary.csv` | add leading `contig` from the source contig; `Contig` → `contig`; `InformativeMarkers` → `informative_markers`; `MeanMinimumAncestryJunctions` → `mean_minimum_ancestry_junctions`; `MeanCallableHaplotypeBins` → `mean_callable_haplotype_bins` |
| `pedigree/chromosome_summary.csv` | `pedigree/chromosome_summary.csv` | add leading `contig` from the source contig; `Contig` → `contig`; `ComponentCount` → `component_count`; `ScoredComponentCount` → `scored_component_count`; `InformativeMarkers` → `informative_markers`; `OmittedReason` → `omitted_reason` |
| `pedigree/relative_contrasts.csv` | `pedigree/relative_contrasts.csv` | `Sample` → `sample`; `CandidateParent` → `candidate_parent`; `Views` → `views`; `ProxyModel` → `proxy_model`; `M1MinusSecondDegreeProxyNats` → `m1_minus_second_degree_proxy_nats`; `M1MinusFullSibProxyNats` → `m1_minus_full_sib_proxy_nats`; `UsableChromosomes` → `usable_chromosomes`; `NegativeChromosomesSecondDegree` → `negative_chromosomes_second_degree`; `NegativeChromosomesFullSib` → `negative_chromosomes_full_sib`; `MinChromosomeContrastSecondDegree` → `min_chromosome_contrast_second_degree` |
| `pedigree/sibling_guard.csv` | `pedigree/sibling_guard.csv` | `Parent` → `parent`; `Child` → `child`; `SupportedMembers` → `supported_members`; `DroppedLinked` → `dropped_linked`; `DroppedHalfSib` → `dropped_half_sib`; `DroppedMembers` → `dropped_members`; `TestedMembers` → `tested_members`; `ExcludedMembers` → `excluded_members`; `Call` → `call`; `EvidenceCall` → `evidence_call`; `WithdrawalIteration` → `withdrawal_iteration`; `WithholdingBasis` → `withholding_basis` |
| `recombination/CONTIG.coverage.csv` | `recombination/coverage.csv` | add leading `contig` from the source contig; `parent`; `child`; `parent_slot`; `left_bp`; `right_bp`; `component_id` → `component` |
| `recombination/CONTIG.crossovers.csv` | `recombination/crossovers.csv` | add leading `contig` from the source contig; `parent`; `child`; `parent_slot`; `left_bp`; `right_bp`; `left_origin_probability`; `right_origin_probability`; `joint_endpoint_probability_lower_bound`; `component_id` → `component`; `reported_left_bp`; `reported_right_bp`; `in_reversed_segment` |
| `recombination/junctions.csv` | `recombination/junctions.csv` | `contig`; `locus`; `left_bp`; `right_bp`; `calls_left_bp`; `calls_right_bp`; `calls`; `parents_with_calls`; `odd_meioses`; `decidable_meioses`; `switch_fraction`; `flip_adjusted_switch_fraction`; `hotspot_ceiling`; `mixed_parents`; `all_or_none_parents`; `parents`; `children`; `mixing_log_lr`; `paired_locus`; `paired_odd_meioses`; `pairing_log_lr`; `pairing_tests`; `junction_log_lr`; `flagged`; `interpretation`; `nearest_supplied_breakpoint`; `breakpoint_distance_bp` |
| `recombination/CONTIG.map.csv` | `recombination/map.csv` | add leading `contig` from the source contig; `left_bp`; `right_bp`; `crossover_count`; `exposure_meiosis_bp`; `expected_crossovers`; `called_exposure_meiosis_bp`; `called_rate_cM_per_Mb` → `called_rate_c_m_per_mb`; `covered_bp`; `effective_meioses`; `rate_cM_per_Mb` → `rate_c_m_per_mb`; `cM_on_covered_bp` → `c_m_on_covered_bp`; `cumulative_observed_cM_at_right` → `cumulative_observed_c_m_at_right` |
| `recombination/CONTIG.meioses.csv` | `recombination/meioses.csv` | add leading `contig` from the source contig; `parent`; `child`; `parent_slot`; `released_chain_bins`; `followed_chain_bins`; `unresolved_chain_bins`; `called_crossovers`; `expected_crossovers`; `exposure_meiosis_bp` |
| `recombination/summary.csv` | `recombination/summary.csv` | `contig`; `meioses`; `observable_meioses`; `crossovers`; `expected_crossovers`; `unresolved_bins`; `exposure_meiosis_bp`; `covered_bp`; `cumulative_observed_cM` → `cumulative_observed_c_m`; `uncovered_bins`; `order_breakpoints`; `order_breakpoint_masked_sites`; `junction_loci`; `junction_loci_flagged`; `family_phase_converged`; `elapsed_seconds` |
| `relatedness/families.csv` | `relatedness/families.csv` | `Sample` → `sample`; `Family` → `family`; `FamilySupport` → `family_support`; `Parent1` → `parent1`; `Parent1Support` → `parent1_support`; `Parent2` → `parent2`; `Parent2Support` → `parent2_support` |
| `relatedness/pairs.csv` | `relatedness/pairs.csv` | `SampleA` → `sample_a`; `SampleB` → `sample_b`; `IBD0` → `ibd0`; `IBD1` → `ibd1`; `IBD2` → `ibd2`; `Sites` → `sites`; `IBD0Evidence` → `ibd0_evidence`; `IBD2Evidence` → `ibd2_evidence` |
| `relatedness/half_sib_links.csv` | `relatedness/half_sib_links.csv` | `FamilyA` → `family_a`; `FamilyB` → `family_b`; `Gain` → `gain` |
| `simulation/truth_pedigree.csv` | `simulation/truth_pedigree.csv` | `sample`; `cohort`; `parent1`; `parent2` |
| `founder_refit/summary.csv` | `founder_refit/summary.csv` | `contig`; `components`; `founder_rows`; `sites`; `called_alleles`; `changed_alleles`; `ambiguous_alleles`; `filled_alleles`; `changed_inference_alleles`; `changed_sites`; `median_changed_gain_nats`; `excluded_samples` |
| `family_phase/summary.csv` | `family_phase/summary.csv` | `contig`; `samples`; `sites`; `components`; `sweeps`; `settled`; `objective`; `junction_candidates`; `withheld_joins`; `tie_breaks`; `called_alleles`; `phased_heterozygotes`; `unphased_heterozygotes`; `phase_sets`; `elapsed_seconds` |
| `ties/decisions.csv` | `ties/decisions.csv` | `contig`; `stage`; `records`; `kind`; `decisions`; `inside_tie`; `wide`; `boundary` |
| `export/CONTIG.founder_identity.tsv` | `export/founder_identity.csv` | `contig`; `column`; `export_component` → `component`; `founder_row` → `founder`; `source_stage`; `source_component`; `source_row`; `block`; `left_position`; `right_position`; `left_local_row`; `right_local_row`; `left_partners`; `right_partners`; `at_release_break`; `reason`; `method`; `status` |
| `export/CONTIG.componentNNNN.founders.tsv.gz` | `export/CONTIG.tracks.csv.gz` | `CHROM` → `contig`; add `component` = NNNN; `POS` → `pos`; `Hk` → `hk`; empty padding for absent rows |
| `recombination/CONTIG.called_coverage.csv` | `recombination/coverage.csv` | Same columns and cells as CONTIG.coverage.csv; retain one copy |
| `recombination/CONTIG.junctions.csv` | `recombination/junctions.csv` | Drop duplicate per-contig copies; retain the existing run-level rows |
| `recombination/CONTIG.png` | `recombination/maps.pdf` | One unchanged figure per page, configuration contig order |
| `export/CONTIG.founders.vcf.gz (+ index), CONTIG.samples.vcf.gz (+ index)` | `unchanged` | VCF fields and sample/founder columns unchanged |
| `export/manifest.json` | `unchanged path` | JSON keys unchanged; schema value released-allele-export-v3 → v4; file paths updated |
| `pedigree/evidence_calibration.json, pedigree/summary.json, relatedness/summary.json, ties/summary.json, evaluation/summary.json` | `unchanged` | JSON keys unchanged |
| `evaluation/chromosomes.csv, phase_groups.csv, founders.csv, founder_components.csv, founder_matches.csv` | `unchanged paths` | founder_matches: `founder_key` → `founder`; other columns unchanged |

Optional pedigree columns: `ParentSlotBasis` → `parent_slot_basis`;
`LogELower`, `LogEUpper`, `LogEExact` → `log_e_lower`, `log_e_upper`, `log_e_exact`;
`WeakestExclusionLogELower`, `WeakestExclusionLogEUpper`, `WeakestExclusionLogEExact`
→ `weakest_exclusion_log_e_lower`, `weakest_exclusion_log_e_upper`,
`weakest_exclusion_log_e_exact`. Assembly-search event/closing-record columns
already in snake_case keep their names; its case-split prefix is in the table.

Coverage stores `callable_spans` and `callable_offsets` once. The duplicate
`informative_spans`/`informative_offsets` fields and posterior-kernel return are
removed; record readers, map aggregation, junctions and evaluation use callable
spans. `coverage.csv` denotes spans between consecutive released markers in a
followed stretch, not all observed sites.

Tracks use the largest segment's founder count for the header. A shorter segment
has empty cells in unused `hk` columns; an existing unknown allele remains `.`.
Segment-local identity, alleles and missingness are unchanged.

## Migrating from 3.5 to 4.0
**Start a new run and checkpoint root. 4.0 does not resume 3.5 records.** The rebuild
preserves the carried models and release rules, with the explicitly documented
numerical conventions and output changes below. A translated TOML file is not a
checkpoint conversion.

## Translate the configuration

Keep the original file and write the translated configuration separately:

```sh
haplotypes migrate-config cross35.toml -o cross40.toml
```

Without `-o`, the translated TOML and its per-key audit comments are printed to
standard output:

```sh
haplotypes migrate-config cross35.toml
```

The translator validates the entire result before writing. It either preserves a
supported value under an explicit translation or refuses with a reason; unknown
keys are not silently ignored. Review the audit comments and set `[run] output`
and `[run] checkpoints` to fresh locations before initialising:

```sh
haplotypes run cross40.toml --cores 8
```

Paths remain relative to the invocation's working directory. Moving the translated
TOML alone does not change their resolution. The generated [settings reference](../reference/settings.md)
and [command reference](../reference/commands.md) describe the accepted 4.0 schema.

| 3.5 input or convention | 4.0 action |
|---|---|
| Omitted/version-3 configuration version | Translator writes `version = 4` |
| Carried settings | Same keys and values are kept and listed in the audit |
| Empty `execution.process_contigs` | Explicitly dropped; scatter work uses workflow step commands |
| Nonempty `execution.process_contigs` | Refused; rewrite the orchestration with `evidence`, `reconstruct` and `phase` |
| Empty `execution.batch_queue` | Explicitly dropped; helper queues are replaced by workflow mode |
| Nonempty `execution.batch_queue` | Refused; use a workflow manager |
| `aulstu` preset | Refused; choose `generic` and specify the desired contigs yourself |
| Sample metadata, eligibility or parent-label sections | Refused; the pipeline reads no sample metadata |
| Unrecognised setting or unsupported value | Refused with the field and reason |

The two presets are `generic` and `tromau`. A preset does not infer known parentage
from sample names. Translation does not add metadata-based eligibility or adjust
scientific values to force a configuration to load.

## Commands and workflow mode

`run CONFIG` initialises a new run automatically, preserving the single-command
entry point. Explicit `init` binds a run before separate workflow step commands.
`run` and the scatter/gather commands execute the same model graph. The [Snakemake example](workflow.md) replaces
shards and the helper batch queue; `batch-worker`, `--process-contigs` and
`--batch-queue` are not 4.0 commands/options. Clean stop points and `--serial` remain
available as graph cuts.

Standalone `haplotypes export RUN DEST` **has no `--vcf` option**. It formats checked
ReleasedSites, FounderRelease and PhasedSamples records through the results reader,
including source REF/ALT and contig metadata. The original input file need not be
present for re-export. `--vcf` still belongs to input-consuming commands such as
`run` and `templates`; its removal is specific to standalone export.

`haplotypes recombination RUN` remains available. It uses the recorded final phase
and recombination binding. Once a contig has a mapped publication, changing its
recombination settings is refused, including in a copy of a completed run: copying
preserves the binding. Alternative settings need an **independent checkpoint root**
and valid prerequisites for that analysis. Copying a run or renaming an output
directory does not create an independent scientific identity.

## Pairwise IBD is an optional output

`relatedness/pairs.csv` is **off by default**. To retain the 3.5 pair report, explicitly
set:

```toml
[relatedness]
pairwise_ibd = true
```

The translator adds `pairwise_ibd = false` when the old configuration omits it and
explains that change in its audit. Read that note before assuming that all former
files will be emitted.

The pair chain costs O(N² L); family fitting does not require it. Selection is
persisted per invocation in `output_selection.json` in the output directory and stays outside every scientific
identity. Switching the flag does not recompute families. When reports are written
with it off, a stale pairs CSV is removed even if the pair record remains in the
checkpoint store. Use the desired selection on the report invocation as well.

## Outputs and record consumers

Founder/sample VCF meanings, missing encodings, FILTER text, phase-set rules and
report cell encodings are preserved. Tables are now rendered from typed records.
The output dictionary includes relatedness tables, painting label-continuity/model
tables and the sibling-guard report; their omission from an older guide was not
an absence of those outputs.

| Item | 4.0 contract |
|---|---|
| Relatedness family numbers | Canonical dense ranks ordered by the family's smallest member (L6); match historical families by membership, not a former sparse integer |
| Assembly `source_decision` names | Deterministic names from the decision sequence (L7), not object identity |
| Phase record internals | `displayed_frame` and `decoder_frame` are not stored public products; use the released phase and ambiguity fields |
| Product/checkpoint schema | Typed, versioned and derivation-checked; no direct 3.5 pickle/checkpoint reuse |
| Export indexing | VCF chooses TBI/CSI by the last **written** position; BCF uses CSI |
| Sample ordering | Canonical name order in records, input order in published sample outputs |
| Standalone export destination | New or empty; re-export does not replace the original run's export record |

Use the supported [results readers](outputs.md#results-readers) instead of reading
private checkpoint entries or internal Python objects. The absence of a product in
a partial run is explicit; it is not reconstructed from a completion marker.

## Identity and numerical conventions

Input identity binds consumed content. Merely touching a file does not change it;
changing consumed AD, fallback GT, REF/ALT, contig length or loader drops does.
Scientific parameters, code dependencies, upstream identities and numerical-library
policy also enter derivation identities. Resource settings and optional report
selection do not. Completion markers describe verified publications; they never
replace their verification.

The [numerical ledger](../developer/ties.md#numerical-ledger) is narrow. It
replaces arithmetic that 3.5 computed in several forms, or in an order chosen by
workers, memory or object identity, with one declared form:

- L1: one Haldane recombination fraction, computed with `expm1`;
- L2: declared reduction orders, independent of worker counts and memory;
- L3: a stable content order for exact-duplicate exchanges;
- L4: family phase's chain test in float64, with float32 only for storage;
- L5: the shadow-domain proof attempted by a dimension rule, not by free memory;
- L6 and L7: canonical family numbers and decision-sequence names;
- L8: tie reversal and the tie census passed as settings, not environment variables;
- L9: a deterministic refinement work witness;
- L10: one direct two-row gain in founder refit;
- L11: the selection context refit's distances through the map;
- L12: one form of the pedigree's powered emissions;
- L13: one family log-likelihood function for relatedness moves;
- L14: the template linker without fast-math reassociation (only when templates
  are rebuilt);
- L15: unphased sample genotypes written in ascending allele order (`0/1`, never
  `1/0`), so their text no longer follows an arbitrary internal order.

No ledger entry changes a model, threshold, candidate set, search order or budget.
A cut-off crossing is reported with its margins and downstream effect rather than
hidden by a tolerance. The [changelog](../../CHANGELOG.md) records measured release
flip counts from the final validation receipts; an unmeasured count is not zero.

## Dependencies and cost exceptions

Use the release's tested Linux/Python dependency set rather than assuming the old
environment will produce the same numerical identity. The installed metadata and
`requirements-tested.txt` record the final envelope. The package does not require
a GPU. Declared dependency removals are listed in the changelog after the packaging
import audit.

The accepted 0, 0b and 0c search/cost exceptions, and the four above-cubic
worst cases of decision 0e, remain scientific limitations in 4.0, not performance
fixes applied during migration. See [limitations](limitations.md)
and the [cost table](../developer/costs.md), including K = 30. The optional pair
output is the explicit 0d output-contract change.

In 4.2, exemption 0b is removed: every eligible count hypothesis is refitted. CountRuleSettings and CountDecisionRecord use schema 2, without the old admission-rank fields; the decision record also drops RC14 cleanup history. Old count-decision payloads are rejected and code/parameter identities prevent their reuse. Recompute the affected assembly/refinement stages. The ordinary count rule replaces the zero-gain cleanup pre-pass. Preferred panels can change where the old admission screen omitted a better endpoint, even when the count verdict agrees.
