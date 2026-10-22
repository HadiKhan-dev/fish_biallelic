# Migration guide (release 1.0 to 2.0)

Release 2.0 runs the same science as release 1.0 (commit 92b3daf) in a rebuilt package. Its commands, settings,
stage names and output names changed; this page maps each old form to the new one. The
[changelog](../CHANGELOG.md) lists the behaviour changes. Checkpoints of 1.0 are refused: start every run in a new
output root.

## 2.7 assembly checkpoints

The completion-retained, refinement-off pipeline requires a new output root for any earlier
refinement-enabled run. Preflight checks both assembly schema 3 and its configuration: a prior
experimental completion-drop candidate also used schema 3, so the number alone does not identify
this pipeline. Historical results remain readable with their original version of the package.

## Commands

The four reconstruction commands are one `run` command; a dataset is a configuration (`preset`), not a command.
`run.py` is gone: use `haplotypes` or `python -m haplotype_reconstruction`. Commands take run directories as
positional arguments.

| old | new |
| --- | --- |
| `python run.py reconstruct --vcf F --output D --contigs C1 C2 C3` | `haplotypes run --vcf F --output D --contigs C1 C2 C3` |
| `python run.py simulate --config S` | `haplotypes run S` (with `[input] kind = "simulation"`) |
| `python run.py simulate --templates T ...` | `haplotypes run --templates T --contigs C1 C2 C3` |
| `python run.py simulate --vcf F` (templates built from a VCF on the fly) | `haplotypes templates --vcf F --output T`, then `haplotypes run --templates T --contigs C1 C2 C3` |
| `python run.py tropheops --config configs/tropheops.toml` | `haplotypes run CONFIG` with `preset = "tromau"` plus the data paths |
| `python run.py astcal --config configs/astcal.toml` | `haplotypes run CONFIG` with `preset = "aulstu"` plus the data paths |
| `recombination --config C --output D --checkpoints K --order-breakpoints F` | `haplotypes recombination RUN --order-breakpoints F` |
| `evaluate --output D [--checkpoints K]` | `haplotypes evaluate RUN` |
| `export --output D --destination E` | `haplotypes export RUN E` |
| `status --output D [--checkpoints K]` | `haplotypes status RUN [--checkpoints K]` |
| `example --output D` | `haplotypes example D` |
| `batch-worker --queue Q` | `haplotypes batch-worker Q` |
| `recombination ... --recombination-map M --rate-cm-per-mb X` | none: `haplotypes recombination RUN` uses the run's own genetic map; only `--set recombination.KEY=VALUE` and `--order-breakpoints FILE` change |
| `recombination ... --shared-family-evidence` / `--no-shared-family-evidence` | `haplotypes recombination RUN --set recombination.shared_family_evidence=true` / `false` |

### Export

Every complete run now ends with an export stage that writes the final products to `OUTPUT/export/`
([usage](usage.md#export)); `haplotypes export RUN DEST` writes the same files for a finished run.

| old | new |
| --- | --- |
| `export --products founders` (per-component founder-track TSVs) | `haplotypes export RUN DEST --products tracks` (`founders` is the founder VCF) |
| `export` defaults: founder-track TSVs and a kept-site BCF of the samples | founder VCF and sample VCF, bgzipped with tabix indexes, every input biallelic SNP that reached the site filter (FILTER = PASS or the rules that removed it) |
| the 1.0 export layout | `haplotypes export RUN DEST --products tracks samples --format bcf --no-full-length` |
| `export --synthetic-alleles` | none: a simulation's alleles are always written as A (0) and C (1) |
| `export --checkpoints K` | none: the checkpoints are found from `RUN` |

A configuration that the TroMau or AulStu commands found in code (contigs, workbook columns, eligibility design,
reference report) is in the presets; the data paths that were hard-coded in the drivers must now be given:

<!-- check_docs: config -->
```toml
preset = "aulstu"

[input]
vcf = "data/AsAc.AulStuGenome.biallelic.bcf.gz"

[samples]
metadata = "data/AsAc_metadata.xlsx"
```

## Command-line options

`run` keeps the execution and input options; scientific switches use `--set SECTION.KEY=VALUE` (any key of the
[configuration](configuration.md)) instead of one flag each.

| old | new |
| --- | --- |
| `--config FILE` | positional `CONFIG` of `haplotypes run` |
| `--output D`, `--checkpoints K`, `--cores N`, `--vcf F`, `--templates T`, `--contigs C ...`, `--max-contigs N`, `--process-contigs C ...`, `--batch-queue Q` | unchanged names: `--output`, `--checkpoints`, `--cores`, `--vcf`, `--templates`, `--contigs`, `--max-contigs`, `--process-contigs`, `--batch-queue` |
| `--stop-after-stage block_discovery\|painting` | `--stop-after discovery\|painting` |
| `--serial-contigs` / `--no-serial-contigs` | `--serial` / `--set execution.serial=false` |
| `--contig-workers N` | `--set execution.contig_lanes=N` |
| `--read-calibration` / `--no-read-calibration` | `--set quality.read_calibration=true` / `false` |
| `--site-quality-filter` / `--no-site-quality-filter` | `--set quality.site_filter=on` / `off` |
| `--sample-quality-filter` / `--no-sample-quality-filter` | `--set quality.sample_filter=on` / `off` |
| `--pedigree-calibration predictive\|off` | `--set pedigree.evidence_calibration=predictive` / `off` |
| `--founder-refinement on\|off` | none: founder-path refinement is disabled in 2.7 |
| `--recombination-map FILE` | `--set genetic_map.file=FILE` |
| `--rate-cm-per-mb X` | `--set genetic_map.rate_cm_per_mb=X` |
| `--shared-family-evidence` / `--no-shared-family-evidence` | `--set recombination.shared_family_evidence=true` / `false` |
| `--order-breakpoints FILE` (simulate, reconstruct, astcal, tropheops) | `--set recombination.order_breakpoints=FILE` (`--order-breakpoints` stays on `haplotypes recombination`) |
| `--eligibility FILE` | `--set pedigree.eligibility.policy=constraints --set pedigree.eligibility.constraints=FILE` |
| `--metadata FILE`, `--metadata-sheet S` | `--set samples.metadata=FILE`, `--set samples.sheet=S` |
| `--seed N`, `--depth X` | `--set simulation.seed=N`, `--set simulation.depth=X` |
| `--generations A B C` | `--set 'simulation.cohorts=[A, B, C]'` |
| `--backcross-fraction`, `--backcross-start-generation`, `--observe-generations`, `--observed-fraction` | `--set simulation.backcross_fraction=X`, `simulation.backcross_start_cohort=N`, `simulation.observe_cohorts=[...]`, `simulation.observed_fraction=X` |
| `--generating-error-rate`, `--depth-cv`, `--heterozygote-alt-probability`, `--dropout-fraction` | `--set simulation.reads.error_rate=X`, `simulation.reads.depth_cv=X`, `simulation.reads.heterozygote_alt_probability=X`, `simulation.reads.dropout_fraction=X` |
| `--generating-map FILE`, `--generating-rate-cm-per-mb X` | `--set simulation.generating_map=FILE`, `--set simulation.generating_rate_cm_per_mb=X` |
| `evaluate --stages block_discovery feedback_initial ...` | `haplotypes evaluate RUN --stages discovery local_selection_initial` (stage names below) |

## Configuration files

Sections are named after what they configure; `[run]` holds only where results go and where to stop. Unknown keys
and sections are errors (1.0 ignored them silently), so a 1.0 file must be converted.

| old | new |
| --- | --- |
| `[run] output`, `[run] checkpoints` | `[run] output`, `[run] checkpoints` |
| `[run] stop_after_stage = "block_discovery"` | `[run] stop_after = "discovery"` (or `"painting"`) |
| `[run] cores`, `[run] max_contigs`, `[run] process_contigs`, `[run] batch_queue` | `[execution] cores`, `[execution] max_contigs`, `[execution] process_contigs`, `[execution] batch_queue` |
| `[run] contig_workers`, `[run] serial_contigs` | `[execution] contig_lanes`, `[execution] serial` |
| `[inputs] vcf`, `[inputs] contigs` | `[input] vcf`, `[input] contigs` (and `[input] kind`) |
| `[inputs] templates` | `[simulation] templates` |
| `[inputs] metadata`, `[inputs] metadata_sheet` | `[samples] metadata`, `[samples] sheet` |
| `[inputs] eligibility` | `[pedigree.eligibility] policy = "constraints"` and `constraints` |
| eligibility designs built into the tropheops and astcal drivers | `[pedigree.eligibility] policy = "cohorts"`, `parent_generations`, `opposite_sex_pairs` (in the presets) |
| `[run] read_calibration`, `[run] site_quality_filter`, `[run] sample_quality_filter` | `[quality] read_calibration`, `[quality] site_filter`, `[quality] sample_filter` |
| `[run] pedigree_calibration` | `[pedigree] evidence_calibration` |
| `[run] founder_refinement` | none (disabled in 2.7) |
| `[recombination] rate_cm_per_mb`, `[recombination] recombination_map` | `[genetic_map] rate_cm_per_mb`, `[genetic_map] file` |
| `[recombination] shared_family_evidence`, `[recombination] order_breakpoints` | `[recombination] shared_family_evidence`, `[recombination] order_breakpoints` |
| environment `BHD_FAMILY_PHASE_PROTECTED_INTERVALS` | `[family_phase] protected_intervals` |
| `[simulation] generations`, `backcross_start_generation`, `observe_generations` | `[simulation] cohorts`, `[simulation] backcross_start_cohort`, `[simulation] observe_cohorts` |
| `[simulation] generating_error_rate`, `depth_cv`, `heterozygote_alt_probability`, `dropout_fraction`, `read_length_bp`, `fragment_mean_bp`, `fragment_sd_bp` | `[simulation.reads] error_rate`, `[simulation.reads] depth_cv`, `[simulation.reads] heterozygote_alt_probability`, `[simulation.reads] dropout_fraction`, `[simulation.reads] read_length_bp`, `[simulation.reads] fragment_mean_bp`, `[simulation.reads] fragment_sd_bp` |
| `[simulation] seed`, `depth`, `backcross_fraction`, `observed_fraction`, `generating_map`, `generating_rate_cm_per_mb` | `[simulation] seed`, `[simulation] depth`, `[simulation] backcross_fraction`, `[simulation] observed_fraction`, `[simulation] generating_map`, `[simulation] generating_rate_cm_per_mb` |
| `[simulation] seed = "none"` | `[simulation] seed = "random"` |
| `[run] founder_scaling`, `[refinement] impute_missing`, `[run] assembly_model`, `assembly_search`, `discovery_search`, `feedback_selection`, `feedback_segment_exchange` | none: unknown keys are errors; delete them |

New settings: `preset`, `[input] kind`, `[execution] memory_limit` (`--memory-limit`), `[execution] profile`, the
`[samples]` columns (`id_column`, `generation_column`, `sex_column`, `reference_generation`,
`generation_prefixes`, `generations`, `sexes`), `[reports] reference_consistency`, and `[export] enabled` and
`[export] full_length`. `[input] contigs` is required for VCF input unless a preset lists
them (the 1.0 `reconstruct` command took every header contig). Paths still resolve against the working
directory.

## Environment variables

No setting is read from the environment any more; every one is in the configuration.

| old | new |
| --- | --- |
| `HAPLOTYPES_*` settings (`OUTPUT_DIR`, `CHECKPOINT_DIR`, `VCF`, `CONTIGS`, `TEMPLATES`, `METADATA`, `METADATA_SHEET`, `ELIGIBILITY`, `STOP_AFTER_STAGE`, `BATCH_QUEUE`, `CONTIG_WORKERS`, `SERIAL_CONTIGS`, `MAX_CONTIGS`, `READ_CALIBRATION`, `SITE_QUALITY_FILTER`, `SAMPLE_QUALITY_FILTER`, `PEDIGREE_CALIBRATION`, `FOUNDER_REFINEMENT`, `GENERATIONS`, `SIMULATION_DESIGN`, `READ_MODEL`) | the corresponding TOML key or `--set` |
| `BHD_*` settings (`NUM_PROCESSES`, `RECOMBINATION_MAP`, `RECOMBINATION_RATE_CM_PER_MB`, `RECOMBINATION_SHARED_FAMILY`, `RECOMBINATION_ORDER_BREAKPOINTS`, `SIMULATION_SEED`, `SIM_READ_DEPTH`, `SIM_CONTIGS`, `SIM_STOP_AFTER_STAGE`, `SIM_CHECKPOINT_DIR`, `SIM_OUTPUT_DIR`, `SIMULATION_RECOMBINATION_MAP`, `SIMULATION_RECOMBINATION_RATE_CM_PER_MB`) | the corresponding TOML key or `--set` |
| `BHD_FAMILY_PHASE_PROTECTED_INTERVALS` | `[family_phase] protected_intervals` |
| `HAPREC_COARSE_MOVE_GUARD` | none: founder-path refinement is disabled in 2.7 |
| `BHD_HIER_PROFILE`, `BHD_HIER_PROFILE_MIN_L`, `BHD_DYNTHREAD_LOG` | `[execution] profile = true` |
| `BHD_MALLOC_TRIM_RSS_MB`, `BHD_MALLOC_TRIM_EVERY_BLOCKS` | none: heap trimming happens at fixed phase boundaries |
| `NUMBA_THREADING_LAYER_OVERRIDE` | none: the threading layer is OpenMP |

The environment facts the package still reads (scheduler, scratch and cache locations) are listed in
[configuration.md](configuration.md#environment).

## Stage names

One vocabulary names a stage everywhere: its record prefix, its output directory, its stop point and its
evaluation stage.

| old | new |
| --- | --- |
| `block_discovery` (stop point, evaluation stage) | `discovery` |
| record `discovery/prepared` (evidence inside discovery) | stage `evidence`, record `evidence` |
| `feedback/initial`, `feedback/l1`, `feedback/l2`; evaluation `feedback_initial`, `feedback_l1`, `feedback_l2` | stage `local_selection`, records `local_selection/initial`, `local_selection/l1`, `local_selection/l2`; evaluation `local_selection_initial`, `local_selection_l1`, `local_selection_l2` |
| `assembly`, `assembly_l1` ... `assembly_l4`, `founder_refinement` | `assembly` is the final joined product; `assembly_l1` ... `assembly_l4` are joined levels; `founder_refinement` is no longer offered |
| `allele_refit` | `founder_refit` |
| `refinement` (package), `phase_correction` (outputs) | `family_phase` |
| `recombination_map` (outputs) | `recombination` |
| `simulation/global`, `simulation/templates` | `simulation/manifest` (run file); templates are read from the template directory |
| (the `export` command only) | stage `export` after recombination: record `export` per contig, `complete/export`, `OUTPUT/export/` |

## Output files

| old | new |
| --- | --- |
| `logs/<command>_<stamp>_<pid>.log` | `logs/run_<stamp>_<pid>.log` (`recombination_<stamp>_<pid>.log` for the recombination command) |
| `logs/reconstruction_<contig>_<stamp>_<pid>.log` | `logs/contigs/reconstruction_<contig>_<stamp>_<pid>.log` |
| run records with the environment settings | `run_records/<stamp>_<pid>.json` with the resolved configuration; `config.toml` holds it too |
| `sample_quality/<contig>.tsv`, `sample_quality/<contig>.segments.tsv` | `quality/<contig>.samples.tsv`, `quality/<contig>.segments.tsv` |
| `reference_consistency.csv` | `discovery/reference_consistency.csv` |
| `reports/painting/<contig>/*.tsv` | `painting/<contig>/*.tsv` (same five tables) |
| `pedigree/scientific_relationships.csv` (a copy of the Tier-B table) | `pedigree/tier_b_relationships.csv` |
| `pedigree/*.csv`, `pedigree/evidence_calibration.json` | unchanged names |
| `allele_refit/summary.csv` | `founder_refit/summary.csv` |
| `phase_correction/family_phase.csv` | `family_phase/summary.csv` |
| `recombination_map/<contig>.{map,crossovers,coverage,called_coverage,meioses,junctions}.csv`, `<contig>.png`, `junctions.csv`, `summary.csv` | `recombination/` with the same file names |
| `ground_truth_pedigree.csv` (columns `Sample`, `Generation`, `Parent1`, `Parent2`) | `simulation/truth_pedigree.csv` (columns `sample`, `cohort`, `parent1`, `parent2`) |
| `evaluation/*` | `evaluation/*` (same files; stage names as above) |
| (exports only from the `export` command) | `export/<contig>.founders.vcf.gz`, `export/<contig>.samples.vcf.gz` with tabix indexes and `export/manifest.json`, written by every complete run |

Columns that changed:

- Pedigree tables: the constant `Generation` column of the relationship tables and constant, duplicate or relict
  diagnostics columns are gone (the list is in the docstring of `pedigree/tables.py`); `chromosome_summary.csv`
  has `Contig`, `ComponentCount`, `ScoredComponentCount`, `InformativeMarkers`, `OmittedReason`.
- `family_phase/summary.csv`: one row per contig with `contig`, `samples`, `sites`, `map_sweeps`, `converged`,
  `map_objective` (release 2.1; 2.0 wrote `iterations` and `final_residual`), `called_alleles`, `changed_observable_phase_sites`, `root_frame_moves`, `root_frame_reverts`,
  `root_frame_transfers`, `elapsed_seconds`; the constant policy columns and `final_checkpoint` are gone (the phase
  is the contig file's `family_phase` record).
- `painting/<contig>/founder_counts.tsv` (`Join*`, `FeedbackJoin*` columns) and `assembly_search.tsv` count each
  join search once, at the level where it ran (1.0 repeated a component's search at later levels it passed
  through unchanged). An empty count cell means that no count step or join search covers the component.
- `founder_refit/summary.csv` has no `minimum_gain_nats` or `elapsed_seconds` column and adds `excluded_samples`.

The protected-interval table (formerly `BHD_FAMILY_PHASE_PROTECTED_INTERVALS`) now needs the header line
`chrom start end reason`, like the order-breakpoint table.

## Checkpoints and Python callers

Checkpoint directories of 1.0 are refused: a 2.0 run marks its checkpoint directory with its layout before it
writes anything, and a directory holding checkpoints without that mark stops the run at once ("written by an
earlier release ... use a new output root"). The container format is unchanged, but the record keys and identities
are new (see [development.md](development.md#checkpoint-store)); rerun into a new output root.

Python callers find the code under stage packages; `core/` and `workflows/` are gone:

| old | new |
| --- | --- |
| `core.checkpoints`, `core.runtime.CheckpointStore` | `store` (`store.store.CheckpointStore`, `store.container.CheckpointFile`) |
| `core.parallel`, `core.chromosome_parallel`, `workflows.contig_parallel` | `execution` (`CpuBudget`, `BudgetedPool`, `ChromosomeExecutor`, contig lanes, memory governor) |
| `core.variants`, `core.read_calibration`, `core.site_quality`, `core.sample_quality` | `evidence` |
| `discovery.path_*`, `workflows.block_feedback` | `local_selection` |
| `painting.allele_refit` | `founder_refit` |
| `refinement` | `family_phase` |
| `simulation.evaluation`, `simulation.metrics`, `simulation.founder_metrics` | `evaluation` |
| `workflows.export`, `workflows.reports`, `assembly.support` | `export`, `reports` |
| `core.products.load_panels(file, stage)` | each stage's `records.founder_panels(file)` |
| `core.genetic_map` | `model.genetic_map` |

## Founder release in 2.3

The founder export manifest schema is `released-allele-export-v3`. A column named
`c<C>_H<k>` identifies row `k` of release segment `C` on that chromosome. Release segments
can subdivide an inference component. Use `source_component` and `segment_index` in
`founder_components` to relate them to assembly and painting; the corresponding VCF
`##component` fields are `SourceComponent` and `SegmentIndex`. `source_founder_rows`
maps displayed rows to the original painting/refit row indices. Identical `H0` labels
in different segments do not assert a common founder identity.

Track filenames retain `<chromosome>.componentNNNN.founders.tsv.gz`, but `NNNN` now
enumerates release segments. An `ambiguity` record gives its source level, half-open
prepared-block span, source decision, status, alternative count, representative prior,
and `uncertainty_interval` in 1-based observed SNP positions. A continuity interval
brackets supported boundaries; it is not a precise crossover estimate. Whole-panel
row permutations alone do not create breaks.

For equal-K terminal alternatives, exact invariant flanks/path anchors can establish
one common row correspondence. Export retains every agreeing known source allele cell
and withholds cells that differ or are missing in any evaluated member. The mapping
uses neither approximate matching nor truth. For unequal-K terminal families, a
displayed path occurring exactly once in every evaluated full-span panel also has
an exact correspondence. Its known prepared cells are retained despite changes in
other paths' local multiplicity. Duplicate paths and clipped fragments do not qualify.
This protection applies within each family before masks from independent families
are combined, and preserves prepared missingness and all continuity cuts.
If unique correspondence is unavailable,
varying duplicate multiplicities receive the same conservative local-row mask under
every representative. Existing founder-refit and carrier-observation unknown calls are
preserved. Full-length r5 filling cannot restore withheld content or bridge segments.

The separate `count_unverified` list reports incomplete computational verification of
count decisions. Entries include cause, domain size, evaluated count, source decision,
block span and observed-position interval. VCF headers expose them as
`##founder_count_verification`. These flags preserve calls and ordinary r5 filling;
they do not establish biological disagreement.

The separate `continuation_unverified` list preserves evaluated source families whose
later continuation was not completed. Its records include `evaluated_source_panels`,
`evaluated_source_founder_counts`, actual `local_disagreement_intervals`, cause, source
scope and stated prior. Source panels are not presented as equally supported final
panels. VCF/BCF headers expose compact `##founder_continuation_verification` records.
Calls remain conditional on the incumbent. Only independently established observed
content disagreement can cause withholding; an unperformed continuation alone cannot.

Per contig, `founder_continuity_breaks` counts additional boundaries within source
components. `assembly_ambiguous_alleles` and segment `withheld_ambiguous_alleles` count
newly withheld panel cells after the existing carrier-observation mask. They exclude
already missing calls and r5 sites outside the panel. Retained/withheld counts must be
reported against same-input baseline called cells, alongside total called/missing cells
and missing-inclusive founder errors. Count and continuation verification record counts
refer to source records once per contig, even if they annotate several segments.

Both `haplotypes export RUN DEST` and pipeline export use this release reader. Older
assembly products without recorded alternative families report
`assembly_ambiguity_metadata = "legacy_unknown"`. Re-export cannot reconstruct discarded
alternatives or establish that no alternatives existed; recompute assembly under 2.3
for the new treatment. Export identity includes assembly records and release code.
Regenerate the merged configuration reference after integrating all owners' changes.

Painting, pedigree, sample phase and recombination remain conditional on the assembly
representative where alternatives are unresolved. Segmentation does not marginalize
those models or change sample VCF calls and phase sets by itself.
