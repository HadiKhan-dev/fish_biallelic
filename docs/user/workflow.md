# Workflow mode

Each workflow command executes a cut of the same dependency graph used by `run`.
All jobs use the same configuration and shared checkpoint directory. Resource
options may differ by job. Scientific settings and input order must match the
initialised run.

| Step | Form | Required published inputs |
|---|---|---|
| Initialise once | `haplotypes init cross.toml` | Configuration and source observations/source configuration |
| Evidence per contig | `haplotypes evidence cross.toml --contig chr1` | Initialisation |
| Observation pilot and relatedness | `haplotypes relatedness cross.toml` | Evidence from every configured contig |
| Assemble per contig | `haplotypes reconstruct cross.toml --contig chr1` | That contig's evidence and the observation pilot published by `relatedness` |
| Calibration gather | `haplotypes calibrate cross.toml` | The final assembly of every analysed contig |
| Paint per contig | `haplotypes reconstruct cross.toml --contig chr1` (again) | The calibration. Publishes the painting and gain histogram for founder refit. |
| Pedigree on one node | `haplotypes pedigree cross.toml` | Relatedness, paintings, refit histograms and contamination statistics for all analysed contigs |
| Contamination decision | `haplotypes pedigree cross.toml --stage contamination` | Contamination statistics for every analysed contig |
| Founder refit priors | `haplotypes pedigree cross.toml --stage refit-priors` | Refit histograms for every analysed contig |
| Pedigree preparation per contig | `haplotypes pedigree cross.toml --stage prepare --contig chr1` | The two run-wide products above and that contig's painting and evidence. Publishes refit, repaint and preparation. |
| Map-scale inputs per contig | `haplotypes pedigree cross.toml --stage map-inputs --contig chr1` | Relatedness and that contig's repaint |
| Map-scale pair guard | `haplotypes pedigree cross.toml --stage map-pairs` | All map-scale contig inputs |
| Map-scale frames per contig | `haplotypes pedigree cross.toml --stage map-frames --contig chr1` | Pair guard and that contig's map-scale inputs |
| Pedigree gather | `haplotypes pedigree cross.toml --stage gather` | All pedigree preparations and map-scale frames |
| Phase per contig | `haplotypes phase cross.toml --contig chr1` | Pedigree and that contig's prerequisites; includes mapping and export |
| Reports gather | `haplotypes report cross.toml` | Available final records; after phase in the complete workflow |

Run `relatedness` before scattering reconstruction. It publishes the run-wide
observation pilot that discovery requires.

Run `reconstruct` twice per contig. The first invocation publishes the contig's
final assembly. `calibrate` uses all analysed contigs' final assemblies to
estimate the painting's outlier rate and evidence strengths. The second
`reconstruct` reuses the assembly and publishes the painting and founder-refit
gain histogram. `haplotypes run` schedules these steps automatically.

## Shared-directory protocol

`init` binds the resolved configuration, source content, sample/contig order and
realised seeds. A repeated identical initialisation is a no-op; a conflicting
binding is refused. Every other command requires initialisation and verifies the
binding. Simulation initialisation materialises all contigs before scatter jobs.

Tasks publish under derivation identities and use atomic claims. Duplicate
submissions of the same unit cannot replace a scientific publication with a
conflicting result. Scatter tasks write their own contig container; gather tasks
publish run-wide products. Invocation logs and run records remain distinct.

Workflow markers live in **CHECKPOINTS/markers**. With the default checkpoint
location this is `run/checkpoints/markers`, not `run/markers`. Global steps use
`init.done`, `relatedness.done`, `calibrate.done` and `pedigree.done`; contig steps use
`evidence.CONTIG.done`, `assembly.CONTIG.done` (a `reconstruct` before the calibration),
`reconstruct.CONTIG.done` (painting and refit histogram) and `phase.CONTIG.done`.
Pedigree scatter steps use `pedigree.STAGE.done` or `pedigree.STAGE.CONTIG.done`. The command
writes a marker only after validating the relevant products. If it dies between
publication and the marker, rerunning verifies the products and recreates the
marker. Never fabricate markers with `touch` as a substitute for running a step.

`report` only reads checkpoints. It writes tables, `output_selection.json`,
invocation records and `report.done` in the output directory. The report marker
is an output completion signal, not a checkpoint publication. To republish from
an archived root, choose a writable output and scratch directory:

```sh
haplotypes report cross.toml --checkpoints /archive/cross/checkpoints --output republished --scratch republished/scratch --cores 8
```

Keep the original scientific configuration and numerical environment. Optional
output choices apply to this publication; pair reports require an existing pair
publication. The original checkpoint files and markers remain unchanged.

## Snakemake example

The complete [Snakefile](../examples/Snakefile) reads `cross.toml` directly so its
contig list and checkpoint root match the CLI. Copy it next to that configuration
and run Snakemake from that directory:

```sh
snakemake --snakefile Snakefile --cores 8
```

The example gives each scientific command its rule's `threads` value through
`--cores`. Adjust those resource requests for your machine or workflow executor.
No scheduler, account or machine-specific paths are embedded. The marker path is
quoted and configured explicitly. Use ordinary filename-safe contig names in this
example; the package's own record identities do not depend on such file naming.

The matching serial shell order for the five-contig example in
[running a cross](running.md) is:

```sh
haplotypes init cross.toml --cores 8
for c in chr1 chr2 chr3 chr4 chr5; do
    haplotypes evidence cross.toml --contig "$c" --cores 8
done
haplotypes relatedness cross.toml --cores 8
for c in chr1 chr2 chr3 chr4 chr5; do
    haplotypes reconstruct cross.toml --contig "$c" --cores 8
done
haplotypes calibrate cross.toml --cores 8
for c in chr1 chr2 chr3 chr4 chr5; do
    haplotypes reconstruct cross.toml --contig "$c" --cores 8
done
haplotypes pedigree cross.toml --cores 8
for c in chr1 chr2 chr3 chr4 chr5; do
    haplotypes phase cross.toml --contig "$c" --cores 8
done
haplotypes report cross.toml --cores 8
```

This is orchestration of supported commands; it does not add a scientific stage.
The release workflow-equivalence check compares these cuts with `run`, including
published records and files. The documentation build checks the example against
the current parser and the integration test's recorded result.

## Several nodes, one checkpoint root

For large runs, scatter chromosomes to separate nodes, then let one `run`
invocation finish the genome-wide stages. Use a shared filesystem for the
configuration, inputs, checkpoint root and output directory. For example, set
`[run] output` and `checkpoints` to shared absolute paths in `cross.toml`.
Initialise once, then produce evidence for every configured chromosome:

```sh
haplotypes init cross.toml --cores 8
haplotypes evidence cross.toml --contig chr1 --cores 8
haplotypes evidence cross.toml --contig chr2 --cores 8
haplotypes evidence cross.toml --contig chr3 --cores 8
haplotypes evidence cross.toml --contig chr4 --cores 8
haplotypes evidence cross.toml --contig chr5 --cores 8
```

Schedule evidence commands concurrently after `init` exits successfully. After
all evidence commands finish, run `haplotypes relatedness cross.toml --cores 8`.
Its observation pilot is required by the reconstruction jobs, which can then
run concurrently on different allocated nodes:

```sh
haplotypes reconstruct cross.toml --contig chr1 --cores 8
haplotypes reconstruct cross.toml --contig chr2 --cores 8
haplotypes reconstruct cross.toml --contig chr3 --cores 8
haplotypes reconstruct cross.toml --contig chr4 --cores 8
haplotypes reconstruct cross.toml --contig chr5 --cores 8
```

Use each job's allocated CPU count instead of the illustrative eight cores. Start
long chromosomes early. Once every scatter job has exited, run the finishing
command on one allocated node:

```sh
haplotypes run cross.toml --cores 8
```

It reuses completed chromosome publications, calibrates the painting, paints
and refits every analysed chromosome, and finishes pedigree, phase,
recombination, export and reports. The scattered jobs above stop at each
chromosome's final assembly so that calibration can use all assemblies. After a
scatter job fails, let the started writers reach a terminal result and withdraw
unstarted jobs before restarting the scatter or resuming with the finishing
command.

To scatter the painting calibration, wait for every analysed chromosome's
final assembly, then run the following rounds in order. Use the same
`run.exclude_contigs` in the relatedness, reconstruction and calibration
commands. Evidence is produced for every configured contig. Wait for all jobs
in a round before starting the next.

1. For each analysed chromosome, run `haplotypes calibrate cross.toml --contig chrN`
   to fit the robustness grid.
2. For each analysed chromosome, run
   `haplotypes calibrate cross.toml --contig chrN --stage refine`
   to fit the rates chosen from the pooled robustness grid.
3. Run `haplotypes calibrate cross.toml --stage outlier` once to publish the
   run-wide outlier estimate.
4. With `[painting] foreign_read_share = "auto"` and multiple read classes, run
   `haplotypes calibrate cross.toml --contig chrN --stage share` for every
   analysed chromosome. Then run `haplotypes calibrate cross.toml --stage share`
   once to pool the likelihood profiles. A numerical share skips this round.
5. List the fit jobs with `haplotypes calibrate cross.toml --stage divisor --list-fits`.
   For read-state calibration, use `fold_fits` and run
   `haplotypes calibrate cross.toml --contig chrN --stage divisor --fold F`
   for each listed chromosome and zero-based fold. Each fold fits its candidates
   concurrently within the listed core count, sharing one emission builder.
   The `fits` list describes individual candidates. To run one, supply
   `--divisor D` in addition to the chromosome and fold. Single-class
   calibration uses these individual fits. Read-aware confidence calibration
   uses the chromosome prefix reaching `calibration_morgans` in configured order.
6. For every analysed chromosome, run
   `haplotypes calibrate cross.toml --contig chrN --stage divisor`.
   This gathers finished divisor fits and publishes an empty record for
   chromosomes outside the read-aware calibration prefix.
7. Run `haplotypes calibrate cross.toml` once to publish the pooled calibration.
   This command can also schedule the entire calibration on one node after assembly.

Run `haplotypes reconstruct cross.toml --contig chrN` again to paint and
publish the refit histogram and contamination statistics for each chromosome.
Then run these steps on the same checkpoint root.

1. Run `haplotypes pedigree cross.toml --stage contamination` and
   `haplotypes pedigree cross.toml --stage refit-priors` once each. These two
   reductions can run concurrently.
2. Submit `haplotypes pedigree cross.toml --stage prepare --contig chrN` for every
   analysed chromosome. Each job refits its founders, repaints them with the
   contamination decision, and publishes the chromosome's pedigree preparation.
3. Submit `haplotypes pedigree cross.toml --stage map-inputs --contig chrN` for
   every analysed chromosome, once its preparation job has finished.
4. Run `haplotypes pedigree cross.toml --stage map-pairs` after all map inputs are
   published. This freezes the initial set of families and targets for the map
   scale estimate.
5. Submit `haplotypes pedigree cross.toml --stage map-frames --contig chrN`
   for every analysed chromosome. Each job builds the first round of frames
   from the pair guard.
6. Run `haplotypes pedigree cross.toml --stage gather` once. This checks and
   gathers the published inputs and frames, computes the map-scale estimate,
   and infers the genome-wide pedigree. If a target guard removes family
   members, the gather schedules another round for those families.
7. Submit `haplotypes phase cross.toml --contig chrN` for every analysed
   chromosome. After all phase jobs finish, run `haplotypes run cross.toml` on
   one allocated node to check the publications and finish the run-wide reports.


Safety conditions:

- All jobs use the same fixed package sources, numerical-library policy, resolved
  configuration, inputs, maps, sample/contig order, realised seeds and tie settings.
  Resource limits may differ. Do not edit sources or inputs during the scatter.
- Initialise successfully before scattering; simulation init materialises every
  contig once. Each reconstruction needs its published evidence and the
  observation pilot from `relatedness`.
- The shared filesystem must support coherent cross-node `flock` and atomic
  rename. Publications use per-derivation claims and append locks. Duplicate
  submissions reuse one publication, but still waste resources.
- Give each job its allocated process/thread budget and sufficient memory and
  scratch capacity. Scratch must be writable; it is disposable and never a
  checkpoint prerequisite.
- Wait for every scatter writer to exit before the final `run`. Run one finishing
  invocation and one report publication at a time for a given output directory.
  Do not remove claims or completion markers to force progress, and recover an
  abandoned claim only after its writer is confirmed terminal.

This uses the same dependency graph and identities as a single-node `run`; the
scheduler changes placement and timing, not scientific decisions.

### Painting calibration barrier

After `relatedness`, run `haplotypes reconstruct CONFIG --contig CONTIG` for each
analysed contig to publish its final assembly, then run
`haplotypes calibrate CONFIG --cores CORES` once. This command schedules per-contig
outlier fits, their run reduction, hidden-window divisor fits, and the final run
reduction. The single-node read-state graph groups candidates per contig and
fold to share the emission builder. Completed products resume by their input identities.
After calibration, run `reconstruct --contig` again for every analysed contig to
reuse the assembly and publish its fine/reference painting and founder-refit
gain histogram and its contamination statistics. Continue with `pedigree`,
which pools the histograms and contamination statistics, completes the refits,
and repaints on the refit founders for inference. Then run per-contig `phase`
and `report`. The combined `run` command schedules
this barrier automatically, with every analysed final assembly required before
calibration and any painting. A fixed singleton robustness or scalar divisor
grid skips that fit round and still publishes its record.
