# Running a cross

## Prepare observations

Use an indexed multi-sample VCF or BCF with biallelic SNPs and `FORMAT/AD` counts
for REF and ALT. Select the samples belonging to the analysis before starting.
The pipeline does not use cohort labels, generation, sex or known-parent metadata
as an eligibility filter: every sequenced sample is a target, and every other
sample is a parent candidate.

Contig names must match the variant header and genetic map exactly. Configure the
contigs explicitly and keep their order fixed. Within records, the sample axis is
canonical by sample name; published sample columns and per-sample tables return to
input order. Loader-dropped variants and site-quality exclusions are separate
quantities. The input VCF FILTER is not the package's site filter; exported r1–r5
FILTER values describe the package's decisions.

A complete analysis needs multiple informative contigs. Pedigree release also
requires enough independent chromosome/map evidence; adding nominal contigs without
informative material does not satisfy that requirement. Check the pedigree
summary's reach and omission fields before interpreting an empty relationship table.

## Configure the run

Save this as `cross.toml`, replacing the observation/map paths and contigs with
your data. This example uses a supplied map and leaves the fallback rate at its
schema default.

<!-- check_docs: config -->
```toml
version = 4
preset = "generic"

[run]
output = "run"

[input]
kind = "vcf"
vcf = "data/cross.bcf"
contigs = ["chr1", "chr2", "chr3", "chr4", "chr5"]

[genetic_map]
file = "data/cross.map"
```

`generic` supplies no contig list. `tromau` supplies the preset's configured
22-contig list; it supplies neither a data path nor sample metadata. A preset is a
base configuration, not a parentage assertion. Only these two named presets are
provided.

Relative paths resolve against the **working directory**, not the directory
containing the TOML file. Keep that directory fixed across invocations, or use
absolute paths appropriate to your own machine. Precedence, from lowest to highest,
is schema defaults, preset, TOML, ordinary CLI options, then repeatable `--set`
overrides. Unknown keys and invalid values are errors. See the generated
[settings reference](../reference/settings.md).

Pedigree decisions use the composite log likelihoods directly. Existing configurations carrying
the pedigree table's `evidence_calibration` key with `"off"` or `"predictive"` load with one removal warning; the key is
discarded and is absent from the resolved configuration. Other legacy values are errors.
The composite evidence is not claimed to give calibrated posterior probabilities.

```sh
haplotypes run cross.toml --cores 8
haplotypes status run
haplotypes status run --json
```


4.12 reads and resumes only checkpoints and exports written by 4.12. Earlier
run directories and exports must be re-run with 4.12 in a new output root. A
refusal names the recorded writer release; artifacts without that provenance
are reported as having no recorded writer release.

`run` initialises a new run automatically and resumes compatible publications on
later invocations. An explicit `init` is needed before separate workflow step
commands; see [workflow mode](workflow.md).

## Supply a genetic map

A map gives cumulative centimorgans, not a local per-base rate. A headerless
four-column file uses `chromosome marker_id cumulative_cM position_bp`; a headed
three-column file uses `chromosome position_bp map_cM`:

```text
chromosome position_bp map_cM
chr1 1 0.0
chr1 1000001 5.0
chr1 2000001 15.0
```

Positions must increase strictly within each contig, and cumulative cM must be
finite and nondecreasing. Coordinates use the same reference as the VCF; no offset
is applied. A plateau is a zero-distance interval, not missing map information.
A supplied map interpolates between knots; `[genetic_map] rate_cm_per_mb` supplies
unmapped contigs and the tails outside the knots.

Genetic coordinates include fallback extensions and determine transition probabilities. The pedigree and the
relatedness support measure material on the same working map: a supplied map, else `rate_cm_per_mb`. The pedigree's
regime reads U, the working Morgans of the informative painted components (below 5 Morgans nothing is assessed,
below 12 Tier B uses the low-material thresholds, Tier A needs 12), so a placeholder rate that is far from the
organism's sets the material scale: a rate twice too high doubles U and halves every cut in true Morgans. The cuts
were calibrated in true Morgans, so they hold only on a measured map or a correct genome-wide rate. Before a
genome-wide run the command line warns when the configured working map of the analysed contigs lies below a cut.
Generating maps in a simulation are separate from the inference map.

## Resources and stopping

`--cores` is the total process/thread ceiling, bounded by the available CPU
allocation. `--contig-lanes` controls concurrent contig work; `--serial` runs it
one contig at a time. `--memory-limit` sets an execution capacity, for example
`--memory-limit 16G`; it is not a scientific search budget. The scheduler observes
memory pressure and reduces admission/concurrency. Resource pressure must not
silently change model decisions. Insufficient resources can stop a task with an
explicit error; valid completed publications remain reusable.

```sh
haplotypes run cross.toml --cores 8 --contig-lanes 1 --memory-limit 16G
haplotypes run cross.toml --cores 8 --stop-after relatedness
haplotypes run cross.toml --cores 8 --serial --max-contigs 1
```

`--max-contigs` limits newly completed contig work in that invocation and implies
serial execution. The available clean stop points are relatedness, discovery and
painting. They are cuts of the same graph, not alternative models. Contig-subset
commands cannot satisfy a genome-wide prerequisite by omitting its other contigs.
For parallel jobs on shared storage, use the [workflow commands](workflow.md).

## Resume and source identity

Resume checks each publication's derivation: consumed inputs, scientific parameters,
upstream identities, code dependencies and numerical policy. A release label
alone does not enter these identities. It does not trust a
completion marker alone. A changed consumed input payload is refused; touching a
file without changing consumed content does not change that identity. The source
binding includes alleles, relevant read counts, fallback GT classes where AD is
unusable, contig length and loader drop counts. Genotypes at usable AD rows and the
input FILTER column are outside that payload. Avoid editing an input while a run
is active.

Changing a scientific setting, map, contig order or numerical environment can
invalidate the binding; use a fresh run/checkpoint root for a different analysis.
Commands require a checkpoint root with the current layout. Resume is at published-product
boundaries, not at an unrecorded intermediate family-phase iteration.

`status` reports durable products and recorded progress without decoding large
scientific payloads or reading the task log. It checks completion certificates
against committed container headers: every referenced record must still be the
version present when completion was checked and published. A later replacement
makes the certificate stale. Scientific resume independently checks derivations.

Stage publications save disposable `*.ckpt.index.json` header snapshots beside
the checkpoint containers. Status reads those snapshots and scans subsequent
appends; a missing, truncated or stale snapshot falls back to scanning headers.
It writes nothing, so archived and read-only checkpoint roots remain usable.
The status latency target is ten seconds for an N = 320 run with saved indexes.
A long unpublished tail, missing indexes or slow storage can require a full
header scan. Status reads only 4.12 invocation records and checkpoint layouts.

The run log prints progress every ten minutes and at each chromosome/stage
completion. It reports completed/known executor calls, running tasks, claimed CPU
slots and the memory governor's sampled live memory. Known totals grow when a
task expands into components or blocks; claims describe scheduler occupancy, not
measured operating-system CPU utilisation. These observations never control
scientific work. Once per contig per invocation, a warning identifies checkpoint
storage contention when publish lock waits exceed half the summed task wall time
across the most recent completed tasks (up to 32, after at least 8). The task log
retains publish lock-wait and fsync seconds for diagnosis. Refinement cap summaries appear once per chromosome and stage
(count or polish), with totals and affected component-level unit counts. The full
fit labels, individual counts and limits stay in component checkpoint records.

A clean stop is distinct from completion. SIGTERM and SIGINT stop workers,
release run-owned resources and record the attempt as `interrupted`, distinct
from `failed`. Resume the same command to reuse committed products. Allow the
process time to clean up before forcing a kill; SIGKILL cannot unwind cleanup. CLI exit codes are 0 for
success/clean stop, 2 for settings or checkpoint mismatch, 3 for input errors,
4 for resource errors, 130 for interruption, and 1 for an unexpected failure
with an error report.

## Leaving contigs out of the final analysis

Set the configured contigs to leave out in the run section:

```toml
[run]
exclude_contigs = ["chr7", "chr12"]
```

You can change this on an existing run without initialising again:

```sh
haplotypes run cross.toml --set 'run.exclude_contigs=["chr7"]' --cores 8
```

Evidence and the relatedness barrier still use every `[input] contigs` entry.
A `run` never schedules discovery, selection rounds, assembly or its contexts,
painting or refit for an excluded contig, including with `stop_after` or
`max_contigs`. Existing reconstruction records stay available. An explicit
`reconstruct --contig` command can work on an excluded contig; removing its
exclusion resumes reconstruction from those records.

Pedigree uses the included contigs; family phase, recombination, export,
evaluation and per-contig reports run only for those contigs. Relatedness families
and the pedigree sibling guard still use evidence from excluded contigs. To
remove a contig from relatedness as well, remove it from `[input] contigs` and
start a new run. Sample eligibility
and all model rules are unchanged. Unknown contig names and an empty remaining
set are errors. The pedigree's existing minimum-contig checks apply to the
remaining contigs, including its informative-contig check after preparation.

Each nonempty exclusion set has separate immutable downstream records. Its
reports and exports are in `RUN/analyses/<included-set digest>/`; the directory's
`analysis_contigs.json`, export `manifest.json` and evaluation `summary.json`
list the configured, included and excluded contigs. Earlier sets remain readable and changing back reuses their
products. `status` lists the current set and that set's workflow markers.
`RUN/analysis_selection.json` owns the active selection. Ordinary shards keep
`RUN/config.toml` unchanged; it describes the initialized settings. Workflow
commands using that recorded file, and the recorded-run commands, read the active
selection from JSON unless explicitly overridden. External configurations and
Python Settings objects specify their own selection. Use the same `--set` on
each shard when scattering a new selection. Do not change the active selection while
submitting shards for another set. A `phase --contig` shard refuses an excluded
contig; `reconstruct --contig` continues to accept it. `pedigree` and `report`
use the same selection as `run`.

The Python reader can select a stored analysis set explicitly:

```python
from haplotype_reconstruction.results.reader import Results
included = Results("RUN", exclude_contigs=("chr7",))
original = Results("RUN", exclude_contigs=())
```

An empty exclusion uses the ordinary output paths and identity parameters. Code
source hashes still change when upgrading software; source identities are checked
independently of the exclusion setting. Use the same software version while
resuming and changing sets. Scientific checkpoints from 4.14 cannot be resumed
with this implementation because changed source hashes change product identities;
use a new run when upgrading from 4.14.

## Re-export

`haplotypes export run destination` writes a finished run's released founders and
samples to a new or empty directory. `--products` selects what is written: the default is
`founders samples`; `--products founders samples tracks` adds per-component founder tracks
(`--products tracks` alone writes only the tracks); `--format bcf` selects BCF; `--no-full-length`
selects kept sites only.
It reads the run's released site metadata and never needs to reopen the original
VCF. See the [output dictionary](outputs.md) for missing calls, phase sets and indexes.
