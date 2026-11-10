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
rate_is_measured = false

[relatedness]
pairwise_ibd = false
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

The models use two coordinate views. **Genetic coordinates** include fallback
extensions and determine transition probabilities. **Measured coordinates** credit
only supplied-map spans unless `[genetic_map] rate_is_measured = true` declares
that the fallback rate is a measured genome-wide rate for this organism. Pedigree
and family-support rules use that measured exposure. Do not label a convenient
placeholder rate as measured: it can overstate independent evidence. Generating
maps in a simulation are separate from the inference map.

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

## Optional pair reports and re-export

To request the pairwise IBD chain and `relatedness/pairs.csv`, set
`[relatedness] pairwise_ibd = true` before the invocation:

```sh
haplotypes relatedness cross.toml --set relatedness.pairwise_ibd=true --cores 8
haplotypes report cross.toml --set relatedness.pairwise_ibd=true --cores 8
```

This is a persisted per-invocation output selection, outside every scientific
identity. It costs O(N² L) and can be substantial. Turning it on does not recompute
the family model. Turning it off hides any retained pair record and removes the
stale CSV when reports are republished. Set the same selection on the report
invocation that should publish the file.

`haplotypes export run destination` writes a finished run's released founders and
samples to a new or empty directory. `--products` selects what is written: the default is
`founders samples`; `--products founders samples tracks` adds per-component founder tracks
(`--products tracks` alone writes only the tracks); `--format bcf` selects BCF; `--no-full-length`
selects kept sites only.
It reads the run's released site metadata and never needs to reopen the original
VCF. See the [output dictionary](outputs.md) for missing calls, phase sets and indexes.
