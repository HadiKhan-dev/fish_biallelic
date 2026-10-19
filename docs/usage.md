# Usage

This guide covers the commands, the inputs, how a run uses an allocation, and everything a run directory
contains. [configuration.md](configuration.md) lists every option and setting with its default; the
[methods](methods/README.md) describe what each stage computes.

## Commands

The program is `haplotypes` (or `python -m haplotype_reconstruction`):

| command | what it does |
| --- | --- |
| `haplotypes run [CONFIG]` | Run or resume the pipeline for a VCF cross, a generic VCF or a simulation. |
| `haplotypes status RUN` | Progress of a run directory from its records, without decoding results. |
| `haplotypes recombination RUN` | Run or resume the recombination stage of an existing run, with recombination overrides. |
| `haplotypes evaluate RUN` | Simulations only: accuracy of every stage against the known truth. |
| `haplotypes export RUN DEST` | The founder VCF and phased sample VCF of a finished run (the files of its export stage), optionally with founder-track TSVs, written to a new directory. |
| `haplotypes example DEST` | A small synthetic example: founder templates and a runnable configuration. |
| `haplotypes templates --vcf FILE --output DIR` | Simulation founder templates built naively from a VCF. |
| `haplotypes batch-worker QUEUE` | A helper allocation serving a running coordinator's batch queue. |

`RUN` is a run directory: its `config.toml` (the resolved configuration) and its checkpoints are found inside.

## Running a cross

A run is described by one TOML file; command-line options and `--set SECTION.KEY=VALUE` override it. Relative
paths are resolved against the working directory, not the file's location. A run of the TroMau cross starts from
the built-in preset, which supplies the contigs, the workbook columns, the parent-eligibility design and the
reference-consistency report; the configuration adds the data paths:

<!-- check_docs: config -->
```toml
preset = "tromau"

[run]
output = "work/runs/tromau"

[input]
vcf = "data/AcTm.bcf"

[samples]
metadata = "data/AcTm_metadata.xlsx"
```

```bash
haplotypes run tromau.toml --cores 112
```

A generic cross needs no workbook. The contigs must be listed (in the order to analyse them); without an
eligibility policy every sample is a candidate parent of every other:

<!-- check_docs: config -->
```toml
[run]
output = "work/runs/my_cross"

[input]
vcf = "cross.bcf"
contigs = ["chr1", "chr2", "chr3", "chr4", "chr5"]
```

The same run without a file:

```bash
haplotypes run --vcf cross.bcf --contigs chr1 chr2 chr3 chr4 chr5 --output work/runs/my_cross
```

A simulation is a run whose input kind is `simulation` (or `--templates DIR`); [simulation.md](simulation.md)
describes its settings, `haplotypes example` and `haplotypes evaluate`.

Scientific switches are changed with `--set`, for example `--set quality.read_calibration=false` or
`--set pedigree.evidence_calibration=off`. Every setting that can change results enters the checkpoint
identities, so a run directory always holds results of one configuration (see
[Checkpoints and resuming](#checkpoints-and-resuming)).

## Inputs

### Variant file

An indexed (`.csi` or `.tbi`), position-sorted VCF or BCF with `FORMAT/AD` (reference and alternate read depths).
GT-only or PL-only files are not supported: the read model is fitted from the allele depths. Every sample of the
header is analysed; to leave samples out (for example unrelated outgroups), subset the file first (`bcftools view
-S`). At least three uniquely named samples are needed.

One record policy applies to every input. Records that are not biallelic SNPs are dropped and counted by reason:
`multiallelic` (more than one ALT allele) and `not_snp` (REF or ALT not one of A, C, G, T: indels, MNPs, symbolic
alleles, `*`, no ALT). When two or more biallelic SNP records share a position, all of them are dropped
(`duplicate_position`), because which of them is the site cannot be decided from the file. The counts appear in
the log line of each contig's evidence stage, in the evidence record and in the run record. Multiallelic depths
are never truncated to a biallelic pair. A missing AD value is zero reads (unobserved, not reference support).
Unreadable files, unknown contigs, records out of position order and malformed AD values are errors.

Select populated physical contigs explicitly: a contig must be declared in the header and have indexed records.
The genome-wide stages need at least three contigs, and Tier-B pedigree calls need at least five informative
chromosomes (see [Checks before computation](#checks-before-computation)).

The input file's path, size and modification time and its sample list enter the evidence identity: a run directory
belongs to one input file. Do not modify or replace the file while its runs are in use.

### Sample metadata

`[samples]` names an optional workbook (`.xlsx`) with one row per sample: the column of the VCF sample IDs
(`id_column`), generation or cohort labels (`generation_column`) and recorded sex (`sex_column`). Every analysed
sample must occur exactly once; rows of other samples are ignored. Labels are read as text; a generation label
that starts with one of `generation_prefixes` is read as that prefix (the AulStu preset reads `G0_...` labels as
`G0`), and when `generations` or `sexes` are listed, other labels are errors. The workbook serves the cohort
eligibility policy and the reference-consistency report; it never supplies individual parentage.

### Parent eligibility

`[pedigree.eligibility] policy` decides who may be whose parent:

- `none` (default): every sample is an inference target and a candidate parent of every other sample.
- `cohorts`: the workbook's generations define the design. `parent_generations` maps each target generation to
  the generation its observed parents may come from, e.g. `{F2 = "F1", F3 = "F2"}`; samples of other generations
  are neither targets nor candidates. With `opposite_sex_pairs = true`, a two-parent configuration needs one
  recorded female and one recorded male (parent order remains arbitrary). The generation steps are design
  chronology, not parentage: cohort labels never assert that a particular sample is a parent.
- `constraints`: a JSON file (`constraints = "FILE"`) for any other design:

```json
{
  "candidate_parents": {"fish_c": ["fish_a", "fish_b"], "fish_d": []},
  "excluded_samples": ["outside_pedigree"],
  "ineligible_children": [],
  "direction_supported_edges": [["fish_a", "fish_c"]]
}
```

  A child not listed in `candidate_parents` keeps every other sample as a candidate; an empty list allows no
  observed parent. Excluded samples are neither parents nor children; `ineligible_children` may still be parents.
  `direction_supported_edges` are `[parent, child]` pairs whose direction is independently established (for example
  by generation), not assumed parent pairs. Candidate restrictions alone assert neither direction nor parentage.

Eligibility affects the pedigree stage only. Discovery, assembly and painting use every analysed sample.

### Genetic map

Inference uses `[genetic_map] rate_cm_per_mb` (5 cM/Mb) everywhere, or a cumulative map given by
`[genetic_map] file` for the chromosomes it covers; [genetic_maps.md](genetic_maps.md) describes the formats.

## Checks before computation

A run checks everything it can before computing anything:

1. The configuration loads strictly: unknown sections or keys, wrong types and inconsistent settings are errors
   naming the valid choices ([consistency rules](configuration.md#consistency-rules)).
2. The variant file has `FORMAT/AD`, at least three uniquely named samples, and every configured contig is declared
   and has indexed records.
3. The workbook, sheet and columns exist, every sample occurs exactly once, and labels are allowed; the
   eligibility policy is built (a constraints file is parsed).
4. `[execution] cores` fits the CPU affinity.
5. A run that reaches the genome-wide stages needs at least `PedigreeConfig.exposure.minimum_exposed_contigs = 3`
   contigs (set `[run] stop_after = "painting"` for fewer). Tier-B pedigree calls need at least five informative
   chromosomes: with fewer than five configured contigs the run warns at the start that it can release no Tier-B
   parent (so family phase and recombination have no parent edges). The pedigree stage states its own count of
   informative chromosomes at its end, which is authoritative (a contig can turn out uninformative).
6. An existing checkpoint directory carries this release's layout mark, and a shard of a simulation finds the
   simulated inputs of every contig recorded.

## What a run computes

| stage | per | product |
| --- | --- | --- |
| [evidence](methods/evidence.md) | contig | allele depths, site and sample QC, calibrated genotype likelihoods, 200-site blocks |
| [discovery](methods/discovery.md) | 200-site block | missing-aware local founder panels |
| [local selection](methods/local_selection.md) | contig | path-selected local panels, refined twice with context assemblies |
| [assembly](methods/assembly.md) | contig | component-preserving founder chromosomes (L1-L4 hierarchy, [founder refinement](methods/founder_refinement.md) after each level) |
| [painting](methods/painting.md) | contig | diploid founder mosaics of every sample |
| [pedigree](methods/pedigree.md) | genome | observed-parent calls with support tiers |
| [founder refit](methods/founder_refit.md) | contig | founder alleles refitted from the painted carriers |
| [family phase](methods/family_phase.md) | contig | pedigree-conditioned, genotype-preserving phase |
| [recombination](methods/recombination.md) | contig | conditional maps, crossover intervals and exposure |
| [export](#export) | contig | the final founder VCF and phased sample VCF in `OUTPUT/export` |

Discovery runs over all pending contigs in one block pool before the per-contig work starts (the evidence of the
next contig is prepared while the current one is discovered). Local selection, assembly and painting then run per
contig in concurrent lanes. The genome-wide stages follow when every contig is painted, and the export writes the
final VCFs last (`[export] enabled = false` skips it). Nothing feeds back upstream: family phase and the maps
never change the founders, paintings or pedigree, and the export changes no record.

## Execution

### CPUs

`[execution] cores` (`--cores`) is the run's CPU ceiling: the product of processes and threads never exceeds it.
The default is the CPU affinity of the process (the allocation). Discovery workers, contig lanes, process pools
and Numba thread teams share the ceiling as leases, and freed CPUs move to the work still running at numerical
boundaries; a running kernel never changes its team. `[execution] contig_lanes` sets how many contigs are
reconstructed at once: `auto` is one lane per 16 cores, at most one per pending contig, limited by memory; `1`
runs one contig at a time with every core.

### Memory

One memory governor per run admits parallel work by its estimated peak bytes. Its limit is the memory available
when the run starts (the node's available memory, every enclosing cgroup limit and the Slurm job's memory) minus a
10% headroom (at least 4 GiB), and at most `[execution] memory_limit` (`--memory-limit`, for example `200G`) when
set. Every parallel construct (process pools, chromosome executors, contig lanes, concurrent fits) reserves its
tasks' estimated bytes before starting them. When memory is tight, work waits instead of failing: fewer tasks run
at once and the run slows down. A task too large to fit even alone runs alone, with a logged warning. The log and
the run record (`memory`) report the limit, the waits and wait time, the peak reserved and peak used memory, and
each task kind's estimate against its measured peak.

### Results never depend on execution settings

Cores, lanes, memory pacing, serial mode, shards and batch helpers change when work runs, not what it computes.
Parallel kernels reduce per-sample (or per-site) rows in a fixed order, so results do not depend on thread counts
or on how work was distributed, and ties between equally supported alternatives are decided by fixed conventions
([ties](methods/ties.md)). These settings are outside every checkpoint identity and may differ between the
invocations of one run. Every search limit that bounds memory is a fixed constant (for example founder
refinement's window-frontier budget), never the memory that happens to be free.

### Compiled kernels

The numerical kernels are compiled once per code version and kept between runs in
`~/.cache/haplotype-reconstruction/numba` (or under `$XDG_CACHE_HOME`); a run of a new code version compiles its
kernels in parallel before its first pool. Nothing needs to be configured; see
[performance.md](performance.md#compiled-kernels).

## Short allocations

By default discovery covers every contig before the lanes start, so an allocation that ends early can leave many
contigs partly done. `[execution] serial = true` (`--serial`) takes one contig at a time, in the configured order,
through evidence, discovery, local selection, assembly and painting with all cores; each contig is recorded before
the next starts. `[execution] max_contigs = N` (`--max-contigs N`, implies serial) finishes at most N more contigs,
counting one resumed from partial records, then exits with status 0, naming the contigs still to do. Rerun the
same command to continue:

```bash
# Each invocation paints at most two more contigs.
haplotypes run my_cross.toml --cores 32 --max-contigs 2
```

When every contig is painted, the run continues as a normal run with the genome-wide stages (unless it stops after
painting). Serial and concurrent invocations may alternate on one run directory: every contig runs the same code
from the same records, so the results are the same. On a large node concurrent lanes finish sooner, because one
contig rarely keeps every core busy; serial mode trades that throughput for small units of durable progress. A
simulation first generates its inputs for every contig.

## Shards

`[execution] process_contigs` (`--process-contigs C ...`) processes only the named contigs (of `[input]
contigs`) through painting and then stops: a shard never runs or marks the genome-wide stages. Independent shards
can run on separate allocations against one run directory, each with its own contigs:

```bash
haplotypes run sim.toml --process-contigs chr1 chr2 --cores 76
haplotypes run sim.toml --process-contigs chr3 chr4 --cores 76
```

Keep `[input] contigs` and every scientific setting identical across shards. A shard of a simulation needs the
simulated inputs of every contig recorded first, and is refused otherwise: run the configuration once without
`process_contigs` (the simulation of every contig is recorded before any other stage; `--max-contigs 1` stops after
one contig). After every shard has finished, run once without `process_contigs`: it finds every contig painted,
records stage completion and continues with the genome-wide stages. Shards cannot be combined with serial mode.

## Batch helpers

Other allocations can help a run with its independent local-selection fits and hierarchy batches. The
coordinator names a directory both can reach; helpers serve it:

```bash
# Coordinator, in its own allocation.
haplotypes run my_cross.toml --cores 76 --batch-queue /shared/my_cross/batches

# Helper, started separately inside another allocation.
haplotypes batch-worker /shared/my_cross/batches --cores 48
```

Both must run the same package source and see the same directory. Set each `--cores` within that process's own
allocation; the package submits no jobs and moves no shell between nodes. Several helpers can join one queue; a
helper exits after `--idle-seconds` (120) without work, so start helpers while batches are available. The
coordinator executes work too, and only the coordinator writes checkpoint records; results keep their task
identities and scientific batch boundaries, so they do not depend on who computed them. Everything else
(discovery, founder refinement, painting and the genome-wide stages) runs in the coordinator's allocation.

Transport files are temporary: a bundle's input is deleted once its result is used, and a completed stage leaves
nothing in the queue. An interrupted hierarchy step keeps its finished results (and failed bundles their inputs
and tracebacks) for the resume, which retries the failed bundles. Before restarting a coordinator, make sure the
previous one has ended; never run two coordinators of one run directory. Use a fresh queue for changed source.
Queue files of a run that will not be resumed can be deleted independently of its checkpoints.

## Checkpoints and resuming

Rerunning the same command resumes: every stage reads what is recorded and computes only what is missing. An
interruption loses only the work in progress since the last record:

- the discovered blocks of a contig since its last saved chunk (64 blocks per chunk);
- the current chunk of a local-selection round;
- the hierarchy batches being joined (each completed join is saved, and shared by every assembly of the contig);
- the components being refined (long components also save their converged fits and every sweep);
- each contig being painted, and each contig in progress in the pedigree preparation, scoring and exclusion,
  founder refit, family phase or recombination;
- up to about two minutes of family-phase iterations.

### Identities

Each stage binds its records to an identity: the stage's configuration, the identities of the upstream records it
read and the source code of the packages it runs. A mismatch stops the run with an error naming the stage: the
directory holds results of another configuration or code version. Use a new output root then; never rename
records or edit identities to force reuse. Records are never removed, so a stage cannot be rerun in place with
other settings. To compare settings of a later stage on the same upstream results, stop after painting
(`--stop-after painting`), copy the run directory (and its checkpoint directory if it lies elsewhere) and finish
each copy with its own settings. Never let two run directories share contig files: both would append to them.

A run marks its checkpoint directory with this release's layout before it writes anything, and a directory
holding checkpoints without that mark (written by an earlier release), or with other subdirectories than
`contigs/` and `work/`, is refused at once: use a new output root.

### Status

`haplotypes status RUN` reports progress from record metadata alone: each invocation's run record (command,
state, host, wall time, the stages active when it was last written, any error), the genome-wide stages completed,
and one line per contig with each stage's progress and any live resume state. `--contigs` selects contigs,
`--checkpoints` names a checkpoint directory outside the run directory and `--json` prints JSON. The exit status
is 1 when the directory cannot be read.

### Checkpoint files

| path | contents |
| --- | --- |
| `checkpoints/contigs/<contig>.ckpt` | everything computed for one contig, from its evidence (and simulated reads) to its recombination map, including every intermediate founder stage |
| `checkpoints/run.ckpt` | genome-wide records: the run manifest, stage completion, the simulation manifest and true pedigree, the pedigree product, family-phase and recombination summaries |
| `checkpoints/work/` | transient resume state (family-phase iterations), present only while it is live |

Each file is a sequence of records appended as results arrive. A record has a key (`evidence`,
`discovery/blocks/000003`, `assembly/l2/join`, `painting`, ...) and is never modified; a later record with the same
key supersedes it. Appends are serialised by a lock on the file and flushed to disk before they count, so threads,
processes and nodes may write to one file at once, and a cut-off write leaves only an incomplete last record that
readers ignore and the next writer removes. Run one invocation per run directory (shards apart): two invocations
cannot corrupt it but repeat each other's work. [development.md](development.md#checkpoint-store) describes the
container and the record keys of each stage.

## Recombination settings

`haplotypes recombination RUN` runs or resumes the recombination stage of a run whose family phase is recorded,
with the run's configuration plus `--order-breakpoints FILE` and `--set recombination.KEY=VALUE` overrides, and
republishes its tables. The run is found from the directory itself (its `config.toml` and checkpoints). Contigs
already mapped with the same settings are read back, not recomputed; the settings of a stage recorded by the
command are written to the run's `config.toml`. A recorded stage keeps the settings it ran with: a contig mapped with
other settings stops with an identity mismatch, so compare recombination settings in copies of the run directory
taken after painting, as described in [Identities](#identities).

## Outputs

Everything a run writes is under its run directory (`[run] output`, default `work/runs/<preset or input
kind>[_seed<seed>]`); the checkpoint directory may live elsewhere (`[run] checkpoints`). Output directories are named
after the stages. Tables are written atomically; list and dict cells are JSON.

| path | contents |
| --- | --- |
| `config.toml` | the resolved configuration of the run (loads again unchanged) |
| `logs/run_<stamp>_<pid>.log` | the log of each `run` invocation (`recombination_<stamp>_<pid>.log` for the recombination command) |
| `logs/contigs/reconstruction_<contig>_<stamp>_<pid>.log` | the log of each contig processed in a concurrent lane |
| `run_records/<stamp>_<pid>.json` | per invocation: command, resolved configuration, source digest and commit, environment and dependency versions, host, CPU affinity, state, exclusive stage timings, per-contig summaries, memory statistics, error |
| `checkpoints/` | the checkpoint store (above) |
| `quality/<contig>.samples.tsv`, `quality/<contig>.segments.tsv` | per-contig library-contamination QC per sample, and inherited heterozygous segments shared by pairs of samples (reported, never flagged); only where the sample QC ran |
| `discovery/reference_consistency.csv` | with `[reports] reference_consistency`: discovered blocks compared with the reference generation's genotypes (descriptive; not founder truth) |
| `painting/<contig>/founder_support.tsv` | per component and founder row: called and missing alleles, observed carriers, called sites without an observed carrier |
| `painting/<contig>/founder_support_windows.tsv` | the same per window of at most 200 sites (1-based inclusive coordinates) |
| `painting/<contig>/assembly_search.tsv` | every record of the final assembly's join searches: budgets, truncations, leading alternatives, count decisions |
| `painting/<contig>/founder_counts.tsv` | per component: founder rows and how firmly the count is determined (ambiguity, count cycles, decisions) |
| `painting/<contig>/sample_switches.tsv` | per sample: painted switches relative to the cohort median |
| `painting/<contig>/sample_segments.tsv` | the released painting per sample: one row per sample, component, track and run of bins with one release (Start/End in bp, status, named founder row, candidate rows); a `SWITCH_INTERVAL` row is the interval in which the track switches between founders the sample cannot tell apart |
| `pedigree/tier_b_relationships.csv` | the primary pedigree calls (one row per sample: parents, parent-count state, status) |
| `pedigree/*.csv` | partial-edge, Tier-A and complete-data views, candidate sets, state calls, explanations, alternative explanations, search and exclusion diagnostics, evidence and chromosome summaries |
| `pedigree/evidence_calibration.json` | the cross-chromosome evidence scaling and its fit |
| `pedigree/summary.json` | informative and omitted contigs, the chromosome-count statement (whether Tier A and Tier B were reachable), the family solve, the exclusion and the eligibility summaries |
| `founder_refit/summary.csv` | per contig: components, founder rows, called and changed alleles, changed sites, median gain of a change, samples excluded from founder construction |
| `family_phase/summary.csv` | per contig: family-MAP sweeps, convergence, final objective, called alleles, observable phase changes, root-frame repairs |
| `recombination/<contig>.map.csv` | per map bin: called and expected crossovers, observable exposure, rates, cumulative cM |
| `recombination/<contig>.crossovers.csv` | called crossover intervals per meiosis, with bin-honest reported intervals |
| `recombination/<contig>.coverage.csv`, `<contig>.called_coverage.csv` | informative and callable spans per meiosis |
| `recombination/<contig>.meioses.csv` | per meiosis: marker counts, calls, expected crossovers, exposure |
| `recombination/<contig>.junctions.csv`, `junctions.csv` | candidate reference-order junction loci per contig and genome-wide |
| `recombination/<contig>.png` | conditional rate and effective meioses along the contig |
| `recombination/summary.csv` | per contig: meioses, crossovers, exposure, coverage, orientation fit, breakpoints, junction loci |
| `export/<contig>.founders.vcf.gz`, `.tbi` | the founder VCF: haploid GT, one column `c<C>_H<k>` per founder row `k` of component `C` ([Export](#export)) |
| `export/<contig>.samples.vcf.gz`, `.tbi` | the final sample phase: GT and PS ([Export](#export)) |
| `export/manifest.json` | every exported file, its components, columns and site counts |
| `ties/decisions.csv`, `ties/summary.json` | when the run completes: decisions taken inside a tie by the fixed conventions, per contig, record group and decision kind, their totals per stage, and the recorded ambiguity flags ([ties](methods/ties.md#what-a-run-records)) |
| `simulation/truth_pedigree.csv` | simulations: the analysed samples and their true parents |
| `evaluation/` | written by `haplotypes evaluate` ([simulation.md](simulation.md#evaluation)) |

The methods pages explain how to read the tables of their stages: [painting](methods/painting.md),
[pedigree](methods/pedigree.md), [family phase](methods/family_phase.md) and
[recombination](methods/recombination.md), and [ties](methods/ties.md) for `ties/`. Readable tables summarise; the complete results (allele calls, phase,
paintings) are in the checkpoint records; the export stage writes the founder alleles and the final phase out
losslessly as VCFs.

## Export

Every complete run ends with the export stage, which writes the final products to `OUTPUT/export/` (`[export]
enabled = false` skips it); `haplotypes export RUN DEST` writes the same files for a finished run to a new or empty
directory, with the options below. Per contig:

- `<contig>.founders.vcf.gz`, the founder VCF: what the run discovered about the founding haplotypes. FORMAT GT is
  haploid, with one column per founder row `k` of every assembly component `C` of the chromosome, named `c<C>_H<k>`
  (founder-refit panels when recorded, else the painted panels). Founder labels are component-local: `c0_H1` and
  `c1_H1` are not implied to be one founder, so a column has calls only inside its component's span (first to last
  panel position, listed in `##component` header lines) and `.` outside it. Inside the span a column holds the
  released allele (`0` REF, `1` ALT), or `.` where the panel withholds it; nothing is imputed.
- `<contig>.samples.vcf.gz`, the final sample phase: GT and PS, phased (`|`) inside a component with PS = the
  component's first site, unphased (`/`, PS `.`) between components; unknown alleles stay `.`.
- a tabix index beside each file (`.tbi`; `.csi` for contigs longer than 2^29 bp), and `manifest.json`, which
  describes every file, component, column and site count.

**Sites.** By default (`[export] full_length = true`, `--full-length`) both files hold every input biallelic SNP
that reached the site-quality filter, once each, in position order. FILTER is `PASS` at sites the filter kept and
otherwise lists the applied rules that removed the site (header-defined: `r1` total depth, `r2` depth
heterogeneity, `r3` wrong-allele reads in homozygotes, `r4` heterozygote allele balance, `r5` uninformative; r3
and r4 are evaluated only at sites passing r1 and r2), so `bcftools view -f PASS` keeps the analysed sites. At a
removed site every sample is `./.`. In the founder VCF, a site removed by `r5` alone (fewer than two expected
carriers of a non-major genotype: effectively monomorphic in the cross) carries the common allele in every column
whose component span contains it: the allele with more reads in total over the samples r5 counts, i.e. the run's
analysed samples that the contamination QC did not flag (FORMAT/AD of the input file, read at export time; those
samples' GT allele counts for a record without AD; equal totals give `.`). A site removed by any of r1-r4 (likely mapping artefacts or paralogs) is `.` in every column. Records the
loader drops never reach the filter and are not written: non-biallelic records (multiallelic records, indels, MNPs,
symbolic alleles) and biallelic SNPs sharing a position; the header counts them. With `full_length = false`
(`--no-full-length`) the files hold only the kept sites, with FILTER `.`. Where the site filter did not run
(simulations by default) every input site is kept.

**Alleles.** REF and ALT come from the run's input file (checked by path, size and modification time; `--vcf`
names it if it moved). A simulation's alleles are abstract and are written as A (0) and C (1), labelled in the
headers and the manifest.

**`haplotypes export`.** `--products` chooses among `founders` and `samples` (the default pair) and `tracks`, the
per-component founder-track TSVs `<contig>.componentNNNN.founders.tsv.gz` (`CHROM`, `POS` and one column `Hk` per
founder row, one row per panel site); `--format bcf` writes BCF with CSI indexes; `--no-full-length` writes the
kept sites only; `--contigs` and `--cores` select contigs and CPUs. An export reads the stored records only: it recomputes nothing
and changes no record. In a run, the export stage records each contig's files (record `export`), so a rerun skips
exported contigs. When the export's identity changes (other `[export]` settings, or a new version of the export
code), the rerun re-exports the contig and removes the files of its earlier export instead of stopping with an
identity mismatch as the other stages do: the files are derived from the stored records, which stay unchanged.

## Stopping processes

To stop a run, interrupt its main process (Ctrl-C, or `kill PID` with the PID from its run record): contig lanes
stop with it, pool workers end after their current task, and every record already written is kept for the resume.
On shared or allocated nodes, signal only the processes of your own run;
[development.md](development.md#stopping-processes-safely) explains why user-wide kill commands are dangerous.
