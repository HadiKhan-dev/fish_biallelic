# Synthetic quick start

Create a new, empty example directory and run these commands from one working
directory:

```sh
haplotypes example example
haplotypes run example/simulation.toml --cores 4
haplotypes status example/run
haplotypes evaluate example/run --cores 4
```

`example` writes a complete coalescent configuration with an exact site count, three diploid founders and three contigs by default. Its source effective size is an explicit generating assumption.

`run` initialises a new run automatically: it resolves and binds the configuration,
contig/sample order and simulation seed, then materialises the observations and
truth before executing the dependency graph. It publishes results under
`example/run`. Explicit `init` is optional here; it is required before individual
workflow step commands. `evaluate` compares a simulated run with
its recorded truth; it is not an evaluation command for a real cross without truth.

Inspect these products after completion:

| Path beneath `example/run` | What it contains |
|---|---|
| `config.toml` | Resolved configuration and checkpoint location |
| `export/manifest.json` | Export options, per-contig summaries and written file lists |
| `export/*.founders.vcf.gz` | Released founder alleles; simulation A/C encodes abstract 0/1 |
| `export/*.samples.vcf.gz` | Released sample genotypes and phase sets (`GT:PS`) |
| `pedigree/tier_b_relationships.csv` | Primary released observed-parent relationships |
| `pedigree/tier_b_candidate_sets.csv` | Retained ambiguity in parent identities/configurations |
| `recombination/` | Conditional maps, event intervals, exposure tables and plots |
| `evaluation/` | Truth-based summaries and metric tables |

Missing founder alleles remain missing. The tiny example can have insufficient
independent evidence for some pedigree release tiers; a successful execution does
not promise a fully resolved pedigree. The [output dictionary](outputs.md) defines
these files and the difference between genotype-site errors and switch events.

Rerun the same `run` command to reuse valid published records after an interruption.
To render additional founder tracks into a separate, empty destination:

```sh
haplotypes export example/run exported --products founders samples tracks
```

Export reads verified released records. It needs no standalone source `--vcf`
argument. See [running a cross](running.md) for inputs and resource controls, or
[workflow mode](workflow.md) for scatter/gather execution.

## Simulation read model

Simulation observations are controlled by `[simulation] depth` and
`[simulation.reads]`. These generating settings are independent of the inference
model, which calibrates from the observed counts when `[quality] read_calibration`
is enabled. The defaults retain independent Poisson site counts, 2% homozygous
read error and balanced heterozygotes; all depth-heterogeneity controls are off.
Existing configurations retain their random draws and read counts when the new
controls are off.

For a plausible short-read low-pass scenario, replace an existing
`[simulation.reads]` table with this block, or add it if absent. Generated examples
and saved configurations contain explicit read defaults: remove the keys you want
the preset to supply, because those explicit defaults override the preset too.

```toml
[simulation.reads]
preset = "realistic"
# Explicit keys override the preset, even if equal to a schema default:
# count_dispersion = 0.0
```

The preset expands to these values in the saved `config.toml`:

| Setting | Value | Rationale |
|---|---:|---|
| `read_length_bp` | 150 | Two 150 bp mates; neighbouring sites can share a fragment and homologue |
| `fragment_mean_bp`, `fragment_sd_bp` | 375, 150 | Retain the existing broad short-insert library model; rounded normal, clipped at 50 bp and the upper six-SD boundary |
| `depth_cv` | 0.3 | Moderate library-to-library depth variation; matches the paired-end depth-sweep scenario |
| `site_depth_cv` | 0.5 | Moderate shared variation in coverage propensity across the genome |
| `site_depth_window_bp` | 1000 | Share site propensity within short physical windows to represent local coverage bias |
| `count_dispersion` | 0.1 | Mild residual extra-Poisson count variance: at conditional mean 2, variance is 2.4 |
| `heterozygote_alt_probability` | 0.48 | Mild reference bias, retaining a 48:52 ALT:REF balance |
| `error_rate` | 0.01 | A 1% effective homozygous error scenario, including mapping/observation errors |
| `dropout_fraction` | 0.0 | No forced missing tracts; heterogeneity and low depth already cause zero counts |

Two 150 bp mates follow [Illumina's WGS read-length guidance](https://www.illumina.com/science/technology/next-generation-sequencing/plan-experiments/read-length.html).
[Benjamini and Speed (2012)](https://doi.org/10.1093/nar/gks001) establish that GC
content and fragment properties affect coverage. The numerical CVs, dispersion,
window width, bias and effective error above are explicit sensitivity assumptions,
not estimates from that paper or a fitted cichlid library. Keep `[simulation] depth`
at the coverage you want to study; the preset does not set it. Very sparse marker
panels may have only one marker per 1 kb window and therefore little window correlation.

`depth_cv` draws one lognormal factor per sample and normalises the sample mean to
one. `site_depth_cv` draws lognormal factors with population mean one, shared by
all samples on a contig. A width of zero makes sites independent; a positive
`site_depth_window_bp` makes factors constant within fixed windows anchored at bp
zero. Window factors are drawn only for occupied windows. This represents local
GC/mappability effects without requiring sequence covariates; it does not compute
GC content or model allele-specific mappability. Hard window edges are an explicit
approximation, chosen over an arbitrary smoothing kernel.

`count_dispersion` is alpha in a gamma-Poisson model: conditional on sample and
site propensity, a cell with mean mu has variance mu + alpha × mu². Zero means
Poisson. It changes total depth, not the heterozygote allele-balance distribution.
In the paired-end sampler the site and cell propensities act through shared
fragment thinning, retaining spatial/homologue dependence. Mean depth is preserved
in expectation; a finite contig's mean can differ, especially with few occupied
windows. Extreme heterogeneity increases fragment-sampling time and temporary
memory because candidate fragments use the largest propensity within a sample.

`dropout_fraction` removes one contiguous fraction of markers per sample and
compensates the surviving depth in expectation. The settings reference lists
[all read-model keys](../reference/settings.md#simulationreads); the
[simulation methods](../methods/simulation.md#ou27) give the count law and draw order.


## Run one chromosome on multiple nodes

Initialize the run and prepare the contig's evidence as usual. On each allocated
node, start the same command using the same fixed source tree, configuration,
output directory and checkpoint root:

```bash
haplotypes reconstruct CONFIG --contig chr3 --cores 76 --memory-limit 240G
```

Additional invocations may join later. Each recreates the graph from checked
publications and computes unclaimed units: discovery and selection blocks,
selection context refits, founder-completion ranges and their final gather,
assembly batches, refinement components, painting and founder refit. A small
founder completion is one whole-product unit. Painting and refit are whole-contig
units. Each process finishes when the reconstruction cut through refit is complete.
Genome-wide stages keep their ordinary workflow; this command launches no nodes.

Each invocation needs the memory of a standalone reconstruction at its declared
CPU budget: evidence, active workspaces and worker runtimes belong to that node.
Memory does not divide by the number of nodes. The example budget above requires
an allocation with more than 240 GiB available; choose both budgets for the
resources actually assigned to the process. Logs and run records are separate
for each invocation.

A busy claim defers the task, releases its CPU seat and lets the scheduler run
other ready work. Retries are five seconds apart. When all remaining work is held
elsewhere, the process waits without occupying task CPUs and logs a heartbeat
every 30 seconds. It resumes published records as on an ordinary restart. There
is no idle timeout; long units can finish. Cold kernel warm-up can take minutes
and occurs independently for every invocation.

Use a shared filesystem with coherent cross-node POSIX flock (RDS Lustre is
mounted with flock). A holder's death releases its claim; workers also watch their
coordinator's death and exit, so survivors can recompute unpublished work. Never
remove claim files to steal live work. Node-loss recovery depends on filesystem
lock reclamation. A live process blocked in filesystem I/O can still hold a lock;
a heartbeat reports coordinator liveness, not remote progress.

All processes must use the same tree throughout the run. Scheduling code is in
the derivation identity, so checkpoints from an older source tree require a new
run root. Concurrent `run` commands restricted to one contig use this same
reconstruction route, but also execute their larger workflow cut; use
`reconstruct` after evidence preparation for a reconstruction-only measurement.
