# Synthetic quick start

Create a new, empty example directory and run these commands from one working
directory:

```sh
haplotypes example example
haplotypes run example/simulation.toml --cores 4
haplotypes status example/run
haplotypes evaluate example/run --cores 4
```

`example` writes templates and a complete configuration. By default it creates
three contigs with 1,200 sites each and six independent synthetic founder
haplotypes, paired into three diploid founders. The templates use Bernoulli marker
alleles at evenly spaced positions. They are a **software wiring example**, not a
reference genome, realistic diversity model or calibrated accuracy benchmark.
The generated configuration records the seed and paths.

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
