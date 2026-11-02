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
