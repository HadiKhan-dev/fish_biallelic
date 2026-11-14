# Haplotype reconstruction

Haplotype reconstruction infers founder haplotypes, relatedness, pedigree,
sample phase and recombination maps from low-coverage biallelic read evidence in
experimental crosses. It reads VCF/BCF allele depths (`FORMAT/AD`) or generates
observations from explicit simulation templates. Sample names identify rows;
no sample metadata, sex labels or known-parent labels enter inference.

The supported mean read depth is 1x per sample or more. Releases are also validated at 0.5x so that data a
little below 1x still reconstruct well; below about 0.5x founder reconstruction degrades sharply (in simulated
crosses of 320 samples: 7 founder alleles per million called wrong at 1x, 89 at 0.5x, 3,365 at 0.35x and 12,500
or more at 0.25x).

The 4.0 implementation separates scientific models, typed products, checkpoint
storage, execution and output formatting. Repeated commands reuse compatible
published records. Changed consumed inputs or incompatible scientific settings
refuse reuse. A run requires a checkpoint root with the current layout.

Start with the [installation guide](docs/user/installation.md) and the
[synthetic quick start](docs/user/quickstart.md). Linux x86_64 and Python 3.14 or
later are required; the [tested dependency envelope](docs/reference/environment.md)
is narrower than the set of all versions that packaging accepts. No GPU is needed.

```sh
haplotypes example example
haplotypes run example/simulation.toml --cores 4
haplotypes evaluate example/run --cores 4
```

This small synthetic example checks software wiring; it is not an accuracy or
scaling benchmark. `run` initialises a new run automatically. Separate workflow
steps start with `init`; the [workflow guide](docs/user/workflow.md) includes a
Snakemake example.

- [Run a cross](docs/user/running.md): observations, maps, presets, resources and resume.
- [Output dictionary](docs/user/outputs.md): files, encodings, schemas and readers.
- [Limitations](docs/user/limitations.md): demonstrated envelope and model approximations.
- [Release notes](CHANGELOG.md).
- [Methods](docs/methods/index.md): ordered scientific rules by semantic unit.
- [Developer guide](docs/developer/architecture.md): architecture, contracts and tests.
- Generated [settings](docs/reference/settings.md), [parameters](docs/reference/parameters.md)
  and [commands](docs/reference/commands.md).

Standalone export reads verified released products and has no `--vcf` option.

See [LICENSE](LICENSE) for licensing and [CITATION.cff](CITATION.cff) for citation metadata.
