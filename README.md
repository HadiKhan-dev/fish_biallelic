# Founder haplotype reconstruction

Reconstruct founder haplotypes, sample ancestry, pedigrees, phase and recombination maps from low-coverage
sequencing of experimental crosses. The pipeline works from allele read depths, keeps missing observations
unknown from local discovery to the final maps, and needs neither pedigree truth nor generation labels (a cross
design can be supplied as parent eligibility).

The primary applications are cichlid crosses; simulations with a known pedigree validate the methods.

## Install

Python 3.11 or newer. In an environment of your choice:

```bash
python -m pip install .
haplotypes --help
```

`python -m haplotype_reconstruction` is the same program. The dependencies (NumPy, Numba, SciPy, pandas,
cyvcf2, Blosc2, openpyxl, Matplotlib, hdbscan) are declared in `pyproject.toml`.

## Quick start

A self-contained synthetic example (no data needed):

```bash
haplotypes example work/examples/quick
haplotypes run work/examples/quick/simulation.toml
haplotypes evaluate work/examples/quick/run
```

The example exercises every stage and file format on a few short, sparse chromosomes; it is not an accuracy
benchmark and may leave relationships unresolved.

A cross: an indexed VCF/BCF with `FORMAT/AD` and the contigs to analyse.

```bash
haplotypes run --vcf cross.bcf --contigs chr1 chr2 chr3 chr4 chr5 --output work/runs/my_cross
haplotypes status work/runs/my_cross
```

A complete run ends by writing its final products to `OUTPUT/export/`: per contig a founder VCF (one haploid
column per founder row of every component) and the phased sample VCF.

Larger runs use a configuration file; the built-in presets `tromau` and `aulstu` describe the two cichlid crosses
(contigs, workbook columns, parent-eligibility design), so their configurations add only the data paths. Rerunning
the same command resumes from the checkpoints. See [usage](docs/usage.md) for inputs, allocations and outputs, and
[configuration](docs/configuration.md) for every setting.

## One pipeline

| stage | product | package |
| --- | --- | --- |
| evidence | allele depths of biallelic SNPs, site and sample quality control, calibrated genotype likelihoods | `evidence/` |
| discovery | missing-aware founder panels of 200-site blocks | `discovery/` |
| local selection | path-selected local panels, refined after context assemblies through L1 and L1+L2 | `local_selection/` |
| assembly | component-preserving founder chromosomes: completion, linker, joins, L1-L4 hierarchy, founder refinement after each level | `assembly/` |
| painting | diploid founder mosaics of every sample, with an explicit unknown state | `painting/` |
| pedigree | observed-parent calls (M0/M1/M2), parent identities, support tiers and ambiguity | `pedigree/` |
| founder refit | founder alleles refitted from the painted carriers | `founder_refit/` |
| family phase | pedigree-conditioned final genotypes, phase and phase sets | `family_phase/` |
| recombination | conditional rates, crossover intervals and observable meiosis exposure | `recombination/` |
| export | per contig a founder VCF and the phased sample VCF in `OUTPUT/export/` | `export/` |

VCF crosses, generic VCFs and simulations run the same stages; they differ only in configuration. Each stage
records its products in the run's checkpoint store, and nothing feeds back upstream. The [methods](docs/methods/README.md)
describe each stage's model, assumptions and cost.

How to read the results:

- Missing alleles remain unknown unless a model supports a call; disconnected components are never silently joined
  into one phase frame.
- The pipeline is designed for up to 30 founder haplotypes per block; no stage's cost grows exponentially with the
  founder count.
- Tier B is the primary pedigree output. **M0 means zero observed parents**, not necessarily a biological founder.
  Support tiers and bootstrap fractions measure internal stability, not calibrated probabilities of parentage.
- Family phase preserves called genotypes and missingness; no imputed genotypes are released. Recombination maps
  are conditional on the final phase and the inferred pedigree.
- Founder reconstruction can be ambiguous where ancestry is unsampled or many descendants repeat one ancestral
  recombinant, and painting errors or structured missingness can affect pedigree direction. Successful simulations
  do not establish individual-level parentage in real crosses.

## Documentation

- [Usage](docs/usage.md): commands, inputs, allocations, checkpoints and every output.
- [Configuration](docs/configuration.md): every option, setting, preset and internal constant (generated from
  the code).
- [Methods](docs/methods/README.md): one chapter per stage.
- [Simulation](docs/simulation.md): simulated crosses, the example, evaluation against the truth, exports.
- [Genetic maps](docs/genetic_maps.md): map formats and what an inferred map means.
- [Performance](docs/performance.md): execution and memory model, caches, determinism.
- [Validation](docs/validation.md): acceptance evidence and known limitations.
- [Development](docs/development.md): code map, stage contracts, checkpoint store, contributor rules.
- [Migration](docs/migration.md): old-to-new commands, settings and output names; [changelog](CHANGELOG.md).

## Licence and citation

The software is MIT licensed, copyright 2026 Hadi Khan. Dependencies and external datasets keep their own terms.
Please cite the software in scientific work.
