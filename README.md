# Founder haplotype reconstruction

Missing-aware reconstruction of founder haplotypes, sample ancestry and
pedigrees from low-coverage sequencing of experimental crosses. The primary
applications are cichlid crosses; simulations provide known-truth validation.

The pipeline discovers haplotypes in 200-SNP blocks, assembles them into
chromosome components, paints each sample as a diploid mosaic, infers parentage,
refines phase using family information, and estimates conditional recombination
maps. Missing alleles remain unknown unless the relevant model supports a call.
Disconnected chromosome components are not silently joined into a single phase
frame.

## Running

Python 3.11 or newer is required. Package dependencies are declared in
`pyproject.toml`; run directly from this checkout in your existing environment.
On CSD3, the project environment is:

```bash
conda activate /rds/user/ahk39/hpc-work/conda_envs/bio-env
python run.py --help
```

Run substantial work only on allocated compute nodes. The default CPU budget
is the process's current affinity; `--cores` can set a smaller ceiling. Worker
pools and numerical thread pools share that budget, rather than multiplying it.

```bash
# Known-pedigree simulation from frozen founder-sequence templates.
python run.py simulate --config configs/simulation.toml --seed 400

# Real-data workflows: edit the input paths in the selected configuration.
python run.py tropheops --config configs/tropheops.toml
python run.py astcal --config configs/astcal.toml

# A supplied cumulative genetic map; other chromosomes retain 5 cM/Mb.
python run.py simulate --config configs/simulation.toml --seed 401 \
  --recombination-map work/data/cross.map
```

Rerun the same command to resume. Use a new output directory when deliberately
changing inputs or scientific settings. Each run keeps logs, readable tables
and atomic chromosome/stage checkpoints together under `work/runs/`; large
runtime data are excluded from Git. See [running and outputs](docs/running.md)
for the checkpoint layout and [genetic maps](docs/genetic_maps.md) for formats,
units and interpretation.

## Scientific workflow

| Step | Implementation | Product |
| --- | --- | --- |
| Block discovery | `discovery/` | Missing-aware reversible-cavity 200-SNP haplotypes |
| L1–L4 assembly | `assembly/` | Supported, component-preserving chromosome haplotypes |
| Sample painting (T09) | `painting/` | Ragged diploid paths with an explicit unknown state |
| Pedigree inference (T10) | `pedigree/` | Genome-wide M0/M1/M2 calls, parent identities and ambiguity |
| Family refinement and phase correction (T11) | `refinement/` | Genotype-preserving final phase; optional imputed view |
| Recombination estimation (T12) | `recombination/` | Posterior rates, crossover intervals and observable exposure |

There is one canonical inference route. T11 does not feed back into painting or
pedigree inference, and the estimated T12 map is not automatically fed upstream.
Shared-family orientation-error evidence is enabled for T12 by default;
`--no-shared-family-evidence` disables it. Family allele imputation is optional
(`--impute-missing`); it does not redefine the genotype-preserving final-phase
product.

Pedigree Tier B is the primary supported output. M0 means **zero observed
parents**, not necessarily a biological founder. Support tiers and bootstrap
fractions are internal stability measures, not calibrated probabilities of
biological correctness. The direction model has limitations in same-depth and
missing-parent designs. Read [scientific assumptions](docs/methods.md) before
interpreting parentage or recombination rates.

The [validation notes](docs/validation.md) distinguish exact implementation
regressions, short wiring checks and fresh full-genome simulation evaluation.

## Repository layout

`haplotype_reconstruction/` contains the implementation, organized by scientific
step. `core/` supplies genotype data structures, VCF/BCF loading, maps,
checkpointing and CPU allocation. `workflows/` connects the steps for simulation,
Tropheops and AstCal. `simulation/` contains the generating model and evaluation.

`configs/` contains readable run examples. `docs/` describes supported behavior
and validation. `deliverables/` preserves the readable
[Tropheops](deliverables/tropheops/README.md) and
[AstCal](deliverables/astcal/README.md) handoffs, including explicit historical
caveats. Those exports are not claims of real-data trio ground truth.

Development experiments and old scripts are not part of the supported package.
Existing local history and large datasets are retained outside the public
source tree, under ignored work storage. Old flat-module APIs and their pickle
checkpoints are not supported by this package.
