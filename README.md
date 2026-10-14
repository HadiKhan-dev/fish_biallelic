# Founder haplotype reconstruction

Reconstruct founder haplotypes, sample ancestry and pedigrees from low-coverage
sequencing of experimental crosses. The pipeline handles missing observations
from local discovery through final phase and recombination-map estimation.

The primary real-data applications are cichlid crosses. Known-pedigree
simulations support validation; the software does not require pedigree truth
or generation metadata for general reconstruction.

## Quick start

Python 3.11 or newer is required. Install into your chosen environment:

```bash
python -m pip install .
haplotypes --help
```

With the declared dependencies already installed, `python run.py` and
`python -m haplotype_reconstruction` provide the same interface.

```bash
# General cross: indexed, sorted, biallelic SNP VCF/BCF with FORMAT/AD.
python run.py reconstruct --vcf cross.bcf --output work/runs/my_cross --contigs chr1 chr2 chr3

# Self-contained synthetic example (no private input data).
python run.py example --output work/examples/quick
python run.py simulate --config work/examples/quick/simulation.toml

# Known-pedigree simulation from supplied founder-sequence templates.
python run.py simulate --config configs/simulation.toml --seed 400

# Dataset-specific sample and breeding-design policies.
python run.py tropheops --config configs/tropheops.toml
python run.py astcal --config configs/astcal.toml
```

Edit the paths in `configs/` for your data. Empirical input datasets and founder
templates are not bundled; the `example` command generates synthetic ones.
General reconstruction requires allele depths; GT-only and PL-only input are
not supported. See [inputs and commands](docs/running.md)
and the [general configuration example](configs/reconstruct.toml).

The small `example` exercises file formats, stage handoffs and checkpointing.
Its few short, sparse chromosomes are not a pedigree-accuracy benchmark and
may leave relationships unresolved. Use the documented whole-genome controls
to assess biological reconstruction accuracy.

On CSD3, activate the existing project environment:

```bash
conda activate /rds/user/ahk39/hpc-work/conda_envs/bio-env
```

Run substantial computation only on allocated compute nodes. The default CPU
ceiling is the process's CPU affinity; `--cores` can reduce it. Process pools
and numerical threads share that ceiling. Dynamic allocation gives freed
threads to surviving workers at supported kernel boundaries.

## One scientific pipeline

| Stage | Product | Implementation |
| --- | --- | --- |
| Observation calibration | Data-fitted read error, sample effects and dispersion | `core/read_calibration.py` |
| Local discovery | Missing-aware 200-SNP founder panels | `discovery/` |
| Local fitting and two feedback rounds | Normalized path selection initially, after L1 and after L1+L2 | `workflows/block_feedback.py` |
| Final L1–L4 assembly | Refined, component-preserving founder chromosomes | `assembly/` |
| Sample painting | Diploid sample mosaics with an explicit unknown state | `painting/` |
| Pedigree inference | Genome-wide M0/M1/M2 calls, parent identities and ambiguity | `pedigree/` |
| Family refinement and final phase | Genotype-preserving final phase | `refinement/` |
| Recombination maps | Conditional rates, crossover intervals and observable exposure | `recombination/` |

All four reconstruction commands share the assembly/painting runner and the
one-way pedigree → family phase → recombination handoff. Dataset adapters supply
observations and eligibility; they do not define alternative inference engines.

Missing alleles remain unknown unless the relevant model supports a call.
Disconnected components are not silently joined into a common phase frame.
Tier B is the primary pedigree output. **M0 means zero observed parents**,
not necessarily a biological founder. Support tiers and bootstrap fractions
measure internal stability, not calibrated probabilities of correct parentage.

Family phase preserves called genotypes and missingness. Phase polishing scores
called scaffold alleles only; no separate imputed-genotype product or full
marginal posterior tensors are released. Neither family phase nor the estimated
recombination map feeds back upstream. Shared-family orientation-error evidence is on by default
for recombination; `--no-shared-family-evidence` disables it.

### Deliberate algorithm choices

- Assembly has one join route at every level and in the feedback context
  assemblies: learned dense transitions, the endpoint beam and the bounded
  panel search. The former structured transition model and broad search were
  removed (old settings at their defaults are ignored with a warning).
- Local panels use normalized path selection before and after **each** L1/L2
  feedback round; this is the only feedback selection.
- Feedback context refits and the missing-allele completion share one site
  posterior over shared founder alleles: exact for at most ten free founders,
  a polynomial uncertain-core approximation above (no founder-count cap). See
  the [local feedback model](docs/methods.md#local-feedback-and-candidate-selection).
- The join decides founder counts with the pipeline's effective-carrier rule
  (at every level and in the feedback context assemblies): a single birth or
  death must pass the test on each sample's total gain and on its sequence
  part, so switch savings alone never justify a path, and a path without a
  private cell or within 1% of another path over the span is deleted. Same-count
  moves are judged by the join likelihood. See the
  [panel search](docs/methods.md#hierarchical-assembly). No biological founder
  count is supplied.
- Founder-path refinement runs after each executed final L1–L4 level: converged
  Gauss-Seidel window and suffix-exchange sweeps with full-site acceptance, and
  founder-count decisions between converged fits by the same rule as the join
  (total gain and its sequence part, the sequence part from an exact shadow
  without the join's 1% near-identity shortcut; founders without a private cell
  are deleted first; additions only at the last level), keeping the best-F
  converged panel at the decided count. See the [model and limitations](docs/methods.md#final-founder-path-refinement).
  `--founder-refinement off` disables it for controlled comparisons.
- Read calibration learns supported observation parameters from allele depths,
  including a nested homozygote-overdispersion model selected on held-out data;
  `--no-read-calibration` retains the fixed-error model. See the
  [calibration model and accepted limitations](docs/read_calibration.md).
  Pedigree evidence separately learns a cross-chromosome predictive scale;
  `--pedigree-calibration off` disables that estimator.
- Inference defaults to 5 cM/Mb. `--recombination-map cross.map` supplies a
  spatially varying cumulative genetic map; unmapped chromosomes retain the
  fallback rate. Simulation-generating maps are configured independently.

Read the [methods and assumptions](docs/methods.md), [founder scaling](docs/founder_scaling.md)
and [genetic-map guide](docs/genetic_maps.md) before choosing a comparison model.

## Outputs and reproducibility

Rerun the same command to resume compatible checkpoints. A run directory
contains logs, readable tables and its checkpoints: one file per chromosome,
growing as its stages complete and holding every intermediate stage, plus one
genome-wide file. An interruption loses only the work in progress, and
`haplotypes status --output RUN` shows how far each chromosome has got
([checkpoints and outputs](docs/running.md#checkpoints-and-outputs)).
For short allocations, `--serial-contigs` and `--max-contigs N` finish whole
chromosomes one at a time ([running in short allocations](docs/running.md#running-in-short-allocations)).
Use a new output root when changing upstream inputs or scientific settings;
never relabel old checkpoints as current.

Pedigree independently versions preparation, genetic scores and decisions.
Direction/release-policy comparisons can reuse compatible preparation and
scores. Outputs include call explanations, candidate ambiguity, search-limit
diagnostics, and founder callability/carrier-support tables.
See the [output and cache guide](docs/running.md#pedigree-cache-reuse-and-explanations).

Use `evaluate --stages all` for local and chromosome-wide founder accuracy,
`export` for lossless founder tracks and sample VCF/BCF, and `status` for saved
progress and exclusive stage timings. See the [reproducibility guide](docs/reproducibility.md)
for incomplete-pedigree simulations, 5× robustness controls and tested dependency versions.

Validation and timing are described separately:

- [Validation record](docs/validation.md): controls, denominators and known
  failures, distinguishing short integration checks from full simulations.
- [Pedigree direction](docs/pedigree_direction.md): finite orientation and
  family evidence, approximations and limits.
- [Performance](docs/performance.md): measured timings and scaling trade-offs.
- [Code map](docs/development.md): module ownership, parallelism and checkpoint
  boundaries.
- [Storage utility](tools/README.md): parallel allocated-space measurements for
  large checkpoint trees, implemented in Python with an optional C accelerator.
  A native-backend validation scan took 2 minutes 8 seconds for 4.66 million
  entries on 27 September 2026; see the utility notes for build instructions,
  validation, and filesystem-cache caveats.

Founder reconstruction can be ambiguous when ancestry is unsampled or many
descendants repeat one ancestral recombinant. Painting errors and structured
missingness can affect downstream pedigree direction. Successful simulations
do not establish individual-level trio truth in real crosses or guarantee
uniform accuracy. See the validation record for the unresolved N320 seed407
chr15 refinement regression and other specific limitations.

## Repository layout

`haplotype_reconstruction/` contains the scientific implementation.
`configs/` contains editable run examples; `docs/` contains methods and usage.
`deliverables/` preserves readable AstCal and Tropheops handoffs with their
original provenance; these are not silently regenerated by a code refactor.

Large inputs, checkpoints and private experiments live under ignored `work/`
and `.work/`. The manuscript is private and ignored. No parallel copy of the
pipeline is shipped as a compatibility implementation.

## Licence and citation

Original project software is [MIT licensed](LICENSE), copyright 2026 Hadi Khan.
Dependencies and external datasets retain their own terms. Please cite the
software in scientific work using the metadata in [CITATION.cff](CITATION.cff).
This citation request is not an additional licence condition.
