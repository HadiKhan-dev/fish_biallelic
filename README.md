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

Python 3.11 or newer is required. From this checkout, install the package into
your chosen environment to obtain its dependencies and the `haplotypes` command:

```bash
python -m pip install .
haplotypes --help
```

Alternatively, use `python run.py` directly when the dependencies declared in
`pyproject.toml` are already installed. Input data and simulation founder
templates are not bundled; see [input requirements](docs/running.md#inputs).
On CSD3, the existing project environment is:

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
| Local feedback selection | `workflows/block_feedback.py`, `discovery/` | Read-supported selection after each L1/L2 feedback round; balanced rescue by default |
| L1–L4 assembly | `assembly/` | Supported, component-preserving chromosome haplotypes |
| Sample painting (T09) | `painting/` | Ragged diploid paths with an explicit unknown state |
| Pedigree inference (T10) | `pedigree/` | Genome-wide M0/M1/M2 calls, parent identities and ambiguity |
| Family refinement and phase correction (T11) | `refinement/` | Stable, genotype-preserving final phase |
| Recombination estimation (T12) | `recombination/` | Posterior rates, crossover intervals and observable exposure |

T11 corrects phase while preserving called genotypes and missingness. It does
not publish a separate family-imputed allele product or full marginal posterior
tensors; internal family-supported fills only inform phase polishing.

Assembly defaults to dense learned transitions plus bounded panel search.
Before final assembly, two checkpointed feedback rounds refine local panels:
L1 → blocks → selection, then L1+L2 → blocks → selection. The first selected
panels feed the second context; original candidates remain available throughout.
Use `--feedback-selection strict` to protect each round's clean feedback calls
and rescue only private alleles instead of the default balanced trade-off.
Entirely uncalled feedback rows are now tested for evidence-supported removal
and refitting after each round; necessary unknown rows remain explicit.
Linking uses partially called founder panels without filling missing alleles
merely to connect blocks. Neither change forces a known founder count.
See [local feedback selection](docs/running.md#local-feedback-selection) and
[partial-founder assumptions](docs/methods.md#hierarchical-linking).
Use `--assembly-model structured` for the
[near-quadratic alternative](docs/founder_scaling.md); this restricts the
transition model and does not change discovery. Use `--assembly-search broad`
to retain the optimized broader path/panel search instead of the default
`bounded` search; this costs more full refits and is not guaranteed to improve
accuracy. Final assembly also reopens the original local row choices in a
component-local founder refinement after **each executed L1–L4 level** of
final assembly, never within the two local feedback rounds. Each complete
refinement feeds the next hierarchy level. If local search stalls, original
and refined L1 pieces supply larger proposals, with a genotype-fit guard
against penalty-only improvements.
The completed beam/dual search is followed by bounded paired exchanges,
one-founder deletion/refitting under the existing complexity cost, and
exact-flank window searches. Paired-interval polishing can repair coordinated
two-founder errors while preserving every local called/missing allele.
These passes use genotype evidence, not pedigree, generation labels or a known
founder count. They are enabled by default and checkpointed separately.
Use `--founder-refinement off` to disable all final-assembly refinement passes
for a comparison.
See [the model and its limits](docs/methods.md#final-founder-path-refinement).
An earlier N320 replay joined all eight fragmented controls but exposed a
final-refinement accuracy regression on seed407 chr15. The progressive N80
validation does not establish that this N320 case is fixed; it remains an
[explicit limitation requiring revalidation](docs/validation.md#n320-fragmentation-replay),
not evidence of a uniformly improved release.
T11 does not feed back into painting or pedigree inference, and the estimated
T12 map is not automatically fed upstream.
Shared-family orientation-error evidence is enabled for T12 by default;
`--no-shared-family-evidence` disables it. T11 releases final phase after five
consecutive unchanged phase checks (starting after 20 family iterations), or
actual family convergence. It preserves called genotypes and missingness; it
does not publish full marginal-posterior tensors. If the 520-iteration solve
remains unstable, it retries from the scaffold up to twice with half damping
each time, retaining separate checkpoints; an unstable final attempt still
produces no release.

Pedigree Tier B is the primary supported output. M0 means **zero observed
parents**, not necessarily a biological founder. Support tiers and bootstrap
fractions are internal stability measures, not calibrated probabilities of
biological correctness. The direction model has limitations in same-depth and
missing-parent designs. Read [scientific assumptions](docs/methods.md) before
interpreting parentage or recombination rates.

The [validation notes](docs/validation.md) distinguish exact implementation
regressions, short wiring checks and fresh full-genome simulation evaluation.
[Performance notes](docs/performance.md) separate numerical speedups from
scientific trade-offs.

The latest [fresh end-to-end validation](docs/validation.md#fresh-seed3000-end-to-end-validation)
uses seed3000, 320 samples and 5× mean depth. All 22 chromosomes contain six long
founder haplotypes; founder mismatches total 2,581 / 53,740,022 called alleles,
and metadata-free pedigree inference recovers all 320 configurations exactly.
This is one successful simulation, not a guarantee across datasets. The
[single-76-core runtime estimate](docs/performance.md#complete-workflow-timing)
is approximately 5 hours 10 minutes.

## Repository layout

The [code map and development guide](docs/development.md) explains module
responsibilities, the two refinement stages, thread allocation and checkpoints.

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
