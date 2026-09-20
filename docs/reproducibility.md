# Reproducible simulations, evaluation and export

These tools report and package scientific results; they do not tune the model
against truth. Founder accuracy, sample-phase accuracy and pedigree accuracy
are different quantities and are reported separately.

## Portable example

No private VCF, metadata workbook or template directory is needed:

```bash
python run.py example --output work/examples/quick --seed 6100
python run.py simulate --config work/examples/quick/simulation.toml
python run.py evaluate --output work/examples/quick/run --stages all
python run.py status --output work/examples/quick/run
```

The example generates six complete, independent Bernoulli founder sequences,
80 samples (20/30/30) and 5× reads. It is a small end-to-end software example,
not an empirical diversity model or a calibrated accuracy benchmark. Three
short chromosomes can leave the entire pedigree unresolved: successful
execution does not imply that this toy genome contains sufficient evidence
for parentage. `example --contigs 22 --sites 2400 --length-bp 4800000` can
generate a larger synthetic example. For scientific comparisons, use fixed,
biologically appropriate founder templates and sufficient chromosome coverage.

Template NPZ files contain `positions` (ordered 1-based coordinates) and
`allele_probabilities` (founders × markers × two alleles). Changing template
density or chromosome length changes the scientific problem; record both.

## Generating crosses and incomplete observation

Options below belong to `[simulation]` in TOML and have equivalent
hyphenated CLI flags. They change data generation, not inference.

| Setting | Default | Meaning |
| --- | --- | --- |
| `generations` | `[20,100,200]` | Numbers of generated individuals in successive cohorts |
| `backcross_fraction` | `0.0` | Fraction of eligible matings between a generated individual and one of its actual parents |
| `backcross_start_generation` | `3` | First generated cohort with backcross matings |
| `observe_generations` | `[]` | Retain only these numbered cohorts; empty means all |
| `observed_fraction` | `1.0` | Random fraction retained after cohort selection |
| `depth` | `5.0` | Expected mean read depth |
| `generating_error_rate` | `0.02` | Read-generating error, independent of the inference likelihood |
| `depth_cv` | `0.0` | Lognormal between-sample depth coefficient of variation |
| `dropout_fraction` | `0.0` | One contiguous missing-marker tract per sample per chromosome |
| `heterozygote_alt_probability` | `0.5` | Alternate-read sampling probability at true heterozygotes |

For example, add the relevant block to a copy of your configuration:

```toml
# 200 generated individuals, exactly 160 observed.
[simulation]
seed = 6200
depth = 5.0
generations = [20, 80, 100]
observed_fraction = 0.8
```

```toml
# Observe 300 later-generation individuals only.
[simulation]
seed = 6201
depth = 5.0
generations = [20, 30, 30, 100, 100, 100]
observe_generations = [4, 5, 6]
```

```toml
# Half of F3 matings are actual-parent backcrosses; the rest are intercrosses.
[simulation]
seed = 6202
depth = 5.0
generations = [20, 30, 30]
backcross_fraction = 0.5
backcross_start_generation = 3
```

Generate the complete biological pedigree before selecting observed samples.
Unobserved individuals are removed from reads and all inference inputs; their
names remain in evaluation truth. M0/M1/M2 therefore mean zero/one/two
**observed** biological parents. Cohort labels never supply inference eligibility.
The global simulation checkpoint retains both the generated pedigree and the
observed truth table.

Random observation selection uses a separate seeded stream. Backcross options
explicitly change the pedigree-generating stream. With all new options at their
defaults, the previous pedigree, ancestry, crossover events, reads and likelihoods
are unchanged for the same seed and resources.

### Read robustness at 5×

A combined stress control might use `depth_cv = 0.6`,
`dropout_fraction = 0.1`, `heterozygote_alt_probability = 0.6` and
`generating_error_rate = 0.04`. Inference still uses its declared 2% read
error likelihood; generating truth is not handed to it.

Dropout is defined in marker count, not physical base pairs. Each chromosome
draws a sample-specific tract location and depth factor. Sample depth factors
are normalized to mean one, and non-dropout depth is increased so expected
depth over **all** markers remains 5×. This is not a model of a permanent
sample-level library-quality effect shared across chromosomes.

Use separate perturbations to attribute a failure; a combined stress control
only measures their joint effect. These controls are not a replacement for
real sequencing data, mapping-bias models or experimental breeding records.

## Founder accuracy and completeness

```bash
# Inspect any completed stages, even before pedigree inference.
python run.py evaluate --output work/reports/my_run \
  --checkpoints work/runs/my_run/checkpoints --stages all

# Explicit local-stage comparison on selected chromosomes.
python run.py evaluate --output work/reports/local \
  --checkpoints work/runs/my_run/checkpoints --contigs chr3 chr10 \
  --stages block_discovery feedback_l1 feedback_l2
```

`available` (default) reports the latest saved founder product per chromosome,
plus available downstream products. `all` reports every saved stage.
An unexecuted hierarchy level is not invented when assembly stopped early.
Explicitly requested products must exist. `--feedback-selection strict`
selects strict-feedback checkpoints when that was the run configuration.

Reports are written under `OUTPUT/evaluation/`:

- `founders.csv`: per-stage/chromosome totals and denominators;
- `founder_components.csv`: component-level accuracy and completeness;
- `founder_matches.csv`: fixed one-to-one founder assignments;
- `chromosomes.csv`: final sample-phase and conditional map metrics;
- `summary.json`: stage availability, pedigree metrics and complete results.

Nearest-founder **called allele errors** match each reconstructed row to one
truth founder across its entire component. Unknowns are excluded from the
called denominator. Duplicate rows can therefore have good precision.

One-to-one matching instead minimizes mismatches plus unknowns and penalizes
an absent founder by its entire component length. **Truth-to-panel
errors/missing** exposes incomplete panels. Extra reconstructed rows and their
called alleles are reported separately; they cannot increase truth completeness.
Founder labels are never rematched at individual SNPs.

Reported represented/absent-ancestry strata use the one-to-one matching.
“Represented” means at least one observed individual carries that ancestry;
it does not establish independent-lineage support or ancestral-phase
identifiability. Final founder calls are obtained from the assembly panel
stored in the painting checkpoint; this does **not** measure errors across
painted children. Final sample-phase metrics remain separate.

Do not compare historical nearest-neighbour completeness figures directly to
the new one-to-one completeness metric. Matching definitions and component
boundaries must agree for a controlled comparison.

## Lossless founder and sample exports

```bash
python run.py export --output work/runs/my_cross \
  --destination work/exports/my_cross --format bcf

# Abstract simulation 0/1 values do not have biological REF/ALT letters.
python run.py export --output work/examples/quick/run \
  --destination work/exports/quick --format vcf --synthetic-alleles
```

An export destination must be new or empty. `--products founders` and
`--products samples` allow separate products. Custom checkpoint roots and
explicit contig selection are supported.

Founders are gzip-compressed TSV tracks: position, then one 0/1/dot column
per released founder row. Each component has its own file. The manifest maps
columns to original row keys. Founder labels are **chromosome- and
component-local**, not identities shared across the genome. No missing allele
is filled in by the exporter.

Sample files are per-chromosome BCF or bgzip VCF with GT and PS. The original
general-reconstruction input identity supplies REF/ALT, with coordinate and
sample checks. Dataset adapters require `--vcf` specifying their original
allele basis. Phase sets reset at supported component boundaries; unsupported
phase is unphased, and unknown alleles remain dots. No GQ/PQ is invented.
Synthetic A/C labels are allowed only for simulations and are explicitly
labelled as abstract in the header and manifest.

Variant indices are not generated. If regional random access is needed, index
the exported files with an installed `bcftools index`. The original input
and scientific checkpoints remain unchanged.

## Run records and timings

Each reconstruction invocation writes a unique JSON record under
`OUTPUT/run_records/`, beside its stdout log. It records:

- command, resolved workflow settings and checkpoint identities;
- Git revision/status and hashes of Python source files, including dirty code;
- Python/platform and installed direct dependency versions;
- verified CPU affinity, host, start/end state and errors;
- stage/contig wall times, with both inclusive and exclusive durations.

Use **exclusive** durations to add stages: feedback calls assembly internally,
so summing inclusive assembly and feedback times counts the same work twice.
Unclassified time includes orchestration and work outside instrumented calls.
Timing begins at workflow entry, after CLI parsing and workflow imports.
Times include checkpoint loading/resume overhead; a resumed invocation is not
a measurement of a fresh run. CPU-hours cannot be inferred by multiplying a
wall time by the allocation size and assuming full utilization.

`status` reads only the small records. “Running” is the last saved state, not
proof that a process survived a node failure. Existing stage completion markers
remain the scientific resume authority; the run record does not replace them.
Historical runs without records are not retroactively assigned timings.

## Tested environment and citation

`pyproject.toml` declares supported dependencies. `requirements-tested.txt`
records the direct versions used on Linux x86_64/Python 3.12.0 for the release
checks. It is a tested constraint set, not a complete transitive lockfile or
a promise of portability to every operating system.

```bash
python -m pip install -c requirements-tested.txt .
```

Installation is an explicit user operation; validation does not modify an
existing environment. Runtime records preserve the versions actually used.

Original project software is MIT licensed; see `LICENSE`. Dependencies and
external datasets retain their own terms. `CITATION.cff` records Hadi Khan's
ORCID and the repository URL. Citation is a scholarly request, not an additional
licence restriction. Private manuscript files are not part of the software
release.
