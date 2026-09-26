# Code map and development guide

Start with [running the pipeline](running.md) for commands and outputs, or
[methods](methods.md) for the statistical models. This guide explains where
the implementation lives and how its parts fit together.

## Follow one run

`run.py`, `python -m haplotype_reconstruction`, and the installed `haplotypes`
command all enter `haplotype_reconstruction/cli.py`. The CLI resolves run
settings before importing the selected workflow. `build_parser()` describes
commands; the inference and simulation setup functions keep model settings
separate from generating parameters. This ordering matters:
workflow configuration and numerical-library limits are read during import.

The four reconstruction drivers live in `workflows/`: `variants.py` provides
the general AD-VCF/BCF route, `simulation.py` generates known-pedigree data, and
`astcal.py` / `tropheops.py` apply their dataset-specific sample policies.
All share `reconstruction.py` for local feedback, assembly and painting, then
`downstream.py` for the one-way pedigree → family phase → recombination handoff.
Simulation truth is retained for evaluation and never supplied to inference.
Plotting libraries are loaded only by the optional simulation-pedigree
plotter; painting types, checkpoint readers and export do not initialize
graphics or font caches.

`reports.py` holds discovery-search summaries and observed-reference
consistency reports. Reporting neither selects haplotypes nor establishes
individual parentage. G0 reference agreement is explicitly distinguished from
known-truth simulation validation.

| Step | Start reading here | Main responsibility |
| --- | --- | --- |
| Input and observations | `core/variants.py`, `core/genotypes.py` | Marker blocks, allele depths and genotype likelihoods |
| Observation calibration | `core/read_calibration.py`, `core/read_model.py`, `core/read_homozygote_model.py` | Fit nested read models, including pooled homozygote overdispersion; kernels and matching GL application live in `read_kernels.py` and `read_likelihoods.py` |
| Local discovery | `discovery/blocks.py`, `discovery/search.py` | Missing-aware reversible search for 200-SNP panels |
| Local path selection | `discovery/path_model.py`, `path_scoring.py`, `path_fitting.py`, `path_selection.py` | Normalized diploid-path objective, prepared scoring, fixed-K fitting and bounded drop/merge/add search |
| Feedback orchestration | `workflows/block_feedback.py`, `discovery/path_blocks.py` | Initial path selection, L1 and rebuilt L1+L2 feedback, block materialization and checkpointed parallel work |
| Candidate banks / alternative selectors | `discovery/candidate_selection.py`, `discovery/candidate_rescue.py` | Partial-row completion and BIC/source-endpoint starts; explicit balanced/strict cavity-rescue alternatives |
| Optional local segment exchange | `discovery/path_exchange.py` | Fixed-K reciprocal suffix starts, exact all-pairs/cuts screening and protected top-eight plus warm-control refits; off by default |
| Founder completion | `assembly/completion.py`, `assembly/joint_completion.py` | Evidence-supported local completion and frozen inference panels |
| L1–L4 assembly | `assembly/pipeline.py`, `assembly/hierarchy.py` | Checkpointed hierarchy, component boundaries and scheduling |
| Founder-path refinement | `assembly/founder_refinement.py` | Coordinate the refinement passes after each final hierarchy level |
| Sample painting | `painting/model.py`, `painting/components.py` | Component-local ragged founder mosaics, including unknown states |
| Pedigree inference | `pedigree/pipeline.py`, `pedigree/inference.py` | Aggregate chromosome evidence and infer observed-parent states and identities |
| Family phase correction | `refinement/pipeline.py`, `refinement/polish.py` | Pedigree-conditioned, genotype-preserving final phase |
| Recombination maps | `recombination/pipeline.py`, `recombination/model.py` | Conditional rates, crossover intervals and observable exposure |
| Truth evaluation | `simulation/evaluation.py`, `simulation/founder_metrics.py`, `simulation/metrics.py`, `simulation/truth.py` | Parallel founder/sample/pedigree metrics and compact cached truth, never inference inputs |
| Portable simulation controls | `simulation/designs.py`, `simulation/example.py` | Backcrosses, observed-only sampling, read perturbations and generated examples |
| Interoperable export | `workflows/export.py`, `core/products.py` | Lossless component-local founder tracks and final sample GT/PS |
| Run provenance and timing | `core/run_record.py` | Coordinator records and non-overlapping wall times |

Paths in this table are relative to `haplotype_reconstruction/`.

Local path selection and optional exchange consume the original calibrated GLs;
assembly panels only propose starts. The path modules contain no workflow paths,
simulation truth or private-pilot imports. `path_blocks.py` preserves kept-marker
axes, ALT probabilities, fractional support and spatial MAP unknown counts.
`PathSelectionConfig` in `core/config.py` owns the shared three-round,
eight-refits-per-kind, twenty-update budgets. Ordinary search reuses prepared
observations and exact panel-plus-frequency score/fit results within one block;
the suffix screen reuses forward/backward messages without assuming equal
founder frequencies. These are numerical optimizations, not reduced proposal
sets.

## Assembly: keep the different problems separate

The linker and founder refiner solve different problems. `linking.py`,
`micro_hmm.py`, `macro_hmm.py` and `edge_counts.py` estimate linkage between
blocks. `paths.py` constructs paths and reconstructs their alleles.
`panel_search.py` owns bounded candidate selection; `chimera_resolution.py`
owns the optional broader search. Both use the declared scoring objective.

`founder_refinement.py` is the coordinator for the next layer. Its numerical
and execution helpers are grouped in `assembly/founder/`:

| Modules | Responsibility |
| --- | --- |
| `path_search`, `beam`, `dual_search`, `windows` | Conditional path and interval proposals |
| `exchanges`, `intervals`, `count`, `count_increase` | Paired-path moves, progressive count reductions and final count-up/refit |
| `scoring`, `predictive`, `evidence` | Cohort likelihoods, missing-founder evidence and model preparation |
| `workspace`, `packing`, `background`, `delta` | Reusable arrays and localized score evaluation |
| `candidates`, `components`, `count_workers` | Bounded concurrent work and straggler thread reallocation |
| `checkpoints` | Resume records for refinement passes |
| `beam_kernels`, `dual_short`, `site_kernels`, `sparse`, `count_bound` | Specialized numerical kernels and proposal screening |

These helpers do not use pedigree truth or replace sample painting. Their
internal sample fits score founder proposals. The separate top-level
`refinement/` package performs **family phase correction after pedigree
inference**. Keeping these two meanings of refinement distinct is important.

## Pedigree evidence and decisions

`pedigree/components.py` prepares ragged painting evidence and source inputs.
`sources.py` represents candidate ancestry; `transmission.py` validates and
coordinates the projected M0/M1/M2 likelihood calculation. Its public model
and score types remain there, while the numerical work is separated into:

- `transmission_projection.py`: transmitted-source marginals and
  persistence-constrained bridges;
- `transmission_scoring.py`: quadratic forward scoring kernels.

`cache.py` independently versions preparation, genetic scores and decisions;
`explanations.py` separates score contributions and release/ambiguity summaries.
`execution.py` keeps chromosome tensors and reusable projections in persistent
workers; `inference.py` combines their scores and applies the decision procedure.
`direction.py`, `eligibility.py`, `states.py`, `bootstrap.py` and `graph.py`
separate chronology/eligibility, parent-count evidence, resampling and acyclic
graph selection. `calibration.py` learns the conditional predictive evidence
scale; `exclusion.py` and `exclusion_patterns.py` supply raw-GL Mendelian
exclusion; `release.py` reconciles released parent-count states without
introducing new edges. `results.py` produces the output tables. Generation labels
and real-data candidate eligibility must not be confused with known parentage.

## Where CPU allocation lives

`core/environment.py` sets numerical-library limits before heavy imports.
`core/parallel.py` owns shared arrays, forkserver pools, Numba thread scopes
and the dynamic worker counter. Its `get_dynamic_threads()` and
`apply_dynamic_threads()` redistribute the existing core budget as workers
finish. An already-running kernel cannot absorb new threads mid-call.

`core/chromosome_parallel.py` supplies persistent chromosome workers and bounded
thread leases for pedigree preparation/scoring, recombination and evaluation.
Its `current_threads()` lets surviving workers claim released cores at explicit
numerical boundaries. Genome-wide candidate selection remains a barrier between
pedigree screening and detailed scoring; parallelism does not change that model.
Recombination and evaluation limit concurrent decoded chromosomes using
`core/runtime.py` memory accounting without reducing the shared CPU ceiling.

Assembly scheduling is in `assembly/hierarchy.py`; refinement candidate
scheduling is in `assembly/founder/candidates.py`. Process count and threads
per process share one ceiling. Do not add independent thread pools or reorder
environment-sensitive imports merely to satisfy a style rule. Checkpoint I/O,
memory bandwidth and ordered acceptance steps can still limit utilization.

## Configuration, types and checkpoints

Run examples live in `configs/`. `cli.py` resolves command-line, TOML and
supported environment settings. Stage-owned configuration classes live beside
their implementation; `core/config.py` contains shared scientific constants.
`pedigree/config.py` owns both `PedigreeConfig` and its environment adapter;
`workflows/design.py` only supplies real-cross eligibility and chronology,
not a second copy of the model defaults. `core/environment.py` supplies the
shared boolean parser used by the CLI, read calibration and recombination.
The two cichlid adapters and empirical-template simulation share their
reference chromosome order in `workflows/__init__.py`; general VCF
reconstruction takes its contigs from the input header or explicit CLI list.

The [running guide](running.md) distinguishes broad-only assembly controls
from bounded-search budgets.

`core/haplotypes.py` owns shared block types. `core/checkpoints.py` implements
compressed atomic serialization with rolling chunk reads and ordered pipelined
writes; checksums and final atomic publication are retained. `core/runtime.py`
owns run-stage stores and scoped logging. Each workflow restores stdout and
closes its log on normal
return, interruption or failure.
Assembly and painting add their own typed checkpoint and scientific-identity
handling. Keep public serialized types at stable module paths when possible.

Moving a numerical helper requires updating its imports **and** every relevant
source-dependency list used for checkpoint identity. Source changes can make
old work incompatible even when mathematical behavior is unchanged. Never
relabel a stored identity to force reuse. Frozen campaign source and its
checkpoints should remain together when development continues in the main
checkout. For an isolated campaign, launch from the frozen source directory
with `PYTHONPATH` pointing there; `PYTHONSAFEPATH=1` also prevents the child
interpreter's current-directory entry from taking precedence. Verify the
package path in a forkserver worker as well as in the parent: a frozen entry
script alone does not isolate preloaded worker imports.

For numerical comparisons, copy a kernel module into an isolated source tree
or use a separate `NUMBA_CACHE_DIR`. Do not import a canonical `@njit(cache=True)`
source file under a temporary module alias: the generated cache can retain
that alias and fail to load in ordinary production imports. Experimental source,
compiled caches and scientific checkpoints are separate kinds of artifact.

## Making a behavior-preserving cleanup

1. Preserve a source snapshot and inspect existing working-tree changes.
2. Move cohesive responsibilities, not arbitrary line-count slices. Keep
   numerical loops intact and multiprocessing workers at module scope.
3. Verify executable syntax-tree equivalence for formatting and mechanical
   moves; check imports and relocated source-dependency paths separately.
4. Exercise the affected public entry paths, serialization and small numerical
   fixtures. Run scientific comparisons when a model or decision changes.
5. Keep validation scripts and generated artifacts under ignored work storage.
   Record what was tested and what was not.

See [validation](validation.md) for scientific evidence and known limitations,
and [performance](performance.md) for measured timings rather than assumptions
about parallelism. `deliverables/` contains preserved readable real-data
handoffs; `manuscript/` is private and ignored. Neither belongs in a code-style
cleanup or generated-data purge.
