# Code map and development guide

Start with [running the pipeline](running.md) for commands and outputs, or
[methods](methods.md) for the statistical models. This guide explains where
the implementation lives and how its parts fit together.

## Follow one run

`run.py`, `python -m haplotype_reconstruction`, and the installed `haplotypes`
command all enter `haplotype_reconstruction/cli.py`. The CLI resolves run
settings before importing the selected workflow. This ordering matters:
workflow configuration and numerical-library limits are read during import.

The three reconstruction drivers live in `workflows/`: `simulation.py`,
`astcal.py`, and `tropheops.py`. They share `reconstruction.py` for local
feedback, component assembly and painting, then call the pedigree, family
refinement and recombination pipelines. The simulation driver additionally
generates observations and retains truth for evaluation; inference does not
receive that truth.

| Step | Start reading here | Main responsibility |
| --- | --- | --- |
| Input and observations | `core/variants.py`, `core/genotypes.py` | Marker blocks, allele depths and genotype likelihoods |
| Local discovery | `discovery/blocks.py`, `discovery/search.py` | Missing-aware reversible search for 200-SNP panels |
| Feedback selection | `workflows/block_feedback.py`, `discovery/candidate_selection.py` | Read-supported selection after each L1 and L1+L2 context round |
| Founder completion | `assembly/completion.py`, `assembly/joint_completion.py` | Evidence-supported local completion and frozen inference panels |
| L1–L4 assembly | `assembly/pipeline.py`, `assembly/hierarchy.py` | Checkpointed hierarchy, component boundaries and scheduling |
| Founder-path refinement | `assembly/founder_refinement.py` | Coordinate the refinement passes after each final hierarchy level |
| Sample painting | `painting/model.py`, `painting/components.py` | Component-local ragged founder mosaics, including unknown states |
| Pedigree inference | `pedigree/pipeline.py`, `pedigree/inference.py` | Aggregate chromosome evidence and infer observed-parent states and identities |
| Family phase correction | `refinement/pipeline.py`, `refinement/polish.py` | Pedigree-conditioned, genotype-preserving final phase |
| Recombination maps | `recombination/pipeline.py`, `recombination/model.py` | Conditional rates, crossover intervals and observable exposure |
| Truth evaluation | `simulation/metrics.py` | Evaluate cached simulated outputs without feeding truth into inference |

Paths in this table are relative to `haplotype_reconstruction/`.

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
| `exchanges`, `intervals`, `count` | Paired-path moves and founder-count comparisons |
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

`inference.py` combines chromosomes and applies the decision procedure.
`direction.py`, `eligibility.py`, `states.py`, `bootstrap.py` and `graph.py`
separate chronology/eligibility, parent-count evidence, resampling and acyclic
graph selection. `results.py` produces the output tables. Generation labels
and real-data candidate eligibility must not be confused with known parentage.

## Where CPU allocation lives

`core/environment.py` sets numerical-library limits before heavy imports.
`core/parallel.py` owns shared arrays, forkserver pools, Numba thread scopes
and the dynamic worker counter. Its `get_dynamic_threads()` and
`apply_dynamic_threads()` redistribute the existing core budget as workers
finish. An already-running kernel cannot absorb new threads mid-call.

Assembly scheduling is in `assembly/hierarchy.py`; refinement candidate
scheduling is in `assembly/founder/candidates.py`. Process count and threads
per process share one ceiling. Do not add independent thread pools or reorder
environment-sensitive imports merely to satisfy a style rule. Checkpoint I/O,
memory bandwidth and ordered acceptance steps can still limit utilization.

## Configuration, types and checkpoints

Run examples live in `configs/`. `cli.py` resolves command-line, TOML and
supported environment settings. Stage-owned configuration classes live beside
their implementation; `core/config.py` contains shared scientific constants.
The [running guide](running.md) distinguishes broad-only assembly controls
from bounded-search budgets.

`core/haplotypes.py` owns shared block types. `core/checkpoints.py` implements
compressed atomic serialization; `core/runtime.py` owns run-stage stores.
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
