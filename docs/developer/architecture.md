# Architecture

Science computes deterministic transitions over typed records. Stages construct
inputs, declare tasks and publish products. The engine schedules those tasks;
results readers expose checked products to exports, reports and evaluation.

## Layers

| Package | Responsibility | Dependencies |
|---|---|---|
| `core` | Numerics, ties, named streams, errors, digests, kernel declarations | Standard library and numerical primitives |
| `contracts` | Domain records, axes, units, roles, schemas, source protocols | Core |
| `store` | Containers, codec, checked readers, identities, publication | Core |
| `science` | Models, searches and shared scientific primitives | Core, contracts, declared scientific APIs |
| `engine` | Generic task execution, workers, runtime policy and run records | Core, store; generic contract validation where declared |
| `stages` | Task graph, configuration mapping, adapters, science calls | Lower layers and the explicit simulation-source route |
| `results` | Typed readers for complete or partial runs | Core, contracts, store, decoder-only stage record modules |
| `outputs` | Exports and report formatting | Core, contracts, results and the engine task protocol |
| `simulation` | Generating design, meiosis, reads and separate truth | Core, contracts |
| `evaluation` | Metrics against recorded simulation truth | Core, contracts, results, simulation truth types |
| `interface` | CLI, settings, API, status and tools | Public interfaces of the other packages |

The repository's layer tests (the unit-test tier) enforce the exact allowed edges.
Results may import stage record decoders, including the explicitly separated
`*_records` modules, but not publication or source-reading orchestration. Output
formatters receive `FounderRelease`, `ReleasedSites`, `PedigreeViews` and other
contracts; they never reach into assembly to reconstruct a missing view.

Cross-science calls use small declared APIs: selection uses discovery fitting and
assembly context functions; pedigree and phase use painting emissions and state
probabilities. The finite selection/context rounds are explicit graph nodes.
Other stage connections carry records, not private implementation imports.

## Model graph

```mermaid
flowchart TD
  I[Initialised source] --> E[Evidence per contig]
  E --> D[Discovery]
  E --> R[Relatedness across contigs]
  D --> S[Selection and context rounds]
  S --> A[Final assembly and refinement]
  A --> P[Painting]
  P --> F[Founder allele refit]
  P --> G[Pedigree across contigs]
  R --> G
  G --> H[Family phase per contig]
  P --> H
  F --> H
  H --> M[Conditional recombination]
  A --> X[Founder release and outputs]
  H --> X
```

Evidence and maps are additional inputs where the stage contract names them.
Relatedness can run as soon as all evidence exists; reconstruction does not wait
for it. Pedigree consumes frozen painting, not refit feedback. Phase and maps never
feed upstream. [Workflow cuts](../user/workflow.md) expose the same dependencies.

## Tasks and resources

A task declares its key, input/output records, importable action, resource model
and optional contig affinity. An expansion returns independent subtasks and a
gather operation under the same resource authority. Results are placed by key;
completion order does not determine array or report order.

The runtime bounds all teams by the verified core budget. It admits outer work
from measured memory pressure; predicted working-set sizes are diagnostics, not
reserved peaks. A task waiting for its children is suspended, while retained
contig state has a resident token. Resources are released on success, cancellation
and failure. A worker with live retained state cannot be recycled.

Science sees explicit teams and validated storage capabilities. It does not read
the environment, files, clocks, logs or available memory. Resident/windowed and
dense/checkpointed routes use the same arithmetic and are tested against each
other. Scientific retry schedules, candidate caps and work accounts remain fixed
parameters even when their names include bytes.

## Identities and publication

A derivation binds source payload content, upstream identities, explicit
parameters, tie settings, relevant code closure, adapters/codec and numerical
policy. Execution resources and output destinations are operational. Source
provenance can include a path or timestamp, but changing only a timestamp does not
change consumed evidence. A changed payload or incompatible derivation refuses
reuse.

The append-only store commits each product atomically. Its envelope separates
scientific products from replaceable output snapshots. Scatter processes write
only their own contig container; init and gather own the run container. Markers
are written last and aid external scheduling; checked products remain the
completion authority. A copied completed root retains all bindings.

## Runtime environment

The complete generated list of actual reads is in the
[environment reference](../reference/environment.md#environment-reads), including
indirect execution-scope variables and the `NUMBA_*` cache-key family. Science has
no environment reads. The runtime fixes numerical-library policy before imports,
then passes explicit tie settings, streams and storage capabilities into tasks.
The docs build fails if a new read is undocumented or occurs outside the runtime (`engine`).

See [contracts](contracts.md), [execution methods](../methods/execution.md),
[kernel declarations](kernels.md) and [testing](testing.md) for the executable checks.
