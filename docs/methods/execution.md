# Execution, storage and interfaces

The dependency graph expresses model prerequisites; one runtime owns process, CPU, memory and compilation effects. Scientific functions receive explicit typed parameters, arrays, streams and capabilities. Products resume only under compatible derivations. Runtime scheduling and validated storage routes must preserve scientific decisions. T is tasks, C contigs, R records and D decoded bytes in the cost notes.

The sections below identify semantic units, not separately configurable pipeline steps. [Outputs](../user/outputs.md), [record contracts](../reference/records.md) and [costs](../developer/costs.md) describe their public boundaries.

<a id="eng-01"></a>
## eng-01 Stage registry and pipeline order

Describe stages, record dependencies and legal command cuts as graph data. Registry order is a naming convention, not execution order. Reject unknown stage names before scheduling. Stage modules supply tasks and typed adapters; the generic graph engine contains no hard-coded scientific imports.

Implementation: [stages/graph.py](../../haplotype_reconstruction/stages/graph.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(C) graph nodes per fixed stage set, plus explicit dynamic tasks.

<a id="eng-02"></a>
## eng-02 Fail-fast pre-run checks and run preparation

Resolve cores within process affinity, validate source/sample/contig requirements, resolve parameters and maps, and check checkpoint layout before work. A full run checks configured-contig reach before pedigree later checks informative reach. Bind one resolved run configuration and realised simulation seed. Invalid user input fails at this boundary with an explicit reason.

Implementation: [interface/run.py](../../haplotype_reconstruction/interface/run.py), [stages/graph.py](../../haplotype_reconstruction/stages/graph.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(contigs). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="eng-03"></a>
## eng-03 Driver stage sequence and global barriers (dependency graph)

Execute only declared dependencies: relatedness waits for all evidence, reconstruction needs its own contig evidence, pedigree needs relatedness and all painting, and phase needs pedigree, painting and refit. Refit may start immediately after painting. Expansion/gather and critical-path scheduling do not create scientific barriers. Downstream phase and recombination never feed back into upstream models.

Implementation: [stages/graph.py](../../haplotype_reconstruction/stages/graph.py), [interface/run.py](../../haplotype_reconstruction/interface/run.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(T) dependency scheduling plus stage work; keyed gather retains order independently of completion.

<a id="eng-04"></a>
## eng-04 Run modes: stop points, shards, serial mode, max_contigs

Implement run, workflow steps, stop points, serial mode and contig limits as cuts of the same graph. Serial visits contigs in bound order; a contig limit counts pending contigs and stops cleanly. Workflow scatter commands consume checked prerequisites after init. Removed shards and helper queues are replaced by these explicit steps. Every mode retains identical derivation checks.

Implementation: [stages/graph.py](../../haplotype_reconstruction/stages/graph.py), [interface/run.py](../../haplotype_reconstruction/interface/run.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Graph-cut selection scales with graph size; serial mode changes concurrency only.

<a id="eng-05"></a>
## eng-05 Per-contig reconstruction unit (contig_work)

Carry one contig-affine founder evidence view through selection rounds, context assemblies and final assembly, reusing checked intermediate publications. Final assembly may consume its private workspace, which must not be reused afterwards. Release it before painting, whose own capability reads all-sample evidence. Reports consume published results rather than retaining assembly internals.

Implementation: [stages/graph.py](../../haplotype_reconstruction/stages/graph.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Resident state scales with decoded discovery/evidence and assembly workspaces; the per-contig view avoids repeated full copies.

<a id="eng-06"></a>
## eng-06 Stage and code identities

Compute derivation identities from explicit parameters, upstream bindings, input construction, adapter/codec code closure, numerical policy and tie settings. Bind consumed source content rather than timestamps. Distinguish the parent-edge conditioning digest from its product derivation. Resource allocations, output paths and output selections do not enter scientific identities.

Implementation: [core/digest.py](../../haplotype_reconstruction/core/digest.py), [store/identity.py](../../haplotype_reconstruction/store/identity.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Source-closure hashing per distinct code identity plus canonical small-record serialisation.

<a id="eng-07"></a>
## eng-07 Identity binding, resume and invalidation graph

Read or publish scientific products only through identity-checked protocol requests. Bind the expected derivation before reusing data and refuse mismatches rather than overwriting scientific history. Propagate upstream derivations through the graph and bind independent source payloads at their first reader. A copied checkpoint root retains its bindings; changing a path alone does not create an independent result.

Implementation: [store/store.py](../../haplotype_reconstruction/store/store.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Identity/control-record reads, without decoding unrelated scientific arrays.

<a id="eng-08"></a>
## eng-08 Completion predicates, completion markers and resume granularity

Decide completion from compatible required products, never from a marker alone. Write workflow markers last after the corresponding cut has checked its products. Resume at each stage’s actual publication granularity; unfinished phase work is recomputed at contig scope. External workflow tools may use markers for scheduling, but stale markers cannot authorise scientific reuse.

Implementation: [store/store.py](../../haplotype_reconstruction/store/store.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Required-key/index checks per task or command cut.

<a id="eng-09"></a>
## eng-09 Configuration loading and validation

Load schema defaults, selected preset, TOML, ordinary CLI values and explicit overrides in declared precedence. Resolve paths against the invocation directory, reject unknown keys and validate types plus cross-key constraints. Translate supported older configurations exactly or refuse them with a reason. The current schema, not an old prose example, defines accepted settings.

Implementation: [interface/settings/migrate.py](../../haplotype_reconstruction/interface/settings/migrate.py), [interface/settings/load.py](../../haplotype_reconstruction/interface/settings/load.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Linear in configuration size; no scientific work.

<a id="eng-10"></a>
## eng-10 User settings: classification and mapping to stage configs

Map user settings to the declared parameter fields and resolve auto values from source kind and validated resources. Keep scientific controls separate from scheduling, storage, diagnostics and output selection. Persist the resolved scientific configuration; allow invocation-specific pairwise/export selections outside its identity. The generated settings reference lists the source mapping and defaults.

Implementation: [stages/configure.py](../../haplotype_reconstruction/stages/configure.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Linear in schema fields and resolved contig descriptors.

<a id="eng-11"></a>
## eng-11 Stage-config constants: classification

Declare every model, search, release and cost parameter by name with provenance in the owning parameters module. Stage configuration supplies those records explicitly. Scientific retry/work budgets remain scientific even when expressed in bytes; validated storage limits affect representation only. Generated references expose defaults without duplicating them in method prose.

Implementation: [stages/configure.py](../../haplotype_reconstruction/stages/configure.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Linear in parameter declarations; actual K-dependent costs belong to their owning methods.

<a id="eng-12"></a>
## eng-12 Environment switches (hidden state)

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Keep environment access at the application/runtime boundary and document every variable actually read. Pass tie reversal and census options as explicit settings, never hidden science state. Establish the fixed numerical policy before importing or starting numerical runtimes. Unsupported historical environment names confer no behaviour.

Implementation: [core/ties.py](../../haplotype_reconstruction/core/ties.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Constant runtime setup plus explicit diagnostic configuration.

<a id="eng-13"></a>
## eng-13 CLI commands and option mapping

Expose commands through one parser and schema, with run and workflow steps selecting the shared graph. Standalone export consumes released records; recombination consumes bound phase and edge products and refuses conflicting mapped settings. Status inspects checked publication state. Generate the command reference directly from parser declarations so help and documentation share one source.

Implementation: [interface/cli.py](../../haplotype_reconstruction/interface/cli.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Parser and command dispatch overhead; selected graph work is separate.

<a id="eng-14"></a>
## eng-14 Checkpoint container: append-only records, crash safety, concurrency

Append checkpoint records under the container’s lock and commit envelope/trailer protocol. Recover only a recognisable torn tail; refuse other corruption. Readers scan complete records and refresh their index when the file changes. Scientific publication rejects a conflicting existing derivation, while explicitly designated output snapshots may be superseded. Atomic record publication is distinct from workflow completion.

Implementation: [store/protocol.py](../../haplotype_reconstruction/store/protocol.py), [store/container.py](../../haplotype_reconstruction/store/container.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(new records). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="eng-15"></a>
## eng-15 Payload codec

Encode typed payload metadata and array buffers losslessly, preserving axes, dtype, shape and required layout. Validate sizes, checksums and structural metadata on decode. Selective reads decompress only requested groups and verify the enclosing identity before exposing data. Compression level and codec threads affect storage/time, not values.

Implementation: [store/codec.py](../../haplotype_reconstruction/store/codec.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(bytes). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="eng-16"></a>
## eng-16 Checked array windows (store/slices.py)

Expose checked sample/site windows whose bytes equal the corresponding decoded array slices. Each needed physical chunk is fully decompressed with the ordinary codec reader and cached by decoded byte size; NumPy performs all range slicing. The shared LRU retains at most 1 GiB (one oversized chunk alone), with one transient compressed input on a miss. This avoids relying on Blosc2 partial-range arithmetic. Validate requested axes, range and stride, and define whether a returned view remains valid across the next read. Consumers must copy or finish using leased views before reuse. A window route is accepted only after forced comparison with its resident route.

Implementation: [store/container.py](../../haplotype_reconstruction/store/container.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(rows × span) copying plus full decoding of physical chunks on cache misses. Dense painting reads its output flags once and traverses sample tiles sharing a GL chunk, within the existing sample/team and whole-bin workspace bounds. Repeated misses can repeat full decompression; logical requested bytes alone do not bound this cost.

<a id="eng-17"></a>
## eng-17 Temporary mapped arrays (store/mapped.py)

Provide temporary mapped arrays through an explicit storage capability. Flush dirty data before eviction or publication and distinguish scratch lifetime from immutable product lifetime. Mapping and eviction may change residency but never the values read by science. The runtime/store owns files; scientific code receives array access, not paths.

Implementation: [store/mapped.py](../../haplotype_reconstruction/store/mapped.py).

Parameters by name: `WINDOW_BYTES`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(size). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="eng-18"></a>
## eng-18 Store layout, layout marker, run manifest, work states

Bind and validate the checkpoint layout before any publication. Init owns the run manifest; scatter tasks can append only their contig container, while gather/init own run-level writes. Require compatible schemas and reject foreign or earlier layouts. Keep markers separate from product authority and temporary kernel/work storage outside the immutable layout.

Implementation: [store/store.py](../../haplotype_reconstruction/store/store.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Directory/layout inspection and incremental record-index reads.

<a id="eng-19"></a>
## eng-19 Process bootstrap and numerical-library set-up

Validate affinity and establish numerical-library policy before starting the forkserver. Preload the declared import closure before workers begin, initialise their threading runtime safely and enforce declared signatures. Ordinary worker graphs compile missing kernels alongside unrelated ready scientific tasks. Each scientific root waits for the missing compile tasks in its numerical ownership group before entering or expanding, as specified in ENG-27; shared callees retain their compile locks. There is no whole-graph compilation barrier. Simulation init and in-process graphs warm the required kernels before scientific tasks start. One application boundary owns these process-wide effects; importing science must perform none.

Implementation: [engine/kernels.py](../../haplotype_reconstruction/engine/kernels.py), [engine/runtime.py](../../haplotype_reconstruction/engine/runtime.py), [interface/run.py](../../haplotype_reconstruction/interface/run.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Process/bootstrap and code-closure overhead; compilation runs beside scientific tasks on ordinary worker graphs or before them on eager routes.

<a id="eng-20"></a>
## eng-20 CPU budget tree and the fair-share rule

Allocate claimed teams within one total core budget. Open the startup gate under its declared worker/admission conditions, split contenders fairly and distribute remainders by the prescribed rules. Rebalance at each task-owned kernel call or independent section, never inside a running native kernel. `engine.scheduler.kernel_team(max_threads=useful_width)` refreshes and applies the live authority; its cap describes useful parallel work rather than the task’s initial integer thread count. Claims protect running teams until the next boundary, so completed peers’ CPUs can be lent to work already executing without oversubscription. Contig leases may borrow free cores when peers are satisfied and lend while waiting. Resources cannot select scientific candidates.

Implementation: [engine/graph.py](../../haplotype_reconstruction/engine/graph.py), [engine/scheduler.py](../../haplotype_reconstruction/engine/scheduler.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(workers). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="eng-21"></a>
## eng-21 BudgetedPool (process pools)

Dispatch independent process tasks through the shared authority, with queued admission followed by a worker CPU seat before running. Consume results only while holding a seat, preserve keyed output placement and release all tokens/resources on failure. Recycle only under the declared safe conditions. Worker exceptions and unexpected exits stop the run while preserving complete published records.

Implementation: [engine/workers.py](../../haplotype_reconstruction/engine/workers.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Process startup, task dispatch and one shared-input copy per pool; scientific task costs dominate.

<a id="eng-22"></a>
## eng-22 ChromosomeExecutor (persistent per-chromosome workers)

Retain contig state on its affinity worker across consecutive tasks. Partition by declared weights or use the longest-first queue, placing results by key. Reloadable identity-keyed caches may be evicted on an idle worker under measured pressure; explicit retained references and expansion continuations remain pinned. A worker with pinned live state cannot recycle; an announced retirement returns its continuation position before replacement. Unexpected exit is a failure, including an unannounced successful process exit.

Implementation: [engine/workers.py](../../haplotype_reconstruction/engine/workers.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(tasks). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="eng-23"></a>
## eng-23 Contig lanes and the straggler floor

Plan concurrent contig lanes from pending work and the configured lane limit. Use declared task costs (or learned per-kind wall cost), the contig cost order and straggler floor with the upper median of positive weights; update remaining work only when whole tasks finish. Borrow unused cores at safe boundaries. On a contig failure cancel sibling work and preserve already committed records. These scheduling heuristics affect latency, not graph semantics.

Implementation: [engine/scheduler.py](../../haplotype_reconstruction/engine/scheduler.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(contigs). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="eng-24"></a>
## eng-24 Shared arrays

Share immutable arrays with explicit ownership and attach/detach lifetimes. Copy exact bytes and avoid object-identity scientific cache keys. Clean abandoned shared segments only after verifying the owning process identity and lack of use. A task cannot mutate another task’s shared input. Identical immutable contiguous backing is copied once across live shared bundles and view shapes. Read-only file/shared mappings are borrowed without a second segment, with an independent descriptor pin per live backing. Each bundle owns a reference; releasing the original owner, closing its mapping or unlinking its path cannot revoke another bundle. Writable and private copy-on-write sources retain snapshot copies. Borrowed inputs remain immutable through their last consumer. Expansion-local result values leave after their gather consumes them; original caller task results remain public. `retain_intermediates=True` retains private expansion results for diagnostics. In-flight holds protect sent arrays, dead attachments are unmapped after messages are consumed, and each unpickler clears its memo at payload completion so its bound loader cannot cyclically retain decoded arrays. Explicit transport ownership, rather than attachment resource trackers, owns unlinking.

Implementation: [engine/workers.py](../../haplotype_reconstruction/engine/workers.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(D) copying/attachment for shared bytes plus process-liveness checks at cleanup.

<a id="eng-25"></a>
## eng-25 Memory governor (pressure admission, ledger, trimming)

Admit outer work from measured process-group memory pressure, maintaining queued, running, waiting, suspended and resident states. Nested continuations retain progress guarantees; only running/queued work blocks a forced progress grant. The token, waiter and label tables double under the interprocess lock when full; attached processes remap before their next ledger operation, preserving slot indices and statistics. Initial table sizes are not execution caps. Working-set estimates are diagnostics, not planned-peak reservations. Outer starts are unrestricted through 90% measured use. Above that point their spacing increases continuously as headroom vanishes: with p = (used − 0.9 limit)/(0.1 limit), delay = sample interval × p/(1 − p). Pressure is refreshed at most every half second; paced starts may share a sample, avoiding a second throughput cliff at the onset of pressure. At the measured limit, only a forced drain can start. Recently polled deeper/older waiters have precedence; an ineligible queue head expires after two poll periods, so a request awaiting its affinity or CPU cannot deadlock other ready work. Idle workers release reloadable caches and trim their heaps before replacement; pinned continuations continue to have a progress grant. Flexible buffers may change batch shape only on a proven value-equivalent route; scientific budgets never depend on free memory.

Implementation: [engine/scheduler.py](../../haplotype_reconstruction/engine/scheduler.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(tokens). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="eng-26"></a>
## eng-26 Memory stall watchdog

Watch admission progress with the declared polling and stall interval. Reset the baseline when grants/starts/releases, process membership, sustained worker CPU activity (at least 0.05 core) or physical I/O changes, or measurements become unavailable. Coordinator admission polling is excluded from CPU activity; owner CPU counts only while an in-process task is executing. A useful serial or fractional-core worker and physical I/O still reset the baseline. Samples and launch-delay expiry are not progress. At the first sustained idle interval, request executor-side idle-state recovery and allow one further interval; only a second idle interval with no progress can fail. Recovery is requested without taking a potentially stuck ledger lock. Idle-worker maintenance is issued once per task boundary, so pressure alone cannot create a replacement loop that masks a hang. Fail only after the sustained low-activity condition is rechecked. This watchdog diagnoses stalled execution; it is not a scientific timeout that returns partial calls.

Implementation: [engine/scheduler.py](../../haplotype_reconstruction/engine/scheduler.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: One bounded process-accounting scan per configured poll while requests wait.

<a id="eng-27"></a>
## eng-27 Kernel cache and warm-up

Derive the kernel cache namespace from the declared code/numerical closure and publish valid cache state even after a later scientific failure. Ordinary worker graphs schedule missing-signature compilation beside unrelated ready work. Each scientific root depends on the missing compile tasks for its numerical modules before it enters or expands; shared callees retain their compile locks. Dependencies cover only the missing functions in that stage’s numerical ownership group, including a general selection task’s context assembly and pedigree/phase painting emissions, rather than every producer reached through record readers or type imports. Application selection rounds consume their separately scheduled context assemblies and wait only for selection, evidence projection and (after the initial round) founder-allele kernels. Context assemblies do not wait for final-role refinement. A warm cache adds no tasks or edges. Existing stage dependencies and scientific order are unchanged; there is no whole-graph warm-up barrier. Simulation init, in-process graphs and explicit warm_up callers compile missing signatures eagerly in the specified cost order before scientific execution. Cache presence may save compilation but cannot change the declared supported signatures or numerical policy. Do not fork after activating an incompatible numerical runtime.

Implementation: [engine/kernels.py](../../haplotype_reconstruction/engine/kernels.py), [engine/runtime.py](../../haplotype_reconstruction/engine/runtime.py), [interface/run.py](../../haplotype_reconstruction/interface/run.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Compilation cost of missing declared signatures; a compatible warm cache avoids this work.

<a id="eng-28"></a>
## eng-28 Kernel manifest and compile audit

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Declare compiled signatures beside kernels and verify the reachable closure against actual supported routes. Audit compilation with explicit runtime hooks, recording missing or undeclared signatures and serial/threaded use. Keep the catalogue generated from declarations. Compile audit is a developer diagnostic, not an environment-controlled scientific mode.

Implementation: [core/kernels.py](../../haplotype_reconstruction/core/kernels.py), [engine/kernels.py](../../haplotype_reconstruction/engine/kernels.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Linear in declared/audited signatures and code-closure entries.

<a id="eng-30"></a>
## eng-30 Run records, run log, stage timers, published-table hook and status

Record invocation/task timing, progress, errors and provenance outside scientific products. Render tables from checked results on each requested report pass and inspect payload-free completion where supported. Published CSV/VCF/JSON files are outputs only and are never read back as inference inputs. Log verbosity and status polling do not alter a derivation.

Implementation: [engine/records.py](../../haplotype_reconstruction/engine/records.py), [interface/status.py](../../haplotype_reconstruction/interface/status.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: One bounded record/log update per task or stage boundary; no scientific result scan for ordinary progress.
